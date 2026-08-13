"""Offline deterministic aggregates for passive Hyperliquid captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SYMBOLS: tuple[str, ...] = ("BTC", "ETH")
SOURCE_STREAMS: tuple[str, ...] = ("trades", "book_5s", "asset_ctx", "candles_5m")
AGGREGATE_SCHEMA_VERSION = "1.0.0"
INTERVALS: tuple[tuple[str, int], ...] = (("1m", 60_000), ("5m", 300_000))

TRADE_FIELDS: tuple[str, ...] = (
    "buy_trade_count",
    "sell_trade_count",
    "buy_volume",
    "sell_volume",
    "buy_notional",
    "sell_notional",
    "signed_notional",
    "trade_count",
    "total_volume",
    "total_notional",
    "trade_vwap",
    "aggressor_imbalance",
)
BOOK_FIELDS: tuple[str, ...] = (
    "mean_spread_bps",
    "median_spread_bps",
    "mean_imbalance_1",
    "mean_imbalance_5",
    "mean_imbalance_10",
    "mean_imbalance_20",
    "last_imbalance_5",
    "last_imbalance_20",
    "mean_microprice_minus_mid_bps",
)
CONTEXT_FIELDS: tuple[str, ...] = (
    "last_funding",
    "last_open_interest",
    "open_interest_change",
    "open_interest_pct_change",
    "last_mark_px",
    "last_oracle_px",
    "last_mark_oracle_basis_bps",
    "mean_mark_oracle_basis_bps",
)
CANDLE_FIELDS: tuple[str, ...] = (
    "candle_open_time_ms",
    "candle_close_time_ms",
    "candle_open",
    "candle_high",
    "candle_low",
    "candle_close",
    "candle_volume",
    "candle_trade_count",
    "candle_latest_receive_time_utc",
    "candle_is_complete_as_captured",
    "candle_may_include_pre_capture_data",
)


class AggregateInputError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def decimal_value(value: Any, field: str, *, allow_none: bool = False) -> Decimal | None:
    if value is None and allow_none:
        return None
    if value is None or isinstance(value, bool):
        raise AggregateInputError(f"{field} is not numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AggregateInputError(f"{field} is not a decimal") from exc
    if not parsed.is_finite():
        raise AggregateInputError(f"{field} is not finite")
    return parsed


def decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def mean(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    with localcontext() as context:
        context.prec = 50
        return sum(values, Decimal(0)) / Decimal(len(values))


def median(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    with localcontext() as context:
        context.prec = 50
        return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def utc_text_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc_ms(value: Any) -> int:
    if not isinstance(value, str):
        raise AggregateInputError("receive_time_utc is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AggregateInputError("receive_time_utc is invalid") from exc
    if timestamp.tzinfo is None:
        raise AggregateInputError("receive_time_utc is not timezone-aware")
    return int(timestamp.timestamp() * 1000)


def atomic_write_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_lines(path, [value])


def _read_manifest(capture_root: Path) -> tuple[dict[str, Any], bytes]:
    path = capture_root / "manifest.json"
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregateInputError(f"capture manifest is unavailable: {path}") from exc
    if not isinstance(value, dict) or not value.get("capture_started_at_utc"):
        raise AggregateInputError("capture manifest has no forward-only start")
    if value.get("symbols") != list(SYMBOLS):
        raise AggregateInputError("capture symbols do not match BTC/ETH contract")
    return value, raw


def _read_records(capture_root: Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    records = {stream: {coin: [] for coin in SYMBOLS} for stream in SOURCE_STREAMS}
    stats: dict[str, Any] = {
        "input_record_counts": {stream: {coin: 0 for coin in SYMBOLS} for stream in SOURCE_STREAMS},
        "malformed_input_count": 0,
    }
    sequence = 0
    for stream in SOURCE_STREAMS:
        for coin in SYMBOLS:
            files = sorted((capture_root / "raw" / stream / coin).glob("*/*.jsonl"))
            for path in files:
                try:
                    handle = path.open("r", encoding="utf-8")
                except FileNotFoundError:
                    continue
                with handle:
                    for line in handle:
                        sequence += 1
                        try:
                            value = json.loads(line)
                            if not isinstance(value, dict) or value.get("coin") != coin:
                                raise AggregateInputError("record symbol does not match partition")
                            value = dict(value)
                            value["_source_sequence"] = sequence
                            records[stream][coin].append(value)
                            stats["input_record_counts"][stream][coin] += 1
                        except (json.JSONDecodeError, AggregateInputError):
                            stats["malformed_input_count"] += 1
    return records, stats


def _record_time_ms(stream: str, record: Mapping[str, Any]) -> int:
    if stream == "candles_5m":
        value = record.get("open_time_ms")
    elif stream == "asset_ctx":
        value = record.get("exchange_time_ms")
        if value is None:
            value = record.get("receive_time_ms")
            if value is None:
                value = parse_utc_ms(record.get("receive_time_utc"))
    else:
        value = record.get("exchange_time_ms")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AggregateInputError(f"{stream} record has no valid event time")
    return value


def _bucket(value: int, interval_ms: int) -> int:
    return value - value % interval_ms


def _validated_trade(record: Mapping[str, Any]) -> dict[str, Any]:
    side = record.get("side")
    if side not in {"BUY", "SELL"}:
        raise AggregateInputError("trade side is invalid")
    price = decimal_value(record.get("price"), "trade.price")
    size = decimal_value(record.get("size"), "trade.size")
    notional = decimal_value(record.get("notional"), "trade.notional")
    signed = decimal_value(record.get("signed_notional"), "trade.signed_notional")
    assert price is not None and size is not None and notional is not None and signed is not None
    expected_notional = price * size
    expected_signed = expected_notional if side == "BUY" else -expected_notional
    if notional != expected_notional or signed != expected_signed:
        raise AggregateInputError("trade notional reconciliation failed")
    return {"side": side, "size": size, "notional": notional, "signed_notional": signed}


def aggregate_trade_flow(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [_validated_trade(record) for record in records]
    buys = [record for record in valid if record["side"] == "BUY"]
    sells = [record for record in valid if record["side"] == "SELL"]
    buy_volume = sum((record["size"] for record in buys), Decimal(0))
    sell_volume = sum((record["size"] for record in sells), Decimal(0))
    buy_notional = sum((record["notional"] for record in buys), Decimal(0))
    sell_notional = sum((record["notional"] for record in sells), Decimal(0))
    signed_notional = sum((record["signed_notional"] for record in valid), Decimal(0))
    total_volume = buy_volume + sell_volume
    total_notional = buy_notional + sell_notional
    with localcontext() as context:
        context.prec = 50
        vwap = total_notional / total_volume if total_volume != 0 else None
        imbalance = (buy_volume - sell_volume) / total_volume if total_volume != 0 else None
    return {
        "buy_trade_count": len(buys),
        "sell_trade_count": len(sells),
        "buy_volume": decimal_text(buy_volume),
        "sell_volume": decimal_text(sell_volume),
        "buy_notional": decimal_text(buy_notional),
        "sell_notional": decimal_text(sell_notional),
        "signed_notional": decimal_text(signed_notional),
        "trade_count": len(valid),
        "total_volume": decimal_text(total_volume),
        "total_notional": decimal_text(total_notional),
        "trade_vwap": decimal_text(vwap),
        "aggressor_imbalance": decimal_text(imbalance),
        "signed_flow_reconciled": signed_notional == buy_notional - sell_notional,
    }


def _decimal_series(records: Sequence[Mapping[str, Any]], field: str) -> list[Decimal]:
    result: list[Decimal] = []
    for record in records:
        value = decimal_value(record.get(field), field, allow_none=True)
        if value is not None:
            result.append(value)
    return result


def aggregate_book(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: (_record_time_ms("book_5s", record), record["_source_sequence"]))
    spread = _decimal_series(ordered, "spread_bps")
    result = {
        "mean_spread_bps": decimal_text(mean(spread)),
        "median_spread_bps": decimal_text(median(spread)),
        "mean_imbalance_1": decimal_text(mean(_decimal_series(ordered, "imbalance_1"))),
        "mean_imbalance_5": decimal_text(mean(_decimal_series(ordered, "imbalance_5"))),
        "mean_imbalance_10": decimal_text(mean(_decimal_series(ordered, "imbalance_10"))),
        "mean_imbalance_20": decimal_text(mean(_decimal_series(ordered, "imbalance_20"))),
        "last_imbalance_5": None,
        "last_imbalance_20": None,
        "mean_microprice_minus_mid_bps": decimal_text(mean(_decimal_series(ordered, "microprice_minus_mid_bps"))),
    }
    if ordered:
        result["last_imbalance_5"] = decimal_text(decimal_value(ordered[-1].get("imbalance_5"), "imbalance_5", allow_none=True))
        result["last_imbalance_20"] = decimal_text(decimal_value(ordered[-1].get("imbalance_20"), "imbalance_20", allow_none=True))
    return result


def aggregate_context(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: (_record_time_ms("asset_ctx", record), record["_source_sequence"]))
    if not ordered:
        return {field: None for field in CONTEXT_FIELDS}
    last = ordered[-1]
    oi_values = [decimal_value(record.get("open_interest"), "open_interest", allow_none=True) for record in ordered]
    oi_values = [value for value in oi_values if value is not None]
    oi_change: Decimal | None = None
    oi_pct_change: Decimal | None = None
    if oi_values:
        oi_change = oi_values[-1] - oi_values[0]
        if oi_values[0] != 0:
            with localcontext() as context:
                context.prec = 50
                oi_pct_change = oi_change / oi_values[0]
    basis_values = _decimal_series(ordered, "mark_oracle_basis_bps")
    return {
        "last_funding": decimal_text(decimal_value(last.get("funding"), "funding", allow_none=True)),
        "last_open_interest": decimal_text(decimal_value(last.get("open_interest"), "open_interest", allow_none=True)),
        "open_interest_change": decimal_text(oi_change),
        "open_interest_pct_change": decimal_text(oi_pct_change),
        "last_mark_px": decimal_text(decimal_value(last.get("mark_px"), "mark_px", allow_none=True)),
        "last_oracle_px": decimal_text(decimal_value(last.get("oracle_px"), "oracle_px", allow_none=True)),
        "last_mark_oracle_basis_bps": decimal_text(decimal_value(last.get("mark_oracle_basis_bps"), "mark_oracle_basis_bps", allow_none=True)),
        "mean_mark_oracle_basis_bps": decimal_text(mean(basis_values)),
    }


def aggregate_candle(
    records: Sequence[Mapping[str, Any]],
    *,
    capture_started_at_ms: int,
) -> dict[str, Any]:
    if not records:
        return {field: None for field in CANDLE_FIELDS}
    ordered = sorted(
        records,
        key=lambda record: (
            int(record.get("receive_time_ms", parse_utc_ms(record.get("receive_time_utc")))),
            record["_source_sequence"],
        ),
    )
    last = ordered[-1]
    receive_ms = last.get("receive_time_ms")
    if not isinstance(receive_ms, int):
        receive_ms = parse_utc_ms(last.get("receive_time_utc"))
    open_time = last.get("open_time_ms")
    close_time = last.get("close_time_ms")
    if not isinstance(open_time, int) or not isinstance(close_time, int):
        raise AggregateInputError("candle timestamps are invalid")
    return {
        "candle_open_time_ms": open_time,
        "candle_close_time_ms": close_time,
        "candle_open": decimal_text(decimal_value(last.get("open"), "candle.open")),
        "candle_high": decimal_text(decimal_value(last.get("high"), "candle.high")),
        "candle_low": decimal_text(decimal_value(last.get("low"), "candle.low")),
        "candle_close": decimal_text(decimal_value(last.get("close"), "candle.close")),
        "candle_volume": decimal_text(decimal_value(last.get("volume"), "candle.volume")),
        "candle_trade_count": int(last.get("trade_count")),
        "candle_latest_receive_time_utc": last.get("receive_time_utc"),
        "candle_is_complete_as_captured": receive_ms > close_time,
        "candle_may_include_pre_capture_data": open_time < capture_started_at_ms,
    }


def _empty_group(fields: Sequence[str]) -> dict[str, Any]:
    return {field: None for field in fields}


def _build_rows_for_symbol(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    coin: str,
    interval_name: str,
    interval_ms: int,
    capture_started_at_ms: int,
    malformed_counter: list[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, list[Mapping[str, Any]]]] = {
        stream: defaultdict(list) for stream in SOURCE_STREAMS
    }
    all_buckets: set[int] = set()
    for stream in SOURCE_STREAMS:
        for record in records[stream]:
            try:
                receive_time = record.get("receive_time_ms")
                if not isinstance(receive_time, int) or isinstance(receive_time, bool):
                    receive_time = parse_utc_ms(record.get("receive_time_utc"))
                all_buckets.add(_bucket(receive_time, interval_ms))
                timestamp = _record_time_ms(stream, record)
                bucket = _bucket(timestamp, interval_ms)
                if stream == "candles_5m" and interval_name != "5m":
                    continue
                grouped[stream][bucket].append(record)
                all_buckets.add(bucket)
            except (AggregateInputError, TypeError, ValueError):
                malformed_counter[0] += 1
    if not all_buckets:
        return []
    minimum = max(min(all_buckets), _bucket(capture_started_at_ms, interval_ms))
    maximum = max(all_buckets)
    rows: list[dict[str, Any]] = []
    for bucket_start in range(minimum, maximum + interval_ms, interval_ms):
        trade_records = grouped["trades"].get(bucket_start, [])
        book_records = grouped["book_5s"].get(bucket_start, [])
        context_records = grouped["asset_ctx"].get(bucket_start, [])
        candle_records = grouped["candles_5m"].get(bucket_start, [])
        try:
            trade_values = aggregate_trade_flow(trade_records) if trade_records else _empty_group(TRADE_FIELDS)
            if not trade_records:
                trade_values.update({"signed_flow_reconciled": None})
            book_values = aggregate_book(book_records) if book_records else _empty_group(BOOK_FIELDS)
            context_values = aggregate_context(context_records)
            candle_values = (
                aggregate_candle(candle_records, capture_started_at_ms=capture_started_at_ms)
                if candle_records
                else _empty_group(CANDLE_FIELDS)
            )
        except (AggregateInputError, TypeError, ValueError):
            malformed_counter[0] += 1
            continue
        row: dict[str, Any] = {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "interval": interval_name,
            "bucket_start_ms": bucket_start,
            "bucket_end_ms": bucket_start + interval_ms,
            "bucket_start_utc": utc_text_from_ms(bucket_start),
            "coin": coin,
            **trade_values,
            **book_values,
            **context_values,
            **candle_values,
            "source_observation_counts": {
                "trades": len(trade_records),
                "book_5s": len(book_records),
                "asset_ctx": len(context_records),
                "candles_5m": len(candle_records),
            },
            "missing": {
                "trade_flow": not bool(trade_records),
                "book": not bool(book_records),
                "context": not bool(context_records),
                "candle": not bool(candle_records),
            },
            "bucket_contains_capture_start": bucket_start <= capture_started_at_ms < bucket_start + interval_ms,
            "forward_fill_applied": False,
        }
        metric_fields = (*TRADE_FIELDS, *BOOK_FIELDS, *CONTEXT_FIELDS, *CANDLE_FIELDS)
        row["missing_fields"] = sorted(field for field in metric_fields if row.get(field) is None)
        rows.append(row)
    return rows


def aggregate_capture(capture_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    capture_root = capture_root.resolve()
    output_root = (output_root or capture_root / "aggregates").resolve()
    manifest, manifest_raw = _read_manifest(capture_root)
    capture_started_at_ms = parse_utc_ms(manifest["capture_started_at_utc"])
    records, stats = _read_records(capture_root)
    pre_capture_input_count = 0
    for stream in SOURCE_STREAMS:
        for coin in SYMBOLS:
            forward_records: list[dict[str, Any]] = []
            for record in records[stream][coin]:
                receive_time = record.get("receive_time_ms")
                if not isinstance(receive_time, int) or isinstance(receive_time, bool):
                    try:
                        receive_time = parse_utc_ms(record.get("receive_time_utc"))
                    except AggregateInputError:
                        forward_records.append(record)
                        continue
                if receive_time < capture_started_at_ms:
                    pre_capture_input_count += 1
                    continue
                forward_records.append(record)
            records[stream][coin] = forward_records
    malformed_counter = [stats["malformed_input_count"]]
    output_counts: dict[str, dict[str, int]] = {name: {} for name, _ in INTERVALS}
    for interval_name, interval_ms in INTERVALS:
        for coin in SYMBOLS:
            symbol_records = {stream: records[stream][coin] for stream in SOURCE_STREAMS}
            rows = _build_rows_for_symbol(
                symbol_records,
                coin,
                interval_name,
                interval_ms,
                capture_started_at_ms,
                malformed_counter,
            )
            atomic_write_lines(output_root / interval_name / f"{coin}.jsonl", rows)
            output_counts[interval_name][coin] = len(rows)
    aggregate_manifest = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_capture_contract_version": manifest.get("capture_contract_version"),
        "capture_started_at_utc": manifest["capture_started_at_utc"],
        "source_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "symbols": list(SYMBOLS),
        "intervals": [name for name, _ in INTERVALS],
        "output_root": str(output_root),
        "input_record_counts": stats["input_record_counts"],
        "pre_capture_input_count": pre_capture_input_count,
        "malformed_input_count": malformed_counter[0],
        "output_row_counts": output_counts,
        "forward_fill_applied": False,
        "network_accessed": False,
    }
    atomic_write_json(output_root / "manifest.json", aggregate_manifest)
    return aggregate_manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline Hyperliquid 1m/5m research aggregates")
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        summary = aggregate_capture(arguments.capture_root, arguments.output_root)
    except AggregateInputError as exc:
        print(f"aggregate failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
