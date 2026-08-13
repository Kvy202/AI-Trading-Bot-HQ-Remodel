from __future__ import annotations

import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools import hyperliquid_market_aggregate as aggregate
from tools import hyperliquid_market_recorder as recorder


BASE = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def raw_trade(received_at, exchange_time, side, price, size, tid):
    code = "B" if side == "BUY" else "A"
    return recorder.parse_trade(
        {
            "coin": "BTC",
            "side": code,
            "px": price,
            "sz": size,
            "time": exchange_time,
            "tid": tid,
            "hash": f"0x{tid}",
        },
        received_at,
    )


def raw_book(received_at, exchange_time, imbalance, spread="2"):
    return {
        "receive_time_utc": recorder.utc_text(received_at),
        "receive_time_ms": recorder.epoch_ms(received_at),
        "exchange_time_ms": exchange_time,
        "coin": "BTC",
        "best_bid": "100",
        "best_ask": "102",
        "mid": "101",
        "spread_bps": spread,
        "bid_depth_1": "1",
        "ask_depth_1": "1",
        "imbalance_1": imbalance,
        "bid_depth_5": "5",
        "ask_depth_5": "5",
        "imbalance_5": imbalance,
        "bid_depth_10": "10",
        "ask_depth_10": "10",
        "imbalance_10": imbalance,
        "bid_depth_20": "20",
        "ask_depth_20": "20",
        "imbalance_20": imbalance,
        "bid_order_count_5": 5,
        "ask_order_count_5": 5,
        "microprice": "101",
        "microprice_minus_mid_bps": imbalance,
    }


def raw_context(received_at, oi, mark, basis):
    return {
        "receive_time_utc": recorder.utc_text(received_at),
        "receive_time_ms": recorder.epoch_ms(received_at),
        "exchange_time_ms": None,
        "coin": "BTC",
        "mark_px": mark,
        "mid_px": "100",
        "oracle_px": "100",
        "funding": "0.0001",
        "open_interest": oi,
        "day_notional_volume": "1000",
        "previous_day_price": "99",
        "mark_oracle_basis_bps": basis,
        "mid_oracle_basis_bps": "0",
        "mark_mid_basis_bps": basis,
    }


def raw_candle(received_at):
    open_ms = recorder.epoch_ms(BASE)
    return recorder.parse_candle(
        {
            "t": open_ms,
            "T": open_ms + 299_999,
            "s": "BTC",
            "i": "5m",
            "o": "100",
            "h": "112",
            "l": "98",
            "c": "110",
            "v": "30",
            "n": 20,
        },
        received_at,
    )


def build_capture(root: Path):
    recorder.initialize_manifest(root, started_at=BASE)
    writer = recorder.HourlyJsonlWriter(root, fsync_interval_seconds=0)
    base_ms = recorder.epoch_ms(BASE)
    writer.append("trades", "BTC", BASE + timedelta(seconds=10), raw_trade(BASE + timedelta(seconds=10), base_ms + 10_000, "BUY", "100", "2", 1))
    writer.append("trades", "BTC", BASE + timedelta(seconds=20), raw_trade(BASE + timedelta(seconds=20), base_ms + 20_000, "SELL", "110", "1", 2))
    writer.append("book_5s", "BTC", BASE + timedelta(seconds=5), raw_book(BASE + timedelta(seconds=5), base_ms + 5_000, "0.2", "1"))
    writer.append("book_5s", "BTC", BASE + timedelta(seconds=35), raw_book(BASE + timedelta(seconds=35), base_ms + 35_000, "0.4", "3"))
    writer.append("asset_ctx", "BTC", BASE + timedelta(seconds=5), raw_context(BASE + timedelta(seconds=5), "100", "101", "100"))
    writer.append("asset_ctx", "BTC", BASE + timedelta(seconds=50), raw_context(BASE + timedelta(seconds=50), "110", "102", "200"))
    writer.append("candles_5m", "BTC", BASE + timedelta(minutes=5), raw_candle(BASE + timedelta(minutes=5)))
    writer.close()


def read_rows(output: Path, interval: str, coin: str = "BTC"):
    path = output / interval / f"{coin}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tree_hash(path: Path):
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def test_one_minute_trade_book_and_context_aggregation(tmp_path):
    capture = tmp_path / "capture"
    output = tmp_path / "aggregate"
    build_capture(capture)
    summary = aggregate.aggregate_capture(capture, output)
    first = read_rows(output, "1m")[0]
    assert first["buy_trade_count"] == 1
    assert first["sell_trade_count"] == 1
    assert first["buy_volume"] == "2"
    assert first["sell_volume"] == "1"
    assert first["buy_notional"] == "200"
    assert first["sell_notional"] == "110"
    assert first["signed_notional"] == "90"
    assert first["total_notional"] == "310"
    assert first["trade_vwap"] == "103.33333333333333333333333333333333333333333333333"
    assert first["aggressor_imbalance"] == "0.33333333333333333333333333333333333333333333333333"
    assert first["signed_flow_reconciled"] is True
    assert first["mean_spread_bps"] == "2"
    assert first["median_spread_bps"] == "2"
    assert first["mean_imbalance_5"] == "0.3"
    assert first["last_imbalance_5"] == "0.4"
    assert first["mean_microprice_minus_mid_bps"] == "0.3"
    assert first["last_open_interest"] == "110"
    assert first["open_interest_change"] == "10"
    assert first["open_interest_pct_change"] == "0.1"
    assert first["last_funding"] == "0.0001"
    assert first["last_mark_px"] == "102"
    assert first["last_oracle_px"] == "100"
    assert first["last_mark_oracle_basis_bps"] == "200"
    assert first["mean_mark_oracle_basis_bps"] == "150"
    assert first["missing"] == {"book": False, "candle": True, "context": False, "trade_flow": False}
    assert first["forward_fill_applied"] is False
    assert summary["output_row_counts"]["1m"]["BTC"] >= 1


def test_five_minute_aggregation_aligns_native_candle(tmp_path):
    capture = tmp_path / "capture"
    output = tmp_path / "aggregate"
    build_capture(capture)
    aggregate.aggregate_capture(capture, output)
    row = read_rows(output, "5m")[0]
    assert row["interval"] == "5m"
    assert row["trade_count"] == 2
    assert row["candle_open"] == "100"
    assert row["candle_high"] == "112"
    assert row["candle_low"] == "98"
    assert row["candle_close"] == "110"
    assert row["candle_volume"] == "30"
    assert row["candle_trade_count"] == 20
    assert row["candle_is_complete_as_captured"] is True
    assert row["candle_may_include_pre_capture_data"] is False
    assert row["missing"]["candle"] is False


def test_missingness_is_explicit_and_no_forward_fill_occurs(tmp_path):
    capture = tmp_path / "capture"
    output = tmp_path / "aggregate"
    build_capture(capture)
    aggregate.aggregate_capture(capture, output)
    rows = read_rows(output, "1m")
    later = rows[-1]
    assert later["forward_fill_applied"] is False
    assert later["missing"]["trade_flow"] is True
    assert later["last_funding"] is None
    assert "last_funding" in later["missing_fields"]


def test_deterministic_output_from_deterministic_input(tmp_path):
    capture = tmp_path / "capture"
    output = tmp_path / "aggregate"
    build_capture(capture)
    first_summary = aggregate.aggregate_capture(capture, output)
    first_hash = tree_hash(output)
    second_summary = aggregate.aggregate_capture(capture, output)
    assert second_summary == first_summary
    assert tree_hash(output) == first_hash


def test_offline_aggregator_never_accesses_network(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    build_capture(capture)

    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    summary = aggregate.aggregate_capture(capture, tmp_path / "aggregate")
    assert summary["network_accessed"] is False


def test_malformed_input_is_skipped_and_reported(tmp_path):
    capture = tmp_path / "capture"
    build_capture(capture)
    trade_path = next((capture / "raw" / "trades" / "BTC").glob("*/*.jsonl"))
    with trade_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    summary = aggregate.aggregate_capture(capture, tmp_path / "aggregate")
    assert summary["malformed_input_count"] == 1
    assert read_rows(tmp_path / "aggregate", "1m")[0]["trade_count"] == 2


def test_pre_capture_records_are_excluded_from_forward_aggregates(tmp_path):
    capture = tmp_path / "capture"
    build_capture(capture)
    writer = recorder.HourlyJsonlWriter(capture, fsync_interval_seconds=0)
    before = BASE - timedelta(milliseconds=1)
    writer.append(
        "trades",
        "BTC",
        before,
        raw_trade(before, recorder.epoch_ms(BASE) + 30_000, "BUY", "100", "9", 99),
    )
    writer.close()
    summary = aggregate.aggregate_capture(capture, tmp_path / "aggregate")
    assert summary["pre_capture_input_count"] == 1
    assert read_rows(tmp_path / "aggregate", "1m")[0]["trade_count"] == 2
