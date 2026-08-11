"""Capture, build, verify, and describe immutable Phase 24 datasets.

Only ``capture`` and ``capture-confirmation`` contain public market-data read
paths.  Feature/label building is performed by the project interpreter; the
canonical scaler worker is explicitly run by the candidate-training Python.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_training_environment import (
    PHASE22_BUNDLE,
    TRAINING_PYTHON,
    CandidateTrainingEnvironmentError,
    atomic_write_json,
    file_digest,
    git_commit,
    json_digest,
    load_training_policy,
    record_incumbent_inventory,
    utc_now,
    validate_phase24_evidence,
    verify_incumbent_inventory,
)


DATASET_ROOT = BASE_DIR / "reports" / "model_training_datasets"
CONFIRMATION_ROOT = BASE_DIR / "reports" / "model_candidate_confirmation"
SELECTION_FREEZE = BASE_DIR / "reports" / "model_candidate_selection_freeze.json"
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
SPLIT_NAMES = {0: "train", 1: "validation", 2: "internal_test", -1: "purged"}
PUBLIC_MARKET_HISTORY_START_UTC = "2009-01-03T00:00:00Z"
BITGET_HISTORY_PAGE_LIMIT = 200


class ModelTrainingDatasetError(ValueError):
    """A dataset integrity, isolation, chronology, or minimum-size gate failed."""


def canonical_utc(value: Any) -> str:
    import pandas as pd
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def phase22_source_bounds(bundle: Path | str = PHASE22_BUNDLE) -> dict[str, str]:
    first: list[datetime] = []
    last: list[datetime] = []
    for path in sorted(Path(bundle).glob("source_bars_*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ModelTrainingDatasetError(f"empty Phase 22 source file: {path.name}")
        parsed = [datetime.fromisoformat(row["bar_open_utc"].replace("Z", "+00:00")) for row in rows]
        first.append(min(parsed))
        last.append(max(parsed))
    if not first:
        raise ModelTrainingDatasetError("Phase 22 source bars unavailable")
    return {
        "earliest_source_bar_open_utc": min(first).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "final_source_bar_open_utc": max(last).astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def specification_contract() -> dict[str, Any]:
    path = BASE_DIR / "reports" / "model_retraining_specification_phase23_1.json"
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    models = value.get("models", {})
    if set(models) != {"lstm", "tcn", "tx"}:
        raise ModelTrainingDatasetError("training_pipeline_contract_incomplete")
    contracts = []
    for kind in ("lstm", "tcn", "tx"):
        model = models[kind]
        contracts.append((
            model.get("canonical_feature_names"), model.get("feature_digest"),
            model.get("symbol_id_map"), model.get("label_contract"), model.get("label_digest"),
            model.get("timeframe"), model.get("sequence_length"),
        ))
    if any(item != contracts[0] for item in contracts[1:]):
        raise ModelTrainingDatasetError("training_pipeline_contract_incomplete")
    columns, feature_digest, symbol_map, label, label_digest, timeframe, seq_len = contracts[0]
    if json_digest(columns) != feature_digest or json_digest(label) != label_digest:
        raise ModelTrainingDatasetError("training_pipeline_contract_incomplete")
    if label != {"type": "triple", "pt": 0.005, "sl": 0.005, "max_hold": 60, "tau": None, "horizon": 12}:
        raise ModelTrainingDatasetError("training_pipeline_contract_incomplete")
    if timeframe != "5m" or seq_len != 64:
        raise ModelTrainingDatasetError("training_pipeline_contract_incomplete")
    return {
        "feature_names": columns, "feature_contract_digest": feature_digest,
        "symbol_id_map": symbol_map, "label_contract": label,
        "label_contract_digest": label_digest, "timeframe": timeframe,
        "sequence_length": seq_len,
    }


def maximum_training_timestamps(
    *, bundle: Path | str = PHASE22_BUNDLE, max_lookahead: int = 60, timeframe_minutes: int = 5
) -> dict[str, str]:
    import pandas as pd
    bounds = phase22_source_bounds(bundle)
    earliest = pd.Timestamp(bounds["earliest_source_bar_open_utc"])
    interval = pd.Timedelta(minutes=timeframe_minutes)
    return {
        **bounds,
        "maximum_training_raw_bar_open_utc": canonical_utc(earliest - interval),
        # An endpoint at earliest-(lookahead+1)*interval uses future bars only
        # through earliest-interval, never a Phase 22 source row.
        "maximum_training_labeled_endpoint_utc": canonical_utc(
            earliest - interval * (max_lookahead + 1)
        ),
    }


def _parse_ohlcv_timestamps(values: Any) -> Any:
    """Parse CCXT numerics as Unix milliseconds and textual values as dates."""
    import pandas as pd
    series = pd.Series(values, copy=False)
    try:
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="raise")
            if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
                raise ModelTrainingDatasetError("numeric OHLCV timestamps must be finite")
            parsed = pd.to_datetime(numeric, unit="ms", utc=True, errors="raise")
        else:
            parsed = pd.to_datetime(series, utc=True, errors="raise")
    except ModelTrainingDatasetError:
        raise
    except Exception as exc:
        raise ModelTrainingDatasetError("OHLCV timestamps cannot be converted cleanly") from exc
    if parsed.isna().any():
        raise ModelTrainingDatasetError("OHLCV timestamps cannot be converted cleanly")
    return pd.DatetimeIndex(parsed)


def _normalize_ohlcv(
    frame: Any,
    *,
    as_of_utc: Any,
    timeframe: str = "5m",
    requested_start_utc: Any | None = None,
    requested_end_exclusive_utc: Any | None = None,
) -> tuple[Any, int, int]:
    import pandas as pd
    value = frame.copy()
    if "timestamp" in value.columns:
        value["timestamp"] = _parse_ohlcv_timestamps(value["timestamp"])
        value = value.set_index("timestamp")
    elif "bar_open_utc" in value.columns:
        value["bar_open_utc"] = _parse_ohlcv_timestamps(value["bar_open_utc"])
        value = value.set_index("bar_open_utc")
    else:
        value.index = _parse_ohlcv_timestamps(value.index)
    missing = [name for name in OHLCV_COLUMNS if name not in value.columns]
    if missing:
        raise ModelTrainingDatasetError(f"OHLCV missing columns: {missing}")
    value = value[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        raise ModelTrainingDatasetError("OHLCV contains non-finite values")
    duplicates = 0
    if value.index.has_duplicates:
        for timestamp, group in value.groupby(level=0, sort=False):
            if len(group.drop_duplicates()) > 1:
                raise ModelTrainingDatasetError(f"conflicting same-timestamp OHLCV: {canonical_utc(timestamp)}")
            duplicates += len(group) - 1
        value = value[~value.index.duplicated(keep="first")]
    value = value.sort_index()
    seconds = 300 if timeframe == "5m" else None
    if seconds is None:
        raise ModelTrainingDatasetError("Phase 24 timeframe must be 5m")
    as_of = pd.Timestamp(as_of_utc)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    plausible_start = pd.Timestamp(PUBLIC_MARKET_HISTORY_START_UTC)
    plausible_end = as_of + pd.Timedelta(seconds=seconds)
    if len(value) and (value.index.min() < plausible_start or value.index.max() > plausible_end):
        raise ModelTrainingDatasetError("OHLCV timestamps outside plausible public market-history range")
    if (requested_start_utc is None) != (requested_end_exclusive_utc is None):
        raise ModelTrainingDatasetError("both requested OHLCV range bounds are required")
    if requested_start_utc is not None:
        requested_start = pd.Timestamp(requested_start_utc)
        requested_end = pd.Timestamp(requested_end_exclusive_utc)
        requested_start = requested_start.tz_localize("UTC") if requested_start.tzinfo is None else requested_start.tz_convert("UTC")
        requested_end = requested_end.tz_localize("UTC") if requested_end.tzinfo is None else requested_end.tz_convert("UTC")
        if requested_start >= requested_end:
            raise ModelTrainingDatasetError("requested OHLCV range is invalid")
        if len(value) and (value.index.min() < requested_start or value.index.max() >= requested_end):
            raise ModelTrainingDatasetError("normalized OHLCV timestamps outside requested capture range")
    if len(value) > 1:
        deltas = np.diff(value.index.asi8) / 1_000_000_000
        if not np.isfinite(deltas).all() or np.any(deltas < seconds):
            raise ModelTrainingDatasetError("invalid sub-timeframe spacing for distinct 5m OHLCV bars")
    complete = value.index + pd.Timedelta(seconds=seconds) <= as_of
    incomplete = int((~complete).sum())
    return value.loc[complete].copy(), int(duplicates), incomplete


def _merge_ohlcv(existing: Any, incoming: Any) -> tuple[Any, int]:
    import pandas as pd
    if existing is None or len(existing) == 0:
        return incoming.sort_index(), 0
    merged = pd.concat([existing, incoming])
    duplicates = 0
    for timestamp, group in merged.groupby(level=0, sort=False):
        if len(group.drop_duplicates()) > 1:
            raise ModelTrainingDatasetError(f"conflicting same-timestamp OHLCV: {canonical_utc(timestamp)}")
        duplicates += max(0, len(group) - 1)
    return merged[~merged.index.duplicated(keep="first")].sort_index(), duplicates


def _write_raw_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("bar_open_utc", *OHLCV_COLUMNS))
        for timestamp, row in frame.iterrows():
            writer.writerow((canonical_utc(timestamp), *(format(float(row[name]), ".17g") for name in OHLCV_COLUMNS)))


def _read_raw_csv(path: Path) -> Any:
    import pandas as pd
    return pd.read_csv(path, parse_dates=["bar_open_utc"]).set_index("bar_open_utc")


def _gap_statistics(index: Any, timeframe_seconds: int = 300) -> tuple[int, float]:
    if len(index) < 2:
        return 0, 0.0
    deltas = np.diff(index.asi8) / 1_000_000_000
    if not np.isfinite(deltas).all() or np.any(deltas < timeframe_seconds):
        raise ModelTrainingDatasetError("invalid sub-timeframe spacing for distinct 5m OHLCV bars")
    missing = int(sum(max(0, round(value / timeframe_seconds) - 1) for value in deltas))
    return missing, float(np.max(deltas))


def _capture_end_contract(
    cutoffs: Mapping[str, Any], capture_end_exclusive_utc: str | None = None,
) -> dict[str, Any]:
    import pandas as pd

    default_end = pd.Timestamp(cutoffs["maximum_training_raw_bar_open_utc"]) + pd.Timedelta(minutes=5)
    default_end = default_end.tz_localize("UTC") if default_end.tzinfo is None else default_end.tz_convert("UTC")
    explicit = capture_end_exclusive_utc is not None
    requested = default_end if not explicit else capture_end_exclusive_utc
    try:
        if isinstance(requested, (float, np.floating)) and not math.isfinite(float(requested)):
            raise ValueError("non-finite timestamp")
        effective_end = pd.Timestamp(requested)
        if pd.isna(effective_end):
            raise ValueError("missing timestamp")
        effective_end = (
            effective_end.tz_localize("UTC")
            if effective_end.tzinfo is None else effective_end.tz_convert("UTC")
        )
        effective_ns = int(effective_end.value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelTrainingDatasetError("capture end must be a finite valid UTC timestamp") from exc
    if effective_ns % (300 * 1_000_000_000) != 0:
        raise ModelTrainingDatasetError("capture end must align to a 5-minute boundary")
    if effective_end > default_end:
        raise ModelTrainingDatasetError("capture end exceeds the Phase-22-safe maximum")
    return {
        "default_safe_end_exclusive_utc": canonical_utc(default_end),
        "effective_end_exclusive_utc": canonical_utc(effective_end),
        "explicit_historical_end_requested": explicit,
    }


def _dataset_id(
    venue: str, cutoff: str, target: int, *, effective_end_exclusive_utc: str | None = None,
) -> str:
    contract = {"phase": 24, "venue": venue, "timeframe": "5m", "symbols": ["BTCUSDT", "ETHUSDT"],
                "cutoff": cutoff, "target": target}
    if effective_end_exclusive_utc is not None:
        contract["effective_end_exclusive_utc"] = canonical_utc(effective_end_exclusive_utc)
    return f"phase24_5m_{json_digest(contract)[:12]}"


def _raw_capture_digest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "venue": manifest["source_venue"], "timeframe": "5m",
        "target": manifest["target_raw_bars_per_symbol"], "phase22_bounds": manifest["phase22_bounds"],
    }
    if manifest.get("explicit_historical_end_requested") is True:
        contract["effective_end_exclusive_utc"] = manifest["effective_end_exclusive_utc"]
    return contract


def _ccxt_timestamp_ms(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelTrainingDatasetError("CCXT OHLCV timestamp is not numeric milliseconds") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ModelTrainingDatasetError("CCXT OHLCV timestamp is not finite integer milliseconds")
    return int(numeric)


def _fetch_ccxt_ohlcv_range(
    exchange: Any,
    market_symbol: str,
    *,
    timeframe: str,
    start_utc: str,
    end_utc: str,
    limit: int,
    params: Mapping[str, Any],
    per_page: int,
    force_history_endpoint: bool,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> Any:
    """Deterministically page an explicitly bounded public CCXT OHLCV range."""
    import pandas as pd
    if timeframe != "5m" or per_page <= 0 or limit <= 0:
        raise ModelTrainingDatasetError("invalid public OHLCV pagination contract")
    start_ms = int(pd.Timestamp(start_utc).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_utc).timestamp() * 1000)
    if start_ms >= end_ms:
        raise ModelTrainingDatasetError("invalid public OHLCV requested range")
    interval_ms = 300_000
    request_params = dict(params)
    if force_history_endpoint:
        request_params["useHistoryEndpoint"] = True
    rows: list[list[float]] = []
    cursor = start_ms
    last_newest: int | None = None
    pages_requested = 0
    pages_returned = 0
    exchange_timestamps: list[int] = []
    stop_reason = "pagination_loop_complete"
    while cursor < end_ms and len(rows) < limit:
        pages_requested += 1
        page = exchange.fetch_ohlcv(
            market_symbol, timeframe=timeframe, since=cursor, limit=per_page,
            params=dict(request_params),
        )
        if not page:
            stop_reason = "empty_page"
            break
        pages_returned += 1
        page_timestamps = [_ccxt_timestamp_ms(row[0]) for row in page]
        exchange_timestamps.extend(page_timestamps)
        rows.extend(
            row for row, timestamp in zip(page, page_timestamps)
            if start_ms <= timestamp < end_ms
        )
        newest = max(page_timestamps)
        if last_newest is not None and newest <= last_newest:
            stop_reason = "no_forward_progress"
            break
        last_newest = newest
        cursor = newest + interval_ms
        if cursor >= end_ms:
            stop_reason = "requested_end_reached"
            break
        if len(rows) >= limit:
            stop_reason = "row_limit_reached"
            break
        sleep_fn((getattr(exchange, "rateLimit", None) or 250) / 1000.0)
    frame = pd.DataFrame(rows[:limit], columns=("timestamp", *OHLCV_COLUMNS))
    first_exchange = min(exchange_timestamps) if exchange_timestamps else None
    last_exchange = max(exchange_timestamps) if exchange_timestamps else None
    frame.attrs["capture_diagnostics"] = {
        "pages_requested": pages_requested,
        "pages_returned": pages_returned,
        "first_exchange_timestamp": (
            canonical_utc(pd.Timestamp(first_exchange, unit="ms", tz="UTC"))
            if first_exchange is not None else None
        ),
        "last_exchange_timestamp": (
            canonical_utc(pd.Timestamp(last_exchange, unit="ms", tz="UTC"))
            if last_exchange is not None else None
        ),
        "requested_start": canonical_utc(start_utc),
        "requested_end": canonical_utc(end_utc),
        "rows_before_normalization": int(len(frame)),
        "rows_after_normalization": None,
        "pagination_stop_reason": stop_reason,
        "public_endpoint": "ccxt_fetch_ohlcv_history" if force_history_endpoint else "ccxt_fetch_ohlcv",
        "page_limit": int(per_page),
    }
    return frame


def _public_fetch_range(
    symbol: str, *, timeframe: str, start_utc: str, end_utc: str, limit: int, venue: str
) -> Any:
    """Public-only paginated CCXT read; no keys, accounts, or order adapter."""
    from data import _close_exchange, _ensure_markets, _market_params, _resolve_market_symbol, get_exchange
    exchange = get_exchange(exchange_id=venue, kind="swap")
    try:
        _ensure_markets(exchange, "swap")
        market_symbol = _resolve_market_symbol(exchange, symbol, "swap")
        params = _market_params(exchange, "swap")
        is_bitget = str(venue).lower() == "bitget"
        return _fetch_ccxt_ohlcv_range(
            exchange, market_symbol, timeframe=timeframe, start_utc=start_utc,
            end_utc=end_utc, limit=limit, params=params,
            # CCXT documents Bitget's public history endpoint at 200 candles.
            # Asking it for 1000 causes its end-time calculation to jump ahead
            # before it clamps the returned page, leaving 800-bar holes.
            per_page=BITGET_HISTORY_PAGE_LIMIT if is_bitget else 1000,
            force_history_endpoint=is_bitget,
        )
    finally:
        _close_exchange(exchange)


def _validate_partial_capture_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse in-place reuse of timestamp-corrupted partial capture evidence."""
    import pandas as pd
    if manifest.get("capture_status") == "complete":
        return
    for info in manifest.get("per_symbol", {}).values():
        first, last = info.get("actual_first_utc"), info.get("actual_last_utc")
        requested_start = info.get("requested_start_utc") or info.get("requested_start")
        requested_end = info.get("requested_end_exclusive_utc") or info.get("requested_end")
        if not all((first, last, requested_start, requested_end)):
            continue
        try:
            rows = int(info.get("rows", info.get("completed_rows", 0)))
            valid = (
                pd.Timestamp(first) >= pd.Timestamp(requested_start)
                and pd.Timestamp(last) < pd.Timestamp(requested_end)
                and (rows < 2 or float(info.get("maximum_gap_seconds", 300.0)) >= 300.0)
            )
        except Exception:
            valid = False
        if not valid:
            raise ModelTrainingDatasetError(
                "invalid_partial_dataset_requires_delete_and_recapture"
            )


def capture_training_data(
    *, dataset_id: str | None = None, venue: str = "bitget", as_of_utc: Any | None = None,
    target_bars: int | None = None, fetcher: Callable[..., Any] | None = None,
    dataset_root: Path | str = DATASET_ROOT,
    capture_end_exclusive_utc: str | None = None,
) -> dict[str, Any]:
    validate_phase24_evidence()
    policy = load_training_policy()
    contract = specification_contract()
    target = int(target_bars or policy["target_raw_bars_per_symbol"])
    if target < int(policy["target_raw_bars_per_symbol"]):
        raise ModelTrainingDatasetError("CLI may not weaken target_raw_bars_per_symbol")
    cutoffs = maximum_training_timestamps(max_lookahead=int(contract["label_contract"]["max_hold"]))
    end_contract = _capture_end_contract(cutoffs, capture_end_exclusive_utc)
    data_id = dataset_id or _dataset_id(
        venue, cutoffs["maximum_training_raw_bar_open_utc"], target,
        effective_end_exclusive_utc=(
            end_contract["effective_end_exclusive_utc"]
            if end_contract["explicit_historical_end_requested"] else None
        ),
    )
    directory = Path(dataset_root) / data_id
    if directory.is_symlink():
        raise ModelTrainingDatasetError("dataset directory may not be a symlink")
    manifest_path = directory / "raw_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for name, expected in (("source_venue", venue), ("timeframe", "5m"), ("target_raw_bars_per_symbol", target)):
            if previous.get(name) != expected:
                raise ModelTrainingDatasetError(f"capture resume contract changed: {name}")
        for name, expected in end_contract.items():
            if name in previous and previous.get(name) != expected:
                raise ModelTrainingDatasetError(f"capture resume contract changed: {name}")
            if end_contract["explicit_historical_end_requested"] and name not in previous:
                raise ModelTrainingDatasetError(f"capture resume contract changed: {name}")
        if previous.get("capture_status") == "complete":
            return verify_raw_capture(directory)
        _validate_partial_capture_manifest(previous)
    directory.mkdir(parents=True, exist_ok=True)
    record_incumbent_inventory()
    import pandas as pd
    end_exclusive = pd.Timestamp(end_contract["effective_end_exclusive_utc"])
    start = end_exclusive - pd.Timedelta(minutes=5 * (target + 250))
    as_of = canonical_utc(as_of_utc or utc_now())
    fetch = fetcher or _public_fetch_range
    per_symbol: dict[str, Any] = {}
    duplicates_total = 0
    for symbol in policy["required_symbols"]:
        raw_path = directory / f"raw_{symbol}.csv"
        existing = _read_raw_csv(raw_path) if raw_path.is_file() else None
        if existing is not None:
            existing, _, _ = _normalize_ohlcv(
                existing, as_of_utc=as_of, requested_start_utc=canonical_utc(start),
                requested_end_exclusive_utc=canonical_utc(end_exclusive),
            )
        try:
            incoming = fetch(
                symbol, timeframe="5m", start_utc=canonical_utc(start),
                end_utc=canonical_utc(end_exclusive), limit=target + 250, venue=venue,
            )
        except TypeError:
            incoming = fetch(symbol, "5m", target + 250)
        diagnostics = dict(getattr(incoming, "attrs", {}).get("capture_diagnostics", {}))
        rows_before_normalization = int(len(incoming))
        normalized, duplicates, incomplete = _normalize_ohlcv(
            incoming, as_of_utc=as_of, requested_start_utc=canonical_utc(start),
            requested_end_exclusive_utc=canonical_utc(end_exclusive),
        )
        diagnostics.setdefault("pages_requested", 1)
        diagnostics.setdefault("pages_returned", 1 if rows_before_normalization else 0)
        diagnostics.setdefault("requested_start", canonical_utc(start))
        diagnostics.setdefault("requested_end", canonical_utc(end_exclusive))
        diagnostics.setdefault("rows_before_normalization", rows_before_normalization)
        diagnostics["rows_after_normalization"] = int(len(normalized))
        diagnostics.setdefault(
            "pagination_stop_reason",
            "injected_fetcher_complete" if fetcher is not None else "unknown",
        )
        diagnostics.setdefault(
            "first_exchange_timestamp",
            canonical_utc(normalized.index.min()) if len(normalized) else None,
        )
        diagnostics.setdefault(
            "last_exchange_timestamp",
            canonical_utc(normalized.index.max()) if len(normalized) else None,
        )
        combined, merge_duplicates = _merge_ohlcv(existing, normalized)
        combined = combined[combined.index < end_exclusive].tail(target)
        if len(combined) and combined.index.max() >= end_exclusive:
            raise ModelTrainingDatasetError("captured raw bar reaches or exceeds exclusive capture end")
        duplicates_total += duplicates + merge_duplicates
        _write_raw_csv(raw_path, combined)
        missing, max_gap = _gap_statistics(combined.index)
        per_symbol[symbol] = {
            "market_symbol": symbol, "requested_start_utc": canonical_utc(start),
            "requested_end_exclusive_utc": canonical_utc(end_exclusive),
            "actual_first_utc": canonical_utc(combined.index[0]) if len(combined) else None,
            "actual_last_utc": canonical_utc(combined.index[-1]) if len(combined) else None,
            "rows": int(len(combined)), "completed_rows": int(len(combined)),
            "incomplete_rows_dropped": int(incomplete), "duplicate_rows": int(duplicates + merge_duplicates),
            "conflicting_rows": 0, "missing_intervals": missing, "maximum_gap_seconds": max_gap,
            "pages_requested": diagnostics["pages_requested"],
            "pages_returned": diagnostics["pages_returned"],
            "first_exchange_timestamp": diagnostics["first_exchange_timestamp"],
            "last_exchange_timestamp": diagnostics["last_exchange_timestamp"],
            "requested_start": diagnostics["requested_start"],
            "requested_end": diagnostics["requested_end"],
            "rows_before_normalization": diagnostics["rows_before_normalization"],
            "rows_after_normalization": diagnostics["rows_after_normalization"],
            "pagination_stop_reason": diagnostics["pagination_stop_reason"],
            "file_sha256": file_digest(raw_path),
        }
    complete = all(value["completed_rows"] >= target for value in per_symbol.values())
    manifest: dict[str, Any] = {
        "schema_version": 1, "dataset_id": data_id,
        "capture_status": "complete" if complete else "historical_capture_range_incomplete",
        "captured_at": utc_now(), "capture_as_of_utc": as_of,
        "public_market_data_only": True, "source_venue": venue, "market_type": "swap",
        "market_symbols": list(policy["required_symbols"]), "timeframe": "5m",
        "target_raw_bars_per_symbol": target, "phase22_bounds": cutoffs,
        **end_contract,
        "per_symbol": per_symbol, "duplicates": duplicates_total, "conflicts": 0,
    }
    manifest["combined_raw_digest"] = json_digest({
        "contract": _raw_capture_digest_contract(manifest),
        "files": {symbol: per_symbol[symbol]["file_sha256"] for symbol in sorted(per_symbol)},
    })
    manifest["manifest_digest"] = json_digest({k: v for k, v in manifest.items() if k not in {"captured_at", "manifest_digest"}})
    atomic_write_json(manifest_path, manifest)
    verify_incumbent_inventory()
    if not complete:
        raise ModelTrainingDatasetError("historical_capture_range_incomplete")
    return verify_raw_capture(directory)


def verify_raw_capture(directory: Path | str) -> dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "raw_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("capture_status") != "complete":
        status = str(manifest.get("capture_status") or "insufficient_training_data")
        raise ModelTrainingDatasetError(status)
    calculated_manifest = json_digest({k: v for k, v in manifest.items() if k not in {"captured_at", "manifest_digest"}})
    if manifest.get("manifest_digest") != calculated_manifest:
        raise ModelTrainingDatasetError("raw manifest digest mismatch")
    cutoffs = maximum_training_timestamps()
    explicit_end = manifest.get("explicit_historical_end_requested") is True
    recorded_end = manifest.get("effective_end_exclusive_utc")
    if explicit_end and recorded_end is None:
        raise ModelTrainingDatasetError("explicit historical capture end evidence missing")
    end_contract = _capture_end_contract(cutoffs, recorded_end if explicit_end else None)
    for name, expected in end_contract.items():
        if name in manifest and manifest.get(name) != expected:
            raise ModelTrainingDatasetError(f"raw capture end contract mismatch: {name}")
    import pandas as pd
    phase22_first = pd.Timestamp(cutoffs["earliest_source_bar_open_utc"])
    file_hashes = {}
    for symbol in manifest["market_symbols"]:
        path = root / f"raw_{symbol}.csv"
        if file_digest(path) != manifest["per_symbol"][symbol]["file_sha256"]:
            raise ModelTrainingDatasetError(f"raw file digest mismatch: {symbol}")
        info = manifest["per_symbol"][symbol]
        if canonical_utc(info["requested_end_exclusive_utc"]) != end_contract["effective_end_exclusive_utc"]:
            raise ModelTrainingDatasetError(f"raw capture end mismatch: {symbol}")
        frame, duplicates, _ = _normalize_ohlcv(
            _read_raw_csv(path), as_of_utc=utc_now(),
            requested_start_utc=info["requested_start_utc"],
            requested_end_exclusive_utc=info["requested_end_exclusive_utc"],
        )
        if (duplicates or frame.index.max() >= phase22_first
                or len(frame) != int(info["completed_rows"])
                or len(frame) < int(manifest["target_raw_bars_per_symbol"])):
            raise ModelTrainingDatasetError("Phase22 excluded contract failed")
        file_hashes[symbol] = file_digest(path)
    observed = json_digest({
        "contract": _raw_capture_digest_contract(manifest),
        "files": {symbol: file_hashes[symbol] for symbol in sorted(file_hashes)},
    })
    if observed != manifest.get("combined_raw_digest"):
        raise ModelTrainingDatasetError("combined raw digest mismatch")
    return manifest


def _raw_manifest_from(source: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    manifest_path = path / "raw_manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise ModelTrainingDatasetError("raw capture manifest required for overlap verification")
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def verify_raw_capture_non_overlap(
    earlier_capture: Mapping[str, Any] | Path | str,
    later_capture: Mapping[str, Any] | Path | str,
    *,
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT"),
) -> dict[str, Any]:
    """Require every earlier last timestamp to precede the later first timestamp."""
    import pandas as pd

    earlier = _raw_manifest_from(earlier_capture)
    later = _raw_manifest_from(later_capture)
    required = tuple(str(symbol) for symbol in symbols)
    if not required or any(
        symbol not in earlier.get("per_symbol", {}) or symbol not in later.get("per_symbol", {})
        for symbol in required
    ):
        raise ModelTrainingDatasetError("raw capture overlap evidence lacks required symbols")
    evidence: dict[str, Any] = {}
    for symbol in required:
        earlier_info = earlier["per_symbol"][symbol]
        later_info = later["per_symbol"][symbol]
        try:
            earlier_first = pd.Timestamp(earlier_info["actual_first_utc"])
            earlier_last = pd.Timestamp(earlier_info["actual_last_utc"])
            later_first = pd.Timestamp(later_info["actual_first_utc"])
            later_last = pd.Timestamp(later_info["actual_last_utc"])
            if any(pd.isna(value) for value in (earlier_first, earlier_last, later_first, later_last)):
                raise ValueError("missing timestamp")
            values = [
                value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
                for value in (earlier_first, earlier_last, later_first, later_last)
            ]
            earlier_first, earlier_last, later_first, later_last = values
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelTrainingDatasetError("invalid raw capture overlap timestamps") from exc
        if earlier_first > earlier_last or later_first > later_last:
            raise ModelTrainingDatasetError("invalid raw capture timestamp range")
        if earlier_last >= later_first:
            raise ModelTrainingDatasetError(f"raw capture timestamp overlap: {symbol}")
        evidence[symbol] = {
            "earlier_first_utc": canonical_utc(earlier_first),
            "earlier_last_utc": canonical_utc(earlier_last),
            "later_first_utc": canonical_utc(later_first),
            "later_last_utc": canonical_utc(later_last),
            "strictly_prior": True,
        }
    return {
        "passed": True,
        "relationship": "strictly_prior_no_timestamp_overlap",
        "symbols": evidence,
    }


def _write_deterministic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    os.replace(temporary, path)


def chronological_split(
    timestamps_by_symbol: Mapping[str, np.ndarray], *, train_fraction: float = 0.70,
    validation_fraction: float = 0.15, purge_bars: int = 60, timeframe_seconds: int = 300,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_times = sorted(set(np.concatenate([np.asarray(v, dtype=np.int64) for v in timestamps_by_symbol.values()]).tolist()))
    if len(all_times) < 3:
        raise ModelTrainingDatasetError("insufficient_training_data")
    first_index = int(math.floor(len(all_times) * train_fraction))
    second_index = int(math.floor(len(all_times) * (train_fraction + validation_fraction)))
    if first_index <= purge_bars or second_index <= first_index + purge_bars or second_index >= len(all_times):
        raise ModelTrainingDatasetError("insufficient_training_data")
    boundary1, boundary2 = int(all_times[first_index]), int(all_times[second_index])
    step_ns = timeframe_seconds * 1_000_000_000
    train_stop = boundary1 - purge_bars * step_ns
    validation_stop = boundary2 - purge_bars * step_ns
    assignments: dict[str, np.ndarray] = {}
    for symbol, timestamps in timestamps_by_symbol.items():
        ts = np.asarray(timestamps, dtype=np.int64)
        split = np.full(len(ts), -1, dtype=np.int8)
        split[ts < train_stop] = 0
        split[(ts >= boundary1) & (ts < validation_stop)] = 1
        split[ts >= boundary2] = 2
        assignments[symbol] = split
    info = {
        "method": "global_utc_chronological",
        "train_fraction": train_fraction, "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "train_validation_boundary_ns": boundary1,
        "validation_test_boundary_ns": boundary2,
        "train_purge_range_ns": [train_stop, boundary1],
        "validation_purge_range_ns": [validation_stop, boundary2],
        "purge_bar_count_each_boundary": purge_bars,
        "global_split_timestamps_utc": {
            "train_start": datetime.fromtimestamp(all_times[0] / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "train_end_exclusive": datetime.fromtimestamp(train_stop / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "validation_start": datetime.fromtimestamp(boundary1 / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "validation_end_exclusive": datetime.fromtimestamp(validation_stop / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "internal_test_start": datetime.fromtimestamp(boundary2 / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "internal_test_end_inclusive": datetime.fromtimestamp(all_times[-1] / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "purge_ranges_utc": [
            [
                datetime.fromtimestamp(train_stop / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                datetime.fromtimestamp(boundary1 / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            ],
            [
                datetime.fromtimestamp(validation_stop / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                datetime.fromtimestamp(boundary2 / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            ],
        ],
    }
    info["split_digest"] = json_digest(info)
    return assignments, info


def _valid_sequence_endpoints(
    split: np.ndarray, finite_label: np.ndarray, sequence_length: int, split_code: int,
) -> tuple[np.ndarray, int]:
    positions = np.flatnonzero(np.asarray(split) == int(split_code))
    endpoints: list[int] = []
    dropped = 0
    segment_start = 0
    for offset, pos in enumerate(positions):
        if offset == 0 or pos != positions[offset - 1] + 1:
            segment_start = offset
        if offset - segment_start < sequence_length - 1:
            dropped += int(bool(finite_label[pos]))
        elif finite_label[pos]:
            endpoints.append(int(pos))
    return np.asarray(endpoints, dtype=np.int64), int(dropped)


def _valid_sequence_count(split: np.ndarray, finite_label: np.ndarray, sequence_length: int) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    context: dict[str, int] = {}
    for code, name in ((0, "train"), (1, "validation"), (2, "internal_test")):
        endpoints, dropped = _valid_sequence_endpoints(split, finite_label, sequence_length, code)
        counts[name], context[name] = int(len(endpoints)), dropped
    return counts, context


def build_dataset(
    dataset: Path | str, *, fit_scaler: bool = True, minimum_usable_rows: int | None = None
) -> dict[str, Any]:
    validate_phase24_evidence()
    policy = load_training_policy()
    contract = specification_contract()
    root = Path(dataset)
    raw_manifest = verify_raw_capture(root)
    if (root / "dataset_manifest.json").exists():
        raise ModelTrainingDatasetError("frozen dataset already exists")
    record_incumbent_inventory()
    from features import build_features, canonical_feature_columns
    from ml_dl.dl_labels import next_k_logret, next_k_rv, triple_barrier_label
    from tools.model_objective_label_audit import resolve_target_contract
    columns = canonical_feature_columns(True)
    if columns != contract["feature_names"] or len(columns) != int(policy["feature_count"]):
        raise ModelTrainingDatasetError("27-feature contract mismatch")
    feature_code_digest = file_digest(BASE_DIR / "features.py")
    label_code_digest = file_digest(BASE_DIR / "ml_dl" / "dl_labels.py")
    import pandas as pd
    resolved_targets = resolve_target_contract(contract["label_contract"])
    maximum_lookahead = int(resolved_targets["maximum_required_purge_bars"])
    max_endpoint_ns = int(pd.Timestamp(
        maximum_training_timestamps(max_lookahead=maximum_lookahead)["maximum_training_labeled_endpoint_utc"]
    ).value)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    eligible_times: dict[str, np.ndarray] = {}
    minimum = int(minimum_usable_rows or policy["minimum_usable_labeled_rows_per_symbol"])
    if minimum < int(policy["minimum_usable_labeled_rows_per_symbol"]):
        raise ModelTrainingDatasetError("CLI may not weaken minimum usable row gate")
    per_symbol: dict[str, Any] = {}
    for symbol in policy["required_symbols"]:
        raw = _read_raw_csv(root / f"raw_{symbol}.csv")
        raw.index = __import__("pandas").to_datetime(raw.index, utc=True)
        features = build_features(raw)
        features = features.copy()
        symbol_id = contract["symbol_id_map"].get(symbol)
        if type(symbol_id) is not int:
            raise ModelTrainingDatasetError(f"persisted symbol ID missing: {symbol}")
        features["symbol_id"] = float(symbol_id)
        matrix_frame = features[columns].astype(np.float32)
        if matrix_frame.shape[1] != 27 or not np.isfinite(matrix_frame.to_numpy()).all():
            raise ModelTrainingDatasetError("feature matrix is not finite width 27")
        prices = raw["close"].to_numpy(dtype=np.float64)
        cfg = contract["label_contract"]
        y = triple_barrier_label(prices, pt=cfg["pt"], sl=cfg["sl"], max_hold=cfg["max_hold"])
        ret = next_k_logret(prices, cfg["max_hold"])
        rv = next_k_rv(np.log(prices), cfg["horizon"])
        raw_ns = raw.index.asi8
        positions = np.searchsorted(raw_ns, matrix_frame.index.asi8)
        y_aligned, ret_aligned, rv_aligned = y[positions], ret[positions], rv[positions]
        timestamps = matrix_frame.index.asi8.astype(np.int64)
        before_phase22 = timestamps <= max_endpoint_ns
        finite_label = np.isfinite(y_aligned) & np.isfinite(ret_aligned) & np.isfinite(rv_aligned) & before_phase22
        usable = int(finite_label.sum())
        if usable < minimum:
            raise ModelTrainingDatasetError("insufficient_training_data")
        eligible_times[symbol] = timestamps[finite_label]
        arrays[symbol] = {
            "features": matrix_frame.to_numpy(dtype=np.float32), "timestamps": timestamps,
            "ret_cls": y_aligned.astype(np.float64), "ret_reg": ret_aligned.astype(np.float64),
            "rv_reg": rv_aligned.astype(np.float64), "finite_label": finite_label,
        }
        classes, class_counts = np.unique(y_aligned[finite_label].astype(np.int64), return_counts=True)
        per_symbol[symbol] = {
            "symbol_id": symbol_id, "feature_rows": int(len(timestamps)), "usable_labeled_rows": usable,
            "timestamp_first_utc": canonical_utc(matrix_frame.index[0]),
            "timestamp_last_utc": canonical_utc(matrix_frame.index[-1]),
            "class_balance": {str(int(key)): int(value) for key, value in zip(classes, class_counts)},
        }
    splits, split_info = chronological_split(
        eligible_times, train_fraction=policy["train_fraction"],
        validation_fraction=policy["validation_fraction"], purge_bars=maximum_lookahead,
    )
    feature_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    training_ret_targets: list[float] = []
    training_rv_targets: list[float] = []
    training_sequence_count_by_symbol: dict[str, int] = {}
    for symbol in policy["required_symbols"]:
        data = arrays[symbol]
        # Recompute split codes over every feature row using the shared exact boundaries.
        ts = data["timestamps"]
        boundary1 = split_info["train_validation_boundary_ns"]
        boundary2 = split_info["validation_test_boundary_ns"]
        train_stop = split_info["train_purge_range_ns"][0]
        val_stop = split_info["validation_purge_range_ns"][0]
        split = np.full(len(ts), -1, dtype=np.int8)
        split[ts < train_stop] = 0
        split[(ts >= boundary1) & (ts < val_stop)] = 1
        split[ts >= boundary2] = 2
        data["split"] = split
        feature_path = root / f"features_{symbol}.npz"
        label_path = root / f"labels_{symbol}.npz"
        _write_deterministic_npz(feature_path, features=data["features"], timestamps=ts, split=split)
        _write_deterministic_npz(
            label_path, timestamps=ts, ret_cls=data["ret_cls"], ret_reg=data["ret_reg"], rv_reg=data["rv_reg"]
        )
        feature_hashes[symbol], label_hashes[symbol] = file_digest(feature_path), file_digest(label_path)
        counts, context = _valid_sequence_count(split, data["finite_label"], policy["sequence_length"])
        train_endpoints, _ = _valid_sequence_endpoints(
            split, data["finite_label"], policy["sequence_length"], 0
        )
        training_ret_targets.extend(data["ret_reg"][train_endpoints].tolist())
        training_rv_targets.extend(data["rv_reg"][train_endpoints].tolist())
        training_sequence_count_by_symbol[symbol] = int(len(train_endpoints))
        per_symbol[symbol]["rows_by_split"] = {
            name: int(np.sum((split == code) & data["finite_label"]))
            for code, name in ((0, "train"), (1, "validation"), (2, "internal_test"))
        }
        per_symbol[symbol]["valid_sequences_by_split"] = counts
        per_symbol[symbol]["sequence_context_drops_by_split"] = context
        per_symbol[symbol]["purged_rows"] = int(np.sum(split == -1))
        if any(counts[name] <= 0 for name in counts):
            raise ModelTrainingDatasetError(f"insufficient per-symbol sequences: {symbol}")
    from tools.model_candidate_objective import compute_training_target_scales
    target_scales = compute_training_target_scales(training_ret_targets, training_rv_targets)
    target_scales.update({
        "training_sequence_count_by_symbol": training_sequence_count_by_symbol,
        "validation_targets_consulted": False,
        "internal_test_targets_consulted": False,
        "legacy_repair_targets_consulted": False,
        "confirmation_targets_consulted": False,
    })
    target_scales["target_scale_digest"] = json_digest({
        key: value for key, value in target_scales.items() if key != "target_scale_digest"
    })
    manifest: dict[str, Any] = {
        "schema_version": 1, "dataset_id": raw_manifest["dataset_id"], "dataset_status": "features_labels_split_built",
        "built_at": utc_now(), "git_commit": git_commit(), "source_venue": raw_manifest["source_venue"],
        "timeframe": "5m", "symbols": list(policy["required_symbols"]),
        "supported_symbols": list(policy["required_symbols"]), "symbol_id_map": contract["symbol_id_map"],
        "sequence_length": policy["sequence_length"], "feature_count": len(columns),
        "ordered_feature_names": columns, "feature_contract_digest": contract["feature_contract_digest"],
        "features_py_digest": feature_code_digest,
        "label_contract": contract["label_contract"], "label_contract_digest": contract["label_contract_digest"],
        "label_implementation_digest": label_code_digest,
        "resolved_target_contract_digest": resolved_targets["target_contract_digest"],
        "maximum_target_lookahead_bars": maximum_lookahead,
        "raw_data_digest": raw_manifest["combined_raw_digest"],
        "feature_file_digests": feature_hashes, "label_file_digests": label_hashes,
        "feature_digest": json_digest(feature_hashes), "label_digest": json_digest(label_hashes),
        "split": split_info, "split_digest": split_info["split_digest"], "per_symbol": per_symbol,
        "phase22_excluded": True, "per_symbol_feature_build": True, "per_symbol_label_build": True,
        "per_symbol_sequence_construction_required": True,
        "target_scales": target_scales, "scaler": None,
    }
    manifest["dataset_digest"] = json_digest({
        "raw": manifest["raw_data_digest"], "features": manifest["feature_digest"],
        "labels": manifest["label_digest"], "split": manifest["split_digest"],
        "feature_contract": manifest["feature_contract_digest"], "label_contract": manifest["label_contract_digest"],
        "target_scale": manifest["target_scales"]["target_scale_digest"],
    })
    manifest["manifest_digest"] = json_digest({k: v for k, v in manifest.items() if k not in {"built_at", "manifest_digest"}})
    atomic_write_json(root / "dataset_manifest.json", manifest)
    if fit_scaler:
        manifest = fit_frozen_scaler(root)
    verify_incumbent_inventory()
    return manifest


def fit_frozen_scaler(
    dataset: Path | str, *, require_canonical_version: bool = True
) -> dict[str, Any]:
    root = Path(dataset)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("scaler") is not None or (root / "scaler.joblib").exists():
        raise ModelTrainingDatasetError("frozen scaler already exists")
    from sklearn import __version__ as sklearn_version
    from sklearn.preprocessing import StandardScaler
    import joblib
    if require_canonical_version and sklearn_version != "1.8.0":
        raise ModelTrainingDatasetError("canonical scikit-learn 1.8.0 required for frozen scaler")
    rows = []
    row_counts = {}
    for symbol in manifest["symbols"]:
        with np.load(root / f"features_{symbol}.npz", allow_pickle=False) as data:
            matrix, split = data["features"], data["split"]
        selected = matrix[split == 0]
        rows.append(selected.astype(np.float64, copy=False))
        row_counts[symbol] = int(len(selected))
    pooled = np.concatenate(rows, axis=0)
    if pooled.shape[1] != 27 or not np.isfinite(pooled).all():
        raise ModelTrainingDatasetError("scaler fit matrix is not finite width 27")
    scaler = StandardScaler().fit(pooled)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    near_zero = int(np.sum(np.abs(scale) <= 1e-12))
    if len(mean) != 27 or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ModelTrainingDatasetError("invalid frozen scaler state")
    if int(np.sum(scale == 0.0)) or near_zero:
        raise ModelTrainingDatasetError("frozen scaler has zero or unacceptable near-zero scales")
    scaler_path = root / "scaler.joblib"
    joblib.dump(scaler, scaler_path, compress=0, protocol=4)
    manifest["scaler"] = {
        "sha256": file_digest(scaler_path), "fit_split": "train_only",
        "fit_rows": int(len(pooled)), "fit_rows_by_symbol": row_counts,
        "symbols": list(manifest["symbols"]), "feature_width": 27,
        "ordered_feature_names": manifest["ordered_feature_names"],
        "mean_digest": hashlib.sha256(mean.tobytes(order="C")).hexdigest(),
        "scale_digest": hashlib.sha256(scale.tobytes(order="C")).hexdigest(),
        "sklearn_version": sklearn_version, "zero_scale_count": 0,
        "near_zero_scale_count": near_zero,
    }
    manifest["dataset_status"] = "frozen_ready"
    manifest["dataset_digest"] = json_digest({
        "raw": manifest["raw_data_digest"], "features": manifest["feature_digest"],
        "labels": manifest["label_digest"], "split": manifest["split_digest"],
        "scaler": manifest["scaler"]["sha256"], "feature_contract": manifest["feature_contract_digest"],
        "label_contract": manifest["label_contract_digest"],
        "target_scale": manifest["target_scales"]["target_scale_digest"],
    })
    manifest["manifest_digest"] = json_digest({k: v for k, v in manifest.items() if k not in {"built_at", "manifest_digest"}})
    atomic_write_json(manifest_path, manifest)
    return manifest


def verify_dataset(dataset: Path | str) -> dict[str, Any]:
    root = Path(dataset)
    raw = json.loads((root / "raw_manifest.json").read_text(encoding="utf-8-sig"))
    if raw.get("capture_status") != "complete":
        raise ModelTrainingDatasetError("insufficient_training_data")
    if raw.get("manifest_digest") != json_digest({
        k: v for k, v in raw.items() if k not in {"captured_at", "manifest_digest"}
    }):
        raise ModelTrainingDatasetError("raw manifest digest mismatch")
    raw_hashes: dict[str, str] = {}
    for symbol in raw.get("market_symbols", []):
        raw_path = root / f"raw_{symbol}.csv"
        observed = file_digest(raw_path)
        if observed != raw["per_symbol"][symbol]["file_sha256"]:
            raise ModelTrainingDatasetError(f"raw file digest mismatch: {symbol}")
        raw_hashes[symbol] = observed
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        if (len(raw_rows) != int(raw["per_symbol"][symbol]["completed_rows"])
                or len(raw_rows) < int(raw["target_raw_bars_per_symbol"])):
            raise ModelTrainingDatasetError(f"raw completed-row count mismatch: {symbol}")
    observed_raw_digest = json_digest({
        "contract": _raw_capture_digest_contract(raw),
        "files": {symbol: raw_hashes[symbol] for symbol in sorted(raw_hashes)},
    })
    if observed_raw_digest != raw.get("combined_raw_digest"):
        raise ModelTrainingDatasetError("combined raw digest mismatch")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("dataset_status") != "frozen_ready" or manifest.get("raw_data_digest") != raw["combined_raw_digest"]:
        raise ModelTrainingDatasetError("dataset is not frozen and ready")
    if manifest.get("manifest_digest") != json_digest({
        k: v for k, v in manifest.items() if k not in {"built_at", "manifest_digest"}
    }):
        raise ModelTrainingDatasetError("dataset manifest digest mismatch")
    if manifest.get("feature_count") != 27 or len(manifest.get("ordered_feature_names", [])) != 27:
        raise ModelTrainingDatasetError("dataset feature width mismatch")
    observed_ret_targets: list[float] = []
    observed_rv_targets: list[float] = []
    observed_train_counts: dict[str, int] = {}
    for symbol in manifest["symbols"]:
        if file_digest(root / f"features_{symbol}.npz") != manifest["feature_file_digests"][symbol]:
            raise ModelTrainingDatasetError(f"feature file digest mismatch: {symbol}")
        if file_digest(root / f"labels_{symbol}.npz") != manifest["label_file_digests"][symbol]:
            raise ModelTrainingDatasetError(f"label file digest mismatch: {symbol}")
        with np.load(root / f"features_{symbol}.npz", allow_pickle=False) as values:
            ts, split, matrix = values["timestamps"], values["split"], values["features"]
        with np.load(root / f"labels_{symbol}.npz", allow_pickle=False) as values:
            label_ts = values["timestamps"]
            labels = {name: values[name] for name in ("ret_cls", "ret_reg", "rv_reg")}
        if matrix.shape[1] != 27 or not np.all(np.diff(ts) > 0) or set(np.unique(split)) - {-1, 0, 1, 2}:
            raise ModelTrainingDatasetError(f"invalid frozen feature arrays: {symbol}")
        if not np.array_equal(ts, label_ts) or any(len(value) != len(ts) for value in labels.values()):
            raise ModelTrainingDatasetError(f"feature/label timestamp mismatch: {symbol}")
        finite_label = np.isfinite(labels["ret_cls"]) & np.isfinite(labels["ret_reg"]) & np.isfinite(labels["rv_reg"])
        counts, context = _valid_sequence_count(split, finite_label, int(manifest["sequence_length"]))
        train_endpoints, _ = _valid_sequence_endpoints(
            split, finite_label, int(manifest["sequence_length"]), 0
        )
        observed_ret_targets.extend(np.asarray(labels["ret_reg"])[train_endpoints].tolist())
        observed_rv_targets.extend(np.asarray(labels["rv_reg"])[train_endpoints].tolist())
        observed_train_counts[symbol] = int(len(train_endpoints))
        if counts != manifest["per_symbol"][symbol]["valid_sequences_by_split"]:
            raise ModelTrainingDatasetError(f"sequence count integrity mismatch: {symbol}")
        if context != manifest["per_symbol"][symbol]["sequence_context_drops_by_split"]:
            raise ModelTrainingDatasetError(f"sequence context integrity mismatch: {symbol}")
    from tools.model_candidate_objective import compute_training_target_scales
    observed_scales = compute_training_target_scales(observed_ret_targets, observed_rv_targets)
    observed_scales.update({
        "training_sequence_count_by_symbol": observed_train_counts,
        "validation_targets_consulted": False,
        "internal_test_targets_consulted": False,
        "legacy_repair_targets_consulted": False,
        "confirmation_targets_consulted": False,
    })
    observed_scales["target_scale_digest"] = json_digest({
        key: value for key, value in observed_scales.items() if key != "target_scale_digest"
    })
    if manifest.get("target_scales") != observed_scales:
        raise ModelTrainingDatasetError("training-sequence target-scale integrity mismatch")
    scaler_path = root / "scaler.joblib"
    if file_digest(scaler_path) != manifest["scaler"]["sha256"]:
        raise ModelTrainingDatasetError("frozen scaler digest mismatch")
    import joblib
    scaler = joblib.load(scaler_path)
    mean, scale = np.asarray(scaler.mean_), np.asarray(scaler.scale_)
    if len(mean) != 27 or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ModelTrainingDatasetError("invalid frozen scaler state")
    return manifest


def _load_selection_freeze(path: Path | str = SELECTION_FREEZE) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ModelTrainingDatasetError("candidate-selection freeze manifest required")
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if value.get("selection_frozen") is not True or set(value.get("candidates", {})) != {"lstm", "tcn", "tx"}:
        raise ModelTrainingDatasetError("candidate-selection freeze manifest incomplete")
    recorded = value.get("freeze_digest")
    if recorded != json_digest({k: v for k, v in value.items() if k not in {"frozen_at", "freeze_digest"}}):
        raise ModelTrainingDatasetError("candidate-selection freeze manifest digest mismatch")
    for item in value["candidates"].values():
        if not item.get("candidate_id") or not item.get("candidate_model_digest") or not item.get("candidate_scaler_digest"):
            raise ModelTrainingDatasetError("candidate-selection freeze lacks immutable digests")
        if item.get("internal_test_recorded") is not True:
            raise ModelTrainingDatasetError("all internal test results must be recorded before confirmation capture")
    return value


def capture_confirmation_data(
    *, confirmation_id: str | None = None, as_of_utc: Any | None = None,
    fetcher: Callable[..., Any] | None = None, freeze_path: Path | str = SELECTION_FREEZE,
    confirmation_root: Path | str = CONFIRMATION_ROOT,
) -> dict[str, Any]:
    validate_phase24_evidence()
    policy = load_training_policy()
    contract = specification_contract()
    freeze = _load_selection_freeze(freeze_path)
    record_incumbent_inventory()
    import pandas as pd
    final_phase22 = pd.Timestamp(phase22_source_bounds()["final_source_bar_open_utc"])
    as_of = canonical_utc(as_of_utc or utc_now())
    capture_id = confirmation_id or (
        f"confirmation_5m_{freeze['freeze_digest'][:8]}_" +
        pd.Timestamp(as_of).strftime("%Y%m%dT%H%M%SZ")
    )
    target = Path(confirmation_root) / capture_id
    if target.exists():
        raise ModelTrainingDatasetError("sealed confirmation set already exists and is immutable")
    Path(confirmation_root).mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{capture_id}.partial-", dir=Path(confirmation_root)))
    fetch = fetcher or _public_fetch_range
    venue = str(freeze.get("source_venue") or "")
    if not venue:
        shutil.rmtree(stage, ignore_errors=True)
        raise ModelTrainingDatasetError("selection freeze source venue missing")
    target_bars = int(policy["confirmation_target_bars_per_symbol"])
    minimum = int(policy["confirmation_minimum_bars_per_symbol"])
    source_hashes: dict[str, str] = {}
    feature_hashes: dict[str, str] = {}
    window_hashes: dict[str, str] = {}
    per_symbol: dict[str, Any] = {}
    try:
        from features import build_features, canonical_feature_columns
        columns = canonical_feature_columns(True)
        start = final_phase22 + pd.Timedelta(minutes=5)
        for symbol in policy["required_symbols"]:
            try:
                incoming = fetch(
                    symbol, timeframe="5m", start_utc=canonical_utc(start), end_utc=as_of,
                    limit=target_bars + policy["sequence_length"] + 100, venue=venue,
                )
            except TypeError:
                incoming = fetch(symbol, "5m", target_bars + policy["sequence_length"] + 100)
            raw, duplicates, incomplete = _normalize_ohlcv(incoming, as_of_utc=as_of)
            raw = raw[raw.index > final_phase22]
            if len(raw) == 0 or raw.index.min() <= final_phase22:
                raise ModelTrainingDatasetError("confirmation timestamps are not strictly post-Phase22")
            features = build_features(raw).copy()
            features["symbol_id"] = float(contract["symbol_id_map"][symbol])
            matrix = features[columns].to_numpy(dtype=np.float32)
            timestamps = features.index.asi8.astype(np.int64)
            possible = np.arange(policy["sequence_length"] - 1, len(matrix), dtype=np.int64)
            retained = possible[-target_bars:]
            if len(retained) < minimum:
                raise ModelTrainingDatasetError("sealed_confirmation_pending")
            windows = np.stack([
                matrix[index - policy["sequence_length"] + 1:index + 1] for index in retained
            ])
            endpoints = timestamps[retained]
            raw_path = stage / f"raw_{symbol}.csv"
            window_path = stage / f"windows_{symbol}.npz"
            _write_raw_csv(raw_path, raw)
            _write_deterministic_npz(window_path, windows=windows, endpoint_timestamps=endpoints)
            source_hashes[symbol], window_hashes[symbol] = file_digest(raw_path), file_digest(window_path)
            feature_hashes[symbol] = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
            per_symbol[symbol] = {
                "first_timestamp_utc": canonical_utc(pd.Timestamp(endpoints[0], unit="ns", tz="UTC")),
                "last_timestamp_utc": canonical_utc(pd.Timestamp(endpoints[-1], unit="ns", tz="UTC")),
                "unique_completed_bars": int(len(endpoints)), "source_rows_with_context": int(len(raw)),
                "duplicates": int(duplicates), "conflicts": 0, "incomplete_rows_dropped": int(incomplete),
                "source_digest": source_hashes[symbol], "feature_digest": feature_hashes[symbol],
                "window_digest": window_hashes[symbol],
            }
        manifest: dict[str, Any] = {
            "schema_version": 1, "confirmation_id": capture_id, "capture_as_of_utc": as_of,
            "captured_at": utc_now(), "source_venue": venue, "public_market_data_only": True,
            "timeframe": "5m", "symbols": list(policy["required_symbols"]),
            "sequence_length": policy["sequence_length"], "feature_count": policy["feature_count"],
            "ordered_feature_names": columns, "feature_contract_digest": contract["feature_contract_digest"],
            "phase22_final_source_bar_open_utc": canonical_utc(final_phase22),
            "selection_freeze_digest": freeze["freeze_digest"], "labels_present": False,
            "per_symbol": per_symbol, "source_digest": json_digest(source_hashes),
            "feature_digest": json_digest(feature_hashes), "window_digest": json_digest(window_hashes),
        }
        manifest["confirmation_digest"] = json_digest({
            "source": manifest["source_digest"], "features": manifest["feature_digest"],
            "windows": manifest["window_digest"], "freeze": manifest["selection_freeze_digest"],
        })
        manifest["manifest_digest"] = json_digest({k: v for k, v in manifest.items() if k not in {"captured_at", "manifest_digest"}})
        atomic_write_json(stage / "confirmation_manifest.json", manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(target)
        result = verify_confirmation(target, freeze_path=freeze_path)
        verify_incumbent_inventory()
        return result
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_confirmation(
    directory: Path | str, *, freeze_path: Path | str = SELECTION_FREEZE
) -> dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "confirmation_manifest.json").read_text(encoding="utf-8-sig"))
    policy = load_training_policy()
    if manifest.get("labels_present") is not False or manifest.get("timeframe") != "5m":
        raise ModelTrainingDatasetError("sealed confirmation integrity failure")
    if manifest.get("manifest_digest") != json_digest({
        k: v for k, v in manifest.items() if k not in {"captured_at", "manifest_digest"}
    }):
        raise ModelTrainingDatasetError("sealed confirmation manifest digest mismatch")
    freeze = _load_selection_freeze(freeze_path)
    if manifest.get("selection_freeze_digest") != freeze.get("freeze_digest"):
        raise ModelTrainingDatasetError("confirmation selection-freeze digest mismatch")
    if manifest.get("source_venue") != freeze.get("source_venue"):
        raise ModelTrainingDatasetError("confirmation source venue differs from training source")
    final_phase22 = datetime.fromisoformat(manifest["phase22_final_source_bar_open_utc"].replace("Z", "+00:00"))
    source_hashes: dict[str, str] = {}
    window_hashes: dict[str, str] = {}
    for symbol in policy["required_symbols"]:
        info = manifest["per_symbol"][symbol]
        if info["unique_completed_bars"] < policy["confirmation_minimum_bars_per_symbol"]:
            raise ModelTrainingDatasetError("sealed_confirmation_pending")
        if datetime.fromisoformat(info["first_timestamp_utc"].replace("Z", "+00:00")) <= final_phase22:
            raise ModelTrainingDatasetError("confirmation overlaps Phase22")
        if file_digest(root / f"raw_{symbol}.csv") != info["source_digest"]:
            raise ModelTrainingDatasetError("confirmation source digest mismatch")
        if file_digest(root / f"windows_{symbol}.npz") != info["window_digest"]:
            raise ModelTrainingDatasetError("confirmation window digest mismatch")
        source_hashes[symbol] = info["source_digest"]
        window_hashes[symbol] = info["window_digest"]
        with (root / f"raw_{symbol}.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        timestamps = [datetime.fromisoformat(row["bar_open_utc"].replace("Z", "+00:00")) for row in rows]
        if len(timestamps) != len(set(timestamps)) or any(value <= final_phase22 for value in timestamps):
            raise ModelTrainingDatasetError("confirmation raw timestamps overlap or duplicate")
        with np.load(root / f"windows_{symbol}.npz", allow_pickle=False) as values:
            windows, endpoints = values["windows"], values["endpoint_timestamps"]
        if (windows.shape[1:] != (64, 27) or len(np.unique(endpoints)) != len(endpoints)
                or not np.isfinite(windows).all()):
            raise ModelTrainingDatasetError("confirmation windows invalid")
    if manifest.get("source_digest") != json_digest(source_hashes):
        raise ModelTrainingDatasetError("confirmation combined source digest mismatch")
    if manifest.get("window_digest") != json_digest(window_hashes):
        raise ModelTrainingDatasetError("confirmation combined window digest mismatch")
    expected = json_digest({
        "source": manifest["source_digest"], "features": manifest["feature_digest"],
        "windows": manifest["window_digest"], "freeze": manifest["selection_freeze_digest"],
    })
    if expected != manifest.get("confirmation_digest"):
        raise ModelTrainingDatasetError("confirmation digest mismatch")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--dataset-id")
    capture.add_argument("--venue", default="bitget")
    capture.add_argument("--as-of-utc")
    capture.add_argument("--capture-end-exclusive-utc")
    build = sub.add_parser("build")
    build.add_argument("--dataset", required=True)
    build.add_argument("--training-python", default=str(TRAINING_PYTHON))
    verify = sub.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--dataset", required=True)
    confirmation = sub.add_parser("capture-confirmation")
    confirmation.add_argument("--confirmation-id")
    confirmation.add_argument("--as-of-utc")
    confirmation.add_argument("--freeze-manifest", default=str(SELECTION_FREEZE))
    verify_confirmation_parser = sub.add_parser("verify-confirmation")
    verify_confirmation_parser.add_argument("--confirmation", required=True)
    # Internal worker: invoked only by build using the visibly separate
    # training interpreter. It has no market-data read path.
    scaler = sub.add_parser("_fit-scaler", help=argparse.SUPPRESS)
    scaler.add_argument("--dataset", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_training_data(
                dataset_id=args.dataset_id, venue=args.venue, as_of_utc=args.as_of_utc,
                capture_end_exclusive_utc=args.capture_end_exclusive_utc,
            )
        elif args.command == "build":
            python = Path(args.training_python)
            if (not python.is_file() or not TRAINING_PYTHON.is_file()
                    or python.resolve() != TRAINING_PYTHON.resolve()
                    or python.resolve() == Path(sys.executable).resolve()):
                raise ModelTrainingDatasetError("canonical training interpreter required for scaler fit")
            dataset_manifest_path = Path(args.dataset) / "dataset_manifest.json"
            if dataset_manifest_path.is_file():
                existing = json.loads(dataset_manifest_path.read_text(encoding="utf-8-sig"))
                if existing.get("dataset_status") == "frozen_ready":
                    result = verify_dataset(args.dataset)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    return 0
                if existing.get("dataset_status") != "features_labels_split_built":
                    raise ModelTrainingDatasetError("partial dataset state is not resumable")
                result = existing
            else:
                result = build_dataset(args.dataset, fit_scaler=False)
            completed = subprocess.run(
                [str(python), str(Path(__file__).resolve()), "_fit-scaler", "--dataset", str(args.dataset)],
                cwd=BASE_DIR,
            )
            if completed.returncode:
                raise ModelTrainingDatasetError("canonical scaler worker failed")
            result = verify_dataset(args.dataset)
        elif args.command == "_fit-scaler":
            if not TRAINING_PYTHON.is_file() or Path(sys.executable).resolve() != TRAINING_PYTHON.resolve():
                raise ModelTrainingDatasetError(
                    "scaler fit must use .venv-model-training/canonical/Scripts/python.exe"
                )
            result = fit_frozen_scaler(args.dataset)
        elif args.command == "verify":
            result = verify_dataset(args.dataset)
        elif args.command == "describe":
            result = verify_dataset(args.dataset)
        elif args.command == "capture-confirmation":
            result = capture_confirmation_data(
                confirmation_id=args.confirmation_id, as_of_utc=args.as_of_utc,
                freeze_path=args.freeze_manifest,
            )
        else:
            result = verify_confirmation(args.confirmation)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ModelTrainingDatasetError, CandidateTrainingEnvironmentError) as exc:
        print(json.dumps({"status": str(exc), "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
