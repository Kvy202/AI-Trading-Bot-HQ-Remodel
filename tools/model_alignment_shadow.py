"""Phase 22 training-serving alignment capture, evaluation, and live shadow.

Every command is research-only.  This module has no executor imports and never
writes signal or trade logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features import build_features, canonical_feature_columns
from ml_dl.dl_ensemble import (
    canonical_utc,
    completed_bar_mask,
    feature_window_digest,
    refresh_live_features_per_symbol,
    source_bar_id,
    timeframe_duration_seconds,
)
from runtime.model_serving_guard import guard_from_snapshot
from tools.model_serving_snapshot import (
    capture_model_serving_snapshot,
    load_model_serving_snapshot,
    write_model_serving_snapshot,
)


DEFAULT_POLICY = BASE_DIR / "research" / "model_alignment_policy.json"
POLICY_TYPES: dict[str, type] = {
    "schema_version": int,
    "required_timeframe": str,
    "required_sequence_length": int,
    "required_served_feature_count": int,
    "historical_unique_bars_required": int,
    "historical_capture_target": int,
    "live_unique_bars_required": int,
    "completed_bar_grace_seconds": int,
    "maximum_gap_bars": int,
    "maximum_missing_rate": float,
    "flat_output_std_threshold": float,
    "flat_output_window": int,
    "extreme_probability_threshold_low": float,
    "extreme_probability_threshold_high": float,
    "extreme_consecutive_limit": int,
    "deterministic_repeat_tolerance": float,
    "require_exact_sklearn_version_for_reproducibility": bool,
    "allow_symbol_subset_of_training": bool,
    "require_completed_bars": bool,
}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
MODEL_OUTPUT_COLUMNS = [
    "source_bar_id", "source_bar_open_utc", "source_bar_close_utc", "symbol",
    "feature_window_digest", "model_kind", "raw_probability",
    "after_bias_probability", "after_temperature_probability", "ret_hat", "rv_hat",
    "model_present", "model_excluded", "exclusion_reason", "consecutive_extreme_count",
    "rolling_probability_std", "deterministic_repeat_error",
]
VARIANT_NAMES = (
    "current_config", "auc_weight_all", "equal_weight_all", "no_tcn",
    "lstm_tx_only", "tcn_only",
)
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|private[_-]?key|wallet|password|token)\s*[:=]\s*\S+"
)


class ModelAlignmentError(ValueError):
    """Returned for contract, source, policy, or deterministic evidence failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().lower()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else "unknown"
    except Exception:
        return "unknown"


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().replace("/", "").split(":", 1)[0].upper()


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    fieldnames = list(fields or [])
    if not fieldnames:
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_alignment_policy(path: Path | str = DEFAULT_POLICY) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ModelAlignmentError(f"unable to load alignment policy: {exc}") from exc
    if not isinstance(value, dict) or set(value) != set(POLICY_TYPES):
        raise ModelAlignmentError("alignment policy fields are not exact")
    for key, expected in POLICY_TYPES.items():
        item = value[key]
        if expected is float:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ModelAlignmentError(f"alignment policy field {key} must be numeric")
            value[key] = float(item)
        elif type(item) is not expected:  # exact bool/int separation matters
            raise ModelAlignmentError(f"alignment policy field {key} must be {expected.__name__}")
    if value["schema_version"] != 1:
        raise ModelAlignmentError("alignment policy schema_version must be 1")
    if value["required_timeframe"].lower() != "5m":
        raise ModelAlignmentError("alignment policy required_timeframe must be 5m")
    if value["required_sequence_length"] != 64:
        raise ModelAlignmentError("alignment policy required_sequence_length must be 64")
    if value["required_served_feature_count"] != 27:
        raise ModelAlignmentError("alignment policy required_served_feature_count must be 27")
    if value["historical_unique_bars_required"] < 100:
        raise ModelAlignmentError("historical policy may not weaken the 100-bar gate")
    if value["live_unique_bars_required"] < 3:
        raise ModelAlignmentError("live policy may not weaken the three-bar gate")
    if value["historical_capture_target"] < value["historical_unique_bars_required"]:
        raise ModelAlignmentError("historical capture target weakens the statistical gate")
    for field_name in (
        "completed_bar_grace_seconds", "maximum_gap_bars",
        "flat_output_window", "extreme_consecutive_limit",
    ):
        if value[field_name] < 0:
            raise ModelAlignmentError(f"alignment policy field {field_name} must be non-negative")
    if value["flat_output_window"] == 0 or value["extreme_consecutive_limit"] == 0:
        raise ModelAlignmentError("alignment health windows must be positive")
    if value["deterministic_repeat_tolerance"] < 0:
        raise ModelAlignmentError("deterministic repeat tolerance must be non-negative")
    if not (0 <= value["maximum_missing_rate"] <= 1):
        raise ModelAlignmentError("maximum_missing_rate must be between zero and one")
    if not (0 < value["extreme_probability_threshold_low"] < 0.5):
        raise ModelAlignmentError("invalid low extreme threshold")
    if not (0.5 < value["extreme_probability_threshold_high"] < 1):
        raise ModelAlignmentError("invalid high extreme threshold")
    return value


def calibrate_probability(raw_probability: float, bias: float, temperature: float) -> tuple[float, float]:
    """Production order: subtract bias, clip, then apply temperature."""

    raw = _finite(raw_probability)
    if raw is None:
        raise ModelAlignmentError("non-finite raw probability")
    biased = float(np.clip(raw - float(bias), 1e-6, 1.0 - 1e-6))
    calibrated = biased
    if temperature > 0 and temperature != 1.0:
        logit = math.log(biased / (1.0 - biased))
        calibrated = 1.0 / (1.0 + math.exp(-logit / float(temperature)))
    return biased, float(calibrated)


def deduplicate_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "source_bar_id",
    digest_field: str = "feature_window_digest",
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate exact source identities and reject conflicting evidence."""

    retained: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates = 0
    for raw in rows:
        row = dict(raw)
        identity = str(row.get(id_field) or "")
        digest = str(row.get(digest_field) or "")
        if not identity or not digest:
            raise ModelAlignmentError("source evidence is missing identity or digest")
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = digest
            retained.append(row)
        elif previous == digest:
            duplicates += 1
        else:
            raise ModelAlignmentError(
                f"conflicting source bar {identity}: {previous} != {digest}"
            )
    return retained, duplicates


def alignment_bundle_digest(manifest: Mapping[str, Any]) -> str:
    return _json_digest({key: manifest[key] for key in sorted(manifest) if key != "bundle_digest"})


def _normalize_ohlcv(
    frame: pd.DataFrame, *, with_duplicate_count: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, int]:
    value = frame.copy()
    if "timestamp" in value.columns:
        value["timestamp"] = pd.to_datetime(value["timestamp"], utc=True)
        value = value.set_index("timestamp")
    value.index = pd.to_datetime(value.index, utc=True)
    missing = [name for name in OHLCV_COLUMNS if name not in value.columns]
    if missing:
        raise ModelAlignmentError(f"OHLCV data missing columns {missing}")
    value = value[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        raise ModelAlignmentError("OHLCV data contains non-finite values")
    # A duplicate is safe only when every canonical value is identical.
    duplicate_count = 0
    if value.index.has_duplicates:
        for timestamp, group in value.groupby(level=0, sort=False):
            if len(group.drop_duplicates()) > 1:
                raise ModelAlignmentError(f"conflicting source bar at {canonical_utc(timestamp)}")
            duplicate_count += max(0, len(group) - 1)
        value = value[~value.index.duplicated(keep="first")]
    normalized = value.sort_index()
    return (normalized, duplicate_count) if with_duplicate_count else normalized


def _maximum_gap(index: pd.DatetimeIndex, timeframe_seconds: int) -> tuple[float, int]:
    if len(index) < 2:
        return 0.0, 0
    seconds = np.diff(index.asi8) / 1_000_000_000
    maximum = float(np.max(seconds))
    missing = max(0, int(round(maximum / timeframe_seconds)) - 1)
    return maximum, missing


def _missing_bar_count(index: pd.DatetimeIndex, timeframe_seconds: int) -> int:
    if len(index) < 2:
        return 0
    seconds = np.diff(index.asi8) / 1_000_000_000
    return int(sum(max(0, int(round(value / timeframe_seconds)) - 1) for value in seconds))


def _safe_bundle_stage(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))


def _assert_capture_contains_no_secrets(root: Path) -> None:
    for path in root.iterdir():
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".log"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if SECRET_RE.search(text):
            raise ModelAlignmentError(f"capture output contains secret-like material: {path.name}")


def _safe_snapshot_for_alignment(symbols: Sequence[str], timeframe: str, sequence_length: int) -> dict[str, Any]:
    return capture_model_serving_snapshot(
        identity="phase22_alignment",
        mode="model_alignment_shadow",
        forced_env_overrides={
            "LIVE_TRADING": "false",
            "PAPER_TRADING": "true",
            "LIVE_MODE": "false",
            "EXEC_PAPER": "true",
            "PLACE_REAL_ORDERS": "false",
            "DL_TIMEFRAME": timeframe,
            "DL_SEQ_LEN": str(sequence_length),
            "DL_SYMBOLS": ",".join(symbols),
            "DL_COMPLETED_ONLY": "true",
        },
    )


def capture_historical_bundle(
    *,
    bundle_out: Path | str,
    symbols: Sequence[str],
    timeframe: str,
    unique_bars: int,
    lookback_bars: int,
    policy: Mapping[str, Any],
    as_of_utc: Any = None,
    fetcher: Optional[Callable[..., pd.DataFrame]] = None,
    snapshot: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Capture one completed, self-contained public OHLCV alignment bundle."""

    final_target = Path(bundle_out)
    if final_target.exists():
        raise ModelAlignmentError("alignment bundle already exists and is immutable")
    tf = str(timeframe).lower()
    if tf != policy["required_timeframe"]:
        raise ModelAlignmentError(f"historical alignment timeframe must be {policy['required_timeframe']}")
    if unique_bars < int(policy["historical_unique_bars_required"]):
        raise ModelAlignmentError("requested bars are below the historical statistical gate")
    sequence_length = int(policy["required_sequence_length"])
    feature_count = int(policy["required_served_feature_count"])
    grace = int(policy["completed_bar_grace_seconds"])
    seconds = timeframe_duration_seconds(tf)
    symbols_norm = list(dict.fromkeys(
        _norm_symbol(value) for value in symbols if _norm_symbol(value)
    ))
    if not symbols_norm:
        raise ModelAlignmentError("at least one serving symbol is required")
    snapshot_value = dict(snapshot or _safe_snapshot_for_alignment(symbols_norm, tf, sequence_length))
    guard = guard_from_snapshot(snapshot_value)
    if guard["status"] != "pass":
        raise ModelAlignmentError("model-serving contract failed: " + "; ".join(guard["critical_mismatches"]))
    add_symbol_id = snapshot_value.get("dl_add_symbol_id")
    if not isinstance(add_symbol_id, bool):
        raise ModelAlignmentError("symbol-id serving setting is unverified")
    columns = canonical_feature_columns(add_symbol_id)
    if len(columns) != feature_count:
        raise ModelAlignmentError(f"generated feature width {len(columns)} != required {feature_count}")
    training_symbols = guard["training_contract"].get("ordered_symbols", [])
    if add_symbol_id and not training_symbols:
        raise ModelAlignmentError("ordered training symbols are unverified")
    symbol_ids = {_norm_symbol(value): index for index, value in enumerate(training_symbols)}
    captured_at = _utc_now()
    as_of = canonical_utc(as_of_utc or captured_at)
    if fetcher is None:
        from data import fetch_ohlcv
        fetcher = fetch_ohlcv

    target = _safe_bundle_stage(final_target)
    write_model_serving_snapshot(snapshot_value, target / "model_serving_snapshot.json")
    window_records: list[dict[str, Any]] = []
    unique_counts: dict[str, int] = {}
    dropped_counts: dict[str, int] = {}
    first_completed: dict[str, str] = {}
    last_completed: dict[str, str] = {}
    gap_seconds: dict[str, float] = {}
    missing_bar_counts: dict[str, int] = {}
    source_duplicate_counts: dict[str, int] = {}
    source_digests: dict[str, str] = {}
    feature_digests: dict[str, str] = {}

    for symbol in symbols_norm:
        raw, source_duplicates = _normalize_ohlcv(
            fetcher(symbol, timeframe=tf, limit=int(lookback_bars)),
            with_duplicate_count=True,
        )
        source_duplicate_counts[symbol] = int(source_duplicates)
        mask = completed_bar_mask(
            raw.index, tf, as_of_utc=as_of,
            completion_grace_seconds=grace,
        )
        completed = raw.loc[mask].copy()
        dropped_counts[symbol] = int(len(raw) - len(completed))
        if completed.empty:
            raise ModelAlignmentError(f"no completed bars captured for {symbol}")
        maximum_gap, missing_gap_bars = _maximum_gap(completed.index, seconds)
        gap_seconds[symbol] = maximum_gap
        missing_bar_counts[symbol] = _missing_bar_count(completed.index, seconds)
        if missing_gap_bars > int(policy["maximum_gap_bars"]):
            raise ModelAlignmentError(
                f"{symbol} has an excessive historical gap of {missing_gap_bars} bars"
            )
        source_path = target / f"source_bars_{symbol}.csv"
        source_frame = completed.reset_index(names="bar_open_utc")
        source_frame["bar_open_utc"] = source_frame["bar_open_utc"].map(canonical_utc)
        source_frame.to_csv(source_path, index=False, lineterminator="\n")
        source_digests[source_path.name] = _file_digest(source_path)

        features = build_features(completed)
        if add_symbol_id:
            if symbol not in symbol_ids:
                raise ModelAlignmentError(f"training metadata has no symbol id for {symbol}")
            features = features.copy()
            features["symbol_id"] = float(symbol_ids[symbol])
        missing_columns = [name for name in columns if name not in features.columns]
        if missing_columns:
            raise ModelAlignmentError(f"feature pipeline missing {missing_columns}")
        matrix = features[columns].astype(np.float32)
        if matrix.shape[1] != feature_count:
            raise ModelAlignmentError("historical feature width is not exactly 27")
        if not np.isfinite(matrix.to_numpy()).all():
            raise ModelAlignmentError("historical feature matrix contains non-finite values")
        feature_path = target / f"features_{symbol}.csv"
        feature_frame = matrix.reset_index(names="feature_open_utc")
        feature_frame["feature_open_utc"] = feature_frame["feature_open_utc"].map(canonical_utc)
        feature_frame.to_csv(feature_path, index=False, lineterminator="\n", float_format="%.9g")
        feature_digests[feature_path.name] = _file_digest(feature_path)

        possible = list(range(sequence_length - 1, len(matrix)))
        if len(possible) < unique_bars:
            raise ModelAlignmentError(
                f"{symbol} produced {len(possible)} unique completed endpoints; need {unique_bars}"
            )
        retained_indices = possible[-int(unique_bars):]
        symbol_records: list[dict[str, Any]] = []
        for endpoint in retained_indices:
            start = endpoint - sequence_length + 1
            window = matrix.iloc[start:endpoint + 1].to_numpy(dtype=np.float32)
            window_index = matrix.index[start:endpoint + 1]
            bar_open = matrix.index[endpoint]
            bar_close = bar_open + pd.Timedelta(seconds=seconds)
            record = {
                "symbol": symbol,
                "source_bar_id": source_bar_id(symbol, bar_close),
                "source_bar_open_utc": canonical_utc(bar_open),
                "source_bar_close_utc": canonical_utc(bar_close),
                "source_bar_completed": True,
                "feature_window_digest": feature_window_digest(
                    symbol, tf, columns, window_index, window
                ),
                "feature_window_row_count": sequence_length,
                "feature_window_first_utc": canonical_utc(window_index[0]),
                "feature_window_last_utc": canonical_utc(window_index[-1]),
            }
            symbol_records.append(record)
        symbol_records, _ = deduplicate_evidence_rows(symbol_records)
        if len(symbol_records) < unique_bars:
            raise ModelAlignmentError(f"{symbol} did not retain enough unique source bars")
        window_records.extend(symbol_records)
        unique_counts[symbol] = len(symbol_records)
        first_completed[symbol] = symbol_records[0]["source_bar_close_utc"]
        last_completed[symbol] = symbol_records[-1]["source_bar_close_utc"]

    window_records, duplicate_windows = deduplicate_evidence_rows(window_records)
    windows_path = target / "evaluation_windows.jsonl"
    windows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in window_records),
        encoding="utf-8",
    )
    (target / "capture_stdout.log").write_text(
        f"captured completed alignment bars symbols={','.join(symbols_norm)} timeframe={tf}\n",
        encoding="utf-8",
    )
    (target / "capture_stderr.log").write_text("", encoding="utf-8")
    source_files = dict(sorted(source_digests.items()))
    feature_files = dict(sorted(feature_digests.items()))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "capture_type": "immutable_historical_alignment",
        "captured_at": captured_at,
        "as_of_utc": as_of,
        "git_commit": _git_commit(),
        "symbols": symbols_norm,
        "market_data_exchange": snapshot_value.get("market_data_exchange"),
        "timeframe": tf,
        "timeframe_seconds": seconds,
        "sequence_length": sequence_length,
        "feature_count": feature_count,
        "add_symbol_id": add_symbol_id,
        "requested_unique_bars": int(unique_bars),
        "unique_completed_bars_by_symbol": unique_counts,
        "incomplete_bars_dropped_by_symbol": dropped_counts,
        "first_completed_bar_by_symbol": first_completed,
        "last_completed_bar_by_symbol": last_completed,
        "maximum_gap_seconds_by_symbol": gap_seconds,
        "missing_bar_count_by_symbol": missing_bar_counts,
        "source_bar_duplicate_count_by_symbol": source_duplicate_counts,
        "source_file_digests": source_files,
        "feature_file_digests": feature_files,
        "window_digest_count": len(window_records),
        "duplicate_source_bar_count": int(
            duplicate_windows + sum(source_duplicate_counts.values())
        ),
        "conflicting_source_bar_count": 0,
        "serving_snapshot_digest": snapshot_value["snapshot_digest"],
        "bundle_digest": None,
    }
    bundle_files = [
        "model_serving_snapshot.json", "evaluation_windows.jsonl",
        "capture_stdout.log", "capture_stderr.log",
        *source_files, *feature_files,
    ]
    manifest["bundle_file_digests"] = {
        name: _file_digest(target / name) for name in sorted(bundle_files)
    }
    manifest["bundle_digest"] = alignment_bundle_digest(manifest)
    _write_json(target / "bundle_manifest.json", manifest)
    _assert_capture_contains_no_secrets(target)
    target.replace(final_target)
    return manifest


def load_historical_bundle(bundle: Path | str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(bundle)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("bundle_digest") != alignment_bundle_digest(manifest):
        raise ModelAlignmentError("historical bundle digest mismatch")
    for name, digest in manifest.get("bundle_file_digests", {}).items():
        if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", str(digest or "")):
            raise ModelAlignmentError(f"historical bundle has an unsafe file inventory: {name}")
        if not (root / name).is_file() or _file_digest(root / name) != digest:
            raise ModelAlignmentError(f"historical bundle file digest mismatch: {name}")
    for name, digest in {
        **manifest.get("source_file_digests", {}),
        **manifest.get("feature_file_digests", {}),
    }.items():
        if _file_digest(root / name) != digest:
            raise ModelAlignmentError(f"historical bundle file digest mismatch: {name}")
    snapshot = load_model_serving_snapshot(root / "model_serving_snapshot.json")
    if snapshot.get("snapshot_digest") != manifest.get("serving_snapshot_digest"):
        raise ModelAlignmentError("historical bundle serving snapshot mismatch")
    records = [
        json.loads(line) for line in (root / "evaluation_windows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records, _ = deduplicate_evidence_rows(records)
    expected_counts = {
        str(symbol): int(count)
        for symbol, count in manifest.get("unique_completed_bars_by_symbol", {}).items()
    }
    actual_counts = {
        symbol: sum(str(row.get("symbol")) == symbol for row in records)
        for symbol in expected_counts
    }
    if actual_counts != expected_counts:
        raise ModelAlignmentError(
            f"historical bundle endpoint inventory mismatch: {actual_counts} != {expected_counts}"
        )
    as_of = pd.Timestamp(manifest["as_of_utc"])
    grace = pd.Timedelta(seconds=int(
        snapshot.get("effective_completed_bar_policy", {}).get("completion_grace_seconds", 5)
    ))
    for row in records:
        if row.get("source_bar_completed") is not True:
            raise ModelAlignmentError(f"incomplete source bar retained: {row.get('source_bar_id')}")
        if pd.Timestamp(row["source_bar_close_utc"]) + grace > as_of:
            raise ModelAlignmentError(f"future or incomplete source bar retained: {row['source_bar_id']}")
    return manifest, snapshot, records


@dataclass
class SimulatedHealthState:
    policy: Mapping[str, Any]
    seen_source_bars: set[str] = field(default_factory=set)
    consecutive_extreme_count: int = 0
    history: deque[float] = field(init=False)
    extreme_event_active: bool = False
    flat_event_active: bool = False

    def __post_init__(self) -> None:
        self.history = deque(maxlen=int(self.policy["flat_output_window"]))

    def observe(self, identity: str, probability: Optional[float]) -> dict[str, Any]:
        if identity in self.seen_source_bars:
            rolling_std = float(np.std(self.history)) if self.history else None
            active_reasons = []
            if self.extreme_event_active:
                active_reasons.append("extreme_collapse")
            if self.flat_event_active:
                active_reasons.append("flat_output")
            return {
                "advanced": False,
                "excluded": bool(active_reasons),
                "reason": "+".join(active_reasons) or "duplicate_source_bar_ignored",
                "consecutive_extreme_count": self.consecutive_extreme_count,
                "rolling_probability_std": rolling_std,
                "events": [],
            }
        self.seen_source_bars.add(identity)
        if probability is None or not math.isfinite(probability):
            return {
                "advanced": True, "excluded": True, "reason": "missing_or_nonfinite_output",
                "consecutive_extreme_count": self.consecutive_extreme_count,
                "rolling_probability_std": None, "events": ["missing_output"],
            }
        low = float(self.policy["extreme_probability_threshold_low"])
        high = float(self.policy["extreme_probability_threshold_high"])
        self.consecutive_extreme_count = (
            self.consecutive_extreme_count + 1 if probability < low or probability > high else 0
        )
        self.history.append(float(probability))
        rolling_std = (
            float(np.std(np.asarray(self.history, dtype=float)))
            if len(self.history) == self.history.maxlen else None
        )
        extreme = self.consecutive_extreme_count >= int(self.policy["extreme_consecutive_limit"])
        flat = rolling_std is not None and rolling_std < float(self.policy["flat_output_std_threshold"])
        events: list[str] = []
        if extreme and not self.extreme_event_active:
            events.append("extreme_collapse")
        if flat and not self.flat_event_active:
            events.append("flat_output")
        self.extreme_event_active = extreme
        self.flat_event_active = flat
        reasons = []
        if extreme:
            reasons.append("extreme_collapse")
        if flat:
            reasons.append("flat_output")
        return {
            "advanced": True,
            "excluded": bool(reasons),
            "reason": "+".join(reasons) if reasons else "",
            "consecutive_extreme_count": self.consecutive_extreme_count,
            "rolling_probability_std": rolling_std,
            "events": events,
        }


def _read_feature_matrices(bundle: Path, manifest: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    columns = canonical_feature_columns(bool(manifest["add_symbol_id"]))
    result: dict[str, pd.DataFrame] = {}
    for symbol in manifest["symbols"]:
        frame = pd.read_csv(bundle / f"features_{symbol}.csv")
        frame["feature_open_utc"] = pd.to_datetime(frame["feature_open_utc"], utc=True)
        frame = frame.set_index("feature_open_utc")
        result[symbol] = frame[columns].astype(np.float32)
    return result


def _window_from_record(frame: pd.DataFrame, record: Mapping[str, Any], sequence_length: int) -> np.ndarray:
    finish = pd.Timestamp(record["feature_window_last_utc"])
    eligible = frame.loc[frame.index <= finish]
    window = eligible.tail(sequence_length)
    if len(window) != sequence_length:
        raise ModelAlignmentError(f"window {record['source_bar_id']} has insufficient rows")
    if canonical_utc(window.index[0]) != record["feature_window_first_utc"]:
        raise ModelAlignmentError(f"window {record['source_bar_id']} first timestamp mismatch")
    return np.ascontiguousarray(window.to_numpy(dtype=np.float32), dtype=np.float32)


def load_snapshot_models_read_only(
    snapshot: Mapping[str, Any], *, base_dir: Path | str = BASE_DIR
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load exactly the snapshotted artifacts on CPU after digest verification."""

    from ml_dl.dl_infer import load_model

    root = Path(base_dir).resolve()
    loaded: dict[str, dict[str, Any]] = {}
    for entry in snapshot.get("model_entries", []):
        kind = str(entry.get("kind") or "").lower()
        if not kind:
            raise ModelAlignmentError("snapshot model kind is missing")
        model_path = (root / str(entry.get("model_filename") or "")).resolve()
        scaler_path = (root / str(entry.get("scaler_filename") or "")).resolve()
        try:
            model_path.relative_to(root)
            scaler_path.relative_to(root)
        except ValueError as exc:
            raise ModelAlignmentError(f"unsafe snapshotted artifact path for {kind}") from exc
        expected_model = str(entry.get("model_sha256") or "")
        expected_scaler = str(entry.get("scaler_sha256") or "")
        if not model_path.is_file() or _file_digest(model_path) != expected_model:
            raise ModelAlignmentError(f"snapshotted model artifact changed or is missing: {kind}")
        if not scaler_path.is_file() or _file_digest(scaler_path) != expected_scaler:
            raise ModelAlignmentError(f"snapshotted scaler artifact changed or is missing: {kind}")
        try:
            scaler, model, device = load_model(
                kind,
                int(entry.get("metadata_n_features") or snapshot.get("feature_count") or 0),
                str(scaler_path),
                str(model_path),
                device="cpu",
            )
        except Exception as exc:
            raise ModelAlignmentError(
                f"snapshotted model/scaler failed read-only CPU load: {kind}: {type(exc).__name__}"
            ) from exc
        loaded[kind] = {
            "scaler": scaler,
            "model": model,
            "metadata": {
                "kind": entry.get("metadata_kind"),
                "timeframe": entry.get("metadata_timeframe"),
                "seq_len": entry.get("metadata_seq_len"),
                "n_features": entry.get("metadata_n_features"),
                "symbols": entry.get("metadata_symbols"),
                "val_auc": entry.get("metadata_val_auc"),
            },
        }
        if str(device) != "cpu":
            raise ModelAlignmentError(f"diagnostic model did not load on CPU: {kind}")
    if not loaded:
        raise ModelAlignmentError("snapshot contains no loadable model artifacts")
    return loaded, "cpu"


def _predict_raw(window: np.ndarray, pack: Mapping[str, Any], device: str) -> tuple[float, float, float]:
    from ml_dl.dl_infer import predict_next
    return predict_next(window, pack["scaler"], pack["model"], device)


def _bias_temperature(snapshot: Mapping[str, Any], kind: str) -> tuple[float, float]:
    return (
        float(snapshot.get(f"dl_bias_{kind}", 0.0) or 0.0),
        float(snapshot.get(f"dl_temp_{kind}", 1.0) or 1.0),
    )


def _percentile(values: np.ndarray, q: float) -> Optional[float]:
    return None if not values.size else float(np.quantile(values, q))


def _longest_repeat(values: Sequence[float]) -> int:
    best = current = 0
    previous: Optional[float] = None
    for value in values:
        if previous is not None and value == previous:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous = value
    return best


def calculate_model_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    unique_rows: list[Mapping[str, Any]] = []
    seen: dict[str, str] = {}
    for row in rows:
        identity = str(row.get("source_bar_id") or "")
        if not identity:
            raise ModelAlignmentError("model output row is missing source_bar_id")
        digest = _json_digest(dict(row))
        if identity in seen:
            if seen[identity] != digest:
                raise ModelAlignmentError(f"conflicting model output for {identity}")
            continue
        seen[identity] = digest
        unique_rows.append(row)
    present = [
        row for row in unique_rows
        if bool(row.get("model_present"))
        and _finite(row.get("raw_probability")) is not None
        and _finite(row.get("after_temperature_probability")) is not None
    ]
    raw = np.asarray([float(row["raw_probability"]) for row in present], dtype=float)
    calibrated = np.asarray(
        [float(row["after_temperature_probability"]) for row in present], dtype=float
    )
    missing = max(0, expected_count - len(present))
    exclusion_reasons = [str(row.get("exclusion_reason") or "") for row in unique_rows]
    flat_windows = sum("flat_output" in reason for reason in exclusion_reasons)
    extreme_events = sum(
        int(row.get("consecutive_extreme_count") or 0) == int(policy["extreme_consecutive_limit"])
        for row in unique_rows
    )
    deterministic_error = max(
        (_finite(row.get("deterministic_repeat_error")) or 0.0 for row in unique_rows), default=0.0
    )
    statuses: list[str] = []
    if expected_count < int(policy["historical_unique_bars_required"]):
        statuses.append("warning_insufficient_unique_bars")
    if (missing / expected_count if expected_count else 1.0) > float(policy["maximum_missing_rate"]):
        statuses.append("failed_missing_output")
    if deterministic_error > float(policy["deterministic_repeat_tolerance"]):
        statuses.append("failed_nondeterministic")
    if flat_windows:
        statuses.append("failed_flat_at_5m")
    if extreme_events:
        statuses.append("failed_extreme_collapse_at_5m")
    if calibrated.size and float(np.std(calibrated)) >= float(policy["flat_output_std_threshold"]):
        bullish = float(np.mean(calibrated > 0.5))
        bearish = float(np.mean(calibrated < 0.5))
        if max(bullish, bearish) > 0.95:
            statuses.append("warning_one_sided")
    failure_statuses = [status for status in statuses if status.startswith("failed_")]
    warning_statuses = [status for status in statuses if status.startswith("warning_")]
    primary_status = (
        failure_statuses[0] if failure_statuses
        else warning_statuses[0] if warning_statuses
        else "healthy_aligned"
    )
    return {
        "unique_completed_bar_count": len(unique_rows),
        "present_count": len(present),
        "missing_count": missing,
        "missing_rate": missing / expected_count if expected_count else 1.0,
        "raw_probability_mean": None if not raw.size else float(np.mean(raw)),
        "raw_probability_std": None if not raw.size else float(np.std(raw)),
        "calibrated_probability_mean": None if not calibrated.size else float(np.mean(calibrated)),
        "calibrated_probability_std": None if not calibrated.size else float(np.std(calibrated)),
        "minimum": None if not calibrated.size else float(np.min(calibrated)),
        "maximum": None if not calibrated.size else float(np.max(calibrated)),
        "median": _percentile(calibrated, 0.5),
        "p05": _percentile(calibrated, 0.05),
        "p95": _percentile(calibrated, 0.95),
        "rounded_unique_count_6dp": len(set(np.round(calibrated, 6).tolist())),
        "near_neutral_rate": None if not calibrated.size else float(np.mean(np.abs(calibrated - 0.5) <= 0.01)),
        "extreme_low_rate": None if not calibrated.size else float(np.mean(calibrated < policy["extreme_probability_threshold_low"])),
        "extreme_high_rate": None if not calibrated.size else float(np.mean(calibrated > policy["extreme_probability_threshold_high"])),
        "bullish_rate": None if not calibrated.size else float(np.mean(calibrated > 0.5)),
        "bearish_rate": None if not calibrated.size else float(np.mean(calibrated < 0.5)),
        "longest_exact_repeat_run": _longest_repeat(calibrated.tolist()),
        "rolling_flat_window_count": flat_windows,
        "extreme_collapse_event_count": extreme_events,
        "simulated_exclusion_count": sum(bool(row.get("model_excluded")) for row in unique_rows),
        "simulated_survival_rate": (
            sum(not bool(row.get("model_excluded")) for row in unique_rows) / expected_count
            if expected_count else 0.0
        ),
        "deterministic_repeat_max_error": deterministic_error,
        "first_source_bar": unique_rows[0]["source_bar_id"] if unique_rows else None,
        "last_source_bar": unique_rows[-1]["source_bar_id"] if unique_rows else None,
        "1m_phase21_comparison_available": False,
        "1m_historical_std": None,
        "5m_historical_std": None if not calibrated.size else float(np.std(calibrated)),
        "collapse_resolved_at_5m": None,
        "collapse_persists_at_5m": None,
        "model_health_status": primary_status,
        "model_health_warnings": warning_statuses,
    }


def _normalize_weights(weights: Mapping[str, Any], kinds: Sequence[str]) -> dict[str, float]:
    positive = {kind: max(0.0, float(weights.get(kind, 0.0) or 0.0)) for kind in kinds}
    total = sum(positive.values())
    return (
        {kind: value / total for kind, value in positive.items()}
        if total > 0 else {kind: 1.0 / len(kinds) for kind in kinds}
    )


def _variant_weights(
    name: str, kinds: Sequence[str], snapshot: Mapping[str, Any], aucs: Mapping[str, float]
) -> dict[str, float]:
    include = list(kinds)
    if name == "no_tcn":
        include = [kind for kind in include if kind != "tcn"]
    elif name == "lstm_tx_only":
        include = [kind for kind in include if kind in {"lstm", "tx"}]
    elif name == "tcn_only":
        include = [kind for kind in include if kind == "tcn"]
    if not include:
        return {}
    if name in {"current_config", "no_tcn"}:
        base = snapshot.get("dl_model_weights", {})
    elif name == "equal_weight_all" or name in {"no_tcn", "lstm_tx_only", "tcn_only"}:
        base = {kind: 1.0 for kind in include}
    else:
        base = {kind: aucs.get(kind, 1.0) for kind in include}
    return _normalize_weights(base, include)


def evaluate_ensemble_variants(
    outputs: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outputs:
        grouped[(str(row["symbol"]), str(row["source_bar_id"]))].append(row)
    kinds = sorted({str(row["model_kind"]) for row in outputs})
    aucs = {
        str(entry["kind"]): float(entry.get("metadata_val_auc") or 1.0)
        for entry in snapshot.get("model_entries", [])
    }
    min_agree = int(snapshot.get("dl_min_agree", 2))
    threshold = float(snapshot.get("dl_p_long", 0.45))
    mode = str(snapshot.get("dl_p_long_mode", "abs"))
    allow_only = str(snapshot.get("dl_allow_only", "1"))
    rows: list[dict[str, Any]] = []
    for (symbol, identity), candidates in sorted(grouped.items()):
        available = {
            str(row["model_kind"]): row for row in candidates
            if row.get("model_present") and not row.get("model_excluded")
        }
        for name in VARIANT_NAMES:
            weights = _variant_weights(name, kinds, snapshot, aucs)
            survivors = [kind for kind in weights if kind in available and weights[kind] > 0]
            normalized = _normalize_weights(weights, survivors) if survivors else {}
            eligible = [kind for kind, weight in weights.items() if weight > 0]
            suppressed = False
            if survivors:
                probability = sum(
                    normalized[kind] * float(available[kind]["after_temperature_probability"])
                    for kind in survivors
                )
                voters = [kind for kind in survivors if normalized.get(kind, 0) > 0]
                if len(voters) >= min_agree:
                    bull = sum(float(available[kind]["after_temperature_probability"]) > 0.5 for kind in voters)
                    bear = sum(float(available[kind]["after_temperature_probability"]) < 0.5 for kind in voters)
                    if bull < min_agree and bear < min_agree:
                        probability = 0.5
                        suppressed = True
                centered = probability - 0.5
                allow = int(
                    (allow_only == "0") or ((abs(centered) if mode == "abs" else centered) >= threshold)
                )
            else:
                probability, centered, allow = 0.5, 0.0, 0
            rows.append({
                "source_bar_id": identity,
                "symbol": symbol,
                "variant": name,
                "configuration_label": (
                    "shadow_configuration_candidate_only"
                    if name in {"no_tcn", "lstm_tx_only"} else "read_only_reference"
                ),
                "model_coverage": len(survivors) / len(eligible) if eligible else 0.0,
                "models_used": ",".join(survivors),
                "probability": probability,
                "centered_probability": centered,
                "allow": allow,
                "direction": "LONG" if centered > 0 else "SHORT" if centered < 0 else "FLAT",
                "agreement_suppressed": suppressed,
            })
    current = {
        (row["symbol"], row["source_bar_id"]): row
        for row in rows if row["variant"] == "current_config"
    }
    summary: dict[str, Any] = {}
    for symbol in sorted({row["symbol"] for row in rows}):
        summary[symbol] = {}
        for name in VARIANT_NAMES:
            selected = [row for row in rows if row["symbol"] == symbol and row["variant"] == name]
            changed_allow = sum(
                row["allow"] != current[(symbol, row["source_bar_id"])]["allow"] for row in selected
            )
            changed_direction = sum(
                row["direction"] != current[(symbol, row["source_bar_id"])]["direction"] for row in selected
            )
            allowed = sum(int(row["allow"]) for row in selected)
            summary[symbol][name] = {
                "configuration_label": selected[0]["configuration_label"] if selected else None,
                "evaluated_unique_bars": len(selected),
                "model_coverage_rate": float(np.mean([row["model_coverage"] for row in selected])) if selected else 0.0,
                "allowed_count": allowed,
                "allow_rate": allowed / len(selected) if selected else 0.0,
                "long_count": sum(row["direction"] == "LONG" for row in selected),
                "short_count": sum(row["direction"] == "SHORT" for row in selected),
                "flat_count": sum(row["direction"] == "FLAT" for row in selected),
                "agreement_suppressed_count": sum(bool(row["agreement_suppressed"]) for row in selected),
                "agreement_suppressed_rate": (
                    sum(bool(row["agreement_suppressed"]) for row in selected) / len(selected)
                    if selected else 0.0
                ),
                "centered_probability_mean": float(np.mean([row["centered_probability"] for row in selected])) if selected else None,
                "centered_probability_std": float(np.std([row["centered_probability"] for row in selected])) if selected else None,
                "changed_allow_vs_current": changed_allow,
                "changed_direction_vs_current": changed_direction,
                "overlap_with_current": len(selected) - changed_direction,
            }
    return rows, summary


def evaluate_historical_bundle(
    *,
    bundle: Path | str,
    json_out: Path | str,
    policy: Mapping[str, Any],
    predictor: Optional[Callable[[np.ndarray, Mapping[str, Any], str], tuple[float, float, float]]] = None,
    models: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    root = Path(bundle)
    manifest, snapshot, records = load_historical_bundle(root)
    guard = guard_from_snapshot(snapshot)
    # Historical diagnostics remain capable of reporting a failed serving
    # contract.  The writer and live-shadow path still fail closed.
    loaded_models, device = (
        (dict(models), "cpu")
        if models is not None
        else load_snapshot_models_read_only(snapshot)
    )
    predictor_fn = predictor or _predict_raw
    matrices = _read_feature_matrices(root, manifest)
    sequence_length = int(manifest["sequence_length"])
    health: dict[tuple[str, str], SimulatedHealthState] = {}
    outputs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    ordered_records = sorted(records, key=lambda row: (row["source_bar_close_utc"], row["symbol"]))
    for record in ordered_records:
        symbol = str(record["symbol"])
        window = _window_from_record(matrices[symbol], record, sequence_length)
        columns = canonical_feature_columns(bool(manifest["add_symbol_id"]))
        recomputed = feature_window_digest(
            symbol, manifest["timeframe"], columns,
            matrices[symbol].loc[
                pd.Timestamp(record["feature_window_first_utc"]):pd.Timestamp(record["feature_window_last_utc"])
            ].tail(sequence_length).index,
            window,
        )
        if recomputed != record["feature_window_digest"]:
            raise ModelAlignmentError(f"feature-window digest mismatch: {record['source_bar_id']}")
        for kind in sorted(loaded_models):
            state = health.setdefault((kind, symbol), SimulatedHealthState(policy))
            ret_hat = rv_hat = raw_probability = None
            repeat_error: Optional[float] = None
            present = False
            failure_reason = ""
            try:
                first = predictor_fn(window, loaded_models[kind], device)
                second = predictor_fn(window.copy(), loaded_models[kind], device)
                values = [*first, *second]
                if not all(_finite(value) is not None for value in values):
                    raise ModelAlignmentError("non-finite model output")
                ret_hat, rv_hat, raw_probability = map(float, first)
                repeat_error = max(abs(float(a) - float(b)) for a, b in zip(first, second))
                present = True
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
            after_bias = after_temperature = None
            if present and raw_probability is not None:
                bias, temperature = _bias_temperature(snapshot, kind)
                try:
                    after_bias, after_temperature = calibrate_probability(
                        raw_probability, bias, temperature
                    )
                except Exception as exc:
                    present = False
                    failure_reason = f"{type(exc).__name__}: {exc}"
            health_result = state.observe(record["source_bar_id"], after_temperature if present else None)
            nondeterministic = bool(
                repeat_error is not None
                and repeat_error > float(policy["deterministic_repeat_tolerance"])
            )
            excluded = bool(health_result["excluded"] or nondeterministic or not present)
            reasons = [health_result["reason"]] if health_result["reason"] else []
            if nondeterministic:
                reasons.append("nondeterministic")
            if failure_reason:
                reasons.append(failure_reason)
            row = {
                **{key: record[key] for key in (
                    "source_bar_id", "source_bar_open_utc", "source_bar_close_utc",
                    "symbol", "feature_window_digest",
                )},
                "model_kind": kind,
                "raw_probability": raw_probability,
                "after_bias_probability": after_bias,
                "after_temperature_probability": after_temperature,
                "ret_hat": ret_hat,
                "rv_hat": rv_hat,
                "model_present": present,
                "model_excluded": excluded,
                "exclusion_reason": ";".join(reasons),
                "consecutive_extreme_count": health_result["consecutive_extreme_count"],
                "rolling_probability_std": health_result["rolling_probability_std"],
                "deterministic_repeat_error": repeat_error,
            }
            outputs.append(row)
            for event in health_result["events"]:
                events.append({
                    "source_bar_id": record["source_bar_id"], "symbol": symbol,
                    "model_kind": kind, "event": event,
                    "probability": after_temperature,
                })
            if nondeterministic:
                events.append({
                    "source_bar_id": record["source_bar_id"], "symbol": symbol,
                    "model_kind": kind, "event": "nondeterministic_output",
                    "probability": after_temperature,
                })

    stats: dict[str, dict[str, Any]] = {}
    for symbol in manifest["symbols"]:
        stats[symbol] = {}
        expected = int(manifest["unique_completed_bars_by_symbol"][symbol])
        for kind in sorted(loaded_models):
            selected = [
                row for row in outputs if row["symbol"] == symbol and row["model_kind"] == kind
            ]
            stats[symbol][kind] = calculate_model_statistics(
                selected, expected_count=expected, policy=policy
            )
            entry = next(
                (item for item in snapshot.get("model_entries", []) if item.get("kind") == kind), {}
            )
            auc = _finite(entry.get("metadata_val_auc"))
            if auc is not None and auc < 0.55:
                stats[symbol][kind]["model_health_warnings"] = sorted(set(
                    stats[symbol][kind].get("model_health_warnings", []) + ["warning_low_auc"]
                ))
                if stats[symbol][kind]["model_health_status"] == "healthy_aligned":
                    stats[symbol][kind]["model_health_status"] = "warning_low_auc"
            if guard["status"] != "pass":
                stats[symbol][kind]["model_health_status"] = "failed_contract"
    variant_rows, variant_summary = evaluate_ensemble_variants(outputs, snapshot)
    out_path = Path(json_out)
    artifact_dir = out_path.with_suffix("")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifact_dir / "model_outputs.csv", outputs, MODEL_OUTPUT_COLUMNS)
    _write_csv(artifact_dir / "model_exclusion_events.csv", events)
    _write_csv(artifact_dir / "ensemble_variants.csv", variant_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "research_only": True,
        "profitability_evidence": False,
        "bundle_digest": manifest["bundle_digest"],
        "serving_snapshot_digest": snapshot["snapshot_digest"],
        "training_serving_contract": guard,
        "unique_completed_bars_by_symbol": manifest["unique_completed_bars_by_symbol"],
        "minimum_statistical_gate_passed": all(
            int(stats[symbol][kind]["unique_completed_bar_count"])
            >= int(policy["historical_unique_bars_required"])
            for symbol in manifest["symbols"] for kind in sorted(loaded_models)
        ),
        "model_results": stats,
        "ensemble_variants": variant_summary,
        "model_output_count": len(outputs),
        "exclusion_event_count": len(events),
        "artifacts": {
            "model_outputs": str(artifact_dir / "model_outputs.csv"),
            "model_exclusion_events": str(artifact_dir / "model_exclusion_events.csv"),
            "ensemble_variants": str(artifact_dir / "ensemble_variants.csv"),
        },
    }
    result["evaluation_digest"] = _json_digest({key: result[key] for key in result if key != "generated_at"})
    _write_json(artifact_dir / "evaluation_manifest.json", result)
    _write_json(out_path, result)
    return result


def _artifact_contract(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_digest": snapshot.get("snapshot_digest"),
        "symbols": snapshot.get("dl_symbols"),
        "timeframe": snapshot.get("dl_timeframe"),
        "sequence_length": snapshot.get("dl_seq_len"),
        "feature_count": snapshot.get("feature_count"),
        "add_symbol_id": snapshot.get("dl_add_symbol_id"),
        "artifact_digests": {
            entry["kind"]: {
                "model": entry.get("model_sha256"), "scaler": entry.get("scaler_sha256")
            } for entry in snapshot.get("model_entries", [])
        },
        "calibration": {
            key: snapshot.get(key) for key in snapshot
            if key.startswith("dl_bias_") or key.startswith("dl_temp_") or key == "dl_model_weights"
        },
        "decision_settings": {
            key: snapshot.get(key) for key in (
                "dl_min_agree", "dl_p_long", "dl_p_long_mode", "dl_allow_only"
            )
        },
    }


def validate_resume_contract(manifest: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    if manifest.get("campaign_contract") != _artifact_contract(snapshot):
        raise ModelAlignmentError("live-shadow resume contract changed")
    started_at = str(manifest.get("started_at") or "")
    expected_id = hashlib.sha256(
        (
            f"{started_at}|{snapshot.get('dl_timeframe')}|{snapshot.get('snapshot_digest')}|"
            f"{','.join(snapshot.get('dl_symbols', []))}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    if manifest.get("campaign_id") != expected_id:
        raise ModelAlignmentError("live-shadow campaign ID is not deterministic")


def _append_campaign_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(message.rstrip("\r\n") + "\n")


def _campaign_evidence_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]
    return _json_digest(canonical)


def live_shadow_campaign(
    *,
    symbols: Sequence[str],
    unique_bars: int,
    output_root: Path | str,
    policy: Mapping[str, Any],
    poll_seconds: float = 5.0,
    campaign_dir: Path | str | None = None,
    dry_run: bool = False,
    refresh_fn: Callable[..., Any] = refresh_live_features_per_symbol,
    predictor: Optional[Callable[[np.ndarray, Mapping[str, Any], str], tuple[float, float, float]]] = None,
    max_polls: Optional[int] = None,
    fresh_logs: bool = False,
) -> dict[str, Any]:
    symbols_norm = sorted({_norm_symbol(value) for value in symbols if _norm_symbol(value)})
    required = int(policy["live_unique_bars_required"])
    if unique_bars < required or unique_bars > 120:
        raise ModelAlignmentError(f"live unique bars must be between {required} and 120")
    timeframe = str(policy["required_timeframe"])
    sequence_length = int(policy["required_sequence_length"])
    snapshot = _safe_snapshot_for_alignment(symbols_norm, timeframe, sequence_length)
    guard = guard_from_snapshot(snapshot)
    if guard["status"] != "pass":
        raise ModelAlignmentError("live-shadow model contract failed")
    plan = {
        "dry_run": bool(dry_run), "symbols": symbols_norm, "timeframe": timeframe,
        "sequence_length": sequence_length, "unique_bars": unique_bars,
        "snapshot_digest": snapshot["snapshot_digest"], "contract_status": guard["status"],
        "writer_started": False, "executor_started": False, "orders_allowed": False,
        "fresh_logs": bool(fresh_logs),
    }
    if dry_run:
        return plan
    start = _utc_now()
    campaign_id = hashlib.sha256(
        f"{start}|{timeframe}|{snapshot['snapshot_digest']}|{','.join(symbols_norm)}".encode("utf-8")
    ).hexdigest()[:20]
    target = Path(campaign_dir) if campaign_dir else Path(output_root) / campaign_id
    manifest_path = target / "campaign_manifest.json"
    if fresh_logs and campaign_dir and manifest_path.exists():
        raise ModelAlignmentError("FreshLogs requires a new campaign directory")
    existing_rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        validate_resume_contract(manifest, snapshot)
        if int(manifest.get("requested_unique_bars", -1)) != int(unique_bars):
            raise ModelAlignmentError("live-shadow resume requested bar count changed")
        start = manifest["started_at"]
        campaign_id = manifest["campaign_id"]
        output_path = target / "completed_bar_outputs.csv"
        if output_path.is_file():
            existing_rows = list(csv.DictReader(output_path.open("r", encoding="utf-8-sig", newline="")))
        expected_evidence_digest = manifest.get("evidence_digest")
        if expected_evidence_digest and expected_evidence_digest != _campaign_evidence_digest(existing_rows):
            raise ModelAlignmentError("live-shadow resume evidence digest changed")
    else:
        target.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1, "campaign_id": campaign_id, "started_at": start,
            "symbols": symbols_norm, "timeframe": timeframe, "sequence_length": sequence_length,
            "requested_unique_bars": unique_bars, "baseline_source_bar_id_by_symbol": {},
            "campaign_contract": _artifact_contract(snapshot), "writer_started": False,
            "executor_started": False, "orders_allowed": False,
            "fresh_logs": bool(fresh_logs), "evidence_digest": _campaign_evidence_digest([]),
        }
        _write_json(manifest_path, manifest)
        write_model_serving_snapshot(snapshot, target / "model_serving_snapshot.json")
        _write_csv(
            target / "completed_bar_outputs.csv", [],
            [
                "source_bar_id", "source_bar_open_utc", "source_bar_close_utc",
        "source_bar_completed", "symbol", "feature_window_digest",
                "model_outputs_json",
            ],
        )
        _write_csv(
            target / "model_exclusion_events.csv", [],
            ["source_bar_id", "symbol", "model_kind", "event"],
        )
        (target / "stdout.log").write_text("", encoding="utf-8")
        (target / "stderr.log").write_text("", encoding="utf-8")
    models, device = load_snapshot_models_read_only(snapshot)
    predictor_fn = predictor or _predict_raw
    evidence, _ = deduplicate_evidence_rows(existing_rows) if existing_rows else ([], 0)
    digest_by_identity = {
        str(row["source_bar_id"]): str(row["feature_window_digest"]) for row in evidence
    }
    counted: dict[str, set[str]] = {symbol: set() for symbol in symbols_norm}
    for row in evidence:
        counted[str(row["symbol"])].add(str(row["source_bar_id"]))
    baseline = dict(manifest.get("baseline_source_bar_id_by_symbol", {}))
    baseline_digests = dict(manifest.get("baseline_window_digest_by_symbol", {}))
    health: dict[tuple[str, str], SimulatedHealthState] = {}
    event_path = target / "model_exclusion_events.csv"
    events = (
        list(csv.DictReader(event_path.open("r", encoding="utf-8-sig", newline="")))
        if event_path.is_file() and event_path.stat().st_size else []
    )
    for row in evidence:
        identity = str(row["source_bar_id"])
        symbol = str(row["symbol"])
        try:
            values = json.loads(str(row.get("model_outputs_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise ModelAlignmentError("live-shadow resume output JSON is malformed") from exc
        for kind, output in values.items():
            probability = _finite(output.get("after_temperature_probability"))
            health.setdefault((str(kind), symbol), SimulatedHealthState(policy)).observe(
                identity, probability
            )
    polls = repeated_polls = 0
    while not all(len(counted[symbol]) >= unique_bars for symbol in symbols_norm):
        if max_polls is not None and polls >= max_polls:
            break
        polls += 1
        meta, windows = refresh_fn(
            seq_len=sequence_length, add_symbol_id=bool(snapshot["dl_add_symbol_id"]),
            symbols=symbols_norm, timeframe=timeframe, completed_only=True,
            completion_grace_seconds=int(policy["completed_bar_grace_seconds"]),
        )
        changed = False
        for symbol in symbols_norm:
            identity = meta["source_bar_id_by_symbol"].get(symbol)
            digest = meta["feature_window_digest_by_symbol"].get(symbol)
            if not identity or not digest or not meta["source_bar_completed_by_symbol"].get(symbol):
                raise ModelAlignmentError(f"live shadow received incomplete evidence for {symbol}")
            previous_digest = digest_by_identity.get(identity)
            if previous_digest is not None and previous_digest != digest:
                raise ModelAlignmentError(f"conflicting live source bar {identity}")
            if symbol not in baseline:
                baseline[symbol] = identity
                baseline_digests[symbol] = digest
                manifest["baseline_source_bar_id_by_symbol"] = baseline
                manifest["baseline_window_digest_by_symbol"] = baseline_digests
                _write_json(manifest_path, manifest)
                _append_campaign_log(target / "stdout.log", f"baseline {identity} digest={digest}")
                continue
            if identity == baseline[symbol]:
                if baseline_digests.get(symbol) != digest:
                    raise ModelAlignmentError(f"conflicting baseline source bar {identity}")
                repeated_polls += 1
                continue
            if identity in counted[symbol]:
                repeated_polls += 1
                continue
            if len(counted[symbol]) >= unique_bars:
                continue
            model_values: dict[str, Any] = {}
            for kind in sorted(models):
                first = predictor_fn(windows[symbol], models[kind], device)
                second = predictor_fn(windows[symbol].copy(), models[kind], device)
                if not all(_finite(value) is not None for value in (*first, *second)):
                    raise ModelAlignmentError(f"non-finite live-shadow model output: {kind}/{symbol}")
                repeat_error = max(abs(float(a) - float(b)) for a, b in zip(first, second))
                if repeat_error > float(policy["deterministic_repeat_tolerance"]):
                    raise ModelAlignmentError(f"non-deterministic live-shadow output: {kind}/{symbol}")
                biased, calibrated = calibrate_probability(
                    float(first[2]), *_bias_temperature(snapshot, kind)
                )
                state = health.setdefault((kind, symbol), SimulatedHealthState(policy))
                health_row = state.observe(identity, calibrated)
                model_values[kind] = {
                    "ret_hat": float(first[0]), "rv_hat": float(first[1]),
                    "raw_probability": float(first[2]), "after_bias_probability": biased,
                    "after_temperature_probability": calibrated,
                    "deterministic_repeat_error": repeat_error,
                    "model_excluded": health_row["excluded"],
                }
                for event in health_row["events"]:
                    events.append({"source_bar_id": identity, "symbol": symbol, "model_kind": kind, "event": event})
            evidence.append({
                "source_bar_id": identity,
                "source_bar_open_utc": meta["source_bar_open_utc_by_symbol"][symbol],
                "source_bar_close_utc": meta["source_bar_close_utc_by_symbol"][symbol],
                "source_bar_completed": True,
                "symbol": symbol,
                "feature_window_digest": digest,
                "model_outputs_json": json.dumps(model_values, sort_keys=True, separators=(",", ":")),
            })
            digest_by_identity[identity] = digest
            counted[symbol].add(identity)
            _append_campaign_log(target / "stdout.log", f"accepted {identity} digest={digest}")
            changed = True
        if changed:
            evidence, _ = deduplicate_evidence_rows(evidence)
            _write_csv(target / "completed_bar_outputs.csv", evidence)
            _write_csv(target / "model_exclusion_events.csv", events)
            manifest["evidence_digest"] = _campaign_evidence_digest(evidence)
            _write_json(manifest_path, manifest)
        if not all(len(counted[symbol]) >= unique_bars for symbol in symbols_norm):
            time.sleep(max(0.0, float(poll_seconds)))
    passed = all(len(counted[symbol]) >= unique_bars for symbol in symbols_norm)
    final = {
        "schema_version": 1, "campaign_id": campaign_id, "completed_at": _utc_now(),
        "status": "pass" if passed else "collection_pending",
        "unique_completed_bars_by_symbol": {symbol: len(counted[symbol]) for symbol in symbols_norm},
        "requested_unique_bars": unique_bars, "poll_count": polls,
        "repeated_poll_count": repeated_polls, "contract_status": guard["status"],
        "timeframe": timeframe, "sequence_length": sequence_length,
        "served_feature_width": int(policy["required_served_feature_count"]),
        "snapshot_digest": snapshot["snapshot_digest"],
        "contract_guard_digest": guard["guard_digest"],
        "all_source_bars_completed": all(
            row.get("source_bar_completed") is True
            or str(row.get("source_bar_completed", "")).strip().lower() == "true"
            for row in evidence
        ),
        "all_source_bar_ids_unique": len(digest_by_identity) == len(evidence),
        "conflicting_source_bar_count": 0, "nonfinite_output_count": 0,
        "nondeterministic_output_count": 0, "writer_started": False,
        "executor_started": False, "orders_placed": 0,
    }
    final["campaign_digest"] = _json_digest({key: final[key] for key in final if key != "completed_at"})
    _write_json(target / "final_report.json", final)
    manifest["status"] = final["status"]
    manifest["final_report_digest"] = final["campaign_digest"]
    _write_json(manifest_path, manifest)
    _append_campaign_log(
        target / "stdout.log",
        f"status={final['status']} unique_completed_bars_by_symbol="
        f"{json.dumps(final['unique_completed_bars_by_symbol'], sort_keys=True)}",
    )
    return final


def _symbols(raw: str) -> list[str]:
    return [_norm_symbol(value) for value in str(raw).split(",") if _norm_symbol(value)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only Phase 22 training-serving alignment shadow validation."
    )
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-history", help="Capture immutable completed historical bars.")
    capture.add_argument("--timeframe", default="5m")
    capture.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    capture.add_argument("--unique-bars", type=int, default=120)
    capture.add_argument("--lookback-bars", type=int, default=800)
    capture.add_argument("--bundle-out", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate an existing bundle without network reads.")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--json-out", required=True)
    live = sub.add_parser("live-shadow", help="Observe new completed bars without a writer or executor.")
    live.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    live.add_argument("--unique-bars", type=int, default=3)
    live.add_argument("--output-root", default=str(BASE_DIR / "reports" / "model_alignment_live"))
    live.add_argument("--campaign-dir")
    live.add_argument("--poll-seconds", type=float, default=5.0)
    live.add_argument("--dry-run", action="store_true")
    live.add_argument("--fresh-logs", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        policy = load_alignment_policy(args.policy)
        if args.command == "capture-history":
            result = capture_historical_bundle(
                bundle_out=args.bundle_out, symbols=_symbols(args.symbols),
                timeframe=args.timeframe, unique_bars=args.unique_bars,
                lookback_bars=args.lookback_bars, policy=policy,
            )
        elif args.command == "evaluate":
            result = evaluate_historical_bundle(
                bundle=args.bundle, json_out=args.json_out, policy=policy,
            )
        else:
            result = live_shadow_campaign(
                symbols=_symbols(args.symbols), unique_bars=args.unique_bars,
                output_root=args.output_root, campaign_dir=args.campaign_dir,
                poll_seconds=args.poll_seconds, policy=policy, dry_run=args.dry_run,
                fresh_logs=args.fresh_logs,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ModelAlignmentError, OSError, ValueError) as exc:
        print(f"model_alignment_error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
