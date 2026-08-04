"""Offline deterministic health audit for the deployed DL ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DEFAULT_POLICY = BASE_DIR / "research" / "model_health_policy.json"
DEFAULT_REPORT = BASE_DIR / "reports" / "model_health_audit.json"
SCHEMA_VERSION = 1
MODEL_KINDS = ("lstm", "tcn", "tx", "adv")

try:
    from tools.evidence_manifest import build_evidence_manifest
    from tools.model_health_probe import load_policy, run_artifact_probes
    from tools.model_serving_snapshot import (
        ModelServingSnapshotError,
        capture_model_serving_snapshot,
        resolve_model_serving_snapshot,
    )
    from tools.replay_bundle import (
        ReplayBundleError,
        collect_model_output_rows,
        parse_timestamp,
        validate_replay_bundle,
    )
    from tools.replay_contract import resolve_replay_contract
except ModuleNotFoundError:
    from evidence_manifest import build_evidence_manifest  # type: ignore
    from model_health_probe import load_policy, run_artifact_probes  # type: ignore
    from model_serving_snapshot import (  # type: ignore
        ModelServingSnapshotError,
        capture_model_serving_snapshot,
        resolve_model_serving_snapshot,
    )
    from replay_bundle import (  # type: ignore
        ReplayBundleError,
        collect_model_output_rows,
        parse_timestamp,
        validate_replay_bundle,
    )
    from replay_contract import resolve_replay_contract  # type: ignore


class ModelHealthAuditError(ValueError):
    """Raised when model-output evidence is contradictory or policy is malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(str(value).strip())
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: np.ndarray, quantile: float) -> Optional[float]:
    return None if values.size == 0 else float(np.quantile(values, quantile, method="linear"))


def _longest_exact_run(values: Sequence[float]) -> int:
    best = current = 0
    sentinel: Optional[float] = None
    for value in values:
        if sentinel is not None and value == sentinel:
            current += 1
        else:
            sentinel = value
            current = 1
        best = max(best, current)
    return best


def probability_statistics(
    values: Sequence[Optional[float]], timestamps: Sequence[Optional[datetime]]
) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    array = np.asarray(present, dtype=np.float64)
    row_count = len(values)
    present_count = len(present)
    valid_times = sorted(value for value in timestamps if value is not None)
    lag1 = None
    if array.size >= 2 and float(np.std(array[:-1])) > 0 and float(np.std(array[1:])) > 0:
        lag1 = float(np.corrcoef(array[:-1], array[1:])[0, 1])
    rounded = np.unique(np.round(array, 6)) if array.size else np.asarray([])
    return {
        "row_count": row_count,
        "present_count": present_count,
        "missing_count": row_count - present_count,
        "missing_rate": None if row_count == 0 else (row_count - present_count) / row_count,
        "mean": None if not array.size else float(np.mean(array)),
        "standard_deviation": None if not array.size else float(np.std(array)),
        "minimum": None if not array.size else float(np.min(array)),
        "maximum": None if not array.size else float(np.max(array)),
        "p01": _percentile(array, 0.01),
        "p05": _percentile(array, 0.05),
        "p25": _percentile(array, 0.25),
        "median": _percentile(array, 0.50),
        "p75": _percentile(array, 0.75),
        "p95": _percentile(array, 0.95),
        "p99": _percentile(array, 0.99),
        "rounded_unique_count_6dp": int(rounded.size),
        "rounded_unique_ratio": None if not array.size else float(rounded.size / array.size),
        "near_neutral_rate": None if not array.size else float(np.mean((array >= 0.49) & (array <= 0.51))),
        "extreme_low_rate": None if not array.size else float(np.mean(array < 0.05)),
        "extreme_high_rate": None if not array.size else float(np.mean(array > 0.95)),
        "bullish_rate": None if not array.size else float(np.mean(array > 0.5)),
        "bearish_rate": None if not array.size else float(np.mean(array < 0.5)),
        "exact_repeat_longest_run": _longest_exact_run(present),
        "lag1_autocorrelation": lag1,
        "first_timestamp": None if not valid_times else _iso(valid_times[0]),
        "last_timestamp": None if not valid_times else _iso(valid_times[-1]),
    }


def analyze_historical_probabilities(
    rows_by_file: Mapping[str, Sequence[Mapping[str, Any]]],
    model_kinds: Sequence[str],
    policy: Mapping[str, Any],
    *,
    symbol_filter: Optional[str] = None,
) -> dict[str, Any]:
    by_symbol: dict[str, list[Mapping[str, Any]]] = {}
    for filename, rows in rows_by_file.items():
        for row in rows:
            symbol = str(row.get("symbol", "__pooled__") or "__pooled__")
            if symbol_filter and symbol != symbol_filter:
                continue
            by_symbol.setdefault(symbol, []).append(row)
    per_model: dict[str, Any] = {}
    for kind in model_kinds:
        symbol_stats: dict[str, Any] = {}
        actual_values: list[float] = []
        actual_symbol_arrays: list[np.ndarray] = []
        for symbol, rows in sorted(by_symbol.items()):
            values = [_finite_float(row.get(f"{kind}_p")) for row in rows]
            timestamps = [parse_timestamp(row.get("ts", row.get("timestamp", ""))) for row in rows]
            stats = probability_statistics(values, timestamps)
            symbol_stats[symbol] = stats
            if symbol != "__pooled__" or len(by_symbol) == 1:
                array = np.asarray([value for value in values if value is not None], dtype=float)
                actual_values.extend(array.tolist())
                if array.size:
                    actual_symbol_arrays.append(array)
        pooled = np.asarray(actual_values, dtype=float)
        symbol_means = np.asarray([np.mean(values) for values in actual_symbol_arrays], dtype=float)
        within_variances = [float(np.var(values)) for values in actual_symbol_arrays]
        row_count = sum(stats["row_count"] for symbol, stats in symbol_stats.items()
                        if symbol != "__pooled__" or len(by_symbol) == 1)
        present_count = len(actual_values)
        missing_rate = None if row_count == 0 else (row_count - present_count) / row_count
        statuses: list[str] = []
        if present_count < int(policy["minimum_rows_for_warning"]):
            statuses.append("unverified")
        elif present_count < int(policy["minimum_rows_for_decision"]):
            statuses.append("warning_insufficient_rows")
        elif float(np.std(pooled)) < float(policy["flat_output_std_threshold"]):
            statuses.append("failed_flat_output")
        if (row_count >= int(policy["minimum_rows_for_decision"])
                and missing_rate is not None
                and missing_rate > float(policy["maximum_missing_rate"])):
            statuses.append("failed_missing_output")
        if pooled.size and float(np.std(pooled)) >= float(policy["flat_output_std_threshold"]):
            if bool(np.all(pooled > 0.5) or np.all(pooled < 0.5)):
                statuses.append("warning_one_sided")
        per_model[kind] = {
            "by_symbol": symbol_stats,
            "pooled_standard_deviation": None if not pooled.size else float(np.std(pooled)),
            "between_symbol_standard_deviation": None if not symbol_means.size else float(np.std(symbol_means)),
            "within_symbol_standard_deviation": None if not within_variances else float(math.sqrt(np.mean(within_variances))),
            "model_missing_rate": missing_rate,
            "model_survival_rate": None if missing_rate is None else 1.0 - missing_rate,
            "present_count": present_count,
            "statuses": sorted(set(statuses)),
        }
    return {"models": per_model, "symbols": sorted(by_symbol), "source_row_count": sum(map(len, rows_by_file.values()))}


AUTO_EXCLUDE_RE = re.compile(r"auto-exclude\s+(?P<model>[a-z0-9_]+)\[(?P<symbol>[^\]]+)\]:\s*(?P<reason>.*)", re.I)
LOG_TIMESTAMP_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2}[^\]]+)\]")
DIAGNOSTIC_PATTERNS = {
    "flat_output": re.compile(r"flat output", re.I),
    "collapsed": re.compile(r"collapsed", re.I),
    "predict_failed": re.compile(r"predict failed", re.I),
    "feature_dim_mismatch": re.compile(r"feature dim mismatch", re.I),
    "ood": re.compile(r"\bOOD\b|off-distribution", re.I),
    "scaler": re.compile(r"scaler", re.I),
    "model_load_failure": re.compile(r"(?:failed to load|load_ensemble_failed|FATAL load_ensemble)", re.I),
}


def parse_writer_diagnostics(
    paths: Iterable[Path], start: Optional[datetime] = None, finish: Optional[datetime] = None
) -> dict[str, Any]:
    auto: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    model_reason_counts: dict[str, Counter[str]] = {}
    diagnostics: Counter[str] = Counter()
    observations: dict[str, Any] = {}
    files: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        files.append(path.name)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stamp_match = LOG_TIMESTAMP_RE.search(line)
            stamp = parse_timestamp(stamp_match.group("ts")) if stamp_match else None
            # Matrix mode-specific stderr is itself run evidence and frequently
            # uses host-local naive timestamps.  Do not reinterpret those as UTC;
            # only apply UTC filtering to shared live logs.
            shared_log = path.name in {"live_writer.err", "live_writer.out"}
            if shared_log and stamp is not None and ((start and stamp < start) or (finish and stamp > finish)):
                continue
            match = AUTO_EXCLUDE_RE.search(line)
            if match:
                model = match.group("model").lower()
                symbol = match.group("symbol")
                reason = match.group("reason")
                label = "flat_output" if "flat output" in reason.lower() else (
                    "collapsed_extreme" if "collapsed" in reason.lower() else "other"
                )
                auto[f"{model}[{symbol}]"] += 1
                reason_counts[label] += 1
                model_reason_counts.setdefault(model, Counter())[label] += 1
            for label, pattern in DIAGNOSTIC_PATTERNS.items():
                if pattern.search(line):
                    diagnostics[label] += 1
            started = re.search(r"writer started symbols=(?P<symbols>.*?)\s+tf=(?P<tf>\S+)", line)
            if started:
                observations["dl_timeframe"] = started.group("tf")
                observations["dl_symbols_text"] = started.group("symbols")
            dims = re.search(r"all models expect (?P<width>\d+) features .*FEATURE_COLS=(?P<base>\d+), add_symbol_id=(?P<sid>True|False)", line)
            if dims:
                observations.update({"scaler_width": int(dims.group("width")),
                                     "feature_count": int(dims.group("base")),
                                     "dl_add_symbol_id": dims.group("sid") == "True"})
    return {
        "auto_exclusion_count": int(sum(auto.values())),
        "auto_exclusions": dict(sorted(auto.items())),
        "auto_exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "auto_exclusion_reason_counts_by_model": {
            model: dict(sorted(counts.items())) for model, counts in sorted(model_reason_counts.items())
        },
        "diagnostic_reason_counts": dict(sorted(diagnostics.items())),
        "runtime_observations": observations,
        "files": sorted(set(files)),
    }


def audit_training_serving_contract(
    snapshot: Mapping[str, Any], *, runtime_observations: Optional[Mapping[str, Any]] = None,
    exact_snapshot: bool = True, low_auc_threshold: float = 0.55,
) -> dict[str, Any]:
    observed = dict(runtime_observations or {})
    effective_timeframe = observed.get("dl_timeframe", snapshot.get("dl_timeframe"))
    generated_width = None
    if snapshot.get("dl_add_symbol_id") is not None:
        generated_width = int(snapshot.get("feature_count", 0)) + int(bool(snapshot["dl_add_symbol_id"]))
    models: dict[str, Any] = {}
    critical_all: list[str] = []
    warnings_all: list[str] = []

    def compare(training: Any, serving: Any) -> str:
        if training is None:
            return "missing_metadata"
        if serving is None:
            return "unverified_runtime_value"
        return "match" if training == serving else "mismatch"

    for entry in snapshot.get("model_entries", []):
        kind = str(entry.get("kind"))
        comparisons = {
            "kind": compare(entry.get("metadata_kind"), kind),
            "timeframe": compare(entry.get("metadata_timeframe"), effective_timeframe),
            "seq_len": compare(entry.get("metadata_seq_len"), snapshot.get("dl_seq_len")),
            "metadata_feature_count_vs_scaler_width": compare(entry.get("metadata_n_features"), entry.get("scaler_n_features_in")),
            "serving_feature_count_vs_scaler_width": (
                "unverified_runtime_value" if entry.get("scaler_n_features_in") is None or generated_width is None
                else ("match" if int(entry["scaler_n_features_in"]) == int(generated_width) else "mismatch")
            ),
            "symbols": "missing_metadata" if entry.get("metadata_symbols") is None else (
                "match" if set(snapshot.get("dl_symbols", [])) == set(entry.get("metadata_symbols") or [])
                else "mismatch"
            ),
        }
        critical: list[str] = []
        warnings: list[str] = []
        if comparisons["timeframe"] == "mismatch":
            critical.append("training_serving_timeframe_mismatch")
        if comparisons["seq_len"] == "mismatch":
            critical.append("training_serving_sequence_length_mismatch")
        if comparisons["metadata_feature_count_vs_scaler_width"] == "mismatch":
            critical.append("metadata_scaler_feature_count_mismatch")
        if comparisons["serving_feature_count_vs_scaler_width"] == "mismatch":
            critical.append("serving_scaler_feature_count_mismatch")
        if entry.get("model_load_status") != "loaded":
            critical.append("model_artifact_load_failure")
        if entry.get("scaler_mean_finite") is not True or entry.get("scaler_scale_finite") is not True:
            critical.append("scaler_nonfinite_or_unverified")
        if comparisons["symbols"] == "mismatch":
            training_symbols = set(entry.get("metadata_symbols") or [])
            serving_symbols = set(snapshot.get("dl_symbols") or [])
            warnings.append("serving_symbols_subset_of_training" if serving_symbols <= training_symbols
                            else "serving_symbols_outside_training_universe")
        if entry.get("scaler_feature_names") is None and comparisons["serving_feature_count_vs_scaler_width"] == "match":
            warnings.append("scaler_feature_names_missing")
        if Path(str(entry.get("scaler_filename") or "")).name == "scaler_latest.joblib":
            warnings.append("shared_fallback_scaler")
        auc = _finite_float(entry.get("metadata_val_auc"))
        if auc is not None and auc < low_auc_threshold:
            warnings.append("low_validation_auc")
        if not exact_snapshot:
            warnings.append("runtime_values_from_current_local_environment")
        models[kind] = {"comparisons": comparisons, "critical_mismatches": critical,
                        "warnings": warnings, "result": "mismatch" if critical else "match"}
        critical_all.extend(f"{kind}:{reason}" for reason in critical)
        warnings_all.extend(f"{kind}:{reason}" for reason in warnings)
    return {
        "effective_serving": {
            "timeframe": effective_timeframe,
            "seq_len": snapshot.get("dl_seq_len"),
            "feature_count": snapshot.get("feature_count"),
            "generated_feature_count": generated_width,
            "symbols": snapshot.get("dl_symbols"),
            "add_symbol_id": snapshot.get("dl_add_symbol_id"),
            "runtime_value_source": "exact_model_serving_snapshot" if exact_snapshot else "current_local_environment_unverified",
        },
        "models": models,
        "critical_mismatches": critical_all,
        "warnings": warnings_all,
        "passed": not critical_all,
    }


def _weight_normalize(weights: Mapping[str, float], kinds: Sequence[str]) -> dict[str, float]:
    positive = {kind: max(0.0, float(weights.get(kind, 0.0))) for kind in kinds}
    total = sum(positive.values())
    if total <= 0 and kinds:
        return {kind: 1.0 / len(kinds) for kind in kinds}
    return {kind: value / total for kind, value in positive.items()}


def _variant_row(
    row: Mapping[str, Any], kinds: Sequence[str], weights: Mapping[str, float],
    min_agree: int, threshold: float, mode: str,
) -> dict[str, Any]:
    values = {kind: _finite_float(row.get(f"{kind}_p")) for kind in kinds}
    values = {kind: value for kind, value in values.items() if value is not None}
    if not values:
        return {"evaluated": False, "allow": False, "direction": "FLAT", "centered": 0.0, "suppressed": False}
    normalized = _weight_normalize(weights, list(values))
    probability = sum(values[kind] * normalized[kind] for kind in values)
    voters = [kind for kind in values if normalized.get(kind, 0) > 0] or list(values)
    suppressed = False
    if len(voters) >= min_agree:
        bulls = sum(values[kind] > 0.5 for kind in voters)
        bears = sum(values[kind] < 0.5 for kind in voters)
        if bulls < min_agree and bears < min_agree:
            probability = 0.5
            suppressed = True
    centered = probability - 0.5
    gate_value = abs(centered) if mode == "abs" else centered
    allowed = gate_value >= threshold
    direction = "LONG" if centered > 0 else ("SHORT" if centered < 0 else "FLAT")
    return {"evaluated": True, "allow": bool(allowed), "direction": direction,
            "centered": float(centered), "suppressed": suppressed}


def analyze_ensemble_variants(
    rows: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any],
    *, threshold: float, mode: str = "abs",
) -> dict[str, Any]:
    entries = {str(entry["kind"]): entry for entry in snapshot.get("model_entries", [])}
    available = [kind for kind in MODEL_KINDS if kind in entries]
    current_weights = {str(k): float(v) for k, v in snapshot.get("dl_model_weights", {}).items()}
    auc_weights = {kind: float(entries[kind].get("metadata_val_auc") or 1.0) for kind in available}
    specs = {
        "current_config": (available, current_weights),
        "equal_weight_all": (available, {kind: 1.0 for kind in available}),
        "auc_weight_all": (available, auc_weights),
        "lstm_tx_only": ([kind for kind in ("lstm", "tx") if kind in available], {"lstm": 1.0, "tx": 1.0}),
        "no_tcn": ([kind for kind in available if kind != "tcn"], {kind: current_weights.get(kind, 0.0) for kind in available if kind != "tcn"}),
        "tcn_only": (["tcn"] if "tcn" in available else [], {"tcn": 1.0}),
    }
    per_variant_rows: dict[str, list[dict[str, Any]]] = {
        name: [_variant_row(row, kinds, weights, int(snapshot.get("dl_min_agree", 2)), threshold, mode)
               for row in rows]
        for name, (kinds, weights) in specs.items()
    }
    current = per_variant_rows["current_config"]
    output: dict[str, Any] = {}
    for name, evaluated in per_variant_rows.items():
        usable = [item for item in evaluated if item["evaluated"]]
        allowed = [item for item in usable if item["allow"]]
        changed_allow = sum(item["allow"] != cur["allow"] for item, cur in zip(evaluated, current)
                            if item["evaluated"] and cur["evaluated"])
        changed_direction = sum(item["direction"] != cur["direction"] for item, cur in zip(evaluated, current)
                                if item["evaluated"] and cur["evaluated"])
        current_signals = {(i, item["direction"]) for i, item in enumerate(current) if item["allow"]}
        variant_signals = {(i, item["direction"]) for i, item in enumerate(evaluated) if item["allow"]}
        union = current_signals | variant_signals
        symbols: dict[str, Counter[str]] = {}
        for row, item in zip(rows, evaluated):
            symbol = str(row.get("symbol", "__pooled__") or "__pooled__")
            counter = symbols.setdefault(symbol, Counter())
            counter["evaluated_rows"] += int(item["evaluated"])
            counter["allowed_count"] += int(item["allow"])
            counter[f"{item['direction'].lower()}_count"] += 1
        centered = np.asarray([item["centered"] for item in usable], dtype=float)
        output[name] = {
            "evaluated_rows": len(usable),
            "allowed_count": len(allowed),
            "allow_rate": None if not usable else len(allowed) / len(usable),
            "long_count": sum(item["direction"] == "LONG" for item in usable),
            "short_count": sum(item["direction"] == "SHORT" for item in usable),
            "flat_count": sum(item["direction"] == "FLAT" for item in usable),
            "agreement_suppressed_count": sum(item["suppressed"] for item in usable),
            "agreement_suppressed_rate": None if not usable else sum(item["suppressed"] for item in usable) / len(usable),
            "centered_mean": None if not centered.size else float(np.mean(centered)),
            "centered_std": None if not centered.size else float(np.std(centered)),
            "changed_allow_vs_current": changed_allow,
            "changed_direction_vs_current": changed_direction,
            "signal_overlap_with_current": None if not union else len(current_signals & variant_signals) / len(union),
            "per_symbol_summary": {symbol: dict(counter) for symbol, counter in sorted(symbols.items())},
            "shadow_configuration_candidate_only": name in {"lstm_tx_only", "no_tcn"},
        }
    return output


def _manifest_window(identity: str, reports_dir: Path) -> tuple[str, str, dict[str, Any]]:
    manifest = build_evidence_manifest(reports_dir)
    for run in manifest.get("runs", []):
        if run.get("identity") == identity:
            return str(run["run_started_utc"]), str(run["finished_at"]), run
    raise ModelHealthAuditError(f"identity not found in evidence manifest: {identity}")


def resolve_historical_model_outputs(
    identity: Optional[str], start_utc: Optional[str], end_utc: Optional[str],
    *, logs_dir: Path, bundle_root: Path, rows_limit: Optional[int] = None,
) -> dict[str, Any]:
    start = parse_timestamp(start_utc) if start_utc else datetime(1970, 1, 1, tzinfo=timezone.utc)
    finish = parse_timestamp(end_utc) if end_utc else datetime(2100, 1, 1, tzinfo=timezone.utc)
    if start is None or finish is None or finish < start:
        raise ModelHealthAuditError("invalid historical audit window")
    if identity:
        mode, stamp = identity.rsplit(":", 1)
        bundle = bundle_root / f"{mode}_{stamp}"
        if (bundle / "bundle_manifest.json").is_file():
            exact = validate_replay_bundle(bundle)
            bundle_start = parse_timestamp(exact["manifest"].get("run_started_utc"))
            bundle_finish = parse_timestamp(exact["manifest"].get("finished_at"))
            if bundle_start != start or bundle_finish != finish:
                raise ModelHealthAuditError("exact bundle window does not match requested audit window")
            rows_by_file = exact.get("model_output_rows", {})
            if any(rows_by_file.values()):
                return {"status": "exact_bundle", "rows_by_file": rows_by_file,
                        "digest": exact["manifest"].get("model_output_digest"),
                        "duplicate_count": exact["manifest"].get("model_output_duplicate_count", 0),
                        "conflict_count": exact["manifest"].get("model_output_conflict_count", 0)}
    collected = collect_model_output_rows(logs_dir, _iso(start), _iso(finish))
    if collected["model_output_conflict_count"]:
        raise ModelHealthAuditError("conflicting timestamp/symbol model-output rows")
    rows_by_file = {name: list(item["rows"]) for name, item in collected["files"].items()}
    if rows_limit is not None and rows_limit > 0:
        for name, values in rows_by_file.items():
            rows_by_file[name] = values[-rows_limit:]
    return {
        "status": "resolved_from_current_logs" if any(rows_by_file.values()) else "missing",
        "rows_by_file": rows_by_file,
        "digest": collected["model_output_digest"],
        "duplicate_count": collected["model_output_duplicate_count"],
        "conflict_count": collected["model_output_conflict_count"],
    }


def decide_model_health(
    kind: str, contract: Mapping[str, Any], artifact: Mapping[str, Any],
    historical: Mapping[str, Any], probe: Mapping[str, Any], architecture: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    statuses: list[str] = []
    contract_failures = list(contract.get("critical_mismatches", []))
    if any("artifact_load" in reason for reason in contract_failures):
        statuses.append("failed_artifact_load")
    if any("scaler" in reason for reason in contract_failures):
        statuses.append("failed_scaler_contract")
    training_mismatches = [reason for reason in contract_failures
                           if "artifact_load" not in reason and "scaler_nonfinite" not in reason]
    if training_mismatches:
        statuses.append("failed_training_serving_mismatch")
    reasons.extend(contract_failures)
    if artifact.get("model_load_status") != "loaded":
        statuses.append("failed_artifact_load")
    statuses.extend(historical.get("statuses", []))
    if probe.get("status") == "failed_flat_output":
        statuses.append("failed_flat_output")
        reasons.append("offline_probe_variance_below_policy")
    elif str(probe.get("status", "")).startswith("failed"):
        statuses.append("failed_artifact_load")
        reasons.append(str(probe.get("status")))
    auc = _finite_float(artifact.get("metadata_val_auc"))
    if auc is not None and auc < float(policy["low_auc_warning_threshold"]):
        statuses.append("warning_low_auc")
    if kind == "tcn" and architecture.get("architecture_issue_suspected"):
        statuses.append("architecture_issue_suspected")
    statuses = sorted(set(statuses))
    failing = any(status.startswith("failed_") for status in statuses)
    warning = any(status.startswith("warning_") or status == "architecture_issue_suspected" for status in statuses)
    unverified = "unverified" in statuses
    decision = "fail" if failing else ("warning" if warning else ("unverified" if unverified else "pass"))
    return {"decision": decision, "statuses": statuses, "reasons": sorted(set(reasons))}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = BASE_DIR
    reports = Path(args.reports_dir)
    logs = Path(args.logs_dir)
    bundle_root = Path(args.bundle_root)
    policy = load_policy(args.policy)
    run_manifest: dict[str, Any] = {}
    start_utc, end_utc = args.start_utc, args.end_utc
    if args.identity and not (start_utc and end_utc):
        start_utc, end_utc, run_manifest = _manifest_window(args.identity, reports)
    start = parse_timestamp(start_utc) if start_utc else None
    finish = parse_timestamp(end_utc) if end_utc else None

    snapshot_resolution = (
        resolve_model_serving_snapshot(args.identity, reports_dir=reports, bundle_root=bundle_root)
        if args.identity and not args.snapshot else
        {"status": "explicit" if args.snapshot else "missing", "path": args.snapshot,
         "snapshot": None, "digest": None}
    )
    if args.snapshot:
        from tools.model_serving_snapshot import load_model_serving_snapshot
        exact_snapshot = load_model_serving_snapshot(args.snapshot)
        snapshot_resolution.update({"snapshot": exact_snapshot, "digest": exact_snapshot["snapshot_digest"]})
    if snapshot_resolution.get("snapshot") is None:
        snapshot = capture_model_serving_snapshot(
            "current_model_serving", "offline_health_audit", base_dir=root,
            forced_env_overrides={"LIVE_TRADING": False, "PAPER_TRADING": True,
                                  "LIVE_MODE": False, "EXEC_PAPER": True,
                                  "PLACE_REAL_ORDERS": False},
        )
        exact_snapshot_available = False
    else:
        snapshot = snapshot_resolution["snapshot"]
        exact_snapshot_available = True

    historical_source = resolve_historical_model_outputs(
        args.identity, start_utc, end_utc, logs_dir=logs, bundle_root=bundle_root,
        rows_limit=args.rows,
    )
    rows_by_file = historical_source["rows_by_file"]
    if args.symbol:
        rows_by_file = {
            name: [row for row in rows if str(row.get("symbol", "__pooled__") or "__pooled__") == args.symbol]
            for name, rows in rows_by_file.items()
        }
    kinds = [args.model] if args.model else [entry["kind"] for entry in snapshot["model_entries"]]
    writer_candidates: list[Path] = []
    if args.identity:
        mode = args.identity.split(":", 1)[0]
        writer_candidates.extend([
            logs / f"{mode}_paper_writer.err", logs / f"matrix_{mode}_writer.err",
        ])
    writer_candidates.extend([logs / "live_writer.err", logs / "live_writer.out"])
    diagnostics = parse_writer_diagnostics(writer_candidates, start, finish)
    contract = audit_training_serving_contract(
        snapshot, runtime_observations=diagnostics["runtime_observations"],
        exact_snapshot=exact_snapshot_available,
        low_auc_threshold=float(policy["low_auc_warning_threshold"]),
    )
    historical = analyze_historical_probabilities(rows_by_file, kinds, policy,
                                                  symbol_filter=args.symbol)
    for kind in kinds:
        historical["models"].setdefault(kind, {})["auto_exclusion_count"] = sum(
            count for key, count in diagnostics["auto_exclusions"].items() if key.startswith(f"{kind}[")
        )
        historical["models"][kind]["auto_exclusion_reason_counts"] = (
            diagnostics["auto_exclusion_reason_counts_by_model"].get(kind, {})
        )

    if args.inventory_only:
        probes = {"models": {}, "calibration": {}, "tcn_architecture": {}}
        variants: dict[str, Any] = {}
    else:
        probes = run_artifact_probes(snapshot, base_dir=root, seed=args.seed,
                                     probe_count=args.probe_count, policy=policy)
        contract_resolution = resolve_replay_contract(args.identity, reports) if args.identity else {"contract": None}
        replay = contract_resolution.get("contract") or {}
        threshold = float(replay.get("exec_thr", 0.08))
        mode = str(replay.get("exec_mode", "abs"))
        per_symbol_rows = rows_by_file.get("live_models_by_symbol.csv", [])
        variants = analyze_ensemble_variants(per_symbol_rows, snapshot, threshold=threshold, mode=mode)

    entries = {entry["kind"]: entry for entry in snapshot["model_entries"]}
    decisions: dict[str, Any] = {}
    for kind in kinds:
        decisions[kind] = decide_model_health(
            kind, contract.get("models", {}).get(kind, {}), entries.get(kind, {}),
            historical.get("models", {}).get(kind, {}), probes["models"].get(kind, {}),
            probes.get("tcn_architecture", {}) if kind == "tcn" else {}, policy,
        )
    critical_mismatch = bool(contract["critical_mismatches"])
    failed_non_tcn = any(value["decision"] == "fail" for kind, value in decisions.items() if kind != "tcn")
    tcn_quarantine = bool(set(decisions.get("tcn", {}).get("statuses", [])) & {
        "failed_flat_output", "failed_missing_output", "failed_artifact_load",
        "failed_scaler_contract", "architecture_issue_suspected",
    })
    missing_snapshot = not exact_snapshot_available
    if critical_mismatch:
        overall_verdict = "health_audit_fail"
        campaign = "blocked_training_serving_mismatch"
    elif failed_non_tcn:
        overall_verdict = "health_audit_fail"
        campaign = "blocked_model_health_failure"
    elif tcn_quarantine and all(value["decision"] in {"pass", "warning"} for kind, value in decisions.items() if kind != "tcn"):
        overall_verdict = "health_audit_warning"
        campaign = "ready_with_tcn_quarantined_candidate"
    elif missing_snapshot or any(value["decision"] == "unverified" for value in decisions.values()):
        overall_verdict = "health_audit_unverified"
        campaign = "insufficient_evidence"
    elif any(value["decision"] == "warning" for value in decisions.values()):
        overall_verdict = "health_audit_warning"
        campaign = "ready_for_replay_enabled_shadow_campaign"
    else:
        overall_verdict = "health_audit_pass"
        campaign = "ready_for_replay_enabled_shadow_campaign"
    overall = {
        "verdict": overall_verdict,
        "campaign_readiness": campaign,
        "exact_serving_snapshot_available": exact_snapshot_available,
        "tcn_not_eligible_for_positive_ensemble_weight": tcn_quarantine,
        "recommendation_only": True,
        "phase17_or_phase18_evidence": False,
    }
    warnings = list(contract["warnings"])
    if missing_snapshot:
        warnings.append("exact_historical_model_serving_snapshot_missing")
    if run_manifest and not run_manifest.get("include_in_strategy_aggregate", False):
        warnings.append("phase19_run_not_strategy_included_model_diagnostic_only")
    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "policy": {"offline_only": True, "paper_only": True, "real_orders_allowed": False,
                   "trading_configuration_changed": False},
        "inputs": {
            "identity": args.identity,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "model_output_source": historical_source["status"],
            "model_output_digest": historical_source["digest"],
            "policy_digest": hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
            "inventory_only": bool(args.inventory_only),
        },
        "serving_snapshot": {
            "status": snapshot_resolution["status"],
            "exact_historical_snapshot_available": exact_snapshot_available,
            "snapshot_digest": snapshot.get("snapshot_digest"),
            "identity": snapshot.get("identity"),
            "effective_settings": {key: snapshot.get(key) for key in (
                "dl_symbols", "dl_timeframe", "dl_seq_len", "dl_add_symbol_id", "dl_min_agree", "dl_model_weights"
            )},
        },
        "training_serving_contract": contract,
        "artifact_inventory": {entry["kind"]: entry for entry in snapshot["model_entries"]},
        "historical_output_analysis": {**historical, "source_status": historical_source["status"],
                                       "duplicate_count": historical_source["duplicate_count"],
                                       "conflict_count": historical_source["conflict_count"],
                                       "writer_diagnostics": diagnostics},
        "offline_probe_analysis": probes["models"],
        "tcn_architecture_analysis": probes["tcn_architecture"],
        "calibration_analysis": probes["calibration"],
        "ensemble_variant_analysis": variants,
        "model_decisions": decisions,
        "overall_decision": overall,
        "warnings": sorted(set(warnings)),
    }
    output["audit_digest"] = _digest({key: value for key, value in output.items() if key != "generated_at"})
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit deployed ensemble health offline.")
    parser.add_argument("--identity")
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--symbol")
    parser.add_argument("--model", choices=MODEL_KINDS)
    parser.add_argument("--logs-dir", default=str(BASE_DIR / "logs"))
    parser.add_argument("--reports-dir", default=str(BASE_DIR / "reports"))
    parser.add_argument("--bundle-root", default=str(BASE_DIR / "reports" / "replay_bundles"))
    parser.add_argument("--snapshot")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = run_audit(args)
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({"status": "completed", "json_out": str(path),
                          "overall_verdict": output["overall_decision"]["verdict"],
                          "campaign_readiness": output["overall_decision"]["campaign_readiness"],
                          "audit_digest": output["audit_digest"]}))
        return 0
    except (ModelHealthAuditError, ModelServingSnapshotError, ReplayBundleError, OSError, ValueError) as exc:
        print(f"model_health_audit_error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
