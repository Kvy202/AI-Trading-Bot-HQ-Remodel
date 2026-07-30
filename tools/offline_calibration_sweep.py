"""Read-only Phase 17 calibration sweep and multi-run outcome aggregation.

The tool reads completed matrix reports plus the current top-level shadow CSVs.
It never changes trading settings, model artifacts, or runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "offline_calibration_sweep.json"

SHADOW_SAFE_MODES = [
    "baseline",
    "iforest_shadow",
    "xgboost_shadow_outcome",
    "survival_shadow",
    "advanced_risk_shadow",
    "combined_shadow",
]
ACTIVE_VALIDATION_MODES = ["iforest_blocking", "survival_active"]
MODE_ORDER = SHADOW_SAFE_MODES + ACTIVE_VALIDATION_MODES
ACTUAL_COUNT_KEYS = [
    "actually_blocked",
    "actually_rejected",
    "actually_exited",
    "actually_paused",
    "actually_reduced",
]
ZERO_ACTUAL_COUNTS = {key: 0 for key in ACTUAL_COUNT_KEYS}

ROW_LEVEL_LOGS = {
    "isolation_forest": "isolation_forest_shadow.csv",
    "xgboost": "xgboost_signal_shadow.csv",
    "survival_exit": "survival_exit_shadow.csv",
    "advanced_risk": "advanced_risk_shadow.csv",
}

REPORT_RE = re.compile(
    r"^matrix_(?P<mode>.+)_(?P<timestamp>\d{14})_"
    r"(?P<kind>unified|shadow_summary|xgboost_audit)\.json$"
)
INDEX_RE = re.compile(r"^matrix_index_(?P<timestamp>\d{14})\.json$")


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: JSON root is not an object"
    return payload, None


def _read_csv(path: Path) -> Tuple[str, List[Dict[str, str]], Optional[str]]:
    if not path.exists():
        return "missing", [], None
    try:
        if path.stat().st_size == 0:
            return "empty", [], None
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return "malformed", [], f"{path}: CSV header is missing"
            rows = [
                {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            ]
        return ("ok" if rows else "empty"), rows, None
    except Exception as exc:
        return "read_error", [], f"{path}: {type(exc).__name__}: {exc}"


def _as_number(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _as_int(value: Any, default: int = 0) -> int:
    number = _as_number(value)
    return default if number is None else int(number)


def _as_rate(value: Any) -> Optional[float]:
    number = _as_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _get(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        number = _as_number(value)
        if number is not None:
            return number
    return None


def _first_int(*values: Any) -> int:
    for value in values:
        number = _as_number(value)
        if number is not None:
            return int(number)
    return 0


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _numeric_values(values: Iterable[Any]) -> List[float]:
    return [number for number in (_as_number(value) for value in values) if number is not None]


def _average(values: Iterable[Any]) -> Optional[float]:
    numbers = _numeric_values(values)
    return None if not numbers else sum(numbers) / len(numbers)


def _sum_counts(mapping: Any) -> Dict[str, int]:
    if not isinstance(mapping, dict):
        return {}
    result: Dict[str, int] = {}
    for key, value in mapping.items():
        count = _as_int(value)
        if count:
            result[str(key)] = count
    return result


def _canonical_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _rate_key(rate: float) -> str:
    return f"{rate:.12g}"


def _ordered_modes(modes: Iterable[str]) -> List[str]:
    found = set(modes)
    ordered = [mode for mode in MODE_ORDER if mode in found]
    ordered.extend(sorted(mode for mode in found if mode not in MODE_ORDER))
    return ordered


def _module_section(run: Dict[str, Any], unified_key: str, shadow_key: Optional[str] = None) -> Dict[str, Any]:
    payload = run.get("data")
    if not isinstance(payload, dict):
        return {}
    value = payload.get(unified_key)
    if isinstance(value, dict):
        return value
    value = payload.get(shadow_key or unified_key)
    return value if isinstance(value, dict) else {}


def _actual_counts(run: Dict[str, Any]) -> Dict[str, int]:
    payload = run.get("data")
    safety = payload.get("safety") if isinstance(payload, dict) else {}
    nested = safety.get("actual_behavior_counts") if isinstance(safety, dict) else {}
    result = dict(ZERO_ACTUAL_COUNTS)
    for key in ACTUAL_COUNT_KEYS:
        if isinstance(nested, dict) and key in nested:
            result[key] = _as_int(nested.get(key))
            continue
        sections = {
            "actually_blocked": ("isolation_forest", "actually_blocked_count"),
            "actually_rejected": ("xgboost", "actually_rejected_count"),
            "actually_exited": ("survival_exit", "actually_exited_count"),
            "actually_paused": ("advanced_risk", "actually_paused_count"),
            "actually_reduced": ("advanced_risk", "actually_reduced_count"),
        }
        section_name, count_name = sections[key]
        result[key] = _as_int(_module_section(run, section_name, "xgboost_signal").get(count_name))
    audit = run.get("audit")
    if isinstance(audit, dict):
        result["actually_rejected"] = max(
            result["actually_rejected"],
            _as_int(audit.get("actually_rejected_count")),
        )
    return result


def _add_actual_counts(total: Dict[str, int], values: Dict[str, int]) -> None:
    for key in ACTUAL_COUNT_KEYS:
        total[key] = total.get(key, 0) + _as_int(values.get(key))


def _parse_report_identity(path: Path) -> Optional[Tuple[str, str, str]]:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    match = REPORT_RE.match(name)
    if match is None:
        return None
    return match.group("mode"), match.group("timestamp"), match.group("kind")


def _index_run_status(payload: Dict[str, Any], fallback_timestamp: str) -> List[Tuple[str, str, int]]:
    statuses: List[Tuple[str, str, int]] = []
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return statuses
    for item in runs:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode") or payload.get("requested_mode") or "").strip()
        if not mode:
            continue
        timestamp = fallback_timestamp
        paths = item.get("report_paths")
        if isinstance(paths, dict):
            for raw_path in paths.values():
                identity = _parse_report_identity(Path(str(raw_path)))
                if identity is not None and identity[0] == mode:
                    timestamp = identity[1]
                    break
        statuses.append((mode, timestamp, _as_int(item.get("exit_status"), default=1)))
    return statuses


def _legacy_audit_mode(path: Path) -> Optional[str]:
    name = path.name.lower()
    if name.startswith("combined_shadow_") and "xgboost_audit" in name:
        return "combined_shadow"
    if name.startswith("xgboost_shadow_outcome_") and "audit" in name:
        return "xgboost_shadow_outcome"
    return None


def _load_inputs(reports_dir: Path, logs_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    report_paths: set[Path] = set()
    for pattern in (
        "matrix_*_unified.json",
        "matrix_*_shadow_summary.json",
        "matrix_*_xgboost_audit.json",
        "matrix_index_*.json",
    ):
        report_paths.update(reports_dir.glob(pattern))
    for name in ("final_experiment_comparison.json", "calibration_recommendation_report.json"):
        path = reports_dir / name
        if path.exists():
            report_paths.add(path)
    for pattern in ("combined_shadow_xgboost_audit*.json", "xgboost_shadow_outcome_paper_audit*.json"):
        report_paths.update(reports_dir.glob(pattern))

    sorted_paths = sorted(report_paths, key=lambda item: item.name)
    payloads: Dict[Path, Dict[str, Any]] = {}
    malformed: List[Dict[str, str]] = []
    read_errors: List[str] = []
    for path in sorted_paths:
        payload, error = _read_json(path)
        if error is not None:
            malformed.append({"path": str(path), "error": error})
            read_errors.append(error)
        elif payload is not None:
            payloads[path] = payload

    index_statuses: Dict[Tuple[str, str], int] = {}
    index_files_used: List[str] = []
    for path, payload in payloads.items():
        match = INDEX_RE.match(path.name)
        if match is None:
            continue
        index_files_used.append(str(path))
        for mode, timestamp, exit_status in _index_run_status(payload, match.group("timestamp")):
            index_statuses[(mode, timestamp)] = exit_status

    groups: Dict[Tuple[str, str], Dict[str, Tuple[Path, Dict[str, Any]]]] = {}
    for path, payload in payloads.items():
        identity = _parse_report_identity(path)
        if identity is None:
            continue
        mode, timestamp, kind = identity
        groups.setdefault((mode, timestamp), {})[kind] = (path, payload)

    duplicate_reports: List[Dict[str, str]] = []
    incomplete_reports: List[Dict[str, str]] = []
    aggregation_files: List[str] = []
    runs: List[Dict[str, Any]] = []
    matrix_audit_payloads: Dict[str, List[Tuple[str, str, str]]] = {}
    for (mode, timestamp), parts in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        exit_status = index_statuses.get((mode, timestamp))
        if exit_status is not None and exit_status != 0:
            for path, _payload in parts.values():
                incomplete_reports.append(
                    {
                        "path": str(path),
                        "identity": f"{mode}:{timestamp}",
                        "reason": f"skipped because matrix index exit_status={exit_status}",
                    }
                )
            continue

        primary_kind: Optional[str] = None
        primary: Optional[Tuple[Path, Dict[str, Any]]] = None
        for candidate in ("unified", "shadow_summary"):
            if candidate in parts:
                primary_kind = candidate
                primary = parts[candidate]
                break
        if "unified" in parts and "shadow_summary" in parts:
            duplicate_reports.append(
                {
                    "path": str(parts["shadow_summary"][0]),
                    "retained_path": str(parts["unified"][0]),
                    "identity": f"{mode}:{timestamp}",
                    "reason": "lower-priority repeated representation of the same run",
                }
            )

        audit_path: Optional[Path] = None
        audit_payload: Dict[str, Any] = {}
        if "xgboost_audit" in parts:
            audit_path, audit_payload = parts["xgboost_audit"]
            aggregation_files.append(str(audit_path))
            matrix_audit_payloads.setdefault(_canonical_payload(audit_payload), []).append(
                (mode, timestamp, str(audit_path))
            )

        if primary is None and not audit_payload:
            continue
        if primary is not None:
            aggregation_files.append(str(primary[0]))
        runs.append(
            {
                "mode": mode,
                "timestamp": timestamp,
                "identity": f"{mode}:{timestamp}",
                "primary_kind": primary_kind,
                "path": None if primary is None else str(primary[0]),
                "data": {} if primary is None else primary[1],
                "audit_path": None if audit_path is None else str(audit_path),
                "audit": audit_payload,
                "completion_status": "completed_index" if exit_status == 0 else "report_present_no_index",
            }
        )

    unverified_reports: List[Dict[str, str]] = []
    for path, payload in payloads.items():
        mode = _legacy_audit_mode(path)
        if mode is None:
            continue
        matches = matrix_audit_payloads.get(_canonical_payload(payload), [])
        if len(matches) == 1:
            match = matches[0]
            duplicate_reports.append(
                {
                    "path": str(path),
                    "retained_path": match[2],
                    "identity": f"{match[0]}:{match[1]}",
                    "reason": "generated audit duplicates the matrix audit for this run",
                }
            )
        else:
            unverified_reports.append(
                {
                    "path": str(path),
                    "reason": (
                        "audit content matches multiple matrix runs, so mode + timestamp is ambiguous"
                        if len(matches) > 1
                        else "audit has no safely verifiable mode + timestamp identity"
                    ),
                }
            )

    supplemental: Dict[str, Dict[str, Any]] = {}
    supplemental_files: List[str] = []
    for name in ("final_experiment_comparison.json", "calibration_recommendation_report.json"):
        path = reports_dir / name
        payload = payloads.get(path)
        if payload is not None:
            supplemental[name] = payload
            supplemental_files.append(str(path))

    logs: Dict[str, Dict[str, Any]] = {}
    log_errors: List[str] = []
    for key, name in ROW_LEVEL_LOGS.items():
        path = logs_dir / name
        status, rows, error = _read_csv(path)
        logs[key] = {
            "path": str(path),
            "status": status,
            "row_count": len(rows),
            "columns": sorted(rows[0].keys()) if rows else [],
            "rows": rows,
        }
        if error is not None:
            log_errors.append(error)
    read_errors.extend(log_errors)

    timestamps_by_mode: Dict[str, List[str]] = {}
    for run in runs:
        timestamps_by_mode.setdefault(run["mode"], []).append(run["timestamp"])
    timestamps_by_mode = {
        mode: sorted(set(timestamps_by_mode.get(mode, [])))
        for mode in _ordered_modes(timestamps_by_mode)
    }

    missing_inputs: List[str] = []
    if not runs:
        missing_inputs.append("completed matrix run reports")
    for mode in MODE_ORDER:
        if mode not in timestamps_by_mode:
            missing_inputs.append(f"completed matrix runs for {mode}")
    for key, item in logs.items():
        if item["status"] != "ok":
            missing_inputs.append(ROW_LEVEL_LOGS[key])

    used_files = sorted(set(aggregation_files + index_files_used + supplemental_files))
    found_logs = [item["path"] for item in logs.values() if item["status"] == "ok"]
    inventory = {
        "reports_dir": str(reports_dir),
        "logs_dir": str(logs_dir),
        "report_files_found": [str(path) for path in sorted_paths],
        "report_file_count_found": len(sorted_paths),
        "report_files_used": used_files,
        "report_file_count_used": len(used_files),
        "aggregation_report_files_used": sorted(set(aggregation_files)),
        "row_level_logs_found": found_logs,
        "row_level_logs": {
            key: {
                "path": value["path"],
                "status": value["status"],
                "row_count": value["row_count"],
                "columns": value["columns"],
            }
            for key, value in logs.items()
        },
        "run_timestamps_by_mode": timestamps_by_mode,
        "run_timestamps_grouped_by_mode": timestamps_by_mode,
        "duplicate_reports_skipped": duplicate_reports,
        "duplicate_report_count": len(duplicate_reports),
        "malformed_reports_skipped": malformed,
        "malformed_report_count": len(malformed),
        "incomplete_reports_skipped": incomplete_reports,
        "incomplete_report_count": len(incomplete_reports),
        "unverified_reports_skipped": unverified_reports,
        "supplemental_reports_read": sorted(supplemental_files),
        "supplemental_report_role": "context_only_not_recounted_as_matrix_runs",
        "missing_inputs": missing_inputs,
        "read_errors": read_errors,
        "archived_csv_policy": "not_aggregated_without a safely verified run identity",
        "trade_csv_policy": (
            "master and dated trade CSVs are not read; matrix audits provide deduplicated outcomes"
        ),
    }
    return inventory, runs, logs


def _percentile(sorted_values: Sequence[float], percentile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    bounded = max(0.0, min(100.0, float(percentile)))
    rank = (bounded / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


def _score_distribution(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    numbers = sorted(_numeric_values(values))
    output: Dict[str, Optional[float]] = {
        "min": numbers[0] if numbers else None,
        "p1": _percentile(numbers, 1),
        "p5": _percentile(numbers, 5),
        "p10": _percentile(numbers, 10),
        "p25": _percentile(numbers, 25),
        "p50": _percentile(numbers, 50),
        "p75": _percentile(numbers, 75),
        "p90": _percentile(numbers, 90),
        "p95": _percentile(numbers, 95),
        "p99": _percentile(numbers, 99),
        "max": numbers[-1] if numbers else None,
        "average": (sum(numbers) / len(numbers)) if numbers else None,
    }
    return output


def _tie_frequency(values: Iterable[Any]) -> Dict[str, Any]:
    numbers = _numeric_values(values)
    counts = Counter(numbers)
    tied_groups = {score: count for score, count in counts.items() if count > 1}
    rows_in_ties = sum(tied_groups.values())
    max_tie = max(counts.values(), default=0)
    return {
        "score_count": len(numbers),
        "unique_score_values": len(counts),
        "tied_score_value_count": len(tied_groups),
        "rows_in_tied_scores": rows_in_ties,
        "tied_row_rate": None if not numbers else rows_in_ties / len(numbers),
        "largest_tie_count": max_tie,
        "largest_tie_rate": None if not numbers else max_tie / len(numbers),
    }


def _threshold_simulations(
    values: Iterable[Any],
    targets: Iterable[float],
    *,
    lower_score_triggers: bool,
    minimum_threshold: Optional[float] = None,
    maximum_threshold: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    numbers = _numeric_values(values)
    counts = Counter(numbers)
    ordered = sorted(counts, reverse=not lower_score_triggers)
    achievable: List[Dict[str, Any]] = []
    if ordered:
        zero_threshold = math.nextafter(
            ordered[0],
            -math.inf if lower_score_triggers else math.inf,
        )
        zero_within_bounds = (
            (minimum_threshold is None or zero_threshold >= minimum_threshold)
            and (maximum_threshold is None or zero_threshold <= maximum_threshold)
        )
        if zero_within_bounds:
            achievable.append(
                {
                    "threshold": zero_threshold,
                    "count": 0,
                    "achieved_rate": 0.0,
                    "score_tie_count_at_threshold": 0,
                }
            )
        cumulative = 0
        for score in ordered:
            cumulative += counts[score]
            if minimum_threshold is not None and score < minimum_threshold:
                continue
            if maximum_threshold is not None and score > maximum_threshold:
                continue
            achievable.append(
                {
                    "threshold": score,
                    "count": cumulative,
                    "achieved_rate": cumulative / len(numbers),
                    "score_tie_count_at_threshold": counts[score],
                }
            )

    result: Dict[str, Dict[str, Any]] = {}
    for raw_target in targets:
        target = float(raw_target)
        below_candidates = [item for item in achievable if item["achieved_rate"] <= target + 1e-12]
        above_candidates = [item for item in achievable if item["achieved_rate"] >= target - 1e-12]
        below = max(below_candidates, key=lambda item: item["achieved_rate"]) if below_candidates else None
        above = min(above_candidates, key=lambda item: item["achieved_rate"]) if above_candidates else None
        candidates = [item for item in (below, above) if item is not None]
        chosen = (
            min(candidates, key=lambda item: (abs(item["achieved_rate"] - target), item["achieved_rate"]))
            if candidates
            else None
        )
        exact = bool(chosen is not None and abs(chosen["achieved_rate"] - target) <= 1e-12)
        tied_limitation = bool(
            not exact
            and above is not None
            and (
                below is None
                or above["achieved_rate"] > below["achieved_rate"]
            )
            and above["score_tie_count_at_threshold"] > 1
        )
        result[_rate_key(target)] = {
            "requested_rate": target,
            "threshold": None if chosen is None else chosen["threshold"],
            "achieved_rate": None if chosen is None else chosen["achieved_rate"],
            "achieved_count": None if chosen is None else chosen["count"],
            "absolute_rate_error": None if chosen is None else abs(chosen["achieved_rate"] - target),
            "target_exactly_achievable": exact,
            "tied_score_limitation": tied_limitation,
            "warning": (
                "tied scores make the requested rate impossible at a single inclusive threshold"
                if tied_limitation
                else ""
            ),
            "nearest_achievable_below": below,
            "nearest_achievable_above": above,
        }
    return result


def _reconstructed_winners(count: int, win_rate: Any) -> Optional[int]:
    rate = _as_rate(win_rate)
    if count < 0 or rate is None:
        return None
    raw = count * rate
    rounded = int(round(raw))
    return rounded if abs(raw - rounded) <= 1e-6 else None


def _coverage(total: int, covered: int) -> Dict[str, Any]:
    return {
        "rows": total,
        "covered_rows": covered,
        "missing_rows": max(0, total - covered),
        "coverage_rate": None if total <= 0 else covered / total,
    }


def _run_classification(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    shadow_runs_by_mode: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in SHADOW_SAFE_MODES}
    active_runs_by_mode: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in ACTIVE_VALIDATION_MODES}
    shadow_runs: List[Dict[str, Any]] = []
    active_runs: List[Dict[str, Any]] = []
    shadow_failures: List[Dict[str, Any]] = []
    for run in runs:
        mode = run["mode"]
        actual = _actual_counts(run)
        any_actual = any(actual.values())
        item = {
            "mode": mode,
            "timestamp": run["timestamp"],
            "identity": run["identity"],
            "actual_behavior_counts": actual,
        }
        if mode in SHADOW_SAFE_MODES:
            item["shadow_safety_passed"] = not any_actual
            shadow_runs_by_mode[mode].append(item)
            shadow_runs.append(item)
            if any_actual:
                shadow_failures.append(item)
        elif mode in ACTIVE_VALIDATION_MODES:
            expected_key = "actually_blocked" if mode == "iforest_blocking" else "actually_exited"
            item.update(
                {
                    "expected_actual_behavior": expected_key,
                    "expected_actual_behavior_observed": actual.get(expected_key, 0) > 0,
                    "not_counted_as_shadow_safety_failure": True,
                }
            )
            active_runs_by_mode[mode].append(item)
            active_runs.append(item)
    return {
        "shadow_safe_modes": SHADOW_SAFE_MODES,
        "intentional_active_validation_modes": ACTIVE_VALIDATION_MODES,
        "shadow_safe_runs": shadow_runs,
        "shadow_safe_runs_by_mode": shadow_runs_by_mode,
        "intentional_active_validation_runs": active_runs,
        "intentional_active_validation_runs_by_mode": active_runs_by_mode,
        "shadow_safety_violations": shadow_failures,
        "shadow_safety_failures": [item["identity"] for item in shadow_failures],
        "shadow_only_safety_passed": not shadow_failures,
        "active_validation_note": (
            "Expected actual blocking/exits in intentional active validation are separated "
            "and are not shadow-only safety failures."
        ),
    }


def _baseline_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_runs: List[Dict[str, Any]] = []
    for run in runs:
        if run["mode"] != "baseline":
            continue
        pnl = _module_section(run, "paper_pnl")
        count = _first_int(pnl.get("closed_trade_count"), pnl.get("closed_trades"))
        total = _first_number(pnl.get("total_pnl"))
        average = _first_number(pnl.get("average_pnl"))
        if total is None and average is not None and count > 0:
            total = average * count
        if average is None and total is not None and count:
            average = total / count
        winners = _reconstructed_winners(count, pnl.get("win_rate"))
        baseline_runs.append(
            {
                "timestamp": run["timestamp"],
                "identity": run["identity"],
                "source_report": run["path"],
                "closed_trades": count,
                "total_pnl": total,
                "average_pnl": average,
                "win_rate": _as_rate(pnl.get("win_rate")),
                "reconstructed_winner_count": winners,
            }
        )
    baseline_runs.sort(key=lambda item: item["timestamp"])
    counts = sum(item["closed_trades"] for item in baseline_runs)
    pnl_supported_counts = sum(
        item["closed_trades"]
        for item in baseline_runs
        if item["total_pnl"] is not None
    )
    totals = [item["total_pnl"] for item in baseline_runs if item["total_pnl"] is not None]
    total_pnl = sum(totals) if totals else None
    reconstructed = [
        item["reconstructed_winner_count"]
        for item in baseline_runs
        if item["reconstructed_winner_count"] is not None
    ]
    reconstruction_supported = bool(baseline_runs) and len(reconstructed) == len(baseline_runs)
    overall_win_rate = (
        sum(reconstructed) / counts
        if reconstruction_supported and counts > 0
        else None
    )
    positive = sum(1 for value in totals if value > 0)
    negative = sum(1 for value in totals if value < 0)
    sample_warning = (
        counts < 30
        or len(baseline_runs) < 3
        or pnl_supported_counts != counts
    )
    if counts <= 0 or total_pnl is None:
        verdict = "baseline_unproven"
    elif positive and negative:
        verdict = "baseline_inconsistent"
    elif total_pnl <= 0 or (overall_win_rate is not None and overall_win_rate < 0.40):
        verdict = "baseline_weak"
    else:
        verdict = "baseline_promising_but_unproven"
    return {
        "number_of_runs": len(baseline_runs),
        "runs": baseline_runs,
        "total_closed_trades": counts,
        "total_pnl": total_pnl,
        "weighted_average_pnl": (
            None
            if not pnl_supported_counts or total_pnl is None
            else total_pnl / pnl_supported_counts
        ),
        "pnl_supported_closed_trade_count": pnl_supported_counts,
        "pnl_coverage_rate": None if counts <= 0 else pnl_supported_counts / counts,
        "overall_win_rate": overall_win_rate,
        "raw_winner_counts_reconstructed": reconstruction_supported,
        "reconstructed_winner_count": sum(reconstructed) if reconstruction_supported else None,
        "min_run_pnl": min(totals) if totals else None,
        "max_run_pnl": max(totals) if totals else None,
        "minimum_run_pnl": min(totals) if totals else None,
        "maximum_run_pnl": max(totals) if totals else None,
        "positive_run_count": positive,
        "negative_run_count": negative,
        "zero_run_count": sum(1 for value in totals if value == 0),
        "latest_run": baseline_runs[-1] if baseline_runs else None,
        "sample_size_warning": sample_warning,
        "sample_size_message": (
            "Baseline evidence is still a small multi-run paper sample; no profitability claim is supported."
            if sample_warning
            else "Multiple paper runs are available, but this remains paper evidence rather than proof of profitability."
        ),
        "verdict": verdict,
        "profitability_claim": False,
    }


def _iforest_run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    section = _module_section(run, "isolation_forest")
    return {
        "mode": run["mode"],
        "timestamp": run["timestamp"],
        "identity": run["identity"],
        "source_report": run["path"],
        "total_rows": _as_int(section.get("total_rows")),
        "would_block_count": _as_int(section.get("would_block_count")),
        "would_block_rate": _as_rate(section.get("would_block_rate")),
        "actually_blocked_count": _as_int(section.get("actually_blocked_count")),
        "actual_block_rate": _as_rate(
            _first_number(section.get("actual_block_rate"), section.get("block_rate"))
        ),
        "anomaly_score_statistics": {
            "min": _first_number(section.get("min_anomaly_score")),
            "p10": _first_number(section.get("p10_anomaly_score")),
            "p50": _first_number(section.get("p50_anomaly_score")),
            "p90": _first_number(section.get("p90_anomaly_score")),
            "max": _first_number(section.get("max_anomaly_score")),
            "average": _first_number(section.get("average_anomaly_score")),
            "latest": _first_number(section.get("latest_anomaly_score")),
        },
    }


def _isolation_forest_analysis(
    runs: List[Dict[str, Any]],
    log: Dict[str, Any],
    targets: Sequence[float],
) -> Dict[str, Any]:
    dedicated_shadow_runs = [
        _iforest_run_metrics(run)
        for run in runs
        if run["mode"] == "iforest_shadow"
    ]
    blocking_runs = [
        _iforest_run_metrics(run)
        for run in runs
        if run["mode"] == "iforest_blocking"
    ]
    combined_runs = [
        _iforest_run_metrics(run)
        for run in runs
        if run["mode"] == "combined_shadow"
    ]
    dedicated_shadow_runs.sort(key=lambda item: item["timestamp"])
    blocking_runs.sort(key=lambda item: item["timestamp"])
    combined_runs.sort(key=lambda item: item["timestamp"])
    shadow_runs = sorted(
        dedicated_shadow_runs + combined_runs,
        key=lambda item: (item["timestamp"], item["identity"]),
    )

    rows = log.get("rows") if isinstance(log.get("rows"), list) else []
    scores = _numeric_values(row.get("anomaly_score") for row in rows)
    tie_frequency = _tie_frequency(scores)
    simulations = _threshold_simulations(scores, targets, lower_score_triggers=True)
    evidence_sufficient = len(scores) >= 20
    minimum_positive_rate = (
        Counter(scores)[min(scores)] / len(scores)
        if scores
        else None
    )
    # A threshold that only allows 0% or a large first jump is not useful calibration.
    threshold_feasible = bool(
        evidence_sufficient
        and minimum_positive_rate is not None
        and minimum_positive_rate <= 0.05 + 1e-12
    )
    largest_tie_rate = tie_frequency.get("largest_tie_rate")
    unique_ratio = (
        tie_frequency["unique_score_values"] / len(scores)
        if scores
        else None
    )
    saturation = bool(
        evidence_sufficient
        and (
            (largest_tie_rate is not None and largest_tie_rate >= 0.10)
            or (unique_ratio is not None and unique_ratio <= 0.20)
        )
    )
    all_report_runs = shadow_runs + blocking_runs
    observed_rates = [
        item["would_block_rate"]
        for item in all_report_runs
        if item["would_block_rate"] is not None and item["total_rows"] > 0
    ]
    extreme = bool(observed_rates and all(rate >= 0.95 for rate in observed_rates))
    hundred_percent_count = sum(abs(rate - 1.0) <= 1e-12 for rate in observed_rates)
    consistent_100 = bool(observed_rates and hundred_percent_count == len(observed_rates))
    retraining = bool(
        evidence_sufficient
        and (saturation or not threshold_feasible)
        and (extreme or saturation)
    )
    if retraining and not observed_rates:
        verdict = "retraining_recommended"
    elif not evidence_sufficient and not observed_rates:
        verdict = "needs_more_shadow_data"
    elif extreme:
        verdict = "unsafe_to_enable"
    elif not evidence_sufficient:
        verdict = "needs_more_shadow_data"
    elif retraining:
        verdict = "retraining_recommended"
    else:
        verdict = "needs_more_shadow_data"

    warnings: List[str] = []
    if not evidence_sufficient:
        warnings.append("Current row-level Isolation Forest evidence is insufficient for calibration.")
    if any(item["tied_score_limitation"] for item in simulations.values()):
        warnings.append("Tied anomaly scores make one or more requested block rates impossible.")
    if not threshold_feasible and evidence_sufficient:
        warnings.append("Threshold-only calibration cannot produce a useful low single-digit positive block rate.")
    if saturation:
        warnings.append("Anomaly-score saturation/ties are present.")

    return {
        "number_of_shadow_runs": len(shadow_runs),
        "number_of_dedicated_shadow_runs": len(dedicated_shadow_runs),
        "number_of_combined_shadow_runs": len(combined_runs),
        "number_of_blocking_validation_runs": len(blocking_runs),
        "shadow_runs": {
            "number_of_runs": len(shadow_runs),
            "runs": shadow_runs,
        },
        "dedicated_shadow_runs": {
            "number_of_runs": len(dedicated_shadow_runs),
            "runs": dedicated_shadow_runs,
        },
        "blocking_validation_runs": {
            "number_of_runs": len(blocking_runs),
            "runs": blocking_runs,
        },
        "shadow_run_details": shadow_runs,
        "blocking_validation_run_details": blocking_runs,
        "combined_shadow_runs": combined_runs,
        "would_block_rate_by_run": {
            item["identity"]: item["would_block_rate"] for item in shadow_runs
        },
        "actual_block_rate_by_blocking_run": {
            item["identity"]: item["actual_block_rate"] for item in blocking_runs
        },
        "anomaly_score_statistics_by_run": {
            item["identity"]: item["anomaly_score_statistics"]
            for item in all_report_runs
        },
        "hundred_percent_anomaly_consistency": {
            "observed_run_count": len(observed_rates),
            "hundred_percent_run_count": hundred_percent_count,
            "consistent": consistent_100,
        },
        "consistent_100_percent_anomaly_behavior": consistent_100,
        "row_level_source": {
            "path": log.get("path"),
            "status": log.get("status"),
            "row_count": len(rows),
            "score_count": len(scores),
        },
        "score_distribution": _score_distribution(scores),
        "unique_score_values": tie_frequency["unique_score_values"],
        "tie_frequency": tie_frequency,
        "threshold_simulations": simulations,
        "row_level_analysis": {
            "source": {
                "path": log.get("path"),
                "status": log.get("status"),
                "row_count": len(rows),
            },
            "score_distribution": {
                **_score_distribution(scores),
                "count": len(scores),
                "unique_score_values": tie_frequency["unique_score_values"],
                "tied_score_frequency": tie_frequency["tied_row_rate"],
            },
            "threshold_simulations": simulations,
            "tie_frequency": tie_frequency,
        },
        "threshold_only_calibration_feasible": threshold_feasible,
        "minimum_positive_achievable_block_rate": minimum_positive_rate,
        "score_saturation_detected": saturation,
        "retraining_recommended": retraining,
        "row_level_evidence_sufficient": evidence_sufficient,
        "warnings": warnings,
        "verdict": verdict,
        "calibration_verdict": (
            "retraining_recommended"
            if retraining
            else ("needs_more_shadow_data" if not evidence_sufficient else "threshold_review_required")
        ),
        "artifact_modified": False,
        "allowed_mode": "shadow_only",
        "prohibited_mode": "blocking",
    }


def _xgb_group_from_audit(audit: Dict[str, Any], group: str) -> Dict[str, Any]:
    join = audit.get("trade_outcome_join")
    join = join if isinstance(join, dict) else {}
    matched = join.get("matched_closed_trade_pnl")
    matched = matched if isinstance(matched, dict) else {}
    if group == "confirmed":
        aliases = ("would_confirm", "allowed")
        count_values = (
            audit.get("would_confirm_matched_count"),
            join.get("would_confirm_matched_count"),
        )
        average_values = (
            audit.get("would_confirm_average_pnl"),
            join.get("would_confirm_average_pnl"),
        )
        win_values = (
            audit.get("would_confirm_win_rate"),
            join.get("would_confirm_win_rate"),
        )
    else:
        aliases = ("would_reject", "rejected")
        count_values = (
            audit.get("would_reject_matched_count"),
            join.get("would_reject_matched_count"),
        )
        average_values = (
            audit.get("would_reject_average_pnl"),
            join.get("would_reject_average_pnl"),
        )
        win_values = (
            audit.get("would_reject_win_rate"),
            join.get("would_reject_win_rate"),
        )

    nested: Dict[str, Any] = {}
    for alias in aliases:
        candidate = matched.get(alias)
        if isinstance(candidate, dict):
            nested = candidate
            break
    count = _first_int(nested.get("count"), *count_values)
    average = _first_number(nested.get("average_pnl"), *average_values)
    total = _first_number(nested.get("total_pnl"))
    if total is None and average is not None:
        total = average * count
    if average is None and total is not None and count > 0:
        average = total / count
    win_rate = _as_rate(_first_number(nested.get("win_rate"), *win_values))
    return {
        "matched_count": count,
        "total_pnl": total,
        "weighted_average_pnl": average,
        "pnl_supported_matched_count": count if total is not None else 0,
        "pnl_coverage_rate": (
            None
            if count <= 0
            else (1.0 if total is not None else 0.0)
        ),
        "win_rate": win_rate,
        "reconstructed_winner_count": _reconstructed_winners(count, win_rate),
    }


def _xgb_run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    audit = run.get("audit")
    audit = audit if isinstance(audit, dict) else {}
    confirmed = _xgb_group_from_audit(audit, "confirmed")
    rejected = _xgb_group_from_audit(audit, "rejected")
    join = audit.get("trade_outcome_join")
    join = join if isinstance(join, dict) else {}
    total = _first_int(audit.get("total_xgboost_rows"), audit.get("total_decision_rows"))
    would_confirm = _first_int(audit.get("would_confirm_count"), audit.get("allowed_signal_count"))
    would_reject = _first_int(audit.get("would_reject_count"), audit.get("rejected_signal_count"))
    matched_count = _first_int(
        join.get("matched_closed_trade_count"),
        confirmed["matched_count"] + rejected["matched_count"],
    )
    unmatched = _first_int(
        join.get("unmatched_decision_rows"),
        max(0, would_confirm + would_reject - matched_count),
    )
    coverage_denominator = matched_count + unmatched
    confirm_average = confirmed["weighted_average_pnl"]
    reject_average = rejected["weighted_average_pnl"]
    relation = (
        reject_average < confirm_average
        if confirm_average is not None
        and reject_average is not None
        and confirmed["matched_count"] > 0
        and rejected["matched_count"] > 0
        else None
    )
    return {
        "timestamp": run["timestamp"],
        "identity": run["identity"],
        "source_mode": run["mode"],
        "source_audit": run.get("audit_path"),
        "total_decision_rows": total,
        "would_confirm_count": would_confirm,
        "would_reject_count": would_reject,
        "would_reject_rate": None if total <= 0 else would_reject / total,
        "reject_reasons": _sum_counts(audit.get("reject_reason_counts")),
        "average_allowed_confidence": _first_number(audit.get("average_confidence_allowed")),
        "average_rejected_confidence": _first_number(audit.get("average_confidence_rejected")),
        "confirmed": confirmed,
        "rejected": rejected,
        "matched_closed_trade_count": matched_count,
        "unmatched_decision_count": unmatched,
        "match_coverage_rate": (
            None if coverage_denominator <= 0 else matched_count / coverage_denominator
        ),
        "outcome_eligible_decision_count": coverage_denominator,
        "pnl_separation": (
            None
            if confirm_average is None or reject_average is None
            else confirm_average - reject_average
        ),
        "rejected_trades_worse_than_confirmed": relation,
    }


def _aggregate_xgb_runs(
    run_metrics: List[Dict[str, Any]],
    min_matched_per_group: int,
) -> Dict[str, Any]:
    reasons: Counter[str] = Counter()
    total_rows = 0
    confirm_decisions = 0
    reject_decisions = 0
    matched = 0
    unmatched = 0
    confidence_allowed_total = 0.0
    confidence_allowed_weight = 0
    confidence_rejected_total = 0.0
    confidence_rejected_weight = 0
    confirmed_count = 0
    confirmed_pnl_supported_count = 0
    confirmed_total = 0.0
    confirmed_winners = 0
    confirmed_win_supported = 0
    rejected_count = 0
    rejected_pnl_supported_count = 0
    rejected_total = 0.0
    rejected_winners = 0
    rejected_win_supported = 0

    for item in run_metrics:
        total_rows += item["total_decision_rows"]
        confirm_decisions += item["would_confirm_count"]
        reject_decisions += item["would_reject_count"]
        reasons.update(item["reject_reasons"])
        matched += item["matched_closed_trade_count"]
        unmatched += item["unmatched_decision_count"]
        allowed_confidence = item["average_allowed_confidence"]
        if allowed_confidence is not None and item["would_confirm_count"] > 0:
            confidence_allowed_total += allowed_confidence * item["would_confirm_count"]
            confidence_allowed_weight += item["would_confirm_count"]
        rejected_confidence = item["average_rejected_confidence"]
        if rejected_confidence is not None and item["would_reject_count"] > 0:
            confidence_rejected_total += rejected_confidence * item["would_reject_count"]
            confidence_rejected_weight += item["would_reject_count"]

        confirmed = item["confirmed"]
        confirmed_count += confirmed["matched_count"]
        confirmed_pnl_supported_count += confirmed["pnl_supported_matched_count"]
        if confirmed["total_pnl"] is not None:
            confirmed_total += confirmed["total_pnl"]
        if confirmed["reconstructed_winner_count"] is not None:
            confirmed_winners += confirmed["reconstructed_winner_count"]
            confirmed_win_supported += confirmed["matched_count"]

        rejected = item["rejected"]
        rejected_count += rejected["matched_count"]
        rejected_pnl_supported_count += rejected["pnl_supported_matched_count"]
        if rejected["total_pnl"] is not None:
            rejected_total += rejected["total_pnl"]
        if rejected["reconstructed_winner_count"] is not None:
            rejected_winners += rejected["reconstructed_winner_count"]
            rejected_win_supported += rejected["matched_count"]

    confirmed_average = (
        confirmed_total / confirmed_count
        if confirmed_count > 0 and confirmed_pnl_supported_count == confirmed_count
        else None
    )
    rejected_average = (
        rejected_total / rejected_count
        if rejected_count > 0 and rejected_pnl_supported_count == rejected_count
        else None
    )
    outcome_bearing_runs = [
        item
        for item in run_metrics
        if (
            item["confirmed"]["matched_count"] > 0
            or item["rejected"]["matched_count"] > 0
        )
    ]
    comparable = [
        item
        for item in outcome_bearing_runs
        if item["rejected_trades_worse_than_confirmed"] is not None
    ]
    relationships = [item["rejected_trades_worse_than_confirmed"] for item in comparable]
    if len(comparable) < 2:
        consistent: Optional[bool] = None
    elif len(comparable) != len(outcome_bearing_runs):
        consistent = False
    else:
        consistent = all(value == relationships[0] for value in relationships)
    pnl_fully_supported = (
        confirmed_count > 0
        and rejected_count > 0
        and confirmed_pnl_supported_count == confirmed_count
        and rejected_pnl_supported_count == rejected_count
    )
    rejected_worse = (
        rejected_average < confirmed_average
        if confirmed_average is not None and rejected_average is not None
        else None
    )
    threshold_satisfied = (
        confirmed_count >= min_matched_per_group
        and rejected_count >= min_matched_per_group
    )
    if consistent is False:
        verdict = "inconsistent_across_runs"
    elif confirmed_count == 0 or rejected_count == 0:
        verdict = "insufficient_matched_outcomes"
    elif not pnl_fully_supported:
        verdict = "insufficient_matched_outcomes"
    elif not threshold_satisfied and rejected_worse:
        verdict = "best_candidate_for_more_shadow_testing"
    elif threshold_satisfied and consistent is True and rejected_worse:
        verdict = "promising_but_unproven"
    else:
        verdict = "not_approved_for_blocking"
    paper_candidate = bool(
        threshold_satisfied
        and consistent is True
        and rejected_worse
        and pnl_fully_supported
    )
    coverage_denominator = matched + unmatched
    return {
        "number_of_runs": len(run_metrics),
        "runs": run_metrics,
        "total_decision_rows": total_rows,
        "would_confirm_count": confirm_decisions,
        "would_reject_count": reject_decisions,
        "would_reject_rate": None if total_rows <= 0 else reject_decisions / total_rows,
        "reject_reasons": dict(reasons.most_common()),
        "average_allowed_confidence": (
            None
            if confidence_allowed_weight <= 0
            else confidence_allowed_total / confidence_allowed_weight
        ),
        "average_rejected_confidence": (
            None
            if confidence_rejected_weight <= 0
            else confidence_rejected_total / confidence_rejected_weight
        ),
        "confirmed": {
            "matched_count": confirmed_count,
            "pnl_supported_matched_count": confirmed_pnl_supported_count,
            "pnl_coverage_rate": (
                None
                if confirmed_count <= 0
                else confirmed_pnl_supported_count / confirmed_count
            ),
            "total_pnl": confirmed_total if confirmed_pnl_supported_count > 0 else None,
            "weighted_average_pnl": confirmed_average,
            "win_rate": (
                confirmed_winners / confirmed_count
                if confirmed_count > 0 and confirmed_win_supported == confirmed_count
                else None
            ),
        },
        "rejected": {
            "matched_count": rejected_count,
            "pnl_supported_matched_count": rejected_pnl_supported_count,
            "pnl_coverage_rate": (
                None
                if rejected_count <= 0
                else rejected_pnl_supported_count / rejected_count
            ),
            "total_pnl": rejected_total if rejected_pnl_supported_count > 0 else None,
            "weighted_average_pnl": rejected_average,
            "win_rate": (
                rejected_winners / rejected_count
                if rejected_count > 0 and rejected_win_supported == rejected_count
                else None
            ),
        },
        "matched_closed_trade_count": matched,
        "unmatched_decision_count": unmatched,
        "match_coverage_rate": (
            None if coverage_denominator <= 0 else matched / coverage_denominator
        ),
        "outcome_eligible_decision_count": coverage_denominator,
        "pnl_separation": (
            None
            if confirmed_average is None or rejected_average is None
            else confirmed_average - rejected_average
        ),
        "rejected_trades_worse_than_confirmed": rejected_worse,
        "comparable_run_count": len(comparable),
        "outcome_bearing_run_count": len(outcome_bearing_runs),
        "relationship_consistent_across_runs": consistent,
        "pnl_fully_supported": pnl_fully_supported,
        "pnl_support_warning": not pnl_fully_supported and (confirmed_count + rejected_count > 0),
        "sample_size_warning": not threshold_satisfied or consistent is not True,
        "unmatched_data_warning": unmatched > 0,
        "minimum_evidence_threshold": {
            "configured_minimum_matched_per_group": min_matched_per_group,
            "configured_per_group": min_matched_per_group,
            "confirmed_matched_count": confirmed_count,
            "rejected_matched_count": rejected_count,
            "satisfied": threshold_satisfied,
            "profitability_guarantee": False,
            "meaning": (
                "This is an evidence threshold for considering a paper blocking test, "
                "not a guarantee of profitability."
            ),
        },
        "minimum_evidence_threshold_status": (
            "satisfied" if threshold_satisfied else "not_satisfied"
        ),
        "verdict": verdict,
        "blocking_approval": (
            "paper-blocking-candidate" if paper_candidate else "not_approved_for_blocking"
        ),
        "blocking_candidate_status": (
            "paper-blocking-candidate" if paper_candidate else "not_approved_for_blocking"
        ),
        "live_approval": "never_live_approved_by_this_analysis",
    }


def _xgboost_analysis(
    runs: List[Dict[str, Any]],
    min_matched_per_group: int,
) -> Dict[str, Any]:
    by_mode_metrics: Dict[str, List[Dict[str, Any]]] = {
        "xgboost_shadow_outcome": [],
        "combined_shadow": [],
    }
    for run in runs:
        if run["mode"] in by_mode_metrics and isinstance(run.get("audit"), dict) and run["audit"]:
            by_mode_metrics[run["mode"]].append(_xgb_run_metrics(run))
    for items in by_mode_metrics.values():
        items.sort(key=lambda item: item["timestamp"])
    dedicated = _aggregate_xgb_runs(
        by_mode_metrics["xgboost_shadow_outcome"],
        min_matched_per_group,
    )
    combined_mode = _aggregate_xgb_runs(
        by_mode_metrics["combined_shadow"],
        min_matched_per_group,
    )
    all_metrics = (
        by_mode_metrics["xgboost_shadow_outcome"]
        + by_mode_metrics["combined_shadow"]
    )
    all_metrics.sort(key=lambda item: (item["timestamp"], item["source_mode"]))
    aggregate = _aggregate_xgb_runs(all_metrics, min_matched_per_group)
    return {
        "by_source_mode": {
            "xgboost_shadow_outcome": dedicated,
            "combined_shadow": combined_mode,
        },
        "combined_aggregate": aggregate,
        "recommendation": (
            "Continue XGBoost shadow outcome collection until both matched groups meet "
            "the configured evidence threshold with consistent separation."
        ),
        "allowed_mode": (
            "paper-blocking-candidate"
            if aggregate["blocking_approval"] == "paper-blocking-candidate"
            else "shadow_outcome_only"
        ),
        "prohibited_mode": "live_blocking",
    }


def _survival_run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    section = _module_section(run, "survival_exit")
    return {
        "mode": run["mode"],
        "timestamp": run["timestamp"],
        "identity": run["identity"],
        "source_report": run["path"],
        "total_rows": _as_int(section.get("total_rows")),
        "would_exit_early_count": _as_int(section.get("would_exit_early_count")),
        "would_exit_rate": _as_rate(section.get("would_exit_rate")),
        "actually_exited_count": _as_int(section.get("actually_exited_count")),
        "actual_exit_rate": _as_rate(section.get("actual_exit_rate")),
        "average_risk_score": _first_number(
            section.get("average_risk_score"),
            section.get("average_survival_risk_score"),
        ),
    }


def _survival_exit_analysis(
    runs: List[Dict[str, Any]],
    log: Dict[str, Any],
    targets: Sequence[float],
) -> Dict[str, Any]:
    dedicated_shadow_runs = [
        _survival_run_metrics(run)
        for run in runs
        if run["mode"] == "survival_shadow"
    ]
    active_runs = [
        _survival_run_metrics(run)
        for run in runs
        if run["mode"] == "survival_active"
    ]
    combined_runs = [
        _survival_run_metrics(run)
        for run in runs
        if run["mode"] == "combined_shadow"
    ]
    for items in (dedicated_shadow_runs, active_runs, combined_runs):
        items.sort(key=lambda item: item["timestamp"])
    shadow_runs = sorted(
        dedicated_shadow_runs + combined_runs,
        key=lambda item: (item["timestamp"], item["mode"]),
    )

    rows = log.get("rows") if isinstance(log.get("rows"), list) else []
    scores = _numeric_values(
        row.get("survival_risk_score") or row.get("risk_score")
        for row in rows
    )
    distribution = _score_distribution(scores)
    tie_frequency = _tie_frequency(scores)
    simulations = _threshold_simulations(
        scores,
        targets,
        lower_score_triggers=False,
        minimum_threshold=0.0,
        maximum_threshold=1.0,
    )
    percent_above: Dict[str, Optional[float]] = {}
    percent_above_details: Dict[str, Dict[str, Any]] = {}
    for threshold in (0.90, 0.95, 0.99, 0.999):
        count = sum(score > threshold for score in scores)
        rate = None if not scores else count / len(scores)
        key = _rate_key(threshold)
        percent_above[key] = rate
        percent_above_details[key] = {
            "threshold": threshold,
            "count": count,
            "rate": rate,
            "percentage": None if rate is None else rate * 100.0,
        }

    evidence_sufficient = len(scores) >= 20
    simulation_errors = [
        item["absolute_rate_error"]
        for item in simulations.values()
        if item["absolute_rate_error"] is not None
    ]
    row_granularity_tolerance = max(0.01, 1.0 / len(scores)) if scores else None
    threshold_feasible = bool(
        evidence_sufficient
        and len(simulation_errors) == len(simulations)
        and row_granularity_tolerance is not None
        and all(
            item["absolute_rate_error"] is not None
            and item["absolute_rate_error"] <= row_granularity_tolerance + 1e-12
            and not (
                item["tied_score_limitation"]
                and item["absolute_rate_error"] > 0.01 + 1e-12
            )
            for item in simulations.values()
        )
    )
    p50 = distribution.get("p50")
    above_090 = percent_above.get("0.9")
    above_099 = percent_above.get("0.99")
    saturation = bool(
        evidence_sufficient
        and (
            (p50 is not None and p50 >= 0.99)
            or (
                above_090 is not None
                and above_099 is not None
                and above_090 >= 0.90
                and above_099 >= 0.50
            )
        )
    )
    report_rates = [
        item["would_exit_rate"]
        for item in shadow_runs
        if item["would_exit_rate"] is not None and item["total_rows"] > 0
    ]
    extreme_report_behavior = bool(report_rates and all(rate >= 0.80 for rate in report_rates))
    probability_calibration = saturation
    retraining = bool(
        evidence_sufficient
        and saturation
        and not threshold_feasible
    )
    if retraining:
        verdict = "retraining_recommended"
    elif probability_calibration:
        verdict = "probability_calibration_required"
    elif extreme_report_behavior:
        verdict = "too_aggressive"
    else:
        verdict = "not_approved_active"
    warnings: List[str] = []
    if not evidence_sufficient:
        warnings.append("Current row-level Survival evidence is insufficient for stable calibration.")
    if any(item["tied_score_limitation"] for item in simulations.values()):
        warnings.append("Tied risk scores make one or more requested exit rates impossible.")
    if saturation:
        warnings.append("Survival risk scores are concentrated near 1.0.")
    if threshold_feasible and saturation:
        warnings.append(
            "Requested rates are numerically reachable only at extreme thresholds; "
            "this does not establish calibrated probabilities."
        )

    row_level_analysis = {
        "source": {
            "path": log.get("path"),
            "status": log.get("status"),
            "row_count": len(rows),
        },
        "score_distribution": {
            **distribution,
            "count": len(scores),
            "unique_score_values": tie_frequency["unique_score_values"],
            "tied_score_frequency": tie_frequency["tied_row_rate"],
        },
        "tie_frequency": tie_frequency,
        "threshold_simulations": simulations,
        "percent_above_thresholds": percent_above,
        "percent_above_threshold_details": percent_above_details,
    }
    return {
        "number_of_shadow_runs": len(shadow_runs),
        "number_of_dedicated_shadow_runs": len(dedicated_shadow_runs),
        "number_of_combined_shadow_runs": len(combined_runs),
        "number_of_active_validation_runs": len(active_runs),
        "shadow_runs": {
            "number_of_runs": len(shadow_runs),
            "runs": shadow_runs,
        },
        "dedicated_shadow_runs": {
            "number_of_runs": len(dedicated_shadow_runs),
            "runs": dedicated_shadow_runs,
        },
        "active_validation_runs": {
            "number_of_runs": len(active_runs),
            "runs": active_runs,
        },
        "combined_shadow_runs": combined_runs,
        "would_exit_rate_by_run": {
            item["identity"]: item["would_exit_rate"] for item in shadow_runs
        },
        "actual_exit_rate_by_active_validation_run": {
            item["identity"]: item["actual_exit_rate"] for item in active_runs
        },
        "average_risk_score_by_run": {
            item["identity"]: item["average_risk_score"]
            for item in shadow_runs + active_runs
        },
        "score_concentration_near_one": {
            "detected": saturation,
            "median_score": p50,
            "shadow_runs_at_or_above_0_95_average": sum(
                item["average_risk_score"] is not None
                and item["average_risk_score"] >= 0.95
                for item in shadow_runs
            ),
        },
        "aggressiveness_status": (
            "too_aggressive" if extreme_report_behavior else "not_established"
        ),
        "row_level_analysis": row_level_analysis,
        "score_distribution": distribution,
        "threshold_simulations": simulations,
        "percent_above_thresholds": percent_above_details,
        "score_saturation_detected": saturation,
        "threshold_only_calibration_feasible": threshold_feasible,
        "threshold_feasibility_tolerance": row_granularity_tolerance,
        "probability_calibration_recommended": probability_calibration,
        "retraining_recommended": retraining,
        "row_level_evidence_sufficient": evidence_sufficient,
        "warnings": warnings,
        "verdict": verdict,
        "recommendation": {
            "keep_survival_exit_active_false": True,
            "next_action": "evaluate probability calibration or retraining offline",
        },
        "allowed_mode": "shadow_only",
        "prohibited_mode": "SURVIVAL_EXIT_ACTIVE",
    }


def _advanced_run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    section = _module_section(run, "advanced_risk")
    total = _as_int(section.get("total_rows"))
    return {
        "mode": run["mode"],
        "timestamp": run["timestamp"],
        "identity": run["identity"],
        "source_report": run["path"],
        "total_rows": total,
        "would_block_count": _as_int(section.get("would_block_count")),
        "would_block_rate": _as_rate(section.get("would_block_rate")),
        "would_pause_count": _as_int(section.get("would_pause_count")),
        "would_reduce_size_count": _as_int(section.get("would_reduce_size_count")),
        "top_reasons": _sum_counts(section.get("top_reasons")),
    }


def _reason_from_row(row: Dict[str, str]) -> str:
    for key in ("top_reason", "reason", "reasons"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.split("|", 1)[0].strip()
    return "unknown"


def _advanced_risk_analysis(
    runs: List[Dict[str, Any]],
    log: Dict[str, Any],
) -> Dict[str, Any]:
    dedicated = [
        _advanced_run_metrics(run)
        for run in runs
        if run["mode"] == "advanced_risk_shadow"
    ]
    combined = [
        _advanced_run_metrics(run)
        for run in runs
        if run["mode"] == "combined_shadow"
    ]
    dedicated.sort(key=lambda item: item["timestamp"])
    combined.sort(key=lambda item: item["timestamp"])
    report_reasons: Counter[str] = Counter()
    report_total_rows = 0
    for item in dedicated:
        report_reasons.update(item["top_reasons"])
        report_total_rows += item["total_rows"]

    report_reason_details = {
        reason: {
            "count": count,
            "share": None if report_total_rows <= 0 else count / report_total_rows,
            "percentage": None if report_total_rows <= 0 else count / report_total_rows * 100.0,
        }
        for reason, count in report_reasons.most_common()
    }

    rows = log.get("rows") if isinstance(log.get("rows"), list) else []
    row_reasons: Counter[str] = Counter(_reason_from_row(row) for row in rows)
    total_rows = len(rows)
    block_rows = [row for row in rows if _truthy(row.get("would_block"))]
    block_count = len(block_rows)

    def counterfactual_rate(excluded: set[str]) -> Optional[float]:
        if total_rows <= 0:
            return None
        remaining = sum(
            1
            for row in block_rows
            if _reason_from_row(row) not in excluded
        )
        return remaining / total_rows

    exclude_each = {
        reason: counterfactual_rate({reason})
        for reason in row_reasons
        if reason not in {"normal", "unknown", ""}
    }
    counterfactual = {
        "approximate": True,
        "method": "top_reason_only",
        "limitation": (
            "Approximate top-reason counterfactual only. A row records one top reason here, "
            "so this subtraction does not capture rule interactions or secondary reasons."
        ),
        "original_would_block_count": block_count,
        "original_would_block_rate": None if total_rows <= 0 else block_count / total_rows,
        "exclude_max_open_positions_limit": counterfactual_rate({"max_open_positions_limit"}),
        "exclude_consecutive_losses_limit": counterfactual_rate({"consecutive_losses_limit"}),
        "exclude_each_reason": exclude_each,
        "exclude_max_open_positions_and_consecutive_losses": counterfactual_rate(
            {"max_open_positions_limit", "consecutive_losses_limit"}
        ),
    }

    evidence_reasons = row_reasons if rows else report_reasons
    evidence_total = total_rows if rows else report_total_rows
    max_count = evidence_reasons.get("max_open_positions_limit", 0)
    consecutive_count = evidence_reasons.get("consecutive_losses_limit", 0)
    report_max_share = (
        report_reasons.get("max_open_positions_limit", 0) / report_total_rows
        if report_total_rows
        else 0.0
    )
    report_consecutive_share = (
        report_reasons.get("consecutive_losses_limit", 0) / report_total_rows
        if report_total_rows
        else 0.0
    )
    row_max_share = max_count / evidence_total if evidence_total else 0.0
    row_consecutive_share = consecutive_count / evidence_total if evidence_total else 0.0
    max_dominates = max(row_max_share, report_max_share) >= 0.50
    consecutive_dominates = max(row_consecutive_share, report_consecutive_share) >= 0.50
    multiple_strict = bool(
        evidence_total
        and max_count > 0
        and consecutive_count > 0
        and (max_count + consecutive_count) / evidence_total >= 0.50
    )
    observed_report_rates = [
        item["would_block_rate"]
        for item in dedicated + combined
        if item["would_block_rate"] is not None
    ]
    needs_calibration = bool(
        max_dominates
        or consecutive_dominates
        or multiple_strict
        or any(rate >= 0.30 for rate in observed_report_rates)
    )
    return {
        "number_of_dedicated_shadow_runs": len(dedicated),
        "dedicated_shadow_runs": dedicated,
        "combined_shadow_runs": combined,
        "would_block_rate_by_run": {
            item["identity"]: item["would_block_rate"] for item in dedicated
        },
        "would_pause_count_by_run": {
            item["identity"]: item["would_pause_count"] for item in dedicated
        },
        "would_reduce_size_count_by_run": {
            item["identity"]: item["would_reduce_size_count"] for item in dedicated
        },
        "top_reasons_aggregated": report_reason_details,
        "top_reasons": dict(report_reasons.most_common()),
        "reason_share_percentages": {
            reason: details["percentage"]
            for reason, details in report_reason_details.items()
        },
        "latest_run_metrics": dedicated[-1] if dedicated else None,
        "combined_shadow_metrics": combined,
        "row_level_analysis": {
            "source": {
                "path": log.get("path"),
                "status": log.get("status"),
                "row_count": total_rows,
            },
            "top_reasons": dict(row_reasons.most_common()),
            "reason_share_percentages": {
                reason: (count / total_rows * 100.0 if total_rows else None)
                for reason, count in row_reasons.most_common()
            },
            "approximate_top_reason_counterfactual": counterfactual,
        },
        "approximate_top_reason_counterfactual": counterfactual,
        "max_open_positions_dominates": max_dominates,
        "consecutive_losses_dominates": consecutive_dominates,
        "dominance_evidence": {
            "report_max_open_positions_share": report_max_share,
            "report_consecutive_losses_share": report_consecutive_share,
            "current_log_max_open_positions_share": row_max_share if rows else None,
            "current_log_consecutive_losses_share": row_consecutive_share if rows else None,
        },
        "multiple_rules_too_strict": multiple_strict,
        "needs_rule_calibration": needs_calibration,
        "recommendation": {
            "keep_advanced_risk_active_false": True,
            "review_maximum_open_positions": True,
            "review_consecutive_loss_reset_and_cooldown": True,
            "preserve_daily_loss_protection_as_candidate": True,
            "activate_daily_loss_without_paper_evidence": False,
        },
        "allowed_mode": "shadow_only",
        "prohibited_mode": "ADVANCED_RISK_ACTIVE",
    }


def _module_present(section: Dict[str, Any]) -> bool:
    status = str(section.get("file_status") or "").strip().lower()
    return status in {"ok", "present", "loaded"} or _as_int(section.get("total_rows")) > 0


def _combined_xgb_coverage(run: Dict[str, Any]) -> Dict[str, int]:
    audit = run.get("audit")
    if isinstance(audit, dict) and audit:
        join = audit.get("trade_outcome_join")
        join = join if isinstance(join, dict) else {}
        joined_matched = _as_number(join.get("matched_closed_trade_count"))
        matched = (
            int(joined_matched)
            if joined_matched is not None
            else (
                _as_int(audit.get("would_confirm_matched_count"))
                + _as_int(audit.get("would_reject_matched_count"))
            )
        )
        unmatched = _first_int(join.get("unmatched_decision_rows"))
        return {"matched": matched, "unmatched": unmatched}
    payload = run.get("data")
    outcome = payload.get("xgboost_outcome") if isinstance(payload, dict) else {}
    outcome = outcome if isinstance(outcome, dict) else {}
    matched = _first_int(
        outcome.get("matched_closed_trade_count"),
        _as_int(outcome.get("would_confirm_matched_count"))
        + _as_int(outcome.get("would_reject_matched_count")),
    )
    unmatched = _first_int(
        outcome.get("unmatched_decision_rows"),
        outcome.get("unmatched_xgboost_rows"),
    )
    return {"matched": matched, "unmatched": unmatched}


def _combined_shadow_integration(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined_runs: List[Dict[str, Any]] = []
    aggregate_actuals = dict(ZERO_ACTUAL_COUNTS)
    paper_rows = 0
    paper_with_id = 0
    closed_rows = 0
    closed_with_id = 0
    xgb_matched = 0
    xgb_unmatched = 0
    for run in runs:
        if run["mode"] != "combined_shadow":
            continue
        isolation = _module_section(run, "isolation_forest")
        xgboost = _module_section(run, "xgboost", "xgboost_signal")
        survival = _module_section(run, "survival_exit")
        advanced = _module_section(run, "advanced_risk")
        presence = {
            "isolation_forest": _module_present(isolation),
            "xgboost": _module_present(xgboost),
            "survival_exit": _module_present(survival),
            "advanced_risk": _module_present(advanced),
        }
        actual = _actual_counts(run)
        _add_actual_counts(aggregate_actuals, actual)
        payload = run.get("data")
        lineage = payload.get("trade_lineage") if isinstance(payload, dict) else {}
        lineage = lineage if isinstance(lineage, dict) else {}
        run_paper_rows = _as_int(lineage.get("paper_trade_rows"))
        run_paper_ids = _as_int(lineage.get("paper_trade_rows_with_signal_id"))
        run_closed_rows = _as_int(lineage.get("closed_trade_rows"))
        run_closed_ids = _as_int(lineage.get("closed_trade_rows_with_signal_id"))
        paper_rows += run_paper_rows
        paper_with_id += run_paper_ids
        closed_rows += run_closed_rows
        closed_with_id += run_closed_ids
        xgb_coverage = _combined_xgb_coverage(run)
        xgb_matched += xgb_coverage["matched"]
        xgb_unmatched += xgb_coverage["unmatched"]
        pnl = payload.get("paper_pnl") if isinstance(payload, dict) else {}
        pnl = pnl if isinstance(pnl, dict) else {}
        combined_runs.append(
            {
                "mode": run["mode"],
                "timestamp": run["timestamp"],
                "identity": run["identity"],
                "source_report": run["path"],
                "module_presence": presence,
                "all_modules_present": all(presence.values()),
                "actual_behavior_counts": actual,
                "signal_id_coverage": {
                    "paper_trades": _coverage(run_paper_rows, run_paper_ids),
                    "closed_trades": _coverage(run_closed_rows, run_closed_ids),
                },
                "xgboost_matched_outcome_coverage": {
                    "matched_count": xgb_coverage["matched"],
                    "unmatched_decision_count": xgb_coverage["unmatched"],
                    "coverage_rate": (
                        None
                        if xgb_coverage["matched"] + xgb_coverage["unmatched"] <= 0
                        else xgb_coverage["matched"]
                        / (xgb_coverage["matched"] + xgb_coverage["unmatched"])
                    ),
                },
                "baseline_strategy_context": {
                    "closed_trade_count": _as_int(pnl.get("closed_trade_count")),
                    "total_pnl": _first_number(pnl.get("total_pnl")),
                    "average_pnl": _first_number(pnl.get("average_pnl")),
                    "win_rate": _as_rate(pnl.get("win_rate")),
                },
            }
        )
    combined_runs.sort(key=lambda item: item["timestamp"])
    any_actual = any(aggregate_actuals.values())
    all_present = bool(combined_runs) and all(item["all_modules_present"] for item in combined_runs)
    signal_coverage = {
        "paper_trades": _coverage(paper_rows, paper_with_id),
        "closed_trades": _coverage(closed_rows, closed_with_id),
    }
    lineage_complete = all(
        item["coverage_rate"] in {None, 1.0}
        for item in signal_coverage.values()
    )
    if any_actual:
        verdict = "integration_failed"
    elif not combined_runs or not all_present or not lineage_complete:
        verdict = "integration_incomplete"
    else:
        verdict = "integration_passed"
    xgb_denominator = xgb_matched + xgb_unmatched
    return {
        "number_of_runs": len(combined_runs),
        "runs": combined_runs,
        "all_modules_present_in_each_run": all_present,
        "actual_behavior_counts": aggregate_actuals,
        "any_actual_blocking_rejection_exit_pause_or_size_reduction": any_actual,
        "signal_id_coverage": signal_coverage,
        "aggregate_signal_id_coverage": signal_coverage,
        "xgboost_matched_outcome_coverage": {
            "matched_count": xgb_matched,
            "unmatched_decision_count": xgb_unmatched,
            "coverage_rate": (
                None if xgb_denominator <= 0 else xgb_matched / xgb_denominator
            ),
        },
        "run_level_pnl_is_baseline_context_only": True,
        "verdict": verdict,
        "calibration_note": (
            "Combined shadow validates integration and safety isolation; "
            "it does not prove profitability."
        ),
        "allowed_mode": "paper_shadow_integration_only",
        "prohibited_mode": "active_or_live_combined_mode",
    }


def _evidence_matrix(
    baseline: Dict[str, Any],
    isolation: Dict[str, Any],
    xgboost: Dict[str, Any],
    survival: Dict[str, Any],
    advanced: Dict[str, Any],
    combined: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    xgb_aggregate = xgboost["combined_aggregate"]
    return {
        "baseline": {
            "technical_status": baseline["verdict"],
            "evidence_strength": "limited" if baseline["sample_size_warning"] else "moderate_paper_only",
            "calibration_status": "more_data_required",
            "allowed_mode": "paper-only",
            "prohibited_mode": "live-or-real-orders",
            "next_action": "continue baseline paper runs and aggregate more closed trades",
        },
        "isolation_forest": {
            "technical_status": isolation["verdict"],
            "evidence_strength": (
                "strong_adverse_shadow_evidence"
                if isolation["consistent_100_percent_anomaly_behavior"]
                else "limited"
            ),
            "calibration_status": (
                "retraining_recommended"
                if isolation["retraining_recommended"]
                else "more_shadow_data_required"
            ),
            "allowed_mode": "shadow-only",
            "prohibited_mode": "blocking",
            "next_action": "evaluate retraining; do not alter or enable the current artifact",
        },
        "xgboost": {
            "technical_status": xgb_aggregate["verdict"],
            "evidence_strength": (
                "threshold_met_paper_evidence"
                if xgb_aggregate["minimum_evidence_threshold"]["satisfied"]
                else "limited_matched_outcomes"
            ),
            "calibration_status": xgb_aggregate["minimum_evidence_threshold_status"],
            "allowed_mode": "shadow-outcome-only",
            "prohibited_mode": "live-blocking",
            "next_action": "aggregate more matched confirmed and rejected closed trades",
        },
        "survival_exit": {
            "technical_status": survival["verdict"],
            "evidence_strength": (
                "strong_adverse_shadow_evidence"
                if survival["score_saturation_detected"]
                else "limited"
            ),
            "calibration_status": (
                "probability_calibration_or_retraining_required"
                if survival["probability_calibration_recommended"]
                else "more_shadow_data_required"
            ),
            "allowed_mode": "shadow-only",
            "prohibited_mode": "active-exits",
            "next_action": "evaluate probability calibration or retraining offline",
        },
        "advanced_risk": {
            "technical_status": (
                "rules_too_strict" if advanced["needs_rule_calibration"] else "unproven"
            ),
            "evidence_strength": "moderate_shadow_rule_evidence",
            "calibration_status": "offline_rule_calibration_required",
            "allowed_mode": "shadow-only",
            "prohibited_mode": "active-risk-blocking",
            "next_action": "review max-open and consecutive-loss reset/cooldown rules",
        },
        "combined_shadow": {
            "technical_status": combined["verdict"],
            "evidence_strength": (
                "integration_evidence_only"
                if combined["number_of_runs"]
                else "missing"
            ),
            "calibration_status": "safety_isolation_only",
            "allowed_mode": "paper-shadow-integration-only",
            "prohibited_mode": "active-or-live",
            "next_action": "continue paper shadow integration after offline proposals",
        },
    }


def _final_recommendation(min_matched_per_group: int) -> Dict[str, Any]:
    return {
        "priority_order": [
            "Continue baseline and XGBoost shadow outcome data collection.",
            (
                "Reach the configured XGBoost evidence threshold of at least "
                f"{min_matched_per_group} matched trades in each outcome group."
            ),
            (
                "Do not activate Isolation Forest; evaluate retraining because "
                "threshold-only calibration is ineffective."
            ),
            "Do not activate Survival Exit; evaluate probability calibration or retraining.",
            "Calibrate Advanced Risk rules offline.",
            "Keep combined mode shadow-only.",
            (
                "Do not use live, mainnet, testnet real orders, real orders, "
                "or PLACE_REAL_ORDERS."
            ),
            "After offline changes are proposed, run a fresh paper-only 60-minute shadow validation.",
            (
                "Only consider active/blocking paper tests after the new shadow evidence passes; "
                "this analysis never grants live approval."
            ),
        ],
        "safety_constraints": {
            "paper_only": True,
            "no_mainnet": True,
            "no_testnet_real_orders": True,
            "no_real_orders": True,
            "place_real_orders_must_remain_false": True,
            "no_active_or_blocking_modules": True,
        },
        "final_verdict": "paper_only_offline_calibration_required",
    }


def summarize_offline_calibration(
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    min_xgb_matched_per_group: int = 30,
    target_if_block_rates: Sequence[float] = (0.01, 0.05, 0.10, 0.15),
    target_survival_exit_rates: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> Dict[str, Any]:
    """Build the complete read-only Phase 17 report."""

    reports_path = Path(reports_dir)
    logs_path = Path(logs_dir)
    if min_xgb_matched_per_group <= 0:
        raise ValueError("min_xgb_matched_per_group must be positive")
    if not target_if_block_rates or not target_survival_exit_rates:
        raise ValueError("at least one calibration target rate is required")
    for name, values in (
        ("target_if_block_rates", target_if_block_rates),
        ("target_survival_exit_rates", target_survival_exit_rates),
    ):
        if any(not 0.0 < float(value) < 1.0 for value in values):
            raise ValueError(f"{name} values must be between 0 and 1")

    inventory, runs, logs = _load_inputs(reports_path, logs_path)
    baseline = _baseline_summary(runs)
    isolation = _isolation_forest_analysis(
        runs,
        logs["isolation_forest"],
        tuple(float(value) for value in target_if_block_rates),
    )
    xgboost = _xgboost_analysis(runs, min_xgb_matched_per_group)
    survival = _survival_exit_analysis(
        runs,
        logs["survival_exit"],
        tuple(float(value) for value in target_survival_exit_rates),
    )
    advanced = _advanced_risk_analysis(runs, logs["advanced_risk"])
    combined = _combined_shadow_integration(runs)
    return {
        "input_inventory": inventory,
        "run_classification": _run_classification(runs),
        "baseline_cross_run_summary": baseline,
        "isolation_forest_analysis": isolation,
        "xgboost_outcome_aggregation": xgboost,
        "survival_exit_analysis": survival,
        "advanced_risk_analysis": advanced,
        "combined_shadow_integration": combined,
        "evidence_matrix": _evidence_matrix(
            baseline,
            isolation,
            xgboost,
            survival,
            advanced,
            combined,
        ),
        "final_recommendation": _final_recommendation(min_xgb_matched_per_group),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _fmt_threshold(value: Any) -> str:
    number = _as_number(value)
    return "n/a" if number is None else format(number, ".17g")


def _format_thresholds(simulations: Dict[str, Dict[str, Any]], label: str) -> List[str]:
    lines: List[str] = []
    for key, item in simulations.items():
        below = item.get("nearest_achievable_below") or {}
        above = item.get("nearest_achievable_above") or {}
        lines.append(
            "  "
            f"target={key} {label}: threshold={_fmt_threshold(item.get('threshold'))} "
            f"achieved={_fmt(item.get('achieved_rate'))} "
            f"tied_limit={_fmt(item.get('tied_score_limitation'))} "
            f"below=({_fmt_threshold(below.get('threshold'))}, {_fmt(below.get('achieved_rate'))}) "
            f"above=({_fmt_threshold(above.get('threshold'))}, {_fmt(above.get('achieved_rate'))})"
        )
    return lines


def _format_xgb_aggregate(label: str, aggregate: Dict[str, Any]) -> List[str]:
    confirmed = aggregate["confirmed"]
    rejected = aggregate["rejected"]
    threshold = aggregate["minimum_evidence_threshold"]
    return [
        (
            f"  {label}: runs={aggregate['number_of_runs']} rows={aggregate['total_decision_rows']} "
            f"confirm={aggregate['would_confirm_count']} reject={aggregate['would_reject_count']} "
            f"reject_rate={_fmt(aggregate['would_reject_rate'])}"
        ),
        (
            f"    confidence_allowed={_fmt(aggregate['average_allowed_confidence'])} "
            f"confidence_rejected={_fmt(aggregate['average_rejected_confidence'])} "
            f"reasons={_fmt(aggregate['reject_reasons'])}"
        ),
        (
            f"    confirmed: matched={confirmed['matched_count']} total_pnl={_fmt(confirmed['total_pnl'])} "
            f"weighted_avg={_fmt(confirmed['weighted_average_pnl'])} "
            f"win_rate={_fmt(confirmed['win_rate'])} "
            f"pnl_coverage={_fmt(confirmed['pnl_coverage_rate'])}"
        ),
        (
            f"    rejected: matched={rejected['matched_count']} total_pnl={_fmt(rejected['total_pnl'])} "
            f"weighted_avg={_fmt(rejected['weighted_average_pnl'])} "
            f"win_rate={_fmt(rejected['win_rate'])} "
            f"pnl_coverage={_fmt(rejected['pnl_coverage_rate'])}"
        ),
        (
            f"    matched={aggregate['matched_closed_trade_count']} "
            f"unmatched={aggregate['unmatched_decision_count']} "
            f"coverage={_fmt(aggregate['match_coverage_rate'])} "
            f"pnl_separation={_fmt(aggregate['pnl_separation'])}"
        ),
        (
            f"    rejected_worse={_fmt(aggregate['rejected_trades_worse_than_confirmed'])} "
            f"consistent={_fmt(aggregate['relationship_consistent_across_runs'])} "
            f"evidence={threshold['confirmed_matched_count']}/{threshold['rejected_matched_count']} "
            f"per-group minimum={threshold['configured_minimum_matched_per_group']} "
            f"satisfied={_fmt(threshold['satisfied'])}"
        ),
        (
            f"    sample_warning={_fmt(aggregate['sample_size_warning'])} "
            f"unmatched_warning={_fmt(aggregate['unmatched_data_warning'])} "
            f"pnl_support_warning={_fmt(aggregate['pnl_support_warning'])} "
            f"verdict={aggregate['verdict']} approval={aggregate['blocking_approval']}"
        ),
    ]


def _format_xgb_run(item: Dict[str, Any]) -> List[str]:
    confirmed = item["confirmed"]
    rejected = item["rejected"]
    return [
        (
            f"    run={item['identity']} rows={item['total_decision_rows']} "
            f"confirm={item['would_confirm_count']} reject={item['would_reject_count']} "
            f"reject_rate={_fmt(item['would_reject_rate'])} "
            f"reasons={_fmt(item['reject_reasons'])}"
        ),
        (
            f"      confidence_allowed={_fmt(item['average_allowed_confidence'])} "
            f"confidence_rejected={_fmt(item['average_rejected_confidence'])} "
            f"confirmed_matched={confirmed['matched_count']} "
            f"confirmed_total_pnl={_fmt(confirmed['total_pnl'])} "
            f"confirmed_avg_pnl={_fmt(confirmed['weighted_average_pnl'])} "
            f"confirmed_win_rate={_fmt(confirmed['win_rate'])}"
        ),
        (
            f"      rejected_matched={rejected['matched_count']} "
            f"rejected_total_pnl={_fmt(rejected['total_pnl'])} "
            f"rejected_avg_pnl={_fmt(rejected['weighted_average_pnl'])} "
            f"rejected_win_rate={_fmt(rejected['win_rate'])} "
            f"matched={item['matched_closed_trade_count']} "
            f"unmatched={item['unmatched_decision_count']} "
            f"coverage={_fmt(item['match_coverage_rate'])} "
            f"separation={_fmt(item['pnl_separation'])} "
            f"rejected_worse={_fmt(item['rejected_trades_worse_than_confirmed'])}"
        ),
    ]


def format_text_summary(summary: Dict[str, Any]) -> str:
    """Format all Phase 17 JSON sections for terminal review."""

    inventory = summary["input_inventory"]
    classification = summary["run_classification"]
    baseline = summary["baseline_cross_run_summary"]
    isolation = summary["isolation_forest_analysis"]
    xgboost = summary["xgboost_outcome_aggregation"]
    survival = summary["survival_exit_analysis"]
    advanced = summary["advanced_risk_analysis"]
    combined = summary["combined_shadow_integration"]
    evidence = summary["evidence_matrix"]
    final = summary["final_recommendation"]

    lines = [
        "Phase 17: Offline Calibration Sweep",
        "",
        "A. Input inventory",
        f"  report_files_found: {inventory['report_file_count_found']}",
    ]
    lines.extend(f"    {path}" for path in inventory["report_files_found"])
    lines.append(f"  report_files_used: {inventory['report_file_count_used']}")
    lines.extend(f"    {path}" for path in inventory["report_files_used"])
    lines.extend(
        [
            f"  row_level_logs_found: {_fmt(inventory['row_level_logs_found'])}",
            f"  run_timestamps_by_mode: {_fmt(inventory['run_timestamps_by_mode'])}",
            f"  duplicate_reports_skipped: {inventory['duplicate_report_count']}",
            f"    details: {_fmt(inventory['duplicate_reports_skipped'])}",
            f"  malformed_reports_skipped: {inventory['malformed_report_count']}",
            f"    details: {_fmt(inventory['malformed_reports_skipped'])}",
            f"  incomplete_reports_skipped: {inventory['incomplete_report_count']}",
            f"    details: {_fmt(inventory['incomplete_reports_skipped'])}",
            f"  unverified_reports_skipped: {_fmt(inventory['unverified_reports_skipped'])}",
            f"  missing_inputs: {_fmt(inventory['missing_inputs'])}",
            f"  read_errors: {_fmt(inventory['read_errors'])}",
            f"  archived_csv_policy: {inventory['archived_csv_policy']}",
            f"  trade_csv_policy: {inventory['trade_csv_policy']}",
            "",
            "B. Run classification",
            f"  shadow-safe modes: {', '.join(classification['shadow_safe_modes'])}",
            (
                "  intentional active validation modes: "
                + ", ".join(classification["intentional_active_validation_modes"])
            ),
            f"  shadow safety violations: {_fmt(classification['shadow_safety_failures'])}",
            f"  shadow-only safety passed: {_fmt(classification['shadow_only_safety_passed'])}",
            f"  note: {classification['active_validation_note']}",
            "",
            "C. Cross-run baseline summary",
            (
                f"  runs={baseline['number_of_runs']} closed_trades={baseline['total_closed_trades']} "
                f"total_pnl={_fmt(baseline['total_pnl'])} "
                f"weighted_average_pnl={_fmt(baseline['weighted_average_pnl'])}"
            ),
            (
                f"  overall_win_rate={_fmt(baseline['overall_win_rate'])} "
                f"winner_counts_reconstructed={_fmt(baseline['raw_winner_counts_reconstructed'])}"
            ),
            (
                f"  min_run_pnl={_fmt(baseline['min_run_pnl'])} "
                f"max_run_pnl={_fmt(baseline['max_run_pnl'])} "
                f"positive={baseline['positive_run_count']} negative={baseline['negative_run_count']}"
            ),
            f"  latest_run={_fmt(baseline['latest_run'])}",
            f"  sample_size_warning={_fmt(baseline['sample_size_warning'])}: {baseline['sample_size_message']}",
            f"  verdict: {baseline['verdict']} (profitability claim: false)",
            "",
            "D. Isolation Forest cross-run analysis",
            (
                f"  shadow_runs={isolation['number_of_shadow_runs']} "
                f"blocking_validation_runs={isolation['number_of_blocking_validation_runs']}"
            ),
            f"  would_block_rate_by_run={_fmt(isolation['would_block_rate_by_run'])}",
            (
                "  actual_block_rate_by_blocking_run="
                f"{_fmt(isolation['actual_block_rate_by_blocking_run'])}"
            ),
            (
                "  anomaly_score_statistics_by_run="
                f"{_fmt(isolation['anomaly_score_statistics_by_run'])}"
            ),
            (
                "  consistent_100_percent_anomaly_behavior="
                f"{_fmt(isolation['consistent_100_percent_anomaly_behavior'])}"
            ),
            f"  row_score_distribution={_fmt(isolation['row_level_analysis']['score_distribution'])}",
            f"  tie_frequency={_fmt(isolation['tie_frequency'])}",
        ]
    )
    lines.extend(_format_thresholds(isolation["threshold_simulations"], "block_rate"))
    lines.extend(
        [
            (
                "  threshold_only_calibration_feasible="
                f"{_fmt(isolation['threshold_only_calibration_feasible'])} "
                f"score_saturation_detected={_fmt(isolation['score_saturation_detected'])} "
                f"retraining_recommended={_fmt(isolation['retraining_recommended'])}"
            ),
            f"  warnings={_fmt(isolation['warnings'])}",
            f"  verdict: {isolation['verdict']}",
            "",
            "E. XGBoost multi-run outcome aggregation",
        ]
    )
    dedicated_xgb = xgboost["by_source_mode"]["xgboost_shadow_outcome"]
    combined_xgb = xgboost["by_source_mode"]["combined_shadow"]
    lines.append("  dedicated xgboost_shadow_outcome per-run metrics:")
    for item in dedicated_xgb["runs"]:
        lines.extend(_format_xgb_run(item))
    lines.extend(_format_xgb_aggregate("dedicated xgboost_shadow_outcome", dedicated_xgb))
    lines.append("  combined_shadow XGBoost per-run metrics:")
    for item in combined_xgb["runs"]:
        lines.extend(_format_xgb_run(item))
    lines.extend(_format_xgb_aggregate("combined_shadow", combined_xgb))
    lines.extend(_format_xgb_aggregate("combined aggregate", xgboost["combined_aggregate"]))
    lines.extend(
        [
            (
                "  Evidence threshold is a review threshold, not a guarantee of profitability. "
                "Even a passing result is paper-blocking-candidate only."
            ),
            "",
            "F. Survival Exit cross-run analysis",
            (
                f"  shadow_runs={survival['number_of_shadow_runs']} "
                f"active_validation_runs={survival['number_of_active_validation_runs']}"
            ),
            f"  would_exit_rate_by_run={_fmt(survival['would_exit_rate_by_run'])}",
            (
                "  actual_exit_rate_by_active_validation_run="
                f"{_fmt(survival['actual_exit_rate_by_active_validation_run'])}"
            ),
            f"  average_risk_score_by_run={_fmt(survival['average_risk_score_by_run'])}",
            f"  row_score_distribution={_fmt(survival['row_level_analysis']['score_distribution'])}",
            f"  percent_above_thresholds={_fmt(survival['percent_above_thresholds'])}",
        ]
    )
    lines.extend(_format_thresholds(survival["threshold_simulations"], "exit_rate"))
    lines.extend(
        [
            (
                f"  score_saturation_detected={_fmt(survival['score_saturation_detected'])} "
                f"threshold_only_calibration_feasible={_fmt(survival['threshold_only_calibration_feasible'])}"
            ),
            (
                "  probability_calibration_recommended="
                f"{_fmt(survival['probability_calibration_recommended'])} "
                f"retraining_recommended={_fmt(survival['retraining_recommended'])}"
            ),
            f"  warnings={_fmt(survival['warnings'])}",
            f"  verdict: {survival['verdict']} (SURVIVAL_EXIT_ACTIVE remains false)",
            "",
            "G. Advanced Risk cross-run analysis",
            f"  dedicated_runs={advanced['number_of_dedicated_shadow_runs']}",
            f"  would_block_rate_by_run={_fmt(advanced['would_block_rate_by_run'])}",
            f"  would_pause_count_by_run={_fmt(advanced['would_pause_count_by_run'])}",
            (
                "  would_reduce_size_count_by_run="
                f"{_fmt(advanced['would_reduce_size_count_by_run'])}"
            ),
            f"  top_reasons_aggregated={_fmt(advanced['top_reasons_aggregated'])}",
            f"  latest_run_metrics={_fmt(advanced['latest_run_metrics'])}",
            f"  combined_shadow_metrics={_fmt(advanced['combined_shadow_metrics'])}",
            (
                "  approximate_top_reason_counterfactual="
                f"{_fmt(advanced['approximate_top_reason_counterfactual'])}"
            ),
            (
                f"  max_open_positions_dominates={_fmt(advanced['max_open_positions_dominates'])} "
                f"consecutive_losses_dominates={_fmt(advanced['consecutive_losses_dominates'])} "
                f"multiple_rules_too_strict={_fmt(advanced['multiple_rules_too_strict'])}"
            ),
            f"  needs_rule_calibration={_fmt(advanced['needs_rule_calibration'])}",
            f"  recommendation={_fmt(advanced['recommendation'])}",
            "",
            "H. Combined shadow integration across runs",
            f"  number_of_runs={combined['number_of_runs']}",
            f"  all_modules_present_in_each_run={_fmt(combined['all_modules_present_in_each_run'])}",
            f"  actual_behavior_counts={_fmt(combined['actual_behavior_counts'])}",
            f"  signal_id_coverage={_fmt(combined['signal_id_coverage'])}",
            (
                "  xgboost_matched_outcome_coverage="
                f"{_fmt(combined['xgboost_matched_outcome_coverage'])}"
            ),
            "  per-run integration details:",
        ]
    )
    for item in combined["runs"]:
        lines.extend(
            [
                (
                    f"    run={item['identity']} all_modules_present="
                    f"{_fmt(item['all_modules_present'])} "
                    f"module_presence={_fmt(item['module_presence'])}"
                ),
                f"      actual_behavior_counts={_fmt(item['actual_behavior_counts'])}",
                f"      signal_id_coverage={_fmt(item['signal_id_coverage'])}",
                (
                    "      xgboost_matched_outcome_coverage="
                    f"{_fmt(item['xgboost_matched_outcome_coverage'])}"
                ),
                f"      baseline_strategy_context={_fmt(item['baseline_strategy_context'])}",
            ]
        )
    lines.extend(
        [
            f"  verdict: {combined['verdict']}",
            f"  {combined['calibration_note']}",
            "",
            "I. Evidence matrix",
            (
                "  module | technical_status | evidence_strength | calibration_status "
                "| allowed_mode | prohibited_mode | next_action"
            ),
        ]
    )
    for module, item in evidence.items():
        lines.append(
            f"  {module} | {item['technical_status']} | {item['evidence_strength']} "
            f"| {item['calibration_status']} | {item['allowed_mode']} "
            f"| {item['prohibited_mode']} | {item['next_action']}"
        )
    lines.extend(["", "J. Final Phase 17 recommendation"])
    for index, recommendation in enumerate(final["priority_order"], start=1):
        lines.append(f"  {index}. {recommendation}")
    lines.extend(
        [
            f"  safety_constraints={_fmt(final['safety_constraints'])}",
            f"  final_verdict: {final['final_verdict']}",
        ]
    )
    return "\n".join(lines)


def write_json_summary(
    summary: Dict[str, Any],
    out_path: Path | str = DEFAULT_JSON_OUT,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _parse_rate_list(value: str) -> Tuple[float, ...]:
    try:
        rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rates must be comma-separated numbers") from exc
    if not rates:
        raise argparse.ArgumentTypeError("at least one rate is required")
    if any(not 0.0 < rate < 1.0 for rate in rates):
        raise argparse.ArgumentTypeError("rates must be between 0 and 1")
    return rates


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only offline calibration sweep across completed matrix runs."
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument(
        "--min-xgb-matched-per-group",
        type=_positive_int,
        default=30,
        help=(
            "Evidence threshold per matched XGBoost outcome group; "
            "it is not a profitability guarantee."
        ),
    )
    parser.add_argument(
        "--target-if-block-rates",
        type=_parse_rate_list,
        default=(0.01, 0.05, 0.10, 0.15),
    )
    parser.add_argument(
        "--target-survival-exit-rates",
        type=_parse_rate_list,
        default=(0.05, 0.10, 0.20, 0.30),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the JSON analysis report.",
    )
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_offline_calibration(
        reports_dir=args.reports_dir,
        logs_dir=args.logs_dir,
        min_xgb_matched_per_group=args.min_xgb_matched_per_group,
        target_if_block_rates=args.target_if_block_rates,
        target_survival_exit_rates=args.target_survival_exit_rates,
    )
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
