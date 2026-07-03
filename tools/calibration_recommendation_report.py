"""Read-only calibration recommendation report for completed experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "calibration_recommendation_report.json"
FINAL_COMPARISON_NAME = "final_experiment_comparison.json"

REQUIRED_MODES = [
    "baseline",
    "iforest_shadow",
    "xgboost_shadow_outcome",
    "survival_shadow",
    "advanced_risk_shadow",
    "combined_shadow",
]
OPTIONAL_MODES = ["iforest_blocking", "survival_active"]
MODE_ORDER = REQUIRED_MODES + OPTIONAL_MODES

REPORT_RE = re.compile(
    r"^matrix_(?P<mode>.+)_(?P<timestamp>\d{14})_"
    r"(?P<kind>unified|shadow_summary|xgboost_audit)\.json$"
)

SHADOW_LOGS = {
    "isolation_forest": "isolation_forest_shadow.csv",
    "xgboost": "xgboost_signal_shadow.csv",
    "survival_exit": "survival_exit_shadow.csv",
    "advanced_risk": "advanced_risk_shadow.csv",
}
ACTUAL_COUNT_KEYS = [
    "actually_blocked",
    "actually_rejected",
    "actually_exited",
    "actually_paused",
    "actually_reduced",
]


def _read_json(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: JSON root is not an object"
    return payload, None


def _read_csv_rows(path: Path) -> tuple[str, List[Dict[str, str]], Optional[str]]:
    if not path.exists():
        return "missing", [], None
    if path.stat().st_size == 0:
        return "empty", [], None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [
                {str(k): "" if v is None else str(v) for k, v in row.items() if k is not None}
                for row in reader
            ]
        return ("empty" if not rows else "ok"), rows, None
    except Exception as exc:
        return f"read_error:{type(exc).__name__}", [], f"{path}: {type(exc).__name__}: {exc}"


def _as_number(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
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


def _get(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _ordered_modes(modes: Iterable[str]) -> List[str]:
    seen = set(modes)
    ordered = [mode for mode in MODE_ORDER if mode in seen]
    ordered.extend(sorted(mode for mode in seen if mode not in MODE_ORDER))
    return ordered


def _latest_timestamp(timestamps: Iterable[str]) -> Optional[str]:
    values = sorted(str(ts) for ts in timestamps if str(ts).strip())
    return values[-1] if values else None


def _discover_report_files(reports_dir: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    groups: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for pattern in (
        "matrix_*_unified.json",
        "matrix_*_shadow_summary.json",
        "matrix_*_xgboost_audit.json",
    ):
        for path in sorted(reports_dir.glob(pattern)):
            match = REPORT_RE.match(path.name)
            if match is None:
                continue
            groups.setdefault(match.group("mode"), {}).setdefault(match.group("timestamp"), {})[
                match.group("kind")
            ] = path
    return groups


def _load_latest_reports(
    report_groups: Dict[str, Dict[str, Dict[str, Path]]],
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    selected: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for mode, by_timestamp in report_groups.items():
        latest = _latest_timestamp(by_timestamp)
        if latest is None:
            continue
        reports = by_timestamp[latest]
        data: Dict[str, Dict[str, Any]] = {}
        for kind, path in reports.items():
            payload, error = _read_json(path)
            if error is not None:
                errors.append(error)
                continue
            data[kind] = payload
        selected[mode] = {
            "timestamp": latest,
            "paths": {kind: str(path) for kind, path in sorted(reports.items())},
            "data": data,
        }
    return selected, errors


def _module_data(mode_report: Dict[str, Any], unified_key: str, shadow_key: Optional[str] = None) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    unified = data.get("unified") if isinstance(data.get("unified"), dict) else {}
    shadow = data.get("shadow_summary") if isinstance(data.get("shadow_summary"), dict) else {}
    primary = unified.get(unified_key)
    if isinstance(primary, dict):
        return primary
    fallback = shadow.get(shadow_key or unified_key)
    return fallback if isinstance(fallback, dict) else {}


def _unified_data(mode_report: Dict[str, Any]) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    unified = data.get("unified")
    return unified if isinstance(unified, dict) else {}


def _audit_data(mode_report: Dict[str, Any]) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    audit = data.get("xgboost_audit")
    return audit if isinstance(audit, dict) else {}


def _numeric_values(values: Iterable[Any]) -> List[float]:
    return [value for value in (_as_number(item) for item in values) if value is not None]


def _avg(values: Iterable[Any]) -> Optional[float]:
    nums = _numeric_values(values)
    return None if not nums else sum(nums) / len(nums)


def _percentile(sorted_values: List[float], percentile: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    bounded = max(0.0, min(100.0, float(percentile)))
    rank = (bounded / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] + ((sorted_values[upper] - sorted_values[lower]) * weight)


def _distribution(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    nums = sorted(_numeric_values(values))
    return {
        "min": None if not nums else nums[0],
        "p10": _percentile(nums, 10),
        "p50": _percentile(nums, 50),
        "p90": _percentile(nums, 90),
        "max": None if not nums else nums[-1],
        "average": None if not nums else sum(nums) / len(nums),
    }


def _threshold_candidates_lower_is_block(values: Iterable[Any], targets: Iterable[float]) -> Dict[str, Dict[str, Any]]:
    nums = sorted(_numeric_values(values))
    out: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        threshold = _percentile(nums, target * 100.0)
        simulated = None if threshold is None else sum(1 for value in nums if value <= threshold) / len(nums)
        out[f"{target:.0%}"] = {
            "target_block_rate": target,
            "anomaly_score_threshold": threshold,
            "simulated_block_rate": simulated,
        }
    return out


def _threshold_candidates_higher_is_exit(values: Iterable[Any], targets: Iterable[float]) -> Dict[str, Dict[str, Any]]:
    nums = sorted(_numeric_values(values))
    out: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        threshold = _percentile(nums, (1.0 - target) * 100.0)
        simulated = None if threshold is None else sum(1 for value in nums if value >= threshold) / len(nums)
        out[f"{target:.0%}"] = {
            "target_exit_rate": target,
            "risk_score_threshold": threshold,
            "simulated_exit_rate": simulated,
        }
    return out


def _top_reasons(rows: Iterable[Dict[str, str]], *keys: str, limit: int = 5) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = ""
        for key in keys:
            raw = str(row.get(key) or "").strip()
            if raw:
                reason = raw.split("|", 1)[0].strip()
                break
        counts[reason or "unknown"] += 1
    return dict(counts.most_common(limit))


def _read_shadow_logs(logs_dir: Path) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    logs: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for key, name in SHADOW_LOGS.items():
        path = logs_dir / name
        status, rows, error = _read_csv_rows(path)
        if error is not None:
            errors.append(error)
        logs[key] = {
            "path": str(path),
            "status": status,
            "rows": len(rows),
            "data": rows,
        }
    return logs, errors


def _log_rates(rows: List[Dict[str, str]], would_key: str, actual_key: str) -> Dict[str, Any]:
    total = len(rows)
    would = sum(1 for row in rows if _truthy(row.get(would_key)))
    actual = sum(1 for row in rows if _truthy(row.get(actual_key)))
    return {
        "total_rows": total,
        "would_count": would,
        "actual_count": actual,
        "would_rate": None if total == 0 else would / total,
        "actual_rate": None if total == 0 else actual / total,
    }


def _baseline_verdict(closed_trades: int, total_pnl: Optional[float], win_rate: Optional[float]) -> str:
    if closed_trades <= 0:
        return "baseline_unproven"
    if (total_pnl is not None and total_pnl < 0.0) or (win_rate is not None and win_rate < 0.40):
        return "baseline_weak"
    if closed_trades < 10:
        return "baseline_needs_more_data"
    return "baseline_unproven"


def _baseline_section(selected: Dict[str, Dict[str, Any]], final: Dict[str, Any]) -> Dict[str, Any]:
    mode_report = selected.get("baseline", {})
    matrix_pnl = _get(_unified_data(mode_report), "paper_pnl", default={})
    phase15 = final.get("baseline_performance") if isinstance(final.get("baseline_performance"), dict) else {}
    closed = _first_int(matrix_pnl.get("closed_trade_count"), phase15.get("closed_trades"))
    total = _first_number(matrix_pnl.get("total_pnl"), phase15.get("total_pnl"))
    average = _first_number(matrix_pnl.get("average_pnl"), phase15.get("average_pnl"))
    win_rate = _first_number(matrix_pnl.get("win_rate"), phase15.get("win_rate"))
    return {
        "closed_trades": closed,
        "total_pnl": total,
        "average_pnl": average,
        "win_rate": win_rate,
        "verdict": _baseline_verdict(closed, total, win_rate),
    }


def _iforest_verdict(would_block_rate: Optional[float], actual_block_rate: Optional[float]) -> str:
    if (actual_block_rate or 0.0) > 0.0 or (would_block_rate or 0.0) >= 0.95:
        return "unsafe_to_enable"
    return "needs_threshold_calibration"


def _iforest_section(
    selected: Dict[str, Dict[str, Any]],
    final: Dict[str, Any],
    logs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mode_report = selected.get("iforest_shadow", {})
    matrix = _module_data(mode_report, "isolation_forest")
    phase15 = final.get("isolation_forest_summary") if isinstance(final.get("isolation_forest_summary"), dict) else {}
    rows = logs["isolation_forest"]["data"]
    log_rates = _log_rates(rows, "would_block", "actually_blocked")
    scores = _numeric_values(row.get("anomaly_score") for row in rows)
    dist = _distribution(scores)
    if not scores:
        dist = {
            "min": _first_number(matrix.get("min_anomaly_score")),
            "p10": _first_number(matrix.get("p10_anomaly_score")),
            "p50": _first_number(matrix.get("p50_anomaly_score")),
            "p90": _first_number(matrix.get("p90_anomaly_score")),
            "max": _first_number(matrix.get("max_anomaly_score")),
            "average": _first_number(matrix.get("average_anomaly_score"), phase15.get("average_anomaly_score")),
        }
    would_rate = _as_rate(_first_number(matrix.get("would_block_rate"), phase15.get("would_block_rate"), log_rates["would_rate"]))
    actual_rate = _as_rate(_first_number(matrix.get("actual_block_rate"), matrix.get("block_rate"), phase15.get("actual_block_rate"), log_rates["actual_rate"]))
    return {
        "current_would_block_rate": would_rate,
        "actual_block_rate": actual_rate,
        "anomaly_score_distribution": dist,
        "threshold_candidates": _threshold_candidates_lower_is_block(scores, [0.01, 0.05, 0.10, 0.15]),
        "recommendation": [
            "do not enable blocking",
            "collect more shadow data",
            "only consider threshold that produces low single-digit block rate",
        ],
        "verdict": _iforest_verdict(would_rate, actual_rate),
    }


def _xgboost_verdict(
    phase15_verdict: str,
    would_reject_rate: Optional[float],
    actual_reject_rate: Optional[float],
    confirm_matches: int,
    reject_matches: int,
    confirm_avg_pnl: Optional[float],
    reject_avg_pnl: Optional[float],
) -> str:
    if phase15_verdict == "best_candidate_for_more_shadow_testing":
        return "best_candidate_for_more_shadow_testing"
    outcome_supports = (
        confirm_matches > 0
        and reject_matches > 0
        and confirm_avg_pnl is not None
        and reject_avg_pnl is not None
        and confirm_avg_pnl >= reject_avg_pnl
    )
    rate_reasonable = would_reject_rate is not None and 0.0 < would_reject_rate <= 0.50
    if (actual_reject_rate or 0.0) == 0.0 and rate_reasonable and outcome_supports:
        return "best_candidate_for_more_shadow_testing"
    return "promising_but_unproven"


def _xgboost_section(
    selected: Dict[str, Dict[str, Any]],
    final: Dict[str, Any],
    logs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mode_report = selected.get("xgboost_shadow_outcome", {})
    matrix = _module_data(mode_report, "xgboost", "xgboost_signal")
    audit = _audit_data(mode_report)
    outcome = _get(_unified_data(mode_report), "xgboost_outcome", default={})
    phase15 = final.get("xgboost_summary") if isinstance(final.get("xgboost_summary"), dict) else {}
    rows = logs["xgboost"]["data"]
    log_rates = _log_rates(rows, "would_reject", "actually_rejected")
    confirm_rows = [row for row in rows if _truthy(row.get("would_confirm"))]
    reject_rows = [row for row in rows if _truthy(row.get("would_reject"))]
    would_confirm = _first_int(matrix.get("would_confirm_count"), audit.get("would_confirm_count"), phase15.get("would_confirm_count"), len(confirm_rows))
    would_reject = _first_int(matrix.get("would_reject_count"), audit.get("would_reject_count"), phase15.get("would_reject_count"), len(reject_rows))
    total_rows = _first_int(matrix.get("total_rows"), audit.get("total_xgboost_rows"), len(rows))
    reject_rate = _as_rate(_first_number(matrix.get("would_reject_rate"), phase15.get("would_reject_rate"), log_rates["would_rate"]))
    actual_rate = _as_rate(_first_number(matrix.get("actual_reject_rate"), phase15.get("actual_reject_rate"), log_rates["actual_rate"]))
    confirm_matches = _first_int(audit.get("would_confirm_matched_count"), outcome.get("would_confirm_matched_count"), phase15.get("would_confirm_matched_count"))
    reject_matches = _first_int(audit.get("would_reject_matched_count"), outcome.get("would_reject_matched_count"), phase15.get("would_reject_matched_count"))
    confirm_avg_pnl = _first_number(audit.get("would_confirm_average_pnl"), outcome.get("would_confirm_average_pnl"), phase15.get("would_confirm_average_pnl"))
    reject_avg_pnl = _first_number(audit.get("would_reject_average_pnl"), outcome.get("would_reject_average_pnl"), phase15.get("would_reject_average_pnl"))
    matched_total = confirm_matches + reject_matches
    return {
        "total_rows": total_rows,
        "would_confirm_count": would_confirm,
        "would_reject_count": would_reject,
        "would_reject_rate": reject_rate,
        "actual_reject_rate": actual_rate,
        "reject_reasons": _first_dict(matrix.get("reject_reason_counts"), matrix.get("reject_reasons"), audit.get("reject_reason_counts"), _top_reasons(reject_rows, "reject_reason", "reason")),
        "average_confidence_allowed": _first_number(matrix.get("average_confidence_allowed"), matrix.get("average_confidence_confirmed"), audit.get("average_confidence_allowed"), _avg(row.get("confidence") or row.get("xgboost_confidence") for row in confirm_rows)),
        "average_confidence_rejected": _first_number(matrix.get("average_confidence_rejected"), audit.get("average_confidence_rejected"), _avg(row.get("confidence") or row.get("xgboost_confidence") for row in reject_rows)),
        "would_confirm_matched_count": confirm_matches,
        "would_reject_matched_count": reject_matches,
        "would_confirm_average_pnl": confirm_avg_pnl,
        "would_reject_average_pnl": reject_avg_pnl,
        "sample_size_warning": matched_total < 30,
        "sample_size_message": "matched closed trade sample is small; do not enable blocking" if matched_total < 30 else "",
        "recommendation": [
            "continue xgboost_shadow_outcome only",
            "do not enable blocking until enough matched closed trades exist",
        ],
        "verdict": _xgboost_verdict(
            str(phase15.get("verdict") or ""),
            reject_rate,
            actual_rate,
            confirm_matches,
            reject_matches,
            confirm_avg_pnl,
            reject_avg_pnl,
        ),
    }


def _survival_verdict(would_exit_rate: Optional[float], actual_exit_rate: Optional[float]) -> str:
    if (would_exit_rate or 0.0) >= 0.80:
        return "too_aggressive"
    if (actual_exit_rate or 0.0) > 0.0:
        return "not_approved_active"
    return "not_approved_active"


def _survival_section(
    selected: Dict[str, Dict[str, Any]],
    final: Dict[str, Any],
    logs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mode_report = selected.get("survival_shadow", {})
    matrix = _module_data(mode_report, "survival_exit")
    phase15 = final.get("survival_exit_summary") if isinstance(final.get("survival_exit_summary"), dict) else {}
    rows = logs["survival_exit"]["data"]
    log_rates = _log_rates(rows, "would_exit_early", "actually_exited")
    scores = _numeric_values(row.get("survival_risk_score") for row in rows)
    would_rate = _as_rate(_first_number(matrix.get("would_exit_rate"), phase15.get("would_exit_rate"), log_rates["would_rate"]))
    actual_rate = _as_rate(_first_number(matrix.get("actual_exit_rate"), phase15.get("actual_exit_rate"), log_rates["actual_rate"]))
    return {
        "would_exit_rate": would_rate,
        "actual_exit_rate": actual_rate,
        "average_risk_score": _first_number(matrix.get("average_risk_score"), matrix.get("average_survival_risk_score"), phase15.get("average_risk_score"), _avg(scores)),
        "risk_score_distribution": _distribution(scores),
        "threshold_candidates": _threshold_candidates_higher_is_exit(scores, [0.05, 0.10, 0.20, 0.30]),
        "recommendation": [
            "do not enable SURVIVAL_EXIT_ACTIVE",
            "current threshold/model exits too much",
        ],
        "verdict": _survival_verdict(would_rate, actual_rate),
    }


def _advanced_risk_verdict(would_block_rate: Optional[float], actual_block_rate: Optional[float], dominance: bool) -> str:
    if (would_block_rate or 0.0) >= 0.70 or (actual_block_rate or 0.0) > 0.0:
        return "too_strict"
    if dominance or (would_block_rate or 0.0) >= 0.25:
        return "needs_rule_calibration"
    return "needs_rule_calibration"


def _advanced_risk_section(
    selected: Dict[str, Dict[str, Any]],
    final: Dict[str, Any],
    logs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mode_report = selected.get("advanced_risk_shadow", {})
    matrix = _module_data(mode_report, "advanced_risk")
    phase15 = final.get("advanced_risk_summary") if isinstance(final.get("advanced_risk_summary"), dict) else {}
    rows = logs["advanced_risk"]["data"]
    log_rates = _log_rates(rows, "would_block", "actually_blocked")
    reasons = _first_dict(matrix.get("top_reasons"), phase15.get("top_reasons"), _top_reasons(rows, "top_reason", "reasons"))
    reason_total = sum(_as_int(value) for value in reasons.values())
    max_open_count = _as_int(reasons.get("max_open_positions_limit"))
    dominance_rate = None if reason_total == 0 else max_open_count / reason_total
    dominance = bool(dominance_rate is not None and dominance_rate >= 0.50)
    would_rate = _as_rate(_first_number(matrix.get("would_block_rate"), phase15.get("would_block_rate"), log_rates["would_rate"]))
    actual_rate = _as_rate(_first_number(matrix.get("actual_block_rate"), phase15.get("actual_block_rate"), log_rates["actual_rate"]))
    return {
        "would_block_rate": would_rate,
        "actual_block_rate": actual_rate,
        "would_pause_count": _first_int(matrix.get("would_pause_count"), sum(1 for row in rows if _truthy(row.get("would_pause")))),
        "would_reduce_size_count": _first_int(matrix.get("would_reduce_size_count"), sum(1 for row in rows if _truthy(row.get("would_reduce_size")))),
        "top_reasons": reasons,
        "max_open_positions_limit_dominates": dominance,
        "max_open_positions_limit_rate": dominance_rate,
        "recommendation": [
            "do not enable ADVANCED_RISK_ACTIVE",
            "review ADVANCED_RISK_MAX_OPEN_POSITIONS and cooldown logic",
            "keep daily loss/consecutive loss protections as candidates only after paper proof",
        ],
        "verdict": _advanced_risk_verdict(would_rate, actual_rate, dominance),
    }


def _actual_counts(mode_report: Dict[str, Any]) -> Dict[str, int]:
    unified = _unified_data(mode_report)
    safety_counts = _get(unified, "safety", "actual_behavior_counts", default={})
    if isinstance(safety_counts, dict) and any(key in safety_counts for key in ACTUAL_COUNT_KEYS):
        return {key: _as_int(safety_counts.get(key)) for key in ACTUAL_COUNT_KEYS}
    isolation = _module_data(mode_report, "isolation_forest")
    xgboost = _module_data(mode_report, "xgboost", "xgboost_signal")
    survival = _module_data(mode_report, "survival_exit")
    advanced = _module_data(mode_report, "advanced_risk")
    return {
        "actually_blocked": _as_int(isolation.get("actually_blocked_count"))
        + _as_int(advanced.get("actually_blocked_count")),
        "actually_rejected": _as_int(xgboost.get("actually_rejected_count")),
        "actually_exited": _as_int(survival.get("actually_exited_count")),
        "actually_paused": _as_int(advanced.get("actually_paused_count")),
        "actually_reduced": _as_int(advanced.get("actually_reduced_count")),
    }


def _module_present(module: Dict[str, Any]) -> bool:
    return str(module.get("file_status") or "").lower() == "ok" and _first_int(module.get("total_rows")) > 0


def _coverage(total: int, with_id: int) -> Dict[str, Any]:
    return {
        "rows": total,
        "with_signal_id": with_id,
        "missing_signal_id": max(0, total - with_id),
        "coverage_rate": None if total == 0 else with_id / total,
    }


def _combined_section(selected: Dict[str, Dict[str, Any]], final: Dict[str, Any]) -> Dict[str, Any]:
    mode_report = selected.get("combined_shadow", {})
    phase15 = final.get("combined_shadow_integration_summary") if isinstance(final.get("combined_shadow_integration_summary"), dict) else {}
    isolation = _module_data(mode_report, "isolation_forest")
    xgboost = _module_data(mode_report, "xgboost", "xgboost_signal")
    survival = _module_data(mode_report, "survival_exit")
    advanced = _module_data(mode_report, "advanced_risk")
    module_presence = {
        "isolation_forest": _module_present(isolation),
        "xgboost": _module_present(xgboost),
        "survival_exit": _module_present(survival),
        "advanced_risk": _module_present(advanced),
    }
    if not any(module_presence.values()) and isinstance(phase15.get("module_presence"), dict):
        module_presence = dict(phase15["module_presence"])
    all_present = bool(all(module_presence.values())) if module_presence else bool(phase15.get("all_modules_present"))
    actual_counts = _actual_counts(mode_report) if mode_report else _first_dict(phase15.get("actual_behavior_counts"))
    lineage = _get(_unified_data(mode_report), "trade_lineage", default={})
    coverage = {
        "paper_trades": _coverage(
            _first_int(lineage.get("paper_trade_rows")),
            _first_int(lineage.get("paper_trade_rows_with_signal_id")),
        ),
        "closed_trades": _coverage(
            _first_int(lineage.get("closed_trade_rows")),
            _first_int(lineage.get("closed_trade_rows_with_signal_id")),
        ),
    }
    if coverage["paper_trades"]["rows"] == 0 and isinstance(phase15.get("signal_id_coverage"), dict):
        coverage = phase15["signal_id_coverage"]
    any_actual = any(_as_int(value) > 0 for value in actual_counts.values())
    verdict = str(phase15.get("verdict") or "")
    if not verdict:
        verdict = "integration_passed" if all_present and not any_actual else "integration_failed"
    return {
        "all_modules_present": all_present,
        "module_presence": module_presence,
        "actual_behavior_counts": actual_counts,
        "signal_id_coverage": coverage,
        "combined_verdict": verdict,
        "calibration_note": "combined shadow proves integration safety, not profitability",
    }


def _missing_inputs(report_groups: Dict[str, Dict[str, Dict[str, Path]]], logs: Dict[str, Dict[str, Any]], final_path: Path) -> List[str]:
    missing = []
    if not final_path.exists():
        missing.append(str(final_path))
    for mode in REQUIRED_MODES:
        if mode not in report_groups:
            missing.append(f"matrix reports for {mode}")
    for log in logs.values():
        if log["status"] == "missing":
            missing.append(log["path"])
    return missing


def _input_inventory(
    reports_dir: Path,
    logs_dir: Path,
    report_groups: Dict[str, Dict[str, Dict[str, Path]]],
    selected: Dict[str, Dict[str, Any]],
    logs: Dict[str, Dict[str, Any]],
    final_path: Path,
    final_loaded: bool,
    read_errors: List[str],
) -> Dict[str, Any]:
    modes = _ordered_modes(report_groups)
    return {
        "reports_dir": str(reports_dir),
        "logs_dir": str(logs_dir),
        "final_experiment_comparison": {
            "path": str(final_path),
            "status": "ok" if final_loaded else ("missing" if not final_path.exists() else "read_error"),
        },
        "reports_found": {mode: selected[mode]["paths"] for mode in _ordered_modes(selected)},
        "logs_found": {
            key: {"path": log["path"], "status": log["status"], "rows": log["rows"]}
            for key, log in logs.items()
        },
        "latest_matrix_timestamp_per_mode": {mode: _latest_timestamp(report_groups.get(mode, {})) for mode in modes},
        "missing_inputs": _missing_inputs(report_groups, logs, final_path),
        "read_errors": read_errors,
    }


def _overall_safety(final: Dict[str, Any]) -> Dict[str, Any]:
    phase15_verdict = _get(final, "final_recommendation", "verdict")
    return {
        "phase15_final_verdict": phase15_verdict,
        "keep_paper_only": True,
        "no_mainnet": True,
        "no_testnet_real_orders": True,
        "no_real_orders": True,
        "do_not_enable_active_or_blocking_modules_yet": True,
        "recommendations": [
            "keep PAPER only",
            "no mainnet",
            "no testnet real orders",
            "no real orders",
            "do not enable active/blocking modules yet",
        ],
    }


def _final_calibration_plan() -> Dict[str, Any]:
    return {
        "priority_order": [
            "Collect longer baseline + XGBoost shadow outcome data",
            "Keep IF, Survival, Advanced Risk shadow-only",
            "Tune thresholds using shadow distributions",
            "Re-run 60-minute paper tests after calibration",
            "Only then consider paper-only active/blocking tests",
        ],
        "final_verdict": "paper_only_calibration_required",
    }


def summarize_calibration_recommendations(
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
) -> Dict[str, Any]:
    reports_root = Path(reports_dir)
    logs_root = Path(logs_dir)
    read_errors: List[str] = []

    final_path = reports_root / FINAL_COMPARISON_NAME
    final: Dict[str, Any] = {}
    final_loaded = False
    if final_path.exists():
        payload, error = _read_json(final_path)
        if error is not None:
            read_errors.append(error)
        else:
            final = payload or {}
            final_loaded = True

    report_groups = _discover_report_files(reports_root)
    selected, report_errors = _load_latest_reports(report_groups)
    read_errors.extend(report_errors)
    logs, log_errors = _read_shadow_logs(logs_root)
    read_errors.extend(log_errors)

    return {
        "input_inventory": _input_inventory(
            reports_root,
            logs_root,
            report_groups,
            selected,
            logs,
            final_path,
            final_loaded,
            read_errors,
        ),
        "overall_safety_recommendation": _overall_safety(final),
        "baseline_strategy_health": _baseline_section(selected, final),
        "isolation_forest_calibration": _iforest_section(selected, final, logs),
        "xgboost_calibration": _xgboost_section(selected, final, logs),
        "survival_exit_calibration": _survival_section(selected, final, logs),
        "advanced_risk_calibration": _advanced_risk_section(selected, final, logs),
        "combined_shadow_calibration_view": _combined_section(selected, final),
        "final_calibration_plan": _final_calibration_plan(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_text_summary(summary: Dict[str, Any]) -> str:
    inventory = summary["input_inventory"]
    safety = summary["overall_safety_recommendation"]
    baseline = summary["baseline_strategy_health"]
    iso = summary["isolation_forest_calibration"]
    xgb = summary["xgboost_calibration"]
    survival = summary["survival_exit_calibration"]
    risk = summary["advanced_risk_calibration"]
    combined = summary["combined_shadow_calibration_view"]
    plan = summary["final_calibration_plan"]

    lines = [
        "Calibration Recommendation Report",
        f"Reports: {inventory['reports_dir']}",
        f"Logs: {inventory['logs_dir']}",
        "",
        "A. Input inventory",
        f"  reports_found: {inventory['reports_found']}",
        f"  logs_found: {inventory['logs_found']}",
        f"  latest_matrix_timestamp_per_mode: {inventory['latest_matrix_timestamp_per_mode']}",
        f"  missing_inputs: {inventory['missing_inputs']}",
        f"  read_errors: {inventory['read_errors']}",
        "",
        "B. Overall safety recommendation",
    ]
    lines.extend(f"  - {item}" for item in safety["recommendations"])
    lines.extend(
        [
            "",
            "C. Baseline strategy health",
            f"  closed_trades: {baseline['closed_trades']}",
            f"  total_pnl: {_fmt(baseline['total_pnl'])}",
            f"  average_pnl: {_fmt(baseline['average_pnl'])}",
            f"  win_rate: {_fmt(baseline['win_rate'])}",
            f"  verdict: {baseline['verdict']}",
            "",
            "D. Isolation Forest calibration",
            f"  current_would_block_rate: {_fmt(iso['current_would_block_rate'])}",
            f"  actual_block_rate: {_fmt(iso['actual_block_rate'])}",
            f"  anomaly_score_distribution: {iso['anomaly_score_distribution']}",
            f"  threshold_candidates: {iso['threshold_candidates']}",
            f"  recommendation: {iso['recommendation']}",
            f"  verdict: {iso['verdict']}",
            "",
            "E. XGBoost calibration",
            f"  would_confirm_count: {xgb['would_confirm_count']}",
            f"  would_reject_count: {xgb['would_reject_count']}",
            f"  would_reject_rate: {_fmt(xgb['would_reject_rate'])}",
            f"  actual_reject_rate: {_fmt(xgb['actual_reject_rate'])}",
            f"  reject_reasons: {xgb['reject_reasons']}",
            f"  average_confidence_allowed: {_fmt(xgb['average_confidence_allowed'])}",
            f"  average_confidence_rejected: {_fmt(xgb['average_confidence_rejected'])}",
            f"  would_confirm_matched_count: {xgb['would_confirm_matched_count']}",
            f"  would_reject_matched_count: {xgb['would_reject_matched_count']}",
            f"  would_confirm_average_pnl: {_fmt(xgb['would_confirm_average_pnl'])}",
            f"  would_reject_average_pnl: {_fmt(xgb['would_reject_average_pnl'])}",
            f"  sample_size_warning: {xgb['sample_size_warning']}",
            f"  recommendation: {xgb['recommendation']}",
            f"  verdict: {xgb['verdict']}",
            "",
            "F. Survival Exit calibration",
            f"  would_exit_rate: {_fmt(survival['would_exit_rate'])}",
            f"  actual_exit_rate: {_fmt(survival['actual_exit_rate'])}",
            f"  average_risk_score: {_fmt(survival['average_risk_score'])}",
            f"  risk_score_distribution: {survival['risk_score_distribution']}",
            f"  threshold_candidates: {survival['threshold_candidates']}",
            f"  recommendation: {survival['recommendation']}",
            f"  verdict: {survival['verdict']}",
            "",
            "G. Advanced Risk calibration",
            f"  would_block_rate: {_fmt(risk['would_block_rate'])}",
            f"  actual_block_rate: {_fmt(risk['actual_block_rate'])}",
            f"  would_pause_count: {risk['would_pause_count']}",
            f"  would_reduce_size_count: {risk['would_reduce_size_count']}",
            f"  top_reasons: {risk['top_reasons']}",
            f"  max_open_positions_limit_dominates: {risk['max_open_positions_limit_dominates']}",
            f"  recommendation: {risk['recommendation']}",
            f"  verdict: {risk['verdict']}",
            "",
            "H. Combined shadow calibration view",
            f"  all_modules_present: {combined['all_modules_present']}",
            f"  actual_behavior_counts: {combined['actual_behavior_counts']}",
            f"  signal_id_coverage: {combined['signal_id_coverage']}",
            f"  combined_verdict: {combined['combined_verdict']}",
            f"  calibration_note: {combined['calibration_note']}",
            "",
            "I. Final calibration plan",
        ]
    )
    for idx, item in enumerate(plan["priority_order"], start=1):
        lines.append(f"  {idx}. {item}")
    lines.append(f"  final_verdict: {plan['final_verdict']}")
    return "\n".join(lines)


def write_json_summary(summary: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibration recommendation report")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--json", action="store_true", help="Write reports/calibration_recommendation_report.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_calibration_recommendations(args.reports_dir, args.logs_dir)
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
