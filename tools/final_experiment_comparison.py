"""Final read-only comparison report for matrix experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "final_experiment_comparison.json"

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
INDEX_RE = re.compile(r"^matrix_index_(?P<timestamp>\d{14})\.json$")

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


def _ordered_modes(modes: Iterable[str]) -> List[str]:
    seen = set(modes)
    ordered = [mode for mode in MODE_ORDER if mode in seen]
    ordered.extend(sorted(mode for mode in seen if mode not in MODE_ORDER))
    return ordered


def _path_str(path: Path) -> str:
    return str(path)


def _as_number(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
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
            mode = match.group("mode")
            timestamp = match.group("timestamp")
            kind = match.group("kind")
            groups.setdefault(mode, {}).setdefault(timestamp, {})[kind] = path
    return groups


def _discover_index_runs(reports_dir: Path) -> tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], List[str]]:
    runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    read_errors: List[str] = []
    for path in sorted(reports_dir.glob("matrix_index_*.json")):
        match = INDEX_RE.match(path.name)
        if match is None:
            continue
        timestamp = match.group("timestamp")
        payload, error = _read_json(path)
        if error is not None:
            read_errors.append(error)
            continue
        for run in payload.get("runs", []):
            if not isinstance(run, dict):
                continue
            mode = str(run.get("mode") or "").strip()
            if not mode:
                continue
            enriched = dict(run)
            enriched["_index_path"] = path
            enriched["_index_timestamp"] = timestamp
            runs.setdefault(mode, {}).setdefault(timestamp, []).append(enriched)
    return runs, read_errors


def _latest_timestamp(timestamps: Iterable[str]) -> Optional[str]:
    values = sorted(str(ts) for ts in timestamps if str(ts).strip())
    return values[-1] if values else None


def _load_selected_reports(
    report_groups: Dict[str, Dict[str, Dict[str, Path]]]
) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
    selected: Dict[str, Dict[str, Any]] = {}
    read_errors: List[str] = []
    for mode, by_timestamp in report_groups.items():
        latest = _latest_timestamp(by_timestamp)
        if latest is None:
            continue
        reports = by_timestamp[latest]
        data: Dict[str, Dict[str, Any]] = {}
        for kind, path in reports.items():
            payload, error = _read_json(path)
            if error is not None:
                read_errors.append(error)
                continue
            data[kind] = payload
        selected[mode] = {
            "timestamp": latest,
            "paths": {kind: _path_str(path) for kind, path in sorted(reports.items())},
            "data": data,
        }
    return selected, read_errors


def _module_data(mode_report: Dict[str, Any], unified_key: str, shadow_key: Optional[str] = None) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    unified = data.get("unified") if isinstance(data.get("unified"), dict) else {}
    shadow = data.get("shadow_summary") if isinstance(data.get("shadow_summary"), dict) else {}
    primary = unified.get(unified_key)
    if isinstance(primary, dict):
        return primary
    fallback = shadow.get(shadow_key or unified_key)
    return fallback if isinstance(fallback, dict) else {}


def _xgboost_audit_data(mode_report: Dict[str, Any]) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    audit = data.get("xgboost_audit")
    return audit if isinstance(audit, dict) else {}


def _unified_data(mode_report: Dict[str, Any]) -> Dict[str, Any]:
    data = mode_report.get("data", {})
    unified = data.get("unified")
    return unified if isinstance(unified, dict) else {}


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


def _mode_safety(mode: str, mode_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if mode_report is None:
        return {
            "inferred_trade_mode": "missing",
            "shadow_only_warning": False,
            "actual_behavior_counts": {key: 0 for key in ACTUAL_COUNT_KEYS},
            "safety_verdict": "missing_reports",
        }

    unified = _unified_data(mode_report)
    safety = unified.get("safety") if isinstance(unified.get("safety"), dict) else {}
    actual_counts = _actual_counts(mode_report)
    inferred = str(safety.get("inferred_trade_mode") or "unknown")
    has_actual = any(count > 0 for count in actual_counts.values())
    shadow_warning = bool(safety.get("shadow_only_warning", has_actual))
    if inferred not in {"paper", "unknown"}:
        verdict = "unsafe_non_paper_mode_detected"
    elif has_actual:
        verdict = "active_behavior_detected"
    elif inferred == "paper":
        verdict = "safe_paper_run"
    else:
        verdict = "no_trade_mode_evidence"
    return {
        "inferred_trade_mode": inferred,
        "shadow_only_warning": shadow_warning,
        "actual_behavior_counts": actual_counts,
        "safety_verdict": verdict,
    }


def _sum_actual_counts(per_mode: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    totals = {key: 0 for key in ACTUAL_COUNT_KEYS}
    for item in per_mode.values():
        counts = item.get("actual_behavior_counts", {})
        for key in ACTUAL_COUNT_KEYS:
            totals[key] += _as_int(counts.get(key))
    return totals


def _baseline_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("baseline", {})
    pnl = _get(_unified_data(mode_report), "paper_pnl", default={})
    total_pnl = _first_number(pnl.get("total_pnl"), 0.0)
    return {
        "source_mode": "baseline",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified"),
        "closed_trades": _first_int(pnl.get("closed_trade_count"), pnl.get("closed_trades")),
        "total_pnl": total_pnl,
        "average_pnl": _first_number(pnl.get("average_pnl")),
        "win_rate": _first_number(pnl.get("win_rate")),
        "performance_note": "negative_pnl" if total_pnl is not None and total_pnl < 0 else "not_negative_pnl",
    }


def _iforest_verdict(would_block_rate: Optional[float], actual_block_rate: Optional[float], rows: int) -> str:
    if rows <= 0:
        return "not_available"
    if (actual_block_rate or 0.0) > 0.0 or (would_block_rate or 0.0) >= 0.95:
        return "unsafe_to_enable"
    if (would_block_rate or 0.0) >= 0.50:
        return "needs_calibration"
    return "shadow_only"


def _isolation_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("iforest_shadow", {})
    iso = _module_data(mode_report, "isolation_forest")
    rows = _first_int(iso.get("total_rows"))
    would = _as_rate(iso.get("would_block_rate"))
    actual = _as_rate(_first_number(iso.get("actual_block_rate"), iso.get("block_rate")))
    return {
        "source_mode": "iforest_shadow",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified") or _get(mode_report, "paths", "shadow_summary"),
        "total_rows": rows,
        "would_block_rate": would,
        "actual_block_rate": actual,
        "average_anomaly_score": _first_number(iso.get("average_anomaly_score")),
        "verdict": _iforest_verdict(would, actual, rows),
    }


def _xgboost_verdict(
    rows: int,
    would_reject_rate: Optional[float],
    actual_reject_rate: Optional[float],
    confirm_matches: int,
    reject_matches: int,
    confirm_avg_pnl: Optional[float],
    reject_avg_pnl: Optional[float],
) -> str:
    if rows <= 0:
        return "not_available"
    if (actual_reject_rate or 0.0) > 0.0:
        return "not_shadow_only"
    outcome_supports = (
        confirm_matches > 0
        and reject_matches > 0
        and confirm_avg_pnl is not None
        and reject_avg_pnl is not None
        and confirm_avg_pnl >= reject_avg_pnl
    )
    rate_reasonable = would_reject_rate is not None and 0.0 < would_reject_rate <= 0.50
    if rate_reasonable and outcome_supports:
        return "best_candidate_for_more_shadow_testing"
    if rate_reasonable:
        return "shadow_only_more_data_needed"
    return "needs_calibration"


def _xgboost_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("xgboost_shadow_outcome", {})
    xgb = _module_data(mode_report, "xgboost", "xgboost_signal")
    audit = _xgboost_audit_data(mode_report)
    outcome = _get(_unified_data(mode_report), "xgboost_outcome", default={})
    rows = _first_int(xgb.get("total_rows"), audit.get("total_xgboost_rows"))
    would_reject_rate = _as_rate(xgb.get("would_reject_rate"))
    actual_reject_rate = _as_rate(xgb.get("actual_reject_rate"))
    confirm_matches = _first_int(audit.get("would_confirm_matched_count"), outcome.get("would_confirm_matched_count"))
    reject_matches = _first_int(audit.get("would_reject_matched_count"), outcome.get("would_reject_matched_count"))
    confirm_avg = _first_number(audit.get("would_confirm_average_pnl"), outcome.get("would_confirm_average_pnl"))
    reject_avg = _first_number(audit.get("would_reject_average_pnl"), outcome.get("would_reject_average_pnl"))
    return {
        "source_mode": "xgboost_shadow_outcome",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified"),
        "audit_report": _get(mode_report, "paths", "xgboost_audit"),
        "total_rows": rows,
        "would_reject_rate": would_reject_rate,
        "actual_reject_rate": actual_reject_rate,
        "would_confirm_count": _first_int(xgb.get("would_confirm_count"), audit.get("would_confirm_count")),
        "would_reject_count": _first_int(xgb.get("would_reject_count"), audit.get("would_reject_count")),
        "would_confirm_matched_count": confirm_matches,
        "would_reject_matched_count": reject_matches,
        "would_confirm_average_pnl": confirm_avg,
        "would_reject_average_pnl": reject_avg,
        "verdict": _xgboost_verdict(
            rows,
            would_reject_rate,
            actual_reject_rate,
            confirm_matches,
            reject_matches,
            confirm_avg,
            reject_avg,
        ),
    }


def _survival_verdict(would_exit_rate: Optional[float], actual_exit_rate: Optional[float], rows: int) -> str:
    if rows <= 0:
        return "not_available"
    if (actual_exit_rate or 0.0) > 0.0:
        return "not_approved_active"
    if (would_exit_rate or 0.0) >= 0.80:
        return "too_aggressive"
    return "shadow_only"


def _survival_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("survival_shadow", {})
    survival = _module_data(mode_report, "survival_exit")
    rows = _first_int(survival.get("total_rows"))
    would = _as_rate(survival.get("would_exit_rate"))
    actual = _as_rate(survival.get("actual_exit_rate"))
    return {
        "source_mode": "survival_shadow",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified") or _get(mode_report, "paths", "shadow_summary"),
        "total_rows": rows,
        "would_exit_rate": would,
        "actual_exit_rate": actual,
        "average_risk_score": _first_number(
            survival.get("average_risk_score"),
            survival.get("average_survival_risk_score"),
        ),
        "verdict": _survival_verdict(would, actual, rows),
    }


def _advanced_risk_verdict(would_block_rate: Optional[float], actual_block_rate: Optional[float], rows: int) -> str:
    if rows <= 0:
        return "not_available"
    if (would_block_rate or 0.0) >= 0.70 or (actual_block_rate or 0.0) > 0.0:
        return "too_strict"
    if (would_block_rate or 0.0) >= 0.25:
        return "needs_rule_calibration"
    return "shadow_only"


def _advanced_risk_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("advanced_risk_shadow", {})
    risk = _module_data(mode_report, "advanced_risk")
    rows = _first_int(risk.get("total_rows"))
    would = _as_rate(risk.get("would_block_rate"))
    actual = _as_rate(_first_number(risk.get("actual_block_rate"), risk.get("block_rate")))
    return {
        "source_mode": "advanced_risk_shadow",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified") or _get(mode_report, "paths", "shadow_summary"),
        "total_rows": rows,
        "would_block_rate": would,
        "actual_block_rate": actual,
        "top_reasons": _first_dict(risk.get("top_reasons"), risk.get("top_reason_counts")),
        "verdict": _advanced_risk_verdict(would, actual, rows),
    }


def _coverage(total: int, with_id: int) -> Dict[str, Any]:
    return {
        "rows": total,
        "with_signal_id": with_id,
        "missing_signal_id": max(0, total - with_id),
        "coverage_rate": None if total == 0 else with_id / total,
    }


def _module_present(module: Dict[str, Any]) -> bool:
    return str(module.get("file_status") or "").lower() == "ok" and _first_int(module.get("total_rows")) > 0


def _combined_summary(selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mode_report = selected.get("combined_shadow", {})
    isolation = _module_data(mode_report, "isolation_forest")
    xgboost = _module_data(mode_report, "xgboost", "xgboost_signal")
    survival = _module_data(mode_report, "survival_exit")
    advanced = _module_data(mode_report, "advanced_risk")
    modules = {
        "isolation_forest": _module_present(isolation),
        "xgboost": _module_present(xgboost),
        "survival_exit": _module_present(survival),
        "advanced_risk": _module_present(advanced),
    }
    actual_counts = _actual_counts(mode_report) if mode_report else {key: 0 for key in ACTUAL_COUNT_KEYS}
    any_actual = any(count > 0 for count in actual_counts.values())
    lineage = _get(_unified_data(mode_report), "trade_lineage", default={})
    paper_coverage = _coverage(
        _first_int(lineage.get("paper_trade_rows")),
        _first_int(lineage.get("paper_trade_rows_with_signal_id")),
    )
    closed_coverage = _coverage(
        _first_int(lineage.get("closed_trade_rows")),
        _first_int(lineage.get("closed_trade_rows_with_signal_id")),
    )
    all_modules_present = all(modules.values())
    signal_ids_ok = paper_coverage["missing_signal_id"] == 0 and closed_coverage["missing_signal_id"] == 0
    return {
        "source_mode": "combined_shadow",
        "timestamp": mode_report.get("timestamp"),
        "source_report": _get(mode_report, "paths", "unified"),
        "all_modules_present": all_modules_present,
        "module_presence": modules,
        "any_actual_blocking_rejection_exit_pause_reduction": any_actual,
        "actual_behavior_counts": actual_counts,
        "signal_id_coverage": {
            "paper_trades": paper_coverage,
            "closed_trades": closed_coverage,
        },
        "verdict": "integration_passed" if all_modules_present and not any_actual and signal_ids_ok else "integration_failed",
    }


def _final_recommendation() -> Dict[str, Any]:
    return {
        "verdict": "paper_only_no_live_or_real_orders",
        "recommendations": [
            "no live/mainnet",
            "no testnet real-order behavior",
            "no real orders",
            "keep paper mode",
            "do not enable IF blocking",
            "do not enable Survival active",
            "do not enable Advanced Risk active",
            "XGBoost can continue in shadow outcome mode only",
        ],
        "next_work": "calibration and longer paper data collection",
    }


def summarize_final_comparison(reports_dir: Path | str = DEFAULT_REPORTS_DIR) -> Dict[str, Any]:
    root = Path(reports_dir)
    report_groups = _discover_report_files(root)
    index_runs, index_read_errors = _discover_index_runs(root)
    selected, report_read_errors = _load_selected_reports(report_groups)

    modes_found = _ordered_modes(set(report_groups) | set(index_runs))
    selected_modes = _ordered_modes(selected)
    latest_timestamp_per_mode = {
        mode: (_latest_timestamp(report_groups.get(mode, {})) if mode in report_groups else None)
        for mode in modes_found
    }
    report_paths_used = {mode: selected[mode]["paths"] for mode in selected_modes}
    index_paths_used: Dict[str, List[str]] = {}
    index_run_status: Dict[str, List[Dict[str, Any]]] = {}
    for mode in modes_found:
        timestamp = latest_timestamp_per_mode.get(mode)
        matching_runs = index_runs.get(mode, {}).get(timestamp, []) if timestamp else []
        if not matching_runs and mode in index_runs:
            latest_index_ts = _latest_timestamp(index_runs[mode])
            matching_runs = index_runs[mode].get(latest_index_ts, []) if latest_index_ts else []
        paths = sorted({_path_str(run["_index_path"]) for run in matching_runs if run.get("_index_path")})
        if paths:
            index_paths_used[mode] = paths
        if matching_runs:
            index_run_status[mode] = [
                {
                    "index_timestamp": run.get("_index_timestamp"),
                    "exit_status": run.get("exit_status"),
                    "duration_minutes": run.get("duration_minutes"),
                    "notes": run.get("notes", []),
                    "index_path": _path_str(run["_index_path"]),
                }
                for run in matching_runs
            ]

    inventory = {
        "reports_dir": _path_str(root),
        "modes_found": modes_found,
        "latest_timestamp_per_mode": latest_timestamp_per_mode,
        "missing_expected_modes": [mode for mode in REQUIRED_MODES if mode not in report_groups],
        "report_paths_used": report_paths_used,
        "index_paths_used": index_paths_used,
        "index_run_status": index_run_status,
        "read_errors": index_read_errors + report_read_errors,
    }

    safety_per_mode = {mode: _mode_safety(mode, selected.get(mode)) for mode in _ordered_modes(set(selected) | set(REQUIRED_MODES))}
    safety = {
        "per_mode": safety_per_mode,
        "aggregate_actual_behavior_counts": _sum_actual_counts(safety_per_mode),
        "shadow_only_warning": any(item["shadow_only_warning"] for item in safety_per_mode.values()),
    }

    return {
        "run_inventory": inventory,
        "safety_summary": safety,
        "baseline_performance": _baseline_summary(selected),
        "isolation_forest_summary": _isolation_summary(selected),
        "xgboost_summary": _xgboost_summary(selected),
        "survival_exit_summary": _survival_summary(selected),
        "advanced_risk_summary": _advanced_risk_summary(selected),
        "combined_shadow_integration_summary": _combined_summary(selected),
        "final_recommendation": _final_recommendation(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_paths(paths: Dict[str, Any]) -> str:
    if not paths:
        return "{}"
    return json.dumps(paths, sort_keys=True)


def format_text_summary(summary: Dict[str, Any]) -> str:
    inventory = summary["run_inventory"]
    safety = summary["safety_summary"]
    baseline = summary["baseline_performance"]
    iso = summary["isolation_forest_summary"]
    xgb = summary["xgboost_summary"]
    survival = summary["survival_exit_summary"]
    risk = summary["advanced_risk_summary"]
    combined = summary["combined_shadow_integration_summary"]
    recommendation = summary["final_recommendation"]

    lines = [
        "Final Experiment Comparison",
        f"Reports: {inventory['reports_dir']}",
        "",
        "A. Run inventory",
        f"  modes_found: {inventory['modes_found']}",
        f"  latest_timestamp_per_mode: {inventory['latest_timestamp_per_mode']}",
        f"  missing_expected_modes: {inventory['missing_expected_modes']}",
        f"  report_paths_used: {_fmt_paths(inventory['report_paths_used'])}",
        f"  index_paths_used: {_fmt_paths(inventory['index_paths_used'])}",
        f"  read_errors: {inventory['read_errors']}",
        "",
        "B. Safety summary",
        f"  aggregate_actual_behavior_counts: {safety['aggregate_actual_behavior_counts']}",
        f"  shadow_only_warning: {safety['shadow_only_warning']}",
    ]
    for mode, item in safety["per_mode"].items():
        lines.append(
            "  {mode}: inferred_trade_mode={mode_value} shadow_only_warning={warning} "
            "actuals={actuals} safety_verdict={verdict}".format(
                mode=mode,
                mode_value=item["inferred_trade_mode"],
                warning=item["shadow_only_warning"],
                actuals=item["actual_behavior_counts"],
                verdict=item["safety_verdict"],
            )
        )

    lines.extend(
        [
            "",
            "C. Baseline performance",
            f"  closed_trades: {baseline['closed_trades']}",
            f"  total_pnl: {_fmt(baseline['total_pnl'])}",
            f"  average_pnl: {_fmt(baseline['average_pnl'])}",
            f"  win_rate: {_fmt(baseline['win_rate'])}",
            "",
            "D. Isolation Forest summary",
            f"  would_block_rate: {_fmt(iso['would_block_rate'])}",
            f"  actual_block_rate: {_fmt(iso['actual_block_rate'])}",
            f"  average_anomaly_score: {_fmt(iso['average_anomaly_score'])}",
            f"  verdict: {iso['verdict']}",
            "",
            "E. XGBoost summary",
            f"  would_reject_rate: {_fmt(xgb['would_reject_rate'])}",
            f"  actual_reject_rate: {_fmt(xgb['actual_reject_rate'])}",
            f"  would_confirm_matched_count: {xgb['would_confirm_matched_count']}",
            f"  would_reject_matched_count: {xgb['would_reject_matched_count']}",
            f"  would_confirm_average_pnl: {_fmt(xgb['would_confirm_average_pnl'])}",
            f"  would_reject_average_pnl: {_fmt(xgb['would_reject_average_pnl'])}",
            f"  verdict: {xgb['verdict']}",
            "",
            "F. Survival Exit summary",
            f"  would_exit_rate: {_fmt(survival['would_exit_rate'])}",
            f"  actual_exit_rate: {_fmt(survival['actual_exit_rate'])}",
            f"  average_risk_score: {_fmt(survival['average_risk_score'])}",
            f"  verdict: {survival['verdict']}",
            "",
            "G. Advanced Risk summary",
            f"  would_block_rate: {_fmt(risk['would_block_rate'])}",
            f"  actual_block_rate: {_fmt(risk['actual_block_rate'])}",
            f"  top_reasons: {risk['top_reasons']}",
            f"  verdict: {risk['verdict']}",
            "",
            "H. Combined shadow integration summary",
            f"  all_modules_present: {combined['all_modules_present']}",
            "  any_actual_blocking_rejection_exit_pause_reduction: "
            f"{combined['any_actual_blocking_rejection_exit_pause_reduction']}",
            f"  signal_id_coverage: {combined['signal_id_coverage']}",
            f"  verdict: {combined['verdict']}",
            "",
            "I. Final recommendation",
            f"  verdict: {recommendation['verdict']}",
        ]
    )
    lines.extend(f"  - {item}" for item in recommendation["recommendations"])
    lines.append(f"  next_work: {recommendation['next_work']}")
    return "\n".join(lines)


def write_json_summary(summary: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Final experiment comparison report")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--json", action="store_true", help="Write reports/final_experiment_comparison.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_final_comparison(args.reports_dir)
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
