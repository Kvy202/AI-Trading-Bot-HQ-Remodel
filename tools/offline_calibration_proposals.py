"""Phase 18 read-only calibration proposals and evidence gates.

This module converts Phase 17 evidence into specifications and paper-only
experiment plans. It never trains a model, writes an artifact, changes runtime
configuration, or enables an active/blocking path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools.offline_calibration_sweep import summarize_offline_calibration
except ModuleNotFoundError:  # Direct execution adds tools/, rather than repo root, to sys.path.
    from offline_calibration_sweep import summarize_offline_calibration

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "offline_calibration_proposals.json"
PHASE17_REPORT_NAME = "offline_calibration_sweep.json"

MATRIX_REPORT_RE = re.compile(
    r"^matrix_(?P<mode>.+)_(?P<timestamp>\d{14})_"
    r"(?P<kind>unified|xgboost_audit)\.json$"
)
ROW_LOG_NAMES = {
    "isolation_forest": "isolation_forest_shadow.csv",
    "xgboost": "xgboost_signal_shadow.csv",
    "survival_exit": "survival_exit_shadow.csv",
    "advanced_risk": "advanced_risk_shadow.csv",
}
SAFE_EXPERIMENT_COMMANDS = {
    "baseline": (
        'powershell -NoProfile -ExecutionPolicy Bypass -File '
        '".\\tools\\run_experiment_matrix.ps1" -Mode baseline -Minutes 60 -FreshLogs'
    ),
    "xgboost_shadow_outcome": (
        'powershell -NoProfile -ExecutionPolicy Bypass -File '
        '".\\tools\\run_experiment_matrix.ps1" '
        "-Mode xgboost_shadow_outcome -Minutes 60 -FreshLogs"
    ),
    "combined_shadow": (
        'powershell -NoProfile -ExecutionPolicy Bypass -File '
        '".\\tools\\run_experiment_matrix.ps1" '
        "-Mode combined_shadow -Minutes 60 -FreshLogs"
    ),
}


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


def _discover_direct_inputs(reports_dir: Path) -> Dict[str, Any]:
    paths: set[Path] = set()
    for pattern in ("matrix_*_unified.json", "matrix_*_xgboost_audit.json"):
        paths.update(reports_dir.glob(pattern))
    for name in ("final_experiment_comparison.json", "calibration_recommendation_report.json"):
        path = reports_dir / name
        if path.exists():
            paths.add(path)

    valid: Dict[Path, Dict[str, Any]] = {}
    malformed: List[Dict[str, str]] = []
    read_errors: List[str] = []
    for path in sorted(paths, key=lambda item: item.name):
        payload, error = _read_json(path)
        if error is not None:
            malformed.append({"path": str(path), "error": error})
            read_errors.append(error)
        elif payload is not None:
            valid[path] = payload

    timestamps: Dict[str, List[str]] = {}
    identities: Dict[Tuple[str, str, str], Path] = {}
    duplicates: List[Dict[str, str]] = []
    for path in sorted(valid, key=lambda item: item.name):
        match = MATRIX_REPORT_RE.match(path.name)
        if match is None:
            continue
        mode = match.group("mode")
        timestamp = match.group("timestamp")
        kind = match.group("kind")
        identity = (mode, timestamp, kind)
        if identity in identities:
            duplicates.append(
                {
                    "path": str(path),
                    "retained_path": str(identities[identity]),
                    "identity": f"{mode}:{timestamp}:{kind}",
                    "reason": "duplicate mode + timestamp + report kind",
                }
            )
            continue
        identities[identity] = path
        timestamps.setdefault(mode, []).append(timestamp)

    return {
        "paths_found": [str(path) for path in sorted(paths, key=lambda item: item.name)],
        "valid": valid,
        "valid_paths": [str(path) for path in sorted(valid, key=lambda item: item.name)],
        "malformed": malformed,
        "read_errors": read_errors,
        "duplicates": duplicates,
        "timestamps": {
            mode: sorted(set(values))
            for mode, values in sorted(timestamps.items())
        },
    }


def _read_current_logs(logs_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    logs: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for key, filename in ROW_LOG_NAMES.items():
        path = logs_dir / filename
        status, rows, error = _read_csv(path)
        logs[key] = {
            "path": str(path),
            "status": status,
            "row_count": len(rows),
            "columns": sorted(rows[0].keys()) if rows else [],
            "rows": rows,
        }
        if error is not None:
            errors.append(error)
    return logs, errors


def _phase17_schema_valid(payload: Dict[str, Any]) -> bool:
    required = {
        "baseline_cross_run_summary",
        "isolation_forest_analysis",
        "xgboost_outcome_aggregation",
        "survival_exit_analysis",
        "advanced_risk_analysis",
        "combined_shadow_integration",
    }
    return required.issubset(payload)


def _load_phase17_or_reconstruct(
    reports_dir: Path,
    logs_dir: Path,
    xgb_min_matched_per_group: int,
    if_target_block_rate: float,
    survival_target_exit_rate: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    phase17_path = reports_dir / PHASE17_REPORT_NAME
    phase17_status: Dict[str, Any] = {
        "path": str(phase17_path),
        "status": "missing",
        "preferred_input_used": False,
        "error": None,
    }
    payload: Optional[Dict[str, Any]] = None
    if phase17_path.exists():
        loaded, error = _read_json(phase17_path)
        if error is not None:
            phase17_status.update({"status": "malformed", "error": error})
        elif loaded is not None and not _phase17_schema_valid(loaded):
            phase17_status.update(
                {
                    "status": "malformed",
                    "error": f"{phase17_path}: required Phase 17 sections are missing",
                }
            )
        else:
            payload = loaded
            phase17_status.update({"status": "ok", "preferred_input_used": True})

    reconstructed = False
    if payload is None:
        payload = summarize_offline_calibration(
            reports_dir=reports_dir,
            logs_dir=logs_dir,
            min_xgb_matched_per_group=xgb_min_matched_per_group,
            target_if_block_rates=(if_target_block_rate,),
            target_survival_exit_rates=(survival_target_exit_rate,),
        )
        reconstructed = True

    return payload, {
        "phase17_report": phase17_status,
        "evidence_source": (
            "phase17_preferred_report"
            if phase17_status["preferred_input_used"]
            else "reconstructed_from_reports_and_current_logs"
        ),
        "fallback_reconstruction_used": reconstructed,
    }


def _input_inventory(
    reports_dir: Path,
    logs_dir: Path,
    phase17: Dict[str, Any],
    phase17_meta: Dict[str, Any],
    direct: Dict[str, Any],
    logs: Dict[str, Dict[str, Any]],
    log_errors: List[str],
) -> Dict[str, Any]:
    phase17_inventory = phase17.get("input_inventory")
    phase17_inventory = phase17_inventory if isinstance(phase17_inventory, dict) else {}
    fallback_used = (
        direct["valid_paths"]
        if phase17_meta["fallback_reconstruction_used"]
        else []
    )
    duplicates = list(direct["duplicates"])
    for item in phase17_inventory.get("duplicate_reports_skipped", []):
        if isinstance(item, dict) and item not in duplicates:
            duplicates.append(item)

    malformed = list(direct["malformed"])
    phase17_status = phase17_meta["phase17_report"]
    if phase17_status["status"] == "malformed":
        malformed.insert(
            0,
            {
                "path": phase17_status["path"],
                "error": str(phase17_status["error"]),
            },
        )
    read_errors = list(direct["read_errors"]) + list(log_errors)
    if phase17_status["status"] == "malformed" and phase17_status["error"] not in read_errors:
        read_errors.insert(0, str(phase17_status["error"]))

    logs_used = [
        {
            "name": key,
            "path": value["path"],
            "status": value["status"],
            "row_count": value["row_count"],
            "columns": value["columns"],
        }
        for key, value in logs.items()
        if value["status"] == "ok"
    ]
    missing_inputs: List[str] = []
    if not fallback_used and phase17_status["status"] != "ok":
        missing_inputs.append("usable Phase 17 or fallback matrix evidence")
    for key, value in logs.items():
        if value["status"] != "ok":
            missing_inputs.append(ROW_LOG_NAMES[key])

    timestamps = phase17_inventory.get("run_timestamps_by_mode")
    if not isinstance(timestamps, dict) or not timestamps:
        timestamps = direct["timestamps"]
    return {
        "reports_dir": str(reports_dir),
        "logs_dir": str(logs_dir),
        "phase17_report_status": phase17_status,
        "preferred_evidence_source": phase17_meta["evidence_source"],
        "fallback_reports_found": direct["paths_found"],
        "fallback_reports_used": fallback_used,
        "logs_used": logs_used,
        "report_timestamps": timestamps,
        "duplicate_inputs_skipped": duplicates,
        "malformed_inputs_skipped": malformed,
        "missing_inputs": missing_inputs,
        "read_errors": read_errors,
        "archived_csv_policy": "not read or aggregated without a safely verified run identity",
    }


def _global_safety_gate() -> Dict[str, Any]:
    return {
        "paper_only": True,
        "no_mainnet": True,
        "no_testnet_real_orders": True,
        "place_real_orders_must_remain_false": True,
        "active_modules_allowed": False,
        "blocking_modules_allowed": False,
        "status": "locked",
        "unlock_condition": "future_explicit_implementation_phase_only",
        "this_report_can_unlock_safety": False,
    }


def _baseline_report_payloads(direct: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    results: List[Tuple[str, Dict[str, Any]]] = []
    for path, payload in direct["valid"].items():
        match = MATRIX_REPORT_RE.match(path.name)
        if (
            match is not None
            and match.group("mode") == "baseline"
            and match.group("kind") == "unified"
        ):
            results.append((match.group("timestamp"), payload))
    return sorted(results, key=lambda item: item[0])


def _baseline_lineage_and_outlier_diagnostics(
    direct: Dict[str, Any],
    aggregate_total_pnl: Optional[float],
) -> Dict[str, Any]:
    extrema: List[Dict[str, Any]] = []
    lineage_missing = 0
    lineage_observed = False
    for timestamp, payload in _baseline_report_payloads(direct):
        paper_pnl = payload.get("paper_pnl")
        paper_pnl = paper_pnl if isinstance(paper_pnl, dict) else {}
        for label in ("best_trade", "worst_trade"):
            trade = paper_pnl.get(label)
            if not isinstance(trade, dict):
                continue
            pnl = _as_number(trade.get("realized_pnl"))
            if pnl is not None:
                extrema.append(
                    {
                        "timestamp": timestamp,
                        "kind": label,
                        "realized_pnl": pnl,
                        "signal_id": trade.get("signal_id"),
                    }
                )
        lineage = payload.get("trade_lineage")
        lineage = lineage if isinstance(lineage, dict) else {}
        missing = lineage.get("signal_id_missing_counts")
        if isinstance(missing, dict):
            lineage_observed = True
            lineage_missing += _as_int(missing.get("paper_trades"))
            lineage_missing += _as_int(missing.get("closed_trades"))

    maximum_extreme = max((abs(item["realized_pnl"]) for item in extrema), default=None)
    outlier_warning: Optional[bool]
    if maximum_extreme is None or aggregate_total_pnl is None:
        outlier_warning = None
    else:
        outlier_warning = maximum_extreme >= max(abs(aggregate_total_pnl), 1e-12)
    return {
        "reported_run_extrema": extrema,
        "maximum_absolute_reported_trade_extreme": maximum_extreme,
        "outlier_warning": outlier_warning,
        "outlier_status": (
            "possible_single_trade_dominance"
            if outlier_warning
            else (
                "not_detected_in_available_run_extrema"
                if outlier_warning is False
                else "insufficient_trade_distribution"
            )
        ),
        "limitation": (
            "Matrix reports expose run best/worst trades, not every trade; "
            "the acceptance gate requires full trade-level review."
        ),
        "lineage_evidence_available": lineage_observed,
        "lineage_missing_signal_id_count": lineage_missing if lineage_observed else None,
        "no_unresolved_lineage_errors": lineage_observed and lineage_missing == 0,
    }


def _baseline_proposal(
    phase17: Dict[str, Any],
    direct: Dict[str, Any],
    minimum_closed_trades: int,
) -> Dict[str, Any]:
    current = phase17.get("baseline_cross_run_summary")
    current = current if isinstance(current, dict) else {}
    runs = _as_int(current.get("number_of_runs"))
    closed = _as_int(current.get("total_closed_trades"))
    total_pnl = _as_number(current.get("total_pnl"))
    weighted = _as_number(current.get("weighted_average_pnl"))
    win_rate = _as_rate(current.get("overall_win_rate"))
    positive = _as_int(current.get("positive_run_count"))
    negative = _as_int(current.get("negative_run_count"))
    timestamps = [
        str(item.get("timestamp"))
        for item in current.get("runs", [])
        if isinstance(item, dict) and item.get("timestamp")
    ]
    multiple_windows = len(set(timestamps)) >= 2 if timestamps else runs >= 2
    positive_consistent = runs >= 2 and positive == runs and negative == 0
    diagnostics = _baseline_lineage_and_outlier_diagnostics(direct, total_pnl)
    additional = max(0, minimum_closed_trades - closed)
    gate_checks = {
        "configured_minimum_closed_trades_reached": closed >= minimum_closed_trades,
        "multiple_independent_runs": runs >= 2 and multiple_windows,
        "no_unresolved_lineage_errors": diagnostics["no_unresolved_lineage_errors"],
        "no_reliance_on_one_outlier_trade": diagnostics["outlier_warning"] is False,
        "positive_performance_stable_across_runs": positive_consistent,
    }
    return {
        "current_evidence": {
            "number_of_runs": runs,
            "closed_trades": closed,
            "aggregate_pnl": total_pnl,
            "weighted_average_pnl": weighted,
            "overall_win_rate": win_rate,
            "positive_run_count": positive,
            "negative_run_count": negative,
            "verdict": str(current.get("verdict") or "baseline_unproven"),
        },
        "evidence_gaps": {
            "configured_minimum_closed_trades": minimum_closed_trades,
            "additional_closed_trades_required": additional,
            "multiple_market_windows_represented": multiple_windows,
            "market_window_assessment": "independent_timestamped_run_proxy",
            "positive_performance_consistent_across_runs": positive_consistent,
            "outlier_assessment": diagnostics,
        },
        "proposal": [
            "continue paper-only baseline collection",
            "do not claim profitability",
            "run repeated 60-minute or longer baseline windows",
            "preserve every run as a unique matrix report",
            "do not replace existing evidence with only the latest run",
        ],
        "acceptance_gate": {
            "checks": gate_checks,
            "gate_satisfied": all(gate_checks.values()),
            "review_gate_note": "These are evidence review gates, not profitability guarantees.",
        },
        "verdict": "collect_more_baseline_evidence",
        "profitability_claim_allowed": False,
    }


def _isolation_forest_specification(
    phase17: Dict[str, Any],
    target_block_rate: float,
) -> Dict[str, Any]:
    current = phase17.get("isolation_forest_analysis")
    current = current if isinstance(current, dict) else {}
    return {
        "problem_statement": {
            "all_shadow_observations_flagged_as_anomalies": bool(
                current.get("consistent_100_percent_anomaly_behavior")
            ),
            "threshold_only_calibration_infeasible": not bool(
                current.get("threshold_only_calibration_feasible")
            ),
            "heavy_score_ties_or_saturation": bool(
                current.get("score_saturation_detected")
            ),
            "summary": (
                "The current model flags essentially all shadow observations as anomalies; "
                "threshold-only calibration is infeasible and score resolution is saturated."
            ),
        },
        "required_training_data": [
            "historical feature rows covering multiple market regimes",
            "normal and volatile periods",
            "time-ordered data with no future leakage",
            "the same feature schema and ordering as the current artifact",
            "existing FEATURE_COLS preserved without modification",
            "missing-value validation and scaler compatibility verification",
        ],
        "proposed_experiment_grid": {
            "contamination": [0.005, 0.01, 0.02, 0.05],
            "random_seeds": [7, 42, 137],
            "estimator_counts": [100, 200, 500],
            "optional_max_samples": ["auto", 0.5, 0.8],
            "validation": {
                "method": "time_ordered_train_validation_split",
                "shuffle": False,
                "placeholder_split": "<TRAIN_END_TIMESTAMP>/<VALIDATION_START_TIMESTAMP>",
            },
        },
        "offline_acceptance_gates": {
            "configured_target_shadow_block_rate": target_block_rate,
            "target_is_low_single_digit": target_block_rate <= 0.05,
            "requirements": [
                "observed shadow block rate is low single-digit",
                "no 100% anomaly behavior",
                "score distribution has useful resolution",
                "tied-score concentration is materially reduced",
                "missing-artifact and model-error safety tests pass",
                "baseline signal generation does not degrade",
                "new artifact receives a new version",
                "existing artifact is never overwritten automatically",
            ],
            "current_artifact_gate_satisfied": False,
        },
        "training_command_template": {
            "execute_in_phase18": False,
            "repository_cli_verified": True,
            "verified_options": [
                "--input-csv",
                "--csv-feature-cols",
                "--csv-symbol-col",
                "--csv-time-col",
                "--feature-width",
                "--seq-len",
                "--step",
                "--contamination",
                "--estimators",
                "--out",
            ],
            "template": (
                'python tools\\train_isolation_forest.py '
                '--input-csv "<TRAINING_CSV>" '
                '--csv-feature-cols "<CURRENT_FEATURE_COLUMNS_IN_EXISTING_ORDER>" '
                '--csv-time-col "<TIMESTAMP_COLUMN>" '
                '--contamination "<CONTAMINATION_CANDIDATE>" '
                '--estimators "<ESTIMATOR_COUNT>" '
                '--out "model_artifacts\\isolation_forest_<NEW_VERSION>.joblib"'
            ),
            "unsupported_grid_placeholders": {
                "random_seed": "<NO_VERIFIED_TRAINING_CLI_OPTION; FUTURE_PHASE_IMPLEMENTATION>",
                "max_samples": "<NO_VERIFIED_TRAINING_CLI_OPTION; FUTURE_PHASE_IMPLEMENTATION>",
                "time_split_orchestrator": "<FUTURE_OFFLINE_EXPERIMENT_DRIVER>",
            },
        },
        "verdict": "retraining_spec_ready",
        "artifact_status": "current_artifact_not_approved",
        "artifact_modified": False,
    }


def _xgb_runs_remaining(
    remaining: int,
    historical_matches: int,
    historical_runs: int,
) -> Dict[str, Any]:
    if remaining <= 0:
        return {
            "estimate_available": True,
            "estimated_runs_remaining": 0,
            "historical_matches_per_run": (
                None if historical_runs <= 0 else historical_matches / historical_runs
            ),
            "approximate": True,
        }
    rate = historical_matches / historical_runs if historical_runs > 0 else 0.0
    if rate <= 0:
        return {
            "estimate_available": False,
            "estimated_runs_remaining": None,
            "historical_matches_per_run": None,
            "approximate": True,
            "reason": "no positive historical matched-trades-per-run rate",
        }
    return {
        "estimate_available": True,
        "estimated_runs_remaining": int(math.ceil(remaining / rate)),
        "historical_matches_per_run": rate,
        "approximate": True,
        "warning": "Estimate assumes future paper runs resemble historical matching rates.",
    }


def _xgboost_proposal(
    phase17: Dict[str, Any],
    minimum_per_group: int,
) -> Dict[str, Any]:
    section = phase17.get("xgboost_outcome_aggregation")
    section = section if isinstance(section, dict) else {}
    aggregate = section.get("combined_aggregate")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    confirmed = aggregate.get("confirmed")
    confirmed = confirmed if isinstance(confirmed, dict) else {}
    rejected = aggregate.get("rejected")
    rejected = rejected if isinstance(rejected, dict) else {}
    confirmed_count = _as_int(confirmed.get("matched_count"))
    rejected_count = _as_int(rejected.get("matched_count"))
    remaining_confirmed = max(0, minimum_per_group - confirmed_count)
    remaining_rejected = max(0, minimum_per_group - rejected_count)

    by_mode = section.get("by_source_mode")
    by_mode = by_mode if isinstance(by_mode, dict) else {}
    dedicated = by_mode.get("xgboost_shadow_outcome")
    dedicated = dedicated if isinstance(dedicated, dict) else {}
    historical_runs = _as_int(dedicated.get("number_of_runs"))
    historical_confirmed = _as_int(_get(dedicated, "confirmed", "matched_count"))
    historical_rejected = _as_int(_get(dedicated, "rejected", "matched_count"))
    if historical_runs <= 0:
        historical_runs = _as_int(aggregate.get("number_of_runs"))
        historical_confirmed = confirmed_count
        historical_rejected = rejected_count

    consistency = aggregate.get("relationship_consistent_across_runs")
    separation = _as_number(aggregate.get("pnl_separation"))
    coverage = _as_rate(aggregate.get("match_coverage_rate"))
    unmatched_warning = bool(aggregate.get("unmatched_data_warning"))
    progress_confirmed = (
        min(100.0, confirmed_count / minimum_per_group * 100.0)
        if minimum_per_group > 0
        else 100.0
    )
    progress_rejected = (
        min(100.0, rejected_count / minimum_per_group * 100.0)
        if minimum_per_group > 0
        else 100.0
    )
    combined = phase17.get("combined_shadow_integration")
    combined = combined if isinstance(combined, dict) else {}
    paper_coverage = _get(combined, "signal_id_coverage", "paper_trades", "coverage_rate")
    closed_coverage = _get(combined, "signal_id_coverage", "closed_trades", "coverage_rate")
    no_missing_signal_ids = paper_coverage == 1.0 and closed_coverage == 1.0
    gate_checks = {
        "minimum_confirmed_matched_reached": confirmed_count >= minimum_per_group,
        "minimum_rejected_matched_reached": rejected_count >= minimum_per_group,
        "rejected_consistently_worse_across_multiple_runs": (
            consistency is True and historical_runs >= 2
        ),
        "positive_pnl_separation": separation is not None and separation > 0,
        "match_coverage_reported_and_understood": coverage is not None and not unmatched_warning,
        "no_missing_signal_ids": no_missing_signal_ids,
        "not_dominated_by_one_trade": False,
        "baseline_performance_reported_separately": bool(
            phase17.get("baseline_cross_run_summary")
        ),
    }
    return {
        "current_evidence": {
            "confirmed_matched_count": confirmed_count,
            "rejected_matched_count": rejected_count,
            "configured_minimum_per_group": minimum_per_group,
            "remaining_confirmed_matches_required": remaining_confirmed,
            "remaining_rejected_matches_required": remaining_rejected,
            "pnl_separation": separation,
            "match_coverage_rate": coverage,
            "relationship_consistent_across_runs": consistency,
            "unmatched_data_warning": unmatched_warning,
            "outlier_assessment": {
                "status": "insufficient_trade_distribution",
                "not_dominated_by_one_trade": False,
                "reason": (
                    "Aggregate matched counts and PnL do not prove that separation is "
                    "not dominated by a single trade."
                ),
            },
        },
        "evidence_progress": {
            "confirmed_percentage": progress_confirmed,
            "rejected_percentage": progress_rejected,
            "review_gate_not_profitability_guarantee": True,
        },
        "estimated_runs_remaining": {
            "confirmed": _xgb_runs_remaining(
                remaining_confirmed,
                historical_confirmed,
                historical_runs,
            ),
            "rejected": _xgb_runs_remaining(
                remaining_rejected,
                historical_rejected,
                historical_runs,
            ),
            "basis": "historical dedicated xgboost_shadow_outcome matched trades per run",
            "approximate": True,
        },
        "proposal": [
            "continue only xgboost_shadow_outcome",
            "collect repeated paper runs",
            "maintain signal_id lineage",
            "do not enable blocking",
            "do not lower the evidence gate merely to obtain approval",
        ],
        "acceptance_gate": {
            "checks": gate_checks,
            "gate_satisfied": all(gate_checks.values()),
            "note": "The configured count is an evidence gate, not a profitability guarantee.",
        },
        "verdict": "continue_shadow_outcome_collection",
        "blocking_status": "not_approved_for_blocking",
        "live_approval": False,
    }


def _survival_specification(
    phase17: Dict[str, Any],
    target_exit_rate: float,
) -> Dict[str, Any]:
    current = phase17.get("survival_exit_analysis")
    current = current if isinstance(current, dict) else {}
    simulations = current.get("threshold_simulations")
    simulations = simulations if isinstance(simulations, dict) else {}
    selected = simulations.get(f"{target_exit_rate:.12g}")
    selected = selected if isinstance(selected, dict) else {}
    return {
        "problem_statement": {
            "risk_scores_concentrated_near_one": bool(
                current.get("score_saturation_detected")
            ),
            "configured_target_exit_rate": target_exit_rate,
            "target_rate_threshold": selected.get("threshold"),
            "target_rate_achieved": selected.get("achieved_rate"),
            "extreme_threshold_required": (
                _as_number(selected.get("threshold")) is not None
                and _as_number(selected.get("threshold")) >= 0.99
            ),
            "raw_scores_are_calibrated_probabilities": False,
            "summary": (
                "Risk scores are concentrated near 1.0. Numeric rate targets are only "
                "reachable at extreme thresholds, so raw scores must not be interpreted "
                "as calibrated probabilities."
            ),
        },
        "calibration_candidates": [
            "isotonic calibration",
            "logistic/Platt-style calibration",
            "empirical percentile mapping",
            "time-split validation",
        ],
        "required_outcome_definition": [
            "define the observed exit event explicitly",
            "define the prediction horizon",
            "avoid future leakage",
            "preserve censoring semantics",
            "do not silently treat every open trade as a failure",
        ],
        "evaluation_metrics_where_supported": [
            "calibration curve",
            "Brier score",
            "expected calibration error",
            "exit-rate stability by time window",
            "score distribution after calibration",
        ],
        "acceptance_gates": {
            "requirements": [
                "target exit rate is reachable without an extreme near-1 threshold",
                "score distribution is not saturated",
                "calibrated scores remain stable across time windows",
                "paper shadow evaluation passes before any active exit test",
                "any later active test remains paper-only",
            ],
            "current_artifact_gate_satisfied": False,
        },
        "command_templates": {
            "execute_in_phase18": False,
            "probability_calibration_cli_verified": False,
            "probability_calibration_template": (
                "<FUTURE_PHASE_PLACEHOLDER: implement a time-split Survival "
                "probability-calibration CLI before execution>"
            ),
            "verified_existing_training_cli_reference": (
                'python tools\\train_survival_exit.py --input-csv "<LABELED_DURATION_EVENT_CSV>" '
                '--duration-col "<DURATION_COLUMN>" --event-col "<EVENT_COLUMN>" '
                '--csv-feature-cols "<CURRENT_FEATURE_COLUMNS_IN_EXISTING_ORDER>" '
                '--out "model_artifacts\\survival_exit_<NEW_VERSION>.joblib"'
            ),
            "warning": (
                "The existing command trains a Survival artifact; it is not a verified "
                "probability-calibration command and must not be executed by Phase 18."
            ),
        },
        "verdict": "probability_calibration_spec_ready",
        "artifact_status": "current_artifact_not_approved_active",
        "artifact_modified": False,
        "survival_exit_active": False,
    }


def _top_reason(row: Dict[str, str]) -> str:
    for key in ("top_reason", "reason", "reasons"):
        value = str(row.get(key) or "").strip()
        if value:
            return value.split("|", 1)[0].strip()
    return "unknown"


def _advanced_row_context(log: Dict[str, Any]) -> Dict[str, Any]:
    rows = log.get("rows") if isinstance(log.get("rows"), list) else []
    columns = set(log.get("columns") or [])
    supported = bool(rows and {"would_block", "top_reason"}.issubset(columns))
    if not supported:
        return {
            "supported": False,
            "row_count": len(rows),
            "columns": sorted(columns),
            "simulation_status": "insufficient_row_context",
            "reason": "would_block and top_reason columns with data are required",
            "top_reasons": {},
            "observed_would_block_rate": None,
            "counterfactual_rates": {},
        }
    total = len(rows)
    blocked = [row for row in rows if _truthy(row.get("would_block"))]
    reasons = Counter(_top_reason(row) for row in rows)

    def rate_after_excluding(excluded: set[str]) -> float:
        remaining = sum(
            1
            for row in blocked
            if _top_reason(row) not in excluded
        )
        return remaining / total

    each = {
        reason: rate_after_excluding({reason})
        for reason in reasons
        if reason not in {"normal", "unknown", ""}
    }
    return {
        "supported": True,
        "row_count": total,
        "columns": sorted(columns),
        "simulation_status": "approximate_top_reason_counterfactual",
        "top_reasons": dict(reasons.most_common()),
        "observed_would_block_count": len(blocked),
        "observed_would_block_rate": len(blocked) / total,
        "counterfactual_rates": {
            "exclude_max_open_positions_limit": rate_after_excluding(
                {"max_open_positions_limit"}
            ),
            "exclude_consecutive_losses_limit": rate_after_excluding(
                {"consecutive_losses_limit"}
            ),
            "exclude_each_reason": each,
            "exclude_both_dominant_rules": rate_after_excluding(
                {"max_open_positions_limit", "consecutive_losses_limit"}
            ),
        },
        "limitation": (
            "Approximate top-reason subtraction only; one top reason per row does not "
            "capture secondary-rule interactions or resulting trade PnL."
        ),
    }


def _candidate_simulation(
    row_context: Dict[str, Any],
    counterfactual_key: Optional[str],
) -> Dict[str, Any]:
    if not row_context["supported"]:
        return {
            "simulation_status": "insufficient_row_context",
            "would_block_rate": None,
            "pnl_effect": "not_available_without_matched_trade_outcomes",
        }
    if counterfactual_key is None:
        return {
            "simulation_status": "observed_current_reference",
            "would_block_rate": row_context["observed_would_block_rate"],
            "pnl_effect": "not_available_without_matched_trade_outcomes",
        }
    value = row_context["counterfactual_rates"].get(counterfactual_key)
    if value is None:
        return {
            "simulation_status": "insufficient_row_context",
            "would_block_rate": None,
            "pnl_effect": "not_available_without_matched_trade_outcomes",
        }
    return {
        "simulation_status": "approximate_top_reason_counterfactual",
        "would_block_rate": value,
        "pnl_effect": "not_available_without_matched_trade_outcomes",
        "does_not_simulate_the_proposed_parameter_value_exactly": True,
        "limitation": row_context["limitation"],
    }


def _advanced_risk_proposal(
    phase17: Dict[str, Any],
    log: Dict[str, Any],
) -> Dict[str, Any]:
    current = phase17.get("advanced_risk_analysis")
    current = current if isinstance(current, dict) else {}
    row_context = _advanced_row_context(log)
    current_reference = {
        "source": "verified repository configuration reference; runtime .env not read",
        "ADVANCED_RISK_ACTIVE": False,
        "ADVANCED_RISK_MAX_DAILY_LOSS_PCT": 3.0,
        "ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES": 3,
        "ADVANCED_RISK_MAX_OPEN_POSITIONS": 1,
        "ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT": 100.0,
        "ADVANCED_RISK_VOLATILITY_GUARD_MULT": 2.0,
    }
    candidates = [
        {
            "id": "set_a",
            "name": "current controls reference",
            "settings_different_from_current": {},
            "daily_loss_protection_retained": True,
            "simulation": _candidate_simulation(row_context, None),
        },
        {
            "id": "set_b",
            "name": "relaxed max-open-position constraint",
            "settings_different_from_current": {
                "ADVANCED_RISK_MAX_OPEN_POSITIONS": {
                    "current_reference": 1,
                    "proposed": 2,
                }
            },
            "daily_loss_protection_retained": True,
            "simulation": _candidate_simulation(
                row_context,
                "exclude_max_open_positions_limit",
            ),
        },
        {
            "id": "set_c",
            "name": "relaxed consecutive-loss threshold",
            "settings_different_from_current": {
                "ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES": {
                    "current_reference": 3,
                    "proposed": 4,
                }
            },
            "daily_loss_protection_retained": True,
            "simulation": _candidate_simulation(
                row_context,
                "exclude_consecutive_losses_limit",
            ),
        },
        {
            "id": "set_d",
            "name": "adjusted consecutive-loss cooldown/reset behavior",
            "settings_different_from_current": {
                "CONSECUTIVE_LOSS_RESET_OR_COOLDOWN_POLICY": {
                    "current_reference": "no verified Advanced Risk setting exists",
                    "proposed": "<FUTURE_IMPLEMENTATION_PLACEHOLDER>",
                }
            },
            "daily_loss_protection_retained": True,
            "simulation": {
                "simulation_status": "insufficient_row_context",
                "would_block_rate": None,
                "reason": (
                    "Current rows do not contain reset/cooldown state transitions, and "
                    "the repository has no verified Advanced Risk cooldown setting."
                ),
                "pnl_effect": "not_available_without_matched_trade_outcomes",
            },
        },
        {
            "id": "set_e",
            "name": "both dominant rules relaxed",
            "settings_different_from_current": {
                "ADVANCED_RISK_MAX_OPEN_POSITIONS": {
                    "current_reference": 1,
                    "proposed": 2,
                },
                "ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES": {
                    "current_reference": 3,
                    "proposed": 4,
                },
            },
            "daily_loss_protection_retained": True,
            "simulation": _candidate_simulation(
                row_context,
                "exclude_both_dominant_rules",
            ),
        },
        {
            "id": "set_f",
            "name": "daily-loss protection retained; dominant entry rules shadow-only",
            "settings_different_from_current": {
                "dominant_entry_rule_enforcement": {
                    "current_reference": "shadow-only Phase 18 safety lock",
                    "proposed": "remain shadow-only",
                }
            },
            "daily_loss_protection_retained": True,
            "daily_loss_reference_pct": 3.0,
            "simulation": {
                "simulation_status": "insufficient_row_context",
                "would_block_rate": None,
                "reason": (
                    "The log cannot simulate alternative enforcement interactions while "
                    "retaining daily-loss protection."
                ),
                "pnl_effect": "not_available_without_matched_trade_outcomes",
            },
        },
    ]
    return {
        "current_reference": current_reference,
        "phase17_findings": {
            "max_open_positions_dominates": bool(
                current.get("max_open_positions_dominates")
            ),
            "consecutive_losses_dominates": bool(
                current.get("consecutive_losses_dominates")
            ),
            "multiple_rules_too_strict": bool(
                current.get("multiple_rules_too_strict")
            ),
            "needs_rule_calibration": bool(
                current.get("needs_rule_calibration")
            ),
        },
        "row_level_context": row_context,
        "candidate_rule_sets": candidates,
        "acceptance_gates": {
            "requirements": [
                "would-block rate is materially below current levels",
                "rules do not block nearly all decisions",
                "daily-loss safety is not silently removed",
                "paper shadow test passes",
                "no actual intervention occurs during shadow validation",
                "ADVANCED_RISK_ACTIVE remains false",
            ],
            "current_gate_satisfied": False,
        },
        "verdict": "offline_rule_candidates_ready",
        "active_status": "advanced_risk_active_not_approved",
        "advanced_risk_active": False,
        "pnl_improvement_claim": False,
    }


def _combined_validation_plan() -> Dict[str, Any]:
    return {
        "stages": [
            {
                "stage": 1,
                "status": "approved_for_paper_data_collection",
                "activities": [
                    "baseline paper collection",
                    "XGBoost shadow outcome collection",
                ],
            },
            {
                "stage": 2,
                "status": "future_artifact_or_rule_phase_required",
                "activities": [
                    (
                        "new Isolation Forest artifact in shadow mode only after a "
                        "separate retraining phase"
                    ),
                    (
                        "calibrated Survival artifact in shadow mode only after a "
                        "separate calibration phase"
                    ),
                    "Advanced Risk candidate rules in shadow mode only",
                ],
            },
            {
                "stage": 3,
                "status": "conditional_paper_shadow_validation",
                "activities": [
                    "run combined shadow for 60 minutes",
                    "verify every module log",
                    "verify signal_id lineage",
                    "verify no actual intervention",
                    "regenerate Phase 15, Phase 16, and Phase 17 reports",
                ],
            },
            {
                "stage": 4,
                "status": "not_currently_approved",
                "activities": [
                    (
                        "only after evidence gates pass, consider isolated paper-only "
                        "active/blocking validation"
                    ),
                    "never grant live approval automatically",
                ],
            },
        ],
        "live_approval_automatic": False,
        "paper_only": True,
    }


def _proposed_commands() -> Dict[str, Any]:
    command_values = list(SAFE_EXPERIMENT_COMMANDS.values())
    prohibited_tokens = [
        "-Mode xgboost_blocking",
        "-Mode iforest_blocking",
        "-Mode survival_active",
        "-Mode advanced_risk_active",
        "PLACE_REAL_ORDERS=true",
        "mainnet",
        "live trading",
    ]
    return {
        "approved_now": {
            "baseline": SAFE_EXPERIMENT_COMMANDS["baseline"],
            "xgboost_shadow_outcome": SAFE_EXPERIMENT_COMMANDS[
                "xgboost_shadow_outcome"
            ],
        },
        "conditional_after_future_offline_implementation": {
            "combined_shadow": SAFE_EXPERIMENT_COMMANDS["combined_shadow"],
        },
        "prohibited_command_categories": [
            "live trading",
            "real orders",
            "mainnet",
            "XGBoost blocking",
            "Isolation Forest blocking",
            "Survival active",
            "Advanced Risk active",
        ],
        "prohibited_commands_generated": any(
            token.lower() in command.lower()
            for token in prohibited_tokens
            for command in command_values
        ),
        "all_commands_are_paper_matrix_commands": True,
    }


def _evidence_gate_matrix(
    baseline: Dict[str, Any],
    isolation: Dict[str, Any],
    xgboost: Dict[str, Any],
    survival: Dict[str, Any],
    advanced: Dict[str, Any],
    combined_evidence: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    combined_runs = _as_int(combined_evidence.get("number_of_runs"))
    combined_verdict = str(
        combined_evidence.get("verdict") or "integration_incomplete"
    )
    baseline_current = baseline["current_evidence"]
    xgb_current = xgboost["current_evidence"]
    return {
        "baseline": {
            "current_status": baseline_current["verdict"],
            "evidence_available": {
                "runs": baseline_current["number_of_runs"],
                "closed_trades": baseline_current["closed_trades"],
            },
            "evidence_required": (
                "configured closed-trade minimum, multiple independent windows, "
                "stable run-level performance, clean lineage, and no outlier reliance"
            ),
            "gate_satisfied": baseline["acceptance_gate"]["gate_satisfied"],
            "allowed_next_action": "continue paper-only baseline collection",
            "prohibited_action": "profitability claim or live trading",
            "reason": "baseline evidence remains inconsistent and below the review minimum",
        },
        "isolation_forest": {
            "current_status": isolation["artifact_status"],
            "evidence_available": isolation["problem_statement"],
            "evidence_required": (
                "separate retraining phase plus low-single-digit shadow block rate, "
                "resolved score saturation, and safety tests"
            ),
            "gate_satisfied": False,
            "allowed_next_action": "prepare retraining inputs and experiment design only",
            "prohibited_action": "retrain now, overwrite artifact, or enable blocking",
            "reason": "current artifact produces saturated near-universal anomaly behavior",
        },
        "xgboost": {
            "current_status": xgboost["blocking_status"],
            "evidence_available": {
                "confirmed_matched": xgb_current["confirmed_matched_count"],
                "rejected_matched": xgb_current["rejected_matched_count"],
                "pnl_separation": xgb_current["pnl_separation"],
            },
            "evidence_required": (
                "configured matched minimum in both groups, stable positive separation, "
                "multiple runs, understood match coverage, and complete lineage"
            ),
            "gate_satisfied": xgboost["acceptance_gate"]["gate_satisfied"],
            "allowed_next_action": "continue xgboost_shadow_outcome paper collection",
            "prohibited_action": "XGBoost blocking or a lowered evidence gate",
            "reason": "matched outcome groups remain below the configured review threshold",
        },
        "survival_exit": {
            "current_status": survival["artifact_status"],
            "evidence_available": survival["problem_statement"],
            "evidence_required": (
                "separate probability-calibration phase, unsaturated calibrated scores, "
                "time-window stability, and paper shadow validation"
            ),
            "gate_satisfied": False,
            "allowed_next_action": "prepare probability-calibration design only",
            "prohibited_action": "activate exits or overwrite the artifact",
            "reason": "raw near-1 scores are not calibrated probabilities",
        },
        "advanced_risk": {
            "current_status": advanced["active_status"],
            "evidence_available": {
                "phase17_findings": advanced["phase17_findings"],
                "row_context_supported": advanced["row_level_context"]["supported"],
            },
            "evidence_required": (
                "materially lower shadow block rate, daily-loss preservation, "
                "no actual intervention, and paper validation"
            ),
            "gate_satisfied": False,
            "allowed_next_action": "compare candidate rule sets offline and in shadow",
            "prohibited_action": "enable ADVANCED_RISK_ACTIVE",
            "reason": "dominant rules require calibration and counterfactuals do not prove PnL",
        },
        "combined_shadow": {
            "current_status": combined_verdict,
            "evidence_available": {
                "runs": combined_runs,
                "actual_behavior_counts": combined_evidence.get(
                    "actual_behavior_counts", {}
                ),
                "signal_id_coverage": combined_evidence.get(
                    "signal_id_coverage", {}
                ),
            },
            "evidence_required": (
                "repeat combined paper shadow after future artifact/rule changes with "
                "all logs, complete lineage, and zero actual interventions"
            ),
            "gate_satisfied": combined_verdict == "integration_passed",
            "allowed_next_action": "continued paper shadow integration only",
            "prohibited_action": "automatic active, blocking, or live approval",
            "reason": (
                "integration evidence is valid for safety isolation, not profitability"
            ),
        },
    }


def _final_recommendation() -> Dict[str, Any]:
    return {
        "priority_order": [
            "Continue baseline paper evidence collection.",
            (
                "Continue XGBoost shadow outcome until at least the configured matched "
                "count exists in both groups."
            ),
            "Prepare Isolation Forest retraining in a future separate phase.",
            "Prepare Survival probability calibration in a future separate phase.",
            "Evaluate Advanced Risk candidate rules offline.",
            "Do not overwrite current artifacts.",
            "Keep all experimental modules inactive or shadow-only.",
            "Re-run paper shadow validation after any future artifact or rule change.",
            (
                "No live, mainnet, testnet real orders, real orders, "
                "or PLACE_REAL_ORDERS."
            ),
        ],
        "review_gates_are_profitability_guarantees": False,
        "artifact_changes_performed": False,
        "trading_behavior_changes_performed": False,
        "final_verdict": "calibration_proposals_ready_paper_only",
    }


def summarize_offline_calibration_proposals(
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    xgb_min_matched_per_group: int = 30,
    baseline_min_closed_trades: int = 100,
    if_target_block_rate: float = 0.05,
    survival_target_exit_rate: float = 0.10,
) -> Dict[str, Any]:
    """Build the Phase 18 report without mutating trading or artifact state."""

    if xgb_min_matched_per_group <= 0:
        raise ValueError("xgb_min_matched_per_group must be positive")
    if baseline_min_closed_trades <= 0:
        raise ValueError("baseline_min_closed_trades must be positive")
    if not 0.0 < if_target_block_rate < 1.0:
        raise ValueError("if_target_block_rate must be between 0 and 1")
    if not 0.0 < survival_target_exit_rate < 1.0:
        raise ValueError("survival_target_exit_rate must be between 0 and 1")

    reports_path = Path(reports_dir)
    logs_path = Path(logs_dir)
    direct = _discover_direct_inputs(reports_path)
    logs, log_errors = _read_current_logs(logs_path)
    phase17, phase17_meta = _load_phase17_or_reconstruct(
        reports_path,
        logs_path,
        xgb_min_matched_per_group,
        if_target_block_rate,
        survival_target_exit_rate,
    )
    inventory = _input_inventory(
        reports_path,
        logs_path,
        phase17,
        phase17_meta,
        direct,
        logs,
        log_errors,
    )
    baseline = _baseline_proposal(
        phase17,
        direct,
        baseline_min_closed_trades,
    )
    isolation = _isolation_forest_specification(
        phase17,
        if_target_block_rate,
    )
    xgboost = _xgboost_proposal(
        phase17,
        xgb_min_matched_per_group,
    )
    survival = _survival_specification(
        phase17,
        survival_target_exit_rate,
    )
    advanced = _advanced_risk_proposal(
        phase17,
        logs["advanced_risk"],
    )
    combined_evidence = phase17.get("combined_shadow_integration")
    combined_evidence = (
        combined_evidence if isinstance(combined_evidence, dict) else {}
    )
    return {
        "input_evidence_inventory": inventory,
        "global_safety_gate": _global_safety_gate(),
        "baseline_evidence_proposal": baseline,
        "isolation_forest_retraining_specification": isolation,
        "xgboost_evidence_collection_proposal": xgboost,
        "survival_probability_calibration_specification": survival,
        "advanced_risk_parameter_proposal": advanced,
        "combined_validation_plan": _combined_validation_plan(),
        "proposed_experiment_commands": _proposed_commands(),
        "evidence_gate_matrix": _evidence_gate_matrix(
            baseline,
            isolation,
            xgboost,
            survival,
            advanced,
            combined_evidence,
        ),
        "final_recommendation": _final_recommendation(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def format_text_summary(summary: Dict[str, Any]) -> str:
    """Render the Phase 18 report for terminal review."""

    inventory = summary["input_evidence_inventory"]
    safety = summary["global_safety_gate"]
    baseline = summary["baseline_evidence_proposal"]
    isolation = summary["isolation_forest_retraining_specification"]
    xgboost = summary["xgboost_evidence_collection_proposal"]
    survival = summary["survival_probability_calibration_specification"]
    advanced = summary["advanced_risk_parameter_proposal"]
    plan = summary["combined_validation_plan"]
    commands = summary["proposed_experiment_commands"]
    matrix = summary["evidence_gate_matrix"]
    final = summary["final_recommendation"]

    lines = [
        "Phase 18: Offline Calibration Proposals and Evidence Gates",
        "",
        "A. Input and evidence inventory",
        f"  phase17_report_status={_fmt(inventory['phase17_report_status'])}",
        f"  preferred_evidence_source={inventory['preferred_evidence_source']}",
        f"  fallback_reports_used={_fmt(inventory['fallback_reports_used'])}",
        f"  logs_used={_fmt(inventory['logs_used'])}",
        f"  report_timestamps={_fmt(inventory['report_timestamps'])}",
        f"  duplicate_inputs_skipped={_fmt(inventory['duplicate_inputs_skipped'])}",
        f"  malformed_inputs_skipped={_fmt(inventory['malformed_inputs_skipped'])}",
        f"  missing_inputs={_fmt(inventory['missing_inputs'])}",
        f"  read_errors={_fmt(inventory['read_errors'])}",
        "",
        "B. Global safety gate",
        (
            f"  paper_only={_fmt(safety['paper_only'])} "
            f"no_mainnet={_fmt(safety['no_mainnet'])} "
            f"no_testnet_real_orders={_fmt(safety['no_testnet_real_orders'])}"
        ),
        (
            "  place_real_orders_must_remain_false="
            f"{_fmt(safety['place_real_orders_must_remain_false'])} "
            f"active_modules_allowed={_fmt(safety['active_modules_allowed'])} "
            f"blocking_modules_allowed={_fmt(safety['blocking_modules_allowed'])}"
        ),
        f"  status={safety['status']}; unlock={safety['unlock_condition']}",
        "",
        "C. Baseline evidence proposal",
        f"  current_evidence={_fmt(baseline['current_evidence'])}",
        f"  evidence_gaps={_fmt(baseline['evidence_gaps'])}",
        f"  proposal={_fmt(baseline['proposal'])}",
        f"  acceptance_gate={_fmt(baseline['acceptance_gate'])}",
        f"  verdict: {baseline['verdict']} (profitability claim not allowed)",
        "",
        "D. Isolation Forest retraining specification",
        f"  problem_statement={_fmt(isolation['problem_statement'])}",
        f"  required_training_data={_fmt(isolation['required_training_data'])}",
        f"  proposed_experiment_grid={_fmt(isolation['proposed_experiment_grid'])}",
        f"  offline_acceptance_gates={_fmt(isolation['offline_acceptance_gates'])}",
        f"  training_command_template={_fmt(isolation['training_command_template'])}",
        (
            f"  verdict: {isolation['verdict']}; "
            f"artifact_status={isolation['artifact_status']}"
        ),
        "",
        "E. XGBoost evidence collection proposal",
        f"  current_evidence={_fmt(xgboost['current_evidence'])}",
        f"  evidence_progress={_fmt(xgboost['evidence_progress'])}",
        f"  estimated_runs_remaining={_fmt(xgboost['estimated_runs_remaining'])}",
        f"  proposal={_fmt(xgboost['proposal'])}",
        f"  acceptance_gate={_fmt(xgboost['acceptance_gate'])}",
        (
            f"  verdict: {xgboost['verdict']}; "
            f"blocking_status={xgboost['blocking_status']}"
        ),
        "",
        "F. Survival probability-calibration specification",
        f"  problem_statement={_fmt(survival['problem_statement'])}",
        f"  calibration_candidates={_fmt(survival['calibration_candidates'])}",
        (
            "  required_outcome_definition="
            f"{_fmt(survival['required_outcome_definition'])}"
        ),
        (
            "  evaluation_metrics_where_supported="
            f"{_fmt(survival['evaluation_metrics_where_supported'])}"
        ),
        f"  acceptance_gates={_fmt(survival['acceptance_gates'])}",
        f"  command_templates={_fmt(survival['command_templates'])}",
        (
            f"  verdict: {survival['verdict']}; "
            f"artifact_status={survival['artifact_status']}"
        ),
        "",
        "G. Advanced Risk parameter proposal",
        f"  current_reference={_fmt(advanced['current_reference'])}",
        f"  phase17_findings={_fmt(advanced['phase17_findings'])}",
        f"  row_level_context={_fmt(advanced['row_level_context'])}",
        f"  candidate_rule_sets={_fmt(advanced['candidate_rule_sets'])}",
        f"  acceptance_gates={_fmt(advanced['acceptance_gates'])}",
        (
            f"  verdict: {advanced['verdict']}; "
            f"active_status={advanced['active_status']}"
        ),
        "",
        "H. Combined validation plan",
    ]
    for item in plan["stages"]:
        lines.append(
            f"  stage_{item['stage']} status={item['status']} "
            f"activities={_fmt(item['activities'])}"
        )
    lines.extend(
        [
            "",
            "I. Proposed experiment commands",
            f"  baseline: {commands['approved_now']['baseline']}",
            (
                "  xgboost_shadow_outcome: "
                f"{commands['approved_now']['xgboost_shadow_outcome']}"
            ),
            (
                "  combined_shadow_after_offline_changes: "
                f"{commands['conditional_after_future_offline_implementation']['combined_shadow']}"
            ),
            (
                "  prohibited_commands_generated="
                f"{_fmt(commands['prohibited_commands_generated'])}"
            ),
            "",
            "J. Evidence-gate matrix",
            (
                "  component | current_status | gate_satisfied | allowed_next_action "
                "| prohibited_action | reason"
            ),
        ]
    )
    for component, item in matrix.items():
        lines.append(
            f"  {component} | {item['current_status']} | "
            f"{_fmt(item['gate_satisfied'])} | {item['allowed_next_action']} | "
            f"{item['prohibited_action']} | {item['reason']}"
        )
        lines.append(
            f"    evidence_available={_fmt(item['evidence_available'])}; "
            f"evidence_required={_fmt(item['evidence_required'])}"
        )
    lines.extend(["", "K. Final Phase 18 recommendation"])
    for index, recommendation in enumerate(final["priority_order"], start=1):
        lines.append(f"  {index}. {recommendation}")
    lines.extend(
        [
            (
                "  artifact_changes_performed="
                f"{_fmt(final['artifact_changes_performed'])}; "
                "trading_behavior_changes_performed="
                f"{_fmt(final['trading_behavior_changes_performed'])}"
            ),
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


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _review_rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not 0.0 < rate < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return rate


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create read-only Phase 18 calibration proposals, evidence gates, "
            "and a paper-only validation plan."
        )
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument(
        "--xgb-min-matched-per-group",
        type=_positive_int,
        default=30,
        help="Evidence review threshold per group; not a profitability guarantee.",
    )
    parser.add_argument(
        "--baseline-min-closed-trades",
        type=_positive_int,
        default=100,
        help="Baseline evidence review threshold; not a profitability guarantee.",
    )
    parser.add_argument("--if-target-block-rate", type=_review_rate, default=0.05)
    parser.add_argument("--survival-target-exit-rate", type=_review_rate, default=0.10)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the JSON proposal report to --json-out.",
    )
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_offline_calibration_proposals(
        reports_dir=args.reports_dir,
        logs_dir=args.logs_dir,
        xgb_min_matched_per_group=args.xgb_min_matched_per_group,
        baseline_min_closed_trades=args.baseline_min_closed_trades,
        if_target_block_rate=args.if_target_block_rate,
        survival_target_exit_rate=args.survival_target_exit_rate,
    )
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
