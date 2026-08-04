"""Tests for the Phase 17 read-only offline calibration sweep."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.offline_calibration_sweep import (
    format_text_summary,
    summarize_offline_calibration,
    write_json_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "offline_calibration_sweep.py"

ZERO_ACTUALS = {
    "actually_blocked": 0,
    "actually_rejected": 0,
    "actually_exited": 0,
    "actually_paused": 0,
    "actually_reduced": 0,
}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _base_unified() -> dict[str, Any]:
    return {
        "safety": {
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": dict(ZERO_ACTUALS),
            "shadow_only_warning": False,
            "warnings": [],
        },
        "isolation_forest": {
            "file_status": "missing",
            "total_rows": 0,
            "would_block_count": 0,
            "actually_blocked_count": 0,
            "would_block_rate": 0.0,
            "actual_block_rate": 0.0,
            "average_anomaly_score": None,
        },
        "xgboost": {
            "file_status": "missing",
            "total_rows": 0,
            "would_confirm_count": 0,
            "would_reject_count": 0,
            "would_reject_rate": 0.0,
            "reject_reason_counts": {},
            "average_confidence_allowed": None,
            "average_confidence_rejected": None,
        },
        "survival_exit": {
            "file_status": "missing",
            "total_rows": 0,
            "would_exit_early_count": 0,
            "actually_exited_count": 0,
            "would_exit_rate": 0.0,
            "actual_exit_rate": 0.0,
            "average_risk_score": None,
        },
        "advanced_risk": {
            "file_status": "missing",
            "total_rows": 0,
            "would_block_count": 0,
            "actually_blocked_count": 0,
            "would_block_rate": 0.0,
            "actual_block_rate": 0.0,
            "would_pause_count": 0,
            "actually_paused_count": 0,
            "would_reduce_size_count": 0,
            "actually_reduced_count": 0,
            "top_reasons": {},
        },
        "trade_lineage": {
            "paper_trade_rows": 0,
            "paper_trade_rows_with_signal_id": 0,
            "closed_trade_rows": 0,
            "closed_trade_rows_with_signal_id": 0,
        },
        "paper_pnl": {
            "closed_trade_count": 0,
            "total_pnl": 0.0,
            "average_pnl": None,
            "win_rate": None,
        },
        "xgboost_outcome": {
            "would_confirm_matched_count": 0,
            "would_confirm_average_pnl": None,
            "would_confirm_win_rate": None,
            "would_reject_matched_count": 0,
            "would_reject_average_pnl": None,
            "would_reject_win_rate": None,
            "matched_closed_trade_count": 0,
            "unmatched_decision_rows": 0,
        },
    }


def _write_unified(
    reports_dir: Path,
    mode: str,
    timestamp: str,
    **sections: dict[str, Any],
) -> Path:
    payload = _base_unified()
    payload.update(sections)
    path = _write_json(
        reports_dir / f"matrix_{mode}_{timestamp}_unified.json",
        payload,
    )
    closed = int(payload.get("paper_pnl", {}).get("closed_trade_count") or 0)
    matched = int(
        payload.get("xgboost_outcome", {}).get("matched_closed_trade_count") or 0
    )
    strategy_outcome = (
        (mode in {"baseline", "combined_shadow"} and closed > 0)
        or (mode == "xgboost_shadow_outcome" and matched > 0)
    )
    _write_index(
        reports_dir,
        mode,
        timestamp,
        duration_minutes=60 if strategy_outcome else 5,
        report_kinds=("unified",),
    )
    return path


def _write_shadow_summary(
    reports_dir: Path,
    mode: str,
    timestamp: str,
    **sections: dict[str, Any],
) -> Path:
    payload = {
        "isolation_forest": _base_unified()["isolation_forest"],
        "xgboost_signal": _base_unified()["xgboost"],
        "survival_exit": _base_unified()["survival_exit"],
        "advanced_risk": _base_unified()["advanced_risk"],
    }
    payload.update(sections)
    return _write_json(
        reports_dir / f"matrix_{mode}_{timestamp}_shadow_summary.json",
        payload,
    )


def _write_index(
    reports_dir: Path,
    mode: str,
    timestamp: str,
    *,
    exit_status: int = 0,
    duration_minutes: int = 60,
    evidence_valid: bool = True,
    stale_entry_count: int = 0,
    report_kinds: tuple[str, ...] = ("unified",),
) -> Path:
    return _write_json(
        reports_dir / f"matrix_index_{timestamp}.json",
        {
            "matrix_timestamp": timestamp,
            "dry_run": False,
            "duration_minutes": duration_minutes,
            "runs": [
                {
                    "mode": mode,
                    "run_timestamp": timestamp,
                    "run_started_utc": "2026-07-01T00:00:00Z",
                    "finished_at": "2026-07-01T01:00:00Z",
                    "duration_minutes": duration_minutes,
                    "exit_status": exit_status,
                    "stale_entry_guard_checked": True,
                    "stale_entry_count": stale_entry_count,
                    "stale_entry_signal_ids": [],
                    "evidence_valid": evidence_valid,
                    "report_paths": {
                        kind: str(
                            reports_dir
                            / f"matrix_{mode}_{timestamp}_{kind}.json"
                        )
                        for kind in report_kinds
                    },
                }
            ],
        },
    )


def _outcome_stats(count: int, total_pnl: float, win_rate: float | None) -> dict[str, Any]:
    return {
        "count": count,
        "total_pnl": total_pnl,
        "average_pnl": None if count == 0 else total_pnl / count,
        "win_rate": win_rate,
    }


def _write_xgb_run(
    reports_dir: Path,
    mode: str,
    timestamp: str,
    *,
    total_rows: int,
    would_confirm_count: int,
    would_reject_count: int,
    confirmed: tuple[int, float, float | None],
    rejected: tuple[int, float, float | None],
    unmatched_decision_count: int,
    reject_reasons: dict[str, int] | None = None,
    average_allowed_confidence: float | None = None,
    average_rejected_confidence: float | None = None,
) -> tuple[Path, Path]:
    confirmed_stats = _outcome_stats(*confirmed)
    rejected_stats = _outcome_stats(*rejected)
    matched_count = confirmed[0] + rejected[0]
    reject_reasons = reject_reasons or {}
    xgboost = {
        "file_status": "ok",
        "total_rows": total_rows,
        "would_confirm_count": would_confirm_count,
        "would_reject_count": would_reject_count,
        "would_reject_rate": 0.0 if total_rows == 0 else would_reject_count / total_rows,
        "actual_reject_rate": 0.0,
        "reject_reason_counts": reject_reasons,
        "average_confidence_allowed": average_allowed_confidence,
        "average_confidence_rejected": average_rejected_confidence,
    }
    outcome = {
        "would_confirm_matched_count": confirmed[0],
        "would_confirm_average_pnl": confirmed_stats["average_pnl"],
        "would_confirm_win_rate": confirmed[2],
        "would_reject_matched_count": rejected[0],
        "would_reject_average_pnl": rejected_stats["average_pnl"],
        "would_reject_win_rate": rejected[2],
        "matched_closed_trade_count": matched_count,
        "unmatched_decision_rows": unmatched_decision_count,
    }
    unified = _write_unified(
        reports_dir,
        mode,
        timestamp,
        xgboost=xgboost,
        xgboost_outcome=outcome,
        **(
            {
                "paper_pnl": {
                    "closed_trade_count": matched_count,
                    "total_pnl": confirmed[1] + rejected[1],
                    "average_pnl": (
                        None
                        if matched_count == 0
                        else (confirmed[1] + rejected[1]) / matched_count
                    ),
                    "win_rate": None,
                },
                "trade_lineage": {
                    "paper_trade_rows": matched_count,
                    "paper_trade_rows_with_signal_id": matched_count,
                    "closed_trade_rows": matched_count,
                    "closed_trade_rows_with_signal_id": matched_count,
                },
            }
            if mode == "combined_shadow"
            else {}
        ),
    )
    audit = _write_json(
        reports_dir / f"matrix_{mode}_{timestamp}_xgboost_audit.json",
        {
            "total_xgboost_rows": total_rows,
            "would_confirm_count": would_confirm_count,
            "would_reject_count": would_reject_count,
            "reject_reason_counts": reject_reasons,
            "average_confidence_allowed": average_allowed_confidence,
            "average_confidence_rejected": average_rejected_confidence,
            "would_confirm_matched_count": confirmed[0],
            "would_confirm_average_pnl": confirmed_stats["average_pnl"],
            "would_confirm_win_rate": confirmed[2],
            "would_reject_matched_count": rejected[0],
            "would_reject_average_pnl": rejected_stats["average_pnl"],
            "would_reject_win_rate": rejected[2],
            "trade_outcome_join": {
                "matched_closed_trade_count": matched_count,
                "unmatched_decision_rows": unmatched_decision_count,
                "matched_closed_trade_pnl": {
                    "allowed": confirmed_stats,
                    "rejected": rejected_stats,
                    "would_confirm": confirmed_stats,
                    "would_reject": rejected_stats,
                },
            },
        },
    )
    return unified, audit


def _names(items: Any) -> set[str]:
    if isinstance(items, dict):
        values = list(items) + list(items.values())
    elif isinstance(items, list):
        values = items
    else:
        values = [items]
    names: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("path") or value.get("file") or value.get("report")
        if value:
            names.add(Path(str(value)).name)
    return names


def test_missing_reports_and_logs_are_reported_safely(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"

    summary = summarize_offline_calibration(reports, logs)

    inventory = summary["input_inventory"]
    assert not inventory["report_files_found"]
    assert not inventory["report_files_used"]
    assert not inventory["row_level_logs_found"]
    assert inventory["duplicate_reports_skipped"] == []
    assert inventory["malformed_reports_skipped"] == []
    assert inventory["missing_inputs"]
    assert inventory["read_errors"] == []
    assert summary["baseline_cross_run_summary"]["number_of_runs"] == 0
    assert summary["final_recommendation"]["final_verdict"] == "paper_only_offline_calibration_required"
    assert "Offline Calibration Sweep" in format_text_summary(summary)


def test_malformed_json_is_skipped_without_hiding_valid_runs(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    reports.mkdir()
    malformed = reports / "matrix_baseline_20260701010101_unified.json"
    malformed.write_text("{not-json", encoding="utf-8")
    non_object = reports / "matrix_iforest_shadow_20260701020202_unified.json"
    non_object.write_text("[]", encoding="utf-8")
    valid = _write_unified(
        reports,
        "baseline",
        "20260701030303",
        paper_pnl={
            "closed_trade_count": 2,
            "total_pnl": 1.0,
            "average_pnl": 0.5,
            "win_rate": 0.5,
        },
    )

    summary = summarize_offline_calibration(reports, logs)

    skipped = _names(summary["input_inventory"]["malformed_reports_skipped"])
    assert malformed.name in skipped
    assert non_object.name in skipped
    assert len(summary["input_inventory"]["read_errors"]) >= 2
    assert valid.name in _names(summary["input_inventory"]["report_files_used"])
    assert summary["baseline_cross_run_summary"]["number_of_runs"] == 1
    assert summary["baseline_cross_run_summary"]["latest_run"]["timestamp"] == "20260701030303"


def test_reports_group_by_mode_and_timestamp_without_double_counting(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    timestamp = "20260702010101"
    _write_unified(
        reports,
        "baseline",
        timestamp,
        paper_pnl={
            "closed_trade_count": 3,
            "total_pnl": 0.3,
            "average_pnl": 0.1,
            "win_rate": 1 / 3,
        },
    )
    _write_shadow_summary(reports, "baseline", timestamp)
    _write_index(reports, "baseline", timestamp)
    _write_unified(reports, "baseline", "20260702020202")
    _write_unified(reports, "iforest_shadow", "20260702030303")
    _write_json(
        reports / "final_experiment_comparison.json",
        {"baseline_performance": {"timestamp": timestamp, "closed_trades": 3, "total_pnl": 0.3}},
    )
    _write_json(
        reports / "calibration_recommendation_report.json",
        {"baseline_strategy_health": {"closed_trades": 3, "total_pnl": 0.3}},
    )

    summary = summarize_offline_calibration(reports, logs)

    grouped = summary["input_inventory"]["run_timestamps_by_mode"]
    assert grouped["baseline"] == [timestamp, "20260702020202"]
    assert grouped["iforest_shadow"] == ["20260702030303"]
    baseline = summary["baseline_cross_run_summary"]
    # The zero-outcome baseline is retained for safety, not PnL aggregation.
    assert baseline["number_of_runs"] == 1
    assert baseline["total_closed_trades"] == 3
    assert summary["input_inventory"]["duplicate_reports_skipped"]


def test_intentional_active_validation_is_separate_from_shadow_safety(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    active_counts = dict(ZERO_ACTUALS)
    active_counts["actually_blocked"] = 4
    _write_unified(
        reports,
        "iforest_blocking",
        "20260703010101",
        safety={
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": active_counts,
            "shadow_only_warning": True,
        },
        isolation_forest={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 10,
            "actually_blocked_count": 4,
            "would_block_rate": 1.0,
            "actual_block_rate": 0.4,
            "average_anomaly_score": -0.5,
        },
    )
    exit_counts = dict(ZERO_ACTUALS)
    exit_counts["actually_exited"] = 2
    _write_unified(
        reports,
        "survival_active",
        "20260703020202",
        safety={
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": exit_counts,
            "shadow_only_warning": True,
        },
        survival_exit={
            "file_status": "ok",
            "total_rows": 8,
            "would_exit_early_count": 8,
            "actually_exited_count": 2,
            "would_exit_rate": 1.0,
            "actual_exit_rate": 0.25,
            "average_risk_score": 0.99,
        },
    )
    _write_unified(reports, "combined_shadow", "20260703030303")

    summary = summarize_offline_calibration(reports, logs)

    classification = summary["run_classification"]
    assert set(classification["shadow_safe_modes"]) == {
        "baseline",
        "iforest_shadow",
        "xgboost_shadow_outcome",
        "survival_shadow",
        "advanced_risk_shadow",
        "combined_shadow",
    }
    assert set(classification["intentional_active_validation_modes"]) == {
        "iforest_blocking",
        "survival_active",
    }
    active_modes = {run["mode"] for run in classification["intentional_active_validation_runs"]}
    assert active_modes == {"iforest_blocking", "survival_active"}
    assert not any(
        run["mode"] in active_modes
        for run in classification["shadow_safety_violations"]
    )
    assert (
        summary["isolation_forest_analysis"]["blocking_validation_runs"]["runs"][0][
            "actual_block_rate"
        ]
        == 0.4
    )
    assert (
        summary["survival_exit_analysis"]["active_validation_runs"]["runs"][0][
            "actual_exit_rate"
        ]
        == 0.25
    )


def test_audit_only_actual_rejections_fail_shadow_safety(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_json(
        reports
        / "matrix_xgboost_shadow_outcome_20260703040404_xgboost_audit.json",
        {
            "total_xgboost_rows": 5,
            "would_confirm_count": 0,
            "would_reject_count": 5,
            "actually_rejected_count": 5,
        },
    )
    _write_index(
        reports,
        "xgboost_shadow_outcome",
        "20260703040404",
        duration_minutes=5,
        report_kinds=("xgboost_audit",),
    )

    classification = summarize_offline_calibration(reports, logs)["run_classification"]

    assert classification["shadow_only_safety_passed"] is False
    assert classification["shadow_safety_violations"][0]["mode"] == (
        "xgboost_shadow_outcome"
    )
    assert classification["shadow_safety_violations"][0][
        "actual_behavior_counts"
    ]["actually_rejected"] == 5


def test_combined_shadow_module_evidence_contributes_to_iforest_and_survival(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    timestamp = "20260703050505"
    _write_unified(
        reports,
        "combined_shadow",
        timestamp,
        isolation_forest={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 10,
            "would_block_rate": 1.0,
            "actual_block_rate": 0.0,
            "average_anomaly_score": -0.5,
        },
        survival_exit={
            "file_status": "ok",
            "total_rows": 10,
            "would_exit_early_count": 10,
            "would_exit_rate": 1.0,
            "actual_exit_rate": 0.0,
            "average_risk_score": 0.99,
        },
    )

    summary = summarize_offline_calibration(reports, logs)
    isolation = summary["isolation_forest_analysis"]
    survival = summary["survival_exit_analysis"]

    assert isolation["number_of_shadow_runs"] == 1
    assert isolation["shadow_runs"]["runs"][0]["mode"] == "combined_shadow"
    assert isolation["consistent_100_percent_anomaly_behavior"] is True
    assert isolation["verdict"] == "unsafe_to_enable"
    assert survival["number_of_shadow_runs"] == 1
    assert survival["would_exit_rate_by_run"][f"combined_shadow:{timestamp}"] == 1.0
    assert survival["probability_calibration_recommended"] is False
    assert survival["verdict"] == "too_aggressive"


def test_multiple_baseline_runs_use_trade_weighted_pnl_and_winners(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "baseline",
        "20260704010101",
        paper_pnl={
            "closed_trade_count": 2,
            "total_pnl": 2.0,
            "average_pnl": 1.0,
            "win_rate": 0.5,
        },
    )
    _write_unified(
        reports,
        "baseline",
        "20260704020202",
        paper_pnl={
            "closed_trade_count": 8,
            "total_pnl": -4.0,
            "average_pnl": -0.5,
            "win_rate": 0.25,
        },
    )

    baseline = summarize_offline_calibration(reports, logs)["baseline_cross_run_summary"]

    assert baseline["number_of_runs"] == 2
    assert baseline["total_closed_trades"] == 10
    assert baseline["total_pnl"] == -2.0
    assert baseline["weighted_average_pnl"] == pytest.approx(-0.2)
    assert baseline["overall_win_rate"] == pytest.approx(0.3)
    assert baseline["minimum_run_pnl"] == -4.0
    assert baseline["maximum_run_pnl"] == 2.0
    assert baseline["positive_run_count"] == 1
    assert baseline["negative_run_count"] == 1
    assert baseline["latest_run"]["closed_trades"] == 8
    assert baseline["sample_size_warning"] is True
    assert baseline["verdict"] == "baseline_inconsistent"


def test_one_short_positive_baseline_run_is_never_called_profitable(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "baseline",
        "20260704030303",
        paper_pnl={
            "closed_trade_count": 6,
            "total_pnl": 0.055706,
            "average_pnl": 0.055706 / 6,
            "win_rate": 1 / 3,
        },
    )

    baseline = summarize_offline_calibration(reports, logs)["baseline_cross_run_summary"]

    assert baseline["verdict"] in {"baseline_unproven", "baseline_weak"}
    assert baseline["sample_size_warning"] is True
    assert "profitable" not in baseline["verdict"]


def test_xgboost_audits_are_deduplicated_and_weighted_by_matched_counts(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705010101",
        total_rows=10,
        would_confirm_count=6,
        would_reject_count=4,
        confirmed=(2, 2.0, 0.5),
        rejected=(1, -1.0, 0.0),
        unmatched_decision_count=7,
        reject_reasons={"low_confidence": 4},
        average_allowed_confidence=0.8,
        average_rejected_confidence=0.3,
    )
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705020202",
        total_rows=20,
        would_confirm_count=10,
        would_reject_count=8,
        confirmed=(8, 0.0, 0.5),
        rejected=(4, -1.0, 0.25),
        unmatched_decision_count=6,
        reject_reasons={"direction_mismatch": 8},
        average_allowed_confidence=0.6,
        average_rejected_confidence=0.5,
    )

    xgb = summarize_offline_calibration(
        reports,
        logs,
        min_xgb_matched_per_group=6,
    )["xgboost_outcome_aggregation"]
    aggregate = xgb["combined_aggregate"]

    assert aggregate["number_of_runs"] == 2
    assert aggregate["total_decision_rows"] == 30
    assert aggregate["would_confirm_count"] == 16
    assert aggregate["would_reject_count"] == 12
    assert aggregate["would_reject_rate"] == pytest.approx(0.4)
    assert aggregate["reject_reasons"] == {
        "low_confidence": 4,
        "direction_mismatch": 8,
    }
    assert aggregate["average_allowed_confidence"] == pytest.approx(0.675)
    assert aggregate["average_rejected_confidence"] == pytest.approx(13 / 30)
    assert aggregate["confirmed"]["matched_count"] == 10
    assert aggregate["confirmed"]["total_pnl"] == 2.0
    assert aggregate["confirmed"]["weighted_average_pnl"] == pytest.approx(0.2)
    assert aggregate["confirmed"]["win_rate"] == pytest.approx(0.5)
    assert aggregate["rejected"]["matched_count"] == 5
    assert aggregate["rejected"]["total_pnl"] == -2.0
    assert aggregate["rejected"]["weighted_average_pnl"] == pytest.approx(-0.4)
    assert aggregate["rejected"]["win_rate"] == pytest.approx(0.2)
    assert aggregate["matched_closed_trade_count"] == 15
    assert aggregate["unmatched_decision_count"] == 13
    assert aggregate["match_coverage_rate"] == pytest.approx(15 / 28)
    assert aggregate["pnl_separation"] == pytest.approx(0.6)
    assert aggregate["rejected_trades_worse_than_confirmed"] is True
    assert aggregate["relationship_consistent_across_runs"] is True
    assert aggregate["minimum_evidence_threshold"]["configured_per_group"] == 6
    assert aggregate["minimum_evidence_threshold"]["satisfied"] is False
    assert aggregate["sample_size_warning"] is True


def test_xgboost_deduplicates_unified_outcomes_and_repeated_summary_inputs(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    timestamp = "20260705030303"
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        timestamp,
        total_rows=10,
        would_confirm_count=6,
        would_reject_count=4,
        confirmed=(3, 0.9, 2 / 3),
        rejected=(2, -0.4, 0.0),
        unmatched_decision_count=5,
    )
    _write_shadow_summary(
        reports,
        "xgboost_shadow_outcome",
        timestamp,
        xgboost_signal={
            "file_status": "ok",
            "total_rows": 10,
            "would_confirm_count": 6,
            "would_reject_count": 4,
        },
    )
    _write_json(
        reports / "final_experiment_comparison.json",
        {
            "xgboost_summary": {
                "source_mode": "xgboost_shadow_outcome",
                "timestamp": timestamp,
                "would_confirm_matched_count": 3,
                "would_reject_matched_count": 2,
            }
        },
    )

    summary = summarize_offline_calibration(reports, logs)
    aggregate = summary["xgboost_outcome_aggregation"]["combined_aggregate"]

    assert aggregate["number_of_runs"] == 1
    assert aggregate["confirmed"]["matched_count"] == 3
    assert aggregate["rejected"]["matched_count"] == 2
    assert summary["input_inventory"]["duplicate_reports_skipped"]


def test_xgboost_separates_dedicated_and_combined_sources(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    for mode, timestamp in (
        ("xgboost_shadow_outcome", "20260705040404"),
        ("combined_shadow", "20260705050505"),
    ):
        _write_xgb_run(
            reports,
            mode,
            timestamp,
            total_rows=10,
            would_confirm_count=5,
            would_reject_count=5,
            confirmed=(2, 0.4, 0.5),
            rejected=(2, -0.2, 0.0),
            unmatched_decision_count=6,
        )

    xgb = summarize_offline_calibration(reports, logs)["xgboost_outcome_aggregation"]

    assert xgb["by_source_mode"]["xgboost_shadow_outcome"]["number_of_runs"] == 1
    assert xgb["by_source_mode"]["combined_shadow"]["number_of_runs"] == 1
    assert xgb["combined_aggregate"]["number_of_runs"] == 2
    assert xgb["combined_aggregate"]["confirmed"]["matched_count"] == 4
    assert xgb["combined_aggregate"]["rejected"]["matched_count"] == 4


def test_xgboost_cross_run_inconsistency_is_explicit(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705060606",
        total_rows=4,
        would_confirm_count=2,
        would_reject_count=2,
        confirmed=(2, 2.0, 1.0),
        rejected=(2, 0.0, 0.0),
        unmatched_decision_count=0,
    )
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705070707",
        total_rows=4,
        would_confirm_count=2,
        would_reject_count=2,
        confirmed=(2, -2.0, 0.0),
        rejected=(2, 1.0, 0.5),
        unmatched_decision_count=0,
    )

    aggregate = summarize_offline_calibration(
        reports,
        logs,
        min_xgb_matched_per_group=1,
    )["xgboost_outcome_aggregation"]["combined_aggregate"]

    assert aggregate["relationship_consistent_across_runs"] is False
    assert aggregate["verdict"] == "inconsistent_across_runs"
    assert aggregate["blocking_candidate_status"] == "not_approved_for_blocking"


def test_xgboost_threshold_is_per_group_and_never_live_approval(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705080808",
        total_rows=10,
        would_confirm_count=5,
        would_reject_count=5,
        confirmed=(3, 0.9, 2 / 3),
        rejected=(3, -0.6, 0.0),
        unmatched_decision_count=4,
    )
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705090909",
        total_rows=10,
        would_confirm_count=5,
        would_reject_count=5,
        confirmed=(3, 0.6, 2 / 3),
        rejected=(3, -0.3, 0.0),
        unmatched_decision_count=4,
    )

    aggregate = summarize_offline_calibration(
        reports,
        logs,
        min_xgb_matched_per_group=3,
    )["xgboost_outcome_aggregation"]["combined_aggregate"]

    assert aggregate["minimum_evidence_threshold"]["satisfied"] is True
    assert aggregate["minimum_evidence_threshold"]["profitability_guarantee"] is False
    assert aggregate["blocking_candidate_status"] == "paper-blocking-candidate"
    assert aggregate["blocking_candidate_status"] != "live-approved"


def test_xgboost_missing_pnl_support_cannot_become_blocking_candidate(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_xgb_run(
        reports,
        "xgboost_shadow_outcome",
        "20260705101010",
        total_rows=2,
        would_confirm_count=1,
        would_reject_count=1,
        confirmed=(1, 0.1, 1.0),
        rejected=(1, -0.1, 0.0),
        unmatched_decision_count=0,
    )
    timestamp = "20260705111111"
    _write_unified(reports, "xgboost_shadow_outcome", timestamp)
    _write_json(
        reports / f"matrix_xgboost_shadow_outcome_{timestamp}_xgboost_audit.json",
        {
            "total_xgboost_rows": 18,
            "would_confirm_count": 9,
            "would_reject_count": 9,
            "would_confirm_matched_count": 9,
            "would_reject_matched_count": 9,
            "trade_outcome_join": {
                "matched_closed_trade_count": 18,
                "unmatched_decision_rows": 0,
            },
        },
    )

    aggregate = summarize_offline_calibration(
        reports,
        logs,
        min_xgb_matched_per_group=5,
    )["xgboost_outcome_aggregation"]["combined_aggregate"]

    # Audit-only counts from an incomplete run cannot inflate strategy evidence.
    assert aggregate["minimum_evidence_threshold"]["satisfied"] is False
    assert aggregate["confirmed"]["pnl_coverage_rate"] == pytest.approx(1.0)
    assert aggregate["rejected"]["pnl_coverage_rate"] == pytest.approx(1.0)
    assert aggregate["confirmed"]["weighted_average_pnl"] == pytest.approx(0.1)
    assert aggregate["rejected"]["weighted_average_pnl"] == pytest.approx(-0.1)
    assert aggregate["pnl_support_warning"] is False
    assert aggregate["blocking_candidate_status"] == "not_approved_for_blocking"


def test_isolation_forest_reports_tied_nearest_achievable_rates(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "iforest_shadow",
        "20260706010101",
        isolation_forest={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 10,
            "actually_blocked_count": 0,
            "would_block_rate": 1.0,
            "actual_block_rate": 0.0,
            "average_anomaly_score": -0.35,
        },
    )
    scores = [-1.0, -1.0, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    _write_csv(
        logs / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score", "would_block", "actually_blocked"],
        [[f"t{i}", score, 1, 0] for i, score in enumerate(scores)],
    )

    analysis = summarize_offline_calibration(
        reports,
        logs,
        target_if_block_rates=(0.25,),
    )["isolation_forest_analysis"]
    row_level = analysis["row_level_analysis"]
    simulation = row_level["threshold_simulations"]["0.25"]

    assert analysis["shadow_runs"]["number_of_runs"] == 1
    assert analysis["shadow_runs"]["runs"][0]["would_block_rate"] == 1.0
    distribution = row_level["score_distribution"]
    assert {
        "count",
        "min",
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "average",
        "unique_score_values",
        "tied_score_frequency",
    } <= set(distribution)
    assert distribution["count"] == 10
    assert distribution["unique_score_values"] == 3
    assert distribution["tied_score_frequency"] > 0
    assert simulation["tied_score_limitation"] is True
    assert simulation["nearest_achievable_below"]["threshold"] == -1.0
    assert simulation["nearest_achievable_below"]["achieved_rate"] == pytest.approx(0.2)
    assert simulation["nearest_achievable_above"]["threshold"] == -0.5
    assert simulation["nearest_achievable_above"]["achieved_rate"] == pytest.approx(0.5)


def test_isolation_forest_saturation_recommends_retraining(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_csv(
        logs / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score", "would_block", "actually_blocked"],
        [[f"t{i}", -0.42, 1, 0] for i in range(20)],
    )

    analysis = summarize_offline_calibration(
        reports,
        logs,
        target_if_block_rates=(0.05,),
    )["isolation_forest_analysis"]
    simulation = analysis["row_level_analysis"]["threshold_simulations"]["0.05"]

    assert simulation["achieved_rate"] != pytest.approx(0.05)
    assert simulation["tied_score_limitation"] is True
    assert analysis["threshold_only_calibration_feasible"] is False
    assert analysis["score_saturation_detected"] is True
    assert analysis["retraining_recommended"] is True
    assert analysis["verdict"] == "retraining_recommended"


def test_archived_shadow_csvs_are_not_automatically_aggregated(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_csv(
        logs / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score"],
        [["current-1", -0.2], ["current-2", -0.1]],
    )
    _write_csv(
        logs / "iforest_archive_20260706" / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score"],
        [[f"archived-{i}", -1.0] for i in range(100)],
    )

    analysis = summarize_offline_calibration(reports, logs)["isolation_forest_analysis"]

    assert analysis["row_level_analysis"]["score_distribution"]["count"] == 2


def test_survival_saturation_and_threshold_ties_require_probability_calibration(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "survival_shadow",
        "20260707010101",
        survival_exit={
            "file_status": "ok",
            "total_rows": 20,
            "would_exit_early_count": 20,
            "actually_exited_count": 0,
            "would_exit_rate": 1.0,
            "actual_exit_rate": 0.0,
            "average_risk_score": 0.999,
        },
    )
    scores = [1.0] * 18 + [0.99] * 2
    _write_csv(
        logs / "survival_exit_shadow.csv",
        ["timestamp", "survival_risk_score", "would_exit_early", "actually_exited"],
        [[f"t{i}", score, 1, 0] for i, score in enumerate(scores)],
    )

    analysis = summarize_offline_calibration(
        reports,
        logs,
        target_survival_exit_rates=(0.1,),
    )["survival_exit_analysis"]
    row_level = analysis["row_level_analysis"]
    simulation = row_level["threshold_simulations"]["0.1"]

    assert analysis["shadow_runs"]["runs"][0]["would_exit_rate"] == 1.0
    distribution = row_level["score_distribution"]
    assert {
        "count",
        "min",
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "average",
        "unique_score_values",
        "tied_score_frequency",
    } <= set(distribution)
    assert distribution["count"] == 20
    assert distribution["unique_score_values"] == 2
    assert row_level["percent_above_thresholds"]["0.9"] == pytest.approx(1.0)
    assert row_level["percent_above_thresholds"]["0.95"] == pytest.approx(1.0)
    assert row_level["percent_above_thresholds"]["0.99"] == pytest.approx(0.9)
    assert row_level["percent_above_thresholds"]["0.999"] == pytest.approx(0.9)
    assert simulation["achieved_rate"] != pytest.approx(0.1)
    assert simulation["tied_score_limitation"] is True
    assert analysis["score_saturation_detected"] is True
    assert analysis["threshold_only_calibration_feasible"] is False
    assert analysis["probability_calibration_recommended"] is True
    assert analysis["retraining_recommended"] is True
    assert analysis["verdict"] in {
        "probability_calibration_required",
        "retraining_recommended",
    }


def test_survival_threshold_simulation_respects_probability_bounds(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_csv(
        logs / "survival_exit_shadow.csv",
        ["timestamp", "survival_risk_score"],
        [[f"t{i}", 1.0] for i in range(20)],
    )

    analysis = summarize_offline_calibration(
        reports,
        logs,
        target_survival_exit_rates=(0.05,),
    )["survival_exit_analysis"]
    simulation = analysis["row_level_analysis"]["threshold_simulations"]["0.05"]

    assert simulation["threshold"] <= 1.0
    assert simulation["achieved_rate"] == 1.0
    assert simulation["nearest_achievable_below"] is None
    assert simulation["tied_score_limitation"] is True
    assert analysis["threshold_only_calibration_feasible"] is False
    assert analysis["retraining_recommended"] is True


def test_advanced_risk_reason_aggregation_and_approximate_counterfactual(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "advanced_risk_shadow",
        "20260708010101",
        advanced_risk={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 8,
            "actually_blocked_count": 0,
            "would_block_rate": 0.8,
            "actual_block_rate": 0.0,
            "would_pause_count": 2,
            "actually_paused_count": 0,
            "would_reduce_size_count": 1,
            "actually_reduced_count": 0,
            "top_reasons": {
                "max_open_positions_limit": 6,
                "consecutive_losses_limit": 2,
                "normal": 2,
            },
        },
    )
    reasons = (
        ["max_open_positions_limit"] * 6
        + ["consecutive_losses_limit"] * 2
        + ["daily_loss_pct_limit"]
        + ["normal"]
    )
    _write_csv(
        logs / "advanced_risk_shadow.csv",
        [
            "timestamp",
            "would_block",
            "actually_blocked",
            "would_pause",
            "would_reduce_size",
            "top_reason",
        ],
        [
            [
                f"t{i}",
                int(reason != "normal"),
                0,
                int(reason in {"consecutive_losses_limit", "daily_loss_pct_limit"}),
                0,
                reason,
            ]
            for i, reason in enumerate(reasons)
        ],
    )

    analysis = summarize_offline_calibration(reports, logs)["advanced_risk_analysis"]
    counterfactual = analysis["row_level_analysis"]["approximate_top_reason_counterfactual"]

    assert analysis["top_reasons_aggregated"]["max_open_positions_limit"]["count"] == 6
    assert analysis["top_reasons_aggregated"]["consecutive_losses_limit"]["count"] == 2
    assert analysis["reason_share_percentages"]["max_open_positions_limit"] == pytest.approx(60.0)
    assert analysis["reason_share_percentages"]["consecutive_losses_limit"] == pytest.approx(20.0)
    assert analysis["row_level_analysis"]["top_reasons"]["daily_loss_pct_limit"] == 1
    assert counterfactual["approximate"] is True
    assert counterfactual["original_would_block_rate"] == pytest.approx(0.9)
    assert counterfactual["exclude_max_open_positions_limit"] == pytest.approx(0.3)
    assert counterfactual["exclude_consecutive_losses_limit"] == pytest.approx(0.7)
    assert counterfactual["exclude_each_reason"]["daily_loss_pct_limit"] == pytest.approx(0.8)
    assert counterfactual[
        "exclude_max_open_positions_and_consecutive_losses"
    ] == pytest.approx(0.1)
    assert analysis["max_open_positions_dominates"] is True
    assert analysis["multiple_rules_too_strict"] is True
    assert analysis["needs_rule_calibration"] is True
    assert analysis["recommendation"]["keep_advanced_risk_active_false"] is True


def _combined_sections(
    *,
    paper_rows: int,
    paper_with_id: int,
    closed_rows: int,
    closed_with_id: int,
    xgb_total: int,
    xgb_matched: int,
    xgb_unmatched: int,
    actuals: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "safety": {
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": dict(actuals or ZERO_ACTUALS),
            "shadow_only_warning": any((actuals or ZERO_ACTUALS).values()),
        },
        "isolation_forest": {
            "file_status": "ok",
            "total_rows": 1,
            "would_block_rate": 1.0,
            "actual_block_rate": 0.0,
        },
        "xgboost": {
            "file_status": "ok",
            "total_rows": xgb_total,
            "would_confirm_count": xgb_total,
            "would_reject_count": 0,
            "would_reject_rate": 0.0,
            "actual_reject_rate": 0.0,
        },
        "survival_exit": {
            "file_status": "ok",
            "total_rows": 1,
            "would_exit_rate": 1.0,
            "actual_exit_rate": 0.0,
        },
        "advanced_risk": {
            "file_status": "ok",
            "total_rows": 1,
            "would_block_rate": 0.8,
            "actual_block_rate": 0.0,
        },
        "trade_lineage": {
            "paper_trade_rows": paper_rows,
            "paper_trade_rows_with_signal_id": paper_with_id,
            "closed_trade_rows": closed_rows,
            "closed_trade_rows_with_signal_id": closed_with_id,
        },
        "xgboost_outcome": {
            "would_confirm_matched_count": xgb_matched,
            "would_reject_matched_count": 0,
            "matched_closed_trade_count": xgb_matched,
            "unmatched_decision_rows": xgb_unmatched,
        },
        "paper_pnl": {
            "closed_trade_count": closed_rows,
            "total_pnl": 0.1,
            "average_pnl": None if closed_rows == 0 else 0.1 / closed_rows,
            "win_rate": 1.0 if closed_rows else None,
        },
    }


def test_combined_shadow_aggregates_signal_id_and_outcome_coverage(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "combined_shadow",
        "20260709010101",
        **_combined_sections(
            paper_rows=2,
            paper_with_id=2,
            closed_rows=1,
            closed_with_id=1,
            xgb_total=10,
            xgb_matched=2,
            xgb_unmatched=8,
        ),
    )
    _write_unified(
        reports,
        "combined_shadow",
        "20260709020202",
        **_combined_sections(
            paper_rows=4,
            paper_with_id=2,
            closed_rows=3,
            closed_with_id=1,
            xgb_total=20,
            xgb_matched=3,
            xgb_unmatched=17,
        ),
    )

    combined = summarize_offline_calibration(reports, logs)["combined_shadow_integration"]

    assert combined["number_of_runs"] == 2
    assert all(run["all_modules_present"] for run in combined["runs"])
    assert combined["actual_behavior_counts"] == ZERO_ACTUALS
    assert combined["signal_id_coverage"]["paper_trades"]["coverage_rate"] == pytest.approx(4 / 6)
    assert combined["signal_id_coverage"]["closed_trades"]["coverage_rate"] == pytest.approx(2 / 4)
    assert combined["xgboost_matched_outcome_coverage"]["matched_count"] == 5
    assert combined["xgboost_matched_outcome_coverage"]["coverage_rate"] == pytest.approx(5 / 30)
    assert combined["verdict"] == "integration_incomplete"
    assert "does not prove profitability" in combined["calibration_note"].lower()


def test_combined_shadow_passes_only_with_complete_safe_integration(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "combined_shadow",
        "20260709030303",
        **_combined_sections(
            paper_rows=2,
            paper_with_id=2,
            closed_rows=1,
            closed_with_id=1,
            xgb_total=2,
            xgb_matched=2,
            xgb_unmatched=0,
        ),
    )

    combined = summarize_offline_calibration(reports, logs)["combined_shadow_integration"]

    assert combined["verdict"] == "integration_passed"
    assert combined["actual_behavior_counts"] == ZERO_ACTUALS


def test_combined_shadow_actual_behavior_is_an_integration_failure(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    actuals = dict(ZERO_ACTUALS)
    actuals["actually_rejected"] = 1
    _write_unified(
        reports,
        "combined_shadow",
        "20260709040404",
        **_combined_sections(
            paper_rows=1,
            paper_with_id=1,
            closed_rows=1,
            closed_with_id=1,
            xgb_total=1,
            xgb_matched=1,
            xgb_unmatched=0,
            actuals=actuals,
        ),
    )

    combined = summarize_offline_calibration(reports, logs)["combined_shadow_integration"]

    assert combined["actual_behavior_counts"]["actually_rejected"] == 1
    assert combined["verdict"] == "integration_failed"


def test_evidence_matrix_has_every_module_and_required_fields(tmp_path):
    summary = summarize_offline_calibration(tmp_path / "reports", tmp_path / "logs")

    matrix = summary["evidence_matrix"]
    assert set(matrix) == {
        "baseline",
        "isolation_forest",
        "xgboost",
        "survival_exit",
        "advanced_risk",
        "combined_shadow",
    }
    required = {
        "technical_status",
        "evidence_strength",
        "calibration_status",
        "allowed_mode",
        "prohibited_mode",
        "next_action",
    }
    assert all(set(row) == required for row in matrix.values())
    assert matrix["baseline"]["allowed_mode"] == "paper-only"
    assert matrix["isolation_forest"]["allowed_mode"] == "shadow-only"
    assert matrix["xgboost"]["allowed_mode"] == "shadow-outcome-only"
    assert matrix["combined_shadow"]["allowed_mode"] == "paper-shadow-integration-only"


def test_json_writer_produces_valid_complete_report(tmp_path):
    summary = summarize_offline_calibration(tmp_path / "reports", tmp_path / "logs")
    out = write_json_summary(summary, tmp_path / "out" / "offline_calibration_sweep.json")

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert set(payload) == {
        "evidence_manifest_schema_version",
        "evidence_manifest_generated_at",
        "evidence_manifest_digest",
        "evidence_runs_total",
        "evidence_runs_strategy_included",
        "evidence_runs_safety_included",
        "evidence_runs_excluded",
        "evidence_exclusions",
        "input_inventory",
        "run_classification",
        "baseline_cross_run_summary",
        "isolation_forest_analysis",
        "xgboost_outcome_aggregation",
        "survival_exit_analysis",
        "advanced_risk_analysis",
        "combined_shadow_integration",
        "evidence_matrix",
        "final_recommendation",
    }
    assert payload["final_recommendation"]["final_verdict"] == (
        "paper_only_offline_calibration_required"
    )
    final = payload["final_recommendation"]
    assert len(final["priority_order"]) == 9
    assert final["safety_constraints"] == {
        "paper_only": True,
        "no_mainnet": True,
        "no_testnet_real_orders": True,
        "no_real_orders": True,
        "place_real_orders_must_remain_false": True,
        "no_active_or_blocking_modules": True,
    }


def test_cli_writes_json_and_prints_final_verdict(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    out = tmp_path / "out" / "offline_calibration_sweep.json"
    _write_unified(reports, "baseline", "20260710010101")
    _write_csv(
        logs / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score"],
        [["t1", -0.5], ["t2", -0.4]],
    )
    _write_csv(
        logs / "survival_exit_shadow.csv",
        ["timestamp", "survival_risk_score"],
        [["t1", 0.9], ["t2", 0.8]],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reports-dir",
            str(reports),
            "--logs-dir",
            str(logs),
            "--min-xgb-matched-per-group",
            "2",
            "--target-if-block-rates",
            "0.25",
            "--target-survival-exit-rates",
            "0.1",
            "--json",
            "--json-out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "paper_only_offline_calibration_required" in result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["final_recommendation"]["final_verdict"] == (
        "paper_only_offline_calibration_required"
    )
    assert payload["xgboost_outcome_aggregation"]["combined_aggregate"][
        "minimum_evidence_threshold"
    ]["configured_per_group"] == 2
    assert "0.25" in payload["isolation_forest_analysis"]["row_level_analysis"][
        "threshold_simulations"
    ]
    assert "0.1" in payload["survival_exit_analysis"]["row_level_analysis"][
        "threshold_simulations"
    ]
