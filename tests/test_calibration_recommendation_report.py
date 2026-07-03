"""Tests for the calibration recommendation report."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from tools.calibration_recommendation_report import (
    REQUIRED_MODES,
    format_text_summary,
    summarize_calibration_recommendations,
    write_json_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "calibration_recommendation_report.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _base_unified() -> dict:
    return {
        "safety": {
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": {
                "actually_blocked": 0,
                "actually_rejected": 0,
                "actually_exited": 0,
                "actually_paused": 0,
                "actually_reduced": 0,
            },
            "shadow_only_warning": False,
        },
        "isolation_forest": {
            "file_status": "missing",
            "total_rows": 0,
            "would_block_rate": 0.0,
            "actual_block_rate": 0.0,
        },
        "xgboost": {
            "file_status": "missing",
            "total_rows": 0,
            "would_confirm_count": 0,
            "would_reject_count": 0,
            "would_reject_rate": 0.0,
            "actual_reject_rate": 0.0,
        },
        "survival_exit": {
            "file_status": "missing",
            "total_rows": 0,
            "would_exit_rate": 0.0,
            "actual_exit_rate": 0.0,
        },
        "advanced_risk": {
            "file_status": "missing",
            "total_rows": 0,
            "would_block_rate": 0.0,
            "actual_block_rate": 0.0,
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
            "would_reject_matched_count": 0,
        },
    }


def _write_unified(reports_dir: Path, mode: str, timestamp: str, **sections: dict) -> Path:
    payload = _base_unified()
    for key, value in sections.items():
        payload[key] = value
    path = reports_dir / f"matrix_{mode}_{timestamp}_unified.json"
    _write_json(path, payload)
    return path


def test_handles_missing_reports_logs(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["input_inventory"]["reports_found"] == {}
    assert summary["input_inventory"]["final_experiment_comparison"]["status"] == "missing"
    assert "matrix reports for baseline" in summary["input_inventory"]["missing_inputs"]
    assert summary["overall_safety_recommendation"]["keep_paper_only"] is True
    assert summary["final_calibration_plan"]["final_verdict"] == "paper_only_calibration_required"
    assert "Calibration Recommendation Report" in format_text_summary(summary)


def test_reads_final_experiment_comparison_json_if_present(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_json(
        reports / "final_experiment_comparison.json",
        {
            "final_recommendation": {"verdict": "paper_only_no_live_or_real_orders"},
            "baseline_performance": {
                "closed_trades": 4,
                "total_pnl": -0.4,
                "average_pnl": -0.1,
                "win_rate": 0.25,
            },
        },
    )

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["input_inventory"]["final_experiment_comparison"]["status"] == "ok"
    assert summary["overall_safety_recommendation"]["phase15_final_verdict"] == "paper_only_no_live_or_real_orders"
    assert summary["baseline_strategy_health"]["closed_trades"] == 4
    assert summary["baseline_strategy_health"]["verdict"] == "baseline_weak"


def test_selects_latest_matrix_reports(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "baseline",
        "20260703010101",
        paper_pnl={"closed_trade_count": 1, "total_pnl": 1.0, "average_pnl": 1.0, "win_rate": 1.0},
    )
    latest = _write_unified(
        reports,
        "baseline",
        "20260703020202",
        paper_pnl={"closed_trade_count": 2, "total_pnl": -2.0, "average_pnl": -1.0, "win_rate": 0.0},
    )

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["input_inventory"]["latest_matrix_timestamp_per_mode"]["baseline"] == "20260703020202"
    assert summary["input_inventory"]["reports_found"]["baseline"]["unified"] == str(latest)
    assert summary["baseline_strategy_health"]["total_pnl"] == -2.0


def test_produces_baseline_verdict(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "baseline",
        "20260703030303",
        paper_pnl={"closed_trade_count": 6, "total_pnl": -0.3, "average_pnl": -0.05, "win_rate": 0.33},
    )

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["baseline_strategy_health"]["closed_trades"] == 6
    assert summary["baseline_strategy_health"]["verdict"] == "baseline_weak"


def test_produces_if_unsafe_and_needs_calibration_verdicts(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "iforest_shadow",
        "20260703040404",
        isolation_forest={"file_status": "ok", "total_rows": 10, "would_block_rate": 1.0, "actual_block_rate": 0.0},
    )
    unsafe = summarize_calibration_recommendations(reports, logs)

    _write_unified(
        reports,
        "iforest_shadow",
        "20260703050505",
        isolation_forest={"file_status": "ok", "total_rows": 10, "would_block_rate": 0.4, "actual_block_rate": 0.0},
    )
    needs = summarize_calibration_recommendations(reports, logs)

    assert unsafe["isolation_forest_calibration"]["verdict"] == "unsafe_to_enable"
    assert needs["isolation_forest_calibration"]["verdict"] == "needs_threshold_calibration"


def test_computes_if_threshold_candidates_from_sample_anomaly_scores(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    rows = [[f"t{i}", -101 + i, 1 if i <= 15 else 0, 0] for i in range(1, 101)]
    _write_csv(
        logs / "isolation_forest_shadow.csv",
        ["timestamp", "anomaly_score", "would_block", "actually_blocked"],
        rows,
    )

    summary = summarize_calibration_recommendations(reports, logs)

    candidates = summary["isolation_forest_calibration"]["threshold_candidates"]
    assert round(candidates["1%"]["anomaly_score_threshold"], 6) == -99.01
    assert candidates["1%"]["simulated_block_rate"] == 0.01
    assert round(candidates["10%"]["simulated_block_rate"], 6) == 0.10


def test_produces_xgboost_best_candidate_and_promising_verdicts(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    timestamp = "20260703060606"
    _write_unified(
        reports,
        "xgboost_shadow_outcome",
        timestamp,
        xgboost={
            "file_status": "ok",
            "total_rows": 20,
            "would_confirm_count": 14,
            "would_reject_count": 4,
            "would_reject_rate": 0.20,
            "actual_reject_rate": 0.0,
        },
    )
    _write_json(
        reports / f"matrix_xgboost_shadow_outcome_{timestamp}_xgboost_audit.json",
        {
            "would_confirm_matched_count": 2,
            "would_confirm_average_pnl": 0.2,
            "would_reject_matched_count": 1,
            "would_reject_average_pnl": -0.1,
        },
    )
    best = summarize_calibration_recommendations(reports, logs)

    _write_unified(
        reports,
        "xgboost_shadow_outcome",
        "20260703070707",
        xgboost={
            "file_status": "ok",
            "total_rows": 20,
            "would_confirm_count": 14,
            "would_reject_count": 4,
            "would_reject_rate": 0.20,
            "actual_reject_rate": 0.0,
        },
    )
    promising = summarize_calibration_recommendations(reports, logs)

    assert best["xgboost_calibration"]["verdict"] == "best_candidate_for_more_shadow_testing"
    assert promising["xgboost_calibration"]["verdict"] == "promising_but_unproven"


def test_warns_when_xgboost_matched_sample_is_small(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    timestamp = "20260703080808"
    _write_unified(
        reports,
        "xgboost_shadow_outcome",
        timestamp,
        xgboost={"file_status": "ok", "total_rows": 10, "would_confirm_count": 6, "would_reject_count": 2, "would_reject_rate": 0.2, "actual_reject_rate": 0.0},
    )
    _write_json(
        reports / f"matrix_xgboost_shadow_outcome_{timestamp}_xgboost_audit.json",
        {"would_confirm_matched_count": 1, "would_reject_matched_count": 1},
    )

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["xgboost_calibration"]["sample_size_warning"] is True
    assert "small" in summary["xgboost_calibration"]["sample_size_message"]


def test_produces_survival_too_aggressive_verdict(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "survival_shadow",
        "20260703090909",
        survival_exit={"file_status": "ok", "total_rows": 10, "would_exit_rate": 1.0, "actual_exit_rate": 0.0, "average_risk_score": 0.99},
    )

    summary = summarize_calibration_recommendations(reports, logs)

    assert summary["survival_exit_calibration"]["would_exit_rate"] == 1.0
    assert summary["survival_exit_calibration"]["verdict"] == "too_aggressive"


def test_computes_survival_threshold_candidates_from_risk_scores(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    rows = [[f"t{i}", i / 100, 1 if i >= 70 else 0, 0] for i in range(1, 101)]
    _write_csv(
        logs / "survival_exit_shadow.csv",
        ["timestamp", "survival_risk_score", "would_exit_early", "actually_exited"],
        rows,
    )

    summary = summarize_calibration_recommendations(reports, logs)

    candidates = summary["survival_exit_calibration"]["threshold_candidates"]
    assert round(candidates["10%"]["risk_score_threshold"], 6) == 0.901
    assert round(candidates["10%"]["simulated_exit_rate"], 6) == 0.10
    assert candidates["30%"]["simulated_exit_rate"] == 0.30


def test_produces_advanced_risk_too_strict_and_detects_max_open_dominance(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "advanced_risk_shadow",
        "20260703101010",
        advanced_risk={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_rate": 0.8,
            "actual_block_rate": 0.0,
            "would_pause_count": 0,
            "would_reduce_size_count": 0,
            "top_reasons": {"max_open_positions_limit": 8, "normal": 2},
        },
    )

    summary = summarize_calibration_recommendations(reports, logs)

    risk = summary["advanced_risk_calibration"]
    assert risk["verdict"] == "too_strict"
    assert risk["max_open_positions_limit_dominates"] is True
    assert risk["max_open_positions_limit_rate"] == 0.8


def test_detects_max_open_positions_limit_dominance_from_log(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_csv(
        logs / "advanced_risk_shadow.csv",
        ["timestamp", "would_block", "actually_blocked", "would_pause", "would_reduce_size", "top_reason"],
        [
            ["t1", 1, 0, 0, 0, "max_open_positions_limit"],
            ["t2", 1, 0, 0, 0, "max_open_positions_limit"],
            ["t3", 0, 0, 0, 0, "normal"],
        ],
    )

    summary = summarize_calibration_recommendations(reports, logs)

    risk = summary["advanced_risk_calibration"]
    assert risk["top_reasons"]["max_open_positions_limit"] == 2
    assert risk["max_open_positions_limit_dominates"] is True
    assert round(risk["max_open_positions_limit_rate"], 6) == round(2 / 3, 6)


def test_combined_shadow_integration_safety_verdict(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(
        reports,
        "combined_shadow",
        "20260703111111",
        isolation_forest={"file_status": "ok", "total_rows": 3},
        xgboost={"file_status": "ok", "total_rows": 3},
        survival_exit={"file_status": "ok", "total_rows": 3},
        advanced_risk={"file_status": "ok", "total_rows": 3},
        trade_lineage={
            "paper_trade_rows": 2,
            "paper_trade_rows_with_signal_id": 2,
            "closed_trade_rows": 1,
            "closed_trade_rows_with_signal_id": 1,
        },
    )

    summary = summarize_calibration_recommendations(reports, logs)

    combined = summary["combined_shadow_calibration_view"]
    assert combined["all_modules_present"] is True
    assert combined["combined_verdict"] == "integration_passed"
    assert combined["calibration_note"] == "combined shadow proves integration safety, not profitability"


def test_json_output_is_valid(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(reports, "baseline", "20260703121212")
    summary = summarize_calibration_recommendations(reports, logs)
    out = write_json_summary(summary, tmp_path / "out" / "calibration_recommendation_report.json")

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["final_calibration_plan"]["final_verdict"] == "paper_only_calibration_required"
    assert payload["input_inventory"]["latest_matrix_timestamp_per_mode"]["baseline"] == "20260703121212"


def test_cli_output_includes_final_calibration_plan(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_unified(reports, "baseline", "20260703131313")

    result = subprocess.run(
        [sys.executable, str(CLI), "--reports-dir", str(reports), "--logs-dir", str(logs)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "I. Final calibration plan" in result.stdout
    assert "paper_only_calibration_required" in result.stdout
    assert "Collect longer baseline + XGBoost shadow outcome data" in result.stdout
