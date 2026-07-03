"""Tests for the final matrix experiment comparison report."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from tools.final_experiment_comparison import (
    REQUIRED_MODES,
    format_text_summary,
    summarize_final_comparison,
    write_json_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "final_experiment_comparison.py"

ZERO_ACTUALS = {
    "actually_blocked": 0,
    "actually_rejected": 0,
    "actually_exited": 0,
    "actually_paused": 0,
    "actually_reduced": 0,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _unified_payload() -> dict:
    return {
        "logs_dir": "logs",
        "safety": {
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": copy.deepcopy(ZERO_ACTUALS),
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
            "top_reasons": {},
        },
        "xgboost": {
            "file_status": "missing",
            "total_rows": 0,
            "would_confirm_count": 0,
            "would_reject_count": 0,
            "actually_rejected_count": 0,
            "would_reject_rate": 0.0,
            "actual_reject_rate": 0.0,
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
            "would_reject_matched_count": 0,
            "would_reject_average_pnl": None,
        },
    }


def _write_unified(reports_dir: Path, mode: str, timestamp: str, **sections: dict) -> Path:
    payload = _unified_payload()
    for key, value in sections.items():
        payload[key] = value
    path = reports_dir / f"matrix_{mode}_{timestamp}_unified.json"
    _write_json(path, payload)
    return path


def test_handles_missing_reports(tmp_path):
    summary = summarize_final_comparison(tmp_path)

    assert summary["run_inventory"]["modes_found"] == []
    assert summary["run_inventory"]["missing_expected_modes"] == REQUIRED_MODES
    assert summary["safety_summary"]["aggregate_actual_behavior_counts"] == ZERO_ACTUALS
    assert summary["combined_shadow_integration_summary"]["verdict"] == "integration_failed"
    assert "Final Experiment Comparison" in format_text_summary(summary)


def test_selects_latest_report_per_mode(tmp_path):
    _write_unified(
        tmp_path,
        "baseline",
        "20260703010101",
        paper_pnl={"closed_trade_count": 1, "total_pnl": 1.0, "average_pnl": 1.0, "win_rate": 1.0},
    )
    latest = _write_unified(
        tmp_path,
        "baseline",
        "20260703020202",
        paper_pnl={"closed_trade_count": 2, "total_pnl": -2.0, "average_pnl": -1.0, "win_rate": 0.0},
    )

    summary = summarize_final_comparison(tmp_path)

    assert summary["run_inventory"]["latest_timestamp_per_mode"]["baseline"] == "20260703020202"
    assert summary["run_inventory"]["report_paths_used"]["baseline"]["unified"] == str(latest)
    assert summary["baseline_performance"]["closed_trades"] == 2
    assert summary["baseline_performance"]["total_pnl"] == -2.0


def test_baseline_pnl_metrics_extracted(tmp_path):
    _write_unified(
        tmp_path,
        "baseline",
        "20260703030303",
        paper_pnl={
            "closed_trade_count": 4,
            "total_pnl": -0.25,
            "average_pnl": -0.0625,
            "win_rate": 0.25,
        },
    )

    summary = summarize_final_comparison(tmp_path)

    baseline = summary["baseline_performance"]
    assert baseline["closed_trades"] == 4
    assert baseline["total_pnl"] == -0.25
    assert baseline["average_pnl"] == -0.0625
    assert baseline["win_rate"] == 0.25


def test_iforest_aggressive_verdict(tmp_path):
    _write_unified(
        tmp_path,
        "iforest_shadow",
        "20260703040404",
        isolation_forest={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 10,
            "actually_blocked_count": 0,
            "would_block_rate": 1.0,
            "actual_block_rate": 0.0,
            "average_anomaly_score": -0.6,
        },
    )

    summary = summarize_final_comparison(tmp_path)

    assert summary["isolation_forest_summary"]["would_block_rate"] == 1.0
    assert summary["isolation_forest_summary"]["average_anomaly_score"] == -0.6
    assert summary["isolation_forest_summary"]["verdict"] == "unsafe_to_enable"


def test_xgboost_best_candidate_verdict(tmp_path):
    timestamp = "20260703050505"
    _write_unified(
        tmp_path,
        "xgboost_shadow_outcome",
        timestamp,
        xgboost={
            "file_status": "ok",
            "total_rows": 20,
            "would_confirm_count": 14,
            "would_reject_count": 4,
            "actually_rejected_count": 0,
            "would_reject_rate": 0.20,
            "actual_reject_rate": 0.0,
        },
    )
    _write_json(
        tmp_path / f"matrix_xgboost_shadow_outcome_{timestamp}_xgboost_audit.json",
        {
            "total_xgboost_rows": 20,
            "would_confirm_count": 14,
            "would_reject_count": 4,
            "would_confirm_matched_count": 3,
            "would_confirm_average_pnl": 0.30,
            "would_reject_matched_count": 2,
            "would_reject_average_pnl": -0.10,
        },
    )

    summary = summarize_final_comparison(tmp_path)

    xgb = summary["xgboost_summary"]
    assert xgb["would_reject_rate"] == 0.20
    assert xgb["would_confirm_matched_count"] == 3
    assert xgb["would_reject_matched_count"] == 2
    assert xgb["would_confirm_average_pnl"] == 0.30
    assert xgb["would_reject_average_pnl"] == -0.10
    assert xgb["verdict"] == "best_candidate_for_more_shadow_testing"


def test_survival_too_aggressive_verdict(tmp_path):
    _write_unified(
        tmp_path,
        "survival_shadow",
        "20260703060606",
        survival_exit={
            "file_status": "ok",
            "total_rows": 8,
            "would_exit_early_count": 8,
            "actually_exited_count": 0,
            "would_exit_rate": 1.0,
            "actual_exit_rate": 0.0,
            "average_risk_score": 0.99,
        },
    )

    summary = summarize_final_comparison(tmp_path)

    survival = summary["survival_exit_summary"]
    assert survival["would_exit_rate"] == 1.0
    assert survival["average_risk_score"] == 0.99
    assert survival["verdict"] == "too_aggressive"


def test_advanced_risk_too_strict_verdict(tmp_path):
    _write_unified(
        tmp_path,
        "advanced_risk_shadow",
        "20260703070707",
        advanced_risk={
            "file_status": "ok",
            "total_rows": 10,
            "would_block_count": 8,
            "actually_blocked_count": 0,
            "would_block_rate": 0.80,
            "actual_block_rate": 0.0,
            "top_reasons": {"max_open_positions_limit": 8, "normal": 2},
        },
    )

    summary = summarize_final_comparison(tmp_path)

    risk = summary["advanced_risk_summary"]
    assert risk["would_block_rate"] == 0.80
    assert risk["top_reasons"]["max_open_positions_limit"] == 8
    assert risk["verdict"] == "too_strict"


def test_combined_shadow_integration_verdict(tmp_path):
    _write_unified(
        tmp_path,
        "combined_shadow",
        "20260703080808",
        isolation_forest={"file_status": "ok", "total_rows": 5, "would_block_rate": 1.0, "actual_block_rate": 0.0},
        xgboost={"file_status": "ok", "total_rows": 5, "would_reject_rate": 0.2, "actual_reject_rate": 0.0},
        survival_exit={"file_status": "ok", "total_rows": 5, "would_exit_rate": 1.0, "actual_exit_rate": 0.0},
        advanced_risk={"file_status": "ok", "total_rows": 5, "would_block_rate": 0.8, "actual_block_rate": 0.0},
        trade_lineage={
            "paper_trade_rows": 3,
            "paper_trade_rows_with_signal_id": 3,
            "closed_trade_rows": 2,
            "closed_trade_rows_with_signal_id": 2,
        },
    )

    summary = summarize_final_comparison(tmp_path)

    combined = summary["combined_shadow_integration_summary"]
    assert combined["all_modules_present"] is True
    assert combined["any_actual_blocking_rejection_exit_pause_reduction"] is False
    assert combined["signal_id_coverage"]["paper_trades"]["coverage_rate"] == 1.0
    assert combined["signal_id_coverage"]["closed_trades"]["coverage_rate"] == 1.0
    assert combined["verdict"] == "integration_passed"


def test_json_output_valid(tmp_path):
    _write_unified(tmp_path, "baseline", "20260703090909")
    summary = summarize_final_comparison(tmp_path)
    out = write_json_summary(summary, tmp_path / "reports" / "final_experiment_comparison.json")

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["run_inventory"]["modes_found"] == ["baseline"]
    assert payload["final_recommendation"]["verdict"] == "paper_only_no_live_or_real_orders"


def test_cli_terminal_output_includes_final_recommendation(tmp_path):
    _write_unified(tmp_path, "baseline", "20260703101010")

    result = subprocess.run(
        [sys.executable, str(CLI), "--reports-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "I. Final recommendation" in result.stdout
    assert "no live/mainnet" in result.stdout
    assert "no real orders" in result.stdout
    assert "keep paper mode" in result.stdout
