"""Tests for Phase 18 read-only calibration proposals and evidence gates."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.evidence_manifest import build_evidence_manifest, evidence_manifest_digest
from tools.offline_calibration_proposals import (
    SAFE_EXPERIMENT_COMMANDS,
    format_text_summary,
    summarize_offline_calibration_proposals,
    write_json_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "offline_calibration_proposals.py"


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


def _phase17_payload() -> dict[str, Any]:
    return {
        "input_inventory": {
            "run_timestamps_by_mode": {
                "baseline": ["20260701010101", "20260702010101", "20260703010101"],
                "xgboost_shadow_outcome": [
                    "20260701020101",
                    "20260702020101",
                    "20260703020101",
                ],
            },
            "duplicate_reports_skipped": [],
        },
        "baseline_cross_run_summary": {
            "number_of_runs": 3,
            "total_closed_trades": 11,
            "total_pnl": 0.027077,
            "weighted_average_pnl": 0.002462,
            "overall_win_rate": 4 / 11,
            "positive_run_count": 2,
            "negative_run_count": 1,
            "verdict": "baseline_inconsistent",
            "runs": [
                {"timestamp": "20260701010101"},
                {"timestamp": "20260702010101"},
                {"timestamp": "20260703010101"},
            ],
        },
        "isolation_forest_analysis": {
            "consistent_100_percent_anomaly_behavior": True,
            "threshold_only_calibration_feasible": False,
            "score_saturation_detected": True,
            "retraining_recommended": True,
            "verdict": "retraining_recommended",
        },
        "xgboost_outcome_aggregation": {
            "by_source_mode": {
                "xgboost_shadow_outcome": {
                    "number_of_runs": 3,
                    "confirmed": {"matched_count": 6},
                    "rejected": {"matched_count": 6},
                },
                "combined_shadow": {
                    "number_of_runs": 1,
                    "confirmed": {"matched_count": 0},
                    "rejected": {"matched_count": 0},
                },
            },
            "combined_aggregate": {
                "number_of_runs": 4,
                "total_decision_rows": 1354,
                "confirmed": {
                    "matched_count": 6,
                    "total_pnl": -0.007356,
                    "weighted_average_pnl": -0.001226,
                },
                "rejected": {
                    "matched_count": 6,
                    "total_pnl": -0.109386,
                    "weighted_average_pnl": -0.018231,
                },
                "pnl_separation": 0.017005,
                "relationship_consistent_across_runs": True,
                "match_coverage_rate": 12 / 1354,
                "unmatched_data_warning": True,
            },
        },
        "survival_exit_analysis": {
            "score_saturation_detected": True,
            "threshold_only_calibration_feasible": False,
            "probability_calibration_recommended": True,
            "threshold_simulations": {
                "0.1": {
                    "threshold": 0.9997,
                    "achieved_rate": 0.10,
                }
            },
            "verdict": "probability_calibration_required",
        },
        "advanced_risk_analysis": {
            "max_open_positions_dominates": True,
            "consecutive_losses_dominates": True,
            "multiple_rules_too_strict": True,
            "needs_rule_calibration": True,
        },
        "combined_shadow_integration": {
            "number_of_runs": 1,
            "actual_behavior_counts": {
                "actually_blocked": 0,
                "actually_rejected": 0,
                "actually_exited": 0,
                "actually_paused": 0,
                "actually_reduced": 0,
            },
            "signal_id_coverage": {
                "paper_trades": {"coverage_rate": 1.0},
                "closed_trades": {"coverage_rate": 1.0},
            },
            "verdict": "integration_passed",
        },
    }


def _write_phase17(reports: Path, payload: dict[str, Any] | None = None) -> Path:
    value = dict(payload or _phase17_payload())
    manifest = build_evidence_manifest(reports)
    value["evidence_manifest_schema_version"] = manifest["schema_version"]
    value["evidence_manifest_generated_at"] = manifest["generated_at"]
    value["evidence_manifest_digest"] = evidence_manifest_digest(manifest)
    return _write_json(
        reports / "offline_calibration_sweep.json",
        value,
    )


def _write_verified_index(reports: Path, mode: str, timestamp: str) -> Path:
    return _write_json(
        reports / f"matrix_index_{timestamp}.json",
        {
            "matrix_timestamp": timestamp,
            "requested_mode": mode,
            "duration_minutes": 60,
            "runs": [
                {
                    "mode": mode,
                    "run_started_utc": "2026-07-01T00:00:00Z",
                    "finished_at": "2026-07-01T01:00:00Z",
                    "duration_minutes": 60,
                    "exit_status": 0,
                    "stale_entry_guard_checked": True,
                    "stale_entry_count": 0,
                    "stale_entry_signal_ids": [],
                    "evidence_valid": True,
                    "report_paths": {
                        "unified": str(
                            reports / f"matrix_{mode}_{timestamp}_unified.json"
                        )
                    },
                }
            ],
        },
    )


def _baseline_unified(
    *,
    closed: int = 3,
    pnl: float = 0.03,
    best_trade: float | None = None,
) -> dict[str, Any]:
    paper_pnl: dict[str, Any] = {
        "closed_trade_count": closed,
        "total_pnl": pnl,
        "average_pnl": pnl / closed if closed else None,
        "win_rate": 2 / closed if closed else None,
    }
    if best_trade is not None:
        paper_pnl["best_trade"] = {
            "realized_pnl": best_trade,
            "signal_id": "signal-best",
        }
        paper_pnl["worst_trade"] = {
            "realized_pnl": -0.01,
            "signal_id": "signal-worst",
        }
    return {
        "safety": {
            "inferred_trade_mode": "paper",
            "actual_behavior_counts": {},
        },
        "paper_pnl": paper_pnl,
        "trade_lineage": {
            "paper_trade_rows": closed,
            "paper_trade_rows_with_signal_id": closed,
            "closed_trade_rows": closed,
            "closed_trade_rows_with_signal_id": closed,
            "signal_id_missing_counts": {
                "paper_trades": 0,
                "closed_trades": 0,
            },
        },
    }


def _summary(
    tmp_path: Path,
    *,
    phase17: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    if phase17:
        _write_phase17(reports)
    return summarize_offline_calibration_proposals(
        reports,
        logs,
        **kwargs,
    )


def test_missing_phase17_json_falls_back_without_failure(tmp_path):
    summary = _summary(tmp_path, phase17=False)
    inventory = summary["input_evidence_inventory"]

    assert inventory["phase17_report_status"]["status"] == "missing"
    assert inventory["preferred_evidence_source"] == (
        "reconstructed_from_reports_and_current_logs"
    )
    assert "usable Phase 17 or fallback matrix evidence" in inventory["missing_inputs"]
    assert summary["final_recommendation"]["final_verdict"] == (
        "calibration_proposals_ready_paper_only"
    )


def test_missing_phase17_reconstructs_fallback_reports(tmp_path):
    reports = tmp_path / "reports"
    path = _write_json(
        reports / "matrix_baseline_20260701010101_unified.json",
        _baseline_unified(),
    )
    _write_verified_index(reports, "baseline", "20260701010101")

    summary = summarize_offline_calibration_proposals(reports, tmp_path / "logs")

    assert str(path) in summary["input_evidence_inventory"]["fallback_reports_used"]
    assert summary["baseline_evidence_proposal"]["current_evidence"][
        "number_of_runs"
    ] == 1


def test_malformed_phase17_json_is_skipped_and_reconstructed(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "offline_calibration_sweep.json").write_text("{bad", encoding="utf-8")

    summary = summarize_offline_calibration_proposals(reports, tmp_path / "logs")
    inventory = summary["input_evidence_inventory"]

    assert inventory["phase17_report_status"]["status"] == "malformed"
    assert inventory["malformed_inputs_skipped"]
    assert inventory["read_errors"]


def test_duplicate_report_representations_are_not_double_counted(tmp_path):
    reports = tmp_path / "reports"
    payload = _baseline_unified()
    _write_json(
        reports / "matrix_baseline_20260701010101_unified.json",
        payload,
    )
    _write_json(
        reports / "matrix_baseline_20260701010101_shadow_summary.json",
        payload,
    )
    _write_verified_index(reports, "baseline", "20260701010101")

    summary = summarize_offline_calibration_proposals(reports, tmp_path / "logs")

    assert summary["baseline_evidence_proposal"]["current_evidence"][
        "number_of_runs"
    ] == 1
    assert any(
        item["identity"] == "baseline:20260701010101"
        for item in summary["input_evidence_inventory"]["duplicate_inputs_skipped"]
    )


def test_global_safety_gate_is_always_locked(tmp_path):
    gate = _summary(tmp_path)["global_safety_gate"]

    assert gate == {
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


def test_baseline_evidence_gap_and_consistency(tmp_path):
    proposal = _summary(
        tmp_path,
        baseline_min_closed_trades=100,
    )["baseline_evidence_proposal"]

    assert proposal["current_evidence"]["closed_trades"] == 11
    assert proposal["evidence_gaps"]["additional_closed_trades_required"] == 89
    assert proposal["evidence_gaps"]["multiple_market_windows_represented"] is True
    assert (
        proposal["evidence_gaps"]["positive_performance_consistent_across_runs"]
        is False
    )
    assert proposal["verdict"] == "collect_more_baseline_evidence"
    assert proposal["profitability_claim_allowed"] is False


def test_baseline_outlier_warning_uses_available_trade_extrema(tmp_path):
    reports = tmp_path / "reports"
    _write_json(
        reports / "matrix_baseline_20260701010101_unified.json",
        _baseline_unified(best_trade=0.2),
    )
    _write_verified_index(reports, "baseline", "20260701010101")
    _write_phase17(reports)

    proposal = summarize_offline_calibration_proposals(
        reports,
        tmp_path / "logs",
    )["baseline_evidence_proposal"]
    diagnostic = proposal["evidence_gaps"]["outlier_assessment"]

    assert diagnostic["outlier_warning"] is True
    assert diagnostic["outlier_status"] == "possible_single_trade_dominance"
    assert proposal["acceptance_gate"]["checks"][
        "no_reliance_on_one_outlier_trade"
    ] is False


def test_isolation_forest_retraining_spec_and_artifact_rejection(tmp_path):
    spec = _summary(tmp_path)[
        "isolation_forest_retraining_specification"
    ]

    assert spec["verdict"] == "retraining_spec_ready"
    assert spec["artifact_status"] == "current_artifact_not_approved"
    assert spec["artifact_modified"] is False
    assert spec["proposed_experiment_grid"]["contamination"] == [
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    assert spec["training_command_template"]["execute_in_phase18"] is False
    assert "<TRAINING_CSV>" in spec["training_command_template"]["template"]
    assert spec["offline_acceptance_gates"][
        "current_artifact_gate_satisfied"
    ] is False


def test_xgboost_progress_remaining_and_approximate_runs(tmp_path):
    proposal = _summary(
        tmp_path,
        xgb_min_matched_per_group=30,
    )["xgboost_evidence_collection_proposal"]

    assert proposal["current_evidence"]["confirmed_matched_count"] == 6
    assert proposal["current_evidence"]["rejected_matched_count"] == 6
    assert proposal["current_evidence"]["remaining_confirmed_matches_required"] == 24
    assert proposal["current_evidence"]["remaining_rejected_matches_required"] == 24
    assert proposal["evidence_progress"]["confirmed_percentage"] == pytest.approx(20)
    assert proposal["evidence_progress"]["rejected_percentage"] == pytest.approx(20)
    confirmed_estimate = proposal["estimated_runs_remaining"]["confirmed"]
    assert confirmed_estimate["estimate_available"] is True
    assert confirmed_estimate["estimated_runs_remaining"] == 12
    assert confirmed_estimate["approximate"] is True


def test_xgboost_gate_is_not_lowered_and_blocking_stays_prohibited(tmp_path):
    proposal = _summary(
        tmp_path,
        xgb_min_matched_per_group=5,
    )["xgboost_evidence_collection_proposal"]

    assert proposal["current_evidence"]["remaining_confirmed_matches_required"] == 0
    assert proposal["current_evidence"]["remaining_rejected_matches_required"] == 0
    assert proposal["blocking_status"] == "not_approved_for_blocking"
    assert proposal["live_approval"] is False
    assert "do not lower the evidence gate" in " ".join(proposal["proposal"])
    assert proposal["acceptance_gate"]["note"].endswith(
        "not a profitability guarantee."
    )


def test_survival_calibration_spec_and_active_rejection(tmp_path):
    spec = _summary(tmp_path)[
        "survival_probability_calibration_specification"
    ]

    assert spec["verdict"] == "probability_calibration_spec_ready"
    assert spec["artifact_status"] == "current_artifact_not_approved_active"
    assert spec["survival_exit_active"] is False
    assert spec["artifact_modified"] is False
    assert "isotonic calibration" in spec["calibration_candidates"]
    assert "Brier score" in spec["evaluation_metrics_where_supported"]
    assert spec["command_templates"]["probability_calibration_cli_verified"] is False
    assert spec["problem_statement"]["raw_scores_are_calibrated_probabilities"] is False


def test_advanced_risk_generates_all_candidate_sets(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_phase17(reports)
    _write_csv(
        logs / "advanced_risk_shadow.csv",
        ["would_block", "top_reason", "daily_loss_pct"],
        [
            [1, "max_open_positions_limit", 0.2],
            [1, "consecutive_losses_limit", 0.3],
            [0, "normal", 0.1],
        ],
    )

    proposal = summarize_offline_calibration_proposals(reports, logs)[
        "advanced_risk_parameter_proposal"
    ]
    candidates = proposal["candidate_rule_sets"]

    assert [item["id"] for item in candidates] == [
        "set_a",
        "set_b",
        "set_c",
        "set_d",
        "set_e",
        "set_f",
    ]
    assert proposal["row_level_context"]["supported"] is True
    assert candidates[4]["simulation"]["would_block_rate"] == pytest.approx(0)
    assert proposal["verdict"] == "offline_rule_candidates_ready"
    assert proposal["advanced_risk_active"] is False


def test_advanced_risk_marks_insufficient_row_context(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    _write_phase17(reports)
    _write_csv(
        logs / "advanced_risk_shadow.csv",
        ["timestamp", "risk_score"],
        [["t1", 0.9]],
    )

    proposal = summarize_offline_calibration_proposals(reports, logs)[
        "advanced_risk_parameter_proposal"
    ]

    assert proposal["row_level_context"]["simulation_status"] == (
        "insufficient_row_context"
    )
    assert all(
        item["simulation"]["simulation_status"] == "insufficient_row_context"
        for item in proposal["candidate_rule_sets"]
    )


def test_advanced_risk_candidates_all_retain_daily_loss_protection(tmp_path):
    candidates = _summary(tmp_path)["advanced_risk_parameter_proposal"][
        "candidate_rule_sets"
    ]

    assert candidates
    assert all(item["daily_loss_protection_retained"] for item in candidates)
    assert all(
        "ADVANCED_RISK_MAX_DAILY_LOSS_PCT"
        not in item["settings_different_from_current"]
        for item in candidates
    )


def test_combined_validation_sequence_is_staged_and_paper_only(tmp_path):
    plan = _summary(tmp_path)["combined_validation_plan"]

    assert [item["stage"] for item in plan["stages"]] == [1, 2, 3, 4]
    assert plan["stages"][0]["activities"] == [
        "baseline paper collection",
        "XGBoost shadow outcome collection",
    ]
    assert "combined shadow for 60 minutes" in " ".join(
        plan["stages"][2]["activities"]
    )
    assert plan["stages"][3]["status"] == "not_currently_approved"
    assert plan["paper_only"] is True
    assert plan["live_approval_automatic"] is False


def test_safe_powershell_commands_and_prohibited_commands_absent(tmp_path):
    commands = _summary(tmp_path)["proposed_experiment_commands"]
    generated = [
        *commands["approved_now"].values(),
        *commands["conditional_after_future_offline_implementation"].values(),
    ]
    joined = "\n".join(generated).lower()

    assert commands["approved_now"]["baseline"] == SAFE_EXPERIMENT_COMMANDS["baseline"]
    assert (
        commands["approved_now"]["xgboost_shadow_outcome"]
        == SAFE_EXPERIMENT_COMMANDS["xgboost_shadow_outcome"]
    )
    assert "-mode combined_shadow" in joined
    for token in (
        "xgboost_blocking",
        "iforest_blocking",
        "survival_active",
        "advanced_risk_active",
        "place_real_orders",
        "mainnet",
    ):
        assert token not in joined
    assert commands["prohibited_commands_generated"] is False


def test_evidence_gate_matrix_has_required_fields_and_prohibitions(tmp_path):
    matrix = _summary(tmp_path)["evidence_gate_matrix"]
    required = {
        "current_status",
        "evidence_available",
        "evidence_required",
        "gate_satisfied",
        "allowed_next_action",
        "prohibited_action",
        "reason",
    }

    assert set(matrix) == {
        "baseline",
        "isolation_forest",
        "xgboost",
        "survival_exit",
        "advanced_risk",
        "combined_shadow",
    }
    assert all(set(item) == required for item in matrix.values())
    assert "blocking" in matrix["xgboost"]["prohibited_action"].lower()
    assert "activate" in matrix["survival_exit"]["prohibited_action"].lower()
    assert matrix["combined_shadow"]["allowed_next_action"] == (
        "continued paper shadow integration only"
    )


def test_json_writer_produces_valid_complete_report(tmp_path):
    summary = _summary(tmp_path)
    out = write_json_summary(
        summary,
        tmp_path / "out" / "offline_calibration_proposals.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert list(payload) == [
        "evidence_manifest_status",
        "phase17_manifest_digest_match",
        "excluded_run_count",
        "excluded_run_identities",
        "input_evidence_inventory",
        "global_safety_gate",
        "baseline_evidence_proposal",
        "isolation_forest_retraining_specification",
        "xgboost_evidence_collection_proposal",
        "survival_probability_calibration_specification",
        "advanced_risk_parameter_proposal",
        "combined_validation_plan",
        "proposed_experiment_commands",
        "evidence_gate_matrix",
        "final_recommendation",
    ]
    assert payload["final_recommendation"]["final_verdict"] == (
        "calibration_proposals_ready_paper_only"
    )


def test_terminal_summary_has_all_sections_and_final_verdict(tmp_path):
    text = format_text_summary(_summary(tmp_path))

    for heading in "ABCDEFGHIJK":
        assert f"{heading}." in text
    assert "calibration_proposals_ready_paper_only" in text
    assert "prohibited_commands_generated=false" in text


def test_cli_writes_json_and_prints_final_verdict(tmp_path):
    reports = tmp_path / "reports"
    logs = tmp_path / "logs"
    out = tmp_path / "out" / "offline_calibration_proposals.json"
    _write_phase17(reports)

    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--reports-dir",
            str(reports),
            "--logs-dir",
            str(logs),
            "--xgb-min-matched-per-group",
            "30",
            "--baseline-min-closed-trades",
            "100",
            "--if-target-block-rate",
            "0.05",
            "--survival-target-exit-rate",
            "0.10",
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
    assert "calibration_proposals_ready_paper_only" in result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["global_safety_gate"]["status"] == "locked"
    assert payload["final_recommendation"]["final_verdict"] == (
        "calibration_proposals_ready_paper_only"
    )
