"""Tests for the Isolation Forest blocking verification workflow."""

from pathlib import Path

from tools.verify_isolation_forest import (
    build_verification_rows,
    format_summary,
    probe_artifact,
    summarize_decisions,
    validate_required_cases,
)


def test_verification_rows_cover_required_blocking_cases(tmp_path):
    artifact = tmp_path / "isolation_forest.joblib"

    rows = build_verification_rows(artifact, base_dir=tmp_path)
    errors = validate_required_cases(rows)

    assert errors == []
    by_case = {row["case"]: row for row in rows}

    assert by_case["normal_prediction_allows_signal"]["final_allow"] == 1
    assert by_case["normal_prediction_allows_signal"]["actually_blocked"] == 0
    assert by_case["normal_prediction_allows_signal"]["reason"] == "normal_market"

    assert by_case["abnormal_prediction_blocks_signal"]["final_allow"] == 0
    assert by_case["abnormal_prediction_blocks_signal"]["actually_blocked"] == 1
    assert by_case["abnormal_prediction_blocks_signal"]["reason"] == "isolation_forest_block"

    assert by_case["missing_artifact_does_not_block"]["final_allow"] == 1
    assert by_case["missing_artifact_does_not_block"]["actually_blocked"] == 0
    assert by_case["missing_artifact_does_not_block"]["isolation_status"] == "disabled_missing_artifact"

    assert by_case["model_error_does_not_block"]["final_allow"] == 1
    assert by_case["model_error_does_not_block"]["actually_blocked"] == 0
    assert by_case["model_error_does_not_block"]["isolation_status"] == "model_error"

    assert by_case["use_isolation_forest_false_never_blocks"]["final_allow"] == 1
    assert by_case["use_isolation_forest_false_never_blocks"]["actually_blocked"] == 0
    assert by_case["use_isolation_forest_false_never_blocks"]["USE_ISOLATION_FOREST"] == 0

    assert by_case["isolation_forest_blocking_false_never_blocks"]["final_allow"] == 1
    assert by_case["isolation_forest_blocking_false_never_blocks"]["actually_blocked"] == 0
    assert by_case["isolation_forest_blocking_false_never_blocks"]["would_block"] == 1
    assert by_case["isolation_forest_blocking_false_never_blocks"]["ISOLATION_FOREST_BLOCKING"] == 0


def test_verification_summary_reports_blocking_counts(tmp_path):
    artifact = tmp_path / "isolation_forest.joblib"
    rows = build_verification_rows(artifact, base_dir=tmp_path)

    summary = summarize_decisions(rows, artifact)
    text = format_summary(rows, summary, probe_artifact(artifact))

    assert summary["total_signals_checked"] == 6
    assert summary["allowed_count"] == 5
    assert summary["blocked_count"] == 1
    assert summary["block_rate"] == 1 / 6
    assert summary["top_block_reasons"] == {"isolation_forest_block": 1}
    assert summary["latest_anomaly_score"] == -0.42
    assert summary["artifact_path"] == str(artifact)
    assert summary["model_version"] == "verify-anomaly"

    assert "total_signals_checked: 6" in text
    assert "allowed_count: 5" in text
    assert "blocked_count: 1" in text
    assert "top_block_reasons: {'isolation_forest_block': 1}" in text
    assert f"artifact_path: {artifact}" in text


def test_validate_required_cases_reports_missing_case():
    errors = validate_required_cases([])

    assert any("normal_prediction_allows_signal: missing" in err for err in errors)
