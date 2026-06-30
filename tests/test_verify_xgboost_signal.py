"""Tests for the XGBoost blocking verification workflow."""

from pathlib import Path

from ml_optional.xgboost_signal import XGBOOST_SHADOW_COLS
from tools.verify_xgboost_signal import (
    build_verification_rows,
    format_summary,
    probe_artifact,
    summarize_decisions,
    validate_required_cases,
)


def test_verification_rows_cover_required_rejection_cases(tmp_path):
    artifact = tmp_path / "xgboost_signal.joblib"

    rows = build_verification_rows(artifact, base_dir=tmp_path)
    errors = validate_required_cases(rows)

    assert errors == []
    for row in rows:
        assert all(col in row for col in XGBOOST_SHADOW_COLS)

    by_case = {row["case"]: row for row in rows}

    assert by_case["high_confidence_direction_agreement_allows_signal"]["final_allow"] == 1
    assert by_case["high_confidence_direction_agreement_allows_signal"]["actually_rejected"] == 0
    assert by_case["high_confidence_direction_agreement_allows_signal"]["would_confirm"] == 1

    assert by_case["low_confidence_rejects_signal_when_blocking"]["final_allow"] == 0
    assert by_case["low_confidence_rejects_signal_when_blocking"]["actually_rejected"] == 1
    assert by_case["low_confidence_rejects_signal_when_blocking"]["reject_reason"] == "low_confidence"

    assert by_case["direction_mismatch_rejects_signal_when_blocking"]["final_allow"] == 0
    assert by_case["direction_mismatch_rejects_signal_when_blocking"]["actually_rejected"] == 1
    assert by_case["direction_mismatch_rejects_signal_when_blocking"]["reject_reason"] == "direction_mismatch"

    assert by_case["missing_artifact_does_not_reject"]["final_allow"] == 1
    assert by_case["missing_artifact_does_not_reject"]["actually_rejected"] == 0
    assert by_case["missing_artifact_does_not_reject"]["xgboost_status"] == "disabled_missing_artifact"

    assert by_case["model_error_does_not_reject"]["final_allow"] == 1
    assert by_case["model_error_does_not_reject"]["actually_rejected"] == 0
    assert by_case["model_error_does_not_reject"]["xgboost_status"] == "model_error"

    assert by_case["use_xgboost_signal_false_never_rejects"]["final_allow"] == 1
    assert by_case["use_xgboost_signal_false_never_rejects"]["actually_rejected"] == 0
    assert by_case["use_xgboost_signal_false_never_rejects"]["USE_XGBOOST_SIGNAL"] == 0

    assert by_case["xgboost_signal_blocking_false_never_rejects"]["final_allow"] == 1
    assert by_case["xgboost_signal_blocking_false_never_rejects"]["actually_rejected"] == 0
    assert by_case["xgboost_signal_blocking_false_never_rejects"]["would_reject"] == 1
    assert by_case["xgboost_signal_blocking_false_never_rejects"]["XGBOOST_SIGNAL_BLOCKING"] == 0


def test_verification_summary_reports_rejection_counts(tmp_path):
    artifact = tmp_path / "xgboost_signal.joblib"
    rows = build_verification_rows(artifact, base_dir=tmp_path)

    summary = summarize_decisions(rows, artifact)
    text = format_summary(rows, summary, probe_artifact(artifact))

    assert summary["total_signals_checked"] == 7
    assert summary["allowed_count"] == 5
    assert summary["rejected_count"] == 2
    assert summary["reject_rate"] == 2 / 7
    assert summary["top_reject_reasons"] == {
        "low_confidence": 1,
        "direction_mismatch": 1,
    }
    assert summary["latest_confidence"] == 0.9
    assert summary["latest_direction"] == "SHORT"
    assert summary["artifact_path"] == str(artifact)
    assert summary["model_version"] == "verify-direction-mismatch"

    assert "total_signals_checked: 7" in text
    assert "allowed_count: 5" in text
    assert "rejected_count: 2" in text
    assert "top_reject_reasons: {'low_confidence': 1, 'direction_mismatch': 1}" in text
    assert f"artifact_path: {artifact}" in text


def test_validate_required_cases_reports_missing_case():
    errors = validate_required_cases([])

    assert any("high_confidence_direction_agreement_allows_signal: missing" in err for err in errors)
