"""Tests for the Survival active-exit verification workflow."""

from pathlib import Path

from ml_optional.survival_exit import SURVIVAL_SHADOW_COLS
from tools.verify_survival_exit import (
    build_verification_rows,
    format_verification_summary,
    validate_required_cases,
)


def test_verification_rows_cover_required_active_exit_cases(tmp_path):
    artifact = tmp_path / "survival_exit.joblib"

    rows = build_verification_rows(artifact, base_dir=tmp_path)
    errors = validate_required_cases(rows)

    assert errors == []
    for row in rows:
        assert all(col in row for col in SURVIVAL_SHADOW_COLS)

    by_case = {row["case"]: row for row in rows}

    assert by_case["high_risk_exits_when_active_paper"]["actually_exited"] == 1
    assert by_case["high_risk_exits_when_active_paper"]["exit_reason"] == "survival_high_exit_risk"
    assert by_case["high_risk_exits_when_active_paper"]["paper_only_guard"] == "paper_only_ok"

    assert by_case["active_false_never_exits"]["actually_exited"] == 0
    assert by_case["active_false_never_exits"]["paper_only_guard"] == "inactive"

    assert by_case["missing_artifact_never_exits"]["actually_exited"] == 0
    assert by_case["missing_artifact_never_exits"]["survival_status"] == "disabled_missing_artifact"

    assert by_case["model_error_never_exits"]["actually_exited"] == 0
    assert by_case["model_error_never_exits"]["survival_status"] == "prediction_error"

    assert by_case["real_live_mode_never_exits"]["actually_exited"] == 0
    assert by_case["real_live_mode_never_exits"]["paper_only_guard"] == "blocked_real_orders"

    assert by_case["dependency_missing_never_exits"]["actually_exited"] == 0
    assert by_case["dependency_missing_never_exits"]["survival_status"] == "disabled_missing_dependency"


def test_verification_summary_includes_active_exit_fields(tmp_path):
    artifact = tmp_path / "survival_exit.joblib"
    rows = build_verification_rows(artifact, base_dir=tmp_path)

    text = format_verification_summary(rows, artifact, "loaded")

    assert "Survival Exit Active Verification" in text
    assert "high_risk_exits_when_active_paper: actually_exited=1" in text
    assert "paper_only_guard=blocked_real_orders" in text


def test_validate_required_cases_reports_missing_case():
    errors = validate_required_cases([])

    assert any("high_risk_exits_when_active_paper: missing" in err for err in errors)
