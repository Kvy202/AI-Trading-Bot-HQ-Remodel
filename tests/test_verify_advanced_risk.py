"""Tests for the Advanced Risk deterministic verifier."""

from tools.verify_advanced_risk import (
    build_verification_rows,
    format_verification_summary,
    validate_required_cases,
)


def test_verification_rows_cover_required_advanced_risk_cases():
    rows = build_verification_rows()
    errors = validate_required_cases(rows)

    assert errors == []

    by_case = {row["case"]: row for row in rows}
    assert by_case["disabled_flag"]["risk_status"] == "disabled"
    assert by_case["normal_risk"]["risk_status"] == "normal"
    assert by_case["daily_loss_would_block"]["would_block"] == 1
    assert by_case["consecutive_losses_would_block"]["top_reason"] == "consecutive_losses_limit"
    assert by_case["max_open_positions_would_block"]["top_reason"] == "max_open_positions_limit"
    assert by_case["volatility_guard_would_block"]["volatility_guard_triggered"] == 1
    assert by_case["active_false_no_actual_block"]["actually_blocked"] == 0
    assert by_case["active_true_still_shadow_only"]["actually_blocked"] == 0
    assert by_case["active_true_still_shadow_only"]["paper_only_guard"] == "phase10_shadow_only"
    assert by_case["missing_context_does_not_crash"]["risk_status"] == "context_missing"


def test_verification_summary_includes_shadow_only_fields():
    text = format_verification_summary(build_verification_rows())

    assert "Advanced Risk Shadow Verification" in text
    assert "active_true_still_shadow_only" in text
    assert "actually_blocked=0" in text
    assert "would_reduce_size=1" in text


def test_validate_required_cases_reports_missing_case():
    errors = validate_required_cases([])

    assert any("disabled_flag: missing" in err for err in errors)
