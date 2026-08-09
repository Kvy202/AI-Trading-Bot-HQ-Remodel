from __future__ import annotations

from tools import model_auxiliary_head_audit as audit


def test_diagnostic_only_consumer_is_distinguished():
    row = audit.classify_consumer("tools/check_model_dist.py", "rv_hat = row['rv_hat']")
    assert row["classification"] == "diagnostic_only"
    assert row["active_in_current_remodel_path"] is False
    assert row["affects_allow_decision"] is False


def test_active_canonical_executor_rv_gate_is_decision_and_risk_affecting():
    row = audit.classify_consumer(
        "tools/live_executor.py", "if abs(sig.rv_mean) > args.rv_max: continue"
    )
    assert row["classification"] == "risk_input"
    assert row["active_in_current_remodel_path"] is True
    assert row["affects_allow_decision"] is True
    assert row["affects_risk"] is True


def test_legacy_bitget_reference_is_not_promoted_to_current_remodel_path():
    row = audit.classify_consumer(
        "trade_multi_bitget.py", "if rv_hat >= DL_MAX_RV: continue"
    )
    assert row["classification"] == "legacy_only"
    assert row["active_in_current_remodel_path"] is False
    assert row["affects_allow_decision"] is True


def test_optional_advanced_risk_is_shadow_only_not_canonical_blocking():
    row = audit.classify_consumer("ml_optional/advanced_risk.py", "abs(rv_mean) >= volatility")
    assert row["classification"] == "shadow_only"
    assert row["affects_risk"] is True
    assert row["active_in_current_remodel_path"] is False


def test_unit_audit_establishes_direct_raw_candidate_comparison_and_current_disable_override():
    consumers = [
        audit.classify_consumer("tools/live_writer.py", "rv_mean = rv_hat"),
        audit.classify_consumer("tools/live_executor.py", "abs(rv_mean) > rv_max"),
    ]
    result = audit.assess_rv_unit_compatibility(consumers)
    assert result["unit_contract_status"] == "compatible_for_resolved_raw_unit_candidate"
    assert result["comparisons"][0]["units_directly_comparable"] is True
    assert result["current_guard_effectively_disabled_by_tracked_override"] is True
    assert result["negative_rv_safety_relevant"] is True
    assert result["downstream_contract_blocker"] is False


def test_unknown_or_missing_unit_path_remains_unverified_and_blocking():
    result = audit.assess_rv_unit_compatibility([
        audit.classify_consumer("tools/live_executor.py", "rv_mean > rv_max")
    ])
    assert result["unit_contract_status"] == "unverified"
    assert result["comparisons"][0]["units_directly_comparable"] is False
    assert result["downstream_contract_blocker"] is True


def test_repository_audit_inventories_active_legacy_and_shadow_consumers():
    result = audit.build_auxiliary_audit()
    paths = {row["path"] for row in result["consumers"]}
    assert {"ml_dl/dl_ensemble.py", "tools/live_writer.py", "tools/live_executor.py"} <= paths
    assert {"trade_multi_bitget.py", "live_ensemble.py", "live_meta_ensemble.py"} <= paths
    assert "ml_optional/advanced_risk.py" in paths
    assert result["rv_hat_affects_current_remodel_decisions"] is True
    assert result["classification_only_safe"] is False
    assert result["decoupling_feasibility"]["implemented"] is False

