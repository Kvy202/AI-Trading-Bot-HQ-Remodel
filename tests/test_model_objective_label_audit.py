from __future__ import annotations

import copy

from tools import model_objective_label_audit as audit


def test_triple_target_semantics_horizons_units_and_purge_are_resolved_from_specification():
    cfg = audit.load_resolved_specification()
    result = audit.resolve_target_contract(cfg)
    assert result["classification_lookahead_bars"] == cfg["max_hold"] == 60
    assert result["ret_reg_lookahead_bars"] == cfg["max_hold"] == 60
    assert result["rv_reg_lookahead_bars"] == cfg["horizon"] == 12
    assert result["maximum_required_purge_bars"] == max(cfg["max_hold"], cfg["horizon"])
    assert result["classification_target"]["horizon_minutes"] == 300
    assert result["return_target"]["horizon_minutes"] == 300
    assert result["volatility_target"]["horizon_minutes"] == 60


def test_ret_target_is_signed_forward_log_return_in_raw_units():
    result = audit.resolve_target_contract(audit.load_resolved_specification())
    target = result["return_target"]
    assert target["mathematical_definition"] == "log(price[t + max_hold]) - log(price[t])"
    assert target["units"] == "dimensionless_log_return"
    assert target["expected_sign_domain"] == "signed_real"
    assert target["minimum_possible_value"] is None


def test_rv_target_is_next_k_root_sum_squared_log_return_and_nonnegative():
    result = audit.resolve_target_contract(audit.load_resolved_specification())
    target = result["volatility_target"]
    assert "sqrt(sum" in target["mathematical_definition"]
    assert target["units"] == "dimensionless_root_sum_squared_log_return"
    assert target["minimum_possible_value"] == 0.0
    assert audit.verify_executable_semantics(result)["rv_target_nonnegative_verified"] is True


def test_rv_horizon_is_independent_and_largest_lookahead_drives_purge():
    cfg = copy.deepcopy(audit.load_resolved_specification())
    cfg["max_hold"] = 8
    cfg["horizon"] = 15
    result = audit.resolve_target_contract(cfg)
    assert result["ret_reg_lookahead_bars"] == 8
    assert result["rv_reg_lookahead_bars"] == 15
    assert result["maximum_required_purge_bars"] == 15


def test_target_contract_digest_is_deterministic_and_changes_with_horizon():
    cfg = audit.load_resolved_specification()
    first = audit.resolve_target_contract(cfg)["target_contract_digest"]
    assert first == audit.resolve_target_contract(cfg)["target_contract_digest"]
    changed = copy.deepcopy(cfg)
    changed["horizon"] += 1
    assert first != audit.resolve_target_contract(changed)["target_contract_digest"]


def test_label_audit_records_source_digest_nan_rules_and_executable_verification():
    result = audit.build_label_audit()
    assert result["classification_target"]["source_code_digest"]
    assert "timeout" in result["classification_target"]["nan_rule"]
    assert result["return_target"]["purge_lookahead_requirement_bars"] == 60
    assert result["volatility_target"]["purge_lookahead_requirement_bars"] == 12
    assert result["executable_semantics"]["ret_formula_verified"] is True

