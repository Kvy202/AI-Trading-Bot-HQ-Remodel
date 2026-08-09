from __future__ import annotations

import json

import pytest

from tools import model_candidate_train as train
from tools import model_objective_contract as contract


@pytest.fixture(scope="module")
def resolved_report():
    return contract.build_contract_report()


def test_git_history_recovers_exact_multitask_objective_with_commit_and_lineage_limits(resolved_report):
    history = resolved_report["historical_objective_search"]
    assert history["classification"] == "exact_historical_objective_recovered"
    record = history["exact_records"][0]
    assert record["source_commit"] == "1c5aac957194b80c44e1689d57f46b7a3c1d4134"
    assert "MSE(ret_reg" in record["loss_formula"]
    assert record["task_weights"] == {
        "return_regression": 1.0, "volatility_regression": 1.0, "classification": 1.0,
    }
    assert history["lineage_assessment"]["plausibly_produced_current_incumbents"] is False
    assert history["lineage_assessment"]["adoption_decision"] == "retain_as_lineage_evidence_only"


def test_partial_history_is_not_exact_no_evidence_falls_back_and_conflicts_fail_closed():
    partial = contract.classify_historical_evidence([
        {"evidence_level": "partial", "source_commit": "a"}
    ])
    assert partial["classification"] == "partial_historical_objective_evidence"
    assert contract.classify_historical_evidence([])["classification"] == "no_historical_multitask_objective_found"
    conflicting = contract.classify_historical_evidence([
        {"evidence_level": "exact", "loss_formula": "a", "task_weights": {}, "target_definitions": {}},
        {"evidence_level": "exact", "loss_formula": "b", "task_weights": {}, "target_definitions": {}},
    ])
    assert conflicting["classification"] == "conflicting_historical_objective_evidence"
    assert conflicting["fails_closed"] is True


def test_conflicting_history_blocks_top_level_resolution(monkeypatch):
    historical = contract.search_git_history()
    historical["classification"] = "conflicting_historical_objective_evidence"
    monkeypatch.setattr(contract, "search_git_history", lambda repository: historical)
    report = contract.build_contract_report()
    assert report["promotion_blockers"]["objective_contract_blocker"] is True
    assert report["overall_decision"]["verdict"] == "candidate_objective_contract_incomplete"


def test_top_level_contract_resolves_new_candidate_only_multitask_loss(resolved_report):
    resolved = resolved_report["resolved_candidate_objective"]
    assert resolved["objective_source"] == "new_candidate_only_contract"
    assert resolved["formula"] == "L_cls + 0.5*L_ret + 0.5*L_rv"
    assert resolved["weights"] == {
        "classification": 1.0, "return_regression": 0.5, "volatility_regression": 0.5,
    }
    assert resolved["historical_formula_restored"] is False
    assert resolved_report["overall_decision"]["verdict"] == (
        "candidate_objective_contract_resolved_multitask_training_required"
    )


def test_contract_digest_is_deterministic_and_expected_by_trainer(resolved_report):
    expected = contract.expected_objective_contract_digest()
    assert resolved_report["objective_contract_digest"] == expected
    assert resolved_report["resolved_candidate_objective"]["objective_contract_digest"] == expected
    assert contract.build_contract_report()["objective_contract_digest"] == expected


def test_real_training_gate_refuses_missing_unresolved_or_wrong_digest(tmp_path):
    with pytest.raises(train.ModelCandidateTrainingError, match="report required"):
        train.validate_training_objective_gate(
            train.OBJECTIVE_NAME, report_path=tmp_path / "missing.json"
        )
    wrong = {
        "overall_decision": {"verdict": "candidate_objective_contract_incomplete"},
        "objective_contract_digest": "0" * 64,
    }
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(train.ModelCandidateTrainingError, match="not resolved"):
        train.validate_training_objective_gate(train.OBJECTIVE_NAME, report_path=path)
    with pytest.raises(train.ModelCandidateTrainingError, match="cannot finalize"):
        train.validate_training_objective_gate(train.LEGACY_OBJECTIVE_NAME, report_path=path)


def test_real_training_gate_accepts_resolved_digest_without_accessing_environment(tmp_path, resolved_report):
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(resolved_report), encoding="utf-8")
    accepted = train.validate_training_objective_gate(train.OBJECTIVE_NAME, report_path=path)
    assert accepted["objective_contract_digest"] == contract.expected_objective_contract_digest()


def test_promotion_blocker_semantics_are_separate_and_no_promotion_is_added(resolved_report):
    blockers = resolved_report["promotion_blockers"]
    assert blockers["objective_contract_blocker"] is False
    assert blockers["candidate_auxiliary_health_blocker"] == "unverified"
    assert blockers["downstream_contract_blocker"] is False
    assert blockers["promotion_ready"] is False
    assert resolved_report["overall_decision"]["promotion_allowed"] is False
    assert resolved_report["overall_decision"]["live_activation_allowed"] is False


def test_report_records_active_rv_decision_use_negative_safety_and_raw_unit_compatibility(resolved_report):
    assert resolved_report["downstream_consumers"]["rv_hat_affects_current_remodel_decisions"] is True
    unit = resolved_report["rv_unit_compatibility"]
    assert unit["unit_contract_status"] == "compatible_for_resolved_raw_unit_candidate"
    assert unit["negative_rv_safety_relevant"] is True
    assert unit["negative_rv_handling"] == "fail candidate auxiliary gate; do not clip production inference"
