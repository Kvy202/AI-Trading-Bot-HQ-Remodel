from __future__ import annotations

from copy import deepcopy

import pytest

import tools.model_retraining_triage as triage
from tools.model_runtime_repro import load_retraining_policy


def _runtime(statuses, verdict="runtime_reproducibility_verified_no_material_delta"):
    return {
        "overall_decision": {"verdict": verdict},
        "collapse_comparisons": {"lstm": {"by_symbol": {
            symbol: {"sklearn180_runtime": {
                "collapse_status": status, "extreme_exclusion_events": int(status != "healthy_aligned"),
                "consecutive_extreme_max": 20 if status != "healthy_aligned" else 0,
                "rolling_flat_window_count": 0,
            }} for symbol, status in statuses.items()
        }}},
        "model_output_comparisons": {"lstm": {"model_digest": "a" * 64}},
        "input_bundle": {"bundle_digest": "b" * 64, "integrity_result": "pass"},
        "policy": {"models_modified": False, "main_environment_modified": False},
        "reproducibility_digest": "r" * 64,
    }


def _phase22(low_auc=False):
    warnings = ["warning_low_auc"] if low_auc else []
    return {"model_results": {"lstm": {"by_symbol": {
        "BTCUSDT": {"model_health_warnings": warnings}, "ETHUSDT": {"model_health_warnings": warnings},
    }}}}


def _failure():
    return {"model_results": {"lstm": {"by_symbol": {}}}}


def _lineage():
    return {"lineage_status": "legacy_lineage_incomplete", "missing_fields": ["raw_data_digests"]}


def test_persistent_collapse_leads_to_retraining():
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "failed_health_gate", "ETHUSDT": "failed_health_gate"}),
        phase22_report=_phase22(), failure_report=_failure(), lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert result["primary_action"] == "retrain_required"
    assert result["live_or_blocking_use_approved"] is False


def test_healthy_model_leads_only_to_shadow_control_retention():
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "healthy_aligned", "ETHUSDT": "healthy_aligned"}),
        phase22_report=_phase22(), failure_report=_failure(), lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert result["primary_action"] == "retain_incumbent_shadow_control"
    assert result["retention_scope"] == "shadow_only"


def test_symbol_specific_failure_is_preserved():
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "healthy_aligned", "ETHUSDT": "failed_health_gate"}),
        phase22_report=_phase22(), failure_report=_failure(), lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert result["primary_action"] == "symbol_specific_retraining_required"
    assert result["required_failed_symbols"] == ["ETHUSDT"]


def test_low_auc_alone_is_warning_only():
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "healthy_aligned", "ETHUSDT": "healthy_aligned"}),
        phase22_report=_phase22(True), failure_report=_failure(), lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert result["primary_action"] == "retain_incumbent_shadow_control"
    assert result["low_auc_is_warning_only"] is True


def test_material_runtime_difference_blocks_final_retraining_decision():
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "failed_health_gate", "ETHUSDT": "failed_health_gate"}, "runtime_reproducibility_material_behavior_delta"),
        phase22_report=_phase22(), failure_report=_failure(), lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert result["primary_action"] == "no_decision_insufficient_evidence"
    assert result["provisional_action_after_runtime_resolution"] == "retrain_required"
    assert "persistent_extreme_collapse" in result["supporting_reasons"]


def test_empty_candidate_registry_is_valid_and_unsafe_target_is_rejected():
    assert triage.validate_candidate_registry({"schema_version": 1, "candidates": {}})["candidates"] == {}
    with pytest.raises(triage.RetrainingTriageError, match="incumbent|unsafe"):
        triage.validate_candidate_registry({"schema_version": 1, "candidates": {
            "lstm-test": {"candidate_id": "lstm-test", "status": "proposed", "reviewed": False,
                          "artifact_manifest_path": "model_artifacts/dl_lstm_latest.pt"}
        }})


def test_candidate_ids_are_deterministic():
    kwargs = dict(parent_model_digest="a" * 64, dataset_digest="b" * 64, feature_digest="c" * 64,
                  label_digest="d" * 64, training_config_digest="e" * 64, seed=42)
    assert triage.deterministic_candidate_id("lstm", **kwargs) == triage.deterministic_candidate_id("lstm", **kwargs)


def test_spec_carries_lineage_gaps_and_never_targets_latest():
    runtime = _runtime({"BTCUSDT": "failed_health_gate", "ETHUSDT": "failed_health_gate"})
    decision = {"lstm": {"primary_action": "retrain_required", "supporting_reasons": ["persistent_extreme_collapse"], "required_failed_symbols": ["BTCUSDT", "ETHUSDT"]}}
    lineage = {"model_results": {"lstm": {"missing_fields": ["raw_data_digests"], "lineage_fields": {"label_configuration": {"type": "triple"}}}}}
    spec = triage.generate_retraining_specification(
        decisions=decision, runtime_report=runtime, phase22_report={"serving_contract": {"training_contract": {"ordered_symbols": ["BTCUSDT", "ETHUSDT"]}}},
        lineage_report=lineage, policy=load_retraining_policy(),
    )
    model = spec["models"]["lstm"]
    assert "raw_data_digests" in model["legacy_lineage_gaps"]
    assert "latest" not in str(model["artifact_naming_policy"]).lower()
    assert model["candidate_created_or_trained_by_phase23"] is False


def test_registry_rejects_secret_command_and_path_traversal_material():
    base = {"candidate_id": "lstm-test", "status": "proposed", "reviewed": False}
    invalid_records = [
        {**base, "reason": "api_key=super-secret"},
        {**base, "reason": "powershell -File activate.ps1"},
        {**base, "reason": "PATH=C:/Users/alice/bin"},
        {**base, "artifact_manifest_path": "model_artifacts/candidates/lstm-test/../../dl_lstm_latest.pt"},
    ]
    for record in invalid_records:
        with pytest.raises(triage.RetrainingTriageError):
            triage.validate_candidate_registry({
                "schema_version": 1, "candidates": {"lstm-test": record}
            })


def test_partial_failure_evidence_is_not_promoted_to_supported_reason():
    failure = {"model_results": {"lstm": {"by_symbol": {
        "BTCUSDT": {"failure_categories": {
            "serving_distribution_ood": {"support": "partially_supported"},
            "learned_classifier_saturation": {"support": "partially_supported"},
        }}
    }}}}
    result = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=_runtime({"BTCUSDT": "healthy_aligned", "ETHUSDT": "healthy_aligned"}),
        phase22_report=_phase22(), failure_report=failure,
        lineage_result=_lineage(), minimum_auc=0.55,
    )
    assert "serving_ood_supported" not in result["supporting_reasons"]
    assert "learned_saturation_supported" not in result["supporting_reasons"]


def test_registry_validates_digest_and_seed_types():
    with pytest.raises(triage.RetrainingTriageError, match="dataset_digest"):
        triage.validate_candidate_registry({"schema_version": 1, "candidates": {
            "lstm-test": {"status": "proposed", "dataset_digest": "not-a-digest"}
        }})
    with pytest.raises(triage.RetrainingTriageError, match="seed"):
        triage.validate_candidate_registry({"schema_version": 1, "candidates": {
            "lstm-test": {"status": "proposed", "seed": True}
        }})


def test_runtime_blocked_retraining_still_gets_a_blocked_contract_specification():
    runtime = _runtime(
        {"BTCUSDT": "failed_health_gate", "ETHUSDT": "failed_health_gate"},
        "runtime_reproducibility_material_behavior_delta",
    )
    decision = triage.decide_model_action(
        "lstm", required_symbols=["BTCUSDT", "ETHUSDT"],
        runtime_report=runtime, phase22_report=_phase22(), failure_report=_failure(),
        lineage_result=_lineage(), minimum_auc=0.55,
    )
    spec = triage.generate_retraining_specification(
        decisions={"lstm": decision}, runtime_report=runtime,
        phase22_report={"serving_contract": {"training_contract": {
            "ordered_symbols": ["BTCUSDT", "ETHUSDT"]
        }}},
        lineage_report={"model_results": {"lstm": _lineage()}},
        policy=load_retraining_policy(),
    )
    assert spec["models"]["lstm"]["specification_status"] == "blocked_pending_runtime_difference_resolution"
    assert spec["models"]["lstm"]["training_dataset_digest"] is None
    assert spec["training_allowed_by_this_specification"] is False


def test_phase24_allowance_requires_complete_deterministic_runtime_evidence():
    runtime = _runtime({"BTCUSDT": "failed_health_gate", "ETHUSDT": "failed_health_gate"})
    runtime["overall_decision"].update({
        "full_required_model_and_symbol_scope": False,
        "worker_runs_deterministic": True,
        "model_forward_passes_deterministic": True,
    })
    runtime["input_bundle"]["artifact_integrity_result"] = "pass"
    phase22 = {**_phase22(), "historical_alignment": {"bundle_digest": "b" * 64}, "alignment_digest": "p" * 64}
    failure = {
        **_failure(), "input_bundle_digest": "b" * 64,
        "runtime_reproducibility_digest": "r" * 64,
        "phase22_alignment_digest": "p" * 64,
        "failure_triage_digest": "f" * 64,
        "overall_decision": {"verdict": "failure_triage_complete"},
    }
    lineage = {
        "model_results": {"lstm": _lineage()},
        "overall_decision": {"verdict": "training_lineage_legacy_incomplete"},
        "training_lineage_digest": "l" * 64,
    }
    report, _ = triage.build_retraining_triage_report(
        phase22_report=phase22, runtime_report=runtime, failure_report=failure,
        lineage_report=lineage, policy=load_retraining_policy(),
        candidate_registry={"schema_version": 1, "candidates": {}},
    )
    assert report["phase24_allowed"] is False
    assert report["runtime_comparison_complete"] is False
