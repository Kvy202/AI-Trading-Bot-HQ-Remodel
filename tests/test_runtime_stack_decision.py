from __future__ import annotations

import copy
import hashlib

import pytest

from tools.runtime_stack_decision import (
    RuntimeStackDecisionError,
    normalized_lock_digest,
    normalized_lock_text,
    select_canonical_runtime,
    write_canonical_outputs,
)
from tools.runtime_stack_isolation import PACKAGE_KEYS, load_isolation_policy
from tools.model_runtime_repro import load_retraining_policy
from tools.model_retraining_triage import build_retraining_triage_report


SERIALIZED = {
    "numpy": "2.3.3", "scipy": "1.16.2", "joblib": "1.5.2",
    "scikit-learn": "1.8.0", "threadpoolctl": "3.6.0",
}
OBSERVED = {
    "numpy": "2.3.2", "scipy": "1.16.1", "joblib": "1.5.1",
    "scikit-learn": "1.7.1", "threadpoolctl": "3.6.0",
}
BEHAVIOR_FIELDS = (
    "changed_raw_direction_count", "changed_calibrated_direction_count",
    "changed_extreme_state_count", "changed_flat_window_count",
    "changed_exclusion_event_count", "changed_excluded_endpoint_count",
    "changed_allow_count", "changed_signal_direction_count",
    "changed_agreement_suppression_count", "changed_ensemble_variant_count",
)


def _sources():
    output = {field: 0 for field in BEHAVIOR_FIELDS}
    output["deterministic_repeat_status"] = "deterministic"
    isolation = {
        "schema_version": 1,
        "isolation_digest": "1" * 64,
        "policy": {"models_modified": False, "main_environment_modified": False},
        "input_bundle": {
            "bundle_digest": "2" * 64, "alignment_digest": "3" * 64,
            "integrity_result": "pass", "feature_window_digest_result": "pass",
        },
        "environment_matrix": {"stacks": {
            "observed_main": {
                "status": "available", "package_versions": copy.deepcopy(OBSERVED),
                "python_version": "3.13.5",
            },
            "serialized_full_stack": {
                "status": "available", "package_versions": copy.deepcopy(SERIALIZED),
                "python_version": "3.13.5",
            },
        }},
        "reproducibility_levels": {"serialized_full_stack": {
            "bitwise_status": "not_bitwise_reproducible",
            "numerical_status": "numerically_material_difference",
            "behavioral_status": "behaviorally_reproducible",
            "deterministic_workers": True,
            "deterministic_model_inference": True,
        }},
        "model_output_comparisons": {"serialized_full_stack": {
            "by_model": {"lstm": {"by_symbol": {"BTCUSDT": output, "ETHUSDT": copy.deepcopy(output)}}}
        }},
    }
    attribution = {
        "schema_version": 1,
        "source_isolation_digest": isolation["isolation_digest"],
        "primary_contributor": "numpy",
        "confidence": "confirmed",
        "attribution_digest": "4" * 64,
    }
    return isolation, attribution


def test_serialized_stack_is_selected_and_numeric_materiality_is_retained():
    isolation, attribution = _sources()
    result = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    assert result["decision_status"] == "canonical_stack_selected"
    assert result["selected_stack_id"] == "serialized_full_stack"
    assert result["numerical_status"] == "numerically_material_difference"
    assert result["behavioral_status"] == "behaviorally_reproducible"
    assert result["phase24_candidate_training_allowed"] is True


def test_current_main_stays_noncanonical_and_migration_and_live_are_blocked():
    result = select_canonical_runtime(*_sources(), load_isolation_policy())
    assert result["current_main_runtime_is_canonical"] is False
    assert result["main_runtime_migration_allowed"] is False
    assert result["live_activation_allowed"] is False
    assert result["phase24_environment_scope"] == ".venv-runtime-isolation/serialized_full_stack"


def test_nondeterministic_stack_cannot_be_selected():
    isolation, attribution = _sources()
    isolation["reproducibility_levels"]["serialized_full_stack"]["deterministic_workers"] = False
    result = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    assert result["decision_status"] == "canonical_stack_selection_blocked_environment"
    assert result["phase24_candidate_training_allowed"] is False


@pytest.mark.parametrize(
    "field",
    [
        "changed_raw_direction_count", "changed_exclusion_event_count",
        "changed_allow_count", "changed_agreement_suppression_count",
        "changed_ensemble_variant_count",
    ],
)
def test_any_behavior_change_blocks_selection(field):
    isolation, attribution = _sources()
    isolation["model_output_comparisons"]["serialized_full_stack"]["by_model"]["lstm"]["by_symbol"]["BTCUSDT"][field] = 1
    result = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    assert result["decision_status"] == "canonical_stack_selection_blocked_behavior_difference"
    assert result["final_implementation_verdict"] == "runtime_stack_isolation_material_behavior_difference"


def test_unavailable_stack_or_wrong_serialized_sklearn_blocks_selection():
    isolation, attribution = _sources()
    isolation["environment_matrix"]["stacks"]["serialized_full_stack"]["status"] = "environment_unavailable"
    result = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    assert result["decision_status"] == "canonical_stack_selection_blocked_environment"
    isolation, attribution = _sources()
    isolation["environment_matrix"]["stacks"]["serialized_full_stack"]["package_versions"]["scikit-learn"] = "1.7.2"
    assert select_canonical_runtime(isolation, attribution, load_isolation_policy())["selected_stack_id"] is None


def test_unresolved_attribution_cannot_generate_a_guessed_lock():
    isolation, attribution = _sources()
    attribution["primary_contributor"] = "unresolved"
    attribution["confidence"] = "unresolved"
    result = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    assert result["decision_status"] == "canonical_stack_selection_pending"
    assert result["package_versions"] == {}
    assert result["canonical_lock_digest"] is None
    with pytest.raises(RuntimeStackDecisionError, match="cannot be written"):
        write_canonical_outputs(result, canonical_out="unused.json", lock_out="unused.txt")


def test_exact_lock_is_normalized_and_generated_from_selected_inventory(tmp_path):
    isolation, attribution = _sources()
    decision = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    canonical = tmp_path / "canonical.json"
    lock = tmp_path / "lock.txt"
    write_canonical_outputs(decision, canonical_out=canonical, lock_out=lock)
    expected = "".join(f"{name}=={SERIALIZED[name]}\n" for name in PACKAGE_KEYS)
    assert lock.read_text(encoding="utf-8") == expected
    assert decision["canonical_lock_digest"] == hashlib.sha256(expected.encode()).hexdigest()
    assert normalized_lock_text(SERIALIZED) == expected
    assert normalized_lock_digest(SERIALIZED) == decision["canonical_lock_digest"]


def test_lock_rejects_incomplete_inventory():
    with pytest.raises(RuntimeStackDecisionError, match="incomplete"):
        normalized_lock_text({"numpy": "2.3.3"})


def test_decision_digest_is_deterministic():
    isolation, attribution = _sources()
    first = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    second = select_canonical_runtime(copy.deepcopy(isolation), copy.deepcopy(attribution), load_isolation_policy())
    assert first["decision_digest"] == second["decision_digest"]


def _triage_sources(canonical, attribution):
    runtime = {
        "overall_decision": {
            "verdict": "runtime_reproducibility_material_behavior_delta",
            "full_required_model_and_symbol_scope": True,
            "worker_runs_deterministic": True,
            "model_forward_passes_deterministic": True,
        },
        "collapse_comparisons": {"lstm": {"by_symbol": {
            symbol: {"sklearn180_runtime": {
                "collapse_status": "failed_health_gate", "extreme_exclusion_events": 1,
                "consecutive_extreme_max": 30, "rolling_flat_window_count": 0,
            }} for symbol in ("BTCUSDT", "ETHUSDT")
        }}},
        "model_output_comparisons": {"lstm": {"model_digest": "a" * 64}},
        "input_bundle": {
            "bundle_digest": "b" * 64, "integrity_result": "pass",
            "artifact_integrity_result": "pass",
        },
        "policy": {"models_modified": False, "main_environment_modified": False},
        "reproducibility_digest": "r" * 64,
    }
    phase22 = {
        "historical_alignment": {"bundle_digest": "b" * 64},
        "alignment_digest": "p" * 64,
        "model_results": {"lstm": {"by_symbol": {
            symbol: {"model_health_warnings": []} for symbol in ("BTCUSDT", "ETHUSDT")
        }}},
        "serving_contract": {"training_contract": {
            "ordered_symbols": ["BTCUSDT", "ETHUSDT"]
        }},
    }
    failure = {
        "input_bundle_digest": "b" * 64,
        "runtime_reproducibility_digest": "r" * 64,
        "phase22_alignment_digest": "p" * 64,
        "failure_triage_digest": "f" * 64,
        "overall_decision": {"verdict": "failure_triage_complete"},
        "model_results": {"lstm": {"by_symbol": {}}},
    }
    lineage = {
        "training_lineage_digest": "l" * 64,
        "overall_decision": {"verdict": "training_lineage_legacy_incomplete"},
        "model_results": {"lstm": {
            "lineage_status": "legacy_lineage_incomplete",
            "missing_fields": ["raw_data_digests", "train_split_boundaries"],
            "lineage_fields": {},
        }},
    }
    return runtime, phase22, failure, lineage, canonical, attribution


def test_canonical_resolution_permits_candidate_training_but_preserves_numeric_delta():
    isolation, attribution = _sources()
    canonical = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    runtime, phase22, failure, lineage, canonical, attribution = _triage_sources(canonical, attribution)
    report, spec = build_retraining_triage_report(
        phase22_report=phase22, runtime_report=runtime, failure_report=failure,
        lineage_report=lineage, policy=load_retraining_policy(),
        candidate_registry={"schema_version": 1, "candidates": {}},
        canonical_runtime_decision=canonical, runtime_attribution_report=attribution,
    )
    assert report["phase24_allowed"] is True
    assert report["numerical_reproducibility_status"] == "numerically_material_difference"
    assert report["behavioral_reproducibility_status"] == "behaviorally_reproducible"
    assert report["model_decisions"]["lstm"]["primary_action"] == "retrain_required"
    assert report["live_or_blocking_use_approved"] is False
    assert spec["training_allowed_by_this_specification"] is True
    assert "raw_data_digests" in spec["models"]["lstm"]["legacy_lineage_gaps"]
    assert spec["models"]["lstm"]["required_canonical_numerical_runtime"]["dedicated_environment_only"] is True


def test_material_behavior_difference_keeps_phase24_blocked():
    isolation, attribution = _sources()
    isolation["model_output_comparisons"]["serialized_full_stack"]["by_model"]["lstm"]["by_symbol"]["BTCUSDT"]["changed_allow_count"] = 1
    canonical = select_canonical_runtime(isolation, attribution, load_isolation_policy())
    runtime, phase22, failure, lineage, canonical, attribution = _triage_sources(canonical, attribution)
    report, _ = build_retraining_triage_report(
        phase22_report=phase22, runtime_report=runtime, failure_report=failure,
        lineage_report=lineage, policy=load_retraining_policy(),
        candidate_registry={"schema_version": 1, "candidates": {}},
        canonical_runtime_decision=canonical, runtime_attribution_report=attribution,
    )
    assert report["phase24_allowed"] is False
    assert report["overall_decision"]["verdict"] == "retraining_triage_blocked_runtime_difference"
