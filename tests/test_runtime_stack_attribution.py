from __future__ import annotations

import copy

import pytest

from tools.runtime_stack_attribution import (
    attribute_runtime_stack,
    determine_decomposition_stage,
)
from tools.runtime_stack_isolation import INTERACTION_STACKS, PRIMARY_STACKS


def _decomposition(*, stage: str = "float32_arithmetic"):
    names = (
        "sklearn_transform_float64_input", "sklearn_transform_float32_input",
        "manual_float64_formula", "manual_float64_then_float32", "manual_float32_formula",
    )
    observed_paths = {name: {"output_digest": f"same-{name}"} for name in names}
    serialized_paths = copy.deepcopy(observed_paths)
    if stage == "float32_arithmetic":
        serialized_paths["manual_float32_formula"]["output_digest"] = "different"
        serialized_paths["sklearn_transform_float32_input"]["output_digest"] = "different-sklearn"
    elif stage == "float64_to_float32_conversion":
        serialized_paths["manual_float64_then_float32"]["output_digest"] = "different"
    elif stage == "sklearn_transform_output_handling":
        serialized_paths["sklearn_transform_float32_input"]["output_digest"] = "different"
    return {
        "observed_main": {
            "lstm": {"scaler_metadata": {"mean_digest": "m", "scale_digest": "s"}, "paths": observed_paths}
        },
        "serialized_full_stack": {
            "lstm": {"scaler_metadata": {"mean_digest": "m", "scale_digest": "s"}, "paths": serialized_paths}
        },
    }


def _report(*, exact_stacks=(), unavailable=(), interaction_required=()):
    all_stacks = set(PRIMARY_STACKS) | set(INTERACTION_STACKS)
    return {
        "schema_version": 1,
        "isolation_digest": "a" * 64,
        "environment_matrix": {"stacks": {
            stack_id: {"status": "environment_unavailable" if stack_id in unavailable else "available"}
            for stack_id in all_stacks
        }},
        "stack_comparisons": {
            stack_id: {"matches_serialized_full_pattern": stack_id in exact_stacks, "by_model": {}}
            for stack_id in all_stacks
        },
        "transform_decomposition": _decomposition(),
        "overall_decision": {"interaction_stacks_required": list(interaction_required)},
    }


@pytest.mark.parametrize(
    ("stack_id", "expected"),
    [
        ("sklearn_only_180", "scikit-learn"),
        ("numpy_only_233", "numpy"),
        ("joblib_only_152", "joblib"),
        ("scipy_only_162", "scipy"),
    ],
)
def test_exact_one_package_stack_is_attributed(stack_id, expected):
    result = attribute_runtime_stack(_report(exact_stacks=[stack_id]))
    assert result["primary_contributor"] == expected
    assert result["interaction_required"] is False
    assert result["confidence"] == "confirmed"
    assert result["scaler_serialization_version_alone_used_as_causal_evidence"] is False


def test_multiple_independent_contributors_are_preserved():
    result = attribute_runtime_stack(_report(exact_stacks=["numpy_only_233", "scipy_only_162"]))
    assert result["primary_contributor"] == "multiple_contributors"
    assert result["secondary_contributors"] == ["numpy", "scipy"]
    assert result["confidence"] == "strongly_supported"


def test_two_package_interaction_is_detected():
    report = _report(
        exact_stacks=["numpy_233_sklearn_180"],
        interaction_required=INTERACTION_STACKS,
    )
    result = attribute_runtime_stack(report)
    assert result["primary_contributor"] == "package_interaction"
    assert result["interaction_required"] is True
    assert result["supporting_stacks"] == ["numpy_233_sklearn_180"]


def test_full_stack_only_result_is_multi_package_interaction():
    result = attribute_runtime_stack(_report(interaction_required=INTERACTION_STACKS))
    assert result["primary_contributor"] == "multi_package_interaction"
    assert result["supporting_stacks"] == ["serialized_full_stack"]


def test_unavailable_or_contradictory_scope_remains_unresolved():
    result = attribute_runtime_stack(_report(
        unavailable=["numpy_only_233", "numpy_233_sklearn_180"],
        interaction_required=INTERACTION_STACKS,
    ))
    assert result["primary_contributor"] == "unresolved"
    assert result["confidence"] == "unresolved"
    assert "numpy_only_233" in result["contradicting_stacks"]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("float32_arithmetic", "float32_arithmetic"),
        ("float64_to_float32_conversion", "float64_to_float32_conversion"),
        ("sklearn_transform_output_handling", "sklearn_transform_output_handling"),
    ],
)
def test_decomposition_stage_uses_path_digests(stage, expected):
    report = _report()
    report["transform_decomposition"] = _decomposition(stage=stage)
    assert determine_decomposition_stage(report)["stage"] == expected


def test_changed_loaded_scaler_metadata_is_deserialization_evidence():
    report = _report()
    report["transform_decomposition"]["serialized_full_stack"]["lstm"]["scaler_metadata"]["mean_digest"] = "other"
    assert determine_decomposition_stage(report)["stage"] == "scaler_deserialization"


def test_attribution_digest_is_deterministic_and_not_confidence_overstated():
    report = _report(exact_stacks=["numpy_only_233"])
    first = attribute_runtime_stack(report)
    second = attribute_runtime_stack(copy.deepcopy(report))
    assert first["attribution_digest"] == second["attribution_digest"]
    report["environment_matrix"]["stacks"]["joblib_only_152"]["status"] = "environment_unavailable"
    partial = attribute_runtime_stack(report)
    assert partial["confidence"] == "strongly_supported"
