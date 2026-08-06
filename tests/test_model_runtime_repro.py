from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import tools.model_runtime_repro as repro


@pytest.fixture
def policy():
    return repro.load_retraining_policy()


def test_requirements_digest_is_deterministic():
    assert repro.requirements_digest() == repro.requirements_digest()
    assert len(repro.requirements_digest()) == 64


def test_version_inventory_distinguishes_roles(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("scikit-learn==1.7.2\n", encoding="utf-8")
    repro_req = tmp_path / "repro.txt"
    repro_req.write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    values = iter([
        {"python": "3.13.1", "numpy": "2", "scipy": "1", "joblib": "1", "scikit_learn": "1.7.1"},
        {"python": "3.13.1", "numpy": "2", "scipy": "1", "joblib": "1", "scikit_learn": "1.8.0"},
    ])
    monkeypatch.setattr(repro, "_interpreter_inventory", lambda path: next(values))
    monkeypatch.setattr(repro, "_pip_freeze_digest", lambda path: "f" * 64)
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")
    result = repro.collect_dependency_inventory(
        current_python=py, repro_python=py, requirements_file=requirements,
        repro_requirements=repro_req, snapshot={"model_entries": [{"scaler_serialized_sklearn_version": "1.8.0"}]},
    )
    assert result["version_roles"] == {
        "declared_runtime_version": "1.7.2",
        "observed_runtime_version": "1.7.1",
        "serialized_artifact_versions": ["1.8.0"],
        "comparison_runtime_version": "1.8.0",
    }


@pytest.mark.parametrize("main,reproduction,message", [
    ("3.13.1", "3.13.1", "scikit-learn"),
    ("3.13.1", "3.12.9", "major/minor"),
])
def test_wrong_reproduction_contract_is_rejected(main, reproduction, message):
    inventory = {
        "main_python_version": main, "repro_python_version": reproduction,
        "repro_sklearn_version": "1.7.2" if message == "scikit-learn" else "1.8.0",
    }
    with pytest.raises(repro.RuntimeReproError, match=message):
        repro.validate_repro_environment(inventory)


def _arrays(delta64=0.0, delta32=0.0):
    current64 = np.zeros((2, 3, 2), dtype=np.float64)
    current32 = np.zeros((2, 3, 2), dtype=np.float32)
    other64, other32 = current64.copy(), current32.copy()
    other64[1, 2, 1] = delta64
    other32[1, 2, 1] = delta32
    return current64, other64, current32, other32


def test_identical_transformed_arrays_are_exact_and_digest_is_repeatable():
    result = repro.compare_scaled_arrays(*_arrays(), source_bar_ids=["a", "b"])
    again = repro.compare_scaled_arrays(*_arrays(), source_bar_ids=["a", "b"])
    assert result["classification"] == "exact_match"
    assert result["comparison_digest"] == again["comparison_digest"]
    assert result["float64_exact_equal_rate"] == result["float32_exact_equal_rate"] == 1


def test_tiny_difference_is_numerically_equivalent_and_precisions_stay_separate():
    result = repro.compare_scaled_arrays(*_arrays(1e-13, 0.0), source_bar_ids=["a", "b"])
    assert result["classification"] == "numerically_equivalent"
    assert result["float64_exact_equal_rate"] < 1
    assert result["float32_exact_equal_rate"] == 1


def test_material_difference_reports_largest_location():
    result = repro.compare_scaled_arrays(*_arrays(1e-4, 1e-4), source_bar_ids=["a", "b"])
    assert result["classification"] == "materially_different"
    assert result["largest_error_source_bar_id"] == "b"
    assert result["largest_error_timestep"] == 2
    assert result["largest_error_feature_index"] == 1


class TinyModel:
    def __init__(self):
        import torch
        self.model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    def eval(self):
        self.model.eval()
        return self

    def cpu(self):
        self.model.cpu()
        return self

    def zero_grad(self, *args, **kwargs):
        return self.model.zero_grad(*args, **kwargs)

    def __call__(self, x):
        logits = self.model(x[:, -1, :])
        return {"ret_reg": logits[:, 0], "rv_reg": logits[:, 1], "ret_cls_logits": logits}


def test_identical_scaled_input_produces_identical_model_output(policy):
    values = np.arange(24, dtype=np.float32).reshape(4, 3, 2) / 10
    result, *_ = repro.compare_model_outputs(
        TinyModel(), values, values.copy(), source_bar_ids=[str(i) for i in range(4)],
        bias=0.0, temperature=1.0, policy=policy,
    )
    assert result["classification"] == "output_exact_match"
    assert result["current_forward_deterministic"] is True


def _record(values):
    return {
        "raw_probability": np.asarray(values), "after_bias_probability": np.asarray(values),
        "after_temperature_probability": np.asarray(values),
        "ret_hat": np.zeros(len(values)), "rv_hat": np.zeros(len(values)),
    }


def test_probability_direction_and_exclusion_changes_are_material(policy):
    current, other = _record([0.49, 0.2]), _record([0.51, 0.2])
    states_a = [{"flat": False, "excluded": False}, {"flat": False, "excluded": False}]
    states_b = [{"flat": False, "excluded": True}, {"flat": False, "excluded": False}]
    result = repro.compare_model_output_records(
        current, other, current_states=states_a, repro_states=states_b, policy=policy
    )
    assert result["changed_direction_count"] == 1
    assert result["changed_exclusion_event_count"] == 1
    assert result["classification"] == "output_materially_different"


def test_regression_tolerance_is_enforced(policy):
    current, other = _record([0.4]), _record([0.4])
    other["ret_hat"][0] = policy["regression_output_max_abs_error"] * 2
    assert repro.compare_model_output_records(current, other, policy=policy)["classification"] == "output_materially_different"


def test_calibration_order_matches_production():
    biased, calibrated = repro.apply_calibration(np.asarray([0.6]), bias=0.1, temperature=2.0)
    assert biased[0] == pytest.approx(0.5)
    assert calibrated[0] == pytest.approx(0.5)


def test_persistent_extreme_and_flat_collapse_are_independent(policy):
    extreme, _ = repro.collapse_statistics(
        [0.01] * 25, [f"e{i}" for i in range(25)], policy=policy
    )
    flat, _ = repro.collapse_statistics(
        [0.4] * 35, [f"f{i}" for i in range(35)], policy=policy
    )
    assert extreme["extreme_exclusion_events"] == 1
    assert flat["rolling_flat_window_count"] > 0
    assert extreme["rolling_flat_window_count"] == 0  # extreme gate starts before the flat window


def test_collapse_resolution_is_strict_and_small_mean_movement_does_not_resolve(policy):
    healthy, _ = repro.collapse_statistics(
        np.linspace(0.2, 0.8, 40), [str(i) for i in range(40)], policy=policy
    )
    assert repro.compare_collapse_status({"model_health_status": "failed_flat_at_5m"}, healthy)["collapse_resolved_under_180"]
    still_failed = deepcopy(healthy)
    still_failed.update({"collapse_status": "failed_health_gate", "rolling_flat_window_count": 1, "excluded_endpoint_count": 1})
    comparison = repro.compare_collapse_status({"model_health_status": "failed_flat_at_5m"}, still_failed)
    assert comparison["collapse_persists_under_180"] and not comparison["collapse_resolved_under_180"]


def test_duplicate_bar_advances_health_once(policy):
    stats, states = repro.collapse_statistics([0.01] * 21, ["same"] * 21, policy=policy, expected_count=1)
    assert stats["consecutive_extreme_max"] == 1
    assert len(states) == 1


def test_probability_difference_inside_tolerance_is_numerically_equivalent(policy):
    current, other = _record([0.4]), _record([0.4 + 1e-9])
    result = repro.compare_model_output_records(current, other, policy=policy)
    assert result["classification"] == "output_numerically_equivalent"


def test_neutral_to_directional_change_is_material(policy):
    result = repro.compare_model_output_records(_record([0.5]), _record([0.49]), policy=policy)
    assert result["changed_direction_count"] == 1
    assert result["classification"] == "output_materially_different"


def test_changed_extreme_classification_is_material(policy):
    result = repro.compare_model_output_records(_record([0.049]), _record([0.051]), policy=policy)
    assert result["changed_extreme_classification_count"] == 1
    assert result["classification"] == "output_materially_different"


def test_changed_exclusion_event_is_material_even_when_endpoint_state_matches(policy):
    states_a = [{"flat": True, "excluded": True, "event_types": ["flat_output"]}]
    states_b = [{"flat": True, "excluded": True, "event_types": []}]
    result = repro.compare_model_output_records(
        _record([0.4]), _record([0.4]),
        current_states=states_a, repro_states=states_b, policy=policy,
    )
    assert result["changed_exclusion_event_count"] == 1
    assert result["classification"] == "output_materially_different"


class NondeterministicModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        result = super().__call__(x)
        result["ret_cls_logits"] = result["ret_cls_logits"] + self.calls * 1e-3
        return result


def test_nondeterministic_cpu_inference_fails_comparison(policy):
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10
    result, *_ = repro.compare_model_outputs(
        NondeterministicModel(), values, values,
        source_bar_ids=["a", "b"], bias=0.0, temperature=1.0, policy=policy,
    )
    assert result["classification"] == "output_comparison_failed"
    assert result["current_forward_deterministic"] is False


def test_stricter_missing_rate_is_used_for_collapse_resolution(policy):
    healthy, _ = repro.collapse_statistics(
        np.linspace(0.2, 0.8, 40), [str(i) for i in range(40)],
        policy=policy, expected_count=41,
    )
    result = repro.compare_collapse_status(
        {"model_health_status": "failed_missing_output"}, healthy,
        maximum_missing_rate=0.01,
    )
    assert result["collapse_resolved_under_180"] is False


def test_report_outputs_cannot_target_incumbent_artifacts(tmp_path):
    outside = repro.ensure_safe_report_output(tmp_path / "report.json")
    assert outside == (tmp_path / "report.json").resolve()
    with pytest.raises(repro.RuntimeReproError, match="reports"):
        repro.ensure_safe_report_output(Path("model_artifacts/dl_lstm_metadata.json"))


def test_retraining_policy_cannot_disable_required_gates(tmp_path):
    value = repro.load_retraining_policy()
    value["require_purged_split"] = False
    path = tmp_path / "policy.json"
    path.write_text(__import__("json").dumps(value), encoding="utf-8")
    with pytest.raises(repro.RuntimeReproError, match="disabled"):
        repro.load_retraining_policy(path)
