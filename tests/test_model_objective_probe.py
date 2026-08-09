from __future__ import annotations

import pytest

from tools import model_objective_probe as probe


@pytest.mark.parametrize("kind", ["lstm", "tcn", "tx"])
def test_every_architecture_activates_backbone_and_all_three_heads(kind):
    result = probe.probe_architecture(kind)
    assert result["passed"] is True
    assert result["all_gradients_finite"] is True
    assert all(value > 0 for value in result["gradient_norms"].values())
    assert all(result["optimizer_step_changed"].values())
    assert result["output_shapes"] == {
        "ret_reg": [6], "ret_cls_logits": [6, 2], "rv_reg": [6]
    }


@pytest.mark.parametrize("kind", ["lstm", "tcn", "tx"])
def test_classification_zero_auxiliary_weight_parity_is_exact(kind):
    result = probe.classification_parity_probe(kind)
    assert result["passed"] is True
    assert result["classification_loss_absolute_error"] == 0.0
    assert result["classification_gradient_max_absolute_error"] == 0.0


def test_probe_is_deterministic_and_does_not_retune_weights():
    first = probe.build_probe_report()
    second = probe.build_probe_report()
    assert first["probe_digest"] == second["probe_digest"]
    assert first["all_architectures_passed"] is True
    assert first["task_weights_changed_by_probe"] is False


def test_probe_reports_large_gradient_imbalance_as_warning_only():
    result = probe.probe_architecture("lstm")
    assert result["maximum_component_gradient_ratio"] > 0
    if result["maximum_component_gradient_ratio"] > 100:
        assert "weighted_component_shared_gradient_ratio_exceeds_100x" in result["warnings"]
    assert result["passed"] is True


def test_probe_preserves_raw_outputs_and_introduces_no_clipping():
    result = probe.probe_architecture("tx")
    assert result["raw_unit_output_semantics_preserved"] is True
    assert result["post_hoc_clipping_present"] is False

