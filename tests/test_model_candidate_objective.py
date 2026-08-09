from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tools import model_candidate_objective as objective


def test_policy_is_exact_candidate_only_and_non_promoting():
    policy = objective.load_objective_policy()
    assert objective.validate_objective_contract(policy) == policy
    assert policy["candidate_only"] is True
    assert policy["modify_incumbent_training"] is False
    assert policy["modify_live_inference"] is False
    assert policy["promotion_allowed"] is False


def test_training_scales_are_population_std_of_finite_training_endpoints_only():
    result = objective.compute_training_target_scales(
        [-0.02, -0.01, 0.01, 0.02, np.nan],
        [0.01, 0.02, 0.03, 0.04, np.nan],
    )
    assert result["ret_target_scale"] == np.std([-0.02, -0.01, 0.01, 0.02])
    assert result["rv_target_scale"] == np.std([0.01, 0.02, 0.03, 0.04])
    assert result["source"] == "training_sequences_only"
    assert result["rv_target_negative_count"] == 0


def test_validation_test_repair_and_confirmation_changes_cannot_affect_training_scales():
    train_ret = np.array([-2.0, -1.0, 1.0, 2.0])
    train_rv = np.array([1.0, 2.0, 3.0, 4.0])
    first = objective.compute_training_target_scales(train_ret, train_rv)
    unrelated = {
        "validation": np.full(20, 1e9),
        "internal_test": np.full(20, -1e9),
        "legacy_repair": np.arange(100),
        "confirmation": np.arange(100) ** 2,
    }
    for key in unrelated:
        changed = {name: value.copy() for name, value in unrelated.items()}
        changed[key] = changed[key] * 999
        second = objective.compute_training_target_scales(train_ret, train_rv)
        assert first == second


@pytest.mark.parametrize(
    "ret,rv,match",
    [
        ([1.0, 1.0], [1.0, 2.0], "target scale"),
        ([1.0, 2.0], [1.0, 1.0], "target scale"),
        ([1.0, np.inf], [1.0, np.nan], "target scale"),
        ([1.0, 2.0], [-0.1, 0.2], "negative"),
    ],
)
def test_invalid_or_negative_training_targets_fail_closed(ret, rv, match):
    with pytest.raises(objective.CandidateObjectiveError, match=match):
        objective.compute_training_target_scales(ret, rv)


def _loss_inputs():
    logits = torch.tensor([[0.1, 0.3], [0.8, -0.2]], dtype=torch.float64, requires_grad=True)
    outputs = {
        "ret_cls_logits": logits,
        "ret_reg": torch.tensor([0.03, -0.01], dtype=torch.float64, requires_grad=True),
        "rv_reg": torch.tensor([0.02, 0.04], dtype=torch.float64, requires_grad=True),
    }
    targets = {
        "y_ret_cls": torch.tensor([1, 0]),
        "y_ret_reg": torch.tensor([0.01, -0.02], dtype=torch.float64),
        "y_rv_reg": torch.tensor([0.01, 0.02], dtype=torch.float64),
    }
    return outputs, targets


def test_classification_loss_matches_legacy_weighted_cross_entropy_exactly():
    outputs, targets = _loss_inputs()
    weights = torch.tensor([1.4, 0.6], dtype=torch.float64)
    actual = objective.classification_loss(outputs["ret_cls_logits"], targets["y_ret_cls"], weights)
    expected = torch.nn.CrossEntropyLoss(weight=weights)(outputs["ret_cls_logits"], targets["y_ret_cls"])
    assert torch.equal(actual, expected)


def test_normalized_residual_losses_and_total_formula_are_exact():
    outputs, targets = _loss_inputs()
    parts = objective.candidate_multitask_loss(
        outputs, targets, ret_scale=0.02, rv_scale=0.01,
    )
    ret = torch.mean(((outputs["ret_reg"] - targets["y_ret_reg"]) / 0.02) ** 2)
    rv = torch.mean(((outputs["rv_reg"] - targets["y_rv_reg"]) / 0.01) ** 2)
    cls = torch.nn.CrossEntropyLoss()(outputs["ret_cls_logits"], targets["y_ret_cls"])
    assert torch.equal(parts["return_regression_loss"], ret)
    assert torch.equal(parts["rv_regression_loss"], rv)
    assert torch.equal(parts["total_loss"], cls + 0.5 * ret + 0.5 * rv)


def test_normalized_huber_uses_normalized_residual_beta_one_and_raw_outputs():
    prediction = torch.tensor([2.0, 4.0])
    target = torch.tensor([1.0, 2.0])
    observed = objective.normalized_huber_return_loss(prediction, target, 2.0, beta=1.0)
    residual = (prediction - target) / 2.0
    expected = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)
    assert torch.equal(observed, expected)
    assert torch.equal(prediction, torch.tensor([2.0, 4.0]))


def test_resolved_formulation_A_exactly_matches_phase24_1_loss_and_classification():
    outputs, targets = _loss_inputs()
    descriptor = {"formulation_id": "normalized_mse_fixed", "classification_weight": 1.0,
                  "return_weight": 0.5, "rv_weight": 0.5, "huber_beta": None}
    old = objective.candidate_multitask_loss(outputs, targets, ret_scale=0.02, rv_scale=0.01)
    new = objective.resolved_candidate_loss(
        outputs, targets, ret_scale=0.02, rv_scale=0.01, formulation=descriptor
    )
    for key in ("total_loss", "classification_loss", "return_regression_loss", "rv_regression_loss"):
        assert torch.equal(old[key], new[key])
    assert torch.equal(new["weighted_return_loss"], new["return_regression_loss"] * 0.5)


def test_resolved_huber_total_formula_and_negative_rv_target_fail_closed():
    outputs, targets = _loss_inputs()
    descriptor = {"formulation_id": "normalized_huber_training_balanced",
                  "classification_weight": 1.0, "return_weight": 0.2,
                  "rv_weight": 0.3, "huber_beta": 1.0}
    losses = objective.resolved_candidate_loss(
        outputs, targets, ret_scale=0.02, rv_scale=0.01, formulation=descriptor
    )
    assert torch.equal(losses["total_loss"], losses["weighted_classification_loss"]
                       + losses["weighted_return_loss"] + losses["weighted_rv_loss"])
    targets["y_rv_reg"][0] = -0.1
    with pytest.raises(objective.CandidateObjectiveError, match="negative"):
        objective.resolved_candidate_loss(
            outputs, targets, ret_scale=0.02, rv_scale=0.01, formulation=descriptor
        )


def test_zero_auxiliary_weights_reduce_exactly_to_legacy_classification_loss():
    outputs, targets = _loss_inputs()
    parts = objective.candidate_multitask_loss(
        outputs, targets, ret_scale=0.02, rv_scale=0.01,
        return_weight=0.0, rv_weight=0.0,
    )
    legacy = torch.nn.CrossEntropyLoss()(outputs["ret_cls_logits"], targets["y_ret_cls"])
    assert torch.equal(parts["total_loss"], legacy)


def test_objective_digest_is_deterministic_and_weight_sensitive():
    policy = objective.load_objective_policy()
    first = objective.candidate_objective_digest(policy=policy, target_contract_digest="a" * 64)
    assert first == objective.candidate_objective_digest(policy=policy, target_contract_digest="a" * 64)
    changed = copy.deepcopy(policy)
    changed["return_regression"]["weight"] = 0.6
    assert first != objective.candidate_objective_digest(policy=changed, target_contract_digest="a" * 64)


def test_auxiliary_metrics_report_baselines_sign_ic_and_raw_units():
    metrics = objective.objective_metrics(
        ret_prediction=[-0.9, -0.1, 1.1, 1.9], ret_target=[-1.0, 0.0, 1.0, 2.0],
        rv_prediction=[1.1, 1.9, 3.1, 3.9], rv_target=[1.0, 2.0, 3.0, 4.0],
        ret_scale=1.2, rv_scale=1.1, ret_train_target_mean=0.5, rv_train_target_mean=2.5,
    )
    assert metrics["ret_reg"]["sign_accuracy"] >= 0.75
    assert metrics["ret_reg"]["information_coefficient"] is not None
    assert metrics["rv_reg"]["candidate_vs_baseline_rmse_ratio"] < 1
    assert metrics["outputs_remain_in_raw_target_units"] is True
    assert metrics["post_hoc_rv_clipping_applied"] is False


def test_negative_rv_prediction_is_recorded_and_hard_fails_without_clipping():
    metrics = objective.objective_metrics(
        ret_prediction=[-1.0, 1.0], ret_target=[-1.0, 1.0],
        rv_prediction=[-0.01, 0.02], rv_target=[0.01, 0.02],
        ret_scale=1.0, rv_scale=0.01, ret_train_target_mean=0.0, rv_train_target_mean=0.015,
    )
    assert metrics["rv_reg"]["negative_prediction_count"] == 1
    assert metrics["rv_reg"]["negative_prediction_rate"] == 0.5
    assert metrics["rv_reg"]["classification"] == "auxiliary_failed_negative_rv"
    assert metrics["auxiliary_head_gate_passed"] is False
    assert metrics["post_hoc_rv_clipping_applied"] is False


def test_nonfinite_and_constant_outputs_are_hard_failures_but_low_skill_is_warning():
    nonfinite = objective.objective_metrics(
        ret_prediction=[np.nan, 1], ret_target=[0, 1], rv_prediction=[1, 2], rv_target=[1, 2],
        ret_scale=1, rv_scale=1, ret_train_target_mean=0, rv_train_target_mean=1.5,
    )
    assert nonfinite["ret_reg"]["classification"] == "auxiliary_failed_nonfinite"
    constant = objective.objective_metrics(
        ret_prediction=[1, 1, 1], ret_target=[-1, 0, 1], rv_prediction=[1, 1, 1], rv_target=[1, 2, 3],
        ret_scale=1, rv_scale=1, ret_train_target_mean=0, rv_train_target_mean=2,
    )
    assert constant["ret_reg"]["classification"] == "auxiliary_failed_constant_output"
    warning = objective.objective_metrics(
        ret_prediction=[1, 0, -1], ret_target=[-1, 0, 1], rv_prediction=[3, 2, 1], rv_target=[1, 2, 3],
        ret_scale=1, rv_scale=1, ret_train_target_mean=0, rv_train_target_mean=2,
    )
    assert warning["ret_reg"]["classification"] == "auxiliary_warning_low_skill"
