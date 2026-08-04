"""Synthetic deterministic-probe and TCN endpoint tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from ml_dl.dl_models import TemporalConvNet
from tools.model_health_probe import (
    apply_probability_calibration,
    calibration_decomposition,
    diagnose_tcn_architecture,
    generate_probe_sequences,
    run_model_probe,
)


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        zeros = torch.zeros(batch, device=x.device)
        return {"ret_reg": zeros, "rv_reg": zeros,
                "ret_cls_logits": torch.stack((zeros, zeros), dim=1)}


class VaryingModel(torch.nn.Module):
    def __init__(self, nonfinite: bool = False):
        super().__init__()
        self.nonfinite = nonfinite

    def forward(self, x):
        score = x[:, -1, 0] * 2.0
        if self.nonfinite:
            score = score * torch.tensor(float("nan"))
        zeros = torch.zeros_like(score)
        return {"ret_reg": score, "rv_reg": score.abs(),
                "ret_cls_logits": torch.stack((zeros, score), dim=1)}


def _scaler(features=3):
    values = np.vstack([np.zeros(features), np.ones(features), -np.ones(features)])
    return StandardScaler().fit(values)


def _probes(features=3):
    return generate_probe_sequences(8, features, seed=123, probe_count=16)


def test_probe_generation_is_deterministic_and_exhaustive():
    first = generate_probe_sequences(8, 3, seed=7, probe_count=16)
    second = generate_probe_sequences(8, 3, seed=7, probe_count=16)

    assert [item["group"] for item in first] == [item["group"] for item in second]
    assert all(np.array_equal(a["values"], b["values"]) for a, b in zip(first, second))
    assert sum(item["group"] == "individual_feature_positive_impulse" for item in first) == 3
    assert sum(item["group"] == "individual_feature_negative_impulse" for item in first) == 3


def test_constant_and_varying_models_have_expected_probe_variance():
    constant = run_model_probe(ConstantModel(), _scaler(), _probes())
    varying = run_model_probe(VaryingModel(), _scaler(), _probes())

    assert constant["p_long_std"] == 0.0
    assert constant["status"] == "failed_flat_output"
    assert varying["p_long_std"] > 0.002
    assert varying["status"] == "passed"
    assert varying["deterministic_repeat_max_error"] == 0.0
    assert varying["deterministic_repeat_passed"] is True


def test_nonfinite_output_is_detected():
    result = run_model_probe(VaryingModel(nonfinite=True), _scaler(), _probes())

    assert result["nonfinite_count"] > 0
    assert result["status"] == "failed_nonfinite_output"


def test_bias_then_temperature_matches_production_order():
    raw = np.asarray([0.2, 0.5, 0.8])
    result = apply_probability_calibration(raw, bias=0.1, temperature=2.0)
    biased = np.clip(raw - 0.1, 1e-6, 1 - 1e-6)
    expected = 1.0 / (1.0 + np.exp(-np.log(biased / (1 - biased)) / 2.0))

    assert np.allclose(result["after_bias"], biased)
    assert np.allclose(result["after_temperature"], expected)


def test_calibration_attribution_detects_preexisting_and_clipped_flatness():
    flat = calibration_decomposition([0.43] * 100, bias=0.0, temperature=1.0)
    clipped = calibration_decomposition(np.linspace(0.01, 0.02, 100), bias=0.5, temperature=1.0)

    assert flat["flatness_attribution"] == "before_calibration"
    assert clipped["flatness_attribution"] == "introduced_by_bias"
    assert clipped["clipping_count"] == 100


def test_feature_and_temporal_sensitivity_locate_the_active_input():
    result = run_model_probe(VaryingModel(), _scaler(4), _probes(4))

    assert result["inactive_feature_count"] == 3
    assert result["last_timestep_sensitivity"] > 0
    assert result["middle_timestep_sensitivity"] == 0
    assert result["first_timestep_sensitivity"] == 0


def _padding_dominated_tcn() -> TemporalConvNet:
    model = TemporalConvNet(1, hid=1, levels=1, kernel=3, dropout=0.0)
    with torch.no_grad():
        first, second = model.net[0], model.net[3]
        first.weight.fill_(1.0)
        first.bias.zero_()
        second.weight.zero_()
        second.weight[:, :, 1:].fill_(1.0)
        second.bias.zero_()
        model.head_ret_cls.weight.zero_()
        model.head_ret_cls.bias.zero_()
        model.head_ret_cls.weight[1, 0] = 5.0
    return model.eval()


def test_tcn_endpoints_record_growth_and_preserve_length_without_state_changes():
    model = TemporalConvNet(1, dropout=0.0).eval()
    before = copy.deepcopy(model.state_dict())
    scaled = np.stack([item["values"] for item in generate_probe_sequences(8, 1, seed=2, probe_count=16)])

    result = diagnose_tcn_architecture(model, scaled)
    endpoints = result["endpoints"]

    assert endpoints["deployed_current_endpoint"]["output_sequence_length_after_each_conv"] == [10, 12, 16, 20, 28, 36, 52, 68]
    assert endpoints["right_cropped_same_length_endpoint"]["output_sequence_length_after_each_conv"] == [8] * 8
    assert endpoints["causal_left_padding_endpoint"]["output_sequence_length_after_each_conv"] == [8] * 8
    assert result["trained_weights_reused"] is True
    assert result["state_dict_modified"] is False
    assert all(torch.equal(before[key], model.state_dict()[key]) for key in before)
    assert all(np.isfinite(value["input_gradient_norm"]) for value in endpoints.values())


def test_padding_dominated_tcn_is_flagged_advisory_only():
    model = _padding_dominated_tcn()
    probes = generate_probe_sequences(8, 1, seed=9, probe_count=16)
    scaled = np.stack([item["values"] for item in probes])

    result = diagnose_tcn_architecture(model, scaled)

    assert result["architecture_issue_suspected"] is True
    assert result["diagnostic_hypothesis_only"] is True


def test_varying_deployed_tcn_is_not_flagged_as_flat_architecture():
    model = _padding_dominated_tcn()
    with torch.no_grad():
        model.net[3].weight[:, :, 0] = 1.0
    scaled = np.stack([item["values"] for item in generate_probe_sequences(8, 1, seed=12, probe_count=16)])

    result = diagnose_tcn_architecture(model, scaled)

    assert result["endpoints"]["deployed_current_endpoint"]["p_long_probe_std"] > 0.002
    assert result["architecture_issue_suspected"] is False
