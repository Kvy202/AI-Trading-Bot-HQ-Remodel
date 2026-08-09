from __future__ import annotations

import numpy as np

from tools import model_candidate_evaluate as evaluate


def test_legacy_bundle_digest_and_windows_are_verified():
    windows = evaluate.load_legacy_repair_windows()
    assert set(windows) == {"BTCUSDT", "ETHUSDT"}
    for values in windows.values():
        assert values["windows"].shape == (120, 64, 27)
        assert len(set(values["source_bar_ids"])) == 120
        assert np.isfinite(values["windows"]).all()


def test_health_statistics_report_required_distribution_and_repeat_fields():
    probabilities = np.linspace(0.2, 0.8, 120)
    stats = evaluate.probability_health_statistics(
        probabilities, [f"bar:{index}" for index in range(120)], expected_count=120
    )
    assert stats["unique_bars"] == 120
    assert stats["probability_mean"] == np.mean(probabilities)
    assert stats["p05"] < stats["median"] < stats["p95"]
    assert stats["rounded_unique_count"] == 120
    assert stats["nonfinite_outputs"] == 0
    assert stats["deterministic_repeat_passed"] is True
    assert stats["critical_failure"] is False


def test_flat_and_extreme_collapse_events_are_detected_without_threshold_weakening():
    ids = [str(index) for index in range(120)]
    flat = evaluate.probability_health_statistics(np.full(120, 0.5), ids)
    extreme = evaluate.probability_health_statistics(np.full(120, 0.99), ids)
    assert flat["flat_exclusion_events"] == 1
    assert flat["rolling_flat_windows"] == 91
    assert extreme["extreme_exclusion_events"] == 1
    assert extreme["maximum_consecutive_extreme"] == 120
    assert flat["critical_failure"] and extreme["critical_failure"]


def test_missing_nonfinite_duplicate_and_deterministic_failure_are_reported():
    stats = evaluate.probability_health_statistics(
        [0.2, np.nan, 0.8], ["a", "b", "a"], expected_count=5,
        deterministic_repeat_error=1e-8,
    )
    assert stats["unique_bars"] == 2
    assert stats["duplicate_bars_ignored"] == 1
    assert stats["nonfinite_outputs"] == 1
    assert stats["missing_rate"] > 0.05
    assert stats["deterministic_repeat_passed"] is False
    assert stats["critical_failure"] is True


def test_incumbent_comparison_reports_repairs_new_failures_and_no_similarity_requirement():
    ids = [str(index) for index in range(120)]
    collapsed = np.full(120, 0.99)
    healthy = np.linspace(0.15, 0.85, 120)
    repaired = evaluate.compare_probability_series(collapsed, healthy, ids)
    regressed = evaluate.compare_probability_series(healthy, collapsed, ids)
    assert repaired["repaired_known_failure"] is True
    assert repaired["candidate_exclusion_events"] == 0
    assert regressed["new_failure"] is True
    assert repaired["mean_absolute_probability_difference"] > 0
    assert 0 <= repaired["direction_overlap"] <= 1


def test_candidate_scaler_is_applied_without_refit():
    import torch

    class CountingScaler:
        n_features_in_ = 27

        def __init__(self):
            self.transform_calls = 0

        def transform(self, values):
            self.transform_calls += 1
            return values + 1

    class Model(torch.nn.Module):
        def forward(self, x):
            score = x[:, -1, 0]
            return {
                "ret_cls_logits": torch.stack([-score, score], dim=1),
                "ret_reg": score * 0,
                "rv_reg": score * 0,
            }

    scaler = CountingScaler()
    result = evaluate.infer_raw_probabilities(Model(), scaler, np.zeros((3, 64, 27), dtype=np.float32))
    assert scaler.transform_calls == 1
    assert np.all(result > 0.5)


def test_raw_inference_returns_unclipped_auxiliary_outputs_in_original_units():
    import torch

    class IdentityScaler:
        n_features_in_ = 27
        def transform(self, values):
            return values

    class Model(torch.nn.Module):
        def forward(self, x):
            raw = x[:, -1, 0]
            return {
                "ret_cls_logits": torch.stack([-raw, raw], dim=1),
                "ret_reg": raw - 3.0,
                "rv_reg": raw - 2.0,
            }

    windows = np.zeros((2, 64, 27), dtype=np.float32)
    windows[1, -1, 0] = 5.0
    outputs = evaluate.infer_raw_outputs(Model(), IdentityScaler(), windows)
    assert outputs["ret_hat"].tolist() == [-3.0, 2.0]
    assert outputs["rv_hat"].tolist() == [-2.0, 3.0]


def test_unlabeled_auxiliary_health_records_negative_rv_without_clipping():
    stats = evaluate.auxiliary_prediction_health({
        "probability": [0.4, 0.6, 0.7],
        "ret_hat": [-0.1, 0.2, 0.3],
        "rv_hat": [-0.01, 0.02, 0.03],
    })
    assert stats["rv_reg"]["negative_prediction_count"] == 1
    assert stats["rv_reg"]["classification"] == "auxiliary_failed_negative_rv"
    assert stats["auxiliary_head_safety_gate_passed"] is False
    assert stats["post_hoc_rv_clipping_applied"] is False
    assert stats["targets_present"] is False


def test_unlabeled_auxiliary_health_fails_nonfinite_constant_and_nondeterministic_outputs():
    nonfinite = evaluate.auxiliary_prediction_health({
        "probability": [0.5, 0.6], "ret_hat": [np.nan, 1.0], "rv_hat": [0.1, 0.2]
    })
    assert nonfinite["ret_reg"]["classification"] == "auxiliary_failed_nonfinite"
    constant = evaluate.auxiliary_prediction_health({
        "probability": [0.5, 0.6], "ret_hat": [1.0, 1.0], "rv_hat": [0.1, 0.1]
    })
    assert constant["ret_reg"]["classification"] == "auxiliary_failed_constant_output"
    repeat = evaluate.auxiliary_prediction_health({
        "probability": [0.5, 0.6], "ret_hat": [1.0, 2.0], "rv_hat": [0.1, 0.2]
    }, deterministic_repeat_error=1e-8)
    assert repeat["auxiliary_head_safety_gate_passed"] is False
