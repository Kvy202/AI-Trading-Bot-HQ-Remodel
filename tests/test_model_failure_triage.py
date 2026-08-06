from __future__ import annotations

import numpy as np

from tools.model_failure_triage import calculate_input_diagnostics, classify_failure


def test_ood_feature_rates_are_calculated_correctly():
    values = np.asarray([[[0.0, 4.0], [6.0, 0.0]]], dtype=float)
    result = calculate_input_diagnostics(values, ["a", "b"])
    assert result["feature_abs_z_gt_3_rate"] == {"a": 0.5, "b": 0.5}
    assert result["feature_abs_z_gt_5_rate"] == {"a": 0.5, "b": 0.0}
    assert result["maximum_absolute_z"] == 6


def _collapse(std=0.1, flat=0):
    return {"sklearn180_runtime": {
        "collapse_status": "failed_health_gate" if flat else "healthy_aligned",
        "calibrated_probability_std": std, "rolling_flat_window_count": flat,
    }}


def test_learned_saturation_is_distinguishable_from_calibration_saturation():
    learned = classify_failure(
        scaler_classification="exact_match", collapse=_collapse(), input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 1},
        classifier_calibration={"raw_std": 0.0001, "saturation_rate": 0.8, "exclusion_events_before_calibration": 4, "exclusion_events_after_calibration": 4, "clipping_count": 0},
    )
    assert learned["learned_classifier_saturation"]["support"] == "supported"
    assert learned["calibration_saturation"]["support"] == "not_supported"


def test_bias_clipping_and_added_exclusions_support_calibration_saturation():
    result = classify_failure(
        scaler_classification="exact_match", collapse=_collapse(), input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 1},
        classifier_calibration={"raw_std": 0.2, "saturation_rate": 0.0, "exclusion_events_before_calibration": 0, "exclusion_events_after_calibration": 2, "clipping_count": 3},
    )
    assert result["calibration_saturation"]["support"] == "supported"


def test_unsupported_causal_claims_remain_unverified():
    result = classify_failure(
        scaler_classification="comparison_failed", collapse={}, input_diagnostics=None,
        classifier_calibration=None,
    )
    assert result["scaler_runtime_difference"]["support"] == "unverified"
    assert result["insufficient_evidence"]["support"] == "supported"


def test_one_sided_varying_output_is_not_low_dynamic_range():
    result = classify_failure(
        scaler_classification="exact_match", collapse=_collapse(std=0.02),
        input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 1},
        classifier_calibration={"raw_std": 0.02, "saturation_rate": 0.0, "exclusion_events_before_calibration": 0, "exclusion_events_after_calibration": 0, "clipping_count": 0},
    )
    assert result["low_dynamic_range"]["support"] == "not_supported"


def test_clean_healthy_evidence_supports_no_failure_detected():
    result = classify_failure(
        scaler_classification="exact_match", collapse=_collapse(std=0.1),
        input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 1},
        classifier_calibration={
            "raw_std": 0.1, "saturation_rate": 0.0,
            "exclusion_events_before_calibration": 0,
            "exclusion_events_after_calibration": 0, "clipping_count": 0,
        },
        symbol_failure_pattern="all_healthy",
    )
    assert result["no_failure_detected"]["support"] == "supported"


def test_mixed_required_symbol_health_supports_symbol_specific_failure():
    result = classify_failure(
        scaler_classification="numerically_equivalent", collapse=_collapse(std=0.1),
        input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 1},
        classifier_calibration={
            "raw_std": 0.1, "saturation_rate": 0.0,
            "exclusion_events_before_calibration": 0,
            "exclusion_events_after_calibration": 0, "clipping_count": 0,
        },
        symbol_failure_pattern="mixed",
    )
    assert result["symbol_specific_failure"]["support"] == "supported"


def test_single_rare_large_z_is_only_partial_ood_support():
    result = classify_failure(
        scaler_classification="exact_match", collapse=_collapse(std=0.1),
        input_diagnostics={"ood_feature_count": 0, "maximum_absolute_z": 6},
        classifier_calibration={
            "raw_std": 0.1, "saturation_rate": 0.0,
            "exclusion_events_before_calibration": 0,
            "exclusion_events_after_calibration": 0, "clipping_count": 0,
        },
    )
    assert result["serving_distribution_ood"]["support"] == "partially_supported"
