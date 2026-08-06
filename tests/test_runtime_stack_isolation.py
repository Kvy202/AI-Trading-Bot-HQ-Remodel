from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from tools.model_runtime_repro import json_digest
from tools.model_scaler_worker import (
    decompose_transform_paths,
    logical_array_digest,
    maximum_ulp_distance,
    windows_payload_digest,
    write_deterministic_npz,
)
from tools.runtime_stack_isolation import (
    BASE_DIR,
    EXPECTED_VERSIONS,
    INTERACTION_STACKS,
    PACKAGE_KEYS,
    PRIMARY_STACKS,
    RuntimeStackIsolationError,
    compare_model_records,
    compare_transform_arrays,
    ensure_safe_working_dir,
    inventory_matrix,
    load_isolation_policy,
    load_stack_matrix,
    stack_python_path,
    validate_stack_inventory,
)


def _inventory(stack_id: str, *, python: str = "3.13.5", forbidden: bool = False):
    values = EXPECTED_VERSIONS[stack_id]
    result = {"python": python}
    result.update(dict(zip(PACKAGE_KEYS, values)))
    result.update({
        "torch_installed": forbidden,
        "ccxt_installed": False,
        "hyperliquid_installed": False,
    })
    return result


def _scaler_and_windows(tmp_path: Path):
    training = np.asarray([
        [1.0, 10.0], [2.0, 20.0], [3.0, 40.0], [8.0, 80.0]
    ], dtype=np.float64)
    scaler = StandardScaler().fit(training)
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    windows = np.asarray([[[1.0, 10.0], [2.0, 30.0]]], dtype=np.float32)
    return scaler_path, scaler, windows


def test_matrix_has_every_exact_stack_and_deterministic_digest():
    first = load_stack_matrix()
    second = load_stack_matrix()
    assert first["matrix_digest"] == second["matrix_digest"]
    assert set(first["stacks"]) == set(EXPECTED_VERSIONS)
    assert set(PRIMARY_STACKS + INTERACTION_STACKS) == set(EXPECTED_VERSIONS)
    for stack_id, versions in EXPECTED_VERSIONS.items():
        observed = first["stacks"][stack_id]["package_versions"]
        assert tuple(observed[name] for name in PACKAGE_KEYS) == versions


def test_policy_is_exact_and_cannot_be_weakened(tmp_path):
    policy = load_isolation_policy()
    assert policy["float32_max_abs_error"] == 1e-7
    policy["float32_max_abs_error"] = 1e-5
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(RuntimeStackIsolationError, match="policy mismatch"):
        load_isolation_policy(path)


def test_isolated_paths_never_target_main_venv():
    for stack_id in EXPECTED_VERSIONS:
        if stack_id == "observed_main":
            continue
        path = stack_python_path(stack_id)
        assert ".venv-runtime-isolation" in str(path)
        assert path != BASE_DIR / ".venv" / "Scripts" / "python.exe"


def test_exact_inventory_accepts_torch_free_isolation_and_main_with_torch():
    matrix = load_stack_matrix()
    validate_stack_inventory("sklearn_only_180", _inventory("sklearn_only_180"), matrix)
    validate_stack_inventory("observed_main", _inventory("observed_main", forbidden=True), matrix)


@pytest.mark.parametrize("field", list(PACKAGE_KEYS))
def test_wrong_or_missing_package_version_refuses_reuse(field):
    matrix = load_stack_matrix()
    inventory = _inventory("sklearn_only_180")
    inventory[field] = None
    with pytest.raises(RuntimeStackIsolationError, match="package mismatch"):
        validate_stack_inventory("sklearn_only_180", inventory, matrix)


def test_wrong_python_minor_is_rejected():
    with pytest.raises(RuntimeStackIsolationError, match="Python major/minor"):
        validate_stack_inventory(
            "sklearn_only_180", _inventory("sklearn_only_180", python="3.12.9"),
            load_stack_matrix(),
        )


@pytest.mark.parametrize("field", ["torch_installed", "ccxt_installed", "hyperliquid_installed"])
def test_trading_or_torch_dependency_is_rejected_from_isolation(field):
    inventory = _inventory("sklearn_only_180")
    inventory[field] = True
    with pytest.raises(RuntimeStackIsolationError, match="prohibited package"):
        validate_stack_inventory("sklearn_only_180", inventory, load_stack_matrix())


def test_unavailable_stack_is_reported_without_substitution():
    inventory = inventory_matrix(load_stack_matrix())
    item = inventory["sklearn_only_180"]
    if not stack_python_path("sklearn_only_180").is_file():
        assert item["status"] == "environment_unavailable"
        assert item["package_versions"]["scikit-learn"] == "1.8.0"


def test_working_directory_guard_accepts_only_ignored_report_work():
    assert ensure_safe_working_dir(BASE_DIR / "reports" / "runtime_stack_synthetic")
    with pytest.raises(RuntimeStackIsolationError):
        ensure_safe_working_dir(BASE_DIR / "model_artifacts" / "runtime_stack_bad")


def test_manual_formula_uses_loaded_mean_scale_and_never_refits(tmp_path):
    scaler_path, scaler, windows = _scaler_and_windows(tmp_path)
    paths, _, _, metadata, decomposition = decompose_transform_paths(scaler_path, windows)
    expected = (
        windows.astype(np.float64) - scaler.mean_.reshape(1, 1, -1)
    ) / scaler.scale_.reshape(1, 1, -1)
    assert np.array_equal(paths["manual_float64_formula"], expected)
    assert metadata["mean_digest"] == logical_array_digest({"mean": scaler.mean_})
    assert decomposition["scaler_refit_performed"] is False


def test_transform_decomposition_records_all_five_paths(tmp_path):
    scaler_path, _, windows = _scaler_and_windows(tmp_path)
    paths, _, _, _, decomposition = decompose_transform_paths(scaler_path, windows)
    assert set(paths) == {
        "sklearn_transform_float64_input", "sklearn_transform_float32_input",
        "manual_float64_formula", "manual_float64_then_float32", "manual_float32_formula",
    }
    for name, values in paths.items():
        assert decomposition["paths"][name]["output_digest"]
        assert decomposition["paths"][name]["output_dtype"] == str(values.dtype)


def test_ulp_distance_and_largest_error_location_are_deterministic():
    baseline32 = np.zeros((2, 2, 2), dtype=np.float32)
    candidate32 = baseline32.copy()
    candidate32[1, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
    baseline64 = baseline32.astype(np.float64)
    candidate64 = baseline64.copy()
    arrays_a = {"transformed_float32": candidate32, "transformed_float64": candidate64}
    arrays_b = {"transformed_float32": baseline32, "transformed_float64": baseline64}
    policy = load_isolation_policy()
    result = compare_transform_arrays(
        arrays_a, arrays_b, mask=np.asarray([True, True]),
        source_bar_ids=["a", "b"], policy=policy,
    )
    assert maximum_ulp_distance(baseline32, candidate32) == 1
    assert result["maximum_ulp_distance"] == 1
    assert result["largest_error_source_bar_id"] == "b"
    assert result["largest_error_timestep"] == 0
    assert result["largest_error_feature_index"] == 1


def test_model_output_levels_keep_bitwise_numeric_and_behavior_separate():
    observed = {
        "raw_probability": np.asarray([0.4, 0.6]),
        "after_bias_probability": np.asarray([0.4, 0.6]),
        "after_temperature_probability": np.asarray([0.4, 0.6]),
        "ret_hat": np.asarray([0.1, 0.2]), "rv_hat": np.asarray([0.3, 0.4]),
    }
    candidate = {name: values.copy() for name, values in observed.items()}
    candidate["raw_probability"][0] += 2e-8
    states = [{"flat": False, "event_types": [], "excluded": False}] * 2
    result = compare_model_records(
        observed, candidate, observed_states=states, candidate_states=states,
        policy=load_isolation_policy(), deterministic=True,
    )
    assert result["bitwise_output_equal"] is False
    assert result["numerical_output_within_tolerance"] is False
    assert result["changed_raw_direction_count"] == 0


@pytest.mark.parametrize("change", ["direction", "exclusion", "flat"])
def test_each_behavioral_change_is_counted(change):
    observed = {
        "raw_probability": np.asarray([0.49]),
        "after_bias_probability": np.asarray([0.49]),
        "after_temperature_probability": np.asarray([0.49]),
        "ret_hat": np.asarray([0.0]), "rv_hat": np.asarray([0.0]),
    }
    candidate = {name: values.copy() for name, values in observed.items()}
    left = [{"flat": False, "event_types": [], "excluded": False}]
    right = [{"flat": change == "flat", "event_types": [], "excluded": False}]
    if change == "direction":
        candidate["raw_probability"][0] = 0.51
        candidate["after_temperature_probability"][0] = 0.51
    if change == "exclusion":
        right[0] = {"flat": False, "event_types": ["extreme"], "excluded": True}
    result = compare_model_records(
        observed, candidate, observed_states=left, candidate_states=right,
        policy=load_isolation_policy(), deterministic=True,
    )
    assert sum(result[field] for field in (
        "changed_raw_direction_count", "changed_calibrated_direction_count",
        "changed_exclusion_event_count", "changed_excluded_endpoint_count",
        "changed_flat_window_count",
    )) > 0


def test_windows_digest_is_content_deterministic(tmp_path):
    arrays = {
        "windows": np.ones((1, 2, 2), dtype=np.float32),
        "symbols": np.asarray(["BTCUSDT"]), "source_bar_ids": np.asarray(["a"]),
        "source_bar_open_utc": np.asarray(["o"]), "source_bar_close_utc": np.asarray(["c"]),
        "feature_window_digests": np.asarray(["f" * 64]),
    }
    digest = windows_payload_digest(arrays)
    arrays["input_windows_digest"] = np.asarray(digest)
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()
    assert json_digest({"digest": digest}) == json_digest({"digest": digest})
