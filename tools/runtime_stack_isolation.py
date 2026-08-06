"""Offline one-dependency-at-a-time numerical runtime isolation.

The isolated interpreters execute only the Phase 23 scaler worker.  All model
inference is performed here with the unchanged main-environment CPU PyTorch
runtime and the same incumbent state dict loaded once per model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_alignment_shadow import evaluate_ensemble_variants
from tools.model_runtime_repro import (
    DEFAULT_BUNDLE,
    MODEL_KINDS,
    _load_model_once,
    _output_rows,
    _safe_artifact_path,
    _worker_pair,
    apply_calibration,
    collapse_statistics,
    compare_scaled_arrays,
    ensure_safe_report_output,
    extract_immutable_windows,
    file_digest,
    forward_scaled_arrays,
    json_digest,
    load_retraining_policy,
    validate_bundle_contract,
    verify_report_digest,
    verify_snapshot_artifacts,
)
from tools.model_scaler_worker import maximum_ulp_distance, write_deterministic_npz


DEFAULT_MATRIX = BASE_DIR / "research" / "runtime_stack_matrix.json"
DEFAULT_POLICY = BASE_DIR / "research" / "runtime_stack_isolation_policy.json"
DEFAULT_PHASE22_REPORT = BASE_DIR / "reports" / "model_alignment_report_final.json"
DEFAULT_PHASE23_REPORT = BASE_DIR / "reports" / "model_runtime_reproducibility.json"
DEFAULT_WORK = BASE_DIR / "reports" / "runtime_stack_work"
DEFAULT_REPORT = BASE_DIR / "reports" / "runtime_stack_isolation.json"
ENVIRONMENT_ROOT = BASE_DIR / ".venv-runtime-isolation"
EXPECTED_BUNDLE_DIGEST = "43597484148a569c4827ff1f1378048264e9a46878ba846efccbb81ee9362843"
EXPECTED_ALIGNMENT_DIGEST = "5f167c7d41e24ca048dd1ea82c3d66ebfd32fed7ed176caddf084a18d7412d4f"
PACKAGE_KEYS = ("numpy", "scipy", "joblib", "scikit-learn", "threadpoolctl")
PRIMARY_STACKS = (
    "observed_main", "declared_sklearn_only", "sklearn_only_180",
    "numpy_only_233", "scipy_only_162", "joblib_only_152", "serialized_full_stack",
)
INTERACTION_STACKS = (
    "numpy_233_sklearn_180", "scipy_162_sklearn_180", "joblib_152_sklearn_180",
    "numpy_233_scipy_162", "numpy_233_joblib_152",
)
EXPECTED_VERSIONS = {
    "observed_main": ("2.3.2", "1.16.1", "1.5.1", "1.7.1", "3.6.0"),
    "declared_sklearn_only": ("2.3.2", "1.16.1", "1.5.1", "1.7.2", "3.6.0"),
    "sklearn_only_180": ("2.3.2", "1.16.1", "1.5.1", "1.8.0", "3.6.0"),
    "numpy_only_233": ("2.3.3", "1.16.1", "1.5.1", "1.7.1", "3.6.0"),
    "scipy_only_162": ("2.3.2", "1.16.2", "1.5.1", "1.7.1", "3.6.0"),
    "joblib_only_152": ("2.3.2", "1.16.1", "1.5.2", "1.7.1", "3.6.0"),
    "serialized_full_stack": ("2.3.3", "1.16.2", "1.5.2", "1.8.0", "3.6.0"),
    "numpy_233_sklearn_180": ("2.3.3", "1.16.1", "1.5.1", "1.8.0", "3.6.0"),
    "scipy_162_sklearn_180": ("2.3.2", "1.16.2", "1.5.1", "1.8.0", "3.6.0"),
    "joblib_152_sklearn_180": ("2.3.2", "1.16.1", "1.5.2", "1.8.0", "3.6.0"),
    "numpy_233_scipy_162": ("2.3.3", "1.16.2", "1.5.1", "1.7.1", "3.6.0"),
    "numpy_233_joblib_152": ("2.3.3", "1.16.1", "1.5.2", "1.7.1", "3.6.0"),
}
POLICY_TEMPLATE = {
    "schema_version": 1,
    "required_python_major_minor": "3.13",
    "required_timeframe": "5m",
    "required_sequence_length": 64,
    "required_feature_count": 27,
    "required_symbols": ["BTCUSDT", "ETHUSDT"],
    "required_windows_per_symbol": 120,
    "float64_max_abs_error": 1e-12,
    "float32_max_abs_error": 1e-7,
    "probability_max_abs_error": 1e-8,
    "regression_max_abs_error": 1e-7,
    "require_zero_direction_changes": True,
    "require_zero_exclusion_changes": True,
    "require_zero_allow_changes": True,
    "require_zero_agreement_changes": True,
    "require_zero_ensemble_decision_changes": True,
    "require_deterministic_workers": True,
    "require_deterministic_model_inference": True,
    "require_immutable_bundle": True,
    "allow_phase24_in_dedicated_canonical_environment": True,
    "allow_main_runtime_migration": False,
    "allow_live_activation": False,
}


class RuntimeStackIsolationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_isolation_policy(path: Path | str = DEFAULT_POLICY) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or set(value) != set(POLICY_TEMPLATE):
        raise RuntimeStackIsolationError("runtime isolation policy fields are not exact")
    for name, expected in POLICY_TEMPLATE.items():
        observed = value[name]
        if isinstance(expected, bool):
            valid_type = type(observed) is bool
        elif isinstance(expected, int):
            valid_type = type(observed) is int
        elif isinstance(expected, float):
            valid_type = type(observed) is float
        else:
            valid_type = type(observed) is type(expected)
        if not valid_type or observed != expected:
            raise RuntimeStackIsolationError(f"runtime isolation policy mismatch: {name}")
    return value


def load_stack_matrix(path: Path | str = DEFAULT_MATRIX) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "python_major_minor", "packages", "stacks"
    }:
        raise RuntimeStackIsolationError("runtime stack matrix fields are not exact")
    if value["schema_version"] != 1 or value["python_major_minor"] != "3.13":
        raise RuntimeStackIsolationError("runtime stack matrix version/Python contract mismatch")
    if value["packages"] != list(PACKAGE_KEYS) or set(value["stacks"]) != set(EXPECTED_VERSIONS):
        raise RuntimeStackIsolationError("runtime stack matrix inventory mismatch")
    for stack_id, versions in EXPECTED_VERSIONS.items():
        stack = value["stacks"][stack_id]
        if set(stack) != {"phase", "environment_required", "changed_packages", "package_versions"}:
            raise RuntimeStackIsolationError(f"stack fields are not exact: {stack_id}")
        expected_phase = "primary" if stack_id in PRIMARY_STACKS else "interaction"
        if stack["phase"] != expected_phase:
            raise RuntimeStackIsolationError(f"stack phase mismatch: {stack_id}")
        if type(stack["environment_required"]) is not bool:
            raise RuntimeStackIsolationError(f"environment_required type mismatch: {stack_id}")
        if stack["environment_required"] != (stack_id != "observed_main"):
            raise RuntimeStackIsolationError(f"environment requirement mismatch: {stack_id}")
        if not isinstance(stack["changed_packages"], list):
            raise RuntimeStackIsolationError(f"changed_packages type mismatch: {stack_id}")
        packages = stack["package_versions"]
        if set(packages) != set(PACKAGE_KEYS):
            raise RuntimeStackIsolationError(f"package fields mismatch: {stack_id}")
        if tuple(packages[name] for name in PACKAGE_KEYS) != versions:
            raise RuntimeStackIsolationError(f"package version mismatch: {stack_id}")
    value["matrix_digest"] = json_digest(value)
    return value


def stack_python_path(stack_id: str) -> Path:
    return ENVIRONMENT_ROOT / stack_id / "Scripts" / "python.exe"


def interpreter_inventory(python: Path | str) -> dict[str, Any]:
    code = r'''
import hashlib, importlib.metadata, json, platform, sys
r = {"python": platform.python_version()}
for key, module in (("numpy", "numpy"), ("scipy", "scipy"), ("joblib", "joblib"), ("scikit-learn", "sklearn"), ("threadpoolctl", "threadpoolctl")):
    try: r[key] = str(__import__(module).__version__)
    except Exception: r[key] = None
for key, module in (("torch_installed", "torch"), ("ccxt_installed", "ccxt"), ("hyperliquid_installed", "hyperliquid")):
    try: __import__(module); r[key] = True
    except Exception: r[key] = False
r["distributions"] = sorted({d.metadata["Name"].lower() for d in importlib.metadata.distributions() if d.metadata.get("Name")})
print(json.dumps(r, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code], cwd=BASE_DIR, capture_output=True, text=True, timeout=60
    )
    if completed.returncode:
        raise RuntimeStackIsolationError("unable to inventory isolated interpreter")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["interpreter_digest"] = file_digest(python) if Path(python).is_file() else None
    return result


def validate_stack_inventory(
    stack_id: str, inventory: Mapping[str, Any], matrix: Mapping[str, Any]
) -> None:
    expected = matrix["stacks"][stack_id]["package_versions"]
    if ".".join(str(inventory.get("python") or "").split(".")[:2]) != matrix["python_major_minor"]:
        raise RuntimeStackIsolationError(f"Python major/minor mismatch: {stack_id}")
    for package in PACKAGE_KEYS:
        if inventory.get(package) != expected[package]:
            raise RuntimeStackIsolationError(
                f"package mismatch for {stack_id}/{package}: {inventory.get(package)!r}"
            )
    # The unchanged project interpreter necessarily contains PyTorch and may
    # contain trading dependencies.  The prohibition applies to the small,
    # scaler-only isolation environments, not to observed_main.
    if stack_id != "observed_main" and any(inventory.get(name) is not False for name in (
        "torch_installed", "ccxt_installed", "hyperliquid_installed"
    )):
        raise RuntimeStackIsolationError(f"prohibited package present: {stack_id}")


def inventory_matrix(matrix: Mapping[str, Any], *, current_python: Path | str = sys.executable) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stack_id, definition in matrix["stacks"].items():
        python = Path(current_python) if stack_id == "observed_main" else stack_python_path(stack_id)
        if not python.is_file():
            result[stack_id] = {
                "status": "environment_unavailable",
                "package_versions": definition["package_versions"],
                "environment_path": None if stack_id == "observed_main" else f".venv-runtime-isolation/{stack_id}",
            }
            continue
        try:
            observed = interpreter_inventory(python)
            validate_stack_inventory(stack_id, observed, matrix)
            environment_manifest: dict[str, Any] | None = None
            if stack_id != "observed_main":
                marker = python.parents[1] / ".runtime-stack-manifest.json"
                if not marker.is_file():
                    raise RuntimeStackIsolationError(f"environment manifest missing: {stack_id}")
                environment_manifest = json.loads(marker.read_text(encoding="utf-8-sig"))
                recorded_digest = environment_manifest.get("manifest_digest")
                calculated_digest = json_digest({
                    key: value for key, value in environment_manifest.items()
                    if key != "manifest_digest"
                })
                if recorded_digest != calculated_digest:
                    raise RuntimeStackIsolationError(f"environment manifest digest mismatch: {stack_id}")
                if environment_manifest.get("stack_id") != stack_id:
                    raise RuntimeStackIsolationError(f"environment manifest stack mismatch: {stack_id}")
                if environment_manifest.get("matrix_digest") != matrix["matrix_digest"]:
                    raise RuntimeStackIsolationError(f"environment matrix contract changed: {stack_id}")
                if environment_manifest.get("package_versions") != definition["package_versions"]:
                    raise RuntimeStackIsolationError(f"environment manifest packages mismatch: {stack_id}")
            result[stack_id] = {
                "status": "available",
                "package_versions": {name: observed[name] for name in PACKAGE_KEYS},
                "python_version": observed["python"],
                "interpreter_digest": observed["interpreter_digest"],
                "torch_installed": observed["torch_installed"],
                "ccxt_installed": observed["ccxt_installed"],
                "hyperliquid_installed": observed["hyperliquid_installed"],
                "environment_path": None if stack_id == "observed_main" else f".venv-runtime-isolation/{stack_id}",
                "environment_manifest_digest": (
                    environment_manifest.get("manifest_digest") if environment_manifest else None
                ),
                "pip_freeze_digest": (
                    environment_manifest.get("pip_freeze_digest") if environment_manifest else None
                ),
            }
        except Exception as exc:
            result[stack_id] = {
                "status": "environment_unavailable",
                "reason": str(exc),
                "package_versions": definition["package_versions"],
                "environment_path": None if stack_id == "observed_main" else f".venv-runtime-isolation/{stack_id}",
            }
    return result


def environment_manifest_path(stack_id: str) -> Path:
    return ENVIRONMENT_ROOT / stack_id / ".runtime-stack-manifest.json"


def record_environment_manifest(
    stack_id: str, python: Path | str, matrix: Mapping[str, Any]
) -> dict[str, Any]:
    if stack_id == "observed_main" or stack_id not in matrix["stacks"]:
        raise RuntimeStackIsolationError("only a declared isolated stack may be recorded")
    expected_python = stack_python_path(stack_id).resolve()
    supplied_python = Path(python).resolve()
    if supplied_python != expected_python or not supplied_python.is_file():
        raise RuntimeStackIsolationError("environment interpreter is outside its dedicated stack path")
    marker = environment_manifest_path(stack_id)
    if marker.exists():
        raise RuntimeStackIsolationError("environment manifest already exists; validate reuse instead")
    observed = interpreter_inventory(supplied_python)
    validate_stack_inventory(stack_id, observed, matrix)
    completed = subprocess.run(
        [str(supplied_python), "-m", "pip", "freeze"], cwd=BASE_DIR,
        capture_output=True, text=True, timeout=60,
    )
    if completed.returncode:
        raise RuntimeStackIsolationError("unable to inventory isolated pip freeze")
    normalized_freeze = "\n".join(sorted(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )) + "\n"
    result: dict[str, Any] = {
        "schema_version": 1,
        "stack_id": stack_id,
        "matrix_digest": matrix["matrix_digest"],
        "python_major_minor": matrix["python_major_minor"],
        "python_version": observed["python"],
        "package_versions": {
            name: observed[name] for name in PACKAGE_KEYS
        },
        "pip_freeze_digest": hashlib.sha256(normalized_freeze.encode("utf-8")).hexdigest(),
        "interpreter_digest": observed["interpreter_digest"],
        "torch_installed": observed["torch_installed"],
        "ccxt_installed": observed["ccxt_installed"],
        "hyperliquid_installed": observed["hyperliquid_installed"],
    }
    result["manifest_digest"] = json_digest(result)
    marker.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def validate_recorded_environment(stack_id: str, matrix: Mapping[str, Any]) -> dict[str, Any]:
    if stack_id == "observed_main" or stack_id not in matrix["stacks"]:
        raise RuntimeStackIsolationError("only a declared isolated stack may be validated")
    result = inventory_matrix(matrix).get(stack_id, {})
    if result.get("status") != "available":
        raise RuntimeStackIsolationError(str(result.get("reason") or "environment unavailable"))
    return result


def ensure_safe_working_dir(path: Path | str) -> Path:
    target = Path(path).resolve()
    try:
        relative = target.relative_to(BASE_DIR.resolve())
    except ValueError:
        return target
    if not relative.parts or relative.parts[0] != "reports" or not relative.parts[-1].startswith("runtime_stack_"):
        raise RuntimeStackIsolationError("working directory must be an ignored reports/runtime_stack_* path")
    return target


def _load_json_report(path: Path | str, digest_field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    verify_report_digest(value, digest_field)
    return value


def validate_immutable_inputs(
    *, bundle: Path | str, phase22_report: Path | str, phase23_report: Path | str,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    phase23_policy = load_retraining_policy()
    manifest, snapshot, records = validate_bundle_contract(bundle, phase23_policy)
    if manifest.get("bundle_digest") != EXPECTED_BUNDLE_DIGEST:
        raise RuntimeStackIsolationError("required Phase 22 bundle digest mismatch")
    phase22 = _load_json_report(phase22_report, "alignment_digest")
    if phase22.get("alignment_digest") != EXPECTED_ALIGNMENT_DIGEST:
        raise RuntimeStackIsolationError("required Phase 22 alignment digest mismatch")
    phase23 = _load_json_report(phase23_report, "reproducibility_digest")
    if phase22.get("historical_alignment", {}).get("bundle_digest") != EXPECTED_BUNDLE_DIGEST:
        raise RuntimeStackIsolationError("Phase 22 report bundle mismatch")
    if phase23.get("input_bundle", {}).get("bundle_digest") != EXPECTED_BUNDLE_DIGEST:
        raise RuntimeStackIsolationError("Phase 23 report bundle mismatch")
    for field, expected in (
        ("timeframe", policy["required_timeframe"]),
        ("sequence_length", policy["required_sequence_length"]),
        ("feature_count", policy["required_feature_count"]),
    ):
        if manifest.get(field) != expected:
            raise RuntimeStackIsolationError(f"immutable bundle {field} mismatch")
    arrays = extract_immutable_windows(bundle, manifest, records)
    for symbol in policy["required_symbols"]:
        if int(np.count_nonzero(np.asarray(arrays["symbols"]) == symbol)) < policy["required_windows_per_symbol"]:
            raise RuntimeStackIsolationError(f"insufficient authenticated windows: {symbol}")
    verify_snapshot_artifacts(snapshot)
    phase23_scalers = phase23.get("scaler_comparisons", {})
    phase23_models = phase23.get("model_output_comparisons", {})
    for entry in snapshot.get("model_entries", []):
        kind = str(entry["kind"])
        if phase23_scalers.get(kind, {}).get("scaler_digest") != entry.get("scaler_sha256"):
            raise RuntimeStackIsolationError(f"Phase 23 scaler digest mismatch: {kind}")
        if phase23_models.get(kind, {}).get("model_digest") != entry.get("model_sha256"):
            raise RuntimeStackIsolationError(f"Phase 23 model digest mismatch: {kind}")
    return manifest, snapshot, records, arrays, phase22, phase23


def map_transform_classification(result: Mapping[str, Any]) -> str:
    return {
        "exact_match": "bitwise_exact",
        "numerically_equivalent": "numerically_equivalent",
        "materially_different": "numerically_material",
    }.get(str(result.get("classification")), "comparison_failed")


def compare_transform_arrays(
    candidate: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
    *, mask: np.ndarray, source_bar_ids: Sequence[str], policy: Mapping[str, Any],
) -> dict[str, Any]:
    base = compare_scaled_arrays(
        np.asarray(reference["transformed_float64"])[mask],
        np.asarray(candidate["transformed_float64"])[mask],
        np.asarray(reference["transformed_float32"])[mask],
        np.asarray(candidate["transformed_float32"])[mask],
        source_bar_ids=source_bar_ids,
        float64_tolerance=policy["float64_max_abs_error"],
        float32_tolerance=policy["float32_max_abs_error"],
    )
    base["maximum_ulp_distance"] = maximum_ulp_distance(
        np.asarray(reference["transformed_float32"])[mask],
        np.asarray(candidate["transformed_float32"])[mask],
    )
    base["classification"] = map_transform_classification(base)
    base["comparison_digest"] = json_digest({
        key: value for key, value in base.items() if key != "comparison_digest"
    })
    return base


def run_model_output(model: Any, scaled: np.ndarray, *, bias: float, temperature: float) -> tuple[dict[str, np.ndarray], bool]:
    first = forward_scaled_arrays(model, scaled)
    repeat = forward_scaled_arrays(model, scaled)
    deterministic = all(np.array_equal(first[name], repeat[name]) for name in first)
    first["after_bias_probability"], first["after_temperature_probability"] = apply_calibration(
        first["raw_probability"], bias, temperature
    )
    return first, deterministic


def compare_model_records(
    observed: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray],
    *, observed_states: Sequence[Mapping[str, Any]], candidate_states: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any], deterministic: bool,
) -> dict[str, Any]:
    raw_error = np.abs(np.asarray(observed["raw_probability"]) - np.asarray(candidate["raw_probability"]))
    calibrated_error = np.abs(
        np.asarray(observed["after_temperature_probability"])
        - np.asarray(candidate["after_temperature_probability"])
    )
    ret_error = np.abs(np.asarray(observed["ret_hat"]) - np.asarray(candidate["ret_hat"]))
    rv_error = np.abs(np.asarray(observed["rv_hat"]) - np.asarray(candidate["rv_hat"]))
    observed_raw = np.asarray(observed["raw_probability"])
    candidate_raw = np.asarray(candidate["raw_probability"])
    observed_cal = np.asarray(observed["after_temperature_probability"])
    candidate_cal = np.asarray(candidate["after_temperature_probability"])
    raw_direction = int(np.count_nonzero((observed_raw > 0.5) != (candidate_raw > 0.5)))
    cal_direction = int(np.count_nonzero((observed_cal > 0.5) != (candidate_cal > 0.5)))
    extreme = int(np.count_nonzero(
        ((observed_cal < 0.05) | (observed_cal > 0.95))
        != ((candidate_cal < 0.05) | (candidate_cal > 0.95))
    ))
    flat = sum(bool(a.get("flat")) != bool(b.get("flat")) for a, b in zip(observed_states, candidate_states))
    events = sum(
        set(a.get("event_types", [])) != set(b.get("event_types", []))
        for a, b in zip(observed_states, candidate_states)
    )
    endpoints = sum(
        bool(a.get("excluded")) != bool(b.get("excluded"))
        for a, b in zip(observed_states, candidate_states)
    )
    exact = all(np.array_equal(np.asarray(observed[name]), np.asarray(candidate[name])) for name in (
        "raw_probability", "after_bias_probability", "after_temperature_probability", "ret_hat", "rv_hat"
    ))
    numerical = (
        (float(np.max(raw_error)) if raw_error.size else 0.0) <= policy["probability_max_abs_error"]
        and (float(np.max(calibrated_error)) if calibrated_error.size else 0.0) <= policy["probability_max_abs_error"]
        and (float(np.max(ret_error)) if ret_error.size else 0.0) <= policy["regression_max_abs_error"]
        and (float(np.max(rv_error)) if rv_error.size else 0.0) <= policy["regression_max_abs_error"]
    )
    result = {
        "raw_probability_max_abs_error": float(np.max(raw_error)) if raw_error.size else 0.0,
        "raw_probability_mean_abs_error": float(np.mean(raw_error)) if raw_error.size else 0.0,
        "calibrated_probability_max_abs_error": float(np.max(calibrated_error)) if calibrated_error.size else 0.0,
        "ret_hat_max_abs_error": float(np.max(ret_error)) if ret_error.size else 0.0,
        "rv_hat_max_abs_error": float(np.max(rv_error)) if rv_error.size else 0.0,
        "changed_raw_direction_count": raw_direction,
        "changed_calibrated_direction_count": cal_direction,
        "changed_extreme_state_count": extreme,
        "changed_flat_window_count": int(flat),
        "changed_exclusion_event_count": int(events),
        "changed_excluded_endpoint_count": int(endpoints),
        "changed_allow_count": 0,
        "changed_signal_direction_count": 0,
        "changed_agreement_suppression_count": 0,
        "changed_ensemble_variant_count": 0,
        "deterministic_repeat_status": "deterministic" if deterministic else "nondeterministic",
        "bitwise_output_equal": exact,
        "numerical_output_within_tolerance": numerical,
        "classification": (
            "bitwise_exact" if exact and deterministic else
            "numerically_equivalent" if numerical and deterministic else
            "numerically_material" if deterministic else "comparison_failed"
        ),
    }
    result["comparison_digest"] = json_digest(result)
    return result


def ensemble_variant_changes(
    observed_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    left, _ = evaluate_ensemble_variants(observed_rows, snapshot)
    right, _ = evaluate_ensemble_variants(candidate_rows, snapshot)
    a = {(row["symbol"], row["source_bar_id"], row["variant"]): row for row in left}
    b = {(row["symbol"], row["source_bar_id"], row["variant"]): row for row in right}
    if set(a) != set(b):
        raise RuntimeStackIsolationError("ensemble endpoint inventory mismatch")
    result: dict[str, dict[str, int]] = defaultdict(lambda: {
        "changed_allow_count": 0,
        "changed_signal_direction_count": 0,
        "changed_agreement_suppression_count": 0,
        "changed_ensemble_variant_count": 0,
    })
    for key, current in a.items():
        other = b[key]
        changed_allow = current["allow"] != other["allow"]
        changed_direction = current["direction"] != other["direction"]
        changed_agreement = current["agreement_suppressed"] != other["agreement_suppressed"]
        symbol = key[0]
        result[symbol]["changed_allow_count"] += int(changed_allow)
        result[symbol]["changed_signal_direction_count"] += int(changed_direction)
        result[symbol]["changed_agreement_suppression_count"] += int(changed_agreement)
        result[symbol]["changed_ensemble_variant_count"] += int(
            changed_allow or changed_direction or changed_agreement
        )
    return dict(result)


def _aggregate_transform_status(comparisons: Mapping[str, Any], field: str) -> bool:
    return all(
        result.get("classification") == field
        for model in comparisons.values()
        for result in model.get("by_symbol", {}).values()
    )


def _stack_matches_serialized(
    candidate: Mapping[str, Mapping[str, np.ndarray]],
    serialized: Mapping[str, Mapping[str, np.ndarray]],
) -> bool:
    if set(candidate) != set(serialized):
        return False
    return all(
        np.array_equal(candidate[kind][name], serialized[kind][name])
        for kind in candidate
        for name in ("transformed_float64", "transformed_float32")
    )


def build_isolation_report(
    *, bundle: Path | str, matrix: Mapping[str, Any], policy: Mapping[str, Any],
    current_python: Path | str = sys.executable, working_dir: Path | str = DEFAULT_WORK,
    phase22_report: Path | str = DEFAULT_PHASE22_REPORT,
    phase23_report: Path | str = DEFAULT_PHASE23_REPORT,
    selected_stacks: Sequence[str] | None = None,
) -> dict[str, Any]:
    if Path(current_python).resolve() != Path(sys.executable).resolve():
        raise RuntimeStackIsolationError("current Python must be the executing main interpreter")
    manifest, snapshot, records, windows, phase22, phase23 = validate_immutable_inputs(
        bundle=bundle, phase22_report=phase22_report, phase23_report=phase23_report, policy=policy
    )
    work = ensure_safe_working_dir(working_dir)
    work.mkdir(parents=True, exist_ok=True)
    windows_npz = work / "immutable-windows.npz"
    write_deterministic_npz(windows_npz, windows)
    environment_matrix = inventory_matrix(matrix, current_python=current_python)
    requested = list(dict.fromkeys(
        (["observed_main", "serialized_full_stack", *(selected_stacks or [])])
        if selected_stacks else PRIMARY_STACKS
    ))
    unknown = sorted(set(requested) - set(matrix["stacks"]))
    if unknown:
        raise RuntimeStackIsolationError(f"unknown stack ids: {unknown}")
    entries = {str(entry["kind"]): entry for entry in snapshot["model_entries"]}
    stack_arrays: dict[str, dict[str, Mapping[str, np.ndarray]]] = {}
    stack_manifests: dict[str, dict[str, Mapping[str, Any]]] = {}
    comparison_errors: dict[str, str] = {}
    single_matches: list[str] = []

    def evaluate_stack(stack_id: str) -> None:
        if stack_id in stack_arrays or environment_matrix[stack_id]["status"] != "available":
            return
        python = Path(current_python) if stack_id == "observed_main" else stack_python_path(stack_id)
        stack_arrays[stack_id], stack_manifests[stack_id] = {}, {}
        try:
            for kind in MODEL_KINDS:
                entry = entries[kind]
                scaler_path = _safe_artifact_path(BASE_DIR, entry["scaler_filename"])
                worker_manifest, output, deterministic = _worker_pair(
                    python, label=stack_id, kind=kind, scaler=scaler_path,
                    windows_npz=windows_npz, work=work,
                )
                observed_versions = worker_manifest["runtime_versions"]
                expected_versions = matrix["stacks"][stack_id]["package_versions"]
                for package in PACKAGE_KEYS:
                    worker_key = "scikit_learn" if package == "scikit-learn" else package
                    if observed_versions.get(worker_key) != expected_versions[package]:
                        raise RuntimeStackIsolationError(f"worker package mismatch: {stack_id}/{package}")
                if not deterministic:
                    raise RuntimeStackIsolationError(f"nondeterministic worker: {stack_id}/{kind}")
                stack_arrays[stack_id][kind] = output
                stack_manifests[stack_id][kind] = worker_manifest
            environment_matrix[stack_id]["comparison_status"] = "comparison_completed"
        except Exception as exc:
            stack_arrays.pop(stack_id, None)
            stack_manifests.pop(stack_id, None)
            comparison_errors[stack_id] = f"{type(exc).__name__}: {exc}"
            environment_matrix[stack_id]["comparison_status"] = "comparison_failed"

    for stack_id in requested:
        evaluate_stack(stack_id)
    if "observed_main" not in stack_arrays or "serialized_full_stack" not in stack_arrays:
        required_interactions: list[str] = []
    else:
        single_matches = [
            stack_id for stack_id in PRIMARY_STACKS
            if stack_id not in {"observed_main", "serialized_full_stack"}
            and len(matrix["stacks"][stack_id]["changed_packages"]) == 1
            and stack_id in stack_arrays
            and _stack_matches_serialized(stack_arrays[stack_id], stack_arrays["serialized_full_stack"])
        ]
        required_interactions = [] if single_matches else list(INTERACTION_STACKS)
    if selected_stacks is None:
        for stack_id in required_interactions:
            requested.append(stack_id)
            if environment_matrix[stack_id]["status"] == "available":
                evaluate_stack(stack_id)

    transform_decomposition: dict[str, Any] = {}
    for stack_id, manifests in stack_manifests.items():
        transform_decomposition[stack_id] = {
            kind: {
                "scaler_metadata": manifest["scaler_metadata"],
                "worker_result_digest": manifest["worker_result_digest"],
                **manifest["transform_decomposition"],
            }
            for kind, manifest in manifests.items()
        }

    stack_comparisons: dict[str, Any] = {}
    observed = stack_arrays.get("observed_main")
    serialized = stack_arrays.get("serialized_full_stack")
    for stack_id in requested:
        if stack_id not in stack_arrays:
            stack_comparisons[stack_id] = {
                "classification": (
                    "comparison_failed" if stack_id in comparison_errors else "environment_unavailable"
                ),
                "reason": comparison_errors.get(stack_id) or environment_matrix[stack_id].get("reason"),
                "by_model": {},
            }
            continue
        by_model: dict[str, Any] = {}
        for kind in MODEL_KINDS:
            by_model[kind] = {"by_symbol": {}}
            for symbol in policy["required_symbols"]:
                mask = np.asarray(windows["symbols"]) == symbol
                ids = np.asarray(windows["source_bar_ids"])[mask].tolist()
                worker_manifest = stack_manifests[stack_id][kind]
                item: dict[str, Any] = {
                    "deserialization_warning_status": (
                        "warnings_captured" if worker_manifest.get("warning_categories") else "no_warnings"
                    ),
                    "deterministic_repeat_status": "deterministic",
                }
                if observed is not None:
                    item["vs_observed_main"] = compare_transform_arrays(
                        stack_arrays[stack_id][kind], observed[kind], mask=mask,
                        source_bar_ids=ids, policy=policy,
                    )
                if serialized is not None:
                    item["vs_serialized_full_stack"] = compare_transform_arrays(
                        stack_arrays[stack_id][kind], serialized[kind], mask=mask,
                        source_bar_ids=ids, policy=policy,
                    )
                by_model[kind]["by_symbol"][symbol] = item
        comparisons = {
            kind: {"by_symbol": {
                symbol: data["vs_observed_main"]
                for symbol, data in model["by_symbol"].items()
                if "vs_observed_main" in data
            }} for kind, model in by_model.items()
        }
        all_vs_observed = [
            item["vs_observed_main"]
            for model in by_model.values() for item in model["by_symbol"].values()
            if "vs_observed_main" in item
        ]
        classification = (
            "bitwise_exact" if all_vs_observed and all(x["classification"] == "bitwise_exact" for x in all_vs_observed)
            else "numerically_equivalent" if all_vs_observed and all(x["classification"] != "numerically_material" for x in all_vs_observed)
            else "numerically_material"
        )
        stack_comparisons[stack_id] = {
            "classification": classification,
            "matches_serialized_full_pattern": bool(
                serialized is not None and _stack_matches_serialized(stack_arrays[stack_id], serialized)
            ),
            "by_model": by_model,
        }
        stack_comparisons[stack_id]["comparison_digest"] = json_digest(stack_comparisons[stack_id])

    model_outputs: dict[str, dict[str, Any]] = {}
    output_values: dict[str, dict[str, dict[str, Mapping[str, np.ndarray]]]] = defaultdict(dict)
    output_states: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(dict)
    output_determinism: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    rows_by_stack_kind: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    health_policy = load_retraining_policy()
    for kind in MODEL_KINDS:
        model = _load_model_once(entries[kind])
        bias = float(snapshot.get(f"dl_bias_{kind}", 0.0) or 0.0)
        temperature = float(snapshot.get(f"dl_temp_{kind}", 1.0) or 1.0)
        for stack_id in requested:
            if stack_id not in stack_arrays:
                continue
            output_values[stack_id].setdefault(kind, {})
            output_states[stack_id].setdefault(kind, {})
            output_determinism[stack_id].setdefault(kind, {})
            for symbol in policy["required_symbols"]:
                mask = np.asarray(windows["symbols"]) == symbol
                ids = np.asarray(windows["source_bar_ids"])[mask].tolist()
                digests = np.asarray(windows["feature_window_digests"])[mask].tolist()
                scaled = np.asarray(stack_arrays[stack_id][kind]["transformed_float32"])[mask]
                output, deterministic = run_model_output(
                    model, scaled, bias=bias, temperature=temperature
                )
                _, states = collapse_statistics(
                    output["after_temperature_probability"], ids,
                    policy=health_policy, deterministic=deterministic,
                )
                output_values[stack_id][kind][symbol] = output
                output_states[stack_id][kind][symbol] = states
                output_determinism[stack_id][kind][symbol] = deterministic
                rows_by_stack_kind[stack_id][kind].extend(
                    _output_rows(kind, symbol, ids, digests, output, states)
                )

    observed_rows = [
        row for rows in rows_by_stack_kind.get("observed_main", {}).values() for row in rows
    ]
    for stack_id in requested:
        if stack_id not in stack_arrays or "observed_main" not in output_values:
            model_outputs[stack_id] = {
                "classification": stack_comparisons[stack_id]["classification"], "by_model": {}
            }
            continue
        by_model: dict[str, Any] = {}
        for kind in MODEL_KINDS:
            by_model[kind] = {"model_digest": entries[kind]["model_sha256"], "by_symbol": {}}
            counterfactual = [
                row for other_kind, rows in rows_by_stack_kind["observed_main"].items()
                if other_kind != kind for row in rows
            ] + list(rows_by_stack_kind[stack_id][kind])
            ensemble_changes = ensemble_variant_changes(observed_rows, counterfactual, snapshot)
            for symbol in policy["required_symbols"]:
                candidate = output_values[stack_id][kind][symbol]
                observed_output = output_values["observed_main"][kind][symbol]
                deterministic = (
                    output_determinism[stack_id][kind][symbol]
                    and output_determinism["observed_main"][kind][symbol]
                )
                result = compare_model_records(
                    observed_output, candidate,
                    observed_states=output_states["observed_main"][kind][symbol],
                    candidate_states=output_states[stack_id][kind][symbol],
                    policy=policy, deterministic=deterministic,
                )
                for field, count in ensemble_changes.get(symbol, {}).items():
                    result[field] = count
                if any(result[name] for name in (
                    "changed_allow_count", "changed_signal_direction_count",
                    "changed_agreement_suppression_count", "changed_ensemble_variant_count",
                )):
                    result["behavioral_difference"] = True
                else:
                    result["behavioral_difference"] = any(result[name] for name in (
                        "changed_raw_direction_count", "changed_calibrated_direction_count",
                        "changed_extreme_state_count", "changed_flat_window_count",
                        "changed_exclusion_event_count", "changed_excluded_endpoint_count",
                    ))
                result["comparison_digest"] = json_digest({
                    key: value for key, value in result.items() if key != "comparison_digest"
                })
                by_model[kind]["by_symbol"][symbol] = result
        all_candidate_rows = [
            row for rows in rows_by_stack_kind[stack_id].values() for row in rows
        ]
        aggregate_ensemble = ensemble_variant_changes(observed_rows, all_candidate_rows, snapshot)
        model_outputs[stack_id] = {
            "same_main_pytorch_runtime": True,
            "models_loaded_once_per_comparison_process": True,
            "by_model": by_model,
            "aggregate_ensemble_changes_by_symbol": aggregate_ensemble,
        }
        model_outputs[stack_id]["comparison_digest"] = json_digest(model_outputs[stack_id])

    reproducibility_levels: dict[str, Any] = {}
    for stack_id in requested:
        if stack_id not in stack_arrays or not model_outputs.get(stack_id, {}).get("by_model"):
            reproducibility_levels[stack_id] = {
                "bitwise_status": "unverified", "numerical_status": "unverified",
                "behavioral_status": "unverified",
            }
            continue
        transform_results = [
            item["vs_observed_main"]
            for model in stack_comparisons[stack_id]["by_model"].values()
            for item in model["by_symbol"].values()
        ]
        output_results = [
            item for model in model_outputs[stack_id]["by_model"].values()
            for item in model["by_symbol"].values()
        ]
        bitwise = (
            all(item["classification"] == "bitwise_exact" for item in transform_results)
            and all(item["bitwise_output_equal"] for item in output_results)
        )
        numerical = (
            all(item["classification"] != "numerically_material" for item in transform_results)
            and all(item["numerical_output_within_tolerance"] for item in output_results)
        )
        behavioral = not any(item["behavioral_difference"] for item in output_results)
        level = {
            "bitwise_status": "bitwise_reproducible" if bitwise else "not_bitwise_reproducible",
            "numerical_status": "numerically_reproducible" if numerical else "numerically_material_difference",
            "behavioral_status": "behaviorally_reproducible" if behavioral else "material_behavior_difference",
            "deterministic_workers": True,
            "deterministic_model_inference": all(
                item["deterministic_repeat_status"] == "deterministic" for item in output_results
            ),
        }
        level["evidence_digest"] = json_digest(level)
        reproducibility_levels[stack_id] = level

    available_required_interactions = [s for s in required_interactions if s in stack_arrays]
    interaction_matches = [
        s for s in available_required_interactions
        if serialized is not None and _stack_matches_serialized(stack_arrays[s], serialized)
    ]
    primary_unavailable = [s for s in PRIMARY_STACKS if s not in stack_arrays]
    behavior_difference = any(
        item.get("behavioral_status") == "material_behavior_difference"
        for item in reproducibility_levels.values()
    )
    if behavior_difference:
        overall = "runtime_stack_behavior_difference"
    elif primary_unavailable or any(s not in stack_arrays for s in required_interactions):
        overall = "runtime_stack_environment_pending"
    elif single_matches:
        overall = "runtime_stack_attributed"
    elif interaction_matches:
        overall = "runtime_stack_multi_package_interaction"
    else:
        overall = "runtime_stack_attribution_unresolved"
    verify_snapshot_artifacts(snapshot)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy": {
            "offline_only": True, "orders_allowed": False,
            "main_environment_modified": False, "models_modified": False,
        },
        "input_bundle": {
            "bundle_digest": manifest["bundle_digest"],
            "alignment_digest": phase22["alignment_digest"],
            "phase23_reproducibility_digest": phase23["reproducibility_digest"],
            "timeframe": manifest["timeframe"],
            "sequence_length": manifest["sequence_length"],
            "feature_width": manifest["feature_count"],
            "authenticated_windows_by_symbol": {
                symbol: int(np.count_nonzero(np.asarray(windows["symbols"]) == symbol))
                for symbol in policy["required_symbols"]
            },
            "input_windows_digest": str(np.asarray(windows["input_windows_digest"]).reshape(-1)[0]),
            "integrity_result": "pass",
            "feature_window_digest_result": "pass",
            "artifact_integrity_verified_before_and_after": True,
        },
        "environment_matrix": {
            "matrix_digest": matrix["matrix_digest"],
            "stacks": environment_matrix,
        },
        "transform_decomposition": transform_decomposition,
        "stack_comparisons": stack_comparisons,
        "model_output_comparisons": model_outputs,
        "reproducibility_levels": reproducibility_levels,
        "overall_decision": {
            "result": overall,
            "primary_single_package_matches": single_matches,
            "interaction_stacks_required": required_interactions,
            "interaction_stack_matches": interaction_matches,
            "full_primary_scope": not primary_unavailable,
            "behavioral_difference_detected": behavior_difference,
        },
        "warnings": [
            "Numerical and behavioral reproducibility are reported independently; zero decisions do not relax numeric tolerances."
        ] + ([
            "Interaction environments are required but unavailable; bootstrap only the listed stacks before final attribution."
        ] if any(s not in stack_arrays for s in required_interactions) else []),
    }
    report["isolation_digest"] = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", "isolation_digest"}
    })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23.1 offline runtime stack isolation")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--working-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--phase22-report", default=str(DEFAULT_PHASE22_REPORT))
    parser.add_argument("--phase23-report", default=str(DEFAULT_PHASE23_REPORT))
    parser.add_argument("--current-python", default=sys.executable)
    parser.add_argument("--stack", action="append", choices=tuple(EXPECTED_VERSIONS))
    parser.add_argument("--record-environment", choices=tuple(EXPECTED_VERSIONS))
    parser.add_argument("--validate-environment", choices=tuple(EXPECTED_VERSIONS))
    parser.add_argument("--environment-python")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        matrix = load_stack_matrix(args.matrix)
        policy = load_isolation_policy(args.policy)
        if args.record_environment:
            if not args.environment_python:
                raise RuntimeStackIsolationError("--environment-python is required when recording")
            print(json.dumps(record_environment_manifest(
                args.record_environment, args.environment_python, matrix
            ), indent=2))
            return 0
        if args.validate_environment:
            print(json.dumps(validate_recorded_environment(
                args.validate_environment, matrix
            ), indent=2))
            return 0
        if args.inventory_only:
            print(json.dumps({
                "schema_version": 1,
                "main_python_version": platform.python_version(),
                "matrix_digest": matrix["matrix_digest"],
                "environment_root": ".venv-runtime-isolation",
                "stacks": inventory_matrix(matrix, current_python=args.current_python),
            }, indent=2))
            return 0
        report = build_isolation_report(
            bundle=args.bundle, matrix=matrix, policy=policy,
            current_python=args.current_python, working_dir=args.working_dir,
            phase22_report=args.phase22_report, phase23_report=args.phase23_report,
            selected_stacks=args.stack,
        )
        target = ensure_safe_report_output(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and report["overall_decision"]["result"] not in {
            "runtime_stack_attributed", "runtime_stack_multi_package_interaction"
        }:
            return 3
        return 0
    except Exception as exc:
        print(f"runtime_stack_isolation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
