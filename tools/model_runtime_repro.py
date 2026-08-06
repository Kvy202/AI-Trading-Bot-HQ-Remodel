"""Phase 23 cross-runtime scaler and model-output reproducibility analysis.

Only immutable Phase 22 windows and snapshotted incumbent artifacts are read.
The reproduction interpreter is used solely to execute model_scaler_worker.py;
all PyTorch inference remains in this process on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features import canonical_feature_columns
from ml_dl.dl_ensemble import canonical_utc, feature_window_digest
from tools.model_alignment_shadow import (
    _read_feature_matrices,
    _window_from_record,
    calibrate_probability,
    evaluate_ensemble_variants,
    load_historical_bundle,
)
from tools.model_scaler_worker import (
    logical_array_digest,
    read_npz,
    windows_payload_digest,
    write_deterministic_npz,
)


DEFAULT_BUNDLE = BASE_DIR / "reports" / "model_alignment_bundles" / "history_5m_final"
DEFAULT_POLICY = BASE_DIR / "research" / "model_retraining_policy.json"
DEFAULT_REQUIREMENTS = BASE_DIR / "requirements" / "model_repro_sklearn180.txt"
DEFAULT_REPORT = BASE_DIR / "reports" / "model_runtime_reproducibility.json"
DEFAULT_REPRO_PYTHON = BASE_DIR / ".venv-repro-sklearn180" / "Scripts" / "python.exe"
MODEL_KINDS = ("adv", "lstm", "tcn", "tx")
REPRO_PACKAGE_CONTRACT = {
    "numpy": "2.3.3",
    "scipy": "1.16.2",
    "joblib": "1.5.2",
    "scikit_learn": "1.8.0",
    "threadpoolctl": "3.6.0",
}
POLICY_TYPES: dict[str, type] = {
    "schema_version": int,
    "required_timeframe": str,
    "required_sequence_length": int,
    "required_feature_count": int,
    "required_serving_symbols": list,
    "minimum_aligned_unique_bars_per_symbol": int,
    "float64_scaler_max_abs_error": float,
    "float32_scaler_max_abs_error": float,
    "probability_max_abs_error": float,
    "regression_output_max_abs_error": float,
    "flat_output_std_threshold": float,
    "extreme_low_threshold": float,
    "extreme_high_threshold": float,
    "extreme_consecutive_limit": int,
    "flat_window": int,
    "maximum_missing_rate": float,
    "minimum_candidate_validation_auc": float,
    "require_time_ordered_split": bool,
    "require_purged_split": bool,
    "require_scaler_fit_train_only": bool,
    "require_dataset_digest": bool,
    "require_feature_digest": bool,
    "require_label_digest": bool,
    "require_versioned_candidate_artifacts": bool,
    "allow_incumbent_artifact_overwrite": bool,
}


class RuntimeReproError(ValueError):
    """Raised for an unsafe, incomplete, or non-reproducible comparison."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def verify_report_digest(report: Mapping[str, Any], digest_field: str) -> None:
    if not isinstance(report, Mapping):
        raise RuntimeReproError("report must be a JSON object")
    observed = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", digest_field}
    })
    if report.get(digest_field) != observed:
        raise RuntimeReproError(f"{digest_field} mismatch")


def file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_safe_report_output(path: Path | str) -> Path:
    """Allow report writes only outside the repository or at reports/<file>.json."""

    target = Path(path).resolve()
    try:
        relative = target.relative_to(BASE_DIR.resolve())
    except ValueError:
        return target
    if len(relative.parts) != 2 or relative.parts[0].lower() != "reports" or target.suffix.lower() != ".json":
        raise RuntimeReproError("generated report paths inside the repository must be reports/<name>.json")
    return target


def ensure_safe_working_directory(path: Path | str) -> Path:
    target = Path(path).resolve()
    try:
        relative = target.relative_to(BASE_DIR.resolve())
    except ValueError:
        return target
    if (
        len(relative.parts) != 2
        or relative.parts[0].lower() != "reports"
        or not relative.parts[1].startswith("model_runtime_reproducibility_")
    ):
        raise RuntimeReproError(
            "in-repository comparison working directories must be ignored reports/model_runtime_reproducibility_* paths"
        )
    return target


def load_retraining_policy(path: Path | str = DEFAULT_POLICY) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeReproError(f"unable to load retraining policy: {type(exc).__name__}") from exc
    if not isinstance(value, dict) or set(value) != set(POLICY_TYPES):
        raise RuntimeReproError("retraining policy fields are not exact")
    for name, expected in POLICY_TYPES.items():
        item = value[name]
        if expected is float:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise RuntimeReproError(f"policy field {name} must be numeric")
            value[name] = float(item)
        elif type(item) is not expected:
            raise RuntimeReproError(f"policy field {name} must be {expected.__name__}")
    fixed = {
        "schema_version": 1,
        "required_timeframe": "5m",
        "required_sequence_length": 64,
        "required_feature_count": 27,
        "required_serving_symbols": ["BTCUSDT", "ETHUSDT"],
    }
    for name, expected in fixed.items():
        if value[name] != expected:
            raise RuntimeReproError(f"policy field {name} may not weaken the Phase 23 contract")
    if value["minimum_aligned_unique_bars_per_symbol"] < 100:
        raise RuntimeReproError("minimum aligned bars may not be less than 100")
    if value["float64_scaler_max_abs_error"] > 1e-12:
        raise RuntimeReproError("float64 scaler tolerance may not be weakened")
    if value["float32_scaler_max_abs_error"] > 1e-7:
        raise RuntimeReproError("float32 scaler tolerance may not be weakened")
    if value["probability_max_abs_error"] > 1e-8:
        raise RuntimeReproError("probability tolerance may not be weakened")
    if value["regression_output_max_abs_error"] > 1e-7:
        raise RuntimeReproError("regression tolerance may not be weakened")
    if value["flat_output_std_threshold"] < 0.002:
        raise RuntimeReproError("flat-output threshold may not be weakened")
    if value["extreme_low_threshold"] < 0.05 or value["extreme_high_threshold"] > 0.95:
        raise RuntimeReproError("extreme-output thresholds may not be weakened")
    if value["extreme_consecutive_limit"] > 20 or value["flat_window"] > 30:
        raise RuntimeReproError("health lookback gates may not be weakened")
    if value["maximum_missing_rate"] > 0.05:
        raise RuntimeReproError("maximum missing rate may not be weakened")
    if value["minimum_candidate_validation_auc"] < 0.55:
        raise RuntimeReproError("candidate validation AUC gate may not be weakened")
    if value["allow_incumbent_artifact_overwrite"] is not False:
        raise RuntimeReproError("incumbent artifact overwrite must be prohibited")
    if not all(value[name] for name in (
        "require_time_ordered_split", "require_purged_split", "require_scaler_fit_train_only",
        "require_dataset_digest", "require_feature_digest", "require_label_digest",
        "require_versioned_candidate_artifacts",
    )):
        raise RuntimeReproError("required retraining gates may not be disabled")
    if not (0 < value["extreme_low_threshold"] < 0.5 < value["extreme_high_threshold"] < 1):
        raise RuntimeReproError("invalid extreme probability thresholds")
    if value["flat_window"] <= 0 or value["extreme_consecutive_limit"] <= 0:
        raise RuntimeReproError("health windows must be positive")
    if not (0 <= value["maximum_missing_rate"] <= 1):
        raise RuntimeReproError("maximum_missing_rate must be between zero and one")
    return value


def requirements_digest(path: Path | str = DEFAULT_REQUIREMENTS) -> str:
    return file_digest(path)


def _declared_sklearn_version(path: Path | str) -> str | None:
    pattern = re.compile(r"^\s*scikit-learn\s*==\s*([^\s#]+)", re.IGNORECASE)
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _interpreter_inventory(python: Path | str) -> dict[str, Any]:
    code = (
        "import json,platform\n"
        "r={'python':platform.python_version()}\n"
        "for k,m in [('numpy','numpy'),('scipy','scipy'),('joblib','joblib'),('scikit_learn','sklearn'),('threadpoolctl','threadpoolctl')]:\n"
        " try:\n  r[k]=str(__import__(m).__version__)\n"
        " except Exception:\n  r[k]=None\n"
        "try:\n import torch\n r['torch_installed']=True\n"
        "except Exception:\n r['torch_installed']=False\n"
        "print(json.dumps(r,sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", code], check=True, capture_output=True, text=True, timeout=30,
            cwd=BASE_DIR,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeReproError(f"unable to inventory Python interpreter: {type(exc).__name__}") from exc


def _pip_freeze_digest(python: Path | str) -> str | None:
    try:
        completed = subprocess.run(
            [str(python), "-m", "pip", "freeze", "--all"], check=True,
            capture_output=True, text=True, timeout=60, cwd=BASE_DIR,
        )
    except Exception:
        return None
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def collect_dependency_inventory(
    *,
    current_python: Path | str = sys.executable,
    repro_python: Path | str | None = None,
    snapshot: Mapping[str, Any] | None = None,
    requirements_file: Path | str = BASE_DIR / "requirements.txt",
    repro_requirements: Path | str = DEFAULT_REQUIREMENTS,
) -> dict[str, Any]:
    main = _interpreter_inventory(current_python)
    repro = None
    if repro_python is not None and Path(repro_python).is_file():
        repro = _interpreter_inventory(repro_python)
    serialized = sorted({
        str(entry.get("scaler_serialized_sklearn_version"))
        for entry in (snapshot or {}).get("model_entries", [])
        if entry.get("scaler_serialized_sklearn_version")
    })
    return {
        "main_python_version": main.get("python"),
        "main_numpy_version": main.get("numpy"),
        "main_scipy_version": main.get("scipy"),
        "main_joblib_version": main.get("joblib"),
        "main_sklearn_version": main.get("scikit_learn"),
        "main_threadpoolctl_version": main.get("threadpoolctl"),
        "requirements_declared_sklearn_version": _declared_sklearn_version(requirements_file),
        "serialized_scaler_sklearn_versions": serialized,
        "repro_python_version": None if repro is None else repro.get("python"),
        "repro_numpy_version": None if repro is None else repro.get("numpy"),
        "repro_scipy_version": None if repro is None else repro.get("scipy"),
        "repro_joblib_version": None if repro is None else repro.get("joblib"),
        "repro_sklearn_version": None if repro is None else repro.get("scikit_learn"),
        "repro_threadpoolctl_version": None if repro is None else repro.get("threadpoolctl"),
        "repro_torch_installed": None if repro is None else bool(repro.get("torch_installed")),
        "repro_requirements_digest": requirements_digest(repro_requirements),
        "repro_pip_freeze_digest": None if repro is None else _pip_freeze_digest(repro_python),
        "version_roles": {
            "declared_runtime_version": _declared_sklearn_version(requirements_file),
            "observed_runtime_version": main.get("scikit_learn"),
            "serialized_artifact_versions": serialized,
            "comparison_runtime_version": None if repro is None else repro.get("scikit_learn"),
        },
    }


def validate_repro_environment(inventory: Mapping[str, Any]) -> None:
    repro_version = inventory.get("repro_sklearn_version")
    if repro_version != REPRO_PACKAGE_CONTRACT["scikit_learn"]:
        raise RuntimeReproError(
            "reproduction scikit-learn must be exactly "
            f"{REPRO_PACKAGE_CONTRACT['scikit_learn']} (observed {repro_version!r})"
        )
    main_python = str(inventory.get("main_python_version") or "")
    repro_python = str(inventory.get("repro_python_version") or "")
    if tuple(main_python.split(".")[:2]) != tuple(repro_python.split(".")[:2]):
        raise RuntimeReproError("Python major/minor mismatch between main and reproduction runtimes")
    inventory_fields = {
        "numpy": "repro_numpy_version",
        "scipy": "repro_scipy_version",
        "joblib": "repro_joblib_version",
        "threadpoolctl": "repro_threadpoolctl_version",
    }
    for package in ("numpy", "scipy", "joblib", "threadpoolctl"):
        expected = REPRO_PACKAGE_CONTRACT[package]
        observed = inventory.get(inventory_fields[package])
        if observed != expected:
            raise RuntimeReproError(
                f"reproduction {package} must be exactly {expected} (observed {observed!r})"
            )
    if inventory.get("repro_torch_installed") is not False:
        raise RuntimeReproError("torch is prohibited in the scaler-only reproduction environment")


def validate_bundle_contract(
    bundle: Path | str, policy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest, snapshot, records = load_historical_bundle(bundle)
    except Exception as exc:
        raise RuntimeReproError(f"Phase 22 bundle integrity failed: {exc}") from exc
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "capture_complete": manifest.get("capture_type") == "immutable_historical_alignment",
        "timeframe": manifest.get("timeframe") == policy["required_timeframe"],
        "sequence_length": manifest.get("sequence_length") == policy["required_sequence_length"],
        "feature_width": manifest.get("feature_count") == policy["required_feature_count"],
        "source_conflicts_zero": int(manifest.get("conflicting_source_bar_count", -1)) == 0,
        "window_conflicts_zero": len({row["source_bar_id"] for row in records}) == len(records),
        "window_digest_count": int(manifest.get("window_digest_count", -1)) == len(records),
        "artifact_digests_recorded": all(
            re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field) or "")) is not None
            for entry in snapshot.get("model_entries", [])
            for field in ("model_sha256", "scaler_sha256")
        ),
        "serialized_scaler_version": all(
            entry.get("scaler_serialized_sklearn_version") == "1.8.0"
            for entry in snapshot.get("model_entries", [])
        ),
        "required_model_inventory": {
            str(entry.get("kind")) for entry in snapshot.get("model_entries", [])
        } == set(MODEL_KINDS),
        "window_records_complete": all(
            set((
                "symbol", "source_bar_id", "source_bar_open_utc", "source_bar_close_utc",
                "feature_window_digest", "feature_window_first_utc", "feature_window_last_utc",
                "feature_window_row_count",
            )).issubset(row)
            and int(row.get("feature_window_row_count", -1)) == policy["required_sequence_length"]
            for row in records
        ),
    }
    required = set(policy["required_serving_symbols"])
    checks["required_symbols"] = required.issubset(set(manifest.get("symbols", [])))
    checks["minimum_bars"] = all(
        int(manifest.get("unique_completed_bars_by_symbol", {}).get(symbol, 0))
        >= int(policy["minimum_aligned_unique_bars_per_symbol"])
        for symbol in required
    )
    if not snapshot.get("model_entries"):
        checks["artifact_digests_recorded"] = False
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeReproError("Phase 22 bundle contract failed: " + ", ".join(failed))
    manifest = dict(manifest)
    manifest["phase23_integrity_checks"] = checks
    return manifest, snapshot, records


def extract_immutable_windows(
    bundle: Path | str,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    root = Path(bundle)
    frames = _read_feature_matrices(root, manifest)
    columns = canonical_feature_columns(bool(manifest.get("add_symbol_id")))
    windows: list[np.ndarray] = []
    ordered_records: list[Mapping[str, Any]] = []
    seen: dict[str, str] = {}
    for record in records:
        symbol = str(record.get("symbol") or "")
        if symbol not in frames:
            raise RuntimeReproError(f"window references an unknown symbol: {symbol}")
        identity = str(record.get("source_bar_id") or "")
        recorded_digest = str(record.get("feature_window_digest") or "")
        if identity in seen:
            if seen[identity] != recorded_digest:
                raise RuntimeReproError(f"conflicting window digest: {identity}")
            continue
        frame = frames[symbol]
        window = _window_from_record(frame, record, int(manifest["sequence_length"]))
        # The frame index is retained solely to authenticate the exact Phase 22 digest.
        selected_index = frame.loc[frame.index <= str(record["feature_window_last_utc"])].tail(
            int(manifest["sequence_length"])
        ).index
        observed_digest = feature_window_digest(
            symbol, str(manifest["timeframe"]), columns, selected_index, window
        )
        if observed_digest != recorded_digest:
            raise RuntimeReproError(f"window digest mismatch: {identity}")
        seen[identity] = recorded_digest
        windows.append(window)
        ordered_records.append(record)
    if len(windows) != int(manifest.get("window_digest_count", -1)):
        raise RuntimeReproError("authenticated window count differs from bundle manifest")
    arrays: dict[str, np.ndarray] = {
        "windows": np.ascontiguousarray(np.stack(windows), dtype=np.float32),
        "symbols": np.asarray([str(row["symbol"]) for row in ordered_records]),
        "source_bar_ids": np.asarray([str(row["source_bar_id"]) for row in ordered_records]),
        "source_bar_open_utc": np.asarray([str(row["source_bar_open_utc"]) for row in ordered_records]),
        "source_bar_close_utc": np.asarray([str(row["source_bar_close_utc"]) for row in ordered_records]),
        "feature_window_digests": np.asarray([
            str(row["feature_window_digest"]) for row in ordered_records
        ]),
    }
    arrays["input_windows_digest"] = np.asarray(windows_payload_digest(arrays))
    return arrays


def _safe_artifact_path(root: Path, filename: Any) -> Path:
    path = (root / str(filename or "")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeReproError("snapshot contains an unsafe artifact path") from exc
    return path


def verify_snapshot_artifacts(snapshot: Mapping[str, Any], *, root: Path = BASE_DIR) -> None:
    for entry in snapshot.get("model_entries", []):
        kind = str(entry.get("kind") or "unknown")
        for field, digest_field in (
            ("model_filename", "model_sha256"), ("scaler_filename", "scaler_sha256")
        ):
            path = _safe_artifact_path(root, entry.get(field))
            if not path.is_file() or file_digest(path) != entry.get(digest_field):
                raise RuntimeReproError(f"incumbent artifact changed or is missing: {kind}/{field}")


def _run_worker(
    python: Path | str,
    *,
    scaler: Path,
    windows_npz: Path,
    output_npz: Path,
    manifest_out: Path,
) -> dict[str, Any]:
    command = [
        str(python), str(BASE_DIR / "tools" / "model_scaler_worker.py"),
        "--scaler", str(scaler), "--windows-npz", str(windows_npz),
        "--output-npz", str(output_npz), "--manifest-out", str(manifest_out),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=300, cwd=BASE_DIR,
    )
    if completed.returncode:
        tail = " ".join(completed.stderr.strip().split())[-1000:]
        raise RuntimeReproError(f"scaler worker failed ({completed.returncode}): {tail}")
    try:
        manifest = json.loads(manifest_out.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeReproError("scaler worker did not write a valid manifest") from exc
    observed_digest = json_digest({
        key: value for key, value in manifest.items() if key != "worker_result_digest"
    })
    if manifest.get("worker_result_digest") != observed_digest:
        raise RuntimeReproError("scaler worker result digest mismatch")
    return manifest


def _worker_pair(
    python: Path | str,
    *,
    label: str,
    kind: str,
    scaler: Path,
    windows_npz: Path,
    work: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], bool]:
    input_arrays = read_npz(windows_npz)
    expected_input_digest = str(np.asarray(input_arrays["input_windows_digest"]).reshape(-1)[0])
    expected_scaler_digest = file_digest(scaler)
    expected_worker_digest = file_digest(BASE_DIR / "tools" / "model_scaler_worker.py")
    manifests: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for repeat in (1, 2):
        output = work / f"{kind}-{label}-{repeat}.npz"
        manifest_path = work / f"{kind}-{label}-{repeat}.json"
        manifest = _run_worker(
            python, scaler=scaler, windows_npz=windows_npz,
            output_npz=output, manifest_out=manifest_path,
        )
        transformed = read_npz(output)
        if (
            manifest.get("scaler_digest") != expected_scaler_digest
            or manifest.get("input_windows_digest") != expected_input_digest
            or manifest.get("worker_code_digest") != expected_worker_digest
            or manifest.get("output_float64_digest")
            != logical_array_digest({"transformed_float64": transformed.get("transformed_float64")})
            or manifest.get("output_float32_digest")
            != logical_array_digest({"transformed_float32": transformed.get("transformed_float32")})
        ):
            raise RuntimeReproError("scaler worker output authentication failed")
        for metadata_name in (
            "symbols", "source_bar_ids", "source_bar_open_utc", "source_bar_close_utc",
            "feature_window_digests", "input_windows_digest",
        ):
            if metadata_name not in transformed or not np.array_equal(
                np.asarray(transformed[metadata_name]), np.asarray(input_arrays[metadata_name])
            ):
                raise RuntimeReproError(f"scaler worker changed immutable metadata: {metadata_name}")
        manifests.append(manifest)
        arrays.append(transformed)
    deterministic = (
        manifests[0].get("worker_result_digest") == manifests[1].get("worker_result_digest")
        and manifests[0].get("output_float64_digest") == manifests[1].get("output_float64_digest")
        and manifests[0].get("output_float32_digest") == manifests[1].get("output_float32_digest")
        and logical_array_digest(arrays[0]) == logical_array_digest(arrays[1])
        and file_digest(work / f"{kind}-{label}-1.npz")
        == file_digest(work / f"{kind}-{label}-2.npz")
    )
    return manifests[0], arrays[0], deterministic


def _comparison_metrics(current: np.ndarray, repro: np.ndarray) -> dict[str, Any]:
    if current.shape != repro.shape:
        raise RuntimeReproError("transformed array shape mismatch")
    if not np.isfinite(current).all() or not np.isfinite(repro).all():
        raise RuntimeReproError("transformed comparison contains non-finite values")
    exact = np.equal(current, repro)
    absolute = np.abs(current - repro)
    denominator = np.maximum(np.maximum(np.abs(current), np.abs(repro)), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    return {
        "exact_equal_count": int(np.count_nonzero(exact)),
        "exact_equal_rate": float(np.mean(exact)) if exact.size else 1.0,
        "max_absolute_error": float(np.max(absolute)) if absolute.size else 0.0,
        "mean_absolute_error": float(np.mean(absolute)) if absolute.size else 0.0,
        "max_relative_error": float(np.max(relative)) if relative.size else 0.0,
        "absolute": absolute,
        "exact": exact,
    }


def compare_scaled_arrays(
    current_float64: np.ndarray,
    repro_float64: np.ndarray,
    current_float32: np.ndarray,
    repro_float32: np.ndarray,
    *,
    source_bar_ids: Sequence[str] | None = None,
    float64_tolerance: float = 1e-12,
    float32_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Compare scaler outputs without rounding and retain the largest-error location."""

    a64 = np.asarray(current_float64, dtype=np.float64)
    b64 = np.asarray(repro_float64, dtype=np.float64)
    a32 = np.asarray(current_float32, dtype=np.float32)
    b32 = np.asarray(repro_float32, dtype=np.float32)
    if a64.ndim != 3 or a32.shape != a64.shape or b64.shape != a64.shape or b32.shape != a64.shape:
        raise RuntimeReproError("scaled comparisons require matching [window,timestep,feature] arrays")
    m64, m32 = _comparison_metrics(a64, b64), _comparison_metrics(a32, b32)
    combined_window_difference = np.any(~m64["exact"], axis=(1, 2)) | np.any(~m32["exact"], axis=(1, 2))
    maximum_error = max(m64["max_absolute_error"], m32["max_absolute_error"])
    index = None
    if maximum_error > 0:
        index = np.unravel_index(int(np.argmax(m64["absolute"])), a64.shape)
        if m32["max_absolute_error"] > m64["max_absolute_error"]:
            index = np.unravel_index(int(np.argmax(m32["absolute"])), a32.shape)
    exact = m64["exact_equal_rate"] == 1.0 and m32["exact_equal_rate"] == 1.0
    inside = (
        m64["max_absolute_error"] <= float(float64_tolerance)
        and m32["max_absolute_error"] <= float(float32_tolerance)
    )
    result = {
        "window_count": int(a64.shape[0]),
        "element_count": int(a64.size),
        "float64_exact_equal_count": m64["exact_equal_count"],
        "float64_exact_equal_rate": m64["exact_equal_rate"],
        "float64_max_absolute_error": m64["max_absolute_error"],
        "float64_mean_absolute_error": m64["mean_absolute_error"],
        "float64_max_relative_error": m64["max_relative_error"],
        "float32_exact_equal_count": m32["exact_equal_count"],
        "float32_exact_equal_rate": m32["exact_equal_rate"],
        "float32_max_absolute_error": m32["max_absolute_error"],
        "float32_mean_absolute_error": m32["mean_absolute_error"],
        "differing_window_count": int(np.count_nonzero(combined_window_difference)),
        "largest_error_source_bar_id": (
            None if source_bar_ids is None or index is None else str(source_bar_ids[index[0]])
        ),
        "largest_error_feature_index": None if index is None else int(index[2]),
        "largest_error_timestep": None if index is None else int(index[1]),
        "classification": (
            "exact_match" if exact else "numerically_equivalent" if inside else "materially_different"
        ),
    }
    result["comparison_digest"] = json_digest(result)
    return result


def _load_model_once(entry: Mapping[str, Any]):
    import torch
    from ml_dl.dl_infer import _build_model

    model_path = _safe_artifact_path(BASE_DIR, entry.get("model_filename"))
    if file_digest(model_path) != entry.get("model_sha256"):
        raise RuntimeReproError(f"model digest mismatch: {entry.get('kind')}")
    model = _build_model(str(entry["kind"]), int(entry["metadata_n_features"]))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().cpu()
    return model


def forward_scaled_arrays(model: Any, values: np.ndarray, *, chunk_size: int = 32) -> dict[str, np.ndarray]:
    import torch

    model.eval().cpu()
    outputs: dict[str, list[np.ndarray]] = {
        name: [] for name in ("ret_hat", "rv_hat", "logits", "raw_probability")
    }
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            tensor = torch.from_numpy(np.ascontiguousarray(values[start:start + chunk_size], dtype=np.float32))
            result = model(tensor)
            for required in ("ret_reg", "rv_reg", "ret_cls_logits"):
                if required not in result:
                    raise RuntimeReproError(f"model output missing {required}")
            outputs["ret_hat"].append(result["ret_reg"].detach().cpu().numpy().reshape(-1))
            outputs["rv_hat"].append(result["rv_reg"].detach().cpu().numpy().reshape(-1))
            outputs["logits"].append(result["ret_cls_logits"].detach().cpu().numpy())
            probabilities = torch.softmax(result["ret_cls_logits"], dim=-1)[:, 1]
            outputs["raw_probability"].append(probabilities.detach().cpu().numpy().reshape(-1))
    ret = np.concatenate(outputs["ret_hat"]).astype(np.float64)
    rv = np.concatenate(outputs["rv_hat"]).astype(np.float64)
    logits = np.concatenate(outputs["logits"]).astype(np.float64)
    probability = np.clip(
        np.concatenate(outputs["raw_probability"]).astype(np.float64), 1e-6, 1.0 - 1e-6
    )
    return {"ret_hat": ret, "rv_hat": rv, "logits": logits, "raw_probability": probability}


def apply_calibration(probabilities: np.ndarray, bias: float, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    biased, calibrated = zip(*(
        calibrate_probability(float(value), float(bias), float(temperature))
        for value in np.asarray(probabilities).reshape(-1)
    ))
    return np.asarray(biased, dtype=np.float64), np.asarray(calibrated, dtype=np.float64)


def collapse_statistics(
    probabilities: Sequence[float],
    source_bar_ids: Sequence[str],
    *,
    policy: Mapping[str, Any],
    deterministic: bool = True,
    expected_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    identities = [str(item) for item in source_bar_ids]
    if len(values) != len(identities):
        raise RuntimeReproError("probabilities and source identities differ in length")
    expected = len(values) if expected_count is None else int(expected_count)
    history: list[float] = []
    seen: set[str] = set()
    consecutive = 0
    max_consecutive = 0
    extreme_active = flat_active = False
    extreme_events = flat_events = excluded = flat_endpoints = 0
    row_states: list[dict[str, Any]] = []
    finite_values: list[float] = []
    for identity, raw in zip(identities, values):
        if identity in seen:
            continue
        seen.add(identity)
        finite = math.isfinite(float(raw))
        if not finite:
            excluded += 1
            row_states.append({"source_bar_id": identity, "extreme": False, "flat": False, "excluded": True})
            continue
        value = float(raw)
        finite_values.append(value)
        is_extreme_value = value < policy["extreme_low_threshold"] or value > policy["extreme_high_threshold"]
        consecutive = consecutive + 1 if is_extreme_value else 0
        max_consecutive = max(max_consecutive, consecutive)
        history.append(value)
        if len(history) > int(policy["flat_window"]):
            history.pop(0)
        rolling_std = float(np.std(history)) if len(history) == int(policy["flat_window"]) else None
        extreme = consecutive >= int(policy["extreme_consecutive_limit"])
        flat = rolling_std is not None and rolling_std < float(policy["flat_output_std_threshold"])
        extreme_started = extreme and not extreme_active
        flat_started = flat and not flat_active
        if extreme_started:
            extreme_events += 1
        if flat_started:
            flat_events += 1
        extreme_active, flat_active = extreme, flat
        if flat:
            flat_endpoints += 1
        if extreme or flat:
            excluded += 1
        row_states.append({
            "source_bar_id": identity, "extreme": extreme, "flat": flat,
            "excluded": bool(extreme or flat), "rolling_std": rolling_std,
            "extreme_event_started": bool(extreme_started),
            "flat_event_started": bool(flat_started),
            "event_types": [
                name for name, active in (
                    ("extreme_collapse", extreme_started), ("flat_output", flat_started)
                ) if active
            ],
        })
    array = np.asarray(finite_values, dtype=np.float64)
    missing_rate = max(0, expected - len(array)) / expected if expected else 1.0
    extreme_low_rate = float(np.mean(array < policy["extreme_low_threshold"])) if array.size else None
    extreme_high_rate = float(np.mean(array > policy["extreme_high_threshold"])) if array.size else None
    standard_deviation = float(np.std(array)) if array.size else None
    one_sided = bool(
        array.size and standard_deviation is not None
        and standard_deviation >= float(policy["flat_output_std_threshold"])
        and max(float(np.mean(array > 0.5)), float(np.mean(array < 0.5))) > 0.95
    )
    failed = (
        excluded > 0 or flat_endpoints > 0 or extreme_events > 0
        or missing_rate > float(policy["maximum_missing_rate"]) or not deterministic
    )
    status = "failed_health_gate" if failed else "healthy_aligned"
    return {
        "calibrated_probability_mean": float(np.mean(array)) if array.size else None,
        "calibrated_probability_std": standard_deviation,
        "minimum": float(np.min(array)) if array.size else None,
        "maximum": float(np.max(array)) if array.size else None,
        "extreme_low_rate": extreme_low_rate,
        "extreme_high_rate": extreme_high_rate,
        "rolling_flat_window_count": flat_endpoints,
        "consecutive_extreme_max": max_consecutive,
        "extreme_exclusion_events": extreme_events,
        "flat_exclusion_events": flat_events,
        "excluded_endpoint_count": excluded,
        "survival_rate": (expected - excluded) / expected if expected else 0.0,
        "one_sided": one_sided,
        "deterministic": bool(deterministic),
        "missing_rate": missing_rate,
        "collapse_status": status,
    }, row_states


def compare_model_output_records(
    current: Mapping[str, np.ndarray],
    repro: Mapping[str, np.ndarray],
    *,
    current_states: Sequence[Mapping[str, Any]] | None = None,
    repro_states: Sequence[Mapping[str, Any]] | None = None,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    required = ("raw_probability", "after_bias_probability", "after_temperature_probability", "ret_hat", "rv_hat")
    if any(np.asarray(current[name]).shape != np.asarray(repro[name]).shape for name in required):
        raise RuntimeReproError("model output comparison shape mismatch")
    if any(
        not np.isfinite(np.asarray(source[name], dtype=float)).all()
        for source in (current, repro) for name in required
    ):
        raise RuntimeReproError("model output comparison contains non-finite values")
    probability_error_by_stage = {
        name: np.abs(np.asarray(current[name], dtype=float) - np.asarray(repro[name], dtype=float))
        for name in required[:3]
    }
    probability_errors = np.concatenate(list(probability_error_by_stage.values()))
    ret_errors = np.abs(np.asarray(current["ret_hat"], dtype=float) - np.asarray(repro["ret_hat"], dtype=float))
    rv_errors = np.abs(np.asarray(current["rv_hat"], dtype=float) - np.asarray(repro["rv_hat"], dtype=float))
    current_p = np.asarray(current["after_temperature_probability"], dtype=float)
    repro_p = np.asarray(repro["after_temperature_probability"], dtype=float)
    current_direction = np.where(current_p > 0.5, 1, np.where(current_p < 0.5, -1, 0))
    repro_direction = np.where(repro_p > 0.5, 1, np.where(repro_p < 0.5, -1, 0))
    changed_direction = int(np.count_nonzero(current_direction != repro_direction))
    current_extreme = (current_p < policy["extreme_low_threshold"]) | (current_p > policy["extreme_high_threshold"])
    repro_extreme = (repro_p < policy["extreme_low_threshold"]) | (repro_p > policy["extreme_high_threshold"])
    changed_extreme = int(np.count_nonzero(current_extreme != repro_extreme))
    current_states = list(current_states or [])
    repro_states = list(repro_states or [])
    changed_flat = changed_exclusion = 0
    if current_states and repro_states:
        if len(current_states) != len(repro_states) or len(current_states) != len(current_p):
            raise RuntimeReproError("model health-state comparison length mismatch")
        changed_flat = sum(bool(a.get("flat")) != bool(b.get("flat")) for a, b in zip(current_states, repro_states))
        if all("event_types" in item for item in [*current_states, *repro_states]):
            changed_exclusion = sum(
                set(a.get("event_types", [])) != set(b.get("event_types", []))
                for a, b in zip(current_states, repro_states)
            )
        else:
            # Backward-compatible synthetic state records: an exclusion change
            # is necessarily a material simulated decision change.
            changed_exclusion = sum(
                bool(a.get("excluded")) != bool(b.get("excluded"))
                for a, b in zip(current_states, repro_states)
            )
    # Allow and agreement decisions are ensemble properties and are populated by
    # the per-model counterfactual ensemble pass after every model is available.
    changed_allow = 0
    decision_change = any((changed_direction, changed_extreme, changed_flat, changed_exclusion, changed_allow))
    # Numeric byte equality is not an exact behavioral match when a downstream
    # health/exclusion state changed.  In particular, event edges may differ
    # while both endpoints remain excluded.
    exact = (
        all(np.array_equal(np.asarray(current[name]), np.asarray(repro[name])) for name in required)
        and not decision_change
    )
    probability_max = float(np.max(probability_errors)) if probability_errors.size else 0.0
    regression_max = max(float(np.max(ret_errors)) if ret_errors.size else 0.0, float(np.max(rv_errors)) if rv_errors.size else 0.0)
    inside = (
        probability_max <= float(policy["probability_max_abs_error"])
        and regression_max <= float(policy["regression_output_max_abs_error"])
        and not decision_change
    )
    result = {
        "probability_max_absolute_error": probability_max,
        "probability_mean_absolute_error": float(np.mean(probability_errors)) if probability_errors.size else 0.0,
        "raw_probability_max_absolute_error": float(
            np.max(probability_error_by_stage["raw_probability"])
        ) if probability_error_by_stage["raw_probability"].size else 0.0,
        "bias_adjusted_probability_max_absolute_error": float(
            np.max(probability_error_by_stage["after_bias_probability"])
        ) if probability_error_by_stage["after_bias_probability"].size else 0.0,
        "temperature_adjusted_probability_max_absolute_error": float(
            np.max(probability_error_by_stage["after_temperature_probability"])
        ) if probability_error_by_stage["after_temperature_probability"].size else 0.0,
        "ret_hat_max_absolute_error": float(np.max(ret_errors)) if ret_errors.size else 0.0,
        "rv_hat_max_absolute_error": float(np.max(rv_errors)) if rv_errors.size else 0.0,
        "changed_direction_count": changed_direction,
        "changed_extreme_classification_count": changed_extreme,
        "changed_flat_window_count": int(changed_flat),
        "changed_exclusion_event_count": int(changed_exclusion),
        "changed_allow_count": changed_allow,
        "changed_agreement_suppression_count": 0,
        "classification": (
            "output_exact_match" if exact else
            "output_numerically_equivalent" if inside else "output_materially_different"
        ),
    }
    result["comparison_digest"] = json_digest(result)
    return result


def compare_model_outputs(
    model: Any,
    current_scaled: np.ndarray,
    repro_scaled: np.ndarray,
    *,
    source_bar_ids: Sequence[str],
    bias: float,
    temperature: float,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    current = forward_scaled_arrays(model, current_scaled)
    current_repeat = forward_scaled_arrays(model, current_scaled)
    repro = forward_scaled_arrays(model, repro_scaled)
    repro_repeat = forward_scaled_arrays(model, repro_scaled)
    current_deterministic = all(np.array_equal(current[name], current_repeat[name]) for name in current)
    repro_deterministic = all(np.array_equal(repro[name], repro_repeat[name]) for name in repro)
    current["after_bias_probability"], current["after_temperature_probability"] = apply_calibration(
        current["raw_probability"], bias, temperature
    )
    repro["after_bias_probability"], repro["after_temperature_probability"] = apply_calibration(
        repro["raw_probability"], bias, temperature
    )
    current_health, current_states = collapse_statistics(
        current["after_temperature_probability"], source_bar_ids,
        policy=policy, deterministic=current_deterministic,
    )
    repro_health, repro_states = collapse_statistics(
        repro["after_temperature_probability"], source_bar_ids,
        policy=policy, deterministic=repro_deterministic,
    )
    comparison = compare_model_output_records(
        current, repro, current_states=current_states, repro_states=repro_states, policy=policy
    )
    comparison["current_forward_deterministic"] = current_deterministic
    comparison["repro_forward_deterministic"] = repro_deterministic
    if not current_deterministic or not repro_deterministic:
        comparison["classification"] = "output_comparison_failed"
    comparison["comparison_digest"] = json_digest({
        key: value for key, value in comparison.items() if key != "comparison_digest"
    })
    return comparison, current, repro, current_health, repro_health


def _phase22_stats(report: Mapping[str, Any], kind: str, symbol: str) -> dict[str, Any]:
    models = report.get("model_results", {})
    if kind in models and isinstance(models[kind], Mapping):
        return dict(models[kind].get("by_symbol", {}).get(symbol, {}))
    if symbol in models and isinstance(models[symbol], Mapping):
        return dict(models[symbol].get(kind, {}))
    return {}


def compare_collapse_status(
    phase22: Mapping[str, Any],
    sklearn180: Mapping[str, Any],
    *,
    maximum_missing_rate: float = 0.05,
) -> dict[str, Any]:
    phase_status = str(phase22.get("model_health_status") or phase22.get("collapse_status") or "unavailable")
    repro_status = str(sklearn180.get("collapse_status") or "unavailable")
    phase_failed = phase_status.startswith("failed_") or phase_status == "failed_health_gate"
    repro_failed = repro_status.startswith("failed_") or repro_status == "failed_health_gate"
    resolved = bool(
        phase_failed and not repro_failed
        and int(sklearn180.get("excluded_endpoint_count", 0)) == 0
        and int(sklearn180.get("rolling_flat_window_count", 0)) == 0
        and int(sklearn180.get("extreme_exclusion_events", 0)) == 0
        and float(sklearn180.get("missing_rate", 1.0)) <= float(maximum_missing_rate)
        and sklearn180.get("deterministic") is True
    )
    return {
        "phase22_collapse_status": phase_status,
        "sklearn180_collapse_status": repro_status,
        "collapse_resolved_under_180": resolved,
        "collapse_persists_under_180": bool(phase_failed and repro_failed),
        "new_failure_under_180": bool(not phase_failed and repro_failed),
    }


def scaled_feature_diagnostics(values: np.ndarray, feature_names: Sequence[str]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(-1, array.shape[-1])
    mean, std = np.mean(flat, axis=0), np.std(flat, axis=0)
    minimum, maximum = np.min(flat, axis=0), np.max(flat, axis=0)
    rate3, rate5 = np.mean(np.abs(flat) > 3, axis=0), np.mean(np.abs(flat) > 5, axis=0)
    constant = std == 0
    near = (std > 0) & (std < 1e-6)
    mapping = lambda values: {str(name): float(value) for name, value in zip(feature_names, values)}
    symbol_index = list(feature_names).index("symbol_id") if "symbol_id" in feature_names else None
    return {
        "scaled_feature_mean_by_feature": mapping(mean),
        "scaled_feature_std_by_feature": mapping(std),
        "scaled_feature_min_by_feature": mapping(minimum),
        "scaled_feature_max_by_feature": mapping(maximum),
        "feature_abs_z_gt_3_rate": mapping(rate3),
        "feature_abs_z_gt_5_rate": mapping(rate5),
        "maximum_absolute_z": float(np.max(np.abs(flat))),
        "ood_feature_count": int(np.count_nonzero(rate3 > 0.05)),
        "constant_feature_count": int(np.count_nonzero(constant)),
        "near_constant_feature_count": int(np.count_nonzero(near)),
        "symbol_id_scaled_mean": None if symbol_index is None else float(mean[symbol_index]),
        "symbol_id_scaled_std": None if symbol_index is None else float(std[symbol_index]),
    }


def classifier_diagnostics(outputs: Mapping[str, np.ndarray], *, bias: float, temperature: float, policy: Mapping[str, Any]) -> dict[str, Any]:
    logits = np.asarray(outputs["logits"], dtype=np.float64)
    raw = np.asarray(outputs["raw_probability"], dtype=np.float64)
    biased = np.asarray(outputs["after_bias_probability"], dtype=np.float64)
    calibrated = np.asarray(outputs["after_temperature_probability"], dtype=np.float64)
    entropy = -(raw * np.log(raw) + (1 - raw) * np.log(1 - raw))
    margin = logits[:, 1] - logits[:, 0]
    raw_health, _ = collapse_statistics(raw, [str(i) for i in range(len(raw))], policy=policy)
    calibrated_health, _ = collapse_statistics(calibrated, [str(i) for i in range(len(raw))], policy=policy)
    return {
        "class0_logit_mean": float(np.mean(logits[:, 0])),
        "class1_logit_mean": float(np.mean(logits[:, 1])),
        "logit_margin_mean": float(np.mean(margin)),
        "logit_margin_std": float(np.std(margin)),
        "probability_entropy_mean": float(np.mean(entropy)),
        "probability_entropy_min": float(np.min(entropy)),
        "saturation_rate": float(np.mean((raw < 0.01) | (raw > 0.99))),
        "raw_std": float(np.std(raw)),
        "after_bias_std": float(np.std(biased)),
        "after_temperature_std": float(np.std(calibrated)),
        "clipping_count": int(np.count_nonzero((raw - bias <= 1e-6) | (raw - bias >= 1 - 1e-6))),
        "exclusion_events_before_calibration": int(
            raw_health["extreme_exclusion_events"] + raw_health["flat_exclusion_events"]
        ),
        "exclusion_events_after_calibration": int(
            calibrated_health["extreme_exclusion_events"] + calibrated_health["flat_exclusion_events"]
        ),
        "configured_bias": float(bias),
        "configured_temperature": float(temperature),
    }


def gradient_sensitivity(model: Any, values: np.ndarray, feature_names: Sequence[str]) -> dict[str, Any]:
    import torch

    sample_indices = np.linspace(0, len(values) - 1, min(8, len(values)), dtype=int)
    tensor = torch.from_numpy(np.ascontiguousarray(values[sample_indices], dtype=np.float32)).requires_grad_(True)
    model.eval().cpu()
    result = model(tensor)
    margin = result["ret_cls_logits"][:, 1] - result["ret_cls_logits"][:, 0]
    # Request only the diagnostic input gradient.  This avoids populating
    # parameter .grad fields and makes the no-training boundary explicit.
    gradient_tensor, = torch.autograd.grad(
        margin.mean(), tensor, retain_graph=False, create_graph=False
    )
    gradient = gradient_tensor.detach().cpu().numpy().astype(np.float64)
    absolute = np.abs(gradient)
    by_feature = np.mean(absolute, axis=(0, 1))
    by_time = np.mean(absolute, axis=(0, 2))
    return {
        "gradient_norm_summary": {
            "l1_mean": float(np.mean(absolute)),
            "l2": float(np.linalg.norm(gradient)),
            "maximum_absolute": float(np.max(absolute)),
            "sample_count": int(len(sample_indices)),
        },
        "temporal_sensitivity": {
            "first_timestep_mean_abs": float(by_time[0]),
            "last_timestep_mean_abs": float(by_time[-1]),
            "maximum_timestep": int(np.argmax(by_time)),
        },
        "feature_sensitivity_summary": {
            "mean_abs_by_feature": {
                str(name): float(value) for name, value in zip(feature_names, by_feature)
            },
            "maximum_feature": str(feature_names[int(np.argmax(by_feature))]),
        },
    }


def compare_ensemble_decisions(
    current_outputs: Sequence[Mapping[str, Any]],
    repro_outputs: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    current_rows, _ = evaluate_ensemble_variants(current_outputs, snapshot)
    repro_rows, _ = evaluate_ensemble_variants(repro_outputs, snapshot)
    current_map = {(row["symbol"], row["source_bar_id"], row["variant"]): row for row in current_rows}
    repro_map = {(row["symbol"], row["source_bar_id"], row["variant"]): row for row in repro_rows}
    if set(current_map) != set(repro_map):
        raise RuntimeReproError("ensemble comparison endpoint inventory differs")
    by_symbol: dict[str, dict[str, int]] = defaultdict(lambda: {
        "changed_allow_count": 0, "changed_direction_count": 0,
        "changed_agreement_suppression_count": 0,
    })
    for key, current in current_map.items():
        repro = repro_map[key]
        symbol = key[0]
        by_symbol[symbol]["changed_allow_count"] += int(current["allow"] != repro["allow"])
        by_symbol[symbol]["changed_direction_count"] += int(current["direction"] != repro["direction"])
        by_symbol[symbol]["changed_agreement_suppression_count"] += int(
            current["agreement_suppressed"] != repro["agreement_suppressed"]
        )
    return dict(by_symbol)


def _output_rows(
    kind: str,
    symbol: str,
    source_ids: Sequence[str],
    feature_digests: Sequence[str],
    outputs: Mapping[str, np.ndarray],
    health_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [{
        "source_bar_id": str(source_ids[index]), "symbol": symbol,
        "feature_window_digest": str(feature_digests[index]), "model_kind": kind,
        "raw_probability": float(outputs["raw_probability"][index]),
        "after_bias_probability": float(outputs["after_bias_probability"][index]),
        "after_temperature_probability": float(outputs["after_temperature_probability"][index]),
        "ret_hat": float(outputs["ret_hat"][index]), "rv_hat": float(outputs["rv_hat"][index]),
        "model_present": True, "model_excluded": bool(health_states[index]["excluded"]),
    } for index in range(len(source_ids))]


def build_runtime_reproducibility_report(
    *,
    bundle: Path | str,
    current_python: Path | str,
    repro_python: Path | str,
    policy: Mapping[str, Any],
    phase22_report: Mapping[str, Any] | None = None,
    working_dir: Path | str | None = None,
    keep_comparison_arrays: bool = False,
    models: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    if Path(current_python).resolve() != Path(sys.executable).resolve():
        raise RuntimeReproError(
            "--current-python must be the interpreter executing model_runtime_repro.py"
        )
    manifest, snapshot, records = validate_bundle_contract(bundle, policy)
    verify_snapshot_artifacts(snapshot)
    if phase22_report is not None:
        verify_report_digest(phase22_report, "alignment_digest")
        if (
            phase22_report.get("historical_alignment", {}).get("bundle_digest")
            != manifest.get("bundle_digest")
        ):
            raise RuntimeReproError("Phase 22 report does not match the immutable bundle")
    inventory = collect_dependency_inventory(
        current_python=current_python, repro_python=repro_python, snapshot=snapshot
    )
    validate_repro_environment(inventory)
    arrays = extract_immutable_windows(bundle, manifest, records)
    selected_models = [kind for kind in MODEL_KINDS if not models or kind in models]
    selected_symbols = [symbol for symbol in manifest["symbols"] if not symbols or symbol in symbols]
    if not selected_models or not selected_symbols:
        raise RuntimeReproError("model/symbol selection is empty")
    temporary = None
    if working_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="phase23-runtime-repro-")
        work = Path(temporary.name)
    else:
        work = ensure_safe_working_directory(working_dir)
        work.mkdir(parents=True, exist_ok=True)
    windows_npz = work / "immutable-windows.npz"
    write_deterministic_npz(windows_npz, arrays)
    entries = {str(entry["kind"]): entry for entry in snapshot["model_entries"]}
    scaler_comparisons: dict[str, Any] = {}
    output_comparisons: dict[str, Any] = {}
    collapse_comparisons: dict[str, Any] = {}
    analysis_evidence: dict[str, Any] = {}
    current_ensemble_rows: list[dict[str, Any]] = []
    repro_ensemble_rows: list[dict[str, Any]] = []
    current_rows_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repro_rows_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worker_deterministic = True
    forward_deterministic = True
    any_material = False
    feature_names = canonical_feature_columns(bool(manifest["add_symbol_id"]))
    for kind in selected_models:
        if kind not in entries:
            raise RuntimeReproError(f"snapshot is missing selected model: {kind}")
        entry = entries[kind]
        scaler_path = _safe_artifact_path(BASE_DIR, entry["scaler_filename"])
        current_manifest, current_arrays, current_deterministic = _worker_pair(
            current_python, label="current", kind=kind, scaler=scaler_path,
            windows_npz=windows_npz, work=work,
        )
        repro_manifest, repro_arrays, repro_deterministic = _worker_pair(
            repro_python, label="sklearn180", kind=kind, scaler=scaler_path,
            windows_npz=windows_npz, work=work,
        )
        current_versions = current_manifest.get("runtime_versions", {})
        repro_versions = repro_manifest.get("runtime_versions", {})
        if (
            current_versions.get("python") != inventory["main_python_version"]
            or current_versions.get("numpy") != inventory["main_numpy_version"]
            or current_versions.get("scipy") != inventory["main_scipy_version"]
            or current_versions.get("joblib") != inventory["main_joblib_version"]
            or current_versions.get("scikit_learn") != inventory["main_sklearn_version"]
            or current_versions.get("threadpoolctl") != inventory["main_threadpoolctl_version"]
        ):
            raise RuntimeReproError("current scaler worker runtime differs from dependency inventory")
        if any(repro_versions.get(name) != version for name, version in {
            "numpy": REPRO_PACKAGE_CONTRACT["numpy"],
            "scipy": REPRO_PACKAGE_CONTRACT["scipy"],
            "joblib": REPRO_PACKAGE_CONTRACT["joblib"],
            "scikit_learn": REPRO_PACKAGE_CONTRACT["scikit_learn"],
            "threadpoolctl": REPRO_PACKAGE_CONTRACT["threadpoolctl"],
        }.items()):
            raise RuntimeReproError("reproduction scaler worker runtime differs from its contract")
        worker_deterministic &= current_deterministic and repro_deterministic
        scaler_comparisons[kind] = {
            "scaler_digest": entry["scaler_sha256"],
            "current_worker_manifest": current_manifest,
            "sklearn180_worker_manifest": repro_manifest,
            "current_worker_deterministic": current_deterministic,
            "sklearn180_worker_deterministic": repro_deterministic,
            "by_symbol": {},
        }
        output_comparisons[kind] = {"model_digest": entry["model_sha256"], "by_symbol": {}}
        collapse_comparisons[kind] = {"by_symbol": {}}
        analysis_evidence[kind] = {"by_symbol": {}}
        model = _load_model_once(entry)
        for symbol in selected_symbols:
            mask = np.asarray(arrays["symbols"]) == symbol
            source_ids = np.asarray(arrays["source_bar_ids"])[mask].tolist()
            feature_digests = np.asarray(arrays["feature_window_digests"])[mask].tolist()
            current64 = np.asarray(current_arrays["transformed_float64"])[mask]
            repro64 = np.asarray(repro_arrays["transformed_float64"])[mask]
            current32 = np.asarray(current_arrays["transformed_float32"])[mask]
            repro32 = np.asarray(repro_arrays["transformed_float32"])[mask]
            scaler_result = compare_scaled_arrays(
                current64, repro64, current32, repro32, source_bar_ids=source_ids,
                float64_tolerance=policy["float64_scaler_max_abs_error"],
                float32_tolerance=policy["float32_scaler_max_abs_error"],
            )
            scaler_comparisons[kind]["by_symbol"][symbol] = scaler_result
            bias = float(snapshot.get(f"dl_bias_{kind}", 0.0) or 0.0)
            temperature = float(snapshot.get(f"dl_temp_{kind}", 1.0) or 1.0)
            output_result, current_output, repro_output, current_health, repro_health = compare_model_outputs(
                model, current32, repro32, source_bar_ids=source_ids,
                bias=bias, temperature=temperature, policy=policy,
            )
            forward_deterministic &= (
                output_result["current_forward_deterministic"]
                and output_result["repro_forward_deterministic"]
            )
            output_comparisons[kind]["by_symbol"][symbol] = output_result
            current_health_states = collapse_statistics(
                current_output["after_temperature_probability"], source_ids,
                policy=policy, deterministic=output_result["current_forward_deterministic"],
            )[1]
            repro_health_states = collapse_statistics(
                repro_output["after_temperature_probability"], source_ids,
                policy=policy, deterministic=output_result["repro_forward_deterministic"],
            )[1]
            phase22 = _phase22_stats(phase22_report or {}, kind, symbol)
            collapse_comparisons[kind]["by_symbol"][symbol] = {
                "current_runtime": current_health,
                "sklearn180_runtime": repro_health,
                **compare_collapse_status(
                    phase22,
                    repro_health,
                    maximum_missing_rate=policy["maximum_missing_rate"],
                ),
            }
            analysis_evidence[kind]["by_symbol"][symbol] = {
                "input_scaler": scaled_feature_diagnostics(repro32, feature_names),
                "classifier_calibration": classifier_diagnostics(
                    repro_output, bias=bias, temperature=temperature, policy=policy
                ),
                "sensitivity": gradient_sensitivity(model, repro32, feature_names),
            }
            current_rows = _output_rows(
                kind, symbol, source_ids, feature_digests, current_output, current_health_states
            )
            repro_rows = _output_rows(
                kind, symbol, source_ids, feature_digests, repro_output, repro_health_states
            )
            current_ensemble_rows.extend(current_rows)
            repro_ensemble_rows.extend(repro_rows)
            current_rows_by_kind[kind].extend(current_rows)
            repro_rows_by_kind[kind].extend(repro_rows)
            any_material |= (
                scaler_result["classification"] == "materially_different"
                or output_result["classification"] == "output_materially_different"
            )
        scaler_exact = all(
            value["classification"] == "exact_match"
            for value in scaler_comparisons[kind]["by_symbol"].values()
        )
        outputs_exact = all(
            value["classification"] == "output_exact_match"
            for value in output_comparisons[kind]["by_symbol"].values()
        )
        kind_material = any(
            value["classification"] == "materially_different"
            for value in scaler_comparisons[kind]["by_symbol"].values()
        ) or any(
            value["classification"] == "output_materially_different"
            for value in output_comparisons[kind]["by_symbol"].values()
        )
        main_exact_version = inventory["main_sklearn_version"] == "1.8.0"
        kind_deterministic = bool(
            current_deterministic
            and repro_deterministic
            and all(
                value.get("current_forward_deterministic") is True
                and value.get("repro_forward_deterministic") is True
                for value in output_comparisons[kind]["by_symbol"].values()
            )
        )
        if not kind_deterministic:
            decision = "comparison_unavailable"
        elif kind_material:
            decision = "runtime_version_changes_behavior"
        elif scaler_exact and outputs_exact and main_exact_version:
            decision = "exact_runtime_reproduction"
        else:
            decision = "numerically_equivalent_runtime"
        scaler_comparisons[kind]["runtime_decision"] = decision
        output_comparisons[kind]["same_model_instance_for_both_inputs"] = True
    ensemble = compare_ensemble_decisions(current_ensemble_rows, repro_ensemble_rows, snapshot)
    if any(value for changes in ensemble.values() for value in changes.values()):
        any_material = True
    per_model_ensemble: dict[str, Any] = {}
    for kind in output_comparisons:
        counterfactual_rows = [
            row for other_kind, rows in current_rows_by_kind.items()
            if other_kind != kind for row in rows
        ] + list(repro_rows_by_kind[kind])
        changes_by_symbol = compare_ensemble_decisions(
            current_ensemble_rows, counterfactual_rows, snapshot
        )
        per_model_ensemble[kind] = changes_by_symbol
        for symbol, changes in changes_by_symbol.items():
            if symbol in output_comparisons[kind]["by_symbol"]:
                comparison = output_comparisons[kind]["by_symbol"][symbol]
                comparison["changed_allow_count"] = changes["changed_allow_count"]
                comparison["changed_agreement_suppression_count"] = changes[
                    "changed_agreement_suppression_count"
                ]
                comparison["changed_ensemble_direction_count"] = changes["changed_direction_count"]
                if any(changes.values()):
                    comparison["classification"] = "output_materially_different"
                    any_material = True
                comparison["comparison_digest"] = json_digest({
                    key: value for key, value in comparison.items() if key != "comparison_digest"
                })
        if any(value for changes in changes_by_symbol.values() for value in changes.values()):
            scaler_comparisons[kind]["runtime_decision"] = "runtime_version_changes_behavior"
        output_comparisons[kind]["runtime_decision"] = scaler_comparisons[kind]["runtime_decision"]
    full_scope = (
        set(selected_models) == set(MODEL_KINDS)
        and set(selected_symbols) == set(policy["required_serving_symbols"])
    )
    comparison_failed = any(
        result.get("classification") == "output_comparison_failed"
        for model in output_comparisons.values() for result in model["by_symbol"].values()
    )
    complete = worker_deterministic and forward_deterministic and full_scope and not comparison_failed
    all_scalers_exact = all(
        result.get("classification") == "exact_match"
        for model in scaler_comparisons.values() for result in model["by_symbol"].values()
    )
    all_outputs_exact = all(
        result.get("classification") == "output_exact_match"
        for model in output_comparisons.values() for result in model["by_symbol"].values()
    )
    exact_artifact_reproducibility = bool(
        complete and not any_material and all_scalers_exact and all_outputs_exact
    )
    # Re-authenticate after every worker and forward pass as well as before
    # comparison.  Phase 23 never writes these paths; this second check makes
    # that immutability claim evidence-backed for the duration of the run.
    verify_snapshot_artifacts(snapshot)
    decision = (
        "runtime_reproducibility_material_difference" if any_material else
        "runtime_reproducibility_verified" if complete else "runtime_reproducibility_failed"
    )
    verdict = (
        "runtime_reproducibility_material_behavior_delta" if any_material else
        "runtime_reproducibility_verified_no_material_delta" if complete else
        "runtime_reproducibility_failed"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy": {
            "offline_only": True, "orders_allowed": False, "models_modified": False,
            "main_environment_modified": False,
        },
        "input_bundle": {
            "bundle_digest": manifest["bundle_digest"],
            "serving_snapshot_digest": snapshot["snapshot_digest"],
            "timeframe": manifest["timeframe"], "sequence_length": manifest["sequence_length"],
            "feature_width": manifest["feature_count"],
            "unique_completed_bars_by_symbol": manifest["unique_completed_bars_by_symbol"],
            "window_count": len(records), "input_windows_digest": str(arrays["input_windows_digest"]),
            "integrity_checks": manifest["phase23_integrity_checks"], "integrity_result": "pass",
            "artifact_integrity_result": "pass",
            "artifact_integrity_verified_before_and_after": True,
        },
        "dependency_inventory": inventory,
        "scaler_comparisons": scaler_comparisons,
        "model_output_comparisons": output_comparisons,
        "collapse_comparisons": collapse_comparisons,
        "ensemble_decision_comparisons": ensemble,
        "per_model_ensemble_decision_comparisons": per_model_ensemble,
        "analysis_evidence": analysis_evidence,
        "overall_decision": {
            "decision": decision, "verdict": verdict,
            "worker_runs_deterministic": worker_deterministic,
            "model_forward_passes_deterministic": forward_deterministic,
            "material_behavior_difference": any_material,
            "full_required_model_and_symbol_scope": full_scope,
            "exact_artifact_reproducibility": exact_artifact_reproducibility,
            "exact_incumbent_dependency_contract": inventory["main_sklearn_version"] == "1.8.0",
        },
        "warnings": (
            ([] if inventory["main_sklearn_version"] == "1.8.0" else [
                "The observed main scikit-learn runtime differs from the serialized 1.8.0 artifact version; behavioral equivalence does not make the incumbent dependency contract exact."
            ])
            + ([] if full_scope else [
                "A model or symbol filter limited this report; partial comparisons cannot verify the Phase 23 runtime contract."
            ])
            + ([] if phase22_report is not None else [
                "The Phase 22 alignment report was unavailable; cross-runtime collapse was calculated, but its Phase 22 status could not be authenticated or compared."
            ])
        ),
    }
    report["reproducibility_digest"] = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", "reproducibility_digest"}
    })
    if temporary is not None and keep_comparison_arrays:
        destination = DEFAULT_REPORT.parent / "model_runtime_reproducibility_arrays"
        if destination.exists():
            raise RuntimeReproError("comparison-array destination already exists")
        shutil.copytree(work, destination)
    if temporary is not None:
        temporary.cleanup()
    return report


def _load_optional_json(path: Path | str | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeReproError("Phase 22 report must be a JSON object")
    verify_report_digest(value, "alignment_digest")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23 runtime reproducibility comparison")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--repro-python", default=str(DEFAULT_REPRO_PYTHON))
    parser.add_argument("--current-python", default=sys.executable)
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--phase22-report", default=str(BASE_DIR / "reports" / "model_alignment_report_final.json"))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--model", action="append", choices=MODEL_KINDS)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--working-dir")
    parser.add_argument("--keep-comparison-arrays", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.inventory_only:
            snapshot = None
            bundle_path = Path(args.bundle)
            if (bundle_path / "model_serving_snapshot.json").is_file():
                snapshot = json.loads((bundle_path / "model_serving_snapshot.json").read_text(encoding="utf-8-sig"))
            inventory = collect_dependency_inventory(
                current_python=args.current_python,
                repro_python=args.repro_python if Path(args.repro_python).is_file() else None,
                snapshot=snapshot,
            )
            print(json.dumps(inventory, indent=2))
            return 0
        policy = load_retraining_policy(args.policy)
        phase22 = _load_optional_json(args.phase22_report)
        report = build_runtime_reproducibility_report(
            bundle=args.bundle, current_python=args.current_python, repro_python=args.repro_python,
            policy=policy, phase22_report=phase22, working_dir=args.working_dir,
            keep_comparison_arrays=args.keep_comparison_arrays, models=args.model, symbols=args.symbol,
        )
        output = ensure_safe_report_output(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and report["overall_decision"]["verdict"] != "runtime_reproducibility_verified_no_material_delta":
            return 3
        return 0
    except Exception as exc:
        print(f"model_runtime_repro: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
