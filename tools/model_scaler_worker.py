"""Deterministic, scaler-only worker for Phases 23 and 23.1.

This module intentionally has no PyTorch or trading-runtime dependencies.  It
loads one immutable scaler, transforms an authenticated float32 window set,
and writes sklearn plus manual StandardScaler decomposition paths.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_INPUT_ARRAYS = (
    "windows",
    "symbols",
    "source_bar_ids",
    "source_bar_open_utc",
    "source_bar_close_utc",
    "feature_window_digests",
    "input_windows_digest",
)
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|private[_-]?key|wallet|password|token)\s*[:=]\s*\S+"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/][^\s]+|\\\\[^\s]+|/(?:home|users|tmp|var)/[^\s]+)"
)


class ScalerWorkerError(ValueError):
    """Raised when an input, scaler, or deterministic-output contract fails."""


def _assert_safe_output_paths(
    *,
    scaler: Path | str,
    windows_npz: Path | str,
    output_npz: Path | str,
    manifest_out: Path | str,
) -> None:
    inputs = {Path(scaler).resolve(), Path(windows_npz).resolve()}
    outputs = {Path(output_npz).resolve(), Path(manifest_out).resolve()}
    if len(outputs) != 2 or inputs & outputs:
        raise ScalerWorkerError("worker output paths must be distinct from each other and all inputs")
    protected_roots = (
        REPOSITORY_ROOT / "model_artifacts",
        REPOSITORY_ROOT / "config",
        REPOSITORY_ROOT / ".venv",
        REPOSITORY_ROOT / ".venv-repro-sklearn180",
        REPOSITORY_ROOT / ".venv-runtime-isolation",
    )
    for output in outputs:
        if output in {REPOSITORY_ROOT / ".env", REPOSITORY_ROOT / "features.py"}:
            raise ScalerWorkerError("worker output path targets a protected repository file")
        for protected in protected_roots:
            try:
                output.relative_to(protected.resolve())
            except ValueError:
                continue
            raise ScalerWorkerError("worker output path targets a protected repository directory")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _canonical_numeric_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        return array
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def logical_array_digest(arrays: Mapping[str, np.ndarray], *, exclude: Sequence[str] = ()) -> str:
    """Hash logical array content without ZIP metadata or host byte order."""

    omitted = set(exclude)
    digest = hashlib.sha256()
    for name in sorted(key for key in arrays if key not in omitted):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        if array.dtype.kind in "OUS":
            values = [str(item) for item in array.reshape(-1).tolist()]
            payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            header = {"kind": "text", "shape": list(array.shape)}
        else:
            canonical = _canonical_numeric_array(array)
            payload = canonical.tobytes(order="C")
            header = {"dtype": canonical.dtype.str, "shape": list(canonical.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(payload)
        digest.update(b"\x00")
    return digest.hexdigest()


def maximum_ulp_distance(left: np.ndarray, right: np.ndarray) -> int | None:
    """Return the deterministic maximum representable-step distance.

    ULP distance is meaningful only for same-shaped float32 or float64 arrays.
    The ordered unsigned representation keeps the calculation vectorized and
    well-defined across negative and positive finite values.
    """

    first, second = np.asarray(left), np.asarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ScalerWorkerError("ULP comparison requires equal shapes and dtypes")
    if first.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ScalerWorkerError("ULP comparison requires finite values")
    if first.dtype == np.dtype(np.float32):
        unsigned = np.uint32
        sign_mask = np.uint32(1 << 31)
        wide = np.uint64
    else:
        unsigned = np.uint64
        sign_mask = np.uint64(1 << 63)
        wide = np.uint64

    def ordered(value: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(value).view(unsigned)
        return np.where((bits & sign_mask) != 0, ~bits, bits ^ sign_mask).astype(wide)

    a, b = ordered(first), ordered(second)
    distance = np.maximum(a, b) - np.minimum(a, b)
    return int(np.max(distance)) if distance.size else 0


def numeric_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    first, second = np.asarray(left), np.asarray(right)
    if first.shape != second.shape:
        raise ScalerWorkerError("decomposition comparison shape mismatch")
    absolute = np.abs(first.astype(np.float64) - second.astype(np.float64))
    return {
        "exact_equal": bool(np.array_equal(first, second)),
        "exact_equal_count": int(np.count_nonzero(first == second)),
        "exact_equal_rate": float(np.mean(first == second)) if first.size else 1.0,
        "max_absolute_error": float(np.max(absolute)) if absolute.size else 0.0,
        "mean_absolute_error": float(np.mean(absolute)) if absolute.size else 0.0,
        "maximum_ulp_distance": (
            maximum_ulp_distance(first, second) if first.dtype == second.dtype else None
        ),
    }


def write_deterministic_npz(path: Path | str, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write an NPZ whose bytes do not depend on timestamps or input ordering."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0
            archive.writestr(info, buffer.getvalue())
    return target


def read_npz(path: Path | str) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    except Exception as exc:
        raise ScalerWorkerError(f"unable to read windows NPZ: {type(exc).__name__}") from exc


def windows_payload_digest(arrays: Mapping[str, np.ndarray]) -> str:
    return logical_array_digest(arrays, exclude=("input_windows_digest",))


def validate_windows_payload(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, str]:
    missing = [name for name in REQUIRED_INPUT_ARRAYS if name not in arrays]
    if missing:
        raise ScalerWorkerError(f"windows NPZ missing arrays: {missing}")
    windows = np.asarray(arrays["windows"])
    if windows.dtype != np.dtype(np.float32):
        raise ScalerWorkerError("raw windows must be ordered float32 values")
    if windows.ndim != 3 or not all(windows.shape):
        raise ScalerWorkerError("raw windows must have shape [window, timestep, feature]")
    if not np.isfinite(windows).all():
        raise ScalerWorkerError("raw windows contain non-finite values")
    count = int(windows.shape[0])
    for name in REQUIRED_INPUT_ARRAYS[1:-1]:
        if np.asarray(arrays[name]).reshape(-1).size != count:
            raise ScalerWorkerError(f"windows metadata length mismatch: {name}")
    recorded = str(np.asarray(arrays["input_windows_digest"]).reshape(-1)[0])
    observed = windows_payload_digest(arrays)
    if not re.fullmatch(r"[0-9a-f]{64}", recorded) or recorded != observed:
        raise ScalerWorkerError("input-window digest mismatch")
    return np.ascontiguousarray(windows, dtype=np.float32), observed


def sanitize_warning(message: str) -> str:
    value = str(message).replace("\r", " ").replace("\n", " ")
    value = _WINDOWS_ABSOLUTE_PATH_RE.sub("<path>", value)
    value = _SECRET_RE.sub(lambda match: match.group(1) + "=<redacted>", value)
    value = " ".join(value.split())
    return value[:2000]


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "joblib": str(joblib.__version__),
        "scikit_learn": str(sklearn.__version__),
        "threadpoolctl": str(threadpoolctl.__version__),
    }


def decompose_transform_paths(
    scaler_path: Path | str,
    windows: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[str], list[str], dict[str, Any], dict[str, Any]]:
    """Load a scaler read-only and execute sklearn/manual arithmetic paths.

    The authenticated source values are float32.  Casting those exact values to
    float64 before one transform preserves the input values while exercising
    scikit-learn's float64 path; the second transform exercises the production
    float32 path.  Merely up-casting one float32 result would make the float64
    comparison incapable of observing runtime differences below float32
    precision.
    """

    source = Path(scaler_path)
    if not source.is_file():
        raise ScalerWorkerError("scaler artifact is missing")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            scaler = joblib.load(source)
        except Exception as exc:
            raise ScalerWorkerError(f"scaler deserialization failed: {type(exc).__name__}") from exc
    expected_width = int(getattr(scaler, "n_features_in_", 0) or 0)
    if expected_width <= 0:
        raise ScalerWorkerError("scaler does not declare n_features_in_")
    if int(windows.shape[2]) != expected_width:
        raise ScalerWorkerError(
            f"scaler width mismatch: input={windows.shape[2]} scaler={expected_width}"
        )
    flat32 = np.ascontiguousarray(windows.reshape(-1, expected_width), dtype=np.float32)
    flat64 = np.ascontiguousarray(flat32, dtype=np.float64)
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ScalerWorkerError("scaler does not expose loaded mean_ and scale_")
    mean_loaded = np.asarray(scaler.mean_)
    scale_loaded = np.asarray(scaler.scale_)
    if (
        mean_loaded.shape != (expected_width,)
        or scale_loaded.shape != (expected_width,)
        or not np.isfinite(mean_loaded).all()
        or not np.isfinite(scale_loaded).all()
        or np.any(scale_loaded == 0)
    ):
        raise ScalerWorkerError("scaler mean_/scale_ decomposition contract is invalid")
    with warnings.catch_warnings(record=True) as transformed_warnings:
        warnings.simplefilter("always")
        transformed64 = np.asarray(scaler.transform(flat64), dtype=np.float64)
        transformed32 = np.asarray(scaler.transform(flat32), dtype=np.float32)
    if (
        transformed64.shape != flat64.shape
        or transformed32.shape != flat32.shape
        or not np.isfinite(transformed64).all()
        or not np.isfinite(transformed32).all()
    ):
        raise ScalerWorkerError("scaler produced an invalid transformed matrix")
    float64 = np.ascontiguousarray(transformed64, dtype=np.float64).reshape(windows.shape)
    float32 = np.ascontiguousarray(transformed32, dtype=np.float32).reshape(windows.shape)
    mean64 = np.ascontiguousarray(mean_loaded, dtype=np.float64)
    scale64 = np.ascontiguousarray(scale_loaded, dtype=np.float64)
    mean32 = np.ascontiguousarray(mean_loaded, dtype=np.float32)
    scale32 = np.ascontiguousarray(scale_loaded, dtype=np.float32)
    manual64_flat = np.ascontiguousarray((flat64 - mean64) / scale64, dtype=np.float64)
    manual64_then32_flat = np.ascontiguousarray(manual64_flat, dtype=np.float32)
    manual32_flat = np.ascontiguousarray((flat32 - mean32) / scale32, dtype=np.float32)
    paths = {
        "sklearn_transform_float64_input": float64,
        "sklearn_transform_float32_input": float32,
        "manual_float64_formula": manual64_flat.reshape(windows.shape),
        "manual_float64_then_float32": manual64_then32_flat.reshape(windows.shape),
        "manual_float32_formula": manual32_flat.reshape(windows.shape),
    }
    all_warnings = [*captured, *transformed_warnings]
    categories = [item.category.__name__ for item in all_warnings]
    messages = [sanitize_warning(str(item.message)) for item in all_warnings]
    metadata = {
        "class": f"{type(scaler).__module__}.{type(scaler).__name__}",
        "n_features_in": expected_width,
        "mean_digest": (
            logical_array_digest({"mean": np.asarray(scaler.mean_)})
            if hasattr(scaler, "mean_") else None
        ),
        "scale_digest": (
            logical_array_digest({"scale": np.asarray(scaler.scale_)})
            if hasattr(scaler, "scale_") else None
        ),
    }
    path_metadata = {
        "sklearn_transform_float64_input": {
            "input_dtype": "float64", "mean_dtype": str(mean_loaded.dtype),
            "scale_dtype": str(scale_loaded.dtype), "arithmetic_dtype": "runtime_managed",
            "output_dtype": "float64",
        },
        "sklearn_transform_float32_input": {
            "input_dtype": "float32", "mean_dtype": str(mean_loaded.dtype),
            "scale_dtype": str(scale_loaded.dtype), "arithmetic_dtype": "runtime_managed",
            "output_dtype": "float32",
        },
        "manual_float64_formula": {
            "input_dtype": "float64", "mean_dtype": "float64", "scale_dtype": "float64",
            "arithmetic_dtype": "float64", "output_dtype": "float64",
        },
        "manual_float64_then_float32": {
            "input_dtype": "float64", "mean_dtype": "float64", "scale_dtype": "float64",
            "arithmetic_dtype": "float64_then_cast", "output_dtype": "float32",
        },
        "manual_float32_formula": {
            "input_dtype": "float32", "mean_dtype": "float32", "scale_dtype": "float32",
            "arithmetic_dtype": "float32", "output_dtype": "float32",
        },
    }
    references = {
        "sklearn_transform_float64_input": "manual_float64_formula",
        "sklearn_transform_float32_input": "manual_float32_formula",
        "manual_float64_formula": "sklearn_transform_float64_input",
        "manual_float64_then_float32": "sklearn_transform_float32_input",
        "manual_float32_formula": "sklearn_transform_float32_input",
    }
    for name, values in paths.items():
        path_metadata[name]["output_digest"] = logical_array_digest({name: values})
        reference = references[name]
        path_metadata[name]["comparison_reference"] = reference
        path_metadata[name]["comparison"] = numeric_comparison(values, paths[reference])
    decomposition = {
        "formula": "(X - mean_) / scale_",
        "scaler_refit_performed": False,
        "paths": path_metadata,
        "comparisons": {
            "sklearn_float64_vs_manual_float64": numeric_comparison(float64, paths["manual_float64_formula"]),
            "sklearn_float32_vs_manual_float32": numeric_comparison(float32, paths["manual_float32_formula"]),
            "sklearn_float32_vs_manual_float64_then_float32": numeric_comparison(
                float32, paths["manual_float64_then_float32"]
            ),
            "manual_float32_vs_manual_float64_then_float32": numeric_comparison(
                paths["manual_float32_formula"], paths["manual_float64_then_float32"]
            ),
        },
    }
    return paths, categories, messages, metadata, decomposition


def transform_windows(
    scaler_path: Path | str,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Backward-compatible Phase 23 entry point."""

    paths, categories, messages, metadata, _ = decompose_transform_paths(scaler_path, windows)
    return (
        paths["sklearn_transform_float64_input"],
        paths["sklearn_transform_float32_input"],
        categories, messages, metadata,
    )


def run_worker(
    *,
    scaler: Path | str,
    windows_npz: Path | str,
    output_npz: Path | str,
    manifest_out: Path | str,
) -> dict[str, Any]:
    _assert_safe_output_paths(
        scaler=scaler,
        windows_npz=windows_npz,
        output_npz=output_npz,
        manifest_out=manifest_out,
    )
    arrays = read_npz(windows_npz)
    windows, input_digest = validate_windows_payload(arrays)
    paths, categories, messages, scaler_metadata, decomposition = decompose_transform_paths(
        scaler, windows
    )
    float64 = paths["sklearn_transform_float64_input"]
    float32 = paths["sklearn_transform_float32_input"]
    output_arrays = {
        "transformed_float64": float64,
        "transformed_float32": float32,
        **paths,
        "symbols": np.asarray(arrays["symbols"]),
        "source_bar_ids": np.asarray(arrays["source_bar_ids"]),
        "source_bar_open_utc": np.asarray(arrays["source_bar_open_utc"]),
        "source_bar_close_utc": np.asarray(arrays["source_bar_close_utc"]),
        "feature_window_digests": np.asarray(arrays["feature_window_digests"]),
        "input_windows_digest": np.asarray(input_digest),
    }
    output64_digest = logical_array_digest({"transformed_float64": float64})
    output32_digest = logical_array_digest({"transformed_float32": float32})
    write_deterministic_npz(output_npz, output_arrays)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runtime_versions": runtime_versions(),
        "scaler_digest": sha256_file(scaler),
        "scaler_metadata": scaler_metadata,
        "input_windows_digest": input_digest,
        "input_window_count": int(windows.shape[0]),
        "input_sequence_length": int(windows.shape[1]),
        "input_feature_width": int(windows.shape[2]),
        "output_float64_digest": output64_digest,
        "output_float32_digest": output32_digest,
        "transform_decomposition": decomposition,
        "warning_categories": categories,
        "warning_messages_sanitized": messages,
        "worker_code_digest": sha256_file(Path(__file__)),
    }
    manifest["worker_result_digest"] = json_digest(manifest)
    target = Path(manifest_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23 scaler-only deterministic worker")
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--windows-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--manifest-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_worker(
            scaler=args.scaler,
            windows_npz=args.windows_npz,
            output_npz=args.output_npz,
            manifest_out=args.manifest_out,
        )
    except Exception as exc:
        print(f"model_scaler_worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
