"""Deterministic, non-secret model-serving artifact snapshots.

The module reads configuration and model artifacts only.  It never mutates the
process environment, imports a trading entrypoint, fetches market data, or
initializes an exchange adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import torch
try:
    import sklearn
except Exception:  # pragma: no cover - a missing runtime is reported in the snapshot
    sklearn = None  # type: ignore

try:
    from tools.replay_contract import (
        _bool,
        _load_dotenv_memory,
        _load_forced_env,
        _load_run_config,
    )
except ModuleNotFoundError:
    from replay_contract import (  # type: ignore
        _bool,
        _load_dotenv_memory,
        _load_forced_env,
        _load_run_config,
    )

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from runtime.model_serving_guard import evaluate_model_serving_contract
SCHEMA_VERSION = 2
KINDS = ("lstm", "tcn", "tx", "adv")
SOURCE_FILES = {
    "live_writer_sha256": "tools/live_writer.py",
    "dl_ensemble_sha256": "ml_dl/dl_ensemble.py",
    "dl_infer_sha256": "ml_dl/dl_infer.py",
    "dl_models_sha256": "ml_dl/dl_models.py",
    "feature_cols_sha256": "features.py",
}
SNAPSHOT_FIELDS_V1 = (
    "schema_version",
    "identity",
    "mode",
    "git_commit",
    "live_writer_sha256",
    "dl_ensemble_sha256",
    "dl_infer_sha256",
    "dl_models_sha256",
    "feature_cols_sha256",
    "feature_count",
    "dl_symbols",
    "dl_timeframe",
    "dl_seq_len",
    "dl_add_symbol_id",
    "dl_min_agree",
    "dl_model_weights",
    "dl_bias_lstm",
    "dl_bias_tcn",
    "dl_bias_tx",
    "dl_temp_lstm",
    "dl_temp_tcn",
    "dl_temp_tx",
    "model_directory",
    "model_entries",
    "paper_mode",
    "place_real_orders",
)
SNAPSHOT_FIELDS = SNAPSHOT_FIELDS_V1 + (
    "dl_p_long",
    "dl_p_long_mode",
    "dl_allow_only",
    "dl_bias_adv",
    "dl_temp_adv",
    "python_version",
    "numpy_version",
    "torch_version",
    "joblib_version",
    "sklearn_runtime_version",
    "training_serving_contract_status",
    "training_serving_critical_mismatches",
    "training_serving_warnings",
    "model_serving_guard_digest",
    "market_data_exchange",
    "effective_completed_bar_policy",
)
DOCUMENT_FIELDS_V1 = set(SNAPSHOT_FIELDS_V1) | {"generated_at", "snapshot_digest"}
DOCUMENT_FIELDS = set(SNAPSHOT_FIELDS) | {"generated_at", "snapshot_digest"}
MODEL_ENTRY_FIELDS_V1 = {
    "kind",
    "model_filename",
    "model_sha256",
    "scaler_filename",
    "scaler_sha256",
    "metadata_filename",
    "metadata_sha256",
    "metadata_status",
    "metadata_kind",
    "metadata_seq_len",
    "metadata_n_features",
    "metadata_timeframe",
    "metadata_symbols",
    "metadata_val_auc",
    "metadata_trained_at",
    "scaler_class",
    "scaler_n_features_in",
    "scaler_feature_names",
    "scaler_mean_finite",
    "scaler_scale_finite",
    "scaler_zero_scale_count",
    "scaler_near_zero_scale_count",
    "model_load_status",
    "model_parameter_count",
    "model_state_key_count",
}
MODEL_ENTRY_FIELDS = MODEL_ENTRY_FIELDS_V1 | {
    "scaler_load_status",
    "scaler_serialized_sklearn_version",
    "scaler_runtime_sklearn_version",
    "sklearn_version_status",
    "scaler_serialization_warnings",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")


class ModelServingSnapshotError(ValueError):
    """Raised for malformed, unsafe, or internally inconsistent snapshots."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path, *, text: bool = False) -> Optional[str]:
    if not path.is_file():
        return None
    if text:
        value = path.read_text(encoding="utf-8-sig", errors="strict")
        data = value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    else:
        data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10,
        )
        value = result.stdout.strip().lower()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else "unknown"
    except Exception:
        return "unknown"


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        value = float(str(env.get(key, default)).strip())
    except Exception as exc:
        raise ModelServingSnapshotError(f"invalid numeric serving setting: {key}") from exc
    if not math.isfinite(value):
        raise ModelServingSnapshotError(f"non-finite serving setting: {key}")
    return value


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    value = _float(env, key, float(default))
    if not value.is_integer():
        raise ModelServingSnapshotError(f"non-integral serving setting: {key}")
    return int(value)


def _relative_or_filename(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def _first_existing(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_path(root: Path, raw: str, fallbacks: Sequence[Path]) -> Path:
    candidates: list[Path] = []
    if raw.strip():
        candidate = Path(raw.strip())
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    candidates.extend(fallbacks)
    return _first_existing(candidates)


def _feature_inventory(root: Path) -> tuple[int, list[str]]:
    # Importing features is calculation-only, but parse the assignment instead so
    # snapshot capture cannot execute unrelated module-level code.
    import ast

    path = root / "features.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FEATURE_COLS"
                for target in node.targets
            ):
                values = ast.literal_eval(node.value)
                if isinstance(values, list) and all(isinstance(v, str) for v in values):
                    return len(values), list(values)
    except Exception as exc:
        raise ModelServingSnapshotError(f"unable to inspect FEATURE_COLS: {exc}") from exc
    raise ModelServingSnapshotError("FEATURE_COLS list not found")


def _parse_weights(raw: str, kinds: Sequence[str], metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    explicit: dict[str, float] = {}
    for part in raw.split(",") if raw.strip() else ():
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        if key not in kinds:
            continue
        try:
            number = float(value.strip())
        except ValueError:
            continue
        if math.isfinite(number):
            explicit[key] = number
    if explicit:
        # Preserve an explicit operator contract verbatim.  The ensemble applies
        # its normal positive-weight normalization at prediction time.
        return {kind: float(explicit.get(kind, 0.0)) for kind in kinds}
    aucs = {
        kind: max(0.0, float(metadata.get(kind, {}).get("val_auc", 1.0) or 1.0))
        for kind in kinds
    }
    total = sum(aucs.values())
    if total <= 0:
        return {kind: 1.0 / len(kinds) for kind in kinds}
    return {kind: value / total for kind, value in aucs.items()}


def _load_metadata(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}, "malformed"
    return (payload, "loaded") if isinstance(payload, dict) else ({}, "malformed")


def _serialized_sklearn_version(path: Path, warning_messages: Sequence[str]) -> Optional[str]:
    warning_pattern = re.compile(
        r"unpickle estimator .*? from version\s+([0-9]+(?:\.[0-9]+){1,3})",
        re.IGNORECASE,
    )
    for message in warning_messages:
        match = warning_pattern.search(message)
        if match:
            return match.group(1)
    try:
        data = path.read_bytes()
        marker = data.find(b"_sklearn_version")
        if marker >= 0:
            match = re.search(rb"([0-9]+\.[0-9]+(?:\.[0-9]+){0,2})", data[marker:marker + 160])
            if match:
                return match.group(1).decode("ascii")
    except Exception:
        pass
    return None


def _sklearn_version_status(serialized: Optional[str], runtime: Optional[str], loaded: bool) -> str:
    if serialized is None:
        return "serialized_version_unknown"
    if runtime is None:
        return "runtime_version_unknown"
    if serialized == runtime:
        return "exact_match"
    return "loadable_version_mismatch" if loaded else "runtime_version_unknown"


def _scaler_details(path: Path) -> tuple[dict[str, Any], Any]:
    runtime_version = None if sklearn is None else str(getattr(sklearn, "__version__", "") or "") or None
    empty = {
        "scaler_class": None,
        "scaler_n_features_in": None,
        "scaler_feature_names": None,
        "scaler_mean_finite": None,
        "scaler_scale_finite": None,
        "scaler_zero_scale_count": None,
        "scaler_near_zero_scale_count": None,
        "scaler_load_status": "missing" if not path.is_file() else "load_failed",
        "scaler_serialized_sklearn_version": None,
        "scaler_runtime_sklearn_version": runtime_version,
        "sklearn_version_status": "serialized_version_unknown",
        "scaler_serialization_warnings": [],
    }
    if not path.is_file():
        return empty, None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            scaler = joblib.load(path)
        warning_messages = [str(item.message) for item in captured]
        serialized_version = _serialized_sklearn_version(path, warning_messages)
        mean = np.asarray(getattr(scaler, "mean_", []), dtype=float)
        scale = np.asarray(getattr(scaler, "scale_", []), dtype=float)
        names = getattr(scaler, "feature_names_in_", None)
        return {
            "scaler_class": f"{type(scaler).__module__}.{type(scaler).__name__}",
            "scaler_n_features_in": int(getattr(scaler, "n_features_in_", 0) or 0),
            "scaler_feature_names": None if names is None else [str(v) for v in names.tolist()],
            "scaler_mean_finite": bool(mean.size and np.isfinite(mean).all()),
            "scaler_scale_finite": bool(scale.size and np.isfinite(scale).all()),
            "scaler_zero_scale_count": int(np.count_nonzero(scale == 0)) if scale.size else None,
            "scaler_near_zero_scale_count": int(np.count_nonzero(np.abs(scale) < 1e-12)) if scale.size else None,
            "scaler_load_status": "loaded",
            "scaler_serialized_sklearn_version": serialized_version,
            "scaler_runtime_sklearn_version": runtime_version,
            "sklearn_version_status": _sklearn_version_status(
                serialized_version, runtime_version, True
            ),
            "scaler_serialization_warnings": warning_messages,
        }, scaler
    except Exception as exc:
        empty["scaler_serialization_warnings"] = [f"{type(exc).__name__}: scaler_load_failed"]
        serialized_version = _serialized_sklearn_version(path, empty["scaler_serialization_warnings"])
        empty["scaler_serialized_sklearn_version"] = serialized_version
        empty["sklearn_version_status"] = _sklearn_version_status(
            serialized_version, runtime_version, False
        )
        return empty, None


def _model_details(kind: str, model_path: Path, scaler: Any, feature_count: int) -> tuple[dict[str, Any], Any]:
    empty = {
        "model_load_status": "missing" if not model_path.is_file() else "load_failed",
        "model_parameter_count": None,
        "model_state_key_count": None,
    }
    if not model_path.is_file() or scaler is None:
        if scaler is None and model_path.is_file():
            empty["model_load_status"] = "scaler_load_failed"
        return empty, None
    try:
        from ml_dl.dl_infer import _build_model

        width = int(getattr(scaler, "n_features_in_", feature_count) or feature_count)
        model = _build_model(kind, width)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise TypeError("state dict is not a mapping")
        model.load_state_dict(state)
        model.eval().cpu()
        return {
            "model_load_status": "loaded",
            "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "model_state_key_count": int(len(state)),
        }, model
    except Exception as exc:
        empty["model_load_status"] = f"load_failed:{type(exc).__name__}"
        return empty, None


def _artifact_entry(kind: str, root: Path, env: Mapping[str, str], feature_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir_raw = str(env.get("DL_MODEL_DIR", "model_artifacts") or "model_artifacts")
    model_dir = Path(model_dir_raw)
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_path = _resolve_path(
        root,
        str(env.get(f"DL_{kind.upper()}_MODEL_PATH", "")),
        [model_dir / f"dl_{kind}_latest.pt", model_dir / f"dl_{kind}.pt"],
    )
    scaler_path = _resolve_path(
        root,
        str(env.get(f"DL_{kind.upper()}_SCALER_PATH", "")),
        [model_dir / f"scaler_{kind}_latest.joblib", model_dir / "scaler_latest.joblib"],
    )
    metadata_path = _resolve_path(
        root,
        str(env.get(f"DL_{kind.upper()}_METADATA_PATH", "")),
        [model_dir / f"dl_{kind}_metadata.json"],
    )
    metadata, metadata_status = _load_metadata(metadata_path)
    scaler_info, scaler = _scaler_details(scaler_path)
    model_info, model = _model_details(kind, model_path, scaler, feature_count)
    entry: dict[str, Any] = {
        "kind": kind,
        "model_filename": _relative_or_filename(model_path, root),
        "model_sha256": _sha256_file(model_path),
        "scaler_filename": _relative_or_filename(scaler_path, root),
        "scaler_sha256": _sha256_file(scaler_path),
        "metadata_filename": _relative_or_filename(metadata_path, root),
        "metadata_sha256": _sha256_file(metadata_path, text=True),
        "metadata_status": metadata_status,
        "metadata_kind": metadata.get("kind"),
        "metadata_seq_len": metadata.get("seq_len"),
        "metadata_n_features": metadata.get("n_features"),
        "metadata_timeframe": metadata.get("timeframe"),
        "metadata_symbols": metadata.get("symbols"),
        "metadata_val_auc": metadata.get("val_auc"),
        "metadata_trained_at": metadata.get("trained_at"),
        **scaler_info,
        **model_info,
    }
    return entry, {"metadata": metadata, "scaler": scaler, "model": model}


def snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    fields = SNAPSHOT_FIELDS_V1 if snapshot.get("schema_version") == 1 else SNAPSHOT_FIELDS
    return _json_digest({key: snapshot.get(key) for key in fields})


def capture_model_serving_snapshot(
    identity: str = "current_model_serving",
    mode: str = "offline_inventory",
    forced_env_json: Path | str | None = None,
    *,
    base_dir: Path | str = BASE_DIR,
    generated_at: Optional[str] = None,
    forced_env_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    root = Path(base_dir)
    env = _load_run_config(root)
    env.update(_load_dotenv_memory(root / ".env"))
    env.update(_load_forced_env(forced_env_json))
    if forced_env_overrides:
        env.update({str(key): str(value) for key, value in forced_env_overrides.items()})
    feature_count, _feature_names = _feature_inventory(root)
    entries: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        entry, loaded = _artifact_entry(kind, root, env, feature_count)
        # Missing optional ADV artifacts are omitted; the three deployed base
        # members are retained even when broken so failures remain visible.
        if kind == "adv" and entry["model_sha256"] is None and entry["metadata_sha256"] is None:
            continue
        entries.append(entry)
        metadata[kind] = loaded["metadata"]
    active_kinds = [entry["kind"] for entry in entries]
    symbols = [value.strip() for value in str(env.get("DL_SYMBOLS", "BTCUSDT,ETHUSDT")).split(",") if value.strip()]
    add_raw = str(env.get("DL_ADD_SYMBOL_ID", "")).strip()
    add_symbol_id: Optional[bool] = None if not add_raw else _bool(env, "DL_ADD_SYMBOL_ID", False)
    if add_symbol_id is None:
        widths = {entry.get("scaler_n_features_in") for entry in entries
                  if entry.get("scaler_n_features_in") is not None}
        if len(widths) == 1:
            width = int(next(iter(widths)))
            if width in {feature_count, feature_count + 1}:
                add_symbol_id = width == feature_count + 1
    model_dir = Path(str(env.get("DL_MODEL_DIR", "model_artifacts") or "model_artifacts"))
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": str(identity),
        "mode": str(mode),
        "git_commit": _git_commit(root),
        **{field: _sha256_file(root / rel, text=True) for field, rel in SOURCE_FILES.items()},
        "feature_count": feature_count,
        "dl_symbols": symbols,
        "dl_timeframe": str(env.get("DL_TIMEFRAME", "1m") or "1m"),
        "dl_seq_len": _int(env, "DL_SEQ_LEN", 64),
        "dl_add_symbol_id": add_symbol_id,
        "dl_min_agree": _int(env, "DL_MIN_AGREE", 2),
        "dl_p_long": _float(env, "DL_P_LONG", 0.45),
        "dl_p_long_mode": str(env.get("DL_P_LONG_MODE", "abs") or "abs").lower(),
        "dl_allow_only": str(env.get("DL_ALLOW_ONLY", "1") or "1"),
        "dl_model_weights": _parse_weights(str(env.get("DL_MODEL_WEIGHTS", "")), active_kinds, metadata),
        "dl_bias_lstm": _float(env, "DL_BIAS_LSTM", 0.0),
        "dl_bias_tcn": _float(env, "DL_BIAS_TCN", 0.0),
        "dl_bias_tx": _float(env, "DL_BIAS_TX", 0.0),
        "dl_temp_lstm": _float(env, "DL_TEMP_LSTM", 1.0),
        "dl_temp_tcn": _float(env, "DL_TEMP_TCN", 1.0),
        "dl_temp_tx": _float(env, "DL_TEMP_TX", 1.0),
        "dl_bias_adv": _float(env, "DL_BIAS_ADV", 0.0),
        "dl_temp_adv": _float(env, "DL_TEMP_ADV", 1.0),
        "model_directory": _relative_or_filename(model_dir, root),
        "model_entries": entries,
        "paper_mode": bool(
            _bool(env, "PAPER_TRADING", True)
            and not _bool(env, "LIVE_TRADING", False)
            and _bool(env, "EXEC_PAPER", True)
            and not _bool(env, "LIVE_MODE", False)
        ),
        "place_real_orders": _bool(env, "PLACE_REAL_ORDERS", False),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "joblib_version": str(getattr(joblib, "__version__", "unknown")),
        "sklearn_runtime_version": (
            None if sklearn is None else str(getattr(sklearn, "__version__", "unknown"))
        ),
        "market_data_exchange": str(env.get("EXCHANGE_ID", "bitget") or "bitget").lower(),
        "effective_completed_bar_policy": {
            "completed_only": _bool(env, "DL_COMPLETED_ONLY", False),
            "completion_grace_seconds": _int(env, "DL_COMPLETION_GRACE_SECONDS", 5),
            "timeframe": str(env.get("DL_TIMEFRAME", "1m") or "1m"),
        },
        "generated_at": generated_at or _utc_now(),
    }
    generated_width = (
        feature_count + int(add_symbol_id) if isinstance(add_symbol_id, bool) else None
    )
    guard = evaluate_model_serving_contract(
        entries,
        serving_timeframe=snapshot["dl_timeframe"],
        serving_sequence_length=snapshot["dl_seq_len"],
        generated_feature_width=generated_width,
        add_symbol_id=add_symbol_id,
        serving_symbols=symbols,
        base_feature_width=feature_count,
    )
    snapshot.update(
        {
            "training_serving_contract_status": guard["status"],
            "training_serving_critical_mismatches": guard["critical_mismatches"],
            "training_serving_warnings": guard["warnings"],
            "model_serving_guard_digest": guard["guard_digest"],
        }
    )
    validate_model_serving_snapshot(snapshot, require_digest=False)
    snapshot["snapshot_digest"] = snapshot_digest(snapshot)
    return snapshot


def validate_model_serving_snapshot(snapshot: Mapping[str, Any], *, require_digest: bool = True) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise ModelServingSnapshotError("model-serving snapshot must be an object")
    schema = snapshot.get("schema_version")
    if schema not in {1, 2}:
        raise ModelServingSnapshotError("snapshot schema_version must be 1 or 2")
    fields = SNAPSHOT_FIELDS_V1 if schema == 1 else SNAPSHOT_FIELDS
    document_fields = DOCUMENT_FIELDS_V1 if schema == 1 else DOCUMENT_FIELDS
    entry_fields = MODEL_ENTRY_FIELDS_V1 if schema == 1 else MODEL_ENTRY_FIELDS
    missing = set(fields) - set(snapshot)
    unknown = set(snapshot) - document_fields
    if missing or unknown:
        raise ModelServingSnapshotError(
            f"snapshot fields invalid; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if snapshot.get("paper_mode") is not True or snapshot.get("place_real_orders") is not False:
        raise ModelServingSnapshotError("unsafe serving snapshot: paper_mode=true and place_real_orders=false required")
    for field in SOURCE_FILES:
        if not SHA_RE.fullmatch(str(snapshot.get(field) or "")):
            raise ModelServingSnapshotError(f"invalid source digest: {field}")
    entries = snapshot.get("model_entries")
    if not isinstance(entries, list):
        raise ModelServingSnapshotError("model_entries must be a list")
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != entry_fields:
            raise ModelServingSnapshotError("model entry fields are invalid")
        for path_field in ("model_filename", "scaler_filename", "metadata_filename"):
            value = str(entry.get(path_field) or "")
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ModelServingSnapshotError(f"unsafe model entry path: {path_field}")
        if schema == 2:
            if entry.get("sklearn_version_status") not in {
                "exact_match", "loadable_version_mismatch",
                "serialized_version_unknown", "runtime_version_unknown",
            }:
                raise ModelServingSnapshotError("invalid scaler sklearn_version_status")
            scaler_warnings = entry.get("scaler_serialization_warnings")
            if not isinstance(scaler_warnings, list) or not all(
                isinstance(value, str) for value in scaler_warnings
            ):
                raise ModelServingSnapshotError("invalid scaler serialization warnings")
    if schema == 2:
        if snapshot.get("training_serving_contract_status") not in {"pass", "fail", "unverified"}:
            raise ModelServingSnapshotError("invalid training_serving_contract_status")
        if not SHA_RE.fullmatch(str(snapshot.get("model_serving_guard_digest") or "")):
            raise ModelServingSnapshotError("invalid model_serving_guard_digest")
        for field in (
            "training_serving_critical_mismatches", "training_serving_warnings"
        ):
            values = snapshot.get(field)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ModelServingSnapshotError(f"invalid {field}")
        completed_policy = snapshot.get("effective_completed_bar_policy")
        if not isinstance(completed_policy, Mapping) or set(completed_policy) != {
            "completed_only", "completion_grace_seconds", "timeframe"
        }:
            raise ModelServingSnapshotError("invalid effective_completed_bar_policy")
        if not isinstance(completed_policy.get("completed_only"), bool):
            raise ModelServingSnapshotError("invalid completed-only setting")
        grace = completed_policy.get("completion_grace_seconds")
        if isinstance(grace, bool) or not isinstance(grace, int) or grace < 0:
            raise ModelServingSnapshotError("invalid completion grace setting")
    expected = snapshot_digest(snapshot)
    if require_digest and snapshot.get("snapshot_digest") != expected:
        raise ModelServingSnapshotError("model-serving snapshot digest mismatch")
    return {"valid": True, "safety_valid": True, "snapshot_digest": expected}


def write_model_serving_snapshot(snapshot: Mapping[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(snapshot), indent=2), encoding="utf-8")
    return out


def load_model_serving_snapshot(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ModelServingSnapshotError(f"malformed model-serving snapshot: {exc}") from exc
    validate_model_serving_snapshot(value)
    return dict(value)


def resolve_model_serving_snapshot(
    identity: str,
    *,
    reports_dir: Path | str = BASE_DIR / "reports",
    bundle_root: Path | str = BASE_DIR / "reports" / "replay_bundles",
) -> dict[str, Any]:
    if ":" not in identity:
        raise ModelServingSnapshotError("historical snapshot identity must contain mode:timestamp")
    mode, timestamp = identity.rsplit(":", 1)
    if not re.fullmatch(r"[a-z0-9_]+", mode) or not re.fullmatch(r"\d{14}", timestamp):
        raise ModelServingSnapshotError("historical snapshot identity is invalid")
    candidates = [
        Path(reports_dir) / f"matrix_{mode}_{timestamp}_model_serving_snapshot.json",
        Path(bundle_root) / f"{mode}_{timestamp}" / "model_serving_snapshot.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            snapshot = load_model_serving_snapshot(candidate)
            if snapshot.get("identity") != identity:
                raise ModelServingSnapshotError("model-serving snapshot identity mismatch")
            return {"status": "exact_snapshot", "path": str(candidate), "snapshot": snapshot,
                    "digest": snapshot["snapshot_digest"]}
    return {"status": "missing", "path": None, "snapshot": None, "digest": None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a safe model-serving snapshot.")
    parser.add_argument("--identity", default="current_model_serving")
    parser.add_argument("--mode", default="offline_inventory")
    parser.add_argument("--forced-env-json")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument(
        "--paper-safe", action="store_true",
        help="Apply paper-safe values in memory for a read-only preflight.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paper_safe = (
            {
                "LIVE_TRADING": "false", "PAPER_TRADING": "true",
                "LIVE_MODE": "false", "EXEC_PAPER": "true",
                "PLACE_REAL_ORDERS": "false",
            }
            if args.paper_safe or not args.forced_env_json else None
        )
        snapshot = capture_model_serving_snapshot(
            args.identity, args.mode, args.forced_env_json, base_dir=args.base_dir,
            forced_env_overrides=paper_safe,
        )
        if args.json_out:
            write_model_serving_snapshot(snapshot, args.json_out)
        if args.json or not args.json_out:
            print(json.dumps(snapshot, indent=2))
        else:
            print(json.dumps({"status": "exact_snapshot", "snapshot_path": str(args.json_out),
                              "snapshot_digest": snapshot["snapshot_digest"], "paper_only": True}))
        return 0
    except (ModelServingSnapshotError, OSError) as exc:
        print(f"model_serving_snapshot_error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
