"""Isolation Forest shadow-mode anomaly filter.

Phase 2 contract:
* optional and default-off via USE_ISOLATION_FOREST
* missing artifacts never crash the writer
* Phase 6 blocking only activates when ISOLATION_FOREST_BLOCKING is also true
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

DEFAULT_ARTIFACT = "model_artifacts/isolation_forest.joblib"
SCORE_THRESHOLD_ENV = "ISOLATION_FOREST_SCORE_THRESHOLD"

_TRUE = {"1", "true", "yes", "y", "on"}

ISOLATION_SHADOW_COLS = [
    "ts",
    "timestamp",
    "symbol",
    "isolation_enabled",
    "isolation_status",
    "anomaly_status",
    "anomaly_score",
    "would_block",
    "actually_blocked",
    "reason",
    "model_version",
    "artifact_path",
]


def artifact_path_from_env(base_dir: Path | str) -> Path:
    raw = (os.getenv("ISOLATION_FOREST_ARTIFACT") or DEFAULT_ARTIFACT).strip()
    path = Path(raw)
    return path if path.is_absolute() else Path(base_dir) / path


def isolation_blocking_from_env(default: bool = False) -> bool:
    raw = os.getenv("ISOLATION_FOREST_BLOCKING")
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in _TRUE


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    except Exception:
        return None


def isolation_score_threshold_from_env(default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(SCORE_THRESHOLD_ENV)
    if raw is None or raw.strip() == "":
        return default
    parsed = _finite_float_or_none(raw)
    return default if parsed is None else parsed


def window_to_isolation_vector(window: Any) -> np.ndarray:
    """Convert a sequence window into a stable 2D vector for Isolation Forest."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"expected non-empty 2D window, got shape={arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    latest = arr[-1]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    max_abs = np.max(np.abs(arr), axis=0)
    return np.concatenate([latest, mean, std, max_abs]).reshape(1, -1)


@dataclass(frozen=True)
class IsolationResult:
    isolation_enabled: bool
    isolation_status: str
    anomaly_status: str
    anomaly_score: Optional[float]
    would_block: bool
    reason: str
    model_version: str
    artifact_path: str

    def to_log_row(self, ts: str, symbol: str, actually_blocked: bool = False) -> Dict[str, Any]:
        reason = "isolation_forest_block" if actually_blocked else self.reason
        return {
            "ts": ts,
            "timestamp": ts,
            "symbol": symbol,
            "isolation_enabled": int(self.isolation_enabled),
            "isolation_status": self.isolation_status,
            "anomaly_status": self.anomaly_status,
            "anomaly_score": "" if self.anomaly_score is None else float(self.anomaly_score),
            "would_block": int(self.would_block),
            "actually_blocked": int(bool(actually_blocked)),
            "reason": reason,
            "model_version": self.model_version,
            "artifact_path": self.artifact_path,
        }


class IsolationFilter:
    def __init__(
        self,
        *,
        enabled: bool,
        artifact_path: Path,
        model: Any = None,
        model_version: str = "",
        isolation_status: str = "disabled_flag_false",
        reason: str = "flag_disabled",
        score_threshold: Optional[float] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.artifact_path = Path(artifact_path)
        self.model = model
        self.model_version = str(model_version or "")
        self.isolation_status = str(isolation_status)
        self.reason = str(reason)
        self.score_threshold = _finite_float_or_none(score_threshold)

    @property
    def ready(self) -> bool:
        return self.enabled and self.model is not None and self.isolation_status == "loaded"

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool,
        base_dir: Path | str,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "IsolationFilter":
        path = artifact_path_from_env(base_dir)
        emit = log_fn or (lambda msg: None)
        score_threshold = isolation_score_threshold_from_env()
        raw_threshold = os.getenv(SCORE_THRESHOLD_ENV)
        if raw_threshold is not None and raw_threshold.strip() and score_threshold is None:
            emit(f"isolation_score_threshold=invalid raw={raw_threshold!r} using=unset")
        elif score_threshold is not None:
            emit(f"isolation_score_threshold={score_threshold}")
        if not enabled:
            return cls(enabled=False, artifact_path=path, score_threshold=score_threshold)
        if not path.exists():
            emit(f"isolation_status=disabled_missing_artifact artifact_path={path}")
            return cls(
                enabled=False,
                artifact_path=path,
                isolation_status="disabled_missing_artifact",
                reason="artifact_missing",
                score_threshold=score_threshold,
            )
        try:
            import joblib

            artifact = joblib.load(path)
            if isinstance(artifact, dict):
                model = artifact.get("model")
                version = artifact.get("model_version") or artifact.get("version") or path.name
            else:
                model = artifact
                version = path.name
            if model is None:
                raise ValueError("artifact does not contain a model")
            emit(f"isolation_status=loaded artifact_path={path} model_version={version}")
            return cls(
                enabled=True,
                artifact_path=path,
                model=model,
                model_version=str(version),
                isolation_status="loaded",
                reason="loaded",
                score_threshold=score_threshold,
            )
        except Exception as exc:
            emit(f"isolation_status=disabled_load_error artifact_path={path} reason={type(exc).__name__}: {exc}")
            return cls(
                enabled=False,
                artifact_path=path,
                isolation_status="disabled_load_error",
                reason=f"load_error:{type(exc).__name__}",
                score_threshold=score_threshold,
            )

    def evaluate(self, symbol: str, window: Any) -> IsolationResult:
        if not self.ready:
            return IsolationResult(
                isolation_enabled=self.enabled,
                isolation_status=self.isolation_status,
                anomaly_status="not_run",
                anomaly_score=None,
                would_block=False,
                reason=self.reason,
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
        if window is None:
            return IsolationResult(
                isolation_enabled=True,
                isolation_status="no_window",
                anomaly_status="not_run",
                anomaly_score=None,
                would_block=False,
                reason="no_window",
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
        try:
            x = window_to_isolation_vector(window)
            score: Optional[float] = None
            if hasattr(self.model, "decision_function"):
                score = float(np.asarray(self.model.decision_function(x)).reshape(-1)[0])
            elif hasattr(self.model, "score_samples"):
                score = float(np.asarray(self.model.score_samples(x)).reshape(-1)[0])

            if hasattr(self.model, "predict"):
                pred = int(np.asarray(self.model.predict(x)).reshape(-1)[0])
                is_anomaly = pred == -1
            elif score is not None:
                is_anomaly = score < 0.0
            else:
                raise ValueError("model has neither predict nor scoring method")

            would_block = bool(is_anomaly)
            if self.score_threshold is not None:
                would_block = bool(is_anomaly and score is not None and score <= self.score_threshold)

            if not is_anomaly:
                reason = "normal_market"
            elif would_block:
                reason = "isolation_anomaly"
            elif score is None:
                reason = "isolation_anomaly_missing_score_threshold"
            else:
                reason = "isolation_anomaly_threshold_allow"

            return IsolationResult(
                isolation_enabled=True,
                isolation_status="loaded",
                anomaly_status="anomaly" if is_anomaly else "normal",
                anomaly_score=score,
                would_block=would_block,
                reason=reason,
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
        except Exception as exc:
            return IsolationResult(
                isolation_enabled=True,
                isolation_status="model_error",
                anomaly_status="error",
                anomaly_score=None,
                would_block=False,
                reason=f"model_error:{type(exc).__name__}",
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )


def should_block_entry(result: IsolationResult, blocking_enabled: bool) -> bool:
    """Return whether Phase 6 should block a new entry.

    This helper is intentionally strict: only a loaded model's anomaly result can
    block, and only when the explicit blocking flag is true.
    """
    return bool(
        blocking_enabled
        and result.isolation_enabled
        and result.isolation_status == "loaded"
        and result.anomaly_status == "anomaly"
        and result.would_block
    )
