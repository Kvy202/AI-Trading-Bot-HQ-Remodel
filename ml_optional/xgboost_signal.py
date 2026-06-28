"""XGBoost signal confirmation in shadow mode.

Phase 3 contract:
* optional and default-off via USE_XGBOOST_SIGNAL
* missing artifacts or missing xgboost dependency never crash the writer
* predictions are logged only; they never block, skip, or modify trades
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

DEFAULT_ARTIFACT = "model_artifacts/xgboost_signal.joblib"
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

XGBOOST_SHADOW_COLS = [
    "timestamp",
    "symbol",
    "xgboost_enabled",
    "xgboost_status",
    "existing_signal",
    "existing_score",
    "xgboost_direction",
    "xgboost_confidence",
    "would_confirm",
    "would_reject",
    "reason",
    "model_version",
    "artifact_path",
]


def _xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def artifact_path_from_env(base_dir: Path | str) -> Path:
    raw = (os.getenv("XGBOOST_SIGNAL_ARTIFACT") or DEFAULT_ARTIFACT).strip()
    path = Path(raw)
    return path if path.is_absolute() else Path(base_dir) / path


def confidence_threshold_from_env(default: float = DEFAULT_CONFIDENCE_THRESHOLD) -> float:
    raw = os.getenv("XGBOOST_CONFIDENCE_THRESHOLD")
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        val = float(raw)
        if math.isfinite(val):
            return min(1.0, max(0.0, val))
    except Exception:
        pass
    return float(default)


def window_to_xgboost_vector(
    window: Any,
    *,
    existing_score: float = 0.0,
    rv_mean: float = 0.0,
    price: float = 0.0,
) -> np.ndarray:
    """Convert a live feature window into a stable 2D vector for XGBoost.

    This is intentionally separate from features.FEATURE_COLS and the deployed
    DL scalers. It summarizes the already-built live feature window without
    changing the model feature contract.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"expected non-empty 2D window, got shape={arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    latest = arr[-1]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    min_v = arr.min(axis=0)
    max_v = arr.max(axis=0)
    context = np.asarray([existing_score, rv_mean, price], dtype=np.float32)
    return np.concatenate([latest, mean, std, min_v, max_v, context]).reshape(1, -1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _class_to_direction(label: Any, fallback: str = "LONG") -> str:
    raw = str(label).strip().upper()
    if raw in {"1", "1.0", "LONG", "BUY", "BULL", "TRUE"}:
        return "LONG"
    if raw in {"0", "0.0", "-1", "-1.0", "SHORT", "SELL", "BEAR", "FALSE"}:
        return "SHORT"
    return fallback


def _normalize_existing_signal(existing_signal: Any, existing_score: float) -> str:
    raw = str(existing_signal or "").strip().upper()
    if raw in {"LONG", "BUY", "BULL"}:
        return "LONG"
    if raw in {"SHORT", "SELL", "BEAR"}:
        return "SHORT"
    score = _safe_float(existing_score, 0.0)
    if score > 0:
        return "LONG"
    if score < 0:
        return "SHORT"
    return "FLAT"


def _predict_direction_confidence(model: Any, x: np.ndarray) -> Tuple[str, float]:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x), dtype=float)
        if probs.ndim == 2:
            probs = probs[0]
        probs = np.nan_to_num(probs.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if probs.size == 0:
            raise ValueError("predict_proba returned no probabilities")

        classes = getattr(model, "classes_", None)
        if classes is not None and len(classes) == probs.size:
            idx = int(np.argmax(probs))
            fallback = "LONG" if idx == probs.size - 1 else "SHORT"
            return _class_to_direction(classes[idx], fallback=fallback), float(probs[idx])

        if probs.size == 1:
            p_long = float(probs[0])
        else:
            p_long = float(probs[-1])
        return ("LONG", p_long) if p_long >= 0.5 else ("SHORT", 1.0 - p_long)

    if hasattr(model, "decision_function"):
        score = float(np.asarray(model.decision_function(x), dtype=float).reshape(-1)[0])
        p_long = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
        return ("LONG", p_long) if p_long >= 0.5 else ("SHORT", 1.0 - p_long)

    if hasattr(model, "predict"):
        pred = np.asarray(model.predict(x)).reshape(-1)[0]
        return _class_to_direction(pred), 1.0

    raise ValueError("model has neither predict_proba, decision_function, nor predict")


@dataclass(frozen=True)
class XGBoostSignalResult:
    xgboost_enabled: bool
    xgboost_status: str
    existing_signal: str
    existing_score: Optional[float]
    xgboost_direction: str
    xgboost_confidence: Optional[float]
    would_confirm: bool
    would_reject: bool
    reason: str
    model_version: str
    artifact_path: str

    def to_log_row(self, timestamp: str, symbol: str) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "xgboost_enabled": int(self.xgboost_enabled),
            "xgboost_status": self.xgboost_status,
            "existing_signal": self.existing_signal,
            "existing_score": "" if self.existing_score is None else float(self.existing_score),
            "xgboost_direction": self.xgboost_direction,
            "xgboost_confidence": "" if self.xgboost_confidence is None else float(self.xgboost_confidence),
            "would_confirm": int(self.would_confirm),
            "would_reject": int(self.would_reject),
            "reason": self.reason,
            "model_version": self.model_version,
            "artifact_path": self.artifact_path,
        }


class XGBoostSignalConfirmer:
    def __init__(
        self,
        *,
        enabled: bool,
        artifact_path: Path,
        model: Any = None,
        model_version: str = "",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        xgboost_status: str = "disabled_flag_false",
        reason: str = "flag_disabled",
    ) -> None:
        self.enabled = bool(enabled)
        self.artifact_path = Path(artifact_path)
        self.model = model
        self.model_version = str(model_version or "")
        self.confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))
        self.xgboost_status = str(xgboost_status)
        self.reason = str(reason)

    @property
    def ready(self) -> bool:
        return self.enabled and self.model is not None and self.xgboost_status == "loaded"

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool,
        base_dir: Path | str,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "XGBoostSignalConfirmer":
        path = artifact_path_from_env(base_dir)
        threshold = confidence_threshold_from_env()
        emit = log_fn or (lambda msg: None)

        if not enabled:
            return cls(enabled=False, artifact_path=path, confidence_threshold=threshold)
        if not path.exists():
            emit(f"xgboost_status=disabled_missing_artifact artifact_path={path}")
            return cls(
                enabled=False,
                artifact_path=path,
                confidence_threshold=threshold,
                xgboost_status="disabled_missing_artifact",
                reason="artifact_missing",
            )
        if not _xgboost_available():
            emit(f"xgboost_status=disabled_missing_dependency dependency=xgboost artifact_path={path}")
            return cls(
                enabled=False,
                artifact_path=path,
                confidence_threshold=threshold,
                xgboost_status="disabled_missing_dependency",
                reason="missing_dependency:xgboost",
            )

        try:
            import joblib

            artifact = joblib.load(path)
            if isinstance(artifact, dict):
                model = artifact.get("model")
                version = artifact.get("model_version") or artifact.get("version") or path.name
                threshold = float(artifact.get("confidence_threshold", threshold))
                threshold = confidence_threshold_from_env(threshold)
            else:
                model = artifact
                version = path.name
            if model is None:
                raise ValueError("artifact does not contain a model")
            emit(
                "xgboost_status=loaded "
                f"artifact_path={path} model_version={version} "
                f"confidence_threshold={threshold:.4f}"
            )
            return cls(
                enabled=True,
                artifact_path=path,
                model=model,
                model_version=str(version),
                confidence_threshold=threshold,
                xgboost_status="loaded",
                reason="loaded",
            )
        except Exception as exc:
            emit(f"xgboost_status=disabled_load_error artifact_path={path} reason={type(exc).__name__}: {exc}")
            return cls(
                enabled=False,
                artifact_path=path,
                confidence_threshold=threshold,
                xgboost_status="disabled_load_error",
                reason=f"load_error:{type(exc).__name__}",
            )

    def evaluate(
        self,
        *,
        symbol: str,
        window: Any,
        existing_signal: Any,
        existing_score: Optional[float],
        rv_mean: float = 0.0,
        price: float = 0.0,
    ) -> XGBoostSignalResult:
        existing_score_f = None if existing_score is None else _safe_float(existing_score, 0.0)
        normalized_signal = _normalize_existing_signal(existing_signal, existing_score_f or 0.0)

        if not self.ready:
            return XGBoostSignalResult(
                xgboost_enabled=self.enabled,
                xgboost_status=self.xgboost_status,
                existing_signal=normalized_signal,
                existing_score=existing_score_f,
                xgboost_direction="",
                xgboost_confidence=None,
                would_confirm=False,
                would_reject=False,
                reason=self.reason,
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )

        try:
            x = window_to_xgboost_vector(
                window,
                existing_score=existing_score_f or 0.0,
                rv_mean=_safe_float(rv_mean, 0.0),
                price=_safe_float(price, 0.0),
            )
            direction, confidence = _predict_direction_confidence(self.model, x)
            confidence = min(1.0, max(0.0, _safe_float(confidence, 0.0)))

            if normalized_signal not in {"LONG", "SHORT"}:
                would_confirm = False
                would_reject = False
                reason = "no_existing_trade_signal"
            elif confidence < self.confidence_threshold:
                would_confirm = False
                would_reject = True
                reason = "low_confidence"
            elif direction != normalized_signal:
                would_confirm = False
                would_reject = True
                reason = "direction_mismatch"
            else:
                would_confirm = True
                would_reject = False
                reason = "confirmed"

            return XGBoostSignalResult(
                xgboost_enabled=True,
                xgboost_status="loaded",
                existing_signal=normalized_signal,
                existing_score=existing_score_f,
                xgboost_direction=direction,
                xgboost_confidence=confidence,
                would_confirm=would_confirm,
                would_reject=would_reject,
                reason=reason,
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
        except Exception as exc:
            return XGBoostSignalResult(
                xgboost_enabled=True,
                xgboost_status="prediction_error",
                existing_signal=normalized_signal,
                existing_score=existing_score_f,
                xgboost_direction="",
                xgboost_confidence=None,
                would_confirm=False,
                would_reject=False,
                reason=f"prediction_error:{type(exc).__name__}",
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
