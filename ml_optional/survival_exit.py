"""Survival Analysis exit timing in shadow mode.

Phase 4 contract:
* optional and default-off via USE_SURVIVAL_EXIT
* missing artifacts or missing survival dependencies never crash the executor
* predictions are logged only; they never close, modify, or manage trades
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

DEFAULT_ARTIFACT = "model_artifacts/survival_exit.joblib"
DEFAULT_RISK_THRESHOLD = 0.60
DEFAULT_RISK_HORIZON_MINUTES = 30.0
SURVIVAL_DEPENDENCY = "lifelines"

SURVIVAL_SHADOW_COLS = [
    "timestamp",
    "symbol",
    "survival_enabled",
    "survival_status",
    "trade_id",
    "side",
    "entry_time",
    "current_age_seconds",
    "current_age_minutes",
    "current_unrealized_pnl",
    "survival_risk_score",
    "estimated_time_to_exit",
    "would_hold",
    "would_exit_early",
    "reason",
    "model_version",
    "artifact_path",
]


def _survival_dependency_available() -> bool:
    return importlib.util.find_spec(SURVIVAL_DEPENDENCY) is not None


def artifact_path_from_env(base_dir: Path | str) -> Path:
    raw = (os.getenv("SURVIVAL_EXIT_ARTIFACT") or DEFAULT_ARTIFACT).strip()
    path = Path(raw)
    return path if path.is_absolute() else Path(base_dir) / path


def risk_threshold_from_env(default: float = DEFAULT_RISK_THRESHOLD) -> float:
    raw = os.getenv("SURVIVAL_EXIT_RISK_THRESHOLD")
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        val = float(raw)
        if math.isfinite(val):
            return min(1.0, max(0.0, val))
    except Exception:
        pass
    return float(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _nullable_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def build_survival_features(
    *,
    side: str,
    current_age_seconds: Optional[float] = None,
    current_unrealized_pnl: Optional[float] = None,
    entry_price: Optional[float] = None,
    current_price: Optional[float] = None,
    qty: Optional[float] = None,
) -> Dict[str, float]:
    """Build model features without touching the signal model feature contract."""
    age_seconds = max(0.0, _safe_float(current_age_seconds, 0.0))
    pnl = _safe_float(current_unrealized_pnl, 0.0)
    entry = _safe_float(entry_price, 0.0)
    price = _safe_float(current_price, 0.0)
    q = _safe_float(qty, 0.0)
    side_norm = str(side or "").strip().lower()
    return {
        "side_long": 1.0 if side_norm == "long" else 0.0,
        "side_short": 1.0 if side_norm == "short" else 0.0,
        "age_minutes": age_seconds / 60.0,
        "age_hours": age_seconds / 3600.0,
        "unrealized_pnl": pnl,
        "abs_unrealized_pnl": abs(pnl),
        "entry_price": entry,
        "current_price": price,
        "price_return": ((price - entry) / entry) if entry > 0.0 and price > 0.0 else 0.0,
        "qty": q,
    }


def _features_as_frame(features: Dict[str, float]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame([features])
    except Exception:
        return np.asarray([[features[k] for k in sorted(features)]], dtype=float)


def _coerce_risk(value: Any) -> float:
    risk = _safe_float(np.asarray(value, dtype=float).reshape(-1)[0], 0.0)
    if risk < 0.0 or risk > 1.0:
        risk = _sigmoid(risk)
    return min(1.0, max(0.0, risk))


def _predict_risk_and_time(model: Any, features: Dict[str, float], horizon_minutes: float) -> Tuple[float, Optional[float]]:
    x = _features_as_frame(features)
    estimated_time: Optional[float] = None

    if hasattr(model, "predict_survival_function"):
        times = [max(0.01, float(horizon_minutes))]
        sf = model.predict_survival_function(x, times=times)
        arr = np.asarray(sf, dtype=float)
        survival_prob = float(arr.reshape(-1)[0]) if arr.size else 1.0
        risk = 1.0 - min(1.0, max(0.0, survival_prob))
        if hasattr(model, "predict_median"):
            try:
                estimated_time = _nullable_float(np.asarray(model.predict_median(x)).reshape(-1)[0])
            except Exception:
                estimated_time = None
        return risk, estimated_time

    if hasattr(model, "predict_partial_hazard"):
        hazard = _safe_float(np.asarray(model.predict_partial_hazard(x)).reshape(-1)[0], 0.0)
        return min(1.0, max(0.0, hazard / (1.0 + abs(hazard)))), estimated_time

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x), dtype=float).reshape(-1)
        if probs.size == 0:
            raise ValueError("predict_proba returned no probabilities")
        return min(1.0, max(0.0, float(probs[-1]))), estimated_time

    if hasattr(model, "predict"):
        return _coerce_risk(model.predict(x)), estimated_time

    raise ValueError("model has no supported survival/risk prediction method")


@dataclass(frozen=True)
class SurvivalExitResult:
    survival_enabled: bool
    survival_status: str
    trade_id: str
    side: str
    entry_time: str
    current_age_seconds: Optional[float]
    current_age_minutes: Optional[float]
    current_unrealized_pnl: Optional[float]
    survival_risk_score: Optional[float]
    estimated_time_to_exit: Optional[float]
    would_hold: bool
    would_exit_early: bool
    reason: str
    model_version: str
    artifact_path: str

    def to_log_row(self, timestamp: str, symbol: str) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "survival_enabled": int(self.survival_enabled),
            "survival_status": self.survival_status,
            "trade_id": self.trade_id,
            "side": self.side,
            "entry_time": self.entry_time,
            "current_age_seconds": "" if self.current_age_seconds is None else float(self.current_age_seconds),
            "current_age_minutes": "" if self.current_age_minutes is None else float(self.current_age_minutes),
            "current_unrealized_pnl": "" if self.current_unrealized_pnl is None else float(self.current_unrealized_pnl),
            "survival_risk_score": "" if self.survival_risk_score is None else float(self.survival_risk_score),
            "estimated_time_to_exit": "" if self.estimated_time_to_exit is None else float(self.estimated_time_to_exit),
            "would_hold": int(self.would_hold),
            "would_exit_early": int(self.would_exit_early),
            "reason": self.reason,
            "model_version": self.model_version,
            "artifact_path": self.artifact_path,
        }


class SurvivalExitModel:
    def __init__(
        self,
        *,
        enabled: bool,
        artifact_path: Path,
        model: Any = None,
        model_version: str = "",
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        risk_horizon_minutes: float = DEFAULT_RISK_HORIZON_MINUTES,
        survival_status: str = "disabled_flag_false",
        reason: str = "flag_disabled",
    ) -> None:
        self.enabled = bool(enabled)
        self.artifact_path = Path(artifact_path)
        self.model = model
        self.model_version = str(model_version or "")
        self.risk_threshold = min(1.0, max(0.0, float(risk_threshold)))
        self.risk_horizon_minutes = max(0.01, float(risk_horizon_minutes))
        self.survival_status = str(survival_status)
        self.reason = str(reason)

    @property
    def ready(self) -> bool:
        return self.enabled and self.model is not None and self.survival_status == "loaded"

    @classmethod
    def from_env(
        cls,
        *,
        enabled: bool,
        base_dir: Path | str,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "SurvivalExitModel":
        path = artifact_path_from_env(base_dir)
        threshold = risk_threshold_from_env()
        emit = log_fn or (lambda msg: None)

        if not enabled:
            return cls(enabled=False, artifact_path=path, risk_threshold=threshold)
        if not path.exists():
            emit(f"survival_status=disabled_missing_artifact artifact_path={path}")
            return cls(
                enabled=False,
                artifact_path=path,
                risk_threshold=threshold,
                survival_status="disabled_missing_artifact",
                reason="artifact_missing",
            )
        if not _survival_dependency_available():
            emit(
                "survival_status=disabled_missing_dependency "
                f"dependency={SURVIVAL_DEPENDENCY} artifact_path={path}"
            )
            return cls(
                enabled=False,
                artifact_path=path,
                risk_threshold=threshold,
                survival_status="disabled_missing_dependency",
                reason=f"missing_dependency:{SURVIVAL_DEPENDENCY}",
            )

        try:
            import joblib

            artifact = joblib.load(path)
            if isinstance(artifact, dict):
                model = artifact.get("model")
                version = artifact.get("model_version") or artifact.get("version") or path.name
                threshold = risk_threshold_from_env(float(artifact.get("risk_threshold", threshold)))
                horizon = float(artifact.get("risk_horizon_minutes", DEFAULT_RISK_HORIZON_MINUTES))
            else:
                model = artifact
                version = path.name
                horizon = DEFAULT_RISK_HORIZON_MINUTES
            if model is None:
                raise ValueError("artifact does not contain a model")
            emit(
                "survival_status=loaded "
                f"artifact_path={path} model_version={version} "
                f"risk_threshold={threshold:.4f}"
            )
            return cls(
                enabled=True,
                artifact_path=path,
                model=model,
                model_version=str(version),
                risk_threshold=threshold,
                risk_horizon_minutes=horizon,
                survival_status="loaded",
                reason="loaded",
            )
        except Exception as exc:
            emit(f"survival_status=disabled_load_error artifact_path={path} reason={type(exc).__name__}: {exc}")
            return cls(
                enabled=False,
                artifact_path=path,
                risk_threshold=threshold,
                survival_status="disabled_load_error",
                reason=f"load_error:{type(exc).__name__}",
            )

    def evaluate(
        self,
        *,
        symbol: str,
        side: str,
        trade_id: str = "",
        entry_time: str = "",
        current_age_seconds: Optional[float] = None,
        current_unrealized_pnl: Optional[float] = None,
        entry_price: Optional[float] = None,
        current_price: Optional[float] = None,
        qty: Optional[float] = None,
    ) -> SurvivalExitResult:
        age_seconds = _nullable_float(current_age_seconds)
        age_minutes = None if age_seconds is None else max(0.0, age_seconds) / 60.0
        pnl = _nullable_float(current_unrealized_pnl)
        side_norm = str(side or "").strip().lower()

        if not self.ready:
            return SurvivalExitResult(
                survival_enabled=self.enabled,
                survival_status=self.survival_status,
                trade_id=str(trade_id or ""),
                side=side_norm,
                entry_time=str(entry_time or ""),
                current_age_seconds=age_seconds,
                current_age_minutes=age_minutes,
                current_unrealized_pnl=pnl,
                survival_risk_score=None,
                estimated_time_to_exit=None,
                would_hold=True,
                would_exit_early=False,
                reason=self.reason,
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )

        try:
            features = build_survival_features(
                side=side_norm,
                current_age_seconds=age_seconds,
                current_unrealized_pnl=pnl,
                entry_price=entry_price,
                current_price=current_price,
                qty=qty,
            )
            risk, estimated_time = _predict_risk_and_time(
                self.model,
                features,
                self.risk_horizon_minutes,
            )
            would_exit = risk >= self.risk_threshold
            return SurvivalExitResult(
                survival_enabled=True,
                survival_status="loaded",
                trade_id=str(trade_id or ""),
                side=side_norm,
                entry_time=str(entry_time or ""),
                current_age_seconds=age_seconds,
                current_age_minutes=age_minutes,
                current_unrealized_pnl=pnl,
                survival_risk_score=risk,
                estimated_time_to_exit=estimated_time,
                would_hold=not would_exit,
                would_exit_early=would_exit,
                reason="high_exit_risk" if would_exit else "hold_risk_below_threshold",
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
        except Exception as exc:
            return SurvivalExitResult(
                survival_enabled=True,
                survival_status="prediction_error",
                trade_id=str(trade_id or ""),
                side=side_norm,
                entry_time=str(entry_time or ""),
                current_age_seconds=age_seconds,
                current_age_minutes=age_minutes,
                current_unrealized_pnl=pnl,
                survival_risk_score=None,
                estimated_time_to_exit=None,
                would_hold=True,
                would_exit_early=False,
                reason=f"prediction_error:{type(exc).__name__}",
                model_version=self.model_version,
                artifact_path=str(self.artifact_path),
            )
