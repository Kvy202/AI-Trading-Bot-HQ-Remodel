"""Advanced Risk shadow evaluator.

Phase 10 contract:
* optional and default-off via USE_ADVANCED_RISK
* shadow-only, even when ADVANCED_RISK_ACTIVE=true
* pure evaluator: no file, environment, order, or executor side effects
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_MAX_DAILY_LOSS_PCT = 3.0
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3
DEFAULT_MAX_OPEN_POSITIONS = 1
DEFAULT_MAX_SYMBOL_EXPOSURE_PCT = 100.0
DEFAULT_VOLATILITY_GUARD_MULT = 2.0

ADVANCED_RISK_SHADOW_COLS = [
    "timestamp",
    "symbol",
    "side",
    "p_meta",
    "price",
    "advanced_risk_enabled",
    "advanced_risk_active",
    "risk_status",
    "risk_score",
    "would_block",
    "actually_blocked",
    "would_reduce_size",
    "actually_reduced",
    "would_pause",
    "actually_paused",
    "reasons",
    "top_reason",
    "daily_loss_pct",
    "consecutive_losses",
    "open_positions_count",
    "symbol_exposure",
    "volatility_guard_triggered",
    "paper_only_guard",
]

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in _TRUE


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


@dataclass(frozen=True)
class AdvancedRiskSettings:
    advanced_risk_enabled: bool = False
    advanced_risk_active: bool = False
    max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT
    max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    max_symbol_exposure_pct: float = DEFAULT_MAX_SYMBOL_EXPOSURE_PCT
    volatility_guard_mult: float = DEFAULT_VOLATILITY_GUARD_MULT
    paper_mode: bool = True
    place_real_orders: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        enabled: Optional[bool] = None,
        paper_mode: bool = True,
        place_real_orders: bool = False,
    ) -> "AdvancedRiskSettings":
        return cls(
            advanced_risk_enabled=(
                _env_bool("USE_ADVANCED_RISK", False) if enabled is None else bool(enabled)
            ),
            advanced_risk_active=_env_bool("ADVANCED_RISK_ACTIVE", False),
            max_daily_loss_pct=max(0.0, _env_float("ADVANCED_RISK_MAX_DAILY_LOSS_PCT", DEFAULT_MAX_DAILY_LOSS_PCT)),
            max_consecutive_losses=max(
                0,
                _env_int("ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES", DEFAULT_MAX_CONSECUTIVE_LOSSES),
            ),
            max_open_positions=max(0, _env_int("ADVANCED_RISK_MAX_OPEN_POSITIONS", DEFAULT_MAX_OPEN_POSITIONS)),
            max_symbol_exposure_pct=max(
                0.0,
                _env_float("ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT", DEFAULT_MAX_SYMBOL_EXPOSURE_PCT),
            ),
            volatility_guard_mult=max(
                0.0,
                _env_float("ADVANCED_RISK_VOLATILITY_GUARD_MULT", DEFAULT_VOLATILITY_GUARD_MULT),
            ),
            paper_mode=bool(paper_mode),
            place_real_orders=bool(place_real_orders),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AdvancedRiskSettings":
        return cls(
            advanced_risk_enabled=_coerce_bool(
                data.get("advanced_risk_enabled", data.get("USE_ADVANCED_RISK")),
                False,
            ),
            advanced_risk_active=_coerce_bool(
                data.get("advanced_risk_active", data.get("ADVANCED_RISK_ACTIVE")),
                False,
            ),
            max_daily_loss_pct=max(
                0.0,
                _safe_float(
                    data.get("max_daily_loss_pct", data.get("ADVANCED_RISK_MAX_DAILY_LOSS_PCT")),
                    DEFAULT_MAX_DAILY_LOSS_PCT,
                ),
            ),
            max_consecutive_losses=max(
                0,
                _safe_int(
                    data.get(
                        "max_consecutive_losses",
                        data.get("ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES"),
                    ),
                    DEFAULT_MAX_CONSECUTIVE_LOSSES,
                ),
            ),
            max_open_positions=max(
                0,
                _safe_int(
                    data.get("max_open_positions", data.get("ADVANCED_RISK_MAX_OPEN_POSITIONS")),
                    DEFAULT_MAX_OPEN_POSITIONS,
                ),
            ),
            max_symbol_exposure_pct=max(
                0.0,
                _safe_float(
                    data.get(
                        "max_symbol_exposure_pct",
                        data.get("ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT"),
                    ),
                    DEFAULT_MAX_SYMBOL_EXPOSURE_PCT,
                ),
            ),
            volatility_guard_mult=max(
                0.0,
                _safe_float(
                    data.get(
                        "volatility_guard_mult",
                        data.get("ADVANCED_RISK_VOLATILITY_GUARD_MULT"),
                    ),
                    DEFAULT_VOLATILITY_GUARD_MULT,
                ),
            ),
            paper_mode=_coerce_bool(data.get("paper_mode"), True),
            place_real_orders=_coerce_bool(data.get("place_real_orders"), False),
        )


@dataclass(frozen=True)
class AdvancedRiskContext:
    symbol: str = ""
    side: str = ""
    p_meta: Optional[float] = None
    price: Optional[float] = None
    rv_mean: Optional[float] = None
    volatility: Optional[float] = None
    open_positions_count: Optional[int] = None
    symbol_exposure: Optional[float] = None
    daily_realized_pnl: Optional[float] = None
    daily_loss_pct: Optional[float] = None
    consecutive_losses: Optional[int] = None
    recent_trade_count: Optional[int] = None
    timestamp: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AdvancedRiskContext":
        score = data.get("p_meta", data.get("score"))
        return cls(
            symbol=str(data.get("symbol") or "").strip(),
            side=str(data.get("side") or "").strip().lower(),
            p_meta=_optional_float(score),
            price=_optional_float(data.get("price", data.get("current_price"))),
            rv_mean=_optional_float(data.get("rv_mean")),
            volatility=_optional_float(data.get("volatility")),
            open_positions_count=(
                None
                if data.get("open_positions_count") is None
                else _safe_int(data.get("open_positions_count"), 0)
            ),
            symbol_exposure=_optional_float(data.get("symbol_exposure")),
            daily_realized_pnl=_optional_float(data.get("daily_realized_pnl")),
            daily_loss_pct=_optional_float(data.get("daily_loss_pct")),
            consecutive_losses=(
                None
                if data.get("consecutive_losses") is None
                else _safe_int(data.get("consecutive_losses"), 0)
            ),
            recent_trade_count=(
                None
                if data.get("recent_trade_count") is None
                else _safe_int(data.get("recent_trade_count"), 0)
            ),
            timestamp=str(data.get("timestamp") or ""),
        )


@dataclass(frozen=True)
class AdvancedRiskDecision:
    advanced_risk_enabled: bool
    advanced_risk_active: bool
    risk_status: str
    risk_score: float
    would_block: bool
    actually_blocked: bool
    would_reduce_size: bool
    actually_reduced: bool
    would_pause: bool
    actually_paused: bool
    reasons: tuple[str, ...]
    top_reason: str
    paper_only_guard: str
    volatility_guard_triggered: bool

    def to_log_row(self, context: AdvancedRiskContext | Mapping[str, Any]) -> dict[str, Any]:
        ctx = context if isinstance(context, AdvancedRiskContext) else AdvancedRiskContext.from_mapping(context)
        return {
            "timestamp": ctx.timestamp,
            "symbol": ctx.symbol,
            "side": ctx.side,
            "p_meta": "" if ctx.p_meta is None else float(ctx.p_meta),
            "price": "" if ctx.price is None else float(ctx.price),
            "advanced_risk_enabled": int(self.advanced_risk_enabled),
            "advanced_risk_active": int(self.advanced_risk_active),
            "risk_status": self.risk_status,
            "risk_score": float(self.risk_score),
            "would_block": int(self.would_block),
            "actually_blocked": int(self.actually_blocked),
            "would_reduce_size": int(self.would_reduce_size),
            "actually_reduced": int(self.actually_reduced),
            "would_pause": int(self.would_pause),
            "actually_paused": int(self.actually_paused),
            "reasons": "|".join(self.reasons),
            "top_reason": self.top_reason,
            "daily_loss_pct": "" if ctx.daily_loss_pct is None else float(ctx.daily_loss_pct),
            "consecutive_losses": "" if ctx.consecutive_losses is None else int(ctx.consecutive_losses),
            "open_positions_count": (
                "" if ctx.open_positions_count is None else int(ctx.open_positions_count)
            ),
            "symbol_exposure": "" if ctx.symbol_exposure is None else float(ctx.symbol_exposure),
            "volatility_guard_triggered": int(self.volatility_guard_triggered),
            "paper_only_guard": self.paper_only_guard,
        }


def _decision(
    settings: AdvancedRiskSettings,
    *,
    risk_status: str,
    risk_score: float = 0.0,
    would_block: bool = False,
    would_reduce_size: bool = False,
    would_pause: bool = False,
    reasons: list[str] | tuple[str, ...] = (),
    volatility_guard_triggered: bool = False,
) -> AdvancedRiskDecision:
    reason_tuple = tuple(reasons)
    top_reason = reason_tuple[0] if reason_tuple else risk_status
    if not settings.advanced_risk_active:
        paper_only_guard = "inactive"
    elif settings.place_real_orders:
        paper_only_guard = "blocked_real_orders"
    elif not settings.paper_mode:
        paper_only_guard = "blocked_not_paper"
    else:
        paper_only_guard = "phase10_shadow_only"

    return AdvancedRiskDecision(
        advanced_risk_enabled=settings.advanced_risk_enabled,
        advanced_risk_active=settings.advanced_risk_active,
        risk_status=risk_status,
        risk_score=max(0.0, min(1.0, float(risk_score))),
        would_block=bool(would_block),
        actually_blocked=False,
        would_reduce_size=bool(would_reduce_size),
        actually_reduced=False,
        would_pause=bool(would_pause),
        actually_paused=False,
        reasons=reason_tuple,
        top_reason=top_reason,
        paper_only_guard=paper_only_guard,
        volatility_guard_triggered=bool(volatility_guard_triggered),
    )


def evaluate_advanced_risk(
    context: AdvancedRiskContext | Mapping[str, Any],
    settings: AdvancedRiskSettings | Mapping[str, Any],
) -> AdvancedRiskDecision:
    """Evaluate shadow risk flags for one candidate decision.

    The returned ``actually_*`` fields are always false in Phase 10. Callers may
    log the decision, but must not use it to alter trading behavior yet.
    """
    ctx = context if isinstance(context, AdvancedRiskContext) else AdvancedRiskContext.from_mapping(context)
    cfg = settings if isinstance(settings, AdvancedRiskSettings) else AdvancedRiskSettings.from_mapping(settings)

    if not cfg.advanced_risk_enabled:
        return _decision(cfg, risk_status="disabled", reasons=("advanced_risk_disabled",))

    missing: list[str] = []
    if not ctx.symbol:
        missing.append("symbol")
    if not ctx.side:
        missing.append("side")
    if ctx.p_meta is None:
        missing.append("p_meta")
    if ctx.price is None or ctx.price <= 0.0:
        missing.append("price")
    if ctx.open_positions_count is None:
        missing.append("open_positions_count")
    if ctx.symbol_exposure is None:
        missing.append("symbol_exposure")
    if ctx.daily_loss_pct is None:
        missing.append("daily_loss_pct")
    if ctx.consecutive_losses is None:
        missing.append("consecutive_losses")

    if missing:
        reasons = tuple(f"context_missing:{name}" for name in missing)
        return _decision(cfg, risk_status="context_missing", reasons=reasons)

    reasons: list[str] = []
    weights: list[float] = []
    would_pause = False
    would_reduce_size = False
    volatility_guard_triggered = False

    daily_loss_pct = abs(_safe_float(ctx.daily_loss_pct, 0.0))
    consecutive_losses = max(0, _safe_int(ctx.consecutive_losses, 0))
    open_positions_count = max(0, _safe_int(ctx.open_positions_count, 0))
    symbol_exposure = max(0.0, _safe_float(ctx.symbol_exposure, 0.0))

    if cfg.max_daily_loss_pct > 0.0 and daily_loss_pct >= cfg.max_daily_loss_pct:
        reasons.append("daily_loss_pct_limit")
        weights.append(1.0)
        would_pause = True

    if cfg.max_consecutive_losses > 0 and consecutive_losses >= cfg.max_consecutive_losses:
        reasons.append("consecutive_losses_limit")
        weights.append(0.85)
        would_pause = True

    if cfg.max_open_positions > 0 and open_positions_count >= cfg.max_open_positions:
        reasons.append("max_open_positions_limit")
        weights.append(0.70)

    if cfg.max_symbol_exposure_pct > 0.0 and symbol_exposure >= cfg.max_symbol_exposure_pct:
        reasons.append("symbol_exposure_limit")
        weights.append(0.65)
        would_reduce_size = True

    rv_mean = ctx.rv_mean
    volatility = ctx.volatility
    if (
        rv_mean is not None
        and volatility is not None
        and volatility > 0.0
        and cfg.volatility_guard_mult > 0.0
        and abs(rv_mean) >= cfg.volatility_guard_mult * abs(volatility)
    ):
        volatility_guard_triggered = True
        reasons.append("volatility_guard")
        weights.append(0.75)
        would_reduce_size = True

    would_block = bool(reasons)
    risk_score = max(weights) if weights else 0.0
    return _decision(
        cfg,
        risk_status="would_block" if would_block else "normal",
        risk_score=risk_score,
        would_block=would_block,
        would_reduce_size=would_reduce_size,
        would_pause=would_pause,
        reasons=reasons or ("normal",),
        volatility_guard_triggered=volatility_guard_triggered,
    )
