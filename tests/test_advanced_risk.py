"""Tests for the Phase 10 Advanced Risk shadow evaluator."""

from __future__ import annotations

import csv
from pathlib import Path

import tools.live_executor as le
from ml_optional.advanced_risk import (
    ADVANCED_RISK_SHADOW_COLS,
    AdvancedRiskSettings,
    evaluate_advanced_risk,
)
from runtime.loader import apply_run_config

ROOT = Path(__file__).resolve().parents[1]


def _settings(**overrides):
    values = {
        "advanced_risk_enabled": True,
        "advanced_risk_active": False,
        "max_daily_loss_pct": 3.0,
        "max_consecutive_losses": 3,
        "max_open_positions": 1,
        "max_symbol_exposure_pct": 100.0,
        "volatility_guard_mult": 2.0,
        "paper_mode": True,
        "place_real_orders": False,
    }
    values.update(overrides)
    return AdvancedRiskSettings(**values)


def _context(**overrides):
    values = {
        "timestamp": "2026-07-03 00:00:00+0000",
        "symbol": "BTCUSDT",
        "side": "long",
        "p_meta": 0.70,
        "price": 100.0,
        "rv_mean": 0.010,
        "volatility": 0.020,
        "open_positions_count": 0,
        "symbol_exposure": 10.0,
        "daily_realized_pnl": 0.0,
        "daily_loss_pct": 0.0,
        "consecutive_losses": 0,
        "recent_trade_count": 0,
    }
    values.update(overrides)
    return values


def test_config_defaults_advanced_risk_false(monkeypatch):
    names = [
        "USE_ADVANCED_RISK",
        "ADVANCED_RISK_ACTIVE",
        "ADVANCED_RISK_MAX_DAILY_LOSS_PCT",
        "ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES",
        "ADVANCED_RISK_MAX_OPEN_POSITIONS",
        "ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT",
        "ADVANCED_RISK_VOLATILITY_GUARD_MULT",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)

    try:
        loaded = apply_run_config(ROOT)
        settings = AdvancedRiskSettings.from_env(enabled=False)

        assert loaded["USE_ADVANCED_RISK"] == "False"
        assert loaded["ADVANCED_RISK_ACTIVE"] == "False"
        assert loaded["ADVANCED_RISK_MAX_DAILY_LOSS_PCT"] == "3.0"
        assert loaded["ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES"] == "3"
        assert loaded["ADVANCED_RISK_MAX_OPEN_POSITIONS"] == "1"
        assert loaded["ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT"] == "100.0"
        assert loaded["ADVANCED_RISK_VOLATILITY_GUARD_MULT"] == "2.0"
        assert settings.advanced_risk_enabled is False
        assert settings.advanced_risk_active is False
    finally:
        for name in names:
            monkeypatch.delenv(name, raising=False)


def test_evaluator_disabled_case():
    decision = evaluate_advanced_risk(_context(), _settings(advanced_risk_enabled=False))

    assert decision.risk_status == "disabled"
    assert decision.would_block is False
    assert decision.actually_blocked is False
    assert decision.top_reason == "advanced_risk_disabled"


def test_evaluator_normal_risk_case():
    decision = evaluate_advanced_risk(_context(), _settings())

    assert decision.risk_status == "normal"
    assert decision.risk_score == 0.0
    assert decision.would_block is False
    assert decision.actually_blocked is False
    assert decision.would_pause is False
    assert decision.would_reduce_size is False


def test_evaluator_accepts_env_style_settings_mapping():
    decision = evaluate_advanced_risk(
        _context(daily_loss_pct=3.5),
        {
            "USE_ADVANCED_RISK": "true",
            "ADVANCED_RISK_ACTIVE": "true",
            "ADVANCED_RISK_MAX_DAILY_LOSS_PCT": "3.0",
            "ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES": "3",
            "ADVANCED_RISK_MAX_OPEN_POSITIONS": "1",
            "ADVANCED_RISK_MAX_SYMBOL_EXPOSURE_PCT": "100.0",
            "ADVANCED_RISK_VOLATILITY_GUARD_MULT": "2.0",
            "paper_mode": "true",
            "place_real_orders": "false",
        },
    )

    assert decision.would_block is True
    assert decision.actually_blocked is False
    assert decision.paper_only_guard == "phase10_shadow_only"


def test_evaluator_daily_loss_risk_case():
    decision = evaluate_advanced_risk(_context(daily_loss_pct=3.5), _settings())

    assert decision.risk_status == "would_block"
    assert decision.would_block is True
    assert decision.would_pause is True
    assert decision.actually_blocked is False
    assert decision.actually_paused is False
    assert decision.top_reason == "daily_loss_pct_limit"


def test_evaluator_consecutive_loss_risk_case():
    decision = evaluate_advanced_risk(_context(consecutive_losses=3), _settings())

    assert decision.would_block is True
    assert decision.would_pause is True
    assert decision.top_reason == "consecutive_losses_limit"


def test_evaluator_max_open_positions_risk_case():
    decision = evaluate_advanced_risk(_context(open_positions_count=1), _settings())

    assert decision.would_block is True
    assert decision.top_reason == "max_open_positions_limit"
    assert decision.actually_blocked is False


def test_evaluator_volatility_guard_risk_case():
    decision = evaluate_advanced_risk(_context(rv_mean=0.050, volatility=0.020), _settings())

    assert decision.would_block is True
    assert decision.would_reduce_size is True
    assert decision.volatility_guard_triggered is True
    assert decision.top_reason == "volatility_guard"
    assert decision.actually_reduced is False


def test_active_flag_does_not_affect_trades_in_phase_10():
    decision = evaluate_advanced_risk(
        _context(daily_loss_pct=4.0),
        _settings(advanced_risk_active=True),
    )

    assert decision.would_block is True
    assert decision.actually_blocked is False
    assert decision.would_pause is True
    assert decision.actually_paused is False
    assert decision.paper_only_guard == "phase10_shadow_only"


def test_missing_context_returns_context_missing_without_crashing():
    decision = evaluate_advanced_risk({"timestamp": "t"}, _settings())

    assert decision.risk_status == "context_missing"
    assert decision.would_block is False
    assert decision.actually_blocked is False
    assert any(reason.startswith("context_missing:") for reason in decision.reasons)


def test_shadow_log_schema_and_executor_append(monkeypatch, tmp_path):
    path = tmp_path / "advanced_risk_shadow.csv"
    monkeypatch.setattr(le, "ADVANCED_RISK_SHADOW_LOG", path)
    ctx = _context(daily_loss_pct=4.0)
    decision = evaluate_advanced_risk(ctx, _settings())

    row = decision.to_log_row(ctx)
    assert list(row) == ADVANCED_RISK_SHADOW_COLS

    le.append_advanced_risk_shadow_row(decision, ctx, ADVANCED_RISK_SHADOW_COLS)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["would_block"] == "1"
    assert rows[0]["actually_blocked"] == "0"
    assert rows[0]["top_reason"] == "daily_loss_pct_limit"
