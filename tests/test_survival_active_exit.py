"""Tests for the optional paper-only Survival active exit gate."""

from pathlib import Path

import csv

import numpy as np

import tools.live_executor as le
from ml_optional.survival_exit import SURVIVAL_SHADOW_COLS, SurvivalExitModel
from runtime.loader import apply_run_config

ROOT = Path(__file__).resolve().parents[1]


class StaticRiskModel:
    def __init__(self, risk: float) -> None:
        self.risk = risk

    def predict(self, x):
        return np.asarray([self.risk], dtype=float)


def _result(risk: float = 0.82):
    model = SurvivalExitModel(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=StaticRiskModel(risk),
        model_version="unit-test",
        risk_threshold=0.60,
        survival_status="loaded",
    )
    return model.evaluate(
        symbol="BTCUSDT",
        side="long",
        trade_id="BTCUSDT:test",
        entry_time="2026-06-28 00:00:00+0000",
        current_age_seconds=600.0,
        current_unrealized_pnl=-0.01,
        entry_price=100.0,
        current_price=99.0,
        qty=0.1,
    )


def test_config_default_survival_exit_active_false(monkeypatch):
    monkeypatch.delenv("SURVIVAL_EXIT_ACTIVE", raising=False)

    try:
        loaded = apply_run_config(ROOT)

        assert loaded["SURVIVAL_EXIT_ACTIVE"] == "False"
        assert le.env_bool("SURVIVAL_EXIT_ACTIVE", False) is False
    finally:
        monkeypatch.delenv("SURVIVAL_EXIT_ACTIVE", raising=False)


def test_survival_shadow_append_includes_active_exit_fields(monkeypatch, tmp_path):
    path = tmp_path / "survival_exit_shadow.csv"
    monkeypatch.setattr(le, "SURVIVAL_SHADOW_LOG", path)

    le.append_survival_shadow_row(
        _result(),
        "2026-06-28 00:00:00+0000",
        "BTCUSDT",
        SURVIVAL_SHADOW_COLS,
        actually_exited=True,
        exit_reason="survival_high_exit_risk",
        survival_active=True,
        paper_only_guard="paper_only_ok",
    )

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["actually_exited"] == "1"
    assert rows[0]["exit_reason"] == "survival_high_exit_risk"
    assert rows[0]["survival_active"] == "1"
    assert rows[0]["paper_only_guard"] == "paper_only_ok"


def test_survival_shadow_append_extends_old_header(monkeypatch, tmp_path):
    path = tmp_path / "survival_exit_shadow.csv"
    old_header = SURVIVAL_SHADOW_COLS[:-4]
    path.write_text(",".join(old_header) + "\n", encoding="utf-8")
    monkeypatch.setattr(le, "SURVIVAL_SHADOW_LOG", path)

    le.append_survival_shadow_row(
        _result(),
        "2026-06-28 00:00:00+0000",
        "BTCUSDT",
        SURVIVAL_SHADOW_COLS,
        actually_exited=True,
        exit_reason="survival_high_exit_risk",
        survival_active=True,
        paper_only_guard="paper_only_ok",
    )

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert "actually_exited" in rows[0]
    assert rows[0]["actually_exited"] == "1"
