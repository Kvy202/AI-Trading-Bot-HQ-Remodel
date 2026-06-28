"""Tests for optional Survival Analysis exit timing in shadow mode."""

from pathlib import Path

import joblib
import numpy as np

import ml_optional.survival_exit as surv
from ml_optional.survival_exit import SURVIVAL_SHADOW_COLS, SurvivalExitModel

ROOT = Path(__file__).resolve().parents[1]


class MockSurvivalRiskModel:
    def __init__(self, risk: float) -> None:
        self.risk = risk
        self.last_shape = None

    def predict(self, x):
        self.last_shape = getattr(x, "shape", None)
        return np.asarray([self.risk], dtype=float)


def _confirmer(model, threshold=0.60):
    return SurvivalExitModel(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=model,
        model_version="unit-test",
        risk_threshold=threshold,
        survival_status="loaded",
    )


def _eval(model: SurvivalExitModel):
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


def test_use_survival_exit_false_behavior(monkeypatch):
    monkeypatch.setenv("SURVIVAL_EXIT_ARTIFACT", "model_artifacts/__missing_survival_disabled__.joblib")
    logs = []
    model = SurvivalExitModel.from_env(enabled=False, base_dir=ROOT, log_fn=logs.append)

    assert model.ready is False
    assert model.enabled is False
    assert model.survival_status == "disabled_flag_false"
    assert logs == []

    result = _eval(model)
    assert result.survival_enabled is False
    assert result.would_hold is True
    assert result.would_exit_early is False
    assert result.reason == "flag_disabled"


def test_missing_artifact_behavior(monkeypatch):
    monkeypatch.setenv("SURVIVAL_EXIT_ARTIFACT", "model_artifacts/__missing_survival_test__.joblib")
    logs = []
    model = SurvivalExitModel.from_env(enabled=True, base_dir=ROOT, log_fn=logs.append)

    assert model.ready is False
    assert model.survival_status == "disabled_missing_artifact"
    assert any("survival_status=disabled_missing_artifact" in msg for msg in logs)

    result = _eval(model)
    assert result.would_hold is True
    assert result.would_exit_early is False
    assert result.reason == "artifact_missing"


def test_missing_survival_dependency_behavior(monkeypatch, tmp_path):
    artifact = tmp_path / "survival_exit.joblib"
    artifact.write_bytes(b"placeholder")
    monkeypatch.setenv("SURVIVAL_EXIT_ARTIFACT", str(artifact))
    monkeypatch.setattr(surv, "_survival_dependency_available", lambda: False)

    logs = []
    model = SurvivalExitModel.from_env(enabled=True, base_dir=ROOT, log_fn=logs.append)

    assert model.ready is False
    assert model.survival_status == "disabled_missing_dependency"
    assert any("survival_status=disabled_missing_dependency" in msg for msg in logs)


def test_normal_prediction_using_mocked_model():
    mock = MockSurvivalRiskModel(0.42)
    model = _confirmer(mock)
    result = _eval(model)

    assert result.survival_enabled is True
    assert result.survival_status == "loaded"
    assert result.survival_risk_score == 0.42
    assert result.current_age_minutes == 10.0
    assert result.would_hold is True
    assert result.would_exit_early is False
    assert result.reason == "hold_risk_below_threshold"
    assert mock.last_shape == (1, 10)


def test_low_risk_would_hold_true():
    result = _eval(_confirmer(MockSurvivalRiskModel(0.20), threshold=0.60))

    assert result.survival_risk_score == 0.20
    assert result.would_hold is True
    assert result.would_exit_early is False
    assert result.reason == "hold_risk_below_threshold"


def test_high_risk_would_exit_early_true():
    result = _eval(_confirmer(MockSurvivalRiskModel(0.82), threshold=0.60))

    assert result.survival_risk_score == 0.82
    assert result.would_hold is False
    assert result.would_exit_early is True
    assert result.reason == "high_exit_risk"


def test_artifact_save_load_behavior(monkeypatch, tmp_path):
    artifact = tmp_path / "survival_exit.joblib"
    joblib.dump(
        {"model": MockSurvivalRiskModel(0.33), "model_version": "unit-artifact", "risk_threshold": 0.60},
        artifact,
    )
    monkeypatch.setenv("SURVIVAL_EXIT_ARTIFACT", str(artifact))
    monkeypatch.setattr(surv, "_survival_dependency_available", lambda: True)

    model = SurvivalExitModel.from_env(enabled=True, base_dir=ROOT)
    assert model.ready is True
    assert model.model_version == "unit-artifact"

    result = _eval(model)
    assert result.survival_risk_score == 0.33
    assert result.would_hold is True


def test_shadow_log_row_format():
    result = _eval(_confirmer(MockSurvivalRiskModel(0.82)))
    row = result.to_log_row("2026-06-28 00:00:00+0000", "BTCUSDT")

    assert list(row.keys()) == SURVIVAL_SHADOW_COLS
    assert row["timestamp"] == "2026-06-28 00:00:00+0000"
    assert row["symbol"] == "BTCUSDT"
    assert row["survival_enabled"] == 1
    assert row["trade_id"] == "BTCUSDT:test"
    assert row["side"] == "long"
    assert row["current_age_seconds"] == 600.0
    assert row["current_age_minutes"] == 10.0
    assert row["survival_risk_score"] == 0.82
    assert row["would_hold"] == 0
    assert row["would_exit_early"] == 1
    assert row["model_version"] == "unit-test"
    assert row["artifact_path"] == "mock.joblib"
