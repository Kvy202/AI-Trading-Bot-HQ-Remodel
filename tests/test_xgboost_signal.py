"""Tests for optional XGBoost signal confirmation in shadow mode."""

from pathlib import Path

import numpy as np

import ml_optional.xgboost_signal as xgs
from ml_optional.xgboost_signal import XGBOOST_SHADOW_COLS, XGBoostSignalConfirmer

ROOT = Path(__file__).resolve().parents[1]


class MockXGBoostModel:
    def __init__(self, probs) -> None:
        self.probs = np.asarray(probs, dtype=float)
        self.classes_ = np.array([0, 1])
        self.last_shape = None

    def predict_proba(self, x):
        self.last_shape = x.shape
        return np.asarray([self.probs], dtype=float)


def _window():
    return np.arange(12, dtype=np.float32).reshape(4, 3)


def _confirmer(model, threshold=0.60):
    return XGBoostSignalConfirmer(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=model,
        model_version="unit-test",
        confidence_threshold=threshold,
        xgboost_status="loaded",
    )


def test_use_xgboost_signal_false_behavior(monkeypatch):
    monkeypatch.setenv("XGBOOST_SIGNAL_ARTIFACT", "model_artifacts/__missing_xgboost_disabled__.joblib")
    logs = []
    confirmer = XGBoostSignalConfirmer.from_env(enabled=False, base_dir=ROOT, log_fn=logs.append)

    assert confirmer.ready is False
    assert confirmer.enabled is False
    assert confirmer.xgboost_status == "disabled_flag_false"
    assert logs == []

    result = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.2,
    )
    assert result.xgboost_enabled is False
    assert result.would_confirm is False
    assert result.would_reject is False
    assert result.reason == "flag_disabled"


def test_missing_artifact_behavior(monkeypatch):
    monkeypatch.setenv("XGBOOST_SIGNAL_ARTIFACT", "model_artifacts/__missing_xgboost_test__.joblib")
    logs = []
    confirmer = XGBoostSignalConfirmer.from_env(enabled=True, base_dir=ROOT, log_fn=logs.append)

    assert confirmer.ready is False
    assert confirmer.xgboost_status == "disabled_missing_artifact"
    assert any("xgboost_status=disabled_missing_artifact" in msg for msg in logs)

    result = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.2,
    )
    assert result.would_confirm is False
    assert result.would_reject is False
    assert result.reason == "artifact_missing"


def test_missing_xgboost_dependency_behavior(monkeypatch, tmp_path):
    artifact = tmp_path / "xgboost_signal.joblib"
    artifact.write_bytes(b"placeholder")
    monkeypatch.setenv("XGBOOST_SIGNAL_ARTIFACT", str(artifact))

    original_find_spec = xgs.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "xgboost":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(xgs.importlib.util, "find_spec", fake_find_spec)
    logs = []
    confirmer = XGBoostSignalConfirmer.from_env(enabled=True, base_dir=ROOT, log_fn=logs.append)

    assert confirmer.ready is False
    assert confirmer.xgboost_status == "disabled_missing_dependency"
    assert any("xgboost_status=disabled_missing_dependency" in msg for msg in logs)


def test_normal_prediction_using_mocked_model():
    model = MockXGBoostModel([0.2, 0.8])
    confirmer = _confirmer(model)

    result = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.31,
        rv_mean=0.04,
        price=100.0,
    )

    assert result.xgboost_enabled is True
    assert result.xgboost_status == "loaded"
    assert result.xgboost_direction == "LONG"
    assert result.xgboost_confidence == 0.8
    assert result.would_confirm is True
    assert result.would_reject is False
    assert result.reason == "confirmed"
    assert model.last_shape == (1, 18)


def test_low_confidence_would_reject_true():
    confirmer = _confirmer(MockXGBoostModel([0.45, 0.55]), threshold=0.60)

    result = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.2,
    )

    assert result.xgboost_direction == "LONG"
    assert result.xgboost_confidence == 0.55
    assert result.would_confirm is False
    assert result.would_reject is True
    assert result.reason == "low_confidence"


def test_high_confidence_would_confirm_true():
    confirmer = _confirmer(MockXGBoostModel([0.1, 0.9]), threshold=0.60)

    result = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.4,
    )

    assert result.would_confirm is True
    assert result.would_reject is False
    assert result.reason == "confirmed"


def test_shadow_log_row_format():
    confirmer = _confirmer(MockXGBoostModel([0.2, 0.8]))
    row = confirmer.evaluate(
        symbol="BTCUSDT",
        window=_window(),
        existing_signal="LONG",
        existing_score=0.25,
    ).to_log_row("2026-06-28 00:00:00+0000", "BTCUSDT")

    assert list(row.keys()) == XGBOOST_SHADOW_COLS
    assert row["timestamp"] == "2026-06-28 00:00:00+0000"
    assert row["symbol"] == "BTCUSDT"
    assert row["xgboost_enabled"] == 1
    assert row["existing_signal"] == "LONG"
    assert row["existing_score"] == 0.25
    assert row["would_confirm"] == 1
    assert row["would_reject"] == 0
    assert row["model_version"] == "unit-test"
    assert row["artifact_path"] == "mock.joblib"