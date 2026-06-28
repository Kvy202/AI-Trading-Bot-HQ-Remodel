"""Tests for optional Isolation Forest shadow filter."""

from pathlib import Path

import joblib
import numpy as np

from ml_optional.isolation_filter import (
    ISOLATION_SHADOW_COLS,
    IsolationFilter,
    isolation_blocking_from_env,
    should_block_entry,
)

ROOT = Path(__file__).resolve().parents[1]


class MockIsolationModel:
    def __init__(self, pred: int, score: float) -> None:
        self.pred = pred
        self.score = score
        self.last_shape = None

    def predict(self, x):
        self.last_shape = x.shape
        return np.array([self.pred])

    def decision_function(self, x):
        return np.array([self.score])


class MockErrorIsolationModel:
    def predict(self, x):
        raise RuntimeError("mock failure")


def _window():
    return np.arange(12, dtype=np.float32).reshape(4, 3)


def test_flag_disabled_behavior():
    flt = IsolationFilter(
        enabled=False,
        artifact_path=ROOT / "model_artifacts" / "missing.joblib",
    )
    res = flt.evaluate("BTCUSDT", _window())
    assert res.isolation_enabled is False
    assert res.anomaly_status == "not_run"
    assert res.would_block is False


def test_missing_artifact_behavior(monkeypatch):
    monkeypatch.setenv("ISOLATION_FOREST_ARTIFACT", "model_artifacts/__missing_isolation_test__.joblib")
    logs = []
    flt = IsolationFilter.from_env(
        enabled=True,
        base_dir=ROOT,
        log_fn=logs.append,
    )
    assert flt.ready is False
    assert flt.isolation_status == "disabled_missing_artifact"
    assert any("isolation_status=disabled_missing_artifact" in msg for msg in logs)
    res = flt.evaluate("BTCUSDT", _window())
    assert res.would_block is False
    assert res.anomaly_status == "not_run"


def test_normal_prediction_behavior_using_mocked_model():
    model = MockIsolationModel(pred=1, score=0.12)
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=model,
        model_version="mock-normal",
        isolation_status="loaded",
    )
    res = flt.evaluate("BTCUSDT", _window())
    assert res.isolation_enabled is True
    assert res.anomaly_status == "normal"
    assert res.anomaly_score == 0.12
    assert res.would_block is False
    assert res.reason == "normal_market"
    assert model.last_shape == (1, 12)


def test_abnormal_prediction_behavior_using_mocked_model():
    model = MockIsolationModel(pred=-1, score=-0.34)
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=model,
        model_version="mock-anomaly",
        isolation_status="loaded",
    )
    res = flt.evaluate("ETHUSDT", _window())
    assert res.anomaly_status == "anomaly"
    assert res.anomaly_score == -0.34
    assert res.would_block is True
    assert res.reason == "isolation_anomaly"
    assert should_block_entry(res, blocking_enabled=False) is False
    assert should_block_entry(res, blocking_enabled=True) is True


def test_blocking_flag_false_abnormal_does_not_block():
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=MockIsolationModel(pred=-1, score=-0.34),
        model_version="mock-anomaly",
        isolation_status="loaded",
    )
    res = flt.evaluate("ETHUSDT", _window())

    assert res.would_block is True
    assert should_block_entry(res, blocking_enabled=False) is False


def test_blocking_flag_true_abnormal_blocks():
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=MockIsolationModel(pred=-1, score=-0.34),
        model_version="mock-anomaly",
        isolation_status="loaded",
    )
    res = flt.evaluate("ETHUSDT", _window())

    assert should_block_entry(res, blocking_enabled=True) is True


def test_normal_never_blocks():
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=MockIsolationModel(pred=1, score=0.12),
        model_version="mock-normal",
        isolation_status="loaded",
    )
    res = flt.evaluate("BTCUSDT", _window())

    assert res.would_block is False
    assert should_block_entry(res, blocking_enabled=True) is False


def test_missing_artifact_never_blocks(monkeypatch):
    monkeypatch.setenv("ISOLATION_FOREST_ARTIFACT", "model_artifacts/__missing_isolation_block_test__.joblib")
    flt = IsolationFilter.from_env(enabled=True, base_dir=ROOT)
    res = flt.evaluate("BTCUSDT", _window())

    assert flt.ready is False
    assert res.would_block is False
    assert should_block_entry(res, blocking_enabled=True) is False


def test_model_error_never_blocks():
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=MockErrorIsolationModel(),
        model_version="mock-error",
        isolation_status="loaded",
    )
    res = flt.evaluate("BTCUSDT", _window())

    assert res.isolation_status == "model_error"
    assert res.anomaly_status == "error"
    assert res.would_block is False
    assert res.reason == "model_error:RuntimeError"
    assert should_block_entry(res, blocking_enabled=True) is False


def test_no_window_logs_without_blocking():
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=MockIsolationModel(pred=-1, score=-0.34),
        model_version="mock-anomaly",
        isolation_status="loaded",
    )
    res = flt.evaluate("BTCUSDT", None)

    assert res.isolation_status == "no_window"
    assert res.anomaly_status == "not_run"
    assert res.would_block is False
    assert res.reason == "no_window"
    assert should_block_entry(res, blocking_enabled=True) is False


def test_default_blocking_flag_false(monkeypatch):
    monkeypatch.delenv("ISOLATION_FOREST_BLOCKING", raising=False)
    assert isolation_blocking_from_env() is False
    monkeypatch.setenv("ISOLATION_FOREST_BLOCKING", "true")
    assert isolation_blocking_from_env() is True


def test_artifact_save_load_behavior(monkeypatch):
    artifact_path = ROOT / "tests" / ".tmp_isolation_filter_test.joblib"
    try:
        joblib.dump(
            {"model": MockIsolationModel(pred=1, score=0.25), "model_version": "unit-test"},
            artifact_path,
        )
        monkeypatch.setenv("ISOLATION_FOREST_ARTIFACT", str(artifact_path))
        flt = IsolationFilter.from_env(enabled=True, base_dir=ROOT)
        assert flt.ready is True
        assert flt.model_version == "unit-test"
        res = flt.evaluate("BTCUSDT", _window())
        assert res.anomaly_status == "normal"
        assert res.anomaly_score == 0.25
    finally:
        if artifact_path.exists():
            artifact_path.unlink()


def test_shadow_log_row_format():
    model = MockIsolationModel(pred=-1, score=-0.2)
    flt = IsolationFilter(
        enabled=True,
        artifact_path=Path("mock.joblib"),
        model=model,
        model_version="row-format",
        isolation_status="loaded",
    )
    row = flt.evaluate("BTCUSDT", _window()).to_log_row("2026-06-28 00:00:00+0000", "BTCUSDT")
    assert list(row.keys()) == ISOLATION_SHADOW_COLS
    assert row["timestamp"] == "2026-06-28 00:00:00+0000"
    assert row["anomaly_status"] == "anomaly"
    assert row["would_block"] == 1
    assert row["actually_blocked"] == 0
    assert row["reason"] == "isolation_anomaly"

    blocked_row = flt.evaluate("BTCUSDT", _window()).to_log_row(
        "2026-06-28 00:00:00+0000",
        "BTCUSDT",
        actually_blocked=True,
    )
    assert blocked_row["actually_blocked"] == 1
    assert blocked_row["reason"] == "isolation_forest_block"
