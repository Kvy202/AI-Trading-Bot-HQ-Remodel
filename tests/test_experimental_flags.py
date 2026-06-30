"""Tests for default-off experimental flags."""

import os
from pathlib import Path

from ml_optional.isolation_filter import isolation_blocking_from_env
from ml_optional.xgboost_signal import xgboost_signal_blocking_from_env
from runtime.experimental_flags import EXPERIMENTAL_FLAG_NAMES, ExperimentalFlags
from runtime.loader import apply_run_config

ROOT = Path(__file__).resolve().parents[1]


def _clear_flags(monkeypatch):
    for name in EXPERIMENTAL_FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_missing_flags_default_false(monkeypatch):
    _clear_flags(monkeypatch)
    flags = ExperimentalFlags.from_env()
    assert flags.as_env_dict() == {name: False for name in EXPERIMENTAL_FLAG_NAMES}


def test_env_bool_parsing(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.setenv("USE_ISOLATION_FOREST", "true")
    monkeypatch.setenv("USE_XGBOOST_SIGNAL", "1")
    monkeypatch.setenv("USE_SURVIVAL_EXIT", "yes")
    monkeypatch.setenv("USE_ADVANCED_RISK", "on")
    flags = ExperimentalFlags.from_env()
    assert flags.as_env_dict() == {name: True for name in EXPERIMENTAL_FLAG_NAMES}


def test_run_config_false_defaults_load_through_existing_loader(monkeypatch):
    _clear_flags(monkeypatch)
    monkeypatch.delenv("ISOLATION_FOREST_BLOCKING", raising=False)
    monkeypatch.delenv("XGBOOST_SIGNAL_BLOCKING", raising=False)

    loaded = apply_run_config(ROOT)
    flags = ExperimentalFlags.from_env()

    assert set(EXPERIMENTAL_FLAG_NAMES).issubset(loaded)
    assert loaded["ISOLATION_FOREST_BLOCKING"] == "False"
    assert loaded["XGBOOST_SIGNAL_BLOCKING"] == "False"
    assert os.getenv("ISOLATION_FOREST_BLOCKING") == "False"
    assert os.getenv("XGBOOST_SIGNAL_BLOCKING") == "False"
    assert isolation_blocking_from_env() is False
    assert xgboost_signal_blocking_from_env() is False
    assert flags.as_env_dict() == {name: False for name in EXPERIMENTAL_FLAG_NAMES}


def test_summary_uses_env_names(monkeypatch):
    _clear_flags(monkeypatch)
    text = ExperimentalFlags.from_env().summary()
    for name in EXPERIMENTAL_FLAG_NAMES:
        assert f"{name}=false" in text
