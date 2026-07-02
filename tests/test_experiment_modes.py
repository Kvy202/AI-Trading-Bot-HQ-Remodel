"""Tests for the read-only experimental mode manager."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.experiment_modes import (
    apply_experiment_mode,
    describe_experiment_mode,
    get_experiment_mode,
    list_experiment_modes,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "experiment_mode.py"

EXPECTED_MODES = [
    "baseline",
    "iforest_shadow",
    "iforest_blocking",
    "xgboost_shadow",
    "xgboost_blocking",
    "xgboost_shadow_outcome",
    "survival_shadow",
    "survival_active_placeholder",
    "advanced_risk_shadow_placeholder",
    "advanced_risk_active_placeholder",
    "combined_shadow",
    "combined_paper",
]

EXPERIMENTAL_SYSTEM_KEYS = [
    "USE_ISOLATION_FOREST",
    "USE_XGBOOST_SIGNAL",
    "USE_SURVIVAL_EXIT",
    "USE_ADVANCED_RISK",
]

BLOCKING_KEYS = [
    "ISOLATION_FOREST_BLOCKING",
    "XGBOOST_SIGNAL_BLOCKING",
]


def _mode_env(name: str) -> dict[str, str]:
    return apply_experiment_mode(name, base_env={})


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_all_expected_modes_exist():
    assert list_experiment_modes() == EXPECTED_MODES


def test_baseline_disables_all_experimental_systems():
    env = _mode_env("baseline")

    for key in EXPERIMENTAL_SYSTEM_KEYS:
        assert env[key] == "false"
    for key in BLOCKING_KEYS:
        assert env[key] == "false"
    assert env["SURVIVAL_EXIT_ACTIVE"] == "false"
    assert env["ADVANCED_RISK_ACTIVE"] == "false"


def test_modes_include_paper_safety_overrides():
    env = _mode_env("xgboost_shadow")

    assert env["LIVE_TRADING"] == "false"
    assert env["PAPER_TRADING"] == "true"
    assert env["LIVE_MODE"] == "false"
    assert env["EXEC_PAPER"] == "true"
    assert env["PLACE_REAL_ORDERS"] == "false"


def test_blocking_modes_only_enable_their_own_blocking_flag():
    iforest = _mode_env("iforest_blocking")
    xgboost = _mode_env("xgboost_blocking")

    assert iforest["ISOLATION_FOREST_BLOCKING"] == "true"
    assert iforest["XGBOOST_SIGNAL_BLOCKING"] == "false"
    assert xgboost["ISOLATION_FOREST_BLOCKING"] == "false"
    assert xgboost["XGBOOST_SIGNAL_BLOCKING"] == "true"


def test_shadow_modes_never_enable_blocking():
    shadow_modes = [name for name in EXPECTED_MODES if "shadow" in name]

    for name in shadow_modes:
        env = _mode_env(name)
        for key in BLOCKING_KEYS:
            assert env[key] == "false", name


def test_placeholder_active_modes_do_not_activate_real_behavior_yet():
    survival = _mode_env("survival_active_placeholder")
    advanced = _mode_env("advanced_risk_active_placeholder")

    assert survival["USE_SURVIVAL_EXIT"] == "true"
    assert survival["SURVIVAL_EXIT_ACTIVE"] == "false"
    assert advanced["USE_ADVANCED_RISK"] == "true"
    assert advanced["ADVANCED_RISK_ACTIVE"] == "false"


def test_combined_modes_are_non_blocking_by_default():
    for name in ("combined_shadow", "combined_paper"):
        env = _mode_env(name)
        assert env["USE_ISOLATION_FOREST"] == "true"
        assert env["USE_XGBOOST_SIGNAL"] == "true"
        assert env["USE_SURVIVAL_EXIT"] == "true"
        assert env["USE_ADVANCED_RISK"] == "true"
        assert env["ISOLATION_FOREST_BLOCKING"] == "false"
        assert env["XGBOOST_SIGNAL_BLOCKING"] == "false"
        assert env["SURVIVAL_EXIT_ACTIVE"] == "false"
        assert env["ADVANCED_RISK_ACTIVE"] == "false"


def test_xgboost_shadow_outcome_matches_xgboost_shadow():
    assert get_experiment_mode("xgboost_shadow_outcome").overrides == get_experiment_mode("xgboost_shadow").overrides


def test_apply_experiment_mode_does_not_mutate_input_dict():
    base = {
        "USE_XGBOOST_SIGNAL": "TRUE",
        "CUSTOM_KEY": "kept",
    }
    original = dict(base)

    result = apply_experiment_mode("baseline", base_env=base)

    assert base == original
    assert result is not base
    assert result["CUSTOM_KEY"] == "kept"
    assert result["USE_XGBOOST_SIGNAL"] == "false"


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown experiment mode"):
        apply_experiment_mode("does_not_exist", base_env={})


def test_describe_includes_mode_and_overrides():
    text = describe_experiment_mode("xgboost_shadow")

    assert "Mode: xgboost_shadow" in text
    assert "USE_XGBOOST_SIGNAL=true" in text
    assert "XGBOOST_SIGNAL_BLOCKING=false" in text


def test_cli_list_works():
    result = _run_cli("--list")

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == EXPECTED_MODES


def test_cli_describe_works():
    result = _run_cli("--describe", "xgboost_shadow")

    assert result.returncode == 0, result.stderr
    assert "Mode: xgboost_shadow" in result.stdout
    assert "USE_XGBOOST_SIGNAL=true" in result.stdout


def test_cli_print_env_includes_expected_key_value_lines():
    result = _run_cli("--print-env", "xgboost_shadow")

    assert result.returncode == 0, result.stderr
    lines = set(result.stdout.splitlines())
    assert "USE_XGBOOST_SIGNAL=true" in lines
    assert "XGBOOST_SIGNAL_BLOCKING=false" in lines
    assert "LIVE_TRADING=false" in lines


def test_cli_json_is_valid_json():
    result = _run_cli("--json", "xgboost_shadow")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["name"] == "xgboost_shadow"
    assert payload["overrides"]["USE_XGBOOST_SIGNAL"] == "true"
    assert payload["overrides"]["XGBOOST_SIGNAL_BLOCKING"] == "false"
