"""Tests for the Isolation Forest blocking paper-test PowerShell runbook."""

import shutil
import subprocess
from pathlib import Path

import pytest

from runtime.experiment_modes import get_experiment_mode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_isolation_forest_blocking_paper_test.ps1"
RUNBOOKS = [
    ROOT / "tools" / "run_xgboost_blocking_paper_test.ps1",
    ROOT / "tools" / "run_xgboost_lineage_paper_test.ps1",
    ROOT / "tools" / "run_xgboost_shadow_outcome_paper_test.ps1",
    SCRIPT,
]


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_isolation_runbook_uses_iforest_blocking_mode():
    text = _script_text()

    for snippet in (
        "apply_experiment_mode.ps1",
        "$experimentMode = 'iforest_blocking'",
        "Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode",
        "$forcedPaperEnv['ISOLATION_FOREST_ARTIFACT'] = $artifactFull",
        "isolation_forest_blocking_mode_env.json",
        "os.environ.update(mode_env)",
        "os.environ.update(forced_env)",
        "tools\\verify_isolation_forest.py",
    ):
        assert snippet in text


def test_isolation_mode_keeps_paper_safe_flags_and_iforest_blocking():
    env = get_experiment_mode("iforest_blocking").overrides

    assert env["LIVE_TRADING"] == "false"
    assert env["PAPER_TRADING"] == "true"
    assert env["LIVE_MODE"] == "false"
    assert env["EXEC_PAPER"] == "true"
    assert env["PLACE_REAL_ORDERS"] == "false"
    assert env["USE_ISOLATION_FOREST"] == "true"
    assert env["ISOLATION_FOREST_BLOCKING"] == "true"
    assert env["USE_XGBOOST_SIGNAL"] == "false"
    assert env["XGBOOST_SIGNAL_BLOCKING"] == "false"
    assert env["USE_SURVIVAL_EXIT"] == "false"


def test_isolation_runbook_refuses_live_or_real_order_mode():
    text = _script_text()

    for snippet in (
        "resolve_trading_mode",
        "live_requested",
        "d.place_real_orders",
        "production_detected",
        "hyperliquid_mainnet_selected",
        "REFUSING: live/mainnet mode detected",
        "guardrail resolves to a real-order mode",
    ):
        assert snippet in text


def test_isolation_runbook_starts_writer_only():
    text = _script_text()

    assert "Starting live_writer only; executor is not started." in text
    assert 'sys.argv = ["tools/live_writer.py"]' in text
    assert 'sys.argv = ["tools/live_executor.py"]' not in text


def test_no_paper_runbook_uses_combined_paper_yet():
    for runbook in RUNBOOKS:
        assert "combined_paper" not in runbook.read_text(encoding="utf-8")


def test_isolation_runbook_parses_when_powershell_available():
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is not available")

    script_arg = str(SCRIPT).replace("'", "''")
    command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{script_arg}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Host $_.Message }; exit 1 "
        "}"
    )
    result = subprocess.run(
        [exe, "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
