"""Tests for the Advanced Risk shadow paper-test PowerShell runbook."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from runtime.experiment_modes import get_experiment_mode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_advanced_risk_shadow_paper_test.ps1"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_advanced_risk_runbook_parser_accepts_required_minutes():
    text = _script_text()

    assert "[ValidateSet(5, 30, 60)] [int]$Minutes = 30" in text
    assert "-FreshShadowLog" in text
    assert "-FreshPaperLogs" in text
    assert "Duration: {0} minutes" in text


def test_advanced_risk_runbook_uses_mode_manager_and_shadow_overrides():
    text = _script_text()

    for snippet in (
        "apply_experiment_mode.ps1",
        "$experimentMode = 'advanced_risk_shadow_placeholder'",
        "Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode",
        "$forcedPaperEnv['USE_ADVANCED_RISK'] = 'true'",
        "$forcedPaperEnv['ADVANCED_RISK_ACTIVE'] = 'false'",
        "$forcedPaperEnv['ADVANCED_RISK_MAX_DAILY_LOSS_PCT'] = '3.0'",
        "$forcedPaperEnv['ADVANCED_RISK_MAX_CONSECUTIVE_LOSSES'] = '3'",
        "$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'",
        "advanced_risk_shadow_mode_env.json",
        "tools\\verify_advanced_risk.py",
        "advanced_risk_shadow_paper_summary.json",
    ):
        assert snippet in text

    assert "combined_paper" not in text
    assert "$forcedPaperEnv['ADVANCED_RISK_ACTIVE'] = 'true'" not in text


def test_advanced_risk_modes_remain_placeholder_safe():
    shadow = get_experiment_mode("advanced_risk_shadow_placeholder").overrides
    placeholder = get_experiment_mode("advanced_risk_active_placeholder").overrides
    combined = get_experiment_mode("combined_paper").overrides

    assert shadow["USE_ADVANCED_RISK"] == "true"
    assert shadow["ADVANCED_RISK_ACTIVE"] == "false"
    assert placeholder["USE_ADVANCED_RISK"] == "true"
    assert placeholder["ADVANCED_RISK_ACTIVE"] == "false"
    assert combined["USE_ADVANCED_RISK"] == "true"
    assert combined["ADVANCED_RISK_ACTIVE"] == "false"


def test_advanced_risk_runbook_refuses_live_or_real_order_mode():
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


def test_advanced_risk_runbook_starts_executor_in_paper_mode_only():
    text = _script_text()

    assert 'sys.argv = ["tools/live_executor.py", "--paper", "--signals", "logs/live_signals.csv"]' in text
    assert '"--live"' not in text
    assert "Starting live_writer and live_executor in paper mode." in text
    assert "Start-Process -FilePath $py" in text
    assert "-WindowStyle Hidden" in text


def test_advanced_risk_runbook_parses_when_powershell_available():
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
