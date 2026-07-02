"""Tests for the XGBoost lineage paper-test PowerShell runbook."""

import shutil
import subprocess
from pathlib import Path

import pytest

from runtime.experiment_modes import get_experiment_mode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_xgboost_lineage_paper_test.ps1"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_lineage_runbook_parser_accepts_required_minutes():
    text = _script_text()

    assert "[ValidateSet(5, 30, 60)] [int]$Minutes = 30" in text
    assert "-Minutes" in text
    assert "Duration: {0} minutes" in text


def test_lineage_runbook_uses_xgboost_blocking_mode():
    text = _script_text()

    for snippet in (
        "apply_experiment_mode.ps1",
        "$experimentMode = 'xgboost_blocking'",
        "Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode",
        "$forcedPaperEnv['XGBOOST_SIGNAL_ARTIFACT'] = $artifactFull",
        "$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'",
        "xgboost_lineage_mode_env.json",
        "os.environ.update(mode_env)",
        "os.environ.update(forced_env)",
        "os.environ.update(FORCED)",
        "tools\\verify_xgboost_signal.py",
        "--missing-artifact-check",
        "tools\\audit_xgboost_rejections.py",
        "reports",
        "xgboost_lineage_paper_audit.json",
    ):
        assert snippet in text

    assert "combined_paper" not in text


def test_lineage_xgboost_blocking_mode_keeps_paper_safe_flags_and_blocking():
    env = get_experiment_mode("xgboost_blocking").overrides

    assert env["LIVE_TRADING"] == "false"
    assert env["PAPER_TRADING"] == "true"
    assert env["LIVE_MODE"] == "false"
    assert env["EXEC_PAPER"] == "true"
    assert env["PLACE_REAL_ORDERS"] == "false"
    assert env["USE_XGBOOST_SIGNAL"] == "true"
    assert env["XGBOOST_SIGNAL_BLOCKING"] == "true"
    assert env["USE_ISOLATION_FOREST"] == "false"
    assert env["ISOLATION_FOREST_BLOCKING"] == "false"
    assert env["USE_SURVIVAL_EXIT"] == "false"


def test_lineage_runbook_refuses_live_or_real_order_mode():
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


def test_lineage_runbook_starts_executor_in_paper_mode_only():
    text = _script_text()

    assert 'sys.argv = ["tools/live_executor.py", "--paper", "--signals", "logs/live_signals.csv"]' in text
    assert '"--live"' not in text
    assert "Starting live_writer and live_executor in paper mode." in text
    assert "Start-Process -FilePath $py" in text
    assert "-WindowStyle Hidden" in text


def test_lineage_runbook_does_not_enable_isolation_or_survival():
    text = _script_text()
    env = get_experiment_mode("xgboost_blocking").overrides

    assert env["USE_ISOLATION_FOREST"] == "false"
    assert env["USE_SURVIVAL_EXIT"] == "false"
    assert "$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'" in text
    assert "USE_ISOLATION_FOREST = 'true'" not in text
    assert "USE_SURVIVAL_EXIT = 'true'" not in text


def test_lineage_runbook_checks_aggregate_and_dated_closed_logs():
    text = _script_text()

    assert "trades_closed.csv" in text
    assert "trades_closed_*.csv" in text
    assert "closed_master_signal_id_count" in text
    assert "closed_dated_signal_id_count" in text
    assert "aggregate closed trade rows occurred but no signal_id was logged" in text
    assert "dated closed trade rows occurred but no signal_id was logged" in text


def test_lineage_runbook_parses_when_powershell_available():
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
