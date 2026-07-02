"""Tests for the combined experimental shadow paper-test PowerShell runbook."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from runtime.experiment_modes import get_experiment_mode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_combined_shadow_paper_test.ps1"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_combined_shadow_runbook_parser_accepts_required_minutes():
    text = _script_text()

    assert "[ValidateSet(5, 30, 60)] [int]$Minutes = 30" in text
    assert "-FreshShadowLogs" in text
    assert "-FreshPaperLogs" in text
    assert "Duration: {0} minutes" in text


def test_combined_shadow_runbook_uses_combined_shadow_mode():
    text = _script_text()

    assert "apply_experiment_mode.ps1" in text
    assert "$experimentMode = 'combined_shadow'" in text
    assert "Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode" in text
    assert "combined_shadow_mode_env.json" in text
    assert "combined_paper" not in text


def test_combined_shadow_mode_expected_flags_are_shadow_safe():
    env = get_experiment_mode("combined_shadow").overrides

    assert env["LIVE_TRADING"] == "false"
    assert env["PAPER_TRADING"] == "true"
    assert env["LIVE_MODE"] == "false"
    assert env["EXEC_PAPER"] == "true"
    assert env["PLACE_REAL_ORDERS"] == "false"
    assert env["USE_ISOLATION_FOREST"] == "true"
    assert env["ISOLATION_FOREST_BLOCKING"] == "false"
    assert env["USE_XGBOOST_SIGNAL"] == "true"
    assert env["XGBOOST_SIGNAL_BLOCKING"] == "false"
    assert env["USE_SURVIVAL_EXIT"] == "true"
    assert env["SURVIVAL_EXIT_ACTIVE"] == "false"
    assert env["USE_ADVANCED_RISK"] == "true"
    assert env["ADVANCED_RISK_ACTIVE"] == "false"


def test_combined_shadow_runbook_contains_paper_only_flags_and_refusal():
    text = _script_text()

    for snippet in (
        "$forcedPaperEnv['LIVE_TRADING'] = 'false'",
        "$forcedPaperEnv['PAPER_TRADING'] = 'true'",
        "$forcedPaperEnv['LIVE_MODE'] = 'false'",
        "$forcedPaperEnv['EXEC_PAPER'] = 'true'",
        "$forcedPaperEnv['PLACE_REAL_ORDERS'] = 'false'",
        "resolve_trading_mode",
        "live_requested",
        "d.place_real_orders",
        "production_detected",
        "hyperliquid_mainnet_selected",
        "REFUSING: live/mainnet mode detected",
        "guardrail resolves to a real-order mode",
    ):
        assert snippet in text


def test_combined_shadow_runbook_does_not_enable_blocking_or_active_flags():
    text = _script_text()

    forbidden = (
        "$forcedPaperEnv['ISOLATION_FOREST_BLOCKING'] = 'true'",
        "$forcedPaperEnv['XGBOOST_SIGNAL_BLOCKING'] = 'true'",
        "$forcedPaperEnv['SURVIVAL_EXIT_ACTIVE'] = 'true'",
        "$forcedPaperEnv['ADVANCED_RISK_ACTIVE'] = 'true'",
    )
    for snippet in forbidden:
        assert snippet not in text

    for snippet in (
        "$forcedPaperEnv['ISOLATION_FOREST_BLOCKING'] = 'false'",
        "$forcedPaperEnv['XGBOOST_SIGNAL_BLOCKING'] = 'false'",
        "$forcedPaperEnv['SURVIVAL_EXIT_ACTIVE'] = 'false'",
        "$forcedPaperEnv['ADVANCED_RISK_ACTIVE'] = 'false'",
    ):
        assert snippet in text


def test_combined_shadow_runbook_starts_writer_and_executor_in_paper_mode():
    text = _script_text()

    assert 'sys.argv = ["tools/live_writer.py"]' in text
    assert 'sys.argv = ["tools/live_executor.py", "--paper", "--signals", "logs/live_signals.csv"]' in text
    assert '"--live"' not in text
    assert "Starting live_writer and live_executor in paper mode." in text
    assert "Start-Process -FilePath $py" in text
    assert "-WindowStyle Hidden" in text


def test_combined_shadow_runbook_runs_all_verifiers():
    text = _script_text()

    for snippet in (
        'tools\\verify_isolation_forest.py" --artifact "model_artifacts/isolation_forest.joblib" --missing-artifact-check',
        'tools\\verify_xgboost_signal.py" --artifact "model_artifacts/xgboost_signal.joblib" --missing-artifact-check',
        'tools\\verify_survival_exit.py" --artifact "model_artifacts/survival_exit.joblib" --missing-artifact-check',
        'tools\\verify_advanced_risk.py"',
    ):
        assert snippet in text


def test_combined_shadow_runbook_generates_both_reports():
    text = _script_text()

    assert "tools\\experimental_shadow_report.py" in text
    assert "--logs-dir $logsDir --json --json-out $reportJson" in text
    assert "combined_shadow_paper_summary.json" in text
    assert "tools\\audit_xgboost_rejections.py" in text
    assert "--json --json-out $auditJson" in text
    assert "combined_shadow_xgboost_audit.json" in text
    assert "Assert-ReportWritten -PathValue $reportJson" in text
    assert "Assert-ReportWritten -PathValue $auditJson" in text


def test_fresh_shadow_logs_include_all_four_shadow_logs():
    text = _script_text()

    assert "$FreshShadowLogs" in text
    for name in (
        "isolation_forest_shadow.csv",
        "xgboost_signal_shadow.csv",
        "survival_exit_shadow.csv",
        "advanced_risk_shadow.csv",
    ):
        assert name in text


def test_fresh_paper_logs_include_paper_and_closed_trade_logs():
    text = _script_text()

    assert "$FreshPaperLogs" in text
    assert "trades_paper_*.csv" in text
    assert "trades_closed.csv" in text
    assert "trades_closed_*.csv" in text
    assert "no paper trades occurred during this run; not failing" in text
    assert "no closed trades occurred during this run; not failing" in text


def test_combined_shadow_runbook_parses_when_powershell_available():
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
