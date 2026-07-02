"""Tests for the controlled experiment matrix PowerShell runner."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_experiment_matrix.ps1"

EXPECTED_MODES = [
    "baseline",
    "iforest_shadow",
    "iforest_blocking",
    "xgboost_shadow_outcome",
    "survival_shadow",
    "survival_active",
    "advanced_risk_shadow",
    "combined_shadow",
]


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _powershell_exe() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _run_ps(*args: str) -> subprocess.CompletedProcess[str]:
    exe = _powershell_exe()
    if exe is None:
        pytest.skip("PowerShell is not available")
    return subprocess.run(
        [exe, "-NoProfile", "-File", str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_matrix_parser_accepts_required_minutes():
    text = _script_text()

    assert "[ValidateSet(5, 30, 60)] [int]$Minutes = 30" in text
    assert "[string]$Mode" in text
    assert "[switch]$All" in text
    assert "[switch]$FreshLogs" in text
    assert "[switch]$DryRun" in text


def test_matrix_requires_mode_or_all():
    result = _run_ps("-DryRun")

    assert result.returncode == 2
    assert "specify either -Mode <mode> or -All" in result.stdout


def test_matrix_invalid_mode_fails_clearly():
    result = _run_ps("-Mode", "not_a_mode", "-Minutes", "5", "-DryRun")

    assert result.returncode == 2
    assert "invalid mode 'not_a_mode'" in result.stdout
    assert "Supported modes:" in result.stdout


def test_matrix_dry_run_does_not_start_processes():
    result = _run_ps("-Mode", "baseline", "-Minutes", "5", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN ONLY" in result.stdout
    assert "no writer/executor processes will be started" in result.stdout
    assert "Starting generic paper mode" not in result.stdout
    assert "Start-Process" not in result.stdout


def test_matrix_each_mode_maps_to_expected_runbook_or_generic_command():
    text = _script_text()

    for mode in ("baseline", "iforest_shadow", "survival_shadow"):
        assert f"{mode} = '{mode}'" in text

    mapping = {
        "iforest_blocking": "run_isolation_forest_blocking_paper_test.ps1",
        "xgboost_shadow_outcome": "run_xgboost_shadow_outcome_paper_test.ps1",
        "survival_active": "run_survival_active_paper_test.ps1",
        "advanced_risk_shadow": "run_advanced_risk_shadow_paper_test.ps1",
        "combined_shadow": "run_combined_shadow_paper_test.ps1",
    }
    for mode, runbook in mapping.items():
        assert mode in text
        assert runbook in text


def test_matrix_commands_force_paper_only_flags_and_refuse_live_mode():
    text = _script_text()

    for snippet in (
        "$overrides['LIVE_TRADING'] = 'false'",
        "$overrides['PAPER_TRADING'] = 'true'",
        "$overrides['LIVE_MODE'] = 'false'",
        "$overrides['EXEC_PAPER'] = 'true'",
        "$overrides['PLACE_REAL_ORDERS'] = 'false'",
        'os.environ["LIVE_TRADING"] = "false"',
        'os.environ["PAPER_TRADING"] = "true"',
        'os.environ["LIVE_MODE"] = "false"',
        'os.environ["EXEC_PAPER"] = "true"',
        'os.environ["PLACE_REAL_ORDERS"] = "false"',
        "resolve_trading_mode",
        "live_requested",
        "d.place_real_orders",
        "production_detected",
        "hyperliquid_mainnet_selected",
        "REFUSING: live/mainnet/real-order mode detected",
    ):
        assert snippet in text


def test_matrix_report_path_naming_is_deterministic_enough():
    result = _run_ps("-Mode", "baseline", "-Minutes", "5", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"matrix_baseline_\d{14}_unified\.json", result.stdout)
    assert re.search(r"matrix_baseline_\d{14}_shadow_summary\.json", result.stdout)
    assert re.search(r"matrix_index_\d{14}\.json", result.stdout)


def test_matrix_all_includes_all_expected_modes():
    result = _run_ps("-All", "-Minutes", "5", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    for mode in EXPECTED_MODES:
        assert f"Mode: {mode}" in result.stdout


def test_matrix_xgboost_and_combined_modes_include_audit_report_paths():
    result = _run_ps("-All", "-Minutes", "5", "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    assert re.search(r"matrix_xgboost_shadow_outcome_\d{14}_xgboost_audit\.json", result.stdout)
    assert re.search(r"matrix_combined_shadow_\d{14}_xgboost_audit\.json", result.stdout)


def test_matrix_runbook_parses_when_powershell_available():
    exe = _powershell_exe()
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
