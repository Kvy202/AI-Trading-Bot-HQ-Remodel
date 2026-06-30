"""Tests for the XGBoost paper-test PowerShell runbook."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_xgboost_blocking_paper_test.ps1"


def test_xgboost_paper_runbook_forces_paper_xgboost_flags():
    text = SCRIPT.read_text(encoding="utf-8")

    for snippet in (
        "LIVE_TRADING = 'false'",
        "PAPER_TRADING = 'true'",
        "LIVE_MODE = 'false'",
        "EXEC_PAPER = 'true'",
        "PLACE_REAL_ORDERS = 'false'",
        "USE_XGBOOST_SIGNAL = 'true'",
        "XGBOOST_SIGNAL_BLOCKING = 'true'",
        "XGBOOST_SIGNAL_ARTIFACT = $artifactFull",
        "USE_ISOLATION_FOREST = 'false'",
        "USE_SURVIVAL_EXIT = 'false'",
        "Starting live_writer only; executor is not started.",
        "tools\\verify_xgboost_signal.py",
    ):
        assert snippet in text

    assert 'sys.argv = ["tools/live_writer.py"]' in text
    assert 'sys.argv = ["tools/live_executor.py"]' not in text


def test_xgboost_paper_runbook_parses_when_powershell_available():
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
