"""Tests for the controlled experiment matrix PowerShell runner."""

from __future__ import annotations

import json
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


def _run_matrix_script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    exe = _powershell_exe()
    if exe is None:
        pytest.skip("PowerShell is not available")
    return subprocess.run(
        [exe, "-NoProfile", "-File", str(script), *args],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _write_fake_report_tool(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "",
                "out = None",
                "for idx, arg in enumerate(sys.argv):",
                "    if arg == '--json-out' and idx + 1 < len(sys.argv):",
                "        out = Path(sys.argv[idx + 1])",
                "        break",
                "if out is None:",
                "    raise SystemExit('missing --json-out')",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(json.dumps({'ok': True}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def _make_fake_matrix_repo(tmp_path: Path, child_exit: int) -> Path:
    root = tmp_path / "matrix_repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "reports").mkdir()

    shutil.copy2(SCRIPT, tools / "run_experiment_matrix.ps1")
    (tools / "apply_experiment_mode.ps1").write_text(
        "\n".join(
            [
                "function Get-ExperimentModeOverrides {",
                "  return @{}",
                "}",
                "",
                "function Set-ExperimentModeEnvironment {",
                "  param([hashtable]$Overrides)",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    (tools / "run_combined_shadow_paper_test.ps1").write_text(
        "\n".join(
            [
                "param(",
                "  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,",
                "  [switch]$FreshShadowLogs,",
                "  [switch]$FreshPaperLogs",
                ")",
                'Write-Output "[fake-child] normal output"',
                '[Console]::Error.WriteLine("[fake-child] stderr output")',
                'Write-Output "0"',
                f"exit {child_exit}",
            ]
        ),
        encoding="utf-8",
    )
    for name in (
        "experimental_shadow_report.py",
        "unified_experimental_report.py",
        "audit_xgboost_rejections.py",
    ):
        _write_fake_report_tool(tools / name)
    return tools / "run_experiment_matrix.ps1"


def _latest_matrix_index(script: Path) -> dict:
    reports = script.parents[1] / "reports"
    indexes = sorted(reports.glob("matrix_index_*.json"))
    assert indexes
    return json.loads(indexes[-1].read_text(encoding="utf-8-sig"))


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


def test_matrix_child_output_with_trailing_zero_is_not_exit_status(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)

    result = _run_matrix_script(script, "-Mode", "combined_shadow", "-Minutes", "5")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[fake-child] normal output" in result.stdout
    assert "[fake-child] stderr output" in result.stderr
    assert re.search(r"(?m)^0$", result.stdout)
    assert "Completed successfully" in result.stdout
    assert "Completed with 1 failed mode(s)" not in result.stdout


def test_matrix_combined_shadow_success_records_zero_child_exit(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)

    result = _run_matrix_script(script, "-Mode", "combined_shadow", "-Minutes", "5")
    index = _latest_matrix_index(script)

    assert result.returncode == 0, result.stdout + result.stderr
    assert index["runs"][0]["mode"] == "combined_shadow"
    assert index["runs"][0]["exit_status"] == 0


def test_matrix_index_exit_status_is_numeric_not_child_output(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)

    result = _run_matrix_script(script, "-Mode", "combined_shadow", "-Minutes", "5")
    index = _latest_matrix_index(script)

    assert result.returncode == 0, result.stdout + result.stderr
    exit_status = index["runs"][0]["exit_status"]
    assert isinstance(exit_status, int)
    assert exit_status == 0


def test_matrix_nonzero_child_exit_fails_and_records_numeric_status(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=7)

    result = _run_matrix_script(script, "-Mode", "combined_shadow", "-Minutes", "5")
    index = _latest_matrix_index(script)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "[fake-child] normal output" in result.stdout
    assert "[fake-child] stderr output" in result.stderr
    assert "[matrix] FAIL: combined_shadow: child runbook exited with status 7" in result.stdout
    assert "child runbook exited with status [fake-child]" not in result.stdout
    assert index["runs"][0]["exit_status"] == 7
    assert isinstance(index["runs"][0]["exit_status"], int)
