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

SHADOW_LOGS = [
    "isolation_forest_shadow.csv",
    "xgboost_signal_shadow.csv",
    "survival_exit_shadow.csv",
    "advanced_risk_shadow.csv",
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
                "shadow_names = [",
                "    'isolation_forest_shadow.csv',",
                "    'xgboost_signal_shadow.csv',",
                "    'survival_exit_shadow.csv',",
                "    'advanced_risk_shadow.csv',",
                "]",
                "logs_dir = Path('logs')",
                "out = None",
                "for idx, arg in enumerate(sys.argv):",
                "    if arg == '--json-out' and idx + 1 < len(sys.argv):",
                "        out = Path(sys.argv[idx + 1])",
                "    if arg == '--logs-dir' and idx + 1 < len(sys.argv):",
                "        logs_dir = Path(sys.argv[idx + 1])",
                "if out is None:",
                "    raise SystemExit('missing --json-out')",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "rows = {}",
                "for name in shadow_names:",
                "    log = logs_dir / name",
                "    if not log.exists():",
                "        rows[name] = 0",
                "        continue",
                "    lines = [line for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]",
                "    rows[name] = max(0, len(lines) - 1)",
                "out.write_text(json.dumps({'ok': True, 'shadow_rows': rows}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )


def _fake_child_runbook(log_names: list[str], child_exit: int) -> str:
    lines = [
        "param(",
        "  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,",
        "  [switch]$FreshShadowLog,",
        "  [switch]$FreshShadowLogs,",
        "  [switch]$FreshPaperLogs",
        ")",
        "$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
        "$root = (Resolve-Path (Join-Path $scriptDir '..')).Path",
        "$logsDir = Join-Path $root 'logs'",
        "New-Item -ItemType Directory -Path $logsDir -Force | Out-Null",
        'Write-Output "[fake-child] normal output"',
        '[Console]::Error.WriteLine("[fake-child] stderr output")',
        'Write-Output "0"',
    ]
    for name in log_names:
        lines.extend(
            [
                f"$path = Join-Path $logsDir '{name}'",
                f"Set-Content -Path $path -Value \"header`n{Path(name).stem}_new_row\" -Encoding UTF8",
            ]
        )
    lines.append(f"exit {child_exit}")
    return "\n".join(lines)


def _make_fake_matrix_repo(tmp_path: Path, child_exit: int) -> Path:
    root = tmp_path / "matrix_repo"
    tools = root / "tools"
    tools.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "reports").mkdir()
    (root / "config").mkdir()
    (root / "v2").mkdir()
    (root / "research").mkdir()

    shutil.copy2(SCRIPT, tools / "run_experiment_matrix.ps1")
    for helper in ("replay_contract.py", "replay_bundle.py", "evidence_manifest.py"):
        shutil.copy2(ROOT / "tools" / helper, tools / helper)
    (tools / "live_executor.py").write_text("# deterministic matrix fixture\n", encoding="utf-8")
    (root / "v2" / "risk_controls.py").write_text("# deterministic matrix fixture\n", encoding="utf-8")
    (root / "config" / "run.json").write_text("{}", encoding="utf-8")
    (root / "research" / "evidence_overrides.json").write_text(
        '{"schema_version":1,"overrides":{}}', encoding="utf-8"
    )
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
    child_logs = {
        "run_isolation_forest_blocking_paper_test.ps1": ["isolation_forest_shadow.csv"],
        "run_xgboost_shadow_outcome_paper_test.ps1": ["xgboost_signal_shadow.csv"],
        "run_survival_active_paper_test.ps1": ["survival_exit_shadow.csv"],
        "run_advanced_risk_shadow_paper_test.ps1": ["advanced_risk_shadow.csv"],
        "run_combined_shadow_paper_test.ps1": [
            "isolation_forest_shadow.csv",
            "xgboost_signal_shadow.csv",
            "survival_exit_shadow.csv",
            "advanced_risk_shadow.csv",
        ],
    }
    for script_name, log_names in child_logs.items():
        (tools / script_name).write_text(
            _fake_child_runbook(log_names=log_names, child_exit=child_exit),
            encoding="utf-8",
        )
    for name in (
        "experimental_shadow_report.py",
        "unified_experimental_report.py",
        "audit_xgboost_rejections.py",
    ):
        _write_fake_report_tool(tools / name)
    return tools / "run_experiment_matrix.ps1"


def _write_stale_log(path: Path) -> None:
    path.write_text(
        "header\nstale_row\n",
        encoding="utf-8",
    )


def _single_archive_dir(script: Path, mode: str) -> Path:
    logs = script.parents[1] / "logs"
    matches = sorted(logs.glob(f"matrix_{mode}_archive_*"))
    assert len(matches) == 1
    return matches[0]


def _latest_report(script: Path, mode: str, report_type: str) -> dict:
    reports = script.parents[1] / "reports"
    matches = sorted(reports.glob(f"matrix_{mode}_*_{report_type}.json"))
    assert matches
    return json.loads(matches[-1].read_text(encoding="utf-8-sig"))


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
    assert index["runs"][0]["replay_contract_status"] == "exact_matrix_snapshot"
    assert index["runs"][0]["replay_bundle_status"] == "exact_bundle"
    assert index["runs"][0]["replay_contract_digest"]
    assert index["runs"][0]["replay_bundle_digest"]


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


def test_matrix_fresh_logs_cleans_all_shadow_logs_before_child_runbook_modes(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)
    logs = script.parents[1] / "logs"
    for name in SHADOW_LOGS:
        _write_stale_log(logs / name)

    result = _run_matrix_script(
        script,
        "-Mode",
        "xgboost_shadow_outcome",
        "-Minutes",
        "5",
        "-FreshLogs",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archive = _single_archive_dir(script, "xgboost_shadow_outcome")
    for name in SHADOW_LOGS:
        archived = archive / name
        assert archived.exists()
        assert "stale_row" in archived.read_text(encoding="utf-8-sig")

    assert not (logs / "isolation_forest_shadow.csv").exists()
    assert not (logs / "survival_exit_shadow.csv").exists()
    assert not (logs / "advanced_risk_shadow.csv").exists()
    xgboost_log = logs / "xgboost_signal_shadow.csv"
    assert xgboost_log.exists()
    assert "xgboost_signal_shadow_new_row" in xgboost_log.read_text(encoding="utf-8-sig")
    assert "stale_row" not in xgboost_log.read_text(encoding="utf-8-sig")


def test_matrix_xgboost_fresh_logs_does_not_retain_isolation_shadow_rows(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)
    logs = script.parents[1] / "logs"
    _write_stale_log(logs / "isolation_forest_shadow.csv")

    result = _run_matrix_script(
        script,
        "-Mode",
        "xgboost_shadow_outcome",
        "-Minutes",
        "5",
        "-FreshLogs",
    )
    unified = _latest_report(script, "xgboost_shadow_outcome", "unified")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (logs / "isolation_forest_shadow.csv").exists()
    assert unified["shadow_rows"]["isolation_forest_shadow.csv"] == 0
    assert unified["shadow_rows"]["xgboost_signal_shadow.csv"] == 1


def test_matrix_combined_shadow_fresh_logs_still_writes_all_shadow_logs(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)
    logs = script.parents[1] / "logs"
    for name in SHADOW_LOGS:
        _write_stale_log(logs / name)

    result = _run_matrix_script(
        script,
        "-Mode",
        "combined_shadow",
        "-Minutes",
        "5",
        "-FreshLogs",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for name in SHADOW_LOGS:
        log_text = (logs / name).read_text(encoding="utf-8-sig")
        assert f"{Path(name).stem}_new_row" in log_text
        assert "stale_row" not in log_text


def test_matrix_fresh_logs_missing_logs_do_not_fail_child_runbook_modes(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)

    result = _run_matrix_script(
        script,
        "-Mode",
        "xgboost_shadow_outcome",
        "-Minutes",
        "5",
        "-FreshLogs",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Completed successfully" in result.stdout


def test_matrix_fresh_logs_archives_trade_logs_before_child_runbook_modes(tmp_path):
    script = _make_fake_matrix_repo(tmp_path, child_exit=0)
    logs = script.parents[1] / "logs"
    trade_logs = [
        "trades_paper_BTC.csv",
        "trades_closed.csv",
        "trades_closed_ETH.csv",
    ]
    for name in trade_logs:
        _write_stale_log(logs / name)

    result = _run_matrix_script(
        script,
        "-Mode",
        "xgboost_shadow_outcome",
        "-Minutes",
        "5",
        "-FreshLogs",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archive = _single_archive_dir(script, "xgboost_shadow_outcome")
    for name in trade_logs:
        assert not (logs / name).exists()
        archived = archive / name
        assert archived.exists()
        assert "stale_row" in archived.read_text(encoding="utf-8-sig")
