from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "tools" / "run_runtime_stack_isolation.ps1"


def _tree_state(path: Path):
    if not path.exists():
        return None
    return sorted(
        (str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
        for item in path.rglob("*") if item.is_file()
    )


def test_runbook_declares_exact_commands_and_dedicated_root():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "[switch]$Bootstrap" in text
    assert "[switch]$Compare" in text
    assert "[switch]$DryRun" in text
    assert ".venv-runtime-isolation" in text
    assert "--no-deps" in text
    assert "--record-environment" in text
    assert "--validate-environment" in text


def test_runbook_never_installs_to_or_replaces_main_venv():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "-m venv $mainVenv" not in text
    assert "pip install" in text
    assert "& $stackPython -m pip install" in text
    assert "requirements.txt" not in text
    assert "Remove-Item" not in text


@pytest.mark.parametrize(
    "prohibited",
    [
        "live_writer.py", "live_executor.py", "run_experiment_matrix.ps1",
        "ccxt", "hyperliquid", "place_order", "approve-live", "activate-candidate",
    ],
)
def test_runbook_contains_no_trading_or_activation_command(prohibited):
    text = RUNBOOK.read_text(encoding="utf-8").lower()
    assert prohibited.lower() not in text


def test_dry_run_is_non_mutating_and_prints_safety_contract():
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    environment_root = ROOT / ".venv-runtime-isolation"
    report = ROOT / "reports" / "runtime_stack_isolation.json"
    before_tree = _tree_state(environment_root)
    before_report = (
        (report.stat().st_size, report.stat().st_mtime_ns) if report.exists() else None
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNBOOK), "-DryRun"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert _tree_state(environment_root) == before_tree
    after_report = (
        (report.stat().st_size, report.stat().st_mtime_ns) if report.exists() else None
    )
    assert after_report == before_report
    output = completed.stdout.lower()
    for line in (
        "writer_started=false", "executor_started=false", "matrix_started=false",
        "exchange_initialized=false", "orders_allowed=false",
        "main_environment_modified=false",
    ):
        assert line in output
    assert "scikit-learn==1.8.0" in output
    assert "isolation_report=" in output


def test_generated_outputs_and_environments_are_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".venv-runtime-isolation/" in ignored
    assert "reports/runtime_stack_isolation.json" in ignored
    assert "reports/runtime_stack_attribution.json" in ignored
    assert "reports/runtime_stack_decision.json" in ignored
    assert "reports/model_retraining_triage_phase23_1.json" in ignored
