from __future__ import annotations

import subprocess
from pathlib import Path


RUNBOOK = Path("tools/run_model_runtime_repro.ps1")


def test_runbook_keeps_main_and_reproduction_environments_separate():
    source = RUNBOOK.read_text(encoding="utf-8")
    assert '.venv-repro-sklearn180' in source
    assert 'Join-Path $RepoRoot ".venv\\Scripts\\python.exe"' in source
    assert "-m pip install" in source
    assert "$ReproPython -m pip install" in source
    assert "$MainPython -m pip install" not in source
    assert "requirements.txt" not in source.replace("model_repro_sklearn180.txt", "")


def test_runbook_requires_exact_versions_and_python_major_minor():
    source = RUNBOOK.read_text(encoding="utf-8")
    for value in ("1.8.0", "2.3.3", "1.16.2", "1.5.2", "3.6.0"):
        assert value in source
    assert "Python major/minor mismatch" in source
    assert "refusing reuse" in source


def test_runbook_has_no_process_or_order_activation_commands():
    source = RUNBOOK.read_text(encoding="utf-8").lower()
    for forbidden in ("start-process", "place_order", "run_experiment_matrix", "live_writer.py", "live_executor.py"):
        assert forbidden not in source


def test_dry_run_creates_no_environment_or_report():
    environment = Path(".venv-repro-sklearn180")
    report = Path("reports/model_runtime_reproducibility.json")
    before = {
        "env_exists": environment.exists(), "report_exists": report.exists(),
        "env_mtime": environment.stat().st_mtime_ns if environment.exists() else None,
        "report_mtime": report.stat().st_mtime_ns if report.exists() else None,
    }
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RUNBOOK), "-DryRun"],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    after = {
        "env_exists": environment.exists(), "report_exists": report.exists(),
        "env_mtime": environment.stat().st_mtime_ns if environment.exists() else None,
        "report_mtime": report.stat().st_mtime_ns if report.exists() else None,
    }
    assert after == before


def test_generated_reports_and_candidate_directories_are_ignored():
    paths = [
        "reports/model_runtime_reproducibility.json", "reports/model_failure_triage.json",
        "reports/training_lineage_audit.json", "reports/model_retraining_triage.json",
        "reports/model_retraining_specification.json", "model_artifacts/candidates/example/model.pt",
        ".venv-repro-sklearn180/Scripts/python.exe",
    ]
    completed = subprocess.run(["git", "check-ignore", *paths], capture_output=True, text=True)
    assert completed.returncode == 0
    assert len(completed.stdout.splitlines()) == len(paths)


def test_reproduction_requirements_are_scaler_only_and_exact():
    lines = Path("requirements/model_repro_sklearn180.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "numpy==2.3.3", "scipy==1.16.2", "scikit-learn==1.8.0",
        "joblib==1.5.2", "threadpoolctl==3.6.0",
    ]
    assert not any("torch" in line.lower() or "ccxt" in line.lower() or "hyperliquid" in line.lower() for line in lines)


def test_runbook_uses_stdin_for_windows_powershell_python_inventory():
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "$Code | & $PythonPath -" in source
    assert "-c $Code" not in source


def test_interrupted_valid_bootstrap_can_recover_without_package_install():
    source = RUNBOOK.read_text(encoding="utf-8")
    recovery = source.index("if ($null -eq $Marker)")
    install = source.index("-m pip install")
    assert recovery < install
    assert "Assert-ReproductionContract $Existing" in source[:recovery]


def test_native_stderr_is_captured_without_overriding_process_exit_code():
    source = RUNBOOK.read_text(encoding="utf-8")
    assert '$ErrorActionPreference = "Continue"' in source
    assert "$CompareExitCode = $LASTEXITCODE" in source
    assert "2> $CompareStderr" in source
