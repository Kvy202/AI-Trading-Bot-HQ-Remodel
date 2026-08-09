from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "tools" / "run_model_candidate_training.ps1"


def test_dry_run_prints_safety_banner_and_creates_nothing():
    protected = [
        ROOT / ".venv-model-training",
        ROOT / "reports/model_training_datasets",
        ROOT / "reports/model_candidate_confirmation",
        ROOT / "model_artifacts/candidates",
    ]
    before = [(path.exists(), path.stat().st_mtime_ns if path.exists() else None) for path in protected]
    completed = subprocess.run(
        [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(RUNBOOK), "-Model", "lstm", "-DryRun",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    for line in (
        "writer_started=false", "executor_started=false", "matrix_started=false",
        "exchange_execution_initialized=false", "orders_allowed=false",
        "incumbent_overwrite_allowed=false", "promotion_allowed=false",
        "live_activation_allowed=false", "dry_run=true", "mutation_performed=false",
    ):
        assert line in completed.stdout
    after = [(path.exists(), path.stat().st_mtime_ns if path.exists() else None) for path in protected]
    assert after == before


def test_runbook_exposes_every_required_operation_and_model_scope():
    source = RUNBOOK.read_text(encoding="utf-8-sig")
    for operation in (
        "DryRun", "Bootstrap", "CaptureDataset", "BuildDataset", "Train", "Evaluate",
        "LegacyRepairGate", "CaptureConfirmation", "ConfirmationGate",
    ):
        assert f"[switch]${operation}" in source
    assert '[ValidateSet("lstm", "tcn", "tx")]' in source


def test_runbook_visibly_separates_project_and_training_interpreters():
    source = RUNBOOK.read_text(encoding="utf-8-sig")
    assert '$ProjectPython = "python"' in source
    assert ".venv-model-training\\canonical\\Scripts\\python.exe" in source
    assert "capture_planned=project Python" in source
    assert "train_planned=training Python" in source


def test_runbook_contains_no_prohibited_process_invocation():
    source = RUNBOOK.read_text(encoding="utf-8-sig")
    for prohibited in (
        "live_writer.py", "live_executor.py", "run_experiment_matrix.ps1",
        "create_order", "place_order", "exchange.py",
    ):
        assert prohibited not in source
