"""Phase 24 canonical candidate-training environment and safety contracts.

Inventory is read-only.  Bootstrap is deliberately explicit and creates only
``.venv-model-training/canonical``; it never installs into the project Python.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_ENV = BASE_DIR / ".venv-model-training" / "canonical"
TRAINING_PYTHON = TRAINING_ENV / "Scripts" / "python.exe"
TRAINING_LOCK = BASE_DIR / "requirements" / "model_training_canonical.txt"
NUMERICAL_LOCK = BASE_DIR / "requirements" / "model_numeric_canonical.txt"
TRAINING_POLICY = BASE_DIR / "research" / "model_candidate_training_policy.json"
CANONICAL_RUNTIME = BASE_DIR / "research" / "canonical_model_runtime.json"
PHASE22_BUNDLE = BASE_DIR / "reports" / "model_alignment_bundles" / "history_5m_final"
ENVIRONMENT_MANIFEST = TRAINING_ENV / ".model-training-manifest.json"
INCUMBENT_INVENTORY = BASE_DIR / "reports" / "model_candidate_incumbent_inventory.json"

EXPECTED_NUMERICAL_LOCK_DIGEST = "9fb582280151766c103fe9723669dd1a86313223ba15b826845a8529569546f0"
EXPECTED_PHASE22_BUNDLE_DIGEST = "43597484148a569c4827ff1f1378048264e9a46878ba846efccbb81ee9362843"
EXPECTED_PHASE22_ALIGNMENT_DIGEST = "5f167c7d41e24ca048dd1ea82c3d66ebfd32fed7ed176caddf084a18d7412d4f"
CANONICAL_NUMERICAL_VERSIONS = {
    "numpy": "2.3.3",
    "scipy": "1.16.2",
    "joblib": "1.5.2",
    "scikit-learn": "1.8.0",
    "threadpoolctl": "3.6.0",
}
PROHIBITED_DISTRIBUTIONS = {
    "ccxt", "eth-account", "hyperliquid", "hyperliquid-python-sdk",
    "hyperliquid-python", "web3",
}
REQUIRED_EVIDENCE = (
    "research/canonical_model_runtime.json",
    "requirements/model_numeric_canonical.txt",
    "research/model_retraining_policy.json",
    "research/model_candidate_registry.json",
    "reports/model_retraining_specification_phase23_1.json",
    "reports/model_retraining_triage_phase23_1.json",
    "reports/runtime_stack_attribution.json",
    "reports/runtime_stack_decision.json",
)

POLICY_TEMPLATE: dict[str, Any] = {
    "schema_version": 1,
    "timeframe": "5m",
    "sequence_length": 64,
    "feature_count": 27,
    "required_symbols": ["BTCUSDT", "ETHUSDT"],
    "target_raw_bars_per_symbol": 50000,
    "minimum_usable_labeled_rows_per_symbol": 25000,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "minimum_validation_auc_pooled": 0.55,
    "minimum_validation_auc_per_symbol": 0.52,
    "minimum_internal_test_auc_pooled": 0.55,
    "minimum_internal_test_auc_per_symbol": 0.52,
    "flat_output_std_threshold": 0.002,
    "flat_window": 30,
    "extreme_low_threshold": 0.05,
    "extreme_high_threshold": 0.95,
    "extreme_consecutive_limit": 20,
    "maximum_missing_rate": 0.05,
    "training_seeds": [24001, 24002, 24003],
    "legacy_repair_bars_per_symbol": 120,
    "confirmation_target_bars_per_symbol": 576,
    "confirmation_minimum_bars_per_symbol": 288,
    "require_train_only_scaler_fit": True,
    "require_chronological_split": True,
    "require_purge": True,
    "require_per_symbol_sequence_construction": True,
    "require_phase22_excluded_from_training": True,
    "require_confirmation_captured_after_candidate_freeze": True,
    "require_deterministic_training": True,
    "candidate_overwrite_allowed": False,
    "incumbent_overwrite_allowed": False,
    "promotion_allowed": False,
    "live_activation_allowed": False,
}


class CandidateTrainingEnvironmentError(ValueError):
    """A Phase 24 environment, evidence, or immutable-artifact contract failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def git_commit(repository: Path | str = BASE_DIR) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().lower()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else "unknown"
    except Exception:
        return "unknown"


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateTrainingEnvironmentError(f"required evidence missing: {path.relative_to(BASE_DIR)}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CandidateTrainingEnvironmentError(f"JSON object required: {path.relative_to(BASE_DIR)}")
    return value


def load_training_policy(path: Path | str = TRAINING_POLICY) -> dict[str, Any]:
    value = _load_json(Path(path))
    if set(value) != set(POLICY_TEMPLATE):
        raise CandidateTrainingEnvironmentError("candidate training policy fields are not exact")
    for name, expected in POLICY_TEMPLATE.items():
        observed = value[name]
        if type(observed) is not type(expected) or observed != expected:
            raise CandidateTrainingEnvironmentError(f"candidate training policy mismatch: {name}")
    if abs(value["train_fraction"] + value["validation_fraction"] + value["test_fraction"] - 1.0) > 1e-12:
        raise CandidateTrainingEnvironmentError("candidate split fractions must sum to one")
    return value


def parse_lock(path: Path | str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            raise CandidateTrainingEnvironmentError(f"unversioned requirement in {Path(path).name}: {line}")
        name, version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        if normalized in pins:
            raise CandidateTrainingEnvironmentError(f"duplicate requirement: {normalized}")
        pins[normalized] = version.strip()
    return pins


def main_torch_contract(python: Path | str = sys.executable) -> dict[str, Any]:
    code = r'''
import importlib.metadata as m, json, torch
d = m.distribution("torch")
print(json.dumps({
  "torch_version": str(torch.__version__),
  "distribution_version": str(d.version),
  "torch_build": str(torch.__version__).split("+", 1)[1] if "+" in str(torch.__version__) else "unsuffixed",
  "cuda_version": torch.version.cuda,
  "git_version": torch.version.git_version,
  "installer": (d.read_text("INSTALLER") or "unknown").strip(),
  "direct_url": d.read_text("direct_url.json"),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code], cwd=BASE_DIR, capture_output=True, text=True, timeout=60
    )
    if completed.returncode:
        raise CandidateTrainingEnvironmentError("current main Torch cannot be inventoried")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result.get("cuda_version") is not None or result.get("torch_build") != "cpu":
        raise CandidateTrainingEnvironmentError("current main Torch is not an exact CPU build")
    return result


def validate_phase24_evidence(repository: Path | str = BASE_DIR) -> dict[str, Any]:
    root = Path(repository)
    for relative in REQUIRED_EVIDENCE:
        if not (root / relative).is_file():
            raise CandidateTrainingEnvironmentError(f"required evidence missing: {relative}")
    if file_digest(root / "requirements/model_numeric_canonical.txt") != EXPECTED_NUMERICAL_LOCK_DIGEST:
        raise CandidateTrainingEnvironmentError("canonical numerical lock digest mismatch")
    canonical = _load_json(root / "research/canonical_model_runtime.json")
    decision = _load_json(root / "reports/runtime_stack_decision.json")
    triage = _load_json(root / "reports/model_retraining_triage_phase23_1.json")
    specification = _load_json(root / "reports/model_retraining_specification_phase23_1.json")
    attribution = _load_json(root / "reports/runtime_stack_attribution.json")
    retraining_policy = _load_json(root / "research/model_retraining_policy.json")
    candidate_registry = _load_json(root / "research/model_candidate_registry.json")
    integrity_reports = (
        (canonical, "decision_digest"),
        (decision, "decision_digest"),
        (attribution, "attribution_digest"),
        (triage, "triage_digest"),
        (specification, "specification_digest"),
    )
    for report, digest_field in integrity_reports:
        observed = json_digest({
            key: value for key, value in report.items()
            if key not in {"generated_at", digest_field}
        })
        if report.get(digest_field) != observed:
            raise CandidateTrainingEnvironmentError(f"Phase 24 evidence digest mismatch: {digest_field}")
    if triage.get("policy_digest") != json_digest(retraining_policy):
        raise CandidateTrainingEnvironmentError("Phase 23.1 retraining policy digest mismatch")
    if triage.get("candidate_registry_digest") != json_digest(candidate_registry):
        raise CandidateTrainingEnvironmentError("tracked candidate registry digest mismatch")
    if decision.get("attribution_digest") != attribution.get("attribution_digest"):
        raise CandidateTrainingEnvironmentError("runtime attribution/decision evidence mismatch")
    if canonical != decision:
        raise CandidateTrainingEnvironmentError("canonical runtime and selected runtime decision differ")
    for source in (canonical, decision):
        required = {
            "selected_stack_id": "serialized_full_stack",
            "behavioral_status": "behaviorally_reproducible",
            "main_runtime_migration_allowed": False,
            "live_activation_allowed": False,
            "phase24_candidate_training_allowed": True,
            "canonical_lock_digest": EXPECTED_NUMERICAL_LOCK_DIGEST,
        }
        for name, expected in required.items():
            if source.get(name) != expected:
                raise CandidateTrainingEnvironmentError(f"Phase 24 evidence mismatch: {name}")
        if source.get("package_versions") != CANONICAL_NUMERICAL_VERSIONS:
            raise CandidateTrainingEnvironmentError("canonical numerical package contract mismatch")
        if source.get("source_bundle_digest") != EXPECTED_PHASE22_BUNDLE_DIGEST:
            raise CandidateTrainingEnvironmentError("Phase 22 bundle digest evidence mismatch")
        if source.get("source_alignment_digest") != EXPECTED_PHASE22_ALIGNMENT_DIGEST:
            raise CandidateTrainingEnvironmentError("Phase 22 alignment digest evidence mismatch")
    if triage.get("phase24_allowed") is not True or triage.get("promotion_allowed") is not False:
        raise CandidateTrainingEnvironmentError("Phase 23.1 triage does not authorize candidate-only work")
    if triage.get("live_or_blocking_use_approved") is not False:
        raise CandidateTrainingEnvironmentError("Phase 23.1 unexpectedly approves live/blocking use")
    expected_actions = {
        "adv": "retain_incumbent_shadow_control",
        "lstm": "retrain_required",
        "tcn": "symbol_specific_retraining_required",
        "tx": "symbol_specific_retraining_required",
    }
    observed_actions = {
        kind: value.get("primary_action")
        for kind, value in triage.get("model_decisions", {}).items()
    }
    if any(observed_actions.get(kind) != action for kind, action in expected_actions.items()):
        raise CandidateTrainingEnvironmentError("Phase 23.1 model decisions mismatch")
    if specification.get("training_allowed_by_this_specification") is not True:
        raise CandidateTrainingEnvironmentError("Phase 23.1 training specification is not complete")
    if specification.get("promotion_implemented") is not False:
        raise CandidateTrainingEnvironmentError("Phase 23.1 unexpectedly implements promotion")
    manifest = _load_json(root / "reports/model_alignment_bundles/history_5m_final/bundle_manifest.json")
    if manifest.get("bundle_digest") != EXPECTED_PHASE22_BUNDLE_DIGEST:
        raise CandidateTrainingEnvironmentError("legacy repair bundle known digest mismatch")
    if json_digest({key: manifest[key] for key in sorted(manifest) if key != "bundle_digest"}) != EXPECTED_PHASE22_BUNDLE_DIGEST:
        raise CandidateTrainingEnvironmentError("legacy repair bundle manifest integrity failure")
    for name, digest in manifest.get("bundle_file_digests", {}).items():
        path = root / "reports/model_alignment_bundles/history_5m_final" / name
        if not path.is_file() or file_digest(path) != digest:
            raise CandidateTrainingEnvironmentError(f"legacy repair bundle file mismatch: {name}")
    alignment = _load_json(root / "reports/model_alignment_report_final.json")
    if alignment.get("alignment_digest") != EXPECTED_PHASE22_ALIGNMENT_DIGEST:
        raise CandidateTrainingEnvironmentError("Phase 22 alignment report digest mismatch")
    if alignment.get("alignment_digest") != json_digest({
        key: value for key, value in alignment.items()
        if key not in {"generated_at", "alignment_digest"}
    }):
        raise CandidateTrainingEnvironmentError("Phase 22 alignment report integrity failure")
    return {
        "status": "phase24_candidate_training_allowed",
        "canonical_numerical_lock_digest": EXPECTED_NUMERICAL_LOCK_DIGEST,
        "legacy_repair_bundle_digest": EXPECTED_PHASE22_BUNDLE_DIGEST,
        "phase22_alignment_digest": EXPECTED_PHASE22_ALIGNMENT_DIGEST,
        "specification_digest": specification.get("specification_digest"),
        "triage_digest": triage.get("triage_digest"),
    }


def interpreter_inventory(python: Path | str) -> dict[str, Any]:
    code = r'''
import hashlib, importlib.metadata as m, json, platform
packages = {}
for key, module, dist in (
  ("numpy", "numpy", "numpy"), ("scipy", "scipy", "scipy"),
  ("joblib", "joblib", "joblib"), ("scikit-learn", "sklearn", "scikit-learn"),
  ("threadpoolctl", "threadpoolctl", "threadpoolctl"), ("torch", "torch", "torch"),
):
  try:
    imported = __import__(module)
    packages[key] = str(imported.__version__)
  except Exception:
    packages[key] = None
dists = sorted({str(d.metadata.get("Name") or "").lower().replace("_", "-") for d in m.distributions()})
import torch
print(json.dumps({
  "python_version": platform.python_version(), "packages": packages,
  "torch_build": str(torch.__version__).split("+", 1)[1] if "+" in str(torch.__version__) else "unsuffixed",
  "torch_cuda_version": torch.version.cuda, "distributions": dists,
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code], cwd=BASE_DIR, capture_output=True, text=True, timeout=60
    )
    if completed.returncode:
        raise CandidateTrainingEnvironmentError("unable to inventory candidate-training interpreter")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def validate_training_inventory(inventory: Mapping[str, Any], torch_contract: Mapping[str, Any]) -> None:
    packages = inventory.get("packages", {})
    for name, expected in CANONICAL_NUMERICAL_VERSIONS.items():
        if packages.get(name) != expected:
            raise CandidateTrainingEnvironmentError(
                f"candidate training package mismatch: {name}={packages.get(name)!r}, expected {expected!r}"
            )
    if packages.get("torch") != torch_contract.get("torch_version"):
        raise CandidateTrainingEnvironmentError("candidate training Torch does not exactly match main Torch")
    if inventory.get("torch_build") != torch_contract.get("torch_build") or inventory.get("torch_cuda_version") is not None:
        raise CandidateTrainingEnvironmentError("candidate training Torch CPU build mismatch")
    installed = set(inventory.get("distributions", []))
    prohibited = sorted(installed & PROHIBITED_DISTRIBUTIONS)
    if prohibited:
        raise CandidateTrainingEnvironmentError(f"prohibited exchange package(s) present: {prohibited}")


def _normalized_pip_freeze(python: Path | str) -> str:
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"], cwd=BASE_DIR,
        capture_output=True, text=True, timeout=60,
    )
    if completed.returncode:
        raise CandidateTrainingEnvironmentError("unable to capture candidate-training pip freeze")
    return "\n".join(sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())) + "\n"


def training_contract(current_python: Path | str = sys.executable) -> dict[str, Any]:
    numerical_pins = parse_lock(NUMERICAL_LOCK)
    if numerical_pins != CANONICAL_NUMERICAL_VERSIONS:
        raise CandidateTrainingEnvironmentError("canonical numerical lock contents mismatch")
    training_pins = parse_lock(TRAINING_LOCK)
    for name, version in CANONICAL_NUMERICAL_VERSIONS.items():
        if training_pins.get(name) != version:
            raise CandidateTrainingEnvironmentError(f"training lock numerical mismatch: {name}")
    torch_contract = main_torch_contract(current_python)
    if training_pins.get("torch") != torch_contract["torch_version"]:
        raise CandidateTrainingEnvironmentError(
            "training lock Torch pin does not exactly match the dynamically observed main CPU build"
        )
    value = {
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "canonical_numerical_versions": CANONICAL_NUMERICAL_VERSIONS,
        "torch": torch_contract,
        "torch_cpu_index": "https://download.pytorch.org/whl/cpu",
        "canonical_numerical_lock_digest": file_digest(NUMERICAL_LOCK),
        "training_lock_digest": file_digest(TRAINING_LOCK),
    }
    value["contract_digest"] = json_digest(value)
    return value


def inventory_only(current_python: Path | str = sys.executable) -> dict[str, Any]:
    evidence = validate_phase24_evidence()
    policy = load_training_policy()
    contract = training_contract(current_python)
    if not TRAINING_PYTHON.is_file():
        environment = {
            "status": "candidate_training_environment_pending",
            "environment_path": ".venv-model-training/canonical",
            "created": False,
        }
    else:
        try:
            observed = interpreter_inventory(TRAINING_PYTHON)
            validate_training_inventory(observed, contract["torch"])
            if not ENVIRONMENT_MANIFEST.is_file():
                raise CandidateTrainingEnvironmentError("candidate-training environment manifest missing")
            manifest = _load_json(ENVIRONMENT_MANIFEST)
            recorded = manifest.get("manifest_digest")
            calculated = json_digest({
                k: v for k, v in manifest.items() if k not in {"created_at", "manifest_digest"}
            })
            if recorded != calculated or manifest.get("contract_digest") != contract["contract_digest"]:
                raise CandidateTrainingEnvironmentError("candidate-training environment manifest mismatch")
            environment = {
                "status": "candidate_training_environment_ready",
                "environment_path": ".venv-model-training/canonical",
                "created": True,
                "inventory": observed,
                "environment_digest": recorded,
                "pip_freeze_digest": manifest.get("pip_freeze_digest"),
            }
        except Exception as exc:
            environment = {
                "status": "candidate_training_environment_pending",
                "environment_path": ".venv-model-training/canonical",
                "created": True,
                "reason": str(exc),
            }
    return {
        "schema_version": 1,
        "operation": "inventory_only",
        "mutated": False,
        "evidence": evidence,
        "policy_digest": json_digest(policy),
        "contract": contract,
        "environment": environment,
        "safety": {
            "writer_started": False, "executor_started": False, "matrix_started": False,
            "exchange_execution_initialized": False, "orders_allowed": False,
            "incumbent_overwrite_allowed": False, "promotion_allowed": False,
            "live_activation_allowed": False,
        },
    }


def resolved_incumbent_artifacts(repository: Path | str = BASE_DIR) -> list[Path]:
    root = Path(repository).resolve()
    snapshot = _load_json(root / "reports/model_alignment_bundles/history_5m_final/model_serving_snapshot.json")
    paths: list[Path] = []
    for entry in snapshot.get("model_entries", []):
        for field in ("model_filename", "scaler_filename"):
            raw = entry.get(field)
            if not raw:
                raise CandidateTrainingEnvironmentError(f"serving snapshot missing {field}")
            path = (root / str(raw)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise CandidateTrainingEnvironmentError("incumbent artifact resolves outside repository") from exc
            if not path.is_file() or path.is_symlink():
                raise CandidateTrainingEnvironmentError(f"incumbent artifact unavailable or symlinked: {path}")
            paths.append(path)
    return sorted(set(paths))


def incumbent_hashes(repository: Path | str = BASE_DIR) -> dict[str, str]:
    root = Path(repository).resolve()
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in resolved_incumbent_artifacts(root)
    }


def record_incumbent_inventory(path: Path | str = INCUMBENT_INVENTORY) -> dict[str, Any]:
    target = Path(path)
    current = incumbent_hashes()
    protected_files = {}
    for relative in (".env", "config/run.json", "features.py", "research/model_candidate_registry.json"):
        candidate = BASE_DIR / relative
        protected_files[relative] = file_digest(candidate) if candidate.is_file() else None
    if target.exists():
        previous = _load_json(target)
        if previous.get("incumbent_artifacts") != current or previous.get("protected_files") != protected_files:
            raise CandidateTrainingEnvironmentError("incumbent/protected artifact inventory changed")
        return previous
    result = {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "serving_snapshot_digest": _load_json(
            PHASE22_BUNDLE / "model_serving_snapshot.json"
        ).get("snapshot_digest"),
        "incumbent_artifacts": current,
        "protected_files": protected_files,
        "incumbent_overwrite_allowed": False,
    }
    result["inventory_digest"] = json_digest({k: v for k, v in result.items() if k != "recorded_at"})
    atomic_write_json(target, result)
    return result


def verify_incumbent_inventory(path: Path | str = INCUMBENT_INVENTORY) -> dict[str, Any]:
    baseline = _load_json(Path(path))
    recorded_digest = baseline.get("inventory_digest")
    observed_digest = json_digest({
        k: v for k, v in baseline.items() if k not in {"recorded_at", "inventory_digest"}
    })
    if recorded_digest != observed_digest:
        raise CandidateTrainingEnvironmentError("incumbent inventory digest mismatch")
    if baseline.get("incumbent_artifacts") != incumbent_hashes():
        raise CandidateTrainingEnvironmentError("incumbent artifact digest changed")
    for relative, expected in baseline.get("protected_files", {}).items():
        candidate = BASE_DIR / relative
        observed = file_digest(candidate) if candidate.is_file() else None
        if observed != expected:
            raise CandidateTrainingEnvironmentError(f"protected file changed: {relative}")
    return baseline


def assert_safe_candidate_directory(path: Path | str) -> Path:
    target = Path(path).resolve()
    root = (BASE_DIR / "model_artifacts" / "candidates").resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise CandidateTrainingEnvironmentError("candidate output is outside the candidate root") from exc
    if not relative.parts or len(relative.parts) != 1:
        raise CandidateTrainingEnvironmentError("candidate output must be one immutable candidate directory")
    if target.is_symlink() or any(parent.is_symlink() for parent in (target, *target.parents) if parent != BASE_DIR.parent):
        raise CandidateTrainingEnvironmentError("candidate output path may not contain symlinks")
    incumbents = {path.resolve() for path in resolved_incumbent_artifacts()}
    forbidden_names = {"dl_lstm_latest.pt", "dl_tcn_latest.pt", "dl_tx_latest.pt", "dl_adv_latest.pt",
                       "scaler_latest.joblib", "scaler_lstm_latest.joblib",
                       "scaler_tcn_latest.joblib", "scaler_tx_latest.joblib", "scaler_adv_latest.joblib"}
    for name in ("model.pt", "scaler.joblib"):
        candidate = (target / name).resolve()
        if candidate in incumbents or candidate.name in forbidden_names:
            raise CandidateTrainingEnvironmentError("candidate write target aliases an incumbent")
    return target


def bootstrap_environment(current_python: Path | str = sys.executable) -> dict[str, Any]:
    validate_phase24_evidence()
    load_training_policy()
    contract = training_contract(current_python)
    record_incumbent_inventory()
    if TRAINING_PYTHON.is_file():
        observed = interpreter_inventory(TRAINING_PYTHON)
        validate_training_inventory(observed, contract["torch"])
        verify_incumbent_inventory()
        return {"status": "reused", "inventory": observed}
    if TRAINING_ENV.exists():
        raise CandidateTrainingEnvironmentError("partial candidate-training environment exists; refusing overwrite")
    TRAINING_ENV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(current_python), "-m", "venv", str(TRAINING_ENV)], cwd=BASE_DIR, check=True)
    subprocess.run(
        [str(TRAINING_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
         "--no-input", "-r", str(TRAINING_LOCK)], cwd=BASE_DIR, check=True,
    )
    observed = interpreter_inventory(TRAINING_PYTHON)
    validate_training_inventory(observed, contract["torch"])
    freeze = _normalized_pip_freeze(TRAINING_PYTHON)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "environment_path": ".venv-model-training/canonical",
        "python_version": observed["python_version"],
        "torch_version": observed["packages"]["torch"],
        "torch_build": observed["torch_build"],
        "numpy_version": observed["packages"]["numpy"],
        "scipy_version": observed["packages"]["scipy"],
        "sklearn_version": observed["packages"]["scikit-learn"],
        "joblib_version": observed["packages"]["joblib"],
        "threadpoolctl_version": observed["packages"]["threadpoolctl"],
        "pip_freeze": freeze.splitlines(),
        "pip_freeze_digest": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "canonical_numerical_lock_digest": contract["canonical_numerical_lock_digest"],
        "training_lock_digest": contract["training_lock_digest"],
        "contract_digest": contract["contract_digest"],
        "torch_package_source": {
            "index": contract["torch_cpu_index"],
            "main_installer": contract["torch"].get("installer"),
            "main_direct_url": contract["torch"].get("direct_url"),
        },
        "prohibited_exchange_packages_present": [],
    }
    manifest["manifest_digest"] = json_digest({
        key: value for key, value in manifest.items() if key != "created_at"
    })
    atomic_write_json(ENVIRONMENT_MANIFEST, manifest)
    verify_incumbent_inventory()
    return {"status": "created", "inventory": observed, "environment_digest": manifest["manifest_digest"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory-only", action="store_true")
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--validate-environment", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.inventory_only:
            result = inventory_only()
        elif args.bootstrap:
            result = bootstrap_environment()
        else:
            result = inventory_only()
            if result["environment"]["status"] != "candidate_training_environment_ready":
                raise CandidateTrainingEnvironmentError(result["environment"].get("reason", "environment pending"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "candidate_training_environment_pending",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
