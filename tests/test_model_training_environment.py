from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import model_training_environment as env


def _inventory(**overrides):
    value = {
        "python_version": "3.13.5",
        "packages": {**env.CANONICAL_NUMERICAL_VERSIONS, "torch": "2.11.0+cpu"},
        "torch_build": "cpu",
        "torch_cuda_version": None,
        "distributions": [*env.CANONICAL_NUMERICAL_VERSIONS, "torch", "pip"],
    }
    value.update(overrides)
    return value


def test_canonical_numerical_lock_and_policy_are_exact():
    assert env.file_digest(env.NUMERICAL_LOCK) == env.EXPECTED_NUMERICAL_LOCK_DIGEST
    assert env.parse_lock(env.NUMERICAL_LOCK) == env.CANONICAL_NUMERICAL_VERSIONS
    assert env.load_training_policy() == env.POLICY_TEMPLATE
    assert env.validate_phase24_evidence()["status"] == "phase24_candidate_training_allowed"


def test_training_lock_matches_dynamically_observed_main_cpu_torch():
    contract = env.training_contract()
    pins = env.parse_lock(env.TRAINING_LOCK)
    assert pins["torch"] == contract["torch"]["torch_version"]
    assert contract["torch"]["torch_build"] == "cpu"
    assert contract["torch"]["cuda_version"] is None
    assert contract["canonical_numerical_versions"] == env.CANONICAL_NUMERICAL_VERSIONS


def test_training_lock_digest_is_deterministic():
    assert env.training_contract()["contract_digest"] == env.training_contract()["contract_digest"]
    assert env.file_digest(env.TRAINING_LOCK) == env.file_digest(env.TRAINING_LOCK)


def test_environment_is_separate_and_exchange_packages_are_forbidden():
    assert env.TRAINING_ENV == env.BASE_DIR / ".venv-model-training" / "canonical"
    assert env.TRAINING_ENV != env.BASE_DIR / ".venv"
    assert ".venv-runtime-isolation" not in env.TRAINING_ENV.parts
    torch_contract = {"torch_version": "2.11.0+cpu", "torch_build": "cpu"}
    env.validate_training_inventory(_inventory(), torch_contract)
    bad = _inventory(distributions=["torch", "ccxt"])
    with pytest.raises(env.CandidateTrainingEnvironmentError, match="prohibited exchange"):
        env.validate_training_inventory(bad, torch_contract)


@pytest.mark.parametrize(
    ("package", "observed"),
    [("scikit-learn", "1.7.1"), ("numpy", "2.3.2")],
)
def test_wrong_canonical_package_is_rejected(package, observed):
    inventory = _inventory()
    inventory["packages"][package] = observed
    with pytest.raises(env.CandidateTrainingEnvironmentError, match="package mismatch"):
        env.validate_training_inventory(inventory, {"torch_version": "2.11.0+cpu", "torch_build": "cpu"})


def test_wrong_torch_version_or_build_is_rejected():
    inventory = _inventory()
    with pytest.raises(env.CandidateTrainingEnvironmentError, match="Torch does not exactly match"):
        env.validate_training_inventory(inventory, {"torch_version": "2.10.0+cpu", "torch_build": "cpu"})
    inventory["torch_build"] = "cu130"
    inventory["torch_cuda_version"] = "13.0"
    with pytest.raises(env.CandidateTrainingEnvironmentError, match="CPU build mismatch"):
        env.validate_training_inventory(inventory, {"torch_version": "2.11.0+cpu", "torch_build": "cpu"})


def test_inventory_only_is_read_only_and_reports_pending_without_bootstrap():
    existed = env.TRAINING_ENV.exists()
    result = env.inventory_only()
    assert result["operation"] == "inventory_only"
    assert result["mutated"] is False
    assert result["safety"] == {
        "writer_started": False,
        "executor_started": False,
        "matrix_started": False,
        "exchange_execution_initialized": False,
        "orders_allowed": False,
        "incumbent_overwrite_allowed": False,
        "promotion_allowed": False,
        "live_activation_allowed": False,
    }
    assert env.TRAINING_ENV.exists() is existed


def test_policy_rejects_missing_or_weakened_fields(tmp_path):
    policy = env.load_training_policy()
    policy["minimum_validation_auc_pooled"] = 0.50
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(env.CandidateTrainingEnvironmentError, match="policy mismatch"):
        env.load_training_policy(path)


def test_candidate_paths_cannot_alias_incumbents():
    safe = env.assert_safe_candidate_directory(env.BASE_DIR / "model_artifacts/candidates/example")
    assert safe.name == "example"
    with pytest.raises(env.CandidateTrainingEnvironmentError):
        env.assert_safe_candidate_directory(env.BASE_DIR / "model_artifacts/dl_lstm_latest.pt")
    assert set(env.incumbent_hashes()) == {
        f"model_artifacts/{name}"
        for name in (
            "dl_lstm_latest.pt", "scaler_lstm_latest.joblib",
            "dl_tcn_latest.pt", "scaler_tcn_latest.joblib",
            "dl_tx_latest.pt", "scaler_tx_latest.joblib",
            "dl_adv_latest.pt", "scaler_adv_latest.joblib",
        )
    }
