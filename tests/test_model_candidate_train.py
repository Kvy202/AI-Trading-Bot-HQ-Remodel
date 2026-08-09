from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from tools import model_candidate_train as train


def _validation(auc=0.60, loss=0.5, btc=0.56, eth=0.57):
    return {
        "split": "validation",
        "pooled": {"auc": auc, "classification_loss": loss, "nonfinite_outputs": 0},
        "per_symbol": {
            "BTCUSDT": {"auc": btc, "class_counts": {"0": 10, "1": 10}, "nonfinite_outputs": 0},
            "ETHUSDT": {"auc": eth, "class_counts": {"0": 10, "1": 10}, "nonfinite_outputs": 0},
        },
    }


def _identity(seed=24001, dataset="a" * 64, learning_rate=0.001, objective="3" * 64, balance="4" * 64):
    manifest = {
        "dataset_digest": dataset,
        "feature_digest": "b" * 64,
        "label_digest": "c" * 64,
        "split_digest": "d" * 64,
    }
    return train.candidate_identity(
        "lstm", architecture_config=train.ARCHITECTURE_DEFAULTS["lstm"],
        dataset_manifest=manifest, scaler_digest="e" * 64,
        training_config={"learning_rate": learning_rate}, seed=seed,
        numerical_lock_digest="f" * 64, training_environment_digest="1" * 64,
        training_code_digest="2" * 64,
        objective_contract_digest=objective,
        balance_contract_digest=balance,
    )


def test_resolved_objective_optimizes_all_heads_and_legacy_mode_stays_research_only():
    contract = train.training_objective_contract(
        target_scales={"ret_target_scale": 0.01, "rv_target_scale": 0.02},
        objective_contract_digest="f" * 64,
    )
    assert contract["formula"] == "fixed_weighted_resolved_candidate_loss"
    assert contract["selected_loss_formulation"] == "normalized_mse_fixed"
    assert contract["ret_reg_optimized"] is True
    assert contract["rv_reg_optimized"] is True
    assert contract["auxiliary_head_training_status"] == "auxiliary_heads_optimized_under_resolved_candidate_objective"
    legacy = train.training_objective_contract(train.LEGACY_OBJECTIVE_NAME)
    assert legacy["loss"] == "CrossEntropyLoss"
    assert legacy["optimized_output"] == "ret_cls_logits"
    assert legacy["ret_reg_optimized"] is False
    assert legacy["rv_reg_optimized"] is False
    assert legacy["candidate_finalization_allowed"] is False
    assert legacy["confirmation_health_pass_allowed"] is False
    source = inspect.getsource(train.train_classification_candidate)
    assert "resolved_candidate_loss" in source
    assert "clip_grad_norm_" in source


def test_downstream_use_sets_auxiliary_head_promotion_blocker():
    audit = train.downstream_auxiliary_head_audit()
    assert audit["candidate_auxiliary_head_promotion_blocker"] is True
    assert audit["objective_contract_blocker"] is False
    assert audit["candidate_auxiliary_health_blocker"] == "unverified"
    legacy = next(row for row in audit["consumers"] if row["path"] == "trade_multi_bitget.py")
    assert legacy["rv_hat_used"] is True and legacy["classification"] == "legacy_only"


@pytest.mark.parametrize("kind", train.ALLOWED_KINDS)
def test_architecture_contract_matches_incumbent_shapes_and_existing_defaults(kind):
    contract = train.architecture_contract(kind)
    assert contract["constructor"] == train.ARCHITECTURE_DEFAULTS[kind]
    assert contract["architecture_mathematics_modified"] is False
    assert contract["incumbent_state_shape_digest"]
    assert len(train.make_candidate_model(kind).state_dict()) > 0


def test_adv_retraining_is_prohibited():
    with pytest.raises(train.ModelCandidateTrainingError, match="ADV"):
        train.make_candidate_model("adv")


def test_candidate_identity_is_deterministic_and_changes_with_inputs():
    first = _identity()
    assert first == _identity()
    assert first[0].startswith("lstm_5m_aaaaaaaa_") and first[0].endswith("_s24001")
    assert first[0] != _identity(seed=24002)[0]
    assert first[0] != _identity(dataset="9" * 64)[0]
    assert first[0] != _identity(learning_rate=0.002)[0]
    assert first[0] != _identity(objective="4" * 64)[0]
    assert first[0] != _identity(balance="5" * 64)[0]
    assert "C:\\" not in first[1]["identity_digest"]


def test_seed_selection_uses_validation_only_with_auc_loss_seed_tiebreaks():
    results = [
        {"seed": 24003, "validation": _validation(0.60, 0.45), "internal_test": {"auc": 1.0}},
        {"seed": 24002, "validation": _validation(0.61, 0.60), "internal_test": {"auc": 0.0}},
        {"seed": 24001, "validation": _validation(0.61, 0.60), "legacy_repair": {"passed": False}},
    ]
    selected = train.select_validation_seed(results)
    assert selected["selected_seed"] == 24001
    assert selected["selection_basis"] == "validation_only"
    assert selected["internal_test_consulted"] is False
    assert selected["legacy_repair_set_consulted"] is False
    assert selected["confirmation_set_consulted"] is False


def test_validation_gates_require_pooled_and_each_symbol_and_both_classes():
    assert train.validation_gate(_validation())["passed"] is True
    assert train.validation_gate(_validation(auc=0.54))["status"] == "validation_failed"
    assert train.validation_gate(_validation(btc=0.51))["status"] == "validation_failed"
    value = _validation()
    value["per_symbol"]["ETHUSDT"]["class_counts"] = {"0": 20}
    assert "ETHUSDT_validation_requires_both_classes" in train.validation_gate(value)["reasons"]
    with pytest.raises(train.ModelCandidateTrainingError, match="validation_failed"):
        train.select_validation_seed([{"seed": 24001, "validation": _validation(auc=0.50)}])


def test_internal_test_gates_pooled_symbols_finiteness_and_repeat():
    metrics = _validation()
    metrics["split"] = "internal_test"
    metrics["deterministic_repeat_passed"] = True
    assert train.internal_test_gate(metrics)["passed"] is True
    metrics["per_symbol"]["BTCUSDT"]["auc"] = 0.51
    assert train.internal_test_gate(metrics)["status"] == "internal_test_failed"
    metrics["per_symbol"]["BTCUSDT"]["auc"] = 0.60
    metrics["pooled"]["nonfinite_outputs"] = 1
    assert train.internal_test_gate(metrics)["status"] == "internal_test_failed"


def test_sequence_dataset_cannot_cross_split_or_symbol_boundaries():
    features = np.arange(12 * 27, dtype=np.float32).reshape(12, 27)
    labels = {
        "ret_cls": np.arange(12) % 2,
        "ret_reg": np.arange(12, dtype=float),
        "rv_reg": np.arange(12, dtype=float),
    }
    split = np.asarray([0] * 6 + [1] * 6, dtype=np.int8)
    btc_train = train.FrozenSequenceDataset(features, labels, split, 0, 4, "BTCUSDT")
    eth_train = train.FrozenSequenceDataset(features + 1000, labels, split, 0, 4, "ETHUSDT")
    validation = train.FrozenSequenceDataset(features, labels, split, 1, 4, "BTCUSDT")
    assert btc_train.endpoints.tolist() == [3, 4, 5]
    assert validation.endpoints.tolist() == [9, 10, 11]
    assert btc_train[0]["x"][0, 0].item() < 1000
    assert eth_train[0]["x"][0, 0].item() >= 1000


def test_deterministic_loader_generator_is_seeded_explicitly():
    torch = train._torch()
    first = torch.randperm(100, generator=train.deterministic_loader_generator(24001))
    second = torch.randperm(100, generator=train.deterministic_loader_generator(24001))
    third = torch.randperm(100, generator=train.deterministic_loader_generator(24002))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_candidate_output_path_cannot_target_incumbents():
    with pytest.raises(Exception):
        train.assert_safe_candidate_directory(train.BASE_DIR / "model_artifacts/dl_lstm_latest.pt")
    source = inspect.getsource(train._write_selected_candidate)
    assert "finalized candidate overwrite prohibited" in source
    assert "scaler.joblib" in source and "model.pt" in source


def test_trainer_has_no_confirmation_data_read_path():
    source = Path(train.__file__).read_text(encoding="utf-8")
    assert "model_candidate_confirmation/" not in source
    assert "capture_confirmation" not in source


def test_target_scales_use_valid_training_sequence_endpoints_only():
    features = np.arange(20 * 27, dtype=np.float32).reshape(20, 27)
    labels = {
        "ret_cls": np.arange(20) % 2,
        "ret_reg": np.linspace(-0.1, 0.1, 20),
        "rv_reg": np.linspace(0.01, 0.2, 20),
    }
    split = np.asarray([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int8)
    datasets = {}
    for symbol, offset in (("BTCUSDT", 0.0), ("ETHUSDT", 0.001)):
        shifted = {name: value + offset if name != "ret_cls" else value for name, value in labels.items()}
        datasets[symbol] = {
            "train": train.FrozenSequenceDataset(features, shifted, split, 0, 4, symbol),
            "validation": train.FrozenSequenceDataset(features, shifted, split, 1, 4, symbol),
            "internal_test": train.FrozenSequenceDataset(features, shifted, split, 2, 4, symbol),
        }
    first = train.training_sequence_target_scales(datasets)
    # Mutating non-training targets cannot alter endpoints or scales.
    labels["ret_reg"][10:] = 1e9
    labels["rv_reg"][10:] = 1e9
    second = train.training_sequence_target_scales(datasets)
    assert first == second
    assert first["source"] == "training_sequences_only"
    assert first["training_sequence_count_by_symbol"] == {"BTCUSDT": 7, "ETHUSDT": 7}


def test_resolved_manifest_contract_records_scales_weights_and_blockers():
    source = inspect.getsource(train._write_selected_candidate)
    for field in (
        "objective_contract_digest", "objective_policy_digest", "ret_target_scale", "rv_target_scale",
        "classification_weight", "return_weight", "rv_weight", "objective_contract_blocker",
        "candidate_auxiliary_health_blocker", "downstream_contract_blocker",
        "balance_contract_digest", "selected_loss_formulation", "balance_statistics",
    ):
        assert field in source
    assert "classification-only mode cannot finalize" in source


def test_default_real_training_objective_is_resolved_and_balance_frozen():
    assert train.OBJECTIVE_NAME == "resolved_candidate_objective"
    assert train.DEFAULT_TRAINING_CONFIG["classification_weight"] == 1.0
    assert train.DEFAULT_TRAINING_CONFIG["return_weight"] == 0.5
    assert train.DEFAULT_TRAINING_CONFIG["rv_weight"] == 0.5
    signature = inspect.signature(train.train_candidate_experiment)
    assert signature.parameters["objective"].default == train.OBJECTIVE_NAME
    assert signature.parameters["balance_freeze_path"].default == train.DEFAULT_BALANCE_FREEZE


def test_real_training_refuses_missing_balance_freeze_before_environment_access(tmp_path):
    with pytest.raises(train.ModelCandidateTrainingError, match="balance_freeze"):
        train.train_candidate_experiment(
            "lstm", tmp_path / "dataset", balance_freeze_path=tmp_path / "missing.json"
        )


def test_history_source_logs_weighted_and_unweighted_components():
    source = inspect.getsource(train.train_classification_candidate)
    for field in (
        "classification_loss", "return_regression_loss", "rv_regression_loss",
        "weighted_classification_loss", "weighted_return_loss", "weighted_rv_loss",
    ):
        assert field in source
    assert "adapt" not in source.lower()
