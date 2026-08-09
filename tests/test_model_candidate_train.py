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


def _identity(seed=24001, dataset="a" * 64, learning_rate=0.001):
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
    )


def test_legacy_objective_is_classification_only_and_auxiliary_heads_are_not_misrepresented():
    contract = train.training_objective_contract()
    assert contract["loss"] == "CrossEntropyLoss"
    assert contract["optimized_output"] == "ret_cls_logits"
    assert contract["ret_reg_optimized"] is False
    assert contract["rv_reg_optimized"] is False
    assert contract["auxiliary_head_training_status"] == "auxiliary_unoptimized_under_legacy_objective"
    source = inspect.getsource(train.train_classification_candidate)
    assert 'model(batch["x"].cpu())["ret_cls_logits"]' in source
    assert "clip_grad_norm_" in source


def test_downstream_use_sets_auxiliary_head_promotion_blocker():
    audit = train.downstream_auxiliary_head_audit()
    assert audit["candidate_auxiliary_head_promotion_blocker"] is True
    assert audit["promotion_blocked_until_objective_contract_resolved"] is True
    assert audit["consumers"]["trade_multi_bitget.py"]["rv_hat_referenced"] is True


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
