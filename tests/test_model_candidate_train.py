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


def _identity(
    seed=24001, dataset="a" * 64, learning_rate=0.001, objective="3" * 64,
    balance="4" * 64, rv_output=None,
):
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
        rv_output_contract_digest=(
            rv_output or train.candidate_rv_output_contract()["rv_output_contract_digest"]
        ),
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
def test_architecture_contract_matches_incumbent_shapes_and_new_candidate_output_contract(kind):
    contract = train.architecture_contract(kind)
    assert contract["incumbent_compatible_base_geometry"] == train.ARCHITECTURE_DEFAULTS[kind]
    assert contract["constructor"] == {
        **train.ARCHITECTURE_DEFAULTS[kind], "rv_output_transform": "softplus"
    }
    assert contract["architecture_mathematics_modified"] is True
    assert contract["rv_output_transform"] == "softplus"
    assert contract["rv_output_support"] == "strictly_positive"
    assert contract["post_hoc_rv_clipping_applied"] is False
    assert contract["rv_output_contract_digest"]
    assert contract["incumbent_state_shape_digest"]
    assert train.make_candidate_model(kind).rv_output_transform == "softplus"


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
    assert first[0] != _identity(rv_output="6" * 64)[0]
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


def _precision_scale_datasets():
    features = np.arange(16 * 27, dtype=np.float32).reshape(16, 27)
    split = np.asarray([0] * 6 + [-1] + [1] * 4 + [-1] + [2] * 4, dtype=np.int8)
    datasets = {}
    for symbol, shift in (("BTCUSDT", 0.0), ("ETHUSDT", 0.0314159265358979)):
        ret_reg = np.asarray([
            -9.1234567890123, -8.2345678901234,
            123456.789012345 + shift, 123456.799012346 + shift,
            123456.819012347 + shift, 123456.849012348 + shift,
            -7.3456789012345,
            10.1111111111111, 10.2222222222222, 10.3333333333333, 10.4444444444444,
            -6.4567890123456,
            20.1111111111111, 20.2222222222222, 20.3333333333333, 20.4444444444444,
        ], dtype=np.float64)
        rv_reg = np.asarray([
            9.1234567890123, 8.2345678901234,
            1.234567890123 + shift, 1.234577890124 + shift,
            1.234607890125 + shift, 1.234697890126 + shift,
            7.3456789012345,
            2.1111111111111, 2.2222222222222, 2.3333333333333, 2.4444444444444,
            6.4567890123456,
            3.1111111111111, 3.2222222222222, 3.3333333333333, 3.4444444444444,
        ], dtype=np.float64)
        labels = {
            "ret_cls": np.arange(16, dtype=np.int64) % 2,
            "ret_reg": ret_reg,
            "rv_reg": rv_reg,
        }
        datasets[symbol] = {
            name: train.FrozenSequenceDataset(features, labels, split, code, 3, symbol)
            for name, code in (("train", 0), ("validation", 1), ("internal_test", 2))
        }
    return datasets


def _build_dataset_style_target_scales(datasets):
    ret_targets = []
    rv_targets = []
    counts = {}
    for symbol in sorted(datasets):
        dataset = datasets[symbol]["train"]
        ret_targets.extend(np.asarray(dataset.labels["ret_reg"])[dataset.endpoints].tolist())
        rv_targets.extend(np.asarray(dataset.labels["rv_reg"])[dataset.endpoints].tolist())
        counts[symbol] = len(dataset.endpoints)
    scales = train.compute_training_target_scales(ret_targets, rv_targets)
    scales.update({
        "training_sequence_count_by_symbol": counts,
        "validation_targets_consulted": False,
        "internal_test_targets_consulted": False,
        "legacy_repair_targets_consulted": False,
        "confirmation_targets_consulted": False,
    })
    scales["target_scale_digest"] = train.json_digest({
        key: value for key, value in scales.items() if key != "target_scale_digest"
    })
    return scales


def test_float64_evidence_differs_from_float32_training_tensor_statistics():
    datasets = _precision_scale_datasets()
    ret64 = []
    rv64 = []
    for symbol in sorted(datasets):
        dataset = datasets[symbol]["train"]
        ret64.extend(np.asarray(dataset.labels["ret_reg"], dtype=np.float64)[dataset.endpoints])
        rv64.extend(np.asarray(dataset.labels["rv_reg"], dtype=np.float64)[dataset.endpoints])
    float64_scales = train.compute_training_target_scales(ret64, rv64)
    float32_scales = train.compute_training_target_scales(
        np.asarray(ret64, dtype=np.float32), np.asarray(rv64, dtype=np.float32)
    )
    assert abs(float64_scales["ret_target_scale"] - float32_scales["ret_target_scale"]) > 1e-6
    assert abs(float64_scales["rv_target_scale"] - float32_scales["rv_target_scale"]) > 1e-10
    assert float64_scales["target_scale_digest"] != float32_scales["target_scale_digest"]


def test_training_tensors_remain_float32_while_scale_evidence_bypasses_getitem(monkeypatch):
    datasets = _precision_scale_datasets()
    row = datasets["BTCUSDT"]["train"][0]
    torch = train._torch()
    assert row["y_ret_reg"].dtype == torch.float32
    assert row["y_rv_reg"].dtype == torch.float32

    def rounded_training_rows_must_not_supply_evidence(self, index):
        raise AssertionError("target-scale evidence must not pass through __getitem__")

    monkeypatch.setattr(
        train.FrozenSequenceDataset, "__getitem__", rounded_training_rows_must_not_supply_evidence
    )
    assert train.training_sequence_target_scales(datasets) == _build_dataset_style_target_scales(datasets)


def test_float64_target_scales_and_digest_match_build_dataset_calculation_exactly():
    datasets = _precision_scale_datasets()
    expected = _build_dataset_style_target_scales(datasets)
    observed = train.training_sequence_target_scales(datasets)
    assert observed["ret_target_scale"] == expected["ret_target_scale"]
    assert observed["rv_target_scale"] == expected["rv_target_scale"]
    assert observed["target_scale_digest"] == expected["target_scale_digest"]
    assert observed == expected
    assert observed["training_sequence_count_by_symbol"] == {"BTCUSDT": 4, "ETHUSDT": 4}


def test_only_selected_endpoint_labels_can_change_target_scale_evidence():
    datasets = _precision_scale_datasets()
    baseline = train.training_sequence_target_scales(datasets)
    train_dataset = datasets["BTCUSDT"]["train"]

    train_dataset.labels["ret_reg"][0] += 1e12
    train_dataset.labels["rv_reg"][0] += 1e12
    assert train.training_sequence_target_scales(datasets) == baseline

    endpoint = int(train_dataset.endpoints[0])
    train_dataset.labels["ret_reg"][endpoint] += 0.25
    changed = train.training_sequence_target_scales(datasets)
    assert changed["target_scale_digest"] != baseline["target_scale_digest"]
    assert changed["training_sequence_count_by_symbol"] == baseline["training_sequence_count_by_symbol"]


@pytest.mark.parametrize("excluded_index", [6, 7, 12], ids=["purged", "validation", "internal_test"])
def test_nontraining_and_purged_labels_cannot_affect_training_target_scales(excluded_index):
    datasets = _precision_scale_datasets()
    baseline = train.training_sequence_target_scales(datasets)
    for symbol in datasets:
        train_dataset = datasets[symbol]["train"]
        train_dataset.labels["ret_reg"][excluded_index] += 1e12
        train_dataset.labels["rv_reg"][excluded_index] += 1e12
    assert train.training_sequence_target_scales(datasets) == baseline


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


def test_real_training_rejects_historical_balance_before_dataset_or_evidence_access(
    tmp_path, monkeypatch
):
    def forbidden_access(*args, **kwargs):
        raise AssertionError("dataset and experiment evidence must remain unopened")

    monkeypatch.setattr(train, "validate_phase24_evidence", forbidden_access)
    monkeypatch.setattr(train, "load_sequence_datasets", forbidden_access)
    with pytest.raises(train.ModelCandidateTrainingError, match="RV-output contract mismatch"):
        train.train_candidate_experiment("lstm", tmp_path / "unopened-dataset")


def test_history_source_logs_weighted_and_unweighted_components():
    source = inspect.getsource(train.train_classification_candidate)
    for field in (
        "classification_loss", "return_regression_loss", "rv_regression_loss",
        "weighted_classification_loss", "weighted_return_loss", "weighted_rv_loss",
    ):
        assert field in source
    assert "adapt" not in source.lower()
