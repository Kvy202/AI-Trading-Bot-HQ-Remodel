from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from tools import model_candidate_loss_balance as balance
from tools import model_candidate_train as candidate_train
from tools import model_loss_balance_probe as probe


def _measurement(cls=1.0, ret=0.5, rv=0.5, *, ret_vector=None, rv_vector=None, finite=True, nonzero=True):
    import torch
    vectors = {
        "classification": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "return": torch.tensor(ret_vector or [ret, 0.0], dtype=torch.float64),
        "rv": torch.tensor(rv_vector or [rv, 0.0], dtype=torch.float64),
    }
    result = {}
    for task, norm in (("classification", cls), ("return", ret), ("rv", rv)):
        result[task] = {
            "shared_gradient_l2": float(norm), "head_gradient_l2": float(norm),
            "finite": finite, "nonzero": nonzero,
            "shared_vector": vectors[task], "full_vector": vectors[task],
        }
    result["cosines"] = {
        "classification_vs_return": balance.cosine_similarity(vectors["classification"], vectors["return"]),
        "classification_vs_rv": balance.cosine_similarity(vectors["classification"], vectors["rv"]),
        "return_vs_rv": balance.cosine_similarity(vectors["return"], vectors["rv"]),
    }
    return result


def test_policy_is_exact_typed_and_parent_digest_is_fixed():
    policy = balance.load_balance_policy()
    assert balance.validate_balance_policy(policy) == policy
    assert policy["parent_objective_contract_digest"] == balance.PARENT_OBJECTIVE_DIGEST
    assert policy["balance_calibration_seed"] == 24201
    wrong = copy.deepcopy(policy)
    wrong["allow_validation_for_balance"] = True
    with pytest.raises(balance.LossBalanceError):
        balance.validate_balance_policy(wrong)


def test_parent_objective_report_verifies_required_digest_and_verdict():
    report = balance.validate_parent_objective()
    assert report["objective_contract_digest"] == balance.PARENT_OBJECTIVE_DIGEST
    assert report["overall_decision"]["verdict"].endswith("multitask_training_required")


def test_formulation_descriptors_are_deterministic_fixed_and_raw_unit():
    first = balance.formulation_descriptor("normalized_mse_fixed")
    assert first == balance.formulation_descriptor("normalized_mse_fixed")
    assert first["classification_weight"] == 1.0
    assert first["return_weight"] == first["rv_weight"] == 0.5
    assert first["huber_beta"] is None
    huber = balance.formulation_descriptor("normalized_huber_fixed")
    assert huber["huber_beta"] == 1.0
    assert huber["outputs_remain_in_raw_runtime_units"] is True
    with pytest.raises(balance.LossBalanceError):
        balance.formulation_descriptor("normalized_huber_fixed", return_weight=0.1)


@pytest.mark.parametrize("kind", balance.ARCHITECTURES)
def test_parameter_groups_exclude_all_heads_and_are_complete(kind):
    model, _ = probe.production_model(kind)
    groups = balance.parameter_groups(model, kind)
    assert all(groups.values())
    shared_names = {name for name, _ in groups["shared_backbone"]}
    head_names = {name for key, values in groups.items() if key != "shared_backbone" for name, _ in values}
    assert not shared_names & head_names
    assert len(shared_names | head_names) == len(list(model.named_parameters()))


def test_component_measurement_uses_identical_parameters_and_never_updates():
    import torch
    model, _ = probe.production_model("lstm")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    batch = probe.synthetic_batch()
    result = balance.measure_task_gradients(
        model, "lstm", batch, formulation_id="normalized_huber_fixed",
        class_weights=torch.ones(2), ret_scale=0.025, rv_scale=0.008,
    )
    assert all(result[task]["finite"] and result[task]["nonzero"] for task in ("classification", "return", "rv"))
    assert all(value.detach().equal(before[name]) for name, value in model.named_parameters())
    assert len({result["starting_parameter_digest"]}) == 1


def test_statistics_ratios_clipping_and_cosines_are_exact():
    values = [1.0, 2.0, 3.0, 4.0]
    stats = balance.deterministic_statistics(values)
    assert stats["median"] == 2.5
    assert stats["p90"] == pytest.approx(3.7)
    report = balance.aggregate_measurements(
        [_measurement()], descriptor=balance.formulation_descriptor("normalized_mse_fixed")
    )
    ratios = report["weighted_gradient_ratios"]
    assert ratios["return_median_to_classification"] == 0.25
    assert ratios["rv_median_to_classification"] == 0.25
    assert ratios["combined_auxiliary_median_to_classification"] == 0.5
    assert report["gradient_clipping"]["expected_clip_activation_rate"] == 1.0
    assert report["pairwise_gradient_cosines"]["classification_vs_return"]["median"] == 1.0


def test_nonfinite_zero_and_projection_gates_fail_closed():
    import torch
    descriptor = balance.formulation_descriptor("normalized_mse_fixed")
    assert not balance.aggregate_measurements([_measurement(finite=False)], descriptor=descriptor)["passed"]
    assert not balance.aggregate_measurements([_measurement(nonzero=False)], descriptor=descriptor)["passed"]
    row = _measurement(ret_vector=[-4.0, 0], rv_vector=[-4.0, 0])
    failed = balance.aggregate_measurements([row], descriptor=descriptor)
    assert failed["classification_projection"]["all_positive"] is False
    cls = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert balance.classification_projection(cls, torch.tensor([0.0, 1.0], dtype=torch.float64))["positive"] is False
    assert balance.classification_projection(cls, torch.tensor([-1.0, 0.0], dtype=torch.float64))["positive"] is False


def test_balanced_weight_formula_bounds_and_no_iterative_search():
    weights = balance.derive_balanced_weights(2.0, 4.0, 8.0)
    assert weights["return"]["raw_weight"] == 0.125
    assert weights["rv"]["raw_weight"] == 0.0625
    assert weights["iterative_search_performed"] is False
    bounded = balance.derive_balanced_weights(1.0, 1e12, 1e-12)
    assert bounded["return"]["lower_bound_applied"] is True
    assert bounded["rv"]["upper_bound_applied"] is True


def test_minimal_change_selection_A_then_B_then_C_and_unresolved():
    safe = [_measurement()]
    a = balance.evaluate_formulations(safe, safe)
    assert a["selected_formulation"] == "normalized_mse_fixed"
    b = balance.evaluate_formulations([_measurement(ret=10, rv=10)], safe)
    assert b["selected_formulation"] == "normalized_huber_fixed"
    c = balance.evaluate_formulations(
        [_measurement(ret=10, rv=10)], [_measurement(ret=4, rv=8)]
    )
    assert c["selected_formulation"] == "normalized_huber_training_balanced"
    assert c["formulations"]["normalized_huber_training_balanced"]["weight_derivation"]["iterative_search_performed"] is False
    failed = balance.evaluate_formulations(
        [_measurement(nonzero=False)], [_measurement(nonzero=False)]
    )
    assert failed["balance_status"] == "loss_balance_unresolved"
    for forbidden in ("validation_metrics_consulted", "internal_test_metrics_consulted",
                      "legacy_repair_metrics_consulted", "confirmation_metrics_consulted"):
        assert failed[forbidden] is False


class _Dataset:
    def __init__(self, symbol, labels):
        import torch
        self.symbol = symbol
        self.endpoints = np.arange(len(labels)) + 63
        self.rows = [{
            "x": torch.zeros(64, 27), "y_ret_cls": torch.tensor(label),
            "y_ret_reg": torch.tensor(float(index)), "y_rv_reg": torch.tensor(float(index + 1)),
        } for index, label in enumerate(labels)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def test_calibration_sample_is_deterministic_training_only_and_has_symbols_classes():
    datasets = {
        "BTCUSDT": {"train": _Dataset("BTCUSDT", [0, 1] * 8)},
        "ETHUSDT": {"train": _Dataset("ETHUSDT", [1, 0] * 8)},
    }
    first = balance.select_calibration_indices(datasets, batches=2, batch_size=8, seed=24201)
    second = balance.select_calibration_indices(datasets, batches=2, batch_size=8, seed=24201)
    assert first["endpoint_digest"] == second["endpoint_digest"]
    assert first["source"] == "training_sequences_only"
    assert first["symbols"] == ["BTCUSDT", "ETHUSDT"] and first["classes"] == [0, 1]


def test_calibration_sample_rejects_missing_symbol_or_class():
    datasets = {"BTCUSDT": {"train": _Dataset("BTCUSDT", [0] * 20)}}
    with pytest.raises(balance.LossBalanceError, match="sample_invalid"):
        balance.select_calibration_indices(datasets, batches=1, batch_size=2, seed=24201)


def _safe_report():
    descriptor = balance.formulation_descriptor("normalized_mse_fixed")
    architectures = {
        kind: {
            "balance_status": "resolved", "selected_formulation": "normalized_mse_fixed",
            "selected_descriptor": descriptor,
            "formulations": {"normalized_mse_fixed": {"passed": True}},
        }
        for kind in balance.ARCHITECTURES
    }
    return {
        "dataset": {"verified": True, "dataset_digest": "d" * 64, "split_digest": "s" * 64,
                    "scaler_digest": "c" * 64,
                    "target_scales": {"target_scale_digest": "t" * 64}},
        "balance_policy": {"digest": balance.balance_policy_digest()},
        "calibration_sample": {"endpoint_digest": "e" * 64, "source": "training_sequences_only"},
        "rv_output_contract": candidate_train.candidate_rv_output_contract(),
        "architectures": architectures,
    }


def test_freeze_is_deterministic_validated_and_cannot_overwrite(tmp_path):
    path = tmp_path / "freeze.json"
    frozen = balance.freeze_balance_contract(_safe_report(), path)
    assert balance.validate_balance_freeze(path)["balance_contract_digest"] == frozen["balance_contract_digest"]
    with pytest.raises(balance.LossBalanceError, match="overwrite"):
        balance.freeze_balance_contract(_safe_report(), path)
    assert frozen["heterogeneous_architecture_objectives"] is False


def test_freeze_requires_verified_dataset_and_training_sequence_source(tmp_path):
    report = _safe_report()
    report["dataset"]["verified"] = False
    with pytest.raises(balance.LossBalanceError, match="verified frozen dataset"):
        balance.freeze_balance_contract(report, tmp_path / "bad.json")
    report = _safe_report()
    report["calibration_sample"]["source"] = "validation"
    with pytest.raises(balance.LossBalanceError, match="training sequences only"):
        balance.freeze_balance_contract(report, tmp_path / "bad.json")


@pytest.mark.parametrize("name", balance.ACCESS_LEDGER_NAMES)
def test_freeze_ordering_rejects_every_nontraining_access_ledger(tmp_path, name):
    (tmp_path / name).write_text(json.dumps({"accesses": [{"dataset_digest": "d" * 64}]}), encoding="utf-8")
    with pytest.raises(balance.LossBalanceError, match="contaminated"):
        balance.assert_balance_freeze_ordering(reports_root=tmp_path, dataset_digest="d" * 64)


def test_freeze_ordering_detects_existing_seed_validation_or_internal_test_access(tmp_path):
    path = tmp_path / "model_candidate_seed_runs" / "candidate" / "selected_freeze.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"candidate_identity": {"dataset_digest": "d" * 64},
                                "internal_test_accessed": True}), encoding="utf-8")
    with pytest.raises(balance.LossBalanceError, match="contaminated"):
        balance.assert_balance_freeze_ordering(reports_root=tmp_path, dataset_digest="d" * 64)


def test_validate_freeze_rejects_dataset_and_expected_digest_mismatch(tmp_path):
    path = tmp_path / "freeze.json"
    balance.freeze_balance_contract(_safe_report(), path)
    manifest = {"dataset_digest": "x" * 64, "split_digest": "s" * 64,
                "scaler": {"sha256": "c" * 64}, "target_scales": {"target_scale_digest": "t" * 64}}
    with pytest.raises(balance.LossBalanceError, match="dataset_digest"):
        balance.validate_balance_freeze(path, dataset_manifest=manifest)
    with pytest.raises(balance.LossBalanceError, match="unexpected"):
        balance.validate_balance_freeze(path, expected_balance_digest="0" * 64)


def test_balance_gate_accepts_exact_target_scale_digest_and_rejects_real_mismatch(tmp_path):
    report = _safe_report()
    path = tmp_path / "freeze.json"
    frozen = balance.freeze_balance_contract(report, path)
    manifest = {
        "dataset_digest": report["dataset"]["dataset_digest"],
        "split_digest": report["dataset"]["split_digest"],
        "scaler": {"sha256": report["dataset"]["scaler_digest"]},
        "target_scales": {
            "target_scale_digest": report["dataset"]["target_scales"]["target_scale_digest"]
        },
    }
    assert balance.validate_balance_freeze(path, dataset_manifest=manifest) == frozen

    manifest["target_scales"]["target_scale_digest"] = "u" * 64
    with pytest.raises(balance.LossBalanceError, match="target-scale digest mismatch"):
        balance.validate_balance_freeze(path, dataset_manifest=manifest)


def test_new_freeze_records_and_authorizes_only_matching_rv_output_contract(tmp_path):
    path = tmp_path / "new-freeze.json"
    frozen = balance.freeze_balance_contract(_safe_report(), path)
    expected = candidate_train.candidate_rv_output_contract()["rv_output_contract_digest"]

    assert frozen["rv_output_contract_digest"] == expected
    assert frozen["rv_output_contract"]["rv_output_transform"] == "softplus"
    assert balance.validate_balance_freeze(
        path, expected_rv_output_contract_digest=expected
    ) == frozen
    with pytest.raises(balance.LossBalanceError, match="RV-output contract mismatch"):
        balance.validate_balance_freeze(
            path, expected_rv_output_contract_digest="0" * 64
        )


def test_historical_freeze_is_readable_but_cannot_authorize_softplus_training():
    historical = balance.validate_balance_freeze(balance.DEFAULT_BALANCE_FREEZE)
    expected = candidate_train.candidate_rv_output_contract()["rv_output_contract_digest"]

    assert historical.get("rv_output_contract_digest") is None
    with pytest.raises(balance.LossBalanceError, match="RV-output contract mismatch"):
        balance.validate_balance_freeze(
            balance.DEFAULT_BALANCE_FREEZE,
            expected_rv_output_contract_digest=expected,
        )


def test_freeze_rejects_missing_or_mismatched_new_rv_output_contract(tmp_path):
    missing = _safe_report()
    missing.pop("rv_output_contract")
    with pytest.raises(balance.LossBalanceError, match="RV-output contract required"):
        balance.freeze_balance_contract(missing, tmp_path / "missing.json")

    mismatched = _safe_report()
    mismatched["rv_output_contract"]["rv_output_transform"] = "identity"
    payload = {
        key: value for key, value in mismatched["rv_output_contract"].items()
        if key != "rv_output_contract_digest"
    }
    mismatched["rv_output_contract"]["rv_output_contract_digest"] = balance._digest(payload)
    with pytest.raises(balance.LossBalanceError, match="RV-output contract mismatch"):
        balance.freeze_balance_contract(mismatched, tmp_path / "mismatched.json")


def test_real_calibration_target_scale_gate_is_exact_without_tolerance_bypass():
    source = inspect.getsource(balance.run_real_calibration)
    assert 'frozen.get("target_scale_digest") != scales["target_scale_digest"]' in source
    assert "isclose" not in source
    assert "allclose" not in source
    assert "tolerance" not in source.lower()


def test_real_calibration_source_has_no_nontraining_fallback_or_auc_selection():
    source = Path(balance.__file__).read_text(encoding="utf-8")
    assert 'datasets_by_symbol[symbol]["train"]' in source
    assert "validation_auc" not in source and "internal_test_auc" not in source
    assert "GradNorm" not in source and "PCGrad" not in source
