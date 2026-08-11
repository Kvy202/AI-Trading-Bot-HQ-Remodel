from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import pytest

from tools import model_candidate_train as train


def _legacy_class_weights(dataset):
    torch = train._torch()
    labels = np.asarray(
        [int(dataset[index]["y_ret_cls"]) for index in range(len(dataset))],
        dtype=np.int64,
    )
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise train.ModelCandidateTrainingError("both training classes are required")
    total = int(counts.sum())
    weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32
    )
    return weights, counts


def _frozen_concat(*, rows=20, sequence_length=4, labels=None):
    torch = train._torch()
    features = np.linspace(-1.0, 1.0, rows * 27, dtype=np.float32).reshape(rows, 27)
    ret_cls = np.asarray(labels if labels is not None else np.arange(rows) % 3 == 0, dtype=np.int64)
    targets = {
        "ret_cls": ret_cls,
        "ret_reg": np.linspace(-0.02, 0.03, rows, dtype=np.float64),
        "rv_reg": np.linspace(0.01, 0.04, rows, dtype=np.float64),
    }
    split = np.zeros(rows, dtype=np.int8)
    parts = [
        train.FrozenSequenceDataset(features + offset, targets, split, 0, sequence_length, symbol)
        for symbol, offset in (("BTCUSDT", 0.0), ("ETHUSDT", 0.25))
    ]
    return torch.utils.data.ConcatDataset(parts)


def test_frozen_concat_fast_path_is_bit_identical_and_bypasses_getitem(monkeypatch):
    torch = train._torch()
    dataset = _frozen_concat(rows=23, sequence_length=5)
    expected_weights, expected_counts = _legacy_class_weights(dataset)
    expected_sequence_count = len(dataset)

    def full_sample_construction_is_not_needed(self, index):
        raise AssertionError("optimized class weights must read endpoint labels directly")

    monkeypatch.setattr(train.FrozenSequenceDataset, "__getitem__", full_sample_construction_is_not_needed)
    actual_weights, actual_counts = train._class_weights(dataset)

    assert len(dataset) == expected_sequence_count
    assert int(actual_counts.sum()) == expected_sequence_count
    assert np.array_equal(actual_counts, expected_counts)
    assert torch.equal(actual_weights, expected_weights)
    assert actual_weights.numpy().tobytes() == expected_weights.numpy().tobytes()


def test_generic_dataset_uses_safe_legacy_compatible_fallback():
    torch = train._torch()

    class GenericDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.labels = [0, 1, 1, 0, 1, 1, 1]
            self.accesses = []

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            self.accesses.append(index)
            return {"y_ret_cls": torch.tensor(self.labels[index], dtype=torch.long)}

    dataset = GenericDataset()
    expected_weights, expected_counts = _legacy_class_weights(dataset)
    dataset.accesses.clear()
    actual_weights, actual_counts = train._class_weights(dataset)

    assert dataset.accesses == list(range(len(dataset)))
    assert np.array_equal(actual_counts, expected_counts)
    assert torch.equal(actual_weights, expected_weights)


def test_frozen_fast_path_preserves_missing_class_failure():
    dataset = _frozen_concat(rows=12, sequence_length=3, labels=np.zeros(12, dtype=np.int64))

    with pytest.raises(train.ModelCandidateTrainingError) as legacy_error:
        _legacy_class_weights(dataset)
    with pytest.raises(train.ModelCandidateTrainingError) as optimized_error:
        train._class_weights(dataset)

    assert str(optimized_error.value) == str(legacy_error.value) == "both training classes are required"


def _small_split_datasets():
    rows = 24
    features = np.linspace(-0.75, 0.85, rows * 27, dtype=np.float32).reshape(rows, 27)
    split = np.asarray([0] * 12 + [1] * 6 + [2] * 6, dtype=np.int8)
    ret_cls = np.asarray([0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 2, dtype=np.int64)
    datasets = {}
    for symbol, shift in (("BTCUSDT", 0.0), ("ETHUSDT", 0.05)):
        labels = {
            "ret_cls": ret_cls,
            "ret_reg": np.linspace(-0.02 + shift, 0.03 + shift, rows, dtype=np.float64),
            "rv_reg": np.linspace(0.01 + shift, 0.04 + shift, rows, dtype=np.float64),
        }
        datasets[symbol] = {
            name: train.FrozenSequenceDataset(features + shift, labels, split, code, 3, symbol)
            for name, code in (("train", 0), ("validation", 1), ("internal_test", 2))
        }
    return datasets


def _tiny_config(*, epochs=1):
    return {
        "batch_size": 4,
        "epochs": epochs,
        "patience": epochs,
        "learning_rate": 0.002,
        "optimizer": {"name": "AdamW", "weight_decay": 0.0001},
        "scheduler": {"name": "CosineAnnealingLR", "eta_min_factor": 0.1},
        "gradient_clipping": 1.0,
    }


class _TinyMultiHeadModel:
    @staticmethod
    def make():
        torch = train._torch()

        class TinyMultiHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(27, 4)
                self.first_training_batch = None

            def forward(self, values):
                if self.training and self.first_training_batch is None:
                    self.first_training_batch = values.detach().clone()
                output = self.head(values.mean(dim=1))
                return {
                    "ret_cls_logits": output[:, :2],
                    "ret_reg": output[:, 2],
                    "rv_reg": output[:, 3],
                }

        return TinyMultiHead()


def test_precomputed_evidence_is_reused_across_seeds_without_other_split_labels(
    monkeypatch, capsys
):
    torch = train._torch()
    datasets = _small_split_datasets()
    training_dataset = train._concat(datasets, "train")
    weights, counts = train._class_weights(training_dataset)
    shared_weights = tuple(float(value) for value in weights.tolist())
    shared_counts = tuple(int(value) for value in counts.tolist())
    seen_weights = []

    def class_weights_must_not_be_recalculated(dataset):
        raise AssertionError("precomputed training-only class weights must be reused")

    def validation_without_label_access(model, datasets_by_symbol, split, **kwargs):
        assert split == "validation"
        seen_weights.append(tuple(kwargs["class_weights"]))
        return {
            "split": "validation",
            "pooled": {"auc": 0.6, "classification_loss": 0.5},
            "per_symbol": {},
            "deterministic_repeat_passed": kwargs.get("deterministic_repeat"),
        }

    monkeypatch.setattr(train, "_class_weights", class_weights_must_not_be_recalculated)
    monkeypatch.setattr(train, "evaluate_classification_model", validation_without_label_access)

    for seed in (701, 702):
        train.set_deterministic_seed(99)
        model = _TinyMultiHeadModel.make()
        history, _ = train.train_classification_candidate(
            model,
            datasets,
            seed=seed,
            config=_tiny_config(),
            objective=train.LEGACY_OBJECTIVE_NAME,
            class_weights=shared_weights,
            class_counts=shared_counts,
        )
        assert history["class_counts"] == {"0": shared_counts[0], "1": shared_counts[1]}

    first_local, _ = train._resolved_class_weight_evidence(
        training_dataset, class_weights=shared_weights, class_counts=shared_counts
    )
    first_local[0] = -123.0
    second_local, _ = train._resolved_class_weight_evidence(
        training_dataset, class_weights=shared_weights, class_counts=shared_counts
    )
    assert tuple(float(value) for value in second_local.tolist()) == shared_weights
    assert all(value == shared_weights for value in seen_weights)
    assert len(seen_weights) == 4  # epoch validation plus selected repeat for both seeds
    assert all(datasets[symbol]["internal_test"].endpoints.size for symbol in datasets)
    assert torch.equal(second_local, weights)
    capsys.readouterr()


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_optimized_weights_preserve_deterministic_training_behavior(monkeypatch, capsys):
    torch = train._torch()
    datasets = _small_split_datasets()
    training_dataset = train._concat(datasets, "train")
    reference_weights, reference_counts = _legacy_class_weights(training_dataset)
    scales = train.training_sequence_target_scales(datasets)

    train.set_deterministic_seed(8001)
    optimized_model = _TinyMultiHeadModel.make()
    reference_model = copy.deepcopy(optimized_model)

    real_loss = train.resolved_candidate_loss
    first_losses = []
    current_run_losses = []

    def capture_first_batch_loss(*args, **kwargs):
        components = real_loss(*args, **kwargs)
        if not current_run_losses:
            current_run_losses.append(float(components["total_loss"].detach().item()))
        return components

    monkeypatch.setattr(train, "resolved_candidate_loss", capture_first_batch_loss)
    optimized_history, optimized_validation = train.train_classification_candidate(
        optimized_model,
        datasets,
        seed=8101,
        config=_tiny_config(epochs=3),
        target_scales=scales,
    )
    first_losses.append(current_run_losses.pop())

    reference_history, reference_validation = train.train_classification_candidate(
        reference_model,
        datasets,
        seed=8101,
        config=_tiny_config(epochs=3),
        target_scales=scales,
        class_weights=reference_weights.tolist(),
        class_counts=reference_counts.tolist(),
    )
    first_losses.append(current_run_losses.pop())

    assert torch.equal(optimized_model.first_training_batch, reference_model.first_training_batch)
    assert first_losses[0] == first_losses[1]
    assert optimized_history["epochs"] == reference_history["epochs"]
    assert optimized_history["best_epoch_auc"] == reference_history["best_epoch_auc"]
    assert optimized_history["best_epoch_loss"] == reference_history["best_epoch_loss"]
    assert optimized_validation == reference_validation
    assert _state_digest(optimized_model) == _state_digest(reference_model)
    assert all(
        torch.equal(optimized_model.state_dict()[name], reference_model.state_dict()[name])
        for name in optimized_model.state_dict()
    )
    capsys.readouterr()


def test_class_weight_calculation_timing_diagnostic():
    rows = 10_000
    labels = (np.arange(rows, dtype=np.int64) % 4 == 0).astype(np.int64)
    dataset = _frozen_concat(rows=rows, sequence_length=32, labels=labels)

    legacy_started = time.perf_counter()
    legacy_weights, legacy_counts = _legacy_class_weights(dataset)
    legacy_elapsed = time.perf_counter() - legacy_started

    optimized_started = time.perf_counter()
    optimized_weights, optimized_counts = train._class_weights(dataset)
    optimized_elapsed = time.perf_counter() - optimized_started

    result = {
        "sequence_count": len(dataset),
        "legacy_class_counts": legacy_counts.tolist(),
        "optimized_class_counts": optimized_counts.tolist(),
        "legacy_weights": legacy_weights.tolist(),
        "optimized_weights": optimized_weights.tolist(),
        "legacy_elapsed_seconds": legacy_elapsed,
        "optimized_elapsed_seconds": optimized_elapsed,
        "speedup_ratio": legacy_elapsed / max(optimized_elapsed, 1e-12),
    }
    print("class_weight_benchmark=" + json.dumps(result, sort_keys=True))

    assert np.array_equal(optimized_counts, legacy_counts)
    assert train._torch().equal(optimized_weights, legacy_weights)
    assert legacy_elapsed >= 0.0
    assert optimized_elapsed >= 0.0
