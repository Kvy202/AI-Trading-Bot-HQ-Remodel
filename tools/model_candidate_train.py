"""Train immutable Phase 24 multitask-health candidates from frozen data.

This module deliberately has no market-data or confirmation-data imports.  The
CLI is permitted to run only under the dedicated canonical training Python.
It reuses the architecture classes from :mod:`ml_dl.dl_models`, but never calls
the production-style ``ml_dl.dl_train.main`` save path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_training_environment import (
    INCUMBENT_INVENTORY,
    TRAINING_ENV,
    TRAINING_PYTHON,
    CandidateTrainingEnvironmentError,
    assert_safe_candidate_directory,
    atomic_write_json,
    file_digest,
    git_commit,
    incumbent_hashes,
    interpreter_inventory,
    json_digest,
    load_training_policy,
    record_incumbent_inventory,
    training_contract,
    utc_now,
    validate_phase24_evidence,
    validate_training_inventory,
    verify_incumbent_inventory,
)
from tools.model_training_dataset import verify_dataset
from tools.model_candidate_objective import (
    LEGACY_AUXILIARY_STATUS,
    LEGACY_OBJECTIVE_NAME,
    OBJECTIVE_NAME,
    RESOLVED_AUXILIARY_STATUS,
    compute_training_target_scales,
    load_objective_policy,
    objective_metrics,
    resolved_candidate_loss,
)
from tools.model_candidate_loss_balance import (
    DEFAULT_BALANCE_FREEZE,
    LossBalanceError,
    formulation_descriptor,
    validate_balance_freeze,
)
from tools.model_objective_contract import (
    OBJECTIVE_REPORT,
    ObjectiveContractError,
    expected_objective_contract_digest,
    validate_objective_report,
)
from tools.model_auxiliary_head_audit import build_auxiliary_audit


CANDIDATE_ROOT = BASE_DIR / "model_artifacts" / "candidates"
SEED_RUN_ROOT = BASE_DIR / "reports" / "model_candidate_seed_runs"
TRAINING_SUMMARY = BASE_DIR / "reports" / "model_candidate_training_summary.json"
SELECTION_FREEZE = BASE_DIR / "reports" / "model_candidate_selection_freeze.json"
VALIDATION_ACCESS_LEDGER = BASE_DIR / "reports" / "model_candidate_validation_access.json"
ALLOWED_KINDS = ("lstm", "tcn", "tx")
AUXILIARY_STATUS = RESOLVED_AUXILIARY_STATUS

DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "batch_size": 256,
    "epochs": 40,
    "patience": 8,
    "learning_rate": 0.001,
    "optimizer": {"name": "AdamW", "weight_decay": 0.0001},
    "scheduler": {"name": "CosineAnnealingLR", "eta_min_factor": 0.1},
    "gradient_clipping": 1.0,
    "class_weighting": "inverse_frequency_from_training_sequences",
    "classification_weight": 1.0,
    "return_weight": 0.5,
    "rv_weight": 0.5,
    "selection_rule": [
        "highest_pooled_validation_auc",
        "lower_pooled_validation_loss",
        "lower_seed",
    ],
}

ARCHITECTURE_DEFAULTS: dict[str, dict[str, Any]] = {
    "lstm": {"in_dim": 27, "hidden": 64, "layers": 2, "dropout": 0.1},
    "tcn": {"in_dim": 27, "hid": 64, "levels": 4, "kernel": 3, "dropout": 0.1},
    "tx": {"in_dim": 27, "d_model": 64, "nhead": 4, "nlayers": 2, "dropout": 0.1},
}


class ModelCandidateTrainingError(ValueError):
    """A frozen-data, selection, test-access, or artifact gate failed."""


def training_objective_contract(
    objective: str = OBJECTIVE_NAME,
    *,
    target_scales: Mapping[str, Any] | None = None,
    objective_contract_digest: str | None = None,
    formulation: Mapping[str, Any] | None = None,
    balance_contract_digest: str | None = None,
) -> dict[str, Any]:
    """Return the resolved contract or the explicitly research-only legacy mode."""
    if objective == LEGACY_OBJECTIVE_NAME:
        return {
            "name": LEGACY_OBJECTIVE_NAME,
            "optimized_output": "ret_cls_logits",
            "target": "y_ret_cls",
            "loss": "CrossEntropyLoss",
            "ret_reg_optimized": False,
            "rv_reg_optimized": False,
            "auxiliary_head_training_status": LEGACY_AUXILIARY_STATUS,
            "research_only": True,
            "candidate_finalization_allowed": False,
            "confirmation_health_pass_allowed": False,
            "objective_contract_blocker": True,
            "source": "ml_dl/dl_train.py:train_once",
        }
    if objective != OBJECTIVE_NAME:
        raise ModelCandidateTrainingError("unknown candidate objective")
    policy = load_objective_policy()
    scales = dict(target_scales or {})
    effective = dict(formulation or formulation_descriptor("normalized_mse_fixed"))
    return {
        "name": OBJECTIVE_NAME,
        "objective_source": "new_candidate_only_contract",
        "objective_schema_version": int(policy["schema_version"]),
        "objective_policy_digest": json_digest(policy),
        "objective_contract_digest": objective_contract_digest,
        "formula": "fixed_weighted_resolved_candidate_loss",
        "classification": dict(policy["classification"]),
        "return_regression": {**dict(policy["return_regression"]),
                              "effective_loss": effective["regression_loss"]},
        "volatility_regression": {**dict(policy["volatility_regression"]),
                                  "effective_loss": effective["regression_loss"]},
        "target_scale_source": policy["target_scale_source"],
        "ret_target_scale": scales.get("ret_target_scale"),
        "rv_target_scale": scales.get("rv_target_scale"),
        "selected_loss_formulation": effective["formulation_id"],
        "classification_weight": effective["classification_weight"],
        "return_weight": effective["return_weight"],
        "rv_weight": effective["rv_weight"],
        "huber_beta": effective["huber_beta"],
        "balance_contract_digest": balance_contract_digest,
        "ret_reg_optimized": True,
        "rv_reg_optimized": True,
        "auxiliary_head_training_status": AUXILIARY_STATUS,
        "research_only": False,
        "candidate_finalization_allowed": True,
        "confirmation_health_pass_allowed": True,
        "objective_contract_blocker": False,
        "optimizer": copy.deepcopy(DEFAULT_TRAINING_CONFIG["optimizer"]),
        "scheduler": copy.deepcopy(DEFAULT_TRAINING_CONFIG["scheduler"]),
        "gradient_clipping": DEFAULT_TRAINING_CONFIG["gradient_clipping"],
        "best_epoch_rule": "highest_validation_auc_then_lower_validation_loss",
        "source": "Phase 24.1 new candidate-only contract",
    }


def downstream_auxiliary_head_audit(repository: Path | str = BASE_DIR) -> dict[str, Any]:
    """Compatibility wrapper around the complete Phase 24.1 downstream audit."""
    audit = build_auxiliary_audit(repository)
    return {
        "audit_type": "phase24_1_complete_downstream_use",
        "audit_digest": audit["audit_digest"],
        "consumers": audit["consumers"],
        "objective_contract_blocker": False,
        "candidate_auxiliary_health_blocker": "unverified",
        "downstream_contract_blocker": audit["downstream_contract_blocker"],
        "candidate_auxiliary_head_promotion_blocker": True,
        "classification_health_research_allowed": True,
        "promotion_blocked_until_auxiliary_health_verified": True,
    }


def validate_training_objective_gate(
    objective: str,
    *,
    report_path: Path | str = OBJECTIVE_REPORT,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    """Fail closed before any real training/environment/data access."""
    if objective == LEGACY_OBJECTIVE_NAME:
        raise ModelCandidateTrainingError("classification_only_legacy cannot finalize a Phase 24 candidate")
    if objective != OBJECTIVE_NAME:
        raise ModelCandidateTrainingError("resolved_candidate_objective is required")
    try:
        return validate_objective_report(
            report_path, expected_digest=expected_digest or expected_objective_contract_digest()
        )
    except ObjectiveContractError as exc:
        raise ModelCandidateTrainingError(str(exc)) from exc


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised in actual training env
        raise ModelCandidateTrainingError("PyTorch is required for candidate training") from exc
    return torch


def set_deterministic_seed(seed: int) -> dict[str, Any]:
    torch = _torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    warnings: list[str] = []
    deterministic = False
    try:
        torch.use_deterministic_algorithms(True)
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
    except Exception as exc:  # pragma: no cover - version dependent
        warnings.append(f"{type(exc).__name__}: {exc}")
    return {
        "seed": seed,
        "random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "cpu_only": True,
        "torch_thread_count": int(torch.get_num_threads()),
        "deterministic_algorithms_enabled": deterministic,
        "deterministic_warnings": warnings,
    }


def deterministic_loader_generator(seed: int):
    generator = _torch().Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Dependency-light AUC with average ranks for exact ties."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ModelCandidateTrainingError("valid finite labels/probabilities required for AUC")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ModelCandidateTrainingError("both classes are required for a valid AUC")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and p[order[end]] == p[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def architecture_contract(kind: str, repository: Path | str = BASE_DIR) -> dict[str, Any]:
    if kind not in ALLOWED_KINDS:
        raise ModelCandidateTrainingError("ADV is retained and may not be retrained in Phase 24")
    torch = _torch()
    from ml_dl.dl_models import TemporalConvNet, TinyLSTM, TinyTransformer

    cls = {"lstm": TinyLSTM, "tcn": TemporalConvNet, "tx": TinyTransformer}[kind]
    config = copy.deepcopy(ARCHITECTURE_DEFAULTS[kind])
    signature = str(inspect.signature(cls.__init__))
    model = cls(**config)
    root = Path(repository)
    incumbent_path = root / "model_artifacts" / f"dl_{kind}_latest.pt"
    metadata_path = root / "model_artifacts" / f"dl_{kind}_metadata.json"
    if not incumbent_path.is_file() or not metadata_path.is_file():
        raise ModelCandidateTrainingError("candidate_architecture_contract_incomplete")
    incumbent_state = torch.load(incumbent_path, map_location="cpu", weights_only=True)
    candidate_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    incumbent_shapes = {name: list(value.shape) for name, value in incumbent_state.items()}
    if candidate_shapes != incumbent_shapes:
        raise ModelCandidateTrainingError("candidate_architecture_contract_incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if int(metadata.get("n_features", -1)) != 27 or int(metadata.get("seq_len", -1)) != 64:
        raise ModelCandidateTrainingError("candidate_architecture_contract_incomplete")
    return {
        "kind": kind,
        "class": f"{cls.__module__}.{cls.__name__}",
        "constructor": config,
        "constructor_signature": signature,
        "derived_from": ["existing_factory", "model_class_defaults", "incumbent_state_shapes", "incumbent_metadata"],
        "incumbent_state_shape_digest": json_digest(incumbent_shapes),
        "model_code_digest": file_digest(root / "ml_dl" / "dl_models.py"),
        "architecture_mathematics_modified": False,
    }


def make_candidate_model(kind: str, config: Mapping[str, Any] | None = None):
    from ml_dl.dl_models import TemporalConvNet, TinyLSTM, TinyTransformer

    if kind not in ALLOWED_KINDS:
        raise ModelCandidateTrainingError("ADV is retained and may not be retrained in Phase 24")
    values = dict(config or ARCHITECTURE_DEFAULTS[kind])
    return {"lstm": TinyLSTM, "tcn": TemporalConvNet, "tx": TinyTransformer}[kind](**values)


class FrozenSequenceDataset:
    """Sequences from exactly one symbol and one split; boundaries cannot cross."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Mapping[str, np.ndarray],
        split_codes: np.ndarray,
        split_code: int,
        sequence_length: int,
        symbol: str,
    ) -> None:
        torch = _torch()
        matrix = np.asarray(features, dtype=np.float32)
        codes = np.asarray(split_codes, dtype=np.int8)
        if matrix.ndim != 2 or matrix.shape[1] != 27 or len(matrix) != len(codes):
            raise ModelCandidateTrainingError("frozen feature shape mismatch")
        positions = np.flatnonzero(codes == int(split_code))
        endpoints: list[int] = []
        for offset, position in enumerate(positions):
            if offset < sequence_length - 1:
                continue
            window_positions = positions[offset - sequence_length + 1:offset + 1]
            if not np.all(np.diff(window_positions) == 1):
                continue
            target_ok = all(np.isfinite(np.asarray(labels[name])[position]) for name in ("ret_cls", "ret_reg", "rv_reg"))
            if target_ok and np.isfinite(matrix[window_positions]).all():
                endpoints.append(int(position))
        self.X = matrix
        self.labels = {key: np.asarray(value) for key, value in labels.items()}
        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.L = int(sequence_length)
        self.symbol = str(symbol)
        self._torch = torch

    def __len__(self) -> int:
        return int(len(self.endpoints))

    def __getitem__(self, index: int) -> dict[str, Any]:
        endpoint = int(self.endpoints[int(index)])
        start = endpoint - self.L + 1
        return {
            "x": self._torch.from_numpy(self.X[start:endpoint + 1]),
            "y_ret_cls": self._torch.tensor(int(self.labels["ret_cls"][endpoint]), dtype=self._torch.long),
            "y_ret_reg": self._torch.tensor(float(self.labels["ret_reg"][endpoint]), dtype=self._torch.float32),
            "y_rv_reg": self._torch.tensor(float(self.labels["rv_reg"][endpoint]), dtype=self._torch.float32),
        }


def load_sequence_datasets(dataset: Path | str) -> tuple[dict[str, dict[str, FrozenSequenceDataset]], dict[str, Any]]:
    import joblib

    root = Path(dataset)
    manifest = verify_dataset(root)
    scaler = joblib.load(root / "scaler.joblib")
    result: dict[str, dict[str, FrozenSequenceDataset]] = {}
    for symbol in manifest["symbols"]:
        with np.load(root / f"features_{symbol}.npz", allow_pickle=False) as values:
            features = np.asarray(values["features"], dtype=np.float32)
            split = np.asarray(values["split"], dtype=np.int8)
        with np.load(root / f"labels_{symbol}.npz", allow_pickle=False) as values:
            labels = {name: np.asarray(values[name]) for name in ("ret_cls", "ret_reg", "rv_reg")}
        scaled = scaler.transform(features).astype(np.float32, copy=False)
        result[symbol] = {
            name: FrozenSequenceDataset(scaled, labels, split, code, int(manifest["sequence_length"]), symbol)
            for name, code in (("train", 0), ("validation", 1), ("internal_test", 2))
        }
        for name, value in result[symbol].items():
            expected = manifest["per_symbol"][symbol]["valid_sequences_by_split"][name]
            if len(value) != expected:
                raise ModelCandidateTrainingError(f"sequence count mismatch: {symbol}/{name}")
    return result, manifest


def _concat(symbol_datasets: Mapping[str, Mapping[str, Any]], split: str):
    torch = _torch()
    return torch.utils.data.ConcatDataset([symbol_datasets[symbol][split] for symbol in sorted(symbol_datasets)])


def _class_weights(dataset: Any):
    torch = _torch()
    labels = np.asarray([int(dataset[index]["y_ret_cls"]) for index in range(len(dataset))], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ModelCandidateTrainingError("both training classes are required")
    total = int(counts.sum())
    return torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32), counts


def training_sequence_target_scales(
    datasets_by_symbol: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive float64 evidence scales at frozen training-sequence endpoints."""
    ret_targets: list[float] = []
    rv_targets: list[float] = []
    per_symbol: dict[str, int] = {}
    for symbol, values in sorted(datasets_by_symbol.items()):
        dataset = values["train"]
        per_symbol[symbol] = len(dataset)
        endpoints = dataset.endpoints
        ret_targets.extend(
            np.asarray(dataset.labels["ret_reg"], dtype=np.float64)[endpoints].tolist()
        )
        rv_targets.extend(
            np.asarray(dataset.labels["rv_reg"], dtype=np.float64)[endpoints].tolist()
        )
    scales = compute_training_target_scales(ret_targets, rv_targets)
    scales["training_sequence_count_by_symbol"] = per_symbol
    scales["validation_targets_consulted"] = False
    scales["internal_test_targets_consulted"] = False
    scales["legacy_repair_targets_consulted"] = False
    scales["confirmation_targets_consulted"] = False
    scales["target_scale_digest"] = json_digest({
        key: value for key, value in scales.items() if key != "target_scale_digest"
    })
    return scales


def _forward_dataset(model: Any, dataset: Any, *, batch_size: int) -> dict[str, np.ndarray]:
    torch = _torch()
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    values: dict[str, list[np.ndarray]] = {key: [] for key in ("probability", "ret_hat", "rv_hat", "label", "ret", "rv")}
    model.eval().cpu()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].cpu())
            values["probability"].append(torch.softmax(out["ret_cls_logits"], dim=-1)[:, 1].cpu().numpy())
            values["ret_hat"].append(out["ret_reg"].reshape(-1).cpu().numpy())
            values["rv_hat"].append(out["rv_reg"].reshape(-1).cpu().numpy())
            values["label"].append(batch["y_ret_cls"].cpu().numpy())
            values["ret"].append(batch["y_ret_reg"].cpu().numpy())
            values["rv"].append(batch["y_rv_reg"].cpu().numpy())
    if not len(dataset):
        return {key: np.asarray([], dtype=np.float64) for key in values}
    return {key: np.concatenate(parts).astype(np.float64) for key, parts in values.items()}


def _metrics(
    outputs: Mapping[str, np.ndarray], class_weights: Sequence[float] | None = None,
    *,
    target_scales: Mapping[str, Any] | None = None,
    objective: str = OBJECTIVE_NAME,
    deterministic_repeat_passed: bool = True,
) -> dict[str, Any]:
    probability = np.asarray(outputs["probability"], dtype=np.float64)
    label = np.asarray(outputs["label"], dtype=np.int64)
    nonfinite = int(sum(np.size(value) - np.isfinite(value).sum() for value in outputs.values()))
    counts = {"0": int(np.sum(label == 0)), "1": int(np.sum(label == 1))}
    if nonfinite:
        return {
            "auc": None, "classification_loss": None, "rows": int(len(label)),
            "class_counts": counts, "nonfinite_outputs": nonfinite,
            "auxiliary_head_gate_passed": False,
            "auxiliary_head_training_status": (
                AUXILIARY_STATUS if objective == OBJECTIVE_NAME else LEGACY_AUXILIARY_STATUS
            ),
        }
    try:
        auc = binary_auc(label, probability)
    except ModelCandidateTrainingError:
        auc = None
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    row_loss = -(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))
    if class_weights is not None:
        weights = np.asarray(class_weights, dtype=np.float64)
        if weights.shape != (2,):
            raise ModelCandidateTrainingError("classification class weights must have width two")
        loss = float(np.sum(row_loss * weights[label]) / np.sum(weights[label]))
    else:
        loss = float(np.mean(row_loss))
    ret_error = np.asarray(outputs["ret_hat"]) - np.asarray(outputs["ret"])
    rv_error = np.asarray(outputs["rv_hat"]) - np.asarray(outputs["rv"])
    result: dict[str, Any] = {
        "auc": None if auc is None else float(auc),
        "classification_loss": loss,
        "rows": int(len(label)),
        "class_counts": counts,
        "nonfinite_outputs": 0,
        "ret_reg_diagnostic_mae": float(np.mean(np.abs(ret_error))),
        "ret_reg_diagnostic_rmse": float(np.sqrt(np.mean(ret_error ** 2))),
        "rv_reg_diagnostic_mae": float(np.mean(np.abs(rv_error))),
        "rv_reg_diagnostic_rmse": float(np.sqrt(np.mean(rv_error ** 2))),
        "auxiliary_head_training_status": (
            AUXILIARY_STATUS if objective == OBJECTIVE_NAME else LEGACY_AUXILIARY_STATUS
        ),
    }
    if objective == OBJECTIVE_NAME and target_scales is not None:
        auxiliary = objective_metrics(
            ret_prediction=outputs["ret_hat"], ret_target=outputs["ret"],
            rv_prediction=outputs["rv_hat"], rv_target=outputs["rv"],
            ret_scale=float(target_scales["ret_target_scale"]),
            rv_scale=float(target_scales["rv_target_scale"]),
            ret_train_target_mean=float(target_scales["ret_train_target_mean"]),
            rv_train_target_mean=float(target_scales["rv_train_target_mean"]),
            deterministic_repeat_passed=deterministic_repeat_passed,
        )
        result["auxiliary_metrics"] = auxiliary
        result["auxiliary_head_gate_passed"] = auxiliary["auxiliary_head_gate_passed"]
        result["ret_reg_metrics"] = auxiliary["ret_reg"]
        result["rv_reg_metrics"] = auxiliary["rv_reg"]
    else:
        result["auxiliary_metrics"] = {
            "ret_reg": {"classification": "auxiliary_unverified"},
            "rv_reg": {"classification": "auxiliary_unverified"},
            "auxiliary_head_gate_passed": False,
            "post_hoc_rv_clipping_applied": False,
        }
        result["auxiliary_head_gate_passed"] = False
    return result


def evaluate_classification_model(
    model: Any,
    datasets_by_symbol: Mapping[str, Mapping[str, Any]],
    split: str,
    *,
    batch_size: int = 256,
    deterministic_repeat: bool = False,
    class_weights: Sequence[float] | None = None,
    target_scales: Mapping[str, Any] | None = None,
    objective: str = OBJECTIVE_NAME,
) -> dict[str, Any]:
    outputs_by_symbol = {
        symbol: _forward_dataset(model, values[split], batch_size=batch_size)
        for symbol, values in sorted(datasets_by_symbol.items())
    }
    pooled_outputs = {
        key: np.concatenate([outputs_by_symbol[symbol][key] for symbol in sorted(outputs_by_symbol)])
        for key in next(iter(outputs_by_symbol.values()))
    }
    repeat_error = None
    repeat_passed = None
    if deterministic_repeat:
        repeated = {
            symbol: _forward_dataset(model, values[split], batch_size=batch_size)
            for symbol, values in sorted(datasets_by_symbol.items())
        }
        errors = []
        for symbol in sorted(outputs_by_symbol):
            for key in ("probability", "ret_hat", "rv_hat"):
                first = outputs_by_symbol[symbol][key]
                second = repeated[symbol][key]
                errors.append(float(np.max(np.abs(first - second))) if len(first) else 0.0)
        repeat_error = max(errors, default=0.0)
        repeat_passed = bool(repeat_error == 0.0)
    metrics_repeat_passed = True if not deterministic_repeat else bool(repeat_passed)
    per_symbol = {
        symbol: _metrics(
            outputs, class_weights, target_scales=target_scales, objective=objective,
            deterministic_repeat_passed=metrics_repeat_passed,
        )
        for symbol, outputs in outputs_by_symbol.items()
    }
    pooled = _metrics(
        pooled_outputs, class_weights, target_scales=target_scales, objective=objective,
        deterministic_repeat_passed=metrics_repeat_passed,
    )
    return {
        "split": split,
        "pooled": pooled,
        "per_symbol": per_symbol,
        "deterministic_repeat_max_absolute_error": repeat_error,
        "deterministic_repeat_passed": repeat_passed,
        "selection_evidence_allowed": split == "validation",
        "auxiliary_metrics_required": objective == OBJECTIVE_NAME,
        "auxiliary_metrics_are_diagnostic_only": objective == LEGACY_OBJECTIVE_NAME,
        "objective": objective,
    }


def train_classification_candidate(
    model: Any,
    datasets_by_symbol: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: Mapping[str, Any],
    objective: str = OBJECTIVE_NAME,
    target_scales: Mapping[str, Any] | None = None,
    objective_contract_digest: str | None = None,
    formulation: Mapping[str, Any] | None = None,
    balance_contract_digest: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train with the resolved objective; retain legacy mode for synthetic research."""
    torch = _torch()
    deterministic = set_deterministic_seed(seed)
    train_dataset = _concat(datasets_by_symbol, "train")
    weights, counts = _class_weights(train_dataset)
    scales = dict(target_scales or training_sequence_target_scales(datasets_by_symbol))
    effective_formulation = dict(formulation or formulation_descriptor("normalized_mse_fixed"))
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=deterministic_loader_generator(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"]), eta_min=float(config["learning_rate"]) * 0.1,
    )
    best_auc, best_loss, best_state, bad = -math.inf, math.inf, None, 0
    epochs: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train().cpu()
        component_sums = {
            "total_loss": 0.0, "classification_loss": 0.0,
            "return_regression_loss": 0.0, "rv_regression_loss": 0.0,
            "weighted_classification_loss": 0.0,
            "weighted_return_loss": 0.0, "weighted_rv_loss": 0.0,
        }
        seen = 0
        for batch in loader:
            outputs = model(batch["x"].cpu())
            if objective == OBJECTIVE_NAME:
                components = resolved_candidate_loss(
                    outputs,
                    {
                        "y_ret_cls": batch["y_ret_cls"].cpu(),
                        "y_ret_reg": batch["y_ret_reg"].cpu(),
                        "y_rv_reg": batch["y_rv_reg"].cpu(),
                    },
                    ret_scale=float(scales["ret_target_scale"]),
                    rv_scale=float(scales["rv_target_scale"]),
                    class_weights=weights, formulation=effective_formulation,
                )
            elif objective == LEGACY_OBJECTIVE_NAME:
                cls = torch.nn.CrossEntropyLoss(weight=weights)(
                    outputs["ret_cls_logits"], batch["y_ret_cls"].cpu()
                )
                zero = cls.detach() * 0.0
                components = {
                    "total_loss": cls, "classification_loss": cls,
                    "return_regression_loss": zero, "rv_regression_loss": zero,
                    "weighted_classification_loss": cls,
                    "weighted_return_loss": zero, "weighted_rv_loss": zero,
                }
            else:
                raise ModelCandidateTrainingError("unknown candidate objective")
            loss = components["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clipping"]))
            optimizer.step()
            batch_rows = len(batch["y_ret_cls"])
            for name in component_sums:
                component_sums[name] += float(components[name].item()) * batch_rows
            seen += batch_rows
        scheduler.step()
        validation = evaluate_classification_model(
            model, datasets_by_symbol, "validation", batch_size=int(config["batch_size"]),
            class_weights=weights.tolist(),
            target_scales=scales, objective=objective,
        )
        if validation["pooled"]["auc"] is None:
            raise ModelCandidateTrainingError("validation_failed")
        auc = float(validation["pooled"]["auc"])
        val_loss = float(validation["pooled"]["classification_loss"])
        epochs.append({
            "epoch": epoch,
            "training_loss": None if not seen else component_sums["total_loss"] / seen,
            "training_loss_components": {
                name: None if not seen else value / seen for name, value in component_sums.items()
            },
            "learning_rate": float(optimizer.param_groups[0]["lr"]), "validation": validation,
        })
        improved = auc > best_auc or (auc == best_auc and val_loss < best_loss)
        if improved:
            best_auc, best_loss = auc, val_loss
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= int(config["patience"]):
                break
    if best_state is None:
        raise ModelCandidateTrainingError("training_failed")
    model.load_state_dict(best_state)
    selected_validation = evaluate_classification_model(
        model, datasets_by_symbol, "validation", batch_size=int(config["batch_size"]),
        deterministic_repeat=True, class_weights=weights.tolist(),
        target_scales=scales, objective=objective,
    )
    history = {
        "seed": int(seed), "epochs": epochs, "best_epoch_auc": best_auc,
        "best_epoch_loss": best_loss, "epochs_completed": len(epochs),
        "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "deterministic_configuration": deterministic,
        "training_objective": training_objective_contract(
            objective, target_scales=scales, objective_contract_digest=objective_contract_digest,
            formulation=effective_formulation, balance_contract_digest=balance_contract_digest,
        ),
        "target_scales": scales,
    }
    return history, selected_validation


def validation_gate(metrics: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(policy or load_training_policy())
    reasons: list[str] = []
    pooled = metrics.get("pooled", {})
    if pooled.get("auc") is None or float(pooled["auc"]) < float(policy["minimum_validation_auc_pooled"]):
        reasons.append("pooled_validation_auc_below_gate")
    for symbol in policy["required_symbols"]:
        value = metrics.get("per_symbol", {}).get(symbol, {})
        if value.get("auc") is None or float(value["auc"]) < float(policy["minimum_validation_auc_per_symbol"]):
            reasons.append(f"{symbol}_validation_auc_below_gate")
        if set(value.get("class_counts", {})) != {"0", "1"} or min(value.get("class_counts", {}).values(), default=0) <= 0:
            reasons.append(f"{symbol}_validation_requires_both_classes")
        if "auxiliary_metrics" in value and value.get("auxiliary_head_gate_passed") is not True:
            reasons.append(f"{symbol}_validation_auxiliary_head_gate_failed")
    if "auxiliary_metrics" in pooled and pooled.get("auxiliary_head_gate_passed") is not True:
        reasons.append("pooled_validation_auxiliary_head_gate_failed")
    return {"passed": not reasons, "status": "passed" if not reasons else "validation_failed", "reasons": reasons}


def select_validation_seed(seed_results: Sequence[Mapping[str, Any]], policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select using only each record's ``validation`` member."""
    if not seed_results:
        raise ModelCandidateTrainingError("validation_failed")
    eligible: list[tuple[float, float, int, Mapping[str, Any], dict[str, Any]]] = []
    evidence: list[dict[str, Any]] = []
    for result in seed_results:
        if "seed" not in result or "validation" not in result:
            raise ModelCandidateTrainingError("seed result lacks validation evidence")
        metrics = result["validation"]
        gate = validation_gate(metrics, policy)
        row = {"seed": int(result["seed"]), "validation": metrics, "gate": gate}
        evidence.append(row)
        if gate["passed"]:
            eligible.append((
                -float(metrics["pooled"]["auc"]),
                float(metrics["pooled"]["classification_loss"]),
                int(result["seed"]), result, gate,
            ))
    if not eligible:
        raise ModelCandidateTrainingError("validation_failed")
    eligible.sort(key=lambda item: item[:3])
    selected = eligible[0][3]
    return {
        "selected_seed": int(selected["seed"]),
        "selection_basis": "validation_only",
        "selection_rule": copy.deepcopy(DEFAULT_TRAINING_CONFIG["selection_rule"]),
        "seed_validation_evidence": evidence,
        "internal_test_consulted": False,
        "legacy_repair_set_consulted": False,
        "confirmation_set_consulted": False,
    }


def internal_test_gate(metrics: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(policy or load_training_policy())
    reasons: list[str] = []
    if metrics.get("split") != "internal_test":
        reasons.append("internal_test_split_required")
    pooled = metrics.get("pooled", {})
    if pooled.get("auc") is None or float(pooled["auc"]) < float(policy["minimum_internal_test_auc_pooled"]):
        reasons.append("pooled_internal_test_auc_below_gate")
    for symbol in policy["required_symbols"]:
        value = metrics.get("per_symbol", {}).get(symbol, {})
        if value.get("auc") is None or float(value["auc"]) < float(policy["minimum_internal_test_auc_per_symbol"]):
            reasons.append(f"{symbol}_internal_test_auc_below_gate")
        if int(value.get("nonfinite_outputs", 1)) != 0:
            reasons.append(f"{symbol}_nonfinite_outputs")
        if "auxiliary_metrics" in value and value.get("auxiliary_head_gate_passed") is not True:
            reasons.append(f"{symbol}_auxiliary_head_gate_failed")
    if int(pooled.get("nonfinite_outputs", 1)) != 0:
        reasons.append("pooled_nonfinite_outputs")
    if metrics.get("deterministic_repeat_passed") is not True:
        reasons.append("deterministic_repeat_failed")
    if "auxiliary_metrics" in pooled and pooled.get("auxiliary_head_gate_passed") is not True:
        reasons.append("pooled_auxiliary_head_gate_failed")
    return {"passed": not reasons, "status": "passed" if not reasons else "internal_test_failed", "reasons": reasons}


def candidate_identity(
    kind: str,
    *,
    architecture_config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    scaler_digest: str,
    training_config: Mapping[str, Any],
    seed: int,
    numerical_lock_digest: str,
    training_environment_digest: str,
    training_code_digest: str,
    objective_contract_digest: str = "unresolved_objective_contract",
    balance_contract_digest: str = "unresolved_balance_contract",
) -> tuple[str, dict[str, Any]]:
    identity = {
        "kind": kind,
        "architecture_config": dict(architecture_config),
        "dataset_digest": dataset_manifest["dataset_digest"],
        "feature_digest": dataset_manifest["feature_digest"],
        "label_digest": dataset_manifest["label_digest"],
        "split_digest": dataset_manifest["split_digest"],
        "scaler_digest": scaler_digest,
        "training_config": dict(training_config),
        "seed": int(seed),
        "numerical_lock_digest": numerical_lock_digest,
        "training_environment_digest": training_environment_digest,
        "training_code_digest": training_code_digest,
        "objective_contract_digest": str(objective_contract_digest),
        "balance_contract_digest": str(balance_contract_digest),
    }
    config_digest = json_digest({
        "architecture": identity["architecture_config"],
        "training": identity["training_config"],
        "feature": identity["feature_digest"],
        "label": identity["label_digest"],
        "split": identity["split_digest"],
        "scaler": identity["scaler_digest"],
        "environment": identity["training_environment_digest"],
        "code": identity["training_code_digest"],
        "objective_contract": identity["objective_contract_digest"],
        "balance_contract": identity["balance_contract_digest"],
    })
    candidate_id = f"{kind}_5m_{str(identity['dataset_digest'])[:8]}_{config_digest[:8]}_s{int(seed)}"
    identity["identity_digest"] = json_digest(identity)
    return candidate_id, identity


def candidate_training_code_digest() -> str:
    """Bind identity to both orchestration and pure objective implementation."""
    return json_digest({
        "model_candidate_train.py": file_digest(Path(__file__)),
        "model_candidate_objective.py": file_digest(BASE_DIR / "tools" / "model_candidate_objective.py"),
        "model_objective_contract.py": file_digest(BASE_DIR / "tools" / "model_objective_contract.py"),
        "model_candidate_loss_balance.py": file_digest(BASE_DIR / "tools" / "model_candidate_loss_balance.py"),
    })


def _manifest_digest(value: Mapping[str, Any], field: str) -> str:
    return json_digest({key: item for key, item in value.items() if key != field})


def refresh_artifact_manifest(candidate_directory: Path | str) -> dict[str, Any]:
    root = Path(candidate_directory)
    required = {
        "model_sha256": "model.pt",
        "scaler_sha256": "scaler.joblib",
        "metadata_sha256": "metadata.json",
        "training_manifest_sha256": "training_manifest.json",
        "training_history_sha256": "training_history.json",
        "evaluation_sha256": "evaluation.json",
        "legacy_repair_gate_sha256": "legacy_repair_gate.json",
        "confirmation_health_gate_sha256": "confirmation_health_gate.json",
    }
    hashes = {field: file_digest(root / filename) for field, filename in required.items()}
    value = {
        "schema_version": 1,
        "candidate_id": root.name,
        **hashes,
        "artifact_directory_digest": json_digest({required[field]: digest for field, digest in sorted(hashes.items())}),
        "incumbent_overwrite_attempted": False,
    }
    atomic_write_json(root / "artifact_manifest.json", value)
    return value


def verify_candidate_artifacts(candidate_directory: Path | str) -> dict[str, Any]:
    root = assert_safe_candidate_directory(candidate_directory)
    if root.is_symlink() or not root.is_dir():
        raise ModelCandidateTrainingError("candidate directory unavailable or symlinked")
    manifest = _verify_candidate_artifacts_read_only(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("candidate_id") != root.name or metadata.get("model_kind") not in ALLOWED_KINDS:
        raise ModelCandidateTrainingError("candidate metadata mismatch")
    return manifest


def _verify_candidate_artifacts_read_only(candidate_directory: Path | str) -> dict[str, Any]:
    root = assert_safe_candidate_directory(candidate_directory)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8-sig"))
    filenames = {
        "model_sha256": "model.pt", "scaler_sha256": "scaler.joblib",
        "metadata_sha256": "metadata.json", "training_manifest_sha256": "training_manifest.json",
        "training_history_sha256": "training_history.json", "evaluation_sha256": "evaluation.json",
        "legacy_repair_gate_sha256": "legacy_repair_gate.json",
        "confirmation_health_gate_sha256": "confirmation_health_gate.json",
    }
    if root.is_symlink() or any((root / filename).is_symlink() for filename in (*filenames.values(), "artifact_manifest.json")):
        raise ModelCandidateTrainingError("candidate artifacts may not be symlinks")
    observed = {field: file_digest(root / filename) for field, filename in filenames.items()}
    if any(manifest.get(field) != digest for field, digest in observed.items()):
        raise ModelCandidateTrainingError("candidate artifact manifest mismatch")
    if manifest.get("artifact_directory_digest") != json_digest({filenames[field]: digest for field, digest in sorted(observed.items())}):
        raise ModelCandidateTrainingError("candidate artifact directory digest mismatch")
    return manifest


def _safe_environment_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = training_contract()
    if not TRAINING_PYTHON.is_file() or Path(sys.executable).resolve() != TRAINING_PYTHON.resolve():
        raise ModelCandidateTrainingError("candidate training must use .venv-model-training/canonical/Scripts/python.exe")
    inventory = interpreter_inventory(TRAINING_PYTHON)
    validate_training_inventory(inventory, contract["torch"])
    manifest_path = TRAINING_ENV / ".model-training-manifest.json"
    if not manifest_path.is_file():
        raise ModelCandidateTrainingError("candidate_training_environment_pending")
    environment = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if environment.get("manifest_digest") != json_digest({
        key: value for key, value in environment.items()
        if key not in {"created_at", "manifest_digest"}
    }):
        raise ModelCandidateTrainingError("candidate training environment manifest mismatch")
    return contract, environment


def _candidate_status_from_internal(gate: Mapping[str, Any]) -> str:
    return "confirmation_pending" if gate.get("passed") else "internal_test_failed"


def record_validation_access(
    *, dataset_digest: str, balance_contract_digest: str, balance_freeze_timestamp: str,
    ledger_path: Path | str = VALIDATION_ACCESS_LEDGER,
) -> dict[str, Any]:
    """Record that validation is being opened only after the immutable balance freeze."""
    path = Path(ledger_path)
    value = {"schema_version": 1, "accesses": []}
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = value.setdefault("accesses", [])
    matching = [item for item in entries if item.get("dataset_digest") == dataset_digest]
    if matching:
        if any(item.get("balance_contract_digest") != balance_contract_digest for item in matching):
            raise ModelCandidateTrainingError("validation access balance contract mismatch")
        if any(str(item.get("first_access_at", "")) < str(balance_freeze_timestamp) for item in matching):
            raise ModelCandidateTrainingError("balance_freeze_contaminated")
        return matching[0]
    record = {
        "dataset_digest": dataset_digest,
        "balance_contract_digest": balance_contract_digest,
        "first_access_at": utc_now(),
        "access_type": "validation_seed_selection",
        "balance_frozen_before_access": True,
        "balance_freeze_timestamp": str(balance_freeze_timestamp),
    }
    entries.append(record)
    value["ledger_digest"] = json_digest({key: item for key, item in value.items() if key != "ledger_digest"})
    atomic_write_json(path, value)
    return record


def _write_selected_candidate(
    *,
    kind: str,
    selected_model: Any,
    dataset_root: Path,
    dataset_manifest: Mapping[str, Any],
    architecture: Mapping[str, Any],
    config: Mapping[str, Any],
    selected_seed: int,
    selection: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    internal_metrics: Mapping[str, Any],
    internal_gate_result: Mapping[str, Any],
    environment_manifest: Mapping[str, Any],
    numerical_contract: Mapping[str, Any],
    started_at: str,
    candidate_root: Path,
    frozen_model_path: Path | None = None,
    expected_candidate_id: str | None = None,
    objective: str = OBJECTIVE_NAME,
    objective_contract_report: Mapping[str, Any] | None = None,
    target_scales: Mapping[str, Any] | None = None,
    balance_freeze: Mapping[str, Any] | None = None,
    formulation: Mapping[str, Any] | None = None,
) -> Path:
    torch = _torch()
    if objective != OBJECTIVE_NAME:
        raise ModelCandidateTrainingError("classification-only mode cannot finalize a Phase 24 candidate")
    if objective_contract_report is None or target_scales is None or balance_freeze is None or formulation is None:
        raise ModelCandidateTrainingError("resolved objective, balance freeze, and training target scales required")
    objective_contract_digest = str(objective_contract_report["objective_contract_digest"])
    balance_contract_digest = str(balance_freeze["balance_contract_digest"])
    training_code_digest = candidate_training_code_digest()
    scaler_digest = file_digest(dataset_root / "scaler.joblib")
    candidate_id, identity = candidate_identity(
        kind,
        architecture_config=architecture["constructor"], dataset_manifest=dataset_manifest,
        scaler_digest=scaler_digest, training_config=config, seed=selected_seed,
        numerical_lock_digest=numerical_contract["canonical_numerical_lock_digest"],
        training_environment_digest=environment_manifest["manifest_digest"],
        training_code_digest=training_code_digest,
        objective_contract_digest=objective_contract_digest,
        balance_contract_digest=balance_contract_digest,
    )
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise ModelCandidateTrainingError("frozen candidate identity changed after internal-test access")
    target = assert_safe_candidate_directory(candidate_root / candidate_id)
    if target.exists():
        raise ModelCandidateTrainingError("finalized candidate overwrite prohibited")
    candidate_root.mkdir(parents=True, exist_ok=True)
    if candidate_root.is_symlink():
        raise ModelCandidateTrainingError("candidate root may not be a symlink")
    stage = Path(tempfile.mkdtemp(prefix=f".{candidate_id}.partial-", dir=candidate_root))
    completed_at = utc_now()
    try:
        if frozen_model_path is not None:
            shutil.copyfile(frozen_model_path, stage / "model.pt")
        else:
            torch.save(selected_model.state_dict(), stage / "model.pt")
        shutil.copyfile(dataset_root / "scaler.joblib", stage / "scaler.joblib")
        if file_digest(stage / "scaler.joblib") != scaler_digest:
            raise ModelCandidateTrainingError("candidate scaler copy mismatch")
        model_digest = file_digest(stage / "model.pt")
        incumbent_digest = incumbent_hashes()[f"model_artifacts/dl_{kind}_latest.pt"]
        objective_record = training_objective_contract(
            objective, target_scales=target_scales,
            objective_contract_digest=objective_contract_digest,
            formulation=formulation, balance_contract_digest=balance_contract_digest,
        )
        auxiliary_audit = downstream_auxiliary_head_audit()
        metadata: dict[str, Any] = {
            "schema_version": 1, "candidate_id": candidate_id, "model_kind": kind,
            "model_sha256": model_digest, "scaler_sha256": scaler_digest,
            "supported_symbols": list(dataset_manifest["supported_symbols"]), "timeframe": "5m",
            "sequence_length": int(dataset_manifest["sequence_length"]), "feature_count": 27,
            "architecture_config": architecture["constructor"], "architecture_contract": architecture,
            "training_objective": objective_record, "objective_contract": objective_record,
            "objective_source": objective_record["objective_source"],
            "objective_schema_version": objective_record["objective_schema_version"],
            "objective_policy_digest": objective_record["objective_policy_digest"],
            "objective_contract_digest": objective_contract_digest,
            "parent_objective_contract_digest": objective_contract_digest,
            "loss_balance_policy_digest": balance_freeze["balance_policy_digest"],
            "balance_contract_digest": balance_contract_digest,
            "balance_calibration_sample_digest": balance_freeze["calibration_sample_digest"],
            "selected_loss_formulation": formulation["formulation_id"],
            "task_weights": {
                "classification": formulation["classification_weight"],
                "return": formulation["return_weight"], "rv": formulation["rv_weight"],
            },
            "classification_weight": formulation["classification_weight"],
            "return_weight": formulation["return_weight"], "rv_weight": formulation["rv_weight"],
            "huber_beta": formulation["huber_beta"],
            "ret_target_scale": target_scales["ret_target_scale"],
            "rv_target_scale": target_scales["rv_target_scale"],
            "ret_target_definition": objective_contract_report["return_target"],
            "rv_target_definition": objective_contract_report["volatility_target"],
            "classification_target_definition": objective_contract_report["classification_target"],
            "ret_horizon_bars": objective_contract_report["return_target"]["horizon_bars"],
            "rv_horizon_bars": objective_contract_report["volatility_target"]["horizon_bars"],
            "classification_max_hold_bars": objective_contract_report["classification_target"]["horizon_bars"],
            "auxiliary_head_training_status": AUXILIARY_STATUS,
            "objective_contract_blocker": False,
            "candidate_auxiliary_health_blocker": "unverified",
            "downstream_contract_blocker": objective_contract_report["promotion_blockers"]["downstream_contract_blocker"],
            "candidate_auxiliary_head_promotion_blocker": True,
            "selected_seed": int(selected_seed), "candidate_identity": identity,
            "candidate_status": _candidate_status_from_internal(internal_gate_result),
            "candidate_model_frozen": True, "candidate_scaler_frozen": True,
            "promotion_allowed": False, "live_activation_allowed": False,
        }
        metadata["metadata_digest"] = _manifest_digest(metadata, "metadata_digest")
        atomic_write_json(stage / "metadata.json", metadata)
        per_symbol = dataset_manifest["per_symbol"]
        rows = {
            split: {symbol: per_symbol[symbol]["rows_by_split"][split] for symbol in dataset_manifest["symbols"]}
            for split in ("train", "validation", "internal_test")
        }
        sequences = {
            split: {symbol: per_symbol[symbol]["valid_sequences_by_split"][split] for symbol in dataset_manifest["symbols"]}
            for split in ("train", "validation", "internal_test")
        }
        training_manifest: dict[str, Any] = {
            "schema_version": 1, "candidate_id": candidate_id, "model_kind": kind,
            "incumbent_parent_digest": incumbent_digest, "git_commit": git_commit(),
            "training_code_digest": training_code_digest, "model_code_digest": architecture["model_code_digest"],
            "dataset_id": dataset_manifest["dataset_id"], "raw_data_digest": dataset_manifest["raw_data_digest"],
            "feature_digest": dataset_manifest["feature_digest"], "label_digest": dataset_manifest["label_digest"],
            "split_digest": dataset_manifest["split_digest"], "scaler_digest": scaler_digest,
            "scaler_fit_contract": "pooled_required_symbols_training_rows_only",
            "symbol_id_map": dataset_manifest["symbol_id_map"],
            "supported_symbols": dataset_manifest["supported_symbols"], "timeframe": "5m",
            "sequence_length": dataset_manifest["sequence_length"], "feature_count": dataset_manifest["feature_count"],
            "architecture_config": architecture["constructor"], "training_objective": objective_record,
            "objective_source": objective_record["objective_source"],
            "objective_schema_version": objective_record["objective_schema_version"],
            "objective_policy_digest": objective_record["objective_policy_digest"],
            "objective_contract_digest": objective_contract_digest,
            "parent_objective_contract_digest": objective_contract_digest,
            "loss_balance_policy_digest": balance_freeze["balance_policy_digest"],
            "balance_contract_digest": balance_contract_digest,
            "balance_calibration_sample_digest": balance_freeze["calibration_sample_digest"],
            "selected_loss_formulation": formulation["formulation_id"],
            "classification_weight": formulation["classification_weight"],
            "return_weight": formulation["return_weight"], "rv_weight": formulation["rv_weight"],
            "huber_beta": formulation["huber_beta"],
            "balance_statistics": balance_freeze["architectures"][kind]["balance_statistics"],
            "heterogeneous_architecture_objectives": balance_freeze["heterogeneous_architecture_objectives"],
            "ret_target_scale": target_scales["ret_target_scale"],
            "rv_target_scale": target_scales["rv_target_scale"],
            "target_scale_contract": dict(target_scales),
            "ret_target_definition": objective_contract_report["return_target"],
            "rv_target_definition": objective_contract_report["volatility_target"],
            "classification_target_definition": objective_contract_report["classification_target"],
            "ret_horizon_bars": objective_contract_report["return_target"]["horizon_bars"],
            "rv_horizon_bars": objective_contract_report["volatility_target"]["horizon_bars"],
            "classification_max_hold_bars": objective_contract_report["classification_target"]["horizon_bars"],
            "auxiliary_head_training_status": AUXILIARY_STATUS, "seed": selected_seed,
            "optimizer": config["optimizer"], "scheduler": config["scheduler"],
            "batch_size": config["batch_size"], "epochs": config["epochs"], "patience": config["patience"],
            "gradient_clipping": config["gradient_clipping"], "class_weighting": config["class_weighting"],
            "train_rows": rows["train"], "train_sequences": sequences["train"],
            "validation_rows": rows["validation"], "validation_sequences": sequences["validation"],
            "test_rows": rows["internal_test"], "test_sequences": sequences["internal_test"],
            "purge_count": {symbol: per_symbol[symbol]["purged_rows"] for symbol in dataset_manifest["symbols"]},
            "scaler_fit_rows": dataset_manifest["scaler"]["fit_rows"],
            "environment_versions": {
                "python": environment_manifest["python_version"], "torch": environment_manifest["torch_version"],
                "numpy": environment_manifest["numpy_version"], "scipy": environment_manifest["scipy_version"],
                "sklearn": environment_manifest["sklearn_version"], "joblib": environment_manifest["joblib_version"],
                "threadpoolctl": environment_manifest["threadpoolctl_version"],
            },
            "training_lock_digest": environment_manifest["training_lock_digest"],
            "deterministic_configuration": next(
                row["history"]["deterministic_configuration"] for row in seed_results if int(row["seed"]) == selected_seed
            ),
            "started_at": started_at, "completed_at": completed_at,
        }
        training_manifest["manifest_digest"] = _manifest_digest(training_manifest, "manifest_digest")
        atomic_write_json(stage / "training_manifest.json", training_manifest)
        history = {
            "schema_version": 1, "candidate_id": candidate_id,
            "selection": selection, "per_seed": list(seed_results),
            "selection_used_validation_only": True,
        }
        history["history_digest"] = _manifest_digest(history, "history_digest")
        atomic_write_json(stage / "training_history.json", history)
        evaluation = {
            "schema_version": 1, "candidate_id": candidate_id,
            "internal_test_access_count": 1, "internal_test_first_access_at": completed_at,
            "internal_test": internal_metrics, "internal_test_gate": internal_gate_result,
            "post_test_tuning_allowed": False, "legacy_incumbent_comparison": "pending",
            "profitability_evidence": False,
        }
        evaluation["evaluation_digest"] = _manifest_digest(evaluation, "evaluation_digest")
        atomic_write_json(stage / "evaluation.json", evaluation)
        atomic_write_json(stage / "legacy_repair_gate.json", {
            "schema_version": 1, "candidate_id": candidate_id,
            "status": "pending" if internal_gate_result["passed"] else "not_run_internal_test_failed",
        })
        atomic_write_json(stage / "confirmation_health_gate.json", {
            "schema_version": 1, "candidate_id": candidate_id, "status": "confirmation_pending",
        })
        stage.replace(target)
        refresh_artifact_manifest(target)
        return target
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _update_summary_and_maybe_freeze(candidate: Path, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8-sig"))
    evaluation = json.loads((candidate / "evaluation.json").read_text(encoding="utf-8-sig"))
    history = json.loads((candidate / "training_history.json").read_text(encoding="utf-8-sig"))
    artifacts = json.loads((candidate / "artifact_manifest.json").read_text(encoding="utf-8-sig"))
    environment_path = TRAINING_ENV / ".model-training-manifest.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8-sig"))
    summary = {
        "schema_version": 1, "phase": 24, "environment": environment,
        "dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_digest": dataset_manifest["dataset_digest"],
            "raw_data_digest": dataset_manifest["raw_data_digest"],
            "feature_digest": dataset_manifest["feature_digest"],
            "label_digest": dataset_manifest["label_digest"],
        },
        "split": dataset_manifest["split"],
        "scaler": dataset_manifest["scaler"], "models": {},
        "auxiliary_head_objective_warning": downstream_auxiliary_head_audit(),
        "profitability_evidence": False,
    }
    if TRAINING_SUMMARY.is_file():
        summary = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8-sig"))
    summary.setdefault("models", {})[metadata["model_kind"]] = {
        "candidate_id": metadata["candidate_id"], "candidate_directory": candidate.relative_to(BASE_DIR).as_posix(),
        "candidate_model_digest": metadata["model_sha256"], "candidate_scaler_digest": metadata["scaler_sha256"],
        "dataset_id": dataset_manifest["dataset_id"], "dataset_digest": dataset_manifest["dataset_digest"],
        "per_model_seed_metrics": [
            {"seed": row["seed"], "validation": row["validation"]} for row in history["per_seed"]
        ],
        "selected_seed": metadata["selected_seed"], "internal_test": evaluation["internal_test"],
        "status": metadata["candidate_status"], "legacy_phase22_repair_result": "pending",
        "sealed_confirmation_result": "pending", "registry_proposal": "pending",
        "candidate_artifacts": artifacts, "incumbent_comparisons": "pending",
        "auxiliary_head_objective_warning": {
            "status": metadata["auxiliary_head_training_status"],
            "objective_contract_digest": metadata["objective_contract_digest"],
            "objective_contract_blocker": metadata["objective_contract_blocker"],
            "candidate_auxiliary_health_blocker": metadata["candidate_auxiliary_health_blocker"],
            "downstream_contract_blocker": metadata["downstream_contract_blocker"],
            "candidate_auxiliary_head_promotion_blocker": metadata["candidate_auxiliary_head_promotion_blocker"],
        },
    }
    statuses = [value["status"] for value in summary["models"].values()]
    summary["final_decision"] = "candidate_training_complete_confirmation_pending" if len(statuses) == 3 else "candidate_training_data_pending"
    summary["artifact_integrity"] = {"incumbents_unchanged": True, "candidates_immutable_models": True}
    summary["summary_digest"] = _manifest_digest(summary, "summary_digest")
    atomic_write_json(TRAINING_SUMMARY, summary)
    if set(summary["models"]) == set(ALLOWED_KINDS) and not SELECTION_FREEZE.exists():
        dataset_ids = {value.get("dataset_id") for value in summary["models"].values()}
        dataset_digests = {value.get("dataset_digest") for value in summary["models"].values()}
        scalers = {value.get("candidate_scaler_digest") for value in summary["models"].values()}
        if len(dataset_ids) != 1 or len(dataset_digests) != 1 or len(scalers) != 1:
            raise ModelCandidateTrainingError("selected candidates do not share one frozen dataset/scaler")
        freeze = {
            "schema_version": 1, "selection_frozen": True,
            "dataset_id": dataset_manifest["dataset_id"], "dataset_digest": dataset_manifest["dataset_digest"],
            "source_venue": dataset_manifest["source_venue"],
            "candidates": {
                kind: {
                    "candidate_id": summary["models"][kind]["candidate_id"],
                    "candidate_model_digest": summary["models"][kind]["candidate_model_digest"],
                    "candidate_scaler_digest": summary["models"][kind]["candidate_scaler_digest"],
                    "internal_test_recorded": True,
                }
                for kind in ALLOWED_KINDS
            },
            "frozen_at": utc_now(),
        }
        freeze["freeze_digest"] = json_digest({key: value for key, value in freeze.items() if key not in {"frozen_at", "freeze_digest"}})
        atomic_write_json(SELECTION_FREEZE, freeze)
    return summary


def train_candidate_experiment(
    kind: str,
    dataset: Path | str,
    *,
    candidate_root: Path | str = CANDIDATE_ROOT,
    config: Mapping[str, Any] | None = None,
    objective: str = OBJECTIVE_NAME,
    objective_report: Path | str = OBJECTIVE_REPORT,
    balance_freeze_path: Path | str = DEFAULT_BALANCE_FREEZE,
) -> dict[str, Any]:
    if kind not in ALLOWED_KINDS:
        raise ModelCandidateTrainingError("ADV is retained and may not be retrained in Phase 24")
    objective_contract_report = validate_training_objective_gate(
        objective, report_path=objective_report
    )
    try:
        balance_freeze = validate_balance_freeze(balance_freeze_path)
    except LossBalanceError as exc:
        raise ModelCandidateTrainingError(str(exc)) from exc
    validate_phase24_evidence()
    policy = load_training_policy()
    numerical, environment = _safe_environment_contract()
    record_incumbent_inventory()
    started_at = utc_now()
    datasets, manifest = load_sequence_datasets(dataset)
    try:
        balance_freeze = validate_balance_freeze(balance_freeze_path, dataset_manifest=manifest)
    except LossBalanceError as exc:
        raise ModelCandidateTrainingError(str(exc)) from exc
    architecture = architecture_contract(kind)
    training_config = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    if config:
        training_config.update(dict(config))
    formulation = dict(balance_freeze["architectures"][kind]["selected_descriptor"])
    expected_weights = {
        "classification_weight": formulation["classification_weight"],
        "return_weight": formulation["return_weight"], "rv_weight": formulation["rv_weight"],
    }
    if config and any(name in config and float(config[name]) != float(value) for name, value in expected_weights.items()):
        raise ModelCandidateTrainingError("training task weights differ from frozen balance contract")
    training_config.update(expected_weights)
    training_config["selected_loss_formulation"] = formulation["formulation_id"]
    training_config["huber_beta"] = formulation["huber_beta"]
    target_scales = training_sequence_target_scales(datasets)
    if manifest.get("target_scales") != target_scales:
        raise ModelCandidateTrainingError("frozen training-sequence target scales mismatch")
    objective_contract_digest = str(objective_contract_report["objective_contract_digest"])
    balance_contract_digest = str(balance_freeze["balance_contract_digest"])
    record_validation_access(
        dataset_digest=manifest["dataset_digest"], balance_contract_digest=balance_contract_digest,
        balance_freeze_timestamp=balance_freeze["freeze_timestamp"],
    )
    seed_results: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    for seed in policy["training_seeds"]:
        set_deterministic_seed(seed)
        model = make_candidate_model(kind, architecture["constructor"]).cpu()
        history, validation = train_classification_candidate(
            model, datasets, seed=seed, config=training_config, objective=objective,
            target_scales=target_scales,
            objective_contract_digest=objective_contract_digest,
            formulation=formulation, balance_contract_digest=balance_contract_digest,
        )
        seed_results.append({"seed": seed, "validation": validation, "history": history})
        models[int(seed)] = model
        verify_incumbent_inventory()
    selection = select_validation_seed(seed_results, policy)
    selected_seed = int(selection["selected_seed"])
    selected_model = models[selected_seed]
    # Freeze identity, exact model bytes, and scaler digest before opening test.
    training_code_digest = candidate_training_code_digest()
    candidate_id, candidate_identity_record = candidate_identity(
        kind, architecture_config=architecture["constructor"], dataset_manifest=manifest,
        scaler_digest=file_digest(Path(dataset) / "scaler.joblib"), training_config=training_config,
        seed=selected_seed, numerical_lock_digest=numerical["canonical_numerical_lock_digest"],
        training_environment_digest=environment["manifest_digest"],
        training_code_digest=training_code_digest,
        objective_contract_digest=objective_contract_digest,
        balance_contract_digest=balance_contract_digest,
    )
    freeze_directory = SEED_RUN_ROOT / candidate_id
    if freeze_directory.exists():
        raise ModelCandidateTrainingError("candidate experiment contract already observed internal test")
    freeze_directory.mkdir(parents=True, exist_ok=False)
    frozen_model_path = freeze_directory / "selected_model.pt"
    _torch().save(selected_model.state_dict(), frozen_model_path)
    selected_freeze = {
        "schema_version": 1, "candidate_id": candidate_id,
        "candidate_identity": candidate_identity_record,
        "candidate_model_digest": file_digest(frozen_model_path),
        "candidate_scaler_digest": file_digest(Path(dataset) / "scaler.joblib"),
        "selection": selection, "internal_test_accessed": False,
    }
    selected_freeze["freeze_digest"] = _manifest_digest(selected_freeze, "freeze_digest")
    atomic_write_json(freeze_directory / "selected_freeze.json", selected_freeze)
    # Candidate contract is immutable at this point. Internal test is opened once.
    selected_freeze["internal_test_accessed"] = True
    selected_freeze["internal_test_first_access_at"] = utc_now()
    selected_freeze["freeze_digest"] = _manifest_digest(selected_freeze, "freeze_digest")
    atomic_write_json(freeze_directory / "selected_freeze.json", selected_freeze)
    internal_weights, _ = _class_weights(_concat(datasets, "train"))
    internal = evaluate_classification_model(
        selected_model, datasets, "internal_test", batch_size=int(training_config["batch_size"]),
        deterministic_repeat=True, class_weights=internal_weights.tolist(),
        target_scales=target_scales, objective=objective,
    )
    internal["selection_evidence_allowed"] = False
    gate = internal_test_gate(internal, policy)
    target = _write_selected_candidate(
        kind=kind, selected_model=selected_model, dataset_root=Path(dataset), dataset_manifest=manifest,
        architecture=architecture, config=training_config, selected_seed=selected_seed,
        selection=selection, seed_results=seed_results, internal_metrics=internal,
        internal_gate_result=gate, environment_manifest=environment, numerical_contract=numerical,
        started_at=started_at, candidate_root=Path(candidate_root),
        frozen_model_path=frozen_model_path, expected_candidate_id=candidate_id,
        objective=objective, objective_contract_report=objective_contract_report,
        target_scales=target_scales, balance_freeze=balance_freeze, formulation=formulation,
    )
    verify_incumbent_inventory()
    summary = _update_summary_and_maybe_freeze(target, manifest)
    return {
        "status": _candidate_status_from_internal(gate), "candidate_id": target.name,
        "candidate_directory": target.as_posix(), "selected_seed": selected_seed,
        "internal_test_gate": gate, "selection": selection, "summary_digest": summary["summary_digest"],
        "objective_contract_digest": objective_contract_digest,
        "balance_contract_digest": balance_contract_digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ALLOWED_KINDS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--objective", choices=(OBJECTIVE_NAME, LEGACY_OBJECTIVE_NAME), default=OBJECTIVE_NAME,
        help="classification_only_legacy is research-only and cannot finalize a candidate",
    )
    parser.add_argument("--objective-contract", default=str(OBJECTIVE_REPORT))
    parser.add_argument("--balance-freeze", default=str(DEFAULT_BALANCE_FREEZE))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_candidate_experiment(
            args.model, args.dataset, objective=args.objective,
            objective_report=args.objective_contract,
            balance_freeze_path=args.balance_freeze,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] != "internal_test_failed" else 3
    except (ModelCandidateTrainingError, CandidateTrainingEnvironmentError) as exc:
        status = str(exc) if str(exc) in {"training_failed", "validation_failed", "internal_test_failed"} else "training_failed"
        print(json.dumps({"status": status, "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
