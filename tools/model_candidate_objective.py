"""Pure Phase 24.1 candidate-objective helpers.

The loss functions read no environment variables and write no artifacts.  Model
outputs remain in their raw target units; only residuals are normalized.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
OBJECTIVE_POLICY = BASE_DIR / "research" / "model_candidate_objective_policy.json"
OBJECTIVE_NAME = "resolved_candidate_objective"
LEGACY_OBJECTIVE_NAME = "classification_only_legacy"
RESOLVED_AUXILIARY_STATUS = "auxiliary_heads_optimized_under_resolved_candidate_objective"
LEGACY_AUXILIARY_STATUS = "auxiliary_unoptimized_under_legacy_objective"


class CandidateObjectiveError(ValueError):
    """The objective policy, targets, loss inputs, or metric inputs are invalid."""


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_objective_policy(path: Path | str = OBJECTIVE_POLICY) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_objective_contract(value)
    return value


def validate_objective_contract(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact Phase 24.1 policy fields, values, and JSON types."""
    expected = {
        "schema_version": 1,
        "classification": {
            "output": "ret_cls_logits", "target": "y_ret_cls",
            "loss": "weighted_cross_entropy", "weight": 1.0,
        },
        "return_regression": {
            "output": "ret_reg", "target": "y_ret_reg",
            "loss": "train_scale_normalized_mse", "weight": 0.5,
            "expected_signed": True,
        },
        "volatility_regression": {
            "output": "rv_reg", "target": "y_rv_reg",
            "loss": "train_scale_normalized_mse", "weight": 0.5,
            "expected_nonnegative": True, "allow_negative_prediction": False,
        },
        "target_scale_source": "training_sequences_only",
        "minimum_target_scale": 1e-12,
        "require_validation_auxiliary_metrics": True,
        "require_internal_test_auxiliary_metrics": True,
        "require_repair_set_auxiliary_metrics": True,
        "require_confirmation_auxiliary_metrics": True,
        "candidate_only": True,
        "modify_incumbent_training": False,
        "modify_live_inference": False,
        "modify_live_risk": False,
        "promotion_allowed": False,
        "live_activation_allowed": False,
    }
    if set(policy) != set(expected):
        raise CandidateObjectiveError("objective policy fields mismatch")
    if dict(policy) != expected:
        raise CandidateObjectiveError("objective policy values or nested fields mismatch")
    # bool is an int subclass, so make the schema's primitive-type checks explicit.
    if type(policy["schema_version"]) is not int:
        raise CandidateObjectiveError("objective schema_version must be an integer")
    for section in ("classification", "return_regression", "volatility_regression"):
        if type(policy[section]["weight"]) is not float:
            raise CandidateObjectiveError("objective weights must be JSON floats")
    if type(policy["minimum_target_scale"]) is not float:
        raise CandidateObjectiveError("minimum_target_scale must be a JSON float")
    return dict(policy)


def objective_policy_digest(policy: Mapping[str, Any] | None = None) -> str:
    return _json_digest(validate_objective_contract(policy or load_objective_policy()))


def _finite_values(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not len(finite):
        raise CandidateObjectiveError(f"{name} has no finite training-sequence targets")
    return finite


def compute_training_target_scales(
    y_ret_reg: Sequence[float] | np.ndarray,
    y_rv_reg: Sequence[float] | np.ndarray,
    *,
    minimum_scale: float = 1e-12,
) -> dict[str, Any]:
    """Compute population scales from valid training-sequence endpoints only."""
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0:
        raise CandidateObjectiveError("minimum target scale must be finite and positive")
    ret = _finite_values(y_ret_reg, "y_ret_reg")
    rv = _finite_values(y_rv_reg, "y_rv_reg")
    if np.any(rv < 0):
        raise CandidateObjectiveError("rv_target_negative_count must be zero")
    ret_scale = float(np.std(ret, ddof=0))
    rv_scale = float(np.std(rv, ddof=0))
    if not math.isfinite(ret_scale) or not math.isfinite(rv_scale):
        raise CandidateObjectiveError("target scale must be finite")
    if ret_scale <= float(minimum_scale) or rv_scale <= float(minimum_scale):
        raise CandidateObjectiveError("target scale must be greater than minimum_target_scale")
    result = {
        "source": "training_sequences_only",
        "ret_target_scale": ret_scale,
        "rv_target_scale": rv_scale,
        "ret_train_target_mean": float(np.mean(ret)),
        "rv_train_target_mean": float(np.mean(rv)),
        "ret_finite_training_sequence_count": int(len(ret)),
        "rv_finite_training_sequence_count": int(len(rv)),
        "rv_target_negative_count": 0,
        "minimum_target_scale": float(minimum_scale),
    }
    result["target_scale_digest"] = _json_digest(result)
    return result


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover - training environment prerequisite
        raise CandidateObjectiveError("PyTorch is required for objective loss calculation") from exc
    return torch, functional


def classification_loss(ret_cls_logits: Any, y_ret_cls: Any, class_weights: Any | None = None):
    """Exactly the legacy weighted CrossEntropyLoss formulation."""
    _, functional = _torch_modules()
    return functional.cross_entropy(ret_cls_logits, y_ret_cls, weight=class_weights)


def _valid_scale(scale: float, name: str, minimum_scale: float = 1e-12) -> float:
    value = float(scale)
    if not math.isfinite(value) or value <= float(minimum_scale):
        raise CandidateObjectiveError(f"{name} must be finite and greater than minimum_target_scale")
    return value


def normalized_return_loss(ret_reg: Any, y_ret_reg: Any, ret_scale: float):
    scale = _valid_scale(ret_scale, "ret_target_scale")
    return (((ret_reg - y_ret_reg) / scale) ** 2).mean()


def normalized_rv_loss(rv_reg: Any, y_rv_reg: Any, rv_scale: float):
    torch, _ = _torch_modules()
    scale = _valid_scale(rv_scale, "rv_target_scale")
    if bool(torch.any(y_rv_reg < 0).item()):
        raise CandidateObjectiveError("rv_target_negative_count must be zero")
    return (((rv_reg - y_rv_reg) / scale) ** 2).mean()


def normalized_huber_return_loss(
    ret_reg: Any, y_ret_reg: Any, ret_scale: float, *, beta: float = 1.0,
):
    """Smooth-L1 on the normalized residual; predictions stay in raw units."""
    _, functional = _torch_modules()
    scale = _valid_scale(ret_scale, "ret_target_scale")
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise CandidateObjectiveError("huber beta must be finite and positive")
    residual = (ret_reg - y_ret_reg) / scale
    return functional.smooth_l1_loss(residual, residual.new_zeros(residual.shape), beta=float(beta))


def normalized_huber_rv_loss(
    rv_reg: Any, y_rv_reg: Any, rv_scale: float, *, beta: float = 1.0,
):
    """Smooth-L1 on normalized RV residuals without clipping raw outputs."""
    torch, functional = _torch_modules()
    scale = _valid_scale(rv_scale, "rv_target_scale")
    if bool(torch.any(y_rv_reg < 0).item()):
        raise CandidateObjectiveError("rv_target_negative_count must be zero")
    if not math.isfinite(float(beta)) or float(beta) <= 0:
        raise CandidateObjectiveError("huber beta must be finite and positive")
    residual = (rv_reg - y_rv_reg) / scale
    return functional.smooth_l1_loss(residual, residual.new_zeros(residual.shape), beta=float(beta))


def candidate_multitask_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    ret_scale: float,
    rv_scale: float,
    class_weights: Any | None = None,
    classification_weight: float = 1.0,
    return_weight: float = 0.5,
    rv_weight: float = 0.5,
) -> dict[str, Any]:
    """Return total and unweighted components without changing output units."""
    for weight, name in (
        (classification_weight, "classification_weight"),
        (return_weight, "return_weight"),
        (rv_weight, "rv_weight"),
    ):
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise CandidateObjectiveError(f"{name} must be finite and non-negative")
    cls = classification_loss(outputs["ret_cls_logits"], targets["y_ret_cls"], class_weights)
    ret = normalized_return_loss(outputs["ret_reg"], targets["y_ret_reg"], ret_scale)
    rv = normalized_rv_loss(outputs["rv_reg"], targets["y_rv_reg"], rv_scale)
    total = float(classification_weight) * cls + float(return_weight) * ret + float(rv_weight) * rv
    return {
        "total_loss": total,
        "classification_loss": cls,
        "return_regression_loss": ret,
        "rv_regression_loss": rv,
    }


def resolved_candidate_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    ret_scale: float,
    rv_scale: float,
    formulation: Mapping[str, Any],
    class_weights: Any | None = None,
) -> dict[str, Any]:
    """Evaluate an explicit immutable Phase 24.2 formulation descriptor."""
    required = {
        "formulation_id", "classification_weight", "return_weight", "rv_weight", "huber_beta",
    }
    if set(formulation) < required:
        raise CandidateObjectiveError("incomplete frozen formulation descriptor")
    formulation_id = str(formulation["formulation_id"])
    if formulation_id not in {
        "normalized_mse_fixed", "normalized_huber_fixed", "normalized_huber_training_balanced",
    }:
        raise CandidateObjectiveError("unknown loss-balance formulation")
    weights = {
        "classification_weight": float(formulation["classification_weight"]),
        "return_weight": float(formulation["return_weight"]),
        "rv_weight": float(formulation["rv_weight"]),
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise CandidateObjectiveError("task weights must be finite and non-negative")
    cls = classification_loss(outputs["ret_cls_logits"], targets["y_ret_cls"], class_weights)
    if formulation_id == "normalized_mse_fixed":
        ret = normalized_return_loss(outputs["ret_reg"], targets["y_ret_reg"], ret_scale)
        rv = normalized_rv_loss(outputs["rv_reg"], targets["y_rv_reg"], rv_scale)
    else:
        beta = float(formulation["huber_beta"])
        ret = normalized_huber_return_loss(outputs["ret_reg"], targets["y_ret_reg"], ret_scale, beta=beta)
        rv = normalized_huber_rv_loss(outputs["rv_reg"], targets["y_rv_reg"], rv_scale, beta=beta)
    weighted_cls = weights["classification_weight"] * cls
    weighted_ret = weights["return_weight"] * ret
    weighted_rv = weights["rv_weight"] * rv
    return {
        "total_loss": weighted_cls + weighted_ret + weighted_rv,
        "classification_loss": cls,
        "return_regression_loss": ret,
        "rv_regression_loss": rv,
        "weighted_classification_loss": weighted_cls,
        "weighted_return_loss": weighted_ret,
        "weighted_rv_loss": weighted_rv,
    }


def candidate_objective_digest(
    *,
    policy: Mapping[str, Any] | None = None,
    target_contract_digest: str,
    objective_source: str = "new_candidate_only_contract",
) -> str:
    value = policy or load_objective_policy()
    return _json_digest({
        "objective_source": objective_source,
        "objective_policy": dict(value),
        "target_contract_digest": str(target_contract_digest),
    })


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(target) < 2 or np.std(target) <= 1e-15 or np.std(prediction) <= 1e-15:
        return None
    return float(np.corrcoef(target, prediction)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _head_metrics(
    prediction: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    *,
    scale: float,
    train_target_mean: float,
    volatility: bool,
    deterministic_repeat_passed: bool,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    true = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(pred) != len(true):
        raise CandidateObjectiveError("prediction and target lengths differ")
    valid_scale = _valid_scale(scale, "target_scale")
    finite_mask = np.isfinite(pred) & np.isfinite(true)
    nonfinite_predictions = int(np.size(pred) - np.isfinite(pred).sum())
    target_nonfinite = int(np.size(true) - np.isfinite(true).sum())
    finite_pred = pred[finite_mask]
    finite_true = true[finite_mask]
    target_negative = int(np.sum(finite_true < 0)) if volatility else 0
    if target_negative:
        raise CandidateObjectiveError("rv_target_negative_count must be zero")
    if not len(finite_pred):
        return {
            "classification": "auxiliary_failed_nonfinite",
            "finite_prediction_count": 0,
            "nonfinite_prediction_count": nonfinite_predictions,
            "target_nonfinite_count": target_nonfinite,
        }
    error = finite_pred - finite_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    baseline_error = float(train_target_mean) - finite_true
    baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
    ratio = None if baseline_rmse <= 1e-15 else float(rmse / baseline_rmse)
    correlation = _pearson(finite_true, finite_pred)
    prediction_std = float(np.std(finite_pred, ddof=0))
    negative_count = int(np.sum(finite_pred < 0)) if volatility else 0
    if nonfinite_predictions or target_nonfinite or not deterministic_repeat_passed:
        health = "auxiliary_failed_nonfinite"
    elif volatility and negative_count:
        health = "auxiliary_failed_negative_rv"
    elif prediction_std <= 1e-12:
        health = "auxiliary_failed_constant_output"
    elif correlation is None or correlation <= 0 or (ratio is not None and ratio >= 1.0):
        health = "auxiliary_warning_low_skill"
    else:
        health = "auxiliary_healthy"
    result: dict[str, Any] = {
        "classification": health,
        "mae": mae,
        "rmse": rmse,
        "normalized_mae": float(mae / valid_scale),
        "normalized_rmse": float(rmse / valid_scale),
        "pearson_correlation": correlation,
        "prediction_mean": float(np.mean(finite_pred)),
        "prediction_std": prediction_std,
        "target_mean": float(np.mean(finite_true)),
        "target_std": float(np.std(finite_true, ddof=0)),
        "finite_prediction_count": int(len(finite_pred)),
        "nonfinite_prediction_count": nonfinite_predictions,
        "target_nonfinite_count": target_nonfinite,
        "candidate_rmse": rmse,
        "baseline": "constant_training_target_mean",
        "baseline_value": float(train_target_mean),
        "baseline_rmse": baseline_rmse,
        "candidate_vs_baseline_rmse_ratio": ratio,
        "target_scale": valid_scale,
        "deterministic_repeat_passed": bool(deterministic_repeat_passed),
    }
    if volatility:
        result.update({
            "negative_prediction_count": negative_count,
            "negative_prediction_rate": float(negative_count / len(finite_pred)),
            "minimum_prediction": float(np.min(finite_pred)),
            "rv_target_negative_count": 0,
        })
    else:
        result.update({
            "information_coefficient": _pearson(
                _average_ranks(finite_true), _average_ranks(finite_pred)
            ),
            "sign_accuracy": float(np.mean(np.sign(finite_pred) == np.sign(finite_true))),
        })
    return result


def objective_metrics(
    *,
    ret_prediction: Sequence[float] | np.ndarray,
    ret_target: Sequence[float] | np.ndarray,
    rv_prediction: Sequence[float] | np.ndarray,
    rv_target: Sequence[float] | np.ndarray,
    ret_scale: float,
    rv_scale: float,
    ret_train_target_mean: float,
    rv_train_target_mean: float,
    deterministic_repeat_passed: bool = True,
) -> dict[str, Any]:
    ret = _head_metrics(
        ret_prediction, ret_target, scale=ret_scale,
        train_target_mean=ret_train_target_mean, volatility=False,
        deterministic_repeat_passed=deterministic_repeat_passed,
    )
    rv = _head_metrics(
        rv_prediction, rv_target, scale=rv_scale,
        train_target_mean=rv_train_target_mean, volatility=True,
        deterministic_repeat_passed=deterministic_repeat_passed,
    )
    hard_failures = {
        "auxiliary_failed_negative_rv", "auxiliary_failed_nonfinite",
        "auxiliary_failed_constant_output", "auxiliary_unverified",
    }
    return {
        "ret_reg": ret,
        "rv_reg": rv,
        "auxiliary_head_gate_passed": (
            ret.get("classification") not in hard_failures
            and rv.get("classification") not in hard_failures
        ),
        "post_hoc_rv_clipping_applied": False,
        "outputs_remain_in_raw_target_units": True,
    }
