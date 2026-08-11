"""Training-only Phase 24.2 multitask loss-balance calibration and freeze.

The mathematical helpers are usable with synthetic batches.  The CLI freeze
path accepts only a verified frozen Phase 24 dataset, samples training
sequences only, and refuses any experiment with recorded non-training access.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_objective import resolved_candidate_loss
from tools.model_objective_contract import validate_objective_report
from tools.model_training_environment import file_digest, json_digest


BALANCE_POLICY = BASE_DIR / "research" / "model_candidate_loss_balance_policy.json"
OBJECTIVE_REPORT = BASE_DIR / "reports" / "model_objective_contract.json"
DEFAULT_BALANCE_REPORT = BASE_DIR / "reports" / "model_candidate_loss_balance.json"
DEFAULT_BALANCE_FREEZE = BASE_DIR / "reports" / "model_candidate_loss_balance_freeze.json"
PARENT_OBJECTIVE_DIGEST = "fe7ea1f3cc6aaa7b20f91cf0e96fc181e3b343bc144fa935a3e4afb700a58700"
FORMULATIONS = (
    "normalized_mse_fixed",
    "normalized_huber_fixed",
    "normalized_huber_training_balanced",
)
ARCHITECTURES = ("lstm", "tcn", "tx")
HEAD_PREFIXES = {
    "lstm": {
        "classification": ("head_cls.",), "return": ("head_ret.",), "rv": ("head_rv.",),
    },
    "tcn": {
        "classification": ("head_ret_cls.",), "return": ("head_ret_reg.",), "rv": ("head_rv_reg.",),
    },
    "tx": {
        "classification": ("head_ret_cls.",), "return": ("head_ret_reg.",), "rv": ("head_rv_reg.",),
    },
}
ACCESS_LEDGER_NAMES = (
    "model_candidate_validation_access.json",
    "model_candidate_internal_test_access.json",
    "model_candidate_legacy_repair_access.json",
    "model_candidate_confirmation_access.json",
)

POLICY_TEMPLATE: dict[str, Any] = {
    "schema_version": 1,
    "parent_objective_contract_digest": PARENT_OBJECTIVE_DIGEST,
    "formulations": list(FORMULATIONS),
    "classification_weight": 1.0,
    "fixed_return_weight": 0.5,
    "fixed_rv_weight": 0.5,
    "huber_beta": 1.0,
    "balance_calibration_seed": 24201,
    "balance_calibration_batches": 16,
    "target_auxiliary_to_classification_gradient_ratio": 0.25,
    "minimum_allowed_auxiliary_ratio": 0.10,
    "maximum_allowed_auxiliary_ratio": 0.50,
    "maximum_auxiliary_p90_ratio": 1.0,
    "maximum_combined_auxiliary_median_ratio": 0.75,
    "minimum_fixed_auxiliary_weight": 0.000001,
    "maximum_fixed_auxiliary_weight": 1.0,
    "require_positive_classification_projection": True,
    "require_all_heads_nonzero_gradient": True,
    "require_all_gradients_finite": True,
    "require_deterministic_balance_calibration": True,
    "balance_source": "training_sequences_only",
    "allow_validation_for_balance": False,
    "allow_internal_test_for_balance": False,
    "allow_legacy_repair_for_balance": False,
    "allow_confirmation_for_balance": False,
    "weights_fixed_after_balance": True,
    "candidate_only": True,
    "promotion_allowed": False,
    "live_activation_allowed": False,
}


class LossBalanceError(ValueError):
    """A balance policy, gradient, sample, ordering, or freeze gate failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _expected_candidate_rv_output_contract() -> dict[str, Any]:
    # Lazy import avoids the candidate-training/loss-balance module import cycle.
    from tools.model_candidate_train import candidate_rv_output_contract

    return candidate_rv_output_contract()


def _validate_rv_output_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    observed = contract.get("rv_output_contract_digest")
    payload = {key: item for key, item in contract.items() if key != "rv_output_contract_digest"}
    if observed != _digest(payload):
        raise LossBalanceError("RV-output contract digest mismatch")
    return contract


def validate_balance_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if set(policy) != set(POLICY_TEMPLATE):
        raise LossBalanceError("loss-balance policy fields are not exact")
    for name, expected in POLICY_TEMPLATE.items():
        value = policy[name]
        if type(value) is not type(expected) or value != expected:
            raise LossBalanceError(f"loss-balance policy mismatch: {name}")
    return copy.deepcopy(dict(policy))


def load_balance_policy(path: Path | str = BALANCE_POLICY) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_balance_policy(value)


def balance_policy_digest(policy: Mapping[str, Any] | None = None) -> str:
    return _digest(validate_balance_policy(policy or load_balance_policy()))


def validate_parent_objective(
    path: Path | str = OBJECTIVE_REPORT, *, expected_digest: str = PARENT_OBJECTIVE_DIGEST,
) -> dict[str, Any]:
    try:
        value = validate_objective_report(path, expected_digest=expected_digest)
    except Exception as exc:
        raise LossBalanceError(str(exc)) from exc
    if value.get("overall_decision", {}).get("verdict") != (
        "candidate_objective_contract_resolved_multitask_training_required"
    ):
        raise LossBalanceError("Phase 24.1 objective verdict does not permit balance calibration")
    return value


def formulation_descriptor(
    formulation_id: str,
    *,
    return_weight: float | None = None,
    rv_weight: float | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p = validate_balance_policy(policy or load_balance_policy())
    if formulation_id not in FORMULATIONS:
        raise LossBalanceError("unknown formulation")
    if formulation_id != "normalized_huber_training_balanced" and (
        return_weight is not None or rv_weight is not None
    ):
        raise LossBalanceError("fixed formulations prohibit overridden task weights")
    descriptor = {
        "formulation_id": formulation_id,
        "classification_loss": "weighted_cross_entropy",
        "regression_loss": (
            "train_scale_normalized_mse" if formulation_id == "normalized_mse_fixed"
            else "train_scale_normalized_smooth_l1"
        ),
        "classification_weight": float(p["classification_weight"]),
        "return_weight": float(
            p["fixed_return_weight"] if return_weight is None else return_weight
        ),
        "rv_weight": float(p["fixed_rv_weight"] if rv_weight is None else rv_weight),
        "huber_beta": None if formulation_id == "normalized_mse_fixed" else float(p["huber_beta"]),
        "outputs_remain_in_raw_runtime_units": True,
        "weights_fixed_for_entire_training_run": True,
        "balance_source": p["balance_source"],
    }
    for name in ("classification_weight", "return_weight", "rv_weight"):
        if not math.isfinite(descriptor[name]) or descriptor[name] < 0:
            raise LossBalanceError("task weights must be finite and non-negative")
    descriptor["formulation_digest"] = _digest(descriptor)
    return descriptor


def parameter_groups(model: Any, kind: str) -> dict[str, list[tuple[str, Any]]]:
    if kind not in HEAD_PREFIXES:
        raise LossBalanceError("unknown model kind")
    result: dict[str, list[tuple[str, Any]]] = {
        "shared_backbone": [], "classification_head": [], "return_head": [], "rv_head": [],
    }
    prefixes = HEAD_PREFIXES[kind]
    for name, parameter in model.named_parameters():
        matched = [task for task, values in prefixes.items() if name.startswith(values)]
        if len(matched) > 1:
            raise LossBalanceError("overlapping head parameter groups")
        key = f"{matched[0]}_head" if matched else "shared_backbone"
        result[key].append((name, parameter))
    if any(not values for values in result.values()):
        raise LossBalanceError("incomplete model parameter groups")
    names = [name for values in result.values() for name, _ in values]
    if len(names) != len(set(names)) or len(names) != len(list(model.named_parameters())):
        raise LossBalanceError("parameter groups are not a complete disjoint partition")
    return result


def _flat_gradients(parameters: Sequence[tuple[str, Any]]) -> Any:
    import torch
    pieces = [
        (parameter.grad.detach().reshape(-1).double() if parameter.grad is not None
         else torch.zeros(parameter.numel(), dtype=torch.float64))
        for _, parameter in parameters
    ]
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)


def _norm(vector: Any) -> float:
    return float(vector.pow(2).sum().sqrt().item())


def cosine_similarity(left: Any, right: Any) -> float | None:
    denominator = _norm(left) * _norm(right)
    if denominator <= 0:
        return None
    return float((left @ right).item() / denominator)


def classification_projection(classification: Any, total: Any) -> dict[str, Any]:
    dot = float((classification @ total).item())
    return {
        "dot": dot,
        "positive": bool(dot > 0),
        "cosine": cosine_similarity(total, classification),
    }


def measure_task_gradients(
    model: Any,
    kind: str,
    batch: Mapping[str, Any],
    *,
    formulation_id: str,
    class_weights: Any,
    ret_scale: float,
    rv_scale: float,
    seed: int = 24201,
) -> dict[str, Any]:
    """Measure component gradients without updating identical starting parameters."""
    import torch
    descriptor = formulation_descriptor(
        "normalized_mse_fixed" if formulation_id == "normalized_mse_fixed"
        else "normalized_huber_fixed"
    )
    groups = parameter_groups(model, kind)
    all_parameters = list(model.named_parameters())
    before = {name: value.detach().clone() for name, value in all_parameters}
    task_key = {
        "classification": "classification_loss",
        "return": "return_regression_loss",
        "rv": "rv_regression_loss",
    }
    task_head = {
        "classification": "classification_head", "return": "return_head", "rv": "rv_head",
    }
    measurements: dict[str, Any] = {}
    for task in ("classification", "return", "rv"):
        torch.manual_seed(int(seed))
        model.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        losses = resolved_candidate_loss(
            outputs,
            {key: batch[key] for key in ("y_ret_cls", "y_ret_reg", "y_rv_reg")},
            ret_scale=ret_scale, rv_scale=rv_scale,
            formulation=descriptor, class_weights=class_weights,
        )
        losses[task_key[task]].backward()
        shared = _flat_gradients(groups["shared_backbone"])
        head = _flat_gradients(groups[task_head[task]])
        full = _flat_gradients(all_parameters)
        finite = bool(torch.isfinite(shared).all().item() and torch.isfinite(head).all().item())
        measurements[task] = {
            "loss": float(losses[task_key[task]].detach().item()),
            "shared_gradient_l2": _norm(shared),
            "head_gradient_l2": _norm(head),
            "finite": finite,
            "nonzero": bool(_norm(shared) > 0 and _norm(head) > 0),
            "shared_vector": shared,
            "full_vector": full,
        }
    if any(not parameter.detach().equal(before[name]) for name, parameter in all_parameters):
        raise LossBalanceError("component gradient measurement changed model parameters")
    shared = {task: measurements[task]["shared_vector"] for task in measurements}
    measurements["cosines"] = {
        "classification_vs_return": cosine_similarity(shared["classification"], shared["return"]),
        "classification_vs_rv": cosine_similarity(shared["classification"], shared["rv"]),
        "return_vs_rv": cosine_similarity(shared["return"], shared["rv"]),
    }
    measurements["starting_parameter_digest"] = _digest({
        name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name, value in before.items()
    })
    return measurements


def deterministic_statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise LossBalanceError("finite nonempty gradient statistics required")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "maximum": float(np.max(array)),
    }


def derive_balanced_weights(
    classification_median: float,
    return_median: float,
    rv_median: float,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p = validate_balance_policy(policy or load_balance_policy())
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in (
        classification_median, return_median, rv_median,
    )):
        raise LossBalanceError("positive finite median gradients required")
    target = float(p["target_auxiliary_to_classification_gradient_ratio"])
    lower, upper = float(p["minimum_fixed_auxiliary_weight"]), float(p["maximum_fixed_auxiliary_weight"])
    result: dict[str, Any] = {}
    for task, gradient in (("return", return_median), ("rv", rv_median)):
        raw = target * float(classification_median) / float(gradient)
        final = min(max(raw, lower), upper)
        result[task] = {
            "raw_weight": float(raw), "final_weight": float(final),
            "lower_bound_applied": bool(raw < lower), "upper_bound_applied": bool(raw > upper),
        }
    result["iterative_search_performed"] = False
    return result


def _conflict(value: float | None) -> str:
    if value is None:
        return "unverified"
    if value >= 0:
        return "aligned"
    return "weak_conflict" if value >= -0.5 else "strong_conflict"


def aggregate_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    descriptor: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    clip_norm: float = 1.0,
) -> dict[str, Any]:
    import torch
    p = validate_balance_policy(policy or load_balance_policy())
    if not measurements:
        raise LossBalanceError("calibration measurements required")
    weights = {
        "classification": float(descriptor["classification_weight"]),
        "return": float(descriptor["return_weight"]),
        "rv": float(descriptor["rv_weight"]),
    }
    gradients = {
        task: deterministic_statistics(item[task]["shared_gradient_l2"] for item in measurements)
        for task in ("classification", "return", "rv")
    }
    cls_median = gradients["classification"]["median"] * weights["classification"]
    ratios = {
        "return_median_to_classification": gradients["return"]["median"] * weights["return"] / cls_median,
        "rv_median_to_classification": gradients["rv"]["median"] * weights["rv"] / cls_median,
        "combined_auxiliary_median_to_classification": (
            gradients["return"]["median"] * weights["return"]
            + gradients["rv"]["median"] * weights["rv"]
        ) / cls_median,
        "return_p90_to_classification_median": gradients["return"]["p90"] * weights["return"] / cls_median,
        "rv_p90_to_classification_median": gradients["rv"]["p90"] * weights["rv"] / cls_median,
    }
    projections, preclip = [], []
    for item in measurements:
        g_cls = item["classification"]["shared_vector"] * weights["classification"]
        g_ret = item["return"]["shared_vector"] * weights["return"]
        g_rv = item["rv"]["shared_vector"] * weights["rv"]
        projections.append(classification_projection(g_cls, g_cls + g_ret + g_rv))
        full = (
            item["classification"]["full_vector"] * weights["classification"]
            + item["return"]["full_vector"] * weights["return"]
            + item["rv"]["full_vector"] * weights["rv"]
        )
        preclip.append(_norm(full))
    cosine_stats = {}
    for key in ("classification_vs_return", "classification_vs_rv", "return_vs_rv"):
        values = [item["cosines"][key] for item in measurements]
        if any(value is None for value in values):
            cosine_stats[key] = {"median": None, "minimum": None, "p10": None, "classification": "unverified"}
        else:
            array = np.asarray(values, dtype=np.float64)
            median = float(np.median(array))
            cosine_stats[key] = {
                "median": median, "minimum": float(np.min(array)),
                "p10": float(np.quantile(array, 0.10, method="linear")),
                "classification": _conflict(median),
            }
    clipping_stats = deterministic_statistics(preclip)
    clipping_rate = float(np.mean(np.asarray(preclip) > float(clip_norm)))
    clipping = {
        "threshold": float(clip_norm), "pre_clip_norm": clipping_stats,
        "expected_clip_activation_rate": clipping_rate,
        "warning": "clipping_almost_always_active" if clipping_rate >= 0.95 else None,
    }
    reasons: list[str] = []
    if any(not item[task]["finite"] for item in measurements for task in ("classification", "return", "rv")):
        reasons.append("nonfinite_task_gradient")
    if any(not item[task]["nonzero"] for item in measurements for task in ("classification", "return", "rv")):
        reasons.append("zero_task_or_head_gradient")
    low, high = float(p["minimum_allowed_auxiliary_ratio"]), float(p["maximum_allowed_auxiliary_ratio"])
    for key in ("return_median_to_classification", "rv_median_to_classification"):
        if not low <= ratios[key] <= high:
            reasons.append(f"{key}_outside_gate")
    for key in ("return_p90_to_classification_median", "rv_p90_to_classification_median"):
        if ratios[key] > float(p["maximum_auxiliary_p90_ratio"]):
            reasons.append(f"{key}_above_gate")
    if ratios["combined_auxiliary_median_to_classification"] > float(
        p["maximum_combined_auxiliary_median_ratio"]
    ):
        reasons.append("combined_auxiliary_median_ratio_above_gate")
    if any(not item["positive"] for item in projections):
        reasons.append("classification_projection_not_positive")
    return {
        "formulation": dict(descriptor),
        "batch_count": len(measurements),
        "unweighted_shared_gradient_statistics": gradients,
        "weighted_gradient_ratios": ratios,
        "pairwise_gradient_cosines": cosine_stats,
        "classification_projection": {
            "all_positive": all(item["positive"] for item in projections),
            "minimum_dot": min(item["dot"] for item in projections),
            "median_cosine": float(np.median([item["cosine"] for item in projections])),
        },
        "gradient_clipping": clipping,
        "balance_status": "balance_safe" if not reasons else "balance_failed",
        "passed": not reasons,
        "failure_reasons": reasons,
        "gradient_surgery_applied": False,
    }


def evaluate_formulations(
    mse_measurements: Sequence[Mapping[str, Any]],
    huber_measurements: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    p = validate_balance_policy(policy or load_balance_policy())
    a = aggregate_measurements(
        mse_measurements, descriptor=formulation_descriptor("normalized_mse_fixed", policy=p), policy=p,
    )
    b = aggregate_measurements(
        huber_measurements, descriptor=formulation_descriptor("normalized_huber_fixed", policy=p), policy=p,
    )
    medians = b["unweighted_shared_gradient_statistics"]
    weights = derive_balanced_weights(
        medians["classification"]["median"], medians["return"]["median"], medians["rv"]["median"], policy=p,
    )
    c_descriptor = formulation_descriptor(
        "normalized_huber_training_balanced",
        return_weight=weights["return"]["final_weight"],
        rv_weight=weights["rv"]["final_weight"], policy=p,
    )
    c = aggregate_measurements(huber_measurements, descriptor=c_descriptor, policy=p)
    c["weight_derivation"] = weights
    formulations = {
        "normalized_mse_fixed": a,
        "normalized_huber_fixed": b,
        "normalized_huber_training_balanced": c,
    }
    selected = next((name for name in FORMULATIONS if formulations[name]["passed"]), None)
    return {
        "formulations": formulations,
        "selected_formulation": selected,
        "selected_descriptor": None if selected is None else formulations[selected]["formulation"],
        "balance_status": "resolved" if selected is not None else "loss_balance_unresolved",
        "selection_rule": "minimal_change_first_safe_A_then_B_then_C",
        "validation_metrics_consulted": False,
        "internal_test_metrics_consulted": False,
        "legacy_repair_metrics_consulted": False,
        "confirmation_metrics_consulted": False,
    }


def select_calibration_indices(
    datasets_by_symbol: Mapping[str, Mapping[str, Any]],
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministically sample endpoint identities from training datasets only."""
    if batches <= 0 or batch_size <= 0:
        raise LossBalanceError("positive calibration batches and batch size required")
    records: list[tuple[str, int, int, int]] = []
    for symbol in sorted(datasets_by_symbol):
        dataset = datasets_by_symbol[symbol]["train"]
        for local_index in range(len(dataset)):
            row = dataset[local_index]
            endpoint = int(dataset.endpoints[local_index]) if hasattr(dataset, "endpoints") else local_index
            records.append((symbol, local_index, endpoint, int(row["y_ret_cls"])))
    if {item[0] for item in records} != {"BTCUSDT", "ETHUSDT"} or {item[3] for item in records} != {0, 1}:
        raise LossBalanceError("loss_balance_calibration_sample_invalid")
    needed = int(batches) * int(batch_size)
    if len(records) < needed:
        raise LossBalanceError("loss_balance_calibration_sample_invalid")
    rng = random.Random(int(seed))
    order = list(range(len(records)))
    rng.shuffle(order)
    mandatory: list[int] = []
    for predicate in (
        lambda r: r[0] == "BTCUSDT", lambda r: r[0] == "ETHUSDT",
        lambda r: r[3] == 0, lambda r: r[3] == 1,
    ):
        index = next(index for index in order if predicate(records[index]) and index not in mandatory)
        mandatory.append(index)
    selected_indices = mandatory + [index for index in order if index not in mandatory]
    selected = [records[index] for index in selected_indices[:needed]]
    identities = [f"{symbol}:{endpoint}:{label}" for symbol, _, endpoint, label in selected]
    return {
        "records": selected,
        "endpoint_identities": identities,
        "endpoint_digest": _digest(identities),
        "symbols": sorted({row[0] for row in selected}),
        "classes": sorted({row[3] for row in selected}),
        "batch_count": int(batches), "batch_size": int(batch_size), "seed": int(seed),
        "source": "training_sequences_only",
    }


def _collate_records(datasets: Mapping[str, Mapping[str, Any]], records: Sequence[tuple[str, int, int, int]]):
    import torch
    rows = [datasets[symbol]["train"][local_index] for symbol, local_index, _, _ in records]
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def nontraining_access_records(
    *, reports_root: Path | str = BASE_DIR / "reports", dataset_digest: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(reports_root)
    found: list[dict[str, Any]] = []
    for name in ACCESS_LEDGER_NAMES:
        path = root / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        entries = value if isinstance(value, list) else value.get("accesses", value.get("candidates", []))
        if isinstance(entries, Mapping):
            entries = list(entries.values())
        if not isinstance(entries, list):
            entries = [value]
        for item in entries:
            if dataset_digest is None or not isinstance(item, Mapping) or item.get("dataset_digest") in (None, dataset_digest):
                found.append({"ledger": name, "record": item})
    seed_root = root / "model_candidate_seed_runs"
    if seed_root.is_dir():
        for path in seed_root.glob("*/selected_freeze.json"):
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            observed = value.get("candidate_identity", {}).get("dataset_digest")
            if dataset_digest is None or observed in (None, dataset_digest):
                found.append({"ledger": path.relative_to(root).as_posix(),
                              "record": {"validation_accessed": True,
                                         "internal_test_accessed": value.get("internal_test_accessed", False),
                                         "dataset_digest": observed}})
    summary_path = root / "model_candidate_training_summary.json"
    if summary_path.is_file():
        value = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        observed = value.get("dataset", {}).get("dataset_digest")
        if dataset_digest is None or observed in (None, dataset_digest):
            found.append({"ledger": summary_path.name,
                          "record": {"validation_and_internal_test_accessed": True,
                                     "dataset_digest": observed}})
    return found


def assert_balance_freeze_ordering(
    *, reports_root: Path | str = BASE_DIR / "reports", dataset_digest: str | None = None,
) -> None:
    if nontraining_access_records(reports_root=reports_root, dataset_digest=dataset_digest):
        raise LossBalanceError("balance_freeze_contaminated")


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise LossBalanceError("loss-balance freeze overwrite prohibited")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def validate_balance_freeze(
    path: Path | str = DEFAULT_BALANCE_FREEZE,
    *,
    dataset_manifest: Mapping[str, Any] | None = None,
    expected_balance_digest: str | None = None,
    expected_rv_output_contract_digest: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise LossBalanceError("model_candidate_loss_balance_freeze.json required")
    value = json.loads(target.read_text(encoding="utf-8-sig"))
    if value.get("freeze_status") != "frozen" or value.get("parent_objective_contract_digest") != PARENT_OBJECTIVE_DIGEST:
        raise LossBalanceError("invalid loss-balance freeze")
    payload = {key: item for key, item in value.items() if key not in {"freeze_timestamp", "balance_contract_digest"}}
    observed = _digest(payload)
    if observed != value.get("balance_contract_digest"):
        raise LossBalanceError("loss-balance freeze digest mismatch")
    if expected_balance_digest is not None and observed != expected_balance_digest:
        raise LossBalanceError("unexpected loss-balance contract digest")
    frozen_rv_digest = value.get("rv_output_contract_digest")
    frozen_rv_contract = value.get("rv_output_contract")
    if frozen_rv_digest is not None or frozen_rv_contract is not None:
        if not isinstance(frozen_rv_contract, Mapping):
            raise LossBalanceError("invalid frozen RV-output contract")
        validated_rv_contract = _validate_rv_output_contract(frozen_rv_contract)
        if validated_rv_contract["rv_output_contract_digest"] != frozen_rv_digest:
            raise LossBalanceError("frozen RV-output contract digest mismatch")
    if (
        expected_rv_output_contract_digest is not None
        and frozen_rv_digest != expected_rv_output_contract_digest
    ):
        raise LossBalanceError("loss-balance freeze RV-output contract mismatch")
    if value.get("balance_policy_digest") != balance_policy_digest():
        raise LossBalanceError("loss-balance policy digest mismatch")
    if set(value.get("architectures", {})) != set(ARCHITECTURES):
        raise LossBalanceError("loss-balance freeze lacks architecture contracts")
    for kind, record in value["architectures"].items():
        if record.get("balance_status") != "balance_safe":
            raise LossBalanceError(f"unsafe frozen balance formulation: {kind}")
        descriptor = record.get("selected_descriptor", {})
        if descriptor.get("formulation_id") not in FORMULATIONS:
            raise LossBalanceError("invalid frozen formulation descriptor")
    if dataset_manifest is not None:
        for field, manifest_field in (("dataset_digest", "dataset_digest"), ("split_digest", "split_digest")):
            if value.get(field) != dataset_manifest.get(manifest_field):
                raise LossBalanceError(f"loss-balance freeze {field} mismatch")
        if value.get("scaler_digest") != dataset_manifest.get("scaler", {}).get("sha256"):
            raise LossBalanceError("loss-balance freeze scaler digest mismatch")
        scales = dataset_manifest.get("target_scales", {})
        if value.get("target_scale_digest") != scales.get("target_scale_digest"):
            raise LossBalanceError("loss-balance freeze target-scale digest mismatch")
    return value


def run_real_calibration(
    dataset: Path | str,
    *,
    calibration_batches: int | None = None,
    batch_size: int = 256,
    reports_root: Path | str = BASE_DIR / "reports",
) -> dict[str, Any]:
    """Calibrate all production architectures from a verified training split."""
    import torch
    from tools.model_candidate_train import (
        _class_weights, _concat, load_sequence_datasets, make_candidate_model,
        candidate_rv_output_contract, training_sequence_target_scales,
    )
    from tools.model_training_dataset import verify_dataset

    parent = validate_parent_objective()
    policy = load_balance_policy()
    rv_output_contract = candidate_rv_output_contract()
    manifest = verify_dataset(dataset)
    assert_balance_freeze_ordering(reports_root=reports_root, dataset_digest=manifest["dataset_digest"])
    datasets, loaded_manifest = load_sequence_datasets(dataset)
    scales = training_sequence_target_scales(datasets)
    frozen = loaded_manifest.get("target_scales")
    if not isinstance(frozen, Mapping) or frozen.get("target_scale_digest") != scales["target_scale_digest"]:
        raise LossBalanceError("verified frozen training target scales required")
    count = int(calibration_batches or policy["balance_calibration_batches"])
    if count < int(policy["balance_calibration_batches"]):
        raise LossBalanceError("CLI may not weaken balance calibration batch count")
    sample = select_calibration_indices(
        datasets, batches=count, batch_size=batch_size, seed=policy["balance_calibration_seed"],
    )
    class_weights, _ = _class_weights(_concat(datasets, "train"))
    architecture_reports = {}
    records = sample["records"]
    for kind in ARCHITECTURES:
        mse, huber = [], []
        for batch_index in range(count):
            start, stop = batch_index * batch_size, (batch_index + 1) * batch_size
            batch = _collate_records(datasets, records[start:stop])
            torch.manual_seed(policy["balance_calibration_seed"])
            model = make_candidate_model(kind).cpu().train()
            state = copy.deepcopy(model.state_dict())
            mse.append(measure_task_gradients(
                model, kind, batch, formulation_id="normalized_mse_fixed",
                class_weights=class_weights, ret_scale=scales["ret_target_scale"],
                rv_scale=scales["rv_target_scale"], seed=policy["balance_calibration_seed"] + batch_index,
            ))
            model.load_state_dict(state)
            huber.append(measure_task_gradients(
                model, kind, batch, formulation_id="normalized_huber_fixed",
                class_weights=class_weights, ret_scale=scales["ret_target_scale"],
                rv_scale=scales["rv_target_scale"], seed=policy["balance_calibration_seed"] + batch_index,
            ))
        architecture_reports[kind] = evaluate_formulations(mse, huber, policy=policy)
    serial = _serializable(architecture_reports)
    safe = all(record["balance_status"] == "resolved" for record in architecture_reports.values())
    report = {
        "schema_version": 1,
        "dataset": {"verified": True, "dataset_digest": manifest["dataset_digest"], "split_digest": manifest["split_digest"],
                    "scaler_digest": manifest["scaler"]["sha256"], "target_scales": scales},
        "parent_objective": {"objective_contract_digest": parent["objective_contract_digest"]},
        "balance_policy": {"digest": balance_policy_digest(policy), "policy": policy},
        "calibration_sample": {key: value for key, value in sample.items() if key != "records"},
        "rv_output_contract": rv_output_contract,
        "architectures": serial,
        "overall_decision": {"status": "all_architectures_resolved" if safe else "candidate_loss_balance_unresolved"},
        "nontraining_evidence_consulted": False,
    }
    report["balance_digest"] = _digest(report)
    return report


def _serializable(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return None
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items() if key not in {"shared_vector", "full_vector"}}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def freeze_balance_contract(report: Mapping[str, Any], path: Path | str) -> dict[str, Any]:
    if report.get("dataset", {}).get("verified") is not True:
        raise LossBalanceError("verified frozen dataset required for loss-balance freeze")
    if report.get("calibration_sample", {}).get("source") != "training_sequences_only":
        raise LossBalanceError("loss-balance freeze requires training sequences only")
    rv_output_contract = report.get("rv_output_contract")
    if not isinstance(rv_output_contract, Mapping):
        raise LossBalanceError("new candidate RV-output contract required")
    validated_rv_contract = _validate_rv_output_contract(rv_output_contract)
    if validated_rv_contract != _expected_candidate_rv_output_contract():
        raise LossBalanceError("new candidate RV-output contract mismatch")
    architectures = report.get("architectures", {})
    if set(architectures) != set(ARCHITECTURES) or any(
        item.get("balance_status") != "resolved" for item in architectures.values()
    ):
        raise LossBalanceError("all architectures must resolve before balance freeze")
    selected = {
        kind: {
            "balance_status": "balance_safe",
            "selected_formulation": item["selected_formulation"],
            "selected_descriptor": item["selected_descriptor"],
            "balance_statistics": item["formulations"][item["selected_formulation"]],
        }
        for kind, item in architectures.items()
    }
    dataset = report["dataset"]
    payload = {
        "schema_version": 1,
        "freeze_status": "frozen",
        "dataset_digest": dataset["dataset_digest"], "split_digest": dataset["split_digest"],
        "scaler_digest": dataset["scaler_digest"],
        "target_scale_digest": dataset["target_scales"]["target_scale_digest"],
        "parent_objective_contract_digest": PARENT_OBJECTIVE_DIGEST,
        "balance_policy_digest": report["balance_policy"]["digest"],
        "calibration_sample_digest": report["calibration_sample"]["endpoint_digest"],
        "rv_output_contract": validated_rv_contract,
        "rv_output_contract_digest": validated_rv_contract["rv_output_contract_digest"],
        "architectures": selected,
        "heterogeneous_architecture_objectives": len({x["selected_formulation"] for x in selected.values()}) > 1,
        "gradient_statistics_digest": _digest(selected),
        "weights_fixed_after_balance": True,
        "nontraining_evidence_consulted": False,
    }
    payload["balance_contract_digest"] = _digest(payload)
    payload["freeze_timestamp"] = _utc_now()
    _atomic_create(Path(path), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--json-out", default=str(DEFAULT_BALANCE_REPORT))
    parser.add_argument("--freeze-out", default=str(DEFAULT_BALANCE_FREEZE))
    parser.add_argument("--calibration-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if Path(args.freeze_out).exists():
            raise LossBalanceError("loss-balance freeze overwrite prohibited")
        report = run_real_calibration(
            args.dataset, calibration_batches=args.calibration_batches, batch_size=args.batch_size,
        )
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        freeze = freeze_balance_contract(report, args.freeze_out)
        print(json.dumps({"status": report["overall_decision"]["status"],
                          "balance_contract_digest": freeze["balance_contract_digest"]}, indent=2))
        return 0
    except (LossBalanceError, ValueError, OSError) as exc:
        print(json.dumps({"status": "loss_balance_calibration_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
