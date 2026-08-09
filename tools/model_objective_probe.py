"""Synthetic deterministic gradient/activity probe for all Phase 24 architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_objective import candidate_multitask_loss


class ObjectiveProbeError(ValueError):
    """A synthetic architecture, gradient, parity, or determinism check failed."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _model(kind: str):
    from ml_dl.dl_models import TemporalConvNet, TinyLSTM, TinyTransformer
    if kind == "lstm":
        return TinyLSTM(in_dim=27, hidden=16, layers=2, dropout=0.0), {
            "in_dim": 27, "hidden": 16, "layers": 2, "dropout": 0.0,
        }
    if kind == "tcn":
        return TemporalConvNet(in_dim=27, hid=16, levels=2, kernel=3, dropout=0.0), {
            "in_dim": 27, "hid": 16, "levels": 2, "kernel": 3, "dropout": 0.0,
        }
    if kind == "tx":
        return TinyTransformer(in_dim=27, d_model=16, nhead=4, nlayers=1, dropout=0.0), {
            "in_dim": 27, "d_model": 16, "nhead": 4, "nlayers": 1, "dropout": 0.0,
        }
    raise ObjectiveProbeError(f"unknown model kind: {kind}")


def _group(kind: str, name: str) -> str:
    if kind == "lstm":
        if name.startswith("head_cls."):
            return "classification_head"
        if name.startswith("head_ret."):
            return "return_head"
        if name.startswith("head_rv."):
            return "rv_head"
        return "shared_backbone"
    if name.startswith("head_ret_cls."):
        return "classification_head"
    if name.startswith("head_ret_reg."):
        return "return_head"
    if name.startswith("head_rv_reg."):
        return "rv_head"
    return "shared_backbone"


def _gradient_norms(model: Any, kind: str) -> dict[str, float]:
    totals = {name: 0.0 for name in ("shared_backbone", "classification_head", "return_head", "rv_head")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        totals[_group(kind, name)] += float(parameter.grad.detach().double().pow(2).sum().item())
    return {name: math.sqrt(value) for name, value in totals.items()}


def _changed_groups(before: Mapping[str, Any], model: Any, kind: str) -> dict[str, bool]:
    changed = {name: False for name in ("shared_backbone", "classification_head", "return_head", "rv_head")}
    for name, parameter in model.named_parameters():
        if not parameter.detach().equal(before[name]):
            changed[_group(kind, name)] = True
    return changed


def _synthetic_batch():
    import torch
    x = torch.linspace(-1.5, 1.5, steps=6 * 64 * 27, dtype=torch.float32).reshape(6, 64, 27)
    y_cls = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    y_ret = torch.tensor([-0.03, 0.01, -0.01, 0.04, -0.02, 0.02], dtype=torch.float32)
    y_rv = torch.tensor([0.010, 0.014, 0.018, 0.022, 0.026, 0.030], dtype=torch.float32)
    return x, {"y_ret_cls": y_cls, "y_ret_reg": y_ret, "y_rv_reg": y_rv}


def classification_parity_probe(kind: str, *, seed: int = 24101) -> dict[str, Any]:
    import torch
    _seed(seed)
    model_a, _ = _model(kind)
    model_b, _ = _model(kind)
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval()
    model_b.eval()
    x, targets = _synthetic_batch()
    weights = torch.tensor([1.2, 0.8], dtype=torch.float32)
    out_a = model_a(x)
    extended = candidate_multitask_loss(
        out_a, targets, ret_scale=0.02, rv_scale=0.01, class_weights=weights,
        classification_weight=1.0, return_weight=0.0, rv_weight=0.0,
    )["total_loss"]
    extended.backward()
    out_b = model_b(x)
    legacy = torch.nn.CrossEntropyLoss(weight=weights)(out_b["ret_cls_logits"], targets["y_ret_cls"])
    legacy.backward()
    max_gradient_error = 0.0
    for (name_a, parameter_a), (name_b, parameter_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a != name_b:
            raise ObjectiveProbeError("parity model parameter order mismatch")
        grad_a = parameter_a.grad
        grad_b = parameter_b.grad
        if grad_a is None and grad_b is None:
            continue
        if grad_a is None:
            grad_a = torch.zeros_like(parameter_a)
        if grad_b is None:
            grad_b = torch.zeros_like(parameter_b)
        max_gradient_error = max(max_gradient_error, float(torch.max(torch.abs(grad_a - grad_b)).item()))
    loss_error = float(torch.abs(extended.detach() - legacy.detach()).item())
    return {
        "kind": kind,
        "classification_loss_absolute_error": loss_error,
        "classification_gradient_max_absolute_error": max_gradient_error,
        "passed": bool(loss_error == 0.0 and max_gradient_error == 0.0),
        "return_weight": 0.0,
        "rv_weight": 0.0,
    }


def probe_architecture(kind: str, *, seed: int = 24101) -> dict[str, Any]:
    import torch
    _seed(seed)
    model, constructor = _model(kind)
    model.eval()
    x, targets = _synthetic_batch()
    weights = torch.tensor([1.2, 0.8], dtype=torch.float32)

    # Component-specific shared-backbone norms are diagnostic only.
    component_shared_norms: dict[str, float] = {}
    for component in ("classification_loss", "return_regression_loss", "rv_regression_loss"):
        model.zero_grad(set_to_none=True)
        outputs = model(x)
        losses = candidate_multitask_loss(
            outputs, targets, ret_scale=0.02, rv_scale=0.01, class_weights=weights,
        )
        multiplier = {"classification_loss": 1.0, "return_regression_loss": 0.5, "rv_regression_loss": 0.5}[component]
        (losses[component] * multiplier).backward()
        component_shared_norms[component] = _gradient_norms(model, kind)["shared_backbone"]

    model.zero_grad(set_to_none=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    outputs = model(x)
    if tuple(outputs["ret_reg"].shape) != (6,) or tuple(outputs["rv_reg"].shape) != (6,):
        raise ObjectiveProbeError(f"{kind} regression output shape changed")
    if tuple(outputs["ret_cls_logits"].shape) != (6, 2):
        raise ObjectiveProbeError(f"{kind} classification output shape changed")
    losses = candidate_multitask_loss(
        outputs, targets, ret_scale=0.02, rv_scale=0.01, class_weights=weights,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    norms = _gradient_norms(model, kind)
    all_gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    optimizer.step()
    changed = _changed_groups(before, model, kind)
    positive_component_norms = [value for value in component_shared_norms.values() if value > 0]
    ratio = max(positive_component_norms) / min(positive_component_norms) if positive_component_norms else math.inf
    warnings = []
    if ratio > 100.0:
        warnings.append("weighted_component_shared_gradient_ratio_exceeds_100x")
    passed = (
        all(value > 0 and math.isfinite(value) for value in norms.values())
        and all_gradients_finite
        and all(changed.values())
    )
    parity = classification_parity_probe(kind, seed=seed)
    if not parity["passed"]:
        passed = False
    return {
        "kind": kind,
        "seed": seed,
        "constructor": constructor,
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "raw_unit_output_semantics_preserved": True,
        "post_hoc_clipping_present": False,
        "gradient_norms": norms,
        "weighted_component_shared_gradient_norms": component_shared_norms,
        "maximum_component_gradient_ratio": float(ratio),
        "all_gradients_finite": all_gradients_finite,
        "optimizer_step_changed": changed,
        "classification_parity": parity,
        "warnings": warnings,
        "passed": bool(passed),
    }


def build_probe_report(*, seed: int = 24101) -> dict[str, Any]:
    models = {kind: probe_architecture(kind, seed=seed) for kind in ("lstm", "tcn", "tx")}
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "synthetic_only": True,
        "seed": seed,
        "models": models,
        "all_architectures_passed": all(value["passed"] for value in models.values()),
        "task_weights_changed_by_probe": False,
    }
    report["probe_digest"] = _digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out")
    parser.add_argument("--seed", type=int, default=24101)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_probe_report(seed=args.seed)
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["all_architectures_passed"] else 3
    except ObjectiveProbeError as exc:
        print(json.dumps({"status": "objective_probe_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
