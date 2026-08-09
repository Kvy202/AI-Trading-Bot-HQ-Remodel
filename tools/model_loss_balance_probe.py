"""Full-production-architecture synthetic Phase 24.2 balance probe.

This is implementation evidence only.  It cannot write a real-data balance
freeze and cannot select final candidate-training weights.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_loss_balance import (
    ARCHITECTURES,
    PARENT_OBJECTIVE_DIGEST,
    _serializable,
    balance_policy_digest,
    evaluate_formulations,
    load_balance_policy,
    measure_task_gradients,
    validate_parent_objective,
)


PRODUCTION_CONSTRUCTORS = {
    "lstm": {"in_dim": 27, "hidden": 64, "layers": 2, "dropout": 0.1},
    "tcn": {"in_dim": 27, "hid": 64, "levels": 4, "kernel": 3, "dropout": 0.1},
    "tx": {"in_dim": 27, "d_model": 64, "nhead": 4, "nlayers": 2, "dropout": 0.1},
}


class LossBalanceProbeError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def production_model(kind: str):
    from ml_dl.dl_models import TemporalConvNet, TinyLSTM, TinyTransformer
    config = copy.deepcopy(PRODUCTION_CONSTRUCTORS[kind])
    cls = {"lstm": TinyLSTM, "tcn": TemporalConvNet, "tx": TinyTransformer}[kind]
    return cls(**config), config


def synthetic_batch():
    import torch
    # Smooth, nonconstant values avoid accidental dead ReLU/head gradients.
    x = torch.linspace(-1.25, 1.75, steps=8 * 64 * 27, dtype=torch.float32).reshape(8, 64, 27)
    return {
        "x": x,
        "y_ret_cls": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        "y_ret_reg": torch.tensor([-0.04, 0.03, -0.02, 0.015, -0.01, 0.02, -0.03, 0.04]),
        "y_rv_reg": torch.tensor([0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027]),
    }


def probe_architecture(kind: str, *, seed: int = 24201) -> dict[str, Any]:
    import torch
    _seed(seed)
    model, constructor = production_model(kind)
    model.train().cpu()
    state = copy.deepcopy(model.state_dict())
    batch = synthetic_batch()
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
    mse = measure_task_gradients(
        model, kind, batch, formulation_id="normalized_mse_fixed",
        class_weights=class_weights, ret_scale=0.025, rv_scale=0.008, seed=seed,
    )
    model.load_state_dict(state)
    huber = measure_task_gradients(
        model, kind, batch, formulation_id="normalized_huber_fixed",
        class_weights=class_weights, ret_scale=0.025, rv_scale=0.008, seed=seed,
    )
    result = evaluate_formulations([mse], [huber])
    with torch.no_grad():
        torch.manual_seed(seed)
        outputs = model(batch["x"])
    result.update({
        "kind": kind,
        "constructor": constructor,
        "exact_production_candidate_architecture": True,
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "outputs_remain_in_raw_runtime_units": True,
        "post_hoc_clipping_applied": False,
    })
    return _serializable(result)


def build_probe_report(*, seed: int = 24201) -> dict[str, Any]:
    parent = validate_parent_objective()
    policy = load_balance_policy()
    architectures = {kind: probe_architecture(kind, seed=seed) for kind in ARCHITECTURES}
    payload = {
        "schema_version": 1,
        "synthetic_only": True,
        "synthetic_balance_only": True,
        "final_weight_evidence": False,
        "eligible_to_freeze_real_weights": False,
        "existing_probe_scope": "reduced_synthetic_architecture",
        "this_probe_scope": "full_production_candidate_architectures_synthetic_data",
        "seed": int(seed),
        "parent_objective_contract_digest": parent["objective_contract_digest"],
        "required_parent_objective_contract_digest": PARENT_OBJECTIVE_DIGEST,
        "balance_policy_digest": balance_policy_digest(policy),
        "architectures": architectures,
        "real_balance_freeze_created": False,
        "real_balance_freeze_status": "pending_training_data",
        "task_weights_frozen_for_real_training": False,
        "all_architectures_numerically_valid": all(
            not any(
                "nonfinite" in reason or "zero_task" in reason
                for form in item["formulations"].values()
                for reason in form["failure_reasons"]
            )
            for item in architectures.values()
        ),
    }
    payload["probe_digest"] = _digest(payload)
    return {"generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out")
    parser.add_argument("--seed", type=int, default=24201)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_probe_report(seed=args.seed)
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["all_architectures_numerically_valid"] else 3
    except Exception as exc:
        print(json.dumps({"status": "loss_balance_probe_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
