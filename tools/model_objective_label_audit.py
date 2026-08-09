"""Resolve and verify the Phase 24.1 target and maximum-lookahead contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SPECIFICATION = BASE_DIR / "reports" / "model_retraining_specification_phase23_1.json"
LABEL_SOURCE = BASE_DIR / "ml_dl" / "dl_labels.py"


class ObjectiveLabelAuditError(ValueError):
    """The source specification or executable label semantics are incomplete."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_resolved_specification(path: Path | str = SPECIFICATION) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    models = value.get("models", {})
    if set(models) != {"lstm", "tcn", "tx"}:
        raise ObjectiveLabelAuditError("training_pipeline_contract_incomplete")
    contracts = [models[kind].get("label_contract") for kind in ("lstm", "tcn", "tx")]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ObjectiveLabelAuditError("conflicting candidate label contracts")
    contract = contracts[0]
    required = {"type", "pt", "sl", "max_hold", "tau", "horizon"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ObjectiveLabelAuditError("training_pipeline_contract_incomplete")
    if contract["type"] != "triple" or contract["tau"] is not None:
        raise ObjectiveLabelAuditError("Phase 24 requires the proven triple-label contract")
    if type(contract["max_hold"]) is not int or type(contract["horizon"]) is not int:
        raise ObjectiveLabelAuditError("label horizons must be integer bars")
    if min(contract["max_hold"], contract["horizon"]) <= 0:
        raise ObjectiveLabelAuditError("label horizons must be positive")
    return dict(contract)


def resolve_target_contract(
    label_contract: Mapping[str, Any], *, timeframe_minutes: int = 5,
    label_source: Path | str = LABEL_SOURCE,
) -> dict[str, Any]:
    cfg = dict(label_contract)
    if cfg.get("type") != "triple":
        raise ObjectiveLabelAuditError("only the resolved triple-label contract is supported")
    max_hold = int(cfg["max_hold"])
    rv_horizon = int(cfg["horizon"])
    maximum = max(max_hold, max_hold, rv_horizon)
    source_digest = _file_digest(Path(label_source))
    targets: dict[str, Any] = {
        "classification_target": {
            "name": "triple_barrier_outcome",
            "model_output_key": "ret_cls_logits",
            "dataset_target_key": "y_ret_cls",
            "mathematical_definition": (
                "1 when +pt is reached before -sl within max_hold future bars; "
                "0 when -sl is reached first; NaN on timeout or insufficient future bars"
            ),
            "units": "binary_class",
            "horizon_bars": max_hold,
            "horizon_minutes": max_hold * int(timeframe_minutes),
            "minimum_possible_value": 0,
            "maximum_theoretical_value": 1,
            "expected_sign_domain": "{0,1}",
            "nan_rule": "timeout_or_fewer_than_max_hold_future_bars",
            "purge_lookahead_requirement_bars": max_hold,
            "barrier_parameters": {"pt": float(cfg["pt"]), "sl": float(cfg["sl"])},
            "source_code_path": "ml_dl/dl_labels.py:triple_barrier_label",
            "source_code_digest": source_digest,
        },
        "return_target": {
            "name": "forward_log_return",
            "model_output_key": "ret_reg",
            "dataset_target_key": "y_ret_reg",
            "mathematical_definition": "log(price[t + max_hold]) - log(price[t])",
            "units": "dimensionless_log_return",
            "horizon_bars": max_hold,
            "horizon_minutes": max_hold * int(timeframe_minutes),
            "minimum_possible_value": None,
            "maximum_theoretical_value": None,
            "expected_sign_domain": "signed_real",
            "nan_rule": "fewer_than_max_hold_future_bars",
            "purge_lookahead_requirement_bars": max_hold,
            "source_code_path": "ml_dl/dl_labels.py:next_k_logret",
            "source_code_digest": source_digest,
        },
        "volatility_target": {
            "name": "forward_realized_log_return_volatility",
            "model_output_key": "rv_reg",
            "dataset_target_key": "y_rv_reg",
            "mathematical_definition": (
                "sqrt(sum((log(price[j + 1]) - log(price[j])) ** 2 "
                "for j=t..t+horizon-1))"
            ),
            "units": "dimensionless_root_sum_squared_log_return",
            "horizon_bars": rv_horizon,
            "horizon_minutes": rv_horizon * int(timeframe_minutes),
            "minimum_possible_value": 0.0,
            "maximum_theoretical_value": None,
            "expected_sign_domain": "nonnegative_real",
            "nan_rule": "fewer_than_horizon_future_bar_differences",
            "purge_lookahead_requirement_bars": rv_horizon,
            "source_code_path": "ml_dl/dl_labels.py:next_k_rv",
            "source_code_digest": source_digest,
        },
    }
    result = {
        **targets,
        "timeframe": "5m",
        "timeframe_minutes": int(timeframe_minutes),
        "classification_lookahead_bars": max_hold,
        "ret_reg_lookahead_bars": max_hold,
        "rv_reg_lookahead_bars": rv_horizon,
        "maximum_required_purge_bars": maximum,
        "horizons_are_intentionally_independent": True,
        "label_contract": cfg,
    }
    result["target_contract_digest"] = _digest(result)
    return result


def verify_executable_semantics(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the actual label functions on deterministic synthetic prices."""
    from ml_dl.dl_labels import next_k_logret, next_k_rv

    cfg = contract["label_contract"]
    length = max(int(cfg["max_hold"]), int(cfg["horizon"])) + 8
    prices = np.exp(np.linspace(0.0, 0.04, length, dtype=np.float64))
    ret = next_k_logret(prices, int(cfg["max_hold"]))
    rv = next_k_rv(np.log(prices), int(cfg["horizon"]))
    expected_ret = np.log(prices[int(cfg["max_hold"])]) - np.log(prices[0])
    increments = np.diff(np.log(prices[:int(cfg["horizon"]) + 1]))
    expected_rv = np.sqrt(np.sum(increments ** 2))
    if not np.isclose(ret[0], expected_ret, rtol=0.0, atol=1e-15):
        raise ObjectiveLabelAuditError("ret_reg executable formula mismatch")
    if not np.isclose(rv[0], expected_rv, rtol=0.0, atol=1e-15):
        raise ObjectiveLabelAuditError("rv_reg executable formula mismatch")
    if np.nanmin(rv) < 0:
        raise ObjectiveLabelAuditError("rv target violated nonnegative domain")
    return {
        "ret_formula_verified": True,
        "rv_formula_verified": True,
        "rv_target_nonnegative_verified": True,
        "raw_units_preserved": True,
    }


def build_label_audit(specification: Path | str = SPECIFICATION) -> dict[str, Any]:
    contract = resolve_target_contract(load_resolved_specification(specification))
    report = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        **contract,
        "executable_semantics": verify_executable_semantics(contract),
    }
    report["audit_digest"] = _digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", default=str(SPECIFICATION))
    parser.add_argument("--json-out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_label_audit(args.specification)
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except ObjectiveLabelAuditError as exc:
        print(json.dumps({"status": "objective_label_audit_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
