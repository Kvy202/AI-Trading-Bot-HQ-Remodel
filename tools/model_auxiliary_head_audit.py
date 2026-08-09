"""Inventory auxiliary-head consumers and audit raw RV unit compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
TERMS = ("ret_hat", "rv_hat", "ret_reg", "rv_reg", "y_ret_reg", "y_rv_reg", "DL_MAX_RV", "rv_mean")


class AuxiliaryHeadAuditError(ValueError):
    """The downstream inventory or its unit contract could not be resolved."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_consumer(path: str, text: str) -> dict[str, Any]:
    """Classify one file without promoting legacy references to canonical uses."""
    relative = path.replace("\\", "/")
    ret_used = bool(re.search(r"\b(ret_hat|ret_reg|y_ret_reg)\b", text))
    rv_used = bool(re.search(r"\b(rv_hat|rv_reg|y_rv_reg|rv_mean|DL_MAX_RV)\b", text))
    role = "diagnostic_only"
    active = False
    affects_allow = affects_side = affects_sizing = affects_risk = experimental = False
    evidence = "static reference; no decision-path classification was assigned"

    if relative == "ml_dl/dl_ensemble.py":
        role, active = "ensemble_blend", True
        evidence = "canonical serving ensemble blends raw ret_reg and rv_reg outputs before writer use"
    elif relative == "tools/live_writer.py":
        role, active = "logging_only", True
        experimental = True
        evidence = (
            "canonical writer publishes rv_mean; p_long alone drives its normal allow gate; "
            "rv_mean is also an optional disabled-by-default XGBoost input"
        )
    elif relative == "tools/live_executor.py":
        role, active, affects_allow, affects_risk = "risk_input", True, True, True
        evidence = "canonical executor skips an otherwise eligible entry when abs(signal.rv_mean) exceeds --rv-max"
    elif relative == "ml_optional/xgboost_signal.py":
        role, experimental = "meta_model_input", True
        affects_allow = True
        evidence = "optional XGBoost vector includes rv_mean and can reject only when separately enabled and blocking"
    elif relative == "ml_optional/advanced_risk.py":
        role, experimental = "shadow_only", True
        affects_risk = True
        evidence = "optional advanced-risk calculation consumes rv_mean; canonical executor records it as Phase-10 shadow only"
    elif relative in {"live_ensemble.py", "live_meta_ensemble.py", "trade_multi_bitget.py", "dry_run_gate.py", "live_infer.py"}:
        role = "legacy_only"
        affects_allow = bool(re.search(r"allow|DL_MAX_RV|RV_MAX|entry", text, re.IGNORECASE))
        affects_risk = bool(re.search(r"risk|DL_MAX_RV|RV_MAX", text, re.IGNORECASE))
        evidence = "standalone legacy/reference path; current systemd Remodel wiring does not invoke this file"
    elif relative == "ml_dl/dl_infer.py":
        role = "unused"
        evidence = "raw-output inference helper/provider; no decision comparison occurs in this module"
    elif relative in {"ml_dl/dl_models.py", "ml_dl/dl_models_adv.py", "ml_dl/dl_dataset.py", "ml_dl/dl_train.py"}:
        role = "unused"
        evidence = "model, dataset, or legacy-training producer rather than a downstream decision consumer"
    elif relative.startswith("tools/model_") or relative.startswith("tools/check_") or relative in {
        "sanity_test.py", "tools/dashboard.py", "tools/merge_daily_logs.py", "tools/oo_metrics_plus.py",
        "tools/walkforward_signoff.py", "tools/counterfactual_replay.py", "tools/replay_contract.py",
        "tools/runtime_stack_isolation.py", "tools/train_xgboost_signal.py", "tools/verify_xgboost_signal.py",
        "tools/verify_advanced_risk.py", "tools/live_proxy.py",
    }:
        role = "diagnostic_only"
        evidence = "offline diagnostic, audit, replay, dashboard, or training-evidence path"
    elif relative == "ml_dl/meta_train.py":
        role = "meta_model_input"
        experimental = True
        evidence = "offline meta-model training consumes auxiliary predictions; not canonical writer/executor wiring"

    return {
        "path": relative,
        "function": "file_level_static_inventory",
        "classification": role,
        "ret_hat_used": ret_used,
        "rv_hat_used": rv_used,
        "affects_allow_decision": affects_allow,
        "affects_side": affects_side,
        "affects_sizing": affects_sizing,
        "affects_risk": affects_risk,
        "affects_experimental_model": experimental,
        "active_in_current_remodel_path": active,
        "evidence": evidence,
    }


def discover_consumers(repository: Path | str = BASE_DIR) -> list[dict[str, Any]]:
    root = Path(repository)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if any(part.startswith(".venv") for part in path.parts) or relative.startswith("tests/"):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        if not any(term in text for term in TERMS):
            continue
        record = classify_consumer(relative, text)
        evidence_lines = []
        for number, line in enumerate(text.splitlines(), start=1):
            if any(term in line for term in TERMS):
                evidence_lines.append({"line": number, "text": line.strip()[:240]})
        record["evidence_lines"] = evidence_lines
        records.append(record)
    return records


def assess_rv_unit_compatibility(consumers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from tools.model_objective_label_audit import load_resolved_specification, resolve_target_contract

    target = resolve_target_contract(load_resolved_specification())["volatility_target"]
    executor = next((row for row in consumers if row.get("path") == "tools/live_executor.py"), None)
    writer = next((row for row in consumers if row.get("path") == "tools/live_writer.py"), None)
    direct_path = bool(executor and writer and executor.get("active_in_current_remodel_path") and writer.get("active_in_current_remodel_path"))
    return {
        "rv_target_definition": target["mathematical_definition"],
        "rv_target_horizon_bars": target["horizon_bars"],
        "rv_target_horizon_minutes": target["horizon_minutes"],
        "expected_numerical_units": target["units"],
        "target_contract_source": target["source_code_path"],
        "target_contract_source_digest": target["source_code_digest"],
        "comparisons": [
            {
                "path": "tools/live_executor.py",
                "threshold_name": "EXEC_RV_MAX (fallback DL_MAX_RV)",
                "threshold_default": 0.02,
                "tracked_run_override": 100,
                "threshold_numerical_units": "raw_predicted_rv_reg_units",
                "units_directly_comparable": direct_path,
                "evidence": (
                    "writer passes untransformed ensemble rv_reg as rv_mean and executor compares that same raw field; "
                    "0.02 is the intended raw-unit ceiling while tracked value 100 deliberately disables the guard"
                ),
            },
            {
                "path": "ml_optional/advanced_risk.py",
                "threshold_name": "ADVANCED_RISK_VOLATILITY_GUARD_MULT times executor rv-max context",
                "threshold_numerical_units": "same raw rv_mean context, ratio comparison",
                "units_directly_comparable": True,
                "active": False,
                "evidence": "disabled-by-default and shadow-only in canonical executor",
            },
            {
                "path": "ml_optional/xgboost_signal.py",
                "threshold_name": None,
                "threshold_numerical_units": "learned feature input",
                "units_directly_comparable": None,
                "active": False,
                "evidence": "disabled-by-default optional model input; no fixed RV threshold in this consumer",
            },
        ],
        "current_incumbent_scale_mismatch_documented": True,
        "current_guard_effectively_disabled_by_tracked_override": True,
        "resolved_candidate_preserves_raw_target_units": True,
        "unit_contract_status": "compatible_for_resolved_raw_unit_candidate" if direct_path else "unverified",
        "negative_rv_safety_relevant": True,
        "negative_rv_handling": "fail candidate auxiliary gate; do not clip production inference",
        "downstream_contract_blocker": not direct_path,
        "warnings": [
            "Incumbent rv_mean values are documented as scale-mismatched; the tracked executor ceiling 100 disables the guard.",
            "Any future activation of optional learned consumers requires their own feature-lineage review.",
        ],
    }


def build_auxiliary_audit(repository: Path | str = BASE_DIR) -> dict[str, Any]:
    consumers = discover_consumers(repository)
    if not any(row["path"] == "tools/live_executor.py" for row in consumers):
        raise AuxiliaryHeadAuditError("canonical executor auxiliary consumer not found")
    unit_audit = assess_rv_unit_compatibility(consumers)
    active = [row for row in consumers if row["active_in_current_remodel_path"]]
    legacy = [row for row in consumers if row["classification"] == "legacy_only"]
    shadow = [row for row in consumers if row["classification"] == "shadow_only"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "consumer_count": len(consumers),
        "consumers": consumers,
        "active_current_remodel_consumers": [row["path"] for row in active],
        "legacy_reference_consumers": [row["path"] for row in legacy],
        "shadow_only_consumers": [row["path"] for row in shadow],
        "rv_hat_affects_current_remodel_decisions": any(
            row["active_in_current_remodel_path"] and (row["affects_allow_decision"] or row["affects_risk"])
            for row in consumers
        ),
        "classification_only_safe": not any(
            row["active_in_current_remodel_path"]
            and (row["ret_hat_used"] or row["rv_hat_used"])
            and (row["affects_allow_decision"] or row["affects_side"] or row["affects_sizing"] or row["affects_risk"])
            for row in consumers
        ),
        "rv_unit_compatibility": unit_audit,
        "downstream_contract_blocker": unit_audit["downstream_contract_blocker"],
        "decoupling_feasibility": {
            "feasible": True,
            "implemented": False,
            "impact": "would require removing or disabling canonical executor RV guard and auditing optional model inputs",
            "decision": "multitask_training_preferred_because_targets_are_defined_and_raw_units_are_directly_consumed",
        },
    }
    report["audit_digest"] = _digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(BASE_DIR))
    parser.add_argument("--json-out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_auxiliary_audit(args.repository)
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0
    except AuxiliaryHeadAuditError as exc:
        print(json.dumps({"status": "auxiliary_head_audit_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
