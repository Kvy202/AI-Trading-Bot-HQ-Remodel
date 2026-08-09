"""Audit history and resolve the immutable Phase 24.1 candidate objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_auxiliary_head_audit import build_auxiliary_audit
from tools.model_candidate_objective import (
    candidate_objective_digest,
    load_objective_policy,
    objective_policy_digest,
)
from tools.model_objective_label_audit import (
    build_label_audit,
    load_resolved_specification,
    resolve_target_contract,
)
from tools.model_objective_probe import build_probe_report


OBJECTIVE_REPORT = BASE_DIR / "reports" / "model_objective_contract.json"
ALLOWED_VERDICTS = {
    "candidate_objective_contract_resolved_multitask_training_required",
    "candidate_objective_contract_resolved_classification_only_safe",
    "candidate_objective_contract_requires_downstream_decoupling",
    "candidate_objective_contract_incomplete",
}


class ObjectiveContractError(ValueError):
    """Objective evidence, digest, or resolution gate failed closed."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repository, check=False, capture_output=True, text=True, encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ObjectiveContractError(f"git history audit failed: {' '.join(args)}: {completed.stderr.strip()}")
    return completed.stdout


def classify_historical_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [dict(row) for row in records if row.get("evidence_level") == "exact"]
    partial = [dict(row) for row in records if row.get("evidence_level") == "partial"]
    formulas = {
        _digest({
            "formula": row.get("loss_formula"), "weights": row.get("task_weights"),
            "targets": row.get("target_definitions"),
        })
        for row in exact
    }
    if len(formulas) > 1:
        classification = "conflicting_historical_objective_evidence"
    elif exact:
        classification = "exact_historical_objective_recovered"
    elif partial:
        classification = "partial_historical_objective_evidence"
    else:
        classification = "no_historical_multitask_objective_found"
    return {
        "classification": classification,
        "exact_records": exact,
        "partial_records": partial,
        "fails_closed": classification == "conflicting_historical_objective_evidence",
    }


def search_git_history(repository: Path | str = BASE_DIR) -> dict[str, Any]:
    root = Path(repository)
    paths = ("ml_dl/dl_train.py", "ml_dl/dl_models.py", "ml_dl/dl_dataset.py")
    commit_lines = _git(root, "log", "--all", "--reverse", "--format=%H%x09%aI%x09%s", "--", *paths)
    commits = []
    for line in commit_lines.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            commits.append(parts)
    # git path simplification can omit a tree-equivalent child, but one complete
    # source lineage is sufficient; all relevant train.py versions are inspected.
    records: list[dict[str, Any]] = []
    searched_terms = [
        "MSELoss", "SmoothL1Loss", "multitask", "loss_ret", "loss_rv",
        "lambda_ret", "lambda_rv", "y_ret_reg", "y_rv_reg",
    ]
    for commit, committed_at, subject in commits:
        shown = subprocess.run(
            ["git", "show", f"{commit}:ml_dl/dl_train.py"], cwd=root,
            check=False, capture_output=True, text=True, encoding="utf-8",
        )
        if shown.returncode != 0:
            continue
        source = shown.stdout
        has_three_head_loss = all(token in source for token in (
            'crit_reg(out["ret_reg"], y_rr)', 'crit_reg(out["rv_reg"], y_rv)',
            'crit_cls(out["ret_cls_logits"], y_rc)', "nn.MSELoss()",
        ))
        mentions = sorted(term for term in searched_terms if term in source)
        if has_three_head_loss:
            weight_match = re.search(
                r"weights:\s*Tuple\[float,\s*float,\s*float\]\s*=\s*\(([^)]+)\)", source
            )
            default_weights = [1.0, 1.0, 1.0]
            if weight_match:
                default_weights = [float(item.strip()) for item in weight_match.group(1).split(",")]
            records.append({
                "evidence_level": "exact",
                "source_commit": commit,
                "source_date": committed_at,
                "source_subject": subject,
                "source_path": "ml_dl/dl_train.py",
                "loss_formula": (
                    "w_ret*MSE(ret_reg,y_ret_reg) + w_rv*MSE(rv_reg,y_rv_reg) + "
                    "w_cls*CrossEntropy(ret_cls_logits,y_ret_cls)"
                ),
                "task_weights": {
                    "return_regression": default_weights[0],
                    "volatility_regression": default_weights[1],
                    "classification": default_weights[2],
                },
                "target_definitions": {
                    "ret_reg": "next_k_logret(prices,horizon)",
                    "rv_reg": "next_k_rv(log(prices),horizon)",
                    "classification": "binarize_return(ret_reg,tau=0.0005)",
                    "common_horizon_bars": 12,
                },
                "optimizer": "AdamW(model.parameters(),lr=lr) with library-default weight_decay",
                "scheduler": None,
                "selection": "lowest total validation loss",
                "mentioned_search_terms": mentions,
                "source_blob_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            })
        elif mentions:
            records.append({
                "evidence_level": "partial",
                "source_commit": commit,
                "source_date": committed_at,
                "source_subject": subject,
                "source_path": "ml_dl/dl_train.py",
                "mentioned_search_terms": mentions,
                "source_blob_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            })
    result = classify_historical_evidence(records)
    result.update({
        "searched_paths": list(paths),
        "searched_terms": searched_terms,
        "commits_inspected": len(commits),
        "lineage_assessment": {
            "historical_heads_intended_to_be_trained": bool(result["exact_records"]),
            "predates_incumbent_artifacts": True,
            "plausibly_produced_current_incumbents": False,
            "evidence": (
                "The recovered objective used simple fixed-horizon labels and was removed by 7cbcac0; "
                "tracked incumbents report triple labels and training timestamps after that removal."
            ),
            "compatible_head_shapes": True,
            "compatible_with_current_target_semantics": False,
            "adoption_decision": "retain_as_lineage_evidence_only",
            "non_adoption_rationale": (
                "raw equal-weight MSE was not normalized, classification was unweighted, all targets shared "
                "a 12-bar horizon, and the formula does not plausibly describe the current incumbents"
            ),
        },
    })
    return result


def expected_objective_contract_digest() -> str:
    target = resolve_target_contract(load_resolved_specification())
    return candidate_objective_digest(
        policy=load_objective_policy(),
        target_contract_digest=target["target_contract_digest"],
        objective_source="new_candidate_only_contract",
    )


def validate_objective_report(
    path: Path | str = OBJECTIVE_REPORT, *, expected_digest: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ObjectiveContractError("candidate objective contract report required")
    report = json.loads(target.read_text(encoding="utf-8-sig"))
    verdict = report.get("overall_decision", {}).get("verdict")
    if verdict != "candidate_objective_contract_resolved_multitask_training_required":
        raise ObjectiveContractError("candidate objective contract is not resolved for multitask training")
    digest = str(expected_digest or expected_objective_contract_digest())
    if report.get("objective_contract_digest") != digest:
        raise ObjectiveContractError("objective contract digest mismatch")
    blockers = report.get("promotion_blockers", {})
    if blockers.get("objective_contract_blocker") is not False:
        raise ObjectiveContractError("objective_contract_blocker must be false")
    if report.get("resolved_candidate_objective", {}).get("objective_contract_digest") != digest:
        raise ObjectiveContractError("resolved candidate objective digest mismatch")
    if report.get("synthetic_gradient_probe", {}).get("all_architectures_passed") is not True:
        raise ObjectiveContractError("synthetic gradient probe must pass")
    return report


def build_contract_report(repository: Path | str = BASE_DIR) -> dict[str, Any]:
    root = Path(repository)
    label = build_label_audit()
    target_digest = label["target_contract_digest"]
    policy = load_objective_policy()
    policy_digest = objective_policy_digest(policy)
    contract_digest = candidate_objective_digest(
        policy=policy, target_contract_digest=target_digest,
        objective_source="new_candidate_only_contract",
    )
    historical = search_git_history(root)
    downstream = build_auxiliary_audit(root)
    probe = build_probe_report()
    target_resolved = all(label[key] for key in (
        "classification_target", "return_target", "volatility_target", "maximum_required_purge_bars",
    ))
    downstream_blocker = bool(downstream["downstream_contract_blocker"])
    objective_blocker = (
        not (target_resolved and probe["all_architectures_passed"])
        or historical["classification"] == "conflicting_historical_objective_evidence"
    )
    if objective_blocker:
        verdict = "candidate_objective_contract_incomplete"
    elif downstream_blocker:
        verdict = "candidate_objective_contract_requires_downstream_decoupling"
    elif downstream["classification_only_safe"]:
        verdict = "candidate_objective_contract_resolved_classification_only_safe"
    else:
        verdict = "candidate_objective_contract_resolved_multitask_training_required"
    if verdict not in ALLOWED_VERDICTS:
        raise ObjectiveContractError("invalid Phase 24.1 verdict")
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "legacy_training_objective": {
            "name": "classification_cross_entropy_only_legacy_objective",
            "formula": "CrossEntropyLoss(ret_cls_logits, y_ret_cls, inverse_frequency_training_class_weights)",
            "ret_reg_optimized": False,
            "rv_reg_optimized": False,
            "optimizer": {"name": "AdamW", "weight_decay": 0.0001},
            "scheduler": {"name": "CosineAnnealingLR", "eta_min_factor": 0.1},
            "gradient_clipping": 1.0,
            "best_epoch_rule": "highest_validation_auc_then_lower_validation_classification_loss",
            "source_path": "ml_dl/dl_train.py:train_once",
            "source_code_digest": _file_digest(root / "ml_dl" / "dl_train.py"),
        },
        "historical_objective_search": historical,
        "classification_target": label["classification_target"],
        "return_target": label["return_target"],
        "volatility_target": label["volatility_target"],
        "resolved_candidate_objective": {
            "name": "resolved_candidate_objective",
            "objective_source": "new_candidate_only_contract",
            "objective_schema_version": 1,
            "objective_policy_digest": policy_digest,
            "objective_contract_digest": contract_digest,
            "formula": "L_cls + 0.5*L_ret + 0.5*L_rv",
            "classification_loss": "weighted CrossEntropyLoss(ret_cls_logits,y_ret_cls)",
            "return_loss": "mean(((ret_reg-y_ret_reg)/ret_target_scale)**2)",
            "rv_loss": "mean(((rv_reg-y_rv_reg)/rv_target_scale)**2)",
            "weights": {"classification": 1.0, "return_regression": 0.5, "volatility_regression": 0.5},
            "raw_model_output_units_preserved": True,
            "post_hoc_rv_clipping": False,
            "historical_formula_restored": False,
            "rationale": (
                "The exact historical formula is not defensible for the current triple-label, mixed-horizon contract; "
                "the explicit candidate-only loss normalizes residuals using training-sequence targets."
            ),
        },
        "loss_normalization": {
            "source": "valid_training_sequence_endpoints_only",
            "ret_target_scale": "population standard deviation of finite y_ret_reg training endpoints",
            "rv_target_scale": "population standard deviation of finite y_rv_reg training endpoints",
            "minimum_target_scale": 1e-12,
            "validation_used": False,
            "internal_test_used": False,
            "legacy_repair_used": False,
            "sealed_confirmation_used": False,
            "outputs_are_z_scores": False,
        },
        "downstream_consumers": {
            "audit_digest": downstream["audit_digest"],
            "consumer_count": downstream["consumer_count"],
            "active_current_remodel_consumers": downstream["active_current_remodel_consumers"],
            "rv_hat_affects_current_remodel_decisions": downstream["rv_hat_affects_current_remodel_decisions"],
            "classification_only_safe": downstream["classification_only_safe"],
            "decoupling_feasibility": downstream["decoupling_feasibility"],
        },
        "rv_unit_compatibility": downstream["rv_unit_compatibility"],
        "target_lookahead": {
            "classification_bars": label["classification_lookahead_bars"],
            "ret_reg_bars": label["ret_reg_lookahead_bars"],
            "rv_reg_bars": label["rv_reg_lookahead_bars"],
            "maximum_required_purge_bars": label["maximum_required_purge_bars"],
            "target_contract_digest": target_digest,
        },
        "synthetic_gradient_probe": {
            "probe_digest": probe["probe_digest"],
            "all_architectures_passed": probe["all_architectures_passed"],
            "per_model": {
                kind: {
                    "passed": value["passed"], "gradient_norms": value["gradient_norms"],
                    "optimizer_step_changed": value["optimizer_step_changed"],
                    "classification_parity": value["classification_parity"], "warnings": value["warnings"],
                }
                for kind, value in probe["models"].items()
            },
        },
        "promotion_blockers": {
            "objective_contract_blocker": objective_blocker,
            "candidate_auxiliary_health_blocker": "unverified",
            "downstream_contract_blocker": downstream_blocker,
            "promotion_ready": False,
        },
        "overall_decision": {
            "verdict": verdict,
            "classification_only_candidate_finalization_allowed": False,
            "real_candidate_training_allowed": verdict == "candidate_objective_contract_resolved_multitask_training_required",
            "promotion_allowed": False,
            "live_activation_allowed": False,
        },
        "warnings": [
            *downstream["rv_unit_compatibility"]["warnings"],
            "Auxiliary predictive skill thresholds beyond hard safety failures remain a later promotion-policy decision.",
            "No Phase 24.1 result is profitability evidence.",
        ],
        "objective_contract_digest": contract_digest,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(BASE_DIR))
    parser.add_argument("--json-out")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="audit and resolve in memory without writing a report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run and args.json_out:
            raise ObjectiveContractError("--dry-run cannot write --json-out")
        if args.verify:
            report = validate_objective_report(args.json_out or OBJECTIVE_REPORT)
        else:
            report = build_contract_report(args.repository)
            if args.dry_run:
                report["dry_run"] = {
                    "enabled": True,
                    "mutation_performed": False,
                    "environment_bootstrap_performed": False,
                    "market_data_fetched": False,
                    "model_training_performed": False,
                    "candidate_artifact_created": False,
                }
            if args.json_out:
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except ObjectiveContractError as exc:
        print(json.dumps({"status": "candidate_objective_contract_incomplete", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
