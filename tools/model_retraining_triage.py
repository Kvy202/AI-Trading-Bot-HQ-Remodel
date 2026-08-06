"""Deterministic incumbent triage and Phase 24 training-contract generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features import canonical_feature_columns
from tools.model_runtime_repro import (
    DEFAULT_POLICY,
    ensure_safe_report_output,
    json_digest,
    load_retraining_policy,
)
from tools.training_lineage_audit import REQUIRED_LINEAGE_FIELDS


DEFAULT_REGISTRY = BASE_DIR / "research" / "model_candidate_registry.json"
ALLOWED_CANDIDATE_STATUSES = {
    "proposed", "training_data_pending", "ready_to_train", "trained_unverified",
    "health_gate_failed", "health_gate_passed", "rejected", "archived",
}
ALLOWED_CANDIDATE_FIELDS = {
    "candidate_id", "kind", "parent_model_digest", "dataset_digest", "feature_digest",
    "label_digest", "training_config_digest", "seed", "symbols", "timeframe",
    "sequence_length", "artifact_manifest_path", "status", "reviewed", "reason",
}
PRIMARY_ACTIONS = {
    "retain_incumbent_shadow_control", "retain_incumbent_with_warning",
    "quarantine_from_positive_weight_candidate", "retrain_required",
    "symbol_specific_retraining_required", "investigate_data_pipeline",
    "lineage_reconstruction_required", "no_decision_insufficient_evidence",
}
RETRAINING_ACTIONS = {"retrain_required", "symbol_specific_retraining_required"}


class RetrainingTriageError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_candidate_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "candidates"}:
        raise RetrainingTriageError("candidate registry fields are not exact")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RetrainingTriageError("candidate registry schema_version must be 1")
    if not isinstance(value["candidates"], Mapping):
        raise RetrainingTriageError("candidate registry candidates must be an object")
    normalized: dict[str, Any] = {"schema_version": 1, "candidates": {}}
    prohibited_key = re.compile(
        r"(?i)(command|secret|password|token|api.?key|activate|promotion|overwrite.?target|binary|environment)"
    )
    secret_value = re.compile(
        r"(?i)(?:api[_ -]?key|secret|private[_ -]?key|wallet|password|token)\s*[:=]\s*\S+"
    )
    command_value = re.compile(
        r"(?i)(?:(?<![A-Za-z0-9_])(?:powershell(?:\.exe)?|pwsh(?:\.exe)?|cmd(?:\.exe)?|bash|sh|python(?:\.exe)?|pip(?:\.exe)?)\s+(?:-|/|\S)|start-process|place[_-]?order|live_(?:writer|executor)\.py|run_experiment_matrix)"
    )
    environment_dump = re.compile(
        r"(?im)(?<![A-Za-z0-9_])(?:path|home|userprofile|codex_home|pythonpath|virtual_env)\s*="
    )
    for key, raw in value["candidates"].items():
        candidate_id = str(key)
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", candidate_id):
            raise RetrainingTriageError(f"invalid candidate id: {candidate_id}")
        if not isinstance(raw, Mapping) or not set(raw).issubset(ALLOWED_CANDIDATE_FIELDS):
            raise RetrainingTriageError(f"candidate {candidate_id} has unsupported fields")
        if any(prohibited_key.search(str(field)) for field in raw):
            raise RetrainingTriageError(f"candidate {candidate_id} contains a prohibited field")
        record = dict(raw)
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if secret_value.search(serialized):
            raise RetrainingTriageError(f"candidate {candidate_id} contains secret-like material")
        if command_value.search(serialized) or environment_dump.search(serialized):
            raise RetrainingTriageError(f"candidate {candidate_id} contains executable or environment material")
        if any(isinstance(item, str) and len(item) > 4096 for item in record.values()):
            raise RetrainingTriageError(f"candidate {candidate_id} contains oversized string data")
        if record.get("candidate_id", candidate_id) != candidate_id:
            raise RetrainingTriageError(f"candidate key/id mismatch: {candidate_id}")
        if record.get("status") not in ALLOWED_CANDIDATE_STATUSES:
            raise RetrainingTriageError(f"candidate {candidate_id} has an invalid status")
        if "kind" in record and record["kind"] not in {"adv", "lstm", "tcn", "tx"}:
            raise RetrainingTriageError(f"candidate {candidate_id} has an invalid model kind")
        for field in (
            "parent_model_digest", "dataset_digest", "feature_digest", "label_digest",
            "training_config_digest",
        ):
            if field in record and re.fullmatch(r"[0-9a-f]{64}", str(record[field])) is None:
                raise RetrainingTriageError(f"candidate {candidate_id} has an invalid {field}")
        if "seed" in record and (type(record["seed"]) is not int or record["seed"] < 0):
            raise RetrainingTriageError(f"candidate {candidate_id} seed must be a non-negative integer")
        if "symbols" in record and (
            not isinstance(record["symbols"], list)
            or not record["symbols"]
            or not all(isinstance(item, str) and item for item in record["symbols"])
            or len(set(record["symbols"])) != len(record["symbols"])
        ):
            raise RetrainingTriageError(f"candidate {candidate_id} has invalid symbols")
        if "timeframe" in record and not isinstance(record["timeframe"], str):
            raise RetrainingTriageError(f"candidate {candidate_id} timeframe must be a string")
        if "sequence_length" in record and (
            type(record["sequence_length"]) is not int or record["sequence_length"] <= 0
        ):
            raise RetrainingTriageError(f"candidate {candidate_id} sequence_length must be positive")
        if "reviewed" in record and type(record["reviewed"]) is not bool:
            raise RetrainingTriageError(f"candidate {candidate_id} reviewed must be boolean")
        path = str(record.get("artifact_manifest_path") or "").replace("\\", "/")
        parts = PurePosixPath(path).parts if path else ()
        if path and (
            not path.startswith(f"model_artifacts/candidates/{candidate_id}/")
            or ".." in parts or "." in parts
            or "latest" in path.lower() or path.startswith("/") or re.match(r"^[A-Za-z]:", path)
        ):
            raise RetrainingTriageError(f"candidate {candidate_id} has an unsafe artifact manifest path")
        text = json.dumps(record, sort_keys=True).lower()
        if "dl_" in text and "_latest.pt" in text or "scaler_" in text and "_latest.joblib" in text:
            raise RetrainingTriageError(f"candidate {candidate_id} targets an incumbent artifact")
        normalized["candidates"][candidate_id] = record
    return normalized


def deterministic_candidate_id(
    kind: str,
    *,
    parent_model_digest: str | None,
    dataset_digest: str | None,
    feature_digest: str,
    label_digest: str,
    training_config_digest: str,
    seed: int,
) -> str:
    digest = json_digest({
        "kind": kind, "parent_model_digest": parent_model_digest,
        "dataset_digest": dataset_digest, "feature_digest": feature_digest,
        "label_digest": label_digest, "training_config_digest": training_config_digest,
        "seed": int(seed),
    })
    return f"{kind}-phase24-{digest[:16]}"


def _verify_report_digest(report: Mapping[str, Any], digest_field: str) -> None:
    observed = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", digest_field}
    })
    if report.get(digest_field) != observed:
        raise RetrainingTriageError(f"{digest_field} mismatch")


def _phase22_kind_stats(report: Mapping[str, Any], kind: str) -> dict[str, Any]:
    value = report.get("model_results", {}).get(kind, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _failure_supported(failure_report: Mapping[str, Any], kind: str, category: str) -> bool:
    by_symbol = failure_report.get("model_results", {}).get(kind, {}).get("by_symbol", {})
    return any(
        data.get("failure_categories", {}).get(category, {}).get("support") == "supported"
        for data in by_symbol.values()
    )


def decide_model_action(
    kind: str,
    *,
    required_symbols: Sequence[str],
    runtime_report: Mapping[str, Any],
    phase22_report: Mapping[str, Any],
    failure_report: Mapping[str, Any],
    lineage_result: Mapping[str, Any],
    minimum_auc: float,
) -> dict[str, Any]:
    runtime_verdict = runtime_report.get("overall_decision", {}).get("verdict")
    runtime_material = runtime_verdict == "runtime_reproducibility_material_behavior_delta"
    if not runtime_material and runtime_verdict != "runtime_reproducibility_verified_no_material_delta":
        return {
            "primary_action": "no_decision_insufficient_evidence",
            "supporting_reasons": ["runtime_comparison_incomplete"],
            "required_failed_symbols": [], "live_or_blocking_use_approved": False,
        }
    collapse = runtime_report.get("collapse_comparisons", {}).get(kind, {}).get("by_symbol", {})
    statuses = {
        symbol: collapse.get(symbol, {}).get("sklearn180_runtime", {}).get("collapse_status")
        for symbol in required_symbols
    }
    evidence_action: str | None = None
    if any(status is None for status in statuses.values()):
        action = "no_decision_insufficient_evidence"
        reasons = ["missing_required_symbol_evidence"]
        failed_symbols: list[str] = []
    else:
        failed_symbols = [symbol for symbol, status in statuses.items() if status == "failed_health_gate"]
        if not failed_symbols:
            evidence_action = "retain_incumbent_shadow_control"
            reasons = ["healthy_aligned"]
        elif len(failed_symbols) == len(required_symbols):
            evidence_action = "retrain_required"
            reasons = []
        else:
            evidence_action = "symbol_specific_retraining_required"
            reasons = ["symbol_specific_failure"]
        action = "no_decision_insufficient_evidence" if runtime_material else evidence_action
        if failed_symbols:
            if any(
                collapse[symbol]["sklearn180_runtime"].get("extreme_exclusion_events", 0)
                or collapse[symbol]["sklearn180_runtime"].get("consecutive_extreme_max", 0) >= 20
                for symbol in failed_symbols
            ):
                reasons.append("persistent_extreme_collapse")
            if any(
                collapse[symbol]["sklearn180_runtime"].get("rolling_flat_window_count", 0)
                for symbol in failed_symbols
            ):
                reasons.append("persistent_flat_output")
    reasons.append(
        "runtime_difference_material" if runtime_material else "runtime_difference_not_material"
    )
    phase22 = _phase22_kind_stats(phase22_report, kind)
    warnings = {
        warning
        for stats in phase22.get("by_symbol", {}).values()
        for warning in stats.get("model_health_warnings", [])
    }
    if "warning_low_auc" in warnings:
        reasons.append("low_validation_auc")
    lineage_status = str(lineage_result.get("lineage_status") or "missing")
    if lineage_status != "complete_reproducible":
        reasons.append("training_lineage_incomplete")
    if _failure_supported(failure_report, kind, "calibration_saturation"):
        reasons.append("calibration_saturation_supported")
    elif failed_symbols:
        reasons.append("calibration_not_root_cause")
    if _failure_supported(failure_report, kind, "serving_distribution_ood"):
        reasons.append("serving_ood_supported")
    if _failure_supported(failure_report, kind, "learned_classifier_saturation"):
        reasons.append("learned_saturation_supported")
    reasons = list(dict.fromkeys(reasons))
    if action not in PRIMARY_ACTIONS:
        raise RetrainingTriageError(f"invalid primary action: {action}")
    return {
        "primary_action": action,
        "supporting_reasons": reasons,
        "required_failed_symbols": failed_symbols,
        "lineage_status": lineage_status,
        "low_auc_is_warning_only": "low_validation_auc" in reasons,
        "retention_scope": "shadow_only" if action.startswith("retain_incumbent") else None,
        "provisional_action_after_runtime_resolution": (
            evidence_action if runtime_material else None
        ),
        "live_or_blocking_use_approved": False,
    }


def _label_contract(lineage: Mapping[str, Any], phase22: Mapping[str, Any], kind: str) -> dict[str, Any]:
    value = lineage.get("lineage_fields", {}).get("label_configuration")
    if isinstance(value, Mapping):
        return dict(value)
    # Phase 22 model statistics do not contain a label contract; keep the gap explicit.
    return {"status": "must_be_explicitly_defined_and_digested", "legacy_value": None}


def generate_retraining_specification(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    runtime_report: Mapping[str, Any],
    phase22_report: Mapping[str, Any],
    lineage_report: Mapping[str, Any],
    policy: Mapping[str, Any],
    canonical_runtime_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature_names = canonical_feature_columns(True)
    feature_digest = json_digest(feature_names)
    training_symbols = phase22_report.get("serving_contract", {}).get("training_contract", {}).get(
        "ordered_symbols", []
    )
    symbol_map = {str(symbol): index for index, symbol in enumerate(training_symbols)} if training_symbols else None
    specifications: dict[str, Any] = {}
    for kind, decision in decisions.items():
        primary_action = decision.get("primary_action")
        provisional_action = decision.get("provisional_action_after_runtime_resolution")
        if primary_action not in RETRAINING_ACTIONS and provisional_action not in RETRAINING_ACTIONS:
            continue
        lineage = lineage_report.get("model_results", {}).get(kind, {})
        label = _label_contract(lineage, phase22_report, kind)
        label_digest = json_digest(label)
        config_contract = {
            "timeframe": policy["required_timeframe"],
            "sequence_length": policy["required_sequence_length"],
            "feature_count": policy["required_feature_count"],
            "symbols": policy["required_serving_symbols"],
            "time_ordered": policy["require_time_ordered_split"],
            "purged": policy["require_purged_split"],
            "scaler_fit_train_only": policy["require_scaler_fit_train_only"],
        }
        config_digest = json_digest(config_contract)
        parent_digest = runtime_report.get("model_output_comparisons", {}).get(kind, {}).get("model_digest")
        # The Phase 22 bundle is immutable evaluation evidence, not a training
        # dataset.  Keep the future training dataset identity explicitly pending.
        dataset_digest = None
        seed = 42
        candidate_id = deterministic_candidate_id(
            kind, parent_model_digest=parent_digest, dataset_digest=dataset_digest,
            feature_digest=feature_digest, label_digest=label_digest,
            training_config_digest=config_digest, seed=seed,
        )
        candidate_root = f"model_artifacts/candidates/{candidate_id}"
        specifications[kind] = {
            "kind": kind,
            "reason": decision["supporting_reasons"],
            "specification_status": (
                "ready_for_explicit_phase24_candidate_work"
                if primary_action in RETRAINING_ACTIONS
                else "blocked_pending_runtime_difference_resolution"
            ),
            "required_symbols": list(policy["required_serving_symbols"]),
            "failed_serving_symbols": list(decision.get("required_failed_symbols", [])),
            "timeframe": policy["required_timeframe"],
            "sequence_length": policy["required_sequence_length"],
            "canonical_feature_names": feature_names,
            "feature_digest": feature_digest,
            "symbol_id_map": symbol_map or {"status": "must_be_explicitly_defined_and_digested"},
            "label_contract": label,
            "label_digest": label_digest,
            "minimum_data_requirements": {
                "dataset_digest_required": True,
                "dataset_digest_status": "must_be_recorded_before_training",
                "raw_file_digests_required": True,
                "raw_start_finish_required": True,
                "exchange_venue_required": True,
                "all_required_symbols_must_be_present": True,
                "minimum_aligned_evaluation_bars_per_symbol": policy["minimum_aligned_unique_bars_per_symbol"],
                "class_distribution_required": True,
                "phase22_evaluation_bundle_is_not_training_data": True,
            },
            "required_time_split": {
                "method": "time_ordered",
                "train_validation_test_boundaries_required": True,
                "no_random_row_split": True,
            },
            "purge_embargo_requirement": {
                "required": True,
                "exact_settings_must_be_recorded": True,
                "must_cover_label_overlap": True,
            },
            "scaler_fit_scope": "training_split_only",
            "deterministic_seeds": [seed],
            "artifact_naming_policy": {
                "candidate_id": candidate_id,
                "model": f"{candidate_root}/model.pt",
                "scaler": f"{candidate_root}/scaler.joblib",
                "metadata": f"{candidate_root}/metadata.json",
                "training_manifest": f"{candidate_root}/training_manifest.json",
                "evaluation": f"{candidate_root}/evaluation.json",
                "incumbent_overwrite_allowed": False,
            },
            "candidate_directory_policy": {
                "root": candidate_root,
                "generated_directories_ignored": True,
                "automatic_promotion": False,
                "future_explicit_review_and_copy_required": True,
            },
            "required_metadata_fields": list(REQUIRED_LINEAGE_FIELDS),
            "legacy_lineage_gaps": list(lineage.get("missing_fields", [])),
            "required_health_gates": {
                "minimum_validation_auc": policy["minimum_candidate_validation_auc"],
                "maximum_missing_rate": policy["maximum_missing_rate"],
                "flat_output_std_threshold": policy["flat_output_std_threshold"],
                "no_extreme_exclusion_events": True,
                "no_flat_exclusion_events": True,
                "deterministic_cpu_inference": True,
                "all_required_symbols_must_pass": True,
            },
            "required_comparison_to_incumbent": {
                "required": True, "same_immutable_windows": True,
                "probability_max_abs_error_documented": True,
                "regression_output_max_abs_error_documented": True,
                "direction_exclusion_and_ensemble_changes_documented": True,
            },
            "required_phase22_rerun": True,
            "required_canonical_numerical_runtime": (
                {
                    "stack_id": canonical_runtime_decision.get("selected_stack_id"),
                    "python_major_minor": canonical_runtime_decision.get("python_major_minor"),
                    "package_versions": canonical_runtime_decision.get("package_versions"),
                    "lock_digest": canonical_runtime_decision.get("canonical_lock_digest"),
                    "dedicated_environment_only": True,
                    "main_runtime_migration_allowed": False,
                }
                if canonical_runtime_decision else None
            ),
            "training_config_digest": config_digest,
            "training_dataset_digest": None,
            "candidate_id_is_pretraining_contract_id": True,
            "candidate_created_or_trained_by_phase23": False,
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy_digest": json_digest(policy),
        "source_runtime_reproducibility_digest": runtime_report.get("reproducibility_digest"),
        "canonical_runtime_decision_digest": (
            canonical_runtime_decision.get("decision_digest")
            if canonical_runtime_decision else None
        ),
        "models": specifications,
        "training_allowed_by_this_specification": bool(
            canonical_runtime_decision
            and canonical_runtime_decision.get("phase24_candidate_training_allowed") is True
        ),
        "promotion_implemented": False,
        "incumbent_overwrite_allowed": False,
    }
    result["specification_digest"] = json_digest({
        key: value for key, value in result.items() if key not in {"generated_at", "specification_digest"}
    })
    return result


def build_retraining_triage_report(
    *,
    phase22_report: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
    failure_report: Mapping[str, Any],
    lineage_report: Mapping[str, Any],
    policy: Mapping[str, Any],
    candidate_registry: Mapping[str, Any],
    models: Sequence[str] | None = None,
    canonical_runtime_decision: Mapping[str, Any] | None = None,
    runtime_attribution_report: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_candidate_registry(candidate_registry)
    bundle_digest = runtime_report.get("input_bundle", {}).get("bundle_digest")
    if phase22_report.get("historical_alignment", {}).get("bundle_digest") != bundle_digest:
        raise RetrainingTriageError("Phase 22 and runtime reports reference different bundles")
    if failure_report.get("input_bundle_digest") != bundle_digest:
        raise RetrainingTriageError("failure triage and runtime reports reference different bundles")
    if (
        failure_report.get("runtime_reproducibility_digest")
        != runtime_report.get("reproducibility_digest")
    ):
        raise RetrainingTriageError("failure triage does not reference the supplied runtime report")
    if failure_report.get("phase22_alignment_digest") != phase22_report.get("alignment_digest"):
        raise RetrainingTriageError("failure triage does not reference the supplied Phase 22 report")
    selected = sorted(runtime_report.get("collapse_comparisons", {}))
    if models:
        selected = [kind for kind in selected if kind in models]
    decisions: dict[str, Any] = {}
    for kind in selected:
        decisions[kind] = decide_model_action(
            kind, required_symbols=policy["required_serving_symbols"],
            runtime_report=runtime_report, phase22_report=phase22_report,
            failure_report=failure_report,
            lineage_result=lineage_report.get("model_results", {}).get(kind, {}),
            minimum_auc=policy["minimum_candidate_validation_auc"],
        )
    runtime_verdict = runtime_report.get("overall_decision", {}).get("verdict")
    lineage_verdict = lineage_report.get("overall_decision", {}).get("verdict")
    integrity_pass = runtime_report.get("input_bundle", {}).get("integrity_result") == "pass"
    artifact_integrity_pass = (
        runtime_report.get("input_bundle", {}).get("artifact_integrity_result") == "pass"
    )
    gaps_documented = all(
        "missing_fields" in lineage_report.get("model_results", {}).get(kind, {}) for kind in selected
    )
    artifact_safe = (
        runtime_report.get("policy", {}).get("models_modified") is False
        and runtime_report.get("policy", {}).get("main_environment_modified") is False
    )
    runtime_complete = bool(
        runtime_report.get("overall_decision", {}).get("full_required_model_and_symbol_scope") is True
        and runtime_report.get("overall_decision", {}).get("worker_runs_deterministic") is True
        and runtime_report.get("overall_decision", {}).get("model_forward_passes_deterministic") is True
    )
    phase23_1_supplied = canonical_runtime_decision is not None or runtime_attribution_report is not None
    if phase23_1_supplied and not (canonical_runtime_decision and runtime_attribution_report):
        raise RetrainingTriageError(
            "canonical runtime decision and runtime attribution report must be supplied together"
        )
    canonical_resolved = False
    numerical_status = None
    behavioral_status = None
    if canonical_runtime_decision and runtime_attribution_report:
        if (
            canonical_runtime_decision.get("attribution_digest")
            != runtime_attribution_report.get("attribution_digest")
        ):
            raise RetrainingTriageError("canonical runtime decision attribution digest mismatch")
        if runtime_attribution_report.get("confidence") == "unresolved":
            raise RetrainingTriageError("runtime stack attribution remains unresolved")
        numerical_status = canonical_runtime_decision.get("numerical_status")
        behavioral_status = canonical_runtime_decision.get("behavioral_status")
        canonical_resolved = bool(
            canonical_runtime_decision.get("decision_status") == "canonical_stack_selected"
            and canonical_runtime_decision.get("selected_stack_id") == "serialized_full_stack"
            and canonical_runtime_decision.get("phase24_candidate_training_allowed") is True
            and canonical_runtime_decision.get("phase24_environment_scope")
            == ".venv-runtime-isolation/serialized_full_stack"
            and canonical_runtime_decision.get("deterministic_workers") is True
            and canonical_runtime_decision.get("deterministic_model_inference") is True
            and canonical_runtime_decision.get("all_behavior_change_counts_zero") is True
            and behavioral_status == "behaviorally_reproducible"
            and canonical_runtime_decision.get("main_runtime_migration_allowed") is False
            and canonical_runtime_decision.get("live_activation_allowed") is False
        )
    phase24_allowed = bool(
        (canonical_resolved or (
            not phase23_1_supplied
            and runtime_verdict == "runtime_reproducibility_verified_no_material_delta"
            and runtime_complete
        ))
        and integrity_pass and artifact_integrity_pass
        and gaps_documented and artifact_safe
        and lineage_verdict != "training_lineage_conflicting"
        and failure_report.get("overall_decision", {}).get("verdict") == "failure_triage_complete"
    )
    if canonical_resolved:
        for decision in decisions.values():
            provisional = decision.get("provisional_action_after_runtime_resolution")
            if provisional in PRIMARY_ACTIONS:
                decision["primary_action"] = provisional
                decision["retention_scope"] = (
                    "shadow_only" if provisional.startswith("retain_incumbent") else None
                )
                decision["runtime_block_resolved_by_canonical_stack"] = True
    if phase23_1_supplied and not canonical_resolved:
        verdict = "retraining_triage_blocked_runtime_difference"
        final = "model_reproducibility_material_difference_requires_resolution"
    elif runtime_verdict == "runtime_reproducibility_material_behavior_delta" and not canonical_resolved:
        verdict = "retraining_triage_blocked_runtime_difference"
        final = "model_reproducibility_material_difference_requires_resolution"
    elif not canonical_resolved and runtime_verdict != "runtime_reproducibility_verified_no_material_delta":
        verdict = "retraining_triage_tooling_ready_comparison_pending"
        final = "model_reproducibility_tooling_ready_runtime_comparison_pending"
    elif lineage_verdict == "training_lineage_conflicting":
        verdict = "retraining_triage_blocked_lineage_conflict"
        final = "model_reproducibility_tooling_ready_runtime_comparison_pending"
    elif any(item["primary_action"] in RETRAINING_ACTIONS or item["primary_action"].startswith("quarantine") for item in decisions.values()):
        verdict = "retraining_triage_complete_candidates_required"
        final = "model_reproducibility_verified_retraining_required"
    else:
        verdict = "retraining_triage_complete_incumbents_healthy_shadow_only"
        final = "model_reproducibility_verified_no_retraining_required"
    specification = generate_retraining_specification(
        decisions=decisions, runtime_report=runtime_report, phase22_report=phase22_report,
        lineage_report=lineage_report, policy=policy,
        canonical_runtime_decision=canonical_runtime_decision,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "runtime_reproducibility_digest": runtime_report.get("reproducibility_digest"),
        "phase22_alignment_digest": phase22_report.get("alignment_digest"),
        "failure_triage_digest": failure_report.get("failure_triage_digest"),
        "training_lineage_digest": lineage_report.get("training_lineage_digest"),
        "policy_digest": json_digest(policy),
        "candidate_registry_digest": json_digest(candidate_registry),
        "retraining_policy_validated": True,
        "runtime_comparison_complete": runtime_complete,
        "runtime_stack_isolation_completed": canonical_resolved,
        "runtime_attribution_digest": (
            runtime_attribution_report.get("attribution_digest")
            if runtime_attribution_report else None
        ),
        "canonical_runtime_decision_digest": (
            canonical_runtime_decision.get("decision_digest")
            if canonical_runtime_decision else None
        ),
        "canonical_stack_id": (
            canonical_runtime_decision.get("selected_stack_id")
            if canonical_runtime_decision else None
        ),
        "numerical_reproducibility_status": numerical_status,
        "behavioral_reproducibility_status": behavioral_status,
        "incumbent_artifact_integrity_result": "pass" if artifact_integrity_pass else "failed_or_unverified",
        "model_decisions": decisions,
        "overall_decision": {"verdict": verdict, "final_implementation_verdict": final},
        "phase24_allowed": phase24_allowed,
        "live_or_blocking_use_approved": False,
        "promotion_allowed": False,
        "warnings": [
            "Phase 24 allowance covers versioned candidate work only; it is not live-trading or promotion approval."
        ],
    }
    report["triage_digest"] = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", "triage_digest"}
    })
    return report, specification


def _load_report(path: Path | str, digest_field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RetrainingTriageError(f"report is not an object: {path}")
    _verify_report_digest(value, digest_field)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23 deterministic model retraining triage")
    parser.add_argument("--phase22-report", required=True)
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--failure-report", required=True)
    parser.add_argument("--lineage-report", required=True)
    parser.add_argument("--spec-out", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--candidate-registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--canonical-runtime-decision")
    parser.add_argument("--runtime-attribution-report")
    parser.add_argument("--model", action="append")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        phase22 = _load_report(args.phase22_report, "alignment_digest")
        runtime = _load_report(args.runtime_report, "reproducibility_digest")
        failure = _load_report(args.failure_report, "failure_triage_digest")
        lineage = _load_report(args.lineage_report, "training_lineage_digest")
        canonical = (
            _load_report(args.canonical_runtime_decision, "decision_digest")
            if args.canonical_runtime_decision else None
        )
        attribution = (
            _load_report(args.runtime_attribution_report, "attribution_digest")
            if args.runtime_attribution_report else None
        )
        registry = json.loads(Path(args.candidate_registry).read_text(encoding="utf-8-sig"))
        report, specification = build_retraining_triage_report(
            phase22_report=phase22, runtime_report=runtime, failure_report=failure,
            lineage_report=lineage, policy=load_retraining_policy(args.policy),
            candidate_registry=registry, models=args.model,
            canonical_runtime_decision=canonical,
            runtime_attribution_report=attribution,
        )
        spec_target = ensure_safe_report_output(args.spec_out)
        report_target = ensure_safe_report_output(args.json_out)
        if spec_target == report_target:
            raise RetrainingTriageError("specification and triage report outputs must be distinct")
        spec_target.parent.mkdir(parents=True, exist_ok=True)
        report_target.parent.mkdir(parents=True, exist_ok=True)
        spec_target.write_text(json.dumps(specification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and not report["phase24_allowed"]:
            return 3
        return 0
    except Exception as exc:
        print(f"model_retraining_triage: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
