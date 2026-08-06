"""Select and lock the dedicated canonical numerical stack for Phase 24."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_runtime_repro import ensure_safe_report_output, json_digest, verify_report_digest
from tools.runtime_stack_isolation import (
    DEFAULT_POLICY,
    PACKAGE_KEYS,
    load_isolation_policy,
)


DEFAULT_ISOLATION_REPORT = BASE_DIR / "reports" / "runtime_stack_isolation.json"
DEFAULT_ATTRIBUTION_REPORT = BASE_DIR / "reports" / "runtime_stack_attribution.json"
DEFAULT_REPORT = BASE_DIR / "reports" / "runtime_stack_decision.json"
DEFAULT_CANONICAL = BASE_DIR / "research" / "canonical_model_runtime.json"
DEFAULT_LOCK = BASE_DIR / "requirements" / "model_numeric_canonical.txt"
SERIALIZED_STACK = "serialized_full_stack"


class RuntimeStackDecisionError(ValueError):
    pass


def normalized_lock_text(package_versions: Mapping[str, str]) -> str:
    if set(package_versions) != set(PACKAGE_KEYS):
        raise RuntimeStackDecisionError("canonical stack package inventory is incomplete")
    return "".join(f"{name}=={package_versions[name]}\n" for name in PACKAGE_KEYS)


def normalized_lock_digest(package_versions: Mapping[str, str]) -> str:
    return hashlib.sha256(normalized_lock_text(package_versions).encode("utf-8")).hexdigest()


def _load_report(path: Path | str, digest_field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    verify_report_digest(value, digest_field)
    return value


def _stack_has_zero_behavior_changes(isolation: Mapping[str, Any], stack_id: str) -> bool:
    comparison = isolation.get("model_output_comparisons", {}).get(stack_id, {})
    results = [
        symbol
        for model in comparison.get("by_model", {}).values()
        for symbol in model.get("by_symbol", {}).values()
    ]
    fields = (
        "changed_raw_direction_count", "changed_calibrated_direction_count",
        "changed_extreme_state_count", "changed_flat_window_count",
        "changed_exclusion_event_count", "changed_excluded_endpoint_count",
        "changed_allow_count", "changed_signal_direction_count",
        "changed_agreement_suppression_count", "changed_ensemble_variant_count",
    )
    return bool(results) and all(
        item.get("deterministic_repeat_status") == "deterministic"
        and all(int(item.get(field, -1)) == 0 for field in fields)
        for item in results
    )


def select_canonical_runtime(
    isolation: Mapping[str, Any], attribution: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    if isolation.get("schema_version") != 1 or attribution.get("schema_version") != 1:
        raise RuntimeStackDecisionError("unsupported source report schema")
    if attribution.get("source_isolation_digest") != isolation.get("isolation_digest"):
        raise RuntimeStackDecisionError("attribution does not reference the supplied isolation report")
    environments = isolation.get("environment_matrix", {}).get("stacks", {})
    levels = isolation.get("reproducibility_levels", {})
    selected_env = environments.get(SERIALIZED_STACK, {})
    selected_level = levels.get(SERIALIZED_STACK, {})
    selected_available = selected_env.get("status") == "available"
    deterministic = bool(
        selected_level.get("deterministic_workers") is True
        and selected_level.get("deterministic_model_inference") is True
    )
    zero_behavior = (
        selected_level.get("behavioral_status") == "behaviorally_reproducible"
        and _stack_has_zero_behavior_changes(isolation, SERIALIZED_STACK)
    )
    attribution_complete = bool(
        attribution.get("primary_contributor") != "unresolved"
        and attribution.get("confidence") != "unresolved"
        and attribution.get("primary_environment_scope_complete", True) is True
        and (
            attribution.get("interaction_required") is not True
            or attribution.get("interaction_environment_scope_complete", True) is True
        )
    )
    package_versions = selected_env.get("package_versions", {}) if selected_available else {}
    serialization_match = package_versions.get("scikit-learn") == "1.8.0"
    current_versions = environments.get("observed_main", {}).get("package_versions", {})
    current_is_canonical = bool(
        selected_available and current_versions == package_versions
        and isolation.get("environment_matrix", {}).get("stacks", {}).get("observed_main", {}).get("python_version", "").startswith(
            str(policy["required_python_major_minor"]) + "."
        )
    )

    if not selected_available or not serialization_match:
        status = "canonical_stack_selection_blocked_environment"
        selected_id: str | None = None
        reasons = ["serialized_full_stack is unavailable or does not use scikit-learn 1.8.0"]
    elif not deterministic:
        status = "canonical_stack_selection_blocked_environment"
        selected_id = None
        reasons = ["serialized_full_stack did not pass deterministic worker and model inference checks"]
    elif not zero_behavior:
        status = "canonical_stack_selection_blocked_behavior_difference"
        selected_id = None
        reasons = ["serialized_full_stack changed one or more model or ensemble decisions"]
    elif not attribution_complete:
        status = "canonical_stack_selection_pending"
        selected_id = None
        reasons = ["runtime attribution remains unresolved"]
    else:
        status = "canonical_stack_selected"
        selected_id = SERIALIZED_STACK
        reasons = [
            "stack matches the incumbent scaler serialization version",
            "all incumbent scalers loaded and all worker repeats were deterministic",
            "main-runtime PyTorch inference repeated deterministically",
            "all required behavioral decision-change counts were zero",
            "the exact numerical package inventory is fully pinned",
        ]

    phase24_allowed = bool(
        status == "canonical_stack_selected"
        and policy["allow_phase24_in_dedicated_canonical_environment"] is True
        and isolation.get("input_bundle", {}).get("integrity_result") == "pass"
        and isolation.get("input_bundle", {}).get("feature_window_digest_result") == "pass"
        and isolation.get("policy", {}).get("models_modified") is False
        and isolation.get("policy", {}).get("main_environment_modified") is False
        and zero_behavior and deterministic
    )
    if status == "canonical_stack_selected":
        final_verdict = "runtime_stack_isolated_canonical_training_stack_selected"
    elif status == "canonical_stack_selection_blocked_behavior_difference":
        final_verdict = "runtime_stack_isolation_material_behavior_difference"
    elif status == "canonical_stack_selection_blocked_environment":
        final_verdict = "runtime_stack_isolation_environment_pending"
    else:
        final_verdict = "runtime_stack_isolation_attribution_unresolved"

    result: dict[str, Any] = {
        "schema_version": 1,
        "decision_status": status,
        "selected_stack_id": selected_id,
        "python_major_minor": policy["required_python_major_minor"],
        "package_versions": package_versions if selected_id else {},
        "source_bundle_digest": isolation.get("input_bundle", {}).get("bundle_digest"),
        "source_alignment_digest": isolation.get("input_bundle", {}).get("alignment_digest"),
        "source_isolation_digest": isolation.get("isolation_digest"),
        "attribution_digest": attribution.get("attribution_digest"),
        "bitwise_status": selected_level.get("bitwise_status", "unverified"),
        "numerical_status": selected_level.get("numerical_status", "unverified"),
        "behavioral_status": selected_level.get("behavioral_status", "unverified"),
        "deterministic_workers": selected_level.get("deterministic_workers") is True,
        "deterministic_model_inference": selected_level.get("deterministic_model_inference") is True,
        "all_behavior_change_counts_zero": zero_behavior,
        "current_main_runtime_is_canonical": current_is_canonical if selected_id else False,
        "main_runtime_migration_allowed": False,
        "phase24_candidate_training_allowed": phase24_allowed,
        "phase24_environment_scope": (
            ".venv-runtime-isolation/serialized_full_stack" if phase24_allowed else None
        ),
        "live_activation_allowed": False,
        "canonical_lock_digest": (
            normalized_lock_digest(package_versions) if selected_id else None
        ),
        "decision_reasons": reasons,
        "final_implementation_verdict": final_verdict,
    }
    result["decision_digest"] = json_digest(result)
    return result


def write_canonical_outputs(
    decision: Mapping[str, Any], *, canonical_out: Path | str, lock_out: Path | str
) -> None:
    if decision.get("decision_status") != "canonical_stack_selected":
        raise RuntimeStackDecisionError("canonical lock cannot be written before a stack is selected")
    canonical = Path(canonical_out).resolve()
    lock = Path(lock_out).resolve()
    allowed_canonical = (BASE_DIR / "research" / "canonical_model_runtime.json").resolve()
    allowed_lock = (BASE_DIR / "requirements" / "model_numeric_canonical.txt").resolve()
    if canonical.is_relative_to(BASE_DIR.resolve()) and canonical != allowed_canonical:
        raise RuntimeStackDecisionError("canonical decision must use its tracked repository path")
    if lock.is_relative_to(BASE_DIR.resolve()) and lock != allowed_lock:
        raise RuntimeStackDecisionError("canonical lock must use its tracked repository path")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    text = normalized_lock_text(decision["package_versions"])
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != decision["canonical_lock_digest"]:
        raise RuntimeStackDecisionError("canonical lock digest mismatch")
    canonical.write_text(json.dumps(dict(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lock.write_text(text, encoding="utf-8", newline="\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the Phase 24 canonical numerical runtime")
    parser.add_argument("--isolation-report", default=str(DEFAULT_ISOLATION_REPORT))
    parser.add_argument("--attribution-report", default=str(DEFAULT_ATTRIBUTION_REPORT))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--canonical-out", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--lock-out", default=str(DEFAULT_LOCK))
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        isolation = _load_report(args.isolation_report, "isolation_digest")
        attribution = _load_report(args.attribution_report, "attribution_digest")
        decision = select_canonical_runtime(
            isolation, attribution, load_isolation_policy(args.policy)
        )
        target = ensure_safe_report_output(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if decision["decision_status"] == "canonical_stack_selected":
            write_canonical_outputs(
                decision, canonical_out=args.canonical_out, lock_out=args.lock_out
            )
        if args.strict and decision["decision_status"] != "canonical_stack_selected":
            return 3
        return 0
    except Exception as exc:
        print(f"runtime_stack_decision: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
