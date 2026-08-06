"""Evidence-based attribution of Phase 23.1 numerical runtime differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_runtime_repro import ensure_safe_report_output, json_digest, verify_report_digest
from tools.runtime_stack_isolation import INTERACTION_STACKS, PRIMARY_STACKS


DEFAULT_ISOLATION_REPORT = BASE_DIR / "reports" / "runtime_stack_isolation.json"
DEFAULT_REPORT = BASE_DIR / "reports" / "runtime_stack_attribution.json"
PACKAGE_LABELS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "joblib": "joblib",
    "scikit-learn": "scikit-learn",
    "threadpoolctl": "threadpoolctl",
}


class RuntimeStackAttributionError(ValueError):
    pass


def _loaded_report(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    verify_report_digest(value, "isolation_digest")
    return value


def _exact_against_serialized(stack: Mapping[str, Any]) -> bool:
    if stack.get("matches_serialized_full_pattern") is True:
        return True
    comparisons = []
    for model in stack.get("by_model", {}).values():
        for symbol in model.get("by_symbol", {}).values():
            item = symbol.get("vs_serialized_full_stack")
            if isinstance(item, Mapping):
                comparisons.append(item)
    return bool(comparisons) and all(item.get("classification") == "bitwise_exact" for item in comparisons)


def _changed_packages(isolation: Mapping[str, Any], stack_id: str) -> list[str]:
    stack = isolation.get("environment_matrix", {}).get("stacks", {}).get(stack_id, {})
    # Runtime reports deliberately do not duplicate the tracked matrix's
    # explanatory labels.  Stack identifiers are an exact, closed contract.
    mapping = {
        "declared_sklearn_only": ["scikit-learn"],
        "sklearn_only_180": ["scikit-learn"],
        "numpy_only_233": ["numpy"],
        "scipy_only_162": ["scipy"],
        "joblib_only_152": ["joblib"],
        "numpy_233_sklearn_180": ["numpy", "scikit-learn"],
        "scipy_162_sklearn_180": ["scipy", "scikit-learn"],
        "joblib_152_sklearn_180": ["joblib", "scikit-learn"],
        "numpy_233_scipy_162": ["numpy", "scipy"],
        "numpy_233_joblib_152": ["numpy", "joblib"],
    }
    if stack.get("status") not in {"available", "environment_unavailable"}:
        return []
    return mapping.get(stack_id, [])


def _path_digest(item: Mapping[str, Any], path_name: str) -> str | None:
    return item.get("paths", {}).get(path_name, {}).get("output_digest")


def determine_decomposition_stage(isolation: Mapping[str, Any]) -> dict[str, Any]:
    decomposition = isolation.get("transform_decomposition", {})
    observed = decomposition.get("observed_main", {})
    serialized = decomposition.get("serialized_full_stack", {})
    if not observed or set(observed) != set(serialized):
        return {"stage": "unresolved", "evidence": [], "consistent": False}
    stages: list[str] = []
    evidence: list[dict[str, Any]] = []
    for kind in sorted(observed):
        left, right = observed[kind], serialized[kind]
        mean_changed = left.get("scaler_metadata", {}).get("mean_digest") != right.get("scaler_metadata", {}).get("mean_digest")
        scale_changed = left.get("scaler_metadata", {}).get("scale_digest") != right.get("scaler_metadata", {}).get("scale_digest")
        equality = {
            name: _path_digest(left, name) == _path_digest(right, name)
            for name in (
                "sklearn_transform_float64_input",
                "sklearn_transform_float32_input",
                "manual_float64_formula",
                "manual_float64_then_float32",
                "manual_float32_formula",
            )
        }
        if mean_changed or scale_changed:
            stage = "scaler_deserialization"
        elif not equality["manual_float64_formula"]:
            stage = "package_interaction"
        elif not equality["manual_float64_then_float32"]:
            stage = "float64_to_float32_conversion"
        elif not equality["manual_float32_formula"]:
            stage = "float32_arithmetic"
        elif not equality["sklearn_transform_float32_input"]:
            stage = "sklearn_transform_output_handling"
        elif not equality["sklearn_transform_float64_input"]:
            stage = "sklearn_transform_implementation"
        else:
            stage = "no_transform_difference"
        stages.append(stage)
        evidence.append({
            "kind": kind,
            "mean_or_scale_digest_changed": bool(mean_changed or scale_changed),
            "cross_stack_path_exact_equal": equality,
            "stage": stage,
        })
    unique = sorted(set(stages))
    return {
        "stage": unique[0] if len(unique) == 1 else "package_interaction",
        "model_stages": unique,
        "evidence": evidence,
        "consistent": len(unique) == 1,
    }


def attribute_runtime_stack(isolation: Mapping[str, Any]) -> dict[str, Any]:
    if isolation.get("schema_version") != 1:
        raise RuntimeStackAttributionError("unsupported isolation report schema")
    stack_results = isolation.get("stack_comparisons", {})
    environments = isolation.get("environment_matrix", {}).get("stacks", {})
    primary_available = all(
        environments.get(stack_id, {}).get("status") == "available" for stack_id in PRIMARY_STACKS
    )
    single_stack_matches = [
        stack_id for stack_id in PRIMARY_STACKS
        if stack_id not in {"observed_main", "serialized_full_stack"}
        and len(_changed_packages(isolation, stack_id)) == 1
        and _exact_against_serialized(stack_results.get(stack_id, {}))
    ]
    contributors = sorted({
        package for stack_id in single_stack_matches
        for package in _changed_packages(isolation, stack_id)
    })
    interaction_matches = [
        stack_id for stack_id in INTERACTION_STACKS
        if environments.get(stack_id, {}).get("status") == "available"
        and _exact_against_serialized(stack_results.get(stack_id, {}))
    ]
    interactions_requested = isolation.get("overall_decision", {}).get(
        "interaction_stacks_required", []
    )
    interaction_scope_complete = all(
        environments.get(stack_id, {}).get("status") == "available"
        for stack_id in interactions_requested
    )
    decomposition = determine_decomposition_stage(isolation)

    contradicting: list[str] = []
    supporting: list[str] = []
    if len(contributors) == 1:
        primary = contributors[0]
        supporting = single_stack_matches
        secondary: list[str] = []
        interaction_required = False
        confidence = "confirmed" if primary_available else "strongly_supported"
    elif len(contributors) > 1:
        primary = "multiple_contributors"
        secondary = contributors
        supporting = single_stack_matches
        interaction_required = False
        confidence = "strongly_supported" if primary_available else "partially_supported"
    elif interaction_matches:
        matched_sets = [set(_changed_packages(isolation, stack_id)) for stack_id in interaction_matches]
        common = set.intersection(*matched_sets) if matched_sets else set()
        primary = "package_interaction"
        secondary = sorted(common or set.union(*matched_sets))
        supporting = interaction_matches
        interaction_required = True
        confidence = "confirmed" if interaction_scope_complete else "strongly_supported"
    elif primary_available and interaction_scope_complete and interactions_requested:
        primary = "multi_package_interaction"
        secondary = []
        supporting = ["serialized_full_stack"]
        interaction_required = True
        confidence = "strongly_supported"
    else:
        primary = "unresolved"
        secondary = []
        interaction_required = bool(interactions_requested)
        confidence = "unresolved"
        contradicting = [
            stack_id for stack_id in (*PRIMARY_STACKS, *interactions_requested)
            if environments.get(stack_id, {}).get("status") != "available"
        ]

    if decomposition["stage"] == "unresolved":
        confidence = "unresolved"
        if primary not in {"package_interaction", "multi_package_interaction"}:
            primary = "unresolved"
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_isolation_digest": isolation.get("isolation_digest"),
        "primary_contributor": primary,
        "secondary_contributors": secondary,
        "interaction_required": interaction_required,
        "decomposition_stage": decomposition["stage"],
        "decomposition_evidence": decomposition,
        "supporting_stacks": supporting,
        "contradicting_stacks": contradicting,
        "confidence": confidence,
        "primary_environment_scope_complete": primary_available,
        "interaction_environment_scope_complete": interaction_scope_complete,
        "causality_basis": "isolated transform arrays and loaded-scaler manual decomposition",
        "scaler_serialization_version_alone_used_as_causal_evidence": False,
    }
    result["attribution_digest"] = json_digest(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute Phase 23.1 runtime stack deltas")
    parser.add_argument("--isolation-report", default=str(DEFAULT_ISOLATION_REPORT))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = attribute_runtime_stack(_loaded_report(args.isolation_report))
        target = ensure_safe_report_output(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and report["confidence"] == "unresolved":
            return 3
        return 0
    except Exception as exc:
        print(f"runtime_stack_attribution: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
