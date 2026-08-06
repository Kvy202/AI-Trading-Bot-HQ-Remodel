"""Evidence-bounded Phase 23 failure triage for incumbent models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_runtime_repro import (
    DEFAULT_BUNDLE,
    DEFAULT_POLICY,
    MODEL_KINDS,
    ensure_safe_report_output,
    json_digest,
    load_retraining_policy,
    scaled_feature_diagnostics,
    validate_bundle_contract,
    verify_report_digest,
)


FAILURE_CATEGORIES = (
    "scaler_runtime_difference",
    "serving_distribution_ood",
    "learned_classifier_saturation",
    "calibration_saturation",
    "low_dynamic_range",
    "symbol_specific_failure",
    "insufficient_evidence",
    "no_failure_detected",
)
SUPPORT_LEVELS = ("supported", "partially_supported", "not_supported", "unverified")


class FailureTriageError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def calculate_input_diagnostics(values: np.ndarray, feature_names: Sequence[str]) -> dict[str, Any]:
    """Public synthetic-test entry point for scaled feature/OOD diagnostics."""

    return scaled_feature_diagnostics(values, feature_names)


def _assessment(level: str, evidence: Sequence[str]) -> dict[str, Any]:
    if level not in SUPPORT_LEVELS:
        raise FailureTriageError(f"unknown support level: {level}")
    return {"support": level, "evidence": list(evidence)}


def classify_failure(
    *,
    scaler_classification: str,
    collapse: Mapping[str, Any],
    input_diagnostics: Mapping[str, Any] | None,
    classifier_calibration: Mapping[str, Any] | None,
    symbol_failure_pattern: str = "unknown",
    flat_threshold: float = 0.002,
) -> dict[str, Any]:
    """Classify measured evidence without turning correlation into causation."""

    input_diagnostics = dict(input_diagnostics or {})
    classifier_calibration = dict(classifier_calibration or {})
    result: dict[str, Any] = {}
    result["scaler_runtime_difference"] = _assessment(
        "supported" if scaler_classification == "materially_different" else
        "not_supported" if scaler_classification in {"exact_match", "numerically_equivalent"} else "unverified",
        [f"cross-runtime scaler classification={scaler_classification}"],
    )
    ood_count = input_diagnostics.get("ood_feature_count")
    maximum_z = input_diagnostics.get("maximum_absolute_z")
    if ood_count is None or maximum_z is None:
        ood_level = "unverified"
    elif int(ood_count) > 0 and float(maximum_z) > 5:
        ood_level = "supported"
    elif int(ood_count) > 0 or float(maximum_z) > 3:
        ood_level = "partially_supported"
    else:
        ood_level = "not_supported"
    result["serving_distribution_ood"] = _assessment(
        ood_level, [f"ood_feature_count={ood_count}", f"maximum_absolute_z={maximum_z}"]
    )
    raw_std = classifier_calibration.get("raw_std")
    raw_saturation = classifier_calibration.get("saturation_rate")
    before = classifier_calibration.get("exclusion_events_before_calibration")
    after = classifier_calibration.get("exclusion_events_after_calibration")
    clipping = classifier_calibration.get("clipping_count")
    if raw_std is None or raw_saturation is None:
        learned_level = "unverified"
    elif float(raw_std) < flat_threshold or float(raw_saturation) >= 0.5 or int(before or 0) > 0:
        learned_level = "supported"
    elif float(raw_saturation) > 0:
        learned_level = "partially_supported"
    else:
        learned_level = "not_supported"
    result["learned_classifier_saturation"] = _assessment(
        learned_level,
        [f"raw_std={raw_std}", f"raw_saturation_rate={raw_saturation}", f"pre_calibration_exclusions={before}"],
    )
    if before is None or after is None or clipping is None:
        calibration_level = "unverified"
    elif int(after) > int(before) or int(clipping) > 0:
        calibration_level = "supported" if int(after) > int(before) else "partially_supported"
    else:
        calibration_level = "not_supported"
    result["calibration_saturation"] = _assessment(
        calibration_level,
        [f"pre_calibration_exclusions={before}", f"post_calibration_exclusions={after}", f"clipping_count={clipping}"],
    )
    calibrated_std = collapse.get("sklearn180_runtime", collapse).get("calibrated_probability_std")
    flat_events = collapse.get("sklearn180_runtime", collapse).get("rolling_flat_window_count")
    if calibrated_std is None:
        range_level = "unverified"
    elif float(calibrated_std) < flat_threshold or int(flat_events or 0) > 0:
        range_level = "supported"
    else:
        range_level = "not_supported"
    result["low_dynamic_range"] = _assessment(
        range_level, [f"calibrated_std={calibrated_std}", f"flat_endpoint_count={flat_events}"]
    )
    result["symbol_specific_failure"] = _assessment(
        "supported" if symbol_failure_pattern == "mixed" else
        "not_supported" if symbol_failure_pattern in {"all_healthy", "all_failed"} else "unverified",
        [f"required-symbol failure pattern={symbol_failure_pattern}"],
    )
    evidence_missing = any(
        result[name]["support"] == "unverified"
        for name in (
            "scaler_runtime_difference", "serving_distribution_ood",
            "learned_classifier_saturation", "calibration_saturation", "low_dynamic_range",
        )
    )
    result["insufficient_evidence"] = _assessment(
        "supported" if evidence_missing else "not_supported",
        ["one or more required diagnostic families are unavailable"] if evidence_missing else ["required diagnostic families are present"],
    )
    failed = str(collapse.get("sklearn180_runtime", collapse).get("collapse_status")) == "failed_health_gate"
    positive_failure = any(
        result[name]["support"] in {"supported", "partially_supported"}
        for name in FAILURE_CATEGORIES[:5]
    )
    result["no_failure_detected"] = _assessment(
        "supported" if not failed and not positive_failure else "not_supported",
        [f"sklearn180 health failed={failed}", f"measured failure evidence={positive_failure}"],
    )
    return result


def _load_report(path: Path | str, digest_field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FailureTriageError(f"report is not an object: {path}")
    recorded = value.get(digest_field)
    observed = json_digest({
        key: item for key, item in value.items() if key not in {"generated_at", digest_field}
    })
    if recorded != observed:
        raise FailureTriageError(f"report digest mismatch: {Path(path).name}")
    return value


def build_failure_triage_report(
    *,
    bundle: Path | str,
    runtime_report: Mapping[str, Any],
    phase22_report: Mapping[str, Any],
    policy: Mapping[str, Any],
    models: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    verify_report_digest(runtime_report, "reproducibility_digest")
    verify_report_digest(phase22_report, "alignment_digest")
    manifest, snapshot, _ = validate_bundle_contract(bundle, policy)
    if runtime_report.get("input_bundle", {}).get("bundle_digest") != manifest.get("bundle_digest"):
        raise FailureTriageError("runtime report does not match the immutable Phase 22 bundle")
    if phase22_report.get("historical_alignment", {}).get("bundle_digest") != manifest.get("bundle_digest"):
        raise FailureTriageError("Phase 22 report does not match the immutable bundle")
    evidence = runtime_report.get("analysis_evidence", {})
    scaler = runtime_report.get("scaler_comparisons", {})
    output_comparisons = runtime_report.get("model_output_comparisons", {})
    collapse = runtime_report.get("collapse_comparisons", {})
    required_symbols = list(policy["required_serving_symbols"])
    selected_models = [kind for kind in sorted(collapse) if not models or kind in models]
    selected_symbols = [symbol for symbol in required_symbols if not symbols or symbol in symbols]
    model_results: dict[str, Any] = {}
    for kind in selected_models:
        statuses = {
            symbol: collapse.get(kind, {}).get("by_symbol", {}).get(symbol, {})
            .get("sklearn180_runtime", {}).get("collapse_status")
            for symbol in required_symbols
        }
        failed_count = sum(status == "failed_health_gate" for status in statuses.values())
        pattern = (
            "mixed" if 0 < failed_count < len(required_symbols) else
            "all_failed" if failed_count == len(required_symbols) else
            "all_healthy" if failed_count == 0 and all(statuses.values()) else "unknown"
        )
        model_results[kind] = {"by_symbol": {}}
        for symbol in selected_symbols:
            scaler_result = scaler.get(kind, {}).get("by_symbol", {}).get(symbol, {})
            collapse_result = collapse.get(kind, {}).get("by_symbol", {}).get(symbol, {})
            diagnostic = evidence.get(kind, {}).get("by_symbol", {}).get(symbol, {})
            assessments = classify_failure(
                scaler_classification=str(scaler_result.get("classification") or "comparison_failed"),
                collapse=collapse_result,
                input_diagnostics=diagnostic.get("input_scaler"),
                classifier_calibration=diagnostic.get("classifier_calibration"),
                symbol_failure_pattern=pattern,
                flat_threshold=float(policy["flat_output_std_threshold"]),
            )
            supported = [
                name for name, item in assessments.items()
                if item["support"] in {"supported", "partially_supported"}
            ]
            model_results[kind]["by_symbol"][symbol] = {
                **diagnostic.get("input_scaler", {}),
                **diagnostic.get("classifier_calibration", {}),
                **diagnostic.get("sensitivity", {}),
                "runtime_scaler_comparison": scaler_result,
                "runtime_model_output_comparison": output_comparisons.get(kind, {}).get(
                    "by_symbol", {}
                ).get(symbol, {}),
                "collapse_comparison": collapse_result,
                "failure_categories": assessments,
                "supported_or_partial_categories": supported,
                "immutable_bundle_only": True,
            }
        model_results[kind]["symbol_failure_pattern"] = pattern
    full_scope = (
        set(selected_models) == set(MODEL_KINDS)
        and set(selected_symbols) == set(required_symbols)
        and all(
            set(result.get("by_symbol", {})) == set(required_symbols)
            for result in model_results.values()
        )
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy": {"offline_only": True, "orders_allowed": False, "causal_claims_require_measured_support": True},
        "input_bundle_digest": manifest["bundle_digest"],
        "runtime_reproducibility_digest": runtime_report.get("reproducibility_digest"),
        "phase22_alignment_digest": phase22_report.get("alignment_digest"),
        "serving_snapshot_digest": snapshot.get("snapshot_digest"),
        "model_results": model_results,
        "overall_decision": {
            "verdict": (
                "failure_triage_complete"
                if model_results and full_scope else "failure_triage_insufficient_evidence"
            ),
            "full_required_scope": full_scope,
            "profitability_evidence": False,
            "live_approval": False,
        },
        "warnings": [
            "Failure categories are evidence assessments, not causal proof; unmeasured causes remain unverified."
        ],
    }
    report["failure_triage_digest"] = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", "failure_triage_digest"}
    })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23 model failure triage")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--phase22-report", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--model", action="append")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime = _load_report(args.runtime_report, "reproducibility_digest")
        phase22 = _load_report(args.phase22_report, "alignment_digest")
        report = build_failure_triage_report(
            bundle=args.bundle, runtime_report=runtime, phase22_report=phase22,
            policy=load_retraining_policy(args.policy), models=args.model, symbols=args.symbol,
        )
        target = ensure_safe_report_output(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and report["overall_decision"]["verdict"] != "failure_triage_complete":
            return 3
        return 0
    except Exception as exc:
        print(f"model_failure_triage: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
