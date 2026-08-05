"""Combine Phase 22 historical and live evidence into a research-only verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from runtime.model_serving_guard import guard_from_snapshot
from tools.model_alignment_shadow import (
    DEFAULT_POLICY,
    ModelAlignmentError,
    load_alignment_policy,
    load_historical_bundle,
)
from tools.model_serving_snapshot import capture_model_serving_snapshot


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path | str | None) -> Optional[dict[str, Any]]:
    if not path:
        return None
    source = Path(path)
    if source.is_dir():
        source = source / "final_report.json"
    if not source.is_file():
        return None
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ModelAlignmentError(f"report input must be an object: {source}")
    return value


def library_compatibility(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    scalers: dict[str, Any] = {}
    statuses: list[str] = []
    remediations: list[str] = []
    for entry in snapshot.get("model_entries", []):
        kind = str(entry.get("kind"))
        serialized = entry.get("scaler_serialized_sklearn_version")
        runtime = entry.get("scaler_runtime_sklearn_version") or snapshot.get("sklearn_runtime_version")
        status = entry.get("sklearn_version_status", "serialized_version_unknown")
        statuses.append(str(status))
        if status == "loadable_version_mismatch":
            remediations.append(
                f"{kind}: expected scikit-learn {serialized}; current scikit-learn {runtime}"
            )
        scalers[kind] = {
            "scaler_serialized_sklearn_version": serialized,
            "scaler_runtime_sklearn_version": runtime,
            "sklearn_version_status": status,
            "serialization_warnings": entry.get("scaler_serialization_warnings", []),
        }
    exact = bool(statuses) and all(status == "exact_match" for status in statuses)
    return {
        "python_version": snapshot.get("python_version"),
        "numpy_version": snapshot.get("numpy_version"),
        "torch_version": snapshot.get("torch_version"),
        "joblib_version": snapshot.get("joblib_version"),
        "sklearn_runtime_version": snapshot.get("sklearn_runtime_version"),
        "scalers": scalers,
        "exact_reproducibility": exact,
        "runtime_remediation_required": remediations,
        "dependencies_changed": False,
    }


def _phase21_model_stats(
    phase21: Optional[Mapping[str, Any]], kind: str, symbol: Optional[str] = None
) -> Optional[dict[str, Any]]:
    if not phase21:
        return None
    value = phase21.get("historical_output_analysis", {}).get("models", {}).get(kind)
    if not isinstance(value, Mapping):
        return None
    result = dict(value)
    if symbol:
        by_symbol = value.get("by_symbol", {})
        selected = by_symbol.get(symbol) or by_symbol.get("__pooled__")
        if isinstance(selected, Mapping):
            result["comparison_standard_deviation"] = selected.get("standard_deviation")
            result["comparison_scope"] = symbol if symbol in by_symbol else "__pooled__"
    result.setdefault("comparison_standard_deviation", value.get("pooled_standard_deviation"))
    return result


def _model_fail(status: Any) -> bool:
    return str(status or "").startswith("failed_")


def build_alignment_report(
    *,
    historical_bundle: Path | str | None,
    historical_evaluation: Path | str | None,
    policy: Mapping[str, Any],
    phase21_report: Path | str | None = None,
    live_campaign: Path | str | None = None,
) -> dict[str, Any]:
    evaluation = _load_json(historical_evaluation)
    phase21 = _load_json(phase21_report)
    live = _load_json(live_campaign)
    manifest = snapshot = records = None
    if historical_bundle:
        manifest, snapshot, records = load_historical_bundle(historical_bundle)
    elif evaluation:
        raise ModelAlignmentError("historical bundle is required with an evaluation")

    if snapshot is None:
        snapshot = capture_model_serving_snapshot(
            identity="phase22_alignment_report_inventory",
            mode="model_alignment_report",
            forced_env_overrides={
                "LIVE_TRADING": "false", "PAPER_TRADING": "true",
                "LIVE_MODE": "false", "EXEC_PAPER": "true",
                "PLACE_REAL_ORDERS": "false",
            },
        )
    guard = guard_from_snapshot(snapshot)
    compatibility = library_compatibility(snapshot)
    model_results: dict[str, Any] = {}
    eval_models = evaluation.get("model_results", {}) if evaluation else {}
    kinds = sorted({
        str(entry.get("kind")) for entry in snapshot.get("model_entries", [])
        if entry.get("kind")
    })
    symbols = list(manifest.get("symbols", [])) if manifest else list(snapshot.get("dl_symbols", []))
    current_weights = snapshot.get("dl_model_weights", {})
    any_critical = False
    for kind in kinds:
        by_symbol: dict[str, Any] = {}
        for symbol in symbols:
            phase1 = _phase21_model_stats(phase21, kind, symbol)
            phase1_std = phase1.get("comparison_standard_deviation") if phase1 else None
            phase1_collapsed = (
                None if phase1_std is None
                else float(phase1_std) < float(policy["flat_output_std_threshold"])
            )
            stats = eval_models.get(symbol, {}).get(kind) if evaluation else None
            if not stats:
                by_symbol[symbol] = {
                    "status": "unverified", "did_1m_collapse": phase1_collapsed,
                    "did_5m_collapse": None, "artifact_eligible_for_later_5m_shadow_campaign": False,
                }
                any_critical = True
                continue
            status = stats.get("model_health_status", "unverified")
            collapsed_5m = status in {
                "failed_flat_at_5m", "failed_extreme_collapse_at_5m"
            }
            repeat_error = stats.get("deterministic_repeat_max_error")
            missing_rate = stats.get("missing_rate")
            five_minute_std = stats.get("5m_historical_std")
            deterministic = repeat_error is not None and float(repeat_error) <= float(
                policy["deterministic_repeat_tolerance"]
            )
            missing_ok = missing_rate is not None and float(missing_rate) <= float(
                policy["maximum_missing_rate"]
            )
            variable = five_minute_std is not None and float(five_minute_std) >= float(
                policy["flat_output_std_threshold"]
            )
            gate = int(stats.get("unique_completed_bar_count") or 0) >= int(
                policy["historical_unique_bars_required"]
            )
            critical = _model_fail(status) or not deterministic or not missing_ok or not gate
            any_critical = any_critical or critical
            by_symbol[symbol] = {
                **stats,
                "did_1m_collapse": phase1_collapsed,
                "did_5m_collapse": collapsed_5m,
                "is_5m_output_sufficiently_variable": variable,
                "is_output_deterministic": deterministic,
                "is_missing_rate_acceptable": missing_ok,
                "did_simulated_production_exclusion_occur": int(stats.get("simulated_exclusion_count") or 0) > 0,
                "artifact_eligible_for_later_5m_shadow_campaign": not critical,
                "1m_phase21_comparison_available": phase1 is not None,
                "1m_historical_std": phase1_std,
                "5m_historical_std": stats.get("5m_historical_std"),
                "collapse_resolved_at_5m": bool(phase1_collapsed and not collapsed_5m) if phase1_collapsed is not None else None,
                "collapse_persists_at_5m": bool(phase1_collapsed and collapsed_5m) if phase1_collapsed is not None else None,
                "current_positive_weight_depends_on_model": float(
                    current_weights.get(kind, 0.0) or 0.0
                ) > 0,
            }
        model_results[kind] = {"by_symbol": by_symbol}

    evaluation_matches_bundle = bool(
        evaluation and manifest and snapshot
        and evaluation.get("bundle_digest", manifest.get("bundle_digest")) == manifest.get("bundle_digest")
        and evaluation.get("serving_snapshot_digest", snapshot.get("snapshot_digest"))
        == snapshot.get("snapshot_digest")
    )
    historical_gate = bool(
        evaluation
        and evaluation.get("minimum_statistical_gate_passed")
        and manifest
        and int(manifest.get("conflicting_source_bar_count", 0)) == 0
        and evaluation_matches_bundle
        and all(
            int(manifest.get("unique_completed_bars_by_symbol", {}).get(symbol, 0))
            >= int(policy["historical_unique_bars_required"])
            for symbol in symbols
        )
    )
    source_conflict = bool(manifest and int(manifest.get("conflicting_source_bar_count", 0)) > 0)
    live_counts = live.get("unique_completed_bars_by_symbol", {}) if live else {}
    live_snapshot_digest = live.get("snapshot_digest") if live else None
    live_snapshot_file_ok = True
    if live_campaign and Path(live_campaign).is_dir():
        live_snapshot_path = Path(live_campaign) / "model_serving_snapshot.json"
        live_snapshot_file_ok = live_snapshot_path.is_file()
    live_pass = bool(
        live and live.get("status") == "pass"
        and set(live_counts) == set(symbols)
        and all(int(live_counts[symbol]) >= int(policy["live_unique_bars_required"]) for symbol in symbols)
        and live.get("timeframe") == policy["required_timeframe"]
        and int(live.get("sequence_length", 0)) == int(policy["required_sequence_length"])
        and int(live.get("served_feature_width", 0)) == int(policy["required_served_feature_count"])
        and live.get("contract_status") == "pass"
        and live_snapshot_file_ok
        and live_snapshot_digest == snapshot.get("snapshot_digest")
        and live.get("all_source_bars_completed") is True
        and live.get("all_source_bar_ids_unique") is True
        and int(live.get("conflicting_source_bar_count", 0)) == 0
        and int(live.get("nonfinite_output_count", 0)) == 0
        and int(live.get("nondeterministic_output_count", 0)) == 0
        and not live.get("executor_started") and not live.get("writer_started")
        and int(live.get("orders_placed", 0)) == 0
    )

    if source_conflict:
        verdict, readiness = "alignment_failed_source_conflict", "blocked_source_conflict"
    elif guard["status"] != "pass":
        verdict, readiness = "alignment_failed_contract", "blocked_contract_mismatch"
    elif not historical_gate:
        verdict, readiness = "alignment_tooling_ready_collection_pending", "insufficient_unique_bars"
    elif any_critical:
        verdict, readiness = "alignment_failed_model_collapse", "blocked_model_collapse"
    elif not live_pass:
        verdict, readiness = "alignment_unverified", "live_smoke_pending"
    elif not compatibility["exact_reproducibility"]:
        verdict = "alignment_validated_runtime_pin_pending"
        readiness = "ready_for_5m_shadow_campaign_runtime_pin_pending"
    else:
        verdict = "alignment_validated_shadow_only"
        readiness = "ready_for_5m_replay_enabled_shadow_campaign"

    warnings: list[str] = []
    if not compatibility["exact_reproducibility"]:
        warnings.append("exact_sklearn_reproducibility_blocked")
    if not live_pass:
        warnings.append("live_completed_bar_smoke_pending_or_failed")
    if phase21:
        warnings.append("1m_and_5m_samples_are_model_health_only_not_profitability_comparisons")
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "policy": {
            "research_only": True, "paper_only": True, "orders_allowed": False,
            "configuration_changed": False, "profitability_claim_allowed": False,
        },
        "serving_contract": guard,
        "historical_alignment": {
            "status": "pass" if historical_gate else "pending_or_failed",
            "bundle_digest": manifest.get("bundle_digest") if manifest else None,
            "unique_completed_bars_by_symbol": (
                manifest.get("unique_completed_bars_by_symbol", {}) if manifest else {}
            ),
            "statistical_gate_passed": historical_gate,
            "evaluation_matches_bundle": evaluation_matches_bundle,
            "conflicting_source_bar_count": (
                manifest.get("conflicting_source_bar_count", 0) if manifest else None
            ),
            "profitability_evidence": False,
        },
        "live_integration_smoke": {
            "status": "pass" if live_pass else "pending",
            "integration_evidence_only": True,
            "statistical_gate_satisfied": False,
            "campaign": live,
        },
        "library_compatibility": compatibility,
        "model_results": model_results,
        "ensemble_variants": evaluation.get("ensemble_variants", {}) if evaluation else {},
        "overall_decision": {
            "verdict": verdict, "campaign_readiness": readiness,
            "production_activation_allowed": False,
            "phase17_or_phase18_promotion_evidence": False,
        },
        "warnings": warnings,
    }
    result["alignment_digest"] = _digest({
        key: result[key] for key in result if key not in {"generated_at", "alignment_digest"}
    })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Phase 22 model alignment report.")
    parser.add_argument("--historical-bundle")
    parser.add_argument("--historical-evaluation")
    parser.add_argument("--phase21-report")
    parser.add_argument("--live-shadow-campaign")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--json-out", default=str(BASE_DIR / "reports" / "model_alignment_report.json"))
    parser.add_argument("--compatibility-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.compatibility_only:
            snapshot = capture_model_serving_snapshot(
                identity="phase22_compatibility_only", mode="compatibility_only",
                forced_env_overrides={
                    "LIVE_TRADING": "false", "PAPER_TRADING": "true",
                    "LIVE_MODE": "false", "EXEC_PAPER": "true",
                    "PLACE_REAL_ORDERS": "false",
                },
            )
            result: dict[str, Any] = library_compatibility(snapshot)
        else:
            result = build_alignment_report(
                historical_bundle=args.historical_bundle,
                historical_evaluation=args.historical_evaluation,
                phase21_report=args.phase21_report,
                live_campaign=args.live_shadow_campaign,
                policy=load_alignment_policy(args.policy),
            )
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except (ModelAlignmentError, OSError, ValueError) as exc:
        print(f"model_alignment_report_error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
