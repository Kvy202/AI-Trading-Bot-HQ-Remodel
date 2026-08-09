"""Run Phase 24 legacy-repair or sealed-confirmation health gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_evaluate import (
    ModelCandidateEvaluationError,
    _load_candidate,
    _load_incumbent,
    infer_raw_probabilities,
    load_legacy_repair_windows,
    probability_health_statistics,
)
from tools.model_candidate_train import (
    SELECTION_FREEZE,
    TRAINING_SUMMARY,
    ModelCandidateTrainingError,
    _manifest_digest,
    refresh_artifact_manifest,
)
from tools.model_training_dataset import verify_confirmation
from tools.model_training_environment import (
    PHASE22_BUNDLE,
    EXPECTED_PHASE22_BUNDLE_DIGEST,
    atomic_write_json,
    file_digest,
    json_digest,
    load_training_policy,
    TRAINING_PYTHON,
    verify_incumbent_inventory,
    utc_now,
    validate_phase24_evidence,
)


CONFIRMATION_ACCESS_LEDGER = BASE_DIR / "reports" / "model_candidate_confirmation_access.json"


class ModelCandidateHealthGateError(ValueError):
    """A health-gate prerequisite, integrity rule, or access rule failed."""


def _critical_reasons(stats: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(stats.get("extreme_exclusion_events", 0)):
        reasons.append("extreme_collapse_events")
    if int(stats.get("flat_exclusion_events", 0)):
        reasons.append("flat_output_events")
    if float(stats.get("missing_rate", 1.0)) > 0.05:
        reasons.append("missing_rate")
    if int(stats.get("nonfinite_outputs", 1)):
        reasons.append("nonfinite_outputs")
    if stats.get("deterministic_repeat_passed") is not True:
        reasons.append("deterministic_repeat")
    return reasons


def gate_acceptance(
    kind: str,
    per_symbol: Mapping[str, Mapping[str, Any]],
    *,
    gate: str,
    incumbent_per_symbol: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply unchanged thresholds and model-specific repair/protection scope."""
    if kind not in {"lstm", "tcn", "tx"}:
        raise ModelCandidateHealthGateError("ADV may not enter a Phase 24 candidate gate")
    if set(per_symbol) != {"BTCUSDT", "ETHUSDT"}:
        raise ModelCandidateHealthGateError("both required symbols must be evaluated")
    target_map = {
        "lstm": {"repair_targets": ["BTCUSDT", "ETHUSDT"], "regression_protection_targets": []},
        "tcn": {"repair_targets": ["BTCUSDT"], "regression_protection_targets": ["ETHUSDT"]},
        "tx": {"repair_targets": ["ETHUSDT"], "regression_protection_targets": ["BTCUSDT"]},
    }
    scope = target_map[kind]
    reasons = {
        symbol: _critical_reasons(per_symbol[symbol]) for symbol in ("BTCUSDT", "ETHUSDT")
    }
    regressions: list[str] = []
    if incumbent_per_symbol:
        for symbol in scope["regression_protection_targets"]:
            incumbent_healthy = not _critical_reasons(incumbent_per_symbol[symbol])
            candidate_failed = bool(reasons[symbol])
            if incumbent_healthy and candidate_failed:
                regressions.append(symbol)
    passed = not any(reasons.values()) and not regressions
    if gate == "legacy-repair":
        status = "legacy_repair_passed" if passed else "legacy_repair_failed"
    elif gate == "confirmation":
        status = "confirmation_health_passed" if passed else "confirmation_health_failed"
    else:
        raise ModelCandidateHealthGateError("unknown health gate")
    return {
        "status": status, "passed": passed, "model_kind": kind,
        "repair_targets": scope["repair_targets"],
        "regression_protection_targets": scope["regression_protection_targets"],
        "per_symbol_failure_reasons": reasons, "healthy_symbol_regressions": regressions,
        "thresholds_weakened": False,
    }


def _load_confirmation_windows(directory: Path | str, *, freeze_path: Path | str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(directory)
    manifest = verify_confirmation(root, freeze_path=freeze_path)
    result: dict[str, dict[str, Any]] = {}
    for symbol in manifest["symbols"]:
        with np.load(root / f"windows_{symbol}.npz", allow_pickle=False) as values:
            windows = np.asarray(values["windows"], dtype=np.float32)
            endpoints = np.asarray(values["endpoint_timestamps"], dtype=np.int64)
        result[symbol] = {
            "windows": windows,
            "source_bar_ids": [f"{symbol}:{int(value)}" for value in endpoints],
            "endpoint_timestamps": endpoints,
        }
    return manifest, result


def _load_freeze(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    digest = json_digest({key: item for key, item in value.items() if key not in {"frozen_at", "freeze_digest"}})
    if value.get("selection_frozen") is not True or value.get("freeze_digest") != digest:
        raise ModelCandidateHealthGateError("candidate-selection freeze manifest invalid")
    return value


def record_confirmation_access(
    candidate_metadata: Mapping[str, Any],
    confirmation_manifest: Mapping[str, Any],
    *,
    freeze_path: Path | str = SELECTION_FREEZE,
    ledger_path: Path | str = CONFIRMATION_ACCESS_LEDGER,
) -> dict[str, Any]:
    freeze = _load_freeze(freeze_path)
    kind = str(candidate_metadata["model_kind"])
    frozen = freeze.get("candidates", {}).get(kind, {})
    expected = {
        "candidate_id": candidate_metadata["candidate_id"],
        "candidate_model_digest": candidate_metadata["model_sha256"],
        "candidate_scaler_digest": candidate_metadata["scaler_sha256"],
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ModelCandidateHealthGateError("confirmation_not_pristine_for_candidate")
    target = Path(ledger_path)
    ledger = {"schema_version": 1, "accesses": []}
    if target.is_file():
        ledger = json.loads(target.read_text(encoding="utf-8-sig"))
    confirmation_digest = str(confirmation_manifest["confirmation_digest"])
    matching = next((
        row for row in ledger.get("accesses", [])
        if row.get("candidate_id") == expected["candidate_id"]
    ), None)
    if matching is None:
        row = {
            **expected, "confirmation_digest": confirmation_digest,
            "first_access_at": utc_now(), "access_count": 1,
            "last_access_type": "first_evaluation",
        }
        ledger.setdefault("accesses", []).append(row)
        access_type = "first_evaluation"
    else:
        immutable = {**expected, "confirmation_digest": confirmation_digest}
        if any(matching.get(key) != value for key, value in immutable.items()):
            raise ModelCandidateHealthGateError("confirmation access digest changed")
        matching["access_count"] = int(matching.get("access_count", 0)) + 1
        matching["last_access_type"] = "deterministic_replay"
        row = matching
        access_type = "deterministic_replay"
    ledger["ledger_digest"] = json_digest({key: value for key, value in ledger.items() if key != "ledger_digest"})
    atomic_write_json(target, ledger)
    return {"access_type": access_type, **row}


def _inference_health(
    candidate_model: Any,
    candidate_scaler: Any,
    windows: Mapping[str, Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stats: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for symbol, values in sorted(windows.items()):
        first = infer_raw_probabilities(candidate_model, candidate_scaler, values["windows"])
        second = infer_raw_probabilities(candidate_model, candidate_scaler, values["windows"])
        error = float(np.max(np.abs(first - second))) if len(first) else 0.0
        predictions[symbol] = first
        stats[symbol] = probability_health_statistics(
            first, values["source_bar_ids"], policy=policy,
            expected_count=len(values["source_bar_ids"]), deterministic_repeat_error=error,
        )
    return stats, predictions


def _incumbent_health(kind: str, windows: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    _, model, scaler = _load_incumbent(kind)
    result: dict[str, Any] = {}
    for symbol, values in sorted(windows.items()):
        probabilities = infer_raw_probabilities(model, scaler, values["windows"])
        result[symbol] = probability_health_statistics(
            probabilities, values["source_bar_ids"], policy=policy,
            expected_count=len(values["source_bar_ids"]),
        )
    return result


def _write_gate_once(candidate: Path, filename: str, result: Mapping[str, Any]) -> str:
    path = candidate / filename
    previous = json.loads(path.read_text(encoding="utf-8-sig"))
    digest = str(result["gate_digest"])
    if previous.get("gate_digest"):
        if previous.get("gate_digest") != digest:
            raise ModelCandidateHealthGateError("frozen health-gate result differs on replay")
        return "deterministic_replay"
    allowed_pending = {
        "legacy_repair_gate.json": {"pending"},
        "confirmation_health_gate.json": {"confirmation_pending"},
    }
    if previous.get("status") not in allowed_pending[filename]:
        raise ModelCandidateHealthGateError("health gate is not eligible to run")
    atomic_write_json(path, dict(result))
    refresh_artifact_manifest(candidate)
    return "first_evaluation"


def _update_candidate_status(candidate: Path, status: str, *, final: bool) -> None:
    metadata_path = candidate / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["candidate_status"] = status
    if final:
        metadata["candidate_directory_finalized"] = True
    metadata["metadata_digest"] = _manifest_digest(metadata, "metadata_digest")
    atomic_write_json(metadata_path, metadata)
    artifacts = refresh_artifact_manifest(candidate)
    if TRAINING_SUMMARY.is_file():
        summary = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8-sig"))
        model = summary.get("models", {}).get(metadata["model_kind"])
        if model and model.get("candidate_id") == metadata["candidate_id"]:
            model["status"] = status
            model["candidate_artifacts"] = artifacts
            if status.startswith("legacy_repair"):
                model["legacy_phase22_repair_result"] = status
            if status.startswith("confirmation_health"):
                model["sealed_confirmation_result"] = status
            statuses = [value["status"] for value in summary["models"].values()]
            if len(statuses) < 3 or any(value == "confirmation_pending" for value in statuses):
                verdict = "candidate_training_complete_confirmation_pending"
            elif all(value == "confirmation_health_passed" for value in statuses):
                verdict = "candidate_training_complete_all_confirmation_gates_passed"
            elif any(value == "confirmation_health_passed" for value in statuses):
                verdict = "candidate_training_complete_partial_confirmation_pass"
            else:
                verdict = "candidate_training_complete_confirmation_gates_failed"
            summary["final_decision"] = verdict
            summary["summary_digest"] = _manifest_digest(summary, "summary_digest")
            atomic_write_json(TRAINING_SUMMARY, summary)


def run_health_gate(
    candidate: Path | str,
    *,
    gate: str,
    legacy_bundle: Path | str = PHASE22_BUNDLE,
    confirmation: Path | str | None = None,
    freeze_path: Path | str = SELECTION_FREEZE,
    ledger_path: Path | str = CONFIRMATION_ACCESS_LEDGER,
) -> dict[str, Any]:
    validate_phase24_evidence()
    root = Path(candidate)
    metadata, model, scaler = _load_candidate(root)
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8-sig"))
    if evaluation.get("internal_test_gate", {}).get("passed") is not True:
        raise ModelCandidateHealthGateError("internal_test_failed")
    policy = load_training_policy()
    access: dict[str, Any] | None = None
    if gate == "legacy-repair":
        if Path(legacy_bundle).resolve() != PHASE22_BUNDLE.resolve():
            # Tests may use a byte-identical copied bundle, but production must use known evidence.
            manifest = json.loads((Path(legacy_bundle) / "bundle_manifest.json").read_text(encoding="utf-8-sig"))
            if manifest.get("bundle_digest") != EXPECTED_PHASE22_BUNDLE_DIGEST:
                raise ModelCandidateHealthGateError("legacy repair bundle digest mismatch")
        windows = load_legacy_repair_windows(legacy_bundle)
        evidence = {
            "name": "legacy_repair_regression_set",
            "digest": EXPECTED_PHASE22_BUNDLE_DIGEST,
        }
        filename = "legacy_repair_gate.json"
    elif gate == "confirmation":
        if confirmation is None:
            raise ModelCandidateHealthGateError("sealed_confirmation_pending")
        legacy = json.loads((root / "legacy_repair_gate.json").read_text(encoding="utf-8-sig"))
        if legacy.get("status") != "legacy_repair_passed":
            raise ModelCandidateHealthGateError("legacy_repair_failed")
        confirmation_manifest, windows = _load_confirmation_windows(confirmation, freeze_path=freeze_path)
        access = record_confirmation_access(
            metadata, confirmation_manifest, freeze_path=freeze_path, ledger_path=ledger_path
        )
        evidence = {
            "name": "sealed_confirmation_health_set",
            "confirmation_id": confirmation_manifest["confirmation_id"],
            "digest": confirmation_manifest["confirmation_digest"],
            "integrity_passed": True,
        }
        filename = "confirmation_health_gate.json"
    else:
        raise ModelCandidateHealthGateError("gate must be legacy-repair or confirmation")
    candidate_stats, _ = _inference_health(model, scaler, windows, policy=policy)
    incumbent_stats = _incumbent_health(metadata["model_kind"], windows, policy)
    acceptance = gate_acceptance(
        metadata["model_kind"], candidate_stats, gate=gate, incumbent_per_symbol=incumbent_stats
    )
    result: dict[str, Any] = {
        "schema_version": 1, "candidate_id": metadata["candidate_id"],
        "candidate_model_digest": metadata["model_sha256"],
        "candidate_scaler_digest": metadata["scaler_sha256"],
        "gate": gate, "status": acceptance["status"], "evaluated_at": utc_now(),
        "evidence": evidence, "per_symbol": candidate_stats,
        "incumbent_per_symbol": incumbent_stats, "acceptance": acceptance,
        "candidate_scaler_used": True, "scaler_fit_performed": False,
        "raw_candidate_probabilities_used": True, "calibration_applied": False,
        "thresholds": {
            "flat_output_std_threshold": policy["flat_output_std_threshold"],
            "flat_window": policy["flat_window"],
            "extreme_low_threshold": policy["extreme_low_threshold"],
            "extreme_high_threshold": policy["extreme_high_threshold"],
            "extreme_consecutive_limit": policy["extreme_consecutive_limit"],
            "maximum_missing_rate": policy["maximum_missing_rate"],
        },
        "confirmation_access": access,
    }
    # Access timestamps are ledger evidence, not calculation identity.
    result["gate_digest"] = json_digest({
        key: value for key, value in result.items() if key not in {"evaluated_at", "gate_digest", "confirmation_access"}
    })
    replay = _write_gate_once(root, filename, result)
    result["execution_type"] = access["access_type"] if access else replay
    _update_candidate_status(root, acceptance["status"], final=(gate == "confirmation"))
    verify_incumbent_inventory()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gate", required=True, choices=("legacy-repair", "confirmation"))
    parser.add_argument("--legacy-bundle", default=str(PHASE22_BUNDLE))
    parser.add_argument("--confirmation")
    parser.add_argument("--freeze-manifest", default=str(SELECTION_FREEZE))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not TRAINING_PYTHON.is_file() or Path(sys.executable).resolve() != TRAINING_PYTHON.resolve():
            raise ModelCandidateHealthGateError(
                "candidate health gates must use .venv-model-training/canonical/Scripts/python.exe"
            )
        result = run_health_gate(
            args.candidate, gate=args.gate, legacy_bundle=args.legacy_bundle,
            confirmation=args.confirmation, freeze_path=args.freeze_manifest,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["acceptance"]["passed"] else 3
    except (ModelCandidateHealthGateError, ModelCandidateEvaluationError, ModelCandidateTrainingError) as exc:
        print(json.dumps({"status": str(exc), "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
