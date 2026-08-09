"""Evaluate a frozen Phase 24 candidate against its incumbent on Phase 22.

Raw probabilities are compared, and raw-unit auxiliary predictions receive
prediction-safety diagnostics.  No calibration, target fitting, strategy-return
calculation, promotion, or serving mutation is implemented here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_train import (
    ALLOWED_KINDS,
    ModelCandidateTrainingError,
    _manifest_digest,
    _verify_candidate_artifacts_read_only,
    make_candidate_model,
    refresh_artifact_manifest,
    TRAINING_SUMMARY,
)
from tools.model_training_environment import (
    PHASE22_BUNDLE,
    EXPECTED_PHASE22_BUNDLE_DIGEST,
    atomic_write_json,
    file_digest,
    json_digest,
    load_training_policy,
    TRAINING_PYTHON,
    validate_phase24_evidence,
)


class ModelCandidateEvaluationError(ValueError):
    """The immutable evaluation contract or an artifact digest failed."""


def load_legacy_repair_windows(bundle: Path | str = PHASE22_BUNDLE) -> dict[str, dict[str, Any]]:
    root = Path(bundle)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("bundle_digest") != EXPECTED_PHASE22_BUNDLE_DIGEST:
        raise ModelCandidateEvaluationError("legacy repair bundle digest mismatch")
    for name, digest in manifest.get("bundle_file_digests", {}).items():
        if not (root / name).is_file() or file_digest(root / name) != digest:
            raise ModelCandidateEvaluationError(f"legacy repair bundle file mismatch: {name}")
    records = [json.loads(line) for line in (root / "evaluation_windows.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_symbol.setdefault(str(record["symbol"]), []).append(record)
    feature_names = None
    result: dict[str, dict[str, Any]] = {}
    for symbol in manifest["symbols"]:
        path = root / f"features_{symbol}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ModelCandidateEvaluationError(f"legacy feature file empty: {symbol}")
        columns = [name for name in rows[0] if name != "feature_open_utc"]
        if feature_names is None:
            feature_names = columns
        elif columns != feature_names:
            raise ModelCandidateEvaluationError("legacy feature order differs by symbol")
        if len(columns) != 27:
            raise ModelCandidateEvaluationError("legacy window feature width mismatch")
        timestamps = [row["feature_open_utc"] for row in rows]
        lookup = {value: index for index, value in enumerate(timestamps)}
        matrix = np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=np.float32)
        windows: list[np.ndarray] = []
        identities: list[str] = []
        endpoints: list[str] = []
        for record in by_symbol.get(symbol, []):
            endpoint = str(record["feature_window_last_utc"])
            index = lookup.get(endpoint)
            length = int(record["feature_window_row_count"])
            if index is None or length != 64 or index - length + 1 < 0:
                raise ModelCandidateEvaluationError("legacy evaluation window cannot be reconstructed")
            window = matrix[index - length + 1:index + 1]
            if window.shape != (64, 27) or not np.isfinite(window).all():
                raise ModelCandidateEvaluationError("legacy evaluation window is invalid")
            windows.append(window)
            identities.append(str(record["source_bar_id"]))
            endpoints.append(str(record["source_bar_open_utc"]))
        if len(windows) != int(manifest["unique_completed_bars_by_symbol"][symbol]):
            raise ModelCandidateEvaluationError("legacy evaluation row count mismatch")
        result[symbol] = {
            "windows": np.stack(windows), "source_bar_ids": identities,
            "endpoint_timestamps": endpoints,
        }
    return result


def _load_candidate(candidate: Path | str):
    import joblib
    import torch

    root = Path(candidate)
    _verify_candidate_artifacts_read_only(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    kind = str(metadata.get("model_kind"))
    if kind not in ALLOWED_KINDS:
        raise ModelCandidateEvaluationError("unsupported candidate model kind")
    if file_digest(root / "model.pt") != metadata.get("model_sha256"):
        raise ModelCandidateEvaluationError("candidate model digest mismatch")
    if file_digest(root / "scaler.joblib") != metadata.get("scaler_sha256"):
        raise ModelCandidateEvaluationError("candidate scaler digest mismatch")
    model = make_candidate_model(kind, metadata["architecture_config"])
    model.load_state_dict(torch.load(root / "model.pt", map_location="cpu", weights_only=True))
    model.eval().cpu()
    scaler = joblib.load(root / "scaler.joblib")
    if int(getattr(scaler, "n_features_in_", -1)) != 27:
        raise ModelCandidateEvaluationError("candidate scaler width mismatch")
    return metadata, model, scaler


def _load_incumbent(kind: str, bundle: Path | str = PHASE22_BUNDLE):
    import joblib
    import torch

    snapshot = json.loads((Path(bundle) / "model_serving_snapshot.json").read_text(encoding="utf-8-sig"))
    entry = next((value for value in snapshot["model_entries"] if value["kind"] == kind), None)
    if entry is None:
        raise ModelCandidateEvaluationError("incumbent snapshot entry missing")
    model_path = BASE_DIR / entry["model_filename"]
    scaler_path = BASE_DIR / entry["scaler_filename"]
    if file_digest(model_path) != entry["model_sha256"] or file_digest(scaler_path) != entry["scaler_sha256"]:
        raise ModelCandidateEvaluationError("incumbent artifact digest mismatch")
    model = make_candidate_model(kind)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval().cpu()
    return entry, model, joblib.load(scaler_path)


def infer_raw_outputs(model: Any, scaler: Any, windows: np.ndarray, *, batch_size: int = 64) -> dict[str, np.ndarray]:
    import torch

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (64, 27):
        raise ModelCandidateEvaluationError("expected [N,64,27] windows")
    transformed = scaler.transform(values.reshape(-1, 27)).reshape(values.shape).astype(np.float32, copy=False)
    parts: dict[str, list[np.ndarray]] = {"probability": [], "ret_hat": [], "rv_hat": []}
    model.eval().cpu()
    with torch.no_grad():
        for start in range(0, len(transformed), int(batch_size)):
            tensor = torch.from_numpy(np.ascontiguousarray(transformed[start:start + batch_size]))
            output = model(tensor)
            parts["probability"].append(torch.softmax(output["ret_cls_logits"], dim=-1)[:, 1].cpu().numpy())
            parts["ret_hat"].append(output["ret_reg"].reshape(-1).cpu().numpy())
            parts["rv_hat"].append(output["rv_reg"].reshape(-1).cpu().numpy())
    return {
        key: np.concatenate(value).astype(np.float64) if value else np.asarray([], dtype=np.float64)
        for key, value in parts.items()
    }


def infer_raw_probabilities(model: Any, scaler: Any, windows: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    return infer_raw_outputs(model, scaler, windows, batch_size=batch_size)["probability"]


def auxiliary_prediction_health(
    outputs: Mapping[str, Sequence[float]], *, deterministic_repeat_error: float = 0.0,
) -> dict[str, Any]:
    """Hard safety checks for unlabeled repair/confirmation windows."""
    result: dict[str, Any] = {
        "targets_present": False,
        "skill_metrics_available": False,
        "post_hoc_rv_clipping_applied": False,
        "outputs_remain_in_raw_target_units": True,
        "deterministic_repeat_max_absolute_error": float(deterministic_repeat_error),
        "deterministic_repeat_passed": bool(float(deterministic_repeat_error) == 0.0),
    }
    hard = set()
    for key, output_name in (("ret_hat", "ret_reg"), ("rv_hat", "rv_reg")):
        values = np.asarray(outputs[key], dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        nonfinite = int(len(values) - len(finite))
        classification = "auxiliary_unverified"
        if nonfinite or not result["deterministic_repeat_passed"]:
            classification = "auxiliary_failed_nonfinite"
        elif len(finite) and float(np.std(finite, ddof=0)) <= 1e-12:
            classification = "auxiliary_failed_constant_output"
        negative_count = int(np.sum(finite < 0)) if key == "rv_hat" else 0
        if key == "rv_hat" and negative_count:
            classification = "auxiliary_failed_negative_rv"
        row = {
            "classification": classification,
            "finite_prediction_count": int(len(finite)),
            "nonfinite_prediction_count": nonfinite,
            "prediction_mean": None if not len(finite) else float(np.mean(finite)),
            "prediction_std": None if not len(finite) else float(np.std(finite, ddof=0)),
            "prediction_min": None if not len(finite) else float(np.min(finite)),
            "prediction_max": None if not len(finite) else float(np.max(finite)),
        }
        if key == "rv_hat":
            row.update({
                "negative_prediction_count": negative_count,
                "negative_prediction_rate": 0.0 if not len(finite) else float(negative_count / len(finite)),
                "rv_target_negative_count": None,
            })
        result[output_name] = row
        if classification.startswith("auxiliary_failed_"):
            hard.add(classification)
    result["hard_failure_reasons"] = sorted(hard)
    result["auxiliary_head_safety_gate_passed"] = not hard
    return result


def _longest_repeat(values: np.ndarray) -> int:
    if not len(values):
        return 0
    longest = current = 1
    for index in range(1, len(values)):
        current = current + 1 if values[index] == values[index - 1] else 1
        longest = max(longest, current)
    return int(longest)


def probability_health_statistics(
    probabilities: Sequence[float],
    identities: Sequence[str],
    *,
    policy: Mapping[str, Any] | None = None,
    expected_count: int | None = None,
    deterministic_repeat_error: float = 0.0,
) -> dict[str, Any]:
    policy = dict(policy or load_training_policy())
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    ids = [str(value) for value in identities]
    if len(values) != len(ids):
        raise ModelCandidateEvaluationError("probabilities and identities differ in length")
    unique_values: list[float] = []
    seen: set[str] = set()
    duplicate_bars = 0
    for identity, value in zip(ids, values):
        if identity in seen:
            duplicate_bars += 1
            continue
        seen.add(identity)
        unique_values.append(float(value))
    array = np.asarray(unique_values, dtype=np.float64)
    finite_mask = np.isfinite(array)
    finite = array[finite_mask]
    expected = int(expected_count if expected_count is not None else len(seen))
    missing = max(0, expected - len(array))
    nonfinite = int((~finite_mask).sum())
    history: list[float] = []
    consecutive = maximum_consecutive = 0
    extreme_active = flat_active = False
    extreme_events = flat_events = rolling_flat_windows = excluded = 0
    for value in finite:
        extreme_value = value < float(policy["extreme_low_threshold"]) or value > float(policy["extreme_high_threshold"])
        consecutive = consecutive + 1 if extreme_value else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
        history.append(float(value))
        if len(history) > int(policy["flat_window"]):
            history.pop(0)
        flat = len(history) == int(policy["flat_window"]) and float(np.std(history)) < float(policy["flat_output_std_threshold"])
        extreme = consecutive >= int(policy["extreme_consecutive_limit"])
        if extreme and not extreme_active:
            extreme_events += 1
        if flat and not flat_active:
            flat_events += 1
        rolling_flat_windows += int(flat)
        excluded += int(extreme or flat)
        extreme_active, flat_active = extreme, flat
    missing_rate = float((missing + nonfinite) / expected) if expected else 1.0
    std = float(np.std(finite)) if len(finite) else None
    one_sided = bool(
        len(finite) and std is not None and std >= float(policy["flat_output_std_threshold"])
        and max(float(np.mean(finite > 0.5)), float(np.mean(finite < 0.5))) > 0.95
    )
    quantiles = np.quantile(finite, [0.05, 0.5, 0.95]) if len(finite) else [None, None, None]
    critical = bool(
        extreme_events or flat_events or nonfinite
        or missing_rate > float(policy["maximum_missing_rate"])
        or float(deterministic_repeat_error) != 0.0
    )
    return {
        "unique_bars": int(len(seen)), "duplicate_bars_ignored": duplicate_bars,
        "probability_mean": None if not len(finite) else float(np.mean(finite)),
        "probability_std": std,
        "minimum": None if not len(finite) else float(np.min(finite)),
        "maximum": None if not len(finite) else float(np.max(finite)),
        "p05": None if not len(finite) else float(quantiles[0]),
        "median": None if not len(finite) else float(quantiles[1]),
        "p95": None if not len(finite) else float(quantiles[2]),
        "rounded_unique_count": int(len(np.unique(np.round(finite, 6)))),
        "extreme_low_rate": None if not len(finite) else float(np.mean(finite < float(policy["extreme_low_threshold"]))),
        "extreme_high_rate": None if not len(finite) else float(np.mean(finite > float(policy["extreme_high_threshold"]))),
        "longest_repeat": _longest_repeat(finite),
        "rolling_flat_windows": int(rolling_flat_windows),
        "maximum_consecutive_extreme": int(maximum_consecutive),
        "extreme_exclusion_events": int(extreme_events),
        "flat_exclusion_events": int(flat_events),
        "excluded_endpoints": int(excluded + nonfinite),
        "missing_rate": missing_rate,
        "nonfinite_outputs": nonfinite,
        "deterministic_repeat_error": float(deterministic_repeat_error),
        "deterministic_repeat_passed": float(deterministic_repeat_error) == 0.0,
        "one_sided_warning": one_sided,
        "critical_failure": critical,
    }


def compare_probability_series(
    incumbent: Sequence[float], candidate: Sequence[float], identities: Sequence[str],
    *, policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inc = np.asarray(incumbent, dtype=np.float64)
    new = np.asarray(candidate, dtype=np.float64)
    if inc.shape != new.shape:
        raise ModelCandidateEvaluationError("incumbent/candidate probability shapes differ")
    finite = np.isfinite(inc) & np.isfinite(new)
    correlation = None
    if int(finite.sum()) >= 2 and float(np.std(inc[finite])) > 0 and float(np.std(new[finite])) > 0:
        correlation = float(np.corrcoef(inc[finite], new[finite])[0, 1])
    inc_health = probability_health_statistics(inc, identities, policy=policy, expected_count=len(identities))
    new_health = probability_health_statistics(new, identities, policy=policy, expected_count=len(identities))
    return {
        "incumbent_probability_mean": None if not finite.any() else float(np.mean(inc[finite])),
        "incumbent_probability_std": None if not finite.any() else float(np.std(inc[finite])),
        "candidate_probability_mean": None if not finite.any() else float(np.mean(new[finite])),
        "candidate_probability_std": None if not finite.any() else float(np.std(new[finite])),
        "correlation": correlation,
        "mean_absolute_probability_difference": None if not finite.any() else float(np.mean(np.abs(inc[finite] - new[finite]))),
        "direction_overlap": None if not finite.any() else float(np.mean((inc[finite] >= 0.5) == (new[finite] >= 0.5))),
        "incumbent_exclusion_events": int(inc_health["extreme_exclusion_events"] + inc_health["flat_exclusion_events"]),
        "candidate_exclusion_events": int(new_health["extreme_exclusion_events"] + new_health["flat_exclusion_events"]),
        "repaired_known_failure": bool(inc_health["critical_failure"] and not new_health["critical_failure"]),
        "new_failure": bool(not inc_health["critical_failure"] and new_health["critical_failure"]),
        "incumbent_health": inc_health, "candidate_health": new_health,
    }


def evaluate_candidate_vs_incumbent(
    candidate: Path | str, *, bundle: Path | str = PHASE22_BUNDLE, write_result: bool = True
) -> dict[str, Any]:
    validate_phase24_evidence()
    root = Path(candidate)
    metadata, candidate_model, candidate_scaler = _load_candidate(root)
    kind = metadata["model_kind"]
    incumbent_entry, incumbent_model, incumbent_scaler = _load_incumbent(kind, bundle)
    windows = load_legacy_repair_windows(bundle)
    policy = load_training_policy()
    per_symbol: dict[str, Any] = {}
    for symbol, values in sorted(windows.items()):
        candidate_outputs = infer_raw_outputs(candidate_model, candidate_scaler, values["windows"])
        repeated = infer_raw_outputs(candidate_model, candidate_scaler, values["windows"])
        repeat_error = max(
            (float(np.max(np.abs(candidate_outputs[key] - repeated[key]))) if len(candidate_outputs[key]) else 0.0)
            for key in candidate_outputs
        )
        candidate_p = candidate_outputs["probability"]
        incumbent_p = infer_raw_probabilities(incumbent_model, incumbent_scaler, values["windows"])
        per_symbol[symbol] = compare_probability_series(
            incumbent_p, candidate_p, values["source_bar_ids"], policy=policy
        )
        per_symbol[symbol]["candidate_auxiliary_prediction_health"] = auxiliary_prediction_health(
            candidate_outputs, deterministic_repeat_error=repeat_error
        )
    result = {
        "schema_version": 1, "candidate_id": metadata["candidate_id"], "model_kind": kind,
        "evidence_name": "legacy_repair_regression_set", "bundle_digest": EXPECTED_PHASE22_BUNDLE_DIGEST,
        "same_immutable_windows": True, "candidate_scaler_used": True,
        "candidate_scaler_refit": False, "raw_probabilities_used": True,
        "incumbent_model_digest": incumbent_entry["model_sha256"],
        "candidate_model_digest": metadata["model_sha256"], "per_symbol": per_symbol,
        "candidate_objective": metadata.get("training_objective", {}).get("name"),
        "auxiliary_targets_present": False,
        "auxiliary_skill_metrics_available": False,
        "strategy_return_calculation_performed": False,
    }
    result["comparison_digest"] = json_digest(result)
    if write_result:
        evaluation_path = root / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
        previous = evaluation.get("legacy_incumbent_comparison")
        if isinstance(previous, dict):
            if previous.get("comparison_digest") != result["comparison_digest"]:
                raise ModelCandidateEvaluationError("incumbent comparison is immutable")
        elif previous == "pending":
            evaluation["legacy_incumbent_comparison"] = result
            evaluation["evaluation_digest"] = json_digest({key: value for key, value in evaluation.items() if key != "evaluation_digest"})
            atomic_write_json(evaluation_path, evaluation)
            refresh_artifact_manifest(root)
        else:
            raise ModelCandidateEvaluationError("unexpected incumbent comparison state")
        if TRAINING_SUMMARY.is_file():
            summary = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8-sig"))
            model_summary = summary.get("models", {}).get(kind)
            if model_summary and model_summary.get("candidate_id") == metadata["candidate_id"]:
                model_summary["incumbent_comparisons"] = result
                summary["summary_digest"] = json_digest({
                    key: value for key, value in summary.items() if key != "summary_digest"
                })
                atomic_write_json(TRAINING_SUMMARY, summary)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--legacy-bundle", default=str(PHASE22_BUNDLE))
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not TRAINING_PYTHON.is_file() or Path(sys.executable).resolve() != TRAINING_PYTHON.resolve():
            raise ModelCandidateEvaluationError(
                "candidate evaluation must use .venv-model-training/canonical/Scripts/python.exe"
            )
        result = evaluate_candidate_vs_incumbent(
            args.candidate, bundle=args.legacy_bundle, write_result=not args.no_write
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ModelCandidateEvaluationError, ModelCandidateTrainingError) as exc:
        print(json.dumps({"status": "candidate_evaluation_failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
