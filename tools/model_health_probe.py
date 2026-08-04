"""Deterministic CPU-only probes for deployed ensemble artifacts.

Probe inputs are constructed in scaler z-space, inverse-transformed through the
paired scaler, and then transformed again at the same inference boundary used by
``predict_next``.  No market source, writer, executor, or exchange is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DEFAULT_POLICY = BASE_DIR / "research" / "model_health_policy.json"
DEFAULT_REPORT = BASE_DIR / "reports" / "model_health_probe.json"
SCHEMA_VERSION = 1


class ModelHealthProbeError(ValueError):
    """Raised when a probe registry or artifact cannot be evaluated safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_policy(path: Path | str = DEFAULT_POLICY) -> dict[str, Any]:
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ModelHealthProbeError(f"malformed model health policy: {exc}") from exc
    required = {
        "schema_version", "minimum_rows_for_decision", "minimum_rows_for_warning",
        "flat_output_std_threshold", "flat_output_window", "minimum_rounded_unique_values",
        "maximum_missing_rate", "maximum_extreme_rate", "minimum_probe_std",
        "low_auc_warning_threshold", "critical_contract_fields",
    }
    if not isinstance(policy, dict) or set(policy) != required or policy.get("schema_version") != 1:
        raise ModelHealthProbeError("model health policy fields or schema are invalid")
    if float(policy["flat_output_std_threshold"]) != 0.002:
        raise ModelHealthProbeError("flat_output_std_threshold must match deployed value 0.002")
    return policy


def apply_probability_calibration(
    probabilities: Sequence[float] | np.ndarray,
    bias: float,
    temperature: float,
) -> dict[str, Any]:
    """Apply production calibration: subtract probability bias, then temperature."""

    raw = np.asarray(probabilities, dtype=np.float64)
    before_clip = raw - float(bias)
    bias_adjusted = np.clip(before_clip, 1e-6, 1.0 - 1e-6)
    clipping_count = int(np.count_nonzero(before_clip != bias_adjusted))
    calibrated = bias_adjusted.copy()
    temp = float(temperature)
    if math.isfinite(temp) and temp > 0 and temp != 1.0:
        logits = np.log(calibrated / (1.0 - calibrated))
        calibrated = 1.0 / (1.0 + np.exp(-logits / temp))
    return {
        "raw_probability": raw,
        "after_bias": bias_adjusted,
        "after_temperature": calibrated,
        "clipping_count": clipping_count,
    }


def calibration_decomposition(
    probabilities: Sequence[float] | np.ndarray,
    bias: float,
    temperature: float,
    flat_threshold: float = 0.002,
) -> dict[str, Any]:
    stages = apply_probability_calibration(probabilities, bias, temperature)
    raw = stages["raw_probability"]
    adjusted = stages["after_bias"]
    calibrated = stages["after_temperature"]
    raw_std = float(np.std(raw)) if raw.size else None
    adjusted_std = float(np.std(adjusted)) if adjusted.size else None
    calibrated_std = float(np.std(calibrated)) if calibrated.size else None
    if not raw.size:
        attribution = "unverified"
    elif raw_std < flat_threshold:
        attribution = "before_calibration"
    elif adjusted_std < flat_threshold:
        attribution = "introduced_by_bias"
    elif calibrated_std < flat_threshold:
        attribution = "introduced_by_temperature"
    else:
        attribution = "unchanged_by_calibration"
    return {
        "raw_std": raw_std,
        "bias_adjusted_std": adjusted_std,
        "calibrated_std": calibrated_std,
        "raw_mean": float(np.mean(raw)) if raw.size else None,
        "calibrated_mean": float(np.mean(calibrated)) if calibrated.size else None,
        "clipping_count": stages["clipping_count"],
        "bias_value": float(bias),
        "temperature_value": float(temperature),
        "flatness_attribution": attribution,
    }


def generate_probe_sequences(
    seq_len: int,
    n_features: int,
    *,
    seed: int = 21021,
    probe_count: int = 128,
) -> list[dict[str, Any]]:
    """Return deterministic z-space probes plus exhaustive feature impulses."""

    if seq_len <= 1 or n_features <= 0 or probe_count < 13:
        raise ModelHealthProbeError("probe dimensions are invalid")
    rng = np.random.default_rng(int(seed))
    probes: list[dict[str, Any]] = []

    def add(group: str, values: np.ndarray, **metadata: Any) -> None:
        probes.append({"group": group, "values": np.asarray(values, dtype=np.float32), **metadata})

    shape = (seq_len, n_features)
    add("all_zero_z", np.zeros(shape))
    add("constant_positive_z", np.ones(shape))
    add("constant_negative_z", -np.ones(shape))
    add("deterministic_gaussian", rng.normal(0, 1, shape))
    add("low_amplitude_gaussian", rng.normal(0, 0.1, shape))
    add("high_amplitude_gaussian", rng.normal(0, 3, shape))
    for group, index in (
        ("last_timestep_impulse", seq_len - 1),
        ("first_timestep_impulse", 0),
        ("middle_timestep_impulse", seq_len // 2),
    ):
        values = np.zeros(shape)
        values[index, :] = 2.0
        add(group, values, timestep=index)
    ramp = np.linspace(-2.0, 2.0, seq_len, dtype=np.float32)[:, None]
    add("temporal_ramp", np.repeat(ramp, n_features, axis=1))
    add("reversed_temporal_ramp", np.repeat(ramp[::-1], n_features, axis=1))
    # Fill the requested multi-feature inventory with seeded Gaussian variants.
    while len(probes) < probe_count:
        index = len(probes)
        amplitude = (0.2, 0.5, 1.0, 2.0, 4.0)[index % 5]
        add("deterministic_gaussian", rng.normal(0, amplitude, shape), amplitude=amplitude)
    for feature in range(n_features):
        for sign, group in ((1.0, "individual_feature_positive_impulse"),
                            (-1.0, "individual_feature_negative_impulse")):
            values = np.zeros(shape)
            values[-1, feature] = sign * 2.0
            add(group, values, feature=feature, timestep=seq_len - 1)
    return probes


def _model_outputs(model: torch.nn.Module, scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(scaled, dtype=np.float32)).cpu()
    with torch.no_grad():
        out = model(tensor)
        logits = out["ret_cls_logits"]
        probs = torch.softmax(logits, dim=-1)[:, 1]
        ret = out["ret_reg"].reshape(-1)
        rv = out["rv_reg"].reshape(-1)
    return (
        ret.detach().cpu().numpy().astype(np.float64),
        rv.detach().cpu().numpy().astype(np.float64),
        probs.detach().cpu().numpy().astype(np.float64),
    )


def run_model_probe(
    model: torch.nn.Module,
    scaler: Any,
    probes: Sequence[Mapping[str, Any]],
    *,
    deterministic_tolerance: float = 1e-10,
    minimum_probe_std: float = 0.002,
    predict_fn: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate every raw-space round-tripped probe twice on CPU."""

    if not probes:
        raise ModelHealthProbeError("probe set is empty")
    z = np.stack([np.asarray(probe["values"], dtype=np.float64) for probe in probes])
    count, seq_len, n_features = z.shape
    raw = scaler.inverse_transform(z.reshape(-1, n_features)).reshape(count, seq_len, n_features)
    scaled = scaler.transform(raw.reshape(-1, n_features)).reshape(count, seq_len, n_features)
    model = model.eval().cpu()
    if predict_fn is None:
        first = _model_outputs(model, scaled)
        second = _model_outputs(model, scaled)
    else:
        def production_pass() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = [predict_fn(raw[index].astype(np.float32), scaler, model, "cpu")
                      for index in range(count)]
            return tuple(np.asarray([item[position] for item in values], dtype=np.float64)
                         for position in range(3))  # type: ignore[return-value]
        first = production_pass()
        second = production_pass()
    repeat_error = max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))
    ret, rv, probabilities = first
    finite = np.isfinite(ret) & np.isfinite(rv) & np.isfinite(probabilities)
    nonfinite_count = int(np.size(ret) * 3 - np.isfinite(ret).sum() - np.isfinite(rv).sum()
                          - np.isfinite(probabilities).sum())
    groups: dict[str, list[int]] = {}
    for index, probe in enumerate(probes):
        groups.setdefault(str(probe["group"]), []).append(index)
    zero_index = groups["all_zero_z"][0]

    def group_delta(name: str) -> Optional[float]:
        indices = groups.get(name, [])
        if not indices:
            return None
        delta = np.abs(probabilities[indices] - probabilities[zero_index])
        return None if not np.isfinite(delta).any() else float(np.nanmean(delta))

    feature_sensitivity: list[float] = []
    positives = groups.get("individual_feature_positive_impulse", [])
    negatives = groups.get("individual_feature_negative_impulse", [])
    for pos, neg in zip(positives, negatives):
        value = float(abs(probabilities[pos] - probabilities[neg]))
        if math.isfinite(value):
            feature_sensitivity.append(value)
    finite_probabilities = probabilities[np.isfinite(probabilities)]
    finite_ret = ret[np.isfinite(ret)]
    finite_rv = rv[np.isfinite(rv)]
    rounded = np.unique(np.round(finite_probabilities, 6))
    result = {
        "probe_count": int(count),
        "p_long_mean": None if not finite_probabilities.size else float(np.mean(finite_probabilities)),
        "p_long_std": None if not finite_probabilities.size else float(np.std(finite_probabilities)),
        "p_long_min": None if not finite_probabilities.size else float(np.min(finite_probabilities)),
        "p_long_max": None if not finite_probabilities.size else float(np.max(finite_probabilities)),
        "rounded_unique_count": int(rounded.size),
        "ret_hat_std": None if not finite_ret.size else float(np.std(finite_ret)),
        "rv_hat_std": None if not finite_rv.size else float(np.std(finite_rv)),
        "nonfinite_count": nonfinite_count,
        "deterministic_repeat_max_error": repeat_error,
        "deterministic_repeat_passed": bool(repeat_error <= deterministic_tolerance),
        "last_timestep_sensitivity": group_delta("last_timestep_impulse"),
        "middle_timestep_sensitivity": group_delta("middle_timestep_impulse"),
        "first_timestep_sensitivity": group_delta("first_timestep_impulse"),
        "feature_sensitivity_min": float(np.min(feature_sensitivity)) if feature_sensitivity else None,
        "feature_sensitivity_median": float(np.median(feature_sensitivity)) if feature_sensitivity else None,
        "feature_sensitivity_max": float(np.max(feature_sensitivity)) if feature_sensitivity else None,
        "inactive_feature_count": int(np.count_nonzero(np.asarray(feature_sensitivity) <= 1e-8)),
        "_raw_probabilities": probabilities,
        "_scaled_probes": scaled,
        "_finite_mask": finite,
    }
    if nonfinite_count:
        result["status"] = "failed_nonfinite_output"
    elif repeat_error > deterministic_tolerance:
        result["status"] = "failed_nondeterministic_output"
    elif result["p_long_std"] is None or float(result["p_long_std"]) < minimum_probe_std:
        result["status"] = "failed_flat_output"
    else:
        result["status"] = "passed"
    return result


def _tcn_endpoint_forward(
    model: torch.nn.Module, x: torch.Tensor, endpoint: str
) -> tuple[dict[str, torch.Tensor], list[int]]:
    h = x.transpose(1, 2)
    lengths: list[int] = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Conv1d):
            incoming = int(h.shape[-1])
            if endpoint == "deployed_current_endpoint":
                h = layer(h)
            elif endpoint == "right_cropped_same_length_endpoint":
                h = layer(h)[..., :incoming]
            elif endpoint == "causal_left_padding_endpoint":
                left = int(layer.dilation[0] * (layer.kernel_size[0] - 1))
                h = F.conv1d(
                    F.pad(h, (left, 0)), layer.weight, layer.bias,
                    stride=layer.stride, padding=0, dilation=layer.dilation, groups=layer.groups,
                )
            else:
                raise ModelHealthProbeError(f"unknown TCN endpoint: {endpoint}")
            lengths.append(int(h.shape[-1]))
        else:
            h = layer(h)
    last = h[:, :, -1]
    return {
        "ret_reg": model.head_ret_reg(last).squeeze(-1),
        "ret_cls_logits": model.head_ret_cls(last),
        "rv_reg": model.head_rv_reg(last).squeeze(-1),
    }, lengths


def diagnose_tcn_architecture(
    model: torch.nn.Module,
    scaled_probes: np.ndarray,
    *,
    flat_threshold: float = 0.002,
) -> dict[str, Any]:
    """Compare three read-only forwards using the same trained TCN parameters."""

    if not hasattr(model, "net") or not hasattr(model, "head_ret_cls"):
        raise ModelHealthProbeError("TCN diagnostic requires the deployed TCN structure")
    model = model.eval().cpu()
    inputs = torch.tensor(np.asarray(scaled_probes, dtype=np.float32), requires_grad=True)
    endpoints = (
        "deployed_current_endpoint",
        "right_cropped_same_length_endpoint",
        "causal_left_padding_endpoint",
    )
    output: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    gradient_norms: dict[str, float] = {}
    for endpoint in endpoints:
        if inputs.grad is not None:
            inputs.grad.zero_()
        out, lengths = _tcn_endpoint_forward(model, inputs, endpoint)
        probability = torch.softmax(out["ret_cls_logits"], dim=-1)[:, 1]
        probability.mean().backward(retain_graph=True)
        gradient = inputs.grad.detach().clone()
        per_time = torch.linalg.vector_norm(gradient, dim=(0, 2)).cpu().numpy()
        total = float(np.sum(per_time))
        seq_len = len(per_time)
        first = float(np.sum(per_time[: seq_len // 2]) / total) if total > 0 else 0.0
        second = float(np.sum(per_time[seq_len // 2 :]) / total) if total > 0 else 0.0
        final = float(per_time[-1] / total) if total > 0 else 0.0
        values = probability.detach().cpu().numpy().astype(np.float64)
        probabilities[endpoint] = values
        gradient_norms[endpoint] = float(torch.linalg.vector_norm(gradient).item())
        profile_total = float(np.sum(per_time))
        profile = (per_time / profile_total).tolist() if profile_total > 0 else [0.0] * seq_len
        output[endpoint] = {
            "output_sequence_length_after_each_conv": lengths,
            "p_long_probe_std": float(np.std(values)),
            "p_long_min": float(np.min(values)),
            "p_long_max": float(np.max(values)),
            "temporal_sensitivity_profile": [float(v) for v in profile],
            "input_gradient_norm": gradient_norms[endpoint],
            "first_half_gradient_fraction": first,
            "second_half_gradient_fraction": second,
            "final_timestep_gradient_fraction": final,
            "zero_padding_dependency_score": None,
            "current_vs_cropped_probability_difference": None,
            "current_vs_causal_probability_difference": None,
        }
    current = probabilities[endpoints[0]]
    cropped_diff = float(np.mean(np.abs(current - probabilities[endpoints[1]])))
    causal_diff = float(np.mean(np.abs(current - probabilities[endpoints[2]])))
    current_std = output[endpoints[0]]["p_long_probe_std"]
    alternate_std = max(output[endpoints[1]]["p_long_probe_std"], output[endpoints[2]]["p_long_probe_std"])
    alternate_grad = max(gradient_norms[endpoints[1]], gradient_norms[endpoints[2]])
    for endpoint in endpoints:
        output[endpoint]["current_vs_cropped_probability_difference"] = cropped_diff
        output[endpoint]["current_vs_causal_probability_difference"] = causal_diff
        output[endpoint]["zero_padding_dependency_score"] = max(cropped_diff, causal_diff)
    materially_more_variable = alternate_std >= flat_threshold and alternate_std > max(current_std * 2.0, current_std + 1e-6)
    materially_more_sensitive = alternate_grad > max(gradient_norms[endpoints[0]] * 2.0, 1e-10)
    suspected = bool(current_std < flat_threshold and (materially_more_variable or materially_more_sensitive))
    return {
        "endpoints": output,
        "architecture_issue_suspected": suspected,
        "diagnostic_hypothesis_only": True,
        "trained_weights_reused": True,
        "state_dict_modified": False,
    }


def _load_artifact(entry: Mapping[str, Any], root: Path) -> tuple[Any, torch.nn.Module]:
    from ml_dl.dl_infer import _build_model

    scaler_path = root / str(entry["scaler_filename"])
    model_path = root / str(entry["model_filename"])
    scaler = joblib.load(scaler_path)
    width = int(getattr(scaler, "n_features_in_"))
    model = _build_model(str(entry["kind"]), width)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return scaler, model.eval().cpu()


def run_artifact_probes(
    snapshot: Mapping[str, Any],
    *,
    base_dir: Path | str = BASE_DIR,
    seed: int = 21021,
    probe_count: int = 128,
    policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    root = Path(base_dir)
    health_policy = dict(policy or load_policy())
    results: dict[str, Any] = {}
    tcn_analysis: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    biases = {"lstm": snapshot.get("dl_bias_lstm", 0.0), "tcn": snapshot.get("dl_bias_tcn", 0.0),
              "tx": snapshot.get("dl_bias_tx", 0.0), "adv": 0.0}
    temperatures = {"lstm": snapshot.get("dl_temp_lstm", 1.0), "tcn": snapshot.get("dl_temp_tcn", 1.0),
                    "tx": snapshot.get("dl_temp_tx", 1.0), "adv": 1.0}
    for entry in snapshot.get("model_entries", []):
        kind = str(entry.get("kind"))
        if entry.get("model_load_status") != "loaded":
            results[kind] = {"status": "failed_artifact_load", "model_load_status": entry.get("model_load_status")}
            continue
        try:
            from ml_dl.dl_infer import predict_next

            scaler, model = _load_artifact(entry, root)
            seq_len = int(entry.get("metadata_seq_len") or snapshot.get("dl_seq_len") or 64)
            width = int(getattr(scaler, "n_features_in_"))
            probes = generate_probe_sequences(seq_len, width, seed=seed, probe_count=probe_count)
            raw_result = run_model_probe(
                model, scaler, probes,
                minimum_probe_std=float(health_policy["minimum_probe_std"]),
                predict_fn=predict_next,
            )
            probabilities = raw_result.pop("_raw_probabilities")
            scaled_probes = raw_result.pop("_scaled_probes")
            raw_result.pop("_finite_mask")
            results[kind] = raw_result
            calibration[kind] = (
                {"status": "unavailable_nonfinite_output"}
                if raw_result["nonfinite_count"] else
                calibration_decomposition(
                    probabilities, float(biases[kind]), float(temperatures[kind]),
                    float(health_policy["flat_output_std_threshold"]),
                )
            )
            if kind == "tcn":
                tcn_analysis = diagnose_tcn_architecture(
                    model, scaled_probes,
                    flat_threshold=float(health_policy["flat_output_std_threshold"]),
                )
        except Exception as exc:
            results[kind] = {"status": "failed_artifact_load", "error": f"{type(exc).__name__}: {exc}"}
    return {"models": results, "calibration": calibration, "tcn_architecture": tcn_analysis}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic offline model probes.")
    parser.add_argument("--snapshot")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--model", choices=("lstm", "tcn", "tx", "adv"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from tools.model_serving_snapshot import (
            capture_model_serving_snapshot,
            load_model_serving_snapshot,
        )
        root = Path(args.base_dir)
        if args.snapshot:
            snapshot = load_model_serving_snapshot(args.snapshot)
        else:
            # Probes are offline even when the local trading environment is live.
            # Artifact inventory is captured with explicit in-memory safety flags;
            # no current settings are written or treated as historical evidence.
            snapshot = capture_model_serving_snapshot(
                "current_model_serving", "offline_probe", base_dir=root,
                forced_env_overrides={"LIVE_TRADING": False, "PAPER_TRADING": True,
                                      "LIVE_MODE": False, "EXEC_PAPER": True,
                                      "PLACE_REAL_ORDERS": False},
            )
        if args.model:
            snapshot = dict(snapshot)
            snapshot["model_entries"] = [entry for entry in snapshot["model_entries"] if entry["kind"] == args.model]
        policy = load_policy(args.policy)
        analysis = run_artifact_probes(snapshot, base_dir=root, seed=args.seed,
                                       probe_count=args.probe_count, policy=policy)
        output = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now(),
            "policy": {"offline_only": True, "market_data_used": False, "artifacts_modified": False},
            "snapshot_digest": snapshot.get("snapshot_digest"),
            "seed": args.seed,
            "multi_feature_probe_count": args.probe_count,
            **analysis,
        }
        output["probe_digest"] = _digest({key: value for key, value in output.items() if key != "generated_at"})
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        if args.json or not args.json_out:
            print(json.dumps(output, indent=2))
        else:
            print(json.dumps({"status": "completed", "json_out": str(args.json_out),
                              "probe_digest": output["probe_digest"]}))
        return 0
    except Exception as exc:
        print(f"model_health_probe_error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
