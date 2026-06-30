"""Verify the optional XGBoost confirmation gate in paper-test mode.

This script does not run market data, the executor, or strategy logic. It
simulates the live_writer XGBoost gate with deterministic inputs so the
blocking contract can be checked safely before any paper-test run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

THIS = Path(__file__).resolve()
BASE_DIR = THIS.parents[1] if THIS.parent.name == "tools" else THIS.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from runtime.loader import apply_run_config

    apply_run_config(BASE_DIR)
except Exception:
    pass

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=True)
except Exception:
    pass

from ml_optional.xgboost_signal import (  # noqa: E402
    DEFAULT_ARTIFACT,
    XGBOOST_SHADOW_COLS,
    XGBoostSignalConfirmer,
    should_reject_signal,
)


class _StaticXGBoostModel:
    n_features_in_ = 18
    classes_ = np.array([0, 1])

    def __init__(self, short_probability: float, long_probability: float) -> None:
        self.probs = np.asarray([short_probability, long_probability], dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self.probs], dtype=float)


class _ErrorXGBoostModel:
    n_features_in_ = 18
    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise RuntimeError("verification model error")


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


@contextmanager
def _temporary_env(values: Dict[str, str]) -> Iterator[None]:
    old = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _append_row(path: Path, row: Dict[str, object]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=XGBOOST_SHADOW_COLS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in XGBOOST_SHADOW_COLS})


def _sample_window(model: Any) -> np.ndarray:
    n_features = int(getattr(model, "n_features_in_", 0) or 0)
    if n_features <= 0 and hasattr(model, "get_booster"):
        try:
            n_features = int(model.get_booster().num_features())
        except Exception:
            n_features = 0
    if n_features <= 0:
        raise RuntimeError("XGBoost model does not expose n_features_in_")
    live_width = (n_features - 3) // 5 if n_features > 3 and (n_features - 3) % 5 == 0 else n_features
    base = np.linspace(-0.5, 0.5, live_width, dtype=np.float32)
    return np.vstack([base * 0.25, base * 0.5, base * 0.75, base]).astype(np.float32)


def _missing_artifact_path(artifact: Path) -> Path:
    parent = artifact.parent if artifact.parent != Path("") else BASE_DIR
    stem = artifact.stem or "xgboost_signal"
    candidate = parent / f"__missing_{stem}_blocking_verify__.joblib"
    idx = 1
    while candidate.exists():
        candidate = parent / f"__missing_{stem}_blocking_verify_{idx}.joblib"
        idx += 1
    return candidate


def _loaded_confirmer(model: Any, artifact: Path, model_version: str) -> XGBoostSignalConfirmer:
    return XGBoostSignalConfirmer(
        enabled=True,
        artifact_path=artifact,
        model=model,
        model_version=model_version,
        confidence_threshold=0.60,
        xgboost_status="loaded",
    )


def _missing_confirmer(artifact: Path, base_dir: Path) -> XGBoostSignalConfirmer:
    missing = _missing_artifact_path(artifact)
    with _temporary_env({"XGBOOST_SIGNAL_ARTIFACT": str(missing)}):
        return XGBoostSignalConfirmer.from_env(enabled=True, base_dir=base_dir)


def simulate_live_writer_decision(
    *,
    case: str,
    confirmer: XGBoostSignalConfirmer,
    use_xgboost_signal: bool,
    xgboost_signal_blocking: bool,
    symbol: str = "VERIFY",
    signal_allow: int = 1,
    existing_signal: str = "LONG",
    existing_score: float = 0.25,
    ts: str = "VERIFY_TS",
) -> Dict[str, Any]:
    """Simulate the live_writer XGBoost allow-gate branch."""
    window = _sample_window(confirmer.model) if confirmer.model is not None else None
    if use_xgboost_signal:
        result = confirmer.evaluate(
            symbol=symbol,
            window=window,
            existing_signal=existing_signal,
            existing_score=existing_score,
            rv_mean=0.0,
            price=0.0,
        )
        actually_rejected = bool(signal_allow and should_reject_signal(result, xgboost_signal_blocking))
    else:
        disabled = XGBoostSignalConfirmer(enabled=False, artifact_path=confirmer.artifact_path)
        result = disabled.evaluate(
            symbol=symbol,
            window=window,
            existing_signal=existing_signal,
            existing_score=existing_score,
            rv_mean=0.0,
            price=0.0,
        )
        actually_rejected = False

    final_allow = 0 if actually_rejected else int(signal_allow)
    row = result.to_log_row(ts, symbol, actually_rejected=actually_rejected)
    row.update(
        {
            "case": case,
            "input_allow": int(signal_allow),
            "final_allow": final_allow,
            "USE_XGBOOST_SIGNAL": int(bool(use_xgboost_signal)),
            "XGBOOST_SIGNAL_BLOCKING": int(bool(xgboost_signal_blocking)),
        }
    )
    return row


def build_verification_rows(artifact: Path, base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    """Build deterministic verification cases for the XGBoost gate."""
    confirmed = _loaded_confirmer(_StaticXGBoostModel(0.10, 0.90), artifact, "verify-confirmed")
    low_confidence = _loaded_confirmer(_StaticXGBoostModel(0.45, 0.55), artifact, "verify-low-confidence")
    direction_mismatch = _loaded_confirmer(_StaticXGBoostModel(0.90, 0.10), artifact, "verify-direction-mismatch")
    error_confirmer = _loaded_confirmer(_ErrorXGBoostModel(), artifact, "verify-model-error")
    missing_confirmer = _missing_confirmer(artifact, base_dir)

    return [
        simulate_live_writer_decision(
            case="high_confidence_direction_agreement_allows_signal",
            confirmer=confirmed,
            use_xgboost_signal=True,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="low_confidence_rejects_signal_when_blocking",
            confirmer=low_confidence,
            use_xgboost_signal=True,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="direction_mismatch_rejects_signal_when_blocking",
            confirmer=direction_mismatch,
            use_xgboost_signal=True,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="missing_artifact_does_not_reject",
            confirmer=missing_confirmer,
            use_xgboost_signal=True,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="model_error_does_not_reject",
            confirmer=error_confirmer,
            use_xgboost_signal=True,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="use_xgboost_signal_false_never_rejects",
            confirmer=direction_mismatch,
            use_xgboost_signal=False,
            xgboost_signal_blocking=True,
        ),
        simulate_live_writer_decision(
            case="xgboost_signal_blocking_false_never_rejects",
            confirmer=direction_mismatch,
            use_xgboost_signal=True,
            xgboost_signal_blocking=False,
        ),
    ]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    except Exception:
        return None


def _latest_non_empty(rows: List[Dict[str, Any]], key: str) -> str:
    for row in reversed(rows):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def _latest_float(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(rows):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def summarize_decisions(rows: List[Dict[str, Any]], artifact: Path) -> Dict[str, Any]:
    rejected_rows = [row for row in rows if bool(int(row.get("actually_rejected") or 0))]
    total = len(rows)
    rejected = len(rejected_rows)
    allowed = sum(1 for row in rows if int(row.get("final_allow") or 0) == 1)
    top_reject_reasons = Counter((row.get("reject_reason") or row.get("reason") or "unknown") for row in rejected_rows)
    return {
        "total_signals_checked": total,
        "allowed_count": allowed,
        "rejected_count": rejected,
        "reject_rate": 0.0 if total == 0 else rejected / total,
        "top_reject_reasons": dict(top_reject_reasons.most_common(5)),
        "latest_confidence": _latest_float(rows, "xgboost_confidence"),
        "latest_direction": _latest_non_empty(rows, "xgboost_direction"),
        "artifact_path": str(artifact),
        "model_version": _latest_non_empty(rows, "model_version"),
    }


def validate_required_cases(rows: List[Dict[str, Any]]) -> List[str]:
    by_case = {row["case"]: row for row in rows}
    errors: List[str] = []

    def check(case: str, condition: bool, detail: str) -> None:
        if case not in by_case:
            errors.append(f"{case}: missing")
        elif not condition:
            errors.append(f"{case}: {detail}")

    confirmed = by_case.get("high_confidence_direction_agreement_allows_signal", {})
    check(
        "high_confidence_direction_agreement_allows_signal",
        confirmed.get("final_allow") == 1
        and confirmed.get("actually_rejected") == 0
        and confirmed.get("would_confirm") == 1,
        (
            "expected allow=1 actually_rejected=0 would_confirm=1 "
            f"got allow={confirmed.get('final_allow')} rejected={confirmed.get('actually_rejected')} "
            f"would_confirm={confirmed.get('would_confirm')}"
        ),
    )

    low_conf = by_case.get("low_confidence_rejects_signal_when_blocking", {})
    check(
        "low_confidence_rejects_signal_when_blocking",
        low_conf.get("final_allow") == 0
        and low_conf.get("actually_rejected") == 1
        and low_conf.get("reject_reason") == "low_confidence",
        (
            "expected allow=0 actually_rejected=1 reject_reason=low_confidence "
            f"got allow={low_conf.get('final_allow')} rejected={low_conf.get('actually_rejected')} "
            f"reason={low_conf.get('reject_reason')}"
        ),
    )

    mismatch = by_case.get("direction_mismatch_rejects_signal_when_blocking", {})
    check(
        "direction_mismatch_rejects_signal_when_blocking",
        mismatch.get("final_allow") == 0
        and mismatch.get("actually_rejected") == 1
        and mismatch.get("reject_reason") == "direction_mismatch",
        (
            "expected allow=0 actually_rejected=1 reject_reason=direction_mismatch "
            f"got allow={mismatch.get('final_allow')} rejected={mismatch.get('actually_rejected')} "
            f"reason={mismatch.get('reject_reason')}"
        ),
    )

    missing = by_case.get("missing_artifact_does_not_reject", {})
    check(
        "missing_artifact_does_not_reject",
        missing.get("final_allow") == 1
        and missing.get("actually_rejected") == 0
        and missing.get("xgboost_status") == "disabled_missing_artifact",
        (
            "expected allow=1 rejected=0 status=disabled_missing_artifact "
            f"got allow={missing.get('final_allow')} rejected={missing.get('actually_rejected')} "
            f"status={missing.get('xgboost_status')}"
        ),
    )

    model_error = by_case.get("model_error_does_not_reject", {})
    check(
        "model_error_does_not_reject",
        model_error.get("final_allow") == 1
        and model_error.get("actually_rejected") == 0
        and model_error.get("xgboost_status") == "model_error",
        (
            "expected allow=1 rejected=0 status=model_error "
            f"got allow={model_error.get('final_allow')} rejected={model_error.get('actually_rejected')} "
            f"status={model_error.get('xgboost_status')}"
        ),
    )

    use_false = by_case.get("use_xgboost_signal_false_never_rejects", {})
    check(
        "use_xgboost_signal_false_never_rejects",
        use_false.get("final_allow") == 1
        and use_false.get("actually_rejected") == 0
        and use_false.get("USE_XGBOOST_SIGNAL") == 0,
        (
            "expected allow=1 rejected=0 USE_XGBOOST_SIGNAL=0 "
            f"got allow={use_false.get('final_allow')} rejected={use_false.get('actually_rejected')} "
            f"flag={use_false.get('USE_XGBOOST_SIGNAL')}"
        ),
    )

    blocking_false = by_case.get("xgboost_signal_blocking_false_never_rejects", {})
    check(
        "xgboost_signal_blocking_false_never_rejects",
        blocking_false.get("final_allow") == 1
        and blocking_false.get("actually_rejected") == 0
        and blocking_false.get("would_reject") == 1
        and blocking_false.get("XGBOOST_SIGNAL_BLOCKING") == 0,
        (
            "expected allow=1 rejected=0 would_reject=1 XGBOOST_SIGNAL_BLOCKING=0 "
            f"got allow={blocking_false.get('final_allow')} rejected={blocking_false.get('actually_rejected')} "
            f"would_reject={blocking_false.get('would_reject')} "
            f"flag={blocking_false.get('XGBOOST_SIGNAL_BLOCKING')}"
        ),
    )

    return errors


def probe_artifact(artifact: Path) -> Dict[str, Any]:
    """Load the configured artifact through the production loader and probe it."""
    if not artifact.exists():
        return {
            "artifact_status": "missing",
            "artifact_model_version": "",
            "artifact_confidence": None,
            "artifact_direction": "",
            "artifact_reason": "artifact_missing",
        }

    logs: List[str] = []
    with _temporary_env({"XGBOOST_SIGNAL_ARTIFACT": str(artifact)}):
        confirmer = XGBoostSignalConfirmer.from_env(enabled=True, base_dir=BASE_DIR, log_fn=logs.append)

    if not confirmer.ready:
        return {
            "artifact_status": confirmer.xgboost_status,
            "artifact_model_version": confirmer.model_version,
            "artifact_confidence": None,
            "artifact_direction": "",
            "artifact_reason": confirmer.reason,
        }

    result = confirmer.evaluate(
        symbol="ARTIFACT_PROBE",
        window=_sample_window(confirmer.model),
        existing_signal="LONG",
        existing_score=0.25,
        rv_mean=0.0,
        price=0.0,
    )
    return {
        "artifact_status": confirmer.xgboost_status,
        "artifact_model_version": confirmer.model_version,
        "artifact_confidence": result.xgboost_confidence,
        "artifact_direction": result.xgboost_direction,
        "artifact_reason": result.reason,
    }


def format_summary(rows: List[Dict[str, Any]], summary: Dict[str, Any], artifact_probe: Dict[str, Any]) -> str:
    lines = [
        "XGBoost Signal Blocking Verification",
        "Environment:",
        "  USE_XGBOOST_SIGNAL=true",
        "  XGBOOST_SIGNAL_BLOCKING=true",
        f"  XGBOOST_SIGNAL_ARTIFACT={summary['artifact_path']}",
        "",
        "Cases:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['case']}: final_allow={row['final_allow']} "
            f"actually_rejected={row['actually_rejected']} "
            f"would_reject={row['would_reject']} "
            f"status={row['xgboost_status']} "
            f"reason={row['reason']} "
            f"reject_reason={row['reject_reason']}"
        )

    lines.extend(
        [
            "",
            "Summary:",
            f"  total_signals_checked: {summary['total_signals_checked']}",
            f"  allowed_count: {summary['allowed_count']}",
            f"  rejected_count: {summary['rejected_count']}",
            f"  reject_rate: {summary['reject_rate']:.2%}",
            f"  top_reject_reasons: {summary['top_reject_reasons']}",
            f"  latest_confidence: {summary['latest_confidence']}",
            f"  latest_direction: {summary['latest_direction']}",
            f"  artifact_path: {summary['artifact_path']}",
            f"  model_version: {summary['model_version']}",
            f"  artifact_status: {artifact_probe['artifact_status']}",
            f"  artifact_probe_model_version: {artifact_probe['artifact_model_version']}",
            f"  artifact_probe_confidence: {artifact_probe['artifact_confidence']}",
            f"  artifact_probe_direction: {artifact_probe['artifact_direction']}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser("Verify optional XGBoost blocking gate")
    parser.add_argument("--artifact", default=os.getenv("XGBOOST_SIGNAL_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--shadow-log", default=str(BASE_DIR / "logs" / "xgboost_signal_shadow.csv"))
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--missing-artifact-check", action="store_true")
    args = parser.parse_args()

    artifact = _resolve(args.artifact)

    if args.missing_artifact_check:
        missing = _missing_artifact_path(artifact)
        with _temporary_env({"XGBOOST_SIGNAL_ARTIFACT": str(missing)}):
            confirmer = XGBoostSignalConfirmer.from_env(enabled=True, base_dir=BASE_DIR)
        print(f"artifact_path={missing}")
        print(f"xgboost_status={confirmer.xgboost_status}")
        print(f"reason={confirmer.reason}")
        if confirmer.xgboost_status != "disabled_missing_artifact":
            raise SystemExit("ERROR missing artifact check did not resolve to disabled_missing_artifact")
        return

    os.environ["USE_XGBOOST_SIGNAL"] = "true"
    os.environ["XGBOOST_SIGNAL_BLOCKING"] = "true"
    os.environ["XGBOOST_SIGNAL_ARTIFACT"] = str(artifact)

    rows = build_verification_rows(artifact)
    for row in rows:
        missing = [c for c in XGBOOST_SHADOW_COLS if c not in row]
        if missing:
            raise SystemExit(f"ERROR shadow row missing columns: {missing}")

    errors = validate_required_cases(rows)
    if errors:
        raise SystemExit("ERROR verification failed:\n" + "\n".join(f"- {err}" for err in errors))

    if args.write_shadow_row:
        for row in rows:
            _append_row(_resolve(args.shadow_log), row)

    artifact_probe = probe_artifact(artifact)
    if artifact_probe["artifact_status"] != "loaded":
        raise SystemExit(
            "ERROR XGBoost artifact did not load: "
            f"status={artifact_probe['artifact_status']} reason={artifact_probe['artifact_reason']} "
            f"artifact={artifact}"
        )

    summary = summarize_decisions(rows, artifact)
    if artifact_probe.get("artifact_model_version"):
        summary["model_version"] = artifact_probe["artifact_model_version"]
    if artifact_probe.get("artifact_confidence") is not None:
        summary["latest_confidence"] = artifact_probe["artifact_confidence"]
    if artifact_probe.get("artifact_direction"):
        summary["latest_direction"] = artifact_probe["artifact_direction"]

    print(format_summary(rows, summary, artifact_probe))


if __name__ == "__main__":
    main()
