"""Verify the optional Isolation Forest blocking gate in paper-test mode.

This script does not run market data, the executor, or strategy logic. It
simulates the live_writer Isolation Forest gate with deterministic inputs so the
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

from ml_optional.isolation_filter import (
    DEFAULT_ARTIFACT,
    ISOLATION_SHADOW_COLS,
    IsolationFilter,
    should_block_entry,
)


class _StaticIsolationModel:
    n_features_in_ = 12

    def __init__(self, pred: int, score: float) -> None:
        self.pred = int(pred)
        self.score = float(score)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.pred])

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.score])


class _ErrorIsolationModel:
    n_features_in_ = 12

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise RuntimeError("verification model error")

    def decision_function(self, x: np.ndarray) -> np.ndarray:
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
        writer = csv.DictWriter(f, fieldnames=ISOLATION_SHADOW_COLS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in ISOLATION_SHADOW_COLS})


def _sample_window(model: Any) -> np.ndarray:
    n_features = int(getattr(model, "n_features_in_", 0) or 0)
    if n_features <= 0:
        raise RuntimeError("loaded model does not expose n_features_in_")
    live_width = n_features // 4 if n_features % 4 == 0 else n_features
    base = np.linspace(-0.5, 0.5, live_width, dtype=np.float32)
    return np.vstack([base * 0.25, base * 0.5, base * 0.75, base]).astype(np.float32)


def _missing_artifact_path(artifact: Path) -> Path:
    parent = artifact.parent if artifact.parent != Path("") else BASE_DIR
    stem = artifact.stem or "isolation_forest"
    candidate = parent / f"__missing_{stem}_blocking_verify__.joblib"
    idx = 1
    while candidate.exists():
        candidate = parent / f"__missing_{stem}_blocking_verify_{idx}.joblib"
        idx += 1
    return candidate


def _loaded_filter(model: Any, artifact: Path, model_version: str) -> IsolationFilter:
    return IsolationFilter(
        enabled=True,
        artifact_path=artifact,
        model=model,
        model_version=model_version,
        isolation_status="loaded",
    )


def _missing_filter(artifact: Path, base_dir: Path) -> IsolationFilter:
    missing = _missing_artifact_path(artifact)
    with _temporary_env({"ISOLATION_FOREST_ARTIFACT": str(missing)}):
        return IsolationFilter.from_env(enabled=True, base_dir=base_dir)



def simulate_live_writer_decision(
    *,
    case: str,
    isolation_filter: IsolationFilter,
    use_isolation_forest: bool,
    isolation_forest_blocking: bool,
    symbol: str = "VERIFY",
    signal_allow: int = 1,
    ts: str = "VERIFY_TS",
) -> Dict[str, Any]:
    """Simulate the live_writer Isolation Forest allow-gate branch."""
    window = _sample_window(isolation_filter.model) if isolation_filter.model is not None else None
    if use_isolation_forest:
        result = isolation_filter.evaluate(symbol, window)
        actually_blocked = bool(signal_allow and should_block_entry(result, isolation_forest_blocking))
    else:
        disabled = IsolationFilter(enabled=False, artifact_path=isolation_filter.artifact_path)
        result = disabled.evaluate(symbol, window)
        actually_blocked = False

    final_allow = 0 if actually_blocked else int(signal_allow)
    row = result.to_log_row(ts, symbol, actually_blocked=actually_blocked)
    row.update(
        {
            "case": case,
            "input_allow": int(signal_allow),
            "final_allow": final_allow,
            "USE_ISOLATION_FOREST": int(bool(use_isolation_forest)),
            "ISOLATION_FOREST_BLOCKING": int(bool(isolation_forest_blocking)),
        }
    )
    return row


def build_verification_rows(artifact: Path, base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    """Build the required deterministic verification cases."""
    normal_filter = _loaded_filter(
        _StaticIsolationModel(pred=1, score=0.18),
        artifact,
        "verify-normal",
    )
    abnormal_filter = _loaded_filter(
        _StaticIsolationModel(pred=-1, score=-0.42),
        artifact,
        "verify-anomaly",
    )
    error_filter = _loaded_filter(
        _ErrorIsolationModel(),
        artifact,
        "verify-model-error",
    )
    missing_filter = _missing_filter(artifact, base_dir)

    return [
        simulate_live_writer_decision(
            case="normal_prediction_allows_signal",
            isolation_filter=normal_filter,
            use_isolation_forest=True,
            isolation_forest_blocking=True,
        ),
        simulate_live_writer_decision(
            case="abnormal_prediction_blocks_signal",
            isolation_filter=abnormal_filter,
            use_isolation_forest=True,
            isolation_forest_blocking=True,
        ),
        simulate_live_writer_decision(
            case="missing_artifact_does_not_block",
            isolation_filter=missing_filter,
            use_isolation_forest=True,
            isolation_forest_blocking=True,
        ),
        simulate_live_writer_decision(
            case="model_error_does_not_block",
            isolation_filter=error_filter,
            use_isolation_forest=True,
            isolation_forest_blocking=True,
        ),
        simulate_live_writer_decision(
            case="use_isolation_forest_false_never_blocks",
            isolation_filter=abnormal_filter,
            use_isolation_forest=False,
            isolation_forest_blocking=True,
        ),
        simulate_live_writer_decision(
            case="isolation_forest_blocking_false_never_blocks",
            isolation_filter=abnormal_filter,
            use_isolation_forest=True,
            isolation_forest_blocking=False,
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
    blocked_rows = [row for row in rows if bool(int(row.get("actually_blocked") or 0))]
    total = len(rows)
    blocked = len(blocked_rows)
    allowed = sum(1 for row in rows if int(row.get("final_allow") or 0) == 1)
    top_block_reasons = Counter((row.get("reason") or "unknown") for row in blocked_rows)
    return {
        "total_signals_checked": total,
        "allowed_count": allowed,
        "blocked_count": blocked,
        "block_rate": 0.0 if total == 0 else blocked / total,
        "top_block_reasons": dict(top_block_reasons.most_common(5)),
        "latest_anomaly_score": _latest_float(rows, "anomaly_score"),
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

    normal = by_case.get("normal_prediction_allows_signal", {})
    check(
        "normal_prediction_allows_signal",
        normal.get("final_allow") == 1 and normal.get("actually_blocked") == 0,
        f"expected allow=1 blocked=0 got allow={normal.get('final_allow')} blocked={normal.get('actually_blocked')}",
    )

    abnormal = by_case.get("abnormal_prediction_blocks_signal", {})
    check(
        "abnormal_prediction_blocks_signal",
        abnormal.get("final_allow") == 0 and abnormal.get("actually_blocked") == 1,
        f"expected allow=0 blocked=1 got allow={abnormal.get('final_allow')} blocked={abnormal.get('actually_blocked')}",
    )

    missing = by_case.get("missing_artifact_does_not_block", {})
    check(
        "missing_artifact_does_not_block",
        missing.get("final_allow") == 1
        and missing.get("actually_blocked") == 0
        and missing.get("isolation_status") == "disabled_missing_artifact",
        (
            "expected allow=1 blocked=0 status=disabled_missing_artifact "
            f"got allow={missing.get('final_allow')} blocked={missing.get('actually_blocked')} "
            f"status={missing.get('isolation_status')}"
        ),
    )

    model_error = by_case.get("model_error_does_not_block", {})
    check(
        "model_error_does_not_block",
        model_error.get("final_allow") == 1
        and model_error.get("actually_blocked") == 0
        and model_error.get("isolation_status") == "model_error",
        (
            "expected allow=1 blocked=0 status=model_error "
            f"got allow={model_error.get('final_allow')} blocked={model_error.get('actually_blocked')} "
            f"status={model_error.get('isolation_status')}"
        ),
    )

    use_false = by_case.get("use_isolation_forest_false_never_blocks", {})
    check(
        "use_isolation_forest_false_never_blocks",
        use_false.get("final_allow") == 1
        and use_false.get("actually_blocked") == 0
        and use_false.get("USE_ISOLATION_FOREST") == 0,
        (
            "expected allow=1 blocked=0 USE_ISOLATION_FOREST=0 "
            f"got allow={use_false.get('final_allow')} blocked={use_false.get('actually_blocked')} "
            f"flag={use_false.get('USE_ISOLATION_FOREST')}"
        ),
    )

    blocking_false = by_case.get("isolation_forest_blocking_false_never_blocks", {})
    check(
        "isolation_forest_blocking_false_never_blocks",
        blocking_false.get("final_allow") == 1
        and blocking_false.get("actually_blocked") == 0
        and blocking_false.get("would_block") == 1
        and blocking_false.get("ISOLATION_FOREST_BLOCKING") == 0,
        (
            "expected allow=1 blocked=0 would_block=1 ISOLATION_FOREST_BLOCKING=0 "
            f"got allow={blocking_false.get('final_allow')} blocked={blocking_false.get('actually_blocked')} "
            f"would_block={blocking_false.get('would_block')} "
            f"flag={blocking_false.get('ISOLATION_FOREST_BLOCKING')}"
        ),
    )

    return errors


def probe_artifact(artifact: Path) -> Dict[str, Any]:
    """Optionally load the configured artifact through the production loader."""
    if not artifact.exists():
        return {
            "artifact_status": "missing",
            "artifact_model_version": "",
            "artifact_anomaly_score": None,
            "artifact_reason": "artifact_missing",
        }

    logs: List[str] = []
    with _temporary_env({"ISOLATION_FOREST_ARTIFACT": str(artifact)}):
        flt = IsolationFilter.from_env(enabled=True, base_dir=BASE_DIR, log_fn=logs.append)

    if not flt.ready:
        return {
            "artifact_status": flt.isolation_status,
            "artifact_model_version": flt.model_version,
            "artifact_anomaly_score": None,
            "artifact_reason": flt.reason,
        }

    result = flt.evaluate("ARTIFACT_PROBE", _sample_window(flt.model))
    return {
        "artifact_status": result.isolation_status,
        "artifact_model_version": result.model_version,
        "artifact_anomaly_score": result.anomaly_score,
        "artifact_reason": result.reason,
    }


def format_summary(rows: List[Dict[str, Any]], summary: Dict[str, Any], artifact_probe: Dict[str, Any]) -> str:
    lines = [
        "Isolation Forest Blocking Verification",
        "Environment:",
        "  USE_ISOLATION_FOREST=true",
        "  ISOLATION_FOREST_BLOCKING=true",
        f"  ISOLATION_FOREST_ARTIFACT={summary['artifact_path']}",
        "",
        "Cases:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['case']}: final_allow={row['final_allow']} "
            f"actually_blocked={row['actually_blocked']} "
            f"would_block={row['would_block']} "
            f"status={row['isolation_status']} "
            f"reason={row['reason']}"
        )

    lines.extend(
        [
            "",
            "Summary:",
            f"  total_signals_checked: {summary['total_signals_checked']}",
            f"  allowed_count: {summary['allowed_count']}",
            f"  blocked_count: {summary['blocked_count']}",
            f"  block_rate: {summary['block_rate']:.2%}",
            f"  top_block_reasons: {summary['top_block_reasons']}",
            f"  latest_anomaly_score: {summary['latest_anomaly_score']}",
            f"  artifact_path: {summary['artifact_path']}",
            f"  model_version: {summary['model_version']}",
            f"  artifact_status: {artifact_probe['artifact_status']}",
            f"  artifact_probe_model_version: {artifact_probe['artifact_model_version']}",
            f"  artifact_probe_anomaly_score: {artifact_probe['artifact_anomaly_score']}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser("Verify optional Isolation Forest blocking gate")
    parser.add_argument("--artifact", default=os.getenv("ISOLATION_FOREST_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--shadow-log", default=str(BASE_DIR / "logs" / "isolation_forest_shadow.csv"))
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--missing-artifact-check", action="store_true", help="Kept for compatibility; always checked.")
    args = parser.parse_args()

    artifact = _resolve(args.artifact)

    os.environ["USE_ISOLATION_FOREST"] = "true"
    os.environ["ISOLATION_FOREST_BLOCKING"] = "true"
    os.environ["ISOLATION_FOREST_ARTIFACT"] = str(artifact)

    rows = build_verification_rows(artifact)
    for row in rows:
        missing = [c for c in ISOLATION_SHADOW_COLS if c not in row]
        if missing:
            raise SystemExit(f"ERROR shadow row missing columns: {missing}")

    errors = validate_required_cases(rows)
    if errors:
        raise SystemExit("ERROR verification failed:\n" + "\n".join(f"- {err}" for err in errors))

    if args.write_shadow_row:
        for row in rows:
            _append_row(_resolve(args.shadow_log), row)

    artifact_probe = probe_artifact(artifact)
    summary = summarize_decisions(rows, artifact)
    if artifact_probe.get("artifact_model_version"):
        summary["model_version"] = artifact_probe["artifact_model_version"]
    if artifact_probe.get("artifact_anomaly_score") is not None:
        summary["latest_anomaly_score"] = artifact_probe["artifact_anomaly_score"]

    print(format_summary(rows, summary, artifact_probe))


if __name__ == "__main__":
    main()

