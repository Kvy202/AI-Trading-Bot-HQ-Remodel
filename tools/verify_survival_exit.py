"""Verify the optional Survival Analysis exit-timing shadow artifact."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

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

from ml_optional.survival_exit import (  # noqa: E402
    DEFAULT_ARTIFACT,
    SURVIVAL_SHADOW_COLS,
    SurvivalExitModel,
    survival_active_exit_decision,
)

SHADOW_LOG = BASE_DIR / "logs" / "survival_exit_shadow.csv"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _ensure_header(path: Path, cols: Iterable[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(cols))


def _append_aligned_row(path: Path, cols: list[str], row: Dict[str, Any]) -> None:
    _ensure_header(path, cols)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in cols])


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


class _StaticSurvivalModel:
    def __init__(self, risk: float) -> None:
        self.risk = risk

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray([self.risk], dtype=float)


class _ErrorSurvivalModel:
    def predict(self, x: Any) -> np.ndarray:
        raise RuntimeError("verification model error")


def _loaded_model(model: Any, artifact: Path, version: str = "verify") -> SurvivalExitModel:
    return SurvivalExitModel(
        enabled=True,
        artifact_path=artifact,
        model=model,
        model_version=version,
        risk_threshold=0.60,
        survival_status="loaded",
        reason="loaded",
    )


def _disabled_model(artifact: Path, status: str, reason: str) -> SurvivalExitModel:
    return SurvivalExitModel(
        enabled=False,
        artifact_path=artifact,
        survival_status=status,
        reason=reason,
    )


def _missing_model(artifact: Path, base_dir: Path) -> SurvivalExitModel:
    missing = artifact.parent / f"__missing_{artifact.stem or 'survival'}_verify__.joblib"
    with _temporary_env({"SURVIVAL_EXIT_ARTIFACT": str(missing)}):
        return SurvivalExitModel.from_env(enabled=True, base_dir=base_dir)


def simulate_active_decision(
    *,
    case: str,
    model: SurvivalExitModel,
    survival_active: bool,
    paper_mode: bool,
    place_real_orders: bool,
    symbol: str = "VERIFY",
) -> Dict[str, Any]:
    result = model.evaluate(
        symbol=symbol,
        side="long",
        trade_id="verify-trade",
        entry_time="2026-06-28 00:00:00+0000",
        current_age_seconds=900.0,
        current_unrealized_pnl=-0.02,
        entry_price=100.0,
        current_price=99.0,
        qty=0.1,
    )
    decision = survival_active_exit_decision(
        result,
        survival_active=survival_active,
        paper_mode=paper_mode,
        place_real_orders=place_real_orders,
    )
    row = result.to_log_row(
        _ts(),
        symbol,
        actually_exited=decision.should_exit,
        exit_reason=decision.exit_reason,
        survival_active=decision.survival_active,
        paper_only_guard=decision.paper_only_guard,
    )
    row.update(
        {
            "case": case,
            "SURVIVAL_EXIT_ACTIVE": int(bool(survival_active)),
            "paper_mode": int(bool(paper_mode)),
            "place_real_orders": int(bool(place_real_orders)),
        }
    )
    return row


def build_verification_rows(artifact: Path, base_dir: Path = BASE_DIR) -> List[Dict[str, Any]]:
    high_risk = _loaded_model(_StaticSurvivalModel(0.82), artifact, "verify-high-risk")
    low_risk = _loaded_model(_StaticSurvivalModel(0.20), artifact, "verify-low-risk")
    model_error = _loaded_model(_ErrorSurvivalModel(), artifact, "verify-model-error")
    missing_artifact = _missing_model(artifact, base_dir)
    missing_dependency = _disabled_model(
        artifact,
        "disabled_missing_dependency",
        "missing_dependency:lifelines",
    )

    return [
        simulate_active_decision(
            case="high_risk_exits_when_active_paper",
            model=high_risk,
            survival_active=True,
            paper_mode=True,
            place_real_orders=False,
        ),
        simulate_active_decision(
            case="active_false_never_exits",
            model=high_risk,
            survival_active=False,
            paper_mode=True,
            place_real_orders=False,
        ),
        simulate_active_decision(
            case="low_risk_never_exits",
            model=low_risk,
            survival_active=True,
            paper_mode=True,
            place_real_orders=False,
        ),
        simulate_active_decision(
            case="missing_artifact_never_exits",
            model=missing_artifact,
            survival_active=True,
            paper_mode=True,
            place_real_orders=False,
        ),
        simulate_active_decision(
            case="model_error_never_exits",
            model=model_error,
            survival_active=True,
            paper_mode=True,
            place_real_orders=False,
        ),
        simulate_active_decision(
            case="real_live_mode_never_exits",
            model=high_risk,
            survival_active=True,
            paper_mode=False,
            place_real_orders=True,
        ),
        simulate_active_decision(
            case="dependency_missing_never_exits",
            model=missing_dependency,
            survival_active=True,
            paper_mode=True,
            place_real_orders=False,
        ),
    ]


def validate_required_cases(rows: List[Dict[str, Any]]) -> List[str]:
    by_case = {row["case"]: row for row in rows}
    errors: List[str] = []

    def check(case: str, condition: bool, detail: str) -> None:
        if case not in by_case:
            errors.append(f"{case}: missing")
        elif not condition:
            errors.append(f"{case}: {detail}")

    high = by_case.get("high_risk_exits_when_active_paper", {})
    check(
        "high_risk_exits_when_active_paper",
        high.get("actually_exited") == 1
        and high.get("exit_reason") == "survival_high_exit_risk"
        and high.get("paper_only_guard") == "paper_only_ok",
        f"expected actual exit in paper mode got {high}",
    )

    inactive = by_case.get("active_false_never_exits", {})
    check(
        "active_false_never_exits",
        inactive.get("actually_exited") == 0
        and inactive.get("paper_only_guard") == "inactive",
        f"expected inactive no-exit got {inactive}",
    )

    low = by_case.get("low_risk_never_exits", {})
    check(
        "low_risk_never_exits",
        low.get("actually_exited") == 0 and low.get("would_exit_early") == 0,
        f"expected low-risk no-exit got {low}",
    )

    missing = by_case.get("missing_artifact_never_exits", {})
    check(
        "missing_artifact_never_exits",
        missing.get("actually_exited") == 0
        and missing.get("survival_status") == "disabled_missing_artifact",
        f"expected missing artifact no-exit got {missing}",
    )

    error = by_case.get("model_error_never_exits", {})
    check(
        "model_error_never_exits",
        error.get("actually_exited") == 0
        and error.get("survival_status") == "prediction_error",
        f"expected model error no-exit got {error}",
    )

    live = by_case.get("real_live_mode_never_exits", {})
    check(
        "real_live_mode_never_exits",
        live.get("actually_exited") == 0
        and live.get("paper_only_guard") == "blocked_real_orders",
        f"expected live-mode no-exit got {live}",
    )

    dependency = by_case.get("dependency_missing_never_exits", {})
    check(
        "dependency_missing_never_exits",
        dependency.get("actually_exited") == 0
        and dependency.get("survival_status") == "disabled_missing_dependency",
        f"expected dependency-missing no-exit got {dependency}",
    )

    return errors


def format_verification_summary(rows: List[Dict[str, Any]], artifact_path: Path, artifact_status: str) -> str:
    lines = [
        "Survival Exit Active Verification",
        f"artifact_path={artifact_path}",
        f"artifact_status={artifact_status}",
        "",
        "Cases:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['case']}: actually_exited={row['actually_exited']} "
            f"would_exit_early={row['would_exit_early']} "
            f"survival_status={row['survival_status']} "
            f"paper_only_guard={row['paper_only_guard']} "
            f"reason={row['reason']} "
            f"exit_reason={row['exit_reason']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser("Verify optional Survival Analysis exit shadow artifact")
    parser.add_argument("--artifact", default=os.getenv("SURVIVAL_EXIT_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--symbol", default="VERIFY")
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--missing-artifact-check", action="store_true")
    args = parser.parse_args()

    if args.missing_artifact_check:
        os.environ["SURVIVAL_EXIT_ARTIFACT"] = "model_artifacts/__missing_survival_verify__.joblib"
    else:
        os.environ["SURVIVAL_EXIT_ARTIFACT"] = args.artifact

    artifact_path = _resolve_path(os.environ["SURVIVAL_EXIT_ARTIFACT"])
    logs: list[str] = []
    model = SurvivalExitModel.from_env(
        enabled=True,
        base_dir=BASE_DIR,
        log_fn=logs.append,
    )
    for msg in logs:
        print(msg)
    print(f"artifact_path={artifact_path}")
    print(f"artifact_exists={artifact_path.exists()}")
    print(f"survival_status={model.survival_status}")

    if args.missing_artifact_check:
        return 0 if model.survival_status == "disabled_missing_artifact" else 1

    rows = build_verification_rows(artifact_path)
    errors = validate_required_cases(rows)
    if errors:
        raise SystemExit("ERROR verification failed:\n" + "\n".join(f"- {err}" for err in errors))

    if not artifact_path.exists():
        print(format_verification_summary(rows, artifact_path, "missing"))
        return 1
    if not model.ready:
        print(format_verification_summary(rows, artifact_path, model.survival_status))
        return 2

    result = model.evaluate(
        symbol=args.symbol,
        side="long",
        trade_id="verify-trade",
        entry_time="2026-06-28 00:00:00+0000",
        current_age_seconds=900.0,
        current_unrealized_pnl=-0.02,
        entry_price=100.0,
        current_price=99.0,
        qty=0.1,
    )
    row = result.to_log_row(_ts(), args.symbol)
    print(
        "prediction_ok=true "
        f"survival_risk_score={result.survival_risk_score} "
        f"estimated_time_to_exit={result.estimated_time_to_exit} "
        f"would_hold={int(result.would_hold)} "
        f"would_exit_early={int(result.would_exit_early)} "
        f"reason={result.reason}"
    )
    if args.write_shadow_row:
        for verify_row in rows:
            _append_aligned_row(SHADOW_LOG, SURVIVAL_SHADOW_COLS, verify_row)
        _append_aligned_row(SHADOW_LOG, SURVIVAL_SHADOW_COLS, row)
        print(f"shadow_log_written={SHADOW_LOG}")
    print(format_verification_summary(rows, artifact_path, model.survival_status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
