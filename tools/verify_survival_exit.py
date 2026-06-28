"""Verify the optional Survival Analysis exit-timing shadow artifact."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

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
    if not artifact_path.exists():
        return 1
    if not model.ready:
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
        _append_aligned_row(SHADOW_LOG, SURVIVAL_SHADOW_COLS, row)
        print(f"shadow_log_written={SHADOW_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
