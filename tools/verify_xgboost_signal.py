"""Verify the optional XGBoost signal-confirmation shadow artifact."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
)

SHADOW_LOG = BASE_DIR / "logs" / "xgboost_signal_shadow.csv"


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


def main() -> int:
    parser = argparse.ArgumentParser("Verify optional XGBoost shadow artifact")
    parser.add_argument("--artifact", default=os.getenv("XGBOOST_SIGNAL_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--symbol", default="VERIFY")
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--missing-artifact-check", action="store_true")
    args = parser.parse_args()

    if args.missing_artifact_check:
        os.environ["XGBOOST_SIGNAL_ARTIFACT"] = "model_artifacts/__missing_xgboost_verify__.joblib"
    else:
        os.environ["XGBOOST_SIGNAL_ARTIFACT"] = args.artifact

    artifact_path = _resolve_path(os.environ["XGBOOST_SIGNAL_ARTIFACT"])
    logs: list[str] = []
    confirmer = XGBoostSignalConfirmer.from_env(
        enabled=True,
        base_dir=BASE_DIR,
        log_fn=logs.append,
    )
    for msg in logs:
        print(msg)
    print(f"artifact_path={artifact_path}")
    print(f"artifact_exists={artifact_path.exists()}")
    print(f"xgboost_status={confirmer.xgboost_status}")

    if args.missing_artifact_check:
        return 0 if confirmer.xgboost_status == "disabled_missing_artifact" else 1

    if not artifact_path.exists():
        return 1
    if not confirmer.ready:
        return 2

    result = confirmer.evaluate(
        symbol=args.symbol,
        window=_sample_window(confirmer.model),
        existing_signal="LONG",
        existing_score=0.25,
        rv_mean=0.0,
        price=0.0,
    )
    row = result.to_log_row(_ts(), args.symbol)
    print(
        "prediction_ok=true "
        f"xgboost_direction={result.xgboost_direction} "
        f"xgboost_confidence={result.xgboost_confidence} "
        f"would_confirm={int(result.would_confirm)} "
        f"would_reject={int(result.would_reject)} "
        f"reason={result.reason}"
    )
    if args.write_shadow_row:
        _append_aligned_row(SHADOW_LOG, XGBOOST_SHADOW_COLS, row)
        print(f"shadow_log_written={SHADOW_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
