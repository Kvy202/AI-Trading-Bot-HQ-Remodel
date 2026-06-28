"""Train the optional Survival Analysis exit-timing shadow artifact.

The repository's local closed-trade logs are small and do not preserve full
entry timestamps. This script therefore prefers paper trade logs, reconstructs
entry/exit durations where possible, and always marks the default artifact as
csv_shadow_verification and not production-ready.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    DEFAULT_RISK_HORIZON_MINUTES,
    DEFAULT_RISK_THRESHOLD,
    SURVIVAL_DEPENDENCY,
)


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _parse_ts(value: object) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _num(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _extract_reason_value(reason: object, name: str) -> float:
    match = re.search(rf"\b{name}=([-+]?\d+(?:\.\d+)?)", str(reason or ""))
    return float(match.group(1)) if match else 0.0


def _side_from_entry_action(action: object) -> str:
    text = str(action or "").strip().upper()
    if text == "BUY":
        return "long"
    if text == "SELL_SHORT":
        return "short"
    return ""


def _is_entry(row: pd.Series) -> bool:
    return _side_from_entry_action(row.get("side")) != "" and "ENTRY" in str(row.get("reason", "")).upper()


def _is_close(row: pd.Series) -> bool:
    text = str(row.get("side", "")).strip().upper()
    return text in {"SELL", "BUY_TO_COVER"}


def paper_logs_to_training_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    open_by_symbol: Dict[str, pd.Series] = {}
    for path in sorted(paths):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ts" not in df.columns or "symbol" not in df.columns:
            continue
        df["_ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["_ts"]).sort_values("_ts")
        for _idx, row in df.iterrows():
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            if _is_entry(row):
                open_by_symbol[sym] = row
                continue
            if not _is_close(row) or sym not in open_by_symbol:
                continue
            entry = open_by_symbol.pop(sym)
            duration = (row["_ts"] - entry["_ts"]).total_seconds() / 60.0
            if not np.isfinite(duration) or duration <= 0:
                continue
            entry_price = _num(entry.get("price"))
            exit_price = _num(row.get("price"))
            qty = _num(row.get("qty"))
            pnl = _extract_reason_value(row.get("reason"), "pnl")
            side = _side_from_entry_action(entry.get("side"))
            rows.append(
                {
                    "duration_minutes": float(duration),
                    "event": 1.0,
                    "side_long": 1.0 if side == "long" else 0.0,
                    "side_short": 1.0 if side == "short" else 0.0,
                    "age_minutes": float(duration),
                    "age_hours": float(duration) / 60.0,
                    "qty": qty,
                    "entry_price": entry_price,
                    "current_price": exit_price,
                    "price_return": ((exit_price - entry_price) / entry_price) if entry_price > 0 else 0.0,
                    "unrealized_pnl": pnl,
                    "abs_unrealized_pnl": abs(pnl),
                    "symbol": sym,
                    "source_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def explicit_csv_to_training_rows(
    path: Path,
    *,
    duration_col: str,
    event_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    if duration_col not in df.columns:
        raise RuntimeError(f"CSV missing duration column: {duration_col}")
    if event_col not in df.columns:
        raise RuntimeError(f"CSV missing event column: {event_col}")
    if not feature_cols:
        blocked = {duration_col, event_col, "ts", "timestamp", "symbol", "reason", "side", "source_file"}
        for col in df.columns:
            if col in blocked:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                feature_cols.append(col)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing feature columns: {missing}")
    out = pd.DataFrame(
        {
            "duration_minutes": pd.to_numeric(df[duration_col], errors="coerce"),
            "event": pd.to_numeric(df[event_col], errors="coerce"),
        }
    )
    for col in feature_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out, feature_cols


def _default_paper_paths() -> List[Path]:
    return sorted((BASE_DIR / "logs").glob("trades_paper*.csv"))


def _default_closed_paths() -> List[Path]:
    return sorted((BASE_DIR / "logs").glob("trades_closed*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser("Train optional Survival Analysis exit shadow artifact")
    parser.add_argument("--input-csv", default=os.getenv("SURVIVAL_TRAIN_INPUT_CSV", ""))
    parser.add_argument("--duration-col", default=os.getenv("SURVIVAL_DURATION_COL", "duration_minutes"))
    parser.add_argument("--event-col", default=os.getenv("SURVIVAL_EVENT_COL", "event"))
    parser.add_argument("--csv-feature-cols", default=os.getenv("SURVIVAL_CSV_FEATURE_COLS", ""))
    parser.add_argument("--out", default=os.getenv("SURVIVAL_EXIT_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--risk-threshold", type=float, default=float(os.getenv("SURVIVAL_EXIT_RISK_THRESHOLD", str(DEFAULT_RISK_THRESHOLD))))
    parser.add_argument("--risk-horizon-minutes", type=float, default=float(os.getenv("SURVIVAL_EXIT_RISK_HORIZON_MINUTES", str(DEFAULT_RISK_HORIZON_MINUTES))))
    parser.add_argument("--min-samples", type=int, default=int(os.getenv("SURVIVAL_MIN_SAMPLES", "100")))
    args = parser.parse_args()

    if importlib.util.find_spec(SURVIVAL_DEPENDENCY) is None:
        raise SystemExit(f"{SURVIVAL_DEPENDENCY} package is not installed. Install it to train a Survival shadow artifact.")

    import joblib
    from lifelines import CoxPHFitter

    feature_cols = [c.strip() for c in args.csv_feature_cols.split(",") if c.strip()]
    if args.input_csv:
        source_path = _resolve_path(args.input_csv)
        train_df, feature_cols = explicit_csv_to_training_rows(
            source_path,
            duration_col=args.duration_col,
            event_col=args.event_col,
            feature_cols=feature_cols,
        )
        source_files = [str(source_path)]
    else:
        train_df = paper_logs_to_training_rows(_default_paper_paths())
        source_files = [str(p) for p in _default_paper_paths()] + [str(p) for p in _default_closed_paths()]
        if train_df.empty:
            raise SystemExit("no paired paper trade entries/exits found; pass --input-csv with duration/event columns")
        feature_cols = [
            "side_long",
            "side_short",
            "age_minutes",
            "age_hours",
            "unrealized_pnl",
            "abs_unrealized_pnl",
            "entry_price",
            "current_price",
            "price_return",
            "qty",
        ]

    cols = ["duration_minutes", "event"] + feature_cols
    train_df = train_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if train_df.empty:
        raise SystemExit("no valid survival training rows after cleaning")
    if train_df["event"].nunique() < 1:
        raise SystemExit("event column has no usable values")

    cph = CoxPHFitter(penalizer=float(os.getenv("SURVIVAL_COX_PENALIZER", "0.1")))
    cph.fit(train_df, duration_col="duration_minutes", event_col="event")

    out = _resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    production_ready = False
    artifact = {
        "model": cph,
        "model_version": datetime.now(timezone.utc).strftime("survival_shadow_%Y%m%d_%H%M%S"),
        "model_family": "cox_ph",
        "dependency": SURVIVAL_DEPENDENCY,
        "source": "csv_shadow_verification",
        "source_files": source_files,
        "production_ready": production_ready,
        "risk_threshold": float(args.risk_threshold),
        "risk_horizon_minutes": float(args.risk_horizon_minutes),
        "feature_cols": feature_cols,
        "n_samples": int(len(train_df)),
        "n_events": int(pd.to_numeric(train_df["event"], errors="coerce").fillna(0).sum()),
        "duration_minutes_summary": {
            "min": float(train_df["duration_minutes"].min()),
            "median": float(train_df["duration_minutes"].median()),
            "max": float(train_df["duration_minutes"].max()),
        },
        "notes": (
            "Shadow verification artifact only. Local samples are small/weak and "
            "reconstructed from executor paper/closed CSV logs."
        ),
    }
    if len(train_df) < args.min_samples:
        artifact["small_sample_warning"] = f"n_samples={len(train_df)} below min_samples={args.min_samples}"
    joblib.dump(artifact, out)
    print(f"saved {out}")
    print(
        "source=csv_shadow_verification production_ready=false "
        f"n_samples={len(train_df)} n_events={artifact['n_events']} "
        f"feature_cols={','.join(feature_cols)}"
    )


if __name__ == "__main__":
    main()
