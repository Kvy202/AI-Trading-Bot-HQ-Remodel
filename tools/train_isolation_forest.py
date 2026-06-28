"""Train the optional Isolation Forest anomaly filter.

This script is separate from live inference and does not change the deployed DL
feature contract. It trains on summary vectors derived from the existing live
feature windows and saves a joblib artifact for shadow-mode use.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

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

from data import load_prices_and_features
from ml_optional.isolation_filter import DEFAULT_ARTIFACT, window_to_isolation_vector


def _symbols(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def infer_live_feature_width() -> int:
    """Infer the current live feature width without modifying the feature set."""
    try:
        import joblib

        preferred = [
            BASE_DIR / "model_artifacts" / "scaler_latest.joblib",
            BASE_DIR / "model_artifacts" / "scaler_tcn_latest.joblib",
            BASE_DIR / "model_artifacts" / "scaler_lstm_latest.joblib",
            BASE_DIR / "model_artifacts" / "scaler_tx_latest.joblib",
        ]
        candidates = preferred + sorted((BASE_DIR / "model_artifacts").glob("scaler_*latest*.joblib"))
        for path in candidates:
            if not path.exists():
                continue
            scaler = joblib.load(path)
            n = int(getattr(scaler, "n_features_in_", 0) or 0)
            if n > 0:
                return n
    except Exception:
        pass

    from features import canonical_feature_columns

    add_symbol_id = _truthy(os.getenv("DL_ADD_SYMBOL_ID", "1"))
    return len(canonical_feature_columns(add_symbol_id))


def _csv_numeric_columns(path: Path, explicit: str = "") -> List[str]:
    if explicit.strip():
        return [c.strip() for c in explicit.split(",") if c.strip()]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= 200:
                break
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    blocked = {"ts", "timestamp", "symbol", "side_hint", "kinds_used", "mode", "reason", "artifact_path", "model_version"}
    cols: List[str] = []
    for col in rows[0].keys():
        if col is None or col.strip().lower() in blocked:
            continue
        vals = []
        for row in rows:
            raw = (row.get(col) or "").strip()
            if raw == "":
                continue
            try:
                vals.append(float(raw))
            except Exception:
                vals = []
                break
        if vals:
            cols.append(col)
    if not cols:
        raise RuntimeError(f"CSV has no numeric feature columns: {path}")
    return cols


def _resize_columns(values: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return values
    if values.shape[1] == target_width:
        return values
    if values.shape[1] > target_width:
        return values[:, :target_width]
    pad = np.zeros((values.shape[0], target_width - values.shape[1]), dtype=values.dtype)
    return np.concatenate([values, pad], axis=1)


def build_training_matrix_from_csv(
    path: Path,
    feature_cols: Iterable[str],
    seq_len: int,
    step: int,
    target_width: int,
    symbol_col: str,
    time_col: str,
) -> np.ndarray:
    df = pd.read_csv(path)
    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing requested feature columns: {missing}")
    if time_col in df.columns:
        df = df.sort_values([symbol_col, time_col] if symbol_col in df.columns else [time_col])
    elif symbol_col in df.columns:
        df = df.sort_values([symbol_col])

    rows = []
    groups = df.groupby(symbol_col, sort=False) if symbol_col in df.columns else [(None, df)]
    for _sym, group in groups:
        X = group[cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) < seq_len:
            continue
        values = _resize_columns(X.values.astype(np.float32, copy=False), target_width)
        for end in range(seq_len, len(values) + 1, max(1, step)):
            rows.append(window_to_isolation_vector(values[end - seq_len:end]).reshape(-1))
    if not rows:
        raise RuntimeError(
            f"no CSV training windows built from {path}; rows may be fewer than seq_len={seq_len}"
        )
    return np.vstack(rows).astype(np.float32, copy=False)


def build_training_matrix(
    symbols: List[str],
    timeframe: str,
    lookback: int,
    seq_len: int,
    step: int,
    add_symbol_id: bool,
) -> np.ndarray:
    X_live, _dfs, lengths = load_prices_and_features(
        symbols=symbols,
        timeframe=timeframe,
        lookback=lookback,
        add_symbol_id=add_symbol_id,
        return_dfs=True,
        return_symbol_lengths=True,
    )
    rows = []
    offset = 0
    for slen in lengths:
        block = X_live[offset:offset + slen, :]
        offset += slen
        if len(block) < seq_len:
            continue
        for end in range(seq_len, len(block) + 1, max(1, step)):
            rows.append(window_to_isolation_vector(block[end - seq_len:end]).reshape(-1))
    if not rows:
        raise RuntimeError("no training windows built; increase lookback or reduce seq_len")
    return np.vstack(rows).astype(np.float32, copy=False)


def main() -> None:
    p = argparse.ArgumentParser("Train optional Isolation Forest shadow artifact")
    p.add_argument("--input-csv", default=os.getenv("ISOLATION_TRAIN_INPUT_CSV", ""),
                   help="Optional local CSV source. Use logs/live_signals.csv for shadow verification.")
    p.add_argument("--csv-feature-cols", default=os.getenv("ISOLATION_CSV_FEATURE_COLS", ""),
                   help="Comma-separated numeric columns. Default: auto-detect numeric columns.")
    p.add_argument("--csv-symbol-col", default=os.getenv("ISOLATION_CSV_SYMBOL_COL", "symbol"))
    p.add_argument("--csv-time-col", default=os.getenv("ISOLATION_CSV_TIME_COL", "ts"))
    p.add_argument("--feature-width", default=os.getenv("ISOLATION_FEATURE_WIDTH", "auto"),
                   help="Live feature width before summary expansion. 'auto' uses scaler metadata.")
    p.add_argument("--symbols", default=os.getenv("DL_SYMBOLS", os.getenv("SYMBOL_WHITELIST", "BTCUSDT,ETHUSDT")))
    p.add_argument("--timeframe", default=os.getenv("DL_TIMEFRAME", os.getenv("TIMEFRAME", "1m")))
    p.add_argument("--lookback", type=int, default=int(os.getenv("ISOLATION_TRAIN_LOOKBACK", "8000")))
    p.add_argument("--seq-len", type=int, default=int(os.getenv("DL_SEQ_LEN", "64")))
    p.add_argument("--step", type=int, default=int(os.getenv("ISOLATION_TRAIN_STEP", "5")))
    p.add_argument("--contamination", type=float, default=float(os.getenv("ISOLATION_CONTAMINATION", "0.02")))
    p.add_argument("--estimators", type=int, default=int(os.getenv("ISOLATION_N_ESTIMATORS", "200")))
    p.add_argument("--add-symbol-id", action="store_true",
                   default=os.getenv("DL_ADD_SYMBOL_ID", "1").strip().lower() in {"1", "true", "yes", "y", "on"})
    p.add_argument("--out", default=os.getenv("ISOLATION_FOREST_ARTIFACT", DEFAULT_ARTIFACT))
    args = p.parse_args()

    from sklearn.ensemble import IsolationForest
    import joblib

    target_width = infer_live_feature_width() if str(args.feature_width).lower() == "auto" else int(args.feature_width)
    source = "live_features"
    source_path: Optional[Path] = None
    feature_cols: List[str] = []
    syms = _symbols(args.symbols)

    if args.input_csv.strip():
        source = "csv_shadow_verification"
        source_path = _resolve_path(args.input_csv)
        feature_cols = _csv_numeric_columns(source_path, args.csv_feature_cols)
        X = build_training_matrix_from_csv(
            source_path,
            feature_cols,
            args.seq_len,
            args.step,
            target_width,
            args.csv_symbol_col,
            args.csv_time_col,
        )
    else:
        X = build_training_matrix(syms, args.timeframe, args.lookback, args.seq_len, args.step, args.add_symbol_id)

    model = IsolationForest(
        n_estimators=args.estimators,
        contamination=args.contamination,
        random_state=int(os.getenv("SEED", "42")),
        n_jobs=-1,
    )
    model.fit(X)

    out = Path(args.out)
    if not out.is_absolute():
        out = BASE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_version": datetime.now(timezone.utc).strftime("iforest_%Y%m%d_%H%M%S"),
        "feature_builder": "window_to_isolation_vector:v1",
        "source": source,
        "source_path": "" if source_path is None else str(source_path),
        "csv_feature_cols": feature_cols,
        "live_feature_width": int(target_width),
        "seq_len": args.seq_len,
        "timeframe": args.timeframe,
        "symbols": syms,
        "n_features": int(X.shape[1]),
        "n_windows": int(X.shape[0]),
        "contamination": float(args.contamination),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    joblib.dump(artifact, out)
    print(
        f"[isolation] saved {out} source={source} windows={X.shape[0]} "
        f"features={X.shape[1]} live_feature_width={target_width}"
    )


if __name__ == "__main__":
    main()
