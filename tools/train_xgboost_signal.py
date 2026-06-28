"""Train the optional XGBoost signal-confirmation shadow artifact.

This script is separate from live inference and does not change the deployed DL
feature contract. The default CSV path is logs/live_signals.csv, which only
provides weak labels from the existing signal direction. Artifacts trained that
way are marked shadow_verification and are not production-ready.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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

from ml_optional.xgboost_signal import (
    DEFAULT_ARTIFACT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    window_to_xgboost_vector,
)


def _truthy(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(0.0, index=df.index, dtype=float)


def infer_live_feature_width() -> int:
    """Infer current live feature width without modifying the feature set."""
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

    return len(canonical_feature_columns(_truthy(os.getenv("DL_ADD_SYMBOL_ID", "1"))))


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

    blocked = {
        "ts",
        "timestamp",
        "symbol",
        "side_hint",
        "kinds_used",
        "mode",
        "reason",
        "artifact_path",
        "model_version",
    }
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


def _labels_from_df(df: pd.DataFrame, label_col: str) -> Tuple[pd.Series, str]:
    if label_col and label_col in df.columns:
        source = label_col
        raw = df[label_col]
    elif "side_hint" in df.columns:
        source = "side_hint"
        raw = df["side_hint"]
    elif "p_meta" in df.columns:
        source = "p_meta_sign"
        raw = df["p_meta"]
    else:
        raise RuntimeError("CSV needs side_hint, p_meta, or an explicit --label-col")

    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        return numeric.apply(lambda v: 1 if v > 0 else (0 if v < 0 else np.nan)), source

    def _map_label(value: object) -> float:
        text = str(value).strip().upper()
        if text in {"LONG", "BUY", "BULL", "1", "1.0", "TRUE"}:
            return 1.0
        if text in {"SHORT", "SELL", "BEAR", "0", "0.0", "-1", "-1.0", "FALSE"}:
            return 0.0
        return np.nan

    return raw.apply(_map_label), source


def build_training_matrix_from_csv(
    path: Path,
    feature_cols: Iterable[str],
    seq_len: int,
    step: int,
    target_width: int,
    symbol_col: str,
    time_col: str,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    df = pd.read_csv(path)
    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing requested feature columns: {missing}")
    if time_col in df.columns:
        df = df.sort_values([symbol_col, time_col] if symbol_col in df.columns else [time_col])
    elif symbol_col in df.columns:
        df = df.sort_values([symbol_col])

    labels, label_source = _labels_from_df(df, label_col)
    rows: List[np.ndarray] = []
    y: List[int] = []
    groups = df.groupby(symbol_col, sort=False) if symbol_col in df.columns else [(None, df)]
    for _sym, group in groups:
        group_labels = labels.loc[group.index]
        X_df = group[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = X_df.notna().all(axis=1) & group_labels.notna()
        X_df = X_df.loc[valid]
        group_labels = group_labels.loc[valid].astype(int)
        if len(X_df) < seq_len:
            continue

        values = _resize_columns(X_df.values.astype(np.float32, copy=False), target_width)
        score_values = _numeric_series(group, "p_meta").loc[valid].fillna(0.0).values
        rv_values = _numeric_series(group, "rv_mean").loc[valid].fillna(0.0).values
        price_values = _numeric_series(group, "px").loc[valid].fillna(0.0).values
        label_values = group_labels.values.astype(int, copy=False)
        for end in range(seq_len, len(values) + 1, max(1, step)):
            idx = end - 1
            rows.append(
                window_to_xgboost_vector(
                    values[end - seq_len:end],
                    existing_score=float(score_values[idx]),
                    rv_mean=float(rv_values[idx]),
                    price=float(price_values[idx]),
                ).reshape(-1)
            )
            y.append(int(label_values[idx]))

    if not rows:
        raise RuntimeError(
            f"no CSV training windows built from {path}; rows may be fewer than seq_len={seq_len}"
        )
    return np.vstack(rows).astype(np.float32, copy=False), np.asarray(y, dtype=np.int64), label_source


def main() -> None:
    parser = argparse.ArgumentParser("Train optional XGBoost signal-confirmation shadow artifact")
    parser.add_argument(
        "--input-csv",
        default=os.getenv("XGBOOST_TRAIN_INPUT_CSV", "logs/live_signals.csv"),
        help="CSV source. Default is logs/live_signals.csv for shadow verification.",
    )
    parser.add_argument(
        "--csv-feature-cols",
        default=os.getenv("XGBOOST_CSV_FEATURE_COLS", ""),
        help="Comma-separated numeric columns. Default: auto-detect numeric columns.",
    )
    parser.add_argument("--csv-symbol-col", default=os.getenv("XGBOOST_CSV_SYMBOL_COL", "symbol"))
    parser.add_argument("--csv-time-col", default=os.getenv("XGBOOST_CSV_TIME_COL", "ts"))
    parser.add_argument("--label-col", default=os.getenv("XGBOOST_LABEL_COL", ""))
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("DL_SEQ_LEN", "64")))
    parser.add_argument("--step", type=int, default=int(os.getenv("XGBOOST_TRAIN_STEP", "1")))
    parser.add_argument(
        "--feature-width",
        default=os.getenv("XGBOOST_FEATURE_WIDTH", "auto"),
        help="Live feature width before summary expansion. 'auto' uses scaler metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.getenv("XGBOOST_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))),
    )
    parser.add_argument("--estimators", type=int, default=int(os.getenv("XGBOOST_N_ESTIMATORS", "200")))
    parser.add_argument("--max-depth", type=int, default=int(os.getenv("XGBOOST_MAX_DEPTH", "3")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("XGBOOST_LEARNING_RATE", "0.05")))
    parser.add_argument("--subsample", type=float, default=float(os.getenv("XGBOOST_SUBSAMPLE", "0.8")))
    parser.add_argument("--colsample-bytree", type=float, default=float(os.getenv("XGBOOST_COLSAMPLE_BYTREE", "0.8")))
    parser.add_argument("--out", default=os.getenv("XGBOOST_SIGNAL_ARTIFACT", DEFAULT_ARTIFACT))
    args = parser.parse_args()

    if importlib.util.find_spec("xgboost") is None:
        raise SystemExit(
            "xgboost package is not installed. Install xgboost to train a real XGBoost shadow artifact."
        )

    source_path = _resolve_path(args.input_csv)
    if not source_path.exists():
        raise SystemExit(f"input CSV not found: {source_path}")

    import joblib
    from xgboost import XGBClassifier

    target_width = infer_live_feature_width() if str(args.feature_width).lower() == "auto" else int(args.feature_width)
    feature_cols = _csv_numeric_columns(source_path, args.csv_feature_cols)
    X, y, label_source = build_training_matrix_from_csv(
        source_path,
        feature_cols,
        args.seq_len,
        args.step,
        target_width,
        args.csv_symbol_col,
        args.csv_time_col,
        args.label_col,
    )
    classes = sorted(set(int(v) for v in y.tolist()))
    if classes != [0, 1]:
        raise SystemExit(f"training labels must contain both SHORT=0 and LONG=1; found {classes}")

    model = XGBClassifier(
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(os.getenv("SEED", "42")),
        n_jobs=-1,
    )
    model.fit(X, y)

    out = _resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_version": datetime.now(timezone.utc).strftime("xgboost_shadow_%Y%m%d_%H%M%S"),
        "feature_builder": "window_to_xgboost_vector:v1",
        "model_family": "xgboost",
        "source": "csv_shadow_verification",
        "source_path": str(source_path),
        "label_source": label_source,
        "production_ready": False,
        "confidence_threshold": float(args.confidence_threshold),
        "csv_feature_cols": feature_cols,
        "live_feature_width": int(target_width),
        "n_features": int(X.shape[1]),
        "n_windows": int(X.shape[0]),
        "class_balance": {str(c): int((y == c).sum()) for c in classes},
    }
    joblib.dump(artifact, out)

    print(f"saved {out}")
    print(
        "source=csv_shadow_verification production_ready=false "
        f"label_source={label_source} n_windows={X.shape[0]} n_features={X.shape[1]}"
    )


if __name__ == "__main__":
    main()
