"""Train the optional Isolation Forest anomaly filter.

This script is separate from live inference and does not change the deployed DL
feature contract. It trains on summary vectors derived from the existing live
feature windows and saves a joblib artifact for shadow-mode use.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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

from data import load_prices_and_features
from ml_optional.isolation_filter import DEFAULT_ARTIFACT, window_to_isolation_vector


def _symbols(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


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

    syms = _symbols(args.symbols)
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
        "seq_len": args.seq_len,
        "timeframe": args.timeframe,
        "symbols": syms,
        "n_features": int(X.shape[1]),
        "n_windows": int(X.shape[0]),
        "contamination": float(args.contamination),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    joblib.dump(artifact, out)
    print(f"[isolation] saved {out} windows={X.shape[0]} features={X.shape[1]}")


if __name__ == "__main__":
    main()

