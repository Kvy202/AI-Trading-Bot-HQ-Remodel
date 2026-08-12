"""Offline, research-only fixed-horizon signal walk-forward evaluation.

This module deliberately does not import candidate-training, candidate-selection,
runtime, exchange, or market-data code.  It reads two caller-supplied frozen raw
OHLCV windows, derives the repository's canonical features, evaluates frozen
simple-model configurations, and writes evidence only below
``reports/model_signal_research``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features import (  # noqa: E402  (repository root is added above)
    FEATURE_COLS,
    SYMBOL_ID_COL,
    build_features,
    canonical_feature_columns,
)


SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = "model-signal-walkforward-v1"
BAR_INTERVAL = pd.Timedelta(minutes=5)
BAR_INTERVAL_NS = int(BAR_INTERVAL.value)
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
RAW_COLUMNS = ("bar_open_utc", *OHLCV_COLUMNS)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SYMBOL_ID_MAP = {"BTCUSDT": 0, "ETHUSDT": 1}
HORIZONS = (6, 12, 24, 60)
FOLD_SPEC = {
    "training_days": 120,
    "test_days": 30,
    "step_days": 30,
    "bar_interval_minutes": 5,
    "anchor": "combined_raw_start_utc",
    "train_window": "rolling",
    "test_window": "strictly_later_chronological",
}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "pipeline": ["StandardScaler", "LogisticRegression"],
        "standard_scaler": {"with_mean": True, "with_std": True},
        "logistic_regression": {
            "C": 1.0,
            "class_weight": None,
            "fit_intercept": True,
            "max_iter": 1000,
            "penalty": "l2",
            "random_state": 1729,
            "solver": "lbfgs",
            "tol": 0.0001,
        },
        "rankable": True,
    },
    "hist_gradient_boosting": {
        "estimator": "HistGradientBoostingClassifier",
        "hist_gradient_boosting": {
            "early_stopping": False,
            "l2_regularization": 0.0,
            "learning_rate": 0.1,
            "max_bins": 255,
            "max_depth": None,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 1729,
            "tol": 1e-7,
        },
        "rankable": True,
    },
    "train_prevalence_baseline": {
        "estimator": "constant probability equal to training positive prevalence",
        "threshold": 0.5,
        "rankable": False,
    },
}

DEFAULT_EARLIER_DATASET = (
    BASE_DIR / "reports" / "model_training_datasets" / "phase24_5m_d0d635ff3f6a"
)
DEFAULT_LATER_DATASET = (
    BASE_DIR / "reports" / "model_training_datasets" / "phase24_5m_cac0baf0b726"
)
RESEARCH_ROOT = BASE_DIR / "reports" / "model_signal_research"


class SignalResearchError(ValueError):
    """Raised when an offline research contract fails closed."""


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: str
    horizon_bars: int
    training_window_start: pd.Timestamp
    training_window_end_exclusive: pd.Timestamp
    fit_train_end_exclusive: pd.Timestamp
    purge_start: pd.Timestamp
    purge_end_exclusive: pd.Timestamp
    test_start: pd.Timestamp
    test_end_exclusive: pd.Timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "horizon_bars": self.horizon_bars,
            "horizon_minutes": self.horizon_bars * 5,
            "training_window_start_utc": canonical_utc(self.training_window_start),
            "training_window_end_exclusive_utc": canonical_utc(
                self.training_window_end_exclusive
            ),
            "fit_train_end_exclusive_utc": canonical_utc(self.fit_train_end_exclusive),
            "purge_start_utc": canonical_utc(self.purge_start),
            "purge_end_exclusive_utc": canonical_utc(self.purge_end_exclusive),
            "purge_bars": self.horizon_bars,
            "test_start_utc": canonical_utc(self.test_start),
            "test_end_exclusive_utc": canonical_utc(self.test_end_exclusive),
        }


def canonical_utc(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise SignalResearchError("timestamp must be finite")
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    return timestamp.isoformat().replace("+00:00", "Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return canonical_utc(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    cleaned = _clean_json(value)
    text = json.dumps(
        cleaned,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_raw_file(path: Path | str) -> pd.DataFrame:
    """Read one frozen raw CSV without mutating or normalizing it in place."""

    raw_path = Path(path)
    if not raw_path.is_file():
        raise SignalResearchError(f"missing frozen raw file: {raw_path}")
    frame = pd.read_csv(raw_path)
    if tuple(frame.columns) != RAW_COLUMNS:
        raise SignalResearchError(
            f"unexpected raw schema in {raw_path}: expected {list(RAW_COLUMNS)}"
        )
    try:
        timestamps = pd.to_datetime(frame["bar_open_utc"], utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise SignalResearchError(f"invalid UTC timestamp in {raw_path}") from exc
    values = frame.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise SignalResearchError(f"nonfinite OHLCV value in {raw_path}")
    if (values.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        raise SignalResearchError(f"nonpositive price in {raw_path}")
    if (values["volume"] < 0).any():
        raise SignalResearchError(f"negative volume in {raw_path}")
    if (values["high"] < values[["open", "close", "low"]].max(axis=1)).any():
        raise SignalResearchError(f"invalid high price in {raw_path}")
    if (values["low"] > values[["open", "close", "high"]].min(axis=1)).any():
        raise SignalResearchError(f"invalid low price in {raw_path}")
    values.index = pd.DatetimeIndex(timestamps, name="bar_open_utc")
    return values


def combine_raw_windows(
    earlier_path: Path | str,
    later_path: Path | str,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Combine contiguous frozen windows and reject any ambiguity or gap."""

    if symbol not in SYMBOLS:
        raise SignalResearchError(f"unsupported research symbol: {symbol}")
    earlier = read_raw_file(earlier_path)
    later = read_raw_file(later_path)
    combined = pd.concat([earlier, later], axis=0).sort_index(kind="mergesort")
    duplicated = combined.index.duplicated(keep=False)
    if bool(duplicated.any()):
        overlap = combined.loc[duplicated]
        conflicts: list[pd.Timestamp] = []
        for timestamp, group in overlap.groupby(level=0, sort=True):
            matrix = group.loc[:, OHLCV_COLUMNS].to_numpy(dtype=np.float64)
            if not np.array_equal(matrix, np.repeat(matrix[[0]], len(matrix), axis=0)):
                conflicts.append(pd.Timestamp(timestamp))
        if conflicts:
            raise SignalResearchError(
                f"conflicting overlapping bar for {symbol}: {canonical_utc(conflicts[0])}"
            )
        first_duplicate = overlap.index[0]
        raise SignalResearchError(
            f"duplicate timestamp for {symbol}: {canonical_utc(first_duplicate)}"
        )
    if not combined.index.is_monotonic_increasing or not combined.index.is_unique:
        raise SignalResearchError(f"non-monotonic raw timestamps for {symbol}")
    if len(combined) < 2:
        raise SignalResearchError(f"insufficient raw rows for {symbol}")
    deltas = np.diff(combined.index.asi8)
    bad = np.flatnonzero(deltas != BAR_INTERVAL_NS)
    if len(bad):
        position = int(bad[0])
        raise SignalResearchError(
            "unexpected 5-minute gap for "
            f"{symbol}: {canonical_utc(combined.index[position])} -> "
            f"{canonical_utc(combined.index[position + 1])}"
        )
    return combined


def build_research_features(
    raw: pd.DataFrame,
    *,
    symbol: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Derive the canonical feature matrix without altering its implementation."""

    before = tuple(FEATURE_COLS)
    if symbol not in SYMBOL_ID_MAP:
        raise SignalResearchError(f"unsupported research symbol: {symbol}")
    derived = build_features(raw.copy())
    if tuple(FEATURE_COLS) != before:
        raise SignalResearchError("canonical FEATURE_COLS ordering changed during research")
    ordered = canonical_feature_columns(True)
    if len(FEATURE_COLS) != 26 or len(ordered) != 27 or SYMBOL_ID_COL not in ordered:
        raise SignalResearchError("canonical 26-market-feature plus symbol_id contract failed")
    derived = derived.copy()
    derived[SYMBOL_ID_COL] = float(SYMBOL_ID_MAP[symbol])
    matrix = derived.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(matrix.to_numpy(dtype=np.float64)).all(axis=1)
    matrix = matrix.loc[finite].astype(np.float64)
    if matrix.empty:
        raise SignalResearchError(f"no finite canonical feature rows for {symbol}")
    diagnostics = {
        "raw_rows": int(len(raw)),
        "feature_rows_before_finite_filter": int(len(derived)),
        "finite_feature_rows": int(len(matrix)),
        "warmup_or_nan_rows_excluded": int(len(raw) - len(derived)),
        "nonfinite_feature_rows_excluded": int((~finite).sum()),
    }
    return matrix, diagnostics


def build_fixed_horizon_rows(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    *,
    symbol: str,
    horizon_bars: int,
) -> pd.DataFrame:
    """Attach close[t+h]-based targets to time-t canonical features."""

    if horizon_bars <= 0:
        raise SignalResearchError("horizon must be positive")
    close = raw["close"].astype(np.float64)
    future_log_return = np.log(close.shift(-horizon_bars)) - np.log(close)
    aligned_return = future_log_return.reindex(features.index)
    usable = np.isfinite(aligned_return.to_numpy(dtype=np.float64))
    rows = features.loc[usable].copy()
    aligned_return = aligned_return.loc[usable].astype(np.float64)
    rows["future_log_return"] = aligned_return
    rows["target"] = (aligned_return > 0.0).astype(np.int8)
    rows["symbol"] = symbol
    rows["timestamp"] = rows.index
    rows["target_timestamp"] = rows.index + horizon_bars * BAR_INTERVAL
    return rows.reset_index(drop=True)


def make_walkforward_folds(
    history_start: Any,
    history_end_inclusive: Any,
    *,
    horizon_bars: int,
    training_days: int = 120,
    test_days: int = 30,
    step_days: int = 30,
) -> list[FoldDefinition]:
    """Create deterministic rolling folds with an h-bar label purge."""

    start = pd.Timestamp(canonical_utc(history_start))
    end = pd.Timestamp(canonical_utc(history_end_inclusive))
    if horizon_bars <= 0 or min(training_days, test_days, step_days) <= 0:
        raise SignalResearchError("walk-forward durations and horizon must be positive")
    if start.value % BAR_INTERVAL_NS or end.value % BAR_INTERVAL_NS:
        raise SignalResearchError("history bounds must align to five-minute bars")
    training_delta = pd.Timedelta(days=training_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)
    folds: list[FoldDefinition] = []
    fold_number = 0
    while True:
        test_start = start + training_delta + fold_number * step_delta
        test_end = test_start + test_delta
        final_required_target = test_end - BAR_INTERVAL + horizon_bars * BAR_INTERVAL
        if final_required_target > end:
            break
        train_start = test_start - training_delta
        purge_start = test_start - horizon_bars * BAR_INTERVAL
        if purge_start <= train_start:
            raise SignalResearchError("horizon purge consumes the training window")
        folds.append(
            FoldDefinition(
                fold_id=f"fold_{fold_number:02d}",
                horizon_bars=horizon_bars,
                training_window_start=train_start,
                training_window_end_exclusive=test_start,
                fit_train_end_exclusive=purge_start,
                purge_start=purge_start,
                purge_end_exclusive=test_start,
                test_start=test_start,
                test_end_exclusive=test_end,
            )
        )
        fold_number += 1
    if not folds:
        raise SignalResearchError("combined history does not permit a complete fold")
    return folds


def select_fold_rows(
    rows: pd.DataFrame,
    fold: FoldDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (
        (rows["timestamp"] >= fold.training_window_start)
        & (rows["timestamp"] < fold.fit_train_end_exclusive)
    )
    test_mask = (
        (rows["timestamp"] >= fold.test_start)
        & (rows["timestamp"] < fold.test_end_exclusive)
    )
    train = rows.loc[train_mask].sort_values(["timestamp", "symbol"], kind="mergesort")
    test = rows.loc[test_mask].sort_values(["timestamp", "symbol"], kind="mergesort")
    if train.empty or test.empty:
        raise SignalResearchError(f"empty train/test rows for {fold.fold_id}")
    if set(train["symbol"].unique()) != set(SYMBOLS):
        raise SignalResearchError(f"training symbol coverage failed for {fold.fold_id}")
    if set(test["symbol"].unique()) != set(SYMBOLS):
        raise SignalResearchError(f"test symbol coverage failed for {fold.fold_id}")
    max_train_label = pd.Timestamp(train["target_timestamp"].max())
    min_test_time = pd.Timestamp(test["timestamp"].min())
    if max_train_label >= min_test_time:
        raise SignalResearchError(
            f"horizon purge label leakage in {fold.fold_id}: "
            f"{canonical_utc(max_train_label)} >= {canonical_utc(min_test_time)}"
        )
    if pd.Timestamp(train["timestamp"].max()) >= min_test_time:
        raise SignalResearchError(f"train/test chronology failed for {fold.fold_id}")
    return train, test


def make_model(model_name: str) -> Any:
    if model_name == "logistic_regression":
        cfg = MODEL_CONFIGS[model_name]
        return Pipeline(
            steps=[
                ("standard_scaler", StandardScaler(**cfg["standard_scaler"])),
                (
                    "logistic_regression",
                    LogisticRegression(**cfg["logistic_regression"]),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **MODEL_CONFIGS[model_name]["hist_gradient_boosting"]
        )
    if model_name == "train_prevalence_baseline":
        return None
    raise SignalResearchError(f"unknown frozen model configuration: {model_name}")


def fit_model_scores(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, Any]:
    """Fit on training arrays only, then score test features only."""

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int8)
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise SignalResearchError("invalid train/test feature matrix")
    if len(y_train) != len(X_train) or set(np.unique(y_train)) != {0, 1}:
        raise SignalResearchError("training fold must contain both classes")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise SignalResearchError("model input contains nonfinite features")
    if model_name == "train_prevalence_baseline":
        prevalence = float(np.mean(y_train))
        return np.full(len(X_test), prevalence, dtype=np.float64), None
    estimator = make_model(model_name)
    estimator.fit(X_train, y_train)
    probabilities = np.asarray(estimator.predict_proba(X_test), dtype=np.float64)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise SignalResearchError(f"positive class missing from fitted {model_name}")
    return probabilities[:, classes.index(1)], estimator


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _accuracy(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(accuracy_score(y_true, scores >= 0.5))


def _is_constant(values: np.ndarray) -> bool:
    """Detect identical values without a roundoff-sensitive std calculation."""

    return len(values) > 0 and float(np.ptp(values)) == 0.0


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or _is_constant(a) or _is_constant(b):
        return None
    return _finite_or_none(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or _is_constant(a) or _is_constant(b):
        return None
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    result = spearmanr(a, b)
    return _finite_or_none(result.statistic)


def calculate_metrics(test: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(test):
        raise SignalResearchError("prediction/test length mismatch")
    finite = np.isfinite(scores)
    nonfinite = int((~finite).sum())
    y = test.loc[finite, "target"].to_numpy(dtype=np.int8)
    returns = test.loc[finite, "future_log_return"].to_numpy(dtype=np.float64)
    usable_scores = scores[finite]
    if len(y) == 0:
        raise SignalResearchError("no finite test predictions")
    result: dict[str, Any] = {
        "pooled_roc_auc": _roc_auc(y, usable_scores),
        "pooled_accuracy_0_5": _accuracy(y, usable_scores),
        "brier_score": float(brier_score_loss(y, usable_scores)),
        "log_loss": float(log_loss(y, usable_scores, labels=[0, 1])),
        "nonfinite_prediction_count": nonfinite,
        "pearson_score_return_ic": _correlation(usable_scores, returns),
        "spearman_score_return_ic": _spearman(usable_scores, returns),
    }
    for symbol, prefix in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        mask = finite & (test["symbol"].to_numpy() == symbol)
        symbol_y = test.loc[mask, "target"].to_numpy(dtype=np.int8)
        symbol_scores = scores[mask]
        result[f"{prefix}_roc_auc"] = _roc_auc(symbol_y, symbol_scores)
        result[f"{prefix}_accuracy_0_5"] = _accuracy(symbol_y, symbol_scores)
    if _is_constant(usable_scores):
        mean_return = float(np.mean(returns))
        result.update(
            {
                "return_bucket_status": "constant_scores_no_distinct_deciles",
                "prediction_decile_row_count": 0,
                "highest_prediction_decile_mean_future_return": mean_return,
                "lowest_prediction_decile_mean_future_return": mean_return,
                "top_minus_bottom_return_spread_bps": 0.0,
            }
        )
    else:
        bucket_size = max(1, int(math.ceil(len(usable_scores) * 0.10)))
        order = np.argsort(usable_scores, kind="mergesort")
        low_mean = float(np.mean(returns[order[:bucket_size]]))
        high_mean = float(np.mean(returns[order[-bucket_size:]]))
        result.update(
            {
                "return_bucket_status": "distinct_score_rank_deciles",
                "prediction_decile_row_count": bucket_size,
                "highest_prediction_decile_mean_future_return": high_mean,
                "lowest_prediction_decile_mean_future_return": low_mean,
                "top_minus_bottom_return_spread_bps": (high_mean - low_mean) * 10_000.0,
            }
        )
    return result


def calculate_regime_descriptors(
    raw_by_symbol: Mapping[str, pd.DataFrame],
    fold: FoldDefinition,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for symbol, prefix in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        raw = raw_by_symbol[symbol]
        train = raw.loc[
            (raw.index >= fold.training_window_start)
            & (raw.index < fold.fit_train_end_exclusive)
        ]
        test = raw.loc[(raw.index >= fold.test_start) & (raw.index < fold.test_end_exclusive)]
        if len(train) == 0 or len(test) < 2:
            raise SignalResearchError(f"insufficient regime rows for {symbol} {fold.fold_id}")
        close = test["close"].to_numpy(dtype=np.float64)
        intraperiod_log_returns = np.diff(np.log(close))
        signed_return = float(np.log(close[-1]) - np.log(close[0]))
        train_volume_median = float(np.median(train["volume"].to_numpy(dtype=np.float64)))
        volume_ratio = (
            float(np.mean(test["volume"].to_numpy(dtype=np.float64)) / train_volume_median)
            if train_volume_median > 0.0
            else None
        )
        result.update(
            {
                f"{prefix}_regime_realized_volatility": float(
                    np.sqrt(np.sum(np.square(intraperiod_log_returns)))
                ),
                f"{prefix}_regime_signed_period_return": signed_return,
                f"{prefix}_regime_absolute_period_return": abs(signed_return),
                f"{prefix}_regime_mean_volume_to_train_median": volume_ratio,
            }
        )
    return result


def _class_counts(values: pd.Series) -> tuple[int, int]:
    counts = values.value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))


def _mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if _finite_or_none(value) is not None]
    return float(np.mean(finite)) if finite else None


def _median(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if _finite_or_none(value) is not None]
    return float(np.median(finite)) if finite else None


def _minimum(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if _finite_or_none(value) is not None]
    return min(finite) if finite else None


def _maximum(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if _finite_or_none(value) is not None]
    return max(finite) if finite else None


def summarize_stability(fold_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in fold_metrics:
        key = (int(row["horizon_bars"]), str(row["model_name"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (horizon, model_name), rows in sorted(groups.items()):
        auc = [row["pooled_roc_auc"] for row in rows]
        finite_auc = [float(value) for value in auc if _finite_or_none(value) is not None]
        btc_mean = _mean([row["btc_roc_auc"] for row in rows])
        eth_mean = _mean([row["eth_roc_auc"] for row in rows])
        symbol_means = [value for value in (btc_mean, eth_mean) if value is not None]
        spreads = [
            float(row["top_minus_bottom_return_spread_bps"])
            for row in rows
            if _finite_or_none(row["top_minus_bottom_return_spread_bps"]) is not None
        ]
        summaries.append(
            {
                "horizon_bars": horizon,
                "horizon_minutes": horizon * 5,
                "model_name": model_name,
                "rankable": bool(MODEL_CONFIGS[model_name]["rankable"]),
                "number_of_folds": len(rows),
                "mean_pooled_auc": _mean(auc),
                "median_pooled_auc": _median(auc),
                "pooled_auc_standard_deviation": (
                    float(np.std(finite_auc, ddof=0)) if finite_auc else None
                ),
                "minimum_pooled_auc": _minimum(auc),
                "maximum_pooled_auc": _maximum(auc),
                "folds_with_pooled_auc_above_0_50": sum(value > 0.50 for value in finite_auc),
                "folds_with_pooled_auc_above_0_52": sum(value > 0.52 for value in finite_auc),
                "mean_btc_auc": btc_mean,
                "mean_eth_auc": eth_mean,
                "minimum_symbol_mean_auc": min(symbol_means) if len(symbol_means) == 2 else None,
                "median_score_return_ic": _median(
                    [row["pearson_score_return_ic"] for row in rows]
                ),
                "median_spearman_score_return_ic": _median(
                    [row["spearman_score_return_ic"] for row in rows]
                ),
                "median_top_bottom_return_spread_bps": _median(spreads),
                "percentage_folds_positive_top_bottom_spread": (
                    100.0 * sum(value > 0.0 for value in spreads) / len(spreads)
                    if spreads
                    else None
                ),
            }
        )
    return summaries


def rank_stability(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rankable = [row for row in summaries if bool(row["rankable"])]
    required = (
        "median_pooled_auc",
        "minimum_pooled_auc",
        "minimum_symbol_mean_auc",
        "median_top_bottom_return_spread_bps",
    )
    if any(any(_finite_or_none(row[field]) is None for field in required) for row in rankable):
        raise SignalResearchError("ranking field is unavailable")
    ordered = sorted(
        rankable,
        key=lambda row: (
            -float(row["median_pooled_auc"]),
            -float(row["minimum_pooled_auc"]),
            -float(row["minimum_symbol_mean_auc"]),
            -float(row["median_top_bottom_return_spread_bps"]),
            str(row["model_name"]),
            int(row["horizon_bars"]),
        ),
    )
    return [
        {
            "rank": index + 1,
            "horizon_bars": int(row["horizon_bars"]),
            "horizon_minutes": int(row["horizon_minutes"]),
            "model_name": str(row["model_name"]),
            **{field: float(row[field]) for field in required},
        }
        for index, row in enumerate(ordered)
    ]


def _source_metadata(
    dataset_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, str]]:
    metadata: dict[str, Any] = {}
    flat_digests: dict[str, str] = {}
    for role, dataset in zip(("earlier", "later"), dataset_paths):
        entry: dict[str, Any] = {
            "role": role,
            "dataset_id": dataset.name,
            "path": str(dataset),
            "raw_files": {},
        }
        for symbol in SYMBOLS:
            path = dataset / f"raw_{symbol}.csv"
            digest = file_sha256(path)
            frame = read_raw_file(path)
            key = f"{dataset.name}/raw_{symbol}.csv"
            flat_digests[key] = digest
            entry["raw_files"][symbol] = {
                "path": str(path),
                "sha256": digest,
                "row_count": int(len(frame)),
                "first_bar_open_utc": canonical_utc(frame.index[0]),
                "last_bar_open_utc": canonical_utc(frame.index[-1]),
            }
        metadata[role] = entry
    return metadata, flat_digests


def _feature_contract() -> dict[str, Any]:
    features_path = BASE_DIR / "features.py"
    contract = {
        "implementation": "features.build_features",
        "features_py_sha256": file_sha256(features_path),
        "market_feature_count": len(FEATURE_COLS),
        "market_feature_columns": list(FEATURE_COLS),
        "symbol_id_column": SYMBOL_ID_COL,
        "symbol_id_map": dict(SYMBOL_ID_MAP),
        "ordered_model_feature_columns": canonical_feature_columns(True),
        "model_feature_count": len(canonical_feature_columns(True)),
        "derived_from_combined_raw_ohlcv": True,
        "future_information_allowed_in_features": False,
    }
    contract["feature_contract_digest"] = json_digest(contract)
    return contract


def _experiment_contract(
    raw_digests: Mapping[str, str],
    feature_contract: Mapping[str, Any],
    *,
    horizons: Sequence[int],
    fold_spec: Mapping[str, Any],
    model_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "tool_code_sha256": file_sha256(Path(__file__)),
        "raw_source_file_sha256": dict(sorted(raw_digests.items())),
        "feature_contract_digest": feature_contract["feature_contract_digest"],
        "horizons_bars": list(horizons),
        "fold_specification": dict(fold_spec),
        "model_configurations": {name: MODEL_CONFIGS[name] for name in model_names},
        "metrics_contract": {
            "classification_threshold": 0.5,
            "return_unit": "log return",
            "return_spread_unit": "basis points of log return; not PnL",
            "prediction_deciles": "stable score ordering; ceil(10% of finite test rows)",
            "auc_std_ddof": 0,
        },
        "dependency_contract": {
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    }


def _ensure_output_root(output_root: Path) -> Path:
    requested = output_root.resolve()
    allowed = RESEARCH_ROOT.resolve()
    if requested != allowed:
        raise SignalResearchError(
            f"research output root must be exactly {allowed}; received {requested}"
        )
    return requested


def _write_fold_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SignalResearchError("no fold metrics to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else format(value, ".17g")
                    if isinstance(value, float)
                    else value
                    for key, value in row.items()
                }
            )


def run_research(
    earlier_dataset: Path | str = DEFAULT_EARLIER_DATASET,
    later_dataset: Path | str = DEFAULT_LATER_DATASET,
    *,
    output_root: Path | str | None = None,
    horizons: Sequence[int] = HORIZONS,
    training_days: int = 120,
    test_days: int = 30,
    step_days: int = 30,
    model_names: Sequence[str] = tuple(MODEL_CONFIGS),
) -> dict[str, Any]:
    """Run one fully offline research experiment and return its summary."""

    earlier = Path(earlier_dataset).resolve()
    later = Path(later_dataset).resolve()
    if earlier == later:
        raise SignalResearchError("earlier and later datasets must differ")
    if tuple(horizons) != tuple(sorted(set(int(value) for value in horizons))):
        raise SignalResearchError("horizons must be unique and sorted")
    if any(name not in MODEL_CONFIGS for name in model_names):
        raise SignalResearchError("unknown model requested")
    output = _ensure_output_root(Path(output_root) if output_root else RESEARCH_ROOT)
    dataset_paths = (earlier, later)
    source_metadata, initial_raw_digests = _source_metadata(dataset_paths)
    raw_by_symbol = {
        symbol: combine_raw_windows(
            earlier / f"raw_{symbol}.csv",
            later / f"raw_{symbol}.csv",
            symbol=symbol,
        )
        for symbol in SYMBOLS
    }
    if not raw_by_symbol["BTCUSDT"].index.equals(raw_by_symbol["ETHUSDT"].index):
        raise SignalResearchError("BTCUSDT and ETHUSDT raw timestamp grids differ")
    history_start = raw_by_symbol["BTCUSDT"].index[0]
    history_end = raw_by_symbol["BTCUSDT"].index[-1]
    features_by_symbol: dict[str, pd.DataFrame] = {}
    feature_diagnostics: dict[str, Any] = {}
    for symbol in SYMBOLS:
        features_by_symbol[symbol], feature_diagnostics[symbol] = build_research_features(
            raw_by_symbol[symbol], symbol=symbol
        )
    feature_contract = _feature_contract()
    fold_spec = {
        **FOLD_SPEC,
        "training_days": int(training_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "purge_rule": (
            "exclude feature timestamps t >= test_start - horizon*5m; "
            "require every training target timestamp t+h*5m < test_start"
        ),
        "complete_test_rule": (
            "last test feature timestamp plus horizon lookahead must exist in combined raw data"
        ),
    }
    experiment_contract = _experiment_contract(
        initial_raw_digests,
        feature_contract,
        horizons=horizons,
        fold_spec=fold_spec,
        model_names=model_names,
    )
    experiment_digest = json_digest(experiment_contract)
    experiment_id = f"walkforward_{experiment_digest[:16]}"
    final_directory = output / experiment_id
    staging_directory = output / f".{experiment_id}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise SignalResearchError(f"research experiment output already exists: {experiment_id}")

    ordered_features = canonical_feature_columns(True)
    fold_metrics: list[dict[str, Any]] = []
    fold_contracts: dict[str, list[dict[str, Any]]] = {}
    regime_diagnostics: list[dict[str, Any]] = []
    for horizon in horizons:
        target_frames = [
            build_fixed_horizon_rows(
                raw_by_symbol[symbol],
                features_by_symbol[symbol],
                symbol=symbol,
                horizon_bars=int(horizon),
            )
            for symbol in SYMBOLS
        ]
        rows = pd.concat(target_frames, ignore_index=True)
        folds = make_walkforward_folds(
            history_start,
            history_end,
            horizon_bars=int(horizon),
            training_days=training_days,
            test_days=test_days,
            step_days=step_days,
        )
        fold_contracts[str(horizon)] = [fold.as_dict() for fold in folds]
        for fold in folds:
            train, test = select_fold_rows(rows, fold)
            X_train = train.loc[:, ordered_features].to_numpy(dtype=np.float64)
            y_train = train["target"].to_numpy(dtype=np.int8)
            X_test = test.loc[:, ordered_features].to_numpy(dtype=np.float64)
            train_zero, train_one = _class_counts(train["target"])
            test_zero, test_one = _class_counts(test["target"])
            regime = calculate_regime_descriptors(raw_by_symbol, fold)
            regime_diagnostics.append(
                {"horizon_bars": int(horizon), **fold.as_dict(), **regime}
            )
            for model_name in model_names:
                scores, _ = fit_model_scores(model_name, X_train, y_train, X_test)
                metrics = calculate_metrics(test, scores)
                row: dict[str, Any] = {
                    "experiment_id": experiment_id,
                    "horizon_bars": int(horizon),
                    "horizon_minutes": int(horizon) * 5,
                    "model_name": model_name,
                    **fold.as_dict(),
                    "actual_train_first_utc": canonical_utc(train["timestamp"].min()),
                    "actual_train_last_utc": canonical_utc(train["timestamp"].max()),
                    "maximum_train_label_timestamp_utc": canonical_utc(
                        train["target_timestamp"].max()
                    ),
                    "actual_test_first_utc": canonical_utc(test["timestamp"].min()),
                    "actual_test_last_utc": canonical_utc(test["timestamp"].max()),
                    "train_row_count": int(len(train)),
                    "test_row_count": int(len(test)),
                    "train_class_0_count": train_zero,
                    "train_class_1_count": train_one,
                    "test_class_0_count": test_zero,
                    "test_class_1_count": test_one,
                    "train_btc_row_count": int((train["symbol"] == "BTCUSDT").sum()),
                    "train_eth_row_count": int((train["symbol"] == "ETHUSDT").sum()),
                    "test_btc_row_count": int((test["symbol"] == "BTCUSDT").sum()),
                    "test_eth_row_count": int((test["symbol"] == "ETHUSDT").sum()),
                    **metrics,
                    **regime,
                }
                fold_metrics.append(row)

    stability = summarize_stability(fold_metrics)
    ranking = rank_stability(stability)
    strongest = ranking[0] if ranking else None
    target_contracts = [
        {
            "name": f"fixed_horizon_direction_{horizon}_bars",
            "horizon_bars": int(horizon),
            "horizon_minutes": int(horizon) * 5,
            "future_log_return": "log(close[t+horizon]) - log(close[t])",
            "classification_target": "1 iff future_log_return > 0, otherwise 0",
            "last_h_raw_rows_excluded": True,
        }
        for horizon in horizons
    ]
    combined_data = {
        "first_bar_open_utc": canonical_utc(history_start),
        "last_bar_open_utc": canonical_utc(history_end),
        "row_counts_by_symbol": {
            symbol: int(len(raw_by_symbol[symbol])) for symbol in SYMBOLS
        },
        "exact_five_minute_spacing": True,
        "duplicate_timestamps": 0,
        "conflicting_overlapping_bars": 0,
        "shared_symbol_timestamp_grid": True,
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "research_only": True,
        "production_candidate": False,
        "promotion_allowed": False,
        "live_execution_allowed": False,
        "source_datasets": source_metadata,
        "raw_source_digests": dict(sorted(initial_raw_digests.items())),
        "combined_raw_data": combined_data,
        "feature_contract": feature_contract,
        "feature_build_diagnostics": feature_diagnostics,
        "target_contracts": {
            "fixed_horizon_primary": target_contracts,
            "triple_barrier_control": {
                "included": False,
                "reason": (
                    "optional control omitted to avoid coupling this research tool to "
                    "candidate-training label/access paths"
                ),
            },
        },
        "walk_forward_contract": {
            **fold_spec,
            "history_anchor_utc": canonical_utc(history_start),
            "fold_count_by_horizon": {
                horizon: len(fold_contracts[str(horizon)]) for horizon in horizons
            },
            "folds_by_horizon": fold_contracts,
        },
        "model_configs": {name: MODEL_CONFIGS[name] for name in model_names},
        "stability_summaries": stability,
        "ranking_contract": {
            "eligible_models": [
                name for name in model_names if MODEL_CONFIGS[name]["rankable"]
            ],
            "excluded_comparators": [
                name for name in model_names if not MODEL_CONFIGS[name]["rankable"]
            ],
            "ordered_fields_descending": [
                "median_pooled_auc",
                "minimum_pooled_auc",
                "minimum_symbol_mean_auc",
                "median_top_bottom_return_spread_bps",
            ],
            "production_acceptance_threshold_defined": False,
        },
        "deterministic_ranking": ranking,
        "strongest_horizon_model": strongest,
        "regime_diagnostics": regime_diagnostics,
        "return_diagnostics_are_pnl": False,
        "historical_periods_pristine_holdout": False,
        "historical_research_exposure_warning": (
            "All historical periods used by this walk-forward scan are research-exposed "
            "and must never again be described as pristine holdout data."
        ),
        "future_untouched_confirmation_required_before_deployment": True,
        "safety_contract": {
            "candidate_training_performed": False,
            "candidate_artifacts_written": False,
            "validation_or_internal_test_access_ledger_written": False,
            "runtime_or_serving_modified": False,
            "exchange_or_network_access_performed": False,
            "models_saved": False,
            "outputs_restricted_to": str(output),
        },
    }

    final_raw_metadata, final_raw_digests = _source_metadata(dataset_paths)
    if final_raw_digests != initial_raw_digests or final_raw_metadata != source_metadata:
        raise SignalResearchError("frozen raw source files changed during research run")
    output.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir()
    try:
        summary_path = staging_directory / "summary.json"
        fold_path = staging_directory / "fold_metrics.csv"
        manifest_path = staging_directory / "experiment_manifest.json"
        _write_json(summary_path, summary)
        _write_fold_metrics(fold_path, fold_metrics)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "experiment_digest": experiment_digest,
            "created_at_utc": utc_now(),
            "research_only": True,
            "production_candidate": False,
            "promotion_allowed": False,
            "live_execution_allowed": False,
            "experiment_contract": experiment_contract,
            "source_files_unchanged_during_run": True,
            "outputs": {
                "summary.json": {"sha256": file_sha256(summary_path)},
                "fold_metrics.csv": {
                    "sha256": file_sha256(fold_path),
                    "row_count": len(fold_metrics),
                },
            },
            "execution_environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "scipy_available": _spearman(
                    np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
                )
                is not None,
            },
        }
        _write_json(manifest_path, manifest)
        staging_directory.rename(final_directory)
    except Exception:
        # Leave any partial staging evidence in the research-only root for inspection.
        raise
    summary["output_directory"] = str(final_directory)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated offline fixed-horizon signal walk-forward research scan."
    )
    parser.add_argument(
        "--earlier-dataset", type=Path, default=DEFAULT_EARLIER_DATASET
    )
    parser.add_argument("--later-dataset", type=Path, default=DEFAULT_LATER_DATASET)
    parser.add_argument("--output-root", type=Path, default=RESEARCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_research(
            args.earlier_dataset,
            args.later_dataset,
            output_root=args.output_root,
        )
    except (SignalResearchError, OSError, ValueError) as exc:
        print(f"model signal walk-forward failed closed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "experiment_id": summary["experiment_id"],
                "experiment_digest": summary["experiment_digest"],
                "output_directory": summary["output_directory"],
                "research_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
