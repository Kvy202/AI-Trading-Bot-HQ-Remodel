"""Offline research for economically aligned executable-return prediction.

The primary target is the signed simple return available from the validated
next-bar execution convention: signal after bar ``t`` completes, entry at
``open[t + 1]``, and exit at ``close[t + h]``.  The task uses only frozen
sklearn configurations and exposed historical data.  It has no runtime,
exchange, network, candidate-training, promotion, or model-saving dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools import model_signal_backtest as bt  # noqa: E402
from tools import model_signal_walkforward as wf  # noqa: E402


SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = "model-executable-return-research-v1"
SOURCE_WALKFORWARD_EXPERIMENT_ID = bt.SOURCE_WALKFORWARD_EXPERIMENT_ID
SOURCE_WALKFORWARD_EXPERIMENT_DIGEST = bt.SOURCE_WALKFORWARD_EXPERIMENT_DIGEST
SOURCE_WALKFORWARD_DIRECTORY = bt.SOURCE_WALKFORWARD_DIRECTORY
SOURCE_WALKFORWARD_DIRECTORY_DIGEST = (
    "46af7ff4e7a2b74d1e8e3a3b6554840dcbd695a55210be07fb86d6016defe71d"
)
SOURCE_BACKTEST_EXPERIMENT_ID = "backtest_5964745630c574fd"
SOURCE_BACKTEST_EXPERIMENT_DIGEST = (
    "5964745630c574fdc728dbaf14bae7def59c3ff5a1fdcfe995e407d9c91084a4"
)
SOURCE_BACKTEST_DIRECTORY = (
    BASE_DIR / "reports" / "model_signal_backtest" / SOURCE_BACKTEST_EXPERIMENT_ID
)
SOURCE_BACKTEST_DIRECTORY_DIGEST = (
    "105d64894fc451d5a38deeb0a9a141965e5179d3b2d5d4c8c1b51c36435a7842"
)
SOURCE_SELECTIVITY_EXPERIMENT_ID = "selectivity_237d071311426f5f"
SOURCE_SELECTIVITY_EXPERIMENT_DIGEST = (
    "237d071311426f5f4758e4bda109959a08e9dbfbbaeb717e380031bd109486b3"
)
SOURCE_SELECTIVITY_DIRECTORY = (
    BASE_DIR
    / "reports"
    / "model_signal_selectivity"
    / SOURCE_SELECTIVITY_EXPERIMENT_ID
)
SOURCE_SELECTIVITY_DIRECTORY_DIGEST = (
    "0c6fc25948f05a0ed973484cf78361e3c9b22867864a365b8dd06089f6c9fcb2"
)
RESEARCH_ROOT = BASE_DIR / "reports" / "model_executable_return_research"

HORIZONS = (6, 12, 24)
HORIZON_LABELS = {6: "30m", 12: "1h", 24: "2h"}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "ridge_regression": {
        "pipeline": ["StandardScaler", "Ridge"],
        "standard_scaler": {"with_mean": True, "with_std": True},
        "ridge": {
            "alpha": 1.0,
            "fit_intercept": True,
            "max_iter": 1000,
            "solver": "lsqr",
            "tol": 0.0001,
        },
        "hyperparameter_sweep_performed": False,
    },
    "hist_gradient_boosting_regressor": {
        "estimator": "HistGradientBoostingRegressor",
        "hist_gradient_boosting_regressor": {
            "early_stopping": False,
            "l2_regularization": 0.0,
            "learning_rate": 0.1,
            "loss": "squared_error",
            "max_bins": 255,
            "max_depth": None,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "random_state": 1729,
            "tol": 1e-7,
        },
        "analogous_classifier_configuration": wf.MODEL_CONFIGS[
            "hist_gradient_boosting"
        ]["hist_gradient_boosting"],
        "hyperparameter_sweep_performed": False,
    },
    "training_mean_baseline": {
        "estimator": "constant arithmetic mean executable return bps from outer training rows",
        "hyperparameter_sweep_performed": False,
    },
}
MODELS = tuple(MODEL_CONFIGS)
POLICIES: tuple[dict[str, Any], ...] = (
    {
        "policy_name": "predicted_sign",
        "definition": "prediction > 0 LONG; prediction < 0 SHORT; zero FLAT",
        "threshold_quantile": None,
        "threshold_source": "fixed zero",
    },
    {
        "policy_name": "train_abs_q90",
        "definition": (
            "abs(test prediction) >= frozen q90 of abs(train predictions) trades "
            "the predicted sign; otherwise FLAT"
        ),
        "threshold_quantile": 0.90,
        "threshold_source": "same-fold fitted-model training predictions only",
    },
    {
        "policy_name": "train_abs_q95",
        "definition": (
            "abs(test prediction) >= frozen q95 of abs(train predictions) trades "
            "the predicted sign; otherwise FLAT"
        ),
        "threshold_quantile": 0.95,
        "threshold_source": "same-fold fitted-model training predictions only",
    },
)
POLICY_ORDER = tuple(row["policy_name"] for row in POLICIES)
COST_SCENARIOS_BPS = bt.COST_SCENARIOS_BPS
EXECUTION_CONTRACT = dict(bt.EXECUTION_CONTRACT)
PORTFOLIO_CONTRACT = dict(bt.PORTFOLIO_CONTRACT)
COST_CONTRACT = dict(bt.COST_CONTRACT)
LIMITATIONS = dict(bt.LIMITATIONS)
TARGET_CONTRACT = {
    "feature_timestamp": "t",
    "signal_available_after": "bar t completes",
    "entry_price": "open[t + 1]",
    "exit_price": "close[t + horizon_bars]",
    "executable_simple_return": "exit_price / entry_price - 1",
    "executable_return_bps": "executable_simple_return * 10000",
    "positive_interpretation": "profitable LONG before costs",
    "negative_interpretation": "profitable SHORT before costs",
    "missing_entry_or_exit": "fail closed",
    "target_fields_in_features": False,
}
DECILE_CONTRACT = {
    "bucket_count": 10,
    "boundaries": "fixed equal-count ranks of predictions; stable order for ties",
    "constant_prediction_behavior": "not rankable; decile metrics are null",
    "optimization_performed": False,
    "magnitude_monotonic_definition": (
        "mean absolute realized executable return is nondecreasing from absolute-"
        "prediction bucket 1 through bucket 10"
    ),
}

RETURN_LEDGER_COLUMNS = (
    *bt.LEDGER_COLUMNS,
    "source_selectivity_experiment_digest",
    "prediction_bps",
    "prediction_origin",
    "training_abs_prediction_threshold_bps",
    "threshold_quantile",
    "threshold_source",
)


class ExecutableReturnResearchError(ValueError):
    """Raised when a target, leakage, source, or accounting invariant fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def strategy_id(model_name: str, horizon_bars: int) -> str:
    if model_name not in MODEL_CONFIGS or horizon_bars not in HORIZON_LABELS:
        raise ExecutableReturnResearchError("unknown model or horizon")
    return f"{model_name}_{HORIZON_LABELS[horizon_bars]}"


def policy_contract(policy_name: str) -> dict[str, Any]:
    for row in POLICIES:
        if row["policy_name"] == policy_name:
            return dict(row)
    raise ExecutableReturnResearchError(f"unknown economic policy: {policy_name}")


def validate_previous_research_source(
    source_directory: Path | str,
    *,
    expected_id: str,
    expected_digest: str,
    expected_directory_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    source = Path(source_directory).resolve()
    summary = bt._load_json(source / "summary.json")
    manifest = bt._load_json(source / "experiment_manifest.json")
    for document, name in ((summary, "summary"), (manifest, "manifest")):
        if document.get("experiment_id") != expected_id:
            raise ExecutableReturnResearchError(
                f"unexpected prior {name} experiment ID"
            )
        if document.get("experiment_digest") != expected_digest:
            raise ExecutableReturnResearchError(
                f"unexpected prior {name} experiment digest"
            )
        if document.get("research_only") is not True:
            raise ExecutableReturnResearchError(f"prior {name} is not research-only")
        if document.get("production_candidate") is not False:
            raise ExecutableReturnResearchError(f"prior {name} production flag is unsafe")
    for name, metadata in manifest.get("outputs", {}).items():
        path = source / name
        if metadata.get("sha256") != bt.file_sha256(path):
            raise ExecutableReturnResearchError(
                f"prior research output hash mismatch: {name}"
            )
    digest = bt.directory_digest(source)
    if digest != expected_directory_digest:
        raise ExecutableReturnResearchError("prior research directory digest changed")
    return summary, manifest, digest


def build_executable_return_rows(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    *,
    symbol: str,
    horizon_bars: int,
) -> pd.DataFrame:
    """Attach next-open-to-horizon-close executable returns to time-t features."""

    if symbol not in wf.SYMBOLS:
        raise ExecutableReturnResearchError(f"unexpected symbol: {symbol}")
    if horizon_bars not in HORIZONS:
        raise ExecutableReturnResearchError(f"unsupported horizon: {horizon_bars}")
    if raw.empty or features.empty or not isinstance(raw.index, pd.DatetimeIndex):
        raise ExecutableReturnResearchError("raw/features input is empty or unindexed")
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ExecutableReturnResearchError("raw timestamp grid is not ordered and unique")
    if not features.index.is_unique:
        raise ExecutableReturnResearchError("feature timestamp grid is not unique")
    required_raw = {"open", "close"}
    if not required_raw <= set(raw.columns):
        raise ExecutableReturnResearchError("raw input lacks open/close")

    last_signal_with_exit = raw.index[-1] - horizon_bars * wf.BAR_INTERVAL
    eligible = features.loc[features.index <= last_signal_with_exit].copy()
    if eligible.empty:
        raise ExecutableReturnResearchError("no feature rows have a complete target")
    signal_times = pd.DatetimeIndex(eligible.index)
    entry_times = signal_times + wf.BAR_INTERVAL
    exit_bar_times = signal_times + horizon_bars * wf.BAR_INTERVAL
    missing_entry = entry_times.difference(raw.index)
    if len(missing_entry):
        raise ExecutableReturnResearchError(
            f"missing next-bar entry for {symbol}: {wf.canonical_utc(missing_entry[0])}"
        )
    missing_exit = exit_bar_times.difference(raw.index)
    if len(missing_exit):
        raise ExecutableReturnResearchError(
            f"missing horizon exit for {symbol}: {wf.canonical_utc(missing_exit[0])}"
        )

    entry = raw.loc[entry_times, "open"].to_numpy(dtype=np.float64)
    exit_price = raw.loc[exit_bar_times, "close"].to_numpy(dtype=np.float64)
    if not (
        np.isfinite(entry).all()
        and np.isfinite(exit_price).all()
        and np.all(entry > 0.0)
        and np.all(exit_price > 0.0)
    ):
        raise ExecutableReturnResearchError("entry/exit prices are invalid")
    executable = exit_price / entry - 1.0
    if not np.isfinite(executable).all():
        raise ExecutableReturnResearchError("executable target is nonfinite")

    eligible["symbol"] = symbol
    eligible["timestamp"] = signal_times
    eligible["entry_timestamp"] = entry_times
    eligible["exit_bar_open_timestamp"] = exit_bar_times
    eligible["target_exit_timestamp"] = exit_bar_times + wf.BAR_INTERVAL
    eligible["target_timestamp"] = eligible["target_exit_timestamp"]
    eligible["entry_price"] = entry
    eligible["exit_price"] = exit_price
    eligible["executable_simple_return"] = executable
    eligible["executable_return_bps"] = executable * 10_000.0
    return eligible.reset_index(drop=True)


def select_executable_fold_rows(
    rows: pd.DataFrame, fold: wf.FoldDefinition
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "timestamp",
        "target_exit_timestamp",
        "symbol",
        "executable_return_bps",
    }
    if not required <= set(rows.columns):
        raise ExecutableReturnResearchError("executable target rows are incomplete")
    train_mask = (
        (rows["timestamp"] >= fold.training_window_start)
        & (rows["timestamp"] < fold.fit_train_end_exclusive)
        & (rows["target_exit_timestamp"] < fold.test_start)
    )
    test_mask = (
        (rows["timestamp"] >= fold.test_start)
        & (rows["timestamp"] < fold.test_end_exclusive)
    )
    train = rows.loc[train_mask].sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    )
    test = rows.loc[test_mask].sort_values(
        ["timestamp", "symbol"], kind="mergesort"
    )
    if train.empty or test.empty:
        raise ExecutableReturnResearchError(f"empty train/test rows for {fold.fold_id}")
    if set(train["symbol"].unique()) != set(wf.SYMBOLS):
        raise ExecutableReturnResearchError(
            f"training symbol coverage failed for {fold.fold_id}"
        )
    if set(test["symbol"].unique()) != set(wf.SYMBOLS):
        raise ExecutableReturnResearchError(
            f"test symbol coverage failed for {fold.fold_id}"
        )
    max_train_exit = pd.Timestamp(train["target_exit_timestamp"].max())
    min_test = pd.Timestamp(test["timestamp"].min())
    if max_train_exit >= min_test:
        raise ExecutableReturnResearchError(
            f"executable target purge leakage in {fold.fold_id}"
        )
    if pd.Timestamp(train["timestamp"].max()) >= min_test:
        raise ExecutableReturnResearchError(
            f"train/test chronology failed for {fold.fold_id}"
        )
    return train, test


def make_regression_model(model_name: str) -> Any:
    if model_name == "ridge_regression":
        config = MODEL_CONFIGS[model_name]
        return Pipeline(
            steps=[
                (
                    "standard_scaler",
                    StandardScaler(**config["standard_scaler"]),
                ),
                ("ridge", Ridge(**config["ridge"])),
            ]
        )
    if model_name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(
            **MODEL_CONFIGS[model_name]["hist_gradient_boosting_regressor"]
        )
    if model_name == "training_mean_baseline":
        return None
    raise ExecutableReturnResearchError(f"unknown regression model: {model_name}")


def fit_regression_predictions(
    model_name: str,
    X_train: np.ndarray,
    y_train_bps: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit on outer training arrays and return train/test predictions."""

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train_bps, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if (
        X_train.ndim != 2
        or X_test.ndim != 2
        or X_train.shape[1] != X_test.shape[1]
        or len(y_train) != len(X_train)
        or len(X_train) == 0
    ):
        raise ExecutableReturnResearchError("invalid regression train/test arrays")
    if not (
        np.isfinite(X_train).all()
        and np.isfinite(X_test).all()
        and np.isfinite(y_train).all()
    ):
        raise ExecutableReturnResearchError("regression input contains nonfinite values")
    if model_name == "training_mean_baseline":
        mean = float(np.mean(y_train))
        return (
            np.full(len(X_train), mean, dtype=np.float64),
            np.full(len(X_test), mean, dtype=np.float64),
            None,
        )
    estimator = make_regression_model(model_name)
    estimator.fit(X_train, y_train)
    train_predictions = np.asarray(estimator.predict(X_train), dtype=np.float64)
    test_predictions = np.asarray(estimator.predict(X_test), dtype=np.float64)
    if not (
        np.isfinite(train_predictions).all()
        and np.isfinite(test_predictions).all()
    ):
        raise ExecutableReturnResearchError("regression model produced nonfinite output")
    return train_predictions, test_predictions, estimator


def derive_absolute_prediction_threshold(
    train_predictions_bps: Sequence[float], policy_name: str
) -> float | None:
    policy = policy_contract(policy_name)
    quantile = policy["threshold_quantile"]
    if quantile is None:
        return None
    predictions = np.asarray(train_predictions_bps, dtype=np.float64)
    if predictions.ndim != 1 or len(predictions) == 0 or not np.isfinite(predictions).all():
        raise ExecutableReturnResearchError("training predictions are invalid")
    threshold = float(np.quantile(np.abs(predictions), float(quantile), method="linear"))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ExecutableReturnResearchError("absolute prediction threshold is invalid")
    return threshold


def return_policy_direction(
    prediction_bps: float,
    policy_name: str,
    *,
    training_abs_threshold_bps: float | None,
) -> str:
    prediction = float(prediction_bps)
    if not math.isfinite(prediction):
        raise ExecutableReturnResearchError("prediction is nonfinite")
    policy = policy_contract(policy_name)
    if prediction == 0.0:
        return "FLAT"
    if policy["threshold_quantile"] is not None:
        if training_abs_threshold_bps is None:
            raise ExecutableReturnResearchError("training prediction threshold is missing")
        threshold = float(training_abs_threshold_bps)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ExecutableReturnResearchError("training prediction threshold is invalid")
        if abs(prediction) < threshold:
            return "FLAT"
    return "LONG" if prediction > 0.0 else "SHORT"


def fixed_decile_buckets(values: Sequence[float]) -> np.ndarray | None:
    """Assign stable equal-count deciles without tuning bucket boundaries."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 10 or not np.isfinite(array).all():
        return None
    if float(np.ptp(array)) == 0.0:
        return None
    order = np.argsort(array, kind="mergesort")
    buckets = np.empty(len(array), dtype=np.int8)
    ranks = np.arange(len(array), dtype=np.int64)
    buckets[order] = np.minimum(9, ranks * 10 // len(array)).astype(np.int8)
    return buckets


def regression_metrics(
    predictions_bps: Sequence[float],
    targets_bps: Sequence[float],
    baseline_predictions_bps: Sequence[float],
) -> dict[str, Any]:
    predictions = np.asarray(predictions_bps, dtype=np.float64)
    targets = np.asarray(targets_bps, dtype=np.float64)
    baseline = np.asarray(baseline_predictions_bps, dtype=np.float64)
    if (
        predictions.ndim != 1
        or targets.ndim != 1
        or baseline.ndim != 1
        or len(predictions) == 0
        or len(predictions) != len(targets)
        or len(predictions) != len(baseline)
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
        or not np.isfinite(baseline).all()
    ):
        raise ExecutableReturnResearchError("regression metric arrays are invalid")
    error = predictions - targets
    baseline_error = baseline - targets
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    ratio = rmse / baseline_rmse if baseline_rmse > 0.0 else None
    buckets = fixed_decile_buckets(predictions)
    top_mean = bottom_mean = spread = top_win = bottom_short_win = None
    if buckets is not None:
        top = targets[buckets == 9]
        bottom = targets[buckets == 0]
        if len(top) and len(bottom):
            top_mean = float(np.mean(top))
            bottom_mean = float(np.mean(bottom))
            spread = top_mean - bottom_mean
            top_win = float(np.mean(top > 0.0))
            bottom_short_win = float(np.mean(bottom < 0.0))
    return {
        "row_count": int(len(predictions)),
        "mae_bps": mae,
        "rmse_bps": rmse,
        "baseline_rmse_bps": baseline_rmse,
        "candidate_baseline_rmse_ratio": ratio,
        "pearson_correlation": wf._correlation(predictions, targets),
        "spearman_correlation": wf._spearman(predictions, targets),
        "absolute_magnitude_pearson_correlation": wf._correlation(
            np.abs(predictions), np.abs(targets)
        ),
        "absolute_magnitude_spearman_correlation": wf._spearman(
            np.abs(predictions), np.abs(targets)
        ),
        "sign_accuracy": float(np.mean(np.sign(predictions) == np.sign(targets))),
        "top_predicted_decile_mean_realized_return_bps": top_mean,
        "bottom_predicted_decile_mean_realized_return_bps": bottom_mean,
        "top_minus_bottom_realized_spread_bps": spread,
        "top_decile_long_win_rate": top_win,
        "bottom_decile_short_win_rate": bottom_short_win,
        "predicted_mean_bps": float(np.mean(predictions)),
        "predicted_std_bps": float(np.std(predictions, ddof=0)),
        "target_mean_bps": float(np.mean(targets)),
        "target_std_bps": float(np.std(targets, ddof=0)),
        "prediction_deciles_rankable": buckets is not None,
    }


def magnitude_bucket_diagnostics(
    predictions_bps: Sequence[float],
    targets_bps: Sequence[float],
) -> tuple[list[dict[str, Any]], bool | None]:
    predictions = np.asarray(predictions_bps, dtype=np.float64)
    targets = np.asarray(targets_bps, dtype=np.float64)
    if (
        predictions.ndim != 1
        or targets.ndim != 1
        or len(predictions) != len(targets)
        or len(predictions) == 0
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
    ):
        raise ExecutableReturnResearchError("magnitude diagnostic arrays are invalid")
    buckets = fixed_decile_buckets(np.abs(predictions))
    if buckets is None:
        return [], None
    rows: list[dict[str, Any]] = []
    realized_means: list[float] = []
    for bucket in range(10):
        selected = buckets == bucket
        if not selected.any():
            raise ExecutableReturnResearchError("fixed magnitude decile is empty")
        realized = float(np.mean(np.abs(targets[selected])))
        realized_means.append(realized)
        rows.append(
            {
                "prediction_magnitude_decile": bucket + 1,
                "row_count": int(selected.sum()),
                "mean_absolute_prediction_bps": float(
                    np.mean(np.abs(predictions[selected]))
                ),
                "mean_absolute_realized_return_bps": realized,
                "median_absolute_realized_return_bps": float(
                    np.median(np.abs(targets[selected]))
                ),
            }
        )
    monotonic = bool(np.all(np.diff(np.asarray(realized_means)) >= -1e-12))
    return rows, monotonic


def build_return_trade_rows(
    *,
    experiment_id: str,
    model_name: str,
    horizon_bars: int,
    policy_name: str,
    fold: wf.FoldDefinition,
    symbol: str,
    scored_test: pd.DataFrame,
    raw: pd.DataFrame,
    training_abs_threshold_bps: float | None,
) -> list[dict[str, Any]]:
    """Use the existing timing mapper, then apply return-prediction direction."""

    required = {"timestamp", "prediction_bps", "prediction_origin"}
    if not required <= set(scored_test.columns):
        raise ExecutableReturnResearchError("scored test frame is incomplete")
    if not scored_test["prediction_origin"].eq("outer_test").all():
        raise ExecutableReturnResearchError(
            "economic results require outer-test predictions only"
        )
    indexed = scored_test.set_index("timestamp")["prediction_bps"]
    if not indexed.index.is_unique:
        raise ExecutableReturnResearchError("duplicate OOS return predictions")
    timing_scores = scored_test.loc[:, ["timestamp"]].copy()
    timing_scores["score"] = 1.0
    strategy = {
        "strategy_id": strategy_id(model_name, horizon_bars),
        "model_name": model_name,
        "horizon_bars": horizon_bars,
        "horizon_minutes": horizon_bars * 5,
    }
    base_rows = bt.build_trade_rows(
        experiment_id=experiment_id,
        strategy=strategy,
        policy_name="directional_0p5",
        fold=fold,
        symbol=symbol,
        scored_test=timing_scores,
        raw=raw,
        training_q20=None,
        training_q80=None,
    )
    policy = policy_contract(policy_name)
    result: list[dict[str, Any]] = []
    for row in base_rows:
        timestamp = pd.Timestamp(row["signal_timestamp_utc"])
        if timestamp not in indexed.index:
            raise ExecutableReturnResearchError("scheduled OOS prediction is missing")
        prediction = float(indexed.loc[timestamp])
        direction = return_policy_direction(
            prediction,
            policy_name,
            training_abs_threshold_bps=training_abs_threshold_bps,
        )
        raw_return = float(row["exit_price"]) / float(row["entry_price"]) - 1.0
        gross = bt.directional_return(raw_return, direction)
        active = direction != "FLAT"
        row.update(
            {
                "policy_name": policy_name,
                "score": prediction,
                "direction": direction,
                "training_q20": None,
                "training_q80": None,
                "gross_simple_return": gross,
                "gross_return_bps": gross * 10_000.0,
                "active_trade": active,
                "source_selectivity_experiment_digest": (
                    SOURCE_SELECTIVITY_EXPERIMENT_DIGEST
                ),
                "prediction_bps": prediction,
                "prediction_origin": "outer_test",
                "training_abs_prediction_threshold_bps": (
                    training_abs_threshold_bps
                ),
                "threshold_quantile": policy["threshold_quantile"],
                "threshold_source": policy["threshold_source"],
            }
        )
        result.append(row)
    return result


def as_return_trade_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    ledger = pd.DataFrame(rows, columns=RETURN_LEDGER_COLUMNS)
    if ledger.empty:
        raise ExecutableReturnResearchError("return-prediction trade ledger is empty")
    for column in (
        "signal_timestamp_utc",
        "signal_bar_available_utc",
        "entry_timestamp_utc",
        "exit_bar_open_timestamp_utc",
        "exit_timestamp_utc",
        "fold_test_start_utc",
        "fold_test_end_exclusive_utc",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    ledger["active_trade"] = ledger["active_trade"].astype(bool)
    bt.validate_trade_ledger(ledger)
    if not ledger["prediction_origin"].eq("outer_test").all():
        raise ExecutableReturnResearchError("non-OOS prediction entered trade ledger")
    return ledger


def gross_trade_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    active = ledger.loc[ledger["active_trade"]]
    returns = active["gross_simple_return"].to_numpy(dtype=np.float64)
    count = int(len(active))
    scheduled = int(len(ledger))
    if scheduled <= 0:
        raise ExecutableReturnResearchError("scheduled signal denominator is empty")
    return {
        "active_trade_count": count,
        "scheduled_signal_count": scheduled,
        "active_fraction": float(count / scheduled),
        "long_trade_count": int((active["direction"] == "LONG").sum()),
        "short_trade_count": int((active["direction"] == "SHORT").sum()),
        "mean_gross_bps_per_active_trade": (
            float(np.mean(returns) * 10_000.0) if count else None
        ),
        "median_gross_bps_per_active_trade": (
            float(np.median(returns) * 10_000.0) if count else None
        ),
        "gross_profit_factor": bt.profit_factor(returns) if count else None,
    }


def calculate_economic_metrics(
    ledger: pd.DataFrame,
    *,
    expected_fold_count: int = 7,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, int], pd.DataFrame],
]:
    fold_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, int], pd.DataFrame] = {}
    for (stream_id, policy_name, fold_id), group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id"], sort=True
    ):
        for cost_bps in COST_SCENARIOS_BPS:
            events = bt.build_portfolio_events(group, cost_bps)
            fold_rows.append(
                {
                    "strategy_id": stream_id,
                    "model_name": str(group["model_name"].iloc[0]),
                    "horizon_bars": int(group["horizon_bars"].iloc[0]),
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "fold_id": fold_id,
                    "cost_bps": int(cost_bps),
                    **gross_trade_metrics(group),
                    **bt.stream_metrics(group, events, cost_bps),
                }
            )
    fold_frame = pd.DataFrame(fold_rows)
    for (stream_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        gross = gross_trade_metrics(group)
        for cost_bps in COST_SCENARIOS_BPS:
            events = bt.build_portfolio_events(group, cost_bps)
            curve = bt.enrich_equity_curve(events)
            curves[(str(stream_id), str(policy_name), int(cost_bps))] = curve
            metrics = bt.stream_metrics(group, events, cost_bps)
            folds = fold_frame.loc[
                (fold_frame["strategy_id"] == stream_id)
                & (fold_frame["policy_name"] == policy_name)
                & (fold_frame["cost_bps"] == cost_bps)
            ].sort_values("fold_id", kind="mergesort")
            if len(folds) != expected_fold_count:
                raise ExecutableReturnResearchError(
                    f"overall economics require {expected_fold_count} folds"
                )
            fold_returns = folds["net_cumulative_return"].to_numpy(dtype=np.float64)
            overall_rows.append(
                {
                    "strategy_id": stream_id,
                    "model_name": str(group["model_name"].iloc[0]),
                    "horizon_bars": int(group["horizon_bars"].iloc[0]),
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "cost_bps": int(cost_bps),
                    **gross,
                    **metrics,
                    "fold_count": int(len(folds)),
                    "positive_fold_percentage": float(
                        100.0 * np.mean(fold_returns > 0.0)
                    ),
                    "median_fold_net_return": float(np.median(fold_returns)),
                    "worst_fold_net_return": float(np.min(fold_returns)),
                    "best_fold_net_return": float(np.max(fold_returns)),
                }
            )
    return fold_rows, overall_rows, curves


def build_policy_summary(
    ledger: pd.DataFrame, overall_metrics: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    overall = pd.DataFrame(overall_metrics)
    rows: list[dict[str, Any]] = []
    for (stream_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        cost_rows = overall.loc[
            (overall["strategy_id"] == stream_id)
            & (overall["policy_name"] == policy_name)
        ]
        by_cost = {int(row.cost_bps): row for row in cost_rows.itertuples(index=False)}
        if set(by_cost) != set(COST_SCENARIOS_BPS):
            raise ExecutableReturnResearchError("economic cost grid is incomplete")
        base: dict[str, Any] = {
            "strategy_id": stream_id,
            "model_name": str(group["model_name"].iloc[0]),
            "horizon_bars": int(group["horizon_bars"].iloc[0]),
            "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
            "policy_name": policy_name,
            **gross_trade_metrics(group),
            "approximate_break_even_round_trip_cost_bps": (
                bt.approximate_break_even_cost_bps(group)
            ),
        }
        for cost_bps in COST_SCENARIOS_BPS:
            metric = by_cost[cost_bps]
            suffix = f"{cost_bps}bps"
            base.update(
                {
                    f"overall_net_return_{suffix}": float(metric.net_cumulative_return),
                    f"daily_sharpe_{suffix}": (
                        None if metric.daily_sharpe is None else float(metric.daily_sharpe)
                    ),
                    f"maximum_drawdown_{suffix}": float(metric.maximum_drawdown),
                    f"positive_fold_percentage_{suffix}": float(
                        metric.positive_fold_percentage
                    ),
                    f"median_fold_return_{suffix}": float(
                        metric.median_fold_net_return
                    ),
                    f"worst_fold_return_{suffix}": float(
                        metric.worst_fold_net_return
                    ),
                    f"btc_net_return_{suffix}": float(
                        metric.btc_net_cumulative_return
                    ),
                    f"eth_net_return_{suffix}": float(
                        metric.eth_net_cumulative_return
                    ),
                }
            )
        base.update(
            {
                "survives_2bps": base["overall_net_return_2bps"] > 0.0,
                "survives_5bps": base["overall_net_return_5bps"] > 0.0,
                "survives_10bps": base["overall_net_return_10bps"] > 0.0,
                "both_symbols_positive_at_5bps": (
                    base["btc_net_return_5bps"] > 0.0
                    and base["eth_net_return_5bps"] > 0.0
                ),
                "positive_majority_of_folds_at_5bps": (
                    base["positive_fold_percentage_5bps"] > 50.0
                ),
                "production_pass_gate_defined": False,
                "diagnostic_only": True,
            }
        )
        rows.append(base)
    return sorted(
        rows,
        key=lambda row: (
            HORIZONS.index(int(row["horizon_bars"])),
            MODELS.index(str(row["model_name"])),
            POLICY_ORDER.index(str(row["policy_name"])),
        ),
    )


def validate_research_results(
    *,
    ledger: pd.DataFrame,
    fold_metrics: Sequence[Mapping[str, Any]],
    overall_metrics: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
    regression_fold_metrics: Sequence[Mapping[str, Any]],
    magnitude_bucket_rows: Sequence[Mapping[str, Any]],
    curves: Mapping[tuple[str, str, int], pd.DataFrame],
) -> dict[str, bool]:
    checks = bt.validate_trade_ledger(ledger)
    if not ledger["prediction_origin"].eq("outer_test").all():
        raise ExecutableReturnResearchError("economics include non-OOS predictions")
    summary = pd.DataFrame(policy_summary)
    overall = pd.DataFrame(overall_metrics)
    folds = pd.DataFrame(fold_metrics)
    regression_folds = pd.DataFrame(regression_fold_metrics)
    buckets = pd.DataFrame(magnitude_bucket_rows)
    for (stream_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=False
    ):
        expected = int(group["active_trade"].sum())
        match = summary.loc[
            (summary["strategy_id"] == stream_id)
            & (summary["policy_name"] == policy_name)
        ]
        if len(match) != 1 or int(match.iloc[0]["active_trade_count"]) != expected:
            raise ExecutableReturnResearchError("active trade count does not reconcile")
        if int(match.iloc[0]["long_trade_count"] + match.iloc[0]["short_trade_count"]) != expected:
            raise ExecutableReturnResearchError("long/short count does not reconcile")
        rows = overall.loc[
            (overall["strategy_id"] == stream_id)
            & (overall["policy_name"] == policy_name)
        ].set_index("cost_bps")
        zero = rows.loc[0]
        if not math.isclose(
            float(zero["net_cumulative_return"]),
            float(zero["gross_cumulative_return"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ExecutableReturnResearchError("zero-cost net differs from gross")
        for lower, higher in zip(COST_SCENARIOS_BPS, COST_SCENARIOS_BPS[1:]):
            lower_curve = curves[(str(stream_id), str(policy_name), lower)]
            higher_curve = curves[(str(stream_id), str(policy_name), higher)]
            if np.any(
                higher_curve["net_equity"].to_numpy(dtype=np.float64)
                > lower_curve["net_equity"].to_numpy(dtype=np.float64) + 1e-12
            ):
                raise ExecutableReturnResearchError("higher cost improved equity")
    for keys, group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id"], sort=False
    ):
        match = folds.loc[
            (folds["strategy_id"] == keys[0])
            & (folds["policy_name"] == keys[1])
            & (folds["fold_id"] == keys[2])
        ]
        if len(match) != len(COST_SCENARIOS_BPS):
            raise ExecutableReturnResearchError("fold economic grid is incomplete")
        if not (match["active_trade_count"] == int(group["active_trade"].sum())).all():
            raise ExecutableReturnResearchError("fold active counts do not reconcile")
    if set(regression_folds["metric_scope"]) != {"ALL", *wf.SYMBOLS}:
        raise ExecutableReturnResearchError("BTC/ETH regression metrics are incomplete")
    if not buckets.empty:
        for _, group in buckets.groupby(
            ["aggregation", "horizon_bars", "model_name", "fold_id", "metric_scope"],
            dropna=False,
        ):
            if set(group["prediction_magnitude_decile"].astype(int)) != set(range(1, 11)):
                raise ExecutableReturnResearchError("magnitude buckets do not reconcile")
    for cost in COST_SCENARIOS_BPS:
        active = ledger["active_trade"].to_numpy(dtype=bool)
        gross = ledger["gross_simple_return"].to_numpy(dtype=np.float64)
        net = np.asarray(
            [
                bt.apply_round_trip_cost(
                    value,
                    active_trade=bool(is_active),
                    round_trip_cost_bps=cost,
                )
                for value, is_active in zip(gross, active)
            ]
        )
        if np.any(net[~active] != 0.0):
            raise ExecutableReturnResearchError("flat signal paid transaction cost")
        if cost == 0 and not np.array_equal(net, gross):
            raise ExecutableReturnResearchError("zero-cost trade return mismatch")
    return {
        **checks,
        "executable_target_uses_next_bar_open": True,
        "executable_target_uses_horizon_close": True,
        "horizon_exit_purge_is_strict": True,
        "thresholds_use_training_predictions_only": True,
        "test_predictions_do_not_set_thresholds": True,
        "test_returns_do_not_set_thresholds": True,
        "economic_predictions_are_outer_test_only": True,
        "accounting_reused_from_validated_backtest": True,
        "transaction_cost_applied_once": True,
        "higher_cost_never_improves_equity": True,
        "zero_cost_net_equals_gross": True,
        "btc_eth_results_remain_separate": True,
        "regression_metrics_reconcile": True,
        "magnitude_buckets_reconcile": True,
        "active_trade_counts_reconcile": True,
    }


def protected_source_hashes(
    *,
    earlier: Path,
    later: Path,
    expected_raw: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "features.py": bt.file_sha256(BASE_DIR / "features.py"),
        "model_artifacts": bt.directory_digest(BASE_DIR / "model_artifacts"),
        "candidates": bt.directory_digest(BASE_DIR / "model_artifacts" / "candidates"),
        "validation_ledger": bt.file_sha256(
            BASE_DIR / "reports" / "model_candidate_validation_access.json"
        ),
        "walkforward_evidence": bt.directory_digest(SOURCE_WALKFORWARD_DIRECTORY),
        "backtest_evidence": bt.directory_digest(SOURCE_BACKTEST_DIRECTORY),
        "selectivity_evidence": bt.directory_digest(SOURCE_SELECTIVITY_DIRECTORY),
        "raw_source_files": bt.raw_source_digests(earlier, later, expected_raw),
        "live_writer.py": "absent",
        "live_executor.py": "absent",
    }


def experiment_contract(
    *, protected_hashes: Mapping[str, Any], feature_contract_digest: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "tool_code_sha256": bt.file_sha256(Path(__file__)),
        "source_walkforward_experiment_id": SOURCE_WALKFORWARD_EXPERIMENT_ID,
        "source_walkforward_experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
        "source_backtest_experiment_id": SOURCE_BACKTEST_EXPERIMENT_ID,
        "source_backtest_experiment_digest": SOURCE_BACKTEST_EXPERIMENT_DIGEST,
        "source_selectivity_experiment_id": SOURCE_SELECTIVITY_EXPERIMENT_ID,
        "source_selectivity_experiment_digest": SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
        "protected_source_hashes": dict(protected_hashes),
        "feature_contract_digest": feature_contract_digest,
        "target_contract": TARGET_CONTRACT,
        "horizons_bars": list(HORIZONS),
        "horizons_minutes": [horizon * 5 for horizon in HORIZONS],
        "fold_specification": dict(wf.FOLD_SPEC),
        "fold_count_per_horizon": 7,
        "model_configurations": MODEL_CONFIGS,
        "policies": list(POLICIES),
        "cost_contract": COST_CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "decile_contract": DECILE_CONTRACT,
        "hyperparameter_or_threshold_sweep_performed": False,
        "production_pass_gate_defined": False,
    }


def ensure_output_root(path: Path | str) -> Path:
    requested = Path(path).resolve()
    allowed = RESEARCH_ROOT.resolve()
    if requested != allowed:
        raise ExecutableReturnResearchError(
            f"return research output root must be exactly {allowed}; received {requested}"
        )
    return requested


def aggregate_regression_summary(
    fold_metrics: Sequence[Mapping[str, Any]],
    pooled_prediction_rows: Sequence[Mapping[str, Any]],
    pooled_baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = pd.DataFrame(fold_metrics)
    predictions = pd.DataFrame(pooled_prediction_rows)
    baselines = pd.DataFrame(pooled_baseline_rows)
    summaries: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for model_name in MODELS:
            for scope in ("ALL", *wf.SYMBOLS):
                selected_predictions = predictions.loc[
                    (predictions["horizon_bars"] == horizon)
                    & (predictions["model_name"] == model_name)
                ].sort_values(["fold_id", "timestamp"], kind="mergesort")
                selected_baseline = baselines.loc[
                    (baselines["horizon_bars"] == horizon)
                ].sort_values(["fold_id", "timestamp"], kind="mergesort")
                if scope != "ALL":
                    selected_predictions = selected_predictions.loc[
                        selected_predictions["symbol"] == scope
                    ]
                    selected_baseline = selected_baseline.loc[
                        selected_baseline["symbol"] == scope
                    ]
                if len(selected_predictions) != len(selected_baseline) or selected_predictions.empty:
                    raise ExecutableReturnResearchError(
                        "pooled regression prediction rows are incomplete"
                    )
                if not selected_predictions[["fold_id", "timestamp", "symbol"]].reset_index(
                    drop=True
                ).equals(
                    selected_baseline[["fold_id", "timestamp", "symbol"]].reset_index(
                        drop=True
                    )
                ):
                    raise ExecutableReturnResearchError(
                        "candidate/baseline prediction streams differ"
                    )
                metrics = regression_metrics(
                    selected_predictions["prediction_bps"],
                    selected_predictions["target_bps"],
                    selected_baseline["baseline_prediction_bps"],
                )
                relevant_folds = folds.loc[
                    (folds["horizon_bars"] == horizon)
                    & (folds["model_name"] == model_name)
                    & (folds["metric_scope"] == scope)
                ]
                pearson = relevant_folds["pearson_correlation"].dropna().to_numpy(
                    dtype=np.float64
                )
                magnitude_pearson = relevant_folds[
                    "absolute_magnitude_pearson_correlation"
                ].dropna().to_numpy(dtype=np.float64)
                spreads = relevant_folds[
                    "top_minus_bottom_realized_spread_bps"
                ].dropna().to_numpy(dtype=np.float64)
                magnitude_rows, monotonic = magnitude_bucket_diagnostics(
                    selected_predictions["prediction_bps"],
                    selected_predictions["target_bps"],
                )
                for row in magnitude_rows:
                    bucket_rows.append(
                        {
                            "aggregation": "pooled_oos",
                            "horizon_bars": horizon,
                            "horizon_minutes": horizon * 5,
                            "model_name": model_name,
                            "fold_id": "ALL",
                            "metric_scope": scope,
                            **row,
                        }
                    )
                summaries.append(
                    {
                        "horizon_bars": horizon,
                        "horizon_minutes": horizon * 5,
                        "model_name": model_name,
                        "metric_scope": scope,
                        **metrics,
                        "fold_count": int(len(relevant_folds)),
                        "percentage_folds_pearson_positive": (
                            float(100.0 * np.mean(pearson > 0.0))
                            if len(pearson)
                            else None
                        ),
                        "percentage_folds_absolute_magnitude_pearson_positive": (
                            float(100.0 * np.mean(magnitude_pearson > 0.0))
                            if len(magnitude_pearson)
                            else None
                        ),
                        "percentage_folds_top_bottom_spread_positive": (
                            float(100.0 * np.mean(spreads > 0.0))
                            if len(spreads)
                            else None
                        ),
                        "magnitude_buckets_monotonic": monotonic,
                        "magnitude_bucket_10_minus_1_realized_absolute_spread_bps": (
                            None
                            if not magnitude_rows
                            else float(
                                magnitude_rows[-1][
                                    "mean_absolute_realized_return_bps"
                                ]
                                - magnitude_rows[0][
                                    "mean_absolute_realized_return_bps"
                                ]
                            )
                        ),
                    }
                )
    return summaries, bucket_rows


def _number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def build_markdown_report(
    *,
    experiment_id: str,
    experiment_digest: str,
    regression_summary: Sequence[Mapping[str, Any]],
    policy_summary: Sequence[Mapping[str, Any]],
) -> str:
    regression = [row for row in regression_summary if row["metric_scope"] == "ALL"]
    policies = list(policy_summary)
    any_survives_5 = any(bool(row["survives_5bps"]) for row in policies)
    any_both = any(bool(row["both_symbols_positive_at_5bps"]) for row in policies)
    best = max(policies, key=lambda row: float(row["overall_net_return_5bps"]))
    monotonic = [
        row
        for row in regression
        if row["model_name"] != "training_mean_baseline"
        and row["magnitude_buckets_monotonic"] is True
    ]
    recommendation = (
        "At least one strategy survives 5 bps; validate it on future untouched data "
        "without adding stricter thresholds."
        if any_survives_5
        else "No return-prediction strategy survives 5 bps. Move to feature "
        "enrichment / regime-aware research rather than further threshold tuning."
    )
    lines = [
        "# Economically Aligned Executable-Return Research",
        "",
        "## Technical summary",
        "",
        f"- Experiment `{experiment_id}` (`{experiment_digest}`) is exposed historical "
        "research only. There is no production pass gate or deployment authorization.",
        f"- Any policy survives 5 bps: **{any_survives_5}**. Any policy has both BTC "
        f"and ETH positive at 5 bps: **{any_both}**.",
        f"- Strongest 5-bps stream: **{best['strategy_id']} / {best['policy_name']}**, "
        f"return {_percent(best['overall_net_return_5bps'])}, average gross edge "
        f"{_number(best['mean_gross_bps_per_active_trade'])} bps/trade.",
        f"- Fully monotonic pooled magnitude buckets among non-baselines: "
        f"**{len(monotonic)} of 6** horizon/model combinations.",
        f"- **Recommendation:** {recommendation}",
        "",
        "## Return-ranking evidence",
        "",
        "Metrics pool all seven outer-test folds. RMSE ratio compares each model with "
        "the matching fold's constant training-mean baseline.",
        "",
        "| Horizon | Model | MAE bps | RMSE bps | RMSE/base | Pearson | Spearman | Sign accuracy | Top-bottom bps | Pearson-positive folds | Spread-positive folds | Magnitude monotonic |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regression:
        lines.append(
            "| {horizon} | {model_name} | {mae} | {rmse} | {ratio} | {pearson} | "
            "{spearman} | {sign} | {spread} | {pearson_folds} | {spread_folds} | "
            "{monotonic} |".format(
                **row,
                horizon=HORIZON_LABELS[int(row["horizon_bars"])],
                mae=_number(row["mae_bps"]),
                rmse=_number(row["rmse_bps"]),
                ratio=_number(row["candidate_baseline_rmse_ratio"]),
                pearson=_number(row["pearson_correlation"]),
                spearman=_number(row["spearman_correlation"]),
                sign=_percent(row["sign_accuracy"]),
                spread=_number(row["top_minus_bottom_realized_spread_bps"]),
                pearson_folds=(
                    "n/a"
                    if row["percentage_folds_pearson_positive"] is None
                    else f"{float(row['percentage_folds_pearson_positive']):.2f}%"
                ),
                spread_folds=(
                    "n/a"
                    if row["percentage_folds_top_bottom_spread_positive"] is None
                    else f"{float(row['percentage_folds_top_bottom_spread_positive']):.2f}%"
                ),
                monotonic=str(row["magnitude_buckets_monotonic"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cost-adjusted policy results",
            "",
            "Active percentage uses all scheduled BTC/ETH sleeve signals. Returns, Sharpe, "
            "drawdown, and symbol results reuse the validated 50/50 sleeve accounting.",
            "",
            "| Horizon | Model | Policy | Active | Active % | Avg gross bps | Median gross bps | Break-even bps | Return 0 | Return 2 | Return 5 | Return 10 | Sharpe 5 | Max DD 5 | Positive folds 5 | BTC 5 | ETH 5 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policies:
        lines.append(
            "| {horizon} | {model_name} | {policy_name} | {active_trade_count} | "
            "{active} | {mean} | {median} | {breakeven} | {r0} | {r2} | {r5} | "
            "{r10} | {sharpe} | {dd} | {positive} | {btc} | {eth} |".format(
                **row,
                horizon=HORIZON_LABELS[int(row["horizon_bars"])],
                active=_percent(row["active_fraction"]),
                mean=_number(row["mean_gross_bps_per_active_trade"]),
                median=_number(row["median_gross_bps_per_active_trade"]),
                breakeven=_number(
                    row["approximate_break_even_round_trip_cost_bps"]
                ),
                r0=_percent(row["overall_net_return_0bps"]),
                r2=_percent(row["overall_net_return_2bps"]),
                r5=_percent(row["overall_net_return_5bps"]),
                r10=_percent(row["overall_net_return_10bps"]),
                sharpe=_number(row["daily_sharpe_5bps"]),
                dd=_percent(row["maximum_drawdown_5bps"]),
                positive=f"{float(row['positive_fold_percentage_5bps']):.2f}%",
                btc=_percent(row["btc_net_return_5bps"]),
                eth=_percent(row["eth_net_return_5bps"]),
            )
        )
    lines.extend(
        [
            "",
            "## Target, chronology, and model specification",
            "",
            "For feature timestamp t and horizon h, the target is "
            "`(close[t+h] / open[t+1] - 1) * 10000` bps. The signal is available only "
            "after bar t completes. Missing next-bar entries or horizon exits fail closed. "
            "All seven folds use a 120-day rolling train, 30-day OOS test, 30-day step, "
            "with every training target exit strictly before test start.",
            "",
            "The fixed models are Ridge(alpha=1.0, solver=lsqr) after StandardScaler; "
            "HistGradientBoostingRegressor with 100 iterations, 0.1 learning rate, 31 "
            "leaves, minimum leaf 20, no early stopping, and random_state 1729; plus a "
            "constant training-mean return baseline. No hyperparameter sweep occurred.",
            "",
            "## Execution and robustness remained unchanged",
            "",
            "Economic policies use outer-test predictions only. q90/q95 thresholds come "
            "only from absolute same-fold training predictions. Trades enter next-bar open, "
            "exit at horizon close, never overlap per symbol, never carry across folds, use "
            "fixed 50% BTC/ETH sleeves, 1x leverage, and no flat-sleeve reallocation. Costs "
            "are synthetic round-trip stress assumptions, not claimed venue fees.",
            "",
            "## Limitations and next research branch",
            "",
            "The history is research-exposed rather than pristine holdout data. Funding, "
            "spread, market impact, additional latency, and liquidation remain unmodeled. "
            "Bucket monotonicity is descriptive and uses fixed pooled OOS deciles.",
            "",
            f"**Next branch:** {recommendation}",
            "",
            "## Further questions",
            "",
            "Future work should test whether enriched state, volatility, liquidity, and "
            "regime features can improve executable-return magnitude ranking on untouched "
            "data without introducing threshold selection on the test set.",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(
    *, output_root: Path | str = RESEARCH_ROOT
) -> dict[str, Any]:
    walkforward_summary, _, _ = bt.validate_walkforward_source(
        SOURCE_WALKFORWARD_DIRECTORY
    )
    validate_previous_research_source(
        SOURCE_BACKTEST_DIRECTORY,
        expected_id=SOURCE_BACKTEST_EXPERIMENT_ID,
        expected_digest=SOURCE_BACKTEST_EXPERIMENT_DIGEST,
        expected_directory_digest=SOURCE_BACKTEST_DIRECTORY_DIGEST,
    )
    validate_previous_research_source(
        SOURCE_SELECTIVITY_DIRECTORY,
        expected_id=SOURCE_SELECTIVITY_EXPERIMENT_ID,
        expected_digest=SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
        expected_directory_digest=SOURCE_SELECTIVITY_DIRECTORY_DIGEST,
    )
    if bt.directory_digest(SOURCE_WALKFORWARD_DIRECTORY) != SOURCE_WALKFORWARD_DIRECTORY_DIGEST:
        raise ExecutableReturnResearchError("walk-forward evidence directory changed")
    earlier, later = bt.source_dataset_paths(walkforward_summary)
    expected_raw = walkforward_summary["raw_source_digests"]
    protected_before = protected_source_hashes(
        earlier=earlier, later=later, expected_raw=expected_raw
    )
    raw_by_symbol = {
        symbol: wf.combine_raw_windows(
            earlier / f"raw_{symbol}.csv",
            later / f"raw_{symbol}.csv",
            symbol=symbol,
        )
        for symbol in wf.SYMBOLS
    }
    if not raw_by_symbol["BTCUSDT"].index.equals(raw_by_symbol["ETHUSDT"].index):
        raise ExecutableReturnResearchError("BTC/ETH raw grids differ")
    feature_contract = wf._feature_contract()
    if feature_contract["feature_contract_digest"] != walkforward_summary[
        "feature_contract"
    ]["feature_contract_digest"]:
        raise ExecutableReturnResearchError("feature contract changed")
    features_by_symbol = {
        symbol: wf.build_research_features(raw_by_symbol[symbol], symbol=symbol)[0]
        for symbol in wf.SYMBOLS
    }
    contract = experiment_contract(
        protected_hashes=protected_before,
        feature_contract_digest=feature_contract["feature_contract_digest"],
    )
    experiment_digest = bt.json_digest(contract)
    experiment_id = f"executable_return_{experiment_digest[:16]}"
    output = ensure_output_root(output_root)
    final_directory = output / experiment_id
    staging_directory = output / f".{experiment_id}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise ExecutableReturnResearchError(
            f"executable-return output already exists: {experiment_id}"
        )

    ordered_features = wf.canonical_feature_columns(True)
    trade_rows: list[dict[str, Any]] = []
    regression_fold_rows: list[dict[str, Any]] = []
    magnitude_bucket_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    pooled_prediction_rows: list[dict[str, Any]] = []
    pooled_baseline_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        target_rows = pd.concat(
            [
                build_executable_return_rows(
                    raw_by_symbol[symbol],
                    features_by_symbol[symbol],
                    symbol=symbol,
                    horizon_bars=horizon,
                )
                for symbol in wf.SYMBOLS
            ],
            ignore_index=True,
        )
        folds = wf.make_walkforward_folds(
            raw_by_symbol["BTCUSDT"].index[0],
            raw_by_symbol["BTCUSDT"].index[-1],
            horizon_bars=horizon,
        )
        if len(folds) != 7:
            raise ExecutableReturnResearchError(
                f"horizon {horizon} does not have seven complete folds"
            )
        for fold in folds:
            train, test = select_executable_fold_rows(target_rows, fold)
            X_train = train.loc[:, ordered_features].to_numpy(dtype=np.float64)
            y_train = train["executable_return_bps"].to_numpy(dtype=np.float64)
            X_test = test.loc[:, ordered_features].to_numpy(dtype=np.float64)
            baseline_mean = float(np.mean(y_train))
            baseline_test = np.full(len(test), baseline_mean, dtype=np.float64)

            for model_name in MODELS:
                train_predictions, test_predictions, _ = fit_regression_predictions(
                    model_name, X_train, y_train, X_test
                )
                scored_test = test.loc[:, ["timestamp", "symbol"]].copy()
                scored_test["prediction_bps"] = test_predictions
                scored_test["prediction_origin"] = "outer_test"
                thresholds = {
                    policy_name: derive_absolute_prediction_threshold(
                        train_predictions, policy_name
                    )
                    for policy_name in POLICY_ORDER
                }
                for policy_name in POLICY_ORDER:
                    policy = policy_contract(policy_name)
                    threshold_rows.append(
                        {
                            "horizon_bars": horizon,
                            "horizon_minutes": horizon * 5,
                            "model_name": model_name,
                            "fold_id": fold.fold_id,
                            "policy_name": policy_name,
                            "training_prediction_count": int(len(train_predictions)),
                            "threshold_quantile": policy["threshold_quantile"],
                            "training_abs_prediction_threshold_bps": thresholds[
                                policy_name
                            ],
                            "threshold_source": policy["threshold_source"],
                            "test_predictions_consulted": False,
                            "test_returns_consulted": False,
                        }
                    )
                    for symbol in wf.SYMBOLS:
                        symbol_scores = scored_test.loc[
                            scored_test["symbol"] == symbol,
                            ["timestamp", "prediction_bps", "prediction_origin"],
                        ]
                        trade_rows.extend(
                            build_return_trade_rows(
                                experiment_id=experiment_id,
                                model_name=model_name,
                                horizon_bars=horizon,
                                policy_name=policy_name,
                                fold=fold,
                                symbol=symbol,
                                scored_test=symbol_scores,
                                raw=raw_by_symbol[symbol],
                                training_abs_threshold_bps=thresholds[policy_name],
                            )
                        )

                for scope in ("ALL", *wf.SYMBOLS):
                    mask = np.ones(len(test), dtype=bool)
                    if scope != "ALL":
                        mask = test["symbol"].to_numpy() == scope
                    predictions_scope = test_predictions[mask]
                    target_scope = test.loc[
                        mask, "executable_return_bps"
                    ].to_numpy(dtype=np.float64)
                    baseline_scope = baseline_test[mask]
                    metrics = regression_metrics(
                        predictions_scope, target_scope, baseline_scope
                    )
                    regression_fold_rows.append(
                        {
                            "horizon_bars": horizon,
                            "horizon_minutes": horizon * 5,
                            "model_name": model_name,
                            "fold_id": fold.fold_id,
                            "metric_scope": scope,
                            **metrics,
                        }
                    )
                    magnitude_rows, monotonic = magnitude_bucket_diagnostics(
                        predictions_scope, target_scope
                    )
                    for row in magnitude_rows:
                        magnitude_bucket_rows.append(
                            {
                                "aggregation": "fold",
                                "horizon_bars": horizon,
                                "horizon_minutes": horizon * 5,
                                "model_name": model_name,
                                "fold_id": fold.fold_id,
                                "metric_scope": scope,
                                **row,
                            }
                        )
                    regression_fold_rows[-1][
                        "magnitude_buckets_monotonic"
                    ] = monotonic
                for index, meta in enumerate(
                    test.loc[:, ["timestamp", "symbol"]].itertuples(index=False)
                ):
                    pooled_prediction_rows.append(
                        {
                            "horizon_bars": horizon,
                            "model_name": model_name,
                            "fold_id": fold.fold_id,
                            "timestamp": bt.canonical_utc(meta.timestamp),
                            "symbol": meta.symbol,
                            "prediction_bps": float(test_predictions[index]),
                            "target_bps": float(
                                test["executable_return_bps"].iloc[index]
                            ),
                        }
                    )
                    if model_name == "training_mean_baseline":
                        pooled_baseline_rows.append(
                            {
                                "horizon_bars": horizon,
                                "fold_id": fold.fold_id,
                                "timestamp": bt.canonical_utc(meta.timestamp),
                                "symbol": meta.symbol,
                                "baseline_prediction_bps": float(
                                    baseline_test[index]
                                ),
                            }
                        )

    ledger = as_return_trade_frame(trade_rows)
    fold_economics, overall_economics, curves = calculate_economic_metrics(ledger)
    policy_summary = build_policy_summary(ledger, overall_economics)
    regression_summary, pooled_buckets = aggregate_regression_summary(
        regression_fold_rows, pooled_prediction_rows, pooled_baseline_rows
    )
    magnitude_bucket_rows.extend(pooled_buckets)
    invariant_checks = validate_research_results(
        ledger=ledger,
        fold_metrics=fold_economics,
        overall_metrics=overall_economics,
        policy_summary=policy_summary,
        regression_fold_metrics=regression_fold_rows,
        magnitude_bucket_rows=magnitude_bucket_rows,
        curves=curves,
    )
    protected_after = protected_source_hashes(
        earlier=earlier, later=later, expected_raw=expected_raw
    )
    if protected_after != protected_before:
        raise ExecutableReturnResearchError("protected sources changed during run")
    invariant_checks.update(
        {
            "protected_hashes_unchanged": True,
            "previous_research_evidence_unchanged": True,
            "frozen_raw_sources_unchanged": True,
            "features_and_feature_ordering_unchanged": True,
            "candidate_and_incumbent_artifacts_unchanged": True,
            "validation_ledger_unchanged": True,
        }
    )

    any_survives_5 = any(bool(row["survives_5bps"]) for row in policy_summary)
    best = max(
        policy_summary, key=lambda row: float(row["overall_net_return_5bps"])
    )
    magnitude_meaningful = any(
        row["model_name"] != "training_mean_baseline"
        and row["metric_scope"] == "ALL"
        and row["absolute_magnitude_pearson_correlation"] is not None
        and float(row["absolute_magnitude_pearson_correlation"]) > 0.0
        and float(
            row["percentage_folds_absolute_magnitude_pearson_positive"] or 0.0
        )
        > 50.0
        and row[
            "magnitude_bucket_10_minus_1_realized_absolute_spread_bps"
        ]
        is not None
        and float(
            row["magnitude_bucket_10_minus_1_realized_absolute_spread_bps"]
        )
        > 0.0
        for row in regression_summary
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "research_only": True,
        "historical_periods_pristine_holdout": False,
        "production_candidate": False,
        "promotion_allowed": False,
        "live_execution_allowed": False,
        "deployment_authorization": False,
        "production_pass_gate_defined": False,
        "source_experiments": {
            "walkforward": {
                "experiment_id": SOURCE_WALKFORWARD_EXPERIMENT_ID,
                "experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
            },
            "backtest": {
                "experiment_id": SOURCE_BACKTEST_EXPERIMENT_ID,
                "experiment_digest": SOURCE_BACKTEST_EXPERIMENT_DIGEST,
            },
            "selectivity": {
                "experiment_id": SOURCE_SELECTIVITY_EXPERIMENT_ID,
                "experiment_digest": SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
            },
        },
        "target_contract": TARGET_CONTRACT,
        "fold_specification": dict(wf.FOLD_SPEC),
        "horizons_bars": list(HORIZONS),
        "model_configurations": MODEL_CONFIGS,
        "policies": list(POLICIES),
        "cost_contract": COST_CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "execution_limitations": LIMITATIONS,
        "decile_contract": DECILE_CONTRACT,
        "feature_contract_digest": feature_contract["feature_contract_digest"],
        "fold_count_per_horizon": 7,
        "regression_fold_metrics": regression_fold_rows,
        "regression_summary": regression_summary,
        "magnitude_bucket_diagnostics": magnitude_bucket_rows,
        "economic_fold_metrics": fold_economics,
        "economic_overall_metrics": overall_economics,
        "policy_summary": policy_summary,
        "policy_thresholds": threshold_rows,
        "decision_diagnostics": {
            "magnitude_prediction_meaningful": magnitude_meaningful,
            "any_strategy_survives_2bps": any(
                bool(row["survives_2bps"]) for row in policy_summary
            ),
            "any_strategy_survives_5bps": any_survives_5,
            "any_strategy_survives_10bps": any(
                bool(row["survives_10bps"]) for row in policy_summary
            ),
            "any_strategy_has_both_symbols_positive_at_5bps": any(
                bool(row["both_symbols_positive_at_5bps"])
                for row in policy_summary
            ),
            "strongest_horizon_model_policy_at_5bps": {
                "horizon_bars": best["horizon_bars"],
                "model_name": best["model_name"],
                "policy_name": best["policy_name"],
                "overall_net_return_5bps": best["overall_net_return_5bps"],
            },
            "automatic_q97_q99_filtering_allowed": False,
            "continue_threshold_tightening": False,
            "next_research_branch": (
                "future untouched confirmation without tighter thresholds"
                if any_survives_5
                else "feature enrichment / regime-aware research rather than further threshold tuning"
            ),
            "observations_are_deployment_authorization": False,
        },
        "protected_source_hashes_before": protected_before,
        "protected_source_hashes_after": protected_after,
        "protected_source_hashes_unchanged": True,
        "invariant_checks": invariant_checks,
        "all_invariants_reconciled": all(invariant_checks.values()),
        "trade_ledger_row_count": int(len(ledger)),
        "historical_research_exposure_warning": (
            "All periods are exposed historical research and are not a pristine holdout."
        ),
        "safety_contract": {
            "candidate_or_incumbent_artifacts_modified_or_written": False,
            "validation_ledger_modified_or_written": False,
            "runtime_or_live_execution_accessed_or_modified": False,
            "exchange_api_or_network_access_performed": False,
            "model_artifacts_written": False,
            "lstm_tcn_transformer_trained": False,
            "outputs_restricted_to": str(output),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir()
    paths = {
        "summary.json": staging_directory / "summary.json",
        "trade_ledger.csv": staging_directory / "trade_ledger.csv",
        "regression_fold_metrics.csv": staging_directory / "regression_fold_metrics.csv",
        "regression_summary.csv": staging_directory / "regression_summary.csv",
        "magnitude_buckets.csv": staging_directory / "magnitude_buckets.csv",
        "economic_fold_metrics.csv": staging_directory / "economic_fold_metrics.csv",
        "economic_overall_metrics.csv": staging_directory / "economic_overall_metrics.csv",
        "policy_summary.csv": staging_directory / "policy_summary.csv",
        "policy_thresholds.csv": staging_directory / "policy_thresholds.csv",
        "report.md": staging_directory / "report.md",
        "experiment_manifest.json": staging_directory / "experiment_manifest.json",
    }
    bt._write_json(paths["summary.json"], summary)
    ledger_output = ledger.copy()
    for column in ledger_output.columns:
        if pd.api.types.is_datetime64_any_dtype(ledger_output[column]):
            ledger_output[column] = ledger_output[column].map(bt.canonical_utc)
    bt._write_csv(
        paths["trade_ledger.csv"],
        ledger_output.to_dict("records"),
        RETURN_LEDGER_COLUMNS,
    )
    bt._write_csv(paths["regression_fold_metrics.csv"], regression_fold_rows)
    bt._write_csv(paths["regression_summary.csv"], regression_summary)
    bt._write_csv(paths["magnitude_buckets.csv"], magnitude_bucket_rows)
    bt._write_csv(paths["economic_fold_metrics.csv"], fold_economics)
    bt._write_csv(paths["economic_overall_metrics.csv"], overall_economics)
    bt._write_csv(paths["policy_summary.csv"], policy_summary)
    bt._write_csv(paths["policy_thresholds.csv"], threshold_rows)
    paths["report.md"].write_text(
        build_markdown_report(
            experiment_id=experiment_id,
            experiment_digest=experiment_digest,
            regression_summary=regression_summary,
            policy_summary=policy_summary,
        ),
        encoding="utf-8",
        newline="\n",
    )
    row_counts: dict[str, int | None] = {
        "summary.json": None,
        "trade_ledger.csv": len(ledger_output),
        "regression_fold_metrics.csv": len(regression_fold_rows),
        "regression_summary.csv": len(regression_summary),
        "magnitude_buckets.csv": len(magnitude_bucket_rows),
        "economic_fold_metrics.csv": len(fold_economics),
        "economic_overall_metrics.csv": len(overall_economics),
        "policy_summary.csv": len(policy_summary),
        "policy_thresholds.csv": len(threshold_rows),
        "report.md": None,
    }
    outputs = {
        name: {"sha256": bt.file_sha256(paths[name]), "row_count": count}
        for name, count in row_counts.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "created_at_utc": utc_now(),
        "research_only": True,
        "production_candidate": False,
        "promotion_allowed": False,
        "live_execution_allowed": False,
        "deployment_authorization": False,
        "production_pass_gate_defined": False,
        "experiment_contract": contract,
        "protected_source_hashes_unchanged_during_run": True,
        "outputs": outputs,
        "execution_environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": wf.sklearn.__version__,
        },
    }
    bt._write_json(paths["experiment_manifest.json"], manifest)
    staging_directory.rename(final_directory)
    summary["output_directory"] = str(final_directory)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline economically aligned executable-return research."
    )
    parser.add_argument("--output-root", type=Path, default=RESEARCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_research(output_root=args.output_root)
    except (
        ExecutableReturnResearchError,
        bt.SignalBacktestError,
        wf.SignalResearchError,
        OSError,
        ValueError,
    ) as exc:
        print(f"executable-return research failed closed: {exc}", file=sys.stderr)
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
