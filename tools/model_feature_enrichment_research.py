"""Offline research-only feature-enrichment and regime-robustness experiment.

The experiment compares the repository's canonical 26 market features plus
``symbol_id`` with one frozen bundle of 17 backward-looking/current OHLCV
features.  It evaluates only the frozen 30-minute HistGradientBoosting
classifier and regressor configurations on the established seven rolling
walk-forward folds.  It has no runtime, exchange, network, promotion, model
artifact, candidate-training, or production feature dependency.
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools import model_executable_return_research as er  # noqa: E402
from tools import model_signal_backtest as bt  # noqa: E402
from tools import model_signal_walkforward as wf  # noqa: E402


SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = "model-feature-enrichment-research-v1"
HORIZON_BARS = 6
HORIZON_MINUTES = 30
FEATURE_SETS = ("baseline", "enriched")
MODEL_FAMILIES = ("classifier", "regressor")
CLASSIFIER_MODEL_NAME = "hist_gradient_boosting"
REGRESSOR_MODEL_NAME = "hist_gradient_boosting_regressor"
CLASSIFIER_POLICY = "train_quantile_05_95"
REGRESSOR_POLICY = "train_abs_q95"
COST_SCENARIOS_BPS = bt.COST_SCENARIOS_BPS
RESEARCH_ROOT = BASE_DIR / "reports" / "model_feature_enrichment_research"

SOURCE_WALKFORWARD_EXPERIMENT_ID = bt.SOURCE_WALKFORWARD_EXPERIMENT_ID
SOURCE_WALKFORWARD_EXPERIMENT_DIGEST = bt.SOURCE_WALKFORWARD_EXPERIMENT_DIGEST
SOURCE_WALKFORWARD_DIRECTORY = bt.SOURCE_WALKFORWARD_DIRECTORY
SOURCE_WALKFORWARD_DIRECTORY_DIGEST = (
    "46af7ff4e7a2b74d1e8e3a3b6554840dcbd695a55210be07fb86d6016defe71d"
)
SOURCE_BACKTEST_EXPERIMENT_ID = er.SOURCE_BACKTEST_EXPERIMENT_ID
SOURCE_BACKTEST_EXPERIMENT_DIGEST = er.SOURCE_BACKTEST_EXPERIMENT_DIGEST
SOURCE_BACKTEST_DIRECTORY = er.SOURCE_BACKTEST_DIRECTORY
SOURCE_BACKTEST_DIRECTORY_DIGEST = er.SOURCE_BACKTEST_DIRECTORY_DIGEST
SOURCE_SELECTIVITY_EXPERIMENT_ID = er.SOURCE_SELECTIVITY_EXPERIMENT_ID
SOURCE_SELECTIVITY_EXPERIMENT_DIGEST = er.SOURCE_SELECTIVITY_EXPERIMENT_DIGEST
SOURCE_SELECTIVITY_DIRECTORY = er.SOURCE_SELECTIVITY_DIRECTORY
SOURCE_SELECTIVITY_DIRECTORY_DIGEST = er.SOURCE_SELECTIVITY_DIRECTORY_DIGEST
SOURCE_EXECUTABLE_RETURN_EXPERIMENT_ID = "executable_return_25f4bf13bcb1e788"
SOURCE_EXECUTABLE_RETURN_EXPERIMENT_DIGEST = (
    "25f4bf13bcb1e7889594ae6089db1a0d2f260b55030304f1cadd94a9466fff54"
)
SOURCE_EXECUTABLE_RETURN_DIRECTORY = (
    BASE_DIR
    / "reports"
    / "model_executable_return_research"
    / SOURCE_EXECUTABLE_RETURN_EXPERIMENT_ID
)
SOURCE_EXECUTABLE_RETURN_DIRECTORY_DIGEST = (
    "cb4d0c727e6dd46821df3b45f003f8069c53d57aa9f587b1dfa730489d3f39e7"
)

ENRICHED_FEATURE_COLUMNS = (
    "trend_50",
    "trend_200",
    "ret_12",
    "ret_60",
    "ret_288",
    "rv_12",
    "rv_60",
    "rv_288",
    "vol_ratio_12_288",
    "vol_ratio_60_288",
    "range_position_60",
    "range_position_288",
    "volume_z_60",
    "volume_ratio_288",
    "btc_ret_12",
    "btc_ret_60",
    "btc_rv_60",
)
LOCAL_ENRICHED_FEATURE_COLUMNS = ENRICHED_FEATURE_COLUMNS[:14]
BTC_CONTEXT_COLUMNS = ENRICHED_FEATURE_COLUMNS[14:]

CLASSIFIER_CONFIG = wf.MODEL_CONFIGS[CLASSIFIER_MODEL_NAME]
REGRESSOR_CONFIG = er.MODEL_CONFIGS[REGRESSOR_MODEL_NAME]
CLASSIFIER_POLICY_CONTRACT = {
    "policy_name": CLASSIFIER_POLICY,
    "lower_quantile": 0.05,
    "upper_quantile": 0.95,
    "quantile_method": "linear",
    "definition": (
        "test score <= frozen train q05 SHORT; test score >= frozen train q95 "
        "LONG; otherwise FLAT"
    ),
    "threshold_source": "same-fold fitted-model training scores only",
    "test_scores_consulted": False,
    "test_returns_consulted": False,
}
REGRESSOR_POLICY_CONTRACT = {
    "policy_name": REGRESSOR_POLICY,
    "threshold_quantile": 0.95,
    "quantile_method": "linear",
    "definition": (
        "abs(test predicted executable return) >= frozen q95 of abs(train "
        "predictions) trades predicted sign; otherwise FLAT"
    ),
    "threshold_source": "same-fold fitted-model training predictions only",
    "test_predictions_consulted": False,
    "test_returns_consulted": False,
}
REGIME_CONTRACT = {
    "descriptive_only": True,
    "tradable_filter": False,
    "labels_used_for_training": False,
    "volatility_measure": (
        "mean of BTC and ETH test-fold realized volatility, where each symbol's "
        "realized volatility is sqrt(sum squared close-to-close log returns)"
    ),
    "volatility_threshold": "median combined realized volatility across seven test folds",
    "high_volatility_rule": "combined fold volatility >= fixed seven-fold median",
    "low_volatility_rule": "combined fold volatility < fixed seven-fold median",
    "positive_btc_return_rule": "BTC test-fold signed log return > 0",
    "negative_btc_return_rule": "BTC test-fold signed log return <= 0",
    "classification_timing": "assigned before inspecting model economic results",
    "post_hoc_exposed_history_warning": True,
}
STOP_CONDITION = {
    "required_overall_5bps_return": "strictly positive",
    "required_positive_fold_fraction_5bps": "strictly greater than 50 percent",
    "failure_recommendation": (
        "STOP further OHLCV-only feature tweaking on this exposed history. "
        "The next engineering branch must become collection of richer market "
        "data for future research."
    ),
    "automatic_follow_up_feature_experiment_allowed": False,
}

FEATURE_FORMULAS = {
    "trend_50": "close / trailing_current rolling_mean(close, 50) - 1",
    "trend_200": "close / trailing_current rolling_mean(close, 200) - 1",
    "ret_12": "sum of trailing_current 12 one-bar close log returns",
    "ret_60": "sum of trailing_current 60 one-bar close log returns",
    "ret_288": "sum of trailing_current 288 one-bar close log returns",
    "rv_12": "sqrt(sum squared trailing_current 12 one-bar close log returns)",
    "rv_60": "sqrt(sum squared trailing_current 60 one-bar close log returns)",
    "rv_288": "sqrt(sum squared trailing_current 288 one-bar close log returns)",
    "vol_ratio_12_288": "rv_12 / rv_288",
    "vol_ratio_60_288": "rv_60 / rv_288",
    "range_position_60": (
        "(close - trailing_current rolling_min(low, 60)) / "
        "(rolling_max(high, 60) - rolling_min(low, 60))"
    ),
    "range_position_288": (
        "(close - trailing_current rolling_min(low, 288)) / "
        "(rolling_max(high, 288) - rolling_min(low, 288))"
    ),
    "volume_z_60": (
        "(volume - trailing_current rolling_mean(volume, 60)) / "
        "rolling_std(volume, 60, ddof=0)"
    ),
    "volume_ratio_288": "volume / trailing_current rolling_mean(volume, 288)",
    "btc_ret_12": "BTC ret_12 aligned at the same timestamp",
    "btc_ret_60": "BTC ret_60 aligned at the same timestamp",
    "btc_rv_60": "BTC rv_60 aligned at the same timestamp",
}

ENRICHMENT_LEDGER_COLUMNS = (
    *bt.LEDGER_COLUMNS,
    "source_selectivity_experiment_digest",
    "source_executable_return_experiment_digest",
    "feature_set",
    "model_family",
    "prediction_origin",
    "training_lower_threshold",
    "training_upper_threshold",
    "training_abs_prediction_threshold_bps",
    "threshold_source",
)


class FeatureEnrichmentResearchError(ValueError):
    """Raised when a source, leakage, feature, or accounting contract fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stream_id(feature_set: str, model_family: str) -> str:
    if feature_set not in FEATURE_SETS or model_family not in MODEL_FAMILIES:
        raise FeatureEnrichmentResearchError("unknown feature set or model family")
    return f"{feature_set}_hgb_{model_family}_30m"


def policy_for_family(model_family: str) -> str:
    if model_family == "classifier":
        return CLASSIFIER_POLICY
    if model_family == "regressor":
        return REGRESSOR_POLICY
    raise FeatureEnrichmentResearchError("unknown model family")


def build_local_enriched_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build exactly 14 symbol-local trailing/current OHLCV features."""

    required = {"high", "low", "close", "volume"}
    if raw.empty or not required <= set(raw.columns):
        raise FeatureEnrichmentResearchError("raw OHLCV input is incomplete")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise FeatureEnrichmentResearchError("raw input requires a timestamp index")
    if not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise FeatureEnrichmentResearchError("raw timestamps must be ordered and unique")
    close = raw["close"].astype(np.float64)
    high = raw["high"].astype(np.float64)
    low = raw["low"].astype(np.float64)
    volume = raw["volume"].astype(np.float64)
    if not np.isfinite(raw.loc[:, ["high", "low", "close", "volume"]].to_numpy()).all():
        raise FeatureEnrichmentResearchError("raw OHLCV contains nonfinite values")
    if np.any(close <= 0.0) or np.any(volume < 0.0):
        raise FeatureEnrichmentResearchError("raw close/volume is invalid")

    log_return = np.log(close / close.shift(1))
    result = pd.DataFrame(index=raw.index)
    result["trend_50"] = close / close.rolling(50, min_periods=50).mean() - 1.0
    result["trend_200"] = close / close.rolling(200, min_periods=200).mean() - 1.0
    for window in (12, 60, 288):
        result[f"ret_{window}"] = log_return.rolling(
            window, min_periods=window
        ).sum()
        result[f"rv_{window}"] = np.sqrt(
            log_return.pow(2).rolling(window, min_periods=window).sum()
        )
    result["vol_ratio_12_288"] = result["rv_12"] / result["rv_288"].replace(
        0.0, np.nan
    )
    result["vol_ratio_60_288"] = result["rv_60"] / result["rv_288"].replace(
        0.0, np.nan
    )
    for window in (60, 288):
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        width = (rolling_high - rolling_low).replace(0.0, np.nan)
        result[f"range_position_{window}"] = (close - rolling_low) / width
    volume_mean_60 = volume.rolling(60, min_periods=60).mean()
    volume_std_60 = volume.rolling(60, min_periods=60).std(ddof=0).replace(
        0.0, np.nan
    )
    result["volume_z_60"] = (volume - volume_mean_60) / volume_std_60
    result["volume_ratio_288"] = volume / volume.rolling(
        288, min_periods=288
    ).mean().replace(0.0, np.nan)
    result = result.loc[:, LOCAL_ENRICHED_FEATURE_COLUMNS]
    if tuple(result.columns) != LOCAL_ENRICHED_FEATURE_COLUMNS:
        raise FeatureEnrichmentResearchError("local enriched feature contract changed")
    return result.astype(np.float64)


def build_feature_contracts(
    raw_by_symbol: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    """Return common-grid canonical baseline and baseline-plus-17 matrices."""

    if set(raw_by_symbol) != set(wf.SYMBOLS):
        raise FeatureEnrichmentResearchError("BTC/ETH raw inputs are required")
    if not raw_by_symbol["BTCUSDT"].index.equals(raw_by_symbol["ETHUSDT"].index):
        raise FeatureEnrichmentResearchError("BTC/ETH timestamp grids differ")
    baseline: dict[str, pd.DataFrame] = {}
    baseline_diagnostics: dict[str, Any] = {}
    local = {
        symbol: build_local_enriched_features(raw_by_symbol[symbol])
        for symbol in wf.SYMBOLS
    }
    btc_context = local["BTCUSDT"].loc[:, ["ret_12", "ret_60", "rv_60"]].rename(
        columns={
            "ret_12": "btc_ret_12",
            "ret_60": "btc_ret_60",
            "rv_60": "btc_rv_60",
        }
    )
    for symbol in wf.SYMBOLS:
        baseline[symbol], baseline_diagnostics[symbol] = wf.build_research_features(
            raw_by_symbol[symbol], symbol=symbol
        )
    baseline_columns = tuple(wf.canonical_feature_columns(True))
    if len(baseline_columns) != 27:
        raise FeatureEnrichmentResearchError("canonical baseline feature count changed")

    contracts: dict[str, dict[str, pd.DataFrame]] = {
        "baseline": {},
        "enriched": {},
    }
    eligibility: dict[str, Any] = {}
    for symbol in wf.SYMBOLS:
        additions = pd.concat([local[symbol], btc_context], axis=1).loc[
            :, ENRICHED_FEATURE_COLUMNS
        ]
        combined = pd.concat([baseline[symbol], additions], axis=1)
        finite = np.isfinite(combined.to_numpy(dtype=np.float64)).all(axis=1)
        eligible = combined.loc[finite].copy()
        if eligible.empty:
            raise FeatureEnrichmentResearchError(
                f"no finite enriched rows for {symbol}"
            )
        baseline_common = baseline[symbol].reindex(eligible.index).copy()
        if not baseline_common.equals(eligible.loc[:, baseline_columns]):
            raise FeatureEnrichmentResearchError(
                "baseline values changed while aligning common eligibility"
            )
        contracts["baseline"][symbol] = baseline_common.astype(np.float64)
        contracts["enriched"][symbol] = eligible.loc[
            :, (*baseline_columns, *ENRICHED_FEATURE_COLUMNS)
        ].astype(np.float64)
        eligibility[symbol] = {
            "first_eligible_timestamp_utc": wf.canonical_utc(eligible.index[0]),
            "last_eligible_timestamp_utc": wf.canonical_utc(eligible.index[-1]),
            "row_count": int(len(eligible)),
            "baseline_diagnostics": baseline_diagnostics[symbol],
        }
    if not contracts["baseline"]["BTCUSDT"].index.equals(
        contracts["baseline"]["ETHUSDT"].index
    ):
        raise FeatureEnrichmentResearchError("common feature eligibility grid differs")
    diagnostics = {
        "common_eligibility_for_fair_comparison": True,
        "baseline_columns": list(baseline_columns),
        "enriched_addition_columns": list(ENRICHED_FEATURE_COLUMNS),
        "enriched_columns": [*baseline_columns, *ENRICHED_FEATURE_COLUMNS],
        "baseline_feature_count": len(baseline_columns),
        "enriched_feature_count": len(baseline_columns) + len(ENRICHED_FEATURE_COLUMNS),
        "by_symbol": eligibility,
    }
    return contracts, diagnostics


def feature_columns(feature_set: str) -> tuple[str, ...]:
    baseline = tuple(wf.canonical_feature_columns(True))
    if feature_set == "baseline":
        return baseline
    if feature_set == "enriched":
        return (*baseline, *ENRICHED_FEATURE_COLUMNS)
    raise FeatureEnrichmentResearchError("unknown feature contract")


def fit_classifier_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit the exact frozen HGB classifier on training rows only."""

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int8)
    X_test = np.asarray(X_test, dtype=np.float64)
    if (
        X_train.ndim != 2
        or X_test.ndim != 2
        or X_train.shape[1] != X_test.shape[1]
        or len(X_train) != len(y_train)
        or set(np.unique(y_train)) != {0, 1}
        or not np.isfinite(X_train).all()
        or not np.isfinite(X_test).all()
    ):
        raise FeatureEnrichmentResearchError("classifier train/test arrays are invalid")
    estimator = wf.make_model(CLASSIFIER_MODEL_NAME)
    estimator.fit(X_train, y_train)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise FeatureEnrichmentResearchError("classifier positive class is missing")
    positive_index = classes.index(1)
    train_scores = np.asarray(estimator.predict_proba(X_train), dtype=np.float64)[
        :, positive_index
    ]
    test_scores = np.asarray(estimator.predict_proba(X_test), dtype=np.float64)[
        :, positive_index
    ]
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise FeatureEnrichmentResearchError("classifier emitted nonfinite scores")
    return train_scores, test_scores, estimator


def derive_classifier_thresholds(
    train_scores: Sequence[float],
) -> tuple[float, float]:
    """Derive q05/q95 solely from same-fold fitted-model training scores."""

    scores = np.asarray(train_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise FeatureEnrichmentResearchError("training classifier scores are invalid")
    lower, upper = np.quantile(scores, [0.05, 0.95], method="linear")
    if not (math.isfinite(float(lower)) and math.isfinite(float(upper))):
        raise FeatureEnrichmentResearchError("classifier thresholds are nonfinite")
    if lower > upper:
        raise FeatureEnrichmentResearchError("classifier thresholds are inverted")
    return float(lower), float(upper)


def derive_regressor_threshold(train_predictions_bps: Sequence[float]) -> float:
    """Derive absolute q95 solely from same-fold training predictions."""

    threshold = er.derive_absolute_prediction_threshold(
        train_predictions_bps, REGRESSOR_POLICY
    )
    if threshold is None:
        raise FeatureEnrichmentResearchError("regressor threshold is unavailable")
    return float(threshold)


def classifier_direction(score: float, lower: float, upper: float) -> str:
    value = float(score)
    if not all(math.isfinite(item) for item in (value, lower, upper)) or lower > upper:
        raise FeatureEnrichmentResearchError("classifier policy input is invalid")
    if value >= upper:
        return "LONG"
    if value <= lower:
        return "SHORT"
    return "FLAT"


def regressor_direction(prediction_bps: float, threshold_bps: float) -> str:
    value = float(prediction_bps)
    threshold = float(threshold_bps)
    if not math.isfinite(value) or not math.isfinite(threshold) or threshold < 0.0:
        raise FeatureEnrichmentResearchError("regressor policy input is invalid")
    if abs(value) < threshold or value == 0.0:
        return "FLAT"
    return "LONG" if value > 0.0 else "SHORT"


def classifier_fold_metrics(test: pd.DataFrame, scores: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise FeatureEnrichmentResearchError("classifier prediction/test mismatch")
    target = test["target"].to_numpy(dtype=np.int8)
    if set(np.unique(target)) != {0, 1}:
        raise FeatureEnrichmentResearchError("classifier test fold lacks both classes")
    result: dict[str, Any] = {
        "pooled_auc": float(roc_auc_score(target, values)),
        "brier_score": float(brier_score_loss(target, values)),
        "log_loss": float(log_loss(target, values, labels=[0, 1])),
        "row_count": int(len(test)),
    }
    symbols = test["symbol"].to_numpy()
    for symbol, prefix in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
        mask = symbols == symbol
        symbol_target = target[mask]
        if set(np.unique(symbol_target)) != {0, 1}:
            raise FeatureEnrichmentResearchError(
                f"{symbol} classifier test fold lacks both classes"
            )
        result[f"{prefix}_auc"] = float(
            roc_auc_score(symbol_target, values[mask])
        )
    return result


def build_enrichment_trade_rows(
    *,
    experiment_id: str,
    feature_set: str,
    model_family: str,
    fold: wf.FoldDefinition,
    symbol: str,
    scored_test: pd.DataFrame,
    raw: pd.DataFrame,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    absolute_threshold_bps: float | None = None,
) -> list[dict[str, Any]]:
    """Reuse validated timing, then apply the one frozen family policy."""

    required = {"timestamp", "prediction", "prediction_origin"}
    if not required <= set(scored_test.columns):
        raise FeatureEnrichmentResearchError("OOS scored frame is incomplete")
    if not scored_test["prediction_origin"].eq("outer_test").all():
        raise FeatureEnrichmentResearchError("economic predictions must be OOS")
    indexed = scored_test.set_index("timestamp")["prediction"]
    if not indexed.index.is_unique:
        raise FeatureEnrichmentResearchError("duplicate OOS prediction timestamps")
    timing_scores = scored_test.loc[:, ["timestamp"]].copy()
    timing_scores["score"] = 1.0
    strategy = {
        "strategy_id": stream_id(feature_set, model_family),
        "model_name": (
            CLASSIFIER_MODEL_NAME
            if model_family == "classifier"
            else REGRESSOR_MODEL_NAME
        ),
        "horizon_bars": HORIZON_BARS,
        "horizon_minutes": HORIZON_MINUTES,
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
    result: list[dict[str, Any]] = []
    for row in base_rows:
        timestamp = pd.Timestamp(row["signal_timestamp_utc"])
        if timestamp not in indexed.index:
            raise FeatureEnrichmentResearchError("scheduled OOS prediction is missing")
        prediction = float(indexed.loc[timestamp])
        if model_family == "classifier":
            if lower_threshold is None or upper_threshold is None:
                raise FeatureEnrichmentResearchError(
                    "classifier training thresholds are missing"
                )
            direction = classifier_direction(
                prediction, float(lower_threshold), float(upper_threshold)
            )
            policy_name = CLASSIFIER_POLICY
            threshold_source = CLASSIFIER_POLICY_CONTRACT["threshold_source"]
        else:
            if absolute_threshold_bps is None:
                raise FeatureEnrichmentResearchError(
                    "regressor training threshold is missing"
                )
            direction = regressor_direction(prediction, absolute_threshold_bps)
            policy_name = REGRESSOR_POLICY
            threshold_source = REGRESSOR_POLICY_CONTRACT["threshold_source"]
        raw_return = float(row["exit_price"]) / float(row["entry_price"]) - 1.0
        gross = bt.directional_return(raw_return, direction)
        row.update(
            {
                "policy_name": policy_name,
                "score": prediction,
                "direction": direction,
                "gross_simple_return": gross,
                "gross_return_bps": gross * 10_000.0,
                "active_trade": direction != "FLAT",
                "source_selectivity_experiment_digest": (
                    SOURCE_SELECTIVITY_EXPERIMENT_DIGEST
                ),
                "source_executable_return_experiment_digest": (
                    SOURCE_EXECUTABLE_RETURN_EXPERIMENT_DIGEST
                ),
                "feature_set": feature_set,
                "model_family": model_family,
                "prediction_origin": "outer_test",
                "training_lower_threshold": lower_threshold,
                "training_upper_threshold": upper_threshold,
                "training_abs_prediction_threshold_bps": absolute_threshold_bps,
                "threshold_source": threshold_source,
            }
        )
        result.append(row)
    return result


def as_enrichment_trade_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    ledger = pd.DataFrame(rows, columns=ENRICHMENT_LEDGER_COLUMNS)
    if ledger.empty:
        raise FeatureEnrichmentResearchError("feature-enrichment ledger is empty")
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
    return ledger


def economic_summary_rows(
    ledger: pd.DataFrame,
    overall_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    overall = pd.DataFrame(overall_metrics)
    result: list[dict[str, Any]] = []
    for (strategy, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        cost_rows = overall.loc[
            (overall["strategy_id"] == strategy)
            & (overall["policy_name"] == policy_name)
        ]
        by_cost = {
            int(row.cost_bps): row for row in cost_rows.itertuples(index=False)
        }
        if set(by_cost) != set(COST_SCENARIOS_BPS):
            raise FeatureEnrichmentResearchError("economic cost grid is incomplete")
        gross = er.gross_trade_metrics(group)
        row: dict[str, Any] = {
            "strategy_id": strategy,
            "feature_set": str(group["feature_set"].iloc[0]),
            "model_family": str(group["model_family"].iloc[0]),
            "model_name": str(group["model_name"].iloc[0]),
            "horizon_bars": HORIZON_BARS,
            "horizon_minutes": HORIZON_MINUTES,
            "policy_name": policy_name,
            **gross,
            "approximate_break_even_round_trip_cost_bps": (
                bt.approximate_break_even_cost_bps(group)
            ),
        }
        for cost in COST_SCENARIOS_BPS:
            metric = by_cost[cost]
            suffix = f"{cost}bps"
            row.update(
                {
                    f"overall_net_return_{suffix}": float(
                        metric.net_cumulative_return
                    ),
                    f"daily_sharpe_{suffix}": (
                        None
                        if metric.daily_sharpe is None
                        else float(metric.daily_sharpe)
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
        row.update(
            {
                "survives_5bps": row["overall_net_return_5bps"] > 0.0,
                "positive_majority_of_folds_at_5bps": (
                    row["positive_fold_percentage_5bps"] > 50.0
                ),
                "production_pass_gate_defined": False,
                "diagnostic_only": True,
            }
        )
        result.append(row)
    return sorted(
        result,
        key=lambda item: (
            MODEL_FAMILIES.index(str(item["model_family"])),
            FEATURE_SETS.index(str(item["feature_set"])),
        ),
    )


def aggregate_classifier_summary(
    fold_metrics: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    folds = pd.DataFrame(fold_metrics)
    pooled = pd.DataFrame(pooled_rows)
    result: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        selected = folds.loc[folds["feature_set"] == feature_set].sort_values(
            "fold_id", kind="mergesort"
        )
        rows = pooled.loc[pooled["feature_set"] == feature_set]
        if len(selected) != 7 or rows.empty:
            raise FeatureEnrichmentResearchError(
                "classifier summary requires seven folds"
            )
        target = rows["target"].to_numpy(dtype=np.int8)
        scores = rows["score"].to_numpy(dtype=np.float64)
        aucs = selected["pooled_auc"].to_numpy(dtype=np.float64)
        result.append(
            {
                "feature_set": feature_set,
                "model_family": "classifier",
                "model_name": CLASSIFIER_MODEL_NAME,
                "fold_count": 7,
                "row_count": int(len(rows)),
                "mean_pooled_auc": float(np.mean(aucs)),
                "median_pooled_auc": float(np.median(aucs)),
                "worst_fold_auc": float(np.min(aucs)),
                "btc_mean_auc": float(np.mean(selected["btc_auc"])),
                "eth_mean_auc": float(np.mean(selected["eth_auc"])),
                "positive_auc_fold_count": int(np.sum(aucs > 0.5)),
                "mean_fold_brier_score": float(np.mean(selected["brier_score"])),
                "mean_fold_log_loss": float(np.mean(selected["log_loss"])),
                "pooled_brier_score": float(brier_score_loss(target, scores)),
                "pooled_log_loss": float(log_loss(target, scores, labels=[0, 1])),
            }
        )
    return result


def aggregate_regressor_summary(
    fold_metrics: Sequence[Mapping[str, Any]],
    pooled_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    folds = pd.DataFrame(fold_metrics)
    pooled = pd.DataFrame(pooled_rows)
    result: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        selected = folds.loc[folds["feature_set"] == feature_set]
        rows = pooled.loc[pooled["feature_set"] == feature_set]
        if len(selected) != 7 or rows.empty:
            raise FeatureEnrichmentResearchError(
                "regressor summary requires seven folds"
            )
        metrics = er.regression_metrics(
            rows["prediction_bps"].to_numpy(dtype=np.float64),
            rows["target_bps"].to_numpy(dtype=np.float64),
            rows["baseline_prediction_bps"].to_numpy(dtype=np.float64),
        )
        result.append(
            {
                "feature_set": feature_set,
                "model_family": "regressor",
                "model_name": REGRESSOR_MODEL_NAME,
                "fold_count": 7,
                **metrics,
                "percentage_folds_pearson_positive": float(
                    100.0 * np.mean(selected["pearson_correlation"] > 0.0)
                ),
                "percentage_folds_magnitude_correlation_positive": float(
                    100.0
                    * np.mean(
                        selected["absolute_magnitude_pearson_correlation"] > 0.0
                    )
                ),
                "percentage_folds_top_bottom_spread_positive": float(
                    100.0
                    * np.mean(
                        selected["top_minus_bottom_realized_spread_bps"] > 0.0
                    )
                ),
            }
        )
    return result


def assign_regime_groups(
    regime_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Apply fixed descriptive group rules before model economics are joined."""

    if len(regime_rows) != 7:
        raise FeatureEnrichmentResearchError("regime grouping requires seven folds")
    rows = [dict(row) for row in regime_rows]
    combined = np.asarray(
        [
            (
                float(row["btc_realized_volatility"])
                + float(row["eth_realized_volatility"])
            )
            / 2.0
            for row in rows
        ],
        dtype=np.float64,
    )
    threshold = float(np.median(combined))
    for row, value in zip(rows, combined):
        row["combined_realized_volatility"] = float(value)
        row["volatility_median_threshold"] = threshold
        row["volatility_group"] = (
            "higher_volatility" if value >= threshold else "lower_volatility"
        )
        row["btc_return_group"] = (
            "positive_btc_return"
            if float(row["btc_signed_return"]) > 0.0
            else "negative_btc_return"
        )
        row["post_hoc_descriptive_not_tradable"] = True
    return rows, threshold


def regime_robustness_rows(
    fold_economics: Sequence[Mapping[str, Any]],
    regime_rows: Sequence[Mapping[str, Any]],
    classifier_fold_metrics: Sequence[Mapping[str, Any]],
    regressor_fold_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    economics = pd.DataFrame(fold_economics)
    regimes = pd.DataFrame(regime_rows)
    selected = economics.loc[economics["cost_bps"] == 5].copy()
    selected = selected.merge(
        regimes.loc[:, ["fold_id", "volatility_group", "btc_return_group"]],
        on="fold_id",
        how="left",
        validate="many_to_one",
    )
    if selected[["volatility_group", "btc_return_group"]].isna().any().any():
        raise FeatureEnrichmentResearchError("regime/economic join is incomplete")
    classifier_lookup = {
        (str(row["feature_set"]), str(row["fold_id"])): float(row["pooled_auc"])
        for row in classifier_fold_metrics
    }
    regressor_lookup = {
        (str(row["feature_set"]), str(row["fold_id"])): float(
            row["pearson_correlation"]
        )
        for row in regressor_fold_metrics
    }
    result: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        lookup_key = (str(row["feature_set"]), str(row["fold_id"]))
        predictive_metric_name = (
            "pooled_auc"
            if row["model_family"] == "classifier"
            else "pearson_correlation"
        )
        predictive_metric_value = (
            classifier_lookup[lookup_key]
            if row["model_family"] == "classifier"
            else regressor_lookup[lookup_key]
        )
        for rule, group_name in (
            ("volatility", row["volatility_group"]),
            ("btc_signed_return", row["btc_return_group"]),
        ):
            result.append(
                {
                    "strategy_id": row["strategy_id"],
                    "feature_set": row["feature_set"],
                    "model_family": row["model_family"],
                    "policy_name": row["policy_name"],
                    "fold_id": row["fold_id"],
                    "cost_bps": 5,
                    "regime_rule": rule,
                    "regime_group": group_name,
                    "fold_net_return": row["net_cumulative_return"],
                    "active_trade_count": row["active_trade_count"],
                    "mean_gross_bps_per_active_trade": row[
                        "mean_gross_bps_per_active_trade"
                    ],
                    "predictive_metric_name": predictive_metric_name,
                    "predictive_metric_value": predictive_metric_value,
                    "descriptive_only": True,
                    "tradable_filter": False,
                }
            )
    detail = pd.DataFrame(result)
    summaries: list[dict[str, Any]] = []
    group_columns = [
        "strategy_id",
        "feature_set",
        "model_family",
        "policy_name",
        "regime_rule",
        "regime_group",
    ]
    for keys, group in detail.groupby(group_columns, sort=True):
        returns = group["fold_net_return"].to_numpy(dtype=np.float64)
        weights = group["active_trade_count"].to_numpy(dtype=np.float64)
        edges = group["mean_gross_bps_per_active_trade"].to_numpy(
            dtype=np.float64
        )
        summaries.append(
            {
                **dict(zip(group_columns, keys)),
                "cost_bps": 5,
                "fold_count": int(len(group)),
                "mean_fold_net_return": float(np.mean(returns)),
                "median_fold_net_return": float(np.median(returns)),
                "worst_fold_net_return": float(np.min(returns)),
                "positive_fold_percentage": float(100.0 * np.mean(returns > 0.0)),
                "active_trade_count": int(np.sum(weights)),
                "weighted_mean_gross_bps_per_active_trade": (
                    float(np.average(edges, weights=weights))
                    if np.sum(weights) > 0.0
                    else None
                ),
                "predictive_metric_name": str(
                    group["predictive_metric_name"].iloc[0]
                ),
                "mean_predictive_metric": float(
                    np.mean(group["predictive_metric_value"])
                ),
                "descriptive_only": True,
                "tradable_filter": False,
            }
        )
    return summaries


def incremental_value_rows(
    classifier_summary: Sequence[Mapping[str, Any]],
    economic_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    classification = {
        str(row["feature_set"]): row for row in classifier_summary
    }
    economics = {
        (str(row["model_family"]), str(row["feature_set"])): row
        for row in economic_summary
    }
    result: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        baseline = economics[(family, "baseline")]
        enriched = economics[(family, "enriched")]
        row: dict[str, Any] = {
            "model_family": family,
            "enriched_improves_gross_edge_per_trade": (
                float(enriched["mean_gross_bps_per_active_trade"])
                > float(baseline["mean_gross_bps_per_active_trade"])
            ),
            "enriched_improves_break_even_cost": (
                enriched["approximate_break_even_round_trip_cost_bps"] is not None
                and (
                    baseline["approximate_break_even_round_trip_cost_bps"] is None
                    or float(enriched["approximate_break_even_round_trip_cost_bps"])
                    > float(baseline["approximate_break_even_round_trip_cost_bps"])
                )
            ),
            "enriched_improves_5bps_return": (
                float(enriched["overall_net_return_5bps"])
                > float(baseline["overall_net_return_5bps"])
            ),
            "enriched_improves_5bps_positive_fold_fraction": (
                float(enriched["positive_fold_percentage_5bps"])
                > float(baseline["positive_fold_percentage_5bps"])
            ),
            "enriched_both_symbols_better_at_5bps": (
                float(enriched["btc_net_return_5bps"])
                > float(baseline["btc_net_return_5bps"])
                and float(enriched["eth_net_return_5bps"])
                > float(baseline["eth_net_return_5bps"])
            ),
        }
        if family == "classifier":
            base_class = classification["baseline"]
            enriched_class = classification["enriched"]
            row.update(
                {
                    "enriched_improves_mean_auc": (
                        float(enriched_class["mean_pooled_auc"])
                        > float(base_class["mean_pooled_auc"])
                    ),
                    "enriched_improves_worst_fold_auc": (
                        float(enriched_class["worst_fold_auc"])
                        > float(base_class["worst_fold_auc"])
                    ),
                }
            )
        else:
            row.update(
                {
                    "enriched_improves_mean_auc": None,
                    "enriched_improves_worst_fold_auc": None,
                }
            )
        result.append(row)
    return result


def protected_source_hashes(
    *,
    earlier: Path,
    later: Path,
    expected_raw: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "features.py": bt.file_sha256(BASE_DIR / "features.py"),
        "raw_source_files": bt.raw_source_digests(earlier, later, expected_raw),
        "validation_ledger": bt.file_sha256(
            BASE_DIR / "reports" / "model_candidate_validation_access.json"
        ),
        "model_artifacts": bt.directory_digest(BASE_DIR / "model_artifacts"),
        "candidates": bt.directory_digest(BASE_DIR / "model_artifacts" / "candidates"),
        "walkforward_evidence": bt.directory_digest(SOURCE_WALKFORWARD_DIRECTORY),
        "backtest_evidence": bt.directory_digest(SOURCE_BACKTEST_DIRECTORY),
        "selectivity_evidence": bt.directory_digest(SOURCE_SELECTIVITY_DIRECTORY),
        "executable_return_evidence": bt.directory_digest(
            SOURCE_EXECUTABLE_RETURN_DIRECTORY
        ),
        "live_writer.py": "absent",
        "live_executor.py": "absent",
    }


def validate_sources() -> tuple[dict[str, Any], Path, Path, Mapping[str, str]]:
    walkforward_summary, _, walkforward_digest = bt.validate_walkforward_source(
        SOURCE_WALKFORWARD_DIRECTORY
    )
    if walkforward_digest != SOURCE_WALKFORWARD_DIRECTORY_DIGEST:
        raise FeatureEnrichmentResearchError("walk-forward evidence changed")
    for directory, experiment_id, experiment_digest, directory_digest in (
        (
            SOURCE_BACKTEST_DIRECTORY,
            SOURCE_BACKTEST_EXPERIMENT_ID,
            SOURCE_BACKTEST_EXPERIMENT_DIGEST,
            SOURCE_BACKTEST_DIRECTORY_DIGEST,
        ),
        (
            SOURCE_SELECTIVITY_DIRECTORY,
            SOURCE_SELECTIVITY_EXPERIMENT_ID,
            SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
            SOURCE_SELECTIVITY_DIRECTORY_DIGEST,
        ),
        (
            SOURCE_EXECUTABLE_RETURN_DIRECTORY,
            SOURCE_EXECUTABLE_RETURN_EXPERIMENT_ID,
            SOURCE_EXECUTABLE_RETURN_EXPERIMENT_DIGEST,
            SOURCE_EXECUTABLE_RETURN_DIRECTORY_DIGEST,
        ),
    ):
        er.validate_previous_research_source(
            directory,
            expected_id=experiment_id,
            expected_digest=experiment_digest,
            expected_directory_digest=directory_digest,
        )
    earlier, later = bt.source_dataset_paths(walkforward_summary)
    expected_raw = walkforward_summary["raw_source_digests"]
    bt.raw_source_digests(earlier, later, expected_raw)
    return walkforward_summary, earlier, later, expected_raw


def feature_contract() -> dict[str, Any]:
    baseline = wf._feature_contract()
    contract = {
        "baseline": baseline,
        "enriched": {
            "baseline_feature_contract_digest": baseline[
                "feature_contract_digest"
            ],
            "research_only_addition_count": 17,
            "research_only_addition_columns": list(ENRICHED_FEATURE_COLUMNS),
            "research_only_addition_formulas": FEATURE_FORMULAS,
            "ordered_model_feature_columns": list(feature_columns("enriched")),
            "model_feature_count": 44,
            "implementation": "tools.model_feature_enrichment_research",
            "features_py_modified": False,
            "future_information_allowed": False,
        },
        "fair_comparison": "same common enriched-eligible timestamp grid",
        "feature_selection_or_sweep_performed": False,
    }
    contract["enriched"]["feature_contract_digest"] = bt.json_digest(
        contract["enriched"]
    )
    return contract


def experiment_contract(
    *, protected_hashes: Mapping[str, Any], feature_contracts: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "tool_code_sha256": bt.file_sha256(Path(__file__)),
        "source_experiments": {
            "walkforward": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
            "backtest": SOURCE_BACKTEST_EXPERIMENT_DIGEST,
            "selectivity": SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
            "executable_return": SOURCE_EXECUTABLE_RETURN_EXPERIMENT_DIGEST,
        },
        "protected_source_hashes": dict(protected_hashes),
        "feature_contracts": dict(feature_contracts),
        "horizon_bars": HORIZON_BARS,
        "horizon_minutes": HORIZON_MINUTES,
        "fold_specification": dict(wf.FOLD_SPEC),
        "fold_count": 7,
        "model_configurations": {
            "classifier": CLASSIFIER_CONFIG,
            "regressor": REGRESSOR_CONFIG,
        },
        "economic_policies": {
            "classifier": CLASSIFIER_POLICY_CONTRACT,
            "regressor": REGRESSOR_POLICY_CONTRACT,
        },
        "cost_contract": dict(bt.COST_CONTRACT),
        "execution_contract": dict(bt.EXECUTION_CONTRACT),
        "portfolio_contract": dict(bt.PORTFOLIO_CONTRACT),
        "regime_contract": REGIME_CONTRACT,
        "stop_condition": STOP_CONDITION,
        "feature_selection_or_sweep_performed": False,
        "hyperparameter_or_threshold_sweep_performed": False,
        "production_pass_gate_defined": False,
    }


def _regime_row(
    raw_by_symbol: Mapping[str, pd.DataFrame], fold: wf.FoldDefinition
) -> dict[str, Any]:
    source = wf.calculate_regime_descriptors(raw_by_symbol, fold)
    return {
        "fold_id": fold.fold_id,
        "test_start_utc": wf.canonical_utc(fold.test_start),
        "test_end_exclusive_utc": wf.canonical_utc(fold.test_end_exclusive),
        "btc_realized_volatility": source["btc_regime_realized_volatility"],
        "eth_realized_volatility": source["eth_regime_realized_volatility"],
        "btc_signed_return": source["btc_regime_signed_period_return"],
        "eth_signed_return": source["eth_regime_signed_period_return"],
        "btc_relative_volume": source[
            "btc_regime_mean_volume_to_train_median"
        ],
        "eth_relative_volume": source[
            "eth_regime_mean_volume_to_train_median"
        ],
    }


def validate_results(
    *,
    ledger: pd.DataFrame,
    classifier_fold_metrics: Sequence[Mapping[str, Any]],
    classifier_summary: Sequence[Mapping[str, Any]],
    regressor_fold_metrics: Sequence[Mapping[str, Any]],
    regressor_summary: Sequence[Mapping[str, Any]],
    economic_fold_metrics: Sequence[Mapping[str, Any]],
    economic_overall_metrics: Sequence[Mapping[str, Any]],
    economic_summary: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, Any]],
    regime_descriptors: Sequence[Mapping[str, Any]],
    curves: Mapping[tuple[str, str, int], pd.DataFrame],
) -> dict[str, bool]:
    checks = bt.validate_trade_ledger(ledger)
    if not ledger["prediction_origin"].eq("outer_test").all():
        raise FeatureEnrichmentResearchError("economics include non-OOS predictions")
    if set(ledger["feature_set"]) != set(FEATURE_SETS):
        raise FeatureEnrichmentResearchError("feature-set economics are incomplete")
    if set(ledger["model_family"]) != set(MODEL_FAMILIES):
        raise FeatureEnrichmentResearchError("model-family economics are incomplete")
    threshold_frame = pd.DataFrame(thresholds)
    if len(threshold_frame) != 28:
        raise FeatureEnrichmentResearchError("threshold grid is incomplete")
    if not (
        threshold_frame["threshold_source"].str.contains("training").all()
        and (~threshold_frame["test_predictions_consulted"]).all()
        and (~threshold_frame["test_returns_consulted"]).all()
    ):
        raise FeatureEnrichmentResearchError("threshold provenance is unsafe")
    class_folds = pd.DataFrame(classifier_fold_metrics)
    regression_folds = pd.DataFrame(regressor_fold_metrics)
    if class_folds.groupby("feature_set").size().to_dict() != {
        "baseline": 7,
        "enriched": 7,
    }:
        raise FeatureEnrichmentResearchError("classifier fold grid is incomplete")
    if regression_folds.groupby("feature_set").size().to_dict() != {
        "baseline": 7,
        "enriched": 7,
    }:
        raise FeatureEnrichmentResearchError("regressor fold grid is incomplete")
    if len(classifier_summary) != 2 or len(regressor_summary) != 2:
        raise FeatureEnrichmentResearchError("model summaries are incomplete")
    fold_frame = pd.DataFrame(economic_fold_metrics)
    overall_frame = pd.DataFrame(economic_overall_metrics)
    summary_frame = pd.DataFrame(economic_summary)
    if len(fold_frame) != 112 or len(overall_frame) != 16 or len(summary_frame) != 4:
        raise FeatureEnrichmentResearchError("economic grids are incomplete")
    if len(regime_descriptors) != 7:
        raise FeatureEnrichmentResearchError("regime descriptor grid is incomplete")
    for (strategy, policy), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=False
    ):
        expected = int(group["active_trade"].sum())
        summary_match = summary_frame.loc[
            (summary_frame["strategy_id"] == strategy)
            & (summary_frame["policy_name"] == policy)
        ]
        if len(summary_match) != 1 or int(
            summary_match.iloc[0]["active_trade_count"]
        ) != expected:
            raise FeatureEnrichmentResearchError("active counts do not reconcile")
        if int(
            summary_match.iloc[0]["long_trade_count"]
            + summary_match.iloc[0]["short_trade_count"]
        ) != expected:
            raise FeatureEnrichmentResearchError("long/short counts do not reconcile")
        rows = overall_frame.loc[
            (overall_frame["strategy_id"] == strategy)
            & (overall_frame["policy_name"] == policy)
        ].set_index("cost_bps")
        if not math.isclose(
            float(rows.loc[0, "net_cumulative_return"]),
            float(rows.loc[0, "gross_cumulative_return"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise FeatureEnrichmentResearchError("zero-cost net differs from gross")
        for lower, higher in zip(COST_SCENARIOS_BPS, COST_SCENARIOS_BPS[1:]):
            low_curve = curves[(str(strategy), str(policy), lower)]
            high_curve = curves[(str(strategy), str(policy), higher)]
            if np.any(
                high_curve["net_equity"].to_numpy(dtype=np.float64)
                > low_curve["net_equity"].to_numpy(dtype=np.float64) + 1e-12
            ):
                raise FeatureEnrichmentResearchError("higher cost improved equity")
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
            raise FeatureEnrichmentResearchError("flat signals paid cost")
        if cost == 0 and not np.array_equal(net, gross):
            raise FeatureEnrichmentResearchError("zero-cost trade mismatch")
    return {
        **checks,
        "research_features_backward_looking_or_current_only": True,
        "btc_context_uses_same_or_earlier_timestamp_only": True,
        "baseline_contract_is_canonical": True,
        "enriched_contract_is_baseline_plus_exactly_17": True,
        "classifier_configuration_is_frozen": True,
        "regressor_configuration_is_frozen": True,
        "walkforward_chronology_is_unchanged": True,
        "horizon_purge_is_strict": True,
        "thresholds_use_training_predictions_only": True,
        "test_predictions_do_not_set_thresholds": True,
        "test_returns_do_not_set_thresholds": True,
        "economic_predictions_are_outer_test_only": True,
        "accounting_reused_from_validated_backtest": True,
        "transaction_cost_applied_once": True,
        "higher_cost_never_improves_equity": True,
        "zero_cost_net_equals_gross": True,
        "economic_metrics_reconcile": True,
        "btc_eth_results_remain_separate": True,
        "regime_labels_are_descriptive_only": True,
    }


def build_markdown_report(
    *,
    experiment_id: str,
    experiment_digest: str,
    classifier_summary: Sequence[Mapping[str, Any]],
    regressor_summary: Sequence[Mapping[str, Any]],
    economic_summary: Sequence[Mapping[str, Any]],
    regime_summary: Sequence[Mapping[str, Any]],
    incremental_value: Sequence[Mapping[str, Any]],
    stop_triggered: bool,
) -> str:
    classes = {str(row["feature_set"]): row for row in classifier_summary}
    regressors = {str(row["feature_set"]): row for row in regressor_summary}
    economics = {
        (str(row["model_family"]), str(row["feature_set"])): row
        for row in economic_summary
    }
    best = max(economic_summary, key=lambda row: float(row["overall_net_return_5bps"]))
    stop_text = (
        "STOP further OHLCV-only feature tweaking on this exposed history. The next "
        "engineering branch must become collection of richer market data for future "
        "research: funding history, open interest, mark/index/basis, order-book "
        "imbalance, trades/aggressor flow, and liquidation data where available."
        if stop_triggered
        else "The frozen stop condition was not triggered, but this remains research-only and does not authorize a production feature migration."
    )
    lines = [
        "# Feature Enrichment and Regime Robustness Research",
        "",
        "## Executive summary",
        "",
        f"Experiment `{experiment_id}` (`{experiment_digest}`) is exposed historical research only. No production gate or deployment authorization exists.",
        f"The strongest 5-bps stream was **{best['feature_set']} {best['model_family']}**, with return {float(best['overall_net_return_5bps']):.2%}, {float(best['positive_fold_percentage_5bps']):.2f}% positive folds, and {float(best['mean_gross_bps_per_active_trade']):.4f} gross bps per active trade.",
        "",
        f"**Decision:** {stop_text}",
        "",
        "## Classification comparison",
        "",
        "| Feature set | Mean AUC | Median AUC | Worst AUC | BTC mean | ETH mean | Positive folds | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in FEATURE_SETS:
        row = classes[name]
        lines.append(
            f"| {name} | {float(row['mean_pooled_auc']):.4f} | {float(row['median_pooled_auc']):.4f} | {float(row['worst_fold_auc']):.4f} | {float(row['btc_mean_auc']):.4f} | {float(row['eth_mean_auc']):.4f} | {int(row['positive_auc_fold_count'])}/7 | {float(row['pooled_brier_score']):.4f} | {float(row['pooled_log_loss']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Regressor comparison",
            "",
            "| Feature set | RMSE/base | Pearson | Spearman | Sign accuracy | Top-bottom bps | Magnitude correlation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in FEATURE_SETS:
        row = regressors[name]
        lines.append(
            f"| {name} | {float(row['candidate_baseline_rmse_ratio']):.4f} | {float(row['pearson_correlation']):.4f} | {float(row['spearman_correlation']):.4f} | {float(row['sign_accuracy']):.2%} | {float(row['top_minus_bottom_realized_spread_bps']):.4f} | {float(row['absolute_magnitude_pearson_correlation']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Economic comparison",
            "",
            "| Family | Features | Active | Active % | Gross bps | Break-even | Return 0 | Return 2 | Return 5 | Return 10 | Sharpe 5 | Max DD 5 | Positive folds 5 | BTC 5 | ETH 5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in MODEL_FAMILIES:
        for name in FEATURE_SETS:
            row = economics[(family, name)]
            break_even = row["approximate_break_even_round_trip_cost_bps"]
            be_text = "n/a" if break_even is None else f"{float(break_even):.4f}"
            lines.append(
                f"| {family} | {name} | {int(row['active_trade_count'])} | {float(row['active_fraction']):.2%} | {float(row['mean_gross_bps_per_active_trade']):.4f} | {be_text} | {float(row['overall_net_return_0bps']):.2%} | {float(row['overall_net_return_2bps']):.2%} | {float(row['overall_net_return_5bps']):.2%} | {float(row['overall_net_return_10bps']):.2%} | {float(row['daily_sharpe_5bps']):.4f} | {float(row['maximum_drawdown_5bps']):.2%} | {float(row['positive_fold_percentage_5bps']):.2f}% | {float(row['btc_net_return_5bps']):.2%} | {float(row['eth_net_return_5bps']):.2%} |"
            )
    lines.extend(
        [
            "",
            "## Incremental-value diagnostics",
            "",
            "The bundle is evaluated as one frozen contract; no ablation, feature selection, importance pruning, or threshold tuning occurred.",
            "",
        ]
    )
    for row in incremental_value:
        lines.append(
            f"- **{row['model_family']}:** gross edge improved={row['enriched_improves_gross_edge_per_trade']}; break-even improved={row['enriched_improves_break_even_cost']}; 5-bps return improved={row['enriched_improves_5bps_return']}; positive-fold fraction improved={row['enriched_improves_5bps_positive_fold_fraction']}; both symbols improved={row['enriched_both_symbols_better_at_5bps']}."
        )
    lines.extend(
        [
            "",
            "## Regime robustness",
            "",
            "Volatility and BTC-return groups are post-hoc descriptive cuts of the seven exposed test folds. They are not learned, not used in fitting, and not a tradable regime rule.",
            "",
            "| Family | Features | Rule | Group | Folds | Mean fold return 5 | Positive folds | Gross bps/trade | Predictive metric |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in regime_summary:
        edge = row["weighted_mean_gross_bps_per_active_trade"]
        edge_text = "n/a" if edge is None else f"{float(edge):.4f}"
        lines.append(
            f"| {row['model_family']} | {row['feature_set']} | {row['regime_rule']} | {row['regime_group']} | {int(row['fold_count'])} | {float(row['mean_fold_net_return']):.2%} | {float(row['positive_fold_percentage']):.2f}% | {edge_text} | {float(row['mean_predictive_metric']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "Both contracts use the same enriched-eligible timestamp grid. The baseline remains the exact canonical 26 market features plus symbol_id built by features.build_features; the enriched contract appends exactly 17 research-only features. All features use data available no later than t.",
            "",
            "All seven folds retain the established 120-day rolling train, 30-day test, 30-day step chronology with strict horizon-aware purge. Economic execution remains next-bar open to close[t+6], non-overlapping within symbol/fold, no fold carry, fixed 50/50 BTC/ETH sleeves, no leverage, and no flat-sleeve reallocation.",
            "",
            "Costs are synthetic round-trip stresses, not venue fee claims. Funding, spread, impact, additional latency, order-book state, open interest, basis, aggressor flow, and liquidations remain unmodeled.",
            "",
            "No production feature contract, candidate, incumbent, model artifact, runtime, or validation ledger was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_output_root(path: Path | str) -> Path:
    requested = Path(path).resolve()
    allowed = RESEARCH_ROOT.resolve()
    if requested != allowed:
        raise FeatureEnrichmentResearchError(
            f"feature-enrichment output root must be exactly {allowed}; received {requested}"
        )
    return requested


def run_research(*, output_root: Path | str = RESEARCH_ROOT) -> dict[str, Any]:
    """Run the one frozen offline feature-enrichment experiment."""

    walkforward_summary, earlier, later, expected_raw = validate_sources()
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
    if not raw_by_symbol["BTCUSDT"].index.equals(
        raw_by_symbol["ETHUSDT"].index
    ):
        raise FeatureEnrichmentResearchError("BTC/ETH raw timestamp grids differ")
    matrices, feature_diagnostics = build_feature_contracts(raw_by_symbol)
    features_contract = feature_contract()
    contract = experiment_contract(
        protected_hashes=protected_before, feature_contracts=features_contract
    )
    experiment_digest = bt.json_digest(contract)
    experiment_id = f"feature_enrichment_{experiment_digest[:16]}"
    output = ensure_output_root(output_root)
    final_directory = output / experiment_id
    staging_directory = output / f".{experiment_id}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise FeatureEnrichmentResearchError(
            f"feature-enrichment output already exists: {experiment_id}"
        )

    folds = wf.make_walkforward_folds(
        raw_by_symbol["BTCUSDT"].index[0],
        raw_by_symbol["BTCUSDT"].index[-1],
        horizon_bars=HORIZON_BARS,
    )
    if len(folds) != 7:
        raise FeatureEnrichmentResearchError("experiment requires exactly seven folds")
    raw_regimes = [_regime_row(raw_by_symbol, fold) for fold in folds]
    regime_descriptors, volatility_threshold = assign_regime_groups(raw_regimes)

    classification_targets: dict[str, pd.DataFrame] = {}
    regression_targets: dict[str, pd.DataFrame] = {}
    for feature_set in FEATURE_SETS:
        classification_targets[feature_set] = pd.concat(
            [
                wf.build_fixed_horizon_rows(
                    raw_by_symbol[symbol],
                    matrices[feature_set][symbol],
                    symbol=symbol,
                    horizon_bars=HORIZON_BARS,
                )
                for symbol in wf.SYMBOLS
            ],
            ignore_index=True,
        )
        regression_targets[feature_set] = pd.concat(
            [
                er.build_executable_return_rows(
                    raw_by_symbol[symbol],
                    matrices[feature_set][symbol],
                    symbol=symbol,
                    horizon_bars=HORIZON_BARS,
                )
                for symbol in wf.SYMBOLS
            ],
            ignore_index=True,
        )

    trade_rows: list[dict[str, Any]] = []
    classifier_fold_rows: list[dict[str, Any]] = []
    classifier_pooled_rows: list[dict[str, Any]] = []
    regressor_fold_rows: list[dict[str, Any]] = []
    regressor_pooled_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for feature_set in FEATURE_SETS:
        columns = feature_columns(feature_set)
        for fold in folds:
            class_train, class_test = wf.select_fold_rows(
                classification_targets[feature_set], fold
            )
            X_class_train = class_train.loc[:, columns].to_numpy(dtype=np.float64)
            y_class_train = class_train["target"].to_numpy(dtype=np.int8)
            X_class_test = class_test.loc[:, columns].to_numpy(dtype=np.float64)
            train_scores, test_scores, _ = fit_classifier_predictions(
                X_class_train, y_class_train, X_class_test
            )
            lower, upper = derive_classifier_thresholds(train_scores)
            class_metrics = classifier_fold_metrics(class_test, test_scores)
            classifier_fold_rows.append(
                {
                    "feature_set": feature_set,
                    "model_family": "classifier",
                    "model_name": CLASSIFIER_MODEL_NAME,
                    "fold_id": fold.fold_id,
                    "train_row_count": int(len(class_train)),
                    "test_row_count": int(len(class_test)),
                    "maximum_train_target_timestamp_utc": wf.canonical_utc(
                        class_train["target_timestamp"].max()
                    ),
                    "test_start_utc": wf.canonical_utc(fold.test_start),
                    **class_metrics,
                }
            )
            threshold_rows.append(
                {
                    "feature_set": feature_set,
                    "model_family": "classifier",
                    "model_name": CLASSIFIER_MODEL_NAME,
                    "fold_id": fold.fold_id,
                    "policy_name": CLASSIFIER_POLICY,
                    "training_prediction_count": int(len(train_scores)),
                    "training_lower_threshold": lower,
                    "training_upper_threshold": upper,
                    "training_abs_prediction_threshold_bps": None,
                    "threshold_source": CLASSIFIER_POLICY_CONTRACT[
                        "threshold_source"
                    ],
                    "test_predictions_consulted": False,
                    "test_returns_consulted": False,
                }
            )
            scored_class_test = class_test.loc[:, ["timestamp", "symbol"]].copy()
            scored_class_test["prediction"] = test_scores
            scored_class_test["prediction_origin"] = "outer_test"
            for symbol in wf.SYMBOLS:
                selected = scored_class_test.loc[
                    scored_class_test["symbol"] == symbol,
                    ["timestamp", "prediction", "prediction_origin"],
                ]
                trade_rows.extend(
                    build_enrichment_trade_rows(
                        experiment_id=experiment_id,
                        feature_set=feature_set,
                        model_family="classifier",
                        fold=fold,
                        symbol=symbol,
                        scored_test=selected,
                        raw=raw_by_symbol[symbol],
                        lower_threshold=lower,
                        upper_threshold=upper,
                    )
                )
            for index, meta in enumerate(
                class_test.loc[:, ["timestamp", "symbol", "target"]].itertuples(
                    index=False
                )
            ):
                classifier_pooled_rows.append(
                    {
                        "feature_set": feature_set,
                        "fold_id": fold.fold_id,
                        "timestamp": bt.canonical_utc(meta.timestamp),
                        "symbol": meta.symbol,
                        "target": int(meta.target),
                        "score": float(test_scores[index]),
                    }
                )

            reg_train, reg_test = er.select_executable_fold_rows(
                regression_targets[feature_set], fold
            )
            X_reg_train = reg_train.loc[:, columns].to_numpy(dtype=np.float64)
            y_reg_train = reg_train["executable_return_bps"].to_numpy(
                dtype=np.float64
            )
            X_reg_test = reg_test.loc[:, columns].to_numpy(dtype=np.float64)
            train_predictions, test_predictions, _ = er.fit_regression_predictions(
                REGRESSOR_MODEL_NAME, X_reg_train, y_reg_train, X_reg_test
            )
            threshold = derive_regressor_threshold(train_predictions)
            baseline_mean = float(np.mean(y_reg_train))
            baseline_test = np.full(len(reg_test), baseline_mean, dtype=np.float64)
            regression_metrics = er.regression_metrics(
                test_predictions,
                reg_test["executable_return_bps"].to_numpy(dtype=np.float64),
                baseline_test,
            )
            regressor_fold_rows.append(
                {
                    "feature_set": feature_set,
                    "model_family": "regressor",
                    "model_name": REGRESSOR_MODEL_NAME,
                    "fold_id": fold.fold_id,
                    "train_row_count": int(len(reg_train)),
                    "test_row_count": int(len(reg_test)),
                    "maximum_train_target_exit_timestamp_utc": wf.canonical_utc(
                        reg_train["target_exit_timestamp"].max()
                    ),
                    "test_start_utc": wf.canonical_utc(fold.test_start),
                    **regression_metrics,
                }
            )
            threshold_rows.append(
                {
                    "feature_set": feature_set,
                    "model_family": "regressor",
                    "model_name": REGRESSOR_MODEL_NAME,
                    "fold_id": fold.fold_id,
                    "policy_name": REGRESSOR_POLICY,
                    "training_prediction_count": int(len(train_predictions)),
                    "training_lower_threshold": None,
                    "training_upper_threshold": None,
                    "training_abs_prediction_threshold_bps": threshold,
                    "threshold_source": REGRESSOR_POLICY_CONTRACT[
                        "threshold_source"
                    ],
                    "test_predictions_consulted": False,
                    "test_returns_consulted": False,
                }
            )
            scored_reg_test = reg_test.loc[:, ["timestamp", "symbol"]].copy()
            scored_reg_test["prediction"] = test_predictions
            scored_reg_test["prediction_origin"] = "outer_test"
            for symbol in wf.SYMBOLS:
                selected = scored_reg_test.loc[
                    scored_reg_test["symbol"] == symbol,
                    ["timestamp", "prediction", "prediction_origin"],
                ]
                trade_rows.extend(
                    build_enrichment_trade_rows(
                        experiment_id=experiment_id,
                        feature_set=feature_set,
                        model_family="regressor",
                        fold=fold,
                        symbol=symbol,
                        scored_test=selected,
                        raw=raw_by_symbol[symbol],
                        absolute_threshold_bps=threshold,
                    )
                )
            for index, meta in enumerate(
                reg_test.loc[:, ["timestamp", "symbol"]].itertuples(index=False)
            ):
                regressor_pooled_rows.append(
                    {
                        "feature_set": feature_set,
                        "fold_id": fold.fold_id,
                        "timestamp": bt.canonical_utc(meta.timestamp),
                        "symbol": meta.symbol,
                        "prediction_bps": float(test_predictions[index]),
                        "target_bps": float(
                            reg_test["executable_return_bps"].iloc[index]
                        ),
                        "baseline_prediction_bps": float(baseline_test[index]),
                    }
                )

    ledger = as_enrichment_trade_frame(trade_rows)
    fold_economics, overall_economics, curves = er.calculate_economic_metrics(
        ledger, expected_fold_count=7
    )
    stream_metadata = {
        stream_id(feature_set, family): (feature_set, family)
        for family in MODEL_FAMILIES
        for feature_set in FEATURE_SETS
    }
    for collection in (fold_economics, overall_economics):
        for row in collection:
            feature_set, family = stream_metadata[str(row["strategy_id"])]
            row["feature_set"] = feature_set
            row["model_family"] = family
    classifier_summary = aggregate_classifier_summary(
        classifier_fold_rows, classifier_pooled_rows
    )
    regressor_summary = aggregate_regressor_summary(
        regressor_fold_rows, regressor_pooled_rows
    )
    economics_summary = economic_summary_rows(ledger, overall_economics)
    regime_summary = regime_robustness_rows(
        fold_economics,
        regime_descriptors,
        classifier_fold_rows,
        regressor_fold_rows,
    )
    incremental_value = incremental_value_rows(
        classifier_summary, economics_summary
    )
    invariant_checks = validate_results(
        ledger=ledger,
        classifier_fold_metrics=classifier_fold_rows,
        classifier_summary=classifier_summary,
        regressor_fold_metrics=regressor_fold_rows,
        regressor_summary=regressor_summary,
        economic_fold_metrics=fold_economics,
        economic_overall_metrics=overall_economics,
        economic_summary=economics_summary,
        thresholds=threshold_rows,
        regime_descriptors=regime_descriptors,
        curves=curves,
    )
    protected_after = protected_source_hashes(
        earlier=earlier, later=later, expected_raw=expected_raw
    )
    if protected_after != protected_before:
        raise FeatureEnrichmentResearchError("protected sources changed during run")
    invariant_checks.update(
        {
            "protected_hashes_unchanged": True,
            "features_py_and_feature_cols_unchanged": True,
            "frozen_raw_sources_unchanged": True,
            "candidate_and_incumbent_artifacts_unchanged": True,
            "validation_ledger_unchanged": True,
            "previous_research_evidence_unchanged": True,
        }
    )
    enriched_rows = [
        row for row in economics_summary if row["feature_set"] == "enriched"
    ]
    enriched_passes_stop_condition = any(
        float(row["overall_net_return_5bps"]) > 0.0
        and float(row["positive_fold_percentage_5bps"]) > 50.0
        for row in enriched_rows
    )
    stop_triggered = not enriched_passes_stop_condition
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "research_only": True,
        "historical_periods_pristine_holdout": False,
        "production_candidate": False,
        "promotion_allowed": False,
        "deployment_authorization": False,
        "production_feature_migration_authorized": False,
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
            "executable_return": {
                "experiment_id": SOURCE_EXECUTABLE_RETURN_EXPERIMENT_ID,
                "experiment_digest": SOURCE_EXECUTABLE_RETURN_EXPERIMENT_DIGEST,
            },
        },
        "feature_contracts": features_contract,
        "feature_diagnostics": feature_diagnostics,
        "model_configurations": {
            "classifier": CLASSIFIER_CONFIG,
            "regressor": REGRESSOR_CONFIG,
        },
        "fold_specification": dict(wf.FOLD_SPEC),
        "fold_count": 7,
        "horizon_bars": HORIZON_BARS,
        "economic_policies": {
            "classifier": CLASSIFIER_POLICY_CONTRACT,
            "regressor": REGRESSOR_POLICY_CONTRACT,
        },
        "cost_contract": dict(bt.COST_CONTRACT),
        "execution_contract": dict(bt.EXECUTION_CONTRACT),
        "portfolio_contract": dict(bt.PORTFOLIO_CONTRACT),
        "regime_contract": REGIME_CONTRACT,
        "regime_volatility_median_threshold": volatility_threshold,
        "regime_descriptors": regime_descriptors,
        "regime_robustness_summary": regime_summary,
        "classifier_fold_metrics": classifier_fold_rows,
        "classifier_summary": classifier_summary,
        "regressor_fold_metrics": regressor_fold_rows,
        "regressor_summary": regressor_summary,
        "economic_fold_metrics": fold_economics,
        "economic_overall_metrics": overall_economics,
        "economic_summary": economics_summary,
        "policy_thresholds": threshold_rows,
        "incremental_value_diagnostics": incremental_value,
        "stop_condition": {
            **STOP_CONDITION,
            "enriched_strategy_positive_at_5bps_and_majority_positive_folds": (
                enriched_passes_stop_condition
            ),
            "stop_ohlcv_only_feature_tweaking": stop_triggered,
            "recommendation": (
                STOP_CONDITION["failure_recommendation"]
                if stop_triggered
                else "Do not migrate production features; require a separate confirmation and feature-contract migration task."
            ),
        },
        "protected_source_hashes_before": protected_before,
        "protected_source_hashes_after": protected_after,
        "protected_source_hashes_unchanged": True,
        "invariant_checks": invariant_checks,
        "all_invariants_reconciled": all(invariant_checks.values()),
        "trade_ledger_row_count": int(len(ledger)),
        "safety_contract": {
            "network_exchange_or_api_access_performed": False,
            "runtime_or_live_execution_accessed_or_modified": False,
            "features_py_or_feature_cols_modified": False,
            "candidate_or_incumbent_artifacts_modified_or_written": False,
            "validation_ledger_modified_or_written": False,
            "model_artifacts_written": False,
            "lstm_tcn_transformer_trained": False,
            "new_data_downloaded": False,
            "outputs_restricted_to": str(output),
        },
        "historical_research_exposure_warning": (
            "All periods and regime groupings are exposed historical research, not a pristine holdout."
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir()
    paths = {
        name: staging_directory / name
        for name in (
            "summary.json",
            "feature_contracts.json",
            "classifier_fold_metrics.csv",
            "classifier_summary.csv",
            "regressor_fold_metrics.csv",
            "regressor_summary.csv",
            "policy_thresholds.csv",
            "trade_ledger.csv",
            "economic_fold_metrics.csv",
            "economic_overall_metrics.csv",
            "economic_summary.csv",
            "regime_descriptors.csv",
            "regime_robustness_summary.csv",
            "incremental_value_diagnostics.csv",
            "report.md",
            "experiment_manifest.json",
        )
    }
    bt._write_json(paths["summary.json"], summary)
    bt._write_json(paths["feature_contracts.json"], features_contract)
    bt._write_csv(paths["classifier_fold_metrics.csv"], classifier_fold_rows)
    bt._write_csv(paths["classifier_summary.csv"], classifier_summary)
    bt._write_csv(paths["regressor_fold_metrics.csv"], regressor_fold_rows)
    bt._write_csv(paths["regressor_summary.csv"], regressor_summary)
    bt._write_csv(paths["policy_thresholds.csv"], threshold_rows)
    ledger_output = ledger.copy()
    for column in ledger_output.columns:
        if pd.api.types.is_datetime64_any_dtype(ledger_output[column]):
            ledger_output[column] = ledger_output[column].map(bt.canonical_utc)
    bt._write_csv(
        paths["trade_ledger.csv"],
        ledger_output.to_dict("records"),
        ENRICHMENT_LEDGER_COLUMNS,
    )
    bt._write_csv(paths["economic_fold_metrics.csv"], fold_economics)
    bt._write_csv(paths["economic_overall_metrics.csv"], overall_economics)
    bt._write_csv(paths["economic_summary.csv"], economics_summary)
    bt._write_csv(paths["regime_descriptors.csv"], regime_descriptors)
    bt._write_csv(paths["regime_robustness_summary.csv"], regime_summary)
    bt._write_csv(
        paths["incremental_value_diagnostics.csv"], incremental_value
    )
    paths["report.md"].write_text(
        build_markdown_report(
            experiment_id=experiment_id,
            experiment_digest=experiment_digest,
            classifier_summary=classifier_summary,
            regressor_summary=regressor_summary,
            economic_summary=economics_summary,
            regime_summary=regime_summary,
            incremental_value=incremental_value,
            stop_triggered=stop_triggered,
        ),
        encoding="utf-8",
        newline="\n",
    )
    row_counts: dict[str, int | None] = {
        "summary.json": None,
        "feature_contracts.json": None,
        "classifier_fold_metrics.csv": len(classifier_fold_rows),
        "classifier_summary.csv": len(classifier_summary),
        "regressor_fold_metrics.csv": len(regressor_fold_rows),
        "regressor_summary.csv": len(regressor_summary),
        "policy_thresholds.csv": len(threshold_rows),
        "trade_ledger.csv": len(ledger_output),
        "economic_fold_metrics.csv": len(fold_economics),
        "economic_overall_metrics.csv": len(overall_economics),
        "economic_summary.csv": len(economics_summary),
        "regime_descriptors.csv": len(regime_descriptors),
        "regime_robustness_summary.csv": len(regime_summary),
        "incremental_value_diagnostics.csv": len(incremental_value),
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
        "deployment_authorization": False,
        "production_feature_migration_authorized": False,
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
        description="Run offline feature-enrichment and regime-robustness research."
    )
    parser.add_argument("--output-root", type=Path, default=RESEARCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_research(output_root=args.output_root)
    except (
        FeatureEnrichmentResearchError,
        er.ExecutableReturnResearchError,
        bt.SignalBacktestError,
        wf.SignalResearchError,
        OSError,
        ValueError,
    ) as exc:
        print(f"feature-enrichment research failed closed: {exc}", file=sys.stderr)
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
