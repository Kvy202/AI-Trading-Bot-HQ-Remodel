from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tools import model_executable_return_research as research
from tools import model_signal_backtest as bt
from tools import model_signal_walkforward as wf


def _raw(periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    position = np.arange(periods, dtype=np.float64)
    open_ = 100.0 + position
    return pd.DataFrame(
        {
            "open": open_,
            "high": open_ + 1.0,
            "low": open_ - 1.0,
            "close": open_ + 0.5,
            "volume": 1000.0 + position,
        },
        index=index,
    )


def _features(raw: pd.DataFrame, start: int = 5) -> pd.DataFrame:
    index = raw.index[start:]
    return pd.DataFrame(
        {"feature_fixture": np.arange(len(index), dtype=np.float64)}, index=index
    )


def _fold(horizon: int = 6, periods: int = 40) -> wf.FoldDefinition:
    test_start = pd.Timestamp("2026-01-03T00:00:00Z")
    return wf.FoldDefinition(
        fold_id="fold_00",
        horizon_bars=horizon,
        training_window_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        training_window_end_exclusive=test_start,
        fit_train_end_exclusive=test_start - horizon * wf.BAR_INTERVAL,
        purge_start=test_start - horizon * wf.BAR_INTERVAL,
        purge_end_exclusive=test_start,
        test_start=test_start,
        test_end_exclusive=test_start + periods * wf.BAR_INTERVAL,
    )


def _execution_raw(fold: wf.FoldDefinition) -> pd.DataFrame:
    start = fold.test_start - wf.BAR_INTERVAL
    end = fold.test_end_exclusive + wf.BAR_INTERVAL
    index = pd.date_range(start, end, freq="5min", inclusive="left", tz="UTC")
    position = np.arange(len(index), dtype=np.float64)
    open_ = 100.0 + position
    return pd.DataFrame(
        {
            "open": open_,
            "high": open_ + 1.0,
            "low": open_ - 1.0,
            "close": open_ + 0.5,
            "volume": 1000.0 + position,
        },
        index=index,
    )


def _scores(fold: wf.FoldDefinition, prediction: float) -> pd.DataFrame:
    times = pd.date_range(
        fold.test_start,
        fold.test_end_exclusive,
        freq="5min",
        inclusive="left",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "timestamp": times,
            "prediction_bps": prediction,
            "prediction_origin": "outer_test",
        }
    )


def _trade_rows(
    symbol: str,
    *,
    prediction: float = 10.0,
    policy: str = "predicted_sign",
    threshold: float | None = None,
) -> list[dict]:
    fold = _fold()
    return research.build_return_trade_rows(
        experiment_id="return_fixture",
        model_name="ridge_regression",
        horizon_bars=fold.horizon_bars,
        policy_name=policy,
        fold=fold,
        symbol=symbol,
        scored_test=_scores(fold, prediction),
        raw=_execution_raw(fold),
        training_abs_threshold_bps=threshold,
    )


def _paired_ledger(
    *,
    btc_prediction: float = 10.0,
    eth_prediction: float = -10.0,
    policy: str = "predicted_sign",
    threshold: float | None = None,
) -> pd.DataFrame:
    rows = _trade_rows(
        "BTCUSDT", prediction=btc_prediction, policy=policy, threshold=threshold
    )
    rows += _trade_rows(
        "ETHUSDT", prediction=eth_prediction, policy=policy, threshold=threshold
    )
    return research.as_return_trade_frame(rows)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def test_executable_target_uses_next_bar_open_and_horizon_close():
    raw = _raw()
    features = _features(raw)
    rows = research.build_executable_return_rows(
        raw, features, symbol="BTCUSDT", horizon_bars=6
    )
    first = rows.iloc[0]
    signal = pd.Timestamp(first["timestamp"])
    entry_time = signal + wf.BAR_INTERVAL
    exit_time = signal + 6 * wf.BAR_INTERVAL
    assert first["entry_timestamp"] == entry_time
    assert first["exit_bar_open_timestamp"] == exit_time
    assert first["entry_price"] == raw.at[entry_time, "open"]
    assert first["exit_price"] == raw.at[exit_time, "close"]
    assert first["executable_simple_return"] == pytest.approx(
        raw.at[exit_time, "close"] / raw.at[entry_time, "open"] - 1.0
    )
    assert first["executable_return_bps"] == pytest.approx(
        first["executable_simple_return"] * 10_000.0
    )


def test_same_bar_close_is_never_used_as_entry():
    raw = _raw()
    raw["close"] = raw["open"] + 500.0
    rows = research.build_executable_return_rows(
        raw, _features(raw), symbol="BTCUSDT", horizon_bars=6
    )
    first = rows.iloc[0]
    signal = pd.Timestamp(first["timestamp"])
    assert first["entry_price"] == raw.at[signal + wf.BAR_INTERVAL, "open"]
    assert first["entry_price"] != raw.at[signal, "close"]


def test_missing_next_bar_entry_fails_closed():
    raw = _raw()
    features = _features(raw)
    missing = features.index[0] + wf.BAR_INTERVAL
    raw = raw.drop(missing)
    with pytest.raises(research.ExecutableReturnResearchError, match="missing next-bar"):
        research.build_executable_return_rows(
            raw, features, symbol="BTCUSDT", horizon_bars=6
        )


def test_missing_horizon_exit_fails_closed():
    raw = _raw()
    features = _features(raw).iloc[[0]]
    missing = features.index[0] + 6 * wf.BAR_INTERVAL
    raw = raw.drop(missing)
    with pytest.raises(research.ExecutableReturnResearchError, match="missing horizon exit"):
        research.build_executable_return_rows(
            raw, features, symbol="BTCUSDT", horizon_bars=6
        )


def test_horizon_purge_requires_training_exit_strictly_before_test():
    fold = _fold(horizon=2)
    timestamps = pd.date_range(
        fold.training_window_start,
        fold.test_end_exclusive,
        freq="5min",
        inclusive="left",
        tz="UTC",
    )
    rows = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_exit_timestamp": timestamps + 3 * wf.BAR_INTERVAL,
            "symbol": np.where(np.arange(len(timestamps)) % 2, "BTCUSDT", "ETHUSDT"),
            "executable_return_bps": 1.0,
        }
    )
    train, test = research.select_executable_fold_rows(rows, fold)
    assert train["target_exit_timestamp"].max() < fold.test_start
    assert test["timestamp"].min() == fold.test_start
    leaked = rows.copy()
    leaked.loc[leaked.index[0], "target_exit_timestamp"] = fold.test_start
    # The leaky row is excluded rather than entering model fitting.
    train_again, _ = research.select_executable_fold_rows(leaked, fold)
    assert (train_again["target_exit_timestamp"] < fold.test_start).all()


def test_scaler_and_regression_fit_see_training_rows_only(monkeypatch):
    X_train = np.arange(60, dtype=np.float64).reshape(20, 3)
    y_train = np.linspace(-2.0, 2.0, 20)
    X_test = np.full((5, 3), 1e9)
    seen = {}
    original_scaler_fit = StandardScaler.fit
    original_ridge_fit = Ridge.fit

    def scaler_fit(self, X, y=None, **kwargs):
        seen["scaler"] = np.asarray(X).copy()
        return original_scaler_fit(self, X, y, **kwargs)

    def ridge_fit(self, X, y, **kwargs):
        seen["ridge_rows"] = len(X)
        seen["ridge_targets"] = np.asarray(y).copy()
        return original_ridge_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", scaler_fit)
    monkeypatch.setattr(Ridge, "fit", ridge_fit)
    train_pred, test_pred, _ = research.fit_regression_predictions(
        "ridge_regression", X_train, y_train, X_test
    )
    assert np.array_equal(seen["scaler"], X_train)
    assert seen["ridge_rows"] == len(X_train)
    assert np.array_equal(seen["ridge_targets"], y_train)
    assert len(train_pred) == len(X_train)
    assert len(test_pred) == len(X_test)


@pytest.mark.parametrize(
    ("policy", "quantile"),
    [("train_abs_q90", 0.90), ("train_abs_q95", 0.95)],
)
def test_absolute_thresholds_use_training_predictions_only(policy, quantile):
    train = np.asarray([-10.0, -5.0, -1.0, 2.0, 7.0, 20.0])
    observed = research.derive_absolute_prediction_threshold(train, policy)
    assert observed == pytest.approx(
        np.quantile(np.abs(train), quantile, method="linear")
    )


def test_test_predictions_and_returns_cannot_change_thresholds():
    signature = inspect.signature(research.derive_absolute_prediction_threshold)
    assert list(signature.parameters) == ["train_predictions_bps", "policy_name"]
    train = np.linspace(-20.0, 20.0, 101)
    first = research.derive_absolute_prediction_threshold(train, "train_abs_q95")
    test_predictions_a = np.asarray([-1000.0, 1000.0])
    test_predictions_b = np.asarray([-0.01, 0.01])
    test_returns_a = np.asarray([500.0, -500.0])
    test_returns_b = -test_returns_a
    assert not np.array_equal(test_predictions_a, test_predictions_b)
    assert not np.array_equal(test_returns_a, test_returns_b)
    assert first == research.derive_absolute_prediction_threshold(
        train, "train_abs_q95"
    )


def test_only_outer_test_predictions_feed_economic_results():
    fold = _fold()
    scores = _scores(fold, 10.0)
    scores["prediction_origin"] = "outer_train"
    with pytest.raises(research.ExecutableReturnResearchError, match="outer-test"):
        research.build_return_trade_rows(
            experiment_id="fixture",
            model_name="ridge_regression",
            horizon_bars=fold.horizon_bars,
            policy_name="predicted_sign",
            fold=fold,
            symbol="BTCUSDT",
            scored_test=scores,
            raw=_execution_raw(fold),
            training_abs_threshold_bps=None,
        )


def test_no_overlap_no_fold_carry_and_next_bar_execution():
    ledger = _paired_ledger()
    checks = bt.validate_trade_ledger(ledger)
    assert checks["next_bar_entry"]
    assert checks["no_same_symbol_overlap"]
    assert checks["no_fold_boundary_carry"]


def test_transaction_cost_once_and_higher_cost_cannot_improve_equity():
    assert bt.apply_round_trip_cost(
        0.01, active_trade=True, round_trip_cost_bps=5
    ) == pytest.approx(0.0095)
    ledger = _paired_ledger()
    curves = []
    for cost in research.COST_SCENARIOS_BPS:
        events = bt.build_portfolio_events(ledger, cost)
        if cost == 0:
            assert np.array_equal(
                events["portfolio_net_event_return"],
                events["portfolio_gross_event_return"],
            )
        curves.append(bt.enrich_equity_curve(events)["net_equity"].to_numpy())
    for lower, higher in zip(curves, curves[1:]):
        assert np.all(higher <= lower + 1e-12)


def test_btc_and_eth_remain_separate():
    ledger = _paired_ledger(btc_prediction=10.0, eth_prediction=-10.0)
    metrics = bt.stream_metrics(ledger, bt.build_portfolio_events(ledger, 0), 0)
    assert metrics["btc_net_cumulative_return"] != metrics["eth_net_cumulative_return"]


def test_regression_metrics_reconcile_exactly():
    prediction = np.asarray([-2.0, -1.0, 1.0, 2.0] * 5)
    target = prediction + np.asarray([1.0, -1.0, 0.5, -0.5] * 5)
    baseline = np.zeros(len(target))
    metrics = research.regression_metrics(prediction, target, baseline)
    error = prediction - target
    assert metrics["mae_bps"] == pytest.approx(np.mean(np.abs(error)))
    assert metrics["rmse_bps"] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert metrics["baseline_rmse_bps"] == pytest.approx(
        np.sqrt(np.mean(target**2))
    )
    assert metrics["candidate_baseline_rmse_ratio"] == pytest.approx(
        metrics["rmse_bps"] / metrics["baseline_rmse_bps"]
    )
    assert metrics["sign_accuracy"] == pytest.approx(
        np.mean(np.sign(prediction) == np.sign(target))
    )


def test_bucket_diagnostics_reconcile_and_are_monotonic_fixture():
    prediction = np.arange(1.0, 101.0)
    target = np.arange(1.0, 101.0)
    rows, monotonic = research.magnitude_bucket_diagnostics(prediction, target)
    assert len(rows) == 10
    assert sum(row["row_count"] for row in rows) == len(prediction)
    assert monotonic is True
    assert all(
        later["mean_absolute_realized_return_bps"]
        >= earlier["mean_absolute_realized_return_bps"]
        for earlier, later in zip(rows, rows[1:])
    )


def test_economic_active_counts_and_btc_eth_metrics_reconcile():
    ledger = _paired_ledger()
    fold_rows, overall_rows, curves = research.calculate_economic_metrics(
        ledger, expected_fold_count=1
    )
    # Fixture has one fold, so validate the lower-level fold/accounting results directly.
    assert len(fold_rows) == len(research.COST_SCENARIOS_BPS)
    assert all(row["active_trade_count"] == int(ledger["active_trade"].sum()) for row in fold_rows)
    assert len(overall_rows) == len(research.COST_SCENARIOS_BPS)
    for row in overall_rows:
        assert row["trade_count"] == int(ledger["active_trade"].sum())
    for cost in research.COST_SCENARIOS_BPS:
        assert (ledger["strategy_id"].iloc[0], "predicted_sign", cost) in curves


def test_exact_horizons_models_policies_and_configs():
    assert research.HORIZONS == (6, 12, 24)
    assert research.MODELS == (
        "ridge_regression",
        "hist_gradient_boosting_regressor",
        "training_mean_baseline",
    )
    assert research.POLICY_ORDER == (
        "predicted_sign",
        "train_abs_q90",
        "train_abs_q95",
    )
    assert research.MODEL_CONFIGS["ridge_regression"]["ridge"] == {
        "alpha": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "solver": "lsqr",
        "tol": 0.0001,
    }
    hgb = research.MODEL_CONFIGS["hist_gradient_boosting_regressor"][
        "hist_gradient_boosting_regressor"
    ]
    assert hgb["random_state"] == 1729
    assert hgb["early_stopping"] is False
    assert hgb["max_iter"] == 100
    assert research.COST_SCENARIOS_BPS == (0, 2, 5, 10)


def test_protected_files_and_previous_evidence_unchanged():
    root = research.BASE_DIR
    files = {
        "features": root / "features.py",
        "validation": root / "reports/model_candidate_validation_access.json",
    }
    directories = {
        "models": root / "model_artifacts",
        "candidates": root / "model_artifacts/candidates",
        "walkforward": research.SOURCE_WALKFORWARD_DIRECTORY,
        "backtest": research.SOURCE_BACKTEST_DIRECTORY,
        "selectivity": research.SOURCE_SELECTIVITY_DIRECTORY,
    }
    before_files = {key: bt.file_sha256(path) for key, path in files.items()}
    before_trees = {key: _tree_digest(path) for key, path in directories.items()}
    research.validate_previous_research_source(
        research.SOURCE_SELECTIVITY_DIRECTORY,
        expected_id=research.SOURCE_SELECTIVITY_EXPERIMENT_ID,
        expected_digest=research.SOURCE_SELECTIVITY_EXPERIMENT_DIGEST,
        expected_directory_digest=research.SOURCE_SELECTIVITY_DIRECTORY_DIGEST,
    )
    research.regression_metrics(np.arange(10), np.arange(10), np.zeros(10))
    assert {key: bt.file_sha256(path) for key, path in files.items()} == before_files
    assert {key: _tree_digest(path) for key, path in directories.items()} == before_trees


def test_no_network_exchange_runtime_or_deep_model_imports():
    path = research.BASE_DIR / "tools/model_executable_return_research.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {
            "ccxt",
            "requests",
            "urllib",
            "httpx",
            "socket",
            "exchange",
            "runtime",
            "live_writer",
            "live_executor",
            "torch",
            "tensorflow",
        }
    )
    assert not (research.BASE_DIR / "live_writer.py").exists()
    assert not (research.BASE_DIR / "live_executor.py").exists()


def test_output_directory_is_gitignored_and_research_only():
    lines = (research.BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "reports/model_executable_return_research/" in lines
    assert research.TARGET_CONTRACT["target_fields_in_features"] is False
    assert research.DECILE_CONTRACT["optimization_performed"] is False
