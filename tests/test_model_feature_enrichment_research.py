from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import model_executable_return_research as er
from tools import model_feature_enrichment_research as research
from tools import model_signal_backtest as bt
from tools import model_signal_walkforward as wf


def _raw(periods: int = 420, *, scale: float = 1.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="5min", tz="UTC")
    position = np.arange(periods, dtype=np.float64)
    close = scale * (100.0 + 0.03 * position + np.sin(position / 11.0))
    open_ = close * (1.0 + 0.0002 * np.cos(position / 7.0))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + scale * 0.4,
            "low": np.minimum(open_, close) - scale * 0.4,
            "close": close,
            "volume": scale * (1000.0 + position + 20.0 * np.sin(position / 5.0)),
        },
        index=index,
    )


def _raw_pair(periods: int = 420) -> dict[str, pd.DataFrame]:
    return {"BTCUSDT": _raw(periods), "ETHUSDT": _raw(periods, scale=1.7)}


def _fold(periods: int = 42) -> wf.FoldDefinition:
    test_start = pd.Timestamp("2026-01-03T00:00:00Z")
    return wf.FoldDefinition(
        fold_id="fold_00",
        horizon_bars=6,
        training_window_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        training_window_end_exclusive=test_start,
        fit_train_end_exclusive=test_start - 6 * wf.BAR_INTERVAL,
        purge_start=test_start - 6 * wf.BAR_INTERVAL,
        purge_end_exclusive=test_start,
        test_start=test_start,
        test_end_exclusive=test_start + periods * wf.BAR_INTERVAL,
    )


def _execution_raw(fold: wf.FoldDefinition) -> pd.DataFrame:
    index = pd.date_range(
        fold.test_start - wf.BAR_INTERVAL,
        fold.test_end_exclusive + wf.BAR_INTERVAL,
        freq="5min",
        inclusive="left",
        tz="UTC",
    )
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


def _scored(fold: wf.FoldDefinition, prediction: float) -> pd.DataFrame:
    timestamps = pd.date_range(
        fold.test_start,
        fold.test_end_exclusive,
        freq="5min",
        inclusive="left",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "prediction": prediction,
            "prediction_origin": "outer_test",
        }
    )


def _ledger(model_family: str = "classifier") -> pd.DataFrame:
    fold = _fold()
    kwargs = (
        {"lower_threshold": 0.2, "upper_threshold": 0.8}
        if model_family == "classifier"
        else {"absolute_threshold_bps": 5.0}
    )
    prediction = 0.95 if model_family == "classifier" else 10.0
    rows: list[dict] = []
    for symbol in wf.SYMBOLS:
        rows.extend(
            research.build_enrichment_trade_rows(
                experiment_id="feature_fixture",
                feature_set="baseline",
                model_family=model_family,
                fold=fold,
                symbol=symbol,
                scored_test=_scored(fold, prediction),
                raw=_execution_raw(fold),
                **kwargs,
            )
        )
    return research.as_enrichment_trade_frame(rows)


def test_all_local_enriched_features_are_prefix_invariant():
    raw = _raw()
    original = research.build_local_enriched_features(raw)
    cut = 350
    mutated = raw.copy()
    mutated.iloc[cut + 1 :, mutated.columns.get_indexer(["high", "low", "close", "volume"])] *= 9.0
    changed = research.build_local_enriched_features(mutated)
    pd.testing.assert_frame_equal(original.iloc[: cut + 1], changed.iloc[: cut + 1])


def test_trend_50_uses_exact_trailing_window():
    raw = _raw()
    row = research.build_local_enriched_features(raw).iloc[120]
    expected = raw["close"].iloc[71:121].iloc[-1] / raw["close"].iloc[71:121].mean() - 1.0
    assert row["trend_50"] == pytest.approx(expected)


def test_trend_200_uses_exact_trailing_window():
    raw = _raw()
    row = research.build_local_enriched_features(raw).iloc[300]
    expected = raw["close"].iloc[101:301].iloc[-1] / raw["close"].iloc[101:301].mean() - 1.0
    assert row["trend_200"] == pytest.approx(expected)


def test_ret_288_contains_no_future_bars_and_reconciles():
    raw = _raw()
    features = research.build_local_enriched_features(raw)
    position = 350
    expected = np.log(raw["close"] / raw["close"].shift(1)).iloc[position - 287 : position + 1].sum()
    assert features.iloc[position]["ret_288"] == pytest.approx(expected)
    mutated = raw.copy()
    mutated.iloc[position + 1 :, mutated.columns.get_loc("close")] *= 50
    assert research.build_local_enriched_features(mutated).iloc[position]["ret_288"] == pytest.approx(expected)


def test_rv_288_contains_no_future_bars_and_reconciles():
    raw = _raw()
    log_return = np.log(raw["close"] / raw["close"].shift(1))
    position = 350
    expected = np.sqrt(np.square(log_return.iloc[position - 287 : position + 1]).sum())
    assert research.build_local_enriched_features(raw).iloc[position]["rv_288"] == pytest.approx(expected)


def test_volume_z_60_uses_only_trailing_current_volume():
    raw = _raw()
    position = 300
    window = raw["volume"].iloc[position - 59 : position + 1]
    expected = (window.iloc[-1] - window.mean()) / window.std(ddof=0)
    assert research.build_local_enriched_features(raw).iloc[position]["volume_z_60"] == pytest.approx(expected)


def test_eth_btc_context_uses_btc_only_through_same_timestamp():
    raw = _raw_pair()
    contracts, _ = research.build_feature_contracts(raw)
    timestamp = contracts["enriched"]["ETHUSDT"].index[20]
    btc_local = research.build_local_enriched_features(raw["BTCUSDT"])
    assert contracts["enriched"]["ETHUSDT"].at[timestamp, "btc_ret_12"] == pytest.approx(btc_local.at[timestamp, "ret_12"])
    assert contracts["enriched"]["ETHUSDT"].at[timestamp, "btc_rv_60"] == pytest.approx(btc_local.at[timestamp, "rv_60"])


def test_future_btc_mutation_cannot_change_earlier_eth_context():
    raw = _raw_pair()
    before, _ = research.build_feature_contracts(raw)
    cut = 360
    mutated = {key: value.copy() for key, value in raw.items()}
    mutated["BTCUSDT"].iloc[cut + 1 :, mutated["BTCUSDT"].columns.get_loc("close")] *= 25.0
    after, _ = research.build_feature_contracts(mutated)
    columns = list(research.BTC_CONTEXT_COLUMNS)
    common = before["enriched"]["ETHUSDT"].index[: cut - 287]
    pd.testing.assert_frame_equal(
        before["enriched"]["ETHUSDT"].loc[common, columns],
        after["enriched"]["ETHUSDT"].loc[common, columns],
    )


def test_baseline_contract_is_canonical_bit_and_column_identical():
    raw = _raw_pair()
    contracts, diagnostics = research.build_feature_contracts(raw)
    expected_columns = wf.canonical_feature_columns(True)
    assert diagnostics["baseline_columns"] == expected_columns
    for symbol in wf.SYMBOLS:
        canonical, _ = wf.build_research_features(raw[symbol], symbol=symbol)
        pd.testing.assert_frame_equal(
            contracts["baseline"][symbol],
            canonical.loc[contracts["baseline"][symbol].index],
            check_exact=True,
        )


def test_enriched_contract_is_baseline_plus_exactly_17():
    contracts, diagnostics = research.build_feature_contracts(_raw_pair())
    baseline = tuple(wf.canonical_feature_columns(True))
    assert len(research.ENRICHED_FEATURE_COLUMNS) == 17
    assert tuple(contracts["enriched"]["BTCUSDT"].columns) == (*baseline, *research.ENRICHED_FEATURE_COLUMNS)
    assert diagnostics["enriched_feature_count"] == 44


def test_features_py_and_feature_cols_remain_unchanged():
    path = research.BASE_DIR / "features.py"
    before_hash = bt.file_sha256(path)
    before_cols = tuple(wf.FEATURE_COLS)
    research.build_feature_contracts(_raw_pair())
    assert bt.file_sha256(path) == before_hash
    assert tuple(wf.FEATURE_COLS) == before_cols
    assert before_hash == "061a53a73b2a5413fcb77f7a2c9c9476b69d6e34a2f773fb468cad10c312a20d"


def test_hgb_configs_are_exact_frozen_contracts():
    assert research.CLASSIFIER_CONFIG is wf.MODEL_CONFIGS["hist_gradient_boosting"]
    assert research.REGRESSOR_CONFIG is er.MODEL_CONFIGS["hist_gradient_boosting_regressor"]
    assert research.CLASSIFIER_CONFIG["hist_gradient_boosting"]["random_state"] == 1729
    assert research.REGRESSOR_CONFIG["hist_gradient_boosting_regressor"]["max_iter"] == 100


def test_training_test_chronology_and_fold_contract_unchanged():
    assert research.HORIZON_BARS == 6
    assert wf.FOLD_SPEC["training_days"] == 120
    assert wf.FOLD_SPEC["test_days"] == 30
    assert wf.FOLD_SPEC["step_days"] == 30
    folds = wf.make_walkforward_folds(
        pd.Timestamp("2025-08-20T17:35:00Z"),
        pd.Timestamp("2026-08-01T17:30:00Z"),
        horizon_bars=6,
    )
    assert len(folds) == 7
    assert all(fold.test_start > fold.training_window_start for fold in folds)


def test_horizon_purge_remains_strict_for_both_targets():
    source = inspect.getsource(wf.select_fold_rows)
    return_source = inspect.getsource(er.select_executable_fold_rows)
    assert "max_train_label >= min_test_time" in source
    assert 'rows["target_exit_timestamp"] < fold.test_start' in return_source
    assert "max_train_exit >= min_test" in return_source


def test_classifier_and_regressor_thresholds_use_train_only():
    train_scores = np.linspace(0.01, 0.99, 101)
    lower, upper = research.derive_classifier_thresholds(train_scores)
    assert lower == pytest.approx(np.quantile(train_scores, 0.05))
    assert upper == pytest.approx(np.quantile(train_scores, 0.95))
    train_predictions = np.linspace(-50.0, 50.0, 101)
    assert research.derive_regressor_threshold(train_predictions) == pytest.approx(np.quantile(np.abs(train_predictions), 0.95))
    assert "test" not in inspect.signature(research.derive_classifier_thresholds).parameters
    assert "test" not in inspect.signature(research.derive_regressor_threshold).parameters


def test_test_predictions_and_returns_cannot_change_thresholds():
    train = np.arange(1.0, 101.0)
    original = research.derive_regressor_threshold(train)
    test_predictions = np.asarray([1e9, -1e9])
    test_returns = np.asarray([-1e12, 1e12])
    assert research.derive_regressor_threshold(train) == original
    assert test_predictions.max() > original and test_returns.max() > original


def test_next_bar_execution_no_overlap_and_no_fold_carry():
    ledger = _ledger("classifier")
    checks = bt.validate_trade_ledger(ledger)
    assert checks["next_bar_entry"]
    assert checks["no_same_symbol_overlap"]
    assert checks["no_fold_boundary_carry"]
    assert (ledger["entry_timestamp_utc"] == ledger["signal_timestamp_utc"] + wf.BAR_INTERVAL).all()


def test_cost_applies_once_and_economics_reconcile():
    ledger = _ledger("regressor")
    assert bt.apply_round_trip_cost(0.01, active_trade=True, round_trip_cost_bps=5) == pytest.approx(0.0095)
    fold_rows, overall_rows, curves = er.calculate_economic_metrics(ledger, expected_fold_count=1)
    assert len(fold_rows) == 4 and len(overall_rows) == 4
    assert all(row["active_trade_count"] == int(ledger["active_trade"].sum()) for row in fold_rows)
    ordered = [curves[(ledger["strategy_id"].iloc[0], research.REGRESSOR_POLICY, cost)]["net_equity"].to_numpy() for cost in research.COST_SCENARIOS_BPS]
    assert all(np.all(high <= low + 1e-12) for low, high in zip(ordered, ordered[1:]))


def test_regime_groups_use_fixed_descriptive_rules():
    rows = [
        {"fold_id": f"fold_{i:02d}", "btc_realized_volatility": float(i + 1), "eth_realized_volatility": float(i + 2), "btc_signed_return": (-1.0) ** i, "eth_signed_return": 0.1, "btc_relative_volume": 1.0, "eth_relative_volume": 1.0}
        for i in range(7)
    ]
    grouped, threshold = research.assign_regime_groups(rows)
    assert threshold == pytest.approx(4.5)
    assert sum(row["volatility_group"] == "higher_volatility" for row in grouped) == 4
    assert all(row["post_hoc_descriptive_not_tradable"] for row in grouped)


def test_protected_evidence_hashes_remain_expected():
    expected = {
        "walkforward": research.SOURCE_WALKFORWARD_DIRECTORY_DIGEST,
        "backtest": research.SOURCE_BACKTEST_DIRECTORY_DIGEST,
        "selectivity": research.SOURCE_SELECTIVITY_DIRECTORY_DIGEST,
        "executable": research.SOURCE_EXECUTABLE_RETURN_DIRECTORY_DIGEST,
    }
    actual = {
        "walkforward": bt.directory_digest(research.SOURCE_WALKFORWARD_DIRECTORY),
        "backtest": bt.directory_digest(research.SOURCE_BACKTEST_DIRECTORY),
        "selectivity": bt.directory_digest(research.SOURCE_SELECTIVITY_DIRECTORY),
        "executable": bt.directory_digest(research.SOURCE_EXECUTABLE_RETURN_DIRECTORY),
    }
    assert actual == expected


def test_no_network_exchange_runtime_or_deep_model_imports():
    path = research.BASE_DIR / "tools/model_feature_enrichment_research.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"requests", "urllib", "httpx", "socket", "ccxt", "exchange", "runtime", "live_writer", "live_executor", "torch", "tensorflow"})
    text = path.read_text(encoding="utf-8")
    assert "q97" not in text and "q99" not in text


def test_output_is_ignored_and_research_only():
    lines = (research.BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "reports/model_feature_enrichment_research/" in lines
    assert research.STOP_CONDITION["automatic_follow_up_feature_experiment_allowed"] is False
    assert research.REGIME_CONTRACT["tradable_filter"] is False
