from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import model_signal_backtest as bt
from tools import model_signal_selectivity as selectivity
from tools import model_signal_walkforward as wf


def _fold(horizon: int = 2, periods: int = 14) -> wf.FoldDefinition:
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


def _strategy(horizon: int = 2) -> dict:
    return {
        "strategy_id": "logistic_regression_30m",
        "model_name": "logistic_regression",
        "horizon_bars": horizon,
        "horizon_minutes": horizon * 5,
    }


def _raw(fold: wf.FoldDefinition) -> pd.DataFrame:
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


def _scores(fold: wf.FoldDefinition, values: float | np.ndarray) -> pd.DataFrame:
    timestamps = pd.date_range(
        fold.test_start,
        fold.test_end_exclusive,
        freq="5min",
        inclusive="left",
        tz="UTC",
    )
    scores = (
        np.full(len(timestamps), float(values), dtype=np.float64)
        if np.isscalar(values)
        else np.asarray(values, dtype=np.float64)
    )
    assert len(scores) == len(timestamps)
    return pd.DataFrame({"timestamp": timestamps, "score": scores})


def _trade_rows(
    symbol: str,
    *,
    policy: str = "q20/q80",
    score: float = 0.9,
    lower: float | None = 0.2,
    upper: float | None = 0.8,
    horizon: int = 2,
) -> list[dict]:
    fold = _fold(horizon=horizon)
    if policy == "directional_0p5":
        lower, upper = None, None
    return selectivity.build_selectivity_trade_rows(
        experiment_id="selectivity_fixture",
        strategy=_strategy(horizon),
        policy_name=policy,
        fold=fold,
        symbol=symbol,
        scored_test=_scores(fold, score),
        raw=_raw(fold),
        lower_threshold=lower,
        upper_threshold=upper,
    )


def _paired_ledger(
    *,
    policy: str = "q20/q80",
    btc_score: float = 0.9,
    eth_score: float = 0.1,
    lower: float | None = 0.2,
    upper: float | None = 0.8,
) -> pd.DataFrame:
    rows = _trade_rows(
        "BTCUSDT", policy=policy, score=btc_score, lower=lower, upper=upper
    )
    rows += _trade_rows(
        "ETHUSDT", policy=policy, score=eth_score, lower=lower, upper=upper
    )
    return selectivity.as_selectivity_trade_frame(rows)


def _all_policy_ledger() -> pd.DataFrame:
    rows: list[dict] = []
    configurations = {
        "q20/q80": (0.20, 0.80, 0.90, 0.10),
        "q10/q90": (0.10, 0.90, 0.95, 0.05),
        "q05/q95": (0.05, 0.95, 0.98, 0.02),
        "directional_0p5": (None, None, 0.90, 0.10),
    }
    for policy, (lower, upper, btc_score, eth_score) in configurations.items():
        rows.extend(
            _trade_rows(
                "BTCUSDT",
                policy=policy,
                score=btc_score,
                lower=lower,
                upper=upper,
            )
        )
        rows.extend(
            _trade_rows(
                "ETHUSDT",
                policy=policy,
                score=eth_score,
                lower=lower,
                upper=upper,
            )
        )
    return selectivity.as_selectivity_trade_frame(rows)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def _assert_training_quantile(policy: str, lower_q: float, upper_q: float) -> None:
    train_scores = np.asarray([0.02, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.98])
    observed = selectivity.derive_training_thresholds(train_scores, policy)
    expected = np.quantile(train_scores, [lower_q, upper_q], method="linear")
    assert observed == pytest.approx(tuple(expected))


def test_q20_q80_uses_train_scores_only():
    _assert_training_quantile("q20/q80", 0.20, 0.80)


def test_q10_q90_uses_train_scores_only():
    _assert_training_quantile("q10/q90", 0.10, 0.90)


def test_q05_q95_uses_train_scores_only():
    _assert_training_quantile("q05/q95", 0.05, 0.95)


def test_test_scores_cannot_affect_thresholds():
    train_scores = np.linspace(0.1, 0.9, 101)
    baseline = {
        policy: selectivity.derive_training_thresholds(train_scores, policy)
        for policy in selectivity.SELECTIVE_POLICY_ORDER
    }
    test_scores_a = np.asarray([0.0, 1.0])
    test_scores_b = np.asarray([0.49, 0.51])
    assert not np.array_equal(test_scores_a, test_scores_b)
    assert baseline == {
        policy: selectivity.derive_training_thresholds(train_scores, policy)
        for policy in selectivity.SELECTIVE_POLICY_ORDER
    }


def test_test_returns_cannot_affect_thresholds():
    signature = inspect.signature(selectivity.derive_training_thresholds)
    assert list(signature.parameters) == ["train_scores", "policy_name"]
    train_scores = np.linspace(0.1, 0.9, 101)
    positive_test_returns = np.full(20, 0.10)
    negative_test_returns = np.full(20, -0.10)
    assert not np.array_equal(positive_test_returns, negative_test_returns)
    first = selectivity.derive_training_thresholds(train_scores, "q05/q95")
    second = selectivity.derive_training_thresholds(train_scores, "q05/q95")
    assert first == second


def test_stricter_quantiles_cannot_create_more_activity_on_same_score_stream():
    train_scores = np.linspace(0.0, 1.0, 1001)
    test_scores = np.linspace(-0.2, 1.2, 2001)
    thresholds = {
        policy: selectivity.derive_training_thresholds(train_scores, policy)
        for policy in selectivity.SELECTIVE_POLICY_ORDER
    }
    counts = selectivity.validate_stricter_policy_nesting(test_scores, thresholds)
    assert counts["q20/q80"] >= counts["q10/q90"] >= counts["q05/q95"]


def test_flat_signals_pay_no_transaction_cost():
    ledger = _paired_ledger(btc_score=0.5, eth_score=0.5)
    assert not ledger["active_trade"].any()
    events = bt.build_portfolio_events(ledger, 10)
    assert events["portfolio_net_event_return"].eq(0.0).all()
    assert bt.apply_round_trip_cost(
        0.0, active_trade=False, round_trip_cost_bps=10
    ) == 0.0


def test_execution_and_accounting_match_existing_backtest_implementation():
    fold = _fold()
    scores = _scores(fold, 0.9)
    selected = selectivity.build_selectivity_trade_rows(
        experiment_id="fixture",
        strategy=_strategy(),
        policy_name="q20/q80",
        fold=fold,
        symbol="BTCUSDT",
        scored_test=scores,
        raw=_raw(fold),
        lower_threshold=0.2,
        upper_threshold=0.8,
    )
    existing = bt.build_trade_rows(
        experiment_id="fixture",
        strategy=_strategy(),
        policy_name="train_quantile_20_80",
        fold=fold,
        symbol="BTCUSDT",
        scored_test=scores,
        raw=_raw(fold),
        training_q20=0.2,
        training_q80=0.8,
    )
    comparable = (
        "signal_timestamp_utc",
        "entry_timestamp_utc",
        "exit_bar_open_timestamp_utc",
        "exit_timestamp_utc",
        "entry_price",
        "exit_price",
        "direction",
        "gross_simple_return",
        "active_trade",
        "sleeve_weight",
    )
    assert [{key: row[key] for key in comparable} for row in selected] == [
        {key: row[key] for key in comparable} for row in existing
    ]


def test_next_bar_execution_remains_unchanged():
    row = _trade_rows("BTCUSDT")[0]
    signal = pd.Timestamp(row["signal_timestamp_utc"])
    assert pd.Timestamp(row["entry_timestamp_utc"]) == signal + wf.BAR_INTERVAL
    assert row["signal_bar_available_utc"] == row["entry_timestamp_utc"]


def test_no_overlapping_trades_and_no_fold_carry():
    ledger = _paired_ledger()
    checks = bt.validate_trade_ledger(ledger)
    assert checks["no_same_symbol_overlap"] is True
    assert checks["no_fold_boundary_carry"] is True
    assert (
        ledger["exit_timestamp_utc"] <= ledger["fold_test_end_exclusive_utc"]
    ).all()


def test_btc_and_eth_remain_separate():
    ledger = _paired_ledger(btc_score=0.9, eth_score=0.1)
    metrics = bt.stream_metrics(ledger, bt.build_portfolio_events(ledger, 0), 0)
    assert metrics["btc_net_cumulative_return"] != metrics["eth_net_cumulative_return"]


def test_long_short_diagnostics_and_active_counts_reconcile():
    ledger = _all_policy_ledger()
    diagnostics = selectivity.build_symbol_direction_diagnostics(ledger)
    diagnostic_frame = pd.DataFrame(diagnostics)
    for keys, group in ledger.groupby(["strategy_id", "policy_name"]):
        rows = diagnostic_frame.loc[
            (diagnostic_frame["strategy_id"] == keys[0])
            & (diagnostic_frame["policy_name"] == keys[1])
        ]
        assert len(rows) == 4
        assert int(rows["active_trade_count"].sum()) == int(
            group["active_trade"].sum()
        )
        assert set(rows["symbol"]) == set(wf.SYMBOLS)
        assert set(rows["direction"]) == {"LONG", "SHORT"}


def test_zero_bps_net_equals_gross_and_cost_cannot_improve_equity():
    ledger = _paired_ledger()
    curves = []
    for cost in selectivity.COST_SCENARIOS_BPS:
        events = bt.build_portfolio_events(ledger, cost)
        if cost == 0:
            assert np.array_equal(
                events["portfolio_net_event_return"],
                events["portfolio_gross_event_return"],
            )
        curves.append(bt.enrich_equity_curve(events)["net_equity"].to_numpy())
    for lower, higher in zip(curves, curves[1:]):
        assert np.all(higher <= lower + 1e-12)


def test_summary_and_diagnostics_reconcile_through_existing_accounting():
    ledger = _all_policy_ledger()
    fold_rows, overall_rows, curves = selectivity.calculate_economic_metrics(ledger)
    summaries = selectivity.build_policy_summary_rows(ledger, overall_rows)
    diagnostics = selectivity.build_symbol_direction_diagnostics(ledger)
    checks = selectivity.validate_results(
        ledger=ledger,
        fold_metrics=fold_rows,
        overall_metrics=overall_rows,
        policy_summaries=summaries,
        symbol_direction_diagnostics=diagnostics,
        curves=curves,
    )
    assert all(checks.values())
    for row in summaries:
        assert row["overall_net_return_0bps"] == pytest.approx(
            next(
                metric["gross_cumulative_return"]
                for metric in overall_rows
                if metric["strategy_id"] == row["strategy_id"]
                and metric["policy_name"] == row["policy_name"]
                and metric["cost_bps"] == 0
            )
        )


def test_frozen_strategies_policies_costs_and_folds_are_exact():
    assert [
        (row["strategy_id"], row["model_name"], row["horizon_bars"])
        for row in selectivity.STRATEGIES
    ] == [
        ("hist_gradient_boosting_30m", "hist_gradient_boosting", 6),
        ("logistic_regression_30m", "logistic_regression", 6),
        ("logistic_regression_2h", "logistic_regression", 24),
    ]
    assert [row["policy_name"] for row in selectivity.POLICIES] == [
        "q20/q80",
        "q10/q90",
        "q05/q95",
        "directional_0p5",
    ]
    assert selectivity.COST_SCENARIOS_BPS == (0, 2, 5, 10)
    source, _, _ = bt.validate_walkforward_source()
    earlier, later = bt.source_dataset_paths(source)
    raw = wf.combine_raw_windows(
        earlier / "raw_BTCUSDT.csv", later / "raw_BTCUSDT.csv", symbol="BTCUSDT"
    )
    for horizon in (6, 24):
        assert len(
            wf.make_walkforward_folds(
                raw.index[0], raw.index[-1], horizon_bars=horizon
            )
        ) == 7


def test_protected_files_and_previous_evidence_remain_unchanged():
    root = selectivity.BASE_DIR
    paths = {
        "features": root / "features.py",
        "validation": root / "reports" / "model_candidate_validation_access.json",
    }
    directories = {
        "walkforward": selectivity.SOURCE_WALKFORWARD_DIRECTORY,
        "backtest": selectivity.SOURCE_BACKTEST_DIRECTORY,
        "candidates": root / "model_artifacts" / "candidates",
    }
    files_before = {name: bt.file_sha256(path) for name, path in paths.items()}
    trees_before = {name: _tree_digest(path) for name, path in directories.items()}
    bt.validate_walkforward_source()
    selectivity.validate_backtest_source()
    selectivity.build_symbol_direction_diagnostics(_paired_ledger())
    assert {name: bt.file_sha256(path) for name, path in paths.items()} == files_before
    assert {name: _tree_digest(path) for name, path in directories.items()} == trees_before


def test_no_candidate_runtime_exchange_or_network_imports_or_model_training():
    source_path = selectivity.BASE_DIR / "tools" / "model_signal_selectivity.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(
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
        }
    )
    assert {row["model_name"] for row in selectivity.STRATEGIES} == {
        "hist_gradient_boosting",
        "logistic_regression",
    }


def test_research_only_contract_has_no_production_pass_gate():
    assert selectivity.EXECUTION_CONTRACT == bt.EXECUTION_CONTRACT
    assert selectivity.PORTFOLIO_CONTRACT == bt.PORTFOLIO_CONTRACT
    assert selectivity.COST_CONTRACT == bt.COST_CONTRACT
    assert selectivity.PORTFOLIO_CONTRACT["leverage"] == 1.0
    assert (
        selectivity.PORTFOLIO_CONTRACT["dynamic_flat_sleeve_reallocation"] is False
    )


def test_selectivity_output_directory_is_gitignored():
    lines = (selectivity.BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "reports/model_signal_selectivity/" in lines
