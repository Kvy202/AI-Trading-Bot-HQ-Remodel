from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import model_signal_backtest as bt
from tools import model_signal_walkforward as wf


def _fold(horizon: int = 2, periods: int = 12) -> wf.FoldDefinition:
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


def _strategy(horizon: int = 2, strategy_id: str = "fixture") -> dict:
    return {
        "strategy_id": strategy_id,
        "model_name": "logistic_regression",
        "horizon_bars": horizon,
        "horizon_minutes": horizon * 5,
    }


def _raw(fold: wf.FoldDefinition, extra_before: int = 1) -> pd.DataFrame:
    start = fold.test_start - extra_before * wf.BAR_INTERVAL
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


def _scores(fold: wf.FoldDefinition, value: float = 0.8) -> pd.DataFrame:
    times = pd.date_range(
        fold.test_start, fold.test_end_exclusive, freq="5min", inclusive="left", tz="UTC"
    )
    return pd.DataFrame({"timestamp": times, "score": value})


def _trade_rows(
    symbol: str,
    *,
    score: float = 0.8,
    policy: str = "directional_0p5",
    q20: float | None = None,
    q80: float | None = None,
    horizon: int = 2,
) -> list[dict]:
    fold = _fold(horizon=horizon)
    return bt.build_trade_rows(
        experiment_id="backtest_fixture",
        strategy=_strategy(horizon),
        policy_name=policy,
        fold=fold,
        symbol=symbol,
        scored_test=_scores(fold, score),
        raw=_raw(fold),
        training_q20=q20,
        training_q80=q80,
    )


def _paired_ledger(
    *, btc_score: float = 0.8, eth_score: float = 0.8, policy: str = "directional_0p5"
) -> pd.DataFrame:
    q20, q80 = (0.2, 0.8) if policy == "train_quantile_20_80" else (None, None)
    rows = _trade_rows("BTCUSDT", score=btc_score, policy=policy, q20=q20, q80=q80)
    rows += _trade_rows("ETHUSDT", score=eth_score, policy=policy, q20=q20, q80=q80)
    return bt._as_trade_frame(rows)


def _regime(horizon: int = 2) -> dict[tuple[int, str], dict]:
    return {
        (horizon, "fold_00"): {
            "btc_regime_realized_volatility": 0.1,
            "eth_regime_realized_volatility": 0.2,
            "btc_regime_signed_period_return": 0.03,
            "eth_regime_signed_period_return": -0.04,
            "btc_regime_mean_volume_to_train_median": 1.1,
            "eth_regime_mean_volume_to_train_median": 1.2,
        }
    }


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def test_next_bar_entry_prevents_same_bar_execution():
    row = _trade_rows("BTCUSDT")[0]
    signal = pd.Timestamp(row["signal_timestamp_utc"])
    entry = pd.Timestamp(row["entry_timestamp_utc"])
    assert entry == signal + wf.BAR_INTERVAL
    assert entry > signal
    assert row["signal_bar_available_utc"] == row["entry_timestamp_utc"]


def test_fixed_horizon_exit_mapping_is_exact():
    horizon = 6
    fold = _fold(horizon=horizon, periods=40)
    rows = bt.build_trade_rows(
        experiment_id="fixture",
        strategy=_strategy(horizon),
        policy_name="directional_0p5",
        fold=fold,
        symbol="BTCUSDT",
        scored_test=_scores(fold),
        raw=_raw(fold),
        training_q20=None,
        training_q80=None,
    )
    row = rows[0]
    signal = pd.Timestamp(row["signal_timestamp_utc"])
    assert pd.Timestamp(row["exit_bar_open_timestamp_utc"]) == signal + horizon * wf.BAR_INTERVAL
    assert pd.Timestamp(row["exit_timestamp_utc"]) == signal + (horizon + 1) * wf.BAR_INTERVAL
    raw = _raw(fold)
    assert row["exit_price"] == raw.at[signal + horizon * wf.BAR_INTERVAL, "close"]


def test_missing_next_bar_entry_fails_closed():
    fold = _fold()
    raw = _raw(fold).drop(fold.test_start + wf.BAR_INTERVAL)
    with pytest.raises(bt.SignalBacktestError, match="missing next-bar entry"):
        bt.build_trade_rows(
            experiment_id="fixture",
            strategy=_strategy(),
            policy_name="directional_0p5",
            fold=fold,
            symbol="BTCUSDT",
            scored_test=_scores(fold),
            raw=raw,
            training_q20=None,
            training_q80=None,
        )


def test_missing_fixed_horizon_exit_fails_closed():
    fold = _fold()
    raw = _raw(fold).drop(fold.test_start + 2 * wf.BAR_INTERVAL)
    with pytest.raises(bt.SignalBacktestError, match="missing fixed-horizon exit"):
        bt.build_trade_rows(
            experiment_id="fixture",
            strategy=_strategy(),
            policy_name="directional_0p5",
            fold=fold,
            symbol="BTCUSDT",
            scored_test=_scores(fold),
            raw=raw,
            training_q20=None,
            training_q80=None,
        )


def test_non_overlapping_schedule_is_horizon_anchored():
    fold = _fold(horizon=2, periods=12)
    scheduled = bt.scheduled_signal_times(fold, 2)
    assert scheduled[0] == fold.test_start
    assert all(b - a == 2 * wf.BAR_INTERVAL for a, b in zip(scheduled, scheduled[1:]))
    ledger = bt._as_trade_frame(_trade_rows("BTCUSDT"))
    ordered = ledger.sort_values("entry_timestamp_utc")
    assert (
        ordered["entry_timestamp_utc"].iloc[1:].reset_index(drop=True)
        >= ordered["exit_timestamp_utc"].iloc[:-1].reset_index(drop=True)
    ).all()


def test_no_trade_carries_across_fold_boundary():
    fold = _fold()
    rows = _trade_rows("BTCUSDT")
    assert max(pd.Timestamp(row["exit_timestamp_utc"]) for row in rows) <= fold.test_end_exclusive
    assert all(pd.Timestamp(row["exit_bar_open_timestamp_utc"]) < fold.test_end_exclusive for row in rows)


def test_long_return_calculation():
    assert bt.directional_return(0.0123, "LONG") == pytest.approx(0.0123)
    row = _trade_rows("BTCUSDT", score=0.8)[0]
    assert row["gross_simple_return"] == pytest.approx(row["exit_price"] / row["entry_price"] - 1)


def test_short_return_calculation():
    assert bt.directional_return(0.0123, "SHORT") == pytest.approx(-0.0123)
    row = _trade_rows("BTCUSDT", score=0.2)[0]
    assert row["direction"] == "SHORT"
    assert row["gross_simple_return"] == pytest.approx(-(row["exit_price"] / row["entry_price"] - 1))


def test_flat_return_calculation():
    row = _trade_rows(
        "BTCUSDT", score=0.5, policy="train_quantile_20_80", q20=0.2, q80=0.8
    )[0]
    assert row["direction"] == "FLAT"
    assert row["active_trade"] is False
    assert row["gross_simple_return"] == 0.0


def test_round_trip_cost_is_applied_exactly_once():
    assert bt.apply_round_trip_cost(0.01, active_trade=True, round_trip_cost_bps=5) == pytest.approx(
        0.0095
    )
    assert bt.apply_round_trip_cost(0.01, active_trade=False, round_trip_cost_bps=5) == 0.0


def test_zero_cost_net_equals_gross():
    ledger = _paired_ledger()
    events = bt.build_portfolio_events(ledger, 0)
    assert np.array_equal(
        events["portfolio_net_event_return"], events["portfolio_gross_event_return"]
    )
    assert np.array_equal(events["btc_net_sleeve_return"], events["btc_gross_sleeve_return"])


def test_increasing_cost_cannot_improve_same_stream_equity():
    ledger = _paired_ledger()
    final = []
    for cost in bt.COST_SCENARIOS_BPS:
        curve = bt.enrich_equity_curve(bt.build_portfolio_events(ledger, cost))
        final.append(curve["net_equity"].iloc[-1])
    assert all(higher <= lower for lower, higher in zip(final, final[1:]))


def test_fixed_half_sleeve_accounting():
    ledger = _paired_ledger(btc_score=0.8, eth_score=0.2)
    event = bt.build_portfolio_events(ledger, 0).iloc[0]
    expected = 0.5 * event["btc_net_sleeve_return"] + 0.5 * event["eth_net_sleeve_return"]
    assert event["portfolio_net_event_return"] == pytest.approx(expected)


def test_flat_sleeve_is_not_reallocated():
    ledger = _paired_ledger(
        btc_score=0.9, eth_score=0.5, policy="train_quantile_20_80"
    )
    event = bt.build_portfolio_events(ledger, 0).iloc[0]
    assert event["eth_active_trade"] is False or event["eth_active_trade"] == False
    assert event["eth_net_sleeve_return"] == 0.0
    assert event["portfolio_net_event_return"] == pytest.approx(
        0.5 * event["btc_net_sleeve_return"]
    )


def test_compounding_is_chronological():
    ledger = _paired_ledger()
    events = bt.build_portfolio_events(ledger, 0)
    curve = bt.enrich_equity_curve(events)
    expected = np.cumprod(1.0 + events.sort_values("exit_timestamp_utc")["portfolio_net_event_return"])
    assert np.allclose(curve["net_equity"], expected)
    assert curve["prior_net_equity"].iloc[0] == 1.0
    assert curve["starting_equity"].eq(1.0).all()
    assert curve["net_equity"].iloc[-1] == pytest.approx(
        1.0 + bt.compound_returns(curve["portfolio_net_event_return"])
    )


def test_maximum_drawdown_calculation():
    equity = [1.10, 0.90, 1.20, 1.00]
    assert bt.maximum_drawdown(equity) == pytest.approx(0.90 / 1.10 - 1.0)


def test_daily_sharpe_handles_zero_or_invalid_variance():
    assert bt.daily_sharpe([0.0, 0.0, 0.0]) is None
    assert bt.daily_sharpe([0.01]) is None
    assert bt.daily_sharpe([0.01, np.nan]) is None
    expected = np.mean([0.01, -0.005, 0.02]) / np.std([0.01, -0.005, 0.02], ddof=1) * np.sqrt(365)
    assert bt.daily_sharpe([0.01, -0.005, 0.02]) == pytest.approx(expected)


def test_training_quantiles_use_training_scores_only():
    train_scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    q20, q80 = bt.derive_training_quantiles(train_scores)
    assert (q20, q80) == pytest.approx(tuple(np.quantile(train_scores, [0.2, 0.8])))


def test_test_scores_do_not_determine_quantile_thresholds():
    train_scores = np.linspace(0.1, 0.9, 101)
    first = bt.derive_training_quantiles(train_scores)
    test_scores_a = np.asarray([0.0, 1.0])
    test_scores_b = np.asarray([0.49, 0.51])
    assert first == bt.derive_training_quantiles(train_scores)
    assert [bt.policy_direction(x, "train_quantile_20_80", training_q20=first[0], training_q80=first[1]) for x in test_scores_a] == ["SHORT", "LONG"]
    assert [bt.policy_direction(x, "train_quantile_20_80", training_q20=first[0], training_q80=first[1]) for x in test_scores_b] == ["FLAT", "FLAT"]


def test_strategy_definitions_are_exact_and_deterministic():
    assert [(row["strategy_id"], row["model_name"], row["horizon_bars"]) for row in bt.STRATEGIES] == [
        ("logistic_30m", "logistic_regression", 6),
        ("hist_gradient_boosting_30m", "hist_gradient_boosting", 6),
        ("logistic_2h", "logistic_regression", 24),
    ]
    assert [row["policy_name"] for row in bt.POLICIES] == [
        "directional_0p5",
        "train_quantile_20_80",
    ]
    assert bt.COST_SCENARIOS_BPS == (0, 2, 5, 10)


def test_fold_metrics_reconcile_with_trade_ledger():
    ledger = _paired_ledger()
    rows, _ = bt.fold_metrics_rows(ledger, _regime())
    assert len(rows) == 4
    for row in rows:
        assert row["trade_count"] == int(ledger["active_trade"].sum())
        assert row["long_count"] + row["short_count"] == row["trade_count"]
        assert row["scheduled_signal_count"] == len(ledger)


def test_equity_curve_reconciles_with_stream_summary():
    ledger = _paired_ledger()
    events = bt.build_portfolio_events(ledger, 5)
    curve = bt.enrich_equity_curve(events)
    metrics = bt.stream_metrics(ledger, events, 5)
    assert metrics["final_net_equity"] == pytest.approx(curve["net_equity"].iloc[-1])
    assert metrics["net_cumulative_return"] == pytest.approx(curve["net_equity"].iloc[-1] - 1)
    assert metrics["maximum_drawdown"] == pytest.approx(bt.maximum_drawdown(curve["net_equity"]))


def test_btc_and_eth_results_remain_separate():
    ledger = _paired_ledger(btc_score=0.8, eth_score=0.2)
    metrics = bt.stream_metrics(ledger, bt.build_portfolio_events(ledger, 0), 0)
    assert metrics["btc_net_cumulative_return"] != metrics["eth_net_cumulative_return"]


def test_break_even_cost_is_deterministic_or_absent_when_gross_loses():
    ledger = _paired_ledger()
    first = bt.approximate_break_even_cost_bps(ledger)
    second = bt.approximate_break_even_cost_bps(ledger)
    assert first == second
    losing = ledger.copy()
    losing.loc[losing["active_trade"], "gross_simple_return"] = -0.01
    assert bt.approximate_break_even_cost_bps(losing) is None


def test_walkforward_source_report_is_not_modified():
    before = _tree_digest(bt.SOURCE_WALKFORWARD_DIRECTORY)
    summary, manifest, digest = bt.validate_walkforward_source()
    after = _tree_digest(bt.SOURCE_WALKFORWARD_DIRECTORY)
    assert before == after
    assert summary["experiment_digest"] == bt.SOURCE_WALKFORWARD_EXPERIMENT_DIGEST
    assert manifest["experiment_digest"] == bt.SOURCE_WALKFORWARD_EXPERIMENT_DIGEST
    assert digest == bt.directory_digest(bt.SOURCE_WALKFORWARD_DIRECTORY)


def test_validation_ledger_is_not_modified():
    path = bt.BASE_DIR / "reports" / "model_candidate_validation_access.json"
    before = bt.file_sha256(path)
    bt.validate_trade_ledger(_paired_ledger())
    assert bt.file_sha256(path) == before


def test_candidate_artifacts_are_not_modified():
    path = bt.BASE_DIR / "model_artifacts" / "candidates"
    before = _tree_digest(path)
    bt.enrich_equity_curve(bt.build_portfolio_events(_paired_ledger(), 5))
    assert _tree_digest(path) == before


def test_frozen_raw_files_are_not_modified():
    source, _, _ = bt.validate_walkforward_source()
    earlier, later = bt.source_dataset_paths(source)
    paths = [dataset / f"raw_{symbol}.csv" for dataset in (earlier, later) for symbol in wf.SYMBOLS]
    before = {path: bt.file_sha256(path) for path in paths}
    for symbol in wf.SYMBOLS:
        wf.combine_raw_windows(
            earlier / f"raw_{symbol}.csv", later / f"raw_{symbol}.csv", symbol=symbol
        )
    assert {path: bt.file_sha256(path) for path in paths} == before


def test_no_runtime_exchange_or_network_imports():
    source_path = bt.BASE_DIR / "tools" / "model_signal_backtest.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    forbidden = {
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
    assert imported_roots.isdisjoint(forbidden)


def test_research_only_and_execution_limitations_are_frozen():
    assert bt.LIMITATIONS == {
        "funding_included": False,
        "order_book_spread_modeled": False,
        "market_impact_modeled": False,
        "latency_beyond_next_bar_execution_modeled": False,
        "liquidation_modeled": False,
    }
    assert bt.PORTFOLIO_CONTRACT["leverage"] == 1.0
    assert bt.PORTFOLIO_CONTRACT["dynamic_flat_sleeve_reallocation"] is False


def test_backtest_output_directory_is_gitignored():
    gitignore = (bt.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert "reports/model_signal_backtest/" in gitignore.splitlines()
