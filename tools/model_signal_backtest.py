"""Offline economic diagnostics for selected walk-forward signal models.

The tool refits only the frozen sklearn research configurations, creates
strictly out-of-sample non-overlapping signals, and writes research evidence
below ``reports/model_signal_backtest``.  It has no runtime, exchange, network,
candidate-training, promotion, or artifact-serving dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features import canonical_feature_columns  # noqa: E402
from tools import model_signal_walkforward as wf  # noqa: E402


SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = "model-signal-economic-backtest-v1"
SOURCE_WALKFORWARD_EXPERIMENT_ID = "walkforward_5e2797926e186f5c"
SOURCE_WALKFORWARD_EXPERIMENT_DIGEST = (
    "5e2797926e186f5c2ce759baf08ff5eabc47acd43eecf00460ba1a6b5b5cf310"
)
SOURCE_WALKFORWARD_DIRECTORY = (
    BASE_DIR
    / "reports"
    / "model_signal_research"
    / SOURCE_WALKFORWARD_EXPERIMENT_ID
)
BACKTEST_ROOT = BASE_DIR / "reports" / "model_signal_backtest"

STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "strategy_id": "logistic_30m",
        "role": "primary",
        "model_name": "logistic_regression",
        "horizon_bars": 6,
        "horizon_minutes": 30,
    },
    {
        "strategy_id": "hist_gradient_boosting_30m",
        "role": "nonlinear_comparator",
        "model_name": "hist_gradient_boosting",
        "horizon_bars": 6,
        "horizon_minutes": 30,
    },
    {
        "strategy_id": "logistic_2h",
        "role": "longer_horizon_comparator",
        "model_name": "logistic_regression",
        "horizon_bars": 24,
        "horizon_minutes": 120,
    },
)
POLICIES: tuple[dict[str, Any], ...] = (
    {
        "policy_name": "directional_0p5",
        "definition": "score >= 0.5 LONG; score < 0.5 SHORT",
        "always_active": True,
        "threshold_source": "fixed",
    },
    {
        "policy_name": "train_quantile_20_80",
        "definition": (
            "score >= train q80 LONG; score <= train q20 SHORT; otherwise FLAT"
        ),
        "always_active": False,
        "threshold_source": "same-fold fitted-model training scores only",
        "lower_quantile": 0.20,
        "upper_quantile": 0.80,
        "quantile_method": "linear",
        "training_profitability_consulted": False,
    },
)
COST_SCENARIOS_BPS = (0, 2, 5, 10)
SLEEVE_WEIGHTS = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
EXECUTION_CONTRACT = {
    "raw_timestamp_semantics": "five-minute bar open timestamp",
    "signal_available_after": "bar at signal timestamp completes",
    "entry": "open price of bar t+1",
    "exit": "close price of bar t+horizon_bars",
    "exit_execution_timestamp": "close time of exit bar: t+(horizon_bars+1) bars",
    "same_bar_execution_allowed": False,
    "schedule": "first test-start signal, then every horizon_bars within symbol/fold",
    "overlapping_trades_allowed": False,
    "fold_boundary_carry_allowed": False,
}
PORTFOLIO_CONTRACT = {
    "starting_equity": 1.0,
    "leverage": 1.0,
    "sleeve_weights": SLEEVE_WEIGHTS,
    "dynamic_flat_sleeve_reallocation": False,
    "portfolio_event_return": "0.5 * BTC sleeve return + 0.5 * ETH sleeve return",
    "compounding": "chronological",
    "borrowing": False,
    "risk_free_rate": 0.0,
    "daily_sharpe_annualization_days": 365,
    "daily_sharpe_standard_deviation_ddof": 1,
}
COST_CONTRACT = {
    "scenarios_round_trip_bps": list(COST_SCENARIOS_BPS),
    "synthetic_stress_scenarios": True,
    "venue_fee_claim": False,
    "active_trade_formula": "gross_simple_return - round_trip_cost_bps / 10000",
    "flat_trade_cost": 0.0,
    "round_trip_cost_applications_per_trade": 1,
}
LIMITATIONS = {
    "funding_included": False,
    "order_book_spread_modeled": False,
    "market_impact_modeled": False,
    "latency_beyond_next_bar_execution_modeled": False,
    "liquidation_modeled": False,
}
LEDGER_COLUMNS = (
    "experiment_id",
    "source_walkforward_experiment_digest",
    "strategy_id",
    "model_name",
    "horizon_bars",
    "horizon_minutes",
    "policy_name",
    "fold_id",
    "symbol",
    "fold_test_start_utc",
    "fold_test_end_exclusive_utc",
    "signal_timestamp_utc",
    "signal_bar_available_utc",
    "entry_timestamp_utc",
    "exit_bar_open_timestamp_utc",
    "exit_timestamp_utc",
    "score",
    "direction",
    "training_q20",
    "training_q80",
    "entry_price",
    "exit_price",
    "gross_simple_return",
    "gross_return_bps",
    "underlying_log_return",
    "active_trade",
    "sleeve_weight",
)


class SignalBacktestError(ValueError):
    """Raised when a source, timing, or accounting invariant fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_utc(value: Any) -> str:
    return wf.canonical_utc(value)


def file_sha256(path: Path | str) -> str:
    return wf.file_sha256(path)


def json_digest(value: Any) -> str:
    return wf.json_digest(value)


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return canonical_utc(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _clean_json(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise SignalBacktestError(f"refusing to write empty CSV: {path.name}")
    fieldnames = list(columns or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if row.get(key) is None
                    else format(float(row[key]), ".17g")
                    if isinstance(row.get(key), (float, np.floating))
                    else int(row[key])
                    if isinstance(row.get(key), np.integer)
                    else row.get(key)
                    for key in fieldnames
                }
            )


def directory_digest(path: Path | str) -> str:
    root = Path(path)
    if not root.is_dir():
        raise SignalBacktestError(f"missing directory for digest: {root}")
    inventory = [
        {
            "relative_path": child.relative_to(root).as_posix(),
            "sha256": file_sha256(child),
        }
        for child in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return json_digest(inventory)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SignalBacktestError(f"invalid JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise SignalBacktestError(f"JSON source must contain an object: {path}")
    return value


def validate_walkforward_source(
    source_directory: Path | str = SOURCE_WALKFORWARD_DIRECTORY,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    source = Path(source_directory).resolve()
    summary_path = source / "summary.json"
    manifest_path = source / "experiment_manifest.json"
    fold_path = source / "fold_metrics.csv"
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    for document, name in ((summary, "summary"), (manifest, "manifest")):
        if document.get("experiment_id") != SOURCE_WALKFORWARD_EXPERIMENT_ID:
            raise SignalBacktestError(f"unexpected source walk-forward {name} experiment ID")
        if document.get("experiment_digest") != SOURCE_WALKFORWARD_EXPERIMENT_DIGEST:
            raise SignalBacktestError(f"unexpected source walk-forward {name} digest")
        if document.get("research_only") is not True:
            raise SignalBacktestError(f"source walk-forward {name} is not research-only")
        if document.get("production_candidate") is not False:
            raise SignalBacktestError(f"source walk-forward {name} production flag is unsafe")
    if summary.get("historical_periods_pristine_holdout") is not False:
        raise SignalBacktestError("source history exposure warning is missing")
    if summary.get("feature_contract", {}).get("feature_contract_digest") != wf._feature_contract()[
        "feature_contract_digest"
    ]:
        raise SignalBacktestError("source feature contract no longer matches repository")
    fold_counts = summary.get("walk_forward_contract", {}).get("fold_count_by_horizon", {})
    if any(int(fold_counts.get(str(horizon), -1)) != 7 for horizon in (6, 24)):
        raise SignalBacktestError("source walk-forward does not contain seven required folds")
    output_manifest = manifest.get("outputs", {})
    for name, path in (("summary.json", summary_path), ("fold_metrics.csv", fold_path)):
        expected = output_manifest.get(name, {}).get("sha256")
        if not isinstance(expected, str) or file_sha256(path) != expected:
            raise SignalBacktestError(f"source walk-forward output hash mismatch: {name}")
    return summary, manifest, directory_digest(source)


def source_dataset_paths(summary: Mapping[str, Any]) -> tuple[Path, Path]:
    datasets = summary.get("source_datasets", {})
    try:
        earlier = Path(datasets["earlier"]["path"]).resolve()
        later = Path(datasets["later"]["path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise SignalBacktestError("source walk-forward dataset paths are missing") from exc
    if earlier.name != "phase24_5m_d0d635ff3f6a" or later.name != "phase24_5m_cac0baf0b726":
        raise SignalBacktestError("unexpected frozen source dataset IDs")
    return earlier, later


def raw_source_digests(
    earlier: Path, later: Path, expected: Mapping[str, str] | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset in (earlier, later):
        for symbol in wf.SYMBOLS:
            path = dataset / f"raw_{symbol}.csv"
            key = f"{dataset.name}/raw_{symbol}.csv"
            result[key] = file_sha256(path)
    if expected is not None and result != dict(expected):
        raise SignalBacktestError("frozen raw source digest mismatch")
    return result


def strategy_contract() -> list[dict[str, Any]]:
    result = []
    for strategy in STRATEGIES:
        model_name = str(strategy["model_name"])
        result.append(
            {
                **strategy,
                "frozen_model_configuration": wf.MODEL_CONFIGS[model_name],
            }
        )
    return result


def derive_training_quantiles(train_scores: Sequence[float]) -> tuple[float, float]:
    scores = np.asarray(train_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise SignalBacktestError("training score distribution is invalid")
    q20, q80 = np.quantile(scores, [0.20, 0.80], method="linear")
    if not (math.isfinite(float(q20)) and math.isfinite(float(q80)) and q20 <= q80):
        raise SignalBacktestError("training quantile thresholds are invalid")
    return float(q20), float(q80)


def policy_direction(
    score: float,
    policy_name: str,
    *,
    training_q20: float | None = None,
    training_q80: float | None = None,
) -> str:
    score = float(score)
    if not math.isfinite(score):
        raise SignalBacktestError("nonfinite signal score")
    if policy_name == "directional_0p5":
        return "LONG" if score >= 0.5 else "SHORT"
    if policy_name == "train_quantile_20_80":
        if training_q20 is None or training_q80 is None:
            raise SignalBacktestError("training quantile policy thresholds are missing")
        if score >= training_q80:
            return "LONG"
        if score <= training_q20:
            return "SHORT"
        return "FLAT"
    raise SignalBacktestError(f"unknown frozen policy: {policy_name}")


def scheduled_signal_times(fold: wf.FoldDefinition, horizon_bars: int) -> list[pd.Timestamp]:
    if horizon_bars != fold.horizon_bars or horizon_bars <= 0:
        raise SignalBacktestError("fold/schedule horizon mismatch")
    result: list[pd.Timestamp] = []
    signal = fold.test_start
    while signal < fold.test_end_exclusive:
        exit_bar_open = signal + horizon_bars * wf.BAR_INTERVAL
        exit_execution = exit_bar_open + wf.BAR_INTERVAL
        if exit_bar_open >= fold.test_end_exclusive or exit_execution > fold.test_end_exclusive:
            break
        result.append(signal)
        signal += horizon_bars * wf.BAR_INTERVAL
    if not result or result[0] != fold.test_start:
        raise SignalBacktestError(f"no test-start-anchored schedule for {fold.fold_id}")
    return result


def directional_return(raw_simple_return: float, direction: str) -> float:
    value = float(raw_simple_return)
    if direction == "LONG":
        return value
    if direction == "SHORT":
        return -value
    if direction == "FLAT":
        return 0.0
    raise SignalBacktestError(f"unknown direction: {direction}")


def apply_round_trip_cost(
    gross_simple_return: float, *, active_trade: bool, round_trip_cost_bps: float
) -> float:
    if round_trip_cost_bps < 0 or not math.isfinite(float(round_trip_cost_bps)):
        raise SignalBacktestError("round-trip cost must be finite and nonnegative")
    if not active_trade:
        return 0.0
    return float(gross_simple_return) - float(round_trip_cost_bps) / 10_000.0


def build_trade_rows(
    *,
    experiment_id: str,
    strategy: Mapping[str, Any],
    policy_name: str,
    fold: wf.FoldDefinition,
    symbol: str,
    scored_test: pd.DataFrame,
    raw: pd.DataFrame,
    training_q20: float | None,
    training_q80: float | None,
) -> list[dict[str, Any]]:
    horizon = int(strategy["horizon_bars"])
    if symbol not in SLEEVE_WEIGHTS:
        raise SignalBacktestError(f"unexpected symbol: {symbol}")
    if "timestamp" not in scored_test or "score" not in scored_test:
        raise SignalBacktestError("scored test frame lacks timestamp or score")
    indexed_scores = scored_test.set_index("timestamp")["score"]
    if not indexed_scores.index.is_unique:
        raise SignalBacktestError("duplicate OOS signal timestamps")
    rows: list[dict[str, Any]] = []
    for signal in scheduled_signal_times(fold, horizon):
        if signal not in indexed_scores.index:
            raise SignalBacktestError(
                f"missing scheduled OOS prediction: {symbol} {canonical_utc(signal)}"
            )
        entry_timestamp = signal + wf.BAR_INTERVAL
        exit_bar_open = signal + horizon * wf.BAR_INTERVAL
        exit_timestamp = exit_bar_open + wf.BAR_INTERVAL
        if entry_timestamp not in raw.index:
            raise SignalBacktestError(
                f"missing next-bar entry: {symbol} {canonical_utc(entry_timestamp)}"
            )
        if exit_bar_open not in raw.index:
            raise SignalBacktestError(
                f"missing fixed-horizon exit: {symbol} {canonical_utc(exit_bar_open)}"
            )
        if not (signal < entry_timestamp < exit_timestamp <= fold.test_end_exclusive):
            raise SignalBacktestError("trade timing or fold-boundary invariant failed")
        score = float(indexed_scores.loc[signal])
        direction = policy_direction(
            score,
            policy_name,
            training_q20=training_q20,
            training_q80=training_q80,
        )
        entry_price = float(raw.at[entry_timestamp, "open"])
        exit_price = float(raw.at[exit_bar_open, "close"])
        if not all(math.isfinite(value) and value > 0 for value in (entry_price, exit_price)):
            raise SignalBacktestError("entry/exit price is invalid")
        underlying_log_return = math.log(exit_price) - math.log(entry_price)
        raw_simple_return = exit_price / entry_price - 1.0
        gross = directional_return(raw_simple_return, direction)
        active = direction != "FLAT"
        rows.append(
            {
                "experiment_id": experiment_id,
                "source_walkforward_experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
                "strategy_id": strategy["strategy_id"],
                "model_name": strategy["model_name"],
                "horizon_bars": horizon,
                "horizon_minutes": int(strategy["horizon_minutes"]),
                "policy_name": policy_name,
                "fold_id": fold.fold_id,
                "symbol": symbol,
                "fold_test_start_utc": canonical_utc(fold.test_start),
                "fold_test_end_exclusive_utc": canonical_utc(fold.test_end_exclusive),
                "signal_timestamp_utc": canonical_utc(signal),
                "signal_bar_available_utc": canonical_utc(entry_timestamp),
                "entry_timestamp_utc": canonical_utc(entry_timestamp),
                "exit_bar_open_timestamp_utc": canonical_utc(exit_bar_open),
                "exit_timestamp_utc": canonical_utc(exit_timestamp),
                "score": score,
                "direction": direction,
                "training_q20": training_q20,
                "training_q80": training_q80,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_simple_return": gross,
                "gross_return_bps": gross * 10_000.0,
                "underlying_log_return": underlying_log_return,
                "active_trade": active,
                "sleeve_weight": float(SLEEVE_WEIGHTS[symbol]),
            }
        )
    return rows


def _as_trade_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if frame.empty:
        raise SignalBacktestError("trade ledger is empty")
    for column in (
        "signal_timestamp_utc",
        "signal_bar_available_utc",
        "entry_timestamp_utc",
        "exit_bar_open_timestamp_utc",
        "exit_timestamp_utc",
        "fold_test_start_utc",
        "fold_test_end_exclusive_utc",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["active_trade"] = frame["active_trade"].astype(bool)
    return frame


def validate_trade_ledger(ledger: pd.DataFrame) -> dict[str, bool]:
    required = set(LEDGER_COLUMNS)
    if not required <= set(ledger.columns) or ledger.empty:
        raise SignalBacktestError("trade ledger contract is incomplete")
    if ledger.duplicated(
        ["strategy_id", "policy_name", "fold_id", "symbol", "signal_timestamp_utc"]
    ).any():
        raise SignalBacktestError("duplicate scheduled signal in ledger")
    if not (ledger["entry_timestamp_utc"] > ledger["signal_timestamp_utc"]).all():
        raise SignalBacktestError("same-bar or earlier entry detected")
    if not (
        ledger["entry_timestamp_utc"]
        == ledger["signal_timestamp_utc"] + wf.BAR_INTERVAL
    ).all():
        raise SignalBacktestError("entry is not exactly the next bar")
    expected_exit_bar = ledger["signal_timestamp_utc"] + pd.to_timedelta(
        ledger["horizon_bars"] * 5, unit="minutes"
    )
    if not (ledger["exit_bar_open_timestamp_utc"] == expected_exit_bar).all():
        raise SignalBacktestError("fixed-horizon exit-bar mapping failed")
    if not (
        ledger["exit_timestamp_utc"]
        == ledger["exit_bar_open_timestamp_utc"] + wf.BAR_INTERVAL
    ).all():
        raise SignalBacktestError("exit execution timestamp mapping failed")
    if not (ledger["exit_timestamp_utc"] > ledger["entry_timestamp_utc"]).all():
        raise SignalBacktestError("exit is not later than entry")
    if not (
        (ledger["signal_timestamp_utc"] >= ledger["fold_test_start_utc"])
        & (ledger["exit_timestamp_utc"] <= ledger["fold_test_end_exclusive_utc"])
    ).all():
        raise SignalBacktestError("fold-boundary carry detected")
    active_from_direction = ledger["direction"].isin(["LONG", "SHORT"])
    if not (active_from_direction == ledger["active_trade"]).all():
        raise SignalBacktestError("active-trade and direction mismatch")
    if not (ledger.loc[~ledger["active_trade"], "gross_simple_return"] == 0.0).all():
        raise SignalBacktestError("flat signal has nonzero gross return")
    if not np.allclose(ledger["sleeve_weight"], 0.5, rtol=0, atol=0):
        raise SignalBacktestError("fixed sleeve weight changed")
    for _, group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id", "symbol"], sort=False
    ):
        ordered = group.sort_values("entry_timestamp_utc", kind="mergesort")
        if len(ordered) > 1:
            prior_exits = ordered["exit_timestamp_utc"].iloc[:-1].reset_index(drop=True)
            next_entries = ordered["entry_timestamp_utc"].iloc[1:].reset_index(drop=True)
            if not (next_entries >= prior_exits).all():
                raise SignalBacktestError("same-symbol trade overlap detected")
    return {
        "next_bar_entry": True,
        "fixed_horizon_exit": True,
        "exit_later_than_entry": True,
        "no_same_symbol_overlap": True,
        "no_fold_boundary_carry": True,
        "long_short_equals_active": True,
        "flat_gross_return_zero": True,
        "fixed_sleeve_weights": True,
    }


def build_portfolio_events(ledger: pd.DataFrame, round_trip_cost_bps: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["strategy_id", "policy_name", "fold_id", "signal_timestamp_utc"]
    for keys, group in ledger.groupby(group_columns, sort=True):
        if set(group["symbol"]) != set(wf.SYMBOLS) or len(group) != 2:
            raise SignalBacktestError("portfolio event lacks exactly one row per sleeve")
        by_symbol = {str(row.symbol): row for row in group.itertuples(index=False)}
        event: dict[str, Any] = {
            "strategy_id": keys[0],
            "policy_name": keys[1],
            "fold_id": keys[2],
            "cost_bps": int(round_trip_cost_bps),
            "signal_timestamp_utc": keys[3],
        }
        gross_portfolio = 0.0
        net_portfolio = 0.0
        active_weight = 0.0
        for symbol, prefix in (("BTCUSDT", "btc"), ("ETHUSDT", "eth")):
            trade = by_symbol[symbol]
            gross = float(trade.gross_simple_return)
            active = bool(trade.active_trade)
            net = apply_round_trip_cost(
                gross,
                active_trade=active,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            weight = float(trade.sleeve_weight)
            gross_portfolio += weight * gross
            net_portfolio += weight * net
            active_weight += weight * int(active)
            event.update(
                {
                    f"{prefix}_gross_sleeve_return": gross,
                    f"{prefix}_net_sleeve_return": net,
                    f"{prefix}_active_trade": active,
                }
            )
        event.update(
            {
                "entry_timestamp_utc": group["entry_timestamp_utc"].iloc[0],
                "exit_timestamp_utc": group["exit_timestamp_utc"].iloc[0],
                "portfolio_gross_event_return": gross_portfolio,
                "portfolio_net_event_return": net_portfolio,
                "active_sleeve_weight": active_weight,
            }
        )
        if not (
            (group["entry_timestamp_utc"] == event["entry_timestamp_utc"]).all()
            and (group["exit_timestamp_utc"] == event["exit_timestamp_utc"]).all()
        ):
            raise SignalBacktestError("BTC/ETH event timestamps do not align")
        if min(1.0 + gross_portfolio, 1.0 + net_portfolio) <= 0.0:
            raise SignalBacktestError("portfolio event would make equity nonpositive")
        rows.append(event)
    events = pd.DataFrame(rows).sort_values(
        ["exit_timestamp_utc", "strategy_id", "policy_name", "fold_id"], kind="mergesort"
    )
    return events.reset_index(drop=True)


def compound_returns(returns: Sequence[float]) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= -1.0):
        raise SignalBacktestError("invalid returns for compounding")
    return float(np.prod(1.0 + values, dtype=np.float64) - 1.0)


def maximum_drawdown(equity: Sequence[float]) -> float:
    values = np.asarray(equity, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise SignalBacktestError("invalid equity curve")
    with_start = np.concatenate(([1.0], values))
    peaks = np.maximum.accumulate(with_start)
    return float(np.min(with_start / peaks - 1.0))


def daily_returns_from_events(events: pd.DataFrame, return_column: str) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=np.float64)
    dates = pd.to_datetime(events["exit_timestamp_utc"], utc=True).dt.floor("D")
    values = events[return_column].astype(np.float64)
    frame = pd.DataFrame({"utc_day": dates, "return": values})
    return frame.groupby("utc_day", sort=True)["return"].apply(compound_returns)


def daily_sharpe(daily_returns: Sequence[float]) -> float | None:
    values = np.asarray(daily_returns, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    standard_deviation = float(np.std(values, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return None
    return float(np.mean(values) / standard_deviation * math.sqrt(365.0))


def enrich_equity_curve(events: pd.DataFrame) -> pd.DataFrame:
    result = events.sort_values("exit_timestamp_utc", kind="mergesort").copy().reset_index(drop=True)
    result["gross_equity"] = np.cumprod(1.0 + result["portfolio_gross_event_return"])
    result["net_equity"] = np.cumprod(1.0 + result["portfolio_net_event_return"])
    result["starting_equity"] = 1.0
    result["prior_gross_equity"] = result["gross_equity"].shift(1, fill_value=1.0)
    result["prior_net_equity"] = result["net_equity"].shift(1, fill_value=1.0)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], result["net_equity"].to_numpy())))
    result["drawdown"] = np.concatenate(([1.0], result["net_equity"].to_numpy()))[1:] / peaks[1:] - 1.0
    result["utc_day"] = pd.to_datetime(result["exit_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
    result["end_of_utc_day"] = False
    result["daily_return"] = np.nan
    daily = daily_returns_from_events(result, "portfolio_net_event_return")
    last_indices = result.groupby("utc_day", sort=True).tail(1).index
    for index in last_indices:
        day = pd.Timestamp(result.at[index, "exit_timestamp_utc"]).floor("D")
        result.at[index, "end_of_utc_day"] = True
        result.at[index, "daily_return"] = float(daily.loc[day])
    return result


def profit_factor(returns: Sequence[float]) -> float | None:
    values = np.asarray(returns, dtype=np.float64)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0.0:
        return None
    return gains / losses


def _symbol_cumulative_return(
    ledger: pd.DataFrame, symbol: str, cost_bps: int, *, gross: bool = False
) -> float:
    rows = ledger.loc[ledger["symbol"] == symbol].sort_values("exit_timestamp_utc")
    if gross:
        returns = rows["gross_simple_return"].to_numpy(dtype=np.float64)
    else:
        returns = np.asarray(
            [
                apply_round_trip_cost(
                    row.gross_simple_return,
                    active_trade=bool(row.active_trade),
                    round_trip_cost_bps=cost_bps,
                )
                for row in rows.itertuples(index=False)
            ],
            dtype=np.float64,
        )
    return compound_returns(returns)


def stream_metrics(
    ledger: pd.DataFrame, events: pd.DataFrame, cost_bps: int
) -> dict[str, Any]:
    active = ledger.loc[ledger["active_trade"]].copy()
    net_trade_returns = np.asarray(
        [
            apply_round_trip_cost(
                row.gross_simple_return,
                active_trade=True,
                round_trip_cost_bps=cost_bps,
            )
            for row in active.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    curve = enrich_equity_curve(events)
    gross_return = float(curve["gross_equity"].iloc[-1] - 1.0)
    net_return = float(curve["net_equity"].iloc[-1] - 1.0)
    daily = curve.loc[curve["end_of_utc_day"], "daily_return"].to_numpy(dtype=np.float64)
    trade_count = int(len(active))
    long_count = int((active["direction"] == "LONG").sum())
    short_count = int((active["direction"] == "SHORT").sum())
    return {
        "trade_count": trade_count,
        "long_count": long_count,
        "short_count": short_count,
        "flat_signal_count": int((~ledger["active_trade"]).sum()),
        "scheduled_signal_count": int(len(ledger)),
        "active_signal_fraction": float(trade_count / len(ledger)),
        "gross_cumulative_return": gross_return,
        "net_cumulative_return": net_return,
        "gross_return_bps": gross_return * 10_000.0,
        "net_return_bps": net_return * 10_000.0,
        "win_rate": float(np.mean(net_trade_returns > 0.0)) if trade_count else None,
        "average_trade_return": float(np.mean(net_trade_returns)) if trade_count else None,
        "median_trade_return": float(np.median(net_trade_returns)) if trade_count else None,
        "profit_factor": profit_factor(net_trade_returns) if trade_count else None,
        "btc_gross_cumulative_return": _symbol_cumulative_return(
            ledger, "BTCUSDT", cost_bps, gross=True
        ),
        "eth_gross_cumulative_return": _symbol_cumulative_return(
            ledger, "ETHUSDT", cost_bps, gross=True
        ),
        "btc_net_cumulative_return": _symbol_cumulative_return(
            ledger, "BTCUSDT", cost_bps
        ),
        "eth_net_cumulative_return": _symbol_cumulative_return(
            ledger, "ETHUSDT", cost_bps
        ),
        "portfolio_active_round_trips": trade_count,
        "portfolio_one_way_notional_turnover_equivalent": float(
            (active["sleeve_weight"] * 2.0).sum()
        ),
        "maximum_drawdown": maximum_drawdown(curve["net_equity"]),
        "daily_sharpe": daily_sharpe(daily),
        "daily_return_count": int(len(daily)),
        "final_gross_equity": float(curve["gross_equity"].iloc[-1]),
        "final_net_equity": float(curve["net_equity"].iloc[-1]),
    }


def fold_metrics_rows(
    ledger: pd.DataFrame,
    regime_map: Mapping[tuple[int, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, int, str], pd.DataFrame] = {}
    for (strategy_id, policy_name, fold_id), group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id"], sort=True
    ):
        horizon = int(group["horizon_bars"].iloc[0])
        for cost_bps in COST_SCENARIOS_BPS:
            events = build_portfolio_events(group, cost_bps)
            curve = enrich_equity_curve(events)
            curves[(strategy_id, policy_name, cost_bps, fold_id)] = curve
            thresholds = group[["training_q20", "training_q80"]].drop_duplicates()
            if len(thresholds) != 1:
                raise SignalBacktestError("fold policy thresholds are not frozen")
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "model_name": group["model_name"].iloc[0],
                    "horizon_bars": horizon,
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "cost_bps": cost_bps,
                    "fold_id": fold_id,
                    "test_start_utc": canonical_utc(group["fold_test_start_utc"].iloc[0]),
                    "test_end_exclusive_utc": canonical_utc(
                        group["fold_test_end_exclusive_utc"].iloc[0]
                    ),
                    "training_q20": thresholds["training_q20"].iloc[0],
                    "training_q80": thresholds["training_q80"].iloc[0],
                    **stream_metrics(group, events, cost_bps),
                    **dict(regime_map[(horizon, fold_id)]),
                }
            )
    return rows, curves


def overall_metrics_rows(
    ledger: pd.DataFrame, fold_metrics: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_frame = pd.DataFrame(fold_metrics)
    overall: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    for (strategy_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        for cost_bps in COST_SCENARIOS_BPS:
            events = build_portfolio_events(group, cost_bps)
            curve = enrich_equity_curve(events)
            for row in curve.to_dict("records"):
                equity_rows.append({key: _clean_json(value) for key, value in row.items()})
            metrics = stream_metrics(group, events, cost_bps)
            folds = fold_frame.loc[
                (fold_frame["strategy_id"] == strategy_id)
                & (fold_frame["policy_name"] == policy_name)
                & (fold_frame["cost_bps"] == cost_bps)
            ].sort_values("fold_id")
            if len(folds) != 7:
                raise SignalBacktestError("overall metrics require exactly seven folds")
            fold_returns = folds["net_cumulative_return"].to_numpy(dtype=np.float64)
            overall.append(
                {
                    "strategy_id": strategy_id,
                    "model_name": group["model_name"].iloc[0],
                    "horizon_bars": int(group["horizon_bars"].iloc[0]),
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "cost_bps": cost_bps,
                    **metrics,
                    "positive_fold_percentage": float(100.0 * np.mean(fold_returns > 0.0)),
                    "median_fold_net_return": float(np.median(fold_returns)),
                    "worst_fold_net_return": float(np.min(fold_returns)),
                    "best_fold_net_return": float(np.max(fold_returns)),
                }
            )
            months = pd.to_datetime(curve["exit_timestamp_utc"], utc=True).dt.strftime("%Y-%m")
            monthly = pd.DataFrame(
                {
                    "month_utc": months,
                    "gross": curve["portfolio_gross_event_return"],
                    "net": curve["portfolio_net_event_return"],
                }
            ).groupby("month_utc", sort=True)
            for month, monthly_group in monthly:
                monthly_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "policy_name": policy_name,
                        "cost_bps": cost_bps,
                        "month_utc": month,
                        "gross_monthly_return": compound_returns(monthly_group["gross"]),
                        "net_monthly_return": compound_returns(monthly_group["net"]),
                        "event_count": int(len(monthly_group)),
                    }
                )
    return overall, equity_rows, monthly_rows


def approximate_break_even_cost_bps(ledger: pd.DataFrame) -> float | None:
    zero_events = build_portfolio_events(ledger, 0)
    gross = zero_events["portfolio_gross_event_return"].to_numpy(dtype=np.float64)
    active_weight = zero_events["active_sleeve_weight"].to_numpy(dtype=np.float64)

    def equity(cost_bps: float) -> float:
        returns = gross - active_weight * cost_bps / 10_000.0
        if np.any(returns <= -1.0):
            return 0.0
        return float(np.prod(1.0 + returns, dtype=np.float64))

    if equity(0.0) <= 1.0:
        return None
    positive = active_weight > 0
    if not positive.any():
        return None
    safe_upper = float(
        np.min((1.0 + gross[positive]) / active_weight[positive] * 10_000.0) * 0.999999
    )
    if safe_upper <= 0.0 or equity(safe_upper) > 1.0:
        return None
    low, high = 0.0, safe_upper
    for _ in range(80):
        middle = (low + high) / 2.0
        if equity(middle) >= 1.0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def rank_overall(overall: Sequence[Mapping[str, Any]], cost_bps: int) -> list[dict[str, Any]]:
    selected = [row for row in overall if int(row["cost_bps"]) == int(cost_bps)]

    def finite_or_bottom(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return -math.inf
        return numeric if math.isfinite(numeric) else -math.inf

    ordered = sorted(
        selected,
        key=lambda row: (
            -float(row["positive_fold_percentage"]),
            -float(row["median_fold_net_return"]),
            -float(row["worst_fold_net_return"]),
            -finite_or_bottom(row["daily_sharpe"]),
            abs(float(row["maximum_drawdown"])),
            str(row["strategy_id"]),
            str(row["policy_name"]),
        ),
    )
    return [
        {
            "rank": index + 1,
            "cost_bps": cost_bps,
            "strategy_id": row["strategy_id"],
            "policy_name": row["policy_name"],
            "positive_fold_percentage": row["positive_fold_percentage"],
            "median_fold_net_return": row["median_fold_net_return"],
            "worst_fold_net_return": row["worst_fold_net_return"],
            "overall_daily_sharpe": row["daily_sharpe"],
            "maximum_drawdown": row["maximum_drawdown"],
        }
        for index, row in enumerate(ordered)
    ]


def regime_analysis(
    fold_metrics: Sequence[Mapping[str, Any]], *, cost_bps: int = 5
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(fold_metrics)
    fields = (
        "btc_regime_realized_volatility",
        "eth_regime_realized_volatility",
        "btc_regime_signed_period_return",
        "eth_regime_signed_period_return",
        "btc_regime_mean_volume_to_train_median",
        "eth_regime_mean_volume_to_train_median",
    )
    result: list[dict[str, Any]] = []
    selected = frame.loc[frame["cost_bps"] == cost_bps]
    for (strategy_id, policy_name), group in selected.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        ordered = group.sort_values("fold_id")
        returns = ordered["net_cumulative_return"].to_numpy(dtype=np.float64)
        correlations: dict[str, float | None] = {}
        for field in fields:
            descriptor = ordered[field].to_numpy(dtype=np.float64)
            correlations[field] = (
                float(np.corrcoef(returns, descriptor)[0, 1])
                if np.ptp(returns) > 0 and np.ptp(descriptor) > 0
                else None
            )
        best = ordered.loc[ordered["net_cumulative_return"].idxmax()]
        worst = ordered.loc[ordered["net_cumulative_return"].idxmin()]
        result.append(
            {
                "strategy_id": strategy_id,
                "policy_name": policy_name,
                "cost_bps": cost_bps,
                "fold_count": int(len(ordered)),
                "pearson_fold_return_correlations": correlations,
                "best_fold": {
                    "fold_id": best["fold_id"],
                    "net_return": best["net_cumulative_return"],
                    **{field: best[field] for field in fields},
                },
                "worst_fold": {
                    "fold_id": worst["fold_id"],
                    "net_return": worst["net_cumulative_return"],
                    **{field: worst[field] for field in fields},
                },
                "regime_filter_fitted": False,
                "regime_optimization_performed": False,
                "interpretation_limit": "seven exposed folds; descriptive association only",
            }
        )
    return result


def validate_accounting(
    ledger: pd.DataFrame,
    fold_metrics: Sequence[Mapping[str, Any]],
    overall: Sequence[Mapping[str, Any]],
    equity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    ledger_checks = validate_trade_ledger(ledger)
    for cost in COST_SCENARIOS_BPS:
        active = ledger["active_trade"].to_numpy(dtype=bool)
        gross = ledger["gross_simple_return"].to_numpy(dtype=np.float64)
        net = np.where(active, gross - cost / 10_000.0, 0.0)
        if np.any(net[active] > gross[active] + 1e-15):
            raise SignalBacktestError("cost improved an active trade")
        if np.any(net[~active] != 0.0):
            raise SignalBacktestError("cost charged on a flat signal")
        if cost == 0 and not np.array_equal(net, gross):
            raise SignalBacktestError("zero-cost net does not equal gross")
    equity = pd.DataFrame(equity_rows)
    key = ["strategy_id", "policy_name", "signal_timestamp_utc", "fold_id"]
    for (strategy_id, policy_name), group in equity.groupby(
        ["strategy_id", "policy_name"], sort=False
    ):
        curves = {
            int(cost): values.sort_values(key, kind="mergesort")
            for cost, values in group.groupby("cost_bps")
        }
        for lower, higher in zip(COST_SCENARIOS_BPS, COST_SCENARIOS_BPS[1:]):
            left, right = curves[lower], curves[higher]
            if left[key].reset_index(drop=True).equals(right[key].reset_index(drop=True)) is False:
                raise SignalBacktestError("cost-scenario event streams differ")
            if np.any(
                right["net_equity"].to_numpy(dtype=np.float64)
                > left["net_equity"].to_numpy(dtype=np.float64) + 1e-12
            ):
                raise SignalBacktestError("higher cost improved same-stream equity")
    for row in equity.itertuples(index=False):
        expected = 0.5 * float(row.btc_net_sleeve_return) + 0.5 * float(
            row.eth_net_sleeve_return
        )
        if not math.isclose(
            expected, float(row.portfolio_net_event_return), rel_tol=0.0, abs_tol=1e-15
        ):
            raise SignalBacktestError("portfolio event sleeve reconciliation failed")
    overall_frame = pd.DataFrame(overall)
    for keys, group in equity.groupby(["strategy_id", "policy_name", "cost_bps"]):
        ordered = group.sort_values("exit_timestamp_utc", kind="mergesort")
        recomputed_equity = 1.0 + compound_returns(ordered["portfolio_net_event_return"])
        reported = overall_frame.loc[
            (overall_frame["strategy_id"] == keys[0])
            & (overall_frame["policy_name"] == keys[1])
            & (overall_frame["cost_bps"] == keys[2])
        ]
        if len(reported) != 1:
            raise SignalBacktestError("missing overall metric row")
        report = reported.iloc[0]
        if not math.isclose(
            recomputed_equity, float(report["final_net_equity"]), rel_tol=0, abs_tol=1e-12
        ):
            raise SignalBacktestError("final equity reconciliation failed")
        recomputed_drawdown = maximum_drawdown(ordered["net_equity"])
        if not math.isclose(
            recomputed_drawdown, float(report["maximum_drawdown"]), rel_tol=0, abs_tol=1e-12
        ):
            raise SignalBacktestError("maximum drawdown reconciliation failed")
    fold_frame = pd.DataFrame(fold_metrics)
    for keys, group in ledger.groupby(["strategy_id", "policy_name", "fold_id"]):
        expected_active = int(group["active_trade"].sum())
        matches = fold_frame.loc[
            (fold_frame["strategy_id"] == keys[0])
            & (fold_frame["policy_name"] == keys[1])
            & (fold_frame["fold_id"] == keys[2])
        ]
        if len(matches) != len(COST_SCENARIOS_BPS) or not (
            matches["trade_count"] == expected_active
        ).all():
            raise SignalBacktestError("fold trade count does not reconcile to ledger")
        if not (
            matches["long_count"] + matches["short_count"] == matches["trade_count"]
        ).all():
            raise SignalBacktestError("long plus short does not equal active trades")
    return {
        **ledger_checks,
        "cost_never_improves_active_trade": True,
        "zero_cost_net_equals_gross": True,
        "higher_cost_never_improves_same_stream_equity": True,
        "portfolio_events_reconcile_from_half_weight_sleeves": True,
        "final_equity_recomputes_from_event_returns": True,
        "maximum_drawdown_recomputes_from_equity_curve": True,
        "trade_counts_reconcile_with_ledger": True,
        "no_cost_on_flat_signals": True,
    }


def _regime_map(summary: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    fields = (
        "btc_regime_realized_volatility",
        "eth_regime_realized_volatility",
        "btc_regime_signed_period_return",
        "eth_regime_signed_period_return",
        "btc_regime_mean_volume_to_train_median",
        "eth_regime_mean_volume_to_train_median",
    )
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for row in summary.get("regime_diagnostics", []):
        horizon = int(row["horizon_bars"])
        if horizon in (6, 24):
            result[(horizon, str(row["fold_id"]))] = {field: row[field] for field in fields}
    expected = {(horizon, f"fold_{index:02d}") for horizon in (6, 24) for index in range(7)}
    if set(result) != expected:
        raise SignalBacktestError("source fold regime diagnostics are incomplete")
    return result


def _positive_scores(estimator: Any, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(values), dtype=np.float64)
    classes = list(estimator.classes_)
    if 1 not in classes or probabilities.ndim != 2:
        raise SignalBacktestError("fitted model cannot produce positive-class scores")
    scores = probabilities[:, classes.index(1)]
    if not np.isfinite(scores).all():
        raise SignalBacktestError("model produced nonfinite scores")
    return scores


def _experiment_contract(
    raw_digests: Mapping[str, str], feature_contract_digest: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "tool_code_sha256": file_sha256(Path(__file__)),
        "source_walkforward_experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
        "raw_source_file_sha256": dict(sorted(raw_digests.items())),
        "feature_contract_digest": feature_contract_digest,
        "strategies": strategy_contract(),
        "policies": list(POLICIES),
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "cost_contract": COST_CONTRACT,
        "ranking_contract": {
            "ranking_cost_bps": 5,
            "ordered_fields": [
                "positive_fold_percentage descending",
                "median_fold_net_return descending",
                "worst_fold_net_return descending",
                "overall_daily_sharpe descending",
                "absolute_maximum_drawdown ascending",
                "strategy_id/policy_name ascending",
            ],
            "production_acceptance_threshold_defined": False,
        },
    }


def _ensure_output_root(path: Path | str) -> Path:
    requested = Path(path).resolve()
    allowed = BACKTEST_ROOT.resolve()
    if requested != allowed:
        raise SignalBacktestError(
            f"backtest output root must be exactly {allowed}; received {requested}"
        )
    return requested


def run_backtest(
    *,
    source_walkforward_directory: Path | str = SOURCE_WALKFORWARD_DIRECTORY,
    output_root: Path | str = BACKTEST_ROOT,
) -> dict[str, Any]:
    source_summary, source_manifest, source_tree_before = validate_walkforward_source(
        source_walkforward_directory
    )
    earlier, later = source_dataset_paths(source_summary)
    expected_raw = source_summary["raw_source_digests"]
    raw_digests_before = raw_source_digests(earlier, later, expected_raw)
    raw_by_symbol = {
        symbol: wf.combine_raw_windows(
            earlier / f"raw_{symbol}.csv",
            later / f"raw_{symbol}.csv",
            symbol=symbol,
        )
        for symbol in wf.SYMBOLS
    }
    if not raw_by_symbol["BTCUSDT"].index.equals(raw_by_symbol["ETHUSDT"].index):
        raise SignalBacktestError("BTC/ETH raw grids differ")
    feature_contract = wf._feature_contract()
    if feature_contract["feature_contract_digest"] != source_summary["feature_contract"][
        "feature_contract_digest"
    ]:
        raise SignalBacktestError("feature contract differs from source experiment")
    features_by_symbol = {
        symbol: wf.build_research_features(raw_by_symbol[symbol], symbol=symbol)[0]
        for symbol in wf.SYMBOLS
    }
    contract = _experiment_contract(
        raw_digests_before, feature_contract["feature_contract_digest"]
    )
    experiment_digest = json_digest(contract)
    experiment_id = f"backtest_{experiment_digest[:16]}"
    output = _ensure_output_root(output_root)
    final_directory = output / experiment_id
    staging_directory = output / f".{experiment_id}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise SignalBacktestError(f"backtest output already exists: {experiment_id}")

    ordered_features = canonical_feature_columns(True)
    trade_rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        horizon = int(strategy["horizon_bars"])
        target_rows = pd.concat(
            [
                wf.build_fixed_horizon_rows(
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
            raise SignalBacktestError("real backtest requires seven complete folds")
        for fold in folds:
            train, test = wf.select_fold_rows(target_rows, fold)
            X_train = train.loc[:, ordered_features].to_numpy(dtype=np.float64)
            y_train = train["target"].to_numpy(dtype=np.int8)
            X_test = test.loc[:, ordered_features].to_numpy(dtype=np.float64)
            test_scores, estimator = wf.fit_model_scores(
                str(strategy["model_name"]), X_train, y_train, X_test
            )
            train_scores = _positive_scores(estimator, X_train)
            q20, q80 = derive_training_quantiles(train_scores)
            scored_test = test.loc[:, ["timestamp", "symbol"]].copy()
            scored_test["score"] = test_scores
            for policy in POLICIES:
                policy_name = str(policy["policy_name"])
                lower = q20 if policy_name == "train_quantile_20_80" else None
                upper = q80 if policy_name == "train_quantile_20_80" else None
                thresholds.append(
                    {
                        "strategy_id": strategy["strategy_id"],
                        "fold_id": fold.fold_id,
                        "policy_name": policy_name,
                        "training_q20": lower,
                        "training_q80": upper,
                        "threshold_source": (
                            "training_scores_only" if lower is not None else "fixed_0.5"
                        ),
                    }
                )
                for symbol in wf.SYMBOLS:
                    symbol_scores = scored_test.loc[
                        scored_test["symbol"] == symbol, ["timestamp", "score"]
                    ]
                    trade_rows.extend(
                        build_trade_rows(
                            experiment_id=experiment_id,
                            strategy=strategy,
                            policy_name=policy_name,
                            fold=fold,
                            symbol=symbol,
                            scored_test=symbol_scores,
                            raw=raw_by_symbol[symbol],
                            training_q20=lower,
                            training_q80=upper,
                        )
                    )
    ledger = _as_trade_frame(trade_rows)
    validate_trade_ledger(ledger)
    fold_metrics, _ = fold_metrics_rows(ledger, _regime_map(source_summary))
    overall, equity_rows, monthly_rows = overall_metrics_rows(ledger, fold_metrics)
    invariant_checks = validate_accounting(ledger, fold_metrics, overall, equity_rows)
    break_even = []
    for (strategy_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        break_even.append(
            {
                "strategy_id": strategy_id,
                "policy_name": policy_name,
                "approximate_break_even_round_trip_cost_bps": approximate_break_even_cost_bps(
                    group
                ),
                "diagnostic_only": True,
                "venue_fee_recommendation": False,
            }
        )
    rankings = {str(cost): rank_overall(overall, cost) for cost in COST_SCENARIOS_BPS}
    base_order = [
        (row["strategy_id"], row["policy_name"]) for row in rankings["5"]
    ]
    ranking_changes = {
        str(cost): [
            (row["strategy_id"], row["policy_name"]) for row in rankings[str(cost)]
        ]
        != base_order
        for cost in (0, 2, 10)
    }
    source_tree_after = directory_digest(source_walkforward_directory)
    raw_digests_after = raw_source_digests(earlier, later, expected_raw)
    if source_tree_after != source_tree_before:
        raise SignalBacktestError("source walk-forward evidence changed during run")
    if raw_digests_after != raw_digests_before:
        raise SignalBacktestError("frozen raw sources changed during run")
    invariant_checks["source_raw_file_hashes_unchanged"] = True
    invariant_checks["source_walkforward_directory_hash_unchanged"] = True

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "research_only": True,
        "production_candidate": False,
        "promotion_allowed": False,
        "live_execution_allowed": False,
        "production_authorization_blocked_reasons": [
            "all evaluated history is research-exposed and is not a pristine holdout",
            "costs are synthetic stress scenarios and are not current venue fees",
            "funding, order-book spread, market impact, additional latency, and liquidation are not modeled",
            "future untouched confirmation is required before any deployment consideration",
        ],
        "source_walkforward": {
            "experiment_id": SOURCE_WALKFORWARD_EXPERIMENT_ID,
            "experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
            "directory": str(Path(source_walkforward_directory).resolve()),
            "directory_digest": source_tree_before,
            "manifest_digest_contract": source_manifest.get("experiment_digest"),
        },
        "raw_source_digests": raw_digests_before,
        "feature_contract_digest": feature_contract["feature_contract_digest"],
        "strategies": strategy_contract(),
        "policies": list(POLICIES),
        "policy_thresholds_by_fold": thresholds,
        "cost_contract": COST_CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "execution_limitations": LIMITATIONS,
        "fold_count": 7,
        "trade_ledger_row_count": int(len(ledger)),
        "overall_metrics": overall,
        "approximate_break_even_costs": break_even,
        "research_ranking_5bps": rankings["5"],
        "rankings_by_cost_bps": rankings,
        "ranking_changes_relative_to_5bps": ranking_changes,
        "regime_analysis_5bps": regime_analysis(fold_metrics, cost_bps=5),
        "accounting_invariants": invariant_checks,
        "all_returns_reconciled": all(invariant_checks.values()),
        "monthly_returns_generated": True,
        "historical_periods_pristine_holdout": False,
        "historical_research_exposure_warning": (
            "All historical periods used here are research-exposed and must never be "
            "described as pristine holdout data."
        ),
        "future_untouched_confirmation_required_before_deployment": True,
        "production_acceptance_threshold_defined": False,
        "safety_contract": {
            "candidate_training_performed": False,
            "candidate_models_or_artifacts_written": False,
            "validation_or_internal_test_ledger_written": False,
            "runtime_or_live_execution_modified": False,
            "exchange_or_network_access_performed": False,
            "models_saved": False,
            "outputs_restricted_to": str(output),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir()
    summary_path = staging_directory / "summary.json"
    ledger_path = staging_directory / "trade_ledger.csv"
    fold_path = staging_directory / "fold_metrics.csv"
    equity_path = staging_directory / "equity_curve.csv"
    monthly_path = staging_directory / "monthly_returns.csv"
    manifest_path = staging_directory / "experiment_manifest.json"
    _write_json(summary_path, summary)
    ledger_output = ledger.copy()
    for column in ledger_output.columns:
        if pd.api.types.is_datetime64_any_dtype(ledger_output[column]):
            ledger_output[column] = ledger_output[column].map(canonical_utc)
    _write_csv(ledger_path, ledger_output.to_dict("records"), LEDGER_COLUMNS)
    _write_csv(fold_path, fold_metrics)
    _write_csv(equity_path, equity_rows)
    _write_csv(monthly_path, monthly_rows)
    outputs = {
        name: {"sha256": file_sha256(path), "row_count": row_count}
        for name, path, row_count in (
            ("summary.json", summary_path, None),
            ("trade_ledger.csv", ledger_path, len(ledger_output)),
            ("fold_metrics.csv", fold_path, len(fold_metrics)),
            ("equity_curve.csv", equity_path, len(equity_rows)),
            ("monthly_returns.csv", monthly_path, len(monthly_rows)),
        )
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
        "experiment_contract": contract,
        "source_walkforward_unchanged_during_run": True,
        "raw_sources_unchanged_during_run": True,
        "outputs": outputs,
        "execution_environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": wf.sklearn.__version__,
        },
    }
    _write_json(manifest_path, manifest)
    staging_directory.rename(final_directory)
    summary["output_directory"] = str(final_directory)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research-only offline signal economic backtest."
    )
    parser.add_argument(
        "--source-walkforward", type=Path, default=SOURCE_WALKFORWARD_DIRECTORY
    )
    parser.add_argument("--output-root", type=Path, default=BACKTEST_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_backtest(
            source_walkforward_directory=args.source_walkforward,
            output_root=args.output_root,
        )
    except (SignalBacktestError, wf.SignalResearchError, OSError, ValueError) as exc:
        print(f"model signal economic backtest failed closed: {exc}", file=sys.stderr)
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
