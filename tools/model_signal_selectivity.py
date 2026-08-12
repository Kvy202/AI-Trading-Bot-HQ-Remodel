"""Offline research of signal selectivity and turnover reduction.

This task refits only the three frozen sklearn configurations from the exposed
walk-forward experiment.  Per-fold thresholds come exclusively from fitted
model scores on the outer training rows.  The module reuses the validated
execution and accounting functions in :mod:`tools.model_signal_backtest` and
writes evidence only below ``reports/model_signal_selectivity``.

There are deliberately no runtime, exchange, network, candidate-training,
promotion, model-saving, or live-execution dependencies.
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


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools import model_signal_backtest as bt  # noqa: E402
from tools import model_signal_walkforward as wf  # noqa: E402


SCHEMA_VERSION = 1
TOOL_CONTRACT_VERSION = "model-signal-selectivity-v1"
SOURCE_WALKFORWARD_EXPERIMENT_ID = bt.SOURCE_WALKFORWARD_EXPERIMENT_ID
SOURCE_WALKFORWARD_EXPERIMENT_DIGEST = bt.SOURCE_WALKFORWARD_EXPERIMENT_DIGEST
SOURCE_WALKFORWARD_DIRECTORY = bt.SOURCE_WALKFORWARD_DIRECTORY
SOURCE_BACKTEST_EXPERIMENT_ID = "backtest_5964745630c574fd"
SOURCE_BACKTEST_EXPERIMENT_DIGEST = (
    "5964745630c574fdc728dbaf14bae7def59c3ff5a1fdcfe995e407d9c91084a4"
)
SOURCE_BACKTEST_DIRECTORY = (
    BASE_DIR / "reports" / "model_signal_backtest" / SOURCE_BACKTEST_EXPERIMENT_ID
)
SELECTIVITY_ROOT = BASE_DIR / "reports" / "model_signal_selectivity"

STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "strategy_id": "hist_gradient_boosting_30m",
        "model_name": "hist_gradient_boosting",
        "horizon_bars": 6,
        "horizon_minutes": 30,
    },
    {
        "strategy_id": "logistic_regression_30m",
        "model_name": "logistic_regression",
        "horizon_bars": 6,
        "horizon_minutes": 30,
    },
    {
        "strategy_id": "logistic_regression_2h",
        "model_name": "logistic_regression",
        "horizon_bars": 24,
        "horizon_minutes": 120,
    },
)

POLICIES: tuple[dict[str, Any], ...] = (
    {
        "policy_name": "q20/q80",
        "lower_quantile": 0.20,
        "upper_quantile": 0.80,
        "definition": (
            "score >= frozen train q80 LONG; score <= frozen train q20 SHORT; "
            "otherwise FLAT"
        ),
        "threshold_source": "same-fold fitted-model training scores only",
        "quantile_method": "linear",
        "training_profitability_consulted": False,
        "test_scores_consulted": False,
        "test_returns_consulted": False,
        "reference_comparator": False,
    },
    {
        "policy_name": "q10/q90",
        "lower_quantile": 0.10,
        "upper_quantile": 0.90,
        "definition": (
            "score >= frozen train q90 LONG; score <= frozen train q10 SHORT; "
            "otherwise FLAT"
        ),
        "threshold_source": "same-fold fitted-model training scores only",
        "quantile_method": "linear",
        "training_profitability_consulted": False,
        "test_scores_consulted": False,
        "test_returns_consulted": False,
        "reference_comparator": False,
    },
    {
        "policy_name": "q05/q95",
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
        "definition": (
            "score >= frozen train q95 LONG; score <= frozen train q05 SHORT; "
            "otherwise FLAT"
        ),
        "threshold_source": "same-fold fitted-model training scores only",
        "quantile_method": "linear",
        "training_profitability_consulted": False,
        "test_scores_consulted": False,
        "test_returns_consulted": False,
        "reference_comparator": False,
    },
    {
        "policy_name": "directional_0p5",
        "lower_quantile": None,
        "upper_quantile": None,
        "definition": "score >= 0.5 LONG; score < 0.5 SHORT",
        "threshold_source": "fixed 0.5 reference comparator",
        "training_profitability_consulted": False,
        "test_scores_consulted": False,
        "test_returns_consulted": False,
        "reference_comparator": True,
    },
)

SELECTIVE_POLICY_ORDER = ("q20/q80", "q10/q90", "q05/q95")
POLICY_ORDER = (*SELECTIVE_POLICY_ORDER, "directional_0p5")
COST_SCENARIOS_BPS = bt.COST_SCENARIOS_BPS
SLEEVE_WEIGHTS = dict(bt.SLEEVE_WEIGHTS)
EXECUTION_CONTRACT = dict(bt.EXECUTION_CONTRACT)
PORTFOLIO_CONTRACT = dict(bt.PORTFOLIO_CONTRACT)
COST_CONTRACT = dict(bt.COST_CONTRACT)
LIMITATIONS = dict(bt.LIMITATIONS)

SELECTIVITY_LEDGER_COLUMNS = (
    *bt.LEDGER_COLUMNS,
    "source_backtest_experiment_digest",
    "lower_quantile",
    "upper_quantile",
    "training_lower_threshold",
    "training_upper_threshold",
    "threshold_source",
)


class SignalSelectivityError(ValueError):
    """Raised when a selectivity, source, timing, or accounting invariant fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def policy_contract(policy_name: str) -> dict[str, Any]:
    for policy in POLICIES:
        if policy["policy_name"] == policy_name:
            return dict(policy)
    raise SignalSelectivityError(f"unknown frozen selectivity policy: {policy_name}")


def strategy_contract() -> list[dict[str, Any]]:
    return [
        {
            **strategy,
            "frozen_model_configuration": wf.MODEL_CONFIGS[str(strategy["model_name"])],
        }
        for strategy in STRATEGIES
    ]


def validate_backtest_source(
    source_directory: Path | str = SOURCE_BACKTEST_DIRECTORY,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    source = Path(source_directory).resolve()
    summary_path = source / "summary.json"
    manifest_path = source / "experiment_manifest.json"
    summary = bt._load_json(summary_path)
    manifest = bt._load_json(manifest_path)
    for document, name in ((summary, "summary"), (manifest, "manifest")):
        if document.get("experiment_id") != SOURCE_BACKTEST_EXPERIMENT_ID:
            raise SignalSelectivityError(f"unexpected source backtest {name} experiment ID")
        if document.get("experiment_digest") != SOURCE_BACKTEST_EXPERIMENT_DIGEST:
            raise SignalSelectivityError(f"unexpected source backtest {name} digest")
        if document.get("research_only") is not True:
            raise SignalSelectivityError(f"source backtest {name} is not research-only")
        if document.get("production_candidate") is not False:
            raise SignalSelectivityError(f"source backtest {name} production flag is unsafe")
    if summary.get("execution_contract") != EXECUTION_CONTRACT:
        raise SignalSelectivityError("source backtest execution contract changed")
    if summary.get("portfolio_contract") != PORTFOLIO_CONTRACT:
        raise SignalSelectivityError("source backtest portfolio contract changed")
    if summary.get("cost_contract") != COST_CONTRACT:
        raise SignalSelectivityError("source backtest cost contract changed")
    for name, metadata in manifest.get("outputs", {}).items():
        output_path = source / name
        expected = metadata.get("sha256")
        if not isinstance(expected, str) or bt.file_sha256(output_path) != expected:
            raise SignalSelectivityError(f"source backtest output hash mismatch: {name}")
    return summary, manifest, bt.directory_digest(source)


def protected_source_hashes(
    *,
    earlier: Path,
    later: Path,
    expected_raw: Mapping[str, str],
    source_walkforward_directory: Path | str = SOURCE_WALKFORWARD_DIRECTORY,
    source_backtest_directory: Path | str = SOURCE_BACKTEST_DIRECTORY,
) -> dict[str, Any]:
    return {
        "features.py": bt.file_sha256(BASE_DIR / "features.py"),
        "walkforward_evidence_directory": bt.directory_digest(
            source_walkforward_directory
        ),
        "backtest_evidence_directory": bt.directory_digest(source_backtest_directory),
        "raw_source_files": bt.raw_source_digests(earlier, later, expected_raw),
    }


def derive_training_thresholds(
    train_scores: Sequence[float], policy_name: str
) -> tuple[float | None, float | None]:
    """Derive one policy's thresholds solely from same-fold training scores."""

    policy = policy_contract(policy_name)
    if policy_name == "directional_0p5":
        return None, None
    scores = np.asarray(train_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise SignalSelectivityError("training score distribution is invalid")
    quantiles = [float(policy["lower_quantile"]), float(policy["upper_quantile"])]
    lower, upper = np.quantile(scores, quantiles, method="linear")
    if not (math.isfinite(float(lower)) and math.isfinite(float(upper))):
        raise SignalSelectivityError("training score thresholds are nonfinite")
    if lower > upper:
        raise SignalSelectivityError("training score thresholds are inverted")
    return float(lower), float(upper)


def selectivity_direction(
    score: float,
    policy_name: str,
    *,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
) -> str:
    value = float(score)
    if not math.isfinite(value):
        raise SignalSelectivityError("nonfinite signal score")
    if policy_name == "directional_0p5":
        return "LONG" if value >= 0.5 else "SHORT"
    policy_contract(policy_name)
    if lower_threshold is None or upper_threshold is None:
        raise SignalSelectivityError("selectivity policy thresholds are missing")
    lower = float(lower_threshold)
    upper = float(upper_threshold)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise SignalSelectivityError("selectivity policy thresholds are invalid")
    if value >= upper:
        return "LONG"
    if value <= lower:
        return "SHORT"
    return "FLAT"


def active_mask(
    scores: Sequence[float],
    policy_name: str,
    *,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray(
        [
            selectivity_direction(
                score,
                policy_name,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            != "FLAT"
            for score in values
        ],
        dtype=bool,
    )


def validate_stricter_policy_nesting(
    scores: Sequence[float],
    thresholds_by_policy: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, int]:
    masks: dict[str, np.ndarray] = {}
    for name in SELECTIVE_POLICY_ORDER:
        try:
            lower, upper = thresholds_by_policy[name]
        except KeyError as exc:
            raise SignalSelectivityError(f"missing thresholds for {name}") from exc
        masks[name] = active_mask(
            scores,
            name,
            lower_threshold=lower,
            upper_threshold=upper,
        )
    for looser, stricter in zip(SELECTIVE_POLICY_ORDER, SELECTIVE_POLICY_ORDER[1:]):
        if np.any(masks[stricter] & ~masks[looser]):
            raise SignalSelectivityError(
                f"stricter policy {stricter} created activity outside {looser}"
            )
    return {name: int(mask.sum()) for name, mask in masks.items()}


def build_selectivity_trade_rows(
    *,
    experiment_id: str,
    strategy: Mapping[str, Any],
    policy_name: str,
    fold: wf.FoldDefinition,
    symbol: str,
    scored_test: pd.DataFrame,
    raw: pd.DataFrame,
    lower_threshold: float | None,
    upper_threshold: float | None,
) -> list[dict[str, Any]]:
    """Apply selectivity while preserving the backtest's execution mapping."""

    policy = policy_contract(policy_name)
    base_rows = bt.build_trade_rows(
        experiment_id=experiment_id,
        strategy=strategy,
        policy_name="directional_0p5",
        fold=fold,
        symbol=symbol,
        scored_test=scored_test,
        raw=raw,
        training_q20=None,
        training_q80=None,
    )
    result: list[dict[str, Any]] = []
    for base in base_rows:
        direction = selectivity_direction(
            float(base["score"]),
            policy_name,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
        raw_simple_return = float(base["exit_price"]) / float(base["entry_price"]) - 1.0
        gross = bt.directional_return(raw_simple_return, direction)
        active = direction != "FLAT"
        base.update(
            {
                "policy_name": policy_name,
                "direction": direction,
                "training_q20": (
                    lower_threshold if policy_name == "q20/q80" else None
                ),
                "training_q80": (
                    upper_threshold if policy_name == "q20/q80" else None
                ),
                "gross_simple_return": gross,
                "gross_return_bps": gross * 10_000.0,
                "active_trade": active,
                "source_backtest_experiment_digest": SOURCE_BACKTEST_EXPERIMENT_DIGEST,
                "lower_quantile": policy["lower_quantile"],
                "upper_quantile": policy["upper_quantile"],
                "training_lower_threshold": lower_threshold,
                "training_upper_threshold": upper_threshold,
                "threshold_source": policy["threshold_source"],
            }
        )
        result.append(base)
    return result


def as_selectivity_trade_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    ledger = pd.DataFrame(rows, columns=SELECTIVITY_LEDGER_COLUMNS)
    if ledger.empty:
        raise SignalSelectivityError("selectivity trade ledger is empty")
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


def approximate_active_stream_break_even_cost_bps(
    gross_returns: Sequence[float],
) -> float | None:
    """Approximate the per-active-trade cost that compounds a stream to flat."""

    values = np.asarray(gross_returns, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        return None
    if np.any(values <= -1.0) or bt.compound_returns(values) <= 0.0:
        return None
    safe_upper = float(np.min(1.0 + values) * 10_000.0 * 0.999999)
    if safe_upper <= 0.0:
        return None

    def final_equity(cost_bps: float) -> float:
        net = values - cost_bps / 10_000.0
        if np.any(net <= -1.0):
            return 0.0
        return float(np.prod(1.0 + net, dtype=np.float64))

    if final_equity(safe_upper) > 1.0:
        return None
    low, high = 0.0, safe_upper
    for _ in range(80):
        middle = (low + high) / 2.0
        if final_equity(middle) >= 1.0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def gross_trade_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    active = ledger.loc[ledger["active_trade"]].copy()
    gross = active["gross_simple_return"].to_numpy(dtype=np.float64)
    gross_bps = gross * 10_000.0
    active_count = int(len(active))
    scheduled_count = int(len(ledger))
    if scheduled_count <= 0:
        raise SignalSelectivityError("scheduled signal denominator is empty")
    return {
        "active_trade_count": active_count,
        "scheduled_signal_count": scheduled_count,
        "active_fraction": float(active_count / scheduled_count),
        "long_trade_count": int((active["direction"] == "LONG").sum()),
        "short_trade_count": int((active["direction"] == "SHORT").sum()),
        "average_gross_bps_per_active_trade": (
            float(np.mean(gross_bps)) if active_count else None
        ),
        "median_gross_bps_per_active_trade": (
            float(np.median(gross_bps)) if active_count else None
        ),
        "gross_profit_factor": bt.profit_factor(gross) if active_count else None,
        "gross_edge_sum_bps": float(np.sum(gross_bps)),
        "gross_edge_per_scheduled_signal_bps": float(np.sum(gross_bps) / scheduled_count),
    }


def calculate_economic_metrics(
    ledger: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int], pd.DataFrame]]:
    """Use the existing accounting implementation for fold and overall metrics."""

    fold_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, int], pd.DataFrame] = {}
    for (strategy_id, policy_name, fold_id), group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id"], sort=True
    ):
        for cost_bps in COST_SCENARIOS_BPS:
            events = bt.build_portfolio_events(group, cost_bps)
            metrics = bt.stream_metrics(group, events, cost_bps)
            fold_rows.append(
                {
                    "strategy_id": strategy_id,
                    "model_name": str(group["model_name"].iloc[0]),
                    "horizon_bars": int(group["horizon_bars"].iloc[0]),
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "cost_bps": int(cost_bps),
                    "fold_id": fold_id,
                    "test_start_utc": bt.canonical_utc(
                        group["fold_test_start_utc"].iloc[0]
                    ),
                    "test_end_exclusive_utc": bt.canonical_utc(
                        group["fold_test_end_exclusive_utc"].iloc[0]
                    ),
                    **gross_trade_metrics(group),
                    **metrics,
                }
            )

    fold_frame = pd.DataFrame(fold_rows)
    for (strategy_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        gross_metrics = gross_trade_metrics(group)
        for cost_bps in COST_SCENARIOS_BPS:
            events = bt.build_portfolio_events(group, cost_bps)
            curve = bt.enrich_equity_curve(events)
            curves[(str(strategy_id), str(policy_name), int(cost_bps))] = curve
            metrics = bt.stream_metrics(group, events, cost_bps)
            folds = fold_frame.loc[
                (fold_frame["strategy_id"] == strategy_id)
                & (fold_frame["policy_name"] == policy_name)
                & (fold_frame["cost_bps"] == cost_bps)
            ].sort_values("fold_id", kind="mergesort")
            if folds.empty:
                raise SignalSelectivityError("overall metrics have no fold rows")
            fold_returns = folds["net_cumulative_return"].to_numpy(dtype=np.float64)
            overall_rows.append(
                {
                    "strategy_id": strategy_id,
                    "model_name": str(group["model_name"].iloc[0]),
                    "horizon_bars": int(group["horizon_bars"].iloc[0]),
                    "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
                    "policy_name": policy_name,
                    "cost_bps": int(cost_bps),
                    **gross_metrics,
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


def build_policy_summary_rows(
    ledger: pd.DataFrame, overall_metrics: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(overall_metrics)
    result: list[dict[str, Any]] = []
    for (strategy_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=True
    ):
        gross = gross_trade_metrics(group)
        cost_rows = frame.loc[
            (frame["strategy_id"] == strategy_id)
            & (frame["policy_name"] == policy_name)
        ]
        by_cost = {int(row.cost_bps): row for row in cost_rows.itertuples(index=False)}
        if set(by_cost) != set(COST_SCENARIOS_BPS):
            raise SignalSelectivityError("policy summary lacks a required cost scenario")
        row: dict[str, Any] = {
            "strategy_id": strategy_id,
            "model_name": str(group["model_name"].iloc[0]),
            "horizon_bars": int(group["horizon_bars"].iloc[0]),
            "horizon_minutes": int(group["horizon_minutes"].iloc[0]),
            "policy_name": policy_name,
            **gross,
            "approximate_break_even_round_trip_cost_bps": (
                bt.approximate_break_even_cost_bps(group)
            ),
        }
        for cost_bps in COST_SCENARIOS_BPS:
            metrics = by_cost[cost_bps]
            suffix = f"{cost_bps}bps"
            row.update(
                {
                    f"overall_net_return_{suffix}": float(metrics.net_cumulative_return),
                    f"daily_sharpe_{suffix}": (
                        None if metrics.daily_sharpe is None else float(metrics.daily_sharpe)
                    ),
                    f"maximum_drawdown_{suffix}": float(metrics.maximum_drawdown),
                    f"positive_fold_percentage_{suffix}": float(
                        metrics.positive_fold_percentage
                    ),
                    f"median_fold_return_{suffix}": float(
                        metrics.median_fold_net_return
                    ),
                    f"worst_fold_return_{suffix}": float(
                        metrics.worst_fold_net_return
                    ),
                    f"btc_net_return_{suffix}": float(
                        metrics.btc_net_cumulative_return
                    ),
                    f"eth_net_return_{suffix}": float(
                        metrics.eth_net_cumulative_return
                    ),
                }
            )
        row.update(
            {
                "survives_2bps": row["overall_net_return_2bps"] > 0.0,
                "survives_5bps": row["overall_net_return_5bps"] > 0.0,
                "survives_10bps": row["overall_net_return_10bps"] > 0.0,
                "positive_majority_of_folds_at_5bps": (
                    row["positive_fold_percentage_5bps"] > 50.0
                ),
                "both_symbols_positive_at_5bps": (
                    row["btc_net_return_5bps"] > 0.0
                    and row["eth_net_return_5bps"] > 0.0
                ),
                "diagnostic_only": True,
                "production_pass_gate_defined": False,
            }
        )
        result.append(row)
    return sorted(
        result,
        key=lambda item: (
            next(
                index
                for index, strategy in enumerate(STRATEGIES)
                if strategy["strategy_id"] == item["strategy_id"]
            ),
            POLICY_ORDER.index(str(item["policy_name"])),
        ),
    )


def build_symbol_direction_diagnostics(
    ledger: pd.DataFrame,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    streams = ledger[["strategy_id", "policy_name"]].drop_duplicates()
    for stream in streams.itertuples(index=False):
        selected = ledger.loc[
            (ledger["strategy_id"] == stream.strategy_id)
            & (ledger["policy_name"] == stream.policy_name)
        ]
        for symbol in wf.SYMBOLS:
            for direction in ("LONG", "SHORT"):
                group = selected.loc[
                    (selected["symbol"] == symbol)
                    & (selected["direction"] == direction)
                    & selected["active_trade"]
                ]
                returns = group["gross_simple_return"].to_numpy(dtype=np.float64)
                count = int(len(group))
                result.append(
                    {
                        "strategy_id": str(stream.strategy_id),
                        "policy_name": str(stream.policy_name),
                        "symbol": symbol,
                        "direction": direction,
                        "active_trade_count": count,
                        "win_rate": float(np.mean(returns > 0.0)) if count else None,
                        "mean_gross_trade_bps": (
                            float(np.mean(returns) * 10_000.0) if count else None
                        ),
                        "gross_profit_factor": (
                            bt.profit_factor(returns) if count else None
                        ),
                        "approximate_break_even_round_trip_cost_bps": (
                            approximate_active_stream_break_even_cost_bps(returns)
                        ),
                        "diagnostic_only": True,
                        "policy_optimized_by_symbol": False,
                    }
                )
    return sorted(
        result,
        key=lambda row: (
            next(
                index
                for index, strategy in enumerate(STRATEGIES)
                if strategy["strategy_id"] == row["strategy_id"]
            ),
            POLICY_ORDER.index(str(row["policy_name"])),
            wf.SYMBOLS.index(str(row["symbol"])),
            ("LONG", "SHORT").index(str(row["direction"])),
        ),
    )


def build_selectivity_comparisons(
    policy_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(policy_summaries)
    result: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        strategy_id = str(strategy["strategy_id"])
        rows = frame.loc[frame["strategy_id"] == strategy_id].set_index("policy_name")
        previous_name: str | None = None
        for policy_name in SELECTIVE_POLICY_ORDER:
            current = rows.loc[policy_name]
            comparison: dict[str, Any] = {
                "strategy_id": strategy_id,
                "policy_name": policy_name,
                "active_trade_count": int(current["active_trade_count"]),
                "active_fraction": float(current["active_fraction"]),
                "average_gross_bps_per_active_trade": current[
                    "average_gross_bps_per_active_trade"
                ],
                "gross_edge_per_scheduled_signal_bps": float(
                    current["gross_edge_per_scheduled_signal_bps"]
                ),
                "looser_policy_name": previous_name,
                "edge_per_trade_increased_vs_looser": None,
                "edge_increase_outpaced_opportunity_reduction_vs_looser": None,
                "active_trade_retention_vs_looser": None,
                "active_trade_reduction_fraction_vs_looser": None,
                "average_gross_edge_change_bps_vs_looser": None,
                "gross_edge_per_scheduled_signal_change_bps_vs_looser": None,
            }
            if previous_name is not None:
                previous = rows.loc[previous_name]
                previous_count = int(previous["active_trade_count"])
                current_count = int(current["active_trade_count"])
                previous_average = float(previous["average_gross_bps_per_active_trade"])
                current_average = float(current["average_gross_bps_per_active_trade"])
                previous_frequency_edge = float(
                    previous["gross_edge_per_scheduled_signal_bps"]
                )
                current_frequency_edge = float(
                    current["gross_edge_per_scheduled_signal_bps"]
                )
                retention = current_count / previous_count if previous_count else 0.0
                comparison.update(
                    {
                        "active_trade_retention_vs_looser": float(retention),
                        "active_trade_reduction_fraction_vs_looser": float(
                            1.0 - retention
                        ),
                        "average_gross_edge_change_bps_vs_looser": float(
                            current_average - previous_average
                        ),
                        "gross_edge_per_scheduled_signal_change_bps_vs_looser": float(
                            current_frequency_edge - previous_frequency_edge
                        ),
                        "edge_per_trade_increased_vs_looser": (
                            current_average > previous_average
                        ),
                        "edge_increase_outpaced_opportunity_reduction_vs_looser": (
                            current_average > previous_average
                            and current_frequency_edge > previous_frequency_edge
                        ),
                    }
                )
            result.append(comparison)
            previous_name = policy_name
    return result


def validate_results(
    *,
    ledger: pd.DataFrame,
    fold_metrics: Sequence[Mapping[str, Any]],
    overall_metrics: Sequence[Mapping[str, Any]],
    policy_summaries: Sequence[Mapping[str, Any]],
    symbol_direction_diagnostics: Sequence[Mapping[str, Any]],
    curves: Mapping[tuple[str, str, int], pd.DataFrame],
) -> dict[str, bool]:
    checks = bt.validate_trade_ledger(ledger)
    fold_frame = pd.DataFrame(fold_metrics)
    overall_frame = pd.DataFrame(overall_metrics)
    summary_frame = pd.DataFrame(policy_summaries)
    diagnostic_frame = pd.DataFrame(symbol_direction_diagnostics)

    for (strategy_id, fold_id, symbol), group in ledger.groupby(
        ["strategy_id", "fold_id", "symbol"], sort=False
    ):
        active_sets = {
            name: set(
                group.loc[
                    (group["policy_name"] == name) & group["active_trade"],
                    "signal_timestamp_utc",
                ]
            )
            for name in SELECTIVE_POLICY_ORDER
        }
        for looser, stricter in zip(SELECTIVE_POLICY_ORDER, SELECTIVE_POLICY_ORDER[1:]):
            if not active_sets[stricter] <= active_sets[looser]:
                raise SignalSelectivityError(
                    f"policy nesting failed for {strategy_id}/{fold_id}/{symbol}"
                )

    for (strategy_id, policy_name), group in ledger.groupby(
        ["strategy_id", "policy_name"], sort=False
    ):
        expected_active = int(group["active_trade"].sum())
        summary_match = summary_frame.loc[
            (summary_frame["strategy_id"] == strategy_id)
            & (summary_frame["policy_name"] == policy_name)
        ]
        if len(summary_match) != 1:
            raise SignalSelectivityError("policy summary does not uniquely reconcile")
        summary = summary_match.iloc[0]
        if int(summary["active_trade_count"]) != expected_active:
            raise SignalSelectivityError("active trade count does not reconcile")
        if int(summary["long_trade_count"] + summary["short_trade_count"]) != expected_active:
            raise SignalSelectivityError("long/short totals do not reconcile")
        diagnostics = diagnostic_frame.loc[
            (diagnostic_frame["strategy_id"] == strategy_id)
            & (diagnostic_frame["policy_name"] == policy_name)
        ]
        if len(diagnostics) != 4:
            raise SignalSelectivityError("BTC/ETH long/short diagnostic grid is incomplete")
        if int(diagnostics["active_trade_count"].sum()) != expected_active:
            raise SignalSelectivityError("diagnostic active trades do not reconcile")
        for symbol in wf.SYMBOLS:
            expected_symbol = int(
                group.loc[
                    (group["symbol"] == symbol) & group["active_trade"]
                ].shape[0]
            )
            reported_symbol = int(
                diagnostics.loc[diagnostics["symbol"] == symbol, "active_trade_count"].sum()
            )
            if expected_symbol != reported_symbol:
                raise SignalSelectivityError("BTC/ETH diagnostics do not remain separate")

        cost_rows = overall_frame.loc[
            (overall_frame["strategy_id"] == strategy_id)
            & (overall_frame["policy_name"] == policy_name)
        ].set_index("cost_bps")
        if set(cost_rows.index.astype(int)) != set(COST_SCENARIOS_BPS):
            raise SignalSelectivityError("cost metric grid is incomplete")
        zero = cost_rows.loc[0]
        if not math.isclose(
            float(zero["net_cumulative_return"]),
            float(zero["gross_cumulative_return"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise SignalSelectivityError("zero-bps net return differs from gross")
        for lower_cost, higher_cost in zip(
            COST_SCENARIOS_BPS, COST_SCENARIOS_BPS[1:]
        ):
            lower_curve = curves[(str(strategy_id), str(policy_name), lower_cost)]
            higher_curve = curves[(str(strategy_id), str(policy_name), higher_cost)]
            if np.any(
                higher_curve["net_equity"].to_numpy(dtype=np.float64)
                > lower_curve["net_equity"].to_numpy(dtype=np.float64) + 1e-12
            ):
                raise SignalSelectivityError("higher cost improved same-stream equity")

    for cost_bps in COST_SCENARIOS_BPS:
        active = ledger["active_trade"].to_numpy(dtype=bool)
        gross = ledger["gross_simple_return"].to_numpy(dtype=np.float64)
        net = np.asarray(
            [
                bt.apply_round_trip_cost(
                    value,
                    active_trade=bool(is_active),
                    round_trip_cost_bps=cost_bps,
                )
                for value, is_active in zip(gross, active)
            ],
            dtype=np.float64,
        )
        if np.any(net[~active] != 0.0):
            raise SignalSelectivityError("flat signal paid transaction cost")
        if cost_bps == 0 and not np.array_equal(net, gross):
            raise SignalSelectivityError("zero-bps trade net differs from gross")

    for keys, group in ledger.groupby(
        ["strategy_id", "policy_name", "fold_id"], sort=False
    ):
        expected_active = int(group["active_trade"].sum())
        matches = fold_frame.loc[
            (fold_frame["strategy_id"] == keys[0])
            & (fold_frame["policy_name"] == keys[1])
            & (fold_frame["fold_id"] == keys[2])
        ]
        if len(matches) != len(COST_SCENARIOS_BPS):
            raise SignalSelectivityError("fold/cost metric rows are incomplete")
        if not (matches["active_trade_count"] == expected_active).all():
            raise SignalSelectivityError("fold active counts do not reconcile")

    return {
        **checks,
        "thresholds_derived_from_training_scores_only": True,
        "test_scores_do_not_set_thresholds": True,
        "test_returns_do_not_set_thresholds": True,
        "stricter_policies_are_activity_subsets": True,
        "flat_signals_pay_no_transaction_cost": True,
        "accounting_reused_from_validated_backtest": True,
        "zero_bps_net_equals_gross": True,
        "higher_cost_never_improves_same_stream_equity": True,
        "btc_eth_results_remain_separate": True,
        "long_short_diagnostics_reconcile": True,
        "active_trade_counts_reconcile": True,
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
        "protected_source_hashes": dict(protected_hashes),
        "feature_contract_digest": feature_contract_digest,
        "strategies": strategy_contract(),
        "policies": list(POLICIES),
        "selective_policy_order_loose_to_strict": list(SELECTIVE_POLICY_ORDER),
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "cost_contract": COST_CONTRACT,
        "decision_diagnostic_contract": {
            "production_pass_gate_defined": False,
            "survives_cost_definition": "overall net cumulative return > 0",
            "positive_majority_of_folds_at_5bps": (
                "positive fold percentage at 5 bps > 50%"
            ),
            "both_symbols_positive_at_5bps": (
                "BTC and ETH standalone sleeve cumulative returns both > 0 at 5 bps"
            ),
            "edge_frequency_definition": (
                "gross edge per scheduled signal = active fraction multiplied by "
                "average gross bps per active trade"
            ),
            "outpaces_definition": (
                "a stricter policy raises average gross bps per active trade and also "
                "raises gross edge per scheduled signal versus the adjacent looser policy"
            ),
        },
    }


def ensure_output_root(path: Path | str) -> Path:
    requested = Path(path).resolve()
    allowed = SELECTIVITY_ROOT.resolve()
    if requested != allowed:
        raise SignalSelectivityError(
            f"selectivity output root must be exactly {allowed}; received {requested}"
        )
    return requested


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def build_markdown_report(
    *,
    experiment_id: str,
    experiment_digest: str,
    policy_summaries: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> str:
    summaries = list(policy_summaries)
    any_2 = any(bool(row["survives_2bps"]) for row in summaries)
    any_5 = any(bool(row["survives_5bps"]) for row in summaries)
    any_10 = any(bool(row["survives_10bps"]) for row in summaries)
    any_both_5 = any(bool(row["both_symbols_positive_at_5bps"]) for row in summaries)
    q05_survives_5 = any(
        row["policy_name"] == "q05/q95" and bool(row["survives_5bps"])
        for row in summaries
    )
    transition_rows = [
        row
        for row in comparisons
        if row["looser_policy_name"] is not None
    ]
    edge_increase_count = sum(
        bool(row["edge_per_trade_increased_vs_looser"]) for row in transition_rows
    )
    outpaced_count = sum(
        bool(row["edge_increase_outpaced_opportunity_reduction_vs_looser"])
        for row in transition_rows
    )
    recommendation = (
        "At least one q05/q95 stream survives 5 bps, so selectivity remains a useful "
        "diagnostic branch; no more-extreme thresholds were evaluated or authorized."
        if q05_survives_5
        else "No q05/q95 stream survives 5 bps. Stop tightening thresholds and move "
        "the next research branch to the predictive objective or feature design."
    )

    lines = [
        "# Signal Selectivity and Turnover Research",
        "",
        "## Technical summary",
        "",
        f"- Experiment `{experiment_id}` (`{experiment_digest}`) is exposed historical "
        "research only; no production pass gate exists.",
        f"- Any strategy survives 2/5/10 bps: **{any_2}/{any_5}/{any_10}**. "
        f"Any strategy has both BTC and ETH positive at 5 bps: **{any_both_5}**.",
        f"- Across {len(transition_rows)} adjacent tightening steps, gross edge per active "
        f"trade increased in {edge_increase_count}; the increase also offset lost "
        f"frequency in {outpaced_count}, using gross edge per scheduled signal.",
        f"- **Recommendation:** {recommendation}",
        "",
        "## Exact strategy-policy results",
        "",
        "Active % uses all scheduled BTC/ETH sleeve signals as the denominator. Returns, "
        "Sharpe, drawdown, and fold diagnostics use the frozen 50/50 sleeve accounting.",
        "",
        "| Strategy | Policy | Active | Active % | Avg gross bps/trade | Median gross bps/trade | Gross PF | Break-even bps | Return 0 bps | Return 2 bps | Return 5 bps | Return 10 bps | Sharpe 5 bps | Max DD 5 bps | Positive folds 5 bps | Median fold 5 bps | Worst fold 5 bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {strategy_id} | {policy_name} | {active_trade_count} | {active_pct} | "
            "{avg} | {median} | {pf} | {be} | {r0} | {r2} | {r5} | {r10} | "
            "{sharpe} | {dd} | {fold_pct} | {fold_median} | {fold_worst} |".format(
                **row,
                active_pct=_format_percent(row["active_fraction"]),
                avg=_format_number(row["average_gross_bps_per_active_trade"]),
                median=_format_number(row["median_gross_bps_per_active_trade"]),
                pf=_format_number(row["gross_profit_factor"]),
                be=_format_number(
                    row["approximate_break_even_round_trip_cost_bps"]
                ),
                r0=_format_percent(row["overall_net_return_0bps"]),
                r2=_format_percent(row["overall_net_return_2bps"]),
                r5=_format_percent(row["overall_net_return_5bps"]),
                r10=_format_percent(row["overall_net_return_10bps"]),
                sharpe=_format_number(row["daily_sharpe_5bps"]),
                dd=_format_percent(row["maximum_drawdown_5bps"]),
                fold_pct=f"{float(row['positive_fold_percentage_5bps']):.2f}%",
                fold_median=_format_percent(row["median_fold_return_5bps"]),
                fold_worst=_format_percent(row["worst_fold_return_5bps"]),
            )
        )

    lines.extend(
        [
            "",
            "## Edge per trade versus opportunity frequency",
            "",
            "The compensation diagnostic is arithmetic gross edge per scheduled signal: "
            "active fraction × average gross bps per active trade. A stricter policy only "
            "outpaces its opportunity reduction when both per-trade edge and this "
            "frequency-adjusted measure increase.",
            "",
            "| Strategy | Policy | Looser policy | Active retention | Avg gross bps/trade | Gross bps/scheduled signal | Edge/trade increased | Outpaced opportunity reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {strategy_id} | {policy_name} | {looser} | {retention} | {average} | "
            "{frequency_edge} | {increased} | {outpaced} |".format(
                **row,
                looser=row["looser_policy_name"] or "baseline",
                retention=(
                    "n/a"
                    if row["active_trade_retention_vs_looser"] is None
                    else _format_percent(row["active_trade_retention_vs_looser"])
                ),
                average=_format_number(row["average_gross_bps_per_active_trade"]),
                frequency_edge=_format_number(
                    row["gross_edge_per_scheduled_signal_bps"]
                ),
                increased=(
                    "n/a"
                    if row["edge_per_trade_increased_vs_looser"] is None
                    else str(bool(row["edge_per_trade_increased_vs_looser"]))
                ),
                outpaced=(
                    "n/a"
                    if row[
                        "edge_increase_outpaced_opportunity_reduction_vs_looser"
                    ]
                    is None
                    else str(
                        bool(
                            row[
                                "edge_increase_outpaced_opportunity_reduction_vs_looser"
                            ]
                        )
                    )
                ),
            )
        )

    diagnostic_frame = pd.DataFrame(diagnostics)
    lines.extend(
        [
            "",
            "## BTC/ETH long-short diagnostics",
            "",
            "Each cell is `trades / win rate / mean gross bps / gross PF / break-even bps`. "
            "Policies remain shared across symbols; these cuts are diagnostic only.",
            "",
            "| Strategy | Policy | BTC long | BTC short | ETH long | ETH short |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        selected = diagnostic_frame.loc[
            (diagnostic_frame["strategy_id"] == summary["strategy_id"])
            & (diagnostic_frame["policy_name"] == summary["policy_name"])
        ]

        def cell(symbol: str, direction: str) -> str:
            match = selected.loc[
                (selected["symbol"] == symbol)
                & (selected["direction"] == direction)
            ].iloc[0]
            return "{} / {} / {} / {} / {}".format(
                int(match["active_trade_count"]),
                _format_percent(match["win_rate"]),
                _format_number(match["mean_gross_trade_bps"]),
                _format_number(match["gross_profit_factor"]),
                _format_number(match["approximate_break_even_round_trip_cost_bps"]),
            )

        lines.append(
            f"| {summary['strategy_id']} | {summary['policy_name']} | "
            f"{cell('BTCUSDT', 'LONG')} | {cell('BTCUSDT', 'SHORT')} | "
            f"{cell('ETHUSDT', 'LONG')} | {cell('ETHUSDT', 'SHORT')} |"
        )

    lines.extend(
        [
            "",
            "## Scope, model specification, and metric definitions",
            "",
            "The analysis is limited to the frozen hist-gradient-boosting 30-minute, "
            "logistic-regression 30-minute, and logistic-regression 2-hour configurations. "
            "All seven outer folds are identical to the source walk-forward experiment. "
            "For every fold, the model is fit on outer training rows, training scores set "
            "the frozen quantiles, and thresholds are then applied to test scores. Test "
            "scores and returns never set thresholds.",
            "",
            "A trade is active only for LONG or SHORT. Gross profit factor divides summed "
            "positive active-trade returns by the absolute sum of negative active-trade "
            "returns. Break-even cost is the approximate synthetic round-trip bps that "
            "compounds final equity to 1.0. Synthetic costs are stress assumptions, not "
            "claimed Hyperliquid fees.",
            "",
            "## Execution and robustness checks remained unchanged",
            "",
            "Signals are available after completed bar t, enter at the next 5-minute open, "
            "and exit at the horizon bar close. Same-symbol trades do not overlap, folds "
            "never carry positions, BTC and ETH keep independent 50% sleeves, leverage is "
            "1×, and flat sleeves are not reallocated. The task reuses the validated "
            "economic backtest accounting functions. Zero-cost net equals gross and higher "
            "cost never improves the same-stream equity path.",
            "",
            "## Limitations and next research branch",
            "",
            "All periods are exposed historical research, not a pristine holdout. Funding, "
            "order-book spread, market impact, latency beyond next-bar execution, and "
            "liquidation remain unmodeled. The diagnostics are observations and provide no "
            "deployment authorization.",
            "",
            f"**Next branch:** {recommendation}",
            "",
            "## Further questions",
            "",
            "The next objective/feature branch should ask whether the predictive target can "
            "produce a materially larger conditional return distribution without relying "
            "on further threshold tightening, and should reserve untouched data for later "
            "confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_selectivity_research(
    *,
    source_walkforward_directory: Path | str = SOURCE_WALKFORWARD_DIRECTORY,
    source_backtest_directory: Path | str = SOURCE_BACKTEST_DIRECTORY,
    output_root: Path | str = SELECTIVITY_ROOT,
) -> dict[str, Any]:
    source_walkforward_summary, source_walkforward_manifest, _ = (
        bt.validate_walkforward_source(source_walkforward_directory)
    )
    source_backtest_summary, source_backtest_manifest, _ = validate_backtest_source(
        source_backtest_directory
    )
    earlier, later = bt.source_dataset_paths(source_walkforward_summary)
    expected_raw = source_walkforward_summary["raw_source_digests"]
    protected_before = protected_source_hashes(
        earlier=earlier,
        later=later,
        expected_raw=expected_raw,
        source_walkforward_directory=source_walkforward_directory,
        source_backtest_directory=source_backtest_directory,
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
        raise SignalSelectivityError("BTC/ETH raw grids differ")
    feature_contract = wf._feature_contract()
    if feature_contract["feature_contract_digest"] != source_walkforward_summary[
        "feature_contract"
    ]["feature_contract_digest"]:
        raise SignalSelectivityError("feature contract differs from source experiment")
    features_by_symbol = {
        symbol: wf.build_research_features(raw_by_symbol[symbol], symbol=symbol)[0]
        for symbol in wf.SYMBOLS
    }

    contract = experiment_contract(
        protected_hashes=protected_before,
        feature_contract_digest=feature_contract["feature_contract_digest"],
    )
    experiment_digest = bt.json_digest(contract)
    experiment_id = f"selectivity_{experiment_digest[:16]}"
    output = ensure_output_root(output_root)
    final_directory = output / experiment_id
    staging_directory = output / f".{experiment_id}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise SignalSelectivityError(
            f"selectivity research output already exists: {experiment_id}"
        )

    ordered_features = wf.canonical_feature_columns(True)
    trade_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    nesting_rows: list[dict[str, Any]] = []
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
            raise SignalSelectivityError("real selectivity run requires seven folds")
        for fold in folds:
            train, test = wf.select_fold_rows(target_rows, fold)
            X_train = train.loc[:, ordered_features].to_numpy(dtype=np.float64)
            y_train = train["target"].to_numpy(dtype=np.int8)
            X_test = test.loc[:, ordered_features].to_numpy(dtype=np.float64)
            test_scores, estimator = wf.fit_model_scores(
                str(strategy["model_name"]), X_train, y_train, X_test
            )
            train_scores = bt._positive_scores(estimator, X_train)
            thresholds_by_policy = {
                policy_name: derive_training_thresholds(train_scores, policy_name)
                for policy_name in POLICY_ORDER
            }
            scored_test = test.loc[:, ["timestamp", "symbol"]].copy()
            scored_test["score"] = test_scores
            nesting_counts = validate_stricter_policy_nesting(
                scored_test["score"].to_numpy(dtype=np.float64), thresholds_by_policy
            )
            for policy_name in POLICY_ORDER:
                lower, upper = thresholds_by_policy[policy_name]
                policy = policy_contract(policy_name)
                threshold_rows.append(
                    {
                        "strategy_id": strategy["strategy_id"],
                        "fold_id": fold.fold_id,
                        "policy_name": policy_name,
                        "training_score_count": int(len(train_scores)),
                        "lower_quantile": policy["lower_quantile"],
                        "upper_quantile": policy["upper_quantile"],
                        "training_lower_threshold": lower,
                        "training_upper_threshold": upper,
                        "threshold_source": policy["threshold_source"],
                        "test_scores_consulted": False,
                        "test_returns_consulted": False,
                    }
                )
                for symbol in wf.SYMBOLS:
                    symbol_scores = scored_test.loc[
                        scored_test["symbol"] == symbol, ["timestamp", "score"]
                    ]
                    trade_rows.extend(
                        build_selectivity_trade_rows(
                            experiment_id=experiment_id,
                            strategy=strategy,
                            policy_name=policy_name,
                            fold=fold,
                            symbol=symbol,
                            scored_test=symbol_scores,
                            raw=raw_by_symbol[symbol],
                            lower_threshold=lower,
                            upper_threshold=upper,
                        )
                    )
            nesting_rows.append(
                {
                    "strategy_id": strategy["strategy_id"],
                    "fold_id": fold.fold_id,
                    **{
                        f"active_test_scores_{name.replace('/', '_')}": count
                        for name, count in nesting_counts.items()
                    },
                    "stricter_policies_cannot_add_active_scores": True,
                }
            )

    ledger = as_selectivity_trade_frame(trade_rows)
    fold_metrics, overall_metrics, curves = calculate_economic_metrics(ledger)
    policy_summaries = build_policy_summary_rows(ledger, overall_metrics)
    diagnostics = build_symbol_direction_diagnostics(ledger)
    comparisons = build_selectivity_comparisons(policy_summaries)
    invariant_checks = validate_results(
        ledger=ledger,
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
        policy_summaries=policy_summaries,
        symbol_direction_diagnostics=diagnostics,
        curves=curves,
    )

    protected_after = protected_source_hashes(
        earlier=earlier,
        later=later,
        expected_raw=expected_raw,
        source_walkforward_directory=source_walkforward_directory,
        source_backtest_directory=source_backtest_directory,
    )
    if protected_after != protected_before:
        raise SignalSelectivityError("a protected source changed during the run")
    invariant_checks.update(
        {
            "features_py_hash_unchanged": True,
            "frozen_raw_source_hashes_unchanged": True,
            "source_walkforward_evidence_hash_unchanged": True,
            "source_backtest_evidence_hash_unchanged": True,
        }
    )

    q05_survives_5bps = any(
        row["policy_name"] == "q05/q95" and row["survives_5bps"]
        for row in policy_summaries
    )
    any_survives = {
        str(cost): any(
            bool(row[f"survives_{cost}bps"]) for row in policy_summaries
        )
        for cost in (2, 5, 10)
    }
    any_both_symbols_positive_5bps = any(
        bool(row["both_symbols_positive_at_5bps"]) for row in policy_summaries
    )
    transition_rows = [
        row for row in comparisons if row["looser_policy_name"] is not None
    ]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "experiment_digest": experiment_digest,
        "research_only": True,
        "historical_periods_pristine_holdout": False,
        "production_candidate": False,
        "promotion_allowed": False,
        "live_execution_allowed": False,
        "production_pass_gate_defined": False,
        "source_walkforward": {
            "experiment_id": SOURCE_WALKFORWARD_EXPERIMENT_ID,
            "experiment_digest": SOURCE_WALKFORWARD_EXPERIMENT_DIGEST,
            "manifest_digest_contract": source_walkforward_manifest.get(
                "experiment_digest"
            ),
            "directory_digest": protected_before[
                "walkforward_evidence_directory"
            ],
        },
        "source_backtest": {
            "experiment_id": SOURCE_BACKTEST_EXPERIMENT_ID,
            "experiment_digest": SOURCE_BACKTEST_EXPERIMENT_DIGEST,
            "manifest_digest_contract": source_backtest_manifest.get(
                "experiment_digest"
            ),
            "directory_digest": protected_before["backtest_evidence_directory"],
            "economic_conclusion_reused_as_context_only": source_backtest_summary.get(
                "production_acceptance_threshold_defined"
            )
            is False,
        },
        "protected_source_hashes_before": protected_before,
        "protected_source_hashes_after": protected_after,
        "protected_source_hashes_unchanged": True,
        "feature_contract_digest": feature_contract["feature_contract_digest"],
        "strategies": strategy_contract(),
        "policies": list(POLICIES),
        "policy_thresholds_by_fold": threshold_rows,
        "policy_nesting_by_fold": nesting_rows,
        "cost_contract": COST_CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "portfolio_contract": PORTFOLIO_CONTRACT,
        "execution_limitations": LIMITATIONS,
        "fold_count": 7,
        "trade_ledger_row_count": int(len(ledger)),
        "policy_summary": policy_summaries,
        "overall_metrics_by_cost": overall_metrics,
        "fold_metrics_by_cost": fold_metrics,
        "symbol_direction_diagnostics": diagnostics,
        "gross_edge_per_trade_versus_trade_frequency": comparisons,
        "research_diagnostics": {
            "any_strategy_survives_2bps": any_survives["2"],
            "any_strategy_survives_5bps": any_survives["5"],
            "any_strategy_survives_10bps": any_survives["10"],
            "any_strategy_has_both_symbols_positive_at_5bps": (
                any_both_symbols_positive_5bps
            ),
            "q05_q95_survives_5bps": q05_survives_5bps,
            "adjacent_tightening_steps": len(transition_rows),
            "steps_where_edge_per_trade_increased": sum(
                bool(row["edge_per_trade_increased_vs_looser"])
                for row in transition_rows
            ),
            "steps_where_edge_increase_outpaced_opportunity_reduction": sum(
                bool(
                    row[
                        "edge_increase_outpaced_opportunity_reduction_vs_looser"
                    ]
                )
                for row in transition_rows
            ),
            "continue_threshold_tightening": False,
            "automatic_more_extreme_quantiles_allowed": False,
            "next_research_branch": (
                "evaluate a changed predictive objective or feature design"
                if not q05_survives_5bps
                else "validate the observed selective edge without automatically adding "
                "more-extreme quantiles"
            ),
            "observations_are_deployment_authorization": False,
        },
        "accounting_and_selectivity_invariants": invariant_checks,
        "all_invariants_reconciled": all(invariant_checks.values()),
        "historical_research_exposure_warning": (
            "All periods are exposed historical research and are not a pristine holdout."
        ),
        "visual_omission_reason": (
            "The requested decision surface is an exact multi-metric strategy-policy "
            "audit table; a chart would obscure the cost and fold fields."
        ),
        "safety_contract": {
            "candidate_training_performed": False,
            "candidate_models_or_artifacts_accessed_or_written": False,
            "validation_or_internal_test_ledger_accessed_or_written": False,
            "runtime_or_live_execution_accessed_or_modified": False,
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
    overall_path = staging_directory / "overall_metrics.csv"
    policy_path = staging_directory / "policy_summary.csv"
    threshold_path = staging_directory / "policy_thresholds.csv"
    nesting_path = staging_directory / "policy_nesting.csv"
    diagnostic_path = staging_directory / "symbol_direction_diagnostics.csv"
    comparison_path = staging_directory / "selectivity_comparisons.csv"
    report_path = staging_directory / "report.md"
    manifest_path = staging_directory / "experiment_manifest.json"

    bt._write_json(summary_path, summary)
    ledger_output = ledger.copy()
    for column in ledger_output.columns:
        if pd.api.types.is_datetime64_any_dtype(ledger_output[column]):
            ledger_output[column] = ledger_output[column].map(bt.canonical_utc)
    bt._write_csv(
        ledger_path, ledger_output.to_dict("records"), SELECTIVITY_LEDGER_COLUMNS
    )
    bt._write_csv(fold_path, fold_metrics)
    bt._write_csv(overall_path, overall_metrics)
    bt._write_csv(policy_path, policy_summaries)
    bt._write_csv(threshold_path, threshold_rows)
    bt._write_csv(nesting_path, nesting_rows)
    bt._write_csv(diagnostic_path, diagnostics)
    bt._write_csv(comparison_path, comparisons)
    report_path.write_text(
        build_markdown_report(
            experiment_id=experiment_id,
            experiment_digest=experiment_digest,
            policy_summaries=policy_summaries,
            comparisons=comparisons,
            diagnostics=diagnostics,
        ),
        encoding="utf-8",
        newline="\n",
    )

    output_rows: dict[str, int | None] = {
        "summary.json": None,
        "trade_ledger.csv": len(ledger_output),
        "fold_metrics.csv": len(fold_metrics),
        "overall_metrics.csv": len(overall_metrics),
        "policy_summary.csv": len(policy_summaries),
        "policy_thresholds.csv": len(threshold_rows),
        "policy_nesting.csv": len(nesting_rows),
        "symbol_direction_diagnostics.csv": len(diagnostics),
        "selectivity_comparisons.csv": len(comparisons),
        "report.md": None,
    }
    outputs = {
        name: {
            "sha256": bt.file_sha256(staging_directory / name),
            "row_count": row_count,
        }
        for name, row_count in output_rows.items()
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
    bt._write_json(manifest_path, manifest)
    staging_directory.rename(final_directory)
    summary["output_directory"] = str(final_directory)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline signal-selectivity and turnover-reduction research."
    )
    parser.add_argument(
        "--source-walkforward", type=Path, default=SOURCE_WALKFORWARD_DIRECTORY
    )
    parser.add_argument(
        "--source-backtest", type=Path, default=SOURCE_BACKTEST_DIRECTORY
    )
    parser.add_argument("--output-root", type=Path, default=SELECTIVITY_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_selectivity_research(
            source_walkforward_directory=args.source_walkforward,
            source_backtest_directory=args.source_backtest,
            output_root=args.output_root,
        )
    except (
        SignalSelectivityError,
        bt.SignalBacktestError,
        wf.SignalResearchError,
        OSError,
        ValueError,
    ) as exc:
        print(f"model signal selectivity research failed closed: {exc}", file=sys.stderr)
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
