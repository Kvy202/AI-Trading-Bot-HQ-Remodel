"""Phase 20 deterministic, offline counterfactual replay engine.

The engine consumes recorded signal ticks only.  It never imports or calls an
executor entry point, initializes an exchange, fetches market data, or loads a
model.  Counterfactual outputs remain separate from Phase 17/18 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from tools.evidence_manifest import (
        DEFAULT_OVERRIDES_PATH as DEFAULT_EVIDENCE_OVERRIDES,
        build_evidence_manifest,
        evidence_manifest_digest,
    )
    from tools.replay_bundle import (
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_LOGS_DIR,
        DEFAULT_REPORTS_DIR,
        ReplayBundleError,
        calculate_coverage,
        canonical_row_digest,
        parse_timestamp,
        resolve_historical_sources,
    )
    from tools.replay_contract import (
        DEFAULT_OVERRIDES_PATH as DEFAULT_CONTRACT_OVERRIDES,
        PARITY_GRADE_STATUSES,
        ReplayContractError,
        resolve_replay_contract,
    )
except ModuleNotFoundError:
    from evidence_manifest import (  # type: ignore
        DEFAULT_OVERRIDES_PATH as DEFAULT_EVIDENCE_OVERRIDES,
        build_evidence_manifest,
        evidence_manifest_digest,
    )
    from replay_bundle import (  # type: ignore
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_LOGS_DIR,
        DEFAULT_REPORTS_DIR,
        ReplayBundleError,
        calculate_coverage,
        canonical_row_digest,
        parse_timestamp,
        resolve_historical_sources,
    )
    from replay_contract import (  # type: ignore
        DEFAULT_OVERRIDES_PATH as DEFAULT_CONTRACT_OVERRIDES,
        PARITY_GRADE_STATUSES,
        ReplayContractError,
        resolve_replay_contract,
    )

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "counterfactual_replay.json"
SCHEMA_VERSION = 1
POLICIES = {"baseline", "xgboost_confirm_only", "xgboost_reject_only"}
ENTRY_ACTIONS = {"BUY", "SELL_SHORT"}
CLOSE_ACTIONS = {"SELL", "BUY_TO_COVER"}
REASON_PREFIXES = ("EXIT_TP", "EXIT_SL", "EXIT_TIME", "FLIP_CLOSE")


class CounterfactualReplayError(ValueError):
    """Raised for malformed replay inputs or internal deterministic failures."""


def _code_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SignalEvent:
    timestamp: datetime
    source_order: int
    ts: str
    symbol: str
    price: float
    p_meta: float
    rv_mean: float
    allow: int
    thr: float
    mode: str
    kinds_used: Optional[str]
    signal_id: str


@dataclass
class ReplayPosition:
    side: str
    qty: float
    avg: float
    entry_timestamp: datetime
    entry_signal_id: str
    entry_slippage_cost: float = 0.0


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def mode_value(p_meta: float, mode: str) -> float:
    return abs(p_meta) if (mode or "abs").lower() == "abs" else p_meta


def threshold_pass(
    signal: SignalEvent,
    exec_thr: float,
    exec_mode: str,
    respect_writer_thr: bool,
) -> tuple[bool, str]:
    if signal.allow != 1:
        return False, "allow=0"
    if signal.kinds_used is not None and not signal.kinds_used:
        return False, "empty_kinds_used"
    value = mode_value(signal.p_meta, exec_mode)
    effective = max(signal.thr, exec_thr) if respect_writer_thr else exec_thr
    if value < effective:
        return False, "below_thr"
    return True, ""


def side_allowed(p_meta: float, sides: str) -> bool:
    value = (sides or "both").lower()
    return value == "both" or (value == "long_only" and p_meta >= 0) or (
        value == "short_only" and p_meta < 0
    )


def apply_slippage(price: float, action: str, slippage_bps: float) -> float:
    if price <= 0 or not slippage_bps:
        return price
    rate = float(slippage_bps) / 1e4
    normalized = (action or "").upper()
    if normalized in {"BUY", "BUY_TO_COVER"}:
        return price * (1.0 + rate)
    if normalized in {"SELL", "SELL_SHORT"}:
        return price * (1.0 - rate)
    return price


def fee_cost(notional: float, fee_bps: float) -> float:
    return abs(notional) * (float(fee_bps) / 1e4)


def _gross_pnl(position: Any, price: float) -> float:
    return (
        (price - position.avg) * position.qty
        if position.side == "long"
        else (position.avg - price) * position.qty
    )


def net_pnl_on_close(
    position: Any,
    exit_mid: float,
    action_close: str,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[float, float]:
    exit_fill = apply_slippage(exit_mid, action_close, slippage_bps)
    fees = fee_cost(position.avg * position.qty, fee_bps) + fee_cost(
        exit_fill * position.qty, fee_bps
    )
    return _gross_pnl(position, exit_fill) - fees, exit_fill


def qty_for(price: float, notional_usdt: float, min_notional: float, min_qty: float) -> float:
    if price <= 0 or notional_usdt <= 0 or notional_usdt < min_notional:
        return 0.0
    quantity = notional_usdt / price
    if quantity < min_qty:
        return 0.0
    return round(quantity, 8)


def prices_close(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))


def check_tp_sl(
    position: Any, price: float, tp_pct: float, sl_pct: float
) -> tuple[bool, bool]:
    if position.side == "long":
        return (
            tp_pct > 0 and price >= position.avg * (1 + tp_pct),
            sl_pct > 0 and price <= position.avg * (1 - sl_pct),
        )
    return (
        tp_pct > 0 and price <= position.avg * (1 - tp_pct),
        sl_pct > 0 and price >= position.avg * (1 + sl_pct),
    )


def parse_signal_events(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[SignalEvent], list[dict[str, Any]]]:
    events: list[SignalEvent] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ts_raw = row.get("ts", row.get("timestamp", ""))
        timestamp = parse_timestamp(ts_raw)
        price = _finite(row.get("px", row.get("price")))
        p_meta = _finite(row.get("p_meta"))
        rv_mean = _finite(row.get("rv_mean"))
        threshold = _finite(row.get("thr"))
        symbol = str(row.get("symbol", "") or "").strip()
        if timestamp is None:
            exclusions.append({"source_order": index, "reason": "malformed_timestamp"})
            continue
        if not symbol or price is None or price <= 0 or p_meta is None or rv_mean is None or threshold is None:
            exclusions.append({"source_order": index, "reason": "malformed_signal_fields"})
            continue
        kinds_used: Optional[str]
        if "kinds_used" not in row:
            kinds_used = None
        else:
            kinds_used = str(row.get("kinds_used", "") or "").strip()
        events.append(
            SignalEvent(
                timestamp=timestamp,
                source_order=index,
                ts=timestamp.isoformat().replace("+00:00", "Z"),
                symbol=symbol,
                price=price,
                p_meta=p_meta,
                rv_mean=rv_mean,
                allow=1 if _truthy(row.get("allow")) else 0,
                thr=threshold,
                mode=str(row.get("mode", "abs") or "abs").strip().lower(),
                kinds_used=kinds_used,
                signal_id=str(row.get("signal_id", "") or "").strip(),
            )
        )
    events.sort(key=lambda event: (event.timestamp, event.source_order))
    previous: dict[str, datetime] = {}
    for event in events:
        prior = previous.get(event.symbol)
        if prior is not None and event.timestamp < prior:
            raise CounterfactualReplayError(
                f"backward timestamp after ordering for symbol {event.symbol}"
            )
        previous[event.symbol] = event.timestamp
    return events, exclusions


def _decision_map(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    decisions: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for index, raw in enumerate(rows):
        row = {str(key): str(value or "").strip() for key, value in raw.items()}
        signal_id = row.get("signal_id", "")
        if not signal_id:
            exclusions.append({"source_order": index, "reason": "missing_signal_id"})
            continue
        digest = canonical_row_digest(row)
        if signal_id in decisions and digests[signal_id] != digest:
            raise CounterfactualReplayError(
                f"conflicting XGBoost decisions for signal_id {signal_id}"
            )
        decisions[signal_id] = row
        digests[signal_id] = digest
    return decisions, exclusions


def _contract_number(contract: Mapping[str, Any], name: str) -> float:
    value = _finite(contract.get(name))
    if value is None:
        raise CounterfactualReplayError(f"contract field {name} is not numeric")
    return value


def _unsupported_runtime_reason(contract: Mapping[str, Any]) -> Optional[str]:
    if contract.get("place_real_orders") is True or contract.get("paper_mode") is not True:
        return "unsafe non-paper replay contract"
    active = [
        field
        for field in (
            "survival_active",
            "xgboost_blocking",
            "iforest_blocking",
            "advanced_risk_active",
        )
        if contract.get(field) is True
    ]
    if active:
        return "unsupported active/blocking behavior: " + ", ".join(active)
    if contract.get("restore_state") is True:
        return "restored position state is unsupported"
    if contract.get("adaptive") is True:
        return "adaptive threshold sequence is unresolved"
    if contract.get("bias_guard") is True:
        return "bias-lock runtime state is unresolved"
    if contract.get("v2_enabled") is True and _contract_number(
        contract, "v2_time_stop_minutes"
    ) <= 0:
        return "V2 entry-gate runtime state is unresolved"
    return None


def _policy_allows_entry(
    policy: str, signal_id: str, decisions: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, str]:
    if policy == "baseline":
        return True, ""
    decision = decisions.get(signal_id)
    if decision is None:
        return False, "xgboost_join_missing"
    confirmed = _truthy(decision.get("would_confirm"))
    rejected = _truthy(decision.get("would_reject"))
    if policy == "xgboost_confirm_only":
        return (confirmed and not rejected, "" if confirmed and not rejected else "xgboost_not_confirmed")
    return (rejected and not confirmed, "" if rejected and not confirmed else "xgboost_not_rejected")


def _portfolio_exposure(positions: Mapping[str, ReplayPosition]) -> float:
    return sum(position.qty * position.avg for position in positions.values())


def _close_position(
    position: ReplayPosition,
    event: SignalEvent,
    reason_prefix: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    action = "SELL" if position.side == "long" else "BUY_TO_COVER"
    pnl, exit_fill = net_pnl_on_close(
        position,
        event.price,
        action,
        _contract_number(contract, "fee_bps"),
        _contract_number(contract, "slippage_bps"),
    )
    fees = fee_cost(position.avg * position.qty, _contract_number(contract, "fee_bps")) + fee_cost(
        exit_fill * position.qty, _contract_number(contract, "fee_bps")
    )
    exit_slippage = abs(exit_fill - event.price) * position.qty
    return {
        "entry_signal_id": position.entry_signal_id,
        "close_signal_id": event.signal_id,
        "entry_timestamp": position.entry_timestamp.isoformat().replace("+00:00", "Z"),
        "timestamp": event.ts,
        "symbol": event.symbol,
        "closed_side": action,
        "position_side": position.side,
        "quantity": position.qty,
        "entry_average": position.avg,
        "exit_mid": event.price,
        "exit_fill_price": exit_fill,
        "exit_reason": reason_prefix,
        "net_pnl": pnl,
        "holding_seconds": (event.timestamp - position.entry_timestamp).total_seconds(),
        "fee_cost": fees,
        "estimated_slippage_cost": position.entry_slippage_cost + exit_slippage,
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "entry_count": 0,
        "closed_trade_count": 0,
        "censored_position_count": 0,
        "total_net_pnl": 0.0,
        "average_net_pnl": None,
        "median_net_pnl": None,
        "win_rate": None,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": None,
        "maximum_drawdown": 0.0,
        "best_trade": None,
        "worst_trade": None,
        "long_trade_count": 0,
        "short_trade_count": 0,
        "exit_reason_counts": {},
        "average_holding_seconds": None,
        "fee_total": 0.0,
        "estimated_slippage_cost": 0.0,
    }


def portfolio_metrics(
    entries: Sequence[Mapping[str, Any]],
    closes: Sequence[Mapping[str, Any]],
    censored: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = _empty_metrics()
    pnls = [float(item["net_pnl"]) for item in closes]
    gains = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    metrics.update(
        {
            "entry_count": len(entries),
            "closed_trade_count": len(closes),
            "censored_position_count": len(censored),
            "total_net_pnl": sum(pnls),
            "average_net_pnl": None if not pnls else statistics.fmean(pnls),
            "median_net_pnl": None if not pnls else statistics.median(pnls),
            "win_rate": None if not pnls else len(gains) / len(pnls),
            "gross_profit": sum(gains),
            "gross_loss": sum(losses),
            "profit_factor": None if not losses else sum(gains) / abs(sum(losses)),
            "maximum_drawdown": drawdown,
            "best_trade": None if not closes else max(closes, key=lambda item: float(item["net_pnl"])),
            "worst_trade": None if not closes else min(closes, key=lambda item: float(item["net_pnl"])),
            "long_trade_count": sum(item.get("position_side") == "long" for item in closes),
            "short_trade_count": sum(item.get("position_side") == "short" for item in closes),
            "exit_reason_counts": dict(Counter(str(item["exit_reason"]) for item in closes)),
            "average_holding_seconds": None
            if not closes
            else statistics.fmean(float(item["holding_seconds"]) for item in closes),
            "fee_total": sum(float(item["fee_cost"]) for item in closes),
            "estimated_slippage_cost": sum(
                float(item["estimated_slippage_cost"]) for item in closes
            ),
        }
    )
    return metrics


def replay_portfolio(
    events: Sequence[SignalEvent],
    xgboost_decisions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    policy: str = "baseline",
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise CounterfactualReplayError(f"unknown replay policy: {policy}")
    unsupported = _unsupported_runtime_reason(contract)
    if unsupported:
        return {
            "replay_status": "unsupported_historical_runtime_state",
            "unsupported_reason": unsupported,
            "exploratory_nonproduction_policy": policy == "xgboost_reject_only",
            "entries": [],
            "closed_trades": [],
            "censored_positions": [],
            "blocked_entries": [],
            "metrics": _empty_metrics(),
        }

    positions: dict[str, ReplayPosition] = {}
    entries: list[dict[str, Any]] = []
    closes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    last_fill_time: dict[str, float] = {}
    last_fill_price: dict[str, float] = {}
    flip_pending: dict[str, tuple[str, int]] = {}
    cooldown = _contract_number(contract, "cooldown_sec")
    notional = _contract_number(contract, "notional_usdt")
    max_portfolio = _contract_number(contract, "max_portfolio_usdt")
    tp_pct = _contract_number(contract, "tp_pct")
    sl_pct = _contract_number(contract, "sl_pct")
    time_stop_seconds = _contract_number(contract, "v2_time_stop_minutes") * 60.0

    for event in events:
        now = event.timestamp.timestamp()
        position = positions.get(event.symbol)

        if position is not None:
            hit_tp, hit_sl = check_tp_sl(position, event.price, tp_pct, sl_pct)
            if hit_tp or hit_sl:
                close = _close_position(
                    position, event, "EXIT_TP" if hit_tp else "EXIT_SL", contract
                )
                closes.append(close)
                positions.pop(event.symbol)
                flip_pending.pop(event.symbol, None)
                last_fill_time[event.symbol] = now
                last_fill_price[event.symbol] = event.price
                continue

        position = positions.get(event.symbol)
        if (
            position is not None
            and contract.get("v2_enabled") is True
            and time_stop_seconds > 0
            and (event.timestamp - position.entry_timestamp).total_seconds()
            >= time_stop_seconds
        ):
            closes.append(_close_position(position, event, "EXIT_TIME", contract))
            positions.pop(event.symbol)
            flip_pending.pop(event.symbol, None)
            last_fill_time[event.symbol] = now
            last_fill_price[event.symbol] = event.price
            continue

        if not side_allowed(event.p_meta, str(contract["sides"])):
            continue
        passed, _reason = threshold_pass(
            event,
            _contract_number(contract, "exec_thr"),
            str(contract["exec_mode"]),
            bool(contract["respect_writer_thr"]),
        )
        if not passed or abs(event.rv_mean) > _contract_number(contract, "rv_max"):
            continue

        wanted = "long" if event.p_meta >= 0 else "short"
        open_action = "BUY" if wanted == "long" else "SELL_SHORT"
        cooldown_ok = now - last_fill_time.get(event.symbol, 0.0) >= cooldown
        position = positions.get(event.symbol)

        if bool(contract["one_position"]) and position is None and len(positions) >= 1:
            blocked.append({"signal_id": event.signal_id, "reason": "one_position_active"})
            continue
        if position is None and len(positions) >= int(contract["max_symbols"]):
            blocked.append({"signal_id": event.signal_id, "reason": "max_symbols"})
            continue

        if position is not None and position.side == wanted:
            flip_pending.pop(event.symbol, None)
            if not bool(contract["scale_in"]):
                continue
            policy_allowed, policy_reason = _policy_allows_entry(
                policy, event.signal_id, xgboost_decisions
            )
            if not policy_allowed:
                blocked.append({"signal_id": event.signal_id, "reason": policy_reason})
                continue
            if not cooldown_ok:
                continue
            if _portfolio_exposure(positions) + notional > max_portfolio:
                blocked.append({"signal_id": event.signal_id, "reason": "portfolio_cap_scale"})
                continue
            quantity = qty_for(
                event.price,
                notional,
                _contract_number(contract, "min_notional"),
                _contract_number(contract, "min_qty"),
            )
            if quantity <= 0:
                continue
            fill = apply_slippage(event.price, open_action, _contract_number(contract, "slippage_bps"))
            combined_qty = position.qty + quantity
            position.avg = (position.avg * position.qty + fill * quantity) / combined_qty
            position.qty = combined_qty
            position.entry_slippage_cost += abs(fill - event.price) * quantity
            entries.append(
                {
                    "signal_id": event.signal_id,
                    "timestamp": event.ts,
                    "symbol": event.symbol,
                    "side": open_action,
                    "quantity": quantity,
                    "entry_mid": event.price,
                    "entry_fill_price": fill,
                    "reason": "SCALE_IN",
                }
            )
            last_fill_time[event.symbol] = now
            last_fill_price[event.symbol] = event.price
            continue

        just_flipped = False
        if position is not None and position.side != wanted:
            confirm_ticks = int(contract["flip_confirm_ticks"])
            if confirm_ticks > 0:
                pending_side, pending_count = flip_pending.get(event.symbol, ("", 0))
                pending_count = pending_count + 1 if pending_side == wanted else 1
                flip_pending[event.symbol] = (wanted, pending_count)
                if pending_count < confirm_ticks:
                    continue
                flip_pending.pop(event.symbol, None)
            closes.append(_close_position(position, event, "FLIP_CLOSE", contract))
            positions.pop(event.symbol)
            last_fill_time[event.symbol] = now
            last_fill_price[event.symbol] = event.price
            just_flipped = True
            if not bool(contract["flip_open"]):
                continue

        policy_allowed, policy_reason = _policy_allows_entry(
            policy, event.signal_id, xgboost_decisions
        )
        if not policy_allowed:
            blocked.append({"signal_id": event.signal_id, "reason": policy_reason})
            continue
        if not cooldown_ok and not just_flipped and event.symbol not in positions:
            blocked.append({"signal_id": event.signal_id, "reason": "cooldown"})
            continue
        previous_price = last_fill_price.get(event.symbol, 0.0)
        if (
            prices_close(previous_price, event.price)
            and now - last_fill_time.get(event.symbol, 0.0) < cooldown * 2
            and event.symbol not in positions
            and not just_flipped
        ):
            blocked.append({"signal_id": event.signal_id, "reason": "dup_price"})
            continue
        if event.symbol not in positions and _portfolio_exposure(positions) + notional > max_portfolio:
            blocked.append({"signal_id": event.signal_id, "reason": "portfolio_cap"})
            continue
        quantity = qty_for(
            event.price,
            notional,
            _contract_number(contract, "min_notional"),
            _contract_number(contract, "min_qty"),
        )
        if quantity <= 0:
            continue
        fill = apply_slippage(event.price, open_action, _contract_number(contract, "slippage_bps"))
        positions[event.symbol] = ReplayPosition(
            side=wanted,
            qty=quantity,
            avg=fill,
            entry_timestamp=event.timestamp,
            entry_signal_id=event.signal_id,
            entry_slippage_cost=abs(fill - event.price) * quantity,
        )
        entries.append(
            {
                "signal_id": event.signal_id,
                "timestamp": event.ts,
                "symbol": event.symbol,
                "side": open_action,
                "quantity": quantity,
                "entry_mid": event.price,
                "entry_fill_price": fill,
                "reason": "ENTRY",
            }
        )
        last_fill_time[event.symbol] = now
        last_fill_price[event.symbol] = event.price

    censored = [
        {
            "symbol": symbol,
            "side": position.side,
            "quantity": position.qty,
            "entry_average": position.avg,
            "entry_signal_id": position.entry_signal_id,
            "entry_timestamp": position.entry_timestamp.isoformat().replace("+00:00", "Z"),
            "status": "censored_open_position",
        }
        for symbol, position in sorted(positions.items())
    ]
    return {
        "replay_status": "completed",
        "exploratory_nonproduction_policy": policy == "xgboost_reject_only",
        "entries": entries,
        "closed_trades": closes,
        "censored_positions": censored,
        "blocked_entries": blocked,
        "metrics": portfolio_metrics(entries, closes, censored),
    }


def _decision_cohort(row: Mapping[str, Any]) -> Optional[str]:
    confirmed = _truthy(row.get("would_confirm"))
    rejected = _truthy(row.get("would_reject"))
    if confirmed and not rejected:
        return "would_confirm"
    if rejected and not confirmed:
        return "would_reject"
    return None


def replay_independent_cohorts(
    events: Sequence[SignalEvent],
    decisions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_events = list(events)
    by_id: dict[str, tuple[int, SignalEvent]] = {}
    for index, event in enumerate(ordered_events):
        if not event.signal_id:
            continue
        previous = by_id.get(event.signal_id)
        if previous is not None and previous[1] != event:
            raise CounterfactualReplayError(
                f"conflicting signal events for signal_id {event.signal_id}"
            )
        by_id[event.signal_id] = (index, event)
    records: dict[str, list[dict[str, Any]]] = {
        "would_confirm": [],
        "would_reject": [],
    }
    exclusions: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    notional = _contract_number(contract, "notional_usdt")
    tp_pct = _contract_number(contract, "tp_pct")
    sl_pct = _contract_number(contract, "sl_pct")
    time_stop_seconds = _contract_number(contract, "v2_time_stop_minutes") * 60.0

    ordered_decisions = sorted(
        decisions.items(),
        key=lambda item: (
            by_id[item[0]][1].timestamp if item[0] in by_id else datetime.max.replace(tzinfo=timezone.utc),
            by_id[item[0]][1].source_order if item[0] in by_id else 0,
            item[0],
        ),
    )
    for signal_id, decision in ordered_decisions:
        cohort = _decision_cohort(decision)
        if cohort is None:
            exclusions.append({"signal_id": signal_id, "reason": "ambiguous_cohort"})
            continue
        decision_counts[cohort] += 1
        joined = by_id.get(signal_id)
        if joined is None:
            exclusions.append({"signal_id": signal_id, "cohort": cohort, "reason": "signal_id_join_missing"})
            continue
        event_index, event = joined
        decision_timestamp_raw = decision.get("timestamp", decision.get("ts", ""))
        if str(decision_timestamp_raw or "").strip():
            decision_timestamp = parse_timestamp(decision_timestamp_raw)
            if decision_timestamp is None:
                exclusions.append(
                    {"signal_id": signal_id, "cohort": cohort, "reason": "malformed_timestamp"}
                )
                continue
            if decision_timestamp != event.timestamp:
                exclusions.append(
                    {"signal_id": signal_id, "cohort": cohort, "reason": "decision_timestamp_mismatch"}
                )
                continue
        decision_symbol = str(decision.get("symbol", "") or "").strip()
        if decision_symbol and decision_symbol != event.symbol:
            exclusions.append(
                {"signal_id": signal_id, "cohort": cohort, "reason": "decision_symbol_mismatch"}
            )
            continue
        side_text = str(decision.get("existing_signal", "") or "").strip().upper()
        if side_text not in {"LONG", "SHORT"}:
            exclusions.append({"signal_id": signal_id, "cohort": cohort, "reason": "flat_or_invalid_direction"})
            continue
        future = [
            candidate
            for candidate in ordered_events[event_index + 1 :]
            if candidate.symbol == event.symbol
        ]
        if not future:
            exclusions.append({"signal_id": signal_id, "cohort": cohort, "reason": "no_future_price"})
            continue
        side = "long" if side_text == "LONG" else "short"
        action_open = "BUY" if side == "long" else "SELL_SHORT"
        quantity = qty_for(
            event.price,
            notional,
            _contract_number(contract, "min_notional"),
            _contract_number(contract, "min_qty"),
        )
        if quantity <= 0:
            exclusions.append({"signal_id": signal_id, "cohort": cohort, "reason": "invalid_entry_quantity"})
            continue
        fill = apply_slippage(event.price, action_open, _contract_number(contract, "slippage_bps"))
        position = ReplayPosition(side, quantity, fill, event.timestamp, signal_id)
        mfe = 0.0
        mae = 0.0
        exit_event: Optional[SignalEvent] = None
        exit_reason: Optional[str] = None
        for candidate in future:
            signed_return = (
                (candidate.price - event.price) / event.price
                if side == "long"
                else (event.price - candidate.price) / event.price
            )
            mfe = max(mfe, signed_return)
            mae = min(mae, signed_return)
            hit_tp, hit_sl = check_tp_sl(position, candidate.price, tp_pct, sl_pct)
            if hit_tp or hit_sl:
                exit_event = candidate
                exit_reason = "EXIT_TP" if hit_tp else "EXIT_SL"
                break
            if (
                contract.get("v2_enabled") is True
                and time_stop_seconds > 0
                and (candidate.timestamp - event.timestamp).total_seconds() >= time_stop_seconds
            ):
                exit_event = candidate
                exit_reason = "EXIT_TIME"
                break
        confidence = _finite(decision.get("confidence"))
        if confidence is None:
            confidence = _finite(decision.get("xgboost_confidence"))
        base = {
            "signal_id": signal_id,
            "timestamp": event.ts,
            "symbol": event.symbol,
            "cohort": cohort,
            "side": side,
            "entry_mid": event.price,
            "entry_fill": fill,
            "signal_tick_mfe": mfe,
            "signal_tick_mae": mae,
            "model_version": decision.get("model_version"),
            "confidence": confidence,
            "reject_reason": decision.get("reject_reason") or decision.get("reason") or "",
        }
        if exit_event is None:
            base.update(
                {
                    "exit_timestamp": None,
                    "exit_mid": None,
                    "exit_fill": None,
                    "exit_reason": "censored_open_position",
                    "net_pnl": None,
                    "return_on_notional": None,
                    "holding_seconds": (future[-1].timestamp - event.timestamp).total_seconds(),
                    "censored": True,
                }
            )
        else:
            action_close = "SELL" if side == "long" else "BUY_TO_COVER"
            pnl, exit_fill = net_pnl_on_close(
                position,
                exit_event.price,
                action_close,
                _contract_number(contract, "fee_bps"),
                _contract_number(contract, "slippage_bps"),
            )
            base.update(
                {
                    "exit_timestamp": exit_event.ts,
                    "exit_mid": exit_event.price,
                    "exit_fill": exit_fill,
                    "exit_reason": exit_reason,
                    "net_pnl": pnl,
                    "return_on_notional": pnl / (fill * quantity),
                    "holding_seconds": (exit_event.timestamp - event.timestamp).total_seconds(),
                    "censored": False,
                }
            )
        records[cohort].append(base)

    return {
        "would_confirm": _cohort_metrics(
            records["would_confirm"], decision_counts["would_confirm"]
        ),
        "would_reject": _cohort_metrics(
            records["would_reject"], decision_counts["would_reject"]
        ),
        "exclusions": exclusions,
        "non_additive": True,
    }


def _cohort_metrics(records: Sequence[Mapping[str, Any]], decision_count: int) -> dict[str, Any]:
    closed = [record for record in records if not record.get("censored")]
    pnls = [float(record["net_pnl"]) for record in closed]
    returns = [float(record["return_on_notional"]) for record in closed]
    confidences = [
        float(value)
        for record in records
        if (value := _finite(record.get("confidence"))) is not None
    ]
    return {
        "decision_count": decision_count,
        "eligible_count": len(records),
        "closed_count": len(closed),
        "censored_count": sum(bool(record.get("censored")) for record in records),
        "average_net_pnl": None if not pnls else statistics.fmean(pnls),
        "median_net_pnl": None if not pnls else statistics.median(pnls),
        "win_rate": None if not pnls else sum(value > 0 for value in pnls) / len(pnls),
        "average_return_on_notional": None if not returns else statistics.fmean(returns),
        "average_signal_tick_mfe": None
        if not records
        else statistics.fmean(float(record["signal_tick_mfe"]) for record in records),
        "average_signal_tick_mae": None
        if not records
        else statistics.fmean(float(record["signal_tick_mae"]) for record in records),
        "confidence_average": None if not confidences else statistics.fmean(confidences),
        "reason_counts": dict(
            Counter(str(record.get("reject_reason") or "unspecified") for record in records)
        ),
        "non_additive": True,
        "records": list(records),
    }


def cohort_separation(
    confirm: Mapping[str, Any],
    reject: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20,
) -> dict[str, Any]:
    confirm_closed = [
        float(record["net_pnl"])
        for record in confirm.get("records", [])
        if not record.get("censored")
    ]
    reject_closed = [
        float(record["net_pnl"])
        for record in reject.get("records", [])
        if not record.get("censored")
    ]
    result = {
        "confirm_minus_reject_average_pnl": _difference(
            confirm.get("average_net_pnl"), reject.get("average_net_pnl")
        ),
        "confirm_minus_reject_win_rate": _difference(
            confirm.get("win_rate"), reject.get("win_rate")
        ),
        "confirm_minus_reject_return": _difference(
            confirm.get("average_return_on_notional"), reject.get("average_return_on_notional")
        ),
        "bootstrap_confidence_interval": None,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "confirm_closed_sample_count": len(confirm_closed),
        "reject_closed_sample_count": len(reject_closed),
        "bootstrap_status": "insufficient_cohort_sample",
        "significance_claimed": False,
        "non_additive": True,
    }
    if len(confirm_closed) < 5 or len(reject_closed) < 5:
        return result
    rng = random.Random(bootstrap_seed)
    differences: list[float] = []
    for _ in range(bootstrap_samples):
        c_mean = statistics.fmean(rng.choice(confirm_closed) for _ in confirm_closed)
        r_mean = statistics.fmean(rng.choice(reject_closed) for _ in reject_closed)
        differences.append(c_mean - r_mean)
    differences.sort()
    lower = differences[int(0.025 * (len(differences) - 1))]
    upper = differences[int(0.975 * (len(differences) - 1))]
    result.update(
        {
            "bootstrap_confidence_interval": [lower, upper],
            "bootstrap_status": "completed",
        }
    )
    return result


def _difference(left: Any, right: Any) -> Optional[float]:
    a, b = _finite(left), _finite(right)
    return None if a is None or b is None else a - b


def _reason_prefix(value: Any) -> str:
    text = str(value or "").strip().upper()
    return next((prefix for prefix in REASON_PREFIXES if text.startswith(prefix)), text.split(" ", 1)[0])


def _entry_actual(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "signal_id": str(row.get("signal_id", "") or "").strip(),
        "timestamp": parse_timestamp(row.get("ts", row.get("timestamp"))),
        "symbol": str(row.get("symbol", "") or "").strip(),
        "side": str(row.get("side", row.get("action", "")) or "").strip().upper(),
        "quantity": _finite(row.get("qty", row.get("quantity"))),
        "price": _finite(row.get("price", row.get("entry_fill_price"))),
    }


def _close_actual(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_signal_id": str(row.get("signal_id", "") or "").strip(),
        "timestamp": parse_timestamp(row.get("ts", row.get("timestamp"))),
        "symbol": str(row.get("symbol", "") or "").strip(),
        "closed_side": str(row.get("closed_side", row.get("side", "")) or "").strip().upper(),
        "quantity": _finite(row.get("qty", row.get("quantity"))),
        "entry_average": _finite(row.get("entry_avg", row.get("entry_average"))),
        "exit_price": _finite(row.get("exit_price", row.get("exit_fill_price"))),
        "pnl": _finite(row.get("realized_pnl", row.get("net_pnl"))),
        "reason": _reason_prefix(row.get("reason", row.get("exit_reason"))),
    }


def _within_timestamp(left: Optional[datetime], right: Optional[datetime], tolerance: float) -> bool:
    return bool(left is not None and right is not None and abs((left - right).total_seconds()) <= tolerance)


def _within_abs(left: Any, right: Any, tolerance: float) -> bool:
    a, b = _finite(left), _finite(right)
    return bool(a is not None and b is not None and abs(a - b) <= tolerance)


def _within_rel(left: Any, right: Any, tolerance: float) -> bool:
    a, b = _finite(left), _finite(right)
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def _match_records(
    actual: Sequence[dict[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    close: bool,
    timestamp_tolerance: float,
    quantity_tolerance: float,
    price_tolerance: float,
    pnl_tolerance: float,
) -> tuple[list[tuple[dict[str, Any], Mapping[str, Any]]], list[dict[str, Any]], list[Mapping[str, Any]]]:
    remaining = list(range(len(replay)))
    matches: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    for item in actual:
        chosen: Optional[int] = None
        for index in remaining:
            candidate = replay[index]
            if close:
                if (
                    item["entry_signal_id"] != candidate.get("entry_signal_id")
                    or item["symbol"] != candidate.get("symbol")
                    or item["closed_side"] != str(candidate.get("closed_side", "")).upper()
                    or item["reason"] != _reason_prefix(candidate.get("exit_reason"))
                ):
                    continue
                replay_ts = parse_timestamp(candidate.get("timestamp"))
                numeric_ok = (
                    _within_abs(item["quantity"], candidate.get("quantity"), quantity_tolerance)
                    and _within_rel(item["entry_average"], candidate.get("entry_average"), price_tolerance)
                    and _within_rel(item["exit_price"], candidate.get("exit_fill_price"), price_tolerance)
                    and _within_abs(item["pnl"], candidate.get("net_pnl"), pnl_tolerance)
                )
            else:
                if (
                    item["signal_id"] != candidate.get("signal_id")
                    or item["symbol"] != candidate.get("symbol")
                    or item["side"] != str(candidate.get("side", "")).upper()
                ):
                    continue
                replay_ts = parse_timestamp(candidate.get("timestamp"))
                numeric_ok = _within_abs(
                    item["quantity"], candidate.get("quantity"), quantity_tolerance
                ) and _within_rel(item["price"], candidate.get("entry_fill_price"), price_tolerance)
            if _within_timestamp(item["timestamp"], replay_ts, timestamp_tolerance) and numeric_ok:
                chosen = index
                break
        if chosen is None:
            missing.append(item)
        else:
            remaining.remove(chosen)
            matches.append((item, replay[chosen]))
    return matches, missing, [replay[index] for index in remaining]


def _logical_parity_pairs(
    actual: Sequence[dict[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    close: bool,
) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    remaining = list(range(len(replay)))
    pairs: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for item in actual:
        for index in remaining:
            candidate = replay[index]
            if close:
                same = bool(
                    item["entry_signal_id"] == candidate.get("entry_signal_id")
                    and item["symbol"] == candidate.get("symbol")
                    and item["closed_side"] == str(candidate.get("closed_side", "")).upper()
                    and item["reason"] == _reason_prefix(candidate.get("exit_reason"))
                )
            else:
                same = bool(
                    item["signal_id"] == candidate.get("signal_id")
                    and item["symbol"] == candidate.get("symbol")
                    and item["side"] == str(candidate.get("side", "")).upper()
                )
            if same:
                pairs.append((item, candidate))
                remaining.remove(index)
                break
    return pairs


def _serializable_actual(item: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(item)
    timestamp = result.get("timestamp")
    if isinstance(timestamp, datetime):
        result["timestamp"] = timestamp.isoformat().replace("+00:00", "Z")
    return result


def _actual_open_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    ordered = sorted(
        rows,
        key=lambda row: parse_timestamp(row.get("ts", row.get("timestamp")))
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    for row in ordered:
        action = str(row.get("side", row.get("action", "")) or "").strip().upper()
        symbol = str(row.get("symbol", "") or "").strip()
        qty = _finite(row.get("qty", row.get("quantity"))) or 0.0
        price = _finite(row.get("price", row.get("entry_fill_price"))) or 0.0
        signal_id = str(row.get("signal_id", "") or "").strip()
        wanted = "long" if action == "BUY" else "short" if action == "SELL_SHORT" else None
        if wanted is not None:
            previous = state.get(symbol)
            if previous and previous["side"] == wanted:
                combined = float(previous["quantity"]) + qty
                average = (
                    (float(previous["entry_average"]) * float(previous["quantity"]) + price * qty)
                    / combined
                    if combined > 0
                    else 0.0
                )
                state[symbol] = {
                    "side": wanted,
                    "quantity": combined,
                    "entry_average": average,
                    "entry_signal_id": previous["entry_signal_id"],
                }
            else:
                state[symbol] = {
                    "side": wanted,
                    "quantity": qty,
                    "entry_average": price,
                    "entry_signal_id": signal_id,
                }
        elif action in CLOSE_ACTIONS:
            state.pop(symbol, None)
    return state


def compare_recorded_parity(
    baseline: Mapping[str, Any],
    paper_rows: Sequence[Mapping[str, Any]],
    closed_rows: Sequence[Mapping[str, Any]],
    *,
    timestamp_tolerance_seconds: float = 1.0,
    quantity_tolerance_abs: float = 1e-8,
    price_tolerance_rel: float = 1e-8,
    pnl_tolerance_abs: float = 1e-6,
) -> dict[str, Any]:
    actual_entries = [
        _entry_actual(row)
        for row in paper_rows
        if str(row.get("side", row.get("action", "")) or "").strip().upper() in ENTRY_ACTIONS
    ]
    actual_closes = [_close_actual(row) for row in closed_rows]
    replay_entries = list(baseline.get("entries", []))
    replay_closes = list(baseline.get("closed_trades", []))
    entry_matches, missing_entries, extra_entries = _match_records(
        actual_entries,
        replay_entries,
        close=False,
        timestamp_tolerance=timestamp_tolerance_seconds,
        quantity_tolerance=quantity_tolerance_abs,
        price_tolerance=price_tolerance_rel,
        pnl_tolerance=pnl_tolerance_abs,
    )
    close_matches, missing_closes, extra_closes = _match_records(
        actual_closes,
        replay_closes,
        close=True,
        timestamp_tolerance=timestamp_tolerance_seconds,
        quantity_tolerance=quantity_tolerance_abs,
        price_tolerance=price_tolerance_rel,
        pnl_tolerance=pnl_tolerance_abs,
    )
    actual_open = _actual_open_state(paper_rows)
    replay_open = {
        str(item["symbol"]): {
            "side": str(item["side"]),
            "quantity": float(item["quantity"]),
            "entry_average": float(item["entry_average"]),
            "entry_signal_id": str(item.get("entry_signal_id", "")),
        }
        for item in baseline.get("censored_positions", [])
    }
    open_state_parity = actual_open.keys() == replay_open.keys() and all(
        actual_open[symbol]["side"] == replay_open[symbol]["side"]
        and abs(
            float(actual_open[symbol]["quantity"])
            - float(replay_open[symbol]["quantity"])
        )
        <= quantity_tolerance_abs
        and _within_rel(
            actual_open[symbol]["entry_average"],
            replay_open[symbol]["entry_average"],
            price_tolerance_rel,
        )
        and actual_open[symbol]["entry_signal_id"]
        == replay_open[symbol]["entry_signal_id"]
        for symbol in actual_open
    )
    logical_entry_pairs = _logical_parity_pairs(actual_entries, replay_entries, close=False)
    logical_close_pairs = _logical_parity_pairs(actual_closes, replay_closes, close=True)
    entry_errors = [
        abs(float(actual["price"]) - float(replay["entry_fill_price"]))
        for actual, replay in logical_entry_pairs
        if _finite(actual.get("price")) is not None
        and _finite(replay.get("entry_fill_price")) is not None
    ]
    exit_errors = [
        abs(float(actual["exit_price"]) - float(replay["exit_fill_price"]))
        for actual, replay in logical_close_pairs
        if _finite(actual.get("exit_price")) is not None
        and _finite(replay.get("exit_fill_price")) is not None
    ]
    pnl_errors = [
        abs(float(actual["pnl"]) - float(replay["net_pnl"]))
        for actual, replay in logical_close_pairs
        if _finite(actual.get("pnl")) is not None and _finite(replay.get("net_pnl")) is not None
    ]
    exact_entry = not missing_entries and not extra_entries and len(entry_matches) == len(actual_entries)
    exact_close = not missing_closes and not extra_closes and len(close_matches) == len(actual_closes)
    mechanical = exact_entry and exact_close and open_state_parity
    if not actual_closes:
        status = "not_testable_no_closed_trades"
        parity_passed = False
    elif mechanical and len(close_matches) < 10:
        status = "mechanically_passed_insufficient_sample"
        parity_passed = True
    elif mechanical:
        status = "parity_verified"
        parity_passed = True
    else:
        status = "parity_failed"
        parity_passed = False
    return {
        "actual_entry_count": len(actual_entries),
        "replay_entry_count": len(replay_entries),
        "matched_entry_count": len(entry_matches),
        "missing_actual_entries": [_serializable_actual(item) for item in missing_entries],
        "extra_replay_entries": extra_entries,
        "actual_closed_count": len(actual_closes),
        "replay_closed_count": len(replay_closes),
        "matched_closed_count": len(close_matches),
        "missing_actual_closes": [_serializable_actual(item) for item in missing_closes],
        "extra_replay_closes": extra_closes,
        "maximum_entry_price_error": max(entry_errors, default=None),
        "maximum_exit_price_error": max(exit_errors, default=None),
        "maximum_pnl_error": max(pnl_errors, default=None),
        "final_actual_open_positions": len(actual_open),
        "final_replay_open_positions": len(replay_open),
        "final_actual_open_state": actual_open,
        "final_replay_open_state": replay_open,
        "exact_entry_parity": exact_entry,
        "exact_close_parity": exact_close,
        "exact_open_state_parity": open_state_parity,
        "parity_passed": parity_passed,
        "parity_status": status,
        "promotion_parity_sample_gate": bool(parity_passed and len(close_matches) >= 10),
        "tolerances": {
            "timestamp_seconds": timestamp_tolerance_seconds,
            "quantity_absolute": quantity_tolerance_abs,
            "price_relative": price_tolerance_rel,
            "pnl_absolute": pnl_tolerance_abs,
        },
    }


def _empty_variant(status: str = "not_run") -> dict[str, Any]:
    return {
        "replay_status": status,
        "entries": [],
        "closed_trades": [],
        "censored_positions": [],
        "metrics": _empty_metrics(),
    }


def _empty_cohort() -> dict[str, Any]:
    return _cohort_metrics([], 0)


def _run_replay(
    manifest_run: Mapping[str, Any],
    contract_result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    inventory_only: bool,
    include_nonstrategy: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    identity = str(manifest_run["identity"])
    strategy = manifest_run.get("include_in_strategy_aggregate") is True
    contract_status = str(contract_result.get("status") or "missing")
    source_status = str(source.get("status") or "missing")
    coverage = source.get("coverage") or {}
    warnings: list[str] = []
    exclusions: list[dict[str, Any]] = []
    baseline = _empty_variant()
    confirm_only = _empty_variant()
    reject_only = _empty_variant()
    confirm_cohort = _empty_cohort()
    reject_cohort = _empty_cohort()
    separation = cohort_separation(
        confirm_cohort,
        reject_cohort,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    parity: dict[str, Any] = {"parity_status": "not_run", "parity_passed": False}
    grade = "exploratory_only"
    approved = False

    if not strategy:
        grade = "excluded_research_diagnostic" if include_nonstrategy else "manifest_excluded"
        exclusions.append(
            {
                "reason": "phase19_manifest_excluded",
                "classification": manifest_run.get("classification"),
            }
        )
    elif contract_status == "missing":
        grade = "contract_missing"
        exclusions.append({"reason": "historical_replay_contract_missing"})
    elif contract_status not in PARITY_GRADE_STATUSES:
        grade = "contract_unverified"
        exclusions.append({"reason": "replay_contract_not_parity_grade"})
    elif source_status not in {"exact_bundle", "resolved_from_archives"}:
        grade = "source_coverage_failed"
        exclusions.append({"reason": f"source_resolution_{source_status}"})
    elif not coverage.get("coverage_passed"):
        grade = "source_coverage_failed"
        exclusions.append({"reason": "source_coverage_gates_failed"})

    may_replay = (
        not inventory_only
        and strategy
        and contract_status in PARITY_GRADE_STATUSES
        and source_status in {"exact_bundle", "resolved_from_archives"}
    )
    if may_replay:
        contract = contract_result.get("contract")
        if not isinstance(contract, Mapping):
            raise CounterfactualReplayError(f"resolved contract missing for {identity}")
        events, event_exclusions = parse_signal_events(source["rows"].get("signals", []))
        decisions, decision_exclusions = _decision_map(source["rows"].get("xgboost", []))
        exclusions.extend(event_exclusions)
        exclusions.extend(decision_exclusions)
        baseline = replay_portfolio(events, decisions, contract, "baseline")
        confirm_only = replay_portfolio(events, decisions, contract, "xgboost_confirm_only")
        reject_only = replay_portfolio(events, decisions, contract, "xgboost_reject_only")
        unsupported = baseline.get("replay_status") == "unsupported_historical_runtime_state"
        if unsupported:
            grade = "unsupported_runtime_state"
            exclusions.append({"reason": baseline.get("unsupported_reason")})
        else:
            cohorts = replay_independent_cohorts(events, decisions, contract)
            confirm_cohort = cohorts["would_confirm"]
            reject_cohort = cohorts["would_reject"]
            exclusions.extend(cohorts["exclusions"])
            separation = cohort_separation(
                confirm_cohort,
                reject_cohort,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            parity = compare_recorded_parity(
                baseline,
                source["rows"].get("paper", []),
                source["rows"].get("closed", []),
            )
            if not coverage.get("coverage_passed"):
                grade = "source_coverage_failed"
            elif parity["parity_status"] == "not_testable_no_closed_trades":
                grade = "parity_not_testable"
            elif not parity["parity_passed"]:
                grade = "parity_failed"
            elif not parity["promotion_parity_sample_gate"]:
                grade = "exploratory_only"
            else:
                safety_flags_ok = bool(
                    contract.get("place_real_orders") is False
                    and contract.get("paper_mode") is True
                    and contract.get("restore_state") is False
                    and contract.get("survival_active") is False
                    and contract.get("xgboost_blocking") is False
                    and contract.get("iforest_blocking") is False
                    and contract.get("advanced_risk_active") is False
                )
                conflict_free = not coverage.get("conflicting_signal_ids")
                if safety_flags_ok and conflict_free:
                    grade = "parity_verified"
                    approved = True
                else:
                    grade = "unsupported_runtime_state"
                    exclusions.append({"reason": "unsafe_or_conflicting_replay_state"})

    if inventory_only:
        warnings.append("inventory_only_no_counterfactual_lifecycle_executed")
    if grade != "parity_verified":
        approved = False
    return {
        "identity": identity,
        "mode": manifest_run.get("mode"),
        "manifest_classification": manifest_run.get("classification"),
        "manifest_strategy_included": strategy,
        "contract_status": contract_status,
        "contract_digest": contract_result.get("digest"),
        "source_resolution_status": source_status,
        "source_digest": source.get("source_digest"),
        "bundle_digest": source.get("bundle_digest"),
        "coverage": coverage,
        "parity": parity,
        "baseline": baseline,
        "xgboost_confirm_only": confirm_only,
        "xgboost_reject_only": reject_only,
        "independent_confirm_cohort": confirm_cohort,
        "independent_reject_cohort": reject_cohort,
        "cohort_separation": separation,
        "evidence_grade": grade,
        "counterfactual_evidence_approved": approved,
        "warnings": warnings,
        "exclusions": exclusions,
    }


def _normalized_for_digest(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_for_digest(item)
            for key, item in sorted(value.items())
            if key not in {"generated_at", "reports_dir", "logs_dir", "bundle_root", "path"}
        }
    if isinstance(value, list):
        return [_normalized_for_digest(item) for item in value]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")):
        return Path(value.replace("\\", "/")).name
    return value


def counterfactual_replay_digest(report: Mapping[str, Any]) -> str:
    relevant = {
        "schema_version": report.get("schema_version"),
        "manifest_digest": report.get("manifest_digest"),
        "replay_engine_sha256": report.get("inputs", {}).get("replay_engine_sha256"),
        "runs": report.get("runs", []),
    }
    encoded = json.dumps(
        _normalized_for_digest(relevant),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_report_path(reports_dir: Path, raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    normalized = Path(text.replace("\\", "/"))
    candidates = [normalized] if normalized.is_absolute() else [
        reports_dir.parent / normalized,
        reports_dir / normalized.name,
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _reported_run_row_counts(
    manifest_run: Mapping[str, Any], reports_dir: Path
) -> dict[str, int]:
    """Extract run-local counts only; cumulative live-signal totals are not authoritative."""

    counts: dict[str, int] = {}
    closed_count = manifest_run.get("closed_trade_count")
    if isinstance(closed_count, int) and not isinstance(closed_count, bool) and closed_count >= 0:
        counts["closed"] = closed_count
    report_paths = manifest_run.get("report_paths")
    if not isinstance(report_paths, Mapping):
        return counts
    for label in ("xgboost_audit", "unified", "shadow_summary"):
        path = _manifest_report_path(reports_dir, report_paths.get(label))
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        if label == "xgboost_audit":
            files = payload.get("files")
            if isinstance(files, Mapping):
                for report_key, kind in (
                    ("xgboost_signal_shadow.csv", "xgboost"),
                    ("trades_paper_*.csv", "paper"),
                    ("trades_closed.csv", "closed"),
                ):
                    item = files.get(report_key)
                    value = item.get("rows") if isinstance(item, Mapping) else None
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        counts[kind] = value
        xgboost_section = payload.get("xgboost") or payload.get("xgboost_signal")
        if isinstance(xgboost_section, Mapping):
            value = xgboost_section.get("total_rows")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                counts.setdefault("xgboost", value)
        lineage = payload.get("trade_lineage")
        if isinstance(lineage, Mapping):
            for report_key, kind in (
                ("paper_trade_rows", "paper"),
                ("closed_trade_rows", "closed"),
            ):
                value = lineage.get(report_key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    counts.setdefault(kind, value)
    return counts


def summarize_counterfactual_replay(
    *,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    manifest_path: Path | str | None = None,
    evidence_overrides_path: Path | str = DEFAULT_EVIDENCE_OVERRIDES,
    contract_overrides_path: Path | str = DEFAULT_CONTRACT_OVERRIDES,
    bundle_root: Path | str = DEFAULT_BUNDLE_ROOT,
    identity: Optional[str] = None,
    all_strategy_runs: bool = False,
    include_nonstrategy_runs: bool = False,
    inventory_only: bool = False,
    max_start_delay_seconds: float = 120.0,
    max_end_delay_seconds: float = 120.0,
    max_signal_gap_seconds: float = 180.0,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20,
) -> dict[str, Any]:
    reports = Path(reports_dir)
    logs = Path(logs_dir)
    manifest = build_evidence_manifest(reports, evidence_overrides_path)
    manifest_digest = evidence_manifest_digest(manifest)
    if manifest_path is not None:
        supplied_path = Path(manifest_path)
        if not supplied_path.is_file():
            raise CounterfactualReplayError(f"supplied evidence manifest is missing: {supplied_path}")
        try:
            supplied = json.loads(supplied_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise CounterfactualReplayError(f"malformed supplied manifest: {exc}") from exc
        if not isinstance(supplied, dict):
            raise CounterfactualReplayError("supplied evidence manifest root must be an object")
        supplied_digest = evidence_manifest_digest(supplied)
        recorded_digest = supplied.get("evidence_manifest_digest")
        if recorded_digest is not None and recorded_digest != supplied_digest:
            raise CounterfactualReplayError("supplied evidence manifest digest is inconsistent")
        if supplied_digest != manifest_digest:
            raise CounterfactualReplayError("supplied evidence manifest is stale")

    available = [run for run in manifest.get("runs", []) if isinstance(run, dict)]
    if identity is not None:
        selected = [run for run in available if run.get("identity") == identity]
        if not selected:
            raise CounterfactualReplayError(f"identity not found in evidence manifest: {identity}")
    elif include_nonstrategy_runs:
        selected = available
    else:
        selected = [run for run in available if run.get("include_in_strategy_aggregate") is True]
    selected.sort(key=lambda run: (str(run.get("matrix_timestamp")), str(run.get("mode"))))

    runs: list[dict[str, Any]] = []
    for manifest_run in selected:
        run_identity = str(manifest_run["identity"])
        started = manifest_run.get("run_started_utc")
        finished = manifest_run.get("finished_at")
        contract = resolve_replay_contract(
            run_identity,
            reports,
            contract_overrides_path,
            bundle_root,
        )
        resolved_contract = contract.get("contract")
        if (
            contract.get("status") in PARITY_GRADE_STATUSES
            and isinstance(resolved_contract, Mapping)
            and started
            and parse_timestamp(resolved_contract.get("run_started_utc"))
            != parse_timestamp(started)
        ):
            raise CounterfactualReplayError(
                f"replay contract start does not match Phase 19 for {run_identity}"
            )
        if not started or not finished:
            source = {
                "status": "missing",
                "rows": {kind: [] for kind in ("signals", "xgboost", "paper", "closed")},
                "source_digest": None,
                "bundle_digest": None,
                "coverage": {"coverage_passed": False, "reason": "run_window_missing"},
            }
        else:
            source = resolve_historical_sources(
                run_identity,
                str(started),
                str(finished),
                mode=str(manifest_run.get("mode")),
                logs_dir=logs,
                bundle_root=bundle_root,
                reported_row_counts=_reported_run_row_counts(manifest_run, reports),
                max_start_delay_seconds=max_start_delay_seconds,
                max_end_delay_seconds=max_end_delay_seconds,
                max_signal_gap_seconds=max_signal_gap_seconds,
            )
        runs.append(
            _run_replay(
                manifest_run,
                contract,
                source,
                inventory_only=inventory_only,
                include_nonstrategy=include_nonstrategy_runs,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )

    grade_counts = Counter(run["evidence_grade"] for run in runs)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_digest": manifest_digest,
        "policy": {
            "offline_only": True,
            "paper_only": True,
            "real_orders_allowed": False,
            "counterfactual_results_enter_promotion_gate": False,
        },
        "inputs": {
            "reports_dir": str(reports),
            "logs_dir": str(logs),
            "bundle_root": str(Path(bundle_root)),
            "inventory_only": inventory_only,
            "include_nonstrategy_runs": include_nonstrategy_runs,
            "replay_engine_sha256": _code_sha256(Path(__file__)),
            "coverage_gates": {
                "max_start_delay_seconds": max_start_delay_seconds,
                "max_end_delay_seconds": max_end_delay_seconds,
                "max_signal_gap_seconds": max_signal_gap_seconds,
            },
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "summary": {
            "run_count": len(runs),
            "strategy_run_count": sum(run["manifest_strategy_included"] for run in runs),
            "counterfactual_evidence_approved_count": sum(
                run["counterfactual_evidence_approved"] for run in runs
            ),
            "evidence_grade_counts": dict(sorted(grade_counts.items())),
            "parity_sample_gate_passed_count": sum(
                bool(run.get("parity", {}).get("promotion_parity_sample_gate")) for run in runs
            ),
        },
        "runs": runs,
    }
    report["replay_digest"] = counterfactual_replay_digest(report)
    return report


def write_counterfactual_report(report: Mapping[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report), indent=2), encoding="utf-8")
    return out


def format_inventory(report: Mapping[str, Any]) -> str:
    lines = [
        "Phase 20 Deterministic Counterfactual Replay",
        f"manifest_digest: {report['manifest_digest']}",
        f"run_count: {report['summary']['run_count']}",
    ]
    for run in report["runs"]:
        coverage = run.get("coverage", {})
        parity = run.get("parity", {})
        lines.append(
            f"  {run['identity']}: contract={run['contract_status']} "
            f"source={run['source_resolution_status']} coverage={coverage.get('coverage_passed')} "
            f"parity={parity.get('parity_status')} grade={run['evidence_grade']}"
        )
    lines.extend(
        [
            f"replay_digest: {report['replay_digest']}",
            "offline_paper_only_no_orders_no_phase17_phase18_promotion",
        ]
    )
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline deterministic signal-tick replay.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--identity")
    selection.add_argument("--all-strategy-runs", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--include-nonstrategy-runs", action="store_true")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--manifest")
    parser.add_argument("--evidence-overrides", default=str(DEFAULT_EVIDENCE_OVERRIDES))
    parser.add_argument("--contract-overrides", default=str(DEFAULT_CONTRACT_OVERRIDES))
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    parser.add_argument("--max-start-delay-seconds", type=_nonnegative_float, default=120.0)
    parser.add_argument("--max-end-delay-seconds", type=_nonnegative_float, default=120.0)
    parser.add_argument("--max-signal-gap-seconds", type=_nonnegative_float, default=180.0)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = summarize_counterfactual_replay(
            reports_dir=args.reports_dir,
            logs_dir=args.logs_dir,
            manifest_path=args.manifest,
            evidence_overrides_path=args.evidence_overrides,
            contract_overrides_path=args.contract_overrides,
            bundle_root=args.bundle_root,
            identity=args.identity,
            all_strategy_runs=args.all_strategy_runs,
            include_nonstrategy_runs=args.include_nonstrategy_runs,
            inventory_only=args.inventory_only,
            max_start_delay_seconds=args.max_start_delay_seconds,
            max_end_delay_seconds=args.max_end_delay_seconds,
            max_signal_gap_seconds=args.max_signal_gap_seconds,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        out = write_counterfactual_report(report, args.json_out)
        print(format_inventory(report))
        print(f"json_written: {out}")
        has_conflict = any(run["source_resolution_status"] == "conflicting" for run in report["runs"])
        return 1 if has_conflict else 0
    except (
        CounterfactualReplayError,
        ReplayBundleError,
        ReplayContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"counterfactual_replay_error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
