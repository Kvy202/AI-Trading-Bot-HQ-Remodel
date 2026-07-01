"""Audit XGBoost shadow rejections against later paper-trade outcomes.

This is a read-only analysis tool for trading inputs and model state. It reads
the shadow/live/trade CSV logs and never changes live_signals, executor state,
model artifacts, or trading behavior. The optional --json flag writes only a
report artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = BASE_DIR / "reports" / "xgboost_rejection_audit.json"

XGBOOST_LOG = "xgboost_signal_shadow.csv"
LIVE_SIGNALS_LOG = "live_signals.csv"
CLOSED_MASTER_LOG = "trades_closed.csv"
PAPER_GLOB = "trades_paper_*.csv"

OPEN_ACTIONS = {"BUY": "LONG", "SELL_SHORT": "SHORT"}
CLOSE_ACTIONS = {"SELL", "BUY_TO_COVER"}
Key = Tuple[str, datetime]
CloseKey = Tuple[str, datetime, str]


@dataclass(frozen=True)
class TradeOutcome:
    entry_key: Key
    signal_id: str
    symbol: str
    side: str
    entry_ts: str
    close_ts: str
    close_side: str
    realized_pnl: float


def _read_csv_rows(path: Path) -> tuple[str, List[Dict[str, str]]]:
    if not path.exists():
        return "missing", []
    if path.stat().st_size == 0:
        return "empty", []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [{str(k): "" if v is None else str(v) for k, v in row.items() if k is not None} for row in reader]
        return ("empty" if not rows else "ok"), rows
    except Exception as exc:
        return f"read_error:{type(exc).__name__}", []


def _read_glob_rows(logs_dir: Path, pattern: str) -> Dict[str, Any]:
    paths = sorted(logs_dir.glob(pattern))
    if not paths:
        return {"status": "missing", "files": [], "rows": []}

    rows: List[Dict[str, str]] = []
    statuses: Dict[str, str] = {}
    had_error = False
    for path in paths:
        status, file_rows = _read_csv_rows(path)
        statuses[path.name] = status
        if status.startswith("read_error"):
            had_error = True
        for idx, row in enumerate(file_rows):
            row["_source_file"] = path.name
            row["_row_index"] = str(idx)
            rows.append(row)

    if rows and had_error:
        status = "partial_read_error"
    elif rows:
        status = "ok"
    elif had_error:
        status = "read_error"
    else:
        status = "empty"
    return {
        "status": status,
        "files": [str(path) for path in paths],
        "file_statuses": statuses,
        "rows": rows,
    }


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _avg(values: Iterable[Any]) -> Optional[float]:
    nums = [x for x in (_float_or_none(v) for v in values) if x is not None]
    return None if not nums else sum(nums) / len(nums)


def _first_non_empty(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _confidence(row: Dict[str, str]) -> Optional[float]:
    return _float_or_none(_first_non_empty(row, "confidence", "xgboost_confidence"))


def _reject_reason(row: Dict[str, str]) -> str:
    value = _first_non_empty(row, "reject_reason", "reason")
    return value or "unknown"


def _row_signal_id(row: Dict[str, str]) -> str:
    return _first_non_empty(row, "signal_id", "decision_id")


def _decision(row: Dict[str, str]) -> str:
    if _truthy(row.get("would_reject")):
        return "rejected"
    if _truthy(row.get("would_confirm")):
        return "allowed"
    return "other"


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _parse_ts(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidates = [
        raw,
        raw.replace("Z", "+00:00"),
        raw.replace("+0000", "+00:00"),
    ]
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None


def _key_from_row(row: Dict[str, str], ts_keys: Sequence[str]) -> Optional[Key]:
    symbol = _normalize_symbol(row.get("symbol"))
    if not symbol:
        return None
    for ts_key in ts_keys:
        dt = _parse_ts(row.get(ts_key))
        if dt is not None:
            return symbol, dt
    return None


def _close_key_from_row(row: Dict[str, str]) -> Optional[CloseKey]:
    key = _key_from_row(row, ("ts", "timestamp"))
    side = _first_non_empty(row, "closed_side", "side").upper()
    if key is None or not side:
        return None
    return key[0], key[1], side


def _index_rows(rows: Iterable[Dict[str, str]], ts_keys: Sequence[str]) -> tuple[Dict[Key, List[Dict[str, str]]], int]:
    index: Dict[Key, List[Dict[str, str]]] = defaultdict(list)
    unparseable = 0
    for row in rows:
        key = _key_from_row(row, ts_keys)
        if key is None:
            unparseable += 1
            continue
        index[key].append(row)
    return index, unparseable


def _index_rows_by_id(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        signal_id = _row_signal_id(row)
        if signal_id:
            index[signal_id].append(row)
    return index


def _direction_from_score(value: Any) -> str:
    score = _float_or_none(value)
    if score is None:
        return ""
    if score > 0:
        return "LONG"
    if score < 0:
        return "SHORT"
    return ""


def _signal_direction(row: Dict[str, str]) -> str:
    raw = _first_non_empty(row, "side_hint", "existing_signal").upper()
    if raw in {"LONG", "BUY", "BULL"}:
        return "LONG"
    if raw in {"SHORT", "SELL", "BEAR"}:
        return "SHORT"
    return _direction_from_score(_first_non_empty(row, "p_meta", "existing_score"))


def _sort_paper_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    max_dt = datetime.max.replace(tzinfo=timezone.utc)

    def sort_key(row: Dict[str, str]) -> tuple[datetime, str, int]:
        dt = _parse_ts(row.get("ts")) or max_dt
        try:
            row_idx = int(row.get("_row_index", "0"))
        except Exception:
            row_idx = 0
        return dt, row.get("_source_file", ""), row_idx

    return sorted(rows, key=sort_key)


def _build_trade_outcomes(
    paper_rows: List[Dict[str, str]],
    closed_rows: List[Dict[str, str]],
) -> tuple[Dict[str, List[TradeOutcome]], Dict[Key, List[TradeOutcome]], Dict[str, int]]:
    diagnostics: Counter[str] = Counter()
    closed_index: Dict[CloseKey, List[Dict[str, str]]] = defaultdict(list)
    closed_id_index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in closed_rows:
        signal_id = _row_signal_id(row)
        if signal_id:
            closed_id_index[signal_id].append(row)
        close_key = _close_key_from_row(row)
        if close_key is None:
            diagnostics["closed_trade_unparseable_key"] += 1
            continue
        closed_index[close_key].append(row)

    positions: Dict[str, Dict[str, Any]] = {}
    outcomes_by_id: Dict[str, List[TradeOutcome]] = defaultdict(list)
    outcomes_by_key: Dict[Key, List[TradeOutcome]] = defaultdict(list)
    for row in _sort_paper_rows(paper_rows):
        key = _key_from_row(row, ("ts", "timestamp"))
        action = _first_non_empty(row, "side").upper()
        if key is None or not action:
            diagnostics["paper_trade_unparseable_key"] += 1
            continue
        symbol, dt = key
        reason = _first_non_empty(row, "reason")

        if action in OPEN_ACTIONS:
            side = OPEN_ACTIONS[action]
            existing = positions.get(symbol)
            if existing is None:
                positions[symbol] = {
                    "entry_key": key,
                    "entry_ts": _first_non_empty(row, "ts", "timestamp"),
                    "signal_id": _row_signal_id(row),
                    "side": side,
                    "fresh_entry": reason.upper().startswith("ENTRY"),
                    "scale_ins": 0,
                }
            elif existing.get("side") == side:
                existing["scale_ins"] = int(existing.get("scale_ins", 0)) + 1
            else:
                diagnostics["paper_position_overlap"] += 1
                positions[symbol] = {
                    "entry_key": key,
                    "entry_ts": _first_non_empty(row, "ts", "timestamp"),
                    "signal_id": _row_signal_id(row),
                    "side": side,
                    "fresh_entry": reason.upper().startswith("ENTRY"),
                    "scale_ins": 0,
                    "unsafe": True,
                }
            continue

        if action not in CLOSE_ACTIONS:
            continue

        position = positions.pop(symbol, None)
        if position is None:
            diagnostics["paper_orphan_close"] += 1
            continue
        close_key = (symbol, dt, action)
        closed_matches = closed_index.get(close_key, [])
        if position.get("unsafe"):
            diagnostics["paper_position_sequence_unsafe"] += 1
            continue
        if not position.get("fresh_entry"):
            diagnostics["paper_open_reason_not_entry"] += 1
            continue
        if int(position.get("scale_ins", 0)) > 0:
            diagnostics["paper_scale_in_attribution_unsafe"] += 1
            continue

        entry_signal_id = str(position.get("signal_id") or "")
        if entry_signal_id:
            id_matches = [
                r
                for r in closed_id_index.get(entry_signal_id, [])
                if _normalize_symbol(r.get("symbol")) == symbol
            ]
            if len(id_matches) == 1:
                closed = id_matches[0]
            elif len(id_matches) > 1:
                diagnostics["closed_trade_duplicate_signal_id"] += 1
                continue
            elif not closed_matches:
                diagnostics["closed_trade_missing_for_signal_id"] += 1
                continue
            elif len(closed_matches) == 1:
                closed = closed_matches[0]
            else:
                diagnostics["closed_trade_duplicate_key"] += 1
                continue
        else:
            if not closed_matches:
                diagnostics["closed_trade_missing_for_paper_close"] += 1
                continue
            if len(closed_matches) > 1:
                diagnostics["closed_trade_duplicate_key"] += 1
                continue
            closed = closed_matches[0]
        pnl = _float_or_none(_first_non_empty(closed, "realized_pnl", "pnl"))
        if pnl is None:
            diagnostics["closed_trade_missing_pnl"] += 1
            continue
        entry_key = position["entry_key"]
        outcome = TradeOutcome(
            entry_key=entry_key,
            signal_id=entry_signal_id,
            symbol=symbol,
            side=str(position["side"]),
            entry_ts=str(position["entry_ts"]),
            close_ts=_first_non_empty(closed, "ts", "timestamp"),
            close_side=action,
            realized_pnl=pnl,
        )
        outcomes_by_key[entry_key].append(outcome)
        if entry_signal_id:
            outcomes_by_id[entry_signal_id].append(outcome)

    diagnostics["paper_open_positions_without_close"] += len(positions)
    for signal_id, rows in outcomes_by_id.items():
        if len(rows) > 1:
            diagnostics["paper_duplicate_outcome_signal_id"] += 1
    for entry_key, rows in outcomes_by_key.items():
        if len(rows) > 1:
            diagnostics["paper_duplicate_outcome_entry_key"] += 1
    return outcomes_by_id, outcomes_by_key, dict(diagnostics)


def _pnl_stats(values: Iterable[float]) -> Dict[str, Optional[float] | int]:
    pnls = list(values)
    if not pnls:
        return {
            "count": 0,
            "total_pnl": 0.0,
            "average_pnl": None,
            "win_rate": None,
        }
    return {
        "count": len(pnls),
        "total_pnl": sum(pnls),
        "average_pnl": sum(pnls) / len(pnls),
        "win_rate": sum(1 for pnl in pnls if pnl > 0) / len(pnls),
    }


def _summarize_trade_outcomes(
    xgboost_rows: List[Dict[str, str]],
    live_rows: List[Dict[str, str]],
    paper_rows: List[Dict[str, str]],
    closed_rows: List[Dict[str, str]],
    *,
    live_status: str,
    paper_status: str,
    closed_status: str,
) -> Dict[str, Any]:
    live_index, live_unparseable = _index_rows(live_rows, ("ts", "timestamp"))
    live_id_index = _index_rows_by_id(live_rows)
    xgb_keys = [_key_from_row(row, ("timestamp", "ts")) for row in xgboost_rows]
    xgb_key_counts = Counter(key for key in xgb_keys if key is not None)
    xgb_ids = [_row_signal_id(row) for row in xgboost_rows]
    xgb_id_counts = Counter(signal_id for signal_id in xgb_ids if signal_id)
    outcomes_by_id, outcomes_by_key, trade_diagnostics = _build_trade_outcomes(paper_rows, closed_rows)

    unmatched_reasons: Counter[str] = Counter()
    matched_pnls: Dict[str, List[float]] = {"allowed": [], "rejected": []}
    matched_closed_trade_count = 0
    id_matched_count = 0
    fallback_matched_count = 0
    decision_rows = 0
    id_decision_rows = 0
    missing_id_count = 0
    unmatched_due_missing_id = 0
    unmatched_due_missing_trade = 0
    unmatched_allowed = 0
    unmatched_rejected = 0

    def mark_unmatched(decision: str, reason: str) -> None:
        nonlocal unmatched_allowed, unmatched_rejected
        unmatched_reasons[reason] += 1
        if decision == "allowed":
            unmatched_allowed += 1
        elif decision == "rejected":
            unmatched_rejected += 1

    def mark_missing_trade(decision: str, reason: str) -> None:
        nonlocal unmatched_due_missing_trade
        unmatched_due_missing_trade += 1
        mark_unmatched(decision, reason)

    def record_match(decision: str, outcome: TradeOutcome, method: str) -> None:
        nonlocal matched_closed_trade_count, id_matched_count, fallback_matched_count
        matched_closed_trade_count += 1
        if method == "id":
            id_matched_count += 1
        else:
            fallback_matched_count += 1
        matched_pnls[decision].append(outcome.realized_pnl)

    for row, key, signal_id in zip(xgboost_rows, xgb_keys, xgb_ids):
        decision = _decision(row)
        if decision not in {"allowed", "rejected"}:
            continue
        decision_rows += 1
        if signal_id:
            id_decision_rows += 1
            if xgb_id_counts[signal_id] > 1:
                mark_unmatched(decision, "xgboost_duplicate_signal_id")
                continue
            live_matches = live_id_index.get(signal_id, [])
            if len(live_matches) > 1:
                mark_unmatched(decision, "live_signal_duplicate_signal_id")
                continue
            outcome_matches = outcomes_by_id.get(signal_id, [])
            if not outcome_matches:
                mark_missing_trade(decision, "paper_entry_or_closed_trade_missing")
                continue
            if len(outcome_matches) > 1:
                mark_unmatched(decision, "paper_duplicate_outcome_signal_id")
                continue
            outcome = outcome_matches[0]
            live = live_matches[0] if live_matches else row
            expected_direction = _signal_direction(live) or _signal_direction(row)
            if expected_direction and outcome.side != expected_direction:
                mark_unmatched(decision, "paper_entry_side_mismatch")
                continue
            record_match(decision, outcome, "id")
            continue

        missing_id_count += 1
        if key is None:
            unmatched_due_missing_id += 1
            mark_unmatched(decision, "xgboost_unparseable_key")
            continue
        if xgb_key_counts[key] > 1:
            unmatched_due_missing_id += 1
            mark_unmatched(decision, "xgboost_duplicate_key")
            continue

        live_matches = live_index.get(key, [])
        if not live_matches:
            mark_unmatched(decision, "live_signal_missing")
            continue
        if len(live_matches) > 1:
            mark_unmatched(decision, "live_signal_duplicate_key")
            continue
        live = live_matches[0]
        if not _truthy(live.get("allow")):
            mark_unmatched(decision, "live_signal_not_allowed")
            continue

        outcome_matches = outcomes_by_key.get(key, [])
        if not outcome_matches:
            mark_missing_trade(decision, "paper_entry_or_closed_trade_missing")
            continue
        if len(outcome_matches) > 1:
            mark_unmatched(decision, "paper_duplicate_outcome_entry_key")
            continue

        outcome = outcome_matches[0]
        expected_direction = _signal_direction(live) or _signal_direction(row)
        if expected_direction and outcome.side != expected_direction:
            mark_unmatched(decision, "paper_entry_side_mismatch")
            continue

        record_match(decision, outcome, "fallback")

    unmatched_decisions = unmatched_allowed + unmatched_rejected
    if id_matched_count > 0 and fallback_matched_count > 0:
        join_method = "signal_id+timestamp_symbol_fallback"
    elif id_matched_count > 0:
        join_method = "signal_id"
    elif fallback_matched_count > 0:
        join_method = "timestamp_symbol_fallback"
    elif id_decision_rows > 0 and missing_id_count > 0:
        join_method = "signal_id+timestamp_symbol_fallback"
    elif id_decision_rows > 0:
        join_method = "signal_id"
    elif missing_id_count > 0:
        join_method = "timestamp_symbol_fallback"
    else:
        join_method = "none"

    if not xgboost_rows:
        status = "not_available"
        message = "No XGBoost shadow rows were available for trade outcome joining."
    elif live_status != "ok":
        status = "unreliable"
        message = (
            "Trade outcome join is not reliable: live_signals.csv is "
            f"{live_status}, so shadow rows cannot be tied to executor inputs."
        )
    elif paper_status != "ok":
        status = "unreliable"
        message = (
            "Trade outcome join is not reliable: trades_paper_*.csv is "
            f"{paper_status}, so entries cannot be reconstructed."
        )
    elif closed_status != "ok":
        status = "unreliable"
        message = (
            "Trade outcome join is not reliable: trades_closed.csv is "
            f"{closed_status}, so realized PnL cannot be verified."
        )
    elif matched_closed_trade_count == 0 and decision_rows > 0:
        status = "unreliable"
        message = (
            "Trade outcome join is not reliable: no XGBoost decision rows matched "
            "a unique paper entry and closed trade by signal_id or conservative timestamp+symbol fallback."
        )
    elif unmatched_decisions > 0:
        status = "partial"
        message = (
            "Trade outcome join is partial and not reliable for unmatched rows; "
            "only uniquely matched closed trades are included in PnL."
        )
    else:
        status = "ok"
        message = "Trade outcome join is reliable for all XGBoost allow/reject decision rows."

    return {
        "status": status,
        "message": message,
        "join_method": join_method,
        "live_unparseable_count": live_unparseable,
        "trade_diagnostics": trade_diagnostics,
        "matched_closed_trade_count": matched_closed_trade_count,
        "id_matched_count": id_matched_count,
        "fallback_matched_count": fallback_matched_count,
        "missing_id_count": missing_id_count,
        "unmatched_due_missing_id": unmatched_due_missing_id,
        "unmatched_due_missing_trade": unmatched_due_missing_trade,
        "matched_closed_trade_pnl": {
            "allowed": _pnl_stats(matched_pnls["allowed"]),
            "rejected": _pnl_stats(matched_pnls["rejected"]),
        },
        "unmatched_xgboost_rows": max(0, len(xgboost_rows) - matched_closed_trade_count),
        "unmatched_decision_rows": unmatched_decisions,
        "unmatched_allowed_signal_count": unmatched_allowed,
        "unmatched_rejected_signal_count": unmatched_rejected,
        "unmatched_reason_counts": dict(unmatched_reasons),
    }


def summarize_audit(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> Dict[str, Any]:
    root = Path(logs_dir)
    xgb_status, xgb_rows = _read_csv_rows(root / XGBOOST_LOG)
    live_status, live_rows = _read_csv_rows(root / LIVE_SIGNALS_LOG)
    closed_status, closed_rows = _read_csv_rows(root / CLOSED_MASTER_LOG)
    paper = _read_glob_rows(root, PAPER_GLOB)
    paper_rows = paper["rows"]

    would_reject_rows = [row for row in xgb_rows if _truthy(row.get("would_reject"))]
    would_confirm_rows = [row for row in xgb_rows if _truthy(row.get("would_confirm"))]
    actually_rejected_rows = [row for row in xgb_rows if _truthy(row.get("actually_rejected"))]
    reject_reason_counts = Counter(_reject_reason(row) for row in would_reject_rows)

    trade_join = _summarize_trade_outcomes(
        xgb_rows,
        live_rows,
        paper_rows,
        closed_rows,
        live_status=live_status,
        paper_status=str(paper["status"]),
        closed_status=closed_status,
    )

    return {
        "logs_dir": str(root),
        "files": {
            XGBOOST_LOG: {"status": xgb_status, "rows": len(xgb_rows), "path": str(root / XGBOOST_LOG)},
            LIVE_SIGNALS_LOG: {"status": live_status, "rows": len(live_rows), "path": str(root / LIVE_SIGNALS_LOG)},
            CLOSED_MASTER_LOG: {"status": closed_status, "rows": len(closed_rows), "path": str(root / CLOSED_MASTER_LOG)},
            PAPER_GLOB: {
                "status": paper["status"],
                "rows": len(paper_rows),
                "files": paper["files"],
                "file_statuses": paper.get("file_statuses", {}),
            },
        },
        "total_xgboost_rows": len(xgb_rows),
        "would_reject_count": len(would_reject_rows),
        "actually_rejected_count": len(actually_rejected_rows),
        "would_confirm_count": len(would_confirm_rows),
        "reject_reason_counts": dict(reject_reason_counts),
        "allowed_signal_count": len(would_confirm_rows),
        "rejected_signal_count": len(would_reject_rows),
        "neutral_signal_count": len(xgb_rows) - len(would_confirm_rows) - len(would_reject_rows),
        "average_confidence_allowed": _avg(_confidence(row) for row in would_confirm_rows),
        "average_confidence_rejected": _avg(_confidence(row) for row in would_reject_rows),
        "direction_mismatch_count": reject_reason_counts.get("direction_mismatch", 0),
        "low_confidence_count": reject_reason_counts.get("low_confidence", 0),
        "trade_outcome_join": trade_join,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_text_summary(summary: Dict[str, Any]) -> str:
    files = summary["files"]
    join = summary["trade_outcome_join"]
    allowed_pnl = join["matched_closed_trade_pnl"]["allowed"]
    rejected_pnl = join["matched_closed_trade_pnl"]["rejected"]
    lines = [
        "XGBoost Rejection Outcome Audit",
        f"Logs: {summary['logs_dir']}",
        "",
        "Inputs",
        f"  {XGBOOST_LOG}: {files[XGBOOST_LOG]['status']} rows={files[XGBOOST_LOG]['rows']}",
        f"  {LIVE_SIGNALS_LOG}: {files[LIVE_SIGNALS_LOG]['status']} rows={files[LIVE_SIGNALS_LOG]['rows']}",
        f"  {CLOSED_MASTER_LOG}: {files[CLOSED_MASTER_LOG]['status']} rows={files[CLOSED_MASTER_LOG]['rows']}",
        f"  {PAPER_GLOB}: {files[PAPER_GLOB]['status']} rows={files[PAPER_GLOB]['rows']}",
        "",
        "XGBoost Decisions",
        "  allowed/rejected here mean XGBoost would_confirm/would_reject shadow decisions.",
        f"  total_xgboost_rows: {summary['total_xgboost_rows']}",
        f"  would_reject_count: {summary['would_reject_count']}",
        f"  actually_rejected_count: {summary['actually_rejected_count']}",
        f"  would_confirm_count: {summary['would_confirm_count']}",
        f"  reject_reason_counts: {summary['reject_reason_counts']}",
        f"  allowed_signal_count: {summary['allowed_signal_count']}",
        f"  rejected_signal_count: {summary['rejected_signal_count']}",
        f"  average_confidence_allowed: {_fmt(summary['average_confidence_allowed'])}",
        f"  average_confidence_rejected: {_fmt(summary['average_confidence_rejected'])}",
        f"  direction_mismatch_count: {summary['direction_mismatch_count']}",
        f"  low_confidence_count: {summary['low_confidence_count']}",
        "",
        "Trade Outcome Join",
        f"  status: {join['status']}",
        f"  join_method: {join['join_method']}",
        f"  message: {join['message']}",
        f"  matched_closed_trade_count: {join['matched_closed_trade_count']}",
        f"  id_matched_count: {join['id_matched_count']}",
        f"  fallback_matched_count: {join['fallback_matched_count']}",
        f"  missing_id_count: {join['missing_id_count']}",
        f"  unmatched_due_missing_id: {join['unmatched_due_missing_id']}",
        f"  unmatched_due_missing_trade: {join['unmatched_due_missing_trade']}",
        f"  allowed_matched_count: {allowed_pnl['count']}",
        f"  allowed_total_pnl: {_fmt(allowed_pnl['total_pnl'])}",
        f"  allowed_average_pnl: {_fmt(allowed_pnl['average_pnl'])}",
        f"  allowed_win_rate: {_fmt(allowed_pnl['win_rate'])}",
        f"  rejected_matched_count: {rejected_pnl['count']}",
        f"  rejected_total_pnl: {_fmt(rejected_pnl['total_pnl'])}",
        f"  rejected_average_pnl: {_fmt(rejected_pnl['average_pnl'])}",
        f"  rejected_win_rate: {_fmt(rejected_pnl['win_rate'])}",
        f"  unmatched_xgboost_rows: {join['unmatched_xgboost_rows']}",
        f"  unmatched_decision_rows: {join['unmatched_decision_rows']}",
        f"  unmatched_allowed_signal_count: {join['unmatched_allowed_signal_count']}",
        f"  unmatched_rejected_signal_count: {join['unmatched_rejected_signal_count']}",
        f"  unmatched_reason_counts: {join['unmatched_reason_counts']}",
    ]
    return "\n".join(lines)


def write_json_summary(summary: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit XGBoost shadow rejection outcomes")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--json", action="store_true", help="Write reports/xgboost_rejection_audit.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_audit(args.logs_dir)
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
