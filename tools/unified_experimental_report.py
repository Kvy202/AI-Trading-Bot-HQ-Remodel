"""Unified read-only report for experimental systems and paper lineage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

THIS = Path(__file__).resolve()
BASE_DIR = THIS.parents[1] if THIS.parent.name == "tools" else THIS.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.audit_xgboost_rejections import summarize_audit  # noqa: E402

DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = BASE_DIR / "reports" / "unified_experimental_report.json"

ISOLATION_LOG = "isolation_forest_shadow.csv"
XGBOOST_LOG = "xgboost_signal_shadow.csv"
SURVIVAL_LOG = "survival_exit_shadow.csv"
ADVANCED_RISK_LOG = "advanced_risk_shadow.csv"
LIVE_SIGNALS_LOG = "live_signals.csv"
CLOSED_MASTER_LOG = "trades_closed.csv"
PAPER_GLOB = "trades_paper_*.csv"
CLOSED_DATED_GLOB = "trades_closed_*.csv"


def _read_csv_rows(path: Path) -> tuple[str, List[Dict[str, str]]]:
    if not path.exists():
        return "missing", []
    if path.stat().st_size == 0:
        return "empty", []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                {str(k): "" if v is None else str(v) for k, v in row.items() if k is not None}
                for row in reader
            ]
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
    return {"status": status, "files": [str(path) for path in paths], "file_statuses": statuses, "rows": rows}


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


def _numeric_values(values: Iterable[Any]) -> List[float]:
    return [v for v in (_float_or_none(x) for x in values) if v is not None]


def _avg(values: Iterable[Any]) -> Optional[float]:
    nums = _numeric_values(values)
    return None if not nums else sum(nums) / len(nums)


def _percentile(values: Iterable[Any], percentile: float) -> Optional[float]:
    nums = sorted(_numeric_values(values))
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    rank = (max(0.0, min(100.0, float(percentile))) / 100.0) * (len(nums) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(nums) - 1)
    weight = rank - lower
    return nums[lower] + ((nums[upper] - nums[lower]) * weight)


def _first_non_empty(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _count_with_signal_id(rows: Iterable[Dict[str, str]]) -> int:
    return sum(1 for row in rows if _first_non_empty(row, "signal_id", "decision_id"))


def _signal_ids(rows: Iterable[Dict[str, str]]) -> set[str]:
    return {sid for sid in (_first_non_empty(row, "signal_id", "decision_id") for row in rows) if sid}


def _top_reason_counts(rows: Iterable[Dict[str, str]], *keys: str, limit: int = 5) -> Dict[str, int]:
    counts = Counter((_first_non_empty(row, *keys) or "unknown") for row in rows)
    return dict(counts.most_common(limit))


def _dedupe_rows(rows: Iterable[Dict[str, str]], keys: tuple[str, ...]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(_first_non_empty(row, k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _score_distribution(values: Iterable[Any], prefix: str) -> Dict[str, Optional[float]]:
    nums = _numeric_values(values)
    return {
        f"min_{prefix}": None if not nums else min(nums),
        f"max_{prefix}": None if not nums else max(nums),
        f"average_{prefix}": None if not nums else sum(nums) / len(nums),
        f"p10_{prefix}": _percentile(nums, 10),
        f"p50_{prefix}": _percentile(nums, 50),
        f"p90_{prefix}": _percentile(nums, 90),
    }


def summarize_isolation(logs_dir: Path, rows: List[Dict[str, str]], status: str) -> Dict[str, Any]:
    total_rows = len(rows)
    abnormal_count = sum(
        1
        for row in rows
        if str(row.get("anomaly_status", "")).strip().lower() in {"anomaly", "abnormal"}
    )
    would_block_count = sum(1 for row in rows if _truthy(row.get("would_block")))
    actually_blocked_count = sum(1 for row in rows if _truthy(row.get("actually_blocked")))
    return {
        "file": str(logs_dir / ISOLATION_LOG),
        "file_status": status,
        "total_rows": total_rows,
        "abnormal_count": abnormal_count,
        "would_block_count": would_block_count,
        "actually_blocked_count": actually_blocked_count,
        "would_block_rate": 0.0 if total_rows == 0 else would_block_count / total_rows,
        "actual_block_rate": 0.0 if total_rows == 0 else actually_blocked_count / total_rows,
        "latest_anomaly_score": _float_or_none(rows[-1].get("anomaly_score")) if rows else None,
        **_score_distribution((row.get("anomaly_score") for row in rows), "anomaly_score"),
        "top_reasons": _top_reason_counts(rows, "reason"),
    }


def _confidence(row: Dict[str, str]) -> Optional[float]:
    return _float_or_none(_first_non_empty(row, "confidence", "xgboost_confidence"))


def summarize_xgboost(logs_dir: Path, rows: List[Dict[str, str]], status: str) -> Dict[str, Any]:
    total_rows = len(rows)
    would_confirm_rows = [row for row in rows if _truthy(row.get("would_confirm"))]
    would_reject_rows = [row for row in rows if _truthy(row.get("would_reject"))]
    actually_rejected_count = sum(1 for row in rows if _truthy(row.get("actually_rejected")))
    reject_reason_counts = Counter(_first_non_empty(row, "reject_reason", "reason") or "unknown" for row in would_reject_rows)
    return {
        "file": str(logs_dir / XGBOOST_LOG),
        "file_status": status,
        "total_rows": total_rows,
        "would_confirm_count": len(would_confirm_rows),
        "would_reject_count": len(would_reject_rows),
        "actually_rejected_count": actually_rejected_count,
        "would_reject_rate": 0.0 if total_rows == 0 else len(would_reject_rows) / total_rows,
        "actual_reject_rate": 0.0 if total_rows == 0 else actually_rejected_count / total_rows,
        "reject_reason_counts": dict(reject_reason_counts),
        "average_confidence_allowed": _avg(_confidence(row) for row in would_confirm_rows),
        "average_confidence_confirmed": _avg(_confidence(row) for row in would_confirm_rows),
        "average_confidence_rejected": _avg(_confidence(row) for row in would_reject_rows),
        "direction_mismatch_count": reject_reason_counts.get("direction_mismatch", 0),
        "low_confidence_count": reject_reason_counts.get("low_confidence", 0),
    }


def summarize_survival(logs_dir: Path, rows: List[Dict[str, str]], status: str) -> Dict[str, Any]:
    total_rows = len(rows)
    would_exit_early_count = sum(1 for row in rows if _truthy(row.get("would_exit_early")))
    actually_exited_count = sum(1 for row in rows if _truthy(row.get("actually_exited")))
    exited_rows = [row for row in rows if _truthy(row.get("actually_exited"))]
    return {
        "file": str(logs_dir / SURVIVAL_LOG),
        "file_status": status,
        "total_rows": total_rows,
        "would_exit_early_count": would_exit_early_count,
        "actually_exited_count": actually_exited_count,
        "would_exit_rate": 0.0 if total_rows == 0 else would_exit_early_count / total_rows,
        "actual_exit_rate": 0.0 if total_rows == 0 else actually_exited_count / total_rows,
        "average_risk_score": _avg(row.get("survival_risk_score") for row in rows),
        "latest_risk_score": _float_or_none(rows[-1].get("survival_risk_score")) if rows else None,
        "exit_reason_counts": _top_reason_counts(exited_rows, "exit_reason", "reason"),
    }


def summarize_advanced_risk(logs_dir: Path, rows: List[Dict[str, str]], status: str) -> Dict[str, Any]:
    total_rows = len(rows)
    would_block_count = sum(1 for row in rows if _truthy(row.get("would_block")))
    actually_blocked_count = sum(1 for row in rows if _truthy(row.get("actually_blocked")))
    would_pause_count = sum(1 for row in rows if _truthy(row.get("would_pause")))
    actually_paused_count = sum(1 for row in rows if _truthy(row.get("actually_paused")))
    would_reduce_size_count = sum(1 for row in rows if _truthy(row.get("would_reduce_size")))
    actually_reduced_count = sum(1 for row in rows if _truthy(row.get("actually_reduced")))
    return {
        "file": str(logs_dir / ADVANCED_RISK_LOG),
        "file_status": status,
        "total_rows": total_rows,
        "would_block_count": would_block_count,
        "actually_blocked_count": actually_blocked_count,
        "would_block_rate": 0.0 if total_rows == 0 else would_block_count / total_rows,
        "actual_block_rate": 0.0 if total_rows == 0 else actually_blocked_count / total_rows,
        "would_pause_count": would_pause_count,
        "actually_paused_count": actually_paused_count,
        "would_reduce_size_count": would_reduce_size_count,
        "actually_reduced_count": actually_reduced_count,
        "average_risk_score": _avg(row.get("risk_score") for row in rows),
        "top_reasons": _top_reason_counts(rows, "top_reason", "reasons"),
    }


def _read_trade_inputs(logs_dir: Path) -> Dict[str, Any]:
    live_status, live_rows = _read_csv_rows(logs_dir / LIVE_SIGNALS_LOG)
    paper = _read_glob_rows(logs_dir, PAPER_GLOB)
    closed_master_status, closed_master_rows = _read_csv_rows(logs_dir / CLOSED_MASTER_LOG)
    for idx, row in enumerate(closed_master_rows):
        row["_source_file"] = CLOSED_MASTER_LOG
        row["_row_index"] = str(idx)
    closed_dated = _read_glob_rows(logs_dir, CLOSED_DATED_GLOB)
    closed_raw_rows = list(closed_master_rows) + list(closed_dated["rows"])
    closed_rows = _dedupe_rows(
        closed_raw_rows,
        ("ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason", "signal_id"),
    )
    return {
        "live_status": live_status,
        "live_rows": live_rows,
        "paper_status": paper["status"],
        "paper_rows": paper["rows"],
        "paper_files": paper["files"],
        "closed_master_status": closed_master_status,
        "closed_dated_status": closed_dated["status"],
        "closed_rows": closed_rows,
        "closed_raw_rows": closed_raw_rows,
        "closed_files": [str(logs_dir / CLOSED_MASTER_LOG)] + list(closed_dated["files"]),
    }


def summarize_trade_lineage(trade_inputs: Dict[str, Any]) -> Dict[str, Any]:
    live_rows = trade_inputs["live_rows"]
    paper_rows = trade_inputs["paper_rows"]
    closed_rows = trade_inputs["closed_rows"]
    live_with_id = _count_with_signal_id(live_rows)
    paper_with_id = _count_with_signal_id(paper_rows)
    closed_with_id = _count_with_signal_id(closed_rows)
    paper_ids = _signal_ids(paper_rows)
    closed_ids = _signal_ids(closed_rows)
    live_ids = _signal_ids(live_rows)
    return {
        "live_signal_rows": len(live_rows),
        "live_signal_rows_with_signal_id": live_with_id,
        "paper_trade_rows": len(paper_rows),
        "paper_trade_rows_with_signal_id": paper_with_id,
        "closed_trade_rows": len(closed_rows),
        "closed_trade_rows_with_signal_id": closed_with_id,
        "signal_id_missing_counts": {
            "live_signals": len(live_rows) - live_with_id,
            "paper_trades": len(paper_rows) - paper_with_id,
            "closed_trades": len(closed_rows) - closed_with_id,
        },
        "matched_closed_trade_count_by_signal_id": sum(
            1 for row in closed_rows if _first_non_empty(row, "signal_id", "decision_id") in paper_ids
        ),
        "unique_live_signal_ids": len(live_ids),
        "unique_paper_trade_signal_ids": len(paper_ids),
        "unique_closed_trade_signal_ids": len(closed_ids),
        "file_statuses": {
            LIVE_SIGNALS_LOG: trade_inputs["live_status"],
            PAPER_GLOB: trade_inputs["paper_status"],
            CLOSED_MASTER_LOG: trade_inputs["closed_master_status"],
            CLOSED_DATED_GLOB: trade_inputs["closed_dated_status"],
        },
    }


def summarize_pnl(closed_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    pnl_rows: List[tuple[float, Dict[str, str]]] = []
    for row in closed_rows:
        pnl = _float_or_none(_first_non_empty(row, "realized_pnl", "pnl"))
        if pnl is not None:
            pnl_rows.append((pnl, row))
    pnls = [pnl for pnl, _row in pnl_rows]
    best = max(pnl_rows, key=lambda item: item[0], default=None)
    worst = min(pnl_rows, key=lambda item: item[0], default=None)
    return {
        "closed_trade_count": len(pnl_rows),
        "total_pnl": sum(pnls) if pnls else 0.0,
        "average_pnl": None if not pnls else sum(pnls) / len(pnls),
        "win_rate": None if not pnls else sum(1 for pnl in pnls if pnl > 0.0) / len(pnls),
        "best_trade": None if best is None else _trade_brief(best[0], best[1]),
        "worst_trade": None if worst is None else _trade_brief(worst[0], worst[1]),
    }


def _trade_brief(pnl: float, row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "ts": _first_non_empty(row, "ts", "timestamp"),
        "symbol": _first_non_empty(row, "symbol"),
        "closed_side": _first_non_empty(row, "closed_side", "side"),
        "realized_pnl": pnl,
        "signal_id": _first_non_empty(row, "signal_id", "decision_id"),
    }


def summarize_safety(
    isolation: Dict[str, Any],
    xgboost: Dict[str, Any],
    survival: Dict[str, Any],
    advanced_risk: Dict[str, Any],
    trade_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    paper_modes = Counter((_first_non_empty(row, "mode") or "unknown").upper() for row in trade_inputs["paper_rows"])
    live_like_modes = {mode: count for mode, count in paper_modes.items() if mode not in {"", "PAPER", "UNKNOWN"}}
    if live_like_modes:
        inferred = "live_or_real"
    elif paper_modes.get("PAPER", 0) > 0:
        inferred = "paper"
    else:
        inferred = "unknown"

    actual_counts = {
        "actually_blocked": int(isolation["actually_blocked_count"]) + int(advanced_risk["actually_blocked_count"]),
        "actually_rejected": int(xgboost["actually_rejected_count"]),
        "actually_exited": int(survival["actually_exited_count"]),
        "actually_paused": int(advanced_risk["actually_paused_count"]),
        "actually_reduced": int(advanced_risk["actually_reduced_count"]),
    }
    warnings: List[str] = []
    if any(count > 0 for count in actual_counts.values()):
        warnings.append("active_or_blocking_behavior_detected_in_shadow_report")
    if inferred == "live_or_real":
        warnings.append("non_paper_trade_mode_detected")
    return {
        "inferred_trade_mode": inferred,
        "paper_trade_mode_counts": dict(paper_modes),
        "actual_behavior_counts": actual_counts,
        "shadow_only_warning": any(count > 0 for count in actual_counts.values()),
        "warnings": warnings,
    }


def summarize_xgboost_outcome(logs_dir: Path) -> Dict[str, Any]:
    audit = summarize_audit(logs_dir)
    join = audit.get("trade_outcome_join", {})
    return {
        "would_confirm_matched_count": audit.get("would_confirm_matched_count", 0),
        "would_confirm_average_pnl": audit.get("would_confirm_average_pnl"),
        "would_confirm_win_rate": audit.get("would_confirm_win_rate"),
        "would_reject_matched_count": audit.get("would_reject_matched_count", 0),
        "would_reject_average_pnl": audit.get("would_reject_average_pnl"),
        "would_reject_win_rate": audit.get("would_reject_win_rate"),
        "join_method": join.get("join_method", "none"),
        "matched_closed_trade_count": join.get("matched_closed_trade_count", 0),
        "unmatched_xgboost_rows": join.get("unmatched_xgboost_rows", 0),
        "unmatched_decision_rows": join.get("unmatched_decision_rows", 0),
        "unmatched_reason_counts": join.get("unmatched_reason_counts", {}),
        "status": join.get("status", "unknown"),
        "message": join.get("message", ""),
    }


def summarize_unified(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> Dict[str, Any]:
    root = Path(logs_dir)
    isolation_status, isolation_rows = _read_csv_rows(root / ISOLATION_LOG)
    xgboost_status, xgboost_rows = _read_csv_rows(root / XGBOOST_LOG)
    survival_status, survival_rows = _read_csv_rows(root / SURVIVAL_LOG)
    advanced_risk_status, advanced_risk_rows = _read_csv_rows(root / ADVANCED_RISK_LOG)
    trade_inputs = _read_trade_inputs(root)

    isolation = summarize_isolation(root, isolation_rows, isolation_status)
    xgboost = summarize_xgboost(root, xgboost_rows, xgboost_status)
    survival = summarize_survival(root, survival_rows, survival_status)
    advanced_risk = summarize_advanced_risk(root, advanced_risk_rows, advanced_risk_status)

    return {
        "logs_dir": str(root),
        "safety": summarize_safety(isolation, xgboost, survival, advanced_risk, trade_inputs),
        "isolation_forest": isolation,
        "xgboost": xgboost,
        "survival_exit": survival,
        "advanced_risk": advanced_risk,
        "trade_lineage": summarize_trade_lineage(trade_inputs),
        "paper_pnl": summarize_pnl(trade_inputs["closed_rows"]),
        "xgboost_outcome": summarize_xgboost_outcome(root),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_text_summary(summary: Dict[str, Any]) -> str:
    safety = summary["safety"]
    iso = summary["isolation_forest"]
    xgb = summary["xgboost"]
    surv = summary["survival_exit"]
    risk = summary["advanced_risk"]
    lineage = summary["trade_lineage"]
    pnl = summary["paper_pnl"]
    outcome = summary["xgboost_outcome"]
    return "\n".join(
        [
            "Unified Experimental Report",
            f"Logs: {summary['logs_dir']}",
            "",
            "Safety / Mode",
            f"  inferred_trade_mode: {safety['inferred_trade_mode']}",
            f"  shadow_only_warning: {safety['shadow_only_warning']}",
            f"  actual_behavior_counts: {safety['actual_behavior_counts']}",
            f"  warnings: {safety['warnings']}",
            "",
            "Isolation Forest",
            f"  total_rows: {iso['total_rows']}",
            f"  abnormal_count: {iso['abnormal_count']}",
            f"  would_block_count: {iso['would_block_count']}",
            f"  actually_blocked_count: {iso['actually_blocked_count']}",
            f"  would_block_rate: {_fmt(iso['would_block_rate'])}",
            f"  actual_block_rate: {_fmt(iso['actual_block_rate'])}",
            f"  latest_anomaly_score: {_fmt(iso['latest_anomaly_score'])}",
            f"  average_anomaly_score: {_fmt(iso['average_anomaly_score'])}",
            f"  top_reasons: {iso['top_reasons']}",
            "",
            "XGBoost",
            f"  total_rows: {xgb['total_rows']}",
            f"  would_confirm_count: {xgb['would_confirm_count']}",
            f"  would_reject_count: {xgb['would_reject_count']}",
            f"  actually_rejected_count: {xgb['actually_rejected_count']}",
            f"  would_reject_rate: {_fmt(xgb['would_reject_rate'])}",
            f"  actual_reject_rate: {_fmt(xgb['actual_reject_rate'])}",
            f"  reject_reason_counts: {xgb['reject_reason_counts']}",
            f"  average_confidence_allowed: {_fmt(xgb['average_confidence_allowed'])}",
            f"  average_confidence_rejected: {_fmt(xgb['average_confidence_rejected'])}",
            "",
            "Survival Exit",
            f"  total_rows: {surv['total_rows']}",
            f"  would_exit_early_count: {surv['would_exit_early_count']}",
            f"  actually_exited_count: {surv['actually_exited_count']}",
            f"  would_exit_rate: {_fmt(surv['would_exit_rate'])}",
            f"  actual_exit_rate: {_fmt(surv['actual_exit_rate'])}",
            f"  average_risk_score: {_fmt(surv['average_risk_score'])}",
            f"  latest_risk_score: {_fmt(surv['latest_risk_score'])}",
            f"  exit_reason_counts: {surv['exit_reason_counts']}",
            "",
            "Advanced Risk",
            f"  total_rows: {risk['total_rows']}",
            f"  would_block_count: {risk['would_block_count']}",
            f"  actually_blocked_count: {risk['actually_blocked_count']}",
            f"  would_pause_count: {risk['would_pause_count']}",
            f"  actually_paused_count: {risk['actually_paused_count']}",
            f"  would_reduce_size_count: {risk['would_reduce_size_count']}",
            f"  actually_reduced_count: {risk['actually_reduced_count']}",
            f"  average_risk_score: {_fmt(risk['average_risk_score'])}",
            f"  top_reasons: {risk['top_reasons']}",
            "",
            "Trade Lineage",
            f"  live_signal_rows: {lineage['live_signal_rows']}",
            f"  live_signal_rows_with_signal_id: {lineage['live_signal_rows_with_signal_id']}",
            f"  paper_trade_rows: {lineage['paper_trade_rows']}",
            f"  paper_trade_rows_with_signal_id: {lineage['paper_trade_rows_with_signal_id']}",
            f"  closed_trade_rows: {lineage['closed_trade_rows']}",
            f"  closed_trade_rows_with_signal_id: {lineage['closed_trade_rows_with_signal_id']}",
            f"  matched_closed_trade_count_by_signal_id: {lineage['matched_closed_trade_count_by_signal_id']}",
            "",
            "Paper PnL",
            f"  closed_trade_count: {pnl['closed_trade_count']}",
            f"  total_pnl: {_fmt(pnl['total_pnl'])}",
            f"  average_pnl: {_fmt(pnl['average_pnl'])}",
            f"  win_rate: {_fmt(pnl['win_rate'])}",
            f"  best_trade: {pnl['best_trade']}",
            f"  worst_trade: {pnl['worst_trade']}",
            "",
            "XGBoost Outcome",
            f"  join_method: {outcome['join_method']}",
            f"  would_confirm_matched_count: {outcome['would_confirm_matched_count']}",
            f"  would_confirm_average_pnl: {_fmt(outcome['would_confirm_average_pnl'])}",
            f"  would_confirm_win_rate: {_fmt(outcome['would_confirm_win_rate'])}",
            f"  would_reject_matched_count: {outcome['would_reject_matched_count']}",
            f"  would_reject_average_pnl: {_fmt(outcome['would_reject_average_pnl'])}",
            f"  would_reject_win_rate: {_fmt(outcome['would_reject_win_rate'])}",
            f"  matched_closed_trade_count: {outcome['matched_closed_trade_count']}",
            f"  unmatched_xgboost_rows: {outcome['unmatched_xgboost_rows']}",
        ]
    )


def write_json_summary(summary: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified experimental shadow and paper-lineage report")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--json", action="store_true", help="Write reports/unified_experimental_report.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_unified(args.logs_dir)
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
