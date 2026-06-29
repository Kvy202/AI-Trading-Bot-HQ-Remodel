"""Experimental shadow-module reporting dashboard.

Read-only summary for optional AI shadow logs. This script never writes to
trading inputs and has no execution path into live decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_JSON_OUT = BASE_DIR / "reports" / "experimental_shadow_summary.json"

ISOLATION_LOG = "isolation_forest_shadow.csv"
XGBOOST_LOG = "xgboost_signal_shadow.csv"
SURVIVAL_LOG = "survival_exit_shadow.csv"


def _read_csv_rows(path: Path) -> tuple[str, List[Dict[str, str]]]:
    if not path.exists():
        return "missing", []
    if path.stat().st_size == 0:
        return "empty", []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return ("empty" if not rows else "ok"), rows
    except Exception as exc:
        return f"read_error:{type(exc).__name__}", []


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    except Exception:
        return None


def _avg(values: Iterable[Any]) -> Optional[float]:
    nums = [v for v in (_float_or_none(x) for x in values) if v is not None]
    return None if not nums else sum(nums) / len(nums)


def _latest(rows: List[Dict[str, str]], key: str) -> str:
    return rows[-1].get(key, "") if rows else ""


def _latest_float(rows: List[Dict[str, str]], key: str) -> Optional[float]:
    return _float_or_none(_latest(rows, key))


def _top_reasons(rows: List[Dict[str, str]], limit: int = 5) -> Dict[str, int]:
    counts = Counter((row.get("reason") or "unknown").strip() or "unknown" for row in rows)
    return dict(counts.most_common(limit))


def summarize_isolation(logs_dir: Path) -> Dict[str, Any]:
    path = logs_dir / ISOLATION_LOG
    status, rows = _read_csv_rows(path)
    normal = sum(1 for row in rows if str(row.get("anomaly_status", "")).strip().lower() == "normal")
    abnormal = sum(
        1
        for row in rows
        if str(row.get("anomaly_status", "")).strip().lower() in {"anomaly", "abnormal"}
    )
    would_block_count = sum(1 for row in rows if _truthy(row.get("would_block")))
    actually_blocked_count = sum(1 for row in rows if _truthy(row.get("actually_blocked")))
    total_rows = len(rows)
    return {
        "file": str(path),
        "file_status": status,
        "total_rows": total_rows,
        "normal_count": normal,
        "abnormal_count": abnormal,
        "would_block_count": would_block_count,
        "actually_blocked_count": actually_blocked_count,
        "block_rate": 0.0 if total_rows == 0 else actually_blocked_count / total_rows,
        "top_reasons": _top_reasons(rows),
        "latest_anomaly_score": _latest_float(rows, "anomaly_score"),
        "latest_model_version": _latest(rows, "model_version"),
    }


def summarize_xgboost(logs_dir: Path) -> Dict[str, Any]:
    path = logs_dir / XGBOOST_LOG
    status, rows = _read_csv_rows(path)
    return {
        "file": str(path),
        "file_status": status,
        "total_rows": len(rows),
        "would_confirm_count": sum(1 for row in rows if _truthy(row.get("would_confirm"))),
        "would_reject_count": sum(1 for row in rows if _truthy(row.get("would_reject"))),
        "average_confidence": _avg(row.get("xgboost_confidence") for row in rows),
        "latest_confidence": _latest_float(rows, "xgboost_confidence"),
        "latest_direction": _latest(rows, "xgboost_direction"),
        "latest_model_version": _latest(rows, "model_version"),
    }


def summarize_survival(logs_dir: Path) -> Dict[str, Any]:
    path = logs_dir / SURVIVAL_LOG
    status, rows = _read_csv_rows(path)
    return {
        "file": str(path),
        "file_status": status,
        "total_rows": len(rows),
        "would_hold_count": sum(1 for row in rows if _truthy(row.get("would_hold"))),
        "would_exit_early_count": sum(1 for row in rows if _truthy(row.get("would_exit_early"))),
        "average_survival_risk_score": _avg(row.get("survival_risk_score") for row in rows),
        "latest_risk_score": _latest_float(rows, "survival_risk_score"),
        "latest_reason": _latest(rows, "reason"),
        "latest_model_version": _latest(rows, "model_version"),
    }


def summarize_all(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> Dict[str, Any]:
    root = Path(logs_dir)
    return {
        "logs_dir": str(root),
        "isolation_forest": summarize_isolation(root),
        "xgboost_signal": summarize_xgboost(root),
        "survival_exit": summarize_survival(root),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_text_summary(summary: Dict[str, Any]) -> str:
    iso = summary["isolation_forest"]
    xgb = summary["xgboost_signal"]
    surv = summary["survival_exit"]
    lines = [
        "Experimental Shadow Report",
        f"Logs: {summary['logs_dir']}",
        "",
        "Isolation Forest",
        f"  file_status: {iso['file_status']}",
        f"  total_rows: {iso['total_rows']}",
        f"  normal_count: {iso['normal_count']}",
        f"  abnormal_count: {iso['abnormal_count']}",
        f"  would_block_count: {iso['would_block_count']}",
        f"  actually_blocked_count: {iso['actually_blocked_count']}",
        f"  block_rate: {_fmt(iso['block_rate'])}",
        f"  top_reasons: {iso['top_reasons']}",
        f"  latest_anomaly_score: {_fmt(iso['latest_anomaly_score'])}",
        f"  latest_model_version: {_fmt(iso['latest_model_version'])}",
        "",
        "XGBoost Signal",
        f"  file_status: {xgb['file_status']}",
        f"  total_rows: {xgb['total_rows']}",
        f"  would_confirm_count: {xgb['would_confirm_count']}",
        f"  would_reject_count: {xgb['would_reject_count']}",
        f"  average_confidence: {_fmt(xgb['average_confidence'])}",
        f"  latest_confidence: {_fmt(xgb['latest_confidence'])}",
        f"  latest_direction: {_fmt(xgb['latest_direction'])}",
        f"  latest_model_version: {_fmt(xgb['latest_model_version'])}",
        "",
        "Survival Exit",
        f"  file_status: {surv['file_status']}",
        f"  total_rows: {surv['total_rows']}",
        f"  would_hold_count: {surv['would_hold_count']}",
        f"  would_exit_early_count: {surv['would_exit_early_count']}",
        f"  average_survival_risk_score: {_fmt(surv['average_survival_risk_score'])}",
        f"  latest_risk_score: {_fmt(surv['latest_risk_score'])}",
        f"  latest_reason: {_fmt(surv['latest_reason'])}",
        f"  latest_model_version: {_fmt(surv['latest_model_version'])}",
    ]
    return "\n".join(lines)


def write_json_summary(summary: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experimental shadow module report")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--json", action="store_true", help="Write reports/experimental_shadow_summary.json")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    summary = summarize_all(args.logs_dir)
    print(format_text_summary(summary))
    if args.json:
        out = write_json_summary(summary, args.json_out)
        print(f"\njson_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
