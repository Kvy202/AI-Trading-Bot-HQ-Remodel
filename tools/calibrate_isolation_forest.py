"""Suggest Isolation Forest score thresholds from shadow logs.

This tool is read-only. It inspects logs/isolation_forest_shadow.csv and prints
candidate thresholds that would have produced approximate target block rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SHADOW_LOG = BASE_DIR / "logs" / "isolation_forest_shadow.csv"
TARGET_BLOCK_RATES = (0.05, 0.10, 0.20, 0.30)
ANOMALY_STATUSES = {"anomaly", "abnormal"}


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


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
        return out if out == out and abs(out) != float("inf") else None
    except Exception:
        return None


def _is_anomaly_row(row: Dict[str, str]) -> bool:
    return str(row.get("anomaly_status", "")).strip().lower() in ANOMALY_STATUSES


def _scored_anomaly_scores(rows: Iterable[Dict[str, str]]) -> List[float]:
    scores: List[float] = []
    for row in rows:
        if not _is_anomaly_row(row):
            continue
        score = _float_or_none(row.get("anomaly_score"))
        if score is not None:
            scores.append(score)
    return scores


def suggest_thresholds(
    rows: Sequence[Dict[str, str]],
    target_rates: Sequence[float] = TARGET_BLOCK_RATES,
) -> List[Dict[str, Any]]:
    total_rows = len(rows)
    scores = sorted(_scored_anomaly_scores(rows))
    suggestions: List[Dict[str, Any]] = []

    for target in target_rates:
        threshold: Optional[float] = None
        expected_block_count = 0
        if total_rows > 0 and scores:
            desired_count = min(len(scores), max(1, int(math.ceil(float(target) * total_rows))))
            threshold = scores[desired_count - 1]
            expected_block_count = sum(1 for score in scores if score <= threshold)
        suggestions.append(
            {
                "target_block_rate": float(target),
                "threshold": threshold,
                "expected_block_count": expected_block_count,
                "expected_block_rate": 0.0 if total_rows == 0 else expected_block_count / total_rows,
            }
        )
    return suggestions


def calibrate(path: Path | str = DEFAULT_SHADOW_LOG) -> Dict[str, Any]:
    shadow_log = Path(path)
    status, rows = _read_csv_rows(shadow_log)
    scores = _scored_anomaly_scores(rows)
    return {
        "file": str(shadow_log),
        "file_status": status,
        "total_rows": len(rows),
        "scored_anomaly_count": len(scores),
        "suggestions": suggest_thresholds(rows),
    }


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_calibration(result: Dict[str, Any]) -> str:
    lines = [
        "Isolation Forest Calibration",
        f"file: {result['file']}",
        f"file_status: {result['file_status']}",
        f"total_rows: {result['total_rows']}",
        f"scored_anomaly_count: {result['scored_anomaly_count']}",
    ]
    if result["total_rows"] == 0 or result["scored_anomaly_count"] == 0:
        lines.append("No scored anomaly rows available for threshold calibration.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "target_block_rate  threshold  expected_block_rate  expected_block_count",
        ]
    )
    for suggestion in result["suggestions"]:
        lines.append(
            "  ".join(
                [
                    _fmt_rate(float(suggestion["target_block_rate"])),
                    _fmt_float(suggestion["threshold"]),
                    _fmt_rate(float(suggestion["expected_block_rate"])),
                    str(suggestion["expected_block_count"]),
                ]
            )
        )
    return "\n".join(lines)


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibrate Isolation Forest score threshold from shadow logs")
    parser.add_argument("--shadow-log", default=str(DEFAULT_SHADOW_LOG))
    parser.add_argument("--json", action="store_true", help="Print calibration as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    result = calibrate(args.shadow_log)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_calibration(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
