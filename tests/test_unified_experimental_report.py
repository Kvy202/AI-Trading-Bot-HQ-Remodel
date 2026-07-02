"""Tests for the unified experimental report."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from tools.unified_experimental_report import (
    ADVANCED_RISK_LOG,
    ISOLATION_LOG,
    LIVE_SIGNALS_LOG,
    SURVIVAL_LOG,
    XGBOOST_LOG,
    format_text_summary,
    summarize_unified,
    write_json_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "unified_experimental_report.py"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_fixture_logs(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / ISOLATION_LOG,
        ["timestamp", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked", "reason"],
        [
            ["t1", "BTCUSDT", "normal", "0.10", "0", "0", "normal_market"],
            ["t2", "BTCUSDT", "anomaly", "-0.30", "1", "0", "isolation_anomaly"],
        ],
    )
    _write_csv(
        tmp_path / XGBOOST_LOG,
        [
            "timestamp",
            "symbol",
            "direction",
            "confidence",
            "would_confirm",
            "would_reject",
            "actually_rejected",
            "reason",
            "reject_reason",
            "signal_id",
        ],
        [
            ["2026-07-03 00:00:00+0000", "BTCUSDT", "LONG", "0.70", "1", "0", "0", "confirmed", "", "sig-1"],
            ["2026-07-03 00:01:00+0000", "ETHUSDT", "SHORT", "0.55", "0", "1", "0", "low_confidence", "low_confidence", "sig-2"],
            ["2026-07-03 00:02:00+0000", "SOLUSDT", "LONG", "0.40", "0", "1", "0", "direction_mismatch", "direction_mismatch", "sig-3"],
        ],
    )
    _write_csv(
        tmp_path / SURVIVAL_LOG,
        [
            "timestamp",
            "symbol",
            "survival_risk_score",
            "would_hold",
            "would_exit_early",
            "actually_exited",
            "exit_reason",
            "reason",
        ],
        [
            ["t1", "BTCUSDT", "0.20", "1", "0", "0", "", "hold_risk_below_threshold"],
            ["t2", "BTCUSDT", "0.80", "0", "1", "0", "", "high_exit_risk"],
        ],
    )
    _write_csv(
        tmp_path / ADVANCED_RISK_LOG,
        [
            "timestamp",
            "symbol",
            "side",
            "risk_score",
            "would_block",
            "actually_blocked",
            "would_reduce_size",
            "actually_reduced",
            "would_pause",
            "actually_paused",
            "top_reason",
        ],
        [
            ["t1", "BTCUSDT", "long", "0.00", "0", "0", "0", "0", "0", "0", "normal"],
            ["t2", "ETHUSDT", "short", "1.00", "1", "0", "0", "0", "1", "0", "daily_loss_pct_limit"],
        ],
    )
    _write_csv(
        tmp_path / LIVE_SIGNALS_LOG,
        ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "side_hint", "signal_id"],
        [
            ["2026-07-03 00:00:00+0000", "BTCUSDT", "100", "0.70", "0.01", "1", "0.55", "abs", "tcn", "LONG", "sig-1"],
            ["2026-07-03 00:01:00+0000", "ETHUSDT", "200", "-0.60", "0.01", "1", "0.55", "abs", "tcn", "SHORT", "sig-2"],
            ["2026-07-03 00:02:00+0000", "SOLUSDT", "10", "0.40", "0.01", "1", "0.55", "abs", "tcn", "LONG", "sig-3"],
        ],
    )
    _write_csv(
        tmp_path / "trades_paper_20260703.csv",
        ["ts", "symbol", "side", "price", "qty", "reason", "mode", "order_id", "signal_id"],
        [
            ["2026-07-03 00:00:00+0000", "BTCUSDT", "BUY", "100", "0.1", "ENTRY", "PAPER", "o1", "sig-1"],
            ["2026-07-03 00:10:00+0000", "BTCUSDT", "SELL", "107", "0.1", "EXIT_TP", "PAPER", "o2", "sig-1"],
            ["2026-07-03 00:01:00+0000", "ETHUSDT", "SELL_SHORT", "200", "0.1", "ENTRY", "PAPER", "o3", "sig-2"],
            ["2026-07-03 00:11:00+0000", "ETHUSDT", "BUY_TO_COVER", "202", "0.1", "EXIT_SL", "PAPER", "o4", "sig-2"],
        ],
    )
    closed_header = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason", "signal_id"]
    closed_rows = [
        ["2026-07-03 00:10:00+0000", "BTCUSDT", "SELL", "0.1", "100", "107", "0.70", "EXIT_TP", "sig-1"],
        ["2026-07-03 00:11:00+0000", "ETHUSDT", "BUY_TO_COVER", "0.1", "200", "202", "-0.20", "EXIT_SL", "sig-2"],
    ]
    _write_csv(tmp_path / "trades_closed.csv", closed_header, closed_rows)
    _write_csv(tmp_path / "trades_closed_20260703.csv", closed_header, closed_rows)


def test_missing_logs_handled_safely(tmp_path):
    summary = summarize_unified(tmp_path)

    assert summary["isolation_forest"]["file_status"] == "missing"
    assert summary["xgboost"]["file_status"] == "missing"
    assert summary["survival_exit"]["file_status"] == "missing"
    assert summary["advanced_risk"]["file_status"] == "missing"
    assert summary["trade_lineage"]["live_signal_rows"] == 0
    assert summary["paper_pnl"]["closed_trade_count"] == 0
    assert summary["safety"]["shadow_only_warning"] is False
    assert "Unified Experimental Report" in format_text_summary(summary)


def test_empty_logs_handled_safely(tmp_path):
    for name in (
        ISOLATION_LOG,
        XGBOOST_LOG,
        SURVIVAL_LOG,
        ADVANCED_RISK_LOG,
        LIVE_SIGNALS_LOG,
        "trades_paper_20260703.csv",
        "trades_closed.csv",
        "trades_closed_20260703.csv",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")

    summary = summarize_unified(tmp_path)

    assert summary["isolation_forest"]["file_status"] == "empty"
    assert summary["xgboost"]["file_status"] == "empty"
    assert summary["survival_exit"]["file_status"] == "empty"
    assert summary["advanced_risk"]["file_status"] == "empty"
    assert summary["paper_pnl"]["total_pnl"] == 0.0
    assert summary["paper_pnl"]["win_rate"] is None


def test_each_section_appears_in_json(tmp_path):
    summary = summarize_unified(tmp_path)
    out = write_json_summary(summary, tmp_path / "reports" / "unified.json")
    data = json.loads(out.read_text(encoding="utf-8"))

    assert set(data) == {
        "logs_dir",
        "safety",
        "isolation_forest",
        "xgboost",
        "survival_exit",
        "advanced_risk",
        "trade_lineage",
        "paper_pnl",
        "xgboost_outcome",
    }


def test_experimental_metrics_trade_lineage_and_pnl_are_calculated(tmp_path):
    _write_fixture_logs(tmp_path)

    summary = summarize_unified(tmp_path)

    iso = summary["isolation_forest"]
    assert iso["total_rows"] == 2
    assert iso["abnormal_count"] == 1
    assert iso["would_block_count"] == 1
    assert iso["actually_blocked_count"] == 0
    assert iso["would_block_rate"] == 0.5
    assert iso["latest_anomaly_score"] == -0.30
    assert iso["min_anomaly_score"] == -0.30
    assert iso["max_anomaly_score"] == 0.10
    assert round(iso["average_anomaly_score"], 6) == -0.10

    xgb = summary["xgboost"]
    assert xgb["total_rows"] == 3
    assert xgb["would_confirm_count"] == 1
    assert xgb["would_reject_count"] == 2
    assert xgb["actually_rejected_count"] == 0
    assert xgb["would_reject_rate"] == 2 / 3
    assert xgb["reject_reason_counts"] == {"low_confidence": 1, "direction_mismatch": 1}
    assert xgb["average_confidence_allowed"] == 0.70
    assert xgb["average_confidence_confirmed"] == 0.70
    assert round(xgb["average_confidence_rejected"], 6) == 0.475
    assert xgb["direction_mismatch_count"] == 1
    assert xgb["low_confidence_count"] == 1

    survival = summary["survival_exit"]
    assert survival["total_rows"] == 2
    assert survival["would_exit_early_count"] == 1
    assert survival["actually_exited_count"] == 0
    assert survival["would_exit_rate"] == 0.5
    assert survival["average_risk_score"] == 0.50
    assert survival["latest_risk_score"] == 0.80

    risk = summary["advanced_risk"]
    assert risk["total_rows"] == 2
    assert risk["would_block_count"] == 1
    assert risk["actually_blocked_count"] == 0
    assert risk["would_pause_count"] == 1
    assert risk["actually_paused_count"] == 0
    assert risk["would_reduce_size_count"] == 0
    assert risk["actually_reduced_count"] == 0
    assert risk["average_risk_score"] == 0.50
    assert risk["top_reasons"]["daily_loss_pct_limit"] == 1

    lineage = summary["trade_lineage"]
    assert lineage["live_signal_rows"] == 3
    assert lineage["live_signal_rows_with_signal_id"] == 3
    assert lineage["paper_trade_rows"] == 4
    assert lineage["paper_trade_rows_with_signal_id"] == 4
    assert lineage["closed_trade_rows"] == 2
    assert lineage["closed_trade_rows_with_signal_id"] == 2
    assert lineage["signal_id_missing_counts"] == {
        "live_signals": 0,
        "paper_trades": 0,
        "closed_trades": 0,
    }
    assert lineage["matched_closed_trade_count_by_signal_id"] == 2

    pnl = summary["paper_pnl"]
    assert pnl["closed_trade_count"] == 2
    assert round(pnl["total_pnl"], 6) == 0.50
    assert round(pnl["average_pnl"], 6) == 0.25
    assert pnl["win_rate"] == 0.5
    assert pnl["best_trade"]["symbol"] == "BTCUSDT"
    assert pnl["worst_trade"]["symbol"] == "ETHUSDT"

    outcome = summary["xgboost_outcome"]
    assert outcome["join_method"] == "signal_id"
    assert outcome["would_confirm_matched_count"] == 1
    assert round(outcome["would_confirm_average_pnl"], 6) == 0.70
    assert outcome["would_confirm_win_rate"] == 1.0
    assert outcome["would_reject_matched_count"] == 1
    assert round(outcome["would_reject_average_pnl"], 6) == -0.20
    assert outcome["would_reject_win_rate"] == 0.0
    assert outcome["matched_closed_trade_count"] == 2
    assert outcome["unmatched_decision_rows"] == 1


def test_shadow_only_warning_if_actual_active_or_blocking_fields_are_nonzero(tmp_path):
    _write_csv(
        tmp_path / ISOLATION_LOG,
        ["timestamp", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked", "reason"],
        [["t1", "BTCUSDT", "anomaly", "-0.3", "1", "1", "isolation_anomaly"]],
    )

    summary = summarize_unified(tmp_path)

    assert summary["safety"]["shadow_only_warning"] is True
    assert summary["safety"]["actual_behavior_counts"]["actually_blocked"] == 1
    assert "active_or_blocking_behavior_detected_in_shadow_report" in summary["safety"]["warnings"]


def test_cli_json_output_is_valid(tmp_path):
    _write_fixture_logs(tmp_path)
    out = tmp_path / "reports" / "unified_experimental_report.json"

    result = subprocess.run(
        [sys.executable, str(CLI), "--logs-dir", str(tmp_path), "--json", "--json-out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["xgboost"]["would_reject_count"] == 2
    assert payload["paper_pnl"]["closed_trade_count"] == 2
    assert "Unified Experimental Report" in result.stdout
