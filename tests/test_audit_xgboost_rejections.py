"""Tests for the XGBoost rejection outcome audit."""

import csv
import json
from pathlib import Path

from tools.audit_xgboost_rejections import (
    CLOSED_MASTER_LOG,
    LIVE_SIGNALS_LOG,
    PAPER_GLOB,
    XGBOOST_LOG,
    format_text_summary,
    summarize_audit,
    write_json_summary,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_xgb_rows(path: Path, rows: list[list[object]]) -> None:
    _write_csv(
        path,
        [
            "timestamp",
            "symbol",
            "existing_signal",
            "existing_score",
            "xgboost_direction",
            "xgboost_confidence",
            "confidence",
            "would_confirm",
            "would_reject",
            "actually_rejected",
            "reason",
            "reject_reason",
        ],
        rows,
    )


def test_missing_logs_do_not_crash(tmp_path):
    summary = summarize_audit(tmp_path)
    text = format_text_summary(summary)

    assert summary["files"][XGBOOST_LOG]["status"] == "missing"
    assert summary["files"][LIVE_SIGNALS_LOG]["status"] == "missing"
    assert summary["files"][CLOSED_MASTER_LOG]["status"] == "missing"
    assert summary["files"][PAPER_GLOB]["status"] == "missing"
    assert summary["total_xgboost_rows"] == 0
    assert summary["would_reject_count"] == 0
    assert summary["actually_rejected_count"] == 0
    assert summary["trade_outcome_join"]["status"] == "not_available"
    assert "XGBoost Rejection Outcome Audit" in text


def test_empty_logs_are_handled(tmp_path):
    for name in (XGBOOST_LOG, LIVE_SIGNALS_LOG, CLOSED_MASTER_LOG):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "trades_paper_20260630.csv").write_text("", encoding="utf-8")

    summary = summarize_audit(tmp_path)

    assert summary["files"][XGBOOST_LOG]["status"] == "empty"
    assert summary["files"][LIVE_SIGNALS_LOG]["status"] == "empty"
    assert summary["files"][CLOSED_MASTER_LOG]["status"] == "empty"
    assert summary["files"][PAPER_GLOB]["status"] == "empty"
    assert summary["total_xgboost_rows"] == 0
    assert summary["average_confidence_allowed"] is None
    assert summary["average_confidence_rejected"] is None


def test_rejected_vs_allowed_counts_and_reasons(tmp_path):
    _write_xgb_rows(
        tmp_path / XGBOOST_LOG,
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "LONG", "0.20", "LONG", "0.80", "0.80", "1", "0", "0", "confirmed", ""],
            ["2026-06-30 10:01:00+0000", "ETHUSDT", "SHORT", "-0.20", "SHORT", "0.55", "0.55", "0", "1", "0", "low_confidence", ""],
            ["2026-06-30 10:02:00+0000", "SOLUSDT", "LONG", "0.30", "SHORT", "0.90", "0.90", "0", "1", "1", "direction_mismatch", "direction_mismatch"],
            ["2026-06-30 10:03:00+0000", "DOGEUSDT", "FLAT", "0", "", "", "", "0", "0", "0", "no_existing_trade_signal", ""],
        ],
    )

    summary = summarize_audit(tmp_path)

    assert summary["total_xgboost_rows"] == 4
    assert summary["would_confirm_count"] == 1
    assert summary["would_reject_count"] == 2
    assert summary["actually_rejected_count"] == 1
    assert summary["allowed_signal_count"] == 1
    assert summary["rejected_signal_count"] == 2
    assert summary["neutral_signal_count"] == 1
    assert summary["reject_reason_counts"] == {
        "low_confidence": 1,
        "direction_mismatch": 1,
    }
    assert summary["direction_mismatch_count"] == 1
    assert summary["low_confidence_count"] == 1


def test_confidence_averages_are_split_by_xgboost_decision(tmp_path):
    _write_xgb_rows(
        tmp_path / XGBOOST_LOG,
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "LONG", "0.20", "LONG", "0.80", "0.80", "1", "0", "0", "confirmed", ""],
            ["2026-06-30 10:01:00+0000", "ETHUSDT", "SHORT", "-0.20", "SHORT", "0.60", "0.60", "1", "0", "0", "confirmed", ""],
            ["2026-06-30 10:02:00+0000", "SOLUSDT", "LONG", "0.30", "SHORT", "0.40", "0.40", "0", "1", "0", "low_confidence", ""],
            ["2026-06-30 10:03:00+0000", "DOGEUSDT", "LONG", "0.30", "SHORT", "0.90", "0.90", "0", "1", "0", "direction_mismatch", ""],
        ],
    )

    summary = summarize_audit(tmp_path)

    assert round(summary["average_confidence_allowed"], 6) == 0.700000
    assert round(summary["average_confidence_rejected"], 6) == 0.650000


def test_unreliable_trade_join_is_reported_without_guessing(tmp_path):
    _write_xgb_rows(
        tmp_path / XGBOOST_LOG,
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "LONG", "0.20", "SHORT", "0.90", "0.90", "0", "1", "0", "direction_mismatch", ""],
        ],
    )
    _write_csv(
        tmp_path / LIVE_SIGNALS_LOG,
        ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "side_hint"],
        [["2026-06-30 10:00:00+0000", "BTCUSDT", "100.0", "0.20", "0.01", "1", "0.08", "abs", "tcn", "LONG"]],
    )
    _write_csv(
        tmp_path / CLOSED_MASTER_LOG,
        ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason"],
        [["2026-06-30 10:05:00+0000", "BTCUSDT", "SELL", "1", "100.0", "101.0", "1.0", "EXIT_TP pnl=1.0"]],
    )

    summary = summarize_audit(tmp_path)
    join = summary["trade_outcome_join"]
    text = format_text_summary(summary)

    assert join["status"] == "unreliable"
    assert join["matched_closed_trade_count"] == 0
    assert join["unmatched_rejected_signal_count"] == 1
    assert join["unmatched_reason_counts"] == {"paper_entry_or_closed_trade_missing": 1}
    assert "Trade outcome join is not reliable" in text


def test_exact_trade_join_reports_matched_closed_pnl(tmp_path):
    _write_xgb_rows(
        tmp_path / XGBOOST_LOG,
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "LONG", "0.20", "LONG", "0.80", "0.80", "1", "0", "0", "confirmed", ""],
            ["2026-06-30 10:01:00+0000", "ETHUSDT", "SHORT", "-0.20", "LONG", "0.90", "0.90", "0", "1", "0", "direction_mismatch", ""],
        ],
    )
    _write_csv(
        tmp_path / LIVE_SIGNALS_LOG,
        ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "side_hint"],
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "100.0", "0.20", "0.01", "1", "0.08", "abs", "tcn", "LONG"],
            ["2026-06-30 10:01:00+0000", "ETHUSDT", "50.0", "-0.20", "0.01", "1", "0.08", "abs", "tcn", "SHORT"],
        ],
    )
    _write_csv(
        tmp_path / "trades_paper_20260630.csv",
        ["ts", "symbol", "side", "price", "qty", "reason", "mode", "order_id"],
        [
            ["2026-06-30 10:00:00+0000", "BTCUSDT", "BUY", "100.0", "1", "ENTRY p=0.2000 rv=0.010000 eff_thr=0.0800", "PAPER", "paper"],
            ["2026-06-30 10:01:00+0000", "ETHUSDT", "SELL_SHORT", "50.0", "1", "ENTRY p=-0.2000 rv=0.010000 eff_thr=0.0800", "PAPER", "paper"],
            ["2026-06-30 10:05:00+0000", "BTCUSDT", "SELL", "101.5", "1", "EXIT_TP pnl=1.500000", "PAPER", "paper"],
            ["2026-06-30 10:06:00+0000", "ETHUSDT", "BUY_TO_COVER", "52.0", "1", "EXIT_SL pnl=-2.000000", "PAPER", "paper"],
        ],
    )
    _write_csv(
        tmp_path / CLOSED_MASTER_LOG,
        ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason"],
        [
            ["2026-06-30 10:05:00+0000", "BTCUSDT", "SELL", "1", "100.0", "101.5", "1.5", "EXIT_TP pnl=1.500000"],
            ["2026-06-30 10:06:00+0000", "ETHUSDT", "BUY_TO_COVER", "1", "50.0", "52.0", "-2.0", "EXIT_SL pnl=-2.000000"],
        ],
    )

    summary = summarize_audit(tmp_path)
    join = summary["trade_outcome_join"]

    assert join["status"] == "ok"
    assert join["matched_closed_trade_count"] == 2
    assert join["matched_closed_trade_pnl"]["allowed"]["count"] == 1
    assert join["matched_closed_trade_pnl"]["allowed"]["total_pnl"] == 1.5
    assert join["matched_closed_trade_pnl"]["rejected"]["count"] == 1
    assert join["matched_closed_trade_pnl"]["rejected"]["total_pnl"] == -2.0


def test_json_output_format(tmp_path):
    summary = summarize_audit(tmp_path)
    out = write_json_summary(summary, tmp_path / "reports" / "xgboost_rejection_audit.json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_xgboost_rows"] == 0
    assert data["files"][XGBOOST_LOG]["status"] == "missing"
    assert data["trade_outcome_join"]["status"] == "not_available"
