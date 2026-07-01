"""Tests for XGBoost signal_id lineage through paper executor logs."""

import csv
import json
from pathlib import Path

import tools.live_executor as le
from tools.audit_xgboost_rejections import (
    CLOSED_MASTER_LOG,
    LIVE_SIGNALS_LOG,
    XGBOOST_LOG,
    summarize_audit,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_paper_entry_signal_id_is_persisted_in_open_position_state(monkeypatch, tmp_path):
    state_path = tmp_path / "executor_state.json"
    monkeypatch.setattr(le, "STATE_JSON", state_path)
    positions = {"BTCUSDT": le.Position("long", 1.0, 100.0)}

    le.write_state_snapshot(
        "PAPER",
        0.08,
        "abs",
        False,
        positions,
        position_signal_ids={"BTCUSDT": "sig-entry-1"},
    )

    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["open_positions"]["BTCUSDT"]["signal_id"] == "sig-entry-1"
    assert data["open_position_signal_ids"] == {"BTCUSDT": "sig-entry-1"}
    assert le.load_positions_from_state(state_path)["BTCUSDT"] == positions["BTCUSDT"]
    assert le.load_position_signal_ids_from_state(state_path) == {"BTCUSDT": "sig-entry-1"}


def test_paper_close_row_includes_signal_id(tmp_path):
    path = tmp_path / "trades_paper_20260701.csv"

    le.record_trade(
        path,
        ["2026-07-01 00:05:00+0000", "BTCUSDT", "SELL", 101.0, 1.0, "EXIT_TP pnl=1.0", "PAPER", "paper"],
        signal_id="sig-entry-1",
    )

    rows = _read_csv(path)
    assert rows[0]["signal_id"] == "sig-entry-1"


def test_closed_aggregate_and_dated_logs_include_signal_id(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    monkeypatch.setattr(le, "LOGS_DIR", logs)
    monkeypatch.setattr(le, "CLOSED_MASTER_CSV", logs / "trades_closed.csv")

    le.record_closed_trade(
        "2026-07-01 00:05:00+0000",
        "BTCUSDT",
        "SELL",
        1.0,
        100.0,
        101.0,
        1.0,
        "EXIT_TP pnl=1.0",
        signal_id="sig-entry-1",
    )

    master_rows = _read_csv(logs / "trades_closed.csv")
    dated_path = le.closed_path_for_day(le.datetime.now(le.timezone.utc).date())
    dated_rows = _read_csv(dated_path)
    assert master_rows[0]["signal_id"] == "sig-entry-1"
    assert dated_rows[0]["signal_id"] == "sig-entry-1"


def test_old_closed_logs_without_signal_id_are_extended_safely(monkeypatch, tmp_path):
    logs = tmp_path / "logs"
    monkeypatch.setattr(le, "LOGS_DIR", logs)
    monkeypatch.setattr(le, "CLOSED_MASTER_CSV", logs / "trades_closed.csv")
    old_header = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason"]
    old_row = ["2026-07-01 00:00:00+0000", "ETHUSDT", "SELL", "1", "50", "51", "1", "EXIT_TP pnl=1.0"]
    dated_path = le.closed_path_for_day(le.datetime.now(le.timezone.utc).date())
    _write_csv(logs / "trades_closed.csv", old_header, [old_row])
    _write_csv(dated_path, old_header, [old_row])

    le.record_closed_trade(
        "2026-07-01 00:05:00+0000",
        "BTCUSDT",
        "SELL",
        1.0,
        100.0,
        101.0,
        1.0,
        "EXIT_TP pnl=1.0",
        signal_id="sig-entry-1",
    )

    for path in (logs / "trades_closed.csv", dated_path):
        rows = _read_csv(path)
        assert "signal_id" in rows[0]
        assert rows[0]["signal_id"] == ""
        assert rows[1]["signal_id"] == "sig-entry-1"


def test_audit_can_match_closed_trade_by_signal_id(tmp_path):
    _write_csv(
        tmp_path / XGBOOST_LOG,
        [
            "timestamp",
            "symbol",
            "existing_signal",
            "existing_score",
            "confidence",
            "would_confirm",
            "would_reject",
            "actually_rejected",
            "reason",
            "reject_reason",
            "signal_id",
        ],
        [["2026-07-01 00:00:00+0000", "BTCUSDT", "LONG", "0.2", "0.8", "1", "0", "0", "confirmed", "", "sig-entry-1"]],
    )
    _write_csv(
        tmp_path / LIVE_SIGNALS_LOG,
        ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "side_hint", "signal_id"],
        [["2026-07-01 00:00:00+0000", "BTCUSDT", "100", "0.2", "0.01", "1", "0.08", "abs", "tcn", "LONG", "sig-entry-1"]],
    )
    _write_csv(
        tmp_path / "trades_paper_20260701.csv",
        ["ts", "symbol", "side", "price", "qty", "reason", "mode", "order_id", "signal_id"],
        [
            ["2026-07-01 00:00:00+0000", "BTCUSDT", "BUY", "100", "1", "ENTRY p=0.2000 rv=0.010000 eff_thr=0.0800", "PAPER", "paper", "sig-entry-1"],
            ["2026-07-01 00:05:00+0000", "BTCUSDT", "SELL", "101", "1", "EXIT_TP pnl=1.000000", "PAPER", "paper", "sig-exit-1"],
        ],
    )
    _write_csv(
        tmp_path / CLOSED_MASTER_LOG,
        ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason", "signal_id"],
        [["2026-07-01 00:05:00+0000", "BTCUSDT", "SELL", "1", "100", "101", "1.0", "EXIT_TP pnl=1.000000", "sig-entry-1"]],
    )

    join = summarize_audit(tmp_path)["trade_outcome_join"]

    assert join["join_method"] == "signal_id"
    assert join["matched_closed_trade_count"] == 1
    assert join["id_matched_count"] == 1
