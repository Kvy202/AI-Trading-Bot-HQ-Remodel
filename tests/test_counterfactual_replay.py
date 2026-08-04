"""Synthetic deterministic lifecycle, cohort, parity, and safety tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import tools.live_executor as live_executor
from exchanges.types import Position
from tools.counterfactual_replay import (
    CounterfactualReplayError,
    _decision_map,
    _run_replay,
    apply_slippage,
    check_tp_sl,
    cohort_separation,
    compare_recorded_parity,
    net_pnl_on_close,
    parse_signal_events,
    qty_for,
    replay_independent_cohorts,
    replay_portfolio,
)
from tools.replay_contract import resolve_replay_contract


def _contract(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "exec_thr": 0.5,
        "exec_mode": "abs",
        "respect_writer_thr": False,
        "rv_max": 1.0,
        "cooldown_sec": 0.0,
        "sides": "both",
        "max_symbols": 2,
        "one_position": False,
        "notional_usdt": 100.0,
        "max_portfolio_usdt": 1000.0,
        "min_notional": 1.0,
        "min_qty": 0.0,
        "tp_pct": 0.10,
        "sl_pct": 0.10,
        "fee_bps": 0.0,
        "slippage_bps": 0.0,
        "flip_open": True,
        "flip_confirm_ticks": 1,
        "scale_in": False,
        "adaptive": False,
        "bias_guard": False,
        "restore_state": False,
        "v2_enabled": False,
        "v2_time_stop_minutes": 0.0,
        "survival_active": False,
        "xgboost_blocking": False,
        "iforest_blocking": False,
        "advanced_risk_active": False,
        "paper_mode": True,
        "place_real_orders": False,
    }
    value.update(updates)
    return value


def _row(
    second: int,
    price: float,
    p_meta: float,
    signal_id: str,
    *,
    symbol: str = "BTC",
    allow: int = 1,
) -> dict[str, object]:
    return {
        "ts": f"2026-01-01T00:{second // 60:02d}:{second % 60:02d}Z",
        "symbol": symbol,
        "px": price,
        "p_meta": p_meta,
        "rv_mean": 0.01,
        "allow": allow,
        "thr": 0.5,
        "mode": "abs",
        "kinds_used": "model",
        "signal_id": signal_id,
    }


def _events(*rows: dict[str, object]):
    parsed, exclusions = parse_signal_events(list(rows))
    assert not exclusions
    return parsed


def _decision(signal_id: str, *, confirm: int = 1, reject: int = 0, side: str = "LONG"):
    return {
        "signal_id": signal_id,
        "would_confirm": str(confirm),
        "would_reject": str(reject),
        "existing_signal": side,
        "confidence": "0.9",
        "model_version": "fixture",
        "reason": "confirmed" if confirm else "direction_mismatch",
    }


def test_slippage_is_adverse_for_long_short_and_exit():
    assert apply_slippage(100, "BUY", 10) == pytest.approx(100.1)
    assert apply_slippage(100, "SELL_SHORT", 10) == pytest.approx(99.9)
    assert apply_slippage(100, "SELL", 10) == pytest.approx(99.9)
    assert apply_slippage(100, "BUY_TO_COVER", 10) == pytest.approx(100.1)


def test_entry_and_exit_fees_are_both_charged():
    position = Position("long", 1.0, 100.0)
    pnl, fill = net_pnl_on_close(position, 110.0, "SELL", 10.0, 0.0)

    assert fill == 110
    assert pnl == pytest.approx(10 - 0.1 - 0.11)


@pytest.mark.parametrize(
    ("p_meta", "exit_price", "expected_reason"),
    [
        (1.0, 111.0, "EXIT_TP"),
        (1.0, 89.0, "EXIT_SL"),
        (-1.0, 89.0, "EXIT_TP"),
        (-1.0, 111.0, "EXIT_SL"),
    ],
)
def test_long_and_short_tp_sl(p_meta, exit_price, expected_reason):
    replay = replay_portfolio(
        _events(_row(0, 100, p_meta, "entry"), _row(10, exit_price, p_meta, "exit")),
        {},
        _contract(),
    )

    assert replay["closed_trades"][0]["exit_reason"] == expected_reason


def test_tp_sl_has_priority_over_flip():
    replay = replay_portfolio(
        _events(_row(0, 100, 1, "entry"), _row(10, 111, -1, "opposite")),
        {},
        _contract(),
    )

    assert [trade["exit_reason"] for trade in replay["closed_trades"]] == ["EXIT_TP"]
    assert replay["metrics"]["entry_count"] == 1


def test_time_stop_follows_tp_sl_priority():
    replay = replay_portfolio(
        _events(_row(0, 100, 1, "entry"), _row(60, 111, 1, "tick")),
        {},
        _contract(v2_enabled=True, v2_time_stop_minutes=1),
    )

    assert replay["closed_trades"][0]["exit_reason"] == "EXIT_TP"


def test_no_end_window_force_close_and_position_is_censored():
    replay = replay_portfolio(_events(_row(0, 100, 1, "entry")), {}, _contract())

    assert replay["closed_trades"] == []
    assert replay["censored_positions"][0]["status"] == "censored_open_position"
    assert replay["metrics"]["censored_position_count"] == 1


def test_same_side_signal_does_not_flip():
    replay = replay_portfolio(
        _events(_row(0, 100, 1, "one"), _row(10, 101, 1, "two")), {}, _contract()
    )

    assert replay["closed_trades"] == []
    assert replay["metrics"]["entry_count"] == 1


def test_flip_confirmation_requires_consecutive_valid_ticks():
    replay = replay_portfolio(
        _events(
            _row(0, 100, 1, "entry"),
            _row(10, 100, -1, "flip-1"),
            _row(20, 100, -1, "flip-2"),
        ),
        {},
        _contract(flip_confirm_ticks=2),
    )

    assert replay["closed_trades"][0]["exit_reason"] == "FLIP_CLOSE"
    assert replay["censored_positions"][0]["side"] == "short"


def test_invalid_opposite_signal_does_not_increment_flip_confirmation():
    replay = replay_portfolio(
        _events(
            _row(0, 100, 1, "entry"),
            _row(10, 100, -1, "invalid", allow=0),
            _row(20, 100, -1, "only-valid"),
        ),
        {},
        _contract(flip_confirm_ticks=2),
    )

    assert replay["closed_trades"] == []


def test_immediate_flip_open_follows_completed_flip():
    replay = replay_portfolio(
        _events(_row(0, 100, 1, "entry"), _row(10, 100, -1, "flip")),
        {},
        _contract(flip_open=True, cooldown_sec=100),
    )

    assert len(replay["entries"]) == 2
    assert replay["censored_positions"][0]["side"] == "short"


def test_cooldown_uses_event_timestamps():
    replay = replay_portfolio(
        _events(
            _row(0, 100, 1, "entry"),
            _row(10, 111, 1, "tp"),
            _row(20, 100, 1, "too-soon"),
            _row(80, 100, 1, "after-cooldown"),
        ),
        {},
        _contract(cooldown_sec=60),
    )

    assert [entry["signal_id"] for entry in replay["entries"]] == ["entry", "after-cooldown"]


def test_portfolio_cap_and_one_position_gates():
    events = _events(_row(0, 100, 1, "btc"), _row(1, 100, 1, "eth", symbol="ETH"))
    cap = replay_portfolio(events, {}, _contract(max_portfolio_usdt=150))
    one = replay_portfolio(events, {}, _contract(one_position=True))

    assert cap["metrics"]["entry_count"] == 1
    assert one["metrics"]["entry_count"] == 1


def test_quantity_and_price_helpers_match_live_executor():
    assert qty_for(123.45, 15, 5, 0) == live_executor.qty_for(123.45, 15, 5, 0)
    assert apply_slippage(123.45, "BUY", 2) == live_executor.apply_slippage(123.45, "BUY", 2)
    ours = net_pnl_on_close(Position("short", 0.2, 100), 90, "BUY_TO_COVER", 5, 2)
    theirs = live_executor.net_pnl_on_close(Position("short", 0.2, 100), 90, "BUY_TO_COVER", 5, 2)
    assert ours == pytest.approx(theirs)
    assert check_tp_sl(Position("long", 1, 100), 111, 0.1, 0.1) == live_executor.check_tp_sl(
        Position("long", 1, 100), 111, 0.1, 0.1
    )


def test_confirm_and_reject_policies_filter_opposite_cohorts():
    events = _events(_row(0, 100, 1, "confirmed"), _row(20, 100, 1, "rejected", symbol="ETH"))
    decisions = {
        "confirmed": _decision("confirmed", confirm=1, reject=0),
        "rejected": _decision("rejected", confirm=0, reject=1),
    }
    confirm = replay_portfolio(events, decisions, _contract(), "xgboost_confirm_only")
    reject = replay_portfolio(events, decisions, _contract(), "xgboost_reject_only")

    assert [entry["signal_id"] for entry in confirm["entries"]] == ["confirmed"]
    assert [entry["signal_id"] for entry in reject["entries"]] == ["rejected"]
    assert reject["exploratory_nonproduction_policy"] is True


def test_missing_xgboost_decision_is_not_confirmation_and_no_timestamp_join():
    events = _events(_row(0, 100, 1, "signal"))
    missing = replay_portfolio(events, {}, _contract(), "xgboost_confirm_only")
    wrong_id = replay_portfolio(
        events, {"other": _decision("other")}, _contract(), "xgboost_confirm_only"
    )

    assert missing["entries"] == []
    assert missing["blocked_entries"][0]["reason"] == "xgboost_join_missing"
    assert wrong_id["entries"] == []


def test_independent_cohorts_are_non_additive_use_future_rows_and_mfe_mae():
    events = _events(
        _row(0, 100, 1, "a"),
        _row(10, 105, 1, "future-1"),
        _row(20, 95, 1, "future-2"),
        _row(30, 111, 1, "future-3"),
    )
    decisions = {
        "a": _decision("a", side="LONG"),
        "future-1": _decision("future-1", side="LONG"),
    }
    cohorts = replay_independent_cohorts(events, decisions, _contract())
    first = cohorts["would_confirm"]["records"][0]

    assert cohorts["non_additive"] is True
    assert cohorts["would_confirm"]["non_additive"] is True
    assert first["exit_timestamp"] > first["timestamp"]
    assert first["signal_tick_mfe"] == pytest.approx(0.11)
    assert first["signal_tick_mae"] == pytest.approx(-0.05)


def test_malformed_signal_timestamp_is_excluded():
    rows = [_row(0, 100, 1, "ok"), {**_row(1, 100, 1, "bad"), "ts": "bad"}]
    events, exclusions = parse_signal_events(rows)

    assert len(events) == 1
    assert exclusions == [{"source_order": 1, "reason": "malformed_timestamp"}]


def test_conflicting_xgboost_decisions_fail():
    with pytest.raises(CounterfactualReplayError):
        _decision_map(
            [
                _decision("same", confirm=1, reject=0),
                _decision("same", confirm=0, reject=1),
            ]
        )


def _baseline_fixture(pnl: float = 9.0) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    entry = {
        "signal_id": "entry",
        "timestamp": "2026-01-01T00:00:00Z",
        "symbol": "BTC",
        "side": "BUY",
        "quantity": 1.0,
        "entry_fill_price": 100.0,
    }
    close = {
        "entry_signal_id": "entry",
        "timestamp": "2026-01-01T00:01:00Z",
        "symbol": "BTC",
        "closed_side": "SELL",
        "quantity": 1.0,
        "entry_average": 100.0,
        "exit_fill_price": 110.0,
        "exit_reason": "EXIT_TP",
        "net_pnl": pnl,
    }
    baseline = {"entries": [entry], "closed_trades": [close], "censored_positions": []}
    paper = [
        {"ts": entry["timestamp"], "symbol": "BTC", "side": "BUY", "price": 100, "qty": 1, "signal_id": "entry"},
        {"ts": close["timestamp"], "symbol": "BTC", "side": "SELL", "price": 110, "qty": 1, "signal_id": "exit"},
    ]
    closed = [
        {
            "ts": close["timestamp"],
            "symbol": "BTC",
            "closed_side": "SELL",
            "qty": 1,
            "entry_avg": 100,
            "exit_price": 110,
            "realized_pnl": 9,
            "reason": "EXIT_TP pnl=9.000000",
            "signal_id": "entry",
        }
    ]
    return baseline, paper, closed


def test_exact_paper_parity_passes_but_one_trade_is_insufficient():
    baseline, paper, closed = _baseline_fixture()
    parity = compare_recorded_parity(baseline, paper, closed)

    assert parity["parity_passed"] is True
    assert parity["parity_status"] == "mechanically_passed_insufficient_sample"
    assert parity["promotion_parity_sample_gate"] is False


def test_mismatched_pnl_extra_and_missing_trades_fail_parity():
    baseline, paper, closed = _baseline_fixture(pnl=8.0)
    mismatch = compare_recorded_parity(baseline, paper, closed)
    baseline2, paper2, closed2 = _baseline_fixture()
    extra = compare_recorded_parity(
        {**baseline2, "entries": baseline2["entries"] * 2}, paper2, closed2
    )
    missing = compare_recorded_parity(
        {**baseline2, "entries": []}, paper2, closed2
    )

    assert mismatch["parity_status"] == "parity_failed"
    assert extra["parity_status"] == "parity_failed"
    assert missing["parity_status"] == "parity_failed"


def test_zero_closed_trades_is_not_testable():
    parity = compare_recorded_parity(
        {"entries": [], "closed_trades": [], "censored_positions": []}, [], []
    )

    assert parity["parity_status"] == "not_testable_no_closed_trades"
    assert parity["parity_passed"] is False


def test_ten_exact_closed_trades_satisfy_only_sample_and_parity_gates():
    baseline, paper, closed = _baseline_fixture()
    entries = []
    paper_rows = []
    closes = []
    closed_rows = []
    for index in range(10):
        suffix = str(index)
        item = dict(baseline["entries"][0])
        item["signal_id"] = f"entry-{suffix}"
        item["symbol"] = f"S{suffix}"
        entries.append(item)
        paper_rows.extend(
            [
                {**paper[0], "signal_id": item["signal_id"], "symbol": item["symbol"]},
                {**paper[1], "symbol": item["symbol"]},
            ]
        )
        close_item = dict(baseline["closed_trades"][0])
        close_item["entry_signal_id"] = item["signal_id"]
        close_item["symbol"] = item["symbol"]
        closes.append(close_item)
        closed_rows.append(
            {**closed[0], "signal_id": item["signal_id"], "symbol": item["symbol"]}
        )
    parity = compare_recorded_parity(
        {"entries": entries, "closed_trades": closes, "censored_positions": []},
        paper_rows,
        closed_rows,
    )

    assert parity["parity_status"] == "parity_verified"
    assert parity["promotion_parity_sample_gate"] is True


def test_manifest_excluded_and_network_runs_remain_excluded():
    source = {"status": "resolved_from_archives", "coverage": {"coverage_passed": True}}
    contract = {"status": "missing", "digest": None}
    for classification in ("unverified_legacy", "network_interrupted"):
        result = _run_replay(
            {
                "identity": "baseline:20260101000000",
                "mode": "baseline",
                "classification": classification,
                "include_in_strategy_aggregate": False,
            },
            contract,
            source,
            inventory_only=True,
            include_nonstrategy=False,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
        assert result["evidence_grade"] == "manifest_excluded"
        assert result["counterfactual_evidence_approved"] is False


def test_missing_historical_contract_produces_missing(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    overrides = tmp_path / "overrides.json"
    overrides.write_text('{"schema_version":1,"contracts":{}}', encoding="utf-8")

    assert resolve_replay_contract("baseline:20260101000000", reports, overrides)["status"] == "missing"


@pytest.mark.parametrize("field", ["adaptive", "bias_guard"])
def test_unsupported_adaptive_or_bias_runtime_state_fails_closed(field):
    replay = replay_portfolio(_events(_row(0, 100, 1, "entry")), {}, _contract(**{field: True}))

    assert replay["replay_status"] == "unsupported_historical_runtime_state"


def test_bootstrap_is_deterministic_and_requires_five_per_cohort():
    records = [
        {"net_pnl": float(value), "censored": False, "return_on_notional": value / 100}
        for value in range(1, 6)
    ]
    confirm = {"records": records, "average_net_pnl": 3, "win_rate": 1, "average_return_on_notional": 0.03}
    reject = {"records": [{**item, "net_pnl": -item["net_pnl"]} for item in records], "average_net_pnl": -3, "win_rate": 0, "average_return_on_notional": -0.03}

    first = cohort_separation(confirm, reject, bootstrap_samples=100, bootstrap_seed=7)
    second = cohort_separation(confirm, reject, bootstrap_samples=100, bootstrap_seed=7)

    assert first == second
    assert first["bootstrap_status"] == "completed"
    assert first["significance_claimed"] is False


def test_no_real_order_or_activation_command_is_generated():
    replay = replay_portfolio(_events(_row(0, 100, 1, "entry")), {}, _contract())
    serialized = json.dumps(replay).lower()

    assert "create_market_order" not in serialized
    assert "place_real_orders=true" not in serialized
    assert ".ps1" not in serialized


def test_confirm_policy_applies_to_scale_in_entries():
    events = _events(_row(0, 100, 1, "entry"), _row(10, 100, 1, "scale"))
    decisions = {
        "entry": _decision("entry", confirm=1, reject=0),
        "scale": _decision("scale", confirm=0, reject=1),
    }

    replay = replay_portfolio(
        events, decisions, _contract(scale_in=True), "xgboost_confirm_only"
    )

    assert [entry["signal_id"] for entry in replay["entries"]] == ["entry"]
    assert replay["blocked_entries"][-1]["reason"] == "xgboost_not_confirmed"


def test_independent_cohort_can_use_later_row_with_equal_timestamp():
    events = _events(_row(0, 100, 1, "entry"), _row(0, 106, 1, "later"))
    cohorts = replay_independent_cohorts(
        events,
        {"entry": _decision("entry")},
        _contract(tp_pct=0.05, sl_pct=0.05),
    )

    assert cohorts["would_confirm"]["closed_count"] == 1
    assert cohorts["would_confirm"]["records"][0]["exit_reason"] == "EXIT_TP"


def test_independent_cohort_excludes_malformed_decision_timestamp():
    decision = _decision("entry")
    decision["timestamp"] = "not-a-timestamp"
    cohorts = replay_independent_cohorts(
        _events(_row(0, 100, 1, "entry"), _row(10, 110, 1, "later")),
        {"entry": decision},
        _contract(),
    )

    assert cohorts["would_confirm"]["eligible_count"] == 0
    assert cohorts["exclusions"][0]["reason"] == "malformed_timestamp"


def test_parity_failure_is_json_serializable_and_reports_pnl_error():
    baseline, paper, closed = _baseline_fixture(pnl=8.0)

    parity = compare_recorded_parity(baseline, paper, closed)

    json.dumps(parity)
    assert parity["maximum_pnl_error"] == 1.0


def test_final_open_state_compares_average_and_entry_signal_id():
    paper = [
        {
            "ts": "2026-01-01T00:00:00Z",
            "symbol": "BTC",
            "side": "BUY",
            "price": 100,
            "qty": 1,
            "signal_id": "actual-entry",
        }
    ]
    baseline = {
        "entries": [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "symbol": "BTC",
                "side": "BUY",
                "entry_fill_price": 100,
                "quantity": 1,
                "signal_id": "actual-entry",
            }
        ],
        "closed_trades": [],
        "censored_positions": [
            {
                "symbol": "BTC",
                "side": "long",
                "quantity": 1,
                "entry_average": 101,
                "entry_signal_id": "different-entry",
            }
        ],
    }

    parity = compare_recorded_parity(baseline, paper, [])

    assert parity["exact_open_state_parity"] is False
