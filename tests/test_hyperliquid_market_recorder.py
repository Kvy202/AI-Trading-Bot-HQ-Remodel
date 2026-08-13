from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import hyperliquid_market_recorder as recorder


BASE = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def trade(*, side: str = "B", price: str = "100.25", size: str = "0.4", tid: int = 7, time: int = 1_765_627_200_000):
    return {"coin": "BTC", "side": side, "px": price, "sz": size, "time": time, "tid": tid, "hash": "0xabc"}


def level(price: str, size: str, count: int = 1):
    return {"px": price, "sz": size, "n": count}


def book(*, time: int = 1_765_627_200_000, bid_shift: int = 0):
    bids = [level(str(100 - index + bid_shift), str(index + 1), index + 1) for index in range(20)]
    asks = [level(str(102 + index), str(2 * (index + 1)), index + 2) for index in range(20)]
    return {"coin": "BTC", "time": time, "levels": [bids, asks]}


def context(*, mark: str = "101", oracle: str = "100", oi: str = "12.5", funding: str = "0.0001"):
    return {
        "coin": "BTC",
        "ctx": {
            "markPx": mark,
            "midPx": "100.5",
            "oraclePx": oracle,
            "funding": funding,
            "openInterest": oi,
            "dayNtlVlm": "123456.7",
            "prevDayPx": "99",
        },
    }


def candle():
    return {
        "t": 1_765_627_200_000,
        "T": 1_765_627_499_999,
        "s": "BTC",
        "i": "5m",
        "o": "100",
        "h": "105",
        "l": "99",
        "c": "104",
        "v": "12.25",
        "n": 42,
    }


def processor(tmp_path: Path):
    writer = recorder.HourlyJsonlWriter(tmp_path, fsync_interval_seconds=0)
    heartbeat = recorder.HeartbeatState(BASE)
    dedupe = recorder.TradeDeduplicator(max_entries=100)
    return recorder.MessageProcessor(writer, heartbeat, dedupe), writer, heartbeat


def jsonl_rows(root: Path, stream: str, coin: str = "BTC"):
    rows = []
    for path in sorted((root / "raw" / stream / coin).glob("*/*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def test_subscriptions_are_exactly_the_authorized_public_set():
    expected = []
    for coin in ("BTC", "ETH"):
        expected.extend(
            [
                {"type": "trades", "coin": coin},
                {"type": "bbo", "coin": coin},
                {"type": "l2Book", "coin": coin},
                {"type": "activeAssetCtx", "coin": coin},
                {"type": "candle", "coin": coin, "interval": "5m"},
            ]
        )
    assert recorder.WS_URL == "wss://api.hyperliquid.xyz/ws"
    assert recorder.SUBSCRIPTIONS == expected


def test_trade_aggressor_mapping_and_exact_notional():
    buy = recorder.parse_trade(trade(side="B", price="100.25", size="0.4"), BASE)
    sell = recorder.parse_trade(trade(side="A", price="100.25", size="0.4", tid=8), BASE)
    assert (buy["side"], buy["notional"], buy["signed_size"], buy["signed_notional"]) == (
        "BUY",
        "40.1",
        "0.4",
        "40.1",
    )
    assert (sell["side"], sell["signed_size"], sell["signed_notional"]) == ("SELL", "-0.4", "-40.1")


def test_trade_deduplication_and_restart_seed(tmp_path):
    parsed = recorder.parse_trade(trade(), BASE)
    dedupe = recorder.TradeDeduplicator(max_entries=10)
    assert dedupe.check(parsed)[0] == "new"
    assert dedupe.check(parsed)[0] == "duplicate"

    writer = recorder.HourlyJsonlWriter(tmp_path, fsync_interval_seconds=0)
    writer.append("trades", "BTC", BASE, parsed)
    writer.close()
    restarted = recorder.TradeDeduplicator(max_entries=10)
    assert restarted.seed_from_capture(tmp_path) == 1
    assert restarted.check(parsed)[0] == "duplicate"


def test_conflicting_duplicate_is_rejected_and_accounted(tmp_path):
    proc, writer, heartbeat = processor(tmp_path)
    proc.handle_message({"channel": "trades", "data": [trade()]}, BASE)
    proc.handle_message({"channel": "trades", "data": [trade(price="100.26")]}, BASE + timedelta(milliseconds=1))
    writer.close()
    assert len(jsonl_rows(tmp_path, "trades")) == 1
    errors = jsonl_rows(tmp_path, "integrity")
    assert errors[0]["kind"] == "conflicting_trade_duplicate"
    assert heartbeat.value["data_quality"]["conflicting_duplicate_count"] == 1
    assert heartbeat.value["integrity_error_count"] == 1


def test_bbo_spread_and_crossed_handling(tmp_path):
    valid, crossed = recorder.parse_bbo(
        {"coin": "BTC", "time": 10, "bbo": [level("100", "2"), level("102", "3")]}, BASE
    )
    assert crossed is False
    assert valid["mid_px"] == "101"
    assert valid["spread"] == "2"
    assert valid["spread_bps"] == "198.0198019801980198019801980198019801980198019802"

    invalid, crossed = recorder.parse_bbo(
        {"coin": "BTC", "time": 11, "bbo": [level("103", "2"), level("102", "3")]}, BASE
    )
    assert crossed is True
    assert invalid["mid_px"] is None and invalid["spread_bps"] is None
    proc, writer, heartbeat = processor(tmp_path)
    proc.handle_message({"channel": "bbo", "data": {"coin": "BTC", "time": 11, "bbo": [level("103", "2"), level("102", "3")]}}, BASE)
    writer.close()
    assert heartbeat.value["data_quality"]["crossed_bbo_count"] == 1
    assert jsonl_rows(tmp_path, "bbo")[0]["is_crossed"] is True


def test_depth_imbalance_and_microprice_formulas():
    sample = recorder.derive_book_sample(book(), BASE)
    assert sample["bid_depth_1"] == "1"
    assert sample["ask_depth_1"] == "2"
    assert sample["imbalance_1"] == "-0.33333333333333333333333333333333333333333333333333"
    assert sample["bid_depth_5"] == "15"
    assert sample["ask_depth_5"] == "30"
    assert sample["imbalance_5"] == "-0.33333333333333333333333333333333333333333333333333"
    assert sample["bid_depth_20"] == "210"
    assert sample["ask_depth_20"] == "420"
    assert sample["bid_order_count_5"] == 15
    assert sample["ask_order_count_5"] == 20
    assert sample["microprice"] == "100.66666666666666666666666666666666666666666666667"
    assert sample["microprice_minus_mid_bps"] == "-33.003300330033003300330033003300330033003300329703"


def test_book_uses_available_levels_without_inventing_missing_ones():
    value = {"coin": "BTC", "time": 10, "levels": [[level("100", "2")], [level("101", "3")]]}
    sample = recorder.derive_book_sample(value, BASE)
    assert sample["bid_depth_1"] == sample["bid_depth_20"] == "2"
    assert sample["ask_depth_1"] == sample["ask_depth_20"] == "3"


def test_five_second_book_sampling(tmp_path):
    proc, writer, _ = processor(tmp_path)
    proc.handle_message({"channel": "l2Book", "data": book(time=10)}, BASE)
    proc.handle_message({"channel": "l2Book", "data": book(time=11, bid_shift=-1)}, BASE + timedelta(seconds=4))
    proc.handle_message({"channel": "l2Book", "data": book(time=12, bid_shift=-2)}, BASE + timedelta(seconds=5))
    writer.close()
    rows = jsonl_rows(tmp_path, "book_5s")
    assert len(rows) == 2
    assert [row["exchange_time_ms"] for row in rows] == [10, 12]
    assert proc.latest_books["BTC"]["time"] == 12


def test_invalid_crossed_book_is_diagnosed(tmp_path):
    proc, writer, heartbeat = processor(tmp_path)
    crossed = {"coin": "BTC", "time": 10, "levels": [[level("102", "1")], [level("101", "1")]]}
    proc.handle_message({"channel": "l2Book", "data": crossed}, BASE)
    writer.close()
    assert jsonl_rows(tmp_path, "book_5s") == []
    assert heartbeat.value["data_quality"]["invalid_book_count"] == 1
    assert jsonl_rows(tmp_path, "integrity")[0]["kind"] == "invalid_book"


def test_active_asset_context_and_basis_parsing():
    parsed = recorder.parse_asset_ctx(context(), BASE)
    assert parsed["exchange_time_ms"] is None
    assert parsed["funding"] == "0.0001"
    assert parsed["open_interest"] == "12.5"
    assert parsed["day_notional_volume"] == "123456.7"
    assert parsed["mark_oracle_basis_bps"] == "100"
    assert parsed["mid_oracle_basis_bps"] == "50"
    assert parsed["mark_mid_basis_bps"] == "49.751243781094527363184079601990049751243781094527"


def test_zero_basis_denominator_returns_null():
    parsed = recorder.parse_asset_ctx(context(mark="1", oracle="0"), BASE)
    assert parsed["mark_oracle_basis_bps"] is None
    assert parsed["mid_oracle_basis_bps"] is None


def test_candle_parsing():
    parsed = recorder.parse_candle(candle(), BASE)
    assert parsed == {
        "receive_time_utc": "2026-08-13T12:00:00.000Z",
        "receive_time_ms": 1_786_622_400_000,
        "open_time_ms": 1_765_627_200_000,
        "close_time_ms": 1_765_627_499_999,
        "coin": "BTC",
        "interval": "5m",
        "open": "100",
        "high": "105",
        "low": "99",
        "close": "104",
        "volume": "12.25",
        "trade_count": 42,
    }


def test_hourly_utc_rotation(tmp_path):
    writer = recorder.HourlyJsonlWriter(tmp_path, fsync_interval_seconds=0)
    first = BASE.replace(hour=23, minute=59, second=59)
    second = first + timedelta(seconds=1)
    path1 = writer.append("trades", "BTC", first, {"row": 1})
    path2 = writer.append("trades", "BTC", second, {"row": 2})
    writer.close()
    assert path1.relative_to(tmp_path).as_posix() == "raw/trades/BTC/2026-08-13/23.jsonl"
    assert path2.relative_to(tmp_path).as_posix() == "raw/trades/BTC/2026-08-14/00.jsonl"
    assert json.loads(path1.read_text(encoding="utf-8"))["row"] == 1
    assert json.loads(path2.read_text(encoding="utf-8"))["row"] == 2


def test_restart_preserves_capture_started_at(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder, "repository_commit", lambda: "abc123")
    first = recorder.initialize_manifest(tmp_path, started_at=BASE)
    restarted = recorder.initialize_manifest(tmp_path, started_at=BASE + timedelta(days=3))
    assert restarted == first
    assert restarted["capture_started_at_utc"] == "2026-08-13T12:00:00.000Z"
    assert restarted["network"] == "mainnet"
    assert restarted["wallet_required"] is False
    assert restarted["private_key_required"] is False
    assert restarted["trading_enabled"] is False


def test_stale_lock_is_recovered(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    lock_path = tmp_path / "recorder.lock"
    lock_path.write_text(
        json.dumps({"pid": 999999, "hostname": socket.gethostname(), "owner_token": "old"}), encoding="utf-8"
    )
    lock = recorder.InstanceLock(tmp_path, pid_checker=lambda pid: False)
    lock.acquire()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["owner_token"] == lock.token
    lock.release()
    assert not lock_path.exists()


def test_active_lock_is_rejected(tmp_path):
    lock_path = tmp_path / "recorder.lock"
    lock_path.write_text(
        json.dumps({"pid": 123, "hostname": socket.gethostname(), "owner_token": "active"}), encoding="utf-8"
    )
    lock = recorder.InstanceLock(tmp_path, pid_checker=lambda pid: True)
    with pytest.raises(recorder.ActiveRecorderError, match="active PID 123"):
        lock.acquire()
    assert lock_path.exists()


def test_atomic_heartbeat_writes_leave_valid_file_and_no_temporary(tmp_path):
    path = tmp_path / "heartbeat.json"
    recorder.atomic_write_json(path, {"value": 1})
    recorder.atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert list(tmp_path.glob(".heartbeat.json.*.tmp")) == []


def test_out_of_order_timestamp_is_counted(tmp_path):
    proc, writer, heartbeat = processor(tmp_path)
    proc.handle_message({"channel": "trades", "data": [trade(time=20, tid=1)]}, BASE)
    proc.handle_message({"channel": "trades", "data": [trade(time=19, tid=2)]}, BASE + timedelta(milliseconds=1))
    writer.close()
    assert heartbeat.value["data_quality"]["out_of_order_timestamp_count"] == 1
    assert any(row["kind"] == "out_of_order_timestamp" for row in jsonl_rows(tmp_path, "integrity"))


def test_malformed_messages_fail_safely_and_count_invalid_numeric(tmp_path):
    proc, writer, heartbeat = processor(tmp_path)
    proc.handle_message({"channel": "trades", "data": [trade(price="NaN")]}, BASE)
    proc.handle_message({"channel": "nonsense", "data": {}}, BASE)
    writer.close()
    assert heartbeat.value["data_quality"]["invalid_numeric_count"] == 1
    assert heartbeat.value["data_quality"]["malformed_message_count"] == 2
    assert len(jsonl_rows(tmp_path, "integrity", "GLOBAL")) == 1


class FakeWebsocket:
    def __init__(self, on_end):
        self.sent = []
        self.closed = False
        self.on_end = on_end

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.closed = True

    async def send_json(self, value):
        self.sent.append(value)

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.on_end()
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.websockets = []
        self.recorder = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def ws_connect(self, *args, **kwargs):
        connection_number = len(self.websockets) + 1

        def on_end():
            if connection_number == 2:
                self.recorder.stop_event.set()

        websocket = FakeWebsocket(on_end)
        self.websockets.append(websocket)
        return websocket


def test_reconnect_resubscribes_with_mocked_websocket(tmp_path):
    session = FakeSession()

    async def fake_sleep(seconds):
        if seconds >= 30:
            await asyncio.Event().wait()

    market_recorder = recorder.MarketRecorder(
        tmp_path,
        session_factory=lambda **kwargs: session,
        sleep=fake_sleep,
        heartbeat_seconds=60,
    )
    session.recorder = market_recorder
    heartbeat = asyncio.run(market_recorder.run())
    assert len(session.websockets) == 2
    expected = [{"method": "subscribe", "subscription": value} for value in recorder.SUBSCRIPTIONS]
    assert session.websockets[0].sent == expected
    assert session.websockets[1].sent == expected
    assert heartbeat["reconnect_count"] == 1


def test_no_secret_environment_or_dotenv_access(monkeypatch, tmp_path):
    monkeypatch.setattr(recorder.os, "getenv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("secret read")))
    monkeypatch.setattr(recorder, "repository_commit", lambda: None)
    recorder.initialize_manifest(tmp_path, started_at=BASE)
    source = inspect.getsource(recorder)
    assert "os.environ" not in source
    assert "dotenv" not in source.lower()
    assert "HYPERLIQUID_PRIVATE_KEY" not in source
    assert "PRIVATE_KEY" not in source
    assert "API_SECRET" not in source


def test_no_exchange_or_signed_request_code_paths():
    source = inspect.getsource(recorder)
    lowered = source.lower()
    assert "hyperliquid.exchange" not in lowered
    assert '"method": "post"' not in source
    assert '"type": "action"' not in source
    assert "/exchange" not in lowered
    assert "eth_account" not in lowered
    assert "sign_transaction" not in lowered
    assert "place_order" not in lowered
    assert "cancel_order" not in lowered


PROTECTED_HASHES = {
    "features.py": "061a53a73b2a5413fcb77f7a2c9c9476b69d6e34a2f773fb468cad10c312a20d",
    "tools/live_writer.py": "8bcd7364f3e932452d94ac258531a35814a7095779b199e622e0f8b680a18f23",
    "tools/live_executor.py": "d3cf6854a5e30915a4628e3b9e00b9f3c2f16ee084ea31cd2581f93a74bd3af9",
    "model_artifacts": "558a702ea5ab75a57f729c69d6fe21b13d73085b88a66fd0ead4d021648a180c",
    "reports/model_candidate_validation_access.json": "c1c8405d5fa701f19f5b765e17509a69f9d2416a1d3a99becd47a0c8b8d29548",
    "reports/model_signal_research": "c9da578fe3ae880bbfa3d7dba05c4b0b46be62f8d08bceccbcbe927d457db4e6",
    "reports/model_signal_backtest": "c6291ba8c3b679ecbc982f805c78a491db235da8526113641ca974b39e9f3fa5",
    "reports/model_signal_selectivity": "39bbc5667867e68c9331382382278ca4bf04fcbdbf97bf92319fbaf34c208411",
    "reports/model_executable_return_research": "535b0c6a415a89580fb5c8c3333a7a998876c59ff111c6a7861b0562b07b6a5e",
    "reports/model_feature_enrichment_research": "13a016c499c2fce7ef9a608069d12be409dca3f542a5618b0b77ef7519fff8e3",
}


def path_digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    lines = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        lines.append(f"{child.relative_to(path).as_posix()}|{hashlib.sha256(child.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_protected_project_files_and_evidence_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert {path: path_digest(root / path) for path in PROTECTED_HASHES} == PROTECTED_HASHES
