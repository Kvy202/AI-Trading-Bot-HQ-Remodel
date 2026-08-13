from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import shutil
import socket
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tools import hyperliquid_market_recorder_status as status


NOW = datetime(2026, 8, 13, 15, 0, 0, tzinfo=timezone.utc)
START = NOW - timedelta(hours=3)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def manifest():
    return {
        "schema_version": status.SCHEMA_VERSION,
        "capture_contract_version": status.CAPTURE_CONTRACT_VERSION,
        "capture_started_at_utc": status.utc_text(START),
        "network": "mainnet",
        "symbols": list(status.SYMBOLS),
        "subscriptions": list(status.EXPECTED_SUBSCRIPTIONS),
        "book_sampling_seconds": 5,
        "repository_commit": "abc",
        "python_version": "3.13.5",
        "output_root": "/capture",
        "data_is_public_market_data": True,
        "wallet_required": False,
        "private_key_required": False,
        "trading_enabled": False,
    }


def heartbeat(now=NOW):
    return {
        "process_started_at_utc": status.utc_text(now - timedelta(hours=2)),
        "last_message_at_utc": status.utc_text(now - timedelta(seconds=2)),
        "last_trade_at_utc": {symbol: status.utc_text(now - timedelta(minutes=5)) for symbol in status.SYMBOLS},
        "last_bbo_at_utc": {symbol: status.utc_text(now - timedelta(seconds=20)) for symbol in status.SYMBOLS},
        "last_book_at_utc": {symbol: status.utc_text(now - timedelta(seconds=10)) for symbol in status.SYMBOLS},
        "last_asset_ctx_at_utc": {symbol: status.utc_text(now - timedelta(seconds=10)) for symbol in status.SYMBOLS},
        "last_candle_at_utc": {symbol: status.utc_text(now - timedelta(minutes=5)) for symbol in status.SYMBOLS},
        "messages_received": {
            "total": 100,
            "control": 10,
            "by_stream_symbol": {
                stream: {symbol: 9 for symbol in status.SYMBOLS} for stream in status.HEARTBEAT_STREAMS
            },
        },
        "reconnect_count": 2,
        "integrity_error_count": 0,
    }


def create_directories(root: Path):
    for stream in status.RAW_STREAMS:
        for symbol in status.SYMBOLS:
            path = root / "raw" / stream / symbol / NOW.strftime("%Y-%m-%d") / f"{NOW:%H}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")


def set_active_lock(root: Path):
    write_json(
        root / "recorder.lock",
        {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "owner_token": "test-owner",
            "acquired_at_utc": status.utc_text(NOW - timedelta(hours=2)),
        },
    )


def create_status_capture(root: Path, hb=None, *, active_lock=True):
    write_json(root / "manifest.json", manifest())
    write_json(root / "heartbeat.json", hb or heartbeat())
    os.utime(root / "heartbeat.json", (NOW.timestamp() - 1, NOW.timestamp() - 1))
    create_directories(root)
    if active_lock:
        set_active_lock(root)


def issue_codes(result):
    return {issue["code"] for issue in result["issues"]}


def test_healthy_heartbeat_reports_all_operational_fields(tmp_path):
    create_status_capture(tmp_path)
    result = status.evaluate_status(tmp_path, now=NOW)
    assert result["health"] == "HEALTHY"
    assert result["capture_appears_healthy"] is True
    assert result["heartbeat_age_seconds"] == 1
    assert result["last_message_age_seconds"] == 2
    assert result["last_trade_age_seconds"] == {"BTC": 300, "ETH": 300}
    assert result["last_bbo_age_seconds"] == {"BTC": 20, "ETH": 20}
    assert result["last_book_age_seconds"] == {"BTC": 10, "ETH": 10}
    assert result["last_asset_ctx_age_seconds"] == {"BTC": 10, "ETH": 10}
    assert result["last_candle_age_seconds"] == {"BTC": 300, "ETH": 300}
    assert result["reconnect_count"] == 2
    assert result["integrity_error_count"] == 0
    assert result["recorder_lock"]["active"] is True
    assert result["messages_by_stream_symbol"]["trades"] == {"BTC": 9, "ETH": 9}
    assert result["latest_hourly_partition"]["trades"]["BTC"]["path"].endswith("15.jsonl")


def test_stale_heartbeat_and_no_recent_message_fail_closed(tmp_path):
    hb = heartbeat()
    hb["last_message_at_utc"] = status.utc_text(NOW - timedelta(minutes=3))
    create_status_capture(tmp_path, hb)
    os.utime(tmp_path / "heartbeat.json", (NOW.timestamp() - 45, NOW.timestamp() - 45))
    result = status.evaluate_status(tmp_path, now=NOW)
    assert result["health"] == "FAILED"
    assert {"HEARTBEAT_STALE", "LAST_MESSAGE_STALE"} <= issue_codes(result)


def test_missing_stream_directory_fails(tmp_path):
    create_status_capture(tmp_path)
    missing = tmp_path / "raw" / "book_5s" / "ETH"
    shutil.rmtree(missing)
    result = status.evaluate_status(tmp_path, now=NOW)
    assert result["health"] == "FAILED"
    assert "STREAM_DIRECTORY_MISSING" in issue_codes(result)


def test_stream_specific_warning_is_distinct_from_failure(tmp_path):
    warning_heartbeat = heartbeat()
    warning_heartbeat["last_trade_at_utc"]["BTC"] = status.utc_text(NOW - timedelta(minutes=20))
    create_status_capture(tmp_path, warning_heartbeat)
    warning = status.evaluate_status(tmp_path, now=NOW)
    assert warning["health"] == "WARNING"
    assert "TRADES_BTC_OLD" in issue_codes(warning)

    failed_heartbeat = heartbeat()
    failed_heartbeat["last_trade_at_utc"]["BTC"] = status.utc_text(NOW - timedelta(hours=2))
    write_json(tmp_path / "heartbeat.json", failed_heartbeat)
    os.utime(tmp_path / "heartbeat.json", (NOW.timestamp() - 1, NOW.timestamp() - 1))
    failed = status.evaluate_status(tmp_path, now=NOW)
    assert failed["health"] == "FAILED"
    assert "TRADES_BTC_STALE" in issue_codes(failed)


def test_missing_trade_during_startup_warns_but_mature_process_fails(tmp_path):
    hb = heartbeat()
    hb["process_started_at_utc"] = status.utc_text(NOW - timedelta(seconds=10))
    hb["last_trade_at_utc"]["BTC"] = None
    create_status_capture(tmp_path, hb)
    warning = status.evaluate_status(tmp_path, now=NOW)
    assert warning["health"] == "WARNING"
    assert "TRADES_BTC_MISSING" in issue_codes(warning)

    hb["process_started_at_utc"] = status.utc_text(NOW - timedelta(hours=2))
    write_json(tmp_path / "heartbeat.json", hb)
    os.utime(tmp_path / "heartbeat.json", (NOW.timestamp() - 1, NOW.timestamp() - 1))
    failed = status.evaluate_status(tmp_path, now=NOW)
    assert failed["health"] == "FAILED"


def test_integrity_error_baseline_fails_on_unexpected_increase(tmp_path):
    hb = heartbeat()
    hb["integrity_error_count"] = 3
    create_status_capture(tmp_path, hb)
    failed = status.evaluate_status(tmp_path, now=NOW, allowed_integrity_errors=2)
    assert failed["health"] == "FAILED"
    assert "INTEGRITY_ERROR_INCREASE" in issue_codes(failed)
    acknowledged = status.evaluate_status(tmp_path, now=NOW, allowed_integrity_errors=3)
    assert acknowledged["health"] == "HEALTHY"


def test_manifest_incompatibility_fails_closed(tmp_path):
    create_status_capture(tmp_path)
    bad = manifest()
    bad["network"] = "testnet"
    write_json(tmp_path / "manifest.json", bad)
    result = status.evaluate_status(tmp_path, now=NOW)
    assert result["health"] == "FAILED"
    assert "MANIFEST_INCOMPATIBLE" in issue_codes(result)


def record_times(hour: datetime, seconds: int):
    value = hour + timedelta(seconds=seconds)
    return {
        "receive_time_utc": status.utc_text(value),
        "receive_time_ms": int(value.timestamp() * 1000),
    }


def trade_record(hour: datetime, seconds: int, symbol: str, tid: int, *, price="100"):
    fields = record_times(hour, seconds)
    size = Decimal("2")
    px = Decimal(price)
    notional = px * size
    return {
        **fields,
        "exchange_time_ms": fields["receive_time_ms"] - 5,
        "coin": symbol,
        "side": "BUY",
        "price": str(px),
        "size": str(size),
        "notional": str(notional),
        "tid": tid,
        "hash": f"0x{tid}",
        "signed_size": str(size),
        "signed_notional": str(notional),
    }


def bbo_record(hour: datetime, seconds: int, symbol: str):
    fields = record_times(hour, seconds)
    return {
        **fields,
        "exchange_time_ms": fields["receive_time_ms"] - 4,
        "coin": symbol,
        "bid_px": "100",
        "bid_sz": "2",
        "ask_px": "102",
        "ask_sz": "3",
        "mid_px": "101",
        "spread": "2",
        "spread_bps": "198.0198019801980198019801980198019801980198019802",
        "is_crossed": False,
    }


def book_record(hour: datetime, seconds: int, symbol: str):
    fields = record_times(hour, seconds)
    with localcontext() as context:
        context.prec = 50
        spread = (Decimal("2") / Decimal("101")) * Decimal("10000")
    value = {
        **fields,
        "exchange_time_ms": fields["receive_time_ms"] - 3,
        "coin": symbol,
        "best_bid": "100",
        "best_ask": "102",
        "mid": "101",
        "spread_bps": str(spread),
        "bid_order_count_5": 5,
        "ask_order_count_5": 5,
        "microprice": "101",
        "microprice_minus_mid_bps": "0",
    }
    for count in (1, 5, 10, 20):
        value[f"bid_depth_{count}"] = str(count)
        value[f"ask_depth_{count}"] = str(count)
        value[f"imbalance_{count}"] = "0"
    return value


def context_record(hour: datetime, seconds: int, symbol: str):
    fields = record_times(hour, seconds)
    return {
        **fields,
        "exchange_time_ms": None,
        "coin": symbol,
        "mark_px": "101",
        "mid_px": "100.5",
        "oracle_px": "100",
        "funding": "0.0001",
        "open_interest": "10",
        "day_notional_volume": "1000",
        "previous_day_price": "99",
        "mark_oracle_basis_bps": "100",
        "mid_oracle_basis_bps": "50",
        "mark_mid_basis_bps": "49.75",
    }


def candle_record(hour: datetime, seconds: int, symbol: str):
    fields = record_times(hour, seconds)
    opened = int(hour.timestamp() * 1000)
    return {
        **fields,
        "open_time_ms": opened,
        "close_time_ms": opened + 299_999,
        "coin": symbol,
        "interval": "5m",
        "open": "100",
        "high": "102",
        "low": "99",
        "close": "101",
        "volume": "10",
        "trade_count": 2,
    }


RECORD_FACTORY = {
    "trades": trade_record,
    "bbo": bbo_record,
    "book_5s": book_record,
    "asset_ctx": context_record,
    "candles_5m": candle_record,
}


def partition_path(root: Path, stream: str, symbol: str, hour: datetime):
    return root / "raw" / stream / symbol / hour.strftime("%Y-%m-%d") / f"{hour:%H}.jsonl"


def append_line(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(value if isinstance(value, str) else canonical(value))
        handle.write("\n")


def create_audit_capture(root: Path, hours=(START,)):
    write_json(root / "manifest.json", manifest())
    tid = 1
    for hour in hours:
        for stream in status.RAW_STREAMS:
            for symbol in status.SYMBOLS:
                factory = RECORD_FACTORY[stream]
                value = factory(hour, 10, symbol, tid) if stream == "trades" else factory(hour, 10, symbol)
                append_line(partition_path(root, stream, symbol, hour), value)
                tid += 1
    for interval in ("1m", "5m"):
        for symbol in status.SYMBOLS:
            write_json(
                root / "aggregates" / interval / f"{symbol}.jsonl",
                {"bucket_start_ms": int(START.timestamp() * 1000), "coin": symbol, "interval": interval},
            )


def tree_digest(path: Path):
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def test_clean_continuity_audit_and_deterministic_summaries(tmp_path):
    create_audit_capture(tmp_path)
    first = status.continuity_audit(tmp_path, now=NOW)
    second = status.continuity_audit(tmp_path, now=NOW)
    assert first == second
    assert first["audit_status"] == "PASSED"
    assert first["lines_scanned"] == 10
    assert first["rows_per_hour"]["trades"]["BTC"] == {"2026-08-13T12:00:00Z": 1}
    assert first["aggregate_coverage"]["1m"]["BTC"]["row_count"] == 1
    assert first["raw_bytes"] > 0
    assert first["capture_duration_seconds"] == 0


def test_missing_hourly_partition_is_reported(tmp_path):
    second_hour = START + timedelta(hours=1)
    create_audit_capture(tmp_path, hours=(START, second_hour))
    missing = partition_path(tmp_path, "bbo", "ETH", second_hour)
    missing.unlink()
    result = status.continuity_audit(tmp_path, now=NOW)
    assert result["audit_status"] == "FAILED"
    assert {"stream": "bbo", "symbol": "ETH", "hour_utc": "2026-08-13T13:00:00.000Z"} in result["missing_hourly_partitions"]


def test_malformed_line_is_counted(tmp_path):
    create_audit_capture(tmp_path)
    append_line(partition_path(tmp_path, "bbo", "BTC", START), "{not-json")
    result = status.continuity_audit(tmp_path, now=NOW)
    assert result["malformed_json_line_count"] == 1
    assert result["audit_status"] == "FAILED"
    assert result["diagnostic_samples"]["malformed_json_lines"][0]["line"] == 2


def test_integrity_partitions_are_scanned_without_being_required(tmp_path):
    create_audit_capture(tmp_path)
    integrity_path = partition_path(tmp_path, "integrity", "BTC", START)
    append_line(
        integrity_path,
        {
            **record_times(START, 20),
            "kind": "invalid_book",
            "coin": "BTC",
            "details": {"error": "crossed"},
        },
    )
    append_line(integrity_path, "{bad-diagnostic")
    result = status.continuity_audit(tmp_path, now=NOW)
    assert result["invalid_book_diagnostic_count"] == 1
    assert result["malformed_json_line_count"] == 1
    assert result["rows_per_hour"]["integrity"]["BTC"] == {"2026-08-13T12:00:00Z": 1}
    assert result["files_scanned"] == 11
    assert result["audit_status"] == "FAILED"


def test_duplicate_and_conflicting_trades_are_detected(tmp_path):
    create_audit_capture(tmp_path)
    path = partition_path(tmp_path, "trades", "BTC", START)
    original = trade_record(START, 10, "BTC", 1)
    append_line(path, original)
    append_line(path, {**original, "price": "101", "notional": "202", "signed_notional": "202"})
    result = status.continuity_audit(tmp_path, now=NOW)
    assert result["duplicate_trade_identity_count"] == 1
    assert result["conflicting_trade_identity_count"] == 1
    assert result["audit_status"] == "FAILED"


def test_receive_and_exchange_time_ordering_issues_are_detected(tmp_path):
    create_audit_capture(tmp_path)
    append_line(partition_path(tmp_path, "bbo", "BTC", START), bbo_record(START, 5, "BTC"))
    result = status.continuity_audit(tmp_path, now=NOW)
    assert result["receive_time_ordering_issue_count"] == 1
    assert result["exchange_time_ordering_issue_count"] == 1
    assert result["audit_status"] == "FAILED"


def test_crossed_bbo_is_warning_while_invalid_book_is_failure(tmp_path):
    create_audit_capture(tmp_path)
    bbo_path = partition_path(tmp_path, "bbo", "BTC", START)
    crossed = bbo_record(START, 20, "BTC")
    crossed.update({"bid_px": "103", "ask_px": "102", "is_crossed": True})
    append_line(bbo_path, crossed)
    warning = status.continuity_audit(tmp_path, now=NOW)
    assert warning["crossed_bbo_count"] == 1
    assert warning["audit_status"] == "WARNING"

    invalid = book_record(START, 20, "BTC")
    invalid["imbalance_5"] = "2"
    append_line(partition_path(tmp_path, "book_5s", "BTC", START), invalid)
    failed = status.continuity_audit(tmp_path, now=NOW)
    assert failed["invalid_book_count"] == 1
    assert failed["audit_status"] == "FAILED"


def test_audit_never_writes_or_repairs_raw_data(tmp_path):
    create_audit_capture(tmp_path)
    before = tree_digest(tmp_path / "raw")
    before_files = sorted(str(path.relative_to(tmp_path)) for path in (tmp_path / "raw").rglob("*") if path.is_file())
    status.continuity_audit(tmp_path, now=NOW)
    after_files = sorted(str(path.relative_to(tmp_path)) for path in (tmp_path / "raw").rglob("*") if path.is_file())
    assert tree_digest(tmp_path / "raw") == before
    assert after_files == before_files


def test_status_and_audit_do_not_access_network_or_secrets(tmp_path, monkeypatch):
    create_status_capture(tmp_path, active_lock=False)
    create_audit_capture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    status.evaluate_status(tmp_path, now=NOW)
    status.continuity_audit(tmp_path, now=NOW)
    source = inspect.getsource(status)
    lowered = source.lower()
    assert "aiohttp" not in lowered
    assert "requests" not in lowered
    assert "urllib" not in lowered
    assert "dotenv" not in lowered
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "PRIVATE_KEY" not in source
    assert "HYPERLIQUID_PRIVATE_KEY" not in source
    assert "API_SECRET" not in source
    assert "hyperliquid.exchange" not in lowered
    assert '"type": "action"' not in source
    assert '"method": "post"' not in source


def test_systemd_unit_is_passive_and_has_no_secret_environment_file():
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy" / "aws" / "hl-market-recorder.service").read_text(encoding="utf-8")
    assert "tools/hyperliquid_market_recorder.py" in unit
    assert "EnvironmentFile=" not in unit
    assert "live_writer" not in unit
    assert "live_executor" not in unit
    assert "--duration-seconds" not in unit
    assert "KillSignal=SIGINT" in unit
    assert "RestartPreventExitStatus=2" in unit
    assert "PRIVATE_KEY" not in unit
    assert "API_SECRET" not in unit


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


def path_digest(path: Path):
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    lines = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        lines.append(f"{child.relative_to(path).as_posix()}|{hashlib.sha256(child.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_protected_project_files_remain_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert {path: path_digest(root / path) for path in PROTECTED_HASHES} == PROTECTED_HASHES
