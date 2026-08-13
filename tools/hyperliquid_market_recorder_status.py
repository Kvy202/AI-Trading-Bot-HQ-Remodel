"""Read-only health status and continuity audit for market recorder captures.

The module intentionally has no network dependency and contains no repair path.
It only opens capture artifacts for reading and inspects process ownership.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CAPTURE_ROOT = Path("data/hyperliquid_market_capture")
SYMBOLS: tuple[str, ...] = ("BTC", "ETH")
RAW_STREAMS: tuple[str, ...] = ("trades", "bbo", "book_5s", "asset_ctx", "candles_5m")
INTEGRITY_SYMBOLS: tuple[str, ...] = ("BTC", "ETH", "GLOBAL")
HEARTBEAT_STREAMS: tuple[str, ...] = ("trades", "bbo", "l2Book", "activeAssetCtx", "candle")
SCHEMA_VERSION = "1.0.0"
CAPTURE_CONTRACT_VERSION = "forward-public-market-data-v1"
EXPECTED_SUBSCRIPTIONS: tuple[dict[str, str], ...] = tuple(
    subscription
    for coin in SYMBOLS
    for subscription in (
        {"type": "trades", "coin": coin},
        {"type": "bbo", "coin": coin},
        {"type": "l2Book", "coin": coin},
        {"type": "activeAssetCtx", "coin": coin},
        {"type": "candle", "coin": coin, "interval": "5m"},
    )
)
LAST_TIME_KEYS: Mapping[str, str] = {
    "trades": "last_trade_at_utc",
    "bbo": "last_bbo_at_utc",
    "l2Book": "last_book_at_utc",
    "activeAssetCtx": "last_asset_ctx_at_utc",
    "candle": "last_candle_at_utc",
}
RAW_TO_HEARTBEAT_STREAM: Mapping[str, str] = {
    "trades": "trades",
    "bbo": "bbo",
    "book_5s": "l2Book",
    "asset_ctx": "activeAssetCtx",
    "candles_5m": "candle",
}


@dataclass(frozen=True)
class AgeThreshold:
    warning_seconds: float
    failure_seconds: float


@dataclass(frozen=True)
class HealthThresholds:
    heartbeat: AgeThreshold = AgeThreshold(15.0, 30.0)
    last_message: AgeThreshold = AgeThreshold(30.0, 90.0)
    trades: AgeThreshold = AgeThreshold(900.0, 3600.0)
    bbo: AgeThreshold = AgeThreshold(120.0, 600.0)
    l2Book: AgeThreshold = AgeThreshold(30.0, 120.0)
    activeAssetCtx: AgeThreshold = AgeThreshold(30.0, 120.0)
    candle: AgeThreshold = AgeThreshold(600.0, 1800.0)
    future_clock_tolerance_seconds: float = 5.0

    def for_stream(self, stream: str) -> AgeThreshold:
        value = getattr(self, stream)
        if not isinstance(value, AgeThreshold):
            raise KeyError(stream)
        return value


class InspectionError(ValueError):
    """A capture artifact cannot be interpreted safely."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise InspectionError(f"{field} is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InspectionError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise InspectionError(f"{field} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def parse_utc_argument(value: str) -> datetime:
    try:
        return parse_utc(value, "--now")
    except InspectionError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def age_seconds(now: datetime, value: datetime) -> float:
    return round((now - value).total_seconds(), 3)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise InspectionError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectionError(f"{label} is not a JSON object: {path}")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected: tuple[tuple[str, Any], ...] = (
        ("schema_version", SCHEMA_VERSION),
        ("capture_contract_version", CAPTURE_CONTRACT_VERSION),
        ("network", "mainnet"),
        ("symbols", list(SYMBOLS)),
        ("subscriptions", list(EXPECTED_SUBSCRIPTIONS)),
        ("book_sampling_seconds", 5),
        ("data_is_public_market_data", True),
        ("wallet_required", False),
        ("private_key_required", False),
        ("trading_enabled", False),
    )
    for field, expected_value in expected:
        if manifest.get(field) != expected_value:
            errors.append(f"manifest {field} expected {expected_value!r}, found {manifest.get(field)!r}")
    try:
        parse_utc(manifest.get("capture_started_at_utc"), "capture_started_at_utc")
    except InspectionError as exc:
        errors.append(str(exc))
    return errors


def _pid_is_active(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return ctypes.windll.kernel32.GetLastError() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_lock(capture_root: Path) -> dict[str, Any]:
    path = capture_root / "recorder.lock"
    if not path.exists():
        return {"state": "absent", "active": False, "path": str(path)}
    try:
        owner = _read_json(path, "recorder lock")
        pid = int(owner.get("pid"))
        hostname = owner.get("hostname")
        token = owner.get("owner_token")
        if not isinstance(hostname, str) or not hostname or not isinstance(token, str) or not token:
            raise InspectionError("recorder lock owner fields are invalid")
    except (InspectionError, TypeError, ValueError) as exc:
        return {"state": "invalid", "active": False, "path": str(path), "error": str(exc)}
    local_hostname = platform.node()
    details = {"path": str(path), "pid": pid, "hostname": hostname, "local_hostname": local_hostname}
    if hostname != local_hostname:
        return {**details, "state": "unknown_remote_owner", "active": None}
    active = _pid_is_active(pid)
    return {**details, "state": "active" if active else "stale", "active": active}


def _safe_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    with contextlib.suppress(OSError):
        for path in root.rglob("*"):
            with contextlib.suppress(OSError):
                if path.is_file() and not path.is_symlink():
                    files.append(path)
    return sorted(files)


def directory_usage(path: Path) -> dict[str, int]:
    files = _safe_files(path)
    total = 0
    for file_path in files:
        with contextlib.suppress(OSError):
            total += file_path.stat().st_size
    return {"bytes": total, "file_count": len(files)}


@dataclass(frozen=True)
class Partition:
    stream: str
    symbol: str
    hour: datetime
    path: Path

    @property
    def hour_text(self) -> str:
        return self.hour.strftime("%Y-%m-%dT%H:00:00Z")


def discover_partitions(capture_root: Path, stream: str, symbol: str) -> tuple[list[Partition], list[str]]:
    symbol_root = capture_root / "raw" / stream / symbol
    partitions: list[Partition] = []
    invalid_paths: list[str] = []
    if not symbol_root.is_dir():
        return partitions, invalid_paths
    for path in _safe_files(symbol_root):
        relative = path.relative_to(symbol_root)
        parts = relative.parts
        if len(parts) != 2 or path.suffix != ".jsonl":
            invalid_paths.append(str(relative).replace("\\", "/"))
            continue
        try:
            hour = datetime.strptime(f"{parts[0]}/{path.stem}", "%Y-%m-%d/%H").replace(tzinfo=timezone.utc)
        except ValueError:
            invalid_paths.append(str(relative).replace("\\", "/"))
            continue
        partitions.append(Partition(stream=stream, symbol=symbol, hour=hour, path=path))
    return sorted(partitions, key=lambda item: (item.hour, str(item.path))), sorted(invalid_paths)


def latest_partitions(capture_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stream in RAW_STREAMS:
        result[stream] = {}
        for symbol in SYMBOLS:
            partitions, invalid = discover_partitions(capture_root, stream, symbol)
            if not partitions:
                result[stream][symbol] = None
                continue
            latest = partitions[-1]
            result[stream][symbol] = {
                "hour_utc": latest.hour_text,
                "path": str(latest.path.relative_to(capture_root)).replace("\\", "/"),
                "bytes": latest.path.stat().st_size,
                "invalid_partition_paths": invalid,
            }
    return result


def _issue(issues: list[dict[str, str]], severity: str, code: str, message: str) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def _check_age(
    issues: list[dict[str, str]],
    *,
    code: str,
    label: str,
    age: float | None,
    threshold: AgeThreshold,
    process_age: float | None,
    future_tolerance: float,
) -> None:
    if age is None:
        severity = "FAILED" if process_age is None or process_age >= threshold.failure_seconds else "WARNING"
        _issue(issues, severity, f"{code}_MISSING", f"{label} has not produced a timestamp")
        return
    if age < -future_tolerance:
        _issue(issues, "FAILED", f"{code}_IN_FUTURE", f"{label} is {-age:.3f}s in the future")
    elif age > threshold.failure_seconds:
        _issue(issues, "FAILED", f"{code}_STALE", f"{label} age {age:.3f}s exceeds {threshold.failure_seconds:.0f}s")
    elif age > threshold.warning_seconds:
        _issue(issues, "WARNING", f"{code}_OLD", f"{label} age {age:.3f}s exceeds {threshold.warning_seconds:.0f}s")


def evaluate_status(
    capture_root: Path,
    *,
    now: datetime | None = None,
    thresholds: HealthThresholds = HealthThresholds(),
    allowed_integrity_errors: int = 0,
) -> dict[str, Any]:
    """Read capture controls and return a deterministic health snapshot."""

    now = (now or utc_now()).astimezone(timezone.utc)
    capture_root = capture_root.resolve()
    issues: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "mode": "status",
        "capture_root": str(capture_root),
        "current_utc": utc_text(now),
        "capture_started_at_utc": None,
        "process_started_at_utc": None,
        "heartbeat_age_seconds": None,
        "last_message_age_seconds": None,
        "last_trade_age_seconds": {symbol: None for symbol in SYMBOLS},
        "last_bbo_age_seconds": {symbol: None for symbol in SYMBOLS},
        "last_book_age_seconds": {symbol: None for symbol in SYMBOLS},
        "last_asset_ctx_age_seconds": {symbol: None for symbol in SYMBOLS},
        "last_candle_age_seconds": {symbol: None for symbol in SYMBOLS},
        "reconnect_count": None,
        "integrity_error_count": None,
        "messages_by_stream_symbol": {stream: {symbol: None for symbol in SYMBOLS} for stream in HEARTBEAT_STREAMS},
        "disk_usage": {
            "capture": directory_usage(capture_root),
            "raw": directory_usage(capture_root / "raw"),
        },
        "latest_hourly_partition": latest_partitions(capture_root),
        "recorder_lock": inspect_lock(capture_root),
    }

    manifest: dict[str, Any] | None = None
    heartbeat: dict[str, Any] | None = None
    try:
        manifest = _read_json(capture_root / "manifest.json", "capture manifest")
        manifest_errors = validate_manifest(manifest)
        for error in manifest_errors:
            _issue(issues, "FAILED", "MANIFEST_INCOMPATIBLE", error)
        if not manifest_errors:
            result["capture_started_at_utc"] = manifest["capture_started_at_utc"]
    except InspectionError as exc:
        _issue(issues, "FAILED", "MANIFEST_UNAVAILABLE", str(exc))

    raw_root = capture_root / "raw"
    if not raw_root.is_dir():
        _issue(issues, "FAILED", "RAW_ROOT_MISSING", f"raw capture directory is missing: {raw_root}")
    for stream in RAW_STREAMS:
        for symbol in SYMBOLS:
            path = raw_root / stream / symbol
            if not path.is_dir():
                _issue(issues, "FAILED", "STREAM_DIRECTORY_MISSING", f"required directory is missing: raw/{stream}/{symbol}")

    heartbeat_path = capture_root / "heartbeat.json"
    try:
        heartbeat = _read_json(heartbeat_path, "capture heartbeat")
        heartbeat_mtime = datetime.fromtimestamp(heartbeat_path.stat().st_mtime, tz=timezone.utc)
        result["heartbeat_age_seconds"] = age_seconds(now, heartbeat_mtime)
        result["process_started_at_utc"] = heartbeat.get("process_started_at_utc")
    except (InspectionError, OSError) as exc:
        _issue(issues, "FAILED", "HEARTBEAT_UNAVAILABLE", str(exc))

    process_age: float | None = None
    if heartbeat is not None:
        try:
            process_started = parse_utc(heartbeat.get("process_started_at_utc"), "process_started_at_utc")
            process_age = age_seconds(now, process_started)
            if process_age < -thresholds.future_clock_tolerance_seconds:
                _issue(issues, "FAILED", "PROCESS_START_IN_FUTURE", "process start timestamp is in the future")
        except InspectionError as exc:
            _issue(issues, "FAILED", "HEARTBEAT_INVALID", str(exc))

        try:
            last_message = parse_utc(heartbeat.get("last_message_at_utc"), "last_message_at_utc")
            result["last_message_age_seconds"] = age_seconds(now, last_message)
        except InspectionError:
            result["last_message_age_seconds"] = None

        age_targets = (
            ("trades", "last_trade_age_seconds"),
            ("bbo", "last_bbo_age_seconds"),
            ("l2Book", "last_book_age_seconds"),
            ("activeAssetCtx", "last_asset_ctx_age_seconds"),
            ("candle", "last_candle_age_seconds"),
        )
        for stream, output_field in age_targets:
            source = heartbeat.get(LAST_TIME_KEYS[stream])
            if not isinstance(source, Mapping):
                source = {}
            for symbol in SYMBOLS:
                try:
                    timestamp = parse_utc(source.get(symbol), f"{LAST_TIME_KEYS[stream]}.{symbol}")
                    result[output_field][symbol] = age_seconds(now, timestamp)
                except InspectionError:
                    result[output_field][symbol] = None

        messages = heartbeat.get("messages_received")
        by_stream = messages.get("by_stream_symbol") if isinstance(messages, Mapping) else None
        if not isinstance(by_stream, Mapping):
            _issue(issues, "FAILED", "MESSAGE_COUNTERS_INVALID", "heartbeat messages_received.by_stream_symbol is missing")
        else:
            for stream in HEARTBEAT_STREAMS:
                stream_counts = by_stream.get(stream)
                if not isinstance(stream_counts, Mapping):
                    _issue(issues, "FAILED", "MESSAGE_COUNTERS_INVALID", f"message counters missing for {stream}")
                    continue
                for symbol in SYMBOLS:
                    value = stream_counts.get(symbol)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        _issue(issues, "FAILED", "MESSAGE_COUNTERS_INVALID", f"message counter invalid for {stream}/{symbol}")
                    else:
                        result["messages_by_stream_symbol"][stream][symbol] = value

        reconnect_count = heartbeat.get("reconnect_count")
        integrity_count = heartbeat.get("integrity_error_count")
        if isinstance(reconnect_count, int) and not isinstance(reconnect_count, bool) and reconnect_count >= 0:
            result["reconnect_count"] = reconnect_count
        else:
            _issue(issues, "FAILED", "RECONNECT_COUNT_INVALID", "heartbeat reconnect_count is invalid")
        if isinstance(integrity_count, int) and not isinstance(integrity_count, bool) and integrity_count >= 0:
            result["integrity_error_count"] = integrity_count
            if integrity_count > allowed_integrity_errors:
                _issue(
                    issues,
                    "FAILED",
                    "INTEGRITY_ERROR_INCREASE",
                    f"integrity errors {integrity_count} exceed allowed baseline {allowed_integrity_errors}",
                )
        else:
            _issue(issues, "FAILED", "INTEGRITY_COUNT_INVALID", "heartbeat integrity_error_count is invalid")

    _check_age(
        issues,
        code="HEARTBEAT",
        label="heartbeat file",
        age=result["heartbeat_age_seconds"],
        threshold=thresholds.heartbeat,
        process_age=process_age,
        future_tolerance=thresholds.future_clock_tolerance_seconds,
    )
    _check_age(
        issues,
        code="LAST_MESSAGE",
        label="last market/control message",
        age=result["last_message_age_seconds"],
        threshold=thresholds.last_message,
        process_age=process_age,
        future_tolerance=thresholds.future_clock_tolerance_seconds,
    )
    for stream, output_field in (
        ("trades", "last_trade_age_seconds"),
        ("bbo", "last_bbo_age_seconds"),
        ("l2Book", "last_book_age_seconds"),
        ("activeAssetCtx", "last_asset_ctx_age_seconds"),
        ("candle", "last_candle_age_seconds"),
    ):
        for symbol in SYMBOLS:
            _check_age(
                issues,
                code=f"{stream}_{symbol}".upper(),
                label=f"{stream}/{symbol}",
                age=result[output_field][symbol],
                threshold=thresholds.for_stream(stream),
                process_age=process_age,
                future_tolerance=thresholds.future_clock_tolerance_seconds,
            )

    for raw_stream, heartbeat_stream in RAW_TO_HEARTBEAT_STREAM.items():
        for symbol in SYMBOLS:
            partition = result["latest_hourly_partition"][raw_stream][symbol]
            if partition is not None and partition["bytes"] > 0:
                continue
            stream_threshold = thresholds.for_stream(heartbeat_stream)
            severity = "FAILED" if process_age is None or process_age >= stream_threshold.failure_seconds else "WARNING"
            _issue(
                issues,
                severity,
                "RAW_PARTITION_MISSING",
                f"no non-empty raw partition exists for {raw_stream}/{symbol}",
            )

    lock = result["recorder_lock"]
    if lock["state"] in {"absent", "stale", "invalid"}:
        _issue(issues, "FAILED", "RECORDER_NOT_ACTIVE", f"recorder lock state is {lock['state']}")
    elif lock["state"] == "unknown_remote_owner":
        _issue(issues, "WARNING", "LOCK_OWNER_UNVERIFIED", "recorder lock belongs to another host and cannot be verified locally")

    severity_order = {"HEALTHY": 0, "WARNING": 1, "FAILED": 2}
    maximum = max((severity_order[issue["severity"]] for issue in issues), default=0)
    health = ("HEALTHY", "WARNING", "FAILED")[maximum]
    result["health"] = health
    result["capture_appears_healthy"] = health == "HEALTHY"
    result["issues"] = issues
    result["thresholds_seconds"] = {
        "heartbeat": vars(thresholds.heartbeat),
        "last_message": vars(thresholds.last_message),
        **{stream: vars(thresholds.for_stream(stream)) for stream in HEARTBEAT_STREAMS},
    }
    return result


def _decimal(value: Any, field: str, *, allow_none: bool = False) -> Decimal | None:
    if value is None and allow_none:
        return None
    if value is None or isinstance(value, bool):
        raise InspectionError(f"{field} is not numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InspectionError(f"{field} is not a decimal") from exc
    if not parsed.is_finite():
        raise InspectionError(f"{field} is not finite")
    return parsed


def _record_receive_time(record: Mapping[str, Any]) -> tuple[int, str]:
    receive_ms = record.get("receive_time_ms")
    receive_text = record.get("receive_time_utc")
    if not isinstance(receive_ms, int) or isinstance(receive_ms, bool) or receive_ms < 0:
        timestamp = parse_utc(receive_text, "receive_time_utc")
        receive_ms = int(timestamp.timestamp() * 1000)
    if not isinstance(receive_text, str):
        receive_text = utc_text(datetime.fromtimestamp(receive_ms / 1000, tz=timezone.utc))
    return receive_ms, receive_text


def _record_exchange_time(stream: str, record: Mapping[str, Any]) -> int | None:
    field = "close_time_ms" if stream == "candles_5m" else "exchange_time_ms"
    value = record.get(field)
    if value is None and stream == "asset_ctx":
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InspectionError(f"{stream}.{field} is invalid")
    return value


def _book_is_invalid(record: Mapping[str, Any]) -> bool:
    try:
        best_bid = _decimal(record.get("best_bid"), "best_bid")
        best_ask = _decimal(record.get("best_ask"), "best_ask")
        mid = _decimal(record.get("mid"), "mid")
        spread_bps = _decimal(record.get("spread_bps"), "spread_bps")
        assert best_bid is not None and best_ask is not None and mid is not None and spread_bps is not None
        if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask or mid <= 0 or spread_bps < 0:
            return True
        with localcontext() as context:
            context.prec = 50
            expected_mid = (best_bid + best_ask) / Decimal(2)
            expected_spread = (best_ask - best_bid) / expected_mid * Decimal(10_000)
        tolerance = Decimal("1e-40")
        if abs(mid - expected_mid) > tolerance or abs(spread_bps - expected_spread) > tolerance:
            return True
        previous_bid = Decimal(-1)
        previous_ask = Decimal(-1)
        for count in (1, 5, 10, 20):
            bid_depth = _decimal(record.get(f"bid_depth_{count}"), f"bid_depth_{count}")
            ask_depth = _decimal(record.get(f"ask_depth_{count}"), f"ask_depth_{count}")
            imbalance = _decimal(record.get(f"imbalance_{count}"), f"imbalance_{count}", allow_none=True)
            assert bid_depth is not None and ask_depth is not None
            if bid_depth < 0 or ask_depth < 0 or bid_depth < previous_bid or ask_depth < previous_ask:
                return True
            denominator = bid_depth + ask_depth
            if denominator == 0:
                if imbalance is not None:
                    return True
            else:
                with localcontext() as context:
                    context.prec = 50
                    expected_imbalance = (bid_depth - ask_depth) / denominator
                if imbalance is None or abs(imbalance - expected_imbalance) > tolerance:
                    return True
            previous_bid, previous_ask = bid_depth, ask_depth
        for field in ("bid_order_count_5", "ask_order_count_5"):
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return True
    except (InspectionError, AssertionError):
        return True
    return False


def _trade_identity(record: Mapping[str, Any], partition_symbol: str) -> tuple[int, str, int]:
    exchange_time = _record_exchange_time("trades", record)
    coin = record.get("coin")
    tid = record.get("tid")
    if exchange_time is None or coin != partition_symbol or not isinstance(tid, int) or isinstance(tid, bool) or tid < 0:
        raise InspectionError("trade identity is invalid")
    return exchange_time, coin, tid


def _trade_fingerprint(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in ("side", "price", "size", "notional", "hash", "signed_size", "signed_notional"))


def _completed_partitions(
    partitions: Iterable[Partition],
    *,
    now: datetime,
    include_current_hour: bool,
    lock_active: bool | None,
) -> list[Partition]:
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    capture_stopped = lock_active is False
    return [
        partition
        for partition in partitions
        if include_current_hour or capture_stopped or partition.hour < current_hour
    ]


def _aggregate_coverage(
    capture_root: Path,
    *,
    capture_started: datetime | None,
    latest_receive_by_symbol: Mapping[str, int | None],
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    malformed = 0
    for interval in ("1m", "5m"):
        interval_ms = 60_000 if interval == "1m" else 300_000
        coverage[interval] = {}
        for symbol in SYMBOLS:
            path = capture_root / "aggregates" / interval / f"{symbol}.jsonl"
            rows = 0
            first_bucket: int | None = None
            last_bucket: int | None = None
            bucket_starts: set[int] = set()
            local_malformed = 0
            if path.is_file() and not path.is_symlink():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                value = json.loads(line)
                                bucket = value.get("bucket_start_ms") if isinstance(value, Mapping) else None
                                if not isinstance(bucket, int) or isinstance(bucket, bool):
                                    raise InspectionError("aggregate bucket is invalid")
                                rows += 1
                                bucket_starts.add(bucket)
                                first_bucket = bucket if first_bucket is None else min(first_bucket, bucket)
                                last_bucket = bucket if last_bucket is None else max(last_bucket, bucket)
                            except (json.JSONDecodeError, InspectionError):
                                malformed += 1
                                local_malformed += 1
                except OSError:
                    malformed += 1
                    local_malformed += 1
            expected_buckets: set[int] | None = None
            latest_receive = latest_receive_by_symbol.get(symbol)
            if capture_started is not None and latest_receive is not None:
                first_expected = int(capture_started.timestamp() * 1000)
                first_expected -= first_expected % interval_ms
                last_expected = latest_receive - latest_receive % interval_ms
                expected_buckets = set(range(first_expected, last_expected + interval_ms, interval_ms))
            missing_buckets = sorted(expected_buckets - bucket_starts) if expected_buckets is not None else []
            extra_buckets = sorted(bucket_starts - expected_buckets) if expected_buckets is not None else []
            complete = (
                path.is_file()
                and local_malformed == 0
                and expected_buckets is not None
                and not missing_buckets
                and not extra_buckets
            )
            coverage[interval][symbol] = {
                "path": str(path.relative_to(capture_root)).replace("\\", "/"),
                "exists": path.is_file(),
                "row_count": rows,
                "expected_row_count": len(expected_buckets) if expected_buckets is not None else None,
                "first_bucket_start_utc": utc_text(datetime.fromtimestamp(first_bucket / 1000, tz=timezone.utc)) if first_bucket is not None else None,
                "last_bucket_start_utc": utc_text(datetime.fromtimestamp(last_bucket / 1000, tz=timezone.utc)) if last_bucket is not None else None,
                "latest_raw_receive_time_utc": utc_text(datetime.fromtimestamp(latest_receive / 1000, tz=timezone.utc)) if latest_receive is not None else None,
                "missing_bucket_starts_utc": [utc_text(datetime.fromtimestamp(value / 1000, tz=timezone.utc)) for value in missing_buckets],
                "unexpected_bucket_starts_utc": [utc_text(datetime.fromtimestamp(value / 1000, tz=timezone.utc)) for value in extra_buckets],
                "malformed_line_count": local_malformed,
                "coverage_complete": complete,
            }
    coverage["malformed_line_count"] = malformed
    return coverage


def continuity_audit(
    capture_root: Path,
    *,
    now: datetime | None = None,
    include_current_hour: bool = False,
    diagnostic_sample_limit: int = 100,
) -> dict[str, Any]:
    """Scan capture files without modifying them and report continuity defects."""

    now = (now or utc_now()).astimezone(timezone.utc)
    capture_root = capture_root.resolve()
    lock = inspect_lock(capture_root)
    issues: list[dict[str, str]] = []
    manifest: dict[str, Any] | None = None
    capture_started: datetime | None = None
    try:
        manifest = _read_json(capture_root / "manifest.json", "capture manifest")
        for error in validate_manifest(manifest):
            _issue(issues, "FAILED", "MANIFEST_INCOMPATIBLE", error)
        capture_started = parse_utc(manifest.get("capture_started_at_utc"), "capture_started_at_utc")
    except InspectionError as exc:
        _issue(issues, "FAILED", "MANIFEST_UNAVAILABLE", str(exc))

    raw_usage = directory_usage(capture_root / "raw")
    missing_directories: list[str] = []
    invalid_partition_paths: list[str] = []
    partition_map: dict[tuple[str, str], list[Partition]] = {}
    all_completed: list[Partition] = []
    for stream in RAW_STREAMS:
        for symbol in SYMBOLS:
            directory = capture_root / "raw" / stream / symbol
            if not directory.is_dir():
                missing_directories.append(f"raw/{stream}/{symbol}")
            partitions, invalid = discover_partitions(capture_root, stream, symbol)
            invalid_partition_paths.extend(f"raw/{stream}/{symbol}/{path}" for path in invalid)
            completed = _completed_partitions(
                partitions,
                now=now,
                include_current_hour=include_current_hour,
                lock_active=lock["active"],
            )
            partition_map[(stream, symbol)] = completed
            all_completed.extend(completed)
    for directory in missing_directories:
        _issue(issues, "FAILED", "STREAM_DIRECTORY_MISSING", f"required directory is missing: {directory}")
    for path in invalid_partition_paths:
        _issue(issues, "FAILED", "INVALID_PARTITION_PATH", f"invalid hourly partition path: {path}")

    missing_hours: list[dict[str, str]] = []
    if capture_started is not None and all_completed:
        start_hour = capture_started.replace(minute=0, second=0, microsecond=0)
        final_hour = max(partition.hour for partition in all_completed)
        for stream in RAW_STREAMS:
            for symbol in SYMBOLS:
                present = {partition.hour for partition in partition_map[(stream, symbol)]}
                expected = start_hour
                while expected <= final_hour:
                    if expected not in present:
                        missing_hours.append({"stream": stream, "symbol": symbol, "hour_utc": utc_text(expected)})
                    expected += timedelta(hours=1)
    for missing in missing_hours:
        severity = "WARNING" if missing["stream"] in {"trades", "candles_5m"} else "FAILED"
        _issue(
            issues,
            severity,
            "MISSING_HOURLY_PARTITION",
            f"missing {missing['stream']}/{missing['symbol']} partition {missing['hour_utc']}",
        )

    rows_per_hour: dict[str, dict[str, dict[str, int]]] = {
        stream: {symbol: {} for symbol in SYMBOLS} for stream in RAW_STREAMS
    }
    rows_per_hour["integrity"] = {symbol: {} for symbol in INTEGRITY_SYMBOLS}
    first_last: dict[str, dict[str, dict[str, Any]]] = {
        stream: {
            symbol: {
                "first_receive_time_utc": None,
                "last_receive_time_utc": None,
                "first_exchange_time_ms": None,
                "last_exchange_time_ms": None,
            }
            for symbol in SYMBOLS
        }
        for stream in RAW_STREAMS
    }
    first_last["integrity"] = {
        symbol: {
            "first_receive_time_utc": None,
            "last_receive_time_utc": None,
            "first_exchange_time_ms": None,
            "last_exchange_time_ms": None,
        }
        for symbol in INTEGRITY_SYMBOLS
    }
    malformed_count = 0
    invalid_record_count = 0
    receive_order_count = 0
    exchange_order_count = 0
    crossed_bbo_count = 0
    invalid_book_record_count = 0
    duplicate_trade_count = 0
    conflicting_trade_count = 0
    files_scanned = 0
    lines_scanned = 0
    audited_raw_bytes = 0
    samples: dict[str, list[dict[str, Any]]] = {
        "malformed_json_lines": [],
        "invalid_records": [],
        "receive_time_ordering": [],
        "exchange_time_ordering": [],
        "duplicate_trade_identities": [],
        "conflicting_trade_identities": [],
        "invalid_books": [],
    }
    seen_trades: dict[tuple[int, str, int], tuple[Any, ...]] = {}
    previous_receive: dict[tuple[str, str], int] = {}
    previous_exchange: dict[tuple[str, str], int] = {}
    global_first_receive: int | None = None
    global_last_receive: int | None = None
    latest_receive_by_symbol: dict[str, int | None] = {symbol: None for symbol in SYMBOLS}

    def add_sample(category: str, value: dict[str, Any]) -> None:
        if len(samples[category]) < diagnostic_sample_limit:
            samples[category].append(value)

    for stream in RAW_STREAMS:
        for symbol in SYMBOLS:
            key = (stream, symbol)
            for partition in partition_map[key]:
                files_scanned += 1
                with contextlib.suppress(OSError):
                    audited_raw_bytes += partition.path.stat().st_size
                row_count = 0
                try:
                    handle = partition.path.open("r", encoding="utf-8")
                except OSError as exc:
                    invalid_record_count += 1
                    add_sample("invalid_records", {"path": str(partition.path), "line": None, "error": str(exc)})
                    continue
                with handle:
                    for line_number, line in enumerate(handle, start=1):
                        lines_scanned += 1
                        location = {
                            "path": str(partition.path.relative_to(capture_root)).replace("\\", "/"),
                            "line": line_number,
                        }
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            malformed_count += 1
                            add_sample("malformed_json_lines", {**location, "error": str(exc)})
                            continue
                        if not isinstance(record, Mapping):
                            invalid_record_count += 1
                            add_sample("invalid_records", {**location, "error": "JSON value is not an object"})
                            continue
                        row_count += 1
                        try:
                            receive_ms, receive_text = _record_receive_time(record)
                            exchange_ms = _record_exchange_time(stream, record)
                            if record.get("coin") != symbol:
                                raise InspectionError("record symbol does not match partition")
                        except InspectionError as exc:
                            invalid_record_count += 1
                            add_sample("invalid_records", {**location, "error": str(exc)})
                            continue

                        prior_receive = previous_receive.get(key)
                        if prior_receive is not None and receive_ms < prior_receive:
                            receive_order_count += 1
                            add_sample(
                                "receive_time_ordering",
                                {**location, "previous_receive_time_ms": prior_receive, "receive_time_ms": receive_ms},
                            )
                        previous_receive[key] = max(receive_ms, prior_receive if prior_receive is not None else receive_ms)
                        if exchange_ms is not None:
                            prior_exchange = previous_exchange.get(key)
                            if prior_exchange is not None and exchange_ms < prior_exchange:
                                exchange_order_count += 1
                                add_sample(
                                    "exchange_time_ordering",
                                    {**location, "previous_exchange_time_ms": prior_exchange, "exchange_time_ms": exchange_ms},
                                )
                            previous_exchange[key] = max(exchange_ms, prior_exchange if prior_exchange is not None else exchange_ms)

                        timestamps = first_last[stream][symbol]
                        first_receive = timestamps.get("_first_receive_ms")
                        last_receive = timestamps.get("_last_receive_ms")
                        if first_receive is None or receive_ms < first_receive:
                            timestamps["_first_receive_ms"] = receive_ms
                            timestamps["first_receive_time_utc"] = receive_text
                        if last_receive is None or receive_ms > last_receive:
                            timestamps["_last_receive_ms"] = receive_ms
                            timestamps["last_receive_time_utc"] = receive_text
                        if exchange_ms is not None:
                            current_first_exchange = timestamps["first_exchange_time_ms"]
                            current_last_exchange = timestamps["last_exchange_time_ms"]
                            timestamps["first_exchange_time_ms"] = exchange_ms if current_first_exchange is None else min(current_first_exchange, exchange_ms)
                            timestamps["last_exchange_time_ms"] = exchange_ms if current_last_exchange is None else max(current_last_exchange, exchange_ms)
                        global_first_receive = receive_ms if global_first_receive is None else min(global_first_receive, receive_ms)
                        global_last_receive = receive_ms if global_last_receive is None else max(global_last_receive, receive_ms)
                        current_symbol_last = latest_receive_by_symbol[symbol]
                        latest_receive_by_symbol[symbol] = receive_ms if current_symbol_last is None else max(current_symbol_last, receive_ms)

                        if stream == "trades":
                            try:
                                identity = _trade_identity(record, symbol)
                                fingerprint = _trade_fingerprint(record)
                            except InspectionError as exc:
                                invalid_record_count += 1
                                add_sample("invalid_records", {**location, "error": str(exc)})
                            else:
                                previous = seen_trades.get(identity)
                                if previous is None:
                                    seen_trades[identity] = fingerprint
                                elif previous == fingerprint:
                                    duplicate_trade_count += 1
                                    add_sample("duplicate_trade_identities", {**location, "identity": list(identity)})
                                else:
                                    conflicting_trade_count += 1
                                    add_sample(
                                        "conflicting_trade_identities",
                                        {**location, "identity": list(identity), "existing_fingerprint": list(previous), "incoming_fingerprint": list(fingerprint)},
                                    )
                        elif stream == "bbo":
                            try:
                                bid = _decimal(record.get("bid_px"), "bid_px", allow_none=True)
                                ask = _decimal(record.get("ask_px"), "ask_px", allow_none=True)
                                crossed = record.get("is_crossed") is True or (bid is not None and ask is not None and bid > ask)
                                if crossed:
                                    crossed_bbo_count += 1
                            except InspectionError as exc:
                                invalid_record_count += 1
                                add_sample("invalid_records", {**location, "error": str(exc)})
                        elif stream == "book_5s" and _book_is_invalid(record):
                            invalid_book_record_count += 1
                            add_sample("invalid_books", location)
                rows_per_hour[stream][symbol][partition.hour_text] = row_count

    invalid_book_diagnostic_count = 0
    for symbol in INTEGRITY_SYMBOLS:
        partitions, invalid = discover_partitions(capture_root, "integrity", symbol)
        for path in invalid:
            qualified = f"raw/integrity/{symbol}/{path}"
            invalid_partition_paths.append(qualified)
            _issue(issues, "FAILED", "INVALID_PARTITION_PATH", f"invalid hourly partition path: {qualified}")
        completed = _completed_partitions(
            partitions,
            now=now,
            include_current_hour=include_current_hour,
            lock_active=lock["active"],
        )
        key = ("integrity", symbol)
        for partition in completed:
            files_scanned += 1
            with contextlib.suppress(OSError):
                audited_raw_bytes += partition.path.stat().st_size
            row_count = 0
            try:
                handle = partition.path.open("r", encoding="utf-8")
            except OSError as exc:
                invalid_record_count += 1
                add_sample("invalid_records", {"path": str(partition.path), "line": None, "error": str(exc)})
                continue
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    lines_scanned += 1
                    location = {
                        "path": str(partition.path.relative_to(capture_root)).replace("\\", "/"),
                        "line": line_number,
                    }
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        malformed_count += 1
                        add_sample("malformed_json_lines", {**location, "error": str(exc)})
                        continue
                    if not isinstance(value, Mapping):
                        invalid_record_count += 1
                        add_sample("invalid_records", {**location, "error": "JSON value is not an object"})
                        continue
                    row_count += 1
                    try:
                        receive_ms, receive_text = _record_receive_time(value)
                        expected_coin = None if symbol == "GLOBAL" else symbol
                        if value.get("coin") != expected_coin:
                            raise InspectionError("integrity record symbol does not match partition")
                        if not isinstance(value.get("kind"), str) or not value.get("kind"):
                            raise InspectionError("integrity record kind is missing")
                    except InspectionError as exc:
                        invalid_record_count += 1
                        add_sample("invalid_records", {**location, "error": str(exc)})
                        continue
                    prior_receive = previous_receive.get(key)
                    if prior_receive is not None and receive_ms < prior_receive:
                        receive_order_count += 1
                        add_sample(
                            "receive_time_ordering",
                            {**location, "previous_receive_time_ms": prior_receive, "receive_time_ms": receive_ms},
                        )
                    previous_receive[key] = max(receive_ms, prior_receive if prior_receive is not None else receive_ms)
                    timestamps = first_last["integrity"][symbol]
                    first_receive = timestamps.get("_first_receive_ms")
                    last_receive = timestamps.get("_last_receive_ms")
                    if first_receive is None or receive_ms < first_receive:
                        timestamps["_first_receive_ms"] = receive_ms
                        timestamps["first_receive_time_utc"] = receive_text
                    if last_receive is None or receive_ms > last_receive:
                        timestamps["_last_receive_ms"] = receive_ms
                        timestamps["last_receive_time_utc"] = receive_text
                    global_first_receive = receive_ms if global_first_receive is None else min(global_first_receive, receive_ms)
                    global_last_receive = receive_ms if global_last_receive is None else max(global_last_receive, receive_ms)
                    if value.get("kind") == "invalid_book":
                        invalid_book_diagnostic_count += 1
            rows_per_hour["integrity"][symbol][partition.hour_text] = row_count

    for stream in (*RAW_STREAMS, "integrity"):
        symbols = INTEGRITY_SYMBOLS if stream == "integrity" else SYMBOLS
        for symbol in symbols:
            first_last[stream][symbol].pop("_first_receive_ms", None)
            first_last[stream][symbol].pop("_last_receive_ms", None)

    counters = (
        (malformed_count, "FAILED", "MALFORMED_JSON", "malformed JSON lines"),
        (invalid_record_count, "FAILED", "INVALID_RECORD", "invalid records"),
        (duplicate_trade_count, "FAILED", "DUPLICATE_TRADE", "duplicate trade identities"),
        (conflicting_trade_count, "FAILED", "CONFLICTING_TRADE", "conflicting trade identities"),
        (receive_order_count, "FAILED", "RECEIVE_TIME_ORDER", "receive-time ordering issues"),
        (exchange_order_count, "FAILED", "EXCHANGE_TIME_ORDER", "exchange-time ordering issues"),
        (invalid_book_record_count + invalid_book_diagnostic_count, "FAILED", "INVALID_BOOK", "invalid books"),
        (crossed_bbo_count, "WARNING", "CROSSED_BBO", "crossed BBO rows"),
    )
    for count, severity, code, label in counters:
        if count:
            _issue(issues, severity, code, f"found {count} {label}")

    coverage = _aggregate_coverage(
        capture_root,
        capture_started=capture_started,
        latest_receive_by_symbol=latest_receive_by_symbol,
    )
    for interval in ("1m", "5m"):
        for symbol in SYMBOLS:
            if not coverage[interval][symbol]["exists"]:
                _issue(issues, "WARNING", "AGGREGATE_MISSING", f"aggregate output missing for {interval}/{symbol}")
            elif not coverage[interval][symbol]["coverage_complete"]:
                _issue(issues, "WARNING", "AGGREGATE_INCOMPLETE", f"aggregate coverage is incomplete for {interval}/{symbol}")
    if coverage["malformed_line_count"]:
        _issue(issues, "WARNING", "AGGREGATE_MALFORMED", f"found {coverage['malformed_line_count']} malformed aggregate lines")

    if not all_completed:
        _issue(issues, "WARNING", "NO_COMPLETED_PARTITIONS", "no completed hourly partitions were available to audit")

    observed_duration = (
        round((global_last_receive - global_first_receive) / 1000, 3)
        if global_first_receive is not None and global_last_receive is not None
        else None
    )
    capture_to_last = (
        round(global_last_receive / 1000 - capture_started.timestamp(), 3)
        if global_last_receive is not None and capture_started is not None
        else None
    )
    severity_order = {"PASSED": 0, "WARNING": 1, "FAILED": 2}
    maximum = max((severity_order[issue["severity"]] for issue in issues), default=0)
    audit_status = ("PASSED", "WARNING", "FAILED")[maximum]
    return {
        "mode": "offline_continuity_audit",
        "capture_root": str(capture_root),
        "current_utc": utc_text(now),
        "capture_started_at_utc": manifest.get("capture_started_at_utc") if manifest else None,
        "completed_files_only": not include_current_hour and lock["active"] is not False,
        "include_current_hour": include_current_hour,
        "recorder_lock": lock,
        "audit_status": audit_status,
        "missing_stream_directories": missing_directories,
        "missing_hourly_partitions": missing_hours,
        "invalid_partition_paths": invalid_partition_paths,
        "malformed_json_line_count": malformed_count,
        "invalid_record_count": invalid_record_count,
        "duplicate_trade_identity_count": duplicate_trade_count,
        "conflicting_trade_identity_count": conflicting_trade_count,
        "receive_time_ordering_issue_count": receive_order_count,
        "exchange_time_ordering_issue_count": exchange_order_count,
        "crossed_bbo_count": crossed_bbo_count,
        "invalid_book_record_count": invalid_book_record_count,
        "invalid_book_diagnostic_count": invalid_book_diagnostic_count,
        "invalid_book_count": invalid_book_record_count + invalid_book_diagnostic_count,
        "rows_per_hour": rows_per_hour,
        "first_last_timestamps": first_last,
        "capture_duration_seconds": observed_duration,
        "capture_start_to_last_receive_seconds": capture_to_last,
        "raw_bytes": raw_usage["bytes"],
        "raw_file_count": raw_usage["file_count"],
        "audited_raw_bytes": audited_raw_bytes,
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "aggregate_coverage": coverage,
        "diagnostic_samples": samples,
        "diagnostic_sample_limit": diagnostic_sample_limit,
        "issues": issues,
    }


def _format_age(value: Any) -> str:
    return "missing" if value is None else f"{value:.3f}s"


def format_status(result: Mapping[str, Any]) -> str:
    lines = [
        f"HEALTH: {result['health']}",
        f"capture root: {result['capture_root']}",
        f"capture started: {result['capture_started_at_utc']}",
        f"process started: {result['process_started_at_utc']}",
        f"current UTC: {result['current_utc']}",
        f"heartbeat age: {_format_age(result['heartbeat_age_seconds'])}",
        f"last message age: {_format_age(result['last_message_age_seconds'])}",
        f"reconnects: {result['reconnect_count']}",
        f"integrity errors: {result['integrity_error_count']}",
        f"lock: {result['recorder_lock']['state']}",
        f"disk: capture={result['disk_usage']['capture']['bytes']} bytes raw={result['disk_usage']['raw']['bytes']} bytes",
    ]
    age_fields = (
        ("trade", "last_trade_age_seconds"),
        ("BBO", "last_bbo_age_seconds"),
        ("book", "last_book_age_seconds"),
        ("asset context", "last_asset_ctx_age_seconds"),
        ("candle", "last_candle_age_seconds"),
    )
    for label, field in age_fields:
        lines.append(f"{label} ages: " + " ".join(f"{symbol}={_format_age(result[field][symbol])}" for symbol in SYMBOLS))
    lines.append("messages by stream/symbol:")
    for stream in HEARTBEAT_STREAMS:
        values = result["messages_by_stream_symbol"][stream]
        lines.append(f"  {stream}: " + " ".join(f"{symbol}={values[symbol]}" for symbol in SYMBOLS))
    lines.append("latest hourly partitions:")
    for stream in RAW_STREAMS:
        for symbol in SYMBOLS:
            value = result["latest_hourly_partition"][stream][symbol]
            lines.append(f"  {stream}/{symbol}: {value['path'] if value else 'missing'}")
    if result["issues"]:
        lines.append("issues:")
        lines.extend(f"  [{issue['severity']}] {issue['code']}: {issue['message']}" for issue in result["issues"])
    return "\n".join(lines)


def format_audit(result: Mapping[str, Any]) -> str:
    lines = [
        f"AUDIT: {result['audit_status']}",
        f"capture root: {result['capture_root']}",
        f"capture started: {result['capture_started_at_utc']}",
        f"current UTC: {result['current_utc']}",
        f"files/lines scanned: {result['files_scanned']}/{result['lines_scanned']}",
        f"raw bytes: {result['raw_bytes']}",
        f"capture duration: {_format_age(result['capture_duration_seconds'])}",
        f"missing hourly partitions: {len(result['missing_hourly_partitions'])}",
        f"malformed JSON lines: {result['malformed_json_line_count']}",
        f"duplicate/conflicting trades: {result['duplicate_trade_identity_count']}/{result['conflicting_trade_identity_count']}",
        f"receive/exchange ordering issues: {result['receive_time_ordering_issue_count']}/{result['exchange_time_ordering_issue_count']}",
        f"crossed BBO/invalid books: {result['crossed_bbo_count']}/{result['invalid_book_count']}",
        "rows per hour:",
    ]
    for stream in (*RAW_STREAMS, "integrity"):
        symbols = INTEGRITY_SYMBOLS if stream == "integrity" else SYMBOLS
        for symbol in symbols:
            values = result["rows_per_hour"][stream][symbol]
            rendered = " ".join(f"{hour}={count}" for hour, count in values.items()) or "none"
            lines.append(f"  {stream}/{symbol}: {rendered}")
    if result["issues"]:
        lines.append("issues:")
        lines.extend(f"  [{issue['severity']}] {issue['code']}: {issue['message']}" for issue in result["issues"])
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Hyperliquid recorder status and continuity audit")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--audit", action="store_true", help="scan completed hourly files offline")
    parser.add_argument("--include-current-hour", action="store_true", help="audit the current UTC partition too")
    parser.add_argument("--allowed-integrity-errors", type=int, default=0)
    parser.add_argument("--now", type=parse_utc_argument, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.allowed_integrity_errors < 0:
        print("--allowed-integrity-errors must be non-negative", file=sys.stderr)
        return 2
    if arguments.audit:
        result = continuity_audit(
            arguments.capture_root,
            now=arguments.now,
            include_current_hour=arguments.include_current_hour,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")) if arguments.json else format_audit(result))
        return {"PASSED": 0, "WARNING": 1, "FAILED": 2}[result["audit_status"]]
    if arguments.include_current_hour:
        print("--include-current-hour requires --audit", file=sys.stderr)
        return 2
    result = evaluate_status(
        arguments.capture_root,
        now=arguments.now,
        allowed_integrity_errors=arguments.allowed_integrity_errors,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")) if arguments.json else format_status(result))
    return {"HEALTHY": 0, "WARNING": 1, "FAILED": 2}[result["health"]]


if __name__ == "__main__":
    sys.exit(main())
