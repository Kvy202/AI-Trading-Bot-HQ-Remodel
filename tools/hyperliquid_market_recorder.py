"""Passive Hyperliquid mainnet public market-data recorder.

This module deliberately has one network capability: an aiohttp WebSocket
connection to the public market-data URL declared below.  It does not import
the Hyperliquid SDK and requires no account identity.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import aiohttp


LOGGER = logging.getLogger("hyperliquid_market_recorder")

WS_URL = "wss://api.hyperliquid.xyz/ws"
NETWORK = "mainnet"
SYMBOLS: tuple[str, ...] = ("BTC", "ETH")
BOOK_SAMPLING_SECONDS = 5
SCHEMA_VERSION = "1.0.0"
CAPTURE_CONTRACT_VERSION = "forward-public-market-data-v1"
DEFAULT_OUTPUT_ROOT = Path("data/hyperliquid_market_capture")
STREAMS: tuple[str, ...] = ("trades", "bbo", "l2Book", "activeAssetCtx", "candle")


def build_subscriptions() -> list[dict[str, str]]:
    """Return the complete, immutable public subscription set."""

    subscriptions: list[dict[str, str]] = []
    for coin in SYMBOLS:
        subscriptions.extend(
            [
                {"type": "trades", "coin": coin},
                {"type": "bbo", "coin": coin},
                {"type": "l2Book", "coin": coin},
                {"type": "activeAssetCtx", "coin": coin},
                {"type": "candle", "coin": coin, "interval": "5m"},
            ]
        )
    return subscriptions


SUBSCRIPTIONS = build_subscriptions()


class RecorderError(RuntimeError):
    """Base class for recorder failures."""


class ValidationError(RecorderError):
    """A public feed message failed schema or integrity validation."""


class NumericValidationError(ValidationError):
    """A numeric value was missing, malformed, or non-finite."""


class ActiveRecorderError(RecorderError):
    """Another process may own the capture directory."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return int(value.timestamp() * 1000)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace a small JSON control file without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _decimal(value: Any, field: str, *, minimum: Decimal | None = None) -> Decimal:
    if value is None or isinstance(value, bool):
        raise NumericValidationError(f"{field} is not a numeric value")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise NumericValidationError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise NumericValidationError(f"{field} is not finite")
    if minimum is not None and parsed < minimum:
        raise NumericValidationError(f"{field} is below {minimum}")
    return parsed


def _optional_decimal(value: Any, field: str, *, minimum: Decimal | None = None) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, field, minimum=minimum)


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise NumericValidationError(f"{field} is not an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip() and value.strip().lstrip("-").isdigit():
        parsed = int(value)
    else:
        raise NumericValidationError(f"{field} is not an integer")
    if minimum is not None and parsed < minimum:
        raise NumericValidationError(f"{field} is below {minimum}")
    return parsed


def decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _ratio_bps(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator * Decimal(10_000)


def _coin(value: Any) -> str:
    if not isinstance(value, str) or value not in SYMBOLS:
        raise ValidationError(f"unsupported coin: {value!r}")
    return value


def _receive_fields(received_at: datetime) -> dict[str, Any]:
    return {"receive_time_utc": utc_text(received_at), "receive_time_ms": epoch_ms(received_at)}


def parse_trade(value: Mapping[str, Any], received_at: datetime) -> dict[str, Any]:
    coin = _coin(value.get("coin"))
    exchange_time_ms = _integer(value.get("time"), "trade.time", minimum=0)
    side_code = value.get("side")
    if side_code == "B":
        side = "BUY"
        sign = Decimal(1)
    elif side_code == "A":
        side = "SELL"
        sign = Decimal(-1)
    else:
        raise ValidationError(f"unknown aggressor side: {side_code!r}")
    price = _decimal(value.get("px"), "trade.px", minimum=Decimal(0))
    size = _decimal(value.get("sz"), "trade.sz", minimum=Decimal(0))
    if price == 0 or size == 0:
        raise NumericValidationError("trade price and size must be positive")
    tid = _integer(value.get("tid"), "trade.tid", minimum=0)
    trade_hash = value.get("hash")
    if not isinstance(trade_hash, str) or not trade_hash:
        raise ValidationError("trade.hash is missing")
    with localcontext() as context:
        context.prec = 50
        notional = price * size
    return {
        **_receive_fields(received_at),
        "exchange_time_ms": exchange_time_ms,
        "coin": coin,
        "side": side,
        "price": decimal_text(price),
        "size": decimal_text(size),
        "notional": decimal_text(notional),
        "tid": tid,
        "hash": trade_hash,
        "signed_size": decimal_text(sign * size),
        "signed_notional": decimal_text(sign * notional),
    }


def _parse_level(value: Any, field: str) -> tuple[Decimal, Decimal, int]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} is not an object")
    price = _decimal(value.get("px"), f"{field}.px", minimum=Decimal(0))
    size = _decimal(value.get("sz"), f"{field}.sz", minimum=Decimal(0))
    count = _integer(value.get("n"), f"{field}.n", minimum=0)
    if price == 0:
        raise NumericValidationError(f"{field}.px must be positive")
    return price, size, count


def parse_bbo(value: Mapping[str, Any], received_at: datetime) -> tuple[dict[str, Any], bool]:
    coin = _coin(value.get("coin"))
    exchange_time_ms = _integer(value.get("time"), "bbo.time", minimum=0)
    bbo = value.get("bbo")
    if not isinstance(bbo, Sequence) or isinstance(bbo, (str, bytes)) or len(bbo) != 2:
        raise ValidationError("bbo.bbo must contain bid and ask")
    bid = None if bbo[0] is None else _parse_level(bbo[0], "bbo.bid")
    ask = None if bbo[1] is None else _parse_level(bbo[1], "bbo.ask")
    bid_px, bid_sz = (bid[0], bid[1]) if bid else (None, None)
    ask_px, ask_sz = (ask[0], ask[1]) if ask else (None, None)
    crossed = bool(bid_px is not None and ask_px is not None and bid_px > ask_px)
    mid: Decimal | None = None
    spread: Decimal | None = None
    spread_bps: Decimal | None = None
    if bid_px is not None and ask_px is not None and not crossed:
        with localcontext() as context:
            context.prec = 50
            mid = (bid_px + ask_px) / Decimal(2)
            spread = ask_px - bid_px
            spread_bps = _ratio_bps(spread, mid)
    return (
        {
            **_receive_fields(received_at),
            "exchange_time_ms": exchange_time_ms,
            "coin": coin,
            "bid_px": decimal_text(bid_px),
            "bid_sz": decimal_text(bid_sz),
            "ask_px": decimal_text(ask_px),
            "ask_sz": decimal_text(ask_sz),
            "mid_px": decimal_text(mid),
            "spread": decimal_text(spread),
            "spread_bps": decimal_text(spread_bps),
            "is_crossed": crossed,
        },
        crossed,
    )


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    size: Decimal
    order_count: int


def _book_levels(values: Any, side: str) -> list[BookLevel]:
    if not isinstance(values, list) or not values:
        raise ValidationError(f"book {side} levels are missing")
    parsed = [BookLevel(*_parse_level(value, f"book.{side}[{index}]")) for index, value in enumerate(values)]
    prices = [level.price for level in parsed]
    if side == "bids" and any(left <= right for left, right in zip(prices, prices[1:])):
        raise ValidationError("book bids are not strictly descending")
    if side == "asks" and any(left >= right for left, right in zip(prices, prices[1:])):
        raise ValidationError("book asks are not strictly ascending")
    return parsed


def _depth(levels: Sequence[BookLevel], count: int) -> Decimal:
    return sum((level.size for level in levels[:count]), Decimal(0))


def _imbalance(bid_depth: Decimal, ask_depth: Decimal) -> Decimal | None:
    denominator = bid_depth + ask_depth
    if denominator == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return (bid_depth - ask_depth) / denominator


def derive_book_sample(value: Mapping[str, Any], received_at: datetime) -> dict[str, Any]:
    coin = _coin(value.get("coin"))
    exchange_time_ms = _integer(value.get("time"), "book.time", minimum=0)
    levels = value.get("levels")
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) != 2:
        raise ValidationError("book.levels must contain bids and asks")
    bids = _book_levels(levels[0], "bids")
    asks = _book_levels(levels[1], "asks")
    best_bid = bids[0].price
    best_ask = asks[0].price
    if best_bid > best_ask:
        raise ValidationError("book is crossed")
    with localcontext() as context:
        context.prec = 50
        mid = (best_bid + best_ask) / Decimal(2)
        spread_bps = _ratio_bps(best_ask - best_bid, mid)
        top_denominator = bids[0].size + asks[0].size
        microprice = (
            (best_ask * bids[0].size + best_bid * asks[0].size) / top_denominator
            if top_denominator != 0
            else None
        )
        microprice_minus_mid_bps = (
            _ratio_bps(microprice - mid, mid) if microprice is not None else None
        )
    record: dict[str, Any] = {
        **_receive_fields(received_at),
        "exchange_time_ms": exchange_time_ms,
        "coin": coin,
        "best_bid": decimal_text(best_bid),
        "best_ask": decimal_text(best_ask),
        "mid": decimal_text(mid),
        "spread_bps": decimal_text(spread_bps),
    }
    for count in (1, 5, 10, 20):
        bid_depth = _depth(bids, count)
        ask_depth = _depth(asks, count)
        record[f"bid_depth_{count}"] = decimal_text(bid_depth)
        record[f"ask_depth_{count}"] = decimal_text(ask_depth)
        record[f"imbalance_{count}"] = decimal_text(_imbalance(bid_depth, ask_depth))
    record.update(
        {
            "bid_order_count_5": sum(level.order_count for level in bids[:5]),
            "ask_order_count_5": sum(level.order_count for level in asks[:5]),
            "microprice": decimal_text(microprice),
            "microprice_minus_mid_bps": decimal_text(microprice_minus_mid_bps),
        }
    )
    return record


def parse_asset_ctx(value: Mapping[str, Any], received_at: datetime) -> dict[str, Any]:
    coin = _coin(value.get("coin"))
    ctx = value.get("ctx")
    if not isinstance(ctx, Mapping):
        raise ValidationError("activeAssetCtx.ctx is missing")
    exchange_time_ms = None
    if value.get("time") is not None:
        exchange_time_ms = _integer(value.get("time"), "activeAssetCtx.time", minimum=0)
    mark = _optional_decimal(ctx.get("markPx"), "activeAssetCtx.markPx", minimum=Decimal(0))
    mid = _optional_decimal(ctx.get("midPx"), "activeAssetCtx.midPx", minimum=Decimal(0))
    oracle = _optional_decimal(ctx.get("oraclePx"), "activeAssetCtx.oraclePx", minimum=Decimal(0))
    funding = _optional_decimal(ctx.get("funding"), "activeAssetCtx.funding")
    open_interest = _optional_decimal(ctx.get("openInterest"), "activeAssetCtx.openInterest", minimum=Decimal(0))
    day_notional = _optional_decimal(ctx.get("dayNtlVlm"), "activeAssetCtx.dayNtlVlm", minimum=Decimal(0))
    previous_day = _optional_decimal(ctx.get("prevDayPx"), "activeAssetCtx.prevDayPx", minimum=Decimal(0))
    return {
        **_receive_fields(received_at),
        "exchange_time_ms": exchange_time_ms,
        "coin": coin,
        "mark_px": decimal_text(mark),
        "mid_px": decimal_text(mid),
        "oracle_px": decimal_text(oracle),
        "funding": decimal_text(funding),
        "open_interest": decimal_text(open_interest),
        "day_notional_volume": decimal_text(day_notional),
        "previous_day_price": decimal_text(previous_day),
        "mark_oracle_basis_bps": decimal_text(_ratio_bps(mark - oracle, oracle) if mark is not None and oracle is not None else None),
        "mid_oracle_basis_bps": decimal_text(_ratio_bps(mid - oracle, oracle) if mid is not None and oracle is not None else None),
        "mark_mid_basis_bps": decimal_text(_ratio_bps(mark - mid, mid) if mark is not None and mid is not None else None),
    }


def parse_candle(value: Mapping[str, Any], received_at: datetime) -> dict[str, Any]:
    coin = _coin(value.get("s"))
    if value.get("i") != "5m":
        raise ValidationError(f"unexpected candle interval: {value.get('i')!r}")
    open_time_ms = _integer(value.get("t"), "candle.t", minimum=0)
    close_time_ms = _integer(value.get("T"), "candle.T", minimum=open_time_ms)
    open_price = _decimal(value.get("o"), "candle.o", minimum=Decimal(0))
    high = _decimal(value.get("h"), "candle.h", minimum=Decimal(0))
    low = _decimal(value.get("l"), "candle.l", minimum=Decimal(0))
    close = _decimal(value.get("c"), "candle.c", minimum=Decimal(0))
    volume = _decimal(value.get("v"), "candle.v", minimum=Decimal(0))
    trade_count = _integer(value.get("n"), "candle.n", minimum=0)
    if min(open_price, high, low, close) == 0:
        raise NumericValidationError("candle prices must be positive")
    if high < max(open_price, close) or low > min(open_price, close) or high < low:
        raise ValidationError("candle OHLC bounds are inconsistent")
    return {
        **_receive_fields(received_at),
        "open_time_ms": open_time_ms,
        "close_time_ms": close_time_ms,
        "coin": coin,
        "interval": "5m",
        "open": decimal_text(open_price),
        "high": decimal_text(high),
        "low": decimal_text(low),
        "close": decimal_text(close),
        "volume": decimal_text(volume),
        "trade_count": trade_count,
    }


class HourlyJsonlWriter:
    """Append canonical JSON records to receive-time UTC hourly partitions."""

    def __init__(self, root: Path, *, fsync_interval_seconds: float = 1.0) -> None:
        self.root = root
        self.raw_root = root / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.fsync_interval_seconds = fsync_interval_seconds
        self._handles: dict[tuple[str, str], tuple[str, Any]] = {}
        self._last_fsync: dict[tuple[str, str], float] = {}

    def path_for(self, stream: str, coin: str, received_at: datetime) -> Path:
        utc = received_at.astimezone(timezone.utc)
        return self.raw_root / stream / coin / utc.strftime("%Y-%m-%d") / f"{utc:%H}.jsonl"

    def append(self, stream: str, coin: str, received_at: datetime, record: Mapping[str, Any]) -> Path:
        key = (stream, coin)
        path = self.path_for(stream, coin, received_at)
        partition = str(path)
        current = self._handles.get(key)
        if current is None or current[0] != partition:
            if current is not None:
                self._close_handle(key, current[1])
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8", newline="\n", buffering=1)
            self._handles[key] = (partition, handle)
            self._last_fsync[key] = 0.0
        handle = self._handles[key][1]
        handle.write(canonical_json(record))
        handle.write("\n")
        handle.flush()
        loop_time = asyncio.get_running_loop().time() if _has_running_loop() else 0.0
        if loop_time - self._last_fsync.get(key, 0.0) >= self.fsync_interval_seconds:
            os.fsync(handle.fileno())
            self._last_fsync[key] = loop_time
        return path

    def flush(self, *, durable: bool = False) -> None:
        for _, handle in self._handles.values():
            handle.flush()
            if durable:
                os.fsync(handle.fileno())

    def _close_handle(self, key: tuple[str, str], handle: Any) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        self._last_fsync.pop(key, None)

    def close(self) -> None:
        for key, (_, handle) in list(self._handles.items()):
            self._close_handle(key, handle)
        self._handles.clear()


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class TradeDeduplicator:
    """Bounded exact-identity trade dedupe, seedable from recent partitions."""

    FINGERPRINT_FIELDS = ("side", "price", "size", "notional", "hash")

    def __init__(self, *, max_entries: int = 1_000_000) -> None:
        self.max_entries = max_entries
        self._seen: OrderedDict[tuple[int, str, int], tuple[Any, ...]] = OrderedDict()

    @staticmethod
    def identity(record: Mapping[str, Any]) -> tuple[int, str, int]:
        return int(record["exchange_time_ms"]), str(record["coin"]), int(record["tid"])

    @classmethod
    def fingerprint(cls, record: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(record.get(field) for field in cls.FINGERPRINT_FIELDS)

    def check(self, record: Mapping[str, Any]) -> tuple[str, tuple[Any, ...] | None]:
        identity = self.identity(record)
        fingerprint = self.fingerprint(record)
        existing = self._seen.get(identity)
        if existing is not None:
            self._seen.move_to_end(identity)
            return ("duplicate" if existing == fingerprint else "conflict"), existing
        self._seen[identity] = fingerprint
        if len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return "new", None

    def seed_from_capture(self, root: Path, *, files_per_symbol: int = 4) -> int:
        loaded = 0
        for coin in SYMBOLS:
            files = sorted((root / "raw" / "trades" / coin).glob("*/*.jsonl"))[-files_per_symbol:]
            for path in files:
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                record = json.loads(line)
                                self.check(record)
                                loaded += 1
                            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                                continue
                except FileNotFoundError:
                    continue
        return loaded


class HeartbeatState:
    def __init__(self, process_started_at: datetime) -> None:
        empty_by_symbol = {coin: None for coin in SYMBOLS}
        self.value: dict[str, Any] = {
            "process_started_at_utc": utc_text(process_started_at),
            "last_message_at_utc": None,
            "last_trade_at_utc": dict(empty_by_symbol),
            "last_bbo_at_utc": dict(empty_by_symbol),
            "last_book_at_utc": dict(empty_by_symbol),
            "last_asset_ctx_at_utc": dict(empty_by_symbol),
            "last_candle_at_utc": dict(empty_by_symbol),
            "messages_received": {
                "total": 0,
                "control": 0,
                "by_stream_symbol": {stream: {coin: 0 for coin in SYMBOLS} for stream in STREAMS},
            },
            "rows_written": {
                "trades": {coin: 0 for coin in SYMBOLS},
                "bbo": {coin: 0 for coin in SYMBOLS},
                "book_5s": {coin: 0 for coin in SYMBOLS},
                "asset_ctx": {coin: 0 for coin in SYMBOLS},
                "candles_5m": {coin: 0 for coin in SYMBOLS},
                "integrity": {coin: 0 for coin in (*SYMBOLS, "GLOBAL")},
            },
            "reconnect_count": 0,
            "integrity_error_count": 0,
            "data_quality": {
                "duplicate_trade_count": 0,
                "conflicting_duplicate_count": 0,
                "invalid_numeric_count": 0,
                "crossed_bbo_count": 0,
                "invalid_book_count": 0,
                "out_of_order_timestamp_count": 0,
                "malformed_message_count": 0,
                "websocket_reconnect_count": 0,
            },
        }

    def note_control(self, received_at: datetime) -> None:
        self.value["last_message_at_utc"] = utc_text(received_at)
        self.value["messages_received"]["total"] += 1
        self.value["messages_received"]["control"] += 1

    def note_message(self, stream: str, coin: str, received_at: datetime) -> None:
        self.value["last_message_at_utc"] = utc_text(received_at)
        self.value["messages_received"]["total"] += 1
        self.value["messages_received"]["by_stream_symbol"][stream][coin] += 1
        last_key = {
            "trades": "last_trade_at_utc",
            "bbo": "last_bbo_at_utc",
            "l2Book": "last_book_at_utc",
            "activeAssetCtx": "last_asset_ctx_at_utc",
            "candle": "last_candle_at_utc",
        }[stream]
        self.value[last_key][coin] = utc_text(received_at)

    def note_row(self, stream: str, coin: str) -> None:
        self.value["rows_written"][stream][coin] += 1

    def note_quality(self, key: str, *, integrity: bool = False) -> None:
        self.value["data_quality"][key] += 1
        if integrity:
            self.value["integrity_error_count"] += 1

    def note_reconnect(self) -> None:
        self.value["reconnect_count"] += 1
        self.value["data_quality"]["websocket_reconnect_count"] += 1


class MessageProcessor:
    def __init__(
        self,
        writer: HourlyJsonlWriter,
        heartbeat: HeartbeatState,
        deduplicator: TradeDeduplicator,
        *,
        book_sampling_seconds: int = BOOK_SAMPLING_SECONDS,
    ) -> None:
        if book_sampling_seconds < BOOK_SAMPLING_SECONDS:
            raise ValueError("book sampling cannot be more frequent than five seconds")
        self.writer = writer
        self.heartbeat = heartbeat
        self.deduplicator = deduplicator
        self.book_sampling_seconds = book_sampling_seconds
        self._last_book_sample_ms: dict[str, int] = {}
        self._last_book_fingerprint: dict[str, str] = {}
        self._last_exchange_time: dict[tuple[str, str], int] = {}
        self.latest_books: dict[str, Mapping[str, Any]] = {}

    def seed_book_sampling_from_capture(self, root: Path) -> None:
        for coin in SYMBOLS:
            files = sorted((root / "raw" / "book_5s" / coin).glob("*/*.jsonl"))
            if not files:
                continue
            last_line = _last_complete_json_line(files[-1])
            if last_line is not None and isinstance(last_line.get("receive_time_ms"), int):
                self._last_book_sample_ms[coin] = last_line["receive_time_ms"]

    def _write(self, stream: str, coin: str, received_at: datetime, record: Mapping[str, Any]) -> None:
        self.writer.append(stream, coin, received_at, record)
        self.heartbeat.note_row(stream, coin)

    def _integrity(
        self,
        kind: str,
        coin: str | None,
        received_at: datetime,
        *,
        details: Mapping[str, Any],
        quality_key: str | None = None,
    ) -> None:
        target = coin if coin in SYMBOLS else "GLOBAL"
        record = {
            **_receive_fields(received_at),
            "kind": kind,
            "coin": coin,
            "details": dict(details),
        }
        self._write("integrity", target, received_at, record)
        if quality_key is not None:
            self.heartbeat.note_quality(quality_key, integrity=True)
        else:
            self.heartbeat.value["integrity_error_count"] += 1

    def _track_timestamp(self, stream: str, coin: str, timestamp: int, received_at: datetime) -> None:
        key = (stream, coin)
        previous = self._last_exchange_time.get(key)
        if previous is not None and timestamp < previous:
            self._integrity(
                "out_of_order_timestamp",
                coin,
                received_at,
                details={"stream": stream, "previous_exchange_time_ms": previous, "exchange_time_ms": timestamp},
                quality_key="out_of_order_timestamp_count",
            )
        self._last_exchange_time[key] = max(timestamp, previous if previous is not None else timestamp)

    def handle_message(self, message: Any, received_at: datetime | None = None) -> None:
        received_at = received_at or utc_now()
        if not isinstance(message, Mapping):
            self.heartbeat.note_control(received_at)
            self._malformed("message is not an object", None, message, received_at)
            return
        channel = message.get("channel")
        if channel in {"pong", "subscriptionResponse"}:
            self.heartbeat.note_control(received_at)
            return
        try:
            if channel == "trades":
                self._handle_trades(message.get("data"), received_at)
            elif channel == "bbo":
                self._handle_bbo(message.get("data"), received_at)
            elif channel == "l2Book":
                self._handle_book(message.get("data"), received_at)
            elif channel in {"activeAssetCtx", "activeSpotAssetCtx"}:
                self._handle_asset_ctx(message.get("data"), received_at)
            elif channel == "candle":
                self._handle_candle(message.get("data"), received_at)
            else:
                self.heartbeat.note_control(received_at)
                self._malformed(f"unexpected channel: {channel!r}", None, message, received_at)
        except Exception as exc:  # A malformed public message must not kill collection.
            coin = _best_effort_coin(message.get("data"))
            self._malformed(str(exc), coin, message, received_at, numeric=isinstance(exc, NumericValidationError))

    def _handle_trades(self, data: Any, received_at: datetime) -> None:
        if not isinstance(data, list):
            raise ValidationError("trades.data is not a list")
        coins = {_best_effort_coin(item) for item in data if isinstance(item, Mapping)} - {None}
        for coin in sorted(coins):
            self.heartbeat.note_message("trades", coin, received_at)
        if not data:
            self.heartbeat.note_control(received_at)
        for item in data:
            try:
                if not isinstance(item, Mapping):
                    raise ValidationError("trade is not an object")
                record = parse_trade(item, received_at)
                coin = record["coin"]
                self._track_timestamp("trades", coin, record["exchange_time_ms"], received_at)
                status, existing = self.deduplicator.check(record)
                if status == "duplicate":
                    self.heartbeat.note_quality("duplicate_trade_count")
                    continue
                if status == "conflict":
                    self._integrity(
                        "conflicting_trade_duplicate",
                        coin,
                        received_at,
                        details={
                            "identity": list(TradeDeduplicator.identity(record)),
                            "existing_fingerprint": list(existing or ()),
                            "incoming_fingerprint": list(TradeDeduplicator.fingerprint(record)),
                        },
                        quality_key="conflicting_duplicate_count",
                    )
                    continue
                self._write("trades", coin, received_at, record)
            except Exception as exc:
                coin = _best_effort_coin(item)
                self._malformed(str(exc), coin, item, received_at, numeric=isinstance(exc, NumericValidationError))

    def _handle_bbo(self, data: Any, received_at: datetime) -> None:
        if not isinstance(data, Mapping):
            raise ValidationError("bbo.data is not an object")
        coin = _coin(data.get("coin"))
        self.heartbeat.note_message("bbo", coin, received_at)
        record, crossed = parse_bbo(data, received_at)
        self._track_timestamp("bbo", coin, record["exchange_time_ms"], received_at)
        self._write("bbo", coin, received_at, record)
        if crossed:
            self._integrity(
                "crossed_bbo",
                coin,
                received_at,
                details={"exchange_time_ms": record["exchange_time_ms"], "bid_px": record["bid_px"], "ask_px": record["ask_px"]},
                quality_key="crossed_bbo_count",
            )

    def _handle_book(self, data: Any, received_at: datetime) -> None:
        if not isinstance(data, Mapping):
            raise ValidationError("l2Book.data is not an object")
        coin = _coin(data.get("coin"))
        self.heartbeat.note_message("l2Book", coin, received_at)
        try:
            sample = derive_book_sample(data, received_at)
        except Exception as exc:
            self._integrity(
                "invalid_book",
                coin,
                received_at,
                details={"error": str(exc), "payload_sha256": _payload_digest(data)},
                quality_key="invalid_book_count",
            )
            if isinstance(exc, NumericValidationError):
                self.heartbeat.note_quality("invalid_numeric_count")
            return
        self.latest_books[coin] = data
        self._track_timestamp("l2Book", coin, sample["exchange_time_ms"], received_at)
        fingerprint = _payload_digest(data)
        if self._last_book_fingerprint.get(coin) == fingerprint:
            return
        self._last_book_fingerprint[coin] = fingerprint
        receive_ms = sample["receive_time_ms"]
        last_sample = self._last_book_sample_ms.get(coin)
        if last_sample is not None and receive_ms - last_sample < self.book_sampling_seconds * 1000:
            return
        self._last_book_sample_ms[coin] = receive_ms
        self._write("book_5s", coin, received_at, sample)

    def _handle_asset_ctx(self, data: Any, received_at: datetime) -> None:
        if not isinstance(data, Mapping):
            raise ValidationError("activeAssetCtx.data is not an object")
        coin = _coin(data.get("coin"))
        self.heartbeat.note_message("activeAssetCtx", coin, received_at)
        record = parse_asset_ctx(data, received_at)
        if record["exchange_time_ms"] is not None:
            self._track_timestamp("activeAssetCtx", coin, record["exchange_time_ms"], received_at)
        self._write("asset_ctx", coin, received_at, record)

    def _handle_candle(self, data: Any, received_at: datetime) -> None:
        if not isinstance(data, Mapping):
            raise ValidationError("candle.data is not an object")
        coin = _coin(data.get("s"))
        self.heartbeat.note_message("candle", coin, received_at)
        record = parse_candle(data, received_at)
        self._track_timestamp("candle", coin, record["close_time_ms"], received_at)
        self._write("candles_5m", coin, received_at, record)

    def _malformed(
        self,
        error: str,
        coin: str | None,
        payload: Any,
        received_at: datetime,
        *,
        numeric: bool = False,
    ) -> None:
        self._integrity(
            "malformed_message",
            coin,
            received_at,
            details={"error": error, "payload_sha256": _payload_digest(payload)},
            quality_key="malformed_message_count",
        )
        if numeric:
            self.heartbeat.note_quality("invalid_numeric_count")


def _payload_digest(payload: Any) -> str:
    try:
        encoded = canonical_json(payload).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(payload).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def _best_effort_coin(value: Any) -> str | None:
    if isinstance(value, Mapping):
        coin = value.get("coin", value.get("s"))
        return coin if coin in SYMBOLS else None
    return None


def _last_complete_json_line(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            buffer = bytearray()
            while position > 0:
                position -= 1
                handle.seek(position)
                byte = handle.read(1)
                if byte == b"\n" and buffer:
                    line = bytes(reversed(buffer)).decode("utf-8")
                    try:
                        value = json.loads(line)
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        buffer.clear()
                        continue
                if byte != b"\n":
                    buffer.extend(byte)
    except FileNotFoundError:
        return None
    if buffer:
        try:
            value = json.loads(bytes(reversed(buffer)).decode("utf-8"))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


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


class InstanceLock:
    """Portable fail-closed lock using atomic creation and PID ownership."""

    def __init__(self, capture_root: Path, *, pid_checker: Callable[[int], bool] = _pid_is_active) -> None:
        self.capture_root = capture_root
        self.path = capture_root / "recorder.lock"
        self.pid_checker = pid_checker
        self.hostname = socket.gethostname()
        self.token = uuid.uuid4().hex
        self._owned = False

    def acquire(self) -> None:
        self.capture_root.mkdir(parents=True, exist_ok=True)
        owner = {
            "pid": os.getpid(),
            "hostname": self.hostname,
            "owner_token": self.token,
            "acquired_at_utc": utc_text(utc_now()),
        }
        for _ in range(4):
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing = self._read_owner()
                if existing is None:
                    raise ActiveRecorderError(f"lock exists but cannot be validated: {self.path}")
                if existing.get("hostname") != self.hostname:
                    raise ActiveRecorderError("capture root is locked by another host; refusing unsafe stale recovery")
                try:
                    existing_pid = int(existing.get("pid"))
                except (TypeError, ValueError) as exc:
                    raise ActiveRecorderError("capture lock has an invalid PID") from exc
                if self.pid_checker(existing_pid):
                    raise ActiveRecorderError(f"capture root is owned by active PID {existing_pid}")
                stale = self.path.with_name(f"recorder.lock.stale.{existing_pid}.{uuid.uuid4().hex}")
                try:
                    os.replace(self.path, stale)
                except FileNotFoundError:
                    continue
                with contextlib.suppress(FileNotFoundError):
                    stale.unlink()
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(canonical_json(owner))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._owned = True
                return
        raise ActiveRecorderError("could not acquire recorder lock after stale-lock races")

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def release(self) -> None:
        if not self._owned:
            return
        existing = self._read_owner()
        if existing is not None and existing.get("owner_token") == self.token:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        self._owned = False

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def repository_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def initialize_manifest(root: Path, *, started_at: datetime | None = None) -> dict[str, Any]:
    path = root / "manifest.json"
    immutable = {
        "schema_version": SCHEMA_VERSION,
        "capture_contract_version": CAPTURE_CONTRACT_VERSION,
        "network": NETWORK,
        "symbols": list(SYMBOLS),
        "subscriptions": SUBSCRIPTIONS,
        "book_sampling_seconds": BOOK_SAMPLING_SECONDS,
        "data_is_public_market_data": True,
        "wallet_required": False,
        "private_key_required": False,
        "trading_enabled": False,
    }
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecorderError(f"existing manifest is unreadable: {path}") from exc
        if not isinstance(manifest, dict) or not manifest.get("capture_started_at_utc"):
            raise RecorderError("existing manifest has no capture start timestamp")
        for key, expected in immutable.items():
            if manifest.get(key) != expected:
                raise RecorderError(f"existing manifest conflicts on {key}")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **immutable,
        "capture_started_at_utc": utc_text(started_at or utc_now()),
        "repository_commit": repository_commit(),
        "python_version": platform.python_version(),
        "output_root": str(root.resolve()),
    }
    atomic_write_json(path, manifest)
    return manifest


class MarketRecorder:
    def __init__(
        self,
        output_root: Path,
        *,
        heartbeat_seconds: float = 5.0,
        max_backoff_seconds: float = 30.0,
        session_factory: Callable[..., Any] = aiohttp.ClientSession,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.output_root = output_root
        self.heartbeat_seconds = heartbeat_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.session_factory = session_factory
        self.sleep = sleep
        self.stop_event = asyncio.Event()
        self.current_websocket: Any = None
        self.heartbeat: HeartbeatState | None = None
        self.writer: HourlyJsonlWriter | None = None
        self.processor: MessageProcessor | None = None

    async def request_stop(self) -> None:
        self.stop_event.set()
        if self.current_websocket is not None and not self.current_websocket.closed:
            await self.current_websocket.close()

    async def _stop_after(self, duration_seconds: float) -> None:
        await self.sleep(duration_seconds)
        await self.request_stop()

    async def _heartbeat_loop(self) -> None:
        assert self.heartbeat is not None and self.writer is not None
        while not self.stop_event.is_set():
            atomic_write_json(self.output_root / "heartbeat.json", self.heartbeat.value)
            self.writer.flush(durable=True)
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.heartbeat_seconds)
            except TimeoutError:
                pass

    async def _application_ping(self, websocket: Any) -> None:
        while not self.stop_event.is_set() and not websocket.closed:
            await self.sleep(30.0)
            if not self.stop_event.is_set() and not websocket.closed:
                await websocket.send_json({"method": "ping"})

    async def _connect_once(self, session: Any) -> None:
        assert self.processor is not None
        async with session.ws_connect(
            WS_URL,
            heartbeat=30.0,
            autoping=True,
            receive_timeout=75.0,
            max_msg_size=16 * 1024 * 1024,
        ) as websocket:
            self.current_websocket = websocket
            for subscription in SUBSCRIPTIONS:
                await websocket.send_json({"method": "subscribe", "subscription": subscription})
            ping_task = asyncio.create_task(self._application_ping(websocket))
            try:
                async for websocket_message in websocket:
                    received_at = utc_now()
                    if websocket_message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            decoded = json.loads(websocket_message.data)
                        except json.JSONDecodeError:
                            self.processor.handle_message(websocket_message.data, received_at)
                            continue
                        self.processor.handle_message(decoded, received_at)
                    elif websocket_message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                        break
                    elif websocket_message.type == aiohttp.WSMsgType.ERROR:
                        error = websocket.exception()
                        raise RecorderError(f"WebSocket receive failed: {error}")
                    if self.stop_event.is_set():
                        break
            finally:
                ping_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ping_task
                self.current_websocket = None

    async def run(self, *, duration_seconds: float | None = None) -> dict[str, Any]:
        process_started = utc_now()
        instance_lock = InstanceLock(self.output_root)
        instance_lock.acquire()
        duration_task: asyncio.Task[Any] | None = None
        heartbeat_task: asyncio.Task[Any] | None = None
        try:
            initialize_manifest(self.output_root, started_at=process_started)
            self.writer = HourlyJsonlWriter(self.output_root)
            self.heartbeat = HeartbeatState(process_started)
            deduplicator = TradeDeduplicator()
            deduplicator.seed_from_capture(self.output_root)
            self.processor = MessageProcessor(self.writer, self.heartbeat, deduplicator)
            self.processor.seed_book_sampling_from_capture(self.output_root)
            atomic_write_json(self.output_root / "heartbeat.json", self.heartbeat.value)
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            if duration_seconds is not None:
                if duration_seconds <= 0:
                    raise ValueError("duration must be positive")
                duration_task = asyncio.create_task(self._stop_after(duration_seconds))

            attempt = 0
            async with self.session_factory(headers={"User-Agent": "passive-hyperliquid-market-recorder/1"}) as session:
                while not self.stop_event.is_set():
                    if attempt > 0:
                        self.heartbeat.note_reconnect()
                    try:
                        await self._connect_once(session)
                        attempt = 0
                        if not self.stop_event.is_set():
                            attempt = 1
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if self.stop_event.is_set():
                            break
                        attempt += 1
                        LOGGER.warning("public WebSocket disconnected: %s", exc)
                    if self.stop_event.is_set():
                        break
                    backoff = min(self.max_backoff_seconds, float(2 ** max(0, attempt - 1)))
                    await self.sleep(backoff)
        finally:
            self.stop_event.set()
            if duration_task is not None:
                duration_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await duration_task
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            if self.writer is not None:
                self.writer.close()
            if self.heartbeat is not None:
                atomic_write_json(self.output_root / "heartbeat.json", self.heartbeat.value)
            instance_lock.release()
        assert self.heartbeat is not None
        return self.heartbeat.value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record passive Hyperliquid mainnet public market data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, arguments.log_level), format="%(asctime)s %(levelname)s %(message)s")
    recorder = MarketRecorder(
        arguments.output_root,
        heartbeat_seconds=arguments.heartbeat_seconds,
        max_backoff_seconds=arguments.max_backoff_seconds,
    )
    try:
        heartbeat = asyncio.run(recorder.run(duration_seconds=arguments.duration_seconds))
    except KeyboardInterrupt:
        LOGGER.info("capture stopped by operator")
        return 130
    except ActiveRecorderError as exc:
        LOGGER.error("%s", exc)
        return 2
    print(canonical_json(heartbeat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
