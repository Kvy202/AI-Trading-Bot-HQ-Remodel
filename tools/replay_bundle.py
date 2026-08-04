"""Deterministic replay-bundle capture and historical source resolution.

All operations are copy/filter operations.  Source evidence is never moved,
deleted, or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from tools.evidence_manifest import build_evidence_manifest, evidence_manifest_digest
    from tools.replay_contract import (
        DEFAULT_OVERRIDES_PATH as _UNUSED_CONTRACT_OVERRIDES,
        IDENTITY_RE,
        ReplayContractError,
        load_replay_contract,
        replay_contract_digest,
    )
    from tools.model_serving_snapshot import (
        ModelServingSnapshotError,
        load_model_serving_snapshot,
    )
except ModuleNotFoundError:
    from evidence_manifest import build_evidence_manifest, evidence_manifest_digest  # type: ignore
    from replay_contract import (  # type: ignore
        DEFAULT_OVERRIDES_PATH as _UNUSED_CONTRACT_OVERRIDES,
        IDENTITY_RE,
        ReplayContractError,
        load_replay_contract,
        replay_contract_digest,
    )
    from model_serving_snapshot import (  # type: ignore
        ModelServingSnapshotError,
        load_model_serving_snapshot,
    )

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_LOGS_DIR = BASE_DIR / "logs"
DEFAULT_BUNDLE_ROOT = DEFAULT_REPORTS_DIR / "replay_bundles"
SCHEMA_VERSION = 1

CSV_NAMES = {
    "signals": "live_signals.csv",
    "xgboost": "xgboost_signal_shadow.csv",
    "paper": "trades_paper.csv",
    "closed": "trades_closed.csv",
}
MODEL_OUTPUT_NAMES = ("live_models_by_symbol.csv", "live_meta_log.csv")
TIMESTAMP_COLUMNS = {
    "signals": ("ts", "timestamp"),
    "xgboost": ("timestamp", "ts"),
    "paper": ("ts", "timestamp"),
    "closed": ("ts", "timestamp"),
}
PATH_ONLY_COLUMNS = {"artifact_path", "model_path", "scaler_path"}
SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>api[_-]?(?:key|secret)|private[_-]?key|"
    r"wallet[_-]?(?:address|key)|mnemonic|seed[_-]?phrase|password|token)\b"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
PRIVATE_HEX_RE = re.compile(r"(?i)(?<![0-9a-f])0x[0-9a-f]{64}(?![0-9a-f])")
WALLET_RE = re.compile(r"(?i)(?<![0-9a-f])0x[0-9a-f]{40}(?![0-9a-f])")


class ReplayBundleError(ValueError):
    """Raised when source evidence is contradictory or a bundle is invalid."""


def parse_timestamp(value: Any) -> Optional[datetime]:
    text = str(value or "").strip().replace("+0000", "+00:00")
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def canonical_timestamp(value: Any) -> Optional[str]:
    parsed = parse_timestamp(value)
    return None if parsed is None else parsed.isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_source_text_file(path: Path) -> str:
    """Hash text evidence with platform line endings and a UTF-8 BOM removed."""

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _sha256_bytes(normalized.encode("utf-8"))


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _normalized_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(key): str(value or "").strip()
        for key, value in row.items()
        if key is not None and str(key).lower() not in PATH_ONLY_COLUMNS
    }


def canonical_row_digest(row: Mapping[str, Any]) -> str:
    return _json_digest(_normalized_row(row))


def _timestamp_value(kind: str, row: Mapping[str, Any]) -> str:
    for column in TIMESTAMP_COLUMNS[kind]:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def _logical_key(kind: str, row: Mapping[str, str], row_digest: str) -> tuple[str, ...]:
    signal_id = str(row.get("signal_id", "") or "").strip()
    if kind == "signals":
        if signal_id:
            return ("signal_id", signal_id)
        return (
            "fallback_inventory",
            _timestamp_value(kind, row),
            row.get("symbol", ""),
            row.get("p_meta", ""),
            row.get("px", row.get("price", "")),
        )
    if kind == "xgboost":
        return ("signal_id", signal_id) if signal_id else ("missing_id", row_digest)
    if kind == "paper":
        return (
            _timestamp_value(kind, row),
            row.get("symbol", ""),
            row.get("side", row.get("action", "")),
            signal_id,
            row.get("reason", ""),
        )
    return (
        _timestamp_value(kind, row),
        row.get("symbol", ""),
        row.get("closed_side", row.get("side", "")),
        signal_id or row.get("entry_signal_id", ""),
    )


def _relative_source_name(path: Path, logs_dir: Path) -> str:
    try:
        return path.resolve().relative_to(logs_dir.resolve()).as_posix()
    except Exception:
        return path.name


def _discover_source_paths(logs_dir: Path) -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {
        "signals": [],
        "xgboost": [],
        "paper": [],
        "closed": [],
    }
    live = logs_dir / "live_signals.csv"
    if live.is_file():
        sources["signals"].append(live)
    sources["xgboost"] = sorted(
        {path for path in logs_dir.rglob("xgboost_signal_shadow.csv") if path.is_file()},
        key=lambda path: _relative_source_name(path, logs_dir),
    )
    sources["paper"] = sorted(
        {path for path in logs_dir.rglob("trades_paper_*.csv") if path.is_file()},
        key=lambda path: _relative_source_name(path, logs_dir),
    )
    closed = {
        path
        for pattern in ("trades_closed.csv", "trades_closed_*.csv")
        for path in logs_dir.rglob(pattern)
        if path.is_file()
    }
    sources["closed"] = sorted(closed, key=lambda path: _relative_source_name(path, logs_dir))
    return sources


def _collect_filtered_rows(
    kind: str,
    paths: Iterable[Path],
    start: datetime,
    finish: datetime,
    logs_dir: Path,
) -> dict[str, Any]:
    retained: list[tuple[dict[str, str], int]] = []
    seen_exact: set[str] = set()
    logical: dict[tuple[str, ...], str] = {}
    duplicate_rows = 0
    conflicts: list[dict[str, Any]] = []
    malformed_timestamps = 0
    source_digests: dict[str, str] = {}
    duplicate_signal_ids: Counter[str] = Counter()
    source_order = 0

    for path in paths:
        source_digests[_relative_source_name(path, logs_dir)] = _sha256_source_text_file(path)
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ReplayBundleError(f"CSV header missing: {path}")
                for raw in reader:
                    current_order = source_order
                    source_order += 1
                    row = _normalized_row(raw)
                    timestamp = parse_timestamp(_timestamp_value(kind, row))
                    if timestamp is None:
                        malformed_timestamps += 1
                        continue
                    if timestamp < start or timestamp > finish:
                        continue
                    digest = canonical_row_digest(row)
                    key = _logical_key(kind, row, digest)
                    if digest in seen_exact:
                        duplicate_rows += 1
                        signal_id = row.get("signal_id", "")
                        if signal_id:
                            duplicate_signal_ids[signal_id] += 1
                        continue
                    seen_exact.add(digest)
                    previous = logical.get(key)
                    if previous is not None and previous != digest:
                        conflicts.append(
                            {
                                "kind": kind,
                                "logical_key": list(key),
                                "first_digest": previous,
                                "conflicting_digest": digest,
                            }
                        )
                        continue
                    logical[key] = digest
                    retained.append((row, current_order))
        except ReplayBundleError:
            raise
        except Exception as exc:
            raise ReplayBundleError(f"unable to read source CSV {path}: {exc}") from exc

    retained.sort(
        key=lambda item: (
            parse_timestamp(_timestamp_value(kind, item[0]))
            or datetime.min.replace(tzinfo=timezone.utc),
            item[1],
        )
    )
    return {
        "rows": [row for row, _source_order in retained],
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_signal_ids": sorted(duplicate_signal_ids),
        "conflicting_rows": conflicts,
        "malformed_timestamps": malformed_timestamps,
        "source_file_digests": source_digests,
    }


def collect_source_rows(
    logs_dir: Path | str,
    run_started_utc: str,
    finished_at: str,
) -> dict[str, Any]:
    logs = Path(logs_dir)
    start = parse_timestamp(run_started_utc)
    finish = parse_timestamp(finished_at)
    if start is None or finish is None or finish < start:
        raise ReplayBundleError("invalid source filtering window")
    paths = _discover_source_paths(logs)
    kinds = {
        kind: _collect_filtered_rows(kind, kind_paths, start, finish, logs)
        for kind, kind_paths in paths.items()
    }
    conflicts = [
        conflict
        for item in kinds.values()
        for conflict in item["conflicting_rows"]
    ]
    source_digests = {
        f"{kind}/{path}": digest
        for kind, item in kinds.items()
        for path, digest in item["source_file_digests"].items()
    }
    normalized_for_digest = {
        kind: [canonical_row_digest(row) for row in item["rows"]]
        for kind, item in kinds.items()
    }
    return {
        "kinds": kinds,
        "source_paths": {
            kind: [str(path) for path in values] for kind, values in paths.items()
        },
        "source_file_digests": dict(sorted(source_digests.items())),
        "source_digest": _json_digest(normalized_for_digest),
        "duplicate_rows_removed": sum(
            item["duplicate_rows_removed"] for item in kinds.values()
        ),
        "conflicting_rows": conflicts,
        "malformed_timestamps": sum(
            item["malformed_timestamps"] for item in kinds.values()
        ),
    }


def _collect_model_output_file(path: Path, start: datetime, finish: datetime) -> dict[str, Any]:
    """Filter one model-output CSV and enforce timestamp+symbol consistency."""

    rows: list[tuple[dict[str, str], int]] = []
    seen_exact: set[str] = set()
    logical: dict[tuple[str, str], str] = {}
    duplicate_count = 0
    conflicts: list[dict[str, Any]] = []
    malformed = 0
    columns: list[str] = []
    if not path.is_file():
        return {"rows": [], "duplicate_count": 0, "conflicts": [], "malformed": 0,
                "columns": [], "source_digest": None}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns:
            raise ReplayBundleError(f"CSV header missing: {path}")
        for order, raw in enumerate(reader):
            row = _normalized_row(raw)
            timestamp_text = str(row.get("ts", row.get("timestamp", "")) or "").strip()
            timestamp = parse_timestamp(timestamp_text)
            if timestamp is None:
                malformed += 1
                continue
            if not (start <= timestamp <= finish):
                continue
            digest = canonical_row_digest(row)
            if digest in seen_exact:
                duplicate_count += 1
                continue
            seen_exact.add(digest)
            symbol = str(row.get("symbol", "__pooled__") or "__pooled__").strip()
            key = (canonical_timestamp(timestamp_text) or timestamp_text, symbol)
            previous = logical.get(key)
            if previous is not None and previous != digest:
                conflicts.append({"file": path.name, "timestamp": key[0], "symbol": symbol,
                                  "first_digest": previous, "conflicting_digest": digest})
                continue
            logical[key] = digest
            rows.append((row, order))
    rows.sort(key=lambda item: (
        parse_timestamp(item[0].get("ts", item[0].get("timestamp", "")))
        or datetime.min.replace(tzinfo=timezone.utc), item[1]))
    return {
        "rows": [row for row, _ in rows],
        "duplicate_count": duplicate_count,
        "conflicts": conflicts,
        "malformed": malformed,
        "columns": columns,
        "source_digest": _sha256_source_text_file(path),
    }


def collect_model_output_rows(
    logs_dir: Path | str, run_started_utc: str, finished_at: str
) -> dict[str, Any]:
    logs = Path(logs_dir)
    start = parse_timestamp(run_started_utc)
    finish = parse_timestamp(finished_at)
    if start is None or finish is None or finish < start:
        raise ReplayBundleError("invalid model-output filtering window")
    files = {
        name: _collect_model_output_file(logs / name, start, finish)
        for name in MODEL_OUTPUT_NAMES
    }
    rows = [row for item in files.values() for row in item["rows"]]
    timestamps = sorted(
        value for row in rows
        if (value := parse_timestamp(row.get("ts", row.get("timestamp", "")))) is not None
    )
    rows_by_symbol = Counter(
        str(row.get("symbol", "__pooled__") or "__pooled__") for row in rows
    )
    columns = sorted({column for item in files.values() for column in item["columns"]})
    normalized = {
        name: [canonical_row_digest(row) for row in item["rows"]]
        for name, item in files.items()
    }
    return {
        "files": files,
        "model_output_row_count": len(rows),
        "model_output_rows_by_symbol": dict(sorted(rows_by_symbol.items())),
        "model_output_first_timestamp": None if not timestamps else timestamps[0].isoformat().replace("+00:00", "Z"),
        "model_output_last_timestamp": None if not timestamps else timestamps[-1].isoformat().replace("+00:00", "Z"),
        "model_output_columns": columns,
        "model_output_digest": _json_digest(normalized),
        "model_output_duplicate_count": sum(item["duplicate_count"] for item in files.values()),
        "model_output_conflict_count": sum(len(item["conflicts"]) for item in files.values()),
        "conflicts": [conflict for item in files.values() for conflict in item["conflicts"]],
        "source_file_digests": {
            f"model_output/{name}": item["source_digest"]
            for name, item in files.items() if item["source_digest"] is not None
        },
    }


def calculate_coverage(
    rows_by_kind: Mapping[str, Sequence[Mapping[str, Any]]],
    run_started_utc: str,
    finished_at: str,
    *,
    max_start_delay_seconds: float = 120.0,
    max_end_delay_seconds: float = 120.0,
    max_signal_gap_seconds: float = 180.0,
    malformed_timestamps: int = 0,
    duplicate_signal_ids: Optional[Sequence[str]] = None,
    conflicting_signal_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    start = parse_timestamp(run_started_utc)
    finish = parse_timestamp(finished_at)
    if start is None or finish is None:
        raise ReplayBundleError("invalid coverage window")
    signals = list(rows_by_kind.get("signals", []))
    xgboost = list(rows_by_kind.get("xgboost", []))
    paper = list(rows_by_kind.get("paper", []))
    closed = list(rows_by_kind.get("closed", []))
    signal_times = sorted(
        parsed
        for row in signals
        if (parsed := parse_timestamp(_timestamp_value("signals", row))) is not None
    )
    gaps = [
        (later - earlier).total_seconds()
        for earlier, later in zip(signal_times, signal_times[1:])
    ]
    signal_ids = [str(row.get("signal_id", "") or "").strip() for row in signals]
    signal_by_id = {
        str(row.get("signal_id", "") or "").strip(): row
        for row in signals
        if str(row.get("signal_id", "") or "").strip()
    }
    xgb_ids = [str(row.get("signal_id", "") or "").strip() for row in xgboost]
    xgb_with_ids = [value for value in xgb_ids if value]
    join_mismatches: list[dict[str, str]] = []
    joins = 0
    for row, signal_id in zip(xgboost, xgb_ids):
        if not signal_id:
            continue
        signal = signal_by_id.get(signal_id)
        if signal is None:
            join_mismatches.append({"signal_id": signal_id, "reason": "signal_id_not_found"})
            continue
        signal_symbol = str(signal.get("symbol", "") or "").strip()
        decision_symbol = str(row.get("symbol", "") or "").strip()
        signal_time = parse_timestamp(_timestamp_value("signals", signal))
        decision_time = parse_timestamp(_timestamp_value("xgboost", row))
        if decision_symbol and signal_symbol != decision_symbol:
            join_mismatches.append({"signal_id": signal_id, "reason": "symbol_mismatch"})
            continue
        if signal_time is None or decision_time is None or signal_time != decision_time:
            join_mismatches.append({"signal_id": signal_id, "reason": "timestamp_mismatch"})
            continue
        joins += 1
    first = signal_times[0] if signal_times else None
    last = signal_times[-1] if signal_times else None
    duplicate_ids = sorted(set(duplicate_signal_ids or []))
    conflict_ids = sorted(set(conflicting_signal_ids or []))
    checks = {
        "first_signal_within_start_gate": bool(
            first is not None
            and start <= first <= finish
            and (first - start).total_seconds() <= max_start_delay_seconds
        ),
        "last_signal_within_end_gate": bool(
            last is not None
            and start <= last <= finish
            and (finish - last).total_seconds() <= max_end_delay_seconds
        ),
        "maximum_signal_gap_within_gate": bool(
            signal_times and (max(gaps, default=0.0) <= max_signal_gap_seconds)
        ),
        "no_conflicting_signal_ids": not conflict_ids,
        "no_malformed_timestamps": malformed_timestamps == 0,
        "all_replayed_xgboost_decisions_joined": (
            joins == len(xgb_with_ids) and not join_mismatches
        ),
    }
    return {
        "signal_row_count": len(signals),
        "signal_rows_with_id": sum(bool(value) for value in signal_ids),
        "xgboost_row_count": len(xgboost),
        "xgboost_rows_with_id": len(xgb_with_ids),
        "xgboost_rows_missing_id": len(xgboost) - len(xgb_with_ids),
        "paper_row_count": len(paper),
        "closed_row_count": len(closed),
        "first_signal_timestamp": None if first is None else first.isoformat().replace("+00:00", "Z"),
        "last_signal_timestamp": None if last is None else last.isoformat().replace("+00:00", "Z"),
        "maximum_signal_gap_seconds": None if not signal_times else max(gaps, default=0.0),
        "xgboost_signal_join_count": joins,
        "xgboost_signal_join_rate": None if not xgb_with_ids else joins / len(xgb_with_ids),
        "xgboost_join_mismatches": join_mismatches,
        "duplicate_signal_ids": duplicate_ids,
        "conflicting_signal_ids": conflict_ids,
        "malformed_timestamp_count": malformed_timestamps,
        "gates": {
            "max_start_delay_seconds": max_start_delay_seconds,
            "max_end_delay_seconds": max_end_delay_seconds,
            "max_signal_gap_seconds": max_signal_gap_seconds,
        },
        "coverage_checks": checks,
        "coverage_passed": all(checks.values()),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return _sha256_bytes(b"")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return _sha256_file(path)


def _sanitize_log_text(text: str) -> str:
    value = PRIVATE_HEX_RE.sub("[redacted_private_key]", text)
    value = WALLET_RE.sub("[redacted_wallet_address]", value)
    return SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}=[redacted]", value
    )


def _copy_relevant_logs(logs_dir: Path, bundle_dir: Path, mode: str) -> dict[str, str]:
    candidates = {
        "executor_stdout.log": [
            logs_dir / f"matrix_{mode}_executor.out",
            logs_dir / f"{mode}_paper_executor.out",
            logs_dir / "live_executor.out",
        ],
        "executor_stderr.log": [
            logs_dir / f"matrix_{mode}_executor.err",
            logs_dir / f"{mode}_paper_executor.err",
            logs_dir / "live_executor.err",
        ],
        "writer_stdout.log": [
            logs_dir / f"matrix_{mode}_writer.out",
            logs_dir / f"{mode}_paper_writer.out",
            logs_dir / "live_writer.out",
        ],
        "writer_stderr.log": [
            logs_dir / f"matrix_{mode}_writer.err",
            logs_dir / f"{mode}_paper_writer.err",
            logs_dir / "live_writer.err",
        ],
    }
    digests: dict[str, str] = {}
    for output_name, options in candidates.items():
        source = next((path for path in options if path.is_file()), None)
        if source is None:
            continue
        text = _sanitize_log_text(source.read_text(encoding="utf-8", errors="replace"))
        data = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        target = bundle_dir / output_name
        target.write_bytes(data)
        digests[output_name] = _sha256_bytes(data)
    return digests


def bundle_digest(manifest: Mapping[str, Any]) -> str:
    relevant = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "identity",
            "run_started_utc",
            "finished_at",
            "manifest_digest",
            "replay_contract_digest",
            "source_file_digests",
            "filtered_file_digests",
            "row_counts",
            "first_timestamp",
            "last_timestamp",
            "duplicate_rows_removed",
            "conflicting_rows",
            "coverage_checks",
            "model_serving_snapshot_digest",
            "model_output_row_count",
            "model_output_rows_by_symbol",
            "model_output_first_timestamp",
            "model_output_last_timestamp",
            "model_output_columns",
            "model_output_digest",
            "model_output_duplicate_count",
            "model_output_conflict_count",
        )
        if key in manifest
    }
    return _json_digest(relevant)


def build_replay_bundle(
    identity: str,
    run_started_utc: str,
    finished_at: str,
    contract_path: Path | str,
    *,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    bundle_root: Path | str = DEFAULT_BUNDLE_ROOT,
    model_serving_snapshot_path: Path | str | None = None,
    manifest_digest_value: Optional[str] = None,
    max_start_delay_seconds: float = 120.0,
    max_end_delay_seconds: float = 120.0,
    max_signal_gap_seconds: float = 180.0,
) -> dict[str, Any]:
    match = IDENTITY_RE.fullmatch(identity)
    if match is None:
        raise ReplayBundleError("identity must follow mode:14-digit-timestamp")
    mode, timestamp = match.group("mode"), match.group("timestamp")
    contract = load_replay_contract(contract_path)
    if contract["identity"] != identity:
        raise ReplayBundleError("replay contract identity does not match bundle identity")
    reports = Path(reports_dir)
    logs = Path(logs_dir)
    root = Path(bundle_root)
    target = root / f"{mode}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ReplayBundleError(f"replay bundle target already exists: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".{mode}_{timestamp}_", dir=root))

    try:
        sources = collect_source_rows(logs, run_started_utc, finished_at)
        model_outputs = collect_model_output_rows(logs, run_started_utc, finished_at)
        if sources["conflicting_rows"]:
            raise ReplayBundleError("conflicting logical source rows prevent bundle capture")
        if model_outputs["conflicts"]:
            raise ReplayBundleError("conflicting timestamp/symbol model output rows prevent bundle capture")
        rows_by_kind = {
            kind: item["rows"] for kind, item in sources["kinds"].items()
        }
        duplicate_ids = sources["kinds"]["signals"]["duplicate_signal_ids"]
        coverage = calculate_coverage(
            rows_by_kind,
            run_started_utc,
            finished_at,
            max_start_delay_seconds=max_start_delay_seconds,
            max_end_delay_seconds=max_end_delay_seconds,
            max_signal_gap_seconds=max_signal_gap_seconds,
            malformed_timestamps=sources["malformed_timestamps"],
            duplicate_signal_ids=duplicate_ids,
            conflicting_signal_ids=[],
        )

        filtered_digests: dict[str, str] = {}
        for kind, output_name in CSV_NAMES.items():
            rows = rows_by_kind[kind]
            if rows:
                filtered_digests[output_name] = _write_rows(stage / output_name, rows)
        for output_name, item in model_outputs["files"].items():
            if item["rows"]:
                filtered_digests[output_name] = _write_rows(stage / output_name, item["rows"])
        contract_copy = dict(contract)
        contract_copy_path = stage / "replay_contract.json"
        contract_copy_path.write_text(json.dumps(contract_copy, indent=2), encoding="utf-8")
        filtered_digests["replay_contract.json"] = _sha256_file(contract_copy_path)
        snapshot_digest_value: Optional[str] = None
        if model_serving_snapshot_path is not None:
            snapshot = load_model_serving_snapshot(model_serving_snapshot_path)
            if snapshot.get("identity") != identity:
                raise ReplayBundleError("model-serving snapshot identity does not match bundle identity")
            snapshot_copy = stage / "model_serving_snapshot.json"
            snapshot_copy.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            filtered_digests["model_serving_snapshot.json"] = _sha256_file(snapshot_copy)
            snapshot_digest_value = str(snapshot["snapshot_digest"])
        filtered_digests.update(_copy_relevant_logs(logs, stage, mode))

        if manifest_digest_value is None:
            evidence = build_evidence_manifest(reports)
            manifest_digest_value = evidence_manifest_digest(evidence)
        first_timestamp: dict[str, Optional[str]] = {}
        last_timestamp: dict[str, Optional[str]] = {}
        for kind, rows in rows_by_kind.items():
            timestamps = [parse_timestamp(_timestamp_value(kind, row)) for row in rows]
            parsed = sorted(value for value in timestamps if value is not None)
            first_timestamp[kind] = (
                None if not parsed else parsed[0].isoformat().replace("+00:00", "Z")
            )
            last_timestamp[kind] = (
                None if not parsed else parsed[-1].isoformat().replace("+00:00", "Z")
            )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "run_started_utc": canonical_timestamp(run_started_utc),
            "finished_at": canonical_timestamp(finished_at),
            "manifest_digest": manifest_digest_value,
            "replay_contract_digest": replay_contract_digest(contract),
            "source_file_digests": dict(sorted({
                **sources["source_file_digests"],
                **model_outputs["source_file_digests"],
            }.items())),
            "filtered_file_digests": dict(sorted(filtered_digests.items())),
            "row_counts": {kind: len(rows) for kind, rows in rows_by_kind.items()},
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "duplicate_rows_removed": sources["duplicate_rows_removed"],
            "conflicting_rows": [],
            "coverage_checks": coverage,
            "model_serving_snapshot_digest": snapshot_digest_value,
            "model_output_row_count": model_outputs["model_output_row_count"],
            "model_output_rows_by_symbol": model_outputs["model_output_rows_by_symbol"],
            "model_output_first_timestamp": model_outputs["model_output_first_timestamp"],
            "model_output_last_timestamp": model_outputs["model_output_last_timestamp"],
            "model_output_columns": model_outputs["model_output_columns"],
            "model_output_digest": model_outputs["model_output_digest"],
            "model_output_duplicate_count": model_outputs["model_output_duplicate_count"],
            "model_output_conflict_count": model_outputs["model_output_conflict_count"],
        }
        manifest["bundle_digest"] = bundle_digest(manifest)
        (stage / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "status": "exact_bundle",
        "bundle_path": str(target),
        "bundle_digest": manifest["bundle_digest"],
        "manifest": manifest,
        "coverage": coverage,
    }


def _read_bundle_rows(bundle_dir: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for kind, name in CSV_NAMES.items():
        path = bundle_dir / name
        if not path.exists() or path.stat().st_size == 0:
            result[kind] = []
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            result[kind] = [_normalized_row(row) for row in csv.DictReader(handle)]
    return result


def _read_bundle_model_rows(bundle_dir: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for name in MODEL_OUTPUT_NAMES:
        path = bundle_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            result[name] = []
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            result[name] = [_normalized_row(row) for row in csv.DictReader(handle)]
    return result


def validate_replay_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    root = Path(bundle_dir)
    manifest_path = root / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReplayBundleError(f"malformed bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReplayBundleError("bundle manifest schema_version must be 1")
    required_fields = {
        "schema_version",
        "identity",
        "run_started_utc",
        "finished_at",
        "manifest_digest",
        "replay_contract_digest",
        "source_file_digests",
        "filtered_file_digests",
        "row_counts",
        "first_timestamp",
        "last_timestamp",
        "duplicate_rows_removed",
        "conflicting_rows",
        "coverage_checks",
        "bundle_digest",
    }
    optional_fields = {
        "model_serving_snapshot_digest",
        "model_output_row_count",
        "model_output_rows_by_symbol",
        "model_output_first_timestamp",
        "model_output_last_timestamp",
        "model_output_columns",
        "model_output_digest",
        "model_output_duplicate_count",
        "model_output_conflict_count",
    }
    missing = required_fields - set(manifest)
    unknown = set(manifest) - required_fields - optional_fields
    if missing or unknown:
        raise ReplayBundleError(
            f"bundle manifest fields invalid; missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    identity_match = IDENTITY_RE.fullmatch(str(manifest.get("identity") or ""))
    if identity_match is None:
        raise ReplayBundleError("bundle manifest identity is invalid")
    start = parse_timestamp(manifest.get("run_started_utc"))
    finish = parse_timestamp(manifest.get("finished_at"))
    if start is None or finish is None or finish < start:
        raise ReplayBundleError("bundle manifest filtering window is invalid")
    filtered = manifest.get("filtered_file_digests")
    if not isinstance(filtered, dict) or "replay_contract.json" not in filtered:
        raise ReplayBundleError("bundle filtered_file_digests are invalid")
    allowed_files = set(CSV_NAMES.values()) | {
        "replay_contract.json",
        "model_serving_snapshot.json",
        *MODEL_OUTPUT_NAMES,
        "executor_stdout.log",
        "executor_stderr.log",
        "writer_stdout.log",
        "writer_stderr.log",
    }
    for name, expected in filtered.items():
        if name not in allowed_files or Path(name).name != name:
            raise ReplayBundleError(f"unexpected or unsafe bundle file name: {name}")
        if re.fullmatch(r"[0-9a-f]{64}", str(expected or "")) is None:
            raise ReplayBundleError(f"invalid bundle file digest: {name}")
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ReplayBundleError(f"bundle file digest mismatch: {name}")
    expected_bundle = bundle_digest(manifest)
    if manifest.get("bundle_digest") != expected_bundle:
        raise ReplayBundleError("bundle digest mismatch")
    contract = load_replay_contract(root / "replay_contract.json")
    if replay_contract_digest(contract) != manifest.get("replay_contract_digest"):
        raise ReplayBundleError("bundle replay contract digest mismatch")
    if contract.get("identity") != manifest.get("identity"):
        raise ReplayBundleError("bundle replay contract identity mismatch")
    if canonical_timestamp(contract.get("run_started_utc")) != canonical_timestamp(
        manifest.get("run_started_utc")
    ):
        raise ReplayBundleError("bundle replay contract start time mismatch")
    snapshot_path = root / "model_serving_snapshot.json"
    if snapshot_path.is_file():
        snapshot = load_model_serving_snapshot(snapshot_path)
        if snapshot.get("identity") != manifest.get("identity"):
            raise ReplayBundleError("bundle model-serving snapshot identity mismatch")
        if snapshot.get("snapshot_digest") != manifest.get("model_serving_snapshot_digest"):
            raise ReplayBundleError("bundle model-serving snapshot digest mismatch")
    rows = _read_bundle_rows(root)
    row_counts = manifest.get("row_counts")
    first_timestamps = manifest.get("first_timestamp")
    last_timestamps = manifest.get("last_timestamp")
    if not all(isinstance(value, dict) for value in (row_counts, first_timestamps, last_timestamps)):
        raise ReplayBundleError("bundle row inventory is invalid")
    for kind, values in rows.items():
        if row_counts.get(kind) != len(values):
            raise ReplayBundleError(f"bundle row count mismatch: {kind}")
        seen_exact: set[str] = set()
        logical: dict[tuple[str, ...], str] = {}
        timestamps: list[datetime] = []
        for row in values:
            timestamp = parse_timestamp(_timestamp_value(kind, row))
            if timestamp is None or not (start <= timestamp <= finish):
                raise ReplayBundleError(f"bundle row outside filtering window: {kind}")
            timestamps.append(timestamp)
            digest = canonical_row_digest(row)
            if digest in seen_exact:
                raise ReplayBundleError(f"exact duplicate row retained in bundle: {kind}")
            seen_exact.add(digest)
            key = _logical_key(kind, row, digest)
            previous = logical.get(key)
            if previous is not None and previous != digest:
                raise ReplayBundleError(f"conflicting logical rows retained in bundle: {kind}")
            logical[key] = digest
        ordered = sorted(timestamps)
        actual_first = None if not ordered else ordered[0].isoformat().replace("+00:00", "Z")
        actual_last = None if not ordered else ordered[-1].isoformat().replace("+00:00", "Z")
        if first_timestamps.get(kind) != actual_first or last_timestamps.get(kind) != actual_last:
            raise ReplayBundleError(f"bundle timestamp inventory mismatch: {kind}")
    model_rows = _read_bundle_model_rows(root)
    if any(name in filtered for name in MODEL_OUTPUT_NAMES):
        total = sum(len(values) for values in model_rows.values())
        if manifest.get("model_output_row_count") != total:
            raise ReplayBundleError("bundle model-output row count mismatch")
        seen_by_file: dict[str, dict[tuple[str, str], str]] = {}
        timestamps: list[datetime] = []
        normalized: dict[str, list[str]] = {}
        rows_by_symbol: Counter[str] = Counter()
        for name, values in model_rows.items():
            seen: dict[tuple[str, str], str] = {}
            normalized[name] = []
            for row in values:
                timestamp_text = row.get("ts", row.get("timestamp", ""))
                timestamp = parse_timestamp(timestamp_text)
                if timestamp is None or not (start <= timestamp <= finish):
                    raise ReplayBundleError("bundle model-output row outside filtering window")
                symbol = str(row.get("symbol", "__pooled__") or "__pooled__")
                digest = canonical_row_digest(row)
                key = (canonical_timestamp(timestamp_text) or str(timestamp_text), symbol)
                if key in seen:
                    raise ReplayBundleError("duplicate or conflicting model-output logical row retained")
                seen[key] = digest
                normalized[name].append(digest)
                rows_by_symbol[symbol] += 1
                timestamps.append(timestamp)
            seen_by_file[name] = seen
        ordered = sorted(timestamps)
        first_model = None if not ordered else ordered[0].isoformat().replace("+00:00", "Z")
        last_model = None if not ordered else ordered[-1].isoformat().replace("+00:00", "Z")
        if manifest.get("model_output_first_timestamp") != first_model or manifest.get("model_output_last_timestamp") != last_model:
            raise ReplayBundleError("bundle model-output timestamp inventory mismatch")
        if manifest.get("model_output_rows_by_symbol") != dict(sorted(rows_by_symbol.items())):
            raise ReplayBundleError("bundle model-output symbol inventory mismatch")
        if manifest.get("model_output_digest") != _json_digest(normalized):
            raise ReplayBundleError("bundle model-output digest mismatch")
    return {
        "status": "exact_bundle",
        "manifest": manifest,
        "rows": rows,
        "model_output_rows": model_rows,
        "bundle_digest": expected_bundle,
    }


def resolve_historical_sources(
    identity: str,
    run_started_utc: str,
    finished_at: str,
    *,
    mode: Optional[str] = None,
    logs_dir: Path | str = DEFAULT_LOGS_DIR,
    bundle_root: Path | str = DEFAULT_BUNDLE_ROOT,
    reported_row_counts: Optional[Mapping[str, int]] = None,
    max_start_delay_seconds: float = 120.0,
    max_end_delay_seconds: float = 120.0,
    max_signal_gap_seconds: float = 180.0,
) -> dict[str, Any]:
    match = IDENTITY_RE.fullmatch(identity)
    if match is None:
        raise ReplayBundleError("invalid historical source identity")
    mode_name = mode or match.group("mode")
    bundle_dir = Path(bundle_root) / f"{match.group('mode')}_{match.group('timestamp')}"
    expected_counts = {
        str(kind): int(count)
        for kind, count in (reported_row_counts or {}).items()
        if kind in CSV_NAMES and isinstance(count, int) and not isinstance(count, bool) and count >= 0
    }
    if (bundle_dir / "bundle_manifest.json").is_file():
        exact = validate_replay_bundle(bundle_dir)
        if canonical_timestamp(exact["manifest"].get("run_started_utc")) != canonical_timestamp(
            run_started_utc
        ) or canonical_timestamp(exact["manifest"].get("finished_at")) != canonical_timestamp(
            finished_at
        ):
            raise ReplayBundleError("exact replay bundle window does not match Phase 19")
        rows = exact["rows"]
        stored_coverage = exact["manifest"].get("coverage_checks", {})
        coverage = calculate_coverage(
            rows,
            run_started_utc,
            finished_at,
            max_start_delay_seconds=max_start_delay_seconds,
            max_end_delay_seconds=max_end_delay_seconds,
            max_signal_gap_seconds=max_signal_gap_seconds,
            malformed_timestamps=int(stored_coverage.get("malformed_timestamp_count", 0) or 0),
            duplicate_signal_ids=stored_coverage.get("duplicate_signal_ids", []),
            conflicting_signal_ids=stored_coverage.get("conflicting_signal_ids", []),
        )
        count_checks = {
            kind: {"reported": count, "resolved": len(rows[kind]), "matched": len(rows[kind]) == count}
            for kind, count in expected_counts.items()
        }
        coverage["reported_row_count_checks"] = count_checks
        coverage["reported_row_counts_passed"] = all(
            item["matched"] for item in count_checks.values()
        )
        return {
            "status": "exact_bundle" if all(item["matched"] for item in count_checks.values()) else "incomplete",
            "rows": rows,
            "source_digest": _json_digest(
                {kind: [canonical_row_digest(row) for row in values] for kind, values in rows.items()}
            ),
            "bundle_digest": exact["bundle_digest"],
            "coverage": coverage,
            "duplicate_rows_removed": exact["manifest"].get("duplicate_rows_removed", 0),
            "conflicting_rows": [],
            "source_paths": {"bundle": [str(bundle_dir)]},
            "reported_row_count_checks": count_checks,
        }

    collected = collect_source_rows(logs_dir, run_started_utc, finished_at)
    rows = {kind: item["rows"] for kind, item in collected["kinds"].items()}
    conflict_ids = [
        str(item.get("logical_key", ["", ""])[1])
        for item in collected["conflicting_rows"]
        if len(item.get("logical_key", [])) > 1
    ]
    coverage = calculate_coverage(
        rows,
        run_started_utc,
        finished_at,
        max_start_delay_seconds=max_start_delay_seconds,
        max_end_delay_seconds=max_end_delay_seconds,
        max_signal_gap_seconds=max_signal_gap_seconds,
        malformed_timestamps=collected["malformed_timestamps"],
        duplicate_signal_ids=collected["kinds"]["signals"]["duplicate_signal_ids"],
        conflicting_signal_ids=conflict_ids,
    )
    count_checks = {
        kind: {"reported": count, "resolved": len(rows[kind]), "matched": len(rows[kind]) == count}
        for kind, count in expected_counts.items()
    }
    coverage["reported_row_count_checks"] = count_checks
    coverage["reported_row_counts_passed"] = all(
        item["matched"] for item in count_checks.values()
    )
    if collected["conflicting_rows"]:
        status = "conflicting"
    elif not rows["signals"]:
        status = "missing"
    elif "xgboost" in mode_name and not rows["xgboost"]:
        status = "incomplete"
    elif not all(item["matched"] for item in count_checks.values()):
        status = "incomplete"
    else:
        status = "resolved_from_archives"
    return {
        "status": status,
        "rows": rows,
        "source_digest": collected["source_digest"],
        "bundle_digest": None,
        "coverage": coverage,
        "duplicate_rows_removed": collected["duplicate_rows_removed"],
        "conflicting_rows": collected["conflicting_rows"],
        "source_paths": collected["source_paths"],
        "reported_row_count_checks": count_checks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate deterministic replay bundles.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--identity", required=True)
    build.add_argument("--run-started-utc", required=True)
    build.add_argument("--finished-at", required=True)
    build.add_argument("--contract", required=True)
    build.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    build.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    build.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE_ROOT))
    build.add_argument("--manifest-digest")
    build.add_argument("--model-serving-snapshot")
    validate = sub.add_parser("validate")
    validate.add_argument("--bundle", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_replay_bundle(
                args.identity,
                args.run_started_utc,
                args.finished_at,
                args.contract,
                reports_dir=args.reports_dir,
                logs_dir=args.logs_dir,
                bundle_root=args.bundle_root,
                manifest_digest_value=args.manifest_digest,
                model_serving_snapshot_path=args.model_serving_snapshot,
            )
        else:
            result = validate_replay_bundle(args.bundle)
            result = {
                "status": result["status"],
                "bundle_digest": result["bundle_digest"],
            }
        print(json.dumps(result, default=str))
        return 0
    except (ReplayBundleError, ReplayContractError, ModelServingSnapshotError, OSError) as exc:
        print(f"replay_bundle_error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
