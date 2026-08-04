"""Phase 19 paper-only evidence registry for experiment matrix runs.

The registry is intentionally read-only with respect to trading configuration,
runtime behavior, and model artifacts.  It discovers generated matrix evidence,
classifies each canonical mode/timestamp identity, and fails closed for legacy
or otherwise unverifiable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_OVERRIDES_PATH = BASE_DIR / "research" / "evidence_overrides.json"
DEFAULT_JSON_OUT = DEFAULT_REPORTS_DIR / "evidence_manifest.json"
SCHEMA_VERSION = 1

CLASSIFICATIONS = (
    "valid_strategy_evidence",
    "valid_safety_only",
    "incomplete_no_outcomes",
    "contaminated_stale_signal",
    "network_interrupted",
    "invalid_matrix_failure",
    "unverified_legacy",
)
INCLUSION_FLAGS = {
    "valid_strategy_evidence": (True, True),
    "valid_safety_only": (False, True),
    "incomplete_no_outcomes": (False, True),
    "contaminated_stale_signal": (False, False),
    "network_interrupted": (False, False),
    "invalid_matrix_failure": (False, False),
    "unverified_legacy": (False, False),
}

IDENTITY_RE = re.compile(r"^(?P<mode>[a-z0-9_]+):(?P<timestamp>\d{14})$")
INDEX_RE = re.compile(r"^matrix_index_(?P<timestamp>\d{14})\.json$")
REPORT_RE = re.compile(
    r"^matrix_(?P<mode>.+)_(?P<timestamp>\d{14})_"
    r"(?P<kind>unified|shadow_summary|xgboost_audit)\.json$"
)
REPORT_KINDS = ("unified", "shadow_summary", "xgboost_audit")
OVERRIDE_KEYS = {"classification", "reason", "reviewed"}
FORBIDDEN_OVERRIDE_KEYS = {
    "command",
    "commands",
    "environment",
    "settings",
    "threshold",
    "risk",
    "fees",
    "slippage",
    "position_size",
    "place_real_orders",
}
EXECUTABLE_OVERRIDE_REASON_RE = re.compile(
    r"(?i)(?:\bpowershell(?:\.exe)?\b|\bcmd(?:\.exe)?\s+/c\b|"
    r"\b(?:python(?:3)?|py)\s+\S+\.py\b|\b(?:bash|sh)\s+\S+|"
    r"\.(?:ps1|bat|cmd|sh)\b|(?:^|\s)--[a-z][a-z0-9-]*)"
)
SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?P<key>api[_-]?(?:key|secret)|private[_-]?key|"
    r"wallet[_-]?(?:address|key)|mnemonic|seed[_-]?phrase|password|token)\b"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
ENV_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\b[A-Z][A-Z0-9_]{2,})\s*=\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
PRIVATE_HEX_RE = re.compile(r"(?i)(?<![0-9a-f])0x[0-9a-f]{64}(?![0-9a-f])")
WALLET_ADDRESS_RE = re.compile(r"(?i)(?<![0-9a-f])0x[0-9a-f]{40}(?![0-9a-f])")


class EvidenceManifestError(ValueError):
    """Raised when authoritative evidence inputs cannot be trusted."""


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: JSON root is not an object"
    return payload, None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        if not number.is_integer():
            return None
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        if value is None or str(value).strip() == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _as_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _notes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _sanitize_evidence_text(value: str) -> str:
    """Remove secrets and environment values while preserving audit context."""

    text = PRIVATE_HEX_RE.sub("[redacted_private_key]", value)
    text = WALLET_ADDRESS_RE.sub("[redacted_wallet_address]", text)
    text = SENSITIVE_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}=[redacted]", text
    )
    return ENV_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('key')}=[redacted]", text
    )


def _sanitize_notes(value: Any) -> List[str]:
    return [_sanitize_evidence_text(note) for note in _notes(value)]


def _nested_int(payload: Optional[Dict[str, Any]], *keys: str) -> int:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    parsed = _as_int(value)
    return 0 if parsed is None else max(0, parsed)


def load_overrides(path: Path | str = DEFAULT_OVERRIDES_PATH) -> Dict[str, Dict[str, Any]]:
    """Load and strictly validate the tracked reviewed override registry."""

    override_path = Path(path)
    payload, error = _read_json(override_path)
    if error is not None or payload is None:
        raise EvidenceManifestError(
            f"unreadable or malformed override registry: {error or override_path}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceManifestError("override registry schema_version must be 1")
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        raise EvidenceManifestError("override registry overrides must be an object")

    validated: Dict[str, Dict[str, Any]] = {}
    for identity, entry in overrides.items():
        if not isinstance(identity, str) or IDENTITY_RE.fullmatch(identity) is None:
            raise EvidenceManifestError(
                f"override identity must follow mode:14-digit-timestamp: {identity!r}"
            )
        if not isinstance(entry, dict):
            raise EvidenceManifestError(f"override {identity} must be an object")
        unknown = set(entry) - OVERRIDE_KEYS
        if unknown or set(entry) & FORBIDDEN_OVERRIDE_KEYS:
            raise EvidenceManifestError(
                f"override {identity} contains prohibited or unknown fields: {sorted(unknown)}"
            )
        classification = entry.get("classification")
        if classification not in CLASSIFICATIONS:
            raise EvidenceManifestError(
                f"override {identity} has unknown classification: {classification!r}"
            )
        if entry.get("reviewed") is not True:
            raise EvidenceManifestError(f"override {identity} must have reviewed=true")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise EvidenceManifestError(f"override {identity} reason must be non-empty")
        if EXECUTABLE_OVERRIDE_REASON_RE.search(reason):
            raise EvidenceManifestError(
                f"override {identity} reason must not contain executable commands"
            )
        if (
            SENSITIVE_ASSIGNMENT_RE.search(reason)
            or PRIVATE_HEX_RE.search(reason)
            or WALLET_ADDRESS_RE.search(reason)
        ):
            raise EvidenceManifestError(
                f"override {identity} reason must not contain sensitive values"
            )
        validated[identity] = {
            "classification": classification,
            "reason": reason.strip(),
            "reviewed": True,
        }
    return validated


def _report_identity(path: Path) -> Optional[Tuple[str, str, str]]:
    match = REPORT_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("mode"), match.group("timestamp"), match.group("kind")


def _matrix_timestamp(
    item: Dict[str, Any],
    payload: Dict[str, Any],
    fallback_timestamp: str,
    mode: str,
) -> str:
    report_paths = item.get("report_paths")
    if isinstance(report_paths, dict):
        for raw_path in report_paths.values():
            parsed = _report_identity(Path(str(raw_path).replace("\\", "/")))
            if parsed is not None and parsed[0] == mode:
                return parsed[1]
    for value in (item.get("matrix_timestamp"), payload.get("matrix_timestamp")):
        text = str(value or "").strip()
        if re.fullmatch(r"\d{14}", text):
            return text
    return fallback_timestamp


def _index_record(
    item: Dict[str, Any],
    payload: Dict[str, Any],
    fallback_timestamp: str,
    index_path: Path,
) -> Optional[Dict[str, Any]]:
    mode = str(item.get("mode") or payload.get("requested_mode") or "").strip()
    if not re.fullmatch(r"[a-z0-9_]+", mode):
        return None
    timestamp = _matrix_timestamp(item, payload, fallback_timestamp, mode)
    expected_kinds: List[str] = []
    report_paths = item.get("report_paths")
    if isinstance(report_paths, dict):
        expected_kinds = sorted(
            str(key) for key in report_paths if str(key) in REPORT_KINDS
        )
    return {
        "identity": f"{mode}:{timestamp}",
        "mode": mode,
        "matrix_timestamp": timestamp,
        "run_started_utc": item.get("run_started_utc") or item.get("run_timestamp"),
        "finished_at": item.get("finished_at"),
        "duration_minutes": _as_number(
            item.get("duration_minutes", payload.get("duration_minutes"))
        ),
        "exit_status": _as_int(item.get("exit_status")),
        "stale_entry_guard_checked": _as_bool(item.get("stale_entry_guard_checked")),
        "stale_entry_count": _as_int(item.get("stale_entry_count")),
        "stale_entry_signal_ids": (
            [str(value) for value in item.get("stale_entry_signal_ids", [])]
            if isinstance(item.get("stale_entry_signal_ids", []), list)
            else []
        ),
        "evidence_valid": _as_bool(item.get("evidence_valid")),
        "notes": _sanitize_notes(item.get("notes")),
        "expected_report_kinds": expected_kinds,
        "index_path": str(index_path),
    }


def _index_signature(record: Dict[str, Any]) -> str:
    fields = {
        key: record.get(key)
        for key in (
            "identity",
            "run_started_utc",
            "finished_at",
            "duration_minutes",
            "exit_status",
            "stale_entry_guard_checked",
            "stale_entry_count",
            "stale_entry_signal_ids",
            "evidence_valid",
            "notes",
            "expected_report_kinds",
        )
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _classification_reason(
    record: Optional[Dict[str, Any]],
    mode: str,
    statuses: Dict[str, str],
    closed_trade_count: int,
    matched_closed_trade_count: int,
) -> Tuple[str, str]:
    if record is None:
        return (
            "unverified_legacy",
            "report exists without a matching verified evidence index",
        )

    exit_status = record.get("exit_status")
    stale_count = record.get("stale_entry_count")
    evidence_valid = record.get("evidence_valid")
    guard_checked = record.get("stale_entry_guard_checked")
    joined_notes = " ".join(record.get("notes") or []).lower()
    if (
        (stale_count is not None and stale_count > 0)
        or "stale_signal_replay_or_prestart_entry_detected" in joined_notes
    ):
        return (
            "contaminated_stale_signal",
            f"stale-entry evidence detected (stale_entry_count={stale_count or 0})",
        )
    if exit_status is not None and exit_status != 0:
        return "invalid_matrix_failure", f"matrix index exit_status={exit_status}"
    malformed_reports = [
        kind for kind in REPORT_KINDS if statuses.get(kind) == "malformed"
    ]
    if malformed_reports:
        return (
            "invalid_matrix_failure",
            "malformed generated report prevents reliable interpretation: "
            + ", ".join(malformed_reports),
        )
    if evidence_valid is None:
        return (
            "unverified_legacy",
            "run predates required evidence_valid and stale-entry verification metadata",
        )
    if exit_status is None:
        return "invalid_matrix_failure", "matrix index exit_status is missing or malformed"
    if evidence_valid is not True:
        return "invalid_matrix_failure", "matrix index evidence_valid is not true"
    if guard_checked is not True or stale_count is None or stale_count < 0:
        return (
            "invalid_matrix_failure",
            "required stale-entry guard result is missing or incomplete",
        )
    duration = record.get("duration_minutes")
    if duration is not None and duration <= 0:
        return "invalid_matrix_failure", "duration_minutes is invalid"

    expected = set(record.get("expected_report_kinds") or [])
    if not expected:
        expected.add("unified")
    bad_reports = [kind for kind in sorted(expected) if statuses.get(kind) != "present"]
    if bad_reports:
        details = ", ".join(f"{kind}={statuses.get(kind, 'missing')}" for kind in bad_reports)
        return "invalid_matrix_failure", f"required generated report unavailable: {details}"

    if mode == "baseline" and closed_trade_count > 0:
        return (
            "valid_strategy_evidence",
            f"clean baseline run has {closed_trade_count} usable closed trade outcome(s)",
        )
    if mode == "xgboost_shadow_outcome" and matched_closed_trade_count > 0:
        return (
            "valid_strategy_evidence",
            "clean XGBoost run has "
            f"{matched_closed_trade_count} uniquely matched closed-trade outcome(s); "
            "this count alone is not sufficient for activation or profitability",
        )
    if mode == "combined_shadow" and closed_trade_count > 0:
        return (
            "valid_strategy_evidence",
            f"clean combined-shadow run has {closed_trade_count} usable closed trade outcome(s)",
        )

    if duration is not None and duration <= 5:
        return (
            "valid_safety_only",
            f"clean {duration:g}-minute safety validation; strategy outcomes are not required",
        )
    if mode == "xgboost_shadow_outcome":
        reason = (
            "clean strategy-evidence run has zero matched closed-trade outcomes; "
            "shadow decision rows are not outcome evidence"
        )
    elif mode in {"baseline", "combined_shadow"}:
        reason = "clean strategy-evidence run has zero usable closed-trade outcomes"
    else:
        reason = (
            "clean completed run has no explicit mode-specific strategy outcome requirement; "
            "retained for safety summary only"
        )
    return "incomplete_no_outcomes", reason


def _run_from_sources(
    identity: str,
    record: Optional[Dict[str, Any]],
    reports: Dict[str, Tuple[Path, Optional[Dict[str, Any]], Optional[str]]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    match = IDENTITY_RE.fullmatch(identity)
    if match is None:  # Defensive: all callers construct canonical identities.
        raise EvidenceManifestError(f"internal invalid evidence identity: {identity}")
    mode = match.group("mode")
    timestamp = match.group("timestamp")
    statuses = {
        kind: (
            "missing"
            if kind not in reports
            else ("malformed" if reports[kind][2] is not None else "present")
        )
        for kind in REPORT_KINDS
    }
    unified = reports.get("unified", (Path(), None, None))[1]
    closed_trade_count = _nested_int(unified, "paper_pnl", "closed_trade_count")
    lineage_closed_rows = _nested_int(unified, "trade_lineage", "closed_trade_rows")
    matched_count = _nested_int(unified, "xgboost_outcome", "matched_closed_trade_count")
    confirm_count = _nested_int(unified, "xgboost_outcome", "would_confirm_matched_count")
    reject_count = _nested_int(unified, "xgboost_outcome", "would_reject_matched_count")
    warnings: List[str] = []
    if closed_trade_count != lineage_closed_rows and unified is not None:
        warnings.append(
            "paper_pnl.closed_trade_count differs from trade_lineage.closed_trade_rows"
        )
    if matched_count != confirm_count + reject_count and unified is not None:
        warnings.append(
            "matched_closed_trade_count differs from confirm plus reject matched counts"
        )

    automatic, automatic_reason = _classification_reason(
        record,
        mode,
        statuses,
        closed_trade_count,
        matched_count,
    )
    if override is not None:
        classification = str(override["classification"])
        source = "reviewed_override"
        reason = str(override["reason"])
        if classification != automatic:
            warnings.append(f"reviewed override replaces automatic classification {automatic}")
    else:
        classification = automatic
        source = "automatic"
        reason = automatic_reason
    if classification not in INCLUSION_FLAGS:
        raise EvidenceManifestError(f"internal unknown classification: {classification}")
    strategy, safety = INCLUSION_FLAGS[classification]

    base = record or {}
    return {
        "identity": identity,
        "mode": mode,
        "matrix_timestamp": timestamp,
        "run_started_utc": base.get("run_started_utc"),
        "finished_at": base.get("finished_at"),
        "duration_minutes": base.get("duration_minutes"),
        "exit_status": base.get("exit_status"),
        "stale_entry_guard_checked": base.get("stale_entry_guard_checked"),
        "stale_entry_count": base.get("stale_entry_count"),
        "stale_entry_signal_ids": base.get("stale_entry_signal_ids", []),
        "evidence_valid": base.get("evidence_valid"),
        "classification": classification,
        "classification_source": source,
        "classification_reason": reason,
        "include_in_strategy_aggregate": strategy,
        "include_in_safety_summary": safety,
        "report_paths": {
            kind: str(value[0]) for kind, value in sorted(reports.items())
        },
        "unified_report_status": statuses["unified"],
        "shadow_summary_status": statuses["shadow_summary"],
        "xgboost_audit_status": statuses["xgboost_audit"],
        "closed_trade_count": closed_trade_count,
        "trade_lineage_closed_trade_rows": lineage_closed_rows,
        "matched_closed_trade_count": matched_count,
        "would_confirm_matched_count": confirm_count,
        "would_reject_matched_count": reject_count,
        "notes": base.get("notes", []),
        "warnings": warnings,
        "reviewed_override": override,
    }


def _digest_content(manifest: Dict[str, Any]) -> Dict[str, Any]:
    digest_runs: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        digest_runs.append(
            {
                key: run.get(key)
                for key in (
                    "identity",
                    "classification",
                    "classification_source",
                    "classification_reason",
                    "include_in_strategy_aggregate",
                    "include_in_safety_summary",
                    "run_started_utc",
                    "finished_at",
                    "duration_minutes",
                    "exit_status",
                    "stale_entry_guard_checked",
                    "stale_entry_count",
                    "stale_entry_signal_ids",
                    "evidence_valid",
                    "unified_report_status",
                    "shadow_summary_status",
                    "xgboost_audit_status",
                    "closed_trade_count",
                    "trade_lineage_closed_trade_rows",
                    "matched_closed_trade_count",
                    "would_confirm_matched_count",
                    "would_reject_matched_count",
                    "notes",
                    "reviewed_override",
                )
            }
        )
    return {"schema_version": manifest.get("schema_version"), "runs": digest_runs}


def evidence_manifest_digest(manifest: Dict[str, Any]) -> str:
    """Return a platform-independent digest of evidence-relevant content."""

    encoded = json.dumps(
        _digest_content(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_evidence_manifest(
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    overrides_path: Path | str = DEFAULT_OVERRIDES_PATH,
    *,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover, classify, and summarize all canonical matrix identities."""

    reports_path = Path(reports_dir)
    overrides = load_overrides(overrides_path)
    index_paths = sorted(reports_path.glob("matrix_index_*.json"), key=lambda path: path.name)
    report_paths: List[Path] = []
    for pattern in (
        "matrix_*_unified.json",
        "matrix_*_shadow_summary.json",
        "matrix_*_xgboost_audit.json",
    ):
        report_paths.extend(reports_path.glob(pattern))
    report_paths = sorted(set(report_paths), key=lambda path: path.name)

    malformed: List[Dict[str, str]] = []
    indexed: Dict[str, Dict[str, Any]] = {}
    for path in index_paths:
        payload, error = _read_json(path)
        if error is not None or payload is None:
            malformed.append({"path": str(path), "error": str(error)})
            continue
        runs = payload.get("runs")
        if not isinstance(runs, list):
            malformed.append({"path": str(path), "error": "top-level runs is not a list"})
            continue
        match = INDEX_RE.fullmatch(path.name)
        if match is None:
            continue
        for position, item in enumerate(runs):
            if not isinstance(item, dict):
                malformed.append(
                    {"path": str(path), "error": f"runs[{position}] is not an object"}
                )
                continue
            record = _index_record(item, payload, match.group("timestamp"), path)
            if record is None:
                malformed.append(
                    {"path": str(path), "error": f"runs[{position}] has invalid or missing mode"}
                )
                continue
            identity = record["identity"]
            if identity in indexed:
                if _index_signature(indexed[identity]) != _index_signature(record):
                    raise EvidenceManifestError(
                        f"duplicate contradictory evidence identity: {identity}"
                    )
                continue
            indexed[identity] = record

    grouped_reports: Dict[
        str, Dict[str, Tuple[Path, Optional[Dict[str, Any]], Optional[str]]]
    ] = {}
    for path in report_paths:
        parsed = _report_identity(path)
        if parsed is None:
            continue
        mode, timestamp, kind = parsed
        identity = f"{mode}:{timestamp}"
        payload, error = _read_json(path)
        if error is not None:
            malformed.append({"path": str(path), "error": error})
        parts = grouped_reports.setdefault(identity, {})
        if kind in parts and parts[kind][0] != path:
            raise EvidenceManifestError(
                f"duplicate contradictory evidence report identity: {identity}:{kind}"
            )
        parts[kind] = (path, payload, error)

    identities = sorted(
        set(indexed) | set(grouped_reports),
        key=lambda value: (value.rsplit(":", 1)[1], value.rsplit(":", 1)[0]),
    )
    unmatched_reports: List[Dict[str, str]] = []
    for identity, parts in grouped_reports.items():
        if identity not in indexed:
            for kind, (path, _payload, _error) in sorted(parts.items()):
                unmatched_reports.append(
                    {
                        "path": str(path),
                        "identity": identity,
                        "kind": kind,
                        "reason": "report has no matching verified matrix index",
                    }
                )

    runs = [
        _run_from_sources(
            identity,
            indexed.get(identity),
            grouped_reports.get(identity, {}),
            overrides.get(identity),
        )
        for identity in identities
    ]
    counts = Counter(run["classification"] for run in runs)
    classification_counts = {
        classification: counts.get(classification, 0)
        for classification in CLASSIFICATIONS
    }
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy": {
            "fail_closed_for_legacy": True,
            "paper_only": True,
            "real_orders_allowed": False,
        },
        "inputs": {
            "reports_dir": str(reports_path),
            "overrides_path": str(Path(overrides_path)),
            "matrix_indexes_found": len(index_paths),
            "report_files_found": len(report_paths),
            "malformed_inputs": sorted(
                malformed, key=lambda item: (item.get("path", ""), item.get("error", ""))
            ),
            "unmatched_reports": sorted(
                unmatched_reports,
                key=lambda item: (item["identity"], item["kind"], item["path"]),
            ),
        },
        "summary": {
            "total_runs": len(runs),
            "classification_counts": classification_counts,
            "strategy_included_count": sum(
                bool(run["include_in_strategy_aggregate"]) for run in runs
            ),
            "safety_included_count": sum(
                bool(run["include_in_safety_summary"]) for run in runs
            ),
            "excluded_count": sum(
                not run["include_in_strategy_aggregate"]
                and not run["include_in_safety_summary"]
                for run in runs
            ),
        },
        "runs": runs,
    }
    manifest["evidence_manifest_digest"] = evidence_manifest_digest(manifest)
    return manifest


def write_evidence_manifest(
    manifest: Dict[str, Any], out_path: Path | str = DEFAULT_JSON_OUT
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _lines_for_runs(runs: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        f"  {run['identity']}: {run['classification']} -- {run['classification_reason']}"
        for run in runs
    ]


def format_text_manifest(manifest: Dict[str, Any]) -> str:
    summary = manifest["summary"]
    runs = manifest["runs"]
    strategy = [run for run in runs if run["include_in_strategy_aggregate"]]
    safety_only = [
        run
        for run in runs
        if run["include_in_safety_summary"]
        and not run["include_in_strategy_aggregate"]
    ]
    excluded = [run for run in runs if not run["include_in_safety_summary"]]
    lines = [
        "Phase 19 Evidence Manifest",
        f"total_runs: {summary['total_runs']}",
        "classification_counts:",
    ]
    lines.extend(
        f"  {classification}: {summary['classification_counts'][classification]}"
        for classification in CLASSIFICATIONS
    )
    lines.append("strategy_included_runs:")
    lines.extend(_lines_for_runs(strategy) or ["  none"])
    lines.append("safety_only_or_incomplete_runs:")
    lines.extend(_lines_for_runs(safety_only) or ["  none"])
    lines.append("excluded_runs:")
    lines.extend(_lines_for_runs(excluded) or ["  none"])
    inputs = manifest["inputs"]
    lines.append(f"malformed_inputs: {json.dumps(inputs['malformed_inputs'], sort_keys=True)}")
    lines.append(f"unmatched_reports: {json.dumps(inputs['unmatched_reports'], sort_keys=True)}")
    lines.append(f"evidence_manifest_digest: {manifest['evidence_manifest_digest']}")
    lines.append("paper_only_evidence_registry_no_trading_changes")
    return "\n".join(lines)


def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper-only Phase 19 matrix evidence registry."
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--overrides", default=str(DEFAULT_OVERRIDES_PATH))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_args(argv)
    try:
        manifest = build_evidence_manifest(args.reports_dir, args.overrides)
        out = write_evidence_manifest(manifest, args.json_out)
    except (EvidenceManifestError, OSError) as exc:
        print(f"evidence_manifest_error: {exc}")
        return 1
    if args.json:
        # Keep stdout machine-readable. The output location is already represented by
        # --json-out and must not be appended after the JSON document.
        print(json.dumps(manifest, indent=2))
    else:
        print(format_text_manifest(manifest))
        print(f"json_written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
