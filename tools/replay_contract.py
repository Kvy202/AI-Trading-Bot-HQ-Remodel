"""Capture and validate deterministic, non-secret executor replay contracts.

This module is deliberately stdlib-only.  It reads configuration files into a
private mapping, never mutates ``os.environ``, and never imports executor,
exchange, market-data, or model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OVERRIDES_PATH = BASE_DIR / "research" / "replay_contract_overrides.json"
SCHEMA_VERSION = 1

CONTRACT_FIELDS = (
    "schema_version",
    "identity",
    "mode",
    "run_started_utc",
    "expected_finished_at",
    "git_commit",
    "live_executor_sha256",
    "v2_risk_controls_sha256",
    "exec_thr",
    "exec_mode",
    "respect_writer_thr",
    "rv_max",
    "cooldown_sec",
    "sides",
    "max_symbols",
    "one_position",
    "notional_usdt",
    "max_portfolio_usdt",
    "min_notional",
    "min_qty",
    "tp_pct",
    "sl_pct",
    "fee_bps",
    "slippage_bps",
    "flip_open",
    "flip_confirm_ticks",
    "scale_in",
    "adaptive",
    "target_pass",
    "window_signals",
    "thr_min",
    "thr_max",
    "thr_alpha",
    "bias_guard",
    "restore_state",
    "v2_enabled",
    "v2_time_stop_minutes",
    "survival_active",
    "xgboost_blocking",
    "iforest_blocking",
    "advanced_risk_active",
    "place_real_orders",
    "paper_mode",
)
CONTRACT_DOCUMENT_FIELDS = set(CONTRACT_FIELDS) | {
    "generated_at",
    "contract_status",
    "contract_digest",
}
PARITY_GRADE_STATUSES = {"exact_matrix_snapshot", "reviewed_historical_override"}
CONTRACT_STATUSES = PARITY_GRADE_STATUSES | {"reconstructed_unverified", "missing"}
IDENTITY_RE = re.compile(r"^(?P<mode>[a-z0-9_]+):(?P<timestamp>\d{14})$")
COMMAND_RE = re.compile(
    r"(?i)(?:\bpowershell(?:\.exe)?\b|\bcmd(?:\.exe)?\s+/c\b|"
    r"\b(?:python(?:3)?|py)\s+\S+\.py\b|\b(?:bash|sh)\s+\S+|"
    r"\.(?:ps1|bat|cmd|sh)\b|(?:^|\s)--[a-z][a-z0-9-]*)"
)
SENSITIVE_RE = re.compile(
    r"(?i)(?:api[_-]?(?:key|secret)|private[_-]?key|wallet[_-]?(?:address|key)|"
    r"mnemonic|seed[_-]?phrase|password|token)\s*[:=]\s*\S+|"
    r"(?<![0-9a-f])0x(?:[0-9a-f]{40}|[0-9a-f]{64})(?![0-9a-f])"
)


class ReplayContractError(ValueError):
    """Raised when a replay contract cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    text = str(value or "").strip().replace("+0000", "+00:00")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReplayContractError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_utc(value: str) -> str:
    return _parse_utc(value).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    except OSError as exc:
        raise ReplayContractError(f"required replay source is unreadable: {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ReplayContractError(f"required replay source is not UTF-8 text: {path}: {exc}") from exc


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        value = result.stdout.strip()
        return value if re.fullmatch(r"[0-9a-fA-F]{40}", value) else "unknown"
    except Exception:
        return "unknown"


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_run_config(root: Path) -> dict[str, str]:
    path = root / "config" / "run.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReplayContractError(f"malformed config/run.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayContractError("config/run.json root must be an object")
    result: dict[str, str] = {}
    for section in payload.values():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if isinstance(key, str) and key and not key.startswith("_"):
                result[key] = _stringify(value)
    return result


def _load_dotenv_memory(path: Path) -> dict[str, str]:
    """Read dotenv syntax without writing values to the process environment."""

    if not path.exists():
        return {}
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise ReplayContractError(f"unable to read .env in memory: {exc}") from exc
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        result[key] = value
    return result


def _load_forced_env(path: Path | str | None) -> dict[str, str]:
    if path is None:
        return {}
    forced_path = Path(path)
    try:
        payload = json.loads(forced_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReplayContractError(f"malformed forced environment JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayContractError("forced environment JSON root must be an object")
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ReplayContractError(f"invalid forced environment key: {key!r}")
        if isinstance(value, (dict, list)):
            raise ReplayContractError(f"forced environment value must be scalar: {key}")
        result[key] = _stringify(value)
    return result


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = str(env.get(name, "true" if default else "false")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        value = float(str(env.get(name, default)).strip())
    except Exception:
        value = float(default)
    if not math.isfinite(value):
        raise ReplayContractError(f"non-finite replay setting: {name}")
    return value


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _float(env, name, float(default))
    if not value.is_integer():
        raise ReplayContractError(f"non-integral replay setting: {name}")
    return int(value)


def _first_float(env: Mapping[str, str], names: Sequence[str], default: float) -> float:
    for name in names:
        if name in env and str(env[name]).strip() != "":
            return _float(env, name, default)
    return float(default)


def _digest_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {key: contract.get(key) for key in CONTRACT_FIELDS}


def replay_contract_digest(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _digest_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inferred_start(identity: str) -> str:
    match = IDENTITY_RE.fullmatch(identity)
    if match is None:
        raise ReplayContractError("identity must follow mode:14-digit-timestamp")
    parsed = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )
    return parsed.isoformat().replace("+00:00", "Z")


def capture_replay_contract(
    identity: str,
    mode: str,
    forced_env_json: Path | str | None = None,
    *,
    base_dir: Path | str = BASE_DIR,
    run_started_utc: Optional[str] = None,
    expected_finished_at: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> dict[str, Any]:
    """Capture the effective allowlisted executor settings for a future run."""

    match = IDENTITY_RE.fullmatch(identity)
    if match is None or match.group("mode") != mode:
        raise ReplayContractError("identity must match mode and use mode:14-digit-timestamp")
    root = Path(base_dir)
    env = _load_run_config(root)
    env.update(_load_dotenv_memory(root / ".env"))
    env.update(_load_forced_env(forced_env_json))

    started = _canonical_utc(run_started_utc or _inferred_start(identity))
    finished = _canonical_utc(expected_finished_at or started)
    if _parse_utc(finished) < _parse_utc(started):
        raise ReplayContractError("expected_finished_at precedes run_started_utc")

    v2_disabled = _bool(env, "V2_RISK_DISABLED", False)
    v2_time = max(0.0, _float(env, "V2_TIME_STOP_MIN", 0.0))
    other_v2_active = any(
        value > 0
        for value in (
            _float(env, "V2_MAX_SL_PER_DAY", 0.0),
            _float(env, "V2_DAILY_LOSS_LIMIT_USDT", 0.0),
            _float(env, "V2_DAILY_DD_LIMIT_USDT", 0.0),
        )
    )
    if other_v2_active and not v2_disabled:
        raise ReplayContractError(
            "active V2 entry-pause controls are outside the replay contract allowlist"
        )

    exec_mode = str(env.get("DL_P_LONG_MODE", "abs") or "abs").strip().lower()
    sides = str(env.get("EXEC_SIDES", "both") or "both").strip().lower()
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "mode": mode,
        "run_started_utc": started,
        "expected_finished_at": finished,
        "git_commit": _git_commit(root),
        "live_executor_sha256": _sha256_file(root / "tools" / "live_executor.py"),
        "v2_risk_controls_sha256": _sha256_file(root / "v2" / "risk_controls.py"),
        "exec_thr": _float(env, "DL_P_LONG", 0.55),
        "exec_mode": exec_mode,
        "respect_writer_thr": _bool(env, "EXEC_RESPECT_WRITER_THR", True),
        "rv_max": _first_float(env, ("EXEC_RV_MAX", "DL_MAX_RV"), 0.02),
        "cooldown_sec": _float(env, "EXEC_COOLDOWN_SEC", 30.0),
        "sides": sides,
        "max_symbols": _int(env, "MAX_CONCURRENT", 1),
        "one_position": _bool(env, "EXEC_ONE_POSITION", False),
        "notional_usdt": _first_float(
            env, ("PER_SYMBOL_NOTIONAL_USDT", "MAX_NOTIONAL_USDT"), 15.0
        ),
        "max_portfolio_usdt": _float(env, "MAX_PORTFOLIO_EXPOSURE_USDT", 30.0),
        "min_notional": _float(env, "EXEC_MIN_NOTIONAL", 5.0),
        "min_qty": _float(env, "EXEC_MIN_QTY", 0.0),
        "tp_pct": _float(env, "EXEC_TP_PCT", 0.01),
        "sl_pct": _float(env, "EXEC_SL_PCT", 0.02),
        "fee_bps": _float(env, "EXEC_FEE_BPS", 5.0),
        "slippage_bps": _float(env, "EXEC_SLIPPAGE_BPS", 2.0),
        "flip_open": _bool(env, "EXEC_FLIP_OPEN", True),
        "flip_confirm_ticks": _int(env, "EXEC_FLIP_CONFIRM_TICKS", 3),
        "scale_in": _bool(env, "EXEC_SCALE_IN", False),
        "adaptive": _bool(env, "EXEC_ADAPTIVE", False),
        "target_pass": _float(env, "EXEC_TARGET_PASS", 0.20),
        "window_signals": _int(env, "EXEC_WINDOW_SIGNALS", 180),
        "thr_min": _float(env, "EXEC_THR_MIN", 0.40),
        "thr_max": _float(env, "EXEC_THR_MAX", 0.60),
        "thr_alpha": _float(env, "EXEC_THR_EMA_ALPHA", 0.20),
        "bias_guard": _bool(env, "EXEC_BIAS_GUARD", True),
        "restore_state": _bool(env, "EXEC_RESTORE_STATE", True),
        "v2_enabled": bool(not v2_disabled and v2_time > 0),
        "v2_time_stop_minutes": v2_time,
        "survival_active": _bool(env, "SURVIVAL_EXIT_ACTIVE", False),
        "xgboost_blocking": _bool(env, "XGBOOST_SIGNAL_BLOCKING", False),
        "iforest_blocking": _bool(env, "ISOLATION_FOREST_BLOCKING", False),
        "advanced_risk_active": _bool(env, "ADVANCED_RISK_ACTIVE", False),
        "place_real_orders": _bool(env, "PLACE_REAL_ORDERS", False),
        "paper_mode": bool(
            _bool(env, "EXEC_PAPER", True)
            and not _bool(env, "LIVE_MODE", False)
            and _bool(env, "PAPER_TRADING", True)
            and not _bool(env, "LIVE_TRADING", False)
        ),
        "generated_at": generated_at or _utc_now(),
        "contract_status": "exact_matrix_snapshot",
    }
    validate_replay_contract(contract, require_digest=False)
    contract["contract_digest"] = replay_contract_digest(contract)
    return contract


def _validate_type_and_range(contract: Mapping[str, Any]) -> None:
    bool_fields = {
        "respect_writer_thr",
        "one_position",
        "flip_open",
        "scale_in",
        "adaptive",
        "bias_guard",
        "restore_state",
        "v2_enabled",
        "survival_active",
        "xgboost_blocking",
        "iforest_blocking",
        "advanced_risk_active",
        "place_real_orders",
        "paper_mode",
    }
    int_fields = {"max_symbols", "flip_confirm_ticks", "window_signals"}
    number_fields = {
        "exec_thr",
        "rv_max",
        "cooldown_sec",
        "notional_usdt",
        "max_portfolio_usdt",
        "min_notional",
        "min_qty",
        "tp_pct",
        "sl_pct",
        "fee_bps",
        "slippage_bps",
        "target_pass",
        "thr_min",
        "thr_max",
        "thr_alpha",
        "v2_time_stop_minutes",
    }
    for field in bool_fields:
        if not isinstance(contract.get(field), bool):
            raise ReplayContractError(f"contract field {field} must be boolean")
    for field in int_fields:
        value = contract.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReplayContractError(f"contract field {field} must be an integer")
    for field in number_fields:
        value = contract.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplayContractError(f"contract field {field} must be numeric")
        if not math.isfinite(float(value)):
            raise ReplayContractError(f"contract field {field} must be finite")
    if contract.get("exec_mode") not in {"abs", "raw"}:
        raise ReplayContractError("exec_mode must be abs or raw")
    if contract.get("sides") not in {"both", "long_only", "short_only"}:
        raise ReplayContractError("sides must be both, long_only, or short_only")
    for field in ("live_executor_sha256", "v2_risk_controls_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(contract.get(field) or "")) is None:
            raise ReplayContractError(f"contract field {field} must be a SHA-256 digest")
    git_commit = str(contract.get("git_commit") or "")
    if git_commit != "unknown" and re.fullmatch(r"[0-9a-fA-F]{40}", git_commit) is None:
        raise ReplayContractError("git_commit must be a Git commit hash or unknown")
    if int(contract["max_symbols"]) <= 0 or int(contract["window_signals"]) <= 0:
        raise ReplayContractError("max_symbols and window_signals must be positive")
    if int(contract["flip_confirm_ticks"]) < 0:
        raise ReplayContractError("flip_confirm_ticks cannot be negative")
    nonnegative = number_fields - {"exec_thr", "rv_max", "thr_min", "thr_max"}
    for field in nonnegative:
        if float(contract[field]) < 0:
            raise ReplayContractError(f"contract field {field} cannot be negative")


def validate_replay_contract(
    contract: Mapping[str, Any], *, require_digest: bool = True
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ReplayContractError("replay contract must be an object")
    unknown = set(contract) - CONTRACT_DOCUMENT_FIELDS
    missing = set(CONTRACT_FIELDS) - set(contract)
    if unknown:
        raise ReplayContractError(f"unknown replay contract fields: {sorted(unknown)}")
    if missing:
        raise ReplayContractError(f"missing replay contract fields: {sorted(missing)}")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ReplayContractError("replay contract schema_version must be 1")
    identity = contract.get("identity")
    mode = contract.get("mode")
    match = IDENTITY_RE.fullmatch(str(identity or ""))
    if match is None or match.group("mode") != mode:
        raise ReplayContractError("contract identity must match mode")
    started = _parse_utc(str(contract.get("run_started_utc")))
    finished = _parse_utc(str(contract.get("expected_finished_at")))
    if finished < started:
        raise ReplayContractError("expected_finished_at precedes run_started_utc")
    _validate_type_and_range(contract)
    unsafe = [
        name
        for name in (
            "place_real_orders",
            "restore_state",
            "survival_active",
            "xgboost_blocking",
            "iforest_blocking",
            "advanced_risk_active",
        )
        if contract.get(name) is True
    ]
    if unsafe:
        raise ReplayContractError(f"unsafe replay contract flags: {', '.join(unsafe)}")
    if contract.get("paper_mode") is not True:
        raise ReplayContractError("unsafe replay contract: paper_mode must be true")
    status = contract.get("contract_status")
    if status is not None and status not in CONTRACT_STATUSES:
        raise ReplayContractError(f"unknown contract_status: {status!r}")
    expected = replay_contract_digest(contract)
    if require_digest:
        supplied = contract.get("contract_digest")
        if supplied != expected:
            raise ReplayContractError("replay contract digest mismatch")
    return {"valid": True, "contract_digest": expected, "safety_valid": True}


def write_replay_contract(contract: Mapping[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(contract), indent=2), encoding="utf-8")
    return out


def load_replay_contract(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReplayContractError(f"malformed replay contract {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayContractError("replay contract JSON root must be an object")
    validate_replay_contract(payload)
    return payload


def load_contract_overrides(
    path: Path | str = DEFAULT_OVERRIDES_PATH,
) -> dict[str, dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ReplayContractError(f"malformed replay contract override registry: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ReplayContractError("replay contract override schema_version must be 1")
    unknown_registry_fields = set(payload) - {"schema_version", "contracts"}
    if unknown_registry_fields:
        raise ReplayContractError(
            "unknown replay contract override registry fields: "
            f"{sorted(unknown_registry_fields)}"
        )
    entries = payload.get("contracts")
    if not isinstance(entries, dict):
        raise ReplayContractError("replay contract overrides must be an object")
    result: dict[str, dict[str, Any]] = {}
    allowed = set(CONTRACT_FIELDS) | {"reviewed", "reason"}
    for identity, entry in entries.items():
        if IDENTITY_RE.fullmatch(str(identity)) is None:
            raise ReplayContractError(f"invalid historical contract identity: {identity!r}")
        if not isinstance(entry, dict):
            raise ReplayContractError(f"historical contract {identity} must be an object")
        unknown = set(entry) - allowed
        missing = allowed - set(entry)
        if unknown:
            raise ReplayContractError(
                f"historical contract {identity} has unknown fields: {sorted(unknown)}"
            )
        if missing:
            raise ReplayContractError(
                f"historical contract {identity} is missing fields: {sorted(missing)}"
            )
        if entry.get("reviewed") is not True:
            raise ReplayContractError(f"historical contract {identity} must be reviewed")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ReplayContractError(f"historical contract {identity} reason must be non-empty")
        serialized = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if COMMAND_RE.search(reason) or SENSITIVE_RE.search(serialized):
            raise ReplayContractError(
                f"historical contract {identity} contains commands or sensitive values"
            )
        contract = {field: entry[field] for field in CONTRACT_FIELDS}
        if contract["identity"] != identity:
            raise ReplayContractError(f"historical contract identity mismatch: {identity}")
        contract["contract_status"] = "reviewed_historical_override"
        validate_replay_contract(contract, require_digest=False)
        contract["contract_digest"] = replay_contract_digest(contract)
        result[identity] = contract
    return result


def resolve_replay_contract(
    identity: str,
    reports_dir: Path | str,
    overrides_path: Path | str = DEFAULT_OVERRIDES_PATH,
    bundle_root: Path | str | None = None,
) -> dict[str, Any]:
    match = IDENTITY_RE.fullmatch(identity)
    if match is None:
        raise ReplayContractError(f"invalid replay identity: {identity!r}")
    reports = Path(reports_dir)
    mode, timestamp = match.group("mode"), match.group("timestamp")
    candidates = [reports / f"matrix_{mode}_{timestamp}_replay_contract.json"]
    root = Path(bundle_root) if bundle_root else reports / "replay_bundles"
    candidates.append(root / f"{mode}_{timestamp}" / "replay_contract.json")
    for candidate in candidates:
        if candidate.exists():
            contract = load_replay_contract(candidate)
            if contract["identity"] != identity:
                raise ReplayContractError(f"replay contract identity mismatch: {candidate}")
            contract = dict(contract)
            contract["contract_status"] = "exact_matrix_snapshot"
            contract["contract_digest"] = replay_contract_digest(contract)
            return {
                "status": "exact_matrix_snapshot",
                "contract": contract,
                "path": str(candidate),
                "digest": contract["contract_digest"],
            }
    overrides = load_contract_overrides(overrides_path)
    if identity in overrides:
        contract = overrides[identity]
        raw_registry = json.loads(Path(overrides_path).read_text(encoding="utf-8-sig"))
        return {
            "status": "reviewed_historical_override",
            "contract": contract,
            "path": str(Path(overrides_path)),
            "digest": contract["contract_digest"],
            "review_reason": raw_registry["contracts"][identity]["reason"].strip(),
        }
    return {"status": "missing", "contract": None, "path": None, "digest": None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture or validate offline replay contracts.")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--identity", required=True)
    capture.add_argument("--mode", required=True)
    capture.add_argument("--forced-env-json")
    capture.add_argument("--base-dir", default=str(BASE_DIR))
    capture.add_argument("--run-started-utc")
    capture.add_argument("--expected-finished-at")
    capture.add_argument("--json-out", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--contract", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "capture":
            contract = capture_replay_contract(
                args.identity,
                args.mode,
                args.forced_env_json,
                base_dir=args.base_dir,
                run_started_utc=args.run_started_utc,
                expected_finished_at=args.expected_finished_at,
            )
            out = write_replay_contract(contract, args.json_out)
            print(
                json.dumps(
                    {
                        "status": "exact_matrix_snapshot",
                        "contract_path": str(out),
                        "contract_digest": contract["contract_digest"],
                        "paper_only": True,
                    }
                )
            )
            return 0
        contract = load_replay_contract(args.contract)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "identity": contract["identity"],
                    "contract_digest": contract["contract_digest"],
                    "paper_only": True,
                }
            )
        )
        return 0
    except (ReplayContractError, OSError) as exc:
        print(f"replay_contract_error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
