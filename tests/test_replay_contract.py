"""Synthetic tests for Phase 20 replay contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.replay_contract import (
    CONTRACT_DOCUMENT_FIELDS,
    CONTRACT_FIELDS,
    ReplayContractError,
    capture_replay_contract,
    load_contract_overrides,
    replay_contract_digest,
    resolve_replay_contract,
    validate_replay_contract,
)


def _root(path: Path) -> Path:
    (path / "tools").mkdir(parents=True)
    (path / "v2").mkdir()
    (path / "config").mkdir()
    (path / "tools" / "live_executor.py").write_text("# pure fixture\n", encoding="utf-8")
    (path / "v2" / "risk_controls.py").write_text("# pure fixture\n", encoding="utf-8")
    (path / "config" / "run.json").write_text(
        json.dumps(
            {
                "executor": {
                    "DL_P_LONG": 0.5,
                    "EXEC_BIAS_GUARD": False,
                    "EXEC_RESTORE_STATE": False,
                    "EXEC_FEE_BPS": 5,
                    "EXEC_SLIPPAGE_BPS": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (path / ".env").write_text(
        "API_KEY=never-serialize\nWALLET_ADDRESS=0x1111111111111111111111111111111111111111\n",
        encoding="utf-8",
    )
    return path


def _forced(path: Path, **updates: object) -> Path:
    values: dict[str, object] = {
        "LIVE_TRADING": False,
        "PAPER_TRADING": True,
        "LIVE_MODE": False,
        "EXEC_PAPER": True,
        "PLACE_REAL_ORDERS": False,
        "EXEC_RESTORE_STATE": False,
        "SURVIVAL_EXIT_ACTIVE": False,
        "XGBOOST_SIGNAL_BLOCKING": False,
        "ISOLATION_FOREST_BLOCKING": False,
        "ADVANCED_RISK_ACTIVE": False,
        "EXEC_BIAS_GUARD": False,
    }
    values.update(updates)
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _capture(tmp_path: Path, **forced: object) -> dict[str, object]:
    root = _root(tmp_path / "repo")
    return capture_replay_contract(
        "xgboost_shadow_outcome:20260803161821",
        "xgboost_shadow_outcome",
        _forced(tmp_path / "forced.json", **forced),
        base_dir=root,
        run_started_utc="2026-08-03T16:18:21Z",
        expected_finished_at="2026-08-03T17:18:21Z",
        generated_at="first",
    )


def test_capture_contains_only_allowlisted_nonsecret_fields(tmp_path):
    contract = _capture(tmp_path)

    assert set(contract) <= CONTRACT_DOCUMENT_FIELDS
    assert set(CONTRACT_FIELDS) <= set(contract)
    serialized = json.dumps(contract)
    assert "never-serialize" not in serialized
    assert "WALLET_ADDRESS" not in serialized


def test_contract_digest_is_deterministic_and_ignores_generated_at_and_paths(tmp_path):
    first = _capture(tmp_path / "one")
    second = _capture(tmp_path / "two")
    second["generated_at"] = "second"
    second["local_path"] = r"C:\ignored"  # digest deliberately reads allowlisted fields only

    assert replay_contract_digest(first) == replay_contract_digest(second)


def test_validate_rejects_real_orders(tmp_path):
    contract = _capture(tmp_path)
    contract["place_real_orders"] = True
    contract["contract_digest"] = replay_contract_digest(contract)

    with pytest.raises(ReplayContractError):
        validate_replay_contract(contract)


@pytest.mark.parametrize(
    "field",
    ["survival_active", "xgboost_blocking", "iforest_blocking", "advanced_risk_active"],
)
def test_validate_rejects_active_or_blocking_flags(tmp_path, field):
    contract = _capture(tmp_path)
    contract[field] = True
    contract["contract_digest"] = replay_contract_digest(contract)

    with pytest.raises(ReplayContractError):
        validate_replay_contract(contract)


def test_current_env_is_not_assumed_to_be_historical_contract(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    registry = tmp_path / "overrides.json"
    registry.write_text('{"schema_version": 1, "contracts": {}}', encoding="utf-8")
    (tmp_path / ".env").write_text("EXEC_PAPER=true\n", encoding="utf-8")

    result = resolve_replay_contract(
        "xgboost_shadow_outcome:20260803161821", reports, registry
    )

    assert result["status"] == "missing"
    assert result["contract"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry.update(reviewed=False),
        lambda entry: entry.update(reason=""),
        lambda entry: entry.update(command="python unsafe.py"),
        lambda entry: entry.pop("fee_bps"),
    ],
)
def test_malformed_reviewed_override_is_rejected(tmp_path, mutation):
    contract = _capture(tmp_path / "capture")
    entry = {field: contract[field] for field in CONTRACT_FIELDS}
    entry.update({"reviewed": True, "reason": "independently verified paper settings"})
    mutation(entry)
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contracts": {contract["identity"]: entry},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReplayContractError):
        load_contract_overrides(path)


def test_config_env_and_forced_override_precedence(tmp_path):
    root = _root(tmp_path / "repo")
    with (root / ".env").open("a", encoding="utf-8") as handle:
        handle.write("DL_P_LONG=0.6\n")
    contract = capture_replay_contract(
        "baseline:20260803161821",
        "baseline",
        _forced(tmp_path / "forced.json", DL_P_LONG=0.7),
        base_dir=root,
    )

    assert contract["exec_thr"] == 0.7


def test_capture_rejects_nonpaper_contract(tmp_path):
    root = _root(tmp_path / "repo")

    with pytest.raises(ReplayContractError):
        capture_replay_contract(
            "baseline:20260803161821",
            "baseline",
            _forced(tmp_path / "forced.json", EXEC_PAPER=False),
            base_dir=root,
        )


def test_reviewed_override_rejects_secret_reason(tmp_path):
    contract = _capture(tmp_path / "capture")
    entry = {field: contract[field] for field in CONTRACT_FIELDS}
    entry.update({"reviewed": True, "reason": "API_KEY=must-not-appear"})
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps({"schema_version": 1, "contracts": {contract["identity"]: entry}}),
        encoding="utf-8",
    )

    with pytest.raises(ReplayContractError):
        load_contract_overrides(path)


def test_reviewed_override_rejects_executable_reason(tmp_path):
    contract = _capture(tmp_path / "capture")
    entry = {field: contract[field] for field in CONTRACT_FIELDS}
    entry.update({"reviewed": True, "reason": "python unsafe.py"})
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps({"schema_version": 1, "contracts": {contract["identity"]: entry}}),
        encoding="utf-8",
    )

    with pytest.raises(ReplayContractError):
        load_contract_overrides(path)


def test_contract_rejects_finish_before_start(tmp_path):
    root = _root(tmp_path / "repo")

    with pytest.raises(ReplayContractError):
        capture_replay_contract(
            "baseline:20260803161821",
            "baseline",
            _forced(tmp_path / "forced.json"),
            base_dir=root,
            run_started_utc="2026-08-03T17:00:00Z",
            expected_finished_at="2026-08-03T16:00:00Z",
        )
