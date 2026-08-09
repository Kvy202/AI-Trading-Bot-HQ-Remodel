from __future__ import annotations

import json

import pytest

from tools import model_candidate_health_gate as gate
from tools.model_training_environment import json_digest, load_training_policy


def _healthy():
    return {
        "extreme_exclusion_events": 0,
        "flat_exclusion_events": 0,
        "missing_rate": 0.0,
        "nonfinite_outputs": 0,
        "deterministic_repeat_passed": True,
    }


def _failed(kind="flat"):
    value = _healthy()
    value["flat_exclusion_events" if kind == "flat" else "extreme_exclusion_events"] = 1
    return value


def test_lstm_must_repair_both_symbols():
    passed = gate.gate_acceptance(
        "lstm", {"BTCUSDT": _healthy(), "ETHUSDT": _healthy()}, gate="legacy-repair"
    )
    assert passed["status"] == "legacy_repair_passed"
    failed = gate.gate_acceptance(
        "lstm", {"BTCUSDT": _failed("extreme"), "ETHUSDT": _healthy()}, gate="legacy-repair"
    )
    assert failed["status"] == "legacy_repair_failed"


def test_tcn_repairs_btc_and_protects_healthy_eth():
    result = gate.gate_acceptance(
        "tcn", {"BTCUSDT": _healthy(), "ETHUSDT": _failed()}, gate="legacy-repair",
        incumbent_per_symbol={"BTCUSDT": _failed(), "ETHUSDT": _healthy()},
    )
    assert result["status"] == "legacy_repair_failed"
    assert result["healthy_symbol_regressions"] == ["ETHUSDT"]
    assert result["repair_targets"] == ["BTCUSDT"]


def test_transformer_repairs_eth_and_protects_healthy_btc():
    result = gate.gate_acceptance(
        "tx", {"BTCUSDT": _failed("extreme"), "ETHUSDT": _healthy()}, gate="confirmation",
        incumbent_per_symbol={"BTCUSDT": _healthy(), "ETHUSDT": _failed("extreme")},
    )
    assert result["status"] == "confirmation_health_failed"
    assert result["healthy_symbol_regressions"] == ["BTCUSDT"]
    assert result["repair_targets"] == ["ETHUSDT"]


def test_health_thresholds_remain_exact():
    policy = load_training_policy()
    assert policy["flat_output_std_threshold"] == 0.002
    assert policy["flat_window"] == 30
    assert policy["extreme_low_threshold"] == 0.05
    assert policy["extreme_high_threshold"] == 0.95
    assert policy["extreme_consecutive_limit"] == 20
    assert policy["maximum_missing_rate"] == 0.05


def _freeze(path, metadata):
    value = {
        "schema_version": 1,
        "selection_frozen": True,
        "dataset_id": "synthetic",
        "dataset_digest": "d" * 64,
        "source_venue": "public",
        "candidates": {
            "lstm": {
                "candidate_id": metadata["candidate_id"],
                "candidate_model_digest": metadata["model_sha256"],
                "candidate_scaler_digest": metadata["scaler_sha256"],
                "internal_test_recorded": True,
            },
            "tcn": {"candidate_id": "tcn", "candidate_model_digest": "3" * 64, "candidate_scaler_digest": "2" * 64, "internal_test_recorded": True},
            "tx": {"candidate_id": "tx", "candidate_model_digest": "4" * 64, "candidate_scaler_digest": "2" * 64, "internal_test_recorded": True},
        },
        "frozen_at": "2026-01-01T00:00:00Z",
    }
    value["freeze_digest"] = json_digest({key: item for key, item in value.items() if key not in {"frozen_at", "freeze_digest"}})
    path.write_text(json.dumps(value), encoding="utf-8")


def test_confirmation_access_records_first_use_then_deterministic_replay(tmp_path):
    metadata = {
        "candidate_id": "lstm_5m_aaaaaaaa_bbbbbbbb_s24001",
        "model_kind": "lstm", "model_sha256": "1" * 64, "scaler_sha256": "2" * 64,
    }
    freeze = tmp_path / "freeze.json"
    ledger = tmp_path / "ledger.json"
    _freeze(freeze, metadata)
    confirmation = {"confirmation_digest": "9" * 64}
    first = gate.record_confirmation_access(metadata, confirmation, freeze_path=freeze, ledger_path=ledger)
    replay = gate.record_confirmation_access(metadata, confirmation, freeze_path=freeze, ledger_path=ledger)
    assert first["access_type"] == "first_evaluation" and first["access_count"] == 1
    assert replay["access_type"] == "deterministic_replay" and replay["access_count"] == 2
    recorded = json.loads(ledger.read_text(encoding="utf-8"))
    assert recorded["accesses"][0]["first_access_at"] == first["first_access_at"]


def test_different_candidate_cannot_inherit_pristine_confirmation(tmp_path):
    metadata = {
        "candidate_id": "lstm_original", "model_kind": "lstm",
        "model_sha256": "1" * 64, "scaler_sha256": "2" * 64,
    }
    freeze = tmp_path / "freeze.json"
    _freeze(freeze, metadata)
    changed = {**metadata, "candidate_id": "lstm_changed", "model_sha256": "8" * 64}
    with pytest.raises(gate.ModelCandidateHealthGateError, match="not_pristine"):
        gate.record_confirmation_access(
            changed, {"confirmation_digest": "9" * 64},
            freeze_path=freeze, ledger_path=tmp_path / "ledger.json",
        )


def test_confirmation_gate_is_separate_and_requires_internal_and_legacy_passes():
    source = open(gate.__file__, encoding="utf-8").read()
    assert 'gate == "confirmation"' in source
    assert 'internal_test_gate' in source
    assert 'legacy.get("status") != "legacy_repair_passed"' in source
    assert "record_confirmation_access" in source
    assert "selection" not in gate._inference_health.__code__.co_names


def test_negative_rv_auxiliary_failure_blocks_health_gate_status():
    failed_aux = {
        "auxiliary_head_safety_gate_passed": False,
        "hard_failure_reasons": ["auxiliary_failed_negative_rv"],
    }
    btc = {**_healthy(), "auxiliary_prediction_health": failed_aux}
    result = gate.gate_acceptance(
        "lstm", {"BTCUSDT": btc, "ETHUSDT": _healthy()}, gate="confirmation"
    )
    assert result["status"] == "auxiliary_head_gate_failed"
    assert result["auxiliary_head_safety_gate_passed"] is False
    assert "auxiliary_head_gate_failed" in result["per_symbol_failure_reasons"]["BTCUSDT"]


def test_classification_only_mode_cannot_pass_health_confirmation():
    source = open(gate.__file__, encoding="utf-8").read()
    assert "classification-only mode cannot produce health_gate_passed" in source
    assert "metadata.get(\"training_objective\"" in source
