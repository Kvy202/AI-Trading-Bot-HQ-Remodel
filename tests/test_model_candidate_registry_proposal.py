from __future__ import annotations

import json

import pytest

from tools import model_candidate_registry_proposal as registry


def _candidate(tmp_path, kind="lstm", status="confirmation_pending"):
    root = tmp_path / f"{kind}_candidate"
    root.mkdir()
    metadata = {
        "candidate_id": root.name,
        "model_kind": kind,
        "model_sha256": "1" * 64,
        "scaler_sha256": "2" * 64,
        "selected_seed": 24001,
        "supported_symbols": ["BTCUSDT", "ETHUSDT"],
        "candidate_status": status,
        "candidate_auxiliary_head_promotion_blocker": True,
        "candidate_identity": {"dataset_digest": "3" * 64},
    }
    files = {
        "metadata.json": metadata,
        "training_manifest.json": {"dataset_id": "synthetic"},
        "evaluation.json": {"internal_test_gate": {"passed": True}},
        "legacy_repair_gate.json": {"status": "legacy_repair_passed"},
        "confirmation_health_gate.json": {"status": status},
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    return root


def test_allowed_proposal_statuses_exclude_activation_states():
    assert "promoted" not in registry.ALLOWED_PROPOSAL_STATUSES
    assert "active" not in registry.ALLOWED_PROPOSAL_STATUSES
    assert "production" not in registry.ALLOWED_PROPOSAL_STATUSES
    assert "confirmation_health_passed" in registry.ALLOWED_PROPOSAL_STATUSES


def test_candidate_record_maps_legacy_pass_to_confirmation_pending(monkeypatch, tmp_path):
    candidate = _candidate(tmp_path, status="legacy_repair_passed")
    monkeypatch.setattr(registry, "_verify_candidate_artifacts_read_only", lambda root: {"verified": True})
    record = registry.candidate_proposal_record(candidate)
    assert record["proposed_status"] == "confirmation_pending"
    assert record["human_review_required"] is True
    assert record["live_activation_allowed"] is False


def test_candidate_record_rejects_unapproved_status(monkeypatch, tmp_path):
    candidate = _candidate(tmp_path, status="active")
    monkeypatch.setattr(registry, "_verify_candidate_artifacts_read_only", lambda root: {})
    with pytest.raises(registry.ModelCandidateRegistryProposalError, match="unsupported"):
        registry.candidate_proposal_record(candidate)


def test_registry_proposal_does_not_modify_tracked_registry(monkeypatch, tmp_path):
    candidates = [
        _candidate(tmp_path, "lstm", "confirmation_health_passed"),
        _candidate(tmp_path, "tcn", "confirmation_health_failed"),
        _candidate(tmp_path, "tx", "confirmation_pending"),
    ]
    monkeypatch.setattr(registry, "_verify_candidate_artifacts_read_only", lambda root: {"candidate_id": root.name})
    tracked = tmp_path / "registry.json"
    tracked.write_text('{"schema_version":1,"candidates":{}}\n', encoding="utf-8")
    before = tracked.read_bytes()
    output = tmp_path / "proposal.json"
    summary = tmp_path / "summary.json"
    proposal = registry.generate_registry_proposal(
        candidates, proposal_path=output, summary_path=summary, tracked_registry=tracked
    )
    assert tracked.read_bytes() == before
    assert proposal["tracked_registry_modified"] is False
    assert proposal["proposal_type"] == "human_review_only"
    assert {row["proposed_status"] for row in proposal["candidates"]} == {
        "confirmation_health_passed", "confirmation_health_failed", "confirmation_pending"
    }
    final = json.loads(summary.read_text(encoding="utf-8"))
    assert final["final_decision"] == "candidate_training_complete_confirmation_pending"
    assert final["profitability_evidence"] is False


@pytest.mark.parametrize(
    ("statuses", "verdict"),
    [
        (["confirmation_health_passed"] * 3, "candidate_training_complete_all_confirmation_gates_passed"),
        (["confirmation_health_passed", "confirmation_health_failed"], "candidate_training_complete_partial_confirmation_pass"),
        (["confirmation_health_failed"] * 3, "candidate_training_complete_confirmation_gates_failed"),
        (["confirmation_pending"] * 3, "candidate_training_complete_confirmation_pending"),
    ],
)
def test_phase24_verdicts(statuses, verdict):
    assert registry.phase24_verdict(statuses) == verdict
