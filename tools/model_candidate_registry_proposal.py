"""Generate a review-only Phase 24 registry proposal and final summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_candidate_train import (
    ALLOWED_KINDS,
    TRAINING_SUMMARY,
    _manifest_digest,
    _verify_candidate_artifacts_read_only,
    downstream_auxiliary_head_audit,
)
from tools.model_training_environment import (
    INCUMBENT_INVENTORY,
    atomic_write_json,
    file_digest,
    json_digest,
    utc_now,
    verify_incumbent_inventory,
)


TRACKED_REGISTRY = BASE_DIR / "research" / "model_candidate_registry.json"
PROPOSAL_PATH = BASE_DIR / "reports" / "model_candidate_registry_update.json"
ALLOWED_PROPOSAL_STATUSES = {
    "trained_unverified",
    "validation_failed",
    "internal_test_failed",
    "legacy_repair_failed",
    "confirmation_pending",
    "confirmation_health_failed",
    "confirmation_health_passed",
}


class ModelCandidateRegistryProposalError(ValueError):
    """A proposal attempted an unsupported state or had incomplete lineage."""


def candidate_proposal_record(candidate: Path | str) -> dict[str, Any]:
    root = Path(candidate)
    artifact_manifest = _verify_candidate_artifacts_read_only(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    training = json.loads((root / "training_manifest.json").read_text(encoding="utf-8-sig"))
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8-sig"))
    legacy = json.loads((root / "legacy_repair_gate.json").read_text(encoding="utf-8-sig"))
    confirmation = json.loads((root / "confirmation_health_gate.json").read_text(encoding="utf-8-sig"))
    status = str(metadata.get("candidate_status", "trained_unverified"))
    if status == "legacy_repair_passed":
        status = "confirmation_pending"
    if status not in ALLOWED_PROPOSAL_STATUSES:
        raise ModelCandidateRegistryProposalError(f"unsupported candidate proposal status: {status}")
    return {
        "candidate_id": metadata["candidate_id"], "model_kind": metadata["model_kind"],
        "proposed_status": status, "model_sha256": metadata["model_sha256"],
        "scaler_sha256": metadata["scaler_sha256"], "artifact_manifest": artifact_manifest,
        "dataset_id": training["dataset_id"], "dataset_digest": metadata["candidate_identity"]["dataset_digest"],
        "selected_seed": metadata["selected_seed"], "supported_symbols": metadata["supported_symbols"],
        "internal_test": evaluation["internal_test_gate"], "legacy_repair": legacy.get("status"),
        "sealed_confirmation": confirmation.get("status"),
        "candidate_auxiliary_head_promotion_blocker": metadata["candidate_auxiliary_head_promotion_blocker"],
        "eligible_for_later_shadow_comparison": status == "confirmation_health_passed",
        "human_review_required": True, "live_activation_allowed": False,
    }


def phase24_verdict(statuses: Sequence[str]) -> str:
    values = list(statuses)
    if not values or any(value in {"trained_unverified", "confirmation_pending"} for value in values):
        return "candidate_training_complete_confirmation_pending"
    passed = sum(value == "confirmation_health_passed" for value in values)
    if passed == len(values):
        return "candidate_training_complete_all_confirmation_gates_passed"
    if passed:
        return "candidate_training_complete_partial_confirmation_pass"
    return "candidate_training_complete_confirmation_gates_failed"


def generate_registry_proposal(
    candidates: Sequence[Path | str],
    *,
    proposal_path: Path | str = PROPOSAL_PATH,
    summary_path: Path | str = TRAINING_SUMMARY,
    tracked_registry: Path | str = TRACKED_REGISTRY,
) -> dict[str, Any]:
    registry = Path(tracked_registry)
    before = file_digest(registry)
    records = [candidate_proposal_record(candidate) for candidate in candidates]
    kinds = [record["model_kind"] for record in records]
    if len(set(kinds)) != len(kinds) or set(kinds) - set(ALLOWED_KINDS):
        raise ModelCandidateRegistryProposalError("candidate kinds must be unique Phase 24 model kinds")
    proposal: dict[str, Any] = {
        "schema_version": 1, "proposal_type": "human_review_only",
        "generated_at": utc_now(), "tracked_registry_sha256": before,
        "tracked_registry_modified": False, "candidates": records,
        "allowed_statuses": sorted(ALLOWED_PROPOSAL_STATUSES),
        "automatic_activation_implemented": False, "human_review_required": True,
    }
    proposal["proposal_digest"] = json_digest({key: value for key, value in proposal.items() if key not in {"generated_at", "proposal_digest"}})
    atomic_write_json(proposal_path, proposal)
    if file_digest(registry) != before:
        raise ModelCandidateRegistryProposalError("tracked registry changed while generating proposal")
    summary_file = Path(summary_path)
    summary: dict[str, Any] = {
        "schema_version": 1, "phase": 24, "models": {}, "environment": "pending",
        "dataset": "pending", "split": "pending", "scaler": "pending",
        "profitability_evidence": False,
    }
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8-sig"))
    for record in records:
        model = summary.setdefault("models", {}).setdefault(record["model_kind"], {})
        model.update({
            "candidate_id": record["candidate_id"], "candidate_artifacts": record["artifact_manifest"],
            "internal_test": record["internal_test"],
            "legacy_phase22_repair_result": record["legacy_repair"],
            "sealed_confirmation_result": record["sealed_confirmation"],
            "registry_proposal": record["proposed_status"], "status": record["proposed_status"],
        })
    summary["registry_proposal"] = {
        "path": Path(proposal_path).as_posix(), "digest": proposal["proposal_digest"],
        "tracked_registry_modified": False,
    }
    summary["auxiliary_head_objective_warning"] = downstream_auxiliary_head_audit()
    summary["artifact_integrity"] = {
        "tracked_registry_sha256": before, "tracked_registry_unchanged": file_digest(registry) == before,
        "candidate_artifacts_verified": True,
    }
    summary["final_decision"] = phase24_verdict([record["proposed_status"] for record in records])
    summary["decision_scope"] = "eligible_for_later_shadow_comparison_only"
    summary["summary_digest"] = _manifest_digest(summary, "summary_digest")
    atomic_write_json(summary_file, summary)
    if INCUMBENT_INVENTORY.is_file():
        verify_incumbent_inventory()
    return proposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", default=str(PROPOSAL_PATH))
    parser.add_argument("--summary", default=str(TRAINING_SUMMARY))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_registry_proposal(
            args.candidate, proposal_path=args.output, summary_path=args.summary
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except ModelCandidateRegistryProposalError as exc:
        print(json.dumps({"status": "registry_proposal_failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
