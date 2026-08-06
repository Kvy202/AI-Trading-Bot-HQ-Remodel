from __future__ import annotations

from tools.training_lineage_audit import REQUIRED_LINEAGE_FIELDS, audit_lineage_sources


def _metadata():
    return {
        "kind": "lstm", "seq_len": 64, "n_features": 27,
        "symbols": ["BTCUSDT", "ETHUSDT"], "timeframe": "5m",
        "label": {"type": "triple"}, "val_auc": 0.7,
    }


def test_basic_metadata_alone_is_not_complete_lineage():
    result = audit_lineage_sources("lstm", metadata=_metadata(), artifact_digests={
        "model_digest": "a" * 64, "scaler_digest": "b" * 64, "metadata_digest": "c" * 64,
    })
    assert result["lineage_status"] == "legacy_lineage_incomplete"
    assert result["inference_reproducibility"] == "artifact_identity_reproducible"
    assert result["training_reproducibility"] != "complete_reproducible"


def test_missing_raw_digest_split_boundaries_and_scaler_scope_are_reported():
    result = audit_lineage_sources("lstm", metadata=_metadata())
    for field in ("raw_data_digests", "train_split_boundaries", "validation_split_boundaries", "test_split_boundaries", "scaler_fit_split"):
        assert field in result["missing_fields"]


def test_current_environment_is_never_historical_evidence():
    result = audit_lineage_sources("lstm", metadata={**_metadata(), "package_versions": None})
    assert result["current_environment_used_as_historical_training_evidence"] is False
    assert result["lineage_fields"]["package_versions"] is None


def test_conflicting_lineage_is_detected():
    result = audit_lineage_sources(
        "lstm", metadata=_metadata(), manifests=[{"kind": "lstm", "timeframe": "1m"}]
    )
    assert result["lineage_status"] == "conflicting_lineage"
    assert "timeframe" in result["conflicting_fields"]


def test_complete_manifest_can_be_complete_only_with_every_field():
    manifest = {field: f"value-{field}" for field in REQUIRED_LINEAGE_FIELDS}
    manifest.update({"kind": "lstm", "sequence_length": 64})
    result = audit_lineage_sources("lstm", manifests=[manifest])
    assert result["lineage_status"] == "complete_reproducible"


def test_basic_metadata_auc_is_recorded_as_validation_evidence():
    result = audit_lineage_sources("lstm", metadata=_metadata())
    assert result["lineage_fields"]["validation_metrics"] == 0.7
    assert result["field_evidence"]["validation_metrics"]["sources"] == ["basic_metadata"]


def test_recorded_artifact_digest_conflict_is_not_silently_resolved():
    result = audit_lineage_sources(
        "lstm",
        manifests=[{"kind": "lstm", "model_digest": "a" * 64}],
        artifact_digests={"kind": "lstm", "model_digest": "b" * 64},
    )
    assert result["lineage_status"] == "conflicting_lineage"
    assert "model_digest" in result["conflicting_fields"]


def test_absent_artifacts_and_metadata_produce_missing_lineage():
    result = audit_lineage_sources("lstm")
    assert result["lineage_status"] == "missing"
    assert result["inference_reproducibility"] == "artifact_identity_incomplete"
