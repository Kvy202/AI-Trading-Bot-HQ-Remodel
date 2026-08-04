"""Synthetic tests for deterministic model-serving snapshots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ml_dl.dl_models import TemporalConvNet

from tools.model_serving_snapshot import (
    DOCUMENT_FIELDS,
    MODEL_ENTRY_FIELDS,
    ModelServingSnapshotError,
    capture_model_serving_snapshot,
    snapshot_digest,
    validate_model_serving_snapshot,
)
from tools.model_health_audit import audit_training_serving_contract


def _root(tmp_path: Path, *, unsafe: bool = False) -> Path:
    root = tmp_path / "repo"
    for directory in ("tools", "ml_dl", "config", "model_artifacts"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in (
        "tools/live_writer.py", "ml_dl/dl_ensemble.py", "ml_dl/dl_infer.py", "ml_dl/dl_models.py"
    ):
        (root / relative).write_text("# fixture\n", encoding="utf-8")
    (root / "features.py").write_text("FEATURE_COLS = ['a', 'b']\n", encoding="utf-8")
    config = {
        "universe": {"DL_SYMBOLS": "BTCUSDT", "DL_TIMEFRAME": "1m", "DL_SEQ_LEN": 8},
        "signal": {"DL_MIN_AGREE": 2},
    }
    (root / "config" / "run.json").write_text(json.dumps(config), encoding="utf-8")
    for kind, auc in (("lstm", 0.6), ("tcn", 0.3), ("tx", 0.6)):
        (root / "model_artifacts" / f"dl_{kind}_metadata.json").write_text(
            json.dumps({"kind": kind, "seq_len": 8, "n_features": 2, "timeframe": "5m",
                        "symbols": ["BTCUSDT", "ETHUSDT"], "val_auc": auc,
                        "trained_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
        )
    (root / ".env").write_text(
        "\n".join([
            f"LIVE_TRADING={'true' if unsafe else 'false'}",
            f"PAPER_TRADING={'false' if unsafe else 'true'}",
            "EXEC_PAPER=true", "LIVE_MODE=false", "PLACE_REAL_ORDERS=false",
            "API_KEY=do-not-leak", "HL_AGENT_PRIVATE_KEY=0x" + "a" * 64,
        ]), encoding="utf-8"
    )
    return root


def test_snapshot_is_allowlisted_and_does_not_expose_secrets(tmp_path):
    snapshot = capture_model_serving_snapshot(base_dir=_root(tmp_path))

    assert set(snapshot) == DOCUMENT_FIELDS
    assert all(set(entry) == MODEL_ENTRY_FIELDS for entry in snapshot["model_entries"])
    encoded = json.dumps(snapshot)
    assert "do-not-leak" not in encoded
    assert "HL_AGENT_PRIVATE_KEY" not in encoded
    assert "a" * 64 not in encoded


def test_snapshot_digest_is_deterministic_and_ignores_generated_at_and_root(tmp_path):
    first_root = _root(tmp_path / "one")
    second_root = tmp_path / "two" / "repo"
    shutil.copytree(first_root, second_root)
    first = capture_model_serving_snapshot(base_dir=first_root, generated_at="2026-01-01T00:00:00Z")
    second = capture_model_serving_snapshot(base_dir=second_root, generated_at="2026-01-02T00:00:00Z")

    assert snapshot_digest(first) == snapshot_digest(second)
    assert first["snapshot_digest"] == second["snapshot_digest"]
    assert not any(Path(entry["model_filename"]).is_absolute() for entry in first["model_entries"])


def test_auc_weights_resolve_when_explicit_weights_are_absent(tmp_path):
    snapshot = capture_model_serving_snapshot(base_dir=_root(tmp_path))

    weights = snapshot["dl_model_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["lstm"] == pytest.approx(weights["tx"])
    assert weights["tcn"] < weights["lstm"]


def test_explicit_weights_are_recorded_without_rewriting(tmp_path):
    root = _root(tmp_path)
    with (root / ".env").open("a", encoding="utf-8") as handle:
        handle.write("\nDL_MODEL_WEIGHTS=lstm:0.7,tcn:0,tx:0.3\n")

    snapshot = capture_model_serving_snapshot(base_dir=root)

    assert snapshot["dl_model_weights"] == {"lstm": 0.7, "tcn": 0.0, "tx": 0.3}


def test_contract_audit_detects_timeframe_sequence_and_feature_mismatches(tmp_path):
    snapshot = capture_model_serving_snapshot(base_dir=_root(tmp_path))
    snapshot["dl_seq_len"] = 9
    snapshot["dl_add_symbol_id"] = True

    result = audit_training_serving_contract(snapshot)

    for model in result["models"].values():
        assert model["comparisons"]["timeframe"] == "mismatch"
        assert model["comparisons"]["seq_len"] == "mismatch"
        assert model["comparisons"]["serving_feature_count_vs_scaler_width"] in {
            "mismatch", "unverified_runtime_value"
        }


def test_unsafe_real_order_or_nonpaper_snapshot_is_rejected(tmp_path):
    with pytest.raises(ModelServingSnapshotError):
        capture_model_serving_snapshot(base_dir=_root(tmp_path, unsafe=True))


def test_snapshot_validation_detects_digest_tampering(tmp_path):
    snapshot = capture_model_serving_snapshot(base_dir=_root(tmp_path))
    snapshot["dl_timeframe"] = "15m"

    with pytest.raises(ModelServingSnapshotError, match="digest mismatch"):
        validate_model_serving_snapshot(snapshot)


def test_artifacts_are_loaded_read_only(tmp_path):
    root = _root(tmp_path)
    scaler_path = root / "model_artifacts" / "scaler_tcn_latest.joblib"
    model_path = root / "model_artifacts" / "dl_tcn_latest.pt"
    joblib.dump(StandardScaler().fit(np.asarray([[0.0, 1.0], [1.0, 2.0]])), scaler_path)
    torch.save(TemporalConvNet(2).state_dict(), model_path)
    before = (scaler_path.read_bytes(), model_path.read_bytes())

    snapshot = capture_model_serving_snapshot(base_dir=root)

    tcn = next(entry for entry in snapshot["model_entries"] if entry["kind"] == "tcn")
    assert tcn["model_load_status"] == "loaded"
    assert tcn["model_parameter_count"] > 0
    assert (scaler_path.read_bytes(), model_path.read_bytes()) == before
