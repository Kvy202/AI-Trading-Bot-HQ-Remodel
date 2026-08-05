from __future__ import annotations

import json
from copy import deepcopy

import pytest

import tools.model_alignment_report as report
from tools.model_alignment_shadow import load_alignment_policy


def _entry(kind, sklearn_status="exact_match"):
    return {
        "kind": kind, "metadata_status": "loaded", "metadata_kind": kind,
        "metadata_timeframe": "5m", "metadata_seq_len": 64,
        "metadata_n_features": 27, "metadata_symbols": ["BTCUSDT"],
        "metadata_val_auc": 0.65, "scaler_n_features_in": 27,
        "scaler_load_status": "loaded", "model_load_status": "loaded",
        "sklearn_version_status": sklearn_status,
        "scaler_serialized_sklearn_version": "1.8.0",
        "scaler_runtime_sklearn_version": "1.7.1" if sklearn_status != "exact_match" else "1.8.0",
        "scaler_serialization_warnings": [],
    }


def _snapshot(status="exact_match"):
    return {
        "dl_timeframe": "5m", "dl_seq_len": 64, "feature_count": 26,
        "dl_add_symbol_id": True, "dl_symbols": ["BTCUSDT"],
        "dl_model_weights": {"lstm": 0.5, "tx": 0.5},
        "model_entries": [_entry("lstm", status), _entry("tx", status)],
        "snapshot_digest": "a" * 64, "python_version": "3.13.0",
        "numpy_version": "2.0.0", "torch_version": "2.0.0",
        "joblib_version": "1.5.0", "sklearn_runtime_version": "1.7.1",
    }


def _stats():
    return {
        "unique_completed_bar_count": 100, "missing_rate": 0.0,
        "deterministic_repeat_max_error": 0.0, "5m_historical_std": 0.01,
        "simulated_exclusion_count": 0, "model_health_status": "healthy_aligned",
    }


def test_library_mismatch_is_reported_without_dependency_change():
    value = report.library_compatibility(_snapshot("loadable_version_mismatch"))
    assert value["exact_reproducibility"] is False
    assert value["dependencies_changed"] is False
    assert "expected scikit-learn 1.8.0" in value["runtime_remediation_required"][0]
    assert "current scikit-learn 1.7.1" in value["runtime_remediation_required"][0]


def test_exact_sklearn_versions_are_reproducible():
    assert report.library_compatibility(_snapshot())["exact_reproducibility"] is True


def test_runtime_pin_pending_readiness(monkeypatch, tmp_path):
    snapshot = _snapshot("loadable_version_mismatch")
    manifest = {
        "symbols": ["BTCUSDT"], "bundle_digest": "b" * 64,
        "conflicting_source_bar_count": 0, "unique_completed_bars_by_symbol": {"BTCUSDT": 100},
    }
    monkeypatch.setattr(report, "load_historical_bundle", lambda path: (manifest, snapshot, []))
    evaluation = {
        "minimum_statistical_gate_passed": True,
        "model_results": {"BTCUSDT": {"lstm": _stats(), "tx": _stats()}},
        "ensemble_variants": {"BTCUSDT": {}},
    }
    live = {
        "status": "pass", "unique_completed_bars_by_symbol": {"BTCUSDT": 3},
        "writer_started": False, "executor_started": False, "orders_placed": 0,
        "timeframe": "5m", "sequence_length": 64, "served_feature_width": 27,
        "contract_status": "pass", "all_source_bars_completed": True,
        "all_source_bar_ids_unique": True, "conflicting_source_bar_count": 0,
        "nonfinite_output_count": 0, "nondeterministic_output_count": 0,
        "snapshot_digest": "a" * 64,
    }
    eval_path, live_path = tmp_path / "eval.json", tmp_path / "live.json"
    eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
    live_path.write_text(json.dumps(live), encoding="utf-8")
    result = report.build_alignment_report(
        historical_bundle="bundle", historical_evaluation=eval_path,
        live_campaign=live_path, policy=load_alignment_policy(),
    )
    assert result["overall_decision"]["verdict"] == "alignment_validated_runtime_pin_pending"
    assert result["overall_decision"]["campaign_readiness"] == "ready_for_5m_shadow_campaign_runtime_pin_pending"
    assert result["policy"]["orders_allowed"] is False
    assert result["historical_alignment"]["profitability_evidence"] is False
    assert len(result["alignment_digest"]) == 64


def test_lstm_collapse_with_positive_weight_blocks_readiness(monkeypatch, tmp_path):
    snapshot = _snapshot()
    manifest = {
        "symbols": ["BTCUSDT"], "bundle_digest": "b" * 64,
        "conflicting_source_bar_count": 0, "unique_completed_bars_by_symbol": {"BTCUSDT": 100},
    }
    monkeypatch.setattr(report, "load_historical_bundle", lambda path: (manifest, snapshot, []))
    collapsed = {**_stats(), "model_health_status": "failed_flat_at_5m", "5m_historical_std": 0.0}
    evaluation = {
        "minimum_statistical_gate_passed": True,
        "model_results": {"BTCUSDT": {"lstm": collapsed, "tx": _stats()}},
        "ensemble_variants": {},
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    result = report.build_alignment_report(
        historical_bundle="bundle", historical_evaluation=path,
        policy=load_alignment_policy(),
    )
    assert result["overall_decision"]["verdict"] == "alignment_failed_model_collapse"
    assert result["overall_decision"]["campaign_readiness"] == "blocked_model_collapse"


def test_report_does_not_mutate_inputs():
    snapshot = _snapshot("loadable_version_mismatch")
    before = deepcopy(snapshot)
    report.library_compatibility(snapshot)
    assert snapshot == before
