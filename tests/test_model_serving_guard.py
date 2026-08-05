from __future__ import annotations

from copy import deepcopy

from runtime.model_serving_guard import evaluate_model_serving_contract


def _entries():
    return [
        {
            "kind": kind,
            "metadata_status": "loaded",
            "metadata_kind": kind,
            "metadata_timeframe": "5m",
            "metadata_seq_len": 64,
            "metadata_n_features": 27,
            "metadata_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "metadata_val_auc": 0.65,
            "scaler_n_features_in": 27,
            "scaler_load_status": "loaded",
            "model_load_status": "loaded",
            "sklearn_version_status": "exact_match",
        }
        for kind in ("lstm", "tcn", "tx")
    ]


def _guard(entries=None, **overrides):
    values = {
        "serving_timeframe": "5m",
        "serving_sequence_length": 64,
        "generated_feature_width": 27,
        "add_symbol_id": True,
        "serving_symbols": ["BTCUSDT", "ETHUSDT"],
        "base_feature_width": 26,
    }
    values.update(overrides)
    return evaluate_model_serving_contract(_entries() if entries is None else entries, **values)


def test_aligned_contract_and_symbol_subset_pass():
    result = _guard()
    assert result["status"] == "pass"
    assert result["critical_mismatches"] == []


def test_serving_timeframe_mismatch_fails():
    result = _guard(serving_timeframe="1m")
    assert result["status"] == "fail"
    assert "serving timeframe 1m != training timeframe 5m" in result["critical_mismatches"]


def test_sequence_mismatch_fails():
    assert _guard(serving_sequence_length=128)["status"] == "fail"


def test_feature_width_mismatch_fails():
    assert _guard(generated_feature_width=26)["status"] == "fail"


def test_symbol_id_setting_mismatch_fails():
    assert _guard(add_symbol_id=False)["status"] == "fail"


def test_unknown_serving_symbol_fails():
    result = _guard(serving_symbols=["BTCUSDT", "UNKNOWNUSDT"])
    assert result["status"] == "fail"
    assert any("unknown serving symbols" in item for item in result["critical_mismatches"])


def test_missing_metadata_fails():
    entries = _entries()
    entries[0]["metadata_status"] = "missing"
    entries[0]["metadata_timeframe"] = None
    assert _guard(entries)["status"] == "fail"


def test_low_auc_and_sklearn_mismatch_are_warnings_only():
    entries = _entries()
    entries[1]["metadata_val_auc"] = 0.4
    entries[1]["sklearn_version_status"] = "loadable_version_mismatch"
    result = _guard(entries)
    assert result["status"] == "pass"
    assert any("validation AUC" in item for item in result["warnings"])
    assert any("scikit-learn" in item for item in result["warnings"])


def test_model_or_scaler_load_failure_is_critical():
    entries = _entries()
    entries[0]["model_load_status"] = "load_failed"
    entries[1]["scaler_load_status"] = "load_failed"
    assert _guard(entries)["status"] == "fail"


def test_loaded_models_must_agree():
    entries = _entries()
    entries[2]["metadata_timeframe"] = "1m"
    entries[1]["metadata_seq_len"] = 32
    assert _guard(entries)["status"] == "fail"


def test_guard_digest_is_deterministic_and_input_is_not_mutated():
    entries = _entries()
    before = deepcopy(entries)
    assert _guard(entries)["guard_digest"] == _guard(deepcopy(entries))["guard_digest"]
    assert entries == before


def test_no_models_is_unverified():
    assert _guard([])["status"] == "unverified"


def test_missing_symbol_id_setting_is_critical():
    assert _guard(add_symbol_id=None)["status"] == "fail"


def test_ordered_training_symbol_mismatch_is_critical_for_persisted_ids():
    entries = _entries()
    entries[1]["metadata_symbols"] = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
    result = _guard(entries)
    assert result["status"] == "fail"
    assert "loaded models disagree on ordered training symbols" in result["critical_mismatches"]


def test_writer_guard_precedes_market_refresh_and_has_no_bypass():
    source = open("tools/live_writer.py", encoding="utf-8").read()
    assert "load_ensemble(X_dim=30, device=None, require_all=True)" in source
    guard_call = source.index("contract = run_model_contract_guard(")
    refresh_call = source.index("meta, windows = refresh_live_features_per_symbol(")
    assert guard_call < refresh_call
    failure = source[source.index('if contract["status"] != "pass"'):refresh_call]
    assert "writer_unlock()" in failure and "sys.exit(1)" in failure
    assert "append_aligned_row(signals_path" not in failure
