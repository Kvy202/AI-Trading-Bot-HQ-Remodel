"""Synthetic tests for historical output and ensemble-variant diagnostics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.model_health_audit import (
    ModelHealthAuditError,
    analyze_ensemble_variants,
    analyze_historical_probabilities,
    audit_training_serving_contract,
    parse_writer_diagnostics,
    probability_statistics,
    resolve_historical_model_outputs,
)
from tools.model_health_probe import load_policy


POLICY = load_policy()


def _rows(values, *, symbol="BTCUSDT", kind="tcn"):
    return [{"ts": f"2026-08-04T16:{index // 60:02d}:{index % 60:02d}Z", "symbol": symbol,
             f"{kind}_p": "" if value is None else value}
            for index, value in enumerate(values)]


def _analyze(values, *, kind="tcn"):
    return analyze_historical_probabilities(
        {"live_models_by_symbol.csv": _rows(values, kind=kind)}, [kind], POLICY
    )["models"][kind]


def _csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ts", "symbol", "tcn_p"])
        writer.writerows(rows)


def test_flat_tcn_fails_only_at_sufficient_sample_size():
    assert "failed_flat_output" in _analyze([0.43] * 100)["statuses"]
    warning = _analyze([0.43] * 30)["statuses"]
    assert "warning_insufficient_rows" in warning
    assert "failed_flat_output" not in warning


def test_one_sided_but_varying_output_is_warning_not_flat_failure():
    result = _analyze(np.linspace(0.55, 0.9, 100))

    assert "warning_one_sided" in result["statuses"]
    assert "failed_flat_output" not in result["statuses"]


def test_missing_values_above_policy_limit_fail_with_sufficient_rows():
    values = list(np.linspace(0.1, 0.9, 90)) + [None] * 10
    result = _analyze(values)

    assert result["model_missing_rate"] == pytest.approx(0.1)
    assert "failed_missing_output" in result["statuses"]


def test_percentiles_and_lag_statistics_are_deterministic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    timestamps = [None] * len(values)

    first = probability_statistics(values, timestamps)
    second = probability_statistics(values, timestamps)

    assert first == second
    assert first["median"] == pytest.approx(0.3)
    assert first["p95"] == pytest.approx(0.48)


def test_auto_exclusion_and_exact_diagnostic_lines_are_counted(tmp_path):
    path = tmp_path / "xgboost_shadow_outcome_paper_writer.err"
    path.write_text(
        "[2026-08-04 21:35:26,154] WARNING auto-exclude tcn[BTCUSDT]: "
        "p_long std=0.0001 < 0.002 over 30 ticks (flat output)\n"
        "[2026-08-04 21:35:27,154] WARNING predict failed for tx: scaler feature dim mismatch\n",
        encoding="utf-8",
    )

    result = parse_writer_diagnostics([path])

    assert result["auto_exclusion_count"] == 1
    assert result["auto_exclusion_reason_counts"] == {"flat_output": 1}
    assert result["diagnostic_reason_counts"]["predict_failed"] == 1
    assert result["diagnostic_reason_counts"]["scaler"] == 1


def test_per_symbol_and_pooled_statistics_remain_separate():
    analysis = analyze_historical_probabilities(
        {"live_models_by_symbol.csv": _rows([0.1, 0.2], symbol="BTC")
         + _rows([0.8, 0.9], symbol="ETH")}, ["tcn"], POLICY
    )["models"]["tcn"]

    assert analysis["by_symbol"]["BTC"]["mean"] == pytest.approx(0.15)
    assert analysis["by_symbol"]["ETH"]["mean"] == pytest.approx(0.85)
    assert analysis["pooled_standard_deviation"] > analysis["within_symbol_standard_deviation"]


def test_latest_rows_filter_and_exact_duplicate_deduplication(tmp_path):
    log = tmp_path / "logs" / "live_models_by_symbol.csv"
    rows = [[f"2026-08-04T16:00:0{i}Z", "BTC", 0.1 + i / 10] for i in range(5)]
    rows.append(rows[-1])
    _csv(log, rows)

    result = resolve_historical_model_outputs(
        None, None, None, logs_dir=log.parent, bundle_root=tmp_path / "bundles", rows_limit=2
    )

    retained = result["rows_by_file"]["live_models_by_symbol.csv"]
    assert len(retained) == 2
    assert retained[-1]["tcn_p"] == "0.5"
    assert result["duplicate_count"] == 1


def test_conflicting_timestamp_symbol_rows_fail(tmp_path):
    log = tmp_path / "logs" / "live_models_by_symbol.csv"
    _csv(log, [["2026-08-04T16:00:00Z", "BTC", 0.4],
               ["2026-08-04T16:00:00Z", "BTC", 0.6]])

    with pytest.raises(ModelHealthAuditError, match="conflicting"):
        resolve_historical_model_outputs(
            None, None, None, logs_dir=log.parent, bundle_root=tmp_path / "bundles"
        )


def _snapshot(weights=None):
    entries = [
        {"kind": kind, "metadata_val_auc": auc}
        for kind, auc in (("lstm", 0.7), ("tcn", 0.55), ("tx", 0.68))
    ]
    return {
        "model_entries": entries,
        "dl_model_weights": weights or {"lstm": 0.5, "tcn": 0.0, "tx": 0.5},
        "dl_min_agree": 2,
    }


def test_zero_weight_model_does_not_vote_and_suppression_is_neutral():
    rows = [{"ts": "2026-08-04T16:00:00Z", "symbol": "BTC",
             "lstm_p": 0.7, "tcn_p": 0.99, "tx_p": 0.3}]

    result = analyze_ensemble_variants(rows, _snapshot(), threshold=0.08)

    current = result["current_config"]
    assert current["agreement_suppressed_count"] == 1
    assert current["centered_mean"] == 0.0
    assert current["changed_allow_vs_current"] == 0


def test_lstm_tx_and_no_tcn_are_shadow_candidates_and_exclude_tcn_effect():
    rows = [{"ts": "2026-08-04T16:00:00Z", "symbol": "BTC",
             "lstm_p": 0.1, "tcn_p": 0.99, "tx_p": 0.1}]
    before = dict(os.environ)

    result = analyze_ensemble_variants(
        rows, _snapshot({"lstm": 0.2, "tcn": 0.6, "tx": 0.2}), threshold=0.08
    )

    assert result["lstm_tx_only"]["short_count"] == 1
    assert result["no_tcn"]["short_count"] == 1
    assert result["lstm_tx_only"]["shadow_configuration_candidate_only"] is True
    assert os.environ == before
    encoded = json.dumps(result).lower()
    assert "pnl" not in encoded
    assert "promotion" not in encoded


def test_loadable_sklearn_mismatch_is_a_reproducibility_warning_only():
    snapshot = {
        "dl_timeframe": "5m", "dl_seq_len": 64, "feature_count": 26,
        "dl_add_symbol_id": True, "dl_symbols": ["BTCUSDT"],
        "model_entries": [{
            "kind": "lstm", "metadata_kind": "lstm", "metadata_timeframe": "5m",
            "metadata_seq_len": 64, "metadata_n_features": 27,
            "metadata_symbols": ["BTCUSDT", "ETHUSDT"], "metadata_val_auc": 0.65,
            "scaler_n_features_in": 27, "scaler_mean_finite": True,
            "scaler_scale_finite": True, "scaler_feature_names": None,
            "scaler_filename": "scaler_lstm_latest.joblib",
            "model_load_status": "loaded",
            "sklearn_version_status": "loadable_version_mismatch",
        }],
    }
    result = audit_training_serving_contract(snapshot)
    model = result["models"]["lstm"]
    assert model["critical_mismatches"] == []
    assert "loadable_sklearn_version_mismatch_reproducibility_warning" in model["warnings"]
