from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import ml_dl.dl_ensemble as ensemble
import tools.model_alignment_shadow as alignment
from features import canonical_feature_columns
from tools.model_alignment_shadow import (
    ModelAlignmentError,
    SimulatedHealthState,
    alignment_bundle_digest,
    calibrate_probability,
    calculate_model_statistics,
    capture_historical_bundle,
    deduplicate_evidence_rows,
    evaluate_ensemble_variants,
    load_alignment_policy,
)


@pytest.fixture
def policy():
    return load_alignment_policy()


def _frame(rows=65):
    columns = canonical_feature_columns(True)
    index = pd.date_range("2026-01-01T04:40:00Z", periods=rows, freq="5min")
    values = np.arange(rows * len(columns), dtype=np.float32).reshape(rows, len(columns)) / 100
    frame = pd.DataFrame(values, index=index, columns=columns)
    frame["close"] = np.linspace(100, 110, rows)
    return frame


def test_completed_only_uses_previous_bar_and_adds_all_provenance(monkeypatch):
    frame = _frame()
    matrix = frame[canonical_feature_columns(True)].to_numpy(dtype=np.float32)

    def loader(**kwargs):
        return matrix, {"BTCUSDT": frame}, [len(frame)]

    monkeypatch.setattr(ensemble, "load_prices_and_features", loader)
    meta, windows = ensemble.refresh_live_features_per_symbol(
        64, True, symbols=["BTCUSDT"], timeframe="5m", completed_only=True,
        as_of_utc="2026-01-01T10:02:00Z", completion_grace_seconds=5,
    )
    assert windows["BTCUSDT"].shape == (64, 27)
    assert meta["source_bar_open_utc_by_symbol"]["BTCUSDT"] == "2026-01-01T09:55:00Z"
    assert meta["source_bar_close_utc_by_symbol"]["BTCUSDT"] == "2026-01-01T10:00:00Z"
    assert meta["source_bar_completed_by_symbol"]["BTCUSDT"] is True
    assert meta["last_px"]["BTCUSDT"] == pytest.approx(frame.iloc[-2]["close"])
    required = {
        "source_bar_id_by_symbol", "feature_window_digest_by_symbol",
        "feature_window_row_count_by_symbol", "feature_window_first_utc_by_symbol",
        "feature_window_last_utc_by_symbol",
    }
    assert required <= set(meta)


def test_default_refresh_window_is_backward_compatible(monkeypatch):
    frame = _frame()
    matrix = frame[canonical_feature_columns(True)].to_numpy(dtype=np.float32)
    monkeypatch.setattr(
        ensemble, "load_prices_and_features",
        lambda **kwargs: (matrix, {"BTCUSDT": frame}, [len(frame)]),
    )
    _meta, windows = ensemble.refresh_live_features_per_symbol(
        64, True, symbols=["BTCUSDT"], timeframe="5m"
    )
    np.testing.assert_array_equal(windows["BTCUSDT"], matrix[-64:])


def test_completed_only_keeps_multi_symbol_blocks_separate_with_extra_rows(monkeypatch):
    columns = canonical_feature_columns(True)
    index = pd.date_range("2026-01-01T00:00:00Z", periods=80, freq="5min")
    btc = pd.DataFrame(np.ones((80, 27), dtype=np.float32), index=index, columns=columns)
    eth = pd.DataFrame(np.full((80, 27), 2.0, dtype=np.float32), index=index, columns=columns)
    btc["close"], eth["close"] = 100.0, 200.0
    matrix = np.concatenate((btc[columns].to_numpy(), eth[columns].to_numpy()))
    monkeypatch.setattr(
        ensemble, "load_prices_and_features",
        lambda **kwargs: (matrix, {"BTCUSDT": btc, "ETHUSDT": eth}, [80, 80]),
    )
    _meta, windows = ensemble.refresh_live_features_per_symbol(
        64, True, symbols=["BTCUSDT", "ETHUSDT"], timeframe="5m",
        completed_only=True, as_of_utc="2026-01-01T06:40:05Z",
    )
    assert np.all(windows["BTCUSDT"] == 1.0)
    assert np.all(windows["ETHUSDT"] == 2.0)


def test_completed_bar_mask_applies_duration_and_grace():
    opens = pd.DatetimeIndex(["2026-01-01T10:00:00Z"])
    assert not ensemble.completed_bar_mask(
        opens, "5m", as_of_utc="2026-01-01T10:05:04Z", completion_grace_seconds=5
    )[0]
    assert ensemble.completed_bar_mask(
        opens, "5m", as_of_utc="2026-01-01T10:05:05Z", completion_grace_seconds=5
    )[0]


def test_source_id_and_window_digest_are_deterministic_and_sensitive():
    columns = ["a", "b"]
    times = pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC")
    values = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    first = ensemble.feature_window_digest("BTCUSDT", "5m", columns, times, values)
    assert first == ensemble.feature_window_digest("BTCUSDT", "5m", columns, times, values.copy())
    changed = values.copy()
    changed[1, 1] += np.float32(0.01)
    assert first != ensemble.feature_window_digest("BTCUSDT", "5m", columns, times, changed)
    assert ensemble.source_bar_id("BTC/USDT", times[-1]) == "BTCUSDT:2026-01-01T00:05:00Z"


def test_digest_uses_little_endian_not_native_representation():
    columns = ["a", "b"]
    times = ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00+00:00"]
    little = np.asarray([[1, 2], [3, 4]], dtype="<f4")
    big = np.asarray([[1, 2], [3, 4]], dtype=">f4")
    assert ensemble.feature_window_digest("BTC", "5m", columns, times, little) == ensemble.feature_window_digest(
        "BTC", "5m", columns, times, big
    )


def test_exact_duplicate_is_deduped_and_conflict_fails():
    row = {"source_bar_id": "BTC:1", "feature_window_digest": "a" * 64}
    retained, duplicates = deduplicate_evidence_rows([row, dict(row)])
    assert len(retained) == 1 and duplicates == 1
    with pytest.raises(ModelAlignmentError, match="conflicting source bar"):
        deduplicate_evidence_rows([row, {**row, "feature_window_digest": "b" * 64}])


def test_alignment_policy_exact_fields_and_types(tmp_path, policy):
    broken = dict(policy)
    broken["extra"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ModelAlignmentError):
        load_alignment_policy(path)
    broken = dict(policy)
    broken["flat_output_window"] = 30.0
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ModelAlignmentError):
        load_alignment_policy(path)


def test_bundle_digest_is_deterministic():
    manifest = {"schema_version": 1, "symbols": ["BTC"], "bundle_digest": None}
    assert alignment_bundle_digest(manifest) == alignment_bundle_digest(deepcopy(manifest))


def _minimal_snapshot():
    entries = []
    for kind in ("lstm", "tcn", "tx"):
        entries.append({
            "kind": kind, "metadata_status": "loaded", "metadata_kind": kind,
            "metadata_timeframe": "5m", "metadata_seq_len": 64,
            "metadata_n_features": 27, "metadata_symbols": ["BTCUSDT", "ETHUSDT"],
            "metadata_val_auc": 0.6, "scaler_n_features_in": 27,
            "scaler_load_status": "loaded", "model_load_status": "loaded",
        })
    return {
        "dl_timeframe": "5m", "dl_seq_len": 64, "feature_count": 26,
        "dl_add_symbol_id": True, "dl_symbols": ["BTCUSDT"],
        "model_entries": entries, "snapshot_digest": "f" * 64,
        "market_data_exchange": "fixture",
    }


def test_historical_capture_is_completed_ordered_and_self_contained(tmp_path, policy):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=260, freq="5min")
    close = 100 + np.linspace(0, 10, len(index)) + np.sin(np.arange(len(index)))
    bars = pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000 + np.arange(len(index)),
    }, index=index)
    out = tmp_path / "bundle"
    manifest = capture_historical_bundle(
        bundle_out=out, symbols=["BTCUSDT"], timeframe="5m", unique_bars=100,
        lookback_bars=260, policy=policy, as_of_utc="2026-01-01T21:40:05Z",
        fetcher=lambda *args, **kwargs: bars, snapshot=_minimal_snapshot(),
    )
    assert manifest["unique_completed_bars_by_symbol"] == {"BTCUSDT": 100}
    assert manifest["feature_count"] == 27 and manifest["add_symbol_id"] is True
    assert manifest["conflicting_source_bar_count"] == 0
    source = pd.read_csv(out / "source_bars_BTCUSDT.csv")
    assert source["bar_open_utc"].is_monotonic_increasing
    features = pd.read_csv(out / "features_BTCUSDT.csv")
    assert "symbol_id" in features and set(features["symbol_id"]) == {0.0}
    encoded = "".join(path.read_text(encoding="utf-8") for path in out.iterdir() if path.suffix in {".json", ".log", ".jsonl"})
    assert "api_key" not in encoded.lower() and "private_key" not in encoded.lower()


def test_historical_capture_rejects_excessive_gap(tmp_path, policy):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=260, freq="5min").delete([100, 101, 102])
    close = 100 + np.arange(len(index)) * 0.01
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.ones(len(index)) * 100,
    }, index=index)
    with pytest.raises(ModelAlignmentError, match="excessive historical gap"):
        capture_historical_bundle(
            bundle_out=tmp_path / "bundle", symbols=["BTCUSDT"], timeframe="5m",
            unique_bars=100, lookback_bars=260, policy=policy,
            as_of_utc="2026-01-01T22:00:00Z", fetcher=lambda *a, **k: bars,
            snapshot=_minimal_snapshot(),
        )


def test_historical_capture_records_safe_duplicates_missing_bars_and_file_inventory(tmp_path, policy):
    index = pd.date_range("2026-01-01T00:00:00Z", periods=262, freq="5min").delete(100)
    close = 100 + np.arange(len(index)) * 0.01
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.ones(len(index)) * 100,
    }, index=index)
    bars = pd.concat((bars, bars.iloc[[50]])).sort_index()
    manifest = capture_historical_bundle(
        bundle_out=tmp_path / "bundle", symbols=["BTCUSDT"], timeframe="5m",
        unique_bars=100, lookback_bars=262, policy=policy,
        as_of_utc="2026-01-01T22:00:05Z", fetcher=lambda *a, **k: bars,
        snapshot=_minimal_snapshot(),
    )
    assert manifest["duplicate_source_bar_count"] == 1
    assert manifest["missing_bar_count_by_symbol"] == {"BTCUSDT": 1}
    assert set(manifest["bundle_file_digests"]) >= {
        "model_serving_snapshot.json", "evaluation_windows.jsonl",
        "capture_stdout.log", "capture_stderr.log",
    }


def test_calibration_matches_bias_clip_temperature_order():
    biased, calibrated = calibrate_probability(0.8, 0.1, 2.0)
    assert biased == pytest.approx(0.7)
    expected = 1 / (1 + np.exp(-np.log(0.7 / 0.3) / 2.0))
    assert calibrated == pytest.approx(expected)


def test_health_counter_ignores_duplicate_and_extreme_triggers_at_20(policy):
    state = SimulatedHealthState(policy)
    for index in range(19):
        result = state.observe(f"BTC:{index}", 0.01)
        assert not result["excluded"]
    duplicate = state.observe("BTC:18", 0.01)
    assert duplicate["advanced"] is False
    result = state.observe("BTC:19", 0.01)
    assert result["excluded"] and "extreme_collapse" in result["events"]
    assert result["consecutive_extreme_count"] == 20


def test_flat_triggers_at_30_and_varying_does_not(policy):
    flat = SimulatedHealthState(policy)
    for index in range(30):
        result = flat.observe(f"BTC:{index}", 0.43)
    assert result["excluded"] and "flat_output" in result["events"]
    varying = SimulatedHealthState(policy)
    for index in range(30):
        result = varying.observe(f"BTC:{index}", 0.4 + 0.01 * (index % 3))
    assert not result["excluded"]


def test_separate_health_instances_keep_model_and_symbol_history_independent(policy):
    lstm_btc = SimulatedHealthState(policy)
    tcn_btc = SimulatedHealthState(policy)
    lstm_eth = SimulatedHealthState(policy)
    for index in range(20):
        lstm_btc.observe(f"BTC:{index}", 0.01)
    assert lstm_btc.consecutive_extreme_count == 20
    assert tcn_btc.consecutive_extreme_count == 0
    assert lstm_eth.consecutive_extreme_count == 0


def _output(identity, kind, probability, *, excluded=False, symbol="BTCUSDT"):
    return {
        "source_bar_id": identity, "source_bar_open_utc": "2026-01-01T00:00:00Z",
        "source_bar_close_utc": "2026-01-01T00:05:00Z", "symbol": symbol,
        "feature_window_digest": "a" * 64, "model_kind": kind,
        "raw_probability": probability, "after_bias_probability": probability,
        "after_temperature_probability": probability, "ret_hat": 0.0, "rv_hat": 0.0,
        "model_present": True, "model_excluded": excluded, "exclusion_reason": "",
        "consecutive_extreme_count": 0, "rolling_probability_std": 0.01,
        "deterministic_repeat_error": 0.0,
    }


def test_statistics_missing_nondeterministic_and_one_sided(policy):
    rows = [_output(f"BTC:{i}", "lstm", 0.6 + i * 0.001) for i in range(95)]
    rows[0]["deterministic_repeat_error"] = 1e-5
    result = calculate_model_statistics(rows, expected_count=100, policy=policy)
    assert result["missing_count"] == 5 and result["missing_rate"] == pytest.approx(0.05)
    assert result["model_health_status"] == "failed_nondeterministic"
    assert result["bullish_rate"] == 1.0


def test_variants_match_voting_exclusions_and_do_not_mutate_configuration():
    outputs = []
    probabilities = {"lstm": 0.8, "tx": 0.2, "tcn": 0.9, "adv": 0.2}
    for kind, probability in probabilities.items():
        outputs.append(_output("BTC:1", kind, probability))
    snapshot = {
        "dl_model_weights": {"lstm": 1, "tx": 1, "tcn": 0, "adv": 0},
        "dl_min_agree": 2, "dl_p_long": 0.1, "dl_p_long_mode": "abs",
        "dl_allow_only": "1",
        "model_entries": [
            {"kind": kind, "metadata_val_auc": 0.6} for kind in probabilities
        ],
    }
    before = deepcopy(snapshot)
    rows, summary = evaluate_ensemble_variants(outputs, snapshot)
    current = next(row for row in rows if row["variant"] == "current_config")
    assert current["agreement_suppressed"] is True and current["centered_probability"] == 0
    no_tcn = next(row for row in rows if row["variant"] == "no_tcn")
    assert "tcn" not in no_tcn["models_used"]
    lstm_tx = next(row for row in rows if row["variant"] == "lstm_tx_only")
    assert set(lstm_tx["models_used"].split(",")) == {"lstm", "tx"}
    assert lstm_tx["configuration_label"] == "shadow_configuration_candidate_only"
    assert "pnl" not in json.dumps(summary).lower() and "sharpe" not in json.dumps(summary).lower()
    assert snapshot == before


def test_no_tcn_preserves_configured_non_tcn_weights():
    outputs = [
        _output("BTC:1", "lstm", 0.6),
        _output("BTC:1", "tx", 0.9),
        _output("BTC:1", "tcn", 0.7),
    ]
    snapshot = {
        "dl_model_weights": {"lstm": 1.0, "tx": 3.0, "tcn": 2.0},
        "dl_min_agree": 1, "dl_p_long": 0.0, "dl_p_long_mode": "abs",
        "dl_allow_only": "1",
        "model_entries": [
            {"kind": kind, "metadata_val_auc": 0.6} for kind in ("lstm", "tx", "tcn")
        ],
    }
    rows, _ = evaluate_ensemble_variants(outputs, snapshot)
    no_tcn = next(row for row in rows if row["variant"] == "no_tcn")
    assert no_tcn["probability"] == pytest.approx(0.825)


def _live_snapshot(symbols, artifact="model-a"):
    normalized = sorted(symbols)
    digest = hashlib.sha256((artifact + "|" + ",".join(normalized)).encode()).hexdigest()
    return {
        "dl_timeframe": "5m", "dl_seq_len": 64, "feature_count": 26,
        "dl_add_symbol_id": True, "dl_symbols": normalized,
        "dl_min_agree": 1, "dl_p_long": 0.1, "dl_p_long_mode": "abs",
        "dl_allow_only": "1", "dl_model_weights": {"lstm": 1.0},
        "dl_bias_lstm": 0.0, "dl_temp_lstm": 1.0,
        "model_entries": [{
            "kind": "lstm", "metadata_status": "loaded", "metadata_kind": "lstm",
            "metadata_timeframe": "5m", "metadata_seq_len": 64,
            "metadata_n_features": 27,
            "metadata_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "metadata_val_auc": 0.65, "scaler_n_features_in": 27,
            "scaler_load_status": "loaded", "model_load_status": "loaded",
            "model_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
            "scaler_sha256": "b" * 64,
        }],
        "snapshot_digest": digest, "market_data_exchange": "fixture",
    }


def _install_live_fixtures(monkeypatch, snapshots):
    sequence = iter(snapshots) if isinstance(snapshots, list) else None
    monkeypatch.setattr(
        alignment, "_safe_snapshot_for_alignment",
        lambda symbols, timeframe, sequence_length: (
            next(sequence) if sequence is not None else _live_snapshot(symbols)
        ),
    )
    monkeypatch.setattr(
        alignment, "load_snapshot_models_read_only", lambda snapshot: ({"lstm": {}}, "cpu")
    )


def _refresh_sequence(symbols):
    polls = {"value": 0}

    def refresh(**kwargs):
        index = min(polls["value"] // 2, 3)
        polls["value"] += 1
        close = pd.Timestamp("2026-01-01T00:05:00Z") + pd.Timedelta(minutes=5 * index)
        identity = {symbol: ensemble.source_bar_id(symbol, close) for symbol in symbols}
        digests = {
            symbol: hashlib.sha256(f"{symbol}:{index}".encode()).hexdigest()
            for symbol in symbols
        }
        return ({
            "source_bar_id_by_symbol": identity,
            "feature_window_digest_by_symbol": digests,
            "source_bar_completed_by_symbol": {symbol: True for symbol in symbols},
            "source_bar_open_utc_by_symbol": {
                symbol: ensemble.canonical_utc(close - pd.Timedelta(minutes=5))
                for symbol in symbols
            },
            "source_bar_close_utc_by_symbol": {
                symbol: ensemble.canonical_utc(close) for symbol in symbols
            },
        }, {
            symbol: np.full((64, 27), index, dtype=np.float32) for symbol in symbols
        })

    return refresh


def test_live_shadow_deduplicates_polls_and_writes_no_trading_files(monkeypatch, tmp_path, policy):
    symbols = ["BTCUSDT", "ETHUSDT"]
    _install_live_fixtures(monkeypatch, _live_snapshot(symbols))
    result = alignment.live_shadow_campaign(
        symbols=symbols, unique_bars=3, output_root=tmp_path, policy=policy,
        poll_seconds=0, max_polls=20, refresh_fn=_refresh_sequence(symbols),
        predictor=lambda window, pack, device: (
            0.0, 0.0, 0.4 + float(window[0, 0]) * 0.01
        ),
    )
    assert result["status"] == "pass"
    assert result["unique_completed_bars_by_symbol"] == {"BTCUSDT": 3, "ETHUSDT": 3}
    assert result["repeated_poll_count"] > 0
    campaign = next(path for path in tmp_path.iterdir() if path.is_dir())
    rows = pd.read_csv(campaign / "completed_bar_outputs.csv")
    assert len(rows) == 6 and rows["source_bar_id"].is_unique
    for forbidden in ("live_signals.csv", "trades_paper.csv", "trades_closed.csv"):
        assert not (campaign / forbidden).exists()


def test_live_shadow_dry_run_creates_no_files(monkeypatch, tmp_path, policy):
    _install_live_fixtures(monkeypatch, _live_snapshot(["BTCUSDT"]))
    result = alignment.live_shadow_campaign(
        symbols=["BTCUSDT"], unique_bars=3, output_root=tmp_path / "new",
        policy=policy, dry_run=True,
    )
    assert result["dry_run"] is True
    assert not (tmp_path / "new").exists()


def test_live_shadow_resume_accepts_same_contract_and_rejects_changes(monkeypatch, tmp_path, policy):
    symbols = ["BTCUSDT"]
    monkeypatch.setattr(
        alignment, "_safe_snapshot_for_alignment",
        lambda values, timeframe, sequence_length: _live_snapshot(values),
    )
    monkeypatch.setattr(
        alignment, "load_snapshot_models_read_only", lambda snapshot: ({"lstm": {}}, "cpu")
    )
    first = alignment.live_shadow_campaign(
        symbols=symbols, unique_bars=3, output_root=tmp_path, policy=policy,
        poll_seconds=0, max_polls=20, refresh_fn=_refresh_sequence(symbols),
        predictor=lambda *args: (0.0, 0.0, 0.4),
    )
    campaign = next(path for path in tmp_path.iterdir() if path.is_dir())
    resumed = alignment.live_shadow_campaign(
        symbols=symbols, unique_bars=3, output_root=tmp_path, campaign_dir=campaign,
        policy=policy, poll_seconds=0, max_polls=0,
        refresh_fn=lambda **kwargs: pytest.fail("completed resume must not poll"),
        predictor=lambda *args: (0.0, 0.0, 0.4),
    )
    assert resumed["campaign_id"] == first["campaign_id"]

    monkeypatch.setattr(
        alignment, "_safe_snapshot_for_alignment",
        lambda values, timeframe, sequence_length: _live_snapshot(values, "model-b"),
    )
    with pytest.raises(ModelAlignmentError, match="resume contract changed"):
        alignment.live_shadow_campaign(
            symbols=symbols, unique_bars=3, output_root=tmp_path, campaign_dir=campaign,
            policy=policy, max_polls=0,
        )

    monkeypatch.setattr(
        alignment, "_safe_snapshot_for_alignment",
        lambda values, timeframe, sequence_length: _live_snapshot(values),
    )
    with pytest.raises(ModelAlignmentError, match="resume contract changed"):
        alignment.live_shadow_campaign(
            symbols=["BTCUSDT", "SOLUSDT"], unique_bars=3,
            output_root=tmp_path, campaign_dir=campaign, policy=policy, max_polls=0,
        )


def test_live_shadow_rejects_changed_digest_for_same_bar(monkeypatch, tmp_path, policy):
    symbols = ["BTCUSDT"]
    _install_live_fixtures(monkeypatch, _live_snapshot(symbols))
    calls = {"value": 0}

    def conflicting(**kwargs):
        calls["value"] += 1
        digest = ("a" if calls["value"] == 1 else "b") * 64
        return ({
            "source_bar_id_by_symbol": {"BTCUSDT": "BTCUSDT:2026-01-01T00:05:00Z"},
            "feature_window_digest_by_symbol": {"BTCUSDT": digest},
            "source_bar_completed_by_symbol": {"BTCUSDT": True},
            "source_bar_open_utc_by_symbol": {"BTCUSDT": "2026-01-01T00:00:00Z"},
            "source_bar_close_utc_by_symbol": {"BTCUSDT": "2026-01-01T00:05:00Z"},
        }, {"BTCUSDT": np.zeros((64, 27), dtype=np.float32)})

    with pytest.raises(ModelAlignmentError, match="conflicting baseline source bar"):
        alignment.live_shadow_campaign(
            symbols=symbols, unique_bars=3, output_root=tmp_path, policy=policy,
            poll_seconds=0, max_polls=3, refresh_fn=conflicting,
            predictor=lambda *args: (0.0, 0.0, 0.4),
        )
