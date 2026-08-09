from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import model_training_dataset as ds


def _ohlcv(rows=2000, start="2020-01-01", step="5min"):
    index = pd.date_range(start, periods=rows, freq=step, tz="UTC")
    positions = np.arange(rows, dtype=float)
    close = 10_000 * np.exp(0.02 * np.sin(positions / 13) + positions * 0.00002)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": 100 + (np.arange(rows) % 31),
        },
        index=index,
    )


def test_incomplete_bars_are_excluded_and_exact_duplicates_deduplicate():
    frame = _ohlcv(4)
    duplicated = pd.concat([frame, frame.iloc[[1]]]).sort_index()
    normalized, duplicates, incomplete = ds._normalize_ohlcv(
        duplicated, as_of_utc=frame.index[-1] + pd.Timedelta(minutes=4)
    )
    assert duplicates == 1
    assert incomplete == 1
    assert len(normalized) == 3
    assert normalized.index.is_unique


def test_conflicting_same_timestamp_ohlcv_fails_closed():
    frame = _ohlcv(3)
    conflict = frame.iloc[[1]].copy()
    conflict["close"] += 1
    with pytest.raises(ds.ModelTrainingDatasetError, match="conflicting same-timestamp"):
        ds._normalize_ohlcv(pd.concat([frame, conflict]), as_of_utc="2030-01-01Z")


def test_deterministic_raw_and_npz_digests(tmp_path):
    frame = _ohlcv(10)
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    ds._write_raw_csv(one, frame)
    ds._write_raw_csv(two, frame)
    assert ds.file_digest(one) == ds.file_digest(two)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    ds._write_deterministic_npz(a, x=np.arange(20).reshape(10, 2), y=np.arange(10))
    ds._write_deterministic_npz(b, y=np.arange(10), x=np.arange(20).reshape(10, 2))
    assert ds.file_digest(a) == ds.file_digest(b)


def test_phase22_cutoff_accounts_for_label_lookahead():
    bounds = ds.maximum_training_timestamps(max_lookahead=60)
    earliest = pd.Timestamp(bounds["earliest_source_bar_open_utc"])
    raw_max = pd.Timestamp(bounds["maximum_training_raw_bar_open_utc"])
    endpoint_max = pd.Timestamp(bounds["maximum_training_labeled_endpoint_utc"])
    assert raw_max == earliest - pd.Timedelta(minutes=5)
    assert endpoint_max == earliest - pd.Timedelta(minutes=305)


def test_global_chronological_split_has_purge_and_no_overlap():
    start = pd.Timestamp("2020-01-01", tz="UTC").value
    step = 300 * 1_000_000_000
    timestamps = start + np.arange(1000, dtype=np.int64) * step
    assignments, info = ds.chronological_split(
        {"BTCUSDT": timestamps, "ETHUSDT": timestamps}, purge_bars=60
    )
    assert info["purge_bar_count_each_boundary"] == 60
    for split in assignments.values():
        groups = {code: set(timestamps[split == code]) for code in (0, 1, 2)}
        assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
        assert int(np.sum(split == -1)) == 120
        assert max(groups[0]) < min(groups[1]) < max(groups[1]) < min(groups[2])


def test_sequence_counts_restart_at_each_split_and_symbol():
    split = np.asarray([0] * 70 + [-1] * 5 + [1] * 70 + [-1] * 5 + [2] * 70, dtype=np.int8)
    finite = np.ones(len(split), dtype=bool)
    counts, context = ds._valid_sequence_count(split, finite, 64)
    assert counts == {"train": 7, "validation": 7, "internal_test": 7}
    assert context == {"train": 63, "validation": 63, "internal_test": 63}
    # Running independently gives the same count for each symbol; no window can
    # consume the tail of the other symbol.
    second, _ = ds._valid_sequence_count(split.copy(), finite.copy(), 64)
    assert second == counts


def _patch_build_contract(monkeypatch):
    policy = ds.load_training_policy().copy()
    policy["minimum_usable_labeled_rows_per_symbol"] = 200
    monkeypatch.setattr(ds, "load_training_policy", lambda: policy)
    monkeypatch.setattr(ds, "validate_phase24_evidence", lambda: {})
    monkeypatch.setattr(ds, "record_incumbent_inventory", lambda: {})
    monkeypatch.setattr(ds, "verify_incumbent_inventory", lambda: {})
    monkeypatch.setattr(
        ds,
        "maximum_training_timestamps",
        lambda **kwargs: {
            "earliest_source_bar_open_utc": "2099-01-02T00:00:00Z",
            "final_source_bar_open_utc": "2099-01-02T10:00:00Z",
            "maximum_training_raw_bar_open_utc": "2099-01-01T23:55:00Z",
            "maximum_training_labeled_endpoint_utc": "2099-01-01T18:55:00Z",
        },
    )
    monkeypatch.setattr(
        ds,
        "verify_raw_capture",
        lambda root: {
            "dataset_id": "synthetic", "source_venue": "synthetic-public",
            "combined_raw_digest": "a" * 64,
        },
    )
    return policy


def test_build_uses_27_features_persisted_ids_per_symbol_labels_and_one_split(monkeypatch, tmp_path):
    _patch_build_contract(monkeypatch)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        ds._write_raw_csv(tmp_path / f"raw_{symbol}.csv", _ohlcv())
    manifest = ds.build_dataset(tmp_path, fit_scaler=False, minimum_usable_rows=200)
    contract = ds.specification_contract()
    assert manifest["ordered_feature_names"] == contract["feature_names"]
    assert manifest["feature_count"] == 27
    assert manifest["per_symbol"]["BTCUSDT"]["symbol_id"] == contract["symbol_id_map"]["BTCUSDT"] == 0
    assert manifest["per_symbol"]["ETHUSDT"]["symbol_id"] == contract["symbol_id_map"]["ETHUSDT"] == 1
    assert manifest["per_symbol_label_build"] is True
    assert manifest["phase22_excluded"] is True
    assert manifest["split"]["purge_bar_count_each_boundary"] == 60
    with np.load(tmp_path / "labels_BTCUSDT.npz") as btc, np.load(tmp_path / "labels_ETHUSDT.npz") as eth:
        assert len(btc["ret_cls"]) == len(eth["ret_cls"])
        assert not np.shares_memory(btc["ret_cls"], eth["ret_cls"])


def test_scaler_fits_only_pooled_training_rows_and_is_finite(monkeypatch, tmp_path):
    _patch_build_contract(monkeypatch)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        ds._write_raw_csv(tmp_path / f"raw_{symbol}.csv", _ohlcv())
    ds.build_dataset(tmp_path, fit_scaler=False, minimum_usable_rows=200)
    training_rows = []
    all_rows = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        with np.load(tmp_path / f"features_{symbol}.npz") as values:
            training_rows.append(values["features"][values["split"] == 0].astype(float))
            all_rows.append(values["features"].astype(float))
    manifest = ds.fit_frozen_scaler(tmp_path, require_canonical_version=False)
    import joblib
    scaler = joblib.load(tmp_path / "scaler.joblib")
    expected = np.concatenate(training_rows).mean(axis=0)
    assert np.allclose(scaler.mean_, expected)
    assert not np.allclose(scaler.mean_, np.concatenate(all_rows).mean(axis=0))
    assert len(scaler.mean_) == len(scaler.scale_) == 27
    assert np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()
    assert (scaler.scale_ > 1e-12).all()
    assert manifest["scaler"]["fit_split"] == "train_only"
    assert set(manifest["scaler"]["fit_rows_by_symbol"]) == {"BTCUSDT", "ETHUSDT"}
    with pytest.raises(ds.ModelTrainingDatasetError, match="already exists"):
        ds.fit_frozen_scaler(tmp_path, require_canonical_version=False)


def test_confirmation_capture_refuses_without_all_candidate_freeze(tmp_path):
    with pytest.raises(ds.ModelTrainingDatasetError, match="freeze manifest required"):
        ds._load_selection_freeze(tmp_path / "missing.json")
    incomplete = {"selection_frozen": True, "candidates": {}}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ds.ModelTrainingDatasetError, match="incomplete"):
        ds._load_selection_freeze(path)


def test_minimum_training_rows_and_confirmation_minimum_cannot_be_weakened():
    policy = ds.load_training_policy()
    assert policy["minimum_usable_labeled_rows_per_symbol"] == 25_000
    assert policy["confirmation_minimum_bars_per_symbol"] == 288
    assert policy["confirmation_target_bars_per_symbol"] == 576


def _selection_freeze(path):
    value = {
        "schema_version": 1, "selection_frozen": True,
        "dataset_id": "synthetic", "dataset_digest": "d" * 64,
        "source_venue": "synthetic-public",
        "candidates": {
            kind: {
                "candidate_id": f"{kind}_candidate",
                "candidate_model_digest": (str(index) * 64)[:64],
                "candidate_scaler_digest": "9" * 64,
                "internal_test_recorded": True,
            }
            for index, kind in enumerate(("lstm", "tcn", "tx"), start=1)
        },
        "frozen_at": "2026-01-01T00:00:00Z",
    }
    value["freeze_digest"] = ds.json_digest({
        key: item for key, item in value.items() if key not in {"frozen_at", "freeze_digest"}
    })
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_confirmation_capture_is_post_phase22_label_free_and_uses_frozen_venue(monkeypatch, tmp_path):
    freeze_path = tmp_path / "freeze.json"
    freeze = _selection_freeze(freeze_path)
    monkeypatch.setattr(ds, "validate_phase24_evidence", lambda: {})
    monkeypatch.setattr(ds, "record_incumbent_inventory", lambda: {})
    monkeypatch.setattr(ds, "verify_incumbent_inventory", lambda: {})
    monkeypatch.setattr(
        ds, "phase22_source_bounds",
        lambda *args, **kwargs: {
            "earliest_source_bar_open_utc": "2020-01-01T00:00:00Z",
            "final_source_bar_open_utc": "2020-01-01T01:00:00Z",
        },
    )
    raw = _ohlcv(760, start="2020-01-02")

    def fetcher(symbol, **kwargs):
        assert kwargs["venue"] == freeze["source_venue"]
        return raw.reset_index(names="timestamp")

    result = ds.capture_confirmation_data(
        confirmation_id="synthetic_confirmation",
        as_of_utc=raw.index[-1] + pd.Timedelta(minutes=5), fetcher=fetcher,
        freeze_path=freeze_path, confirmation_root=tmp_path / "confirmations",
    )
    root = tmp_path / "confirmations/synthetic_confirmation"
    assert result["labels_present"] is False
    assert result["selection_freeze_digest"] == freeze["freeze_digest"]
    assert not list(root.glob("*label*"))
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert result["per_symbol"][symbol]["unique_completed_bars"] == 576
        assert pd.Timestamp(result["per_symbol"][symbol]["first_timestamp_utc"]) > pd.Timestamp("2020-01-01T01:00:00Z")
        with np.load(root / f"windows_{symbol}.npz") as values:
            assert values["windows"].shape == (576, 64, 27)


def test_confirmation_below_288_unique_bars_is_pending_not_candidate_failure(monkeypatch, tmp_path):
    freeze_path = tmp_path / "freeze.json"
    _selection_freeze(freeze_path)
    monkeypatch.setattr(ds, "validate_phase24_evidence", lambda: {})
    monkeypatch.setattr(ds, "record_incumbent_inventory", lambda: {})
    monkeypatch.setattr(
        ds, "phase22_source_bounds",
        lambda *args, **kwargs: {
            "earliest_source_bar_open_utc": "2020-01-01T00:00:00Z",
            "final_source_bar_open_utc": "2020-01-01T01:00:00Z",
        },
    )
    raw = _ohlcv(300, start="2020-01-02")
    with pytest.raises(ds.ModelTrainingDatasetError, match="sealed_confirmation_pending"):
        ds.capture_confirmation_data(
            confirmation_id="too_short",
            as_of_utc=raw.index[-1] + pd.Timedelta(minutes=5),
            fetcher=lambda symbol, **kwargs: raw.reset_index(names="timestamp"),
            freeze_path=freeze_path, confirmation_root=tmp_path / "confirmations",
        )
    assert not (tmp_path / "confirmations/too_short").exists()
