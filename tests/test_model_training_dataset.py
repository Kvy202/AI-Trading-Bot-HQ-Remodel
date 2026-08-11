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


def _ccxt_ohlcv(timestamps):
    rows = len(timestamps)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.arange(rows, dtype=float) + 100,
        "high": np.arange(rows, dtype=float) + 101,
        "low": np.arange(rows, dtype=float) + 99,
        "close": np.arange(rows, dtype=float) + 100.5,
        "volume": np.arange(rows, dtype=float) + 10,
    })


def test_ccxt_numeric_milliseconds_convert_to_2026_utc_not_1970():
    frame = _ccxt_ohlcv([1770876300000])
    normalized, duplicates, incomplete = ds._normalize_ohlcv(
        frame, as_of_utc="2027-01-01T00:00:00Z"
    )
    assert normalized.index[0] == pd.Timestamp(1770876300000, unit="ms", tz="UTC")
    assert normalized.index[0].year == 2026
    assert duplicates == incomplete == 0


def test_ccxt_five_minute_millisecond_sequence_has_300_second_deltas():
    start_ms = 1770876300000
    normalized, _, _ = ds._normalize_ohlcv(
        _ccxt_ohlcv([start_ms, start_ms + 300_000, start_ms + 600_000]),
        as_of_utc="2027-01-01T00:00:00Z",
    )
    assert (np.diff(normalized.index.asi8) / 1_000_000_000).tolist() == [300.0, 300.0]
    assert ds._gap_statistics(normalized.index) == (0, 300.0)


@pytest.mark.parametrize("timestamp", [
    "2026-02-12T05:05:00Z",
    "2026-02-12T10:35:00+05:30",
])
def test_iso_and_timezone_aware_timestamp_input_remains_utc_normalized(timestamp):
    normalized, _, _ = ds._normalize_ohlcv(
        _ccxt_ohlcv([timestamp]), as_of_utc="2027-01-01T00:00:00Z"
    )
    assert normalized.index[0] == pd.Timestamp("2026-02-12T05:05:00Z")


@pytest.mark.parametrize("timestamp,match", [
    (np.inf, "finite"),
    (np.nan, "finite"),
    ("definitely-not-a-timestamp", "converted cleanly"),
])
def test_nonfinite_or_malformed_timestamps_fail_closed(timestamp, match):
    with pytest.raises(ds.ModelTrainingDatasetError, match=match):
        ds._normalize_ohlcv(
            _ccxt_ohlcv([timestamp]), as_of_utc="2027-01-01T00:00:00Z"
        )


def test_numeric_1970_and_requested_range_mismatch_fail_closed():
    with pytest.raises(ds.ModelTrainingDatasetError, match="plausible"):
        ds._normalize_ohlcv(_ccxt_ohlcv([0]), as_of_utc="2026-08-09T00:00:00Z")
    with pytest.raises(ds.ModelTrainingDatasetError, match="requested capture range"):
        ds._normalize_ohlcv(
            _ccxt_ohlcv([1770876300000]), as_of_utc="2027-01-01T00:00:00Z",
            requested_start_utc="2026-03-01T00:00:00Z",
            requested_end_exclusive_utc="2026-04-01T00:00:00Z",
        )


def test_sub_timeframe_spacing_for_distinct_5m_bars_fails():
    start_ms = 1770876300000
    with pytest.raises(ds.ModelTrainingDatasetError, match="sub-timeframe"):
        ds._normalize_ohlcv(
            _ccxt_ohlcv([start_ms, start_ms + 1]),
            as_of_utc="2027-01-01T00:00:00Z",
        )


def test_numeric_identical_duplicates_deduplicate_and_conflicts_still_fail():
    start_ms = 1770876300000
    frame = pd.concat([_ccxt_ohlcv([start_ms])] * 2, ignore_index=True)
    normalized, duplicates, _ = ds._normalize_ohlcv(frame, as_of_utc="2027-01-01T00:00:00Z")
    assert len(normalized) == 1 and duplicates == 1
    conflict = frame.copy()
    conflict.loc[1, "close"] += 1
    with pytest.raises(ds.ModelTrainingDatasetError, match="conflicting same-timestamp"):
        ds._normalize_ohlcv(conflict, as_of_utc="2027-01-01T00:00:00Z")


def test_numeric_incomplete_current_candle_logic_is_preserved():
    start = pd.Timestamp("2026-02-12T05:00:00Z")
    timestamps = [(start + pd.Timedelta(minutes=5 * index)).value // 1_000_000 for index in range(3)]
    normalized, _, incomplete = ds._normalize_ohlcv(
        _ccxt_ohlcv(timestamps), as_of_utc=start + pd.Timedelta(minutes=14)
    )
    assert len(normalized) == 2 and incomplete == 1


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


def test_bitget_public_history_pagination_uses_200_bar_chunks_without_skips():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    start_ms = start.value // 1_000_000
    end = start + pd.Timedelta(minutes=5 * 450)

    class FakeExchange:
        rateLimit = 250
        def __init__(self):
            self.calls = []
        def fetch_ohlcv(self, symbol, *, timeframe, since, limit, params):
            self.calls.append({"since": since, "limit": limit, "params": params})
            return [
                [since + index * 300_000, 1.0, 2.0, 0.5, 1.5, 10.0]
                for index in range(limit)
            ]

    exchange = FakeExchange()
    frame = ds._fetch_ccxt_ohlcv_range(
        exchange, "BTC/USDT:USDT", timeframe="5m", start_utc=ds.canonical_utc(start),
        end_utc=ds.canonical_utc(end), limit=450, params={"productType": "USDT-FUTURES"},
        per_page=ds.BITGET_HISTORY_PAGE_LIMIT, force_history_endpoint=True,
        sleep_fn=lambda _: None,
    )
    assert len(frame) == 450
    assert all(call["limit"] == 200 for call in exchange.calls)
    assert all(call["params"]["useHistoryEndpoint"] is True for call in exchange.calls)
    assert [call["since"] for call in exchange.calls] == [
        start_ms, start_ms + 200 * 300_000, start_ms + 400 * 300_000,
    ]
    diagnostics = frame.attrs["capture_diagnostics"]
    assert diagnostics["pages_requested"] == diagnostics["pages_returned"] == 3
    assert diagnostics["pagination_stop_reason"] == "requested_end_reached"
    normalized, _, _ = ds._normalize_ohlcv(
        frame, as_of_utc="2027-01-01T00:00:00Z",
        requested_start_utc=start, requested_end_exclusive_utc=end,
    )
    assert set(np.diff(normalized.index.asi8) / 1_000_000_000) == {300.0}


def _patch_capture_contract(monkeypatch, target):
    policy = ds.load_training_policy().copy()
    policy["target_raw_bars_per_symbol"] = target
    monkeypatch.setattr(ds, "load_training_policy", lambda: policy)
    monkeypatch.setattr(ds, "validate_phase24_evidence", lambda: {})
    monkeypatch.setattr(ds, "record_incumbent_inventory", lambda: {})
    monkeypatch.setattr(ds, "verify_incumbent_inventory", lambda: {})
    monkeypatch.setattr(ds, "specification_contract", lambda: {"label_contract": {"max_hold": 60}})
    monkeypatch.setattr(ds, "maximum_training_timestamps", lambda **kwargs: {
        "earliest_source_bar_open_utc": "2026-02-02T00:00:00Z",
        "final_source_bar_open_utc": "2026-02-03T00:00:00Z",
        "maximum_training_raw_bar_open_utc": "2026-02-01T23:55:00Z",
        "maximum_training_labeled_endpoint_utc": "2026-02-01T18:55:00Z",
    })
    monkeypatch.setattr(
        ds, "verify_raw_capture",
        lambda root: json.loads((Path(root) / "raw_manifest.json").read_text(encoding="utf-8")),
    )


def _bounded_fetcher(calls):
    def fetcher(symbol, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        end = pd.Timestamp(kwargs["end_utc"])
        return _ccxt_ohlcv([
            (end - pd.Timedelta(minutes=5 * offset)).value // 1_000_000
            for offset in (4, 3, 2, 1)
        ])

    return fetcher


def test_default_capture_end_and_dataset_identity_are_unchanged(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 4)
    calls = []
    manifest = ds.capture_training_data(
        target_bars=4, fetcher=_bounded_fetcher(calls), dataset_root=tmp_path,
        as_of_utc="2026-02-03T00:00:00Z",
    )
    cutoff = "2026-02-01T23:55:00Z"
    legacy_contract = {
        "phase": 24, "venue": "bitget", "timeframe": "5m",
        "symbols": ["BTCUSDT", "ETHUSDT"], "cutoff": cutoff, "target": 4,
    }
    expected_id = f"phase24_5m_{ds.json_digest(legacy_contract)[:12]}"

    assert manifest["dataset_id"] == expected_id == ds._dataset_id("bitget", cutoff, 4)
    assert {call["end_utc"] for call in calls} == {"2026-02-02T00:00:00Z"}
    assert manifest["default_safe_end_exclusive_utc"] == "2026-02-02T00:00:00Z"
    assert manifest["effective_end_exclusive_utc"] == "2026-02-02T00:00:00Z"
    assert manifest["explicit_historical_end_requested"] is False
    assert "effective_end_exclusive_utc" not in ds._raw_capture_digest_contract(manifest)


def test_earlier_explicit_capture_end_is_accepted_and_changes_identity(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 4)
    calls = []
    explicit_end = "2026-01-15T00:00:00Z"
    manifest = ds.capture_training_data(
        target_bars=4, fetcher=_bounded_fetcher(calls), dataset_root=tmp_path,
        as_of_utc="2026-02-03T00:00:00Z", capture_end_exclusive_utc=explicit_end,
    )
    default_id = ds._dataset_id("bitget", "2026-02-01T23:55:00Z", 4)
    explicit_id = ds._dataset_id(
        "bitget", "2026-02-01T23:55:00Z", 4,
        effective_end_exclusive_utc=explicit_end,
    )

    assert manifest["dataset_id"] == explicit_id
    assert explicit_id != default_id
    assert manifest["default_safe_end_exclusive_utc"] == "2026-02-02T00:00:00Z"
    assert manifest["effective_end_exclusive_utc"] == explicit_end
    assert manifest["explicit_historical_end_requested"] is True
    assert ds._raw_capture_digest_contract(manifest)["effective_end_exclusive_utc"] == explicit_end
    assert {call["end_utc"] for call in calls} == {explicit_end}
    assert all(
        pd.Timestamp(info["actual_last_utc"]) < pd.Timestamp(explicit_end)
        for info in manifest["per_symbol"].values()
    )


def test_later_than_safe_and_non_aligned_capture_ends_fail_closed(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 4)
    with pytest.raises(ds.ModelTrainingDatasetError, match="Phase-22-safe maximum"):
        ds.capture_training_data(
            target_bars=4, fetcher=lambda *args, **kwargs: None, dataset_root=tmp_path,
            capture_end_exclusive_utc="2026-02-02T00:05:00Z",
        )
    with pytest.raises(ds.ModelTrainingDatasetError, match="5-minute boundary"):
        ds.capture_training_data(
            target_bars=4, fetcher=lambda *args, **kwargs: None, dataset_root=tmp_path,
            capture_end_exclusive_utc="2026-01-15T00:01:00Z",
        )
    assert list(tmp_path.iterdir()) == []


def test_fetcher_rows_at_or_above_explicit_end_are_rejected(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 4)
    explicit_end = pd.Timestamp("2026-01-15T00:00:00Z")

    def fetcher(symbol, **kwargs):
        return _ccxt_ohlcv([
            (explicit_end - pd.Timedelta(minutes=15)).value // 1_000_000,
            (explicit_end - pd.Timedelta(minutes=10)).value // 1_000_000,
            (explicit_end - pd.Timedelta(minutes=5)).value // 1_000_000,
            explicit_end.value // 1_000_000,
        ])

    with pytest.raises(ds.ModelTrainingDatasetError, match="requested capture range"):
        ds.capture_training_data(
            dataset_id="out_of_range", target_bars=4, fetcher=fetcher,
            dataset_root=tmp_path, as_of_utc="2026-02-03T00:00:00Z",
            capture_end_exclusive_utc=ds.canonical_utc(explicit_end),
        )


def _raw_range(first, last):
    return {
        "market_symbols": ["BTCUSDT", "ETHUSDT"],
        "per_symbol": {
            symbol: {"actual_first_utc": first, "actual_last_utc": last}
            for symbol in ("BTCUSDT", "ETHUSDT")
        },
    }


def test_raw_capture_non_overlap_requires_strictly_prior_ranges(tmp_path):
    earlier = _raw_range("2025-08-20T17:35:00Z", "2026-02-10T08:10:00Z")
    later = _raw_range("2026-02-10T08:15:00Z", "2026-08-02T22:50:00Z")
    earlier_dir = tmp_path / "earlier"
    later_dir = tmp_path / "later"
    earlier_dir.mkdir()
    later_dir.mkdir()
    (earlier_dir / "raw_manifest.json").write_text(json.dumps(earlier), encoding="utf-8")
    (later_dir / "raw_manifest.json").write_text(json.dumps(later), encoding="utf-8")

    result = ds.verify_raw_capture_non_overlap(earlier_dir, later_dir)
    assert result["passed"] is True
    assert all(value["strictly_prior"] for value in result["symbols"].values())


@pytest.mark.parametrize("earlier_last", [
    "2026-02-10T08:15:00Z",
    "2026-02-10T08:20:00Z",
])
def test_raw_capture_non_overlap_rejects_touching_or_overlapping_ranges(earlier_last):
    earlier = _raw_range("2025-08-20T17:35:00Z", earlier_last)
    later = _raw_range("2026-02-10T08:15:00Z", "2026-08-02T22:50:00Z")
    with pytest.raises(ds.ModelTrainingDatasetError, match="timestamp overlap"):
        ds.verify_raw_capture_non_overlap(earlier, later)


def test_capture_cli_exposes_optional_exclusive_end():
    args = ds.build_parser().parse_args([
        "capture", "--capture-end-exclusive-utc", "2026-02-10T08:15:00Z",
    ])
    assert args.capture_end_exclusive_utc == "2026-02-10T08:15:00Z"


def test_capture_manifest_records_public_pagination_coverage_diagnostics(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 4)

    def fetcher(symbol, **kwargs):
        end = pd.Timestamp(kwargs["end_utc"])
        timestamps = [(end - pd.Timedelta(minutes=5 * offset)).value // 1_000_000 for offset in (4, 3, 2, 1)]
        frame = _ccxt_ohlcv(timestamps)
        frame.attrs["capture_diagnostics"] = {
            "pages_requested": 2, "pages_returned": 2,
            "first_exchange_timestamp": ds.canonical_utc(end - pd.Timedelta(minutes=20)),
            "last_exchange_timestamp": ds.canonical_utc(end - pd.Timedelta(minutes=5)),
            "requested_start": kwargs["start_utc"], "requested_end": kwargs["end_utc"],
            "rows_before_normalization": 4, "rows_after_normalization": None,
            "pagination_stop_reason": "requested_end_reached",
        }
        return frame

    manifest = ds.capture_training_data(
        dataset_id="synthetic_capture", target_bars=4, fetcher=fetcher,
        dataset_root=tmp_path, as_of_utc="2026-02-03T00:00:00Z",
    )
    for info in manifest["per_symbol"].values():
        assert info["pages_requested"] == info["pages_returned"] == 2
        assert info["rows_before_normalization"] == info["rows_after_normalization"] == 4
        assert info["pagination_stop_reason"] == "requested_end_reached"
        assert info["first_exchange_timestamp"].startswith("2026-")


def test_incomplete_public_range_reports_specific_status_without_weakening_target(monkeypatch, tmp_path):
    _patch_capture_contract(monkeypatch, 5)

    def fetcher(symbol, **kwargs):
        end = pd.Timestamp(kwargs["end_utc"])
        return _ccxt_ohlcv([
            (end - pd.Timedelta(minutes=10)).value // 1_000_000,
            (end - pd.Timedelta(minutes=5)).value // 1_000_000,
        ])

    with pytest.raises(ds.ModelTrainingDatasetError, match="historical_capture_range_incomplete"):
        ds.capture_training_data(
            dataset_id="short_capture", target_bars=5, fetcher=fetcher,
            dataset_root=tmp_path, as_of_utc="2026-02-03T00:00:00Z",
        )
    manifest = json.loads((tmp_path / "short_capture/raw_manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture_status"] == "historical_capture_range_incomplete"
    assert manifest["target_raw_bars_per_symbol"] == 5


def test_timestamp_corrupted_partial_dataset_requires_manual_recapture():
    manifest = {
        "capture_status": "insufficient_training_data",
        "per_symbol": {"BTCUSDT": {
            "requested_start_utc": "2026-02-09T11:25:00Z",
            "requested_end_exclusive_utc": "2026-08-02T22:55:00Z",
            "actual_first_utc": "1970-01-01T00:29:30.876300Z",
            "actual_last_utc": "1970-01-01T00:29:45.711000Z",
            "rows": 15043, "maximum_gap_seconds": 0.2403,
        }},
    }
    with pytest.raises(ds.ModelTrainingDatasetError, match="delete_and_recapture"):
        ds._validate_partial_capture_manifest(manifest)


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


def _build_frozen_verifier_dataset(monkeypatch, root, *, explicit):
    real_verify_raw_capture = ds.verify_raw_capture
    policy = _patch_build_contract(monkeypatch)
    policy["target_raw_bars_per_symbol"] = 2000
    monkeypatch.setattr(ds, "verify_raw_capture", real_verify_raw_capture)
    frames = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frames[symbol] = _ohlcv()
        ds._write_raw_csv(root / f"raw_{symbol}.csv", frames[symbol])
    bounds = ds.maximum_training_timestamps()
    default_end = "2099-01-02T00:00:00Z"
    effective_end = "2098-12-31T00:00:00Z" if explicit else default_end
    per_symbol = {
        symbol: {
            "requested_start_utc": "2019-12-31T00:00:00Z",
            "requested_end_exclusive_utc": effective_end,
            "actual_first_utc": ds.canonical_utc(frame.index[0]),
            "actual_last_utc": ds.canonical_utc(frame.index[-1]),
            "completed_rows": len(frame),
            "file_sha256": ds.file_digest(root / f"raw_{symbol}.csv"),
        }
        for symbol, frame in frames.items()
    }
    raw = {
        "schema_version": 1,
        "dataset_id": "synthetic_explicit" if explicit else "synthetic_legacy",
        "capture_status": "complete",
        "captured_at": "2026-01-01T00:00:00Z",
        "source_venue": "synthetic-public",
        "market_symbols": ["BTCUSDT", "ETHUSDT"],
        "timeframe": "5m",
        "target_raw_bars_per_symbol": 2000,
        "phase22_bounds": bounds,
        "per_symbol": per_symbol,
    }
    if explicit:
        raw.update({
            "default_safe_end_exclusive_utc": default_end,
            "effective_end_exclusive_utc": effective_end,
            "explicit_historical_end_requested": True,
        })
    raw["combined_raw_digest"] = ds.json_digest({
        "contract": ds._raw_capture_digest_contract(raw),
        "files": {
            symbol: per_symbol[symbol]["file_sha256"] for symbol in sorted(per_symbol)
        },
    })
    raw["manifest_digest"] = ds.json_digest({
        key: value for key, value in raw.items() if key not in {"captured_at", "manifest_digest"}
    })
    (root / "raw_manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    ds.build_dataset(root, fit_scaler=False, minimum_usable_rows=200)
    ds.fit_frozen_scaler(root, require_canonical_version=False)
    return raw


def test_verify_dataset_uses_canonical_raw_contract_for_bounded_capture(
    monkeypatch, tmp_path
):
    raw = _build_frozen_verifier_dataset(monkeypatch, tmp_path, explicit=True)
    canonical_helper = ds._raw_capture_digest_contract
    calls = []

    def observed_contract(value):
        calls.append(value)
        return canonical_helper(value)

    monkeypatch.setattr(ds, "_raw_capture_digest_contract", observed_contract)
    manifest = ds.verify_dataset(tmp_path)

    assert manifest["dataset_status"] == "frozen_ready"
    assert calls and calls[0]["combined_raw_digest"] == raw["combined_raw_digest"]
    assert canonical_helper(raw)["effective_end_exclusive_utc"] == "2098-12-31T00:00:00Z"


def test_verify_dataset_legacy_raw_contract_remains_bit_compatible(monkeypatch, tmp_path):
    raw = _build_frozen_verifier_dataset(monkeypatch, tmp_path, explicit=False)
    legacy_contract = {
        "venue": raw["source_venue"], "timeframe": "5m",
        "target": raw["target_raw_bars_per_symbol"], "phase22_bounds": raw["phase22_bounds"],
    }
    expected = ds.json_digest({
        "contract": legacy_contract,
        "files": {
            symbol: raw["per_symbol"][symbol]["file_sha256"]
            for symbol in sorted(raw["per_symbol"])
        },
    })

    assert ds._raw_capture_digest_contract(raw) == legacy_contract
    assert raw["combined_raw_digest"] == expected
    assert ds.verify_dataset(tmp_path)["dataset_status"] == "frozen_ready"


def test_verify_dataset_rejects_tampered_explicit_capture_end(monkeypatch, tmp_path):
    _build_frozen_verifier_dataset(monkeypatch, tmp_path, explicit=True)
    path = tmp_path / "raw_manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["effective_end_exclusive_utc"] = "2098-12-30T23:55:00Z"
    raw["manifest_digest"] = ds.json_digest({
        key: value for key, value in raw.items() if key not in {"captured_at", "manifest_digest"}
    })
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ds.ModelTrainingDatasetError, match="combined raw digest mismatch"):
        ds.verify_dataset(tmp_path)


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
