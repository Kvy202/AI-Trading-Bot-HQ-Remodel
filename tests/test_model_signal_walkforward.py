from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLS, canonical_feature_columns
from tools import model_signal_walkforward as wf


def _ohlcv(rows: int = 200, start: str = "2024-01-01T00:00:00Z", phase: float = 0.0):
    index = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    position = np.arange(rows, dtype=np.float64)
    log_close = (
        np.log(10_000.0)
        + 0.00002 * position
        + 0.004 * np.sin(position / 4.0 + phase)
        + 0.002 * np.sin(position / 19.0 + phase)
    )
    close = np.exp(log_close)
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": 100.0 + (position % 31.0) + 2.0 * np.sin(position / 7.0),
        },
        index=index,
    )


def _write_raw(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.index.name = "bar_open_utc"
    output.reset_index().to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")


def _two_windows(tmp_path: Path, *, rows: int = 120, split: int = 60):
    earlier = tmp_path / "earlier"
    later = tmp_path / "later"
    frame = _ohlcv(rows)
    _write_raw(earlier / "raw_BTCUSDT.csv", frame.iloc[:split])
    _write_raw(later / "raw_BTCUSDT.csv", frame.iloc[split:])
    return earlier, later, frame


def _minimal_rows(start: str, periods: int, horizon: int) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="5min", tz="UTC")
    rows = []
    for timestamp in times:
        for symbol in wf.SYMBOLS:
            rows.append(
                {
                    "timestamp": timestamp,
                    "target_timestamp": timestamp + horizon * wf.BAR_INTERVAL,
                    "symbol": symbol,
                    "target": int(timestamp.minute % 10 == 0),
                }
            )
    return pd.DataFrame(rows)


def _path_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def mini_research_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("signal_walkforward")
    earlier = root / "earlier_dataset"
    later = root / "later_dataset"
    rows = 1_400
    split = 700
    for symbol, phase in (("BTCUSDT", 0.0), ("ETHUSDT", 0.8)):
        frame = _ohlcv(rows, phase=phase)
        _write_raw(earlier / f"raw_{symbol}.csv", frame.iloc[:split])
        _write_raw(later / f"raw_{symbol}.csv", frame.iloc[split:])
    source_paths = sorted([*earlier.glob("raw_*.csv"), *later.glob("raw_*.csv")])
    source_before = {str(path): _path_digest(path) for path in source_paths}
    ledger = wf.BASE_DIR / "reports" / "model_candidate_validation_access.json"
    candidates = wf.BASE_DIR / "model_artifacts" / "candidates"
    ledger_before = _path_digest(ledger)
    candidates_before = _path_digest(candidates)
    original_research_root = wf.RESEARCH_ROOT
    research_root = root / "reports" / "model_signal_research"
    wf.RESEARCH_ROOT = research_root
    try:
        returned = wf.run_research(
            earlier,
            later,
            output_root=research_root,
            horizons=(6,),
            training_days=2,
            test_days=1,
            step_days=1,
        )
    finally:
        wf.RESEARCH_ROOT = original_research_root
    output = Path(returned["output_directory"])
    return {
        "returned": returned,
        "summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
        "manifest": json.loads(
            (output / "experiment_manifest.json").read_text(encoding="utf-8")
        ),
        "fold_metrics": pd.read_csv(output / "fold_metrics.csv"),
        "source_before": source_before,
        "source_after": {str(path): _path_digest(path) for path in source_paths},
        "ledger_before": ledger_before,
        "ledger_after": _path_digest(ledger),
        "candidates_before": candidates_before,
        "candidates_after": _path_digest(candidates),
    }


def test_earlier_and_later_raw_windows_combine_chronologically(tmp_path):
    earlier, later, expected = _two_windows(tmp_path)
    combined = wf.combine_raw_windows(
        earlier / "raw_BTCUSDT.csv",
        later / "raw_BTCUSDT.csv",
        symbol="BTCUSDT",
    )
    assert combined.index.equals(expected.index)
    assert len(combined) == len(expected)
    assert np.all(np.diff(combined.index.asi8) == wf.BAR_INTERVAL_NS)


def test_duplicate_timestamp_is_rejected(tmp_path):
    earlier, later, frame = _two_windows(tmp_path)
    duplicated_later = pd.concat([frame.iloc[[59]], frame.iloc[60:]])
    _write_raw(later / "raw_BTCUSDT.csv", duplicated_later)
    with pytest.raises(wf.SignalResearchError, match="duplicate timestamp"):
        wf.combine_raw_windows(
            earlier / "raw_BTCUSDT.csv",
            later / "raw_BTCUSDT.csv",
            symbol="BTCUSDT",
        )


def test_conflicting_overlapping_bar_is_rejected(tmp_path):
    earlier, later, frame = _two_windows(tmp_path)
    conflict = frame.iloc[[59]].copy()
    conflict["close"] *= 0.9995
    conflict["low"] = np.minimum(conflict["low"], conflict["close"] * 0.999)
    _write_raw(later / "raw_BTCUSDT.csv", pd.concat([conflict, frame.iloc[60:]]))
    with pytest.raises(wf.SignalResearchError, match="conflicting overlapping bar"):
        wf.combine_raw_windows(
            earlier / "raw_BTCUSDT.csv",
            later / "raw_BTCUSDT.csv",
            symbol="BTCUSDT",
        )


def test_unexpected_five_minute_gap_is_rejected(tmp_path):
    earlier, later, frame = _two_windows(tmp_path)
    _write_raw(later / "raw_BTCUSDT.csv", frame.iloc[61:])
    with pytest.raises(wf.SignalResearchError, match="unexpected 5-minute gap"):
        wf.combine_raw_windows(
            earlier / "raw_BTCUSDT.csv",
            later / "raw_BTCUSDT.csv",
            symbol="BTCUSDT",
        )


def test_target_uses_close_at_t_plus_h_without_future_features():
    raw = _ohlcv(100)
    horizon = 6
    timestamp = raw.index[40]
    feature_frame = pd.DataFrame({"only_time_t_feature": np.arange(len(raw))}, index=raw.index)
    before = wf.build_fixed_horizon_rows(
        raw, feature_frame, symbol="BTCUSDT", horizon_bars=horizon
    ).set_index("timestamp")
    changed = raw.copy()
    changed.loc[raw.index[40 + horizon], "close"] *= 1.02
    after = wf.build_fixed_horizon_rows(
        changed, feature_frame, symbol="BTCUSDT", horizon_bars=horizon
    ).set_index("timestamp")
    expected = np.log(raw["close"].iloc[40 + horizon]) - np.log(raw["close"].iloc[40])
    assert before.loc[timestamp, "future_log_return"] == pytest.approx(expected)
    assert before.loc[timestamp, "only_time_t_feature"] == after.loc[
        timestamp, "only_time_t_feature"
    ]
    assert before.loc[timestamp, "future_log_return"] != after.loc[
        timestamp, "future_log_return"
    ]
    assert before.loc[timestamp, "target_timestamp"] == raw.index[40 + horizon]


def test_final_h_rows_are_excluded_for_horizon_h():
    raw = _ohlcv(80)
    features = pd.DataFrame({"f": np.arange(len(raw))}, index=raw.index)
    for horizon in (6, 12, 24):
        rows = wf.build_fixed_horizon_rows(
            raw, features, symbol="BTCUSDT", horizon_bars=horizon
        )
        assert len(rows) == len(raw) - horizon
        assert rows["timestamp"].iloc[-1] == raw.index[-horizon - 1]


def test_train_test_chronology_is_strict():
    history_start = pd.Timestamp("2024-01-01T00:00:00Z")
    horizon = 6
    rows = _minimal_rows(str(history_start), 4 * 288, horizon)
    fold = wf.make_walkforward_folds(
        history_start,
        history_start + (4 * 288 - 1) * wf.BAR_INTERVAL,
        horizon_bars=horizon,
        training_days=2,
        test_days=1,
        step_days=1,
    )[0]
    train, test = wf.select_fold_rows(rows, fold)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert train["target_timestamp"].max() < test["timestamp"].min()


def test_horizon_purge_removes_h_bars_and_fails_closed_on_label_overlap():
    history_start = pd.Timestamp("2024-01-01T00:00:00Z")
    horizon = 12
    rows = _minimal_rows(str(history_start), 4 * 288, horizon)
    fold = wf.make_walkforward_folds(
        history_start,
        history_start + (4 * 288 - 1) * wf.BAR_INTERVAL,
        horizon_bars=horizon,
        training_days=2,
        test_days=1,
        step_days=1,
    )[0]
    train, _ = wf.select_fold_rows(rows, fold)
    assert len(train) == (2 * 288 - horizon) * len(wf.SYMBOLS)
    tampered = rows.copy()
    last_train_index = tampered.index[
        tampered["timestamp"] < fold.fit_train_end_exclusive
    ][-1]
    tampered.loc[last_train_index, "target_timestamp"] = fold.test_start
    with pytest.raises(wf.SignalResearchError, match="label leakage"):
        wf.select_fold_rows(tampered, fold)


def test_standard_scaler_is_fit_from_training_rows_only():
    X_train = np.asarray([[-2.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    y_train = np.asarray([0, 0, 1, 1])
    X_test = np.asarray([[100.0, 200.0], [200.0, 400.0]])
    _, estimator = wf.fit_model_scores(
        "logistic_regression", X_train, y_train, X_test
    )
    scaler = estimator.named_steps["standard_scaler"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, np.vstack([X_train, X_test]).mean(axis=0))


def test_test_features_and_labels_never_enter_model_fitting(monkeypatch):
    captured = {}

    class SpyEstimator:
        classes_ = np.asarray([0, 1])

        def fit(self, X, y):
            captured["fit_X"] = np.asarray(X).copy()
            captured["fit_y"] = np.asarray(y).copy()
            return self

        def predict_proba(self, X):
            captured["predict_X"] = np.asarray(X).copy()
            return np.tile(np.asarray([[0.4, 0.6]]), (len(X), 1))

    monkeypatch.setattr(wf, "make_model", lambda _: SpyEstimator())
    X_train = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.asarray([0, 1, 0, 1])
    X_test = np.asarray([[999.0], [1000.0]])
    wf.fit_model_scores("logistic_regression", X_train, y_train, X_test)
    assert np.array_equal(captured["fit_X"], X_train)
    assert np.array_equal(captured["fit_y"], y_train)
    assert np.array_equal(captured["predict_X"], X_test)


def test_same_input_config_produces_deterministic_fold_definitions():
    kwargs = {
        "history_start": "2025-08-20T17:35:00Z",
        "history_end_inclusive": "2026-08-02T22:50:00Z",
        "horizon_bars": 60,
        "training_days": 120,
        "test_days": 30,
        "step_days": 30,
    }
    first = [fold.as_dict() for fold in wf.make_walkforward_folds(**kwargs)]
    second = [fold.as_dict() for fold in wf.make_walkforward_folds(**kwargs)]
    assert first == second
    assert len(first) == 7


def test_same_deterministic_model_run_matches_metrics_within_tolerance():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(240, 5))
    y_train = ((X_train[:, 0] - 0.5 * X_train[:, 1]) > 0).astype(np.int8)
    X_test = rng.normal(size=(80, 5))
    symbols = np.asarray(["BTCUSDT"] * 40 + ["ETHUSDT"] * 40)
    y_test = ((X_test[:, 0] - 0.5 * X_test[:, 1]) > 0).astype(np.int8)
    test = pd.DataFrame(
        {
            "target": y_test,
            "future_log_return": (2 * y_test - 1) * 0.002 + X_test[:, 2] * 0.0001,
            "symbol": symbols,
        }
    )
    first_scores, _ = wf.fit_model_scores(
        "hist_gradient_boosting", X_train, y_train, X_test
    )
    second_scores, _ = wf.fit_model_scores(
        "hist_gradient_boosting", X_train, y_train, X_test
    )
    assert np.allclose(first_scores, second_scores, rtol=0, atol=1e-15)
    first = wf.calculate_metrics(test, first_scores)
    second = wf.calculate_metrics(test, second_scores)
    for key in first:
        if isinstance(first[key], float):
            assert first[key] == pytest.approx(second[key], rel=1e-13, abs=1e-13)
        else:
            assert first[key] == second[key]


def test_btc_and_eth_metrics_remain_separate():
    test = pd.DataFrame(
        {
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
            "future_log_return": [-2, -1, 1, 2, -2, -1, 1, 2],
            "symbol": ["BTCUSDT"] * 4 + ["ETHUSDT"] * 4,
        }
    )
    scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.2, 0.1])
    metrics = wf.calculate_metrics(test, scores)
    assert metrics["btc_roc_auc"] == pytest.approx(1.0)
    assert metrics["eth_roc_auc"] == pytest.approx(0.0)


def test_constant_baseline_has_no_artificial_ic_or_return_deciles():
    test = pd.DataFrame(
        {
            "target": [0, 1] * 50,
            "future_log_return": np.linspace(-0.01, 0.02, 100),
            "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
        }
    )
    scores = np.full(len(test), 0.5123, dtype=np.float64)
    metrics = wf.calculate_metrics(test, scores)
    assert metrics["pearson_score_return_ic"] is None
    assert metrics["spearman_score_return_ic"] is None
    assert metrics["return_bucket_status"] == "constant_scores_no_distinct_deciles"
    assert metrics["prediction_decile_row_count"] == 0
    assert metrics["top_minus_bottom_return_spread_bps"] == 0.0
    assert metrics["highest_prediction_decile_mean_future_return"] == pytest.approx(
        test["future_log_return"].mean()
    )
    assert metrics["lowest_prediction_decile_mean_future_return"] == pytest.approx(
        test["future_log_return"].mean()
    )


def test_source_frozen_files_are_not_modified(mini_research_run):
    assert mini_research_run["source_before"] == mini_research_run["source_after"]
    assert mini_research_run["manifest"]["source_files_unchanged_during_run"] is True


def test_validation_access_ledger_is_not_modified(mini_research_run):
    assert mini_research_run["ledger_before"] == mini_research_run["ledger_after"]
    assert (
        mini_research_run["summary"]["safety_contract"][
            "validation_or_internal_test_access_ledger_written"
        ]
        is False
    )


def test_candidate_artifact_directory_is_not_modified(mini_research_run):
    assert mini_research_run["candidates_before"] == mini_research_run["candidates_after"]
    assert mini_research_run["summary"]["safety_contract"]["candidate_artifacts_written"] is False


def test_generated_reports_are_research_only(mini_research_run):
    summary = mini_research_run["summary"]
    manifest = mini_research_run["manifest"]
    assert summary["research_only"] is manifest["research_only"] is True
    assert summary["production_candidate"] is manifest["production_candidate"] is False
    assert summary["promotion_allowed"] is manifest["promotion_allowed"] is False
    assert summary["live_execution_allowed"] is manifest["live_execution_allowed"] is False
    assert summary["historical_periods_pristine_holdout"] is False
    assert summary["future_untouched_confirmation_required_before_deployment"] is True
    assert set(mini_research_run["fold_metrics"]["model_name"]) == set(wf.MODEL_CONFIGS)


def test_feature_contract_ordering_is_unchanged():
    original = tuple(FEATURE_COLS)
    features, diagnostics = wf.build_research_features(_ohlcv(100), symbol="BTCUSDT")
    assert tuple(FEATURE_COLS) == original
    assert list(features.columns) == canonical_feature_columns(True)
    assert len(original) == 26
    assert diagnostics["finite_feature_rows"] == len(features)


def test_research_output_directory_is_gitignored():
    gitignore = (wf.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert "reports/model_signal_research/" in gitignore.splitlines()
