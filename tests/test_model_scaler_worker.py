from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import tools.model_scaler_worker as worker


def _payload(rows=3, timesteps=4, features=2):
    windows = np.arange(rows * timesteps * features, dtype=np.float32).reshape(rows, timesteps, features)
    arrays = {
        "windows": windows,
        "symbols": np.asarray(["BTCUSDT"] * rows),
        "source_bar_ids": np.asarray([f"BTC:{i}" for i in range(rows)]),
        "source_bar_open_utc": np.asarray([f"2026-01-01T00:0{i}:00Z" for i in range(rows)]),
        "source_bar_close_utc": np.asarray([f"2026-01-01T00:0{i + 1}:00Z" for i in range(rows)]),
        "feature_window_digests": np.asarray([str(i) * 64 for i in range(rows)]),
    }
    arrays["input_windows_digest"] = np.asarray(worker.windows_payload_digest(arrays))
    return arrays


def test_worker_imports_no_torch_or_trading_modules():
    tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert "torch" not in names
    assert not any(name.startswith(("runtime", "exchanges", "ml_dl", "tools.live")) for name in names)


def test_worker_transforms_deterministically_and_records_both_precisions(tmp_path):
    arrays = _payload()
    scaler = StandardScaler().fit(arrays["windows"].reshape(-1, 2))
    scaler_path, windows_path = tmp_path / "scaler.joblib", tmp_path / "windows.npz"
    joblib.dump(scaler, scaler_path)
    worker.write_deterministic_npz(windows_path, arrays)
    results = []
    files = []
    for suffix in ("a", "b"):
        output, manifest = tmp_path / f"out-{suffix}.npz", tmp_path / f"manifest-{suffix}.json"
        results.append(worker.run_worker(
            scaler=scaler_path, windows_npz=windows_path, output_npz=output, manifest_out=manifest
        ))
        files.append(output.read_bytes())
        loaded = worker.read_npz(output)
        assert loaded["transformed_float64"].dtype == np.float64
        assert loaded["transformed_float32"].dtype == np.float32
    assert results[0]["worker_result_digest"] == results[1]["worker_result_digest"]
    assert files[0] == files[1]
    assert not any(":" in str(value) and "\\" in str(value) for value in results[0].values())


def test_scaler_width_mismatch_fails(tmp_path):
    arrays = _payload(features=3)
    scaler = StandardScaler().fit(np.zeros((5, 2)))
    path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, path)
    with pytest.raises(worker.ScalerWorkerError, match="width mismatch"):
        worker.transform_windows(path, arrays["windows"])


def test_input_window_digest_mismatch_fails():
    arrays = _payload()
    arrays["windows"][0, 0, 0] = 99
    with pytest.raises(worker.ScalerWorkerError, match="digest mismatch"):
        worker.validate_windows_payload(arrays)


def test_serialization_warnings_are_captured_and_sanitized(monkeypatch, tmp_path):
    arrays = _payload()
    scaler = StandardScaler().fit(arrays["windows"].reshape(-1, 2))
    path = tmp_path / "scaler.joblib"
    path.write_bytes(b"fixture")

    def fake_load(_):
        warnings.warn(r"C:\Users\alice\secret api_key=abc", UserWarning)
        return scaler

    monkeypatch.setattr(worker.joblib, "load", fake_load)
    _, _, categories, messages, _ = worker.transform_windows(path, arrays["windows"])
    assert categories == ["UserWarning"]
    assert "C:\\Users" not in messages[0]
    assert "abc" not in messages[0]
    assert "<redacted>" in messages[0]


def test_deterministic_npz_ignores_mapping_order(tmp_path):
    arrays = _payload()
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    worker.write_deterministic_npz(first, arrays)
    worker.write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
    assert first.read_bytes() == second.read_bytes()


def test_float64_and_float32_transform_paths_are_executed_independently(monkeypatch, tmp_path):
    arrays = _payload()

    class DtypeAwareScaler:
        n_features_in_ = 2
        mean_ = np.zeros(2)
        scale_ = np.ones(2)

        def transform(self, values):
            offset = 1.0 if values.dtype == np.float64 else 2.0
            return values + offset

    scaler_path = tmp_path / "scaler.joblib"
    scaler_path.write_bytes(b"fixture")
    monkeypatch.setattr(worker.joblib, "load", lambda _: DtypeAwareScaler())
    float64, float32, *_ = worker.transform_windows(scaler_path, arrays["windows"])
    np.testing.assert_array_equal(float64, arrays["windows"].astype(np.float64) + 1.0)
    np.testing.assert_array_equal(float32, arrays["windows"] + np.float32(2.0))


def test_worker_refuses_output_collision_before_loading_inputs(tmp_path):
    scaler = tmp_path / "scaler.joblib"
    windows = tmp_path / "windows.npz"
    scaler.write_bytes(b"immutable")
    windows.write_bytes(b"immutable")
    with pytest.raises(worker.ScalerWorkerError, match="distinct"):
        worker.run_worker(
            scaler=scaler,
            windows_npz=windows,
            output_npz=scaler,
            manifest_out=tmp_path / "manifest.json",
        )
    assert scaler.read_bytes() == b"immutable"


def test_worker_refuses_protected_repository_output():
    with pytest.raises(worker.ScalerWorkerError, match="protected"):
        worker._assert_safe_output_paths(
            scaler=Path("model_artifacts/scaler_lstm_latest.joblib"),
            windows_npz=Path("reports/windows.npz"),
            output_npz=Path("model_artifacts/worker-output.npz"),
            manifest_out=Path("reports/worker-manifest.json"),
        )
