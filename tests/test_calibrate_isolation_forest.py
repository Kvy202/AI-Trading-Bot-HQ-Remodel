"""Tests for Isolation Forest threshold calibration."""

import csv
from pathlib import Path

from tools.calibrate_isolation_forest import calibrate, format_calibration


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_calibration_handles_missing_log(tmp_path):
    result = calibrate(tmp_path / "missing.csv")

    assert result["file_status"] == "missing"
    assert result["total_rows"] == 0
    assert result["scored_anomaly_count"] == 0
    assert all(item["threshold"] is None for item in result["suggestions"])
    assert "No scored anomaly rows" in format_calibration(result)


def test_calibration_handles_empty_log(tmp_path):
    path = tmp_path / "isolation_forest_shadow.csv"
    path.write_text("", encoding="utf-8")

    result = calibrate(path)

    assert result["file_status"] == "empty"
    assert result["total_rows"] == 0
    assert result["scored_anomaly_count"] == 0
    assert all(item["expected_block_rate"] == 0.0 for item in result["suggestions"])


def test_calibration_suggests_thresholds(tmp_path):
    path = tmp_path / "isolation_forest_shadow.csv"
    rows = []
    for i in range(100):
        rows.append([f"t{i}", "BTCUSDT", "anomaly", f"{-1.0 + (i * 0.01):.2f}", "1", "0"])
    _write_csv(
        path,
        ["ts", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked"],
        rows,
    )

    result = calibrate(path)
    by_target = {item["target_block_rate"]: item for item in result["suggestions"]}

    assert result["file_status"] == "ok"
    assert result["total_rows"] == 100
    assert result["scored_anomaly_count"] == 100
    assert by_target[0.05]["threshold"] == -0.96
    assert by_target[0.05]["expected_block_count"] == 5
    assert by_target[0.05]["expected_block_rate"] == 0.05
    assert by_target[0.10]["threshold"] == -0.91
    assert by_target[0.20]["threshold"] == -0.81
    assert by_target[0.30]["threshold"] == -0.71
    text = format_calibration(result)
    assert "target_block_rate" in text
    assert "-0.960000" in text
