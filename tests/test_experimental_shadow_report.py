"""Tests for experimental shadow report summaries."""

import csv
import json
from pathlib import Path

from tools.experimental_shadow_report import (
    ISOLATION_LOG,
    SURVIVAL_LOG,
    XGBOOST_LOG,
    format_text_summary,
    summarize_all,
    write_json_summary,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_missing_log_files_do_not_crash(tmp_path):
    summary = summarize_all(tmp_path)

    assert summary["isolation_forest"]["file_status"] == "missing"
    assert summary["isolation_forest"]["total_rows"] == 0
    assert summary["xgboost_signal"]["file_status"] == "missing"
    assert summary["survival_exit"]["file_status"] == "missing"
    assert "Experimental Shadow Report" in format_text_summary(summary)


def test_valid_log_files_are_summarized(tmp_path):
    _write_csv(
        tmp_path / ISOLATION_LOG,
        ["ts", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked", "reason", "model_version"],
        [
            ["t1", "BTCUSDT", "normal", "0.12", "0", "0", "normal_market", "iso-v1"],
            ["t2", "BTCUSDT", "anomaly", "-0.30", "1", "1", "isolation_anomaly", "iso-v2"],
        ],
    )
    _write_csv(
        tmp_path / XGBOOST_LOG,
        ["timestamp", "symbol", "xgboost_direction", "xgboost_confidence", "would_confirm", "would_reject", "reason", "model_version"],
        [
            ["t1", "BTCUSDT", "LONG", "0.70", "1", "0", "confirmed", "xgb-v1"],
            ["t2", "BTCUSDT", "SHORT", "0.90", "0", "1", "direction_mismatch", "xgb-v2"],
        ],
    )
    _write_csv(
        tmp_path / SURVIVAL_LOG,
        ["timestamp", "symbol", "survival_risk_score", "would_hold", "would_exit_early", "reason", "model_version"],
        [
            ["t1", "BTCUSDT", "0.20", "1", "0", "hold_risk_below_threshold", "surv-v1"],
            ["t2", "BTCUSDT", "0.80", "0", "1", "high_exit_risk", "surv-v2"],
        ],
    )

    summary = summarize_all(tmp_path)

    iso = summary["isolation_forest"]
    assert iso["total_rows"] == 2
    assert iso["normal_count"] == 1
    assert iso["abnormal_count"] == 1
    assert iso["would_block_count"] == 1
    assert iso["actually_blocked_count"] == 1
    assert iso["would_block_rate"] == 0.5
    assert iso["actual_block_rate"] == 0.5
    assert iso["block_rate"] == iso["actual_block_rate"]
    assert iso["latest_anomaly_score"] == -0.30
    assert iso["min_anomaly_score"] == -0.30
    assert iso["max_anomaly_score"] == 0.12
    assert iso["average_anomaly_score"] == -0.09
    assert round(iso["p10_anomaly_score"], 6) == -0.258
    assert round(iso["p50_anomaly_score"], 6) == -0.09
    assert round(iso["p90_anomaly_score"], 6) == 0.078
    assert iso["latest_model_version"] == "iso-v2"
    assert iso["top_reasons"]["isolation_anomaly"] == 1

    xgb = summary["xgboost_signal"]
    assert xgb["would_confirm_count"] == 1
    assert xgb["would_reject_count"] == 1
    assert xgb["average_confidence"] == 0.80
    assert xgb["latest_confidence"] == 0.90
    assert xgb["latest_direction"] == "SHORT"
    assert xgb["latest_model_version"] == "xgb-v2"

    survival = summary["survival_exit"]
    assert survival["would_hold_count"] == 1
    assert survival["would_exit_early_count"] == 1
    assert survival["average_survival_risk_score"] == 0.50
    assert survival["latest_risk_score"] == 0.80
    assert survival["latest_reason"] == "high_exit_risk"
    assert survival["latest_model_version"] == "surv-v2"


def test_isolation_report_distinguishes_would_and_actual_block_rates(tmp_path):
    _write_csv(
        tmp_path / ISOLATION_LOG,
        ["ts", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked", "reason", "model_version"],
        [
            ["t1", "BTCUSDT", "normal", "0.10", "0", "0", "normal_market", "iso-v1"],
            ["t2", "BTCUSDT", "anomaly", "-0.40", "1", "0", "isolation_anomaly", "iso-v1"],
            ["t3", "BTCUSDT", "anomaly", "-0.35", "1", "0", "isolation_anomaly", "iso-v1"],
        ],
    )

    summary = summarize_all(tmp_path)
    iso = summary["isolation_forest"]

    assert iso["would_block_count"] == 2
    assert iso["actually_blocked_count"] == 0
    assert iso["would_block_rate"] == 2 / 3
    assert iso["actual_block_rate"] == 0.0
    assert iso["block_rate"] == iso["actual_block_rate"]

    text = format_text_summary(summary)
    assert "would_block_rate" in text
    assert "actual_block_rate" in text


def test_isolation_report_includes_score_distribution(tmp_path):
    _write_csv(
        tmp_path / ISOLATION_LOG,
        ["ts", "symbol", "anomaly_status", "anomaly_score", "would_block", "actually_blocked", "reason", "model_version"],
        [
            ["t1", "BTCUSDT", "anomaly", "-0.50", "1", "0", "isolation_anomaly", "iso-v1"],
            ["t2", "BTCUSDT", "anomaly", "-0.40", "1", "0", "isolation_anomaly", "iso-v1"],
            ["t3", "BTCUSDT", "anomaly", "-0.30", "1", "0", "isolation_anomaly", "iso-v1"],
            ["t4", "BTCUSDT", "anomaly", "-0.20", "1", "0", "isolation_anomaly", "iso-v1"],
            ["t5", "BTCUSDT", "anomaly", "-0.10", "1", "0", "isolation_anomaly", "iso-v1"],
        ],
    )

    summary = summarize_all(tmp_path)
    iso = summary["isolation_forest"]

    assert iso["min_anomaly_score"] == -0.50
    assert iso["max_anomaly_score"] == -0.10
    assert iso["average_anomaly_score"] == -0.30
    assert round(iso["p10_anomaly_score"], 6) == -0.46
    assert round(iso["p50_anomaly_score"], 6) == -0.30
    assert round(iso["p90_anomaly_score"], 6) == -0.14
    text = format_text_summary(summary)
    assert "min_anomaly_score" in text
    assert "p90_anomaly_score" in text


def test_empty_log_files_are_handled(tmp_path):
    for name in (ISOLATION_LOG, XGBOOST_LOG, SURVIVAL_LOG):
        (tmp_path / name).write_text("", encoding="utf-8")

    summary = summarize_all(tmp_path)

    assert summary["isolation_forest"]["file_status"] == "empty"
    assert summary["xgboost_signal"]["file_status"] == "empty"
    assert summary["survival_exit"]["file_status"] == "empty"
    assert summary["survival_exit"]["average_survival_risk_score"] is None
    assert summary["isolation_forest"]["min_anomaly_score"] is None
    assert summary["isolation_forest"]["p50_anomaly_score"] is None


def test_json_output_format(tmp_path):
    summary = summarize_all(tmp_path)
    out = write_json_summary(summary, tmp_path / "reports" / "experimental_shadow_summary.json")

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"logs_dir", "isolation_forest", "xgboost_signal", "survival_exit"}
    assert data["isolation_forest"]["total_rows"] == 0
    assert data["xgboost_signal"]["file_status"] == "missing"
    assert data["survival_exit"]["would_exit_early_count"] == 0
