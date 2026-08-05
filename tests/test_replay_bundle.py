"""Synthetic tests for Phase 20 replay bundles and coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.replay_bundle import (
    ReplayBundleError,
    bundle_digest,
    build_replay_bundle,
    calculate_coverage,
    collect_source_rows,
    resolve_historical_sources,
    validate_replay_bundle,
)
from tools.replay_contract import capture_replay_contract, write_replay_contract
from tools.model_serving_snapshot import capture_model_serving_snapshot, write_model_serving_snapshot


START = "2026-08-03T16:00:00Z"
FINISH = "2026-08-03T16:05:00Z"
IDENTITY = "xgboost_shadow_outcome:20260803160000"


def _csv(path: Path, fields: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
    return path


def _contract(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "v2").mkdir()
    (root / "config").mkdir()
    (root / "tools" / "live_executor.py").write_text("fixture\n", encoding="utf-8")
    (root / "v2" / "risk_controls.py").write_text("fixture\n", encoding="utf-8")
    (root / "config" / "run.json").write_text("{}", encoding="utf-8")
    forced = tmp_path / "forced.json"
    forced.write_text(
        json.dumps(
            {
                "LIVE_TRADING": False,
                "PAPER_TRADING": True,
                "LIVE_MODE": False,
                "EXEC_PAPER": True,
                "PLACE_REAL_ORDERS": False,
                "EXEC_RESTORE_STATE": False,
                "EXEC_BIAS_GUARD": False,
            }
        ),
        encoding="utf-8",
    )
    contract = capture_replay_contract(
        IDENTITY,
        "xgboost_shadow_outcome",
        forced,
        base_dir=root,
        run_started_utc=START,
        expected_finished_at=FINISH,
    )
    return write_replay_contract(contract, tmp_path / "contract.json")


def _logs(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    signal_fields = ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "signal_id"]
    inside = ["2026-08-03T16:00:30Z", "BTC", 100, 0.8, 0.01, 1, 0.5, "abs", "model", "sig-1"]
    _csv(
        logs / "live_signals.csv",
        signal_fields,
        [
            ["2026-08-03T15:59:00Z", "BTC", 99, 0.8, 0.01, 1, 0.5, "abs", "model", "outside"],
            inside,
            inside,
            ["2026-08-03T16:03:00Z", "BTC", 102, -0.8, 0.01, 1, 0.5, "abs", "model", "sig-2"],
            ["2026-08-03T16:04:30Z", "BTC", 101, -0.8, 0.01, 1, 0.5, "abs", "model", "sig-3"],
            ["2026-08-03T16:06:00Z", "BTC", 103, 0.8, 0.01, 1, 0.5, "abs", "model", "outside-2"],
        ],
    )
    _csv(
        logs / "archive" / "xgboost_signal_shadow.csv",
        ["timestamp", "symbol", "existing_signal", "would_confirm", "would_reject", "signal_id", "artifact_path"],
        [["2026-08-03T16:00:30Z", "BTC", "LONG", 1, 0, "sig-1", r"C:\model.joblib"]],
    )
    _csv(
        logs / "trades_paper_20260803.csv",
        ["ts", "symbol", "side", "price", "qty", "reason", "signal_id"],
        [["2026-08-03T16:00:30Z", "BTC", "BUY", 100, 1, "ENTRY", "sig-1"]],
    )
    _csv(
        logs / "trades_closed.csv",
        ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price", "realized_pnl", "reason", "signal_id"],
        [["2026-08-03T16:03:00Z", "BTC", "SELL", 1, 100, 102, 2, "EXIT_TP", "sig-1"]],
    )
    (logs / "executor_state.json").write_text("secret state", encoding="utf-8")
    (logs / "matrix_xgboost_shadow_outcome_mode_env.json").write_text("env", encoding="utf-8")
    model_fields = ["ts", "symbol", "lstm_p", "tcn_p", "tx_p"]
    inside_model = ["2026-08-03T16:00:30Z", "BTC", 0.4, 0.5, 0.6]
    _csv(
        logs / "live_models_by_symbol.csv",
        model_fields,
        [["2026-08-03T15:59:30Z", "BTC", 0.1, 0.2, 0.3], inside_model,
         inside_model, ["2026-08-03T16:04:30Z", "BTC", 0.45, 0.55, 0.65],
         ["2026-08-03T16:06:00Z", "BTC", 0.7, 0.8, 0.9]],
    )
    _csv(
        logs / "live_meta_log.csv",
        ["ts", "p_meta", "lstm_p", "tcn_p", "tx_p"],
        [["2026-08-03T16:00:30Z", 0.5, 0.4, 0.5, 0.6]],
    )
    return logs


def _model_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot_repo"
    for directory in ("tools", "ml_dl", "config", "model_artifacts"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative in ("tools/live_writer.py", "ml_dl/dl_ensemble.py", "ml_dl/dl_infer.py", "ml_dl/dl_models.py"):
        (root / relative).write_text("# fixture\n", encoding="utf-8")
    (root / "features.py").write_text("FEATURE_COLS = ['a', 'b']\n", encoding="utf-8")
    (root / "config" / "run.json").write_text("{}", encoding="utf-8")
    snapshot = capture_model_serving_snapshot(
        IDENTITY, "xgboost_shadow_outcome", base_dir=root,
        forced_env_overrides={"LIVE_TRADING": False, "PAPER_TRADING": True,
                              "LIVE_MODE": False, "EXEC_PAPER": True,
                              "PLACE_REAL_ORDERS": False},
    )
    return write_model_serving_snapshot(snapshot, tmp_path / "snapshot.json")


def test_bundle_filters_window_deduplicates_and_preserves_sources(tmp_path):
    logs = _logs(tmp_path)
    before = (logs / "live_signals.csv").read_bytes()
    result = build_replay_bundle(
        IDENTITY,
        START,
        FINISH,
        _contract(tmp_path),
        reports_dir=tmp_path / "reports",
        logs_dir=logs,
        bundle_root=tmp_path / "bundles",
        manifest_digest_value="manifest",
    )
    bundle = Path(result["bundle_path"])
    rows = list(csv.DictReader((bundle / "live_signals.csv").open(encoding="utf-8")))

    assert [row["signal_id"] for row in rows] == ["sig-1", "sig-2", "sig-3"]
    assert result["manifest"]["duplicate_rows_removed"] == 1
    assert (logs / "live_signals.csv").read_bytes() == before
    assert (logs / "live_signals.csv").exists()
    assert not (bundle / "executor_state.json").exists()
    assert not any(path.name.endswith("mode_env.json") for path in bundle.iterdir())
    closed = list(csv.DictReader((bundle / "trades_closed.csv").open(encoding="utf-8")))
    assert closed[0]["signal_id"] == "sig-1"


def test_conflicting_signal_ids_are_rejected(tmp_path):
    logs = _logs(tmp_path)
    with (logs / "live_signals.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            ["2026-08-03T16:00:30Z", "BTC", 999, 0.8, 0.01, 1, 0.5, "abs", "model", "sig-1"]
        )

    with pytest.raises(ReplayBundleError):
        build_replay_bundle(
            IDENTITY,
            START,
            FINISH,
            _contract(tmp_path),
            reports_dir=tmp_path / "reports",
            logs_dir=logs,
            bundle_root=tmp_path / "bundles",
            manifest_digest_value="manifest",
        )


def test_bundle_digest_is_deterministic(tmp_path):
    logs = _logs(tmp_path)
    contract = _contract(tmp_path)
    first = build_replay_bundle(
        IDENTITY, START, FINISH, contract, reports_dir=tmp_path / "reports", logs_dir=logs,
        bundle_root=tmp_path / "one", manifest_digest_value="manifest"
    )
    second = build_replay_bundle(
        IDENTITY, START, FINISH, contract, reports_dir=tmp_path / "reports", logs_dir=logs,
        bundle_root=tmp_path / "two", manifest_digest_value="manifest"
    )

    assert first["bundle_digest"] == second["bundle_digest"]
    assert validate_replay_bundle(first["bundle_path"])["bundle_digest"] == first["bundle_digest"]


def test_coverage_records_first_last_and_maximum_gap(tmp_path):
    source = collect_source_rows(_logs(tmp_path), START, FINISH)
    rows = {kind: item["rows"] for kind, item in source["kinds"].items()}
    coverage = calculate_coverage(rows, START, FINISH)

    assert coverage["first_signal_timestamp"] == "2026-08-03T16:00:30Z"
    assert coverage["last_signal_timestamp"] == "2026-08-03T16:04:30Z"
    assert coverage["maximum_signal_gap_seconds"] == 150
    assert coverage["signal_row_count"] == 3
    assert coverage["xgboost_signal_join_rate"] == 1.0


def test_equal_timestamp_signal_rows_preserve_source_order(tmp_path):
    logs = tmp_path / "logs"
    _csv(
        logs / "live_signals.csv",
        ["ts", "symbol", "px", "p_meta", "rv_mean", "allow", "thr", "mode", "kinds_used", "signal_id"],
        [
            ["2026-08-03T16:01:00Z", "BTC", 100, 1, 0, 1, 0.5, "abs", "model", "z-first"],
            ["2026-08-03T16:01:00Z", "ETH", 100, 1, 0, 1, 0.5, "abs", "model", "a-second"],
        ],
    )

    source = collect_source_rows(logs, START, FINISH)

    assert [row["signal_id"] for row in source["kinds"]["signals"]["rows"]] == [
        "z-first",
        "a-second",
    ]


def test_xgboost_join_with_timestamp_mismatch_fails_coverage(tmp_path):
    logs = _logs(tmp_path)
    path = logs / "archive" / "xgboost_signal_shadow.csv"
    text = path.read_text(encoding="utf-8").replace("16:00:30Z", "16:00:31Z")
    path.write_text(text, encoding="utf-8")
    source = collect_source_rows(logs, START, FINISH)
    rows = {kind: item["rows"] for kind, item in source["kinds"].items()}

    coverage = calculate_coverage(rows, START, FINISH)

    assert coverage["xgboost_signal_join_count"] == 0
    assert coverage["coverage_passed"] is False


def test_reported_row_count_mismatch_marks_resolution_incomplete(tmp_path):
    result = resolve_historical_sources(
        IDENTITY,
        START,
        FINISH,
        logs_dir=_logs(tmp_path),
        bundle_root=tmp_path / "bundles",
        reported_row_counts={"xgboost": 999},
    )

    assert result["status"] == "incomplete"
    assert result["coverage"]["reported_row_counts_passed"] is False


def test_bundle_validation_rejects_self_consistent_wrong_row_inventory(tmp_path):
    result = build_replay_bundle(
        IDENTITY,
        START,
        FINISH,
        _contract(tmp_path),
        reports_dir=tmp_path / "reports",
        logs_dir=_logs(tmp_path),
        bundle_root=tmp_path / "bundles",
        manifest_digest_value="manifest",
    )
    path = Path(result["bundle_path"]) / "bundle_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["row_counts"]["signals"] += 1
    manifest["bundle_digest"] = bundle_digest(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReplayBundleError):
        validate_replay_bundle(path.parent)


def test_bundle_filters_model_logs_and_records_model_inventory(tmp_path):
    result = build_replay_bundle(
        IDENTITY, START, FINISH, _contract(tmp_path), reports_dir=tmp_path / "reports",
        logs_dir=_logs(tmp_path), bundle_root=tmp_path / "bundles", manifest_digest_value="manifest",
    )
    bundle = Path(result["bundle_path"])
    rows = list(csv.DictReader((bundle / "live_models_by_symbol.csv").open(encoding="utf-8")))

    assert [row["ts"] for row in rows] == ["2026-08-03T16:00:30Z", "2026-08-03T16:04:30Z"]
    assert result["manifest"]["model_output_duplicate_count"] == 1
    assert result["manifest"]["model_output_conflict_count"] == 0
    assert result["manifest"]["model_output_rows_by_symbol"] == {"BTC": 2, "__pooled__": 1}
    assert result["manifest"]["model_output_first_timestamp"] == "2026-08-03T16:00:30Z"
    assert result["manifest"]["model_output_last_timestamp"] == "2026-08-03T16:04:30Z"


def test_bundle_includes_validated_model_snapshot(tmp_path):
    result = build_replay_bundle(
        IDENTITY, START, FINISH, _contract(tmp_path), reports_dir=tmp_path / "reports",
        logs_dir=_logs(tmp_path), bundle_root=tmp_path / "bundles", manifest_digest_value="manifest",
        model_serving_snapshot_path=_model_snapshot(tmp_path),
    )
    bundle = Path(result["bundle_path"])

    assert (bundle / "model_serving_snapshot.json").is_file()
    validated = validate_replay_bundle(bundle)
    assert validated["manifest"]["model_serving_snapshot_digest"]


def test_new_bundle_preserves_source_bar_provenance_when_present(tmp_path):
    logs = _logs(tmp_path)
    _csv(
        logs / "live_models_by_symbol.csv",
        [
            "ts", "symbol", "source_bar_id", "source_bar_open_utc",
            "source_bar_close_utc", "feature_window_digest", "lstm_p",
        ],
        [[
            "2026-08-03T16:00:30Z", "BTC", "BTC:2026-08-03T16:00:00Z",
            "2026-08-03T15:55:00Z", "2026-08-03T16:00:00Z", "a" * 64, 0.6,
        ]],
    )
    result = build_replay_bundle(
        IDENTITY, START, FINISH, _contract(tmp_path),
        reports_dir=tmp_path / "reports", logs_dir=logs,
        bundle_root=tmp_path / "bundles", manifest_digest_value="manifest",
    )
    manifest = result["manifest"]
    assert manifest["bar_unique_model_health_evidence"] is True
    assert manifest["source_bar_provenance_columns"] == [
        "source_bar_id", "source_bar_open_utc", "source_bar_close_utc",
        "feature_window_digest",
    ]
    validated = validate_replay_bundle(result["bundle_path"])
    assert validated["manifest"]["bar_unique_model_health_evidence"] is True


def test_conflicting_model_probabilities_fail_bundle_capture(tmp_path):
    logs = _logs(tmp_path)
    with (logs / "live_models_by_symbol.csv").open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["2026-08-03T16:00:30Z", "BTC", 0.9, 0.5, 0.6])

    with pytest.raises(ReplayBundleError, match="conflicting timestamp/symbol"):
        build_replay_bundle(
            IDENTITY, START, FINISH, _contract(tmp_path), reports_dir=tmp_path / "reports",
            logs_dir=logs, bundle_root=tmp_path / "bundles", manifest_digest_value="manifest",
        )
