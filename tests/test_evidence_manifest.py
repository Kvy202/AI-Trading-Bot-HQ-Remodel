"""Phase 19 evidence registry and Phase 17/18 integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.evidence_manifest import (
    CLASSIFICATIONS,
    DEFAULT_OVERRIDES_PATH,
    EvidenceManifestError,
    build_evidence_manifest,
    evidence_manifest_digest,
    load_overrides,
    main,
    write_evidence_manifest,
)
from tools.offline_calibration_proposals import summarize_offline_calibration_proposals
from tools.offline_calibration_sweep import summarize_offline_calibration, write_json_summary


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _overrides(path: Path, entries: dict[str, Any] | None = None) -> Path:
    return _write_json(
        path,
        {"schema_version": 1, "overrides": entries or {}},
    )


def _unified(
    reports: Path,
    mode: str,
    timestamp: str,
    *,
    closed: int = 0,
    matched: int = 0,
    confirm: int = 0,
    reject: int = 0,
) -> Path:
    return _write_json(
        reports / f"matrix_{mode}_{timestamp}_unified.json",
        {
            "paper_pnl": {
                "closed_trade_count": closed,
                "total_pnl": float(closed),
                "average_pnl": None if not closed else 1.0,
                "win_rate": None if not closed else 1.0,
            },
            "trade_lineage": {"closed_trade_rows": closed},
            "xgboost_outcome": {
                "matched_closed_trade_count": matched,
                "would_confirm_matched_count": confirm,
                "would_reject_matched_count": reject,
            },
        },
    )


def _index(
    reports: Path,
    mode: str,
    timestamp: str,
    *,
    duration: int = 60,
    exit_status: int = 0,
    evidence_valid: bool = True,
    guard: bool = True,
    stale: int = 0,
    notes: list[str] | None = None,
    kinds: tuple[str, ...] = ("unified",),
) -> Path:
    report_paths = {
        kind: str(reports / f"matrix_{mode}_{timestamp}_{kind}.json")
        for kind in kinds
    }
    return _write_json(
        reports / f"matrix_index_{timestamp}.json",
        {
            "matrix_timestamp": timestamp,
            "requested_mode": mode,
            "duration_minutes": duration,
            "runs": [
                {
                    "mode": mode,
                    "run_started_utc": "2026-08-01T00:00:00Z",
                    "finished_at": "2026-08-01T01:00:00Z",
                    "duration_minutes": duration,
                    "exit_status": exit_status,
                    "stale_entry_guard_checked": guard,
                    "stale_entry_count": stale,
                    "stale_entry_signal_ids": (["stale-signal"] if stale else []),
                    "evidence_valid": evidence_valid,
                    "notes": notes or [],
                    "report_paths": report_paths,
                }
            ],
        },
    )


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return build_evidence_manifest(
        tmp_path / "reports",
        _overrides(tmp_path / "research" / "evidence_overrides.json"),
        generated_at="2026-08-04T00:00:00Z",
    )


def _run(manifest: dict[str, Any], identity: str) -> dict[str, Any]:
    return next(run for run in manifest["runs"] if run["identity"] == identity)


def test_matching_report_family_is_one_canonical_run(tmp_path):
    reports = tmp_path / "reports"
    timestamp = "20260801010101"
    _unified(reports, "baseline", timestamp, closed=1)
    _write_json(reports / f"matrix_baseline_{timestamp}_shadow_summary.json", {})
    _write_json(reports / f"matrix_baseline_{timestamp}_xgboost_audit.json", {})
    _index(
        reports,
        "baseline",
        timestamp,
        kinds=("unified", "shadow_summary", "xgboost_audit"),
    )

    manifest = _manifest(tmp_path)

    assert manifest["summary"]["total_runs"] == 1
    run = manifest["runs"][0]
    assert set(run["report_paths"]) == {"unified", "shadow_summary", "xgboost_audit"}
    assert run["classification"] == "valid_strategy_evidence"


@pytest.mark.parametrize(
    ("index_kwargs", "expected"),
    [
        ({"exit_status": 7}, "invalid_matrix_failure"),
        ({"stale": 1, "evidence_valid": False}, "contaminated_stale_signal"),
    ],
)
def test_failure_and_stale_classifications_fail_closed(tmp_path, index_kwargs, expected):
    reports = tmp_path / "reports"
    timestamp = "20260801020202"
    _unified(reports, "baseline", timestamp, closed=1)
    _index(reports, "baseline", timestamp, **index_kwargs)

    run = _manifest(tmp_path)["runs"][0]

    assert run["classification"] == expected
    assert run["include_in_strategy_aggregate"] is False
    assert run["include_in_safety_summary"] is False


@pytest.mark.parametrize(
    ("mode", "duration", "closed", "matched", "expected"),
    [
        ("baseline", 5, 0, 0, "valid_safety_only"),
        ("baseline", 60, 0, 0, "incomplete_no_outcomes"),
        ("baseline", 60, 2, 0, "valid_strategy_evidence"),
        ("xgboost_shadow_outcome", 60, 1, 0, "incomplete_no_outcomes"),
        ("xgboost_shadow_outcome", 60, 1, 1, "valid_strategy_evidence"),
    ],
)
def test_mode_aware_outcome_classification(
    tmp_path, mode, duration, closed, matched, expected
):
    reports = tmp_path / "reports"
    timestamp = "20260801030303"
    _unified(
        reports,
        mode,
        timestamp,
        closed=closed,
        matched=matched,
        reject=matched,
    )
    _index(reports, mode, timestamp, duration=duration)

    run = _manifest(tmp_path)["runs"][0]

    assert run["classification"] == expected
    assert run["include_in_strategy_aggregate"] is (
        expected == "valid_strategy_evidence"
    )
    assert run["include_in_safety_summary"] is True


def test_report_without_verified_index_is_unmatched_legacy_and_excluded(tmp_path):
    reports = tmp_path / "reports"
    path = _unified(reports, "baseline", "20260801040404", closed=3)

    manifest = _manifest(tmp_path)
    run = manifest["runs"][0]

    assert run["classification"] == "unverified_legacy"
    assert run["include_in_strategy_aggregate"] is False
    assert run["include_in_safety_summary"] is False
    assert manifest["inputs"]["unmatched_reports"][0]["path"] == str(path)


def test_reviewed_override_precedes_automatic_classification(tmp_path):
    reports = tmp_path / "reports"
    timestamp = "20260801050505"
    identity = f"baseline:{timestamp}"
    _unified(reports, "baseline", timestamp, closed=2)
    _index(reports, "baseline", timestamp)
    override = _overrides(
        tmp_path / "overrides.json",
        {
            identity: {
                "classification": "network_interrupted",
                "reason": "review confirmed a market connectivity interruption",
                "reviewed": True,
            }
        },
    )

    run = build_evidence_manifest(reports, override)["runs"][0]

    assert run["classification"] == "network_interrupted"
    assert run["classification_source"] == "reviewed_override"
    assert not run["include_in_strategy_aggregate"]
    assert not run["include_in_safety_summary"]


@pytest.mark.parametrize(
    "entry",
    [
        {
            "classification": "valid_strategy_evidence",
            "reason": "reviewed evidence",
            "reviewed": False,
        },
        {"classification": "unknown", "reason": "reviewed evidence", "reviewed": True},
        {
            "classification": "valid_strategy_evidence",
            "reason": "",
            "reviewed": True,
        },
    ],
)
def test_malformed_override_entries_are_rejected(tmp_path, entry):
    path = _overrides(
        tmp_path / "overrides.json", {"baseline:20260801060606": entry}
    )
    with pytest.raises(EvidenceManifestError):
        load_overrides(path)


def test_invalid_override_identity_and_settings_are_rejected(tmp_path):
    bad_identity = _overrides(
        tmp_path / "bad_identity.json",
        {
            "baseline:not-a-timestamp": {
                "classification": "unverified_legacy",
                "reason": "reviewed evidence",
                "reviewed": True,
            }
        },
    )
    settings = _overrides(
        tmp_path / "settings.json",
        {
            "baseline:20260801060606": {
                "classification": "unverified_legacy",
                "reason": "reviewed evidence",
                "reviewed": True,
                "threshold": 0.5,
            }
        },
    )
    with pytest.raises(EvidenceManifestError):
        load_overrides(bad_identity)
    with pytest.raises(EvidenceManifestError):
        load_overrides(settings)


def test_executable_override_reason_is_rejected(tmp_path):
    path = _overrides(
        tmp_path / "commands.json",
        {
            "baseline:20260801060606": {
                "classification": "unverified_legacy",
                "reason": "powershell -File tools/run_experiment_matrix.ps1",
                "reviewed": True,
            }
        },
    )

    with pytest.raises(EvidenceManifestError):
        load_overrides(path)


def test_manifest_notes_redact_environment_and_sensitive_values(tmp_path):
    reports = tmp_path / "reports"
    timestamp = "20260801060607"
    _unified(reports, "baseline", timestamp, closed=1)
    _index(
        reports,
        "baseline",
        timestamp,
        notes=[
            "PLACE_REAL_ORDERS=true api_key=not-for-output "
            "wallet=0x1111111111111111111111111111111111111111"
        ],
    )

    serialized = json.dumps(_manifest(tmp_path))

    assert "PLACE_REAL_ORDERS=true" not in serialized
    assert "not-for-output" not in serialized
    assert "0x1111111111111111111111111111111111111111" not in serialized
    assert "PLACE_REAL_ORDERS=[redacted]" in serialized


def test_malformed_index_is_recorded_without_hiding_unmatched_report(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "matrix_index_20260801070707.json").write_text("{bad", encoding="utf-8")
    _unified(reports, "baseline", "20260801070707", closed=1)

    manifest = _manifest(tmp_path)

    assert manifest["inputs"]["malformed_inputs"]
    assert manifest["inputs"]["unmatched_reports"]
    assert manifest["runs"][0]["classification"] == "unverified_legacy"


def test_output_ordering_and_digest_are_deterministic(tmp_path):
    reports = tmp_path / "reports"
    for mode, timestamp in (
        ("xgboost_shadow_outcome", "20260803000000"),
        ("baseline", "20260802000000"),
        ("advanced_risk_shadow", "20260803000000"),
    ):
        _unified(reports, mode, timestamp, closed=1, matched=1, reject=1)
        _index(reports, mode, timestamp)

    first = _manifest(tmp_path)
    second = _manifest(tmp_path)

    assert [run["identity"] for run in first["runs"]] == [
        "baseline:20260802000000",
        "advanced_risk_shadow:20260803000000",
        "xgboost_shadow_outcome:20260803000000",
    ]
    assert evidence_manifest_digest(first) == evidence_manifest_digest(second)


def test_digest_ignores_generated_at_and_absolute_path_differences(tmp_path):
    manifests = []
    for folder in (tmp_path / "windows", tmp_path / "linux"):
        reports = folder / "reports"
        timestamp = "20260801080808"
        _unified(reports, "baseline", timestamp, closed=1)
        _index(reports, "baseline", timestamp)
        manifests.append(_manifest(folder))
    manifests[0]["generated_at"] = "first"
    manifests[1]["generated_at"] = "second"
    manifests[0]["inputs"]["reports_dir"] = r"C:\absolute\reports"
    manifests[1]["inputs"]["reports_dir"] = "/absolute/reports"

    assert evidence_manifest_digest(manifests[0]) == evidence_manifest_digest(manifests[1])


def test_phase17_aggregates_only_strategy_runs_and_records_exclusions(tmp_path):
    reports = tmp_path / "reports"
    overrides = _overrides(tmp_path / "overrides.json")
    good_ts = "20260801090909"
    stale_ts = "20260801101010"
    _unified(reports, "baseline", good_ts, closed=2)
    _index(reports, "baseline", good_ts)
    _unified(reports, "baseline", stale_ts, closed=99)
    _index(reports, "baseline", stale_ts, stale=1, evidence_valid=False)

    summary = summarize_offline_calibration(
        reports, tmp_path / "logs", overrides_path=overrides
    )

    assert summary["baseline_cross_run_summary"]["total_closed_trades"] == 2
    assert summary["evidence_runs_strategy_included"] == 1
    assert summary["evidence_exclusions"] == [
        {
            "identity": f"baseline:{stale_ts}",
            "classification": "contaminated_stale_signal",
            "reason": "stale-entry evidence detected (stale_entry_count=1)",
        }
    ]
    assert any(
        item["identity"] == f"baseline:{stale_ts}"
        for item in summary["input_inventory"]["incomplete_reports_skipped"]
    )


def test_phase18_rejects_stale_phase17_digest_and_reconstructs(tmp_path):
    reports = tmp_path / "reports"
    overrides = _overrides(tmp_path / "overrides.json")
    phase17 = summarize_offline_calibration(
        reports, tmp_path / "logs", overrides_path=overrides
    )
    phase17["evidence_manifest_digest"] = "0" * 64
    write_json_summary(phase17, reports / "offline_calibration_sweep.json")

    proposal = summarize_offline_calibration_proposals(
        reports, tmp_path / "logs", overrides_path=overrides
    )

    status = proposal["input_evidence_inventory"]["phase17_report_status"]
    assert status["status"] == "stale"
    assert proposal["phase17_manifest_digest_match"] is False
    assert proposal["input_evidence_inventory"]["preferred_evidence_source"] == (
        "reconstructed_from_reports_and_current_logs"
    )


def test_phase18_reuses_matching_phase17_digest(tmp_path):
    reports = tmp_path / "reports"
    overrides = _overrides(tmp_path / "overrides.json")
    phase17 = summarize_offline_calibration(
        reports, tmp_path / "logs", overrides_path=overrides
    )
    write_json_summary(phase17, reports / "offline_calibration_sweep.json")

    proposal = summarize_offline_calibration_proposals(
        reports, tmp_path / "logs", overrides_path=overrides
    )

    assert proposal["phase17_manifest_digest_match"] is True
    assert proposal["input_evidence_inventory"]["phase17_report_status"]["status"] == "ok"
    assert proposal["input_evidence_inventory"]["preferred_evidence_source"] == (
        "phase17_preferred_report"
    )


def test_manifest_write_changes_no_runtime_settings_or_artifacts(tmp_path):
    runtime = _write_json(tmp_path / "config" / "run.json", {"trade_mode": "paper"})
    env = tmp_path / ".env"
    env.write_text("PLACE_REAL_ORDERS=false\n", encoding="utf-8")
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"unchanged-model")
    before = (runtime.read_bytes(), env.read_bytes(), artifact.read_bytes())

    manifest = _manifest(tmp_path)
    out = write_evidence_manifest(manifest, tmp_path / "reports" / "evidence_manifest.json")

    assert out.exists()
    assert before == (runtime.read_bytes(), env.read_bytes(), artifact.read_bytes())
    serialized = out.read_text(encoding="utf-8")
    assert "command_run" not in serialized
    assert ".ps1" not in serialized
    assert "--place-real-orders" not in serialized.lower()


def test_json_cli_stdout_is_one_machine_readable_document(tmp_path, capsys):
    reports = tmp_path / "reports"
    overrides = _overrides(tmp_path / "overrides.json")
    out = reports / "evidence_manifest.json"

    assert main(
        [
            "--reports-dir",
            str(reports),
            "--overrides",
            str(overrides),
            "--json-out",
            str(out),
            "--json",
        ]
    ) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["policy"] == {
        "fail_closed_for_legacy": True,
        "paper_only": True,
        "real_orders_allowed": False,
    }
    assert out.exists()


def test_known_reviewed_override_identities_have_expected_classifications():
    overrides = load_overrides(DEFAULT_OVERRIDES_PATH)
    expected = {
        "baseline:20260730122136": "contaminated_stale_signal",
        "baseline:20260730162028": "unverified_legacy",
        "xgboost_shadow_outcome:20260801173114": "contaminated_stale_signal",
        "baseline:20260801192203": "valid_safety_only",
        "baseline:20260802112352": "incomplete_no_outcomes",
        "xgboost_shadow_outcome:20260802124342": "network_interrupted",
        "xgboost_shadow_outcome:20260803161821": "valid_strategy_evidence",
    }

    assert set(CLASSIFICATIONS) == {
        "valid_strategy_evidence",
        "valid_safety_only",
        "incomplete_no_outcomes",
        "contaminated_stale_signal",
        "network_interrupted",
        "invalid_matrix_failure",
        "unverified_legacy",
    }
    assert {key: value["classification"] for key, value in overrides.items()} == expected
