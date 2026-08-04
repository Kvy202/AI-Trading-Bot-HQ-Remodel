"""Phase 18.1 regression coverage for stale signals and matrix evidence."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tools.live_executor as live_executor


ROOT = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = ROOT / "tools" / "run_experiment_matrix.ps1"
SIGNAL_HEADER = [
    "ts",
    "symbol",
    "price",
    "p_meta",
    "rv_mean",
    "allow",
    "thr",
    "mode",
    "kinds_used",
    "unused",
    "signal_id",
]
TRADE_HEADER = "ts,symbol,side,price,qty,reason,mode,order_id,signal_id"


def _signal_row(ts: str, symbol: str = "BTC", signal_id: str = "sig") -> list[str]:
    return [ts, symbol, "100", "0.9", "0.01", "1", "0.5", "abs", "model", "", signal_id]


def _write_signals(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SIGNAL_HEADER)
        writer.writerows(rows)


def _append_signal(path: Path, row: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(row)


def _timestamps(signals: list[live_executor.SignalRow]) -> list[str]:
    return [signal.ts for signal in signals]


def test_equal_timestamp_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(path, [_signal_row("2026-08-01 10:02:00")])

    result = live_executor.read_recent_signals(
        path, {"BTC": "2026-08-01 10:02:00"}, 100
    )

    assert result == []


def test_older_timestamp_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(path, [_signal_row("2026-08-01 10:01:00")])

    result = live_executor.read_recent_signals(
        path, {"BTC": "2026-08-01 10:02:00"}, 100
    )

    assert result == []


def test_multiple_older_rows_are_not_replayed_one_by_one(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(
        path,
        [
            _signal_row("2026-08-01 10:00:00", signal_id="sig-0"),
            _signal_row("2026-08-01 10:01:00", signal_id="sig-1"),
            _signal_row("2026-08-01 10:02:00", signal_id="sig-2"),
        ],
    )
    high_water = {"BTC": "2026-08-01 10:02:00"}

    assert live_executor.read_recent_signals(path, high_water, 100) == []
    assert live_executor.read_recent_signals(path, high_water, 100) == []
    assert live_executor.read_recent_signals(path, high_water, 100) == []


def test_repeated_polling_without_append_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(path, [_signal_row("2026-08-01 10:02:00")])
    high_water = {"BTC": "2026-08-01 10:02:00"}

    for _ in range(5):
        assert live_executor.read_recent_signals(path, high_water, 100) == []


def test_one_appended_newer_row_is_returned_once(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(
        path,
        [
            _signal_row("2026-08-01 10:00:00"),
            _signal_row("2026-08-01 10:01:00"),
            _signal_row("2026-08-01 10:02:00"),
        ],
    )
    high_water = {"BTC": "2026-08-01 10:02:00"}
    _append_signal(path, _signal_row("2026-08-01 10:03:00", signal_id="sig-new"))

    result = live_executor.read_recent_signals(path, high_water, 100)
    assert _timestamps(result) == ["2026-08-01 10:03:00"]
    assert result[0].signal_id == "sig-new"

    high_water[result[0].symbol] = result[0].ts
    assert live_executor.read_recent_signals(path, high_water, 100) == []


def test_multiple_symbols_have_independent_high_water_marks(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(
        path,
        [
            _signal_row("2026-08-01 10:02:00", "BTC", "btc-2"),
            _signal_row("2026-08-01 10:05:00", "ETH", "eth-5"),
            _signal_row("2026-08-01 10:03:00", "BTC", "btc-3"),
            _signal_row("2026-08-01 10:04:00", "ETH", "eth-old"),
        ],
    )

    result = live_executor.read_recent_signals(
        path,
        {"BTC": "2026-08-01 10:02:00", "ETH": "2026-08-01 10:05:00"},
        100,
    )

    assert [(signal.symbol, signal.ts) for signal in result] == [
        ("BTC", "2026-08-01 10:03:00")
    ]


def test_out_of_order_appended_old_row_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(path, [_signal_row("2026-08-01 10:03:00", signal_id="newest")])
    high_water = {"BTC": "2026-08-01 10:03:00"}
    _append_signal(path, _signal_row("2026-08-01 10:01:00", signal_id="late-old"))

    assert live_executor.read_recent_signals(path, high_water, 100) == []


def test_malformed_timestamp_fallback_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(
        path,
        [
            _signal_row("bad-ts-001", signal_id="one"),
            _signal_row("bad-ts-003", signal_id="three"),
            _signal_row("bad-ts-002", signal_id="two"),
        ],
    )
    high_water = {"BTC": "bad-ts-001"}

    first = live_executor.read_recent_signals(path, high_water, 100)
    second = live_executor.read_recent_signals(path, high_water, 100)

    assert _timestamps(first) == ["bad-ts-003"]
    assert _timestamps(second) == ["bad-ts-003"]
    high_water["BTC"] = first[0].ts
    assert live_executor.read_recent_signals(path, high_water, 100) == []


def test_startup_drain_prevents_historical_replay(tmp_path: Path) -> None:
    path = tmp_path / "live_signals.csv"
    _write_signals(
        path,
        [
            _signal_row("2026-08-01 10:00:00", "BTC"),
            _signal_row("2026-08-01 10:05:00", "ETH"),
            _signal_row("2026-08-01 10:02:00", "BTC"),
        ],
    )

    high_water = live_executor.initialize_signal_high_water_marks(path, 100)

    assert high_water == {
        "BTC": "2026-08-01 10:02:00",
        "ETH": "2026-08-01 10:05:00",
    }
    assert live_executor.read_recent_signals(path, high_water, 100) == []
    _append_signal(path, _signal_row("2026-08-01 10:03:00", "BTC", "btc-new"))
    assert _timestamps(live_executor.read_recent_signals(path, high_water, 100)) == [
        "2026-08-01 10:03:00"
    ]
    assert "signal_drain: initialized high-water marks" in MATRIX_SCRIPT.with_name(
        "live_executor.py"
    ).read_text(encoding="utf-8")


def _powershell_exe() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _fake_report_tool() -> str:
    return "\n".join(
        [
            "import json",
            "import sys",
            "from pathlib import Path",
            "out = Path(sys.argv[sys.argv.index('--json-out') + 1])",
            "out.parent.mkdir(parents=True, exist_ok=True)",
            "out.write_text(json.dumps({'ok': True}), encoding='utf-8')",
        ]
    )


def _make_fake_matrix_repo(tmp_path: Path, trade_rows: list[str]) -> Path:
    root = tmp_path / "matrix_repo"
    tools = root / "tools"
    logs = root / "logs"
    tools.mkdir(parents=True)
    logs.mkdir()
    (root / "reports").mkdir()
    (root / "config").mkdir()
    (root / "v2").mkdir()
    (root / "research").mkdir()
    shutil.copy2(MATRIX_SCRIPT, tools / MATRIX_SCRIPT.name)
    for helper in ("replay_contract.py", "replay_bundle.py", "evidence_manifest.py"):
        shutil.copy2(ROOT / "tools" / helper, tools / helper)
    (tools / "live_executor.py").write_text("# deterministic matrix fixture\n", encoding="utf-8")
    (root / "v2" / "risk_controls.py").write_text("# deterministic matrix fixture\n", encoding="utf-8")
    (root / "config" / "run.json").write_text("{}", encoding="utf-8")
    (root / "research" / "evidence_overrides.json").write_text(
        '{"schema_version":1,"overrides":{}}', encoding="utf-8"
    )
    (tools / "apply_experiment_mode.ps1").write_text(
        "function Get-ExperimentModeOverrides { return @{} }\n",
        encoding="utf-8",
    )

    csv_text = "`n".join([TRADE_HEADER, *trade_rows])
    child = "\n".join(
        [
            "param(",
            "  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,",
            "  [switch]$FreshShadowLogs,",
            "  [switch]$FreshPaperLogs",
            ")",
            "$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            "$root = (Resolve-Path (Join-Path $scriptDir '..')).Path",
            "$logsDir = Join-Path $root 'logs'",
            f'$csv = "{csv_text}"',
            "$csv | Set-Content -Path (Join-Path $logsDir 'trades_paper_fixture.csv') -Encoding UTF8",
            "exit 0",
        ]
    )
    (tools / "run_combined_shadow_paper_test.ps1").write_text(child, encoding="utf-8")

    for name in (
        "experimental_shadow_report.py",
        "unified_experimental_report.py",
        "audit_xgboost_rejections.py",
    ):
        (tools / name).write_text(_fake_report_tool(), encoding="utf-8")
    return tools / MATRIX_SCRIPT.name


def _run_fake_matrix(script: Path) -> subprocess.CompletedProcess[str]:
    executable = _powershell_exe()
    if executable is None:
        pytest.skip("PowerShell is not available")
    return subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Mode",
            "combined_shadow",
            "-Minutes",
            "5",
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _matrix_run(script: Path) -> dict:
    indexes = sorted((script.parents[1] / "reports").glob("matrix_index_*.json"))
    assert indexes
    index = json.loads(indexes[-1].read_text(encoding="utf-8-sig"))
    return index["runs"][0]


def _trade_row(ts: str, action: str, signal_id: str) -> str:
    return f"{ts},BTC,{action},100,1,fixture,PAPER,paper-{signal_id},{signal_id}"


def test_matrix_stale_entry_detector_finds_entry_before_run_start(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path, [_trade_row("2000-01-01 00:00:00", "BUY", "stale-buy")]
    )

    result = _run_fake_matrix(script)
    run = _matrix_run(script)

    assert result.returncode == 1, result.stdout + result.stderr
    assert run["stale_entry_guard_checked"] is True
    assert run["stale_entry_count"] == 1


def test_matrix_detector_ignores_valid_post_start_entry(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path, [_trade_row("2099-01-01 00:00:00+00:00", "SELL_SHORT", "valid-short")]
    )

    result = _run_fake_matrix(script)
    run = _matrix_run(script)

    assert result.returncode == 0, result.stdout + result.stderr
    assert run["stale_entry_guard_checked"] is True
    assert run["stale_entry_count"] == 0


def test_matrix_detector_ignores_exit_only_rows(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path,
        [
            _trade_row("2000-01-01 00:00:00", "SELL", "exit-long"),
            _trade_row("2000-01-01 00:00:01", "BUY_TO_COVER", "exit-short"),
        ],
    )

    result = _run_fake_matrix(script)
    run = _matrix_run(script)

    assert result.returncode == 0, result.stdout + result.stderr
    assert run["stale_entry_count"] == 0


def test_contaminated_matrix_run_sets_evidence_invalid(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path, [_trade_row("2000-01-01 00:00:00", "SELL_SHORT", "stale-short")]
    )

    _run_fake_matrix(script)
    run = _matrix_run(script)

    assert run["evidence_valid"] is False
    assert "stale_signal_replay_or_prestart_entry_detected" in run["notes"]
    assert all(Path(path).exists() for path in run["report_paths"].values())


def test_valid_matrix_run_sets_evidence_valid(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path, [_trade_row("2099-01-01 00:00:00Z", "BUY", "valid-buy")]
    )

    result = _run_fake_matrix(script)
    run = _matrix_run(script)

    assert result.returncode == 0, result.stdout + result.stderr
    assert run["run_started_utc"]
    assert run["evidence_valid"] is True


def test_signal_ids_are_in_contamination_diagnostics(tmp_path: Path) -> None:
    script = _make_fake_matrix_repo(
        tmp_path, [_trade_row("2000-01-01 00:00:00", "BUY", "diagnostic-id")]
    )

    result = _run_fake_matrix(script)
    run = _matrix_run(script)

    assert "signal_id=diagnostic-id" in result.stdout
    assert run["stale_entry_signal_ids"] == ["diagnostic-id"]


def test_matrix_guard_is_shared_by_generic_and_child_runbook_modes() -> None:
    text = MATRIX_SCRIPT.read_text(encoding="utf-8")
    branch_start = text.index("if ($plan.kind -eq 'generic')")
    generic_call = text.index("Invoke-GenericPaperMode", branch_start)
    child_call = text.index("Invoke-ChildRunbook", generic_call)
    guard_call = text.index("Test-MatrixStalePaperEntries", child_call)
    report_call = text.index("Write-MatrixReports", guard_call)

    assert generic_call < child_call < guard_call < report_call


def test_existing_paper_live_safety_preflight_remains_enabled() -> None:
    text = MATRIX_SCRIPT.read_text(encoding="utf-8")

    for required in (
        "$overrides['LIVE_TRADING'] = 'false'",
        "$overrides['PAPER_TRADING'] = 'true'",
        "$overrides['PLACE_REAL_ORDERS'] = 'false'",
        "$overrides['CONFIRM_LIVE_TRADING'] = ''",
        "resolve_trading_mode",
        "live_requested",
        "d.place_real_orders",
        "production_detected",
        "hyperliquid_mainnet_selected",
        "REFUSING: live/mainnet/real-order mode detected",
    ):
        assert required in text
