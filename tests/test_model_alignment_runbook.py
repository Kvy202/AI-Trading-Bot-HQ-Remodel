from __future__ import annotations

from pathlib import Path


RUNBOOK = Path("tools/run_model_alignment_shadow.ps1")


def test_runbook_starts_only_alignment_tool_and_has_safe_child_environment():
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "tools/model_alignment_shadow.py" in source
    assert "live_writer.py" not in source
    assert "live_executor.py" not in source
    for line in (
        '$env:LIVE_TRADING = "false"', '$env:PAPER_TRADING = "true"',
        '$env:LIVE_MODE = "false"', '$env:EXEC_PAPER = "true"',
        '$env:PLACE_REAL_ORDERS = "false"', '$env:DL_TIMEFRAME = "5m"',
        '$env:DL_SEQ_LEN = "64"',
    ):
        assert line in source


def test_runbook_defaults_and_maximum_are_bounded():
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "[ValidateRange(3, 120)]" in source
    assert "[int]$UniqueBars = 3" in source
    assert "[switch]$DryRun" in source
    assert 'if ($FreshLogs)' in source and '"--fresh-logs"' in source


def test_runbook_has_no_order_or_activation_command():
    source = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "start-process" not in source
    assert "place_order" not in source
    assert "run_experiment_matrix" not in source
    assert "live_signals.csv" not in source
    assert "trades_paper" not in source
    assert "trades_closed" not in source
