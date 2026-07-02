"""Verify the Phase 10 Advanced Risk shadow evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

THIS = Path(__file__).resolve()
BASE_DIR = THIS.parents[1] if THIS.parent.name == "tools" else THIS.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from runtime.loader import apply_run_config

    apply_run_config(BASE_DIR)
except Exception:
    pass

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=True)
except Exception:
    pass

from ml_optional.advanced_risk import (  # noqa: E402
    ADVANCED_RISK_SHADOW_COLS,
    AdvancedRiskSettings,
    evaluate_advanced_risk,
)

SHADOW_LOG = BASE_DIR / "logs" / "advanced_risk_shadow.csv"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _ensure_header(path: Path, cols: Iterable[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(cols))


def _append_aligned_row(path: Path, cols: list[str], row: Dict[str, Any]) -> None:
    _ensure_header(path, cols)
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([row.get(c, "") for c in cols])


def _settings(**overrides: Any) -> AdvancedRiskSettings:
    values = {
        "advanced_risk_enabled": True,
        "advanced_risk_active": False,
        "max_daily_loss_pct": 3.0,
        "max_consecutive_losses": 3,
        "max_open_positions": 1,
        "max_symbol_exposure_pct": 100.0,
        "volatility_guard_mult": 2.0,
        "paper_mode": True,
        "place_real_orders": False,
    }
    values.update(overrides)
    return AdvancedRiskSettings(**values)


def _context(**overrides: Any) -> Dict[str, Any]:
    values = {
        "timestamp": _ts(),
        "symbol": "VERIFY",
        "side": "long",
        "p_meta": 0.72,
        "price": 100.0,
        "rv_mean": 0.010,
        "volatility": 0.020,
        "open_positions_count": 0,
        "symbol_exposure": 10.0,
        "daily_realized_pnl": 0.0,
        "daily_loss_pct": 0.0,
        "consecutive_losses": 0,
        "recent_trade_count": 0,
    }
    values.update(overrides)
    return values


def simulate_case(
    case: str,
    *,
    context: Dict[str, Any],
    settings: AdvancedRiskSettings,
) -> Dict[str, Any]:
    decision = evaluate_advanced_risk(context, settings)
    row = decision.to_log_row(context)
    row["case"] = case
    return row


def build_verification_rows() -> List[Dict[str, Any]]:
    return [
        simulate_case(
            "disabled_flag",
            context=_context(),
            settings=_settings(advanced_risk_enabled=False),
        ),
        simulate_case(
            "normal_risk",
            context=_context(),
            settings=_settings(),
        ),
        simulate_case(
            "daily_loss_would_block",
            context=_context(daily_realized_pnl=-4.0, daily_loss_pct=4.0),
            settings=_settings(),
        ),
        simulate_case(
            "consecutive_losses_would_block",
            context=_context(consecutive_losses=3),
            settings=_settings(),
        ),
        simulate_case(
            "max_open_positions_would_block",
            context=_context(open_positions_count=1),
            settings=_settings(),
        ),
        simulate_case(
            "volatility_guard_would_block",
            context=_context(rv_mean=0.050, volatility=0.020),
            settings=_settings(),
        ),
        simulate_case(
            "active_false_no_actual_block",
            context=_context(daily_realized_pnl=-4.0, daily_loss_pct=4.0),
            settings=_settings(advanced_risk_active=False),
        ),
        simulate_case(
            "active_true_still_shadow_only",
            context=_context(daily_realized_pnl=-4.0, daily_loss_pct=4.0),
            settings=_settings(advanced_risk_active=True),
        ),
        simulate_case(
            "missing_context_does_not_crash",
            context={"timestamp": _ts(), "symbol": "VERIFY"},
            settings=_settings(),
        ),
    ]


def validate_required_cases(rows: List[Dict[str, Any]]) -> List[str]:
    by_case = {row["case"]: row for row in rows}
    errors: List[str] = []

    def check(case: str, condition: bool, detail: str) -> None:
        if case not in by_case:
            errors.append(f"{case}: missing")
        elif not condition:
            errors.append(f"{case}: {detail}")

    disabled = by_case.get("disabled_flag", {})
    check(
        "disabled_flag",
        disabled.get("risk_status") == "disabled"
        and disabled.get("would_block") == 0
        and disabled.get("actually_blocked") == 0,
        f"expected disabled no-block got {disabled}",
    )

    normal = by_case.get("normal_risk", {})
    check(
        "normal_risk",
        normal.get("risk_status") == "normal"
        and normal.get("risk_score") == 0.0
        and normal.get("would_block") == 0,
        f"expected normal no-block got {normal}",
    )

    daily = by_case.get("daily_loss_would_block", {})
    check(
        "daily_loss_would_block",
        daily.get("would_block") == 1
        and daily.get("would_pause") == 1
        and daily.get("top_reason") == "daily_loss_pct_limit",
        f"expected daily-loss would-block got {daily}",
    )

    losses = by_case.get("consecutive_losses_would_block", {})
    check(
        "consecutive_losses_would_block",
        losses.get("would_block") == 1
        and losses.get("would_pause") == 1
        and losses.get("top_reason") == "consecutive_losses_limit",
        f"expected consecutive-loss would-block got {losses}",
    )

    open_positions = by_case.get("max_open_positions_would_block", {})
    check(
        "max_open_positions_would_block",
        open_positions.get("would_block") == 1
        and open_positions.get("top_reason") == "max_open_positions_limit",
        f"expected max-open-positions would-block got {open_positions}",
    )

    volatility = by_case.get("volatility_guard_would_block", {})
    check(
        "volatility_guard_would_block",
        volatility.get("would_block") == 1
        and volatility.get("would_reduce_size") == 1
        and volatility.get("volatility_guard_triggered") == 1,
        f"expected volatility would-block got {volatility}",
    )

    inactive = by_case.get("active_false_no_actual_block", {})
    check(
        "active_false_no_actual_block",
        inactive.get("would_block") == 1
        and inactive.get("actually_blocked") == 0
        and inactive.get("paper_only_guard") == "inactive",
        f"expected inactive shadow-only got {inactive}",
    )

    active = by_case.get("active_true_still_shadow_only", {})
    check(
        "active_true_still_shadow_only",
        active.get("would_block") == 1
        and active.get("actually_blocked") == 0
        and active.get("paper_only_guard") == "phase10_shadow_only",
        f"expected active flag still shadow-only got {active}",
    )

    missing = by_case.get("missing_context_does_not_crash", {})
    check(
        "missing_context_does_not_crash",
        missing.get("risk_status") == "context_missing"
        and missing.get("actually_blocked") == 0,
        f"expected context_missing no-actual-block got {missing}",
    )

    return errors


def format_verification_summary(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "Advanced Risk Shadow Verification",
        "",
        "Cases:",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['case']}: risk_status={row['risk_status']} "
            f"risk_score={row['risk_score']} "
            f"would_block={row['would_block']} "
            f"actually_blocked={row['actually_blocked']} "
            f"would_pause={row['would_pause']} "
            f"actually_paused={row['actually_paused']} "
            f"would_reduce_size={row['would_reduce_size']} "
            f"actually_reduced={row['actually_reduced']} "
            f"top_reason={row['top_reason']} "
            f"paper_only_guard={row['paper_only_guard']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Verify Phase 10 Advanced Risk shadow evaluator")
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print verification rows as JSON")
    args = parser.parse_args(argv)

    rows = build_verification_rows()
    errors = validate_required_cases(rows)
    if errors:
        raise SystemExit("ERROR verification failed:\n" + "\n".join(f"- {err}" for err in errors))

    if args.write_shadow_row:
        for row in rows:
            _append_aligned_row(SHADOW_LOG, ADVANCED_RISK_SHADOW_COLS, row)
        print(f"shadow_log_written={SHADOW_LOG}")

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_verification_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
