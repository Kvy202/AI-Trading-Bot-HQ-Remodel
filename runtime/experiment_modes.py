"""Named experimental mode definitions.

This module is intentionally read-only: it builds environment dictionaries for
callers, but it does not write ``os.environ`` or touch ``.env`` files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

EnvValue = bool | str

SAFE_PAPER_OVERRIDES: Mapping[str, EnvValue] = {
    "LIVE_TRADING": False,
    "PAPER_TRADING": True,
    "LIVE_MODE": False,
    "EXEC_PAPER": True,
    "PLACE_REAL_ORDERS": False,
}

EXPERIMENTAL_BASELINE_OVERRIDES: Mapping[str, EnvValue] = {
    "USE_ISOLATION_FOREST": False,
    "ISOLATION_FOREST_BLOCKING": False,
    "USE_XGBOOST_SIGNAL": False,
    "XGBOOST_SIGNAL_BLOCKING": False,
    "USE_SURVIVAL_EXIT": False,
    "SURVIVAL_EXIT_ACTIVE": False,
    "USE_ADVANCED_RISK": False,
    "ADVANCED_RISK_ACTIVE": False,
}


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    description: str
    overrides: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": dict(self.overrides),
        }


def _normalize_env_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    raw = str(value)
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered
    return raw


def _normalized_overrides(overrides: Mapping[str, EnvValue]) -> Mapping[str, str]:
    return MappingProxyType({key: _normalize_env_value(value) for key, value in overrides.items()})


def _mode(name: str, description: str, overrides: Mapping[str, EnvValue] | None = None) -> ExperimentMode:
    values: dict[str, EnvValue] = {}
    values.update(SAFE_PAPER_OVERRIDES)
    values.update(EXPERIMENTAL_BASELINE_OVERRIDES)
    if overrides:
        values.update(overrides)
    return ExperimentMode(name=name, description=description, overrides=_normalized_overrides(values))


EXPERIMENT_MODES: Mapping[str, ExperimentMode] = MappingProxyType(
    {
        "baseline": _mode(
            "baseline",
            "Paper-safe baseline with every experimental system disabled.",
        ),
        "iforest_shadow": _mode(
            "iforest_shadow",
            "Isolation Forest enabled in shadow mode only.",
            {
                "USE_ISOLATION_FOREST": True,
                "ISOLATION_FOREST_BLOCKING": False,
            },
        ),
        "iforest_blocking": _mode(
            "iforest_blocking",
            "Isolation Forest enabled with its blocking gate active for paper tests.",
            {
                "USE_ISOLATION_FOREST": True,
                "ISOLATION_FOREST_BLOCKING": True,
            },
        ),
        "xgboost_shadow": _mode(
            "xgboost_shadow",
            "XGBoost signal confirmer enabled in shadow mode only.",
            {
                "USE_XGBOOST_SIGNAL": True,
                "XGBOOST_SIGNAL_BLOCKING": False,
            },
        ),
        "xgboost_blocking": _mode(
            "xgboost_blocking",
            "XGBoost signal confirmer enabled with its blocking gate active for paper tests.",
            {
                "USE_XGBOOST_SIGNAL": True,
                "XGBOOST_SIGNAL_BLOCKING": True,
            },
        ),
        "xgboost_shadow_outcome": _mode(
            "xgboost_shadow_outcome",
            "Same flags as xgboost_shadow, named for shadow-outcome paper audits.",
            {
                "USE_XGBOOST_SIGNAL": True,
                "XGBOOST_SIGNAL_BLOCKING": False,
            },
        ),
        "survival_shadow": _mode(
            "survival_shadow",
            "Survival exit model enabled for shadow observations only.",
            {
                "USE_SURVIVAL_EXIT": True,
                "SURVIVAL_EXIT_ACTIVE": False,
            },
        ),
        "survival_active_placeholder": _mode(
            "survival_active_placeholder",
            "Survival exit placeholder; active exit behavior remains disabled until Phase 9.",
            {
                "USE_SURVIVAL_EXIT": True,
                "SURVIVAL_EXIT_ACTIVE": False,
            },
        ),
        "advanced_risk_shadow_placeholder": _mode(
            "advanced_risk_shadow_placeholder",
            "Advanced Risk placeholder enabled for shadow configuration only.",
            {
                "USE_ADVANCED_RISK": True,
                "ADVANCED_RISK_ACTIVE": False,
            },
        ),
        "advanced_risk_active_placeholder": _mode(
            "advanced_risk_active_placeholder",
            "Advanced Risk active placeholder; active behavior remains disabled in Phase 10.",
            {
                "USE_ADVANCED_RISK": True,
                "ADVANCED_RISK_ACTIVE": False,
            },
        ),
        "combined_shadow": _mode(
            "combined_shadow",
            "Isolation Forest, XGBoost, Survival Exit, and Advanced Risk all enabled in non-blocking shadow mode.",
            {
                "USE_ISOLATION_FOREST": True,
                "ISOLATION_FOREST_BLOCKING": False,
                "USE_XGBOOST_SIGNAL": True,
                "XGBOOST_SIGNAL_BLOCKING": False,
                "USE_SURVIVAL_EXIT": True,
                "SURVIVAL_EXIT_ACTIVE": False,
                "USE_ADVANCED_RISK": True,
                "ADVANCED_RISK_ACTIVE": False,
            },
        ),
        "combined_paper": _mode(
            "combined_paper",
            "Paper-safe combined observation mode; all experimental systems are non-blocking.",
            {
                "USE_ISOLATION_FOREST": True,
                "ISOLATION_FOREST_BLOCKING": False,
                "USE_XGBOOST_SIGNAL": True,
                "XGBOOST_SIGNAL_BLOCKING": False,
                "USE_SURVIVAL_EXIT": True,
                "SURVIVAL_EXIT_ACTIVE": False,
                "USE_ADVANCED_RISK": True,
                "ADVANCED_RISK_ACTIVE": False,
            },
        ),
    }
)


def list_experiment_modes() -> list[str]:
    """Return supported experiment mode names in stable display order."""
    return list(EXPERIMENT_MODES)


def _unknown_mode_error(name: object) -> ValueError:
    supported = ", ".join(list_experiment_modes())
    return ValueError(f"Unknown experiment mode {name!r}. Supported modes: {supported}")


def validate_experiment_mode(name: str) -> str:
    """Return the normalized mode name or raise ``ValueError``."""
    normalized = str(name).strip().lower()
    if normalized not in EXPERIMENT_MODES:
        raise _unknown_mode_error(name)
    return normalized


def get_experiment_mode(name: str) -> ExperimentMode:
    """Return a mode definition by name."""
    return EXPERIMENT_MODES[validate_experiment_mode(name)]


def apply_experiment_mode(name: str, base_env: Mapping[str, object] | None = None) -> dict[str, str]:
    """Return a new environment dict with the mode overrides applied.

    ``base_env=None`` copies the current process environment without mutating it.
    Pass ``base_env={}`` when only the mode's override lines are needed.
    """
    source = os.environ if base_env is None else base_env
    env = {str(key): _normalize_env_value(value) for key, value in source.items()}
    env.update(get_experiment_mode(name).overrides)
    return env


def describe_experiment_mode(name: str) -> str:
    """Return a human-readable description of a mode and its overrides."""
    mode = get_experiment_mode(name)
    lines = [
        f"Mode: {mode.name}",
        mode.description,
        "",
        "Overrides:",
    ]
    lines.extend(f"{key}={value}" for key, value in mode.overrides.items())
    return "\n".join(lines)
