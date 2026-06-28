"""Default-off experimental feature flags.

These flags only report configuration state for now. Live writer/executor code
must keep behavior unchanged unless later phases add explicit opt-in hooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

EXPERIMENTAL_FLAG_NAMES = (
    "USE_ISOLATION_FOREST",
    "USE_XGBOOST_SIGNAL",
    "USE_SURVIVAL_EXIT",
    "USE_ADVANCED_RISK",
)

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE


@dataclass(frozen=True)
class ExperimentalFlags:
    use_isolation_forest: bool = False
    use_xgboost_signal: bool = False
    use_survival_exit: bool = False
    use_advanced_risk: bool = False

    @classmethod
    def from_env(cls) -> "ExperimentalFlags":
        return cls(
            use_isolation_forest=_env_bool("USE_ISOLATION_FOREST", False),
            use_xgboost_signal=_env_bool("USE_XGBOOST_SIGNAL", False),
            use_survival_exit=_env_bool("USE_SURVIVAL_EXIT", False),
            use_advanced_risk=_env_bool("USE_ADVANCED_RISK", False),
        )

    def as_env_dict(self) -> Dict[str, bool]:
        return {
            "USE_ISOLATION_FOREST": self.use_isolation_forest,
            "USE_XGBOOST_SIGNAL": self.use_xgboost_signal,
            "USE_SURVIVAL_EXIT": self.use_survival_exit,
            "USE_ADVANCED_RISK": self.use_advanced_risk,
        }

    def summary(self) -> str:
        vals = self.as_env_dict()
        return " ".join(f"{name}={str(vals[name]).lower()}" for name in EXPERIMENTAL_FLAG_NAMES)
