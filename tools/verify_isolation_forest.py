"""Verify the optional Isolation Forest shadow artifact.

Checks:
* artifact exists
* artifact loads through the same IsolationFilter path live_writer uses
* prediction works on a synthetic live-shaped sample window
* optional shadow CSV row has the expected schema
* optional missing-artifact check degrades without raising
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

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

from ml_optional.isolation_filter import (
    DEFAULT_ARTIFACT,
    ISOLATION_SHADOW_COLS,
    IsolationFilter,
)


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _append_row(path: Path, row: Dict[str, object]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ISOLATION_SHADOW_COLS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in ISOLATION_SHADOW_COLS})


def _sample_window(model) -> np.ndarray:
    n_features = int(getattr(model, "n_features_in_", 0) or 0)
    if n_features <= 0:
        raise RuntimeError("loaded model does not expose n_features_in_")
    live_width = n_features // 4 if n_features % 4 == 0 else n_features
    base = np.linspace(-0.5, 0.5, live_width, dtype=np.float32)
    return np.vstack([base * 0.25, base * 0.5, base * 0.75, base]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser("Verify optional Isolation Forest artifact")
    parser.add_argument("--artifact", default=os.getenv("ISOLATION_FOREST_ARTIFACT", DEFAULT_ARTIFACT))
    parser.add_argument("--shadow-log", default=str(BASE_DIR / "logs" / "isolation_forest_shadow.csv"))
    parser.add_argument("--write-shadow-row", action="store_true")
    parser.add_argument("--missing-artifact-check", action="store_true")
    args = parser.parse_args()

    artifact = _resolve(args.artifact)
    if not artifact.exists():
        raise SystemExit(f"ERROR artifact missing: {artifact}")

    old_artifact_env = os.environ.get("ISOLATION_FOREST_ARTIFACT")
    os.environ["ISOLATION_FOREST_ARTIFACT"] = str(artifact)
    logs = []
    try:
        flt = IsolationFilter.from_env(enabled=True, base_dir=BASE_DIR, log_fn=logs.append)
    finally:
        if old_artifact_env is None:
            os.environ.pop("ISOLATION_FOREST_ARTIFACT", None)
        else:
            os.environ["ISOLATION_FOREST_ARTIFACT"] = old_artifact_env

    if not flt.ready:
        raise SystemExit(f"ERROR artifact did not load: status={flt.isolation_status} reason={flt.reason}")

    window = _sample_window(flt.model)
    result = flt.evaluate("VERIFY", window)
    row = result.to_log_row("VERIFY_TS", "VERIFY")
    missing = [c for c in ISOLATION_SHADOW_COLS if c not in row]
    if missing:
        raise SystemExit(f"ERROR shadow row missing columns: {missing}")

    if args.write_shadow_row:
        _append_row(_resolve(args.shadow_log), row)

    print(
        "OK artifact_loaded "
        f"artifact_path={artifact} model_version={result.model_version} "
        f"anomaly_status={result.anomaly_status} anomaly_score={result.anomaly_score} "
        f"would_block={result.would_block} reason={result.reason}"
    )

    if args.missing_artifact_check:
        old_artifact_env = os.environ.get("ISOLATION_FOREST_ARTIFACT")
        os.environ["ISOLATION_FOREST_ARTIFACT"] = str(BASE_DIR / "model_artifacts" / "__missing_isolation_verify__.joblib")
        miss_logs = []
        try:
            missing_filter = IsolationFilter.from_env(enabled=True, base_dir=BASE_DIR, log_fn=miss_logs.append)
        finally:
            if old_artifact_env is None:
                os.environ.pop("ISOLATION_FOREST_ARTIFACT", None)
            else:
                os.environ["ISOLATION_FOREST_ARTIFACT"] = old_artifact_env
        if missing_filter.isolation_status != "disabled_missing_artifact":
            raise SystemExit(f"ERROR missing-artifact check failed: {missing_filter.isolation_status}")
        print("OK missing_artifact_check isolation_status=disabled_missing_artifact")


if __name__ == "__main__":
    main()

