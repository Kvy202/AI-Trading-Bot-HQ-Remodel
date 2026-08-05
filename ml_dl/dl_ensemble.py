"""
ml_dl/dl_ensemble.py

Loads and runs the deep-learning ensemble used by tools/live_writer.py.

Public API (must not change without updating live_writer.py):
    load_ensemble(X_dim, device=None) -> (models: dict, device_str: str)
    refresh_live_features(seq_len, add_symbol_id, lookback_pad,
                          symbols, timeframe) -> (meta: dict, xw: ndarray)
    predict_ensemble(xw, models, device, weights=None)
        -> (per_model: dict, aggregate: tuple)

Changelog vs the previous version
---------------------------------
- CRITICAL FIX: refresh_live_features now returns a META DICT as the first
  element, not the raw feature array. The previous version returned the
  numpy array, which silently broke the writer: `extract_last_prices(meta)`
  in live_writer.py looks up `meta["last_px"]`, and on a numpy array this
  always returned an empty dict. The downstream effect was every signal
  written with px=0 and the executor skipping every row with `bad_price`.

  The meta dict now carries:
      meta["last_px"]   -> {symbol: last_close_price}  (used by writer)
      meta["symbols"]   -> list of symbols actually present in the window
      meta["rows"]      -> int, how many rows we ended up with
      meta["pad_used"]  -> int, the pad value that finally produced enough rows
      meta["X_live"]    -> the full feature array, in case a caller wants it

- Added NaN/inf sanitisation at the prediction boundary so a model output
  containing NaN can't poison the blended aggregate.
- Replaced bare `print` with a module-level logger that respects DL_LOG_LEVEL.
  Errors still appear in stderr; service managers and the writer's log files
  pick them up automatically.
- _align_feat_dim now warns once per (model, in_dim, target_dim) combination
  instead of silently chopping or padding every tick.
- _pick_device now recognises Apple Silicon's MPS backend (DL_DEVICE=mps).
- Per-model failures during predict_ensemble are caught and logged; the
  ensemble keeps running with the remaining models instead of crashing the
  whole writer for one bad model.
- Weights normalisation is unchanged in behaviour but the parsing is more
  permissive about whitespace.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .dl_infer import load_model, predict_next
from .dl_dataset import load_prices_and_features  # real loader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# We use a module-level logger rather than print so that messages can be
# filtered by DL_LOG_LEVEL and so that service managers / the writer's own
# log capture them as a stream.
_LOG = logging.getLogger("dl_ensemble")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
    _LOG.addHandler(_h)
    _LOG.setLevel(os.getenv("DL_LOG_LEVEL", "INFO").upper())

# Feature-dim warnings can be very noisy if the scaler shape disagrees with
# the live feature pipeline. Emit each unique (kind, in, out) only once.
_FEAT_WARN_SEEN: set = set()

# Degenerate model detection: count consecutive ticks where p_long is collapsed
_DEGEN_COUNTS: Dict[str, int] = {}
_DEGEN_THRESHOLD = 20   # ticks of p < 0.05 or p > 0.95 before auto-excluding
_DEGEN_RECOVER = 5      # consecutive normal ticks needed to re-enable a model

# Variance-based degenerate detection: catches flat p~=0.43 that extreme-p gate misses
_DEGEN_HISTORY: Dict[str, deque] = {}
_DEGEN_VAR_WINDOW = 30  # rolling window of p_long values per model
_DEGEN_VAR_THR = 0.002  # std below this = flat/collapsed output (0.01 was too aggressive,
                         # caught stable-but-healthy models with slow drift; 0.002 only fires
                         # on truly stuck outputs like p=0.43 +/- 0.0001)

# AUC-weight cache: loaded once from metadata JSON, used by _resolve_weights.
_AUC_CACHE: Dict[str, float] = {}

# symbol_id map cache: {compact_symbol: id} as used during training. Loaded once
# from model metadata so serving assigns the SAME id a model learned, instead of
# a position-in-the-live-list id (which silently differs from training).
_SYMBOL_ID_MAP_CACHE: Optional[Dict[str, int]] = None


def _norm_symbol(s: str) -> str:
    """Compact, uppercase symbol key (matches data.executor_symbol semantics)."""
    try:
        from data import executor_symbol
        return executor_symbol(s).upper()
    except Exception:
        return str(s).replace("/", "").split(":")[0].upper()


def _load_symbol_id_map() -> Dict[str, int]:
    """Return the training-time {compact_symbol: id} map, cached.

    Prefers an explicit ``symbol_id_map`` key in any model's metadata JSON; falls
    back to deriving ids from the ordered ``symbols`` list (id = list index),
    which is exactly how data.load_prices_and_features assigned them at train
    time. Returns {} if no metadata is found (caller then keeps positional ids).
    """
    global _SYMBOL_ID_MAP_CACHE
    if _SYMBOL_ID_MAP_CACHE is not None:
        return _SYMBOL_ID_MAP_CACHE
    d = _default_model_dir()
    out: Dict[str, int] = {}
    for kind in ("adv", "tx", "lstm", "tcn"):
        path = os.path.join(d, f"dl_{kind}_metadata.json")
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except Exception:
            continue
        explicit = meta.get("symbol_id_map")
        if isinstance(explicit, dict) and explicit:
            out = {_norm_symbol(k): int(v) for k, v in explicit.items()}
            break
        syms = meta.get("symbols") or []
        if syms:
            out = {_norm_symbol(s): i for i, s in enumerate(syms)}
            break
    _SYMBOL_ID_MAP_CACHE = out
    if out:
        _LOG.info("symbol_id map loaded (%d symbols): %s", len(out), out)
    else:
        _LOG.warning("symbol_id map: no metadata symbols found; serving will use "
                     "positional ids (only correct if live symbol list == training).")
    return out


def common_scaler_dim(models: Dict[str, dict]) -> int:
    """Return the single feature width every loaded scaler expects.

    Raises if the models were trained on different widths (which would make a
    single feature pipeline impossible).
    """
    dims = sorted({int(getattr(p.get("scaler"), "n_features_in_", -1))
                   for p in models.values()})
    if len(dims) != 1:
        raise RuntimeError(
            f"loaded scalers disagree on feature width: {dims}. All models must "
            f"expect the same number of features; retrain the odd one out."
        )
    return dims[0]


def resolve_add_symbol_id(scaler_n_features: int) -> bool:
    """Decide whether to append the symbol_id channel for THIS run.

    Reconciles the DL_ADD_SYMBOL_ID env var with the scaler's actual width:
      - env unset/empty -> auto-derive from the scaler (base -> False, base+1 -> True).
      - env set         -> honour it, but FAIL LOUD if it contradicts the scaler.

    Supports both the deployed 27-feature artifacts (DL_ADD_SYMBOL_ID=0) and a
    future 28-feature retrain (DL_ADD_SYMBOL_ID=1) with no other code change.
    """
    from features import FEATURE_COLS
    base = len(FEATURE_COLS)
    valid = (base, base + 1)
    if scaler_n_features not in valid:
        raise RuntimeError(
            f"Scaler expects {scaler_n_features} features but FEATURE_COLS has {base} "
            f"(valid widths {valid}). features.py and the model artifacts disagree — "
            f"retrain or align features.py."
        )
    env_raw = os.getenv("DL_ADD_SYMBOL_ID")
    if env_raw is None or str(env_raw).strip() == "":
        add_sid = (scaler_n_features == base + 1)
        _LOG.info("DL_ADD_SYMBOL_ID unset; auto-derived add_symbol_id=%s "
                  "from scaler width %d", add_sid, scaler_n_features)
        return add_sid
    add_sid = str(env_raw).strip().lower() not in ("0", "false", "no", "off")
    expected = base + (1 if add_sid else 0)
    if expected != scaler_n_features:
        raise RuntimeError(
            f"DL_ADD_SYMBOL_ID mismatch: env generates {expected} features "
            f"(DL_ADD_SYMBOL_ID={env_raw!r} -> add_symbol_id={add_sid}) but the scaler "
            f"expects {scaler_n_features}. Set DL_ADD_SYMBOL_ID="
            f"{'1' if scaler_n_features == base + 1 else '0'} to match the deployed artifacts."
        )
    return add_sid


def feature_ood(window: np.ndarray, scaler, z_warn: float = 2.0) -> Tuple[int, float]:
    """Serving-time feature-distribution guard.

    Returns (n_features_with_|mean z|>z_warn, max_|mean z|) for the scaled window.
    Dimension-match is NOT feature-match: a train/serve feature-SET mismatch passes
    every dim check but produces enormous z here (the deployed scalers expected a
    different feature at each column). Use this to refuse to trade on garbage.
    """
    z = scaler.transform(np.asarray(window, dtype=np.float32))
    zmean = np.abs(z.mean(axis=0))
    if zmean.size == 0:
        return 0, 0.0
    return int((zmean > z_warn).sum()), float(np.nanmax(zmean))


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    """Pick the torch device based on DL_DEVICE env var.

    Values: 'cpu', 'cuda', 'mps', or 'auto' (default).
    'auto' prefers cuda > mps > cpu.
    """
    pref = os.getenv("DL_DEVICE", "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        # MPS is the Apple Silicon backend.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _default_model_dir() -> str:
    return os.getenv("DL_MODEL_DIR", "model_artifacts")


def _fallback_scaler_paths(kind: str) -> List[str]:
    d = _default_model_dir()
    return [
        os.path.join(d, f"scaler_{kind}_latest.joblib"),  # per-kind preferred
        os.path.join(d, "scaler_latest.joblib"),           # shared fallback
    ]


def _fallback_model_paths(kind: str) -> List[str]:
    d = _default_model_dir()
    return [
        os.path.join(d, f"dl_{kind}_latest.pt"),
        os.path.join(d, f"dl_{kind}.pt"),
    ]


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _load_auc_from_metadata(kind: str) -> float:
    """Read val_auc from dl_{kind}_metadata.json. Cached; returns 1.0 on missing file."""
    if kind in _AUC_CACHE:
        return _AUC_CACHE[kind]
    path = os.path.join(_default_model_dir(), f"dl_{kind}_metadata.json")
    try:
        with open(path) as fh:
            val = float(json.load(fh).get("val_auc", 1.0))
    except Exception:
        val = 1.0
    _AUC_CACHE[kind] = val
    _LOG.debug("auc-weight %s: %.4f", kind, val)
    return val


# ---------------------------------------------------------------------------
# Load all base models
# ---------------------------------------------------------------------------

def load_ensemble(
    X_dim: int,
    device: Optional[str] = None,
    *,
    require_all: bool = False,
) -> Tuple[Dict[str, dict], str]:
    """Load every base model that has on-disk artifacts. Returns (models, device).

    `models` is a dict keyed by kind ('tcn', 'lstm', 'tx') where each value is
    {"scaler": ..., "model": ...}. Models that fail to load are skipped and
    a warning is logged. If NO model loads, RuntimeError is raised.  The live
    writer uses ``require_all=True`` so a configured/deployed artifact failure
    cannot silently degrade to a smaller ensemble before contract validation.
    """
    dev = device or _pick_device()
    load_failures: Dict[str, str] = {}

    def maybe_load(kind: str, scaler_env: str, model_env: str) -> Optional[Dict[str, Any]]:
        scaler_path_env = os.getenv(scaler_env, "").strip()
        model_path_env = os.getenv(model_env, "").strip()

        scaler_path = _first_existing([scaler_path_env] + _fallback_scaler_paths(kind))
        model_path = _first_existing([model_path_env] + _fallback_model_paths(kind))

        if not scaler_path or not model_path:
            _LOG.warning("skip %s: missing files scaler=%r model=%r",
                         kind, scaler_path, model_path)
            load_failures[kind] = "missing model or scaler artifact"
            return None

        try:
            scaler, model, _dev = load_model(kind, X_dim, scaler_path, model_path, device=dev)
            metadata_path_raw = os.getenv(f"DL_{kind.upper()}_METADATA_PATH", "").strip()
            metadata_path = metadata_path_raw or os.path.join(
                _default_model_dir(), f"dl_{kind}_metadata.json"
            )
            metadata: Optional[Dict[str, Any]] = None
            metadata_status = "missing"
            try:
                with open(metadata_path, encoding="utf-8-sig") as handle:
                    loaded_metadata = json.load(handle)
                if isinstance(loaded_metadata, dict):
                    metadata = loaded_metadata
                    metadata_status = "loaded"
                else:
                    metadata_status = "malformed"
            except FileNotFoundError:
                metadata_status = "missing"
            except Exception:
                metadata_status = "malformed"
            _LOG.info("loaded %s: scaler=%s model=%s dev=%s",
                      kind, os.path.basename(scaler_path),
                      os.path.basename(model_path), _dev)
            return {
                "scaler": scaler,
                "model": model,
                "metadata": metadata,
                "metadata_status": metadata_status,
                "scaler_load_status": "loaded",
                "model_load_status": "loaded",
            }
        except Exception as e:
            _LOG.warning("failed to load %s: %s", kind, e)
            load_failures[kind] = f"{type(e).__name__}: {e}"
            return None

    models: Dict[str, dict] = {}
    models["tcn"] = maybe_load("tcn", "DL_TCN_SCALER_PATH", "DL_TCN_MODEL_PATH")
    models["lstm"] = maybe_load("lstm", "DL_LSTM_SCALER_PATH", "DL_LSTM_MODEL_PATH")
    models["tx"] = maybe_load("tx", "DL_TX_SCALER_PATH", "DL_TX_MODEL_PATH")
    models["adv"] = maybe_load("adv", "DL_ADV_SCALER_PATH", "DL_ADV_MODEL_PATH")

    models = {k: v for k, v in models.items() if v is not None}
    if require_all and load_failures:
        details = "; ".join(f"{kind}={load_failures[kind]}" for kind in sorted(load_failures))
        raise RuntimeError(f"required ensemble model/scaler load failure: {details}")
    if not models:
        raise RuntimeError(
            "No ensemble members loaded. Provide *_SCALER_PATH and *_MODEL_PATH envs, "
            "or ensure defaults exist: model_artifacts/scaler_latest.joblib and "
            "model_artifacts/dl_{tcn,lstm,tx}_latest.pt"
        )

    _LOG.info("ensemble loaded: %d model(s) on %s -> %s",
              len(models), dev, ",".join(sorted(models.keys())))
    return models, dev


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _align_feat_dim(x_window: np.ndarray, target_dim: int, kind: str = "?") -> np.ndarray:
    """Validate that x_window already has exactly target_dim feature columns.

    Previously this silently truncated extra columns or zero-padded missing ones,
    which is exactly how a train/serve feature-set divergence stays hidden for
    weeks while feeding the model scrambled inputs. We now FAIL LOUDLY: a
    dimension mismatch raises, so the per-model try/except in predict_ensemble
    drops that model (logged) instead of quietly corrupting its features. The
    startup scaler-dim check in tools/live_writer.py catches the same class of
    bug at load time.
    """
    x = np.asarray(x_window, dtype=np.float32)
    _T, F = x.shape
    if F == target_dim:
        return x
    raise RuntimeError(
        f"feature dim mismatch for model {kind!r}: window has {F} features but "
        f"the scaler/model expects {target_dim}. Refusing to silently pad/truncate "
        f"— train and serve feature sets have diverged. Retrain, or align features.py."
    )


# ---------------------------------------------------------------------------
# NaN guards
# ---------------------------------------------------------------------------

def _safe_finite(x: float, default: float = 0.0) -> float:
    """Return x as a finite float, or `default` if NaN/inf/parse-fail."""
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


# ---------------------------------------------------------------------------
# Weight parsing
# ---------------------------------------------------------------------------

def _resolve_weights(kinds: List[str],
                     weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Return a normalised, non-negative weight dict covering all kinds.

    Priority:
    1. Explicit ``weights`` argument from the caller.
    2. DL_MODEL_WEIGHTS env var (comma-separated ``kind:weight`` pairs).
    3. val_auc from each model's metadata JSON (auto-loaded, cached).
    4. Equal weights fallback if nothing else is available.
    """
    if weights is None:
        weights_env = os.getenv("DL_MODEL_WEIGHTS", "").strip()
        parsed: Dict[str, float] = {}
        if weights_env:
            for part in weights_env.split(","):
                part = part.strip()
                if ":" not in part:
                    continue
                k, v = part.split(":", 1)
                try:
                    parsed[k.strip()] = float(v.strip())
                except Exception:
                    continue
        if parsed:
            weights = parsed
        else:
            # Auto-load from metadata -- higher validation AUC -> more weight.
            weights = {k: _load_auc_from_metadata(k) for k in kinds}

    # Clamp to non-negative, normalise to sum to 1, default to equal weights
    # if every relevant kind has weight 0.
    pos = {k: max(0.0, float(weights.get(k, 0.0))) for k in kinds}
    total = sum(pos.values())
    if total <= 0:
        return {k: 1.0 / len(kinds) for k in kinds}
    return {k: v / total for k, v in pos.items()}


# ---------------------------------------------------------------------------
# Per-model + blended preds
# ---------------------------------------------------------------------------

def predict_ensemble(
    x_window: np.ndarray,
    models: Dict[str, dict],
    device: str,
    weights: Optional[Dict[str, float]] = None,
    symbol: Optional[str] = None,
) -> Tuple[Dict[str, tuple], tuple]:
    """Run every loaded model and return (per_model, aggregate).

    per_model: {kind: (ret_hat, rv_hat, p_long)}  - one entry per model that
               successfully predicted. Models that throw are dropped with a
               warning instead of aborting the whole tick.
    aggregate: (ret, rv, p_long) - weight-blended across the surviving models.

    NaN/inf in any model output is replaced with 0.0 before blending.
    """
    kinds: List[str] = list(models.keys())
    if not kinds:
        raise ValueError("predict_ensemble: empty models dict")

    weights = _resolve_weights(kinds, weights)

    per_model: Dict[str, tuple] = {}
    failed: List[str] = []
    for k, pack in models.items():
        scaler = pack["scaler"]
        target_dim = int(getattr(scaler, "n_features_in_", x_window.shape[1]))
        try:
            xw_aligned = _align_feat_dim(x_window, target_dim, kind=k)
            ret_hat, rv_hat, p_long = predict_next(xw_aligned, scaler, pack["model"], device)
            per_model[k] = (
                _safe_finite(ret_hat),
                _safe_finite(rv_hat),
                _safe_finite(p_long),
            )
        except Exception as e:
            _LOG.warning("predict failed for %s: %s", k, e)
            failed.append(k)
            continue

        # Per-model calibration: bias first, then temperature.
        # Order matches tools/calibrate_temperature.py:
        #   raw p -> subtract DL_BIAS_<MODEL> -> apply DL_TEMP_<MODEL>
        #
        # Step 1: bias offset (probability-space median offset).
        # Subtracts the model's historical median deviation from 0.5 so that
        # exactly half its predictions land each side of the decision boundary.
        # Configure via DL_BIAS_LSTM / DL_BIAS_TCN / DL_BIAS_TX in .env.
        # Use tools/calibrate_temperature.py to compute the right values.
        _p_raw = per_model[k][2]
        _bias = _safe_finite(float(os.getenv(f"DL_BIAS_{k.upper()}", "0.0")), 0.0)
        if _bias != 0.0:
            _p_raw = max(1e-6, min(1.0 - 1e-6, _p_raw - _bias))

        # Step 2: temperature (logit-space sharpness scaling).
        # T > 1 compresses overconfident probabilities toward 0.5.
        # Applied after bias so the calibration script's pipeline is matched.
        # Configure via DL_TEMP_LSTM / DL_TEMP_TCN / DL_TEMP_TX in .env.
        _temp = _safe_finite(float(os.getenv(f"DL_TEMP_{k.upper()}", "1.0")), 1.0)
        if _temp > 0.0 and _temp != 1.0 and 0.0 < _p_raw < 1.0:
            _logit = math.log(_p_raw / (1.0 - _p_raw))
            _p_raw = 1.0 / (1.0 + math.exp(-_logit / _temp))

        per_model[k] = (per_model[k][0], per_model[k][1], _p_raw)

        # degenerate detection: p < 0.05 or p > 0.95 repeatedly -- exclude.
        # Counters are namespaced by (kind, symbol) so scoring multiple symbols
        # per tick doesn't mix their histories into one collapse counter.
        p_val = per_model[k][2]
        dk = (k, symbol)
        if p_val < 0.05 or p_val > 0.95:
            _DEGEN_COUNTS[dk] = _DEGEN_COUNTS.get(dk, 0) + 1
        else:
            _DEGEN_COUNTS[dk] = max(0, _DEGEN_COUNTS.get(dk, 0) - _DEGEN_RECOVER)

        if _DEGEN_COUNTS.get(dk, 0) >= _DEGEN_THRESHOLD:
            if _DEGEN_COUNTS[dk] == _DEGEN_THRESHOLD:  # log once at threshold
                _LOG.warning(
                    "auto-exclude %s[%s]: p_long=%.6f collapsed for %d consecutive ticks",
                    k, symbol, p_val, _DEGEN_COUNTS[dk],
                )
            del per_model[k]
            failed.append(k)
            continue

        # variance-based: catches flat p~=0.43 that the extreme-p gate misses
        hist = _DEGEN_HISTORY.setdefault(dk, deque(maxlen=_DEGEN_VAR_WINDOW))
        hist.append(p_val)
        if len(hist) >= _DEGEN_VAR_WINDOW and float(np.std(hist)) < _DEGEN_VAR_THR:
            _LOG.warning(
                "auto-exclude %s[%s]: p_long std=%.5f < %.3f over %d ticks (flat output)",
                k, symbol, float(np.std(hist)), _DEGEN_VAR_THR, _DEGEN_VAR_WINDOW,
            )
            del per_model[k]
            failed.append(k)
            continue

    if not per_model:
        # Nothing succeeded - return a safe neutral aggregate so the writer can
        # log "tick_failed" and keep going, instead of crashing the loop.
        # p_long=0.5 centers to 0.0 (FLAT) in the writer; the writer also
        # forces allow=0 because per_model is empty.
        _LOG.error("all %d models failed to predict on this tick", len(kinds))
        return {}, (0.0, 0.0, 0.5)

    if failed:
        # Re-normalise weights over the survivors so the blend isn't biased
        # by a missing model's slot.
        survivors = list(per_model.keys())
        weights = _resolve_weights(survivors, weights)
        kinds = survivors

    rets = np.array([per_model[k][0] for k in kinds], dtype=np.float32)
    rvs = np.array([per_model[k][1] for k in kinds], dtype=np.float32)
    plons = np.array([per_model[k][2] for k in kinds], dtype=np.float32)
    ws = np.array([weights[k] for k in kinds], dtype=np.float32)

    blend = (
        _safe_finite(float(np.dot(rets, ws))),
        _safe_finite(float(np.dot(rvs, ws))),
        _safe_finite(float(np.dot(plons, ws))),
    )

    # Agreement gate: require DL_MIN_AGREE models to share the same directional
    # conviction (all bullish p>0.5, OR all bearish p<0.5).  Mixed conviction
    # (e.g. one bull + two bears) is suppressed to NEUTRAL 0.5 so the centered
    # signal emitted by the writer is 0.0 -> side_hint FLAT, allow=0, no trade.
    # (It used to suppress to 0.0, which the writer centered to -0.5 — a fake
    # maximum-conviction SHORT that PASSED the allow gate. Disagreement was
    # being traded as a short signal.)
    # Only models with a positive blend weight get a vote: a model zeroed via
    # DL_MODEL_WEIGHTS (e.g. tcn:0) still predicts and is logged for
    # diagnostics, but cannot tip the gate. A flat/dead model that always
    # emits ~0.5+eps would otherwise act as a permanent tiebreaker.
    # Set DL_MIN_AGREE=1 to disable.
    _min_agree = int(os.getenv("DL_MIN_AGREE", "2"))
    _voters = [k for k in kinds if weights.get(k, 0.0) > 0.0] or kinds
    if len(_voters) >= _min_agree:
        _n_bull = sum(1 for k in _voters if per_model[k][2] > 0.5)
        _n_bear = sum(1 for k in _voters if per_model[k][2] < 0.5)
        if _n_bull < _min_agree and _n_bear < _min_agree:
            _LOG.info(
                "agree-gate: %d bull %d bear of %d voters (need %d for either "
                "direction); blended p=%.3f -> suppressed to neutral",
                _n_bull, _n_bear, len(_voters), _min_agree, blend[2],
            )
            blend = (blend[0], blend[1], 0.5)

    return per_model, blend


# ---------------------------------------------------------------------------
# Live feature refresh
# ---------------------------------------------------------------------------

def timeframe_duration_seconds(timeframe: str) -> int:
    """Return a positive bar duration for standard fixed CCXT timeframes."""

    value = str(timeframe or "").strip().lower()
    if len(value) < 2 or not value[:-1].isdigit():
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}.get(value[-1])
    if scale is None or int(value[:-1]) <= 0:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return int(value[:-1]) * scale


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def canonical_utc(value: Any) -> str:
    """Canonical, platform-independent UTC identity string."""

    timestamp = _utc_timestamp(value)
    if timestamp.microsecond:
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def source_bar_id(symbol: str, bar_close_utc: Any) -> str:
    return f"{_norm_symbol(symbol)}:{canonical_utc(bar_close_utc)}"


def feature_window_digest(
    symbol: str,
    timeframe: str,
    feature_columns: List[str],
    feature_timestamps: Any,
    values: np.ndarray,
) -> str:
    """Hash an ordered float32 feature window without machine-specific data."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("feature window must be a non-empty 2-D matrix")
    timestamps = list(feature_timestamps)
    if len(timestamps) != matrix.shape[0]:
        raise ValueError("feature timestamp count does not match window rows")
    if len(feature_columns) != matrix.shape[1]:
        raise ValueError("ordered feature-column count does not match window width")
    header = {
        "symbol": _norm_symbol(symbol),
        "timeframe": str(timeframe).strip().lower(),
        "feature_columns": [str(value) for value in feature_columns],
        "first_feature_utc": canonical_utc(timestamps[0]),
        "last_feature_utc": canonical_utc(timestamps[-1]),
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    digest.update(b"\x00")
    little_endian = np.ascontiguousarray(matrix, dtype=np.dtype("<f4"))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def completed_bar_mask(
    bar_open_times: Any,
    timeframe: str,
    *,
    as_of_utc: Any = None,
    completion_grace_seconds: int = 5,
) -> np.ndarray:
    """Return the completed-candle mask defined by the Phase 22 contract."""

    if int(completion_grace_seconds) < 0:
        raise ValueError("completion_grace_seconds must be non-negative")
    as_of = _utc_timestamp(
        datetime.now(timezone.utc) if as_of_utc is None else as_of_utc
    )
    duration = pd.Timedelta(seconds=timeframe_duration_seconds(timeframe))
    grace = pd.Timedelta(seconds=int(completion_grace_seconds))
    index = pd.DatetimeIndex([_utc_timestamp(value) for value in bar_open_times])
    return np.asarray(index + duration + grace <= as_of, dtype=bool)


def _provenance_for_windows(
    windows: Mapping[str, np.ndarray],
    dfs: Any,
    *,
    timeframe: str,
    add_symbol_id: bool,
    as_of_utc: Any,
    completion_grace_seconds: int,
) -> Dict[str, Dict[str, Any]]:
    from features import canonical_feature_columns

    columns = canonical_feature_columns(add_symbol_id)
    duration = pd.Timedelta(seconds=timeframe_duration_seconds(timeframe))
    maps: Dict[str, Dict[str, Any]] = {
        "source_bar_open_utc_by_symbol": {},
        "source_bar_close_utc_by_symbol": {},
        "source_bar_completed_by_symbol": {},
        "source_bar_id_by_symbol": {},
        "feature_window_digest_by_symbol": {},
        "feature_window_row_count_by_symbol": {},
        "feature_window_first_utc_by_symbol": {},
        "feature_window_last_utc_by_symbol": {},
    }
    if not isinstance(dfs, Mapping):
        return maps
    for symbol, window in windows.items():
        frame = dfs.get(symbol)
        if frame is None or len(frame) < len(window):
            continue
        index = pd.DatetimeIndex([_utc_timestamp(value) for value in frame.index[-len(window):]])
        bar_open = index[-1]
        bar_close = bar_open + duration
        complete = bool(
            completed_bar_mask(
                [bar_open], timeframe, as_of_utc=as_of_utc,
                completion_grace_seconds=completion_grace_seconds,
            )[0]
        )
        maps["source_bar_open_utc_by_symbol"][symbol] = canonical_utc(bar_open)
        maps["source_bar_close_utc_by_symbol"][symbol] = canonical_utc(bar_close)
        maps["source_bar_completed_by_symbol"][symbol] = complete
        maps["source_bar_id_by_symbol"][symbol] = source_bar_id(symbol, bar_close)
        maps["feature_window_digest_by_symbol"][symbol] = feature_window_digest(
            symbol, timeframe, columns, index, window
        )
        maps["feature_window_row_count_by_symbol"][symbol] = int(len(window))
        maps["feature_window_first_utc_by_symbol"][symbol] = canonical_utc(index[0])
        maps["feature_window_last_utc_by_symbol"][symbol] = canonical_utc(index[-1])
    return maps

def _extract_last_prices_from_dfs(dfs: Any, symbols: Optional[List[str]]) -> Dict[str, float]:
    """Best-effort extraction of {symbol: last_close} from whatever
    load_prices_and_features returned as its second value.

    Handles three shapes that load_prices_and_features might use, so this
    keeps working even if the dataset module changes its return type:
      1. dict {symbol: DataFrame-like with a 'close' column}
      2. dict {symbol: dict with 'close' key (list/array) or scalar 'last_px'}
      3. single DataFrame-like with 'symbol' and 'close' columns
    Anything we can't parse is silently skipped - missing prices are
    handled correctly downstream by the writer (writes "0", executor skips).
    """
    out: Dict[str, float] = {}
    if dfs is None:
        return out

    def _safe_last_close(obj: Any) -> Optional[float]:
        # Pandas Series / DataFrame
        try:
            if hasattr(obj, "columns"):
                # DataFrame-like
                if "close" in obj.columns:
                    val = obj["close"].iloc[-1]
                    return _safe_finite(val) if val is not None else None
                # Maybe last numeric column is a price
                last_col = obj.columns[-1]
                val = obj[last_col].iloc[-1]
                return _safe_finite(val) if val is not None else None
            if hasattr(obj, "iloc"):
                # Series-like
                val = obj.iloc[-1]
                return _safe_finite(val)
        except Exception:
            pass
        # Plain dict
        if isinstance(obj, dict):
            for key in ("last_px", "last_price", "last_close"):
                if key in obj:
                    return _safe_finite(obj[key])
            if "close" in obj:
                seq = obj["close"]
                try:
                    return _safe_finite(seq[-1])
                except Exception:
                    return None
        # Plain sequence -> assume close prices
        try:
            return _safe_finite(obj[-1])
        except Exception:
            return None

    # Shape 1 & 2: dict keyed by symbol
    if isinstance(dfs, dict):
        for sym, frame in dfs.items():
            v = _safe_last_close(frame)
            if v is not None and v > 0:
                out[str(sym)] = float(v)
        return out

    # Shape 3: a single combined DataFrame
    try:
        if hasattr(dfs, "columns") and "symbol" in dfs.columns and "close" in dfs.columns:
            for sym in (symbols or dfs["symbol"].unique().tolist()):
                rows = dfs[dfs["symbol"] == sym]
                if len(rows) == 0:
                    continue
                v = _safe_finite(rows["close"].iloc[-1])
                if v > 0:
                    out[str(sym)] = float(v)
    except Exception:
        pass

    return out


def refresh_live_features(
    seq_len: int,
    add_symbol_id: bool,
    lookback_pad: int = 200,
    symbols: Optional[list] = None,
    timeframe: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Pull a recent feature window plus a symbol -> last_price map.

    Returns
    -------
    meta : dict
        {
          "last_px":  {symbol: last_close_price},
          "symbols":  [symbol, ...],
          "rows":     int,
          "pad_used": int,
          "X_live":   np.ndarray,    # full window we drew from
        }
    xw : np.ndarray
        The most recent `seq_len` rows of features, ready to feed the model.

    The function auto-doubles the lookback pad until it has at least seq_len
    rows, up to DL_MAX_LOOKBACK_PAD (default 5000).

    IMPORTANT: meta["last_px"] is what live_writer.py reads to populate the
    `px` column in live_signals.csv. If this dict is empty, the writer will
    correctly write "0" and the executor will skip those signals - which
    is the safe behaviour but means no trades. If you see persistent "WARN
    missing price" messages from the writer, the issue is in
    load_prices_and_features (it's not returning price data we can recognise).
    """
    # Ceiling on the doubling loop. The previous version read DL_MAX_LOOKBACK_PAD
    # for this, but live_writer.py ALSO reads DL_MAX_LOOKBACK_PAD as the *starting*
    # lookback_pad with default 6000 - while this ceiling defaulted to 5000.
    # When the user didn't set the env var, the writer passed pad=6000 which was
    # already > the 5000 ceiling, so the doubling loop never ran and we raised
    # "insufficient rows" with last_err=None.
    #
    # We now read DL_LOOKBACK_PAD_CEILING (a distinct name), and we also defensively
    # ensure the ceiling is at least the starting pad so the loop runs at least once.
    cfg_ceiling = int(os.getenv("DL_LOOKBACK_PAD_CEILING", os.getenv("DL_MAX_LOOKBACK_PAD", "8000")))
    pad = max(int(lookback_pad), 64)
    max_pad = max(cfg_ceiling, pad)

    last_err: Optional[str] = None
    X_live: Optional[np.ndarray] = None
    dfs: Any = None
    pad_used = pad

    while pad <= max_pad:
        try:
            # Try to get DataFrames too so we can populate last_px. Some
            # implementations of load_prices_and_features may not accept
            # return_dfs=True - if that's the case we fall back to the
            # original signature on the next iteration.
            try:
                X_live, dfs = load_prices_and_features(
                    symbols=symbols,
                    timeframe=timeframe,
                    lookback=seq_len + pad,
                    add_symbol_id=add_symbol_id,
                    return_dfs=True,
                )
            except TypeError:
                # Older signature without return_dfs
                X_live, dfs = load_prices_and_features(
                    symbols=symbols,
                    timeframe=timeframe,
                    lookback=seq_len + pad,
                    add_symbol_id=add_symbol_id,
                )

            if X_live is not None and X_live.shape[0] >= seq_len:
                pad_used = pad
                break

            last_err = (f"got {0 if X_live is None else X_live.shape[0]} rows "
                        f"with pad={pad} (need {seq_len})")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        pad = int(pad * 2)  # exponential backoff

    if X_live is None or X_live.shape[0] < seq_len:
        raise RuntimeError(
            f"refresh_live_features: insufficient rows for seq_len={seq_len} "
            f"even after pad up to {max_pad} (last error: {last_err})"
        )

    xw = X_live[-seq_len:, :]
    last_px = _extract_last_prices_from_dfs(dfs, symbols)

    # The symbol list we report is whatever we managed to extract prices for,
    # falling back to the requested symbols if extraction failed entirely.
    # The writer prefers meta["symbols"] but degrades gracefully if missing.
    syms_out = list(last_px.keys()) if last_px else (list(symbols) if symbols else [])

    meta: Dict[str, Any] = {
        "last_px": last_px,
        "symbols": syms_out,
        "rows": int(X_live.shape[0]),
        "pad_used": int(pad_used),
        "X_live": X_live,
    }

    if not last_px:
        # Don't crash - the writer will handle empty prices safely - but make
        # the cause visible. Without this, "missing price" warnings in the
        # writer's log are mysterious.
        _LOG.warning("refresh_live_features: extracted 0 prices from dataset return; "
                     "symbols=%r, dfs type=%s",
                     symbols, type(dfs).__name__)

    return meta, xw


def refresh_live_features_per_symbol(
    seq_len: int,
    add_symbol_id: bool = True,
    lookback_pad: int = 200,
    symbols: Optional[list] = None,
    timeframe: Optional[str] = None,
    symbol_id_map: Optional[Dict[str, int]] = None,
    completed_only: bool = False,
    as_of_utc: Any = None,
    completion_grace_seconds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Build ONE feature window PER SYMBOL so each is scored independently.

    Returns
    -------
    meta : dict   {"last_px": {sym: px}, "symbols": [...], "rows": int, "pad_used": int}
    windows : dict  {compact_symbol: np.ndarray[seq_len, F]}

    This replaces the old single-window path used by the live writer, where only
    the LAST symbol's window was scored and its signal copied to every symbol.
    Here every symbol gets its own most-recent `seq_len` rows, with the correct
    persisted symbol_id baked in, so ETH and SOL produce genuinely separate
    predictions.
    """
    if add_symbol_id and symbol_id_map is None:
        symbol_id_map = _load_symbol_id_map() or None

    cfg_ceiling = int(os.getenv("DL_LOOKBACK_PAD_CEILING", os.getenv("DL_MAX_LOOKBACK_PAD", "8000")))
    pad = max(int(lookback_pad), 64)
    max_pad = max(cfg_ceiling, pad)

    last_err: Optional[str] = None
    X_live: Optional[np.ndarray] = None
    dfs: Any = None
    sym_lengths: List[int] = []
    pad_used = pad

    while pad <= max_pad:
        try:
            X_live, dfs, sym_lengths = load_prices_and_features(
                symbols=symbols,
                timeframe=timeframe,
                lookback=seq_len + pad,
                add_symbol_id=add_symbol_id,
                symbol_id_map=symbol_id_map,
                return_dfs=True,
                return_symbol_lengths=True,
            )
            if X_live is not None and sym_lengths and min(sym_lengths) >= seq_len:
                pad_used = pad
                break
            last_err = (f"min per-symbol rows="
                        f"{min(sym_lengths) if sym_lengths else 0} with pad={pad} "
                        f"(need {seq_len})")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        pad = int(pad * 2)

    if X_live is None or not sym_lengths or min(sym_lengths) < seq_len:
        raise RuntimeError(
            f"refresh_live_features_per_symbol: insufficient rows for seq_len={seq_len} "
            f"even after pad up to {max_pad} (last error: {last_err})"
        )

    effective_timeframe = str(timeframe or "1m").lower()
    if completed_only:
        if not isinstance(dfs, Mapping):
            raise RuntimeError("completed-only serving requires timestamped per-symbol frames")
        from features import canonical_feature_columns
        ordered_columns = canonical_feature_columns(add_symbol_id)
        filtered_dfs: Dict[str, Any] = {}
        completed_blocks: List[np.ndarray] = []
        completed_lengths: List[int] = []
        for symbol, frame in dfs.items():
            mask = completed_bar_mask(
                frame.index,
                effective_timeframe,
                as_of_utc=as_of_utc,
                completion_grace_seconds=completion_grace_seconds,
            )
            completed = frame.loc[mask].copy()
            if len(completed) < seq_len:
                raise RuntimeError(
                    f"refresh_live_features_per_symbol: {symbol} has {len(completed)} "
                    f"completed feature rows (need {seq_len})"
                )
            missing = [column for column in ordered_columns if column not in completed.columns]
            if missing:
                raise RuntimeError(f"completed feature frame for {symbol} missing columns {missing}")
            filtered_dfs[symbol] = completed
            completed_lengths.append(len(completed))
            completed_blocks.append(np.ascontiguousarray(
                completed[ordered_columns].to_numpy(dtype=np.float32),
                dtype=np.float32,
            ))
        dfs = filtered_dfs
        sym_lengths = completed_lengths
        X_live = np.concatenate(completed_blocks, axis=0)

    last_px = _extract_last_prices_from_dfs(dfs, symbols)

    # dfs insertion order == the order symbols were stacked into X_live, and
    # sym_lengths is in that same order, so each contiguous block of X_live is one
    # symbol. Slice the block and take its most-recent seq_len rows.
    sym_keys = list(dfs.keys()) if isinstance(dfs, dict) else []
    windows: Dict[str, np.ndarray] = {}
    offset = 0
    for i, slen in enumerate(sym_lengths):
        start, end = offset, offset + slen
        offset = end
        if i >= len(sym_keys):
            break
        block = X_live[start:end, :]
        if block.shape[0] < seq_len:
            continue
        windows[sym_keys[i]] = block[-seq_len:, :]

    if not windows:
        raise RuntimeError(
            "refresh_live_features_per_symbol: produced no per-symbol windows "
            f"(sym_lengths={sym_lengths}, keys={sym_keys})"
        )

    meta: Dict[str, Any] = {
        "last_px": last_px,
        "symbols": list(windows.keys()),
        "rows": int(X_live.shape[0]),
        "pad_used": int(pad_used),
    }
    meta.update(
        _provenance_for_windows(
            windows,
            dfs,
            timeframe=effective_timeframe,
            add_symbol_id=add_symbol_id,
            as_of_utc=as_of_utc,
            completion_grace_seconds=completion_grace_seconds,
        )
    )
    if not last_px:
        _LOG.warning("refresh_live_features_per_symbol: extracted 0 prices; symbols=%r", symbols)

    return meta, windows
