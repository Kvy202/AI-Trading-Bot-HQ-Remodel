"""Pure training/serving contract validation for deployed DL artifacts.

The guard consumes already-loaded model/scaler descriptions.  It performs no
I/O, reads no environment variables, and has no exchange or trading imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence


VALID_STATUSES = {"pass", "fail", "unverified"}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _norm_symbol(value: Any) -> str:
    return str(value or "").strip().replace("/", "").split(":", 1)[0].upper()


def _integer(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _entry_from_loaded(kind: str, pack: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an in-memory ensemble pack into a guard entry."""

    metadata = pack.get("metadata")
    scaler = pack.get("scaler")
    model = pack.get("model")
    return {
        "kind": kind,
        "metadata_status": pack.get(
            "metadata_status", "loaded" if isinstance(metadata, Mapping) else "missing"
        ),
        "metadata_kind": metadata.get("kind") if isinstance(metadata, Mapping) else None,
        "metadata_timeframe": metadata.get("timeframe") if isinstance(metadata, Mapping) else None,
        "metadata_seq_len": metadata.get("seq_len") if isinstance(metadata, Mapping) else None,
        "metadata_n_features": metadata.get("n_features") if isinstance(metadata, Mapping) else None,
        "metadata_symbols": metadata.get("symbols") if isinstance(metadata, Mapping) else None,
        "metadata_val_auc": metadata.get("val_auc") if isinstance(metadata, Mapping) else None,
        "metadata_min_auc_gate": metadata.get("min_auc_gate") if isinstance(metadata, Mapping) else None,
        "scaler_n_features_in": getattr(scaler, "n_features_in_", None),
        "scaler_load_status": pack.get(
            "scaler_load_status", "loaded" if scaler is not None else "missing"
        ),
        "model_load_status": pack.get(
            "model_load_status", "loaded" if model is not None else "missing"
        ),
        "sklearn_version_status": pack.get("sklearn_version_status"),
    }


def _normalize_entries(models: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(models, Mapping):
        entries: list[dict[str, Any]] = []
        for raw_kind, pack in sorted(models.items(), key=lambda item: str(item[0])):
            kind = str(raw_kind)
            if isinstance(pack, Mapping) and (
                "metadata" in pack or "scaler" in pack or "model" in pack
            ):
                entries.append(_entry_from_loaded(kind, pack))
            elif isinstance(pack, Mapping):
                value = dict(pack)
                value.setdefault("kind", kind)
                entries.append(value)
            else:
                entries.append({"kind": kind, "metadata_status": "malformed"})
        return entries
    return [dict(entry) for entry in models if isinstance(entry, Mapping)]


def evaluate_model_serving_contract(
    models: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    serving_timeframe: Any,
    serving_sequence_length: Any,
    generated_feature_width: Any,
    add_symbol_id: Any,
    serving_symbols: Sequence[Any],
    base_feature_width: Any | None = None,
    low_auc_warning_threshold: float = 0.55,
) -> dict[str, Any]:
    """Compare the effective serving contract with every loaded artifact.

    ``models`` may be either the mapping returned by ``load_ensemble`` or the
    ``model_entries`` array from a serving snapshot.
    """

    entries = _normalize_entries(models)
    serving_tf = str(serving_timeframe or "").strip().lower()
    serving_seq = _integer(serving_sequence_length)
    generated_width = _integer(generated_feature_width)
    base_width = _integer(base_feature_width)
    sid = add_symbol_id if isinstance(add_symbol_id, bool) else None
    serving_syms = sorted({_norm_symbol(value) for value in serving_symbols if _norm_symbol(value)})
    critical: list[str] = []
    warnings: list[str] = []
    per_model: list[dict[str, Any]] = []

    if not entries:
        result = {
            "status": "unverified",
            "critical_mismatches": [],
            "warnings": ["no loaded model entries were supplied"],
            "training_contract": {},
            "serving_contract": {
                "timeframe": serving_tf or None,
                "sequence_length": serving_seq,
                "generated_feature_width": generated_width,
                "base_feature_width": base_width,
                "add_symbol_id": sid,
                "symbols": serving_syms,
            },
            "per_model_results": [],
        }
        result["guard_digest"] = _digest(result)
        return result

    training_tfs: set[str] = set()
    training_seqs: set[int] = set()
    metadata_widths: set[int] = set()
    scaler_widths: set[int] = set()
    training_symbol_sets: list[set[str]] = []
    training_symbol_sequences: list[list[str]] = []
    kinds_seen: set[str] = set()

    if sid is None:
        critical.append("symbol-id serving setting missing or malformed")
    if base_width is None or base_width <= 0:
        critical.append("base serving feature width missing or malformed")
    if not serving_syms:
        critical.append("serving symbols missing or malformed")

    for raw in entries:
        kind = str(raw.get("kind") or raw.get("metadata_kind") or "unknown").strip().lower()
        item_critical: list[str] = []
        item_warnings: list[str] = []
        metadata_status = str(raw.get("metadata_status") or "missing").lower()
        metadata_kind = str(raw.get("metadata_kind") or "").strip().lower()
        tf = str(raw.get("metadata_timeframe") or "").strip().lower()
        seq = _integer(raw.get("metadata_seq_len"))
        meta_width = _integer(raw.get("metadata_n_features"))
        scaler_width = _integer(raw.get("scaler_n_features_in"))
        symbols_raw = raw.get("metadata_symbols")
        symbols = (
            list(dict.fromkeys(
                _norm_symbol(value) for value in symbols_raw if _norm_symbol(value)
            ))
            if isinstance(symbols_raw, (list, tuple, set))
            else []
        )
        model_status = str(raw.get("model_load_status") or "missing").lower()
        scaler_status = str(
            raw.get("scaler_load_status")
            or ("loaded" if scaler_width is not None else "missing")
        ).lower()

        if kind in kinds_seen:
            item_critical.append(f"{kind}: duplicate model kind")
        kinds_seen.add(kind)

        if metadata_status != "loaded":
            item_critical.append(f"{kind}: metadata {metadata_status}")
        if not metadata_kind or metadata_kind != kind:
            item_critical.append(
                f"{kind}: metadata model kind {metadata_kind or 'missing'} does not match {kind}"
            )
        if not tf:
            item_critical.append(f"{kind}: metadata timeframe missing")
        else:
            training_tfs.add(tf)
        if seq is None or seq <= 0:
            item_critical.append(f"{kind}: metadata sequence length missing or malformed")
        else:
            training_seqs.add(seq)
        if meta_width is None or meta_width <= 0:
            item_critical.append(f"{kind}: metadata feature width missing or malformed")
        else:
            metadata_widths.add(meta_width)
        if scaler_width is None or scaler_width <= 0:
            item_critical.append(f"{kind}: scaler feature width missing or malformed")
        else:
            scaler_widths.add(scaler_width)
        if not symbols:
            item_critical.append(f"{kind}: training symbols missing or malformed")
        else:
            training_symbol_sets.append(set(symbols))
            training_symbol_sequences.append(symbols)
        if model_status != "loaded":
            item_critical.append(f"{kind}: model load status {model_status}")
        if scaler_status != "loaded":
            item_critical.append(f"{kind}: scaler load status {scaler_status}")
        if meta_width is not None and scaler_width is not None and meta_width != scaler_width:
            item_critical.append(
                f"{kind}: metadata feature width {meta_width} != scaler feature width {scaler_width}"
            )
        if generated_width is not None and scaler_width is not None and generated_width != scaler_width:
            item_critical.append(
                f"{kind}: generated serving feature width {generated_width} != scaler feature width {scaler_width}"
            )
        if tf and serving_tf and tf != serving_tf:
            item_critical.append(
                f"{kind}: serving timeframe {serving_tf} != training timeframe {tf}"
            )
        if seq is not None and serving_seq is not None and seq != serving_seq:
            item_critical.append(
                f"{kind}: serving sequence length {serving_seq} != training sequence length {seq}"
            )
        unknown = sorted(set(serving_syms) - set(symbols)) if symbols else list(serving_syms)
        if unknown:
            item_critical.append(f"{kind}: unknown serving symbols {','.join(unknown)}")
        auc = _number(raw.get("metadata_val_auc"))
        auc_gate = _number(raw.get("metadata_min_auc_gate"))
        threshold = max(float(low_auc_warning_threshold), auc_gate or float("-inf"))
        if auc is not None and auc < threshold:
            item_warnings.append(
                f"{kind}: validation AUC {auc:.6f} below warning threshold {threshold:.6f}"
            )
        sklearn_status = str(raw.get("sklearn_version_status") or "")
        if sklearn_status == "loadable_version_mismatch":
            item_warnings.append(f"{kind}: loadable scikit-learn artifact/runtime version mismatch")

        critical.extend(item_critical)
        warnings.extend(item_warnings)
        per_model.append(
            {
                "kind": kind,
                "status": "fail" if item_critical else "pass",
                "critical_mismatches": sorted(set(item_critical)),
                "warnings": sorted(set(item_warnings)),
                "metadata_timeframe": tf or None,
                "metadata_sequence_length": seq,
                "metadata_feature_width": meta_width,
                "scaler_feature_width": scaler_width,
                "training_symbols": symbols,
                "model_load_status": model_status,
                "scaler_load_status": scaler_status,
            }
        )

    if len(training_tfs) != 1:
        critical.append(f"loaded models disagree on training timeframe: {sorted(training_tfs)}")
    if len(training_seqs) != 1:
        critical.append(f"loaded models disagree on sequence length: {sorted(training_seqs)}")
    if len(metadata_widths) != 1:
        critical.append(f"loaded models disagree on metadata feature width: {sorted(metadata_widths)}")
    if len(scaler_widths) != 1:
        critical.append(f"loaded scalers disagree on feature width: {sorted(scaler_widths)}")
    if training_symbol_sequences and any(
        sequence != training_symbol_sequences[0] for sequence in training_symbol_sequences[1:]
    ):
        critical.append("loaded models disagree on ordered training symbols")

    training_tf = next(iter(training_tfs)) if len(training_tfs) == 1 else None
    training_seq = next(iter(training_seqs)) if len(training_seqs) == 1 else None
    training_width = next(iter(metadata_widths)) if len(metadata_widths) == 1 else None
    scaler_width = next(iter(scaler_widths)) if len(scaler_widths) == 1 else None
    common_training_symbols = (
        sorted(set.intersection(*training_symbol_sets)) if training_symbol_sets else []
    )
    ordered_training_symbols = (
        list(training_symbol_sequences[0])
        if training_symbol_sequences
        and all(sequence == training_symbol_sequences[0] for sequence in training_symbol_sequences)
        else []
    )
    if training_tf is not None and serving_tf != training_tf:
        critical.append(f"serving timeframe {serving_tf or 'missing'} != training timeframe {training_tf}")
    if training_seq is not None and serving_seq != training_seq:
        critical.append(
            f"serving sequence length {serving_seq if serving_seq is not None else 'missing'} != training sequence length {training_seq}"
        )
    if generated_width is None or generated_width <= 0:
        critical.append("generated serving feature width missing or malformed")
    if base_width is not None and sid is not None and generated_width != base_width + int(sid):
        critical.append(
            f"symbol-id setting generates {base_width + int(sid)} features != supplied serving width {generated_width}"
        )
    inferred_training_symbol_id = (
        None
        if base_width is None or scaler_width is None
        else scaler_width == base_width + 1
    )
    if sid is not None and inferred_training_symbol_id is not None and sid != inferred_training_symbol_id:
        critical.append(
            f"serving symbol-id setting {str(sid).lower()} != training symbol-id setting "
            f"{str(inferred_training_symbol_id).lower()}"
        )
    if training_width is not None and generated_width != training_width:
        critical.append(
            f"generated serving feature width {generated_width} != training feature width {training_width}"
        )
    if scaler_width is not None and generated_width != scaler_width:
        critical.append(
            f"generated serving feature width {generated_width} != common scaler feature width {scaler_width}"
        )

    critical = sorted(set(critical))
    warnings = sorted(set(warnings))
    result = {
        "status": "fail" if critical else "pass",
        "critical_mismatches": critical,
        "warnings": warnings,
        "training_contract": {
            "timeframe": training_tf,
            "sequence_length": training_seq,
            "metadata_feature_width": training_width,
            "scaler_feature_width": scaler_width,
            "symbols": common_training_symbols,
            "ordered_symbols": ordered_training_symbols,
            "add_symbol_id": inferred_training_symbol_id,
        },
        "serving_contract": {
            "timeframe": serving_tf or None,
            "sequence_length": serving_seq,
            "generated_feature_width": generated_width,
            "base_feature_width": base_width,
            "add_symbol_id": sid,
            "symbols": serving_syms,
        },
        "per_model_results": per_model,
    }
    result["guard_digest"] = _digest(result)
    return result


def guard_model_serving_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Stable public alias for :func:`evaluate_model_serving_contract`."""

    return evaluate_model_serving_contract(*args, **kwargs)


def guard_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a schema-1 or schema-2 model-serving snapshot."""

    add_sid = snapshot.get("dl_add_symbol_id")
    base_width = _integer(snapshot.get("feature_count"))
    generated = None if base_width is None or not isinstance(add_sid, bool) else base_width + int(add_sid)
    return evaluate_model_serving_contract(
        snapshot.get("model_entries", []),
        serving_timeframe=snapshot.get("dl_timeframe"),
        serving_sequence_length=snapshot.get("dl_seq_len"),
        generated_feature_width=generated,
        add_symbol_id=add_sid,
        serving_symbols=snapshot.get("dl_symbols", []),
        base_feature_width=base_width,
    )
