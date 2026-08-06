"""Audit whether incumbent training can be reproduced from recorded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.model_runtime_repro import ensure_safe_report_output, file_digest, json_digest


MODEL_KINDS = ("adv", "lstm", "tcn", "tx")
REQUIRED_LINEAGE_FIELDS = (
    "kind",
    "model_digest",
    "scaler_digest",
    "metadata_digest",
    "training_code_path",
    "training_code_digest_or_git_commit",
    "raw_data_source",
    "exchange_venue",
    "symbols_exact_order",
    "timeframe",
    "raw_data_start",
    "raw_data_finish",
    "raw_data_digests",
    "feature_names_order",
    "feature_code_digest",
    "symbol_id_mapping",
    "sequence_length",
    "label_configuration",
    "label_code_digest",
    "class_distribution",
    "split_method",
    "train_split_boundaries",
    "validation_split_boundaries",
    "test_split_boundaries",
    "purge_embargo_settings",
    "scaler_fit_split",
    "random_seeds",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "early_stopping_rule",
    "loss_weights",
    "package_versions",
    "training_hardware_device",
    "validation_metrics",
    "model_selection_rule",
)


class TrainingLineageError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_text_digest(path: Path | str) -> str:
    value = Path(path).read_text(encoding="utf-8-sig")
    return hashlib.sha256(value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def _pick(source: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = source
        found = True
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value is not None:
            return value
    return None


def _canonical_fields(kind: str, source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": _pick(source, "kind", "model_kind") or kind,
        "model_digest": _pick(source, "model_digest", "model_sha256", "artifacts.model_digest"),
        "scaler_digest": _pick(source, "scaler_digest", "scaler_sha256", "artifacts.scaler_digest"),
        "metadata_digest": _pick(source, "metadata_digest", "metadata_sha256", "artifacts.metadata_digest"),
        "training_code_path": _pick(source, "training_code_path", "code.training_path"),
        "training_code_digest_or_git_commit": _pick(
            source, "training_code_digest_or_git_commit", "training_code_digest",
            "training_git_commit", "git_commit", "code.digest", "code.git_commit"
        ),
        "raw_data_source": _pick(source, "raw_data_source", "data.source"),
        "exchange_venue": _pick(source, "exchange_venue", "exchange", "venue", "data.exchange"),
        "symbols_exact_order": _pick(source, "symbols_exact_order", "symbols", "data.symbols"),
        "timeframe": _pick(source, "timeframe", "data.timeframe"),
        "raw_data_start": _pick(source, "raw_data_start", "data.start"),
        "raw_data_finish": _pick(source, "raw_data_finish", "data.finish", "data.end"),
        "raw_data_digests": _pick(source, "raw_data_digests", "dataset_digests", "data.digests"),
        "feature_names_order": _pick(source, "feature_names_order", "feature_names", "features.names"),
        "feature_code_digest": _pick(source, "feature_code_digest", "features.code_digest"),
        "symbol_id_mapping": _pick(source, "symbol_id_mapping", "symbol_id_map", "features.symbol_id_map"),
        "sequence_length": _pick(source, "sequence_length", "seq_len", "training.sequence_length"),
        "label_configuration": _pick(source, "label_configuration", "label", "labels.configuration"),
        "label_code_digest": _pick(source, "label_code_digest", "labels.code_digest"),
        "class_distribution": _pick(source, "class_distribution", "labels.class_distribution"),
        "split_method": _pick(source, "split_method", "split.method"),
        "train_split_boundaries": _pick(source, "train_split_boundaries", "split.train"),
        "validation_split_boundaries": _pick(source, "validation_split_boundaries", "split.validation"),
        "test_split_boundaries": _pick(source, "test_split_boundaries", "split.test"),
        "purge_embargo_settings": _pick(source, "purge_embargo_settings", "split.purge_embargo"),
        "scaler_fit_split": _pick(source, "scaler_fit_split", "scaler.fit_split"),
        "random_seeds": _pick(source, "random_seeds", "seed", "training.seeds"),
        "optimizer": _pick(source, "optimizer", "training.optimizer"),
        "learning_rate": _pick(source, "learning_rate", "lr", "training.learning_rate"),
        "batch_size": _pick(source, "batch_size", "batch", "training.batch_size"),
        "epochs": _pick(source, "epochs", "training.epochs"),
        "early_stopping_rule": _pick(source, "early_stopping_rule", "training.early_stopping"),
        "loss_weights": _pick(source, "loss_weights", "class_weights", "training.loss_weights"),
        "package_versions": _pick(source, "package_versions", "dependencies", "environment.package_versions"),
        "training_hardware_device": _pick(source, "training_hardware_device", "device", "training.device"),
        "validation_metrics": _pick(
            source, "validation_metrics", "metrics", "val_metrics", "val_auc"
        ),
        "model_selection_rule": _pick(
            source, "model_selection_rule", "training.model_selection_rule", "min_auc_gate"
        ),
    }


def audit_lineage_sources(
    kind: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    manifests: Sequence[Mapping[str, Any]] = (),
    artifact_digests: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge recorded lineage, flag conflicts, and never infer absent history."""

    sources: list[tuple[str, Mapping[str, Any]]] = []
    if metadata is not None:
        sources.append(("basic_metadata", metadata))
    sources.extend((f"training_manifest_{index + 1}", value) for index, value in enumerate(manifests))
    if artifact_digests:
        sources.append(("observed_artifact_identity", artifact_digests))
    values_by_field: dict[str, list[tuple[str, Any]]] = {field: [] for field in REQUIRED_LINEAGE_FIELDS}
    for source_name, source in sources:
        canonical = _canonical_fields(kind, source)
        for field, value in canonical.items():
            if value is not None:
                values_by_field[field].append((source_name, value))
    fields: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    conflicts: list[str] = []
    for field in REQUIRED_LINEAGE_FIELDS:
        observations = values_by_field[field]
        distinct: dict[str, Any] = {}
        for source_name, value in observations:
            key = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            distinct[key] = value
        if len(distinct) > 1:
            conflicts.append(field)
            fields[field] = None
            status = "conflicting"
        else:
            fields[field] = observations[0][1] if observations else None
            status = "recorded" if observations else "missing"
        evidence[field] = {
            "status": status,
            "sources": [source for source, _ in observations],
        }
    missing = [field for field, value in fields.items() if value is None and field not in conflicts]
    manifest_present = bool(manifests)
    if conflicts:
        status = "conflicting_lineage"
    elif not sources:
        status = "missing"
    elif not missing and manifest_present:
        status = "complete_reproducible"
    elif manifest_present:
        status = "partially_reproducible"
    else:
        status = "legacy_lineage_incomplete"
    identity_fields = ("kind", "model_digest", "scaler_digest", "metadata_digest")
    inference_complete = all(fields.get(name) is not None for name in identity_fields) and not any(
        name in conflicts for name in identity_fields
    )
    return {
        "lineage_fields": fields,
        "field_evidence": evidence,
        "missing_fields": missing,
        "conflicting_fields": conflicts,
        "lineage_status": status,
        "inference_reproducibility": (
            "artifact_identity_reproducible" if inference_complete else "artifact_identity_incomplete"
        ),
        "training_reproducibility": status,
        "basic_metadata_is_sufficient_for_complete_lineage": False,
        "current_environment_used_as_historical_training_evidence": False,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _manifest_candidates(root: Path, kind: str) -> list[Path]:
    candidates = (
        root / "model_artifacts" / f"dl_{kind}_training_manifest.json",
        root / "model_artifacts" / f"training_manifest_{kind}.json",
        root / "model_artifacts" / kind / "training_manifest.json",
        root / "reports" / f"dl_{kind}_training_manifest.json",
        root / "reports" / f"training_manifest_{kind}.json",
        root / "reports" / f"model_training_manifest_{kind}.json",
    )
    return [path for path in candidates if path.is_file()]


def audit_repository_lineage(
    *, root: Path | str = BASE_DIR, models: Sequence[str] | None = None
) -> dict[str, Any]:
    repository = Path(root)
    kinds = [kind for kind in MODEL_KINDS if not models or kind in models]
    results: dict[str, Any] = {}
    training_code = repository / "ml_dl" / "dl_train.py"
    label_code = repository / "ml_dl" / "dl_labels.py"
    feature_code_candidates = (
        repository / "features.py",
        repository / "ml_dl" / "dl_dataset.py",
    )
    architecture_candidates = (
        repository / "ml_dl" / "dl_models.py",
        repository / "ml_dl" / "dl_models_adv.py",
    )
    current_code_evidence = {
        "training_code_path": "ml_dl/dl_train.py" if training_code.is_file() else None,
        "training_code_digest": file_digest(training_code) if training_code.is_file() else None,
        "label_code_digest": file_digest(label_code) if label_code.is_file() else None,
        "feature_code_candidates": {
            str(path.relative_to(repository)).replace("\\", "/"): file_digest(path)
            for path in feature_code_candidates if path.is_file()
        },
        "architecture_code_candidates": {
            str(path.relative_to(repository)).replace("\\", "/"): file_digest(path)
            for path in architecture_candidates if path.is_file()
        },
        "artifact_linkage": "unverified",
    }
    for kind in kinds:
        model = repository / "model_artifacts" / f"dl_{kind}_latest.pt"
        scaler = repository / "model_artifacts" / f"scaler_{kind}_latest.joblib"
        metadata_path = repository / "model_artifacts" / f"dl_{kind}_metadata.json"
        metadata = _load_json(metadata_path)
        manifests = [value for path in _manifest_candidates(repository, kind) if (value := _load_json(path))]
        identity = {
            "kind": kind,
            "model_digest": file_digest(model) if model.is_file() else None,
            "scaler_digest": file_digest(scaler) if scaler.is_file() else None,
            "metadata_digest": file_digest(metadata_path) if metadata_path.is_file() else None,
        }
        observed_identity = identity if any(
            value is not None for key, value in identity.items() if key != "kind"
        ) else None
        result = audit_lineage_sources(
            kind, metadata=metadata, manifests=manifests, artifact_digests=observed_identity
        )
        related_checkpoints = sorted(
            path for path in (repository / "model_artifacts").glob(f"dl_{kind}_*.pt")
            if path != model
        ) if (repository / "model_artifacts").is_dir() else []
        result["repository_evidence"] = {
            "basic_metadata_present": metadata is not None,
            "training_manifests": [str(path.relative_to(repository)).replace("\\", "/") for path in _manifest_candidates(repository, kind)],
            "unlinked_related_checkpoints": {
                str(path.relative_to(repository)).replace("\\", "/"): file_digest(path)
                for path in related_checkpoints
            },
            "current_training_code_candidate": current_code_evidence,
            "current_code_not_assumed_to_be_historical_code": True,
            "unlinked_evidence_not_used_to_fill_lineage_fields": True,
        }
        results[kind] = result
    statuses = [result["lineage_status"] for result in results.values()]
    if "conflicting_lineage" in statuses:
        verdict = "training_lineage_conflicting"
    elif statuses and all(status == "complete_reproducible" for status in statuses):
        verdict = "training_lineage_complete"
    elif any(status == "legacy_lineage_incomplete" for status in statuses):
        verdict = "training_lineage_legacy_incomplete"
    else:
        verdict = "training_lineage_partial"
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy": {
            "current_env_is_historical_evidence": False,
            "current_requirements_are_historical_evidence": False,
            "current_feature_code_is_historical_evidence_without_digest_link": False,
            "metadata_date_ranges_inferred": False,
            "scaler_train_only_fit_assumed": False,
            "validation_auc_assumed_leakage_free": False,
        },
        "model_results": results,
        "overall_decision": {"verdict": verdict},
        "warnings": [
            "Current repository code is listed only as a candidate source; no incumbent artifact records a digest or commit linking it to that exact code."
        ],
    }
    report["training_lineage_digest"] = json_digest({
        key: value for key, value in report.items() if key not in {"generated_at", "training_lineage_digest"}
    })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 23 original training-lineage audit")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--model", action="append", choices=MODEL_KINDS)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_repository_lineage(models=args.model)
        target = ensure_safe_report_output(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.strict and report["overall_decision"]["verdict"] == "training_lineage_conflicting":
            return 3
        return 0
    except Exception as exc:
        print(f"training_lineage_audit: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
