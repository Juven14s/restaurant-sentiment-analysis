"""Strict, dependency-free evaluation for aspect sentiment predictions.

All scores are fractions between zero and one. ``json_valid_rate`` measures
conformance to the prediction schema (all aspects and known labels), rather
than whether an arbitrary raw response can be parsed as JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .data import ASPECTS, LABELS, load_jsonl
from .reporting import experiment_signature


def _index_rows(rows: list[dict], *, kind: str) -> dict[str, dict]:
    indexed = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{kind} row {position} must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"{kind} row {position} must have a nonempty string id")
        if row_id in indexed:
            raise ValueError(f"Duplicate {kind} id: {row_id}")
        indexed[row_id] = row
    return indexed


def _valid_labels(labels: Any) -> bool:
    return (
        isinstance(labels, dict)
        and set(labels) == set(ASPECTS)
        and all(isinstance(labels[aspect], str) and labels[aspect] in LABELS for aspect in ASPECTS)
    )


def evaluate_predictions(references: list[dict], predictions: list[dict]) -> dict:
    """Evaluate by id; retain invalid outputs in every metric denominator.

    Reference rows contain ``id`` and ``labels``. Prediction rows contain
    ``id`` and ``prediction``; a failed generation should use ``None``.
    Duplicate ids, missing/extra predictions, and malformed references raise
    ``ValueError``. A malformed prediction map counts as wrong for all aspects.
    """
    reference_index = _index_rows(references, kind="reference")
    prediction_index = _index_rows(predictions, kind="prediction")
    if not reference_index:
        raise ValueError("Evaluation requires at least one reference")
    missing = set(reference_index) - set(prediction_index)
    extra = set(prediction_index) - set(reference_index)
    if missing or extra:
        raise ValueError(
            f"Prediction ids do not match references: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for row_id, row in reference_index.items():
        if not _valid_labels(row.get("labels")):
            raise ValueError(f"Invalid reference labels for id: {row_id}")

    n_samples = len(reference_index)
    label_indexes = {label: index for index, label in enumerate(LABELS)}
    matrices = {aspect: [[0 for _ in LABELS] for _ in LABELS] for aspect in ASPECTS}
    supports = {aspect: {label: 0 for label in LABELS} for aspect in ASPECTS}
    valid_count = 0
    joint_correct = 0
    for row_id, reference in reference_index.items():
        expected = reference["labels"]
        predicted = prediction_index[row_id].get("prediction")
        valid = _valid_labels(predicted)
        valid_count += int(valid)
        joint_correct += int(valid and predicted == expected)
        for aspect in ASPECTS:
            supports[aspect][expected[aspect]] += 1
            if valid:
                matrices[aspect][label_indexes[expected[aspect]]][label_indexes[predicted[aspect]]] += 1

    per_aspect = {}
    for aspect in ASPECTS:
        matrix = matrices[aspect]
        per_class = {}
        for index, label in enumerate(LABELS):
            true_positive = matrix[index][index]
            predicted_count = sum(row[index] for row in matrix)
            support = supports[aspect][label]
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        per_aspect[aspect] = {
            "accuracy": sum(matrix[index][index] for index in range(len(LABELS))) / n_samples,
            "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
            "per_class": per_class,
            "confusion_matrix": matrix,
            "invalid_count": n_samples - valid_count,
        }
    return {
        "n_samples": n_samples,
        "labels": list(LABELS),
        "confusion_matrix_axes": {"rows": "reference", "columns": "prediction"},
        "per_aspect": per_aspect,
        "macro_accuracy": sum(row["accuracy"] for row in per_aspect.values()) / len(ASPECTS),
        "macro_f1": sum(row["macro_f1"] for row in per_aspect.values()) / len(ASPECTS),
        "joint_exact_match": joint_correct / n_samples,
        "json_valid_rate": valid_count / n_samples,
    }


def evaluate_files(reference_path: Path, predictions_path: Path, output_path: Path) -> dict:
    """Evaluate JSONL artifacts and write a machine-readable JSON result."""
    reference_path = Path(reference_path)
    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    data_fingerprint = None
    manifest_path = reference_path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_fingerprint = manifest.get("fingerprint")
        if not isinstance(data_fingerprint, str) or not data_fingerprint:
            raise ValueError("Data manifest must contain a fingerprint")
        expected_hash = manifest.get("files", {}).get(reference_path.stem, {}).get("sha256")
        if reference_sha256 != expected_hash:
            raise ValueError(f"Data fingerprint mismatch for {reference_path.name}")
    references = load_jsonl(reference_path)
    # Prediction JSONL has a different schema from labelled data. Keep the
    # loader local so failed generations (prediction=null) remain evaluable.
    predictions = []
    with Path(predictions_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank prediction line {line_number}")
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSON at line {line_number}: {exc.msg}") from exc
    results = evaluate_predictions(references, predictions)
    results.update({"reference_sha256": reference_sha256, "data_fingerprint": data_fingerprint})
    prediction_manifest_path = Path(predictions_path).with_suffix(".manifest.json")
    if prediction_manifest_path.is_file():
        provenance = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        if provenance.get("input_sha256") != reference_sha256:
            raise ValueError("Prediction manifest input_sha256 does not match the reference file")
        backend = provenance.get("backend")
        if backend not in {"camembert", "llm", "lora"}:
            raise ValueError("Prediction manifest has an unknown backend")
        source_run = provenance.get("source_run") or {}
        if not isinstance(source_run, dict) or not isinstance(source_run.get("config", {}), dict):
            raise ValueError("Prediction manifest source_run.config must be an object")
        seed = source_run.get("config", {}).get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("Prediction manifest training seed must be an integer")
        results.update({"method": backend, "seed": seed})
        training_config = {key: value for key, value in source_run.get("config", {}).items() if key != "seed"}
        results["experiment_signature"] = experiment_signature({
            "backend": backend,
            "training_config": training_config,
            "base_revision": source_run.get("base_model_revision"),
            "resolved_revision": provenance.get("resolved_revision"),
            "prompt_version": provenance.get("prompt_version"),
            "system_prompt_sha256": provenance.get("system_prompt_sha256"),
            "max_length": provenance.get("max_length"),
            "max_new_tokens": provenance.get("max_new_tokens"),
            "batch_size": provenance.get("batch_size"),
            "quantize_4bit": provenance.get("quantize_4bit"),
            "code_commit": (provenance.get("runtime") or {}).get("git_commit"),
        })
        inference_seconds = provenance.get("inference_seconds")
        if inference_seconds is not None:
            if not isinstance(inference_seconds, (int, float)) or not math.isfinite(inference_seconds) or inference_seconds < 0:
                raise ValueError("Prediction manifest inference_seconds must be finite and nonnegative")
            results["inference_seconds"] = inference_seconds
        runtime = provenance.get("runtime")
        if isinstance(runtime, dict):
            # Keep aggregate execution facts; do not copy raw responses, review
            # text, model paths, or the complete training manifest into metrics.
            runtime_keys = {"python", "platform", "packages", "git_commit", "git_dirty", "device", "gpu", "cuda"}
            results["runtime"] = {key: runtime[key] for key in runtime_keys if key in runtime}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results
