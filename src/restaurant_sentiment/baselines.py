"""Train-only majority and TF-IDF baselines with the same strict evaluator."""

from __future__ import annotations

import hashlib
import json
import importlib.metadata
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

from .data import ASPECTS, LABELS, load_jsonl
from .metrics import evaluate_predictions
from .reporting import experiment_signature


def _load_verified_splits(data_dir: Path, evaluation_split: str = "test", evaluation_path: Path | None = None) -> tuple[list[dict], list[dict], dict]:
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Data manifest must contain a fingerprint")
    file_hashes = {}
    for split in ("train",) if evaluation_path else ("train", evaluation_split):
        path = data_dir / f"{split}.jsonl"
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = manifest.get("files", {}).get(split, {}).get("sha256")
        if expected_hash != actual_hash:
            raise ValueError(f"Data fingerprint mismatch for {split}.jsonl")
        file_hashes[split] = actual_hash
    train = load_jsonl(data_dir / "train.jsonl")
    reference_path = Path(evaluation_path) if evaluation_path else data_dir / f"{evaluation_split}.jsonl"
    evaluation_manifest = manifest
    if evaluation_path:
        evaluation_manifest = json.loads((reference_path.parent / "manifest.json").read_text(encoding="utf-8"))
        expected_hash = evaluation_manifest.get("files", {}).get(reference_path.stem, {}).get("sha256")
        actual_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if expected_hash != actual_hash or not evaluation_manifest.get("fingerprint"):
            raise ValueError("External evaluation data fingerprint mismatch")
        file_hashes["evaluation"] = actual_hash
    test = load_jsonl(reference_path)
    if not train or not test:
        raise ValueError("Both train and test splits must be nonempty")
    if {row["id"] for row in train} & {row["id"] for row in test}:
        raise ValueError("Train and test splits contain overlapping review ids")
    return train, test, {
        "data_fingerprint": evaluation_manifest["fingerprint"],
        "training_data_fingerprint": fingerprint,
        "evaluation_split": "external" if evaluation_path else evaluation_split,
        "data_files_sha256": file_hashes,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    }


def _majority_labels(train: list[dict]) -> dict[str, str]:
    labels = {}
    for aspect in ASPECTS:
        counts = Counter(row["labels"][aspect] for row in train)
        # Fixed label order also makes ties deterministic across platforms.
        labels[aspect] = max(LABELS, key=lambda label: counts[label])
    return labels


def run_baseline(data_dir: Path, output_dir: Path, method: str = "majority", seed: int = 42, *,
                 evaluation_split: str = "test", evaluation_path: Path | None = None) -> dict:
    """Fit on train only and score a frozen validation, test or external set.

    ``majority`` uses the standard library. ``tfidf`` imports scikit-learn
    lazily and fits one classifier per aspect, with a constant fallback for
    an aspect that has only one observed training label.
    """
    if method not in {"majority", "tfidf"}:
        raise ValueError(f"Unknown baseline method: {method}")
    train, test, provenance = _load_verified_splits(Path(data_dir), evaluation_split, evaluation_path)
    train_texts = [row["text"] for row in train]
    test_texts = [row["text"] for row in test]
    fit_start = perf_counter()
    if method == "majority":
        majority = _majority_labels(train)
        fit_seconds = perf_counter() - fit_start
        infer_start = perf_counter()
        label_predictions = [dict(majority) for _ in test]
        model_config = {"majority_labels": majority, "tie_break_order": list(LABELS)}
    else:
        try:
            from sklearn.dummy import DummyClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError("The tfidf baseline requires scikit-learn; install the project's baseline dependencies") from exc
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
        train_features = vectorizer.fit_transform(train_texts)
        classifiers = {}
        constant_aspects = []
        for aspect in ASPECTS:
            labels = [row["labels"][aspect] for row in train]
            if len(set(labels)) == 1:
                classifier = DummyClassifier(strategy="constant", constant=labels[0])
                constant_aspects.append(aspect)
            else:
                classifier = LogisticRegression(max_iter=1_000, random_state=seed)
            classifier.fit(train_features, labels)
            classifiers[aspect] = classifier
        fit_seconds = perf_counter() - fit_start
        infer_start = perf_counter()
        test_features = vectorizer.transform(test_texts)
        by_aspect = {aspect: classifiers[aspect].predict(test_features).tolist() for aspect in ASPECTS}
        label_predictions = [{aspect: by_aspect[aspect][index] for aspect in ASPECTS} for index in range(len(test))]
        model_config = {
            "ngram_range": [1, 2],
            "max_features": 50_000,
            "sublinear_tf": True,
            "logistic_regression_max_iter": 1_000,
            "constant_aspects": constant_aspects,
            "vocabulary_size": len(vectorizer.vocabulary_),
        }
    infer_seconds = perf_counter() - infer_start
    predictions = [{"id": row["id"], "prediction": predicted} for row, predicted in zip(test, label_predictions)]
    results = evaluate_predictions(test, predictions)
    results.update({
        "method": method,
        "seed": seed,
        "training_samples": len(train),
        "labels": list(LABELS),
        "fit_seconds": fit_seconds,
        "infer_seconds": infer_seconds,
        "model_config": model_config,
        **provenance,
    })
    packages = {}
    if method == "tfidf":
        packages["scikit-learn"] = importlib.metadata.version("scikit-learn")
    results["runtime"] = {"python": platform.python_version(), "device": "cpu", "packages": packages}
    results["experiment_signature"] = experiment_signature({
        "method": method, "model_config": model_config, "packages": packages,
    })
    results["implementation_sha256"] = hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in ("data.py", "metrics.py", "baselines.py", "reporting.py")
    )).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results
