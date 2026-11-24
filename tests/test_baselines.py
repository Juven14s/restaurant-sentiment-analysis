import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from restaurant_sentiment.baselines import run_baseline
from restaurant_sentiment.data import ASPECTS, LABELS, normalize_text


def row(row_id, label, text=None):
    text = text or f"Sample review {row_id}"
    return {
        "id": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(),
        "text": text,
        "restaurant": None,
        "labels": {aspect: label for aspect in ASPECTS},
    }


def write_data(directory, train, test):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"fingerprint": "fixture-fingerprint", "files": {}}
    for split, rows in (("train", train), ("test", test)):
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        manifest["files"][split] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": len(rows)}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class BaselineTests(unittest.TestCase):
    def test_majority_is_fit_only_on_train_despite_opposite_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_dir, output_dir = directory / "data", directory / "results"
            write_data(data_dir, [row("train-1", LABELS[0]), row("train-2", LABELS[0]), row("train-3", LABELS[1])], [row(f"test-{index}", LABELS[1]) for index in range(5)])
            report = run_baseline(data_dir, output_dir, method="majority", seed=17)
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(set(item["prediction"].values()) == {LABELS[0]} for item in predictions))
            self.assertEqual(report["macro_accuracy"], 0)
            self.assertEqual(report["training_samples"], 3)
            self.assertEqual(report["seed"], 17)
            self.assertEqual(report["data_fingerprint"], "fixture-fingerprint")
            self.assertEqual(json.loads((output_dir / "metrics.json").read_text(encoding="utf-8")), report)
            self.assertGreaterEqual(report["fit_seconds"], 0)
            self.assertGreaterEqual(report["infer_seconds"], 0)

    def test_modified_split_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_data(directory / "data", [row("train", LABELS[0])], [row("test", LABELS[0])])
            with (directory / "data" / "test.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row("extra", LABELS[1]), ensure_ascii=False) + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                run_baseline(directory / "data", directory / "results")

    def test_overlapping_train_test_reviews_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            repeated = row("repeated", LABELS[0])
            write_data(directory / "data", [repeated], [repeated])
            with self.assertRaisesRegex(ValueError, "overlapping review ids"):
                run_baseline(directory / "data", directory / "results")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn optional dependency is unavailable")
    def test_tfidf_handles_an_aspect_with_one_training_class(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train = [row("one", LABELS[0], "excellent cuisine delicieuse"), row("two", LABELS[1], "mauvais repas desastreux")]
            for item in train:
                item["labels"][ASPECTS[1]] = LABELS[3]
            write_data(directory / "data", train, [row("test", LABELS[3], "excellent repas")])
            report = run_baseline(directory / "data", directory / "results", method="tfidf")
            prediction = json.loads((directory / "results" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prediction["prediction"][ASPECTS[1]], LABELS[3])
            self.assertEqual(report["json_valid_rate"], 1)
            self.assertEqual(report["model_config"]["constant_aspects"], [ASPECTS[1]])


if __name__ == "__main__":
    unittest.main()
