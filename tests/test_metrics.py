import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from restaurant_sentiment.data import ASPECTS, LABELS, normalize_text
from restaurant_sentiment.metrics import evaluate_files, evaluate_predictions


def reference(row_id, label):
    return {
        "id": row_id,
        "text": f"Review {row_id}",
        "restaurant": None,
        "labels": {aspect: label for aspect in ASPECTS},
    }


class MetricsTests(unittest.TestCase):
    def test_invalid_prediction_remains_in_accuracy_and_recall_denominators(self):
        references = [reference(str(index), label) for index, label in enumerate(LABELS)]
        references.append(reference("invalid", LABELS[0]))
        predictions = [{"id": row["id"], "prediction": row["labels"]} for row in references[:4]]
        # Correct on two aspects, but the complete prediction is invalid.
        invalid = dict(references[-1]["labels"])
        invalid[ASPECTS[0]] = "unknown"
        predictions.append({"id": "invalid", "prediction": invalid})
        report = evaluate_predictions(references, list(reversed(predictions)))
        self.assertAlmostEqual(report["macro_accuracy"], 0.8)
        self.assertAlmostEqual(report["macro_f1"], 11 / 12)
        self.assertAlmostEqual(report["joint_exact_match"], 0.8)
        self.assertAlmostEqual(report["json_valid_rate"], 0.8)
        for aspect in ASPECTS:
            result = report["per_aspect"][aspect]
            self.assertEqual(result["invalid_count"], 1)
            self.assertEqual(sum(map(sum, result["confusion_matrix"])), 4)
            positive = result["per_class"][LABELS[0]]
            self.assertEqual(positive["support"], 2)
            self.assertEqual(positive["precision"], 1)
            self.assertEqual(positive["recall"], 0.5)
            self.assertAlmostEqual(positive["f1"], 2 / 3)

    def test_none_prediction_is_incorrect_for_every_aspect(self):
        report = evaluate_predictions([reference("a", LABELS[0])], [{"id": "a", "prediction": None}])
        self.assertEqual(report["macro_accuracy"], 0)
        self.assertEqual(report["macro_f1"], 0)
        self.assertEqual(report["json_valid_rate"], 0)

    def test_macro_f1_keeps_absent_classes(self):
        row = reference("a", LABELS[0])
        report = evaluate_predictions([row], [{"id": "a", "prediction": row["labels"]}])
        self.assertEqual(report["macro_accuracy"], 1)
        self.assertEqual(report["macro_f1"], 0.25)

    def test_duplicate_missing_and_extra_ids_are_rejected(self):
        rows = [reference("a", LABELS[0]), reference("b", LABELS[1])]
        valid = [{"id": row["id"], "prediction": row["labels"]} for row in rows]
        invalid_predictions = [valid[:1], valid + [{"id": "extra", "prediction": None}], [valid[0], valid[0]]]
        for predictions in invalid_predictions:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_predictions(rows, predictions)
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            evaluate_predictions([rows[0], rows[0]], valid)

    def test_empty_input_and_bad_reference_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])
        row = reference("a", "unknown")
        with self.assertRaisesRegex(ValueError, "Invalid reference"):
            evaluate_predictions([row], [{"id": "a", "prediction": None}])

    def test_file_evaluation_preserves_failed_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference_path = directory / "test.jsonl"
            predictions_path = directory / "predictions.jsonl"
            output_path = directory / "reports" / "metrics.json"
            row = reference("a", LABELS[0])
            row["id"] = hashlib.sha256(normalize_text(row["text"]).encode("utf-8")).hexdigest()
            reference_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            predictions_path.write_text(json.dumps({"id": row["id"], "prediction": None}) + "\n", encoding="utf-8")
            report = evaluate_files(reference_path, predictions_path, output_path)
            self.assertEqual(report["n_samples"], 1)
            self.assertEqual(report["json_valid_rate"], 0)
            self.assertEqual(report["reference_sha256"], hashlib.sha256(reference_path.read_bytes()).hexdigest())
            self.assertIsNone(report["data_fingerprint"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), report)

    def test_file_evaluation_verifies_and_uses_prediction_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            row = reference("a", LABELS[0])
            row["id"] = hashlib.sha256(normalize_text(row["text"]).encode("utf-8")).hexdigest()
            reference_path = directory / "test.jsonl"
            reference_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(json.dumps({"fingerprint": "prepared-fixture", "files": {"test": {"sha256": reference_hash}}}), encoding="utf-8")
            predictions_path = directory / "predictions.jsonl"
            predictions_path.write_text(json.dumps({"id": row["id"], "prediction": row["labels"]}, ensure_ascii=False) + "\n", encoding="utf-8")
            provenance = {
                "backend": "lora", "input_sha256": reference_hash,
                "source_run": {"config": {"seed": 43}}, "inference_seconds": 1.5,
                "runtime": {"device": "cpu", "raw_output": "must not be copied"},
                "raw_output": "must not be copied",
            }
            predictions_path.with_suffix(".manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
            report = evaluate_files(reference_path, predictions_path, directory / "metrics.json")
            self.assertEqual(report["data_fingerprint"], "prepared-fixture")
            self.assertEqual(report["method"], "lora")
            self.assertEqual(report["seed"], 43)
            self.assertEqual(report["inference_seconds"], 1.5)
            self.assertEqual(report["runtime"], {"device": "cpu"})
            self.assertNotIn("raw_output", report)
            provenance["input_sha256"] = "wrong-input"
            predictions_path.with_suffix(".manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                evaluate_files(reference_path, predictions_path, directory / "metrics.json")


if __name__ == "__main__":
    unittest.main()
