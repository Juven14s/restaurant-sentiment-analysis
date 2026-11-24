"""Data invariants independent of optional model-training dependencies."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from restaurant_sentiment.data import ASPECTS, LABELS, load_jsonl, normalize_text, prepare_data


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_tsv(self, name, rows, header=("Avis", *ASPECTS, "Restaurant")):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def rows(self, count=30):
        return [[f"Avis synthétique numéro {index}", *(LABELS[index % 4] for _ in ASPECTS), f"Restaurant {index // 3}"] for index in range(count)]

    def test_split_counts_hashes_and_no_leakage(self):
        source = self.write_tsv("input.tsv", self.rows())
        output = self.root / "prepared"
        manifest = prepare_data([source], output)
        self.assertEqual(manifest["split_counts"], {"train": 24, "validation": 3, "test": 3})
        sets = []
        for split in ("train", "validation", "test"):
            rows = load_jsonl(output / f"{split}.jsonl")
            sets.append({row["id"] for row in rows})
            self.assertEqual(hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest(), manifest["files"][split]["sha256"])
            self.assertEqual(set(rows[0]["labels"]), set(ASPECTS))
        self.assertTrue(all(not sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2))))
        self.assertEqual(len(set.union(*sets)), 30)
        self.assertEqual(manifest["sources"][0]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertNotIn("Avis synthétique", (output / "manifest.json").read_text())

    def test_input_order_and_idempotence_and_frozen_refusal(self):
        first = self.write_tsv("a.tsv", self.rows()[:15])
        second = self.write_tsv("b.tsv", self.rows()[15:])
        output = self.root / "prepared"
        manifest = prepare_data([first, second], output, seed=11)
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(manifest, prepare_data([second, first], output, seed=11))
        alternate = self.root / "alternate"
        prepare_data([second, first], alternate, seed=11)
        self.assertEqual(original, {path.name: path.read_bytes() for path in alternate.iterdir()})
        with self.assertRaises(FileExistsError):
            prepare_data([first, second], output, seed=12)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_normalized_duplicates_and_all_conflicts_removed(self):
        rows = self.rows()
        rows.extend([
            ["  ＡＶＩＳ   synthétique numéro 0  ", *rows[0][1:]],
            ["AVIS SYNTHÉTIQUE NUMÉRO 1", "NE", "NE", "NE", "Restaurant 0"],
            ["avis synthétique numéro 1", *rows[1][1:]],
        ])
        manifest = prepare_data([self.write_tsv("input.tsv", rows)], self.root / "out")
        self.assertEqual(normalize_text(rows[0][0]), normalize_text(rows[30][0]))
        self.assertEqual(manifest["counts"]["input_rows"], 33)
        self.assertEqual(manifest["counts"]["retained_rows"], 29)
        self.assertEqual(manifest["counts"]["duplicate_rows"], 3)
        self.assertEqual(manifest["counts"]["concordant_duplicate_rows_removed"], 1)
        self.assertEqual(manifest["counts"]["conflicting_texts"], 1)
        self.assertEqual(manifest["counts"]["conflicting_rows_removed"], 3)
        retained = [row for split in ("train", "validation", "test") for row in load_jsonl(self.root / "out" / f"{split}.jsonl")]
        self.assertNotIn(normalize_text(rows[1][0]), {normalize_text(row["text"]) for row in retained})

    def test_relocating_sources_preserves_fingerprint_and_artifact_bytes(self):
        first = self.write_tsv("a.tsv", self.rows()[:15])
        second = self.write_tsv("b.tsv", self.rows()[15:])
        original = self.root / "original"
        expected = prepare_data([first, second], original)
        relocated = self.root / "colab" / "data"
        relocated.mkdir(parents=True)
        for source in (first, second):
            shutil.copyfile(source, relocated / source.name)
        copied = self.root / "copied"
        actual = prepare_data([relocated / "b.tsv", relocated / "a.tsv"], copied)
        self.assertEqual(expected["fingerprint"], actual["fingerprint"])
        self.assertEqual(
            {path.name: path.read_bytes() for path in original.iterdir()},
            {path.name: path.read_bytes() for path in copied.iterdir()},
        )
        self.assertTrue(all(Path(source["path"]).name == source["path"] for source in actual["sources"]))

    def test_invalid_annotation_empty_text_and_schema_raise_without_examples(self):
        for index, changed in enumerate(("", "positif", "unknown")):
            rows = self.rows()
            rows[0][1] = changed
            source = self.write_tsv(f"invalid{index}.tsv", rows)
            with self.assertRaisesRegex(ValueError, "invalid annotation") as caught:
                prepare_data([source], self.root / "out")
            self.assertNotIn(rows[0][0], str(caught.exception))
        rows = self.rows()
        rows[0][0] = " \t "
        with self.assertRaisesRegex(ValueError, "nonempty text"):
            prepare_data([self.write_tsv("empty.tsv", rows)], self.root / "out")
        with self.assertRaisesRegex(ValueError, "missing required"):
            prepare_data([self.write_tsv("schema.tsv", [["x"]], header=("Avis",))], self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_row_width_and_duplicate_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, "TSV fields"):
            prepare_data([self.write_tsv("short.tsv", [["x", "Positive"]])], self.root / "out")
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_data([self.write_tsv("columns.tsv", [], ("Avis", "Prix", "Cuisine", "Service", "Avis"))], self.root / "out")

    def test_extra_metadata_and_optional_restaurant(self):
        rows = [[row[0], *row[1:4], "5.0", "https://example.invalid"] for row in self.rows()]
        source = self.write_tsv("metadata.tsv", rows, ("Avis", *ASPECTS, "Note", "URL"))
        prepare_data([source], self.root / "out")
        self.assertTrue(all(row["restaurant"] is None for row in load_jsonl(self.root / "out/train.jsonl")))
        with self.assertRaisesRegex(ValueError, "Restaurant column required"):
            prepare_data([source], self.root / "restaurant", split_by="restaurant")

    def test_restaurants_are_disjoint_even_with_shared_duplicate_review(self):
        rows = self.rows(60)
        rows.append([rows[0][0].upper(), *rows[0][1:4], "Restaurant 1"])
        source = self.write_tsv("restaurants.tsv", rows)
        manifest = prepare_data([source], self.root / "out", split_by="restaurant")
        self.assertEqual(manifest["counts"]["texts_with_multiple_restaurants"], 1)
        self.assertTrue(all(check["restaurant_overlap"] == 0 and check["text_id_overlap"] == 0 for check in manifest["overlap_checks"].values()))
        restaurants = []
        for split in ("train", "validation", "test"):
            records = load_jsonl(self.root / "out" / f"{split}.jsonl")
            self.assertTrue(records)
            restaurants.append({normalize_text(row["restaurant"]) for row in records})
        self.assertTrue(all(not restaurants[left] & restaurants[right] for left, right in ((0, 1), (0, 2), (1, 2))))
        with self.assertRaisesRegex(ValueError, "at least three independent"):
            prepare_data([self.write_tsv("one.tsv", [[row[0], *row[1:4], "Only restaurant"] for row in self.rows()])], self.root / "one", split_by="restaurant")

    def test_changed_source_and_partial_artifacts_are_not_overwritten(self):
        source = self.write_tsv("input.tsv", self.rows())
        output = self.root / "out"
        prepare_data([source], output)
        original_test = (output / "test.jsonl").read_bytes()
        self.write_tsv("input.tsv", self.rows(33))
        with self.assertRaises(FileExistsError):
            prepare_data([source], output)
        self.assertEqual(original_test, (output / "test.jsonl").read_bytes())
        (output / "validation.jsonl").unlink()
        with self.assertRaises(FileExistsError):
            prepare_data([source], output)

    def test_loader_rejects_corruption_and_duplicate_ids(self):
        source = self.write_tsv("input.tsv", self.rows())
        prepare_data([source], self.root / "out")
        path = self.root / "out/train.jsonl"
        row = load_jsonl(path)[0]
        corrupted = self.root / "corrupted.jsonl"
        corrupted.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            load_jsonl(corrupted)
        row["id"] = "arbitrary"
        corrupted.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid or duplicate ID"):
            load_jsonl(corrupted)


if __name__ == "__main__":
    unittest.main()
