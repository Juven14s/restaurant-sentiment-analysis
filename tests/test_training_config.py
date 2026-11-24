"""Protect data isolation, resumability and completion masking without torch."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from restaurant_sentiment.data import ASPECTS, normalize_text
from restaurant_sentiment.training import (
    _bounded_prompt, _load_training_data, _prepare_run, _render_prompt,
    _sft_records, predict_model, train_camembert, train_sft,
)


class CharacterTokenizer:
    """Transparent stand-in: a token is a Unicode character, with no ML dependency."""

    eos_token = "§"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("enable_thinking") is not False:
            raise AssertionError("Thinking must stay disabled in every prompt")
        return "\n".join(message["content"] for message in messages) + "\nassistant:"


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def row(self, text):
        return {"id": hashlib.sha256(normalize_text(text).encode()).hexdigest(), "text": text,
                "labels": {aspect: "NE" for aspect in ASPECTS}, "restaurant": None}

    def write_split(self, split, rows):
        (self.root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )

    def test_training_only_reads_train_validation_even_if_test_is_unreadable(self):
        self.write_split("train", [self.row(f"Review {index}") for index in range(10)])
        self.write_split("validation", [self.row("Independent validation")])
        (self.root / "test.jsonl").write_text("THIS IS NOT JSON", encoding="utf-8")
        train, validation, fingerprints, data_fingerprint = _load_training_data(self.root, seed=42, maximum=3)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 1)
        self.assertEqual(set(fingerprints), {"train", "validation"})
        self.assertIsNone(data_fingerprint)
        self.assertEqual(train, _load_training_data(self.root, seed=42, maximum=3)[0])

    def test_frozen_manifest_rejects_valid_annotation_tampering_without_reading_test(self):
        train = [self.row("Independent training review")]
        validation = [self.row("Independent validation review")]
        self.write_split("train", train)
        self.write_split("validation", validation)
        (self.root / "test.jsonl").write_text("UNREADABLE TEST JSON", encoding="utf-8")
        originals = {split: (self.root / f"{split}.jsonl").read_bytes() for split in ("train", "validation")}
        manifest = {"fingerprint": "frozen-dataset", "files": {
            split: {"sha256": hashlib.sha256(content).hexdigest()} for split, content in originals.items()
        }}
        # The test is deliberately absent from manifest.files as well.
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _, _, hashes, fingerprint = _load_training_data(self.root, seed=42, maximum=None)
        self.assertEqual(fingerprint, "frozen-dataset")
        self.assertEqual(set(hashes), {"train", "validation", "data_manifest"})
        for split, rows in (("train", train), ("validation", validation)):
            with self.subTest(split=split):
                changed = [{**row, "labels": {**row["labels"], "Prix": "Positive"}} for row in rows]
                self.write_split(split, changed)
                with self.assertRaisesRegex(ValueError, f"fingerprint mismatch for {split}"):
                    _load_training_data(self.root, seed=42, maximum=None)
                (self.root / f"{split}.jsonl").write_bytes(originals[split])

    def test_overlap_rejected_before_training(self):
        self.write_split("train", [self.row("Très bon service")])
        self.write_split("validation", [self.row("  TRÈS   BON SERVICE ")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            _load_training_data(self.root, seed=42, maximum=None)

    def test_invalid_configuration_fails_before_loading_data_or_dependencies(self):
        for function in (train_camembert, train_sft):
            for options in ({"epochs": 0}, {"batch_size": 0}, {"learning_rate": float("nan")}, {"max_train_samples": -1}):
                with self.assertRaises(ValueError):
                    function(self.root / "absent", self.root / "run", **options)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            predict_model(self.root / "data.jsonl", self.root / "data.jsonl", "unused", "llm")

    def test_resume_requires_matching_frozen_data_and_configuration(self):
        output = self.root / "run"
        output.mkdir()
        checkpoint = output / "checkpoint-1"
        checkpoint.mkdir()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_text("{}")
        config = {"seed": 42, "max_length": 512}
        fingerprints = {"train": "a", "validation": "b"}
        (output / "run_manifest.json").write_text(json.dumps({
            "config": config, "data_sha256": fingerprints, "base_model_revision": "pinned-commit",
        }))
        with self.assertRaisesRegex(ValueError, "not empty"):
            _prepare_run(output, config, fingerprints, None)
        with self.assertRaisesRegex(ValueError, "changed"):
            _prepare_run(output, {**config, "seed": 12}, fingerprints, str(checkpoint))
        with self.assertRaisesRegex(ValueError, "changed"):
            _prepare_run(output, config, {**fingerprints, "train": "edited"}, str(checkpoint))
        manifest = _prepare_run(output, config, fingerprints, str(checkpoint))
        self.assertEqual(manifest["base_model_revision"], "pinned-commit")
        (checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "complete Trainer checkpoint"):
            _prepare_run(output, config, fingerprints, str(checkpoint))

    def test_long_review_keeps_full_target_and_masks_all_prompt_tokens(self):
        tokenizer = CharacterTokenizer()
        review = "Très bon service. " * 100
        row = self.row(review)
        # Leave enough space for instructions + the full response, but not the review.
        target = json.dumps(row["labels"], ensure_ascii=False) + tokenizer.eos_token
        budget = len(_render_prompt(tokenizer, "Court")) + len(target) + 12
        records, truncated = _sft_records([row], tokenizer, budget)
        self.assertEqual(truncated, 1)
        record = records[0]
        self.assertLessEqual(len(record["input_ids"]), budget)
        supervised = [token for token, mask in zip(record["input_ids"], record["completion_mask"]) if mask]
        self.assertEqual(tokenizer.decode(supervised), target)
        prompt = tokenizer.decode(record["input_ids"][:-len(supervised)])
        self.assertIn("Tu analyses les opinions", prompt)
        self.assertNotIn(review, prompt)
        self.assertTrue(all(value == 0 for value in record["completion_mask"][:-len(supervised)]))

    def test_too_short_context_fails_instead_of_removing_the_target(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            _bounded_prompt(CharacterTokenizer(), "Avis long " * 20, max_length=5, completion='{"Prix":"NE"}')


if __name__ == "__main__":
    unittest.main()
