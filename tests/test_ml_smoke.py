"""Offline integration checks for real, tiny CPU training and model reloading.

No weights, tokenizer, or dataset are downloaded. Base installations skip these
tests; install the training extra to exercise the optional Hugging Face stack.
Random models verify pipeline invariants, never the scientific model quality.
"""

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from restaurant_sentiment.data import ASPECTS, LABELS, load_jsonl, prepare_data
from restaurant_sentiment.metrics import evaluate_files
from restaurant_sentiment.prompts import SYSTEM_PROMPT


def _available(*packages):
    return all(importlib.util.find_spec(package) is not None for package in packages)


_ENCODER_AVAILABLE = _available("torch", "transformers", "tokenizers", "datasets", "accelerate")
_SFT_AVAILABLE = _ENCODER_AVAILABLE and _available("trl", "peft", "safetensors")


class _OfflineTrainingFixtures:
    """Small, local fixtures shared by optional integration test classes."""

    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(self.root / "hf-cache"),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        old = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)

        def restore_environment():
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.addCleanup(restore_environment)
        source = self.root / "synthetic.tsv"
        self.review_texts = [f"Avis synthétique {index} : cuisine agréable et service rapide." for index in range(20)]
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["Avis", *ASPECTS, "Restaurant"])
            for index, text in enumerate(self.review_texts):
                writer.writerow([text, *(LABELS[(index + aspect) % len(LABELS)] for aspect in range(len(ASPECTS))), f"Restaurant {index // 2}"])
        self.data_dir = self.root / "data"
        self.data_manifest = prepare_data([source], self.data_dir)

    def tokenizer(self, directory):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
        from transformers import PreTrainedTokenizerFast

        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"]
        raw = Tokenizer(models.BPE(unk_token="<unk>"))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        raw.train_from_iterator(
            [SYSTEM_PROMPT, *self.review_texts, *[
                json.dumps({aspect: label for aspect in ASPECTS}, ensure_ascii=False)
                for label in LABELS
            ]],
            trainers.BpeTrainer(vocab_size=384, min_frequency=1,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=special_tokens),
        )
        raw.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[("<s>", raw.token_to_id("<s>")), ("</s>", raw.token_to_id("</s>"))],
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="<unk>", bos_token="<s>",
            eos_token="<|im_end|>", pad_token="<pad>", model_max_length=2048,
            model_input_names=["input_ids", "attention_mask"],
        )
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        tokenizer.save_pretrained(directory)
        return tokenizer

    def assert_completed_run(self, manifest, output_dir):
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["training_examples"], 4)
        self.assertEqual(manifest["validation_examples"], 2)
        self.assertTrue(math.isfinite(manifest["best_validation_loss"]))
        self.assertEqual(manifest["runtime"]["device"], "cpu")
        self.assertIsNone(manifest["peak_gpu_memory_bytes"])
        model_dir = Path(manifest["model_dir"])
        self.assertEqual(model_dir, output_dir / "final")
        checkpoint = Path(manifest["best_checkpoint"])
        self.assertTrue((checkpoint / "trainer_state.json").is_file())
        checkpoint_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertGreater(checkpoint_state["global_step"], 0)
        self.assertEqual(checkpoint_state["best_model_checkpoint"], str(checkpoint))
        saved = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["config"], manifest["config"])
        self.assertEqual(saved["data_sha256"], manifest["data_sha256"])
        self.assertEqual(saved["best_checkpoint"], manifest["best_checkpoint"])
        self.assertEqual(saved["best_validation_loss"], manifest["best_validation_loss"])
        return model_dir, checkpoint

    def assert_predictions_evaluate(self, path, prediction_manifest):
        references = load_jsonl(self.data_dir / "test.jsonl")
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["id"] for row in predictions}, {row["id"] for row in references})
        self.assertEqual(prediction_manifest["n_predictions"], len(references))
        self.assertEqual(len(predictions), len(references))
        scores = evaluate_files(self.data_dir / "test.jsonl", path, self.root / "metrics.json")
        self.assertEqual(scores["n_samples"], len(references))
        self.assertEqual(scores["data_fingerprint"], self.data_manifest["fingerprint"])
        self.assertTrue(math.isfinite(scores["macro_f1"]))
        return predictions, scores


@unittest.skipUnless(_ENCODER_AVAILABLE, "Optional torch/transformers/datasets training dependencies are not installed")
class EncoderTrainingSmokeTests(_OfflineTrainingFixtures, unittest.TestCase):
    def test_actual_classifier_training_best_checkpoint_reload_and_prediction(self):
        from transformers import AutoModelForSequenceClassification, RobertaConfig
        from restaurant_sentiment.training import predict_model, train_camembert

        base_dir = self.root / "tiny-roberta"
        tokenizer = self.tokenizer(base_dir)
        self.torch.manual_seed(7)
        config = RobertaConfig(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=64, max_position_embeddings=130,
            num_labels=12, pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        AutoModelForSequenceClassification.from_config(config).save_pretrained(base_dir)
        output_dir = self.root / "encoder-run"
        manifest = train_camembert(
            self.data_dir, output_dir, model_name=str(base_dir), device="cpu",
            epochs=0.5, max_train_samples=4, batch_size=2, max_length=64,
            gradient_accumulation_steps=1,
        )
        model_dir, checkpoint = self.assert_completed_run(manifest, output_dir)
        restored = AutoModelForSequenceClassification.from_pretrained(model_dir)
        best = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.assertEqual(restored.config.restaurant_sentiment, {"aspects": list(ASPECTS), "labels": list(LABELS)})
        self.assertEqual(restored.config.num_labels, 12)
        for name, tensor in restored.state_dict().items():
            self.assertTrue(self.torch.equal(tensor, best.state_dict()[name]), name)
        prediction_path = self.root / "encoder-predictions.jsonl"
        prediction_manifest = predict_model(
            self.data_dir / "test.jsonl", prediction_path, str(model_dir), "camembert",
            device="cpu", batch_size=2, max_length=64,
        )
        predictions, scores = self.assert_predictions_evaluate(prediction_path, prediction_manifest)
        self.assertEqual(scores["json_valid_rate"], 1.0)
        self.assertTrue(all(set(row["prediction"]) == set(ASPECTS) for row in predictions))
        self.assertEqual(prediction_manifest["source_run"]["data_sha256"], manifest["data_sha256"])


@unittest.skipUnless(_SFT_AVAILABLE, "Optional TRL/PEFT training dependencies are not installed")
class SFTTrainingSmokeTests(_OfflineTrainingFixtures, unittest.TestCase):
    def save_tiny_causal_model(self):
        from transformers import AutoModelForCausalLM, Qwen3Config

        base_dir = self.root / "tiny-qwen3"
        tokenizer = self.tokenizer(base_dir)
        self.torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            intermediate_size=64, max_position_embeddings=1152,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=True,
        )
        AutoModelForCausalLM.from_config(config).save_pretrained(base_dir)
        return base_dir, tokenizer

    def test_actual_completion_only_lora_training_reload_and_prediction(self):
        from safetensors.torch import load_file
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        from restaurant_sentiment import training

        base_dir, tokenizer = self.save_tiny_causal_model()
        records, truncated = training._sft_records(load_jsonl(self.data_dir / "train.jsonl")[:4], tokenizer, 1024)
        self.assertEqual(truncated, 0)
        for record in records:
            mask = record["completion_mask"]
            self.assertEqual(len(mask), len(record["input_ids"]))
            self.assertGreater(sum(mask), 0)
            self.assertGreater(mask.count(0), 0)
            self.assertEqual(mask, sorted(mask))
            target = tokenizer.decode([token for token, selected in zip(record["input_ids"], mask) if selected], skip_special_tokens=True)
            self.assertEqual(set(json.loads(target)), set(ASPECTS))
        collated = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id, completion_only_loss=True,
        )(records)
        for index, record in enumerate(records):
            expected = [token if selected else -100 for token, selected in zip(record["input_ids"], record["completion_mask"])]
            expected.extend([-100] * (collated["labels"].shape[1] - len(expected)))
            self.assertEqual(collated["labels"][index].tolist(), expected)
        output_dir = self.root / "lora-run"
        # Production requires CUDA. This is the sole backend mock: all tokenizer,
        # model, TRL loss masking, optimization, save/load, and inference are real.
        with patch.object(training, "_torch_and_device", return_value=(self.torch, "cpu")) as device_check:
            manifest = training.train_sft(
                self.data_dir, output_dir, model_name=str(base_dir), device="cpu",
                epochs=1.0, max_train_samples=4, batch_size=2, max_length=1024,
                gradient_accumulation_steps=1, lora_rank=2, learning_rate=1e-3,
            )
        device_check.assert_called_once_with("cpu", require_cuda=True)
        model_dir, checkpoint = self.assert_completed_run(manifest, output_dir)
        self.assertTrue((model_dir / "adapter_config.json").is_file())
        final_weights = load_file(model_dir / "adapter_model.safetensors")
        checkpoint_weights = load_file(checkpoint / "adapter_model.safetensors")
        self.assertEqual(set(final_weights), set(checkpoint_weights))
        self.assertTrue(any(self.torch.count_nonzero(value).item() for name, value in final_weights.items() if "lora_B" in name))
        for name, value in final_weights.items():
            self.assertTrue(self.torch.equal(value, checkpoint_weights[name]), name)
        prediction_path = self.root / "lora-predictions.jsonl"
        prediction_manifest = training.predict_model(
            self.data_dir / "test.jsonl", prediction_path, str(model_dir), "lora",
            device="cpu", batch_size=2, max_length=1024, max_new_tokens=4,
        )
        predictions, _ = self.assert_predictions_evaluate(prediction_path, prediction_manifest)
        self.assertTrue(all("raw_output" in row for row in predictions))
        self.assertEqual(prediction_manifest["source_run"]["best_checkpoint"], manifest["best_checkpoint"])
        self.assertFalse(prediction_manifest["enable_thinking"])

    def test_actual_gradient_accumulation_checkpoint_resume_matches_uninterrupted(self):
        from safetensors.torch import load_file
        from trl import SFTTrainer
        from restaurant_sentiment import training

        base_dir, _ = self.save_tiny_causal_model()
        arguments = dict(
            model_name=str(base_dir), device="cpu", seed=23,
            epochs=2.0, max_train_samples=4, batch_size=2, max_length=1024,
            gradient_accumulation_steps=2, lora_rank=2, learning_rate=1e-3,
        )
        resumed_dir = self.root / "resumed-run"
        control_dir = self.root / "uninterrupted-run"
        saved_checkpoints = []

        class DeliberateInterruption(RuntimeError):
            pass

        original_save_checkpoint = SFTTrainer._save_checkpoint

        def interrupt_after_real_checkpoint(trainer, *args, **kwargs):
            original_save_checkpoint(trainer, *args, **kwargs)
            saved_checkpoints.append(Path(trainer.args.output_dir) / f"checkpoint-{trainer.state.global_step}")
            raise DeliberateInterruption("simulate a Colab interruption after a complete checkpoint")

        # Preserve the real checkpoint implementation and fail only after model,
        # optimizer, scheduler and RNG states have actually reached disk.
        with patch.object(training, "_torch_and_device", return_value=(self.torch, "cpu")):
            with patch.object(SFTTrainer, "_save_checkpoint", new=interrupt_after_real_checkpoint):
                with self.assertRaises(DeliberateInterruption):
                    training.train_sft(self.data_dir, resumed_dir, **arguments)
            self.assertEqual(len(saved_checkpoints), 1)
            checkpoint = saved_checkpoints[0]
            checkpoint_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_state["global_step"], 1)
            self.assertEqual(checkpoint_state["max_steps"], 2)
            for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
                self.assertTrue((checkpoint / filename).is_file(), filename)
            resumed = training.train_sft(
                self.data_dir, resumed_dir, resume_from_checkpoint=str(checkpoint), **arguments,
            )
            control = training.train_sft(self.data_dir, control_dir, **arguments)
        resumed_model, _ = self.assert_completed_run(resumed, resumed_dir)
        control_model, _ = self.assert_completed_run(control, control_dir)
        self.assertEqual(resumed["resumed_from_checkpoint"], str(checkpoint))
        self.assertEqual(resumed["config"], control["config"])
        self.assertEqual(resumed["data_sha256"], control["data_sha256"])
        self.assertEqual(resumed["config"]["gradient_accumulation_steps"], 2)
        for directory in (resumed_dir, control_dir):
            state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["global_step"], 2)
            self.assertEqual(state["epoch"], 2.0)
        resumed_weights = load_file(resumed_model / "adapter_model.safetensors")
        control_weights = load_file(control_model / "adapter_model.safetensors")
        self.assertEqual(set(resumed_weights), set(control_weights))
        for name, value in resumed_weights.items():
            self.assertTrue(self.torch.equal(value, control_weights[name]), name)
        self.assertEqual(resumed["best_validation_loss"], control["best_validation_loss"])


if __name__ == "__main__":
    unittest.main()
