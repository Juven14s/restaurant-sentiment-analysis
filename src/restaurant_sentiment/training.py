"""Hugging Face training and inference, imported only when explicitly requested.

The lightweight data and evaluation commands do not need torch installed. Model
artifacts and resumable Trainer checkpoints are kept outside the source tree.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import ASPECTS, LABELS, normalize_text



def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_options(*, epochs: float, learning_rate: float, batch_size: int,
                      max_length: int, gradient_accumulation_steps: int,
                      max_train_samples: int | None) -> None:
    for name, value in (("epochs", epochs), ("learning_rate", learning_rate),
                        ("batch_size", batch_size), ("max_length", max_length),
                        ("gradient_accumulation_steps", gradient_accumulation_steps)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError("max_train_samples must be positive when specified.")


def _load_training_data(data_dir: Path, seed: int, maximum: int | None):
    from .data import load_jsonl

    data_dir = Path(data_dir)
    # Deliberately no test.jsonl read here: model selection uses validation only.
    fingerprints = {name: _sha256(data_dir / f"{name}.jsonl") for name in ("train", "validation")}
    manifest_path = data_dir / "manifest.json"
    data_fingerprint = None
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("fingerprint"), str) or not manifest["fingerprint"]:
            raise ValueError("Data manifest must contain a fingerprint.")
        data_fingerprint = manifest["fingerprint"]
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Data manifest must contain frozen split hashes.")
        for split in ("train", "validation"):
            entry = files.get(split)
            if not isinstance(entry, dict) or entry.get("sha256") != fingerprints[split]:
                raise ValueError(f"Data fingerprint mismatch for {split}.jsonl. Prepare a new frozen dataset instead of editing it.")
        fingerprints["data_manifest"] = hashlib.sha256(manifest_bytes).hexdigest()
    train = load_jsonl(data_dir / "train.jsonl")
    validation = load_jsonl(data_dir / "validation.jsonl")
    if not train or not validation:
        raise ValueError("Training needs nonempty train.jsonl and validation.jsonl.")
    train_ids = {row["id"] for row in train}
    val_ids = {row["id"] for row in validation}
    train_texts = {normalize_text(row["text"]) for row in train}
    val_texts = {normalize_text(row["text"]) for row in validation}
    if train_ids & val_ids or train_texts & val_texts:
        raise ValueError("Train and validation overlap. Prepare disjoint splits before training.")
    if maximum is not None and maximum < len(train):
        indices = sorted(random.Random(seed).sample(range(len(train)), maximum))
        train = [train[index] for index in indices]
    return train, validation, fingerprints, data_fingerprint


def _torch_and_device(device: str, *, require_cuda: bool = False):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the training dependencies first: pip install -e '.[training]'") from exc
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab, or use --device cpu for CamemBERT.")
    if require_cuda and selected != "cuda":
        raise RuntimeError("SFT requires a CUDA GPU in this project. In Colab choose Runtime > Change runtime type > T4 GPU.")
    return torch, selected


def _runtime(torch, device: str) -> dict:
    versions = {}
    for package in ("torch", "transformers", "datasets", "accelerate", "trl", "peft", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "git_commit": revision, "git_dirty": dirty,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda": torch.version.cuda if device == "cuda" else None,
    }


def _prepare_run(output_dir: Path, config: dict, fingerprints: dict, resume: str | None,
                 data_fingerprint: str | None = None) -> dict:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError("Output directory is not empty. Use a new run directory or resume_from_checkpoint.")
    old = {}
    if resume:
        checkpoint = Path(resume)
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        if checkpoint.resolve().parent != output_dir.resolve():
            raise ValueError("Resume checkpoint must belong to the original output directory.")
        if any(not (checkpoint / name).is_file() for name in required):
            raise ValueError("Resume requires a complete Trainer checkpoint: trainer, optimizer, scheduler and RNG state.")
        if not manifest_path.exists():
            raise ValueError("Resume requires the original output directory and its run_manifest.json.")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config") != config or old.get("data_sha256") != fingerprints or old.get("data_fingerprint") != data_fingerprint:
            raise ValueError("Resume configuration or data changed. Start a separate run for a new experiment.")
    return {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "data_sha256": fingerprints,
        "data_fingerprint": data_fingerprint,
        "resumed_from_checkpoint": resume,
        "base_model_revision": old.get("base_model_revision"),
    }


def _training_arguments(output_dir: Path, n_train: int, config: dict, torch, device: str) -> dict:
    updates_per_epoch = math.ceil(math.ceil(n_train / config["batch_size"]) / config["gradient_accumulation_steps"])
    total_updates = max(1, math.ceil(updates_per_epoch * config["epochs"]))
    interval = min(50, updates_per_epoch, total_updates)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    return dict(
        output_dir=str(output_dir), num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"], per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        save_total_limit=2, save_only_model=False, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=config["seed"], data_seed=config["seed"],
        logging_steps=min(10, interval), logging_first_step=True, report_to="none",
        fp16=device == "cuda" and not bf16, bf16=bf16, use_cpu=device == "cpu",
        optim="adamw_torch", weight_decay=0.01, warmup_ratio=0.05,
        dataloader_num_workers=0, dataloader_pin_memory=device == "cuda",
        push_to_hub=False,
    )


def _finish_training(trainer, tokenizer, manifest: dict, output_dir: Path, started: float, torch, device: str) -> dict:
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    validation_metrics = trainer.evaluate()
    manifest.update(
        status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        model_dir=str(final_dir), best_checkpoint=trainer.state.best_model_checkpoint,
        best_validation_loss=trainer.state.best_metric, validation_metrics=validation_metrics,
    )
    _write_json(Path(output_dir) / "run_manifest.json", manifest)
    # Copy provenance beside the model so inference remains reproducible after moving it.
    _write_json(final_dir / "run_manifest.json", manifest)
    return {**manifest, "manifest_path": str(Path(output_dir) / "run_manifest.json")}


def train_camembert(data_dir: Path, output_dir: Path, *, model_name: str = "almanach/camembert-base",
                    seed: int = 42, epochs: float = 3.0, learning_rate: float = 3e-5,
                    batch_size: int = 8, max_length: int = 256, gradient_accumulation_steps: int = 1,
                    device: str = "auto", max_train_samples: int | None = None,
                    model_revision: str | None = None, resume_from_checkpoint: str | None = None) -> dict:
    """Fine-tune a 12-logit classifier as three independent four-class tasks."""
    config = dict(backend="camembert", model_name=model_name, model_revision=model_revision, seed=seed,
                  epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, max_length=max_length,
                  gradient_accumulation_steps=gradient_accumulation_steps, max_train_samples=max_train_samples)
    _validate_options(**{key: config[key] for key in ("epochs", "learning_rate", "batch_size", "max_length", "gradient_accumulation_steps", "max_train_samples")})
    train, validation, fingerprints, data_fingerprint = _load_training_data(data_dir, seed, max_train_samples)
    manifest = _prepare_run(Path(output_dir), config, fingerprints, resume_from_checkpoint, data_fingerprint)
    torch, selected = _torch_and_device(device)
    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("Install dependencies with pip install -e '.[training]'.") from exc
    set_seed(seed)
    effective_revision = manifest.get("base_model_revision") or model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=effective_revision)
    flat_labels = [f"{aspect}:{label}" for aspect in ASPECTS for label in LABELS]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=effective_revision, num_labels=12,
        id2label=dict(enumerate(flat_labels)), label2id={value: index for index, value in enumerate(flat_labels)},
    )
    model.config.restaurant_sentiment = {"aspects": list(ASPECTS), "labels": list(LABELS)}
    manifest.update(runtime=_runtime(torch, selected), base_model_revision=getattr(model.config, "_commit_hash", None),
                    training_examples=len(train), validation_examples=len(validation))
    _write_json(Path(output_dir) / "run_manifest.json", manifest)

    def dataset(rows):
        records = [{"text": row["text"], "labels": [LABELS.index(row["labels"][aspect]) for aspect in ASPECTS]} for row in rows]
        return Dataset.from_list(records).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["text"],
        )

    def multi_aspect_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.reshape(-1, len(LABELS))
        loss = torch.nn.functional.cross_entropy(logits, labels.reshape(-1), reduction="sum")
        return loss / (num_items_in_batch if num_items_in_batch is not None else labels.numel())

    trainer = Trainer(
        model=model, args=TrainingArguments(**_training_arguments(Path(output_dir), len(train), config, torch, selected)),
        train_dataset=dataset(train), eval_dataset=dataset(validation), processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_loss_func=multi_aspect_loss,
    )
    if selected == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return _finish_training(trainer, tokenizer, manifest, Path(output_dir), started, torch, selected)


def _render_prompt(tokenizer, text: str) -> str:
    from .prompts import build_messages

    return tokenizer.apply_chat_template(
        build_messages(text if text.strip() else "…"), tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _bounded_prompt(tokenizer, text: str, max_length: int, completion: str = "") -> tuple[str, bool]:
    """Truncate only the review, retaining the instructions and complete target."""
    def fits(prompt):
        return len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]) <= max_length

    prompt = _render_prompt(tokenizer, text)
    if fits(prompt):
        return prompt, False
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    low, high = 0, len(tokens)
    best = _render_prompt(tokenizer, "")
    if not fits(best):
        raise ValueError("max_length is too small for the instructions and complete JSON target. Increase it.")
    while low <= high:
        middle = (low + high) // 2
        candidate = _render_prompt(tokenizer, tokenizer.decode(tokens[:middle], skip_special_tokens=True))
        if fits(candidate):
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best, True


def _sft_records(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    records, truncated = [], 0
    if not tokenizer.eos_token:
        raise ValueError("SFT requires a tokenizer with an EOS token.")
    for row in rows:
        completion = json.dumps({aspect: row["labels"][aspect] for aspect in ASPECTS}, ensure_ascii=False) + tokenizer.eos_token
        prompt, cut = _bounded_prompt(tokenizer, row["text"], max_length, completion)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        combined_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        if combined_ids[:len(prompt_ids)] != prompt_ids or len(combined_ids) <= len(prompt_ids):
            raise ValueError("Tokenizer changed the prompt/completion boundary; completion-only training would be incorrect.")
        # Already tokenized to avoid a second implicit chat template or BOS token.
        records.append({"input_ids": combined_ids, "completion_mask": [0] * len(prompt_ids) + [1] * (len(combined_ids) - len(prompt_ids))})
        truncated += int(cut)
    return records, truncated


def _causal_model_kwargs(torch, device: str, quantize_4bit: bool, revision: str | None) -> dict:
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32
    kwargs = {"revision": revision, "torch_dtype": dtype}
    if quantize_4bit:
        if device != "cuda":
            raise ValueError("4-bit quantization requires CUDA in this project.")
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install QLoRA dependencies with pip install -e '.[quantization]'.") from exc
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ), device_map={"": torch.cuda.current_device()})
    return kwargs


def train_sft(data_dir: Path, output_dir: Path, *, model_name: str = "Qwen/Qwen3-0.6B",
              seed: int = 42, epochs: float = 1.0, learning_rate: float = 2e-4,
              batch_size: int = 1, max_length: int = 512, gradient_accumulation_steps: int = 8,
              device: str = "auto", quantize_4bit: bool = False, lora_rank: int = 8,
              max_train_samples: int | None = None, model_revision: str | None = None,
              resume_from_checkpoint: str | None = None) -> dict:
    """Train a LoRA adapter; only JSON completion tokens contribute to loss."""
    from .prompts import PROMPT_VERSION, SYSTEM_PROMPT

    config = dict(backend="lora", model_name=model_name, model_revision=model_revision, seed=seed,
                  epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, max_length=max_length,
                  gradient_accumulation_steps=gradient_accumulation_steps, max_train_samples=max_train_samples,
                  quantize_4bit=quantize_4bit, lora_rank=lora_rank, enable_thinking=False,
                  prompt_version=PROMPT_VERSION, system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest())
    _validate_options(**{key: config[key] for key in ("epochs", "learning_rate", "batch_size", "max_length", "gradient_accumulation_steps", "max_train_samples")})
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive.")
    train, validation, fingerprints, data_fingerprint = _load_training_data(data_dir, seed, max_train_samples)
    manifest = _prepare_run(Path(output_dir), config, fingerprints, resume_from_checkpoint, data_fingerprint)
    torch, selected = _torch_and_device(device, require_cuda=True)
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install dependencies with pip install -e '.[training]'.") from exc
    set_seed(seed)
    effective_revision = manifest.get("base_model_revision") or model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=effective_revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, **_causal_model_kwargs(torch, selected, quantize_4bit, effective_revision))
    base_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    if quantize_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    train_records, train_truncated = _sft_records(train, tokenizer, max_length)
    val_records, val_truncated = _sft_records(validation, tokenizer, max_length)
    manifest.update(runtime=_runtime(torch, selected), base_model_revision=base_revision,
                    training_examples=len(train), validation_examples=len(validation),
                    truncated_reviews={"train": train_truncated, "validation": val_truncated},
                    truncation="review tokens only; system prompt and full JSON completion retained")
    _write_json(Path(output_dir) / "run_manifest.json", manifest)
    arguments = _training_arguments(Path(output_dir), len(train), config, torch, selected)
    trainer = SFTTrainer(
        model=model, args=SFTConfig(**arguments, max_length=max_length, completion_only_loss=True,
            packing=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
        processing_class=tokenizer, train_dataset=Dataset.from_list(train_records), eval_dataset=Dataset.from_list(val_records),
        peft_config=LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                               bias="none", target_modules="all-linear", task_type=TaskType.CAUSAL_LM,
                               revision=base_revision or model_revision),
    )
    if selected == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return _finish_training(trainer, tokenizer, manifest, Path(output_dir), started, torch, selected)


def predict_model(data_path: Path, output_path: Path, model_path: str, backend: str, *,
                  device: str = "auto", batch_size: int = 8, max_length: int = 512,
                  max_new_tokens: int = 96, quantize_4bit: bool = False,
                  model_revision: str | None = None) -> dict:
    """Write ID-aligned predictions and separately measured inference provenance."""
    if backend not in ("camembert", "llm", "lora"):
        raise ValueError("backend must be camembert, llm, or lora.")
    if min(batch_size, max_length, max_new_tokens) <= 0:
        raise ValueError("batch_size, max_length and max_new_tokens must be positive.")
    data_path, output_path = Path(data_path), Path(output_path)
    if data_path.resolve() == output_path.resolve():
        raise ValueError("Prediction output must not overwrite the source data.")
    if backend == "camembert" and quantize_4bit:
        raise ValueError("4-bit inference is supported for llm and lora only.")
    from .data import load_jsonl
    from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, parse_prediction

    rows = load_jsonl(data_path)
    if not rows:
        raise ValueError("Prediction data is empty.")
    torch, selected = _torch_and_device(device)
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install dependencies with pip install -e '.[training]'.") from exc
    source_manifest = Path(model_path) / "run_manifest.json"
    provenance = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else None
    loading_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision)
    if backend == "camembert":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, revision=model_revision)
        expected = {"aspects": list(ASPECTS), "labels": list(LABELS)}
        if model.config.num_labels != 12 or getattr(model.config, "restaurant_sentiment", None) != expected:
            raise ValueError("Use a CamemBERT model trained with this project's three-aspect classification head.")
    else:
        kwargs = _causal_model_kwargs(torch, selected, quantize_4bit, model_revision)
        if backend == "lora":
            from peft import PeftConfig, PeftModel

            adapter = PeftConfig.from_pretrained(model_path, revision=model_revision)
            base_revision = (provenance or {}).get("base_model_revision") or adapter.revision
            kwargs["revision"] = base_revision
            model = AutoModelForCausalLM.from_pretrained(adapter.base_model_name_or_path, **kwargs)
            model = PeftModel.from_pretrained(model, model_path, revision=model_revision)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
    if not quantize_4bit:
        model.to(selected)
    model.eval()
    if backend != "camembert":
        model.config.use_cache = True
    if selected == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.perf_counter() - loading_started
    started = time.perf_counter()
    predictions, truncated = [], 0
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            if backend == "camembert":
                encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt").to(selected)
                indices = model(**encoded).logits.reshape(-1, len(ASPECTS), len(LABELS)).argmax(-1).cpu().tolist()
                predictions.extend({"id": row["id"], "prediction": {aspect: LABELS[index] for aspect, index in zip(ASPECTS, labels)}}
                                   for row, labels in zip(batch, indices))
            else:
                prompts = []
                for row in batch:
                    prompt, cut = _bounded_prompt(tokenizer, row["text"], max_length)
                    prompts.append(prompt)
                    truncated += int(cut)
                encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(selected)
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                raw_outputs = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                predictions.extend({"id": row["id"], "prediction": parse_prediction(raw), "raw_output": raw}
                                   for row, raw in zip(batch, raw_outputs))
    if selected == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    result = dict(
        backend=backend, model_path=model_path, requested_revision=model_revision,
        resolved_revision=getattr(model.config, "_commit_hash", None), source_run=provenance,
        input_sha256=_sha256(data_path), n_predictions=len(predictions), runtime=_runtime(torch, selected),
        batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens,
        quantize_4bit=quantize_4bit, enable_thinking=False if backend != "camembert" else None,
        prompt_version=PROMPT_VERSION if backend != "camembert" else None,
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest() if backend != "camembert" else None,
        truncated_reviews=truncated if backend != "camembert" else None,
        model_load_seconds=load_seconds, inference_seconds=elapsed,
        mean_seconds_per_example=elapsed / len(rows), examples_per_second=len(rows) / elapsed,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if selected == "cuda" else None,
        output_path=str(output_path), manifest_path=str(output_path.with_suffix(".manifest.json")),
    )
    _write_json(output_path.with_suffix(".manifest.json"), result)
    return result
