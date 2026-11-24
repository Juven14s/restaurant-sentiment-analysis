"""Command-line workflow; heavyweight dependencies load only when requested."""

import argparse
import json
from pathlib import Path
import sys


def _training_parser(commands):
    p = commands.add_parser("train", help="Train using train/validation only")
    p.add_argument("--backend", choices=["camembert", "sft"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name")
    p.add_argument("--model-revision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-length", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--quantize-4bit", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)


def parser():
    p = argparse.ArgumentParser(description="French aspect sentiment research workflow")
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate, deduplicate and freeze splits")
    prep.add_argument("--input", type=Path, nargs="+", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--split-by", choices=["text", "restaurant"], default="text")

    baseline = commands.add_parser("baseline", help="Fit a baseline and evaluate it on the frozen test")
    baseline.add_argument("--data-dir", type=Path, required=True)
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--method", choices=["majority", "tfidf"], default="majority")
    baseline.add_argument("--seed", type=int, default=42)
    _training_parser(commands)

    pred = commands.add_parser("predict", help="Generate predictions with a saved or Hub model")
    pred.add_argument("--data-path", type=Path, required=True)
    pred.add_argument("--output-path", type=Path, required=True)
    pred.add_argument("--model-path", required=True)
    pred.add_argument("--backend", choices=["camembert", "llm", "lora"], required=True)
    pred.add_argument("--model-revision")
    pred.add_argument("--device", default="auto")
    pred.add_argument("--batch-size", type=int, default=1)
    pred.add_argument("--max-length", type=int, default=512)
    pred.add_argument("--max-new-tokens", type=int, default=96)
    pred.add_argument("--quantize-4bit", action="store_true")

    ev = commands.add_parser("evaluate", help="Score ID-aligned predictions, counting invalid outputs")
    ev.add_argument("--references", type=Path, required=True)
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--output-path", type=Path, required=True)

    summary = commands.add_parser("summarize", help="Aggregate compatible metric files")
    summary.add_argument("--metrics", type=Path, nargs="+", required=True)
    summary.add_argument("--output-path", type=Path, required=True)
    return p


def run(args):
    if args.command == "prepare":
        from .data import prepare_data

        return prepare_data(args.input, args.output_dir, args.seed, args.split_by)
    if args.command == "baseline":
        from .baselines import run_baseline

        return run_baseline(args.data_dir, args.output_dir, args.method, args.seed)
    if args.command == "train":
        from .training import train_camembert, train_sft

        values = vars(args).copy()
        backend = values.pop("backend")
        values.pop("command")
        if backend == "camembert":
            if values["quantize_4bit"]:
                raise ValueError("--quantize-4bit applies only to SFT")
            values.pop("quantize_4bit")
            values.pop("lora_rank")
        # Backend defaults are the versioned recipe; explicit flags override them.
        values = {k: v for k, v in values.items() if v is not None}
        return (train_camembert if backend == "camembert" else train_sft)(**values)
    if args.command == "predict":
        from .training import predict_model

        values = {k: v for k, v in vars(args).items() if k != "command" and v is not None}
        return predict_model(**values)
    if args.command == "evaluate":
        from .metrics import evaluate_files

        return evaluate_files(args.references, args.predictions, args.output_path)
    if args.command == "summarize":
        from .reporting import summarize

        return summarize(args.metrics, args.output_path)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
