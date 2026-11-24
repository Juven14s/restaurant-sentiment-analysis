# French aspect sentiment: from encoders to small LLMs

[Open in Google Colab](https://colab.research.google.com/github/Juven14s/restaurant-sentiment-analysis/blob/main/notebooks/colab_training.ipynb)

A reproducible study of **data quality, supervised adaptation and evaluation**
for French restaurant reviews. Predict `Positive`, `Négative`, `Neutre` or `NE`
for each of **Prix, Cuisine, Service**. Here `Neutre` means **mixed positive and
negative opinions**, and `NE` means the aspect is not expressed.

Research question: **how do a linear baseline, a fine-tuned French encoder and a
small generative model compare in quality and compute cost, and what does LoRA
adaptation change?** Full CamemBERT and LoRA adaptation scores remain to be measured.
The old coursework's reported 85% accuracy is not a result of this protocol.
See [experiment status](docs/experiments.md).

## CPU-only diagnostic study

For a path without neural training, see the [study guide](docs/light_study.md)
and [English research report](docs/light_research_report.md). It includes a
50-case validation error review, an original 80-case challenge, and bounded
Qwen inference on CPU with an explicit strict-versus-Markdown-format ablation.
It also audits 12 train-only paraphrases without downstream training.
Assistant-written annotations are not an independently validated human gold set.

```bash
python -m pip install -e '.[baseline]'
python scripts/run_light_study.py --data-dir data/processed-v1
```

## Start with Colab

No personal GPU is required. The notebook starts with a **32-example training
smoke test**, retains the full validation set and skips test evaluation.
After checking the run, set `SMOKE_TEST = False` for full training. Test scoring
remains disabled until you freeze the choices and set `RUN_TEST_EVALUATION = True`.
Colab GPU availability and session duration vary.

1. Open the notebook and select a GPU runtime.
2. Mount your Google Drive when prompted.
3. Upload the two original TSV files when prompted. They stay in your Drive.
4. Run the cells in order. Checkpoints, frozen splits and results persist in Drive.

The first run records the Git commit and Hugging Face model revisions; later
sessions reuse them. Training can resume from a complete Trainer checkpoint
with identical configuration and data. See the [French guide](docs/colab.md).

## Local installation

Python 3.12 is recommended. Data preparation, metrics and the majority baseline
use the standard library. Install extra dependencies only for the requested work.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m restaurant_sentiment --help
```

For TF-IDF: `python -m pip install -e '.[baseline]'`.
For training: `python -m pip install -r requirements-colab.txt`.
For optional 4-bit CUDA adaptation: `python -m pip install -e '.[quantization]'`.
On Windows, activate with `.venv\Scripts\activate`.

## Freeze the data protocol

The original data is not redistributed. Obtain the coursework TSVs and follow
[data/README.md](data/README.md). Pool the former train/validation sources and
create a **new**, deterministic 80/10/10 split:

```bash
sentiment prepare \
  --input data/raw/ftdataset_train.tsv data/raw/ftdataset_val.tsv \
  --output-dir data/processed-v1 --seed 42
```

Preparation validates annotations, normalizes text for duplicate detection,
removes concordant duplicates and excludes all copies of conflicting annotations.
It saves source hashes, per-class counts and overlap checks. Frozen splits cannot
be overwritten with changed data or settings. Near-duplicates are not detected;
splits are not stratified, so inspect per-class support.

The default split permits restaurants in multiple splits. For a separate
unseen-restaurant experiment, use `--split-by restaurant` with a **different
output directory**; its scores answer a different generalization question.
Repartitioning previously studied coursework does not provide a pristine external
test. Prior scores are not comparable. Select models on validation before
reporting final test scores.

## Run the benchmark

```bash
sentiment baseline --data-dir data/processed-v1 --method majority --output-dir outputs/majority
sentiment baseline --data-dir data/processed-v1 --method tfidf --output-dir outputs/tfidf

sentiment train --backend camembert --data-dir data/processed-v1 \
  --output-dir outputs/camembert-42 --seed 42 --device cuda

sentiment train --backend sft --data-dir data/processed-v1 \
  --output-dir outputs/lora-42 --seed 42 --device cuda
```

CamemBERT uses twelve logits reshaped into three four-class tasks. This Trainer
implementation is not an exact reproduction of the coursework head. LoRA uses
`Qwen/Qwen3-0.6B`, completion-only supervised loss, rank 8 and a shared versioned
French prompt with thinking disabled. Only review text is truncated; task
instructions and the full JSON target are retained. `--quantize-4bit` selects
optional QLoRA on compatible CUDA hardware.

Starting recipes: CamemBERT 3 epochs, batch 8, length 256, LR 3e-5; LoRA 1 epoch,
batch 1, accumulation 8, length 512, LR 2e-4. Both select the best checkpoint by
**validation loss**. These are starting settings, not claimed optimal values.
See `sentiment train --help` for overrides and checkpoint resumption.

Training reads **train and validation only**, saves resumable checkpoints and
exports the selected model/tokenizer to `OUTPUT_DIR/final`. Inference is separate:

```bash
sentiment predict --backend camembert --model-path outputs/camembert-42/final \
  --data-path data/processed-v1/test.jsonl --output-path outputs/camembert-42/predictions.jsonl \
  --max-length 256 --batch-size 8

sentiment predict --backend llm --model-path Qwen/Qwen3-0.6B \
  --data-path data/processed-v1/test.jsonl --output-path outputs/qwen-prompt/predictions.jsonl

sentiment predict --backend lora --model-path outputs/lora-42/final \
  --data-path data/processed-v1/test.jsonl --output-path outputs/lora-42/predictions.jsonl

sentiment evaluate --references data/processed-v1/test.jsonl \
  --predictions outputs/lora-42/predictions.jsonl --output-path outputs/lora-42/metrics.json
```

For a controlled Qwen comparison, pass `--model-revision` matching LoRA's base
model commit. Colab does this automatically. Both methods share task prompts,
deterministic decoding and nominal token limits. SFT reserves target-token space,
so effective review lengths can differ; see [the protocol](docs/protocol.md).

## Evaluation and provenance

- Accuracy and macro-F1 by aspect, plus precision, recall, F1 and support per class.
- Mean macro-F1 across aspects, joint exact match and structured-output validity.
- Invalid JSON/schema predictions are errors for **all** aspects, kept in the denominator.
- Strict ID alignment rejects missing, extra or duplicate predictions.
- Manifests record seeds, settings, data hashes, model/code revisions, dependency
  versions, hardware and timing. Inference records latency and peak allocated GPU
  memory; comparisons must account for batch size, precision and hardware.

Metrics use fractions from 0 to 1. Compare compatible metric files with:

```bash
sentiment summarize --metrics outputs/majority/metrics.json outputs/tfidf/metrics.json \
  outputs/lora-42/metrics.json --output-path outputs/comparison.json
```

For a later multi-run study use training seeds 42, 43 and 44 while keeping the
**split seed fixed**. Standard deviation is reported only for multiple seeds.
Summaries reject different frozen tests.

## Tests and layout

```bash
python -m unittest discover -s tests -v
```

Core tests need no GPU or downloads. With training dependencies installed, tiny
random local models are trained and reloaded on CPU. This checks integration,
not model quality or CUDA/4-bit compatibility.

```text
src/restaurant_sentiment/  Data, baselines, prompts, training, scoring and CLI
notebooks/                Colab workflow
tests/                    Protocol and offline integration tests
docs/                     Experimental protocol, status and Colab guide
legacy/                   Archived coursework outside the supported workflow
data/                     Local sources and frozen splits (ignored by Git)
outputs/                  Local checkpoints and predictions (ignored by Git)
```

The archived paraphrase module is unvalidated. Downstream augmentation studies,
DPO and cluster runs are future work, not demonstrated capabilities.

Author: Rassem DJIMADOUM.
