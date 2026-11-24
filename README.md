# French aspect sentiment: data quality and LLM evaluation

A reproducible **CPU-based NLP/LLM study** of French restaurant reviews:
clean the data, compare TF-IDF with a frozen small language model, and diagnose
prediction and structured-output failures. **No GPU is needed for this study.**
Qwen is used for inference with unchanged pretrained weights; only the linear
baseline is fitted on the training data.

Predict `Positive`, `Négative`, `Neutre` or `NE` for each of **Prix, Cuisine,
Service**. Here `Neutre` means **mixed positive and negative opinions**, and `NE`
means no opinion is expressed about the aspect.

Read the [English research report](docs/light_research_report.md),
[French reproduction guide](docs/light_study.md) and
[experiment status](docs/experiments.md).

## What was completed

- Validated and deduplicated 4,609 source rows into **4,598 reviews**, with frozen
  train/validation/test splits of **3,678 / 460 / 460** and no normalized text overlap.
- Compared majority and TF-IDF baselines with **Qwen3-0.6B inference on CPU**,
  using a frozen 40-review validation probe and an original 80-case challenge.
- Reviewed **50 validation errors** for mixed opinions, aspect scope, absent
  opinions and possible annotation ambiguities.
- Audited **12 train-only paraphrases** for semantic preservation; none were
  used for downstream training.
- Added strict ID-aligned scoring, provenance manifests, **44 automated tests**
  and GitHub Actions checks.

Qualitative reviews and challenge annotations were produced by the assistant;
they still need independent human validation. The original coursework reviews
and generated paraphrase pairs are not redistributed.

## Measured results

On all **460 validation reviews**, TF-IDF reaches **75.94% mean per-aspect
accuracy** and **0.4449 macro-F1**. Mixed-opinion F1 is only 0.00 for Prix, 0.05
for Cuisine and 0.00 for Service: overall accuracy hides rare-class failures.

The matched comparisons below use the same examples for each method. Accuracy
and macro-F1 are averaged across the three aspects.

| Evaluation set | Method | Mean accuracy | Macro-F1 | Schema validity |
|---|---|---:|---:|---:|
| Validation probe, 40 reviews | TF-IDF | 72.50% | 0.4032 | 100% |
| Validation probe, 40 reviews | Qwen, strict JSON | 0.00% | 0.0000 | 0% |
| Validation probe, 40 reviews | Qwen, single-fence analysis | 47.50% | 0.2689 | 95% |
| Diagnostic challenge, 80 cases | TF-IDF | 46.25% | 0.2775 | 100% |
| Diagnostic challenge, 80 cases | Qwen, strict JSON | 0.00% | 0.0000 | 0% |
| Diagnostic challenge, 80 cases | Qwen, single-fence analysis | 62.92% | 0.4129 | 95% |

Qwen wrapped every answer in Markdown fences, violating the JSON-only contract.
A **separate format analysis** removes one complete JSON code fence from the
same generations, without repairing labels or dropping invalid examples. This
rule was chosen after inspecting validation outputs; the production parser
remains strict. Even after normalization, Qwen trails TF-IDF on the real-review
probe. Its higher challenge score does not establish a general advantage.

The challenge covers eight difficulties, including negation, mixed sentiment,
aspect separation and instruction-like text. It is an assistant-authored
diagnostic set, not an independently validated external benchmark. See the
[challenge and annotation policy](benchmarks/fr_aspect_challenge_v1/README.md),
[public result artifacts](reports/light_study/) and
[full report](docs/light_research_report.md) for category scores and limitations.

## Reproduce the CPU study

Python 3.12 is recommended. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baseline]'
```

On Windows, activate with `.venv\Scripts\activate`.

Obtain the two original coursework TSVs and place them in `data/raw/`, following
[data/README.md](data/README.md). The private review corpus is required to refit
the baselines and reproduce the validation analyses.

```bash
sentiment prepare \
  --input data/raw/ftdataset_train.tsv data/raw/ftdataset_val.tsv \
  --output-dir data/processed-v1 --seed 42 --split-by text

python scripts/run_light_study.py --data-dir data/processed-v1
```

This runs the CPU baselines, freezes the 40-review probe, creates a private
50-error review worksheet and scores the public 80-case challenge. Outputs go
to `outputs/light-study-v1/`. Qualitative coding requires a separate review;
the script does not reproduce the published judgments automatically.

To also run **frozen Qwen inference**, install the ML dependencies. On
Linux/Windows, install the CPU-only PyTorch build first:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[training]'
python scripts/run_light_study.py --data-dir data/processed-v1 --run-llm
```

The first run downloads the public model. The recipe pins the model revision,
prompt and decoding settings and uses four CPU threads by default. On an Intel
i7-11370H, measured inference took **231 seconds for the 40-review probe** and
**417 seconds for the 80-case challenge**, excluding setup and model download.
These are single-run timings, not a hardware guarantee. No neural training is
called by this script. See the [French guide](docs/light_study.md) for the
separate paraphrase audit and targeted analysis commands.

## Data and evaluation safeguards

Preparation validates labels, normalizes text for duplicate detection, removes
agreeing duplicates and excludes every copy of texts with conflicting labels.
Frozen artifacts record source hashes, class counts and overlap checks and
reject changes to existing splits.

The text split permits restaurants in multiple partitions. Splits are not
stratified, and semantic near-duplicates are not detected. An unseen-restaurant
experiment is available through `--split-by restaurant` in a new output directory.
Previously studied coursework is not a pristine external test collection.
The CPU diagnostic study uses validation for development and does not rerun the
coursework test to choose prompts or settings.

Scoring reports per-aspect accuracy, four-class macro-F1, per-class support,
confusion matrices, joint exact match and schema validity. Missing, extra or
duplicate prediction IDs are rejected. Invalid schema predictions count as
errors for all three aspects and remain in the denominator. Manifests record
data hashes, seeds, model/code revisions, settings, versions, hardware and timing.
See the [experimental protocol](docs/protocol.md).

## Optional future training: CamemBERT and LoRA

The repository also includes CamemBERT training and Qwen LoRA/optional QLoRA
backends, with best-validation checkpoints and resumable training. These are
**available workflows, not completed full pretrained-model experiments** in
this study. No new CamemBERT or LoRA quality score is claimed. The archived
coursework's approximately 85% accuracy is not comparable to this protocol.

For a future GPU run, use the
[Colab notebook](https://colab.research.google.com/github/Juven14s/restaurant-sentiment-analysis/blob/main/notebooks/colab_training.ipynb)
and [French Colab guide](docs/colab.md). It starts with a 32-example smoke test,
persists checkpoints in Drive and keeps final test evaluation disabled until
explicitly enabled. Colab is optional; the CPU study above is independent of it.

## Tests and repository layout

```bash
python -m unittest discover -s tests -v
```

The suite contains 44 tests. Core checks need no GPU or model download; optional
ML checks are skipped when their dependencies are absent. Tiny random local
models exercise actual CPU training, checkpoint reload and LoRA resumption.
They verify integration, not pretrained-model quality or CUDA compatibility.

```text
src/restaurant_sentiment/  Data, baselines, prompts, analysis, training and CLI
scripts/                  CPU study and train-only paraphrase audit
benchmarks/               Public original challenge, labels and rationales
reports/                  Published scores and aggregate diagnostics
tests/                    Protocol and offline integration checks
docs/                     Research report, protocol and reproduction guides
notebooks/                Optional Colab training workflow
legacy/                   Archived coursework outside the supported workflow
data/                     Local private sources and frozen splits (ignored by Git)
outputs/                  Local predictions and checkpoints (ignored by Git)
```

Downstream augmentation gains, DPO and cluster/multi-GPU execution remain future
work. Author: Rassem DJIMADOUM.
