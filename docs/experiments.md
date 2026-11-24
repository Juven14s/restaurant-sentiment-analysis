# Experiment status

The CPU baselines below were measured locally on 24 November 2026 with Python
3.12 and scikit-learn 1.7.2 for TF-IDF. Full CamemBERT/LoRA training runs and
CUDA/4-bit compatibility have **not** been measured. Tiny random-model smoke
tests validate software integration only; they are not research results.

A subsequent [CPU-only diagnostic study](light_research_report.md) evaluates
the frozen small Qwen model without training, reviews validation errors and
scores an original targeted challenge. Its validation/challenge scores are
separate from the historical frozen-test results below.

## Data audit

The 4,609 original rows yield 4,602 normalized unique texts. Three concordant
duplicate rows are removed; four conflicting texts (eight rows) are entirely
excluded. The 4,598 retained reviews are split into 3,678 train, 460 validation
and 460 test rows with split seed 42. No normalized text IDs overlap. Restaurants
can overlap in this text-grouped protocol. No stratification or semantic
near-duplicate detection is performed.

This is a new split of previously studied coursework data, not an external
unseen collection. It is not comparable to the historical approximately 85%
accuracy in the archived report.

[Audit and source hashes](../reports/data_audit.json)

## Fixed CPU baselines, seed 42

| Method | Mean aspect accuracy | Mean aspect macro-F1 | Joint exact match |
|---|---:|---:|---:|
| majority | 65.80% | 0.1984 | 36.96% |
| tfidf | 76.23% | 0.4531 | 49.57% |

All outputs have a valid label schema. Per-class support, confusion matrices,
settings, package versions, fit/inference timings and implementation hashes are
in [majority](../reports/majority.json) and [TF-IDF](../reports/tfidf.json).
The [comparison JSON](../reports/cpu_comparison.json) uses scores from 0 to 1.
No hyperparameter search was performed against this test. One run provides no
seed-variance estimate; the two baseline methods are not evidence about LoRA.

## Ready to run in Colab

- CamemBERT training with validation checkpoint selection.
- The pinned Qwen3-0.6B base checkpoint prompted without adaptation.
- Completion-only SFT with LoRA; optional QLoRA on compatible CUDA hardware.
- Shared scoring, timing, memory and provenance, plus checkpoint resumption.

The notebook defaults to a pilot that never evaluates test. Full training and
final test evaluation are separate switches. Real model metrics must be added
only after their runs finish. Subsequent choices must use validation, not these
fixed test scores.

## Future work

Controlled filtered synthetic-data ablations, justified DPO preferences,
external test data, paper reproduction and actual Slurm/multi-GPU runs are not
implemented or claimed by this release.
