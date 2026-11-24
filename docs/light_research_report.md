# CPU-only research note: data quality and structured sentiment evaluation

**Rassem DJIMADOUM — 24 November 2026.** This study asks what can be learned
about French aspect sentiment without further neural training. It combines a
validation error review, an original diagnostic challenge, frozen-model Qwen
inference and a small train-only paraphrase audit. No neural parameters were
updated. A linear baseline was fitted on train only.

## Protocol and methods

The existing data protocol retains 4,598 reviews after duplicate/conflict removal:
3,678 train, 460 validation and 460 test. This study uses validation for development;
it does not rerun the coursework test to choose prompts or settings. Previous
coursework exposure means this corpus is not a pristine external evaluation.

Methods are a train-derived majority baseline, fixed TF-IDF/logistic regression,
and Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with unchanged weights and the existing
French task prompt. Inference uses CPU FP32, four PyTorch threads, batch one,
greedy decoding, thinking disabled, 512 input tokens and up to 96 new tokens.
The execution host reports an Intel i7-11370H. Torch/Transformers versions and
runtime hashes accompany the reports. Timings are single runs, not controlled
hardware benchmarks.

For each of Prix, Cuisine and Service, `Neutre` means **both positive and negative
opinions**; `NE` means no opinion. Scores include mean per-aspect accuracy and
four-class macro-F1, joint exact match and schema validity. Invalid predictions
remain errors for every aspect; no rows are dropped. Predictions align by ID.

## Validation errors and annotation questions

TF-IDF reaches **75.94% mean accuracy, 0.4449 macro-F1 and 47.17% joint exact
match** on all 460 validation reviews. The gap between accuracy and macro-F1
matters: mixed-opinion F1 is 0.00 for Prix, 0.05 for Cuisine and 0.00 for Service,
with supports 19, 39 and 22. There are 243 reviews with at least one wrong aspect.

A deterministic, error-pattern-balanced sample of **50 failed reviews** was read
and coded by the assistant. Categories overlap: 20 cases involve mixed polarity,
16 aspect scope, 12 absent-aspect predictions and six price expressions. Common
patterns are collapsing praise and complaints into one polarity, spreading food
praise to unmentioned service, and missing value judgments.

The review flagged **23 cases for possible annotation clarification**, including
explicit opinions marked absent and mixed evidence labelled with a single
polarity. These are hypotheses from one assistant, not corrected gold labels.
Original annotations and scores remain unchanged. The selected error sample is
not representative, so 23/50 is **not** an estimate of corpus annotation error.
Private review text is retained locally; only aggregate diagnostics and coded
notes are public. Independent human adjudication is still needed.

## Frozen-model probe: 40 validation reviews

Examples were selected by seeded ID hashing, without label-based selection.
Qwen needed 231.4 seconds for inference; two inputs were truncated. The model
wrapped every response in a Markdown code fence, failing the original JSON-only
contract. A separate, post-hoc validation analysis removes one complete JSON
fence, without changing keys or labels. It recovers 38/40 schemas; two responses
still use unsupported label values. The strict parser is unchanged.

| Method | Mean accuracy | Macro-F1 | Joint exact match | Schema validity |
|---|---:|---:|---:|---:|
| Majority | 65.00% | 0.1968 | 32.50% | 100.0% |
| TF-IDF | 72.50% | 0.4032 | 45.00% | 100.0% |
| Qwen, strict contract | 0.00% | 0.0000 | 0.00% | 0.0% |
| Qwen, single-fence analysis | 47.50% | 0.2689 | 15.00% | 95.0% |

Even with explicit formatting normalization, this small Qwen model does not
outperform TF-IDF on the natural-review probe. This is a bounded diagnostic
result, not a population ranking or a test-set improvement claim.

## Original targeted challenge: 80 cases

Ten cases cover each of eight difficulties. Every case has labels and a short
rationale. The challenge was assistant-authored and frozen before model scoring;
there is no independent human gold validation. Exact normalized text overlap
with every coursework split is zero, but semantic overlap is not excluded.
Qwen generation settings and prompt were unchanged. Inference took 417.3 seconds,
with no truncated inputs. The single-fence rule was chosen on validation and
then applied unchanged to these diagnostic outputs.

| Method | Mean accuracy | Macro-F1 | Joint exact match | Schema validity |
|---|---:|---:|---:|---:|
| Majority | 34.17% | 0.1168 | 0.00% | 100.0% |
| TF-IDF | 46.25% | 0.2775 | 6.25% | 100.0% |
| Qwen, strict contract | 0.00% | 0.0000 | 0.00% | 0.0% |
| Qwen, single-fence analysis | 62.92% | 0.4129 | 26.25% | 95.0% |

After the declared format normalization, Qwen exceeds TF-IDF here, reversing the
probe ranking. The short controlled language, domain shift and explicit task
policy are plausible explanations, not established causes. The challenge has
convenience sampling and shared templates, so these observations do not support
statistical significance or broad French-language capability claims.

| Category, 10 cases each | TF-IDF mean accuracy | Qwen single-fence mean accuracy |
|---|---:|---:|
| absent opinions | 46.7% | 96.7% |
| aspect separation | 50.0% | 40.0% |
| attribution and scope | 53.3% | 76.7% |
| implicit aspects | 43.3% | 53.3% |
| instruction like text | 43.3% | 46.7% |
| long context | 33.3% | 63.3% |
| mixed sentiment | 40.0% | 50.0% |
| negation | 60.0% | 76.7% |

Four challenge generations remain invalid after normalization. They include
unsupported label variants and an embedded instruction copied into a label.
The public synthetic inputs and raw predictions allow these failures to be
inspected without distributing coursework reviews. Per-category macro-F1 is
omitted because several slices have absent classes.

## Train-only synthetic-data quality audit

Qwen generated 12 paraphrases, three per Cuisine reference class, from short
training reviews. Validation/test were not generation sources. In a single-pass
assistant review, semantic preservation passed for five, failed for four and
was uncertain for three. Failures included a price-polarity flip, removal of an
explicit cost complaint and factual drift from noisy spelling. One output was
an exact normalized copy; none exactly overlapped validation/test.

Seven source annotations also needed clarification. Under the declared review
rule—semantic pass, no flagged source annotation, no exact copy/held-out overlap
or output truncation—**two pairs remain candidates**. This deliberate tiny audit
does not estimate a production acceptance rate. Candidates were not used to
train anything, and no downstream gain is claimed. Raw source/paraphrase pairs
remain private. The aggregate audit and coding notes are public.

## Interpretation and reproducibility

The main evidence is practical: data checks, ID-safe scoring, rare-class failure
analysis, an observable JSON-contract failure, and a small synthetic-data audit.
A larger training run would not by itself resolve ambiguous label policy or
unsafe automatic label copying. The next steps are independent annotation
review, validation-only experiments on rare-class handling, and output-schema
controls tested with a new untouched evaluation collection.

The archived coursework describes CamemBERT fine-tuning, but its old approximately
85% accuracy is not comparable to this protocol. No new CamemBERT/SFT/DPO results,
cluster experience or multi-GPU execution are claimed here. Tiny random-model
tests verify integration and checkpoint resumption, not pretrained-model quality.

Reproduce the CPU baselines and diagnostic artifacts with:

```bash
python -m pip install -e '.[baseline]'
python scripts/run_light_study.py --data-dir data/processed-v1
```

Add `--run-llm` after installing `.[training]` for the frozen model inference.
See [the French guide](light_study.md),
[challenge policy](../benchmarks/fr_aspect_challenge_v1/README.md),
[validation review](../reports/light_study/validation_review.json),
[inference provenance](../reports/light_study/qwen_inference_provenance.json)
and [paraphrase audit](../reports/light_study/paraphrase_quality_audit.json).

Implementation references: [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B)
and [Transformers 4.56.2 generation](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/text_generation).
