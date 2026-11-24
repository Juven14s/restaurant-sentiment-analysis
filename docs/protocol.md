# Experimental protocol

Question: how does a small instruction model adapted with LoRA compare to a
French encoder and a linear baseline in sentiment quality, compute cost and
structured-output reliability?

## Data

Pool former coursework splits. Reject malformed annotations. Normalize with
NFKC/casefold/whitespace, retain a deterministic copy of agreeing duplicates and
remove all copies of conflicting ones. Do not arbitrate conflicts with the
model under evaluation.

Freeze an 80/10/10 split with seed 42. Publish aggregate counts and hashes only.
Default text grouping permits restaurant overlap. Restaurant grouping is a
separate domain-shift experiment. Hashes prove artifact identity, not annotation
correctness or absence of semantic near-duplicates. There is no stratification;
report missing and low-support classes. This is previously studied coursework,
not a pristine external test; previous tuning can introduce optimism.

## Comparisons

1. Majority labels learned independently per aspect from train.
2. TF-IDF plus per-aspect logistic regression with a fixed starting configuration.
3. CamemBERT with three four-class outputs.
4. Qwen3-0.6B prompted with the task definition, no parameter updates.
5. The identical Qwen checkpoint adapted with LoRA on supervised JSON completions.

Pin model revisions; use identical Qwen prompts and decoding limits. Only review
tokens may be truncated. SFT also reserves the complete target in its sequence
budget, so effective review lengths can differ from prompting. Record truncation
and interpret long-review differences accordingly.

Select checkpoints by validation loss. Debug and tune on validation only. Do not
choose ranks, prompts or seeds using test scores. Freeze final choices before
test evaluation. Keep split seed 42 while varying training seeds 42, 43 and 44.
Run one seed first to establish feasibility and cost.

## Reporting

Primary metric: mean per-aspect macro-F1 over four fixed labels. Also report
per-class support/F1, aspect accuracy, joint exact match, structured JSON validity
and confusion matrices. Invalid outputs are errors for every aspect. Strict ID
alignment prevents dropped or reordered examples.

Record hardware, precision, seed, data hashes, code/model commits, selection
rule, truncation, duration, inference batch size, latency and peak allocated
GPU memory. Separate loading from inference time. CPU baselines and GPU runs
are not equivalent hardware comparisons. One seed has no estimated seed variance.

Inspect a fixed validation sample for negation, mixed opinions, implicit aspects,
rare classes and long reviews. Keep analysis notes separate from the final test.
Do not publish review text without redistribution permission.

## Follow-up work

After the core benchmark, choose one bounded study: LoRA rank or augmentation
quality. For augmentation, generate only from train, retain source IDs, reject
copied/evaluation-overlapping texts, manually verify label preservation and
compare original, unfiltered and filtered data. A bounded train-only generation
and quality audit is now documented in the CPU-only report; downstream training
comparisons with original/unfiltered/filtered data are not implemented.

DPO needs a justified preference dataset and SFT comparison. Slurm/multi-GPU
experience must come from actual runs; Colab does not demonstrate those skills.

## References

- [TRL SFTTrainer 0.23.1](https://huggingface.co/docs/trl/v0.23.1/en/sft_trainer)
- [Transformers Trainer 4.56.2](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer)
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Colab resource limits](https://research.google.com/colaboratory/faq.html)
