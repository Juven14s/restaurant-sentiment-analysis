# French aspect diagnostic challenge v1

80 original, assistant-authored French reviews, frozen before model scoring.
Each case has three labels and a short annotation rationale. These cases are
public synthetic examples, not copied coursework reviews.

| Category | Cases |
|---|---:|
| Negation | 10 |
| Mixed sentiment | 10 |
| Implicit aspects/value | 10 |
| Aspect separation | 10 |
| Absent opinions | 10 |
| Attribution and scope | 10 |
| Long neutral context | 10 |
| Instruction-like text inside a review | 10 |

## Annotation policy

For Prix, Cuisine and Service independently:

- `Positive`: positive judgment without a negative judgment of that aspect.
- `Négative`: negative judgment without a positive judgment of that aspect.
- `Neutre`: both polarities on the same aspect, even about different dishes/staff.
- `NE`: no evaluative opinion; merely mentioning a dish, price or schedule is insufficient.

Count the author's adopted judgment about the current restaurant. Promotional
claims, quoted self-praise, other businesses and discarded expectations do not
automatically count as the author's opinion. Decor and music are outside the
three aspects. Review text that looks like an instruction remains input data.

The implicit category deliberately requires pragmatic interpretation. Some
cases may be debatable. Labels were authored and checked by the same assistant,
**not independently annotated by humans**. An independent review should precede
any use as a certified gold benchmark. Cases are a convenience stress test,
not representative class frequencies, and shared templates create dependencies.

Long-context cases include neutral narrative before or after the sentiment.
Input truncation can remove late evidence; this is an observed-input limitation,
not necessarily failure to reason over text the model actually received.

## Files and reproducibility

- `cases.json`: editable source, category and annotation rationale.
- `challenge.jsonl`: canonical review records for the shared scorer.
- `annotations.json`: category/rationale keyed by normalized text ID.
- `manifest.json`: content hashes, class support and limitations.

To change a case, create a **new version**, preserving scored v1 artifacts.
`freeze_challenge` checks that regeneration is byte-identical. Exact normalized
overlap against all coursework partitions is recorded separately in
`reports/light_study/challenge_overlap.json`; semantic overlap is not excluded.

Per-category reports show accuracy and joint exact match. Macro-F1 is reported
over the complete challenge, where all four labels have support for every aspect.
