# Local data

The coursework data is not included. Place your authorized copies in `data/raw/`:

```text
data/raw/ftdataset_train.tsv
data/raw/ftdataset_val.tsv
```

Required UTF-8 TSV columns: `Avis`, `Prix`, `Cuisine`, `Service`.
Allowed labels: `Positive`, `Négative`, `Neutre`, `NE`. `Neutre` means mixed
positive/negative opinions. Named extra columns are permitted; `Restaurant`
enables restaurant-group splits. `Note` and `URL` are not model inputs.

`sentiment prepare` pools these sources, validates annotations, removes duplicates
and conflicts, and creates a new frozen train/validation/test protocol. This does
not recreate the course split or provide a newly collected external test.
Normalization checks NFKC, case and whitespace, not semantic near-duplicates.
Inspect per-class support: splits are not stratified.

Processed JSONL records contain ID, text, three labels and optional restaurant.
Manifests contain source hashes, aggregate counts and overlap checks. A changed
seed, source or grouping requires a new output directory.

All source and processed data are ignored by Git. Redistribution requires
appropriate permission; raw predictions can also contain review text.
