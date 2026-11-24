"""Reproducible validation diagnostics; surface cues are not causal explanations."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from .data import ASPECTS, LABELS, load_jsonl
from .metrics import evaluate_files, evaluate_predictions
from .prompts import parse_prediction
from .reporting import experiment_signature


def single_fence_prediction(raw):
    """Explicit analysis-only normalization; production parsing stays strict."""
    strict = parse_prediction(raw)
    if strict is not None:
        return strict
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", raw, re.S)
    return parse_prediction(match.group(1)) if match else None


def format_ablation(reference_path, predictions_path, output_dir):
    """Rescore identical generations with one declared formatting rule."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strict = evaluate_files(reference_path, predictions_path, output_dir/'strict.json')
    rows = [json.loads(line) for line in Path(predictions_path).read_text(encoding='utf-8').splitlines()]
    normalized = []
    recovered = 0
    for row in rows:
        if 'raw_output' not in row:
            raise ValueError('Format ablation requires raw model outputs')
        if row.get('prediction') != parse_prediction(row['raw_output']):
            raise ValueError('Stored strict prediction does not match its raw output')
        parsed = single_fence_prediction(row['raw_output'])
        recovered += int(row.get('prediction') is None and parsed is not None)
        normalized.append(dict(id=row['id'],prediction=parsed))
    normalized_path = output_dir/'predictions_single_fence.jsonl'
    normalized_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in normalized),encoding='utf-8')
    scores = evaluate_files(reference_path, normalized_path, output_dir/'single_fence.json')
    scores.update(method='llm_single_fence',seed=strict.get('seed'),
        experiment_signature=experiment_signature({'base_signature':strict.get('experiment_signature'),
            'normalization':'unwrap_one_complete_json_markdown_fence_v1'}))
    (output_dir/'single_fence.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report = dict(n_samples=len(rows), recovered_by_single_fence=recovered,
        reference_sha256=strict['reference_sha256'],data_fingerprint=strict['data_fingerprint'],
        raw_predictions_sha256=hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
        normalization='unwrap_one_complete_json_markdown_fence_v1; no label/key repair',
        strict={k:strict[k] for k in ('macro_f1','macro_accuracy','joint_exact_match','json_valid_rate')},
        single_fence={k:scores[k] for k in ('macro_f1','macro_accuracy','joint_exact_match','json_valid_rate')},
        limitation='Normalization was introduced after inspecting validation outputs; strict scores remain primary for the original API contract.')
    (output_dir/'format_ablation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def surface_cues(text):
    cues = []
    if re.search(r"\b(?:pas|jamais|aucun[es]*|sans|ni)\b", text, re.I):
        cues.append("negation_word")
    if re.search(r"\b(?:mais|pourtant|cependant|toutefois|en revanche|alors que)\b", text, re.I):
        cues.append("contrast_word")
    if len(text.split()) >= 100:
        cues.append("long_100_words")
    if len(text.split()) <= 15:
        cues.append("short_15_words")
    return cues


def diagnose(reference_path: Path, predictions_path: Path, output_dir: Path,
             sample_size: int = 50, seed: int = 42) -> dict:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if Path(reference_path).stem != "validation":
        raise ValueError("Error-review sampling is restricted to validation.jsonl, not the final test")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worksheet = output_dir / "error_review.csv"
    if worksheet.exists():
        with worksheet.open(encoding='utf-8') as stream:
            if any(r.get('review_category') or r.get('notes') for r in csv.DictReader(stream)):
                raise FileExistsError('Reviewed worksheet exists; choose a new diagnostic output directory')
    metrics = evaluate_files(reference_path, predictions_path, output_dir / "metrics.json")
    references = load_jsonl(reference_path)
    predictions = [json.loads(line) for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in predictions}
    error_groups = defaultdict(list)
    error_histogram = Counter()
    cue_rows = defaultdict(list)
    missed = {a: Counter() for a in ASPECTS}
    all_errors = []
    for row in references:
        prediction = by_id[row["id"]].get("prediction")
        valid = isinstance(prediction, dict) and set(prediction) == set(ASPECTS) and all(prediction[a] in LABELS for a in ASPECTS)
        wrong = tuple(a for a in ASPECTS if not valid or row["labels"][a] != prediction[a])
        error_histogram[len(wrong)] += 1
        for a in wrong:
            missed[a][row["labels"][a]] += 1
        for cue in surface_cues(row["text"]):
            cue_rows[cue].append(row)
        if wrong:
            case = {"row": row, "prediction": prediction, "wrong_aspects": wrong}
            error_groups[wrong].append(case)
            all_errors.append(case)
    # Round-robin across aspect-error patterns, then seeded ID ranking within
    # each pattern. This is diagnostic sampling, not prevalence estimation.
    queues = []
    for pattern in sorted(error_groups):
        queues.append(sorted(error_groups[pattern], key=lambda c: hashlib.sha256(
            f"{seed}:{c['row']['id']}".encode()).hexdigest()))
    selected = []
    while len(selected) < min(sample_size, len(all_errors)):
        for queue in queues:
            if queue and len(selected) < sample_size:
                selected.append(queue.pop(0))
    with worksheet.open("w", encoding="utf-8", newline="") as stream:
        fields = ["id", "text", "reference", "prediction", "wrong_aspects", "surface_cues", "review_category", "annotation_check", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in selected:
            row = c["row"]
            writer.writerow(dict(id=row["id"], text=row["text"],
                reference=json.dumps(row["labels"], ensure_ascii=False),
                prediction=json.dumps(c["prediction"], ensure_ascii=False),
                wrong_aspects=";".join(c["wrong_aspects"]), surface_cues=";".join(surface_cues(row["text"])),
                review_category="", annotation_check="", notes=""))
    slices = {}
    for cue, rows in sorted(cue_rows.items()):
        scores = evaluate_predictions(rows, [by_id[row["id"]] for row in rows])
        slices[cue] = {k: scores[k] for k in ("n_samples", "macro_accuracy", "macro_f1", "joint_exact_match")}
    report = {
        "reference_sha256": metrics["reference_sha256"], "data_fingerprint": metrics["data_fingerprint"],
        "n_samples": len(references), "error_reviews": len(all_errors),
        "wrong_aspect_histogram": {str(i): error_histogram[i] for i in range(4)},
        "missed_reference_labels": {a: {label: missed[a][label] for label in LABELS} for a in ASPECTS},
        "surface_slices": slices, "review_sample_size": len(selected), "review_sample_seed": seed,
        "sampling": "round_robin_aspect_error_patterns_then_sha256_seeded_id_rank",
        "limitations": ["Surface cues are lexical indicators, not semantic labels or causal explanations.",
                         "Error-pattern-balanced review sample is not a random prevalence estimate.",
                         "Worksheet includes private source text and must not be published."],
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
