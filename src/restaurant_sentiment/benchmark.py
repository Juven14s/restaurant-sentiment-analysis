"""Freeze original diagnostic cases and label-independent validation probes."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import ASPECTS, LABELS, load_jsonl, normalize_text
from .metrics import evaluate_predictions


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _freeze(output_dir, content):
    output_dir = Path(output_dir)
    present = [name for name in content if (output_dir / name).exists()]
    if present:
        if len(present) != len(content) or any((output_dir/name).read_bytes() != data for name, data in content.items()):
            raise FileExistsError("Frozen benchmark differs; create a new version directory")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in content.items():
            with (output_dir/name).open('xb') as stream:
                stream.write(data)


def _jsonl(rows):
    return b''.join((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode() for row in rows)


def freeze_validation_probe(reference_path: Path, output_dir: Path, size=40, seed=42):
    reference_path = Path(reference_path)
    if reference_path.stem != 'validation' or size <= 0:
        raise ValueError("Probe sampling requires validation.jsonl and a positive sample size")
    manifest = json.loads((reference_path.parent/'manifest.json').read_text())
    source_sha = _sha(reference_path.read_bytes())
    if manifest['files']['validation']['sha256'] != source_sha:
        raise ValueError('Validation data no longer matches the frozen manifest')
    rows = load_jsonl(reference_path)
    if size > len(rows):
        raise ValueError('Probe size exceeds available validation examples')
    selected = sorted(rows, key=lambda r: _sha(f"{seed}:{r['id']}".encode()))[:size]
    payload = _jsonl(selected)
    options = dict(source_sha256=source_sha, size=size, seed=seed,
                   sampling='sha256_seed_id_rank_without_labels')
    report = dict(fingerprint=_sha(_json(options)), options=options,
        files={'probe': dict(path='probe.jsonl', sha256=_sha(payload), count=size)},
        purpose='development_probe_not_final_test')
    _freeze(output_dir, {'probe.jsonl':payload, 'manifest.json':_json(report)})
    return report


def freeze_challenge(cases_path: Path, output_dir: Path):
    cases = json.loads(Path(cases_path).read_text(encoding='utf-8'))
    rows, annotations, seen, case_ids = [], [], set(), set()
    for case in cases:
        if not isinstance(case.get('text'), str) or not case['text'].strip():
            raise ValueError('Challenge requires nonempty texts')
        if set(case.get('labels', {})) != set(ASPECTS) or any(case['labels'][a] not in LABELS for a in ASPECTS):
            raise ValueError('Invalid challenge labels')
        identifier = _sha(normalize_text(case['text']).encode())
        if identifier in seen or case['case_id'] in case_ids:
            raise ValueError('Duplicate challenge text or case ID')
        if not case.get('category') or not case.get('rationale'):
            raise ValueError('Challenge cases need a category and annotation rationale')
        seen.add(identifier); case_ids.add(case['case_id'])
        rows.append(dict(id=identifier,text=case['text'],labels=case['labels'],restaurant=None))
        annotations.append(dict(id=identifier,case_id=case['case_id'],category=case['category'],rationale=case['rationale']))
    if not rows:
        raise ValueError('Challenge must not be empty')
    payload = _jsonl(rows)
    metadata = _json(annotations)
    provenance = dict(kind='assistant_authored_diagnostic_challenge', version=1,
        reference_sha256=_sha(payload),annotations_sha256=_sha(metadata))
    report = dict(fingerprint=_sha(_json(provenance)), provenance=provenance,
        files={'challenge':dict(path='challenge.jsonl',sha256=_sha(payload),count=len(rows))},
        categories=dict(Counter(a['category'] for a in annotations)),
        label_support={a:dict(Counter(r['labels'][a] for r in rows)) for a in ASPECTS},
        limitations=['Assistant-authored and assistant-reviewed labels; not independently human validated.',
                     'Diagnostic convenience sample, not representative restaurant-review prevalence.',
                     'Frozen before model scoring; do not tune prompts against these reported scores.'])
    _freeze(output_dir, {'challenge.jsonl':payload,'annotations.json':metadata,'manifest.json':_json(report)})
    return report


def category_scores(reference_path, predictions_path, annotations_path):
    manifest=json.loads((Path(reference_path).parent/'manifest.json').read_text())
    if (_sha(Path(reference_path).read_bytes()) != manifest['files'][Path(reference_path).stem]['sha256']
        or _sha(Path(annotations_path).read_bytes()) != manifest['provenance']['annotations_sha256']):
        raise ValueError('Challenge reference or annotation hash mismatch')
    rows = load_jsonl(reference_path)
    predictions = [json.loads(line) for line in Path(predictions_path).read_text().splitlines()]
    evaluate_predictions(rows, predictions)  # Strict global alignment first.
    by_id = {r['id']:r for r in predictions}
    annotations = json.loads(Path(annotations_path).read_text())
    annotation_ids = [a['id'] for a in annotations]
    if len(set(annotation_ids)) != len(annotation_ids) or set(annotation_ids) != {r['id'] for r in rows}:
        raise ValueError('Challenge annotations must cover every reference ID exactly once')
    categories = {a['id']:a['category'] for a in annotations}
    result = {}
    for category in sorted(set(categories.values())):
        subset = [r for r in rows if categories[r['id']] == category]
        scores = evaluate_predictions(subset,[by_id[r['id']] for r in subset])
        # Per-category macro-F1 is omitted: several small slices have absent
        # classes, making fixed-four-class F1 particularly hard to interpret.
        result[category] = {k:scores[k] for k in ('n_samples','macro_accuracy','joint_exact_match','json_valid_rate')}
    return result
