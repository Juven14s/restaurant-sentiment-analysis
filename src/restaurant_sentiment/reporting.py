"""Small comparison tables with explicit provenance and no invented scores."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS = ("macro_f1", "macro_accuracy", "joint_exact_match", "json_valid_rate")


def experiment_signature(specification: dict) -> str:
    """Compare settings without exposing local paths in aggregate reports."""
    return hashlib.sha256(json.dumps(specification, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def summarize(paths: list[Path], output_path: Path) -> dict:
    if not paths or len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("Provide distinct metric files")
    groups = defaultdict(list)
    test_sets = set()
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if not all(type(result.get(k)) in (int, float) and math.isfinite(result[k])
                   and 0 <= result[k] <= 1 for k in METRICS):
            raise ValueError(f"Incomplete metric file: {path}")
        # Evaluation helper supplies the fingerprint of actual reference bytes.
        test_hash = result.get("reference_sha256")
        fingerprint = result.get("data_fingerprint")
        if not test_hash or not fingerprint:
            raise ValueError(f"Missing dataset provenance in {path}; use evaluate on prepared test.jsonl")
        test_sets.add((test_hash, fingerprint))
        method = result.get("method", "unspecified")
        groups[method].append(result)
    if len(test_sets) != 1:
        raise ValueError("Cannot compare metrics from different frozen test sets")
    rows = []
    for method, values in sorted(groups.items()):
        signatures = {r.get("experiment_signature") for r in values}
        if len(values) > 1 and (None in signatures or len(signatures) != 1):
            raise ValueError(f"Different or missing experiment configurations for {method}")
        seeds = [r.get("seed") for r in values]
        if len(values) > 1 and (None in seeds or len(set(seeds)) != len(seeds)):
            raise ValueError(f"Repeated/unspecified seeds for {method}")
        rows.append({
            "method": method,
            "n_runs": len(values),
            "seeds": seeds,
            "experiment_signature": values[0].get("experiment_signature"),
            **{k: {"mean": mean(v[k] for v in values),
                   "sample_std": stdev(v[k] for v in values) if len(values) > 1 else None}
               for k in METRICS},
        })
    reference_sha256, data_fingerprint = next(iter(test_sets))
    report = {"scale": "0 to 1", "reference_sha256": reference_sha256,
              "data_fingerprint": data_fingerprint, "results": rows}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
