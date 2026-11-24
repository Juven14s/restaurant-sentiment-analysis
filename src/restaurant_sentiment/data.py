"""Validate TSV annotations and create deterministic, frozen evaluation splits.

This module intentionally uses only the Python standard library. Review text is
kept in JSONL artifacts; manifests contain hashes and aggregate statistics only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import unicodedata
from typing import Any

ASPECTS = ("Prix", "Cuisine", "Service")
LABELS = ("Positive", "Négative", "Neutre", "NE")
SPLITS = ("train", "validation", "test")
_RATIOS = (0.8, 0.1, 0.1)
_SCHEMA_VERSION = 1


def normalize_text(text: str) -> str:
    """Canonical text used for leakage detection and stable review IDs."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _display_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent) + "\n").encode("utf-8")


def _read_tsv(path: Path, split_by: str) -> tuple[list[dict], dict]:
    content = path.read_bytes()
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: expected a UTF-8 TSV") from error
    reader = csv.reader(io.StringIO(decoded, newline=""), delimiter="\t", strict=True)
    records = []
    try:
        raw_header = next(reader, None)
        if not raw_header:
            raise ValueError(f"{path}: missing TSV header")
        header = [column.strip() for column in raw_header]
        if any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError(f"{path}: column names must be nonempty and unique")
        missing = set(("Avis", *ASPECTS)) - set(header)
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
        if split_by == "restaurant" and "Restaurant" not in header:
            raise ValueError(f"{path}: Restaurant column required for restaurant grouping")
        for row in reader:
            line = reader.line_num
            if len(row) != len(header):
                raise ValueError(f"{path}:{line}: expected {len(header)} TSV fields, got {len(row)}")
            values = dict(zip(header, row))
            text = _display_text(values["Avis"])
            if not text or "\x00" in text:
                raise ValueError(f"{path}:{line}: Avis must contain nonempty text without NUL characters")
            labels = {aspect: values[aspect].strip() for aspect in ASPECTS}
            for aspect, label in labels.items():
                if label not in LABELS:
                    raise ValueError(f"{path}:{line}: missing or invalid annotation for {aspect}")
            restaurant = _display_text(values.get("Restaurant", "")) or None
            if restaurant and "\x00" in restaurant:
                raise ValueError(f"{path}:{line}: Restaurant contains a NUL character")
            if split_by == "restaurant" and restaurant is None:
                raise ValueError(f"{path}:{line}: Restaurant cannot be empty for restaurant grouping")
            normalized = normalize_text(text)
            records.append({
                "id": _sha256(normalized.encode("utf-8")),
                "text": text,
                "labels": labels,
                "restaurant": restaurant,
                "_normalized": normalized,
            })
    except csv.Error as error:
        raise ValueError(f"{path}:{reader.line_num}: malformed TSV") from error
    # A filename identifies the source without exposing local machine paths;
    # moving identical inputs between a laptop and Colab preserves provenance.
    return records, {"path": path.name, "sha256": _sha256(content), "rows": len(records)}


def _deduplicate(records: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["id"], []).append(record)
    retained = []
    counts = {
        "input_rows": len(records),
        "unique_normalized_texts": len(groups),
        "duplicate_rows": len(records) - len(groups),
        "concordant_duplicate_rows_removed": 0,
        "conflicting_texts": 0,
        "conflicting_rows_removed": 0,
        "texts_with_multiple_restaurants": 0,
    }
    for identifier, members in sorted(groups.items()):
        signatures = {tuple(member["labels"][aspect] for aspect in ASPECTS) for member in members}
        if len(signatures) != 1:
            counts["conflicting_texts"] += 1
            counts["conflicting_rows_removed"] += len(members)
            continue
        # A deterministic representative preserves casing for model inputs.
        selected = min(members, key=lambda member: (member["text"], member["restaurant"] or ""))
        selected = dict(selected)
        restaurants = sorted({normalize_text(member["restaurant"]) for member in members if member["restaurant"]})
        selected["_restaurants"] = restaurants
        if len(restaurants) > 1:
            counts["texts_with_multiple_restaurants"] += 1
        counts["concordant_duplicate_rows_removed"] += len(members) - 1
        retained.append(selected)
    counts["retained_rows"] = len(retained)
    return retained, counts


def _targets(size: int) -> dict[str, int]:
    values = [math.floor(size * ratio) for ratio in _RATIOS]
    remainder_order = sorted(range(3), key=lambda index: (-(size * _RATIOS[index] - values[index]), index))
    for index in remainder_order[:size - sum(values)]:
        values[index] += 1
    # A tiny smoke-test corpus should still exercise all three splits.
    for index in (1, 2):
        if values[index] == 0 and size >= 3:
            values[0] -= 1
            values[index] += 1
    return dict(zip(SPLITS, values))


def _partition(records: list[dict], seed: int, split_by: str) -> tuple[dict[str, list[dict]], dict[str, int]]:
    def rank(key: str) -> str:
        return _sha256(f"{seed}\0{split_by}\0{key}".encode("utf-8"))

    targets = _targets(len(records))
    output: dict[str, list[dict]] = {split: [] for split in SPLITS}
    if split_by == "text":
        ordered = sorted(records, key=lambda row: (rank(row["id"]), row["id"]))
        offset = 0
        for split in SPLITS:
            output[split] = ordered[offset:offset + targets[split]]
            offset += targets[split]
        return output, {split: len(output[split]) for split in SPLITS}

    # Connect restaurants sharing a duplicate review. Grouping only the retained
    # representative's restaurant would otherwise leak those restaurants.
    parents: dict[str, str] = {}

    def find(value: str) -> str:
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    for record in records:
        restaurants = record["_restaurants"]
        root = min(find(value) for value in restaurants)
        for value in restaurants:
            parents[find(value)] = root
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = find(record["_restaurants"][0])
        groups.setdefault(key, []).append(record)
    if len(groups) < 3:
        raise ValueError("Restaurant grouping requires at least three independent restaurant groups")
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), rank(item[0]), item[0]))
    group_counts = {split: 0 for split in SPLITS}
    for index, (_, members) in enumerate(ordered_groups):
        empty = [split for split in SPLITS if not output[split]]
        candidates = empty if len(ordered_groups) - index == len(empty) else list(SPLITS)

        def error_if_assigned(split: str) -> tuple[float, int]:
            error = sum(
                (len(output[name]) + (len(members) if name == split else 0) - targets[name]) ** 2
                / max(targets[name], 1)
                for name in SPLITS
            )
            return error, SPLITS.index(split)

        selected = min(candidates, key=error_if_assigned)
        output[selected].extend(members)
        group_counts[selected] += 1
    return output, group_counts


def _label_stats(records: list[dict]) -> dict:
    return {
        aspect: {label: sum(row["labels"][aspect] == label for row in records) for label in LABELS}
        for aspect in ASPECTS
    }


def prepare_data(input_paths: list[Path], output_dir: Path, seed: int = 42, split_by: str = "text") -> dict:
    """Create frozen train/validation/test JSONL and a provenance manifest.

    All source TSVs are pooled before splitting, including any former validation
    set. A changed source, seed or grouping requires a *new* output directory.
    Invalid rows fail the entire operation. Conflicting duplicate texts are
    excluded. Restaurant grouping may approximate 80/10/10 due to group sizes.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if split_by not in ("text", "restaurant"):
        raise ValueError("split_by must be 'text' or 'restaurant'")
    paths = sorted({Path(path).expanduser().resolve() for path in input_paths}, key=str)
    if not paths:
        raise ValueError("At least one source TSV is required")
    records, sources = [], []
    for path in paths:
        rows, provenance = _read_tsv(path, split_by)
        records.extend(rows)
        sources.append(provenance)
    sources.sort(key=lambda source: (source["sha256"], source["path"]))
    records, counts = _deduplicate(records)
    if len(records) < 3:
        raise ValueError("At least three nonconflicting, unique review texts are required")
    partitions, group_counts = _partition(records, seed, split_by)
    options = {
        "seed": seed,
        "split_by": split_by,
        "requested_ratios": dict(zip(SPLITS, _RATIOS)),
        "normalization": "NFKC + casefold + whitespace",
        "deduplication": "discard_all_conflicting_labels; deterministic_representative",
        "assignment": "sha256_rank_v1" if split_by == "text" else "connected_restaurants_greedy_v1",
    }
    source_identity = sorted(
        ({"sha256": source["sha256"], "rows": source["rows"]} for source in sources),
        key=lambda source: (source["sha256"], source["rows"]),
    )
    fingerprint = _sha256(_json_bytes({"schema_version": _SCHEMA_VERSION, "sources": source_identity, "options": options}))
    content: dict[str, bytes] = {}
    files, overlaps = {}, {}
    for split, rows in partitions.items():
        public_rows = [
            {key: row[key] for key in ("id", "text", "labels", "restaurant")}
            for row in sorted(rows, key=lambda row: row["id"])
        ]
        data = b"".join(_json_bytes(row) for row in public_rows)
        content[f"{split}.jsonl"] = data
        files[split] = {"path": f"{split}.jsonl", "sha256": _sha256(data), "count": len(rows)}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            ids_left = {row["id"] for row in partitions[left]}
            ids_right = {row["id"] for row in partitions[right]}
            restaurants_left = {value for row in partitions[left] for value in row["_restaurants"]}
            restaurants_right = {value for row in partitions[right] for value in row["_restaurants"]}
            overlaps[f"{left}/{right}"] = {
                "text_id_overlap": len(ids_left & ids_right),
                "restaurant_overlap": len(restaurants_left & restaurants_right),
            }
    if any(value["text_id_overlap"] for value in overlaps.values()):
        raise RuntimeError("Internal split error: text leakage detected")
    if split_by == "restaurant" and any(value["restaurant_overlap"] for value in overlaps.values()):
        raise RuntimeError("Internal split error: restaurant leakage detected")
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "sources": sources,
        "options": options,
        "counts": counts,
        "files": files,
        "split_counts": {split: len(rows) for split, rows in partitions.items()},
        "actual_ratios": {split: len(rows) / len(records) for split, rows in partitions.items()},
        "group_counts": group_counts,
        "label_statistics": {"all": _label_stats(records), **{split: _label_stats(rows) for split, rows in partitions.items()}},
        "overlap_checks": overlaps,
        "limitations": [
            "Pooled sources are repartitioned; prior reported validation scores are not comparable.",
            "Splits are not label-stratified; inspect per-class support before interpreting metrics.",
            "Text grouping allows restaurant overlap; use restaurant grouping for unseen restaurants."
            if split_by == "text" else
            "Restaurant components remain intact; actual split ratios may differ from 80/10/10.",
        ],
    }
    content["manifest.json"] = _json_bytes(manifest, indent=2)
    output_dir = Path(output_dir)
    present = [name for name in content if (output_dir / name).exists()]
    if present:
        if len(present) != len(content) or any((output_dir / name).read_bytes() != data for name, data in content.items()):
            raise FileExistsError(
                f"Frozen split artifacts already exist in {output_dir}; use a new output directory for changed data or options"
            )
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites a concurrently-created frozen split.
    for name, data in content.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(data)
    return manifest


def load_jsonl(path: Path) -> list[dict]:
    """Load prepared reviews, validating schema, labels, and canonical text IDs."""
    result = []
    identifiers = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"id", "text", "labels", "restaurant"}:
                    raise ValueError("unexpected fields")
                if not isinstance(row["text"], str) or not normalize_text(row["text"]) or "\x00" in row["text"]:
                    raise ValueError("invalid text")
                if not isinstance(row["labels"], dict) or set(row["labels"]) != set(ASPECTS):
                    raise ValueError("invalid aspects")
                if any(row["labels"][aspect] not in LABELS for aspect in ASPECTS):
                    raise ValueError("invalid label")
                if row["restaurant"] is not None and (not isinstance(row["restaurant"], str) or not normalize_text(row["restaurant"])):
                    raise ValueError("invalid restaurant")
                expected_id = _sha256(normalize_text(row["text"]).encode("utf-8"))
                if row["id"] != expected_id or row["id"] in identifiers:
                    raise ValueError("invalid or duplicate ID")
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid prepared review ({error})") from error
            identifiers.add(row["id"])
            result.append(row)
    return result
