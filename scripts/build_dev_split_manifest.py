#!/usr/bin/env python3
"""Build a stratified chunk-level development manifest.

This is useful when the local subset has only one PCAP per class, so a
capture-disjoint validation split is not yet possible. It should be treated as
development validation, not as the final reported evaluation protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.labels import CICIOT2023_SCHEMA  # noqa: E402
from activity_patterns.manifest import ManifestRow, write_manifest  # noqa: E402
from activity_patterns.schema import LabelSchema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-root",
        default="data/sequences",
        help="Root directory containing generated sequence JSONL files",
    )
    parser.add_argument(
        "--output",
        default="manifests/dev_chunk_manifest.csv",
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation fraction per class for the development split",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--stratify-by",
        choices=("coarse", "fine"),
        default="coarse",
        help="Label level used for stratified splitting",
    )
    parser.add_argument(
        "--split-unit",
        choices=("chunk", "capture", "auto"),
        default="chunk",
        help=(
            "Split individual chunks, whole capture files, or prefer capture "
            "splits and fall back to chunks when a class has one capture"
        ),
    )
    parser.add_argument(
        "--label-schema",
        help=(
            "Optional JSON label schema. Without it, CICIoT2023 labels are used."
        ),
    )
    return parser


def iter_chunk_rows(
    sequence_root: Path,
    *,
    label_schema: LabelSchema = CICIOT2023_SCHEMA,
    project_root: Path = PROJECT_ROOT,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for sequence_path in sorted(sequence_root.rglob("*.jsonl")):
        label_source = sequence_path.parent.name
        fine_label = label_schema.canonical_fine_label(label_source)
        coarse_label = label_schema.coarse_label_for_fine(fine_label)
        try:
            relative_path = sequence_path.relative_to(project_root)
        except ValueError:
            relative_path = sequence_path

        with sequence_path.open("r", encoding="utf-8") as handle:
            for fallback_index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunk_index = int(payload.get("chunk_index", fallback_index))
                sequence_id = str(payload.get("sequence_id") or sequence_path.stem)
                rows.append(
                    ManifestRow(
                        sequence_path=relative_path,
                        sequence_id=f"{sequence_id}:chunk-{chunk_index:06d}",
                        coarse_label=coarse_label,
                        fine_label=fine_label,
                        split="train",
                        chunk_index=chunk_index,
                    )
                )
    return rows


def assign_development_splits(
    rows: list[ManifestRow],
    *,
    val_ratio: float,
    stratify_by: str,
    seed: int,
    split_unit: str = "chunk",
) -> list[ManifestRow]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")

    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = row.coarse_label if stratify_by == "coarse" else row.fine_label
        groups[key].append(index)

    if split_unit not in {"chunk", "capture", "auto"}:
        raise ValueError("split_unit must be chunk, capture, or auto")

    val_indices: set[int] = set()
    for indices in groups.values():
        capture_groups: dict[Path, list[int]] = defaultdict(list)
        for index in indices:
            capture_groups[rows[index].sequence_path].append(index)

        use_capture_split = split_unit == "capture" or (
            split_unit == "auto" and len(capture_groups) > 1
        )
        units = (
            list(capture_groups.values())
            if use_capture_split
            else [[index] for index in indices]
        )
        rng.shuffle(units)
        if len(units) <= 1:
            val_count = 0
        else:
            val_count = max(1, round(len(units) * val_ratio))
            val_count = min(val_count, len(units) - 1)
        for unit in units[:val_count]:
            val_indices.update(unit)

    split_rows = []
    for index, row in enumerate(rows):
        split_rows.append(
            ManifestRow(
                sequence_path=row.sequence_path,
                sequence_id=row.sequence_id,
                coarse_label=row.coarse_label,
                fine_label=row.fine_label,
                split="val" if index in val_indices else "train",
                chunk_index=row.chunk_index,
            )
        )
    return split_rows


def print_distribution(rows: list[ManifestRow]) -> None:
    counts = Counter((row.coarse_label, row.split) for row in rows)
    labels = sorted({row.coarse_label for row in rows})
    print("Development split distribution")
    print(f"{'coarse_label':<14} {'train':>6} {'val':>6} {'total':>6}")
    print("-" * 36)
    for label in labels:
        train = counts[(label, "train")]
        val = counts[(label, "val")]
        print(f"{label:<14} {train:>6} {val:>6} {train + val:>6}")


def main() -> None:
    args = build_parser().parse_args()
    label_schema = (
        LabelSchema.load(args.label_schema)
        if args.label_schema
        else CICIOT2023_SCHEMA
    )
    sequence_root = Path(args.sequence_root)
    if not sequence_root.is_absolute():
        sequence_root = PROJECT_ROOT / sequence_root
    rows = iter_chunk_rows(
        sequence_root,
        label_schema=label_schema,
    )
    if not rows:
        raise ValueError(f"No sequence chunks found under {args.sequence_root}")

    split_rows = assign_development_splits(
        rows,
        val_ratio=args.val_ratio,
        stratify_by=args.stratify_by,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    write_manifest(split_rows, PROJECT_ROOT / args.output)
    print(f"Wrote {len(split_rows)} chunk row(s) to {args.output}")
    print_distribution(split_rows)
    print(f"split_unit={args.split_unit}")
    if args.split_unit == "chunk":
        print(
            "Note: this is a chunk-level development split. Use capture-disjoint "
            "splits for final reported results once multiple captures per class "
            "are available."
        )
    elif args.split_unit == "auto":
        print(
            "Note: auto mode keeps captures disjoint where a class has multiple "
            "captures and falls back to chunks for single-capture classes."
        )
    else:
        print("Capture files are kept disjoint between train and validation.")


if __name__ == "__main__":
    main()
