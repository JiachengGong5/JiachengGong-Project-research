"""Build and read sequence manifests for model training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .labels import CICIOT2023_SCHEMA
from .schema import LabelSchema


MANIFEST_FIELDS = (
    "sequence_path",
    "sequence_id",
    "coarse_label",
    "fine_label",
    "split",
    "chunk_index",
)
REQUIRED_MANIFEST_FIELDS = tuple(
    field for field in MANIFEST_FIELDS if field != "chunk_index"
)


@dataclass(frozen=True)
class ManifestRow:
    sequence_path: Path
    sequence_id: str
    coarse_label: str
    fine_label: str
    split: str = "train"
    chunk_index: int | None = None


def _first_sequence_id(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                return str(payload.get("sequence_id") or path.stem)
    return path.stem


def infer_manifest_row(
    sequence_path: str | Path,
    *,
    root: str | Path = ".",
    split: str = "train",
    label_schema: LabelSchema = CICIOT2023_SCHEMA,
) -> ManifestRow:
    """Infer labels from a generated sequence JSONL path."""

    path = Path(sequence_path)
    label_source = path.parent.name
    fine_label = label_schema.canonical_fine_label(label_source)
    coarse_label = label_schema.coarse_label_for_fine(fine_label)
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        relative_path = path
    return ManifestRow(
        sequence_path=relative_path,
        sequence_id=_first_sequence_id(path),
        coarse_label=coarse_label,
        fine_label=fine_label,
        split=split,
    )


def build_manifest_rows(
    sequence_root: str | Path,
    *,
    project_root: str | Path = ".",
    split: str = "train",
    label_schema: LabelSchema = CICIOT2023_SCHEMA,
) -> list[ManifestRow]:
    root = Path(sequence_root)
    return [
        infer_manifest_row(
            path,
            root=project_root,
            split=split,
            label_schema=label_schema,
        )
        for path in sorted(root.rglob("*.jsonl"))
    ]


def write_manifest(rows: Iterable[ManifestRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_path": row.sequence_path.as_posix(),
                    "sequence_id": row.sequence_id,
                    "coarse_label": row.coarse_label,
                    "fine_label": row.fine_label,
                    "split": row.split,
                    "chunk_index": "" if row.chunk_index is None else row.chunk_index,
                }
            )


def read_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> list[ManifestRow]:
    root = Path(project_root)
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        rows = []
        for record in reader:
            raw_chunk_index = (record.get("chunk_index") or "").strip()
            rows.append(
                ManifestRow(
                    sequence_path=root / record["sequence_path"],
                    sequence_id=record["sequence_id"],
                    coarse_label=record["coarse_label"],
                    fine_label=record["fine_label"],
                    split=record.get("split") or "train",
                    chunk_index=int(raw_chunk_index) if raw_chunk_index else None,
                )
            )
    return rows
