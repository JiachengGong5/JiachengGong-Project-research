"""PyTorch datasets and collators for generated sequence JSONL files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .labels import CATEGORY_NAMES, category_index, fine_index_within_category
from .manifest import ManifestRow, read_manifest
from .vocab import TokenVocab

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - depends on optional package
    raise ImportError(
        "PyTorch is required for activity_patterns.dataset. "
        "Install the project with the 'model' optional dependency."
    ) from exc


@dataclass(frozen=True)
class ChunkRef:
    row_index: int
    byte_offset: int


class SequenceChunkDataset(Dataset[dict[str, Any]]):
    """Lazy dataset where each JSONL line is one sequence chunk."""

    def __init__(
        self,
        rows: list[ManifestRow],
        vocab: TokenVocab,
        *,
        split: str | None = None,
        max_events: int | None = None,
    ) -> None:
        self.rows = [row for row in rows if split is None or row.split == split]
        self.vocab = vocab
        self.max_events = max_events
        self.refs = self._build_refs()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        vocab: TokenVocab,
        *,
        project_root: str | Path = ".",
        split: str | None = None,
        max_events: int | None = None,
    ) -> "SequenceChunkDataset":
        return cls(
            read_manifest(manifest_path, project_root=project_root),
            vocab,
            split=split,
            max_events=max_events,
        )

    def _build_refs(self) -> list[ChunkRef]:
        refs_by_row: list[list[ChunkRef]] = [[] for _ in self.rows]
        rows_by_path: dict[Path, list[int]] = {}
        for row_index, row in enumerate(self.rows):
            rows_by_path.setdefault(row.sequence_path, []).append(row_index)

        for sequence_path, row_indices in rows_by_path.items():
            file_level_rows = [
                row_index
                for row_index in row_indices
                if self.rows[row_index].chunk_index is None
            ]
            chunk_rows: dict[int, list[int]] = {}
            for row_index in row_indices:
                chunk_index = self.rows[row_index].chunk_index
                if chunk_index is not None:
                    chunk_rows.setdefault(chunk_index, []).append(row_index)

            with sequence_path.open("r", encoding="utf-8") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        for row_index in file_level_rows:
                            refs_by_row[row_index].append(
                                ChunkRef(row_index=row_index, byte_offset=offset)
                            )
                        if chunk_rows:
                            payload = json.loads(line)
                            chunk_index = int(payload.get("chunk_index", 0))
                            for row_index in chunk_rows.get(chunk_index, []):
                                refs_by_row[row_index].append(
                                    ChunkRef(row_index=row_index, byte_offset=offset)
                                )

        for row_index, row in enumerate(self.rows):
            if row.chunk_index is not None and not refs_by_row[row_index]:
                raise ValueError(
                    f"Could not find chunk_index={row.chunk_index} "
                    f"in {row.sequence_path}"
                )
        return [ref for row_refs in refs_by_row for ref in row_refs]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        row = self.rows[ref.row_index]
        with row.sequence_path.open("r", encoding="utf-8") as handle:
            handle.seek(ref.byte_offset)
            payload = json.loads(handle.readline())

        events = payload.get("events", [])
        if self.max_events is not None:
            events = events[: self.max_events]

        event_start = int(payload.get("event_start", 0))

        token_ids = [
            self.vocab.encode_event(event.get("tokens", []))
            for event in events
        ]
        return {
            "token_ids": token_ids,
            "length": len(token_ids),
            "sequence_id": payload.get("sequence_id", row.sequence_id),
            "chunk_index": int(payload.get("chunk_index", 0)),
            "source_chunk_index": int(
                payload.get("source_chunk_index", payload.get("chunk_index", 0))
            ),
            "event_start": event_start,
            "event_stop": event_start + len(events),
            "coarse_label": row.coarse_label,
            "fine_label": row.fine_label,
            "coarse_target": category_index(row.coarse_label),
            "fine_target": fine_index_within_category(row.fine_label),
        }


def collate_sequence_chunks(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_time = max(item["length"] for item in batch)
    max_fields = max(
        len(event)
        for item in batch
        for event in item["token_ids"]
    )
    token_ids = torch.zeros((len(batch), max_time, max_fields), dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    coarse_targets = torch.zeros((len(batch),), dtype=torch.long)
    fine_targets = torch.zeros((len(batch),), dtype=torch.long)

    for batch_index, item in enumerate(batch):
        lengths[batch_index] = item["length"]
        coarse_targets[batch_index] = item["coarse_target"]
        fine_targets[batch_index] = item["fine_target"]
        for time_index, event_tokens in enumerate(item["token_ids"]):
            token_ids[batch_index, time_index, : len(event_tokens)] = torch.tensor(
                event_tokens,
                dtype=torch.long,
            )

    return {
        "token_ids": token_ids,
        "lengths": lengths,
        "coarse_targets": coarse_targets,
        "fine_targets": fine_targets,
        "coarse_labels": [item["coarse_label"] for item in batch],
        "fine_labels": [item["fine_label"] for item in batch],
        "sequence_ids": [item["sequence_id"] for item in batch],
        "chunk_indices": [item["chunk_index"] for item in batch],
        "source_chunk_indices": [item["source_chunk_index"] for item in batch],
        "event_starts": [item["event_start"] for item in batch],
        "event_stops": [item["event_stop"] for item in batch],
        "category_names": CATEGORY_NAMES,
    }
