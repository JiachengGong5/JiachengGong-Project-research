"""Rechunk existing protocol-event sequences without losing split boundaries."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

from .manifest import ManifestRow


def rechunk_manifest_rows(
    rows: Iterable[ManifestRow],
    *,
    output_root: str | Path,
    project_root: str | Path = ".",
    max_events: int,
) -> tuple[list[ManifestRow], dict[str, int]]:
    """Write contiguous, non-overlapping event segments for manifest rows.

    Every child segment inherits the split and labels of its parent manifest
    row. This preserves the existing development split while ensuring that all
    events, rather than only the first training-time prefix, reach the model.
    """

    if max_events < 1:
        raise ValueError("max_events must be positive")

    root = Path(project_root).resolve()
    destination_root = Path(output_root)
    if not destination_root.is_absolute():
        destination_root = root / destination_root

    rows_by_path: dict[Path, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        source_path = row.sequence_path
        if not source_path.is_absolute():
            source_path = root / source_path
        rows_by_path[source_path].append(row)

    output_rows: list[ManifestRow] = []
    input_events = 0
    output_events = 0
    source_chunks = 0

    for source_path, path_rows in sorted(rows_by_path.items()):
        file_rows = [row for row in path_rows if row.chunk_index is None]
        chunk_rows: dict[int, list[ManifestRow]] = defaultdict(list)
        for row in path_rows:
            if row.chunk_index is not None:
                chunk_rows[row.chunk_index].append(row)

        output_path = destination_root / source_path.parent.name / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            manifest_output_path = output_path.relative_to(root)
        except ValueError:
            manifest_output_path = output_path

        next_chunk_index = 0
        with source_path.open("r", encoding="utf-8") as source_handle:
            with output_path.open("w", encoding="utf-8") as output_handle:
                for fallback_index, line in enumerate(source_handle):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    parent_chunk_index = int(
                        payload.get("chunk_index", fallback_index)
                    )
                    matching_rows = [
                        *file_rows,
                        *chunk_rows.get(parent_chunk_index, []),
                    ]
                    if not matching_rows:
                        continue

                    unique_rows = {
                        (
                            row.coarse_label,
                            row.fine_label,
                            row.split,
                        ): row
                        for row in matching_rows
                    }
                    if len(unique_rows) != 1:
                        raise ValueError(
                            "Conflicting manifest rows for "
                            f"{source_path} chunk {parent_chunk_index}"
                        )
                    parent_row = next(iter(unique_rows.values()))
                    events = payload.get("events", [])
                    source_chunks += 1
                    input_events += len(events)

                    for event_start in range(0, len(events), max_events):
                        segment = events[event_start : event_start + max_events]
                        event_stop = event_start + len(segment)
                        source_sequence_id = str(
                            payload.get("source_sequence_id")
                            or payload.get("sequence_id")
                            or parent_row.sequence_id
                        )
                        child_sequence_id = (
                            f"{source_sequence_id}:chunk-{parent_chunk_index:06d}"
                            f":events-{event_start:04d}-{event_stop:04d}"
                        )
                        child_payload = {
                            **payload,
                            "sequence_id": child_sequence_id,
                            "source_sequence_id": source_sequence_id,
                            "source_chunk_index": parent_chunk_index,
                            "event_start": event_start,
                            "event_stop": event_stop,
                            "chunk_index": next_chunk_index,
                            "events": segment,
                        }
                        output_handle.write(
                            json.dumps(child_payload, separators=(",", ":")) + "\n"
                        )
                        output_rows.append(
                            ManifestRow(
                                sequence_path=manifest_output_path,
                                sequence_id=child_sequence_id,
                                coarse_label=parent_row.coarse_label,
                                fine_label=parent_row.fine_label,
                                split=parent_row.split,
                                chunk_index=next_chunk_index,
                            )
                        )
                        next_chunk_index += 1
                        output_events += len(segment)

    return output_rows, {
        "source_files": len(rows_by_path),
        "source_chunks": source_chunks,
        "output_chunks": len(output_rows),
        "input_events": input_events,
        "output_events": output_events,
    }
