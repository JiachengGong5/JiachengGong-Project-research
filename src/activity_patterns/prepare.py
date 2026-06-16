"""CLI for writing labeled, contiguous protocol-event sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from .events import ProtocolEvent, iter_zeek_events


def contiguous_chunks(
    events: Iterable[ProtocolEvent], max_events: int
) -> Iterator[list[ProtocolEvent]]:
    """Split only for memory limits; never overlap or aggregate events."""

    if max_events < 1:
        raise ValueError("max_events must be positive")

    chunk: list[ProtocolEvent] = []
    for event in events:
        chunk.append(event)
        if len(chunk) == max_events:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_sequences(
    log_directory: str | Path,
    output_path: str | Path,
    *,
    label: str,
    sequence_id: str,
    max_events: int,
) -> int:
    """Write one JSON object per contiguous sequence chunk."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chunks_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for chunk_index, chunk in enumerate(
            contiguous_chunks(iter_zeek_events(log_directory), max_events)
        ):
            payload = {
                "sequence_id": sequence_id,
                "chunk_index": chunk_index,
                "label": label,
                "events": [event.as_dict() for event in chunk],
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            chunks_written += 1
    return chunks_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_directory", help="Directory containing Zeek JSON logs")
    parser.add_argument("output_path", help="Output JSONL path")
    parser.add_argument("--label", required=True, help="Capture-level class label")
    parser.add_argument(
        "--sequence-id",
        required=True,
        help="Stable original-capture ID used to prevent split leakage",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=4096,
        help="Maximum events per non-overlapping contiguous chunk",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = write_sequences(
        args.log_directory,
        args.output_path,
        label=args.label,
        sequence_id=args.sequence_id,
        max_events=args.max_events,
    )
    print(f"Wrote {count} sequence chunk(s) to {args.output_path}")


if __name__ == "__main__":
    main()

