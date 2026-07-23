#!/usr/bin/env python3
"""Rechunk sequence JSONL files while preserving parent manifest splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.manifest import read_manifest, write_manifest  # noqa: E402
from activity_patterns.rechunk import rechunk_manifest_rows  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/dev_chunk_manifest.csv")
    parser.add_argument("--output-root", default="data/sequences_windowed")
    parser.add_argument(
        "--output-manifest",
        default="manifests/dev_event_manifest.csv",
    )
    parser.add_argument("--max-events", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_manifest(args.manifest, project_root=PROJECT_ROOT)
    output_rows, stats = rechunk_manifest_rows(
        rows,
        output_root=args.output_root,
        project_root=PROJECT_ROOT,
        max_events=args.max_events,
    )
    write_manifest(output_rows, PROJECT_ROOT / args.output_manifest)

    print(f"source_files={stats['source_files']}")
    print(f"source_chunks={stats['source_chunks']}")
    print(f"output_chunks={stats['output_chunks']}")
    print(f"input_events={stats['input_events']}")
    print(f"output_events={stats['output_events']}")
    print(f"event_coverage={stats['output_events'] / max(stats['input_events'], 1):.3f}")
    print(f"output_manifest={args.output_manifest}")


if __name__ == "__main__":
    main()
