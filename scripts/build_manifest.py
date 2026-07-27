#!/usr/bin/env python3
"""Build a CSV manifest from generated sequence JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.labels import CICIOT2023_SCHEMA  # noqa: E402
from activity_patterns.manifest import build_manifest_rows, write_manifest  # noqa: E402
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
        default="manifests/smoke_manifest.csv",
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name assigned to discovered rows",
    )
    parser.add_argument(
        "--label-schema",
        help=(
            "Optional JSON label schema. Without it, CICIoT2023 labels are used."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_schema = (
        LabelSchema.load(args.label_schema)
        if args.label_schema
        else CICIOT2023_SCHEMA
    )
    rows = build_manifest_rows(
        args.sequence_root,
        project_root=PROJECT_ROOT,
        split=args.split,
        label_schema=label_schema,
    )
    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
