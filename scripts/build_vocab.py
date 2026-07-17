#!/usr/bin/env python3
"""Build a protocol-event token vocabulary from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.manifest import read_manifest  # noqa: E402
from activity_patterns.vocab import build_vocab  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="manifests/smoke_manifest.csv",
        help="Input CSV manifest path",
    )
    parser.add_argument(
        "--output",
        default="artifacts/vocab.json",
        help="Output vocabulary JSON path",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum token count required to enter the vocabulary",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Only use this split to build the vocabulary; use 'all' for every row",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_manifest(args.manifest, project_root=PROJECT_ROOT)
    if args.split != "all":
        rows = [row for row in rows if row.split == args.split]
    vocab = build_vocab(rows, min_freq=args.min_freq)
    vocab.save(args.output)
    print(f"Wrote {len(vocab)} token(s) from {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
