#!/usr/bin/env python3
"""Batch process labeled PCAP folders into protocol-event sequence JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.events import FIELDS_BY_LOG  # noqa: E402
from activity_patterns.labels import CICIOT2023_SCHEMA  # noqa: E402
from activity_patterns.prepare import write_sequences  # noqa: E402
from activity_patterns.schema import LabelSchema  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--zeek-root", default="data/zeek")
    parser.add_argument("--sequence-root", default="data/sequences")
    parser.add_argument("--zeek-sample-root", default="data/zeek_sample")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Process only this raw label folder. May be repeated.",
    )
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument(
        "--sample-conn-lines",
        type=int,
        default=0,
        help="If set, build sequences from a sampled Zeek directory with only the first N conn.log lines.",
    )
    parser.add_argument("--force-zeek", action="store_true")
    parser.add_argument("--force-sequences", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--label-schema",
        help=(
            "Optional JSON label schema. Without it, CICIoT2023 labels are used."
        ),
    )
    return parser


def discover_pcaps(raw_root: Path, only: set[str]) -> list[Path]:
    pcaps = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )
    if only:
        pcaps = [path for path in pcaps if path.parent.name in only]
    return pcaps


def run_zeek(pcap: Path, output_dir: Path, *, force: bool, dry_run: bool) -> None:
    if output_dir.exists() and any(output_dir.glob("*.log")) and not force:
        print(f"zeek: skip existing {output_dir}")
        return

    print(f"zeek: {pcap} -> {output_dir}")
    if dry_run:
        return

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["zeek", "-C", "-r", str(pcap.resolve()), "LogAscii::use_json=T"],
        cwd=output_dir,
        check=True,
    )


def copy_lines(source: Path, destination: Path, max_lines: int) -> None:
    with source.open("r", encoding="utf-8") as src, destination.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if line_number > max_lines:
                break
            dst.write(line)


def build_sample_log_dir(
    source_dir: Path,
    sample_dir: Path,
    *,
    sample_conn_lines: int,
    dry_run: bool,
) -> Path:
    if sample_conn_lines <= 0:
        return source_dir

    print(f"sample: {source_dir} -> {sample_dir} conn_lines={sample_conn_lines}")
    if dry_run:
        return sample_dir

    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for log_type in FIELDS_BY_LOG:
        source = source_dir / f"{log_type}.log"
        if not source.exists():
            continue
        destination = sample_dir / source.name
        if log_type == "conn":
            copy_lines(source, destination, sample_conn_lines)
        else:
            shutil.copy2(source, destination)
    return sample_dir


def process_sequences(
    log_dir: Path,
    sequence_path: Path,
    *,
    raw_label: str,
    sequence_id: str,
    max_events: int,
    label_schema: LabelSchema,
    force: bool,
    dry_run: bool,
) -> None:
    if sequence_path.exists() and not force:
        print(f"sequence: skip existing {sequence_path}")
        return

    print(f"sequence: {log_dir} -> {sequence_path}")
    if dry_run:
        return

    fine_label = label_schema.canonical_fine_label(raw_label)
    write_sequences(
        log_dir,
        sequence_path,
        label=fine_label,
        sequence_id=sequence_id,
        max_events=max_events,
    )


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root)
    zeek_root = Path(args.zeek_root)
    sequence_root = Path(args.sequence_root)
    sample_root = Path(args.zeek_sample_root)
    label_schema = (
        LabelSchema.load(args.label_schema)
        if args.label_schema
        else CICIOT2023_SCHEMA
    )
    only = set(args.only)

    pcaps = discover_pcaps(raw_root, only)
    if not pcaps:
        raise SystemExit(f"No PCAP files found under {raw_root}")

    for pcap in pcaps:
        raw_label = pcap.parent.name
        stem = pcap.stem
        zeek_dir = zeek_root / raw_label / stem
        sequence_suffix = ".sample.jsonl" if args.sample_conn_lines else ".jsonl"
        sequence_path = sequence_root / raw_label / f"{stem}{sequence_suffix}"

        run_zeek(pcap, zeek_dir, force=args.force_zeek, dry_run=args.dry_run)
        effective_log_dir = build_sample_log_dir(
            zeek_dir,
            sample_root / raw_label / stem,
            sample_conn_lines=args.sample_conn_lines,
            dry_run=args.dry_run,
        )
        process_sequences(
            effective_log_dir,
            sequence_path,
            raw_label=raw_label,
            sequence_id=f"{stem}-sample" if args.sample_conn_lines else stem,
            max_events=args.max_events,
            label_schema=label_schema,
            force=args.force_sequences,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
