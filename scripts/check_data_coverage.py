#!/usr/bin/env python3
"""Report CICIoT2023 raw/sequence coverage for coarse and fine labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.coverage import (  # noqa: E402
    collect_fine_coverage,
    missing_download_rows,
    summarize_coarse_coverage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--sequence-root", default="data/sequences")
    parser.add_argument(
        "--output-dir",
        default="artifacts/coverage",
        help="Directory for CSV and Markdown coverage reports",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print coverage; do not write report artifacts",
    )
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    coarse_rows: list[dict[str, object]],
    fine_rows: list[dict[str, object]],
    missing_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# CICIoT2023 Local Coverage",
        "",
        "## Coarse Coverage",
        "",
        "| Coarse label | Fine ready | Raw PCAPs | Sequences | Chunks | Complete |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in coarse_rows:
        lines.append(
            "| {coarse_label} | {fine_labels_ready}/{fine_labels_total} | "
            "{raw_pcaps} | {sequences} | {chunks} | {complete} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Missing PCAP Folders",
            "",
            "| Coarse label | Fine label | Download folder |",
            "| --- | --- | --- |",
        ]
    )
    if missing_rows:
        for row in missing_rows:
            lines.append(
                "| {coarse_label} | {fine_label} | {dataset_folder} |".format(**row)
            )
    else:
        lines.append("| - | - | None |")

    lines.extend(
        [
            "",
            "## Fine Coverage",
            "",
            "| Coarse label | Fine label | Folder | Raw PCAPs | Sequences | Chunks | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fine_rows:
        lines.append(
            "| {coarse_label} | {fine_label} | {dataset_folder} | "
            "{raw_pcaps} | {sequences} | {chunks} | {status} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_coarse_coverage(coarse_rows: list[dict[str, object]]) -> None:
    print("CICIoT2023 coarse-class coverage")
    print()
    print(
        f"{'category':<14} {'fine_ready':>10} {'raw_pcaps':>9} "
        f"{'sequences':>9} {'chunks':>8} complete"
    )
    print("-" * 72)
    for row in coarse_rows:
        ready = f"{row['fine_labels_ready']}/{row['fine_labels_total']}"
        print(
            f"{row['coarse_label']:<14} {ready:>10} "
            f"{row['raw_pcaps']:>9} {row['sequences']:>9} "
            f"{row['chunks']:>8} {row['complete']}"
        )


def print_missing(missing_rows: list[dict[str, object]]) -> None:
    print()
    if not missing_rows:
        print("All canonical CICIoT2023 fine labels have raw PCAP and sequence files.")
        return

    print("Missing fine-label PCAP folders:")
    for row in missing_rows:
        print(
            f"  - {row['coarse_label']}: {row['fine_label']} "
            f"(download folder: {row['dataset_folder']})"
        )


def main() -> None:
    args = build_parser().parse_args()
    fine_rows = collect_fine_coverage(
        raw_root=args.raw_root,
        sequence_root=args.sequence_root,
    )
    coarse_rows = summarize_coarse_coverage(fine_rows)
    missing_rows = missing_download_rows(fine_rows)

    print_coarse_coverage(coarse_rows)
    print_missing(missing_rows)

    if not args.no_write:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "coarse_coverage.csv", coarse_rows)
        write_csv(output_dir / "fine_coverage.csv", fine_rows)
        write_csv(output_dir / "missing_downloads.csv", missing_rows)
        write_markdown(
            output_dir / "coverage_report.md",
            coarse_rows,
            fine_rows,
            missing_rows,
        )
        print()
        print(f"Wrote coverage artifacts to {output_dir}")


if __name__ == "__main__":
    main()
