#!/usr/bin/env python3
"""Import manually downloaded CICIoT2023 PCAP files into data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.coverage import FINE_LABEL_TO_FOLDER  # noqa: E402
from activity_patterns.labels import canonical_fine_label  # noqa: E402


EXPECTED_FILE_OVERRIDES = {
    "Benign_Final": "BenignTraffic.pcap",
}


def expected_pcap_files() -> dict[str, tuple[str, str]]:
    return {
        folder: (EXPECTED_FILE_OVERRIDES.get(folder, f"{folder}.pcap"), folder)
        for folder in FINE_LABEL_TO_FOLDER.values()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads-dir",
        default=str(Path.home() / "Downloads"),
        help="Directory where browser-downloaded PCAP files are located",
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default is copy.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def find_download(downloads_dir: Path, expected_name: str) -> Path | None:
    direct = downloads_dir / expected_name
    if direct.exists():
        return direct

    matches = sorted(downloads_dir.rglob(expected_name))
    if matches:
        return matches[0]

    stem = Path(expected_name).stem
    suffix = Path(expected_name).suffix
    alternate_name = f"{stem}-{suffix}"
    alternate = downloads_dir / alternate_name
    if alternate.exists():
        return alternate

    alternate_matches = sorted(downloads_dir.rglob(alternate_name))
    if alternate_matches:
        return alternate_matches[0]
    return None


def main() -> None:
    args = build_parser().parse_args()
    downloads_dir = Path(args.downloads_dir)
    raw_root = Path(args.raw_root)

    print(f"downloads_dir={downloads_dir}")
    print(f"raw_root={raw_root}")
    print()

    missing = []
    imported = []
    for folder, (expected_name, raw_label) in expected_pcap_files().items():
        canonical_fine_label(raw_label)
        source = find_download(downloads_dir, expected_name)
        destination = raw_root / folder / expected_name

        if destination.exists():
            print(f"exists: {destination}")
            imported.append(destination)
            continue

        if source is None:
            print(f"missing: {expected_name}")
            missing.append(expected_name)
            continue

        print(f"{'move' if args.move else 'copy'}: {source} -> {destination}")
        imported.append(destination)
        if args.dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)

    print()
    print(f"ready_files={len(imported)} missing_files={len(missing)}")
    if missing:
        print("Still need to download:")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
