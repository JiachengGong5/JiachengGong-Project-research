#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 INPUT.pcap OUTPUT_DIRECTORY" >&2
  exit 2
fi

pcap="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_dir="$2"

if ! command -v zeek >/dev/null 2>&1; then
  echo "zeek is not installed or is not on PATH" >&2
  exit 1
fi

mkdir -p "$output_dir"
cd "$output_dir"

# -C ignores invalid checksums commonly found in offline captures.
# JSON preserves field names and types for deterministic preprocessing.
zeek -C -r "$pcap" LogAscii::use_json=T

