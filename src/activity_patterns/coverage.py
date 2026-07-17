"""Coverage helpers for CICIoT2023 labels and generated sequence chunks."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable

from .labels import (
    CATEGORY_NAMES,
    CATEGORY_TO_FINE_LABELS,
    canonical_fine_label,
    coarse_label_for_fine,
)


FINE_LABEL_TO_FOLDER: dict[str, str] = {
    "Benign": "Benign_Final",
    "DDoS-ACK fragmentation": "DDoS-ACK_Fragmentation",
    "DDoS-UDP flood": "DDoS-UDP_Flood",
    "DDoS-SlowLoris": "DDoS-SlowLoris",
    "DDoS-ICMP flood": "DDoS-ICMP_Flood",
    "DDoS-RSTFIN flood": "DDoS-RSTFINFlood",
    "DDoS-PSHACK flood": "DDoS-PSHACK_Flood",
    "DDoS-HTTP flood": "DDoS-HTTP_Flood",
    "DDoS-UDP fragmentation": "DDoS-UDP_Fragmentation",
    "DDoS-TCP flood": "DDoS-TCP_Flood",
    "DDoS-SYN flood": "DDoS-SYN_Flood",
    "DDoS-SynonymousIP flood": "DDoS-SynonymousIP_Flood",
    "DoS-TCP flood": "DoS-TCP_Flood",
    "DoS-HTTP flood": "DoS-HTTP_Flood",
    "DoS-SYN flood": "DoS-SYN_Flood",
    "DoS-UDP flood": "DoS-UDP_Flood",
    "Recon-Ping sweep": "Recon-PingSweep",
    "Recon-OS scan": "Recon-OSScan",
    "Recon-Vulnerability scan": "VulnerabilityScan",
    "Recon-Port scan": "Recon-PortScan",
    "Recon-Host discovery": "Recon-HostDiscovery",
    "Web-Sql injection": "SqlInjection",
    "Web-Command injection": "CommandInjection",
    "Web-Backdoor malware": "Backdoor_Malware",
    "Web-Uploading attack": "Uploading_Attack",
    "Web-XSS": "XSS",
    "Web-Browser hijacking": "BrowserHijacking",
    "BruteForce-Dictionary brute force": "DictionaryBruteForce",
    "Spoofing-Arp spoofing": "MITM-ArpSpoofing",
    "Spoofing-DNS spoofing": "DNS_Spoofing",
    "Mirai-GREIP flood": "Mirai-greip_flood",
    "Mirai-Greeth flood": "Mirai-greeth_flood",
    "Mirai-UDPPlain": "Mirai-udpplain",
}


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def collect_fine_coverage(
    *,
    raw_root: str | Path = "data/raw",
    sequence_root: str | Path = "data/sequences",
) -> list[dict[str, object]]:
    raw_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    unknown_raw: list[str] = []
    unknown_sequences: list[str] = []

    for path in sorted(Path(raw_root).glob("*/*.pcap")):
        try:
            raw_counts[canonical_fine_label(path.parent.name)] += 1
        except KeyError:
            unknown_raw.append(path.as_posix())

    for path in sorted(Path(sequence_root).glob("*/*.jsonl")):
        try:
            fine_label = canonical_fine_label(path.parent.name)
        except KeyError:
            unknown_sequences.append(path.as_posix())
            continue
        sequence_counts[fine_label] += 1
        chunk_counts[fine_label] += count_jsonl_lines(path)

    rows = []
    for category in CATEGORY_NAMES:
        for fine_label in CATEGORY_TO_FINE_LABELS[category]:
            raw_pcaps = raw_counts[fine_label]
            sequences = sequence_counts[fine_label]
            rows.append(
                {
                    "coarse_label": category,
                    "fine_label": fine_label,
                    "dataset_folder": FINE_LABEL_TO_FOLDER[fine_label],
                    "raw_pcaps": raw_pcaps,
                    "sequences": sequences,
                    "chunks": chunk_counts[fine_label],
                    "status": "ready" if raw_pcaps > 0 and sequences > 0 else "missing",
                }
            )

    for path in unknown_raw:
        rows.append(
            {
                "coarse_label": "UNKNOWN",
                "fine_label": "UNKNOWN",
                "dataset_folder": path,
                "raw_pcaps": 1,
                "sequences": 0,
                "chunks": 0,
                "status": "unknown_raw",
            }
        )
    for path in unknown_sequences:
        rows.append(
            {
                "coarse_label": "UNKNOWN",
                "fine_label": "UNKNOWN",
                "dataset_folder": path,
                "raw_pcaps": 0,
                "sequences": 1,
                "chunks": 0,
                "status": "unknown_sequence",
            }
        )
    return rows


def summarize_coarse_coverage(
    fine_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in fine_rows:
        category = str(row["coarse_label"])
        if category != "UNKNOWN":
            by_category[category].append(row)

    summaries = []
    for category in CATEGORY_NAMES:
        rows = by_category[category]
        ready = sum(1 for row in rows if row["status"] == "ready")
        summaries.append(
            {
                "coarse_label": category,
                "fine_labels_ready": ready,
                "fine_labels_total": len(rows),
                "raw_pcaps": sum(int(row["raw_pcaps"]) for row in rows),
                "sequences": sum(int(row["sequences"]) for row in rows),
                "chunks": sum(int(row["chunks"]) for row in rows),
                "complete": ready == len(rows),
            }
        )
    return summaries


def missing_download_rows(
    fine_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    return [row for row in fine_rows if row["status"] == "missing"]


def rows_to_json(rows: Iterable[dict[str, object]]) -> str:
    return json.dumps(list(rows), indent=2, sort_keys=True) + "\n"
