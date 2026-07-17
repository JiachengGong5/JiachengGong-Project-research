"""Map salient sequence spans back to Zeek activity traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


TRACE_LOG_TYPES = ("conn", "dns", "http", "ssl", "ssh", "dhcp", "weird")
TRACE_FIELDS: dict[str, tuple[str, ...]] = {
    "conn": (
        "ts",
        "uid",
        "id.orig_h",
        "id.orig_p",
        "id.resp_h",
        "id.resp_p",
        "proto",
        "service",
        "conn_state",
        "history",
    ),
    "dns": (
        "ts",
        "uid",
        "id.orig_h",
        "id.resp_h",
        "query",
        "qtype_name",
        "rcode_name",
        "opcode_name",
        "rejected",
    ),
    "http": (
        "ts",
        "uid",
        "id.orig_h",
        "id.resp_h",
        "method",
        "host",
        "uri",
        "status_code",
        "user_agent",
    ),
    "ssl": (
        "ts",
        "uid",
        "id.orig_h",
        "id.resp_h",
        "version",
        "cipher",
        "resumed",
        "established",
        "validation_status",
    ),
    "ssh": (
        "ts",
        "uid",
        "id.orig_h",
        "id.resp_h",
        "version",
        "auth_success",
        "direction",
    ),
    "dhcp": ("ts", "uid", "client_addr", "server_addr", "msg_types"),
    "weird": ("ts", "uid", "id.orig_h", "id.resp_h", "name"),
}


def resolve_zeek_log_dir(
    sequence_path: str | Path,
    *,
    project_root: str | Path = ".",
    zeek_root: str | Path = "data/zeek",
    zeek_sample_root: str | Path = "data/zeek_sample",
) -> Path:
    """Infer the Zeek log directory used to create a sequence JSONL file."""

    root = Path(project_root)
    path = Path(sequence_path)
    if not path.is_absolute():
        path = root / path

    raw_label = path.parent.name
    if path.name.endswith(".sample.jsonl"):
        capture_stem = path.name[: -len(".sample.jsonl")]
        return root / zeek_sample_root / raw_label / capture_stem

    if path.name.endswith(".jsonl"):
        capture_stem = path.name[: -len(".jsonl")]
    else:
        capture_stem = path.stem
    return root / zeek_root / raw_label / capture_stem


def iter_zeek_json_records(log_dir: str | Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield raw Zeek JSON records with their log type."""

    directory = Path(log_dir)
    if not directory.exists():
        return

    for log_type in TRACE_LOG_TYPES:
        path = directory / f"{log_type}.log"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                payload = json.loads(line)
                yield log_type, payload


def summarize_trace_record(log_type: str, record: dict[str, Any]) -> dict[str, Any]:
    """Keep useful tracking metadata without changing model inputs."""

    fields = TRACE_FIELDS.get(log_type, ("ts", "uid"))
    summary = {"log_type": log_type}
    for field in fields:
        if field in record:
            summary[field] = record[field]
    return summary


def reconstruct_activity_trace(
    log_dir: str | Path,
    *,
    start_ts: float,
    end_ts: float,
    context_seconds: float = 1.0,
    uid_context_seconds: float = 5.0,
    max_records: int = 40,
) -> list[dict[str, Any]]:
    """Return Zeek records in and around a salient model span.

    The first pass finds records in the time window and collects their Zeek
    ``uid`` values. The second pass includes nearby protocol records linked by
    those ``uid`` values, so HTTP/DNS/SSL records can be shown together with
    the related connection where available without pulling in a long-lived
    flow's entire history.
    """

    lower = start_ts - context_seconds
    upper = end_ts + context_seconds
    window_records: list[tuple[str, dict[str, Any]]] = []
    linked_uids: set[str] = set()

    for log_type, record in iter_zeek_json_records(log_dir):
        timestamp = record.get("ts")
        if timestamp is None:
            continue
        timestamp_value = float(timestamp)
        if lower <= timestamp_value <= upper:
            window_records.append((log_type, record))
            uid = record.get("uid")
            if uid:
                linked_uids.add(str(uid))

    linked_records: list[tuple[str, dict[str, Any]]] = []
    if linked_uids:
        linked_lower = start_ts - context_seconds - uid_context_seconds
        linked_upper = end_ts + context_seconds + uid_context_seconds
        for log_type, record in iter_zeek_json_records(log_dir):
            uid = record.get("uid")
            timestamp = record.get("ts")
            if timestamp is None:
                continue
            timestamp_value = float(timestamp)
            if (
                uid
                and str(uid) in linked_uids
                and linked_lower <= timestamp_value <= linked_upper
            ):
                linked_records.append((log_type, record))

    seen: set[tuple[str, float | None, str | None, str]] = set()
    merged = []
    for log_type, record in window_records + linked_records:
        key = (
            log_type,
            float(record["ts"]) if "ts" in record else None,
            str(record.get("uid")) if record.get("uid") else None,
            json.dumps(record, sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(summarize_trace_record(log_type, record))

    merged.sort(key=lambda item: float(item.get("ts", 0.0)))
    return merged[:max_records]
