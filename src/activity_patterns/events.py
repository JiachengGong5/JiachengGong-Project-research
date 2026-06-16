"""Convert Zeek JSON logs into ordered, categorical protocol events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Iterator


# These are direct protocol semantics, not aggregate traffic statistics.
FIELDS_BY_LOG: dict[str, tuple[tuple[str, str], ...]] = {
    "conn": (
        ("proto", "proto"),
        ("service", "service"),
        ("conn_state", "conn_state"),
        ("history", "history"),
        ("id.resp_p", "resp_port"),
    ),
    "http": (
        ("method", "method"),
        ("version", "version"),
        ("status_code", "status_code"),
    ),
    "dns": (
        ("qtype_name", "qtype_name"),
        ("rcode_name", "rcode_name"),
        ("opcode_name", "opcode_name"),
        ("rejected", "rejected"),
    ),
    "ssl": (
        ("version", "version"),
        ("cipher", "cipher"),
        ("resumed", "resumed"),
        ("established", "established"),
        ("validation_status", "validation_status"),
    ),
    "ssh": (
        ("version", "version"),
        ("auth_success", "auth_success"),
        ("direction", "direction"),
    ),
    "dhcp": (("msg_types", "msg_type"),),
    "weird": (("name", "name"),),
}


@dataclass(frozen=True)
class ProtocolEvent:
    """One chronological LSTM time step."""

    timestamp: float
    log_type: str
    tokens: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "log_type": self.log_type,
            "tokens": list(self.tokens),
        }


def _token_values(value: Any) -> Iterable[str]:
    if value is None or value == "" or value == "-":
        return ()
    if isinstance(value, bool):
        return ("true" if value else "false",)
    if isinstance(value, list):
        return (str(item) for item in value if item is not None and item != "")
    return (str(value),)


def record_to_event(log_type: str, record: dict[str, Any]) -> ProtocolEvent:
    """Build an event using only the allow-listed direct semantic fields."""

    if "ts" not in record:
        raise ValueError(f"{log_type}.log record has no ts field")

    tokens = [f"log={log_type}"]
    for source_name, token_name in FIELDS_BY_LOG[log_type]:
        for value in _token_values(record.get(source_name)):
            tokens.append(f"{token_name}={value}")

    return ProtocolEvent(
        timestamp=float(record["ts"]),
        log_type=log_type,
        tokens=tuple(tokens),
    )


def iter_json_log(path: Path, log_type: str) -> Iterator[ProtocolEvent]:
    """Read one chronological Zeek JSON log."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                yield record_to_event(log_type, record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def iter_zeek_events(log_directory: str | Path) -> Iterator[ProtocolEvent]:
    """Globally sort supported Zeek logs into one chronological sequence.

    A Zeek log file is not guaranteed to be ordered by its ``ts`` field. For
    example, a connection record can be written when the connection ends while
    ``ts`` still represents its start time. SQLite provides a disk-backed sort
    so large captures do not need to fit in Python memory.
    """

    directory = Path(log_directory)
    paths: list[tuple[str, Path]] = []
    for log_type in FIELDS_BY_LOG:
        path = directory / f"{log_type}.log"
        if path.exists():
            paths.append((log_type, path))

    if not paths:
        supported = ", ".join(f"{name}.log" for name in FIELDS_BY_LOG)
        raise FileNotFoundError(
            f"No supported Zeek JSON logs found in {directory}. Expected: {supported}"
        )

    with tempfile.TemporaryDirectory(prefix="activity-patterns-sort-") as temp_dir:
        database_path = Path(temp_dir) / "events.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """
                CREATE TABLE events (
                    ordinal INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    log_type TEXT NOT NULL,
                    tokens TEXT NOT NULL
                )
                """
            )

            ordinal = 0
            for log_type, path in paths:
                rows = []
                for event in iter_json_log(path, log_type):
                    rows.append(
                        (
                            ordinal,
                            event.timestamp,
                            event.log_type,
                            json.dumps(event.tokens, separators=(",", ":")),
                        )
                    )
                    ordinal += 1
                    if len(rows) == 10_000:
                        connection.executemany(
                            "INSERT INTO events VALUES (?, ?, ?, ?)", rows
                        )
                        rows = []
                if rows:
                    connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", rows)
            connection.commit()

            cursor = connection.execute(
                """
                SELECT timestamp, log_type, tokens
                FROM events
                ORDER BY timestamp, ordinal
                """
            )
            for timestamp, log_type, tokens_json in cursor:
                yield ProtocolEvent(
                    timestamp=timestamp,
                    log_type=log_type,
                    tokens=tuple(json.loads(tokens_json)),
                )
        finally:
            connection.close()
