"""Token vocabulary for protocol-event sequences."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .manifest import ManifestRow


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True)
class TokenVocab:
    token_to_id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def encode_event(self, tokens: Iterable[str]) -> list[int]:
        ids = [self.encode_token(token) for token in tokens]
        return ids or [self.unk_id]

    def to_json(self) -> dict[str, object]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "TokenVocab":
        token_to_id = payload.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("Vocabulary JSON must contain token_to_id")
        return cls({str(token): int(index) for token, index in token_to_id.items()})

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "TokenVocab":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))


def iter_sequence_tokens(
    sequence_path: str | Path,
    *,
    chunk_index: int | None = None,
) -> Iterable[str]:
    chunk_indices = None if chunk_index is None else {chunk_index}
    yield from iter_selected_sequence_tokens(sequence_path, chunk_indices=chunk_indices)


def iter_selected_sequence_tokens(
    sequence_path: str | Path,
    *,
    chunk_indices: set[int] | None = None,
) -> Iterable[str]:
    with Path(sequence_path).open("r", encoding="utf-8") as handle:
        for fallback_index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            payload_chunk_index = int(payload.get("chunk_index", fallback_index))
            if chunk_indices is not None and payload_chunk_index not in chunk_indices:
                continue
            for event in payload.get("events", []):
                for token in event.get("tokens", []):
                    yield str(token)


def build_vocab(
    rows: Iterable[ManifestRow],
    *,
    min_freq: int = 1,
) -> TokenVocab:
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1")

    rows_by_path: dict[Path, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        rows_by_path[row.sequence_path].append(row)

    counter: Counter[str] = Counter()
    for sequence_path, path_rows in rows_by_path.items():
        chunk_indices = {
            row.chunk_index
            for row in path_rows
            if row.chunk_index is not None
        }
        read_all_chunks = any(row.chunk_index is None for row in path_rows)
        counter.update(
            iter_selected_sequence_tokens(
                sequence_path,
                chunk_indices=None if read_all_chunks else set(chunk_indices),
            )
        )

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counter.items()):
        if count >= min_freq:
            token_to_id[token] = len(token_to_id)
    return TokenVocab(token_to_id)
