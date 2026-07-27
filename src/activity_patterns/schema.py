"""Dataset-independent label schemas for hierarchical classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import re
from typing import Iterable, Mapping


def label_key(label: str) -> str:
    """Return a forgiving comparison key for folder and class labels."""

    return re.sub(r"[^a-z0-9]+", "", label.lower())


@dataclass(frozen=True)
class LabelSchema:
    """Describe coarse categories, fine labels, and dataset folder aliases."""

    name: str
    category_to_fine_labels: dict[str, tuple[str, ...]]
    aliases: dict[str, str]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Label schema name cannot be empty")
        if not self.category_to_fine_labels:
            raise ValueError("Label schema must contain at least one category")

        seen_fine: set[str] = set()
        for category, fine_labels in self.category_to_fine_labels.items():
            if not category:
                raise ValueError("Category names cannot be empty")
            if not fine_labels:
                raise ValueError(f"Category {category!r} has no fine labels")
            for fine_label in fine_labels:
                if not fine_label:
                    raise ValueError("Fine-label names cannot be empty")
                if fine_label in seen_fine:
                    raise ValueError(f"Duplicate fine label: {fine_label!r}")
                seen_fine.add(fine_label)

        for alias, target in self.aliases.items():
            if not alias:
                raise ValueError("Label aliases cannot be empty")
            if target not in seen_fine:
                raise ValueError(
                    f"Alias {alias!r} points to unknown fine label {target!r}"
                )

        canonical_by_key: dict[str, str] = {}
        for fine_label in seen_fine:
            key = label_key(fine_label)
            previous = canonical_by_key.get(key)
            if previous is not None and previous != fine_label:
                raise ValueError(
                    f"Fine labels {previous!r} and {fine_label!r} normalize "
                    "to the same key"
                )
            canonical_by_key[key] = fine_label

        for alias, target in self.aliases.items():
            key = label_key(alias)
            previous = canonical_by_key.get(key)
            if previous is not None and previous != target:
                raise ValueError(
                    f"Alias {alias!r} conflicts with canonical label {previous!r}"
                )
            canonical_by_key[key] = target

    @cached_property
    def category_names(self) -> tuple[str, ...]:
        return tuple(self.category_to_fine_labels)

    @cached_property
    def fine_label_names(self) -> tuple[str, ...]:
        return tuple(
            fine_label
            for fine_labels in self.category_to_fine_labels.values()
            for fine_label in fine_labels
        )

    @cached_property
    def fine_label_to_category(self) -> dict[str, str]:
        return {
            fine_label: category
            for category, fine_labels in self.category_to_fine_labels.items()
            for fine_label in fine_labels
        }

    @cached_property
    def category_to_num_fine(self) -> dict[str, int]:
        return {
            category: len(fine_labels)
            for category, fine_labels in self.category_to_fine_labels.items()
        }

    @cached_property
    def canonical_by_key(self) -> dict[str, str]:
        lookup = {
            label_key(fine_label): fine_label
            for fine_label in self.fine_label_names
        }
        lookup.update(
            {label_key(alias): target for alias, target in self.aliases.items()}
        )
        return lookup

    def canonical_fine_label(self, label: str) -> str:
        key = label_key(label)
        if key not in self.canonical_by_key:
            raise KeyError(
                f"Unknown fine label for schema {self.name!r}: {label!r}"
            )
        return self.canonical_by_key[key]

    def coarse_label_for_fine(self, fine_label: str) -> str:
        canonical = self.canonical_fine_label(fine_label)
        return self.fine_label_to_category[canonical]

    def category_index(self, category: str) -> int:
        return self.category_names.index(category)

    def fine_index_within_category(self, fine_label: str) -> int:
        canonical = self.canonical_fine_label(fine_label)
        category = self.fine_label_to_category[canonical]
        return self.category_to_fine_labels[category].index(canonical)

    def fine_label_from_category_index(
        self,
        category: str,
        fine_index: int,
    ) -> str:
        return self.category_to_fine_labels[category][fine_index]

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category_to_fine_labels": {
                category: list(fine_labels)
                for category, fine_labels in self.category_to_fine_labels.items()
            },
            "aliases": dict(self.aliases),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "LabelSchema":
        raw_hierarchy = payload.get("category_to_fine_labels")
        if not isinstance(raw_hierarchy, Mapping):
            raise ValueError(
                "Label schema must contain category_to_fine_labels"
            )

        hierarchy: dict[str, tuple[str, ...]] = {}
        for category, raw_fine_labels in raw_hierarchy.items():
            if not isinstance(raw_fine_labels, (list, tuple)):
                raise ValueError(
                    f"Fine labels for category {category!r} must be a list"
                )
            hierarchy[str(category)] = tuple(
                str(fine_label) for fine_label in raw_fine_labels
            )

        raw_aliases = payload.get("aliases", {})
        if not isinstance(raw_aliases, Mapping):
            raise ValueError("Label schema aliases must be an object")

        return cls(
            name=str(payload.get("name") or "unnamed-dataset"),
            category_to_fine_labels=hierarchy,
            aliases={
                str(alias): str(target)
                for alias, target in raw_aliases.items()
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabelSchema":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("Label schema JSON must contain an object")
        return cls.from_json(payload)

    @classmethod
    def from_flat_labels(
        cls,
        labels: Iterable[str],
        *,
        name: str,
    ) -> "LabelSchema":
        """Create a portable schema where every folder is one output class."""

        ordered_labels = tuple(sorted({str(label) for label in labels}))
        if not ordered_labels:
            raise ValueError("Cannot infer a schema without labels")
        return cls(
            name=name,
            category_to_fine_labels={
                label: (label,) for label in ordered_labels
            },
            aliases={},
        )


def label_schema_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    fallback: LabelSchema,
) -> LabelSchema:
    """Load a schema payload or reconstruct a compatible legacy schema."""

    payload = checkpoint.get("label_schema")
    if isinstance(payload, Mapping):
        return LabelSchema.from_json(payload)

    raw_counts = checkpoint.get("category_to_num_fine")
    raw_names = checkpoint.get("category_names")
    if not isinstance(raw_counts, Mapping) or not isinstance(
        raw_names,
        (list, tuple),
    ):
        return fallback

    categories = tuple(str(category) for category in raw_names)
    if any(category not in fallback.category_to_fine_labels for category in categories):
        return fallback

    hierarchy: dict[str, tuple[str, ...]] = {}
    for category in categories:
        requested_count = int(raw_counts.get(category, 0))
        available = fallback.category_to_fine_labels[category]
        if requested_count < 1 or requested_count > len(available):
            return fallback
        hierarchy[category] = available[:requested_count]

    retained_fine_labels = {
        fine_label
        for fine_labels in hierarchy.values()
        for fine_label in fine_labels
    }
    return LabelSchema(
        name=f"{fallback.name}-checkpoint",
        category_to_fine_labels=hierarchy,
        aliases={
            alias: target
            for alias, target in fallback.aliases.items()
            if target in retained_fine_labels
        },
    )
