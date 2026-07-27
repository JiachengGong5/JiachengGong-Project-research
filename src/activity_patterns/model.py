"""LSTM model that learns directly from categorical protocol-event fields."""

from __future__ import annotations

from collections.abc import Mapping

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError as exc:  # pragma: no cover - depends on optional package
    raise ImportError(
        "PyTorch is required for activity_patterns.model. "
        "Install the project with the 'model' optional dependency."
    ) from exc


class ProtocolSequenceEncoder(nn.Module):
    """Encode protocol-event sequences without precomputed traffic statistics.

    token_ids has shape [batch, time, fields]. Zero is the padding token.
    lengths contains the number of real events in each sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * directions
        self.bidirectional = bidirectional

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        # Summing learned field embeddings is a neural representation layer,
        # not a manually calculated traffic statistic.
        event_embeddings = self.embedding(token_ids).sum(dim=2)
        packed = pack_padded_sequence(
            event_embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            return torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden[-1]


class ProtocolEventLSTM(nn.Module):
    """Single-head classifier for binary, 8-class, or flat fine labels."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ProtocolSequenceEncoder(
            vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, num_classes),
        )

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        return self.classifier(self.encoder(token_ids, lengths))


class HierarchicalProtocolEventLSTM(nn.Module):
    """Coarse category classifier plus category-specific fine classifiers."""

    def __init__(
        self,
        vocab_size: int,
        category_to_num_fine: Mapping[str, int],
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.category_names = tuple(category_to_num_fine)
        self.encoder = ProtocolSequenceEncoder(
            vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.coarse_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, len(self.category_names)),
        )
        module_dict_attributes = set(dir(nn.ModuleDict()))
        self._fine_classifier_keys = {}
        for index, category in enumerate(self.category_names):
            key_is_safe = (
                bool(category)
                and "." not in category
                and category not in module_dict_attributes
            )
            self._fine_classifier_keys[category] = (
                category if key_is_safe else f"category_{index}"
            )
        self.fine_classifiers = nn.ModuleDict(
            {
                self._fine_classifier_keys[category]: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(self.encoder.output_dim, num_fine),
                )
                for category, num_fine in category_to_num_fine.items()
            }
        )

    def forward(self, token_ids: Tensor, lengths: Tensor) -> dict[str, object]:
        sequence_embedding = self.encoder(token_ids, lengths)
        return {
            "coarse": self.coarse_classifier(sequence_embedding),
            "fine": {
                category: self.fine_classifiers[
                    self._fine_classifier_keys[category]
                ](sequence_embedding)
                for category in self.category_names
            },
        }


def hierarchical_loss(
    outputs: Mapping[str, object],
    coarse_targets: Tensor,
    fine_targets: Tensor,
    category_names: tuple[str, ...],
    *,
    coarse_weight: float = 1.0,
    fine_weight: float = 1.0,
    fine_loss_reduction: str = "sample_mean",
    coarse_class_weights: Tensor | None = None,
    fine_class_weights_by_category: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Cross-entropy for the coarse head plus the matching fine head.

    ``fine_targets`` must contain indices local to the true category's fine
    classifier. For example, `Recon-Port scan` is indexed within the Recon
    head, not within a global 34-class list.
    """

    coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)
    coarse_logits = outputs["coarse"]
    fine_logits_by_category = outputs["fine"]
    if not isinstance(coarse_logits, Tensor):
        raise TypeError("outputs['coarse'] must be a tensor")
    if not isinstance(fine_logits_by_category, Mapping):
        raise TypeError("outputs['fine'] must be a mapping")

    if coarse_weight < 0.0 or fine_weight < 0.0:
        raise ValueError("coarse_weight and fine_weight must be non-negative")
    if coarse_weight == 0.0 and fine_weight == 0.0:
        raise ValueError("At least one hierarchical loss weight must be positive")
    if fine_loss_reduction not in {"sample_mean", "category_sum"}:
        raise ValueError(
            "fine_loss_reduction must be 'sample_mean' or 'category_sum'"
        )

    loss = coarse_criterion(coarse_logits, coarse_targets) * coarse_weight
    if fine_weight == 0.0:
        return loss

    fine_loss_numerator = coarse_logits.new_zeros(())
    fine_loss_denominator = coarse_logits.new_zeros(())
    for category_index, category in enumerate(category_names):
        mask = coarse_targets == category_index
        if torch.any(mask):
            fine_logits = fine_logits_by_category[category]
            if not isinstance(fine_logits, Tensor):
                raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
            fine_class_weights = None
            if fine_class_weights_by_category is not None:
                fine_class_weights = fine_class_weights_by_category.get(category)
            category_targets = fine_targets[mask]
            if fine_loss_reduction == "category_sum":
                fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
                loss = loss + fine_criterion(
                    fine_logits[mask], category_targets
                ) * fine_weight
                continue

            category_losses = nn.functional.cross_entropy(
                fine_logits[mask],
                category_targets,
                weight=fine_class_weights,
                reduction="none",
            )
            fine_loss_numerator = fine_loss_numerator + category_losses.sum()
            if fine_class_weights is None:
                fine_loss_denominator = (
                    fine_loss_denominator + category_targets.numel()
                )
            else:
                fine_loss_denominator = (
                    fine_loss_denominator
                    + fine_class_weights[category_targets].sum()
                )

    if fine_loss_reduction == "sample_mean":
        loss = loss + fine_weight * (
            fine_loss_numerator / fine_loss_denominator.clamp_min(1e-12)
        )
    return loss
