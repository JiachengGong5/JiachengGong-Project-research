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
        self.fine_classifiers = nn.ModuleDict(
            {
                category: nn.Sequential(
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
                category: classifier(sequence_embedding)
                for category, classifier in self.fine_classifiers.items()
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

    loss = coarse_criterion(coarse_logits, coarse_targets) * coarse_weight
    for category_index, category in enumerate(category_names):
        mask = coarse_targets == category_index
        if torch.any(mask):
            fine_logits = fine_logits_by_category[category]
            if not isinstance(fine_logits, Tensor):
                raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
            fine_class_weights = None
            if fine_class_weights_by_category is not None:
                fine_class_weights = fine_class_weights_by_category.get(category)
            fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
            loss = (
                loss
                + fine_criterion(fine_logits[mask], fine_targets[mask]) * fine_weight
            )
    return loss
