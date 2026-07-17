#!/usr/bin/env python3
"""Run a small hierarchical LSTM training smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from activity_patterns.dataset import (  # noqa: E402
    SequenceChunkDataset,
    collate_sequence_chunks,
)
from activity_patterns.labels import CATEGORY_NAMES, CATEGORY_TO_NUM_FINE  # noqa: E402
from activity_patterns.manifest import read_manifest  # noqa: E402
from activity_patterns.model import (  # noqa: E402
    HierarchicalProtocolEventLSTM,
    hierarchical_loss,
)
from activity_patterns.vocab import TokenVocab  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/smoke_manifest.csv")
    parser.add_argument("--vocab", default="artifacts/smoke_vocab.json")
    parser.add_argument("--output-dir", default="runs/smoke")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = dict(batch)
    for key in ("token_ids", "lengths", "coarse_targets", "fine_targets"):
        value = moved[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a tensor")
        moved[key] = value.to(device)
    return moved


def predicted_fine_indices(
    outputs: dict[str, object],
    coarse_predictions: torch.Tensor,
) -> torch.Tensor:
    fine_logits_by_category = outputs["fine"]
    if not isinstance(fine_logits_by_category, dict):
        raise TypeError("outputs['fine'] must be a dict")

    predictions = torch.zeros_like(coarse_predictions)
    for category_index, category in enumerate(CATEGORY_NAMES):
        mask = coarse_predictions == category_index
        if torch.any(mask):
            logits = fine_logits_by_category[category]
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
            predictions[mask] = torch.argmax(logits[mask], dim=1)
    return predictions


def run_epoch(
    model: HierarchicalProtocolEventLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    coarse_correct = 0
    path_correct = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["token_ids"], batch["lengths"])
        loss = hierarchical_loss(
            outputs,
            batch["coarse_targets"],
            batch["fine_targets"],
            CATEGORY_NAMES,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        coarse_logits = outputs["coarse"]
        if not isinstance(coarse_logits, torch.Tensor):
            raise TypeError("outputs['coarse'] must be a tensor")

        batch_size = int(batch["coarse_targets"].shape[0])
        coarse_predictions = torch.argmax(coarse_logits, dim=1)
        fine_predictions = predicted_fine_indices(outputs, coarse_predictions)
        coarse_matches = coarse_predictions == batch["coarse_targets"]
        fine_matches = fine_predictions == batch["fine_targets"]

        total_loss += float(loss.detach()) * batch_size
        total += batch_size
        coarse_correct += int(torch.sum(coarse_matches).detach().cpu())
        path_correct += int(torch.sum(coarse_matches & fine_matches).detach().cpu())

    return {
        "loss": total_loss / max(total, 1),
        "coarse_accuracy": coarse_correct / max(total, 1),
        "path_accuracy": path_correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: HierarchicalProtocolEventLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    coarse_correct = 0
    path_correct = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["token_ids"], batch["lengths"])
        loss = hierarchical_loss(
            outputs,
            batch["coarse_targets"],
            batch["fine_targets"],
            CATEGORY_NAMES,
        )
        coarse_logits = outputs["coarse"]
        if not isinstance(coarse_logits, torch.Tensor):
            raise TypeError("outputs['coarse'] must be a tensor")

        batch_size = int(batch["coarse_targets"].shape[0])
        coarse_predictions = torch.argmax(coarse_logits, dim=1)
        fine_predictions = predicted_fine_indices(outputs, coarse_predictions)
        coarse_matches = coarse_predictions == batch["coarse_targets"]
        fine_matches = fine_predictions == batch["fine_targets"]

        total_loss += float(loss.detach()) * batch_size
        total += batch_size
        coarse_correct += int(torch.sum(coarse_matches).detach().cpu())
        path_correct += int(torch.sum(coarse_matches & fine_matches).detach().cpu())

    return {
        "loss": total_loss / max(total, 1),
        "coarse_accuracy": coarse_correct / max(total, 1),
        "path_accuracy": path_correct / max(total, 1),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = choose_device()

    rows = read_manifest(args.manifest, project_root=PROJECT_ROOT)
    vocab = TokenVocab.load(args.vocab)
    dataset = SequenceChunkDataset(
        rows,
        vocab,
        split=args.split,
        max_events=args.max_events,
    )
    if len(dataset) == 0:
        raise ValueError(f"No sequence chunks found for split {args.split!r}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequence_chunks,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_chunks,
    )

    model = HierarchicalProtocolEventLSTM(
        len(vocab),
        CATEGORY_TO_NUM_FINE,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"device={device}")
    print(f"chunks={len(dataset)} vocab={len(vocab)} max_events={args.max_events}")
    print(
        "model="
        f"embedding_dim={args.embedding_dim} hidden_dim={args.hidden_dim} "
        f"num_layers={args.num_layers}"
    )

    best_path_accuracy = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loader, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)
        if eval_metrics["path_accuracy"] > best_path_accuracy:
            best_path_accuracy = eval_metrics["path_accuracy"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "category_names": CATEGORY_NAMES,
                "category_to_num_fine": CATEGORY_TO_NUM_FINE,
                "args": vars(args),
                "eval_metrics": eval_metrics,
            }

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_coarse_acc={train_metrics['coarse_accuracy']:.3f} "
            f"train_path_acc={train_metrics['path_accuracy']:.3f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_coarse_acc={eval_metrics['coarse_accuracy']:.3f} "
            f"eval_path_acc={eval_metrics['path_accuracy']:.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    elapsed = time.time() - start
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
