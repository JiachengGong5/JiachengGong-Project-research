#!/usr/bin/env python3
"""Train the hierarchical protocol-event LSTM with validation metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from activity_patterns.dataset import (  # noqa: E402
    SequenceChunkDataset,
    collate_sequence_chunks,
)
from activity_patterns.labels import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_FINE_LABELS,
    CATEGORY_TO_NUM_FINE,
    category_index,
    fine_index_within_category,
)
from activity_patterns.manifest import read_manifest  # noqa: E402
from activity_patterns.model import (  # noqa: E402
    HierarchicalProtocolEventLSTM,
    hierarchical_loss,
)
from activity_patterns.vocab import TokenVocab  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/dev_chunk_manifest.csv")
    parser.add_argument("--vocab", default="artifacts/dev_vocab.json")
    parser.add_argument("--output-dir", default="runs/hierarchical_dev")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Use inverse-frequency weights for the coarse classification loss",
    )
    parser.add_argument(
        "--fine-class-weighting",
        choices=("balanced", "none"),
        default="balanced",
        help="Use inverse-frequency weights inside each category-specific fine head",
    )
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


def coarse_counts(dataset: SequenceChunkDataset) -> list[int]:
    counts = [0 for _ in CATEGORY_NAMES]
    for ref in dataset.refs:
        row = dataset.rows[ref.row_index]
        counts[category_index(row.coarse_label)] += 1
    return counts


def fine_counts_by_category(dataset: SequenceChunkDataset) -> dict[str, list[int]]:
    counts = {
        category: [0 for _ in fine_labels]
        for category, fine_labels in CATEGORY_TO_FINE_LABELS.items()
    }
    for ref in dataset.refs:
        row = dataset.rows[ref.row_index]
        counts[row.coarse_label][fine_index_within_category(row.fine_label)] += 1
    return counts


def balanced_weights(counts: list[int], device: torch.device) -> torch.Tensor:
    present_counts = [count for count in counts if count > 0]
    if not present_counts:
        raise ValueError("Cannot build class weights for an empty dataset")

    total = sum(present_counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (len(present_counts) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def balanced_fine_weights_by_category(
    counts_by_category: dict[str, list[int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        category: balanced_weights(counts, device)
        for category, counts in counts_by_category.items()
    }


def predicted_fine_indices(
    outputs: dict[str, object],
    coarse_predictions: torch.Tensor,
) -> torch.Tensor:
    fine_logits_by_category = outputs["fine"]
    if not isinstance(fine_logits_by_category, dict):
        raise TypeError("outputs['fine'] must be a dict")

    predictions = torch.zeros_like(coarse_predictions)
    for category_index_value, category in enumerate(CATEGORY_NAMES):
        mask = coarse_predictions == category_index_value
        if torch.any(mask):
            logits = fine_logits_by_category[category]
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
            predictions[mask] = torch.argmax(logits[mask], dim=1)
    return predictions


def build_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def macro_metrics(confusion: list[list[int]]) -> dict[str, float]:
    f1_scores = []
    recalls = []
    for index, row in enumerate(confusion):
        true_positive = row[index]
        support = sum(row)
        predicted = sum(confusion[row_index][index] for row_index in range(len(row)))
        if support == 0:
            continue
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_scores.append(f1)
        recalls.append(recall)
    return {
        "macro_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
    }


def per_class_metrics(confusion: list[list[int]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for index, label in enumerate(CATEGORY_NAMES):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(confusion[row_index][index] for row_index in range(len(confusion)))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def run_epoch(
    model: HierarchicalProtocolEventLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    coarse_class_weights: torch.Tensor | None,
    fine_class_weights_by_category: dict[str, torch.Tensor] | None,
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
            coarse_class_weights=coarse_class_weights,
            fine_class_weights_by_category=fine_class_weights_by_category,
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
    coarse_class_weights: torch.Tensor | None,
    fine_class_weights_by_category: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    coarse_correct = 0
    path_correct = 0
    coarse_targets_all: list[int] = []
    coarse_predictions_all: list[int] = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["token_ids"], batch["lengths"])
        loss = hierarchical_loss(
            outputs,
            batch["coarse_targets"],
            batch["fine_targets"],
            CATEGORY_NAMES,
            coarse_class_weights=coarse_class_weights,
            fine_class_weights_by_category=fine_class_weights_by_category,
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
        coarse_targets_all.extend(batch["coarse_targets"].detach().cpu().tolist())
        coarse_predictions_all.extend(coarse_predictions.detach().cpu().tolist())

    confusion = build_confusion_matrix(
        coarse_targets_all,
        coarse_predictions_all,
        len(CATEGORY_NAMES),
    )
    metrics = macro_metrics(confusion)
    metrics.update(
        {
            "loss": total_loss / max(total, 1),
            "coarse_accuracy": coarse_correct / max(total, 1),
            "path_accuracy": path_correct / max(total, 1),
            "confusion_matrix": confusion,
            "per_class": per_class_metrics(confusion),
        }
    )
    return metrics


def print_counts(name: str, counts: list[int]) -> None:
    rendered = ", ".join(
        f"{label}={count}" for label, count in zip(CATEGORY_NAMES, counts)
    )
    print(f"{name}_counts={rendered}")


def print_fine_counts(name: str, counts_by_category: dict[str, list[int]]) -> None:
    for category in CATEGORY_NAMES:
        rendered = ", ".join(
            f"{fine_label}={count}"
            for fine_label, count in zip(
                CATEGORY_TO_FINE_LABELS[category],
                counts_by_category[category],
            )
        )
        print(f"{name}_{category}_fine_counts={rendered}")


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = choose_device()

    rows = read_manifest(args.manifest, project_root=PROJECT_ROOT)
    vocab = TokenVocab.load(args.vocab)
    train_dataset = SequenceChunkDataset(
        rows,
        vocab,
        split=args.train_split,
        max_events=args.max_events,
    )
    val_dataset = SequenceChunkDataset(
        rows,
        vocab,
        split=args.val_split,
        max_events=args.max_events,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"No chunks found for train split {args.train_split!r}")
    if len(val_dataset) == 0:
        raise ValueError(f"No chunks found for val split {args.val_split!r}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequence_chunks,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_chunks,
    )

    train_counts = coarse_counts(train_dataset)
    val_counts = coarse_counts(val_dataset)
    train_fine_counts = fine_counts_by_category(train_dataset)
    val_fine_counts = fine_counts_by_category(val_dataset)
    coarse_class_weights = None
    if args.class_weighting == "balanced":
        coarse_class_weights = balanced_weights(train_counts, device)
    fine_class_weights_by_category = None
    if args.fine_class_weighting == "balanced":
        fine_class_weights_by_category = balanced_fine_weights_by_category(
            train_fine_counts,
            device,
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
    print(
        f"train_chunks={len(train_dataset)} val_chunks={len(val_dataset)} "
        f"vocab={len(vocab)} max_events={args.max_events}"
    )
    print_counts("train", train_counts)
    print_counts("val", val_counts)
    print_fine_counts("train", train_fine_counts)
    print_fine_counts("val", val_fine_counts)
    if coarse_class_weights is not None:
        weights_for_display = coarse_class_weights.detach().cpu().tolist()
        rendered_weights = ", ".join(
            f"{label}={float(weight):.3f}"
            for label, weight in zip(CATEGORY_NAMES, weights_for_display)
        )
        print(f"coarse_class_weights={rendered_weights}")
    if fine_class_weights_by_category is not None:
        for category in CATEGORY_NAMES:
            weights_for_display = (
                fine_class_weights_by_category[category].detach().cpu().tolist()
            )
            rendered_weights = ", ".join(
                f"{fine_label}={float(weight):.3f}"
                for fine_label, weight in zip(
                    CATEGORY_TO_FINE_LABELS[category],
                    weights_for_display,
                )
            )
            print(f"{category}_fine_class_weights={rendered_weights}")

    best_macro_f1 = -1.0
    best_state: dict[str, Any] | None = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            coarse_class_weights,
            fine_class_weights_by_category,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            coarse_class_weights,
            fine_class_weights_by_category,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "vocab_size": len(vocab),
                "category_names": CATEGORY_NAMES,
                "category_to_num_fine": CATEGORY_TO_NUM_FINE,
                "args": vars(args),
                "train_counts": dict(zip(CATEGORY_NAMES, train_counts)),
                "val_counts": dict(zip(CATEGORY_NAMES, val_counts)),
                "train_fine_counts": train_fine_counts,
                "val_fine_counts": val_fine_counts,
                "class_weighting": args.class_weighting,
                "fine_class_weighting": args.fine_class_weighting,
                "coarse_class_weights": (
                    None
                    if coarse_class_weights is None
                    else coarse_class_weights.detach().cpu().tolist()
                ),
                "fine_class_weights_by_category": (
                    None
                    if fine_class_weights_by_category is None
                    else {
                        category: weights.detach().cpu().tolist()
                        for category, weights in fine_class_weights_by_category.items()
                    }
                ),
                "val_metrics": val_metrics,
            }

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_path_acc={train_metrics['path_accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_path_acc={val_metrics['path_accuracy']:.3f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(best_state, checkpoint_path)
    best_metrics = {
        key: value
        for key, value in best_state.items()
        if key != "model_state_dict"
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best": best_metrics}, handle, indent=2)
        handle.write("\n")

    elapsed = time.time() - start
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_metrics={metrics_path}")
    print(f"best_val_macro_f1={best_macro_f1:.3f}")
    print(f"elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
