#!/usr/bin/env python3
"""Evaluate a trained hierarchical LSTM and write analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from activity_patterns.dataset import (  # noqa: E402
    SequenceChunkDataset,
    collate_sequence_chunks,
)
from activity_patterns.labels import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_NUM_FINE,
    FINE_LABEL_NAMES,
    FINE_LABEL_TO_CATEGORY,
    fine_label_from_category_index,
)
from activity_patterns.manifest import read_manifest  # noqa: E402
from activity_patterns.model import HierarchicalProtocolEventLSTM  # noqa: E402
from activity_patterns.vocab import TokenVocab  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/dev_chunk_manifest.csv")
    parser.add_argument("--vocab", default="artifacts/dev_vocab.json")
    parser.add_argument("--checkpoint", default="runs/hierarchical_dev/best_model.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--output-dir", default="artifacts/evaluation_dev")
    return parser


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_load(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(
    checkpoint_path: str | Path,
    vocab_size: int,
    device: torch.device,
) -> HierarchicalProtocolEventLSTM:
    checkpoint = torch_load(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {})
    model = HierarchicalProtocolEventLSTM(
        vocab_size,
        CATEGORY_TO_NUM_FINE,
        embedding_dim=int(checkpoint_args.get("embedding_dim", 32)),
        hidden_dim=int(checkpoint_args.get("hidden_dim", 64)),
        num_layers=int(checkpoint_args.get("num_layers", 1)),
        dropout=float(checkpoint_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = dict(batch)
    for key in ("token_ids", "lengths", "coarse_targets", "fine_targets"):
        value = moved[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field {key!r} must be a tensor")
        moved[key] = value.to(device)
    return moved


def fine_prediction(
    outputs: dict[str, object],
    coarse_prediction: int,
) -> tuple[str, int]:
    category = CATEGORY_NAMES[coarse_prediction]
    fine_logits_by_category = outputs["fine"]
    if not isinstance(fine_logits_by_category, dict):
        raise TypeError("outputs['fine'] must be a dict")
    logits = fine_logits_by_category[category]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
    return category, int(torch.argmax(logits[0]).detach().cpu())


def category_fine_prediction(
    outputs: dict[str, object],
    category: str,
) -> tuple[int, str]:
    fine_logits_by_category = outputs["fine"]
    if not isinstance(fine_logits_by_category, dict):
        raise TypeError("outputs['fine'] must be a dict")
    logits = fine_logits_by_category[category]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
    index = int(torch.argmax(logits[0]).detach().cpu())
    return index, fine_label_from_category_index(category, index)


def build_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def per_class_metrics(
    confusion: list[list[int]],
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(labels):
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


def aggregate_metrics(confusion: list[list[int]], labels: tuple[str, ...]) -> dict[str, float]:
    class_rows = [
        row for row in per_class_metrics(confusion, labels) if row["support"] > 0
    ]
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(row["f1"] for row in class_rows) / max(len(class_rows), 1),
        "balanced_accuracy": (
            sum(row["recall"] for row in class_rows) / max(len(class_rows), 1)
        ),
    }


@torch.no_grad()
def evaluate_dataset(
    model: HierarchicalProtocolEventLSTM,
    dataset: SequenceChunkDataset,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    coarse_targets = []
    coarse_predicted = []
    fine_targets = []
    fine_predicted_for_true_category = []
    path_correct = 0

    for index in range(len(dataset)):
        item = dataset[index]
        row = dataset.rows[dataset.refs[index].row_index]
        batch = move_batch(collate_sequence_chunks([item]), device)
        outputs = model(batch["token_ids"], batch["lengths"])
        coarse_logits = outputs["coarse"]
        if not isinstance(coarse_logits, torch.Tensor):
            raise TypeError("outputs['coarse'] must be a tensor")
        coarse_probs = torch.softmax(coarse_logits[0], dim=0)
        pred_index = int(torch.argmax(coarse_probs).detach().cpu())
        true_index = int(item["coarse_target"])
        pred_category, pred_fine_index = fine_prediction(outputs, pred_index)
        pred_fine_label = fine_label_from_category_index(pred_category, pred_fine_index)
        true_category = item["coarse_label"]
        true_category_fine_index, true_category_fine_label = category_fine_prediction(
            outputs,
            true_category,
        )
        fine_target_index = FINE_LABEL_NAMES.index(item["fine_label"])
        fine_pred_index = FINE_LABEL_NAMES.index(true_category_fine_label)
        coarse_correct = pred_index == true_index
        fine_correct_given_true_category = true_category_fine_label == item["fine_label"]
        path_is_correct = coarse_correct and pred_fine_label == item["fine_label"]
        path_correct += int(path_is_correct)

        record = {
            "sequence_path": row.sequence_path.as_posix(),
            "sequence_id": item["sequence_id"],
            "chunk_index": item["chunk_index"],
            "true_coarse": item["coarse_label"],
            "true_fine": item["fine_label"],
            "pred_coarse": pred_category,
            "pred_fine_index": pred_fine_index,
            "pred_fine": pred_fine_label,
            "pred_fine_given_true_category": true_category_fine_label,
            "pred_fine_given_true_category_index": true_category_fine_index,
            "pred_coarse_confidence": float(coarse_probs[pred_index].detach().cpu()),
            "coarse_correct": coarse_correct,
            "fine_correct_given_true_category": fine_correct_given_true_category,
            "path_correct": path_is_correct,
            "correct": path_is_correct,
        }
        predictions.append(record)
        coarse_targets.append(true_index)
        coarse_predicted.append(pred_index)
        fine_targets.append(fine_target_index)
        fine_predicted_for_true_category.append(fine_pred_index)

    coarse_confusion = build_confusion_matrix(
        coarse_targets,
        coarse_predicted,
        len(CATEGORY_NAMES),
    )
    fine_confusion = build_confusion_matrix(
        fine_targets,
        fine_predicted_for_true_category,
        len(FINE_LABEL_NAMES),
    )
    metrics = {
        "coarse": aggregate_metrics(coarse_confusion, CATEGORY_NAMES),
        "fine_given_true_category": aggregate_metrics(fine_confusion, FINE_LABEL_NAMES),
        "path_accuracy": path_correct / max(len(dataset), 1),
    }
    metrics["coarse_confusion_matrix"] = coarse_confusion
    metrics["fine_confusion_matrix"] = fine_confusion
    metrics["coarse_per_class"] = per_class_metrics(coarse_confusion, CATEGORY_NAMES)
    metrics["fine_per_class"] = per_class_metrics(fine_confusion, FINE_LABEL_NAMES)
    metrics["total_samples"] = len(dataset)
    return predictions, metrics


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "support", "precision", "recall", "f1"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(
    path: Path,
    confusion: list[list[int]],
    labels: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, confusion):
            writer.writerow([label, *row])


def main() -> None:
    args = build_parser().parse_args()
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
        raise ValueError(f"No chunks found for split {args.split!r}")

    model = load_model(args.checkpoint, len(vocab), device)
    predictions, metrics = evaluate_dataset(model, dataset, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = [record for record in predictions if not record["path_correct"]]

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "errors.jsonl", errors)
    write_per_class_csv(
        output_dir / "coarse_per_class_metrics.csv",
        metrics["coarse_per_class"],
    )
    write_per_class_csv(
        output_dir / "fine_per_class_metrics.csv",
        metrics["fine_per_class"],
    )
    write_confusion_csv(
        output_dir / "coarse_confusion_matrix.csv",
        metrics["coarse_confusion_matrix"],
        CATEGORY_NAMES,
    )
    write_confusion_csv(
        output_dir / "fine_confusion_matrix.csv",
        metrics["fine_confusion_matrix"],
        FINE_LABEL_NAMES,
    )

    print(f"evaluated_samples={len(dataset)}")
    print(f"errors={len(errors)}")
    print(f"coarse_accuracy={metrics['coarse']['accuracy']:.3f}")
    print(f"coarse_macro_f1={metrics['coarse']['macro_f1']:.3f}")
    print(
        "fine_true_category_macro_f1="
        f"{metrics['fine_given_true_category']['macro_f1']:.3f}"
    )
    print(f"path_accuracy={metrics['path_accuracy']:.3f}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
