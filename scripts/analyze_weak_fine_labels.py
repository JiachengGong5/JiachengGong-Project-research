#!/usr/bin/env python3
"""Summarize weak fine-label performance and likely error causes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.labels import (  # noqa: E402
    FINE_LABEL_NAMES,
    FINE_LABEL_TO_CATEGORY,
)
from activity_patterns.manifest import read_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/dev_chunk_manifest.csv")
    parser.add_argument(
        "--evaluation-dir",
        default="artifacts/evaluation_fine_weighted",
        help="Directory containing metrics.json, predictions.jsonl, and fine metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/error_analysis_fine_weighted",
    )
    parser.add_argument(
        "--min-val-support",
        type=int,
        default=5,
        help="Validation chunks below this threshold are treated as unstable.",
    )
    parser.add_argument(
        "--min-total-support",
        type=int,
        default=10,
        help="Total chunks below this threshold are treated as severe scarcity.",
    )
    parser.add_argument("--low-f1-threshold", type=float, default=0.6)
    parser.add_argument("--low-recall-threshold", type=float, default=0.6)
    return parser


def read_csv_by_label(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            label = row["label"]
            rows[label] = {
                "support": int(row["support"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
        return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_field(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def format_counter(counter: Counter[str], *, limit: int = 3) -> str:
    if not counter:
        return "-"
    return "; ".join(
        f"{label} ({count})"
        for label, count in counter.most_common(limit)
    )


def recommended_action(diagnosis: str) -> str:
    actions = {
        "severe_data_scarcity": (
            "Do not overclaim fixed-split fine accuracy; use repeated splits or "
            "leave-one-chunk-out and keep trace evidence qualitative."
        ),
        "no_validation_support": (
            "Exclude from fixed-split macro interpretation and evaluate with "
            "a rare-class protocol."
        ),
        "low_validation_support": (
            "Report as unstable; rerun repeated fine-stratified splits before "
            "making a class-level claim."
        ),
        "coarse_gate_confusion": (
            "Inspect coarse mistakes first; improve category separation before "
            "tuning the fine head."
        ),
        "fine_head_confusion": (
            "Inspect within-category confusions and salient Zeek traces; tune "
            "fine loss, sequence length, or per-class sampling."
        ),
        "low_recall": (
            "Prioritize false negatives and inspect top predicted alternatives."
        ),
        "low_precision": (
            "Check whether this label is being over-predicted for other classes."
        ),
        "acceptable": (
            "Keep current result; use extracted activity patterns as evidence."
        ),
    }
    return actions[diagnosis]


def diagnose_label(
    *,
    total_chunks: int,
    val_chunks: int,
    precision: float,
    recall: float,
    f1: float,
    path_recall: float,
    coarse_gate_errors: int,
    fine_head_errors: int,
    args: argparse.Namespace,
) -> str:
    if total_chunks < args.min_total_support:
        return "severe_data_scarcity"
    if val_chunks == 0:
        return "no_validation_support"
    if val_chunks < args.min_val_support:
        return "low_validation_support"
    if path_recall < args.low_recall_threshold:
        if coarse_gate_errors > fine_head_errors:
            return "coarse_gate_confusion"
        return "fine_head_confusion"
    if f1 < args.low_f1_threshold:
        if coarse_gate_errors > fine_head_errors:
            return "coarse_gate_confusion"
        return "fine_head_confusion"
    if recall < args.low_recall_threshold:
        return "low_recall"
    if precision < args.low_recall_threshold:
        return "low_precision"
    return "acceptable"


def build_manifest_counts(manifest_path: str | Path) -> dict[str, Counter[str]]:
    rows = read_manifest(manifest_path, project_root=PROJECT_ROOT)
    counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "total": Counter(),
    }
    for row in rows:
        counts["total"][row.fine_label] += 1
        if row.split in counts:
            counts[row.split][row.fine_label] += 1
    return counts


def build_prediction_stats(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "val_predictions": 0,
            "path_correct": 0,
            "path_errors": 0,
            "coarse_gate_errors": 0,
            "fine_head_errors": 0,
            "other_path_errors": 0,
            "predicted_coarse_on_errors": Counter(),
            "predicted_path_fine_on_errors": Counter(),
            "predicted_true_category_fine_on_errors": Counter(),
        }
    )

    for record in predictions:
        true_fine = record["true_fine"]
        label_stats = stats[true_fine]
        label_stats["val_predictions"] += 1
        if bool_field(record, "path_correct"):
            label_stats["path_correct"] += 1
            continue

        label_stats["path_errors"] += 1
        label_stats["predicted_coarse_on_errors"][record["pred_coarse"]] += 1
        label_stats["predicted_path_fine_on_errors"][record["pred_fine"]] += 1
        label_stats["predicted_true_category_fine_on_errors"][
            record["pred_fine_given_true_category"]
        ] += 1

        if not bool_field(record, "coarse_correct"):
            label_stats["coarse_gate_errors"] += 1
        elif not bool_field(record, "fine_correct_given_true_category"):
            label_stats["fine_head_errors"] += 1
        else:
            label_stats["other_path_errors"] += 1
    return stats


def severity_rank(diagnosis: str) -> int:
    order = {
        "severe_data_scarcity": 0,
        "no_validation_support": 1,
        "low_validation_support": 2,
        "coarse_gate_confusion": 3,
        "fine_head_confusion": 4,
        "low_recall": 5,
        "low_precision": 6,
        "acceptable": 7,
    }
    return order[diagnosis]


def build_analysis_rows(
    fine_metrics: dict[str, dict[str, Any]],
    manifest_counts: dict[str, Counter[str]],
    prediction_stats: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for label in FINE_LABEL_NAMES:
        metric = fine_metrics.get(
            label,
            {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        )
        stats = prediction_stats.get(label, {})
        val_predictions = int(stats.get("val_predictions", 0))
        path_correct = int(stats.get("path_correct", 0))
        path_recall = path_correct / val_predictions if val_predictions else 0.0
        diagnosis = diagnose_label(
            total_chunks=manifest_counts["total"][label],
            val_chunks=manifest_counts["val"][label],
            precision=metric["precision"],
            recall=metric["recall"],
            f1=metric["f1"],
            path_recall=path_recall,
            coarse_gate_errors=int(stats.get("coarse_gate_errors", 0)),
            fine_head_errors=int(stats.get("fine_head_errors", 0)),
            args=args,
        )
        rows.append(
            {
                "fine_label": label,
                "coarse_label": FINE_LABEL_TO_CATEGORY[label],
                "train_chunks": manifest_counts["train"][label],
                "val_chunks": manifest_counts["val"][label],
                "total_chunks": manifest_counts["total"][label],
                "metric_support": metric["support"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "path_correct": path_correct,
                "path_recall": path_recall,
                "path_errors": int(stats.get("path_errors", 0)),
                "coarse_gate_errors": int(stats.get("coarse_gate_errors", 0)),
                "fine_head_errors": int(stats.get("fine_head_errors", 0)),
                "other_path_errors": int(stats.get("other_path_errors", 0)),
                "top_predicted_coarse_on_errors": format_counter(
                    stats.get("predicted_coarse_on_errors", Counter())
                ),
                "top_predicted_path_fine_on_errors": format_counter(
                    stats.get("predicted_path_fine_on_errors", Counter())
                ),
                "top_predicted_true_category_fine_on_errors": format_counter(
                    stats.get("predicted_true_category_fine_on_errors", Counter())
                ),
                "diagnosis": diagnosis,
                "recommended_action": recommended_action(diagnosis),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row[field]).replace("|", "/") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def write_markdown(
    path: Path,
    *,
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    total_chunks = sum(row["total_chunks"] for row in rows)
    train_chunks = sum(row["train_chunks"] for row in rows)
    val_chunks = sum(row["val_chunks"] for row in rows)
    weak_rows = [row for row in rows if row["diagnosis"] != "acceptable"]
    weak_rows = sorted(
        weak_rows,
        key=lambda row: (
            severity_rank(row["diagnosis"]),
            row["val_chunks"],
            row["path_recall"],
            row["f1"],
            row["fine_label"],
        ),
    )
    ready_rows = [
        row
        for row in rows
        if row["diagnosis"] == "acceptable"
        and row["val_chunks"] >= args.min_val_support
        and row["f1"] >= 0.7
        and row["path_recall"] >= 0.7
    ]
    low_f1_with_support = [
        row
        for row in rows
        if row["val_chunks"] >= args.min_val_support
        and row["f1"] < args.low_f1_threshold
    ]
    low_path_with_support = [
        row
        for row in rows
        if row["val_chunks"] >= args.min_val_support
        and row["path_recall"] < args.low_recall_threshold
    ]
    scarce_rows = [
        row for row in rows if row["total_chunks"] < args.min_total_support
    ]
    low_val_rows = [
        row for row in rows if row["val_chunks"] < args.min_val_support
    ]

    total_path_errors = sum(row["path_errors"] for row in rows)
    coarse_gate_errors = sum(row["coarse_gate_errors"] for row in rows)
    fine_head_errors = sum(row["fine_head_errors"] for row in rows)
    other_errors = sum(row["other_path_errors"] for row in rows)

    lines = [
        "# Fine-Label Error Analysis",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Evaluation directory: `{args.evaluation_dir}`",
        f"- Low validation support threshold: `{args.min_val_support}` chunks",
        f"- Severe total support threshold: `{args.min_total_support}` chunks",
        f"- Low F1 threshold: `{args.low_f1_threshold}`",
        "",
        "## Headline Metrics",
        "",
        f"- Validation samples: `{metrics['total_samples']}`",
        f"- Coarse accuracy: `{format_metric(metrics['coarse']['accuracy'])}`",
        f"- Coarse macro F1: `{format_metric(metrics['coarse']['macro_f1'])}`",
        "- Fine macro F1 given true category: "
        f"`{format_metric(metrics['fine_given_true_category']['macro_f1'])}`",
        f"- Full path accuracy: `{format_metric(metrics['path_accuracy'])}`",
        "",
        "## Data Sufficiency",
        "",
        f"- Fine labels included in the pipeline: `{len(rows)}`",
        f"- Chunk rows: `{total_chunks}` total, `{train_chunks}` train, `{val_chunks}` val",
        f"- Severe-scarcity fine labels: `{len(scarce_rows)}`",
        f"- Low-validation-support fine labels: `{len(low_val_rows)}`",
        f"- Low-F1 labels with enough validation support: `{len(low_f1_with_support)}`",
        "- Low full-path-recall labels with enough validation support: "
        f"`{len(low_path_with_support)}`",
        "",
        "## Error Mechanism",
        "",
        f"- Full path errors: `{total_path_errors}`",
        f"- Coarse gate errors: `{coarse_gate_errors}`",
        f"- Fine-head errors after correct coarse category: `{fine_head_errors}`",
        f"- Other path errors: `{other_errors}`",
        "",
        "Interpretation: a coarse gate error means the model selected the wrong "
        "top-level category first, so the final fine label also becomes wrong. "
        "A fine-head error means the coarse category was correct, but the "
        "category-specific fine classifier confused sibling attacks.",
        "",
        "## Weak Or Unstable Fine Labels",
        "",
    ]

    weak_table_rows = []
    for row in weak_rows:
        weak_table_rows.append(
            {
                "fine_label": row["fine_label"],
                    "chunks": f"{row['train_chunks']}/{row['val_chunks']}",
                    "f1": format_metric(row["f1"]),
                    "path_recall": format_metric(row["path_recall"]),
                    "diagnosis": row["diagnosis"],
                    "top_error": row["top_predicted_path_fine_on_errors"],
                }
        )
    lines.extend(
        markdown_table(
            weak_table_rows,
            ["fine_label", "chunks", "f1", "path_recall", "diagnosis", "top_error"],
        )
        if weak_table_rows
        else ["No weak labels under the configured thresholds."]
    )

    lines.extend(
        [
            "",
            "## Stable Fine Labels For Pattern Evidence",
            "",
        ]
    )
    if ready_rows:
        lines.extend(
            markdown_table(
                [
                    {
                        "fine_label": row["fine_label"],
                        "val_chunks": row["val_chunks"],
                        "f1": format_metric(row["f1"]),
                        "path_recall": format_metric(row["path_recall"]),
                    }
                    for row in ready_rows
                ],
                ["fine_label", "val_chunks", "f1", "path_recall"],
            )
        )
    else:
        lines.append("No label met the stable-evidence threshold.")

    lines.extend(
        [
            "",
            "## Recommended Next Experiment",
            "",
            "1. Keep the current result as the first complete 33-fine-label "
            "hierarchical run. It proves the end-to-end Zeek-to-LSTM pipeline "
            "works across the full local CICIoT2023 hierarchy.",
            "2. Do not treat one fixed split as the final authority for labels "
            "with fewer than five validation chunks. Use repeated fine-stratified "
            "splits or leave-one-chunk-out for those rare labels.",
            "3. For labels with enough validation support but low F1, focus on "
            "error mechanism: DDoS/Recon sibling confusions need fine-head "
            "analysis; Web-based failures are mainly data scarcity plus coarse "
            "gate confusion.",
            "4. Continue using activity-pattern extraction and Zeek trace "
            "reconstruction for class-level evidence, especially for the stable "
            "labels listed above.",
            "",
            "## Full Recommendations",
            "",
        ]
    )
    recommendation_rows = [
        {
            "fine_label": row["fine_label"],
            "diagnosis": row["diagnosis"],
            "recommended_action": row["recommended_action"],
        }
        for row in weak_rows
    ]
    lines.extend(
        markdown_table(
            recommendation_rows,
            ["fine_label", "diagnosis", "recommended_action"],
        )
        if recommendation_rows
        else ["No special recommendations."]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    fine_metrics = read_csv_by_label(evaluation_dir / "fine_per_class_metrics.csv")
    metrics = read_json(evaluation_dir / "metrics.json")
    predictions = read_jsonl(evaluation_dir / "predictions.jsonl")
    manifest_counts = build_manifest_counts(args.manifest)
    prediction_stats = build_prediction_stats(predictions)
    rows = build_analysis_rows(
        fine_metrics,
        manifest_counts,
        prediction_stats,
        args,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weak_fine_labels.csv"
    markdown_path = output_dir / "weak_fine_labels.md"
    write_csv(csv_path, rows)
    write_markdown(markdown_path, metrics=metrics, rows=rows, args=args)

    weak_count = sum(1 for row in rows if row["diagnosis"] != "acceptable")
    print(f"fine_labels={len(rows)}")
    print(f"weak_or_unstable_labels={weak_count}")
    print(f"output_csv={csv_path}")
    print(f"output_markdown={markdown_path}")


if __name__ == "__main__":
    main()
