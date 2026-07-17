#!/usr/bin/env python3
"""Run repeated fine-stratified development splits for stability analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="7,11,23",
        help="Comma-separated split/training seeds.",
    )
    parser.add_argument("--sequence-root", default="data/sequences")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--class-weighting", choices=("balanced", "none"), default="balanced")
    parser.add_argument(
        "--fine-class-weighting",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/repeated_fine_splits",
        help="Root directory for manifests, vocabs, evaluations, and summaries.",
    )
    parser.add_argument(
        "--run-root",
        default="runs/repeated_fine_splits",
        help="Root directory for checkpoints.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Read existing seed outputs and regenerate summaries without retraining.",
    )
    return parser


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def run_step(
    command: list[str],
    *,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        completed.stdout
        + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip().splitlines()[-1], flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_analysis_counts(path: Path) -> dict[str, int]:
    counts = {
        "acceptable_labels": 0,
        "weak_or_unstable_labels": 0,
        "severe_data_scarcity": 0,
        "low_validation_support": 0,
        "coarse_gate_confusion": 0,
        "fine_head_confusion": 0,
        "low_recall": 0,
        "low_precision": 0,
        "stable_evidence_labels": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            diagnosis = row["diagnosis"]
            if diagnosis == "acceptable":
                counts["acceptable_labels"] += 1
            else:
                counts[diagnosis] = counts.get(diagnosis, 0) + 1
                counts["weak_or_unstable_labels"] += 1

            val_chunks = int(row["val_chunks"])
            f1 = float(row["f1"])
            path_recall = float(row["path_recall"])
            if diagnosis == "acceptable" and val_chunks >= 5 and f1 >= 0.7 and path_recall >= 0.7:
                counts["stable_evidence_labels"] += 1
    return counts


def metric_value(summary: dict[str, Any], key: str) -> float:
    if key == "coarse_accuracy":
        return float(summary["coarse_accuracy"])
    if key == "coarse_macro_f1":
        return float(summary["coarse_macro_f1"])
    if key == "fine_macro_f1":
        return float(summary["fine_macro_f1"])
    if key == "path_accuracy":
        return float(summary["path_accuracy"])
    raise KeyError(key)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summaries[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_markdown(path: Path, summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    metric_keys = [
        "coarse_accuracy",
        "coarse_macro_f1",
        "fine_macro_f1",
        "path_accuracy",
    ]
    lines = [
        "# Repeated Fine-Stratified Split Summary",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{args.seeds}`",
        f"- Split: fine-stratified chunk split, validation ratio `{args.val_ratio}`",
        f"- Epochs per run: `{args.epochs}`",
        f"- Max events per chunk: `{args.max_events}`",
        f"- Class weighting: coarse `{args.class_weighting}`, fine `{args.fine_class_weighting}`",
        "",
        "This is a development stability protocol. It does not create new traffic "
        "statistics or manual aggregate features; it repeats the same end-to-end "
        "protocol-event LSTM workflow under different fine-label splits.",
        "",
        "## Seed Results",
        "",
        "| seed | coarse_acc | coarse_macro_f1 | fine_macro_f1 | path_acc | weak/unstable | stable_evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {seed} | {coarse_accuracy:.3f} | {coarse_macro_f1:.3f} | "
            "{fine_macro_f1:.3f} | {path_accuracy:.3f} | "
            "{weak_or_unstable_labels} | {stable_evidence_labels} |".format(**summary)
        )

    lines.extend(["", "## Stability Ranges", ""])
    lines.append("| metric | min | median | max |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in metric_keys:
        values = [metric_value(summary, key) for summary in summaries]
        lines.append(
            f"| {key} | {min(values):.3f} | {median(values):.3f} | {max(values):.3f} |"
        )

    weak_counts = [int(summary["weak_or_unstable_labels"]) for summary in summaries]
    stable_counts = [int(summary["stable_evidence_labels"]) for summary in summaries]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Weak or unstable labels ranged from `{min(weak_counts)}` to `{max(weak_counts)}`.",
            f"- Stable evidence labels ranged from `{min(stable_counts)}` to `{max(stable_counts)}`.",
            "- If a label is weak in every repeated split, treat it as a real "
            "limitation or data-scarcity issue. If it changes substantially by "
            "seed, report it as split-sensitive rather than a final model claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_existing_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    seed_name = f"seed_{seed}"
    seed_output = Path(args.output_root) / seed_name
    run_dir = Path(args.run_root) / seed_name
    eval_dir = seed_output / "evaluation"
    analysis_dir = seed_output / "error_analysis"
    metrics = read_json(eval_dir / "metrics.json")
    analysis_counts = read_analysis_counts(analysis_dir / "weak_fine_labels.csv")
    return {
        "seed": seed,
        "val_samples": metrics["total_samples"],
        "coarse_accuracy": metrics["coarse"]["accuracy"],
        "coarse_macro_f1": metrics["coarse"]["macro_f1"],
        "fine_macro_f1": metrics["fine_given_true_category"]["macro_f1"],
        "path_accuracy": metrics["path_accuracy"],
        **analysis_counts,
        "manifest": (seed_output / "manifest.csv").as_posix(),
        "evaluation_dir": eval_dir.as_posix(),
        "analysis_dir": analysis_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
    }


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def diagnosis_counts(diagnoses: list[str]) -> str:
    counts: dict[str, int] = {}
    for diagnosis in diagnoses:
        counts[diagnosis] = counts.get(diagnosis, 0) + 1
    return "; ".join(
        f"{diagnosis}={count}"
        for diagnosis, count in sorted(counts.items())
    )


def write_label_stability(output_root: Path, seeds: list[int]) -> None:
    rows_by_label: dict[str, list[dict[str, str]]] = {}
    for seed in seeds:
        analysis_path = (
            output_root
            / f"seed_{seed}"
            / "error_analysis"
            / "weak_fine_labels.csv"
        )
        for row in read_label_rows(analysis_path):
            row = dict(row)
            row["seed"] = str(seed)
            rows_by_label.setdefault(row["fine_label"], []).append(row)

    stability_rows: list[dict[str, Any]] = []
    for label, label_rows in sorted(rows_by_label.items()):
        diagnoses = [row["diagnosis"] for row in label_rows]
        f1_values = [float(row["f1"]) for row in label_rows]
        path_values = [float(row["path_recall"]) for row in label_rows]
        weak_count = sum(diagnosis != "acceptable" for diagnosis in diagnoses)
        severe_count = sum(diagnosis == "severe_data_scarcity" for diagnosis in diagnoses)
        stability_rows.append(
            {
                "fine_label": label,
                "coarse_label": label_rows[0]["coarse_label"],
                "weak_splits": weak_count,
                "acceptable_splits": len(label_rows) - weak_count,
                "severe_scarcity_splits": severe_count,
                "path_recall_min": min(path_values),
                "path_recall_median": median(path_values),
                "path_recall_max": max(path_values),
                "f1_min": min(f1_values),
                "f1_median": median(f1_values),
                "f1_max": max(f1_values),
                "diagnoses": diagnosis_counts(diagnoses),
            }
        )

    stability_rows.sort(
        key=lambda row: (
            -int(row["weak_splits"]),
            float(row["path_recall_median"]),
            row["fine_label"],
        )
    )
    csv_path = output_root / "label_stability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(stability_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(stability_rows)

    lines = [
        "# Fine-Label Stability Across Repeated Splits",
        "",
        "| fine_label | weak_splits | path_recall min/median/max | f1 min/median/max | diagnoses |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in stability_rows:
        lines.append(
            "| {fine_label} | {weak_splits} | "
            "{path_recall_min:.3f}/{path_recall_median:.3f}/{path_recall_max:.3f} | "
            "{f1_min:.3f}/{f1_median:.3f}/{f1_max:.3f} | {diagnoses} |".format(
                **row
            )
        )
    (output_root / "label_stability.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_seed(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    seed_name = f"seed_{seed}"
    seed_output = Path(args.output_root) / seed_name
    run_dir = Path(args.run_root) / seed_name
    manifest_path = seed_output / "manifest.csv"
    vocab_path = seed_output / "vocab.json"
    eval_dir = seed_output / "evaluation"
    analysis_dir = seed_output / "error_analysis"

    python = sys.executable
    run_step(
        [
            python,
            "scripts/build_dev_split_manifest.py",
            "--sequence-root",
            args.sequence_root,
            "--output",
            manifest_path.as_posix(),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(seed),
            "--stratify-by",
            "fine",
        ],
        log_path=seed_output / "logs" / "build_manifest.log",
    )
    run_step(
        [
            python,
            "scripts/build_vocab.py",
            "--manifest",
            manifest_path.as_posix(),
            "--split",
            "train",
            "--output",
            vocab_path.as_posix(),
        ],
        log_path=seed_output / "logs" / "build_vocab.log",
    )
    run_step(
        [
            python,
            "scripts/train_hierarchical.py",
            "--manifest",
            manifest_path.as_posix(),
            "--vocab",
            vocab_path.as_posix(),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--max-events",
            str(args.max_events),
            "--embedding-dim",
            str(args.embedding_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--class-weighting",
            args.class_weighting,
            "--fine-class-weighting",
            args.fine_class_weighting,
            "--seed",
            str(seed),
            "--output-dir",
            run_dir.as_posix(),
        ],
        log_path=seed_output / "logs" / "train.log",
    )
    run_step(
        [
            python,
            "scripts/evaluate_hierarchical.py",
            "--manifest",
            manifest_path.as_posix(),
            "--vocab",
            vocab_path.as_posix(),
            "--checkpoint",
            (run_dir / "best_model.pt").as_posix(),
            "--split",
            "val",
            "--max-events",
            str(args.max_events),
            "--output-dir",
            eval_dir.as_posix(),
        ],
        log_path=seed_output / "logs" / "evaluate.log",
    )
    run_step(
        [
            python,
            "scripts/analyze_weak_fine_labels.py",
            "--manifest",
            manifest_path.as_posix(),
            "--evaluation-dir",
            eval_dir.as_posix(),
            "--output-dir",
            analysis_dir.as_posix(),
        ],
        log_path=seed_output / "logs" / "analyze.log",
    )

    metrics = read_json(eval_dir / "metrics.json")
    analysis_counts = read_analysis_counts(analysis_dir / "weak_fine_labels.csv")
    return {
        "seed": seed,
        "val_samples": metrics["total_samples"],
        "coarse_accuracy": metrics["coarse"]["accuracy"],
        "coarse_macro_f1": metrics["coarse"]["macro_f1"],
        "fine_macro_f1": metrics["fine_given_true_category"]["macro_f1"],
        "path_accuracy": metrics["path_accuracy"],
        **analysis_counts,
        "manifest": manifest_path.as_posix(),
        "evaluation_dir": eval_dir.as_posix(),
        "analysis_dir": analysis_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
    }


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    summaries = []
    for seed in seeds:
        if args.summarize_only:
            summaries.append(summarize_existing_seed(seed, args))
        else:
            print(f"\n=== seed {seed} ===", flush=True)
            summaries.append(run_seed(seed, args))

    output_root = Path(args.output_root)
    write_summary_csv(output_root / "summary.csv", summaries)
    write_summary_markdown(output_root / "summary.md", summaries, args)
    write_label_stability(output_root, seeds)
    print(f"\nsummary_csv={output_root / 'summary.csv'}")
    print(f"summary_markdown={output_root / 'summary.md'}")
    print(f"label_stability={output_root / 'label_stability.md'}")


if __name__ == "__main__":
    main()
