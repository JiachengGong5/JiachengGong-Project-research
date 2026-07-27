#!/usr/bin/env python3
"""Run the complete PCAP-to-model workflow for a labeled network dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_patterns.schema import LabelSchema  # noqa: E402
from activity_patterns.labels import CICIOT2023_SCHEMA  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Folder containing one subdirectory per class and PCAP/PCAPNG files",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name used for the output workspace; defaults to the input folder name",
    )
    parser.add_argument(
        "--output-root",
        default="workflow_runs",
        help="Parent directory for self-contained workflow workspaces",
    )
    parser.add_argument(
        "--label-schema",
        help=(
            "Optional hierarchical label schema JSON, or the built-in name "
            "'ciciot2023'. If omitted, class folders become flat classes."
        ),
    )
    parser.add_argument("--initial-max-events", type=int, default=4096)
    parser.add_argument("--segment-events", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-unit",
        choices=("auto", "capture", "chunk"),
        default="auto",
    )
    parser.add_argument(
        "--stratify-by",
        choices=("fine", "coarse"),
        default="fine",
    )
    parser.add_argument("--min-token-frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--coarse-loss-weight", type=float, default=1.0)
    parser.add_argument("--fine-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--fine-class-weighting",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--force-processing",
        action="store_true",
        help="Regenerate Zeek logs and sequence files even when they exist",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip metrics evaluation; optional pattern extraction can still run",
    )
    parser.add_argument(
        "--extract-patterns",
        action="store_true",
        help="Run occlusion explanation and Zeek trace reconstruction after evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved schema and commands without executing them",
    )
    return parser


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not slug:
        raise ValueError("Dataset name must contain at least one letter or number")
    return slug


def discover_pcaps(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )


def discover_folder_labels(input_root: Path, pcaps: list[Path]) -> tuple[str, ...]:
    labels = []
    for pcap in pcaps:
        if pcap.parent == input_root:
            raise ValueError(
                f"{pcap} is directly under the input root. Put every capture "
                "inside a class folder such as INPUT/Benign/capture.pcap."
            )
        labels.append(pcap.parent.name)
    return tuple(sorted(set(labels)))


def resolve_schema(
    schema_path: str | None,
    *,
    dataset_name: str,
    folder_labels: tuple[str, ...],
) -> LabelSchema:
    if schema_path:
        schema = (
            CICIOT2023_SCHEMA
            if schema_path.lower() in {"ciciot2023", "cic-iot-2023"}
            else LabelSchema.load(schema_path)
        )
        unknown = []
        for label in folder_labels:
            try:
                schema.canonical_fine_label(label)
            except KeyError:
                unknown.append(label)
        if unknown:
            raise ValueError(
                "The label schema does not recognize these input folders: "
                + ", ".join(unknown)
            )
        return schema
    return LabelSchema.from_flat_labels(folder_labels, name=dataset_name)


def run_step(
    stage: str,
    command: list[str],
    *,
    workspace: Path,
    dry_run: bool,
) -> None:
    rendered = " ".join(command)
    print(f"\n[{stage}]\n{rendered}", flush=True)
    if dry_run:
        return

    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stage}.log"
    environment = dict(os.environ)
    source_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not capture output for stage {stage}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Workflow stage {stage!r} failed with exit code {return_code}. "
            f"See {log_path}."
        )


def command_paths(workspace: Path) -> dict[str, Path]:
    return {
        "schema": workspace / "label_schema.json",
        "zeek": workspace / "zeek",
        "sequences": workspace / "sequences",
        "segments": workspace / "segments",
        "chunk_manifest": workspace / "manifests" / "chunks.csv",
        "event_manifest": workspace / "manifests" / "events.csv",
        "vocab": workspace / "artifacts" / "vocab.json",
        "model_dir": workspace / "model",
        "evaluation": workspace / "artifacts" / "evaluation",
        "patterns": workspace / "artifacts" / "activity_patterns",
    }


def build_commands(
    args: argparse.Namespace,
    *,
    input_root: Path,
    paths: dict[str, Path],
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    process_command = [
        python,
        "scripts/process_raw_pcaps.py",
        "--raw-root",
        str(input_root),
        "--zeek-root",
        str(paths["zeek"]),
        "--sequence-root",
        str(paths["sequences"]),
        "--zeek-sample-root",
        str(paths["sequences"].parent / "zeek_sample"),
        "--max-events",
        str(args.initial_max_events),
        "--label-schema",
        str(paths["schema"]),
    ]
    if args.force_processing:
        process_command.extend(["--force-zeek", "--force-sequences"])

    commands = [
        ("01-process-pcaps", process_command),
        (
            "02-build-split",
            [
                python,
                "scripts/build_dev_split_manifest.py",
                "--sequence-root",
                str(paths["sequences"]),
                "--output",
                str(paths["chunk_manifest"]),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--stratify-by",
                args.stratify_by,
                "--split-unit",
                args.split_unit,
                "--label-schema",
                str(paths["schema"]),
            ],
        ),
        (
            "03-segment-events",
            [
                python,
                "scripts/rechunk_sequences.py",
                "--manifest",
                str(paths["chunk_manifest"]),
                "--output-root",
                str(paths["segments"]),
                "--output-manifest",
                str(paths["event_manifest"]),
                "--max-events",
                str(args.segment_events),
            ],
        ),
        (
            "04-build-vocabulary",
            [
                python,
                "scripts/build_vocab.py",
                "--manifest",
                str(paths["event_manifest"]),
                "--split",
                "train",
                "--min-freq",
                str(args.min_token_frequency),
                "--output",
                str(paths["vocab"]),
            ],
        ),
        (
            "05-train-model",
            [
                python,
                "scripts/train_hierarchical.py",
                "--manifest",
                str(paths["event_manifest"]),
                "--vocab",
                str(paths["vocab"]),
                "--label-schema",
                str(paths["schema"]),
                "--output-dir",
                str(paths["model_dir"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-events",
                str(args.segment_events),
                "--embedding-dim",
                str(args.embedding_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--dropout",
                str(args.dropout),
                "--learning-rate",
                str(args.learning_rate),
                "--seed",
                str(args.seed),
                "--coarse-loss-weight",
                str(args.coarse_loss_weight),
                "--fine-loss-weight",
                str(args.fine_loss_weight),
                "--class-weighting",
                args.class_weighting,
                "--fine-class-weighting",
                args.fine_class_weighting,
            ],
        ),
    ]
    if not args.skip_evaluation:
        commands.append(
            (
                "06-evaluate-model",
                [
                    python,
                    "scripts/evaluate_hierarchical.py",
                    "--manifest",
                    str(paths["event_manifest"]),
                    "--vocab",
                    str(paths["vocab"]),
                    "--checkpoint",
                    str(paths["model_dir"] / "best_model.pt"),
                    "--label-schema",
                    str(paths["schema"]),
                    "--split",
                    "val",
                    "--max-events",
                    str(args.segment_events),
                    "--batch-size",
                    str(args.batch_size),
                    "--output-dir",
                    str(paths["evaluation"]),
                ],
            )
        )
    if args.extract_patterns:
        commands.append(
            (
                "07-extract-patterns",
                [
                    python,
                    "scripts/extract_activity_patterns.py",
                    "--manifest",
                    str(paths["event_manifest"]),
                    "--vocab",
                    str(paths["vocab"]),
                    "--checkpoint",
                    str(paths["model_dir"] / "best_model.pt"),
                    "--label-schema",
                    str(paths["schema"]),
                    "--split",
                    "val",
                    "--max-events",
                    str(args.segment_events),
                    "--zeek-root",
                    str(paths["zeek"]),
                    "--zeek-sample-root",
                    str(paths["sequences"].parent / "zeek_sample"),
                    "--output-dir",
                    str(paths["patterns"]),
                ],
            )
        )
    return commands


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_name: str,
    input_root: Path,
    workspace: Path,
    schema: LabelSchema,
    pcaps: list[Path],
    paths: dict[str, Path],
    status: str,
    error: str | None = None,
) -> None:
    metrics: dict[str, Any] | None = None
    metrics_path = paths["evaluation"] / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    payload = {
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "input_root": str(input_root),
        "workspace": str(workspace),
        "pcap_count": len(pcaps),
        "label_schema": schema.to_json(),
        "parameters": vars(args),
        "outputs": {name: str(value) for name, value in paths.items()},
        "evaluation_metrics": metrics,
        "error": error,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.is_dir():
        raise ValueError(f"Input root is not a directory: {input_root}")

    pcaps = discover_pcaps(input_root)
    if not pcaps:
        raise ValueError(f"No PCAP or PCAPNG files found under {input_root}")
    folder_labels = discover_folder_labels(input_root, pcaps)
    dataset_name = args.dataset_name or input_root.name
    dataset_slug = slugify(dataset_name)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    workspace = (output_root / dataset_slug).resolve()
    paths = command_paths(workspace)
    schema = resolve_schema(
        args.label_schema,
        dataset_name=dataset_name,
        folder_labels=folder_labels,
    )

    workspace.mkdir(parents=True, exist_ok=True)
    schema.save(paths["schema"])
    commands = build_commands(args, input_root=input_root, paths=paths)

    print(f"dataset={dataset_name}")
    print(f"workspace={workspace}")
    print(f"pcaps={len(pcaps)}")
    print(f"folder_labels={', '.join(folder_labels)}")
    print(f"coarse_classes={len(schema.category_names)}")
    print(f"fine_classes={len(schema.fine_label_names)}")

    if not args.dry_run and shutil.which("zeek") is None:
        raise RuntimeError(
            "Zeek is required but was not found on PATH. Install Zeek or add "
            "its executable directory to PATH."
        )

    summary_path = workspace / "workflow_summary.json"
    try:
        for stage, command in commands:
            run_step(stage, command, workspace=workspace, dry_run=args.dry_run)
    except Exception as exc:
        write_summary(
            summary_path,
            args=args,
            dataset_name=dataset_name,
            input_root=input_root,
            workspace=workspace,
            schema=schema,
            pcaps=pcaps,
            paths=paths,
            status="failed",
            error=str(exc),
        )
        raise

    write_summary(
        summary_path,
        args=args,
        dataset_name=dataset_name,
        input_root=input_root,
        workspace=workspace,
        schema=schema,
        pcaps=pcaps,
        paths=paths,
        status="dry-run" if args.dry_run else "completed",
    )
    print(f"\nworkflow_summary={summary_path}")


if __name__ == "__main__":
    main()
