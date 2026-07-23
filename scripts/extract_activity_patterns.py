#!/usr/bin/env python3
"""Extract salient event spans and reconstruct their Zeek activity traces."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from activity_patterns.dataset import SequenceChunkDataset  # noqa: E402
from activity_patterns.labels import (  # noqa: E402
    CATEGORY_NAMES,
    CATEGORY_TO_NUM_FINE,
    FINE_LABEL_NAMES,
    fine_label_from_category_index,
)
from activity_patterns.manifest import read_manifest  # noqa: E402
from activity_patterns.model import HierarchicalProtocolEventLSTM  # noqa: E402
from activity_patterns.trace import (  # noqa: E402
    reconstruct_activity_trace,
    resolve_zeek_log_dir,
)
from activity_patterns.vocab import TokenVocab  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/dev_chunk_manifest.csv")
    parser.add_argument("--vocab", default="artifacts/dev_vocab.json")
    parser.add_argument("--checkpoint", default="runs/hierarchical_dev/best_model.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--occlusion-window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-samples-per-class", type=int, default=1)
    parser.add_argument(
        "--group-by",
        choices=("coarse", "fine"),
        default="coarse",
        help="Choose representative samples per coarse category or per fine label.",
    )
    parser.add_argument("--trace-context-seconds", type=float, default=1.0)
    parser.add_argument("--trace-uid-context-seconds", type=float, default=5.0)
    parser.add_argument("--max-trace-records", type=int, default=30)
    parser.add_argument("--output-dir", default="artifacts/activity_patterns_dev")
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include incorrectly predicted samples when correct examples are scarce.",
    )
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


def read_payload(dataset: SequenceChunkDataset, index: int) -> dict[str, Any]:
    ref = dataset.refs[index]
    row = dataset.rows[ref.row_index]
    with row.sequence_path.open("r", encoding="utf-8") as handle:
        handle.seek(ref.byte_offset)
        return json.loads(handle.readline())


def encode_events(
    events: list[dict[str, Any]],
    vocab: TokenVocab,
) -> list[list[int]]:
    return [
        vocab.encode_event(event.get("tokens", []))
        for event in events
    ]


def tensor_from_token_ids(
    token_ids: list[list[int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_ids:
        raise ValueError("Cannot build a tensor from an empty event sequence")

    max_fields = max(len(event_tokens) for event_tokens in token_ids)
    tensor = torch.zeros((1, len(token_ids), max_fields), dtype=torch.long)
    for time_index, event_tokens in enumerate(token_ids):
        tensor[0, time_index, : len(event_tokens)] = torch.tensor(
            event_tokens,
            dtype=torch.long,
        )
    lengths = torch.tensor([len(token_ids)], dtype=torch.long)
    return tensor.to(device), lengths.to(device)


@torch.no_grad()
def hierarchical_prediction(
    model: HierarchicalProtocolEventLSTM,
    token_ids: list[list[int]],
    device: torch.device,
) -> dict[str, Any]:
    tensor, lengths = tensor_from_token_ids(token_ids, device=device)
    outputs = model(tensor, lengths)
    coarse_logits = outputs["coarse"]
    fine_logits_by_category = outputs["fine"]
    if not isinstance(coarse_logits, torch.Tensor):
        raise TypeError("outputs['coarse'] must be a tensor")
    if not isinstance(fine_logits_by_category, Mapping):
        raise TypeError("outputs['fine'] must be a mapping")

    coarse_probabilities = torch.softmax(coarse_logits[0], dim=0)
    coarse_index = int(torch.argmax(coarse_probabilities).detach().cpu())
    coarse_label = CATEGORY_NAMES[coarse_index]
    coarse_confidence = float(coarse_probabilities[coarse_index].detach().cpu())

    fine_logits = fine_logits_by_category[coarse_label]
    if not isinstance(fine_logits, torch.Tensor):
        raise TypeError(f"outputs['fine'][{coarse_label!r}] must be a tensor")
    fine_probabilities = torch.softmax(fine_logits[0], dim=0)
    fine_index = int(torch.argmax(fine_probabilities).detach().cpu())
    fine_confidence = float(fine_probabilities[fine_index].detach().cpu())
    fine_label = fine_label_from_category_index(coarse_label, fine_index)

    return {
        "coarse_index": coarse_index,
        "coarse_label": coarse_label,
        "coarse_confidence": coarse_confidence,
        "fine_index": fine_index,
        "fine_label": fine_label,
        "fine_confidence": fine_confidence,
        "path_confidence": coarse_confidence * fine_confidence,
        "coarse_probabilities": [
            float(value) for value in coarse_probabilities.detach().cpu()
        ],
    }


@torch.no_grad()
def target_probability(
    model: HierarchicalProtocolEventLSTM,
    token_ids: list[list[int]],
    target_coarse_index: int,
    device: torch.device,
    target_fine_index: int | None = None,
) -> float:
    tensor, lengths = tensor_from_token_ids(token_ids, device=device)
    outputs = model(tensor, lengths)
    coarse_logits = outputs["coarse"]
    fine_logits_by_category = outputs["fine"]
    if not isinstance(coarse_logits, torch.Tensor):
        raise TypeError("outputs['coarse'] must be a tensor")
    coarse_probabilities = torch.softmax(coarse_logits[0], dim=0)
    coarse_probability = coarse_probabilities[target_coarse_index]
    if target_fine_index is None:
        return float(coarse_probability.detach().cpu())

    if not isinstance(fine_logits_by_category, Mapping):
        raise TypeError("outputs['fine'] must be a mapping")
    category = CATEGORY_NAMES[target_coarse_index]
    fine_logits = fine_logits_by_category[category]
    if not isinstance(fine_logits, torch.Tensor):
        raise TypeError(f"outputs['fine'][{category!r}] must be a tensor")
    fine_probabilities = torch.softmax(fine_logits[0], dim=0)
    path_probability = coarse_probability * fine_probabilities[target_fine_index]
    return float(path_probability.detach().cpu())


def occlusion_search(
    model: HierarchicalProtocolEventLSTM,
    token_ids: list[list[int]],
    *,
    target_coarse_index: int,
    target_fine_index: int | None,
    unk_id: int,
    window: int,
    stride: int,
    device: torch.device,
) -> dict[str, Any]:
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")

    baseline = target_probability(
        model,
        token_ids,
        target_coarse_index,
        device,
        target_fine_index=target_fine_index,
    )
    best = {
        "start": 0,
        "end": min(window, len(token_ids)),
        "baseline_probability": baseline,
        "occluded_probability": baseline,
        "probability_drop": 0.0,
    }
    for start in range(0, len(token_ids), stride):
        end = min(start + window, len(token_ids))
        occluded = [list(event_tokens) for event_tokens in token_ids]
        for index in range(start, end):
            occluded[index] = [unk_id]
        probability = target_probability(
            model,
            occluded,
            target_coarse_index,
            device,
            target_fine_index=target_fine_index,
        )
        drop = baseline - probability
        if drop > best["probability_drop"]:
            best = {
                "start": start,
                "end": end,
                "baseline_probability": baseline,
                "occluded_probability": probability,
                "probability_drop": drop,
            }
        if end == len(token_ids):
            break
    return best


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": event.get("timestamp"),
        "log_type": event.get("log_type"),
        "tokens": event.get("tokens", []),
    }


def select_sample_indices(
    model: HierarchicalProtocolEventLSTM,
    dataset: SequenceChunkDataset,
    vocab: TokenVocab,
    *,
    device: torch.device,
    max_events: int,
    max_samples_per_class: int,
    include_incorrect: bool,
    group_by: str,
) -> list[tuple[int, int, int | None, float]]:
    candidates_by_class: dict[str, list[tuple[int, int, int | None, float]]] = (
        defaultdict(list)
    )
    class_order = FINE_LABEL_NAMES if group_by == "fine" else CATEGORY_NAMES

    for index in range(len(dataset)):
        item = dataset[index]
        payload = read_payload(dataset, index)
        events = payload.get("events", [])[:max_events]
        if not events:
            continue
        prediction = hierarchical_prediction(
            model,
            encode_events(events, vocab),
            device,
        )

        if group_by == "fine":
            true_label = item["fine_label"]
            correct = prediction["fine_label"] == true_label
            confidence = prediction["path_confidence"]
            target_fine_index: int | None = prediction["fine_index"]
        else:
            true_label = item["coarse_label"]
            correct = prediction["coarse_label"] == true_label
            confidence = prediction["coarse_confidence"]
            target_fine_index = None

        if not correct and not include_incorrect:
            continue

        candidates_by_class[true_label].append(
            (
                index,
                prediction["coarse_index"],
                target_fine_index,
                confidence,
            )
        )

    selected: list[tuple[int, int, int | None, float]] = []
    for label in class_order:
        candidates = sorted(
            candidates_by_class.get(label, []),
            key=lambda candidate: candidate[3],
            reverse=True,
        )
        selected.extend(candidates[:max_samples_per_class])
    return selected


def build_pattern_record(
    model: HierarchicalProtocolEventLSTM,
    dataset: SequenceChunkDataset,
    vocab: TokenVocab,
    *,
    dataset_index: int,
    target_coarse_index: int,
    target_fine_index: int | None,
    target_confidence: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    item = dataset[dataset_index]
    row = dataset.rows[dataset.refs[dataset_index].row_index]
    payload = read_payload(dataset, dataset_index)
    events = payload.get("events", [])[: args.max_events]
    token_ids = encode_events(events, vocab)

    occlusion = occlusion_search(
        model,
        token_ids,
        target_coarse_index=target_coarse_index,
        target_fine_index=target_fine_index,
        unk_id=vocab.unk_id,
        window=args.occlusion_window,
        stride=args.stride,
        device=device,
    )
    span_events = events[occlusion["start"] : occlusion["end"]]
    start_ts = float(span_events[0]["timestamp"])
    end_ts = float(span_events[-1]["timestamp"])
    log_dir = resolve_zeek_log_dir(row.sequence_path, project_root=PROJECT_ROOT)
    trace_records = reconstruct_activity_trace(
        log_dir,
        start_ts=start_ts,
        end_ts=end_ts,
        context_seconds=args.trace_context_seconds,
        uid_context_seconds=args.trace_uid_context_seconds,
        max_records=args.max_trace_records,
    )

    prediction = hierarchical_prediction(model, token_ids, device)

    return {
        "sequence_path": row.sequence_path.as_posix(),
        "sequence_id": item["sequence_id"],
        "chunk_index": item["chunk_index"],
        "source_chunk_index": item["source_chunk_index"],
        "event_start": item["event_start"],
        "event_stop": item["event_stop"],
        "true_coarse": item["coarse_label"],
        "true_fine": item["fine_label"],
        "pred_coarse": prediction["coarse_label"],
        "pred_fine": prediction["fine_label"],
        "pred_coarse_confidence": prediction["coarse_confidence"],
        "pred_fine_confidence": prediction["fine_confidence"],
        "pred_path_confidence": prediction["path_confidence"],
        "target_confidence": target_confidence,
        "explanation_target": "path" if target_fine_index is not None else "coarse",
        "span_start_index": item["event_start"] + occlusion["start"],
        "span_end_index": item["event_start"] + occlusion["end"],
        "span_start_ts": start_ts,
        "span_end_ts": end_ts,
        "baseline_probability": occlusion["baseline_probability"],
        "occluded_probability": occlusion["occluded_probability"],
        "probability_drop": occlusion["probability_drop"],
        "span_events": [compact_event(event) for event in span_events],
        "zeek_log_dir": log_dir.as_posix(),
        "trace_records": trace_records,
    }


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# Activity Pattern Extraction", ""]
    for index, record in enumerate(records, start=1):
        heading = record["pred_coarse"]
        if record.get("explanation_target") == "path":
            heading = f"{heading} / {record['pred_fine']}"
        lines.append(f"## Example {index}: {heading}")
        lines.append("")
        lines.append(
            f"- true: {record['true_coarse']} / {record['true_fine']}"
        )
        lines.append(
            f"- predicted: {record['pred_coarse']} / {record['pred_fine']}"
        )
        lines.append(f"- sequence: {record['sequence_id']} chunk {record['chunk_index']}")
        lines.append(f"- explanation target: {record['explanation_target']}")
        lines.append(
            "- occlusion drop: "
            f"{record['probability_drop']:.4f} "
            f"({record['baseline_probability']:.4f} -> "
            f"{record['occluded_probability']:.4f})"
        )
        lines.append("")
        lines.append("Salient events:")
        lines.append("")
        for event in record["span_events"]:
            tokens = " ".join(event["tokens"])
            lines.append(
                f"- {event['timestamp']} {event['log_type']}: `{tokens}`"
            )
        lines.append("")
        lines.append("Reconstructed Zeek trace:")
        lines.append("")
        for trace in record["trace_records"]:
            rendered = ", ".join(
                f"{key}={value}"
                for key, value in trace.items()
                if key != "log_type"
            )
            lines.append(f"- {trace['log_type']}: {rendered}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    selected = select_sample_indices(
        model,
        dataset,
        vocab,
        device=device,
        max_events=args.max_events,
        max_samples_per_class=args.max_samples_per_class,
        include_incorrect=args.include_incorrect,
        group_by=args.group_by,
    )
    records = [
        build_pattern_record(
            model,
            dataset,
            vocab,
            dataset_index=index,
            target_coarse_index=target_coarse_index,
            target_fine_index=target_fine_index,
            target_confidence=confidence,
            args=args,
            device=device,
        )
        for index, target_coarse_index, target_fine_index, confidence in selected
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "patterns.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_markdown(output_dir / "patterns.md", records)

    print(f"selected_examples={len(records)}")
    print(f"output_jsonl={jsonl_path}")
    print(f"output_markdown={output_dir / 'patterns.md'}")


if __name__ == "__main__":
    main()
