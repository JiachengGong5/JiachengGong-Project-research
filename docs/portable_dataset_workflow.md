# Portable PCAP Dataset Workflow

`scripts/run_dataset_workflow.py` runs the complete project pipeline for a
dataset organized as labeled PCAP folders. It keeps each dataset in a
self-contained workspace and removes the CICIoT2023 label dependency from the
model and training data path.

## Input Contract

Place every capture under a class folder:

```text
/path/to/new-dataset/
├── Normal/
│   ├── capture-01.pcap
│   └── capture-02.pcapng
├── PortScan/
│   ├── scan-01.pcap
│   └── scan-02.pcap
└── SYN_Flood/
    └── flood-01.pcap
```

The workflow accepts `.pcap` and `.pcapng`, including uppercase extensions.
The immediate parent folder of each capture is its source label.

## One-Command Flat Classification

When no label schema is supplied, every class folder becomes one output class:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dataset_workflow.py \
  --input-root /path/to/new-dataset \
  --dataset-name new-iot-dataset \
  --epochs 10 \
  --extract-patterns
```

This mode still uses `HierarchicalProtocolEventLSTM`, but each coarse category
has one fine label. The path prediction is therefore equivalent to flat
classification and can later be upgraded to a true hierarchy without changing
the event pipeline.

## Hierarchical Labels

Pass a JSON schema when folder names must map to canonical fine labels and
coarse categories:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dataset_workflow.py \
  --input-root /path/to/new-dataset \
  --dataset-name new-iot-dataset \
  --label-schema configs/datasets/example_hierarchical.json \
  --epochs 10 \
  --extract-patterns
```

The schema format is:

```json
{
  "name": "example",
  "category_to_fine_labels": {
    "Benign": ["Benign"],
    "Flood": ["SYN Flood", "UDP Flood"]
  },
  "aliases": {
    "Normal": "Benign",
    "syn_flood": "SYN Flood",
    "udp_flood": "UDP Flood"
  }
}
```

For the existing dataset, use the built-in hierarchy:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dataset_workflow.py \
  --input-root data/raw \
  --dataset-name ciciot2023-portable \
  --label-schema ciciot2023
```

## Automatic Stages

The orchestrator runs:

```text
PCAP/PCAPNG
  -> process_raw_pcaps.py
  -> Zeek JSON protocol logs
  -> chronological sequence JSONL
  -> build_dev_split_manifest.py
  -> capture-disjoint or fallback chunk split
  -> rechunk_sequences.py
  -> non-overlapping event segments
  -> build_vocab.py using train only
  -> train_hierarchical.py
  -> best_model.pt and training metrics
  -> evaluate_hierarchical.py
  -> predictions, metrics, and confusion matrices
  -> optional extract_activity_patterns.py
  -> occlusion spans and Zeek traces
```

No stage calculates traffic means, standard deviations, rates, or rolling
aggregate features.

## Output Workspace

The default output is `workflow_runs/<dataset-name>/`:

```text
workflow_runs/new-iot-dataset/
├── label_schema.json
├── zeek/
├── sequences/
├── segments/
├── manifests/
│   ├── chunks.csv
│   └── events.csv
├── artifacts/
│   ├── vocab.json
│   ├── evaluation/
│   └── activity_patterns/
├── model/
│   ├── best_model.pt
│   └── metrics.json
├── logs/
└── workflow_summary.json
```

The checkpoint stores the complete label schema, so evaluation and activity
pattern extraction can reconstruct the correct model heads without importing
CICIoT2023 constants.

## Split Policy

`--split-unit auto` is the default:

- If a fine label has two or more capture files, entire captures are assigned
  to either train or validation.
- If a fine label has only one capture but multiple chunks, it falls back to a
  chunk-level development split.
- If a label has only one capture and one chunk, no reliable validation example
  can be created; add another capture or produce more contiguous chunks.

For final research claims, use multiple captures per class and
`--split-unit capture`.

## Useful Options

Preview all resolved commands:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dataset_workflow.py \
  --input-root /path/to/new-dataset \
  --dry-run
```

Regenerate existing Zeek logs and sequences:

```bash
... --force-processing
```

Write the workspace outside the repository:

```bash
... --output-root /external/disk/workflows
```

## Portability Boundary

The workflow is portable across PCAP datasets that Zeek can decode. Adding
protocols not currently represented in `src/activity_patterns/events.py`
requires adding direct protocol-semantic fields to `FIELDS_BY_LOG`. A dataset
that contains only precomputed statistical CSV features is not accepted by this
PCAP workflow.
