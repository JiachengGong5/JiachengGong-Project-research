# Identifying Activity Patterns in Intrusion Detection Data Sets

This repository is a research scaffold for learning intrusion activity patterns
directly from protocol-semantic event sequences in the
[CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

The primary path is:

```text
PCAP
-> Zeek JSON protocol logs
-> chronological semantic events
-> LSTM classification
-> salient event span
-> Zeek-based activity trace reconstruction
```

The project intentionally does **not** use manually aggregated traffic
statistics such as means, standard deviations, rates, flow totals, or rolling
window features. CICIoT2023's supplied CSV files are reserved for comparison
with the traditional feature-engineering baseline; primary experiments start
from the original PCAP files.

## Research Scope

The recommended first task is hierarchical classification. The coarse head uses
8 categories:

- Benign
- DDoS
- DoS
- Recon
- Web-based
- Brute Force
- Spoofing
- Mirai

The fine heads classify attacks only inside the predicted or true category.
For example, the Recon head distinguishes ping sweep, OS scan, vulnerability
scan, port scan, and host discovery; the Web-based head distinguishes SQL
injection, command injection, XSS, and related attacks.

The current local checkpoint covers all 33 official CICIoT2023 attacks plus
Benign, for 34 output labels. This is complete label coverage, but several rare
labels still have too few chunks for a reliable class-level claim.

Protocol state alone is not sufficient to reliably distinguish every fine
attack type. For example, SQL injection and XSS can share the same HTTP method
and connection state. The hierarchical 33-attack extension should add an
end-to-end byte or character encoder for raw application fields such as HTTP
URI data.

See [docs/research_plan.md](docs/research_plan.md) for the full experimental
design and rationale.

Tracking related data flows is part of the interpretation layer, not a
separate project. The LSTM is still trained only on protocol-semantic tokens,
but salient spans are mapped back to Zeek timestamps, `uid` values, and
protocol logs to reconstruct the associated activity trace. See
[docs/multi_connection_sessionization.md](docs/multi_connection_sessionization.md).

## Event Representation

Each Zeek log record becomes one time step. A time step contains categorical
tokens such as:

```text
log=conn proto=tcp service=http conn_state=S0 history=S resp_port=80
log=http method=GET version=1.1 status_code=200
log=dns qtype_name=A rcode_name=NOERROR opcode_name=query
```

The sequence preserves order and repetition. The LSTM must learn the useful
transitions and repeated patterns itself.

The preprocessor excludes identifiers and traditional flow summaries:

- Excluded: IP addresses, origin ephemeral ports, UID, duration, bytes,
  packet counts, rates, means, standard deviations, and rolling statistics.
- Included: protocol, service, protocol state, TCP history, responder service
  port, HTTP method/status, DNS operation/result, TLS state, SSH state, DHCP
  message type, and Zeek protocol anomalies.

## Quick Start

Requirements:

- Zeek for PCAP protocol analysis
- Python 3.10+
- PyTorch for the LSTM model

Extract Zeek JSON logs from one capture:

```bash
./scripts/pcap_to_zeek.sh data/raw/DDoS/example.pcap data/zeek/DDoS/example
```

Convert the logs into contiguous event sequences:

```bash
PYTHONPATH=src python -m activity_patterns.prepare \
  data/zeek/DDoS/example \
  data/sequences/ddos-example.jsonl \
  --label DDoS \
  --sequence-id ddos-example \
  --max-events 4096
```

Run the current tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

`--max-events` only splits a very long capture into contiguous memory-sized
chunks. It does not calculate window statistics or create overlapping rolling
features. The development workflow below assigns parent chunks to a split and
then ensures that every smaller child segment inherits that parent split.

## Portable One-Command PCAP Workflow

For another dataset organized as `DATASET/<class>/*.pcap`, the complete
PCAP-to-model workflow can be run with one command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dataset_workflow.py \
  --input-root /path/to/another-pcap-dataset \
  --dataset-name another-dataset \
  --epochs 10 \
  --extract-patterns
```

Without a schema, class folder names are inferred automatically as flat output
classes. For a coarse/fine hierarchy, pass a JSON schema:

```bash
... --label-schema configs/datasets/example_hierarchical.json
```

The command generates Zeek logs, chronological event sequences, capture-aware
train/validation splits, non-overlapping segments, a train-only vocabulary,
the hierarchical LSTM checkpoint, evaluation tables, and optional Occlusion
plus Trace outputs. All products and stage logs are stored under
`workflow_runs/<dataset-name>/`.

See [docs/portable_dataset_workflow.md](docs/portable_dataset_workflow.md) for
the input contract, schema format, output tree, and split policy.

## Phase 2 Smoke Data

After a few PCAP files have been converted into sequence JSONL files, build a
training manifest:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_manifest.py \
  --sequence-root data/sequences \
  --output manifests/smoke_manifest.csv
```

Build a vocabulary from the training split only:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_vocab.py \
  --manifest manifests/smoke_manifest.csv \
  --output artifacts/smoke_vocab.json
```

The manifest is small and can be versioned. The vocabulary is generated from
local data and is stored under `artifacts/`, which is ignored by Git.

Run a small training smoke test:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_smoke.py \
  --epochs 10 \
  --batch-size 16 \
  --max-events 128 \
  --output-dir runs/smoke_overfit
```

This run intentionally evaluates on the same smoke data. It is an engineering
sanity check for the training loop, not a publishable model score.

Generate development evaluation tables from a trained checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_hierarchical.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --checkpoint runs/hierarchical_dev/best_model.pt \
  --output-dir artifacts/evaluation_dev
```

Extract salient activity spans and reconstruct their Zeek traces:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_activity_patterns.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --checkpoint runs/hierarchical_dev/best_model.pt \
  --output-dir artifacts/activity_patterns_dev
```

## Development Train/Validation Split

When the local subset has only one downloaded PCAP per class, use a
stratified chunk-level split only as a development check:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dev_split_manifest.py \
  --sequence-root data/sequences \
  --output manifests/dev_chunk_manifest.csv

PYTHONPATH=src .venv/bin/python scripts/rechunk_sequences.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --output-root data/sequences_windowed \
  --output-manifest manifests/dev_event_manifest.csv \
  --max-events 256

PYTHONPATH=src .venv/bin/python scripts/build_vocab.py \
  --manifest manifests/dev_event_manifest.csv \
  --split train \
  --output artifacts/dev_vocab.json

PYTHONPATH=src .venv/bin/python scripts/train_hierarchical.py \
  --manifest manifests/dev_event_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --epochs 4 \
  --batch-size 64 \
  --max-events 256 \
  --class-weighting balanced \
  --fine-class-weighting balanced \
  --coarse-loss-weight 1.0 \
  --fine-loss-weight 0.25 \
  --fine-loss-reduction sample_mean \
  --output-dir runs/hierarchical_event256_joint
```

This split is useful for catching modeling and imbalance issues early. It is
not a final evaluation protocol because chunks from the same capture can land
in both train and validation. Final reported results should use
capture-disjoint splits after multiple captures per class are available.

The split-preserving re-chunking stage retains all 13,149,952 current protocol
events exactly once in non-overlapping 256-event segments. It raises model
event coverage from 3.1% in the previous prefix-truncation baseline to 100%
without adding manual statistical features.

## Expanding To 8 Classes

Check which coarse classes are currently present:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_data_coverage.py
```

After downloading new PCAP folders under `data/raw/<label>/`, process them:

If the browser downloads the PCAP files to `~/Downloads`, import them first:

```bash
PYTHONPATH=src .venv/bin/python scripts/import_downloaded_pcaps.py
```

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_pcaps.py \
  --only DoS-SYN_Flood \
  --only DNS_Spoofing \
  --only DictionaryBruteForce \
  --only Mirai-udpplain \
  --only SqlInjection
```

For very large flood captures, use a Zeek-log sample for the first validation
run:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_pcaps.py \
  --only DoS-SYN_Flood \
  --sample-conn-lines 200000
```

See [docs/8class_expansion_plan.md](docs/8class_expansion_plan.md) for the
download checklist and full command sequence.

## Expanding To Full Fine Labels

The current hierarchical model supports the full CICIoT2023 label hierarchy,
but local data coverage must be complete before reporting 33-attack
fine-label results.

Generate a coarse and fine coverage report:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_data_coverage.py \
  --output-dir artifacts/coverage
```

The report identifies missing PCAP folders and writes:

```text
artifacts/coverage/coverage_report.md
artifacts/coverage/missing_downloads.csv
```

After full fine-label coverage is downloaded and processed, use
`scripts/evaluate_hierarchical.py` to report:

- coarse 8-class metrics,
- fine-label metrics within the true coarse category,
- path accuracy where both coarse and fine labels are correct.

See [docs/full_dataset_expansion_plan.md](docs/full_dataset_expansion_plan.md)
for the current missing-download checklist and command sequence.
