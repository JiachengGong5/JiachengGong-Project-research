# Full CICIoT2023 Expansion Plan

> Status update, July 20, 2026: this expansion has been completed. The local
> pipeline now contains all 33 official attacks plus Benign (34 output labels),
> including DDoS-ICMP Fragmentation. The coverage lists and commands below are
> retained as a historical record of the expansion process; current results are
> documented in `docs/progress_report.md` and `docs/professor_demo_guide.md`.

The current local data validates the end-to-end pipeline and covers most
coarse categories, but it is not yet the full CICIoT2023 fine-label set.

## Current Coverage

Ready:

- Benign: 1/1 fine label
- Recon: 5/5 fine labels
- Web-based: 6/6 fine labels
- Brute Force: 1/1 fine label
- Spoofing: 2/2 fine labels
- Mirai: 3/3 fine labels

Incomplete:

- DDoS: 1/11 fine labels
- DoS: 1/4 fine labels

The generated coverage report is:

```text
artifacts/coverage/coverage_report.md
artifacts/coverage/missing_downloads.csv
```

## Remaining PCAP Folders To Download

Download these folders from the CICIoT2023 `PCAP` directory.

### DDoS

- `DDoS-ACK_Fragmentation`
- `DDoS-UDP_Flood`
- `DDoS-SlowLoris`
- `DDoS-ICMP_Flood`
- `DDoS-RSTFINFlood`
- `DDoS-PSHACK_Flood`
- `DDoS-HTTP_Flood`
- `DDoS-UDP_Fragmentation`
- `DDoS-TCP_Flood`
- `DDoS-SynonymousIP_Flood`

Already present:

- `DDoS-SYN_Flood`

### DoS

- `DoS-TCP_Flood`
- `DoS-HTTP_Flood`
- `DoS-UDP_Flood`

Already present:

- `DoS-SYN_Flood`

## Processing Steps After Download

Keep the downloaded `.pcap` files in `~/Downloads`, then run:

```bash
PYTHONPATH=src .venv/bin/python scripts/import_downloaded_pcaps.py
```

Process the newly downloaded DDoS/DoS captures. For first-pass development
validation, use Zeek-log sampling on large flood captures:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_pcaps.py \
  --sample-conn-lines 200000 \
  --only DDoS-ACK_Fragmentation \
  --only DDoS-UDP_Flood \
  --only DDoS-SlowLoris \
  --only DDoS-ICMP_Flood \
  --only DDoS-RSTFINFlood \
  --only DDoS-PSHACK_Flood \
  --only DDoS-HTTP_Flood \
  --only DDoS-UDP_Fragmentation \
  --only DDoS-TCP_Flood \
  --only DDoS-SynonymousIP_Flood \
  --only DoS-TCP_Flood \
  --only DoS-HTTP_Flood \
  --only DoS-UDP_Flood
```

Then rebuild the development split, vocabulary, model, and reports:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_dev_split_manifest.py \
  --sequence-root data/sequences \
  --output manifests/dev_chunk_manifest.csv \
  --val-ratio 0.2 \
  --seed 7

PYTHONPATH=src .venv/bin/python scripts/build_vocab.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --split train \
  --output artifacts/dev_vocab.json

PYTHONPATH=src .venv/bin/python scripts/train_hierarchical.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --epochs 10 \
  --batch-size 16 \
  --max-events 128 \
  --embedding-dim 32 \
  --hidden-dim 64 \
  --class-weighting balanced \
  --output-dir runs/hierarchical_dev

PYTHONPATH=src .venv/bin/python scripts/evaluate_hierarchical.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --checkpoint runs/hierarchical_dev/best_model.pt \
  --split val \
  --max-events 128 \
  --output-dir artifacts/evaluation_dev

PYTHONPATH=src .venv/bin/python scripts/extract_activity_patterns.py \
  --manifest manifests/dev_chunk_manifest.csv \
  --vocab artifacts/dev_vocab.json \
  --checkpoint runs/hierarchical_dev/best_model.pt \
  --split val \
  --max-events 128 \
  --output-dir artifacts/activity_patterns_dev
```

## Reporting Scope

After full fine-label coverage is processed, report:

- Coarse 8-class metrics.
- Fine-label metrics within the true coarse category.
- Path accuracy where both the coarse and fine labels are correct.
- Per-fine-label F1, especially for DDoS and DoS subtypes.
- Activity patterns and reconstructed traces for correctly predicted labels.

The project should still avoid claiming capture-disjoint generalization until
multiple independent captures per fine label are available or a grouped split
can be constructed.
