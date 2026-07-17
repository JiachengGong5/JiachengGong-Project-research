# 8-Class Expansion Plan

The current local smoke data covers three coarse classes:

- Benign: `Benign_Final`
- DDoS: `DDoS-SYN_Flood`
- Recon: `Recon-PortScan`

To run the first real 8-class coarse experiment, add one representative PCAP
folder for each missing coarse class:

| Coarse class | Download this PCAP folder first |
| --- | --- |
| DoS | `DoS-SYN_Flood` |
| Web-based | `SqlInjection` |
| Brute Force | `DictionaryBruteForce` |
| Spoofing | `DNS_Spoofing` |
| Mirai | `Mirai-udpplain` |

Place them under:

```text
data/raw/<folder_name>/<pcap_file>.pcap
```

Example:

```text
data/raw/DoS-SYN_Flood/DoS-SYN_Flood.pcap
data/raw/DNS_Spoofing/DNS_Spoofing.pcap
data/raw/DictionaryBruteForce/DictionaryBruteForce.pcap
data/raw/Mirai-udpplain/Mirai-udpplain.pcap
data/raw/SqlInjection/SqlInjection.pcap
```

After downloading, check coverage:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_data_coverage.py
```

If the files were downloaded into the browser's Downloads folder, import them
into the project:

```bash
PYTHONPATH=src .venv/bin/python scripts/import_downloaded_pcaps.py
```

Process the newly downloaded samples:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_pcaps.py \
  --only DoS-SYN_Flood \
  --only DNS_Spoofing \
  --only DictionaryBruteForce \
  --only Mirai-udpplain \
  --only SqlInjection
```

If a flood PCAP produces a very large Zeek `conn.log`, process a first
validation sample instead:

```bash
PYTHONPATH=src .venv/bin/python scripts/process_raw_pcaps.py \
  --only DoS-SYN_Flood \
  --sample-conn-lines 200000
```

Then rebuild training artifacts:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_manifest.py \
  --sequence-root data/sequences \
  --output manifests/smoke_manifest.csv

PYTHONPATH=src .venv/bin/python scripts/build_vocab.py \
  --manifest manifests/smoke_manifest.csv \
  --output artifacts/smoke_vocab.json
```

Finally run the smoke training command again. This will become an 8-class
coarse sanity check once all eight categories are present:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_smoke.py \
  --epochs 10 \
  --batch-size 16 \
  --max-events 128 \
  --output-dir runs/smoke_8class
```
