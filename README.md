# Identifying Activity Patterns in Intrusion Detection Data Sets

This repository is a research scaffold for learning intrusion activity patterns
directly from protocol-semantic event sequences in the
[CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

The primary path is:

```text
PCAP -> Zeek JSON protocol logs -> chronological semantic events -> LSTM -> class + salient activity subsequence
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

Protocol state alone is not sufficient to reliably distinguish every fine
attack type. For example, SQL injection and XSS can share the same HTTP method
and connection state. The hierarchical 33-attack extension should add an
end-to-end byte or character encoder for raw application fields such as HTTP
URI data.

See [docs/research_plan.md](docs/research_plan.md) for the full experimental
design and rationale.

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
features. During evaluation, every chunk from the same original capture must
remain in the same train, validation, or test partition.
