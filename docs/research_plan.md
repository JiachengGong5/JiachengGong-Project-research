# Research Plan: End-to-End Protocol Activity Sequence Learning

## 1. Core Research Question

Can an LSTM identify and classify normal and malicious IoT activity patterns
from chronological protocol-semantic events, without manually engineered flow
statistics?

This reframes intrusion detection from classifying independent flow-feature
rows to learning a language of network behavior:

```text
connection attempt -> protocol negotiation -> application operation -> result
```

The model learns order, repetition, and context directly from event sequences.

## 2. Important Technical Boundary

End-to-end learning does not mean "no representation decisions." PCAP bytes
still need to become model inputs, sequences need boundaries, and labels need
alignment. The defensible claim is:

> No manually aggregated traffic features are supplied to the model. The model
> receives ordered protocol-semantic events and learns useful activity
> representations jointly with the classification task.

Do not claim that Zeek performs no abstraction. Zeek is deliberately used as a
protocol parser that converts packets into interpretable semantic events.

## 3. Why Use CICIoT2023 PCAP Instead of Its CSV

CICIoT2023 provides both original PCAP traffic and CSV files intended for ML.
The official CSV schema includes engineered quantities such as flow duration,
rates, packet counts, average, standard deviation, variance, and covariance.
Those fields conflict with this project's research premise.

Use:

- **PCAP:** primary end-to-end experiments.
- **Official CSV:** traditional-feature baseline only.
- **Attack directory / capture metadata:** source of labels.

The dataset contains 33 attacks grouped into seven attack categories. Add
Benign to obtain the recommended first-stage 8-class problem.

## 4. Tool Roles

### Zeek: primary semantic event extractor

Zeek produces linked protocol logs such as `conn.log`, `http.log`, `dns.log`,
`ssl.log`, `ssh.log`, and `dhcp.log`. Useful direct semantic fields include
connection state/history, HTTP method, DNS query type/result, and TLS state.

### Wireshark/tshark: validation

Use Wireshark or tshark to inspect a small sample and verify that Zeek's event
sequence corresponds to the original packets. It should not be the main
feature generator.

### Suricata or Snort: rule-based baseline

Use one signature-based IDS as a baseline. Compare its alert recall and false
positive rate with the sequence model. Do not feed alert labels into the LSTM,
because that would leak expert rules into the proposed input.

### ntopng: optional demonstration

Use ntopng only for analyst-facing visualization if time permits. It is not
required for the core experiment.

## 5. Input Representation

One protocol log record is one event. One event is one LSTM time step.

Initial categorical fields:

| Log | Direct semantic fields |
| --- | --- |
| `conn.log` | `proto`, `service`, `conn_state`, `history`, responder port |
| `http.log` | `method`, HTTP version, exact status code |
| `dns.log` | query type, response code, opcode, rejected flag |
| `ssl.log` | TLS version, cipher, resumed, established, validation status |
| `ssh.log` | SSH version, authentication success, direction |
| `dhcp.log` | DHCP message types |
| `weird.log` | Zeek anomaly name |

Explicitly exclude IP addresses and origin ephemeral ports to reduce identity
leakage. Exclude flow duration, bytes, packet totals, rates, averages,
standard deviations, and all rolling statistics.

The model embeds each categorical field, sums the field embeddings to form an
event representation, and passes the ordered event representations to an
LSTM. This embedding operation is part of the neural network, not manual
traffic-statistic engineering.

## 6. Sequence Boundaries

Start with one ordered sequence per original PCAP capture. Very long captures
may be split into non-overlapping contiguous chunks solely for memory limits.
Never randomly split events or chunks from the same capture across partitions.

Later, sequences can be organized per IoT device when reliable device metadata
is available. Device-level grouping is preferable for measuring
generalization to unseen devices.

## 7. Activity Trace Reconstruction

Tracking related connections is part of the activity-pattern interpretation
layer. The main representation remains globally timestamp-sorted Zeek events
split into non-overlapping contiguous chunks, and the model is still trained
without leakage-prone identifiers such as IP addresses or Zeek `uid`.

After prediction, salient event spans are mapped back to the original Zeek
logs using timestamps, log type, and `uid` values. This reconstruction links
related `conn.log`, `dns.log`, `http.log`, `ssl.log`, and `weird.log` records
so the output is not just a class label, but a readable activity trace.

This design keeps model learning and trace reconstruction separate:

- The LSTM input receives only protocol-semantic event tokens.
- The interpretation layer may use Zeek `uid`, timestamps, and raw log fields
  to explain which connections and protocol records formed the identified
  activity.

See `docs/multi_connection_sessionization.md` for the trace-reconstruction
note.

## 8. Model and Pattern Identification

### Hierarchical classification model

```text
field token IDs
  -> learned field embeddings
  -> event embedding
  -> LSTM
  -> sequence representation
  -> coarse category head: Benign / DDoS / DoS / Recon / Web-based / Brute Force / Spoofing / Mirai
  -> category-specific fine head: attack type inside that category
```

The hierarchy matches the CICIoT2023 design: the official data set describes
33 attacks grouped into seven attack categories, and Benign traffic is added as
its own category. This is preferable to one flat 34-class classifier because
the model first learns broad activity behavior and then resolves fine-grained
differences within the relevant category.

During training, use the true category to select the fine head for the
fine-grained loss. During inference, first choose the highest-probability
coarse category, then choose the highest-probability label from that
category's fine head. Also report the coarse result independently, since a
fine-label mistake inside the correct category is less severe than confusing
Recon with DDoS.

### Activity-pattern output

A class label alone does not satisfy "identifying activity patterns." After
training, identify the most influential contiguous event spans using
gradient-based attribution or occlusion. Translate the selected event spans
back into readable protocol sequences.

Example output:

```text
Recon:
  conn/tcp/S0 -> conn/tcp/S0 -> conn/tcp/REJ -> conn/tcp/S0

DNS spoofing:
  dns/query/A -> dns/NOERROR -> weird/dns_* -> dns/query/A
```

These are examples of the desired output form, not hard-coded rules.

## 9. Experiments

### Primary experiments

1. Binary: Benign vs. malicious.
2. Eight-class: Benign plus seven attack categories.
3. Hierarchical: coarse 8-class plus category-specific fine attack labels.

The fine-grained Web-based experiment needs a raw application-content branch
for attacks whose protocol states are otherwise indistinguishable. Use a
character/byte encoder rather than manually designed payload features.

### Baselines

1. Majority-class baseline.
2. Suricata or Snort signature baseline.
3. LSTM using protocol semantics only.
4. Optional LSTM using protocol semantics plus direct per-event time gaps.
5. Published/traditional CICIoT2023 CSV-feature result, clearly marked as a
   non-comparable feature-engineering reference.

### Ablation studies

- Remove `conn_state` and `history`.
- Remove application-layer protocol events.
- Remove responder service port.
- Add direct event time gaps.
- Compare seen-device and unseen-device splits when metadata permits.
- Compare flat 34-class classification against hierarchical classification.

These ablations test representation choices without calculating statistical
traffic summaries.

## 10. Evaluation Design

Use capture/scenario/device-grouped splits, not random row or event splits.
Random splits can place nearly identical flood traffic from one attack run in
both training and test data and produce misleadingly high scores.

During early local development, a stratified chunk-level validation split may
be used only to debug the training loop and class-imbalance strategy when
there is only one downloaded capture per class. Do not report that score as
the final experimental result.

Report:

- Macro F1 as the primary metric.
- Per-class precision, recall, and F1.
- Balanced accuracy.
- False positive rate on Benign traffic.
- Confusion matrix.
- Hierarchical metrics: coarse accuracy/F1, fine accuracy/F1 within each true
  category, and path accuracy where both coarse and fine labels are correct.
- Inference throughput and sequence length.
- Examples of salient activity subsequences.

Accuracy alone is not sufficient because CICIoT2023 is highly imbalanced.

## 11. Risks and Mitigations

### Flood classes dominate

Many CICIoT2023 classes are floods. If sequence chunks are too short, the
model may not observe enough repetition. Evaluate several memory-safe chunk
lengths while keeping chunks non-overlapping.

### Label leakage

File names, IP addresses, device identity, and capture-specific artifacts can
reveal labels. Exclude them from model input and split by original capture.

### Zeek semantic coverage

Encrypted or unsupported traffic may only produce connection events. Document
the percentage of events from each Zeek log type and inspect representative
captures with Wireshark.

### Fine-grained Web attacks

HTTP method and connection state cannot reliably separate SQL injection, XSS,
command injection, and related attacks. Limit the initial claim to categories,
then add a learned raw URI/payload encoder for fine-grained classification.

### Multi-connection activity ambiguity

Some samples may involve related events across multiple connections or Zeek
logs, while unrelated background traffic may be interleaved. The baseline
preserves chronological order. If this ambiguity affects a class or error
case, handle it locally through inspection or an optional ablation, while
avoiding leakage-prone identifiers in model input.

## 12. Proposed Milestones

### Phase 1: Data and validity

- Download a small representative subset of CICIoT2023 PCAP.
- Generate Zeek JSON logs.
- Validate event sequences against Wireshark.
- Confirm label mapping and grouped split strategy.

### Phase 2: Primary model

- Train binary, 8-class, and hierarchical protocol-only LSTM models.
- Measure class imbalance and select a loss strategy.
- Compare with a signature IDS baseline.

### Phase 3: Pattern extraction and ablation

- Extract influential contiguous event spans.
- Run representation ablations.
- Test unseen-capture and, if possible, unseen-device generalization.

### Phase 4: Extension and report

- Optionally add a learned byte/character branch for 34-class classification.
- Complete error analysis, performance evaluation, report, and presentation.

## 13. Recommended Thesis Claim

> This project investigates whether ordered protocol-semantic events extracted
> from raw CICIoT2023 packet captures can support interpretable intrusion
> activity classification with an LSTM, without manually aggregated flow
> statistics. The model jointly learns event representations and temporal
> patterns, while salient subsequences provide analyst-readable descriptions
> of the activity patterns associated with each class.
