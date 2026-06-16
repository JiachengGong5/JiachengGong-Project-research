# Proposal Revision

## Project Title

Identifying Activity Patterns in Intrusion Detection Data Sets Through
End-to-End Protocol Event Sequence Learning

## 1. Project Purpose

Intrusion detection data sets contain large volumes of low-level network
traffic in which normal device behavior and malicious activity are
interleaved. Traditional intrusion detection research often converts this
traffic into manually engineered flow statistics, such as averages, standard
deviations, packet rates, and sliding-window aggregates. This process is
computationally expensive, can discard temporal structure, and requires
researchers to decide in advance which statistics represent an attack.

The purpose of this project is to investigate whether high-level normal and
malicious activity patterns can instead be learned directly from ordered
protocol-semantic events extracted from raw packet captures. The project will
use the CICIoT2023 data set, which contains original PCAP traffic for 33 IoT
attacks grouped into seven attack categories. Protocol analysis tools such as
Zeek will convert packets into interpretable events, including connection
states, protocol operations, HTTP methods, DNS operations, and TLS states. An
LSTM will then learn useful representations and temporal activity patterns
jointly with the classification task.

## 2. Main Objectives

1. Develop a reproducible pipeline that converts CICIoT2023 PCAP files into
   chronological protocol-semantic event sequences without calculating
   manually aggregated traffic statistics.
2. Design and train an LSTM-based model that learns event representations and
   temporal activity patterns end to end.
3. Classify sequences through a hierarchical model: first as Benign or one of
   the seven CICIoT2023 attack categories, then as a fine-grained attack type
   within the predicted category.
4. Identify influential contiguous event subsequences and translate them into
   analyst-readable activity patterns.
5. Evaluate whether the learned sequence representation improves
   interpretability and generalization while maintaining competitive
   classification performance.
6. Compare the proposed model with a signature-based IDS baseline and
   traditional CICIoT2023 feature-based results.

## 3. Proposed Methods

### Data preparation and protocol event extraction

The primary experiments will begin with CICIoT2023 PCAP files rather than the
data set's precomputed CSV features. Zeek will parse the PCAP traffic and
produce protocol logs such as connection, HTTP, DNS, TLS, SSH, and DHCP logs.
Wireshark or tshark will be used to validate representative event sequences
against the original packets.

Each supported Zeek log record will become one chronological event. Direct
protocol semantics, such as connection state, TCP history, HTTP method, DNS
operation, and TLS state, will be preserved as categorical inputs. IP
addresses, capture identifiers, and traditional flow summaries such as
duration, byte totals, packet counts, rates, averages, standard deviations,
and sliding-window aggregates will not be supplied to the proposed model.

### End-to-end hierarchical sequence learning

The model will use learned embeddings to represent the categorical fields
within each event. An LSTM will process the ordered event representations and
learn transitions, repetition, and longer temporal dependencies associated
with normal and malicious activities. The learned sequence representation will
feed a coarse classification head for Benign and the seven attack categories,
as well as category-specific fine classification heads for the attack types
inside each category.

The initial study will validate binary and eight-class classification before
training the hierarchical model. Fine-grained Web-based attacks may require a
learned byte- or character-level encoder for raw application fields. This
extension remains end to end and avoids manually designed payload features.

### Activity-pattern identification

To produce more than a class label, the project will apply model attribution
or occlusion analysis to locate the most influential contiguous event spans.
The selected spans will be translated back into readable protocol-event
sequences. These sequences will describe the activity patterns that most
strongly support each prediction.

### Experimental validation

Training, validation, and test partitions will be grouped by original capture,
scenario, and device when metadata permits. Events or chunks from one capture
will never be randomly distributed across partitions. This strategy reduces
the risk of leakage from highly repetitive attack traffic.

The primary metric will be macro F1. The evaluation will also report
per-class precision, recall, and F1, balanced accuracy, false positive rate,
confusion matrices, inference throughput, and representative identified
activity patterns. Ablation experiments will measure the contribution of
connection states, application-layer events, responder service ports, and
direct per-event timing.

## 4. Novel Methodological Elements

- The proposed hierarchical model learns from ordered protocol-semantic events instead of
  manually aggregated flow statistics.
- Protocol fields and temporal patterns are learned jointly by the neural
  model rather than selected through a manual feature-engineering pipeline.
- The project evaluates generalization with capture- and device-grouped data
  splits rather than random event-level splits.
- The system identifies influential event subsequences to provide
  analyst-readable descriptions of learned intrusion activity patterns.

## 5. Anticipated Outcomes and Impact

The project is expected to produce:

1. A reproducible PCAP-to-protocol-event processing pipeline.
2. An LSTM model for binary, category-level, and hierarchical IoT intrusion
   activity classification.
3. A method for extracting and presenting influential protocol activity
   subsequences.
4. A rigorous evaluation comparing sequence learning, signature-based
   detection, and traditional feature-based approaches.
5. A documented analysis of when protocol state is sufficient and when raw
   application content is required for fine-grained attack classification.

The practical impact is an intrusion detection approach that presents network
behavior as understandable activity sequences rather than isolated feature
rows. The research contribution is an evaluation of how far end-to-end
protocol event sequence learning can replace manual traffic feature
engineering in a large-scale IoT attack data set.

## 6. Revised Work Plan

### Phase 1: Literature review and data validation

- Review sequence-based intrusion detection, end-to-end traffic learning, and
  protocol-semantic representation methods.
- Analyze CICIoT2023 PCAP organization, labels, class balance, and metadata.
- Configure Zeek and validate extracted events with Wireshark.

**Output:** literature review, data audit, and validated event schema.

### Phase 2: Event pipeline and experimental split

- Implement the PCAP-to-event pipeline.
- Define reproducible capture- and device-grouped partitions.
- Measure event coverage and verify the absence of prohibited aggregate
  features and label leakage.

**Output:** protocol-event sequence data set and preprocessing documentation.

### Phase 3: LSTM model and primary experiments

- Implement and train the protocol-event LSTM.
- Evaluate binary, eight-class, and hierarchical classification.
- Compare with a signature-based IDS and traditional feature-based results.

**Output:** trained models and primary experimental results.

### Phase 4: Pattern identification and ablation

- Extract influential contiguous event subsequences.
- Run representation and timing ablations.
- Analyze errors, false positives, and generalization limitations.

**Output:** interpretable activity-pattern examples and ablation results.

### Phase 5: Final evaluation and reporting

- Optionally add an end-to-end raw application-content encoder for
  fine-grained classification.
- Complete the final report, reproducibility package, and presentation.

**Output:** final research report, code, and presentation.
