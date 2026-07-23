# Progress Report

## Identifying Activity Patterns in Intrusion Detection Data Sets

**Student:** Jiacheng Gong

**Course:** ECE 9065 M.Eng Project

**Supervisor:** Jagath

**Date:** July 22, 2026
**Repository:** https://github.com/JiachengGong5/JiachengGong-Project-research

## Executive Summary

This project investigates whether IoT intrusion activity can be identified from ordered protocol-semantic events without supplying manually aggregated traffic statistics to the model. The primary data source is the original CICIoT2023 PCAP collection. Zeek parses each packet capture into protocol logs, which are converted into chronological categorical events. A hierarchical LSTM first predicts Benign or one of seven attack categories and then predicts an attack subtype within the selected category.

The implemented pipeline covers PCAP import, Zeek processing, event generation, manifest and vocabulary construction, hierarchical training, evaluation, error diagnosis, salient event-span extraction, and Zeek trace reconstruction. A July 22 sequence-coverage improvement replaced prefix truncation with 256-event, contiguous, non-overlapping segments and normalized the joint fine loss across samples. On the resulting 10,340-segment development validation set, the model achieved 91.77% coarse accuracy, 81.26% coarse Macro F1, 69.74% conditional fine-label Macro F1, and 75.98% full hierarchical path accuracy. These are within-capture development results, not final unseen-capture results.

The current checkpoint contains all 33 official attack types plus Benign, for 34 model output labels. The most important limitations are severe class imbalance, one local PCAP per label, within-capture chunk splitting, sampled development sequences for very large flood captures, and insufficient application-content information for fine-grained Web attacks.

## 1. Research Question and Scope

The central research question is:

> Can an LSTM learn normal and malicious IoT activity patterns from chronological protocol-semantic events extracted from raw packet captures, without manually engineered flow statistics?

The defensible end-to-end claim is from protocol-semantic events to learned activity representations and predictions. Zeek still performs deterministic protocol parsing, and sequence boundaries remain explicit design choices. The project does not claim zero preprocessing or direct learning from every raw packet byte.

The model input deliberately excludes IP addresses, Zeek UIDs, flow duration, byte totals, packet counts, rates, averages, standard deviations, variances, covariances, and rolling-window aggregates. It includes direct protocol semantics such as transport protocol, service, responder service port, connection state, TCP history, HTTP method and status, DNS operation and result, TLS state, SSH state, DHCP message type, and Zeek anomaly names.

## 2. Existing Approaches and Project Difference

### 2.1 Precomputed flow-feature classification

The CICIoT2023 authors provide CSV data containing engineered flow and packet statistics. Typical benchmark classifiers operate on independent rows containing quantities such as flow duration, rates, packet totals, averages, standard deviations, and variance. This is efficient for conventional tabular machine learning, but the representation is selected before model training and can discard the original protocol-event order.

### 2.2 Signature-based intrusion detection

Tools such as Suricata and Snort match traffic against expert-written signatures. They provide operationally useful alerts and are suitable baselines, but their detection logic is manually specified rather than learned from the activity sequence.

### 2.3 Sequence learning over flow records

Another approach applies recurrent or attention models to sequences of flow-feature rows. Although the classifier is sequential, each row may still contain manually aggregated statistics. The temporal model therefore learns over an engineered representation rather than direct protocol operations and states.

### 2.4 Raw-byte or payload learning

Raw packet-byte models reduce protocol parsing assumptions but require larger models and more data, and their learned patterns can be difficult to explain to a network analyst. Payload learning is also affected by encryption and privacy constraints.

### 2.5 Difference of this work

This project combines five design choices:

1. Primary experiments begin with original PCAP rather than the official feature CSV.
2. One Zeek protocol record becomes one ordered model time step.
3. Manually aggregated traffic statistics and identifying addresses are excluded.
4. The classifier follows the dataset hierarchy through an eight-category head and category-specific fine heads.
5. Important event spans are mapped back to timestamp- and UID-linked Zeek records to produce an analyst-readable activity trace.

The project therefore focuses on learned temporal activity patterns while preserving protocol-level interpretability.

## 3. Work Completed and Contribution Scope

External resources used by the project are CICIoT2023, Zeek, PyTorch, Python, and standard scientific software. They are not presented as original contributions.

My direct project work included defining and refining the research direction, obtaining and organizing the local data, configuring and operating the Zeek pipeline, running the training and evaluation experiments, inspecting intermediate data and errors, integrating the activity-tracking requirement into the interpretation design, and preparing the technical documentation and demonstration.

Development assistance disclosure: I used an AI coding assistant to help scaffold and revise parts of the Python implementation and documentation. I supplied the project requirements, made the research and evaluation decisions, managed the local data, executed the software and experiments, reviewed intermediate and final outputs, and interpreted the results. I remain responsible for verifying the submitted code and report, and I will describe this assistance in accordance with course and university policy.

The project work completed to date includes:

- selecting and refining the research question around protocol-event learning rather than manual traffic statistics;
- downloading, organizing, and checking local CICIoT2023 PCAP coverage;
- configuring and running Zeek processing on approximately 40 GB of raw PCAP data;
- defining a direct protocol-semantic event schema and prohibited-field boundary;
- building chronological, contiguous, non-overlapping sequence chunks and a split-preserving event re-chunking stage;
- defining the coarse/fine label hierarchy and normalization of dataset folder names;
- building reusable manifest, vocabulary, PyTorch dataset, collation, and training components;
- implementing and executing hierarchical LSTM training with coarse/fine class weighting, sample-normalized fine loss, and configurable task weights;
- producing confusion matrices, per-class precision/recall/F1, balanced accuracy, path accuracy, and prediction-level error records;
- running three repeated fine-stratified development splits for stability analysis;
- diagnosing coarse-gate errors, fine-head errors, low-support classes, and severe data scarcity;
- extracting influential contiguous spans through occlusion;
- reconstructing related Zeek connection and application-protocol records through timestamps and UIDs;
- adding automated tests and research/design documentation; and
- publishing the implementation to the Git repository.

The in-person demonstration will execute and explain the repository components, data path, model outputs, and interpretation results.

## 4. Implemented System

The processing and learning workflow is:

```text
CICIoT2023 PCAP
  -> Zeek JSON protocol logs
  -> chronological categorical protocol events
  -> contiguous memory-safe sequence chunks
  -> split-preserving 256-event segments
  -> training manifest and token vocabulary
  -> learned field embeddings
  -> LSTM sequence encoder
  -> 8-category coarse prediction
  -> category-specific fine prediction
  -> full hierarchical path prediction
  -> occlusion-based important event span
  -> Zeek activity trace reconstruction
```

### 4.1 Event preparation

The event converter merges supported Zeek logs by timestamp. It currently reads connection, HTTP, DNS, TLS, SSH, DHCP, and anomaly records. Events retain direct categorical states and operations. Long captures are split only into non-overlapping contiguous chunks for memory limits; no sliding statistical window is calculated.

### 4.2 Hierarchical LSTM

Each categorical token is mapped to a learned 32-dimensional embedding. Event fields are combined into event representations, and a one-layer LSTM with hidden dimension 64 processes each 256-event segment. The sequence representation feeds an eight-category coarse head and a fine head for each category. Balanced class weighting is applied at both levels. The current main run uses coarse loss weight 1.0 and fine loss weight 0.25; fine losses are normalized across samples instead of summing one equally weighted mean loss per category present in a batch.

During training, the true category selects the fine head for the fine loss. During inference, the predicted category selects the fine head. This produces two interpretable error mechanisms: a coarse-gate error selects the wrong category, while a fine-head error confuses sibling attacks inside the correct category.

### 4.3 Activity interpretation and connection tracking

After prediction, the interpretation script occludes contiguous event spans and measures the reduction in target path probability. The most influential span is mapped back to the original Zeek log directory. Timestamps identify the local context, while Zeek UIDs link connection, DNS, HTTP, TLS, and anomaly records.

Identifiers are used only after prediction for explanation. IP addresses and UIDs are not supplied to the LSTM, reducing label leakage while allowing multi-log and multi-connection activity traces to be displayed.

## 5. Data and Experimental Configuration

The official dataset describes 33 attacks in seven attack categories. The current local experiment contains 34 PCAP files representing all 33 official attacks plus Benign, and the retrained checkpoint therefore has the complete 34-output-label hierarchy.

Current processed development data:

| Category | Fine labels | Parent chunks | 256-event segments |
| --- | ---: | ---: | ---: |
| Benign | 1 | 99 | 1,584 |
| DDoS | 12 | 1,494 | 23,813 |
| DoS | 4 | 1,119 | 17,874 |
| Recon | 5 | 329 | 5,215 |
| Web-based | 6 | 12 | 152 |
| Brute Force | 1 | 4 | 52 |
| Spoofing | 2 | 64 | 998 |
| Mirai | 3 | 107 | 1,695 |
| **Total** | **34 output labels** | **3,228** | **51,383** |

The improved manifest contains 41,043 training segments and 10,340 validation segments. All child segments inherit the split of their parent chunk, and the re-chunking audit confirms that all 13,149,952 input events are retained exactly once. Segment counts are not independent-capture counts: the data still come from 34 local PCAP files, generally one PCAP per label. Large flood captures use sampled Zeek connection logs for the current memory-safe development sequences, while complete local Zeek outputs remain stored outside Git.

## 6. Results

### 6.1 Main hierarchical run

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Coarse accuracy | 91.77% | Overall correctness across eight categories |
| Coarse balanced accuracy | 85.19% | Macro average of category recall |
| Coarse Macro F1 | 81.26% | Equal category weighting in F1 |
| Fine accuracy given true category | 82.65% | Fine head evaluated with the correct category supplied |
| Fine balanced accuracy given true category | 72.99% | Macro fine-label recall under the true category |
| Fine Macro F1 given true category | 69.74% | Macro fine-label F1 under the true category |
| Full hierarchical path accuracy | 75.98% | Both coarse and fine prediction are correct |

The conditional fine score must not be reported as complete system accuracy because it uses the true category to select the fine head. Full path accuracy is the stricter end-to-end hierarchical metric.

If all attack categories are merged into one Malicious class, the improved eight-class predictions yield 98.48% binary accuracy, 99.98% attack precision, 98.45% attack recall, 99.21% attack F1, and a 0.63% false-positive rate on the 320 Benign validation segments. These values are a binary projection of the hierarchical predictions, not a separately trained binary model.

### 6.2 Improvement ablation

The previous 128-event prefix baseline achieved 85.08% coarse accuracy, 65.63% balanced accuracy, and 59.99% Macro F1. Re-chunking exposes 100% rather than 3.1% of the available protocol events to training. A one-epoch coarse-only probe reached 88.25% accuracy and 68.56% Macro F1, while the one-epoch equally weighted joint probe reached only 58.31% Macro F1. This showed that the unnormalized fine objective was interfering with the shared encoder. The final 1.0/0.25 task weighting and sample-normalized fine loss improved coarse Macro F1 by 21.27 percentage points while preserving and improving the hierarchical path result.

### 6.3 Category-level behavior

Coarse recall is 99.38% for Benign, 91.26% for DDoS, 92.98% for DoS, 86.55% for Recon, 65.67% for Web-based, 50.00% for Brute Force, 98.97% for Spoofing, and 96.73% for Mirai. DDoS-to-DoS errors fell from 15.00% to 5.31% of DDoS validation samples, although DoS-to-DDoS errors increased from 4.91% to 6.46%. Web and Brute Force results remain based on correlated segments from one local capture per label and must not be interpreted as independent generalization evidence.

### 6.4 Stable fine-label evidence

The improved run provides stronger conditional fine-label evidence for Benign, DDoS-ACK Fragmentation, DDoS-UDP Flood, DDoS-SlowLoris, DDoS-RSTFIN, DDoS-HTTP, DDoS-SynonymousIP, all four DoS fine labels, Recon-Host Discovery, both Spoofing labels, and Mirai-UDPPlain.

DDoS-ICMP Fragmentation now contributes 89 training and 32 validation segments after re-chunking, but these all derive from one PCAP. Its conditional fine-head precision, recall, and F1 are 47.37%, 56.25%, and 51.43%, and its full-path recall remains 0% because its validation segments fail at the coarse gate. Re-chunking provides more temporal coverage but does not create independent behavior diversity.

Important recurring errors include DDoS-SYN versus DoS-SYN, DDoS-HTTP versus DoS-HTTP, DDoS-PSHACK versus DDoS-TCP, and confusion among Recon OS Scan, Port Scan, and Vulnerability Scan.

### 6.5 Repeated development splits

The earlier 128-event baseline was repeated with three fine-stratified seeds and produced path accuracies of 66.31%, 64.00%, and 65.54%. The improved full-event configuration has not yet been repeated across seeds. The old runs remain useful as baseline stability evidence, but they must not be presented as stability evidence for the new 91.77% model.

### 6.6 Activity-pattern output

The improved interpretation output contains 25 correctly predicted fine-label examples with salient event spans and reconstructed Zeek traces. Every extracted span retains its parent chunk and global event offsets, and all 25 examples successfully map back to Zeek records. DDoS-ICMP Fragmentation is still absent because it has no correct full-path validation prediction. These patterns are learned model sensitivities rather than hard-coded detection signatures.

## 7. Limitations

1. **Within-capture validation.** Each current label has one local PCAP, so chunks from the same capture can appear in training and validation. Current scores are development results only.
2. **Class imbalance and independent-capture scarcity.** Re-chunking increases segment counts but each label generally still has one local PCAP. Class weighting and additional segments cannot create independent device, scenario, or capture diversity.
3. **Development sampling.** Several large flood sequences were built from at most 200,000 Zeek connection records for memory-safe iteration.
4. **Limited Web content.** Protocol state alone cannot reliably separate SQL Injection, XSS, Command Injection, and related attacks. A learned URI/payload character or byte encoder is still required.
5. **Segment boundary context.** The 256-event re-chunking removes prefix event loss, but an activity crossing a segment boundary is still divided. No overlapping or statistical windows are used.
6. **Post-hoc session reconstruction.** The model sees chronological chunks rather than an explicit session graph. UID-based linkage is currently used only in the interpretation layer.
7. **Incomplete baselines and ablations.** A signature IDS baseline, formal majority baseline table, flat-versus-hierarchical comparison, representation ablations, inference throughput, and formal Wireshark validation record remain future work.
8. **Attribution boundary.** Occlusion identifies model-sensitive spans but does not prove that a span is the unique or causal source of an attack.

## 8. Git Repository and Reproducibility

The repository records the following main milestones:

| Date | Commit | Recorded work |
| --- | --- | --- |
| 2026-06-16 | `87ee240` | Initial research plan, event schema, LSTM scaffold, preprocessing CLI, and tests |
| 2026-06-25 | `2eb182a` | Ignore generated Zeek sample outputs |
| 2026-07-16 | `2658711` | Hierarchical training/evaluation pipeline, full current label processing, repeated splits, error analysis, trace reconstruction, documentation, manifests, and tests |

The July 16 commit adds 8,294 lines across 32 changed files. The Git history is less incremental than it should be because local work was consolidated before being pushed. Future experiments and reporting changes should be committed in smaller, dated units.

Raw PCAP, Zeek logs, generated sequences, model checkpoints, and large experiment artifacts are intentionally excluded from Git because the local directories exceed 80 GB. The repository includes the reproducible source code, development manifests, label definitions, tests, and research documentation. Selected quantitative results are reproduced in this report.

The automated test suite currently contains 15 tests covering event-field exclusion, timestamp ordering, contiguous chunking, split-preserving re-chunking and event conservation, hierarchy mapping, dataset collation, model forward/loss behavior, and Zeek trace reconstruction.

## 9. Immediate Next Work

The next work should be completed in this order:

1. Obtain multiple captures per label or define a defensible capture/scenario/device-disjoint test.
2. Add a majority baseline and a Suricata or Snort signature baseline.
3. Repeat the improved 256-event configuration across multiple split and training seeds.
4. Run two focused ablations: remove connection state/history and remove application-layer events.
5. Integrate session-aware or activity-level sequence construction for captures with interleaved connections.
6. Add an end-to-end character or byte branch for Web URI/payload content.
7. Measure inference throughput and document representative Zeek-versus-Wireshark validation cases.

## 10. Conclusion

The project has progressed from a research concept to a working and testable software pipeline that processes original PCAP files, covers all 33 official attacks plus Benign, learns hierarchical activity classifications from ordered protocol semantics, and produces interpretable Zeek activity traces without manually aggregated traffic statistics. The results demonstrate method feasibility and expose meaningful error mechanisms. They do not yet support a final claim of unseen-capture or real-world generalization. Improving the evaluation split and adding baselines and ablations are necessary for the final report.

## References

1. E. C. P. Neto et al., “CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,” *Sensors*, 23(13), 5941, 2023. https://doi.org/10.3390/s23135941
2. Canadian Institute for Cybersecurity, “CIC IoT Dataset 2023.” https://www.unb.ca/cic/datasets/iotdataset-2023.html
3. Zeek Project, “Zeek Documentation.” https://docs.zeek.org/
4. Open Information Security Foundation, “Suricata Documentation.” https://docs.suricata.io/
5. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 9(8), 1735-1780, 1997. https://doi.org/10.1162/neco.1997.9.8.1735
