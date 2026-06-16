# Hierarchical Classification Design

## Why Hierarchical Classification

CICIoT2023 has a natural label hierarchy: attacks are grouped into seven
semantic categories, and Benign traffic can be treated as an additional
top-level category. A flat 34-class classifier forces the model to compare
every fine attack against every other fine attack, even when the real
distinction is first category-level behavior.

Hierarchical classification matches the problem structure:

```text
Level 1: Benign / DDoS / DoS / Recon / Web-based / Brute Force / Spoofing / Mirai
Level 2: attack type inside the selected category
```

This is especially useful because some fine labels are very similar across the
full data set. For example, flood attacks appear under both DDoS and DoS, and
several Web-based attacks can share similar protocol states.

## Model Structure

The model uses one shared protocol-sequence encoder:

```text
Zeek event tokens
  -> field embeddings
  -> event embeddings
  -> LSTM sequence encoder
```

The shared sequence embedding is then sent to multiple prediction heads:

```text
sequence embedding
  -> coarse head: 8 categories
  -> DDoS fine head
  -> DoS fine head
  -> Recon fine head
  -> Web-based fine head
  -> Brute Force fine head
  -> Spoofing fine head
  -> Mirai fine head
```

Only the fine head belonging to the true category is used for the fine-grained
training loss.

## Training Objective

For each sequence, training uses two labels:

```text
coarse_label = Recon
fine_label_within_category = Port scan
```

The loss is:

```text
total_loss = coarse_cross_entropy + fine_cross_entropy_for_true_category
```

This lets the model learn broad behavior first while still learning
fine-grained distinctions inside each category.

## Inference

Inference is a two-step path:

```text
1. Predict the coarse category.
2. Use that category's fine head to predict the specific attack.
```

Example:

```text
coarse prediction: Recon
Recon fine head: Port scan
final path: Recon -> Port scan
```

For analysis, also keep the top alternatives:

```text
coarse top-3: Recon, DDoS, DoS
Recon fine top-3: Port scan, Host discovery, OS scan
```

## Metrics

Report three levels of performance:

- Coarse category Macro F1.
- Fine-label Macro F1 within each true category.
- Path accuracy, where both the coarse category and fine label must be correct.

Also report a flat 34-class baseline if time permits. The comparison can show
whether the hierarchy improves fine-grained recognition or mainly improves
interpretability and error analysis.

## Recommended Experimental Order

1. Binary: Benign vs. malicious.
2. Coarse: Benign plus seven attack categories.
3. Hierarchical: coarse category plus fine attack type.
4. Optional flat 34-class baseline for comparison.
5. Optional raw URI/payload encoder for hard Web-based fine labels.

