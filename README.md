# GUI-Intent: Multimodal Behavioral Pipeline

> Predict what users want to do next. Algorithmically detect where they're struggling.

---

## Overview

GUI-Intent is an end-to-end machine learning pipeline built on a **7.87 GB multimodal dataset** of 3,167 human-computer interaction tasks. It combines spatial-temporal data engineering, sequence modeling, and supervised classification to solve two complementary problems:

1. **Intent Prediction** — Given a user's last N actions, predict their next action
2. **Friction Diagnostics** — Identify UI paths where users are confused or struggling

The dataset spans **294 unique websites** and **173 desktop applications**, with multimodal coverage including parquet event logs, screenshots, and DOM snapshots.

---

## Architecture

```
[Raw Parquet Data]
       │
       ▼
┌─────────────────────────────┐
│  STAGE 1                    │
│  Spatial-Temporal Feature   │
│  Engine                     │
│  - GMM coordinate clusters  │
│  - Temporal delta features  │
│  - DOM text embeddings      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 2                    │
│  Next-Action Intent Model   │
│  (PyTorch GRU/Transformer)  │
│  - Tokenized clickstreams   │
│  - Multi-class prediction   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 3                    │
│  Cognitive Friction Engine  │
│  (LightGBM Classifier)      │
│  - Backtracking index       │
│  - Stall rate analysis      │
│  - Path tortuosity          │
└─────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| Total Tasks | 3,167 |
| Parquet Size | 7.87 GB |
| Total Dataset | 49.2 GB |
| Browser Tasks | 2,220 (70.1%) |
| Computer Tasks | 947 (29.9%) |
| Unique Websites | 294 |
| Unique Applications | 173 |

**Difficulty Distribution:**

| Label | Count | Share |
|---|---|---|
| Easy | ~2,515 | 79.4% |
| Medium | ~529 | 16.7% |
| Hard | ~123 | 3.9% |

**Modality Coverage:**

| Modality | Coverage | Size |
|---|---|---|
| Interaction Events (Parquet) | 100% | 7.87 GB |
| Videos | 100% | 16.9 GB |
| DOM Snapshots | 55.8% | 24.4 GB |
| Screenshots | 42.6% | embedded in parquet |

---

## Pipeline Details

### Stage 1 — Spatial-Temporal Feature Engine

Raw `(x, y)` coordinates are meaningless across different screens and resolutions. This stage transforms them into semantically meaningful features.

**Spatial Quantization**
- Fit a **Gaussian Mixture Model (GMM)** per domain across all click coordinates
- Cluster coordinates into `K` functional UI zones (search bar, checkout button, nav menu, etc.)
- Each click is mapped to a discrete `spatial_cluster_id` instead of raw pixels

**Temporal Features**
- `Δt` — time delta between consecutive actions
- Mouse velocity vectors over movement sequences
- Action frequency per session window

**DOM Text Processing**
- Parse HTML structure from DOM snapshots (55.8% coverage)
- Embed text labels and HTML tags near the interaction point using **TF-IDF** or a **sentence-transformer**
- Learned null-embedding fallback for tasks without DOM snapshots (~44%)

Each action is represented as a concatenated vector:

```
v = [Action Type Group | Spatial Cluster ID | Temporal Deltas | DOM Context Embedding]
```

---

### Stage 2 — Next-Action Intent Model

Frames user interaction history as a **sequence-to-sequence prediction problem** — treating clickstreams exactly like words in a sentence.

**Architecture:** PyTorch GRU (preferred over Transformer for short sequences typical of easy tasks) with a multi-class output head over action token vocabulary.

**Input:** Last `N` action vectors from Stage 1  
**Output:** Probability distribution over next action tokens

```python
# Conceptual input shape
# (batch_size, sequence_length, feature_dim)
# feature_dim = action_type + spatial_cluster + temporal_deltas + dom_embedding
```

**Why GRU over Transformer?**  
79.4% of tasks are labeled Easy, meaning sequences are typically short. Attention mechanisms provide limited benefit at short sequence lengths and add unnecessary compute. A GRU is faster to train and converges more reliably here.

---

### Stage 3 — Cognitive Friction Engine

Uses the dataset's built-in difficulty labels (`Easy` / `Medium` / `Hard`) as a proxy for cognitive load and interface friction.

**Model:** LightGBM gradient-boosting classifier

**Engineered Features:**

| Feature | Definition |
|---|---|
| Backtracking Index | Click an element → go back → click a different one, counted per short time window |
| Stall Rate | Frequency of `Δt` spikes significantly above per-app median (indicates confusion) |
| Action Density | Total interactions ÷ minimum path length required to complete the task |
| Path Tortuosity | Deviation of actual mouse path from optimal straight-line path |

**Class Imbalance Handling:**  
The dataset has a severe ~20:1 imbalance (Easy vs Hard). The pipeline uses:
- SMOTE oversampling on the minority class during training
- `scale_pos_weight` in LightGBM config
- Macro-F1 as the primary optimization metric (ROC-AUC reported as secondary)

**Result:** ROC-AUC of **0.82** on held-out test split

---

## Project Structure

```
gui-intent/
│
├── data/
│   ├── raw/                  # Parquet files from dataset
│   └── processed/            # Feature-engineered outputs
│
├── stage1_features/
│   ├── gmm_clustering.py     # Spatial quantization via GMM
│   ├── temporal_features.py  # Delta time, velocity, frequency
│   └── dom_embeddings.py     # TF-IDF / sentence-transformer on DOM
│
├── stage2_intent/
│   ├── dataset.py            # Clickstream tokenization & batching
│   ├── model.py              # PyTorch GRU architecture
│   └── train.py              # Training loop
│
├── stage3_friction/
│   ├── feature_engineering.py  # Backtracking, stall rate, tortuosity
│   ├── train_lgbm.py           # LightGBM training with SMOTE
│   └── evaluate.py             # ROC-AUC, macro-F1, confusion matrix
│
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
│
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/gui-intent.git
cd gui-intent
pip install -r requirements.txt
```

**Core dependencies:**

```
torch
lightgbm
scikit-learn
imbalanced-learn      # SMOTE
sentence-transformers
pandas
pyarrow               # Parquet reading
numpy
```

---

## Usage

**Stage 1 — Generate features:**
```bash
python stage1_features/gmm_clustering.py --input data/raw/ --output data/processed/
python stage1_features/temporal_features.py --input data/processed/
python stage1_features/dom_embeddings.py --input data/processed/
```

**Stage 2 — Train intent model:**
```bash
python stage2_intent/train.py \
  --data data/processed/ \
  --seq_len 20 \
  --hidden_dim 256 \
  --epochs 30
```

**Stage 3 — Train friction classifier:**
```bash
python stage3_friction/train_lgbm.py \
  --data data/processed/ \
  --smote true \
  --eval_metric macro_f1
```

---

## Results

| Stage | Task | Metric | Score |
|---|---|---|---|
| Stage 2 | Next-action prediction | Top-3 Accuracy | TBD |
| Stage 3 | Friction classification | ROC-AUC | 0.82 |
| Stage 3 | Friction classification | Macro-F1 | TBD |

---

## Key Design Decisions

**Why GMM over simple normalization for coordinates?**  
Normalizing `(x, y)` to `[0, 1]` still loses semantic meaning — clicking the top-left of Amazon and the top-left of Google are structurally different interactions. GMM clusters capture *functional zones* (search bar, nav, cart), not just relative positions.

**Why not use the video modality?**  
The 16.9 GB of MP4 recordings are intentionally excluded. The parquet event logs already contain precise timestamped action data. Video-based optical flow would add significant compute for marginal gain given the structured event data available.

**Why GRU over Transformer for Stage 2?**  
Short sequences (most tasks are Easy difficulty) make attention less impactful. GRUs train faster, converge more reliably at this sequence length, and are sufficient for the prediction task.

---

## Dataset Source

[Paradigm Shift AI — GUI Interaction Dataset](https://huggingface.co/datasets/ParadigmShiftAI/gui-interaction-dataset)  
Created for advancing computer-use AI agent research.

---

## License

MIT