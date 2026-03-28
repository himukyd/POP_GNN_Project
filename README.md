# CUDA Graph Coarsening for Scalable GNN Training

A GPU-accelerated graph coarsening pipeline that reduces large graphs before training Graph Neural Networks (GNNs), enabling faster training on large-scale datasets without significant accuracy loss.

---

## Overview

Training GNNs on large graphs (millions of nodes/edges) is computationally expensive. This project tackles that by **coarsening** the graph — merging similar nodes into super-nodes — before training, dramatically reducing the graph size while preserving structural properties.

The pipeline consists of two stages:

1. **`him_coarsening.cu`** — A CUDA kernel that performs iterative heavy-edge matching to coarsen the graph on the GPU.
3. **`preprocess.py`** — A Python script that loads graph datasets (OGB, Reddit, Yelp, IGB, etc.), generates CSR input files, runs the CUDA coarsening binary, and aggregates node features, labels, and masks onto the coarsened graph.
4. **`train.py`** — Trains a GraphSAGE model on the coarsened graph and evaluates it on the original graph by mapping predictions back via the vertex mapping file.

---

## Pipeline Architecture

```
Original Graph (DGL)
        │
        ▼
  preprocess.py
  ├── Load dataset (OGB / Reddit / Yelp / IGB)
  ├── Clean graph (to_simple + to_bidirected)
  ├── Export CSR files (row.txt, column.txt, input_coarsening_weights.txt)
  └── Run roy_coarsening (CUDA)
        │
        ▼
  CoarsedGraphOutput/
  ├── coarse_row.txt
  ├── coarse_column.txt
  ├── coarse_weight.txt
  └── final_vertex_mapping.txt
        │
        ▼
  preprocess.py (aggregation)
  ├── Aggregate features (mean / sum / max)
  ├── Aggregate labels   (mode / majority / first / max / min)
  └── Aggregate masks    (any / all / majority)
        │
        ▼
  <dataset>.dgl  (coarsened graph)
        │
        ▼
  train.py
  ├── Train GraphSAGE on coarsened graph
  └── Evaluate on original graph via vertex mapping
```

---

## Supported Datasets

| Dataset       | Type              | Source       |
|---------------|-------------------|--------------|
| `ogbn-arxiv`  | Citation network  | OGB          |
| `ogbn-products` | Co-purchase graph | OGB         |
| `ogbn-papers100M` | Large citation | OGB         |
| `reddit`      | Social network    | DGL built-in |
| `pubmed`      | Citation network  | DGL built-in |
| `yelp`        | Fraud detection   | DGL built-in |
| `igb-*`       | Large-scale graph | IGB (custom) |

---

## Requirements

### System
- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA Toolkit installed (`nvcc` must be in `PATH`)
- CUDA 11.x or higher recommended

### Python
```
torch
dgl
ogb
numpy
scipy
torchmetrics
tqdm
```

Install Python dependencies:
```bash
pip install torch dgl ogb numpy scipy torchmetrics tqdm
```

---

## Usage

### Step 1 — Compile the CUDA Kernel

```bash
nvcc him_coarsening.cu -o him_coarsening
```

### Step 2 — Run the Preprocessing Pipeline

```bash
python preprocess.py \
  --dataset ogbn-arxiv \
  --path /path/to/datasets \
  --retain_fraction 0.5 \
  --feature_agg mean \
  --label_agg mode \
  --mask_agg any
```

**Arguments:**

| Argument            | Default        | Description                                              |
|---------------------|----------------|----------------------------------------------------------|
| `--dataset`         | `ogbn-arxiv`   | Dataset name                                             |
| `--path`            | `/data/dgl_lab`| Path to dataset root directory                           |
| `--retain_fraction` | `0.5`          | Fraction of nodes to retain after coarsening (0.0 – 1.0)|
| `--feature_agg`     | `mean`         | Feature aggregation: `mean`, `sum`, `max`                |
| `--label_agg`       | `mode`         | Label aggregation: `mode`, `majority`, `first`, `max`, `min` |
| `--mask_agg`        | `any`          | Mask aggregation: `any`, `all`, `majority`               |

This produces a `<dataset>.dgl` file (e.g., `arxiv.dgl`) and the `CoarsedGraphOutput/` directory.

### Step 3 — Train on the Coarsened Graph

```bash
python train.py \
  --dataset ogbn-arxiv \
  --path /path/to/datasets \
  --epoch 50 \
  --num_layers 2 \
  --fan_out 10,10 \
  --batch_size 1024 \
  --mode puregpu
```

**Arguments:**

| Argument       | Default     | Description                              |
|----------------|-------------|------------------------------------------|
| `--mode`       | `puregpu`   | Training mode: `cpu`, `mixed`, `puregpu` |
| `--dataset`    | `ogbn-arxiv`| Dataset name (must match preprocessing)  |
| `--epoch`      | `50`        | Number of training epochs                |
| `--num_layers` | `2`         | Number of GraphSAGE layers               |
| `--fan_out`    | `10,10,10`  | Neighbor sampling fanout per layer       |
| `--batch_size` | `1024`      | Mini-batch size                          |

---

## Output Files

After preprocessing, the following files are generated:

```
<dataset_name>_input/
├── row.txt                        # CSR row pointers
├── column.txt                     # CSR column indices
└── input_coarsening_weights.txt   # Edge weights for coarsening

CoarsedGraphOutput/
├── coarse_row.txt                 # Coarsened graph CSR row pointers
├── coarse_column.txt              # Coarsened graph CSR column indices
├── coarse_weight.txt              # Coarsened graph edge weights
└── final_vertex_mapping.txt       # Maps coarse node → original nodes

MergeParts/
└── merge_level_<N>.txt            # Per-level merge info (for debugging)

<dataset>.dgl                      # Final coarsened DGL graph (used for training)
```

---

## How the Coarsening Works

The CUDA coarsening kernel (`him_coarsening.cu`) implements **iterative heavy-edge matching**:

1. **Match** — Each node finds its highest-weight unmatched neighbor in parallel (`matchVertices`).
2. **Rematch** — Conflicting or unmatched nodes are rematched iteratively until convergence (`rematchVertices`).
3. **Map** — Matched pairs are assigned a shared representative (`mapVertices`).
4. **Relabel** — Representatives receive dense new IDs in two atomic stages to avoid race conditions.
5. **Build coarse graph** — Edges between coarse nodes are aggregated and self-loops are removed.
6. **Repeat** — The process iterates until the graph reaches the target size (`retain_fraction × original_nodes`).

---

## Timing Output

Both scripts report detailed timing:

```
--- COARSENING TIMING SUMMARY ---
Data Prep Time:            X.XXXX seconds
Coarsening Execution Time: X.XXXX seconds
Total Preprocessing Time:  X.XXXX seconds

TRAIN_TIME: X.XXXX
TEST_TIME:  X.XXXX
```

---


---

## Institution

Developed at the **Indian Institute of Technology Bhilai**.

---

## License

This project is released for academic and research use.
