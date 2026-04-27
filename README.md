# POP_GNN_Project

Graph reduction + node classification pipeline using CUDA-based graph coarsening and DGL/PyTorch GraphSAGE.

This repository compares:
- Baseline training on the original graph.
- Training on a reduced (coarsened) graph, then testing on the original graph.

The coarsening pipeline combines CUDA kernels for edge weighting and node merging with DGL-based dataset handling and GNN training.

## Main Components

- `him_file.py`
  - End-to-end coarsening pipeline.
  - Compiles and runs CUDA kernels:
    - `him_spectralweight.cu`
    - `him_coarsening.cu`
  - Loads datasets (OGB, Reddit, Yelp, Pubmed, Flickr), prepares CSR inputs, runs weighting + coarsening, and writes `<dataset>.dgl`.

- `node_classification.py`
  - GraphSAGE training/evaluation script.
  - Supports:
    - Baseline mode (`--baseline`) on original graph.
    - Coarsened mode (loads generated `<dataset>.dgl`).
  - Always evaluates final test accuracy on the original graph using full-graph inference.

- `Run_commands_sh/*.sh`
  - SLURM job scripts for running baseline + coarsening sweeps on datasets:
    - Reddit, OGBN-Arxiv, OGBN-Products, Yelp, Pubmed.

- Output folders
  - `output/`, `output_1/`, `output_2/`, `output_3/`, `output_4/` contain experiment logs.

## Workflow

1. Load dataset and preprocess graph.
2. Convert graph to CSR format.
3. Compute edge weights via `him_spectralweight`.
4. Coarsen graph via `him_coarsening` (controlled by `retain_fraction`).
5. Save reduced graph as `<dataset_alias>.dgl`.
6. Train GraphSAGE on reduced graph (or original graph in baseline mode).
7. Test on original graph.

## Requirements

Typical dependencies used by the code:
- Python 3.9+
- PyTorch
- DGL (GPU build recommended)
- OGB (`ogb`)
- torchmetrics
- numpy, scipy, tqdm
- cupy (used in sampling/coarsening helpers)
- CUDA Toolkit with `nvcc` available in `PATH`

The scripts are designed for GPU environments (for example SLURM + CUDA modules).

## Quick Start

### 1) Run baseline training (original graph)

Example (Reddit):

python node_classification.py --dataset reddit --epoch 50 --mode puregpu --baseline

### 2) Generate coarsened graph

Example (Reddit, keep ~70% target retention setting):

python him_file.py --dataset reddit --method coarsening --retain_fraction 0.7 --boost_h 0

This creates `reddit.dgl` (or `arxiv.dgl`, `products.dgl`, `yelp.dgl`, `pubmed.dgl` depending on dataset).

### 3) Train on coarsened graph and test on original graph

python node_classification.py --dataset reddit --epoch 50 --mode puregpu

### 4) Run full SLURM sweeps

Use scripts under `Run_commands_sh/`:
- `run_1_coarsening.sh` (reddit)
- `run_arxiv.sh`
- `run_product.sh`
- `run_yelp.sh`
- `run_pubmad.sh`

Each script runs:
- 1 baseline experiment.
- 4 coarsening settings (`R = 0.3, 0.5, 0.7, 0.85`).

## Important CLI Arguments

### `him_file.py`

- `--dataset`: dataset name (`reddit`, `pubmed`, `yelp`, `ogbn-arxiv`, `ogbn-products`, etc.)
- `--method`: `coarsening`, `sampling`, or `hybrid`
- `--retain_fraction`: target fraction of nodes to retain after coarsening
- `--sample_fraction`: used in hybrid/sampling mode
- `--boost_h`: homophily boost added to same-label edges
- `--path`: dataset root path (default `/data/dgl_lab`)

### `node_classification.py`

- `--dataset`
- `--epoch`
- `--mode` (`cpu`, `mixed`, `puregpu`)
- `--num_layers`
- `--fan_out`
- `--batch_size`
- `--baseline` (train on original graph)

## Experimental Results (From Current Logs)

Below are baseline vs best coarsened results from existing log files in this repo.

| Dataset | Baseline Final Test Acc | Best Coarsened Final Test Acc | Best Reduction Ratio | Best R file |
|---|---:|---:|---:|---|
| reddit | 0.9627 | 0.9610 | 48.17% | `output/coarse_reddit_R0.7_1264.txt` |
| ogbn-arxiv | 0.5452 | 0.5135 | 36.07% | `output_1/coarse_ogbn-arxiv_R0.85_1258.txt` |
| ogbn-products | 0.7594 | 0.7761 | 73.35% | `output_2/coarse_ogbn-products_R0.3_1259.txt` |
| yelp | 0.8820 | 0.8668 | 49.62% | `output_3/coarse_yelp_R0.7_1261.txt` |
| pubmed | 0.7610 | 0.7440 | 52.02% | `output_4/coarse_pubmed_R0.5_1260.txt` |

Notes:
- Coarsening improves speed and memory use by reducing training graph size.
- Accuracy impact depends on dataset and effective reduction ratio.
- In current logs, OGBN-Products shows a positive accuracy gain at strong reduction.
- Some datasets show non-linear behavior: configured `R` does not always map one-to-one to final measured reduction in output logs.

## Output Log Pattern

Most logs end with:
- Original/Training graph size
- Reduction ratio
- Best validation accuracy (on training graph)
- Final test accuracy (on original graph)
- Train/test timing

This makes downstream comparison easy by parsing lines:
- `Reduction Ratio:`
- `BEST_VAL_ACC (on G_t):`
- `FINAL_TEST_ACC (on G):`

## License

See `LICENSE`.
