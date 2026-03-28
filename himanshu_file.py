import os

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
# Redirect DGL config directory to a writable location if the home directory is restricted
if not os.access(os.path.expanduser("~"), os.W_OK):
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import dgl
import torch
import numpy as np
import scipy.sparse as sp
import os
import subprocess
import sys
import time
import shutil # Added for file operations

from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, PubmedGraphDataset, FraudYelpDataset as LegacyFraudYelpDataset
import dgl.graphbolt as gb

# Define the names of the source files and their desired executable names
KERNELS = {
    "him_coarsening.cu": "him_coarsening",
}

# Define the output directories that the C++ programs will create
OUTPUT_DIRS = [
    "CoarsedGraphOutput",
    "MergeParts",
]

def check_nvcc():
    """Checks if nvcc (NVIDIA CUDA Compiler) is available in the system path."""
    if not shutil.which("nvcc"):
        print("--> [ERROR] nvcc not found. Please ensure the CUDA Toolkit is installed and configured in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    print("--> [SUCCESS] nvcc found.")

def compile_kernels():
    """Compiles all CUDA source files defined in the KERNELS dictionary."""
    print("\n--- Compiling CUDA Kernels ---")
    check_nvcc()
    for source, executable in KERNELS.items():
        if not os.path.exists(source):
            print(f"--> [ERROR] Source file '{source}' not found. Cannot compile.", file=sys.stderr)
            sys.exit(1)
        
        print(f"--> Compiling '{source}' into '{executable}'...")
        compile_command = [
            "nvcc", source, "-o", executable,
            # Add any other required flags like -arch=sm_XX if needed
        ]
        try:
            subprocess.run(compile_command, check=True, capture_output=True, text=True)
            print(f"--> [SUCCESS] Compiled '{executable}'.")
        except subprocess.CalledProcessError as e:
            print(f"--> [ERROR] Failed to compile '{source}'.", file=sys.stderr)
            print("NVCC Error Output:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)

def setup_cuda_output_directories():
    """Creates the necessary output directories for CUDA programs."""
    print("\n--- Setting up CUDA Output Directories ---")
    for directory in OUTPUT_DIRS:
        os.makedirs(directory, exist_ok=True)
        print(f"--> Ensured directory '{directory}' exists.")

def get_processed_graphs_and_times(dataset_name, weight_type, retain_fraction, mask_agg, feature_agg, label_agg, dataset_path, boost_h=0.0):
    # --- CONFIGURATION ---
    # Clean name for folder creation (e.g., "arxiv" from "ogbn-arxiv")
    if dataset_name.startswith("ogbn-"):
        name_for_folders = dataset_name.replace("ogbn-", "")
    elif dataset_name.startswith("igb-"):
        name_for_folders = dataset_name.replace("igb-", "")
    else:
        name_for_folders = dataset_name # For reddit, yelp, etc.
        
    INPUT_DIR = name_for_folders + "_input"
    OUTPUT_DGL_FILE = name_for_folders + ".dgl"

    # Executables (Must be compiled from C++ files) - now reference KERNELS
    COARSENING_EXE = f"./{KERNELS['him_coarsening.cu']}"

    COARSE_OUTPUT_DIR = OUTPUT_DIRS[0] # "CoarsedGraphOutput"

    # Initialize timing variables
    data_prep_time = 0.0
    coarsening_exec_time = 0.0

    # --- PIPELINE SETUP (CUDA) ---
    compile_kernels()
    setup_cuda_output_directories()
    
    # --- DATA PREPARATION (PYTHON) ---

    print(f"--- 1. Preparing {dataset_name} Graph Data ---")
    data_prep_start_time = time.time()
    try:
        print("--> [LOG] Loading dataset...")
        dataset = None # Initialize dataset variable
        if dataset_name.startswith("ogbn-"):
            dataset = DglNodePropPredDataset(name=dataset_name, root=dataset_path)
            graph, labels = dataset[0]
            graph.ndata['label'] = labels
        elif dataset_name == "yelp":
            dataset = LegacyFraudYelpDataset(raw_dir=dataset_path) # Store dataset object
            g_raw = dataset[0] # Get the raw graph from the dataset
            graph = dgl.to_homogeneous(g_raw)
            graph.ndata['feat'] = g_raw.ndata['feature']
            graph.ndata['label'] = g_raw.ndata['label'].long()
            graph.ndata['train_mask'] = g_raw.ndata['train_mask']
            graph.ndata['val_mask'] = g_raw.ndata['val_mask']
            graph.ndata['test_mask'] = g_raw.ndata['test_mask']
        elif dataset_name.startswith("igb-"):
            dgl_file_path = os.path.join(dataset_path, dataset_name.replace('-', '_') + '.dgl')
            print(f"--> [LOG] Loading pre-processed IGB graph from: {dgl_file_path}")
            graphs, _ = dgl.load_graphs(dgl_file_path)
            graph = graphs[0]
        else: # For reddit, pubmed
            if dataset_name == "reddit": dataset = RedditDataset()
            elif dataset_name == "pubmed": dataset = PubmedGraphDataset(raw_dir=dataset_path)
            else: raise ValueError(f"Unknown or unsupported dataset: {dataset_name}")
            graph = dataset[0]

        print("--> [LOG] ...Dataset loaded.")
            
        graph_original_dgl = graph.clone() # Store the original graph object
        
        # Store original data before cleaning the structure
        feat_original = graph.ndata['feat'].clone()
        
        # --- Handle label_original consistently ---
        if hasattr(dataset, 'labels') and dataset.labels is not None:
            label_original = dataset.labels.clone().squeeze()
        elif 'label' in graph.ndata:
            label_original = graph.ndata['label'].clone().squeeze()
        else:
            raise ValueError(f"Could not find labels for dataset {dataset_name}")
        
        # --- Store original masks for boost logic ---
        train_mask_original = graph.ndata.get('train_mask', torch.zeros(graph.num_nodes(), dtype=torch.bool))

        # --- ADDED: Detect if the dataset is multi-label ---
        is_multilabel = label_original.ndim > 1 and label_original.shape[1] > 1
        if is_multilabel:
            print("--> [LOG] Detected multi-label dataset. Using multi-label aggregation.")

        # Clean the graph structure for coarsening input
        print("--> [LOG] Cleaning graph (to_simple, to_bidirected)...")
        graph = dgl.to_simple(graph)
        print("--> [LOG] to_simple Done ")
        graph = dgl.to_bidirected(graph)
        print("--> [LOG] to_bidirected Done ")
        print("--> [LOG] ...Graph cleaning done.")
        
        # Use DGL's optimized C++ function for CSR conversion for performance
        print("--> [LOG] Converting graph to CSR format...")
        row_ptr, col_ind, _ = graph.adj_tensors('csr')
        xadj = row_ptr.cpu().numpy().astype(np.int32)
        adjncy = col_ind.cpu().numpy().astype(np.int32)
        print("--> [LOG] ...CSR conversion done.")

        nvtxs = graph.num_nodes()
        nedges = graph.num_edges()
        
        print(f"Graph Size: {nvtxs} nodes, {nedges} edges (for CSR input).")

        data_prep_end_time = time.time()
        data_prep_time = data_prep_end_time - data_prep_start_time

    except Exception as e:
        print(f"Error during DGL data preparation: {e}")
        sys.exit(1)


    # --- 2. GENERATE INITIAL C++ INPUT FILES ---

    print(f"\n--- 2. Generating Initial Input Files in {INPUT_DIR} ---")
    os.makedirs(INPUT_DIR, exist_ok=True)

    try:
        # 1. CSR Row Pointers (xadj)
        np.savetxt(os.path.join(INPUT_DIR, "row.txt"), xadj, fmt='%d')
        # 2. CSR Column Indices (adjncy)
        np.savetxt(os.path.join(INPUT_DIR, "column.txt"), adjncy, fmt='%d')
        
        print("Initial input files (row.txt, column.txt) generated.")
    except Exception as e:
        print(f"Error saving initial input files: {e}")
        sys.exit(1)


    # --- 3. GENERATE UNIT WEIGHTS FOR COARSENING ---

    print(f"\n--- 3. Preparing Unit Weights for Coarsening ---")
    coarsening_input_weight_file = os.path.join(INPUT_DIR, "input_coarsening_weights.txt")

    try:
        # Use unit weights (all ones), scaled to integer format expected by coarsening tool
        unit_weights = np.ones(nedges, dtype=np.int32)
        np.savetxt(coarsening_input_weight_file, unit_weights, fmt='%d')
        print("--> [SUCCESS] Coarsening input weights (unit) are ready.")
    except Exception as e:
        print(f"--> [ERROR] Failed to prepare weights: {e}", file=sys.stderr)
        sys.exit(1)


    # --- 4. EXECUTE GRAPH COARSENING (him_coarsening.cu) ---

    print("\n--- 4. Running Graph Coarsening ---")
    coarsening_exec_start_time = time.time()
    if not os.path.exists(COARSENING_EXE):
        print(f"Skipping coarsening execution: Executable '{COARSENING_EXE}' not found.")
    else:
        cmd = [COARSENING_EXE, INPUT_DIR, str(nvtxs), str(nedges), str(retain_fraction)]
        print(f"Executing: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Coarsening Output (stdout):")
            print(result.stdout)
            if result.stderr:
                print("Coarsening Errors (stderr):")
                print(result.stderr)
            print(f"Coarsening complete. Output files expected in {COARSE_OUTPUT_DIR}/")

        except subprocess.CalledProcessError as e:
            print(f"Error running Coarsening program: {e}")
            print(e.stderr)
            sys.exit(1)
    coarsening_exec_end_time = time.time()
    if os.path.exists(COARSENING_EXE):
        coarsening_exec_time = coarsening_exec_end_time - coarsening_exec_start_time


    # --- 5. CONSUME OUTPUT & AGGREGATE DATA (PYTHON) ---

    print("\n--- 5. Reading Coarsened Output and Aggregating Features ---")

    # Helper functions
    def read_txt_file(path, dtype=int):
        with open(path, 'r') as f:
            return list(map(dtype, f.read().strip().split()))

    def load_node_groups(file_path):
        groups = {}
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    key_part, val_part = line.strip().split(':')
                    key = int(key_part.replace('vertex', '').strip())
                    values = [int(v.strip()) for v in val_part.strip().split(',') if v.strip()]
                    groups[key] = values
                except ValueError:
                    print(f"Skipping malformed line in mapping file: {line.strip()}")
                    continue
        return groups

    # Define paths to output files
    mapping_path = os.path.join(COARSE_OUTPUT_DIR, "final_vertex_mapping.txt")
    row_path = os.path.join(COARSE_OUTPUT_DIR, "coarse_row.txt")

    if not os.path.exists(mapping_path) or not os.path.exists(row_path):
        print(f"ERROR: Final coarsening output files not found in {COARSE_OUTPUT_DIR}/. Check CUDA execution.")
        sys.exit(1)

    try:
        # Load Coarsened CSR Structure
        row_ptr = read_txt_file(row_path, dtype=int)
        num_new_nodes = len(row_ptr) - 1
        if num_new_nodes <= 0:
            raise ValueError("Coarse row pointer file is empty or invalid.")

        # Re-load original graph to get masks
        print("--> [LOG] Re-loading dataset for mask aggregation...")
        if dataset_name.startswith("ogbn-"):
            d = AsNodePredDataset(DglNodePropPredDataset(dataset_name, root=dataset_path))
            orig_g_for_masks = d[0]
        elif dataset_name == "yelp":
            d = LegacyFraudYelpDataset(raw_dir=dataset_path); g_raw = d[0]
            orig_g_for_masks = dgl.to_homogeneous(g_raw)
            orig_g_for_masks.ndata['train_mask'] = g_raw.ndata['train_mask']
            orig_g_for_masks.ndata['val_mask'] = g_raw.ndata['val_mask']
            orig_g_for_masks.ndata['test_mask'] = g_raw.ndata['test_mask']
        elif dataset_name.startswith("igb-"):
            dgl_file_path = os.path.join(dataset_path, dataset_name.replace('-', '_') + '.dgl')
            graphs, _ = dgl.load_graphs(dgl_file_path)
            orig_g_for_masks = graphs[0]
        else:
            if dataset_name == "reddit": d = RedditDataset()
            elif dataset_name == "pubmed": d = PubmedGraphDataset(raw_dir=dataset_path)
            else: raise ValueError(f"Original dataset loader not implemented for {dataset_name} for masks extraction.")
            orig_g_for_masks = d[0]
        
        # Load Node Grouping Information
        groups = load_node_groups(mapping_path)
        
        # Initialize aggregated data structures
        feat_dim = feat_original.shape[1]
        new_features = torch.zeros((num_new_nodes, feat_dim))
        if is_multilabel:
            new_labels = torch.zeros((num_new_nodes, label_original.shape[1]), dtype=torch.float32)
        else:
            new_labels = torch.zeros(num_new_nodes, dtype=label_original.dtype)
        new_train_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
        new_val_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
        new_test_mask = torch.zeros(num_new_nodes, dtype=torch.bool)

        # Get original masks
        train_mask_original = orig_g_for_masks.ndata.get('train_mask', torch.zeros(nvtxs, dtype=torch.bool))
        val_mask_original = orig_g_for_masks.ndata.get('val_mask', torch.zeros(nvtxs, dtype=torch.bool))
        test_mask_original = orig_g_for_masks.ndata.get('test_mask', torch.zeros(nvtxs, dtype=torch.bool))

        # Aggregate features, labels, and masks
        for new_node, old_nodes in groups.items():
            if new_node >= num_new_nodes or new_node < 0:
                continue
            valid_old_nodes = [node for node in old_nodes if node != -1]
            if valid_old_nodes:
                old_nodes_tensor = torch.tensor(valid_old_nodes, dtype=torch.long)
                
                # --- Feature Aggregation ---
                if feature_agg == "mean":
                    new_features[new_node] = feat_original[old_nodes_tensor].mean(dim=0)
                elif feature_agg == "sum":
                    new_features[new_node] = feat_original[old_nodes_tensor].sum(dim=0)
                elif feature_agg == "max":
                    new_features[new_node] = feat_original[old_nodes_tensor].max(dim=0).values

                # --- Label Aggregation ---
                if is_multilabel:
                    if label_agg in ["any", "max"]:
                        new_labels[new_node] = label_original[old_nodes_tensor].max(dim=0).values.float()
                    elif label_agg in ["all", "min"]:
                        new_labels[new_node] = label_original[old_nodes_tensor].min(dim=0).values.float()
                    elif label_agg == "majority":
                        new_labels[new_node] = (label_original[old_nodes_tensor].sum(dim=0) > len(old_nodes_tensor) / 2).float()
                    else:
                        new_labels[new_node] = label_original[old_nodes_tensor].max(dim=0).values.float()
                else: # Multi-class
                    if label_agg == "mode":
                        flat_labels = label_original[old_nodes_tensor].flatten()
                        new_labels[new_node] = torch.mode(flat_labels).values
                    elif label_agg == "first":
                        new_labels[new_node] = label_original[old_nodes_tensor][0]
                    elif label_agg == "max":
                        new_labels[new_node] = label_original[old_nodes_tensor].max()
                    elif label_agg == "min":
                        new_labels[new_node] = label_original[old_nodes_tensor].min()
                    else:
                        flat_labels = label_original[old_nodes_tensor].flatten()
                        new_labels[new_node] = torch.mode(flat_labels).values

                # --- Mask Aggregation ---
                if mask_agg == "any":
                    new_train_mask[new_node] = train_mask_original[old_nodes_tensor].any()
                    new_val_mask[new_node] = val_mask_original[old_nodes_tensor].any()
                    new_test_mask[new_node] = test_mask_original[old_nodes_tensor].any()
                elif mask_agg == "all":
                    new_train_mask[new_node] = train_mask_original[old_nodes_tensor].all()
                    new_val_mask[new_node] = val_mask_original[old_nodes_tensor].all()
                    new_test_mask[new_node] = test_mask_original[old_nodes_tensor].all()
                elif mask_agg == "majority":
                    new_train_mask[new_node] = (train_mask_original[old_nodes_tensor].sum() > len(old_nodes_tensor) / 2)
                    new_val_mask[new_node] = (val_mask_original[old_nodes_tensor].sum() > len(old_nodes_tensor) / 2)
                    new_test_mask[new_node] = (test_mask_original[old_nodes_tensor].sum() > len(old_nodes_tensor) / 2)

        # Reconstruct the full Coarsened Graph
        col_idx = read_txt_file(os.path.join(COARSE_OUTPUT_DIR, "coarse_column.txt"), dtype=int)
        weights = read_txt_file(os.path.join(COARSE_OUTPUT_DIR, "coarse_weight.txt"), dtype=float)
        
        g_coarse = dgl.graph(('csr', (torch.tensor(row_ptr, dtype=torch.int64), 
                                      torch.tensor(col_idx, dtype=torch.int64), 
                                      [])))
        g_coarse.ndata['feat'] = new_features
        g_coarse.ndata['label'] = new_labels
        g_coarse.ndata['train_mask'] = new_train_mask
        g_coarse.ndata['val_mask'] = new_val_mask
        g_coarse.ndata['test_mask'] = new_test_mask
        g_coarse.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

        print(f"\nFinal Coarsened Graph: {g_coarse}")
        dgl.save_graphs(OUTPUT_DGL_FILE, g_coarse)
        print(f"Final coarsened graph saved to {OUTPUT_DGL_FILE}.")

    except Exception as e:
        print(f"Critical error during aggregation/consumption: {e}")
        sys.exit(1)

    # --- COARSENING TIMING SUMMARY ---
    total_preprocess_time = data_prep_time + coarsening_exec_time
    print("\n--- COARSENING TIMING SUMMARY ---")
    print(f"Data Prep Time: {data_prep_time:.4f} seconds")
    print(f"Coarsening Execution Time: {coarsening_exec_time:.4f} seconds")
    print(f"Total Preprocessing Time: {total_preprocess_time:.4f} seconds")
    
    return graph_original_dgl, g_coarse, data_prep_time, coarsening_exec_time, total_preprocess_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="A pipeline to prepare graph data and perform CUDA-based coarsening.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="ogbn-arxiv",
        help="Dataset to process (e.g., 'ogbn-arxiv', 'reddit', 'yelp', 'igb-full')."
    )
    parser.add_argument(
        "--path",
        type=str,
        default='/data/dgl_lab',
        help="Path to dataset directory."
    )
    parser.add_argument(
        "-r", "--retain_fraction",
        type=float,
        default=0.5,
        help="The target fraction of vertices to retain after coarsening.\n"
             "Must be between 0.0 and 1.0 (exclusive)."
    )
    parser.add_argument(
        "--mask_agg",
        default="any",
        choices=["any", "all", "majority"],
        help="Strategy for aggregating node masks."
    )
    parser.add_argument(
        "--feature_agg",
        default="mean",
        choices=["mean", "sum", "max"],
        help="Strategy for aggregating node features."
    )
    parser.add_argument(
        "--label_agg",
        default="mode",
        choices=["mode", "any", "all", "majority", "first", "max", "min"],
        help="Strategy for aggregating node labels."
    )
    args = parser.parse_args()
    
    if not 0 < args.retain_fraction < 1:
        print(f"--> [ERROR] --retain_fraction must be between 0.0 and 1.0 (exclusive).", file=sys.stderr)
        sys.exit(1)
        
    # Call the main function with the parsed arguments
    graph_original, graph_coarsened, data_prep_t, coarsening_t, total_preprocess_t = \
        get_processed_graphs_and_times(args.dataset, None, args.retain_fraction, args.mask_agg, args.feature_agg, args.label_agg, args.path)
    
    print("\n--- FINAL SUMMARY ---")
    if graph_original:
        print(f"Original Graph Nodes: {graph_original.num_nodes()}, Edges: {graph_original.num_edges() // 2}")
    if graph_coarsened:
        print(f"Coarsened Graph Nodes: {graph_coarsened.num_nodes()}, Edges: {graph_coarsened.num_edges() // 2}")
    print(f"Data Preparation Time: {data_prep_t:.4f}s")
    print(f"Coarsening Execution Time: {coarsening_t:.4f}s")
    print(f"Total Preprocessing Time (G to G''): {total_preprocess_t:.4f}s")