import os

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
# Redirect DGL config directory to a writable location if the home directory is restricted
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import argparse, time, torch, torch.nn as nn, torch.nn.functional as F, torchmetrics.functional as MF, dgl, os, tqdm, sys
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset, FlickrDataset, PubmedGraphDataset, FraudYelpDataset as LegacyFraudYelpDataset
from ogb.nodeproppred import DglNodePropPredDataset

# Set DGL backend to PyTorch
os.environ['DGLBACKEND'] = 'pytorch'

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size, self.out_size = hid_size, out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Layer-wise inference to save memory. Intermediate features stay on CPU."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1)
        # We use a DataLoader to iterate over all nodes
        dataloader = DataLoader(g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device, 
                                batch_size=batch_size, shuffle=False, drop_last=False)
        
        for l, layer in enumerate(self.layers):
            out_dim = self.hid_size if l != len(self.layers) - 1 else self.out_size
            y = torch.empty(g.num_nodes(), out_dim, device="cpu")
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc=f"Inference Layer {l+1}"):
                # Ensure indexing happens on CPU to avoid device mismatch, then move to GPU for computation
                x = feat[input_nodes.cpu()].to(device)
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes.cpu()] = h.cpu()
            feat = y
        return y

def train(args, device, g_coarse, model, num_classes, is_multilabel):
    train_idx = g_coarse.ndata['train_mask'].nonzero().squeeze()
    val_idx = g_coarse.ndata['val_mask'].nonzero().squeeze()
    
    sampler = NeighborSampler([int(f) for f in args.fan_out.split(",")[:args.num_layers]])
    train_loader = DataLoader(g_coarse, train_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(g_coarse, val_idx.to(device), sampler, device=device, batch_size=args.batch_size, shuffle=False)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_acc = torch.tensor(0.0).to(device)
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            
            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                loss = F.cross_entropy(y_hat, y.squeeze().long())
                
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        
        # Validation on Coarsened Graph
        model.eval(); ys, y_hats = [], []
        with torch.no_grad():
            for _, (_, _, blocks) in enumerate(val_loader):
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, blocks[0].srcdata["feat"]))
        
        val_preds, val_labels = torch.cat(y_hats), torch.cat(ys)
        acc = MF.accuracy(val_preds, val_labels.squeeze(), task="multiclass", num_classes=num_classes)
        
        if acc > best_val_acc: best_val_acc = acc
        print(f"Epoch {epoch:03d} | Loss {total_loss/(it+1):.4f} | Val Acc: {acc.item():.4f}")
        
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="reddit")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--path', type=str, default='/data/dgl_lab')
    parser.add_argument("--baseline", action="store_true", help="Train on the original graph instead of the coarsened one.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.mode != "cpu" else "cpu")

    # --- Smart Path Detection ---
    server_path = "/data/dgl_lab"
    local_path = os.path.expanduser("~/.dgl")
    data_path = args.path if args.path else (server_path if os.path.exists(server_path) else local_path)
    print(f"--> [LOG] Data Root: {data_path}")

    # --- 1. Load Original Graph ---
    print(f"Loading original {args.dataset} dataset...")
    if args.dataset.startswith("ogbn-"):
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=data_path))
        g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "flickr":
        dataset = FlickrDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes
    elif args.dataset == "yelp":
        # Use Legacy Fraud Yelp (Single-label binary)
        dataset = LegacyFraudYelpDataset(raw_dir=data_path); g_raw = dataset[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'], g.ndata['label'] = g_raw.ndata['feature'], g_raw.ndata['label'].long()
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = g_raw.ndata['train_mask'], g_raw.ndata['val_mask'], g_raw.ndata['test_mask']
        num_classes = 2
    elif args.dataset.startswith("igb"):
        # Search for .dgl file
        dgl_path = os.path.join(data_path, args.dataset.replace('-', '_') + '.dgl')
        if not os.path.exists(dgl_path):
            raise FileNotFoundError(f"IGB DGL file not found at {dgl_path}")
        g = dgl.load_graphs(dgl_path)[0][0]; num_classes = 19
        # Mapping for IGB consistency
        if 'feat' not in g.ndata and 'features' in g.ndata: g.ndata['feat'] = g.ndata['features']
        if 'label' not in g.ndata and 'labels' in g.ndata: g.ndata['label'] = g.ndata['labels']
    else:
        dataset = PubmedGraphDataset(raw_dir=data_path); g, num_classes = dataset[0], dataset.num_classes

    # Automatically detect multi-label based on label dimensions
    is_multilabel = g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    print(f"--> [LOG] Multi-label mode: {is_multilabel}")
    
    # --- 2. Select Graph for Training ---
    if args.baseline:
        print("--> [MODE] BASELINE: Training on the ORIGINAL graph.")
        g_train = g
    else:
        coarsened_graph_name = args.dataset.split("-")[-1] + ".dgl"
        if not os.path.exists(coarsened_graph_name):
            print(f"Error: Coarsened graph file '{coarsened_graph_name}' not found. Run 'him_file.py' first.")
            sys.exit(1)
        print(f"--> [MODE] COARSENED: Training on {coarsened_graph_name}")
        g_train = dgl.load_graphs(coarsened_graph_name)[0][0]
    
    # Ensure labels are long for multi-class cross entropy
    g_train.ndata['label'] = g_train.ndata['label'].long()
    g.ndata['label'] = g.ndata['label'].long()
    
    # Mode-based placement
    if args.mode == "puregpu":
        g, g_train = g.to(device), g_train.to(device)
    else:
        g_train = g_train.to(device)
        # For mixed mode, original graph stays on CPU but we optimize its format
        g.create_formats_()
    
    model = SAGE(g_train.ndata["feat"].shape[1], 256, num_classes, args.num_layers).to(device)

    # --- 3. Training ---
    print(f"\nTraining on {'Original' if args.baseline else 'Coarsened'} Graph ({args.dataset})...")
    train_start = time.time()
    best_val_acc = train(args, device, g_train, model, num_classes, is_multilabel)
    train_time = time.time() - train_start

    # --- 4. Testing (Directly on Original Graph) ---
    print("\nTesting on Original Graph via direct layer-wise inference...")
    test_start = time.time()
    model.eval()
    with torch.no_grad():
        # Full inference produces predictions for every node in the original graph
        full_preds = model.inference(g, device, 4096)
        test_idx = g.ndata["test_mask"].nonzero().squeeze()
        test_preds = full_preds[test_idx.cpu()].to(device)
        test_labels = g.ndata["label"][test_idx].to(device)
        
        # Always use Accuracy for testing
        test_acc = MF.accuracy(test_preds, test_labels.squeeze(), task="multiclass", num_classes=num_classes)
    test_time = time.time() - test_start

    # --- 5. Report ---
    print("\n--- RESULTS SUMMARY ---")
    print(f"Original Graph (G):   Nodes: {g.num_nodes():<10} Edges: {g.num_edges() // 2}")
    print(f"Training Graph (G_t): Nodes: {g_train.num_nodes():<10} Edges: {g_train.num_edges() // 2}")
    print(f"Reduction Ratio:      {1.0 - (g_train.num_nodes() / g.num_nodes()):.2%}")
    print("-" * 30)
    print(f"BEST_VAL_ACC (on G_t): {best_val_acc.item():.4f}")
    print(f"FINAL_TEST_ACC (on G): {test_acc.item():.4f}")
    print(f"TRAIN_TIME: {train_time:.4f}s")
    print(f"TEST_TIME: {test_time:.4f}s")
    print("-" * 30)
    print("Workflow: Trained on reduced G_t, Tested on Original G (Correct).")
