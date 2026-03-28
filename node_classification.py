import os

# Set DGL backend and home directory before importing DGL to avoid permission errors on clusters
os.environ['DGLBACKEND'] = 'pytorch'
# Redirect DGL config directory to a writable location if the home directory is restricted
try:
    if not os.access(os.path.expanduser("~"), os.W_OK):
        os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")
except Exception:
    os.environ['DGL_HOME'] = os.path.join(os.getcwd(), ".dgl")

import argparse
import time

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
import re
import os
import sys
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
import dgl.graphbolt as gb
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, FlickrDataset, ActorDataset, YelpDataset
from dgl.data import FraudYelpDataset as LegacyFraudYelpDataset # Rename to avoid conflict

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, "mean"))
        else:
            # input layer
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            # hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            # output layer
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader, num_classes, is_multilabel):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    
    y_hats_cat = torch.cat(y_hats)
    ys_cat = torch.cat(ys)

    if is_multilabel:
        preds = torch.sigmoid(y_hats_cat)
        return MF.f1_score(
            preds,
            ys_cat.int(),
            task="multilabel",
            num_labels=num_classes,
            average="macro",
        )
    else:
        return MF.accuracy(
            y_hats_cat,
            ys_cat,
            task="multiclass",
            num_classes=num_classes,
        )

def layerwise_infer(
    device, g_orig, g_coarse, node_map, nid, model, num_classes, batch_size, is_multilabel
):
    model.eval()
    with torch.no_grad():
        # Step 1: Run inference on the coarse graph to get predictions for all coarse nodes.
        coarse_preds = model.inference(g_coarse, device, batch_size)

        # Step 2: Map the original test node IDs to the predictions from the coarse graph.
        mapped_pred_list = []
        valid_nid_list = []
        for orig_node_id in nid.tolist():
            coarse_node_id = node_map.get(orig_node_id)
            if coarse_node_id is not None:
                mapped_pred_list.append(coarse_preds[coarse_node_id])
                valid_nid_list.append(orig_node_id)
        
        pred = torch.stack(mapped_pred_list).to(device)
        valid_nid = torch.tensor(valid_nid_list, device=device)
        label = g_orig.ndata["label"][valid_nid].to(device)

        if is_multilabel:
            preds = torch.sigmoid(pred)
            return MF.f1_score(
                preds,
                label.int(),
                task="multilabel",
                num_labels=num_classes,
                average="macro",
            )
        else:
            return MF.accuracy(pred, label, task="multiclass", num_classes=num_classes)

def train(args, device, g, g_loaded, model, num_classes, is_multilabel):
    train_mask = g_loaded.ndata['train_mask']
    val_mask = g_loaded.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze()
    val_idx = torch.nonzero(val_mask).squeeze()

    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")[0:args.num_layers]],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g_loaded,
        train_idx.to(device),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g_loaded,
        val_idx.to(device),
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_acc = 0
    
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        
        loop_start_time = time.time()
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            else:
                loss = F.cross_entropy(y_hat, y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        loop_time = time.time() - loop_start_time
        
        metric_name = "Val F1-Score" if is_multilabel else "Val Accuracy"
        acc = evaluate(model, g_loaded, val_dataloader, num_classes, is_multilabel)
        if acc > best_val_acc:
            best_val_acc = acc
        
        print(
            f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} | "
            f"{metric_name} {acc.item():.4f} (Best: {best_val_acc.item():.4f}) | "
            f"Time {loop_time:.4f}"
        )
        
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="puregpu", choices=["cpu", "mixed", "puregpu"])
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--path', type=str, default='/data/dgl_lab', help='Path to dataset')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    print(f"Loading original {args.dataset} dataset...")
    
    if args.dataset.startswith("ogbn-"):
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=args.path))
        g = dataset[0]
        num_classes = dataset.num_classes
    elif args.dataset == 'yelp':
        d = LegacyFraudYelpDataset(raw_dir=args.path); g_raw = d[0]
        g = dgl.to_homogeneous(g_raw)
        g.ndata['feat'] = g_raw.ndata['feature']
        g.ndata['label'] = g_raw.ndata['label'].long()
        num_classes = 2
        g.ndata['train_mask'] = g_raw.ndata['train_mask']
        g.ndata['val_mask'] = g_raw.ndata['val_mask']
        g.ndata['test_mask'] = g_raw.ndata['test_mask']
    elif args.dataset.startswith("igb"):
        dgl_file_path = os.path.join(args.path, args.dataset.replace('-', '_') + '.dgl')
        print(f"--> [LOG] Loading pre-processed IGB graph from: {dgl_file_path}")
        graphs, _ = dgl.load_graphs(dgl_file_path)
        g = graphs[0]
        num_classes = 19
    else:
        if args.dataset == "cora": dataset = CoraGraphDataset()
        elif args.dataset == "citeseer": dataset = CiteseerGraphDataset()
        elif args.dataset == "pubmed": dataset = PubmedGraphDataset()
        elif args.dataset == "reddit": dataset = RedditDataset()
        else: raise ValueError(f"Unknown dataset: {args.dataset}")
        g = dataset[0]
        num_classes = dataset.num_classes

    is_multilabel = g.ndata['label'].ndim > 1 and g.ndata['label'].shape[1] > 1
    if is_multilabel:
        print("Detected multi-label dataset. Using F1 score and BCE loss.")

    test_idx = g.ndata["test_mask"].nonzero().squeeze()

    coarsened_graph_name = args.dataset.split("-")[-1] + ".dgl"
    print(f"Loading coarsened graph from: {coarsened_graph_name}")
    try:
        graphs, _ = dgl.load_graphs(coarsened_graph_name)
        g_coarse = graphs[0]
        print("Coarsened graph loaded successfully:")
        print(g_coarse)
    except Exception as e:
        print(f"Error loading coarsened graph: {e}")
        print("Please ensure the coarsening pipeline was run successfully for this dataset.")
        sys.exit(1)

    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    if args.mode == "puregpu":
        g = g.to(device)
        g_coarse = g_coarse.to(device)

    in_size = g_coarse.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size, args.num_layers).to(device)

    print("\nTraining on coarsened graph...")
    train_start_time = time.time()
    best_val_acc = train(args, device, g, g_coarse, model, num_classes, is_multilabel)
    train_time = time.time() - train_start_time
    
    print("\nLoading vertex mapping for evaluation...")
    orig_to_coarse_node_map = {}
    mapping_path = "CoarsedGraphOutput/final_vertex_mapping.txt"
    try:
        with open(mapping_path, 'r') as f:
            for line in f:
                parts = re.match(r"vertex (\d+): ([\d,]+)", line)
                if parts:
                    coarse_node_id = int(parts.group(1))
                    original_nodes = [int(x) for x in parts.group(2).split(',')]
                    for orig_node_id in original_nodes:
                        if orig_node_id != -1:
                            orig_to_coarse_node_map[orig_node_id] = coarse_node_id
    except FileNotFoundError:
        print(f"ERROR: Mapping file not found at {mapping_path}.")
        print("Please ensure the coarsening pipeline was run successfully and the 'CoarsedGraphOutput' directory is present.")
        sys.exit(1)

    print("\nTesting on original graph by mapping coarse predictions...")
    test_start_time = time.time()
    test_acc = layerwise_infer(
        device,
        g,
        g_coarse,
        orig_to_coarse_node_map,
        test_idx,
        model,
        num_classes,
        batch_size=4096,
        is_multilabel=is_multilabel,
    )
    test_time = time.time() - test_start_time
    
    val_metric_name = "BEST_VAL_F1" if is_multilabel else "BEST_VAL_ACC"
    test_metric_name = "FINAL_TEST_F1" if is_multilabel else "FINAL_TEST_ACC"
    print("\n--- METRICS ---")
    print(f"{val_metric_name}: {best_val_acc.item():.4f}")
    print(f"{test_metric_name}: {test_acc.item():.4f}")
    print(f"TRAIN_TIME: {train_time:.4f}")
    print(f"TEST_TIME: {test_time:.4f}")