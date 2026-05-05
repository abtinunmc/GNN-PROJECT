"""
Test multiple solutions to improve GNN AUC:
1. MLP baseline (no graph structure)
2. Better graph construction (higher threshold)
3. Stacking ensemble (RF + GNN -> meta-learner)
4. RF predictions as node feature
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import dropout_edge

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

GRAPHS_PT_COMBINED = PROJECT_DIR / "data" / "processed_combined" / "graphs" / "graphs_combined.pt"
GRAPHS_PT_ORIGINAL = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs.pt"
GRAPHS_PT = GRAPHS_PT_COMBINED if GRAPHS_PT_COMBINED.exists() else GRAPHS_PT_ORIGINAL
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.21
CLASSIFIER_DROPOUT = 0.55
LR = 6.1e-4
WEIGHT_DECAY = 9e-5
BATCH_SIZE = 16
EPOCHS = 150
PATIENCE = 20

SEEDS = [42, 123, 456, 789, 2024]


def load_data():
    all_graphs = torch.load(GRAPHS_PT, weights_only=False)
    labeled = [g for g in all_graphs if g.n_soz > 0]
    train = [g for g in labeled if g.split == "train"]
    val = [g for g in labeled if g.split == "val"]
    test = [g for g in labeled if g.split == "test"]
    return train, val, test


def graphs_to_flat_arrays(graphs):
    X_list, y_list = [], []
    for g in graphs:
        X_list.append(g.x.numpy())
        y_list.append(g.y.numpy())
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def evaluate_metrics(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


def compute_class_weights(graphs):
    n_soz = sum(g.y.sum().item() for g in graphs)
    n_total = sum(g.y.shape[0] for g in graphs)
    n_non_soz = n_total - n_soz
    return torch.tensor([n_total / (2 * n_non_soz), n_total / (2 * n_soz)], dtype=torch.float32)


# ==============================================================================
# SOLUTION 1: MLP BASELINE (NO GRAPH)
# ==============================================================================

class MLPClassifier(nn.Module):
    """MLP without any graph structure - just node features."""
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(train_graphs, val_graphs, test_graphs, seed):
    """Train MLP without graph structure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Flatten to arrays
    X_train, y_train = graphs_to_flat_arrays(train_graphs)
    X_val, y_val = graphs_to_flat_arrays(val_graphs)
    X_test, y_test = graphs_to_flat_arrays(test_graphs)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    n_features = X_train.shape[1]
    class_weights = compute_class_weights(train_graphs).to(DEVICE)

    model = MLPClassifier(n_features, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    # Mini-batch training
    batch_size = 512
    n_batches = (len(X_train_t) + batch_size - 1) // batch_size

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(len(X_train_t))

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            x_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(x_batch)
            weights = class_weights[1] * y_batch + class_weights[0] * (1 - y_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch, weight=weights)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = torch.sigmoid(val_logits).cpu().numpy()

        val_auc = roc_auc_score(y_val, val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = torch.sigmoid(test_logits).cpu().numpy()

    return evaluate_metrics(y_test, test_preds)


def run_solution_1():
    """Solution 1: MLP baseline without graph structure."""
    print("\n" + "=" * 70)
    print("SOLUTION 1: MLP BASELINE (NO GRAPH STRUCTURE)")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs = load_data()

    all_results = []
    for seed in SEEDS:
        metrics = train_mlp(train_graphs, val_graphs, test_graphs, seed)
        all_results.append(metrics)
        print(f"  Seed {seed}: AUC={metrics['auc']:.4f}, Recall={metrics['recall']:.4f}")

    stats = {metric: {"mean": np.mean([r[metric] for r in all_results]),
                      "std": np.std([r[metric] for r in all_results])}
             for metric in ["auc", "f1", "precision", "recall"]}

    print(f"\nMLP (5 seeds):")
    print(f"  Test AUC:    {stats['auc']['mean']:.4f} ± {stats['auc']['std']:.4f}")
    print(f"  Test Recall: {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")

    return stats


# ==============================================================================
# SOLUTION 3: BETTER GRAPH CONSTRUCTION (HIGHER THRESHOLD)
# ==============================================================================

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.norms.append(BatchNorm(out_channels))

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SOZClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, classifier_dropout):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.classifier(h).squeeze(-1)


def prune_edges_by_threshold(graphs, threshold):
    """Remove edges below correlation threshold."""
    from torch_geometric.data import Data
    pruned_graphs = []
    for g in graphs:
        # Get edge weights (correlations)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            edge_attr = g.edge_attr
            if edge_attr.dim() == 2:
                edge_weights = edge_attr[:, 0]  # Use first column
            else:
                edge_weights = edge_attr

            mask = edge_weights >= threshold

            # Handle case where mask might reduce to scalar
            if mask.sum() == 0:
                # Keep at least some edges
                mask = edge_weights >= edge_weights.median()

            new_edge_index = g.edge_index[:, mask]
            new_edge_attr = edge_attr[mask] if edge_attr.dim() == 1 else edge_attr[mask, :]
        else:
            new_edge_index = g.edge_index
            new_edge_attr = g.edge_attr

        # Create new graph with pruned edges
        new_g = Data(
            x=g.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=g.y,
            n_soz=g.n_soz,
            split=g.split
        )
        pruned_graphs.append(new_g)

    return pruned_graphs


def train_gnn(train_graphs, val_graphs, test_graphs, seed):
    """Train standard GNN."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    n_features = train_graphs[0].x.shape[1]
    class_weights = compute_class_weights(train_graphs)

    model = SOZClassifier(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CLASSIFIER_DROPOUT).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            labels = batch.y.float()
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                all_preds.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        val_auc = roc_auc_score(all_labels, all_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    return evaluate_metrics(all_labels, all_preds)


def run_solution_3():
    """Solution 3: Higher correlation threshold for edges."""
    print("\n" + "=" * 70)
    print("SOLUTION 3: BETTER GRAPH CONSTRUCTION (HIGHER THRESHOLD)")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs = load_data()

    # Test different thresholds
    thresholds = [0.4, 0.5, 0.6]
    results = {}

    for thresh in thresholds:
        print(f"\n  Threshold: {thresh}")
        train_pruned = prune_edges_by_threshold(train_graphs, thresh)
        val_pruned = prune_edges_by_threshold(val_graphs, thresh)
        test_pruned = prune_edges_by_threshold(test_graphs, thresh)

        # Check edge reduction
        orig_edges = sum(g.edge_index.shape[1] for g in train_graphs)
        new_edges = sum(g.edge_index.shape[1] for g in train_pruned)
        print(f"    Edges: {orig_edges} -> {new_edges} ({100*new_edges/orig_edges:.1f}%)")

        all_results = []
        for seed in SEEDS:
            metrics = train_gnn(train_pruned, val_pruned, test_pruned, seed)
            all_results.append(metrics)

        stats = {metric: {"mean": np.mean([r[metric] for r in all_results]),
                          "std": np.std([r[metric] for r in all_results])}
                 for metric in ["auc", "f1", "precision", "recall"]}

        print(f"    AUC: {stats['auc']['mean']:.4f} ± {stats['auc']['std']:.4f}, "
              f"Recall: {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")

        results[thresh] = stats

    return results


# ==============================================================================
# SOLUTION 4: STACKING ENSEMBLE (RF + GNN -> META-LEARNER)
# ==============================================================================

def run_solution_4():
    """Solution 4: Stacking ensemble with RF + GNN predictions."""
    print("\n" + "=" * 70)
    print("SOLUTION 4: STACKING ENSEMBLE (RF + GNN -> META-LEARNER)")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs = load_data()

    X_train, y_train = graphs_to_flat_arrays(train_graphs)
    X_val, y_val = graphs_to_flat_arrays(val_graphs)
    X_test, y_test = graphs_to_flat_arrays(test_graphs)

    # Train RF
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_train_pred = rf.predict_proba(X_train)[:, 1]
    rf_val_pred = rf.predict_proba(X_val)[:, 1]
    rf_test_pred = rf.predict_proba(X_test)[:, 1]

    # Train GNN and get predictions
    print("  Training GNN...")

    # We need to get GNN predictions for all nodes
    # Train on train, predict on train/val/test
    all_gnn_preds = {split: [] for split in ["train", "val", "test"]}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)

        n_features = train_graphs[0].x.shape[1]
        class_weights = compute_class_weights(train_graphs)

        model = SOZClassifier(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CLASSIFIER_DROPOUT).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_auc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index)
                labels = batch.y.float()
                weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
                weights = weights.to(DEVICE)
                loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            val_preds = []
            with torch.no_grad():
                for g in val_graphs:
                    g = g.to(DEVICE)
                    logits = model(g.x, g.edge_index)
                    val_preds.extend(torch.sigmoid(logits).cpu().tolist())

            val_auc = roc_auc_score(y_val, val_preds)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Get predictions for all splits
        model.load_state_dict(best_state)
        model.eval()

        for split, graphs in [("train", train_graphs), ("val", val_graphs), ("test", test_graphs)]:
            preds = []
            with torch.no_grad():
                for g in graphs:
                    g = g.to(DEVICE)
                    logits = model(g.x, g.edge_index)
                    preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_gnn_preds[split].append(preds)

    # Average GNN predictions across seeds
    gnn_train_pred = np.mean(all_gnn_preds["train"], axis=0)
    gnn_val_pred = np.mean(all_gnn_preds["val"], axis=0)
    gnn_test_pred = np.mean(all_gnn_preds["test"], axis=0)

    print(f"  RF Test AUC: {roc_auc_score(y_test, rf_test_pred):.4f}")
    print(f"  GNN Test AUC: {roc_auc_score(y_test, gnn_test_pred):.4f}")

    # Stack: combine RF and GNN predictions
    print("\n  Training meta-learner...")

    # Option A: Just RF + GNN probs
    X_meta_train = np.column_stack([rf_train_pred, gnn_train_pred])
    X_meta_val = np.column_stack([rf_val_pred, gnn_val_pred])
    X_meta_test = np.column_stack([rf_test_pred, gnn_test_pred])

    meta = LogisticRegression(class_weight="balanced", random_state=42)
    meta.fit(X_meta_train, y_train)

    meta_test_pred = meta.predict_proba(X_meta_test)[:, 1]
    meta_metrics = evaluate_metrics(y_test, meta_test_pred)

    print(f"\n  Stacking (RF + GNN):")
    print(f"    Test AUC:    {meta_metrics['auc']:.4f}")
    print(f"    Test Recall: {meta_metrics['recall']:.4f}")

    # Option B: RF + GNN + original features
    X_meta_train_full = np.column_stack([rf_train_pred, gnn_train_pred, X_train])
    X_meta_test_full = np.column_stack([rf_test_pred, gnn_test_pred, X_test])

    meta_full = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    meta_full.fit(X_meta_train_full, y_train)

    meta_full_test_pred = meta_full.predict_proba(X_meta_test_full)[:, 1]
    meta_full_metrics = evaluate_metrics(y_test, meta_full_test_pred)

    print(f"\n  Stacking (RF + GNN + features):")
    print(f"    Test AUC:    {meta_full_metrics['auc']:.4f}")
    print(f"    Test Recall: {meta_full_metrics['recall']:.4f}")

    return {"stacking_simple": meta_metrics, "stacking_full": meta_full_metrics}


# ==============================================================================
# SOLUTION 5: RF PREDICTIONS AS NODE FEATURE
# ==============================================================================

def run_solution_5():
    """Solution 5: Add RF predictions as additional node feature."""
    print("\n" + "=" * 70)
    print("SOLUTION 5: RF PREDICTIONS AS NODE FEATURE")
    print("=" * 70)

    train_graphs, val_graphs, test_graphs = load_data()

    X_train, y_train = graphs_to_flat_arrays(train_graphs)
    X_val, y_val = graphs_to_flat_arrays(val_graphs)
    X_test, y_test = graphs_to_flat_arrays(test_graphs)

    # Train RF
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Get RF predictions for each graph's nodes
    def add_rf_feature(graphs, rf):
        enhanced_graphs = []
        for g in graphs:
            X = g.x.numpy()
            rf_prob = rf.predict_proba(X)[:, 1]
            rf_feat = torch.tensor(rf_prob, dtype=torch.float32).unsqueeze(-1)
            new_x = torch.cat([g.x, rf_feat], dim=-1)

            from torch_geometric.data import Data
            new_g = Data(
                x=new_x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                y=g.y,
                n_soz=g.n_soz,
                split=g.split
            )
            enhanced_graphs.append(new_g)
        return enhanced_graphs

    train_enhanced = add_rf_feature(train_graphs, rf)
    val_enhanced = add_rf_feature(val_graphs, rf)
    test_enhanced = add_rf_feature(test_graphs, rf)

    print(f"  Features: {train_graphs[0].x.shape[1]} -> {train_enhanced[0].x.shape[1]}")

    # Train GNN with enhanced features
    print("  Training GNN with RF feature...")

    all_results = []
    for seed in SEEDS:
        metrics = train_gnn(train_enhanced, val_enhanced, test_enhanced, seed)
        all_results.append(metrics)
        print(f"    Seed {seed}: AUC={metrics['auc']:.4f}, Recall={metrics['recall']:.4f}")

    stats = {metric: {"mean": np.mean([r[metric] for r in all_results]),
                      "std": np.std([r[metric] for r in all_results])}
             for metric in ["auc", "f1", "precision", "recall"]}

    print(f"\n  GNN + RF Feature (5 seeds):")
    print(f"    Test AUC:    {stats['auc']['mean']:.4f} ± {stats['auc']['std']:.4f}")
    print(f"    Test Recall: {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")

    return stats


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("TESTING SOLUTIONS TO IMPROVE GNN AUC")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nBaseline Reference:")
    print("  Random Forest:  AUC = 0.847")
    print("  GNN (baseline): AUC = 0.763 ± 0.008")

    results = {}

    # Solution 1: MLP baseline
    results["mlp"] = run_solution_1()

    # Solution 3: Better graph construction
    results["threshold"] = run_solution_3()

    # Solution 4: Stacking ensemble
    results["stacking"] = run_solution_4()

    # Solution 5: RF as feature
    results["rf_feature"] = run_solution_5()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n| Solution | Test AUC | Test Recall |")
    print("|----------|----------|-------------|")
    print(f"| Random Forest (baseline) | 0.847 | 0.436 |")
    print(f"| GNN (baseline) | 0.763 ± 0.008 | 0.623 ± 0.014 |")
    print(f"| MLP (no graph) | {results['mlp']['auc']['mean']:.3f} ± {results['mlp']['auc']['std']:.3f} | {results['mlp']['recall']['mean']:.3f} ± {results['mlp']['recall']['std']:.3f} |")

    for thresh, stats in results["threshold"].items():
        print(f"| GNN (thresh={thresh}) | {stats['auc']['mean']:.3f} ± {stats['auc']['std']:.3f} | {stats['recall']['mean']:.3f} ± {stats['recall']['std']:.3f} |")

    print(f"| Stacking (RF+GNN) | {results['stacking']['stacking_simple']['auc']:.3f} | {results['stacking']['stacking_simple']['recall']:.3f} |")
    print(f"| Stacking (RF+GNN+feat) | {results['stacking']['stacking_full']['auc']:.3f} | {results['stacking']['stacking_full']['recall']:.3f} |")
    print(f"| GNN + RF feature | {results['rf_feature']['auc']['mean']:.3f} ± {results['rf_feature']['auc']['std']:.3f} | {results['rf_feature']['recall']['mean']:.3f} ± {results['rf_feature']['recall']['std']:.3f} |")

    # Save results
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "results_solutions.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
