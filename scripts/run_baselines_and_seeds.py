"""
Baselines + Multi-Seed GNN Evaluation
======================================

1. Classical ML baselines (Random Forest, Logistic Regression, SVM)
2. GNN with multiple seeds for confidence intervals

Output: results_baselines_seeds.json with all metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

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

# Try combined dataset first, fall back to original
GRAPHS_PT_COMBINED = PROJECT_DIR / "data" / "processed_combined" / "graphs" / "graphs_combined.pt"
GRAPHS_PT_ORIGINAL = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs.pt"
GRAPHS_PT = GRAPHS_PT_COMBINED if GRAPHS_PT_COMBINED.exists() else GRAPHS_PT_ORIGINAL
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GNN hyperparameters (from previous tuning)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.21
CLASSIFIER_DROPOUT = 0.55
FINETUNE_LR = 6.1e-4
WEIGHT_DECAY = 9e-5
BATCH_SIZE = 16
AUG_EDGE_DROP = 0.076
AUG_FEAT_NOISE = 0.011
AUG_FEAT_MASK = 0.10
FINETUNE_EPOCHS = 150
PATIENCE = 20

# Seeds for multi-run evaluation
SEEDS = [42, 123, 456, 789, 2024]


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


def load_data():
    """Load graphs and return train/val/test splits."""
    all_graphs = torch.load(GRAPHS_PT, weights_only=False)
    labeled = [g for g in all_graphs if g.n_soz > 0]
    train = [g for g in labeled if g.split == "train"]
    val = [g for g in labeled if g.split == "val"]
    test = [g for g in labeled if g.split == "test"]
    return train, val, test


def graphs_to_flat_arrays(graphs):
    """Convert PyG graphs to flat numpy arrays for sklearn."""
    X_list, y_list = [], []
    for g in graphs:
        X_list.append(g.x.numpy())
        y_list.append(g.y.numpy())
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def compute_class_weights(graphs):
    n_soz = sum(g.y.sum().item() for g in graphs)
    n_total = sum(g.y.shape[0] for g in graphs)
    n_non_soz = n_total - n_soz
    return torch.tensor([n_total / (2 * n_non_soz), n_total / (2 * n_soz)], dtype=torch.float32)


def augment_batch(x, edge_index):
    if AUG_EDGE_DROP > 0:
        edge_index, _ = dropout_edge(edge_index, p=AUG_EDGE_DROP, training=True)
    if AUG_FEAT_NOISE > 0:
        x = x + torch.randn_like(x) * AUG_FEAT_NOISE
    if AUG_FEAT_MASK > 0:
        mask = torch.rand_like(x) > AUG_FEAT_MASK
        x = x * mask.float()
    return x, edge_index


def evaluate_metrics(y_true, y_pred_proba):
    """Compute all metrics."""
    auc = roc_auc_score(y_true, y_pred_proba)
    y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


# ============================================================================
# CLASSICAL ML BASELINES
# ============================================================================

def run_baselines(train_graphs, val_graphs, test_graphs):
    """Train and evaluate classical ML baselines."""
    print("\n" + "=" * 70)
    print("Classical ML Baselines")
    print("=" * 70)

    X_train, y_train = graphs_to_flat_arrays(train_graphs)
    X_val, y_val = graphs_to_flat_arrays(val_graphs)
    X_test, y_test = graphs_to_flat_arrays(test_graphs)

    print(f"Train: {X_train.shape[0]} nodes, {y_train.sum()} SOZ ({100*y_train.mean():.1f}%)")
    print(f"Val:   {X_val.shape[0]} nodes, {y_val.sum()} SOZ ({100*y_val.mean():.1f}%)")
    print(f"Test:  {X_test.shape[0]} nodes, {y_test.sum()} SOZ ({100*y_test.mean():.1f}%)")

    results = {}

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = evaluate_metrics(y_test, rf_pred)
    print(f"  Test AUC: {results['RandomForest']['auc']:.4f}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_test)[:, 1]
    results["LogisticRegression"] = evaluate_metrics(y_test, lr_pred)
    print(f"  Test AUC: {results['LogisticRegression']['auc']:.4f}")

    # SVM
    print("\nTraining SVM (RBF kernel)...")
    svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    svm.fit(X_train, y_train)
    svm_pred = svm.predict_proba(X_test)[:, 1]
    results["SVM"] = evaluate_metrics(y_test, svm_pred)
    print(f"  Test AUC: {results['SVM']['auc']:.4f}")

    return results


# ============================================================================
# GNN WITH MULTIPLE SEEDS
# ============================================================================

def train_gnn_single_seed(train_graphs, val_graphs, test_graphs, seed):
    """Train GNN with a specific seed and return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    n_features = train_graphs[0].x.shape[1]
    class_weights = compute_class_weights(train_graphs)

    model = SOZClassifier(
        n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT, CLASSIFIER_DROPOUT
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(FINETUNE_EPOCHS):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            x, edge_index = augment_batch(batch.x, batch.edge_index)
            logits = model(x, edge_index)
            labels = batch.y.float()
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                all_preds.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Evaluate on test
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


def run_gnn_multi_seed(train_graphs, val_graphs, test_graphs):
    """Run GNN with multiple seeds and compute mean ± std."""
    print("\n" + "=" * 70)
    print(f"GNN Multi-Seed Evaluation ({len(SEEDS)} seeds)")
    print("=" * 70)

    all_results = []
    for seed in tqdm(SEEDS, desc="Running seeds"):
        metrics = train_gnn_single_seed(train_graphs, val_graphs, test_graphs, seed)
        all_results.append(metrics)
        print(f"  Seed {seed}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

    # Compute statistics
    stats = {}
    for metric in ["auc", "f1", "precision", "recall"]:
        values = [r[metric] for r in all_results]
        stats[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values
        }

    return stats, all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Baselines + Multi-Seed GNN Evaluation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t_start = time.perf_counter()

    # Load data
    print("\nLoading data...")
    train_graphs, val_graphs, test_graphs = load_data()
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Run baselines
    baseline_results = run_baselines(train_graphs, val_graphs, test_graphs)

    # Run GNN with multiple seeds
    gnn_stats, gnn_all_results = run_gnn_multi_seed(train_graphs, val_graphs, test_graphs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nClassical ML Baselines:")
    for name, metrics in baseline_results.items():
        print(f"  {name:20s}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    print(f"\nGNN (GraphSAGE) - {len(SEEDS)} seeds:")
    print(f"  Test AUC:       {gnn_stats['auc']['mean']:.4f} ± {gnn_stats['auc']['std']:.4f}")
    print(f"  Test F1:        {gnn_stats['f1']['mean']:.4f} ± {gnn_stats['f1']['std']:.4f}")
    print(f"  Test Precision: {gnn_stats['precision']['mean']:.4f} ± {gnn_stats['precision']['std']:.4f}")
    print(f"  Test Recall:    {gnn_stats['recall']['mean']:.4f} ± {gnn_stats['recall']['std']:.4f}")

    # Save results
    results = {
        "baselines": baseline_results,
        "gnn_stats": gnn_stats,
        "gnn_all_seeds": gnn_all_results,
        "seeds": SEEDS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "results_baselines_seeds.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
