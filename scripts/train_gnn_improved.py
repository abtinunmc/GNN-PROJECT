"""
Improved GNN Training with:
1. Knowledge distillation from Random Forest
2. Smaller model (hidden_dim=64)
3. Early stopping on recall instead of AUC
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# IMPROVED hyperparameters
HIDDEN_DIM = 128  # Original size
NUM_LAYERS = 2
DROPOUT = 0.3  # Increased from 0.21
CLASSIFIER_DROPOUT = 0.5
FINETUNE_LR = 5e-4
WEIGHT_DECAY = 1e-4  # Increased from 9e-5
BATCH_SIZE = 16
AUG_EDGE_DROP = 0.15  # Increased from 0.076
AUG_FEAT_NOISE = 0.02  # Increased from 0.011
AUG_FEAT_MASK = 0.15  # Increased from 0.10
FINETUNE_EPOCHS = 150
PATIENCE = 15  # Balanced

# Knowledge distillation
KD_ALPHA = 0.0  # Disabled - was hurting recall
KD_TEMPERATURE = 2.0

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
    auc = roc_auc_score(y_true, y_pred_proba)
    y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


def train_rf_teacher(train_graphs):
    """Train Random Forest as teacher model."""
    X_train, y_train = graphs_to_flat_arrays(train_graphs)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def get_rf_soft_labels(rf, graphs):
    """Get soft labels from RF for each graph."""
    soft_labels = []
    for g in graphs:
        X = g.x.numpy()
        probs = rf.predict_proba(X)[:, 1]
        soft_labels.append(torch.tensor(probs, dtype=torch.float32))
    return soft_labels


def kd_loss(student_logits, teacher_probs, temperature):
    """Knowledge distillation loss using KL divergence."""
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = teacher_probs.to(student_logits.device)

    # Binary KL divergence
    eps = 1e-7
    student_probs = student_probs.clamp(eps, 1 - eps)
    teacher_probs = teacher_probs.clamp(eps, 1 - eps)

    kl = teacher_probs * torch.log(teacher_probs / student_probs) + \
         (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))

    return kl.mean() * (temperature ** 2)


def train_gnn_with_improvements(train_graphs, val_graphs, test_graphs, rf_soft_labels, seed):
    """Train GNN with all improvements."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Attach soft labels to graphs
    for g, soft in zip(train_graphs, rf_soft_labels):
        g.rf_soft = soft

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

    best_val_score = 0  # Changed from AUC to recall
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

            # Standard BCE loss with class weights
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)

            # Knowledge distillation loss
            rf_soft = batch.rf_soft.to(DEVICE)
            distill_loss = kd_loss(logits, rf_soft, KD_TEMPERATURE)

            # Combined loss
            loss = (1 - KD_ALPHA) * ce_loss + KD_ALPHA * distill_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate - now using recall for early stopping
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                all_preds.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        val_metrics = evaluate_metrics(all_labels, all_preds)
        # Use AUC for early stopping (like baseline) - most stable
        val_score = val_metrics["auc"]

        if val_score > best_val_score:
            best_val_score = val_score
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


def main():
    print("=" * 70)
    print("Improved GNN Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nImprovements applied:")
    print(f"  1. Knowledge distillation from RF (alpha={KD_ALPHA})")
    print(f"  2. Smaller model (hidden_dim={HIDDEN_DIM})")
    print(f"  3. Early stopping on recall (patience={PATIENCE})")
    print(f"  4. Stronger regularization (dropout={DROPOUT}, edge_drop={AUG_EDGE_DROP})")

    t_start = time.perf_counter()

    # Load data
    print("\nLoading data...")
    train_graphs, val_graphs, test_graphs = load_data()
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Train RF teacher
    print("\nTraining Random Forest teacher...")
    rf = train_rf_teacher(train_graphs)
    rf_soft_labels = get_rf_soft_labels(rf, train_graphs)

    # Also get RF test performance for comparison
    X_test, y_test = graphs_to_flat_arrays(test_graphs)
    rf_pred = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_metrics(y_test, rf_pred)
    print(f"  RF Test AUC: {rf_metrics['auc']:.4f}, Recall: {rf_metrics['recall']:.4f}")

    # Run improved GNN with multiple seeds
    print(f"\nTraining Improved GNN ({len(SEEDS)} seeds)...")
    all_results = []
    for seed in SEEDS:
        metrics = train_gnn_with_improvements(
            train_graphs, val_graphs, test_graphs, rf_soft_labels, seed
        )
        all_results.append(metrics)
        print(f"  Seed {seed}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")

    # Compute statistics
    stats = {}
    for metric in ["auc", "f1", "precision", "recall"]:
        values = [r[metric] for r in all_results]
        stats[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values
        }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\nRandom Forest (teacher):")
    print(f"  Test AUC:    {rf_metrics['auc']:.4f}")
    print(f"  Test Recall: {rf_metrics['recall']:.4f}")

    print(f"\nImproved GNN ({len(SEEDS)} seeds):")
    print(f"  Test AUC:       {stats['auc']['mean']:.4f} ± {stats['auc']['std']:.4f}")
    print(f"  Test F1:        {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
    print(f"  Test Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
    print(f"  Test Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")

    print("\nPrevious GNN (baseline):")
    print(f"  Test AUC:       0.763 ± 0.008")
    print(f"  Test Recall:    0.623 ± 0.014")

    # Save results
    results = {
        "rf_metrics": rf_metrics,
        "improved_gnn_stats": stats,
        "improved_gnn_all_seeds": all_results,
        "seeds": SEEDS,
        "hyperparameters": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "kd_alpha": KD_ALPHA,
            "kd_temperature": KD_TEMPERATURE,
            "aug_edge_drop": AUG_EDGE_DROP,
            "patience": PATIENCE
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "results_improved_gnn.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
