"""
Train GNN with Augmented Data
=============================

Uses the time-shift augmented graphs with best hyperparameters
from previous tuning.

Input
-----
    data/processed/graphs/graphs_augmented.pt
    data/processed/h5/features.h5  (for pretraining)

Output
------
    data/processed/models/soz_classifier_augmented.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import dropout_edge

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"
GRAPHS_PT = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs_augmented.pt"
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best hyperparameters from tuning (with augmentation)
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.23
CLASSIFIER_DROPOUT = 0.52
PRETRAIN_LR = 1.7e-4
FINETUNE_LR = 3.9e-4
WEIGHT_DECAY = 1.6e-5
BATCH_SIZE = 8
FREEZE_EPOCHS = 5
CORRELATION_THRESHOLD = 0.44

# Augmentation during training
AUG_EDGE_DROP = 0.017
AUG_FEAT_NOISE = 0.037
AUG_FEAT_MASK = 0.010

# Training settings
PRETRAIN_EPOCHS = 150
FINETUNE_EPOCHS = 100
PATIENCE = 15


# ============================================================================
# MODELS
# ============================================================================

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


class PretrainModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.predictor(h)


class SOZClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout,
                 classifier_dropout, pretrain_in_channels=None):
        super().__init__()
        self.input_proj = None
        encoder_in = in_channels
        if pretrain_in_channels is not None and pretrain_in_channels != in_channels:
            self.input_proj = nn.Linear(in_channels, pretrain_in_channels)
            encoder_in = pretrain_in_channels

        self.encoder = GNNEncoder(encoder_in, hidden_channels, hidden_channels, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index):
        if self.input_proj is not None:
            x = self.input_proj(x)
        h = self.encoder(x, edge_index)
        return self.classifier(h).squeeze(-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# ============================================================================
# DATA LOADING
# ============================================================================

def build_edge_index(corr_matrix, threshold):
    n = corr_matrix.shape[0]
    mask = np.triu(np.abs(corr_matrix) >= threshold, k=1)
    src, dst = np.where(mask)
    if len(src) == 0:
        idx = np.arange(n)
        return torch.tensor(np.stack([idx, idx]), dtype=torch.long)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge_index, dtype=torch.long)


def zscore(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((features - mean) / std).astype(np.float32)


def load_pretrain_data():
    """Load window pairs for pretraining."""
    pairs = []
    with h5py.File(FEATURES_H5, "r") as f:
        for rec_name in f.keys():
            grp = f[rec_name]
            window_feats = grp["window_node_features"][:]
            n_windows = window_feats.shape[0]
            if n_windows < 2:
                continue
            corr_matrix = grp["edge_features/correlation"][:, :, 0]
            edge_index = build_edge_index(corr_matrix, CORRELATION_THRESHOLD)
            for t in range(n_windows - 1):
                x_in = zscore(window_feats[t])
                x_out = zscore(window_feats[t + 1])
                pairs.append(Data(x=torch.from_numpy(x_in), edge_index=edge_index,
                                  y=torch.from_numpy(x_out)))
    return pairs


def load_labeled_graphs():
    """Load augmented labeled graphs for fine-tuning."""
    all_graphs = torch.load(GRAPHS_PT, weights_only=False)
    labeled = [g for g in all_graphs if g.n_soz > 0]
    train = [g for g in labeled if g.split == "train"]
    val = [g for g in labeled if g.split == "val"]
    test = [g for g in labeled if g.split == "test"]
    return train, val, test


def compute_class_weights(graphs):
    n_soz = sum(g.y.sum().item() for g in graphs)
    n_total = sum(g.y.shape[0] for g in graphs)
    n_non_soz = n_total - n_soz
    return torch.tensor([n_total / (2 * n_non_soz), n_total / (2 * n_soz)], dtype=torch.float32)


# ============================================================================
# AUGMENTATION
# ============================================================================

def augment_batch(x, edge_index):
    """Apply augmentation during training."""
    if AUG_EDGE_DROP > 0:
        edge_index, _ = dropout_edge(edge_index, p=AUG_EDGE_DROP, training=True)
    if AUG_FEAT_NOISE > 0:
        x = x + torch.randn_like(x) * AUG_FEAT_NOISE
    if AUG_FEAT_MASK > 0:
        mask = torch.rand_like(x) > AUG_FEAT_MASK
        x = x * mask.float()
    return x, edge_index


# ============================================================================
# TRAINING
# ============================================================================

def pretrain(model, train_loader, val_loader):
    """Pretrain and return best encoder state."""
    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(PRETRAIN_EPOCHS), desc="Pretraining")
    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch.x, batch.edge_index)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

        pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.encoder.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return best_state, best_loss


def finetune(model, encoder_state, train_loader, val_loader, class_weights):
    """Fine-tune and return best model state."""
    model.encoder.load_state_dict(encoder_state)
    optimizer = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    best_auc = 0
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(FINETUNE_EPOCHS), desc="Fine-tuning")
    for epoch in pbar:
        if epoch == 0 and FREEZE_EPOCHS > 0:
            model.freeze_encoder()
        elif epoch == FREEZE_EPOCHS:
            model.unfreeze_encoder()

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

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                all_preds.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(batch.y.cpu().tolist())

        if len(set(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_preds)
        else:
            val_auc = 0.5

        pbar.set_postfix({"val_auc": f"{val_auc:.4f}"})

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return best_state, best_auc


def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    test_auc = roc_auc_score(all_labels, all_preds)
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    test_f1 = f1_score(all_labels, preds_binary)
    return test_auc, test_f1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Training GNN with Augmented Data")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 70)

    t_start = time.perf_counter()

    # Load pretrain data
    print("\nLoading pretraining data...")
    pretrain_data = load_pretrain_data()
    n_train = int(0.8 * len(pretrain_data))
    indices = torch.randperm(len(pretrain_data)).tolist()
    pretrain_train = [pretrain_data[i] for i in indices[:n_train]]
    pretrain_val = [pretrain_data[i] for i in indices[n_train:]]
    print(f"  Pretrain: {len(pretrain_train)} train, {len(pretrain_val)} val")

    pretrain_train_loader = DataLoader(pretrain_train, batch_size=BATCH_SIZE, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val, batch_size=BATCH_SIZE)

    # Pretrain
    print("\nPretraining...")
    n_features = pretrain_data[0].x.shape[1]
    pretrain_model = PretrainModel(n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    encoder_state, pretrain_loss = pretrain(pretrain_model, pretrain_train_loader, pretrain_val_loader)
    print(f"  Best pretrain val loss: {pretrain_loss:.4f}")

    # Load fine-tune data (augmented)
    print("\nLoading augmented fine-tuning data...")
    train_graphs, val_graphs, test_graphs = load_labeled_graphs()
    print(f"  Train: {len(train_graphs)} (augmented), Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    # Fine-tune
    print("\nFine-tuning...")
    graph_n_features = train_graphs[0].x.shape[1]
    class_weights = compute_class_weights(train_graphs)
    print(f"  Class weights: {class_weights.tolist()}")

    classifier = SOZClassifier(
        graph_n_features, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
        CLASSIFIER_DROPOUT, pretrain_in_channels=n_features
    ).to(DEVICE)

    best_state, val_auc = finetune(classifier, encoder_state, train_loader, val_loader, class_weights)
    print(f"  Best val AUC: {val_auc:.4f}")

    # Evaluate on test
    classifier.load_state_dict(best_state)
    test_auc, test_f1 = evaluate(classifier, test_loader)

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC:       {test_auc:.4f}")
    print(f"  Test F1:        {test_f1:.4f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "soz_classifier_augmented.pt"
    torch.save(best_state, output_path)
    print(f"\nSaved: {output_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Log
    backend = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "CPU"
    append_project_log(
        stage="train_augmented",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Training graphs: {len(train_graphs)} (augmented)",
            f"Best val AUC: {val_auc:.4f}",
            f"Test AUC: {test_auc:.4f}",
            f"Test F1: {test_f1:.4f}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
