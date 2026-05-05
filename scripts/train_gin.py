"""
GIN (Graph Isomorphism Network) for SOZ Localization
=====================================================

Uses Graph Isomorphism Networks - the most expressive GNN architecture
under the Weisfeiler-Lehman test. GIN uses MLPs and sum aggregation
for maximum expressiveness.

Key advantages of GIN:
    1. Provably most expressive among message-passing GNNs
    2. Uses MLPs instead of linear transformations
    3. Sum aggregation captures multiset structure
    4. Learnable epsilon for self-loop weighting

Architecture
------------
    Input → GIN (MLP) → GIN → Classifier → SOZ prediction

Input
-----
    data/processed/h5/features.h5    (for pretraining)
    data/processed/graphs/graphs.pt  (for fine-tuning)

Output
------
    data/processed/models/gin_encoder.pt
    data/processed/models/gin_classifier.pt
    data/processed/figures/gin_training.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.utils import dropout_edge

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"
GRAPHS_PT = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs.pt"
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_DIR / "data" / "processed" / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
NUM_LAYERS = 3          # GIN often benefits from more layers
MLP_HIDDEN_RATIO = 2    # MLP hidden = HIDDEN_DIM * ratio
DROPOUT = 0.2
CLASSIFIER_DROPOUT = 0.5
TRAIN_EPS = True        # Learn epsilon parameter

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
PRETRAIN_EPOCHS = 250
FINETUNE_EPOCHS = 150
BATCH_SIZE = 32
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 25
FREEZE_EPOCHS = 15

# ---------------------------------------------------------------------------
# Data augmentation (gentler for GIN)
# ---------------------------------------------------------------------------
USE_AUGMENTATION = True
EDGE_DROP_RATE = 0.05   # Lighter augmentation
FEATURE_MASK_RATE = 0.05

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD = 0.277


# ============================================================================
# GIN MODEL
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron for GIN."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        mlp_hidden_ratio: int = 2,
        dropout: float = 0.2,
        train_eps: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        mlp_hidden = hidden_channels * mlp_hidden_ratio

        # First layer
        mlp1 = MLP(in_channels, mlp_hidden, hidden_channels, dropout)
        self.convs.append(GINConv(mlp1, train_eps=train_eps))
        self.norms.append(BatchNorm(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            mlp = MLP(hidden_channels, mlp_hidden, hidden_channels, dropout)
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.norms.append(BatchNorm(hidden_channels))

        # Last layer
        if num_layers > 1:
            mlp_last = MLP(hidden_channels, mlp_hidden, out_channels, dropout)
            self.convs.append(GINConv(mlp_last, train_eps=train_eps))
            self.norms.append(BatchNorm(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GINPretrainModel(nn.Module):
    """GIN model for self-supervised pretraining."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        mlp_hidden_ratio: int = 2,
        dropout: float = 0.2,
        train_eps: bool = True,
    ):
        super().__init__()
        self.encoder = GINEncoder(
            in_channels, hidden_channels, hidden_channels,
            num_layers, mlp_hidden_ratio, dropout, train_eps
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        return self.predictor(h)


class GINClassifier(nn.Module):
    """GIN model for SOZ classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        mlp_hidden_ratio: int = 2,
        dropout: float = 0.2,
        classifier_dropout: float = 0.5,
        train_eps: bool = True,
        pretrain_in_channels: int = None,
    ):
        super().__init__()

        # Input projection if needed
        self.input_proj = None
        encoder_in = in_channels
        if pretrain_in_channels is not None and pretrain_in_channels != in_channels:
            self.input_proj = nn.Linear(in_channels, pretrain_in_channels)
            encoder_in = pretrain_in_channels

        self.encoder = GINEncoder(
            encoder_in, hidden_channels, hidden_channels,
            num_layers, mlp_hidden_ratio, dropout, train_eps
        )

        # Deeper classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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

def build_edge_index(corr_matrix: np.ndarray, threshold: float) -> torch.Tensor:
    n = corr_matrix.shape[0]
    mask = np.triu(np.abs(corr_matrix) >= threshold, k=1)
    src, dst = np.where(mask)
    if len(src) == 0:
        idx = np.arange(n)
        return torch.tensor(np.stack([idx, idx]), dtype=torch.long)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge_index, dtype=torch.long)


def zscore(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((features - mean) / std).astype(np.float32)


def load_pretrain_data() -> list[Data]:
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
                pairs.append(Data(
                    x=torch.from_numpy(x_in),
                    edge_index=edge_index,
                    y=torch.from_numpy(x_out),
                ))
    return pairs


def load_labeled_graphs():
    all_graphs = torch.load(GRAPHS_PT, weights_only=False)
    labeled = [g for g in all_graphs if g.n_soz > 0]
    train = [g for g in labeled if g.split == "train"]
    val = [g for g in labeled if g.split == "val"]
    test = [g for g in labeled if g.split == "test"]
    return train, val, test


def compute_class_weights(graphs: list) -> torch.Tensor:
    n_soz = sum(g.y.sum().item() for g in graphs)
    n_total = sum(g.y.shape[0] for g in graphs)
    n_non_soz = n_total - n_soz
    return torch.tensor([n_total / (2 * n_non_soz), n_total / (2 * n_soz)])


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def pretrain(model: GINPretrainModel, train_data: list, val_data: list) -> dict:
    """Self-supervised pretraining."""
    print("\n" + "=" * 70)
    print("Stage 1: Self-Supervised Pretraining (GIN)")
    print("=" * 70)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    best_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            edge_index = batch.edge_index
            x = batch.x

            # Gentle augmentation
            if USE_AUGMENTATION:
                edge_index, _ = dropout_edge(edge_index, p=EDGE_DROP_RATE, training=True)
                mask = torch.rand_like(x) > FEATURE_MASK_RATE
                x = x * mask.float()

            optimizer.zero_grad()
            pred = model(x, edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch.x, batch.edge_index)
                val_loss += F.mse_loss(pred, batch.y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.encoder.state_dict()
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        "encoder_state": best_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_loss": best_loss,
    }


def finetune(
    model: GINClassifier,
    encoder_state: dict,
    train_graphs: list,
    val_graphs: list,
    test_graphs: list,
) -> dict:
    """Fine-tune for SOZ classification."""
    print("\n" + "=" * 70)
    print("Stage 2: Fine-tuning for SOZ Classification (GIN)")
    print("=" * 70)

    model.encoder.load_state_dict(encoder_state)
    print("Loaded pretrained encoder weights")

    train_loader = DataLoader(train_graphs, batch_size=min(BATCH_SIZE, len(train_graphs)), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=min(BATCH_SIZE, len(val_graphs)))
    test_loader = DataLoader(test_graphs, batch_size=min(BATCH_SIZE, len(test_graphs)))

    # Class-weighted BCE loss
    class_weights = compute_class_weights(train_graphs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(DEVICE))
    print(f"Class weights: non-SOZ={class_weights[0]:.2f}, SOZ={class_weights[1]:.2f}")

    optimizer = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    best_auc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    best_state = None

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        # Freeze/unfreeze encoder
        if epoch == 1 and FREEZE_EPOCHS > 0:
            model.freeze_encoder()
            print(f"Encoder frozen for first {FREEZE_EPOCHS} epochs")
        elif epoch == FREEZE_EPOCHS + 1:
            model.unfreeze_encoder()
            print("Encoder unfrozen")

        # Train
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(DEVICE)
            edge_index = batch.edge_index
            x = batch.x

            # Light augmentation during finetuning
            if USE_AUGMENTATION:
                edge_index, _ = dropout_edge(edge_index, p=EDGE_DROP_RATE, training=True)

            optimizer.zero_grad()
            logits = model(x, edge_index)
            labels = batch.y.float()
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch.x, batch.edge_index)
                labels = batch.y.float()
                loss = criterion(logits, labels)
                val_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Early stopping on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} AUC: {train_auc:.3f} | Val AUC: {val_auc:.3f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    model.load_state_dict(best_state)
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
    test_f1 = f1_score(all_labels, preds_binary, zero_division=0)
    test_precision = precision_score(all_labels, preds_binary, zero_division=0)
    test_recall = recall_score(all_labels, preds_binary, zero_division=0)

    print(f"\nTest Results:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")

    return {
        "model_state": best_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "best_val_auc": best_auc,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training(pretrain_results: dict, finetune_results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GIN (Graph Isomorphism Network) Training Results", fontsize=14, fontweight="bold")

    # Pretrain loss
    ax = axes[0, 0]
    epochs = range(1, len(pretrain_results["train_losses"]) + 1)
    ax.plot(epochs, pretrain_results["train_losses"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, pretrain_results["val_losses"], "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Pretraining: Signal Forecasting")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Finetune loss
    ax = axes[0, 1]
    epochs = range(1, len(finetune_results["train_losses"]) + 1)
    ax.plot(epochs, finetune_results["train_losses"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, finetune_results["val_losses"], "r-", label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Fine-tuning: Classification Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Finetune AUC
    ax = axes[1, 0]
    ax.plot(epochs, finetune_results["train_aucs"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, finetune_results["val_aucs"], "r-", label="Val", linewidth=2)
    ax.axhline(finetune_results["test_auc"], color="g", linestyle="--", linewidth=2,
               label=f"Test: {finetune_results['test_auc']:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Fine-tuning: ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 1]
    ax.axis("off")
    summary = f"""
    GIN Model Summary
    ─────────────────────────────
    Architecture:
      Hidden dim:       {HIDDEN_DIM}
      Num layers:       {NUM_LAYERS}
      MLP hidden ratio: {MLP_HIDDEN_RATIO}x
      Train epsilon:    {TRAIN_EPS}
      Dropout:          {DROPOUT}

    Training:
      Pretrain epochs:  {len(pretrain_results['train_losses'])}
      Finetune epochs:  {len(finetune_results['train_losses'])}
      Augmentation:     {USE_AUGMENTATION}

    Results:
      Best Val AUC:     {finetune_results['best_val_auc']:.4f}
      Test AUC:         {finetune_results['test_auc']:.4f}
      Test F1:          {finetune_results['test_f1']:.4f}
      Test Precision:   {finetune_results['test_precision']:.4f}
      Test Recall:      {finetune_results['test_recall']:.4f}
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()
    fig_path = FIGURES_DIR / "gin_training.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("GIN (Graph Isomorphism Network) for SOZ Localization")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {NUM_LAYERS} layers, {HIDDEN_DIM} hidden, MLP ratio {MLP_HIDDEN_RATIO}x")
    print(f"Train epsilon: {TRAIN_EPS}")
    print(f"Augmentation: {USE_AUGMENTATION} (edge_drop={EDGE_DROP_RATE}, feat_mask={FEATURE_MASK_RATE})")

    t_start = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Pretraining ----
    print("\nLoading pretraining data...")
    pretrain_data = load_pretrain_data()
    n_train = int(0.8 * len(pretrain_data))
    indices = torch.randperm(len(pretrain_data)).tolist()
    train_data = [pretrain_data[i] for i in indices[:n_train]]
    val_data = [pretrain_data[i] for i in indices[n_train:]]
    print(f"Pretrain: {len(train_data)} train, {len(val_data)} val window pairs")

    n_features = pretrain_data[0].x.shape[1]
    pretrain_model = GINPretrainModel(
        n_features, HIDDEN_DIM, NUM_LAYERS, MLP_HIDDEN_RATIO, DROPOUT, TRAIN_EPS
    ).to(DEVICE)

    n_params = sum(p.numel() for p in pretrain_model.parameters())
    print(f"Pretrain model parameters: {n_params:,}")

    pretrain_results = pretrain(pretrain_model, train_data, val_data)

    # Save encoder
    torch.save(pretrain_results["encoder_state"], MODELS_DIR / "gin_encoder.pt")

    # ---- Stage 2: Fine-tuning ----
    print("\nLoading labeled graphs...")
    train_graphs, val_graphs, test_graphs = load_labeled_graphs()
    print(f"Finetune: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    graph_n_features = train_graphs[0].x.shape[1]
    classifier = GINClassifier(
        graph_n_features, HIDDEN_DIM, NUM_LAYERS, MLP_HIDDEN_RATIO,
        DROPOUT, CLASSIFIER_DROPOUT, TRAIN_EPS,
        pretrain_in_channels=n_features
    ).to(DEVICE)

    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier parameters: {n_params:,}")

    finetune_results = finetune(
        classifier, pretrain_results["encoder_state"],
        train_graphs, val_graphs, test_graphs
    )

    # Save classifier
    torch.save(finetune_results["model_state"], MODELS_DIR / "gin_classifier.pt")

    # Save config
    config = {
        "architecture": "GIN",
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "mlp_hidden_ratio": MLP_HIDDEN_RATIO,
        "train_eps": TRAIN_EPS,
        "dropout": DROPOUT,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "use_augmentation": USE_AUGMENTATION,
        "best_val_auc": finetune_results["best_val_auc"],
        "test_auc": finetune_results["test_auc"],
        "test_f1": finetune_results["test_f1"],
    }
    with open(MODELS_DIR / "gin_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Plot
    plot_training(pretrain_results, finetune_results)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'=' * 70}")
    print("GIN Training Complete!")
    print(f"{'=' * 70}")
    print(f"Test AUC: {finetune_results['test_auc']:.4f}")
    print(f"Test F1:  {finetune_results['test_f1']:.4f}")
    print(f"Runtime:  {elapsed:.1f}s")

    # Log
    backend = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "CPU"
    append_project_log(
        stage="train_gin",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Architecture: GIN ({NUM_LAYERS} layers, MLP ratio {MLP_HIDDEN_RATIO}x)",
            f"Hidden dim: {HIDDEN_DIM}",
            f"Train epsilon: {TRAIN_EPS}",
            f"Augmentation: {USE_AUGMENTATION}",
            f"Best val AUC: {finetune_results['best_val_auc']:.4f}",
            f"Test AUC: {finetune_results['test_auc']:.4f}",
            f"Test F1: {finetune_results['test_f1']:.4f}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
