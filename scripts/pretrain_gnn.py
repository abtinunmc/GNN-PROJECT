"""
Self-Supervised GNN Pre-training for SOZ Localization
======================================================

Pre-trains a GNN encoder on ALL recordings (labeled + unlabeled) using a
signal forecasting objective: predict next-window node features from the
current window.

This approach follows recent literature (NeuroGNN, Self-Supervised GNN for EEG)
which shows that self-supervised pre-training substantially improves downstream
performance, especially with limited labeled data.

Pre-training Task
-----------------
For each recording with multiple windows:
    Input:  node features at window t      (n_channels, n_features)
    Target: node features at window t+1    (n_channels, n_features)

The GNN encoder learns to capture temporal dynamics and spatial relationships
in iEEG signals without requiring SOZ labels.

Input
-----
    data/processed/h5/features.h5   (window-level features from Stage 4)

Output
------
    data/processed/models/pretrained_encoder.pt   (encoder state_dict)
    data/processed/models/pretrain_config.json    (hyperparameters)
    data/processed/figures/pretrain_loss.png      (training curves)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SAGEConv, BatchNorm
except ImportError as exc:
    raise ImportError(
        "PyTorch and PyTorch Geometric required.\n"
        "Install: pip install torch torch_geometric"
    ) from exc

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_DIR / "data" / "processed" / "figures"

OUTPUT_ENCODER = MODELS_DIR / "pretrained_encoder.pt"
OUTPUT_CONFIG = MODELS_DIR / "pretrain_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model hyperparameters (validated from eeg-gnn-ssl / NeuroGNN repos)
# ---------------------------------------------------------------------------
HIDDEN_DIM = 64       # Smaller dim prevents overfitting on limited data
NUM_LAYERS = 3        # 3 layers for pretraining (eeg-gnn-ssl)
DROPOUT = 0.3         # Moderate dropout for pretraining

# ---------------------------------------------------------------------------
# Training hyperparameters (from eeg-gnn-ssl self-supervised pretraining)
# ---------------------------------------------------------------------------
EPOCHS = 350          # More epochs for self-supervised learning
BATCH_SIZE = 32       # Standard batch size
LEARNING_RATE = 5e-4  # Higher LR for pretraining (eeg-gnn-ssl, NeuroGNN)
WEIGHT_DECAY = 1e-4   # Standard L2 regularization
PATIENCE = 25         # More patience for longer training

# ---------------------------------------------------------------------------
# Graph construction (same threshold as build_graph.py)
# ---------------------------------------------------------------------------
CORRELATION_THRESHOLD = 0.3


# ============================================================================
# GNN ENCODER
# ============================================================================

class GNNEncoder(nn.Module):
    """GraphSAGE encoder for node-level representation learning."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.norms.append(BatchNorm(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PretrainModel(nn.Module):
    """Full pre-training model: encoder + prediction head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Prediction head: maps encoder output to next-window features
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, edge_index)
        return self.predictor(h)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)


# ============================================================================
# DATA LOADING
# ============================================================================

def build_edge_index_from_correlation(
    corr_matrix: np.ndarray,
    threshold: float,
) -> torch.Tensor:
    """Build edge index from correlation matrix using threshold."""
    n = corr_matrix.shape[0]
    corr_abs = np.abs(corr_matrix)

    # Upper triangle mask above threshold
    mask = np.triu(corr_abs >= threshold, k=1)
    src, dst = np.where(mask)

    if len(src) == 0:
        # Fallback: self-loops
        idx = np.arange(n)
        return torch.tensor(np.stack([idx, idx]), dtype=torch.long)

    # Undirected: both directions
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ])
    return torch.tensor(edge_index, dtype=torch.long)


def zscore_features(features: np.ndarray) -> np.ndarray:
    """Z-score normalize features."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((features - mean) / std).astype(np.float32)


def load_pretrain_data() -> list[Data]:
    """Load window pairs for self-supervised pre-training.

    Creates (input, target) pairs where:
        input  = node features at window t
        target = node features at window t+1
    """
    print("Loading pre-training data...")

    if not FEATURES_H5.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_H5}")

    pairs = []
    skipped = 0

    with h5py.File(FEATURES_H5, "r") as f:
        rec_names = list(f.keys())
        print(f"Found {len(rec_names)} recordings")

        for rec_name in tqdm(rec_names, desc="Loading windows"):
            grp = f[rec_name]

            # Window-level node features: (n_windows, n_channels, n_features)
            window_feats = grp["window_node_features"][:]
            n_windows = window_feats.shape[0]

            if n_windows < 2:
                skipped += 1
                continue

            # Get correlation matrix for edge construction (mean aggregation)
            corr_matrix = grp["edge_features/correlation"][:, :, 0]
            edge_index = build_edge_index_from_correlation(
                corr_matrix, CORRELATION_THRESHOLD
            )

            # Create consecutive window pairs
            for t in range(n_windows - 1):
                x_in = zscore_features(window_feats[t])    # (n_channels, n_features)
                x_out = zscore_features(window_feats[t + 1])

                data = Data(
                    x=torch.from_numpy(x_in),
                    edge_index=edge_index,
                    y=torch.from_numpy(x_out),  # Target is next window
                )
                pairs.append(data)

    print(f"Created {len(pairs)} window pairs from {len(rec_names) - skipped} recordings")
    print(f"Skipped {skipped} recordings with < 2 windows")

    return pairs


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: PretrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        pred = model(batch.x, batch.edge_index)

        # MSE loss between predicted and actual next-window features
        loss = F.mse_loss(pred, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples


@torch.no_grad()
def eval_epoch(model: PretrainModel, loader: DataLoader) -> float:
    """Evaluate model, return mean loss."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index)
        loss = F.mse_loss(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs
        n_samples += batch.num_graphs

    return total_loss / n_samples


def pretrain() -> dict:
    """Main pre-training loop."""
    print("=" * 70)
    print("Self-Supervised GNN Pre-training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Task: Signal forecasting (predict next window features)")
    print("-" * 70)

    # Load data
    all_data = load_pretrain_data()

    if len(all_data) == 0:
        raise ValueError("No training pairs found!")

    # Get feature dimension from first sample
    n_features = all_data[0].x.shape[1]
    print(f"Node features: {n_features}")

    # Train/val split (80/20)
    n_train = int(0.8 * len(all_data))
    indices = torch.randperm(len(all_data)).tolist()
    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:]]

    print(f"Train pairs: {len(train_data)}")
    print(f"Val pairs: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = PretrainModel(
        in_channels=n_features,
        hidden_channels=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop with early stopping
    print("-" * 70)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch(model, val_loader)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best encoder
            best_encoder_state = model.encoder.state_dict()
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best: {best_val_loss:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        "encoder_state": best_encoder_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "n_features": n_features,
        "epochs_trained": len(train_losses),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(train_losses: list, val_losses: list) -> None:
    """Plot and save training curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Self-Supervised Pre-training: Signal Forecasting", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7)
    ax.scatter([best_epoch], [best_val], color="green", s=100, zorder=5)
    ax.annotate(
        f"Best: {best_val:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 5, best_val + 0.01),
        fontsize=10,
    )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "pretrain_loss.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    t_start = time.perf_counter()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-train
    results = pretrain()

    # Save encoder
    torch.save(results["encoder_state"], OUTPUT_ENCODER)
    print(f"\nSaved encoder: {OUTPUT_ENCODER}")

    # Save config
    config = {
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "n_features": results["n_features"],
        "epochs_trained": results["epochs_trained"],
        "best_val_loss": results["best_val_loss"],
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "correlation_threshold": CORRELATION_THRESHOLD,
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {OUTPUT_CONFIG}")

    # Plot curves
    plot_training_curves(results["train_losses"], results["val_losses"])

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("Pre-training complete!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Log
    backend = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "CPU"
    append_project_log(
        stage="pretrain_gnn",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Task: Signal forecasting (predict next window)",
            f"Input: {FEATURES_H5}",
            f"Output: {OUTPUT_ENCODER}",
            f"Hidden dim: {HIDDEN_DIM}",
            f"Num layers: {NUM_LAYERS}",
            f"Epochs trained: {results['epochs_trained']}",
            f"Best val loss: {results['best_val_loss']:.4f}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
