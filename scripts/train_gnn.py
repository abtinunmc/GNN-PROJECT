"""
GNN Training for SOZ Localization (Fine-tuning)
================================================

Fine-tunes the pre-trained GNN encoder on labeled graphs for binary
node-level classification: SOZ vs non-SOZ channels.

This script:
1. Loads the pre-trained encoder from pretrain_gnn.py
2. Adds a classification head
3. Fine-tunes on graphs with SOZ labels (29 graphs)
4. Evaluates on held-out test set

The pre-trained encoder has learned general iEEG representations from ALL
recordings; now we specialize it for SOZ detection using the limited labels.

Input
-----
    data/processed/graphs/graphs.pt              (from build_graph.py)
    data/processed/models/pretrained_encoder.pt  (from pretrain_gnn.py)
    data/processed/models/pretrain_config.json

Output
------
    data/processed/models/soz_classifier.pt      (fine-tuned model)
    data/processed/models/train_config.json      (hyperparameters + metrics)
    data/processed/figures/train_loss.png        (training curves)
    data/processed/figures/confusion_matrix.png  (test set results)
    data/processed/figures/roc_curve.png         (ROC curve)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
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

GRAPHS_PT = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs.pt"
PRETRAINED_ENCODER = PROJECT_DIR / "data" / "processed" / "models" / "pretrained_encoder.pt"
PRETRAIN_CONFIG = PROJECT_DIR / "data" / "processed" / "models" / "pretrain_config.json"
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"
FIGURES_DIR = PROJECT_DIR / "data" / "processed" / "figures"

OUTPUT_MODEL = MODELS_DIR / "soz_classifier.pt"
OUTPUT_CONFIG = MODELS_DIR / "train_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Training hyperparameters (validated from eeg-gnn-ssl / NeuroGNN repos)
# ---------------------------------------------------------------------------
EPOCHS = 100          # 60-100 epochs for fine-tuning (eeg-gnn-ssl)
BATCH_SIZE = 8        # Smaller batch for limited labeled data
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning (eeg-gnn-ssl, NeuroGNN)
WEIGHT_DECAY = 1e-4   # Standard L2 regularization
PATIENCE = 20         # Early stopping patience
FREEZE_ENCODER_EPOCHS = 10  # Freeze encoder for first N epochs

# ---------------------------------------------------------------------------
# Class imbalance handling
# ---------------------------------------------------------------------------
USE_CLASS_WEIGHTS = True  # Weight loss by inverse class frequency

# ---------------------------------------------------------------------------
# Classifier dropout (from eeg-gnn-ssl: 0.5 for classification tasks)
# ---------------------------------------------------------------------------
CLASSIFIER_DROPOUT = 0.5


# ============================================================================
# MODEL
# ============================================================================

class GNNEncoder(nn.Module):
    """GraphSAGE encoder - must match pretrain_gnn.py architecture."""

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

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))

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


class SOZClassifier(nn.Module):
    """Full model: pre-trained encoder + classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        classifier_dropout: float = 0.5,  # Higher dropout for classifier (eeg-gnn-ssl)
        pretrain_in_channels: int = None,  # Input dim the encoder was pretrained on
    ):
        super().__init__()
        # Input projection: maps aggregated features (33) to pretrain dim (11)
        # This allows using pretrained encoder weights trained on different input dim
        self.input_proj = None
        encoder_in = in_channels
        if pretrain_in_channels is not None and pretrain_in_channels != in_channels:
            self.input_proj = nn.Linear(in_channels, pretrain_in_channels)
            encoder_in = pretrain_in_channels

        self.encoder = GNNEncoder(
            in_channels=encoder_in,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Classification head for binary node classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),  # 0.5 for classification (literature)
            nn.Linear(hidden_channels // 2, 1),  # Binary: SOZ or not
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)
        h = self.encoder(x, edge_index)
        return self.classifier(h).squeeze(-1)  # (n_nodes,)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# ============================================================================
# DATA LOADING
# ============================================================================

def load_labeled_graphs() -> tuple[list, list, list]:
    """Load graphs and split into train/val/test (only labeled graphs)."""
    print("Loading graphs...")

    if not GRAPHS_PT.exists():
        raise FileNotFoundError(f"Graphs not found: {GRAPHS_PT}")

    all_graphs = torch.load(GRAPHS_PT, weights_only=False)
    print(f"Total graphs: {len(all_graphs)}")

    # Filter to only graphs with SOZ labels
    labeled_graphs = [g for g in all_graphs if g.n_soz > 0]
    print(f"Graphs with SOZ labels: {len(labeled_graphs)}")

    # Split by pre-assigned split attribute (subject-level split from build_graph.py)
    train_graphs = [g for g in labeled_graphs if g.split == "train"]
    val_graphs = [g for g in labeled_graphs if g.split == "val"]
    test_graphs = [g for g in labeled_graphs if g.split == "test"]

    print(f"Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    # Print class balance
    for name, graphs in [("Train", train_graphs), ("Val", val_graphs), ("Test", test_graphs)]:
        if graphs:
            n_soz = sum(g.y.sum().item() for g in graphs)
            n_total = sum(g.y.shape[0] for g in graphs)
            print(f"  {name}: {n_soz}/{n_total} SOZ nodes ({100*n_soz/n_total:.1f}%)")

    return train_graphs, val_graphs, test_graphs


def compute_class_weights(graphs: list) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    n_soz = sum(g.y.sum().item() for g in graphs)
    n_total = sum(g.y.shape[0] for g in graphs)
    n_non_soz = n_total - n_soz

    # Weight inversely proportional to frequency
    w_non_soz = n_total / (2 * n_non_soz)
    w_soz = n_total / (2 * n_soz)

    return torch.tensor([w_non_soz, w_soz], dtype=torch.float32)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: SOZClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor | None,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)
        labels = batch.y.float()

        # Binary cross-entropy with optional class weights
        if class_weights is not None:
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(
    model: SOZClassifier,
    loader: DataLoader,
    class_weights: torch.Tensor | None,
) -> tuple[float, float, list, list]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index)
        labels = batch.y.float()

        if class_weights is not None:
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc, all_preds, all_labels


def train() -> dict:
    """Main training loop."""
    print("=" * 70)
    print("SOZ Classification - Fine-tuning Pre-trained GNN")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("-" * 70)

    # Load data
    train_graphs, val_graphs, test_graphs = load_labeled_graphs()

    if len(train_graphs) == 0:
        raise ValueError("No training graphs with SOZ labels!")

    # Load pretrain config
    if not PRETRAIN_CONFIG.exists():
        raise FileNotFoundError(f"Pretrain config not found: {PRETRAIN_CONFIG}")

    with open(PRETRAIN_CONFIG) as f:
        pretrain_cfg = json.load(f)

    # Get actual n_features from graph data (33 aggregated features)
    n_features = train_graphs[0].x.shape[1]
    # Pretrain used window-level features (11)
    pretrain_n_features = pretrain_cfg["n_features"]
    hidden_dim = pretrain_cfg["hidden_dim"]
    num_layers = pretrain_cfg["num_layers"]
    dropout = pretrain_cfg["dropout"]

    print(f"Loaded pretrain config: hidden={hidden_dim}, layers={num_layers}")
    print(f"Graph features: {n_features}, Pretrain features: {pretrain_n_features}")

    # Create dataloaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    # pretrain_in_channels: the encoder was pretrained on 11 window-level features,
    # but graphs.pt has 33 aggregated features. Add input projection to bridge this.
    model = SOZClassifier(
        in_channels=n_features,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        classifier_dropout=CLASSIFIER_DROPOUT,
        pretrain_in_channels=pretrain_n_features,
    ).to(DEVICE)

    # Load pre-trained encoder weights
    if PRETRAINED_ENCODER.exists():
        print(f"Loading pre-trained encoder from {PRETRAINED_ENCODER}")
        state_dict = torch.load(PRETRAINED_ENCODER, map_location=DEVICE, weights_only=True)
        model.encoder.load_state_dict(state_dict)
        print("Pre-trained weights loaded successfully!")
        use_pretrained = True
    else:
        print("WARNING: No pre-trained encoder found. Training from scratch.")
        use_pretrained = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Class weights for imbalanced data
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_graphs)
        print(f"Class weights: non-SOZ={class_weights[0]:.2f}, SOZ={class_weights[1]:.2f}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print("-" * 70)
    best_val_auc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # Freeze encoder for first N epochs (warm up classifier head)
        if use_pretrained and epoch == 1:
            model.freeze_encoder()
            print(f"Encoder frozen for first {FREEZE_ENCODER_EPOCHS} epochs")
        elif use_pretrained and epoch == FREEZE_ENCODER_EPOCHS + 1:
            model.unfreeze_encoder()
            print("Encoder unfrozen - full fine-tuning")

        train_loss, train_auc = train_epoch(model, train_loader, optimizer, class_weights)
        val_loss, val_auc, _, _ = eval_epoch(model, val_loader, class_weights)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        # Early stopping on val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} AUC: {train_auc:.3f} | "
                f"Val Loss: {val_loss:.4f} AUC: {val_auc:.3f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    print("-" * 70)
    print("Evaluating on test set...")
    model.load_state_dict(best_model_state)

    test_loss, test_auc, test_preds, test_labels = eval_epoch(
        model, test_loader, class_weights
    )

    # Compute metrics at threshold 0.5
    test_preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
    accuracy = accuracy_score(test_labels, test_preds_binary)
    precision = precision_score(test_labels, test_preds_binary, zero_division=0)
    recall = recall_score(test_labels, test_preds_binary, zero_division=0)
    f1 = f1_score(test_labels, test_preds_binary, zero_division=0)

    print(f"\nTest Results:")
    print(f"  AUC:       {test_auc:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    return {
        "model_state": best_model_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_aucs": train_aucs,
        "val_aucs": val_aucs,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "epochs_trained": len(train_losses),
        "use_pretrained": use_pretrained,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(results: dict) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(results["train_losses"]) + 1)

    # Loss
    axes[0].plot(epochs, results["train_losses"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, results["val_losses"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(epochs, results["train_aucs"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, results["val_aucs"], "r-", label="Val", linewidth=2)
    axes[1].axhline(results["test_auc"], color="green", linestyle="--",
                    label=f"Test AUC: {results['test_auc']:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("ROC-AUC Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "train_loss.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


def plot_roc_curve(results: dict) -> None:
    """Plot ROC curve for test set."""
    fpr, tpr, _ = roc_curve(results["test_labels"], results["test_preds"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {results['test_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - SOZ Classification (Test Set)", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


def plot_confusion_matrix(results: dict) -> None:
    """Plot confusion matrix for test set."""
    preds_binary = [1 if p > 0.5 else 0 for p in results["test_preds"]]
    cm = confusion_matrix(results["test_labels"], preds_binary)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-SOZ", "SOZ"])
    ax.set_yticklabels(["Non-SOZ", "SOZ"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Test Set", fontsize=14)

    # Annotate
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "confusion_matrix.png"
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

    # Train
    results = train()

    # Save model
    torch.save(results["model_state"], OUTPUT_MODEL)
    print(f"\nSaved model: {OUTPUT_MODEL}")

    # Save config
    config = {
        "use_pretrained": results["use_pretrained"],
        "epochs_trained": results["epochs_trained"],
        "best_val_auc": results["best_val_auc"],
        "test_auc": results["test_auc"],
        "test_accuracy": results["test_accuracy"],
        "test_precision": results["test_precision"],
        "test_recall": results["test_recall"],
        "test_f1": results["test_f1"],
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "freeze_encoder_epochs": FREEZE_ENCODER_EPOCHS,
        "use_class_weights": USE_CLASS_WEIGHTS,
    }
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {OUTPUT_CONFIG}")

    # Plots
    plot_training_curves(results)
    plot_roc_curve(results)
    plot_confusion_matrix(results)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("Training complete!")
    print(f"Test AUC: {results['test_auc']:.3f}")
    print(f"Test F1:  {results['test_f1']:.3f}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Log
    backend = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "CPU"
    append_project_log(
        stage="train_gnn",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Pretrained encoder: {results['use_pretrained']}",
            f"Input: {GRAPHS_PT}",
            f"Output: {OUTPUT_MODEL}",
            f"Epochs trained: {results['epochs_trained']}",
            f"Best val AUC: {results['best_val_auc']:.3f}",
            f"Test AUC: {results['test_auc']:.3f}",
            f"Test F1: {results['test_f1']:.3f}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
