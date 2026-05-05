"""
Hyperparameter Tuning for GAT (Graph Attention Network)
========================================================

Uses Optuna to tune GAT-specific hyperparameters including:
    - Number of attention heads
    - Attention dropout
    - Hidden dimensions
    - Learning rates
    - Augmentation settings

Output
------
    data/processed/models/best_gat_hyperparams.json
    data/processed/models/tuned_gat_encoder.pt
    data/processed/models/tuned_gat_classifier.pt
    data/processed/figures/optuna_gat.png
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import optuna
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.utils import dropout_edge

from log_utils import append_project_log

warnings.filterwarnings("ignore")

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

# Tuning settings
N_TRIALS = 60
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 80
PATIENCE = 15


# ============================================================================
# GAT MODEL
# ============================================================================

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, num_heads, dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(
            in_channels, hidden_channels // num_heads, heads=num_heads,
            dropout=attention_dropout, concat=True
        ))
        self.norms.append(BatchNorm(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels, hidden_channels // num_heads, heads=num_heads,
                dropout=attention_dropout, concat=True
            ))
            self.norms.append(BatchNorm(hidden_channels))

        # Last layer
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels, out_channels, heads=num_heads,
                dropout=attention_dropout, concat=False
            ))
            self.norms.append(BatchNorm(out_channels))

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATPretrainModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, dropout, attention_dropout):
        super().__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, hidden_channels,
                                   num_layers, num_heads, dropout, attention_dropout)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.predictor(h)


class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads,
                 dropout, attention_dropout, classifier_dropout, pretrain_in_channels=None):
        super().__init__()

        self.input_proj = None
        encoder_in = in_channels
        if pretrain_in_channels is not None and pretrain_in_channels != in_channels:
            self.input_proj = nn.Linear(in_channels, pretrain_in_channels)
            encoder_in = pretrain_in_channels

        self.encoder = GATEncoder(encoder_in, hidden_channels, hidden_channels,
                                   num_layers, num_heads, dropout, attention_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
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


def load_pretrain_data(threshold):
    pairs = []
    with h5py.File(FEATURES_H5, "r") as f:
        for rec_name in f.keys():
            grp = f[rec_name]
            window_feats = grp["window_node_features"][:]
            n_windows = window_feats.shape[0]
            if n_windows < 2:
                continue
            corr_matrix = grp["edge_features/correlation"][:, :, 0]
            edge_index = build_edge_index(corr_matrix, threshold)
            for t in range(n_windows - 1):
                x_in = zscore(window_feats[t])
                x_out = zscore(window_feats[t + 1])
                pairs.append(Data(x=torch.from_numpy(x_in), edge_index=edge_index,
                                  y=torch.from_numpy(x_out)))
    return pairs


def load_labeled_graphs():
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
# TRAINING FUNCTIONS
# ============================================================================

def pretrain_model(model, train_loader, val_loader, lr, weight_decay, epochs, patience,
                   use_augmentation, edge_drop_rate):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            edge_index = batch.edge_index
            x = batch.x

            if use_augmentation:
                edge_index, _ = dropout_edge(edge_index, p=edge_drop_rate, training=True)

            optimizer.zero_grad()
            pred = model(x, edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch.x, batch.edge_index)
                val_loss += F.mse_loss(pred, batch.y).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.encoder.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_loss


def finetune_model(model, encoder_state, train_loader, val_loader, class_weights,
                   lr, weight_decay, epochs, patience, freeze_epochs,
                   use_augmentation, edge_drop_rate):
    model.encoder.load_state_dict(encoder_state)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        if epoch == 0 and freeze_epochs > 0:
            model.freeze_encoder()
        elif epoch == freeze_epochs:
            model.unfreeze_encoder()

        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            edge_index = batch.edge_index
            x = batch.x

            if use_augmentation:
                edge_index, _ = dropout_edge(edge_index, p=edge_drop_rate * 0.5, training=True)

            optimizer.zero_grad()
            logits = model(x, edge_index)
            labels = batch.y.float()
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_auc


def evaluate_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
    return roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: Trial) -> float:
    # GAT-specific hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.3)
    classifier_dropout = trial.suggest_float("classifier_dropout", 0.3, 0.7)

    # Training hyperparameters
    pretrain_lr = trial.suggest_float("pretrain_lr", 5e-5, 5e-3, log=True)
    finetune_lr = trial.suggest_float("finetune_lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    freeze_epochs = trial.suggest_categorical("freeze_epochs", [0, 5, 10, 15])

    # Augmentation
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    edge_drop_rate = trial.suggest_float("edge_drop_rate", 0.0, 0.15) if use_augmentation else 0.0

    # Graph construction
    corr_threshold = trial.suggest_float("correlation_threshold", 0.2, 0.5)

    try:
        # Load pretrain data
        pretrain_data = load_pretrain_data(corr_threshold)
        n_train = int(0.8 * len(pretrain_data))
        indices = torch.randperm(len(pretrain_data)).tolist()
        pretrain_train = [pretrain_data[i] for i in indices[:n_train]]
        pretrain_val = [pretrain_data[i] for i in indices[n_train:]]

        pretrain_train_loader = DataLoader(pretrain_train, batch_size=batch_size, shuffle=True)
        pretrain_val_loader = DataLoader(pretrain_val, batch_size=batch_size)

        # Pretrain
        n_features = pretrain_data[0].x.shape[1]
        pretrain_model_inst = GATPretrainModel(
            n_features, hidden_dim, num_layers, num_heads, dropout, attention_dropout
        ).to(DEVICE)

        encoder_state, _ = pretrain_model(
            pretrain_model_inst, pretrain_train_loader, pretrain_val_loader,
            pretrain_lr, weight_decay, PRETRAIN_EPOCHS, PATIENCE,
            use_augmentation, edge_drop_rate
        )

        # Load fine-tune data
        train_graphs, val_graphs, _ = load_labeled_graphs()
        train_loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)), shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=min(batch_size, len(val_graphs)))

        graph_n_features = train_graphs[0].x.shape[1]
        class_weights = compute_class_weights(train_graphs)

        classifier = GATClassifier(
            graph_n_features, hidden_dim, num_layers, num_heads,
            dropout, attention_dropout, classifier_dropout,
            pretrain_in_channels=n_features
        ).to(DEVICE)

        _, val_auc = finetune_model(
            classifier, encoder_state, train_loader, val_loader, class_weights,
            finetune_lr, weight_decay, FINETUNE_EPOCHS, PATIENCE, freeze_epochs,
            use_augmentation, edge_drop_rate
        )

        trial.report(val_auc, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_auc

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Hyperparameter Tuning for GAT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trials: {N_TRIALS}")
    print(f"Pretrain epochs/trial: {PRETRAIN_EPOCHS}")
    print(f"Finetune epochs/trial: {FINETUNE_EPOCHS}")
    print("-" * 70)

    t_start = time.perf_counter()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_path = MODELS_DIR / "best_gat_hyperparams.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved: {best_params_path}")

    # Retrain with best hyperparameters
    print("\n" + "-" * 70)
    print("Retraining with best hyperparameters...")

    params = study.best_params

    # Pretrain
    pretrain_data = load_pretrain_data(params["correlation_threshold"])
    n_train = int(0.8 * len(pretrain_data))
    indices = torch.randperm(len(pretrain_data)).tolist()
    pretrain_train = [pretrain_data[i] for i in indices[:n_train]]
    pretrain_val = [pretrain_data[i] for i in indices[n_train:]]

    pretrain_train_loader = DataLoader(pretrain_train, batch_size=params["batch_size"], shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val, batch_size=params["batch_size"])

    n_features = pretrain_data[0].x.shape[1]
    pretrain_model_inst = GATPretrainModel(
        n_features, params["hidden_dim"], params["num_layers"],
        params["num_heads"], params["dropout"], params["attention_dropout"]
    ).to(DEVICE)

    edge_drop = params.get("edge_drop_rate", 0.0)
    encoder_state, _ = pretrain_model(
        pretrain_model_inst, pretrain_train_loader, pretrain_val_loader,
        params["pretrain_lr"], params["weight_decay"], PRETRAIN_EPOCHS * 2, PATIENCE,
        params["use_augmentation"], edge_drop
    )

    torch.save(encoder_state, MODELS_DIR / "tuned_gat_encoder.pt")

    # Fine-tune
    train_graphs, val_graphs, test_graphs = load_labeled_graphs()
    train_loader = DataLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=params["batch_size"])
    test_loader = DataLoader(test_graphs, batch_size=params["batch_size"])

    graph_n_features = train_graphs[0].x.shape[1]
    class_weights = compute_class_weights(train_graphs)

    classifier = GATClassifier(
        graph_n_features, params["hidden_dim"], params["num_layers"],
        params["num_heads"], params["dropout"], params["attention_dropout"],
        params["classifier_dropout"], pretrain_in_channels=n_features
    ).to(DEVICE)

    best_state, val_auc = finetune_model(
        classifier, encoder_state, train_loader, val_loader, class_weights,
        params["finetune_lr"], params["weight_decay"], FINETUNE_EPOCHS * 2, PATIENCE,
        params["freeze_epochs"], params["use_augmentation"], edge_drop
    )

    classifier.load_state_dict(best_state)
    test_auc = evaluate_test(classifier, test_loader)

    torch.save(best_state, MODELS_DIR / "tuned_gat_classifier.pt")

    print(f"\nFinal Results (with best hyperparameters):")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC:       {test_auc:.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GAT Hyperparameter Tuning", fontsize=14, fontweight="bold")

        # Optimization history
        trials = [t for t in study.trials if t.value is not None and t.value > 0]
        values = [t.value for t in trials]
        axes[0].plot(range(len(values)), values, "b.-", alpha=0.7)
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Validation AUC")
        axes[0].set_title("Optimization History")
        axes[0].axhline(max(values), color="r", linestyle="--", label=f"Best: {max(values):.4f}")
        axes[0].axhline(test_auc, color="g", linestyle="--", label=f"Test: {test_auc:.4f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(study)
            names = list(importances.keys())[:10]
            vals = [importances[n] for n in names]
            axes[1].barh(names, vals, color="steelblue")
            axes[1].set_xlabel("Importance")
            axes[1].set_title("Hyperparameter Importance")
            axes[1].grid(True, alpha=0.3, axis="x")
        except:
            axes[1].text(0.5, 0.5, "Could not compute\nimportances", ha="center", va="center")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "optuna_gat.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {fig_path}")
        plt.close()
    except Exception as e:
        print(f"Could not plot: {e}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    # Log
    backend = f"GPU ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "CPU"
    append_project_log(
        stage="tune_gat",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Trials: {N_TRIALS}",
            f"Best val AUC: {study.best_value:.4f}",
            f"Test AUC (tuned): {test_auc:.4f}",
            f"Best hidden_dim: {params['hidden_dim']}",
            f"Best num_heads: {params['num_heads']}",
            f"Best num_layers: {params['num_layers']}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
