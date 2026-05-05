"""
Hyperparameter Tuning for SOZ Localization GNN
===============================================

Uses Optuna for Bayesian optimization of GNN hyperparameters.
Tunes both pretraining and fine-tuning stages jointly.

Tuned Parameters
----------------
Model:
    - hidden_dim: [32, 64, 128]
    - num_layers: [2, 3, 4]
    - dropout: [0.1, 0.3, 0.5]
    - classifier_dropout: [0.3, 0.5, 0.7]

Training:
    - pretrain_lr: [1e-4, 1e-2]
    - finetune_lr: [1e-5, 1e-3]
    - weight_decay: [1e-5, 1e-3]
    - batch_size: [8, 16, 32]
    - freeze_epochs: [0, 5, 10, 15]

Graph:
    - correlation_threshold: [0.2, 0.5]

Data Augmentation:
    - aug_edge_drop: [0.0, 0.25]   - probability of dropping edges
    - aug_feat_noise: [0.0, 0.1]  - std of Gaussian noise on features
    - aug_feat_mask: [0.0, 0.2]   - probability of masking features
    - mixup_alpha: [0.0, 0.5]     - Beta distribution param for mixup

Objective: Maximize validation AUC

Output
------
    data/processed/models/best_hyperparams.json
    data/processed/models/tuned_encoder.pt
    data/processed/models/tuned_classifier.pt
    data/processed/figures/optuna_optimization.png
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
N_TRIALS = 50
PRETRAIN_EPOCHS = 100  # Reduced for faster tuning
FINETUNE_EPOCHS = 60
PATIENCE = 10

# Data augmentation settings
USE_AUGMENTATION = True
EDGE_DROP_RATE = 0.15      # Drop 15% of edges randomly
FEATURE_NOISE_STD = 0.05   # Add Gaussian noise with std=0.05
FEATURE_MASK_RATE = 0.10   # Mask 10% of features

# Mixup settings
USE_MIXUP = False           # Disabled - didn't help
MIXUP_ALPHA = 0.2          # Beta distribution parameter for mixup

# SMOTE settings
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 5      # Number of neighbors for SMOTE
SMOTE_RATIO = 1.0          # Target ratio of minority to majority class


# ============================================================================
# MODEL DEFINITIONS
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


def load_pretrain_data(threshold):
    """Load window pairs for pretraining with given correlation threshold."""
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
    """Load labeled graphs for fine-tuning."""
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
# DATA AUGMENTATION
# ============================================================================

def augment_batch(x, edge_index, edge_drop=EDGE_DROP_RATE,
                  feat_noise=FEATURE_NOISE_STD, feat_mask=FEATURE_MASK_RATE):
    """Apply data augmentation to a batch during training.

    Args:
        x: Node features (N, F)
        edge_index: Edge indices (2, E)
        edge_drop: Probability of dropping each edge
        feat_noise: Std of Gaussian noise to add to features
        feat_mask: Probability of masking each feature value

    Returns:
        Augmented x and edge_index
    """
    # Edge dropout - randomly remove edges
    if edge_drop > 0:
        edge_index, _ = dropout_edge(edge_index, p=edge_drop, training=True)

    # Feature noise - add Gaussian noise
    if feat_noise > 0:
        noise = torch.randn_like(x) * feat_noise
        x = x + noise

    # Feature masking - zero out random features
    if feat_mask > 0:
        mask = torch.rand_like(x) > feat_mask
        x = x * mask.float()

    return x, edge_index


def mixup_batch(x, y, alpha=MIXUP_ALPHA):
    """Apply mixup augmentation to node features and labels.

    Mixup creates virtual training examples by interpolating between
    pairs of samples: x_new = λ*x + (1-λ)*x_shuffled

    Args:
        x: Node features (N, F)
        y: Labels (N,)
        alpha: Beta distribution parameter (lower = less mixing)

    Returns:
        Mixed x, mixed y (as float for soft labels), lambda value
    """
    if alpha <= 0:
        return x, y.float(), 1.0

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Shuffle indices
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Mix features
    mixed_x = lam * x + (1 - lam) * x[index]

    # Mix labels (convert to float for soft labels)
    y_float = y.float()
    mixed_y = lam * y_float + (1 - lam) * y_float[index]

    return mixed_x, mixed_y, lam


def smote_oversample(x, y, k_neighbors=SMOTE_K_NEIGHBORS, target_ratio=SMOTE_RATIO):
    """Apply SMOTE to oversample minority class (SOZ nodes).

    SMOTE generates synthetic minority samples by interpolating between
    existing minority samples and their k-nearest neighbors.

    Args:
        x: Node features (N, F) as tensor
        y: Labels (N,) as tensor (1 = SOZ, 0 = non-SOZ)
        k_neighbors: Number of neighbors for interpolation
        target_ratio: Target ratio of minority to majority (1.0 = balanced)

    Returns:
        Augmented x and y tensors with synthetic SOZ samples added
    """
    minority_mask = y == 1
    majority_mask = y == 0
    n_minority = minority_mask.sum().item()
    n_majority = majority_mask.sum().item()

    if n_minority == 0 or n_minority >= n_majority * target_ratio:
        return x, y

    # Number of synthetic samples to generate
    n_synthetic = int(n_majority * target_ratio) - n_minority

    # Get minority class features
    minority_x = x[minority_mask].cpu().numpy()
    n_minority_samples = minority_x.shape[0]

    if n_minority_samples < 2:
        return x, y

    # Compute pairwise distances for k-NN
    from sklearn.neighbors import NearestNeighbors
    k = min(k_neighbors, n_minority_samples - 1)
    if k < 1:
        return x, y

    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because includes self
    nn.fit(minority_x)
    _, indices = nn.kneighbors(minority_x)

    # Generate synthetic samples
    synthetic_samples = []
    for _ in range(n_synthetic):
        # Pick a random minority sample
        idx = np.random.randint(n_minority_samples)
        sample = minority_x[idx]

        # Pick a random neighbor (exclude self at index 0)
        neighbor_idx = indices[idx, np.random.randint(1, k + 1)]
        neighbor = minority_x[neighbor_idx]

        # Interpolate
        alpha = np.random.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)

    synthetic_x = torch.tensor(np.array(synthetic_samples), dtype=x.dtype, device=x.device)
    synthetic_y = torch.ones(n_synthetic, dtype=y.dtype, device=y.device)

    # Concatenate with original
    augmented_x = torch.cat([x, synthetic_x], dim=0)
    augmented_y = torch.cat([y, synthetic_y], dim=0)

    return augmented_x, augmented_y


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def pretrain_model(model, train_loader, val_loader, lr, weight_decay, epochs, patience):
    """Pretrain and return best encoder state."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch.x, batch.edge_index)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
        val_loss /= len(val_loader.dataset)

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
                   aug_edge_drop=EDGE_DROP_RATE, aug_feat_noise=FEATURE_NOISE_STD,
                   aug_feat_mask=FEATURE_MASK_RATE, mixup_alpha=MIXUP_ALPHA,
                   smote_k=SMOTE_K_NEIGHBORS, smote_ratio=SMOTE_RATIO):
    """Fine-tune and return best val AUC."""
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

        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            # Apply augmentation during training
            x, edge_index = batch.x, batch.edge_index
            labels = batch.y

            if USE_AUGMENTATION:
                x, edge_index = augment_batch(x, edge_index,
                                              edge_drop=aug_edge_drop,
                                              feat_noise=aug_feat_noise,
                                              feat_mask=aug_feat_mask)

            # Apply SMOTE oversampling
            if USE_SMOTE and smote_ratio > 0:
                x, labels = smote_oversample(x, labels, k_neighbors=smote_k,
                                             target_ratio=smote_ratio)

            # Apply mixup
            if USE_MIXUP and mixup_alpha > 0:
                x, labels, lam = mixup_batch(x, labels, alpha=mixup_alpha)
            else:
                labels = labels.float()

            # For SMOTE-augmented data, we only predict on original nodes
            # but train with all (original + synthetic) nodes
            if USE_SMOTE and smote_ratio > 0 and x.size(0) > batch.x.size(0):
                # Create dummy edge_index for synthetic nodes (self-loops)
                n_orig = batch.x.size(0)
                n_total = x.size(0)
                n_synth = n_total - n_orig

                # Add self-loop edges for synthetic nodes
                synth_edges = torch.stack([
                    torch.arange(n_orig, n_total, device=DEVICE),
                    torch.arange(n_orig, n_total, device=DEVICE)
                ])
                edge_index = torch.cat([edge_index, synth_edges], dim=1)

            logits = model(x, edge_index)
            weights = class_weights[1] * labels + class_weights[0] * (1 - labels)
            weights = weights.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
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

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_auc


def evaluate_test(model, test_loader, class_weights):
    """Evaluate on test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index)
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())

    if len(set(all_labels)) > 1:
        return roc_auc_score(all_labels, all_preds)
    return 0.5


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: Trial) -> float:
    """Optuna objective: maximize validation AUC."""

    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    classifier_dropout = trial.suggest_float("classifier_dropout", 0.3, 0.7)
    pretrain_lr = trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True)
    finetune_lr = trial.suggest_float("finetune_lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    freeze_epochs = trial.suggest_categorical("freeze_epochs", [0, 5, 10, 15])
    corr_threshold = trial.suggest_float("correlation_threshold", 0.2, 0.5)

    # Augmentation hyperparameters
    aug_edge_drop = trial.suggest_float("aug_edge_drop", 0.0, 0.25)
    aug_feat_noise = trial.suggest_float("aug_feat_noise", 0.0, 0.1)
    aug_feat_mask = trial.suggest_float("aug_feat_mask", 0.0, 0.2)

    # Mixup hyperparameter
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.5)

    # SMOTE hyperparameters
    smote_k = trial.suggest_int("smote_k_neighbors", 2, 10)
    smote_ratio = trial.suggest_float("smote_ratio", 0.3, 1.5)

    try:
        # Load pretrain data with sampled threshold
        pretrain_data = load_pretrain_data(corr_threshold)
        n_train = int(0.8 * len(pretrain_data))
        indices = torch.randperm(len(pretrain_data)).tolist()
        pretrain_train = [pretrain_data[i] for i in indices[:n_train]]
        pretrain_val = [pretrain_data[i] for i in indices[n_train:]]

        pretrain_train_loader = DataLoader(pretrain_train, batch_size=batch_size, shuffle=True)
        pretrain_val_loader = DataLoader(pretrain_val, batch_size=batch_size)

        # Pretrain
        n_features = pretrain_data[0].x.shape[1]  # 11
        pretrain_model_inst = PretrainModel(n_features, hidden_dim, num_layers, dropout).to(DEVICE)
        encoder_state, pretrain_loss = pretrain_model(
            pretrain_model_inst, pretrain_train_loader, pretrain_val_loader,
            pretrain_lr, weight_decay, PRETRAIN_EPOCHS, PATIENCE
        )

        # Load fine-tune data
        train_graphs, val_graphs, test_graphs = load_labeled_graphs()
        train_loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)), shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=min(batch_size, len(val_graphs)))

        # Fine-tune
        graph_n_features = train_graphs[0].x.shape[1]  # 33
        class_weights = compute_class_weights(train_graphs)

        classifier = SOZClassifier(
            graph_n_features, hidden_dim, num_layers, dropout,
            classifier_dropout, pretrain_in_channels=n_features
        ).to(DEVICE)

        best_state, val_auc = finetune_model(
            classifier, encoder_state, train_loader, val_loader, class_weights,
            finetune_lr, weight_decay, FINETUNE_EPOCHS, PATIENCE, freeze_epochs,
            aug_edge_drop=aug_edge_drop, aug_feat_noise=aug_feat_noise,
            aug_feat_mask=aug_feat_mask, mixup_alpha=mixup_alpha,
            smote_k=smote_k, smote_ratio=smote_ratio
        )

        # Report intermediate value for pruning
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
    print("Hyperparameter Tuning with Optuna")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trials: {N_TRIALS}")
    print(f"Pretrain epochs per trial: {PRETRAIN_EPOCHS}")
    print(f"Finetune epochs per trial: {FINETUNE_EPOCHS}")
    print(f"Data Augmentation: {USE_AUGMENTATION}")
    if USE_AUGMENTATION:
        print(f"  - Edge dropout: tuned [0.0, 0.25]")
        print(f"  - Feature noise: tuned [0.0, 0.1]")
        print(f"  - Feature mask: tuned [0.0, 0.2]")
    print(f"Mixup: {USE_MIXUP}")
    if USE_MIXUP:
        print(f"  - Alpha: tuned [0.0, 0.5]")
    print(f"SMOTE: {USE_SMOTE}")
    if USE_SMOTE:
        print(f"  - K neighbors: tuned [2, 10]")
        print(f"  - Target ratio: tuned [0.3, 1.5]")
    print("-" * 70)

    t_start = time.perf_counter()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation AUC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best hyperparameters
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_path = MODELS_DIR / "best_hyperparams.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved: {best_params_path}")

    # Retrain with best hyperparameters and evaluate on test
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
    pretrain_model_inst = PretrainModel(
        n_features, params["hidden_dim"], params["num_layers"], params["dropout"]
    ).to(DEVICE)

    encoder_state, _ = pretrain_model(
        pretrain_model_inst, pretrain_train_loader, pretrain_val_loader,
        params["pretrain_lr"], params["weight_decay"], PRETRAIN_EPOCHS * 2, PATIENCE
    )

    # Save tuned encoder
    torch.save(encoder_state, MODELS_DIR / "tuned_encoder.pt")

    # Fine-tune
    train_graphs, val_graphs, test_graphs = load_labeled_graphs()
    train_loader = DataLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=params["batch_size"])
    test_loader = DataLoader(test_graphs, batch_size=params["batch_size"])

    graph_n_features = train_graphs[0].x.shape[1]
    class_weights = compute_class_weights(train_graphs)

    classifier = SOZClassifier(
        graph_n_features, params["hidden_dim"], params["num_layers"], params["dropout"],
        params["classifier_dropout"], pretrain_in_channels=n_features
    ).to(DEVICE)

    best_state, val_auc = finetune_model(
        classifier, encoder_state, train_loader, val_loader, class_weights,
        params["finetune_lr"], params["weight_decay"], FINETUNE_EPOCHS * 2, PATIENCE,
        params["freeze_epochs"],
        aug_edge_drop=params.get("aug_edge_drop", EDGE_DROP_RATE),
        aug_feat_noise=params.get("aug_feat_noise", FEATURE_NOISE_STD),
        aug_feat_mask=params.get("aug_feat_mask", FEATURE_MASK_RATE),
        mixup_alpha=params.get("mixup_alpha", MIXUP_ALPHA),
        smote_k=params.get("smote_k_neighbors", SMOTE_K_NEIGHBORS),
        smote_ratio=params.get("smote_ratio", SMOTE_RATIO)
    )

    classifier.load_state_dict(best_state)
    test_auc = evaluate_test(classifier, test_loader, class_weights)

    # Save tuned classifier
    torch.save(best_state, MODELS_DIR / "tuned_classifier.pt")

    print(f"\nFinal Results (with best hyperparameters):")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC:       {test_auc:.4f}")

    # Plot optimization history
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Optimization history
        trials = [t for t in study.trials if t.value is not None]
        values = [t.value for t in trials]
        axes[0].plot(range(len(values)), values, "b.-", alpha=0.7)
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Validation AUC")
        axes[0].set_title("Optimization History")
        axes[0].axhline(max(values), color="r", linestyle="--", label=f"Best: {max(values):.4f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Parameter importance
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[:10]
        vals = [importances[n] for n in names]
        axes[1].barh(names, vals, color="steelblue")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Hyperparameter Importance")
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "optuna_optimization.png"
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
        stage="tune_gnn",
        status="success",
        lines=[
            f"Backend: {backend}",
            f"Trials: {N_TRIALS}",
            f"Augmentation: {USE_AUGMENTATION}",
            f"Best val AUC: {study.best_value:.4f}",
            f"Test AUC (tuned): {test_auc:.4f}",
            f"Best hidden_dim: {params['hidden_dim']}",
            f"Best num_layers: {params['num_layers']}",
            f"Best finetune_lr: {params['finetune_lr']:.2e}",
            f"Best aug_edge_drop: {params.get('aug_edge_drop', 'N/A')}",
            f"Best aug_feat_noise: {params.get('aug_feat_noise', 'N/A')}",
            f"Best aug_feat_mask: {params.get('aug_feat_mask', 'N/A')}",
            f"Best mixup_alpha: {params.get('mixup_alpha', 'N/A')}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
