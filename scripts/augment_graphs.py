"""
Time-Shift Data Augmentation for SOZ Localization GNN
======================================================

Creates multiple augmented versions of each training graph by:
1. Using different window subsets (simulates time-shifted epochs)
2. Adding slight noise to features
3. Dropping random windows before aggregation

This multiplies the training data without re-running preprocessing.

Input
-----
    data/processed/h5/features.h5
    data/processed/graphs/graphs.pt

Output
------
    data/processed/graphs/graphs_augmented.pt
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"
GRAPHS_PT = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs.pt"
OUTPUT_PT = PROJECT_DIR / "data" / "processed" / "graphs" / "graphs_augmented.pt"
MODELS_DIR = PROJECT_DIR / "data" / "processed" / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation settings
N_AUGMENTED_VERSIONS = 4      # Number of augmented copies per training graph
WINDOW_DROP_RATE = 0.2        # Drop 20% of windows randomly
FEATURE_NOISE_STD = 0.02      # Small noise on aggregated features
CORRELATION_THRESHOLD = 0.3   # Same as original graphs
RANDOM_SEED = 42

# ============================================================================
# UTILITIES
# ============================================================================

def zscore(features: np.ndarray) -> np.ndarray:
    """Z-score normalize features."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((features - mean) / std).astype(np.float32)


def aggregate_features(window_features: np.ndarray) -> np.ndarray:
    """Aggregate window features using mean, std, max."""
    return np.concatenate([
        window_features.mean(axis=0),
        window_features.std(axis=0),
        window_features.max(axis=0)
    ], axis=-1)


def build_edge_index(corr_matrix: np.ndarray, threshold: float) -> torch.Tensor:
    """Build edge index from correlation matrix."""
    n = corr_matrix.shape[0]
    mask = np.triu(np.abs(corr_matrix) >= threshold, k=1)
    src, dst = np.where(mask)
    if len(src) == 0:
        idx = np.arange(n)
        return torch.tensor(np.stack([idx, idx]), dtype=torch.long)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(edge_index, dtype=torch.long)


def create_augmented_graph(
    window_node_features: np.ndarray,
    correlation_windows: np.ndarray,
    labels: np.ndarray,
    channels: list,
    soz_channels: list,
    rec_name: str,
    aug_idx: int,
    original_graph: Data,
    window_drop_rate: float = WINDOW_DROP_RATE,
    feature_noise_std: float = FEATURE_NOISE_STD,
    threshold: float = CORRELATION_THRESHOLD,
) -> Data:
    """Create an augmented graph from window-level features.

    Augmentation:
    1. Randomly drop some windows (simulates time shift)
    2. Re-aggregate features from remaining windows
    3. Add small noise to features
    """
    n_windows = window_node_features.shape[0]
    n_channels = window_node_features.shape[1]

    # Randomly select windows (drop some)
    n_keep = max(1, int(n_windows * (1 - window_drop_rate)))
    rng = np.random.RandomState(RANDOM_SEED + aug_idx * 1000 + hash(rec_name) % 10000)
    keep_idx = rng.choice(n_windows, size=n_keep, replace=False)
    keep_idx.sort()

    # Subset windows
    window_feats_subset = window_node_features[keep_idx]
    corr_subset = correlation_windows[keep_idx]

    # Re-aggregate
    node_features = aggregate_features(window_feats_subset)
    corr_agg = corr_subset.mean(axis=0)

    # Add noise
    noise = rng.randn(*node_features.shape).astype(np.float32) * feature_noise_std
    node_features = node_features + noise

    # Z-score normalize
    node_features = zscore(node_features)

    # Build edge index
    edge_index = build_edge_index(corr_agg, threshold)

    # Create PyG Data object
    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    n_soz = int(labels.sum())

    # Build edge_attr from correlation (simplified - use mean correlation as edge weight)
    n_edges = edge_index.shape[1]
    edge_attr = torch.zeros(n_edges, 4, dtype=torch.float32)
    for i in range(n_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        corr_val = float(corr_agg[src, dst])
        edge_attr[i, 0] = corr_val  # correlation
        edge_attr[i, 1] = corr_val  # placeholder for coherence_low_gamma
        edge_attr[i, 2] = corr_val  # placeholder for coherence_hfo
        edge_attr[i, 3] = corr_val  # placeholder for plv_low_gamma

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.rec_name = f"{rec_name}_aug{aug_idx}"
    data.channels = channels
    data.soz_channels = soz_channels
    data.n_soz = n_soz
    data.n_channels = n_channels
    # Copy other attributes from original graph
    data.sfreq = original_graph.sfreq
    data.n_windows = len(keep_idx)
    data.engel = original_graph.engel
    data.subject = original_graph.subject
    data.outcome = original_graph.outcome

    return data


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Time-Shift Data Augmentation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Augmented versions per training graph: {N_AUGMENTED_VERSIONS}")
    print(f"Window drop rate: {WINDOW_DROP_RATE}")
    print(f"Feature noise std: {FEATURE_NOISE_STD}")
    print("-" * 70)

    t_start = time.perf_counter()

    # Load original graphs
    if not GRAPHS_PT.exists():
        raise FileNotFoundError(f"Original graphs not found: {GRAPHS_PT}")

    original_graphs = torch.load(GRAPHS_PT, weights_only=False)
    print(f"Loaded {len(original_graphs)} original graphs")

    # Separate by split
    train_graphs = [g for g in original_graphs if g.split == "train"]
    val_graphs = [g for g in original_graphs if g.split == "val"]
    test_graphs = [g for g in original_graphs if g.split == "test"]
    unlabeled_graphs = [g for g in original_graphs if g.split == "unlabeled"]

    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}, Unlabeled: {len(unlabeled_graphs)}")

    # Load features for augmentation
    if not FEATURES_H5.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_H5}")

    augmented_train = []

    with h5py.File(FEATURES_H5, "r") as f:
        # Create mapping from graph name to recording name
        rec_names = list(f.keys())

        for graph in tqdm(train_graphs, desc="Augmenting training graphs"):
            # Find matching recording in features
            graph_rec_name = graph.rec_name

            # Try to find the recording
            matched_rec = None
            for rec_name in rec_names:
                if rec_name in graph_rec_name or graph_rec_name in rec_name:
                    matched_rec = rec_name
                    break

            if matched_rec is None:
                # Try exact match
                if graph_rec_name in rec_names:
                    matched_rec = graph_rec_name

            if matched_rec is None:
                print(f"  Warning: No matching recording for {graph_rec_name}")
                continue

            grp = f[matched_rec]

            # Check if we have window-level features
            if "window_node_features" not in grp:
                print(f"  Warning: No window features for {matched_rec}")
                continue

            window_node_features = grp["window_node_features"][:]
            n_windows = window_node_features.shape[0]

            if n_windows < 2:
                print(f"  Warning: Only {n_windows} window(s) for {matched_rec}, skipping")
                continue

            # Get correlation windows
            if "edge_features/correlation_windows" in grp:
                correlation_windows = grp["edge_features/correlation_windows"][:]
            else:
                # Fall back to computing from aggregated
                correlation_windows = np.tile(
                    grp["edge_features/correlation"][:, :, 0],
                    (n_windows, 1, 1)
                )

            labels = grp["labels"][:]
            channels = list(grp.attrs["channels"])
            soz_channels = list(grp.attrs["soz_channels"])

            # Create augmented versions
            for aug_idx in range(N_AUGMENTED_VERSIONS):
                aug_graph = create_augmented_graph(
                    window_node_features=window_node_features,
                    correlation_windows=correlation_windows,
                    labels=labels,
                    channels=channels,
                    soz_channels=soz_channels,
                    rec_name=matched_rec,
                    aug_idx=aug_idx,
                    original_graph=graph,
                )
                aug_graph.split = "train"
                augmented_train.append(aug_graph)

    print(f"\nCreated {len(augmented_train)} augmented training graphs")

    # Combine: original train + augmented train + val + test + unlabeled
    all_graphs = train_graphs + augmented_train + val_graphs + test_graphs + unlabeled_graphs

    # Stats
    n_train_total = len(train_graphs) + len(augmented_train)
    n_train_with_soz = len([g for g in train_graphs + augmented_train if g.n_soz > 0])

    print(f"\nFinal dataset:")
    print(f"  Total graphs: {len(all_graphs)}")
    print(f"  Training graphs: {n_train_total} ({len(train_graphs)} original + {len(augmented_train)} augmented)")
    print(f"  Training graphs with SOZ: {n_train_with_soz}")
    print(f"  Validation graphs: {len(val_graphs)}")
    print(f"  Test graphs: {len(test_graphs)}")

    # Save
    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_graphs, OUTPUT_PT)
    print(f"\nSaved: {OUTPUT_PT}")

    elapsed = time.perf_counter() - t_start
    print(f"\nRuntime: {elapsed:.1f}s")

    # Log
    append_project_log(
        stage="augment_graphs",
        status="success",
        lines=[
            f"Augmented versions per graph: {N_AUGMENTED_VERSIONS}",
            f"Window drop rate: {WINDOW_DROP_RATE}",
            f"Original train graphs: {len(train_graphs)}",
            f"Augmented train graphs: {len(augmented_train)}",
            f"Total train graphs: {n_train_total}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )

    return len(augmented_train)


if __name__ == "__main__":
    main()
