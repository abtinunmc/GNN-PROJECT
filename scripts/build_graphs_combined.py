"""
Build PyTorch Geometric graphs from combined dataset features
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from log_utils import append_project_log


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FEATURES_H5 = PROJECT_DIR / "data" / "processed_combined" / "h5" / "features_extracted.h5"
OUTPUT_PT = PROJECT_DIR / "data" / "processed_combined" / "graphs" / "graphs_combined.pt"

CORRELATION_THRESHOLD = 0.3
RANDOM_SEED = 42


def zscore(features: np.ndarray) -> np.ndarray:
    """Z-score normalize features."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((features - mean) / std).astype(np.float32)


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


def build_edge_attr(corr_agg: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
    """Build edge attributes from correlation features."""
    n_edges = edge_index.shape[1]
    n_edge_features = corr_agg.shape[-1]
    edge_attr = torch.zeros(n_edges, n_edge_features, dtype=torch.float32)

    for i in range(n_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_attr[i] = torch.tensor(corr_agg[src, dst])

    return edge_attr


def main():
    print("=" * 70)
    print("Building Graphs for Combined Dataset")
    print("=" * 70)

    t_start = time.perf_counter()

    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)

    # Load features and build graphs
    graphs = []
    stats = {"total": 0, "with_soz": 0, "ds003029": 0, "ds004100": 0}

    with h5py.File(FEATURES_H5, "r") as f:
        rec_names = list(f.keys())
        print(f"Processing {len(rec_names)} recordings...")

        # Get all subjects for split assignment
        subjects = {}
        for rec_name in rec_names:
            grp = f[rec_name]
            subj = grp.attrs.get("subject", rec_name.split("_")[1])
            dataset = grp.attrs.get("dataset", "unknown")
            key = f"{dataset}_{subj}"
            if key not in subjects:
                subjects[key] = []
            subjects[key].append(rec_name)

        # Split subjects into train/val/test (60/20/20)
        np.random.seed(RANDOM_SEED)
        subject_list = list(subjects.keys())
        np.random.shuffle(subject_list)

        n_subjects = len(subject_list)
        n_train = int(0.6 * n_subjects)
        n_val = int(0.2 * n_subjects)

        train_subjects = set(subject_list[:n_train])
        val_subjects = set(subject_list[n_train:n_train + n_val])
        test_subjects = set(subject_list[n_train + n_val:])

        print(f"Subjects: {n_subjects} total ({len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test)")

        for rec_name in tqdm(rec_names, desc="Building graphs"):
            grp = f[rec_name]

            # Load features
            node_features = grp["node_features"][:]
            corr_agg = grp["edge_features/correlation"][:]
            labels = grp["labels"][:]

            # Get metadata
            channels = list(grp.attrs["channels"])
            soz_channels = list(grp.attrs["soz_channels"])
            sfreq = float(grp.attrs["sfreq"])
            n_windows = int(grp.attrs["n_windows"])
            dataset = grp.attrs.get("dataset", "unknown")
            subject = grp.attrs.get("subject", rec_name.split("_")[1])
            engel = grp.attrs.get("engel", "n/a")
            outcome = grp.attrs.get("outcome", "n/a")

            # Normalize features
            node_features = zscore(node_features)

            # Build graph structure
            corr_mean = corr_agg[:, :, 0]  # Use mean correlation
            edge_index = build_edge_index(corr_mean, CORRELATION_THRESHOLD)
            edge_attr = build_edge_attr(corr_agg, edge_index)

            # Create PyG Data object
            x = torch.tensor(node_features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.rec_name = rec_name
            data.channels = channels
            data.soz_channels = soz_channels
            data.n_soz = int(labels.sum())
            data.n_channels = len(channels)
            data.sfreq = sfreq
            data.n_windows = n_windows
            data.dataset = dataset
            data.subject = subject
            data.engel = engel
            data.outcome = outcome

            # Assign split
            subj_key = f"{dataset}_{subject}"
            if subj_key in train_subjects:
                data.split = "train"
            elif subj_key in val_subjects:
                data.split = "val"
            else:
                data.split = "test"

            # Mark unlabeled if no SOZ
            if data.n_soz == 0:
                data.split = "unlabeled"

            graphs.append(data)
            stats["total"] += 1
            if data.n_soz > 0:
                stats["with_soz"] += 1
            if dataset == "ds003029":
                stats["ds003029"] += 1
            else:
                stats["ds004100"] += 1

    # Save graphs
    torch.save(graphs, OUTPUT_PT)

    # Count splits
    train_graphs = [g for g in graphs if g.split == "train"]
    val_graphs = [g for g in graphs if g.split == "val"]
    test_graphs = [g for g in graphs if g.split == "test"]
    unlabeled_graphs = [g for g in graphs if g.split == "unlabeled"]

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total graphs: {stats['total']}")
    print(f"  - ds003029: {stats['ds003029']}")
    print(f"  - ds004100: {stats['ds004100']}")
    print(f"With SOZ labels: {stats['with_soz']}")
    print(f"Splits:")
    print(f"  - Train: {len(train_graphs)}")
    print(f"  - Val: {len(val_graphs)}")
    print(f"  - Test: {len(test_graphs)}")
    print(f"  - Unlabeled: {len(unlabeled_graphs)}")
    print(f"Output: {OUTPUT_PT}")
    print(f"Runtime: {elapsed:.1f}s")

    append_project_log(
        stage="build_graphs_combined",
        status="success",
        lines=[
            f"Total graphs: {stats['total']}",
            f"ds003029: {stats['ds003029']}",
            f"ds004100: {stats['ds004100']}",
            f"With SOZ: {stats['with_soz']}",
            f"Train: {len(train_graphs)}",
            f"Val: {len(val_graphs)}",
            f"Test: {len(test_graphs)}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
