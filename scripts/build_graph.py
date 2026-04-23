"""
Graph Construction for SOZ Localization GNN
=============================================

This script builds PyTorch Geometric (PyG) graph objects from the features
extracted in Stage 4.  Each recording becomes one graph where:

- **Nodes** = iEEG channels
- **Node features** = aggregated spectral / statistical features
- **Edge index + edge attr** = functional connectivity (thresholded correlation)
- **Labels** = binary SOZ indicator per channel (node-level classification)

The output is a list of PyG ``Data`` objects saved as a single ``.pt`` file,
ready for data-loading in a GNN training loop.

Graph construction choices
--------------------------
1. **Edges from correlation**: We threshold the absolute aggregated Pearson
   correlation to decide which channel pairs are connected.  A moderate
   threshold (default 0.3) keeps the graph sparse enough for efficient message
   passing while preserving physiologically meaningful connections.

2. **Edge attributes**: For every retained edge we store a small feature
   vector consisting of the mean aggregated correlation, coherence (low-gamma,
   HFO), and PLV (low-gamma).  These give the GNN richer information about the
   nature of each connection.

3. **Node features are z-scored per graph**: This removes cross-recording
   amplitude differences while preserving the relative feature pattern within
   a recording — important because electrode impedance and amplifier gain
   vary across sessions and subjects.

4. **Subject-level split**: We partition recordings into train / val / test at
   the *subject* level so no data from the same patient leaks between splits.

Input
-----
    data/processed/h5/features.h5      (from extract_features.py)
    data/processed/csv/metadata.csv    (for outcome / Engel labels)

Output
------
    data/processed/graphs/graphs.pt        – list[Data]
    data/processed/graphs/graph_info.csv   – per-graph summary
    data/processed/figures/graph_demo.png  – sample graph visualisation
"""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from log_utils import append_project_log

# ============================================================================
# Try importing torch / torch_geometric; give a clear error if missing.
# ============================================================================
try:
    import torch
    from torch_geometric.data import Data
except ImportError as exc:
    raise ImportError(
        "PyTorch and PyTorch Geometric are required for graph construction.\n"
        "Install them with:\n"
        "  pip install torch torch_geometric\n"
    ) from exc

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"
METADATA_CSV = PROJECT_DIR / "data" / "processed" / "csv" / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "graphs"
FIGURES_DIR = PROJECT_DIR / "data" / "processed" / "figures"

OUTPUT_PT = OUTPUT_DIR / "graphs.pt"
OUTPUT_CSV = OUTPUT_DIR / "graph_info.csv"

# ---------------------------------------------------------------------------
# Edge construction parameters
# ---------------------------------------------------------------------------
# Threshold on |mean correlation| to create an edge.  Pairs with weaker
# correlation are dropped, keeping the graph sparse.
CORRELATION_THRESHOLD = 0.3

# If True the graph is undirected — each edge (i, j) also appears as (j, i).
# Standard for functional-connectivity graphs.
UNDIRECTED = True

# ---------------------------------------------------------------------------
# Node feature normalisation
# ---------------------------------------------------------------------------
# Z-score node features within each graph so that GNN layers see standardised
# inputs regardless of absolute amplifier gain.
ZSCORE_NODE_FEATURES = True

# ---------------------------------------------------------------------------
# Train / Val / Test split (subject-level)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42


# ============================================================================
# METADATA HELPERS
# ============================================================================

def load_metadata(csv_path: Path) -> dict[str, dict]:
    """Load metadata.csv into a lookup keyed by (subject, session, run)."""
    meta = {}
    if not csv_path.exists():
        return meta
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = f"{row['subject']}_{row['session']}_{row['run']}"
            meta[key] = row
    return meta


def _subject_from_rec_name(rec_name: str) -> str:
    """Extract the subject ID from an HDF5 recording name.

    Recording names follow the BIDS pattern, e.g.
    ``sub-jh101_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg``
    """
    for part in rec_name.split("_"):
        if part.startswith("sub-"):
            return part
    # Fallback: use the first segment before underscore
    return rec_name.split("_")[0]


# ============================================================================
# GRAPH BUILDING
# ============================================================================

def build_edge_index_and_attr(
    edge_features: dict[str, np.ndarray],
    threshold: float,
    edge_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge index and edge attribute matrix from connectivity matrices.

    Parameters
    ----------
    edge_features : dict[str, np.ndarray]
        Aggregated edge feature matrices.  Each value has shape
        ``(n_channels, n_channels, n_aggs)``; we use the *mean* aggregation
        (index 0) for the correlation-based thresholding decision.
    threshold : float
        Minimum absolute mean-correlation to keep an edge.
    edge_feature_names : list[str]
        Ordered list of edge feature names to include in edge_attr.

    Returns
    -------
    edge_index : np.ndarray, shape (2, n_edges)
    edge_attr  : np.ndarray, shape (n_edges, n_edge_features)
    """
    # Use mean-aggregated correlation (agg index 0) for thresholding.
    corr_mean = edge_features["correlation"][:, :, 0]  # (C, C)
    n_channels = corr_mean.shape[0]

    src_list, dst_list, attr_list = [], [], []

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if abs(corr_mean[i, j]) < threshold:
                continue

            # Collect one feature per connectivity metric (mean agg only).
            feat = []
            for name in edge_feature_names:
                if name in edge_features:
                    feat.append(edge_features[name][i, j, 0])
                else:
                    feat.append(0.0)
            feat = np.array(feat, dtype=np.float32)

            # Undirected: add both directions.
            src_list.extend([i, j])
            dst_list.extend([j, i])
            attr_list.extend([feat, feat])

    if len(src_list) == 0:
        # Degenerate: no edges pass threshold → create self-loops so the
        # graph is not completely empty (GNN layers need at least something).
        src_list = list(range(n_channels))
        dst_list = list(range(n_channels))
        n_feat = len(edge_feature_names)
        attr_list = [np.zeros(n_feat, dtype=np.float32)] * n_channels

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr = np.stack(attr_list, axis=0).astype(np.float32)
    return edge_index, edge_attr


def zscore_features(features: np.ndarray) -> np.ndarray:
    """Z-score features along the node axis (axis 0)."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # Avoid division by zero for constant features.
    return ((features - mean) / std).astype(np.float32)


# ============================================================================
# SPLIT LOGIC
# ============================================================================

def subject_level_split(
    subjects: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign each subject to train / val / test.

    Returns a dict mapping subject → split name.
    """
    rng = random.Random(seed)
    subjects_sorted = sorted(set(subjects))
    rng.shuffle(subjects_sorted)

    n = len(subjects_sorted)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    # Remainder goes to test.

    split_map: dict[str, str] = {}
    for i, subj in enumerate(subjects_sorted):
        if i < n_train:
            split_map[subj] = "train"
        elif i < n_train + n_val:
            split_map[subj] = "val"
        else:
            split_map[subj] = "test"
    return split_map


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def build_all_graphs() -> list:
    """Read features.h5, construct PyG graphs, and save them."""
    print("=" * 70)
    print("Building PyTorch Geometric Graphs")
    print("=" * 70)
    print(f"Input features: {FEATURES_H5}")
    print(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    print(f"Z-score node features: {ZSCORE_NODE_FEATURES}")
    print(f"Output: {OUTPUT_PT}")

    if not FEATURES_H5.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_H5}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load optional metadata for outcome / Engel info.
    meta_lookup = load_metadata(METADATA_CSV)

    graphs: list[Data] = []
    info_rows: list[dict] = []
    subjects_seen: list[str] = []
    skipped: list[tuple[str, str]] = []

    with h5py.File(FEATURES_H5, "r") as f:
        # Read global attributes written by extract_features.
        node_feature_names = list(f.attrs.get("node_feature_names", []))
        edge_feature_names = list(f.attrs.get("edge_feature_names", []))
        n_node_features = int(f.attrs.get("n_node_features", 0))
        agg_names = list(f.attrs.get("aggregation_names", []))

        rec_names = list(f.keys())
        print(f"Found {len(rec_names)} recordings in features.h5")
        print("-" * 70)

        for rec_name in tqdm(rec_names, desc="Building graphs"):
            grp = f[rec_name]

            # ----- Node features (aggregated) -----
            node_features = grp["node_features"][:]  # (C, F)
            labels = grp["labels"][:]                 # (C,)
            channels = list(grp.attrs["channels"])
            sfreq = float(grp.attrs["sfreq"])
            n_channels = int(grp.attrs["n_channels"])
            n_soz = int(grp.attrs["n_soz"])
            n_windows = int(grp.attrs["n_windows"])
            soz_channels = list(grp.attrs["soz_channels"])

            if n_channels < 2:
                skipped.append((rec_name, "fewer than 2 channels"))
                continue

            # ----- Edge features (aggregated) -----
            edge_grp = grp["edge_features"]
            edge_features_dict: dict[str, np.ndarray] = {}
            for ename in edge_feature_names:
                if ename in edge_grp:
                    edge_features_dict[ename] = edge_grp[ename][:]

            if "correlation" not in edge_features_dict:
                skipped.append((rec_name, "no correlation edge feature"))
                continue

            # ----- Build edges -----
            edge_index, edge_attr = build_edge_index_and_attr(
                edge_features_dict, CORRELATION_THRESHOLD, edge_feature_names
            )

            # ----- Normalise node features -----
            x = node_features.astype(np.float32)
            if ZSCORE_NODE_FEATURES:
                x = zscore_features(x)

            # ----- Identify subject -----
            subject = _subject_from_rec_name(rec_name)
            subjects_seen.append(subject)

            # ----- Optional metadata -----
            # Try to match metadata row for outcome / Engel.
            outcome, engel = "NR", -1.0
            for mkey, mrow in meta_lookup.items():
                # Match on subject + run if possible.
                if mrow["subject"] == subject:
                    if mrow.get("run", "") in rec_name:
                        outcome = mrow.get("outcome", "NR")
                        try:
                            engel = float(mrow.get("engel", -1))
                        except (ValueError, TypeError):
                            engel = -1.0
                        break

            # ----- Assemble PyG Data object -----
            data = Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
                y=torch.from_numpy(labels.astype(np.int64)),
            )
            # Store extra metadata as plain attributes (not tensors).
            data.rec_name = rec_name
            data.subject = subject
            data.channels = channels
            data.soz_channels = soz_channels
            data.n_channels = n_channels
            data.n_soz = n_soz
            data.n_windows = n_windows
            data.sfreq = sfreq
            data.outcome = outcome
            data.engel = engel

            graphs.append(data)

            n_edges = edge_index.shape[1] // 2 if UNDIRECTED else edge_index.shape[1]
            info_rows.append({
                "rec_name": rec_name,
                "subject": subject,
                "n_channels": n_channels,
                "n_soz": n_soz,
                "n_edges": n_edges,
                "n_node_features": x.shape[1],
                "n_edge_features": edge_attr.shape[1],
                "n_windows": n_windows,
                "outcome": outcome,
                "engel": engel,
            })

    # ------------------------------------------------------------------
    # Subject-level split
    # ------------------------------------------------------------------
    unique_subjects = sorted(set(subjects_seen))
    split_map = subject_level_split(unique_subjects, TRAIN_RATIO, VAL_RATIO, SPLIT_SEED)

    for i, g in enumerate(graphs):
        g.split = split_map.get(g.subject, "train")
        info_rows[i]["split"] = g.split

    # ------------------------------------------------------------------
    # Save graphs
    # ------------------------------------------------------------------
    torch.save(graphs, OUTPUT_PT)

    # Save info CSV
    if info_rows:
        fieldnames = list(info_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(info_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("-" * 70)
    print(f"Graphs built: {len(graphs)}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    split_counts = defaultdict(int)
    for g in graphs:
        split_counts[g.split] += 1
    print(f"\nSubject-level split (seed={SPLIT_SEED}):")
    print(f"  Subjects: {len(unique_subjects)}")
    for sp in ["train", "val", "test"]:
        n_subj = sum(1 for s in unique_subjects if split_map.get(s) == sp)
        print(f"  {sp:>5s}: {n_subj} subjects, {split_counts[sp]} graphs")

    soz_counts = sum(1 for g in graphs if g.n_soz > 0)
    print(f"\nGraphs with SOZ labels: {soz_counts}/{len(graphs)}")
    if graphs:
        avg_nodes = np.mean([g.n_channels for g in graphs])
        avg_edges = np.mean([g.edge_index.shape[1] / 2 for g in graphs])
        print(f"Average nodes/graph: {avg_nodes:.1f}")
        print(f"Average edges/graph: {avg_edges:.1f}")

    print(f"\nOutput saved to:")
    print(f"  {OUTPUT_PT}")
    print(f"  {OUTPUT_CSV}")

    return graphs


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_graphs(graphs: list) -> None:
    """Print verification info for the first graph."""
    print("\n" + "=" * 70)
    print("Verifying Graph Construction")
    print("=" * 70)

    if not graphs:
        print("No graphs to verify.")
        return

    g = graphs[0]
    print(f"Sample graph: {g.rec_name}")
    print(f"  subject:        {g.subject}")
    print(f"  split:          {g.split}")
    print(f"  x (node feat):  {tuple(g.x.shape)}")
    print(f"  edge_index:     {tuple(g.edge_index.shape)}")
    print(f"  edge_attr:      {tuple(g.edge_attr.shape)}")
    print(f"  y (labels):     {tuple(g.y.shape)}  (SOZ={g.y.sum().item()})")
    print(f"  n_windows:      {g.n_windows}")
    print(f"  outcome:        {g.outcome}")
    print(f"  engel:          {g.engel}")

    # Quick sanity checks
    assert g.x.shape[0] == g.y.shape[0], "Node count mismatch between x and y"
    assert g.edge_index.max() < g.x.shape[0], "Edge index out of bounds"
    assert not torch.isnan(g.x).any(), "NaN in node features"
    assert not torch.isnan(g.edge_attr).any(), "NaN in edge attributes"
    print("  ✓ All sanity checks passed")


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_graph_demo(graphs: list) -> None:
    """Create a diagnostic figure for a sample graph."""
    print("\n" + "=" * 70)
    print("Generating Graph Visualisation")
    print("=" * 70)

    if not graphs:
        print("No graphs available for plotting.")
        return

    # Pick a graph that has SOZ labels for a more interesting plot.
    demo_graph = None
    for g in graphs:
        if g.n_soz > 0:
            demo_graph = g
            break
    if demo_graph is None:
        demo_graph = graphs[0]

    g = demo_graph
    n = g.x.shape[0]
    ei = g.edge_index.numpy()
    labels = g.y.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Graph Demo: {g.rec_name}\n"
        f"{n} nodes, {ei.shape[1] // 2} edges, "
        f"{g.n_soz} SOZ channels, split={g.split}",
        fontsize=14, fontweight="bold",
    )

    # ---- Top-left: adjacency matrix ----
    adj = np.zeros((n, n), dtype=np.float32)
    for k in range(ei.shape[1]):
        adj[ei[0, k], ei[1, k]] = 1.0
    im = axes[0, 0].imshow(adj, cmap="Blues", aspect="auto")
    axes[0, 0].set_title("Adjacency Matrix")
    axes[0, 0].set_xlabel("Channel")
    axes[0, 0].set_ylabel("Channel")
    plt.colorbar(im, ax=axes[0, 0])

    # ---- Top-right: node degree distribution ----
    degree = np.array([(ei[0] == i).sum() for i in range(n)])
    axes[0, 1].hist(degree, bins=max(10, int(np.sqrt(n))), color="steelblue",
                    edgecolor="white", alpha=0.85)
    axes[0, 1].set_title("Node Degree Distribution")
    axes[0, 1].set_xlabel("Degree")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].axvline(degree.mean(), color="red", linestyle="--",
                       label=f"mean={degree.mean():.1f}")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ---- Bottom-left: node feature heatmap ----
    x_np = g.x.numpy()
    im2 = axes[1, 0].imshow(x_np.T, aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_title("Node Features (z-scored)")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("Feature")
    plt.colorbar(im2, ax=axes[1, 0])

    # Highlight SOZ channels on x-axis
    soz_mask = labels == 1
    for idx in np.where(soz_mask)[0]:
        axes[1, 0].axvline(idx, color="red", linewidth=0.5, alpha=0.4)

    # ---- Bottom-right: label distribution across all graphs ----
    total_soz = sum(g2.y.sum().item() for g2 in graphs)
    total_non_soz = sum(g2.y.shape[0] - g2.y.sum().item() for g2 in graphs)
    bars = axes[1, 1].bar(["Non-SOZ", "SOZ"], [total_non_soz, total_soz],
                          color=["steelblue", "crimson"], alpha=0.85,
                          edgecolor="white")
    axes[1, 1].set_title("Label Distribution (All Graphs)")
    axes[1, 1].set_ylabel("Node Count")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    # Annotate bars
    for bar, val in zip(bars, [total_non_soz, total_soz]):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{int(val)}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "graph_demo.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main entry point for graph construction."""
    t_start = time.perf_counter()

    graphs = build_all_graphs()

    if graphs:
        verify_graphs(graphs)
        plot_graph_demo(graphs)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("Graph construction complete!")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    append_project_log(
        stage="build_graph",
        status="success",
        lines=[
            f"Input: {FEATURES_H5}",
            f"Output: {OUTPUT_PT}",
            f"Correlation threshold: {CORRELATION_THRESHOLD}",
            f"Z-score node features: {ZSCORE_NODE_FEATURES}",
            f"Total graphs: {len(graphs)}",
            f"Graphs with SOZ: {sum(1 for g in graphs if g.n_soz > 0)}",
            f"Split seed: {SPLIT_SEED}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
