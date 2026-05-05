"""
Feature extraction for combined dataset (ds003029 + ds004100)
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False
    DEVICE = None

from log_utils import append_project_log


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_H5 = PROJECT_DIR / "data" / "processed_combined" / "h5" / "features_combined.h5"
OUTPUT_H5 = PROJECT_DIR / "data" / "processed_combined" / "h5" / "features_extracted.h5"

WINDOW_SEC = 12.0
WINDOW_STEP_SEC = 12.0
MIN_WINDOW_SAMPLES = 1024

FREQUENCY_BANDS = {
    "delta":     (1,   4),
    "theta":     (4,   8),
    "alpha":     (8,  12),
    "beta":      (15, 25),
    "low_gamma": (35, 50),
    "hfo":       (80, 150),
}


def bandpower_welch(data: np.ndarray, sfreq: float, band: tuple, nperseg: int = 256) -> np.ndarray:
    """Compute band power using Welch's method."""
    fmin, fmax = band
    freqs, psd = scipy_signal.welch(data, fs=sfreq, nperseg=min(nperseg, data.shape[-1]))
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def line_length(data: np.ndarray) -> np.ndarray:
    """Compute line length feature."""
    return np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)


def compute_node_features(window: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute node features for a single window."""
    n_channels = window.shape[0]
    features = []

    # Band powers
    for band_name, band_range in FREQUENCY_BANDS.items():
        bp = bandpower_welch(window, sfreq, band_range)
        features.append(bp)

    # Statistical features
    features.append(np.var(window, axis=1))
    features.append(line_length(window))
    features.append(kurtosis(window, axis=1))
    features.append(skew(window, axis=1))

    # Relative powers
    total_power = bandpower_welch(window, sfreq, (1, min(150, sfreq/2 - 1)))
    total_power[total_power < 1e-10] = 1e-10
    hfo_power = bandpower_welch(window, sfreq, (80, min(150, sfreq/2 - 1)))
    features.append(hfo_power / total_power)

    return np.column_stack(features).astype(np.float32)


def compute_correlation_matrix(window: np.ndarray) -> np.ndarray:
    """Compute correlation matrix for a window."""
    n_channels = window.shape[0]
    corr = np.corrcoef(window)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def aggregate_features(window_features: np.ndarray) -> np.ndarray:
    """Aggregate window features using mean, std, max."""
    return np.concatenate([
        window_features.mean(axis=0),
        window_features.std(axis=0),
        window_features.max(axis=0)
    ], axis=-1).astype(np.float32)


def process_recording(data: np.ndarray, sfreq: float, channels: list,
                      soz_channels: list) -> dict:
    """Process a single recording."""
    n_channels, n_samples = data.shape
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(WINDOW_STEP_SEC * sfreq)

    if window_samples < MIN_WINDOW_SAMPLES:
        return None

    # Create windows
    windows = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(data[:, start:start + window_samples])
        start += step_samples

    if len(windows) == 0:
        return None

    # Compute features for each window
    window_node_features = []
    correlation_windows = []

    for window in windows:
        node_feats = compute_node_features(window, sfreq)
        window_node_features.append(node_feats)
        correlation_windows.append(compute_correlation_matrix(window))

    window_node_features = np.stack(window_node_features)
    correlation_windows = np.stack(correlation_windows)

    # Aggregate features
    node_features = aggregate_features(window_node_features)
    correlation_agg = np.stack([
        correlation_windows.mean(axis=0),
        correlation_windows.std(axis=0),
        correlation_windows.max(axis=0)
    ], axis=-1)

    # Create labels
    labels = np.array([1 if ch in soz_channels else 0 for ch in channels], dtype=np.int32)

    return {
        "labels": labels,
        "window_node_features": window_node_features,
        "node_features": node_features,
        "correlation_windows": correlation_windows,
        "correlation": correlation_agg,
        "n_windows": len(windows),
    }


def main():
    print("=" * 70)
    print("Feature Extraction for Combined Dataset")
    print("=" * 70)
    print(f"Device: {DEVICE if USE_GPU else 'CPU'}")
    print(f"Input: {INPUT_H5}")
    print(f"Output: {OUTPUT_H5}")
    print("-" * 70)

    t_start = time.perf_counter()

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(INPUT_H5, "r") as f_in, h5py.File(OUTPUT_H5, "w") as f_out:
        rec_names = list(f_in.keys())
        print(f"Processing {len(rec_names)} recordings...")

        stats = {"processed": 0, "with_soz": 0, "skipped": 0}

        for rec_name in tqdm(rec_names, desc="Extracting features"):
            grp_in = f_in[rec_name]

            data = grp_in["data"][:]
            sfreq = float(grp_in.attrs["sfreq"])
            channels = list(grp_in.attrs["channels"])
            soz_channels = list(grp_in.attrs["soz_channels"])

            result = process_recording(data, sfreq, channels, soz_channels)

            if result is None:
                stats["skipped"] += 1
                continue

            # Save to output
            grp_out = f_out.create_group(rec_name)
            grp_out.create_dataset("labels", data=result["labels"])
            grp_out.create_dataset("window_node_features", data=result["window_node_features"])
            grp_out.create_dataset("node_features", data=result["node_features"])
            grp_out.create_dataset("edge_features/correlation_windows", data=result["correlation_windows"])
            grp_out.create_dataset("edge_features/correlation", data=result["correlation"])

            # Copy attributes
            grp_out.attrs["sfreq"] = sfreq
            grp_out.attrs["channels"] = channels
            grp_out.attrs["soz_channels"] = soz_channels
            grp_out.attrs["n_windows"] = result["n_windows"]

            # Copy additional attributes
            for attr in ["subject", "dataset", "engel", "outcome"]:
                if attr in grp_in.attrs:
                    grp_out.attrs[attr] = grp_in.attrs[attr]

            stats["processed"] += 1
            if len(soz_channels) > 0:
                stats["with_soz"] += 1

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed: {stats['processed']}")
    print(f"With SOZ labels: {stats['with_soz']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Output: {OUTPUT_H5}")
    print(f"Runtime: {elapsed:.1f}s")

    append_project_log(
        stage="extract_features_combined",
        status="success",
        lines=[
            f"Processed: {stats['processed']}",
            f"With SOZ: {stats['with_soz']}",
            f"Skipped: {stats['skipped']}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
