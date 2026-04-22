"""
Windowed feature extraction for the SOZ localization GNN pipeline.

This stage reads CAR + bandpass filtered iEEG data, extracts features from
clean time windows, and aggregates those window-level features per recording.

Output structure per recording:
- labels: binary SOZ labels per channel
- window_node_features: (n_windows, n_channels, n_node_features)
- node_features: aggregated node features, (n_channels, n_node_features * n_aggs)
- edge_features/<name>_windows: (n_windows, n_channels, n_channels)
- edge_features/<name>: aggregated edge features, (n_channels, n_channels, n_aggs)
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from log_utils import append_project_log


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg_car_bp.h5"
OUTPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"

# Use 12-second non-overlapping windows to match recent interictal SOZ
# connectivity work more closely, while keeping enough data per window for
# stable correlation/coherence/PLV estimates.
WINDOW_SEC = 12.0
WINDOW_STEP_SEC = 12.0

# Guard against accidentally using windows that are too short for spectral and
# connectivity features, especially on lower-sampling-rate recordings.
MIN_WINDOW_SAMPLES = 1024

# Frequency bands are chosen to stay close to SOZ/interictal connectivity work
# while keeping one broad HFO band that matches the current preprocessing.
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (15, 25),
    "low_gamma": (35, 50),
    "hfo": (80, 250),
}

NODE_FEATURE_NAMES = list(FREQUENCY_BANDS.keys()) + [
    "variance",
    "skewness",
    "kurtosis",
    "line_length",
    "spectral_entropy",
]

EDGE_FEATURE_NAMES = [
    "correlation",
    "coherence_low_gamma",
    "coherence_hfo",
    "plv_low_gamma",
]

# Aggregations convert window-level features into one recording-level summary.
AGGREGATIONS = ["mean", "std", "max"]
EPS = 1e-10


def compute_band_power(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Compute Welch band power per channel."""
    # Estimate power spectral density and integrate within the requested band.
    n_channels, n_samples = data.shape
    low_freq, high_freq = band
    nyquist = sfreq / 2

    if low_freq >= nyquist:
        return np.zeros(n_channels)
    if high_freq > nyquist:
        high_freq = nyquist - 1e-6

    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return np.zeros(n_channels)

    band_power = np.zeros(n_channels)
    for ch in range(n_channels):
        freqs, psd = signal.welch(data[ch, :], fs=sfreq, nperseg=nperseg)
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if np.any(band_mask):
            band_power[ch] = np.trapz(psd[band_mask], freqs[band_mask])
    return band_power


def compute_line_length(data: np.ndarray) -> np.ndarray:
    """Compute normalized line length per channel."""
    # Capture signal roughness with the average absolute first difference.
    diffs = np.abs(np.diff(data, axis=1))
    line_length = np.sum(diffs, axis=1)
    if data.shape[1] > 1:
        line_length = line_length / (data.shape[1] - 1)
    return line_length


def compute_spectral_entropy(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute spectral entropy per channel."""
    # Measure how concentrated or diffuse each channel's spectrum is.
    n_channels, n_samples = data.shape
    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return np.zeros(n_channels)

    entropy = np.zeros(n_channels)
    for ch in range(n_channels):
        _, psd = signal.welch(data[ch, :], fs=sfreq, nperseg=nperseg)
        psd_norm = psd / (np.sum(psd) + EPS)
        entropy[ch] = -np.sum(psd_norm * np.log2(psd_norm + EPS))
    return entropy


def compute_node_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute node features for one window."""
    # Build one feature vector per channel from spectral and statistical summaries.
    features = []
    for _, band_range in FREQUENCY_BANDS.items():
        features.append(compute_band_power(data, sfreq, band_range))

    features.append(np.var(data, axis=1))
    features.append(skew(data, axis=1))
    features.append(kurtosis(data, axis=1, fisher=True))
    features.append(compute_line_length(data))
    features.append(compute_spectral_entropy(data, sfreq))

    out = np.column_stack(features)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix."""
    # Use linear correlation as a simple functional-connectivity baseline.
    corr_matrix = np.corrcoef(data)
    return np.nan_to_num(corr_matrix, nan=0.0)


def compute_coherence_matrix(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Compute coherence matrix averaged within a band."""
    # Estimate frequency-specific coupling for every channel pair.
    n_channels, n_samples = data.shape
    low_freq, high_freq = band
    nyquist = sfreq / 2

    if low_freq >= nyquist:
        return np.zeros((n_channels, n_channels))
    if high_freq > nyquist:
        high_freq = nyquist - 1e-6

    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return np.zeros((n_channels, n_channels))

    coh_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = signal.coherence(data[i, :], data[j, :], fs=sfreq, nperseg=nperseg)
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                coh_matrix[i, j] = np.mean(coh[band_mask])
                coh_matrix[j, i] = coh_matrix[i, j]
    np.fill_diagonal(coh_matrix, 1.0)
    return coh_matrix


def compute_plv_matrix(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Compute PLV matrix for one band."""
    # Filter into one band, extract phase, then summarize phase synchrony by PLV.
    n_channels, n_samples = data.shape
    low_freq, high_freq = band
    nyquist = sfreq / 2

    if low_freq >= nyquist or high_freq <= low_freq:
        return np.zeros((n_channels, n_channels))
    if high_freq > nyquist - 1:
        high_freq = nyquist - 1

    try:
        sos = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], btype="bandpass", output="sos")
    except ValueError:
        return np.zeros((n_channels, n_channels))

    phases = np.zeros_like(data)
    for ch in range(n_channels):
        try:
            filtered = signal.sosfiltfilt(sos, data[ch, :])
            phases[ch, :] = np.angle(hilbert(filtered))
        except ValueError:
            phases[ch, :] = 0

    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i, :] - phases[j, :]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    np.fill_diagonal(plv_matrix, 1.0)
    return plv_matrix


def compute_edge_features(data: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Compute edge features for one window."""
    # Assemble all pairwise connectivity features for one analysis window.
    return {
        "correlation": compute_correlation_matrix(data),
        "coherence_low_gamma": compute_coherence_matrix(data, sfreq, FREQUENCY_BANDS["low_gamma"]),
        "coherence_hfo": compute_coherence_matrix(data, sfreq, FREQUENCY_BANDS["hfo"]),
        "plv_low_gamma": compute_plv_matrix(data, sfreq, FREQUENCY_BANDS["low_gamma"]),
    }


def extract_labels(soz_channels: list, all_channels: list) -> np.ndarray:
    """Create binary SOZ labels per channel."""
    # Convert channel names into a channel-aligned binary supervision vector.
    labels = np.zeros(len(all_channels), dtype=np.int32)
    for i, ch in enumerate(all_channels):
        if ch in soz_channels:
            labels[i] = 1
    return labels


def get_clean_windows(
    data: np.ndarray,
    sfreq: float,
    global_valid_times: np.ndarray | None,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Split clean valid intervals into overlapping analysis windows."""
    # Turn valid-time intervals into overlapping fixed-length windows for features.
    n_samples = data.shape[1]
    window_samples = int(round(WINDOW_SEC * sfreq))
    step_samples = int(round(WINDOW_STEP_SEC * sfreq))

    if window_samples < MIN_WINDOW_SAMPLES or step_samples <= 0:
        return [], []

    if global_valid_times is None or len(global_valid_times) == 0:
        intervals = np.array([[0.0, n_samples / sfreq]])
    else:
        intervals = global_valid_times

    windows = []
    window_times = []
    for start_sec, end_sec in intervals:
        start_idx = max(0, int(round(start_sec * sfreq)))
        end_idx = min(n_samples, int(round(end_sec * sfreq)))
        if end_idx - start_idx < window_samples:
            continue

        for win_start in range(start_idx, end_idx - window_samples + 1, step_samples):
            win_end = win_start + window_samples
            windows.append(data[:, win_start:win_end])
            window_times.append((win_start / sfreq, win_end / sfreq))

    return windows, window_times


def aggregate_window_features(features: np.ndarray) -> np.ndarray:
    """Aggregate windowed features across the window axis."""
    # Summarize each channel feature across windows with simple descriptive stats.
    if features.shape[0] == 0:
        raise ValueError("Cannot aggregate zero windows")

    aggs = [
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.max(features, axis=0),
    ]
    return np.concatenate(aggs, axis=-1)


def aggregate_window_edge_features(features: np.ndarray) -> np.ndarray:
    """Aggregate edge features across the window axis."""
    # Summarize each pairwise connectivity feature across windows.
    aggs = [
        np.mean(features, axis=0),
        np.std(features, axis=0),
        np.max(features, axis=0),
    ]
    return np.stack(aggs, axis=-1)


def process_all_recordings() -> int:
    """Extract windowed and aggregated features for all recordings."""
    print("=" * 70)
    print("Extracting Windowed Features for GNN")
    print("=" * 70)
    print(f"Window size: {WINDOW_SEC:.1f}s, step: {WINDOW_STEP_SEC:.1f}s")
    print(f"Input:  {INPUT_H5}")
    print(f"Output: {OUTPUT_H5}")

    if not INPUT_H5.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_H5}")

    processed_recordings = 0
    skipped_recordings: list[tuple[str, str]] = []

    with h5py.File(INPUT_H5, "r") as f_in, h5py.File(OUTPUT_H5, "w") as f_out:
        f_out.attrs["node_feature_names"] = NODE_FEATURE_NAMES
        f_out.attrs["edge_feature_names"] = EDGE_FEATURE_NAMES
        f_out.attrs["aggregation_names"] = AGGREGATIONS
        f_out.attrs["window_sec"] = WINDOW_SEC
        f_out.attrs["window_step_sec"] = WINDOW_STEP_SEC
        f_out.attrs["n_node_features"] = len(NODE_FEATURE_NAMES)

        for rec_name in tqdm(list(f_in.keys()), desc="Extracting features"):
            # Step 1: load one recording and its metadata from the bandpass HDF5.
            grp_in = f_in[rec_name]
            data = grp_in["data"][:]
            sfreq = float(grp_in.attrs["sfreq"])
            channels = list(grp_in.attrs["channels"])
            soz_channels = list(grp_in.attrs["soz_channels"])
            global_valid_times = grp_in["global_valid_times"][:] if "global_valid_times" in grp_in else None

            # Step 2: generate overlapping clean windows from valid-time intervals.
            windows, window_times = get_clean_windows(data, sfreq, global_valid_times)
            if not windows:
                skipped_recordings.append((rec_name, "No clean windows after valid-time windowing"))
                continue

            window_node_features = []
            edge_windows: dict[str, list[np.ndarray]] = {name: [] for name in EDGE_FEATURE_NAMES}

            # Step 3: compute node and edge features independently for each window.
            for window_data in windows:
                window_node_features.append(compute_node_features(window_data, sfreq))
                edge_features = compute_edge_features(window_data, sfreq)
                for name, matrix in edge_features.items():
                    edge_windows[name].append(matrix)

            # Step 4: aggregate window-level features into one per-recording summary.
            window_node_features_arr = np.stack(window_node_features, axis=0)
            aggregated_node_features = aggregate_window_features(window_node_features_arr)

            # Step 5: build channel labels aligned with the recording's channel order.
            labels = extract_labels(soz_channels, channels)
            n_soz = int(np.sum(labels))

            # Step 6: save both window-level and aggregated features to the output HDF5.
            grp_out = f_out.create_group(rec_name)
            grp_out.create_dataset("window_node_features", data=window_node_features_arr, compression="gzip")
            grp_out.create_dataset("node_features", data=aggregated_node_features, compression="gzip")
            grp_out.create_dataset("labels", data=labels)
            grp_out.create_dataset("window_times", data=np.array(window_times, dtype=np.float32))

            edge_grp = grp_out.create_group("edge_features")
            for name in EDGE_FEATURE_NAMES:
                windows_arr = np.stack(edge_windows[name], axis=0)
                edge_grp.create_dataset(f"{name}_windows", data=windows_arr, compression="gzip")
                edge_grp.create_dataset(name, data=aggregate_window_edge_features(windows_arr), compression="gzip")

            grp_out.attrs["channels"] = channels
            grp_out.attrs["sfreq"] = sfreq
            grp_out.attrs["n_channels"] = data.shape[0]
            grp_out.attrs["n_soz"] = n_soz
            grp_out.attrs["soz_channels"] = soz_channels
            grp_out.attrs["n_windows"] = len(windows)

            processed_recordings += 1

    print("-" * 70)
    print(f"Successfully processed {processed_recordings} recordings")
    print(f"Skipped: {len(skipped_recordings)} recordings")
    if skipped_recordings:
        print("Skipped recordings:")
        for rec_name, reason in skipped_recordings:
            print(f"  - {rec_name}: {reason}")
    print(f"Output saved to: {OUTPUT_H5}")

    return processed_recordings


def verify_features() -> None:
    """Print a compact verification summary."""
    print("\n" + "=" * 70)
    print("Verifying Extracted Features")
    print("=" * 70)

    if not OUTPUT_H5.exists():
        print("Output file not found")
        return

    with h5py.File(OUTPUT_H5, "r") as f:
        if len(f.keys()) == 0:
            print("No recordings present")
            return

        rec_name = list(f.keys())[0]
        grp = f[rec_name]
        print(f"Sample recording: {rec_name}")
        print(f"  windows: {grp.attrs['n_windows']}")
        print(f"  window_node_features: {grp['window_node_features'].shape}")
        print(f"  node_features: {grp['node_features'].shape}")
        print(f"  labels: {grp['labels'].shape}")
        for edge_name in EDGE_FEATURE_NAMES:
            print(f"  edge {edge_name}: {grp['edge_features'][edge_name].shape}")


def main() -> None:
    """Main entry point."""
    t_start = time.perf_counter()
    n_processed = process_all_recordings()
    if n_processed > 0:
        verify_features()
    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("Feature extraction complete!")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    append_project_log(
        stage="extract_features",
        status="success",
        lines=[
            f"Input: {INPUT_H5}",
            f"Output: {OUTPUT_H5}",
            f"Window size (s): {WINDOW_SEC:.1f}",
            f"Window step (s): {WINDOW_STEP_SEC:.1f}",
            f"Processed recordings: {n_processed}",
            f"Node features: {', '.join(NODE_FEATURE_NAMES)}",
            f"Edge features: {', '.join(EDGE_FEATURE_NAMES)}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
