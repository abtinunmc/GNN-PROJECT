"""
Windowed feature extraction for the SOZ localization GNN pipeline.
GPU-accelerated version using PyTorch (falls back to CPU if unavailable).

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
INPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg_car_bp.h5"
OUTPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "features.h5"

WINDOW_SEC = 12.0
WINDOW_STEP_SEC = 12.0
MIN_WINDOW_SAMPLES = 1024

FREQUENCY_BANDS = {
    "delta":     (1,   4),
    "theta":     (4,   8),
    "alpha":     (8,  12),
    "beta":      (15, 25),
    "low_gamma": (35, 50),
    "hfo":       (80, 250),
}

NODE_FEATURE_NAMES = list(FREQUENCY_BANDS.keys()) + [
    "variance", "skewness", "kurtosis", "line_length", "spectral_entropy",
]
EDGE_FEATURE_NAMES = [
    "correlation", "coherence_low_gamma", "coherence_hfo", "plv_low_gamma",
]
AGGREGATIONS = ["mean", "std", "max"]
EPS = 1e-10


# ============================================================================
# GPU UTILITIES
# ============================================================================

def to_tensor(arr: np.ndarray) -> "torch.Tensor":
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)


def welch_gpu(data_t: "torch.Tensor", sfreq: float, nperseg: int) -> tuple:
    """Batch Welch PSD on GPU. data_t: (n_channels, n_samples)."""
    n_channels, n_samples = data_t.shape
    step = nperseg // 2
    hann = torch.hann_window(nperseg, device=DEVICE)

    # Build segment indices
    starts = list(range(0, n_samples - nperseg + 1, step))
    if not starts:
        freqs = torch.linspace(0, sfreq / 2, nperseg // 2 + 1, device=DEVICE)
        return freqs, torch.zeros(n_channels, nperseg // 2 + 1, device=DEVICE)

    # Stack segments: (n_channels, n_segs, nperseg)
    segs = torch.stack([data_t[:, s:s + nperseg] for s in starts], dim=1)
    segs = segs * hann.unsqueeze(0).unsqueeze(0)

    # FFT → power spectrum
    spec = torch.fft.rfft(segs, dim=-1)
    power = (spec.real ** 2 + spec.imag ** 2) / (sfreq * (hann ** 2).sum())
    power[:, :, 1:-1] *= 2  # one-sided correction

    psd = power.mean(dim=1)  # (n_channels, n_freqs)
    freqs = torch.linspace(0, sfreq / 2, psd.shape[-1], device=DEVICE)
    return freqs, psd


def band_power_gpu(data_t: "torch.Tensor", sfreq: float,
                   bands: dict) -> "torch.Tensor":
    """Compute band powers for all bands at once. Returns (n_channels, n_bands)."""
    n_channels, n_samples = data_t.shape
    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return torch.zeros(n_channels, len(bands), device=DEVICE)

    freqs, psd = welch_gpu(data_t, sfreq, nperseg)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    nyquist = sfreq / 2

    out = []
    for _, (lo, hi) in bands.items():
        if lo >= nyquist:
            out.append(torch.zeros(n_channels, device=DEVICE))
            continue
        hi = min(hi, nyquist - 1e-6)
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            out.append((psd[:, mask] * df).sum(dim=-1))
        else:
            out.append(torch.zeros(n_channels, device=DEVICE))

    return torch.stack(out, dim=-1)  # (n_channels, n_bands)


def spectral_entropy_gpu(data_t: "torch.Tensor", sfreq: float) -> "torch.Tensor":
    """Spectral entropy per channel."""
    n_channels, n_samples = data_t.shape
    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return torch.zeros(n_channels, device=DEVICE)

    _, psd = welch_gpu(data_t, sfreq, nperseg)
    psd_norm = psd / (psd.sum(dim=-1, keepdim=True) + EPS)
    entropy = -(psd_norm * torch.log2(psd_norm + EPS)).sum(dim=-1)
    return entropy


def correlation_matrix_gpu(data_t: "torch.Tensor") -> "torch.Tensor":
    """Pearson correlation matrix. data_t: (n_channels, n_samples)."""
    x = data_t - data_t.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=EPS)
    x_norm = x / std
    n = x_norm.shape[-1]
    corr = (x_norm @ x_norm.T) / (n - 1)
    corr = corr.clamp(-1.0, 1.0)
    corr.fill_diagonal_(1.0)
    return torch.nan_to_num(corr, nan=0.0)


def coherence_matrix_gpu(data_t: "torch.Tensor", sfreq: float,
                          band: tuple, nperseg_sec: float = 2.0) -> "torch.Tensor":
    """Mean coherence matrix for a frequency band."""
    n_channels, n_samples = data_t.shape
    lo, hi = band
    nyquist = sfreq / 2
    if lo >= nyquist:
        return torch.zeros(n_channels, n_channels, device=DEVICE)
    hi = min(hi, nyquist - 1e-6)

    nperseg = min(n_samples, int(sfreq * nperseg_sec))
    if nperseg < 4:
        return torch.zeros(n_channels, n_channels, device=DEVICE)

    step = nperseg // 2
    hann = torch.hann_window(nperseg, device=DEVICE)
    starts = list(range(0, n_samples - nperseg + 1, step))
    if not starts:
        return torch.zeros(n_channels, n_channels, device=DEVICE)

    # Stack segments: (n_channels, n_segs, nperseg)
    segs = torch.stack([data_t[:, s:s + nperseg] for s in starts], dim=1)
    segs = segs * hann.unsqueeze(0).unsqueeze(0)
    spec = torch.fft.rfft(segs, dim=-1)  # (n_ch, n_segs, n_freqs)

    freqs = torch.linspace(0, nyquist, spec.shape[-1], device=DEVICE)
    mask = (freqs >= lo) & (freqs <= hi)

    spec_band = spec[:, :, mask]  # (n_ch, n_segs, n_band_freqs)

    # Cross-spectral density: Sxy = mean over segs of conj(X) * Y
    # shape tricks: (n_ch, 1, n_segs, f) × (1, n_ch, n_segs, f)
    X = spec_band.unsqueeze(1)   # (n_ch, 1, n_segs, f)
    Y = spec_band.unsqueeze(0)   # (1, n_ch, n_segs, f)

    Sxy = (X.conj() * Y).mean(dim=2)          # (n_ch, n_ch, f)
    Sxx = (X.abs() ** 2).mean(dim=2)           # (n_ch, 1, f)
    Syy = (Y.abs() ** 2).mean(dim=2)           # (1, n_ch, f)

    coh = (Sxy.abs() ** 2) / (Sxx * Syy + EPS)
    coh_mean = coh.mean(dim=-1).clamp(0.0, 1.0)
    coh_mean.fill_diagonal_(1.0)
    return torch.nan_to_num(coh_mean, nan=0.0)


def hilbert_gpu(x: "torch.Tensor") -> "torch.Tensor":
    """Analytic signal via FFT (Hilbert transform) on GPU."""
    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return torch.fft.ifft(X * h, dim=-1)


def plv_matrix_gpu(data_t: "torch.Tensor", sfreq: float,
                   band: tuple) -> "torch.Tensor":
    """Phase Locking Value matrix for one band."""
    n_channels, n_samples = data_t.shape
    lo, hi = band
    nyquist = sfreq / 2
    if lo >= nyquist or hi <= lo:
        return torch.zeros(n_channels, n_channels, device=DEVICE)
    hi = min(hi, nyquist - 1)

    # Bandpass on CPU (scipy), then back to GPU for phase/PLV
    data_np = data_t.cpu().numpy()
    try:
        sos = scipy_signal.butter(
            4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos"
        )
        filtered = np.zeros_like(data_np)
        for ch in range(n_channels):
            filtered[ch] = scipy_signal.sosfiltfilt(sos, data_np[ch])
    except Exception:
        return torch.zeros(n_channels, n_channels, device=DEVICE)

    filtered_t = to_tensor(filtered)
    analytic = hilbert_gpu(filtered_t)
    phases = torch.angle(analytic)  # (n_channels, n_samples)

    # PLV: |mean(exp(i * phase_diff))| for all pairs — vectorised
    # phase_i - phase_j for all i,j: (n_ch, n_ch, n_samples)
    ph_i = phases.unsqueeze(1)  # (n_ch, 1, n_samples)
    ph_j = phases.unsqueeze(0)  # (1, n_ch, n_samples)
    phase_diff = ph_i - ph_j

    plv = torch.abs(torch.mean(torch.exp(1j * phase_diff.to(torch.complex64)), dim=-1))
    plv = plv.real.clamp(0.0, 1.0)
    plv.fill_diagonal_(1.0)
    return torch.nan_to_num(plv, nan=0.0)


# ============================================================================
# NODE & EDGE FEATURES (GPU)
# ============================================================================

def compute_node_features_gpu(data_t: "torch.Tensor", sfreq: float) -> np.ndarray:
    """GPU node features. Returns (n_channels, n_node_features) numpy array."""
    bp = band_power_gpu(data_t, sfreq, FREQUENCY_BANDS)         # (C, 6)
    var = data_t.var(dim=-1, keepdim=True)                       # (C, 1)
    data_np = data_t.cpu().numpy()
    sk = torch.as_tensor(skew(data_np, axis=1), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    kt = torch.as_tensor(kurtosis(data_np, axis=1, fisher=True), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    ll = (data_t[:, 1:] - data_t[:, :-1]).abs().sum(dim=-1, keepdim=True) / max(data_t.shape[-1] - 1, 1)
    se = spectral_entropy_gpu(data_t, sfreq).unsqueeze(-1)       # (C, 1)

    features = torch.cat([bp, var, sk, kt, ll, se], dim=-1)     # (C, 11)
    out = features.cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_edge_features_gpu(data_t: "torch.Tensor", sfreq: float) -> dict:
    return {
        "correlation":        correlation_matrix_gpu(data_t).cpu().numpy(),
        "coherence_low_gamma": coherence_matrix_gpu(data_t, sfreq, FREQUENCY_BANDS["low_gamma"]).cpu().numpy(),
        "coherence_hfo":      coherence_matrix_gpu(data_t, sfreq, FREQUENCY_BANDS["hfo"]).cpu().numpy(),
        "plv_low_gamma":      plv_matrix_gpu(data_t, sfreq, FREQUENCY_BANDS["low_gamma"]).cpu().numpy(),
    }


# ============================================================================
# CPU FALLBACK (original implementation)
# ============================================================================

def compute_band_power_cpu(data: np.ndarray, sfreq: float, band: tuple) -> np.ndarray:
    n_channels, n_samples = data.shape
    lo, hi = band
    nyquist = sfreq / 2
    if lo >= nyquist:
        return np.zeros(n_channels)
    hi = min(hi, nyquist - 1e-6)
    nperseg = min(n_samples, int(sfreq * 2))
    if nperseg < 4:
        return np.zeros(n_channels)
    out = np.zeros(n_channels)
    for ch in range(n_channels):
        freqs, psd = scipy_signal.welch(data[ch], fs=sfreq, nperseg=nperseg)
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            out[ch] = np.trapz(psd[mask], freqs[mask])
    return out


def compute_node_features_cpu(data: np.ndarray, sfreq: float) -> np.ndarray:
    feats = [compute_band_power_cpu(data, sfreq, b) for b in FREQUENCY_BANDS.values()]
    feats.append(np.var(data, axis=1))
    feats.append(skew(data, axis=1))
    feats.append(kurtosis(data, axis=1, fisher=True))
    diffs = np.abs(np.diff(data, axis=1))
    feats.append(diffs.sum(axis=1) / max(data.shape[1] - 1, 1))
    nperseg = min(data.shape[1], int(sfreq * 2))
    entropy = np.zeros(data.shape[0])
    if nperseg >= 4:
        for ch in range(data.shape[0]):
            _, psd = scipy_signal.welch(data[ch], fs=sfreq, nperseg=nperseg)
            pn = psd / (psd.sum() + EPS)
            entropy[ch] = -(pn * np.log2(pn + EPS)).sum()
    feats.append(entropy)
    return np.nan_to_num(np.column_stack(feats), nan=0.0, posinf=0.0, neginf=0.0)


def compute_edge_features_cpu(data: np.ndarray, sfreq: float) -> dict:
    n_ch = data.shape[0]
    corr = np.nan_to_num(np.corrcoef(data), nan=0.0)

    def coh_matrix(band):
        lo, hi = band
        nyquist = sfreq / 2
        if lo >= nyquist:
            return np.zeros((n_ch, n_ch))
        hi = min(hi, nyquist - 1e-6)
        nperseg = min(data.shape[1], int(sfreq * 2))
        mat = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                freqs, coh = scipy_signal.coherence(data[i], data[j], fs=sfreq, nperseg=nperseg)
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    mat[i, j] = mat[j, i] = np.mean(coh[mask])
        np.fill_diagonal(mat, 1.0)
        return mat

    def plv_matrix(band):
        lo, hi = band
        nyquist = sfreq / 2
        if lo >= nyquist:
            return np.zeros((n_ch, n_ch))
        hi = min(hi, nyquist - 1)
        try:
            sos = scipy_signal.butter(4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos")
        except ValueError:
            return np.zeros((n_ch, n_ch))
        phases = np.zeros_like(data)
        for ch in range(n_ch):
            try:
                phases[ch] = np.angle(scipy_signal.hilbert(scipy_signal.sosfiltfilt(sos, data[ch])))
            except ValueError:
                pass
        mat = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                plv = np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
                mat[i, j] = mat[j, i] = plv
        np.fill_diagonal(mat, 1.0)
        return mat

    return {
        "correlation":         corr,
        "coherence_low_gamma": coh_matrix(FREQUENCY_BANDS["low_gamma"]),
        "coherence_hfo":       coh_matrix(FREQUENCY_BANDS["hfo"]),
        "plv_low_gamma":       plv_matrix(FREQUENCY_BANDS["low_gamma"]),
    }


# ============================================================================
# WINDOWING & AGGREGATION
# ============================================================================

def get_clean_windows(data: np.ndarray, sfreq: float,
                      global_valid_times) -> tuple[list, list]:
    n_samples = data.shape[1]
    window_samples = int(round(WINDOW_SEC * sfreq))
    step_samples = int(round(WINDOW_STEP_SEC * sfreq))

    if window_samples < MIN_WINDOW_SAMPLES or step_samples <= 0:
        return [], []

    intervals = (np.array([[0.0, n_samples / sfreq]])
                 if global_valid_times is None or len(global_valid_times) == 0
                 else global_valid_times)

    windows, window_times = [], []
    for start_sec, end_sec in intervals:
        start_idx = max(0, int(round(start_sec * sfreq)))
        end_idx = min(n_samples, int(round(end_sec * sfreq)))
        if end_idx - start_idx < window_samples:
            continue
        for ws in range(start_idx, end_idx - window_samples + 1, step_samples):
            windows.append(data[:, ws:ws + window_samples])
            window_times.append((ws / sfreq, (ws + window_samples) / sfreq))

    return windows, window_times


def aggregate_window_features(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features.mean(0), features.std(0), features.max(0)], axis=-1)


def aggregate_window_edge_features(features: np.ndarray) -> np.ndarray:
    return np.stack([features.mean(0), features.std(0), features.max(0)], axis=-1)


def extract_labels(soz_channels: list, all_channels: list) -> np.ndarray:
    labels = np.zeros(len(all_channels), dtype=np.int32)
    for i, ch in enumerate(all_channels):
        if ch in soz_channels:
            labels[i] = 1
    return labels


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_all_recordings() -> int:
    print("=" * 70)
    print("Extracting Windowed Features for GNN")
    print(f"Backend: {'GPU (' + torch.cuda.get_device_name(0) + ')' if USE_GPU else 'CPU'}")
    print("=" * 70)
    print(f"Window size: {WINDOW_SEC:.1f}s, step: {WINDOW_STEP_SEC:.1f}s")
    print(f"Input:  {INPUT_H5}")
    print(f"Output: {OUTPUT_H5}")

    if not INPUT_H5.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_H5}")

    processed = 0
    skipped: list[tuple[str, str]] = []

    with h5py.File(INPUT_H5, "r") as f_in, h5py.File(OUTPUT_H5, "w") as f_out:
        f_out.attrs["node_feature_names"] = NODE_FEATURE_NAMES
        f_out.attrs["edge_feature_names"] = EDGE_FEATURE_NAMES
        f_out.attrs["aggregation_names"] = AGGREGATIONS
        f_out.attrs["window_sec"] = WINDOW_SEC
        f_out.attrs["window_step_sec"] = WINDOW_STEP_SEC
        f_out.attrs["n_node_features"] = len(NODE_FEATURE_NAMES)

        for rec_name in tqdm(list(f_in.keys()), desc="Extracting features"):
            grp_in = f_in[rec_name]
            data = grp_in["data"][:]
            sfreq = float(grp_in.attrs["sfreq"])
            channels = list(grp_in.attrs["channels"])
            soz_channels = list(grp_in.attrs["soz_channels"])
            gvt = grp_in["global_valid_times"][:] if "global_valid_times" in grp_in else None

            windows, window_times = get_clean_windows(data, sfreq, gvt)
            if not windows:
                skipped.append((rec_name, "No clean windows"))
                continue

            window_node_features = []
            edge_windows: dict[str, list] = {n: [] for n in EDGE_FEATURE_NAMES}

            for window_data in windows:
                if USE_GPU:
                    data_t = to_tensor(window_data)
                    window_node_features.append(compute_node_features_gpu(data_t, sfreq))
                    ef = compute_edge_features_gpu(data_t, sfreq)
                else:
                    window_node_features.append(compute_node_features_cpu(window_data, sfreq))
                    ef = compute_edge_features_cpu(window_data, sfreq)

                for name, mat in ef.items():
                    edge_windows[name].append(mat)

            window_node_arr = np.stack(window_node_features, axis=0)
            agg_node = aggregate_window_features(window_node_arr)
            labels = extract_labels(soz_channels, channels)

            grp_out = f_out.create_group(rec_name)
            grp_out.create_dataset("window_node_features", data=window_node_arr, compression="gzip")
            grp_out.create_dataset("node_features", data=agg_node, compression="gzip")
            grp_out.create_dataset("labels", data=labels)
            grp_out.create_dataset("window_times", data=np.array(window_times, dtype=np.float32))

            edge_grp = grp_out.create_group("edge_features")
            for name in EDGE_FEATURE_NAMES:
                arr = np.stack(edge_windows[name], axis=0)
                edge_grp.create_dataset(f"{name}_windows", data=arr, compression="gzip")
                edge_grp.create_dataset(name, data=aggregate_window_edge_features(arr), compression="gzip")

            grp_out.attrs["channels"] = channels
            grp_out.attrs["sfreq"] = sfreq
            grp_out.attrs["n_channels"] = data.shape[0]
            grp_out.attrs["n_soz"] = int(labels.sum())
            grp_out.attrs["soz_channels"] = soz_channels
            grp_out.attrs["n_windows"] = len(windows)

            processed += 1

    print("-" * 70)
    print(f"Successfully processed {processed} recordings")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for r, reason in skipped:
            print(f"  - {r}: {reason}")
    print(f"Output saved to: {OUTPUT_H5}")
    return processed


def verify_features() -> None:
    print("\n" + "=" * 70)
    print("Verifying Extracted Features")
    print("=" * 70)
    if not OUTPUT_H5.exists():
        print("Output file not found")
        return
    with h5py.File(OUTPUT_H5, "r") as f:
        if not f.keys():
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
            f"Backend: {'GPU' if USE_GPU else 'CPU'}",
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
