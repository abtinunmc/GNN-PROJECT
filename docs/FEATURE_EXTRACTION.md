# Feature Extraction - SOZ Localization GNN

## Overview

Extracts windowed node and edge features from CAR + bandpass filtered iEEG data.
Features are aggregated across clean time windows and saved per recording for GNN input.

## Backend

GPU-accelerated using PyTorch (falls back to CPU if unavailable).
Tested on NVIDIA RTX A4000 (16 GB VRAM, CUDA 11.4).

## Window Parameters

| Parameter | Value |
|-----------|-------|
| Window size | 12 seconds |
| Window step | 12 seconds (non-overlapping) |
| Min window samples | 1024 |

Windows are extracted only from globally valid time intervals (all channels clean simultaneously).

## Node Features (per channel)

11 features per channel per window:

| Feature | Type | Description |
|---------|------|-------------|
| delta | Band power | 1–4 Hz |
| theta | Band power | 4–8 Hz |
| alpha | Band power | 8–12 Hz |
| beta | Band power | 15–25 Hz |
| low_gamma | Band power | 35–50 Hz |
| hfo | Band power | 80–250 Hz |
| variance | Statistical | Signal variance |
| skewness | Statistical | Distribution skew |
| kurtosis | Statistical | Distribution kurtosis |
| line_length | Statistical | Mean absolute first difference |
| spectral_entropy | Spectral | Shannon entropy of PSD |

## Edge Features (per channel pair)

4 pairwise connectivity features:

| Feature | Description | Band |
|---------|-------------|------|
| correlation | Pearson correlation | Broadband |
| coherence_low_gamma | Spectral coherence | 35–50 Hz |
| coherence_hfo | Spectral coherence | 80–250 Hz |
| plv_low_gamma | Phase Locking Value | 35–50 Hz |

## Aggregation

Window-level features are aggregated across all windows using **mean**, **std**, and **max**.

- `node_features` shape: `(n_channels, 11 * 3)` = `(n_channels, 33)`
- `edge_features` shape: `(n_channels, n_channels, 3)` per edge type

## GPU Implementation

| Operation | GPU Method |
|-----------|-----------|
| Welch PSD | Batched `torch.fft.rfft` across all channels |
| Correlation | Vectorised matrix multiply |
| Coherence | Batched cross-spectral density via `torch.fft` |
| PLV | Vectorised `exp(i·Δphase)` across all channel pairs |

**Speedup:** ~200x vs CPU (43.9s vs ~2.5h for 99 recordings)

## Output HDF5 Structure

```
features.h5
├── attrs: node_feature_names, edge_feature_names, aggregation_names,
│          window_sec, window_step_sec, n_node_features
└── <recording_name>/
    ├── attrs: channels, sfreq, n_channels, n_soz, soz_channels, n_windows
    ├── labels                    # (n_channels,) binary SOZ labels
    ├── window_times              # (n_windows, 2) start/stop seconds
    ├── window_node_features      # (n_windows, n_channels, 11)
    ├── node_features             # (n_channels, 33) aggregated
    └── edge_features/
        ├── correlation           # (n_channels, n_channels, 3)
        ├── correlation_windows   # (n_windows, n_channels, n_channels)
        ├── coherence_low_gamma   # (n_channels, n_channels, 3)
        ├── coherence_hfo         # (n_channels, n_channels, 3)
        ├── plv_low_gamma         # (n_channels, n_channels, 3)
        └── ...
```

## Results Summary

| Metric | Value |
|--------|-------|
| Input recordings | 99 |
| Successfully processed | 97 |
| Skipped (no clean windows) | 2 |
| Runtime (GPU) | 43.9 seconds |
| Runtime (CPU estimate) | ~2.5 hours |

## Skipped Recordings

- `sub-jh102_ses-presurgery_task-ictal_acq-ecog_run-04_ieeg` — no clean windows
- `sub-ummc008_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg` — no clean windows

## Usage

```bash
cd ~/Desktop/GNN-PROJECT-main/scripts
python3 extract_features.py
```

## Dependencies

- numpy, scipy, h5py, tqdm
- torch (with CUDA for GPU acceleration)
