# SOZ Localization GNN - Complete Pipeline

## Overview

Complete preprocessing pipeline for seizure onset zone (SOZ) localization using Graph Neural Networks on OpenNeuro ds003029 (HUP iEEG) dataset.

## Pipeline Stages

```
Raw Data (BrainVision)
        │
        ▼
┌───────────────────┐
│  1. scripts/preprocess.py │  Notch filter (60 Hz), extract epochs, valid times
└─────────┬─────────┘
          │
          ▼
    preprocessed_ieeg.h5
          │
          ▼
┌───────────────────┐
│  2. scripts/apply_car.py  │  Common Average Reference
└─────────┬─────────┘
          │
          ▼
    preprocessed_ieeg_car.h5
          │
          ▼
┌───────────────────────┐
│  3. scripts/apply_bandpass.py │  Elliptic bandpass (1-250 Hz)
└─────────┬─────────────┘
          │
          ▼
    preprocessed_ieeg_car_bp.h5
          │
          ▼
┌─────────────────────────┐
│  4. scripts/extract_features.py │  Windowed node and edge features
└─────────┬───────────────┘
          │
          ▼
    features.h5
          │
          ▼
┌───────────────────┐
│  5. build_graph.py │  Graph construction (TODO)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  6. train_gnn.py  │  GNN training (TODO)
└───────────────────┘
```

## Stage 1: Preprocessing (`scripts/preprocess.py`)

**Purpose**: Load raw data, apply notch filter, extract peri-ictal epochs

| Parameter | Value |
|-----------|-------|
| Input | BrainVision .vhdr files |
| Output | `data/processed/h5/preprocessed_ieeg.h5` |
| Notch filter | 60 Hz + harmonics |
| Epoch window | +/- 30 seconds around seizure |

**Key operations**:
- Load BrainVision data via MNE
- Parse channels.tsv for good/bad channels
- Parse events.tsv for seizure onset
- Apply notch filter (60, 120, 180 Hz)
- Extract 60-second epoch around seizure
- Compute global valid times (all channels clean)

## Stage 2: CAR (`scripts/apply_car.py`)

**Purpose**: Remove common signal across all electrodes

| Parameter | Value |
|-----------|-------|
| Input | `data/processed/h5/preprocessed_ieeg.h5` |
| Output | `data/processed/h5/preprocessed_ieeg_car.h5` |
| Method | Common Average Reference |

**Formula**:
```
x_car[ch, t] = x_raw[ch, t] - mean(x_raw[:, t])
```

**Effects**:
- Removes global noise
- Reduces volume conduction
- Preserves focal (local) activity

## Stage 3: Bandpass Filter (`scripts/apply_bandpass.py`)

**Purpose**: Keep frequencies 1-250 Hz, preserve HFO band

| Parameter | Value |
|-----------|-------|
| Input | `data/processed/h5/preprocessed_ieeg_car.h5` |
| Output | `data/processed/h5/preprocessed_ieeg_car_bp.h5` |
| Filter type | Elliptic (Cauer) |
| Low cutoff | 1 Hz |
| High cutoff | 250 Hz (adjusted for Nyquist) |
| Order | 4 |
| Passband ripple | 0.1 dB |
| Stopband attenuation | 40 dB |

**Key features**:
- Steepest roll-off filter type
- Zero-phase filtering (sosfiltfilt)
- Automatic Nyquist adjustment

## Stage 4: Feature Extraction (`scripts/extract_features.py`)

**Purpose**: Compute clean-window features and aggregate them per recording

**Node features (per channel)**:
- Band powers: delta, theta, alpha, beta, low_gamma, hfo
- Statistical: variance, skewness, kurtosis, line_length
- Spectral entropy

**Edge features (channel pairs)**:
- Correlation matrix
- Coherence (low_gamma, hfo bands)
- Phase Locking Value (PLV) in low_gamma

## Results Summary

| Stage | Recordings | Runtime | Output Size |
|-------|------------|---------|-------------|
| 1. Preprocess | 103/106 | ~130s | 3.0 GB |
| 2. CAR | 103/103 | ~80s | 3.0 GB |
| 3. Bandpass | 103/103 | ~82s | 3.1 GB |
| 4. Features | - | - | - |

## Output Files

```
data/processed/
├── h5/
│   ├── preprocessed_ieeg.h5           # Stage 1: Notch filtered
│   ├── preprocessed_ieeg_car.h5       # Stage 2: CAR applied
│   ├── preprocessed_ieeg_car_bp.h5    # Stage 3: Bandpass filtered
│   └── features.h5                    # Stage 4: Windowed + aggregated features
├── csv/
│   ├── metadata.csv                   # Recording metadata
│   └── failed_recordings.csv          # QC failures
└── figures/
    ├── preprocessing_demo.png         # Notch filter visualization
    ├── preprocessing_multichannel.png
    ├── car_demo.png                   # CAR visualization
    ├── car_multichannel.png
    ├── car_statistics.png
    ├── bandpass_filter_response.png   # Bandpass visualization
    ├── bandpass_demo.png
    └── bandpass_bands.png
```

## Usage

Run each stage sequentially:

```bash
cd ~/Documents/gnn\ project

# Stage 1: Preprocessing
python3 scripts/preprocess.py

# Stage 2: CAR
python3 scripts/apply_car.py

# Stage 3: Bandpass
python3 scripts/apply_bandpass.py

# Stage 4: Feature extraction
python3 scripts/extract_features.py
```

## Documentation

- `docs/PREPROCESSING.md` - Stage 1 details
- `docs/CAR.md` - Stage 2 details
- `docs/BANDPASS.md` - Stage 3 details

## Logs

- `logs/preprocessing_log_2026-04-21.txt`
- `logs/car_log_2026-04-21.txt`
- `logs/bandpass_log_2026-04-21.txt`

## Dependencies

```
numpy
pandas
mne
h5py
scipy
matplotlib
tqdm
```

## Dataset Information

- **Source**: OpenNeuro ds003029 (HUP iEEG)
- **Subjects**: 35 from 4 sites (JHH, NIH, UMF, UMMC)
- **Total recordings**: 106 (103 successful)
- **Total channels**: 7,840
- **Sampling frequency**: 1000 Hz (most), 250-500 Hz (some UMMC)
- **SOZ annotations**: 30 recordings
