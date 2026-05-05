# SOZ Localization GNN - Complete Pipeline

## Overview

Complete preprocessing pipeline for seizure onset zone (SOZ) localization using Graph Neural Networks on OpenNeuro ds003029 (HUP iEEG) dataset.

## Status

| Stage | Script | Status | Date | Runtime |
|-------|--------|--------|------|---------|
| 1. Preprocess | `preprocess.py` | ✅ Complete | 2026-04-25 | 200s |
| 2. CAR | `apply_car.py` | ✅ Complete | 2026-04-25 | 95s |
| 3. Bandpass | `apply_bandpass.py` | ✅ Complete | 2026-04-25 | 113s |
| 4. Feature Extraction | `extract_features.py` | ✅ Complete (GPU) | 2026-04-25 | 44s |
| 5. Build Graph | `build_graph.py` | ✅ Complete (GPU) | 2026-04-25 | 2.5s |
| 6. Pretrain GNN | `pretrain_gnn.py` | ✅ Complete (GPU) | 2026-04-25 | 12s |
| 7. Train GNN | `train_gnn.py` | ✅ Complete (GPU) | 2026-04-25 | 6s |
| 8. Tune GraphSAGE | `tune_gnn.py` | ✅ Complete (GPU) | 2026-04-25 | 455s |
| 9. Train GAT | `train_gat.py` | ✅ Complete (GPU) | 2026-04-25 | 8s |
| 10. Train GIN | `train_gin.py` | ✅ Complete (GPU) | 2026-04-25 | 18s |
| 11. Tune GAT | `tune_gat.py` | ✅ Complete (GPU) | 2026-04-25 | 1062s |

## Pipeline Stages

```
Raw Data (BrainVision)
        │
        ▼
┌───────────────────┐
│  1. preprocess.py │  Notch filter (60 Hz), extract epochs  ✅
└─────────┬─────────┘
          │
          ▼
    preprocessed_ieeg.h5
          │
          ▼
┌───────────────────┐
│  2. apply_car.py  │  Common Average Reference  ✅
└─────────┬─────────┘
          │
          ▼
    preprocessed_ieeg_car.h5
          │
          ▼
┌─────────────────────┐
│  3. apply_bandpass.py │  Elliptic bandpass (1-250 Hz)  ✅
└─────────┬───────────┘
          │
          ▼
    preprocessed_ieeg_car_bp.h5
          │
          ▼
┌───────────────────────┐
│  4. extract_features.py │  Windowed node/edge features (GPU)  ✅
└─────────┬─────────────┘
          │
          ▼
    features.h5
          │
          ▼
┌───────────────────┐
│  5. build_graph.py │  PyG graph construction (GPU)  ✅
└─────────┬─────────┘
          │
          ▼
    graphs.pt (97 graphs, 29 with SOZ)
          │
          ▼
┌───────────────────┐
│  6. pretrain_gnn.py │  Self-supervised pretraining  ✅
└─────────┬─────────┘
          │
          ▼
    pretrained_encoder.pt
          │
          ▼
┌───────────────────┐
│  7. train_gnn.py  │  Fine-tune SOZ classifier  ✅
└─────────┬─────────┘
          │
          ▼
    soz_classifier.pt (Test AUC: 0.697)
          │
          ▼
┌───────────────────┐
│  8. tune_gnn.py   │  Optuna hyperparameter tuning  ✅
└─────────┬─────────┘
          │
          ▼
    Best GraphSAGE (Test AUC: 0.730)
          │
          ▼
┌───────────────────────────────────────────────────┐
│  9-11. Alternative architectures (GAT, GIN)       │
│        tune_gat.py (Test AUC: 0.702)             │  ✅
│        train_gin.py (Test AUC: 0.562)            │
└───────────────────────────────────────────────────┘
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
- Apply notch filter (60, 120, 180, 240 Hz)
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

### Preprocessing Stages

| Stage | Recordings | Runtime | Output Size |
|-------|------------|---------|-------------|
| 1. Preprocess | 99/106 | 200s | ~3.0 GB |
| 2. CAR | 99/99 | 95s | ~3.0 GB |
| 3. Bandpass | 99/99 | 113s | ~3.1 GB |
| 4. Features (GPU) | 97/99 | 44s | — |
| 5. Build Graph (GPU) | 97 graphs | 2.5s | 1.5 MB |

### GNN Training Results

| Model | Architecture | Val AUC | Test AUC | Test F1 |
|-------|--------------|---------|----------|---------|
| GraphSAGE (baseline) | 3 layers, 64 hidden | 0.920 | 0.697 | 0.478 |
| **GraphSAGE (tuned)** | 2 layers, 128 hidden | 0.983 | **0.730** | — |
| GAT (baseline) | 2 layers, 4 heads | 0.814 | 0.685 | 0.411 |
| GAT (tuned) | 3 layers, 4 heads | 0.925 | 0.702 | — |
| GIN | 3 layers, MLP 2x | 0.660 | 0.562 | 0.000 |

**Best model**: Tuned GraphSAGE with Test AUC 0.730

## Output Files

```
data/processed/
├── h5/
│   ├── preprocessed_ieeg.h5           # Stage 1: Notch filtered
│   ├── preprocessed_ieeg_car.h5       # Stage 2: CAR applied
│   ├── preprocessed_ieeg_car_bp.h5    # Stage 3: Bandpass filtered
│   └── features.h5                    # Stage 4: Windowed + aggregated features
├── graphs/
│   └── graphs.pt                      # Stage 5: PyG Data objects
├── models/
│   ├── pretrained_encoder.pt          # Stage 6: Self-supervised encoder
│   ├── pretrain_config.json           # Pretraining hyperparameters
│   ├── soz_classifier.pt              # Stage 7: Fine-tuned classifier
│   ├── best_hyperparams.json          # Stage 8: Best GraphSAGE params
│   └── best_gat_hyperparams.json      # Stage 11: Best GAT params
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
    ├── bandpass_bands.png
    ├── optuna_study.png               # GraphSAGE tuning results
    └── optuna_gat.png                 # GAT tuning results
```

## Usage

Run each stage sequentially:

```bash
cd ~/Desktop/GNN-PROJECT-main

# Stage 1-4: Preprocessing
python3 scripts/preprocess.py
python3 scripts/apply_car.py
python3 scripts/apply_bandpass.py
python3 scripts/extract_features.py

# Stage 5: Graph construction
python3 scripts/build_graph.py

# Stage 6-7: Self-supervised pretraining + fine-tuning
python3 scripts/pretrain_gnn.py
python3 scripts/train_gnn.py

# Stage 8: Hyperparameter tuning (optional)
python3 scripts/tune_gnn.py

# Alternative architectures (optional)
python3 scripts/train_gat.py
python3 scripts/train_gin.py
python3 scripts/tune_gat.py
```

## Stage 5: Graph Construction (`scripts/build_graph.py`)

**Purpose**: Convert feature matrices to PyTorch Geometric graphs

| Parameter | Value |
|-----------|-------|
| Input | `features.h5` |
| Output | `graphs/graphs.pt` |
| Correlation threshold | 0.3 |
| Z-score normalization | True |

**Graph structure**:
- Nodes: iEEG channels (electrodes)
- Node features: 33 aggregated features (mean, std, max per window feature)
- Edges: Channel pairs with correlation > threshold
- Edge weights: Correlation values
- Labels: SOZ binary mask (1 = SOZ channel, 0 = non-SOZ)

## Stage 6: Self-Supervised Pretraining (`scripts/pretrain_gnn.py`)

**Purpose**: Learn graph representations using signal forecasting

| Parameter | Value |
|-----------|-------|
| Task | Predict next window features |
| Architecture | GraphSAGE encoder |
| Hidden dim | 64 |
| Layers | 3 |
| Epochs | 350 (early stopping) |
| Learning rate | 5e-4 |

**Key insight**: Train on ALL 97 graphs (not just 29 with SOZ labels)

## Stage 7: Fine-Tuning (`scripts/train_gnn.py`)

**Purpose**: Train SOZ classifier using pretrained encoder

| Parameter | Value |
|-----------|-------|
| Encoder | Frozen for first 10 epochs |
| Classifier | 2-layer MLP |
| Loss | Class-weighted BCE |
| Train graphs | 19 |
| Val graphs | 5 |
| Test graphs | 5 |

## Stage 8: Hyperparameter Tuning (`scripts/tune_gnn.py`)

**Purpose**: Bayesian optimization with Optuna

| Parameter | Search Range |
|-----------|--------------|
| hidden_dim | [64, 128, 256] |
| num_layers | [2, 3, 4] |
| finetune_lr | [1e-5, 1e-3] |
| Trials | 50 |

**Best configuration**: hidden_dim=128, num_layers=2, finetune_lr=1.97e-4

## Documentation

- `docs/PREPROCESSING.md` - Stage 1 details
- `docs/CAR.md` - Stage 2 details
- `docs/BANDPASS.md` - Stage 3 details
- `docs/FEATURE_EXTRACTION.md` - Stage 4 details (GPU-accelerated)

## Logs

- `logs/preprocessing_log_2026-04-25.txt`
- `logs/car_log_2026-04-25.txt`
- `logs/bandpass_log_2026-04-25.txt`
- `logs/feature_extraction_log_2026-04-25.txt`
- `logs/project_log.txt` (cumulative)

## Dependencies

```
# Preprocessing
numpy
pandas
mne
h5py
scipy
matplotlib
tqdm

# GNN Training
torch==2.0.1+cu117
torch-geometric
scikit-learn
optuna
```

## Dataset Information

- **Source**: OpenNeuro ds003029 (HUP iEEG)
- **Subjects**: 35 from 4 sites (JHH, NIH, UMF, UMMC)
- **Total recordings**: 106 (103 successful)
- **Total channels**: 7,840
- **Sampling frequency**: 1000 Hz (most), 250-500 Hz (some UMMC)
- **SOZ annotations**: 30 recordings
