# Scripts Reference

## Original Pipeline (ds003029 only)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `preprocess.py` | Load BrainVision, notch filter, extract epochs | ds003029 raw | `h5/preprocessed_ieeg.h5` |
| `apply_car.py` | Common Average Reference | preprocessed_ieeg.h5 | preprocessed_ieeg_car.h5 |
| `apply_bandpass.py` | Bandpass filter 1-250 Hz | preprocessed_ieeg_car.h5 | preprocessed_ieeg_car_bp.h5 |
| `extract_features.py` | Window features + edge features | preprocessed_ieeg_car_bp.h5 | features.h5 |
| `build_graph.py` | PyG graph construction | features.h5 | graphs.pt |
| `pretrain_gnn.py` | Self-supervised pretraining | features.h5 | pretrained_encoder.pt |
| `train_gnn.py` | Supervised fine-tuning | graphs.pt | soz_classifier.pt |
| `tune_gnn.py` | Optuna hyperparameter tuning | features.h5, graphs.pt | best_hyperparams.json |
| `train_gat.py` | GAT architecture | graphs.pt | soz_classifier_gat.pt |
| `train_gin.py` | GIN architecture | graphs.pt | soz_classifier_gin.pt |
| `tune_gat.py` | GAT tuning | graphs.pt | best_hyperparams_gat.json |

## Combined Dataset Pipeline (ds003029 + ds004100)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `preprocess_combined.py` | Unified preprocessing | ds003029 + ds004100 | `processed_combined/h5/features_combined.h5` |
| `extract_features_combined.py` | Feature extraction | features_combined.h5 | features_extracted.h5 |
| `build_graphs_combined.py` | Graph construction | features_extracted.h5 | graphs_combined.pt |
| `train_combined.py` | Train on combined data | graphs_combined.pt | soz_classifier_combined.pt |

## Augmentation Scripts (Experimental)

| Script | Purpose | Result |
|--------|---------|--------|
| `augment_graphs.py` | Time-shift augmentation | Did not improve (0.718) |
| `train_augmented.py` | Train on augmented graphs | Did not improve |

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `log_utils.py` | Logging utilities |

## Running the Combined Pipeline

```bash
# 1. Preprocess both datasets
/usr/local/mne-python/1.0.0_0/bin/python3 scripts/preprocess_combined.py

# 2. Extract features
/usr/local/mne-python/1.0.0_0/bin/python3 scripts/extract_features_combined.py

# 3. Build graphs
/usr/bin/python3.9 scripts/build_graphs_combined.py

# 4. Train
/usr/bin/python3.9 scripts/train_combined.py
```

## Environment Notes

Two Python environments are used:
- **MNE Python** (`/usr/local/mne-python/1.0.0_0/bin/python3`): mne, h5py, numpy, pandas, scipy
- **System Python** (`/usr/bin/python3.9`): torch, torch_geometric, sklearn

Preprocessing and feature extraction use MNE Python.
Graph building and training use System Python.
