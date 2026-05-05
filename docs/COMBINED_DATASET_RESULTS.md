# Combined Dataset Results: SOZ Localization with GNN

## Overview

This document summarizes the work done to combine two OpenNeuro iEEG datasets for improved SOZ (Seizure Onset Zone) localization using Graph Neural Networks.

## Datasets Combined

| Dataset | Source | Patients | Recordings | SOZ Labels |
|---------|--------|----------|------------|------------|
| ds003029 | OpenNeuro/HUP | 23 | 103 | From events.tsv |
| ds004100 | OpenNeuro/HUP | 58 | 109 | From channels.tsv |
| **Combined** | - | **81** | **212** | **126 labeled** |

## Pipeline Scripts Created

### Preprocessing
- `scripts/preprocess_combined.py` - Unified preprocessing for both datasets
  - Handles BrainVision (.vhdr) and EDF (.edf) formats
  - Parses SOZ labels from different annotation formats
  - Applies notch filtering (60 Hz)
  - Extracts peri-ictal epochs (±30s around seizure onset)

### Feature Extraction
- `scripts/extract_features_combined.py` - Windowed feature extraction
  - 12-second windows
  - Band powers: delta, theta, alpha, beta, low_gamma, HFO
  - Statistical features: variance, line length, kurtosis, skew
  - Correlation matrices for edge features

### Graph Construction
- `scripts/build_graphs_combined.py` - PyTorch Geometric graph building
  - Node features: aggregated spectral + statistical (33 features)
  - Edge features: correlation-based connectivity
  - Train/Val/Test split by subject (60/20/20)

### Training
- `scripts/train_combined.py` - GNN training pipeline
  - Self-supervised pretraining (next-window prediction)
  - Supervised fine-tuning with class-weighted BCE loss
  - Online augmentation (edge dropout, feature noise, masking)

## Model Architecture

- **Encoder**: GraphSAGE with BatchNorm
- **Layers**: 2
- **Hidden dim**: 128
- **Dropout**: 0.21 (encoder), 0.55 (classifier)
- **Pretraining**: 200 epochs, early stopping
- **Fine-tuning**: 150 epochs, early stopping

## Results

### Combined Dataset Performance

| Metric | Value |
|--------|-------|
| Validation AUC | 0.751 |
| **Test AUC** | **0.768** |
| Test F1 | 0.296 |
| Test Precision | 0.192 |
| Test Recall | 0.650 |

### Comparison to Single Dataset

| Configuration | Test AUC | Training Graphs |
|---------------|----------|-----------------|
| ds003029 only | 0.730 | ~20 |
| ds003029 + augmentation | 0.761 | ~20 |
| **Combined (ds003029 + ds004100)** | **0.768** | 70 |

### Key Improvements
- Test AUC: 0.730 → 0.768 (+5.2% from baseline)
- Training data: 3.5x increase (20 → 70 graphs)
- Recall: 48% → 65% (+35% relative improvement)

## Output Files

```
data/processed_combined/
├── h5/
│   ├── features_combined.h5      # Raw preprocessed data
│   └── features_extracted.h5     # Extracted features
├── graphs/
│   └── graphs_combined.pt        # PyTorch Geometric graphs
└── models/
    ├── soz_classifier_combined.pt    # Trained model
    └── metrics_combined.json         # Performance metrics
```

## Data Augmentation Techniques Tested

| Technique | Test AUC | Notes |
|-----------|----------|-------|
| None (baseline) | 0.730 | Original ds003029 |
| Online augmentation | 0.761 | Edge drop + noise + mask |
| Time-shift | 0.718 | Did not help |
| Mixup | 0.731 | Did not help |
| SMOTE | 0.696 | Did not help |
| **Combined dataset** | **0.768** | Best result |

## Best Hyperparameters

```python
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.21
CLASSIFIER_DROPOUT = 0.55
PRETRAIN_LR = 1.2e-4
FINETUNE_LR = 6.1e-4
WEIGHT_DECAY = 9e-5
BATCH_SIZE = 16
CORRELATION_THRESHOLD = 0.32

# Augmentation
AUG_EDGE_DROP = 0.076
AUG_FEAT_NOISE = 0.011
AUG_FEAT_MASK = 0.10
```

## Next Steps for Preprint

1. Add baseline comparisons (Random Forest, Logistic Regression)
2. Run multiple seeds (3-5x) for confidence intervals
3. Cross-dataset generalization test
4. Write 4-5 page paper

## Runtime

- Preprocessing: ~3.5 min
- Feature extraction: ~1.7 min
- Graph building: ~13 sec
- Training: ~35 sec
- **Total: ~6 min**

## Date

Generated: 2026-04-26
