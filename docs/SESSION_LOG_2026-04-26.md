# Session Log: 2026-04-26

## Objective
Combine OpenNeuro ds003029 and ds004100 datasets for improved SOZ localization

## Work Completed

### 1. Dataset Discovery
- Searched for compatible iEEG datasets with SOZ labels
- Identified ds004100 (58 patients) as ideal match for ds003029 (23 patients)
- Both from Hospital of University of Pennsylvania (HUP)
- Same BIDS format, compatible SOZ labeling

### 2. Dataset Download
- Downloaded ds004100 from OpenNeuro via AWS S3
- Size: ~3GB
- 109 ictal recordings with SOZ labels in channels.tsv

### 3. Scripts Created

#### preprocess_combined.py
- Unified preprocessing for both datasets
- Handles different file formats (BrainVision vs EDF)
- Handles different SOZ annotation formats:
  - ds003029: SOZ in events.tsv (complex parsing)
  - ds004100: SOZ in channels.tsv status_description column
- Output: 212 recordings processed, 126 with SOZ labels

#### extract_features_combined.py
- Window-based feature extraction (12s windows)
- Band powers: delta, theta, alpha, beta, low_gamma, HFO
- Statistical: variance, line length, kurtosis, skew
- Correlation matrices for edges
- Output: 208 recordings with features

#### build_graphs_combined.py
- PyTorch Geometric graph construction
- Subject-level train/val/test split (60/20/20)
- 62 subjects total (37 train, 12 val, 13 test)
- Output: 70 train, 36 val, 20 test graphs (labeled)

#### train_combined.py
- Self-supervised pretraining (661 window pairs)
- Supervised fine-tuning with augmentation
- Class-weighted loss for imbalance
- Best hyperparameters from previous tuning

### 4. Training Results

```
Pretraining: 661 train, 166 val window pairs
Best pretrain val loss: 0.2819 (early stop epoch 99)

Fine-tuning: 70 train, 36 val, 20 test graphs
Train split: 13 ds003029, 57 ds004100
Class weights: [0.55, 5.62]
Best val AUC: 0.7511 (early stop epoch 25)

Final Results:
- Validation AUC: 0.7511
- Test AUC: 0.7677
- Test F1: 0.2964
- Test Precision: 0.1920
- Test Recall: 0.6500
```

### 5. Comparison to Previous Work

| Approach | Test AUC | Notes |
|----------|----------|-------|
| Baseline (ds003029) | 0.730 | Original |
| + Online augmentation | 0.761 | Best single-dataset |
| + Time-shift aug | 0.718 | Didn't help |
| + Mixup | 0.731 | Didn't help |
| + SMOTE | 0.696 | Didn't help |
| **Combined dataset** | **0.768** | Best overall |

### 6. Augmentation Experiments (Earlier in Session)

Implemented and tested:
1. Online augmentation (edge dropout, feature noise, feature masking) - **Helped**
2. Time-shift augmentation (window dropout + re-aggregation) - Did not help
3. Mixup augmentation - Did not help
4. SMOTE oversampling - Did not help

## Files Created

### Scripts
- `/scripts/preprocess_combined.py`
- `/scripts/extract_features_combined.py`
- `/scripts/build_graphs_combined.py`
- `/scripts/train_combined.py`
- `/scripts/augment_graphs.py` (time-shift, not used)
- `/scripts/train_augmented.py` (not used)

### Data
- `/data/processed_combined/h5/features_combined.h5`
- `/data/processed_combined/h5/features_extracted.h5`
- `/data/processed_combined/graphs/graphs_combined.pt`
- `/data/processed_combined/models/soz_classifier_combined.pt`
- `/data/processed_combined/models/metrics_combined.json`

### Documentation
- `/docs/COMBINED_DATASET_RESULTS.md`
- `/docs/SESSION_LOG_2026-04-26.md` (this file)

## Discussion: Preprint Readiness

Current state is sufficient for a basic preprint with:
- Add 1 baseline (Random Forest)
- Run 3 seeds for confidence intervals
- Write 4-5 page paper

User plans:
1. Publish preprint with public data (OpenNeuro)
2. Later publish full paper with UNMC PLV data

This is a valid strategy - preprint establishes method, paper validates on clinical data.

## Technical Notes

### Python Environments
- MNE Python (`/usr/local/mne-python/1.0.0_0/bin/python3`): mne, h5py, numpy, pandas
- System Python (`/usr/bin/python3.9`): torch, torch_geometric, sklearn

### Key Differences Between Datasets

| Aspect | ds003029 | ds004100 |
|--------|----------|----------|
| File format | BrainVision (.vhdr) | EDF (.edf) |
| SOZ location | events.tsv | channels.tsv |
| SOZ format | "Z LAT1-2, 6-7" | status_description="soz" |
| Subjects | sub-RID* | sub-HUP* |

### Runtime Summary
- ds004100 download: ~2 min
- Preprocessing: 209 sec
- Feature extraction: 105 sec
- Graph building: 13 sec
- Training: 35 sec
- **Total pipeline: ~6 min**
