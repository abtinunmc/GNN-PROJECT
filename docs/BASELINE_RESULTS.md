# Baseline Comparison Results

## Overview

Comparison of classical ML baselines against GNN for SOZ localization on the combined dataset (ds003029 + ds004100).

## Dataset Split

- **Train**: 70 graphs, 6,198 nodes (551 SOZ, 8.9%)
- **Val**: 36 graphs, 3,618 nodes (292 SOZ, 8.1%)
- **Test**: 20 graphs, 1,685 nodes (140 SOZ, 8.3%)

## Results

| Method | Test AUC | Test F1 | Test Precision | Test Recall |
|--------|----------|---------|----------------|-------------|
| Random Forest | **0.847** | **0.373** | **0.326** | 0.436 |
| Logistic Regression | 0.741 | 0.336 | 0.238 | 0.571 |
| Lasso (L1) | 0.738 | 0.338 | 0.240 | 0.571 |
| Ridge (L2) | 0.741 | 0.336 | 0.238 | 0.571 |
| LR (tuned, CV) | 0.742 | 0.340 | 0.242 | 0.571 |
| SVM (RBF) | 0.713 | 0.097 | 0.200 | 0.064 |
| **GNN (5 seeds)** | 0.763 ± 0.008 | 0.273 ± 0.007 | 0.175 ± 0.006 | **0.623 ± 0.014** |

## GNN Per-Seed Results

| Seed | AUC | F1 | Recall |
|------|-----|-----|--------|
| 42 | 0.753 | 0.260 | 0.621 |
| 123 | 0.776 | 0.278 | 0.643 |
| 456 | 0.762 | 0.281 | 0.629 |
| 789 | 0.756 | 0.270 | 0.600 |
| 2024 | 0.769 | 0.274 | 0.621 |

## Key Findings

### Random Forest Achieves Highest AUC
- RF: 0.847 vs GNN: 0.763 (+11% relative)
- RF benefits from: small sample robustness, strong engineered features, no graph construction uncertainty

### GNN Achieves Highest Recall
- GNN: 62.3% vs RF: 43.6% (+43% relative improvement)
- Critical for clinical SOZ screening where missing true positives is costly
- Graph structure helps identify connectivity patterns associated with SOZ

### Regularization Doesn't Help Linear Models
- Lasso/Ridge/Tuned LR all perform similarly to default LR (~0.74 AUC)
- Features are well-conditioned; regularization provides minimal benefit

### SVM Performs Poorly
- Lowest AUC (0.713) and near-zero recall (6.4%)
- RBF kernel may not suit the feature space

## Interpretation

The performance gap between RF and GNN likely reflects:
1. **Sample size**: 70 training graphs is small for deep learning
2. **Feature quality**: Band powers, HFO, line length are highly discriminative
3. **Graph construction**: Correlation-based edges may not optimally capture SOZ-relevant connectivity

However, GNN's recall advantage suggests graph structure helps identify true SOZ electrodes even if overall discrimination is lower.

## Output Files

- `data/processed/models/results_baselines_seeds.json` - Full results
- `scripts/run_baselines_and_seeds.py` - Evaluation script

## Date

Generated: 2026-05-05
