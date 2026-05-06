# SOZ Localization GNN

Graph Neural Network pipeline for Seizure Onset Zone (SOZ) localization using intracranial EEG data from OpenNeuro datasets (ds003029 + ds004100).

**Best Result**: Test AUC **0.785** with hybrid GNN + RF feature (81 patients, 126 labeled recordings, 63% recall)

## Structure

```text
docs/            Pipeline and stage documentation
scripts/         Runnable pipeline scripts
data/processed/  Generated artifacts
  ├── h5/        HDF5 files (preprocessed signals, features)
  ├── graphs/    PyTorch Geometric graph objects
  ├── models/    Trained model checkpoints
  ├── csv/       Metadata and QC failures
  └── figures/   Visualizations
logs/            Historical run logs
```

## Run Order

```bash
cd ~/Desktop/GNN-PROJECT-main

# Preprocessing (Stages 1-4)
python3 scripts/preprocess.py        # Notch filter, epoch extraction
python3 scripts/apply_car.py         # Common Average Reference
python3 scripts/apply_bandpass.py    # Bandpass 1-250 Hz
python3 scripts/extract_features.py  # Node/edge features (GPU)

# Graph Construction (Stage 5)
python3 scripts/build_graph.py       # PyG graphs (GPU)

# GNN Training (Stages 6-8)
python3 scripts/pretrain_gnn.py      # Self-supervised pretraining
python3 scripts/train_gnn.py         # Fine-tune SOZ classifier
python3 scripts/tune_gnn.py          # Hyperparameter tuning (Optuna)

# Alternative Architectures (optional)
python3 scripts/train_gat.py         # Graph Attention Network
python3 scripts/train_gin.py         # Graph Isomorphism Network
python3 scripts/tune_gat.py          # GAT hyperparameter tuning

# Baselines & Multi-seed Evaluation (for paper)
python3 scripts/run_baselines_and_seeds.py  # RF, LR, SVM + GNN with 5 seeds
```

## Results

| Model | Test AUC | Test Recall | Dataset |
|-------|----------|-------------|---------|
| **GNN + RF feature (hybrid)** | **0.785** | **0.630** | ds003029 + ds004100 |
| Random Forest | 0.847 | 0.436 | ds003029 + ds004100 |
| GraphSAGE (combined) | 0.768 | 0.650 | ds003029 + ds004100 |
| GraphSAGE (5 seeds) | 0.763 ± 0.008 | 0.623 ± 0.014 | ds003029 + ds004100 |
| GraphSAGE + augmentation | 0.761 | 0.520 | ds003029 |
| GraphSAGE (tuned) | 0.730 | 0.400 | ds003029 |
| GAT (tuned) | 0.702 | 0.411 | ds003029 |
| GIN | 0.562 | 0.000 | ds003029 |

## Notes

- Raw dataset location defaults to `~/Desktop/ds003029`
- Override dataset path with `DS003029_DIR=/path/to/ds003029`
- GPU (CUDA) required for feature extraction and GNN training
- PyTorch 2.0.1+cu117 required (driver 470+ compatible)

## Documentation

See `docs/PIPELINE.md` for complete pipeline documentation.
