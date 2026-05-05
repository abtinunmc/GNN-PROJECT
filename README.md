# SOZ Localization GNN

Graph Neural Network pipeline for Seizure Onset Zone (SOZ) localization using intracranial EEG data from OpenNeuro ds003029 (HUP iEEG dataset).

**Best Result**: Test AUC 0.730 with tuned GraphSAGE

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
```

## Results

| Model | Test AUC |
|-------|----------|
| **GraphSAGE (tuned)** | **0.730** |
| GAT (tuned) | 0.702 |
| GraphSAGE (baseline) | 0.697 |
| GAT (baseline) | 0.685 |
| GIN | 0.562 |

## Notes

- Raw dataset location defaults to `~/Desktop/ds003029`
- Override dataset path with `DS003029_DIR=/path/to/ds003029`
- GPU (CUDA) required for feature extraction and GNN training
- PyTorch 2.0.1+cu117 required (driver 470+ compatible)

## Documentation

See `docs/PIPELINE.md` for complete pipeline documentation.
