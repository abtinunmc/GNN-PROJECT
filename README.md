# SOZ Localization GNN

Clean project layout for seizure onset zone localization with iEEG preprocessing,
feature extraction, and planned graph-learning stages.

## Structure

```text
docs/            Pipeline and stage documentation
scripts/         Runnable pipeline scripts
data/processed/  Generated HDF5 files, CSVs, and figures
  ├── h5/        Generated HDF5 files
  ├── csv/       Metadata and failures CSVs
  └── figures/   Generated figures
logs/            Historical run logs
```

## Run Order

```bash
cd ~/Documents/gnn\ project
python3 scripts/preprocess.py
python3 scripts/apply_car.py
python3 scripts/apply_bandpass.py
python3 scripts/extract_features.py
```

## Notes

- Raw dataset location defaults to `~/Desktop/ds003029`
- Override dataset path with `DS003029_DIR=/path/to/ds003029`
- Generated artifacts are written under `data/processed/`

## Version Control

This project is tracked via Git and hosted on GitHub.
**Important:** Due to GitHub's strict file size limits (100 MB), large intermediate data files (e.g., `*.h5` artifacts in `data/processed/h5/`) are explicitly excluded via `.gitignore`. The repository tracks all scripts, Markdown documentation, logs, output figures, and metadata CSVs.
