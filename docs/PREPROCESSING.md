# SOZ Localization GNN - Preprocessing Pipeline

## Overview

Preprocessing pipeline for OpenNeuro ds003029 (HUP iEEG) dataset, adapted from the HFO-GLISKE pipeline approach.

## Dataset

- **Source**: OpenNeuro ds003029
- **Location**: `~/Desktop/ds003029`
- **Format**: BrainVision (.vhdr/.eeg/.vmrk) with BIDS metadata
- **Subjects**: 35 subjects from JHH, NIH, UMF, UMMC sites
- **Recordings**: 106 total, 103 successfully processed

## Pipeline Steps

### 1. Load BrainVision Data
- Uses MNE `read_raw_brainvision()` to load .vhdr files
- Preserves original sampling rate (typically 1000 Hz)

### 2. Parse Good/Bad Channels
- Reads `channels.tsv` for each recording
- Filters for ECOG/SEEG type channels
- Separates good vs bad channels based on `status` column

### 3. Apply Notch Filter (60 Hz)
- Removes line noise at 60 Hz and harmonics (120 Hz, 180 Hz)
- Only applies frequencies below Nyquist (handles low sampling rate recordings)

### 4. Extract Peri-ictal Epochs
- Parses seizure onset from `events.tsv` (looks for "SZ EVENT" markers)
- Extracts +/- 30 seconds around seizure onset
- Total epoch duration: 60 seconds

### 5. Parse SOZ Labels
- Extracts SOZ channel annotations from event markers
- Looks for "Z " prefixed annotations indicating seizure activity

### 6. Compute Global Valid Times
- **Per-channel**: Identifies 1-second windows with clean signal
  - Rejects windows with extreme variance (>5x global std)
  - Rejects flat windows (<0.01x global std)
- **Global intersection**: Uses two-pointer merge algorithm to find times where ALL channels are clean simultaneously
- Similar to HFO-GLISKE validTimes approach

### 7. Save to HDF5
- Output: `data/processed/h5/preprocessed_ieeg.h5`
- Structure per recording:
  - Group name: Uses the original `.vhdr` filename stem to guarantee uniqueness and prevent HDF5 name collisions.
  - `data`: (n_channels x n_samples) preprocessed signal
  - `global_valid_times`: (n_intervals x 2) start/stop times in seconds
  - Attributes: channels, sfreq, onset, soz_channels, n_channels, total_valid_sec

### 8. Metadata and Outcomes
- Gracefully handles surgical outcome loading from `participants.tsv` (handles missing files or missing `participant_id` columns).
- Outputs metadata to `metadata.csv` and records failures to `failed_recordings.csv`.
- Robust error handling skips recordings that end up with 0 good channels in the raw data without crashing.

## Configuration

```python
DATA_DIR = ~/Desktop/ds003029
OUTPUT_DIR = ~/Documents/gnn project/data/processed

NOTCH_FREQ = 60.0 Hz
PRE_SEIZURE = 30 seconds
POST_SEIZURE = 30 seconds
```

## Output Files

```
data/processed/
├── h5/preprocessed_ieeg.h5       # All preprocessed data
├── csv/metadata.csv              # Recording metadata
└── figures/
    ├── preprocessing_demo.png    # Single-channel before/after visualization
    └── preprocessing_multichannel.png # Multi-channel visualization
```

## Results Summary

| Metric | Value |
|--------|-------|
| Total recordings | 106 |
| Successfully processed | 103 |
| Failed (no seizure onset) | 3 |
| Total channels | 7,840 |
| Sampling frequency | 1000 Hz (most recordings) |
| Recordings with SOZ annotations | 30 |
| Recordings with global valid times | 99 |
| Total valid signal time | 5,521 seconds |

## Failed Recordings

- `sub-pt11_ses-presurgery_task-ictal_acq-ecog_run-03`: No seizure onset found
- `sub-pt17_ses-presurgery_task-ictal_acq-ecog_run-01`: No seizure onset found
- `sub-pt17_ses-presurgery_task-ictal_acq-ecog_run-03`: No seizure onset found

## Key Differences from Original step1.py

| Original step1.py | New preprocess.py |
|-------------------|-------------------|
| Bandpass 1-100 Hz | No bandpass (preserves HFO band) |
| Downsample to 250 Hz | Keep original 1000 Hz |
| Simple valid times | Global valid times (all channels) |
| Combined artifact check | Per-channel + intersection |

## Key Differences from HFO-GLISKE Pipeline

| HFO-GLISKE | This Pipeline |
|------------|---------------|
| Persyst .lay/.dat format | BrainVision .vhdr format |
| Pre-computed HFO events | No HFO detection (use valid times directly) |
| CAR in separate step | CAR to be added in next step |
| 80-500 Hz bandpass | No bandpass yet |

## Usage

```bash
cd ~/Documents/gnn\ project
python3 scripts/preprocess.py
```

## Next Steps

1. **CAR (Common Average Reference)**: Apply using good channels only
2. **Feature Extraction**: Compute band powers, connectivity features
3. **Graph Construction**: Build graphs with channels as nodes
4. **GNN Training**: Train model for SOZ prediction

## Dependencies

- numpy
- pandas
- mne
- h5py
- scipy
- matplotlib
- tqdm
