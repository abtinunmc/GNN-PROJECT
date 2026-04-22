# Common Average Reference (CAR) - SOZ Localization GNN

## Overview

CAR is a spatial filtering technique that removes the common signal present across all electrodes, improving signal-to-noise ratio for focal neural activity detection.

## What is CAR?

For each time point, CAR computes the mean across all channels and subtracts it from each channel:

```
x_car[ch, t] = x_raw[ch, t] - mean(x_raw[:, t])
```

## Why Use CAR?

1. **Removes global noise**: Movement artifacts, distant physiological noise
2. **Reduces volume conduction**: Common signal from distant sources
3. **Removes reference bias**: Eliminates reference electrode influence
4. **Preserves local activity**: Focal signals (like SOZ) remain intact

## Implementation

```python
def apply_car(data: np.ndarray) -> np.ndarray:
    # data shape: (n_channels, n_samples)

    # Step 1: Compute mean across all channels for each time point
    common_average = np.mean(data, axis=0, keepdims=True)  # (1, n_samples)

    # Step 2: Subtract common average from each channel
    data_car = data - common_average  # Broadcasting: (n_channels, n_samples)

    return data_car
```

## Input/Output

| Parameter | Value |
|-----------|-------|
| Input file | `data/processed/h5/preprocessed_ieeg.h5` |
| Output file | `data/processed/h5/preprocessed_ieeg_car.h5` |
| Processing time | ~80 seconds |
| Recordings processed | 103/103 |

## Visualizations Generated

1. `car_demo.png` - Single channel before/after CAR (time domain + spectrum)
2. `car_multichannel.png` - Multi-channel view showing common signal removal
3. `car_statistics.png` - Per-channel variance and correlation changes

## Effects of CAR

### What CAR Removes
- Global fluctuations common to all electrodes
- Reference electrode artifacts
- Distant volume-conducted activity

### What CAR Preserves
- Local/focal activity (important for SOZ)
- Channel-specific signals
- High-frequency oscillations (HFOs)

## Usage

```bash
cd ~/Documents/gnn\ project
python3 scripts/apply_car.py
```

## Script Location

`scripts/apply_car.py` - Main CAR application script

## Dependencies

- numpy
- h5py
- scipy
- matplotlib
- tqdm
