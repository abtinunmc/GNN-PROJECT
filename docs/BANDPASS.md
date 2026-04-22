# Elliptic Bandpass Filter - SOZ Localization GNN

## Overview

This step applies an elliptic (Cauer) bandpass filter to the CAR-referenced data, keeping frequencies from 1-250 Hz to preserve HFO activity while removing DC drift and high-frequency noise.

## Why Elliptic Filter?

Elliptic filters offer the steepest roll-off of any filter type for a given order:

| Filter Type | Passband | Stopband | Roll-off |
|-------------|----------|----------|----------|
| Butterworth | Flat | Flat | Slowest |
| Chebyshev I | Ripple | Flat | Medium |
| Chebyshev II | Flat | Ripple | Medium |
| **Elliptic** | Ripple | Ripple | **Steepest** |

For iEEG/HFO analysis, elliptic is ideal because:
- Sharp cutoffs preserve the HFO band (80-500 Hz)
- Small ripples are acceptable for neural signals
- Computational efficiency for large datasets

## Filter Parameters

```python
LOW_FREQ = 1.0       # Hz - removes DC drift
HIGH_FREQ = 250.0    # Hz - preserves HFOs (adjusted for Nyquist)
FILTER_ORDER = 4     # Balance of sharpness vs stability
RIPPLE_DB = 0.1      # Maximum passband ripple (very small)
STOP_ATTEN_DB = 40   # Stopband attenuation (good rejection)
```

## Implementation

```python
from scipy import signal

def design_elliptic_bandpass(low_freq, high_freq, sfreq, order=4, rp=0.1, rs=40):
    # Normalize frequencies to Nyquist
    nyquist = sfreq / 2.0
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    # Design elliptic bandpass filter (SOS format for stability)
    sos = signal.ellip(
        N=order,
        rp=rp,             # Passband ripple in dB
        rs=rs,             # Stopband attenuation in dB
        Wn=[low_normalized, high_normalized],
        btype='bandpass',
        output='sos'       # Second-order sections (more stable)
    )
    return sos

def apply_bandpass_filter(data, sos):
    # Zero-phase filtering (no phase distortion)
    for ch in range(data.shape[0]):
        data_filtered[ch, :] = signal.sosfiltfilt(sos, data[ch, :])
    return data_filtered
```

## Key Design Choices

### Why SOS Format?
- Second-order sections are more numerically stable
- Better for high-order filters
- Recommended by scipy for IIR filters

### Why sosfiltfilt (Zero-Phase)?
- Applies filter forward and backward
- Results in zero phase distortion
- Essential for preserving event timing in neural signals
- Doubles effective filter order

### Nyquist Handling
Some recordings have low sampling rates (~250-500 Hz). The script automatically adjusts:
```python
if high_freq >= nyquist:
    high_freq = nyquist - 1  # Stay below Nyquist
```

## Input/Output

| Parameter | Value |
|-----------|-------|
| Input file | `data/processed/h5/preprocessed_ieeg_car.h5` |
| Output file | `data/processed/h5/preprocessed_ieeg_car_bp.h5` |
| Processing time | ~82 seconds |
| Recordings processed | 103/103 |

## Recordings with Adjusted Cutoffs

| Recording Pattern | Original sfreq | Adjusted High Cutoff |
|-------------------|----------------|---------------------|
| sub-umf003 | 499.75 Hz | 248.88 Hz |
| sub-ummc001-003 | 499.75 Hz | 248.88 Hz |
| sub-ummc003-004, 006, 008 | 249.88 Hz | 123.94 Hz |

## Visualizations Generated

1. `bandpass_filter_response.png` - Filter frequency/phase response
2. `bandpass_demo.png` - Single channel before/after comparison
3. `bandpass_bands.png` - Power by frequency band analysis

## Frequency Bands Preserved

| Band | Frequency Range | Relevance |
|------|-----------------|-----------|
| Delta | 1-4 Hz | Slow oscillations |
| Theta | 4-8 Hz | Memory, navigation |
| Alpha | 8-13 Hz | Idle rhythm |
| Beta | 13-30 Hz | Motor, active processing |
| Gamma | 30-80 Hz | Local processing |
| High Gamma | 80-150 Hz | Very local activity |
| HFO | 150-250 Hz | **Epileptogenic** |

## Usage

```bash
cd ~/Documents/gnn\ project
python3 scripts/apply_bandpass.py
```

## Script Location

`scripts/apply_bandpass.py` - Main bandpass filtering script

## Dependencies

- numpy
- h5py
- scipy
- matplotlib
- tqdm
