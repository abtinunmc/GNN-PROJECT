"""
Bandpass Filtering for SOZ Localization GNN
============================================

This script applies bandpass filtering to the CAR-referenced iEEG data
using an Elliptic (Cauer) filter.

Why Elliptic Filter?
--------------------
Elliptic filters (also called Cauer filters) offer:
1. Steepest roll-off of any filter for a given order
2. Equiripple behavior in both passband and stopband
3. Most efficient transition band (narrowest)
4. Trade-off: has ripples in both passband and stopband

Comparison of filter types:
- Butterworth: Maximally flat passband, slow roll-off
- Chebyshev I: Ripple in passband, steeper than Butterworth
- Chebyshev II: Ripple in stopband, flat passband
- Elliptic: Ripple in both, STEEPEST roll-off (most efficient)

For iEEG/HFO analysis, elliptic is good because:
- We need sharp cutoffs to preserve HFO band (80-500 Hz)
- Small ripples are acceptable for neural signal analysis
- Computational efficiency matters for large datasets

Filter Parameters
-----------------
- Low cutoff: 1 Hz (removes DC drift and slow artifacts)
- High cutoff: 250 Hz (preserves HFOs, stays below Nyquist for 1000 Hz data)
- Order: 4 (good balance of sharpness vs stability)
- Passband ripple: 0.1 dB (very small)
- Stopband attenuation: 40 dB (good rejection)

Input:
    data/processed/h5/preprocessed_ieeg_car.h5 (CAR-referenced data)

Output:
    data/processed/h5/preprocessed_ieeg_car_bp.h5 (bandpass filtered data)
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
import time

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Input: CAR-referenced data
INPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg_car.h5"

# Output: Bandpass filtered data
OUTPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg_car_bp.h5"

# Output directory for visualizations
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "figures"

# ============================================================================
# BANDPASS FILTER PARAMETERS
# ============================================================================

# Frequency band to keep (in Hz)
LOW_FREQ = 1.0      # High-pass cutoff: removes DC drift, slow movement artifacts
HIGH_FREQ = 250.0   # Low-pass cutoff: keeps HFOs (80-500 Hz range), must be < Nyquist

# Elliptic filter parameters
FILTER_ORDER = 10    # Filter order (higher = sharper cutoff but more ringing)
RIPPLE_DB = 0.1     # Maximum ripple in passband (dB) - very small
STOP_ATTEN_DB = 60  # Minimum attenuation in stopband (dB) - good rejection


# ============================================================================
# FILTER DESIGN FUNCTIONS
# ============================================================================

def design_elliptic_bandpass(low_freq: float, high_freq: float, sfreq: float,
                              order: int = 4, rp: float = 0.1, rs: float = 40):
    """
    Design an elliptic (Cauer) bandpass filter.

    Parameters
    ----------
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Filter order. Higher = sharper cutoff but more potential instability.
    rp : float
        Maximum ripple allowed in the passband (dB).
        Smaller = flatter passband but wider transition band.
    rs : float
        Minimum attenuation in the stopband (dB).
        Higher = better rejection but wider transition band.

    Returns
    -------
    sos : np.ndarray
        Second-order sections representation of the filter.
        SOS format is more numerically stable than transfer function (b, a).

    Notes
    -----
    We use SOS (second-order sections) format because:
    1. More numerically stable than transfer function coefficients
    2. Better for high-order filters
    3. Recommended by scipy for IIR filters
    """

    # Calculate Nyquist frequency (half the sampling rate)
    # All frequencies must be normalized to Nyquist for digital filter design
    nyquist = sfreq / 2.0

    # Check that our frequencies are valid
    if high_freq >= nyquist:
        # If high_freq exceeds Nyquist, cap it slightly below
        print(f"  Warning: high_freq ({high_freq} Hz) >= Nyquist ({nyquist} Hz)")
        print(f"  Adjusting high_freq to {nyquist - 1} Hz")
        high_freq = nyquist - 1

    if low_freq <= 0:
        print(f"  Warning: low_freq must be > 0, adjusting to 0.5 Hz")
        low_freq = 0.5

    # Normalize frequencies to Nyquist (scipy convention)
    # Wn should be in range (0, 1) where 1 = Nyquist
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    # Design the elliptic bandpass filter
    # Output='sos' gives second-order sections (more stable)
    sos = signal.ellip(
        N=order,           # Filter order
        rp=rp,             # Passband ripple in dB
        rs=rs,             # Stopband attenuation in dB
        Wn=[low_normalized, high_normalized],  # Normalized cutoff frequencies
        btype='bandpass',  # Bandpass filter
        analog=False,      # Digital filter (not analog)
        output='sos'       # Return second-order sections
    )

    return sos, low_freq, high_freq


def apply_bandpass_filter(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    Apply bandpass filter to multi-channel data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_channels, n_samples).
    sos : np.ndarray
        Second-order sections of the filter (from design_elliptic_bandpass).

    Returns
    -------
    data_filtered : np.ndarray
        Filtered data with same shape as input.

    Notes
    -----
    We use sosfiltfilt (zero-phase filtering) because:
    1. Applies filter forward and backward
    2. Results in zero phase distortion (important for timing analysis)
    3. Doubles the effective filter order
    4. Essential for analyzing event timing in neural signals

    The alternative sosfilt() would introduce phase delay, which would
    shift the timing of neural events - bad for SOZ localization!
    """

    n_channels, n_samples = data.shape

    # sosfiltfilt requires minimum samples for padding
    # Padding length = 3 * max(len(sos sections)) = 3 * order * 2 = 3 * 4 * 2 = 24
    # We need at least padlen + 1 samples, so minimum ~30 samples
    MIN_SAMPLES = 30

    if n_samples < MIN_SAMPLES:
        # Data too short for filtering, return zeros (will be flagged)
        print(f"    Warning: Data too short ({n_samples} samples) for filtering")
        return np.zeros_like(data)

    # Allocate output array
    data_filtered = np.zeros_like(data)

    # Apply filter to each channel
    # sosfiltfilt applies the filter twice (forward + backward) for zero phase
    for ch in range(n_channels):
        data_filtered[ch, :] = signal.sosfiltfilt(sos, data[ch, :])

    return data_filtered


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_all_recordings():
    """
    Apply bandpass filter to all recordings in the CAR-referenced HDF5 file.

    Workflow
    --------
    1. Open input HDF5 (CAR-referenced data)
    2. Create output HDF5
    3. For each recording:
       a. Read the CAR-referenced data
       b. Design filter appropriate for this recording's sampling rate
       c. Apply bandpass filter
       d. Save filtered data with metadata
    4. Close files
    """

    print("=" * 70)
    print("Applying Elliptic Bandpass Filter")
    print("=" * 70)
    print(f"\nFilter parameters:")
    print(f"  - Low cutoff:  {LOW_FREQ} Hz (removes DC drift)")
    print(f"  - High cutoff: {HIGH_FREQ} Hz (preserves HFOs)")
    print(f"  - Filter type: Elliptic (Cauer)")
    print(f"  - Order:       {FILTER_ORDER}")
    print(f"  - Passband ripple:      {RIPPLE_DB} dB")
    print(f"  - Stopband attenuation: {STOP_ATTEN_DB} dB")
    print(f"\nInput:  {INPUT_H5}")
    print(f"Output: {OUTPUT_H5}")

    # Check input exists
    if not INPUT_H5.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_H5}")

    # Track statistics
    total_recordings = 0
    processed_recordings = 0
    skipped_recordings = 0

    # Cache for filter designs (key = sfreq)
    # Avoids redesigning filter for each recording with same sampling rate
    filter_cache = {}

    # Open input and output files
    with h5py.File(INPUT_H5, 'r') as f_in, h5py.File(OUTPUT_H5, 'w') as f_out:

        recording_names = list(f_in.keys())
        total_recordings = len(recording_names)

        print(f"\nFound {total_recordings} recordings to process")
        print("-" * 70)

        for rec_name in tqdm(recording_names, desc="Bandpass filtering"):

            grp_in = f_in[rec_name]

            # Read data and metadata
            data = grp_in['data'][:]
            sfreq = grp_in.attrs['sfreq']
            channels = grp_in.attrs['channels']
            onset = grp_in.attrs['onset']
            soz_channels = grp_in.attrs['soz_channels']
            n_channels = grp_in.attrs['n_channels']
            n_good_channels = grp_in.attrs['n_good_channels']
            n_samples = grp_in.attrs['n_samples']
            total_valid_sec = grp_in.attrs['total_valid_sec']

            # Read global valid times if they exist
            global_valid_times = None
            if 'global_valid_times' in grp_in:
                global_valid_times = grp_in['global_valid_times'][:]

            # ----------------------------------------------------------------
            # CHECK NYQUIST CONSTRAINT
            # ----------------------------------------------------------------
            # Nyquist frequency = sfreq / 2
            # Our high cutoff must be less than Nyquist
            nyquist = sfreq / 2.0

            if HIGH_FREQ >= nyquist:
                # For low sampling rate recordings, adjust high cutoff
                actual_high = nyquist - 1  # Stay 1 Hz below Nyquist
                print(f"\n  {rec_name}: sfreq={sfreq} Hz, adjusting high_freq to {actual_high} Hz")
            else:
                actual_high = HIGH_FREQ

            # ----------------------------------------------------------------
            # GET OR CREATE FILTER
            # ----------------------------------------------------------------
            # Use cached filter if we've seen this sampling rate before
            cache_key = (sfreq, actual_high)

            if cache_key not in filter_cache:
                # Design new filter for this sampling rate
                sos, _, _ = design_elliptic_bandpass(
                    low_freq=LOW_FREQ,
                    high_freq=actual_high,
                    sfreq=sfreq,
                    order=FILTER_ORDER,
                    rp=RIPPLE_DB,
                    rs=STOP_ATTEN_DB
                )
                filter_cache[cache_key] = sos

            sos = filter_cache[cache_key]

            # ----------------------------------------------------------------
            # APPLY BANDPASS FILTER
            # ----------------------------------------------------------------
            data_filtered = apply_bandpass_filter(data, sos)

            # ----------------------------------------------------------------
            # SAVE TO OUTPUT FILE
            # ----------------------------------------------------------------
            grp_out = f_out.create_group(rec_name)

            # Save filtered data
            grp_out.create_dataset('data', data=data_filtered, compression='gzip')

            # Copy global valid times (unchanged by filtering)
            if global_valid_times is not None and len(global_valid_times) > 0:
                grp_out.create_dataset('global_valid_times', data=global_valid_times)

            # Copy all attributes
            grp_out.attrs['channels'] = channels
            grp_out.attrs['sfreq'] = sfreq
            grp_out.attrs['onset'] = onset
            grp_out.attrs['soz_channels'] = soz_channels
            grp_out.attrs['n_channels'] = n_channels
            grp_out.attrs['n_good_channels'] = n_good_channels
            grp_out.attrs['n_samples'] = n_samples
            grp_out.attrs['total_valid_sec'] = total_valid_sec

            # Add new attributes to track processing
            grp_out.attrs['car_applied'] = True
            grp_out.attrs['bandpass_applied'] = True
            grp_out.attrs['bandpass_low'] = LOW_FREQ
            grp_out.attrs['bandpass_high'] = actual_high
            grp_out.attrs['filter_type'] = 'elliptic'
            grp_out.attrs['filter_order'] = FILTER_ORDER

            processed_recordings += 1

    print("-" * 70)
    print(f"Successfully processed {processed_recordings}/{total_recordings} recordings")
    print(f"Skipped: {skipped_recordings}")
    print(f"Output saved to: {OUTPUT_H5}")

    return processed_recordings


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_filter_response():
    """
    Plot the frequency response of the elliptic bandpass filter.

    This shows:
    1. Magnitude response (how much each frequency is attenuated)
    2. Phase response (phase shift at each frequency)
    3. Passband detail (zoom into the passband region)
    """

    print("\n" + "=" * 70)
    print("Generating Filter Response Visualization")
    print("=" * 70)

    # Design filter for typical 1000 Hz sampling rate
    sfreq = 1000.0
    sos, low, high = design_elliptic_bandpass(
        LOW_FREQ, HIGH_FREQ, sfreq,
        FILTER_ORDER, RIPPLE_DB, STOP_ATTEN_DB
    )

    # Compute frequency response
    # worN = number of frequency points to compute
    w, h = signal.sosfreqz(sos, worN=2000, fs=sfreq)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Elliptic Bandpass Filter Response\n'
                 f'{LOW_FREQ}-{HIGH_FREQ} Hz, Order {FILTER_ORDER}, '
                 f'Ripple {RIPPLE_DB} dB, Atten {STOP_ATTEN_DB} dB',
                 fontsize=14, fontweight='bold')

    # Top-left: Full magnitude response (dB)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)  # Add small value to avoid log(0)
    axes[0, 0].plot(w, magnitude_db, 'b', linewidth=1)
    axes[0, 0].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7, label=f'Low cutoff ({LOW_FREQ} Hz)')
    axes[0, 0].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7, label=f'High cutoff ({HIGH_FREQ} Hz)')
    axes[0, 0].axhline(-3, color='g', linestyle=':', alpha=0.7, label='-3 dB')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Magnitude Response (Full Range)')
    axes[0, 0].set_xlim([0, sfreq/2])
    axes[0, 0].set_ylim([-60, 5])
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Passband detail
    passband_mask = (w >= LOW_FREQ * 0.5) & (w <= HIGH_FREQ * 1.5)
    axes[0, 1].plot(w[passband_mask], magnitude_db[passband_mask], 'b', linewidth=1)
    axes[0, 1].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(-RIPPLE_DB, color='orange', linestyle=':', alpha=0.7, label=f'Ripple limit ({RIPPLE_DB} dB)')
    axes[0, 1].axhline(RIPPLE_DB, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_title('Passband Detail')
    axes[0, 1].set_ylim([-1, 0.5])
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)

    # Shade passband
    axes[0, 1].axvspan(LOW_FREQ, HIGH_FREQ, alpha=0.2, color='green', label='Passband')

    # Bottom-left: Phase response
    phase = np.unwrap(np.angle(h))
    axes[1, 0].plot(w, np.degrees(phase), 'b', linewidth=1)
    axes[1, 0].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Phase (degrees)')
    axes[1, 0].set_title('Phase Response (note: filtfilt gives zero phase)')
    axes[1, 0].set_xlim([0, sfreq/2])
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Impulse response
    # Create impulse and filter it
    impulse = np.zeros(1000)
    impulse[500] = 1.0
    impulse_response = signal.sosfilt(sos, impulse)

    time_impulse = np.arange(len(impulse)) / sfreq * 1000  # Convert to ms
    axes[1, 1].plot(time_impulse, impulse_response, 'b', linewidth=1)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Impulse Response')
    axes[1, 1].set_xlim([400, 600])  # Zoom around impulse
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "bandpass_filter_response.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_bandpass_demo():
    """
    Generate before/after plots showing the effect of bandpass filtering.
    """

    print("\nGenerating Bandpass Effect Visualization")

    # Check files exist
    if not INPUT_H5.exists() or not OUTPUT_H5.exists():
        print("Input or output HDF5 not found, skipping visualization")
        return

    with h5py.File(INPUT_H5, 'r') as f_before, h5py.File(OUTPUT_H5, 'r') as f_after:

        # Use first recording
        rec_name = list(f_before.keys())[0]
        print(f"Using recording: {rec_name}")

        # Read data
        data_before = f_before[rec_name]['data'][:]
        data_after = f_after[rec_name]['data'][:]
        sfreq = f_before[rec_name].attrs['sfreq']
        channels = f_before[rec_name].attrs['channels']

        n_channels, n_samples = data_before.shape

        # Time vector (2 seconds for time domain)
        plot_samples = min(int(2 * sfreq), n_samples)
        time_vec = np.arange(plot_samples) / sfreq

        # Select channel
        demo_ch = 0
        demo_ch_name = channels[demo_ch] if len(channels) > demo_ch else f"Ch{demo_ch}"

        # ====================================================================
        # FIGURE 1: Single channel comparison
        # ====================================================================

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Bandpass Filter Effect: {rec_name}\n'
                     f'Channel: {demo_ch_name} | Filter: {LOW_FREQ}-{HIGH_FREQ} Hz Elliptic',
                     fontsize=14, fontweight='bold')

        # Top-left: Time domain BEFORE
        axes[0, 0].plot(time_vec, data_before[demo_ch, :plot_samples] * 1e6,
                        'b', linewidth=0.5)
        axes[0, 0].set_title('Before Bandpass: Time Domain (CAR only)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude (µV)')
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: Time domain AFTER
        axes[0, 1].plot(time_vec, data_after[demo_ch, :plot_samples] * 1e6,
                        'g', linewidth=0.5)
        axes[0, 1].set_title(f'After Bandpass: Time Domain ({LOW_FREQ}-{HIGH_FREQ} Hz)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude (µV)')
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: Power spectrum BEFORE
        freqs_before, psd_before = signal.welch(
            data_before[demo_ch, :], fs=sfreq,
            nperseg=min(n_samples, int(sfreq * 2))
        )
        axes[1, 0].semilogy(freqs_before, psd_before, 'b', linewidth=0.8)
        axes[1, 0].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7, label=f'{LOW_FREQ} Hz')
        axes[1, 0].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7, label=f'{HIGH_FREQ} Hz')
        axes[1, 0].set_title('Before Bandpass: Power Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power (V²/Hz)')
        axes[1, 0].set_xlim([0, min(400, sfreq/2)])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Shade the passband region
        axes[1, 0].axvspan(LOW_FREQ, HIGH_FREQ, alpha=0.2, color='green')

        # Bottom-right: Power spectrum AFTER
        freqs_after, psd_after = signal.welch(
            data_after[demo_ch, :], fs=sfreq,
            nperseg=min(n_samples, int(sfreq * 2))
        )
        axes[1, 1].semilogy(freqs_after, psd_after, 'g', linewidth=0.8)
        axes[1, 1].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('After Bandpass: Power Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Power (V²/Hz)')
        axes[1, 1].set_xlim([0, min(400, sfreq/2)])
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvspan(LOW_FREQ, HIGH_FREQ, alpha=0.2, color='green')

        plt.tight_layout()
        fig1_path = OUTPUT_DIR / "bandpass_demo.png"
        plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig1_path}")
        plt.close()

        # ====================================================================
        # FIGURE 2: Frequency bands comparison
        # ====================================================================

        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle(f'Frequency Band Analysis: {rec_name}',
                      fontsize=14, fontweight='bold')

        # Define frequency bands of interest for iEEG
        bands = {
            'Delta (1-4 Hz)': (1, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-80 Hz)': (30, 80),
            'High Gamma (80-150 Hz)': (80, 150),
            'HFO (150-250 Hz)': (150, 250),
        }

        # Compute band powers before and after for all channels
        band_power_before = {}
        band_power_after = {}

        for band_name, (fmin, fmax) in bands.items():
            # Find frequency indices
            freq_mask = (freqs_before >= fmin) & (freqs_before <= fmax)
            if np.any(freq_mask):
                band_power_before[band_name] = np.mean(psd_before[freq_mask])
                band_power_after[band_name] = np.mean(psd_after[freq_mask])
            else:
                band_power_before[band_name] = 0
                band_power_after[band_name] = 0

        # Top-left: Band powers comparison
        x = np.arange(len(bands))
        width = 0.35
        band_names = list(bands.keys())

        powers_before = [band_power_before[b] for b in band_names]
        powers_after = [band_power_after[b] for b in band_names]

        axes2[0, 0].bar(x - width/2, powers_before, width, label='Before BP', color='blue', alpha=0.7)
        axes2[0, 0].bar(x + width/2, powers_after, width, label='After BP', color='green', alpha=0.7)
        axes2[0, 0].set_xlabel('Frequency Band')
        axes2[0, 0].set_ylabel('Power (V²/Hz)')
        axes2[0, 0].set_title('Band Powers Before/After Bandpass')
        axes2[0, 0].set_xticks(x)
        axes2[0, 0].set_xticklabels([b.split(' ')[0] for b in band_names], rotation=45)
        axes2[0, 0].legend()
        axes2[0, 0].set_yscale('log')
        axes2[0, 0].grid(True, alpha=0.3)

        # Top-right: Ratio (after/before)
        ratios = [powers_after[i] / (powers_before[i] + 1e-20) for i in range(len(bands))]
        colors = ['green' if r > 0.5 else 'red' for r in ratios]
        axes2[1, 0].bar(x, ratios, color=colors, alpha=0.7)
        axes2[1, 0].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        axes2[1, 0].set_xlabel('Frequency Band')
        axes2[1, 0].set_ylabel('Power Ratio (After/Before)')
        axes2[1, 0].set_title('Power Retention by Band')
        axes2[1, 0].set_xticks(x)
        axes2[1, 0].set_xticklabels([b.split(' ')[0] for b in band_names], rotation=45)
        axes2[1, 0].grid(True, alpha=0.3)

        # Bottom: Multi-channel spectrogram comparison
        # Average spectrum across channels
        psd_multi_before = []
        psd_multi_after = []

        for ch in range(min(20, n_channels)):
            _, psd_b = signal.welch(data_before[ch, :], fs=sfreq,
                                    nperseg=min(n_samples, int(sfreq * 2)))
            _, psd_a = signal.welch(data_after[ch, :], fs=sfreq,
                                    nperseg=min(n_samples, int(sfreq * 2)))
            psd_multi_before.append(psd_b)
            psd_multi_after.append(psd_a)

        psd_multi_before = np.array(psd_multi_before)
        psd_multi_after = np.array(psd_multi_after)

        # Plot as heatmap
        im1 = axes2[0, 1].imshow(10 * np.log10(psd_multi_before + 1e-20),
                                  aspect='auto', origin='lower',
                                  extent=[0, freqs_before[-1], 0, psd_multi_before.shape[0]],
                                  cmap='viridis')
        axes2[0, 1].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7)
        axes2[0, 1].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7)
        axes2[0, 1].set_xlabel('Frequency (Hz)')
        axes2[0, 1].set_ylabel('Channel')
        axes2[0, 1].set_title('Before Bandpass: Multi-channel Spectrum')
        axes2[0, 1].set_xlim([0, min(300, sfreq/2)])
        plt.colorbar(im1, ax=axes2[0, 1], label='Power (dB)')

        im2 = axes2[1, 1].imshow(10 * np.log10(psd_multi_after + 1e-20),
                                  aspect='auto', origin='lower',
                                  extent=[0, freqs_after[-1], 0, psd_multi_after.shape[0]],
                                  cmap='viridis')
        axes2[1, 1].axvline(LOW_FREQ, color='r', linestyle='--', alpha=0.7)
        axes2[1, 1].axvline(HIGH_FREQ, color='r', linestyle='--', alpha=0.7)
        axes2[1, 1].set_xlabel('Frequency (Hz)')
        axes2[1, 1].set_ylabel('Channel')
        axes2[1, 1].set_title('After Bandpass: Multi-channel Spectrum')
        axes2[1, 1].set_xlim([0, min(300, sfreq/2)])
        plt.colorbar(im2, ax=axes2[1, 1], label='Power (dB)')

        plt.tight_layout()
        fig2_path = OUTPUT_DIR / "bandpass_bands.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig2_path}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for bandpass filtering.

    Steps:
    1. Plot filter response (to verify filter design)
    2. Process all recordings (apply bandpass)
    3. Generate before/after visualization
    """

    t_start = time.perf_counter()

    # Step 1: Visualize filter response
    plot_filter_response()

    # Step 2: Apply bandpass to all recordings
    n_processed = process_all_recordings()

    # Step 3: Generate before/after demo
    if n_processed > 0:
        plot_bandpass_demo()

    # Report timing
    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"Bandpass filtering complete!")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    append_project_log(
        stage="apply_bandpass",
        status="success",
        lines=[
            f"Input: {INPUT_H5}",
            f"Output: {OUTPUT_H5}",
            f"Bandpass: {LOW_FREQ:.1f}-{HIGH_FREQ:.1f} Hz",
            f"Processed recordings: {n_processed}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
