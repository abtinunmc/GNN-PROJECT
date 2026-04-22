"""
Common Average Reference (CAR) Application for SOZ Localization GNN
====================================================================

This script applies Common Average Reference (CAR) to the preprocessed iEEG data.

What is CAR?
------------
CAR is a spatial filtering technique that removes the common signal present across
all electrodes. For each time point, we compute the mean across all channels and
subtract it from each channel. This removes:
- Common noise sources (e.g., movement artifacts, distant physiological noise)
- Volume conduction effects
- Reference electrode bias

Mathematical formulation:
    x_car[ch, t] = x_raw[ch, t] - mean(x_raw[:, t])

Why use CAR?
------------
1. Removes global noise while preserving local neural activity
2. Standard practice in iEEG/ECoG analysis
3. Improves signal-to-noise ratio for high-frequency activity
4. Helps isolate focal activity (important for SOZ localization)

Input:
    data/processed/h5/preprocessed_ieeg.h5 (from preprocess.py)

Output:
    data/processed/h5/preprocessed_ieeg_car.h5 (CAR-referenced data)
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

# Input: preprocessed data from preprocess.py (notch filtered only)
INPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg.h5"

# Output: CAR-referenced data
OUTPUT_H5 = PROJECT_DIR / "data" / "processed" / "h5" / "preprocessed_ieeg_car.h5"

# Output directory for visualizations
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "figures"


# ============================================================================
# CAR FUNCTIONS
# ============================================================================

def apply_car(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to multi-channel data.

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_channels, n_samples).
        Each row is a channel, each column is a time point.

    Returns
    -------
    data_car : np.ndarray
        CAR-referenced data with same shape as input.

    Algorithm
    ---------
    For each time point t:
        1. Compute the mean across all channels: mean_t = mean(data[:, t])
        2. Subtract this mean from each channel: data_car[ch, t] = data[ch, t] - mean_t

    This is equivalent to: data_car = data - mean(data, axis=0, keepdims=True)

    Notes
    -----
    - We use keepdims=True to maintain the shape for broadcasting
    - The mean is computed using only the good channels (bad channels were
      already excluded in preprocessing)
    - This assumes all channels in the data array are valid/good channels
    """

    # Get the number of channels and samples for reference
    n_channels, n_samples = data.shape

    # Step 1: Compute the mean across all channels for each time point
    # axis=0 means we average along the channel dimension
    # keepdims=True keeps the result as shape (1, n_samples) for broadcasting
    common_average = np.mean(data, axis=0, keepdims=True)

    # Step 2: Subtract the common average from each channel
    # Broadcasting: (n_channels, n_samples) - (1, n_samples) = (n_channels, n_samples)
    data_car = data - common_average

    return data_car


def apply_car_weighted(data: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Apply weighted CAR (optional alternative approach).

    In standard CAR, all channels contribute equally to the average.
    In weighted CAR, channels can have different weights based on:
    - Signal quality
    - Electrode impedance
    - Distance from region of interest

    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_channels, n_samples).
    weights : np.ndarray, optional
        Weight for each channel, shape (n_channels,).
        If None, uses uniform weights (standard CAR).

    Returns
    -------
    data_car : np.ndarray
        Weighted CAR-referenced data.

    Notes
    -----
    This is provided for completeness but we use standard CAR in this pipeline.
    """
    n_channels, n_samples = data.shape

    # Default: uniform weights (equivalent to standard CAR)
    if weights is None:
        weights = np.ones(n_channels) / n_channels
    else:
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)

    # Compute weighted average: weights are (n_channels,), need to reshape
    # weights[:, np.newaxis] makes it (n_channels, 1) for proper multiplication
    weighted_avg = np.sum(data * weights[:, np.newaxis], axis=0, keepdims=True)

    # Subtract weighted average from each channel
    data_car = data - weighted_avg

    return data_car


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_all_recordings():
    """
    Apply CAR to all recordings in the preprocessed HDF5 file.

    Workflow
    --------
    1. Open input HDF5 (read-only)
    2. Create output HDF5 (write)
    3. For each recording:
       a. Read the preprocessed data
       b. Apply CAR
       c. Save CAR-referenced data with same metadata
    4. Close files
    """

    print("=" * 70)
    print("Applying Common Average Reference (CAR)")
    print("=" * 70)
    print(f"\nInput:  {INPUT_H5}")
    print(f"Output: {OUTPUT_H5}")

    # Check input exists
    if not INPUT_H5.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_H5}")

    # Track statistics for reporting
    total_recordings = 0
    processed_recordings = 0

    # Open input file for reading, output file for writing
    with h5py.File(INPUT_H5, 'r') as f_in, h5py.File(OUTPUT_H5, 'w') as f_out:

        # Get list of all recording groups in the input file
        recording_names = list(f_in.keys())
        total_recordings = len(recording_names)

        print(f"\nFound {total_recordings} recordings to process")
        print("-" * 70)

        # Process each recording with progress bar
        for rec_name in tqdm(recording_names, desc="Applying CAR"):

            # Get the input group (contains data and attributes)
            grp_in = f_in[rec_name]

            # Read the preprocessed data
            # Shape: (n_channels, n_samples)
            data = grp_in['data'][:]

            # Get metadata from attributes
            channels = grp_in.attrs['channels']
            sfreq = grp_in.attrs['sfreq']
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
            # APPLY CAR
            # ----------------------------------------------------------------
            # This is the core operation: subtract common average from each channel
            data_car = apply_car(data)

            # ----------------------------------------------------------------
            # SAVE TO OUTPUT FILE
            # ----------------------------------------------------------------

            # Create output group with same name
            grp_out = f_out.create_group(rec_name)

            # Save CAR-referenced data (compressed to save space)
            grp_out.create_dataset('data', data=data_car, compression='gzip')

            # Copy global valid times (unchanged by CAR)
            if global_valid_times is not None and len(global_valid_times) > 0:
                grp_out.create_dataset('global_valid_times', data=global_valid_times)

            # Copy all attributes (metadata unchanged by CAR)
            grp_out.attrs['channels'] = channels
            grp_out.attrs['sfreq'] = sfreq
            grp_out.attrs['onset'] = onset
            grp_out.attrs['soz_channels'] = soz_channels
            grp_out.attrs['n_channels'] = n_channels
            grp_out.attrs['n_good_channels'] = n_good_channels
            grp_out.attrs['n_samples'] = n_samples
            grp_out.attrs['total_valid_sec'] = total_valid_sec

            # Add new attribute to indicate CAR was applied
            grp_out.attrs['car_applied'] = True

            processed_recordings += 1

    # Report completion
    print("-" * 70)
    print(f"Successfully processed {processed_recordings}/{total_recordings} recordings")
    print(f"Output saved to: {OUTPUT_H5}")

    return processed_recordings


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_car_demo():
    """
    Generate before/after plots showing the effect of CAR.

    Creates visualization comparing:
    1. Single channel: before vs after CAR
    2. Power spectrum: before vs after CAR
    3. Multi-channel view: showing how common signals are removed
    """

    print("\n" + "=" * 70)
    print("Generating CAR Visualization")
    print("=" * 70)

    # Check both files exist
    if not INPUT_H5.exists() or not OUTPUT_H5.exists():
        print("Input or output HDF5 not found, skipping visualization")
        return

    # Open both files to compare
    with h5py.File(INPUT_H5, 'r') as f_before, h5py.File(OUTPUT_H5, 'r') as f_after:

        # Use first recording for demo
        rec_name = list(f_before.keys())[0]
        print(f"Using recording: {rec_name}")

        # Read data before and after CAR
        data_before = f_before[rec_name]['data'][:]
        data_after = f_after[rec_name]['data'][:]
        sfreq = f_before[rec_name].attrs['sfreq']
        channels = f_before[rec_name].attrs['channels']

        n_channels, n_samples = data_before.shape

        # Time vector for plotting (5 seconds window for clarity)
        plot_samples = min(int(5 * sfreq), n_samples)
        time_vec = np.arange(plot_samples) / sfreq

        # Select a channel to visualize (first channel)
        demo_ch = 0
        demo_ch_name = channels[demo_ch] if len(channels) > demo_ch else f"Ch{demo_ch}"

        # ====================================================================
        # FIGURE 1: Single channel before/after CAR
        # ====================================================================

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'CAR Effect: {rec_name}\nChannel: {demo_ch_name}',
                     fontsize=14, fontweight='bold')

        # Top-left: Time domain BEFORE CAR
        axes[0, 0].plot(time_vec, data_before[demo_ch, :plot_samples] * 1e6,
                        'b', linewidth=0.5)
        axes[0, 0].set_title('Before CAR: Time Domain')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude (µV)')
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: Time domain AFTER CAR
        axes[0, 1].plot(time_vec, data_after[demo_ch, :plot_samples] * 1e6,
                        'g', linewidth=0.5)
        axes[0, 1].set_title('After CAR: Time Domain')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Amplitude (µV)')
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: Power spectrum BEFORE CAR
        # Using Welch's method for spectral estimation
        freqs_before, psd_before = signal.welch(
            data_before[demo_ch, :],  # Use full data for spectrum
            fs=sfreq,
            nperseg=min(n_samples, int(sfreq * 2))  # 2-second windows
        )
        axes[1, 0].semilogy(freqs_before, psd_before, 'b', linewidth=0.8)
        axes[1, 0].set_title('Before CAR: Power Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power (V²/Hz)')
        axes[1, 0].set_xlim([0, min(200, sfreq/2)])
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: Power spectrum AFTER CAR
        freqs_after, psd_after = signal.welch(
            data_after[demo_ch, :],
            fs=sfreq,
            nperseg=min(n_samples, int(sfreq * 2))
        )
        axes[1, 1].semilogy(freqs_after, psd_after, 'g', linewidth=0.8)
        axes[1, 1].set_title('After CAR: Power Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Power (V²/Hz)')
        axes[1, 1].set_xlim([0, min(200, sfreq/2)])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig1_path = OUTPUT_DIR / "car_demo.png"
        plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig1_path}")
        plt.close()

        # ====================================================================
        # FIGURE 2: Multi-channel comparison
        # ====================================================================

        fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))
        fig2.suptitle(f'Multi-Channel CAR Effect: {rec_name}',
                      fontsize=14, fontweight='bold')

        # Number of channels to show (max 10 for visibility)
        n_show = min(10, n_channels)

        # Top: Before CAR (multiple channels stacked)
        for i in range(n_show):
            # Offset each channel vertically for visualization
            offset = i * np.std(data_before[i, :plot_samples]) * 8
            axes2[0].plot(time_vec,
                          data_before[i, :plot_samples] * 1e6 + offset * 1e6,
                          linewidth=0.5)
        axes2[0].set_title(f'Before CAR: First {n_show} Channels')
        axes2[0].set_xlabel('Time (s)')
        axes2[0].set_ylabel('Channels (offset)')
        axes2[0].set_yticks([])
        axes2[0].grid(True, alpha=0.3)

        # Middle: After CAR (multiple channels stacked)
        for i in range(n_show):
            offset = i * np.std(data_after[i, :plot_samples]) * 8
            axes2[1].plot(time_vec,
                          data_after[i, :plot_samples] * 1e6 + offset * 1e6,
                          linewidth=0.5)
        axes2[1].set_title(f'After CAR: First {n_show} Channels')
        axes2[1].set_xlabel('Time (s)')
        axes2[1].set_ylabel('Channels (offset)')
        axes2[1].set_yticks([])
        axes2[1].grid(True, alpha=0.3)

        # Bottom: The common average that was removed
        # This shows what CAR subtracts from each channel
        common_avg = np.mean(data_before, axis=0)
        axes2[2].plot(time_vec, common_avg[:plot_samples] * 1e6,
                      'r', linewidth=0.8)
        axes2[2].set_title('Common Average (what CAR removes)')
        axes2[2].set_xlabel('Time (s)')
        axes2[2].set_ylabel('Amplitude (µV)')
        axes2[2].grid(True, alpha=0.3)
        axes2[2].fill_between(time_vec, 0, common_avg[:plot_samples] * 1e6,
                              alpha=0.3, color='red')

        plt.tight_layout()
        fig2_path = OUTPUT_DIR / "car_multichannel.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig2_path}")
        plt.close()

        # ====================================================================
        # FIGURE 3: Statistical summary
        # ====================================================================

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        fig3.suptitle('CAR Effect: Signal Statistics', fontsize=14, fontweight='bold')

        # Left: Variance per channel before/after
        var_before = np.var(data_before, axis=1)
        var_after = np.var(data_after, axis=1)

        x = np.arange(min(30, n_channels))  # Show first 30 channels max
        width = 0.35

        axes3[0].bar(x - width/2, var_before[:len(x)] * 1e12, width,
                     label='Before CAR', color='blue', alpha=0.7)
        axes3[0].bar(x + width/2, var_after[:len(x)] * 1e12, width,
                     label='After CAR', color='green', alpha=0.7)
        axes3[0].set_xlabel('Channel Index')
        axes3[0].set_ylabel('Variance (µV²)')
        axes3[0].set_title('Per-Channel Variance')
        axes3[0].legend()
        axes3[0].grid(True, alpha=0.3)

        # Right: Correlation matrix change (shows reduced common signal)
        # Compute correlation between first 10 channels
        n_corr = min(20, n_channels)
        corr_before = np.corrcoef(data_before[:n_corr, :])
        corr_after = np.corrcoef(data_after[:n_corr, :])

        # Show difference: CAR should reduce correlations
        corr_diff = corr_before - corr_after

        im = axes3[1].imshow(corr_diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes3[1].set_title(f'Correlation Reduction (first {n_corr} ch)')
        axes3[1].set_xlabel('Channel')
        axes3[1].set_ylabel('Channel')
        plt.colorbar(im, ax=axes3[1], label='Correlation Change')

        plt.tight_layout()
        fig3_path = OUTPUT_DIR / "car_statistics.png"
        plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig3_path}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for CAR application.

    Steps:
    1. Process all recordings (apply CAR)
    2. Generate visualization plots
    """

    t_start = time.perf_counter()

    # Step 1: Apply CAR to all recordings
    n_processed = process_all_recordings()

    # Step 2: Generate visualization
    if n_processed > 0:
        plot_car_demo()

    # Report timing
    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"CAR application complete!")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    append_project_log(
        stage="apply_car",
        status="success",
        lines=[
            f"Input: {INPUT_H5}",
            f"Output: {OUTPUT_H5}",
            f"Processed recordings: {n_processed}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
