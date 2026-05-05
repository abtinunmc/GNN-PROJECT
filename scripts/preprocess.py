"""
Preprocessing Pipeline for SOZ Localization GNN
Dataset: OpenNeuro ds003029 (HUP iEEG)

Steps:
1. Load BrainVision iEEG data
2. Parse good channels from channels.tsv
3. Apply notch filter (60 Hz line noise)
4. Extract peri-ictal epochs around seizure onset
5. Parse SOZ labels from events
6. Save preprocessed data to HDF5
"""

import os
import csv
import time
import re
import numpy as np
import pandas as pd
import mne
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path(os.environ.get("DS003029_DIR", str(Path.home() / "Desktop" / "ds003029")))
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
H5_DIR = OUTPUT_DIR / "h5"
CSV_DIR = OUTPUT_DIR / "csv"
FIGURES_DIR = OUTPUT_DIR / "figures"

for d in [H5_DIR, CSV_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
NOTCH_FREQ = 60.0     # Hz - line noise (US)

# Epoch parameters (seconds relative to seizure onset)
PRE_SEIZURE = 30      # seconds before onset
POST_SEIZURE = 30     # seconds after onset


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_subject_sessions(data_dir: Path) -> list:
    """Find all subject/session pairs with iEEG data."""
    sessions = []
    for sub_dir in sorted(data_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        for ses_dir in sub_dir.glob("ses-*"):
            ieeg_dir = ses_dir / "ieeg"
            if ieeg_dir.exists():
                vhdr_files = list(ieeg_dir.glob("*_ieeg.vhdr"))
                for vhdr in vhdr_files:
                    sessions.append({
                        "subject": sub_dir.name,
                        "session": ses_dir.name,
                        "vhdr_path": vhdr
                    })
    return sessions


def load_channels_tsv(vhdr_path: Path) -> pd.DataFrame:
    """Load channel information from BIDS channels.tsv."""
    channels_tsv = str(vhdr_path).replace("_ieeg.vhdr", "_channels.tsv")
    if os.path.exists(channels_tsv):
        return pd.read_csv(channels_tsv, sep="\t")
    return None


def load_events_tsv(vhdr_path: Path) -> pd.DataFrame:
    """Load events from BIDS events.tsv."""
    events_tsv = str(vhdr_path).replace("_ieeg.vhdr", "_events.tsv")
    if os.path.exists(events_tsv):
        return pd.read_csv(events_tsv, sep="\t")
    return None


def get_good_channels(channels_df: pd.DataFrame) -> tuple:
    """
    Extract good ECOG/SEEG channels from channels.tsv.
    Returns (good_names, bad_names).
    """
    if channels_df is None:
        return [], []

    # Filter for ECOG or SEEG type channels
    ieeg_mask = channels_df["type"].isin(["ECOG", "SEEG"])
    ieeg_df = channels_df[ieeg_mask]

    good_names = ieeg_df[ieeg_df["status"] == "good"]["name"].tolist()
    bad_names = ieeg_df[ieeg_df["status"] == "bad"]["name"].tolist()

    return good_names, bad_names


def get_seizure_onset(events_df: pd.DataFrame) -> float:
    """Extract seizure onset time from events.tsv."""
    if events_df is None:
        return None

    # Look for seizure event markers
    sz_keywords = ["SZ EVENT", "SEIZURE", "SZ START", "ONSET"]
    for _, row in events_df.iterrows():
        trial_type = str(row.get("trial_type", "")).upper()
        if any(kw in trial_type for kw in sz_keywords):
            return row["onset"]

    return None


def parse_soz_channels(events_df: pd.DataFrame, all_channels: list) -> list:
    """
    Parse SOZ channels from event annotations.

    The events contain annotations like "Z SWC LAT1-2, 6-7" indicating
    which electrodes show seizure activity.
    """
    if events_df is None:
        return []

    def split_channel_name(ch_name: str) -> tuple[str | None, int | None]:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", ch_name.upper())
        if not match:
            return None, None
        return match.group(1), int(match.group(2))

    channels_by_prefix: dict[str, list[tuple[int, str]]] = {}
    ordered_prefixes = []

    for ch in all_channels:
        prefix, number = split_channel_name(ch)
        if prefix is None:
            continue
        channels_by_prefix.setdefault(prefix, []).append((number, ch))

    ordered_prefixes = sorted(channels_by_prefix.keys(), key=len, reverse=True)
    prefix_channel_maps = {
        prefix: {num: ch for num, ch in entries}
        for prefix, entries in channels_by_prefix.items()
    }

    soz_channels = set()

    for _, row in events_df.iterrows():
        trial_type = str(row.get("trial_type", "")).upper()

        # Look for electrode mentions in seizure annotations (Z prefix)
        if not trial_type.startswith("Z "):
            continue

        annotation = trial_type[2:]

        for prefix in ordered_prefixes:
            pattern = rf"(?<![A-Z0-9]){re.escape(prefix)}(?=(?:\d|[^A-Z0-9]|$))"
            matches = list(re.finditer(pattern, annotation))
            if not matches:
                continue

            prefix_numbers = set()
            matched_any_numbers = False

            for match in matches:
                tail = annotation[match.end():]
                parsed_numbers = []

                range_match = re.match(r"\s*(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*)", tail)
                if range_match:
                    matched_any_numbers = True
                    number_text = range_match.group(1)
                    for part in re.split(r"\s*,\s*", number_text):
                        if "-" in part:
                            start_num, end_num = part.split("-", 1)
                            start_num = int(start_num)
                            end_num = int(end_num)
                            if end_num < start_num:
                                start_num, end_num = end_num, start_num
                            parsed_numbers.extend(range(start_num, end_num + 1))
                        else:
                            parsed_numbers.append(int(part))

                if parsed_numbers:
                    prefix_numbers.update(parsed_numbers)

            if matched_any_numbers and prefix_numbers:
                for number in sorted(prefix_numbers):
                    channel_name = prefix_channel_maps[prefix].get(number)
                    if channel_name is not None:
                        soz_channels.add(channel_name)
            elif not matched_any_numbers:
                # If an electrode group is mentioned without specific contacts,
                # assign the whole prefix group rather than using unsafe substring matches.
                for _, channel_name in channels_by_prefix[prefix]:
                    soz_channels.add(channel_name)

    return list(soz_channels)


def preprocess_raw(raw: mne.io.Raw, good_channels: list, bad_channels: list) -> mne.io.Raw:
    """
    Apply preprocessing steps to raw data.
    """
    # Mark bad channels
    raw.info['bads'] = [ch for ch in bad_channels if ch in raw.ch_names]

    # Load data into memory
    raw.load_data()

    # Apply notch filter for line noise (60 Hz and harmonics up to 240 Hz)
    # Only use frequencies below Nyquist
    nyquist = raw.info['sfreq'] / 2
    notch_freqs = [f for f in [NOTCH_FREQ, NOTCH_FREQ*2, NOTCH_FREQ*3, NOTCH_FREQ*4] if f < nyquist]
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, picks='all', verbose=False)

    return raw


def extract_epoch(raw: mne.io.Raw, onset: float, pre: float, post: float) -> np.ndarray:
    """Extract a single epoch around seizure onset."""
    sfreq = raw.info['sfreq']

    start_sample = int((onset - pre) * sfreq)
    end_sample = int((onset + post) * sfreq)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(raw.times), end_sample)

    data = raw.get_data(start=start_sample, stop=end_sample)

    return data


def intersect_intervals(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Two-pointer merge to find overlapping time between two sorted interval sets.
    From HFO-GLISKE pipeline.
    """
    if len(A) == 0 or len(B) == 0:
        return np.empty((0, 2))

    out = []
    i, j = 0, 0
    while i < len(A) and j < len(B):
        lo = max(A[i, 0], B[j, 0])
        hi = min(A[i, 1], B[j, 1])
        if lo < hi:
            out.append([lo, hi])
        if A[i, 1] < B[j, 1]:
            i += 1
        else:
            j += 1
    return np.array(out).reshape(-1, 2) if out else np.empty((0, 2))


def compute_valid_times_per_channel(data: np.ndarray, sfreq: float,
                                     threshold_std: float = 5.0,
                                     window_sec: float = 1.0) -> list:
    """
    Compute valid time intervals per channel.

    Returns list of arrays, one per channel, each with shape (n_intervals, 2).
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    if window_samples <= 0:
        return [np.empty((0, 2)) for _ in range(n_channels)]

    window_bounds = []
    start = 0
    while start < n_samples:
        end = min(start + window_samples, n_samples)
        window_bounds.append((start, end))
        start = end

    n_windows = len(window_bounds)

    # Compute global std per channel
    global_std = np.std(data, axis=1)

    per_channel_intervals = []

    for ch in range(n_channels):
        valid_windows = np.ones(n_windows, dtype=bool)
        ch_data = data[ch, :]
        ch_global_std = global_std[ch]

        for w in range(n_windows):
            start, end = window_bounds[w]
            window_data = ch_data[start:end]
            window_std = np.std(window_data)

            # Reject if extreme variance
            if window_std > threshold_std * ch_global_std:
                valid_windows[w] = False
            # Reject if flat
            if ch_global_std > 0 and window_std < 0.01 * ch_global_std:
                valid_windows[w] = False

        # Convert to intervals
        intervals = []
        in_valid = False
        start_idx = 0

        for w in range(n_windows):
            window_start, window_end = window_bounds[w]
            if valid_windows[w] and not in_valid:
                start_idx = window_start
                in_valid = True
            elif not valid_windows[w] and in_valid:
                intervals.append([start_idx / sfreq, window_start / sfreq])
                in_valid = False

        if in_valid:
            intervals.append([start_idx / sfreq, window_bounds[-1][1] / sfreq])

        per_channel_intervals.append(
            np.array(intervals) if intervals else np.empty((0, 2))
        )

    return per_channel_intervals


def compute_global_valid_times(data: np.ndarray, sfreq: float,
                                threshold_std: float = 5.0,
                                window_sec: float = 1.0) -> tuple:
    """
    Compute globally valid time intervals where ALL channels have clean signal.
    Similar to HFO-GLISKE validTimes intersection approach.

    Returns:
        global_valid: array of shape (n_intervals, 2) - times where all channels are valid
        per_channel_valid: list of arrays, one per channel
    """
    # Get per-channel valid intervals
    per_channel_valid = compute_valid_times_per_channel(
        data, sfreq, threshold_std, window_sec
    )

    n_channels = len(per_channel_valid)
    if n_channels == 0:
        return np.empty((0, 2)), []

    # Intersect across all channels
    global_valid = per_channel_valid[0].copy()
    for ch in range(1, n_channels):
        global_valid = intersect_intervals(global_valid, per_channel_valid[ch])
        if len(global_valid) == 0:
            break

    return global_valid, per_channel_valid


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_recording(session_info: dict) -> dict:
    """Process a single iEEG recording."""
    vhdr_path = session_info["vhdr_path"]
    subject = session_info["subject"]

    # Parse run from filename
    run_match = vhdr_path.stem.split("_")
    run = [p for p in run_match if p.startswith("run-")]
    run = run[0] if run else "run-01"

    result = {
        "subject": subject,
        "session": session_info["session"],
        "run": run,
        "name": vhdr_path.stem,
        "success": False,
        "n_channels": 0,
        "n_good_channels": 0,
        "n_samples": 0,
        "sfreq": 0,
        "soz_channels": [],
        "valid_times": None,
        "failure_reason": "",
    }

    try:
        # Load metadata
        channels_df = load_channels_tsv(vhdr_path)
        events_df = load_events_tsv(vhdr_path)

        # Get good/bad channels from channels.tsv
        good_channels, bad_channels = get_good_channels(channels_df)

        if len(good_channels) == 0:
            print(f"  No good channels found in {vhdr_path.name}")
            result["failure_reason"] = "no_good_channels"
            return result

        # Get seizure onset
        onset = get_seizure_onset(events_df)
        if onset is None:
            print(f"  No seizure onset found in {vhdr_path.name}")
            result["failure_reason"] = "no_seizure_onset"
            return result

        # Load raw data
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose=False)

        # Validate that the annotated seizure onset is inside the recording.
        duration_sec = raw.n_times / raw.info["sfreq"]
        if onset < 0 or onset >= duration_sec:
            print(
                f"  Invalid onset for {vhdr_path.name}: onset={onset:.3f}s, "
                f"duration={duration_sec:.3f}s"
            )
            result["sfreq"] = raw.info["sfreq"]
            result["failure_reason"] = "invalid_onset_outside_recording"
            result["recording_duration_sec"] = duration_sec
            result["onset"] = onset
            return result

        # Get channel info before preprocessing
        all_channels = raw.ch_names

        # Parse SOZ channels from annotations
        soz_channels = parse_soz_channels(events_df, all_channels)

        # Preprocess raw data (notch filtering and bad-channel marking)
        raw = preprocess_raw(raw, good_channels, bad_channels)

        # Pick only good iEEG channels for output
        good_in_raw = [ch for ch in good_channels if ch in raw.ch_names]
        raw = raw.pick(good_in_raw)

        # Extract epoch around seizure
        data = extract_epoch(raw, onset, PRE_SEIZURE, POST_SEIZURE)

        # Compute valid time intervals (global and per-channel)
        global_valid, per_channel_valid = compute_global_valid_times(
            data, raw.info['sfreq']
        )

        # Calculate total valid duration
        if len(global_valid) > 0:
            total_valid_sec = np.sum(global_valid[:, 1] - global_valid[:, 0])
        else:
            total_valid_sec = 0

        # Update result
        result["success"] = True
        result["n_channels"] = data.shape[0]
        result["n_good_channels"] = len(good_in_raw)
        result["n_samples"] = data.shape[1]
        result["sfreq"] = raw.info['sfreq']
        result["soz_channels"] = [ch for ch in soz_channels if ch in good_in_raw]
        result["global_valid_times"] = global_valid
        result["per_channel_valid_times"] = per_channel_valid
        result["total_valid_sec"] = total_valid_sec
        result["data"] = data
        result["channels"] = raw.ch_names
        result["onset"] = onset

    except Exception as e:
        print(f"  Error processing {vhdr_path.name}: {e}")
        result["failure_reason"] = f"exception: {e}"

    return result


def save_to_hdf5(results: list, output_path: Path):
    """Save all preprocessed data to HDF5."""
    with h5py.File(output_path, 'w') as f:
        for res in results:
            if not res["success"]:
                continue

            grp_name = res.get("name", f"{res['subject']}_{res['session']}_{res['run']}")
            grp = f.create_group(grp_name)

            # Store data
            grp.create_dataset("data", data=res["data"], compression="gzip")

            # Store global valid times (where ALL channels are clean)
            if res["global_valid_times"] is not None and len(res["global_valid_times"]) > 0:
                grp.create_dataset("global_valid_times", data=res["global_valid_times"])

            # Attributes
            grp.attrs["channels"] = res["channels"]
            grp.attrs["sfreq"] = res["sfreq"]
            grp.attrs["onset"] = res["onset"]
            grp.attrs["soz_channels"] = res["soz_channels"]
            grp.attrs["n_channels"] = res["n_channels"]
            grp.attrs["n_good_channels"] = res["n_good_channels"]
            grp.attrs["n_samples"] = res["n_samples"]
            grp.attrs["total_valid_sec"] = res["total_valid_sec"]

    print(f"\nSaved to {output_path}")


def plot_preprocessing_demo(data_dir: Path, output_dir: Path):
    """
    Generate before/after plots showing preprocessing effect.
    Uses the first available recording as an example.
    """
    print("\n" + "=" * 60)
    print("Generating Preprocessing Visualization")
    print("=" * 60)

    # Find first valid recording
    sessions = find_subject_sessions(data_dir)
    demo_session = None

    for session_info in sessions:
        events_df = load_events_tsv(session_info["vhdr_path"])
        onset = get_seizure_onset(events_df)
        if onset is not None:
            demo_session = session_info
            break

    if demo_session is None:
        print("No valid recording found for demo")
        return

    vhdr_path = demo_session["vhdr_path"]
    print(f"\nUsing: {demo_session['subject']} / {vhdr_path.name}")

    # Load metadata
    channels_df = load_channels_tsv(vhdr_path)
    events_df = load_events_tsv(vhdr_path)
    onset = get_seizure_onset(events_df)
    good_channels, bad_channels = get_good_channels(channels_df)

    # Load raw data (before processing)
    raw_before = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    sfreq = raw_before.info['sfreq']

    # Pick a good channel for visualization
    demo_channel = good_channels[0] if good_channels else raw_before.ch_names[0]
    ch_idx = raw_before.ch_names.index(demo_channel)

    # Extract 10 seconds around seizure onset for visualization
    viz_duration = 10  # seconds
    start_sample = int((onset - viz_duration / 2) * sfreq)
    end_sample = int((onset + viz_duration / 2) * sfreq)
    start_sample = max(0, start_sample)
    end_sample = min(raw_before.n_times, end_sample)

    # Get data before processing
    data_before = raw_before.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)[0]
    time_vec = np.arange(len(data_before)) / sfreq

    # Apply notch filter (after processing) - 60, 120, 180, 240 Hz
    raw_after = raw_before.copy()
    nyquist = sfreq / 2
    notch_freqs = [f for f in [NOTCH_FREQ, NOTCH_FREQ * 2, NOTCH_FREQ * 3, NOTCH_FREQ * 4] if f < nyquist]
    if notch_freqs:
        raw_after.notch_filter(freqs=notch_freqs, picks='all', verbose=False)
    data_after = raw_after.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)[0]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Preprocessing Demo: {demo_session["subject"]} - Channel: {demo_channel}',
                 fontsize=14, fontweight='bold')

    # Top left: Raw signal (before)
    axes[0, 0].plot(time_vec, data_before * 1e6, 'b', linewidth=0.5)
    axes[0, 0].set_title('Before: Raw Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (uV)')
    axes[0, 0].axvline(viz_duration / 2, color='red', linestyle='--', alpha=0.7, label='Seizure onset')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # Top right: After notch filter
    axes[0, 1].plot(time_vec, data_after * 1e6, 'g', linewidth=0.5)
    axes[0, 1].set_title('After: Notch Filter (60 Hz + harmonics)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (uV)')
    axes[0, 1].axvline(viz_duration / 2, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom left: Power spectrum before
    freqs_before, psd_before = signal.welch(data_before, sfreq, nperseg=min(len(data_before), int(sfreq * 2)))
    axes[1, 0].semilogy(freqs_before, psd_before, 'b', linewidth=0.8)
    axes[1, 0].set_title('Before: Power Spectrum')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power (V^2/Hz)')
    axes[1, 0].set_xlim([0, min(150, nyquist)])
    axes[1, 0].axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
    axes[1, 0].axvline(120, color='orange', linestyle='--', alpha=0.5, label='120 Hz')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom right: Power spectrum after
    freqs_after, psd_after = signal.welch(data_after, sfreq, nperseg=min(len(data_after), int(sfreq * 2)))
    axes[1, 1].semilogy(freqs_after, psd_after, 'g', linewidth=0.8)
    axes[1, 1].set_title('After: Power Spectrum (60 Hz removed)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power (V^2/Hz)')
    axes[1, 1].set_xlim([0, min(150, nyquist)])
    axes[1, 1].axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
    axes[1, 1].axvline(120, color='orange', linestyle='--', alpha=0.5, label='120 Hz')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "preprocessing_demo.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")
    plt.close()

    # Multi-channel view
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    fig2.suptitle(f'Multi-Channel View: Before vs After\n{demo_session["subject"]}',
                  fontsize=14, fontweight='bold')

    # Show first 10 good channels
    n_show = min(10, len(good_channels))
    show_channels = good_channels[:n_show]

    # Before (raw)
    for i, ch in enumerate(show_channels):
        if ch in raw_before.ch_names:
            ch_idx_i = raw_before.ch_names.index(ch)
            data_i = raw_before.get_data(picks=[ch_idx_i], start=start_sample, stop=end_sample)[0]
            offset = i * np.std(data_i) * 8
            axes2[0].plot(time_vec, data_i * 1e6 + offset * 1e6, linewidth=0.5, label=ch)
    axes2[0].set_title('Before Preprocessing (Raw)')
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Channels (offset for visibility)')
    axes2[0].axvline(viz_duration / 2, color='red', linestyle='--', alpha=0.7, label='Seizure onset')
    axes2[0].set_yticks([])
    axes2[0].grid(True, alpha=0.3)

    # After (notch filtered)
    for i, ch in enumerate(show_channels):
        if ch in raw_after.ch_names:
            ch_idx_i = raw_after.ch_names.index(ch)
            data_i = raw_after.get_data(picks=[ch_idx_i], start=start_sample, stop=end_sample)[0]
            offset = i * np.std(data_i) * 8
            axes2[1].plot(time_vec, data_i * 1e6 + offset * 1e6, linewidth=0.5, label=ch)
    axes2[1].set_title('After Preprocessing (Notch Filtered)')
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Channels (offset for visibility)')
    axes2[1].axvline(viz_duration / 2, color='red', linestyle='--', alpha=0.7)
    axes2[1].set_yticks([])
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()

    fig2_path = output_dir / "preprocessing_multichannel.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-channel view to {fig2_path}")
    plt.close()


def load_participant_outcomes(data_dir: Path) -> dict:
    """Load surgical outcomes from participants.tsv."""
    participants_tsv = data_dir / "participants.tsv"
    if not participants_tsv.exists():
        return {}
        
    try:
        df = pd.read_csv(participants_tsv, sep="\t")
    except Exception:
        return {}

    outcomes = {}
    for _, row in df.iterrows():
        if "participant_id" not in row:
            continue
        outcomes[row["participant_id"]] = {
            "outcome": row.get("outcome", ""),
            "engel_score": row.get("engel_score", ""),
            "ilae_score": row.get("ilae_score", "")
        }
    return outcomes


def main():
    print("=" * 60)
    print("SOZ Localization GNN - Preprocessing Pipeline")
    print("(Notch 60 Hz)")
    print("=" * 60)

    t_start = time.perf_counter()

    # Find all sessions
    print(f"\nScanning {DATA_DIR} for iEEG data...")
    sessions = find_subject_sessions(DATA_DIR)
    print(f"Found {len(sessions)} recordings from {len(set(s['subject'] for s in sessions))} subjects")

    # Load outcomes
    outcomes = load_participant_outcomes(DATA_DIR)

    # Process each recording
    results = []
    for session_info in tqdm(sessions, desc="Processing"):
        print(f"\n{session_info['subject']} / {session_info['vhdr_path'].name}")
        result = process_single_recording(session_info)

        # Add outcome info
        sub_id = session_info["subject"]
        if sub_id in outcomes:
            result["outcome"] = outcomes[sub_id]

        results.append(result)

    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    valid_times_count = 0
    total_valid_sec = 0.0
    print(f"\n{'=' * 60}")
    print(f"Successfully processed: {len(successful)}/{len(results)} recordings")

    if failed:
        print("Failed recordings:")
        for r in failed:
            rec_id = f"{r['subject']}_{r['session']}_{r['run']}"
            reason = r.get("failure_reason", "unknown")
            print(f"  - {rec_id}: {reason}")

    if successful:
        total_channels = sum(r["n_channels"] for r in successful)
        print(f"Total channels: {total_channels}")
        print(f"Sampling frequency: {successful[0]['sfreq']} Hz")

        soz_count = sum(1 for r in successful if r["soz_channels"])
        print(f"Recordings with SOZ annotations: {soz_count}")

        # Global valid times stats
        valid_times_count = sum(
            1 for r in successful
            if r["global_valid_times"] is not None and len(r["global_valid_times"]) > 0
        )
        total_valid_sec = sum(r["total_valid_sec"] for r in successful)
        print(f"Recordings with global valid time intervals: {valid_times_count}")
        print(f"Total valid signal time (all channels clean): {total_valid_sec:.1f}s")

        # Save to HDF5
        output_path = H5_DIR / "preprocessed_ieeg.h5"
        save_to_hdf5(successful, output_path)

        # Save metadata CSV
        meta_df = pd.DataFrame([{
            "subject": r["subject"],
            "session": r["session"],
            "run": r["run"],
            "n_channels": r["n_channels"],
            "n_good_channels": r["n_good_channels"],
            "n_samples": r["n_samples"],
            "sfreq": r["sfreq"],
            "n_global_valid_intervals": len(r["global_valid_times"]) if r["global_valid_times"] is not None else 0,
            "total_valid_sec": r["total_valid_sec"],
            "soz_channels": ";".join(r["soz_channels"]) if r["soz_channels"] else "",
            "outcome": r.get("outcome", {}).get("outcome", ""),
            "engel": r.get("outcome", {}).get("engel_score", ""),
        } for r in successful])

        meta_path = CSV_DIR / "metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"Saved metadata to {meta_path}")

    if failed:
        failed_df = pd.DataFrame([{
            "subject": r["subject"],
            "session": r["session"],
            "run": r["run"],
            "sfreq": r.get("sfreq", ""),
            "onset": r.get("onset", ""),
            "recording_duration_sec": r.get("recording_duration_sec", ""),
            "failure_reason": r.get("failure_reason", ""),
        } for r in failed])
        failed_path = CSV_DIR / "failed_recordings.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Saved failed-recordings QC to {failed_path}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal runtime: {elapsed:.1f}s")

    append_project_log(
        stage="preprocess",
        status="success",
        lines=[
            f"Dataset dir: {DATA_DIR}",
            f"Output dir: {OUTPUT_DIR}",
            f"Processed recordings: {len(successful)}/{len(results)}",
            f"Failed recordings: {len(failed)}",
            f"Recordings with SOZ annotations: {sum(1 for r in successful if r['soz_channels'])}",
            f"Recordings with valid intervals: {valid_times_count}",
            f"Total valid signal time (s): {total_valid_sec:.1f}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )

    # Generate visualization
    plot_preprocessing_demo(DATA_DIR, FIGURES_DIR)


if __name__ == "__main__":
    main()
