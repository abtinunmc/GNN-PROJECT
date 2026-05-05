"""
Combined Preprocessing Pipeline for SOZ Localization GNN
=========================================================

Processes both OpenNeuro datasets:
- ds003029: BrainVision format, SOZ from events.tsv
- ds004100: EDF format, SOZ from channels.tsv

Output: Unified HDF5 files ready for graph construction
"""

import os
import re
import time
import numpy as np
import pandas as pd
import mne
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from log_utils import append_project_log

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Dataset directories
DS003029_DIR = Path(os.environ.get("DS003029_DIR", str(Path.home() / "Desktop" / "ds003029")))
DS004100_DIR = Path(os.environ.get("DS004100_DIR", str(Path.home() / "Desktop" / "ds004100")))

OUTPUT_DIR = PROJECT_DIR / "data" / "processed_combined"
H5_DIR = OUTPUT_DIR / "h5"
CSV_DIR = OUTPUT_DIR / "csv"

for d in [H5_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
NOTCH_FREQ = 60.0     # Hz - line noise (US)
PRE_SEIZURE = 30      # seconds before onset
POST_SEIZURE = 30     # seconds after onset


# ============================================================================
# DS003029 FUNCTIONS (BrainVision format)
# ============================================================================

def find_ds003029_sessions(data_dir: Path) -> list:
    """Find all subject/session pairs with iEEG data in ds003029."""
    sessions = []
    if not data_dir.exists():
        return sessions

    for sub_dir in sorted(data_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        for ses_dir in sub_dir.glob("ses-*"):
            ieeg_dir = ses_dir / "ieeg"
            if ieeg_dir.exists():
                vhdr_files = list(ieeg_dir.glob("*_ieeg.vhdr"))
                for vhdr in vhdr_files:
                    sessions.append({
                        "dataset": "ds003029",
                        "subject": sub_dir.name,
                        "session": ses_dir.name,
                        "data_path": vhdr,
                        "format": "brainvision"
                    })
    return sessions


def parse_soz_from_events(events_df: pd.DataFrame, all_channels: list) -> list:
    """Parse SOZ channels from ds003029 event annotations."""
    if events_df is None:
        return []

    def split_channel_name(ch_name: str):
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", ch_name.upper())
        if not match:
            return None, None
        return match.group(1), int(match.group(2))

    channels_by_prefix = {}
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
                range_match = re.match(r"\s*(\d+(?:-\d+)?(?:\s*,\s*\d+(?:-\d+)?)*)", tail)
                if range_match:
                    matched_any_numbers = True
                    number_text = range_match.group(1)
                    for part in re.split(r"\s*,\s*", number_text):
                        if "-" in part:
                            start_num, end_num = part.split("-", 1)
                            start_num, end_num = int(start_num), int(end_num)
                            if end_num < start_num:
                                start_num, end_num = end_num, start_num
                            prefix_numbers.update(range(start_num, end_num + 1))
                        else:
                            prefix_numbers.add(int(part))

            if matched_any_numbers and prefix_numbers:
                for number in prefix_numbers:
                    channel_name = prefix_channel_maps[prefix].get(number)
                    if channel_name:
                        soz_channels.add(channel_name)

    return list(soz_channels)


# ============================================================================
# DS004100 FUNCTIONS (EDF format)
# ============================================================================

def find_ds004100_sessions(data_dir: Path) -> list:
    """Find all subject/session pairs with iEEG data in ds004100."""
    sessions = []
    if not data_dir.exists():
        return sessions

    for sub_dir in sorted(data_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        for ses_dir in sub_dir.glob("ses-*"):
            ieeg_dir = ses_dir / "ieeg"
            if ieeg_dir.exists():
                # Look for ictal EDF files (prioritize ictal over interictal)
                edf_files = list(ieeg_dir.glob("*task-ictal*_ieeg.edf"))
                for edf in edf_files:
                    sessions.append({
                        "dataset": "ds004100",
                        "subject": sub_dir.name,
                        "session": ses_dir.name,
                        "data_path": edf,
                        "format": "edf"
                    })
    return sessions


def parse_soz_from_channels_tsv(channels_df: pd.DataFrame) -> list:
    """Parse SOZ channels from ds004100 channels.tsv status_description."""
    if channels_df is None:
        return []

    soz_channels = []
    for _, row in channels_df.iterrows():
        status_desc = str(row.get("status_description", "")).lower()
        if "soz" in status_desc:
            soz_channels.append(row["name"])

    return soz_channels


# ============================================================================
# COMMON FUNCTIONS
# ============================================================================

def load_channels_tsv(data_path: Path, data_format: str) -> pd.DataFrame:
    """Load channel information from BIDS channels.tsv."""
    if data_format == "brainvision":
        channels_tsv = str(data_path).replace("_ieeg.vhdr", "_channels.tsv")
    else:  # edf
        channels_tsv = str(data_path).replace("_ieeg.edf", "_channels.tsv")

    if os.path.exists(channels_tsv):
        return pd.read_csv(channels_tsv, sep="\t")
    return None


def load_events_tsv(data_path: Path, data_format: str) -> pd.DataFrame:
    """Load events from BIDS events.tsv."""
    if data_format == "brainvision":
        events_tsv = str(data_path).replace("_ieeg.vhdr", "_events.tsv")
    else:  # edf
        events_tsv = str(data_path).replace("_ieeg.edf", "_events.tsv")

    if os.path.exists(events_tsv):
        return pd.read_csv(events_tsv, sep="\t")
    return None


def get_good_channels(channels_df: pd.DataFrame) -> tuple:
    """Extract good ECOG/SEEG channels from channels.tsv."""
    if channels_df is None:
        return [], []

    ieeg_mask = channels_df["type"].isin(["ECOG", "SEEG"])
    ieeg_df = channels_df[ieeg_mask]

    good_names = ieeg_df[ieeg_df["status"] == "good"]["name"].tolist()
    bad_names = ieeg_df[ieeg_df["status"] == "bad"]["name"].tolist()

    return good_names, bad_names


def get_seizure_onset(events_df: pd.DataFrame) -> float:
    """Extract seizure onset time from events.tsv."""
    if events_df is None:
        return None

    sz_keywords = ["SZ EVENT", "SEIZURE", "SZ START", "ONSET", "SZ ONSET"]
    for _, row in events_df.iterrows():
        trial_type = str(row.get("trial_type", "")).upper()
        if any(kw in trial_type for kw in sz_keywords):
            return row["onset"]

    return None


def load_raw(data_path: Path, data_format: str) -> mne.io.Raw:
    """Load raw iEEG data."""
    if data_format == "brainvision":
        return mne.io.read_raw_brainvision(data_path, preload=False, verbose=False)
    else:  # edf
        return mne.io.read_raw_edf(data_path, preload=False, verbose=False)


def preprocess_raw(raw: mne.io.Raw, good_channels: list, bad_channels: list) -> mne.io.Raw:
    """Apply preprocessing steps to raw data."""
    raw.info['bads'] = [ch for ch in bad_channels if ch in raw.ch_names]
    raw.load_data()

    nyquist = raw.info['sfreq'] / 2
    notch_freqs = [f for f in [NOTCH_FREQ, NOTCH_FREQ*2, NOTCH_FREQ*3] if f < nyquist]
    if notch_freqs:
        raw.notch_filter(freqs=notch_freqs, picks='all', verbose=False)

    return raw


def extract_epoch(raw: mne.io.Raw, onset: float, pre: float, post: float) -> np.ndarray:
    """Extract a single epoch around seizure onset."""
    sfreq = raw.info['sfreq']
    start_sample = int((onset - pre) * sfreq)
    end_sample = int((onset + post) * sfreq)
    start_sample = max(0, start_sample)
    end_sample = min(len(raw.times), end_sample)
    return raw.get_data(start=start_sample, stop=end_sample)


def load_participants_tsv(data_dir: Path) -> pd.DataFrame:
    """Load participants.tsv for outcome info."""
    tsv_path = data_dir / "participants.tsv"
    if tsv_path.exists():
        return pd.read_csv(tsv_path, sep="\t")
    return None


def process_session(session: dict, participants_df: pd.DataFrame = None) -> dict:
    """Process a single session and return data dict."""
    dataset = session["dataset"]
    subject = session["subject"]
    data_path = session["data_path"]
    data_format = session["format"]

    # Load metadata
    channels_df = load_channels_tsv(data_path, data_format)
    events_df = load_events_tsv(data_path, data_format)

    if channels_df is None:
        return None

    good_channels, bad_channels = get_good_channels(channels_df)
    if len(good_channels) < 5:
        return None

    # Get seizure onset
    onset = get_seizure_onset(events_df)
    if onset is None:
        return None

    # Get SOZ channels based on dataset
    if dataset == "ds003029":
        soz_channels = parse_soz_from_events(events_df, good_channels)
    else:  # ds004100
        soz_channels = parse_soz_from_channels_tsv(channels_df)

    # Filter SOZ to only include good channels
    soz_channels = [ch for ch in soz_channels if ch in good_channels]

    # Load and preprocess raw data
    try:
        raw = load_raw(data_path, data_format)
    except Exception as e:
        print(f"  Error loading {data_path}: {e}")
        return None

    # Pick only good channels that exist in raw
    available_good = [ch for ch in good_channels if ch in raw.ch_names]
    if len(available_good) < 5:
        return None

    raw.pick_channels(available_good)
    raw = preprocess_raw(raw, available_good, bad_channels)

    # Extract epoch
    data = extract_epoch(raw, onset, PRE_SEIZURE, POST_SEIZURE)

    # Get participant info if available
    engel = "n/a"
    outcome = "n/a"
    if participants_df is not None:
        subj_row = participants_df[participants_df["participant_id"].str.strip() == subject]
        if len(subj_row) > 0:
            engel = str(subj_row.iloc[0].get("engel", "n/a"))
            outcome = str(subj_row.iloc[0].get("outcome", "n/a"))

    return {
        "data": data,
        "sfreq": raw.info['sfreq'],
        "channels": available_good,
        "soz_channels": soz_channels,
        "subject": subject,
        "dataset": dataset,
        "engel": engel,
        "outcome": outcome
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Combined Preprocessing Pipeline")
    print("=" * 70)

    t_start = time.perf_counter()

    # Find all sessions from both datasets
    print("\nFinding sessions...")
    sessions_003029 = find_ds003029_sessions(DS003029_DIR)
    sessions_004100 = find_ds004100_sessions(DS004100_DIR)

    print(f"  ds003029: {len(sessions_003029)} recordings")
    print(f"  ds004100: {len(sessions_004100)} recordings")

    all_sessions = sessions_003029 + sessions_004100

    if len(all_sessions) == 0:
        print("No sessions found!")
        return

    # Load participants info
    participants_003029 = load_participants_tsv(DS003029_DIR)
    participants_004100 = load_participants_tsv(DS004100_DIR)

    # Process all sessions
    print("\nProcessing sessions...")
    stats = {
        "total": 0,
        "with_soz": 0,
        "ds003029": 0,
        "ds004100": 0
    }

    h5_path = H5_DIR / "features_combined.h5"

    with h5py.File(h5_path, "w") as h5f:
        for session in tqdm(all_sessions, desc="Processing"):
            dataset = session["dataset"]
            participants = participants_003029 if dataset == "ds003029" else participants_004100

            result = process_session(session, participants)
            if result is None:
                continue

            # Create unique recording name
            rec_name = f"{dataset}_{result['subject']}_{session['session']}"
            run_match = re.search(r"run-(\d+)", str(session["data_path"]))
            if run_match:
                rec_name += f"_run-{run_match.group(1)}"

            # Save to HDF5
            grp = h5f.create_group(rec_name)
            grp.create_dataset("data", data=result["data"].astype(np.float32))
            grp.attrs["sfreq"] = result["sfreq"]
            grp.attrs["channels"] = result["channels"]
            grp.attrs["soz_channels"] = result["soz_channels"]
            grp.attrs["subject"] = result["subject"]
            grp.attrs["dataset"] = result["dataset"]
            grp.attrs["engel"] = result["engel"]
            grp.attrs["outcome"] = result["outcome"]

            stats["total"] += 1
            if len(result["soz_channels"]) > 0:
                stats["with_soz"] += 1
            if dataset == "ds003029":
                stats["ds003029"] += 1
            else:
                stats["ds004100"] += 1

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total recordings processed: {stats['total']}")
    print(f"  - ds003029: {stats['ds003029']}")
    print(f"  - ds004100: {stats['ds004100']}")
    print(f"Recordings with SOZ labels: {stats['with_soz']}")
    print(f"Output: {h5_path}")
    print(f"Runtime: {elapsed:.1f}s")

    # Log
    append_project_log(
        stage="preprocess_combined",
        status="success",
        lines=[
            f"Total recordings: {stats['total']}",
            f"ds003029: {stats['ds003029']}",
            f"ds004100: {stats['ds004100']}",
            f"With SOZ labels: {stats['with_soz']}",
            f"Runtime (s): {elapsed:.1f}",
        ],
    )


if __name__ == "__main__":
    main()
