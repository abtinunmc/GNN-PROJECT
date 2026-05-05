#!/bin/bash
set -e

PYTHON=/usr/local/mne-python/1.0.0_0/bin/python3
SCRIPTS_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"

run_stage() {
    local stage="$1"
    local script="$2"
    echo ""
    echo "============================================================"
    echo "STAGE: $stage"
    echo "Started: $(date)"
    echo "============================================================"
    cd "$SCRIPTS_DIR"
    $PYTHON "$script" 2>&1
    echo ""
    echo "DONE: $stage at $(date)"
}

run_stage "2 - CAR"             apply_car.py
run_stage "3 - Bandpass"        apply_bandpass.py
run_stage "4 - Feature Extract" extract_features.py

echo ""
echo "============================================================"
echo "ALL STAGES COMPLETE: $(date)"
echo "============================================================"
