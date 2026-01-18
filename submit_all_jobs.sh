#!/bin/bash
# Script to submit all training jobs for different model configurations
# Usage: ./submit_all_jobs.sh [--muon-only | --ranger-only | --all]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:---all}"

echo "Submitting training jobs..."
echo "============================"

if [[ "$MODE" == "--ranger-only" ]] || [[ "$MODE" == "--all" ]]; then
    echo "Submitting Ranger optimizer jobs..."
    sbatch sbatch.sh ParT full Pythia ranger
    sbatch sbatch.sh ParT full Herwig ranger
    sbatch sbatch.sh ParT_gated full Pythia ranger
    sbatch sbatch.sh ParT_gated_v1 full Pythia ranger
    sbatch sbatch.sh ParT_gated_no_mask full Pythia ranger
    sbatch sbatch.sh ParT_gated full Herwig ranger
    sbatch sbatch.sh ParT_gated_v1 full Herwig ranger
    sbatch sbatch.sh ParT_gated_no_mask full Herwig ranger
fi

if [[ "$MODE" == "--muon-only" ]] || [[ "$MODE" == "--all" ]]; then
    echo "Submitting Muon optimizer jobs..."
    sbatch sbatch.sh ParT full Pythia muon
    sbatch sbatch.sh ParT full Herwig muon
    sbatch sbatch.sh ParT_gated full Pythia muon
    sbatch sbatch.sh ParT_gated_v1 full Pythia muon
    sbatch sbatch.sh ParT_gated_no_mask full Pythia muon
    sbatch sbatch.sh ParT_gated full Herwig muon
    sbatch sbatch.sh ParT_gated_v1 full Herwig muon
    sbatch sbatch.sh ParT_gated_no_mask full Herwig muon
fi

echo "============================"
echo "All jobs submitted successfully!"

