#!/bin/bash
# =============================================================================
# Launch full grid of scaling law study jobs
#
# Dimensions:
#   Model sizes:    nano, micro, tiny, small, base, large, xlarge  (7)
#   Data budgets:   5M, 10M, 25M, 50M, 100M, 250M, 500M           (7)
#   Sample types:   Pythia, Herwig, Mixed                           (3)
#   Feature types:  kin, kinpid, full                               (3)
#   Pairwise:       1 (with), 0 (without)                           (2)
#
# Full grid = 7 * 7 * 3 * 3 * 2 = 882 jobs
#
# Usage:
#   ./launch_scaling_grid.sh [--dry-run] [--subset SUBSET]
#
# Subsets:
#   all         Full grid (882 jobs)
#   baseline    Pythia/full/pair only (49 jobs) - matches v1 study
#   sample      All sample types, full features, pair (147 jobs)
#   features    Pythia, all feature types, pair+nopair (294 jobs)
#   quick       Small grid: {nano,small,base} x {5M,50M,500M} x Pythia x full x pair (9 jobs)
# =============================================================================

set -euo pipefail

DRY_RUN=false
SUBSET="baseline"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --subset)    SUBSET="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/sbatch_scaling.sh"

# Ensure output dirs exist
mkdir -p /pscratch/sd/s/sqian/part_training_output/scaling_study_v2/slurm_logs

# ---- Define grid based on subset ----
case "$SUBSET" in
    all)
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Pythia Herwig Mixed)
        FEATURES=(kin kinpid full)
        PAIRS=(1 0)
        ;;
    baseline)
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Pythia)
        FEATURES=(full)
        PAIRS=(1)
        ;;
    sample)
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Pythia Herwig Mixed)
        FEATURES=(full)
        PAIRS=(1)
        ;;
    features)
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Pythia)
        FEATURES=(kin kinpid full)
        PAIRS=(1 0)
        ;;
    quick)
        MODELS=(nano small base)
        BUDGETS=(5M 50M 500M)
        SAMPLES=(Pythia)
        FEATURES=(full)
        PAIRS=(1)
        ;;
    *)
        echo "Unknown subset: $SUBSET"
        echo "Valid: all, baseline, sample, features, quick"
        exit 1
        ;;
esac

# ---- Count and submit ----
TOTAL=0
SUBMITTED=0
SKIPPED=0

for sample in "${SAMPLES[@]}"; do
    for feature in "${FEATURES[@]}"; do
        for pair in "${PAIRS[@]}"; do
            for model in "${MODELS[@]}"; do
                for budget in "${BUDGETS[@]}"; do
                    TOTAL=$((TOTAL + 1))

                    pair_tag="pair"
                    [[ "$pair" == "0" ]] && pair_tag="nopair"
                    run_name="${model}_${budget}_${sample}_${feature}_${pair_tag}"

                    # Skip if already completed
                    CHECKPOINT_BASE="/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/checkpoints"
                    if ls "$CHECKPOINT_BASE"/${run_name}_*/training_complete.flag 1>/dev/null 2>&1; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    if $DRY_RUN; then
                        echo "[DRY-RUN] SAMPLE_TYPE=$sample FEATURE_TYPE=$feature PAIR_FEATURES=$pair sbatch $SBATCH_SCRIPT $model $budget"
                    else
                        SAMPLE_TYPE=$sample FEATURE_TYPE=$feature PAIR_FEATURES=$pair \
                            sbatch --job-name="scl-${run_name}" \
                            "$SBATCH_SCRIPT" "$model" "$budget"
                    fi
                    SUBMITTED=$((SUBMITTED + 1))
                done
            done
        done
    done
done

echo ""
echo "Grid: $SUBSET"
echo "Total configurations: $TOTAL"
echo "Already completed: $SKIPPED"
echo "Submitted: $SUBMITTED"
if $DRY_RUN; then
    echo "(dry run - no jobs actually submitted)"
fi
