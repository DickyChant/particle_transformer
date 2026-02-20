#!/bin/bash
# =============================================================================
# Chinchilla Scaling Law Study - Master Launcher
# Submits all (model_size, data_budget) combinations to SLURM.
#
# Usage:
#   ./launch_scaling_study.sh [options]
#
# Options:
#   --models SIZES     Comma-separated model sizes (default: all)
#   --budgets BUDGETS  Comma-separated data budgets (default: all)
#   --dry-run          Print commands without submitting
#   --help             Show this help
#
# Examples:
#   ./launch_scaling_study.sh                           # Submit all 49 jobs
#   ./launch_scaling_study.sh --dry-run                 # Preview all commands
#   ./launch_scaling_study.sh --models nano,micro,tiny  # Only small models
#   ./launch_scaling_study.sh --budgets 5M,10M,25M      # Only small budgets
#   ./launch_scaling_study.sh --models nano --budgets 5M # Single test job
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_MODELS="nano micro tiny small base large xlarge"
ALL_BUDGETS="5M 10M 25M 50M 100M 250M 500M"

MODELS="$ALL_MODELS"
BUDGETS="$ALL_BUDGETS"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODELS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --budgets)
            BUDGETS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure output directories exist
OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output/scaling_study"
mkdir -p "$OUTPUT_BASE/slurm_logs"

# Track submitted jobs
JOB_LOG="$OUTPUT_BASE/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"
job_count=0

echo "============================================"
echo " Chinchilla Scaling Law Study - Launcher"
echo "============================================"
echo "Models:  $MODELS"
echo "Budgets: $BUDGETS"
echo "Dry run: $DRY_RUN"
echo "============================================"
echo ""

for model in $MODELS; do
    for budget in $BUDGETS; do
        run_name="${model}_${budget}"
        cmd="sbatch --job-name=scale-${run_name} ${SCRIPT_DIR}/sbatch_scaling.sh ${model} ${budget}"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] $cmd"
        else
            echo "Submitting: $run_name"
            job_output=$($cmd)
            job_id=$(echo "$job_output" | grep -oP '\d+')
            echo "  -> Job ID: $job_id"
            echo "${job_id} ${run_name}" >> "$JOB_LOG"
        fi
        job_count=$((job_count + 1))
    done
done

echo ""
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete: $job_count jobs would be submitted"
else
    echo "Submitted $job_count jobs"
    echo "Job log: $JOB_LOG"
    echo "Monitor: squeue -u \$USER --name=scale-*"
fi
echo "============================================"
