#!/bin/bash
# =============================================================================
#
#    ____            _   _      _        _____                     __
#   |  _ \ __ _ _ __| |_(_) ___| | ___  |_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __
#   | |_) / _` | '__| __| |/ __| |/ _ \   | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
#   |  __/ (_| | |  | |_| | (__| |  __/   | || | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |
#   |_|   \__,_|_|   \__|_|\___|_|\___|   |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_|
#
#        _____ _     _            _     _ _ _          ____            _ _
#       / ____| |   (_)          | |   (_) | |        / ___|  ___ __ _| (_)_ __   __ _
#      | |    | |__  _ _ __   ___| |__  _| | | __ _   \___ \ / __/ _` | | | '_ \ / _` |
#      | |    | '_ \| | '_ \ / __| '_ \| | | |/ _` |  ___) | (_| (_| | | | | | | (_| |
#       \____|_| |_|_|_| |_|\___|_| |_|_|_|_|\__,_| |____/ \___\__,_|_|_|_| |_|\__, |
#                                                                                 |___/
#               L(N, D) = A * N^(-alpha) + B * D^(-beta) + E
#
# =============================================================================
#
# Launch full grid of scaling law study jobs
#
#   Model sizes  :  nano  micro  tiny  small  base  large  xlarge   (7)
#   Data budgets :  5M  10M  25M  50M  100M  250M  500M            (7)
#   Sample types :  Pythia  Herwig  Mixed                           (3)
#   Feature types:  kin  kinpid  full                               (3)
#   Pairwise     :  pair  nopair                                    (2)
#
#   Full grid = 7 x 7 x 3 x 3 x 2 = 882 jobs
#
#        model   data
#        size    budget   sample    features   pair?
#       +------+--------+---------+----------+------+
#       | nano |  5M    | Pythia  | kin      | pair |
#       | ...  |  ...   | Herwig  | kinpid   |nopair|
#       |xlarge| 500M   | Mixed   | full     |      |
#       +------+--------+---------+----------+------+
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
#   oneepoch    Like baseline but NUM_EPOCHS=1: single pass through data (49 jobs)
#   nopair        Full 7x7 grid, Pythia x full x nopair (49 jobs)
#   nopair_quick  {nano,small,base} x {5M,50M,500M} x Pythia x full x nopair (9 jobs)
#   nopair_oneepoch  Full 7x7 grid, nopair, NUM_EPOCHS=1 (49 jobs)
# =============================================================================

set -euo pipefail

DRY_RUN=false
SUBSET="baseline"
EXTRA_ENV=""

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
mkdir -p "${SCRATCH}/part_scaling_output/slurm_logs"

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
    oneepoch)
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M)
        SAMPLES=(Pythia)
        FEATURES=(full)
        PAIRS=(1)
        EXTRA_ENV="NUM_EPOCHS=1"
        ;;
    nopair)
        # 49 jobs: full 7x7 grid, no pairwise features
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Herwig)
        FEATURES=(full)
        PAIRS=(0)
        ;;
    nopair_quick)
        # 9 jobs: quick sanity-check grid, no pairwise features
        MODELS=(nano small base)
        BUDGETS=(5M 50M 500M)
        SAMPLES=(Herwig)
        FEATURES=(full)
        PAIRS=(0)
        ;;
    nopair_oneepoch)
        # 49 jobs: full 7x7 grid, no pairwise features, single epoch per budget
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Herwig)
        FEATURES=(full)
        PAIRS=(0)
        EXTRA_ENV="NUM_EPOCHS=1"
        ;;
    v2_nopair_oneepoch)
        # 49 jobs: full 7x7, Herwig, no pairwise features, 1 epoch, v2 arch
        MODELS=(nano micro tiny small base large xlarge)
        BUDGETS=(5M 10M 25M 50M 100M 250M 500M)
        SAMPLES=(Herwig)
        FEATURES=(full)
        PAIRS=(0)
        EXTRA_ENV="NUM_EPOCHS=1 MODEL_CONFIG_DIR=scaling/model_configs_v2 MODEL_VERSION=v2"
        ;;
    *)
        echo "Unknown subset: $SUBSET"
        echo "Valid: all, baseline, sample, features, quick, oneepoch, nopair, nopair_quick, nopair_oneepoch, v2_nopair_oneepoch"
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
                    epoch_tag=""
                    [[ -n "$EXTRA_ENV" ]] && epoch_tag=$(echo "$EXTRA_ENV" | grep -oP 'NUM_EPOCHS=\K\d+' | head -1)
                    [[ -n "$epoch_tag" ]] && epoch_tag="_${epoch_tag}ep"
                    version_tag=$(echo "${EXTRA_ENV:-}" | grep -oP 'MODEL_VERSION=\K\S+' | head -1 || true)
                    [[ -n "$version_tag" ]] && version_tag="_${version_tag}"
                    run_name="${model}_${budget}_${sample}_${feature}_${pair_tag}${epoch_tag}${version_tag}"

                    # Dynamic wall time for 1-epoch jobs; fall back to 12h otherwise.
                    # Calibrated from actual per-epoch throughput in existing nopair runs:
                    #   nano/micro/tiny ~9000/s, small/base ~5200/s, large ~2170/s, xlarge ~1076/s
                    # "infeasible" = estimated >12h even without margin; job is skipped.
                    wall_time="12:00:00"
                    if echo "${EXTRA_ENV:-}" | grep -q 'NUM_EPOCHS=1'; then
                        case "$model" in
                            nano|micro|tiny)
                                case "$budget" in
                                    5M|10M|25M|50M|100M) wall_time="01:00:00" ;;
                                    250M)                wall_time="03:00:00" ;;
                                    500M)                wall_time="06:00:00" ;;
                                esac ;;
                            small|base)
                                case "$budget" in
                                    5M|10M|25M|50M) wall_time="01:00:00" ;;
                                    100M)           wall_time="02:00:00" ;;
                                    250M)           wall_time="05:00:00" ;;
                                    500M)           wall_time="10:00:00" ;;
                                esac ;;
                            large)
                                case "$budget" in
                                    5M|10M)   wall_time="01:00:00" ;;
                                    25M|50M)  wall_time="03:00:00" ;;
                                    100M)     wall_time="05:00:00" ;;
                                    250M)     wall_time="12:00:00" ;;
                                    500M)     wall_time="infeasible" ;;
                                esac ;;
                            xlarge)
                                case "$budget" in
                                    5M|10M)    wall_time="01:00:00" ;;
                                    25M|50M)   wall_time="05:00:00" ;;
                                    100M)      wall_time="10:00:00" ;;
                                    250M|500M) wall_time="infeasible" ;;
                                esac ;;
                        esac
                    fi

                    # Skip infeasible 1-epoch jobs (estimated >12h to complete)
                    if [[ "$wall_time" == "infeasible" ]]; then
                        SKIPPED=$((SKIPPED + 1))
                        $DRY_RUN && echo "[SKIP] infeasible (>12h): $model $budget"
                        continue
                    fi

                    # Skip if already completed (check timestamped run dirs)
                    RUNS_BASE="${SCRATCH}/part_scaling_output/runs"
                    if ls "$RUNS_BASE"/${run_name}_*/training_complete.flag 1>/dev/null 2>&1; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi

                    if $DRY_RUN; then
                        echo "[DRY-RUN] [time=$wall_time] ${EXTRA_ENV:+$EXTRA_ENV }SAMPLE_TYPE=$sample FEATURE_TYPE=$feature PAIR_FEATURES=$pair sbatch $SBATCH_SCRIPT $model $budget"
                    else
                        env SAMPLE_TYPE=$sample FEATURE_TYPE=$feature PAIR_FEATURES=$pair $EXTRA_ENV \
                            sbatch --job-name="scl-${run_name}" \
                            --time="$wall_time" \
                            "$SBATCH_SCRIPT" "$model" "$budget"
                    fi
                    SUBMITTED=$((SUBMITTED + 1))
                done
            done
        done
    done
done

echo ""
echo "  ================================================================"
echo "       _                        _                   _"
echo "      | | ___  _ __ ___   ___  | |__   _____      _| |"
echo "      | |/ _ \| '_ \` _ \ / _ \ | '_ \ / _ \ \ /\ / / |"
echo "      | | (_) | | | | | |  __/ | | | | (_) \ V  V /|_|"
echo "      |_|\___/|_| |_| |_|\___| |_| |_|\___/ \_/\_/ (_)"
echo ""
echo "         Grid subset:  $SUBSET"
echo "         Total configs: $TOTAL"
printf "         Completed:     %s\n" "$SKIPPED"
printf "         Submitted:     %s\n" "$SUBMITTED"
if $DRY_RUN; then
    echo ""
    echo "         (dry run - no jobs actually submitted)"
fi
echo "  ================================================================"
