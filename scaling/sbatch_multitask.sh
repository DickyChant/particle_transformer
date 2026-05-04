#!/bin/bash
#SBATCH --job-name=part-mt
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --time=12:00:00
#SBATCH --module=cvmfs
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --output=/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/slurm_logs/slurm-mt-%j.out
#SBATCH --error=/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/slurm_logs/slurm-mt-%j.err

# =============================================================================
# Multi-Task Scaling Study: Classification + Regression
#
# Usage:
#   sbatch sbatch_multitask.sh <model_size> <data_budget>
#   NUM_EPOCHS=1 sbatch sbatch_multitask.sh nano 5M
#
# Same interface as sbatch_scaling.sh but uses:
#   - networks/multitask_ParT.py (cls + reg heads, uncertainty weighting)
#   - data/JetClass/JetClass_multitask.yaml (cls_label + reg_target=log(jet_pt))
#   - Custom train/eval with gradient cosine similarity logging
# =============================================================================

set -euo pipefail

MODEL_SIZE="${1:?Error: model_size required (nano/micro/tiny/small/base/large/xlarge)}"
DATA_BUDGET="${2:?Error: data_budget required (5M/10M/25M/50M/100M/250M/500M)}"

SAMPLE_TYPE="${SAMPLE_TYPE:-Pythia}"
PAIR_FEATURES="${PAIR_FEATURES:-1}"
GRAD_COS_FREQ="${GRAD_COS_FREQ:-50}"

# ---- Validation ----
case "$MODEL_SIZE" in
    nano|micro|tiny|small|base|large|xlarge) ;;
    *) echo "Error: Invalid model size '$MODEL_SIZE'"; exit 1 ;;
esac

case "$DATA_BUDGET" in
    5M|10M|25M|50M|100M|250M|500M) ;;
    *) echo "Error: Invalid data budget '$DATA_BUDGET'"; exit 1 ;;
esac

# ---- Environment Setup ----
REPO_DIR="/global/homes/s/sqian/jetclass_dir/particle_transformer"
cd "$REPO_DIR"

export DDP_NGPUS=4
NGPUS=$DDP_NGPUS

# ---- Build run name ----
PAIR_TAG="pair"
[[ "$PAIR_FEATURES" == "0" ]] && PAIR_TAG="nopair"
EPOCH_TAG=""
[[ -n "${NUM_EPOCHS:-}" ]] && EPOCH_TAG="_${NUM_EPOCHS}ep"
RUN_NAME="${MODEL_SIZE}_${DATA_BUDGET}_${SAMPLE_TYPE}_full_${PAIR_TAG}_multitask${EPOCH_TAG}"

# ---- Timestamped run directory ----
export OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output/scaling_study_v2"
mkdir -p "$OUTPUT_BASE/slurm_logs"

TIMESTAMP_FILE="$OUTPUT_BASE/timestamps/TIMESTAMP_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$TIMESTAMP_FILE")"
if [ -f "$TIMESTAMP_FILE" ]; then
    TIMESTAMP=$(cat "$TIMESTAMP_FILE")
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "$TIMESTAMP" > "$TIMESTAMP_FILE"
fi

RUN_DIR="$OUTPUT_BASE/runs/${RUN_NAME}_${TIMESTAMP}"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/tensorboard"
export MODEL_PREFIX="$RUN_DIR/checkpoints/net"

# ---- Config snapshot ----
cat > "$RUN_DIR/config.txt" <<EOF
model_size: $MODEL_SIZE
data_budget: $DATA_BUDGET
sample_type: $SAMPLE_TYPE
feature_type: full
pair_features: $PAIR_FEATURES
pair_tag: $PAIR_TAG
task_type: multitask
run_name: $RUN_NAME
timestamp: $TIMESTAMP
slurm_job_id: ${SLURM_JOB_ID:-local}
repo_dir: $REPO_DIR
grad_cos_freq: $GRAD_COS_FREQ
EOF

echo "Run: $RUN_NAME"
echo "Run dir: $RUN_DIR"

# ---- Requeue Handler ----
export max_restarts=9
function requeue () {
    if [ -f "$RUN_DIR/training_complete.flag" ]; then
        echo "Training complete. Not requeuing."
        exit 0
    fi
    export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
    if [ "$restarts" -ge "$max_restarts" ]; then
        exit 0
    else
        scontrol requeue ${SLURM_JOB_ID}
    fi
}

# ---- Conda Environment ----
module load conda
conda activate weaver
pip install -e weaver-core/ --quiet 2>/dev/null

# ---- Dataset paths ----
DATADIR="/pscratch/sd/s/sqian/part_datasets/JetClass"
JET_CLASSES="HToBB HToCC HToGG HToWW2Q1L HToWW4Q TTBar TTBarLep WToQQ ZToQQ ZJetsToNuNu"

DATA_TRAIN_ARGS=()
DATA_VAL_ARGS=()
DATA_TEST_ARGS=()

add_sample_paths() {
    local stype="$1"
    for cls in $JET_CLASSES; do
        DATA_TRAIN_ARGS+=("${cls}_${stype}:${DATADIR}/${stype}/train_100M/${cls}_*.root")
    done
    DATA_VAL_ARGS+=("${DATADIR}/${stype}/val_5M/*.root")
    DATA_TEST_ARGS+=("${DATADIR}/${stype}/test_20M/*.root")
}

case "$SAMPLE_TYPE" in
    Pythia)  add_sample_paths "Pythia"; UNIQUE_SAMPLES=100000000 ;;
    Herwig)  add_sample_paths "Herwig"; UNIQUE_SAMPLES=100000000 ;;
    Mixed)   add_sample_paths "Pythia"; add_sample_paths "Herwig"; UNIQUE_SAMPLES=200000000 ;;
esac

# ---- Model Configuration ----
# Use multitask wrapper — maps to scaling model configs for embed_dims etc.
network_config="networks/multitask_ParT.py"

# Model size -> embed_dims, pair_embed_dims, num_heads, num_layers, num_cls_layers
declare -A EMBED_DIMS PAIR_DIMS HEADS LAYERS CLS_LAYERS
EMBED_DIMS=([nano]='[32,128,32]'   [micro]='[64,128,64]'   [tiny]='[64,256,64]'
            [small]='[128,256,128]' [base]='[128,512,128]'  [large]='[256,512,256]'
            [xlarge]='[512,1024,512]')
PAIR_DIMS=( [nano]='[16,16,16]'    [micro]='[32,32,32]'    [tiny]='[32,32,32]'
            [small]='[64,64,64]'   [base]='[64,64,64]'     [large]='[128,128,128]'
            [xlarge]='[128,128,128]')
HEADS=(     [nano]=4  [micro]=4  [tiny]=8  [small]=8  [base]=8   [large]=8   [xlarge]=16)
LAYERS=(    [nano]=2  [micro]=4  [tiny]=4  [small]=6  [base]=8   [large]=8   [xlarge]=12)
CLS_LAYERS=([nano]=1  [micro]=1  [tiny]=2  [small]=2  [base]=2   [large]=2   [xlarge]=2)

case "$MODEL_SIZE" in
    nano|micro|tiny|small|base) batch_size=512; start_lr=1e-3 ;;
    large)                      batch_size=256; start_lr=5e-4 ;;
    xlarge)                     batch_size=128; start_lr=5e-4 ;;
esac

# Network options for the multitask wrapper
NETWORK_OPTS=(
    --network-option embed_dims "${EMBED_DIMS[$MODEL_SIZE]}"
    --network-option pair_embed_dims "${PAIR_DIMS[$MODEL_SIZE]}"
    --network-option num_heads "${HEADS[$MODEL_SIZE]}"
    --network-option num_layers "${LAYERS[$MODEL_SIZE]}"
    --network-option num_cls_layers "${CLS_LAYERS[$MODEL_SIZE]}"
    --network-option grad_cos_freq "$GRAD_COS_FREQ"
)

if [[ "$PAIR_FEATURES" == "0" ]]; then
    NETWORK_OPTS+=(--network-option pair_input_dim 0 --network-option pair_embed_dims None)
fi

# ---- Data Budget Configuration ----
NUM_EPOCHS_OVERRIDE="${NUM_EPOCHS:-}"

case "$DATA_BUDGET" in
    5M)   samples_per_epoch=320000;   num_epochs=4  ;;
    10M)  samples_per_epoch=640000;   num_epochs=4  ;;
    25M)  samples_per_epoch=1600000;  num_epochs=4  ;;
    50M)  samples_per_epoch=1600000;  num_epochs=8  ;;
    100M) samples_per_epoch=1600000;  num_epochs=16 ;;
    250M) samples_per_epoch=1600000;  num_epochs=40 ;;
    500M) samples_per_epoch=2560000;  num_epochs=50 ;;
esac

if [[ -n "$NUM_EPOCHS_OVERRIDE" ]]; then
    num_epochs=$NUM_EPOCHS_OVERRIDE
    budget_num=${DATA_BUDGET%M}
    samples_per_epoch=$(( budget_num * 1000000 / NGPUS / num_epochs ))
fi

samples_per_epoch_val=1280000
TOTAL_SAMPLES=$(( samples_per_epoch * NGPUS * num_epochs ))

# Append to config
cat >> "$RUN_DIR/config.txt" <<EOF
network_config: $network_config
batch_size: $batch_size
start_lr: $start_lr
samples_per_epoch: $samples_per_epoch
num_epochs: $num_epochs
total_samples: $TOTAL_SAMPLES
unique_samples: $UNIQUE_SAMPLES
ngpus: $NGPUS
EOF

echo "Model: $MODEL_SIZE (multitask), Budget: $DATA_BUDGET, Epochs: $num_epochs"

# ---- Resume support ----
RESUME_ARGS=()
LAST_EPOCH=""
if ls "$RUN_DIR/checkpoints"/net_epoch-*_state.pt 1> /dev/null 2>&1; then
    for ckpt in "$RUN_DIR/checkpoints"/net_epoch-*_state.pt; do
        n=$(basename "$ckpt" | sed -E 's/^net_epoch-([0-9]+)_state\.pt$/\1/')
        opt="$RUN_DIR/checkpoints/net_epoch-${n}_optimizer.pt"
        if [ -f "$opt" ]; then
            if [ -z "$LAST_EPOCH" ] || [ "$n" -gt "$LAST_EPOCH" ]; then
                LAST_EPOCH=$n
            fi
        fi
    done
fi
if [ -n "$LAST_EPOCH" ]; then
    echo "Resuming from epoch $LAST_EPOCH (will continue at epoch $((LAST_EPOCH + 1)))"
    RESUME_ARGS=(--load-epoch "$LAST_EPOCH")
else
    echo "No usable checkpoint found; starting from scratch"
fi

# ---- DDP Command ----
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

# ---- Launch Training ----
# Disable -e around training so a non-zero exit reaches the requeue path
# instead of killing the script under `set -e`.
set +e
$CMD \
    --data-train "${DATA_TRAIN_ARGS[@]}" \
    --data-val "${DATA_VAL_ARGS[@]}" \
    --data-test "${DATA_TEST_ARGS[@]}" \
    --data-config "data/JetClass/JetClass_multitask.yaml" \
    --network-config "$network_config" \
    "${NETWORK_OPTS[@]}" \
    --model-prefix "$MODEL_PREFIX" \
    --batch-size $batch_size --start-lr $start_lr \
    --num-workers 2 --fetch-step 0.01 \
    --samples-per-epoch $samples_per_epoch \
    --samples-per-epoch-val $samples_per_epoch_val \
    --num-epochs $num_epochs \
    --gpus 0 \
    --optimizer ranger \
    --use-amp \
    --log "$RUN_DIR/logs/{auto}.log" \
    --predict-output pred.root \
    --tensorboard "$RUN_DIR/tensorboard" \
    --save-steps "${SAVE_STEPS:-200}" \
    "${RESUME_ARGS[@]}" \
    "${@:3}"
TRAIN_EXIT_CODE=$?
set -e

# ---- Post-training ----
WEAVER_LOG=$(ls -t "$RUN_DIR"/logs/*.log.000 2>/dev/null | head -1)
if [ -n "$WEAVER_LOG" ] && [ -f "$WEAVER_LOG" ]; then
    echo "Final log: $WEAVER_LOG"
    grep "Train AvgLoss:" "$WEAVER_LOG" | tail -1
    grep "CE:" "$WEAVER_LOG" | tail -1
fi

if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    touch "$RUN_DIR/training_complete.flag"
    echo "Training complete!"
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    requeue
fi
