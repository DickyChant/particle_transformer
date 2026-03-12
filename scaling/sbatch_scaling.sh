#!/bin/bash
#SBATCH --job-name=part-scaling
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
#SBATCH --output=/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/slurm_logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/slurm_logs/slurm-%j.err

# =============================================================================
# Chinchilla Scaling Law Study v2 - Single Run Script
#
# Each run creates a self-contained timestamped directory under OUTPUT_BASE:
#   {OUTPUT_BASE}/runs/{RUN_NAME}_{TIMESTAMP}/
#       checkpoints/     model .pt files
#       logs/            weaver text logs (.log.000, .log.001, ...)
#       tensorboard/     TensorBoard event files
#       config.txt       run configuration snapshot
#       training_complete.flag   (created on success)
#
# Usage:
#   sbatch sbatch_scaling.sh <model_size> <data_budget> [extra weaver args...]
#
# Required args:
#   model_size:   nano, micro, tiny, small, base, large, xlarge
#   data_budget:  5M, 10M, 25M, 50M, 100M, 250M, 500M
#
# Optional env vars (set before sbatch or export):
#   SAMPLE_TYPE:   Pythia (default), Herwig, Mixed
#   FEATURE_TYPE:  full (default), kinpid, kin
#   PAIR_FEATURES: 1 (default, with pairwise), 0 (without pairwise)
#
# Examples:
#   sbatch sbatch_scaling.sh base 100M
#   SAMPLE_TYPE=Mixed FEATURE_TYPE=kinpid sbatch sbatch_scaling.sh small 50M
#   PAIR_FEATURES=0 sbatch sbatch_scaling.sh base 100M
# =============================================================================

set -euo pipefail

MODEL_SIZE="${1:?Error: model_size required (nano/micro/tiny/small/base/large/xlarge)}"
DATA_BUDGET="${2:?Error: data_budget required (5M/10M/25M/50M/100M/250M/500M)}"

# ---- Defaults for optional env vars ----
SAMPLE_TYPE="${SAMPLE_TYPE:-Pythia}"
FEATURE_TYPE="${FEATURE_TYPE:-full}"
PAIR_FEATURES="${PAIR_FEATURES:-1}"

# ---- Validation ----
case "$MODEL_SIZE" in
    nano|micro|tiny|small|base|large|xlarge) ;;
    *) echo "Error: Invalid model size '$MODEL_SIZE'"; exit 1 ;;
esac

case "$DATA_BUDGET" in
    5M|10M|25M|50M|100M|250M|500M) ;;
    *) echo "Error: Invalid data budget '$DATA_BUDGET'"; exit 1 ;;
esac

case "$SAMPLE_TYPE" in
    Pythia|Herwig|Mixed) ;;
    *) echo "Error: Invalid SAMPLE_TYPE '$SAMPLE_TYPE' (Pythia/Herwig/Mixed)"; exit 1 ;;
esac

case "$FEATURE_TYPE" in
    full|kinpid|kin) ;;
    *) echo "Error: Invalid FEATURE_TYPE '$FEATURE_TYPE' (full/kinpid/kin)"; exit 1 ;;
esac

case "$PAIR_FEATURES" in
    0|1) ;;
    *) echo "Error: Invalid PAIR_FEATURES '$PAIR_FEATURES' (0 or 1)"; exit 1 ;;
esac

# ---- Environment Setup ----
REPO_DIR="/global/homes/s/sqian/jetclass_dir/particle_transformer"
cd "$REPO_DIR"

export DDP_NGPUS=4
NGPUS=$DDP_NGPUS

# ---- Build run name ----
PAIR_TAG="pair"
[[ "$PAIR_FEATURES" == "0" ]] && PAIR_TAG="nopair"
RUN_NAME="${MODEL_SIZE}_${DATA_BUDGET}_${SAMPLE_TYPE}_${FEATURE_TYPE}_${PAIR_TAG}"

# ---- Timestamped run directory (preserved across requeueing) ----
export OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output/scaling_study_v2"
mkdir -p "$OUTPUT_BASE/slurm_logs"

TIMESTAMP_FILE="$OUTPUT_BASE/timestamps/TIMESTAMP_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$TIMESTAMP_FILE")"
if [ -f "$TIMESTAMP_FILE" ]; then
    TIMESTAMP=$(cat "$TIMESTAMP_FILE")
    echo "Resuming with timestamp: $TIMESTAMP"
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "$TIMESTAMP" > "$TIMESTAMP_FILE"
    echo "New timestamp: $TIMESTAMP"
fi

# Everything for this run lives under one directory
RUN_DIR="$OUTPUT_BASE/runs/${RUN_NAME}_${TIMESTAMP}"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/tensorboard"
export MODEL_PREFIX="$RUN_DIR/checkpoints/net"

# ---- Structured header (parseable by plot_scaling_laws.py) ----
echo "===== SCALING_RUN_CONFIG ====="
echo "model_size: $MODEL_SIZE"
echo "data_budget: $DATA_BUDGET"
echo "sample_type: $SAMPLE_TYPE"
echo "feature_type: $FEATURE_TYPE"
echo "pair_features: $PAIR_FEATURES"
echo "slurm_job_id: ${SLURM_JOB_ID:-local}"
echo "run_dir: $RUN_DIR"
echo "===== END_CONFIG ====="

# Save config snapshot to run dir
cat > "$RUN_DIR/config.txt" <<EOF
model_size: $MODEL_SIZE
data_budget: $DATA_BUDGET
sample_type: $SAMPLE_TYPE
feature_type: $FEATURE_TYPE
pair_features: $PAIR_FEATURES
pair_tag: $PAIR_TAG
run_name: $RUN_NAME
timestamp: $TIMESTAMP
slurm_job_id: ${SLURM_JOB_ID:-local}
repo_dir: $REPO_DIR
EOF

echo "Run: $RUN_NAME"
echo "Run dir: $RUN_DIR"

# ---- Check for existing checkpoints (resume support) ----
if ls "$RUN_DIR/checkpoints"/*.pt 1> /dev/null 2>&1; then
    echo "Found existing checkpoints"
fi

# ---- Requeue Handler ----
export max_restarts=9
function requeue () {
    if [ -f "$RUN_DIR/training_complete.flag" ]; then
        echo "Training complete. Not requeuing."
        exit 0
    fi
    export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
    if [ "$restarts" -ge "$max_restarts" ]; then
        echo "Max restarts reached ($max_restarts). Not requeuing."
        exit 0
    else
        echo "Requeuing (restart $restarts)"
        scontrol requeue ${SLURM_JOB_ID}
    fi
}

# ---- Conda Environment ----
module load conda
conda activate weaver

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
    Pythia)
        add_sample_paths "Pythia"
        UNIQUE_SAMPLES=100000000
        ;;
    Herwig)
        add_sample_paths "Herwig"
        UNIQUE_SAMPLES=100000000
        ;;
    Mixed)
        add_sample_paths "Pythia"
        add_sample_paths "Herwig"
        UNIQUE_SAMPLES=200000000
        ;;
esac

echo "Unique training samples: $UNIQUE_SAMPLES"

# ---- Model Configuration ----
network_config="scaling/model_configs/ParT_${MODEL_SIZE}.py"

case "$MODEL_SIZE" in
    nano|micro|tiny|small|base)
        batch_size=512
        start_lr=1e-3
        ;;
    large)
        batch_size=256
        start_lr=5e-4
        ;;
    xlarge)
        batch_size=128
        start_lr=5e-4
        ;;
esac

# Pairwise feature control via --network-option
NETWORK_OPTS=()
if [[ "$PAIR_FEATURES" == "0" ]]; then
    NETWORK_OPTS+=(--network-option pair_input_dim 0 --network-option pair_embed_dims None)
    echo "Pairwise features: DISABLED"
else
    echo "Pairwise features: ENABLED"
fi

# ---- Data Budget Configuration ----
# Total samples = samples_per_epoch * NGPUS * num_epochs
case "$DATA_BUDGET" in
    5M)   samples_per_epoch=320000;   num_epochs=4  ;;
    10M)  samples_per_epoch=640000;   num_epochs=4  ;;
    25M)  samples_per_epoch=1600000;  num_epochs=4  ;;
    50M)  samples_per_epoch=1600000;  num_epochs=8  ;;
    100M) samples_per_epoch=1600000;  num_epochs=16 ;;
    250M) samples_per_epoch=1600000;  num_epochs=40 ;;
    500M) samples_per_epoch=2560000;  num_epochs=50 ;;
esac

samples_per_epoch_val=1280000
TOTAL_SAMPLES=$(( samples_per_epoch * NGPUS * num_epochs ))
DATA_REUSE="no"
if (( TOTAL_SAMPLES > UNIQUE_SAMPLES )); then
    DATA_REUSE="yes ($(echo "scale=1; $TOTAL_SAMPLES / $UNIQUE_SAMPLES" | bc)x)"
fi

echo "Network config: $network_config"
echo "Batch size: $batch_size, LR: $start_lr"
echo "Samples/epoch (per GPU): $samples_per_epoch, Epochs: $num_epochs"
echo "Total training samples: $TOTAL_SAMPLES"
echo "Data reuse: $DATA_REUSE"

# Append training params to config snapshot
cat >> "$RUN_DIR/config.txt" <<EOF
network_config: $network_config
batch_size: $batch_size
start_lr: $start_lr
samples_per_epoch: $samples_per_epoch
num_epochs: $num_epochs
total_samples: $TOTAL_SAMPLES
unique_samples: $UNIQUE_SAMPLES
data_reuse: $DATA_REUSE
ngpus: $NGPUS
EOF

# ---- DDP Command ----
if ((NGPUS > 1)); then
    if command -v torchrun &> /dev/null; then
        CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
    elif python -c "import torch.distributed" &> /dev/null; then
        CMD="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
    else
        echo "Error: torchrun not found"
        exit 1
    fi
else
    CMD="weaver"
fi

# ---- Get restart count ----
export restarts=$(scontrol show jobid $SLURM_JOB_ID 2>/dev/null | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
restarts=${restarts:-0}
echo "Restart count: $restarts"

# ---- Launch Training ----
# All outputs go into the timestamped RUN_DIR:
#   checkpoints -> RUN_DIR/checkpoints/net_epoch-N_state.pt
#   text logs   -> RUN_DIR/logs/{auto}.log.000
#   tensorboard -> RUN_DIR/tensorboard/  (absolute path -> log_dir, not runs/)
$CMD \
    --data-train "${DATA_TRAIN_ARGS[@]}" \
    --data-val "${DATA_VAL_ARGS[@]}" \
    --data-test "${DATA_TEST_ARGS[@]}" \
    --data-config "data/JetClass/JetClass_${FEATURE_TYPE}.yaml" \
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
    "${@:3}"

TRAIN_EXIT_CODE=$?

# ---- Post-training summary (logged to SLURM stdout for easy parsing) ----
echo ""
echo "===== SCALING_RUN_RESULT ====="
echo "run_name: $RUN_NAME"
echo "run_dir: $RUN_DIR"
echo "exit_code: $TRAIN_EXIT_CODE"

# Extract final losses from the rank-0 weaver log
WEAVER_LOG=$(ls -t "$RUN_DIR"/logs/*.log.000 2>/dev/null | head -1)
if [ -n "$WEAVER_LOG" ] && [ -f "$WEAVER_LOG" ]; then
    BEST_TRAIN_LOSS=$(grep "Train AvgLoss:" "$WEAVER_LOG" | \
        sed 's/.*Train AvgLoss: \([0-9.]*\).*/\1/' | sort -n | head -1)
    LAST_TRAIN_LOSS=$(grep "Train AvgLoss:" "$WEAVER_LOG" | \
        sed 's/.*Train AvgLoss: \([0-9.]*\).*/\1/' | tail -1)
    BEST_VAL_METRIC=$(grep "Current validation metric:" "$WEAVER_LOG" | \
        sed 's/.*best: \([0-9.]*\)).*/\1/' | tail -1)
    NUM_EPOCHS_DONE=$(grep -c "Train AvgLoss:" "$WEAVER_LOG")

    echo "best_train_loss: ${BEST_TRAIN_LOSS:-N/A}"
    echo "last_train_loss: ${LAST_TRAIN_LOSS:-N/A}"
    echo "best_val_metric: ${BEST_VAL_METRIC:-N/A}"
    echo "epochs_completed: ${NUM_EPOCHS_DONE:-0}"
    echo "weaver_log: $WEAVER_LOG"
fi
echo "===== END_RESULT ====="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    touch "$RUN_DIR/training_complete.flag"
    echo "Training completed successfully"
else
    echo "Training exited with code $TRAIN_EXIT_CODE"
fi

requeue
