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
#SBATCH --output=/pscratch/sd/s/sqian/part_training_output/scaling_study/slurm_logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/s/sqian/part_training_output/scaling_study/slurm_logs/slurm-%j.err

# =============================================================================
# Chinchilla Scaling Law Study - Single Run Script
# Usage: sbatch sbatch_scaling.sh <model_size> <data_budget> [additional_args...]
#   model_size:  nano, micro, tiny, small, base, large, xlarge
#   data_budget: 5M, 10M, 25M, 50M, 100M, 250M, 500M
# =============================================================================

MODEL_SIZE="${1:?Error: model_size required (nano/micro/tiny/small/base/large/xlarge)}"
DATA_BUDGET="${2:?Error: data_budget required (5M/10M/25M/50M/100M/250M/500M)}"

# Validate model size
case "$MODEL_SIZE" in
    nano|micro|tiny|small|base|large|xlarge)
        echo "Model size: $MODEL_SIZE"
        ;;
    *)
        echo "Error: Invalid model size '$MODEL_SIZE'"
        echo "Valid sizes: nano, micro, tiny, small, base, large, xlarge"
        exit 1
        ;;
esac

# Validate data budget
case "$DATA_BUDGET" in
    5M|10M|25M|50M|100M|250M|500M)
        echo "Data budget: $DATA_BUDGET"
        ;;
    *)
        echo "Error: Invalid data budget '$DATA_BUDGET'"
        echo "Valid budgets: 5M, 10M, 25M, 50M, 100M, 250M, 500M"
        exit 1
        ;;
esac

# ---- Environment Setup ----
REPO_DIR="/global/homes/s/sqian/jetclass_dir/particle_transformer"
cd "$REPO_DIR"

export DDP_NGPUS=4
NGPUS=$DDP_NGPUS

# ---- Output Directories ----
export OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output/scaling_study"
mkdir -p "$OUTPUT_BASE/slurm_logs"

# Unique timestamp (preserved across requeueing)
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

RUN_NAME="${MODEL_SIZE}_${DATA_BUDGET}"
export CHECKPOINT_DIR="$OUTPUT_BASE/checkpoints/${RUN_NAME}_${TIMESTAMP}"
export LOG_DIR="$OUTPUT_BASE/logs"
export TENSORBOARD_DIR="$OUTPUT_BASE/tensorboard"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"
export MODEL_PREFIX="$CHECKPOINT_DIR/net"

echo "Run: $RUN_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"

# ---- Check for existing checkpoints (resume support) ----
CHECKPOINT_EXISTS=false
if ls "$CHECKPOINT_DIR"/*.pt 1> /dev/null 2>&1; then
    CHECKPOINT_EXISTS=true
    echo "Found existing checkpoints"
fi

# ---- Requeue Handler ----
export max_restarts=9
function requeue () {
    if [ -f "$CHECKPOINT_DIR/training_complete.flag" ]; then
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

# ---- Dataset ----
DATADIR="/pscratch/sd/s/sqian/part_datasets/JetClass"
SAMPLE_TYPE="Pythia"
FEATURE_TYPE="full"

# ---- Model Configuration ----
# Map model_size -> network config, batch size, learning rate
case "$MODEL_SIZE" in
    nano)
        network_config="scaling/model_configs/ParT_nano.py"
        batch_size=512
        start_lr=1e-3
        ;;
    micro)
        network_config="scaling/model_configs/ParT_micro.py"
        batch_size=512
        start_lr=1e-3
        ;;
    tiny)
        network_config="scaling/model_configs/ParT_tiny.py"
        batch_size=512
        start_lr=1e-3
        ;;
    small)
        network_config="scaling/model_configs/ParT_small.py"
        batch_size=512
        start_lr=1e-3
        ;;
    base)
        network_config="scaling/model_configs/ParT_base.py"
        batch_size=512
        start_lr=1e-3
        ;;
    large)
        network_config="scaling/model_configs/ParT_large.py"
        batch_size=256
        start_lr=5e-4
        ;;
    xlarge)
        network_config="scaling/model_configs/ParT_xlarge.py"
        batch_size=128
        start_lr=5e-4
        ;;
esac

# ---- Data Budget Configuration ----
# Map data_budget -> samples_per_epoch (per GPU) and num_epochs
# Total samples = samples_per_epoch * NGPUS * num_epochs
case "$DATA_BUDGET" in
    5M)
        samples_per_epoch=320000
        num_epochs=4
        ;;
    10M)
        samples_per_epoch=640000
        num_epochs=4
        ;;
    25M)
        samples_per_epoch=1600000
        num_epochs=4
        ;;
    50M)
        samples_per_epoch=1600000
        num_epochs=8
        ;;
    100M)
        samples_per_epoch=1600000
        num_epochs=16
        ;;
    250M)
        samples_per_epoch=1600000
        num_epochs=40
        ;;
    500M)
        samples_per_epoch=2560000
        num_epochs=50
        ;;
esac

samples_per_epoch_val=1280000

echo "Network config: $network_config"
echo "Batch size: $batch_size, LR: $start_lr"
echo "Samples/epoch (per GPU): $samples_per_epoch, Epochs: $num_epochs"
echo "Total training samples: $(( samples_per_epoch * NGPUS * num_epochs ))"

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
$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_100M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_100M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_100M/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_5M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_20M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_20M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_20M/ZJetsToNuNu_*.root" \
    --data-config "data/JetClass/JetClass_${FEATURE_TYPE}.yaml" \
    --network-config "$network_config" \
    --model-prefix "$MODEL_PREFIX" \
    --batch-size $batch_size --start-lr $start_lr \
    --num-workers 2 --fetch-step 0.01 \
    --samples-per-epoch $samples_per_epoch \
    --samples-per-epoch-val $samples_per_epoch_val \
    --num-epochs $num_epochs \
    --gpus 0 \
    --optimizer ranger \
    --use-amp \
    --log "${LOG_DIR}/${RUN_NAME}_{auto}.log" \
    --predict-output pred.root \
    --tensorboard "${TENSORBOARD_DIR}/${RUN_NAME}" \
    "${@:3}"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    touch "$CHECKPOINT_DIR/training_complete.flag"
    echo "Training completed successfully"
else
    echo "Training exited with code $TRAIN_EXIT_CODE"
fi

requeue
