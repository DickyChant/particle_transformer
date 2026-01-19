#!/bin/bash
#SBATCH --job-name=peft-downstream
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --time=02:00:00
#SBATCH --module=cvmfs
#SBATCH --output=/pscratch/sd/s/sqian/part_training_output/slurm_logs/peft_downstream-%j.out
#SBATCH --error=/pscratch/sd/s/sqian/part_training_output/slurm_logs/peft_downstream-%j.err

# PEFT Fine-tuning on downstream tasks (TopLandscape, QuarkGluon)
# Usage: sbatch sbatch_peft_downstream.sh [dataset] [pretrain_model] [optimizer]
# dataset: TopLandscape or QuarkGluon
# pretrain_model: full, kin, or kinpid (default: full)
# optimizer: ranger or muon (default: ranger)

DATASET="${1:-TopLandscape}"
PRETRAIN_MODEL="${2:-full}"
OPTIMIZER="${3:-ranger}"

# Validate dataset
case "$DATASET" in
    TopLandscape|QuarkGluon)
        echo "Dataset: $DATASET"
        ;;
    *)
        echo "Error: Invalid dataset '$DATASET'"
        echo "Usage: $0 [TopLandscape|QuarkGluon] [full|kin|kinpid] [ranger|muon]"
        exit 1
        ;;
esac

# Validate pretrain model
case "$PRETRAIN_MODEL" in
    full|kin|kinpid)
        echo "Pretrain model: $PRETRAIN_MODEL"
        ;;
    *)
        echo "Error: Invalid pretrain model '$PRETRAIN_MODEL'"
        exit 1
        ;;
esac

# Validate optimizer
case "$OPTIMIZER" in
    ranger|muon)
        echo "Optimizer: $OPTIMIZER"
        ;;
    *)
        echo "Error: Invalid optimizer '$OPTIMIZER'"
        exit 1
        ;;
esac

# Set up environment
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

module load conda
conda activate weaver

# Source environment for dataset paths
source env.sh

# Output directories on pscratch
OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CHECKPOINT_DIR="$OUTPUT_BASE/checkpoints/PEFT_${DATASET}_${PRETRAIN_MODEL}_${OPTIMIZER}_${TIMESTAMP}"
LOG_DIR="$OUTPUT_BASE/logs"
TENSORBOARD_DIR="$OUTPUT_BASE/tensorboard"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$TENSORBOARD_DIR"

echo "=========================================="
echo "PEFT Fine-tuning on $DATASET"
echo "Pretrained model: ParT_${PRETRAIN_MODEL}.pt"
echo "Optimizer: $OPTIMIZER"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "=========================================="

# Set dataset-specific options
if [[ "$DATASET" == "TopLandscape" ]]; then
    DATADIR=${DATADIR_TopLandscape}
    [[ -z $DATADIR ]] && DATADIR='./datasets/TopLandscape'
    
    # TopLandscape uses "kin" features only
    FEATURE_TYPE="kin"
    DATA_CONFIG="data/TopLandscape/top_${FEATURE_TYPE}.yaml"
    DATA_TRAIN="--data-train ${DATADIR}/train_file.parquet"
    DATA_VAL="--data-val ${DATADIR}/val_file.parquet"
    DATA_TEST="--data-test ${DATADIR}/test_file.parquet"
    EXTRA_OPTS="--num-workers 1 --fetch-step 1 --in-memory"
    EPOCHS=20
    SAMPLES_PER_EPOCH=$((2400 * 512))
    SAMPLES_PER_EPOCH_VAL=$((800 * 512))
    
elif [[ "$DATASET" == "QuarkGluon" ]]; then
    DATADIR=${DATADIR_QuarkGluon}
    [[ -z $DATADIR ]] && DATADIR='./datasets/QuarkGluon'
    
    # QuarkGluon uses "kinpid" features
    FEATURE_TYPE="kinpid"
    DATA_CONFIG="data/QuarkGluon/qg_${FEATURE_TYPE}.yaml"
    DATA_TRAIN="--data-train ${DATADIR}/train_file_*.parquet"
    DATA_VAL=""
    DATA_TEST="--data-test ${DATADIR}/test_file_*.parquet"
    EXTRA_OPTS="--num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889"
    EPOCHS=20
    SAMPLES_PER_EPOCH=1600000
    SAMPLES_PER_EPOCH_VAL=200000
fi

# PEFT model configuration
NETWORK_CONFIG="networks/example_ParticleTransformer_peft.py"
PRETRAIN_WEIGHTS="models/ParT_${PRETRAIN_MODEL}.pt"

# LoRA configuration
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Learning rate (lower for fine-tuning)
START_LR=1e-4

echo "Data directory: $DATADIR"
echo "Feature type: $FEATURE_TYPE"
echo "LoRA config: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"

# Run training
weaver \
    $DATA_TRAIN \
    $DATA_VAL \
    $DATA_TEST \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --model-prefix "$CHECKPOINT_DIR/net" \
    $EXTRA_OPTS \
    --batch-size 512 \
    --samples-per-epoch $SAMPLES_PER_EPOCH \
    --samples-per-epoch-val $SAMPLES_PER_EPOCH_VAL \
    --num-epochs $EPOCHS \
    --gpus 0 \
    --start-lr $START_LR \
    --optimizer $OPTIMIZER \
    --log "${LOG_DIR}/PEFT_${DATASET}_${PRETRAIN_MODEL}_${OPTIMIZER}_{auto}.log" \
    --predict-output pred.root \
    --tensorboard "${TENSORBOARD_DIR}/PEFT_${DATASET}_${PRETRAIN_MODEL}_${OPTIMIZER}" \
    --load-model-weights "$PRETRAIN_WEIGHTS" \
    --network-option lora_r $LORA_R \
    --network-option lora_alpha $LORA_ALPHA \
    --network-option lora_dropout $LORA_DROPOUT \
    --use-amp \
    "${@:4}"

echo "Training completed!"
