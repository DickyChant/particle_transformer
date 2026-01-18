#!/bin/bash
#SBATCH --job-name=particle-transformer-train
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --time=06:00:00
#SBATCH --module=cvmfs
#SBATCH --open-mode=append     # Append output to log files
#SBATCH --requeue
#SBATCH --output=/pscratch/sd/s/sqian/part_training_output/slurm_logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/s/sqian/part_training_output/slurm_logs/slurm-%j.err

# Parse command line arguments
# Usage: sbatch sbatch.sh [model] [feature_type] [sample_type] [optimizer] [additional_args...]
# model: ParT, ParT_gated, ParT_gated_v1, ParT_gated_no_mask, ParT_addnodes, ParT_no_mask, ParT_no_mask_aug, PN, PFN, or PCNN (default: ParT)
# feature_type: full, kin, or kinpid (default: full)
# sample_type: Pythia or Herwig (default: Pythia)
# optimizer: ranger or muon (default: ranger)
MODEL="${1:-ParT}"
FEATURE_TYPE="${2:-full}"
SAMPLE_TYPE="${3:-Pythia}"
OPTIMIZER="${4:-ranger}"

# Validate model
case "$MODEL" in
    ParT|ParT_gated|ParT_gated_v1|ParT_gated_no_mask|ParT_addnodes|ParT_no_mask|ParT_no_mask_aug|PN|PFN|PCNN)
        echo "Model: $MODEL"
        ;;
    *)
        echo "Error: Invalid model '$MODEL'"
        echo "Usage: $0 [model] [full|kin|kinpid] [Pythia|Herwig] [ranger|muon] [additional_args...]"
        exit 1
        ;;
esac

# Validate feature type
case "$FEATURE_TYPE" in
    full|kin|kinpid)
        echo "Feature type: $FEATURE_TYPE"
        ;;
    *)
        echo "Error: Invalid feature type '$FEATURE_TYPE'"
        echo "Usage: $0 [model] [full|kin|kinpid] [Pythia|Herwig] [ranger|muon] [additional_args...]"
        exit 1
        ;;
esac

# Validate sample type
case "$SAMPLE_TYPE" in
    Pythia|Herwig)
        echo "Sample type: $SAMPLE_TYPE"
        ;;
    *)
        echo "Error: Invalid sample type '$SAMPLE_TYPE'"
        echo "Usage: $0 [model] [full|kin|kinpid] [Pythia|Herwig] [ranger|muon] [additional_args...]"
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
        echo "Usage: $0 [model] [full|kin|kinpid] [Pythia|Herwig] [ranger|muon] [additional_args...]"
        exit 1
        ;;
esac

# Export SAMPLE_TYPE so train_JetClass.sh can use it
export SAMPLE_TYPE

# Set up environment
# Get the directory where this script is located
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

# Set number of GPUs for DDP training
export DDP_NGPUS=4

# Base output directory on pscratch (faster I/O, more storage)
export OUTPUT_BASE="/pscratch/sd/s/sqian/part_training_output"

# Create slurm logs directory (needed for SBATCH output/error directives)
mkdir -p "$OUTPUT_BASE/slurm_logs"

# Create unique timestamp for this job submission
# Check if TIMESTAMP file exists (from previous restart)
TIMESTAMP_FILE="$OUTPUT_BASE/timestamps/TIMESTAMP_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$TIMESTAMP_FILE")"
if [ -f "$TIMESTAMP_FILE" ]; then
    # Use existing timestamp (for requeue/restart)
    TIMESTAMP=$(cat "$TIMESTAMP_FILE")
    echo "Using existing timestamp for job restart: $TIMESTAMP"
else
    # Create new timestamp for this submission
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "$TIMESTAMP" > "$TIMESTAMP_FILE"
    echo "Created new timestamp for this job: $TIMESTAMP"
fi

# Create unique checkpoint directory for this job submission
export CHECKPOINT_DIR="$OUTPUT_BASE/checkpoints/${MODEL}_${FEATURE_TYPE}_${SAMPLE_TYPE}_${OPTIMIZER}_${TIMESTAMP}"
mkdir -p "$CHECKPOINT_DIR"

# Create log directory alongside checkpoints
export LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

# Tensorboard logs directory
export TENSORBOARD_DIR="$OUTPUT_BASE/tensorboard"
mkdir -p "$TENSORBOARD_DIR"

# Set model prefix to use our checkpoint directory
# This will be passed to train_JetClass.sh to override the default model-prefix
export MODEL_PREFIX="$CHECKPOINT_DIR/net"

echo "Output base (pscratch): $OUTPUT_BASE"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Model prefix: $MODEL_PREFIX"
echo "Log directory: $LOG_DIR"
echo "Tensorboard directory: $TENSORBOARD_DIR"
echo "Slurm logs: $OUTPUT_BASE/slurm_logs"

# Check if checkpoints exist for resume
CHECKPOINT_EXISTS=false
if ls "$CHECKPOINT_DIR"/*.pt 1> /dev/null 2>&1 || ls "$CHECKPOINT_DIR"/*.pth 1> /dev/null 2>&1; then
    CHECKPOINT_EXISTS=true
    echo "Found existing checkpoints in $CHECKPOINT_DIR"
fi

# Watch for the job ending to resubmit
export max_restarts=9
function requeue () {
    # Check if training completed successfully (all epochs done)
    # This is a simple check - you might want to enhance this based on your needs
    if [ -f "$CHECKPOINT_DIR/training_complete.flag" ]; then
        echo "Training appears complete. Not requeuing."
        exit 0
    fi
    
    export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
    if [ "$restarts" -ge "$max_restarts" ]; then
        echo "Max restarts reached - restarts at $max_restarts. Not requeuing."
        exit 0
    else
        echo "Going to requeue - at $restarts restarts"
        scontrol requeue ${SLURM_JOB_ID}
    fi
}

module load conda
conda activate weaver

# Set environment variables (embedded from env.sh)
export DATADIR_JetClass=/pscratch/sd/s/sqian/part_datasets/JetClass
export DATADIR_TopLandscape=
export DATADIR_QuarkGluon=

# Get restart count
export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
echo "Restart count: $restarts"

# Determine if we should resume training
SHOULD_RESUME=false
if [ "$restarts" -gt 0 ] || [ "$CHECKPOINT_EXISTS" = true ]; then
    SHOULD_RESUME=true
    echo "Resuming training from checkpoint..."
else
    echo "Starting new training run..."
fi

# Set up training parameters
# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1

# Check for torchrun - use python -m torch.distributed.run as fallback
if ((NGPUS > 1)); then
    # Try torchrun first
    if command -v torchrun &> /dev/null; then
        CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
    # Fallback to python -m torch.distributed.run (works if PyTorch is installed)
    elif python -c "import torch.distributed" &> /dev/null; then
        CMD="python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
    else
        echo "Error: torchrun not found and PyTorch distributed not available"
        echo "Please ensure PyTorch is installed in the weaver conda environment"
        exit 1
    fi
    echo "Using DDP command: $CMD"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 2 --fetch-step 0.01"

# Set model options based on MODEL (includes all ParT variants)
model=$MODEL
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_gated" ]]; then
    modelopts="networks/example_ParticleTransformer_gated.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_gated_v1" ]]; then
    modelopts="networks/example_ParticleTransformer_gated_v1.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_gated_no_mask" ]]; then
    modelopts="networks/example_ParticleTransformer_gated_no_mask.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_addnodes" ]]; then
    modelopts="networks/example_ParticleTransformer_addnodes.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_no_mask" ]]; then
    modelopts="networks/example_ParticleTransformer_no_mask.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "ParT_no_mask_aug" ]]; then
    modelopts="networks/example_ParticleTransformer_no_mask_augmented.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    echo "Valid models: ParT, ParT_gated, ParT_gated_v1, ParT_gated_no_mask, ParT_addnodes, ParT_no_mask, ParT_no_mask_aug, PN, PFN, PCNN"
    exit 1
fi

# Run the training
echo "Running ParticleTransformer training with $DDP_NGPUS GPUs"
echo "Model: $MODEL, Feature type: $FEATURE_TYPE, Sample type: $SAMPLE_TYPE, Optimizer: $OPTIMIZER"
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Build the weaver command
# Use MODEL_PREFIX instead of the default model-prefix from train_JetClass.sh
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
    --data-config "$SCRIPT_DIR/data/JetClass/JetClass_${FEATURE_TYPE}.yaml" --network-config $modelopts \
    --model-prefix "$MODEL_PREFIX" \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer $OPTIMIZER --log "${LOG_DIR}/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_${OPTIMIZER}_{auto}${suffix}.log" --predict-output pred.root \
    --tensorboard "${TENSORBOARD_DIR}/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_${OPTIMIZER}${suffix}" \
    "${@:5}"

TRAIN_EXIT_CODE=$?

# Create completion flag if training finished successfully (exit code 0)
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    touch "$CHECKPOINT_DIR/training_complete.flag"
    echo "Training completed successfully"
else
    echo "Training exited with code $TRAIN_EXIT_CODE"
fi

requeue

