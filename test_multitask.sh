#!/bin/bash
# Quick interactive test for multi-task ParT training
# Run on a GPU node: salloc -A m3246 -C gpu -q interactive -t 30 -n 1 --gpus 1
# Then: bash test_multitask.sh

set -euo pipefail

module load conda 2>/dev/null
conda activate weaver

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Use local weaver-core
pip install -e weaver-core/ --quiet 2>/dev/null

DATA_DIR="/pscratch/sd/s/sqian/part_datasets/JetClass/Pythia/train_100M"
DATA_CONFIG="data/JetClass/JetClass_multitask.yaml"
NETWORK_CONFIG="networks/multitask_ParT.py"
OUTPUT_DIR="/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/test_multitask"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Multi-task ParT Quick Test"
echo "============================================"
echo "  Data:    $DATA_DIR"
echo "  Config:  $DATA_CONFIG"
echo "  Network: $NETWORK_CONFIG"
echo "  Output:  $OUTPUT_DIR"
echo "============================================"

weaver \
    --data-train \
    "HToBB:${DATA_DIR}/HToBB_000.root" \
    "ZToQQ:${DATA_DIR}/ZToQQ_000.root" \
    "TTBar:${DATA_DIR}/TTBar_000.root" \
    --data-test \
    "HToBB:${DATA_DIR}/HToBB_001.root" \
    "ZToQQ:${DATA_DIR}/ZToQQ_001.root" \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --network-option embed_dims '[32,128,32]' \
    --network-option pair_embed_dims '[16,16,16]' \
    --network-option num_heads 4 \
    --network-option num_layers 2 \
    --network-option num_cls_layers 1 \
    --network-option grad_cos_freq 10 \
    --model-prefix "$OUTPUT_DIR/test_{auto}" \
    --batch-size 512 \
    --start-lr 1e-3 \
    --num-epochs 2 \
    --optimizer ranger \
    --fetch-step 1 \
    --gpus 0 \
    --samples-per-epoch 10000 \
    --samples-per-epoch-val 5000 \
    --log "$OUTPUT_DIR/test_{auto}.log" \
    --tensorboard "$OUTPUT_DIR/tb" \
    "$@"

echo ""
echo "============================================"
echo "  Test complete! Check:"
echo "  - Log: $OUTPUT_DIR/test_*.log"
echo "  - TensorBoard: tensorboard --logdir $OUTPUT_DIR/tb"
echo "============================================"
