#!/bin/bash

set -e
set -x

echo "=== SINGLE ITERATION KNOWLEDGE DISTILLATION ==="

# Student model khởi tạo từ GPT-2 nhỏ
CUSTOM_MODEL_PATH="/kaggle/working/gpt2"
ITER=1
MODEL=$CUSTOM_MODEL_PATH

# Checkpoint output
OUTPUT_DIR="/kaggle/working/checkpoints/gpt2-kd-qwen-dolly-iter${ITER}"

# Prompts input dataset
PROMPT="/kaggle/working/data/dolly/train.jsonl"

# OUT: intermediate output (generated + ranking + combined)
OUT="/kaggle/working/kd-gpt2-qwen-dolly-iter${ITER}"

# Dataset sau compute_prob
DATASET="/kaggle/working/synthetic_data_gpt2-qwen-dolly-iter${ITER}_score"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Output directory: $OUTPUT_DIR"
echo "  Intermediate OUT: $OUT"
echo "  Dataset: $DATASET"
echo "  Prompts: $PROMPT"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: Model directory does not exist: $MODEL"
    exit 1
fi

if [ ! -f "$PROMPT" ]; then
    echo "ERROR: Prompts file does not exist: $PROMPT"
    exit 1
fi

echo "=== Step 1: Generating and ranking responses ==="
bash /kaggle/working/sppo-dskd-v3/scripts/generate.sh \
    --model $MODEL \
    --prompts $PROMPT \
    --output_dir $OUT \
    --use_teacher_llm \
    --teacher_model Qwen/Qwen1.5-1.8B-Chat \
    --batch_size 8 \
    --tensor_parallel_size 1 \
    --bt_conversion_method bradley_terry_mle


echo "=== Step 3: Computing probabilities and preparing dataset ==="
python3 /kaggle/working/sppo-dskd-v3/scripts/compute_prob.py \
    --gpu_ids "0,1" \
    --output_dir $OUT \
    --pairs 5 \
    --frac_len 6000 \
    --prompts $PROMPT

echo "=== Step 4: Training GPT-2 student model with SPPO ==="
bash /kaggle/working/sppo-dskd-v3/scripts/pipeline.sh \
    --model $MODEL \
    --iter $ITER \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --num 1 \
    --learning_rate 1.0e-6 \
    --batch_size 16

echo "=== Single iteration complete! ==="
echo "Checkpoint saved to: $OUTPUT_DIR"
