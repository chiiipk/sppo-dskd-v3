#!/bin/bash

set -e
set -x

echo "=== SINGLE ITERATION KNOWLEDGE DISTILLATION ==="

CUSTOM_MODEL_PATH="/kaggle/working/gpt2"
ITER=1
MODEL=$CUSTOM_MODEL_PATH
OUTPUT_DIR="checkpoints/gpt2-kd-qwen-dolly-iter${ITER}"
PROMPT="/kaggle/working/data/dolly/train.jsonl"
OUT="kd-gpt2-qwen-dolly-iter${ITER}"
DATASET="synthetic_data_gpt2-qwen-dolly-iter${ITER}_score"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Output directory: $OUTPUT_DIR"
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
bash scripts/generate.sh \
    --model $MODEL \
    --prompt $PROMPT \
    --out_path $OUT \
    --use_teacher_llm \
    --teacher_model Qwen/Qwen1.5-1.8B-Chat \
    --batch_size 8 \
    --tensor_parallel_size 1 \
    --bt_conversion_method bradley_terry_mle

echo "=== Step 2: Computing probabilities and preparing dataset ==="
python3 scripts/compute_prob.py \
    --gpu_ids "0,1" \
    --output_dir $OUT \
    --pairs 5 \
    --frac_len 6000 \
    --prompts $PROMPT

echo "=== Step 3: Training GPT-2 student model with SPPO ==="
bash scripts/pipeline.sh \
    --model $MODEL \
    --iter $ITER \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --num 1 \
    --learning_rate 1.0e-6 \
    --batch_size 16

echo "=== Single iteration complete! ==="
echo "Checkpoint saved to: $OUTPUT_DIR"
