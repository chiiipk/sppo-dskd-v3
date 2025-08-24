set -e
set -x

export CUDA_VISIBLE_DEVICES="0,1"
AVAILABLE_GPUS=(0 1)
# HF_ORG=UCLA-AGI

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="data-mistral-7b-instruct-sppo-iter1"

PAIRS=5
FRAC=0
PROMPTS="/home/hungpv/projects/lab/DTW-v2/data/dolly/train.jsonl"

USE_TEACHER_LLM=""
TEACHER_MODEL=""
BATCH_SIZE=""
TENSOR_PARALLEL_SIZE=""
BT_METHOD=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --pairs)
        PAIRS="$2"
        shift
        ;;
    --frac)
        FRAC="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --out_path)
        OUTDIR="$2"
        shift
        ;;
    --prompt)
        PROMPTS="$2"
        shift
        ;;
    --use_teacher_llm)
        USE_TEACHER_LLM="--use_teacher_llm"
        ;;
    --teacher_model)
        TEACHER_MODEL="--teacher_model $2"
        shift
        ;;
    --batch_size)
        BATCH_SIZE="--batch_size $2"
        shift
        ;;
    --tensor_parallel_size)
        TENSOR_PARALLEL_SIZE="--tensor_parallel_size $2"
        shift
        ;;
    --bt_conversion_method)
        BT_METHOD="--bt_conversion_method $2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

#####################
# Generate Data
#####################

#frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes
echo "Using model ${MODEL}"
FRAC_LEN=6000
echo "Using frac_len ${FRAC_LEN}"
(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/generate.py --model $MODEL --maxlen 256 --output_dir "generated/$OUTDIR" --prompts $PROMPTS --pairs $PAIRS --world_size 1 --frac_len $FRAC_LEN --data_frac $data_frac > output_log_${gpu_id}.txt 2>&1 &
        ((data_frac+=1));
    done
    wait
) &
all_gen=$!

wait $all_gen

python3 scripts/combine_generate.py --output_dir "generated/$OUTDIR" --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --pairs $PAIRS


# #####################
# # Rank Data
# #####################

# # frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes

python3 scripts/preload.py

(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py --model $MODEL --output_dir $OUTDIR --pairs $PAIRS --numgpu ${#AVAILABLE_GPUS[@]} --frac_len $FRAC_LEN --data_frac $data_frac --gpu $gpu_id --prompts $PROMPTS $USE_TEACHER_LLM $TEACHER_MODEL $BATCH_SIZE $TENSOR_PARALLEL_SIZE $BT_METHOD > rank_log_${gpu_id}.txt 2>&1 &
        ((data_frac+=1));
    done
    wait
) &
all_rank=$!

wait $all_rank

python3 scripts/compute_prob.py --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --output_dir $OUTDIR --pairs $PAIRS --frac_len $FRAC_LEN --prompts $PROMPTS
