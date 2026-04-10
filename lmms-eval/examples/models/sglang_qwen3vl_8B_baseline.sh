#!/bin/bash
set -uo pipefail

# Qwen3-VL Baseline Evaluation Script with SGLang Backend
#
# Requirements:
# - sglang>=0.4.6
# - qwen-vl-utils
# - CUDA-enabled GPU(s)

# ============================================================================
# Configuration
# ============================================================================

# Model to evaluate (HuggingFace model ID or local path)
MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Tasks to evaluate
TASKS="mme,mmstar,chartqa,realworldqa,mathvista_testmini_solution,mathverse_testmini,ai2d"

# Output directory (relative to working directory)
OUTPUT_PATH="./results/$(basename ${MODEL})"

# Parallelization Settings
TENSOR_PARALLEL_SIZE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Memory and Performance Settings
GPU_MEMORY_UTILIZATION=0.75
BATCH_SIZE=1

# SGLang Specific Settings
MAX_PIXELS=1605632
MIN_PIXELS=784
MAX_FRAME_NUM=32
THREADS=16

# Generation Configuration
GEN_KWARGS="max_new_tokens=4096,until="

# Reduce the chance of one TP worker crashing and taking down the whole scheduler.
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Evaluation
# ============================================================================

mkdir -p "${OUTPUT_PATH}"

echo "Model:  ${MODEL}"
echo "Tasks:  ${TASKS}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python -m lmms_eval \
    --model sglang \
    --model_args model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS},max_frame_num=${MAX_FRAME_NUM},threads=${THREADS} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --gen_kwargs ${GEN_KWARGS} \
    --log_samples \
    --log_samples_suffix baseline \
    --output_path ${OUTPUT_PATH}
