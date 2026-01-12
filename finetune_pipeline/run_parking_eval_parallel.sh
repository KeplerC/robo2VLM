#!/bin/bash
# Evaluate all checkpoints on the parking benchmark in parallel using 8 GPUs
# Usage: bash run_parking_eval_parallel.sh

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate unsloth_env 2>/dev/null || true

RESULTS_DIR="results_parking"
OUTPUTS_DIR="outputs"

mkdir -p $RESULTS_DIR

echo "====================================="
echo "Parking Benchmark Evaluation (8 GPUs)"
echo "====================================="

# Build list of all evaluation jobs
declare -a JOBS=()

# Base models (0 samples) - zero_shot mode
JOBS+=("Qwen/Qwen2.5-VL-3B-Instruct|${RESULTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct_0samples.json|zero_shot|model")
JOBS+=("Qwen/Qwen2.5-VL-7B-Instruct|${RESULTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct_0samples.json|zero_shot|model")

# Finetuned models (zero-shot training) - zero_shot mode
for samples in 5000 10000; do
    if [ -d "${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct/${samples}samples/checkpoint-final" ]; then
        JOBS+=("${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct/${samples}samples/checkpoint-final|${RESULTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct_${samples}samples.json|zero_shot|checkpoint")
    fi
    if [ -d "${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct/${samples}samples/checkpoint-final" ]; then
        JOBS+=("${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct/${samples}samples/checkpoint-final|${RESULTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct_${samples}samples.json|zero_shot|checkpoint")
    fi
done

# COT finetuned models - cot mode
for samples in 2500 5000 7500 10000 15000 20000; do
    if [ -d "${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct_cot/${samples}samples/checkpoint-final" ]; then
        JOBS+=("${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct_cot/${samples}samples/checkpoint-final|${RESULTS_DIR}/Qwen_Qwen2.5-VL-3B-Instruct_cot_${samples}samples.json|cot|checkpoint")
    fi
    if [ -d "${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct_cot/${samples}samples/checkpoint-final" ]; then
        JOBS+=("${OUTPUTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct_cot/${samples}samples/checkpoint-final|${RESULTS_DIR}/Qwen_Qwen2.5-VL-7B-Instruct_cot_${samples}samples.json|cot|checkpoint")
    fi
done

NUM_JOBS=${#JOBS[@]}
NUM_GPUS=8

echo "Total jobs: $NUM_JOBS"
echo "Using $NUM_GPUS GPUs"
echo ""

# Function to run a single job
run_job() {
    local gpu_id=$1
    local job_spec=$2

    IFS='|' read -r path output_file mode type <<< "$job_spec"

    if [ "$type" == "model" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id python eval_parking.py \
            --model "$path" \
            --output-file "$output_file" \
            --mode "$mode"
    else
        CUDA_VISIBLE_DEVICES=$gpu_id python eval_parking.py \
            --checkpoint "$path" \
            --output-file "$output_file" \
            --mode "$mode"
    fi
}

export -f run_job
export RESULTS_DIR OUTPUTS_DIR

# Run jobs in parallel, 8 at a time
job_idx=0
pids=()

for job in "${JOBS[@]}"; do
    gpu_id=$((job_idx % NUM_GPUS))

    echo "[GPU $gpu_id] Starting: $job"
    run_job $gpu_id "$job" &
    pids+=($!)

    job_idx=$((job_idx + 1))

    # If we've launched NUM_GPUS jobs, wait for all to complete before launching more
    if [ ${#pids[@]} -ge $NUM_GPUS ]; then
        echo "Waiting for batch to complete..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        pids=()
        echo "Batch complete. Launching next batch..."
    fi
done

# Wait for remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo "Waiting for final batch to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
fi

echo ""
echo "====================================="
echo "All evaluations complete!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "====================================="

# Print summary
echo ""
echo "Summary of results:"
for f in ${RESULTS_DIR}/*.json; do
    if [ -f "$f" ]; then
        accuracy=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']*100:.2f}%\")")
        echo "  $(basename $f): $accuracy"
    fi
done
