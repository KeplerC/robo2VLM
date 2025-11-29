#!/bin/bash
#
# Parallel Evaluation Script
#
# Evaluates all checkpoints on all test datasets in parallel.
# Each evaluation job runs on one GPU.
#
# Usage:
#   ./run_eval.sh                        # Evaluate all checkpoints
#   ./run_eval.sh --dataset merged_dataset  # Evaluate on specific dataset
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#######################
# CONFIGURATION
#######################

# Total GPUs available
NUM_GPUS=8

# Models (format: "model_name:num_gpus" - gpus not used for eval, but used for checkpoint detection)
# Note: InternVL models not supported by Unsloth
MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct:1"
    "Qwen/Qwen2.5-VL-7B-Instruct:2"
    # "meta-llama/Llama-3.2-11B-Vision-Instruct:2"
    "google/gemma-3-4b-it:1"      # Add this
    "google/gemma-3-12b-it:2"     # Add this
)

# Training sample sizes (0 = base model)
SAMPLE_SIZES=(0 5000 10000 15000 20000)

# Test datasets
DATASETS=("merged_dataset" "ERQA")

# Directories
OUTPUT_DIR="outputs"
RESULTS_DIR="results"
LOG_DIR="$SCRIPT_DIR/logs/eval"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Conda environment
CONDA_ENV="unsloth_env"

# Evaluation mode: zero_shot or cot
MODE="cot"

#######################
# PARSE ARGUMENTS
#######################

FILTER_DATASET=""
MAX_SAMPLES="500"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            FILTER_DATASET="$2"
            shift 2
            ;;
        --samples)
            IFS=',' read -ra SAMPLE_SIZES <<< "$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --models)
            IFS=' ' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Parallel Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME         Evaluate only on specific dataset"
            echo "  --samples 0,10000,...  Comma-separated sample sizes to evaluate"
            echo "  --max-samples N        Max samples per evaluation"
            echo "  --models \"m1:g1 m2:g2\" Space-separated model list"
            echo "  --mode MODE            Evaluation mode: zero_shot or cot (default: zero_shot)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Filter datasets if specified
if [[ -n "$FILTER_DATASET" ]]; then
    DATASETS=("$FILTER_DATASET")
fi

#######################
# HELPER FUNCTIONS
#######################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Get model slug (replace / with _)
get_slug() {
    echo "$1" | tr '/' '_'
}

# Get dataset slug (replace / and - with _)
get_dataset_slug() {
    echo "$1" | tr '/-' '_'
}

# Parse model string to get name
parse_model() {
    local model_str="$1"
    echo "${model_str%:*}"
}

# Activate conda
activate_conda() {
    if command -v conda &> /dev/null; then
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
}

#######################
# PARALLEL JOB MANAGEMENT
#######################

# Track running jobs
declare -A RUNNING_JOBS
declare -A JOB_GPUS

# Clean up finished jobs and return available GPUs
get_available_gpus() {
    local used=""
    for pid in "${!JOB_GPUS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            used="$used ${JOB_GPUS[$pid]}"
        else
            unset JOB_GPUS[$pid]
            unset RUNNING_JOBS[$pid]
        fi
    done

    local available=()
    for ((i=0; i<NUM_GPUS; i++)); do
        if [[ ! " $used " =~ " $i " ]]; then
            available+=($i)
        fi
    done
    echo "${available[@]}"
}

# Wait for a GPU to become available
wait_for_gpu() {
    while true; do
        local available=($(get_available_gpus))
        if [[ ${#available[@]} -gt 0 ]]; then
            echo "${available[0]}"
            return
        fi
        sleep 5
    done
}

# Run evaluation job
run_eval_job() {
    local model_name="$1"
    local num_samples="$2"
    local dataset="$3"
    local gpu_id="$4"

    local slug=$(get_slug "$model_name")
    # Add mode suffix for cot checkpoints and results
    local slug_with_mode="$slug"
    if [[ "$MODE" == "cot" ]]; then
        slug_with_mode="${slug}_cot"
    fi

    local ds_slug=$(get_dataset_slug "$dataset")
    local result_file="$RESULTS_DIR/${slug_with_mode}_${num_samples}samples_${ds_slug}.json"
    local log_file="$LOG_DIR/${slug_with_mode}_${num_samples}samples_${ds_slug}.log"

    # Build command
    local cmd="python eval_single.py"

    if [[ "$num_samples" -eq 0 ]]; then
        # Base model
        cmd="$cmd --model \"$model_name\""
    else
        # Finetuned model - use slug_with_mode for checkpoint path
        local checkpoint="$OUTPUT_DIR/$slug_with_mode/${num_samples}samples/checkpoint-final"
        cmd="$cmd --checkpoint \"$checkpoint\""
    fi

    cmd="$cmd --dataset \"$dataset\" --output-file \"$result_file\" --mode \"$MODE\""

    if [[ -n "$MAX_SAMPLES" ]]; then
        cmd="$cmd --max-samples $MAX_SAMPLES"
    fi

    log "Starting eval: $model_name ($num_samples samples, mode=$MODE) on $dataset [GPU $gpu_id]"

    CUDA_VISIBLE_DEVICES="$gpu_id" eval $cmd > "$log_file" 2>&1 &

    local pid=$!
    RUNNING_JOBS[$pid]="$model_name:$num_samples:$dataset"
    JOB_GPUS[$pid]="$gpu_id"
}

# Wait for all jobs
wait_all_jobs() {
    for pid in "${!RUNNING_JOBS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Waiting for PID $pid: ${RUNNING_JOBS[$pid]}"
            wait "$pid" || log "Job $pid failed: ${RUNNING_JOBS[$pid]}"
        fi
    done
    RUNNING_JOBS=()
    JOB_GPUS=()
}

#######################
# MAIN
#######################

main() {
    log "=============================================="
    log "Parallel Evaluation Pipeline"
    log "=============================================="
    log "Models: ${#MODELS[@]}"
    log "Mode: $MODE"
    log "Sample sizes: ${SAMPLE_SIZES[*]}"
    log "Datasets: ${DATASETS[*]}"
    log "GPUs: $NUM_GPUS"
    log "Results: $RESULTS_DIR"
    log "=============================================="
    echo ""

    activate_conda

    # Create job queue
    declare -a JOB_QUEUE

    for model_str in "${MODELS[@]}"; do
        model_name=$(parse_model "$model_str")
        slug=$(get_slug "$model_name")
        # Add mode suffix for cot checkpoints and results
        slug_with_mode="$slug"
        if [[ "$MODE" == "cot" ]]; then
            slug_with_mode="${slug}_cot"
        fi

        for samples in "${SAMPLE_SIZES[@]}"; do
            # For non-zero samples, check if checkpoint exists
            if [[ "$samples" -ne 0 ]]; then
                checkpoint="$OUTPUT_DIR/$slug_with_mode/${samples}samples/checkpoint-final"
                if [[ ! -d "$checkpoint" ]]; then
                    log "Skipping (no checkpoint): $model_name with $samples samples (mode=$MODE)"
                    continue
                fi
            fi

            for dataset in "${DATASETS[@]}"; do
                ds_slug=$(get_dataset_slug "$dataset")
                result_file="$RESULTS_DIR/${slug_with_mode}_${samples}samples_${ds_slug}.json"

                # Skip if result already exists
                if [[ -f "$result_file" ]]; then
                    log "Skipping (exists): $model_name ($samples, mode=$MODE) on $dataset"
                    continue
                fi

                JOB_QUEUE+=("$model_name:$samples:$dataset")
            done
        done
    done

    log "Total evaluation jobs: ${#JOB_QUEUE[@]}"
    echo ""

    if [[ ${#JOB_QUEUE[@]} -eq 0 ]]; then
        log "No jobs to run. All evaluations complete or no checkpoints found."
        log "Run ./run_training.sh first to create checkpoints."
        exit 0
    fi

    # Process job queue
    for job in "${JOB_QUEUE[@]}"; do
        IFS=':' read -r model_name num_samples dataset <<< "$job"

        # Wait for GPU
        gpu_id=$(wait_for_gpu)

        # Run job
        run_eval_job "$model_name" "$num_samples" "$dataset" "$gpu_id"

        # Small delay
        sleep 2
    done

    # Wait for all remaining jobs
    wait_all_jobs

    log "=============================================="
    log "EVALUATION COMPLETE"
    log "=============================================="
    log "Results saved to: $RESULTS_DIR"
    log "Logs saved to: $LOG_DIR"
    log ""
    log "Next step: Run python aggregate_results.py to generate summary"
}

main
