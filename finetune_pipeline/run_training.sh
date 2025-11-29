#!/bin/bash
#
# Parallel Training Script
#
# Trains each model ONCE, saving checkpoints at intervals (10k, 20k, ..., 100k samples).
# Each model runs on its own GPU(s) in parallel.
#
# Usage:
#   ./run_training.sh                    # Train all models
#   ./run_training.sh --quick            # Quick test (10k samples only)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#######################
# CONFIGURATION
#######################

# Total GPUs available
NUM_GPUS=8

# Models and their GPU requirements (format: "model_name:num_gpus")
# Note: InternVL models not supported by Unsloth
MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct:1"
    "Qwen/Qwen2.5-VL-7B-Instruct:2"
    # "meta-llama/Llama-3.2-11B-Vision-Instruct:2"
    "google/gemma-3-4b-it:1"      # Add this
    "google/gemma-3-12b-it:2"     # Add this
)

# Checkpoint intervals (save checkpoint after processing this many samples)
CHECKPOINT_INTERVALS="10000,15000,20000"

# Max samples to train on
MAX_SAMPLES=20000

# Output directory
OUTPUT_DIR="outputs"

# Logs directory
LOG_DIR="$SCRIPT_DIR/logs/training"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Conda environment
CONDA_ENV="unsloth_env"

# Wandb
USE_WANDB=true

# Training mode: zero_shot or cot
MODE="cot"

#######################
# PARSE ARGUMENTS
#######################

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            CHECKPOINT_INTERVALS="10000"
            MAX_SAMPLES=10000
            shift
            ;;
        --intervals)
            CHECKPOINT_INTERVALS="$2"
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
        --no-wandb)
            USE_WANDB=false
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Parallel Training Script"
            echo ""
            echo "Trains each model once, saving checkpoints at specified intervals."
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick                Quick test (10k samples only)"
            echo "  --intervals 10000,50000,100000  Checkpoint intervals"
            echo "  --max-samples N        Max training samples (default: 100000)"
            echo "  --models \"m1:g1 m2:g2\" Space-separated model list"
            echo "  --no-wandb             Disable wandb logging"
            echo "  --mode MODE            Training mode: zero_shot or cot (default: zero_shot)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

#######################
# HELPER FUNCTIONS
#######################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Parse model string to get name and GPU count
parse_model() {
    local model_str="$1"
    local name="${model_str%:*}"
    local gpus="${model_str##*:}"
    if [[ "$name" == "$gpus" ]]; then
        gpus=1
    fi
    echo "$name $gpus"
}

# Get model slug (replace / with _)
get_slug() {
    echo "$1" | tr '/' '_'
}

# Activate conda
activate_conda() {
    if command -v conda &> /dev/null; then
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
}

#######################
# GPU ALLOCATION
#######################

# Track GPU usage: GPU_USED[i]=PID or empty
declare -a GPU_USED
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_USED[$i]=""
done

# Track all PIDs
declare -a ALL_PIDS
declare -A PID_TO_MODEL

# Update GPU availability by checking if processes are still running
update_gpu_status() {
    for ((i=0; i<NUM_GPUS; i++)); do
        local pid="${GPU_USED[$i]}"
        if [[ -n "$pid" ]]; then
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished, GPU is free
                GPU_USED[$i]=""
            fi
        fi
    done
}

# Try to allocate N contiguous GPUs, return comma-separated list or empty
try_allocate_gpus() {
    local needed=$1
    update_gpu_status

    local count=0
    local gpus=""

    for ((i=0; i<NUM_GPUS; i++)); do
        if [[ -z "${GPU_USED[$i]}" ]]; then
            if [[ -z "$gpus" ]]; then
                gpus="$i"
            else
                gpus="$gpus,$i"
            fi
            ((count++))
            if [[ $count -ge $needed ]]; then
                echo "$gpus"
                return 0
            fi
        fi
    done

    echo ""
    return 1
}

# Mark GPUs as used by PID
mark_gpus_used() {
    local gpu_list="$1"
    local pid="$2"

    IFS=',' read -ra gpus <<< "$gpu_list"
    for gpu in "${gpus[@]}"; do
        GPU_USED[$gpu]="$pid"
    done
}

# Wait for GPUs to become available
wait_for_gpus() {
    local needed=$1
    while true; do
        local gpus=$(try_allocate_gpus "$needed")
        if [[ -n "$gpus" ]]; then
            echo "$gpus"
            return
        fi
        sleep 10
    done
}

#######################
# JOB MANAGEMENT
#######################

# Run a single training job (one model, all checkpoints)
run_training_job() {
    local model_name="$1"
    local gpu_ids="$2"

    local slug=$(get_slug "$model_name")
    local log_file="$LOG_DIR/${slug}.log"

    local wandb_flag=""
    if [[ "$USE_WANDB" == "false" ]]; then
        wandb_flag="--no-wandb"
    fi

    log "Starting: $model_name on GPUs: $gpu_ids"

    CUDA_VISIBLE_DEVICES="$gpu_ids" python train_single.py \
        --model "$model_name" \
        --max-samples "$MAX_SAMPLES" \
        --checkpoint-intervals "$CHECKPOINT_INTERVALS" \
        --output-dir "$OUTPUT_DIR" \
        --mode "$MODE" \
        $wandb_flag \
        > "$log_file" 2>&1 &

    local pid=$!

    # Mark GPUs as used
    mark_gpus_used "$gpu_ids" "$pid"

    # Track PID
    ALL_PIDS+=($pid)
    PID_TO_MODEL[$pid]="$model_name"

    log "Started PID $pid for $model_name on GPUs $gpu_ids"
}

# Wait for all jobs to complete
wait_all_jobs() {
    log "Waiting for all jobs to complete..."
    for pid in "${ALL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Waiting for PID $pid: ${PID_TO_MODEL[$pid]}"
            wait "$pid" && log "PID $pid completed: ${PID_TO_MODEL[$pid]}" || log "PID $pid FAILED: ${PID_TO_MODEL[$pid]}"
        fi
    done
}

#######################
# MAIN
#######################

main() {
    log "=============================================="
    log "Parallel Training Pipeline"
    log "=============================================="
    log "Models: ${#MODELS[@]}"
    log "Mode: $MODE"
    log "Checkpoint intervals: $CHECKPOINT_INTERVALS"
    log "Max samples: $MAX_SAMPLES"
    log "GPUs: $NUM_GPUS"
    log "Output: $OUTPUT_DIR"
    log "Logs: $LOG_DIR"
    log "=============================================="
    echo ""

    activate_conda

    # Launch one job per model
    for model_str in "${MODELS[@]}"; do
        read model_name num_gpus <<< $(parse_model "$model_str")

        log "Requesting $num_gpus GPU(s) for $model_name..."

        # Wait for GPUs
        gpu_ids=$(wait_for_gpus "$num_gpus")

        # Run job
        run_training_job "$model_name" "$gpu_ids"

        # Small delay
        sleep 2
    done

    # Wait for all remaining jobs
    wait_all_jobs

    log "=============================================="
    log "TRAINING COMPLETE"
    log "=============================================="
    log "Checkpoints saved to: $OUTPUT_DIR"
    log "Logs saved to: $LOG_DIR"
    log ""
    log "Next step: Run ./run_eval.sh to evaluate all checkpoints"
}

main
