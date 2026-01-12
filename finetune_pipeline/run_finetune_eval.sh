#!/bin/bash
#
# SGLang-based Evaluation for Finetuned Models
#
# Automatically cycles through finetuned checkpoints: start server -> evaluate -> stop -> next
#
# Usage:
#   ./run_finetune_eval.sh                        # Run all checkpoints
#   ./run_finetune_eval.sh --quick                # Quick test (100 samples)
#   ./run_finetune_eval.sh --checkpoints "path1 path2"  # Custom checkpoint list
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#######################
# CONFIGURATION
#######################

# Server settings
PORT=30020
HOST_BIND="127.0.0.1"
SERVER_URL="http://localhost:${PORT}/v1"

# Base models for each checkpoint type
declare -A BASE_MODELS
BASE_MODELS["Qwen_Qwen2.5-VL-3B-Instruct"]="Qwen/Qwen2.5-VL-3B-Instruct"
BASE_MODELS["Qwen_Qwen2.5-VL-3B-Instruct_cot"]="Qwen/Qwen2.5-VL-3B-Instruct"
BASE_MODELS["Qwen_Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"
BASE_MODELS["Qwen_Qwen2.5-VL-7B-Instruct_cot"]="Qwen/Qwen2.5-VL-7B-Instruct"
BASE_MODELS["google_gemma-3-4b-it_cot"]="google/gemma-3-4b-it"

# Default checkpoints to evaluate (model_dir:samples:tp:mode)
# Format: model_directory:sample_count:tensor_parallel:evaluation_mode
DEFAULT_CHECKPOINTS=(
    # Qwen 3B CoT checkpoints
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:2500:8:cot"
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:5000:8:cot"
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:7500:8:cot"
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:10000:8:cot"
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:15000:8:cot"
    "Qwen_Qwen2.5-VL-3B-Instruct_cot:20000:8:cot"
    # Qwen 7B CoT checkpoints
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:2500:8:cot"
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:5000:8:cot"
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:7500:8:cot"
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:10000:8:cot"
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:15000:8:cot"
    "Qwen_Qwen2.5-VL-7B-Instruct_cot:20000:8:cot"
)

# Datasets to evaluate on
DATASETS=("merged_dataset")

# Evaluation settings
MAX_SAMPLES=""  # Empty = all samples
NUM_WORKERS=8
OUTPUT_DIR="results_sglang"

# Timing
SERVER_STARTUP_TIMEOUT=300
SERVER_SHUTDOWN_WAIT=10

# Conda environment
CONDA_ENV="robodm"

# Logging
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

#######################
# PARSE ARGUMENTS
#######################

QUICK_MODE=false
CUSTOM_CHECKPOINTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            MAX_SAMPLES=100
            shift
            ;;
        --checkpoints)
            IFS=' ' read -ra CUSTOM_CHECKPOINTS <<< "$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --datasets)
            IFS=' ' read -ra DATASETS <<< "$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            SERVER_URL="http://localhost:${PORT}/v1"
            shift 2
            ;;
        --help|-h)
            echo "SGLang-based Evaluation for Finetuned Models"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick                Quick test with 100 samples"
            echo "  --checkpoints \"c1 c2\"  Space-separated checkpoint configs"
            echo "                         Format: model_dir:samples:tp:mode"
            echo "  --max-samples N        Limit samples per evaluation"
            echo "  --datasets \"d1 d2\"     Datasets to evaluate on"
            echo "  --port PORT            Server port (default: 30020)"
            echo ""
            echo "Checkpoint format: model_dir:sample_count:tensor_parallel:mode"
            echo "  Example: Qwen_Qwen2.5-VL-3B-Instruct_cot:2500:1:cot"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use custom checkpoints if provided
if [[ ${#CUSTOM_CHECKPOINTS[@]} -gt 0 ]]; then
    DEFAULT_CHECKPOINTS=("${CUSTOM_CHECKPOINTS[@]}")
fi

#######################
# HELPER FUNCTIONS
#######################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

get_server_pid() {
    pgrep -f "sglang.launch_server.*--port $PORT" 2>/dev/null | head -1 || echo ""
}

stop_server() {
    log "Stopping any existing server on port $PORT..."

    local pid=$(get_server_pid)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi

    fuser -k ${PORT}/tcp 2>/dev/null || true
    sleep "$SERVER_SHUTDOWN_WAIT"
    log "Server stopped"
}

start_server() {
    local base_model="$1"
    local lora_path="$2"
    local tp="$3"
    local log_file="$LOG_DIR/server_$(basename $lora_path).log"

    log "Starting server for $base_model with LoRA: $lora_path (TP=$tp)..."

    # Fix conda HOST variable bug
    unset HOST

    # Launch server with LoRA
    # --lora-target-modules all is required because unsloth uses regex patterns
    nohup python3 -m sglang.launch_server \
        --model-path "$base_model" \
        --lora-path "$lora_path" \
        --lora-target-modules all \
        --host "$HOST_BIND" \
        --port "$PORT" \
        --tp "$tp" \
        --mem-fraction-static 0.7 \
        > "$log_file" 2>&1 &

    local server_pid=$!
    log "Server PID: $server_pid, log: $log_file"

    # Wait for server to be ready
    log "Waiting for server to be ready (timeout: ${SERVER_STARTUP_TIMEOUT}s)..."

    local waited=0
    while [[ $waited -lt $SERVER_STARTUP_TIMEOUT ]]; do
        if curl -s "${SERVER_URL}/models" > /dev/null 2>&1; then
            log "Server is ready!"
            return 0
        fi

        if ! ps -p "$server_pid" > /dev/null 2>&1; then
            error "Server process died. Check log: $log_file"
            tail -50 "$log_file"
            return 1
        fi

        sleep 5
        waited=$((waited + 5))

        if ((waited % 30 == 0)); then
            log "  Still waiting... ($waited/${SERVER_STARTUP_TIMEOUT}s)"
        fi
    done

    error "Server startup timeout after ${SERVER_STARTUP_TIMEOUT}s"
    return 1
}

run_evaluation() {
    local checkpoint_path="$1"
    local mode="$2"
    local checkpoint_name="$3"

    log "Running evaluation for $checkpoint_name (mode: $mode)"

    for dataset in "${DATASETS[@]}"; do
        log "  Dataset: $dataset"

        local output_file="${OUTPUT_DIR}/${checkpoint_name}_${dataset}.json"

        local cmd="python sglang_eval.py \
            --server-url $SERVER_URL \
            --dataset $dataset \
            --output-file $output_file \
            --mode $mode \
            --num-workers $NUM_WORKERS \
            --checkpoint-name $checkpoint_name"

        if [[ -n "$MAX_SAMPLES" ]]; then
            cmd="$cmd --max-samples $MAX_SAMPLES"
        fi

        eval $cmd || {
            error "Evaluation failed for $checkpoint_name on $dataset"
        }
    done
}

#######################
# MAIN
#######################

main() {
    log "=============================================="
    log "SGLang Finetuned Model Evaluation"
    log "=============================================="
    log "Checkpoints: ${#DEFAULT_CHECKPOINTS[@]}"
    log "Datasets: ${DATASETS[*]}"
    log "Max samples: ${MAX_SAMPLES:-all}"
    log "Output: $OUTPUT_DIR"
    log "=============================================="
    echo ""

    # Activate conda
    if command -v conda &> /dev/null; then
        log "Activating conda environment: $CONDA_ENV"
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi

    local total=${#DEFAULT_CHECKPOINTS[@]}
    local current=0
    local failed=()

    for checkpoint_config in "${DEFAULT_CHECKPOINTS[@]}"; do
        current=$((current + 1))

        # Parse config: model_dir:samples:tp:mode
        IFS=':' read -r model_dir samples tp mode <<< "$checkpoint_config"

        local checkpoint_path="outputs/${model_dir}/${samples}samples/checkpoint-final"
        local checkpoint_name="${model_dir}_${samples}samples"
        local base_model="${BASE_MODELS[$model_dir]}"

        if [[ -z "$base_model" ]]; then
            error "Unknown base model for $model_dir, skipping..."
            failed+=("$checkpoint_name")
            continue
        fi

        if [[ ! -d "$checkpoint_path" ]]; then
            error "Checkpoint not found: $checkpoint_path, skipping..."
            failed+=("$checkpoint_name")
            continue
        fi

        echo ""
        log "=============================================="
        log "[$current/$total] $checkpoint_name"
        log "Base model: $base_model"
        log "LoRA path: $checkpoint_path"
        log "Mode: $mode, TP: $tp"
        log "=============================================="

        # Stop any existing server
        stop_server

        # Start server with this checkpoint
        if ! start_server "$base_model" "$checkpoint_path" "$tp"; then
            error "Failed to start server for $checkpoint_name, skipping..."
            failed+=("$checkpoint_name")
            continue
        fi

        # Run evaluation
        run_evaluation "$checkpoint_path" "$mode" "$checkpoint_name"

        # Stop server before next checkpoint
        stop_server
    done

    # Final summary
    echo ""
    log "=============================================="
    log "EVALUATION COMPLETE"
    log "=============================================="
    log "Results saved to: $OUTPUT_DIR/"

    if [[ ${#failed[@]} -gt 0 ]]; then
        log "Failed: ${failed[*]}"
    fi

    # Generate summary
    log "Generating summary..."
    python3 -c "
import json
import glob
import os

results_dir = '$OUTPUT_DIR'
files = glob.glob(os.path.join(results_dir, '*.json'))

if not files:
    print('No results found')
    exit()

print()
print('=' * 90)
print('RESULTS SUMMARY')
print('=' * 90)
print(f'{\"Checkpoint\":<50} | {\"Dataset\":<15} | {\"Accuracy\":<10} | {\"Speed\":<12}')
print('-' * 90)

results = []
for f in sorted(files):
    try:
        with open(f) as fp:
            data = json.load(fp)
            results.append({
                'checkpoint': data.get('checkpoint', os.path.basename(f)),
                'dataset': data.get('dataset', 'N/A'),
                'accuracy': data.get('accuracy', 0) * 100,
                'speed': data.get('samples_per_second', 0),
            })
    except:
        pass

for r in sorted(results, key=lambda x: (x['checkpoint'], x['dataset'])):
    print(f\"{r['checkpoint'][:48]:<50} | {r['dataset']:<15} | {r['accuracy']:>8.2f}% | {r['speed']:>8.2f} s/s\")

print('=' * 90)
" 2>/dev/null || true
}

# Run main
main
