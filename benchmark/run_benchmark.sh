#!/bin/bash
#
# Automatic SGLang VLM Benchmark Pipeline
#
# Automatically cycles through models: start server -> benchmark -> stop server -> next
#
# Usage:
#   ./run_benchmark.sh                    # Run all models with all prompt types
#   ./run_benchmark.sh --quick            # Quick test (100 samples per model)
#   ./run_benchmark.sh --models "model1 model2"  # Custom model list
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#######################
# CONFIGURATION
#######################

# Server settings
PORT=30015
HOST_BIND="127.0.0.1"  # Named HOST_BIND to avoid conflict with conda's HOST env var
SERVER_URL="http://localhost:${PORT}/v1"

# Default model list (edit this to change models)
DEFAULT_MODELS=(
    # Qwen 2.5 VL series
    # "Qwen/Qwen2.5-VL-3B-Instruct:1"
    # "Qwen/Qwen2.5-VL-7B-Instruct:4"
    # "Qwen/Qwen2.5-VL-32B-Instruct:8"
    # Qwen3 VL series
    "Qwen/Qwen3-VL-4B-Instruct:1"
    "Qwen/Qwen3-VL-8B-Instruct:2"
    "Qwen/Qwen3-VL-32B-Instruct:8"
    # Gemma 3 (multimodal)
    "google/gemma-3-4b-it:1"
    "google/gemma-3-12b-it:4"
    # InternVL 3.5
    "OpenGVLab/InternVL3_5-4B:1"
    "OpenGVLab/InternVL3_5-8B:2"
    # # Llama Vision
    # "meta-llama/Llama-3.2-11B-Vision-Instruct:4"
    # # LLaVA
    # "llava-hf/llava-v1.6-mistral-7b-hf:2"
    # "llava-hf/llava-v1.6-34b-hf:4"
)

# Prompt types to test
PROMPT_TYPES=("zero_shot" "cot")

# Dataset settings - use local path from run_pipeline.sh
DATASET="/home/syx/robo2VLM/merged_dataset"
SPLIT="test"
MAX_SAMPLES=""  # Empty = all samples

# Benchmark settings
NUM_WORKERS=8
OUTPUT_DIR="results"

# Timing
SERVER_STARTUP_TIMEOUT=300  # 5 minutes max to wait for server
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
CUSTOM_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            MAX_SAMPLES=100
            shift
            ;;
        --models)
            IFS=' ' read -ra CUSTOM_MODELS <<< "$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --prompt-types)
            IFS=' ' read -ra PROMPT_TYPES <<< "$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            SERVER_URL="http://localhost:${PORT}/v1"
            shift 2
            ;;
        --help|-h)
            echo "Automatic SGLang VLM Benchmark"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Quick test with 100 samples"
            echo "  --models \"m1 m2\"     Space-separated model list (format: model:tp)"
            echo "  --max-samples N      Limit samples per benchmark"
            echo "  --prompt-types \"p1 p2\"  Prompt types (zero_shot, cot, direct)"
            echo "  --port PORT          Server port (default: 30000)"
            echo ""
            echo "Edit DEFAULT_MODELS array in script to change model list."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use custom models if provided
if [[ ${#CUSTOM_MODELS[@]} -gt 0 ]]; then
    DEFAULT_MODELS=("${CUSTOM_MODELS[@]}")
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
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi

    # Also kill by port
    fuser -k ${PORT}/tcp 2>/dev/null || true

    sleep "$SERVER_SHUTDOWN_WAIT"
    log "Server stopped"
}

start_server() {
    local model="$1"
    local tp="$2"
    local log_file="$LOG_DIR/server_${model//\//_}.log"

    log "Starting server for $model (TP=$tp)..."

    # Fix conda HOST variable bug (conda sets HOST=x86_64-conda-linux-gnu)
    # This breaks socket.gethostname() used by sglang/uvicorn
    unset HOST

    # Launch server in background
    nohup python3 -m sglang.launch_server \
        --model-path "$model" \
        --host "$HOST_BIND" \
        --port "$PORT" \
        --tp "$tp" \
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

        # Check if process died
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

run_benchmark_for_model() {
    local model="$1"

    log "Running benchmark for $model"

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        log "  Prompt type: $prompt_type"

        local cmd="python sglang_benchmark.py \
            --server-url $SERVER_URL \
            --model $model \
            --dataset $DATASET \
            --split $SPLIT \
            --prompt-type $prompt_type \
            --num-workers $NUM_WORKERS \
            --output-dir $OUTPUT_DIR"

        if [[ -n "$MAX_SAMPLES" ]]; then
            cmd="$cmd --max-samples $MAX_SAMPLES"
        fi

        eval $cmd || {
            error "Benchmark failed for $model with $prompt_type"
            # Continue with next prompt type
        }
    done
}

#######################
# MAIN
#######################

main() {
    log "=============================================="
    log "SGLang VLM Benchmark - Automatic Mode"
    log "=============================================="
    log "Models: ${#DEFAULT_MODELS[@]}"
    log "Prompt types: ${PROMPT_TYPES[*]}"
    log "Dataset: $DATASET ($SPLIT)"
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

    local total_models=${#DEFAULT_MODELS[@]}
    local current=0
    local failed_models=()

    for model_config in "${DEFAULT_MODELS[@]}"; do
        current=$((current + 1))

        # Parse model:tp format
        local model=$(echo "$model_config" | cut -d: -f1)
        local tp=$(echo "$model_config" | cut -d: -f2)
        tp=${tp:-4}  # Default to 4 if not specified

        echo ""
        log "=============================================="
        log "[$current/$total_models] $model"
        log "=============================================="

        # Stop any existing server
        stop_server

        # Start server for this model
        if ! start_server "$model" "$tp"; then
            error "Failed to start server for $model, skipping..."
            failed_models+=("$model")
            continue
        fi

        # Run benchmark
        run_benchmark_for_model "$model"

        # Stop server before next model
        stop_server
    done

    # Final summary
    echo ""
    log "=============================================="
    log "BENCHMARK COMPLETE"
    log "=============================================="
    log "Results saved to: $OUTPUT_DIR/"

    if [[ ${#failed_models[@]} -gt 0 ]]; then
        log "Failed models: ${failed_models[*]}"
    fi

    # Generate summary
    log "Generating comparison summary..."
    python -c "
import json
import glob
import os

results_dir = '$OUTPUT_DIR'
metrics_files = glob.glob(os.path.join(results_dir, '*_metrics.json'))

if not metrics_files:
    print('No results found')
    exit()

print()
print('=' * 80)
print('RESULTS SUMMARY')
print('=' * 80)
print(f'{\"Model\":<45} | {\"Prompt\":<12} | {\"Accuracy\":<10}')
print('-' * 80)

results = []
for f in sorted(metrics_files):
    try:
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    except:
        pass

for r in sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True):
    model = r.get('model', 'N/A')[:43]
    prompt = r.get('prompt_type', 'N/A')[:12]
    acc = r.get('accuracy', 0)
    print(f'{model:<45} | {prompt:<12} | {acc:>8.2f}%')

print('=' * 80)
" 2>/dev/null || true
}

# Run main
main
