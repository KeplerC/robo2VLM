#!/bin/bash
#
# Reasoning Dataset Generation Pipeline
#
# Generates Chain-of-Thought reasoning for the merged VQA dataset
# using Qwen2.5-VL-72B via SGLang server.
#
# Usage:
#   ./run_reasoning.sh                    # Full pipeline (requires server running)
#   ./run_reasoning.sh --start-server     # Start server in background, then generate
#   ./run_reasoning.sh --test             # Test with 100 samples
#
# Prerequisites:
#   1. Start the SGLang server first (in a separate terminal):
#      ./launch_server.sh --model Qwen/Qwen2.5-VL-72B-Instruct --tp 4
#
#   2. Or use --start-server flag to start automatically
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#######################
# CONFIGURATION
#######################

# Server settings
SERVER_URL="http://localhost:30000/v1"
MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
TP=8

# Dataset paths
INPUT_DATASET="/home/syx/robo2VLM/merged_dataset"
OUTPUT_DATASET="/home/syx/robo2VLM/reasoning_dataset"
CHECKPOINT_DIR="/home/syx/robo2VLM/reasoning_checkpoints"

# Generation settings
NUM_WORKERS=16
TEMPERATURE=0.7
MAX_TOKENS=10240
MAX_TRAIN_SAMPLES=100000  # Max samples to randomly sample from training set

# HuggingFace (optional)
HF_REPO=""  # Set to push to HuggingFace, e.g., "keplerccc/robo2vlm-reasoning"

# Conda environment
CONDA_ENV="robodm"

#######################
# PARSE ARGUMENTS
#######################

START_SERVER=false
TEST_MODE=false
CUSTOM_MAX_TRAIN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-server)
            START_SERVER=true
            shift
            ;;
        --test)
            TEST_MODE=true
            CUSTOM_MAX_TRAIN="1000"
            shift
            ;;
        --max-train-samples)
            CUSTOM_MAX_TRAIN="$2"
            shift 2
            ;;
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --push-to-hub)
            HF_REPO="$2"
            shift 2
            ;;
        --help|-h)
            echo "Reasoning Dataset Generation Pipeline"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --start-server          Start SGLang server in background before generating"
            echo "  --test                  Test mode: process only 1000 training samples"
            echo "  --max-train-samples N   Max samples to randomly sample from training (default: 100000)"
            echo "  --server-url URL        SGLang server URL (default: http://localhost:30000/v1)"
            echo "  --workers N             Number of concurrent workers (default: 16)"
            echo "  --push-to-hub REPO      Push to HuggingFace repo"
            echo "  --help                  Show this help message"
            echo ""
            echo "Note: Only training set gets reasoning generation."
            echo "      Validation and test sets are kept unchanged."
            echo ""
            echo "Examples:"
            echo "  $0                              # Run with server already started (100k samples)"
            echo "  $0 --start-server               # Start server and run"
            echo "  $0 --test                       # Quick test with 1000 samples"
            echo "  $0 --max-train-samples 50000    # Process 50k training samples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -n "$CUSTOM_MAX_TRAIN" ]]; then
    MAX_TRAIN_SAMPLES="$CUSTOM_MAX_TRAIN"
fi

#######################
# HELPER FUNCTIONS
#######################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

wait_for_server() {
    local url="$1"
    local max_attempts=60
    local attempt=0

    log "Waiting for server at $url..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "${url}/models" > /dev/null 2>&1; then
            log "Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    error "Server did not start within timeout"
}

check_server() {
    if ! curl -s "${SERVER_URL}/models" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

#######################
# MAIN
#######################

main() {
    log "=============================================="
    log "Reasoning Dataset Generation Pipeline"
    log "=============================================="
    log "Model: $MODEL"
    log "Server: $SERVER_URL"
    log "Input: $INPUT_DATASET"
    log "Output: $OUTPUT_DATASET"
    log "Max train samples: $MAX_TRAIN_SAMPLES"
    log "Workers: $NUM_WORKERS"
    log "=============================================="
    log "Note: Only training set gets reasoning."
    log "      Validation/test sets kept unchanged."
    log "=============================================="

    # Activate conda environment
    log "Activating conda environment: $CONDA_ENV"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"

    # Start server if requested
    if $START_SERVER; then
        log "Starting SGLang server in background..."

        # Check if server already running
        if check_server; then
            log "Server already running at $SERVER_URL"
        else
            # Start server in background
            nohup ./launch_server.sh --model "$MODEL" --tp "$TP" > server.log 2>&1 &
            SERVER_PID=$!
            log "Server started with PID: $SERVER_PID"
            log "Server logs: $SCRIPT_DIR/server.log"

            # Wait for server to be ready
            wait_for_server "$SERVER_URL"
        fi
    else
        # Check server is running
        if ! check_server; then
            error "Server not running at $SERVER_URL. Start with: ./launch_server.sh"
        fi
        log "Server is running at $SERVER_URL"
    fi

    # Build command
    CMD="python generate_reasoning.py"
    CMD="$CMD --server-url \"$SERVER_URL\""
    CMD="$CMD --model \"$MODEL\""
    CMD="$CMD --input \"$INPUT_DATASET\""
    CMD="$CMD --output \"$OUTPUT_DATASET\""
    CMD="$CMD --checkpoint-dir \"$CHECKPOINT_DIR\""
    CMD="$CMD --num-workers $NUM_WORKERS"
    CMD="$CMD --temperature $TEMPERATURE"
    CMD="$CMD --max-tokens $MAX_TOKENS"
    CMD="$CMD --max-train-samples $MAX_TRAIN_SAMPLES"

    if [[ -n "$HF_REPO" ]]; then
        CMD="$CMD --push-to-hub \"$HF_REPO\""
    fi

    # Run generation
    log "Starting reasoning generation..."
    log "Command: $CMD"
    echo ""

    eval $CMD

    log "=============================================="
    log "Pipeline completed successfully!"
    log "=============================================="
    log "Output saved to: $OUTPUT_DATASET"
    if [[ -n "$HF_REPO" ]]; then
        log "Pushed to: https://huggingface.co/datasets/$HF_REPO"
    fi
}

main
