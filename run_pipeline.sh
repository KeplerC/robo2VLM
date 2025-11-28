#!/bin/bash
#
# VQA Dataset Pipeline: Generate -> Sample -> Reasoning -> Merge -> Save
#
# Usage:
#   ./run_pipeline.sh                    # Run full pipeline
#   ./run_pipeline.sh --generate-only    # Only generate VQA data
#   ./run_pipeline.sh --merge-only       # Sample, reason, merge (skip VQA generation)
#   ./run_pipeline.sh --skip-reasoning   # Skip reasoning generation
#

set -e  # Exit on error

#######################
# CONFIGURATION
#######################

# Agibot dataset paths
AGIBOT_DATA_ROOT="/shared/projects/agibot/agibot_alpha_full"
AGIBOT_VQA_OUTPUT="/home/syx/robo2VLM/vqa_output14000new3"

# VQA generation settings
NUM_EPISODES=30000
NUM_WORKERS=128
STATE_SAMPLES=1
SEGMENT_SAMPLES=5
TOTAL_VQAS=50000

# HuggingFace source
HF_SOURCE_REPO="keplerccc/ManipulationVQA-60k"

# Output path
MERGED_DATASET_PATH="/home/syx/robo2VLM/merged_dataset2"

# Sampling settings
MANIPULATION_SAMPLES=16065 # Samples from ManipulationVQA-60k
AGIBOT_SAMPLES=10000        # Samples from local Agibot VQA

# Reasoning settings
REASONING_SERVER_URL="http://localhost:30000/v1"
REASONING_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
REASONING_WORKERS=16
REASONING_TP=8
REASONING_SERVER_LOG="/home/syx/robo2VLM/reasoning_pipeline/server.log"

# Conda environment
CONDA_ENV="robodm"

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

check_reasoning_server() {
    if ! curl -s "${REASONING_SERVER_URL}/models" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

wait_for_reasoning_server() {
    local max_attempts=120
    local attempt=0

    log "Waiting for reasoning server to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        if check_reasoning_server; then
            log "Reasoning server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $((attempt % 12)) -eq 0 ]; then
            log "Still waiting for server... (${attempt}/${max_attempts})"
        fi
        sleep 5
    done

    error "Reasoning server did not start within timeout (10 minutes)"
}

start_reasoning_server() {
    log "Starting reasoning server in background..."
    log "Model: $REASONING_MODEL"
    log "TP: $REASONING_TP GPUs"

    cd /home/syx/robo2VLM/reasoning_pipeline

    nohup python3 -m sglang.launch_server \
        --model-path "$REASONING_MODEL" \
        --host "0.0.0.0" \
        --port 30000 \
        --tp "$REASONING_TP" \
        --trust-remote-code \
        --mem-fraction-static 0.8 \
        > "$REASONING_SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    log "Server started with PID: $SERVER_PID"

    wait_for_reasoning_server
}

#######################
# PIPELINE STEPS
#######################

step_generate_vqa() {
    log "=========================================="
    log "STEP 1: Generating VQA data from Agibot"
    log "=========================================="

    log "Data root: $AGIBOT_DATA_ROOT"
    log "Output dir: $AGIBOT_VQA_OUTPUT"
    log "Num episodes: $NUM_EPISODES"
    log "Target VQAs: $TOTAL_VQAS"

    cd /home/syx/robo2VLM/generation_agibot

    python generate_dataset.py \
        --data-root "$AGIBOT_DATA_ROOT" \
        --output-dir "$AGIBOT_VQA_OUTPUT" \
        --all-tasks \
        --num-episodes "$NUM_EPISODES" \
        --num-workers "$NUM_WORKERS" \
        --state-samples "$STATE_SAMPLES" \
        --segment-samples "$SEGMENT_SAMPLES" \
        --total-vqas "$TOTAL_VQAS"

    log "VQA generation complete!"
}

step_sample_reason_merge() {
    log "=========================================="
    log "STEP 2: Sample -> Reasoning -> Merge"
    log "=========================================="

    log "ManipulationVQA samples: $MANIPULATION_SAMPLES"
    log "Agibot VQA samples: $AGIBOT_SAMPLES"
    log "Output: $MERGED_DATASET_PATH"

    # Start reasoning server if needed
    if ! check_reasoning_server; then
        log "Starting reasoning server..."
        start_reasoning_server
    else
        log "Reasoning server already running"
    fi

    cd /home/syx/robo2VLM

    python merge_and_reason.py \
        --local-vqa-path "$AGIBOT_VQA_OUTPUT" \
        --hf-source-repo "$HF_SOURCE_REPO" \
        --manipulation-samples "$MANIPULATION_SAMPLES" \
        --agibot-samples "$AGIBOT_SAMPLES" \
        --output "$MERGED_DATASET_PATH" \
        --server-url "$REASONING_SERVER_URL" \
        --model "$REASONING_MODEL" \
        --num-workers "$REASONING_WORKERS"

    log "Pipeline complete! Output: $MERGED_DATASET_PATH"
}

step_sample_merge_only() {
    log "=========================================="
    log "Sample and Merge (no reasoning)"
    log "=========================================="

    cd /home/syx/robo2VLM

    python merge_and_reason.py \
        --local-vqa-path "$AGIBOT_VQA_OUTPUT" \
        --hf-source-repo "$HF_SOURCE_REPO" \
        --manipulation-samples "$MANIPULATION_SAMPLES" \
        --agibot-samples "$AGIBOT_SAMPLES" \
        --output "$MERGED_DATASET_PATH" \
        --skip-reasoning

    log "Complete! Output: $MERGED_DATASET_PATH"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --generate-only    Only generate VQA data"
    echo "  --merge-only       Sample, reason, merge (skip VQA generation)"
    echo "  --skip-reasoning   Sample and merge without reasoning"
    echo "  --help             Show this help"
    echo ""
    echo "Default: generate -> sample -> reasoning -> merge"
}

#######################
# MAIN
#######################

main() {
    log "=========================================="
    log "VQA Dataset Pipeline"
    log "=========================================="

    # Activate conda
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"

    case "${1:-full}" in
        --generate-only)
            step_generate_vqa
            ;;
        --merge-only)
            step_sample_reason_merge
            ;;
        --skip-reasoning)
            step_generate_vqa
            step_sample_merge_only
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        full|"")
            step_generate_vqa
            step_sample_reason_merge
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac

    log "=========================================="
    log "Pipeline completed!"
    log "=========================================="
}

main "$@"
