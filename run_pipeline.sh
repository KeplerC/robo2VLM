#!/bin/bash
#
# VQA Dataset Pipeline: Generation -> Merge -> Upload
#
# Usage:
#   ./run_pipeline.sh                    # Run full pipeline
#   ./run_pipeline.sh --generate-only    # Only generate VQA data
#   ./run_pipeline.sh --merge-only       # Only merge and upload (skip generation)
#   ./run_pipeline.sh --push-only        # Only push existing local dataset
#
# Configuration can be modified in the CONFIGURATION section below
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
NUM_WORKERS=256
STATE_SAMPLES=1
SEGMENT_SAMPLES=5
TOTAL_VQAS=50000

# HuggingFace settings
HF_SOURCE_REPO="keplerccc/ManipulationVQA-60k"
HF_TARGET_REPO="keplerccc/robo2vlm-2"

# Local merged dataset path
MERGED_DATASET_PATH="/home/syx/robo2VLM/merged_dataset"

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

check_hf_login() {
    log "Checking HuggingFace authentication..."
    if ! hf auth whoami &>/dev/null; then
        error "Not logged in to HuggingFace. Run: hf auth login"
    fi
    log "HuggingFace authentication OK"
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
    log "Num workers: $NUM_WORKERS"
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
    log "Output saved to: $AGIBOT_VQA_OUTPUT"
}

step_merge_and_push() {
    log "=========================================="
    log "STEP 2: Merging datasets and pushing to HuggingFace"
    log "=========================================="

    check_hf_login

    log "Merging:"
    log "  - HuggingFace source: $HF_SOURCE_REPO"
    log "  - Local Agibot VQA: $AGIBOT_VQA_OUTPUT"
    log "Target repo: $HF_TARGET_REPO"

    cd /home/syx/robo2VLM

    python merge_datasets.py

    log "Merge and push complete!"
    log "Dataset available at: https://huggingface.co/datasets/$HF_TARGET_REPO"
}

step_push_only() {
    log "=========================================="
    log "Pushing existing local dataset to HuggingFace"
    log "=========================================="

    check_hf_login

    if [ ! -d "$MERGED_DATASET_PATH" ]; then
        error "Local dataset not found at: $MERGED_DATASET_PATH"
    fi

    log "Local dataset: $MERGED_DATASET_PATH"
    log "Target repo: $HF_TARGET_REPO"

    cd /home/syx/robo2VLM

    python merge_datasets.py --push-only

    log "Push complete!"
    log "Dataset available at: https://huggingface.co/datasets/$HF_TARGET_REPO"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --generate-only    Only generate VQA data (skip merge/upload)"
    echo "  --merge-only       Only merge and upload (skip generation)"
    echo "  --push-only        Only push existing local dataset"
    echo "  --help             Show this help message"
    echo ""
    echo "Without options, runs the full pipeline: generate -> merge -> upload"
}

#######################
# MAIN
#######################

main() {
    log "=========================================="
    log "VQA Dataset Pipeline"
    log "=========================================="

    # Activate conda environment
    log "Activating conda environment: $CONDA_ENV"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"

    case "${1:-full}" in
        --generate-only)
            step_generate_vqa
            ;;
        --merge-only)
            step_merge_and_push
            ;;
        --push-only)
            step_push_only
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        full|"")
            step_generate_vqa
            step_merge_and_push
            ;;
        *)
            error "Unknown option: $1. Use --help for usage."
            ;;
    esac

    log "=========================================="
    log "Pipeline completed successfully!"
    log "=========================================="
}

main "$@"
