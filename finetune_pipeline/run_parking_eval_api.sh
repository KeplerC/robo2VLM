#!/bin/bash
# Evaluate API models (OpenAI GPT, Google Gemini) on the parking benchmark
# Usage: bash run_parking_eval_api.sh
#
# Required environment variables:
#   OPENAI_API_KEY  - For GPT models
#   GOOGLE_API_KEY  - For Gemini models
#
# You can set these in your shell or create a .env file and source it:
#   export OPENAI_API_KEY="your-openai-api-key"
#   export GOOGLE_API_KEY="your-google-api-key"

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate unsloth_env 2>/dev/null || true

RESULTS_DIR="results_parking"
SCRIPT_DIR="$(dirname "$0")"

mkdir -p $RESULTS_DIR

echo "====================================="
echo "Parking Benchmark - API Models"
echo "====================================="

# Check for API keys
OPENAI_AVAILABLE=false
GEMINI_AVAILABLE=false

if [ -n "$OPENAI_API_KEY" ]; then
    echo "OpenAI API key found"
    OPENAI_AVAILABLE=true
else
    echo "Warning: OPENAI_API_KEY not set - skipping OpenAI models"
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    echo "Google API key found"
    GEMINI_AVAILABLE=true
else
    echo "Warning: GOOGLE_API_KEY not set - skipping Gemini models"
fi

echo ""

# Define models to evaluate
# OpenAI models (GPT-5 family + GPT-4 for comparison)
OPENAI_MODELS=()

# Gemini models
GEMINI_MODELS=("gemini-2.5-flash")

# Evaluation modes
MODES=("zero_shot" "cot")

# Function to run a single evaluation
run_eval() {
    local model=$1
    local mode=$2
    local output_file="${RESULTS_DIR}/${model}_${mode}.json"

    # Skip if already exists
    if [ -f "$output_file" ]; then
        echo "[SKIP] $output_file already exists"
        return 0
    fi

    echo "[RUN] Model: $model, Mode: $mode"
    echo "      Output: $output_file"

    python "${SCRIPT_DIR}/eval_parking_api.py" \
        --model "$model" \
        --output-file "$output_file" \
        --mode "$mode"

    if [ $? -eq 0 ]; then
        accuracy=$(python -c "import json; d=json.load(open('$output_file')); print(f\"{d['accuracy']*100:.2f}%\")")
        echo "[DONE] $model ($mode): $accuracy"
    else
        echo "[ERROR] Failed: $model ($mode)"
    fi
    echo ""
}

# Run OpenAI models
if [ "$OPENAI_AVAILABLE" = true ]; then
    echo "====================================="
    echo "Evaluating OpenAI Models"
    echo "====================================="
    for model in "${OPENAI_MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            run_eval "$model" "$mode"
        done
    done
fi

# Run Gemini models
if [ "$GEMINI_AVAILABLE" = true ]; then
    echo "====================================="
    echo "Evaluating Gemini Models"
    echo "====================================="
    for model in "${GEMINI_MODELS[@]}"; do
        for mode in "${MODES[@]}"; do
            run_eval "$model" "$mode"
        done
    done
fi

echo ""
echo "====================================="
echo "All API evaluations complete!"
echo "Results saved to: ${RESULTS_DIR}/"
echo "====================================="

# Print summary of all results
echo ""
echo "Summary of API model results:"
echo "-----------------------------"
for f in ${RESULTS_DIR}/gpt-5*.json ${RESULTS_DIR}/gpt-4*.json ${RESULTS_DIR}/gemini*.json ${RESULTS_DIR}/o1*.json ${RESULTS_DIR}/o3*.json; do
    if [ -f "$f" ]; then
        accuracy=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['accuracy']*100:.2f}%\")" 2>/dev/null || echo "N/A")
        echo "  $(basename $f): $accuracy"
    fi
done
