#!/bin/bash
#
# Launch SGLang Server for VLM Benchmarking
#
# Usage:
#   ./launch_sglang_server.sh                          # Default: Qwen2.5-VL-7B with 4 GPUs
#   ./launch_sglang_server.sh --model Qwen/Qwen2.5-VL-32B-Instruct --tp 8
#   ./launch_sglang_server.sh --list-models            # Show common models
#

set -e

#######################
# CONFIGURATION
#######################

DEFAULT_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PORT=30000
DEFAULT_HOST="0.0.0.0"
DEFAULT_TP=4  # tensor parallelism

# Common models for benchmarking
COMMON_MODELS=(
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen2.5-VL-72B-Instruct"
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "llava-hf/llava-v1.6-34b-hf"
    "llava-hf/llava-next-72b-hf"
    "microsoft/Phi-4-multimodal-instruct"
)

#######################
# PARSE ARGUMENTS
#######################

MODEL="$DEFAULT_MODEL"
PORT="$DEFAULT_PORT"
HOST="$DEFAULT_HOST"
TP="$DEFAULT_TP"
EXTRA_ARGS=""

show_usage() {
    echo "Launch SGLang Server for VLM Benchmarking"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL      Model to serve (default: $DEFAULT_MODEL)"
    echo "  --port PORT        Server port (default: $DEFAULT_PORT)"
    echo "  --host HOST        Server host (default: $DEFAULT_HOST)"
    echo "  --tp NUM           Tensor parallelism / number of GPUs (default: $DEFAULT_TP)"
    echo "  --mem-fraction F   GPU memory fraction (default: 0.9)"
    echo "  --max-model-len N  Maximum model length (default: auto)"
    echo "  --list-models      List common models for benchmarking"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                              # Start with defaults"
    echo "  $0 --model Qwen/Qwen2.5-VL-32B-Instruct --tp 8  # 32B model on 8 GPUs"
    echo "  $0 --port 30001                                 # Different port"
}

list_models() {
    echo "Common VLM models for benchmarking:"
    echo ""
    for model in "${COMMON_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Recommended tensor parallelism (--tp):"
    echo "  7B models:  1-2 GPUs"
    echo "  11B models: 2-4 GPUs"
    echo "  32B models: 4-8 GPUs"
    echo "  72B models: 8 GPUs"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --mem-fraction)
            EXTRA_ARGS="$EXTRA_ARGS --mem-fraction-static $2"
            shift 2
            ;;
        --max-model-len)
            EXTRA_ARGS="$EXTRA_ARGS --max-model-len $2"
            shift 2
            ;;
        --list-models)
            list_models
            exit 0
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

#######################
# LAUNCH SERVER
#######################

echo "=============================================="
echo "Launching SGLang Server"
echo "=============================================="
echo "Model:    $MODEL"
echo "Host:     $HOST"
echo "Port:     $PORT"
echo "TP:       $TP GPUs"
echo "Extra:    $EXTRA_ARGS"
echo "=============================================="
echo ""
echo "Server will be available at: http://$HOST:$PORT/v1"
echo "Press Ctrl+C to stop the server"
echo ""

# Launch the server
python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP" \
    $EXTRA_ARGS
