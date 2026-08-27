#!/bin/bash
# Start a vLLM server. The reported extraction experiments use Qwen2.5.
#
# Usage:
#   bash vllm_pipeline/start_vllm.sh              # Qwen2.5 (reported model)
#   bash vllm_pipeline/start_vllm.sh qwen3.5      # optional exploratory model
#   bash vllm_pipeline/start_vllm.sh org/model-id # explicit model
#
# The server will listen on port 8000.
# Wait until "Uvicorn running on http://0.0.0.0:8000" appears before starting the pipeline.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medllm

PROFILE="${1:-qwen2.5}"
EXTRA_ARGS=()
case "$PROFILE" in
  qwen2.5)
    MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
    ;;
  qwen3.5)
    MODEL_NAME="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    EXTRA_ARGS+=(--reasoning-parser qwen3)
    ;;
  *)
    MODEL_NAME="$PROFILE"
    ;;
esac

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Port: 8000"
echo ""

vllm serve "$MODEL_NAME" \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --dtype float16 \
  "${EXTRA_ARGS[@]}"
