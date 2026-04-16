#!/bin/bash
# Start vLLM server for Qwen3.5-35B-A3B-GPTQ-Int4
#
# Usage:
#   bash vllm_pipeline/start_vllm.sh
#
# The server will listen on port 8000.
# Wait until "Uvicorn running on http://0.0.0.0:8000" appears before starting the pipeline.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medllm

echo "Starting vLLM server..."
echo "Model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
echo "Port: 8000"
echo ""

vllm serve Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --port 8000 \
  --enable-prefix-caching \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --dtype float16 \
  --reasoning-parser qwen3
