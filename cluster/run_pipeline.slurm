#!/bin/bash
# Mode A: self-contained Slurm job — starts vLLM server on the allocated GPU node,
# waits until ready, runs the pipeline against localhost:8000, then tears down.
# Fully reproducible: one `sbatch cluster/run_pipeline.slurm [exp/config.yaml]`.
#
# Usage:  sbatch cluster/run_pipeline.slurm exp/test_fix_breast8.yaml
#
#SBATCH --job-name=meddict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=cluster/slurm_%j.out

set -euo pipefail
CONFIG="${1:-exp/test_fix_breast8.yaml}"
echo "=== job $SLURM_JOB_ID on $(hostname); config=$CONFIG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load anaconda3-2023.03 cuda-12.4
source activate medllm

export HF_HOME=/user_data/yizhouc3/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_LOGGING_LEVEL=WARNING
cd "$SLURM_SUBMIT_DIR"

# --- start vLLM server in background ---
echo "=== starting vLLM (Qwen2.5-32B-Instruct-AWQ) ==="
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --dtype float16 \
  --trust-remote-code > "cluster/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# --- wait until server is ready (model load + first-run download can take a while) ---
echo "=== waiting for vLLM readiness ==="
READY=0
for i in $(seq 1 120); do
  if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen; then
    READY=1; echo "vLLM ready after ${i} checks"; break
  fi
  sleep 15
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM did not become ready; see cluster/vllm_${SLURM_JOB_ID}.log"; tail -40 "cluster/vllm_${SLURM_JOB_ID}.log"; exit 1
fi

# --- run the extraction pipeline ---
echo "=== running pipeline: $CONFIG ==="
python -u vllm_pipeline/run_vllm.py "$CONFIG"

echo "=== done; results under results/ ==="
