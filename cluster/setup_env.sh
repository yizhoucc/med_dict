#!/bin/bash
# One-time conda env setup for CMU Mind cluster (run on LOGIN node — pip/conda only, no GPU compute).
# Creates env `medllm` with vLLM + pipeline deps. Re-run is idempotent-ish.
set -e
module load anaconda3-2023.03 cuda-12.4

# create env if missing
conda env list | grep -q "^medllm " || conda create -y -n medllm python=3.11
source activate medllm

pip install --upgrade pip
# vLLM (pins its own torch+cuda runtime; transformers pulled in as dep). Adjust version if incompat.
pip install "vllm==0.6.6"
pip install pandas pyyaml requests openai textstat

python -c "import vllm, pandas, yaml, textstat, openai; print('env OK, vllm', vllm.__version__)"
echo "=== medllm env ready ==="
