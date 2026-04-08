#!/bin/bash
# V26 full comparison: notool vs tool
# Run sequentially (single GPU)

set -e
cd /home/yc/repo/med_dict
source /home/yc/miniconda3/etc/profile.d/conda.sh
conda activate medllm

echo "=== Starting v26_full_notool at $(date) ==="
python run.py exp/v26_full_notool.yaml
echo "=== Finished v26_full_notool at $(date) ==="

echo ""
echo "=== Starting v26_full_tool at $(date) ==="
python run.py exp/v26_full_tool.yaml
echo "=== Finished v26_full_tool at $(date) ==="

echo ""
echo "=== Both runs complete at $(date) ==="
