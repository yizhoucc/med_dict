#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medllm
cd ~/repo/med_dict
git pull
nohup python run.py exp/v15a_verify.yaml > v15a_run.log 2>&1 &
echo "v15a pipeline started at $(date), PID=$!" >> ~/repo/med_dict/v15a_cron.log
