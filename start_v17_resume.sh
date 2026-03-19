#!/bin/bash
# v17 pipeline full rerun - scheduled for 2026-03-19 08:00
# POST-VISIT-TYPE logic changed, need clean rerun
eval "$(/home/yc/miniconda3/bin/conda shell.bash hook)"
conda activate medllm
cd ~/repo/med_dict
rm -rf results/v17_verify_20260318_184026/
python run.py exp/v17_verify.yaml >> results/v17_verify_rerun.log 2>&1
