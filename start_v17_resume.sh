#!/bin/bash
# v17 pipeline resume - scheduled for 2026-03-19 08:00
eval "$(/home/yc/miniconda3/bin/conda shell.bash hook)"
conda activate medllm
cd ~/repo/med_dict
python run.py --resume results/v17_verify_20260318_184026/progress.json exp/v17_verify.yaml >> results/v17_verify_resume.log 2>&1
