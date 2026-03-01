# med_dict

Structured extraction from medical oncology notes using Llama 3.1 8B Instruct.

## Quick Start

```bash
# Run an experiment
python run.py exp/default.yaml

# Run 2-row test
python run.py exp/test_2row.yaml

# Resume an interrupted run
python run.py exp/default.yaml --resume results/default_20260228_103000/
```

## Project Structure

```
med_dict/
├── run.py                  # Experiment runner (CLI entry point)
├── ult.py                  # Utility library (model inference, KV cache, JSON repair)
├── prompts/
│   ├── extraction.yaml     # Extraction prompts (4 keys: visit reason, findings, treatment, goals)
│   └── plan_extraction.yaml # Plan extraction prompts (12 keys: meds, procedures, imaging, etc.)
├── exp/
│   ├── default.yaml        # Default config (10 rows, bfloat16)
│   ├── test_2row.yaml      # Quick test config (2 rows)
│   └── test_mac.yaml       # Mac MPS config (float16)
├── results/                # Auto-generated run outputs
│   └── <name>_<timestamp>/
│       ├── config.yaml     # Config snapshot
│       ├── results.txt     # Extraction results
│       ├── progress.json   # Checkpoint for resume
│       └── run.log         # Terminal output log
├── results.txt             # Copy of latest run's results
├── notebooks/              # Legacy notebooks
└── data/                   # CORAL dataset (not in repo)
```

## Experiment Config

Configs live in `exp/`. Key sections:

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  dtype: "bfloat16"
  device_map: "auto"
  # quantization:          # Optional 4-bit quantization
  #   type: "4bit"
  #   quant_type: "nf4"
  #   double_quant: true

data:
  dataset_path: "data/CORAL/.../breastca_unannotated.csv"
  row_range: [0, 10]       # Process rows 0-9
```

## Pipeline

1. **Assessment/Plan extraction** - Copy-paste from note (3 retries + LLM verification)
2. **Keypoint extraction** - From full note: visit reason, findings, treatment, goals
3. **Plan extraction** - From A/P section: meds, procedures, imaging, labs, genetics, referrals, follow-up
4. **Faithfulness check** - Verify each extraction against source text

## Features

- **Checkpoint/resume**: Saves `progress.json` after each row. Auto-resumes matching incomplete runs.
- **Logging**: All terminal output saved to `results/<run>/run.log`
- **4-bit/8-bit quantization**: Via `model.quantization` in config
- **KV cache reuse**: Note encoded once, shared across extraction tasks

## Setup

```bash
# HuggingFace token (required for Llama model access)
echo 'hf_YOUR_TOKEN' > hf.token

# Install dependencies
pip install torch transformers accelerate pyyaml pandas huggingface_hub nltk
```
