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
│   ├── default.yaml        # Default config (5 rows, bfloat16, v2 pipeline)
│   ├── default_v1.yaml     # V1 pipeline config (for comparison)
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

Two pipeline versions are available, selected via `extraction.pipeline` in config:

### V2 (default) — 5-Gate Agentic Extraction

Each gate fixes one specific issue by trimming/correcting, not re-extracting:

1. **FORMAT** — Parse JSON; if invalid, LLM reformats to match schema
2. **SCHEMA** — Validate output keys against expected schema; LLM fixes mismatched keys
3. **FAITHFULNESS** — Trim unsupported claims (removes only unfaithful parts, keeps the rest)
4. **TEMPORAL** — Plan keys only: remove past/completed items, keep current/future
5. **SPECIFICITY** — Conditional: replace vague terms ("staging workup", "as above") with specific details from the note

### V1 — 3-Gate Pipeline

1. **FORMAT** — Parse JSON; LLM repair if needed
2. **FAITHFULNESS** — Full re-extraction if unfaithful (may introduce new errors)
3. **TEMPORAL** — Remove past items from plan keys

### Switching Pipelines

```bash
# V2 (default)
python run.py exp/default.yaml

# V1 (for comparison)
python run.py exp/default_v1.yaml
```

Or set `extraction.pipeline` to `"v1"` or `"v2"` in any config YAML.

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
