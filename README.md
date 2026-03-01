# med_dict

Structured extraction from medical oncology notes using Llama 3.1 8B Instruct.

## Quick Start

```bash
# Run an experiment (V2 pipeline, default)
python run.py exp/default.yaml

# Run V1 pipeline for comparison
python run.py exp/default_v1.yaml

# Run 2-row test
python run.py exp/test_2row.yaml

# Resume an interrupted run
python run.py exp/default.yaml --resume results/default_20260228_103000/
```

## Project Structure

```
med_dict/
├── run.py                  # Experiment runner (CLI entry point)
├── ult.py                  # Utility library (model inference, KV cache, JSON repair, pipelines)
├── prompts/
│   ├── extraction.yaml     # Extraction prompts (4 keys: visit reason, findings, treatment, goals)
│   └── plan_extraction.yaml # Plan extraction prompts (12 keys: meds, procedures, imaging, etc.)
├── exp/
│   ├── default.yaml        # Default config (v2 pipeline)
│   ├── default_v1.yaml     # V1 pipeline config (for comparison)
│   ├── test_2row.yaml      # Quick test config (2 rows)
│   └── test_mac.yaml       # Mac MPS config (float16)
├── results/                # Auto-generated run outputs
│   └── <name>_<timestamp>/
│       ├── config.yaml     # Config snapshot
│       ├── results.txt     # Extraction results
│       ├── progress.json   # Checkpoint for resume
│       └── run.log         # Terminal output + per-gate logs
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

data:
  dataset_path: "data/CORAL/.../breastca_unannotated.csv"
  row_range: [0, 20]

extraction:
  pipeline: "v2"       # "v1" or "v2"
  verify: true          # enable faithfulness/temporal/specificity gates
```

## Pipeline

Two pipeline versions, selected via `extraction.pipeline` in config.

### V2 (default) — 5-Gate Agentic Extraction

Each gate fixes one specific issue by trimming/correcting, never re-extracting from scratch:

1. **FORMAT** — Parse JSON; if invalid, LLM reformats to match schema
2. **SCHEMA** — Validate output keys against expected schema from prompt; LLM fixes mismatched keys
3. **FAITHFULNESS** — Review each value against the original note. Only empty values that clearly contradict or are fabricated. "When in doubt, keep." Dropped keys are auto-restored from the original extraction.
4. **TEMPORAL** — Plan keys only (`Therapy_plan`, `Procedure_Plan`, `Imaging_Plan`, `Lab_Plan`, `Medication_Plan`, `Medication_Plan_chatgpt`): remove past/completed items, keep current/future
5. **SPECIFICITY** — Conditional trigger: only runs if vague terms detected ("staging workup", "as above", etc.). Replaces with specific details from the note.

Key design decisions:
- Gate 3 uses "keep unless clearly wrong" policy (not "keep only if explicitly stated"), avoiding over-trimming of reasonable clinical summaries
- Gates 3-5 all validate key overlap before accepting changes, preventing schema leakage (e.g. `{"faithful": true}`)
- Per-gate logging in `run.log` shows before/after for every field change

### V1 — 3-Gate Pipeline (legacy)

1. **FORMAT** — Parse JSON; LLM repair if needed
2. **FAITHFULNESS** — Full re-extraction if unfaithful (may introduce new errors)
3. **TEMPORAL** — Remove past items from plan keys

V1 issues: re-extraction can introduce new hallucinations; no schema validation; no specificity check.

### V1 vs V2 Comparison (15-sample test)

| Metric | V1 | V2 |
|--------|----|----|
| Speed | ~295s/5rows | ~296s/5rows (comparable) |
| re-extract calls | ~25/batch | 0 |
| Schema key preservation | OK | OK (with auto-restore) |
| Content richness | Baseline | +7% average |
| "No X planned" false negatives | Common | Reduced |
| response_assessment | Has content (some speculative) | Has content (after Gate 3 fix) |

## Per-Gate Logging

V2 logs detailed gate activity to `run.log`:

```
Reason_for_Visit: 3.0s [faith-trimmed]
  [EXTRACT] raw={"Patient type": "follow up", ...
  [G1-FORMAT] ok, keys=['Patient type', 'second opinion', ...]
  [G2-SCHEMA] ok
  [G3-FAITH] summary: "Follow up for ER+/PR+ IDC..." -> "Follow up for IDC..."
  [G3-FAITH] EMPTIED: []
```

Each gate reports: `ok`, field-level changes (`before -> after`), `EMPTIED: [fields]`, `FAILED`, or `REJECTED`.

## Features

- **Checkpoint/resume**: Saves `progress.json` after each row. Auto-resumes matching incomplete runs.
- **Logging**: All terminal output + per-gate details saved to `results/<run>/run.log`
- **4-bit/8-bit quantization**: Via `model.quantization` in config
- **KV cache reuse**: Note encoded once, shared across extraction tasks

## Setup

```bash
# HuggingFace token (required for Llama model access)
echo 'hf_YOUR_TOKEN' > hf.token

# Install dependencies
pip install torch transformers accelerate pyyaml pandas huggingface_hub nltk
```
