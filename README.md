# med_dict — Oncology Patient Letter Generation via Inference Harness

Generate patient-friendly summary letters from oncology clinical notes using an open-source LLM (Qwen2.5-32B-Instruct-AWQ via vLLM), enhanced by a structured inference harness — prompt engineering, multi-gate verification, and rule-based safety hooks — without modifying model weights.

## Core Principles

1. **Faithful** — Never hallucinate. Better to say less than to say something wrong.
2. **Complete** — Cover all clinically important information.
3. **Simple** — Eighth-grade reading level for patient-facing output.
4. **Clear** — Avoid jargon; explain medical terms when unavoidable.

## For Reviewers

- **Architecture overview**: [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md)
- **Research proposal**: [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md)
- **Evaluation rubric**: [`results/doctor_review_final/EVALUATION_RUBRIC.md`](results/doctor_review_final/EVALUATION_RUBRIC.md)
- **Power analysis**: [`results/doctor_review_final/POWER_ANALYSIS.md`](results/doctor_review_final/POWER_ANALYSIS.md)

## Current Status

| | Breast Cancer (v31) | PDAC (v32) |
|---|---|---|
| **Dev set** | 100 unannotated, 15 iterations → 100% clean | 100 unannotated, 18 iterations → 99% clean |
| **Test set** | 20 annotated held-out → P0=0, P1=0 | 20 annotated held-out → generated |
| **Optimization** | Physician-guided (15 rounds) | Model-only (no physician feedback) |

### 3-Way Baseline Comparison (Pipeline vs Qwen Baseline vs ChatGPT)

| Metric | Pipeline (Qwen + Harness) | Qwen Baseline (bare) | ChatGPT (GPT-4o) |
|---|---|---|---|
| Deployable letters | 97.5% | 0% | 0% |
| Hallucination | 0% | 12.5% | has speculation |
| REDACTED leakage | 0/40 | 45% | 2/20 |
| HIPAA compliant | Yes (local) | Yes (local) | No (cloud API) |
| Reading level (FK) | — | 6.6 | 8.8 |

## Pipeline Architecture

```
Clinical Note
  → A/P Extraction (regex + LLM fallback)
  → Phase 1: Key Point Extraction (6 independent prompts)
  → Phase 2: Dependent Reasoning (2 prompts, injected Phase 1 context)
  → Plan Extraction (from A/P section)
  → 5-Gate Verification Cascade
      1. FORMAT — JSON parse repair
      2. SCHEMA — key name validation
      3. IMPROVE — specificity + semantic alignment
      4. FAITHFUL — hallucination pruning ("keep if supported")
      5. TEMPORAL — filter past/completed items from plans
  → Rule-Based POST Hooks (40+ corrections)
  → Source Attribution
  → Patient Letter Generation
  → Letter POST Checks (terminology, dosage, voice, grammar)
```

## Quick Start

```bash
# Start vLLM server (on GPU machine)
bash vllm_pipeline/start_vllm.sh

# Run pipeline
python run.py exp/default_qwen.yaml

# Run with vLLM (remote inference)
python vllm_pipeline/run_vllm.py exp/v32_vllm.yaml

# Resume an interrupted run
python run.py exp/default_qwen.yaml --resume results/<run_dir>/
```

## Project Structure

```
med_dict/
├── run.py                     # Main pipeline + POST hooks (3500 lines)
├── ult.py                     # Model inference, 5-gate cascade (1600 lines)
├── letter_generation.py       # Letter generation + post-checks (550 lines)
├── source_attribution.py      # Source attribution to original note
├── auto_review.py             # LLM-based automated review
├── baseline_generate.py       # Baseline generation (bare model + single prompt)
├── prompts/
│   ├── extraction.yaml        # Breast cancer extraction (8 fields)
│   ├── plan_extraction.yaml   # Plan extraction (7 fields)
│   ├── letter_generation.yaml # Breast letter template
│   └── pdac/                  # PDAC-specific prompts (same structure)
├── vllm_pipeline/
│   ├── vllm_client.py         # vLLM HTTP API client
│   ├── run_vllm.py            # vLLM-based pipeline runner
│   └── start_vllm.sh          # vLLM server launch script
├── exp/                       # Experiment configs (YAML)
├── data/
│   ├── CORAL/                 # Clinical notes dataset (UCSF, de-identified)
│   ├── formaldef.txt           # Medical term dictionary (9,331 terms)
│   ├── oncology_drugs.txt      # Drug whitelist (~140 drugs)
│   └── supportive_care_drugs.txt
├── results/
│   └── doctor_review_final/   # Physician evaluation package
│       ├── breast_pipeline/   # 20 pipeline letters (breast)
│       ├── breast_baseline/   # 20 Qwen baseline letters
│       ├── breast_chatgptbaseline/ # 20 ChatGPT letters
│       ├── pdac_pipeline/     # 20 pipeline letters (PDAC)
│       ├── pdac_baseline/     # 20 Qwen baseline letters
│       ├── pdac_chatgptbaseline/   # 20 ChatGPT letters
│       ├── EVALUATION_RUBRIC.md
│       └── POWER_ANALYSIS.md
├── PIPELINE_OVERVIEW.md       # Architecture doc
└── RESEARCH_PROPOSAL.md       # Research proposal v3.0
```

## Model & Infrastructure

- **Model**: Qwen/Qwen2.5-32B-Instruct-AWQ (4-bit quantized)
- **Inference**: vLLM server (OpenAI-compatible API)
- **Hardware**: Single GPU (WSL, Ubuntu 22.04)
- **Data**: CORAL dataset — 200 unannotated (dev) + 40 annotated (test) oncology notes

## Evaluation Design

- **Primary contrast**: Pipeline vs Qwen Baseline (superiority, Accuracy dimension)
- **Descriptive comparison**: ChatGPT (GPT-4o) for context, no formal hypothesis test
- **Raters**: 6 oncology clinicians, blinded, independent ratings
- **Rubric**: 5-point Likert (Accuracy, Completeness, Comprehensibility, Usefulness) + binary hallucination + deployment readiness
- **Statistical test**: Paired Wilcoxon signed-rank, α=0.05
- **Power**: 40 scenarios at OR=2.0 → 94% power

## License

Dataset: [CORAL](https://physionet.org/content/curated-oncology-reports/1.0/) (PhysioNet credentialed access required).
