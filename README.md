# med_dict — Oncology Note Structured Extraction

Extract structured clinical information from breast cancer oncology notes using a 32B LLM (Qwen2.5-32B-Instruct-AWQ), with rule-based post-processing for quality assurance.

## For Reviewers

**Start here**: [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md) — explains input/output, architecture, quality data, and examples without requiring code knowledge.

**Quality report**: [`results/v23_audit_report.md`](results/v23_audit_report.md) — full audit of 61 samples with per-sample findings.

## Core Principles

1. **Faithful** — Never hallucinate. Better to say less than to say something wrong.
2. **Complete** — Cover all important clinical information.
3. **Simple** — Eighth-grade reading level for patient-facing output.
4. **Clear** — Avoid jargon; explain when unavoidable.

## Current Version: v23

- **Model**: Qwen/Qwen2.5-32B-Instruct-AWQ (4-bit quantized, single GPU)
- **Dataset**: 61 breast cancer notes from CORAL (UCSF, de-identified)
- **Quality**: P0=0, P1=2 (3.3%), P2=28, OK=31 (51%)
- **POST hooks**: 22 rule-based corrections for known LLM failure patterns

## Quick Start

```bash
# Run extraction on a config
python run.py exp/v23_remaining.yaml

# Resume an interrupted run
python run.py exp/v23_remaining.yaml --resume results/v23_remaining_<timestamp>/
```

## Project Structure

```
med_dict/
├── run.py                  # Pipeline entry point (2154 lines)
├── ult.py                  # Utilities: model inference, KV cache, gates
├── PIPELINE_OVERVIEW.md    # Architecture doc for reviewers
├── prompts/
│   ├── extraction.yaml     # Phase 1+2 extraction prompts (8 fields)
│   └── plan_extraction.yaml # Plan extraction prompts (7 fields)
├── exp/                    # Experiment configs (YAML)
├── results/                # Run outputs + audit reports
│   ├── v23_audit_report.md # Latest quality audit
│   ├── v23_*_results.txt   # Raw extraction outputs
│   └── error_notebook.md   # Cumulative error tracking
└── data/                   # CORAL dataset (not in repo)
```

## Pipeline Architecture

```
Clinical Note → A/P Extraction → Phase 1 (6 prompts) → Phase 2 (2 prompts)
                                → Plan Extraction (8 prompts from A/P)
             → 5-Gate Verification (format, schema, improve, faithfulness, temporal)
             → 22 POST Hooks (rule-based corrections)
             → Source Attribution → Structured JSON Output
```

See [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md) for the full architecture diagram.
