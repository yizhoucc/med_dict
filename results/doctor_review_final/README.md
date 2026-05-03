# Doctor Review — Baseline vs Pipeline Comparison

**Date:** 2026-05-03
**Models:** Qwen2.5-32B-Instruct-AWQ (Pipeline + Qwen Baseline) and GPT-4o (ChatGPT Baseline)

## Conditions

| # | Folder | Cancer Type | Model | Condition | Description |
|---|--------|------------|-------|-----------|-------------|
| 1 | `1_breast_pipeline/` | Breast Cancer | Qwen2.5-32B | **Pipeline** | Full harness: 5-gate verification + 40+ POST hooks + RAG |
| 2 | `2_breast_baseline/` | Breast Cancer | Qwen2.5-32B | **Qwen Baseline** | Same model, single prompt, no processing |
| 2b | `2_breast_chatgptbaseline/` | Breast Cancer | GPT-4o | **ChatGPT Baseline** | Proprietary model, single prompt, no processing |
| 3 | `3_pdac_pipeline/` | Pancreatic Cancer | Qwen2.5-32B | **Pipeline** | Full harness adapted for PDAC |
| 4 | `4_pdac_baseline/` | Pancreatic Cancer | Qwen2.5-32B | **Qwen Baseline** | Same model, single prompt, no processing |

## Instructions

1. Each folder contains 20 samples with original clinical note + patient letter
2. Pipeline folders also include structured extraction (JSON)
3. For each letter, rate 1-5 on: Accuracy, Completeness, Safety, Simplification, Overall Quality
4. Also mark Yes/No: Fabricated info? Missing critical info? Harmful content?
5. Reviewers are blinded to condition — do not look at folder names until after rating

## Samples

- 20 annotated breast cancer notes (CORAL dataset, held-out test set)
- 20 annotated pancreatic cancer notes (CORAL dataset, held-out test set)
- Total: 100 letters to review (20 breast × 3 conditions + 20 PDAC × 2 conditions)
