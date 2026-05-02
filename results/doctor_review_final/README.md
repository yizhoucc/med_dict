# Doctor Review — Baseline vs Pipeline Comparison

**Date:** 2026-05-02
**Model:** Qwen2.5-32B-Instruct-AWQ (same model for both conditions)

## Conditions

| # | Folder | Cancer Type | Condition | Description |
|---|--------|------------|-----------|-------------|
| 1 | `1_breast_pipeline/` | Breast Cancer | **Pipeline** | Full harness: 5-gate verification + 40+ POST hooks + RAG |
| 2 | `2_breast_baseline/` | Breast Cancer | **Baseline** | Same model, single prompt, no processing |
| 3 | `3_pdac_pipeline/` | Pancreatic Cancer | **Pipeline** | Full harness adapted for PDAC |
| 4 | `4_pdac_baseline/` | Pancreatic Cancer | **Baseline** | Same model, single prompt, no processing |

## Instructions

1. Each folder contains 20 samples with original clinical note + patient letter
2. Pipeline folders also include structured extraction (JSON)
3. For each letter, rate 1-5 on: Accuracy, Completeness, Safety, Simplification, Overall Quality
4. Also mark Yes/No: Fabricated info? Missing critical info? Harmful content?
5. Reviewers are blinded to condition — do not look at folder names until after rating

## Samples

- 20 annotated breast cancer notes (CORAL dataset, held-out test set)
- 20 annotated pancreatic cancer notes (CORAL dataset, held-out test set)
- Total: 80 letters to review (20 × 2 cancers × 2 conditions)
