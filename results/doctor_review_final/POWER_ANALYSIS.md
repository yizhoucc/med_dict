# Statistical Design — Sample Size and Power

## Study Design

- **Primary contrast**: Pipeline (Qwen-DA) vs Qwen Baseline (superiority)
- **Primary endpoint**: Accuracy (single endpoint, no multiple testing correction)
- **Descriptive comparison**: ChatGPT (GPT-4o) included for context; no formal hypothesis test
- **Raters**: 6 clinicians, independent ratings
- **Rating scale**: 5-point Likert (Accuracy, Completeness, Comprehensibility, Usefulness)
- **Statistical test**: Paired Wilcoxon signed-rank on scenario-level rater-mean differences
- **Alpha**: 0.05, two-sided

## Effect Size Estimation

An automated evaluation using an LLM judge (Claude) scored 20 breast cancer samples on the Accuracy dimension. The observed distributions were:

| Score | Pipeline | Baseline |
|-------|----------|----------|
| 3     | 0        | 2        |
| 4     | 1        | 12       |
| 5     | 19       | 6        |
| **Mean** | **4.95** | **4.20** |

Paired comparison: Pipeline wins 14, ties 5, baseline wins 1.
Mann-Whitney proportional-odds **OR ≈ 4.8** (Wilcoxon p = 0.001).

Human raters typically show smaller effect sizes than automated judges due to greater variability and subjective interpretation. We conservatively assume **OR = 2.0** for human raters, approximately 2.4× smaller than the LLM judge estimate.

## Power Results

Monte Carlo simulation (10,000 iterations per cell, paired Wilcoxon signed-rank, single primary endpoint, α = 0.05 two-sided).

Baseline distribution assumed: P(score 3) = 0.10, P(score 4) = 0.60, P(score 5) = 0.30.

### n = 20 scenarios (single cancer type)

| OR  | Power |
|-----|-------|
| 1.5 | 0.30  |
| 2.0 | 0.69  |
| 2.3 | 0.83 ✓ |
| 2.5 | 0.89 ✓ |
| 3.0 | 0.97 ✓ |

**20 scenarios reach 80% power at OR ≥ 2.3.**

### n = 40 scenarios (both cancer types pooled)

| OR  | Power |
|-----|-------|
| 1.5 | 0.55  |
| 2.0 | 0.94 ✓ |
| 2.5 | 1.00 ✓ |

**40 scenarios reach 80% power at OR ≥ 2.0.**

## Analysis Plan

### Available Data

- 20 breast cancer scenarios (physician-guided iterative optimization, 15 rounds)
- 20 pancreatic cancer scenarios (model-only optimization, no physician feedback)
- Total: 40 clinical scenarios × 3 systems × 6 raters

### Primary Analysis (Breast Cancer, n = 20)

The breast cancer pipeline was developed through 15 rounds of iterative optimization with physician feedback. This is the primary analysis.

- **Hypothesis**: Pipeline produces more accurate patient letters than the same base model without the inference harness (superiority, OR > 1)
- **Required effect**: OR ≥ 2.3 for 80% power at n = 20
- **Estimated effect**: OR ≈ 2.0–3.0 (conservative range based on LLM judge OR = 4.8)

### Stretch Goal (PDAC, n = 20)

The pancreatic cancer pipeline was adapted from the breast cancer pipeline without physician involvement — only model-driven optimization. If the PDAC subset independently achieves statistical significance (OR ≥ 2.3), this demonstrates that the harness generalizes to new cancer types without requiring additional physician iteration.

### Fallback (Pooled, n = 40)

If neither cancer type alone reaches significance, pooling both subsets (n = 40) achieves 94% power at OR = 2.0. This is the most conservative analysis and requires the smallest effect size to detect.

## ChatGPT Comparison (Descriptive Only)

ChatGPT (GPT-4o) letters are included in the evaluation package for descriptive context. Dimension-level means, distributions, and qualitative observations will be reported, but **no formal superiority or non-inferiority hypothesis is tested** against ChatGPT.

Rationale: The research question is whether an inference harness improves a base open-source model (Pipeline vs Qwen Baseline, same model). The ChatGPT comparison addresses an anticipated reader question ("how does it compare to a commercial model?") but is not the scientific claim of the paper. Removing ChatGPT from formal testing eliminates a secondary contrast that would otherwise inflate the required sample size.

## Code

Power analysis code: [`power_analysis.py`](power_analysis.py)
