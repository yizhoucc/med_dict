# Adjusted clinician-preference analysis

Updated: 2026-10-05

## Primary model

The primary analysis excludes ties and models whether a directional rating favors the inference harness. It uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. This approach accounts for repeated ratings of fields and evaluators within a note without treating all directional ratings as independent.

| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted harness probability | Odds ratio | 95% CI | p value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 737 | 596 | 141 | 40 | 80.6% | 4.15 | 3.25-5.29 | 2.56e-30 |
| Breast cancer | 443 | 363 | 80 | 20 | 81.7% | 4.45 | 3.17-6.25 | 7.04e-18 |
| PDAC | 294 | 233 | 61 | 20 | 78.8% | 3.71 | 2.64-5.23 | 5.97e-14 |

The odds ratio compares the adjusted probability of a harness preference with the probability of a baseline preference among non-tie judgments. It is not an odds ratio for clinical correctness.

## Note-level sensitivity analysis

For each evaluator-cancer combination, field ratings were reduced to one harness-minus-baseline margin per note. Tied note margins were excluded from the exact two-sided sign test.

| Evaluation | Positive notes | Negative notes | Tied notes | Exact p value |
|---|---:|---:|---:|---:|
| Oncologist 01, Breast cancer | 18 | 1 | 1 | 7.63e-05 |
| Oncologist 01, PDAC | 19 | 1 | 0 | 4.01e-05 |
| Oncologist 02, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| Oncologist 02, PDAC | 18 | 0 | 2 | 7.63e-06 |
| Oncologist 03, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| Oncologist 04, Breast cancer | 16 | 2 | 2 | 0.00131 |
| Oncologist 04, PDAC | 13 | 4 | 3 | 0.049 |
| All clinicians pooled within note | 39 | 0 | 1 | 3.64e-12 |

## Interpretation boundary

The observed-note result is statistically strong, but four oncologists remain too few for a precise estimate of variation across the broader oncologist population. Evaluator-level generalization should therefore remain a pilot claim.
