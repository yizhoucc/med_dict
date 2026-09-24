# Adjusted clinician-preference analysis

Updated: 2026-09-23

## Primary model

The primary analysis excludes ties and models whether a directional rating favors the inference harness. It uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. This approach accounts for repeated ratings of fields and evaluators within a note without treating all 520 directional ratings as independent.

| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted harness probability | Odds ratio | 95% CI | p value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 520 | 443 | 77 | 40 | 85.1% | 5.70 | 4.24-7.66 | 6.79e-31 |
| Breast cancer | 336 | 282 | 54 | 20 | 83.8% | 5.18 | 3.61-7.46 | 6.58e-19 |
| PDAC | 184 | 161 | 23 | 20 | 87.4% | 6.94 | 4.19-11.48 | 4.87e-14 |

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
| All clinicians pooled within note | 40 | 0 | 0 | 1.82e-12 |

## Interpretation boundary

The observed-note result is statistically strong, but three oncologists remain too few for a precise estimate of variation across the broader oncologist population. Evaluator-level generalization should therefore remain a pilot claim.
