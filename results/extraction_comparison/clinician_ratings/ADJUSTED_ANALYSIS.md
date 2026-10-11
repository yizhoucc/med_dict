# Adjusted clinician-preference analysis

Updated: 2026-10-10

## Primary model

The primary analysis excludes ties and models whether a directional rating favors the inference harness. It uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. This approach accounts for repeated ratings of fields and evaluators within a note without treating all directional ratings as independent.

| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted harness probability | Odds ratio | 95% CI | p value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 870 | 699 | 171 | 40 | 80.0% | 4.00 | 3.12-5.11 | 2.67e-28 |
| Breast cancer | 443 | 363 | 80 | 20 | 81.7% | 4.45 | 3.17-6.25 | 7.04e-18 |
| PDAC | 427 | 336 | 91 | 20 | 78.1% | 3.58 | 2.50-5.10 | 2.3e-12 |

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
| Oncologist 03, PDAC | 17 | 1 | 2 | 0.000145 |
| Oncologist 04, Breast cancer | 16 | 2 | 2 | 0.00131 |
| Oncologist 04, PDAC | 13 | 4 | 3 | 0.049 |
| All clinicians pooled, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| All clinicians pooled, PDAC | 19 | 1 | 0 | 4.01e-05 |
| All clinicians pooled, both cancers | 39 | 1 | 0 | 7.46e-11 |

## Interpretation boundary

The observed-note result is statistically strong, but four oncologists remain too few for a precise estimate of variation across the broader oncologist population. Evaluator-level generalization should therefore remain a pilot claim.
