# Adjusted physician-preference analysis

Updated: 2026-10-11

## Primary model

The primary estimand is P(harness preference | directional judgment). The analysis excludes ties and uses a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator is included as a fixed effect, and the overall model also includes cancer type. Adjusted probabilities are standardized over the observed covariate distribution. Delta-method confidence limits are calculated on the logit scale and transformed back to probabilities. The absolute effect is the adjusted probability minus 50 percentage points; the odds ratio is the same standardized estimate on the odds scale.

| Scope | Directional judgments | Harness | Baseline | Clusters | Adjusted probability (95% CI) | Difference from 50% (95% CI), pp | Odds ratio (95% CI) | p value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 870 | 699 | 171 | 40 | 79.98% (75.75%-83.63%) | +29.98 (+25.75 to +33.63) | 3.995 (3.124-5.110) | 2.67e-28 |
| Breast cancer | 443 | 363 | 80 | 20 | 81.66% (76.02%-86.21%) | +31.66 (+26.02 to +36.21) | 4.452 (3.169-6.253) | 7.04e-18 |
| PDAC | 427 | 336 | 91 | 20 | 78.15% (71.47%-83.62%) | +28.15 (+21.47 to +33.62) | 3.576 (2.505-5.105) | 2.3e-12 |

The odds ratio compares the adjusted probability of a harness preference with the probability of a baseline preference among non-tie judgments. It is not an odds ratio for clinical correctness.

## Descriptive and tie-inclusive effects

The raw three-category distribution was 699/2160 harness preferences (32.36%), 1290/2160 ties (59.72%), and 171/2160 baseline preferences (7.92%).

The crude harness share was 80.34% among 870 directional judgments (note-cluster robust 95% CI 76.14%-83.96%). The tie-inclusive net preference margin, calculated as the mean of harness=1, tie=0, and baseline=-1 over all required judgments, was +24.44 percentage points (note-cluster robust 95% CI +20.42 to +28.47). Both intervals use sandwich standard errors clustered by note; they are not bootstrap intervals.

## Note-level sensitivity analysis

For each evaluator-cancer combination, field ratings were reduced to one harness-minus-baseline margin per note. Tied note margins were excluded from the exact two-sided sign test.

| Evaluation | Positive notes | Negative notes | Tied notes | Exact p value |
|---|---:|---:|---:|---:|
| Physician 01, Breast cancer | 18 | 1 | 1 | 7.63e-05 |
| Physician 01, PDAC | 19 | 1 | 0 | 4.01e-05 |
| Physician 02, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| Physician 02, PDAC | 18 | 0 | 2 | 7.63e-06 |
| Physician 03, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| Physician 03, PDAC | 17 | 1 | 2 | 0.000145 |
| Physician 04, Breast cancer | 16 | 2 | 2 | 0.00131 |
| Physician 04, PDAC | 13 | 4 | 3 | 0.049 |
| All physicians pooled, Breast cancer | 20 | 0 | 0 | 1.91e-06 |
| All physicians pooled, PDAC | 19 | 1 | 0 | 4.01e-05 |
| All physicians pooled, both cancers | 39 | 1 | 0 | 7.46e-11 |

## Interpretation boundary

The observed-note result is statistically strong, but four physicians remain too few for a precise estimate of variation across the broader physician population. Evaluator-level generalization should therefore remain a pilot claim.
