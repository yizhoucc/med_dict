# Four-clinician extraction analysis

Updated: 2026-10-05

## Data included

The analysis uses the current exports listed in `CLINICIAN_RATINGS_MANIFEST.md`:

- Oncologist 1 (Simo): 280 breast ratings and 260 PDAC ratings.
- Oncologist 2 (Kevin): 280 breast ratings and 260 PDAC ratings.
- Oncologist 3 (Bolun): 280 breast ratings. No PDAC ratings were submitted.
- Oncologist 4 (Zhengrui): 280 breast ratings and 260 PDAC ratings.

The blind mapping is fixed: `A` is the inference harness (PL), `B` is the same-model single-prompt baseline (BL), and `TIE` records no preference. The comparison evaluates the presented systems as a whole. Source attribution was shown with PL and was not isolated as a separate component.

## Main result

| Scope | Judgments | PL | BL | Tie | PL share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 1, breast | 280 | 84 | 22 | 174 | 79.2% |
| Oncologist 1, PDAC | 260 | 75 | 14 | 171 | 84.3% |
| Oncologist 2, breast | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 2, PDAC | 260 | 86 | 9 | 165 | 90.5% |
| Oncologist 3, breast | 280 | 119 | 24 | 137 | 83.2% |
| Oncologist 4, breast | 280 | 81 | 26 | 173 | 75.7% |
| Oncologist 4, PDAC | 260 | 72 | 38 | 150 | 65.5% |
| **All completed evaluations** | **1,900** | **596** | **141** | **1,163** | **80.9%** |

The breast subtotal is PL 363, BL 80, and 677 ties across 1,120 judgments. PL received 81.9% of the 443 directional breast judgments. The PDAC subtotal is PL 233, BL 61, and 486 ties across 780 judgments. PL received 79.3% of the 294 directional PDAC judgments.

Across all judgments, PL preference, BL preference, and tie account for 31.4%, 7.4%, and 61.2%, respectively. The 80.9% figure is conditional on a directional judgment.

## Replication across clinicians and cancer types

All seven evaluator-by-cancer analyses favor PL among directional judgments. At the note level:

- Oncologist 1: breast 18 PL wins, 1 tie, and 1 BL win; PDAC 19 PL wins and 1 BL win.
- Oncologist 2: breast 20 PL wins; PDAC 18 PL wins and 2 ties.
- Oncologist 3: breast 20 PL wins.
- Oncologist 4: breast 16 PL wins, 2 ties, and 2 BL wins; PDAC 13 PL wins, 3 ties, and 4 BL wins.

After pooling ratings within each note, all 20 breast notes have a positive PL-minus-BL margin. Nineteen PDAC notes have a positive margin and one is tied. No pooled note has a negative margin.

## Adjusted analysis

Among 737 directional judgments, a population-averaged logistic generalized estimating equation clustered by note and adjusted for evaluator and cancer type estimated an odds ratio of 4.15 for harness preference (95% CI 3.25-5.29; `p<0.001`). Cancer-specific estimates were 4.45 for breast cancer (95% CI 3.17-6.25) and 3.71 for PDAC (95% CI 2.64-5.23), both `p<0.001`. Full output is in `ADJUSTED_ANALYSIS.md` and `adjusted_gee_results.csv`.

The fourth clinician is more critical than the first three, especially for PDAC. This lowers the adjusted effect estimate from the earlier three-clinician analysis, but the direction and statistical conclusion remain unchanged. The result is statistically strong for the observed notes. Four clinicians are still too few for a precise estimate of variation across the broader oncologist population.

## Breast-cancer agreement across four oncologists

Each pair shares all 280 required breast comparisons.

| Pair | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Oncologist 1 vs 2 | 82.9% | 0.646 |
| Oncologist 1 vs 3 | 80.0% | 0.644 |
| Oncologist 1 vs 4 | 81.4% | 0.644 |
| Oncologist 2 vs 3 | 73.6% | 0.511 |
| Oncologist 2 vs 4 | 77.1% | 0.533 |
| Oncologist 3 vs 4 | 73.2% | 0.527 |

All four oncologists gave the same verdict on 176 of 280 comparisons (62.9%). A strict three-of-four majority favored PL in 76 comparisons, BL in 6, and tie in 155. The remaining 43 comparisons had no three-vote majority.

## Field-level pattern

The seven clinician-prioritized core fields contribute 920 judgments.

| Core field | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| Active anticancer medications | 114 | 2 | 24 | +112 |
| Stage | 56 | 6 | 78 | +50 |
| Distant metastasis | 20 | 4 | 116 | +16 |
| Regional or overall metastasis | 71 | 17 | 52 | +54 |
| Treatment response | 47 | 10 | 83 | +37 |
| Breast cancer type and receptors | 24 | 14 | 42 | +10 |
| Completed molecular or genetic results | 30 | 9 | 101 | +21 |
| **Overall** | **362** | **62** | **496** | **+300** |

PL received 85.4% of the 424 directional core judgments. Every core field has a positive aggregate margin. Active anticancer medications remain the clearest strength. Medication planning, outside the seven core fields, also has a large margin: PL 83, BL 2, and 55 ties.

Breast type and receptor status remains the weakest core result. Procedure planning is nearly balanced at PL 19 versus BL 17, and laboratory planning is exactly balanced at 17 versus 17. These fields should not be presented as major strengths.

## Manuscript implications

The fourth clinician strengthens the replication claim while making the effect estimate more conservative. The draft can report that all seven clinician-by-cancer evaluations favor PL among directional judgments and that the adjusted observed-note result is statistically significant. It should continue to describe the work as a pilot and avoid a broad population-level claim about oncologists.

The fourth review also sharpens the negative findings. PL's advantage is strongest for active medication status, staging, metastasis, response, and medication planning. Procedure and laboratory plans show little or no aggregate advantage. This pattern is more informative than a uniformly positive result because it identifies where the harness contributes and where it does not.

## Figure updates

- Figure 1 now records four breast evaluators and three PDAC evaluators.
- Figure 2 contains seven evaluator-by-cancer distributions.
- Figure 3 shows all six pairwise breast agreement estimates.
- Figure 4 pools 920 core-field judgments.
- Figure 5 pools four breast evaluators and three PDAC evaluators at the note level.
- Figure 6 uses the four-clinician adjusted model.
- Supplementary Figure S1 uses the updated clinician field margins.
