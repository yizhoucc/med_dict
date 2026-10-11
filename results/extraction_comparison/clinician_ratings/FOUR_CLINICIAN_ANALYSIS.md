# Four-clinician extraction analysis

Updated: 2026-10-11

## Data included

The analysis uses the current exports listed in `CLINICIAN_RATINGS_MANIFEST.md`:

- Oncologist 1 (Simo): 280 breast ratings and 260 PDAC ratings.
- Oncologist 2 (Kevin): 280 breast ratings and 260 PDAC ratings.
- Oncologist 3 (Bolun): 280 breast ratings and 260 PDAC ratings, stored in separate source workbooks.
- Oncologist 4 (Zhengrui): 280 breast ratings and 260 PDAC ratings.

Each breast-cancer evaluation contains 20 notes by 14 required fields, and each PDAC evaluation contains 20 notes by 13 required fields. The blind mapping is fixed: `A` is the inference harness (PL), `B` is the same-model single-prompt baseline (BL), and `TIE` records no preference. The comparison evaluates the presented systems as a whole. Source attribution was shown with PL and was not isolated as a separate component.

Bolun's new workbook also contains 20 forced-choice patient-letter selections. That sheet uses a different task and response instrument, so it is archived as exploratory data and excluded from every extraction result below.

## Main result

| Scope | Judgments | PL | BL | Tie | PL share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 1, breast | 280 | 84 | 22 | 174 | 79.2% |
| Oncologist 1, PDAC | 260 | 75 | 14 | 171 | 84.3% |
| Oncologist 2, breast | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 2, PDAC | 260 | 86 | 9 | 165 | 90.5% |
| Oncologist 3, breast | 280 | 119 | 24 | 137 | 83.2% |
| Oncologist 3, PDAC | 260 | 103 | 30 | 127 | 77.4% |
| Oncologist 4, breast | 280 | 81 | 26 | 173 | 75.7% |
| Oncologist 4, PDAC | 260 | 72 | 38 | 150 | 65.5% |
| **All eight evaluations** | **2,160** | **699** | **171** | **1,290** | **80.3%** |

The breast subtotal is PL 363, BL 80, and 677 ties across 1,120 judgments. PL received 81.9% of the 443 directional breast judgments. The PDAC subtotal is PL 336, BL 91, and 613 ties across 1,040 judgments. PL received 78.7% of the 427 directional PDAC judgments.

Across all judgments, PL preference, tie, and BL preference account for 32.36%, 59.72%, and 7.92%, respectively. The 80.34% figure is conditional on a directional judgment.

## Replication across clinicians and cancer types

All eight evaluator-by-cancer analyses favor PL among directional judgments. At the note level:

- Oncologist 1: breast 18 PL wins, 1 tie, and 1 BL win; PDAC 19 PL wins and 1 BL win.
- Oncologist 2: breast 20 PL wins; PDAC 18 PL wins and 2 ties.
- Oncologist 3: breast 20 PL wins; PDAC 17 PL wins, 2 ties, and 1 BL win.
- Oncologist 4: breast 16 PL wins, 2 ties, and 2 BL wins; PDAC 13 PL wins, 3 ties, and 4 BL wins.

After pooling the four ratings within each note, all 20 breast notes have a positive PL-minus-BL margin. Nineteen PDAC notes have a positive margin and one has a negative margin (`p4`, pooled margin -1). No pooled note is tied.

## Adjusted analysis

Among 870 directional judgments, a population-averaged logistic generalized estimating equation clustered by note and adjusted for evaluator and cancer type estimated an adjusted harness-preference probability of 79.98% (95% CI 75.75%-83.63%). This is +29.98 percentage points above equal preference (95% CI +25.75 to +33.63). Cancer-specific probabilities were 81.66% for breast cancer (95% CI 76.02%-86.21%) and 78.15% for PDAC (95% CI 71.47%-83.62%). The same standardized estimates on the odds scale were 3.995 overall (95% CI 3.124-5.110; `p=2.67e-28`), 4.452 for breast cancer (95% CI 3.169-6.253; `p=7.04e-18`), and 3.576 for PDAC (95% CI 2.505-5.105; `p=2.30e-12`). Full output is in `ADJUSTED_ANALYSIS.md` and `adjusted_gee_results.csv`.

The crude directional share was 699/870, or 80.34% (note-cluster robust 95% CI 76.14%-83.96%). Across all 2,160 judgments, PL, tie, and BL accounted for 32.36%, 59.72%, and 7.92%. The tie-inclusive net preference margin was +24.44 percentage points (note-cluster robust 95% CI +20.42 to +28.47). These are preference effects, not accuracy estimates.

Bolun's PDAC review lowers the overall adjusted odds ratio from 4.15 to 4.00 and the PDAC estimate from 3.71 to 3.58. The direction and statistical conclusion remain unchanged. The result is statistically strong for the observed notes. Four clinicians are still too few for a precise estimate of variation across the broader oncologist population.

## Agreement across four oncologists

Every pair shares all 280 required breast comparisons and all 260 required PDAC comparisons.

| Pair | Breast exact agreement | Breast kappa | PDAC exact agreement | PDAC kappa |
|---|---:|---:|---:|---:|
| Oncologist 1 vs 2 | 82.9% | 0.646 | 80.0% | 0.588 |
| Oncologist 1 vs 3 | 80.0% | 0.644 | 73.8% | 0.532 |
| Oncologist 1 vs 4 | 81.4% | 0.644 | 75.8% | 0.545 |
| Oncologist 2 vs 3 | 73.6% | 0.511 | 73.8% | 0.529 |
| Oncologist 2 vs 4 | 77.1% | 0.533 | 74.2% | 0.520 |
| Oncologist 3 vs 4 | 73.2% | 0.527 | 73.1% | 0.545 |

All four oncologists gave the same verdict on 176 of 280 breast comparisons (62.9%) and 155 of 260 PDAC comparisons (59.6%). In breast cancer, a strict three-of-four majority favored PL in 76 comparisons, BL in 6, and tie in 155; 43 comparisons had no three-vote majority. In PDAC, a strict majority favored PL in 67 comparisons, BL in 7, and tie in 139; 47 had no three-vote majority.

## Field-level pattern

The seven clinician-prioritized core fields contribute 1,040 judgments.

| Core field | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| Active anticancer medications | 132 | 3 | 25 | +129 |
| Stage | 69 | 7 | 84 | +62 |
| Distant metastasis | 23 | 5 | 132 | +18 |
| Regional or overall metastasis | 74 | 26 | 60 | +48 |
| Treatment response | 62 | 12 | 86 | +50 |
| Breast cancer type and receptors | 24 | 14 | 42 | +10 |
| Completed molecular or genetic results | 34 | 10 | 116 | +24 |
| **Overall** | **418** | **77** | **545** | **+341** |

PL received 84.4% of the 495 directional core judgments. Every core field has a positive aggregate margin. Active anticancer medications remain the clearest strength. Medication planning, outside the seven core fields, also has a large margin: PL 98, BL 2, and 60 ties.

Distant metastasis has the smallest normalized core-field margin, followed by breast type and receptor status. Procedure planning is nearly balanced at PL 21 versus BL 20, and laboratory planning has a modest margin at PL 24 versus BL 19. These fields should not be presented as major strengths.

## Manuscript implications

The completed fourth PDAC review closes the coverage gap: all four oncologists evaluated both cancer types. It adds replication without changing the central result. All eight clinician-by-cancer comparisons favor PL among directional judgments, and the adjusted observed-note result remains statistically significant.

The new review also adds a useful negative observation. The pooled PDAC margin is slightly negative for one note, and procedure and laboratory planning remain close to balanced. The manuscript should report these limits alongside the stronger effects for active medication status, stage, metastasis, response, and medication planning. Population-level claims about oncologists should remain framed as pilot findings.

## Figure updates

- Figure 1 separates data partitions, breast development, PDAC adaptation, and locked holdout evaluation in four panels.
- Figure 2 retains all eight evaluator-by-cancer distributions and adds directional harness share as a second panel.
- Figure 3 reports exact agreement and Cohen's kappa together in each populated evaluator-pair cell.
- Figure 4 pools 1,040 core-field judgments in a diverging preference profile.
- Figure 5 ranks note-level margins after pooling four evaluators within each cancer type.
- Figure 6 shows the four-clinician adjusted effect on both probability and odds scales. Panel A reports adjusted probabilities and absolute differences from 50%; Panel B reports the corresponding ORs, p values, and directional sample sizes.
- Figure 7 places the locked staged-ablation counts in the same visual system.
- Figure 8 is a results-informed clinical handoff concept. It does not use unmeasured workload, time, or risk axes.
- The v0.20 manuscript does not use the former technical-audit versus clinician Supplementary Figure S1 because the final test-set LLM audit is not reported as a result.
