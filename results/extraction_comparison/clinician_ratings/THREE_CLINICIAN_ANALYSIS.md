# Three-clinician extraction analysis

Updated: 2026-09-23

## Data included

The analysis uses the current complete exports listed in `CLINICIAN_RATINGS_MANIFEST.md`:

- Oncologist 1 (Simo): 280 breast ratings and 260 PDAC ratings.
- Oncologist 2 (Kevin): 280 breast ratings and 259 PDAC ratings. The missing item is `p7 / lab_plan`.
- Oncologist 3 (Bolun): 280 breast ratings. No PDAC extraction ratings were submitted.

The blind mapping is fixed: `A` is the inference harness (PL), `B` is the same-model single-prompt baseline (BL), and `TIE` records no preference.

The project owner confirmed that all three exports were completed against the intended finalized baseline package. Evaluators were not told the system identities, whether the outputs shared a base model, or which side was expected to perform better. Source attribution was displayed with PL as part of the complete harness output. The comparison therefore evaluates the presented systems as a whole and does not isolate attribution as a component.

## Main result

| Scope | Judgments | PL | BL | Tie | PL share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 1, breast | 280 | 84 | 22 | 174 | 79.2% |
| Oncologist 1, PDAC | 260 | 75 | 14 | 171 | 84.3% |
| Oncologist 2, breast | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 2, PDAC | 259 | 86 | 9 | 164 | 90.5% |
| Oncologist 3, breast | 280 | 119 | 24 | 137 | 83.2% |
| **All completed evaluations** | **1,359** | **443** | **77** | **839** | **85.2%** |

The breast-cancer subtotal is PL 282, BL 54, and 504 ties across 840 judgments. PL received 83.9% of the 336 directional breast judgments.

The PDAC subtotal is PL 161, BL 23, and 335 ties across 519 judgments. PL received 87.5% of the 184 directional PDAC judgments.

Across all 1,359 judgments, PL preference, BL preference, and tie account for 32.6%, 5.7%, and 61.7%, respectively. The 85.2% figure is conditional on a directional judgment.

## Replication across clinicians and cancer types

Every completed evaluator-by-cancer analysis favored PL among directional judgments. At the note level:

- Oncologist 1: breast 18 PL wins, 1 tie, 1 BL win; PDAC 19 PL wins and 1 BL win.
- Oncologist 2: breast 20 PL wins; PDAC 18 PL wins and 2 ties.
- Oncologist 3: breast 20 PL wins.

After pooling ratings within each note, all 20 breast notes and all 20 PDAC notes have a positive PL-minus-BL margin. This is descriptive because ratings from the same note and clinician are correlated.

## Current significance status

The adjusted primary analysis is now complete. Among 520 directional judgments, a population-averaged logistic generalized estimating equation clustered by note and adjusted for evaluator and cancer type estimated an odds ratio of 5.70 for harness preference (95% CI 4.24-7.66; `p<0.001`). Cancer-specific estimates were 5.18 for breast cancer (95% CI 3.61-7.46) and 6.94 for PDAC (95% CI 4.19-11.48), both `p<0.001`. Full reproducible output is in `ADJUSTED_ANALYSIS.md` and `adjusted_gee_results.csv`.

Two-sided exact sign tests on within-note PL-minus-BL margins give the following results:

| Evaluation | Positive notes | Negative notes | Tied notes | Exact p value |
|---|---:|---:|---:|---:|
| Oncologist 1, breast | 18 | 1 | 1 | 0.000076 |
| Oncologist 1, PDAC | 19 | 1 | 0 | 0.000040 |
| Oncologist 2, breast | 20 | 0 | 0 | 0.0000019 |
| Oncologist 2, PDAC | 18 | 0 | 2 | 0.0000076 |
| Oncologist 3, breast | 20 | 0 | 0 | 0.0000019 |

Thus, the result is statistically significant for the observed notes within every completed evaluator-by-cancer analysis. Pooling clinicians within each cancer type also gives 20 positive and 0 negative note margins for both breast cancer and PDAC (`p=0.0000019` for each exact sign test).

This does not provide the same level of evidence for generalization across oncologists. If each clinician is reduced to one independent directional result, all three breast oncologists favor PL, but a two-sided exact sign test with 3 of 3 positive gives `p=0.25`. The two PDAC oncologists give `p=0.50`. The completed GEE uses repeated note-field observations without treating them as independent, but it does not remove the uncertainty caused by having only three clinicians.

For a simple clinician-level two-sided sign test, 6 of 6 clinicians favoring PL is the smallest result below 0.05 (`p=0.03125`). If one clinician favors BL, 8 of 9 clinicians are required (`p=0.0391`). These calculations discard effect size and are conservative, but they show why three clinicians support a pilot result more strongly than a broad claim about oncologists in general.

## Breast-cancer agreement across three oncologists

Each pair shares all 280 required breast comparisons.

| Pair | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Oncologist 1 vs 2 | 82.9% | 0.646 |
| Oncologist 1 vs 3 | 80.0% | 0.644 |
| Oncologist 2 vs 3 | 73.6% | 0.511 |

Across the 280 breast note-field comparisons, all three oncologists gave the same verdict on 192 (68.6%). A simple three-rater majority favored PL in 95 comparisons, BL in 15, and tie in 168. Two comparisons had one PL, one BL, and one tie, so no majority existed.

Agreement is substantial enough to show that the direction is not driven by one evaluator, but it is not perfect. The adjusted model retains evaluator identity and clusters repeated field and evaluator judgments within each note.

## Field-level pattern

The seven clinician-prioritized core fields contribute 660 judgments.

| Core field | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| Active anticancer medications | 83 | 2 | 15 | +81 |
| Stage | 41 | 4 | 55 | +37 |
| Distant metastasis | 15 | 2 | 83 | +13 |
| Regional or overall metastasis | 60 | 5 | 35 | +55 |
| Treatment response | 37 | 3 | 60 | +34 |
| Breast cancer type and receptors | 18 | 12 | 30 | +6 |
| Completed molecular or genetic results | 22 | 6 | 72 | +16 |
| **Overall** | **276** | **34** | **350** | **+242** |

PL received 89.0% of the 310 directional core judgments. Every core field has a positive aggregate margin.

The clearest strengths are active anticancer medications, regional or overall metastatic involvement, and treatment response. Medication planning, outside the seven core fields, also has a large margin: PL 59, BL 0, and 41 ties.

Breast type and receptor status remains the weakest core result at PL 18 versus BL 12. Procedure planning (14 versus 12) and laboratory planning (13 versus 9) also have small margins. These fields should not be presented as major strengths.

## Manuscript implications

The three-clinician data support stronger pilot wording than the previous two-clinician draft:

1. The breast result now has three independent clinical evaluations.
2. The PDAC transfer result now has two independent clinical evaluations.
3. The overall direction appears in all five evaluator-by-cancer analyses, not only in pooled counts.
4. The strongest gains match the intended safeguards for treatment status, metastatic classification, temporal reasoning, and medication plans.
5. Receptor status and several plan fields remain useful negative controls because the advantage is smaller.

The draft can report a statistically significant preference among the observed ratings. It should still avoid claiming precise generalization to the wider oncologist population because only three oncologists participated. A naive test that treats all 1,359 judgments as independent would overstate precision.

## Figure updates

- Figure 2 now contains five evaluator-by-cancer distributions.
- Figure 3 now shows pairwise exact agreement and Cohen's kappa for all three breast oncologists. It replaces the old two-rater confusion matrix.
- Figure 4 pools the updated 660 core-field judgments.
- Figure 5 pools three breast evaluators and two PDAC evaluators at the note level.
- Supplementary Figure S1 uses the updated clinician field margins.
