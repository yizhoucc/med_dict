# Oncologist 02 and inter-rater blind-rating analysis

Date received: 2026-09-11

Download source filename: `Result_Kevin.csv`

Preserved raw export: `oncologist_02_breast_pdac_blind_scores_20260911.csv`

SHA-256: `b60862f308c956d66739d9044d65bf246eafb426620e86df4a9c3d684aae9386`

## Interpretation of labels

The scoring interface uses a fixed blind mapping:

- `A` = PL, the pipeline / inference harness
- `B` = BL, the same-model single-prompt baseline
- `TIE` = neither output is better

This mapping is defined in `build_scoring_blind.py` and was hidden from the clinician.

## Is this a second independent submission?

The new file is not a duplicate of the first clinician export:

- The raw SHA-256 values differ.
- The first export has 291 records; the new export has 539 records.
- The new export covers all 20 breast samples and all 20 PDAC samples. The first export is almost entirely breast-only.
- Among the 278 required breast ratings present in both files, 230 verdicts match and 48 differ. Thus the two files have 82.7% exact agreement, not identical answer vectors.
- The new file contains 10 written comments, including comments and judgments not present in the first file.
- Neither file contains duplicate `(sample, field)` keys.

On the 278 overlapping required breast ratings, Cohen's kappa is 0.645 (observed agreement 82.7%; chance-expected agreement 51.3%). This is consistent with two raters who often agree but make materially different judgments.

The CSV itself does not contain a formal evaluator identifier, so it cannot prove personal identity by itself. However, its distinct content, added PDAC evaluation, different comments, and download filename provide strong evidence that this is a separate clinician submission. It is therefore stored as **Oncologist 02**.

## Oncologist 02: breast-cancer result

All 280 required ratings are present.

| Verdict | Count | Share |
|---|---:|---:|
| PL better (`A`) | 79 | 28.2% |
| BL better (`B`) | 8 | 2.9% |
| Tie | 193 | 68.9% |

Among the 87 decisive breast comparisons, PL won **79/87 (90.8%)**. At the sample level, PL had more field wins than BL in all 20 breast cases.

| Breast field | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| current_meds | 17 | 0 | 3 | +17 |
| stage | 6 | 0 | 14 | +6 |
| distant_met | 2 | 1 | 17 | +1 |
| metastasis | 14 | 0 | 6 | +14 |
| response | 6 | 0 | 14 | +6 |
| type_receptor | 3 | 1 | 16 | +2 |
| genetic_results | 4 | 2 | 14 | +2 |
| genetic_plan | 3 | 0 | 17 | +3 |
| supportive_meds | 5 | 0 | 15 | +5 |
| procedure_plan | 2 | 2 | 16 | 0 |
| imaging_plan | 4 | 2 | 14 | +2 |
| lab_plan | 2 | 0 | 18 | +2 |
| medication_plan | 7 | 0 | 13 | +7 |
| recent_changes | 4 | 0 | 16 | +4 |

For the seven prespecified core breast fields, the result is PL 52 / BL 4 / TIE 84. PL won **92.9% of decisive core comparisons**. Every core field has a nonnegative aggregate PL margin.

## Oncologist 02: PDAC result

The file contains 259 of 260 expected PDAC ratings. The only missing item is `p7 / lab_plan`.

| Verdict | Count | Share |
|---|---:|---:|
| PL better (`A`) | 86 | 33.2% |
| BL better (`B`) | 9 | 3.5% |
| Tie | 164 | 63.3% |

Among the 95 decisive PDAC comparisons, PL won **86/95 (90.5%)**. At the sample level, PL won 18 cases and tied 2; BL won none.

| PDAC field | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| current_meds | 16 | 0 | 4 | +16 |
| stage | 10 | 1 | 9 | +9 |
| distant_met | 3 | 0 | 17 | +3 |
| metastasis | 12 | 0 | 8 | +12 |
| response | 9 | 0 | 11 | +9 |
| genetic_results | 3 | 1 | 16 | +2 |
| genetic_plan | 1 | 0 | 19 | +1 |
| supportive_meds | 5 | 1 | 14 | +4 |
| procedure_plan | 4 | 1 | 15 | +3 |
| imaging_plan | 3 | 0 | 17 | +3 |
| lab_plan | 5 | 1 | 13 | +4 |
| medication_plan | 11 | 0 | 9 | +11 |
| recent_changes | 4 | 4 | 12 | 0 |

## All completed clinician ratings

Across the three completed clinician-by-cancer evaluations, there are 817 required-field judgments:

| Evaluation | PL | BL | Tie | PL share among decisive judgments |
|---|---:|---:|---:|---:|
| Oncologist 01, breast | 84 | 22 | 172 | 79.2% |
| Oncologist 02, breast | 79 | 8 | 193 | 90.8% |
| Oncologist 02, PDAC | 86 | 9 | 164 | 90.5% |
| **All available ratings** | **249** | **39** | **529** | **86.5%** |

The seven prespecified categories contribute 400 ratings across their applicable cancer types. They total PL 156 / BL 18 / TIE 226, so PL accounts for **89.7% of the 174 decisive core judgments**.

| Core category | PL | BL | Tie | Net PL-BL |
|---|---:|---:|---:|---:|
| Active anticancer medications | 50 | 1 | 9 | +49 |
| Stage | 24 | 2 | 34 | +22 |
| Distant metastasis | 8 | 2 | 50 | +6 |
| Regional or overall metastasis | 35 | 1 | 24 | +34 |
| Treatment response | 18 | 1 | 41 | +17 |
| Breast type and receptor status | 8 | 6 | 26 | +2 |
| Completed molecular or genetic results | 13 | 5 | 42 | +8 |

All seven core categories have a positive aggregate PL margin in the available clinician data. The most stable advantages are active anticancer medication and regional or overall metastatic involvement. Breast type and receptor status remains the weakest category because its pooled margin is only 8 to 6.

## Agreement with Oncologist 01

The inter-rater comparison uses the 278 required breast ratings present in both exports. It excludes the first file's optional `findings` and `lab_summary` rows and its stray `p8 / supportive_meds` row.

| Old verdict -> new verdict | Count |
|---|---:|
| A -> A | 62 |
| A -> B | 2 |
| A -> TIE | 20 |
| B -> A | 7 |
| B -> B | 6 |
| B -> TIE | 9 |
| TIE -> A | 10 |
| TIE -> TIE | 162 |

No first-rater `TIE` became a second-rater `B`; 10 became `A`. Direct reversals were uncommon: 2 ratings changed from PL to BL, while 7 changed from BL to PL.

Field-level exact agreement ranged from 60% to 100%:

- 100%: `genetic_plan` and `procedure_plan`
- 90%: `current_meds`, `distant_met`, and `lab_plan`
- 85%: `genetic_results`, `supportive_meds`, and `imaging_plan`
- 75%: `stage`, `metastasis`, and `response`
- 60%: `type_receptor` and `medication_plan`

The lower agreement for receptor extraction and medication planning is a useful target for adjudication and clearer scoring guidance.

## Breast-cancer combined result

Pooling the two clinicians' recorded required breast ratings gives 558 rating decisions:

- PL better: 163
- BL better: 30
- Tie: 365
- PL share among decisive judgments: **163/193 (84.5%)**

This pooled count is descriptive. Ratings are clustered within samples, fields, and clinicians, so the 558 rows should not be treated as independent observations in a naive significance test. The final analysis should use a paired or hierarchical method and should preserve clinician identity as a grouping variable.

As an exploratory sample-level check, the combined PL-minus-BL field margin was positive for all 20 breast notes. A two-sided exact sign test at the note level gives `p=1.9e-6`. For the second oncologist's PDAC review, 18 note-level margins were positive and two were tied, giving `p=7.6e-6` after excluding ties. These tests avoid treating every field as independent, but they were not the prespecified final model and do not resolve the limited number of clinicians. They should support internal interpretation, not replace the planned evaluator-note-field analysis.

## Data-quality notes

- The raw second export is preserved unchanged.
- It contains 539 unique rating records: 280 breast and 259 PDAC.
- One expected PDAC rating is missing: `p7 / lab_plan`.
- There are no unexpected fields or sample identifiers in the second export.
- The first export remains separately preserved and is not overwritten.
