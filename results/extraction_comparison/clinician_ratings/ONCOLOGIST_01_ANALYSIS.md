# Oncologist 01 blind-rating analysis

Date received: 2026-09-07

Raw export: `oncologist_01_breast_blind_scores_20260907.csv`

SHA-256: `6484eb1d3c1ac9f8d4ff5f19e1eecbd628f0e7599e1c4a29fe6d574e1baeae33`

## Interpretation of labels

The scoring interface uses a fixed blind mapping:

- `A` = PL, the pipeline / inference harness
- `B` = BL, the single-prompt baseline
- `TIE` = neither output is better

This mapping is verified in `build_scoring_blind.py`; it was hidden from the clinician in the scoring interface.

## Evaluation provenance

The project owner confirmed that the oncologist reviewed the **newer PL and BL outputs**. The scoring page/template retained stale filenames and was not renamed when its displayed results were updated. Therefore, the legacy filenames referenced by the page-building code must not be used to infer which output version the clinician saw.

The exported CSV contains judgments but does not embed the hashes or version identifiers of the displayed PL and BL artifacts. Those identifiers should be recorded separately before final manuscript submission so the evaluated run can be reproduced exactly.

## Main result

On the 14 required breast-cancer extraction fields, the oncologist recorded:

| Verdict | Count | Share of all 278 recorded required ratings |
|---|---:|---:|
| PL better (`A`) | 84 | 30.2% |
| BL better (`B`) | 22 | 7.9% |
| Tie | 172 | 61.9% |

Among the 106 non-tied comparisons, PL won **84/106 (79.2%)** and BL won 22/106 (20.8%).

At the sample level, comparing the number of PL and BL field wins within each case:

- PL won 18 of 20 samples.
- BL won 1 sample (`b13`).
- 1 sample was tied (`b10`).

This is strong preliminary evidence from this rater that the harness adds clinical extraction value over the same-model baseline. It is not yet an inter-rater result because this file contains only one clinician's judgments.

## Core clinical questions

For the seven central clinical fields, PL had 51 wins, BL had 12 wins, and 77 ties. PL therefore won **81.0% of decisive core-field comparisons**. PL was at least as good as BL in every core field when aggregated across the 20 samples: six fields favored PL and one was even.

| Core field | PL | BL | Tie | Net PL−BL |
|---|---:|---:|---:|---:|
| Current medications | 17 | 1 | 2 | +16 |
| Cancer stage | 8 | 1 | 11 | +7 |
| Distant metastasis | 3 | 1 | 16 | +2 |
| Metastasis, including regional nodes | 9 | 1 | 10 | +8 |
| Treatment response | 3 | 1 | 16 | +2 |
| Type / ER-PR-HER2 receptors | 5 | 5 | 10 | 0 |
| Molecular / genetic results | 6 | 2 | 12 | +4 |

The strongest core advantage is current anticancer medications. Type/receptor extraction is the main unresolved core area: the rater split 5:5, with one comment expressing uncertainty about whether PR positivity was supported.

## All fields

| Field | Ratings | PL | BL | Tie | Net PL−BL |
|---|---:|---:|---:|---:|---:|
| current_meds | 20 | 17 | 1 | 2 | +16 |
| stage | 20 | 8 | 1 | 11 | +7 |
| distant_met | 20 | 3 | 1 | 16 | +2 |
| metastasis | 20 | 9 | 1 | 10 | +8 |
| response | 20 | 3 | 1 | 16 | +2 |
| type_receptor | 20 | 5 | 5 | 10 | 0 |
| genetic_results | 20 | 6 | 2 | 12 | +4 |
| genetic_plan | 19 | 3 | 0 | 16 | +3 |
| supportive_meds | 20 | 4 | 2 | 14 | +2 |
| procedure_plan | 20 | 2 | 2 | 16 | 0 |
| imaging_plan | 20 | 5 | 3 | 12 | +2 |
| lab_plan | 20 | 1 | 2 | 17 | −1 |
| medication_plan | 20 | 13 | 0 | 7 | +13 |
| recent_changes | 19 | 5 | 1 | 13 | +4 |
| lab_summary (optional) | 6 | 0 | 1 | 5 | −1 |
| findings (optional) | 6 | 5 | 0 | 1 | +5 |

Outside the core fields, medication planning is another major PL strength. Lab planning is the only required field with a negative aggregate margin, although the difference is only one rating. Procedure planning is even.

## Clinician comments

Only two ratings contain written comments:

- `b2 / type_receptor` — BL preferred; “unsure if PR positive”. This should be manually checked against the source note before using receptor results as a paper example.
- `b2 / imaging_plan` — PL preferred; “B gives results but didn't mention PET/CT”. This supports the claim that the baseline sometimes summarizes completed findings instead of extracting the future plan.

## Data-quality notes

- The file contains 291 rating records: 290 labeled as breast samples and one stray `p8` record.
- Two required breast ratings are missing: `b12 / genetic_plan` and `b19 / recent_changes`. Therefore the required-field total is 278 rather than 280.
- The stray record is `p8 / supportive_meds / A`. Because the file otherwise covers breast samples `b1`–`b20`, and `b7 / supportive_meds / A` is already present, this row is treated as an accidental extra entry and excluded from every aggregate above. The raw CSV is preserved unchanged.
- Optional `lab_summary` and `findings` ratings appear for six samples each. They are reported separately and are not included in the 14-field primary result.

## Suggested paper-ready summary

> In a blinded comparison by one oncologist, the inference harness was preferred over the same-model single-prompt baseline in 84 of 106 decisive comparisons across required extraction fields (79.2%), with 172 additional ties. Aggregated by case, the harness won 18 of 20 cases, tied one, and lost one. These results are preliminary pending additional clinician ratings.
