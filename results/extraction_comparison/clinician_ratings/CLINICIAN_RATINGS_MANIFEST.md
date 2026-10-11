# Clinician rating files

Last updated: 2026-10-10

Current combined results are documented in `FOUR_CLINICIAN_ANALYSIS.md`. `THREE_CLINICIAN_ANALYSIS.md` and the earlier one- and two-clinician analyses are retained as dated intermediate results.

The project owner confirmed the evaluator names below. Raw rating files are preserved byte-for-byte. In the blind scoring interface, `A` denotes PL, `B` denotes BL, and `TIE` denotes no preference.

The project owner confirmed that all four exports were completed against the intended finalized clinician-review package and baseline version. Evaluators were not told which output was the harness, whether the systems shared a base model, or which side was expected to perform better. The harness side included source attribution as part of the complete system output being evaluated. The exported score files do not embed the HTML or input-file hashes, so this confirmation is recorded here as study provenance rather than cryptographic verification.

## Current files

| Evaluator | File | Breast coverage | PDAC coverage | SHA-256 | Notes |
|---|---|---:|---:|---|---|
| Simo | `simo_breast_pdac_blind_scores_20260921.csv` | 280/280 required ratings, plus 12 optional ratings | 260/260 required ratings | `9c5bf126accdb45f8ef55da3ebaa873cc75a8998efac965f1af39abe0f9ee684` | Complete replacement for the earlier partial Simo export. |
| Kevin | `kevin_breast_pdac_blind_scores_20261005.csv` | 280/280 required ratings | 260/260 required ratings | `58d2fa1144e35a831af399a1db9cbe3bd00bd869ff0376591be8385a3c3f84d8` | Complete replacement for the earlier export. The added `p7 / lab_plan` judgment is `TIE`; the other 539 judgments are unchanged. |
| Bolun | `bolun_breast_blind_scores_20260921.xlsx`; `bolun_pdac_blind_scores_and_letter_selection_20261006.xlsx` | 280/280 required ratings | 260/260 required ratings | `e68344183a63ec43044c2f9546c08d67425b95bf3b4f3a941eaa40c17925bc0e`; `6173f85fdbb4737102b735474d02b7d74294eeb463ac668f45b9ff0ef7ee635d` | Breast and PDAC extraction ratings are complete. The files have no overlapping extraction keys. The PDAC workbook also contains a separate exploratory letter-selection sheet. |
| Zhengrui | `zhengrui_breast_pdac_blind_scores_20261005.csv` | 280/280 required ratings | 260/260 required ratings | `33bad2e5594cb932aeff8c253dd137269cd8d232c5dbf7e75e0b551d8191db74` | Fourth clinician export; both cancer types are complete. |

## Source and legacy files

- Simo's source download was `/Users/yizhoucc/Downloads/Result_Simo_Full.csv`.
- Kevin's complete source download was `/Users/yizhoucc/Downloads/Result_Kevin_full.csv`.
- Bolun's source download was `/Users/yizhoucc/Downloads/blind_scores_breast 20.xlsx`.
- Bolun's PDAC source download was `/Users/yizhoucc/Downloads/Result_Bolun_ancreatic_blind_scores and letter selection.csv.xlsx`.
- Zhengrui's source download was `/Users/yizhoucc/Downloads/Result_Zhengrui.csv`.
- `oncologist_01_breast_blind_scores_20260907.csv` is Simo's earlier partial export. It remains unchanged for provenance and should not be used as the current Simo dataset.
- `kevin_breast_pdac_blind_scores_20260911.csv` and its byte-identical legacy alias `oncologist_02_breast_pdac_blind_scores_20260911.csv` contain Kevin's earlier 539-item export. They remain unchanged for provenance and should not be used in the current analysis.

## Exploratory letter data

Bolun's PDAC workbook contains 20 forced-choice letter selections in a separate `letter` sheet: A 18, B 1, and C 1. In the letter-review instrument, A is the direct ChatGPT letter, B is the direct same-model Qwen baseline, and C is the harness-based letter. These observations use a different task and response instrument from the extraction ratings. They are archived for exploratory follow-up and are not included in the primary extraction totals, adjusted models, or clinician-rating figures.
