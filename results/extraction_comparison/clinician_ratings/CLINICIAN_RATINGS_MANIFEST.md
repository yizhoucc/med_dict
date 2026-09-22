# Clinician rating files

Last updated: 2026-09-21

The project owner confirmed the evaluator names below. Raw rating files are preserved byte-for-byte. In the blind scoring interface, `A` denotes PL, `B` denotes BL, and `TIE` denotes no preference.

## Current files

| Evaluator | File | Breast coverage | PDAC coverage | SHA-256 | Notes |
|---|---|---:|---:|---|---|
| Simo | `simo_breast_pdac_blind_scores_20260921.csv` | 280/280 required ratings, plus 12 optional ratings | 260/260 required ratings | `9c5bf126accdb45f8ef55da3ebaa873cc75a8998efac965f1af39abe0f9ee684` | Complete replacement for the earlier partial Simo export. |
| Kevin | `kevin_breast_pdac_blind_scores_20260911.csv` | 280/280 required ratings | 259/260 required ratings | `b60862f308c956d66739d9044d65bf246eafb426620e86df4a9c3d684aae9386` | Both cancer types are present. The only missing rating is `p7 / lab_plan`. |
| Bolun | `bolun_breast_blind_scores_20260921.xlsx` | 280/280 required ratings | Not evaluated | `e68344183a63ec43044c2f9546c08d67425b95bf3b4f3a941eaa40c17925bc0e` | Breast extraction ratings are complete. The workbook contains no PDAC extraction ratings. |

## Source and legacy files

- Simo's source download was `/Users/yizhoucc/Downloads/Result_Simo_Full.csv`.
- Bolun's source download was `/Users/yizhoucc/Downloads/blind_scores_breast 20.xlsx`.
- `oncologist_01_breast_blind_scores_20260907.csv` is Simo's earlier partial export. It remains unchanged for provenance and should not be used as the current Simo dataset.
- `oncologist_02_breast_pdac_blind_scores_20260911.csv` is byte-identical to the named Kevin copy. It remains unchanged so existing analyses and links continue to work.
