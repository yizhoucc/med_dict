# V32 Full Run Review (61 samples)

> Run: v32_full_20260416_224708
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (POST hooks: imaging header, old labs)
> Results: results/v32_full_20260416_224708_results.txt
> Status: **审查中 — 17/61 完成**
> POST hooks fired: 1x imaging_plan (ROW 1), 2x lab_summary old labs (ROW 64 + 1 other)

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 7 | — |

## 已审查 (from test set review + partial)

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅ imaging/lab/referral POST hook |
| 2 | 141 | 1 | medication_plan 漏 pRBC (test set 已知) |
| 3 | 142 | 0 | ✅ Stage IIA, second opinion, full code |
| 5 | 144 | 0 | ✅ Mixed response, Rad Onc referral |
| 8 | 147 | 0 | ✅ pCR + N1 correct |
| 17 | 156 | 0 | ✅ |
| 29 | 168 | 0 | ✅ multifocal + Oncotype Low Risk |
| 46 | 185 | 0 | ✅ sarcoidosis 关键测试通过 |
| 64 | 203 | 0 | ✅ dd AC (no taxol), POST hook |
| 100 | 239 | 0 | ✅ Gemzar + labs |
| 6 | 145 | 1 | Referral Genetics: past genetics referral (already done) |
| 7 | 146 | 1 | medication_plan: "restart Pertuzumab" wrong (it's being STOPPED) |
| 9 | 148 | 1 | therapy_plan: "currently on taxol" wrong (s/p, completed) |
| 10 | 149 | 1 | Type: didn't capture "HR+/HER2-" from A/P |
| 11 | 150 | 0 | ✅ Faslodex+Denosumab, bone mets |
| 12 | 151 | 0 | ✅ DNR/DNI, brain mets GK, 4 imaging plans |
| 14 | 153 | 2 | Response_Assessment error + medication_plan contradictions |

## 待审查

ROW 18, 20, 22, 27, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 49, 50, 52, 53, 54, 57, 59, 61, 63, 65, 66, 68, 70, 72, 73, 78, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97

---

### ROW 7 (coral_idx 146) — 0 P1, 1 P2
- MBC since 2008. ER-/PR-/HER2+ IDC. T2N1 left. Metastatic to supraclavicular + mediastinum. Multiple tx lines (AC+T, taxotere/xeloda/herceptin, tykerb/herceptin, cape/herceptin, pertuzumab/herceptin/taxotere). Off rx since last wk. Second opinion. LVEF 52%.
- P2: **medication_plan "restart Pertuzumab"** — pertuzumab is part of the current rx being STOPPED (A/P: "d/c current rx *****/Herceptin/Taxotere"). The "Rec *****" is a DIFFERENT redacted next-line drug. therapy_plan guesses "T-DM1" — inconsistent with medication_plan. Letter inherits error.
- ✅ Type ER-/PR-/HER2+ ✅。Stage II→IV ✅。Mets: supraclavicular+mediastinum+CW ✅
- ✅ response_assessment: probable minor PD, SUV 2.1 vs 1.8, markers stable ✅
- ✅ current_meds "" (off rx) ✅。lab_summary "Values redacted" ✅ (labs genuinely redacted)
- ✅ lab_plan: recheck markers ✅。goals: palliative ✅

### ROW 6 (coral_idx 145) — 0 P1, 1 P2
- 34yo female, ER+/PR+/HER2-(IHC 2, FISH non-amp) IDC grade I + DCIS, s/p bilateral mastectomy 06/21/19, Oncotype Low Risk, premenopausal, bipolar 2. Started zoladex 06/08 + letrozole at this visit.
- P2: **Referral Genetics "genetics referral"** — 原文 genetics referral 是 04/24/2019 的旧转诊（Myriad 结果已回: Negative）。A/P 无新 genetics referral。Letter 错误说 "referred to genetics specialist"
- ✅ Type ER+/PR+/HER2- ✅。Stage I (1.5cm, 0/1 nodes) ✅。Goals curative ✅
- ✅ lab_summary: Estradiol 172 + Vit D 24 + full CMP/CBC ✅
- ✅ current_meds: letrozole + zoladex ✅。medication_plan: letrozole + zoladex + estradiol monthly + gabapentin prn ✅
- ✅ F/u 3 months ✅

