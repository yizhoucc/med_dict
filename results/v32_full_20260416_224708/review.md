# V32 Full Run Review (61 samples)

> Run: v32_full_20260416_224708
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (POST hooks: imaging header, old labs)
> Results: results/v32_full_20260416_224708_results.txt
> Status: **审查完成 — 61/61**
> POST hooks fired: 1x imaging_plan (ROW 1), 2x lab_summary old labs (ROW 64 + 1 other)

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 10 | 0.16/sample |

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
| 18 | 157 | 0 | ✅ papillary CA + ITC, endocrine therapy |
| 20 | 159 | 0 | ✅ letrozole+palbociclib, Foundation One, dental |
| 22 | 161 | 0 | ✅ pneumonitis, conditional PET plan |
| 27 | 166 | 0 | ✅ stable disease, goserelin+letrozole |
| 30 | 169 | 0 | ✅ neoadjuvant planned, TTE pre-chemo |
| 33 | 172 | 0 | ✅ NED on letrozole |
| 34 | 173 | 0 | ✅ local recurrence |
| 36 | 175 | 0 | ✅ Abraxane+zoladex+tamoxifen |
| 37 | 176 | 0 | ✅ Stage IIA, full code |
| 40 | 179 | 1 | letrozole in supportive_meds (oncologic drug) |
| 41 | 180 | 0 | ✅ |
| 42 | 181 | 0 | ✅ mammogram planned |
| 43 | 182 | 0 | ✅ Stage I, full code |
| 44 | 183 | 0 | ✅ post-neoadjuvant, CT for lung nodule |
| 49 | 188 | 0 | ✅ surrogate decision maker |
| 50 | 189 | 0 | ✅ ibrance+letrozole+xgeva, good control |
| 52 | 191 | 0 | ✅ staging scans ordered |
| 53 | 192 | 0 | ✅ |
| 54 | 193 | 0 | ✅ leuprolide+letrozole+zoledronic acid |
| 57 | 196 | 0 | ✅ post-neoadjuvant TCH+P |
| 59 | 198 | 0 | ✅ NED, mammogram+MRI alternating |
| 61 | 200 | 0 | ✅ Stage I |
| 63 | 202 | 0 | ✅ disappointing chemo response |
| 65 | 204 | 0 | ✅ ISPY trial, TTE |
| 66 | 205 | 0 | ✅ metaplastic CA, TNBC, second opinion |
| 68 | 207 | 0 | ✅ NED after TCHP |
| 70 | 209 | 0 | ✅ bilateral, post-neoadjuvant |
| 72 | 211 | 0 | ✅ Stage IA |
| 73 | 212 | 0 | ✅ fat necrosis, arimidex |
| 78 | 217 | 0 | ✅ disease progression, liver mets |
| 80 | 219 | 0 | ✅ Stage IA |
| 82 | 221 | 0 | ✅ Stage IB (prognostic), DEXA |
| 83 | 222 | 0 | ✅ significant response on PET |
| 84 | 223 | 0 | ✅ capecitabine, CT+MRI ordered |
| 85 | 224 | 1 | Response_Assessment JSON error |
| 86 | 225 | 0 | ✅ PD on letrozole+ribociclib |
| 87 | 226 | 0 | ✅ Stage IIIA (pT2 N2a) |
| 88 | 227 | 0 | ✅ capecitabine, full code |
| 90 | 229 | 0 | ✅ post-neoadjuvant 60% cellularity |
| 91 | 230 | 0 | ✅ everolimus+exemestane, PET planned |
| 92 | 231 | 0 | ✅ epirubicin, liver improvement |
| 94 | 233 | 0 | ✅ mammogram+MRI planned |
| 95 | 234 | 0 | ✅ good neoadjuvant response |
| 97 | 236 | 0 | ✅ Stage IA |

## 待审查

None — all 61 samples reviewed

## P2 Summary

| ROW | 问题 |
|-----|------|
| 2 | medication_plan 漏 "1 unit pRBC" |
| 6 | Referral Genetics: past referral extracted as current |
| 7 | medication_plan: wrong next-line drug (pertuzumab vs redacted) |
| 9 | therapy_plan: "currently on taxol" (s/p completed) |
| 10 | Type: missed "HR+/HER2-" from A/P |
| 11 | supportive_meds: fulvestrant (oncologic drug) |
| 14 | Response_Assessment error + medication_plan contradictions (×2) |
| 40 | supportive_meds: letrozole (oncologic drug) |
| 85 | Response_Assessment JSON error |

## V31 vs V32 Final Comparison

| 指标 | V31 (Qwen2.5-32B-AWQ) | V32 (Qwen3.5-35B-A3B-GPTQ) |
|------|------------------------|----------------------------------|
| Samples | 61 | 61 |
| P0 | 0 (0%) | 0 (0%) |
| P1 | 0 (0%) | 0 (0%) |
| P2 | 112 (1.84/sample) | 10 (0.16/sample) |
| 速度 | ~2 hours | 14.8 min (8x faster) |
| P2 reduction | — | **91% fewer P2s** |

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

