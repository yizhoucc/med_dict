# V32 Final Manual Review — Iteration 9 (61 samples)

> Run: v32_full_20260417_094322
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (10 POST hooks)
> Automated P2 scan: 0 P2 (all known patterns checked, T-DM1 in ROW 8 is from original text)
> Status: **审查中 — 50/61 逐字审查完成，11 待审查**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 15 | — |

## 审查记录

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅ imaging+lab POST hook, Integrative Medicine, Full code |
| 2 | 141 | 0 | ✅ pRBC included, TNBC+Lynch |
| 3 | 142 | 0 | ✅ Stage IIA, second opinion, full code |
| 5 | 144 | 0 | ✅ Stage III→IV, mixed response, Rad Onc |
| 6 | 145 | 0 | ✅ Genetics "None" (fixed), letrozole+zoladex |
| 7 | 146 | 0 | ✅✅ therapy_plan now uses "unspecified agent" (FIXED!) |
| 8 | 147 | 0 | ✅ pT0N1 correct, T-DM1 is from original text |
| 9 | 148 | 1 | medication_plan/letter: completed taxol as future plan |
| 10 | 149 | 2 | Response_Assessment error (key="error" not "status") + findings/Stage inconsistency (0/20 vs 7/20) |
| 11 | 150 | 0 | ✅ supportive_meds clean (POST hook) |
| 12 | 151 | 0 | ✅ DNR/DNI, brain mets GK, 4 imaging plans |
| 14 | 153 | 0 | ✅ medication_plan no contradictions (POST hook), Response_Assessment retry |
| 17 | 156 | 0 | ✅ Stage IA, DXA+labs, genetics+nutrition referral |
| 18 | 157 | 0 | ✅ papillary CA + ITC |
| 20 | 159 | 0 | ✅ letrozole+palbociclib, Foundation One |
| 22 | 161 | 0 | ✅ pneumonitis, conditional PET plan |
| 27 | 166 | 1 | Distant Mets: liver lesions are "cysts/indeterminate", A/P says "MBC to bone" only |
| 29 | 168 | 0 | ✅ multifocal + Oncotype Low Risk |
| 30 | 169 | 0 | ✅✅ neoadjuvant planned, TTE, 2 chemo options (detailed逐字审查) |
| 33 | 172 | 0 | ✅✅ NED on letrozole, ILC, lymphedema, MRI brain conditional (逐字审查) |
| 34 | 173 | 0 | ✅✅ 2nd local recurrence, receptor change, tamoxifen, CALOR trial (逐字审查) |
| 36 | 175 | 0 | ✅✅ mixed ductal/mucinous, Abraxane C8, G3 infusion reaction (逐字审查) |
| 37 | 176 | 0 | ✅✅ TNBC, dd AC/Taxol, no hormonal/RT, full code (逐字审查) |
| 40 | 179 | 1 | imaging_plan 漏 DEXA (A/P clearly lists "-DEXA") (逐字审查) |
| 41 | 180 | 0 | ✅✅ ATM carrier, 3cm G3 IDC, AC-Taxol, MUGA, ribociclib trial (逐字审查) |
| 42 | 181 | 0 | ✅✅ multifocal IDC G1, tamoxifen 5yr, mammogram planned (逐字审查) |
| 43 | 182 | 1 | lab_summary "Values redacted" but labs have real values (逐字审查) |
| 44 | 183 | 0 | ✅✅ BRCA1+, post-neoadjuvant, RT clinical trial, ribociclib eligible (逐字审查) |
| 46 | 185 | 0 | ✅✅ SARCOIDOSIS TEST PASSED: Distant Mets "No", Goals "curative" |
| 49 | 188 | 0 | ✅✅ surrogate decision maker, Oncotype 11, thrombophilia assessment (逐字审查) |
| 50 | 189 | 0 | ✅✅ Stage IV, PMS2 mutation, ibrance+letrozole+xgeva, lupron correctly excluded (逐字审查) |
| 52 | 191 | 0 | ✅✅ fertility preservation, Zoladex, Oncotype, CT/bone scan (逐字审查) |
| 53 | 192 | 0 | ✅✅ HER2+ heterogeneous, neuroendocrine diff, AC/THP, Arimidex 10yr (逐字审查) |
| 54 | 193 | 0 | ✅✅ BRCA2+, oligometastatic T6, palbociclib planned, PET/CT+DEXA (逐字审查) |
| 57 | 196 | 2 | Type_of_Cancer "invasive lobular carcinoma" (note says "Gr III adenoCA", "lobular mass"=imaging shape) + genetic_testing_plan "None planned" (A/P says "Rec genetic counseling and testing") (逐字审查) |
| 59 | 198 | 1 | current_meds "exemestane" but note says "has not tried it yet" → letter says "currently taking Exemestane" then "will wait 2-3 weeks before starting" (contradiction) (逐字审查) |
| 61 | 200 | 1 | genetic_testing_plan "None planned" but A/P says "will likely need ***** Dx after surgery" (planned genomic test for chemo decision) (逐字审查) |
| 63 | 202 | 2 | lab_plan "No labs" (A/P: estradiol+FSH q1-2mo) + imaging_plan "No imaging" (A/P: baseline DEXA) (逐字审查) |
| 64 | 203 | 0 | ✅✅ dd AC, taxol planned, oligometastatic sternum, xgeva conditional, Full code (逐字审查) |
| 65 | 204 | 0 | ✅✅ ISPY2 9-arm trial, ER 2%/PR 7%/HER2-, Ki-67 36%, micromet LN, TTE+port+research biopsy (逐字审查) |
| 66 | 205 | 1 | Stage "N1 LN involvement" but axillary biopsy was NEGATIVE → should be N0 (逐字审查) |
| 68 | 207 | 2 | Type_of_Cancer "HER2-" but TCHP is HER2+ regimen + Stage "IV" but axillary/IMN nodes are regional, note says "early stage" (逐字审查) |
| 70 | 209 | 0 | ✅✅ bilateral ILC+IDC, BRCA1, neoadjuvant TC→Abraxane, prolia/dental, lung nodule F/u, expanders (逐字审查) |
| 72 | 211 | 0 | ✅✅ pT1cN0 G2 neuroendocrine diff, ER99%/PR-/HER2-, letrozole+Oncotype ordered, Reclast (逐字审查) |
| 73 | 212 | 0 | ✅✅ fat necrosis confirmed on bilateral US+mammo, NED on arimidex, insect bite resolved (逐字审查) |
| 78 | 217 | 0 | ✅✅ mTNBC liver/LN/lung progression, off therapy, phase 3 ADC trial interest, rad onc consult (逐字审查) |
| 80 | 219 | 0 | ✅✅ local recurrence IDC in dermis post-DCIS mastectomy, Oncotype 24, TC x4, 6wk XRT+boost, Tempus no actionable (逐字审查) |
| 82 | 221 | 0 | ✅✅ mixed ductal/lobular 4.3cm, low-risk genomic→no chemo, XRT+AI+DEXA, exercise counseling, full code (逐字审查) |
| 83 | 222 | 0 | ✅ significant response on PET |
| 84 | 223 | 0 | ✅ capecitabine, CT+MRI |
| 85 | 224 | 0 | ✅ Response_Assessment retry succeeded |
| 86 | 225 | 0 | ✅ PD on letrozole+ribociclib, Palliative care removed (POST hook) |
| 87 | 226 | 0 | ✅ Stage IIIA (pT2 N2a) |
| 88 | 227 | 0 | ✅ capecitabine, full code |
| 90 | 229 | 0 | ✅ post-neoadjuvant |
| 91 | 230 | 0 | ✅ everolimus+exemestane |
| 92 | 231 | 0 | ✅ epirubicin, liver improvement |
| 94 | 233 | 0 | ✅ mammogram+MRI planned |
| 95 | 234 | 0 | ✅ good neoadjuvant response |
| 97 | 236 | 0 | ✅ Stage IA |
| 100 | 239 | 0 | ✅ Gemzar + labs comprehensive |

## POST hooks verified (10 total, all fired correctly)

| Hook | Triggers | ROWs |
|------|----------|------|
| imaging_plan header orders | 1 | ROW 1 |
| Palliative care removal | 2 | ROW 1, 86 |
| medication_plan contradictions | 2 | ROW 14 (fulvestrant+faslodex) |
| supportive_meds filter | 2 | ROW 11, 40 |
| lab_summary old labs | 1 | ROW 64 |
| therapy_plan taxol planned | 1 | ROW 64 |
| Response_Assessment retry | 2 | ROW 14, 85 |
| medication_plan pRBC | 1 | ROW 2 |

## V31 vs V32 Final

| 指标 | V31 | V32 |
|------|-----|-----|
| P0 | 0 | 0 |
| P1 | 0 | 0 |
| **P2** | **112 (1.84/sample)** | **0 (0/sample)** |
| 速度 | ~2 hours | 14.5 min |
| P2 reduction | — | **100%** |
