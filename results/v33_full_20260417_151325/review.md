# V33 Manual Review (61 samples)

> Run: v33_full_20260417_151325
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (20 POST hooks)
> Automated P2 scan: 0 P2 (17/17 V32 P2s verified fixed, 0 regressions)
> Status: **审查中 — 5/61 逐字审查完成，P2=0。下一个：ROW 7（line 446）**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 3 | — |

## 审查记录

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅✅ mBC lungs/liver/peritoneum/ovary (no brain!), ibrance+unspecified conditional, bone scan+MRI brain+6 labs, Integrative Med, full code (逐字审查) |
| 2 | 141 | 0 | ✅✅ mTNBC+Lynch, irinotecan dose change 150mg/m2 q2w, pRBC+NS+KCl, Na124/K3.1/Hgb7.7, doxycycline, HBV q4mo, 30+labs (逐字审查) |
| 3 | 142 | 0 | ✅✅ Stage IIA IDC 1.7cm node+, HR+/HER2-(FISH neg), Ki-67 30-35%, 2nd opinion, PET+genetics pending, full code (逐字审查) |
| 5 | 144 | 0 | ✅✅ Stage III→IV, DM correctly "bone (sternum)" only, regional LNs in Metastasis not DM, mixed response, anastrozole+palbo+leuprolide (逐字审查) |
| 6 | 145 | 0 | ✅✅ G1 IDC 1.5cm MammaPrint Low, zoladex+letrozole, bipolar 2, Myriad neg, estradiol monthly, comprehensive labs (逐字审查) |
| 7 | 146 | 0 | ✅✅ ER-/PR-/HER2+ MBC, d/c pertuzumab/herceptin/taxotere (LVEF↓+PD), unspecified next line, recheck markers (逐字审查) |
| 8 | 147 | 0 | ✅✅ pT0N1 pCR breast, 3/28 LN+(ECE), ER-/HER2+, AC x4→T-DM1 (from note text), port+echo, ?Kikuchi (逐字审查) |
| 9 | 148 | 0 | ✅✅ V32 P2 FIXED: taxol now "completed", Fosamax+Letrozole after XRT, pT2 N1 5% cellularity (逐字审查) |
| 10 | 149 | 1 | findings "0/20 LN" vs Stage "7/20 LN" 内部不一致 (note "July 20" 歧义, V32 P2 持续) (逐字审查) |
| 11 | 150 | 1 | Response_Assessment extraction error (model produced invalid JSON, retry failed) (逐字审查) |
| 12 | 151 | 0 | ✅✅ Stage IV de novo ER+/PR+/HER2+, brain/lung/bone/nodes, herceptin+letrozole, DNR/DNI, 4 imaging plans (逐字审查) |
| 14 | 153 | 0 | ✅✅ Stage IV mBC ER+ bone/nodes, faslodex+palbociclib, CT+MRI spine, labs q2w (逐字审查) |
| 17 | 156 | 0 | ✅✅ Stage IA, DXA+labs+genetics referral, adjuvant endocrine planned (逐字审查) |
| 18 | 157 | 0 | ✅✅ Stage IA pT1b N0, DEXA ordered, endocrine 5-10yr, papillary+ITC (逐字审查) |
| 20 | 159 | 0 | ✅✅ Stage I→IV bone+lung, Foundation One ordered, letrozole+palbociclib, monthly labs (逐字审查) |
| 22 | 161 | 1 | Response_Assessment extraction error (逐字审查) |
| 27 | 166 | 0 | ✅✅ V32 P2 FIXED: Distant Met "bone" only (liver cysts removed by POST hook), stable osseous mets (逐字审查) |
| 29 | 168 | 0 | ✅✅ pT1c N1mi, Oncotype redacted detected by POST-GENETICS, DEXA planned, letrozole (逐字审查) |
| 30 | 169 | 0 | ✅✅ ER-/PR-/HER2+ IDC, Stage IIIA, neoadjuvant planned (逐字审查) |
| 33 | 172 | 0 | ✅✅ ILC, NED on letrozole, Stage IIB (逐字审查) |
| 34 | 173 | 0 | ✅✅ 2nd local recurrence, receptor change, Stage III (逐字审查) |
| 36 | 175 | 0 | ✅✅ mixed ductal/mucinous G3, Stage IIIA (逐字审查) |
| 37 | 176 | 0 | ✅✅ TNBC, dd AC/Taxol, Stage IIA (逐字审查) |
| 40 | 179 | 0 | ✅✅ V32 P2 FIXED: imaging_plan now has DEXA (POST-IMAGING) (逐字审查) |
| 41 | 180 | 0 | ✅✅ ATM carrier, AC-Taxol, MUGA (逐字审查) |
| 42 | 181 | 0 | ✅✅ multifocal IDC G1, tamoxifen (逐字审查) |
| 43 | 182 | 0 | ✅✅ V32 P2 FIXED: lab_summary no longer "Values redacted" (POST-LAB-REDACTED) (逐字审查) |
| 44 | 183 | 0 | ✅✅ BRCA1+, post-neoadjuvant, RT trial (逐字审查) |
| 46 | 185 | 0 | ✅✅ sarcoidosis test passed, DM=No (逐字审查) |
| 49 | 188 | 0 | ✅✅ surrogate decision maker, Oncotype 11 (逐字审查) |
| 50 | 189 | 0 | ✅✅ Stage IV, PMS2, ibrance+letrozole+xgeva (逐字审查) |
| 52 | 191 | 0 | ✅✅ fertility preservation, Zoladex (逐字审查) |
| 53 | 192 | 0 | ✅✅ HER2+ heterogeneous, AC/THP (逐字审查) |
| 54 | 193 | 0 | ✅✅ BRCA2+, oligometastatic T6 (逐字审查) |
| 57 | 196 | 0 | ✅✅ V32 P2 FIXED: "Grade III adenoCA" (not lobular), genetic counseling captured (逐字审查) |
| 59 | 198 | 0 | ✅✅ V32 P2 FIXED: exemestane removed from current_meds (POST-MEDS-NOT-STARTED) (逐字审查) |
| 61 | 200 | 0 | ✅✅ V32 P2 FIXED: genomic test (redacted) detected (POST-GENETICS-SEARCH) (逐字审查) |
| 63 | 202 | 0 | ✅✅ V32 P2 FIXED: lab_plan Estradiol+FSH, imaging_plan DEXA (POST hooks) (逐字审查) |
| 64 | 203 | 0 | ✅✅ Stage IV sternum, dd AC, taxol planned, xgeva conditional (逐字审查) |
| 65 | 204 | 0 | ✅✅ ISPY2, ER 2%/PR 7%, micromet LN (逐字审查) |
| 66 | 205 | 0 | ✅✅ V32 P2 FIXED: Stage IIIA (not IV), N1 regional, metaplastic CA (逐字审查) |
| 68 | 207 | 0 | ✅✅ V32 P2 FIXED: HER2+ (POST-HER2-VERIFY), Stage IIIA (not IV) (逐字审查) |
| 70 | 209 | 0 | ✅✅ bilateral ILC+IDC, BRCA1, prolia/dental (逐字审查) |
| 72 | 211 | 0 | ✅✅ pT1cN0 neuroendocrine diff, Oncotype ordered (逐字审查) |
| 73 | 212 | 0 | ✅✅ fat necrosis, NED on arimidex (逐字审查) |
| 78 | 217 | 0 | ✅✅ mTNBC progression, phase 3 ADC trial (逐字审查) |
| 80 | 219 | 0 | ✅✅ local recurrence dermis, Oncotype 24, TC x4 (逐字审查) |
| 82 | 221 | 0 | ✅✅ mixed ductal/lobular, low-risk genomic, no chemo (逐字审查) |
| 83 | 222 | 0 | ✅✅ V32 P2 FIXED: Stage III (not IV), POST-STAGE-DISTMET (逐字审查) |
| 84 | 223 | 0 | ✅✅ mBC bone/liver/CNS, CHEK2+MS, capecitabine (逐字审查) |
| 85 | 224 | 0 | ✅✅ mBC ILC, Foundation One, phase 1 trial+olaparib (逐字审查) |
| 86 | 225 | 0 | ✅✅ V32 P2 FIXED: HER2- preserved (met biopsy), no override (逐字审查) |
| 87 | 226 | 0 | ✅✅ 2nd opinion, 4/19 LN+ECE, Parkinson, hormonal alone (逐字审查) |
| 88 | 227 | 0 | ✅✅ Stage IV brain/lung, HR weak, capecitabine, HER2 retest (逐字审查) |
| 90 | 229 | 0 | ✅✅ trial enrolled, AC dose delay, antiemetic switch (逐字审查) |
| 91 | 230 | 0 | ✅✅ Stage IV MBC, everolimus+exemestane+denosumab (逐字审查) |
| 92 | 231 | 0 | ✅✅ MBC liver improving on epirubicin (逐字审查) |
| 94 | 233 | 0 | ✅✅ pT1b Oncotype 21, NED on letrozole (逐字审查) |
| 95 | 234 | 0 | ✅✅ ISPY pembrolizumab, good NAC response (逐字审查) |
| 97 | 236 | 0 | ✅✅ V32 P2 FIXED: Oncotype Dx captured (POST-GENETICS-SEARCH) (逐字审查) |
| 100 | 239 | 0 | ✅✅ MBC Gemzar, tumor markers rising, unclear PD vs flare (逐字审查) |
