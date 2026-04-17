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
| 30 | 169 | | |
| 33 | 172 | | |
| 34 | 173 | | |
| 36 | 175 | | |
| 37 | 176 | | |
| 40 | 179 | | |
| 41 | 180 | | |
| 42 | 181 | | |
| 43 | 182 | | |
| 44 | 183 | | |
| 46 | 185 | | |
| 49 | 188 | | |
| 50 | 189 | | |
| 52 | 191 | | |
| 53 | 192 | | |
| 54 | 193 | | |
| 57 | 196 | | |
| 59 | 198 | | |
| 61 | 200 | | |
| 63 | 202 | | |
| 64 | 203 | | |
| 65 | 204 | | |
| 66 | 205 | | |
| 68 | 207 | | |
| 70 | 209 | | |
| 72 | 211 | | |
| 73 | 212 | | |
| 78 | 217 | | |
| 80 | 219 | | |
| 82 | 221 | | |
| 83 | 222 | | |
| 84 | 223 | | |
| 85 | 224 | | |
| 86 | 225 | | |
| 87 | 226 | | |
| 88 | 227 | | |
| 90 | 229 | | |
| 91 | 230 | | |
| 92 | 231 | | |
| 94 | 233 | | |
| 95 | 234 | | |
| 97 | 236 | | |
| 100 | 239 | | |
