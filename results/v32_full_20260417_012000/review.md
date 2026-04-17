# V32 Final Manual Review (61 samples, iteration 8)

> Run: v32_full_20260417_012000
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (10 POST hooks)
> Results: results/v32_full_20260417_012000_results.txt
> Automated P2 scan: 0 P2 (all known patterns checked)
> Status: **审查完成 — 61/61**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 1 | 0.016/sample |

## 审查记录

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅ imaging+lab POST hook, Integrative Medicine, Full code |
| 2 | 141 | 0 | ✅ pRBC included! TNBC+Lynch, comprehensive |
| 3 | 142 | 0 | ✅ Stage IIA, second opinion, HER2-(2+ FISH neg), full code |
| 5 | 144 | 0 | ✅ Stage III→IV, micropapillary, mixed response, Rad Onc |
| 6 | 145 | 0 | ✅ Genetics "None" (fixed), letrozole+zoladex |
| 7 | 146 | 1 | therapy_plan guesses "T-DM1" for redacted drug |
| 8 | 147 | 0 | ✅ pT0N1 correct, pCR + nodal |
| 9 | 148 | 0 | ✅ "status post...completed" (fixed!) |
| 10 | 149 | 0 | ✅ HR+ implied HER2- |
| 11 | 150 | 0 | ✅ supportive_meds clean, Faslodex+Denosumab |
| 12 | 151 | 0 | ✅ DNR/DNI, brain mets GK |
| 14 | 153 | 0 | ✅ medication_plan no contradictions, Response_Assessment fixed |
| 17 | 156 | 0 | ✅ Stage IA, DXA+labs, genetics+nutrition referral |
| 18 | 157 | 0 | ✅ papillary CA + ITC |
| 20 | 159 | 0 | ✅ letrozole+palbociclib, Foundation One |
| 22 | 161 | 0 | ✅ pneumonitis, conditional PET plan |
| 27 | 166 | 0 | ✅ stable disease |
| 29 | 168 | 0 | ✅ multifocal + Oncotype Low Risk |
| 30 | 169 | 0 | ✅ neoadjuvant planned, TTE |
| 33 | 172 | 0 | ✅ NED on letrozole |
| 34 | 173 | 0 | ✅ local recurrence |
| 36 | 175 | 0 | ✅ |
| 37 | 176 | 0 | ✅ Stage IIA, full code |
| 40 | 179 | 0 | ✅ supportive_meds clean (POST hook) |
| 41 | 180 | 0 | ✅ |
| 42 | 181 | 0 | ✅ mammogram planned |
| 43 | 182 | 0 | ✅ Stage I, full code |
| 44 | 183 | 0 | ✅ post-neoadjuvant |
| 46 | 185 | 0 | ✅✅ SARCOIDOSIS TEST PASSED |
| 49 | 188 | 0 | ✅ surrogate decision maker |
| 50 | 189 | 0 | ✅ ibrance+letrozole+xgeva |
| 52 | 191 | 0 | ✅ staging scans |
| 53 | 192 | 0 | ✅ |
| 54 | 193 | 0 | ✅ |
| 57 | 196 | 0 | ✅ post-neoadjuvant TCH+P |
| 59 | 198 | 0 | ✅ NED, mammogram+MRI |
| 61 | 200 | 0 | ✅ Stage I |
| 63 | 202 | 0 | ✅ disappointing chemo response |
| 64 | 203 | 0 | ✅ dd AC, taxol planned (POST hook), Full code |
| 65 | 204 | 0 | ✅ ISPY trial |
| 66 | 205 | 0 | ✅ metaplastic CA |
| 68 | 207 | 0 | ✅ NED after TCHP |
| 70 | 209 | 0 | ✅ bilateral, post-neoadjuvant |
| 72 | 211 | 0 | ✅ Stage IA |
| 73 | 212 | 0 | ✅ fat necrosis |
| 78 | 217 | 0 | ✅ disease progression |
| 80 | 219 | 0 | ✅ Stage IA |
| 82 | 221 | 0 | ✅ Stage IB (prognostic) |
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

---

### ROW 1 (coral_idx 140) — 0 P1, 0 P2 ✅
- 56yo female。2013 Stage IIA R breast ca (2.3+2.4cm, G2, ER+/PR+/HER2-, 0 SLN)。Refused tamoxifen。Widely metastatic 2019 (lungs+liver+peritoneum+ovary+axilla)。Full code。ECOG 0。
- ✅ 全字段准确。POST hooks: imaging header orders + no palliative care FP
- ✅ Letter 出色: "belly lining" + appendix + "unspecified agent" (not guessing) + all plans + emotional support

### ROW 2 (coral_idx 141) — 0 P1, 0 P2 ✅
- 44yo TNBC + Lynch syndrome + colon ca + endometrial ca。Irinotecan C3D1。Chest wall cellulitis, worsening back pain, Na 124, K 3.1, Hgb 7.7。
- ✅ **pRBC 已修复**: medication_plan 包含 "1 unit pRBC today" ✅, Letter 说 "received a blood transfusion" ✅
- ✅ Response: "Disease progression on PET/CT... worsening back pain + chest wall" ✅
- ✅ Labs 全面: CMP+CBC+HBV 逐值匹配 ✅
- ✅ All plans: irinotecan dose change + doxycycline + effexor increase + NS/K+ + scans 3mo + MRI brain + HBV q4mo + Rad Onc + Home health + F/u 2wk ✅

