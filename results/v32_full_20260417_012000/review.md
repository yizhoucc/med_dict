# V32 Final Manual Review (61 samples, iteration 8)

> Run: v32_full_20260417_012000
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (10 POST hooks)
> Results: results/v32_full_20260417_012000_results.txt
> Automated P2 scan: 0 P2 (all known patterns checked)
> Status: **手工审查中 — 2/61 完成**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 0 | 0/sample |

## 审查记录

| ROW | coral_idx | P2 | 备注 |
|-----|-----------|-----|------|
| 1 | 140 | 0 | ✅ imaging+lab POST hook, Integrative Medicine, Full code |
| 2 | 141 | 0 | ✅ pRBC included! TNBC+Lynch, comprehensive |

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

