# V32 Final Review (8 samples, iteration 8)

> Run: v32_vllm_test_20260416_222902
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (独立 pipeline, POST hooks)
> Reviewer: Claude (逐字逐句手工审查)
> Status: **审查完成 — 全部 8/8 samples 已审查**

## 汇总统计

| 严重度 | 数量 | 比率 |
|--------|------|------|
| **P0** | 0 | 0% |
| **P1** | 0 | 0% |
| **P2** | 1 | 0.125/sample | ROW 2: medication_plan 漏 pRBC transfusion |

---

### ROW 1 (coral_idx 140) — 0 P1, 0 P2 ✅
- 56yo female。2013 Stage IIA R breast ca (2.3+2.4cm, G2, ER+/PR+/HER2-, 0 SLN)。S/p R mastectomy+SLN+implant。Refused tamoxifen, no chemo/RT/imaging for 6 years。2019/12 ER visit: CT widely metastatic — lungs+liver+peritoneum+ovaries+R axillary recurrence (1.8cm)。New patient to establish care。Very anxious, asks if curable。Wants natural therapies。Full code。ECOG 0。
- ✅ Type: ER+/PR+/HER2- (inferred from ibrance plan) ✅。Stage: IIA→IV ✅。Mets: lung+liver+peritoneum+ovary+axilla(regional) ✅
- ✅ findings 极其全面: CT 所有 8 条 + PE (axillary mass 3cm, hepatomegaly, omental masses) ✅
- ✅ current_meds: "" (no meds on file) ✅。goals: palliative (原文 explicit) ✅
- ✅ imaging_plan: "Bone Scan; MR Brain (ordered in note header)" — POST hook 正确提取 header orders ✅
- ✅ lab_plan: CBC, CMP, CA 15-3, CEA, APTT, PT — header orders 完全匹配 ✅
- ✅ Referral: Integrative Medicine ✅ (header + HPI + A/P #6 一致)
- ✅ Advance care: "Code status: Full code." ✅
- ✅ procedure_plan: biopsy R axilla ✅。medication_plan: ibrance+[redacted] if HR+/HER2- ✅
- ✅ **Letter 出色**: "belly lining"=peritoneum ✅ + appendix finding ✅ + "no signs of cancer in your bones" ✅ + palliative explained ✅ + "natural therapies" referral ✅ + bone scan+MRI+blood tests all mentioned ✅ + full code ✅ + source tags all correct ✅

### ROW 2 (coral_idx 141) — 0 P1, 1 P2
- 44yo female。极其复杂：Lynch syndrome (MSH2 mutation) + 3 cancers: TNBC (2013 IIB→IV to liver+bone+skull base) + colon ca (2018 Stage I) + endometrial ca (2018 FIGO 1)。On irinotecan C3D1 (missed D8 every cycle d/t transaminitis/diarrhea)。Chest wall cellulitis, worsening back pain (bed-bound), confusion, fever 103 history, Na 124, K 3.1, Hgb 7.7, ALP 183。ECOG 1。Tachycardia 136。DVT hx on rivaroxaban。
- P2: **medication_plan 漏 "1 unit pRBC today"** — A/P 明确 ordered pRBC for anemia, prompt 要求 "include any blood transfusion plan"
- ✅ Type: ER-/PR-/HER2- TNBC grade 3 ✅。Stage: IIB→IV ✅。Mets: liver(caudate+segVII)+bone(S1+femoral necks)+skull base ✅
- ✅ lab_summary 极其全面: 所有 CMP+CBC+HBV labs 逐值列出，全部核对匹配 ✅
- ✅ findings 极其全面: 2x PET/CT with SUV + MRI spine + CT CAP + pelvic US + PE + vitals + symptoms ✅
- ✅ response_assessment 出色: "Disease progression on PET/CT 05/31/19 + worsening back pain + chest wall" ✅
- ✅ current_meds: irinotecan+morphine+flexeril+oxycodone ✅。medication_plan: 10+ changes captured ✅
- ✅ imaging_plan: scans 3mo + MRI brain if worse ✅。lab_plan: HBV monitoring q4mo ✅
- ✅ Referral: Rad Onc + Home health ✅。F/u 2 weeks ✅
- ✅ **Letter 出色**: chemo调整 + 抗生素 + 疼痛管理 + 电解质 + Rad Onc + home health + scans + HBV monitoring ✅ + source tags correct。唯缺输血信息。

### ROW 8 (coral_idx 147) — 0 P1, 0 P2 ✅
- 29yo premenopausal。Clinical stage II-III ER-/PR-/HER2+(IHC 3+) L breast IDC。Non-adherent: only 3 incomplete TCHP cycles, family opposed chemo。S/p L lumpectomy+ALND (08/26/19): breast pCR (0% cellularity) but 3/28 LN+ (2 macro+1 micro, largest 2.4cm, ENE+)。LN receptors: ER-/PR-/HER2+(IHC 3, FISH 5.7), Ki-67 75%。Necrotizing lymphadenitis (Kikuchi)。PET/CTs negative。Zoom visit。ECOG 0。3 young boys。
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Stage: II-III→pT0N1 (3/28 = N1, not N2!) ✅。Mets: No ✅
- ✅ response_assessment 出色: pCR in breast (0% cellularity) + 3/28 LN+ with ENE — 完美匹配病理 ✅
- ✅ findings 全面: pathology synoptic details + imaging size changes + PET negative + Kikuchi ✅
- ✅ medication_plan: AC x4 → T-DM1 1yr ✅。procedure_plan: port ✅。imaging_plan: Echo ✅
- ✅ radiotherapy_plan: RT after AC ✅。goals: curative ✅
- ✅ **Letter 极其出色**: IDC "milk ducts" ✅ + pCR "no cancer in breast, good sign" ✅ + "3 of 28 lymph nodes" ✅ + ENE "grown outside the lymph node wall" ✅ + Kikuchi explained ✅ + AC side effects ✅ + port "easier to receive treatments" ✅ + Echo "ultrasound of your heart" ✅ + source tags correct ✅

### ROW 17 (coral_idx 156) — 0 P1, 0 P2 ✅
- 53yo female (Televisit)。L breast IDC 0.8cm, G2, ER+(>95%)/PR+(>95%)/HER2-(IHC 0), Ki-67 5%。S/p L lumpectomy+SLN: margins neg, 0/5 LN。Menopausal status uncertain (s/p hysterectomy)。Family hx: sister ovarian ca @40s, paternal aunt breast ca @60s。
- ✅ 全字段准确: Stage IA (0.8cm, 0/5 LN) ✅。Goals: curative ✅
- ✅ medication_plan: hormonal ≥5yr + tamoxifen vs AI + menopausal status decision ✅
- ✅ radiotherapy_plan: RT + "postponed if prophylactic mastectomy" ✅
- ✅ genetic_testing_plan: genetics referral + prophylactic mastectomy implication ✅
- ✅ imaging_plan: DXA ✅。lab_plan: labs+hormones ✅。Referral: Nutrition+Genetics+[redacted] ✅
- ✅ **Letter 出色**: Stage IA "small, not spread" + hormonal "at least five years" + tamoxifen vs AI choice + side effects + RT + "if you decide to remove both breasts" + DXA "bone health" + genetics + nutritionist ✅

### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅
- 59yo postmenopausal female。Multifocal R breast IDC with micropapillary features, G2。1.6cm+0.6cm (positive margin for 0.6cm)。SLN micromet 0.5mm。pT1c(m)N1mi(sn)。ER+(>90%)/PR+(30%)/HER2-。Oncotype Low Risk。No chemo。Works abroad。
- ✅ Stage I (pT1c N1mi) ✅。Goals: curative ✅。Mets: No ✅
- ✅ findings 极其全面: 两灶 (1.6+0.6cm) + margins + DCIS span + LN micromet 0.5mm + LVI + MRI BI-RADS + PE ✅
- ✅ medication_plan: letrozole + Ca + vit D + moisturizer + bisphosphonate if osteopenia + tamoxifen fallback ✅
- ✅ procedure_plan: re-excision for positive margin ✅。imaging_plan: DEXA September ✅
- ✅ radiotherapy_plan: post-lumpectomy RT ✅
- ✅ **Letter 出色**: two foci + Grade 2 "moderate speed" + DCIS explained + seroma "pocket of clear fluid" + "chemo not recommended, low-risk" + letrozole + Ca/vit D + bisphosphonate + tamoxifen fallback + re-excision + RT + DEXA + source tags correct ✅

### ROW 46 (coral_idx 185, sarcoidosis case) — 0 P1, 0 P2 ✅
- 48yo postmenopausal female (s/p TAH-BSO)。R breast IDC, ER+(95%)/PR-(0%)/HER2-(1+), Ki-67<5%。S/p neoadjuvant Taxol+[redacted]→R lumpectomy+SLN: 3.5cm residual (ypT2), POSITIVE margins (multifocal), 2/2 SLN macro (max 6mm, ENE>2mm)。Sarcoidosis (FNA: non-necrotizing granulomatous inflammation)。Renal artery aneurysm。
- ✅ **关键 sarcoidosis 测试**: Distant Metastasis "No" ✅✅✅, Goals "curative" ✅ — 媒纵隔/肺门淋巴结病变正确识别为 sarcoidosis 非 distant mets!
- ✅ Stage IIB (ypT2 N1a) ✅。response_assessment: residual 35mm (10-20% cellularity) + 2/2 nodes + MRI decrease ✅
- ✅ lab_summary: 完整 CMP+CBC + Vit D + CRP + ESR，部分 redacted 值正确标注 ✅
- ✅ imaging_plan: DEXA + MRA abdomen Jan 2022 ✅✅。lab_plan: iron panel + sarcoid blood tests ✅✅
- ✅ medication_plan: letrozole + naproxen + APAP + allegra + tramadol + PO iron ✅
- ✅ procedure_plan: re-excision ✅。radiotherapy_plan: after re-excision ✅。abemaciclib after XRT ✅
- ✅ **Letter 出色**: surgery results + re-excision + letrozole + abemaciclib after RT + DEXA + MRA "kidney artery" + iron panel + sarcoid tests + PT referral + anemia explained ✅

### ROW 64 (coral_idx 203) — 0 P1, 0 P2 ✅
- 28yo premenopausal。L breast IDC ER+/PR+/HER2-。MRI: 10.3×4.5×3.5cm。Axillary biopsy positive。Bone scan: suspicious sternal lesion (manubrium)。On dd AC, tolerating okay。Video visit second opinion。Full code。ECOG 0。
- ✅ second opinion "yes" ✅, Televisit ✅。current_meds: "dd AC" (taxol 正确排除) ✅
- ✅ Stage IV (metastatic to sternum) ✅。lab_summary: "No labs in note" (POST hook 修复 old labs) ✅
- ✅ Advance care: "Full code." ✅。procedure: biopsy sternal lesion + surgery ✅
- ✅ medication_plan: [redacted] + taxol planned + xgeva if bone+ ✅
- ✅ **Letter 出色**: second opinion + IDC + sternum biopsy + dd AC + xgeva conditional + surgery/RT + full code + emotional support ✅

### ROW 100 (coral_idx 239) — 0 P1, 0 P2 ✅
- 68yo female。Metastatic breast cancer (originally ER+/PR+/HER2-)。Complex history: lumpectomy→taxol→XRT→arimidex→metastatic→abraxane+bevacizumab→PD→faslodex→xeloda→Gemzar。On Gemzar C2D8, cancelled by patient due to fatigue。
- ✅ current_meds: "Gemzar" ✅ (cancelled dose ≠ stopped)
- ✅ lab_summary: 完整 — CA 15-3: 119, CA 27.29: 181, CEA: 319.9, ALP: 179, Hgb: 9.9, MCV: 104 等全部匹配最新 labs ✅
- ✅ response_assessment 出色: "Unclear if progressing or tumor flare; scan too early; tumor markers risen" — 完美匹配 A/P ✅
- ✅ medication_plan: Focalin prn + continue treatment ✅
- ✅ **Letter 出色**: fatigue + tumour markers "higher than normal" + anemia "low red blood cells" + "not sure if cancer getting worse or temporary reaction" + Focalin + discuss break ✅

---

## 完整统计

| ROW | coral_idx | P0 | P1 | P2 | 说明 |
|-----|-----------|----|----|-----|------|
| 1 | 140 | 0 | 0 | 0 | ✅ imaging/lab/referral 全部通过 (POST hook) |
| 2 | 141 | 0 | 0 | 1 | medication_plan 漏 1 unit pRBC transfusion |
| 8 | 147 | 0 | 0 | 0 | ✅ Stage N1 correct, pCR + nodal disease |
| 17 | 156 | 0 | 0 | 0 | ✅ 全字段准确 |
| 29 | 168 | 0 | 0 | 0 | ✅ 多灶性 + Oncotype Low Risk |
| 46 | 185 | 0 | 0 | 0 | ✅ Sarcoidosis 关键测试通过！ |
| 64 | 203 | 0 | 0 | 0 | ✅ dd AC (no taxol), Full code, POST hook |
| 100 | 239 | 0 | 0 | 0 | ✅ Gemzar + labs + response assessment |
| **Total** | | **0** | **0** | **1** | **0.125/sample** |

## 与 V31 对比

| 指标 | V31 (Qwen2.5-32B-AWQ) | V32 Final (Qwen3.5-35B-A3B-GPTQ) |
|------|------------------------|----------------------------------|
| P0 | 0 (0%) | 0 (0%) |
| P1 | 0 (0%) | 0 (0%) |
| P2 | 112 (1.84/sample) | 1 (0.125/sample) |
| 速度 | ~15min/8 samples | 2.0min/8 samples (8x faster) |
| Letter 质量 | 良好 | 出色 |
| 关键测试 | — | Sarcoidosis ✅ |

