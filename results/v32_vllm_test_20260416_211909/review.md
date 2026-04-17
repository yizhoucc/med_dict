# V32 vLLM Test Review (8 samples)

> Run: v32_vllm_test_20260416_211909
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (独立 pipeline)
> Reviewer: Claude (逐字逐句手工审查)
> Status: **审查完成 — 全部 8/8 samples 已审查**
> Results 文件: `results/v32_vllm_test_20260416_211909_results.txt`（本地已下载）

### 系统性问题（全 8 个 sample 共有）
- **`<think>` 标签污染**：所有 assessment_and_plan 和 letter 输出都包含 `<think>\n\n</think>` 前缀（Qwen3.5 thinking mode 默认开启）

## 汇总统计

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 0 | 0% | |
| **P2** | 10 | 1.25/sample | ROW 1×5, 2×1, 8×1, 17×0, 29×0, 46×0, 64×1, 100×2 |

---

### ROW 1 (coral_idx 140) — 0 P1, 5 P2
- 56yo female。2013 Stage IIA R breast ca (2.3+2.4cm, G2, ER+/PR+/HER2-, 0 SLN)。S/p R mastectomy+SLN+implant。**Refused tamoxifen, no chemo/RT/imaging for 6 years**。2019/12 ER visit: CT shows widely metastatic — lungs+liver+peritoneum+ovaries+R axillary recurrence (1.8cm near implant)。New patient to establish care。Very anxious, asks if curable。Wants natural therapies。Husband supportive。ECOG 0。Full code。PE: R axillary mass 3cm, hepatomegaly+omental masses。A/P: needs tissue biopsy (Dr [redacted] Thursday) + staging workup (MRI brain+bone scan+labs) + if HR+/HER2- → ibrance+[redacted] + integrative medicine referral + RTC after workup。
- P2: **imaging_plan "No imaging planned"** — A/P 明确 ordered MRI brain + bone scan
- P2: **lab_plan "No labs planned"** — A/P 明确 ordered CBC, CMP, CA 15-3, CEA, PT/APTT
- P2: **Advance care "Not discussed"** — Note 写了 "Full code"
- P2: **Referral 全 "None"** — A/P 明确 "[redacted] center Referral asap" (integrative medicine)
- P2: **Letter 误称 "No imaging/labs planned"** — 继承 extraction 错误
- ✅ Type 出色: "current status inferred as ER+/PR+/HER2- based on plan to treat with Ibrance" ✅
- ✅ Stage: IIA→IV ✅。Mets: lung+liver+peritoneum+ovary+axilla ✅。Goals: palliative ✅
- ✅ findings 极其全面（CT 每条+PE）✅。procedure_plan: biopsy ✅
- ✅ **Letter 内容好**（除 imaging/labs 错误外）: "belly lining" = peritoneum + "help you feel better and live longer" + source tags 正确

### ROW 2 (coral_idx 141) — 0 P1, 1 P2
- 44yo female。**极其复杂**: Lynch syndrome (MSH2 mutation) + 3 cancers: TNBC (2013, grade 3, metastatic 2019 to liver+bone+skull base) + colon cancer (2018, Stage I) + endometrial ca (2018, FIGO 1)。On irinotecan cycle 3 D1 (missed D8 q cycle due to transaminitis/diarrhea)。Presenting: chest wall cellulitis, worsening back pain (bed-bound), confusion, fever 103 (resolved), anxiety/depression worse, Na 124(LL), K 3.1(L), Hgb 7.7(L), ALP 183(H)。ECOG 1。Tachycardia 136。DVT hx (rivaroxaban)。A/P: doxycycline + change irinotecan q2w 150mg/m2 + 1U pRBC + urgent Rad Onc + HBV monitoring q4mo + increase effexor 75mg + MRI brain if worse + NS IV+K+ + home health + F/U 2wk。
- P2: **Response_Assessment 提取失败** — `{"status":"error","message":"..."}` — `<think>` 标签污染导致 JSON 修复出错
- ✅ Type: TNBC ✅。Stage: IIB→IV ✅。Mets: liver+bone+skull base ✅（极其精确）
- ✅ lab_summary 极其全面: Na 124, K 3.1, Hgb 7.7, Albumin 2.1, ALP 183, HBV labs ✅
- ✅ findings 极其全面: 所有 PET/CT+MRI+CT+US+PE with SUV 值 ✅
- ✅ current_meds: irinotecan+morphine+flexeril+oxycodone+effexor+ativan ✅
- ✅ medication_plan 极其全面: 10+ changes all captured ✅
- ✅ imaging_plan: scans 3mo + MRI brain if worse + correctly labeled past imaging ✅
- ✅ lab_plan: HBV monitoring q4mo ✅。Referral: Rad Onc + Home health ✅
- ✅ **Letter 极其出色**: 每个治疗变化有原因 + "low levels of salt" + "cellulitis, which is a skin infection" + source tags 正确 + 通俗完整

### ROW 8 (coral_idx 147) — 0 P1, 1 P2
- 29yo premenopausal female。Clinical stage III ER-/PR-/HER2+(IHC 3+) L breast IDC。Non-adherent: incomplete TCHP (3 partial cycles), multiple provider changes, family opposed chemo。S/p L lumpectomy+ALND (08/26/19): **breast pCR (0% cellularity)** but 3/28 LN+ (2 macro+1 micro, largest 2.4cm, ENE+)。LN receptors: ER-/PR-/HER2+(IHC 3+, FISH 5.7), Ki-67 75%。Necrotizing lymphadenitis (Kikuchi disease)。PET/CTs negative for distant mets。Zoom visit。ECOG 0。3 young boys。A/P: adjuvant AC×4→T-DM1, radiation after AC, port+Echo needed, patient considering。
- P2: **Stage "pT0N2 (Stage IIIA)"** — ypT0 with 3/28 LN+ = ypN1a (1-3 nodes)。ypT0 N1a = Stage **IIA**, not IIIA (N2 requires 4-9 nodes)
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment 出色: "no residual invasive carcinoma (0% cellularity), but 3/28 LN+" ✅
- ✅ findings 极其全面: pathology details + imaging + PE + necrotizing lymphadenitis ✅
- ✅ medication_plan: AC×4→T-DM1 ✅。procedure_plan: port ✅。imaging_plan: Echo ✅
- ✅ **Letter 出色**: pCR "no cancer in breast tissue, good sign" + "cancer in lymph nodes under arm" + AC+T-DM1+RT + port "help with treatments" + echo "check heart" + source tags ✅

### ROW 17 (coral_idx 156) — 0 P1, 0 P2 ✅
- 53yo female (Televisit)。L breast IDC 0.8cm, grade 2, ER+(>95%)/PR+(>95%)/HER2-(IHC 0), Ki-67 5%。S/p L lumpectomy+SLN: margins neg, no DCIS, 0/5 LN。Menopausal status uncertain (s/p hysterectomy)。Family: sister ovarian ca @40s, paternal aunt breast ca @60s。A/P: 10-15% recurrence risk。Adjuvant hormonal ≥5yr (tamoxifen vs AI → check hormone levels)。Breast RT (may postpone if genetics+→prophylactic mastectomy)。Genetics referral (family hx)。DXA。Check labs+hormones。Nutritionist。F/U after RT。
- ✅ **全字段准确**: Stage IA (0.8cm, 0/5 LN) ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan 全面: hormonal ≥5yr + tamoxifen vs AI + menopausal status + side effects ✅
- ✅ radiotherapy_plan 出色: RT + "positive genetic test may lead to prophylactic mastectomy, thus RT postponed" ✅
- ✅ genetic_testing_plan: genetics referral + prophylactic mastectomy implication ✅
- ✅ imaging_plan: DXA ✅。lab_plan: labs+hormones ✅。Referral: Nutrition+Genetics+Specialty ✅
- ✅ **Letter 极其出色**: IDC "milk ducts" + Stage IA "small, not spread" + hormonal therapy "prevent coming back" + "Tamoxifen or AI depending on menopausal status" + "check hormone levels" + side effects explained + RT + genetics "removing other breast" + nutritionist + DXA "bone density" + all source tags correct

### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅
- 59yo postmenopausal female。Multifocal R breast IDC with micropapillary features, grade 2。1.6cm+0.6cm (positive margin for 0.6cm)。SLN micromet 0.5mm。pT1c(m)N1mi(sn)。ER+(>90%)/PR+(30%)/HER2-。Oncotype Low Risk (+0.046)。No chemo recommended。Works abroad。A/P: start letrozole 2.5mg daily + Ca 1200mg + vit D + DEXA September + re-excision September + RT locally + vaginal moisturizer。
- ✅ 全字段准确：Stage I (pT1c, N1mi) ✅。Goals: curative ✅。Mets: No ✅
- ✅ medication_plan 极其全面：letrozole + Ca + vit D + moisturizer + bisphosphonate if osteopenia + tamoxifen fallback ✅
- ✅ procedure_plan: re-excision for positive margin ✅。imaging_plan: DEXA ✅
- ✅ radiotherapy_plan: post-lumpectomy RT + "may pursue closer to home" ✅
- ✅ **Letter 极其出色**: two tumor foci + seroma explained + letrozole "prevent coming back" + Ca+vit D + bisphosphonate "if weak bones" + tamoxifen fallback + re-excision "remove remaining non-invasive cancer" + RT + DEXA "bone density" + all source tags correct

### ROW 46 (coral_idx 185, sarcoidosis case) — 0 P1, 0 P2 ✅
- 48yo postmenopausal female (s/p TAH-BSO)。R breast IDC, ER+(95%)/PR-(0%)/HER2-(1+), Ki-67<5%。S/p neoadjuvant Taxol+[redacted]→R partial mastectomy+SLN: 3.5cm residual (ypT2), POSITIVE margins, 2/2 SLN macro (6mm, ENE>2mm)。Sarcoidosis (bronchoscopy FNA: non-necrotizing granulomatous inflammation)。Renal artery aneurysm。Neuropathy, anemia, joint pain。A/P: re-excision + start letrozole + abemaciclib after XRT + DEXA + MRA Jan 2022 + sarcoid blood tests + Rad Onc + iron repeat 3-4mo + F/U 2-3mo。
- ✅ **关键 sarcoidosis 测试**: Metastasis "No" ✅✅, Distant "No" ✅✅, Goals "curative" ✅ — 媒纵隔/肺门淋巴结病变正确识别为 sarcoidosis 非 distant mets!
- ✅ Stage: IIB (ypT2 N1) ✅（2/2 SLN = N1a，合理）
- ✅ response_assessment 出色: neoadjuvant response + residual disease + imaging interval decrease ✅
- ✅ imaging_plan: DEXA + MRA abdomen Jan 2022 ✅。lab_plan: sarcoid blood tests + iron repeat ✅
- ✅ **Letter 极其出色**: "cancer responded well but some cells still found" + "edges still have cancer, more surgery needed" + "lymph nodes have cancer" + aneurysm "bulge in blood vessel" + letrozole + naproxen/APAP/allegra/iron + abemaciclib after RT + re-excision + DEXA + MRA + sarcoid blood tests + iron repeat + all source tags correct

### ROW 64 (coral_idx 203) — 0 P1, 1 P2
- 28yo premenopausal female。L breast IDC ER+/PR+/HER2-。MRI: 10.3×4.5×3.5cm。Axillary biopsy positive。Bone scan: suspicious lesion manubrium (solitary bone met?)。Currently on dd AC, tolerating okay。Video visit (Zoom) second opinion。Cousin with her。Full code。ECOG 0。Situational anxiety。A/P: Stage III-IV, oligo-metastatic approach (aggressive treatment), biopsy sternal lesion, AC→taxol→surgery→RT→treat sternum, add xgeva if bone biopsy positive, hormonal blockade discussed, keep chemo on time with current oncologist。
- P2: **Advance care "Not discussed"** — Note 写了 "Full code" under Advance Care Planning
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV (metastatic to sternum) ✅。Goals: palliative ✅（Stage IV per decision tree，尽管医生说 "treat aggressively"）
- ✅ second opinion: "yes" ✅。Televisit ✅。current_meds: "dd AC" ✅
- ✅ findings 全面: MRI tumor + axillary biopsy + bone scan manubrium + PE ✅
- ✅ medication_plan 出色: current [redacted] + Taxol planned + Xgeva if bone positive + hormonal blockade ✅
- ✅ radiotherapy_plan: RT for primary + sternum ✅。procedure_plan: biopsy + surgery ✅
- ✅ **Letter 出色**: "second opinion" + IDC explained + oligo-met approach + biopsy sternum + surgery + RT + Xgeva + "keep chemo on schedule" + source tags correct

### ROW 100 (coral_idx 239) — 0 P1, 2 P2
- 68yo female。Metastatic breast cancer (originally 2002 ER+(80%)/PR+(50%)/HER2-, later recurrence ER/PR negative)。Complex history: lumpectomy+AC+taxol+XRT+arimidex→metastatic (liver+bone)→abraxane+bevacizumab→PD→faslodex→xeloda→now Gemzar。On Gemzar Cycle 2 Day 8, cancelled by patient。Fatigue, rising tumor markers, scan too early to assess。ECOG 1。A/P: unclear if progressing or tumor flare, rec exercise 10min 3x/day, Focalin prn, continue treatment, discuss break with Dr [redacted]。
- P2: **lab_summary "Values redacted"** — Labs are NOT redacted! CA 15-3: 118-119(H), CA 27.29: 178-181(H), CEA: 312-320(H), ALP: 172-196(H), AST: 49-63(H), Hgb: 9.6-9.9(L), MCV: 104(H)。Lab 值被放入 findings 而非 lab_summary
- P2: **current_meds empty** — Gemzar (gemcitabine) is the active oncologic treatment, should be in current_meds
- ✅ Type: ER+/PR+/HER2- IDC ✅（based on original diagnosis, faslodex use supports ER+ working dx）
- ✅ Stage: Metastatic (Stage IV) ✅。Goals: palliative ✅
- ✅ response_assessment 极其出色: "Unclear if progressing or tumor flare, tumor markers rose, scan too early" — 完美匹配 A/P ✅
- ✅ findings 内容准确（PE + lab abnormalities），只是 lab 值放错了字段
- ✅ medication_plan: Focalin prn + continue treatment + discuss break ✅
- ✅ **Letter 出色**: "markers for cancer are higher" + anemia explained "not enough healthy red blood cells" + "not sure if getting worse or temporary reaction" + Focalin for energy + discuss break + source tags correct

---

## 完整统计

| ROW | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|-----|---------|
| 1 | 140 | 0 | 0 | 5 | imaging/lab/referral/advance care 全 missed, letter 继承错误 |
| 2 | 141 | 0 | 0 | 1 | Response_Assessment JSON 修复失败（`<think>` 标签污染） |
| 8 | 147 | 0 | 0 | 1 | Stage ypT0N1a=IIA 误判为 IIIA |
| 17 | 156 | 0 | 0 | 0 | ✅ 全字段准确 |
| 29 | 168 | 0 | 0 | 0 | ✅ 全字段准确 |
| 46 | 185 | 0 | 0 | 0 | ✅ 关键 sarcoidosis 测试通过 |
| 64 | 203 | 0 | 0 | 1 | Advance care missed "Full code" |
| 100 | 239 | 0 | 0 | 2 | lab_summary "Values redacted" (实际有大量 labs), current_meds 漏 Gemzar |
| **Total** | | **0** | **0** | **10** | **1.25/sample** |

## V31 vs V32 对比

| 指标 | V31 (Qwen2.5-32B-AWQ) | V32 (Qwen3.5-35B-A3B-GPTQ) |
|------|------------------------|----------------------------|
| 模型 | Qwen2.5-32B-Instruct-AWQ | Qwen3.5-35B-A3B-GPTQ-Int4 |
| Pipeline | run.py (HF 直接加载, KV Cache 分叉) | vllm_pipeline (vLLM HTTP API, Prefix Caching) |
| 样本数 | 61 | 8 (test) |
| P0 | 0 (0%) | 0 (0%) |
| P1 | 0 (0%) | 0 (0%) |
| P2 | 112 (1.84/sample) | 10 (1.25/sample) |
| 速度 | ~15-20min/8 samples | 1.9min/8 samples (10x faster) |
| 系统性问题 | 无 | `<think>` 标签污染所有输出 |
| Letter 质量 | 良好 | 出色（通俗化更好） |
| 关键测试 | — | Sarcoidosis 正确识别为非 distant mets ✅ |
