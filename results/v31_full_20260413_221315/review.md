# V31 Full Run Review (61 samples)

> Run: v31_full_20260413_221315
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v31) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **✅ 全部 61/61 审查完成！**
> Results 文件: `results/v31_full_20260413_221315/results.txt`

### v31 改进（相对 v30）
1. POST-PROCEDURE-FILTER: 扩展黑名单 (~40 新药物/影像/RT 关键词)
2. POST-RESPONSE-PRETREATMENT: 新 hook 纠正 pre-treatment consultation 的 "On treatment"
3. Letter [REDACTED] handling: facility/anemia context rules
4. Stage 推断: T/N→Stage 映射表
5. Sarcoidosis/P1 预防: metastasis 字段加入活检确认规则

### POST hook 触发统计
- POST-PROCEDURE-FILTER: 触发 9 次
- POST-RESPONSE-PRETREATMENT: 触发 5 次

### 全量 ROW 列表（61 个）
ROW: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 20, 22, 27, 29, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 46, 49, 50, 52, 53, 54, 57, 59, 61, 63, 64, 65, 66, 68, 70, 72, 73, 78, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97, 100

### v30→v31 快速对比（自动检测）
- procedure_plan 混入: v30=5 → v31=0 ✅
- response_assessment "On treatment" + empty meds: v30=5 → v31=0 ✅
- Stage 空/Not mentioned: v30=9 → v31=7（小改善）

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 0 | 0% | v30 P1(ROW 46 sarcoidosis) **已修复** ✅ |
| **P2** | 112 | 1.84/sample | ROW 1×2, 2×2, 3×1, 5×0, 6×3, 7×2, 8×0, 9×0, 10×0, 11×3, 12×2, 14×2, 17×0, 18×4, 20×3, 22×5, 27×3, 29×3, 30×3, 33×1, 34×4, 36×2, 37×1, 40×1, 41×4, 42×5, 43×1, 44×2, 46×2, 49×0, 50×4, 52×2, 53×2, 54×0, 57×3, 59×1, 61×1, 63×1, 64×3, 65×0, 66×4, 68×1, 70×4, 72×2, 73×0, 78×1, 80×3, 82×1, 83×1, 84×1, 85×2, 86×1, 87×0, 88×1, 90×2, 91×3, 92×2, 94×0, 95×2, 97×1, 100×2 |

---

## v30→v31 改善总结

### 确认修复的 P2 (19个) + P1 (1个)
| ROW | 修复内容 |
|-----|---------|
| 3 | Letter [REDACTED] garbling → 不再生成 "a medication" |
| 8 | procedure_plan chemo 混入 → 清除 |
| 12 | procedure_plan GK → 清除 |
| 17 | Stage "Not mentioned" → "Stage IA (inferred)" + procedure_plan labs → 清除 |
| 30 | Letter "at a medication" → 修复 |
| 34 | procedure_plan labs → 清除 |
| 36 | procedure_plan 混入 → 清除 |
| 37 | Letter "at a medication" → 修复 |
| **46** | **P1 sarcoidosis 全修复: Stage IV→IIIA, Mets Yes→No, Goals palliative→curative** |
| 49 | RA "On treatment" → "Not yet on treatment" |
| 53 | procedure_plan RT+chemo+hormone → 清除 |
| 54 | procedure_plan acupuncture → 清除 |
| 63 | procedure_plan labs → 清除 |
| 65 | RA "On treatment" → "Not yet on treatment" |
| 66 | RA "On treatment" → "Not yet on treatment" |
| 70 | procedure_plan expanders → 捕获为 procedure |
| 80 | RA "On treatment" → "Not yet on treatment" |
| 87 | Stage 空 → "Stage IIIA (inferred from pT2 N2a)" |
| 88 | procedure_plan garbled → 清除 |

### 仍未修复的 P2（7 个确认❌）
| ROW | 残留问题 |
|-----|---------|
| 7 | procedure_plan 仍含 GK (Gamma Knife = 放射治疗) |
| 20 | procedure_plan 仍含 "Abdomen, Pelvis" |
| 41 | Stage 仍空 |
| 42 | Stage 仍为 "Not mentioned" |
| 57 | procedure_plan 仍含非 procedure |
| 80 | procedure_plan 仍 garbled "with [REDACTED]" |
| 92 | procedure_plan 仍含 chemo "8" (garbled) |

---

## 逐 Sample 问题清单

### ROW 1 (coral_idx 140) — 0 P1, 2 P2（同 v30）
- 56yo, Stage IIA→IV ER+/PR+/HER2- IDC。2013 R mastectomy+SLN(2.3+2.4cm, node neg, G2), declined tamoxifen。Dec 2019 widespread mets: lungs/peritoneum/liver/ovary + axillary recurrence 1.8cm。Hepatomegaly+omental masses on exam。ECOG 0。Full code。A/P: biopsy R axilla Thursday → if HR+/HER2- → ibrance+[REDACTED]。Brain MRI+bone scan+labs ordered。Integrative Medicine referral。
- P2: imaging_plan "Brain MRI" — 漏了 bone scan（A/P "MRI of brain and bone scan"）— 同 v30
- P2: lab_plan 混入影像（"MRI of brain and bone scan as well as labs"）— MRI+bone scan 应在 imaging_plan。lab_plan 应仅为 "labs to complete" — 同 v30
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IIA→IV ✅。Mets: lungs/peritoneum/liver/ovary ✅。Goals: palliative ✅
- ✅ procedure_plan: "biopsy mass in right axilla" ✅（真正的 procedure）
- ✅ response_assessment: "progressing, widespread metastases" ✅ — 详细
- ✅ Referral: Integrative Medicine ✅。Advance care: Full code ✅
- ✅ **Letter 出色**（16句）: IDC "milk ducts" + ER/PR + HER2 "protein"(**v31✅**) + mets to lungs/"lining of abdomen"/liver/ovaries + axilla recurrence + palliative "make you feel better, not cure" + Ibrance + "another medication"([REDACTED] 处理✅) + biopsy "armpit" + brain MRI + bone scan(letter 正确包含!) + Integrative Medicine + full code + emotional support + closing complete。全面准确通俗



### ROW 2 (coral_idx 141) — 0 P1, 2 P2（v30:1P2 + v31 新增1P2 回归）
- 44yo, Lynch Syndrome(PMS2), metastatic TNBC Stage IIB→IV(liver/S1 bone/chest wall/nodes)。Also colon ca Stage I + endometrial ca FIGO 1。S/p abraxane+pembrolizumab(PD) → irinotecan C3D1(poor tolerance, missing D8)。Confused。Chest wall cellulitis recurrence。Severe back pain(S1 fracture)。Anemia Hgb 7.7。Hyponatremia Na 124!!。Hypokalemia K 3.1。Hep B prior exposure。ECOG 1。HR 136。A/P: change irinotecan to q2w 150mg/m2, doxycycline, 1u pRBC, KCl, NS IV, Rad Onc urgent, increase effexor, HBV q4mo, scans 3mo, social work/home health。F/U 2wks。
- P2: lab_summary 漏了关键电解质 — Na 124(LL), K 3.1(L), Ca 8.2(L), Cl 87(L) 全部缺失。只列了 CBC diff + Albumin + ALP。**v31 回归**（v30 这些值都有）
- P2: Letter 截断 — "Please feel free to contact us if you have" 断了，缺 "any questions" + "Sincerely, Your Care Team" — 同 v30
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: IIB→IV ✅。Mets: liver/bone/chest wall ✅。Goals: palliative ✅
- ✅ response_assessment: "progressing — bone marrow replacing lesions, chest wall worse, back pain worse" ✅ — 详细
- ✅ procedure_plan: "No procedures planned." ✅。imaging_plan: scans 3mo + MRI brain ✅
- ✅ medication_plan: comprehensive(doxycycline+morphine+flexeril+oxycodone+effexor+NS+KCl+pRBC) ✅
- ✅ radiotherapy_plan: "urgently needs Rad Onc" ✅。Referral: Rad Onc + social work + home health ✅
- ✅ **Letter（除截断外）出色**: chest wall infection + back pain "cancer growing" + anemia + Hep B + hyponatremia/K + irinotecan schedule change + doxycycline + transfusion + KCl + Rad Onc + scans + HBV monitoring + 2wks + social work/home health + emotional support。全面准确通俗

### ROW 3 (coral_idx 142) — 0 P1, 1 P2（同 v30，未修复！之前自动检测误判为✅）
- 53yo postmenopausal(Televisit, second opinion), Stage IIA R breast IDC 1.7cm, ER+/PR+/HER2-(IHC 2+, FISH neg), Ki-67 30-35%, LN+。Pre-diabetes。PET CT+Oncotype+genetic testing all pending。Not ISPY eligible。No current meds。ECOG 0。Full code。A/P: discussed chemo/surgery/RT/hormonal/lifestyle。F/U after PET+Oncotype。
- P2: Letter 写 "follow-up after the PET scan and a medication are back" — [REDACTED] = Oncotype Dx（基因组检测），被误解为 "a medication"。v31 的 [REDACTED] 规则修了 facility 但**未覆盖检测名称**。同 v30
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IIA ✅。response_assessment: "Not yet on treatment" ✅
- ✅ genetic_testing_plan: "Genetic testing sent and is pending" ✅。imaging_plan: PET ✅
- ✅ **Letter（除 "a medication" 外）**: IDC "milk ducts" + ER + HER2 "protein"(**v31✅**) + neoadjuvant "treatment before surgery to shrink" + PET + genetics + full code + emotional support + closing complete

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅（同 v30）
- 31yo premenopausal(Televisit), Stage III→IV ER+/PR+/HER2- IDC(micropapillary), metastatic recurrence to cervical/axillary/supraclavicular/IM LN。On goserelin+anastrozole+palbociclib since 2019。CT 08/19: mixed response（cervical LN decreased 1.8→1.2cm, axillary LN increased 1.5→1.9cm, breast nodule decreased 3.2→2.5cm, new suspicious sternal lesion）。MRI: L brachial plexus involvement → arm pain 4/10。MRI brain normal。ECOG 1。Full code。A/P: continue therapy, Rad Onc referral for symptomatic L neck/brachial plexus, CT+bone scan ordered, labs monthly on lupron day。
- ✅ Type: ER+/PR+/HER2- IDC micropapillary + recurrence receptor confirmed ✅（出色）
- ✅ response_assessment: **出色** — 详细 CT 比较 with measurements（mixed response 准确描述）
- ✅ current_meds: anastrozole+palbociclib+leuprolide ✅（全部三个）
- ✅ radiotherapy_plan: Rad Onc for symptomatic brachial plexus ✅
- ✅ **Letter 出色**（9句）: recurrent cancer + LN neck/chest + "nerves in left arm, causing pain" + mixed response "smaller...but other areas have grown" + continue meds + nausea med + Rad Onc + CT+bone scan + monthly labs "on lupron injection day" + full code + closing complete。准确通俗

### ROW 6 (coral_idx 145) — 0 P1, 3 P2（v30:2P2, v31新增Stage空1P2）
- 34yo premenopausal, bipolar 2。R breast 1.5cm grade I IDC ER+(>95%)/PR+(~90%)/HER2-(IHC 2+, FISH non-amp), Ki-67 ~10%。MammaPrint low risk。0/1 SLN。Myriad neg。S/p bilateral mastectomy+expanders 06/21/19。Zoladex 06/08, letrozole starting today。Night sweats post-injection。Labs: Estradiol 172(high, zoladex may not fully suppressed), Vit D 24。A/P: letrozole ≥3yr → tamoxifen。Estradiol monthly。RTC 3mo。
- P2: Patient type "New patient" — zoladex 06/08 已由该提供者开始。应为 "Follow up" — 同 v30
- P2: Stage 空 — 1.5cm(pT1c)+0/1 N0 = Stage IA。v31 推断规则未触发此案例
- P2: Referral-Genetics "Dr. [REDACTED] genetics referral" — 历史转诊 04/24/2019（Myriad 已 negative）混入当前 referrals — 同 v30
- ✅ Type: ER+/PR+/HER2- grade 1 IDC with DCIS ✅。Mets: No ✅。Goals: curative ✅
- ✅ current_meds: zoladex+letrozole ✅。lab_summary: comprehensive(Estradiol+VitD+CMP+CBC) ✅
- ✅ **Letter**: bilateral mastectomy + L benign/R cancer "small area, no LN spread" + letrozole "prevent coming back" ≥3yr + gabapentin + estradiol labs + 3mo + closing complete。准确通俗

### ROW 7 (coral_idx 146) — 0 P1, 2 P2（同 v30）
- Stage II→IV ER-/PR-/HER2+(IHC) IDC L breast, metastatic since 2008(supraclavicular+mediastinal)。Multiple lines(Taxotere/Herceptin→Tykerb→capecitabine→pertuzumab/Herceptin/Taxotere)。PET-CT: probable mild PD(SUV 2.1 vs 1.8)。[REDACTED] elevated 14.8。LVEF decreased 52%。Brain MRI neg。BRCA neg。Second opinion。A/P: d/c current(PD+decreased LVEF), recommend [REDACTED/T-DM1] next, recheck [REDACTED/LVEF] prior, no hormonal therapy。
- P2: procedure_plan "Would recheck [REDACTED]" — [REDACTED] = LVEF/echo（imaging, not procedure）。POST-PROCEDURE-FILTER 无法捕获（关键词被 REDACTED）— 同 v30
- P2: Letter 写 "test to check the levels of a medication" — [REDACTED] LVEF/echo 被误解为 "medication levels"。v31 [REDACTED] 规则未覆盖 LVEF/echo 场景 — 同 v30
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Stage: II→IV ✅。second opinion: yes ✅
- ✅ response_assessment: "probable mild progression, SUV 2.1 vs 1.8, [REDACTED] elevated 14.8" ✅ — 准确有细微区分
- ✅ medication_plan: d/c current + recommend [REDACTED] next line ✅
- ✅ **Letter（除 garbled 外）**: mets sites + "grown a little bit" + "chest not changed" + brain clear + LVEF "heart working less well 52%" + d/c treatment + new medication + closing complete

### ROW 8 (coral_idx 147) — 0 P1, 0 P2 ✅ ← **v30 的 2 P2 全部修复！**
- 29yo premenopausal(Televisit), clinical Stage III ER-/PR-/HER2+(IHC 3+, FISH 5.7) IDC L breast。Incomplete neoadjuvant TCHP(3 partial cycles, non-adherent)。S/p lumpectomy+ALND: **breast pCR!**(0% cellularity) but 3/28 LN+(largest 2.4cm, ECE, Ki-67 75%)。Necrotizing lymphadenitis(Kikuchi's disease)。PET/CTs no distant mets。ECOG 0。3 young kids。Family discouraged chemo。A/P: high risk → adjuvant AC x4 → T-DM1 + radiation。Port placement + echocardiogram。Social work referral。
- **v30 P2 #1 修复**: procedure_plan v30 "adjuvant AC x4, T-DM1, needs port placement"(chemo混入) → v31 **"port placement"** ✅（POST-PROCEDURE-FILTER 清除 chemo）
- **v30 P2 #2 修复**: response_assessment v30 "Not yet on treatment"(过度应用) → v31 **正确描述 post-neoadjuvant pathology: "no residual disease in breast but 3/28 LN positive, largest 2.4cm ECE"** ✅
- ✅ Type: ER-/PR-/HER2+(IHC 3+, FISH 5.7) IDC ✅（含 FISH ratio）。Stage: III ✅。Goals: curative ✅
- ✅ imaging_plan: echocardiogram ✅。medication_plan: AC x4 → T-DM1 ✅。radiotherapy_plan: radiation after AC ✅
- ✅ **Letter 出色**（15句）: IDC "milk ducts" + Stage III "spread to LN but not other parts" + **"no cancer in breast tissue (good sign), but still in some lymph nodes"（pCR 通俗解释出色！）** + not on active treatment + oxycodone + AC x4 → T-DM1 + radiation + port "easier to give medication" + echocardiogram "check heart" + social work + closing complete。准确全面通俗

### ROW 9 (coral_idx 148) — 0 P1, 0 P2 ✅（同 v30）
- 63yo, kidney transplant recipient(identical twin 1990)。Stage II R breast IDC ER+(85%)/PR-(<1%)/HER2-(IHC 0, FISH 0.89), Ki-67 1-2%。S/p neoadjuvant AC x4+taxol x12 → bilateral mastectomies 06/11/21: R breast 3.84cm residual(5% cellularity), margins neg, 3/4 SLN+(1 macro 0.21cm ECE + 1 micro 0.025cm + 1 ITC)。L breast neg。Neuropathy improving。Drains+expanders in。ECOG 0。Full code。A/P: RT referral, letrozole after RT, Fosamax(osteopenia), drains Thursday。
- ✅ Type: ER+/PR-/HER2- IDC ✅。response_assessment: **出色** — "3.84cm, 5% cellularity, 1 LN macro 0.21cm + 1 micro 0.025cm, treatment effect"
- ✅ procedure_plan: "Drains removed Thursday" ✅（真正 procedure）。medication_plan: letrozole+Fosamax ✅
- ✅ **Letter 出色**: bilateral mastectomy + 3.84cm + chemo before surgery + margins neg "no cancer at edges" + LN+ + ER+/PR- "respond to estrogen but not progesterone" + recovering+drains + letrozole "prevent coming back" + Fosamax "protect bones" + RT + drains Thursday + full code + emotional support + closing complete

### ROW 10 (coral_idx 149) — 0 P1, 0 P2 ✅ ← **v30 的 2 P2 全部修复！**
- 66yo, Stage II L breast HR+/HER2- IDC。S/p neoadjuvant letrozole(April 2021) → L mastectomy 07/24/21(8.8cm residual, LN involved) → bilateral reductions+re-excision 08/07/21。Oncotype low risk → no chemo。Recovering well。Stopped smoking。ECOG 0。Full code。Phone visit(failed video)。A/P: continue letrozole, RT to L chest wall+surrounding LN, DEXA, RTC [REDACTED]。
- **v30 P2 #1 修复**: response_assessment v30 "No specific evidence" → v31 正确包含 post-neoadjuvant pathology "8.8cm residual + LN involved" ✅
- **v30 P2 #2 修复**: Letter v30 "referred to for a follow-up visit"(garbled [REDACTED] 医生名) → v31 "another visit with us in the future"(自然表达) ✅
- ✅ Type: HR+/HER2- IDC ✅。Stage: II ✅。Goals: curative ✅。current_meds: letrozole ✅
- ✅ radiotherapy_plan: "RT to L chest wall + surrounding LN" ✅。imaging_plan: DEXA ✅
- ✅ **Letter**: surgery + cancer removed + LN + continue letrozole + RT "left side of chest and nearby lymph nodes" + DEXA "bone health" + future visit + full code + closing complete。简洁准确通俗

### ROW 11 (coral_idx 150) — 0 P1, 3 P2（同 v30）
- 68yo, Stage IIIC→IV IDC L breast, bone mets(mandible/T-spine/R femur)。S/p mastectomy+ALND+TC x4+XRT。Femara NOT TAKEN。11/2011 bone mets → T6 fixation+XRT。Then Letrozole。10/10/12 PET CT: mandibular mass increased → **10/16/12 started Faslodex+Denosumab**。Current visit(12/11/12): s/p jaw XRT(healing, numbness remains)。Worsening R leg pain/numbness 2wks。Thrush。ECOG 1。Labs normal。A/P: **Exam stable**。Continue Faslodex+Denosumab。MRI lumbar/pelvis/R femur + PETCT to toes ordered。Mycelex for thrush。
- P2: response_assessment 时态错误 — "PET/CT showed increased activity...cancer is progressing on Faslodex" 但该 PET CT(10/10/12)是 Faslodex **开始前**(10/16/12)做的。PET CT 反映的是 Letrozole 上的进展。当前 Faslodex 2 个月后 A/P 说 "Exam stable" — 同 v30
- P2: imaging_plan 漏了 MRI — A/P 明确 "MRI of lumbar, pelvis and right femur"，但 imaging_plan 只写了 PETCT — 同 v30
- P2: Letter 时态误导 — "cancer in your jaw has grown" 呈现 2 个月前 Faslodex 前的 PET CT 为当前消息。Jaw 已经接受放疗并在愈合中 — 同 v30
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IIIC→IV ✅。current_meds: Faslodex+Denosumab ✅
- ✅ lab_summary: comprehensive(CBC+CMP all values) ✅。medication_plan: continue+Mycelex ✅
- ✅ **Letter（除时态外）**: bone mets + R leg numbness worsening + continue meds + Mycelex "thrush 5x/day 14 days" + PETCT femur/toes + closing complete

### ROW 12 (coral_idx 151) — 0 P1, 2 P2（v30:4P2→v31:2P2, 修复2个）
- 51yo, de novo Stage IV ER+/PR+/HER2+(IHC 3+, FISH 5.4) IDC。Mets to [REDACTED]/lung/nodes/brain/bone(widespread osseous+pathologic fractures+cord compression)。S/p XRT T-spine → herceptin+[REDACTED]+letrozole → taxotere/taxol(severe sepsis x3, intolerant) → herceptin+[REDACTED]+letrozole since 08/17。GK x3(23→19→17 lesions)。Current: MRI brain 01/31/19: 2 new foci(3mm+1mm), previously treated resolved。CT CAP: stable osseous, no pulmonary nodules, celiac node 9→7mm。Doing "very well," off walker! ECOG 1。DNR/DNI POLST。CHF/HFpEF, DM, HTN, Hep B。A/P: continue all, CT CAP q4mo, bone scan 4mo, MRI brain q4mo, Echo q6mo, await GK/Rad Onc, F/U 6wks。
- **v30 P2 #1 修复**: summary 不再幻觉 "liver" 作为 [REDACTED] 站点 → v31 直接跳过 ✅
- **v30 P2 #4 修复**: procedure_plan v30 "await GK/Rad Onc"(GK=radiation) → v31 **"No procedures planned."** ✅（POST-PROCEDURE-FILTER 清除 GK）
- P2: response_assessment 仍有时态/模态混淆 — 引用旧日期(08/15/18, 09/05/18)而非最新(01/31/19), 将 CT 发现归到 "MRI brain" — 同 v30
- P2: imaging_plan 漏了 Echo q6mo — A/P 明确 "Echo q6 months, repeat again in April 2019" — 同 v30（🩺 医生确认）
- ✅ Type: ER+/PR+/HER2+ IDC ✅。current_meds: herceptin+letrozole ✅。Advance care: POLST+DNR/DNI ✅
- ✅ **Letter 出色**: "brain has new small spots, but other parts stable"(mixed response 出色) + herceptin+letrozole+pain med + "no chemo because it did not work well"(intolerance 通俗化) + CT+MRI brain+bone scan q4mo + Rad Onc + 6wks + closing complete

### ROW 14 (coral_idx 153) — 0 P1, 2 P2（同 v30）
- 58yo, de novo Stage IV ER+(99%)/PR+(25%)/HER2-(FISH neg) to bone/liver/nodes。S/p multiple spine surgeries+XRT+faslodex+palbociclib → patient STOPPED(sensitivity test) → Mexico treatment → now self-administering at home: doxorubicin+gemcitabine+docetaxel weekly + pamidronate+metabolic therapy+vaccines。US oncologist monitoring role。CT cancelled, scans May。Nerve pain improving, mobility better。Weight loss -2 lbs/mo。ECOG 2, wt 48.2kg。CA 27.29 trending down 193→48。A/P: monitoring, cannabis+sulfur, Cymbalta Rx, PT, CT+MRI May, MRI 6wks, labs q2wks。
- P2: current_meds 空 — 但患者正在服用 cancer meds（gemcitabine/docetaxel/doxorubicin/pamidronate from Mexico），second doc 确认 — 同 v30
- P2: Next visit "2 months" — doc 1 "F/u 3 months" vs doc 2 "return in 2 months"（from Mexico）— 同 v30
- ✅ Stage: IV ✅。response_assessment: "stable, no significant imaging changes" ✅
- ✅ imaging_plan: CT CAP+spine MRI May + repeat MRI 6wks ✅。lab_plan: labs q2wks ✅。lab_summary: comprehensive incl CA 27.29 ✅
- ✅ **Letter 出色**（21句）: stable + low dose chemo at home + vaccines+pamidronate + chills from bone drug + mobility improved + cancelled CT + stopped palbo/fulvestrant + Cymbalta + cannabis/sulfur + CT+MRI May + MRI 6wks + labs q2wks + PT + 2mo + closing complete。全面准确通俗（缺 weight loss dietary advice）

### ROW 17 (coral_idx 156) — 0 P1, 0 P2 ✅ ← **v30 的 3 P2 全部修复！**
- 53yo(Televisit), L breast IDC 0.8cm grade 2, ER+(>95%)/PR+(>95%)/HER2-(IHC 0, FISH 1.1X), Ki-67 5%。0/5 LN, margins neg, no DCIS。Chest CT neg。Menopausal status uncertain(s/p hysterectomy)。No current meds。Family hx: sister ovarian ca @40s, aunt breast ca @60s。A/P: 10-15% recurrence risk。Adjuvant hormonal ≥5yr(tamoxifen or AI based on menopausal status)。Breast RT scheduled tomorrow。Check hormone labs。DXA。Genetics referral。Nutritionist。F/U after RT。
- **v30 P2 #1 修復**: Stage v30 "Not mentioned" → v31 **"Stage IA (inferred from pT1b N0)"** ✅（v31 Stage 推断规则生效！）
- **v30 P2 #2 修復**: procedure_plan v30 "check labs" → v31 **"No procedures planned."** ✅（POST-PROCEDURE-FILTER 清除 labs）
- **v30 P2 #3 修復**: Letter v30 "no cancer found in removed tissue"(误导) → v31 不再包含此表述 ✅
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: adjuvant hormonal ≥5yr tamoxifen/AI ✅。radiotherapy_plan: breast RT ✅
- ✅ imaging_plan: DXA ✅。lab_plan: "check labs including hormones" ✅。genetic_testing_plan: genetics referral ✅
- ✅ Referral: Nutrition+Genetics+RT consult ✅
- ✅ **Letter 出色**: IDC "milk ducts" + "early stage" + ER/PR "sensitive to hormones" + HER2 "protein" + adjuvant hormonal "prevent coming back" + "depends on menopause" + hormone test + RT + DXA "bone health" + genetics + nutritionist + F/U after RT。全面准确通俗

### ROW 18 (coral_idx 157) — 0 P1, 4 P2
- 65yo female, L breast cancer。2017/9 screening mammogram → 12mm asymmetry L UOQ。2017/10 US-guided core bx → atypical papillary lesion, suspicious for intracystic papillary carcinoma。2017/10 left lumpectomy → IDC 8mm grade 1, ER+(~100%), PR+(95%), HER2-(1+), Ki-67 5%。合并 encapsulated papillary carcinoma (=DCIS, 8mm) + additional DCIS。Margins: invasive 2mm from deep, submitted margins neg。SLN bx: isolated tumor cells in 1/3 LN (0/3 H&E+) = pN0(i+)。pT1b。PMH: papillary thyroid ca, s/p thyroidectomy 2015, on levothyroxine。Half-sister breast ca @45。Allergies: ibuprofen, penicillins, sulfa。Initial med onc consultation, with husband。A/P: reviewed pathology+treatment principles。Strongly rec adjuvant endocrine therapy 5-10yr with [REDACTED]。Patient NOT interested in chemo → will NOT pursue molecular profiling。DEXA ordered。RTC after Rad Onc eval +/- XRT。Discussed with UCSF Cancer Risk re family hx — they will reach out to pt today。BMD: "some thinning" previously。
- P2: **genetic_testing_plan temporal** — 写的是过去失败的尝试 ("Plan to obtain blood sample... has not occurred") 而非当前活跃的转诊 "UCSF Cancer Risk will reach out to pt today"
- P2: **Referral.Genetics = "None"** — 笔记明确 "discussed with UCSF Cancer Risk. They will reach out to pt today"，这是遗传学转诊/协调
- P2: **Letter "med onc" jargon** — 使用 "med onc consultation" 缩写术语，患者看不懂
- P2: **Letter Stage "I-II"** — Stage_of_Cancer 是 Stage IA，Letter 写成 "Stage I-II" 不精确（应为 Stage I）
- ✅ Patient type: New patient ✅。Type: ER+/PR+/HER2- IDC + encapsulated papillary CA ✅。Stage: IA (inferred pT1b N0) ✅
- ✅ Goals: curative ✅。response_assessment: "Not yet on treatment" ✅。current_meds: 空 ✅（尚未开始癌症药物）
- ✅ medication_plan: adjuvant endocrine 5-10yr ✅。radiotherapy_plan: Rad Onc eval +/- XRT ✅。imaging_plan: DEXA ✅
- ✅ Specialty: Rad Onc eval ✅。follow up: RTC after Rad Onc eval ✅
- ✅ **Letter 较好**: IDC "started in the milk ducts" + new patient diagnosis section appropriate + adjuvant therapy "prevent cancer from coming back" + DEXA "bone health" + Rad Onc referral + genetic testing mentioned。Medical jargon和Stage精度可改善

### ROW 20 (coral_idx 159) — 0 P1, 3 P2
- 75yo postmenopausal female。2009 L breast IDC 0.9cm grade II, ER+PR+HER2-, 0/2 SLN, 1.8cm DCIS。S/p bilateral mastectomies + 5yr tamoxifen。2015 genetic testing (21 genes) neg。2020 metastatic recurrence: innumerable bone mets + R axillary/mediastinal/hilar LN。R iliac crest bx: ER+(80%)/PR+(50%)/HER2-(FISH 1.05)。L rib pain。ECOG 0。Non-cancer meds: aspirin, risedronate, calcium, vit D, B12, estradiol(Estrace), metformin, rosuvastatin, telmisartan。3 sisters breast ca (42/48/51yo)。A/P: HR+/HER2- Stage IV → start letrozole+palbociclib(Rx sent)。Denosumab after dental clearance。MRI spine + CT CAP。Rad Onc referral。Foundation One testing。Labs + tumor markers + monthly CBC。RTC ~1mo。
- P2: **procedure_plan = "Abdomen, Pelvis"** — CT CAP 是影像不是 procedure（已知 v30 残留 ❌）
- P2: **current_meds = "letrozole, palbociclib"** — 本次就诊新开处方，患者尚未服用。当前无癌症药物
- P2: **lab_summary 报告 2013 年 glucose** — 笔记中唯一的 lab 来自 2013/03/01（"Results for orders placed on 03/01/13"），非本次 2021/02 就诊。应为 "No labs in note"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV ✅。Mets: Yes, bone+LN ✅。Goals: palliative ✅
- ✅ response_assessment: "Not mentioned" ✅（初次就诊未治疗）
- ✅ medication_plan: letrozole+palbo+denosumab(dental clearance) 全面 ✅
- ✅ imaging_plan: MRI spine+CT CAP+repeat 3mo ✅。genetic_testing_plan: Foundation One ✅
- ✅ **Letter 出色**: bilateral mastectomy 2009 + cancer spread to bones/LN + Stage IV + letrozole+palbociclib+denosumab "protect bones" + dental clearance + MRI+CT+blood tests+tumor markers+genetic testing + Rad Onc referral + 1mo F/U。全面准确通俗

### ROW 22 (coral_idx 161) — 0 P1, 5 P2
- 72yo female, complex history: 1994 L DCIS s/p lumpectomy+XRT (ER+, no endocrine). 2000 R Stage II IDC (1.5cm, 14 LN, HR+) s/p lumpectomy+ALND+AC×3+C×1+XRT+6yr tamoxifen/[redacted](到2007). 2020/5 metastatic relapse: R chest wall recurrence + bone mets + R infraclavicular/IM nodes (HR+/HER2-). 2020/6 XRT to T10+L4 + abemaciclib+letrozole (→7月换anastrozole因皮疹). Abemaciclib dose reduced 150mg bid (10/2020). PET/CT 11/2020+04/2021 good response. 2021/7 pneumonitis→abemaciclib discontinued, prednisone started. Second opinion visit. Current meds: anastrozole, denosumab(Xgeva), prednisone, Lomotil, Bactrim. Labs 01/29/21: WBC 3.16L, Hgb 10.7L, Cr 1.19H, eGFR 46L. ECOG 0. Full code. A/P: recommend PET CT now → if stable continue arimidex alone; if progression → faslodex+[redacted] if [redacted] mutation; future options: afinitor/xeloda/trial.
- P2: **Type_of_Cancer "metastatic biopsy ER+/PR-/HER2-"** — 原文对转移活检只说 "HR+" 不能推断 PR-。HR+ = ER+ and/or PR+
- P2: **lab_summary "No labs in note"** — 笔记有 01/29/2021 labs：WBC 3.16L, Hgb 10.7L, Hct 32.6L, RBC 2.96L, Cr 1.19H, eGFR 46L 等多项异常
- P2: **Distant Metastasis "Yes, to bones, chest wall"** — chest wall recurrence是局部复发（locoregional）非远处转移。Distant mets 应为 "Yes, to bones"。也漏了淋巴结
- P2: **imaging_plan 混入药物方案** — 应为 "PET CT recommended"，但写成了整段 A/P contingency plan
- P2: **Letter 截断** — 结尾 "[source:none" 未闭合，缺少 "Please feel free to contact us..." 和 "Sincerely, Your Care Team"
- ✅ Patient type: New patient + second opinion ✅。Stage: II→IV ✅。Goals: palliative ✅
- ✅ current_meds: anastrozole+denosumab ✅。recent_changes: abemaciclib d/c+letrozole→anastrozole ✅
- ✅ response_assessment: "PET/CT good response" ✅
- ✅ medication_plan: conditional plan (stable→arimidex; progression→faslodex+[redacted]) ✅
- ✅ Advance care: "Full code" ✅
- ✅ **Letter 内容好**（除截断外）: 详细治疗史通俗化 + 药物变更原因 + PET scan conditional plan + future options + emotional support。全面通俗

### ROW 27 (coral_idx 166) — 0 P1, 3 P2
- 41yo female, metastatic breast cancer to bone。1999 early stage node neg IDC, ER/PR+, HER2 neg, lumpectomy+XRT。2006 mets to L1。Tx: Tamoxifen 2005-2008, Zoladex 2009, Femara 2011。Follow-up。新症状: lightheadedness, new reading glasses, low back pain, easy bruising, urinary frequency。Meds: letrozole+Zoladex+zolendronic acid+Ca/vit D。PET/CT 9/27/11: Stable to slightly decreased osseous mets. L1 SUV 6/9 unchanged, L2 decreased. No new lesions。A/P: stable disease, continue meds。Reassess back pain 2wks consider MRI spine。Obtain UA (UTI vs estrogen depletion)。Obtain CBC w/ plt (easy bruising)。
- P2: **Type_of_Cancer "HER2: not tested"** — 原文 "ER/PR + [REDACTED] neg" = HER2 negative，非 "not tested"
- P2: **lab_summary "CBC with platelets. No specific values provided"** — CBC 是计划的检查（lab_plan），不是已有结果。笔记中无实验室数据。应为 "No labs in note"
- P2: **lab_plan 漏了 UA** — A/P 明确 "Obtain UA"（possible UTI vs estrogen depletion），但 lab_plan 只写了 CBC
- ✅ Stage: IV ✅。Goals: palliative ✅。Mets: Yes, to bone ✅
- ✅ current_meds: letrozole+Zoladex+zolendronic acid ✅。response_assessment: 详细 PET/CT stable disease ✅
- ✅ imaging_plan: consider MRI spine at 2 weeks ✅
- ✅ **Letter 简洁出色** (grade 4.4): stable + slightly decreased activity + no new cancer in bones + continue same meds + MRI if pain persists + blood test for bruising + return if pain worsens。通俗完整

### ROW 29 (coral_idx 168) — 0 P1, 3 P2
- 59yo female, R breast multifocal IDC grade 2, ER+(>90%)/PR+(30%weak)/HER2-(FISH neg), Ki-67 10%。S/p R partial mastectomy×2+SLN bx: site1 1.6cm IDC micropapillary+ductal neg margins LVI+; site2 0.6cm IDC POSITIVE margin; SLN 1/1 micromet 0.5mm。pT1c(m)N1mi(sn)。DCIS intermediate。Oncotype Low Risk (+0.046)。BRCA neg (2014)。工作在海外。A/P: no chemo (Oncotype Low Risk)。Start letrozole 2.5mg PO daily (Rx sent)。Ca 1200mg+vit D。Bone density scan on return。Dental eval for bisphosphonate。Re-excision September 2019。RT locally in [redacted]。Vaginal moisturizer。
- P2: **current_meds = "letrozole 2.5mg PO daily"** — 本次就诊新开处方，患者尚未服用。应为空
- P2: **response_assessment = "On treatment"** — 患者尚未开始癌症治疗，letrozole 刚刚开出。应为 "Not yet on treatment"（POST-RESPONSE-PRETREATMENT hook 因 current_meds 错误未触发）
- P2: **imaging_plan 末尾 "Bone scan" garbled** — 应为 "Bone density scan when she returns"，末尾多出 "Bone scan"
- ✅ Patient type: New patient ✅。Type: ER+/PR+/HER2- IDC ✅。Stage: pT1c(m)N1mi(sn) ✅
- ✅ Goals: curative ✅。Mets: No ✅。procedure_plan: surgery September ✅
- ✅ medication_plan: letrozole+Ca+vit D+vaginal moisturizer 全面 ✅。radiotherapy_plan: RT locally ✅
- ✅ **Letter 简洁好**: first visit + IDC "milk ducts" + early stage + letrozole "hormone therapy prevent coming back" + RT risk + surgery Sept + bone density scan + F/U Sept + long-term follow up closer to home。通俗完整

### ROW 30 (coral_idx 169) — 0 P1, 3 P2
- 64yo postmenopausal female。Complex history: 2006 R breast DCIS (untreated), 2012 biopsy DCIS+possible invasion (untreated)。2016/11 invasive: 0.35cm grade 2 IDC, ER-(0%)/PR-(0%)/HER2+(IHC 3+, FISH 8.9), Ki-67 30%。PET/CT: 9.0×3.8cm R breast mass (SUV 7.8), R axillary LN (SUV 1.8) but FNA **negative** for metastatic carcinoma。MRI: NME throughout R breast。Clinical stage II-III, node-negative。PE: 6×11cm mass, denuded R nipple。Invitae neg。Tumor markers: CA 27.29 67.7-106.2, CA 15-3 60.9-74.5。No meds。A/P: neoadjuvant THP→AC or TCHP, then trastuzumab 1yr。TTE。Mediport。Surgery+RT after chemo。Patient deciding where to proceed。
- P2: **lab_summary 漏 tumor markers** — 只提 Cr 0.74，漏了 CA 27.29 (67.7-106.2, 升高↑) 和 CA 15-3 (60.9-74.5, 升高↑)
- P2: **procedure_plan garbled** — "Mediport placement, and if treated here at [REDACTED]" 截断不完整
- P2: **Letter 错误称 LN involvement** — "lymph nodes in the armpit area that are also involved" 但 A/P 明确 "high-risk **node-negative**"，R axillary LN FNA negative
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Stage: II-III ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: 空 ✅
- ✅ medication_plan: 详细 neoadjuvant regimen (THP→AC or TCHP) + 1yr trastuzumab ✅
- ✅ imaging_plan: TTE ✅。Invitae genetic testing already done ✅
- ✅ **Letter 内容丰富**（除LN错误外）: ER-/PR-/HER2+ explained + neoadjuvant "before surgery to shrink" + TTE "heart test" + Mediport explained + "Stage II-III not very early" + RT referral

### ROW 33 (coral_idx 172) — 0 P1, 1 P2
- 63yo female, L breast ER+/PR+/HER2- invasive lobular carcinoma, Stage IIB→IIIA。S/p bilateral mastectomies + TC×6 + XRT。On adjuvant letrozole since 2/2011 (brand name, generic GI intolerance)。Follow-up。Headaches 1x/month (?tension)。Joint stiffness AM (improves walking)。ECOG 0。+L neck <1cm LN soft mobile。+ lymphedema。A/P: continue letrozole >5yr (patient preference)。Exercise+weight+alcohol risk reduction。NSAIDs PRN for joints。Ca+vit D。If headaches continue → consider MRI brain。F/U 6 months。
- P2: **Letter clinical tone** — "Continue letrozole daily. Continue calcium and vitamin D daily. NSAIDs with food as needed." 像临床医嘱非通俗语言。"No evidence of recurrence" 重复出现两段
- ✅ Type: ER+/PR+/HER2- ILC ✅。Stage: IIB→IIIA(staging update) ✅。Mets: No ✅。Goals: curative ✅
- ✅ current_meds: letrozole ✅。response_assessment: no evidence of recurrence ✅
- ✅ medication_plan: letrozole+Ca+vit D+NSAIDs ✅。imaging_plan: consider MRI brain ✅
- ✅ Next visit: 6 months ✅

### ROW 34 (coral_idx 173) — 0 P1, 4 P2
- 71yo female, Stage III L breast IDC。2011: L lumpectomy 1.4cm IDC ER+/PR low/HER2- (refused SLN/reexcision/chemo)。2012: rapid local recurrence → bilateral mastectomies+implants, 3.3cm 11+LN ER+/PR-/HER2-, shortened AC/T, anastrozole(self-D/C'd), declined CW XRT。2020: 2nd local recurrence → FNA ER+(100%)/PR+(50%)/HER2-(FISH 1.6X)。Excision: 1.7cm IDC grade 3, skeletal muscle invasion, margins neg。PET-CT: L breast implant hypermetabolic, L 6th rib unclear。Brain MRI neg。SOB eval: echo normal。ECOG 0。A/P: CALOR study → no chemo (ER+)。Switch anastrozole→tamoxifen 20mg。CW RT referral (now accepts)。Check labs。RTC 6mo。
- P2: **Type_of_Cancer "2020 biopsy ER+/PR-"** — 2020 FNA 显示 PR+(50%)，非 PR-。PR- 是 2012 的结果
- P2: **current_meds = "arimidex"** — A/P 说 "resumption" 暗示当前未在服，患者之前 self-D/C'd anastrozole
- P2: **lab_plan = "No labs planned"** — A/P 明确 "check labs"
- P2: **Letter "referred to a medication for consultation"** — [REDACTED] 是医生名字不是药物名。应为 "referred to a specialist/radiation specialist"
- ✅ v30 fix confirmed: procedure_plan "check labs" → "No procedures planned." ✅（POST-PROCEDURE-FILTER）
- ✅ Stage: III + local recurrence (not metastatic) ✅。Goals: adjuvant ✅。Mets: No ✅
- ✅ medication_plan: tamoxifen 20mg ✅。radiotherapy_plan: CW RT referral ✅。Next visit: 6mo ✅
- ✅ findings: 全面 PET/MRI/biopsy+PE ✅。response_assessment: local recurrence + no distant mets ✅

### ROW 36 (coral_idx 175) — 0 P1, 2 P2
- 27yo premenopausal female, pT3N0 R breast ER+/PR+/HER2- grade III mixed ductal+mucinous carcinoma。S/p bilateral mastectomies+expanders (12/6/20)。Post-op infection resolved (cellulitis, washout, Bactrim)。Tamoxifen 1/29/21 + Zoladex 2/6/21。Taxol→Abraxane (grade 3 infusion reaction)。Cycle 8 of 12。R arm/hand swelling (improved elevation)。Nausea managed (Zofran+Compazine+Ativan)。Lexapro+Ambien for anxiety/sleep。ECOG 1。BMI 43.1。Labs: CBC adequate, mild anemia Hgb 11.8L, Albumin 3.3L。PET/CT incidental: R thyroid nodule 2.1cm (US recommended)。A/P: Doppler r/o DVT; continue Abraxane×12; continue zoladex; Rad Onc referral next week; antiemetics; valtrex ppx; lexapro+ativan+ambien; RTC 2wk。
- P2: **Stage "IIIA (inferred from pT3 N0)"** — pT3N0 = Stage **IIB** (AJCC 8th), NOT IIIA (IIIA = T3N1 or T0-3N2)
- P2: **current_meds missing tamoxifen** — 患者 1/29/21 开始 tamoxifen，正在服用中。只写了 "Abraxane, zoladex"
- ✅ v30 fix confirmed: procedure_plan 清除 ✅
- ✅ Type: ER+/PR+/HER2- mixed ductal+mucinous ✅。Goals: curative ✅。Mets: No ✅
- ✅ lab_summary: CBC values comprehensive ✅。findings: arm swelling+thyroid nodule+labs ✅
- ✅ medication_plan: 非常全面（Abraxane+zoladex+antiemetics+prilosec+valtrex+lexapro+ativan+ambien）✅
- ✅ imaging_plan: Doppler for DVT ✅。radiotherapy_plan: Rad Onc referral next week ✅
- ✅ **Letter 出色**: arm swelling + no infection + thyroid nodule+US + Abraxane switch reason + Zoladex "protect ovaries" + Doppler "blood clot" + Rad Onc + 2wk F/U + emotional support sentence ✅

### ROW 37 (coral_idx 176) — 0 P1, 1 P2
- 61yo female, Video visit (Televisit)。Newly diagnosed Stage IIA L triple negative IDC (ER-/PR-/HER2-), 2.3cm, node neg, grade 3。S/p bilateral mastectomies + L SLN July 2020。Core bx: grade 3, no LVI, high grade DCIS。PMH: asthma, headaches, GI (dexilant)。Former smoker 38 py。Mother breast cancer。Meds: dexlansoprazole。A/P: agree with outside oncologist: dd AC→Taxol。No indication for RT or hormone blockade。Lifestyle: Mediterranean diet, exercise, stress, sleep。Will proceed with chemo at outside facility。Full code。
- P2: **Letter "Stage I-II"** — Stage 是 IIA，应写 "Stage II" 而非 "Stage I-II"
- ✅ v30 fix confirmed: Letter "at a medication" → 已修复 ✅
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: IIA ✅。Goals: curative ✅。Mets: No ✅
- ✅ in-person: "Televisit" ✅。response_assessment: "Not yet on treatment" ✅。current_meds: 空 ✅
- ✅ medication_plan: dd AC+Taxol ✅。Advance care: Full code ✅
- ✅ **Letter 较好**: TNBC "does not have receptors for estrogen, progesterone, or HER2" + adjuvant "after surgery to prevent coming back" + no RT + full code explained。简洁通俗

### ROW 40 (coral_idx 179) — 0 P1, 1 P2
- 62yo female with MS (25yr, secondary progressive, teriflunomide, cane/scooter)。Newly diagnosed Stage 2 ER+(95%)/PR+(5%)/HER2-(FISH 1.2) G1 IDC R breast, 2.3cm。S/p R partial mastectomy+SLN: 1/2 SLN micromet 0.04cm (+1 LN direct extension)。Margins widely neg。DCIS G1。PMH: osteoporosis(Prolia), Graves(thyroidectomy), fractures, DDD, neurogenic bladder。Extensive meds: gabapentin, baclofen, modafinil, levothyroxine, teriflunomide, letrozole(just Rx'd)。A/P: adjuvant letrozole (patient declines chemo→no molecular testing)。DEXA。If no RT→start letrozole immediately。Appt with Dr[redacted]。PT referral。RTC 3mo。
- P2: **response_assessment "On treatment"** — 新患者初次会诊，letrozole 刚开处方尚未开始。应为 "Not yet on treatment"
- ✅ Type: ER+/PR+/HER2- G1 IDC ✅。Stage: II ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: letrozole (on med list) ✅。medication_plan: letrozole+Prolia ✅
- ✅ imaging_plan: DEXA ✅。Referral.Others: PT referral ✅。Next visit: 3mo ✅
- ✅ findings: 全面包括 path+MRI+PE ✅
- ✅ **Letter 简洁好**: IDC "milk ducts" + early stage + letrozole + DEXA "bone density" + PT + 3mo F/U。通俗完整

### ROW 41 (coral_idx 180) — 0 P1, 4 P2
- 32yo female, ATM mutation carrier。L breast 3cm grade 3 IDC, ER+(90%)/PR weakly+(1%)/HER2 1+ (FISH unavail), Ki-67 30%。S/p bilateral mastectomies + L SLN: 1/3 SLN micromet 0.022cm, LVI+。R breast benign。Oncotype High Risk。LVEF 79%。Labs: Hgb 11.8L, Ferritin 27, HCG neg。Premenopausal。Prior visits 03/17+04/21/18。Decided AC-Taxol (Taxol first 12wk→AC)。Port placement scheduled。Chemo at outside facility。After chemo: ovarian suppression+AI, possibly ribociclib trial。Medical marijuana (vape/edibles, no smoking)。
- P2: **Patient type "New patient"** — 已有先前就诊 (03/17, 04/21/18)，应为 "Follow up"
- P2: **Stage 空** — 已知问题❌。pT2 N1mi M0 = Stage IB
- P2: **medication_plan 缺 chemo** — 只写了 post-chemo AI+ribociclib，缺少当前决定的 AC-Taxol
- P2: **Letter "regimenaxol" garbled** — "a chemotherapy regimenaxol" 是 "regimen"+"axol"(Taxol) 拼接错误
- ✅ Type: ER+/PR-/HER2- IDC ✅（PR 1% clinically treated as PR-）。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: 空 ✅
- ✅ procedure_plan: port placement ✅。findings 全面 ✅。lab_summary 全面 ✅
- ✅ therapy_plan: Taxol 12wk→AC + post-chemo ovarian suppression+AI+ribociclib trial ✅

### ROW 42 (coral_idx 181) — 0 P1, 5 P2
- 41yo premenopausal female。R breast multifocal IDC 0.9cm+0.3cm, grade 1。Margins clear (deep reexcised)。0/5 SLN。PR strongly+(95%), HER2/neu negative。ER not explicitly stated but tamoxifen implies ER+。S/p R excisional bx+reexcision+SLN (Sept 2018)。Completed 3-wk RT to R breast (Jan 2019)。Doing well。Regular periods。A/P: start tamoxifen 5yr (Rx written)。RTC 4-6wk。Diagnostic mammogram at next visit。
- P2: **Type_of_Cancer "PR+ IDC, HER2: not tested"** — Note says [redacted]/neu negative = HER2 neg (not "not tested")。ER omitted 但 tamoxifen implies ER+
- P2: **Stage 空** — 已知问题❌。pT1b(m) N0 = Stage IA
- P2: **current_meds = "tamoxifen"** — Rx 刚写出，尚未服用
- P2: **response_assessment "On treatment"** — 刚完成 RT，tamoxifen 尚未开始
- P2: **Letter "responds well to progesterone"** — PR+=有孕激素受体，非 "respond well to"。且 "No new imaging findings mentioned" 是 meta 语言不适合患者信
- ✅ Mets: No ✅。Goals: curative ✅。medication_plan: tamoxifen 5yr ✅
- ✅ imaging_plan: diagnostic mammogram at next visit ✅。Next visit: 4-6wk ✅
- ✅ **Letter 较好**（除 PR 表述外）: IDC "milk ducts" + margins clear + LN no cancer + tamoxifen "prevent coming back" + mammogram + 4-6wk F/U

### ROW 43 (coral_idx 182) — 0 P1, 1 P2
- 38yo female。History: Stage I TNBC at 27 (2010) s/p lumpectomy+SLN+dd AC→T+XRT。BRCA neg。2020 second primary: bilateral mastectomies + L ALND (02/22/21)。Path: 1.3cm grade 3 IDC, ER-/PR-/HER2-(IHC 1, FISH neg), Ki-67>80%。0/1 LN。Neg margins。R breast benign。Post-op severe anemia Hgb 5.4→transfusion 2U。Pre-chemo labs normal。ECOG 0。Full code。PMH: depression, IPV history, C-section w/ PPH。A/P: taxol+carboplatin ×4 cycles adjuvant。RTC 2 days prior to cycle 1 (Oct 14) for lab draw+visit。
- P2: **lab_plan "No labs planned"** — A/P 明确 "RTC 2 days prior to cycle [REDACTED] draw and visit"
- ✅ Type: ER-/PR-/HER2- IDC (TNBC) ✅。Stage: I (pT1c N0) ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: 空 ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: taxol+carboplatin ×4 ✅。supportive_meds: granisetron+compazine+senna ✅
- ✅ lab_summary: 全面 pre-chemo labs (02/17) ✅。Advance care: Full code ✅
- ✅ **Letter 出色** (grade 5.3): bilateral mastectomy + small cancer removed + no cancer in LN + R breast healthy + taxol+carboplatin "prevent coming back" + nausea/constipation meds + 4 cycles + blood test Oct 14 + full medical care + emotional support。通俗完整

### ROW 44 (coral_idx 183) — 0 P1, 2 P2
- 33yo female, BRCA1+。ER+/PR+/HER2- node+ L breast cancer。S/p neoadjuvant dd AC×4→Taxol×4 + bilateral mastectomy + L SLN bx (10/07/18)。Residual: 1cm grade 2 IDC (15% cellularity), DCIS, 1/8 SLN micromet 0.07cm。R breast benign。ECOG 2。Wt 41.3kg (BMI 16.76)。Neck/back pain (ED neg)。Bilateral shoulder stiffness, limited ROM。Anxiety improving (Klonopin PRN, Lexapro not taking)。A/P: RT trial (3 vs 5wk, 12/16/18)。AI after RT。BSO discussion 12/02/18。Zoladex backup。CT chest 1yr (4mm nodule stable)。Nutrition 11/30。PT。F/U 01/05/19。
- P2: **Stage 空** — 已知问题❌。Clinical Stage IIA+ pre-neoadjuvant
- P2: **Letter "cancer not completely removed"** — 残留是新辅助化疗不完全缓解(residual, cellularity 15%)，非手术不完全。Margins NEGATIVE。Letter 误导
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。Mets: No ✅
- ✅ response_assessment: 准确描述 residual disease + no recurrence ✅
- ✅ radiotherapy_plan: clinical trial 3 vs 5wk ✅。procedure_plan: BSO ✅。imaging_plan: CT chest 1yr ✅
- ✅ medication_plan: AI after RT + Zoladex backup ✅
- ✅ Referral: Nutrition 11/30 ✅, PT ✅, Rad Onc ✅。Next visit: 01/05/19 ✅
- ✅ **Letter 内容丰富**（除 "not removed" 外）: residual + R breast healthy + pain neg + stiffness + stable nodule + AI after RT + RT trial 3/5wk + BSO + CT 1yr + PT + nutrition + F/U + emotional support

### ROW 46 (coral_idx 185) — 0 P1, 2 P2 ← **v30 P1 sarcoidosis 全修复确认 ✅**
- 48yo postmenopausal female (s/p TAH-BSO)。R breast IDC, ER+(95%)/PR-(0% residual)/HER2-(1+), Ki-67<5%。Complex history: 10/2020 ED→TAH-BSO (benign)→intra-op R breast bx→IDC。12/2020 PET/CT: R breast mass + R axillary/IM/chest wall mets + **extensive bilateral hilar/mediastinal adenopathy**。01/2021 bronchoscopy+FNA mediastinal nodes: **SARCOIDOSIS (non-necrotizing granulomatous inflammation)** NOT cancer。Invitae: pathogenic [REDACTED]+VUS WRN。S/p neoadjuvant Taxol+[redacted]→[redacted]。06/2021 R partial mastectomy+SLN: 3.5cm residual (ypT2), POSITIVE margins (multifocal), 2/2 SLN macrometastases (6mm, extranodal extension)。Labs: Hgb 10.1L, CRP 5.6H, ESR 60H。This visit: Re-excision needed。Start letrozole (s/p BSO)。Abemaciclib after XRT。DEXA。MRA Jan 2022。Naproxen+APAP+iron。F/U 2-3mo。
- **v30 P1 全修复 ✅**:
  - Stage: v30 **IV** → v31 **IIIA** ✅ (不再 metastatic)
  - Metastasis: v30 **Yes** → v31 **No** ✅ (sarcoidosis 正确识别)
  - Distant Mets: v30 **Yes** → v31 **No** ✅
  - Goals: v30 **palliative** → v31 **curative** ✅
- P2: **Stage "IIIA (pT2 N2a)"** — N2a 需 4-9 positive nodes，实际 2/2 = **N1a**。ypT2 N1a = Stage IIB
- P2: **Referral.Others "Exercise counseling referral"** — Note 只说 "Continue exercise"，非正式转诊
- ✅ Type: ER+/PR-/HER2- IDC ✅。current_meds: letrozole (started today) ✅
- ✅ response_assessment: partial response + residual disease detailed ✅
- ✅ medication_plan 全面: letrozole+naproxen+APAP+allegra+iron+abemaciclib(after XRT) ✅
- ✅ procedure_plan: re-excision ✅。imaging_plan: DEXA+MRA Jan 2022 ✅
- ✅ lab_summary 全面: CBC+CMP+Vit D+CRP+ESR ✅。findings 全面 ✅
- ✅ **Letter 出色**: residual cancer smaller (good sign) + letrozole + abemaciclib after RT + re-excision + RT + DEXA + MRA abdomen + naproxen/APAP/allegra/iron + exercise + 2-3mo F/U + emotional support

### ROW 49 (coral_idx 188) — 0 P1, 0 P2 ✅
- 50yo premenopausal female。L breast multifocal IDC: 12:00 ER+(100%)/PR+(100%)/HER2-(IHC 2+, FISH 1.4); 10:00 ER+(100%)/PR+(100%)/HER2-(IHC 0)。L axillary LN met: ER+(60%)/PR+(70%)/HER2-(IHC 2+, FISH 1.4)。Oncotype Low Risk (score 11)。MRI: 4×2.5×3cm NME upper L breast + 1cm bilobed nodule。PET/CT: PET-negative disease, T7-L1 bone marrow uptake (unclear)。Thoracic spine MRI: no mets。Two outside opinions (neoadjuvant vs surgery first)。Planned L mastectomy 01/06/17。Father PE hx→thrombophilia check before tamoxifen。Surrogate: spouse。
- ✅ v30 fix confirmed: response_assessment "On treatment" → "Not yet on treatment" ✅
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: "Likely stage 2" ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: 空 ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: tamoxifen + thrombophilia assessment ✅。procedure_plan: L mastectomy ✅
- ✅ radiotherapy_plan: XRT discussed for node+ disease ✅。imaging_plan: none ✅
- ✅ Advance care: surrogate decision maker (spouse) detailed ✅
- ✅ findings 全面: 全部 imaging+biopsy+PE ✅。lab_summary: Dec 2015 CBC (per note) ✅
- ✅ **Letter 出色**: IDC "milk ducts" + LN spread "armpit" + "likely stage 2" + ER/PR "proteins" + HER2 "protein" + tamoxifen + "blood clots" risk check + mastectomy explained + RT after surgery + surrogate decision maker。全面通俗

### ROW 50 (coral_idx 189) — 0 P1, 4 P2
- 58yo female, de novo metastatic IDC (2013) to lung/LN/liver/bone, HR+/HER2-, Stage IV (T2 N1 M1)。S/p AC×4→tamoxifen+lupron→progression 10/2014→letrozole+lupron→ibrance added 1/2015。XRT: R pelvis+sternum bone mets (2014), L breast (2019)。Lumpectomy+XRT mid 2019。Pathogenic PMS2 mutation (no genetic counseling yet, no colonoscopy)。Sister breast ca @49。Restaging 12/2021: disease under good control。New progression in L breast: biopsy DCIS+IDC→considering observation vs mastectomy。Video visit second opinion。Full code。ECOG 0。
- P2: **lab_summary "No labs in note"** — Note HAS 11/25/2020 CMP (Total Protein 6.5L, eGFR 57-70)
- P2: **medication_plan includes tamoxifen** — Tamoxifen stopped at progression in 10/2014！Current meds: letrozole+lupron+ibrance+xgeva
- P2: **Referral.Genetics = "None"** — A/P #6 明确 "Referral to genetics for pathogenic PMS 2 mutation"（genetic_testing_plan 有但 Referral 缺）
- P2: **Letter "cancer has also spread to the left breast"** — 原始癌症就在 L breast（"felt a left breast mass in 2012"），L breast progression 是局部复发/进展非"扩散到"
- ✅ Patient type: New patient, second opinion: yes ✅。in-person: Televisit ✅
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV ✅。Mets: Yes, lung/LN/liver/bone ✅。Goals: palliative ✅
- ✅ current_meds: ibrance+xgeva+letrozole ✅。response_assessment: "under good control" ✅
- ✅ procedure_plan: Mastectomy pending ✅。genetic_testing_plan: PMS2 referral ✅
- ✅ Advance care: full code ✅。Next visit: PRN ✅

### ROW 52 (coral_idx 191) — 0 P1, 2 P2
- 35yo premenopausal female, G0P0。L breast IDC grade II, ER+(>95%)/PR+(>95%)/HER2-(FISH 1.1), Ki-67 <10%。S/p L partial mastectomy+SLN+bilateral mastopexy (12/04/20)。Path: 1.7cm grade II IDC, SLN micromet 0.18cm (N1mi), minimal extranodal extension, margins neg。MammaPredict +0.298。Invitae: VUS×3。Menarche 9, premenstrual dysphoric disorder, writer, G0P0。ECOG 0。A/P: Ovarian suppression+AI (Zoladex+[redacted]) after egg harvesting。CT CAP+bone scan for staging。Order Oncotype。Fertility preservation referral to reproductive health。RTC 3wk。
- P2: **Stage "II/III"** — 来自 RxPONDER trial 引用，非患者实际 Stage。pT1c N1mi = Stage IB/IIA
- P2: **Referral.Others = "None"** — A/P 明确 "Referral for fertility preservation asap"+"referred to reproductive health"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: 空 ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: OS+AI after egg harvesting ✅。imaging_plan: CT CAP+bone scan ✅
- ✅ genetic_testing_plan: Oncotype ordered ✅。lab_summary: urine pregnancy neg ✅
- ✅ **Letter 较好** (grade 4.9): [REDACTED] handling 好 ("new medications as discussed") + fertility referral mentioned + CT+bone scan "check stage" + test "learn more about cancer" + 3wk

### ROW 53 (coral_idx 192) — 0 P1, 2 P2
- 59yo female。L breast IDC with neuroendocrine differentiation, 4.5cm, grade 3, LVI+。ER+(>95%)/PR+(30%)/HER2+ (IHC heterogeneous 2+/3+ 90:10%, FISH 4.9X)。Ki-67 25%。DCIS int/high grade 4.5cm。SLN positive 1/2 (6mm met)。Core bx initially HER2-(1+)→excision HER2+(FISH 4.9X)。S/p L lumpectomy+SLN (03/01/17)。PMH benign, no meds, NKDA。Lives in [redacted island]。Family: aunt/cousin breast ca, mother uterine ca, father esophageal ca。A/P: Stage II/III, ER/PR+/HER2+。High risk ~60%。Rec: AC/THP or TCHP + 1yr trastuzumab/pertuzumab + neratinib yr2 + Arimidex 10yr + bisphosphonates + adjuvant breast XRT + genetic counseling。Patient considering options, may treat locally。
- P2: **Letter "What's new" section 空** — New patient consultation但该段无内容，直接跳到 treatment section
- P2: **Letter "AC/targeted therapy with chemotherapy chemotherapy"** — garbled/redundant text
- ✅ v31 fix confirmed: procedure_plan "RT+chemo+hormone" → "No procedures planned." ✅
- ✅ Type: ER+/PR+/HER2+ IDC with neuroendocrine differentiation ✅。Stage: II/III ✅。Goals: curative ✅
- ✅ medication_plan 极其全面: AC/THP+TCHP alternative+neratinib yr2+Arimidex 10yr+bisphosphonates ✅
- ✅ radiotherapy_plan: adjuvant breast RT after chemo ✅。Genetics: counseling referral ✅
- ✅ findings 全面: core bx→excision receptor heterogeneity ✅

### ROW 54 (coral_idx 193) — 0 P1, 0 P2 ✅
- 39yo premenopausal female, BRCA2+。Oligometastatic ER+(90%)/PR+(10%)/HER2-(FISH 1.3) IDC L breast + T6 bone met。S/p neoadjuvant AC×4→paclitaxel×12wk + XRT to T6 + bilateral mastectomies+L ALND。Residual: 8.2cm grade 1 IDC (cellularity ~10%), ER+(50%), Ki-67 1%, margins neg, 1/24 SLN+ (0.15cm, no ENE)。R breast benign。Interval: incision healing, joint pain (letrozole), R foot neuropathy, hot flashes, weaning opioids+Ativan, hair thinning。Meds: leuprolide+letrozole+zoledronic acid。A/P: continue leuprolide+letrozole, Rad Onc referral (post-mastectomy RT), start palbociclib after RT, PET/CT 3-4mo, DEXA, acupuncture, Ca+vit D+exercise, RTC 4wk。
- ✅ v31 fix confirmed: procedure_plan "acupuncture" → "No procedures planned." ✅
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV (T6 met) ✅。Goals: palliative ✅。Mets: Yes, bone ✅
- ✅ current_meds: leuprolide+letrozole+zoledronic acid ✅。response_assessment: residual disease detailed ✅
- ✅ medication_plan 全面: leuprolide+letrozole+palbociclib(after RT)+zoledronic acid+Ca+vit D ✅
- ✅ imaging_plan: PET/CT 3-4mo+DEXA ✅。radiotherapy_plan: post-mastectomy RT ✅
- ✅ **Letter 出色** (grade 5.8): healing+pain+numbness + residual "shrunk a lot" + palbociclib after RT + letrozole+leuprolide + zoledronic acid "bones" + pain med + Ca+vit D + PET/CT + DEXA + 4wk F/U + emotional support

### ROW 57 (coral_idx 196) — 0 P1, 3 P2
- 59yo postmenopausal female。L breast locally advanced cancer, HER2 status controversy: core bx HER2+(3+)→neoadjuvant TCH+P×6 (dose reduced 25% C1)→lumpectomy+ALND→surgical specimen HER2 NEGATIVE (TNBC confirmed on review)。3.7cm residual, 0/6 nodes。Post-op AC×4。PMH: HTN, Crohn's (asacol)。Mother lung ca, sister breast ca @60。Multiple allergies。ECOG ~0。Second opinion visit。A/P: treatment appropriate given initial dx (TCH+P covers all TNBC drug classes)。Rec: additional path review at UCSF for HER2 (if HER2+ → resume trastuzumab), genetic counseling, XRT (scheduled)。
- P2: **procedure_plan "which pt is scheduled to receive"** — garbled XRT fragment，放疗非 procedure（known ❌）
- P2: **recent_changes "Dose reduction 25% after C1"** — 新辅助化疗期间历史事件，非本次就诊变化
- P2: **Letter 漏 pathology re-review 建议** — 第二意见核心建议（UCSF path review for HER2 question）未出现在 letter
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: locally advanced ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: 空 ✅（chemo 已完成，TNBC 无内分泌治疗）
- ✅ radiotherapy_plan: XRT scheduled ✅。genetic_testing_plan: genetic counseling ✅
- ✅ **Letter 较好**: TNBC explained + residual 3.7cm + 0/6 nodes + healing + LE swelling + neuropathy + dose reduction explained + XRT + genetic counseling

### ROW 59 (coral_idx 198) — 0 P1, 1 P2
- 52yo female。Stage 1 R sided ER+(>95%)/PR+(50%)/HER2-(equivocal IHC, neg FISH) IDC grade 3, 1.5cm+high grade DCIS。0/5 nodes, margins neg。Oncotype High Risk (-0.294)。S/p lumpectomy+SLN (11/16)。TC×3+Abraxane/Cytoxan×1 (docetaxel allergy)。XRT completed 05/17。Tamoxifen 05/17→stopped 12/17 (weight gain/arthralgias)→letrozole 03/18→worsened→exemestane rec'd 06/18 (patient unaware of recommendation)。Depression: Pristiq 75mg。PMH: asthma, hypothyroid, allergic rhinitis。Labs: Vit D 32, FSH 32.5, Estradiol 5 (postmenopausal), TSH 0.68。No evidence of recurrence。ECOG 1。A/P: D/C letrozole, wait 2-3wk→exemestane。Mammogram July+alternating MRI+q6mo breast exam。Psychiatry referral (Pristiq→duloxetine transition, drug interaction)。F/U 6mo。
- P2: **current_meds = "exemestane"** — 患者尚未开始 exemestane（"not aware of this recommendation"）。当前无 hormonal therapy
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: I ✅。Goals: curative ✅。Mets: No ✅
- ✅ response_assessment: no evidence of recurrence ✅。lab_summary 全面 (Vit D/FSH/E2/TSH) ✅
- ✅ medication_plan: D/C letrozole→exemestane + Pristiq + duloxetine consideration ✅
- ✅ imaging_plan: mammogram July + alternating MRI + q6mo exam ✅。Referral: psychiatry ✅
- ✅ **Letter 出色**: no recurrence + weight stable + letrozole→exemestane + Pristiq + duloxetine "joint pain" + psychiatry "safe" + mammogram/MRI + 6mo F/U + emotional support

### ROW 61 (coral_idx 200) — 0 P1, 1 P2
- 43yo premenopausal female。Newly diagnosed L breast IDC, ER+(100)/PR+(100)/HER2-(1+), grade 2, ≥11mm。Routine screening mammogram finding (no palpable lump)。MRI: 1.5cm mass 0630 + 0.6cm enhancement 0900 (repeat biopsy neg)。CT chest: slight fissure thickening LUL, no adenopathy。CT abd: liver cyst。Invitae: negative。Surgery scheduled 04/12/21: lumpectomy+IORT+reconstruction。Televisit (Zoom)。A/P: early stage IDC, lumpectomy+IORT (no post-op RT), Oncotype Dx after surgery for chemo decision, adjuvant endocrine (Tamoxifen vs OS+[redacted])。RTC after surgery+pathology。
- P2: **Letter "What's new" section 空** — 新患者但该段无内容（同 ROW 53 pattern）
- ✅ Stage 空但可接受（手术前无法确定，需 pathology）
- ✅ Type: ER+/PR+/HER2-(1+) IDC ✅。Goals: curative ✅。Mets: No ✅
- ✅ in-person: Televisit ✅。current_meds: 空 ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: adjuvant endocrine therapy (Tamoxifen vs OS+AI) ✅
- ✅ radiotherapy_plan: IORT (no post-op RT) ✅。procedure_plan: lumpectomy+IORT+recon 04/12 ✅
- ✅ findings 全面: biopsy+MRI+repeat bx+CT chest+CT abd ✅
- ✅ **Letter 较好**: IDC "milk ducts"+"sensitive to hormones" + endocrine therapy "prevent coming back" + surgery April 12 + IORT explained + "will not need additional radiation" + "review results and decide next steps"

### ROW 63 (coral_idx 202) — 0 P1, 1 P2
- 49yo female。Locally advanced L breast IDC, ER+(90-95%)/PR-(surgical)/HER2-(IHC 2+, FISH neg)。S/p neoadjuvant dd-AC×4→Abraxane→L partial mastectomy+SLN: 3.8×3.5cm grade 2 IDC, LVI+, positive margins, 3/4 LN (15mm macro+micromet+ITC, ENE)→re-excision (neg)→XRT L breast+nodes (11/26/19)。Letrozole since 10/04/19。Invitae neg。MRI dramatic response but disappointing surgical pathology (3.8cm residual)。Post-XRT skin changes。Arthralgias, hot flashes, sleep issues。Televisit second opinion。A/P: continue letrozole, E2/FSH q1-2mo monitoring, if premenopausal→OS/oophorectomy, DEXA, discuss abemaciclib (MonarchE+/PALLAS-), Mychart F/U。
- P2: **Referral.Others "Physical therapy referral"** — A/P 无 PT 转诊。Note 只说 "has not met with physical therapy"
- ✅ v31 fix confirmed: procedure_plan "labs" → "No procedures planned." ✅
- ✅ Type: ER+/PR-/HER2- IDC ✅。Stage: IIIA ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: letrozole ✅。response_assessment: stable on treatment ✅
- ✅ medication_plan 全面: letrozole+E2/FSH monitoring+OS/oophorectomy+DEXA+abemaciclib ✅
- ✅ lab_plan: E2/FSH q1-2mo ✅。imaging_plan: DEXA ✅
- ✅ **Letter 出色**: second opinion + IDC "milk ducts" + Stage IIIA + letrozole + E2/FSH "check post-menopausal" + OS/oophorectomy "ovaries removed or shots" + abemaciclib + DEXA + emotional support

### ROW 64 (coral_idx 203) — 0 P1, 3 P2
- 28yo premenopausal female, G2P2。L breast large IDC, ER+/PR+/HER2-。MRI: 10.3×4.5×3.5cm。Axillary bx positive。Bone scan: suspicious manubrium lesion→probable solitary bone met (bx planned)。Stage III-IV (oligo-metastatic)。Currently on dd-AC, taxol planned。Televisit second opinion。Anxious (cousin present)。Full code。Maternal aunt breast ca。A/P: continue AC→taxol→surgery→RT→treat sternum。If bone bx+→add xgeva。Check insurance for care transfer。
- P2: **Stage "IV- IV"** — garbled, should be "III-IV"
- P2: **current_meds 空** — 患者当前正在 dd-AC 化疗
- P2: **response_assessment "Not yet on treatment"** — 患者 IS on dd-AC
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative (oligo-metastatic, aggressive approach) ✅
- ✅ Mets: Yes, to sternum ✅。findings: tumor size+axillary met+bone scan ✅
- ✅ medication_plan: AC+taxol ✅。radiotherapy_plan: sternum treatment ✅。procedure_plan: sternal bx ✅
- ✅ Advance care: Full code ✅。second opinion: yes ✅

### ROW 65 (coral_idx 204) — 0 P1, 0 P2 ✅
- 48yo female。R breast IDC, ER weak+(2%)/PR low+(7%)/HER2-(IHC 2+, FISH 1.4), Ki-67 36%, grade 2-3。Focal LVI。High-grade DCIS。MRI: 2.6×1.9×2.3cm。R axillary LN: 0.2mm micromet, no ENE。PET/CT: SUV 5.1, no hypermetabolic LN, no distant mets。Televisit。RN, 3 children, BMI 29.7。A/P: locally advanced ER/PR low+/HER2- disease。Neoadjuvant: AC/T or ISPY trial (9 arms detailed)。After surgery: post-op RT + 5-10yr endocrine。Plan: TTE, port, chemo teaching, ISPY screening (research bx+MRI+labs), F/U genetic testing。RTC 1-2wk。
- ✅ v31 fix confirmed: response_assessment "On treatment" → "Not yet on treatment" ✅
- ✅ 全字段准确：Type(ER/PR low+/HER2-) ✅, Stage II ✅, Goals curative ✅, Mets No ✅
- ✅ medication_plan 极全面: AC/T or ISPY + post-op endocrine 5-10yr ✅
- ✅ procedure_plan: port+research bx+lumpectomy ✅。imaging_plan: TTE ✅
- ✅ **Letter 出色**: IDC "milk ducts" + LN "under arm" + neoadjuvant "shrink before surgery" + clinical trial + port+bx+lumpectomy+RT+endocrine 5-10yr + TTE "check heart" + genetic test + 1-2wk F/U

### ROW 66 (coral_idx 205) — 0 P1, 4 P2
- 53yo female。R breast metaplastic carcinoma with squamous differentiation, ER 5-10%(weak+)/PR 0%/HER2 0%。Treated as TNBC (ER<10%)。Grade III (3+3+3)。Two sites: R1 7:00 (3.8cm MRI) + R2 6:00 (1.4cm MRI)。Skin thickening/enhancement 9-10:00 (indeterminate)。R axillary LN bx: NEGATIVE。CT: no distant disease, pulmonary nodules ≤4mm, thyroid nodule 4mm。BRCA neg。Invitae germline pending。Adopted (no FHx)。Afro-Caribbean + Ashkenazi Jewish ancestry。ECOG 0。Second opinion (Experimental Therapeutics/Phase I)。A/P: neoadjuvant [redacted]→bilateral mastectomy considered→adjuvant RT。Discussed pembrolizumab (TNBC)。Delay reconstruction until after RT。Continue Tx with local oncologist。
- P2: **medication_plan "None"** — A/P 明确 neoadjuvant chemo planned + pembrolizumab discussed
- P2: **imaging_plan "Mammogram"** — A/P 未下新 mammogram（fabricated）
- P2: **genetic_testing_plan garbled** — "germline, germline testing, invitae" 应为 "Awaiting Invitae germline panel results"
- P2: **Letter "What's new" section 空** — 同 ROW 53/61 pattern
- ✅ v31 fix confirmed: RA "On treatment" → "Not yet on treatment" ✅
- ✅ Type: metaplastic carcinoma with squamous differentiation ✅。Mets: No ✅。Goals: curative ✅
- ✅ findings 极全面: biopsy+MRI+CT+PE ✅。second opinion: yes ✅
- ✅ therapy_plan: neoadjuvant+mastectomy+RT+pembrolizumab ✅
- ✅ **Letter 部分出色**: metaplastic "rare type where cells look like skin cells" + HER2 "protein" + neoadjuvant "shrink before surgery" + mastectomy considered + RT + reconstruction delay + genetic test

### ROW 68 (coral_idx 207) — 0 P1, 1 P2
- 63yo postmenopausal female。Multifocal ER+/PR+/HER2+ R breast cancer with BRCA mutation。Mother died breast ca @41。Baseline MRI: 3.9cm+1.5cm+2.2cm masses R breast, multiple R axillary LN + 0.7cm R IM LN。S/p 6 cycles TCHP (neoadjuvant)。Follow-up MRI: NO LESIONS (complete clinical response!)。PMH: HTN。PE: no palpable lesions, facial rash。A/P: bilateral mastectomy recommended (disease extent+BRCA)。If lumpectomy→RT; post-mastectomy RT if extensive residual/LN。Sons need BRCA testing (Fanconi anemia risk + pancreatic ca @50)。Diet/exercise。RTC as needed。
- P2: **Patient type "New patient"** — Note 明确 "return visit"/"presents for a return visit"，应为 "Follow up"
- ✅ Type: ER+/PR+/HER2+ multifocal ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: complete clinical response (no lesions on follow-up MRI) ✅
- ✅ procedure_plan: bilateral mastectomy ✅。radiotherapy_plan: conditional (lumpectomy→RT) ✅
- ✅ genetic_testing_plan: sons should be tested ✅
- ✅ **Letter 较好**: ER+/PR+/HER2+ multifocal + "responding well, no signs of cancer" + bilateral mastectomy explained + lumpectomy explained + RT conditional + sons testing + return as needed

### ROW 70 (coral_idx 209) — 0 P1, 4 P2
- 61yo postmenopausal female, BRCA1+。BILATERAL breast cancer: Left ILC ER+(>95%)/PR+(2-5%→90% residual)/HER2-; Right IDC ER+(95%)/PR-(<1%)/HER2 equivocal→likely neg。Oncotype High Risk (L)。S/p neoadjuvant TC×4→Abraxane/cyclophosphamide×2 (docetaxel→eye toxicity)。S/p bilateral mastectomies + preventive BSO (BRCA1, benign) May 2020。Post-op: L 4.4cm residual ILC G2 (5-10% cellularity), 2/5 SLN+ (macro+micro), neg margins; R 1cm IDC G2, 0/2 SLN neg, neg margins。Both with treatment effect。PMH: Graves, anxiety/depression (sertraline), osteoporosis, asthma。Labs: CBC+CMP normal, Lymph 0.73L。ECOG 0。Full code。A/P: expanders→radiation→restart letrozole→Prolia after dental clearance→CT lung nodules June→RTC September。
- P2: **Stage "pT4 N1 on left side"** — Left residual 4.4cm = ypT2 (>2-5cm)，pT4 需 chest wall/skin invasion
- P2: **procedure_plan "No procedures planned"** — A/P 明确 expander placement planned
- P2: **medication_plan 漏 Prolia** — A/P "start prolia after dental clearance" for osteoporosis
- P2: **Letter 过于临床化** — "non-mass enhancement""scintigraphic""T12 superior endplate""sub-4mm pulmonary nodules" 非通俗语言
- ✅ Type: bilateral ILC(L)+IDC(R) with different receptor profiles ✅。Mets: No ✅。Goals: curative ✅
- ✅ current_meds: letrozole (restarting) ✅。response_assessment: responding (MRI improvement) ✅
- ✅ lab_summary 全面 ✅。imaging_plan: CT lung nodules June ✅。Specialty: Radiation consult ✅
- ✅ Advance care: Full code ✅。Next visit: September ✅

### ROW 72 (coral_idx 211) — 0 P1, 2 P2
- 72yo postmenopausal female。L breast IDC with focal neuroendocrine differentiation, grade 2 (core bx grade 1→excision grade 2), 1.2cm。ER+(99%)/PR-(<1%)/HER2-(IHC 1+, FISH 1.0)。Ki-67 20%。0/2 SLN。No LVI。pT1cN0(sn)。S/p L mastectomy+SLN (03/03/22)。Televisit。PMH: osteoporosis (Reclast), AR, cataracts, glaucoma, subdural hematoma, syncope。Cousin breast ca @33。A/P: begin letrozole ≥5yr (AI>tamoxifen)。Order Oncotype (Ki-67 20%+PR-+grade upgrade)。RTC 3wk review Oncotype。
- P2: **current_meds = "letrozole"** — 刚开处方尚未服用
- P2: **response_assessment "On treatment"** — 刚完成手术+letrozole 刚开。应为 "Not yet on treatment"
- ✅ Type: ER+/PR-/HER2- IDC with neuroendocrine differentiation ✅。Stage: pT1cN0 ✅。Goals: curative ✅
- ✅ genetic_testing_plan: Oncotype ordered ✅。Next visit: 3wk ✅
- ✅ **Letter 清晰通俗** (grade 5.5): IDC "milk ducts" + ER/PR/HER2 explained + early stage + letrozole "prevent coming back" + "test to see if you might benefit from chemotherapy" + 3wk F/U

### ROW 73 (coral_idx 212) — 0 P1, 0 P2 ✅
- 63yo postmenopausal female。Stage III L breast cancer ER/PR+/HER2-。S/p bilateral mastectomies+L ALND+TC/abraxane+CW XRT+arimidex (Aug 2017)。Implant reconstruction (exchanged May 2018)。Follow-up。New nodule L breast→today bilateral breast US+mammogram: all 3 areas = **fat necrosis** (not cancer)。R breast 1cm nodule 7OClock + L breast 2cm lump 1-2OClock (previously 3cm) + new linear nodularity 7-9OClock = all fat necrosis。Insect bite rash (resolving, clobetasol)。Labs/markers Aug 2018 neg。PET-CT Nov 2017 neg。Genetics VUS only。ECOG 0。A/P: continue arimidex, continue [redacted] (Aug 2019), check labs, RTC 4mo。
- ✅ Type: ER+/PR+/HER2- ✅。Stage: III ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: arimidex ✅。response_assessment: fat necrosis not recurrence ✅
- ✅ medication_plan: continue arimidex ✅。lab_plan: check labs ✅。Next visit: 4mo ✅
- ✅ **Letter 出色**: new lump + smaller than before + insect bite resolving + "fat necrosis, which is not cancer" + continue arimidex + blood tests + 4mo F/U。简洁通俗

### ROW 78 (coral_idx 217) — 0 P1, 1 P2
- 79yo female。De novo PD-L1(-) metastatic TNBC to liver+periportal LNs (dx 07/2017)。L breast 4.2cm grade 3 IDC, ER0/PR0/HER2-, Ki-67 60%。S/p L partial mastectomy+ALND (pT2N1a)+re-excision。S/p capecitabine×10→OPERA trial→gemcitabine×4 (d/c'd: fatigue+pericardial effusion)。Off systemic therapy since 03/15/19。08/2019 CT: WORSENING — liver lesions enlarging, portocaval LN 2.9cm(from 1.7cm), new 7mm RML nodule (suspicious met)。PMH: DM/HTN/hyperlipidemia/hemorrhagic brainstem CVA (cavernous malformation)。Cr 1.27H, eGFR 40-46, Glucose 152。Patient prefers clinical trial over chemo。Rad Onc consult tomorrow (XRT liver/periportal)。Echo 09/08 (pericardial effusion)。
- P2: **medication_plan lists non-cancer meds** — "Mag-Ox, lisinopril, norvasc" instead of cancer treatment options (trial/doxil/eribulin discussed)
- ✅ Type: TNBC ✅。Stage: IV ✅。Mets: Yes, liver+periportal LN ✅。Goals: palliative ✅
- ✅ current_meds: 空 ✅ (off all cancer therapy since 03/15/19)
- ✅ response_assessment: worsening detailed (liver+LN enlarging+new lung nodule) ✅
- ✅ lab_summary 全面: CBC+CMP including Cr 1.27H, eGFR 40-46, Glucose 152 ✅
- ✅ therapy_plan: trial interest (phase 1+pembrolizumab, phase 3 eribulin, doxil option) ✅
- ✅ radiotherapy_plan: Rad Onc consult for liver/periportal XRT ✅。imaging_plan: Echo 09/08 ✅
- ✅ **Letter 较好**: progression explained + "new spot in chest might be cancer" + Rad Onc referral + echo + clinical trial interest + dentist F/U

### ROW 80 (coral_idx 219) — 0 P1, 3 P2
- 53yo female。Local recurrence IDC in dermis 7yr post bilateral mastectomy for DCIS (2012→2019)。0.8cm grade 3 IDC, ER+(95%)/PR+(70%)/HER2-(IHC 1+, FISH not amplified), Ki-67 15-20%。Close margin。Wide excision benign。Oncotype 24。PET/CT: left renal cyst only。Genetic testing neg。Tempus: no actionable mutations。Televisit。Mother had bilateral DCIS @68。A/P: TC×4 start 04/11/19 (cold cap+cold gloves+Claritin)。6wk RT (5wk+1wk boost, L axilla+SC fields)。Labs normal。RTC cycle 2。
- P2: **Patient type "New patient"** — Note says "return visit"/"follow-up consult"
- P2: **Type "HER2: not tested"** — HER2 WAS tested: IHC 1+, FISH not amplified = HER2 negative
- P2: **procedure_plan "with [REDACTED]" garbled** — known v30 residual ❌
- ✅ v31 fix confirmed: RA "On treatment" → "Not yet on treatment" ✅
- ✅ Mets: No ✅。Goals: curative ✅。Stage: local recurrence (no primary staging)
- ✅ lab_summary 全面 (CBC+CMP) ✅。medication_plan: TC×4+Claritin+cold cap ✅
- ✅ radiotherapy_plan: 6wk RT L axilla+SC fields ✅。Specialty: Rad Onc ✅
- ✅ **Letter 较好**: IDC "milk ducts" + ER/PR "sensitive to hormones" + labs normal + TC start date + side effect meds + Claritin+cold gloves + 6wk RT "armpit and skin areas" + emotional support

### ROW 82 (coral_idx 221) — 0 P1, 1 P2
- 52yo postmenopausal female。R breast mixed ductal+lobular carcinoma, 4.3cm, G2-3, 1/24 LN+, ER+/PR+/HER2-, Ki-67 15%。Neg margins。S/p R lumpectomy+SLN (11/16/20)。Oncotype Low Risk→no chemo。Stage IB (pT2 N1mi(sn))。PMH: DM2/HTN/hyperlipidemia/GERD/liver enzyme elevation/vit D def/anxiety。ECOG 0。Full code。A/P: no chemo。RT referral (appt tomorrow)。AI +/- bone medication after RT。DEXA ordered。Exercise counseling。Lifestyle (diet/exercise/stress/sleep)。RTC after RT。
- P2: **medication_plan lists non-cancer meds** — HCTZ/lisinopril/metformin instead of cancer treatment plan (AI+bone medication)
- ✅ Type: ER+/PR+/HER2- mixed ductal+lobular ✅。Stage: II ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: 空 ✅。response_assessment: "Not yet on treatment" ✅
- ✅ imaging_plan: DEXA ✅。radiotherapy_plan: RT referral ✅。Advance care: Full code ✅
- ✅ Referral.Others: Exercise counseling ✅。Next visit: after radiation ✅

### ROW 83 (coral_idx 222) — 0 P1, 1 P2
- 77yo female。R breast invasive lobular carcinoma, grade I, ER+(3+/100%)/PR+(3+/10%)。R axillary LN metastatic。Extensive R axillary/subpectoral adenopathy。Workup: bone scan neg, CT no distant mets。Neoadjuvant letrozole since Dec 2019。PET/CT 04/25/2020: **significant response** — R axillary SUV 15.1→1.9。No active skeletal metastatic disease。PMH: HTN/hypothyroidism/anxiety-depression/aortic stenosis/rosacea。ECOG 1。Televisit。A/P: continue neoadjuvant letrozole→breast surgery (interested in conservation)→reassessment after surgery。
- P2: **Patient type "New patient"** — Note says "Last visit December 2019"，应为 "Follow up"
- ✅ Type: lobular carcinoma ER+/PR+ ✅。Stage: III ✅。Mets: Yes, R axilla LN ✅。Distant: No ✅
- ✅ current_meds: letrozole ✅。response_assessment: substantially improved ✅。Goals: curative ✅
- ✅ medication_plan: continue neoadjuvant letrozole ✅。procedure_plan: breast surgery ✅
- ✅ **Letter 出色**: lobular "milk-producing glands" + "improved with treatment" + letrozole + breast surgery + F/U after surgery + emotional support

### ROW 84 (coral_idx 223) — 0 P1, 1 P2
- 60yo female, CHEK2 biallelic mutation, MS (1985, wheelchair since 2002, Avonex)。Metastatic ER+(71-80%)/PR-(<1%)/HER2-(IHC 2+ equivocal, FISH 1.3=NEG) breast cancer to bone (extensive lytic/blastic, pathologic C2 fracture)+soft tissue (R axillary/mediastinal LN)+liver (multiple, progressing)+possibly meninges (dural enhancement, CN VI palsy, hearing loss R ear)。Hx: 1999 lumpectomy→2006 R mastectomy+CAF×6+5yr tamoxifen→2019 metastatic: letrozole+palbociclib (12/19-07/20, PD)→capecitabine (08/20)。Current: capecitabine 1500mg BID+zolendronic acid。Labs: Hgb 8.4, Plt 80。ECOG 2。Televisit。A/P: repeat CT CAP+LP cytology+MRI spine (r/o LMD)+Rad Onc (focal CNS RT)+consider steroids+continue xeloda+zolendronic acid。At PD→fulvestrant+[redacted] ([redacted] mutation)。
- P2: **Type "HER2 equivocal"** — FISH ratio 1.3 = HER2 NEGATIVE (FISH resolved equivocal IHC)
- ✅ Stage: IV ✅。Mets: Yes, bone+liver+LN+possibly meninges ✅。Goals: palliative ✅
- ✅ current_meds: capecitabine+zolendronic acid ✅。lab_summary with Hgb 8.4, Plt 80 ✅
- ✅ medication_plan: continue xeloda+steroids+zolendronic acid ✅。procedure_plan: repeat LP ✅
- ✅ imaging_plan: CT CAP+MRI spine ✅。radiotherapy_plan: focal CNS RT ✅。Specialty: Rad Onc ✅
- ✅ therapy_plan 极其全面: all recommendations captured ✅

### ROW 85 (coral_idx 224) — 0 P1, 2 P2
- 61yo female。ER+/PR-/HER2- ILC R breast, originally Stage IIIA。Extensive Tx: bilateral mastectomy→DD AC→paclitaxel→exemestane→XRT→metastatic to bone+muscle+brain (1yr post-chemo)→RT spine→GK brain→fulvestrant+palbociclib→PD: new **liver** mets+increased bone→palliative RT。Foundation One: FGFR1 amp, TMB 14, BRCA2 VUS。Brain: leptomeningeal+Meckel's cave→CN V (facial numbness/headache)→prednisone+morphine。CA 15-3 trending up (45→360)。Currently no active cancer therapy。A/P: phase 1 trial [redacted]+olaparib。Rad Onc (brain)。Steroid taper+pain meds。F/U 2wk。
- P2: **Metastasis 漏 liver** — "Yes, to bone, muscle, and brain" 但 liver mets 是关键 PD 事件
- P2: **Letter "steroid dose was decreased to day"** — garbled text
- ✅ Type: ER+/PR-/HER2- ILC ✅。Stage: IIIA→IV ✅。Goals: palliative ✅
- ✅ response_assessment: PD on fulvestrant+palbociclib, new liver mets ✅
- ✅ therapy_plan: phase 1 trial+olaparib ✅。radiotherapy_plan: Rad Onc brain ✅。F/U: 2wk ✅
- ✅ **Letter 内容好（除 garble）**: brain "cancer that has spread" + headaches/numbness + olaparib "certain changes in cancer cells" + brain MRI + radiation doctor + 2wk F/U

### ROW 86 (coral_idx 225) — 0 P1, 1 P2
- 53yo female。R breast mixed IDC/ILC, grade III。Original HER2+(FISH 4.37)→neoadjuvant TCHP×6→bilateral MRM (3 masses, 4/15 LN+)→XRT→adjuvant [redacted]。2019 metastatic: bone+liver+?brain dural。Metastatic bx L pelvis: ER+(95%)/PR+(2%)/HER2-(IHC 1+, FISH neg)=**RECEPTOR CONVERSION**。CHEK2 mutation。Letrozole+ribociclib→PD (increasing bone mets 04/2020)。Televisit。ECOG 1。A/P: fulvestrant+/-everolimus。Continue denosumab。Palliative XRT cervical spine+L mandible。F/U 6wk。
- P2: **Type "HER2+"** — Metastatic bx shows HER2-(IHC 1+, FISH neg)。Receptor conversion from original HER2+
- ✅ Stage: IV ✅。Mets: Yes, bone+liver ✅。Goals: palliative ✅
- ✅ current_meds: letrozole+ribociclib+denosumab ✅。response_assessment: PD on letrozole+ribociclib ✅
- ✅ medication_plan: fulvestrant+/-everolimus+continue denosumab ✅。radiotherapy_plan: palliative XRT ✅
- ✅ **Letter 出色**: mixed IDC/ILC + bone+liver + progression + palliative goals + fulvestrant+everolimus + denosumab "bone health" + XRT + 6wk F/U

### ROW 87 (coral_idx 226) — 0 P1, 0 P2 ✅
- 79yo female。R breast IDC grade 2, 2.2cm multifocal (separate 0.6cm incidental adenoCA), ER+/PR+/HER2-。S/p excisional bx+R ALND: 4/19 LN+ (ENE 1 node 0.5cm)。Clear margins。PMH: Parkinson's disease (R side), mild HTN。Family: daughter breast+CRC @40, GM ovarian, mother leukemia。WHI estrogen-only arm。Second opinion。A/P: high risk ~40-45%。Hormonal therapy only (chemo adds 3-4% but risk-benefit unfavorable with age+Parkinson's)。Return to local oncologist。
- ✅ **Stage "IIIA (pT2 N2a)" 正确** — 4/19 LN+ = N2a ✅ (v31 Stage 推断正确!)
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。Mets: No ✅。second opinion: yes ✅
- ✅ medication_plan: hormonal therapy alone ✅。radiotherapy_plan: discussed ✅

### ROW 88 (coral_idx 227) — 0 P1, 1 P2
- 36yo female。Stage III L breast IDC grade III, HR weak+/HER2-。S/p neoadjuvant AC→Taxol→Taxol/carbo (PD)→bilateral mastectomies+L ALND (2 tumors, 23/30 LN+, ENE+)→gemzar+carbo×4→XRT。Brain mets (2)→resection 1+SRS both (brain met ER-/PR-)。Lung+LN mets。On xeloda。COVID-19 diagnosed。Genetic testing neg。Full code。ECOG 0。Televisit。A/P: need HER2 on brain met+residual disease。Continue xeloda, restage 3mo。If PD→immunotherapy。F/U PRN。
- P2: **radiotherapy_plan 写过去治疗** — "had stereotactic XRT" 是历史，无新 RT planned
- ✅ Type: ER+/PR+/HER2- IDC, metastatic biopsy ER-/PR- (receptor discordance) ✅。Stage: III→IV ✅
- ✅ Mets: Yes, brain+lungs+LN ✅。Goals: palliative ✅。current_meds: xeloda ✅
- ✅ genetic_testing_plan: repeat HER2 on brain met+residual ✅。Advance care: Full code ✅

### ROW 90 (coral_idx 229) — 0 P1, 2 P2
- 51yo female。R breast adenocarcinoma, Clinical stage II/III。ISPY trial。S/p Taxol→R lumpectomy (2.2cm residual, 60% cellularity)。Oncotype High Risk。Currently cycle 3/4 AC (adjuvant)。BLM gene carrier。Adopted。PMH: allergies/arthritis/GERD。Labs: TSH 6.01H (hypothyroidism), ALP 125H, WBC 11.8H, Hgb 11.3L。Port extravasation (reinserted)。Neuropathy resolved。Televisit。A/P: cycle 4 AC 1wk (delay), reduce GCSF 50%, granisetron+dex+olanzapine for N/V, gabapentin, RTC after XRT。
- P2: **Patient type "New patient"** — 明确 Follow-up (on cycle 3 AC, ISPY trial)
- P2: **Letter truncated** — "Sincerely," missing "Your Care Team"
- ✅ Stage: II/III ✅。Mets: No ✅。Goals: curative ✅
- ✅ lab_summary 全面: TSH+CMP+CBC ✅。medication_plan 全面: AC+GCSF+antiemetics ✅
- ✅ **Letter 内容好**: adenocarcinoma + TSH/blood tests explained + GCSF/granisetron/dex/olanzapine + cycle 4 delay + RT after + emotional support

### ROW 91 (coral_idx 230) — 0 P1, 3 P2
- 53yo female。Stage 4 (originally Stage I 2003: R breast 1.3cm IDC ER+/PR+/HER2-, 0/4 LN)。2005: bone mets PD。Extensive Tx history: multiple XRT, letrozole, zoladex, BSO, fulvestrant, phase I trials, then exemestane+everolimus since 04/2012+denosumab monthly。MRI/PET 2011: increasing bone disease (pre-current therapy)。Current: RLE edema improved (lasix), 1cm R iliac LN (unclear significance), fungal dermatitis。Labs: AST 60H。A/P: continue exemestane+everolimus+denosumab。PET/CT next week。Labs monthly。Lasix+KCL。Topical antifungal。RTC 1mo。
- P2: **Type "HER2: not tested"** — Note says "*****-" = HER2 negative
- P2: **response_assessment "not responding"** — Pre-therapy imaging conflated with current; A/P ordering PET to evaluate
- P2: **medication_plan misses cancer therapy** — Lists lasix/denosumab but not exemestane+everolimus (core cancer Tx)
- ✅ Stage: I→IV ✅。Mets: Yes, bone ✅。Goals: palliative ✅
- ✅ current_meds: everolimus+exemestane+denosumab ✅。imaging_plan: PET/CT next week ✅
- ✅ lab_plan: labs monthly ✅。Next visit: 1 month ✅

### ROW 92 (coral_idx 231) — 0 P1, 2 P2
- 67yo female。Metastatic breast cancer to multiple sites (liver+bone)。Original 1991: ER+/HER2-, 7/7 LN+。2003 recurrence: ER%/PR-。2011 liver mets: ER+(60%)/HER2-/PR-。>10 lines of therapy。Currently epirubicin 25mg/m2 D1,8,15 cycle 2 + Neupogen + denosumab。Labs: CA 27.29=3332!!!, CEA 380.8!!, Albumin 2.8L, AST 72H, ANC 1.70L。ECOG 1。A/P: continue epirubicin, liver improving on exam, labs+tumor markers, okay to proceed。
- P2: **Type "ER+/PR+/HER2-"** — Metastatic bx 和 2003 recurrence 都是 PR-。应为 ER+/PR-/HER2-
- P2: **procedure_plan = "8"** — garbled (known v30 residual ❌ — chemo "8" from D1,8,15)
- ✅ Stage: IV ✅。Mets: Yes, multiple sites ✅。Goals: palliative ✅
- ✅ current_meds: epirubicin+denosumab ✅。lab_summary 全面 (CA 27.29+CEA+CBC+CMP) ✅
- ✅ response_assessment: stable, liver improved on exam ✅

### ROW 94 (coral_idx 233) — 0 P1, 0 P2 ✅
- 75yo female。L breast cancer 1.6cm, 3 LN+, ER+/PR+/HER2-, grade 2。Oncotype RS 21 (no chemo)。S/p lumpectomy+radiation+letrozole。Genetic testing neg。Mammogram 12/2020 normal。Joint pain from AI (CBD)。Televisit。ECOG 0。Full code。A/P: continue letrozole, mammogram Nov 2021, high risk MRI, bone protection, RTC 6mo。
- ✅ Type: ER+/PR+/HER2- ✅。Stage: IIA (pT1b pN1 G2) ✅。Goals: curative ✅。Mets: No ✅
- ✅ current_meds: letrozole ✅。response_assessment: no recurrence ✅。Advance care: full code ✅
- ✅ imaging_plan: mammogram Nov + high risk MRI ✅。Next visit: 6mo ✅

### ROW 95 (coral_idx 234) — 0 P1, 2 P2
- 49yo female。L breast ER+/PR+/HER2- IDC with papillary features。ISPY trial (Pembrolizumab+[redacted])→T→AC→L lumpectomy: 3 foci residual IDC (largest 0.9cm, 20% cellularity), margins neg, 1/6 SLN+ (0.9cm macro, no ENE)。Residual receptors: ER+(>95%)/PR-(0%)/HER2 equivocal(2+), Ki-67 5%。A/P: breast+axilla XRT (preferred over ALND), then capecitabine (CREATE-X), then adjuvant endocrine。RTC after XRT。
- P2: **Stage "IIA (pT2 N1a)"** — Post-neoadjuvant应用 ypT（0.9cm residual=ypT1，非pT2）
- P2: **imaging_plan "breast and axilla XRT"** — XRT 是放疗非影像检查
- ✅ Type: residual IDC ER+/PR-/HER2 equivocal ✅。Goals: curative ✅。Mets: No ✅
- ✅ medication_plan: prilosec+capecitabine after XRT ✅。radiotherapy_plan: breast+axilla XRT ✅。Specialty: Rad Onc ✅

### ROW 97 (coral_idx 236) — 0 P1, 1 P2
- 53yo female with relapsing-remitting MS (fingolimod/GILENYA)。L breast grade 1 IDC, 0.8cm, ER+(>95%)/PR+(~60%)/HER2-(IHC 1+), Ki-67 10%。S/p L partial mastectomy+SLN: margins neg, 0/3 SLN。pT1bN0(sn)。Current smoker。Hx crack cocaine (stopped 3yr)。PMH: MS/anxiety/depression/insomnia。Televisit。A/P: low risk, Oncotype Dx ordered。Rad Onc eval referred。Strongly recommend adjuvant AI。No problem GILENYA+AI。Drain management。RTC 3-4wk after Oncotype。
- P2: **medication_plan focuses on GILENYA** — 漏了 planned adjuvant endocrine therapy (AI)
- ✅ Type: ER+/PR+/HER2- grade 1 IDC ✅。Stage: pT1bN0 ✅。Goals: curative ✅。Mets: No ✅
- ✅ genetic_testing_plan: molecular profiling ✅。Specialty: Rad Onc ✅
- ✅ **Letter 出色**: IDC "milk ducts" + early stage + surgery explained + "no cancer in lymph nodes" + GILENYA continuation + "molecular profiling to learn more" + Rad Onc + emotional support

### ROW 100 (coral_idx 239) — 0 P1, 2 P2
- 68yo female。Metastatic breast cancer to multiple sites (liver+skin+bone)。Original 2002: ER+(80%)/PR+(50%)/HER2-, 3.0cm grade 2 IDC, 10/10 LN+。2011 skin+liver recurrence (receptor conversion to TNBC)。>10 lines of therapy。Currently gemcitabine cycle 2 Day 8 cancelled by patient (fatigue)。Labs: CA 15-3 118H, CA 27.29 178H, CEA 312H, ALP 172-196H, AST 49-63H, Hgb 9.6-9.9L。ECOG 1。A/P: unclear PD vs tumor flare。Rec exercise+Focalin+continue treatment。Patient wants break。
- P2: **therapy_plan "None"** — A/P recommends "continue with treatment"
- P2: **Letter "Gemzar stopped"** — 治疗未停止，仅本次 day 8 cancelled by patient，计划继续
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV ✅。Mets: Yes, liver+multiple ✅。Goals: palliative ✅
- ✅ current_meds: gemzar ✅。lab_summary 极其全面 (tumor markers+CBC+CMP) ✅
- ✅ response_assessment: "unclear if PD or tumor flare" ✅ — 准确传达不确定性
