# V30 Full Run Review (61 samples)

> Run: v30_full_20260413_101511
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v30) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-9 完成 (8/61)，ROW 10 待审查**
> Results 文件: `results/v30_full_20260413_101511/results.txt`

### v30 改进（相对 v29）
1. Response_Assessment: 新增"刚开处方≠On treatment" + A/P评估优先于旧影像
2. Cancer_Diagnosis: 最新受体状态优先 + redacted HER2推断规则
3. Letter: [REDACTED] garbled text跳过 + 事实性准确(HER2是蛋白不是药物)
4. Procedure/Imaging Plan: 字段混入严格检查 + imaging不含XRT/推测

### 全量 ROW 列表（61 个）
ROW: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 20, 22, 27, 29, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 46, 49, 50, 52, 53, 54, 57, 59, 61, 63, 64, 65, 66, 68, 70, 72, 73, 78, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97, 100

### v29→v30 对比目标
- v29: P0=0, P1=0, P2=92
- v30 目标: P2 < 60（减少 ~35%）
- 重点关注: response_assessment "On treatment" 问题是否修复, 受体状态优先级, letter garbled text, procedure/imaging 字段混入

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 0 | — | |
| **P2** | 10 | — | ROW 1×2, 2×1, 3×1, 5×0, 6×2, 7×2, 8×2, 9×0 (ROW 10+ 待审查) |

---

## 逐 Sample 问题清单

### ROW 1 (coral_idx 140) — 0 P1, 2 P2
- 56yo, Stage IIA→IV ER+/PR+/HER2- IDC。2013 mastectomy, declined tamoxifen。Now metastatic to lungs/peritoneum/liver/ovary + axillary recurrence。Biopsy planned。Brain MRI + bone scan ordered。If HR+/HER2- → ibrance+[letrozole]。Integrative Medicine referral。Full code。ECOG 0。
- P2: imaging_plan "Brain MRI" 漏了 bone scan（A/P 明确 "MRI of brain and bone scan"）— 同 v29
- P2: lab_plan 混入了 imaging（"MRI of brain and bone scan as well as labs"）— 同 v29
- **v30 改进效果**:
  - ✅ response_assessment: "cancer is currently progressing" + 具体影像证据 — v29 没这么具体
  - ✅ Letter [REDACTED] 处理: "ibrance and another medication" 而非 garbled text
  - ✅ Letter HER2: "does not have a protein called HER2" — 正确描述为蛋白
  - ✅ procedure_plan: 只有 biopsy（真正的 procedure），无 chemo 混入
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IIA→IV ✅。Goals: palliative ✅
- ✅ findings: 非常详细 — CT 全部发现 + 体检（hepatomegaly, omental masses, axilla mass 3cm）
- ✅ Advance care: Full code ✅。Referral: Integrative Medicine ✅
- ✅ Letter 逐句(16句): first visit + IDC "milk ducts" + ER/PR/HER2 "protein" + metastases listed + peritoneum "lining of abdomen" + palliative "feel better, live longer" + biopsy "armpit" + brain MRI + bone scan + Integrative Medicine + full code + emotional support。通俗准确

### ROW 2 (coral_idx 141) — 0 P1, 1 P2
- 44yo, Lynch Syndrome + colon ca (Stage I) + endometrial ca + metastatic TNBC Stage IIB→IV。Mets to liver/bone/chest wall。S/p neoadjuvant + carboplatin/paclitaxel (PD) + abraxane/pembrolizumab (PD) → now irinotecan C3D1。Severe: chest wall infection, sacral pain (S1 fracture), anemia Hgb 7.7, Na 124 (LL), K 3.1 (L)。Confused。Hep B prior exposure (HBsAb+, HBV DNA neg)。ECOG 1。
- P2: Letter 被截断 — 结尾 "You will have a follow-up visit" 断了，缺少 "in 2 weeks" 和 closing ("Sincerely, Your Care Team")。**v30 regression**（v29 没有此问题）
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: IIB→IV ✅。Goals: palliative ✅
- ✅ response_assessment: "cancer is not responding well" + 胸壁恶化 + 背痛加重 + 贫血 + 电解质紊乱 + MRI 骨转移 — 详细的临床证据
- ✅ current_meds: Irinotecan ✅。recent_changes: dose/schedule change ✅
- ✅ supportive_meds: 5 drugs listed ✅。medication_plan: 极其全面（6 items）✅
- ✅ imaging_plan: scans 3 months + MRI brain if worse ✅。lab_plan: HBV monitoring q4mo ✅
- ✅ lab_summary: 全面（列出所有异常值 Albumin 2.1L, ALP 183H, Na 124LL, K 3.1L, Hgb 7.7L 等）
- ✅ findings: 极其详细 — 胸壁感染 + 骨转移MRI + Hep B + neuropathy + cellulitis + 体检 + labs ✅
- ✅ Referral: Rad Onc + Social work + Home health ✅
- ✅ procedure_plan: "No procedures planned" ✅（无 chemo 混入 — v30 字段改进确认）
- ✅ Letter（除截断外）: 极其全面 — chest wall infection + back pain PD + labs explained + anemia "tired" + Hep B + neuropathy improved + irinotecan change + doxycycline + effexor + potassium + Rad Onc + scans + MRI brain + HBV monitoring。通俗准确

### ROW 3 (coral_idx 142) — 0 P1, 1 P2
- 53yo postmenopausal, Stage IIA R breast IDC 1.7cm, LN+, ER+/PR+/HER2-(IHC 2+, FISH neg), Ki-67 30-35%。Multiple opinions (second opinion)。PET + Oncotype pending。Genetic testing sent。Pre-diabetes。Video consult。Full code。ECOG 0。
- P2: Letter 写 "after the results of the PET scan and a medication are back" — [REDACTED] Oncotype Dx 被误解为 "a medication"。Oncotype 是检测不是药物。v30 [REDACTED] handling 未完全生效
- ✅ second opinion: yes ✅。Type: ER+/PR+/HER2- IDC ✅。Stage: IIA ✅
- ✅ response_assessment: "Not yet on treatment — no response to assess." ✅ — **v30 改进确认！** 正确识别 pre-treatment
- ✅ Goals: curative ✅。current_meds: empty ✅。Advance care: full code ✅
- ✅ genetic_testing_plan: "sent and pending" ✅。imaging_plan: PET follow-up ✅
- ✅ procedure_plan: "No procedures planned" ✅（无混入）
- ✅ Letter（除"medication"外）: IDC "milk ducts" + HER2 "protein" + neoadjuvant "treatments before surgery to shrink cancer" + PET + genetic testing + chemo discussed + full code + emotional support。通俗

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅
- 31yo premenopausal, Stage III→IV ER+/PR+/HER2- IDC left breast。Metastatic recurrence to cervical LN + brachial plexus + possible sternal bone met。On anastrozole + palbociclib + leuprolide。Televisit follow-up。Full code。ECOG 1。
- ✅ Type: "ER+/PR+/HER2- IDC (originally ER+/PR+/HER2-, metastatic biopsy ER+/PR+/HER2-)" ✅ — **v30 受体优先级改进确认！** 两次活检状态都列出
- ✅ current_meds: anastrozole + palbociclib + leuprolide ✅（三个药全部）
- ✅ response_assessment: CT cervical LN decreased + axillary LN increased + MRI brachial plexus + bone scan sternal lesion ✅ — 详细且准确反映 mixed response
- ✅ procedure_plan: "No procedures planned" ✅（v30 字段改进确认 — 无 chemo 混入）
- ✅ imaging_plan: CT + bone scan ✅。lab_plan: monthly ✅。radiotherapy_plan: Rad Onc referral ✅
- ✅ Letter 逐句(9句): follow-up + spread to neck/arm + mixed response "some smaller, others grown" + continue meds + ondansetron + Rad Onc + CT+bone scan + monthly labs on lupron day + closing complete。通俗准确

### ROW 6 (coral_idx 145) — 0 P1, 2 P2
- 34yo, ER+/PR+/HER2- IDC 1.5cm grade 1, 0/1 node。S/p bilateral mastectomy + expanders。On zoladex (1 month) + letrozole (started today)。Oncotype low risk。Bipolar 2 disorder。Myriad negative。ECOG 0。
- P2: Patient type "New patient" — 应为 "Follow up"（zoladex 06/08 已由该提供者开始）— 同 v29
- P2: Referral-Genetics 历史转诊（04/24/2019，Myriad already negative）混入当前 referrals — 同 v29
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。current_meds: zoladex + letrozole ✅
- ✅ medication_plan: letrozole ≥3yr → tamoxifen + gabapentin + estradiol monthly ✅
- ✅ lab_summary: 全面（Estradiol 172 + Vitamin D 24 + CMP + CBC）✅
- ✅ procedure_plan: "No procedures planned" ✅（v30 字段改进确认）
- ✅ Letter 逐句(10句): bilateral mastectomy + left benign/right IDC + "grows slowly, responds to hormones" + letrozole + gabapentin + estradiol monthly + genetic counseling + 3 months + emotional support + closing complete。通俗准确

### ROW 7 (coral_idx 146) — 0 P1, 2 P2 ← **v28 Stage regression 修复确认**
- Stage II→IV ER-/PR-/HER2+ IDC left breast。Metastatic since 2008 to supraclavicular LN + mediastinum。Multiple lines (Taxotere/xeloda+Herceptin → Tykerb+Herceptin → capecitabine/Herceptin → pertuzumab/Herceptin/Taxotere)。Probable mild PD (SUV 2.1 vs 1.8), LVEF decreased 52%。D/c current regimen, recommend [T-DM1]。Second opinion。
- **v28 regression FIXED**: Stage "Originally Stage II, now Stage IV" ✅
- P2: procedure_plan "Would recheck [REDACTED]" — LVEF/echo 是 imaging 不是 procedure。同 v29
- P2: Letter 写 "check your levels of a specific medication" — [REDACTED] LVEF 被误解为 "levels of a medication"。v30 [REDACTED] handling 部分生效（没生成 garbled drug name）但仍不完美
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Goals: palliative ✅。second opinion: yes ✅
- ✅ response_assessment: "probable mild progression...SUV 2.1 (was 1.8)...[REDACTED] 14.8 persistently elevated" ✅ — 详细
- ✅ medication_plan: d/c current regimen + recommend [REDACTED] next line ✅
- ✅ Letter: progression described + LVEF 52% + d/c Herceptin/Taxotere + new medication + closing complete

### ROW 8 (coral_idx 147) — 0 P1, 2 P2
- 29yo premenopausal, Stage III ER-/PR-/HER2+(IHC 3+, FISH 5.7) IDC left breast。Incomplete neoadjuvant TCHP（3 partial cycles, non-adherent）。S/p lumpectomy + ALND: **breast pCR** but 3/28 LN+（largest 2.4cm, ECE, Ki-67 75%）。Kikuchi's disease。Plan: adjuvant AC x4 → T-DM1 + radiation。
- P2: procedure_plan "adjuvant AC x 4 cycles, to be followed by T-DM1, needs port placement" — 化疗混入 procedure（虽然 port placement 被正确捕获）。v30 字段改进部分生效但仍有 chemo 混入
- P2: response_assessment "Not yet on treatment — no response to assess" — **v30 REGRESSION!** 患者完成了不完整的新辅助化疗+手术，病理显示 breast pCR + 3/28 LN+。v29 正确捕获了 post-neoadjuvant pathology。v30 的"刚开处方"规则过度应用了
- ✅ Type: ER-/PR-/HER2+(IHC 3+, FISH 5.7) ✅。Goals: curative ✅
- ✅ medication_plan: AC x4 → T-DM1 ✅。radiotherapy_plan: radiation after AC ✅
- ✅ imaging_plan: echocardiogram ✅。Referral: social work ✅
- ✅ Letter(12句): AC + T-DM1 + radiation + echo + port + social work + closing complete。通俗准确

### ROW 9 (coral_idx 148) — 0 P1, 0 P2 ✅
- 63yo, kidney transplant recipient, Stage II R breast IDC ER+(85%)/PR-(<1%)/HER2-(IHC 0, FISH neg)。S/p neoadjuvant [AC] x4 + taxol x12 → bilateral mastectomies: 3.84cm residual (~5% cellularity), 1 macro + 1 micro + 1 ITC in 4 SLN with extranodal extension。Plan: letrozole after radiation + Fosamax。Drains still in。Full code。
- ✅ Type: ER+/PR-/HER2- IDC ✅。Stage: II ✅。Goals: curative ✅
- ✅ response_assessment: "responded to neoadjuvant therapy" + 详细病理（3.84cm/5% cellularity/LN details）✅ — 出色！
- ✅ procedure_plan: "Drains to be removed on Thursday" ✅（真正的 procedure！）
- ✅ medication_plan: Letrozole after radiation + Fosamax ✅
- ✅ Letter: bilateral mastectomy + residual cancer "mostly gone" + LN + ER "responds to estrogen" + Letrozole + radiation + drains Thursday + full code + emotional support。通俗准确


