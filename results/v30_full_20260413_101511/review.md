# V30 Full Run Review (61 samples)

> Run: v30_full_20260413_101511
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v30) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-3 完成 (3/61)，ROW 5 待审查**
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
| **P2** | 4 | — | ROW 1×2, 2×1, 3×1 (ROW 5+ 待审查) |

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


