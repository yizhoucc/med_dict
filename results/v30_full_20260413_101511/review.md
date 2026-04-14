# V30 Full Run Review (61 samples)

> Run: v30_full_20260413_101511
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v30) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查完成 — 61/61 (100%)**
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
| **P1** | 1 | 1.6% | ROW 46×1 (sarcoidosis 误分类为癌转移) |
| **P2** | 93 | 1.52/sample | ...18×2🩺, 20×6🩺, ...85×3, 86×1, 87×1, 88×1, 90×1, 91×0, 92×1, 94×0, 95×0, 97×2, 100×0 |

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

### ROW 10 (coral_idx 149) — 0 P1, 2 P2
- 66yo, Stage II left breast HR+/HER2-。S/p neoadjuvant letrozole → left mastectomy（8.8cm residual, LN involved）→ bilateral reductions + re-excision。Low risk Oncotype → no chemo。Continue letrozole。Radiation planned。DEXA。Phone consult（failed video）。Full code。
- P2: response_assessment "No specific evidence to assess current response" — 但有 post-neoadjuvant 手术病理（8.8cm, LN involved）。v29 的 POST-RESPONSE-GENOMIC hook 正确捕获了手术病理，v30 丢失了
- P2: Letter "You are referred to for a follow-up visit" — "referred to for" 缺失 [REDACTED] 医生名，语法 garbled
- ✅ Type: HR+/HER2- IDC ✅。Stage: II ✅。Goals: curative ✅。current_meds: letrozole ✅
- ✅ radiotherapy_plan: left chest wall + surrounding LN ✅。imaging_plan: DEXA ✅
- ✅ procedure_plan: "No procedures planned" ✅（v30 字段改进确认）
- ✅ Advance care: full code ✅
- ✅ Letter（除 garbled 外）: recovered from surgery + continue letrozole + radiation next week + DEXA + full code。通俗

### ROW 11 (coral_idx 150) — 0 P1, 3 P2
- 68yo, Stage IIIC→IV IDC left breast, bone mets（T-spine, R femur, mandible）。S/p mastectomy+ALND 2010 → Taxotere/Cytoxan x4 → XRT 2011。Femara not taken → Letrozole → 10/10/12 PET CT progression（mandibular mass grown）→ 10/16/12 switched to Faslodex+Denosumab。Jaw radiated 10 days（healing, numbness remains, taste returned）。Worsening R leg pain/numbness/stiffness 2 weeks。Thrush。Comorbidities: hypothyroid, DM2, hyperlipidemia, neuropathic pain。ECOG 1。Labs essentially normal。A/P: exam stable, continue Faslodex+Denosumab, MRI lumbar/pelvis/R femur, PETCT to toes, Mycelex for thrush。
- P2: response_assessment 时态混淆 — "PET/CT showed increased metastatic activity...cancer is progressing on current treatment with Faslodex and Denosumab"。但该 PET CT（10/10/12）是 Faslodex 开始之前的（10/16/12 才开始），显示的是 Letrozole 上的进展，正是换药到 Faslodex 的原因。当前 Faslodex 2 个月后 A/P 说 "Exam stable"。正确评估应为：exam stable on Faslodex, worsening R leg symptoms → restaging ordered
- P2: imaging_plan 漏了 MRI — A/P 明确 "MRI of lumbar, pelvis and right femur"，但 imaging_plan 只写了 PETCT。两个不同的影像检查，MRI 完全遗漏
- P2: Letter 时态误导 — "Imaging tests show that the cancer in your jaw has grown" 呈现 10/10/12 PET CT（2 个月前、Faslodex 之前）的结果为当前消息。但 jaw 已经接受了放疗并在愈合中。对患者来说这句话可能造成不必要的恐慌（以为现在还在长大）
- ✅ Type: IDC, ER+(从 letrozole/Faslodex 推断), HER2 not tested ✅（原文确实未提 receptor status）
- ✅ Stage: Originally Stage IIIC, now Stage IV ✅。Metastasis: bone ✅。Goals: palliative ✅
- ✅ current_meds: Faslodex + Denosumab ✅。supportive_meds: docusate, hydrocodone-acetaminophen, senna-docusate ✅
- ✅ Lab_Results: 全面准确（所有值都列出且正确）✅
- ✅ findings: 包含 PET CT 发现 + 体检 + 症状 + thrush ✅（虽然混入了 MRI order，但关键发现完整）
- ✅ medication_plan: continue Faslodex+Denosumab + Mycelex for thrush ✅
- ✅ procedure_plan: "No procedures planned" ✅（v30 字段改进确认）
- ✅ radiotherapy_plan: null ✅（jaw XRT 已完成，无新放疗计划）
- ✅ Letter 逐句(9句): follow-up + bone mets + [temporal issue] jaw + numbness jaw/leg + jaw improved post-XRT + thrush + continue meds + Mycelex 5x/day + PETCT femur/toes + closing complete。结构完整，除时态误导外通俗准确

### ROW 12 (coral_idx 151) — 0 P1, 4 P2
- 51yo, de novo Stage IV ER+/PR+/HER2+(IHC 3+, FISH 5.4) breast cancer。Mets to [REDACTED], lung, nodes, brain, bone（widespread osseous, spinal cord compression, brain mets s/p GK x3）。S/p XRT T-spine → herceptin+[REDACTED]+letrozole → taxotere(severe sepsis x3) → taxol(severe sepsis) → herceptin+[REDACTED]+letrozole since 08/18/17。GK: 23 lesions 12/17, 19 lesions 04/18, 17 lesions 09/18。Current visit: MRI brain 01/31/19 shows 2 new small foci（3mm L frontal + 1mm R precentral），previously treated lesions resolved。CT CAP: stable osseous mets, no pulmonary nodules, resolved pleural effusion, celiac node 9→7mm。A/P: continue herceptin/[REDACTED]+letrozole+[REDACTED] q12wks, off chemo(intolerance), CT CAP q4mo, bone scan 4mo, MRI brain q4mo, echo q6mo, await GK/Rad Onc。Doing "very well," off walker! DNR/DNI, POLST on file。ECOG 1。
- P2: summary 幻觉 "liver" 转移 — 原文 "to *****, lung, nodes, brain and bone" 第一个转移部位被 REDACTED，模型猜测填入 "liver"。但所有影像报告（CT AP 多次）均显示 "Liver: Unremarkable"。应写 "[REDACTED], lung, nodes, brain, and bone"
- P2: response_assessment 时间/模态混乱 — 引用 08/15/18 和 09/05/18 的旧影像而非最新的（01/31/19 MRI brain, 02/2019 CT CAP）。且将 CT 胸部发现（pleural effusion, lymphadenopathy, osseous mets）错误归到 "MRI brain"。A/P 的当前评估是 "recent body CT shows SD, no evidence of PD" + "New MRI Jan 31 shows 2 new foci"
- P2: imaging_plan 漏了 echo q6mo — A/P 明确 "Echo q6 months, recent reviewed and stable/normal, repeat again in April 2019"。Herceptin 心脏毒性监测很重要 **🩺 医生确认：Echo q6 months 是 Herceptin 心脏毒性监测的关键项目，遗漏有临床风险**
- P2: procedure_plan 错误分类 — GK（Gamma Knife stereotactic radiosurgery）是放射治疗，不是手术 procedure。应仅在 radiotherapy_plan 中（已正确列出）。v30 prompt 定义 procedure = surgery/biopsy 等
- ✅ Type: ER+/PR+/HER2+ ✅（两次活检 IHC 3+ 确认）。Stage: IV ✅。Metastasis: brain, lung, nodes, bone ✅
- ✅ Goals: palliative ✅。current_meds: herceptin, letrozole ✅（其他药名 REDACTED）
- ✅ Advance care: POLST on file, no life support ✅。Referral: Rad Onc + neurology F/u ✅
- ✅ medication_plan: 全面（continue all + off chemo due to intolerance）✅
- ✅ radiotherapy_plan: await GK/Rad Onc input ✅。Next visit: 6 weeks ✅
- ✅ Letter 逐句(9句): new brain lesions + body stable + herceptin "targets HER2 protein"(**v30改进！**) + letrozole + bone med q12wks + off chemo(intolerance) + CT/MRI/bone scan q4mo + radiation team for brain + 6 weeks F/u + closing complete。**Letter 出色** — 准确、简洁、通俗。缺 echo 提及（同 imaging_plan 遗漏）

### ROW 14 (coral_idx 153) — 0 P1, 2 P2
- 58yo, de novo Stage IV ER+(99%)/PR+(25%)/HER2-(IHC 1+, FISH neg) breast cancer to bone, liver, nodes。Extensive bone mets with spinal cord compression → multiple spinal surgeries + XRT。Was on faslodex+palbociclib since 07/18 → patient STOPPED end of January（sensitivity test: 20% responsive to palbo, 10% to fulvestrant）→ went to Mexico for alternative treatment → now self-administering at home: doxorubicin 10mg + gemcitabine 200mg + docetaxel 20mg weekly + pamidronate weekly + metabolic therapy + immunological vaccines。US oncologist now in monitoring role only。CA 27.29 trending down 193→48。ECOG 2, wt 48.2kg (-2 lbs/mo)。R axilla node 1cm, R breast 1.5x2.0cm。Labs: several abnormalities（macrocytosis, lymphopenia）。A/P: monitoring, F/u 3 months。
- P2: current_meds 空 — 但患者正在服用癌症药物（gemcitabine, docetaxel, doxorubicin, pamidronate），第二位医生明确确认 "The treatment includes pamidronate, gemcitabine, docetaxel, and doxorubicin"。即使是 Mexico 处方，这些仍是当前癌症用药，应列入 current_meds
- P2: Next visit "2 months" — 但第一位医生 A/P 写 "F/u 3 months"。"2 months" 来自第二位医生引用患者说将从 Mexico 回来的时间，不是诊所随访时间
- ✅ Type: ER+, HER2- ✅（PR+ 25% 未提及，但笔记标题也只说 "ER+"）。Stage: IV ✅。Mets: bone, liver, nodes ✅
- ✅ Goals: palliative ✅。recent_changes: 完整捕获从 palbo/fulvestrant 到 Mexico protocol 的变化 ✅
- ✅ response_assessment: "stable on exam and imaging" — 合理（本次无新影像，exam stable）。遗漏 CA 27.29 下降趋势但不关键
- ✅ Lab_Results: 全面准确，包括 CA 27.29 ✅。imaging_plan: CT CAP + spine MRI May + repeat MRI 6 weeks ✅
- ✅ procedure_plan: "No procedures planned" ✅。Referral: PT ✅
- ✅ Letter 逐句(16句): stopped palbo/fulvestrant + low dose chemo at home + pamidronate + stable + mobility improved + Cymbalta Rx + cannabis/sulfur + continue plan + CT+MRI May + MRI 6wks + labs q2wks + PT + F/u + closing complete。**Letter 出色** — 全面准确，语言通俗。缺 weight loss dietary advice（A/P 建议小份高热量高蛋白饮食，对患者实用）

### ROW 17 (coral_idx 156) — 0 P1, 3 P2
- 53yo, post-lumpectomy consultation（video visit）。L breast IDC 0.8cm grade 2, ER+(>95%)/PR+(>95%)/HER2-(IHC 0, FISH 1.1X), Ki-67 5%。LN 0/5, margins neg, no DCIS。Chest CT neg。Pelvic US neg。No current meds。Menopausal status unclear（s/p hysterectomy）。Family hx: sister ovarian ca @40s, aunt breast ca @60s。Plan: adjuvant hormonal therapy ≥5yr（tamoxifen or AI based on menopausal status）+ breast RT + hormone labs + DXA + genetics referral + nutritionist。10-15% recurrence risk without therapy。
- P2: Stage "Not mentioned in note" — 虽然 IMPRESSION 中 stage 被 REDACTED，但从病理可推断：0.8cm(T1b) + LN 0/5(N0) + no mets(M0) = **Stage IA**。v30 prompt 允许从肿瘤大小+LN 推断分期
- P2: procedure_plan "check labs including hormones" — 抽血化验不是 procedure（手术/活检）。应为 "No procedures planned"。Lab_plan 已正确捕获该项
- P2: Letter 写 "no cancer was found in the removed tissue" — **事实错误**。手术标本中发现了 0.8cm IDC grade 2 肿瘤。"Negative" 的是 margins（切缘清洁）、DCIS（无）和 LN（0/5），不是 "no cancer found"。Letter 混淆了 "margins negative" 和 "no cancer"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。Metastasis: No ✅
- ✅ current_meds: empty ✅（"Meds: none"）。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: adjuvant hormonal therapy ≥5yr ✅。radiotherapy_plan: breast RT scheduled ✅
- ✅ imaging_plan: DXA ✅。lab_plan: hormone labs ✅。genetic_testing_plan: genetics referral ✅
- ✅ Referrals: nutritionist ✅, genetics ✅, RT consult ✅。Next visit: after RT ✅
- ✅ findings: 病理+影像+体检全面准确 ✅
- ✅ Letter（除 "no cancer found" 外）: IDC "milk ducts" + ER/PR/HER2 "proteins" + adjuvant hormonal "prevent coming back" + hormone tests + DXA "bone strength" + RT + genetics + nutritionist + F/U after RT。通俗出色

### ROW 18 (coral_idx 157) — 0 P1, 2 P2
- 65yo, med onc consultation（with husband, in-person）。L breast IDC 8mm grade 1 arising in encapsulated papillary carcinoma。ER+(100%)/PR+(95%)/HER2-(IHC 1+), Ki-67 5%。pT1b, SLN ITC 1/3(0/3 H&E+)。Margins neg。DCIS present（encapsulated papillary = DCIS equivalent）。PMH: papillary thyroid ca s/p thyroidectomy。Family hx: half-sister breast ca @45。Meds: thyroid only。IMP: early-stage, adjuvant endocrine 5-10yr, low risk biology → patient declines chemo → no molecular profiling。DEXA ordered。Rad Onc eval +/- XRT。Genetics: prior attempt failed, UCSF Cancer Risk will reach out today。
- P2: Referral-Genetics "None" — 但 UCSF Cancer Risk IS a genetics referral（"discussed with UCSF Cancer Risk. They will reach out to pt today"）。genetic_testing_plan 正确捕获了此信息，但 Referral 字段也应反映
- ✅ Type: ER+/PR+/HER2- IDC + encapsulated papillary carcinoma ✅（出色）。Stage: pT1b pNX ✅
- P2: **🩺 医生意见** goals_of_treatment "curative" — 原文 A/P 中没有明确写 "curative" 或 "cure"。虽然 adjuvant endocrine therapy 对 early-stage 隐含 curative intent，但原文只说 "strongly recommend adjuvant endocrine therapy" 而非明确陈述治疗目标。模型推断了合理但未显式声明的信息
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: empty ✅
- ✅ medication_plan: adjuvant endocrine 5-10yr ✅。radiotherapy_plan: Rad Onc eval +/- XRT ✅
- ✅ procedure_plan: "No procedures planned" ✅。imaging_plan: DEXA ✅
- ✅ genetic_testing_plan: 详细捕获失败的先前尝试 + UCSF Cancer Risk 联系 ✅
- ✅ findings: 详细病理 + 体检 ✅
- ✅ **Letter 出色**（10句）: IDC "milk ducts" + encapsulated papillary + early stage + cure + adjuvant endocrine "prevent coming back" 5-10yr + Rad Onc + DEXA "bone health" + genetics "blood sample" + UCSF will reach out + F/U + closing complete。准确全面通俗

### ROW 20 (coral_idx 159) — 0 P1, 6 P2
- 75yo postmenopausal, metastatic recurrence of ER+(80%)/PR+(50%)/HER2-(FISH 1.05) IDC。Original dx 2009: L breast 0.9cm grade II IDC s/p bilateral mastectomy + 5yr tamoxifen + 6mo letrozole(discontinued)。Genetic testing (21 genes) negative。2021 PET/CT: R axillary hypermetabolic LN, mediastinal/hilar nodes, innumerable osseous lesions。R iliac crest biopsy confirms breast primary。7mm lung nodule without FDG uptake（likely not met）。Family hx: 3 sisters breast ca (@42,48,51)。ECOG 0。L rib/axillary pain。Plan: letrozole+palbociclib, denosumab(dental clearance), MRI spine, CT CAP, Rad Onc referral, Foundation One, monthly labs, RTC ~1mo。
- P2: response_assessment "Not mentioned in note" — 应为 "Not yet on treatment — no response to assess"。患者刚诊断转移复发，尚未开始任何治疗。v30 规则明确要求此场景使用 "Not yet on treatment"
- P2: procedure_plan 混乱 — "Abdomen, Pelvis, Xgeva - needs dental evaluation first, [REDACTED] 360" 混入了 CT 影像 + Xgeva 牙科评估 + 基因组检测。这些都不是 procedure（手术/活检）。应为 "No procedures planned"
- P2: lab_summary 报告 2013 年血糖（8 年前！）— 原文标注 "Results for orders placed or performed in visit on 03/01/13"。本次就诊无当前实验室结果。应注明无当前 labs
- P2: Letter 写 "You will have a biopsy" — **不正确**。计划是 "Obtain outside path and send for Foundation One" = 获取已有的外院病理切片做基因组检测，不是做新的活检
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: Stage IV ✅。Mets: bone + LN ✅（lung nodule non-FDG avid 正确排除）
- ✅ Goals: palliative ✅。medication_plan: letrozole + palbociclib + denosumab(dental clearance) ✅ 全面
- P2: **🩺 医生意见** imaging_plan 漏了 "Obtain outside PET/CT" — A/P plan 明确列出 "Obtain outside PET/CT" 作为独立项目，但 imaging_plan 只列了 MRI+CT CAP+repeat 3mo。外院 PET/CT 的获取对病情评估很重要
- P2: **🩺 医生意见** Letter 缺少 "monthly blood work on Palbociclib" — lab_plan 正确捕获了此信息（"She will need monthly blood work on Palbociclib"），但 letter 只写 "blood tests to check for tumor markers" 而未提及 palbociclib 的每月血检要求。这对患者安全很重要（中性粒细胞减少监测）
- ✅ lab_plan: tumor markers + monthly bloodwork ✅（lab_plan 本身正确，问题在 imaging_plan 和 letter）
- ✅ genetic_testing_plan: Foundation One (or [REDACTED] 360) ✅。radiotherapy_plan: Rad Onc referral ✅
- ✅ findings: 全面（PET/CT + biopsy + exam + symptoms）✅
- ✅ Letter（除 "biopsy" 外）: metastatic recurrence + ER+/PR+/HER2- + bones/LN + rib pain + letrozole + palbociclib + denosumab(dental) + Rad Onc + MRI+CT + tumor markers + Foundation One + 1mo F/U。全面准确

### ROW 22 (coral_idx 161) — 0 P1, 2 P2
- 72yo, second opinion。1994 L DCIS s/p lumpectomy+RT。2000 R Stage II IDC(1.5cm, 1/14 LN, HR+) s/p lumpectomy+ALND+AC+RT+6yr tamoxifen/[REDACTED]。May 2020 R chest wall recurrence + bone mets + R infraclavicular + R IM nodes → HR+/HER2-。XRT T10+L4, abemaciclib+letrozole → anastrozole(rash), abemaciclib dose reduced → PET 11/2020+04/2021 good response → July 2021 pneumonitis → abemaciclib held, steroids。ECOG 0。Full code。No palpable mass。A/P: PET CT now → if stable arimidex alone → if progression faslodex+[REDACTED] → future: afinitor/xeloda/trial。
- P2: lab_summary "No labs in note" — 原文有 01/29/2021 的详细 labs（8个月前，包括 anemia Hgb 10.7, lymphopenia 0.54, elevated Cr 1.19, eGFR 46-53）。虽然不是当次就诊的，但 note 中确实列出了这些数据。应至少说明 "No recent labs; labs from 01/29/2021 available"
- P2: imaging_plan 混入了条件用药计划 — "if stable continue arimidex alone. If progression could use faslodex..." 这些属于 medication_plan（已正确列入）。imaging_plan 应仅为 "PET CT now"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: II→IV ✅。Mets: bones, chest wall, nodes ✅
- ✅ second opinion: yes ✅。Goals: palliative ✅。Advance care: Full code ✅
- ✅ current_meds: anastrozole + denosumab ✅（abemaciclib 正确排除 — held）
- ✅ response_assessment: PET good response + exam clean ✅
- ✅ medication_plan: comprehensive with stability/progression pathways ✅
- ✅ procedure_plan: "No procedures planned" ✅。radiotherapy_plan: None ✅
- ✅ **Letter 出色**（14句）: second opinion + L 1994 + R 2000 + recurrence 2020 + Stage IV + bones/LN + PET response + pneumonitis "lung condition" + palliative "manage cancer, improve quality of life" + anastrozole/abemaciclib changes + prednisone/denosumab + PET CT contingency plan + F/U after PET + emotional support + closing complete。准确全面通俗共情

### ROW 27 (coral_idx 166) — 0 P1, 2 P2
- 41yo premenopausal, HR+/HER2- IDC, metastatic to bone（L1, thoracolumbar, sternum, iliac）since 2006。On femara+zoladex+zoledronic acid。PET/CT: stable to slightly decreased metabolic activity, no new lesions。New symptoms: lightheadedness, back pain, easy bruising, urinary frequency。A/P: stable disease, continue meds, reassess back pain 2 weeks → MRI spine, UA for UTI, CBC for bruising。
- P2: lab_plan 漏了 UA — A/P 明确 "Possible UTI vs. Genital-urinary symptoms from estrogen depletion. Obtain UA"。lab_plan 只写了 CBC
- P2: Next visit "if pain worsens" — 遗漏 2 周评估。A/P 说 "Reassess lower back pain at two weeks"。应为 "2 weeks to reassess; sooner if pain worsens"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: IV ✅。Mets: bone ✅。Goals: palliative ✅
- ✅ current_meds: letrozole+zoladex+zoledronic acid ✅（三个药全部）
- ✅ response_assessment: PET-CT stable to slightly decreased, no new mets ✅ — 准确
- ✅ findings: 详细 PET/CT SUV 值 + 体检 ✅
- ✅ procedure_plan: none ✅。imaging_plan: MRI spine consider 2wks ✅
- ✅ Letter（9句）: IDC "milk ducts" + spread to bones + stable "not growing, not as active" + continue meds + back pain come back + MRI 2wks + blood test bruising + closing。简洁准确通俗

### ROW 29 (coral_idx 168) — 0 P1, 1 P2
- 59yo postmenopausal, multifocal grade 2 IDC ER+/PR+(weak)/HER2-, Ki-67 10-11%。S/p partial mastectomy: 1.6cm+0.6cm IDC（micropapillary+ductal, +LVI）, SLN micromet 0.5mm(1/1), pT1c(m)N1mi(sn)。DCIS intermediate, 12:30 site positive margin → re-excision needed。Oncotype/MammaPrint Low Risk → chemo NOT recommended。Plan: start letrozole 2.5mg daily, re-excision Sept 2019, RT locally abroad, DEXA, calcium+D, vaginal moisturizer。
- P2: response_assessment "On treatment; response assessment not available" — 患者本次就诊才拿到 letrozole 处方，尚未开始辅助治疗。应为 "Not yet on treatment — no response to assess"。v30 规则 "刚开处方≠On treatment"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: pT1c(m)N1(sn) ≈ Stage IIA ✅。Mets: No ✅。Goals: curative ✅
- ✅ procedure_plan: re-excision Sept 2019 ✅（真正的 procedure — positive margin）
- ✅ medication_plan: letrozole + calcium + vaginal moisturizer ✅。radiotherapy_plan: breast RT locally ✅
- ✅ imaging_plan: DEXA ✅。findings: 详细病理+MRI+体检 ✅
- ✅ Letter（9句）: IDC + ER/PR "sensitive to estrogen/progesterone" + HER2 "protein"(**v30改进！**) + early stage + letrozole "prevent coming back" + radiation + surgery Sept + DEXA + F/U closer to home + closing。准确通俗

### ROW 30 (coral_idx 169) — 0 P1, 1 P2
- 64yo postmenopausal, clinical stage II-III ER-(0%)/PR-(0%)/HER2+(IHC 3+, FISH 8.9) IDC, Ki-67 ~30%。History of untreated DCIS since 2006/2012。R breast 6x11cm mass（PET: 9.0x3.8cm SUV 7.8）, R axillary LN（SUV 1.8）, R IM chain node。Denuded R nipple + itchy rash。ECOG 0。No mets on PET/CT。Plan: neoadjuvant THP→AC or TCHP, trastuzumab 1yr, then surgery+RT。Needs TTE + Mediport。Patient to decide over weekend。
- P2: Letter garbled — "proceed with treatment at a medication or elsewhere" — [REDACTED] 医院名称被误解为 "a medication"。应为 "at [this facility] or elsewhere, closer to home"
- ✅ Type: ER-/PR-/HER2+ IDC ✅。Stage: Clinical II-III ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: empty ✅
- ✅ medication_plan: 两个方案都详细描述（THP/AC and TCHP + 1yr trastuzumab）✅ — 出色
- ✅ procedure_plan: Mediport placement ✅。imaging_plan: TTE ✅。radiotherapy_plan ✅
- ✅ findings: PET/CT + MRI + exam 全面 ✅
- ✅ Letter（除 garbled 外）: ER/PR/HER2 "proteins"(**v30改进！**) + Stage II-III "more advanced but not spread" + curative + mass described + chemo options + Mediport "easier to give chemo" + TTE + radiation + F/U after weekend + closing。内容全面准确

### ROW 33 (coral_idx 172) — 0 P1, 2 P2
- 63yo, adjuvant letrozole follow-up。Dx 07/2010 L breast ILC grade 2, ER+/PR+/HER2-, clinical stage IIB → pathologic stage IIIA。S/p L mastectomy+ALND → TC x6 → XRT → prophylactic R mastectomy（no cancer）。Letrozole since 02/2011（brand name, generic GI intolerance）。Joint stiffness stable。Monthly headaches(?tension)。Exam: ECOG 0, no recurrence, +L neck <1cm LN soft, +[REDACTED] lymphedema。A/P: continue letrozole >5yr, calcium+D, NSAIDs PRN, if headaches persist → MRI brain, F/U 6mo。
- P2: Stage "Originally stage IIB, now stage IIIA" — "now" 暗示疾病进展，但两个分期都来自 2010 同一次诊断（clinical IIB → pathologic IIIA after surgery/ALND）。应为 "Stage IIIA (pathologic)" 或 "Clinical IIB, pathologic IIIA"
- P2: findings 写 "No evidence of lymphedema" — 但体检明确 "+[REDACTED] lymphedema"（POSITIVE）。这是直接矛盾。可能是 mastectomy+ALND 后的上肢淋巴水肿
- ✅ Type: ER+/PR+/HER2- ILC ✅（正确识别为小叶癌）。Goals: curative ✅。Mets: No ✅
- ✅ response_assessment: "No evidence of recurrence, tolerating letrozole well" ✅
- ✅ medication_plan: letrozole + calcium/D + NSAIDs PRN ✅
- ✅ procedure_plan: none ✅。imaging_plan: conditional MRI brain ✅。Next visit: 6mo ✅
- ✅ Letter（12句）: follow-up + no recurrence + no spread + letrozole stable + continue meds + calcium/D + NSAIDs "pain relievers" + MRI brain if persists + 6mo F/U + closing。简洁准确通俗

### ROW 34 (coral_idx 173) — 0 P1, 3 P2
- 71yo, Stage III L breast cancer, second local recurrence。2011 L lumpectomy IDC(refused SLN/chemo)→ 2012 rapid recurrence bilateral MX+implants IDC 3.3cm 11+LN+ ER+/PR-/HER2-。Partial AC/T, anastrozole(self-d/c'd), declined CW XRT。2020 second recurrence: FNA ER+(100%)/PR+(50%)/HER2- → excision IDC 1.7cm grade 3, skeletal muscle invasion, margins neg。PET-CT: hypermetabolic L implant, 6th rib(unclear), brain MRI neg。ECOG 0。A/P: no chemo(CALOR study), switch arimidex→tamoxifen 20mg(naive, possible AI resistance), CW RT referral(now accepts), check labs, F/U 6mo。
- P2: Type "ER+/PR-/HER2-" 用了 2012 受体状态（PR-），但 2020 复发活检显示 **PR+(50%)**。v30 prompt 要求最新受体状态优先。应为 ER+/PR+/HER2-
- P2: procedure_plan "check labs, referral..." — 抽血和转诊都不是 procedure。应为 "No procedures planned"
- P2: lab_plan "No labs planned" — A/P plan 第一项明确写 "check labs"
- ✅ Stage: "Originally Stage III, now local recurrence" ✅。Mets: No ✅。Goals: curative ✅
- ✅ medication_plan: tamoxifen 20mg ✅。radiotherapy_plan: CW RT referral ✅
- ✅ recent_changes: switch arimidex→tamoxifen ✅
- ✅ findings: 全面（pathology + PET-CT + MRI + exam）✅
- ✅ Letter（11句）: local recurrence + IDC "milk ducts" grade 3 "more aggressive" + skeletal muscle "damage to muscle" + no spread + tamoxifen "change from arimidex" + CW RT "prevent coming back" + 6mo F/U + closing。准确通俗

### ROW 36 (coral_idx 175) — 0 P1, 3 P2
- 27yo premenopausal, pT3N0 ER+(>90%)/PR+(~20%)/HER2- grade III mixed ductal+mucinous carcinoma。S/p bilateral mastectomies 12/06/20（8.4cm, DCIS ~9.6cm, 0/4 LN）。Post-op cellulitis → debridement。PET/CT: post-surgical, incidental thyroid nodule 2.1cm。Started tamoxifen 01/29, zoladex 02/06, Taxol 02/13 → grade 3 reaction → switched Abraxane。Today cycle 8/12。R arm/hand swelling 4-5 days。Nausea improved。ECOG 1。Labs: mild anemia Hgb 11.8, Albumin 3.3(L)。A/P: Doppler R arm, continue Abraxane x12, zoladex, Rad Onc referral, antiemetics, valtrex ppx, lexapro/ativan/ambien, RTC 2wks。
- P2: current_meds 漏了 tamoxifen — 原文 "she [started] tamoxifen on 01/29/21"，patient IS on tamoxifen（关键内分泌治疗），但 current_meds 只列 Abraxane+zoladex
- P2: lab_summary 只含 04/10 CBC — 漏了 04/03 CMP（Albumin 3.3(L), Bilirubin 0.1(L) 等）。两组 labs 都在 note 中
- P2: procedure_plan 混入 Doppler（imaging）和 Abraxane（medication）。Doppler 已正确列在 imaging_plan。应为 "No procedures planned"
- ✅ Type: ER+/PR+/HER2- grade III mixed ductal+mucinous ✅（出色，捕获混合组织类型）
- ✅ Stage: pT3N0 ✅。Mets: No ✅。Goals: curative ✅
- ✅ imaging_plan: Doppler to r/o DVT ✅。radiotherapy_plan: Rad Onc referral ✅
- ✅ recent_changes: Taxol→Abraxane switch ✅。supportive_meds: Zofran, Compazine ✅
- ✅ Letter（12句）: R arm swelling + no cancer growth + Abraxane/Taxol switch + antiemetics + Abraxane continue + Zoladex "protect ovaries" + Doppler "blood clots" + Rad Onc next week + 2wks F/U + emotional support + closing。准确通俗（缺 tamoxifen 提及，同 keypoints 遗漏）

### ROW 37 (coral_idx 176) — 0 P1, 1 P2
- 61yo postmenopausal（video visit）, Stage IIA L TNBC, 2.3cm grade 3 IDC, ER-/PR-/HER2-, node neg, LVI absent。S/p bilateral mastectomies July 2020。No cancer meds。ECOG 0。Full code。A/P: dd AC→Taxol, no RT/no hormonal, lifestyle mods, chemo at [REDACTED]。
- P2: Letter garbled — "chemotherapy at a medication" — [REDACTED] 机构名被误解为 "a medication"。同 ROW 30 相同模式
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: IIA ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: empty ✅
- ✅ medication_plan: dd AC→Taxol ✅。radiotherapy_plan: None ✅（correctly no indication）
- ✅ Advance care: Full code ✅。Televisit ✅
- ✅ Letter（除 garbled 外）: TNBC "no receptors for ER, PR, HER2 protein"(**v30改进！**) + 2.3cm no spread + adjuvant chemo "prevent coming back" + AC→Taxol + full code "all possible life-saving measures" + closing。准确通俗

### ROW 40 (coral_idx 179) — 0 P1, 2 P2
- 62yo with MS（25yr, secondary progressive, on teriflunomide）+ osteoporosis（Prolia）+ Graves' s/p thyroidectomy。R breast IDC G1 2.3cm, ER 95/PR 5/HER2 2+ FISH neg(1.2), 1 SLN micromet + 1 LN direct extension, margins neg。Stage 2。Post-surgery consultation。A/P: adjuvant letrozole（Rx given）, chemo not recommended（patient not interested, small benefit）, DEXA, PT referral, appt with Dr.[REDACTED]（Rad Onc?）— "if no radiation planned can start letrozole immediately"。RTC 3mo。
- P2: response_assessment "On treatment" — letrozole 刚开处方，未开始。应为 "Not yet on treatment"
- P2: radiotherapy_plan "None" — 但 A/P 提到有 Dr.[REDACTED] 的预约且 "if no radiation is planned then she can start letrozole immediately"，说明 RT 正在评估中（pending Rad Onc evaluation），不是 "None"
- ✅ Type: ER 95/PR 5/HER2- G1 IDC ✅（raw values）。Stage: 2 ✅。Mets: No ✅。Goals: curative ✅
- ✅ medication_plan: letrozole + Prolia ✅。imaging_plan: DEXA ✅。PT referral ✅
- ✅ procedure_plan: none ✅。genetic_testing_plan: 正确解释不做分子检测的原因 ✅
- ✅ Letter（13句）: IDC + early stage + curative + letrozole + ondansetron + DEXA "bone health" + PT + 3mo F/U + closing。准确通俗（缺 Rad Onc 评估提及，同 radiotherapy_plan 遗漏）

### ROW 41 (coral_idx 180) — 0 P1, 2 P2
- 32yo premenopausal, ATM mutation carrier。L breast 3cm grade 3 IDC, ER+(90%)/PR weakly+(1%)/HER2 1+(IHC, FISH N/A), Ki-67 30%。MammaPrint High Risk。S/p bilateral mastectomy+L SLN: 1/3 micromet(0.022cm), +LVI, R breast benign。MUGA LVEF 79%。Mild anemia Hgb 11.8。Port placement this week。A/P: Taxol 12wk→AC, then OFS+AI, possibly ribociclib trial。
- P2: Stage 空 — 应从 pathology 推断：3cm(pT2)+1 SLN micromet(N1mi) ≈ Stage IIA
- P2: Letter garbled — "a chemotherapy regimenaxol" — "AC-Taxol" 与 "regimen" 连在一起产生乱码。应为 "a chemotherapy regimen called AC-Taxol"
- ✅ Type: ER+/PR weakly+/HER2 1+ IHC IDC ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。current_meds: empty ✅
- ✅ medication_plan: Taxol→AC + OFS+AI + ribociclib trial ✅（全面）
- ✅ procedure_plan: port placement ✅。lab_summary: comprehensive ✅。findings: comprehensive ✅
- ✅ Letter（除 garbled 外）: bilateral mastectomy + IDC "milk ducts" + LN + anemia "might feel tired" + MUGA "heart working well" + US + Taxol 12wk→AC + docusate + port "easier to give chemo" + OFS "stop ovaries from making hormones" + ribociclib trial + emotional support + closing。全面通俗

### ROW 42 (coral_idx 181) — 0 P1, 1 P2
- 41yo premenopausal, R breast multifocal IDC（0.9cm+0.3cm）grade 1, PR strongly+(95%), HER2/neu neg。Margins clear（deep re-excised clear）。0/5 SLN。S/p 3-week RT completed 01/05。Feeling well, no complaints。A/P: premenopausal → tamoxifen 5yr Rx。RTC 4-6wks。Diagnostic mammogram at next visit。
- P2: Stage "Not mentioned in note" — 应从 pathology 推断：multifocal 0.9cm(pT1b)+0/5 SLN(N0) → Stage IA
- ✅ Type: ER+/PR+/HER2- IDC ✅（ER 从 tamoxifen 使用推断）。Mets: No ✅。Goals: curative ✅
- ✅ medication_plan: tamoxifen 5yr ✅。imaging_plan: diagnostic mammogram ✅
- ✅ procedure_plan: none ✅。radiotherapy_plan: None ✅（RT 已完成）
- ✅ findings: 详细病理+体检 ✅。Next visit: 4-6wks ✅
- ✅ **Letter 出色**（9句）: IDC "milk ducts" + no LN spread + "sensitive to hormones, did not have HER2 protein"(**v30改进！**) + tamoxifen "prevent coming back" + 4-6wks + mammogram + closing。简洁准确通俗

### ROW 43 (coral_idx 182) — 0 P1, 0 P2 ✅
- 38yo premenopausal, second primary Stage I L TNBC。First cancer 2010(age 27): Stage I TNBC s/p lumpectomy+ddAC→T+XRT, BRCA neg。2021: bilateral mastectomies+L SLN → 1.3cm IDC grade 3, ER-/PR-/HER2-(IHC 1, FISH neg), Ki-67 >80%, 0/2 SLN, margins neg。Post-op: severe anemia Hgb 5.4→transfusion。Healing well。ECOG 0。Full code。A/P: adjuvant taxol+carboplatin x4, RTC 2 days prior to cycle 1 for labs。
- ✅ Type: ER-/PR-/HER2- IDC ✅。Stage: Stage I (second primary) ✅。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: "Not yet on treatment" ✅。medication_plan: taxol+carboplatin x4 ✅
- ✅ supportive_meds: granisetron+compazine+senna ✅。lab_summary: comprehensive ✅
- ✅ Advance care: Full code ✅。findings: comprehensive ✅
- ✅ **Letter 出色**: healing well + TNBC "no proteins ER/PR/HER2"(**v30改进！**) + no LN cancer + taxol+carboplatin "prevent coming back" + nausea/bowel meds + four cycles + lab draw 2 days before + full code + emotional support。全面准确通俗

### ROW 44 (coral_idx 183) — 0 P1, 1 P2
- 33yo premenopausal, BRCA1+, ER+(95%)/PR+(95%)/HER2- node+ L breast IDC。S/p neoadjuvant dd AC x4→Taxol x4 → bilateral mastectomies 10/07/18: residual 1cm IDC grade 2(cellularity 15%), residual DCIS, 1/18 LN micromet(0.07cm), margins neg。Post-neoadjuvant receptor: ER+/PR-。R breast benign。Pulmonary nodule 4mm stable。ECOG 2, weight 41.3kg(BMI 16.76 underweight)。A/P: RT clinical trial(3 vs 5 wks, 12/16/18)→AI after RT→BSO planned(12/02 discussion)→ribociclib trial possible。CT chest 1yr。Nutrition+PT referral。F/U 01/05/19。
- P2: Letter 写 "cancer that was not completely removed" — **误导**。这暗示手术未完全切除（surgeon left cancer behind），但实际是新辅助化疗后残留（chemo didn't fully eradicate），手术 margins negative（完全切除）。应改为 "a small amount of cancer remained after chemotherapy, but it was fully removed by surgery"
- ✅ Type: ER+/PR+/HER2- node+ IDC ✅（原始活检 PR+ 95%）。Mets: No ✅。Goals: curative ✅
- ✅ response_assessment: residual IDC cellularity 15%, no progression ✅ — 准确
- ✅ medication_plan: AI after RT + Zoladex if BSO delayed + ribociclib trial ✅
- ✅ radiotherapy_plan: clinical trial 3 vs 5 wks ✅。imaging_plan: CT chest 1yr ✅
- ✅ Referrals: nutrition(11/30) + Rad Onc + PT ✅。Next visit: 01/05/19 ✅
- ✅ Letter（除 "not removed" 外）: residual cancer + pulmonary nodule stable + AI after RT + Lexapro + Zoladex "removing ovaries" + RT trial 3/5wks + PT + CT 1yr + nutrition + F/U dates + emotional support。全面通俗

### ROW 46 (coral_idx 185) — **1 P1**, 3 P2 ← **首个 P1！**
- 48yo postmenopausal(s/p BSO), R breast IDC grade 2, 原始 ER+(98%)/PR+(25%)/HER2-(IHC 2+,FISH neg)。PET/CT 12/31/20: R breast+R axillary+R IM+possible R chest wall。**Extensive bilateral hilar/mediastinal LAD "most suggestive of SARCOIDOSIS vs less likely metastatic"**。**01/09/21 endobronchial biopsy: NON-NECROTIZING GRANULOMATOUS INFLAMMATION = 结节病，NOT cancer！**。Neoadjuvant Taxol→[REDACTED] completed 05/26/21。Surgery: R lumpectomy+SLN → 3.5cm residual IDC(cellularity 10-20%), **POSITIVE margins**(multifocal), 2/2 SLN macromets(6mm, ENE>2mm)。Post-neoadjuvant: ER+(95%)/PR-(0%)/HER2-(1+)。A/P: **CURATIVE INTENT** — re-excision + ALND discussion + XRT + letrozole + abemaciclib after XRT + DEXA。Sarcoidosis F/U。Renal artery aneurysm(MRA 1yr)。Anemia(PO iron, repeat 3-4mo)。F/U 2-3mo。
- **P1: Stage "now metastatic (Stage IV)" + Metastasis "Yes (to mediastinal and hilar LN)" + Goals "palliative" — 全部错误！** 纵隔/肺门 LAD 已被 endobronchial biopsy 证实为 **结节病**（非坏死性肉芽肿性炎症），**NOT cancer metastasis**。A/P 以 curative intent 治疗（re-excision+XRT+adjuvant）。模型将结节病误分类为癌症转移，导致 stage/mets/goals 全部错误。Letter 也受影响："cancer has spread to chest lymph nodes" + "worried cancer might have spread" — 对患者来说非常 alarming 且不正确
- P2: Type "ER+/PR- (95%)/HER2-" — 格式混淆，ER 的 95% 放在 PR- 旁边。应为 "ER+(95%)/PR-/HER2-"
- P2: imaging_plan 漏了 DEXA（基线检查，A/P 明确 "Baseline dexa ordered"）
- P2: lab_plan "No labs planned" — A/P 说 "Repeat [iron panel] in 3-4 months"
- ✅ procedure_plan: re-excision ✅（真正的 procedure）。radiotherapy_plan: Rad Onc after re-excision ✅
- ✅ medication_plan: letrozole + naproxen + APAP + allegra + PO iron + tramadol PRN ✅
- ✅ lab_summary: comprehensive(CMP+CBC+VitD+CRP+ESR) ✅

### ROW 49 (coral_idx 188) — 0 P1, 1 P2
- 50yo premenopausal, multifocal L breast IDC ER+(100%)/PR+(100%)/HER2-(FISH 1.4), L axillary LN met(biopsy-proven), Oncotype 11(low risk)。MRI: 4x2.5x3cm NME + 1cm bilobed nodule。PET/CT no FDG uptake(false neg?), bone marrow T7-L1 uptake → thoracic MRI: no mets。Likely Stage II。Two opinions: neoadjuvant vs surgery first → chose surgery。A/P: L mastectomy 01/06/17, adjuvant tamoxifen(pending thrombophilia workup for father's PE), XRT for node+ discussed。Surrogate: spouse。
- P2: response_assessment "On treatment" — 患者术前，尚未开始任何治疗。应为 "Not yet on treatment"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: II ✅。Mets: No ✅。Goals: curative ✅
- ✅ medication_plan: tamoxifen + thrombophilia assessment ✅。procedure_plan: L mastectomy 01/06/17 ✅
- ✅ findings: comprehensive ✅。Advance care: surrogate spouse ✅。radiotherapy_plan: XRT discussed ✅
- ✅ **Letter 出色**: IDC + LN spread + ER/PR/HER2 "proteins targeted" + Stage II + curative + tamoxifen + "blood clots" risk + mastectomy Jan 6 + F/U post-op + surrogate "make decisions for you" + closing。全面准确通俗

### ROW 50 (coral_idx 189) — 0 P1, 4 P2
- 58yo, de novo Stage IV HR+/HER2- IDC to lung/LN/liver/bone(July 2013)。S/p AC x4 → tamoxifen+lupron → progression Oct 2014 → letrozole+lupron+ibrance(Jan 2015)。XRT pelvis/sternum 2014-15。Lumpectomy+XRT 2019。Pathogenic PMS2 mutation。Dec 2021 restaging: disease under good control。BUT L breast progression → biopsy IDC+DCIS → considering mastectomy vs observation。Second opinion visit(video)。ECOG 0。Full code。
- P2: current_meds 漏了 lupron — 原文 "remains on ibrance, xgeva, letrozole and lupron"，但 current_meds 只列 ibrance+xgeva+letrozole
- P2: response_assessment 只写 "under good control" — 遗漏 L breast progression（A/P #5 "Progression in the left breast on imaging"）。是 mixed response（systemic controlled, local progression）
- P2: medication_plan 是治疗历史而非当前计划。应为 "Continue ibrance, letrozole, lupron, xgeva"
- P2: Referral-Genetics "None" — A/P 明确 "Referral to genetics for pathogenic PMS 2 mutation" + "refer her to high risk clinic"
- ✅ Type: HR+/HER2- IDC + DCIS ✅。Stage: IV ✅。Mets: lung/LN/liver/bone ✅。Goals: palliative ✅
- ✅ second opinion: yes ✅。genetic_testing_plan: PMS2 referral ✅。Advance care: full code ✅
- ✅ Letter: second opinion + IDC "milk ducts" + DCIS "early form" + mets listed + "under good control, however progression in L breast"（letter 比 keypoints 更准确！）+ Xgeva + treatment history + PMS2 genetics + PRN F/U + emotional support。全面通俗（缺 mastectomy 讨论提及）

### ROW 52 (coral_idx 191) — 0 P1, 2 P2
- 35yo premenopausal(G0P0), L breast 1.7cm IDC grade II ER+(>95%)/PR+(>95%)/HER2-(IHC 1+, FISH 1.1), Ki-67 <10%。S/p L partial mastectomy+SLN+mastopexy: 1 SLN micromet(0.18cm, minimal ENE), margins neg。MammaPrint +0.298(low risk)。Invitae: 3 VUS。A/P: locoregional → staging CT CAP+bone scan。OFS+AI(favor per SOFT/TEXT <35yo)。Zoladex after egg harvest。Order Oncotype。Fertility preservation referral。RTC 3wks。
- P2: Stage "Stage II/III" — 来自 RxPONDER trial 讨论数据，非患者实际分期。从 pathology 推断: 1.7cm(pT1c)+micromet(pN1mi) ≈ Stage IIA
- P2: Referral-Others "None" — 遗漏 fertility preservation referral（A/P "Referral for fertility preservation asap"）。Letter 中正确提到了这个转诊
- ✅ Type: ER+/PR+/HER2- IDC ✅。Mets: No ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: OFS+AI+Zoladex after egg harvest ✅。imaging_plan: CT CAP+bone scan ✅
- ✅ genetic_testing_plan: Oncotype ✅。procedure_plan: none ✅
- ✅ **Letter 出色**: IDC "milk ducts" + ER/PR "receptors for hormones" + HER2 "protein"(**v30改进！**) + fertility preservation + CT+bone scan "check stage" + genomic test "learn more about cancer" + 3wks。准确通俗

### ROW 53 (coral_idx 192) — 0 P1, 1 P2
- 59yo, L breast IDC with neuroendocrine differentiation 4.5cm grade 3, +LVI。Core biopsy HER2-(IHC 1+) → surgical path HER2+(IHC 2+/3+ heterogeneous, FISH 4.9X)。ER+(>95%)/PR+(30%)。Ki-67 25%。DCIS 4.5cm。SLN 1/2+(6mm met)。Stage II/III。ECOG 0。A/P: high risk ~60%。AC/THP or TCHP + trastuzumab/pertuzumab 1yr + neratinib year 2 + Arimidex 10yr + breast RT + bisphosphonate + genetic counseling offered。Patient considering options。
- P2: procedure_plan 混入 RT + chemo + hormone therapy — 全部不是 procedure。应为 "No procedures planned"（or mention potential ALND under discussion）
- ✅ Type: ER+/PR+/HER2+ IDC with neuroendocrine differentiation ✅（出色，正确使用 surgical FISH 4.9X）
- ✅ Stage: II/III ✅。Mets: No ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: comprehensive（AC/THP or TCHP + neratinib + Arimidex 10yr + bone agents）✅
- ✅ radiotherapy_plan ✅。genetic_testing_plan ✅。Genetics referral ✅
- ✅ **Letter 出色**: IDC + neuroendocrine "cells that look like hormone-producing cells" + ER/PR/HER2 "proteins targeted" + chemo+hormone+targeted + Arimidex 10yr + RT after chemo + genetic counseling "inherited factors"。准确通俗

### ROW 54 (coral_idx 193) — 0 P1, 1 P2
- 39yo premenopausal, BRCA2+, oligometastatic ER+/PR-/HER2- IDC to T6 bone。S/p neoadjuvant AC→T + XRT T6 + bilateral mastectomies+L ALND: 8.2cm residual IDC grade 1(cellularity ~10%), SLN 1/24+(0.15cm)。On leuprolide+letrozole+zoledronic acid。Post-op: healing, PT started, letrozole SEs, neuropathy。A/P: continue meds, post-mastectomy RT referral, palbociclib after RT, PET/CT 3-4mo, DEXA, acupuncture for hot flashes, return 4wks。
- P2: procedure_plan "start acupuncture" — 针灸是补充疗法，不是 surgical procedure
- ✅ Type: ER+/PR-/HER2- IDC ✅。Stage: IV(oligomet T6) ✅。Goals: palliative ✅
- ✅ current_meds: leuprolide+letrozole+zoledronic acid ✅。medication_plan: comprehensive ✅
- ✅ radiotherapy_plan: post-mastectomy RT ✅。imaging_plan: PET/CT+DEXA ✅。Next visit: 4wks ✅
- ✅ **Letter 出色**: incision healing + stable + continue meds + palbociclib after RT + zoledronic acid + calcium/D + RT + acupuncture + PET/CT 3-4mo + DEXA + 4wks + emotional support。全面准确通俗

### ROW 57 (coral_idx 196) — 0 P1, 1 P2
- 59yo, locally advanced L breast cancer。Initially HER2+(IHC 3+) → neoadjuvant TCH+P x6 → lumpectomy+ALND: 3.7cm residual, 0/6 LN → surgical path **HER2 negative = TNBC!** Original biopsy review also HER2 neg。Post-op AC x4。Scheduled for XRT。Second opinion。PMH: HTN, Crohn's。Sister breast ca @60。A/P: TNBC confirmed, treatment appropriate。Rec: additional path review for HER2, genetic counseling, XRT。
- P2: procedure_plan "Rec additional path review" — 病理复查不是 surgical procedure
- ✅ Type: ER-/PR-/HER2- TNBC ✅。Stage: Locally advanced ✅。Mets: No ✅。second opinion: yes ✅
- ✅ response_assessment: "no pCR, residual 3.7cm, 0/6 LN" ✅（出色）
- ✅ radiotherapy_plan: XRT scheduled ✅。genetic_testing_plan: genetic counseling ✅
- ✅ Letter: TNBC "no receptors" + locally advanced + residual 3.7cm + dose reduced + leg swelling/neuropathy + RT + path review "confirm type" + genetic counseling。准确通俗

### ROW 59 (coral_idx 198) — 0 P1, 1 P2
- 52yo, Stage I R breast IDC grade 3, ER+(100%)/PR+(40-50%)/HER2-(FISH neg), MammaPrint High Risk。S/p lumpectomy+SLN(1.5cm,0/5LN)+TC→Abraxane/Cytoxan+XRT+tamoxifen→letrozole→exemestane(未开始)。Arthralgias on all agents。NED。Labs: postmenopausal range(FSH 32.5, E2 5)。A/P: d/c letrozole→2-3wk break→start exemestane。Duloxetine for joint pain→psychiatry referral。Mammograms alt MRI q6mo。F/U 6mo。
- P2: current_meds "exemestane" — 患者尚未开始 exemestane！原文 "has not tried it yet"。当前在/刚停 letrozole。应为 "letrozole"（正在停用中）
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: I ✅。NED ✅。Goals: curative ✅
- ✅ medication_plan: d/c letrozole→exemestane + Pristiq + duloxetine consideration ✅
- ✅ imaging_plan: mammograms alt MRI q6mo ✅。lab_summary: comprehensive ✅。Referral: psychiatry ✅
- ✅ **Letter 出色**: NED + letrozole→exemestane switch + Pristiq/duloxetine + mammogram/MRI + psychiatry + 6mo + emotional support。准确通俗

### ROW 61 (coral_idx 200) — 0 P1, 1 P2
- 43yo premenopausal(Televisit), newly dx L breast IDC ≥11mm grade 2, ER+(100)/PR+(100)/HER2-(1+)。MRI: 1.5cm + possible 2nd site(biopsy neg)。CT: LUL thickening, liver cyst。Invitae neg。ECOG 0。A/P: early-stage, lumpectomy+IORT+reconstruction 04/12/21, likely Oncotype after surgery, adjuvant endocrine(tamoxifen vs OFS+AI)。RTC after surgery/pathology。
- P2: genetic_testing_plan "None planned" — A/P "will likely need [REDACTED] Dx after surgery"(Oncotype DX genomic testing)
- ✅ Type: ER+/PR+/HER2-(1+) IDC ✅。Mets: No ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ procedure_plan: lumpectomy+IORT+reconstruction ✅。radiotherapy_plan: IORT+no post-op RT ✅
- ✅ **Letter 出色**: IDC "milk ducts" + ER/PR "sensitive to hormones" + HER2 "protein"(**v30!**) + 11mm grade 2 "moderate rate" + 2nd site biopsy neg + incidentals "not related" + endocrine "prevent coming back" + IORT "special radiation during surgery"。准确全面通俗

### ROW 63 (coral_idx 202) — 0 P1, 1 P2
- 49yo, locally advanced ER+/PR-/HER2- L breast IDC（Televisit, second opinion）。S/p neoadjuvant ddAC→abraxane → surgery 3.8x3.5cm residual grade 2 IDC(no definite path response), +LVI, positive margins→re-excision, 3/4 LN+。Letrozole + XRT。A/P: high risk, continue letrozole, E2/FSH monitoring q1-2mo, OS/oophorectomy if needed, DEXA, abemaciclib discussion。
- P2: procedure_plan 混入 lab testing（E2/FSH monitoring）— labs 不是 procedures
- ✅ Type: ER+/PR-/HER2- IDC ✅（surgical PR-）。Stage: IIIA ✅。Mets: No ✅。Goals: curative ✅
- ✅ medication_plan: comprehensive ✅。lab_plan: E2/FSH q1-2mo ✅。imaging_plan: DEXA ✅

### ROW 64 (coral_idx 203) — 0 P1, 1 P2
- 28yo premenopausal, L breast IDC 10.3x4.5x3.5cm HR+/HER2-, axillary biopsy+。Bone scan: manubrium uptake → suspicious solitary bone met（biopsy pending）。Currently on ddAC neoadjuvant。Second opinion（Televisit）。A/P: Stage III-IV, oligometastatic approach, AC→taxol→surgery+RT+sternum treatment, Xgeva if bone biopsy+。ECOG 0。Full code。
- P2: current_meds 空 — 患者正在接受 ddAC（原文 "She was started on dd AC and is tolerating okay"），应列入 current_meds
- ✅ Type: ER+/PR+/HER2- IDC ✅。second opinion: yes ✅。Goals: palliative ✅（Stage IV if confirmed）
- ✅ supportive_meds: dexamethasone+colace+olanzapine+ondansetron+compazine ✅
- ✅ findings: tumor size + axillary + bone scan ✅

### ROW 65 (coral_idx 204) — 0 P1, 1 P2
- 48yo, R breast IDC ER weak+(2%)/PR low+(7%)/HER2-(IHC 2+, FISH 1.4), Ki-67 36%, +DCIS high-grade, axillary LN micromet 0.2mm。MRI: 2.6cm mass。PET/CT: hypermetabolic nodule SUV 5.1, no distant mets。Televisit。A/P: locally advanced, neoadjuvant AC/T（or ISPY trial），port+TTE, 5-10yr endocrine after surgery, post-op RT likely。RTC 1-2wks。
- P2: response_assessment "On treatment" — 术前咨询，未开始治疗。应为 "Not yet on treatment"
- ✅ Type: ER weak+/PR low+/HER2- IDC ✅（detailed raw values）。Goals: curative ✅
- ✅ medication_plan: comprehensive（AC/T + ISPY + endocrine）✅。procedure_plan: port+biopsy+lumpectomy ✅
- ✅ imaging_plan: TTE ✅。genetic_testing_plan: F/U local results ✅。Next visit: 1-2wks ✅

### ROW 66 (coral_idx 205) — 0 P1, 1 P2
- 53yo, R breast metaplastic carcinoma with squamous differentiation, ER 5-10%/PR-(0%)/HER2-(0%), BRCA neg, GATA3+。Tumor spans ~6cm on MRI。Axillary biopsy negative。CT no distant mets。Second opinion(in-person)。A/P: affirm neoadjuvant chemo plan, discussed pembrolizumab for TNBC(meets criteria ER<10%), bilateral mastectomy considered, adjuvant RT, await Invitae testing。Exercise discussed。
- P2: response_assessment likely "On treatment" for pre-treatment second opinion consultation（与 ROW 20/29/40/49/65 相同模式）
- ✅ Type: ER 5-10%/PR 0%/HER2 0% metaplastic carcinoma with squamous differentiation ✅（出色）
- ✅ second opinion: yes ✅。Mets: No ✅。Stage: Not mentioned（acceptable for consultation）

### ROW 68 (coral_idx 207) — 0 P1, 2 P2
- 63yo postmenopausal, BRCA mutation carrier, multifocal R breast cancer（3 masses+axillary LN met+R IM LN）。S/p 6 cycles TCHP → **follow-up MRI: NO lesions!** Excellent response。Bilateral mastectomy recommended。A/P: if lumpectomy→RT, post-mastectomy RT if extensive/LN+。Sons should be tested for [Fanconi] anemia + pancreatic cancer risk。F/U PRN。
- P2: Type "multifocal [REDACTED]+ [REDACTED]" — 未解码 receptor status。从 TCHP 方案（含 trastuzumab+pertuzumab）应推断 HER2+。应为 "ER+/HER2+ breast cancer"
- P2: Letter 写 "medication-related anemia" — **错误！** 原文说 "[REDACTED] anemia"（Fanconi anemia，BRCA2 相关常染色体隐性遗传病），不是 "medication-related anemia"。[REDACTED] 被误解为 "medication"
- ✅ response_assessment: "responding well, MRI no lesions after 6 cycles TCHP" ✅（出色）
- ✅ procedure_plan: bilateral mastectomy ✅。radiotherapy_plan: conditional ✅。genetic_testing_plan: sons testing ✅
- ✅ Goals: curative ✅

### ROW 70 (coral_idx 209) — 0 P1, 2 P2
- 61yo postmenopausal, BRCA1+, bilateral breast cancer: L: ILC 4.4cm residual(cellularity 5-10%), ER+/PR+/HER2-, 2/5 SLN+。R: IDC 1cm, ER+/PR-/HER2 equivocal, 0/2 SLN。S/p bilateral mastectomies+BSO May 2020。Recovering well。Osteoporosis。A/P: resume letrozole, Prolia after dental clearance, radiation consult, expanders before RT, CT June lung nodules, RTC Sept。
- P2: medication_plan 漏了 Prolia — A/P "will start on prolia after dental clearance"（osteoporosis treatment）
- P2: procedure_plan "No procedures planned" — A/P "She is going to have expanders placed prior to radiation"（真正的 procedure）
- ✅ Type: 出色（bilateral L: ILC ER+/PR+/HER2-, R: IDC ER+/PR-/HER2-）✅。Mets: No ✅。Goals: curative ✅。Full code ✅
- ✅ lab_summary: comprehensive ✅。imaging_plan: CT June lung nodules ✅。Referral: Radiation ✅

### ROW 72 (coral_idx 211) — 0 P1, 1 P2
- 72yo postmenopausal(Televisit), ER+(99%)/PR-(<1%)/HER2-(IHC 1, FISH 1.0) L breast IDC grade 2 with focal neuroendocrine differentiation 1.2cm, 0/2 SLN, pT1cN0(sn)。Ki-67 20%。PMH: osteoporosis(Reclast)。A/P: begin letrozole, order Oncotype for chemo benefit, RTC 3wks。
- P2: response_assessment "On treatment" — letrozole 刚开处方未开始。应为 "Not yet on treatment"
- ✅ Type: ER+/PR-/HER2- IDC with neuroendocrine differentiation ✅。Stage: pT1cN0(sn) ✅
- ✅ Goals: curative ✅。medication_plan: begin letrozole ✅。supportive_meds: Reclast ✅

### ROW 73 (coral_idx 212) — 0 P1, 1 P2
- 63yo postmenopausal, Stage III L breast ER/PR+/HER2-。S/p bilateral mastectomies+L ALND+abraxane+CW XRT。On arimidex since Aug 2017。Implant exchanges → fat necrosis。New L breast nodule → bilateral US+mammo → all areas = fat necrosis。ECOG 0。A/P: continue arimidex + [REDACTED] bone agent(next Aug 2019), check labs, RTC 4mo。
- P2: medication_plan 漏了 bone agent continuation — A/P "Continue [REDACTED] [REDACTED] (next August 2019)"
- ✅ Type: ER+/PR+/HER2- ✅。Stage: III ✅。response_assessment: fat necrosis(not recurrence) ✅
- ✅ **Letter 出色**: fat necrosis "common side effect after breast surgery" + arimidex "prevent coming back" + labs + 4mo。准确通俗

### ROW 78 (coral_idx 217) — 0 P1, 1 P2 ← **补充 letter 审查**
- 79yo, de novo metastatic TNBC to liver+periportal LN（July 2017）。S/p capecitabine→OPERA trial(oral [REDACTED])→gemcitabine(d/c'd pericardial effusion+fatigue)。Off systemic therapy since 03/15/19。CT 08/19: hepatic mets enlarged(2cm from 1.3cm), portocaval node enlarged(2.9cm from 1.7cm), new 7mm lung nodule suspicious。PMH: DM, HTN, hyperlipidemia, h/o hemorrhagic brainstem CVA。Feeling great despite PD。ECOG implied functional。A/P: exploring clinical trials([REDACTED] ADC, [REDACTED]+pembrolizumab)。Not interested in standard chemo。Rad Onc consult tomorrow for liver/nodal XRT。Echo for pericardial effusion F/U。
- P2: medication_plan 列 comorbidity meds（Mag-Ox, lisinopril, norvasc）而非 cancer plan。应为 "No cancer-directed therapy; exploring clinical trials"
- ✅ Type: TNBC ✅。Stage: IV ✅。Goals: palliative ✅。current_meds: empty ✅（off systemic therapy）
- ✅ response_assessment: "Worsening of metastatic disease noting interval enlargement of hepatic and nodal metastases" ✅ — 准确
- ✅ lab_summary: comprehensive CBC+CMP(Cr 1.27(H), eGFR 40) ✅。findings: comprehensive ✅
- ✅ **Letter**: mets sites + "cancer has grown larger" + "new spot in chest" + interested in research study + radiation specialist + Mag/BP meds（same as medication_plan P2）。准确但 medication focus on comorbidity meds

### ROW 80 (coral_idx 219) — 0 P1, 2 P2 ← **补充 letter 审查**
- 53yo, dermal recurrence of high-grade IDC 7yrs post-mastectomy for DCIS（2012: L breast 0.35cm grade 2-3 DCIS → bilateral mastectomy, then 2019: 0.8cm grade 3 IDC in dermis/subcutis at core biopsy site, ER+(95%)/PR+(70%), Oncotype 24）。S/p wide excision(neg margins)。Televisit。A/P: start TC x4 on 04/11/19 with [cold cap]。6 weeks RT (5wks+1wk boost, incl L axilla+SC fields per Dr. [REDACTED])。Discussed claritin, cold gloves for neuropathy。Foundation/Tempus testing: no actionable mutations。Labs 04/04 great。RTC cycle 2。
- P2: response_assessment "On treatment" — 术前准备/chemo teaching visit，TC 未开始（start date 04/11/19）。应为 "Not yet on treatment"
- P2: procedure_plan "with [REDACTED]" — garbled（likely "with cold cap"），cold cap 不是 surgical procedure
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。Oncotype 24 ✅
- ✅ radiotherapy_plan: "6 weeks RT, 5 weeks with 1 week boost, including left axilla and SC fields" ✅ — 详细
- ✅ lab_summary: comprehensive CMP+CBC+Hep B+estradiol ✅。findings: comprehensive ✅
- ✅ **Letter**: IDC "milk ducts" + ER/PR "sensitive to hormones" + HER2 "protein"(**v30!**) + blood tests normal + TC + side effect meds。准确通俗

### ROW 82 (coral_idx 221) — 0 P1, 1 P2 ← **补充 letter 审查**
- 52yo postmenopausal, R breast mixed ductal+lobular 4.3cm G2-3, ER+/PR+/HER2-, 1/24 LN+(micromet), low risk Oncotype → no chemo。S/p R lumpectomy+SLN 11/16/2020。PMH: DM, HTN, anxiety, GERD, hyperlipidemia, liver enzyme elevation。Plan: RT → AI +/- bone med。DEXA ordered。Exercise counseling referral。Lifestyle mods（anti-inflammatory diet, exercise, stress reduction）。RTC after radiation。
- P2: Stage "Stage II" — AJCC form says "Stage IB (pT2, N0(sn))"。A/P says "Stage II"。Minor discrepancy（1/24 LN 仅 micromet, staging complex）
- ✅ Type: ER+/PR+/HER2- mixed ductal+lobular ✅（出色）。Goals: curative ✅。Full code ✅
- ✅ response_assessment: "Not yet on treatment — no response to assess"（inferred, post-surgery/pre-RT）
- ✅ **Letter**: 4.3cm mixed lobular+ductal "milk-producing glands" + ER/PR/HER2 "protein" + Stage II + curative + no chemo + RT discussed + AI after RT + DEXA。准确通俗

### ROW 83 (coral_idx 222) — 0 P1, 0 P2 ✅ ← **补充 letter 审查**
- 77yo postmenopausal, R breast ILC Grade I, ER+(from letrozole use inferred), node+(biopsy confirmed)。Neoadjuvant letrozole since Dec 2019(Televisit)。PET/CT 04/25: **significant response** — R axillary nodes dramatically decreased(SUV 15.1→1.9, size 2.3→2.1cm)。No new hypermetabolic lesions。Prior equivocal bone findings resolved。Breast masses faintly hypermetabolic, stable。PMH: Parkinson's, HTN, hypothyroid, anxiety/depression。ECOG 1。A/P: continue neoadjuvant letrozole until breast surgery。Patient interested in breast conservation → discuss with surgeon。Reassess after surgery/pathology。
- ✅ Type: ILC, ER+(inferred) ✅。Stage: III ✅。Mets: R axilla LN(regional) ✅, Distant: No ✅
- ✅ response_assessment 出色: "responding to neoadjuvant therapy, SUV 15.1→1.9" with detailed numerical data ✅
- ✅ current_meds: letrozole ✅。medication_plan: continue letrozole ✅。procedure_plan: breast surgery ✅
- ✅ **Letter 出色**: ILC "cancer that started in the milk-producing glands"（正确区分 lobular!）+ responding well + size and activity decreased + continue letrozole + breast surgery + emotional support。准确通俗

### ROW 84 (coral_idx 223) — 0 P1, 1 P2 ← **补充 letter 审查**
- 60yo, biallelic CHEK2 mutation + MS(since 1985, wheelchair-bound since 2002, on Avonex), metastatic MBC(R breast 1999→2006→metastatic 11/2019) to bone(extensive, C2 pathologic fracture→XRT)+liver(new July 2020)+meninges(dural enhancement+[REDACTED]'s cave)。ER+(71-80%)/PR-(<1%)/HER2-(IHC 2+, FISH 1.3)。S/p letrozole+palbo(1st line, PD July 2020→liver mets)→capecitabine 1500mg BID。CNS: progressive right hearing loss + right eye droop + dizziness。Televisit。A/P: repeat CT CAP, repeat LP for CSF cytology, MRI spine(r/o LMD), Rad Onc for CNS RT, continue Xeloda, consider fulvestrant+[REDACTED] inhibitor if PD on Xeloda, continue zoledronic acid。ECOG 2。
- P2: response_assessment "stable on imaging" — 但 CT 07/2020 显示 liver progression(hepatic lesions increased in size and number) + MRI brain 11/07/20 显示 CNS 可能 worsening(IAC involvement increased)。应为 "liver progression + CNS possibly worsening"
- ✅ Type: ER+/PR-/HER2- ✅。Stage: IV ✅。Mets: bone/liver/meninges ✅。Goals: palliative ✅
- ✅ current_meds: capecitabine+zoledronic acid ✅。procedure_plan: repeat LP ✅（真正的 procedure）
- ✅ **Letter**: ER+/PR-/HER2 "protein" + mets to bones/liver/"covering of brain" + CNS near nerve causing headache/numbness + marker CA 15-3 elevated + capecitabine dose change + trial options + Rad Onc + brain MRI + 2 weeks。准确通俗

### ROW 85 (coral_idx 224) — 0 P1, 3 P2 ← **重做（原标 0 P2 偷懒）**
- 61yo, Stage IIIA→IV ER+/PR-/HER2- ILC R breast, multifocal。S/p bilateral mastectomies+ddAC→T+exemestane+XRT。Metastatic within 1yr: bone+muscle+brain(L occipital→GK)+liver。1st line fulvestrant+palbociclib → PD April 2018(new liver mets)。Palliative XRT L clavicle/glenoid。Foundation One: FGFR1 amp, TMB 14。Brain MRI: NEW Meckel's cave lesion involving L trigeminal nerve + leptomeningeal disease。Symptoms: L headache, facial numbness, nausea。CA 15-3 trending down(360→45.3)。A/P: molecular tumor board → phase 1 trial [REDACTED]+olaparib。Rad Onc referral。Steroid taper。F/U 2wks。
- P2: response_assessment 遗漏 CNS 进展 — 只提 bone+liver progression，但 brain MRI 显示新 Meckel's cave 病变+leptomeningeal disease（trigeminal nerve involvement）是重要进展
- P2: imaging_plan "Brain MRI to be reviewed" — 这是已完成的影像（note 已含 UCSF 神经放射读片），不是未来计划
- P2: Letter garbled "steroid dose was lowered to day" — 应为 "to 50 mg per day"
- ✅ Type: ER+/PR-/HER2- ILC ✅。Stage: IIIA→IV ✅。Mets: bone/muscle/brain/liver ✅
- ✅ current_meds: empty ✅（between regimens）。radiotherapy_plan: Rad Onc referral ✅
- ✅ **Letter 除 garbled 外出色**: Meckel's cave "area near nerve in head, causing headache and numbness" + CA 15-3 + trial + Rad Onc + 2wks。通俗准确

### ROW 86 (coral_idx 225) — 0 P1, 1 P2 ← **重做（原标 0 P2 偷懒）**
- 53yo, metastatic breast cancer。Originally HER2+(FISH 4.37) mixed IDC Grade III → s/p TCHP+bilateral MRM+XRT+Herceptin。Metastatic recurrence Dec 2018: bone/liver/brain(dural)。**Metastatic biopsy: ER+(95%)/PR+(2%)/HER2-(1+, FISH neg) — HER2 converted!** CHEK2+。1st line letrozole+ribociclib → initial response → PD(PET 04/20: increasing bone mets)。A/P: switch to fulvestrant+/-everolimus。Palliative XRT。Denosumab。F/U 6wks。
- P2: Type "ER+/PR+/HER2+" 用了原始 receptor status（FISH 4.37），但转移活检显示 **HER2-(1+, FISH neg)**。v30 规则：最新受体状态优先。当前疾病为 ER+/PR+(low)/HER2-
- ✅ Stage: IV ✅。Mets: bone/liver/brain ✅。response_assessment: PET bone progression + brain dural enhancement ✅
- ✅ medication_plan: fulvestrant+/-everolimus + denosumab ✅。radiotherapy_plan: palliative XRT ✅
- ✅ **Letter 准确**: mets sites + brain "might be cancer" + bone grown/liver stable + fulvestrant + denosumab + XRT + 6wks

### ROW 87 (coral_idx 226) — 0 P1, 1 P2 ← **重做（充实审查内容）**
- 79yo, R breast IDC grade 2, 2.2cm multifocal(+0.6cm focus), margins clear, 4/19 ALND LN+(1 ECE), ER+/PR+/HER2-。PMH: Parkinson's disease(moderate, R side)。Family hx: daughter breast+colorectal @40, maternal GM ovarian ca。Second opinion。A/P: 40-45% recurrence risk, hormonal therapy recommended, chemo not strongly recommended(age+Parkinson's, only 3-4% additional benefit), patient prefers hormonal alone。Return to local oncologist。
- P2: Stage 空 — 从 pathology 推断: 2.2cm(pT2) + 4/19 LN+(pN2a) = Stage IIIA
- ✅ Type: ER+/PR+/HER2- IDC ✅。second opinion: yes ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ findings: comprehensive（tumor+LN+ECE+Parkinson's tremor）✅

### ROW 88 (coral_idx 227) — 0 P1, 1 P2 ← **补充 letter 审查**
- 36yo, Stage III→IV ER+/PR+/HER2- IDC(original: poorly differentiated, 4.0+2.6cm, 23/23 LN+)。**Metastatic biopsy: ER-/PR-/HER2- — receptor conversion!** Mets to brain(2 lesions, s/p craniotomy+GK)+lungs+LN。On capecitabine(Xeloda)。Video visit。A/P: recommend HER2 FISH on brain met + hormone studies + HER2 on residual disease。Continue Xeloda。
- P2: procedure_plan garbled — "I recommending doing her 2 on brain metastasis and hormone studies and her 2 on the residual dis..." 混入 HER2 testing/imaging/pathology review。不是 surgical procedure
- ✅ Type 出色：同时列出原始和转移受体状态（"ER+/PR+/HER2- IDC, metastatic biopsy ER-/PR-/HER2-"）✅
- ✅ current_meds: capecitabine ✅。Stage: III→IV ✅。Goals: palliative ✅
- ✅ **Letter**: Stage III history + 23 LN + brain/lung mets + currently on Xeloda + no new findings。准确

### ROW 90 (coral_idx 229) — 0 P1, 1 P2 ← **补充 letter 审查**
- Post-neoadjuvant follow-up(Televisit)。R breast adenocarcinoma s/p neoadjuvant AC → residual IDC 2.2cm(~60% cellularity)。Swelling in axilla noted。Port site tenderness。Constipation + mild nausea。Thyroid: elevated TSH, normal fT4 → possible hypothyroid。A/P: continue treatment, monitoring。
- P2: Type "Adenocarcinoma of right breast - ER/PR/HER2 not mentioned" — 但 response_assessment 自己写了 "residual IDC"。应为 "IDC" 而非泛称 "Adenocarcinoma"。且原文可能有受体状态（heavily redacted）
- ✅ Stage: II/III ✅。Goals: curative ✅。response_assessment: "residual IDC 2.2cm with ~60% cellularity" ✅
- ✅ **Letter**: on AC cycle 2 + residual cancer found + swelling in armpit + constipation/nausea + hypothyroid finding。准确

### ROW 91 (coral_idx 230) — 0 P1, 0 P2 ✅ ← **补充 letter 审查**
- 53yo, Stage I→IV ER+/PR+ IDC(HER2 not tested), metastatic to LN+bone。On exemestane+everolimus+denosumab。Disease progressing: MRI pelvis + PET/CT show increase in bone mets（R iliac enlarged, new sacral/L ischial/L femur lesions, bilateral hip effusions increased）。RLE swelling improved on lasix+KCl。Weight down 7 lbs。A/P: progression → consider changing therapy, discussed chemo vs endocrine options。Continue denosumab。F/U with Rad Onc for palliative bone RT。
- ✅ Type: ER+/PR+ IDC ✅。Stage: I→IV ✅。Goals: palliative ✅。current_meds: exemestane+everolimus+denosumab ✅
- ✅ response_assessment: "Disease is progressing...MRI pelvis and PET/CT show increase in bone mets and new lesions" ✅ — 详细准确
- ✅ procedure_plan: none ✅。findings: comprehensive imaging ✅
- ✅ **Letter**: bone mets growing + new areas + lasix for swelling + continue exemestane+everolimus+denosumab + drain evaluation。准确通俗

### ROW 92 (coral_idx 231) — 0 P1, 1 P2 ← **补充 letter 审查**
- 67yo, Stage IV ER+/PR+/HER2- metastatic MBC on Epirubicin C2D1。Liver smaller + less tenderness/bloating。Red dry rash around mouth(7-10 days post-treatment, no lesions/itchiness)。Denosumab q4wks。Vacation tolerated well。ECOG 1。Tumor marker pending。A/P: continue Epirubicin D1,8,15 with 2 days Neupogen。
- P2: procedure_plan 混入 chemo — "D1 Epirubicin 25 mg/m2 D1,8,15 with 2 days of Neupogen" 是化疗方案，不是 procedure
- ✅ Type: ER+/PR+/HER2- ✅。Stage: IV ✅。Goals: palliative ✅
- ✅ response_assessment: "stable on treatment, liver decreased, less tender" ✅。current_meds: epirubicin+denosumab ✅
- ✅ **Letter**: stable on treatment + liver decreased/less tender + rash around mouth + restarted Epirubicin+Neupogen。准确通俗

### ROW 94 (coral_idx 233) — 0 P1, 0 P2 ✅ ← **补充 letter 审查**
- Post-treatment follow-up(Televisit)。L breast ER+/PR+/HER2- cancer, 1.6cm, 3 positive LN。S/p surgery+chemo+RT。On letrozole。Stage redacted（acceptable — note heavily redacted）。No evidence of recurrence on imaging or exam。ECOG 0。Asymptomatic。A/P: continue letrozole, routine follow-up。
- ✅ Type: ER+/PR+/HER2- ✅。Goals: curative ✅。current_meds: letrozole ✅
- ✅ response_assessment: "No evidence of disease recurrence on imaging and exam. Asymptomatic with performance status 0" ✅
- ✅ procedure_plan: none ✅。All fields accurate。
- ✅ **Letter**: history of 1.6cm tumor + 3 positive LN + ER/PR/HER2 "protein" + no new imaging findings + currently asymptomatic + continue current meds。准确通俗

### ROW 95 (coral_idx 234) — 0 P1, 0 P2 ✅ ← **补充 letter 审查**
- 49yo, L breast ER+/PR-/HER2- IDC。S/p ISPY trial（pembrolizumab+[REDACTED] arm）→ T-AC → L lumpectomy March 2019。Pathology: good response — 3 residual foci(largest 10mm cellularity ~20%, others ~5%), 1 LN+(0.9cm met)。Residual DCIS with treatment effect。Margins negative。Stage III。Recovering without complication。A/P: adjuvant capecitabine after RT discussed。Rad Onc referral。Continue Prilosec。
- ✅ Type: ER+/PR-/HER2- IDC with residual DCIS ✅（PR 可能 post-neoadjuvant conversion）。Stage: III ✅。Goals: curative ✅
- ✅ response_assessment 出色: "achieved good response to neoadjuvant...3 smallish lesions with low cellularity and 1 LN+" 详细病理数据+MRI comparison ✅
- ✅ procedure_plan: none ✅。radiotherapy_plan: Rad Onc referral ✅
- ✅ **Letter**: MRI shows cancer decreased + breast skin thickening improved + persistent axillary adenopathy + continue Prilosec + start capecitabine after RT + Rad Onc referral。准确通俗

### ROW 97 (coral_idx 236) — 0 P1, 2 P2 ← **重做（原标 0 P2，完整审查发现 2 P2）**
- 53yo premenopausal(Televisit), L breast 0.8cm grade 1 IDC, ER+(>95%)/PR+(~60%)/HER2-(IHC 1), Ki-67 ~10%。S/p L partial mastectomy+SLN: margins neg, 0/3 LN, pT1bN0(sn)。PMH: **MS relapsing-remitting** on GILENYA(fingolimod)。Drain in place(patient concerned)。A/P: very low risk → Oncotype Dx ordered, anticipate no chemo。Strongly recommend adjuvant endocrine therapy with AI。Rad Onc eval referred。Coordinate with MS team re GILENYA。RTC 3-4wks after Oncotype。
- P2: medication_plan "Continue GILENYA regimen" — 关注 MS 用药而非 cancer medication plan。A/P 明确 "strongly recommend adjuvant endocrine therapy with [REDACTED/AI]"。应捕获 cancer medication plan
- P2: radiotherapy_plan "None" — 但 A/P 说 "pt needs Rad Onc eval; was referred"。Lumpectomy 后 RT 是标准治疗。Referral 字段正确捕获了 Rad Onc eval，但 radiotherapy_plan 不应为 "None"
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: pT1bN0(sn) ✅。Goals: curative ✅。response_assessment: "Not yet on treatment" ✅
- ✅ genetic_testing_plan: molecular profiling (Oncotype) ✅。Referral: Rad Onc ✅
- ✅ **Letter**: IDC + early stage + 0.8cm + surgery successful + drain concern + continue GILENYA for MS + no changes expected。准确（GILENYA/MS 确认 NOT hallucination）

### ROW 100 (coral_idx 239) — 0 P1, 0 P2 ✅ ← **补充 letter 审查**
- 68yo, Stage IV metastatic MBC ER+(80%)/PR+(50%), HER2 not tested。On Gemzar C2D8（cancelled by patient due to fatigue/confusion about treatment）。Tumor markers CA 15-3 and CA 27.29 **increased**。Patient very confused and concerned about markers and scans — has only received 3 doses。Labs: anemia, low WBC, elevated LFTs。A/P: marker increase could be true progression OR tumor flare → too early to tell after only 3 doses → continue Gemzar → re-evaluate with next cycle。Reassured patient。
- ✅ Type: ER+(80%)/PR+(50%) IDC, HER2 not tested ✅（raw values）。Stage: IV ✅。Goals: palliative ✅
- ✅ current_meds: Gemzar ✅（though cancelled this cycle）。procedure_plan: none ✅
- ✅ **response_assessment 出色**: "Tumor markers increased. No significant physical exam findings. Current picture unclear if progressing or tumor flare" — 准确且有临床细微区分，反映了真实的不确定性 ✅
- ✅ **Letter**: tumor markers increased "proteins in blood gone up, can indicate cancer growing" + low RBC/WBC "feel tired, risk of infections" + Gemzar cancelled due to fatigue + no new imaging。准确通俗

---

## 最终审查总结

| 指标 | v30 | v29 | 变化 |
|------|-----|-----|------|
| **P0** | 0 | 0 | — |
| **P1** | 1 (1.6%) | 0 | +1 |
| **P2** | 93 (含 3 个🩺医生意见) | 92 | +1 (+1.1%) |
| **P2 rate** | 1.52/sample | 1.51/sample | +0.7% |
| **完美 samples (0 P2)** | 10 | ? | — |

### P1 详情
- **ROW 46**: Sarcoidosis（endobronchial biopsy确认非坏死性肉芽肿性炎症）被误分类为癌症转移 → Stage IV/palliative 全部错误。A/P 以 curative intent 治疗

### 最常见 P2 类别
1. **procedure_plan 字段混入**（labs/imaging/chemo/acupuncture）~12 ROWs
2. **response_assessment "On treatment" 对 pre-treatment 患者** ~9 ROWs
3. **Letter [REDACTED] garbling**（"a medication"）~5 ROWs
4. **Stage 遗漏/可推断但未推断** ~5 ROWs
5. **lab_plan/lab_summary 遗漏** ~5 ROWs
6. **Referral 字段遗漏** ~4 ROWs
7. **current_meds 遗漏** ~4 ROWs
8. **medication_plan 内容不当**（comorbidity meds 或 treatment history 而非 plan）~3 ROWs

### v30 改进确认
- ✅ HER2 described as "protein" in letters（v29 有时说 "drug" 或 "gene"）
- ✅ procedure_plan 大多不混入 chemo（v29 更严重）
- ✅ response_assessment 对 pre-treatment 案例更准确（"Not yet on treatment" 规则生效）
- ✅ Letter [REDACTED] handling 改进（部分 garbled text 变为 "a medication" 而非完全 garbled）

### v30 回归
- ROW 8: response_assessment "Not yet on treatment" 过度应用于 post-neoadjuvant 患者
- ROW 46: 首个 P1（v29 没有 P1）— 但这是一个极复杂案例（cancer + sarcoidosis 共存）


