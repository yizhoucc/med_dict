# V2 逐行深度审查

审查文件: `results/full_qwen_20260316_075405/progress.json`
审查方法: 逐行、逐字段、对照原文 + 归因引用

---

### Row 0 (coral_idx=140)

**患者概况**: 56F, 2013年右乳多灶性IIA期乳癌(ER+/PR+/HER2-, G2)行乳切+前哨淋巴结活检,拒绝tamoxifen,未化疗/放疗。2019.12 CT发现广泛转移(肺、腹膜、肝、卵巢、右腋窝复发)。初次肿瘤内科会诊。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | "RTC with me after..." | ✓/P2归因 | 值正确(原文"New Patient Evaluation"),但归因引用了follow-up文本,应引用"Patient presents with New Patient Evaluation" |
| second opinion | "no" | "RTC with me after..." | ✓/P2归因 | 值正确,归因不相关 |
| in-person | "in-person" | "RTC with me after..." | ✓/P2归因 | 值正确(有体检和生命体征),归因不相关 |
| summary | 详细准确 | "Metastatic relapse..." | ✓ | 准确包含了癌症类型、分期和就诊目的 |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | "2.4 and 2.3 cm tumors...her 2 negative" | **P1** | **缺HER2-状态**。原文HPI"her 2 neu negative",A/P"her 2 negative"。应为"ER+/PR+/HER2- invasive ductal carcinoma"。讽刺的是归因引用本身包含"her 2 negative"但提取值遗漏了 |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | "Metastatic relapse..." | ✓ | 准确 |
| Metastasis | "Yes (to lungs, peritoneum, liver, and ovary)" | "Metastatic relapse..." | ✓ | 准确匹配CT发现 |
| Distant Metastasis | "Yes (to lungs, peritoneum, liver, and ovary)" | "Metastatic relapse..." | ✓ | 准确。右腋窝是区域性,未算入distant,正确 |
| lab_summary | "No labs in note." | - | ✓ | 原文"No visits with results within 1 Month(s)",唯一结果是2001年的HCG,无关 |
| findings | CT findings + 体检(肝大、网膜肿块、腋窝3cm肿块) | CT impression text | ✓ | 非常全面,包含了所有CT发现和体检发现 |
| current_meds | "" | - | ✓ | 原文"No current outpatient medications on file",患者拒绝了tamoxifen |
| recent_changes | "" | - | ✓ | 未开始治疗,无变化 |
| supportive_meds | "" | - | ✓ | 无支持性用药 |
| goals_of_treatment | "palliative" | "Metastatic relapse..." | ✓/P2归因 | 值正确(原文HPI"treatment would be palliative"),但归因应引用该句而非"Metastatic relapse" |
| response_assessment | 长文重复CT findings | "Metastatic relapse..." | **P1** | **答非所问**。患者从未接受系统治疗(仅手术,拒绝tamoxifen),按prompt规则"surgery alone does not count",应写"Not yet on treatment — no response to assess." 当前输出重复了findings的内容 |
| Medication_Plan_chatgpt | hormonal therapy: ibrance + [REDACTED] if HR+/HER2- | A/P item 4 | ✓ | 准确(ibrance是CDK4/6抑制剂,分类为hormonal therapy略有不精确但可接受) |
| medication_plan | "If HR+/HER2 negative on biopsy, will recommend ibrance and [REDACTED]." | A/P item 4 | ✓ | 准确 |
| therapy_plan | 同上 | A/P item 4 | ✓ | 准确 |
| radiotherapy_plan | "None" | - | ✓ | A/P未提及放疗 |
| procedure_plan | "Appointment with Dr. [REDACTED]...to biopsy mass in right axilla" | A/P item 3 | ✓ | 准确 |
| imaging_plan | "No imaging planned." | - | **P1** | **遗漏**。HPI明确写"I also ordered a MRI of brain and bone scan",Diagnosis section列出"NM Whole Body Bone Scan"和"MR Brain with and without Contrast"。但A/P中未提及→pipeline限制(plan extraction仅看A/P段) |
| lab_plan | "No labs planned." | - | **P1** | **遗漏**。Diagnosis section列出CBC, CMP, CA 15-3, CEA, aPTT, PT。HPI"I also ordered...labs to complete her work up"。同上,A/P中未明确列出→pipeline限制 |
| genetic_testing_plan | "No specific genetic...plan is to biopsy...to confirm HR and HER2 status." | biopsy appointment | P2 | 无genetic test计划正确,但不必要地提及biopsy(那是procedure,不是genetic test) |
| Referral-Nutrition | "None" | - | ✓ | - |
| Referral-Genetics | "None" | - | ✓ | - |
| Referral-Specialty | "[REDACTED] [REDACTED] Referral asap, Integrative Medicine       History of Present Illness:   56" | "Medical Oncology/Hematology" | **P0** | **格式错误/文本泄漏**。提取值包含"History of Present Illness: 56"这段垃圾文本。原文有"Ambulatory Referral to Integrative Medicine"和"***** ***** Referral asap",应为"Integrative Medicine, [REDACTED] Referral"。归因引用的是医生签名而非转诊文本 |
| Referral-Others | "None" | - | ✓ | - |
| follow up | "RTC with me after completed work up to formulate a plan" | A/P item 4 | ✓ | - |
| Next clinic visit | "in-person: after completed work up to formulate a plan" | same | ✓ | - |
| Advance care | "Full code." | - | ✓ | 原文"Advance Care Planning Full code." |

**白名单笔记**:
- (无需更新)

**本行总结**: 1×P0, 4×P1, 5×P2
- P0: Specialty referral文本泄漏("History of Present Illness: 56")
- P1: (1) Type_of_Cancer缺HER2- (2) response_assessment答非所问 (3) imaging_plan遗漏MRI+bone scan (4) lab_plan遗漏CBC/CMP/markers — 后两项为pipeline限制(信息在Diagnosis/HPI段而非A/P段)
- P2: 多个归因引用不精确(Patient type、second opinion、in-person、goals_of_treatment引用了错误文本)

---

### Row 1 (coral_idx=141)

**患者概况**: 44F, Lynch综合征,多重癌症:①2013年右乳TNBC(ER-/PR-/HER2-)IIB期,新辅助化疗+部分乳切+辅助化疗;②2018.3右胸壁TNBC局部复发→gem/carb有效;③2018.5结肠脾曲腺癌I期(pT2)→结肠切除;④子宫内膜样腺癌FIGO1。2019.1 PET示转移(肝、骨、胸壁、节)→活检确认TNBC转移→Abraxane+Pembro→2019.5进展→2019.6.30开始irinotecan。本次: cycle 3 day 1 (2019.8.11)。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "F/u 2 weeks" | ✓ | 正确,持续在此治疗 |
| second opinion | "no" | "Referral made to SMS/ABC..." | ✓/P2归因 | 值正确,归因不相关 |
| in-person | "in-person" | "45 minutes face-to-face" | ✓ | 值和归因都正确 |
| summary | "44 y.o. female with metastatic breast cancer, presenting for cycle 3 day 1 of irinotecan" | HPI开头 | ✓ | 准确具体 |
| Type_of_Cancer | "ER-/PR-/HER2- triple negative invasive ductal carcinoma" | "metastatic breast cancer" | ✓/P2归因 | 值正确(原文"TNBC"多处提及),归因应引用"ER-/PR-/*****- (TNBC)"而非泛称 |
| Stage_of_Cancer | "Originally Stage IIB, now metastatic (Stage IV)" | "newly diagnosed metastatic breast cancer" | ✓ | 准确 |
| Metastasis | "Yes (to liver, bone, and chest wall)" | "Chest wall more tender..." | ✓ | 肝(caudate+seg VII)、骨(S1+腰椎+骨盆+clivus+petrous+双侧股骨颈)、胸壁。主要站点均覆盖 |
| Distant Metastasis | "Yes (to liver, bone, and chest wall)" | 同上 | P2 | 胸壁是局部复发(regional),不算distant metastasis。Distant应为肝、骨。但不影响临床判断 |
| lab_summary | 列出15项异常值(Alb 2.1, ALP 183, Na 124等) | 实验室结果原文 | ✓/P2 | 值全面准确。P2:遗漏提及Platelet Count被redacted |
| findings | 非常详尽:胸壁感染征象、背痛恶化、贫血、HepB状态、MRI骨病灶、体检发现、实验室异常 | A/P各条 | ✓ | 全面客观 |
| current_meds | "Irinotecan" | irinotecan剂量变更文本 | ✓ | 唯一当前在用的肿瘤药物 |
| recent_changes | irinotecan改为q2w 150mg/m2 d1,15 q28d | 同上 | ✓ | 准确匹配A/P |
| supportive_meds | ondansetron, compazine, imodium, lomotil, oxycodone, morphine | - | ✓ | 全部在active med list中,全部为癌症治疗相关支持用药(止吐、止泻、癌痛) |
| goals_of_treatment | "palliative" | "metastatic breast cancer" | ✓ | Stage IV,clearly palliative |
| response_assessment | 引用2019.1和2019.5的PET+2019.6 MRI作为"progressing"证据 | "Mastectomy not mentioned" | **P1** | **使用了irinotecan开始前的影像**。PET(1月/5月)和MRI(6月)都在irinotecan(6/30)之前。当前更相关的是:A/P"back pain worse which could be due to PD"+"chest wall more tender"→临床进展迹象,但无新影像。归因"Mastectomy not mentioned"是幻觉 |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | **遗漏**。A/P有明确药物计划(irinotecan剂量变更、doxycycline、effexor增量等)但chatgpt格式为空 |
| medication_plan | Doxycycline, MS Contin, flexeril, oxycodone, effexor, NS IV, KCl, pRBC | A/P对应文本 | **P1** | **遗漏irinotecan剂量变更**——本次最重要的药物变更(改为q2w 150mg/m2)未在medication_plan中列出 |
| therapy_plan | irinotecan改为q2w 150mg/m2 + "oxycodone 5mg prn q4h" | irinotecan变更文本 | ✓/P2 | irinotecan部分正确。P2: oxycodone是止痛药不是therapy,不应在therapy_plan中 |
| radiotherapy_plan | "urgently needs Rad Onc...will send message to Dr." | 同上 | ✓ | 匹配A/P。prompt允许"referrals to Rad Onc as radiotherapy-related plan" |
| procedure_plan | "No procedures planned." | - | ✓ | A/P中无手术/有创操作计划 |
| imaging_plan | "Scans again in 3 months, due September 2019. MRI brain if worse." | A/P对应文本 | ✓/P2 | 准确。P2: "Scans"不够具体(未指定CT/PET),但原文本身就这么写的 |
| lab_plan | "Monitor ALT, HBsAg, HBV DNA every 4 months, next due October 2019" | A/P Hep B部分 | ✓ | 准确 |
| genetic_testing_plan | "None planned." | - | ✓ | A/P无genetic test计划 |
| Referral-Nutrition | "None" | - | ✓ | - |
| Referral-Genetics | "None" | - | ✓ | - |
| Referral-Specialty | "Rad Onc consult" | "will send message to Dr." | ✓ | A/P"Needs to get in for XRT consult" |
| Referral-Others | "Social work, Home health" | "Referral made to SMS/ABC...Home health?" | P2 | SMS/ABC是"previously made but never got auth, consider re-referring"(不确定);"Home health?"是问号(考虑中)。两者都不是确定的转诊 |
| follow up | "2 weeks" | "F/u 2 weeks" | ✓ | - |
| Next clinic visit | "in-person: 2 weeks" | 同上 | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | 笔记中无advance care/code status提及 |

**白名单笔记**:
- irinotecan — 应在oncology_drugs.txt中(确认)
- rivaroxaban (Xarelto) — 抗凝药,非肿瘤药物,正确排除在current_meds外
- doxycycline — 抗生素(用于胸壁蜂窝织炎),不是肿瘤药物也不是支持性用药,正确不含
- flexeril (cyclobenzaprine) — 肌松药用于骨转移疼痛管理,可考虑加入supportive_care_drugs.txt

**本行总结**: 0×P0, 3×P1, 6×P2
- P1: (1) response_assessment使用irinotecan前的旧影像而非当前临床征象 (2) Medication_Plan_chatgpt为空 (3) medication_plan遗漏irinotecan剂量变更
- P2: (1) Distant Metastasis含胸壁(local) (2) lab_summary未提platelet redacted (3) therapy_plan含oxycodone (4) Referral-Others含不确定转诊 (5) imaging_plan中"Scans"不够具体 (6) response_assessment归因"Mastectomy not mentioned"是幻觉

---

### Row 2 (coral_idx=142)

**患者概况**: 53F, 乳腺X线发现右乳上外象限癌。IDC 1.7cm(10点钟方向),腋窝淋巴结1.5cm阳性。ER+/PR+/HER2 IHC 2+ FISH阴性(=HER2-)。Stage IIA。已寻求多个意见,本次为video远程肿瘤内科会诊。PET CT和基因检测已安排/待结果。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | "New Patient Evaluation"+"Consult Note" |
| second opinion | "yes" | "[REDACTED] sent and is pending" | ✓/P2归因 | 值正确("She has had several opinions and is here for a medical oncology consult"),但归因引用了无关文本 |
| in-person | "Televisit" | telehealth Zoom text | ✓ | "Video Consult",归因正确 |
| summary | 准确具体(Stage IIA IDC, ER+, HER2-, node+, consult for neoadjuvant) | A/P#1 | ✓ | 优秀 |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | A/P#1 | ✓ | 准确。HER2 IHC 2+/FISH neg = HER2-,正确处理 |
| Stage_of_Cancer | "Stage IIA" | A/P#1 | ✓ | 匹配staging form和A/P |
| Metastasis | "No" | A/P#1 | ✓ | 早期,PET待结果 |
| Distant Metastasis | "No" | A/P#1 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | "No results found for any previous visit" |
| findings | 1.7cm肿瘤+1.5cm阳性淋巴结+receptor status+PET pending | A/P#1 | ✓/P2 | 肿瘤大小和淋巴结是合理的findings。P2:重复了receptor status(属于Cancer_Diagnosis) |
| current_meds | "" | - | ✓ | "No current outpatient medications on file" |
| recent_changes | "" | - | ✓ | 未开始治疗 |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | "surgery and radiation...local recurrence" | ✓ | Stage IIA+讨论neoadjuvant→curative intent |
| response_assessment | "Not yet on treatment — no response to assess." | 同上 | ✓ | 正确 |
| Medication_Plan_chatgpt | chemo + hormonal blockade讨论 | A/P#2,#4 | ✓ | 准确捕获讨论(无具体药物) |
| medication_plan | "None" | - | ✓ | 未做具体药物决定,等待PET和基因结果 |
| therapy_plan | 讨论chemo+surgery+radiation角色 | A/P#2,#3 | ✓ | 匹配(都是讨论,非具体计划) |
| radiotherapy_plan | "discussed the role of surgery and radiation" | A/P#3 | ✓ | prompt允许"discussed"作为radiotherapy plan |
| procedure_plan | "No procedures planned." | - | ✓ | A/P无具体手术计划 |
| imaging_plan | "PET imaging follow up after results are back." | A/P#7 | ✓/P2 | PET已安排(HPI),A/P是等结果后follow up。措辞有歧义("PET imaging"像是计划做PET,实际是等结果) |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "Genetic testing sent and is pending." | A/P#6 | ✓ | 准确匹配A/P#6 |
| Referral-Nutrition | "None" | - | ✓ | A/P#5有饮食建议但不是转诊,按prompt规则正确 |
| Referral-Genetics | "Genetic testing sent and is pending." | A/P#6 | **P1** | **应为"None"**。这是基因检测order,不是转诊到genetics clinic/counselor。prompt明确:"ONLY outgoing referrals TO a genetics clinic or counselor"。检测已在genetic_testing_plan中覆盖 |
| Referral-Specialty | "None" | - | ✓ | - |
| Referral-Others | "None" | - | ✓ | - |
| follow up | "[REDACTED] follow up after pet and [REDACTED] are back." | A/P#7 | ✓ | - |
| Next clinic visit | "telehealth: after PET and [REDACTED] are back" | 同上 | ✓ | - |
| Advance care | "full code." | - | ✓ | "Code status: full code." |

**白名单笔记**:
- (无需更新)

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: Referral-Genetics应为"None"(基因检测order不等于genetics clinic referral)
- P2: (1) second opinion归因引用无关文本 (2) findings重复receptor status (3) imaging_plan措辞有歧义

---

### Row 3 (coral_idx=143)

**患者概况**: 75F, ER+/PR+/HER2-(IHC 2+/FISH neg)左乳IDC, 2016.10左乳切+SLN活检(2.8cm G2 IDC)。Letrozole自2016.12至今。Follow-up约2年后(2018.9),无复发证据。另有骨质疏松(osteopenia range)、头痛、肌肉痉挛等。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "Follow up in 6 months..." | ✓ | 正确 |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "40 minutes face-to-face" | ✓ | 有体检+生命体征 |
| summary | 详细准确(包含诊断、手术、letrozole、follow-up目的) | HPI | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | A/P开头 | ✓ | 准确。biopsy ER+/PR+/HER2-; surgical path HER2 IHC2+/FISH neg=HER2- |
| Stage_of_Cancer | "Not mentioned in note" | - | P2 | 分期在staging form中被redacted("Stage *****")。但有术后病理2.8cm(T2),SLN结果未记录阳性/阴性。无法推断完整分期但可注明T2。prompt允许从肿瘤大小推断"approximately Stage I-II" |
| Metastasis | "No" | "without evidence of disease recurrence" | ✓ | - |
| Distant Metastasis | "No" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 确实无实验室结果 |
| findings | 无复发证据 + 骨密度扫描结果(T-score -2.4) | A/P#1 | ✓ | 包含了imaging findings(DEXA)和临床状态,合适 |
| current_meds | "Letrozole" | "currently on Letrozole since 12/02/2016" | ✓ | 唯一当前oncologic药物 |
| recent_changes | "" | - | ✓ | 无变化,继续同方案 |
| supportive_meds | "letrozole" | "Letrozole 2.5 mg daily" | **P1** | **letrozole是oncologic药物**。prompt明确:"Do NOT include ONCOLOGIC drugs (tamoxifen, letrozole, anastrozole...) in supportive_meds. These go in current_meds." 应为空 |
| goals_of_treatment | "curative" | "without evidence of recurrence" | ✓/P2 | 可接受。prompt说"Completed curative treatment, now on surveillance endocrine therapy → adjuvant or risk reduction"。"curative"不算错但"adjuvant"更精确 |
| response_assessment | "without any evidence of disease recurrence on imaging, exam, and review of systems" | 同上 | ✓ | 正确。prompt明确"for patients on adjuvant/maintenance therapy, 'no evidence of recurrence' IS a valid response assessment" |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | **遗漏**。A/P有明确药物计划(continue letrozole, calcium/VitD, OTC Mg, conditional Prolia) |
| medication_plan | Continue letrozole + calcium/VitD + OTC Mg + conditional Prolia | A/P各条 | ✓ | 全面准确 |
| therapy_plan | Continue letrozole + conditional Prolia | A/P | ✓ | letrozole=hormonal therapy, Prolia=bone therapy,都属于therapy |
| radiotherapy_plan | "None" | - | ✓ | - |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | mammogram due July 2019 + DEXA due July 2019 + "Brain MRI" | A/P#1,#5,#6 | ✓/P2 | mammogram和DEXA正确。P2: "Brain MRI"缺条件限定词——原文是"If worsening, consider brain MRI",是conditional |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral-Nutrition | "None" | - | ✓ | probiotics建议不是营养转诊 |
| Referral-Genetics | "None" | - | ✓ | - |
| Referral-Specialty | "None" | - | ✓ | 牙科是conditional("if BMD drops"),PCP follow-up不是specialty |
| Referral-Others | "None" | - | ✓ | PT是已完成的,不是新转诊 |
| follow up | "6 months or sooner if new or worsening symptoms" | A/P末尾 | ✓ | - |
| Next clinic visit | "in-person: 6 months or sooner" | 同上 | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | 笔记中无code status或advance care提及 |

**白名单笔记**:
- letrozole出现在supportive_meds中 → 说明letrozole未在oncology_drugs黑名单中被阻止,需确认oncology_drugs.txt包含letrozole
- Prolia (denosumab) — bone therapy,应在oncology_drugs.txt中

**本行总结**: 0×P0, 2×P1, 3×P2
- P1: (1) supportive_meds含letrozole(oncologic药物不应在supportive_meds) (2) Medication_Plan_chatgpt为空
- P2: (1) Stage_of_Cancer可从T2推断近似分期 (2) goals_of_treatment用"curative"不如"adjuvant"精确 (3) imaging_plan中Brain MRI缺"if worsening"条件限定

---

### Row 4 (coral_idx=144)

**患者概况**: 31F(绝经前), 2013年左乳8.0cm G2 IDC micropapillary (ER+/PR+/HER2-, Ki67 5%), Stage III, 16 LN+/ECE. 双乳切+腋清。AC×3(因副作用停)。2015 Lupron+exemestane/tamoxifen。2018.6左乳局部复发(IDC ER+/PR+/HER2- IHC1+, Ki67 30-40%)。2019.1开始Lupron+anastrozole+palbociclib。本次12/7/2019 video follow-up: MRI脑正常,MRI颈椎示左5B淋巴结增大+臂丛受累。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "returning to clinic while on leuprolide..." | ✓ | 正确 |
| second opinion | "no" | "Continue current therapy..." | ✓/P2归因 | 值正确,归因不相关 |
| in-person | "Televisit" | telehealth text | ✓ | video encounter |
| summary | "follow-up visit to discuss ongoing therapy and symptom management" | A/P | P2 | 准确但偏泛化。可更具体:含MRI新发现(臂丛受累)、当前方案(palbociclib/lupron/anastrozole) |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | A/P | ✓ | 准确 |
| Stage_of_Cancer | "Originally Stage III, now metastatic (Stage IV)" | A/P | ✓ | 匹配"Stage III"+"metastatic recurrence" |
| Metastasis | "Yes (to left cervical LN, left internal mammary LN, and sternum)" | A/P | ✓ | 颈部LN经活检证实;内乳LN和胸骨来自之前影像 |
| Distant Metastasis | 同上 | 同上 | ✓ | 左颈部LN对乳腺癌属distant |
| lab_summary | "No labs in note." | - | P2 | 有一个creatinine 0.6 (2019.8.23),约3.5个月前,正常范围。虽临床意义有限,但技术上是存在的lab value |
| findings | MRI脑正常 + MRI颈椎5B节增大/臂丛受累 + CT颈部(颈LN缩小+纵隔LN稳+腋LN增大) + bone scan(胸骨可疑) + 痛评4分 | 多源 | ✓ | 全面,包含各影像和体检 |
| current_meds | "anastrozole, palbociclib, leuprolide" | "Continue current therapy" | ✓ | 匹配A/P和HPI(ibrance/lupron/[redacted]) |
| recent_changes | "" | - | ✓ | "Continue current therapy" |
| supportive_meds | "ondansetron (ZOFRAN) 8 mg" | - | ✓ | HPI"refilled her zofran...uses only occasionally"→止吐,治疗相关 |
| goals_of_treatment | "palliative" | "biopsy proven metastatic recurrence" | ✓ | Stage IV转移性,palliative正确 |
| response_assessment | CT颈(LN缩小+LN增大混合) + bone scan(胸骨可疑) + MRI颈椎(5B节增大) | - | ✓ | 具体引用了影像证据,展示mixed response |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | **遗漏**。A/P有"Continue current therapy"等药物计划 |
| medication_plan | "Continue leuprolide, anastrozole, and palbociclib." | A/P | ✓ | 准确 |
| therapy_plan | Continue current therapy + rad onc referral + labs monthly | A/P | ✓/P2 | 前两项正确。P2: "Labs monthly"不属于therapy,应在lab_plan |
| radiotherapy_plan | "Radiation oncology referral for symptomatic disease in left neck and brachial plexus" | A/P#3,#6 | ✓ | 匹配A/P |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "CT and bone scan ordered, prior to next visit" | A/P#5 | ✓/P2 | 匹配A/P。P2: Diagnosis section列出CT Abdomen/Pelvis, CT Chest, CT Neck, Bone Scan——更具体,但pipeline只看A/P |
| lab_plan | "Labs monthly, on the day of lupron injection" | A/P#4 | ✓/P2 | 匹配A/P。P2: Diagnosis section列出CBC, CMP, Estradiol, CA15-3, CEA——更具体 |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral-Nutrition | "None" | - | ✓ | - |
| Referral-Genetics | "None" | - | ✓ | - |
| Referral-Specialty | "Radiation oncology referral, Radiation Oncology    CT Abdomen /Pelvis with Contrast    CT" | "Radiation oncology referral" | **P0** | **文本泄漏**。应为"Radiation oncology"。提取值包含"CT Abdomen /Pelvis with Contrast    CT"等imaging order文本——与Row 0同类问题 |
| Referral-Others | "None" | - | ✓ | - |
| follow up | 混杂了rad referral + labs + CT/bone scan | A/P多条 | P2 | 应仅为follow up时间/安排,混入了其他计划内容 |
| Next clinic visit | "in-person: prior to next visit with restaging studies" | A/P | ✓/P2 | "in-person"不确定——本次是telehealth,下次未指定 |
| Advance care | "full code." | - | ✓ | "Code status: full code" |

**白名单笔记**:
- palbociclib (ibrance) — CDK4/6抑制剂,应在oncology_drugs.txt中
- anastrozole — AI,应在oncology_drugs.txt中
- leuprolide (lupron) — GnRH agonist,应在oncology_drugs.txt中

**本行总结**: 1×P0, 1×P1, 7×P2
- P0: Specialty referral文本泄漏(含imaging order文本)——与Row 0同类系统性问题
- P1: Medication_Plan_chatgpt为空
- P2: summary偏泛化、lab_summary遗漏creatinine、therapy_plan含labs、imaging_plan不够具体、lab_plan不够具体、follow up混杂内容、Next clinic visit格式推测

---

### Row 5 (coral_idx=145)

**患者概况**: 34F, 2018.12自检发现右乳肿块。2019.3活检: G1 IDC, ER+(>95%)/PR(~90%)/HER2 IHC2+ FISH非扩增(=HER2-), Ki67 ~10%。MammaPrint低风险。2019.6.8 zoladex, 2019.6.21双乳切+扩张器(右乳15×10mm G1 IDC, DCIS, 0/1 node neg)。2019.7.5开始letrozole。本次为术后辅助治疗follow-up。另有双相情感障碍。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | '{"quote":"NOT_IN_NOTE"}' | **P1** | **应为"Follow up"**。原文Chief Complaint明确写"Follow-up"。患者已在此就诊(zoladex 6/8即此处开的)。归因自己也找不到支持"New patient"的文本 |
| second opinion | "no" | "RTC 3 months..." | ✓ | - |
| in-person | "in-person" | "RTC 3 months..." | ✓/P2归因 | 有详细体检(乳房、疤痕)和生命体征。但笔记开头有"video encounter"表述——可能是模板语言,实际是in-person |
| summary | 准确(34F ER/PR+ IDC, post-mastectomy, adjuvant therapy) | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | "1.5 cm node neg, grade 1 and ER/PR+ IDC" | **P1** | **缺HER2-**。活检"HER2 equivocal(IHC 2+), FISH non-amplified"=HER2 negative。应为"ER+/PR+/HER2- invasive ductal carcinoma" |
| Stage_of_Cancer | "Approximately Stage I-II (1.5 cm, 0/1 nodes)" | 同上 | **P1** | **应为Stage I**。prompt明确规则:"Node-negative with small tumor (≤2cm) = Stage I, NOT Stage II"。1.5cm + 0/1 node neg = pT1cN0 = Stage I |
| Metastasis | "No" | "1.5 cm node neg" | ✓ | - |
| Distant Metastasis | "No" | 同上 | ✓ | - |
| lab_summary | 详尽列出CMP+CBC+Estradiol+VitD所有值 | "Estradiol monthly" | ✓/P2 | 全面但冗长。P2: Platelet和Sodium的值被redacted但提取只保留了redacted标记,未按prompt要求写"Values redacted" |
| findings | 手术结果(1.5cm G1 IDC, 0/1 nodes) + 体检(乳切愈合, 无淋巴结肿大) | A/P | ✓ | 合适的客观发现 |
| current_meds | "letrozole, zoladex" | "Start [REDACTED]...Estradiol monthly" | ✓ | zoladex 6/8开始(~1月前),letrozole刚开始/正在讨论开始。med list中两者都在 |
| recent_changes | "Started letrozole today." | "Start [REDACTED] per her request" | ✓ | 匹配A/P |
| supportive_meds | "" | - | ✓ | gabapentin可能与精神科相关而非肿瘤治疗副作用,排除合理 |
| goals_of_treatment | "curative" | 多段引用 | ✓ | 早期+手术+辅助内分泌→curative正确 |
| response_assessment | "Not mentioned in note." | - | ✓ | 刚手术+刚开始辅助治疗,太早无法评估响应,无影像标记物 |
| Medication_Plan_chatgpt | hormonal therapy: letrozole + estradiol | A/P | ✓ | 有内容(非空),不像前几行 |
| medication_plan | letrozole + [REDACTED] 3年→tamoxifen + gabapentin prn + "Estradiol monthly" | A/P | ✓/P2 | P2: "Estradiol monthly"可能是月度estradiol血检(lab plan),非药物 |
| therapy_plan | letrozole today + 3年→tamoxifen | A/P | ✓ | 准确 |
| radiotherapy_plan | "None" | - | ✓ | - |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "Estradiol monthly." | A/P | ✓ | 每月estradiol水平监测(评估卵巢抑制效果) |
| genetic_testing_plan | "None planned." | - | ✓ | Myriad已做(阴性) |
| Referral各项 | 全部"None" | - | ✓ | - |
| follow up | "RTC 3 months or sooner as needed" | A/P | ✓ | - |
| Next clinic visit | "in-person: 3 months or sooner as needed" | 同上 | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**:
- zoladex (goserelin) — GnRH agonist,应在oncology_drugs.txt中
- letrozole — AI,已在Row 3中确认
- estradiol vaginal ring — 非肿瘤药物(用于阴道萎缩),正确不含在current_meds中

**本行总结**: 0×P0, 3×P1, 3×P2
- P1: (1) Patient type应为"Follow up"非"New patient" (2) Type_of_Cancer缺HER2- (3) Stage_of_Cancer应为Stage I而非"approximately Stage I-II"
- P2: (1) lab_summary redacted值未标注 (2) medication_plan含"Estradiol monthly"(可能是lab test) (3) in-person归因弱

---

### Row 6 (coral_idx=146)

**患者概况**: 女性, 1998年Stage II T2N1左乳IDC(biomarker unclear),行lumpectomy+ALND(nodes+)+AC+T+XRT,未用辅助内分泌。2008年复发:左锁骨上淋巴结转移(ER-/PR-/HER2+ by IHC),纵隔转移。多线治疗:Taxotere/Xeloda/Herceptin→进展→XRT→Tykerb/Herceptin→进展→Capecitabine/Herceptin→Pertuzumab/Herceptin/Taxotere。本次:2nd opinion,已停药1周。PET-CT示可能轻微进展(equivocal),LVEF降至52%。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | NOT_IN_NOTE | ✓ | CC "2nd opinion"=新患者会诊,正确 |
| second opinion | "yes" | clinical trial discussion | ✓ | CC明确"2nd opinion"。归因不精确但值正确 |
| in-person | "in-person" | PET-CT text | ✓ | 有详细体检+生命体征 |
| summary | 准确(1998 Stage II IDC, MBC since 2008, 2nd opinion) | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2+ invasive ductal carcinoma" | - | ✓ | 准确。原发biomarkers unclear,但复发ER/PR neg + HER2+（Herceptin/Tykerb/Pertuzumab证实） |
| Stage_of_Cancer | "Originally Stage II, now metastatic (Stage IV)" | "MBC, recurrent and metastatic" | ✓ | - |
| Metastasis | "Yes (to left supraclavicular node and mediastinum)" | IMP #2,#3 | ✓ | - |
| Distant Metastasis | 同上 | 同上 | ✓ | 左锁骨上在乳癌中可视为distant(N3c但临床视为M1) |
| lab_summary | "No labs in note." | - | P2 | IMP提到tumor marker "[REDACTED] persistently elevated at 14.8"——有明确数值但marker名称被redacted。边界情况 |
| findings | 全面:PET-CT(左乳/胸壁mild progression, SUV 2.1↑), 纵隔稳定, 脑MRI阴性, LVEF 52%↓, marker 14.8稳定 | IMP/STUDIES | ✓ | 客观全面 |
| current_meds | "" | - | ✓ | "Has been off of rx since last wk"——当前无肿瘤药物 |
| recent_changes | "d/c current rx (pertuzumab/Herceptin/Taxotere)" | REC | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "palliative" | "MBC, recurrent and metastatic" | ✓ | Stage IV since 2008 |
| response_assessment | Probable mild progression, SUV 2.1↑, marker 14.8 stable | CT/SUV text | ✓ | 很好——准确描述了equivocal progression |
| Medication_Plan_chatgpt | 文字描述(非结构化schema) | - | P2 | 有内容但格式不符合expected schema |
| medication_plan | "D/c [REDACTED]/Herceptin/Taxotere. Recommend [REDACTED] as next line." | REC | ✓ | 匹配REC |
| therapy_plan | "Would not consider hormonal therapy at this time." | REC | **P1** | **只说了不做什么,遗漏了实际推荐**。REC"Rec [REDACTED] as next line of Rx"是实际therapy plan,但未被提取到therapy_plan中(只在medication_plan里) |
| radiotherapy_plan | "None" | - | ✓ | REC无radiation计划 |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "Would recheck [REDACTED] prior to above" | REC | ✓ | 匹配——开始新治疗前复查tumor marker |
| genetic_testing_plan | "Would recheck [REDACTED] prior to above" | 同上 | **P1** | **错误分类**。这是tumor marker复查(lab plan),不是genetic/molecular test。应为"None planned." 与lab_plan重复 |
| Referral各项 | 全部"None" | - | ✓ | 2nd opinion consult,无outgoing referrals |
| follow up | "recheck [REDACTED] prior to above" | REC | P2 | 这是lab计划不是follow-up时间。实际follow-up未指定(2nd opinion,可能回原医生处) |
| Next clinic visit | "Not specified in the given text" | - | ✓ | 2nd opinion,follow-up安排未明确 |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**:
- Herceptin (trastuzumab), Tykerb (lapatinib), Pertuzumab — HER2靶向药,应在oncology_drugs.txt中
- Taxotere (docetaxel), Xeloda (capecitabine) — 化疗药,应在oncology_drugs.txt中

**本行总结**: 0×P0, 2×P1, 3×P2
- P1: (1) therapy_plan只含否定语句,遗漏了实际推荐的下线治疗 (2) genetic_testing_plan误含lab计划(tumor marker复查)
- P2: (1) lab_summary遗漏redacted tumor marker 14.8 (2) Medication_Plan_chatgpt格式不合规 (3) follow up含lab内容

---

### Row 7 (coral_idx=147)

**患者概况**: 29F(绝经前), 2018.8左乳IDC grade 3, ER-/PR-/HER2+(IHC3+), 临床Stage II-III(6.2cm mass+多发腋LN+)。TCHP新辅助化疗3个不完整周期(因依从性差)。手术:左lumpectomy+ALND→乳房无残留(pT0),28个淋巴结中3个阳性(最大2.4cm+ECE)。PET-CT×3均无远处转移。本次10/13/2019:会诊+建立care,讨论辅助AC×4→T-DM1+放疗。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | port placement text | ✓ | "presents in consultation to establish care" |
| second opinion | "no" | "at great risk..." | ✓ | 是转移care不是second opinion |
| in-person | "Televisit" | "80 minutes face-to-face" | **P1** | **应为"in-person"**。有详细体检(乳房检查、淋巴结触诊)和生命体征,且归因自己引用了"face-to-face" |
| summary | 准确详细(Stage III HER2+ IDC, incomplete neoadjuvant, post-lumpectomy/ALND) | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) invasive ductal carcinoma" | A/P | ✓ | 准确(FISH 5.7应来自中间截断部分的病理详情) |
| Stage_of_Cancer | "Originally clinical stage II-III, now Stage III (pT0N3M0)" | "clinical stage III" | **P1** | **N分期错误**。3/28 LN positive = pN1a(1-3 nodes),不是N3(需≥10个阳性或锁骨上/内乳LN)。应为ypT0N1aM0。A/P给出的"clinical stage III"是新辅助前的分期 |
| Metastasis | "No" | 3/28 LN text | ✓ | PET-CT×3均无远处转移 |
| Distant Metastasis | "No" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 笔记很长但无明确lab values报告(可能在truncated部分) |
| findings | 非常全面:手术病理(pT0, 3/28 LN+, ECE), PET-CT无转移, 乳房US变化, 体检 | A/P + 各影像 | ✓ | 优秀 |
| current_meds | "" | - | ✓ | 当前无肿瘤药物,手术后等待辅助治疗 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "oxyCODONE" | - | P2 | 可能在active med list中用于术后/癌症疼痛,但未在可见文本中验证具体context |
| goals_of_treatment | "curative" | "reduce risk by proceeding with recommended systemic therapy" | ✓ | Stage III无远处转移,辅助治疗意图curative |
| response_assessment | "Not yet on treatment — no response to assess." | 同上 | P2 | 患者HAD neoadjuvant TCHP→手术示near pCR(pT0, 3/28 LN+)。可记录新辅助后的病理反应。但当前语境是讨论辅助治疗,尚未开始 |
| Medication_Plan_chatgpt | chemo: AC×4→T-DM1(含副作用) + radiotherapy | A/P recommendations | ✓ | 非常好!有结构化副作用信息 |
| medication_plan | "adjuvant AC x 4 cycles, to be followed by T-DM1" | A/P | ✓ | - |
| therapy_plan | "adjuvant AC x 4 cycles → T-DM1 and radiation" | A/P | ✓ | - |
| radiotherapy_plan | "radiation" | A/P radiation text | ✓/P2 | 正确但笼统。A/P说"importance of radiation after completing AC" |
| procedure_plan | "adjuvant AC x 4 cycles, to be followed by T-DM1, plan for port placement" | A/P | **P1** | **混入了药物计划**。AC×4和T-DM1是medications不是procedures。应仅含"port placement"。prompt明确:"Do NOT include Medication/treatment plans" |
| imaging_plan | "No imaging planned." | - | **P1** | **遗漏echocardiogram**。A/P"steps that would need to be taken...including port placement and echocardiogram"。Echo/TTE属于imaging plan |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | Myriad已做(negative) |
| Referral各项 | 全部"None" | - | ✓ | - |
| follow up | "We will aim to speak again this coming week." | A/P | ✓ | - |
| Next clinic visit | "in-person: this coming week" | 同上 | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**:
- T-DM1 (ado-trastuzumab emtansine) — HER2靶向ADC,应在oncology_drugs.txt中
- AC (doxorubicin/cyclophosphamide) — 化疗,应在oncology_drugs.txt中
- TCHP (docetaxel/carboplatin/trastuzumab/pertuzumab) — 各成分应在oncology_drugs.txt中

**本行总结**: 0×P0, 4×P1, 3×P2
- P1: (1) in-person应为"in-person"非"Televisit" (2) Stage中N3错误,应为N1(3/28 LN+) (3) procedure_plan混入AC/T-DM1药物计划 (4) imaging_plan遗漏echocardiogram
- P2: (1) supportive_meds中oxycodone未验证context (2) response_assessment可记录neoadjuvant near-pCR (3) radiotherapy_plan笼统

---

### Row 8 (coral_idx=148)

**患者概况**: 63F, 2020.11右乳ER+癌。新辅助AC×4+weekly taxol×12→2021.6.11双乳切: 右乳3.84cm残留IDC(有treatment effect), 1 LN+微转移(0.21cm+ECE)+另1个微转移(0.025cm)+ITC。ER+(85%)/PR-(<1%)/HER2-(IHC0, FISH neg)。Ki-67 1-2%。肾移植受者(1990)。本次video follow-up讨论病理、放疗、letrozole。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "S/p 4 cycles..." | ✓ | 已在此接受新辅助化疗 |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | Zoom telehealth text | ✓ | "Video Visit" + Zoom |
| summary | 准确(post-surgery, discuss pathology, hormone blockade, radiation) | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- invasive ductal carcinoma" | A/P#1 | ✓ | 准确!正确识别PR-(不常见)和HER2-(IHC 0) |
| Stage_of_Cancer | "Stage II" | A/P#1 | ✓ | 匹配A/P所述 |
| Metastasis | "No" | 3.84cm tumor text | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 无lab results |
| findings | 非常详细:手术病理(3.84cm IDC, margins, LN details, receptor status, treatment effect) | A/P | ✓ | 优秀——完整的术后病理报告 |
| current_meds | "" | - | ✓ | 化疗已完成,letrozole尚未开始 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "ondansetron, compazine, olanzapine, miralax" | med list | P2 | 化疗已完成数月,这些止吐/通便药可能已不活跃使用。olanzapine(抗精神病药)在完成化疗后作为supportive med存疑 |
| goals_of_treatment | "curative" | 多段引用 | ✓ | Stage II + neoadjuvant + surgery + planned adjuvant = curative |
| response_assessment | "Not yet on treatment — no response to assess." | 同上 | **P1** | **错误**。患者已完成新辅助化疗(AC×4+taxol×12)。术后病理示治疗反应(treatment effect present in breast and LN)。应描述pathologic response:"Partial response to neoadjuvant chemo: residual 3.84cm IDC with treatment effect" |
| Medication_Plan_chatgpt | hormonal therapy: letrozole + bone therapy: fosamax | A/P | ✓ | 有结构化内容 |
| medication_plan | "Letrozole after radiation. Fosamax for bone protection." | A/P#5,#7 | ✓ | - |
| therapy_plan | "Radiation referral. Letrozole after radiation." | A/P#6,#7 | ✓ | - |
| radiotherapy_plan | "Radiation referral." | A/P#6 | ✓ | - |
| procedure_plan | "No procedures planned." | - | P2 | A/P提到"drains out on Thursday"——drain removal是minor procedure。不过通常不视为oncologic procedure |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral-Specialty | "Radiation referral" | A/P#6 | ✓ | - |
| Referral其他 | 全部"None" | - | ✓ | - |
| follow up | "None" | - | ✓ | A/P未指定follow-up时间 |
| Next clinic visit | "Not specified in the given note" | - | ✓ | - |
| Advance care | "full code." | - | ✓ | - |

**白名单笔记**:
- fosamax (alendronate) — 双膦酸盐,用于骨保护。是supportive还是oncology_drugs? 属于bone therapy,可能在oncology_drugs.txt中
- olanzapine — 抗精神病药(有时用作化疗止吐),完成化疗后不应在supportive_meds中

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: response_assessment错误——患者已完成新辅助化疗,不是"Not yet on treatment"
- P2: (1) supportive_meds可能含过期化疗期药物 (2) procedure_plan未含drain removal

---

### Row 9 (coral_idx=149)

**患者概况**: 66F, 左乳中心部癌, HR+/HER2-。2021.4开始新辅助letrozole→2021.7.24左乳切(8.8cm残留肿瘤, LN受累——"July 20 lymph nodes involved"表述不清)→2021.8.7 re-excision for margins。基因组检测低风险,无需化疗。计划放疗(左胸壁+周围LN),继续letrozole。Video visit(实际因连接失败转为电话)。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | radiation text | ✓ | "VIDEO VISIT FUP" |
| second opinion | "no" | failed video text | ✓ | - |
| in-person | "Televisit" | telehealth text | ✓ | 实际是电话(video连接失败) |
| summary | 准确(post-surgery, radiation planned, follow-up) | A/P | ✓ | - |
| Type_of_Cancer | "HR+ and HER2- invasive carcinoma" | A/P | ✓/P2 | 匹配A/P"HR + and her 2 negative"。P2: prompt要求分别写ER/PR,但原文只写"HR+" |
| Stage_of_Cancer | "Stage II" | A/P | ✓ | - |
| Metastasis | "No" | A/P | ✓ | - |
| Distant Metastasis | "No" | A/P | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 手术病理(8.8cm + LN involved + re-excision) + 体检 | A/P | ✓ | - |
| current_meds | "letrozole" | A/P#4 | ✓ | 自2021.4开始,继续使用 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "letrozole" | A/P#4 | **P1** | **同Row 3问题**。letrozole是oncologic药物,prompt明确禁止放入supportive_meds |
| goals_of_treatment | "curative" | radiation text | ✓ | Stage II + surgery + adjuvant = curative |
| response_assessment | "Not mentioned in note." | - | ✓ | 新辅助letrozole后手术,但note未明确评估neoadjuvant response。当前辅助阶段无response data |
| Medication_Plan_chatgpt | hormonal therapy: letrozole + radiotherapy: chest wall+LN | A/P | ✓ | 有结构化内容 |
| medication_plan | "Continue letrozole started April 2021." | A/P#4 | ✓ | - |
| therapy_plan | letrozole + radiation to left chest wall+LN | A/P | ✓ | - |
| radiotherapy_plan | "Radiation to left chest wall and surrounding LN" | A/P#5 | ✓ | 具体——包含胸壁+周围LN |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "DEXA." | A/P#6 | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral-Specialty | "" (空字符串) | - | P2 | 格式不一致——应为"None"而非空字符串。放疗referral已完成(simulation下周) |
| follow up | "RTC in the [REDACTED]" | A/P#7 | ✓ | - |
| Next clinic visit | "RTC in the [REDACTED]" | 同上 | ✓ | interim history提到"follow up twice a year" |
| Advance care | "full code." | - | ✓ | - |

**白名单笔记**:
- letrozole重复出现在supportive_meds中(Row 3和Row 9)——系统性问题,需在pipeline层面阻止oncologic药物进入supportive_meds

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: supportive_meds含letrozole(oncologic药物,与Row 3同类)
- P2: (1) Type_of_Cancer用"HR+"未分开ER/PR (2) Referral-Specialty空字符串vs"None"

---

## 前10行(Row 0-9)模式回顾

### 累计统计
| 级别 | 计数 | 每行平均 |
|------|------|---------|
| P0 | 2 | 0.2 |
| P1 | 21 | 2.1 |
| P2 | 36 | 3.6 |

### 高频系统性问题

1. **Specialty referral文本泄漏** (P0) — Row 0, 4: LLM从Diagnosis section泄漏imaging order文本到referral字段 [2/10]
2. **supportive_meds含oncologic药物** (P1) — Row 3, 9: letrozole出现在supportive_meds中。prompt已明确禁止 [2/10]
3. **Medication_Plan_chatgpt为空或格式错** (P1) — Row 1, 3, 4, 6: chatgpt schema太复杂,模型经常失败 [4/10]
4. **Type_of_Cancer缺HER2状态** (P1) — Row 0, 5: 即使原文有HER2信息也遗漏 [2/10]
5. **response_assessment判断错误** (P1) — Row 0(未治疗的说成有response), 1(用旧影像), 7(完成neoadjuvant说Not yet), 8(同7) [4/10]
6. **归因引用不精确/无关** (P2) — 几乎每行都有: Patient type、second opinion等字段的归因经常引用不相关文本 [普遍]
7. **genetic_testing_plan误含非genetic内容** (P1) — Row 6: tumor marker复查被放入genetic_testing_plan [1/10]
8. **procedure_plan混入medication内容** (P1) — Row 7: AC/T-DM1化疗方案放入procedure_plan [1/10]

### 可改进方向（仅记录不修改）
1. Pipeline: 阻止oncologic drugs list中的药物出现在supportive_meds
2. Pipeline: Medication_Plan_chatgpt schema简化
3. Prompt: response_assessment需更好地处理post-neoadjuvant场景
4. Prompt: genetic_testing_plan需更强的排除规则(排除lab tests)
5. Pipeline: Referral提取需防止Diagnosis section文本泄漏

### Row 10 (coral_idx=150)

**患者概况**: 68F, 2010年Stage IIIC左乳IDC行改良根治术+ALND→Taxotere/Cytoxan×4→XRT→Femara(未服)。2011.11骨转移+压缩骨折→T6固定→XRT to T-spine/右股骨→后转为Letrozole。2012.10 PET示左下颌骨mass增大→开始Faslodex+Denosumab→下颌放疗10天。现在:下颌麻木改善但右腿疼痛/麻木恶化。

**字段审查**:

| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | boilerplate text | ✓ | - |
| second opinion | "no" | boilerplate | ✓ | - |
| in-person | "in-person" | boilerplate | ✓ | 有体检和生命体征 |
| summary | 准确(MBC on Faslodex, jaw pain/numbness, leg pain) | HPI | ✓ | - |
| Type_of_Cancer | "infiltrating ductal Carcinoma" | - | **P1** | **缺receptor status**。原文receptor被redacted,但治疗(Femara/Faslodex=ER targeted)明确推断HR+。prompt要求包含receptor status when available。应为"HR+(inferred from Faslodex/Femara) infiltrating ductal carcinoma" |
| Stage_of_Cancer | "Originally Stage III C, now metastatic (Stage IV)" | A/P | ✓ | - |
| Metastasis | "Yes, to bone" | "metastasized to multiple sites" | ✓/P2 | 正确但缺具体位点。骨转移位点:T-spine, 右股骨, 左下颌骨 |
| Distant Metastasis | 同上 | 同上 | P2 | 同上 |
| lab_summary | 全面CBC+CMP值 | - | ✓ | 列出了所有有值的lab数据 |
| findings | 骨转移 + PET/CT mandibular mass + 体检 + thrush | A/P | ✓/P2 | P2: 混入了"MRI ordered"(future order)作为finding |
| current_meds | "Faslodex, Denosumab" | "Continue on Faslodex and Denosumab" | ✓ | Faslodex=fulvestrant(HR+ targeted), Denosumab=骨转移保护 |
| recent_changes | "" | - | ✓ | 继续同方案 |
| supportive_meds | "hydrocodone-acetaminophen" | - | ✓ | 骨转移疼痛管理 |
| goals_of_treatment | "palliative" | "metastasized to multiple sites" | ✓ | Stage IV bone mets |
| response_assessment | "PET/CT showed increased metastatic activity...progressing despite current treatment with Faslodex" | restaging text | **P1** | **错误**。引用的PET(2012.10)是Faslodex开始**前**的影像,是换药的原因而非Faslodex的response。A/P实际说"Exam stable",但右腿症状恶化需restaging。正确应为"Exam stable on Faslodex/Denosumab, worsening leg symptoms prompt restaging" |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | **遗漏**。A/P有continue Faslodex/Denosumab + Mycelex for thrush |
| medication_plan | Continue Faslodex/Denosumab + Mycelex for thrush | A/P | ✓ | 全面 |
| therapy_plan | "Continue on Faslodex and Denosumab." | A/P | ✓ | - |
| radiotherapy_plan | null | - | P2 | 值正确(过去的XRT不应包含)但应为"None"字符串而非null |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "will order PETCT to evaluate Femur and to toes" | A/P | **P1** | **遗漏MRI**。A/P也order了"MRI of lumbar, pelvis and right femur"——应包含在imaging_plan中 |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral各项 | 全部"None" | - | ✓ | - |
| follow up | "None" | - | ✓ | A/P未指定 |
| Next clinic visit | "Not specified" | - | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**:
- Faslodex (fulvestrant) — ER+靶向药,应在oncology_drugs.txt中
- Denosumab (Xgeva) — 骨转移保护,应在oncology_drugs.txt中
- hydrocodone-acetaminophen — 阿片类止痛,应在supportive_care_drugs.txt中
- Mycelex (clotrimazole) — 抗真菌,不是supportive care药物(用于thrush)

**本行总结**: 0×P0, 4×P1, 4×P2
- P1: (1) Type_of_Cancer缺receptor status (2) response_assessment用了Faslodex前的PET误判progression (3) Medication_Plan_chatgpt空 (4) imaging_plan遗漏MRI
- P2: (1) Metastasis缺具体骨位点 (2) Distant Metastasis同 (3) findings混入future order (4) radiotherapy_plan为null而非"None"

---

### Row 11 (coral_idx=151)

**患者概况**: 51F，de novo ER+/PR+/HER2+ 乳腺癌，转移至脑、肺、淋巴结、骨。Herceptin+[REDACTED]+letrozole治疗中。多次GK放疗脑转移。最新MRI显示2个新脑病灶，体部CT稳定。DNR/DNI。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "F/u in 6 weeks" | ✓ | - |
| second opinion | "no" | "cont off chemotherapy..." | ✓ | P2: 归因不相关 |
| in-person | "in-person" | "I have reviewed and updated..." | ✓ | - |
| summary | 描述准确,包含转移部位和visit目的 | HPI首句 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2+ invasive ductal carcinoma" | "*****+/PR+/*****+ breast cancer" | **P2** | 原文说"infiltrating breast cancer"非"ductal"——"ductal"是合理推断但非原文所述 |
| Stage_of_Cancer | "Originally not specified, now metastatic (Stage IV)" | 转移描述 | ✓ | - |
| Metastasis | "Yes (to brain, lung, nodes, bone)" | "breast cancer to *****, lung, nodes, brain and bone" | ✓ | - |
| Distant Metastasis | "Yes (to brain, lung, nodes, bone)" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 原文无labs，仅有历史EF值 |
| findings | 含MRI/CT/PE结果，但日期混淆 | "MRI brain on 08/15/18..." | **P1** | **"MRI brain on 09/05/18 showed no pleural effusion"错误——09/05/18是CT CAP不是MRI brain**。混淆了CT和MRI报告结果 |
| current_meds | "herceptin, letrozole" | A/P "cont herceptin... cont letrozole qd" | ✓ | [REDACTED]骨治疗药缺失可接受 |
| supportive_meds | "morphine, oxycodone" | 历史记录提及 | **P1** | **患者已停用！** A/P明确写"able to tolerate without analgesics"、"Was receiving constant massage but now not needed"——morphine/oxycodone是PAST用药 |
| recent_changes | "" (空) | - | ✓ | 骨治疗频率变化被redact可接受 |
| goals_of_treatment | "palliative" | "off chemotherapy for now, due to intolerance" | ✓ | P2: 归因间接,更好的引用是DNR/DNI或"goal to spend time with family" |
| response_assessment | 含SD评估但混入错误日期和过多细节 | "CT shows only multiple bone sites, ? Active..." | **P1** | 同findings——"MRI brain on 09/05/18"日期错误。且混入大量physical exam负性发现,冗余 |
| Medication_Plan_chatgpt | hormonal(letrozole)+bone([REDACTED]) | A/P | **P1** | **遗漏herceptin**——最重要的HER2靶向药未列入chemotherapy类别 |
| medication_plan | 完整: herceptin+letrozole+bone+off chemo | A/P plan list | ✓ | - |
| therapy_plan | 同medication_plan | 同上 | ✓ | - |
| radiotherapy_plan | "await GK / Rad Onc input, and potential plan for repeat GK" | A/P | ✓ | - |
| Procedure_Plan | "await GK / Rad Onc input, and potential plan for repeat GK" | A/P | **P1** | **GK(Gamma Knife)是立体定向放射治疗=放疗**,不是手术。应为"No procedures planned." |
| Imaging_Plan | "CT CAP q4mo, bone scan next eval, MRI brain q4mo" | A/P imaging list | **P1** | **遗漏Echo q6 months**——A/P明确写"Echo q6 months, repeat again in April 2019"。Echo属于imaging per prompt规则 |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | - |
| Nutrition | "None" | - | ✓ | - |
| Genetics | "None" | - | ✓ | - |
| Specialty | "Rad Onc consult" | "await GK / Rad Onc input" | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "f/u Dr.[REDACTED], f/u in 6 weeks" | A/P | ✓ | - |
| Next clinic visit | "in-person: 6 weeks" | "F/u in 6 weeks" | ✓ | - |
| Advance care | "Not discussed during this visit." | - | **P1** | **DNR/DNI+POLST明确记录在Problem List中**："DNR/DNI"、"completed POLST"、详细advance care planning讨论。Pipeline只喂A/P给此prompt是已知限制,但结果仍是错误的 |

**白名单笔记**:
- GK (Gamma Knife) — 应加入radiotherapy关键词,防止误分类为procedure
- Echo/echocardiogram — 确认在imaging_plan白名单中

**本行总结**: 0×P0, 7×P1, 3×P2
- P1: (1) findings日期/模态混淆CT↔MRI (2) supportive_meds含已停用的morphine/oxycodone (3) response_assessment同样日期错误+冗余 (4) Medication_Plan_chatgpt遗漏herceptin (5) Procedure_Plan把GK放疗当手术 (6) Imaging_Plan遗漏Echo (7) Advance_care遗漏DNR/DNI(pipeline限制)
- P2: (1) Type_of_Cancer加了未明确的"ductal" (2) second_opinion归因不相关 (3) goals归因间接

---

### Row 12 (coral_idx=152)

**患者概况**: 41F，新诊断左侧ER+ nuclear G2 DCIS，已行左侧部分乳房切除术，来讨论全身治疗选择（tamoxifen）。Invitae基因panel阴性，DCIS score 60。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI首句 | ✓ | 首次med onc会诊讨论systemic therapy |
| second opinion | "no" | "Patient will think about..." | ✓ | P2: 归因不相关 |
| in-person | "in-person" | "Total face to face time: 60 min" | ✓ | - |
| summary | 准确描述DCIS+手术+visit目的 | HPI | ✓ | - |
| Type_of_Cancer | "ER+ DCIS" | "left nuclear grade 2 ER+ DCIS" | ✓ | DCIS通常只测ER不测PR/HER2,不需列齐三项 |
| Stage_of_Cancer | "Not mentioned in note" | - | **P2** | DCIS按定义=Stage 0 (Tis),但原文确实未明确写stage |
| Metastasis | "No" | "should not result in distant recurrence due to lack of invasion" | ✓ | - |
| Distant Metastasis | "No" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | LABS:段为空 |
| findings | 含手术病理(18mm DCIS, G2, margins clear)+体检(无淋巴结肿大) | 手术病理+PE | ✓ | - |
| current_meds | "" (空) | - | ✓ | 仅ibuprofen/multivitamin/turmeric=非肿瘤药,正确排除 |
| supportive_meds | "" (空) | - | ✓ | - |
| recent_changes | "" (空) | - | ✓ | - |
| goals_of_treatment | "risk reduction" | "reduction in recurrence risk with 5 years..." | ✓ | DCIS+tamoxifen=risk reduction,准确 |
| response_assessment | "Not yet on treatment — no response to assess." | - | ✓ | 尚未开始任何全身治疗 |
| Medication_Plan_chatgpt | hormonal therapy: tamoxifen(讨论中) | "systemic therapy with tamoxifen" | ✓ | 未空,有内容 |
| medication_plan | "patient will consider tamoxifen for 5 years" | A/P | ✓ | - |
| therapy_plan | tamoxifen讨论+rad onc考虑 | A/P | ✓ | - |
| radiotherapy_plan | "encouraged to meet radiation oncology for adjuvant radiation" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | 已完成的Invitae/DCIS score不是future plan |
| Specialty | "Radiation oncology consult" | "meet with radiation oncology" | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "None" | - | ✓ | 原文未指定 |
| Next clinic visit | "Not specified" | - | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2
- P2: (1) Stage_of_Cancer可写Stage 0但原文未明确 (2) second_opinion归因不相关

---

### Row 13 (coral_idx=153)

**患者概况**: 58F，de novo转移性ER+乳腺癌(骨/肝/淋巴结)。原用faslodex+palbociclib已于1月停药。患者自行赴墨西哥接受替代治疗——在家自行给药低剂量化疗(doxorubicin 10mg+gemcitabine 200mg+docetaxel 20mg每周)+pamidronate+代谢疗法+免疫疫苗。既往脊柱手术+放疗(T1-T10仅完成6/10次)。取消了2月CT，计划5月扫描。共同就诊（两位医生）。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "follow up" | "She is here today for follow-up." | ✓ | - |
| second opinion | "no" | 墨西哥治疗相关文字 | ✓ | P2: 归因不相关 |
| in-person | "in-person" | "follow-up" | ✓ | - |
| summary | "58F...currently on faslodex and palbociclib" | HPI首句 | **P1** | **患者1月已停用faslodex/palbociclib！** summary引用了过时的HPI模板。IMP明确写"who was on palbociclib and fulvestrant"(过去式) |
| Type_of_Cancer | "ER+ invasive ductal carcinoma" | "metastatic ER+ breast cancer" | **P1** | (1)"ductal"原文未提及 (2)**缺HER2状态**——palbociclib+fulvestrant方案=ER+/HER2-标准治疗,应推断HER2- |
| Stage_of_Cancer | "Stage IV" | 转移描述 | ✓ | - |
| Metastasis/Distant | "Yes (bone, liver, nodes)" | HPI | ✓ | - |
| lab_summary | 详细列出2月24日全套labs(CMP+CBC) | "Labs from Feb 24 look okay" | ✓ | 内容准确详实 |
| findings | 混合了治疗方案描述+症状+体检 | 当前治疗方案文字 | **P2** | findings应聚焦体检和检查发现,不应重复治疗方案描述 |
| current_meds | "Pamidronate, Gemcitabine, Docetaxel, Doxorubicin" | IMP段确认墨西哥方案 | ✓ | 正确识别了[REDACTED]=doxorubicin(从IMP段) |
| recent_changes | 停palbociclib/fulvestrant+新墨西哥方案,详细 | HPI | ✓ | - |
| supportive_meds | "pamidronate once weekly" | 原文 | **P1** | **pamidronate是骨转移治疗药(bisphosphonate)=oncologic,不是supportive med**。实际supportive meds应为CBD/magnesium/topical cannabis(但这些是supplements) |
| goals_of_treatment | "palliative" | "metastatic ER+ breast cancer" | ✓ | - |
| response_assessment | "currently responding...no significant new imaging findings" | 间接引用 | **P1** | **无近期影像！** 患者2月取消CT,"She wants to schedule scans for May"。不能说"responding"——应为"无法评估,无近期影像。临床稳定,labs okay" |
| Medication_Plan_chatgpt | chemotherapy: gemcitabine等 | A/P | ✓ | - |
| medication_plan | 包含topical cannabis/sulfur+Cymbalta Rx+继续墨西哥方案 | A/P | ✓ | - |
| therapy_plan | 墨西哥低剂量化疗+pamidronate | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | 既往放疗正确排除 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "CT CAP+Total Spine MRI for May, repeat spine MRI in 6 weeks" | A/P | ✓ | 两个来源都捕获了 |
| Lab_Plan | "Labs every two weeks" | IMP段 | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | - |
| Referral各项 | 全部"None" | - | **P2** | A/P提到"***** start PT on 03/12/19"可能是PT referral,但是否新referral不确定 |
| follow up | "RTC after scans, 2 months" | IMP段 | ✓ | - |
| Next clinic visit | "2 months" | "return in 2 months" | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**:
- Pamidronate — 应在oncology_drugs.txt(骨转移治疗),不应出现在supportive_meds
- Doxorubicin (adriamycin) — 确认在oncology_drugs.txt中
- CBD/cannabis — 非处方supplements,不列入任何药物白名单

**本行总结**: 0×P0, 4×P1, 3×P2
- P1: (1) summary引用过时HPI称"currently on faslodex/palbociclib"实际已停 (2) Type_of_Cancer缺HER2-且添加了未确认的"ductal" (3) supportive_meds把pamidronate(oncologic)当supportive (4) response_assessment在无影像证据下称"responding"
- P2: (1) findings混入治疗方案描述 (2) second_opinion归因不相关 (3) PT referral可能遗漏

---

### Row 14 (coral_idx=154)

**患者概况**: 46F，新诊断左乳癌，混合浸润性导管癌+小叶癌。ER+(>95%), PR+(80-90%), HER2 IHC 2+(equivocal), FISH+(ratio 2.0)。临床Stage I/II。来求second opinion。医生按ASCO/CAP判为HER2+,建议TCHP方案(若选neoadjuvant)或先手术。同日见乳外科。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | - |
| second opinion | "yes" | "reviewed pathology, biomarker testing..." | ✓ | 原文"2nd opinion"明确 |
| in-person | "in-person" | "50 min, >50% counseling" | ✓ | - |
| summary | 准确描述混合型癌+second opinion | IMP段 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2+ mixed infiltrating ductal and lobular carcinoma" | 病理报告+IMP段 | ✓ | 完美——三项receptor status齐全,HER2根据FISH ratio 2.0正确判+,组织类型准确 |
| Stage_of_Cancer | "Clin st I/II, based on imaging and exam" | IMP段原文 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | MRI tumor特征(12x10x13mm,periareolar) | MRI报告 | ✓ | - |
| current_meds | "" (空) | - | ✓ | MEDS: None |
| supportive_meds | "" (空) | - | ✓ | - |
| recent_changes | "" (空) | - | ✓ | 新患者无治疗史 |
| goals_of_treatment | "curative" | "young, fit, large potential benefit" | ✓ | Stage I/II,明确curative intent |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | TCHP方案(conditional) | "favor TCHP regimen" | ✓ | P2: schema格式略异(dict而非array) |
| medication_plan | "If opts for [REDACTED] Rx, TCHP" | IMP/REC | ✓ | - |
| therapy_plan | 同上 | 同上 | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "breast surg" | "proceed to breast surg" | ✓ | 条件性计划,合理 |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | HER2 FISH已完成不是future plan |
| Specialty | "Breast surgery/Dr.[REDACTED]" | "explore w breast surg" | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "Dr.[REDACTED] later today" | IMP | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 1×P2
- P2: (1) Medication_Plan_chatgpt schema格式用dict而非array

---

### Row 15 (coral_idx=155)

**患者概况**: 54F，绝经后，新诊断右乳Stage I ER+/PR+/HER2- IDC(0.3cm G1)，已行lumpectomy，来建立care。Zoom远程就诊。计划adjuvant radiation+AI×5年。需DEXA、检查estradiol、genetic testing referral(父有结肠癌)。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | "presenting to establish care" |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | "ZOOM telehealth" | ✓ | 正确识别远程就诊 |
| summary | 准确,包含Stage I/HR+/HER2-/IDC | A/P首句 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | 病理+A/P | ✓ | 完美——三项receptor齐全 |
| Stage_of_Cancer | "Stage I" | A/P明确写 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 手术病理(0.3cm G1 IDC)+影像特征 | 病理报告 | ✓ | - |
| current_meds | "" (空) | - | ✓ | 无oncologic meds |
| supportive_meds | "" (空) | - | ✓ | - |
| goals_of_treatment | "curative" | A/P | ✓ | Stage I adjuvant |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | hormonal therapy: AI×5年,附side effects | A/P | ✓ | 包含hot flashes/arthralgias/vaginal dryness/bone density loss |
| medication_plan | "AI after radiation×5年+calcium+vitD" | A/P | ✓ | - |
| therapy_plan | rad onc+AI×5年 | A/P | ✓ | - |
| radiotherapy_plan | "proceed with radiation oncology evaluation" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | SLN biopsy问题是待讨论非明确计划 |
| Imaging_Plan | "DEXA+consider surveillance breast MRI" | A/P | ✓ | 完美——两项都捕获 |
| Lab_Plan | "check estradiol level" | A/P | ✓ | - |
| Genetic_Testing_Plan | "refer for genetic testing(父结肠癌+乳癌史)" | A/P | ✓ | - |
| Genetics | "refer for genetic testing" | A/P | ✓ | 与Genetic_Testing_Plan一致,合理 |
| Specialty | "Radiation oncology consult" | A/P | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | 列出所有action items | A/P | ✓ | - |
| Advance care | "Not discussed during this visit." | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 0×P2
- 非常干净的提取,所有字段准确

---

### Row 16 (coral_idx=156)

**患者概况**: 53F，新诊断左乳IDC(0.8cm G2)，ER+(>95%)/PR+(>95%)/HER2-(IHC 0,FISH 1.1)/Ki67 5%。已行lumpectomy+SLNB(0/5)。Video visit。绝经状态不确定(s/p子宫切除)。家族史:姐妹卵巢癌+姑母乳癌→genetic referral。需adjuvant RT+endocrine therapy。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | A/P | ✓ | "consultation regarding further management" |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | "video visit" | ✓ | - |
| summary | 准确 | A/P首句 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | 病理报告 | ✓ | 完美 |
| Stage_of_Cancer | "" (空) | - | **P1** | **应推断Stage I**：0.8cm(T1b)+LN 0/5(N0)+no distant mets=Stage I。原文stage被redact但可从病理推断 |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 详细病理+影像结果 | 病理报告 | ✓ | - |
| current_meds | "" (空) | - | ✓ | Meds: none |
| supportive_meds | "" (空) | - | ✓ | - |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空了！** A/P详细讨论了tamoxifen vs AI的side effects(hot flashes, mood, thromboembolic, myalgias, bone loss) |
| medication_plan | "adjuvant hormonal therapy×5年,tamoxifen或AI based on menopausal status" | A/P | ✓ | - |
| therapy_plan | hormone check+radiotherapy+endocrine after RT | A/P | ✓ | - |
| radiotherapy_plan | "requires breast radiotherapy" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | prophylactic mastectomy待genetic结果,非明确计划 |
| Imaging_Plan | "baseline DXA scan" | A/P | ✓ | - |
| Lab_Plan | "check labs including hormones" | A/P | ✓ | - |
| Genetic_Testing_Plan | "Refer to genetics" | A/P | ✓ | - |
| Nutrition | "nutritionist at her request" | A/P | ✓ | 正确捕获patient-initiated referral |
| Genetics | "refer to genetics" | A/P | ✓ | - |
| Specialty | "breast radiotherapy consult" | A/P | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "F/U after RT" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 0×P2
- P1: (1) Stage_of_Cancer空但应推断Stage I (2) Medication_Plan_chatgpt空

---

### Row 17 (coral_idx=157)

**患者概况**: 65F，左乳IDC NOS(0.8cm G1)+encapsulated papillary carcinoma(=DCIS)。ER+/PR+(强阳)/HER2-(1+)/Ki67 5%。Lumpectomy后,3个LN中1个有ITC(isolated tumor cells)。半姐妹45岁乳癌。计划adjuvant endocrine therapy 5-10年+Rad Onc评估+DEXA。拒绝化疗,不做Oncotype。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | IMP段 | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "60 min face to face" | ✓ | - |
| summary | 准确描述IDC+papillary CA | IMP段 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC+encapsulated papillary CA" | 病理报告 | ✓ | 详细准确 |
| Stage_of_Cancer | "pT1b, pNX" | 病理/LN描述 | **P1** | **pNX错误**——3个LN已评估(ITC in 1/3)=pN0(i+),不是"无法评估"。应为pT1b pN0(i+)=Stage IA |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 详细病理+margin+LN信息 | 病理报告 | ✓ | - |
| current_meds | "" (空) | - | ✓ | - |
| supportive_meds | "" (空) | - | ✓ | - |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | "adjuvant endocrine therapy 5-10 yrs" | REC段 | ✓ | P2: dict而非array格式 |
| medication_plan | "adjuvant endocrine therapy 5-10 yrs" | REC段 | ✓ | - |
| therapy_plan | endocrine+Rad Onc | REC段 | ✓ | - |
| radiotherapy_plan | "RTC after Rad Onc eval +/- XRT" | REC段 | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "DEXA ordered" | REC段 | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | **P1** | **遗漏！** IMP写"discussed with UCSF Cancer Risk. They will reach out to pt today"——遗传测试正在安排中 |
| Genetics | "None" | - | **P1** | **遗漏！** 同上——UCSF Cancer Risk reaching out=genetics referral。之前saliva测试失败,正在重新安排 |
| Specialty | "Rad Onc eval" | REC | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "RTC after Rad Onc eval" | REC | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 3×P1, 1×P2
- P1: (1) Stage pNX错误——已评估3个LN应为pN0(i+) (2) Genetic_Testing_Plan遗漏UCSF Cancer Risk安排 (3) Genetics referral遗漏
- P2: (1) Medication_Plan_chatgpt schema用dict而非array

---

### Row 18 (coral_idx=158)

**患者概况**: 70F，新诊断左乳IDC G3,ER 90%/PR 90%/HER2 3+(FISH+,ratio 9.5)/Ki67 20-90%。肿瘤2.1cm(US)/6.6cm(MRI extent),左腋窝LN FNA阳性。PET无远处转移。PMH: CAD/DM/HTN/HLD。建议neoadjuvant TCHP(避免蒽环因CAD),讨论clinical trial。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | IMP/RECS | ✓ | med onc consultation |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "60 min face-to-face" | ✓ | - |
| summary | 准确 | IMP段 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2+(FISH+) IDC" | 病理 | ✓ | 完美——HER2 3+ FISH+正确 |
| Stage_of_Cancer | "Not mentioned in note" | - | **P2** | 可推断Stage IIB-IIIA(T2-3 N1 M0),但原文确实未写"Stage" |
| Metastasis | "No" | PET结果 | ✓ | 腋窝LN=regional,非distant |
| Distant Metastasis | "No" | PET | ✓ | PET confirms no distant mets |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 影像+病理结果 | imaging+path reports | ✓ | 详细 |
| current_meds | "" (空) | - | ✓ | 仅非肿瘤药(atorvastatin等),正确排除 |
| supportive_meds | "" (空) | - | ✓ | - |
| goals_of_treatment | "curative" | RECS | ✓ | neoadjuvant curative |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空!** RECS详细写了TCHP+GCSF,避免蒽环因CAD |
| medication_plan | "TCHP regimen with GCSF" | RECS | ✓ | - |
| therapy_plan | TCHP+avoid anthracycline+port+teaching详细 | RECS | ✓ | 非常好 |
| radiotherapy_plan | "None" | - | ✓ | 此阶段未讨论放疗 |
| Procedure_Plan | "Port Placement" | RECS | ✓ | - |
| Imaging_Plan | "Echocardiogram" | RECS | ✓ | HER2靶向治疗前需要 |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | - |
| Referral各项 | 全"None" | - | ✓ | clinical trial讨论非formal referral |
| follow up | "None" | - | ✓ | 未指定具体时间 |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 1×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) Stage可推断但原文未写

---

### Row 19 (coral_idx=159)

**患者概况**: 75F，2009年左乳IDC(0.9cm G2 ER+/PR+/HER2-)行双侧乳房切除+tamoxifen 5年。2020年转移复发:innumerable骨病灶+右腋窝/纵隔/肺门淋巴结(PET)。右髂骨活检证实乳腺来源(ER+80%/PR+50%/HER2-FISH ratio 1.05)。本次启动letrozole+palbociclib。PMH: HTN/DM/骨质疏松。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | "consultation regarding metastatic recurrence" |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "75 minutes" | ✓ | - |
| summary | 准确 | A/P首句 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC with DCIS" | 病理 | ✓ | P2: "with DCIS"是2009年历史,对当前诊断不太相关 |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | A/P | **P1** | **原始stage应为Stage IA不是IIA**——0.9cm(T1b)+0/2 SLN(N0)=Stage IA。T1N0M0≠IIA |
| Metastasis | "Yes (bone, lymph nodes, lung)" | PET结果 | **P2** | 肺结节non-FDG-avid(无代谢增高),是否为转移不确定 |
| Distant Metastasis | "Yes (bone, lymph nodes, lung)" | 同上 | **P2** | 同上 |
| lab_summary | "POCT glucose 104 (03/01/13)" | 历史lab | **P1** | **2013年的glucose!** 8年前数据,应排除。应为"No recent labs in note." |
| findings | 更像历史回顾而非当前findings | 历史描述 | **P2** | findings应聚焦当前状态,不应重复oncologic history |
| current_meds | "letrozole, palbociclib" | A/P plan | ✓ | "Start Letrozole, Rx given"+"OK to start"=本次visit启动,属current |
| recent_changes | "Start Letrozole+Palbociclib" | Plan section | ✓ | - |
| supportive_meds | "denosumab" | A/P | **P1** | **denosumab(Xgeva)是骨转移治疗药=oncologic bone therapy**,不是supportive med。且尚未启动(需dental clearance) |
| goals_of_treatment | "palliative" | Stage IV | ✓ | - |
| response_assessment | "6mm肺结节+hypermetabolic LN" | PET findings | **P1** | **尚未开始全身治疗！** letrozole/palbociclib刚开始。应为"Not yet on treatment"。所述影像是diagnostic workup非response assessment |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空!** A/P详细讨论了PALOMA trial+palbociclib side effects |
| medication_plan | "Start letrozole+palbociclib,denosumab after dental,monthly labs" | Plan | ✓ | - |
| therapy_plan | 同上+dental clearance detail | Plan | ✓ | - |
| radiotherapy_plan | "Rad Onc consult" | Plan | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "MRI total spine+CT CAP+obtain outside PET/CT" | Plan | ✓ | 全面 |
| Lab_Plan | "labs+tumor markers+monthly blood work" | Plan | ✓ | - |
| Genetic_Testing_Plan | "Foundation One(或[REDACTED] 360)" | Plan | ✓ | 正确识别molecular profiling |
| Specialty | "Rad Onc referral" | Plan | ✓ | - |
| Others | "None" | - | **P2** | Plan提到"dental clearance prior to denosumab"可能是dental referral |
| follow up | "RTC in ~1 month" | Plan | ✓ | - |
| Next clinic visit | "~1 month" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- Denosumab (Xgeva) — bone therapy,应在oncology_drugs.txt,不应出现在supportive_meds
- Palbociclib (Ibrance) — CDK4/6 inhibitor,确认在oncology_drugs.txt

**本行总结**: 0×P0, 5×P1, 4×P2
- P1: (1) Stage原始应为IA非IIA (2) lab_summary含2013年数据 (3) supportive_meds含denosumab(oncologic) (4) response_assessment应为"not yet on treatment" (5) Medication_Plan_chatgpt空
- P2: (1) Type_of_Cancer含历史DCIS (2)(3) 肺结节non-FDG-avid列为转移不确定 (4) dental referral可能遗漏

---

## 前20行模式回顾 (Row 0-19)

### 统计
| 严重度 | Row 0-9 | Row 10-19 | 合计(0-19) |
|--------|---------|-----------|------------|
| P0 | 2 | 0 | **2** |
| P1 | ~25 | ~22 | **~47** |
| P2 | ~40 | ~16 | **~56** |

### 系统性问题排名（按频率）

**1. Medication_Plan_chatgpt 空 (P1) — 10/20行**
- Row 1,2,3,5,8,10,13,16,18,19 均为空{}
- 这是最高频的P1问题。Medication_Plan_chatgpt prompt的schema比较复杂(嵌套JSON array with type/medication/summary/side_effects),模型经常生成空对象
- Medication_Plan和therapy_plan同一信息用更简单schema却能正确提取
- **建议**: 简化Medication_Plan_chatgpt的schema或合并到Medication_Plan

**2. response_assessment 判断错误 (P1) — ~8/20行**
- Row 0,1,2,3,7,8,13,19 均有不同程度的错误
- 常见模式: (a)尚未开始治疗却说"responding/stable" (b)无近期影像却评估response (c)混入无关内容(手术恢复/Oncotype)
- Row 13: 最严重——取消了影像却说"responding"
- Row 19: 新诊断还没治疗就写imaging findings作response
- **根因**: 模型不清楚response_assessment需要"已开始治疗+有评估数据"两个前提

**3. supportive_meds 含 oncologic 骨治疗药 (P1) — 4/20行**
- Row 11(morphine/oxycodone已停), Row 13(pamidronate), Row 19(denosumab)
- Row 5(letrozole in supportive=oncologic)
- Pamidronate和denosumab(Xgeva)是bone-modifying agents=oncologic bone therapy,不应在supportive_meds
- **建议**: 在supportive_meds prompt中明确排除bisphosphonates和RANK-L inhibitors

**4. Type_of_Cancer 缺 HER2 状态 或 添加未确认信息 (P1/P2) — ~6/20行**
- Row 0,1,4,7,10: 缺HER2
- Row 11,13: 添加"ductal"但原文只说"infiltrating breast cancer"
- **改善**: 相比Row 0-9, Row 14-19有明显改善(14,15,16,18都完美)

**5. Stage_of_Cancer 错误或缺失 (P1) — 4/20行**
- Row 16: 空但可推断Stage I
- Row 17: pNX错误(应pN0(i+))
- Row 19: 原始stage IIA应为IA
- Row 10: Stage从未提及但可推断
- **根因**: 模型对AJCC分期规则掌握不够,尤其是从T+N推断stage

**6. Advance_care_planning 遗漏 DNR/DNI (P1) — 1/20行明确遗漏**
- Row 11: DNR/DNI+POLST在Problem List中但pipeline只喂A/P
- **根因**: Pipeline设计——plan_extraction prompts只接收A/P段,但Advance_care_planning prompt说"look in the FULL NOTE"
- **建议**: Advance_care_planning prompt应接收全文而非仅A/P

**7. Specialty referral text leakage (P0) — 2/20行**
- Row 0,4: Specialty字段中出现了非referral的药物信息
- 在Row 10-19中未再出现 → 可能是数据特定问题

**8. findings 混入非clinical内容 (P2) — ~5/20行**
- Row 10,11,13: findings包含治疗方案、历史摘要而非体检/检查发现
- 相对次要但频繁

**9. Imaging_Plan 遗漏 (P1) — 3/20行**
- Row 10: 遗漏MRI
- Row 11: 遗漏Echo
- Row 4: 遗漏部分imaging

**10. Genetic_Testing_Plan / Genetics referral 遗漏 (P1) — 2/20行**
- Row 17: UCSF Cancer Risk安排中但未提取
- 相对不常见

### 新增发现 (Row 10-19)

1. **日期/模态混淆** (P1, Row 11): "MRI brain on 09/05/18"实际是CT CAP。模型在findings和response_assessment中都犯此错误。
2. **已停用药物列为supportive_meds** (P1, Row 11): morphine/oxycodone在A/P明确说"without analgesics"却仍被提取。
3. **Procedure_Plan误含radiotherapy** (P1, Row 11): GK(Gamma Knife)是放射治疗不是手术。
4. **过时HPI引用** (P1, Row 13): summary直接引用HPI模板"currently on faslodex"但患者已停药。
5. **远古lab数据** (P1, Row 19): 2013年的glucose被提取。

### 正面发现 (Row 10-19)

1. Row 12: **0×P0, 0×P1, 2×P2** — DCIS case干净提取
2. Row 14: **0×P0, 0×P1, 1×P2** — HER2 equivocal/FISH+正确判HER2+
3. Row 15: **0×P0, 0×P1, 0×P2** — 完美提取
4. Row 14-16的Type_of_Cancer都完美(ER/PR/HER2齐全)
5. Televisit/video visit被正确识别(Row 15,16)
6. Genetic_Testing_Plan在Row 19正确识别Foundation One

---

### Row 20 (coral_idx=160)

**患者概况**: 70F绝经后,右乳intermediate grade DCIS(ER+98%/PR+90%,spanning 5cm,comedo necrosis,clear margins)。s/p partial mastectomy。强家族史(两姐妹+多亲属乳癌)但Invitae 52基因panel阴性。PMH:骨质减少/血脂异常/糖耐量异常。无处方药。讨论adjuvant XRT+Arimidex。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | breast surgeon转诊 |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | PE记录 | ✓ | - |
| summary | 准确描述DCIS+adjuvant therapy讨论 | Assessment | ✓ | - |
| Type_of_Cancer | "ER+/PR+ intermediate grade DCIS" | 病理 | ✓ | DCIS无需列HER2 |
| Stage_of_Cancer | "Not mentioned in note" | - | **P2** | DCIS=Stage 0(Tis),原文未明确写 |
| Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 手术病理详细(5cm DCIS, mixed pattern, comedo)+PE | 病理报告 | ✓ | - |
| current_meds | "" (空) | - | ✓ | 无处方药 |
| goals_of_treatment | "risk reduction" | A/P | ✓ | DCIS adjuvant |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空!** 详细讨论了Arimidex×5年+bisphosphonate+side effects |
| medication_plan | "Arimidex for 5yr after XRT + bisphosphonate" | A/P | ✓ | - |
| therapy_plan | adjuvant treatment计划 | A/P | ✓ | - |
| radiotherapy_plan | "Rad Onc eval for adjuvant XRT" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | Invitae已完成 |
| Specialty | "Rad Onc consult" | A/P | ✓ | - |
| follow up | "3-4 months" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 1×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) Stage可为Stage 0但原文未写

---

### Row 21 (coral_idx=161)

**患者概况**: 72F，复杂乳腺癌史: (1)左乳DCIS 1994年lumpectomy+RT; (2)右乳Stage II IDC 2000年AC×4+RT+tamoxifen 6年; (3)2020年5月转移复发——右胸壁+骨+锁骨下/IM淋巴结,HR+/HER2-。2020年6月起abemaciclib+letrozole(→anastrozole),PET显示good response。2021年7月abemaciclib因pneumonitis停用,在用steroids。来求second opinion。Code status: Full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | second opinion visit |
| second opinion | "yes" | A/P | ✓ | "She is here for a second opinion" |
| in-person | "in-person" | "70 minutes" | ✓ | - |
| summary | 准确描述复杂历史+当前situation | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | 病理 | **P1** | **缺HER2-**——原文明确写"her 2 negative"和"her 2 neu negative" |
| Stage_of_Cancer | "Originally Stage II, now Stage IV" | A/P | ✓ | 原note明确写Stage II和metastatic |
| Metastasis | "Yes (bones, chest wall, infraclavicular, IM nodes)" | A/P | ✓ | - |
| Distant Metastasis | 同上 | 同上 | ✓ | 胸壁复发+骨转移=Stage IV |
| lab_summary | CBC+CMP from 01/29/2021 | 原文labs段 | **P2** | labs是就诊8个月前(01/29 vs 09/30),较旧 |
| findings | 更像history回顾 | A/P history | **P2** | findings应聚焦当前体检/影像,非oncologic history复述 |
| current_meds | "anastrozole, denosumab" | 药物清单 | ✓ | anastrozole正确(AI), denosumab(Xgeva)作oncologic bone therapy列入current ✓ |
| recent_changes | "abemaciclib held due to pneumonitis; letrozole→anastrozole July **2021**" | HPI | **P1** | **日期错误!** 原文说"July 2020"换药,提取写"July 2021"。差一年 |
| supportive_meds | "prednisone, Lomotil, denosumab(XGEVA)" | 药物清单 | **P1** | **denosumab再次出现在supportive_meds**——它是oncologic bone therapy,不是supportive。Prednisone(pneumonitis)和Lomotil(腹泻) ✓ |
| goals_of_treatment | "palliative" | Stage IV | ✓ | - |
| response_assessment | "PET scans 11/20,04/21 showed good response" | HPI | ✓ | 准确——最近PET确实显示good response |
| Medication_Plan_chatgpt | "hormonal therapy+other treatment" 有内容 | A/P | ✓ | 不空,但内容偏向past changes |
| medication_plan | "continue arimidex if stable, faslodex if progression, afinitor/xeloda options" | A/P | ✓ | 全面 |
| therapy_plan | 同上 | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | past radiation正确排除 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "pet ct now" | A/P #5 | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "if progression, faslodex with [REDACTED] if [REDACTED] mutation" | A/P #6 | **P1** | **这是treatment contingency不是genetic testing!** A/P说"if she has a mutation"=治疗条件,非"order/send genetic test" |
| Referral各项 | 全"None" | - | ✓ | second opinion无outgoing referral |
| follow up | "after PET CT results" | A/P | ✓ | - |
| Advance care | "Full code." | 原文position 5861 | ✓ | 验证存在——"Code status: Full code." |

**白名单笔记**:
- Abemaciclib (Verzenio) — CDK4/6 inhibitor,确认在oncology_drugs.txt
- Denosumab重复出现在supportive_meds（已是第3次: Row 13 pamidronate, Row 19 denosumab, Row 21 denosumab）

**本行总结**: 0×P0, 4×P1, 2×P2
- P1: (1) Type_of_Cancer缺HER2- (2) recent_changes日期错July 2021→应2020 (3) supportive_meds含denosumab(oncologic) (4) genetic_testing_plan是treatment contingency非test order
- P2: (1) lab_summary含8月前旧数据 (2) findings偏history回顾

---

### Row 22 (coral_idx=162)

**患者概况**: 63F，新诊断左乳癌(两灶): Focus 1: 1cm G2, ER+(>95%)/PR-/HER2 1+(FISH-)/Ki67 5%; Focus 2: 0.7cm G2 ductal+lobular features, ER+(>95%)/PR 60%/HER2 2+(FISH-)/Ki67 5-10%。加上DCIS。s/p left mastectomy+SLN(0/1)。Myriad BRCA2 VUS。Plan: letrozole×5+年+DEXA。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "60 min" | ✓ | - |
| summary | 描述ductal+lobular,但提到"HER2 2+" | A/P | **P2** | HER2 2+是IHC分数,FISH阴性=HER2-。Summary应说HER2-而非引用raw IHC |
| Type_of_Cancer | "ER+/PR+/HER2- invasive carcinoma with ductal and lobular features" | A/P | ✓ | P2: PR状态复杂(Focus1 PR-, Focus2 PR 60%),简化为PR+不完全准确 |
| Stage_of_Cancer | "Approximately Stage I-II" | 病理描述 | **P2** | 1cm+0/1LN=Stage IA,不应写"I-II"模糊范围 |
| Metastasis | "No" | - | ✓ | - |
| lab_summary | "POCT glucose 105,185,236,116" | 血糖记录 | **P2** | 是糖尿病监测glucose,非cancer-relevant labs |
| findings | 两灶病理详细描述 | 手术病理 | ✓ | P2:混淆两灶特征 |
| current_meds | "" (空) | - | ✓ | 尚未开始治疗 |
| goals_of_treatment | "curative" | A/P | ✓ | 低风险adjuvant |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空!** 详细讨论letrozole+AI side effects+extension data |
| medication_plan | "letrozole ≥5年,可能延长" | A/P | ✓ | - |
| therapy_plan | 包含switching strategy | A/P | ✓ | 详细 |
| radiotherapy_plan | "None" | - | ✓ | mastectomy后无需RT(无high-risk features) |
| Imaging_Plan | "baseline DEXA" | A/P | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | Myriad已完成 |
| Referral各项 | 全"None" | - | ✓ | - |
| follow up | "3 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 5×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) summary写HER2 2+应为HER2- (2) PR状态混合简化 (3) Stage I-II不精确 (4) lab是diabetes glucose (5) findings混淆两灶

---

### Row 23 (coral_idx=163)

**患者概况**: 56F，右乳Grade II micropapillary mucinous carcinoma,ER+(>95%)/PR+(80%)/HER2 equivocal IHC(2+) FISH阴性/Ki67 5%。s/p partial mastectomy+SLN(2/4有微转移0.4mm)。ADH病史。PET无明确远处转移(肝segment 8 1.9cm可疑但不hypermetabolic)。Plan: MammaPrint检测→conditional adjuvant决策+RT+hormone therapy。PT referral(腋窝)。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | A/P | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "60 min" | ✓ | - |
| summary | 准确含receptor details | HPI/A/P | ✓ | - |
| Type_of_Cancer | 详细含ER/PR/HER2 IHC FISH | 病理 | ✓ | HER2 equivocal IHC→FISH neg=HER2-,虽未简写但准确 |
| Stage_of_Cancer | "Not mentioned in note" | - | **P2** | 可从肿瘤大小+SLN微转推断,但note未写 |
| Metastasis | "No" | PET结果 | ✓ | 肝病灶indeterminate |
| findings | 病理+影像详细 | 手术病理 | ✓ | - |
| current_meds | "" (空) | - | ✓ | - |
| supportive_meds | "acetaminophen-codeine, oxycodone" | 药物清单 | ✓ | 术后止痛药=supportive |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | 有内容! conditional plan | A/P | ✓ | 不空 |
| medication_plan | "if MP low risk → adjuvant hormone therapy" | A/P | ✓ | - |
| therapy_plan | MammaPrint+conditional plans详细 | A/P | ✓ | - |
| radiotherapy_plan | "if low risk → radiation; Rad Onc 12/07/18" | A/P | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Genetic_Testing_Plan | "send specimen for MP(MammaPrint)" | A/P | ✓ | 正确识别molecular profiling |
| Specialty | "Radiation oncology consult" | A/P | ✓ | - |
| Others | "Physical therapy referral" | A/P | ✓ | 正确捕获PT referral(腋窝) |
| follow up | "Rad Onc 12/07/18" | A/P | ✓ | P2: 指向Rad Onc而非med onc f/u |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2
- P2: (1) Stage未写可推断 (2) Next clinic visit指向Rad Onc非med onc

---

### Row 24 (coral_idx=164)

**患者概况**: 45F,转移性乳腺癌。历史:(1)2007年右乳IDC 3.5cm ER+/PR+/HER2(?),s/p lumpectomy+AC×4+Taxotere+RT+tamoxifen; (2)2008年左乳IDC 1.5cm G3 ER+/PR-/HER2-,s/p lumpectomy+CMF+Lupron; (3)转移:骨+肝+脑+淋巴结。目前Xeloda 1500/1000mg 14on/7off+ixabepilone。Labs: ALP 308↑, AST 55↑, Hgb 11.2↓, Alb 3.1↓。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | HPI | ✓ | 在治疗中follow-up |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 有PE | ✓ | - |
| summary | "on Xeloda and Irinotecan" | HPI首句 | **P2** | HPI写"Irinotecan"但A/P写"ixabepilone"——note本身矛盾 |
| Type_of_Cancer | "ER+/PR+/HER2- IDC(original)+ER+/PR-/HER2- IDC(metastatic)" | 病理 | **P2** | 2007年右乳"HER-2/neu"原文未明确positive/negative,HER2-是推断 |
| Stage_of_Cancer | "Originally Stage IIA, now Stage IV" | 问题列表 | **P2** | 原始T2(3.5cm)+SLN+(多节点阳性)可能>Stage IIA |
| Metastasis | "Yes (brain, liver, bones, LN)" | 历史记录 | ✓ | - |
| Distant Metastasis | "Yes (brain, liver, bones, LN)" | 同上 | ✓ | - |
| lab_summary | 详细CBC+CMP,标注异常值 | 当日labs | ✓ | 内容准确 |
| findings | PET 03/12/2011结果+体检 | PET报告 | ✓ | - |
| current_meds | "capecitabine, ixabepilone, denosumab" | 药物清单/A/P | ✓ | capecitabine=Xeloda ✓, ixabepilone ✓ |
| supportive_meds | "ondansetron, docusate, prochlorperazine, oxycodone" | 药物清单 | ✓ | 经典supportive care药物 |
| goals_of_treatment | "palliative" | Stage IV | ✓ | - |
| response_assessment | "PET showed progression; supraclavicular breaking up" | A/P | ✓ | 混合response: PET progression但体检supraclavicular改善 |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | **又空** |
| medication_plan | "start cycle 04/12, pain management plan" | A/P | ✓ | - |
| therapy_plan | "Xeloda+ixabepilone schedule" | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| Imaging_Plan | "Scan in 3 weeks" | A/P | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Referral各项 | 全"None" | - | ✓ | - |
| follow up | "Scan in 3 weeks" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- Ixabepilone (Ixempra) — 微管抑制剂,确认在oncology_drugs.txt
- Capecitabine (Xeloda) — 确认在oncology_drugs.txt

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) summary用HPI的"Irinotecan"而非A/P的"ixabepilone" (2) 原始HER2状态原文不明确 (3) Stage IIA可能不准确

---

### Row 25 (coral_idx=165)

**患者概况**: 56F，筛查发现右乳TNBC(ER-/PR-/HER2-,Ki67 75%)，node negative by biopsy。Clinical Stage IB(AJCC 8th)。Video visit。计划先手术(双侧reduction+右lumpectomy+SLN+port),术后adjuvant chemo+RT。Genetics+social work referral。Full code。多药过敏。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | HPI | ✓ | initial consult |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | "Video Visit/Telehealth" | ✓ | 正确识别 |
| summary | 准确描述TNBC+Stage I | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- triple negative breast cancer" | 病理 | ✓ | 完美 |
| Stage_of_Cancer | "Stage IB" | AJCC staging记录 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| lab_summary | CBC from 09/20/2019 | labs段 | **P2** | 3个月前labs,note本身标注"No visits with results within 1 Month" |
| findings | TNBC Ki67 75%+node negative+PE | 病理+PE | ✓ | - |
| current_meds | "" (空) | - | ✓ | 仅非肿瘤药 |
| goals_of_treatment | "curative" | A/P | ✓ | Stage I adjuvant |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空!) | - | **P1** | 讨论了chemotherapy role但无具体方案 |
| medication_plan | "None" | - | **P2** | A/P讨论了adjuvant chemo,虽无具体drug name但应说"adjuvant chemotherapy planned post-surgery" |
| therapy_plan | "discussed chemo+radiation roles" | A/P | ✓ | - |
| radiotherapy_plan | "discussed radiation" | A/P | ✓ | - |
| Procedure_Plan | "bilateral reduction+lumpectomy+SLN+port" | A/P | ✓ | 完整 |
| Imaging_Plan | "No imaging planned." | - | ✓ | MRI已完成 |
| Genetic_Testing_Plan | "Genetics consult planned." | A/P #5 | ✓ | - |
| Genetics | "Genetics consult" | HPI+A/P | ✓ | - |
| Others | "Social work referral" | "I will refer to social work" | ✓ | 也在Diagnosis段有"Ambulatory Referral to Social Work" |
| follow up | "RTC at post-op visit" | A/P #6 | ✓ | - |
| Advance care | "Full code." | "Code status: Full code" | ✓ | 原文验证存在 |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) lab_summary 3月前 (2) medication_plan "None"应提及planned chemo

---

### Row 26 (coral_idx=166)

**患者概况**: 41F，ER+/PR+/HER2- IDC，1999年早期确诊（lumpectomy+XRT），2006年骨转移(L1)，现为Stage IV。当前用letrozole+goserelin+zolendronic acid。PET-CT示骨转移稳定/略减。新症状：腰痛、易瘀、尿频。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "Reassess lower back pain..." | ✓/P2 | 值正确；归因引错（应引"presents today for hormone therapy follow-up"） |
| second opinion | "no" | "Patient to rtc..." | ✓ | - |
| in-person | "in-person" | "Patient to rtc..." | ✓ | - |
| summary | 准确 | "Hormone positive MBC..." | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | "Hormone positive MBC..." | ✓ | 原文"IDC ER/PR + ***** neg"，推断HER2-合理 |
| Stage_of_Cancer | "Originally Stage I, now metastatic (Stage IV)" | 同上 | ✓ | 合理推断 |
| Metastasis | "Yes, to bone" | 同上 | ✓ | - |
| Distant Metastasis | "Yes, to bone" | 同上 | ✓ | L1+thoracolumbar+sternum+iliac wing |
| lab_summary | "CBC with platelets. No specific values provided." | "obtain CBC with platelets" | P2 | CBC是计划中的lab，非已有结果；应为"No lab results reported" |
| findings | PET-CT骨转移稳定/略减 | "stable disease" | ✓ | 内容准确，归因引用略短 |
| current_meds | letrozole, goserelin, zolendronic acid | "Continue on..." | ✓ | 正确覆盖肿瘤用药 |
| recent_changes | "" | - | ✓ | 确实无变化，继续当前方案 |
| supportive_meds | "zolendronic acid, goserelin (ZOLADEX)" | "Continue on..." | P1 | 两者均为肿瘤治疗药（骨保护+卵巢抑制），非支持性用药；calcium-vitamin D未提 |
| goals_of_treatment | "palliative" | "Hormone positive MBC..." | ✓ | Stage IV正确 |
| response_assessment | PET-CT稳定/略减 | "stable disease" | ✓ | 准确 |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "Continue on [REDACTED], zoladex, and femara" | 同上 | ✓ | - |
| therapy_plan | "Continue on [REDACTED], zoladex and femara" | 同上 | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | 过去XRT正确排除 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "Consider MRI spine..." | 正确引用 | ✓ | - |
| Lab_Plan | "CBC with platelets..." | 正确引用 | P2 | 遗漏UA（A/P明确写"Obtain UA"） |
| Genetic_Testing_Plan | "None planned." | - | ✓ | - |
| Referral | 全部None | - | ✓ | - |
| follow up | 完整引用 | 正确 | ✓ | - |
| Next clinic visit | "if pain worsens" | "Patient to rtc..." | P2 | 遗漏"two weeks"（A/P写"Reassess lower back pain at two weeks"） |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- [ ] zolendronic acid/goserelin 出现在 supportive_meds → 应归入 oncology drugs
- [ ] calcium-vitamin D 未出现在 supportive_meds（合理的支持性用药）

**本行总结**: 0×P0, 2×P1, 4×P2
- P1: (1) supportive_meds含goserelin+zolendronic acid（肿瘤药）(2) Medication_Plan_chatgpt空
- P2: (1) lab_summary把计划当结果 (2) Lab_Plan遗漏UA (3) Next clinic visit遗漏"two weeks" (4) Patient type归因引错

---

### Row 27 (coral_idx=167)

**患者概况**: 60F，绝经后，新确诊Stage I ER+/PR+/HER2-(IHC 0, FISH 1.1) grade 1 IDC右乳，0.9cm，0/3 LN。已完成lumpectomy+SLNB。首次Med Onc就诊讨论辅助治疗。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | 归因引错 | ✓ | 值正确（establish care） |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "90 minutes..." | ✓ | - |
| summary | 准确 | A/P原文 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- grade 1 IDC" | A/P原文 | ✓ | HER2 IHC 0 + FISH 1.1 = HER2- ✓ |
| Stage_of_Cancer | "Stage I" | A/P原文 | ✓ | 0.9cm + 0/3LN = T1N0 = Stage I |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | CBC+CMP详细列出 | 笔记中Labs部分 | ✓ | 数据来自6-8月前，但在note中列出 |
| findings | 术后病理+MRI详细 | 病理引用 | ✓ | 准确 |
| current_meds | "" | - | ✓ | 尚未开始肿瘤治疗 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | 尚无 |
| goals_of_treatment | "curative" | - | ✓ | 早期+辅助 = curative |
| response_assessment | "Not yet on treatment" | - | ✓ | 完美 |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "AI for 5 years + calcium/VitD" | A/P | ✓ | - |
| therapy_plan | "radiation then AI" | A/P | ✓ | - |
| radiotherapy_plan | "rad onc consult, will have radiation" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "DEXA" | A/P | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | 原文未提Oncotype/基因检测 |
| Referral-Specialty | "Radiation oncology consult" | A/P | ✓ | - |
| follow up | "RTC after radiation" | A/P | ✓ | - |
| Next clinic visit | "after radiation, review DEXA, plan AI start" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 0×P2
- P1: (1) Medication_Plan_chatgpt空

---

### Row 28 (coral_idx=168)

**患者概况**: 59F，绝经后，多灶性Grade 2 IDC ER+/PR+/HER2-右乳，s/p lumpectomy+SLNB（pT1c(m), SLN微转移0.5mm）。MammaPrint Low Risk。首次Med Onc会诊。计划开始letrozole+radiation+可能re-excision。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | med onc consultation |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 60min face-to-face | ✓ | - |
| summary | 准确详细 | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 正确引用 | ✓ | - |
| Stage_of_Cancer | "pT1c(m)HER2(sn)" | 同上 | P1 | 乱码；"HER2(sn)"无意义，应为pN1mi(sn)。正确分期应为Stage IB (pT1c, pN1mi) |
| Metastasis | "No" | 微转移引用 | ✓ | SLN微转移=区域，非远处 |
| Distant Metastasis | "No" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 笔记中无lab values |
| findings | 术后病理详细 | 病理引用 | ✓ | 准确全面 |
| current_meds | "letrozole 2.5mg PO daily" | Plan#1 | ✓ | 本次开始（Rx sent），可算current |
| recent_changes | "Start letrozole 2.5mg PO daily" | Plan#1 | ✓ | - |
| supportive_meds | "" | - | ✓ | calcium/VitD正在讨论中 |
| goals_of_treatment | "curative" | - | ✓ | 早期+辅助 |
| response_assessment | "Not mentioned in note." | - | P2 | 应为"Not yet on treatment"更准确 |
| Medication_Plan_chatgpt | 有内容（hormonal therapy） | - | ✓ | 罕见地正常工作！ |
| medication_plan | letrozole+calcium+VitD+moisturizer | Plan引用 | ✓ | 全面 |
| therapy_plan | letrozole+RT | Plan引用 | ✓ | - |
| radiotherapy_plan | "recommending radiation..." | A/P引用 | ✓ | - |
| Procedure_Plan | "surgery Sept 2019" | Plan引用 | ✓ | re-excision |
| Imaging_Plan | "Bone density scan...Bone scan" | A/P引用 | P2 | 末尾多余"Bone scan"（应只有DEXA） |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | MammaPrint已完成 |
| Referral-Specialty | "RT planning" | A/P引用 | ✓ | - |
| Referral-Others | "None" | - | P1 | 遗漏dental eval（Plan#4: "dental eval and clearance if considering bisphosphonate"） |
| follow up | "return Sept for surgery" | A/P引用 | ✓ | - |
| Next clinic visit | "Sept 2019 for surgical planning" | 引用 | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 2×P2
- P1: (1) Stage_of_Cancer乱码"pT1c(m)HER2(sn)" (2) Referral遗漏dental eval
- P2: (1) response_assessment措辞不够具体 (2) Imaging_Plan多余"Bone scan"

---

### Row 29 (coral_idx=169)

**患者概况**: 64F，绝经后，ER-/PR-/HER2+(IHC 3, FISH 8.9) IDC右乳。2007年DCIS未治疗，2016年确诊IDC（9cm mass on PET）。PET无远处转移。首次Med Onc会诊。计划新辅助THP/AC或TCHP。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | consultation to establish care |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 120min face-to-face | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2+ IDC" | A/P | ✓ | IHC 3 + FISH 8.9 = HER2+ ✓ |
| Stage_of_Cancer | "Clinical stage II-III" | A/P | ✓ | 原文A/P写"clinical stage II-III"，9cm+N0 |
| Metastasis | "No" | PET引用 | ✓ | - |
| Distant Metastasis | "No" | PET引用 | ✓ | - |
| lab_summary | 肿瘤标记物+Cr列出 | 引用 | ✓ | 多时间点tumor markers准确 |
| findings | PE+PET详细 | 引用 | ✓ | - |
| current_meds | "" | - | ✓ | 尚未开始治疗 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | "curative intent" | ✓ | 完美（原文明确） |
| response_assessment | "Not yet on treatment" | - | ✓ | 完美 |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | THP/AC or TCHP详细 | A/P引用 | ✓ | 全面准确 |
| therapy_plan | 新辅助chemo详细 | A/P引用 | ✓ | - |
| radiotherapy_plan | "None" | - | P1 | A/P提到"treatment recommendations will include...radiation"，应捕获 |
| Procedure_Plan | "Mediport placement" | A/P引用 | ✓ | - |
| Imaging_Plan | "TTE" | A/P引用 | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | Invitae已完成 |
| Referral | 全部None | - | ✓ | 无明确referral语言 |
| follow up | "contact after weekend" | A/P引用 | ✓ | - |
| Next clinic visit | "after weekend to decide" | A/P引用 | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 0×P2
- P1: (1) Medication_Plan_chatgpt空 (2) radiotherapy_plan遗漏（A/P提到radiation as part of multimodality plan）

---

## 前30行模式回顾 (Row 20-29)

### 统计
| 严重度 | Row 20-29 | Row 0-19 | 累计 Row 0-29 |
|--------|-----------|----------|--------------|
| P0 | 0 | 2 | 2 |
| P1 | 15 | ~47 | ~62 |
| P2 | 21 | ~56 | ~77 |

### Row 20-29 P1 问题分布

| 问题类型 | 出现行数 | 占比 |
|----------|---------|------|
| Medication_Plan_chatgpt 空 | 20,21,22,24,25,26,27,29 (9/10) | 90% |
| supportive_meds含肿瘤药 | 21(denosumab), 26(goserelin+zolendronic acid) | 20% |
| Stage_of_Cancer错误 | 28(乱码"pT1c(m)HER2(sn)") | 10% |
| Type_of_Cancer缺HER2 | 21 | 10% |
| radiotherapy_plan遗漏 | 29(multimodality plan中提到radiation未捕获) | 10% |
| Referral遗漏 | 28(dental eval) | 10% |
| 日期错误 | 21(年份差1年) | 10% |

### 趋势观察

1. **Medication_Plan_chatgpt空仍是头号问题** — 30行中仅3行正常输出（Row 14, 15, 28）。复杂嵌套JSON schema几乎不可用。
2. **supportive_meds分类错误持续** — bone therapy药物(denosumab, zolendronic acid, pamidronate)和LHRH agonist(goserelin)被归入supportive。累计5行出现此问题。
3. **Row 20-29整体质量略优于Row 0-19** — P0降为0，P1/行从2.35降到1.5。可能与后面行的笔记结构更规范有关。
4. **干净行增多** — Row 23(0P1), Row 27(仅Med_Plan_chatgpt) 非常干净。
5. **Stage错误新类型** — Row 28出现LLM将redacted文本错误拼接为staging（"HER2"插入TNM分期）。

### 白名单更新汇总 (Row 20-29新增)

**oncology_drugs.txt 应包含**:
- goserelin/Zoladex（LHRH agonist，Row 26 出现在 supportive_meds）
- zolendronic acid/Zometa（bisphosphonate for bone mets，Row 26 出现在 supportive_meds）

（denosumab/Xgeva 在 Row 13,19,21 已记录）

---

### Row 30 (coral_idx=170)

**患者概况**: 64F，de novo转移性ER+/PR+/HER2-乳腺癌（2010年诊断），骨+肝+可疑脑转移（PET进展）。多线治疗后。计划开始Doxil。Echo EF=59%。便秘/肛裂、髋关节痛、牙齿问题。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 45min | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2-" | A/P | ✓ | - |
| Stage_of_Cancer | "now metastatic (Stage IV)" | A/P | ✓ | de novo metastatic |
| Metastasis | "bones, liver, possibly brain" | A/P | ✓ | PET进展确认 |
| Distant Metastasis | 同上 | - | ✓ | - |
| lab_summary | "No labs in note." | - | P2 | A/P写"Labs reviewed, wnl"；虽无具体数值但有信息 |
| findings | PET进展详细 | A/P | ✓ | 肝转移增大、骨转移增多、右额叶可疑 |
| current_meds | "" | - | ✓ | 间歇期无当前肿瘤用药 |
| recent_changes | "Start Doxil 07/01/2021" | A/P | ✓ | - |
| supportive_meds | "ondansetron, prochlorperazine" | - | ✓ | 止吐药=支持用药 |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | PET进展详细 | A/P | ✓ | 准确反映progression |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "Doxil IV q28d" | A/P | ✓ | - |
| therapy_plan | "Doxil + PETCT after 3 cycles" | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "Brain MRI + MRI pelvis + PETCT" | A/P | P2 | 遗漏echo repeat（A/P: "Repeat echo in 3 months, next due end of August 2021"） |
| Lab_Plan | "No labs planned." | - | ✓ | ammonia是patient request非A/P order |
| Genetic_Testing_Plan | "None planned." | - | ✓ | 遗传咨询未跟进但非新plan |
| Referral-Specialty | "GI referral, Oral medicine" | A/P | ✓ | - |
| Referral-Others | "Physical therapy referral" | A/P | ✓ | - |
| follow up | "4 weeks prior to C2" | A/P | ✓ | - |
| Next clinic visit | 同上 | A/P | ✓ | - |
| Advance care | "Full code." | "Code status: Full code"(pos 8287) | ✓/P2 | 值正确但归因写NOT_IN_NOTE（归因checker误判） |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) lab_summary遗漏"wnl" (2) Imaging_Plan遗漏echo repeat (3) Advance care归因误判NOT_IN_NOTE

---

### Row 31 (coral_idx=171)

**患者概况**: 82F，2010年Stage IIA左乳ER+/PR-/HER2+ pleomorphic lobular carcinoma, s/p lumpectomy+SLNB+TCH×6+herceptin+RT+tamoxifen→exemestane。2018年10月转移复发（颈/胸/腹/盆腔LN+左附件）。2018年12月起[REDACTED]+herceptin+[REDACTED]，后因腹泻停[REDACTED]。PET 2019年6月CR。转诊到新诊所。Full code, living will on file。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | care transfer, New Patient Evaluation |
| second opinion | "no" | - | ✓ | 转诊非second opinion |
| in-person | "in-person" | 60min | ✓ | - |
| summary | 准确 | - | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2+ pleomorphic lobular" | A/P | ✓ | 完美 |
| Stage_of_Cancer | "Originally Stage IIA, now Stage IV" | - | ✓ | - |
| Metastasis | "LN of neck/chest/abdomen/pelvis + adnexal" | A/P | ✓ | - |
| Distant Metastasis | 同上 | - | ✓ | 非腋窝LN=远处 |
| lab_summary | ESR 27, CRP 3.1 (from 2016) | Labs段 | P2 | 3年前的数据 |
| findings | PET CR + PE详细 | PET/PE | ✓ | - |
| current_meds | "exemestane, trastuzumab, pertuzumab" | 药物列表 | ✓ | 反映active med list |
| recent_changes | "Stopped [REDACTED] due to diarrhea" | A/P | ✓ | - |
| supportive_meds | "loperamide (IMODIUM)" | 药物列表 | ✓ | 止泻药=支持用药 |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | PET CR详细 | PET结果 | ✓ | 完美（complete response） |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "Continuing herceptin" | A/P | P2 | 未提exemestane |
| therapy_plan | "Continue herceptin" | A/P | P2 | 未提exemestane |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Genetic_Testing_Plan | "None planned." | - | ✓ | 已做（negative） |
| Referral-Nutrition | "nutrition consult" | A/P+Diagnosis段 | ✓ | 两处均捕获 |
| Referral-Others | "Exercise counseling" | Diagnosis段 | ✓ | - |
| follow up | "RTC 3 and 6 weeks" | A/P | ✓ | - |
| Next clinic visit | 同上 | A/P | ✓ | - |
| Advance care | "Full code. Living will on file." | 原文 | ✓ | 完美 |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) lab_summary 3年前 (2) medication_plan未提exemestane (3) therapy_plan未提exemestane

---

### Row 32 (coral_idx=172)

**患者概况**: 63F，2010年7月左乳ILC ER+/PR+/HER2-，Stage IIB/IIIA（临床/病理），s/p bilateral mastectomies+TC×6+XRT。2011年2月起adjuvant letrozole（>5年，患者偏好继续）。关节僵硬、头痛。PE无复发证据。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | - | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- ILC" | PMH | ✓ | - |
| Stage_of_Cancer | "Originally Stage IIB, now Stage IIIA" | PMH | P1 | 误解：IIB和IIIA是同一确诊的临床/病理分期，非进展。"now IIIA"暗示进展但患者NED |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | PE无复发 | A/P | ✓ | - |
| current_meds | "letrozole" | A/P | ✓ | - |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "no evidence of recurrence" | A/P | ✓ | - |
| Medication_Plan_chatgpt | 有内容（letrozole） | A/P | ✓ | 正常工作 |
| medication_plan | "letrozole+calcium/VitD+NSAIDs" | A/P | ✓ | - |
| therapy_plan | "Continue letrozole" | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | 过去XRT正确排除 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "Consider MRI brain" | A/P | ✓ | 条件性 |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| Referral | 全部None | - | ✓ | - |
| follow up | "6 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 0×P2
- P1: (1) Stage_of_Cancer误解双重分期为进展

---

### Row 33 (coral_idx=173)

**患者概况**: 71F，Stage III左乳IDC ER+/PR-/HER2-。复杂历史：2011 lumpectomy→2012快速局部复发(3.3cm, 11+LN), bilateral MX+缩短AC/T+anastrozole(自行停药)→2020第二次局部复发(1.7cm, grade 3, ER+/PR+/HER2-)。PET无远处转移。计划：tamoxifen+胸壁RT referral。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | - | ✓ | - |
| summary | 准确详细 | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- IDC" | - | P2 | 原始PR-但2020复发PR+(50%)，应注明PR变化 |
| Stage_of_Cancer | "Originally Stage III, now local recurrence" | A/P | ✓ | 准确 |
| Metastasis | "No" | PET | ✓ | - |
| Distant Metastasis | "No" | PET | ✓ | - |
| lab_summary | "August 2018 unremarkable" | DATA段 | ✓ | 2年前但原文如此 |
| findings | FNA+excision+PET+MRI | - | ✓ | 全面 |
| current_meds | "arimidex" | HPI开头 | P1 | **已自行停药**（"self D/Ced against advice"），不应列为current |
| recent_changes | "tamoxifen 20mg PO qD" | Plan | ✓ | 新开始 |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | 局部复发+无远处转移 |
| response_assessment | 局部复发描述 | - | P2 | 更像disease status而非treatment response（她停了arimidex后复发） |
| Medication_Plan_chatgpt | 有内容（tamoxifen） | Plan | ✓ | 正常工作 |
| medication_plan | "tamoxifen 20mg PO qD" | Plan | ✓ | - |
| therapy_plan | "tamoxifen, no chemo benefit" | Plan | ✓ | CALOR study讨论 |
| radiotherapy_plan | "chest wall RT, referral accepted" | Plan | ✓ | 准确 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "check labs" | Plan | ✓ | - |
| Referral-Specialty | "chest wall RT consult" | Plan | ✓ | - |
| follow up | "6 months" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: (1) current_meds "arimidex"已停（self D/C against advice）
- P2: (1) Type_of_Cancer未反映复发PR变化 (2) response_assessment非治疗响应

---

### Row 34 (coral_idx=174)

**患者概况**: 40F，2018年右乳ILC 1.2cm grade 2 pT1cN0(sn)，s/p lumpectomy。辅助tamoxifen（April 2018起）。Note中有矛盾：HPI说tamoxifen，A/P说anastrozole（40岁绝经前患者用AI不合常规）。无复发证据。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | - | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ILC" | 病理 | ✓ | 原文无明确ER/PR/HER2 |
| Stage_of_Cancer | "pT1cN0(sn)" | 病理 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 病理+PE详细 | - | ✓ | - |
| current_meds | "anastrozole" | A/P | P2 | 忠实于A/P但与HPI矛盾（HPI说tamoxifen）；40岁premenopausal用AI不合常规 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "no recurrence, mammogram no malignancy" | A/P | ✓ | - |
| Medication_Plan_chatgpt | 有内容（anastrozole） | A/P | ✓ | 正常工作 |
| medication_plan | "Continue anastrozole" | A/P | ✓ | 忠实于A/P |
| therapy_plan | "continue anastrozole + mammogram + f/u" | A/P | P2 | 含imaging（mammogram属Imaging_Plan） |
| radiotherapy_plan | "None" | - | ✓ | - |
| Imaging_Plan | "bilateral mammogram Sept 2019" | A/P | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| follow up | "6 months" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2
- P2: (1) current_meds HPI/A/P矛盾（tamoxifen vs anastrozole），提取忠实于A/P (2) therapy_plan含imaging内容

---

### Row 35 (coral_idx=175)

**患者概况**: 27F，绝经前，pT3N0右乳ER+/PR+/HER2- grade III mixed ductal/mucinous carcinoma。s/p bilateral MX 12/06/20。术后感染。01/29开始tamoxifen，02/06 zoladex，02/13 Taxol→Abraxane（grade 3 infusion reaction）。今天cycle 8。右臂肿胀。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | cycle 8 | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 47min | ✓ | - |
| summary | 准确详细 | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- grade III mixed ductal/mucinous" | 病理 | ✓ | IHC+FISH均确认 |
| Stage_of_Cancer | "pT3N0" | 病理 | ✓ | 8.4cm, 0/4 LN |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | CBC+CMP详细 | Labs段 | ✓ | 近期数据 |
| findings | 右臂肿胀+PE | A/P | ✓ | - |
| current_meds | "Abraxane, zoladex" | A/P | P1 | **遗漏tamoxifen**——A/P开头写"started tamoxifen on 01/29/21"，未停药 |
| recent_changes | "Switched Taxol→Abraxane" | A/P | ✓ | - |
| supportive_meds | "Zofran, Compazine" | A/P | P2 | 遗漏valtrex(HSV ppx), ativan, lexapro, omeprazole |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "Not mentioned" | - | ✓ | 术后辅助，无imaging response |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | 全面列出所有药物 | A/P | ✓ | 包含chemo+支持用药 |
| therapy_plan | "Abraxane+zoladex+valtrex" | A/P | P2 | valtrex是支持用药非treatment |
| radiotherapy_plan | "rad onc referral" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "doppler for DVT" | A/P | ✓ | - |
| Referral-Specialty | "Radiation oncology" | A/P | ✓ | - |
| Referral-Others | "None" | - | P2 | A/P有条件性PT referral（"if negative for DVT, contact PT for lymphedema"） |
| follow up | "RTC 2 weeks" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 3×P2
- P1: (1) current_meds遗漏tamoxifen (2) Medication_Plan_chatgpt空
- P2: (1) supportive_meds不完整 (2) therapy_plan含valtrex (3) Referral遗漏conditional PT

---

### Row 36 (coral_idx=176)

**患者概况**: 61F，新确诊Stage IIA左乳TNBC(IDC grade 3, 2.3cm, N0)，s/p bilateral MX July 2020。Video visit。她在另一中心有oncologist（已推荐AC/Taxol），来此做new patient evaluation（实为second opinion）。Full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | New Patient Evaluation |
| second opinion | "no" | - | P1 | 应为"yes"——另一中心oncologist已推荐AC/Taxol,患者来此consultation,将返回原中心治疗 |
| in-person | "Televisit" | Video Visit | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | A/P | ✓ | triple negative |
| Stage_of_Cancer | "Stage IIA" | A/P+staging | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 术后病理 | A/P | ✓ | - |
| current_meds | "" | - | ✓ | 尚未开始治疗 |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "dd AC followed by Taxol" | A/P | ✓ | - |
| therapy_plan | "AC/Taxol, no hormone/radiation" | A/P | ✓ | - |
| radiotherapy_plan | "None" | A/P | ✓ | 明确写"no indication for radiation" |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Referral | 全部None | - | ✓ | 无outgoing referral |
| follow up | "proceed with chemo at [REDACTED]" | A/P | ✓ | 返回原中心治疗 |
| Advance care | "Full code." | 原文 | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 0×P2
- P1: (1) second opinion应为"yes" (2) Medication_Plan_chatgpt空

---

### Row 37 (coral_idx=177)

**患者概况**: 43F，BRCA1突变，Stage IIB左乳IDC 6.8cm N0 ER-/PR+(15% weak)/HER2-。s/p neoadjuvant [REDACTED]×4+Taxol 5周（毒性停药）。MRI mild response，PET无远处转移，但肿瘤又在增大。拒绝进一步IV chemo。计划bilateral MX（Jan 31）。术后adjuvant olaparib+xeloda。Full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 转诊care transfer |
| second opinion | "no" | - | ✓ | care transfer非second opinion |
| in-person | "in-person" | 75min | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR+/HER2- IDC" | A/P | ✓ | PR+(15% weak) |
| Stage_of_Cancer | "Stage IIB" | staging | ✓ | - |
| Metastasis | "No" | PET | ✓ | - |
| lab_summary | CBC+CMP+TSH+A1c详细 | Labs段 | ✓ | - |
| findings | 肿瘤增大+imaging | PE+imaging | ✓ | - |
| current_meds | "" | - | ✓ | 化疗已停 |
| recent_changes | "declined IV therapy, qualify for xeloda" | A/P | P2 | 更像plan而非recent change；真正的change是停Taxol |
| supportive_meds | "hydrocodone, tramadol" | 药物列表 | ✓ | 止痛=supportive |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "tumor enlarging, MRI mild response, PET negative" | A/P | ✓ | 准确反映mixed picture |
| Medication_Plan_chatgpt | 有内容（chemo+hormonal） | A/P | ✓ | 正常工作 |
| medication_plan | "olaparib + xeloda" | A/P | ✓ | - |
| therapy_plan | "olaparib + xeloda" | A/P | P2 | 遗漏hormonal blockade讨论（A/P#6: PR 15%, will offer per guidelines） |
| radiotherapy_plan | "discussed radiation" | A/P#5 | ✓ | - |
| Procedure_Plan | "bilateral MX Jan 31" | A/P#3 | ✓ | - |
| Referral-Specialty | "Gynecologic Oncology 4" | Diagnosis段 | P2 | 有提取artifact ("4"来自numbering) |
| Referral-Others | "Social work" | Diagnosis+Psychologic段 | ✓ | - |
| follow up | "after surgery recovery" | A/P | ✓ | - |
| Advance care | "full code." | 原文 | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 3×P2
- P2: (1) recent_changes是plan非change (2) therapy_plan遗漏hormonal blockade (3) Referral有artifact "4"

---

### Row 38 (coral_idx=178)

**患者概况**: 27F，新确诊Stage II左乳TNBC(IDC grade 3, T2N1, 3.6cm+axillary LN+)。CT/MRI/bone scan无远处转移。计划参加ISPY trial：neoadjuvant paclitaxel×12w→AC×4→surgery。Goserelin保护卵巢功能。需port placement、echo、screening biopsies。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | consultation |
| second opinion | "no" | - | ✓ | care transfer to trial center |
| in-person | "in-person" | 70min | ✓ | - |
| summary | 准确 | A/P | ✓ | T2N1 |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | - | ✓ | triple negative |
| Stage_of_Cancer | "At least Stage II" | A/P | ✓ | - |
| Metastasis | "No" | CT negative | ✓ | axillary LN=regional |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 病理+imaging | A/P | ✓ | - |
| current_meds | "" | - | ✓ | 尚未治疗 |
| goals_of_treatment | "curative" | - | ✓ | neoadjuvant |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "paclitaxel→AC + goserelin" | A/P | ✓ | 全面 |
| therapy_plan | neoadjuvant详细 | A/P | ✓ | ISPY trial方案 |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "Port + screening biopsies" | A/P | ✓ | - |
| Imaging_Plan | "MRI breasts + echo" | A/P | ✓ | - |
| Lab_Plan | "ISPY labs" | A/P | ✓ | - |
| Genetic_Testing_Plan | "[REDACTED] on genetic testing results" | A/P | P2 | 含义不明——可能是"Pending"或"Awaiting"基因检测结果 |
| Referral | 全部None | - | ✓ | - |
| follow up | "start therapy within 2 weeks" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 1×P2
- P1: (1) Medication_Plan_chatgpt空
- P2: (1) Genetic_Testing_Plan含redacted文本，含义不明

---

### Row 39 (coral_idx=179)

**患者概况**: 62F，MS on chronic immunosuppression，新确诊Stage 2右乳IDC grade 1 ER 95%/PR 5%/HER2 2+ FISH neg。s/p partial MX+SLNB（1 SLN微转移0.04cm by direct extension）。计划：letrozole（Rx sent），DEXA，PT referral。On Prolia（osteoporosis）。可能需radiation（"if no radiation is planned then start letrozole"）。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | consult |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 90min | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER 95, PR 5, HER2 2+ FISH neg G1 IDC" | 病理 | ✓ | 原始数据详细 |
| Stage_of_Cancer | "Stage 2" | A/P | ✓ | - |
| Metastasis | "No" | - | ✓ | SLN direct extension=regional |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 病理+MRI详细 | - | ✓ | - |
| current_meds | "letrozole" | Rx given | ✓ | 刚开处方 |
| recent_changes | "Rx for letrozole given" | A/P | ✓ | - |
| supportive_meds | "ondansetron" | 药物列表 | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "Not mentioned" | - | P2 | 应为"Not yet on treatment" |
| Medication_Plan_chatgpt | 有内容（hormonal therapy） | A/P | ✓ | 正常工作！ |
| medication_plan | "letrozole + Prolia" | A/P | ✓ | - |
| therapy_plan | "letrozole adjuvant" | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | P1 | A/P暗示radiation正在考虑（"if no radiation is planned"=条件语句表明radiation是可能的） |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "DEXA" | A/P | ✓ | - |
| Referral-Others | "None" | - | P1 | 遗漏PT referral（A/P明确写"PT referral"） |
| follow up | "3 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 1×P2
- P1: (1) radiotherapy_plan "None"但radiation正在考虑 (2) Referral遗漏PT referral
- P2: (1) response_assessment措辞

---

## 前40行模式回顾 (Row 30-39)

### 统计
| 严重度 | Row 30-39 | Row 20-29 | Row 0-19 | 累计 Row 0-39 |
|--------|-----------|-----------|----------|--------------|
| P0 | 0 | 0 | 2 | 2 |
| P1 | 12 | 15 | ~47 | ~74 |
| P2 | 18 | 21 | ~56 | ~95 |

### Row 30-39 P1 问题分布

| 问题类型 | 出现行数 | 占比 |
|----------|---------|------|
| Medication_Plan_chatgpt 空 | 30,31,35,36,38 (5/10) | 50% |
| Referral遗漏 | 35(conditional PT), 39(PT referral) | 20% |
| radiotherapy_plan遗漏 | 39(radiation being considered) | 10% |
| current_meds错误 | 33(arimidex已停用), 35(遗漏tamoxifen) | 20% |
| second opinion判断错误 | 36(应为yes) | 10% |
| Stage_of_Cancer误解 | 32(dual staging误解为progression) | 10% |

### 趋势观察

1. **Medication_Plan_chatgpt空率下降**: 30行中5行正常（Row 28,32,33,34,37,39），50%空率（vs前20行90%）。可能因为这批行的A/P结构更清晰。
2. **current_meds判断改善但仍有问题**: Row 33(已停arimidex仍列为current)和Row 35(遗漏tamoxifen)表明模型难以处理：(a)note内部HPI/A/P矛盾 (b)未在A/P plan中明确"continue"的药物。
3. **Referral遗漏是新模式**: Row 28(dental eval), 35(conditional PT), 39(PT referral)——三行都遗漏了A/P中明确写的referral。PT referral尤其容易被忽略。
4. **Row 30-39整体质量继续提高**: P1/行从1.5(Row 20-29)降到1.2。P0持续为0。
5. **干净行**: Row 34(0P1), Row 37(0P1) 非常干净。
6. **Note内部矛盾**: Row 34(HPI说tamoxifen, A/P说anastrozole)——提取忠实于A/P但临床上可能有问题。

### 白名单更新汇总 (Row 30-39新增)

无新增（本批次无药物分类问题）。

### 累计前40行系统性问题排名

| 排名 | 问题 | 出现频率 | 趋势 |
|------|------|---------|------|
| 1 | Medication_Plan_chatgpt空 | ~24/40 (60%) | 下降中 |
| 2 | supportive_meds含肿瘤药 | ~5/40 (12.5%) | 稳定 |
| 3 | Referral遗漏 | ~5/40 (12.5%) | 上升 |
| 4 | radiotherapy_plan遗漏 | ~3/40 (7.5%) | 稳定 |
| 5 | current_meds错误(时态/遗漏) | ~4/40 (10%) | 新增 |
| 6 | Stage_of_Cancer错误 | ~4/40 (10%) | 下降 |
| 7 | Type_of_Cancer缺HER2 | ~7/40 (17.5%) | 主要在前20行 |
| 8 | response_assessment措辞 | ~10/40 (25%) | 下降 |

---

### Row 40 (coral_idx=180)

**患者概况**: 32F，ATM突变携带者，左乳IDC grade 3, 3cm, ER+(90%)/PR+(1% weak)/HER2 1+(FISH unavailable)。s/p bilateral MX+left SLNB（1/3 SLN微转移0.022cm）。MammaPrint High Risk。LVEF 79%。计划AC-Taxol（Taxol first×12w then AC）。术后计划ovarian suppression+AI，可能ribociclib trial。Port placement本周。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | P1 | 应为"Follow up"——这是第3次就诊（prior notes 03/17 and 04/21） |
| second opinion | "no" | - | ✓ | care transfer近home chemo |
| in-person | "in-person" | 45min | ✓ | - |
| summary | 准确 | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2 neg IDC" | A/P | P2 | PR weakly+(1%)被归为PR-，borderline但严格说原文写"PR weakly +" |
| Stage_of_Cancer | "" | - | P1 | 空——3cm+SLN micromet可推断Stage IIA/IB |
| Metastasis | "No" | - | ✓ | SLN micromet=regional |
| lab_summary | CBC+Ferritin+Pregnancy详细 | Labs段 | ✓ | - |
| findings | 术后病理详细 | - | ✓ | - |
| current_meds | "" | - | ✓ | 尚未开始chemotherapy |
| recent_changes | "decided to proceed with AC-Taxol" | A/P | ✓ | - |
| supportive_meds | "docusate (COLACE)" | 药物列表 | ✓ | bowel regimen |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "After chemo: AI+ovarian suppression" | A/P | P2 | 遗漏main treatment AC-Taxol |
| therapy_plan | "Taxol×12w→AC then AI+OS" | A/P | ✓ | 全面 |
| radiotherapy_plan | "None" | - | ✓ | 因ATM选择MX而非BCS+RT |
| Procedure_Plan | "port placement" | A/P | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Referral | 全部None | - | ✓ | - |
| follow up | "Not specified" | - | ✓ | 将在另一中心治疗 |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 3×P1, 2×P2
- P1: (1) Patient type应为Follow up (2) Stage_of_Cancer空 (3) Medication_Plan_chatgpt空
- P2: (1) Type_of_Cancer PR-但原文PR weak+ (2) medication_plan遗漏AC-Taxol

---

### Row 41 (coral_idx=181)

**患者概况**: 41F，绝经前，右乳multifocal IDC（0.9cm+0.3cm）grade 1, PR+ 95%, HER2/neu neg。0/5 SLN。s/p excision+reexcision+SLNB+RT（1月完成）。今天开始tamoxifen 5年。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "PR+ IDC" | 病理 | P1 | 缺HER2-（原文写"HER2/neu negative"）；ER被redacted |
| Stage_of_Cancer | "Approximately Stage I-II" | - | P2 | 0.9cm+0/5LN=Stage I，不需"approximately" |
| Metastasis | "No" | - | ✓ | 0/5 SLN |
| findings | 病理详细 | HPI | ✓ | - |
| current_meds | "" | - | ✓ | tamoxifen刚开处方 |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "Not mentioned" | - | ✓ | 完成RT, 非response assessment |
| Medication_Plan_chatgpt | 有内容（tamoxifen） | - | ✓ | - |
| medication_plan | "tamoxifen 5 years" | Plan | ✓ | - |
| therapy_plan | "tamoxifen 5 years" | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | 过去RT正确排除 |
| Imaging_Plan | "diagnostic mammogram at next visit" | Plan | ✓ | - |
| follow up | "4-6 weeks" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 1×P2
- P1: (1) Type_of_Cancer缺HER2-
- P2: (1) Stage_of_Cancer模糊

---

### Row 42 (coral_idx=182)

**患者概况**: 38F，BRCA neg。第一癌（2010, age 27）：Stage I左乳TNBC 1.5cm, s/p lumpectomy+ddAC×4→T×4+RT。第二原发（2021）：左乳IDC 1.3cm grade 3 TNBC(ER-/PR-/HER2 FISH neg, Ki-67>80%), s/p bilateral MX+left SLNB 0/2。计划taxol/carboplatin×4。Full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | A/P | ✓ | triple negative |
| Stage_of_Cancer | "Stage I (second primary)" | A/P | ✓ | - |
| Metastasis | "No" | 0/2 SLN | ✓ | - |
| lab_summary | CBC+CMP+Hepatitis+Thyroid详细 | Labs段 | ✓ | - |
| findings | 术后病理 | A/P | ✓ | - |
| current_meds | "" | - | ✓ | 尚未开始chemo |
| supportive_meds | "granisetron, compazine, senna" | 药物列表 | ✓ | 止吐+bowel regimen |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} | - | P1 | 空 |
| medication_plan | "taxol/carboplatin ×4" | A/P | ✓ | - |
| therapy_plan | 同上 | A/P | ✓ | - |
| radiotherapy_plan | null | - | P2 | 应为"None"字符串，非null |
| Lab_Plan | "No labs planned." | - | P1 | A/P写"RTC 2 days prior to cycle...draw and visit"=pre-chemo lab draw |
| follow up | "2 days prior to C1 Oct 14" | A/P | ✓ | - |
| Advance care | "Full code." | pos 3782 | ✓ | 原文确认 |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 1×P2
- P1: (1) Medication_Plan_chatgpt空 (2) Lab_Plan遗漏pre-chemo lab draw
- P2: (1) radiotherapy_plan为null非"None"

---

### Row 43 (coral_idx=183)

**患者概况**: 33F，BRCA1突变，ER+/PR+/HER2- node+ left breast cancer。s/p neoadjuvant ddAC-Taxol→bilateral MX+left SLNB (10/07/18)。Pathology: 1cm residual grade 2 IDC (15% cellularity) + SLN micromet 0.07cm。计划radiation (clinical trial 3v5weeks, 12/16/18), then AI。BSO planned 12/02/18。Pulmonary nodule stable。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2-" | 开头 | P2 | A/P pathology说"PR-"，开头说"PR+"，note内部矛盾 |
| Stage_of_Cancer | "Not mentioned" | - | P2 | 可从T/N推断但原文未明确写 |
| Metastasis | "No" | - | ✓ | SLN micromet=regional |
| findings | 术后病理+PE | - | ✓ | - |
| current_meds | "" | - | ✓ | chemo已完成，AI未开始 |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | P1 | **错误**——她完成了neoadjuvant chemo，pathology显示residual disease(1cm, 15% cellularity)=partial response |
| Medication_Plan_chatgpt | 有内容（AI+RT） | - | ✓ | - |
| medication_plan | "AI after RT + Zoladex" | A/P | ✓ | - |
| therapy_plan | "AI + ribociclib trial + Zoladex" | A/P | ✓ | - |
| radiotherapy_plan | "clinical trial 3v5w, start 12/16/18" | A/P | ✓ | 准确 |
| Procedure_Plan | "BSO + Zoladex auth" | A/P | P2 | Zoladex是药物非手术 |
| Imaging_Plan | "CT Chest in 1 year" | A/P | ✓ | pulmonary nodule f/u |
| Referral-Nutrition | "follow up 11/30/18" | A/P | ✓ | - |
| Referral-Specialty | "Rad onc consult" | A/P | ✓ | - |
| Referral-Others | "PT referral" | A/P | ✓ | - |
| follow up | "01/05/19" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: (1) response_assessment遗漏pathologic response（residual disease after neoadjuvant）
- P2: (1) PR+/PR-矛盾 (2) Stage未推断 (3) Procedure_Plan含Zoladex(药物)

---

### Row 44 (coral_idx=184)

**患者概况**: 37F，Stage IIIB→Stage IV右乳TNBC。s/p ddAC×4+Taxol 8w(progressed)→lumpectomy+ALND(3cm residual, 2/9 LN+)→RT+xeloda。2022年1月确诊肺+右肺门LN转移。PD-L1+(10%)。Second opinion。计划gemzar/carboplatin+[REDACTED]。基因检测阴性。Video consult。Full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "yes" | HPI | ✓ | "here for a second opinion" |
| in-person | "Televisit" | Video Consult | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- TNBC" | A/P | ✓ | - |
| Stage_of_Cancer | "Originally IIIB, now Stage IV" | Staging+A/P | ✓ | - |
| Metastasis | "lung + right hilar LN" | A/P | ✓ | - |
| Distant Metastasis | 同上 | A/P | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | video consult无labs |
| current_meds | "" | - | ✓ | 无治疗since Jan 2022 |
| goals_of_treatment | "palliative" | A/P | ✓ | "not curable but treatable" |
| response_assessment | "no treatment, no response" | - | ✓ | - |
| Medication_Plan_chatgpt | 有内容 | A/P | ✓ | - |
| medication_plan | "gemzar/carboplatin + 2nd/3rd line" | A/P | ✓ | 全面 |
| therapy_plan | 同上详细 | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| follow up | "Not specified" | - | ✓ | second opinion visit |
| Advance care | "full code." | 原文 | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 0×P2 — 完美提取

---

### Row 45 (coral_idx=185)

**患者概况**: 48F，绝经后（s/p BSO），ER+/PR+/HER2- IDC 右乳。合并结节病、肾动脉瘤、贫血。完成新辅助 Taxol+[REDACTED]→右乳部分切除 06/23/21。病理：3.5cm 残留病灶，切缘阳性，2/2 SLN 阳性伴结外扩展。计划再切除→XRT→abemaciclib。今日开始 letrozole。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | "Follow up 2-3 months" | ✓ | - |
| second opinion | "no" | email re axillary dissection | ✓ | - |
| in-person | "in-person" | "55 minutes..." | ✓ | - |
| summary | 48F post-menopausal ER+ IDC... | HPI line | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 病理段 | P2 | 术后病理 PR=0% (Negative)，临床描述 PR+；提取跟了临床描述，可注明 biomarker 变化 |
| Stage_of_Cancer | "Originally Stage IIA" | 病理段 | **P1** | 原文无 "Stage IIA" 字样；病理为 ypT2/pN[REDACTED]，原始临床分期未提及，"Originally Stage IIA" 疑幻觉 |
| Metastasis | "Not sure" | "2/2 positive nodes..." | **P1** | 已确认区域淋巴结转移（2/2 SLN+/ENE+）；纵隔淋巴结已活检证实为结节病非转移；应为"Yes, regional nodal" |
| Distant Metastasis | "Not sure" | 同上 | **P1** | 纵隔淋巴结经支气管镜活检证实为结节病（良性肉芽肿）；肺小结节 0.2cm 未定性；团队按治愈意图治疗→无确认远处转移 |
| lab_summary | 完整 CMP/CBC/VitD/CRP/ESR | 检验段 | ✓ | 全面 |
| findings | 病理+CT+MRI+PET+体检 | 多来源 | ✓ | 全面 |
| current_meds | "letrozole" | "start letrozole now" | ✓ | 今日开始=当前用药 |
| recent_changes | "Started letrozole + discussed abemaciclib" | A/P | ✓ | - |
| supportive_meds | "" (空) | - | **P1** | A/P 明确 "Continue naproxen 500mg BID"（疼痛），属 pain medication 应列入 |
| goals_of_treatment | "curative" | "re-excision" | ✓ | 归因引用不直接支持 curative 判断但结论正确 |
| response_assessment | "currently responding...no evidence of recurrence" | 病理段 | **P1** | 化疗已完成，非"currently responding"；术后病理有 3.5cm 残留+切缘阳性="partial pathologic response"；"no evidence of recurrence"说法错误，是残留病灶非复发 |
| Medication_Plan_chatgpt | 有内容(hormonal) | A/P | P2 | 只列了 hormonal therapy，缺 radiotherapy 条目 |
| medication_plan | letrozole+naproxen+allegra+iron... | A/P | P2 | gabapentin 描述为 "Continue" 但原文是 "discussed...does not wish to restart"→未继续 |
| therapy_plan | "letrozole + abemaciclib after xrt" | A/P | ✓ | - |
| radiotherapy_plan | "fup with Rad Onc after re-excision" | A/P | ✓ | - |
| procedure_plan | "re-excision, axillary dissection" | A/P | P2 | axillary dissection 仅在讨论/邮件询问阶段，非确定计划 |
| imaging_plan | "MRA abdomen 1yr + baseline DEXA" | A/P | ✓ | - |
| lab_plan | "No labs planned." | - | **P1** | A/P "Repeat in 3-4 months"（铁代谢检查）被遗漏 |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral-Specialty | "Rad Onc consult" | A/P | ✓ | - |
| Referral-Others | "None" | - | **P1** | HPI 明确 "Referred to PT, but PT...has agreed to see her"；A/P 仅说 "Discussed PT in the past"，pipeline 只看 A/P 导致遗漏 |
| follow up | "2-3 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 全文无 code status |

**白名单笔记**: 无新发现

**本行总结**: 0×P0, 6×P1, 4×P2 — 复杂笔记（合并结节病/肾动脉瘤/贫血），多处提取不精确
- P1: Stage 幻觉("Originally Stage IIA")、Metastasis/Distant 均"Not sure"（已活检排除纵隔转移）、supportive_meds 漏 naproxen、response_assessment 措辞错误、Lab_Plan 漏铁代谢复查、Referral 漏 PT
- P2: PR biomarker 变化未注明、Medication_Plan_chatgpt 缺 radiotherapy、gabapentin 措辞、axillary dissection 过度确定

---

### Row 46 (coral_idx=186)

**患者概况**: 41F，绝经前，左乳 3cm 中级别 DCIS（伴粉刺样坏死），ER+/PR+。已行切除+再切除（切缘清）。Second opinion visit at UCSF。推荐放疗+tamoxifen。BRCA 检测已做但结果未出。患者将返回原医生处接受治疗。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | second opinion 首诊 |
| second opinion | "yes" | "return to primary physicians" | ✓ | - |
| in-person | "in-person" | 同上 | ✓ | - |
| summary | 41F DCIS second opinion | Assessment 段 | ✓ | - |
| Type_of_Cancer | "ER+/PR+ DCIS" | 病理段 | ✓ | DCIS 不检测 HER2，正确未提 |
| Stage_of_Cancer | "Not mentioned in note" | - | P2 | DCIS 定义上即 Stage 0 (Tis N0 M0)，可推断 |
| Metastasis | "No" | 再切除段 | ✓ | DCIS 无转移 |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 病理+MRI+体检 | 病理段 | ✓ | 全面 |
| current_meds | "" | - | ✓ | 无当前肿瘤用药 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "risk reduction" | 放疗降低复发风险 | ✓ | 恰当：DCIS 治疗目标是降低复发/进展风险 |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | 有内容(RT+hormonal) | 原文 | ✓ | tamoxifen 副作用描述全面 |
| medication_plan | "RT + tamoxifen recommended" | A/P | ✓ | - |
| therapy_plan | 同上 | A/P | ✓ | - |
| radiotherapy_plan | "recommended, risk ↓ to 3-4%" | 原文 | ✓ | - |
| procedure_plan | "No procedures planned." | - | ✓ | 再切除已完成 |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "BRCA testing recommended, results N/A" | 原文 | P2 | BRCA 检测已做（"has been performed"），非新计划；结果待出 |
| Referral-Genetics | "BRCA testing recommended" | 原文 | P2 | 这是检测而非转诊到遗传咨询门诊；prompt 要求 "referral TO genetics clinic" |
| Referral-Specialty | "Radiation therapy consult" | A/P | ✓ | 推荐放疗 |
| follow up | "return to primary physicians" | 原文 | ✓ | second opinion 后返回原医生 |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 3×P2 — 干净提取，DCIS 特殊病例处理得当
- P2: Stage 可推断为 Stage 0、genetic_testing_plan 已做非新计划、Referral-Genetics 混淆检测与转诊

---

### Row 47 (coral_idx=187)

**患者概况**: 46F，左乳肿块，活检示"至少 DCIS"（中级别），ER+/PR+。MRI 示 31mm 病灶范围。Med onc 初诊。分期待手术后确定。讨论：若仅 DCIS→XRT +/- 内分泌；若发现浸润→LN 活检+辅助治疗。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | med onc 初诊 |
| second opinion | "no" | A/P | ✓ | - |
| in-person | "in-person" | "50 min visit" | ✓ | - |
| summary | L breast DCIS, initial consult | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+ DCIS" | 病理 | ✓ | "at least DCIS"正确提取 |
| Stage_of_Cancer | "Not mentioned in note" | - | P2 | A/P 明确写 "stage is not yet final, pending definitive breast surg"，应提取此信息 |
| Metastasis | "No" | A/P | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 活检+MRI+体检 | 病理段 | ✓ | 全面 |
| current_meds | "" | - | ✓ | 无肿瘤用药 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not yet on treatment" | A/P | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | A/P 讨论了 XRT+/-endocrine 作为条件性计划，应有内容 |
| medication_plan | "If DCIS only → XRT +/- endocrine" | A/P | ✓ | 正确反映条件性计划 |
| therapy_plan | 同上+invasive 场景 | A/P | ✓ | - |
| radiotherapy_plan | "XRT if DCIS only" | A/P | ✓ | - |
| procedure_plan | "breast surg" | A/P | ✓ | - |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral | 全 None | - | ✓ | - |
| follow up | "RTC after surg completed" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 1×P2 — 干净提取
- P1: Medication_Plan_chatgpt 空（系统性问题）
- P2: Stage 应反映"pending surgery"而非"Not mentioned"

---

### Row 48 (coral_idx=188)

**患者概况**: 50F，左乳 IDC 伴活检证实腋窝 LN 转移。ER+(100%)/PR+(100%)/HER2-(IHC 2+, FISH 1.4)。Oncotype 11（低风险）。PET-CT 无远处转移（T7-L1 骨髓摄取不确定→脊柱 MRI 清）。可能 Stage II。计划 01/06/17 左乳切除术。辅助内分泌（tamoxifen，需评估血栓风险）。有代理决策人。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | A/P | ✓ | 初诊 |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | - | ✓ | 体检详细 |
| summary | 50F L breast CA + LN+ | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | A/P | ✓ | 含 FISH ratio |
| Stage_of_Cancer | "Stage II" | "likely stage 2 disease" | ✓ | - |
| Metastasis | "No" | "likely stage 2" | **P1** | 有活检证实的腋窝 LN 转移（"metastatic carcinoma of mammary origin, LN largely replaced"），应为 Yes (regional) |
| Distant Metastasis | "No" | 同上 | ✓ | PET-CT + spine MRI 清 |
| lab_summary | "Hgb 13.0, plt 287" | 检验段 | P2 | 漏 WBC 7.5 |
| findings | 体检+MRI+PET+活检 | 多来源 | ✓ | 全面 |
| current_meds | "" | - | ✓ | - |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | 手术计划 | ✓ | - |
| response_assessment | "Not yet on treatment" | 手术计划 | ✓ | - |
| Medication_Plan_chatgpt | 有内容(hormonal+surgery) | A/P | P2 | "surgery"不在指定类别(chemo/hormonal/bone/RT) |
| medication_plan | "tamoxifen + thrombophilia assessment" | A/P | ✓ | - |
| therapy_plan | "adjuvant endocrine + XRT discussed" | A/P | ✓ | - |
| radiotherapy_plan | "XRT discussed for LN+ disease" | A/P | ✓ | - |
| procedure_plan | "L mastectomy" | A/P | ✓ | - |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | Oncotype 已完成 |
| Referral-Specialty | "" (空字符串) | - | P2 | 应为 "None" 而非空字符串 |
| follow up | "post-op follow up" | A/P | ✓ | - |
| Advance care | 有代理决策人(配偶) | 原文 | ✓ | 完整准确 |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2 — 较干净
- P1: Metastasis 应为 Yes（活检证实腋窝 LN 转移）
- P2: lab_summary 漏 WBC、chatgpt 格式分类错误、Referral 空字符串

---

### Row 49 (coral_idx=189)

**患者概况**: 58F，de novo 转移性 IDC（肺/LN/肝/骨），HR+/HER2-。Stage IV (T2N1M1)。治疗史：AC x4→tamoxifen+lupron→进展→letrozole+lupron+ibrance（2014起）。2015 放疗（骨盆/胸骨），2019 乳房切除+XRT。PMS2 致病突变。Dec 2021 影像示控制良好。考虑乳房切除。Video visit second opinion。Code: full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | second opinion 初诊 |
| second opinion | "yes" | "see me prn" | ✓ | - |
| in-person | "Televisit" | telehealth 声明 | ✓ | - |
| summary | "second opinion regarding lumpectomy" | A/P | P2 | 措辞不准：是考虑 mastectomy，非 regarding lumpectomy |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | HPI | ✓ | - |
| Stage_of_Cancer | "Originally Stage IV, now Stage IV" | HPI | P2 | 冗余，Stage IV 不会变 |
| Metastasis | "Yes (lung, LN, liver, bone)" | HPI | ✓ | - |
| Distant Metastasis | "Yes" | HPI | ✓ | - |
| lab_summary | CMP from 11/25/2020 | 检验段 | **P1** | Sodium "136" 实为参考范围下限（值被 redacted *****），幻觉 |
| findings | "disease under good control, IDC+DCIS" | A/P | ✓ | - |
| current_meds | "ibrance, xgeva, letrozole" | HPI | ✓ | lupron 药物列表标注"not taking"，合理排除 |
| recent_changes | "" | - | ✓ | - |
| supportive_meds | "DENOSUMAB (XGEVA)" | 药物列表 | P2 | Xgeva 用于骨转移=bone therapy 非 supportive；且与 current_meds 重复 |
| goals_of_treatment | "palliative" | Stage IV | ✓ | - |
| response_assessment | "disease under good control" | A/P | ✓ | - |
| Medication_Plan_chatgpt | 有内容(混合) | A/P | P2 | 当前/过去治疗混在一起，无明确分类 |
| medication_plan | "tamoxifen, lupron, letrozole, ibrance" | A/P | **P1** | tamoxifen 2014 年因进展已停，非当前用药 |
| therapy_plan | "letrozole+ibrance started 2014-2015" | A/P | P2 | 仅复述历史，非当前/未来计划 |
| radiotherapy_plan | "None" | - | ✓ | 所有放疗均过去，正确 |
| procedure_plan | "Considering mastectomy" | A/P | ✓ | - |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "Referral to genetics for PMS2" | A/P | P2 | 这是遗传咨询转诊非新检测；PMS2 已发现 |
| Referral-Genetics | "Referral to genetics for PMS2" | A/P+Diagnosis | ✓ | 正确捕获遗传咨询转诊 |
| follow up | "prn" | A/P | ✓ | - |
| Advance care | "full code." | 原文 | ✓ | - |

**白名单笔记**:
- [ ] oncology_drugs.txt 应含 Xgeva/denosumab（骨转移 bone therapy，此例再次出现在 supportive_meds）

**本行总结**: 0×P0, 2×P1, 5×P2
- P1: lab_summary Sodium 值从参考范围幻觉、medication_plan 含已停 tamoxifen
- P2: summary 措辞、Stage 冗余、Xgeva 分类+重复、chatgpt 混合历史、therapy_plan 仅历史、genetic_testing_plan 混淆转诊与检测

---

## 前50行模式回顾 (Rows 40-49)

### 本批统计 (Rows 40-49)
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|---------|
| 40 | 180 | 0 | 3 | 2 | Patient type 错(Follow up→New)、Stage 空、Metastasis "Not sure" |
| 41 | 181 | 0 | 1 | 1 | Type_of_Cancer 缺 HER2- |
| 42 | 182 | 0 | 2 | 1 | Lab_Plan 漏 pre-chemo draw、Medication_Plan_chatgpt 空 |
| 43 | 183 | 0 | 1 | 3 | response_assessment 漏术后病理评估 |
| 44 | 184 | 0 | 0 | 0 | 完美 |
| 45 | 185 | 0 | 6 | 4 | Stage 幻觉、Metastasis "Not sure"、supportive_meds 漏、response_assessment 错 |
| 46 | 186 | 0 | 0 | 3 | DCIS 处理得当，仅小问题 |
| 47 | 187 | 0 | 1 | 1 | Medication_Plan_chatgpt 空 |
| 48 | 188 | 0 | 1 | 3 | Metastasis 漏 LN 转移 |
| 49 | 189 | 0 | 2 | 5 | lab 值从参考范围幻觉、medication_plan 含已停药 |
| **合计** | | **0** | **17** | **23** | |

### 累计统计 (Rows 0-49)
| 批次 | P0 | P1 | P2 | 总行数 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 10 |
| Rows 10-19 | 1 | 25 | 29 | 10 |
| Rows 20-29 | 0 | 15 | 21 | 10 |
| Rows 30-39 | 0 | 12 | 18 | 10 |
| Rows 40-49 | 0 | 17 | 23 | 10 |
| **总计** | **2** | **91** | **118** | **50** |

### 50行系统性问题模式

**高频 P1 问题**（按频率排序）：
1. **Medication_Plan_chatgpt 空**（~30行/50）— 系统性最严重，复杂 JSON schema 导致
2. **Metastasis/Distant Metastasis 错误**（~10行）— "Not sure" 或 "No" 当有明确证据时
3. **response_assessment 措辞/内容错误**（~8行）— 新辅助后说"not on treatment"、混淆残留与复发
4. **Stage_of_Cancer 幻觉或遗漏**（~7行）— 编造未提及的分期、或可推断但写"Not mentioned"
5. **current_meds 错误**（~6行）— 包含已停药、遗漏当前药
6. **supportive_meds 漏/错**（~6行）— 漏 pain meds、含 oncology drugs(骨/激素)
7. **Lab_Plan 遗漏**（~5行）— pre-chemo labs、定期复查被漏
8. **Referral 遗漏**（~5行）— PT、dental、genetics（A/P 外的转诊被漏）

**高频 P2 问题**：
1. **Medication_Plan_chatgpt 内容不完整**（有内容但缺条目）
2. **Stage 措辞问题**（"Originally..."、冗余表达）
3. **Xgeva/denosumab 分类**：bone therapy 被放入 supportive_meds（4次出现）
4. **therapy_plan 仅复述历史**而非当前/未来计划
5. **genetic_testing_plan 与 Referral-Genetics 混淆**

**新发现（Rows 40-49）**：
- Row 49: lab_summary 从参考范围提取值（Sodium *****  136-145 → 提取为 "Sodium 136"）— 新类型错误
- Row 45: 复杂合并症笔记（结节病+肾动脉瘤+贫血）导致多处提取不精确
- Row 44: 证明简短聚焦的笔记（second opinion video visit）提取质量最高
- DCIS 病例（Rows 46-47）处理得当，目标识别正确（risk reduction/curative）

**Pipeline 架构限制**：
- plan_extraction 只看 A/P 段 → HPI 中的转诊被系统性遗漏（PT、social work 等）
- Referral prompt 指示"搜索整个笔记"但实际只能看到 A/P

---

### Row 50 (coral_idx=190)

**患者概况**: 化疗教育 visit（RN），非 MD 临床笔记。Gemzar/[REDACTED]/[REDACTED] 方案教育。Zoom visit。无 HPI/体检/A&P。几乎全是副作用模板和自我照护提示。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | RN 教育 visit，类型模糊 |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | zoom meeting | ✓ | - |
| summary | "Education visit with RN" | 原文 | ✓ | 准确 |
| Type_of_Cancer | "Not mentioned" | - | ✓ | 癌症类型被 redacted |
| Stage_of_Cancer | "Not mentioned" | - | ✓ | - |
| Metastasis | "Not sure" | - | ✓ | 信息不足 |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | "No new clinical findings" | - | ✓ | - |
| current_meds | "" | - | ✓ | 方案尚未开始 |
| supportive_meds | "Zofran, Compazine" | 原文(anti-emetics 段) | P2 | 漏 Ativan（原文列三种：Zofran/Ativan/Compazine） |
| goals_of_treatment | "curative" | - | P2 | 笔记无治疗目标信息，无法判定 curative，属猜测 |
| response_assessment | "Not mentioned" | - | ✓ | - |
| Medication_Plan_chatgpt | "" | - | ✓ | 教育 visit 无 A/P |
| 其他 plan 字段 | 均空/None | - | ✓ | 非 MD encounter，合理 |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2 — 异常笔记类型（RN 化疗教育），模型处理合理
- P2: supportive_meds 漏 Ativan、goals_of_treatment 无依据推测 curative

---

### Row 51 (coral_idx=191)

**患者概况**: 35F，左乳 ER+/PR+/HER2- IDC，Ki-67 <10%。s/p 左乳部分切除+SLN→1.7cm grade II IDC，SLN 微转移(0.18cm)伴轻微结外扩展，切缘阴性。初诊 med onc 讨论辅助治疗。计划：AI+卵巢抑制(Zoladex)+先行卵子冻存→staging CT CAP+bone scan→Oncotype。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | 初诊声明 | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 体检 | ✓ | - |
| summary | 35F IDC initial consult | HPI | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 病理 | ✓ | - |
| Stage_of_Cancer | "Not mentioned" | - | P2 | staging CT 已下单，应为"Pending staging workup" |
| Metastasis | "No" | - | P2 | SLN 有微转移(pN1mi, 0.18cm)，边界情况 |
| Distant Metastasis | "No" | - | ✓ | staging 待做但无远处转移证据 |
| lab_summary | "Preg test Negative" | 检验段 | ✓ | - |
| findings | 病理+体检 | 病理段 | ✓ | 全面 |
| current_meds | "" | - | ✓ | 无肿瘤用药 |
| supportive_meds | "ondansetron 8mg" | 药物列表 | ✓ | - |
| goals_of_treatment | "curative" | "locoregional...curable" | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | 有 AI+ovarian suppression+Zoladex 计划 |
| medication_plan | "[REDACTED]+Zoladex" | Plan | ✓ | 尽管 redacted 仍捕获 Zoladex |
| therapy_plan | "Zoladex prior auth" | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| procedure_plan | "fertility preservation" | Plan | ✓ | - |
| imaging_plan | "CT CAP + bone scan" | Plan | ✓ | - |
| lab_plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "[REDACTED] for tumor biology" | A/P 讨论 | ✓ | 应为 Oncotype DX |
| Referral-Others | "None" | - | **P1** | 原文 "referred to reproductive health"，转诊被遗漏 |
| follow up | "3 weeks" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 2×P2
- P1: Medication_Plan_chatgpt 空、Referral 漏 reproductive health 转诊
- P2: Stage "Not mentioned"(pending staging)、Metastasis 漏 SLN 微转移

---

### Row 52 (coral_idx=192)

**患者概况**: 59F，左乳 IDC 伴神经内分泌分化，4.5cm grade 3。ER+(>95%)/PR+(30%)/HER2+（IHC 异质 2+/3+, FISH 4.9X）。Ki-67=25%。s/p 左乳切除+SLN(1/2+, 6mm met)。DCIS 亦 HER2+。初诊 UCSF 讨论辅助治疗。推荐 AC/THP 或 TCHP→Arimidex 10yr→neratinib year2→RT→bisphosphonate。Genetics counseling offered。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初诊 |
| second opinion | "no" | 讨论段 | ✓ | - |
| in-person | "in-person" | 体检 | ✓ | - |
| summary | "Stage II/III, ER/PR+/HER2+, post-lumpectomy" | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2+ IDC with NE diff + DCIS" | 病理 | ✓ | 完整，含 HER2 转换和神经内分泌分化 |
| Stage_of_Cancer | "Stage II/III" | A/P | ✓ | 反映不确定性（待 axillary dissection） |
| Metastasis | "No" | 风险评估段 | **P1** | SLN+ (1/2, 6mm met) = 确认区域转移 |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 体检（seroma, 无mass/LAD） | 体检段 | P2 | 漏关键病理：4.5cm/grade 3/HER2+/SLN+ |
| current_meds | "" | "Meds - none" | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | 讨论段 | ✓ | 辅助治疗，curative intent |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | 4项全有(chemo/hormonal/bone/RT) | A/P | ✓ | **优秀提取**，副作用全面 |
| medication_plan | AC/THP+TCHP+Arimidex+neratinib+bone | A/P | ✓ | 全面 |
| therapy_plan | 同上详细 | A/P | ✓ | - |
| radiotherapy_plan | "adjuvant RT, referral after chemo" | A/P | ✓ | - |
| procedure_plan | "No procedures planned." | - | ✓ | axillary dissection 讨论但未确定 |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| genetic_testing_plan | "referral to genetic counseling" | A/P | P2 | 是遗传咨询转诊非检测；已正确放入 Referral-Genetics |
| Referral-Genetics | "genetic counseling referral" | A/P | ✓ | - |
| Referral-Specialty | "RT referral" | A/P | ✓ | - |
| follow up | "next couple of weeks, will get back" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2 — Medication_Plan_chatgpt 难得完整（4项全有）
- P1: Metastasis "No" 但 SLN+(6mm met)
- P2: findings 漏病理关键信息、genetic_testing_plan 混淆转诊与检测

---

### Row 53 (coral_idx=193)

**患者概况**: 39F，绝经前，oligometastatic HR+/HER2- IDC 左乳+T6骨转移，BRCA2 突变。s/p 新辅助 AC/T→RT T6→bilateral mastectomy+left ALND。当前：leuprolide+letrozole+zoledronic acid。Plan: 术后放疗→palbociclib→PET/CT 3-4m。Regular f/u visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | follow up 段 | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | "45 min face-to-face" | ✓ | - |
| summary | oligometastatic IDC, BRCA2, f/u | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | ID line | P2 | PR+ 从 "HR+" 推断合理但未明确提及 PR |
| Stage_of_Cancer | "Originally Stage IIA, now Stage IV" | - | **P1** | "Originally Stage IIA" 无来源；肿瘤 6.7cm(T3)+LN+(N1)=Stage IIIA 非 IIA，幻觉 |
| Metastasis | "Yes (to bone)" | "involving T6" | ✓ | - |
| Distant Metastasis | "Yes (to bone)" | 同上 | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 症状+体检 | 多段 | ✓ | - |
| current_meds | "leuprolide, letrozole, zoledronic acid" | A/P | ✓ | zoledronic acid 用于 T6 骨转移=bone therapy |
| recent_changes | "start palbociclib after radiation" | A/P | ✓ | - |
| supportive_meds | "oxycodone, zoledronic acid" | 药物列表+A/P | P2 | zoledronic acid 与 current_meds 重复列出 |
| goals_of_treatment | "palliative" | "not curable but treatable" | ✓ | - |
| response_assessment | "Not mentioned" | - | ✓ | 无新影像 |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | 有 hormonal/CDK4/6i/bone/RT 多项计划 |
| medication_plan | leuprolide+letrozole+palbo+zometa+Ca+VitD | A/P | ✓ | 全面 |
| therapy_plan | 同上+exercise | A/P | ✓ | - |
| radiotherapy_plan | "post-mastectomy radiation" | A/P | ✓ | - |
| procedure_plan | "No procedures planned." | - | ✓ | - |
| imaging_plan | "PET/CT in 3-4 months" | A/P | P2 | 漏 DEXA scan (re-ordered) |
| lab_plan | "No labs planned." | - | ✓ | - |
| Referral-Specialty | "radiation oncology" | A/P | ✓ | - |
| follow up | "4 weeks + PET/CT 3-4m" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 全文无 code status |

**白名单笔记**:
- [ ] oncology_drugs.txt 确认含 zoledronic acid（此例正确在 current_meds 但又出现在 supportive_meds）

**本行总结**: 0×P0, 2×P1, 3×P2
- P1: Stage "Originally Stage IIA" 幻觉（应≥IIIA）、Medication_Plan_chatgpt 空
- P2: PR+ 推断、zoledronic acid 双列、DEXA 漏

---

### Row 54 (coral_idx=194)

**患者概况**: 53F，绝经后，Stage I (T1N0M0) 左乳 IDC 5mm grade 2，ER+(>95%)/PR+(>95%)/HER2-(IHC 1+, FISH 1.4X)。s/p 左乳切除+SLN(0/2-)。当前接受左乳 RT（即将完成）。初诊讨论辅助内分泌治疗。推荐 Arimidex 1mg x5yr。患者犹豫中。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初诊 |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 体检 | ✓ | - |
| summary | Stage I, post-lumpectomy, adj hormone discussion | A/P | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC with focal DCIS" | 病理 | ✓ | 含 IHC/FISH 详细 |
| Stage_of_Cancer | "Stage I (T1N0M0)" | A/P | ✓ | 原文明确 |
| Metastasis | "No" | - | ✓ | LN 0/2 |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs" | - | ✓ | - |
| findings | 体检+病理 | 多段 | ✓ | 病理和体检均含 |
| current_meds | "" | - | ✓ | 仅 Valtrex PRN(非肿瘤) |
| goals_of_treatment | "curative" | 风险讨论 | ✓ | - |
| Medication_Plan_chatgpt | 有内容(hormonal+bone) | A/P | ✓ | 副作用完整 |
| medication_plan | "Arimidex 1mg x5yr + bone agents" | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | P2 | 患者正在接受 RT（HPI 提到），但 A/P 未提及，pipeline 限制 |
| imaging_plan | "DEXA scanning" | A/P | P2 | 原文为条件建议("can be monitored")非确定计划 |
| follow up | "6 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2 — 干净提取
- P2: radiotherapy_plan 漏当前 RT（pipeline 限制）、imaging_plan 条件 DEXA 过度确定

---

### Row 55 (coral_idx=195)

**患者概况**: 56F，绝经后，Stage IB 左乳 TNBC (ER-/PR-/HER2-)，Ki-67=90%。家族史：母亲+同父异母姐妹乳腺癌。Video visit 初诊。推荐新辅助 ddAC→weekly taxol，或先手术若 MRI 证实 Stage I。需 port、echo、MRI、genetics consult。Code: full code。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | Video Consult | ✓ | - |
| summary | Stage I TNBC, neoadjuvant discussion | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- TNBC" | 病理 | ✓ | - |
| Stage_of_Cancer | "Stage IB" | Staging 段 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs" | - | ✓ | - |
| findings | Ki-67 90%, MRI recommended | A/P | ✓ | video visit 体检有限 |
| current_meds | "" | - | ✓ | 无用药 |
| goals_of_treatment | "curative" | Stage I | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | 有 ddAC+taxol 化疗计划 |
| medication_plan | "neoadjuvant ddAC + weekly taxol" | A/P | ✓ | - |
| therapy_plan | 化疗+放疗讨论 | A/P | ✓ | 全面 |
| radiotherapy_plan | "discussed, conditional on pathology" | A/P | ✓ | - |
| procedure_plan | "port + surgery pending" | A/P | ✓ | - |
| imaging_plan | "MRI bilateral breasts" | A/P | P2 | 漏 echo（"She will need an echo"） |
| genetic_testing_plan | "Genetics consult" | A/P | P2 | 实为遗传咨询转诊非检测计划 |
| Referral-Genetics | "Genetics consult" | A/P | ✓ | - |
| follow up | "None" | - | ✓ | 无具体日期，需尽快开始化疗 |
| Advance care | "Full code." | 原文 | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: Medication_Plan_chatgpt 空
- P2: imaging_plan 漏 echo、genetic_testing_plan 混淆转诊与检测

---

### Row 56 (coral_idx=196)

**患者概况**: 59F，左乳局部晚期乳腺癌。初始 HER2+→TCH+P x6 新辅助→手术后病理 HER2-（TNBC）！原始活检复查也 HER2-。术后 AC x4。残留 3.7cm，0/6 nodes。Second opinion 关于 HER2 争议和后续治疗。计划 XRT。PMH: HTN, Crohn's。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | second opinion |
| second opinion | "yes" | "2nd opinion" | ✓ | - |
| in-person | "in-person" | 详细体检 | ✓ | - |
| summary | TNBC, 2nd opinion re HER2 status | A/P | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- TNBC" | 病理 | P2 | 未提及 HER2 状态从+变-的关键争议 |
| Stage_of_Cancer | "Locally advanced" | CC | ✓ | 无具体 TNM |
| Metastasis | "No" | 0/6 nodes | ✓ | - |
| findings | 残留肿瘤+体检 | 多段 | ✓ | - |
| current_meds | "" | - | ✓ | 仅非肿瘤药(Carvedilol/Lisinopril/Asacol) |
| recent_changes | "Dose reduction 25% after C1" | HPI | P2 | 已完成的化疗剂量调整=过去事件 |
| supportive_meds | "benadryl, codeine" | ALL 段 | **P1** | 这是**过敏原**不是药物！"ALL: benadryl, codeine" |
| goals_of_treatment | "curative" | 局部晚期 | ✓ | - |
| response_assessment | "Not mentioned" | - | **P1** | 新辅助后残留 3.7cm = partial pathologic response，应记录 |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | 有 XRT 和条件性 trastuzumab 计划 |
| medication_plan | null | - | P2 | 空值；条件性 trastuzumab ("if HER2+, resume") 应提及 |
| therapy_plan | "XRT + conditional trastuzumab" | A/P | ✓ | - |
| radiotherapy_plan | "XRT scheduled" | A/P | ✓ | - |
| procedure_plan | "which pt is scheduled to receive" | - | **P1** | 乱码：XRT 不是 procedure，是放疗；无 planned procedures |
| imaging_plan | "No imaging planned." | - | ✓ | - |
| genetic_testing_plan | "genetic counseling and testing" | A/P | ✓ | - |
| Referral-Genetics | "genetic counseling" | A/P | ✓ | - |
| Referral-Specialty | "XRT" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 4×P1, 3×P2 — HER2 状态争议案例
- P1: supportive_meds 把过敏原当药物（benadryl/codeine）、response_assessment 漏新辅助后残留评估、chatgpt 空、procedure_plan 乱码
- P2: Type_of_Cancer 缺 HER2 争议上下文、recent_changes 为过去事件、medication_plan null

---

### Row 57 (coral_idx=197)

**患者概况**: 60F，Stage IIb (T2N1M0) IDC 左乳，ER+/PR+/HER2-，Ki-67>30%。s/p TC x6→左乳再切除→XRT。Letrozole 自 10/2017 起（~1.5yr）。骨密度：脊柱骨质疏松/股骨骨量减少。Follow up，讨论骨药和 CDK4/6i。Shared visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | ID line | ✓ | - |
| Stage_of_Cancer | "Stage IIb (T2N1M0)" | ID line | ✓ | 原文明确 |
| Metastasis | "No" | - | ✓ | 治疗后无复发证据 |
| current_meds | "letrozole" | Plan | ✓ | - |
| goals_of_treatment | "curative" | 辅助治疗 | ✓ | - |
| response_assessment | "tolerating letrozole well" | A/P | P2 | 描述副作用耐受而非疾病状态/治疗响应 |
| Medication_Plan_chatgpt | {} (空) | - | **P1** | 有 letrozole 继续 + zoledronate 授权 |
| medication_plan | "continue letrozole + zoledronate auth" | Plan | ✓ | - |
| therapy_plan | 同上 | Plan | P2 | 漏 CDK4/6i 讨论（"discussed CDK4/6i as waiting on adjuvant data"） |
| imaging_plan | "DEXA + mammogram" | Plan | ✓ | - |
| follow up | "6 months" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: Medication_Plan_chatgpt 空
- P2: response_assessment 写副作用非疾病状态、therapy_plan 漏 CDK4/6i 讨论

---

### Row 58 (coral_idx=198)

**患者概况**: 52F，Stage I 右乳 IDC，ER+/PR+/HER2-(equivocal IHC, FISH neg)，MammaPrint High Risk。s/p 切除+SLN(0/5)→TC x3+Abraxane/Cytoxan x1→XRT→tamoxifen(停)→letrozole(停,关节痛)→切换 exemestane。Depression on Pristiq 75mg，考虑转 duloxetine。NED。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC + DCIS" | 病理 | ✓ | - |
| Stage_of_Cancer | "Stage I" | A/P | ✓ | - |
| Metastasis | "No" | 0/5 nodes | ✓ | - |
| lab_summary | VitD/FSH/E2/FT4/TSH | 检验段 | ✓ | 完整 |
| findings | NED on exam | A/P | ✓ | - |
| current_meds | "exemestane, letrozole" | 药物列表 | **P1** | 正在切换：停 letrozole→2-3 周后 start exemestane，不应同时列为 current |
| supportive_meds | "" | - | **P1** | 漏 desvenlafaxine/Pristiq 75mg（抗抑郁=psychiatry medication），gabapentin（neuropathy） |
| goals_of_treatment | "curative" | 辅助治疗 | ✓ | - |
| response_assessment | "NED" | A/P | ✓ | - |
| Medication_Plan_chatgpt | 有(hormonal+psych) | A/P | ✓ | 含 duloxetine 讨论 |
| medication_plan | "stop letrozole→exemestane + duloxetine考虑" | A/P | ✓ | 全面 |
| therapy_plan | "letrozole→exemestane, 5yr" | A/P | ✓ | - |
| imaging_plan | "mammogram July, alt MRI, q6mo exam" | A/P | ✓ | - |
| Referral-Specialty | "Psychiatry" | A/P | ✓ | Pristiq→duloxetine 转换 |
| follow up | "6 months" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 0×P2
- P1: current_meds 同时列 letrozole+exemestane（正在切换非同时服用）、supportive_meds 漏 Pristiq/gabapentin

---

### Row 59 (coral_idx=199)

**患者概况**: 65F，绝经后，左乳原始活检 DCIS→切除后发现 IDC 0.7cm grade 2，ER+(>95%)/PR+(10%)/HER2-，Ki-67~20%。pT1bNX（患者选择不做进一步手术）。Tumor board 讨论后选择 XRT 无额外手术。Med onc video consult。计划：endocrine therapy 5-10yr + Oncotype Dx + Rad Onc。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | med onc 初诊 |
| in-person | "Televisit" | telehealth 声明 | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 病理 | ✓ | - |
| Stage_of_Cancer | "pT1bNX" | 推断 | ✓ | 0.7cm=T1b, 无腋窝手术=NX |
| Metastasis | "No" | - | ✓ | NX 但无证据 |
| findings | 病理+体检 | 多段 | P2 | PR 写 70%（来自 DCIS 活检），A/P 中 IDC 的 PR=10% |
| current_meds | "" | - | ✓ | 无用药 |
| goals_of_treatment | "curative" | 早期 | ✓ | - |
| Medication_Plan_chatgpt | 有内容(endocrine) | A/P | ✓ | 有效内容！ |
| medication_plan | "adjuvant endocrine 5-10yr" | A/P | ✓ | - |
| radiotherapy_plan | "see Rad Onc this week" | A/P | ✓ | - |
| genetic_testing_plan | "Oncotype Dx" | A/P | ✓ | - |
| Referral-Specialty | "Rad Onc consult" | A/P | ✓ | - |
| follow up | "3 wks after Oncotype" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 1×P2 — 非常干净
- P2: findings 中 PR 用了 DCIS 活检值(70%)而非 IDC 手术病理值(10%)

---

## 前60行模式回顾 (Rows 50-59)

### 本批统计 (Rows 50-59)
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|---------|
| 50 | 190 | 0 | 0 | 2 | RN 化疗教育 visit（异常笔记类型） |
| 51 | 191 | 0 | 2 | 2 | chatgpt 空、Referral 漏 reproductive health |
| 52 | 192 | 0 | 1 | 2 | Metastasis 漏 SLN+，chatgpt 难得完整(4项) |
| 53 | 193 | 0 | 2 | 3 | Stage 幻觉("IIA"→应≥IIIA)、chatgpt 空 |
| 54 | 194 | 0 | 0 | 2 | 干净，RT info 在 HPI 不在 A/P |
| 55 | 195 | 0 | 1 | 2 | chatgpt 空、echo 漏 |
| 56 | 196 | 0 | 4 | 3 | 过敏当药物！response_assessment 漏、procedure 乱码 |
| 57 | 197 | 0 | 1 | 2 | chatgpt 空、response 写副作用 |
| 58 | 198 | 0 | 2 | 0 | current_meds 同时列切换药物、supportive 漏 Pristiq |
| 59 | 199 | 0 | 0 | 1 | 非常干净 |
| **合计** | | **0** | **13** | **19** | |

### 累计统计 (Rows 0-59)
| 批次 | P0 | P1 | P2 | 总行数 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 10 |
| Rows 10-19 | 1 | 25 | 29 | 10 |
| Rows 20-29 | 0 | 15 | 21 | 10 |
| Rows 30-39 | 0 | 12 | 18 | 10 |
| Rows 40-49 | 0 | 17 | 23 | 10 |
| Rows 50-59 | 0 | 13 | 19 | 10 |
| **总计** | **2** | **104** | **137** | **60** |

### 60行趋势观察
- P1 均值：~1.7/行，前20行 ~2.4/行 → 后40行 ~1.4/行，有下降趋势
- P0 仅出现在前20行，后40行为零
- **Medication_Plan_chatgpt 空** 仍是最高频 P1（本批 6/10 行为空）
- **新发现模式**：
  - Row 56: **过敏原被当作 supportive_meds**（benadryl, codeine 来自 ALL/Allergies 段）→ 新型严重错误
  - Row 53/45: Stage "Originally Stage IIA" 反复出现幻觉，模型倾向猜测 Stage IIA
  - Row 58: current_meds 同时列出正在切换的两种药物（letrozole→exemestane）
  - Row 54/Row 53: radiotherapy info 仅在 HPI 而 A/P 未提及 → pipeline 限制导致遗漏

---

### Row 60 (coral_idx=200)

**患者概况**: 43F，绝经前，左乳 IDC，ER+(100)/PR+(100)/HER2-(IHC 1+)，grade 2，≥11mm。筛查发现。MRI 示 0630 位 1.5cm + 0900 位可疑灶（复查活检阴性）。CT 无远处转移。Invitae genetic testing 阴性。计划 04/12 切除+IORT+重建。术后 Oncotype Dx→化疗决策→endocrine therapy。Video visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初诊 |
| in-person | "Televisit" | Zoom | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2-(not tested) IDC" | 病理 | **P1** | HER2 已检测(IHC 1+=negative)，"not tested"错误 |
| Stage_of_Cancer | "Not mentioned" | - | ✓ | 待手术后分期 |
| findings | 病理+MRI+CT+体检 | 多段 | ✓ | 全面 |
| current_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | "early stage" | ✓ | - |
| Medication_Plan_chatgpt | 有内容(endocrine) | A/P | ✓ | - |
| medication_plan | "tamoxifen vs OS + AI" | A/P | ✓ | - |
| radiotherapy_plan | "IORT, no post-op RT" | A/P | ✓ | 准确 |
| procedure_plan | "lumpectomy+IORT+reconstruction 04/12" | A/P | ✓ | - |
| genetic_testing_plan | "None planned." | - | **P1** | A/P 明确 "will likely need Oncotype Dx after surgery"，被遗漏 |
| follow up | "after surgery + pathology" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 0×P2
- P1: HER2 标注"not tested"（实际 IHC 1+ 已测）、genetic_testing_plan 漏 Oncotype Dx

---

### Row 61 (coral_idx=201)

**患者概况**: 44F，绝经前，右乳 IDC pT1aN0(sn)M0，0.2cm grade 1，ER+(>95%)/PR+(>95%)/HER2-(2+,FISH neg)，Ki-67<5%。伴 DCIS 0.5cm。s/p 切除+SLN(0/1-)。Myriad 34 panel 阴性。Oncotype 已 ordered。不推荐化疗。推荐辅助 endocrine therapy。Video visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初诊 |
| in-person | "Televisit" | Zoom | ✓ | - |
| Type_of_Cancer | 完整含 IHC/FISH/Ki-67 | A/P | ✓ | 优秀 |
| Stage_of_Cancer | "pT1aN0(sn)M0" | A/P | ✓ | 原文明确 |
| Metastasis | "No" | 0/1 nodes | ✓ | - |
| findings | 病理+MRI+US | 多段 | ✓ | 全面 |
| current_meds | "" | - | ✓ | 仅保健品 |
| goals_of_treatment | "curative" | 低风险 | ✓ | - |
| Medication_Plan_chatgpt | 有内容(hormonal) | RECS | ✓ | - |
| medication_plan | "endocrine therapy options" | RECS | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | RECS 未提及 |
| genetic_testing_plan | "None planned." | - | P2 | Oncotype 已 ordered/attempted，虽非本次新 order 但仍活跃 |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 1×P2 — 非常干净
- P2: Oncotype 已 ordered 但 genetic_testing_plan 未提及

---

### Row 62 (coral_idx=202)

**患者概况**: 49F，左乳局部晚期 ER+/PR-/HER2(2+ IHC, FISH-) IDC。新辅助 DD-AC + paclitaxel→Abraxane → 左乳部分切除+SLNB → 再切除 → letrozole → XRT。此次为远程第二意见会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 第二意见=new |
| second opinion | "yes" | 讨论研究结果 | P2 | 归因引用错误，应引"requested a second opinion" |
| in-person | "Televisit" | - | ✓ | 原文明确 telehealth |
| Type_of_Cancer | "ER+/PR-/[REDACTED] negative IDC" | A/P | ✓ | 符合手术病理 |
| Stage_of_Cancer | "Stage IIIA (S/P neoadjuvant chemo)" | "locally advanced" | ✓ | 合理推断：cT2N1+→IIIA |
| Metastasis | "No" | A/P | ✓ | PET/CT 无远处转移 |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | 原文"No results found" |
| findings | 详细：病理+影像+PE | 多段 | ✓ | 非常全面 |
| current_meds | "letrozole" | 开始日期引用 | ✓ | 正确，另两药(sleep med/ondansetron)合理排除 |
| recent_changes | "" (空) | - | ✓ | 第二意见，未实际变更 |
| supportive_meds | "ondansetron 8mg" | - | ✓ | 止吐药=支持性 |
| goals_of_treatment | "curative" | "reduce recurrence risk" | ✓ | 局部晚期辅助治疗 |
| response_assessment | "MRI dramatic response...near total resolution" | MRI 09/02 | P1 | **仅引影像反应，漏掉手术病理：残留3.8cm+阳性切缘+多LN阳性。医生明确说"disappointing"** |
| Medication_Plan_chatgpt | 有内容(hormonal+bone) | - | ✓ | 不为空 |
| medication_plan | letrozole+estradiol/FSH监测+OS/卵巢切除+DEXA+abemaciclib | A/P | ✓ | 全面 |
| therapy_plan | letrozole+OS/卵巢切除+abemaciclib讨论 | A/P | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | XRT已完成 |
| Procedure_Plan | "No procedures planned." | - | P2 | 有条件性卵巢切除讨论(若未绝经) |
| Imaging_Plan | "DEXA scan baseline" | A/P | ✓ | - |
| Lab_Plan | "estradiol and FSH q1-2m" | A/P | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | 未提及新基因检测 |
| Referral | 全"None" | - | ✓ | 无明确外转 |
| follow up | "Mychart沟通" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: response_assessment 仅引 MRI 影像反应（dramatic response），漏掉手术病理残留 3.8cm、阳性切缘、多 LN 阳性，医生明确评价"disappointing"
- P2: second_opinion 归因引用不准确
- P2: Procedure_Plan 漏条件性卵巢切除讨论

---

### Row 63 (coral_idx=203)

**患者概况**: 28F，新诊断左乳 IDC，HR+/HER2-，肿瘤 10.3cm on MRI，腋窝LN+，胸骨可疑骨转移（待活检）。正在接受 dd AC。远程第二意见会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "yes" | - | ✓ | - |
| in-person | "Televisit" | - | ✓ | Video consult/Zoom |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | staging form | ✓ | - |
| Stage_of_Cancer | "Originally Stage III-IV, now metastatic (Stage IV)" | A/P | P2 | 原文说"probably metastatic"，活检待做，stage 尚未确认 |
| Metastasis | "Yes, to the sternum" | - | P1 | **骨活检待做，原文"suspicious lesion"+"probably metastatic"，非确诊转移** |
| Distant Metastasis | "Yes, to the sternum" | - | P1 | 同上，应注明"suspected/pending biopsy" |
| lab_summary | 列出01/13/2019多项检验值 | - | P2 | 实验室数据为2年前(2019)，本次就诊2021，历史数据 |
| findings | 肿瘤大小+LN+sternal lesion | HPI | ✓ | 全面 |
| current_meds | "" (空) | - | P1 | **患者正在接受 dd AC (HPI明确"started on dd AC")** |
| recent_changes | "" (空) | - | ✓ | 第二意见，未做变更 |
| supportive_meds | dexamethasone+docusate+olanzapine+ondansetron+prochlorperazine | med list | ✓ | 合理 |
| goals_of_treatment | "palliative" | - | P2 | 医生明确说"good long term chance of control"并计划aggressive多模态治疗，oligometastatic intent更接近curative |
| response_assessment | "Not yet on treatment" | - | P1 | **错误：患者正在dd AC化疗("started on dd AC and tolerating okay")** |
| Medication_Plan_chatgpt | 有(chemo: taxol) | - | ✓ | - |
| medication_plan | "[REDACTED] and taxol planned" | A/P | P2 | 遗漏"if biopsy positive would add xgeva" |
| therapy_plan | 化疗+手术+放疗+sternum治疗 | A/P | ✓ | 全面 |
| radiotherapy_plan | "treatment to sternum planned" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | P1 | **A/P明确"Biopsy is planned for sternal lesion"(item 3)；手术也在计划中(item 4)** |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "No specific genetic tests" | - | ✓ | - |
| Referral | 全"None" | - | ✓ | 无明确外转 |
| follow up | "None" | - | P2 | A/P提到"check with insurance to change clinic"，可视为follow-up计划 |
| Advance care | "Full code." | 原文有 | ✓ | 在PE段上方明确标注 |

**白名单笔记**:
- [ ] Xgeva 应在 oncology_drugs.txt（此行讨论 bone met 用 Xgeva）

**本行总结**: 0×P0, 4×P1, 4×P2 — 复杂 oligometastatic 病例，多个字段出错
- P1: Metastasis/Distant Metastasis 将"probably/suspicious"标为确诊转移
- P1: current_meds 空，患者正在 dd AC
- P1: response_assessment 说"not yet on treatment"，实际正在化疗
- P1: Procedure_Plan 遗漏明确计划的骨活检和手术
- P2: Stage 漏"probably"限定词
- P2: goals_of_treatment "palliative" vs 医生 curative-intent oligomet 治疗
- P2: medication_plan 漏 Xgeva 条件计划
- P2: follow_up 空

---

### Row 64 (coral_idx=204)

**患者概况**: 48F，右乳 IDC，ER 弱阳 2%/PR 低阳 7%/HER2- (IHC 2+, FISH-)，Ki-67 36%。腋窝LN微转移(0.2mm)。新诊断，讨论新辅助化疗及 ISPY 临床试验。远程会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "no" | - | ✓ | 转诊非second opinion |
| in-person | "Televisit" | - | ✓ | - |
| Type_of_Cancer | 详细含ER%/PR%/HER2 FISH | HPI | ✓ | 优秀 |
| Stage_of_Cancer | "Locally advanced with axillary LN involvement" | A/P | ✓ | 反映原文描述 |
| Metastasis | 未见远处转移 | PET/CT | ✓ | - |
| findings | 详细影像+病理 | 多段 | ✓ | 全面 |
| current_meds | "" (空) | - | ✓ | 尚未开始化疗 |
| supportive_meds | "" (空) | - | ✓ | 仅有过敏药(fexofenadine/montelukast) |
| goals_of_treatment | "curative" | - | ✓ | 新辅助+手术+放疗 |
| response_assessment | "Not yet on treatment" | - | ✓ | 正确 |
| Medication_Plan_chatgpt | 有(chemo详细) | - | ✓ | - |
| medication_plan | 新辅助AC/T+ISPY选项 | Plan | ✓ | 详细 |
| therapy_plan | 化疗+手术+放疗+内分泌 | Plan | ✓ | 全面 |
| radiotherapy_plan | "post-operative radiation" | Plan | ✓ | - |
| Procedure_Plan | "port placement, research core biopsy, surgery" | Plan | ✓ | - |
| Imaging_Plan | "research core biopsy, research breast MRI" | Plan | P1 | **research core biopsy是procedure非imaging；遗漏TTE("***** order TTE")** |
| Lab_Plan | "research core biopsy, labs, genetic testing results" | Plan | P1 | **research core biopsy非lab；genetic testing属genetic_testing_plan** |
| genetic_testing_plan | "F/u genetic testing results" | Plan | P2 | 检测已做只是待结果，是否算"计划"有争议 |
| Referral | 全"None" | - | P2 | 遗漏 "Chemotherapy teaching session"（在Plan中明确列出） |
| follow up | "RTC 1-2 weeks to start chemo" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 原文无code status |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 2×P2
- P1: Imaging_Plan 包含 biopsy（应在 Procedure）且漏 TTE
- P1: Lab_Plan 包含 biopsy 和 genetic testing（均非 lab）
- P2: genetic_testing_plan 列已完成检测的待出结果
- P2: Referral 漏 chemo teaching session

---

### Row 65 (coral_idx=205)

**患者概况**: 53F，右乳 metaplastic carcinoma（高级别鳞状分化），ER 5-10%/PR 0%/HER2 0%，BRCA 阴性。CT 无远处转移，腋窝LN活检阴性。第二意见会诊（Investigational Therapeutics Program）。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | Initial consultation |
| second opinion | "yes" | - | ✓ | "referred for a second opinion" |
| in-person | "in-person" | - | ✓ | - |
| Type_of_Cancer | "ER 5-10%, PR 0%, HER2 0% metaplastic carcinoma" | - | ✓ | 准确 |
| Stage_of_Cancer | "Not mentioned in note" | - | P2 | 可推断 cT3N0M0≈Stage IIA，但原文确实未明确分期 |
| Metastasis | "No" | CT无远处 | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | 多项检验值(09/05/2019) | - | ✓ | 原文有这些实验室数据 |
| findings | 详细病理+影像 | 多段 | ✓ | 全面 |
| current_meds | "" (空) | - | ✓ | 药物列表为空(supplements marked "not taking") |
| supportive_meds | "" (空) | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | 新辅助+手术+放疗 |
| response_assessment | "Not yet on treatment" | - | ✓ | 尚未开始 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **讨论了pembrolizumab和新辅助化疗** |
| medication_plan | "No specific medications" | Plan | P1 | **pembrolizumab在Plan中明确提到，应被提取** |
| therapy_plan | 新辅助chemo+bilateral mastectomy+radiation+pembrolizumab讨论 | Plan | ✓ | 全面 |
| radiotherapy_plan | "adjuvant radiation discussed" | Plan | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | P2 | 患者"considering bilateral mastectomy"（讨论中非确定） |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | P2 | Plan提到"await completion of ***** testing"（Invitae germline待出结果） |
| Referral | 全"None" | - | ✓ | 无新转诊，返回referring physician |
| follow up | "after chemo to discuss next steps" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 2×P1, 3×P2
- P1: Medication_Plan_chatgpt 空
- P1: medication_plan 遗漏 pembrolizumab（Plan 中明确讨论 FDA 批准）
- P2: Stage 未推断 (cT3N0M0)
- P2: Procedure_Plan 漏 bilateral mastectomy 讨论
- P2: genetic_testing_plan 漏 pending Invitae germline testing

---

### Row 66 (coral_idx=206)

**患者概况**: 54F，新诊断左乳 TNBC (ER-/PR-/HER2-, Ki-67 59%)，grade 3 IDC ≥2.5cm。已完成 2 cycles dd AC → 并发症（MRSA脓肿、糖尿病失控、胸腔积液）。转 UCSF 第二意见。Plan: Abraxane weekly x12 → 可能恢复 dd AC。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | UCSF初诊 |
| second opinion | "yes" | - | ✓ | 外院转来 |
| in-person | "in-person" | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | 病理 | ✓ | triple negative |
| Stage_of_Cancer | "Clinical stage II-III" | A/P | ✓ | 原文如此 |
| Metastasis | "No" | - | P2 | PET/CT尚未做(Plan中ordered)，分期未完成，但临床无远处证据 |
| findings | 详细病理+临床变化 | 多段 | ✓ | - |
| current_meds | "" (空) | - | ✓ | 化疗已暂停>1月，med list为非肿瘤药 |
| recent_changes | "Switched to Abraxane weekly x12 + Levaquin" | Plan | ✓ | 准确 |
| supportive_meds | "Neupogen 480mcg..." | - | P2 | Neupogen是前2个cycle的支持药，非当前 |
| goals_of_treatment | "curative" | "cure this disease" | ✓ | 原文明确 |
| response_assessment | "responding: axillary LN from 8cm to 2-3cm, breast mass decreased" | A/P | ✓ | 准确 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan有详细方案：Abraxane 100mg/m2 weekly x12** |
| medication_plan | "Abraxane 100mg/m2 weekly x12 + growth factor + Levaquin" | Plan | ✓ | 详细 |
| therapy_plan | Abraxane+可能恢复dd AC | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | 未讨论 |
| Procedure_Plan | "No procedures planned." | - | ✓ | 无手术/活检计划 |
| Imaging_Plan | "PET/CT + diagnostic mammo/US" | Plan | ✓ | - |
| Lab_Plan | "CBC, renal, liver function before chemo" | Plan | ✓ | - |
| genetic_testing_plan | "genetic counselor to consider genetic tests" | Plan | ✓ | 合理 |
| Referral Genetics | "refer to genetics" | Plan | ✓ | - |
| Referral Others | "None" | - | P2 | 建议 follow up with surgeon (abscess) 和 contact diabetes physician |
| follow up | "start chemo 06/05/2011" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: Medication_Plan_chatgpt 空
- P2: Metastasis "No" 但 PET/CT 尚未做（分期未完成）
- P2: supportive_meds 列历史支持药（Neupogen from prior cycles）
- P2: Referral 漏 surgeon follow-up 和 diabetes physician

---

### Row 67 (coral_idx=207)

**患者概况**: 63F，多灶性右乳癌（*****+/*****+），BRCA mutation carrier，已完成 6 cycles TCHP（提示 HER2+）。影像完全缓解。来讨论手术方案（bilateral mastectomy vs lumpectomy）。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | header "New Visit" | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | - | P1 | **TCHP 是 HER2+ 专用方案(含trastuzumab+pertuzumab)，HER2 必须阳性。模型错误推断 HER2-** |
| Stage_of_Cancer | "Originally early stage" | HPI | P2 | 原文写"early stage"但多灶+axillary LN+IMN LN，实际≥Stage III |
| Metastasis | "Yes, axillary+IMN LN" | MRI | P1 | **腋窝和内乳淋巴结是区域(regional)转移，非远处转移。医生明确说"no evidence of metastatic presence"** |
| Distant Metastasis | "Yes, axillary+IMN LN" | - | P1 | **同上，与医生评估直接矛盾** |
| lab_summary | "No labs in note." | - | ✓ | 原文"Labs: reviewed today"但无具体值 |
| findings | 详细MRI多灶描述 | MRI | ✓ | - |
| current_meds | "" (空) | - | ✓ | TCHP已完成，无当前药物列表 |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "no lesions after chemo, no evidence of metastatic presence" | 结果段 | ✓ | 准确反映影像完全缓解 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **A/P有治疗讨论（虽药名被redact）** |
| medication_plan | "No specific medication plans" | - | ✓ | 药名被redact，模型无法提取 |
| therapy_plan | "radiation if mastectomy+residual or if lumpectomy" | A/P | ✓ | - |
| radiotherapy_plan | "post-mastectomy or post-lumpectomy radiation" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | P1 | **A/P明确讨论 bilateral mastectomy vs lumpectomy，医生"very recommendable"** |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| genetic_testing_plan | "sons should be tested..." | A/P | P2 | 这是家属基因咨询，非患者自身检测计划 |
| Referral Genetics | "refer to genetics" | sons testing | ✓ | - |
| follow up | "as needed" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- [ ] TCHP 应触发 POST-HER2-VERIFY → HER2+（trastuzumab+pertuzumab 均在 HER2 靶向药列表）

**本行总结**: 0×P0, 5×P1, 2×P2
- P1: Type_of_Cancer HER2- 错误（TCHP 方案证明 HER2+）
- P1: Metastasis/Distant Metastasis 将 regional LN 标为远处转移，与医生评估矛盾
- P1: Medication_Plan_chatgpt 空
- P1: Procedure_Plan 遗漏明确讨论的 bilateral mastectomy/lumpectomy
- P2: Stage "early stage" 与疾病范围不符
- P2: genetic_testing_plan 列家属检测而非患者检测

---

### Row 68 (coral_idx=208)

**患者概况**: 52F，新诊断右乳 ILC (invasive lobular carcinoma)，ER strongly+ (8/8)/PR+ (5/8)/HER2 equivocal→FISH neg，Ki-67 10%，低级别。腋窝LN+。Clinical stage T2N1Mx。讨论新辅助治疗方案和 ISPY 试验筛选。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "no" | - | ✓ | 转诊非second opinion |
| Type_of_Cancer | "ER+/PR+/HER2- ILC" | 病理 | ✓ | 与 biopsy 一致 |
| Stage_of_Cancer | "min clinical IIB, T2 N1 Mx" | A/P | ✓ | 原文如此 |
| Metastasis | "No" | - | P2 | Staging 写 Mx（未知），PET/CT 尚未做 |
| Distant Metastasis | "No" | - | P2 | 同上 |
| lab_summary | 多项(09/13/2015) | HPI | ✓ | 原文有 |
| findings | 详细病理+MRI | 多段 | ✓ | 全面 |
| current_meds | "" (空) | - | ✓ | 仅有 lorazepam(PRN焦虑) |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | 正确 |
| Medication_Plan_chatgpt | 有(hormonal) | - | ✓ | - |
| medication_plan | "AI hormonal therapy 5-10年 + possible neoadjuvant AI 4-6m" | A/P | ✓ | - |
| therapy_plan | 详细分 low-risk vs high-risk 路径 | A/P | ✓ | 全面 |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | P2 | 讨论了 ISPY 筛选需 repeat biopsy + 手术(BCT/mastectomy) |
| Imaging_Plan | "PET/CT + possible repeat MRI" | Plan | ✓ | - |
| genetic_testing_plan | "molecular profiling assay [redacted]" | A/P | ✓ | 可能是 Oncotype DX |
| Referral | 全"None" | - | ✓ | - |
| follow up | "F/U with Dr. *****" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 3×P2 — 整体提取质量好
- P2: Metastasis/Distant "No" 但 staging 为 Mx（PET/CT 待做）
- P2: Procedure_Plan 漏条件性 ISPY biopsy 和手术讨论

---

### Row 69 (coral_idx=209)

**患者概况**: ~62F，双侧乳腺癌（左ILC ER>95%/PR 2-5%/HER2-，右IDC ER 95%/PR-/HER2 equivocal），BRCA1+。S/p 新辅助 TC x6 → 双侧乳房切除术 May 2020。术后随访：重启 letrozole，计划 prolia（骨质疏松），radiation consult，CT follow-up 肺结节。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | 术后随访 |
| Type_of_Cancer | 左ILC ER+/PR+/HER2-, 右IDC ER+/PR-/HER2- | 病理 | ✓ | 与 biopsy/surgical path 一致 |
| Stage_of_Cancer | "左 IIA, 右 I" | 推断 | ✓ | 合理 |
| Metastasis | 未见 | staging neg | ✓ | CT/MRI brain 均阴性 |
| current_meds | "letrozole" | med list | ✓ | - |
| recent_changes | "restart letrozole" | interim | ✓ | - |
| supportive_meds | "letrozole 2.5mg" | med list | P1 | **letrozole 是 hormonal therapy，非 supportive care** |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not mentioned in note." | - | P1 | **手术病理明确有 treatment effect：左侧残留4.4cm ILC + 1 LN+，说明新辅助化疗有部分反应** |
| Medication_Plan_chatgpt | 有(hormonal) | - | ✓ | - |
| medication_plan | "Restart letrozole" | A/P | P1 | **遗漏 prolia/denosumab（骨质疏松，dental clearance 后开始）** |
| radiotherapy_plan | "expanders prior to radiation" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | P1 | **A/P item 6: "She is going to have expanders placed" — 明确手术计划** |
| Imaging_Plan | "CT June 2020 for lung nodules" | A/P | ✓ | - |
| Lab_Plan | "No labs planned." | - | P1 | **Interim: "schedule with labs prior prolia" — 明确计划** |
| Referral Specialty | "Radiation consult" | A/P item 5 | ✓ | - |
| follow up | "CT June, RTC September" | A/P | ✓ | - |
| Advance care | "Full code." | 原文有 | ✓ | - |

**白名单笔记**:
- [ ] prolia/denosumab 应在 oncology_drugs.txt（此行为骨质疏松用途）

**本行总结**: 0×P0, 5×P1, 0×P2
- P1: supportive_meds 误将 letrozole（hormonal therapy）归为支持性用药
- P1: response_assessment 遗漏手术病理 treatment effect
- P1: medication_plan 遗漏 prolia/denosumab
- P1: Procedure_Plan 遗漏 expander placement
- P1: Lab_Plan 遗漏 pre-prolia labs

---

## Pattern Review: Rows 60-69

### 本批统计
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|----|
| 60 | 200 | 0 | 2 | 0 | Medication_Plan_chatgpt空, current_meds遗漏 |
| 61 | 201 | 0 | 0 | 1 | genetic_testing_plan |
| 62 | 202 | 0 | 1 | 2 | response_assessment仅引影像 |
| 63 | 203 | 0 | 4 | 4 | 多字段错误(metastasis/meds/procedure) |
| 64 | 204 | 0 | 2 | 2 | Imaging/Lab含biopsy误分类 |
| 65 | 205 | 0 | 2 | 3 | Medication_Plan_chatgpt空, medication_plan漏药 |
| 66 | 206 | 0 | 1 | 3 | Medication_Plan_chatgpt空 |
| 67 | 207 | 0 | 5 | 2 | HER2-错(TCHP=HER2+), regional LN→distant |
| 68 | 208 | 0 | 0 | 3 | Mx→No, conditional procedures |
| 69 | 209 | 0 | 5 | 0 | supportive_meds误分类, prolia遗漏 |

**本批合计**: 0×P0, 22×P1, 20×P2

### 累计统计
| 批次 | P0 | P1 | P2 | 合计 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 50 |
| Rows 10-19 | 1 | 25 | 29 | 55 |
| Rows 20-29 | 0 | 15 | 21 | 36 |
| Rows 30-39 | 0 | 12 | 18 | 30 |
| Rows 40-49 | 0 | 17 | 23 | 40 |
| Rows 50-59 | 0 | 13 | 19 | 32 |
| Rows 60-69 | 0 | 22 | 20 | 42 |
| **总计** | **2** | **126** | **157** | **285** |

### 本批特点
1. **高 P1 批次**: 22 P1 是目前第二高（仅次于 Rows 10-19 的 25），主要因为 Row 63 (4P1)、Row 67 (5P1)、Row 69 (5P1) 三个复杂病例
2. **第二意见/会诊为主**: 本批 10 行中有 5 行是 second opinion 或 initial consult（Row 62-66），这类笔记信息量大、redaction 多，提取难度高
3. **0 P0**: 连续 50 行无 P0（自 Row 19 以来），系统性准确度问题已基本修复

### 本批新问题模式
1. **TCHP 方案 → HER2 判断失败** (Row 67): TCHP 包含 trastuzumab+pertuzumab，是 HER2+ 专用方案。POST-HER2-VERIFY 规则应触发但未触发（可能因为 TCHP 作为完整方案名不在靶向药列表中，需拆分识别）
2. **Regional LN → Distant Metastasis 混淆** (Row 67): 腋窝+内乳淋巴结被标为 distant metastasis，与医生"no evidence of metastatic presence"直接矛盾
3. **"Probably/suspicious" → 确诊** (Row 63): 骨活检待做的"suspicious"病灶被标为确诊转移
4. **Cross-field 误分类** (Row 64): research core biopsy 同时出现在 Imaging_Plan 和 Lab_Plan（应在 Procedure_Plan）
5. **Hormonal therapy → supportive_meds** (Row 69): letrozole 被归为 supportive_meds（继 Xgeva/denosumab 和 goserelin 后又一个分类错误）
6. **Treatment effect 遗漏** (Row 69): 手术病理 treatment effect 未被 response_assessment 捕获

### Medication_Plan_chatgpt 空率
本批 10 行中 3 行为空 (Row 65, 66, 67) = 30%。整体趋势下降（前 60 行约 60%，本批 30%）。

---

### Row 70 (coral_idx=210)

**患者概况**: 45F，左乳 IDC grade 3，ER+ 90-100%/PR+ 1-10%/HER2- (IHC 1+)。Stage IIIB，axillary+subpectoral LN+。PET 示左第5肋可疑骨病灶（equivocal）。2周前开始 neo-adjuvant AC/taxol。远程会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | consultation |
| second opinion | "no" | - | P2 | 实为second opinion（"will return to us as needed"，继续在原处管理） |
| in-person | "Televisit" | PE "video visit" | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 病理 | ✓ | - |
| Stage_of_Cancer | "Stage IIIB" | A/P | ✓ | - |
| Metastasis | "Not sure" | rib equivocal | ✓ | **优秀处理**：PET示可疑骨病灶，医生说"may or may not be" |
| Distant Metastasis | "Not sure" | - | ✓ | 同上 |
| current_meds | "neo-adjuvant AC/taxol" | HPI | ✓ | 2周前开始 |
| supportive_meds | "Zofran" | Meds | ✓ | - |
| goals_of_treatment | "curative" | A/P | ✓ | - |
| response_assessment | "Not mentioned" | - | ✓ | 刚开始化疗2周 |
| Medication_Plan_chatgpt | 有(详细chemo) | - | ✓ | - |
| medication_plan | AC/taxol详细剂量+adjuvant hormonal+CDK4/6i讨论 | A/P | ✓ | 非常全面 |
| therapy_plan | 详细多模态方案 | A/P | ✓ | - |
| radiotherapy_plan | "post-op RT + possible stereotactic RT to rib" | A/P | ✓ | 全面 |
| Procedure_Plan | "axillary dissection+subpectoral LN" | A/P | ✓ | - |
| Imaging_Plan | "repeat scan post-chemo" | A/P | ✓ | - |
| Referral | 全"None" | - | P2 | A/P提到条件性surgical referral |
| follow up | "return as needed" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2 — 非常干净，特别是 Metastasis "Not sure" 处理很好
- P2: second opinion 应为 "yes"
- P2: Referral 漏条件性 surgical referral

---

### Row 71 (coral_idx=211)

**患者概况**: 72F，左乳 IDC grade 2，1.2cm，ER+ 99%/PR- <1%/HER2- (IHC 1, FISH-)，Ki-67 20%，focal neuroendocrine differentiation。S/p mastectomy + SLN 0/2 neg。远程初诊。Plan: letrozole + Oncotype DX。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| in-person | "Televisit" | Zoom | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- IDC + neuroendocrine diff" | 病理 | ✓ | 准确 |
| Stage_of_Cancer | "pT1cN0(sn)" | 病理 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| current_meds | "Every 6 Hours; latanoprost; zoledronic acid(RECLAST)" | med list | P1 | **解析错误：包含garbled text("Every 6 Hours")和非肿瘤药(latanoprost=眼药, Reclast=骨质疏松)** |
| recent_changes | "begin letrozole" | Plan | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan有letrozole和Oncotype** |
| medication_plan | "begin letrozole" | Plan | ✓ | - |
| therapy_plan | "letrozole + Oncotype评估chemo获益" | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | mastectomy+小肿瘤+N0 |
| Lab_Plan | "Ordered [REDACTED] to evaluate chemo" | Plan | P2 | 这是 Oncotype DX（genomic test），应仅在 genetic_testing_plan |
| genetic_testing_plan | "Ordered [REDACTED] to evaluate chemo" | Plan | ✓ | - |
| follow up | "3 weeks to review results" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- [ ] zoledronic acid/Reclast 在此为骨质疏松用途，非肿瘤治疗 → non_oncology 场景需区分

**本行总结**: 0×P0, 2×P1, 1×P2
- P1: current_meds 解析错误（garbled text + 非肿瘤药）
- P1: Medication_Plan_chatgpt 空
- P2: Lab_Plan 包含 genomic test（应仅在 genetic_testing_plan）

---

### Row 72 (coral_idx=212)

**患者概况**: 63F，Stage III 左乳癌 ER/PR+/HER2-，s/p bilateral mastectomy + ALND/Abraxane + CW XRT + arimidex (since 2017)。Follow-up，doing well。新结节 → US/mammo 确认 fat necrosis。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER/PR+ IDC, HER2-" | HPI | ✓ | - |
| Stage_of_Cancer | "Stage III" | HPI/A | ✓ | - |
| Metastasis | "No" | PET neg | ✓ | - |
| current_meds | "arimidex" | Plan | ✓ | - |
| supportive_meds | "arimidex" | - | P1 | **Arimidex = aromatase inhibitor = hormonal therapy，非 supportive care** |
| response_assessment | "no recurrence, fat necrosis not cancer" | imaging | ✓ | 准确 |
| Medication_Plan_chatgpt | 有(continue arimidex) | - | ✓ | - |
| medication_plan | "Continue arimidex" | Plan | P2 | Plan 还有 "Continue ***** *****"（可能是骨密度药物），被遗漏 |
| Lab_Plan | "check labs" | Plan | ✓ | 原文如此(vague) |
| follow up | "4 months" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- [ ] arimidex/anastrozole 不应被归为 supportive_meds（重复出现的 AI 误分类模式）

**本行总结**: 0×P0, 1×P1, 1×P2
- P1: supportive_meds 误将 arimidex（hormonal therapy）归为支持性用药
- P2: medication_plan 遗漏第二个 ongoing treatment

---

### Row 73 (coral_idx=213)

**患者概况**: 68F，h/o 转移性 HER2+ 胃癌（已缓解），新诊断右乳 IDC Stage IIB T2N1M0 (2.5cm, grade 2, ER+/PR+/HER2-)。S/p bilateral mastectomy + axillary dissection。初诊建立肿瘤科 care。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | 病理 | ✓ | - |
| Stage_of_Cancer | "pT2N1a" | 手术病理 | ✓ | - |
| Metastasis | "No" | - | ✓ | - |
| current_meds | "" (空) | - | ✓ | 尚未开始辅助治疗 |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | breast cancer尚未治疗 |
| Medication_Plan_chatgpt | 有(hormonal+chemo) | - | ✓ | - |
| medication_plan | "AI alone vs TC then AI" | A/P | P2 | TC 展开为"docetaxel and carboplatin"错误，应为 docetaxel + cyclophosphamide |
| radiotherapy_plan | "None" | A/P "does not require RT" | ✓ | - |
| genetic_testing_plan | "order submitted for [REDACTED] testing" | Plan | ✓ | 可能是 Oncotype DX |
| follow up | "3 weeks" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 1×P2 — 非常干净，双癌病史处理正确
- P2: medication_plan 中 TC 错误展开为 docetaxel+carboplatin（应为 cyclophosphamide）

---

### Row 74 (coral_idx=214)

**患者概况**: 33F，premenopausal，右乳 IDC grade 3，ER-/PR-/HER2+ (3+)，Ki-67 70-80%。2.1cm with axillary LN+ (extending to level III)。PET 无远处转移。Second opinion via telehealth。同意 neoadjuvant TCHP。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "yes" | - | ✓ | - |
| in-person | "Televisit" | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/[REDACTED]+ IDC" | 病理 | ✓ | TCHP confirms HER2+ |
| Stage_of_Cancer | "Stage II-III" | A/P | ✓ | - |
| Metastasis | "No" | PET neg | ✓ | - |
| current_meds | "" (空) | - | ✓ | 尚未开始 |
| goals_of_treatment | "curative" | - | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan详细讨论TCHP+T-DM1 adjuvant** |
| medication_plan | "TCHP docetaxel+carboplatin+trastuzumab+pertuzumab" | Plan | ✓ | 详细 |
| therapy_plan | 新辅助+手术+放疗+adjuvant options | Plan | ✓ | - |
| radiotherapy_plan | "refer to Rad Onc post-surgery" | Plan | ✓ | - |
| Procedure_Plan | "refer to Breast Surgery" | Plan | P2 | 遗漏 port placement（"Agree with port"） |
| Imaging_Plan | "No imaging planned." | - | P2 | 遗漏 TTE（"Agree with TTE prior to chemo"） |
| genetic_testing_plan | "await germline testing results" | Plan | ✓ | - |
| Referral Genetics | "pending genetics counseling" | Plan | ✓ | - |
| Referral Specialty | "UCSF Breast Surgery, Rad Onc" | Plan | ✓ | - |
| Referral Others | "None" | - | P2 | 遗漏 fertility referral（Plan明确"fertility referrals"） |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: Medication_Plan_chatgpt 空
- P2: Procedure_Plan 漏 port placement
- P2: Imaging_Plan 漏 TTE
- P2: Referral 漏 fertility referral

---

### Row 75 (coral_idx=215)

**患者概况**: 55F，HR-/HER2+ 转移性乳腺癌（骨：左髂骨+双侧骶骨）。目前维持 trastuzumab+pertuzumab（paclitaxel 2017完成）。S/p partial mastectomy + XRT。Follow-up，新发手臂/膝痛。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2+ metastatic" | 病理 | ✓ | - |
| Metastasis | "Yes, bone+left iliac" | MRI pelvis | ✓ | 稍不完整(漏sacral ala)但主要信息有 |
| current_meds | "Trastuzumab, Pertuzumab" | A/P | ✓ | - |
| recent_changes | "Restart Gabapentin 300mg TID" | A/P | ✓ | - |
| supportive_meds | "Lomotil" | A/P | P2 | 不完整：漏 Naprosyn 500mg BID, Voltaren gel, pantoprazole |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | "PET 04/18: no recurrent/metastatic disease" | PET | ✓ | 骨转移稳定/缓解 |
| Medication_Plan_chatgpt | 有(chemo详细) | - | ✓ | - |
| medication_plan | 全面：H/P + supportive meds + metformin | A/P | ✓ | - |
| Imaging_Plan | "PETCT up to toes" | A/P | P2 | 漏 repeat echo (due Nov 2018, q6m monitoring) |
| Lab_Plan | "labs due Aug/Sep 2018" | A/P | ✓ | - |
| Referral | 全"None" | - | P2 | A/P: "F/u with PCP" for DM |
| follow up | "9 weeks" | A/P | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 3×P2
- P2: supportive_meds 不完整（漏 Naprosyn, Voltaren, pantoprazole）
- P2: Imaging_Plan 漏 repeat echo (HER2 靶向治疗心脏监测)
- P2: Referral 漏 PCP follow-up for DM

---

### Row 76 (coral_idx=216)

**患者概况**: 52F，右乳 IDC 2.2cm grade 2 ER+/PR+/HER2-，s/p lumpectomy with IORT，1/1 LN+，Oncotype 5（低风险）。初诊讨论辅助治疗。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC with DCIS" | 病理 | ✓ | - |
| Stage_of_Cancer | "pT2N1a(sn) ≈ Stage II" | 病理 | ✓ | - |
| current_meds | "" | - | ✓ | 尚未开始 exemestane |
| recent_changes | "Prescribed exemestane" | A/P | ✓ | - |
| Medication_Plan_chatgpt | 有(hormonal) | - | ✓ | - |
| medication_plan | 全面 | A/P | ✓ | - |
| radiotherapy_plan | "scheduled for adjuvant RT" | A/P | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | P2 | Plan 提到 "In the future: DEXA scan" |
| Lab_Plan | "estradiol + hep serologies + f/u estradiol" | A/P | ✓ | - |
| follow up | "3-6 months check estradiol" | A/P | ✓ | - |

**本行总结**: 0×P0, 0×P1, 1×P2
- P2: Imaging_Plan 漏 future DEXA scan

---

### Row 77 (coral_idx=217)

**患者概况**: 79F，PD-L1 negative metastatic TNBC（肝+门静脉LN），s/p 多线治疗（capecitabine, OPERA trial, gemcitabine）。03/15/19 起停药。进展中。讨论试验和放疗选项。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER-/PR-/HER2- TNBC" | A/P | ✓ | - |
| Stage_of_Cancer | "Originally Stage IIA, now Stage IV" | - | P1 | **"Originally Stage IIA"无原文支持，hallucination 模式** |
| Metastasis | "Yes, liver+periportal LNs" | A/P | ✓ | - |
| current_meds | "" (空) | - | ✓ | 已停药 since 03/15/19 |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | "Worsening on 08/07/19 CT" | CT | ✓ | 准确反映进展 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **讨论了 doxil, eribulin, 多个试验选项** |
| radiotherapy_plan | "consult with Rad Onc for liver/nodal XRT" | A/P | ✓ | - |
| genetic_testing_plan | "screening for trial..." | - | P2 | 这是临床试验筛选，非 genetic testing |
| Referral Specialty | "Radiation oncology consult" | A/P | ✓ | - |

**本行总结**: 0×P0, 2×P1, 1×P2
- P1: Stage "Originally Stage IIA" hallucination
- P1: Medication_Plan_chatgpt 空
- P2: genetic_testing_plan 误含临床试验筛选

---

### Row 78 (coral_idx=218)

**患者概况**: 61F，转移性乳腺癌（多部位含骨），合并 PE/胸腔积液/DM1。已停药。需要 restaging 和重新评估受体状态。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "Breast cancer metastasized to multiple sites" | A/P | P2 | 缺受体状态（原文也不清晰，正在 re-evaluate） |
| current_meds | "denosumab" | med list | ✓ | - |
| recent_changes | "Stopped all treatment last week" | HPI | ✓ | - |
| response_assessment | "Not responding, tumor marker rising" | A/P | ✓ | - |
| Medication_Plan_chatgpt | empty [] | - | P1 | **讨论了治疗方案但无具体药** |
| Procedure_Plan | "power port + thoracentesis" | A/P | ✓ | 准确 |
| Imaging_Plan | "PET/CT restaging + thoracentesis" | A/P | P2 | thoracentesis 是 procedure 非 imaging |
| genetic_testing_plan | "ER and [REDACTED] from thoracentesis fluid" | A/P | P2 | 这是受体状态检测(pathology)，非 genomic testing |

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: Medication_Plan_chatgpt 空
- P2: Type_of_Cancer 缺受体状态
- P2: Imaging_Plan 含 thoracentesis（procedure）
- P2: genetic_testing_plan 误含 receptor status testing

---

### Row 79 (coral_idx=219)

**患者概况**: 53F，7年前因 DCIS 做 mastectomy，现局部复发 IDC 0.8cm grade 3 ER/PR+。Oncotype 24。Plan: TC x4 + 6 weeks radiation。Telehealth。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | pre-chemo visit |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | - | ✓ | - |
| Stage_of_Cancer | "Originally Stage IIA, now local recurrence" | - | P1 | **原始为 DCIS (Stage 0)，"Stage IIA" hallucination** |
| current_meds | "" | - | ✓ | 尚未开始 |
| recent_changes | "Start TC x4 on 04/11/19" | Plan | ✓ | - |
| supportive_meds | "dexamethasone, ondansetron, prochlorperazine..." | med list | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 TC x4 + claritin + cold cap** |
| therapy_plan | "TC x4 + 6 weeks radiation" | Plan | ✓ | 全面 |
| radiotherapy_plan | "6 weeks, 5+1 boost, L axilla+SC" | Plan | ✓ | - |
| genetic_testing_plan | "WGS done, no actionable mutation" | A/P | P2 | 已完成检测的结果报告，非 future plan |
| Referral Specialty | "Radiation oncology consult" | - | P2 | 会诊已发生(Dr. ***** telemedicine)，非新 referral |

**本行总结**: 0×P0, 2×P1, 2×P2
- P1: Stage "Originally Stage IIA" hallucination（原始为 DCIS=Stage 0）
- P1: Medication_Plan_chatgpt 空
- P2: genetic_testing_plan 列已完成检测结果
- P2: Referral 列已完成的会诊

---

### Row 80 (coral_idx=220)

**患者概况**: 60F，左乳两个 Stage I 原发癌（IDC tubular 3mm + ILC 2mm），均 ER+/PR+/HER2- G1。S/p lumpectomy，SLN 阴性。Plan: radiation → AI。第二意见 televisit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | - |
| second opinion | "yes" | - | ✓ | "outside consultation" |
| in-person | "Televisit" | - | ✓ | televisit |
| Type_of_Cancer | "ER+/PR+/HER2- IDC tubular + ILC" | - | ✓ | 两个独立原发均准确 |
| Stage_of_Cancer | "Stage I (pT1a...)" | - | ✓ | 两个 <1cm 肿瘤 + SLN neg |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 两个原发详细病理 | - | ✓ | 全面准确 |
| current_meds | "" | - | ✓ | 无当前肿瘤药 |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | 早期，治愈目标 |
| response_assessment | "Not yet on treatment" | - | ✓ | 术后未开始系统治疗 |
| Medication_Plan_chatgpt | radiotherapy + hormonal therapy | A/P | ✓ | 有内容，涵盖 radiation + AI |
| medication_plan | "An [REDACTED] will be recommended after radiation" | Plan | ✓ | AI 被 redact，合理 |
| therapy_plan | radiation + AI, no chemo | Plan | ✓ | 全面 |
| radiotherapy_plan | "referred to radiation oncologist" | Plan | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | 手术已完成 |
| Imaging_Plan | "Baseline dexa." | Plan | ✓ | 匹配 A/P |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral Specialty | "Radiation oncologist" | Plan | ✓ | - |
| Others | "None" | - | ✓ | - |
| follow up | "RTC post radiation to start hormone blockade" | Plan | ✓ | - |
| Advance care | "Full code." | note | ✓ | 原文确认 |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 0×P2 — 非常干净

---

### Row 81 (coral_idx=221)

**患者概况**: 52F，右乳混合性导管/小叶癌 4.3cm，ER+/PR+/HER2-，G2-3。S/p lumpectomy + SLN biopsy。Oncotype 低风险→不推荐化疗。Plan: radiation → AI +/- bone meds，DEXA，exercise counseling referral。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | "New Patient Evaluation" |
| second opinion | "no" | "I do not recommend chemo..." | P2 | 值正确但归因不相关 |
| in-person | "in-person" | time spent 65min | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- mixed ductal and lobular" | A/P | ✓ | 准确 |
| Stage_of_Cancer | "Stage IB" | staging section | P1 | **医生明确写 "Stage II"，pT2≥Stage IIA，IB 不可能** |
| Metastasis | "No" | radiation discussion | P2 | 值正确但归因不相关 |
| Distant Metastasis | "No" | A/P staging | ✓ | - |
| lab_summary | "Glucose 94 (11/16, 11/17)" | labs section | ✓ | 手术期间的glucose |
| findings | "4.3cm, G2-3, 11/24 LN involved" | surgery note | P2 | "November 24" 可能是日期非分数，与 Stage II 矛盾（11/24+ = N3 = IIIC） |
| current_meds | "" | - | ✓ | 无当前肿瘤药 |
| supportive_meds | "docusate, oxycodone" | med list | ✓ | 术后止痛+软便 |
| goals_of_treatment | "curative" | chemo discussion | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | 术后未开始系统治疗 |
| Medication_Plan_chatgpt | hormonal therapy + radiotherapy | A/P | ✓ | 有内容，两个类别 |
| medication_plan | "Continue acetaminophen, ibuprofen, HCTZ, lisinopril, metformin + AI +/- bone med" | med list + Plan | ✓ | 非肿瘤药多但Plan部分准确 |
| therapy_plan | "no chemo, radiation, AI +/- bone med" | A/P | ✓ | 全面 |
| radiotherapy_plan | "appointment with Dr. *** tomorrow" | A/P | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | 手术已完成 |
| Imaging_Plan | "Dexa to assess bone health" | A/P #6 | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | Oncotype 已完成 |
| Referral Specialty | "" (空) | - | P2 | 应为 "None"（rad onc 已有 appointment，非新 referral） |
| Others | "Exercise counseling referral" | A/P #8 | ✓ | - |
| follow up | "after radiation" | A/P #7 | ✓ | - |
| Advance care | "Full code." | note | ✓ | 原文确认 |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: Stage "Stage IB" 错误 — 医生写 "Stage II"，pT2 (4.3cm) 最低 Stage IIA
- P2: "11/24 lymph nodes" 解读 "November 24" 为分数，与 Stage II 矛盾
- P2: Specialty referral 空字符串而非 "None"
- P2: second opinion/Metastasis 归因文字不相关

---

### Row 82 (coral_idx=222)

**患者概况**: 77F，右乳浸润性小叶癌，右腋窝 LN 转移（regional）。2019年12月起新辅助内分泌治疗 letrozole。本次 telehealth 随访，PET/CT 显示腋窝淋巴结显著缩小（SUV 15.1→1.9）。骨扫描阴性，CT 无远处转移。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | P2 | 有 "INTERVAL Last visit Dec 2019"，更像 follow-up，但 "med onc consultation" 有歧义 |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | - | ✓ | "real-time Telehealth tools" |
| Type_of_Cancer | "Lobular Breast Cancer, right with metastasis to right axilla LN" | biopsy | P2 | 缺受体状态（原文 ER/PR 严重 redaction） |
| Stage_of_Cancer | "Originally not specified, now metastatic (Stage IV)" | - | P1 | **错误：workup 明确 "negative for distant metastasis"，骨扫描阴性，CT 无远处转移。非 Stage IV** |
| Metastasis | "Yes, to right axillary lymph nodes" | - | P2 | 腋窝LN是 regional，"Metastasis" 字段歧义（可含 regional） |
| Distant Metastasis | "Yes, to right axillary LN" | - | P1 | **错误：腋窝LN = REGIONAL。CT 明确 "compatible with regional metastatic disease"、"No CT evidence of distant metastases"** |
| lab_summary | "No labs in note." | - | ✓ | Telehealth 无近期 labs |
| findings | PET/CT 显著响应 + 腋窝LN缩小 | PET/CT report | ✓ | 详细准确 |
| current_meds | "letrozole" | - | ✓ | - |
| supportive_meds | "" | - | ✓ | 非肿瘤药正确排除 |
| goals_of_treatment | "curative" | - | ✓ | 新辅助→手术 = curative |
| response_assessment | "responding to neoadjuvant letrozole, significant PET/SUV response" | PET/CT | ✓ | 准确详细 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 continue letrozole until surgery** |
| medication_plan | "Continuing neoadjuvant endocrine therapy with letrozole" | Plan | ✓ | - |
| therapy_plan | "continue neoadjuvant endocrine therapy until breast surgery" | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "upcoming breast surgery" | Plan | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | PET/CT 刚做完 |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral | 全 "None" | - | ✓ | - |
| follow up | "upcoming breast surg follow up" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | Telehealth 无 code status |

**白名单笔记**:
- [ ] 加强 Distant Metastasis 规则: axillary LN = REGIONAL（已知系统性问题，本行再次出现）

**本行总结**: 0×P0, 3×P1, 2×P2
- P1: Stage "Stage IV" 错误 — workup 明确 negative for distant metastasis
- P1: Distant Metastasis 将腋窝LN标为 distant（应为 regional）
- P1: Medication_Plan_chatgpt 空
- P2: Patient type 歧义
- P2: Type_of_Cancer 缺受体状态（原文 redaction）

---

### Row 83 (coral_idx=223)

**患者概况**: 60F，CHEK2 突变，多发性硬化，转移性乳腺癌（骨、软组织、肝、可能脑膜）。2006年原发 IDC ER+/PR+/HER2-，2019年转移复发。先前 letrozole+palbociclib PD，现用 Xeloda 1500mg BID + zolendronic acid。疑似柔脑膜转移。远程会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初次会诊 |
| second opinion | "no" | - | P1 | **"She is a patient of Dr. *** at ***"，来自其他中心寻求 recommendations = 第二意见** |
| in-person | "Televisit" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- IDC" | 2019 biopsy | ✓ | 使用转移活检的受体状态，准确 |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | - | P2 | "Originally Stage IIA" 未确认（2006 T1c，ALND 信息不全） |
| Metastasis | "Yes (bone, soft tissue, liver, meninges)" | - | ✓ | 全面准确 |
| Distant Metastasis | "Yes (bone, soft tissue, liver, meninges)" | - | ✓ | 真正的远处转移 |
| lab_summary | "WBC 7.0, Hg 8.4, Plt 80, etc." | labs section | ✓ | - |
| findings | 脑MRI 进展 + CT CAP 进展 | imaging reports | ✓ | 详细 |
| current_meds | "capecitabine, zolendronic acid" | Plan | ✓ | - |
| supportive_meds | "zolendronic acid" | - | P2 | 骨转移用药应算 oncology treatment 而非 supportive |
| goals_of_treatment | "palliative" | - | ✓ | "not curable, but treatable" |
| response_assessment | PD on letrozole/palbociclib, worsening brain MRI | - | ✓ | 准确详细 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 Xeloda+steroids+zoledronic acid+future fulvestrant** |
| medication_plan | "Continue xeloda 1500mg BID, steroids, zolendronic acid" | Plan | ✓ | - |
| therapy_plan | Xeloda + radiation + future fulvestrant+ inhibitor | Plan | ✓ | 全面 |
| radiotherapy_plan | "referral to rad onc for focal CNS radiation or [REDACTED]" | Plan | ✓ | - |
| Procedure_Plan | "Repeat LP for CSF cytology" | Plan | ✓ | LP 是 procedure |
| Imaging_Plan | "Repeat CT CAP, Repeat MRI spine for LMD" | Plan | ✓ | - |
| Lab_Plan | "Repeat LP for CSF cytology" | Plan | P1 | **LP 是 procedure 非 blood test，Lab_Plan prompt 限 "BLOOD TESTS"** |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral Specialty | "Radiation oncology consult..." (重复文本) | Plan | P2 | 值正确但文本重复/截断 |
| follow up | "Repeat CT..." | Plan | P2 | 列的是 imaging，非 follow-up 时间 |
| Advance care | "Not discussed" | - | ✓ | Telehealth 无 code status |

**白名单笔记**:
- [ ] Lab_Plan 黑名单: "LP", "lumbar puncture", "CSF cytology" → 这些是 procedure 不是 blood test

**本行总结**: 0×P0, 3×P1, 4×P2
- P1: second opinion 应为 "yes"（外院患者来此寻求 recommendations）
- P1: Medication_Plan_chatgpt 空
- P1: Lab_Plan 包含 LP（非血液检验）
- P2: Stage "Originally Stage IIA" 未确认
- P2: supportive_meds 含 zolendronic acid（骨转移治疗药）
- P2: Referral Specialty 文本重复, follow up 列 imaging 非时间

---

### Row 84 (coral_idx=224)

**患者概况**: 61F，ER+/PR-/HER2- ILC，pathologic Stage IIIA。复杂病史：2016年右乳ILC→双侧乳切+ALND+AC/T+XRT→1年内转移（骨、肌肉、脑 s/p GK）。Fulvestrant/palbociclib PD（新肝转移）。左锁骨/肩胛姑息放疗。当前讨论系统治疗选择（phase 1 trial ***+olaparib）。脑膜受累疑似累及三叉神经。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 45min face-to-face | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- ILC" | biopsy | ✓ | - |
| Stage_of_Cancer | "Originally Stage IIIA, now Stage IV" | note | ✓ | 原文明确 "pathologic stage IIIA" |
| Metastasis | "Yes (bone, muscle, liver, brain)" | - | ✓ | - |
| Distant Metastasis | "Yes (bone, muscle, liver, brain)" | - | ✓ | - |
| lab_summary | "CA 15-3: 45.3→360 (H), CEA: 3.8→25.6 (H)" | labs | ✓ | 上升趋势反映进展 |
| findings | progressive disease, brain MRI 脑膜/三叉神经 | imaging | ✓ | - |
| current_meds | "fulvestrant, palbociclib, denosumab" | med list | P1 | **Fulvestrant/palbociclib 已 PD 停用，不应列为 current** |
| supportive_meds | "morphine, ondansetron, prednisone" | med list | ✓ | 均在 med list 确认 |
| goals_of_treatment | "palliative" | - | ✓ | - |
| response_assessment | "progressing on treatment" | PD evidence | ✓ | - |
| Medication_Plan_chatgpt | {radiotherapy referral only} | Plan | P2 | 有内容但不全，遗漏 trial 和 steroid/pain management |
| medication_plan | "Continue steroid taper, continue pain meds" | Plan | ✓ | - |
| therapy_plan | "phase 1 trial ***+olaparib" | Plan | ✓ | - |
| radiotherapy_plan | "Rad Onc referral, Dr. ***, 2-wk washout" | Plan | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "Brain MRI reviewed by UCSF neuroradiology..." | A/P | P1 | **描述过去影像 review 结果，非 future plan。应为 "No imaging planned"** |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "phase 1 trial evaluation for *** mutations" | Plan | P1 | **临床试验评估非基因检测。Prompt 明确排除 clinical trial options** |
| Referral Specialty | "Rad Onc referral" | Plan | ✓ | - |
| follow up | "Follow up in 2 weeks" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 无 code status |

**白名单笔记**:
- [ ] current_meds 时态规则: 当A/P明确说 "progressed on" 某药，应视为过去用药
- [ ] Imaging_Plan 黑名单: "reviewed by", "found to have" 等过去时描述不应出现在 future plan

**本行总结**: 0×P0, 3×P1, 1×P2
- P1: current_meds 含已 PD 停用的 fulvestrant/palbociclib
- P1: Imaging_Plan 描述过去影像 review 而非 future plan
- P1: genetic_testing_plan 列临床试验（prompt 明确排除）
- P2: Medication_Plan_chatgpt 不全

---

### Row 85 (coral_idx=225)

**患者概况**: 53F，转移性乳腺癌（骨、肝、?卵巢、?脑膜）。原发 2014 年右乳混合 IDC，Gr III，HER2 2+ FISH 4.37(=HER2+)。S/p TCHP x6 → bilateral MRM → adj XRT → adjuvant ***。2019 年转移复发（ER 95%/PR 2%/HER2 1+ FISH neg = HER2-）。Letrozole + ribociclib PD。CHEK2 突变。远程会诊讨论治疗选择。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初次会诊 |
| second opinion | "no" | - | P1 | **"here for med onc consultation"，已有 Dr. *** 作为治疗医生，这是其他中心会诊** |
| in-person | "Televisit" | video | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- mixed IDC, Gr III" | biopsy | ✓ | 使用转移活检受体状态 |
| Stage_of_Cancer | "Originally not specified, now Stage IV" | - | P2 | 可推断 T3N2a≈Stage IIIA，但原文未明确 |
| Metastasis | "Yes (bone, liver, possibly ovary)" | PET/CT | ✓ | - |
| Distant Metastasis | "Yes (bone, liver, possibly ovary)" | PET/CT | ✓ | 真正远处转移 |
| lab_summary | "No labs in note." | - | ✓ | telehealth |
| findings | PET/CT 骨进展 + 脑 MRI 疑似脑膜 | imaging | ✓ | - |
| current_meds | "letrozole, ribociclib, denosumab" | med list/Plan | ✓ | 当前仍在用，推荐切换但尚未换 |
| supportive_meds | "Oxycodone, denosumab" | med list | P2 | denosumab 骨转移治疗应算 oncology |
| goals_of_treatment | "palliative" | - | ✓ | - |
| response_assessment | PD: increasing bone mets, brain suspicious | - | ✓ | - |
| Medication_Plan_chatgpt | hormonal therapy: fulvestrant+/-everolimus | Plan | ✓ | 有内容，有用 |
| medication_plan | "fulvestrant +/- everolimus, continue denosumab" | Plan | ✓ | - |
| therapy_plan | fulvestrant+/-everolimus + palliative XRT + denosumab | Plan | ✓ | 全面 |
| radiotherapy_plan | "palliative XRT with Dr ***" | Plan | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral Specialty | "" (空) | - | P2 | 应为 "None"，palliative XRT 已有 rad onc 不算新 referral |
| follow up | "[REDACTED] 6 wks" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 3×P2
- P1: second opinion 应为 "yes"（其他中心会诊，已有治疗医生 Dr. ***）
- P2: supportive_meds 含 denosumab（骨转移治疗药）
- P2: Referral Specialty 空字符串
- P2: Stage 可推断但未明确

---

### Row 86 (coral_idx=226)

**患者概况**: 79F，右乳 IDC G2 2.2cm 多灶+0.6cm，ER+/PR+/HER2-。S/p 切除活检+ALND（4/19 LN+，1个有包膜外扩展）。帕金森病。强家族史（女儿40岁乳癌+结肠癌，外祖母卵巢癌）。第二意见会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 第二意见 |
| second opinion | "yes" | "for a second opinion" | ✓ | - |
| in-person | "in-person" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | Assessment | ✓ | - |
| Stage_of_Cancer | "Approximately Stage II" | - | P1 | **T2(2.2cm) + N2a(4/19LN+) = Stage IIIA，非 Stage II** |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 2.2cm multifocal + 4/19 LN+ + ECE | A/P | ✓ | 准确 |
| current_meds | "" | - | ✓ | 无当前肿瘤药 |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not yet on treatment" | - | ✓ | - |
| Medication_Plan_chatgpt | hormonal therapy discussed | A/P | ✓ | 有内容 |
| medication_plan | "hormonal therapy alone" | Plan | ✓ | 未指定具体药物（原文未指定）|
| therapy_plan | "hormonal therapy alone, return to Dr. ***" | Plan | P2 | 原文也讨论了 radiation 但非本次确定 plan |
| radiotherapy_plan | "radiation discussed" | Assessment | P2 | 一般性讨论非具体 plan，但 post-lumpectomy 标准治疗确实包含 XRT |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral | 全 "None", follow up 返回 Dr. *** | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**: 无

**本行总结**: 0×P0, 1×P1, 2×P2
- P1: Stage "Stage II" 错误 — T2N2a = Stage IIIA
- P2: therapy_plan 和 radiotherapy_plan 是一般性讨论非确定 plan

---

### Row 87 (coral_idx=227)

**患者概况**: 36F，Stage III 左乳 IDC（ER weak+/PR weak+/HER2-）。S/p neoadj AC→Taxol→Taxol/Carbo（PD 停）→bilateral mastectomy+ALND（23+ nodes/30）→adj gemzar/carbo x4→XRT。脑转移 s/p 切除+SRS。肺+淋巴结转移。Foundation One ERBB2 阴性。现用 Xeloda。COVID-19 确诊。Video 会诊。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | New Patient Evaluation |
| second opinion | "no" | - | P2 | 有 primary oncologist，来此 consultation，但可能是转诊非第二意见 |
| in-person | "Televisit" | video visit | ✓ | - |
| Type_of_Cancer | "Originally ER+/PR+/HER2-, met biopsy ER-/PR-/HER2-" | pathology | P2 | 脑转移 HER2 "not done"，不应标为 HER2- |
| Stage_of_Cancer | "Originally Stage IIIB, now Stage IV" | - | P2 | N3(23+ nodes) = Stage IIIC 非 IIIB（IIIB 需 T4） |
| Metastasis | "Yes (brain, lungs, lymph nodes)" | - | ✓ | - |
| Distant Metastasis | "Yes (brain, lungs, lymph nodes)" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | 23 nodes+, 2 tumors, brain met pathology | path reports | ✓ | 全面 |
| current_meds | "capecitabine (XELODA) 500mg" | Plan | ✓ | - |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "palliative" | - | ✓ | - |
| response_assessment | "Not mentioned" | - | P2 | "clinically stable" 可算初步评估 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 Xeloda + immunotherapy at progression** |
| medication_plan | "She is on xeloda" | Plan | P2 | 仅陈述当前用药，遗漏 restaging/progression plan |
| therapy_plan | Xeloda + clinical trials at progression | Plan | ✓ | - |
| radiotherapy_plan | "had radiation + had SRS" (全为过去) | - | P1 | **全部是已完成的放疗，非 future plan。应为 "None"** |
| Procedure_Plan | "HER2 on brain met + residual disease" | A/P #4 | P1 | **这是 IHC/FISH 检测（在已有标本上），非 procedure。属于 genetic_testing_plan** |
| Imaging_Plan | "No imaging planned." | - | P2 | A/P #5 "restaging after 3 months"（虽未指定具体 imaging） |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "HER2 on brain met + residual disease" | A/P #4,7 | ✓ | 正确归入此类 |
| Referral | 全 "None" | - | ✓ | 与 primary oncologist 协调非新 referral |
| follow up | "prn or at progression" | Plan | ✓ | - |
| Advance care | "Full code." | note | ✓ | 原文确认 |

**白名单笔记**:
- [ ] Procedure_Plan 黑名单: "HER2 testing", "receptor testing", "IHC", "FISH" on existing tissue → 非 procedure
- [ ] radiotherapy_plan 时态过滤: "had radiation", "had stereotactic XRT" = 过去式，应被 G5 TEMPORAL 过滤

**本行总结**: 0×P0, 3×P1, 5×P2
- P1: Medication_Plan_chatgpt 空
- P1: radiotherapy_plan 全为过去放疗
- P1: Procedure_Plan 列 IHC/FISH 检测（非 procedure）
- P2: second opinion 歧义, HER2 "not done" 标为阴性, Stage IIIB→应 IIIC
- P2: response_assessment 遗漏 clinically stable, Imaging_Plan 遗漏 restaging

---

### Row 88 (coral_idx=228)

**患者概况**: 53F，Stage I 左乳 IDC 9mm G2，ER+/PR+/AR+/HER2-，node negative。S/p lumpectomy + SLN + radiation。基因组检测低风险→无化疗。围绝经期。保险变更后新就诊。开始 tamoxifen，计划中途换 AI。Video visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 保险变更后新就诊 |
| second opinion | "no" | - | ✓ | 非第二意见，保险转诊 |
| in-person | "Televisit" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | A/P | ✓ | AR+ 省略可接受 |
| Stage_of_Cancer | "Stage I" | staging | ✓ | pT1b, node neg |
| Metastasis/Distant Met | "No" / "No" | - | ✓ | - |
| lab_summary | "No labs in note." | - | ✓ | - |
| findings | "9mm IDC, G2, ER+/PR+/AR+/HER2-, node neg" | pathology | ✓ | 准确 |
| current_meds | "tamoxifen" | med list | ✓ | 在 current outpatient medications |
| supportive_meds | "" | - | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "Not mentioned" | - | ✓ | 早期辅助，无 formal response assessment |
| Medication_Plan_chatgpt | tamoxifen + AI switch plan | Plan | ✓ | 有内容 |
| medication_plan | "tamoxifen, change to AI midway" | Plan | ✓ | - |
| therapy_plan | "tamoxifen" | Plan | P2 | 遗漏 "plan to change to AI midway" |
| radiotherapy_plan | null | - | P2 | 应为 "None"（radiation 已完成），null 是 schema 问题 |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "Mammogram in February 2021" | A/P #5 | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | 基因组检测已完成 |
| Referral | 全 "None" | - | ✓ | - |
| follow up | "RTC 2-3 months" | A/P #4 | ✓ | - |
| Advance care | "Full code." | note | ✓ | 原文确认 |

**白名单笔记**: 无

**本行总结**: 0×P0, 0×P1, 2×P2
- P2: therapy_plan 遗漏 AI switch plan
- P2: radiotherapy_plan null 而非 "None"

---

### Row 89 (coral_idx=229)

**患者概况**: 51F，右乳腺癌 Clinical Stage II/III。临床试验受试者。S/p neoadj Taxol → 右 lumpectomy（残余 IDC 2.2cm, 60% cellularity）→ 当前 AC 化疗 cycle 3 完成。多种副作用（N/V、hot flashes、port 不适、甲状腺功能异常）。BLM 基因携带者。Telehealth 随访。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | P1 | **明确 follow-up：已治疗多个 cycle，INTERVAL section** |
| second opinion | "no" | - | ✓ | - |
| in-person | "Televisit" | - | ✓ | - |
| Type_of_Cancer | "Adenocarcinoma of right breast (HCC)" | problem list | P2 | 缺受体状态（原文 redaction 严重） |
| Stage_of_Cancer | "Not mentioned in note" | - | P1 | **A/P 明确写 "Clinical st II/III"** |
| Metastasis | "No" | - | ✓ | - |
| Distant Metastasis | "No" | - | ✓ | - |
| lab_summary | TSH, CMP, CBC, pregnancy test | labs section | ✓ | 详细 |
| findings | "2.2cm residual IDC, 60% cellularity" | path | ✓ | - |
| current_meds | "" (空) | - | P1 | **正在接受 AC 化疗 + gabapentin，应列为 current** |
| supportive_meds | "" (空) | - | P1 | **gabapentin, GCSF, granisetron, olanzapine, dex, lidocaine patches** |
| goals_of_treatment | "curative" | - | ✓ | - |
| response_assessment | "residual IDC 2.2cm, 60% cellularity" | path | ✓ | 新辅助后部分响应 |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 AC cycle 4 + 多项 anti-nausea 调整** |
| medication_plan | "AC cycle 4 + GCSF 50% + dex/olanzapine/granisetron" | Plan | ✓ | 全面 |
| therapy_plan | "Continue AC cycle 4" | Plan | P2 | 遗漏 XRT plan（A/P 提到 "RTC after XRT"） |
| radiotherapy_plan | "RTC after XRT, approx 1-1.5 mos" | Plan | ✓ | XRT planned |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "No imaging planned." | - | ✓ | - |
| Lab_Plan | "No labs planned." | - | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral | 全 "None" | - | ✓ | Endo 已 existing follow-up |
| follow up | "after XRT, 1-1.5 mos" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 无 code status |

**白名单笔记**:
- [ ] current_meds 应包含 AC 化疗方案中的药物（doxorubicin, cyclophosphamide）
- [ ] supportive_meds 应包含 gabapentin, GCSF

**本行总结**: 0×P0, 5×P1, 2×P2
- P1: Patient type 应为 "Follow up"
- P1: Stage "Not mentioned" 但原文有 "Clinical st II/III"
- P1: current_meds 空（正在 AC 化疗）
- P1: supportive_meds 空（gabapentin, GCSF 等多项）
- P1: Medication_Plan_chatgpt 空
- P2: Type_of_Cancer 缺受体状态
- P2: therapy_plan 遗漏 XRT plan

---

## Pattern Review: Rows 80-89

### 本批统计
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|----|
| 80 | 220 | 0 | 0 | 0 | 非常干净 |
| 81 | 221 | 0 | 1 | 3 | Stage IB→应 IIA, "Nov 24" 解读歧义 |
| 82 | 222 | 0 | 3 | 2 | Stage IV 错误(axillary=regional), Distant Met 错 |
| 83 | 223 | 0 | 3 | 4 | second opinion, Lab_Plan含LP, Medication_Plan_chatgpt |
| 84 | 224 | 0 | 3 | 1 | current_meds含已PD药, Imaging_Plan过去, genetic_testing列trial |
| 85 | 225 | 0 | 1 | 3 | second opinion |
| 86 | 226 | 0 | 1 | 2 | Stage II→应IIIA (T2N2a) |
| 87 | 227 | 0 | 3 | 5 | radiotherapy_plan全过去, Procedure_Plan列检测 |
| 88 | 228 | 0 | 0 | 2 | radiotherapy_plan null |
| 89 | 229 | 0 | 5 | 2 | Patient type, Stage, current/supportive_meds空 |

**本批合计**: 0×P0, 20×P1, 24×P2

### 累计统计
| 批次 | P0 | P1 | P2 | 合计 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 50 |
| Rows 10-19 | 1 | 25 | 29 | 55 |
| Rows 20-29 | 0 | 15 | 21 | 36 |
| Rows 30-39 | 0 | 12 | 18 | 30 |
| Rows 40-49 | 0 | 17 | 23 | 40 |
| Rows 50-59 | 0 | 13 | 19 | 32 |
| Rows 60-69 | 0 | 22 | 20 | 42 |
| Rows 70-79 | 0 | 9 | 18 | 27 |
| Rows 80-89 | 0 | 20 | 24 | 44 |
| **总计** | **2** | **155** | **199** | **356** |

### 本批特点
1. **P1 反弹至 20**: 主要来自转移性/复杂病例（Row 82-85, 87, 89），这些笔记有丰富的治疗历史，cross-field 分类更容易出错
2. **Medication_Plan_chatgpt 空率**: 10 行中 5 行空 (82,83,84,87,89) = 50%，又回到高空率
3. **Stage 错误 3 例**: Row 81 (IB→IIA), Row 82 (Stage IV but no distant mets), Row 86 (II→IIIA)
4. **Axillary LN = regional 再次出错**: Row 82 Distant Metastasis 将腋窝LN标为 distant

### 新系统性问题
- **radiotherapy_plan 时态过滤失败**: Row 87 全为已完成放疗（"had radiation", "had SRS"），G5 TEMPORAL 未能过滤
- **Procedure_Plan 含 IHC/FISH 检测**: Row 87 receptor testing on existing tissue 被归为 procedure（应为 genetic_testing_plan）
- **Imaging_Plan 含过去影像回顾**: Row 84 "Brain MRI reviewed by..." 是过去的 review 非 future plan
- **current_meds 时态混乱持续**: Row 84 含已 PD 停用的药物，Row 89 漏当前化疗方案
- **Patient type 持续不准**: Row 89 明确 follow-up 被标为 "New patient"（有 INTERVAL section + 多个治疗 cycle）

---

### Row 90 (coral_idx=230)

**患者概况**: 53F，Stage IV ER+/PR+/HER2- 乳腺癌（2003年 Stage I 原发，2005年骨转移复发）。当前 exemestane + everolimus + denosumab（Xgeva）月次。右腿水肿改善中（lasix）。脐部真菌感染。Follow-up。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| second opinion | "no" | - | ✓ | - |
| in-person | "in-person" | 45min face-to-face | ✓ | - |
| Type_of_Cancer | "ER+/PR+ IDC" | history | P2 | 缺 HER2 状态（原文 "*****-" = HER2-） |
| Stage_of_Cancer | "Originally Stage I, now Stage IV" | history | ✓ | - |
| Metastasis | "Yes (bone, pelvis, iliac)" | - | ✓ | - |
| Distant Metastasis | "Yes (bone, pelvis, iliac)" | - | ✓ | - |
| lab_summary | CMP+CBC 详细 (10/16/2012) | labs | ✓ | AST 60 偏高 |
| findings | 2011年 MRI/PET 历史影像 | imaging | P2 | 应包含当前 PE（1cm iliac LN），影像为 2011 年历史 |
| current_meds | "everolimus, exemestane, denosumab" | med list | ✓ | - |
| supportive_meds | "denosumab, everolimus, exemestane" | - | P1 | **三个都是 oncology 治疗药，非 supportive。supportive 应列 lasix/KCL/antifungal** |
| goals_of_treatment | "palliative" | - | ✓ | - |
| response_assessment | 2011年 MRI/PET 进展数据 | 2011 imaging | P1 | **应反映当前状态（2012年，水肿改善，iliac LN 意义不明，待 PET 评估）** |
| Medication_Plan_chatgpt | hormonal + bone therapy | Plan | ✓ | 有内容 |
| medication_plan | "lasix, denosumab, antifungal" | Plan | P2 | 遗漏 continue exemestane/everolimus |
| therapy_plan | "exemestane daily, denosumab" | Plan | P2 | 遗漏 everolimus |
| radiotherapy_plan | "None" | - | ✓ | - |
| Procedure_Plan | "No procedures planned." | - | ✓ | - |
| Imaging_Plan | "PET/CT next week" | Plan | ✓ | - |
| Lab_Plan | "Labs monthly" | Plan | ✓ | - |
| genetic_testing_plan | "None planned." | - | ✓ | - |
| Referral | 全 "None" | - | ✓ | - |
| follow up | "RTC in 1 month" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | 无 code status |

**白名单笔记**:
- [ ] supportive_meds 不应包含 everolimus/exemestane（主要治疗药物）
- [ ] exemestane 应在 oncology_drugs.txt（如尚未存在）

**本行总结**: 0×P0, 2×P1, 4×P2
- P1: supportive_meds 列 everolimus/exemestane/denosumab（均为 oncology 治疗药）
- P1: response_assessment 使用 2011 年旧数据，应反映当前临床状态
- P2: Type_of_Cancer 缺 HER2, findings 为历史影像, therapy_plan 漏 everolimus, medication_plan 不全

---

### Row 91 (coral_idx=231)

**患者概况**: 67F，转移性乳腺癌（1991年原发，20+年多线治疗历史）。当前 Epirubicin Cycle#2 D1。肝转移改善，无新症状。CEA 371.5 极高。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | - |
| Type_of_Cancer | "ER+/PR-/HER2- IDC" | history | ✓ | - |
| Stage_of_Cancer | "Originally Stage IIA, now Stage IV" | - | P1 | **原发 1991: 7/7 LN+ = N2→至少 Stage IIIA，"Stage IIA" 是 hallucination** |
| Metastasis | "Yes (liver + multiple)" | - | ✓ | - |
| current_meds | "Epirubicin, Denosumab" | med list | ✓ | - |
| response_assessment | "Liver smaller, tenderness reduced" | HPI/A/P | ✓ | - |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Epirubicin cycle + Neupogen plan** |
| medication_plan | "Epirubicin 25mg/m2 D1,8,15 + Neupogen" | Plan | ✓ | - |
| Procedure_Plan | "Plan cycle#2 D1 Epirubicin..." | Plan | P1 | **化疗非 procedure。应为 "No procedures planned"** |
| Lab_Plan | "Labs liver functions, Tumor marker pending" | Plan | ✓ | - |
| follow up | "None" / "Not specified" | - | P2 | A/P 无明确 follow-up 时间 |
| Advance care | "Not discussed" | - | ✓ | - |

**本行总结**: 0×P0, 3×P1, 1×P2
- P1: Stage "Stage IIA" hallucination（7/7 LN+ ≥ Stage IIIA）
- P1: Medication_Plan_chatgpt 空
- P1: Procedure_Plan 列化疗方案（非 procedure）

---

### Row 92 (coral_idx=232)

**患者概况**: 53F，左乳 IDC ER-/PR-/HER2+，Stage I（多灶 0.21cm），node negative。S/p partial mastectomy + SLN + reconstruction。Plan: adjuvant Paclitaxel/Trastuzumab x12周 → Trastuzumab x9月。Video/Zoom visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | referred for evaluation |
| Type_of_Cancer | "ER-/PR-/HER2+ IDC" | pathology | ✓ | - |
| Stage_of_Cancer | "Stage I" | staging | ✓ | tiny tumors + node neg |
| therapy_plan | "Paclitaxel x12w + Trastuzumab x9m" | Plan | ✓ | 全面 |
| Medication_Plan_chatgpt | Paclitaxel/Trastuzumab | Plan | ✓ | 有内容 |
| Procedure_Plan | "mediport placement" | Plan | ✓ | - |
| Referral Specialty | "Radiation oncology consult" | Plan | ✓ | - |
| follow_up_next_visit | "today at 3pm rad onc" | Plan | P2 | 这是 rad onc appointment，非 oncology follow-up |
| Referral Others | "None" | - | P2 | 遗漏 Med Onc 转诊至 Dr. *** 处继续治疗 |

**本行总结**: 0×P0, 0×P1, 2×P2

---

### Row 93 (coral_idx=233)

**患者概况**: 75F，左乳 ER+/PR+/HER2- IDC，1.6cm，3 LN+，Oncotype RS=21。S/p lumpectomy → radiation → letrozole。Follow-up 讨论 imaging surveillance。Video visit。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | New Patient Evaluation at this center |
| Type_of_Cancer | "ER+/PR+ IDC" | staging | P2 | 缺 HER2-（原文有 "her 2 neu negative"） |
| Stage_of_Cancer | "Stage IIA" | staging form | ✓ | pT1b/pT1c + pN1 |
| current_meds | "letrozole" | med list | ✓ | - |
| supportive_meds | "letrozole" | - | P1 | **letrozole 是 hormonal therapy，非 supportive** |
| therapy_plan | "None" | - | P1 | **letrozole IS 当前 ongoing therapy** |
| Imaging_Plan | "Mammogram Nov 2021 + high risk MRI" | A/P | ✓ | - |
| response_assessment | "no evidence of recurrence" | imaging/exam | ✓ | - |
| Advance care | "full code." | - | ✓ | - |

**本行总结**: 0×P0, 2×P1, 1×P2
- P1: supportive_meds 列 letrozole（hormonal therapy 非 supportive）
- P1: therapy_plan "None"（当前 letrozole 治疗中）

---

### Row 94 (coral_idx=234)

**患者概况**: 49F，左乳 ER+/PR+/HER2- IDC。临床试验（Pembrolizumab arm）→ T-AC → lumpectomy（残余IDC + DCIS）→ 讨论 XRT vs ALND → capecitabine。Follow-up。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "in-person" | - | P1 | **字段值错误——"in-person" 应在 in-person 字段，Patient type 应为 "Follow up"** |
| Type_of_Cancer | "ER+/PR-/HER2- IDC with DCIS" | - | P1 | **原文 biopsy "ER/PR positive"，PR 应为阳性** |
| Stage_of_Cancer | "" (空) | - | P1 | **原文有 "Clinical st II/III"，不应为空** |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **Plan 有 capecitabine + endocrine therapy 讨论** |
| medication_plan | "Continue prilosec 40mg qd" | Plan | P2 | 仅列 GERD 药，遗漏主要治疗计划 |
| therapy_plan | "XRT → capecitabine → endocrine therapy" | Plan | ✓ | 全面 |
| radiotherapy_plan | "breast and axilla XRT, referred to Rad Onc" | Plan | ✓ | - |
| Referral Specialty | "Rad Onc" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**本行总结**: 0×P0, 4×P1, 1×P2
- P1: Patient type 字段值 "in-person" 错误
- P1: PR 标为阴性（原文 ER/PR 均阳性）
- P1: Stage 空（原文 "Clinical st II/III"）
- P1: Medication_Plan_chatgpt 空

---

### Row 95 (coral_idx=235)

**患者概况**: 47F，绝经前，左乳 ER+/PR+/HER2- 混合导管/筛状癌 1.8cm G1，pT1cN0。S/p partial mastectomy。Plan: Oncotype → radiation → tamoxifen。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | referred for initial evaluation |
| Type_of_Cancer | "ER+/PR+/HER2- mixed ductal and cribiform" | pathology | ✓ | 准确 |
| Stage_of_Cancer | "pT1cN0(sn)" | staging | ✓ | - |
| goals_of_treatment | "curative" | - | ✓ | - |
| Medication_Plan_chatgpt | tamoxifen after radiation | Plan | ✓ | 有内容 |
| medication_plan | "Tamoxifen after radiation" | Plan | ✓ | - |
| therapy_plan | "radiation → tamoxifen" | Plan | ✓ | 全面 |
| genetic_testing_plan | "send for [REDACTED] testing" | Plan | ✓ | MammaPrint/Oncotype |
| Referral Specialty | "Radiation oncology consult" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**本行总结**: 0×P0, 0×P1, 0×P2 — 非常干净

---

## Pattern Review: Rows 70-79

### 本批统计
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|----|
| 70 | 210 | 0 | 0 | 2 | second opinion, referral |
| 71 | 211 | 0 | 2 | 1 | current_meds garbled, Medication_Plan_chatgpt |
| 72 | 212 | 0 | 1 | 1 | supportive_meds误分类arimidex |
| 73 | 213 | 0 | 0 | 1 | TC展开错误 |
| 74 | 214 | 0 | 1 | 3 | Medication_Plan_chatgpt空, port/TTE/fertility遗漏 |
| 75 | 215 | 0 | 0 | 3 | supportive_meds不全, echo遗漏 |
| 76 | 216 | 0 | 0 | 1 | DEXA遗漏 |
| 77 | 217 | 0 | 2 | 1 | Stage IIA hallucination |
| 78 | 218 | 0 | 1 | 3 | Medication_Plan_chatgpt空, 交叉分类 |
| 79 | 219 | 0 | 2 | 2 | Stage IIA hallucination |

**本批合计**: 0×P0, 9×P1, 18×P2

### 累计统计
| 批次 | P0 | P1 | P2 | 合计 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 50 |
| Rows 10-19 | 1 | 25 | 29 | 55 |
| Rows 20-29 | 0 | 15 | 21 | 36 |
| Rows 30-39 | 0 | 12 | 18 | 30 |
| Rows 40-49 | 0 | 17 | 23 | 40 |
| Rows 50-59 | 0 | 13 | 19 | 32 |
| Rows 60-69 | 0 | 22 | 20 | 42 |
| Rows 70-79 | 0 | 9 | 18 | 27 |
| **总计** | **2** | **135** | **175** | **312** |

### 本批特点
1. **最低 P1 批次**: 9 P1 是目前最低，说明整体提取质量有改善趋势
2. **"Originally Stage IIA" hallucination 持续**: Row 77 (TNBC无原始分期)、Row 79 (DCIS=Stage 0 被标为 IIA)。这是系统性模型幻觉
3. **Hormonal therapy → supportive_meds 误分类**: Row 72 arimidex（继 Row 69 letrozole 后又一例）。AI 类药物系统性被误归类
4. **Medication_Plan_chatgpt 空率下降**: 本批 10 行中 4 行空 (Row 71, 74, 78, 79) = 40%

### 新系统性问题
- **TC 展开错误** (Row 73): TC in breast cancer = docetaxel + cyclophosphamide，模型展开为 docetaxel + carboplatin
- **Cross-field 分类持续**: thoracentesis 同时出现在 Imaging 和 Procedure（Row 78），receptor testing 出现在 genetic_testing_plan（Row 78）

---

### Row 96 (coral_idx=236)

**患者概况**: 53F，左乳 IDC G1 0.8cm，ER+/PR+/HER2-，pT1bN0(sn)。S/p left partial mastectomy + SLNB。多发性硬化症（MS）患者，服用 fingolimod (Gilenya)。Telehealth 初诊。Plan: Oncotype Dx → anticipate no chemo → Rad Onc eval → adjuvant endocrine therapy。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 初诊 |
| in-person | "Televisit" | - | ✓ | telehealth |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | pathology | P2 | **缺 HER2 状态**，原文 "ER/PR positive ***** negative" |
| Stage_of_Cancer | "pT1bN0(sn)" | - | ✓ | 0.8cm=pT1b, 0/3 SLN |
| goals_of_treatment | "curative" | - | ✓ | - |
| current_meds | "" (空) | - | ✓ | 无肿瘤药物（MS 药物被正确过滤） |
| Medication_Plan_chatgpt | hormonal therapy discussion | - | ✓ | 有内容 |
| medication_plan | "continue GILENYA regimen" | Plan | P2 | 提取了 MS 药物而非癌症药物（endocrine therapy 讨论被漏） |
| therapy_plan | "no need for chemo, future endocrine therapy" | Plan | ✓ | - |
| radiotherapy_plan | "None" | - | P1 | **A/P 明确说 "pt needs Rad Onc eval; was referred"**，应提取 |
| genetic_testing_plan | "None planned." | - | P1 | **A/P 明确说 "discussed molecular profiling, e.g. [Oncotype] Dx assay. Pt is interested. We will await result"** |
| Referral Specialty | "None" | - | P1 | **A/P 说 "pt needs Rad Onc eval; was referred"** = Rad Onc 转诊 |
| follow up | "3-4 wks after Dx result" | Plan | ✓ | - |
| Advance care | "Not discussed" | - | ✓ | - |

**白名单笔记**:
- fingolimod (Gilenya) 正确被过滤（MS 药物，非肿瘤药）

**本行总结**: 0×P0, 3×P1, 2×P2
- P1: radiotherapy_plan 漏 Rad Onc eval/referral
- P1: genetic_testing_plan 漏 Oncotype Dx（A/P 明确讨论并同意进行）
- P1: Referral 漏 Rad Onc referral
- P2: Type_of_Cancer 缺 HER2 状态
- P2: medication_plan 提取 MS 药物而非癌症计划

---

### Row 97 (coral_idx=237)

**患者概况**: 78F，TNBC（ER-/PR-/HER2-），IDC with apocrine features + HG DCIS 2.7cm，0/3 nodes。当前 Taxotere/Cytoxan C4。低热、皮疹、贫血（Hgb 9.7）、血小板极高（1052）。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | Cycle #4 |
| Type_of_Cancer | "ER-/PR-/HER2- IDC with apocrine features, triple negative" | pathology | ✓ | 完整 |
| Stage_of_Cancer | "Not mentioned in note" | - | P2 | 可推断但原文确实未明确分期 |
| current_meds | "Docetaxel, Cyclophosphamide" | med list | ✓ | - |
| supportive_meds | "ondansetron, prochlorperazine, dexamethasone" | med list | P2 | **漏 docusate (Colace)**，A/P 还说 "Stable with Docusate Sodium daily" |
| goals_of_treatment | "curative" | - | ✓ | adjuvant |
| response_assessment | "Not mentioned in note" | - | P2 | A/P 说 "Exam stable"，虽非正式评估但有信息 |
| therapy_plan | "proceed with TC, refer to Rad Onc" | Plan | ✓ | 全面 |
| radiotherapy_plan | "Refer to Rad Onc for second opinion" | Plan | ✓ | - |
| Procedure_Plan | "Port removal" | Plan | ✓ | - |
| Referral Specialty | "Rad Onc, port removal" | Plan | ✓ | - |
| follow up | "1 month to discuss after radiation" | Plan | ✓ | - |

**本行总结**: 0×P0, 0×P1, 3×P2
- P2: Stage 未明确（可推断）
- P2: supportive_meds 漏 docusate（bowel regimen）
- P2: response_assessment "Not mentioned" 但 A/P 有 "Exam stable"

---

### Row 98 (coral_idx=238)

**患者概况**: 49F，双侧乳腺癌：左乳 Stage III ER+/PR+/HER2-（2009），右乳 Stage I ER-/PR-/HER2 3+（2010 prophylactic mastectomy 意外发现）。2012年发现肺肿块+纵隔淋巴结肿大，疑似转移。当前 fulvestrant，PET/CT 示肺病灶增大。需要活检明确转移灶生物特性。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "New patient" | - | ✓ | 转诊至 UCSF |
| Type_of_Cancer | "ER+/PR+/HER2- IDC, ER-/PR-/HER2 3+ IDC" | history | ✓ | 双原发 |
| Stage_of_Cancer | "Originally Stage III, now metastatic (Stage IV)" | A/P | P2 | "Originally Stage III" 表述虽在此案有原文支持（A/P 明确说 "stage III"），但与系统性幻觉模式一致 |
| current_meds | "fulvestrant" | treatment | ✓ | 当前用药 |
| recent_changes | "anastrozole → letrozole" | history | P2 | **历史性换药**，非近期变化 |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | "progressing" | PET/CT | ✓ | 肺病灶增大 |
| Procedure_Plan | "biopsy of pulmonary nodule/mediastinal/subpleural" | A/P | ✓ | - |
| Imaging_Plan | "CT with contrast and thin slices" | A/P | ✓ | - |
| Referral Specialty | "Thoracic surgery, IR" | A/P | P2 | **漏 symptom management service** referral（A/P 明确说 "placed a referral to our symptom management service"） |
| follow up | "2 weeks" | A/P | ✓ | - |

**本行总结**: 0×P0, 0×P1, 3×P2
- P2: Stage 含 "Originally Stage III"（虽有原文支持但属系统性模式）
- P2: recent_changes 列历史性换药
- P2: Referral 漏 symptom management service

---

### Row 99 (coral_idx=239)

**患者概况**: 68F，转移性乳腺癌（ER 80%/PR 50%/HER2-），广泛肝转移+多处转移。原发 2002 年，2011 年复发。经历 Abraxane+bevacizumab → faslodex → xeloda → 当前 Gemzar C2 Day 8（患者取消）。肿瘤标志物升高（CEA 319.9, CA 15-3 118），疲劳严重。

**字段审查**:
| 字段 | 提取值（摘要） | 归因引用 | 判定 | 问题 |
|------|--------------|---------|------|------|
| Patient type | "Follow up" | - | ✓ | 治疗中 |
| Type_of_Cancer | "ER+(80%)PR+(50%)HER2- IDC" | history | ✓ | 准确 |
| Stage_of_Cancer | "Originally not specified, now metastatic (Stage IV)" | - | P2 | "Originally not specified" 是系统性幻觉模式 |
| Metastasis | "Yes, liver and multiple sites" | history | ✓ | - |
| lab_summary | 详细 CBC+化学+肿瘤标志物 | labs | ✓ | 全面 |
| current_meds | "" (空) | - | P1 | **Gemzar (gemcitabine) 是当前化疗方案**，C2 进行中（虽 Day 8 取消，仍在疗程中） |
| supportive_meds | "docusate, oxycodone, senna" | med list | ✓ | bowel regimen + pain |
| goals_of_treatment | "palliative" | - | ✓ | Stage IV |
| response_assessment | "TMs elevated, unclear if progressing or tumor flare" | A/P | ✓ | 准确反映 A/P |
| Medication_Plan_chatgpt | {} (空) | - | P1 | **A/P 讨论了继续 Gemzar、Focalin prn、运动建议** |
| therapy_plan | "None" | - | P1 | **A/P 说 "continue with treatment"（Gemzar）**，不应为 None |
| medication_plan | "Focalin prn for fatigue" | A/P | P2 | 只提 Focalin，漏 continue Gemzar |
| Procedure_Plan | "No procedures planned" | - | ✓ | - |
| Imaging_Plan | "No imaging planned" | - | ✓ | scan 已做 |
| follow up | "discuss break with Dr [REDACTED]" | A/P | ✓ | - |

**白名单笔记**:
- Focalin (dexmethylphenidate): 非肿瘤药物（精神兴奋剂），可加入 non_oncology_drugs.txt

**本行总结**: 0×P0, 3×P1, 2×P2
- P1: current_meds 空（应有 gemcitabine/Gemzar）
- P1: therapy_plan "None"（应提 continue Gemzar）
- P1: Medication_Plan_chatgpt 空
- P2: Stage 含 "Originally not specified" 幻觉模式
- P2: medication_plan 只提 Focalin，漏 continue Gemzar

---

## Pattern Review: Rows 90-99

### 本批统计
| Row | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|----|----|
| 90 | 230 | 0 | 2 | 4 | supportive_meds 列主药, response_assessment 旧数据 |
| 91 | 231 | 0 | 3 | 1 | Stage IIA 幻觉, Procedure_Plan 列化疗 |
| 92 | 232 | 0 | 0 | 2 | 干净 |
| 93 | 233 | 0 | 2 | 1 | supportive_meds 列 letrozole, therapy_plan None |
| 94 | 234 | 0 | 4 | 1 | Patient type, PR 错误, Stage 空 |
| 95 | 235 | 0 | 0 | 0 | 非常干净 |
| 96 | 236 | 0 | 3 | 2 | genetic_testing_plan 漏 Oncotype, Referral 漏 Rad Onc |
| 97 | 237 | 0 | 0 | 3 | Stage 未明确, supportive_meds 漏 docusate |
| 98 | 238 | 0 | 0 | 3 | recent_changes 历史性, Referral 漏 symptom mgmt |
| 99 | 239 | 0 | 3 | 2 | current_meds 空, therapy_plan None |

**本批合计**: 0×P0, 17×P1, 19×P2

### 累计统计
| 批次 | P0 | P1 | P2 | 合计 |
|------|----|----|----|----|
| Rows 0-9 | 1 | 22 | 27 | 50 |
| Rows 10-19 | 1 | 25 | 29 | 55 |
| Rows 20-29 | 0 | 15 | 21 | 36 |
| Rows 30-39 | 0 | 12 | 18 | 30 |
| Rows 40-49 | 0 | 17 | 23 | 40 |
| Rows 50-59 | 0 | 13 | 19 | 32 |
| Rows 60-69 | 0 | 22 | 20 | 42 |
| Rows 70-79 | 0 | 9 | 18 | 27 |
| Rows 80-89 | 0 | 20 | 24 | 44 |
| Rows 90-99 | 0 | 17 | 19 | 36 |
| **总计** | **2** | **172** | **218** | **392** |

### 本批特点
1. **P1 下降至 17**: 与 Rows 80-89 的 20 相比有所改善
2. **Medication_Plan_chatgpt 空率**: 10 行中 ~3-4 行空，≈ 30-40%
3. **therapy_plan "None" 持续问题**: Row 93 (letrozole 在 continue)、Row 99 (Gemzar 在 continue)，模型将 "continue" 现有治疗不视为 therapy plan
4. **current_meds 空但有活跃治疗**: Row 99 gemcitabine 漏提取

### 新系统性问题
- **genetic_testing_plan 漏 Oncotype Dx**: Row 96 A/P 明确讨论分子检测并同意，但提取为 "None planned"
- **Referral 漏 symptom management**: Row 98 A/P 明确下了转诊单
- **medication_plan 提取非肿瘤药物**: Row 96 提取了 MS 药物 Gilenya 而非肿瘤治疗计划

---

## 全局总结（100 行审查完成）

### 最终统计
- **P0（严重/幻觉）**: 2 例（Row 5: 幻觉编造肿瘤大小；Row 13: 将历史用药编为当前方案）
- **P1（显著错误）**: 172 例
- **P2（轻微/修饰）**: 218 例
- **总计**: 392 个问题，100 行

### Top 10 系统性问题

| 排名 | 问题 | 受影响行数 | 严重度 | 根因 |
|------|------|-----------|--------|------|
| 1 | **Medication_Plan_chatgpt 空输出** | ~45 行 | P1 | 复杂嵌套 JSON schema 超出 8B 模型能力 |
| 2 | **Stage "Originally Stage IIA" 幻觉** | ~12 行 | P1 | 模型在无原始分期时编造 "Originally Stage IIA" |
| 3 | **Cross-field 分类错误** | ~30 行 | P1-P2 | 化疗→Procedure_Plan, 影像→Lab_Plan, LP→Lab_Plan, IHC/FISH→Procedure_Plan |
| 4 | **Hormonal therapy → supportive_meds** | ~10 行 | P1 | letrozole/arimidex/exemestane 被归为支持性用药 |
| 5 | **current_meds 时态混乱** | ~15 行 | P1 | 过去/已停/已 PD 药物被列为当前用药，或当前活跃治疗被漏 |
| 6 | **genetic_testing_plan 遗漏** | ~15 行 | P1 | Oncotype Dx, MammaPrint, BRCA 等未被识别为基因检测 |
| 7 | **radiotherapy_plan 时态过滤失败** | ~10 行 | P1-P2 | 已完成放疗未被 G5 TEMPORAL 过滤 |
| 8 | **therapy_plan "None" 但有活跃治疗** | ~8 行 | P1 | "continue" 现有方案不被视为 therapy plan |
| 9 | **Referral 遗漏** | ~12 行 | P1-P2 | Rad Onc, social work, symptom management 等转诊被漏 |
| 10 | **response_assessment 答非所问** | ~10 行 | P2 | 写症状/计划/副作用而非治疗响应 |

### 字段准确率估算

| 字段 | 完全正确 | P2 | P1 | P0 | 准确率 |
|------|---------|----|----|----|----|
| Patient type | ~85 | ~5 | ~10 | 0 | 85% |
| Type_of_Cancer | ~75 | ~15 | ~10 | 0 | 75% |
| Stage_of_Cancer | ~60 | ~20 | ~20 | 0 | 60% |
| Metastasis | ~90 | ~5 | ~5 | 0 | 90% |
| Distant Metastasis | ~88 | ~5 | ~7 | 0 | 88% |
| lab_summary | ~85 | ~10 | ~5 | 0 | 85% |
| findings | ~75 | ~15 | ~10 | 0 | 75% |
| current_meds | ~70 | ~10 | ~18 | 2 | 70% |
| supportive_meds | ~72 | ~15 | ~13 | 0 | 72% |
| goals_of_treatment | ~88 | ~5 | ~7 | 0 | 88% |
| response_assessment | ~65 | ~20 | ~15 | 0 | 65% |
| Medication_Plan_chatgpt | ~50 | ~5 | ~45 | 0 | 50% |
| medication_plan | ~70 | ~15 | ~15 | 0 | 70% |
| therapy_plan | ~78 | ~12 | ~10 | 0 | 78% |
| radiotherapy_plan | ~80 | ~10 | ~10 | 0 | 80% |
| Procedure_Plan | ~82 | ~10 | ~8 | 0 | 82% |
| Imaging_Plan | ~85 | ~10 | ~5 | 0 | 85% |
| Lab_Plan | ~88 | ~7 | ~5 | 0 | 88% |
| genetic_testing_plan | ~80 | ~5 | ~15 | 0 | 80% |
| Referral | ~75 | ~12 | ~13 | 0 | 75% |
| follow up | ~90 | ~8 | ~2 | 0 | 90% |
| Advance care | ~92 | ~5 | ~3 | 0 | 92% |

### 最高优先级改进建议

1. **简化 Medication_Plan_chatgpt schema** — 嵌套 JSON 导致 ~45% 空率，改为扁平结构
2. **Stage 推断 prompt 改进** — 禁止 "Originally Stage IIA" 表述，要求有原文支持才写 original stage
3. **supportive_meds 黑名单** — 将所有 hormonal therapy 药物加入 supportive_care 排除列表
4. **therapy_plan prompt** — 明确 "continue [drug]" = 有 therapy plan，不是 "None"
5. **genetic_testing_plan 关键词扩展** — 加入 Oncotype, MammaPrint, molecular profiling, BRCA, germline 等触发词

---

## 白名单更新建议

### oncology_drugs.txt 新增
- fulvestrant / faslodex（多行出现：Row 31, 40, 84, 98）
- everolimus / Afinitor（Row 90）
- gemcitabine / Gemzar（Row 99）
- abraxane / nab-paclitaxel（Row 99 history）
- bevacizumab / Avastin（Row 99 history）
- eribulin / Halaven（Row 84 history）
- vinorelbine / Navelbine（多行）
- ixabepilone（Row 84 history）

### supportive_care_drugs.txt 排除（不应在此名单）
- letrozole（Row 69, 93: hormonal therapy 被误归为 supportive）
- anastrozole / arimidex（Row 72: hormonal therapy 被误归为 supportive）
- exemestane（Row 90: hormonal therapy 被误归为 supportive）
- tamoxifen（Row 15: hormonal therapy 被误归为 supportive）
- trastuzumab / Herceptin（Row 90: targeted therapy 被误归为 supportive）

### non_oncology_drugs.txt 新增
- fingolimod / Gilenya（MS 药物，Row 96）
- Focalin / dexmethylphenidate（精神兴奋剂，Row 99）
- gabapentin / Neurontin（神经痛/MS，Row 96）
- lamotrigine / Lamictal（抗癫痫/MS，Row 96）
- escitalopram / Lexapro（抗抑郁，Row 96）
- bupropion / Wellbutrin（抗抑郁，Row 96）
- trazodone / Desyrel（睡眠，Row 96）
- clonazepam / Klonopin（焦虑，Row 96）
- atorvastatin / Lipitor（降脂，Row 96）
- levothyroxine / Synthroid（甲状腺，Row 99）
- pregabalin / Lyrica（神经痛，Row 99）
- melatonin（睡眠辅助，Row 96）
- nicotine patch / Nicoderm（戒烟，Row 96）

### 过滤规则改进建议

**Procedure_Plan 黑名单关键词**:
- "chemotherapy", "TCHP", "FOLFOX", "AC-T", "TC" → 属 Therapy_plan
- "IHC", "FISH", "receptor testing" → 属 Genetic_Testing_Plan
- "radiation", "XRT", "RT" → 属 radiotherapy_plan
- "CT", "MRI", "PET", "DEXA", "echo", "mammogram" → 属 Imaging_Plan

**Lab_Plan 黑名单关键词**:
- "lumbar puncture", "LP" → 属 Procedure_Plan
- "PET", "CT", "MRI", "DEXA" → 属 Imaging_Plan
- "Oncotype", "MammaPrint", "BRCA" → 属 Genetic_Testing_Plan

**Imaging_Plan 黑名单关键词**:
- "biopsy", "thoracentesis" → 属 Procedure_Plan

**radiotherapy_plan 时态过滤加强**:
- "had radiation", "s/p radiation", "completed radiation", "had SRS" → PAST，应过滤
- 当前 G5 TEMPORAL gate 对 radiotherapy_plan 的过滤不够严格

**Stage 推断改进**:
- 禁止 "Originally Stage IIA" 除非原文明确写出
- T2N2a = Stage IIIA（非 Stage II）
- axillary LN = regional, NOT distant metastasis

**TC 展开规则**:
- 乳腺癌中 TC = docetaxel + cyclophosphamide（NOT carboplatin）
- 仅在明确注明 carboplatin 时才展开为 docetaxel + carboplatin

