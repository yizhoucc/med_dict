# V14 逐行审查

审查对象: `results/v14_verify_20260316_222417_progress.json` (61 行)
对比基线: `results/full_qwen_20260316_075405/review_v2.md` (v13a 审查)
审查重点: (1) v13a 问题是否修复 (2) 是否引入新问题 (3) 归因质量

v14 改进项:
1. 删除 Medication_Plan_chatgpt
2. 修复 supportive_meds 过滤
3. Stage 幻觉修复
4. Cross-field 白名单过滤
5. therapy_plan "continue" 修复
6. genetic_testing_plan 白名单搜索
7. Referral 全文提取
8. current_meds 时态 prompt

---

## Row 0 (coral_idx=140)

**患者概况**: 56F, 2013年右乳多灶性IIA期ER+/PR+/HER2-乳癌,行乳切+SLN+植入物重建。拒绝tamoxifen,未化疗/放疗。2019.12 CT发现广泛转移(肺、腹膜、肝、卵巢+右腋窝局部复发)。初次肿瘤内科会诊。

### v13a 已知问题
- P0: Specialty referral 文本泄漏 ("History of Present Illness: 56")
- P1: (1) Type_of_Cancer 缺 HER2- (2) response_assessment 答非所问 (3) imaging_plan 遗漏 MRI+bone scan (4) lab_plan 遗漏 CBC/CMP/markers
- P2: 多个归因不精确

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "New patient" | ✓ | 同 |
| second opinion | "no" | ✓ | 同 |
| in-person | "in-person" | ✓ | 同 |
| summary | 详细准确(56F, metastatic ER+ breast cancer, 转移部位) | ✓ | 同 |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | **P1 未修** | 仍缺HER2-。原文"her 2 neu negative"。归因引用包含"her 2 negative"但值遗漏 |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | ✓ | 同。原文HPI"multifocal Stage IIA",有支持 |
| Metastasis | "Yes (to lungs, peritoneum, liver, and ovaries)" | ✓ | 同 |
| Distant Metastasis | "Yes (to lungs, peritoneum, liver, and ovaries)" | ✓ | 同。腋窝=regional,正确排除 |
| lab_summary | "No labs in note." | ✓ | 同 |
| findings | CT findings + 体检发现(肝大/腋窝3cm肿块),非常全面 | ✓ | 同 |
| current_meds | "" | ✓ | 同("No current outpatient medications on file") |
| recent_changes | "" | ✓ | 同 |
| supportive_meds | "" | ✓ | 同 |
| goals_of_treatment | "palliative" | ✓ | 同 |
| response_assessment | 长文重复CT findings | **P1 未修** | 仍答非所问。未开始系统治疗(仅2013手术+拒绝tamoxifen),应为"Not yet on treatment" |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a有此字段,v14已删除 |
| medication_plan | "If HR+/HER2 negative on biopsy, will recommend ibrance and [REDACTED]." | ✓ | 同 |
| therapy_plan | "If the patient is HR+/HER2 negative on biopsy, will recommend ibrance and [REDACTED]." | ✓ | 同 |
| radiotherapy_plan | "None" | ✓ | 同 |
| procedure_plan | "Appointment with Dr. [REDACTED]...to biopsy mass in right axilla" | ✓ | 同 |
| imaging_plan | "No imaging planned." | **P1 未修** | HPI"I also ordered a MRI of brain and bone scan",Diagnosis列出bone scan+MRI brain。A/P未提→pipeline限制 |
| lab_plan | "No labs planned." | **P1 未修** | Diagnosis列出CBC, CMP, CA15-3, CEA, aPTT, PT。A/P未提→pipeline限制 |
| genetic_testing_plan | "No specific genetic...plan is to biopsy...to confirm HR and HER2 status." | P2 | 无genetic test正确,但不必要提及biopsy |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "None" | ✓ | 同 |
| Referral-Specialty | "Integrative Medicine, Integrative Medicine       History of Present Illness:   56" | **P0 未修** | 文本泄漏仍在。应为"Integrative Medicine, [REDACTED] Referral" |
| Referral-Others | "None" | ✓ | 同 |
| follow up | "RTC with me after completed work up to formulate a plan." | ✓ | 同 |
| Next clinic visit | "in-person: after completed work up to formulate a plan" | ✓ | 同 |
| Advance care | "Full code." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| Patient type | "RTC with me after completed work up to formulate a plan." | **P2 错误** — 应引用"Patient presents with New Patient Evaluation" |
| second opinion | "RTC with me after completed work up to formulate a plan." | **P2 错误** — 应引用patient history或"no"的否定依据 |
| in-person | "RTC with me after completed work up to formulate a plan." | **P2 错误** — 应引用体检或"TIME SPENT: 75 minutes examining patient" |
| summary | "Metastatic relapse of her breast cancer. She needs tissue confirmation..." | ✓ 合理 |
| Type_of_Cancer | "2.4 and 2.3 cm tumors, node negative ER and PR positive and her 2 negative." | ✓ 相关(但值漏了HER2-) |
| Stage_of_Cancer | "Metastatic relapse of her breast cancer." | P2 — 应引用HPI"Stage IIA"或staging form |
| Metastasis | "Metastatic relapse of her breast cancer." | P2 — 太笼统,应引用CT findings |
| Distant Metastasis | "Metastatic relapse of her breast cancer." | P2 — 同上 |
| findings | "Widespread metastases consistent with metastatic breast carcinoma." | ✓ CT impression开头 |
| goals_of_treatment | "Metastatic relapse...tissue confirmation..." | P2 — 应引用HPI"treatment would be palliative" |
| response_assessment | "Metastatic relapse of her breast cancer." | P2 — 值本身就有问题 |
| medication_plan | "If she is HR+/ her 2 negative on biopsy will recommend ibrance and [REDACTED]." | ✓ 精确匹配A/P |
| therapy_plan | 同上 | ✓ |
| procedure_plan | "Appointment with Dr. [REDACTED]...to biopsy mass in right axilla in the office." | ✓ 精确 |
| genetic_testing_plan | "Appointment with Dr. [REDACTED]...to biopsy mass in right axilla..." | P2 — biopsy不是genetic testing |
| Next clinic visit | "RTC with me after completed work up to formulate a plan." | ✓ |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓
- **未修复**: P0 Specialty referral泄漏, P1 Type_of_Cancer缺HER2-, P1 response_assessment答非所问, P1 imaging_plan遗漏, P1 lab_plan遗漏
- **新问题**: 无
- **归因质量**: 差。Patient type/second opinion/in-person都引用了无关的"RTC"文本(3个P2归因错误)
- **统计**: 1×P0, 4×P1, 7×P2归因

---

## Row 1 (coral_idx=141)

**患者概况**: 44F, Lynch综合征,多重癌症:①2013右乳TNBC(ER-/PR-/HER2-) IIB期→新辅助化疗+部分乳切+辅助化疗;②2018.3右胸壁TNBC复发→gem/carb有效;③2018.5结肠脾曲腺癌I期→结肠切除;④子宫内膜样腺癌FIGO1。2019.1 PET示转移(肝、骨、胸壁)→Abraxane+Pembro→进展→2019.6.30开始irinotecan。本次:cycle 3 day 1(2019.8.11)。

### v13a 已知问题
- P1: (1) response_assessment使用旧影像 (2) Medication_Plan_chatgpt空 (3) medication_plan遗漏irinotecan剂量变更
- P2: (1) Distant Metastasis含胸壁(local) (2) therapy_plan含oxycodone (3) Referral-Others含不确定转诊 (4) response_assessment归因幻觉

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "Follow up" | ✓ | 同 |
| second opinion | "no" | ✓ | 同 |
| in-person | "in-person" | ✓ | 同 |
| summary | "44 y.o. female with metastatic breast cancer, presenting for cycle 3 day 1 of irinotecan treatment." | ✓ | 同,准确具体 |
| Type_of_Cancer | "ER-/PR-/HER2- triple negative invasive ductal carcinoma" | ✓ | 同 |
| Stage_of_Cancer | "Originally Stage IIB, now metastatic (Stage IV)" | ✓ | 同 |
| Metastasis | "Yes (to liver, bone, and chest wall)" | ✓ | 同 |
| Distant Metastasis | "Yes (to liver, bone, and chest wall)" | **P2 未修** | 胸壁=local recurrence不算distant,但不影响临床 |
| lab_summary | 15项异常值(Alb 2.1, ALP 183, Na 124等) | ✓ | 同,全面 |
| findings | 非常详尽:胸壁感染、背痛加重、HepB、MRI骨病灶 | ✓ | 同 |
| current_meds | "" | **P1 新问题** | v13a有"irinotecan",v14变空!原文medication list只有支持用药,但A/P明确irinotecan cycle 3。可能是改进8(cross-reference medication list)导致——medication list中没有irinotecan(化疗不在outpatient meds中)→被排除。**回归** |
| recent_changes | "Changed irinotecan to every other week and increased dose to 150mg/m2..." | ✓ | 同 |
| supportive_meds | "ondansetron, prochlorperazine, loperamide, morphine, oxycodone" | ✓ | v13a也有这些,过滤正确 |
| goals_of_treatment | "palliative" | ✓ | 同 |
| response_assessment | "chest wall more tender...Back pain is worse which could be due to disease progression." | **✓ 改善** | v13a使用旧PET,v14使用当前临床征象(胸壁变化+背痛加重→可能PD)。**好得多** |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a为空(P1),现在字段不存在 |
| medication_plan | "Doxycycline, MS Contin, flexeril, oxycodone, Effexor-XR, NS IV, KCl, pRBC" | ✓ 改善 | v13a遗漏irinotecan剂量变更(P1)。v14仍无irinotecan在medication_plan中,但irinotecan更属于therapy_plan。其余都准确 |
| therapy_plan | "irinotecan...changed to every other week with increased dose to 150mg/m2 on days 1 and 15 every 28 days." | ✓ | 同,准确 |
| radiotherapy_plan | "She urgently needs to get in with Rad Onc..." | ✓ | 同 |
| procedure_plan | "No procedures planned." | ✓ | 同 |
| imaging_plan | "Scans again in 3 months, due September 2019. MRI brain if worse." | ✓ | 同 |
| lab_plan | "Monitor with ALT, HBsAg and HBV DNA every 4 months, next due October 2019" | ✓ | 同 |
| genetic_testing_plan | "None planned." | ✓ | 同 |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "None" | ✓ | 同 |
| Referral-Specialty | "Rad Onc consult" | ✓ | 同 |
| Referral-Others | "Social work referral, Home health referral" | P2 | v13a也有。"Social work"是re-referring(不确定);"Home health?"是问号(考虑中) |
| follow up | "2 weeks" | ✓ | 同 |
| Next clinic visit | "in-person: 2 weeks" | ✓ | 同 |
| Advance care | "Not discussed during this visit." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| Patient type | "F/u 2 weeks" | ✓ 合理 |
| second opinion | "Referral made to SMS/ABC made previously, but never go auth." | P2 — 不相关 |
| in-person | "I spent a total of 45 minutes face-to-face with the patient..." | ✓ 精确 |
| summary | A/P开头文本 | ✓ 精确 |
| Type_of_Cancer | "metastatic breast cancer" | P2 — 太笼统,应引用"ER-/PR-/[REDACTED]- (TNBC)" |
| Stage_of_Cancer | "newly diagnosed metastatic breast cancer" | P2 — 应引用"Stage IIB" |
| Metastasis | "Chest wall more tender, erythematous, and edematous" | P2 — 这是感染征象不是转移证据 |
| Distant Metastasis | 同上 | P2 |
| lab_summary | 实验室结果原文 | ✓ |
| findings | "Chest wall more tender..." | ✓ |
| recent_changes | "will change her irinotecan to every other week..." | ✓ 精确 |
| supportive_meds | "Continue [REDACTED] 30mg TID and flexeril TID. Continue oxycodone 5mg prn" | ✓ |
| goals_of_treatment | "metastatic breast cancer" | P2 — 太笼统 |
| response_assessment | "chest wall more tender...Back pain is worse which could be due..." | ✓ 准确 |
| medication_plan | "Rx Doxycycline...Continue [REDACTED]...Increase effexor-XR..." | ✓ 全面 |
| therapy_plan | "will change her irinotecan..." | ✓ 精确 |
| radiotherapy_plan | "She urgently needs to get in with Rad Onc..." | ✓ |
| imaging_plan | "Scans again in 3 months...MRI brain if worse" | ✓ |
| lab_plan | "monitor with ALT, hepatitis B surface antigen and HBV DNA..." | ✓ |
| Specialty | "so I will send a message to Dr. [REDACTED]..." | ✓ |
| Others | "Referral made to SMS/ABC...Home health?" | ✓ 准确引用 |
| follow up | "F/u 2 weeks" | ✓ |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓, response_assessment显著改善(从旧影像→当前临床征象)
- **未修复**: P2 Distant Metastasis含胸壁
- **新问题**: **P1 current_meds变空**(v13a有irinotecan)——疑似改进8(cross-reference medication list)导致回归。化疗药不在outpatient medication list中
- **归因质量**: 中等。计划相关字段归因好,诊断相关字段归因偏笼统
- **统计**: 0×P0, 1×P1(新), 6×P2归因

---

## Row 2 (coral_idx=142)

**患者概况**: 53F, 乳腺X线发现右乳上外象限癌。IDC 1.7cm, 腋窝LN 1.5cm阳性。ER+/PR+/HER2 IHC2+ FISH neg(=HER2-)。Stage IIA。多处就诊seeking opinions,本次video远程肿瘤内科会诊。PET CT和基因检测已安排/待结果。

### v13a 已知问题
- P1: Referral-Genetics 误把基因检测order当转诊
- P2: (1) second opinion归因不相关 (2) findings重复receptor status (3) imaging_plan措辞有歧义

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "New patient" | ✓ | 同 |
| second opinion | "yes" | ✓ | 同。"She has had several opinions" |
| in-person | "Televisit" | ✓ | 同 |
| summary | "53F, Stage IIA right breast IDC, ER+, HER2 negative, node positive. Consult for neoadjuvant." | ✓ | 同,优秀 |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | ✓ | 同 |
| Stage_of_Cancer | "Stage IIA" | ✓ | 同 |
| Metastasis | "No" | ✓ | 同 |
| Distant Metastasis | "No" | ✓ | 同 |
| lab_summary | "No labs in note." | ✓ | 同 |
| findings | 1.7cm肿瘤 + 1.5cm LN + receptor + PET pending | ✓/P2 | 同。findings重复receptor info |
| current_meds | "" | ✓ | 同 |
| recent_changes | "" | ✓ | 同 |
| supportive_meds | "" | ✓ | 同 |
| goals_of_treatment | "curative" | ✓ | 同 |
| response_assessment | "Not yet on treatment — no response to assess." | ✓ | 同 |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a有此字段(chemo+hormonal讨论) |
| medication_plan | "None" | ✓ | 同 |
| therapy_plan | "Discussed chemo + surgery + radiation roles" | ✓ | 同 |
| radiotherapy_plan | "discussed the role of surgery and radiation" | ✓ | 同 |
| procedure_plan | "No procedures planned." | ✓ | 同 |
| imaging_plan | "PET imaging follow up after results are back." | ✓/P2 | 同。措辞有歧义(PET已做还是计划做?) |
| lab_plan | "No labs planned." | ✓ | 同 |
| genetic_testing_plan | "Genetic testing sent and is pending." | ✓ | 同 |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "Genetic testing sent and is pending." | **P1 未修** | 仍把基因检测order当genetics clinic referral |
| Referral-Specialty | "None" | ✓ | 同 |
| Referral-Others | "None" | ✓ | 同 |
| follow up | "[REDACTED] follow up after pet and [REDACTED] are back." | ✓ | 同 |
| Next clinic visit | "telehealth: after PET and [REDACTED] are back" | ✓ | 同 |
| Advance care | "full code." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| second opinion | "[REDACTED] sent and is pending." | **P2** — 不相关 |
| in-person | "I performed this evaluation using real-time telehealth tools..." | ✓ 精确 |
| summary | "Stage IIA right IDC of the breast 1.7 cm, node positive, HR+, her 2 neu negative by FISH." | ✓ A/P#1 |
| Type_of_Cancer | 同上 | ✓ |
| Stage_of_Cancer | 同上 | ✓ |
| Metastasis | 同上 | P2 — 虽然文本正确,但这是诊断信息不是转移证据 |
| goals_of_treatment | "discussed the role of surgery and radiation..." | ✓ 合理 |
| response_assessment | "Not yet on treatment — no response to assess." | ✓ |
| therapy_plan | "Discussed the role of chemotherapy..." | ✓ |
| radiotherapy_plan | "the role of surgery and radiation..." | ✓ |
| imaging_plan | "[REDACTED] follow up after pet and [REDACTED] are back." | ✓ |
| genetic_testing_plan | "Genetic testing sent and is pending." | ✓ |
| Genetics | "Genetic testing sent and is pending." | ✓ (但值本身应为None) |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓
- **未修复**: P1 Referral-Genetics(基因检测order≠genetics clinic referral)
- **新问题**: 无
- **归因质量**: 好。大部分归因精确引用A/P文本,少量P2
- **统计**: 0×P0, 1×P1, 3×P2

---

## Row 4 (coral_idx=144)

**患者概况**: 31F(绝经前), 2013左乳8cm G2 IDC micropapillary(ER+/PR+/HER2-, Ki67 5%), Stage III, 16LN+/ECE。双乳切+腋清。AC×3(副作用停)。2015 Lupron+exemestane/tamoxifen。2018.6左乳复发(IDC ER+/PR+/HER2- IHC1+, Ki67 30-40%)。2019.1 PET→颈LN活检证实转移。当前Lupron+anastrozole+palbociclib。本次12/7/2019 video follow-up。

### v13a 已知问题
- P0: Specialty referral文本泄漏(含imaging order)
- P1: Medication_Plan_chatgpt空
- P2: summary偏泛化、lab_summary遗漏creatinine、therapy_plan含labs、follow up混杂

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "Follow up" | ✓ | 同 |
| second opinion | "no" | ✓ | 同 |
| in-person | "Televisit" | ✓ | 同 |
| summary | "recurrent breast cancer...follow-up to continue therapy with ibrance/lupron, manage symptoms, review imaging" | ✓ 改善 | v13a偏泛化,v14更具体(含药物名) |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | ✓ | 同 |
| Stage_of_Cancer | "Originally Stage III, now metastatic (Stage IV)" | ✓ | 同 |
| Metastasis | "Yes (to left cervical LN, left IM LN, and sternum)" | ✓ | 同 |
| Distant Metastasis | 同上 | ✓ | 同 |
| lab_summary | "Creatinine 0.6 mg/dL (from 08/23/2019)" | ✓ 改善 | v13a为空("No labs")P2,v14找到了creatinine |
| findings | MRI脑/颈+CT+bone scan+体检,非常全面 | ✓ | 同 |
| current_meds | "anastrozole, palbociclib" | **P1 新问题** | v13a有"anastrozole, palbociclib, leuprolide",v14丢失leuprolide!A/P明确"on leuprolide",但lupron是clinic注射不在outpatient med list→疑似改进8(cross-reference)导致回归 |
| recent_changes | "" | ✓ | 同("Continue current therapy") |
| supportive_meds | "ondansetron (ZOFRAN) 8 mg tablet" | ✓ | 同 |
| goals_of_treatment | "palliative" | ✓ | 同 |
| response_assessment | CT颈+bone scan+MRI颈椎的mixed response详细信息 | ✓ | 同 |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a空(P1) |
| medication_plan | "Continue leuprolide, anastrozole, and palbociclib." | ✓ | 同 |
| therapy_plan | "Continue current therapy...Radiation referral...Labs monthly" | ✓/P2 | 同,P2 labs不属于therapy |
| radiotherapy_plan | "Radiation oncology referral for symptomatic disease..." | ✓ | 同 |
| procedure_plan | "No procedures planned." | ✓ | 同 |
| imaging_plan | "CT and bone scan ordered...prior to next visit." | ✓ | 同 |
| lab_plan | "Labs monthly. On the day of lupron injection." | ✓ | 同 |
| genetic_testing_plan | "None planned." | ✓ | 同 |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "None" | ✓ | 同 |
| Referral-Specialty | "Radiation oncology referral, Radiation Oncology    CT Abdomen /Pelvis with Contrast    CT" | **P0 未修** | 文本泄漏仍在。同Row 0——系统性问题 |
| Referral-Others | "None" | ✓ | 同 |
| follow up | 混杂rad referral+labs+imaging内容 | P2 | 同 |
| Next clinic visit | "in-person: prior to next visit with restaging studies" | P2 | 同,"in-person"推测不确定 |
| Advance care | "full code." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| Patient type | "returning to clinic while on leuprolide...anastrozole...palbociclib" | ✓ 合理 |
| second opinion | "Continue current therapy..." | P2 — 不相关 |
| in-person | "real-time Telehealth tools, including a live video connection..." | ✓ 精确 |
| summary | "biopsy proven metastatic recurrence...returning to clinic while on..." | ✓ |
| Type_of_Cancer | "[REDACTED]+/[REDACTED]- IDC" | ✓ 引用A/P |
| Stage_of_Cancer | "Stage III [REDACTED]+/[REDACTED]- IDC...now with metastatic recurrence" | ✓ |
| Metastasis | "metastatic recurrence (involving left cervical LN)" | ✓ |
| lab_summary | "Labs monthly. The patient agrees..." | **P2** — 这是lab plan不是lab results |
| findings | "biopsy proven metastatic recurrence...on leuprolide" | P2 — 太笼统 |
| current_meds | "anastrozole...palbociclib" | ✓ 但遗漏了leuprolide |
| goals_of_treatment | "biopsy proven metastatic recurrence" | ✓ |
| response_assessment | "CT and bone scan ordered...prior to next visit" | **P2** — 这是imaging plan不是response assessment |
| medication_plan | "Continue current therapy..." | ✓ |
| therapy_plan | "Continue current therapy..." | ✓ |
| radiotherapy_plan | "Radiation referral for symptomatic disease..." | ✓ 精确 |
| imaging_plan | "CT and bone scan ordered..." | ✓ |
| lab_plan | "Labs monthly..." | ✓ |
| Specialty | "send a message to Dr. [REDACTED]..." | ✓ |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓, summary改善, lab_summary改善(找到creatinine)
- **未修复**: P0 Specialty referral文本泄漏(系统性问题)
- **新问题**: **P1 current_meds丢失leuprolide**(clinic注射不在outpatient med list→改进8回归)
- **归因质量**: 中等。计划类归因好,诊断和评估类归因有交叉引用问题
- **统计**: 1×P0, 1×P1(新), 5×P2

---

## Row 5 (coral_idx=145)

**患者概况**: 34F, 2018.12自检右乳肿块。2019.3活检:G1 IDC, ER+/PR+/HER2 IHC2+ FISH非扩增(=HER2-)。MammaPrint低风险。2019.6.21双乳切+扩张器(右乳15×10mm G1 IDC, 0/1 node neg)。2019.6.8 zoladex, 2019.7.5 letrozole。另有双相情感障碍。

### v13a 已知问题
- P1: (1) Patient type应为Follow up非New patient (2) Type_of_Cancer缺HER2- (3) Stage应为Stage I非"Approximately Stage I-II"
- P2: (1) lab_summary redacted值 (2) medication_plan含"Estradiol monthly"(lab?) (3) in-person归因弱

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "New patient" | **P1 未修** | 仍然错。Chief Complaint"Follow-up",已在此就诊(zoladex) |
| second opinion | "no" | ✓ | 同 |
| in-person | "in-person" | ✓ | 同 |
| summary | "34F, ER+/PR+ IDC, post-bilateral mastectomy with expanders, follow-up for adjuvant therapy" | ✓ | 同 |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma" | **P1 未修** | 仍缺HER2-。HER2 IHC2+/FISH non-amplified=HER2- |
| Stage_of_Cancer | "" (空) | **P1 变化** | v13a"Approximately Stage I-II"(P1应为Stage I)。v14变空,更差。1.5cm+0/1 node neg=pT1cN0=Stage I |
| Metastasis | "No" | ✓ | 同 |
| Distant Metastasis | "No" | ✓ | 同 |
| lab_summary | 详尽列出CMP+CBC+Estradiol+VitD所有值 | ✓ | 同 |
| findings | 手术结果+体检(乳切愈合、无淋巴结肿大) | ✓ | 同 |
| current_meds | "letrozole" | **P1 新问题** | v13a有"letrozole, zoladex"。v14丢失zoladex!A/P"Started zoladex one month ago"但zoladex(clinic注射)不在outpatient med list→改进8回归 |
| recent_changes | "Started letrozole today." | ✓ | 同 |
| supportive_meds | "" | ✓ | 同 |
| goals_of_treatment | "curative" | ✓ | 同 |
| response_assessment | "Not mentioned in note." | ✓ | 同 |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a有内容 |
| medication_plan | "Start letrozole...3年→tamoxifen...gabapentin prn...Estradiol monthly" | ✓/P2 | 同。P2: "Estradiol monthly"可能是lab monitoring |
| therapy_plan | "letrozole today...3年→tamoxifen" | ✓ | 同 |
| radiotherapy_plan | "None" | ✓ | 同 |
| procedure_plan | "No procedures planned." | ✓ | 同 |
| imaging_plan | "No imaging planned." | ✓ | 同 |
| lab_plan | "Estradiol monthly." | ✓ | 同 |
| genetic_testing_plan | "myriad" | **P1 新问题** | Myriad检测已于04/25/2019完成(结果阴性)。POST-GENETICS-SEARCH搜到了"myriad"关键词但这是**已完成的检测**,不是计划。应为"None planned" |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "Dr. [REDACTED] at [REDACTED]. genetics referral" | **P1** | 这是04/24/2019的历史转诊,已完成。不是当前计划 |
| Referral-Specialty | "None" | ✓ | 同 |
| Referral-Others | "None" | ✓ | 同 |
| follow up | "RTC 3 months or sooner as needed" | ✓ | 同 |
| Next clinic visit | "in-person: 3 months or sooner as needed" | ✓ | 同 |
| Advance care | "Not discussed during this visit." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| Patient type | '{"quote":"NOT_IN_NOTE"}' | **P2** — 归因自己承认找不到支持"New patient"的文本! |
| second opinion | "RTC 3 months" | P2 — 不相关 |
| in-person | "RTC 3 months" | P2 — 不相关。应引用体检或"Total face to face time: 40" |
| summary | "34 year old woman with bipolar disorder, and ER/PR+..." | ✓ |
| Type_of_Cancer | "right breast with 1.5 cm node neg, grade 1 and ER/PR+ IDC." | ✓ 但缺HER2 |
| Metastasis | "1.5 cm node neg" | ✓ |
| lab_summary | "Estradiol monthly." | **P2** — 这是lab plan不是lab results |
| findings | "Underwent bilateral mastectomy...1.5 cm node neg..." | ✓ |
| current_meds | "Start [REDACTED] per her request" | ✓ |
| goals_of_treatment | 多段引用(zoladex+surgery+letrozole) | ✓ 全面 |
| medication_plan | "Start [REDACTED]...Estradiol monthly...Gabapentin..." | ✓ |
| therapy_plan | "Discussed starting letrozole...3年→tamoxifen" | ✓ |
| lab_plan | "Estradiol monthly." | ✓ |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓
- **未修复**: P1 Patient type(仍New patient), P1 Type_of_Cancer(仍缺HER2-)
- **新问题**: (1) **P1 Stage空**(v13a至少有"Approximately I-II") (2) **P1 current_meds丢失zoladex**(改进8回归) (3) **P1 genetic_testing_plan"myriad"是已完成检测**(改进6误触发) (4) **P1 Referral-Genetics是历史转诊**
- **归因质量**: 差。Patient type归因自己标了NOT_IN_NOTE,多个字段引用RTC follow-up文本
- **统计**: 0×P0, 6×P1(3未修+3新), 5×P2归因

---

## Row 6 (coral_idx=146)

**患者概况**: 女性, 1998年Stage II T2N1左乳IDC(biomarker unclear),lumpectomy+ALND(nodes+)+AC+T+XRT,未用辅助内分泌。2008年复发:左锁骨上LN转移(ER-/PR-/HER2+),纵隔转移。多线治疗(Taxotere/Xeloda/Herceptin→Tykerb→Pertuzumab/Herceptin/Taxotere)。本次:2nd opinion。PET-CT可能轻微进展(equivocal),LVEF 52%。

### v13a 已知问题
- P1: (1) therapy_plan只含否定语句,遗漏实际推荐 (2) genetic_testing_plan误含lab(tumor marker复查)
- P2: (1) lab_summary遗漏redacted marker (2) Medication_Plan_chatgpt格式 (3) follow up含lab

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "New patient" | ✓ | 同。CC"2nd opinion"=新患者会诊 |
| second opinion | "yes" | ✓ | 同 |
| in-person | "in-person" | ✓ | 同。有详细PE+生命体征 |
| summary | "MBC since 1998 Stage II...2nd opinion..." | ✓ | 同 |
| Type_of_Cancer | "Originally ER+/PR+/HER2+, metastatic biopsy ER-/PR-/HER2+ IDC" | **P1 新问题** | v13a"ER-/PR-/HER2+"正确。v14声称原发"ER+/PR+/HER2+"是**幻觉**——原文"Biomarker results unclear"。模型编造了原发receptor status |
| Stage_of_Cancer | "Originally Stage II, now metastatic (Stage IV)" | ✓ | 同 |
| Metastasis | "Yes (left supraclavicular node, mediastinum)" | ✓ | 同 |
| Distant Metastasis | 同上 | ✓ | 同 |
| lab_summary | "No labs in note." | P2 | 同(marker 14.8被redacted) |
| findings | 全面:PET-CT,纵隔稳定,脑MRI阴性,LVEF 52%,SUV 2.1 | ✓ | 同 |
| current_meds | "" | ✓ | 同("off rx since last wk") |
| recent_changes | "d/c current rx ([REDACTED]/Herceptin/Taxotere)" | ✓ | 同 |
| supportive_meds | "" | ✓ | 同 |
| goals_of_treatment | "palliative" | ✓ | 同 |
| response_assessment | "Probable mild progression...SUV 2.1(was 1.8)" | ✓ | 同 |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a P2格式问题 |
| medication_plan | "D/c [REDACTED]/Herceptin/Taxotere. Rec [REDACTED] as next line." | ✓ | 同 |
| therapy_plan | "Do not consider hormonal therapy at this time." | **P1 未修** | 仍只含否定,遗漏了实际推荐的下线治疗([REDACTED] as next line) |
| radiotherapy_plan | "None" | ✓ | 同 |
| procedure_plan | "No procedures planned." | ✓ | 同 |
| imaging_plan | "No imaging planned." | ✓ | 同 |
| lab_plan | "Would recheck [REDACTED] prior to above" | ✓ | 同 |
| genetic_testing_plan | "Would recheck [REDACTED] prior to above" | **P1 未修** | 仍把tumor marker复查误分类为genetic testing |
| Referral各项 | 全部"None" | ✓ | 同 |
| follow up | "Would recheck [REDACTED] prior to above" | P2 | 同,lab内容不是follow-up |
| Next clinic visit | "Not specified in the given text" | ✓ | 同 |
| Advance care | "Not discussed during this visit." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| second opinion | "Discussed current clinical trial possibilities..." | P2 — 不精确,应引用CC"2nd opinion" |
| in-person | "Pt appears to have minor degree of progression..." | P2 — 不相关 |
| summary | 同上 | P2 |
| Stage_of_Cancer | "MBC, recurrent and metastatic" | ✓ |
| Metastasis | "[REDACTED] metastasis. Mediastinal metastasis." | ✓ 精确 |
| response_assessment | "CT shows increased size of left breast tail nodule, with SUV 2.1 (was 1.8)." | ✓ 精确 |
| medication_plan | "would d/c current rx...Rec [REDACTED] as next line of Rx" | ✓ 全面 |
| therapy_plan | "Would not consider hormonal therapy at this time..." | ✓ 匹配(但值本身有问题) |
| genetic_testing_plan | "Would recheck [REDACTED] prior to above" | ✓ 匹配(但值分类错误) |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓
- **未修复**: P1 therapy_plan只含否定, P1 genetic_testing_plan误分类
- **新问题**: **P1 Type_of_Cancer幻觉原发ER+/PR+/HER2+**(原文"Biomarker results unclear")
- **归因质量**: 中等。Metastasis/response_assessment精确,其他偏笼统
- **统计**: 0×P0, 3×P1(1新+2未修), 5×P2

---

## Row 7 (coral_idx=147)

**患者概况**: 29F(绝经前), 2018.8左乳IDC grade 3, ER-/PR-/HER2+(IHC3+, FISH 5.7), 临床Stage II-III。TCHP新辅助×3不完整(依从性差)。手术:左lumpectomy+ALND→乳房无残留(ypT0), 3/28 LN+(最大2.4cm+ECE)。PET-CT×3无远处转移。本次10/13/2019:ZOOM会诊,讨论辅助AC×4→T-DM1+放疗。

### v13a 已知问题
- P1: (1) in-person应为in-person非Televisit (2) Stage N3错误应为N1 (3) procedure_plan混入chemo (4) imaging_plan遗漏echo
- P2: (1) oxycodone未验证 (2) response_assessment可记录near-pCR (3) radiotherapy_plan笼统

### v14 审查

| 字段 | v14 提取值 | 判定 | v13a对比 |
|------|-----------|------|---------|
| Patient type | "New patient" | ✓ | 同。"presents in consultation to establish care" |
| second opinion | "no" | ✓ | 同 |
| in-person | "Televisit" | ✓ 修正v13a | v13a标P1错误——实际上"presents through ZOOM" + "Vital Signs: None taken" + PE极简(无乳房/LN触诊)→确实是Televisit。v13a审查误判 |
| summary | "29F, clinical stage III HER2+/ER- IDC, incomplete neoadjuvant TCHP, post-lumpectomy/ALND" | ✓ | 同,优秀 |
| Type_of_Cancer | "ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) IDC" | ✓ | 同,包含FISH比值 |
| Stage_of_Cancer | "Originally Stage II-III, now Stage III" | P2 | v13a"pT0N3M0"(P1 N3错),v14去掉了N分期(避免错误但不够具体) |
| Metastasis | "No" | ✓ | 同 |
| Distant Metastasis | "No" | ✓ | 同 |
| lab_summary | "No labs in note." | ✓ | 同 |
| findings | 手术病理详细(ypT0, 3/28 LN+, ECE) + PET无转移 + 术前US变化 | ✓ | 同,优秀 |
| current_meds | "" | ✓ | 同 |
| recent_changes | "" | ✓ | 同 |
| supportive_meds | "oxyCODONE (ROXICODONE)" | P2 | 同。在med list中,用于术后疼痛。可接受 |
| goals_of_treatment | "curative" | ✓ | 同 |
| response_assessment | "Not yet on treatment — no response to assess." | P2 | 同。可记录近pCR但当前语境是讨论新辅助后治疗 |
| ~~Medication_Plan_chatgpt~~ | 已删除 | **改进1 ✓** | v13a有结构化内容 |
| medication_plan | "adjuvant AC x 4 cycles, to be followed by T-DM1" | ✓ | 同 |
| therapy_plan | "adjuvant AC x 4 → T-DM1 and radiation" | ✓ | 同 |
| radiotherapy_plan | "radiation after completing AC" | ✓ 改善 | v13a只有"radiation"(P2),v14加了"after completing AC"更具体 |
| procedure_plan | "adjuvant AC x 4 cycles" | **P1 未修** | 仍把化疗混入procedure。应仅含"port placement" |
| imaging_plan | "No imaging planned." | **P1 未修** | A/P提到echocardiogram,应在imaging plan |
| lab_plan | "No labs planned." | ✓ | 同 |
| genetic_testing_plan | "None planned." | ✓ | 同 |
| Referral-Nutrition | "None" | ✓ | 同 |
| Referral-Genetics | "None" | ✓ | 同 |
| Referral-Specialty | "None" | ✓ | 同 |
| Referral-Others | "Social work" | **P1 新问题** | 原文A/P中未提及social work referral。可能是模型从social context推断出来的——但是**幻觉referral** |
| follow up | "aim to speak again this coming week" | ✓ | 同 |
| Next clinic visit | "in-person: this coming week" | P2 | "speak again"可能是phone/video而非in-person |
| Advance care | "Not discussed during this visit." | ✓ | 同 |

### 归因审查

| 字段 | 归因引用 | 质量 |
|------|---------|------|
| Patient type | "{port placement and echocardiogram}" (JSON格式) | P2 — 不相关,归因格式异常 |
| in-person | "{80 minutes face-to-face...}" (JSON格式) | P2 — 归因格式异常,但内容相关 |
| summary | "29 y.o. premenopausal patient with...stage III..." | ✓ |
| Type_of_Cancer | "history of clinical stage III [REDACTED]-/[REDACTED]+ IDC" | ✓ |
| Metastasis | "no residual disease in breast but 3 of 28 LN positive" | ✓ 精确 |
| goals_of_treatment | "reduce risk by proceeding with...T-DM1" | ✓ |
| response_assessment | "Not yet on treatment" | ✓ |
| medication_plan | "adjuvant [REDACTED] followed by T-DM1" | ✓ |
| therapy_plan | "adjuvant [REDACTED] followed by T-DM1...radiation" | ✓ |
| radiotherapy_plan | "importance of radiation after completing AC" | ✓ |

### 本行总结
- **v14 修复**: 改进1(删Chatgpt) ✓, radiotherapy_plan更具体, v13a误判in-person已确认Televisit正确
- **未修复**: P1 procedure_plan混入chemo, P1 imaging_plan遗漏echo
- **新问题**: **P1 Referral-Others幻觉social work**(原文无此referral)
- **归因质量**: 中等。病理/治疗归因好,Patient type归因格式异常
- **统计**: 0×P0, 3×P1(1新+2未修), 4×P2

---

## Row 8 (coral_idx=148)

**患者概况**: 63F, 肾移植受者(1990)。2020.11右乳ER+癌。新辅助AC×4+weekly taxol×12→2021.6.11双乳切:右乳3.84cm残留IDC, 1 LN+/ECE。ER+(85%)/PR-(<1%)/HER2-(IHC0)。本次video follow-up。

### v14 审查

| 字段 | v14 值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | "Follow up" | ✓ | |
| in-person | "Televisit" | ✓ | Video Visit+Zoom |
| Type_of_Cancer | "ER+/PR-/HER2- IDC" | ✓ | 正确识别罕见的PR-和HER2-(IHC0) |
| Stage_of_Cancer | "Stage II" | ✓ | |
| current_meds | "" | ✓ | 术后/放疗前,letrozole计划在放疗后 |
| supportive_meds | "ondansetron, compazine, olanzapine, miralax" | P2 | 化疗已完成 |
| goals_of_treatment | "curative" | ✓ | |
| response_assessment | "Not yet on treatment" | P2 | 可记录新辅助后病理残留 |
| medication_plan | "Letrozole after radiation. Fosamax." | ✓ | |
| Referral-Specialty | "Radiation referral" | ✓ | |
| Advance care | "full code." | ✓ | |

### 本行总结
- **v14 修复**: 改进1 ✓
- **统计**: 0×P0, 0×P1, 3×P2

---

## Row 9 (coral_idx=149)

**患者概况**: 66F, 左乳中央区ER+/HER2-癌Stage II。Neoadjuvant letrozole→左乳切(8.8cm, LN受累)→re-excision。当前letrozole。笔记含两次visit合并。

### v14 审查

| 字段 | v14 值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | "Follow up" | ✓ | |
| in-person | "Televisit" | ✓ | Video→failed→telephone |
| Type_of_Cancer | "" (空) | **P1** | A/P"HR + and her 2 negative"→应为"ER+/HER2-" |
| Stage_of_Cancer | "Stage II" | ✓ | |
| current_meds | "letrozole" | ✓ | 在med list和A/P中 |
| medication_plan | "continue on letrozole" | ✓ | |
| therapy_plan | "continue letrozole, radiation to left chest wall and LN" | ✓ | 具体 |
| radiotherapy_plan | "Radiation to left chest wall and surrounding lymph nodes." | ✓ | |
| imaging_plan | "DEXA." | ✓ | A/P#6 |
| Advance care | "full code." | ✓ | |

### 本行总结
- **v14 修复**: 改进1 ✓
- **新问题**: P1 Type_of_Cancer空(A/P有ER+/HER2-信息但未提取)
- **统计**: 0×P0, 1×P1, 2×P2

---

## 中期汇总 (9行: Row 0, 1, 2, 4, 5, 6, 7, 8, 9)

### v14 改进效果

| 改进 | 状态 | 证据 |
|------|------|------|
| 1. 删Chatgpt | ✅ 完全生效 | 所有行不再有此字段 |
| 2. supportive_meds过滤 | ✅ 部分生效 | Row 8支持药正确分类。但未覆盖所有行的检验 |
| 3. Stage幻觉 | ⚠️ 部分生效 | Row 5 Stage变空(v13a有"Approximately I-II")——可能过度修正 |
| 4. Cross-field白名单 | ✅ 未在已审查行中触发大问题 | |
| 5. therapy_plan修复 | ❓ 未在已审查行中验证(需看有"continue"的行) | |
| 6. POST-GENETICS-SEARCH | ⚠️ 误触发 | Row 5 "myriad"是已完成的检测非计划 |
| 7. Referral全文提取 | ⚠️ 部分问题 | Row 7幻觉出social work referral |
| 8. current_meds cross-reference | ❌ **系统性回归** | Row 1丢irinotecan, Row 4丢leuprolide, Row 5丢zoladex |

### 系统性P0/P1问题

| 问题 | 出现行 | 频率 | 类型 |
|------|--------|------|------|
| Specialty Referral文本泄漏 | Row 0, 4 | 2/9 | P0 未修 |
| Type_of_Cancer缺HER2- | Row 0, 5 | 2/9 | P1 未修 |
| Type_of_Cancer空 | Row 9 | 1/9 | P1 |
| Type_of_Cancer幻觉receptor | Row 6 | 1/9 | P1 新 |
| current_meds丢失clinic注射药 | Row 1, 4, 5 | 3/9 | P1 回归(改进8) |
| response_assessment答非所问 | Row 0 | 1/9 | P1 未修 |
| imaging/lab_plan遗漏(A/P外) | Row 0, 7 | 2/9 | P1 pipeline限制 |
| procedure_plan混入chemo | Row 7 | 1/9 | P1 未修 |
| therapy_plan只含否定 | Row 6 | 1/9 | P1 未修 |
| genetic_testing_plan误分类 | Row 5, 6 | 2/9 | P1 |
| Referral幻觉 | Row 7 | 1/9 | P1 新 |

### 归因质量总结

| 模式 | 频率 | 描述 |
|------|------|------|
| Reason_for_Visit归因差 | 高(6/9行) | Patient type/second opinion/in-person引用无关文本(如"RTC","Continue therapy") |
| 诊断类归因笼统 | 中(4/9行) | Stage/Metastasis归因引用泛泛文本而非具体数据 |
| 计划类归因好 | 高(7/9行) | medication_plan/therapy_plan/imaging_plan归因精确引用A/P |
| 归因格式异常 | 低(2/9行) | 部分归因包含JSON格式或NOT_IN_NOTE |

### 关键发现:改进8(cross-reference med list)需要修复

最严重的回归来自改进8。问题根因:
- 化疗药(irinotecan)通过infusion center给药,不在outpatient med list
- GnRH agonist(leuprolide/zoladex)是clinic注射,不在outpatient med list
- cross-reference instruction让模型只看medication list,排除了A/P明确提到的当前用药

**建议修复**: 在prompt中加: "If a drug is explicitly mentioned in the Assessment/Plan as a CURRENT medication (e.g., 'currently on X', 'continues X'), include it even if it does not appear in the Current Outpatient Medications list. Clinic-administered drugs (injections, infusions) may not appear in the outpatient medication list."

---

