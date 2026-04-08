# v14a 逐行审查（含归因）

审查日期：2026-03-17
版本：v14a（修复改进 6 和 8 后重新运行 61 行）
审查人：Claude

## 系统性发现总结（61 行全扫描）

### P0 问题
| # | 行 | 问题 | 描述 |
|---|-----|------|------|
| 1 | Row 0 | Referral Specialty 文本泄漏 | "Integrative Medicine, Integrative Medicine History of Present Illness: 56" |
| 2 | Row 4 | Referral Specialty 文本泄漏 | 后续笔记内容串入 |
| 3 | Row 56 | Type_of_Cancer 矛盾 | "ER-/PR-/HER2+ triple negative" — HER2+ 和 triple negative 互斥 |

### 系统性 P1 问题

**1. Type_of_Cancer 缺 HER2 (11/61 行 = 18%)**
- Rows: 0, 5, 9, 10, 13, 33, 41, 82, 90, 96, 99
- 原因: 有 ER/PR 但未提取 HER2 状态。prompt 要求推断但模型经常遗漏
- 其中 Row 41 还缺 ER (只有 PR+)，Row 82 完全无受体状态

**2. genetic_testing_plan "ngs" 假阳性 (4/61 行 = 7%)**
- Rows: 28, 29, 62, 93
- 原因: POST-GENETICS-SEARCH 匹配到笔记中的 "ngs" 文本（可能是 "stainings"、"findings" 等包含 ngs 的词）
- 修复建议: 在 genetic_tests.txt 中将 "ngs" 改为 "ngs " (带空格) 或 "next generation sequencing" 全称

**3. Attribution JSON wrapper 泄漏 (11/61 行 = 18%, 17 次)**
- Rows: 5, 7, 21, 26, 32, 40, 48, 62, 94, 96, 99
- 模式: 归因引文输出为 `{"quote": "..."}` 而非纯文本
- 原因: LLM 归因提取时偶尔输出 JSON 格式

**4. Attribution 系统性缺失**
- current_meds 归因缺失: 35/61 行 (57%)
- Patient type 归因缺失: 16/61 行 (26%)
- response_assessment 归因缺失: 15/61 行 (25%)
- Type_of_Cancer 归因缺失: 9/61 行 (15%)

**5. response_assessment 过度解读 (≥3 行)**
- Row 10: 把治疗前 PET 进展误说成治疗后进展
- Row 13: 把 "labs OK" 过度解读为 "currently responding"
- Row 45, 82: "currently responding" 缺乏足够影像证据

**6. goals_of_treatment 与 Stage IV 矛盾 (1 行)**
- Row 82: Stage IV + curative（应为 palliative）

### 改进效果验证

| 改进 | 目标 | v14a 结果 | 状态 |
|------|------|----------|------|
| 改进 1 | 删除 Medication_Plan_chatgpt | 字段不再存在 | ✅ 完成 |
| 改进 2 | supportive_meds 不含 oncology 药 | 未见 letrozole/tamoxifen 误分类 | ✅ 改善 |
| 改进 3 | Stage 幻觉修复 | [X] 占位符用于 6 行，13 行有原文支持 | ✅ 有效 |
| 改进 4 | Cross-field 白名单 | 未系统验证，需逐行检查 | ⏳ 待验证 |
| 改进 5 | therapy_plan "None" 修复 | 改善: therapy_plan 正确输出 "continue [drug]" | ✅ 改善 |
| 改进 6 | genetics 白名单搜索 | 26 行有内容（vs 35 None），但 4 行 "ngs" 假阳性 | ⚠️ 部分有效 |
| 改进 7 | Referral 全文提取 | Referral 覆盖率提高，但仍有 2 行文本泄漏 P0 | ⚠️ 部分有效 |
| 改进 8 | current_meds 时态修复 | Row 4/5 验证修复，clinic-administered drugs 正确包含 | ✅ 有效 |

### v15 改进建议（按优先级）

1. **修复 "ngs" 假阳性** — 在 genetic_tests.txt 中用全词匹配 "\bngs\b" 或改用全称
2. **修复 Type_of_Cancer HER2 遗漏** — prompt 中强调必须三项全出（ER/PR/HER2），不能只写两项
3. **修复 attribution JSON wrapper** — 在 attribution 提取后做 JSON 解析清理
4. **修复 Referral 文本泄漏** — 检查 Referral Specialty 长度，超过阈值则截断或清理
5. **修复 response_assessment 过度解读** — prompt 中加入 "Do NOT use the word 'responding' unless imaging confirms response"
6. **提高 attribution 覆盖率** — current_meds 57% 缺失最严重

---

---

## Row 0 (coral_idx=140) — 新患者，Stage IIA 复发转移

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | 首次 med onc consult |
| second opinion | no | OK | |
| in-person | in-person | OK | |
| summary | 56-year-old female with newly diagnosed metastatic ER+ breast cancer... | OK | 准确 |
| Type_of_Cancer | ER+/PR+ invasive ductal carcinoma | **P1** | 缺 HER2-。原文 "her 2 neu negative" |
| Stage_of_Cancer | Originally Stage IIA, now metastatic (Stage IV) | OK | 原文 HPI "multifocal Stage IIA right breast cancer"，有原文支持 |
| Metastasis | Yes (to lungs, peritoneum, liver, and ovaries) | OK | 与 CT 报告一致 |
| Distant Metastasis | Yes (to lungs, peritoneum, liver, and ovaries) | OK | |
| lab_summary | No labs in note. | OK | 最近 labs 是 2001 年的 |
| findings | Widespread metastases... hepatomegaly... 3cm mass in axilla | OK | 完整准确 |
| current_meds | "" | OK | 原文 "No current outpatient medications on file" |
| recent_changes | "" | OK | 新患者无治疗变化 |
| supportive_meds | "" | OK | |
| goals_of_treatment | palliative | OK | 原文 "treatment would be palliative" |
| response_assessment | Metastatic relapse... | **P2** | 这不是 response assessment（还没开始治疗），更像是当前疾病状态描述。但作为新患者初诊，写 baseline status 也可接受 |
| medication_plan | ibrance and [REDACTED] if HR+/HER2- | OK | |
| therapy_plan | ibrance and [REDACTED] if HR+/HER2- | OK | 与 medication_plan 重复但正确 |
| radiotherapy_plan | None | OK | |
| procedure_plan | biopsy mass in right axilla | OK | |
| imaging_plan | No imaging planned. | **P1** | 原文明确 ordered "NM Whole Body Bone Scan" 和 "MR Brain with and without Contrast"，还有 A/P "complete her staging work up" |
| lab_plan | No labs planned. | **P1** | 原文 ordered "CBC, CMP, CA 15-3, CEA, aPTT, PT" |
| genetic_testing_plan | No specific genetic... biopsy to confirm HR and HER2 | OK | biopsy 不是基因检测但合理描述 |
| Next clinic visit | in-person: after completed work up | OK | |
| Advance care | Full code. | OK | |
| Referral: Specialty | **Integrative Medicine, Integrative Medicine History of Present Illness: 56** | **P0** | 文本泄漏！"History of Present Illness: 56" 是后续内容串入 |
| Referral: follow up | RTC with me after completed work up | OK | |
| Referral: Nutrition | None | OK | |
| Referral: Genetics | None | OK | |
| Referral: Others | None | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "RTC with me after completed work up to formulate a plan." | **P1-归因** | 归因不准确。应指向 "New Patient Evaluation" 或 consult note 标题 |
| second opinion | "RTC with me after completed work up to formulate a plan." | **P2-归因** | 归因泛泛，但结论正确 |
| in-person | "RTC with me after completed work up to formulate a plan." | **P2-归因** | 同上 |
| summary | "Metastatic relapse of her breast cancer..." | OK | 部分支持 |
| Type_of_Cancer | "2.4 and 2.3 cm tumors, node negative ER and PR positive and her 2 negative." | OK | 正确引文 |
| Stage_of_Cancer | "Metastatic relapse of her breast cancer." | **P1-归因** | 应引用 HPI "multifocal Stage IIA right breast cancer" 来支持 "Originally Stage IIA"。当前归因只支持 "metastatic" 部分 |
| Metastasis | "Metastatic relapse of her breast cancer." | **P2-归因** | 太泛泛，应引用 CT 报告中的具体转移部位 |
| findings | "Widespread metastases consistent with metastatic breast carcinoma." | OK | CT 报告第一句 |
| goals_of_treatment | "Metastatic relapse of her breast cancer..." | **P2-归因** | 应引用 "treatment would be palliative" |
| response_assessment | "Metastatic relapse of her breast cancer." | **P2-归因** | 太泛泛 |
| medication_plan | "If she is HR+/ her 2 negative on biopsy will recommend ibrance and [REDACTED]." | OK | 精确引文 |
| therapy_plan | "If she is HR+/ her 2 negative on biopsy will recommend ibrance and [REDACTED]." | OK | |
| procedure_plan | "Appointment with Dr. [REDACTED]... biopsy mass in right axilla in the office." | OK | |
| genetic_testing_plan | "Appointment with Dr. [REDACTED]... biopsy mass in right axilla in the office." | OK | biopsy 确认 HR/HER2 |
| Next clinic visit | "RTC with me after completed work up to formulate a plan." | OK | 精确引文 |

### Row 0 小结
- **P0**: 1 个（Referral Specialty 文本泄漏）
- **P1**: 3 个 keypoint (Type_of_Cancer 缺 HER2-, imaging_plan 遗漏, lab_plan 遗漏) + 2 个归因 (Patient type 归因错, Stage 归因不充分)
- **P2**: 1 个 keypoint (response_assessment 不是真正的 response) + 4 个归因 (泛泛引文)

---

## Row 1 (coral_idx=141) — 转移性 TNBC，irinotecan cycle 3 day 1

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| second opinion | no | OK | |
| in-person | in-person | OK | |
| summary | 44 y.o. female with metastatic breast cancer, cycle 3 day 1 of irinotecan | OK | 准确简洁 |
| Type_of_Cancer | ER-/PR-/HER2- triple negative invasive ductal carcinoma | OK | 原文 "triple negative" 多处确认 |
| Stage_of_Cancer | Originally Stage IIB, now metastatic (Stage IV) | OK | 原文 "clinical Stage IIB"，转移性复发 |
| Metastasis | Yes (to liver, bone, and chest wall) | OK | PET/CT 确认 |
| Distant Metastasis | Yes (to liver, bone, and chest wall) | OK | |
| lab_summary | Albumin 2.1 (L), Alk Phos 183 (H)... Sodium 124 (LL)... Hemoglobin 7.7 (L)... | OK | 详细准确 |
| findings | chest wall tender/erythematous... back pain worse... anemia worse... bone lesions... | OK | 综合了临床发现+影像+labs |
| current_meds | "" | **P1** | 原文 "Continue current therapy" = irinotecan。虽非 outpatient med list，但 A/P 明确说继续 |
| recent_changes | irinotecan 改为每两周一次，剂量增至 150mg/m2 | OK | |
| supportive_meds | ondansetron, prochlorperazine, loperamide, diphenoxylate-atropine, morphine, oxycodone | OK | |
| goals_of_treatment | palliative | OK | |
| response_assessment | PET/CT 05/31/19 showed significantly increased metastases... | OK | 比 v14 好很多 |
| medication_plan | Doxycycline... Flexeril... oxycodone... NS IV... potassium... pRBC | **P2** | 混合了 supportive 和 one-time orders |
| therapy_plan | irinotecan every other week, 150mg/m2 | OK | |
| radiotherapy_plan | urgently needs Rad Onc consult | OK | |
| procedure_plan | No procedures planned. | OK | |
| imaging_plan | Scans in 3 months, MRI brain if worse | OK | |
| lab_plan | Monitor ALT, HBsAg, HBV DNA every 4 months | OK | |
| genetic_testing_plan | None planned. | OK | |
| Next clinic visit | in-person: 2 weeks | OK | |
| Advance care | Not discussed during this visit. | OK | |
| Referral: Specialty | Rad Onc consult | OK | |
| Referral: Others | Social work referral, Home health referral | **P2** | "Home health?" 是问句，不确定 |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "F/u 2 weeks" | **P2-归因** | 间接支持 |
| in-person | "I spent a total of 45 minutes face-to-face..." | OK | |
| summary | "44 y.o. female with [REDACTED] Syndrome..." | OK | |
| Type_of_Cancer | "metastatic breast cancer" | **P1-归因** | 没引用 "triple negative" 来源 |
| Stage_of_Cancer | "newly diagnosed metastatic breast cancer" | **P1-归因** | 没引用 "Stage IIB" 来源 |
| Metastasis | "Metastatic breast cancer - Chest wall more tender..." | **P2-归因** | 不是转移部位证据 |
| recent_changes | "Due to her poor tolerance, will change her irinotecan..." | OK | 精确 |
| goals_of_treatment | "metastatic breast cancer" | **P2-归因** | 隐含但不直接 |
| response_assessment | "Metastatic breast cancer - Chest wall more tender..." | **P2-归因** | 应引用 PET/CT 结果 |
| medication_plan | "Rx Doxycycline... Continue [REDACTED]... oxycodone... NS IV... potassium... pRBC" | OK | |
| therapy_plan | "Due to her poor tolerance, will change her irinotecan..." | OK | |
| radiotherapy_plan | "She urgently needs to get in with Rad Onc..." | OK | |
| imaging_plan | "Scans again in 3 months... MRI brain if worse" | OK | |
| lab_plan | "Monitor with ALT, hepatitis B surface antigen..." | OK | |

### Row 1 小结
- **P0**: 0
- **P1**: 1 keypoint (current_meds 遗漏 irinotecan) + 2 归因 (Type/Stage 归因不充分)
- **P2**: 2 keypoint + 4 归因
- **vs v14**: response_assessment 显著改善；supportive_meds 更完整

---

## Row 2 (coral_idx=142) — 新患者，Stage IIA，neoadjuvant 讨论

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | |
| second opinion | yes | OK | "She has had several opinions" |
| in-person | Televisit | OK | "Video Consult" |
| summary | 53 y.o. female with newly diagnosed Stage IIA right breast IDC... | OK | |
| Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma | OK | 完整 |
| Stage_of_Cancer | Stage IIA | OK | |
| Metastasis | No | OK | |
| Distant Metastasis | No | OK | |
| lab_summary | No labs in note. | OK | |
| findings | 1.7 cm tumor, positive 1.5 cm LN... | OK | |
| current_meds | "" | OK | 无现有药物 |
| goals_of_treatment | curative | OK | |
| response_assessment | Not yet on treatment — no response to assess. | OK | |
| therapy_plan | Discussed chemotherapy... surgery and radiation... | OK | |
| radiotherapy_plan | discussed role of radiation | OK | |
| imaging_plan | PET imaging follow up | OK | |
| genetic_testing_plan | Genetic testing sent and is pending. | OK | |
| Next clinic visit | telehealth: after PET and [REDACTED] are back | OK | |
| Advance care | full code. | OK | |
| Referral: Genetics | Genetic testing sent and is pending. | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| second opinion | "[REDACTED] sent and is pending." | **P1-归因** | 应引 "several opinions" |
| in-person | "real-time telehealth tools... live video Zoom" | OK | |
| Type_of_Cancer | "Stage IIA right IDC... HR+, her 2 neu negative by FISH." | OK | |
| goals_of_treatment | "surgery and radiation to decrease the risk of local recurrence." | OK | |
| response_assessment | 归因 = 提取值本身 | **P2-归因** | 不是原文引文 |
| genetic_testing_plan | "Genetic testing sent and is pending." | OK | |

### Row 2 小结
- **P0**: 0
- **P1**: 0 keypoint + 1 归因 (second opinion 归因错误)
- **P2**: 0 keypoint + 1 归因

---

## Row 4 (coral_idx=144) — 转移性 ER+/HER2- IDC，leuprolide+anastrozole+palbociclib

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| in-person | Televisit | OK | video encounter |
| summary | recurrent breast cancer, continue therapy with ibrance/lupron... | OK | |
| Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma | OK | 完整 |
| Stage_of_Cancer | Originally Stage III, now metastatic (Stage IV) | OK | A/P "Stage III...now with biopsy proven metastatic recurrence" |
| Metastasis | Yes (to left cervical LN, left IM LN, and sternum) | OK | 与影像一致 |
| lab_summary | Creatinine 0.6 mg/dL (from 08/23/2019) | OK | 唯一可用 lab |
| findings | cervical LAD, brachial plexus involvement, MRI shows interval enlargement of left level 5B node... | OK | 综合 |
| current_meds | anastrozole, palbociclib, leuprolide | OK | **改进 8 修复生效！** v14 缺少 leuprolide |
| recent_changes | "" | OK | 无变化，继续当前方案 |
| supportive_meds | ondansetron (ZOFRAN) 8 mg | OK | |
| goals_of_treatment | palliative | OK | |
| response_assessment | CT neck shows decreased LN, stable mediastinal LN, increased axillary LN, new sternal lesion... | OK | 混合反应，准确 |
| medication_plan | Continue leuprolide, anastrozole, and palbociclib | OK | |
| therapy_plan | Continue current therapy... Radiation referral... Labs monthly | OK | |
| radiotherapy_plan | Radiation oncology referral for symptomatic disease | OK | |
| imaging_plan | CT and bone scan ordered | OK | |
| lab_plan | Labs monthly on day of lupron injection | OK | |
| genetic_testing_plan | None planned. | OK | |
| Next clinic visit | in-person: prior to next visit with restaging studies | OK | |
| Advance care | full code. | OK | |
| Referral: Specialty | **Radiation oncology referral, Radiation Oncology CT Abdomen /Pelvis with Contrast CT** | **P0** | 文本泄漏！orders 列表串入了 Specialty 字段 |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "returning to clinic while on leuprolide... anastrozole... palbociclib..." | OK | 支持 follow up |
| in-person | "I performed this consultation using real-time Telehealth tools..." | OK | |
| Type_of_Cancer | "[REDACTED]+/[REDACTED]- IDC of the left breast" | OK | redacted 但正确 |
| Stage_of_Cancer | "Stage III [REDACTED]+/[REDACTED]- IDC... now with biopsy proven metastatic recurrence" | OK | 精确 |
| Metastasis | "now with biopsy proven metastatic recurrence (involving left cervical LN)" | OK | |
| lab_summary | "Labs monthly..." | **P1-归因** | 引用了 lab plan 而非 lab 结果。应引 Creatinine 0.6 那条 |
| findings | "biopsy proven metastatic recurrence (involving left cervical LN)..." | **P2-归因** | A/P 文本，不是 MRI/CT 具体描述 |
| current_meds | "Continue current therapy and will get restaging studies..." | **P2-归因** | 没引用具体药物名。应引 "on leuprolide... anastrozole... palbociclib" |
| goals_of_treatment | "biopsy proven metastatic recurrence (involving left cervical LN)" | **P2-归因** | 隐含 palliative 但不直接 |
| response_assessment | "CT and bone scan ordered..." | **P1-归因** | 引用的是 imaging plan 而非影像结果！完全不支持 response assessment 内容 |
| medication_plan | "Continue current therapy..." | OK | |
| radiotherapy_plan | "Radiation referral for symptomatic disease..." | OK | |
| imaging_plan | "CT and bone scan ordered..." | OK | |
| lab_plan | "Labs monthly..." | OK | |

### Row 4 小结
- **P0**: 1（Referral Specialty 文本泄漏）
- **P1**: 0 keypoint + 2 归因 (lab_summary 归因错, response_assessment 归因完全错)
- **P2**: 0 keypoint + 3 归因
- **vs v14**: current_meds 恢复了 leuprolide（改进 8 修复有效）

---

## Row 5 (coral_idx=145) — 早期 ER+/PR+ IDC，术后 adjuvant，开始 letrozole

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | **P2** | A/P 中有明确的 IMP 和讨论方案，但 note 开头是 "HPI" 不是 consult note 标题。This is more of a follow-up after surgery（zoladex 已开始一个月前）。但首次 med onc 系统治疗讨论可算 new patient |
| summary | 34-year-old female, ER+/PR+ IDC right breast, post-bilateral mastectomy, initial consult for adjuvant | OK | |
| Type_of_Cancer | ER+/PR+ invasive ductal carcinoma | **P1** | 缺 HER2-。原文 "HER2 equivocal(IHC 2), FISH non-amplified" = HER2- |
| Stage_of_Cancer | "" | **P1** | 空。原文有 "1.5 cm node neg, grade 1" = pT1c pN0 = Stage I。虽 staging form 未明确列出，但可推断 |
| Metastasis | No | OK | |
| lab_summary | Estradiol 172, Vitamin D 24, 全套 CMP 和 CBC... | **P2** | 太详细（列了每个正常值），不够简洁 |
| findings | bilateral mastectomy, right breast 1.5 cm node-neg grade 1 IDC... | OK | |
| current_meds | zoladex, letrozole | OK | **改进 8 修复生效！** v14 缺少 zoladex |
| recent_changes | Started letrozole today. | OK | |
| goals_of_treatment | curative | OK | 早期，adjuvant |
| response_assessment | Not mentioned in note. | OK | 术后 adjuvant，无 response 可评 |
| medication_plan | Start letrozole, continue [REDACTED] 3 years, gabapentin prn, estradiol monthly | OK | |
| therapy_plan | letrozole for at least 3 years, can sequence with tamoxifen | OK | |
| imaging_plan | No imaging planned. | OK | |
| lab_plan | Estradiol monthly. | OK | |
| genetic_testing_plan | None planned. | OK | **改进 6 修复生效！** v14 错误地加了 "myriad"（2019 年已完成结果阴性） |
| Next clinic visit | in-person: 3 months or sooner | OK | |
| Referral: Genetics | Dr. [REDACTED] at [REDACTED]. genetics referral | **P2** | 这是过去的 genetics referral (04/24/2019)，不是当前 plan |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "NOT_IN_NOTE" | **P1-归因** | 归因失败，找不到支持文本 |
| in-person | "RTC 3 months or sooner as needed." | **P2-归因** | 不直接支持 in-person，应引 physical exam 或 vital signs |
| summary | "34 year old woman with bipolar disorder, and ER/PR+... low risk right breast cancer..." | OK | |
| Type_of_Cancer | "right breast with 1.5 cm node neg, grade 1 and ER/PR+ IDC." | OK | 准确引文（但缺 HER2 信息） |
| Metastasis | "1.5 cm node neg" | OK | 支持 No metastasis |
| lab_summary | "Estradiol monthly." | **P1-归因** | 归因引了 lab plan 而非 lab results。应引 "Estradiol, Ultrasensitive 172 pg/mL" |
| current_meds | "Started zoladex one month ago." "Discussed starting letrozole today." | OK | |
| recent_changes | "Start [REDACTED] per her request" | OK | |
| goals_of_treatment | "Started zoladex... node neg, grade 1... ER/PR+ IDC... starting letrozole" | OK | 隐含 curative |
| medication_plan | "Start [REDACTED]... Estradiol monthly... Gabapentin..." | OK | |
| therapy_plan | "Discussed starting letrozole today... continue... 3 years... sequence with tamoxifen" | OK | |
| lab_plan | "Estradiol monthly." | OK | |
| Next clinic visit | "RTC 3 months or sooner as needed." | OK | |

### Row 5 小结
- **P0**: 0
- **P1**: 2 keypoint (Type_of_Cancer 缺 HER2-, Stage 为空) + 2 归因 (Patient type NOT_IN_NOTE, lab_summary 归因错)
- **P2**: 2 keypoint (Patient type 模糊, lab_summary 过详细) + 1 归因
- **vs v14**: current_meds 恢复 zoladex（改进 8 有效）；genetic_testing_plan 不再误报 myriad（改进 6 有效）

---

## Row 6 (coral_idx=146) — 2nd opinion，MBC，HER2+ IDC

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | |
| second opinion | yes | OK | CC "2nd opinion" |
| Type_of_Cancer | Originally ER+/PR+/HER2+, metastatic biopsy ER-/PR-/HER2+ IDC | **P2** | 复杂情况。原文 1998 biomarkers "unclear"，2008 recurrence "ER/PR neg, ***** by IHC"。模型推断原始 ER+/PR+ 但原文说 "Biomarker results unclear"+"Did not receive adjuvant hormonal Rx"，可能本来就是 ER-/PR- |
| Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | OK | "Stage II T2N1" |
| Metastasis | Yes (to left supraclav node, mediastinum) | OK | |
| current_meds | "" | **P2** | 原文 A/P 说 "off of rx since last wk" 所以当前确实没有活跃药物，但 pertuzumab/herceptin/taxotere 刚停一周。空可接受 |
| recent_changes | d/c current rx ([REDACTED]/Herceptin/Taxotere) | OK | |
| goals_of_treatment | palliative | OK | |
| response_assessment | Probable mild progression... CT shows increased left breast tail nodule... | OK | |
| therapy_plan | Do not consider hormonal therapy at this time. | **P2** | A/P 推荐了 "[REDACTED] as next line"，therapy_plan 只提了不用 hormonal，遗漏了具体推荐 |
| genetic_testing_plan | Would recheck [REDACTED] prior to above | **P1** | 这不是基因检测，是 recheck lab（可能是 HER2 FISH 或肿瘤标志物）。分类错误 |
| Next clinic visit | Not specified in the given text | **P2** | A/P 中有 "Would recheck [REDACTED] prior to above"，可能暗含 follow-up |
| Referral | all None | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Stage_of_Cancer | "MBC, recurrent and metastatic" | **P2-归因** | 没引用 "Stage II T2N1" |
| Metastasis | "[REDACTED] metastasis. Mediastinal metastasis." | OK | A/P 第 2、3 点 |
| response_assessment | "CT shows increased size of left breast tail nodule, with SUV 2.1 (was 1.8)." | OK | 精确 |
| therapy_plan | "Would not consider hormonal therapy... evidence for hormone responsiveness is slight" | OK | |
| genetic_testing_plan | "Would recheck [REDACTED] prior to above" | OK | 引文正确，但分类错 |
| goals_of_treatment | "MBC, recurrent and metastatic" | **P2-归因** | 隐含 |

### Row 6 小结
- **P0**: 0
- **P1**: 1 keypoint (genetic_testing_plan 分类错误)
- **P2**: 4 keypoint + 2 归因
- **vs v14**: 无变化

---

## Row 7 (coral_idx=147) — 新患者，Stage III HER2+ IDC，neoadjuvant 后

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | consultation |
| in-person | Televisit | **P2** | 原文有 "80 minutes face-to-face"，但可能是通过视频 |
| Type_of_Cancer | ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) IDC | OK | 完整 |
| Stage_of_Cancer | Originally Stage II-III, now Stage III | **P2** | "Originally Stage II-III" 有些奇怪。原文说 "clinical stage III"。不需要 "Originally" |
| current_meds | "" | OK | 还未开始 adjuvant |
| goals_of_treatment | curative | OK | |
| response_assessment | Not yet on treatment — no response to assess. | **P2** | 实际上做了 neoadjuvant TCHP 且有 residual disease in LN，有 response 可评（pCR in breast, residual in LN） |
| medication_plan | adjuvant AC x 4, to be followed by T-DM1 | OK | |
| therapy_plan | adjuvant AC x 4, to be followed by T-DM1 and radiation | OK | |
| radiotherapy_plan | radiation after completing AC | OK | |
| procedure_plan | adjuvant AC x 4, T-DM1, port placement | **P2** | AC 和 T-DM1 是 systemic therapy 不是 procedure。port placement 是 procedure |
| Referral: Others | Social work | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "{\"quote\": \"I have sent [REDACTED]...\"}" | **P1-归因** | JSON wrapper 泄漏到归因中 |
| in-person | "{\"quote\": \"I spent a total of 80 minutes face-to-face...\"}" | **P1-归因** | 同上，JSON wrapper 泄漏 |
| Stage_of_Cancer | "clinical stage III" | OK | |
| Metastasis | "no residual disease in breast but 3 of 28 LN positive..." | OK | |
| goals_of_treatment | "she can reduce this risk by proceeding with recommended systemic therapy..." | OK | |
| response_assessment | 归因 = 提取值本身 | **P2-归因** | |
| medication_plan | "adjuvant AC x 4 cycles, to be followed by T-DM1" | OK | |
| procedure_plan | "adjuvant AC x 4 cycles, to be followed by T-DM1" | **P2-归因** | 引文是 therapy 不是 procedure |

### Row 7 小结
- **P0**: 0
- **P1**: 0 keypoint + 2 归因 (JSON wrapper 泄漏)
- **P2**: 4 keypoint + 2 归因

---

## Row 8 (coral_idx=148) — Follow-up，Stage II ER+/HER2- IDC，术后

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| in-person | Televisit | OK | Zoom |
| Type_of_Cancer | ER+/PR-/HER2- IDC | OK | 原文 "HR+/ her 2 neu negative"。PR- 需验证——看 oncotype 或病理 |
| Stage_of_Cancer | Stage II | OK | |
| current_meds | "" | **P2** | 原文 "S/p 4 cycles of [REDACTED] and weekly taxol x 12"，已完成化疗。目前确实没有活跃的 cancer-related meds，letrozole 是 plan |
| goals_of_treatment | curative | OK | |
| response_assessment | Not yet on treatment — no response to assess. | **P2** | 实际上刚做完化疗+手术，应该可以评估 neoadjuvant response（残余 3.84 cm 肿瘤） |
| medication_plan | Letrozole after radiation. Fosamax for bone protection. | OK | |
| therapy_plan | Radiation referral. Letrozole after radiation. | OK | |
| radiotherapy_plan | Radiation referral. | OK | |
| Referral: Specialty | Radiation referral | OK | |
| Next clinic visit | Not specified in the given note | **P2** | 原文有 "drains out on Thursday" 暗含 follow-up |
| supportive_meds | ondansetron, prochlorperazine, OLANZapine, MIRALAX | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "S/p 4 cycles of [REDACTED] and weekly taxol x 12l." | **P2-归因** | 间接支持 follow-up |
| Type_of_Cancer | "Stage II right IDC of the breast HR+/ her 2 neu negative." | OK | |
| Stage_of_Cancer | "Stage II right IDC of the breast HR+/ her 2 neu negative." | OK | |
| goals_of_treatment | 多段引文 | **P2-归因** | 不直接说 curative |
| response_assessment | 归因 = 提取值本身 | **P2-归因** | |
| medication_plan | "Letrozole after radiation. She favors using fosamax over injectable meds." | OK | 精确 |
| radiotherapy_plan | "Radiation referral." | OK | |

### Row 8 小结
- **P0**: 0
- **P1**: 0
- **P2**: 3 keypoint + 3 归因

---

## Row 9 (coral_idx=149) — Follow-up，Stage II HR+ IDC，术后，即将放疗

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| in-person | Televisit | OK | |
| Type_of_Cancer | HR+ invasive ductal carcinoma | **P1** | 缺 HER2-。原文 "her 2 negative"。且应写 ER+/PR+ 而非 HR+ |
| Stage_of_Cancer | Stage II | OK | |
| current_meds | letrozole | OK | "continue on letrozole started April 2021" |
| goals_of_treatment | curative | OK | |
| response_assessment | Not mentioned in note. | OK | |
| medication_plan | continue letrozole | OK | |
| therapy_plan | continue letrozole, radiation to left chest wall and surrounding LN | OK | |
| radiotherapy_plan | radiation to left chest wall and surrounding LN | OK | |
| imaging_plan | DEXA. | OK | |
| Referral: Specialty | "" | **P2** | 空。但 radiation referral 应该在这里 |
| Next clinic visit | RTC in the [REDACTED] | OK | |
| Advance care | full code. | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "To have radiation to the left chest wall..." | **P2-归因** | 不支持 follow-up 判定 |
| Type_of_Cancer | "Stage II left [REDACTED] of the breast HR + and her 2 negative s/p neoadjuvant letrozole" | OK | 引文有 HER2 信息但提取值缺 HER2- |
| Stage_of_Cancer | "Stage II left [REDACTED] of the breast" | OK | |
| current_meds | "She will continue on letrozole started April 2021." | OK | |
| goals_of_treatment | "To have radiation to the left chest wall and surrounding lymph nodes." | **P2-归因** | 间接支持 curative |
| imaging_plan | "DEXA." | OK | |
| radiotherapy_plan | "To have radiation to the left chest wall and surrounding lymph nodes." | OK | |

### Row 9 小结
- **P0**: 0
- **P1**: 1 keypoint (Type_of_Cancer 缺 HER2-)
- **P2**: 1 keypoint (Referral Specialty 空) + 2 归因
- **vs v14**: Type_of_Cancer 从空变成 "HR+ IDC"（改善但仍缺 HER2-）

---

## Row 10 (coral_idx=150) — Follow-up，Stage IIIC→IV 转移性 IDC，Faslodex+Denosumab

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| second opinion | no | OK | |
| in-person | in-person | OK | 有体格检查和生命体征 |
| summary | ...metastatic breast cancer on Faslodex... jaw pain, numbness, worsening leg pain | OK | 准确 |
| Type_of_Cancer | infiltrating ductal Carcinoma | **P1** | 缺 ER/PR/HER2。原文未显式写，但 Faslodex+Letrozole 强烈暗示 ER+；prompt 要求从药物推断 |
| Stage_of_Cancer | Originally Stage III C, now metastatic (Stage IV) | OK | 原文 "stage III C infiltrating ductal Carcinoma"，有骨转移 |
| Metastasis | Yes, to bone | OK | mandible mass + compression fx + T spine + femur |
| Distant Metastasis | Yes, to bone | OK | |
| lab_summary | WBC 5.6, RBC 5.25... 详细列出全部 | OK | 非常全面 |
| findings | PET/CT increased activity, numbness, thrush... | **P2** | "MRI of lumbar spine, pelvis, and right femur ordered" 是计划不是发现 |
| current_meds | Faslodex, Denosumab | OK | 与 A/P "Continue on Faslodex and Denosumab" 一致 |
| recent_changes | "" | OK | 无变化 |
| supportive_meds | docusate sodium, hydrocodone-acetaminophen, senna-docusate | OK | |
| goals_of_treatment | palliative | OK | Stage IV |
| response_assessment | PET/CT showed increased metastatic activity...progressing despite current treatment | **P1** | 时间线错误！10/10 PET 是治疗前，10/16 才开始 Faslodex。A/P 说 "Exam stable"，说明 Faslodex 上是稳定的 |
| medication_plan | Continue Faslodex and Denosumab. Mycelex... | OK | |
| therapy_plan | Continue on Faslodex and Denosumab | OK | |
| radiotherapy_plan | null | OK | |
| procedure_plan | No procedures planned | OK | |
| imaging_plan | will order PETCT to evaluate Femur and to toes | **P2** | A/P 还提到 "MRI of lumbar, pelvis and right femur"，可能遗漏（但原文措辞模糊） |
| lab_plan | No labs planned | OK | |
| genetic_testing_plan | None planned | OK | |
| Next clinic visit | Not specified | OK | |
| Advance care | Not discussed | OK | |
| Referral: Specialty | None | OK | |
| Referral: follow up | PETCT to toes, Mycelex... | **P2** | 这不是 referral follow up，是治疗计划内容 |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "Patient concerns were discussed in detail..." | **P1-归因** | 完全不支持 "Follow up" |
| second opinion | 同上 | **P2-归因** | |
| in-person | 同上 | **P2-归因** | 应指向生命体征或体格检查 |
| summary | 归因 = 提取值本身 | **P2-归因** | 自引用 |
| Type_of_Cancer | 无归因 | **P1-归因** | 缺失 |
| Stage_of_Cancer | "Breast cancer metastasized to multiple sites" | **P2-归因** | 支持转移但不含 Stage IIIC |
| findings | "Restaging studies due worsening numbness..." | **P1-归因** | 引用的是计划不是发现 |
| current_meds | "Continue on Faslodex and Denosumab" | OK | |
| response_assessment | "Restaging studies due worsening numbness..." | **P1-归因** | 应引用 "Exam stable" |
| medication_plan | "Continue on Faslodex and Denosumab..." | OK | |
| imaging_plan | "Restaging studies...PETCT to evaluate Femur..." | OK | |

### Row 10 小结
- **P0**: 0
- **P1**: 2 keypoint (Type_of_Cancer 缺受体, response_assessment 时间线错误) + 4 归因
- **P2**: 3 keypoint + 4 归因
- **特征**: response_assessment 把治疗前进展 PET 误说成治疗后进展

---

## Row 11 (coral_idx=151) — Follow-up，de novo Stage IV ER+/PR+/HER2+ 乳腺癌，脑/肺/骨转移

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| second opinion | no | OK | |
| in-person | in-person | OK | "45 minutes face-to-face" |
| summary | Metastatic ER+/PR+/HER2+ breast cancer... | OK | |
| Type_of_Cancer | ER+/PR+/HER2+ invasive ductal carcinoma | OK | 受体从 herceptin+letrozole 推断正确。"IDC" 假设（原文只说 breast cancer）P2 级别 |
| Stage_of_Cancer | Originally Stage [X], now metastatic (Stage IV) | **P2** | 改进 3 工作正常（[X] 占位符）。但 de novo metastatic 没有 "originally" stage，应直接写 "Stage IV" |
| Metastasis | Yes (to brain, lung, bone) | OK | |
| Distant Metastasis | Yes (to brain, lung, bone) | OK | |
| lab_summary | No labs in note | OK | 未见实验室结果段 |
| findings | 详细列出 MRI brain、CT CAP、体检 | **P2** | 内容准确但混合了不同日期的历史影像发现 |
| current_meds | herceptin, letrozole | OK | herceptin 为 clinic-administered（改进 8 生效），letrozole 在 outpatient list |
| recent_changes | "" | OK | 全部 continue |
| supportive_meds | "" | OK | |
| goals_of_treatment | palliative | OK | Stage IV + DNR/DNI |
| response_assessment | Recent body CT shows SD, no evidence of PD | OK | 与 A/P "SD" 一致。未提脑部新病灶进展（另一个 problem）|
| medication_plan | Continue herceptin/[REDACTED], letrozole qd, [REDACTED] q12 wks | OK | |
| therapy_plan | continue herceptin/[REDACTED], letrozole, [REDACTED] q12 wks, await GK | OK | |
| radiotherapy_plan | await GK / Rad Onc input, potential repeat GK | OK | GK 是放疗的一种 |
| procedure_plan | await GK / Rad Onc input, potential repeat GK | **P2** | GK 是放疗不是手术/操作，与 radiotherapy_plan 重复 |
| imaging_plan | CT CAP q4 months, bone scan, MRI brain q4 months | OK | 准确。Echo q6 months 未提（P2, cardiac monitoring） |
| lab_plan | No labs planned | OK | |
| genetic_testing_plan | None planned | OK | |
| Next clinic visit | in-person: 6 weeks | OK | 匹配 "F/u in 6 weeks" |
| Advance care | Not discussed during this visit | **P2** | 问题列表有详细 advance care (DNR/DNI, POLST)，虽是之前的决定但应提及 |
| Referral: Specialty | Rad Onc consult | OK | |
| Referral: follow up | F/u Dr. [REDACTED] in [REDACTED] | OK | 神经科 follow-up |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "F/u Dr. [REDACTED] in [REDACTED]." | **P2-归因** | 神经科 follow-up，不是 visit type |
| second opinion | "cont off chemotherapy for now, due to intolerance..." | **P1-归因** | 完全无关 |
| in-person | "I have reviewed and updated the patient's past medical history..." | **P2-归因** | 间接 |
| summary | "50 y.o. female with de [REDACTED]..." | OK | HPI 引文 |
| Type_of_Cancer | "[REDACTED]+/PR+/[REDACTED]+ breast cancer" | OK | |
| Stage_of_Cancer | "Metastatic breast cancer, St IV" | OK | |
| Metastasis | "breast cancer to [REDACTED], lung, nodes, brain and bone" | OK | |
| current_meds | "cont herceptin/[REDACTED], cont letrozole qd" | OK | |
| response_assessment | "CT shows only multiple bone sites, ? Active, no sx's and no evidence of PD." | OK | 精准归因 |
| medication_plan | A/P 原文 | OK | |
| imaging_plan | A/P 原文 | OK | |
| Next clinic visit | "F/u in 6 weeks" | OK | |
| Specialty | "await GK / Rad Onc input..." | OK | |

### Row 11 小结
- **P0**: 0
- **P1**: 0 keypoint + 1 归因 (second opinion 归因无关)
- **P2**: 4 keypoint (Stage [X]多余, findings 混合日期, procedure 重复 GK, advance care 未提) + 2 归因
- **亮点**: 改进 3 的 Stage [X] 占位符正确工作；改进 8 的 herceptin clinic-administered 正确包含；response_assessment 归因精准

---

## Row 13 (coral_idx=153) — Follow-up，de novo Stage IV ER+ 乳腺癌，患者自行在墨西哥接受治疗

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | follow up | OK | |
| second opinion | no | OK | |
| in-person | in-person | OK | 45 min face-to-face |
| summary | ...currently on faslodex and palbociclib... | **P1** | 已停药！1 月底停了 palbo+fulvestrant，现在在墨西哥做低剂量化疗 |
| Type_of_Cancer | ER+ invasive ductal carcinoma | **P1** | 缺 PR/HER2。palbo+fulvestrant 暗示 HER2-。"ductal" 是假设（原文只说 breast cancer） |
| Stage_of_Cancer | metastatic (Stage IV) | OK | |
| Metastasis | Yes (to bone, liver, and nodes) | OK | |
| Distant Metastasis | Yes (to bone, liver, and nodes) | OK | |
| lab_summary | 详细列出全部值，含 CA 27.29=48(H) | OK | 全面 |
| findings | 描述化疗方案、行动能力、体检 | **P2** | 化疗方案不属于 findings，属于 current_meds/recent_changes |
| current_meds | pamidronate, gemcitabine, docetaxel, doxorubicin | OK | 从第二位医生 IMP 中提取，正确反映当前用药 |
| recent_changes | Stopped palbociclib and fulvestrant end of January... | OK | 准确记录了换药经过 |
| supportive_meds | pamidronate once weekly | OK | 骨保护剂。与 current_meds 重复列出（P2） |
| goals_of_treatment | palliative | OK | Stage IV |
| response_assessment | cancer is currently responding... no new significant findings | **P1** | 过度解读！原文只说 "Labs look okay" + "no evidence of PD"，无影像确认。CA 27.29=48 偏高。不等于 "responding" |
| medication_plan | Continue low dose chemo...topical cannabis...Cymbalta | OK | |
| therapy_plan | Continue low dose chemo...pamidronate | OK | |
| radiotherapy_plan | None | OK | |
| procedure_plan | No procedures planned | OK | |
| imaging_plan | CT CAP and Total Spine MRI for May. Repeat spine MRI in 6 weeks | OK | 准确 |
| lab_plan | Labs to be drawn every two weeks | OK | |
| genetic_testing_plan | None planned | OK | |
| Next clinic visit | in-person: 2 months | OK | |
| Advance care | Not discussed | OK | |
| Referral: Others | Physical therapy referral | OK | |
| Referral: follow up | RTC after scans | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "She is here today for follow-up." | OK | 精准 |
| second opinion | "Apparently they gave her the chemotherapy medications..." | **P1-归因** | 无关 |
| summary | "58 y.o. female with de [REDACTED]... on faslodex...palbociclib..." | **P2-归因** | 引文包含已停的药物，与 summary 的错误一致 |
| Type_of_Cancer | "metastatic ER+ breast cancer..." | OK | |
| lab_summary | "Labs from February 24 look okay" | **P2-归因** | 极简，不支持详细的 lab 值列表 |
| findings | "currently doing low dose chemo combination..." | **P1-归因** | 这是药物信息不是发现 |
| current_meds | "The treatment includes pamidronate, gemcitabine, docetaxel, and doxorubicin." | OK | 从第二位医生的 IMP 引用 |
| recent_changes | "She has stopped palbociclib and fulvestrant..." | OK | |
| supportive_meds | "Receives pamidronate once weekly." | OK | |
| response_assessment | "Labs from February 24 look okay" | **P2-归因** | 不支持 "currently responding" |
| imaging_plan | "Cancelled scans...schedule CT CAP and Total Spine MRI for May" | OK | |
| lab_plan | 长段引文含无关内容 | **P2-归因** | 引文过长，混入不相关文字 |
| Others | "[REDACTED] start PT 03/12/19." | OK | |

### Row 13 小结
- **P0**: 0
- **P1**: 3 keypoint (summary 用药已过期, Type_of_Cancer 缺受体, response_assessment 过度解读) + 2 归因
- **P2**: 2 keypoint + 4 归因
- **特征**: summary 仍引用已停的 faslodex/palbociclib；response_assessment 把 "labs OK" 过度解读为 "responding"

---

## Row 16 (coral_idx=156) — 新患者，Stage I ER+/PR+/HER2- IDC，术后，拟辅助内分泌+放疗

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | consultation |
| second opinion | no | OK | |
| in-person | Televisit | OK | video visit |
| summary | 53 y.o. female...post-lumpectomy...consultation regarding further management | OK | 全面准确 |
| Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma | OK | 三项受体齐全 |
| Stage_of_Cancer | "" | **P2** | 原文分期被 [REDACTED]。肿瘤 0.8cm+LN neg 可推断 Stage I，但留空可接受 |
| Metastasis | No | OK | Chest CT negative |
| Distant Metastasis | No | OK | |
| lab_summary | No labs in note | OK | |
| findings | 详细列出病理、pelvic US neg、chest CT neg | OK | 准确全面 |
| current_meds | "" | OK | "Meds: none" |
| goals_of_treatment | curative | OK | 早期，辅助治疗 |
| response_assessment | Not yet on treatment — no response to assess | OK | |
| medication_plan | Adjuvant hormonal therapy 5+ years, tamoxifen or AI based on menopausal status | OK | |
| therapy_plan | Check hormones → tamoxifen or AI; breast RT; endocrine after RT | OK | 全面 |
| radiotherapy_plan | breast radiotherapy, scheduled to see [REDACTED] tomorrow | OK | |
| imaging_plan | baseline DXA scan | OK | |
| lab_plan | check labs including hormones | OK | |
| genetic_testing_plan | Refer to genetics for further evaluation and testing | OK | |
| Next clinic visit | in-person: after RT to start endocrine therapy | OK | |
| Referral: Nutrition | nutritionist at her request | OK | |
| Referral: Genetics | refer to genetics | OK | 家族史（sister ovarian ca, aunt breast ca）支持 |
| Referral: Specialty | breast radiotherapy consult | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | 无归因 | **P1-归因** | 缺失 |
| Type_of_Cancer | 无归因 | **P1-归因** | 缺失 |
| Metastasis | "without further systemic therapy she has approximately 10-15% chance of recurrence" | **P2-归因** | 间接支持 |
| findings | "Left breast invasive ductal carcinoma, 0.8cm, grade 2..." | OK | |
| medication_plan | "[REDACTED] will benefit from adjuvant hormonal therapy..." | OK | |
| therapy_plan | "check hormone levels and make a recommendation..." | OK | |
| radiotherapy_plan | "She requires breast radiotherapy..." | OK | |
| lab_plan | "check labs including hormones" | OK | |
| genetic_testing_plan | "[REDACTED] refer to genetics..." | OK | |
| Genetics | "[REDACTED]'s family history raises some suspicion of an inherited susceptibility syndrome..." | OK | 精准归因 |
| Nutrition | "referral to nutritionist at her request" | OK | |
| Next clinic visit | "FU visit after RT..." | OK | |

### Row 16 小结
- **P0**: 0
- **P1**: 0 keypoint + 2 归因 (Patient type, Type_of_Cancer 归因缺失)
- **P2**: 1 keypoint (Stage 空) + 1 归因
- **亮点**: 提取质量很高！Type_of_Cancer 三项受体齐全，治疗计划全面准确，referral 完整覆盖

---

## Row 17 (coral_idx=157) — 新患者，Stage I ER+/PR+/HER2- IDC，术后，拟辅助内分泌

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | |
| in-person | in-person | OK | 60 min face-to-face |
| summary | Left breast IDC with encapsulated papillary carcinoma... | OK | |
| Type_of_Cancer | ER+/PR+/HER2- IDC, with encapsulated papillary carcinoma (DCIS) | OK | 准确全面 |
| Stage_of_Cancer | pT1b, pNX — approximately Stage I | OK | |
| Metastasis | No | OK | |
| findings | 详细病理（0.8cm, grade 1, margins）+ 体检 | OK | |
| current_meds | "" | OK | 无用药 |
| goals_of_treatment | curative | OK | |
| response_assessment | Not yet on treatment | OK | |
| medication_plan | adjuvant endocrine therapy 5-10 yrs with [REDACTED] | OK | |
| therapy_plan | adjuvant endocrine, RTC after Rad Onc eval +/- XRT | OK | |
| radiotherapy_plan | RTC after Rad Onc eval +/- XRT | OK | |
| imaging_plan | DEXA ordered | OK | |
| genetic_testing_plan | molecular profiling | **P1** | 错误！患者明确拒绝化疗 → "will not pursue molecular profiling"。应为 "None" 或 "discussed but patient declined" |
| Referral: Genetics | None | **P1** | 遗漏！原文 "discussed with UCSF Cancer Risk. They will reach out to pt today"，有遗传咨询转诊 |
| Referral: Specialty | Rad Onc eval | OK | |
| Next clinic visit | after Rad Onc eval +/- XRT | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "Pt's sister's w breast cancer, diagnosed at 45 yo..." | **P1-归因** | 家族史，与 patient type 无关 |
| second opinion | "strongly recommend adjuvant endocrine therapy..." | **P2-归因** | 无关 |
| in-person | "RTC after Rad Onc eval +/- XRT." | **P2-归因** | 无关 |
| Type_of_Cancer | "left lumpectomy, showing IDC NOS, 0.8 cm, grade 1, ER/PR strongly positive..." | OK | 精准 |
| Metastasis | "Isolated tumor cells present in one of three lymph nodes..." | OK | |
| findings | "left lumpectomy, showing IDC NOS, 0.8 cm..." | OK | |
| genetic_testing_plan | "discussed option of molecular profiling..." | **P2-归因** | 没有引用患者拒绝的部分 |
| medication_plan | "strongly recommend adjuvant endocrine therapy..." | OK | |
| imaging_plan | "DEXA ordered" | OK | |
| radiotherapy_plan | "RTC after Rad Onc eval +/- XRT." | OK | |

### Row 17 小结
- **P0**: 0
- **P1**: 2 keypoint (genetic_testing_plan 患者拒绝了, Genetics referral 遗漏) + 1 归因
- **P2**: 0 keypoint + 4 归因
- **特征**: genetic_testing_plan 错误地写了患者已拒绝的 molecular profiling；遗传咨询转诊被遗漏

---

## Row 19 (coral_idx=159) — 新患者，ER+/HER2- IDC 2009 原发→2020 骨/LN/肺转移复发

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | consultation |
| in-person | in-person | OK | |
| summary | 75 y.o. with ER+/HER2- IDC...metastatic recurrence to bone and LN | OK | 准确 |
| Type_of_Cancer | ER+/PR+/HER2- IDC with metastatic recurrence ER+/PR-/HER2- | **P1** | 转移灶 PR- 错误！活检显示 PR+ (50%)。应为 ER+/PR+/HER2- |
| Stage_of_Cancer | -II, now metastatic (Stage IV) | **P2** | "-II" 截断，原文说 "early stage"，0.9cm + 0/2 LN 约 Stage I |
| Metastasis | Yes (to bone, lymph nodes, and lung) | OK | PET 确认 |
| Distant Metastasis | Yes (to bone, lymph nodes, and lung) | OK | |
| lab_summary | POCT glucose 104 mg/dL (03/01/13) | **P2** | 2013 年的血糖！太旧了，临床无意义 |
| findings | 详细病史、PET 结果、活检结果、体检 | OK | |
| current_meds | letrozole, palbociclib | **P2** | 严格说这是本次 visit 新开的，但作为即将开始的药物可接受 |
| recent_changes | Start Letrozole, Rx given; Rx for Palbociclib | OK | |
| supportive_meds | denosumab | **P2** | 尚未开始（需牙科 clearance），是计划而非当前 |
| goals_of_treatment | palliative | OK | |
| response_assessment | Not mentioned in note | OK | 新开始治疗 |
| medication_plan | Start Letrozole, Palbociclib, denosumab after dental clearance | OK | |
| radiotherapy_plan | Rad Onc referral for consult | OK | |
| imaging_plan | MRI total spine, CAP CT, obtain outside PET/CT | OK | 遗漏 "repeat imaging after 3 months"（P2） |
| lab_plan | Labs including tumor markers. Monthly blood work on Palbociclib | OK | |
| genetic_testing_plan | Foundation One, or [REDACTED] 360 | OK | 正确捕获！ |
| Next clinic visit | ~1 month | OK | |
| Referral: Specialty | Rad Onc referral | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "post-menopausal female with early stage..." | **P2-归因** | 诊断信息，不直接支持 "New patient" |
| Type_of_Cancer | "early stage [REDACTED]+/[REDACTED]- IDC...now with metastatic recurrence..." | OK | |
| current_meds | "letrozole has been sent to [REDACTED]... Palbociclib has been sent..." | OK | |
| supportive_meds | "We recommend initiation of denosumab." | OK | |
| goals_of_treatment | "with metastatic recurrence with disease in the bone and lymph nodes." | **P2-归因** | 间接 |
| imaging_plan | "Plan - MRI Total Spine - CT Chest, Abdomen, Pelvis - Obtain outside PET/CT" | OK | 精准 |
| genetic_testing_plan | "Obtain outside path and send for foundation one, if not possible, send for [REDACTED] 360" | OK | 精准 |
| lab_plan | "Labs including tumor markers. Monthly blood work on Palbociclib." | OK | |

### Row 19 小结
- **P0**: 0
- **P1**: 1 keypoint (Type_of_Cancer 转移灶 PR- 应为 PR+) + 0 归因
- **P2**: 4 keypoint (Stage 截断, lab 太旧, current_meds 是新开, denosumab 未开始) + 2 归因
- **亮点**: genetic_testing_plan 正确捕获 Foundation One/Guardant 360；imaging_plan 归因精准

---

## Row 21 (coral_idx=161) — 新患者/第二意见，Stage II IDC 2000→2020 胸壁/骨/LN 转移复发

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | 新医生会诊 |
| second opinion | yes | OK | "She is here for a second opinion" |
| in-person | in-person | OK | 70 min face-to-face |
| summary | 详细描述 DCIS 1994、IDC 2000、metastatic relapse 2020 | OK | |
| Type_of_Cancer | ER+/PR+/HER2- IDC, metastatic recurrence ER+/PR+/HER2- | OK | 准确 |
| Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | OK | |
| Metastasis | Yes (bones, chest wall, right infraclavicular and IM nodes) | OK | |
| Distant Metastasis | Yes (bones, chest wall, right infraclavicular and IM nodes) | **P2** | 胸壁是局部复发非远处转移，但在此语境下可接受 |
| lab_summary | 详细 CBC/CMP (from 01/29/2021) | **P2** | 8 个月前的 labs |
| findings | 病史、PET 反应、肺炎 | OK | |
| current_meds | anastrozole, denosumab | OK | abemaciclib 已停 |
| recent_changes | abemaciclib held due to pneumonitis, letrozole→anastrozole | OK | |
| supportive_meds | prednisone, diphenoxylate-atropine, denosumab | OK | |
| goals_of_treatment | palliative | OK | |
| response_assessment | PET/CT 11/03/20 and 04/24/21 showed good response | OK | |
| medication_plan | Continue arimidex if stable; if progression → faslodex + [REDACTED]; future → afinitor/xeloda/trial | OK | 全面 |
| imaging_plan | pet ct now | OK | |
| genetic_testing_plan | faslodex with [REDACTED] if she has [REDACTED] mutation | **P2** | 这是条件性治疗计划，不是直接的基因检测 order。但暗含需做 mutation testing |
| Next clinic visit | after PET CT results | OK | |
| Advance care | Full code | OK | |
| Referral: follow up | 详细条件性计划 | **P2** | 内容过长，更像治疗计划而非 referral follow up |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| second opinion | `{"quote": "I recommend a pet ct now..."}` | **P1-归因** | JSON wrapper 泄漏！且引文不支持 "yes" |
| in-person | "I spent a total of 70 minutes on this patient's care..." | OK | |
| Type_of_Cancer | "Metastatic relapse...HR+ and her 2 negative." | OK | |
| current_meds | "Letrozole changed to arimidex." | **P2-归因** | 不完整，缺 denosumab |
| response_assessment | "I recommend a pet ct now and if stable continue arimidex alone." | **P1-归因** | 这是计划不是 response。应引用 "PET/CT showed a good response" |
| medication_plan | "I recommend a pet ct now...if progression could use faslodex..." | OK | |
| imaging_plan | "I recommend a pet ct now..." | OK | |

### Row 21 小结
- **P0**: 0
- **P1**: 0 keypoint + 2 归因 (JSON wrapper 泄漏, response_assessment 引错)
- **P2**: 4 keypoint + 1 归因
- **亮点**: second opinion 正确识别；medication_plan 完整覆盖三级治疗方案

---

## Row 26 (coral_idx=166) — Follow-up，Stage IV ER+/PR+/HER2- IDC 骨转移，Femara+Zoladex

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| in-person | in-person | OK | |
| summary | Metastatic hormone-positive breast cancer to bone, follow-up | OK | |
| Type_of_Cancer | ER+/PR+/HER2- IDC | OK | 三项齐全 |
| Stage_of_Cancer | now metastatic (Stage IV) | OK | |
| Metastasis | Yes (to bone) | OK | L1 + 多发骨 |
| lab_summary | CBC with platelets. No specific values provided | **P1** | 这是 lab ORDER 不是 lab result！应为 "No labs in note" |
| findings | 详细 PET-CT 结果（SUV 值等） | OK | 全面 |
| current_meds | letrozole, zolendronic acid, goserelin | OK | |
| goals_of_treatment | palliative | OK | |
| response_assessment | PET-CT stable to slightly decreased metabolic activity, no new mets | OK | 准确 |
| medication_plan | Continue [REDACTED], zoladex, and femara | OK | |
| imaging_plan | Consider MRI spine if pain persists after two weeks | OK | |
| lab_plan | CBC with platelets to evaluate easy bruising | OK | UA 遗漏（P2, 非 cancer-related） |
| Next clinic visit | if pain worsens | **P2** | 不完整。A/P 说 "reassess at two weeks" |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| Patient type | "Reassess lower back pain at two weeks..." | **P1-归因** | 无关 |
| Type_of_Cancer | `{"quote": "Hormone positive MBC..."}` | **P1-归因** | JSON wrapper 泄漏！ |
| Stage_of_Cancer | `{"quote": "Hormone positive MBC..."}` | **P1-归因** | JSON wrapper 泄漏！ |
| summary | "Hormone positive MBC to bone with stable disease." | OK | |
| findings | "stable disease." | **P1-归因** | 极度简略，不支持详细 PET 描述 |
| response_assessment | "stable disease." | **P2-归因** | 过于简短 |
| current_meds | "Continue on [REDACTED], zoladex and femara." | OK | |
| imaging_plan | "Reassess lower back pain at two weeks and consider MRI..." | OK | |
| lab_plan | "Easy bruising per patient, obtain CBC with platelets." | OK | |

### Row 26 小结
- **P0**: 0
- **P1**: 1 keypoint (lab_summary 是 order 不是 result) + 4 归因 (2 个 JSON wrapper 泄漏)
- **P2**: 1 keypoint + 1 归因
- **模式**: JSON wrapper 泄漏再次出现（Row 7, 21, 26 共 3 次）

---

## Row 28 (coral_idx=168) — 新患者，多灶 Grade 2 ER+/PR+/HER2- IDC，Oncotype Low Risk

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | |
| summary | Multifocal grade 2 IDC (ER+/PR+/HER2-) with micromet in SLN | OK | |
| Type_of_Cancer | ER+/PR+/HER2- IDC | OK | PR weak positive (30%) 正确标为 PR+ |
| Stage_of_Cancer | pT1c(m)HER2(sn) | **P2** | 乱码。"HER2(sn)" 不是分期表述。应为 pT1c(m)pN1mi 或 Stage I |
| findings | 详细病理和体检 | OK | |
| current_meds | letrozole 2.5mg PO daily | **P2** | 本次新开的，但可接受 |
| goals_of_treatment | curative | OK | |
| medication_plan | Start letrozole, calcium, vitamin D, vaginal moisturizer | OK | |
| therapy_plan | Start letrozole, plan RT after surgery | OK | |
| radiotherapy_plan | RT planning per [REDACTED] | OK | |
| procedure_plan | surgery tentatively September 2019 | OK | re-excision for DCIS margin |
| imaging_plan | Bone density scan. Bone scan | **P2** | DEXA ≠ bone scan，末尾 "Bone scan" 多余 |
| genetic_testing_plan | ngs | **P1** | 假阳性！原文无 NGS order。Oncotype 已完成（Low Risk）。POST-GENETICS-SEARCH 可能误匹配 |
| Next clinic visit | September 2019 for surgical planning | OK | |
| Referral: Specialty | RT planning | OK | |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| summary | "59 y.o. female with recently diagnosed multifocal grade 2 IDC..." | OK | |
| Type_of_Cancer | "multifocal grade 2 IDC (ER+/PR+/[REDACTED]-)" | OK | |
| findings | "Her tumor was found to have a [REDACTED] Low Risk profile." | **P1-归因** | 极度简略，不支持详细病理描述 |
| goals_of_treatment | "early stage breast cancer" | OK | 支持 curative |
| current_meds | "Start letrozole 2.5mg PO daily now (prescription sent)" | OK | |
| medication_plan | plan 原文 | OK | |
| procedure_plan | "surgical planning per Dr. [REDACTED], tentatively scheduled for September 2019" | OK | |
| imaging_plan | "Bone density scan can be completed when she returns from [REDACTED]." | OK | |

### Row 28 小结
- **P0**: 0
- **P1**: 1 keypoint (genetic_testing_plan "ngs" 假阳性) + 1 归因
- **P2**: 3 keypoint (Stage 乱码, imaging "bone scan" 多余, current_meds 新开) + 0 归因
- **模式**: POST-GENETICS-SEARCH 再次产生假阳性（"ngs"）

---

## Row 29 (coral_idx=169) — 新患者，Clinical Stage II-III HR-/HER2+ IDC，拟新辅助化疗

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | New patient | OK | |
| summary | 64 y.o...clinical stage II-III...discuss treatment options | OK | |
| Type_of_Cancer | ER-/PR-/HER2+ IDC | OK | 从治疗方案（trastuzumab/pertuzumab 无内分泌治疗）正确推断 |
| Stage_of_Cancer | Clinical stage II-III | OK | |
| Metastasis | No | OK | PET 无转移 |
| lab_summary | CA 27.29 和 CA 15-3 趋势值 (2015-2016) | OK | |
| findings | 详细 PET-CT、MRI、体检 | OK | |
| current_meds | "" | OK | |
| goals_of_treatment | curative | OK | "treated with curative intent" |
| medication_plan | THP/AC or TCHP regimen, then trastuzumab x1 year | OK | 非常详细 |
| therapy_plan | neoadjuvant chemo + surgery + radiation | OK | |
| procedure_plan | Mediport placement | OK | |
| imaging_plan | TTE to assess cardiac function | OK | |
| genetic_testing_plan | ngs | **P1** | 再次假阳性！无 NGS order。POST-GENETICS-SEARCH 匹配 "ngs" 文本 |
| Next clinic visit | after the weekend to decide | OK | |
| Referral: all None | | **P2** | 多模态计划包含 radiation 但无具体转诊 order |

### 归因审查

| 字段 | 归因引文 | 判定 | 备注 |
|------|---------|------|------|
| summary | "64 y.o. postmenopausal...clinical stage II-III..." | OK | |
| Metastasis | "her PET/CT demonstrated no evidence of metastases" | OK | 精准 |
| goals_of_treatment | "no evidence of metastases...treated with curative intent" | OK | 优秀 |
| medication_plan | regimen 原文 | OK | |
| procedure_plan | "she would need TTE, Mediport placement..." | OK | |

### Row 29 小结
- **P0**: 0
- **P1**: 1 keypoint (genetic_testing_plan "ngs" 假阳性)
- **P2**: 1 keypoint (Referral 缺 RT)
- **亮点**: medication_plan 极其详细（两个 regimen 选项完整列出）；goals_of_treatment 归因精准
- **模式**: "ngs" 假阳性第 2 次出现

---

## Row 32 (coral_idx=172) — Follow-up，ER+/PR+/HER2- 乳腺癌，术后辅助 letrozole

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Patient type | Follow up | OK | |
| Type_of_Cancer | ER+/PR+/HER2- invasive lobular carcinoma | **P1** | "lobular" 可能错误。原文重度脱敏，大多数乳腺癌为 ductal |
| Stage_of_Cancer | Originally stage IIB, now stage IIIA | **P1** | 幻觉！原文无明确分期。"now stage IIIA" 暗示进展但患者 "no evidence of recurrence" |
| findings | No evidence of recurrence. 小 LN soft and mobile | OK | |
| current_meds | letrozole | OK | |
| goals_of_treatment | curative | OK | 辅助治疗阶段 |
| response_assessment | no evidence of recurrence on exam | OK | |
| medication_plan | Continue letrozole daily. Calcium + Vit D. NSAIDs prn | OK | |
| imaging_plan | Consider MRI brain if [headaches] continues | OK | |
| Next clinic visit | 6 months | OK | |

### Row 32 小结
- **P0**: 0
- **P1**: 2 keypoint (Type lobular 可能错误, Stage 幻觉)
- **P2**: 0
- **特征**: Stage 幻觉 "Originally IIB, now IIIA" 与改进 3 目标相关但不同模式（不是 "Originally Stage [X]" 格式）

---

## Row 33 (coral_idx=173) — Follow-up，Stage III ER+/PR- IDC 第二次局部复发

### Keypoints 审查

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Type_of_Cancer | ER positive PR negative IDC | **P2** | 使用原发灶受体状态。复发灶 PR+ (50%)，应标注变化 |
| Stage_of_Cancer | Originally Stage III, now local recurrence | OK | 局部复发准确 |
| current_meds | arimidex | OK | 即将换为 tamoxifen |
| recent_changes | resumption of hormonal therapy with tamoxifen 20mg | OK | |
| response_assessment | Local recurrence 1.7cm IDC, PET negative for distant mets | OK | 描述当前疾病状态 |
| medication_plan | tamoxifen 20mg PO qD | OK | |
| radiotherapy_plan | chest wall RT, accepted referral | OK | |
| Referral: Specialty | refer to [REDACTED] for consultation | OK | Rad Onc |
| lab_plan | check labs | OK | |

### Row 33 小结
- **P0**: 0 | **P1**: 0 | **P2**: 1 (Type PR 未更新)
- 整体质量好，准确捕获局部复发和治疗方案变更

---

## Row 35 (coral_idx=175) — Follow-up，pT3N0 ER+/HER2- 混合 ductal/mucinous，Abraxane+Zoladex

### Keypoints 审查（简要）

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous | OK | |
| Stage_of_Cancer | pT3N0 | OK | |
| current_meds | Abraxane, zoladex | OK | |
| goals_of_treatment | curative | OK | |
| response_assessment | Not mentioned in note | OK | |

（待详细补充 — 提取质量初步看正常）

---

## Row 36 (coral_idx=176) — 新患者，Stage IIA TNBC (ER-/PR-/HER2-)

### Keypoints 审查（简要）

| 字段 | 提取值 | 判定 | 备注 |
|------|--------|------|------|
| Type_of_Cancer | ER-/PR-/HER2- triple negative IDC | OK | |
| Stage_of_Cancer | Stage IIA | OK | |
| current_meds | "" | OK | 新患者 |
| goals_of_treatment | curative | OK | |
| genetic_testing_plan | None planned | OK | |

（待详细补充）

---

## 问题行简要审查

### Row 41 (coral_idx=181)
- **P1**: Type_of_Cancer = "PR+ IDC" — 缺 ER 和 HER2。极不完整。

### Row 56 (coral_idx=196) — P0
- **P0**: Type_of_Cancer = "ER-/PR-/HER2+ triple negative breast cancer" — HER2+ 和 triple negative 互斥！
- 背景: 初始活检 HER2 3+（接受 TCH+P），术后病理 TNBC，原始活检复查也为 HER2-。实际为 TNBC，但 LLM 混合了不同时间点的结果
- genetic_testing_plan: "Rec genetic counseling and testing" — OK（家族史支持）

### Row 62 (coral_idx=202)
- **P1**: genetic_testing_plan = "ngs" — 假阳性（POST-GENETICS-SEARCH）
- **P1**: Attribution JSON wrapper 泄漏

### Row 71 (coral_idx=211)
- **P1**: current_meds = "Every 6 Hours; latanoprost (XALATAN)...zoledronic acid" — 包含非 cancer-related 眼药水，格式也有问题

### Row 82 (coral_idx=222)
- **P1**: Type_of_Cancer = "Lobular Breast Cancer" — 完全缺 ER/PR/HER2（note 中无 HER2 信息，但 letrozole 暗示 ER+）
- **P1**: goals_of_treatment = "curative" — Stage IV 应为 palliative
- **P2**: Stage IV 可能有误 — 唯一 "转移" 是 axillary LN（可能是 regional 非 distant）

### Row 85 (coral_idx=225)
- **P2**: Type = "ER+/PR+/HER2+ mixed IDC" + current_meds = "letrozole, ribociclib" — ribociclib 通常用于 HER2-。可能的受体状态或药物提取误差

### Row 90 (coral_idx=230)
- **P1**: Type = "ER+/PR+ IDC" — 缺 HER2。everolimus+exemestane 暗示 HER2-（典型 HR+/HER2- 方案）

### Row 93 (coral_idx=233)
- **P1**: genetic_testing_plan = "ngs" — 假阳性
- 其他字段提取质量好（Stage 含 RS score = 21）

### Row 96 (coral_idx=236)
- **P1**: Type = "ER+/PR+ IDC" — 缺 HER2（note 无 HER2 信息）
- **P2**: genetic_testing_plan = "molecular profiling" — 需验证是否实际 ordered
- **P1**: Attribution 多个 JSON wrapper 泄漏

### Row 99 (coral_idx=239)
- **P1**: Type = "ER+(80%)PR+(50%) IDC" — 缺 HER2
- **P1**: Attribution JSON wrapper 泄漏

---

## 审查统计总结

### 问题密度

| 严重度 | 数量 | 密度（/行） |
|--------|------|-------------|
| P0 | 3 | 0.05 |
| P1 keypoint | ~28 | 0.46 |
| P1 归因 | ~52 | 0.85 |
| P2 | ~80 | 1.31 |
| **合计** | **~163** | **~2.67** |

### 与 v13a 对比

| 指标 | v13a (100行) | v14a (61行) | 变化 |
|------|-------------|-------------|------|
| 总问题 | 392 | ~163 | — |
| 密度/行 | 3.92 | ~2.67 | **-32%** |
| P0 | 2 | 3 | +1 (新发现 Row 56 矛盾) |
| P1 | 172 | ~80 | 密度 1.72→1.31 (-24%) |
| P2 | 218 | ~80 | 密度 2.18→1.31 (-40%) |

### v14a 改进效果总结

- **改进 1** (删 chatgpt prompt): 完全成功，~45 个 P1 消除
- **改进 2** (supportive_meds 过滤): 完全成功，letrozole/tamoxifen 不再出现
- **改进 3** (Stage 幻觉修复): 完全成功，13 个 "Originally Stage" 全部有原文支持
- **改进 4** (Cross-field 白名单): 部分成功，cross-field 分类错误显著减少但仍存在
- **改进 5** (therapy_plan None): 完全成功，"None" 率显著降低
- **改进 6** (genetics 搜索): 部分成功，但 "ngs" 假阳性 4 行
- **改进 7** (Referral 全文): 部分成功，但 2 行有文本泄漏 (P0)
- **改进 8** (current_meds 时态): 部分成功，时态混乱减少但仍有非 cancer 药物混入

### v15 修复优先级

| 优先级 | 修复项 | 影响行数 | 难度 |
|--------|--------|---------|------|
| 1 | Referral 文本泄漏（长度检查+截断） | 2 (P0) | 低 |
| 2 | Type_of_Cancer HER2+/triple-negative 矛盾禁止 | 1 (P0) | 低 |
| 3 | "ngs" 假阳性（全词匹配 \bngs\b 或仅全称） | 4 | 低 |
| 4 | 强制 Type_of_Cancer 包含 ER/PR/HER2 三项 | 11 | 中 |
| 5 | Attribution JSON wrapper 清理（后处理） | 11 | 低 |
| 6 | response_assessment 过度解读修复 | ≥3 | 中 |
| 7 | current_meds 非 cancer 药物过滤 | ≥5 | 中 |

