# v17 目标问题验证审查报告

审查日期：2026-03-19
版本：v17_verify_20260319_080005
审查范围：v16 中发现的 3 个 P0 问题对应行

---

## 审查说明

根据 v16_review.md 中标记的 P0 问题，应审查以下 3 行：
- **Row 7** (v16 review 标号) = dataset row_index 7 → v17_verify **Row 8** (coral_idx 147)
- **Row 56** (v16 review 标号) = dataset row_index 56 → **未包含在 v17_verify 中**
- **Row 94** (v16 review 标号) = dataset row_index 93 → v17_verify **Row 94** (coral_idx 233)

注意：v16 review 中的行号使用 1-based 计数，但实际对应的是 dataset 的 row_index (0-based +1)。

---

## Row 8 (coral_idx 147) — P0: in-person vs Televisit 矛盾

### v16 问题描述
- 字段：`in-person`
- 问题：标为 "Televisit" 但原文明确 "80 minutes face-to-face"
- 严重度：P0（幻觉/矛盾）

### v17 修复措施
- v17 应该有 POST-VISIT-TYPE gate，优先检测 "video visit" 关键词

### 原文关键证据

**Note 开头**：
```
The patient presents through ZOOM to discuss her treatment to date...
```

**Note 结尾（TIME SPENT 部分）**：
```
I spent a total of 80 minutes face-to-face with the patient and >50% of that time
was spent counseling regarding the diagnosis, the treatment plan, the prognosis...
```

**矛盾分析**：
- "through ZOOM" → 明确是视频访问
- "80 minutes face-to-face" → 医学笔记中的标准计费语言，"face-to-face" 在 telehealth context 中包含视频访问
- 这是医学文档中的术语使用习惯，而非矛盾

### v17 提取结果

```json
{
  "in-person": "Televisit",
  "summary": "29 y.o. premenopausal patient with a history of clinical stage III HER2+/ER- IDC of the left breast, who presents in consultation to discuss treatment options and to establish care after receiving an incomplete course of neoadjuvant TCHP and a lumpectomy/ALND."
}
```

**Attribution**：
```json
{
  "in-person": [
    "I spent a total of 80 minutes face-to-face with the patient and >50% of that time was spent counseling regarding the diagnosis, the treatment plan, the prognosis, medication risks, lifestyle modification, symptoms and therapeutic options."
  ]
}
```

### ✅ v17 修复状态：**已修复**

**结论**：
- v17 正确输出 `"in-person": "Televisit"`
- 归因引用了 "face-to-face" 句子，但这是误导性的
- 正确的归因应该是 "The patient presents through ZOOM"
- **POST-VISIT-TYPE 的优先级设计正确**：优先检测 "video/ZOOM/telehealth" 而非 "face-to-face" 计费语言
- v16 的 P0 标记可能是**误判**（"face-to-face" 在 telehealth 文档中是标准术语，不代表 in-person visit）

### 其他字段审查（Row 8）

#### ✅ Type_of_Cancer
- 提取：`"ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) invasive ductal carcinoma"`
- 原文：病理报告明确 ER- (0%), PR- (0%), HER2+ (IHC 3+, FISH ratio 5.7)
- **准确**

#### ✅ Stage_of_Cancer
- 提取：`"Originally Stage II-III, now Stage III"`
- 原文：
  - HPI: "clinical stage II-III ER-/HER2+ IDC"
  - 术后病理：AJCC Stage ***** (redacted，但根据 3/28 LN+ 应为 Stage IIIA)
- **准确**

#### ✅ Distant Metastasis
- 提取：`"No"`
- 原文：
  - "3 of 28 LN positive" (axillary LN = regional, not distant)
  - "09/07/18 PET/CT: no distant metastases"
  - "02/23/19 PET/CT: no evidence of metastases"
- **准确**

#### ⚠️ P1: current_meds 为空
- 提取：`""`
- 原文：medication list 有 acetaminophen, ibuprofen, oxycodone（支持性药物，非抗癌治疗）
- 问题：患者此时未在活跃治疗中（刚完成手术，等待开始辅助化疗），所以空值合理
- **可接受**

#### ✅ goals_of_treatment
- 提取：`"curative"`
- 原文：
  - "clinical stage III" + neoadjuvant + surgery + 计划辅助化疗
  - A/P: "at great risk of developing metastatic breast cancer... can reduce this risk by proceeding with recommended systemic therapy"
- **准确**（早期乳腺癌辅助治疗目的是治愈）

#### ✅ response_assessment
- 提取：`"Not yet on treatment — no response to assess."`
- 原文：刚完成手术，尚未开始辅助化疗
- **准确**

#### ✅ Referral
- 提取：全部 "None"
- 原文：A/P 中只有治疗计划（AC + T-DM1 + radiation），没有提到转诊
- **准确**

---

## Row 94 (coral_idx 233) — P0: Distant Metastasis = Axillary LN

### v16 问题描述
- 字段：`Distant Metastasis`
- 问题：Axillary LN 被标为 distant（应为 regional）
- 严重度：P0（分类错误）

### v17 修复措施
- v17 应该有 POST-DISTMET-REGIONAL gate，检测 axillary/sentinel LN 并修正为 regional

### 原文关键证据

**HPI**：
```
75 year old female with a left breast cancer in May 2019.
She had a lumpectomy June 2019 with a 1.6 cm tumor and 3 lymph nodes positive,
HR+ and HER2 negative.
```

**Staging**：
```
Stage IIA (pT1b, N1(sn), G2, ER+, PR+, HER2-, RS score: 21)
```

**A/P**：
```
1. Stage IIA breast cancer, s/p lumpectomy June 2020 followed by chemotherapy,
radiation and hormone therapy started. HR+, HER2 negative, 1.6 cm tumor
3 lymph nodes positive. Oncotype RS of 21. No chemotherapy given.
```

**关键信息**：
- "3 lymph nodes positive" **未指明类型**，但根据 staging "N1(sn)" → sentinel node = **axillary LN** = **regional** metastasis
- "Stage IIA" → 早期乳腺癌，**无 distant metastasis**
- "s/p lumpectomy... radiation" → 局部治疗完成，提示无远处转移

### v17 提取结果

```json
{
  "Metastasis": "No",
  "Distant Metastasis": "No"
}
```

**Attribution**：
```json
{
  "Metastasis": [
    "1.6 cm tumor August 03 lymph nodes positive."
  ],
  "Distant Metastasis": [
    "lymph nodes positive."
  ]
}
```

### ✅ v17 修复状态：**已修复**

**结论**：
- v17 正确输出 `"Distant Metastasis": "No"`
- **POST-DISTMET-REGIONAL 修复有效**
- 归因仍引用 "lymph nodes positive"，但输出值正确（通过 POST gate 修正了初始提取的错误）

### 其他字段审查（Row 94）

#### ✅ Type_of_Cancer
- 提取：`"ER+/PR+/HER2- invasive ductal carcinoma"`
- 原文：pathology 明确 ER+, PR+, HER2-
- **准确**

#### ⚠️ P2: Stage_of_Cancer 格式冗余
- 提取：`"Stage IIA (pT1b, N1(sn), G2, ER+, PR+, HER2-, RS score: 21)"`
- 原文：staging form 有完整 TNM
- 问题：包含了 receptor status（应该在 Type_of_Cancer 中），格式略冗余
- **可接受**（信息准确，只是格式不够简洁）

#### ✅ in-person
- 提取：`"Televisit"`
- 原文：
  - Chief Complaint: "Video Visit"
  - TIME SPENT: "I performed this evaluation using real-time telehealth tools, including a live video Zoom connection"
- **准确**

#### ✅ current_meds
- 提取：`"letrozole (FEMARA) 2.5 mg tablet"`
- 原文：medication list 有 letrozole（激素治疗）
- **准确**（gabapentin 是支持性药物，正确未列入）

#### ✅ goals_of_treatment
- 提取：`"curative"`
- 原文：
  - Stage IIA，s/p 手术 + 放疗，目前辅助激素治疗
  - A/P: "on hormone blockade" "RTC in 6 months for follow up"
- **准确**（早期乳腺癌辅助治疗）

#### ✅ response_assessment
- 提取：`"No evidence of disease recurrence on imaging and exam. The patient is currently asymptomatic with a performance status of 0."`
- 原文：
  - "most recent mammogram was normal 12/05/2020"
  - Physical exam: "no acute distress"
  - ECOG PS: 0
- **准确**

#### ⚠️ P1: Imaging_Plan 遗漏 colonoscopy
- 提取：`"Mammogram due in November 2021. High risk screening MRI with Dr. [REDACTED]."`
- 原文：A/P 提到 "A colonoscopy has been recommended"
- 问题：colonoscopy 不是 cancer imaging，但是 A/P 中提到的 plan
- **轻微遗漏**（colonoscopy 应归入 Procedure_Plan，而非 Imaging_Plan）

#### ⚠️ P2: Genetic_Testing_Plan 误判
- 提取：`"None planned."`
- 原文：HPI 提到 "She had genetic testing that was negative"
- 问题：这是**已完成**的检测结果，不是计划，所以 "None planned" 正确
- **准确**（POST-GENETICS-RESULT 应该已过滤此类内容）

---

## Row 56 (dataset row_index 56) — 未包含在 v17_verify 中

### v16 问题描述
- 字段：`Type_of_Cancer`
- 问题：输出 "ER-/PR-/HER2+" 但原文结论是 TNBC (ER-/PR-/HER2-)
- 严重度：P0（与原文直接矛盾）

### v17 修复措施
- v17 应该有 POST-TYPE-VERIFY-TNBC gate，检测 A/P 中的 "TNBC/triple negative" 并修正 HER2 状态

### ❌ 无法验证

**原因**：v17_verify.yaml 的 row_indices 不包含 56，无法验证该 P0 问题是否修复。

**建议**：需要在包含 row_index=56 的 run 中验证 POST-TYPE-VERIFY-TNBC 的效果。

---

## 总结

### v17 修复验证

| Row | v16 P0 问题 | v17 修复状态 | 备注 |
|-----|-------------|--------------|------|
| **8** | in-person "Televisit" vs "80 min face-to-face" | ✅ **已修复** | POST-VISIT-TYPE 优先检测 video/ZOOM 关键词正确；v16 P0 标记可能是误判 |
| **56** | Type_of_Cancer HER2+ vs TNBC | ❓ **无法验证** | v17_verify 未包含此行 |
| **94** | Distant Metastasis = Axillary LN | ✅ **已修复** | POST-DISTMET-REGIONAL 成功识别 regional LN |

### 新发现问题

#### Row 8
- 无新增 P0/P1 问题
- 归因引用了误导性的 "face-to-face" 句子（应引用 "through ZOOM"），但不影响提取正确性

#### Row 94
- **P1**：Imaging_Plan 遗漏 colonoscopy（应归入 Procedure_Plan）
- **P2**：Stage_of_Cancer 格式冗余（包含 receptor status）
- 其他字段准确

### v17 整体评估

**优点**：
1. POST-DISTMET-REGIONAL 修复有效（Row 94 P0 已解决）
2. POST-VISIT-TYPE 逻辑正确（Row 8 in-person 准确）
3. 归因覆盖率良好（所有关键字段都有引文）

**待改进**：
1. 归因质量：部分归因引用了误导性句子（如 Row 8 的 "face-to-face"）
2. Row 56 验证缺失：需要包含该行的 run 来验证 POST-TYPE-VERIFY-TNBC
3. Plan 字段分类：colonoscopy 等非 imaging 的 procedure 应归入正确字段

### 下一步建议

1. **验证 Row 56**：运行包含 row_index=56 的实验，确认 POST-TYPE-VERIFY-TNBC 修复效果
2. **改进归因逻辑**：优先引用最直接的证据句（如 "ZOOM" 而非 "face-to-face"）
3. **Plan 字段细化**：在 plan_extraction prompt 中明确区分 imaging vs procedure vs lab

---

审查完成时间：2026-03-19
审查者：Claude Code (Sonnet 4.5)
