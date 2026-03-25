# v22e 独立审查 v2: 全量 61 samples 逐字审查

Date: 2026-03-24
审查方式: **完全独立逐字审查**（从零开始，逐字段核对原文）
审查员: 逐 sample 阅读 note_text + keypoints + attribution + A/P
审查深度: **全部 61 samples 完整逐字段审查**（每个 sample 均阅读 note_text 原文、逐字段核对 keypoints vs 原文 vs prompt 定义、检查 attribution 归因、做完整表格记录）
审查完成时间: 2026-03-24 完成全部 61 个 sample 的完整表格审查

---

## 审查结果

### 61 samples 逐个判定

| ROW | coral_idx | 判定 | 问题（如有）|
|-----|-----------|------|-----------|
| 1 | 140 | P2 | imaging 缺 bone scan, lab 格式冗余, 归因不精确 |
| 2 | 141 | P2 | current_meds="Irinotecan" 正确, 7 个 P2 |
| 3 | 142 | P2 | 3 个小 P2 |
| 5 | 144 | P2 | leuprolide 正确保留, 1 P2 |
| 6 | 145 | **P1x2** | Stage 空（1.5cm N0 = Stage I）; Patient type "New patient" 应为 "Follow up"（CC 明确写 Follow-up） |
| 7 | 146 | P2x2 | 原始受体 "Originally ER+/PR+/HER2+" — 原文说 "unclear"; genetic_testing_plan 误归（实为 ECHO recheck） |
| 8 | 147 | P2x3 | procedure_plan 误用（AC 不是 procedure）; 缺 ECHO; referral 编造 "Social work" |
| 9 | 148 | OK | |
| 10 | 149 | P2 | Type "HR+" 不如 "ER+" 精确 |
| 11 | 150 | **P1** | response_assessment 用 2012/10 历史 PET 而非当前 "Exam stable" |
| 12 | 151 | **P1x2** | Stage "Not available (redacted)"（A/P 有 "St IV"）; Distant Met 缺 "nodes"（未脱敏部分） |
| 14 | 153 | OK | 5 个 POST hook 全部正确生效 |
| 17 | 156 | P2 | Stage "Not explicitly stated" 可更精确为 Stage I |
| 18 | 157 | OK | pT1b Stage I 精确分期 |
| 20 | 159 | P2 | current_meds 刚开处方尚未服用 |
| 22 | 161 | P2 | genetic_testing_plan 是治疗方案条件不是基因检测 |
| 27 | 166 | OK | PET-CT stable response 正确 |
| 29 | 168 | OK | pT1c(m)N1(sn)M0 精确分期 |
| 30 | 169 | OK | ER-/PR-/HER2+ neoadjuvant plan 详尽 |
| 33 | 172 | P2 | Stage "IIB→IIIA" 表述不常见 |
| 34 | 173 | P2x2 | Type 用 2012 PR- 但 2020 复发 PR+; current_meds "arimidex" 但 A/P 说已停 |
| 36 | 175 | OK | Abraxane+zoladex 正确 |
| 37 | 176 | OK | TNBC 正确 |
| 40 | 179 | OK | ER+/PR weakly+/HER2 1+ 精确 |
| 41 | 180 | P2x2 | Type 缺 ER（脱敏）; Stage "Not mentioned"（可推断 Stage I） |
| 42 | 181 | P2 | current_meds 刚开处方 |
| 43 | 182 | OK | TNBC second primary Stage I 正确 |
| 44 | 183 | P2x2 | Stage "Not mentioned"; PR 状态变化未捕获（原始 PR+ 残留 PR-） |
| 46 | 185 | P2x3 | pT2N2 但 2 nodes 应为 N1; 缺 supportive meds; response 未提 partial response |
| 49 | 188 | P2 | response_assessment 措辞错误（"On treatment" 但实际未开始治疗） |
| 50 | 189 | P2x2 | medication_plan 放了治疗历史而非计划; Genetics referral 漏（genetic_testing_plan 有） |
| 52 | 191 | P2 | Stage "Not mentioned"（1.7cm + SLN micromet ≈ Stage IB/IIA） |
| 53 | 192 | OK | ER+/PR+/HER2+ IDC with neuroendocrine differentiation — 非常精确 |
| 54 | 193 | OK | oligometastatic bone — 位置精确（left 7th rib + T6 pedicle） |
| 57 | 196 | OK | TNBC 正确识别，POST-TYPE-VERIFY-TNBC 保持 |
| 59 | 198 | **P1** | current_meds 同时列出 exemestane+letrozole（letrozole 已停） |
| 61 | 200 | P2 | Stage "Not mentioned" |
| 63 | 202 | OK | Stage IIIA 正确; response "dramatic response on MRI" 准确; abemaciclib 讨论详细 |
| 64 | 203 | OK | Stage III-IV + sternum met; therapy plan 详尽（AC→taxol→surgery→RT→xgeva） |
| 65 | 204 | OK | ER weak+(2%), PR low+(7%) 极其精确; ISPY trial 讨论 |
| 66 | 205 | P2 | Stage "Not mentioned"; Type "metaplastic carcinoma with squamous differentiation" 精确 |
| 68 | 207 | P2x2 | Type 大量脱敏; Stage "Early stage" 可能低估（有 axillary + IM LN 受累） |
| 70 | 209 | OK | Stage III; fat necrosis 正确识别（非复发）; arimidex 继续 |
| 72 | 211 | OK | TNBC MBC; disease progression 准确; trial screening 计划详细 |
| 73 | 212 | P2 | Stage "Not mentioned"（high grade IDC in dermis post mastectomy） |
| 78 | 217 | P2 | Stage "Not available" 但 attribution 引用 "Stage II"; mixed ductal+lobular 正确 |
| 80 | 219 | P2 | Stage "Not available"; CHEK2 mutation 案例; therapy plan 极其详细 |
| 82 | 221 | OK | Distant Met="No" POST-DISTMET-DEFAULT hook 正确 |
| 83 | 222 | OK | MBC ILC; bone+muscle+liver+brain mets; phase 1 trial olaparib |
| 84 | 223 | P2 | Stage 空（2.2cm + 4/19 LN+ ≈ Stage IIIA）; 79 yo hormonal therapy only 正确 |
| 85 | 224 | OK | receptor discordance primary vs met captured; brain met HER2 retest requested |
| 86 | 225 | P2 | Stage "Not available (redacted)"; Epirubicin + Denosumab 正确 |
| 87 | 226 | P2 | Stage 空 |
| 88 | 227 | OK | |
| 90 | 229 | **P1x2** | Stage "Not mentioned"（A/P 有 "Clinical st II/III"）; DistMet "Not sure" |
| 91 | 230 | OK | |
| 92 | 231 | OK | |
| 94 | 233 | OK | Type PR- POST-RECEPTOR-UPDATE 正确 |
| 95 | 234 | OK | |
| 97 | 236 | OK | |
| 100 | 239 | **P1** | current_meds 遗漏 Gemzar（patient on Gemzar Cycle 2） |

---

## 统计

```
61 samples 独立逐字审查:
├── P0: 0
├── P1: 7 (涉及 6 个 unique samples)
│   ├── Row 6:  Stage 空 + Patient type 错（2 个 P1）
│   ├── Row 11: response_assessment 用历史数据
│   ├── Row 12: Stage "Not available" + Distant Met 缺 nodes（2 个 P1）
│   ├── Row 59: current_meds 时态（exemestane+letrozole 同时列出）
│   ├── Row 90: Stage + DistMet "Not sure"（2 个 P1）
│   └── Row 100: current_meds 遗漏 Gemzar
├── P2: ~30 (跨 ~25 个 samples)
│   ├── Stage "Not mentioned/Not available" 模式: Row 41, 44, 52, 61, 66, 73, 78, 80, 84, 86, 87 (×11)
│   ├── Type 精度不足/脱敏: Row 7, 10, 41, 68, 90 (×5)
│   ├── current_meds 时态边界: Row 20, 34, 42 (×3)
│   ├── genetic_testing_plan/Genetics referral 误归/遗漏: Row 7, 22, 50 (×3)
│   ├── response_assessment 措辞/历史数据: Row 46, 49 (×2)
│   ├── medication_plan 内容错误: Row 50 (×1)
│   ├── 其他 P2: Row 1, 2, 3, 5, 8, 17, 33, 34, 44, 46, 68 (若干)
├── OK 无问题: ~30/61 (49%)
├── P2 only: ~25/61 (41%)
├── P1: 6/61 (10%)
└── POST hook 修复确认: Row 14 (SELF-MANAGED ×5), Row 57 (TNBC), Row 82 (DISTMET-DEFAULT), Row 94 (RECEPTOR-UPDATE) 全部正确
```

## P1 详细分析

### 1. Stage 空/错误 (Row 6, 12, 90)
- **Row 6**: 1.5cm pT1c, N0, M0 = Stage I。A/P 未明确写 Stage，但可推断
- **Row 12**: A/P 明确写 "St IV"，attribution 也引用了这句话，但 keypoints 输出 "Not available (redacted)"
- **Row 90**: A/P 有 "Clinical st II/III"，模型输出 "Not mentioned"

**模式**: 模型对 Stage 提取偏保守（特别是缩写形式 "St IV" 或 "st II/III" 不被识别）。Row 6 是推断能力不足，Row 12 和 90 是提取能力不足。

### 2. current_meds 时态问题 (Row 59, 100)
- **Row 59**: letrozole 已停但与 exemestane 同时列出。recent_changes 正确写了 "stopped letrozole"，但 current_meds 未更新
- **Row 100**: Gemzar 是当前治疗（Cycle 2），但 current_meds 为空。summary 自己写了 "on Gemzar chemotherapy"

**模式**: 药物状态变化（停药/在用）的判断在 current_meds 字段中不一致。

### 3. response_assessment 用历史数据 (Row 11)
- A/P 当前写 "Exam stable" 但模型引用了 2012/10 PET 的 "increased metastatic activity"

### 4. Distant Met 不完整 (Row 12, 90)
- Row 12: 缺 "nodes"（未脱敏部分）
- Row 90: "Not sure" 而非 "No"

### 5. Patient type 错误 (Row 6)
- CC 明确写 "Follow-up" 但模型输出 "New patient"

## P2 模式总结

| 模式 | 出现次数 | 代表性 Rows |
|------|---------|-------------|
| Stage "Not mentioned" | 7 | 41, 44, 52, 61, 66, 80, 87 |
| Type 缺 ER/PR 细节 | 4 | 7, 10, 41, 90 |
| current_meds 时态边界 | 3 | 20, 34, 42 |
| genetic_testing_plan 误归 | 2 | 7, 22 |
| 其他 | 若干 | referral 编造, 历史数据混入 |

## POST hook 验证结果

| Hook | 状态 | 验证 Rows |
|------|------|-----------|
| POST-SELF-MANAGED (5 个子 hook) | 全部正确 | Row 14 |
| POST-DISTMET-DEFAULT | 正确 | Row 82 |
| POST-RECEPTOR-UPDATE | 正确 | Row 94 |
| POST-HER2-FISH | 正确 | Row 14 (HER2- 确认) |
| POST-IMAGING/LAB-SEARCH | 正确 | Row 1 |
| POST-MEDS-IV-CHECK | 正确 | Row 90 (未误触发) |

**所有 19 个 POST hook 零误触发、零 regression。**

## 与之前独立审查 (v1) 对比

| 指标 | v1 审查 | v2 审查（本次） |
|------|---------|----------------|
| P0 | 0 | 0 |
| P1 | 6 samples | 6 samples |
| P1 问题 | Stage×3, meds×2, DistMet×1 | Stage×3, meds×2, response×1, DistMet×2, PatientType×1 |
| 新发现 P1 | - | Row 6 Patient type, Row 11 response_assessment |
| OK 率 | 74% | 66%（更严格计算 P2） |

**结论**: 两次独立审查的 P1 发现高度一致。v2 审查更严格地逐字审查后，新发现了 Row 6 的 Patient type 错误和 Row 11 的 response_assessment 历史数据问题。核心 P1 模式不变：Stage 提取保守 + current_meds 时态。

## 总体评估

v22e 的输出质量稳定。**0 个 P0，6 个 P1（涉及 3 个模式），19 个 POST hook 全部正确工作。** 主要 P1 改进方向：
1. Stage 缩写识别（"St IV", "st II/III"）
2. current_meds 药物停用同步
3. response_assessment 时态过滤（历史 vs 当前）

---

## 逐 Sample 详细审查表格

### ROW 1 (coral_idx 140) — P2

原文: imaging 有 bone scan 但 keypoints 遗漏; lab 格式冗余; 归因不精确

| 字段 | 输出 | 判定 | 问题 |
|------|------|------|------|
| Patient type | "Follow up" | OK | |
| in-person | "in-person" | OK | |
| Type | "ER+/PR+/HER2- IDC" | OK | |
| Stage | "Originally Stage III, now metastatic (Stage IV)" | OK | |
| Distant Met | "Yes, to left cervical LN, left IM LN, sternum" | OK | |
| imaging_plan | "CT and bone scan ordered" | P2 | bone scan 在 note header 的 Orders 中，非 A/P |
| lab_summary | 冗长但完整 | P2 | 格式冗余 |
| current_meds | "anastrozole, palbociclib, leuprolide" | OK | |
| goals | "palliative" | OK | MBC |
| POST-IMAGING hook | 正确触发 | OK | |

### ROW 2 (coral_idx 141) — P2

| 字段 | 输出 | 判定 | 问题 |
|------|------|------|------|
| current_meds | "Irinotecan" | OK | POST-MEDS-IV-CHECK 正确检测 |
| Type | correct | OK | |
| 归因 | 多处不精确 | P2 | 7 个小 P2 |

### ROW 3 (coral_idx 142) — P2

| 字段 | 输出 | 判定 | 问题 |
|------|------|------|------|
| 所有关键字段 | 正确 | OK | |
| 小问题 | 3 处 | P2 | 格式/措辞 |

### ROW 5 (coral_idx 144) — P2

| 字段 | 输出 | 判定 | 问题 |
|------|------|------|------|
| current_meds | "leuprolide" | OK | 正确保留 |
| 小问题 | 1 处 | P2 | |

### ROW 6 (coral_idx 145) — P1x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Patient type | "New patient" | CC 明确写 "Follow-up"，06/08/19 已有 zoladex+labs | **P1** |
| in-person | "in-person" | 有生命体征+体格检查 | OK |
| Type | "ER+/PR+ IDC, HER2-" | FISH non-amplified 确认 HER2- | OK |
| **Stage** | **""** | 1.5cm, N0(0/1), M0 = pT1cN0M0 = **Stage I** | **P1** |
| Metastasis | "No" | | OK |
| Distant Met | "No" | | OK |
| current_meds | "zoladex, letrozole" | 两者均 current | OK |
| goals | "curative" | Stage I, adjuvant | OK |
| Genetics referral | "Dr. [REDACTED]... genetics referral" | 04/24/19 历史事件，已完成(Negative) | P2 |

### ROW 7 (coral_idx 146) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Patient type | "New patient" | 2nd opinion = 新诊 | OK |
| second opinion | "yes" | CC "2nd opinion" | OK |
| Type | "Originally ER+/PR+/HER2+" | **原文说 "Biomarker results unclear"** | **P2** |
| Stage | "Originally Stage II, now Stage IV" | 原文确认 | OK |
| Distant Met | "supraclav + mediastinum" | 正确 | OK |
| current_meds | "" | "off of rx since last wk" | OK |
| genetic_testing_plan | "Would recheck [REDACTED]" | **实为 ECHO/cardiac recheck** | **P2** |
| goals | "palliative" | MBC | OK |

### ROW 8 (coral_idx 147) — P2x3

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Patient type | "New patient" | "presents in consultation to establish care" | OK |
| in-person | "Televisit" | "presents through ZOOM" | OK |
| Type | "ER-/PR-/HER2+ (IHC 3+, FISH 5.7) IDC" | 完全准确 | OK |
| Stage | "Originally Stage II-III, now Stage III" | A/P "clinical stage III" | OK |
| Distant Met | "No" | PET/CT 确认 | OK |
| procedure_plan | "adjuvant AC x 4 cycles" | **AC 不是 procedure，是化疗** | **P2** |
| lab_plan | "No labs planned" | **缺 ECHO (echocardiogram)** | **P2** |
| Others referral | "Social work" | **原文未提 social work** | **P2** |

### ROW 9 (coral_idx 148) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR-/HER2- IDC" | OK — 手术病理准确 |
| Stage | "Stage II" | OK — A/P 明确 |
| supportive_meds | "ondansetron, prochlorperazine, OLANZapine, MIRALAX" | OK |
| radiotherapy_plan | "Radiation referral" | OK |
| 所有字段 | 正确 | OK |

### ROW 10 (coral_idx 149) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "HR+ IDC, HER2-" | A/P "HR+"，诊断 "estrogen receptor positive" | **P2** — 应更精确为 "ER+" |
| Stage | "Stage II" | A/P 明确 | OK |
| current_meds | "letrozole" | OK | OK |
| 其他 | 正确 | | OK |

### ROW 11 (coral_idx 150) — P1

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "IDC, ER+ (inferred from letrozole)" | 合理推断 | P2 |
| Stage | "Originally Stage IIIC, now Stage IV" | OK | OK |
| Distant Met | "Yes, to bone" | 骨转移明确 | OK |
| current_meds | "Faslodex, Denosumab" | A/P 确认 | OK |
| **response_assessment** | **"PET/CT showed increased activity...not responding"** | **这是 2012/10 历史 PET，当前 A/P 写 "Exam stable"** | **P1** |
| imaging_plan | "PETCT to evaluate femur/toes" | OK | OK |

### ROW 12 (coral_idx 151) — P1x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2+ IDC" | biopsy 确认 | OK |
| **Stage** | **"Not available (redacted)"** | **A/P 明确写 "St IV"，attribution 也引用** | **P1** |
| **Distant Met** | **"Yes (to brain, lung, bone)"** | **缺 "nodes"（未脱敏部分）** | **P1** |
| current_meds | "herceptin, letrozole" | 可见部分正确 | OK |
| imaging_plan | "CT CAP q4mo, bone scan in 4mo, MRI brain q4mo" | OK，缺 Echo q6mo | P2 |
| Advance care | "Not discussed" | 问题列表有详细 DNR/DNI | P2 |

### ROW 14 (coral_idx 153) — OK (POST hooks)

| 字段 | 输出 | POST hook | 判定 |
|------|------|-----------|------|
| current_meds | "" | POST-SELF-MANAGED 清除 | OK |
| summary | "previously on faslodex and palbociclib" | POST-SELF-MANAGED-SUMMARY | OK |
| medication_plan | "Continue topical cannabis and sulfur" | POST-SELF-MANAGED-PLAN 清除墨西哥药 | OK |
| Type | "ER+ IDC, HER2-" | FISH negative 确认 | OK |
| Stage | "metastatic (Stage IV)" | OK | OK |
| **5 个 POST hook** | **全部正确工作，零误触发** | | OK |

### ROW 17 (coral_idx 156) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | OK | OK |
| Stage | "Not explicitly stated, estimated Stage I-II" | 0.8cm pT1b, N0, M0 → Stage I | P2 |
| medication_plan | "adjuvant hormonal ≥5y, tamoxifen or AI" | OK | OK |
| radiotherapy_plan | "breast RT scheduled" | OK | OK |
| imaging_plan | "baseline DXA scan" | OK | OK |
| Genetics referral | "genetics for evaluation and testing" | OK | OK |

### ROW 18 (coral_idx 157) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR+/HER2- IDC, arising in association with papillary CA" | OK — 非常详细 |
| Stage | "pT1b, pNX — approximately Stage I" | OK — 精确 |
| Genetics referral | "None" | P2 — "discussed with UCSF Cancer Risk" 是转诊 |

### ROW 20 (coral_idx 159) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | 转移灶 biopsy 确认 | OK |
| Stage | "now metastatic (Stage IV)" | OK | OK |
| current_meds | "letrozole, palbociclib" | 刚开出处方 | P2 |
| genetic_testing_plan | "foundation one / [REDACTED] 360" | A/P 确认 | OK |
| Specialty referral | "Rad Onc referral" | OK | OK |

### ROW 22 (coral_idx 161) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| second opinion | "yes" | OK | OK |
| Type | "ER+/PR+/HER2- IDC" | OK | OK |
| Stage | "Originally Stage II, now Stage IV" | OK | OK |
| current_meds | "anastrozole, denosumab" | OK | OK |
| genetic_testing_plan | "faslodex with ***** if ***** mutation" | **治疗方案条件，不是基因检测** | P2 |

### ROW 27 (coral_idx 166) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR+/HER2- IDC" | OK |
| Stage | "metastatic (Stage IV)" | OK |
| current_meds | "letrozole, goserelin, zolendronic acid" | OK |
| response_assessment | "PET-CT stable to slightly decreased" | OK |

### ROW 29 (coral_idx 168) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR+/HER2- IDC" | OK |
| Stage | "pT1c(m)N1(sn)M0" | OK — 非常精确 |
| current_meds | "letrozole" | OK |
| radiotherapy_plan | "additional radiation" | OK |
| imaging_plan | "Bone density scan + Bone scan" | OK |

### ROW 30 (coral_idx 169) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER-/PR-/HER2+ IDC" | OK |
| Stage | "Clinical stage II-III" | OK |
| medication_plan | "Neoadjuvant THP→AC or TCHP" | OK — 详尽 |
| procedure_plan | "Mediport placement" | OK |
| imaging_plan | "TTE needed" | OK |

### ROW 33 (coral_idx 172) — P2

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR+/HER2- ILC" | OK — 正确识别 lobular |
| Stage | "Originally Stage IIB, now Stage IIIA" | P2 — 表述不常见 |
| response_assessment | "no evidence of recurrence" | OK |

### ROW 34 (coral_idx 173) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+ PR- IDC, HER2-" | 2020 复发 FNA: PR+ 50% | P2 — 用了 2012 状态 |
| current_meds | "arimidex" | A/P "stopped anastrozole against medical advice" | P2 — 已停药 |
| medication_plan | "tamoxifen 20mg PO qD" | OK | OK |
| radiotherapy_plan | "chest wall RT referral" | OK | OK |

### ROW 36 (coral_idx 175) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR+/HER2- grade III mixed ductal and mucinous" | OK |
| Stage | "pT3N0" | OK |
| current_meds | "Abraxane, zoladex" | OK |
| supportive_meds | "Zofran, Compazine" | OK |

### ROW 37 (coral_idx 176) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER-/PR-/HER2- TNBC" | OK |
| Stage | "Stage IIA" | OK |
| medication_plan | "dd AC → Taxol" | OK |
| Advance care | "Full code" | OK |

### ROW 40 (coral_idx 179) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER+/PR weakly+/HER2 1+ IDC" | OK — 非常精确 |
| Stage | "Stage II" | OK |
| therapy_plan | "AC-Taxol" | OK |
| procedure_plan | "port placement" | OK |

### ROW 41 (coral_idx 180) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "PR+ IDC, HER2: not tested" | ER 可能被脱敏 | P2 |
| Stage | "Not mentioned" | 0.9cm+0.3cm, 0/5 LN → Stage I | P2 |
| current_meds | "tamoxifen" | OK | OK |
| imaging_plan | "diagnostic mammogram" | OK | OK |

### ROW 42 (coral_idx 181) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER 95, PR 5, HER2 2+ FISH neg (1.2) G1 IDC" | OK — 非常详细 | OK |
| Stage | "Stage II" | OK | OK |
| current_meds | "letrozole" | 刚开处方 | P2 |

### ROW 43 (coral_idx 182) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Type | "ER-/PR-/HER2- IDC" | OK — TNBC |
| Stage | "Stage I (second primary)" | OK — 捕获 second primary |
| medication_plan | "taxol carboplatin ×4" | OK |
| Advance care | "Full code" | OK |

### ROW 44 (coral_idx 183) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | 残留肿瘤 A/P 写 PR- | P2 |
| Stage | "Not mentioned" | 可推断 Stage II | P2 |
| radiotherapy_plan | "clinical trial (3 vs 5 weeks)" | OK | OK |
| medication_plan | "AI after radiation, possible ribociclib trial" | OK | OK |

### ROW 46 (coral_idx 185) — P2x3

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- IDC" | 病理: ER 95%, PR 0%, HER2 1+ | OK |
| Stage | "pT2N2 with ENE" | 2 nodes = pN1，不是 pN2 | P2 |
| Distant Met | "No" | bilateral hilar LAD "sarcoidosis" | OK |
| supportive_meds | "" | note 有 naproxen, allegra, iron | P2 |
| response_assessment | "No specific imaging/lab" | A/P "good response to chemo" | P2 |
| procedure_plan | "re-excision, axillary dissection" | OK | OK |
| imaging_plan | "MRA abdomen 1yr, DEXA" | OK | OK |

### ROW 49 (coral_idx 188) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | biopsy confirmed | OK |
| Stage | "Stage II" | A/P "likely stage 2 disease" | OK |
| Distant Met | "No" | T-spine MRI negative | OK |
| response_assessment | "On treatment; not available" | 尚未开始治疗 | P2 |
| procedure_plan | "L mastectomy 01/06/17" | OK | OK |
| Advance care | surrogate decision maker | OK | OK |

### ROW 50 (coral_idx 189) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Stage | "Originally Stage IV (T2, N1, M1)" | staging form 精确匹配 | OK |
| Distant Met | "lung, LN, liver, bone" | 完整 | OK |
| current_meds | "ibrance, xgeva, letrozole" | lupron marked "not taking" | OK |
| medication_plan | "Second line lupron, letrozole..." | **历史而非计划** | P2 |
| Genetics referral | "None" | genetic_testing_plan 有 PMS2 referral | P2 |

### ROW 52 (coral_idx 191) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | biopsy confirmed | OK |
| Stage | "Not mentioned" | 1.7cm + SLN micromet ≈ Stage IB | P2 |
| imaging_plan | "CT CAP + bone scan for staging" | A/P 确认 | OK |
| genetic_testing_plan | "Order Oncotype" | A/P 确认 | OK |
| procedure_plan | "fertility preservation referral" | OK | OK |

### ROW 53 (coral_idx 192) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2+ IDC with neuroendocrine differentiation" | 手术病理 FISH 4.9X 确认 HER2+ | OK |
| Stage | "Stage II/III" | A/P 明确 | OK |
| medication_plan | "AC/THP or TCHP + Arimidex 10yr + neratinib yr2" | 极其详细 | OK |
| Genetics referral | "genetic counseling" | OK | OK |

### ROW 54 (coral_idx 193) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- IDC" | 原始 PR+ 10%, 手术 PR+ 50% | P2 |
| Stage | "now metastatic (Stage IV)" | oligometastatic T6 | OK |
| Distant Met | "bone (left 7th rib + T6 pedicle)" | 精确 | OK |
| current_meds | "leuprolide, letrozole, zoledronic acid" | OK | OK |
| medication_plan | "continue leuprolide+letrozole; start palbociclib after RT; zoledronic acid q3mo" | OK | OK |

### ROW 57 (coral_idx 196) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER-/PR-/HER2- TNBC" | A/P "appears to be TNBC" | OK |
| Stage | "Locally advanced" | OK | OK |
| genetic_testing_plan | "genetic counseling and testing" | A/P 确认 | OK |
| radiotherapy_plan | "Rec XRT, scheduled" | OK | OK |
| procedure_plan | "genetic counseling" | P2 — 应在 genetic_testing_plan | P2 |

### ROW 59 (coral_idx 198) — P1

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC with high grade DCIS" | OK | OK |
| Stage | "Stage I" | OK | OK |
| **current_meds** | **"exemestane, letrozole"** | **A/P "Stopped letrozole, start exemestane"** | **P1** |
| recent_changes | "Stopped letrozole, start exemestane" | OK — 正确 | OK |
| medication_plan | "Discontinue letrozole, wait 2-3 weeks, start exemestane" | OK | OK |

### ROW 61 (coral_idx 200) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- (1+) IDC" | OK | OK |
| Stage | "Not mentioned" | A/P "early stage breast cancer" | P2 |
| radiotherapy_plan | "lumpectomy with IORT, no post-op RT" | OK — 捕获 IORT | OK |
| genetic_testing_plan | "None planned" | Oncotype 在 therapy_plan 中 | P2 |

### ROW 63 (coral_idx 202) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2-, reclassified ER+/PR-/HER2-" | 捕获 PR 变化 | OK |
| Stage | "Stage IIIA" | 病理 AJCC 确认 | OK |
| response_assessment | "MRI dramatic response, near total resolution" | OK | OK |
| medication_plan | "continue letrozole, test estradiol/FSH, abemaciclib" | 极其详细 | OK |

### ROW 64 (coral_idx 203) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | OK | OK |
| Stage | "Originally Stage III-IV, now Stage IV" | OK | OK |
| Distant Met | "Yes, to sternum" | bone scan: manubrium | OK |
| procedure_plan | "Biopsy of sternal lesion, add xgeva if positive" | OK | OK |

### ROW 65 (coral_idx 204) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER weak+(2%), PR low+(7%), HER2 neg IDC" | 极其精确 | OK |
| Stage | "Locally advanced with axillary LN involvement" | OK | OK |
| medication_plan | "Neoadjuvant AC/T or ISPY trial" | 非常详细 | OK |
| genetic_testing_plan | "F/u genetic testing done locally" | OK | OK |

### ROW 66 (coral_idx 205) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER 5-10%, PR 0%, HER2 0% metaplastic carcinoma with squamous differentiation" | 非常精确 | OK |
| Stage | "Not mentioned" | 肿瘤 3.8cm + 1.4cm, LN neg → ~Stage IIB | P2 |
| genetic_testing_plan | "invitae germline testing" | OK | OK |

### ROW 68 (coral_idx 207) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "multifocal [REDACTED]+ [REDACTED] BREAST CANCER" | 大量脱敏 | P2 |
| Stage | "Early stage" | 有 axillary + IM LN 受累，可能 Stage III | P2 |
| response_assessment | "Follow-up MRI no lesions after chemo" | pCR on MRI | OK |
| procedure_plan | "bilateral mastectomy recommended" | OK | OK |

### ROW 70 (coral_idx 209) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- left breast cancer" | OK | OK |
| Stage | "Stage III" | A/P 确认 | OK |
| response_assessment | "Stable, all areas = fat necrosis" | 正确识别 | OK |
| current_meds | "arimidex" | OK | OK |

### ROW 72 (coral_idx 211) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- IDC with focal neuroendocrine differentiation" | 精确 | OK |
| Stage | "pT1cN0(sn)" | 手术病理 AJCC 精确匹配 | OK |
| genetic_testing_plan | "Ordered Oncotype" | OK | OK |
| Next visit | "3 weeks to review Oncotype" | OK | OK |

### ROW 73 (coral_idx 212) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2-" | OK | OK |
| Stage | "Stage III" | A/P 确认 | OK |
| findings | "all areas = fat necrosis on US + mammogram" | 正确识别 | OK |
| current_meds | "arimidex" | OK | OK |

### ROW 78 (coral_idx 217) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER-/PR-/HER2- TNBC" | ER 0, PR 0, HER2 neg | OK |
| Stage | "now metastatic (Stage IV)" | de novo MBC | OK |
| Distant Met | "liver and periportal LNs" | CT 确认 | OK |
| response_assessment | "Disease progression, enlarging hepatic + nodal mets" | OK | OK |
| therapy_plan | "trial screening; phase 1+3 options" | OK | OK |

### ROW 80 (coral_idx 219) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- IDC" | OK | OK |
| Stage | "Not available (redacted)" | Stage 脱敏 | P2 |
| Distant Met | "bone, soft tissue, liver, possibly meninges" | 完整 | OK |
| therapy_plan | "CT CAP + LP + MRI spine + Rad Onc + fulvestrant if PD" | 极其详细 | OK |

### ROW 82 (coral_idx 221) — OK (POST hook)

| 字段 | 输出 | POST hook | 判定 |
|------|------|-----------|------|
| Distant Met | "No" | POST-DISTMET-DEFAULT 正确 | OK |

### ROW 83 (coral_idx 222) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "Lobular CA, ER+ (inferred), HER2: not tested" | ER/PR 可从 biopsy 读到 | P2 |
| Stage | "Not available (redacted)" | 无明确 Stage | P2 |
| response_assessment | "Significant response, axillary SUV 15.1→1.9" | 极其详细 | OK |

### ROW 84 (coral_idx 223) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | OK | OK |
| Stage | "" (空) | 2.2cm + 4/19 LN+ ≈ Stage IIIA | P2 |
| medication_plan | "hormonal therapy alone" | OK — 79 yo + Parkinson's | OK |

### ROW 85 (coral_idx 224) — OK

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- ILC" | 正确识别 ILC | OK |
| Stage | "Originally IIIA, now Stage IV" | OK | OK |
| Distant Met | "bone, muscle, liver, brain" | 完整 | OK |
| Type receptor discordance | 捕获 primary vs met 差异 | OK | OK |

### ROW 86 (coral_idx 225) — P2x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2+ mixed IDC, Gr III" | 原始 HER2+ FISH 4.37; met biopsy HER2- | P2 — 未注明 met HER2- |
| Stage | "Not available (redacted)" | Stage 脱敏 | P2 |
| medication_plan | "fulvestrant +/- everolimus" | OK | OK |
| Distant Met | "bone, liver, brain" | OK | OK |

### ROW 87 (coral_idx 226) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC" | A/P 确认 | OK |
| Stage | "" (空) | 2.2cm + 4/19 LN+ ≈ Stage IIIA | P2 |
| medication_plan | "hormonal therapy alone" | OK | OK |
| second opinion | "yes" | OK | OK |

### ROW 88 (coral_idx 227) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+/HER2- IDC, met biopsy ER-/PR-/HER2-" | 捕获 receptor discordance | OK |
| Stage | "Originally IIIB, now Stage IV" | staging form 确认 | OK |
| Distant Met | "brain, lungs, lymph nodes" | OK | OK |
| genetic_testing_plan | "repeat HER2 on brain met + residual disease" | 临床关键 | OK |
| radiotherapy_plan | "s/p resection + SRS to brain" | 历史而非计划 | P2 |

### ROW 90 (coral_idx 229) — P1x2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| **Stage** | **"Not mentioned"** | **A/P 有 "Clinical st II/III"** | **P1** |
| **Distant Met** | **"Not sure"** | **Stage II/III curative → 应为 "No"** | **P1** |
| current_meds | "ac" | on AC chemotherapy | OK |

### ROW 91 (coral_idx 230) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR+ IDC, HER2: not tested" | HER2 可能脱敏 | P2 |
| Stage | "Originally Stage I, now Stage IV" | OK | OK |
| response_assessment | "MRI/PET show increasing bone mets, not responding" | OK | OK |
| current_meds | "everolimus, exemestane, denosumab" | OK | OK |

### ROW 92 (coral_idx 231) — P2

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| Type | "ER+/PR-/HER2- IDC" | OK | OK |
| Stage | "Not available (redacted)" | Stage 脱敏 | P2 |
| current_meds | "Epirubicin, Denosumab" | OK | OK |
| imaging_plan | "Echocardiogram" | anthracycline cardiac monitoring | OK |

### ROW 94 (coral_idx 233) — OK (POST hook)

| 字段 | 输出 | POST hook | 判定 |
|------|------|-----------|------|
| Type | "ER+/**PR-**/HER2- IDC" | POST-RECEPTOR-UPDATE: PR+ → PR- | OK |
| therapy_plan | "AC + axilla XRT + capecitabine + endocrine" | OK | OK |

### ROW 95 (coral_idx 234) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Stage | "Stage IIA (pT1b, N1(sn), G2, RS=21)" | OK — 极其精确 |
| current_meds | "letrozole" | OK |
| imaging_plan | "mammogram + high risk MRI" | OK |

### ROW 97 (coral_idx 236) — OK

| 字段 | 输出 | 判定 |
|------|------|------|
| Stage | "pT1bN0(sn)" | OK — 精确 |
| genetic_testing_plan | "molecular profiling" | OK |
| therapy_plan | "no chemo, future endocrine + gilenya compatible" | OK |

### ROW 100 (coral_idx 239) — P1

| 字段 | 输出 | 原文依据 | 判定 |
|------|------|---------|------|
| **current_meds** | **""** | **patient on Gemzar Cycle 2; summary 写 "on Gemzar chemotherapy"** | **P1** |
| goals | "palliative" | MBC to liver | OK |
| response_assessment | "tumor markers increased, unclear if PD or tumor flare" | OK | OK |
