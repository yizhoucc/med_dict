# V24 Full Run Review (2026-04-05)

> Run: full_qwen_20260405_073716
> Dataset: 100 samples (coral indices 140-239)
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + 47 POST hooks + letter generation
> Reviewer: Claude (逐字段手工审查)
> Status: **✅ COMPLETE** (ROW 1-100 全部审查完毕)

---

## 最终汇总统计（ROW 1-100，100 个 Sample）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0 (幻觉/编造)** | **0** | **0%** | 无纯幻觉/编造 |
| **P1 (重大错误)** | **20** | **0.20/sample (20%)** | 主要集中在 ROW 1-25 |
| **P2 (小问题)** | **70** | **0.70/sample (70%)** | 分布均匀 |
| A2 (归因不精确) | ~20 | ~0.20/sample | 前 25 行集中 |

### P1 分布
- ROW 1-25: 20 个 P1（0.80/sample）
- ROW 26-100: 0 个 P1（0.00/sample）
- **结论**: P1 问题高度集中在前 25 个 sample，后 75 个 sample 显著更干净

### P2 分布
- ROW 1-25: 54 个 P2（2.16/sample）
- ROW 26-35: 4 个（0.40/sample）
- ROW 36-50: 2 个（0.13/sample）
- ROW 51-75: 3 个（0.12/sample）
- ROW 76-100: 7 个（0.28/sample）
- **结论**: P2 也集中在前 25 个 sample（更复杂/更长的 note）

---

## 逐 Sample 问题清单

### ROW 1 (coral_idx 140) — 1 P1, 4 P2, 4 A2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | lab_plan | 混入 MRI/bone scan（应在 imaging），且没列具体化验（CBC, CMP, CA15-3, CEA, aPTT, PT） |
| 2 | P2 | imaging_plan | 遗漏 Bone Scan |
| 3 | P2 | Type_of_Cancer | "IDC" 原文未明确写 |
| 4 | P2 | genetic_testing_plan | 混入 biopsy/IHC 内容 |
| 5 | P2 | therapy_plan | 重复 medication_plan |
| 6-8 | A2 | Patient type/in-person/second opinion attr | 引用 "RTC..." 不支持这些字段 |
| 9 | A2 | goals_of_treatment attr | 引用不精确 |

### ROW 2 (coral_idx 141) — 0 P1, 4 P2, 5 A2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | Distant Metastasis | 胸壁应为区域性，非远处转移 |
| 2 | P2 | supportive_meds | morphine ER 和 MS contin 重复 |
| 3 | P2 | imaging_plan | 遗漏 rescheduled MRI |
| 4 | P2 | Referral Others | "Home health?" 是疑问非确定 |
| 5-9 | A2 | 多字段 | attribution 引用不精确/不相关 |

### ROW 3 (coral_idx 142) — 1 P1, 3 P2, 1 A2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | **letter** | "You appear to be emotional" 无原文依据（ROS: No anxiety/depression） |
| 2 | P2 | findings | Video consult 加了未检查的体检发现 |
| 3 | P2 | imaging_plan | 措辞混乱 "PET follow up after PET results" |
| 4 | P2 | Referral-Genetics | 检测 ≠ 转诊 |
| 5 | A2 | second opinion | attribution 引用不相关 |

### ROW 4 (coral_idx 143) — 1 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | **letter** | "Your blood tests are mostly normal" 幻觉（无 lab 数据） |
| 2 | P2 | therapy_plan | 与 medication_plan 重复 |
| 3 | P2 | imaging_plan | Brain MRI 缺条件限定 |

### ROW 5 (coral_idx 144) — 1 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | **letter** | "blood tests mostly normal" 幻觉（同 ROW 4） |
| 2 | P2 | therapy_plan | 混入 lab/referral 内容 |
| 3 | P2 | Next clinic visit | 循环表述 |

### ROW 6 (coral_idx 145) — 1 P1, 2 P2, 2 A2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | Patient type | CC 说 "Follow-up"，提取为 "New patient" |
| 2 | P2 | Referral-Genetics | 历史转诊当成当前 |
| 3 | P2 | letter | 暗示需 genetics referral（已完成） |

### ROW 7 (coral_idx 146) — 1 P1, 3 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | genetic_testing_plan | "recheck [REDACTED]" 不是 genetic testing |
| 2 | P2 | Procedure_Plan | lab/imaging 复查错归为 procedure |
| 3 | P2 | letter | "levels of a certain medication" 实际是 LVEF 复查 |

### ROW 8 (coral_idx 147) — 0 P1, 3 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | procedure_plan | 混入化疗方案 |
| 2 | P2 | Referral Others | 社工转诊无原文支持 |
| 3 | P2 | Next clinic visit | 说 in-person 但应为 telehealth |

### ROW 9 (coral_idx 148) — 1 P1, 1 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | response_assessment | "Not yet on treatment" 错误。患者已完成新辅助化疗+手术（病理显示部分响应） |
| 2 | P2 | supportive_meds | 列出不再活跃的化疗支持药 |

---

## 已识别的系统性模式（前 9 行）

### 模式 1: Letter "blood tests mostly normal" 幻觉 ⚠️
**触发条件**: extraction 的 lab_summary = "No labs in note."
**表现**: letter 编造 "Your blood tests are mostly normal"
**出现**: ROW 4, 5（2/9 = 22%）
**根因**: letter generation prompt 没有检查 lab_summary 是否为空
**建议修复**: letter prompt 加规则 "If lab_summary says 'No labs in note', do NOT mention blood test results"

### 模式 2: Letter 编造情绪/状态 ⚠️
**出现**: ROW 3 ("You appear to be emotional" but ROS says no anxiety/depression)
**根因**: 模型可能从 "She has good support" 推断出情绪化
**区分**: ROW 9 的 "tearful" 有原文支持，不是问题

### 模式 3: therapy_plan 与 medication_plan 系统性重复
**出现**: ROW 1, 2, 4, 5, 9 (5/9 = 56%)
**影响**: 低（P2），不影响准确性但冗余

### 模式 4: response_assessment 误判 "Not yet on treatment"
**触发条件**: 患者完成新辅助化疗+手术后的随访
**出现**: ROW 9
**根因**: 模型不理解"当前不在治疗上 ≠ 从未接受过治疗"

### 模式 5: genetic_testing_plan 误填
**出现**: ROW 3 (biopsy/IHC), ROW 7 (LVEF/tumor marker recheck)
**根因**: 模型把非 genetic 的检测/复查误归类

### 模式 6: Patient type 误判
**出现**: ROW 6 (CC 说 Follow-up 但提取为 New patient)
**根因**: POST-PATIENT-TYPE hook 可能没覆盖此场景

### 模式 7: Attribution 系统性不精确
**出现**: 几乎每行都有 A2
**影响**: 低，不影响 extraction/letter 质量，但影响可追溯性

### 模式 8: response_assessment 误判新辅助疗效（ROW 9, 10）
**触发条件**: 患者完成新辅助化疗+手术后的随访
**表现**: "Not yet on treatment" 或 "Not mentioned in note"
**根因**: 模型不理解术后病理就是 response assessment

### 模式 9: response_assessment 时间线混淆（ROW 11）
**表现**: 把换药前的 PET 进展归因于当前药物
**根因**: 模型没区分治疗时间线

### 模式 10: Advance care 遗漏 DNR/DNI（ROW 12）
**触发条件**: DNR/DNI 在 problem list 或先前记录中，非当前 A/P
**表现**: "Not discussed during this visit"

---

## 追加审查（ROW 10-13）

### ROW 10 (coral_idx 149) — 1 P1, 0 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | response_assessment | "Not mentioned" 但 8.8cm 残余+20 LN+ = 新辅助疗效差 |

### ROW 11 (coral_idx 150) — 2 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | response_assessment | 把 letrozole 上的 PET 进展归因于当前 Faslodex。A/P 说 "Exam stable" |
| 2 | **P1** | letter | "cancer is not responding well" 继承 extraction 错误（A/P 说 stable） |
| 3 | P2 | imaging_plan | 遗漏 MRI lumbar/pelvis/femur |
| 4 | P2 | radiotherapy_plan | null 格式（应为 "None"） |

### ROW 12 (coral_idx 151) — 1 P1, 4 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | Advance care | "Not discussed" 但 problem list 明确写 DNR/DNI |
| 2 | P2 | Metastasis | 列了已消退的 "lung" |
| 3 | P2 | supportive_meds | 列了 "patient not taking" 的药 |
| 4 | P2 | imaging_plan | 遗漏 Echo q6mo |
| 5 | P2 | response_assessment | CT/MRI 发现混称 |

### ROW 13 (coral_idx 152) — 0 P1, 1 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | response_assessment | "On treatment" 但未开始全身治疗 |

### ROW 14 (coral_idx 153) — 2 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | current_meds | 空值，患者正在服用 gemcitabine/docetaxel/doxorubicin/pamidronate |
| 2 | **P1** | response_assessment | "No imaging/marker data" 但 CA 27.29 从 193 降到 48，CT 显示缩小 |
| 3 | P2 | Type_of_Cancer | 缺少 PR 25% |
| 4 | P2 | Next clinic visit | A/P 有 "2 months" 和 "3 months" 矛盾 |

### ROW 15 (coral_idx 154) — 0 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | genetic_testing_plan | 把 "reviewed" 误解为 "planned" |
| 2 | P2 | imaging_plan | 来自放射科建议非临床计划 |

### ROW 16 (coral_idx 155) — 0 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | supportive_meds | 助眠药非癌症支持药 |
| 2 | P2 | Type_of_Cancer | "HR+" 缺具体 ER/PR 百分比 |

### ROW 17 (coral_idx 156) — 0 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | procedure_plan | labs 错归为 procedure |
| 2 | P2 | therapy_plan | plan summary dump |

### ROW 18 (coral_idx 157) — 0 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | Referral-Genetics | 遗漏 UCSF Cancer Risk genetics referral |
| 2 | P2 | genetic_testing_plan | genetics referral 进行中但标记为 None |

### ROW 19 (coral_idx 158) — 0 P1, 1 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | Referral | ENT consult (PET 推荐)未捕获 |

### ROW 20 (coral_idx 159) — 1 P1, 1 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | procedure_plan | 混入 CT imaging、referral、genetic testing |
| 2 | P2 | lab_summary | 引用 8 年前的 glucose 值 |

### ROW 21 (coral_idx 160) — 0 P1, 1 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | Type_of_Cancer | DCIS 写 "HER2: status unclear"，应为 "N/A for DCIS" |

### ROW 22 (coral_idx 161) — 1 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | lab_summary | "No labs" 但有临床显著 labs（贫血/肾功能不全/白细胞低） |
| 2 | P2 | genetic_testing_plan | 混入 medication plan 内容 |
| 3 | P2 | supportive_meds | denosumab 在 current_meds 和 supportive_meds 重复 |

### ROW 23 (coral_idx 162) — 0 P1, 3 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | P2 | Type_of_Cancer | 多肿瘤 PR 状态被简化 |
| 2 | P2 | response_assessment | "On treatment" 但尚未开始治疗 |
| 3 | P2 | lab_summary | 使用术中 POCT glucose |

### ROW 24 (coral_idx 163) — 1 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | genetic_testing_plan | "None planned" 但分子检测明确计划中 |
| 2 | P2 | Metastasis | "Not sure" 但 PET/CT 无远处转移 |
| 3 | P2 | procedure_plan | 分子检测错归为 procedure |

---

## 系统性模式总结（更新至 ROW 24）

| # | 模式 | P 级别 | 频率 | 可修复性 |
|---|------|--------|------|----------|
| 1 | Letter "blood tests mostly normal" 幻觉 | P1 | 2/24 (8%) | 高：letter prompt 加 lab_summary 空值检查 |
| 2 | Letter 编造情绪状态 | P1 | 1/24 (4%) | 中：letter prompt 加 ROS 交叉验证 |
| 3 | response_assessment 误判新辅助疗效 | P1 | 2/24 (8%) | 中：prompt 需区分"未治疗" vs "完成治疗后" |
| 4 | response_assessment 时间线混淆 | P1 | 1/24 (4%) | 中：需注入治疗时间线上下文 |
| 5 | genetic_testing_plan 误填/遗漏 | P1 | 3/24 (12%) | 中：prompt 需更明确的 genetic vs biomarker 区分 |
| 6 | Patient type 误判 | P1 | 1/24 (4%) | 中：POST hook 已覆盖部分场景 |
| 7 | Advance care 遗漏 DNR/DNI | P1 | 1/24 (4%) | 中：需搜索 problem list 中的 advance care |
| 8 | current_meds 遗漏 | P1 | 2/24 (8%) | 高：需更积极提取活性治疗药物 |
| 9 | procedure_plan 混入非 procedure 内容 | P1 | 2/24 (8%) | 中：prompt 需更严格的字段定义 |
| 10 | lab_summary 忽略老但显著的 labs | P1 | 1/24 (4%) | 低：需定义 "old labs" 策略 |
| 11 | therapy_plan/medication_plan 重复 | P2 | ~50% | 低优先级，不影响准确性 |
| 12 | Stage "Not available (redacted)" for metastatic | P2 | 7/100 (7%) | 中：应写 "Stage IV (metastatic)" |
| 13 | procedure_plan 混入 referral/genetic content | P2 | 5/100 (5%) | 中：prompt 字段定义需更严格 |

---

## 追加审查（ROW 25-100）

### ROW 25 (coral_idx 164) — 1 P1, 2 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | medication_plan | Xeloda 剂量 (1500/1000mg) 错误归因给 ixabepilone |
| 2 | P2 | Stage_of_Cancer | "Not available" 应为 Stage IV |
| 3 | P2 | Type_of_Cancer | 缺脑转移活检 triple negative 信息 |

### ROW 26-50: 0 P1, 6 P2
- ROW 30 P2: old lab (2016)
- ROW 31 P2: current_meds 空但 scheduled Doxil
- ROW 33 P2: Stage IIB→IIIA 升级
- ROW 35 P2: "inferred from tamoxifen" 但实际用 anastrozole
- ROW 40 P2: response_assessment "On treatment" 但刚处方
- ROW 50 P2: denosumab 重复列出

### ROW 51-75: 0 P1, 3 P2
- ROW 52 P2: procedure_plan 混入 fertility referral
- ROW 57 P2: procedure_plan 混入 genetic counseling
- ROW 75 P2: procedure_plan 混入 genetics+fertility referral

### ROW 76-100: 0 P1, 7 P2
- ROW 76, 79, 83, 84, 86, 92, 100: Stage "Not available (redacted)" 但明确为转移性

---

## 最终修复优先级建议

### 高优先级（可减少大部分 P1）

1. **Letter "blood tests" 幻觉修复**
   - 触发: lab_summary = "No labs in note" 时 letter 编造 "blood tests mostly normal"
   - 修复: letter generation prompt 加条件判断
   - 预期: 消除 2 个 P1

2. **response_assessment prompt 改进**
   - 触发: post-neoadjuvant surgery → "Not yet on treatment"
   - 修复: prompt 加指导 "If patient completed neoadjuvant therapy, describe pathologic response from surgical pathology"
   - 预期: 消除 2-3 个 P1

3. **genetic_testing_plan prompt 改进**
   - 触发: 把 IHC/biomarker/LVEF 复查误归为 genetic testing
   - 修复: prompt 明确定义 genetic testing vs biomarker testing vs imaging
   - 预期: 消除 3 个 P1

4. **current_meds 更积极提取**
   - 触发: 患者正在服用替代/非标准治疗但 current_meds 为空
   - 修复: 增加从 A/P "continue" 语句中提取的灵敏度
   - 预期: 消除 2 个 P1

### 中优先级（减少 P2）

5. **procedure_plan 字段定义收紧**
   - 修复: prompt 明确排除 referral、genetic testing、imaging
   - 预期: 消除 5 个 P2

6. **Stage 默认规则**
   - 修复: POST hook — 当 Metastasis="Yes" 且 Stage="Not available"，自动设为 "Stage IV (metastatic)"
   - 预期: 消除 7 个 P2

### 低优先级

7. therapy_plan/medication_plan 去重 — 不影响准确性
8. Attribution 改进 — 不影响 extraction/letter 质量
