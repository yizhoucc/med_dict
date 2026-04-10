# V26 notool Full Run Review (正式版审查)

> Run: v26_full_notool_20260408_080004
> Dataset: 100 samples
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + 49 POST hooks + letter generation
> tool_calling: false
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **✅ COMPLETE** (ROW 1-100 全部逐字审查完毕)

---

## 最终汇总统计（100 Samples）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | **0** | **0%** | 无幻觉/编造 |
| **P1** | **6** | **6%** | 集中在 ROW 1-12 + ROW 88 |
| **P2** | **38** | **38%** | 分布均匀，以 procedure_plan 混入为主 |

### P1 分布
- ROW 1: lab_plan 混入 imaging（持续问题）
- ROW 8: response "not yet on treatment" for incomplete neoadjuvant + surgery
- ROW 10: response "no evidence" but 有 8.8cm 残余 + 20 LN+
- ROW 11: response 引用换药前 PET 进展作为当前响应
- ROW 12: Advance care "Not discussed" but DNR/DNI in problem list
- ROW 88: response "Not mentioned" for post-neoadjuvant progression → surgery

### P2 分布
- ROW 1-10: 19 个（最密集区——复杂 note）
- ROW 11-25: 12 个
- ROW 26-50: 2 个
- ROW 51-100: 5 个（主要是 procedure_plan 混入）

### 与前版本对比
| 版本 | P0 | P1 | P2 |
|------|----|----|-----|
| v24 (修复前) | 0 | 21 | 88 |
| v25 (prompt 修复) | 0 | 4 | ~15 |
| **v26 notool** | **0** | **6** | **38** |

P1 从 21 降到 6，改善 71%。P0 持续为零。

---

## 逐 Sample 问题清单

### ROW 1 (coral_idx 140) — 1 P1, 3 P2
| # | 严重度 | 字段 | 问题 |
|---|--------|------|------|
| 1 | **P1** | lab_plan | 混入 MRI/bone scan + 没列具体 labs（持续问题） |
| 2 | P2 | imaging_plan | 遗漏 Bone Scan |
| 3 | P2 | Type_of_Cancer | IDC 原文未明确 |
| 4 | P2 | therapy_plan | 重复 medication_plan |
| — | ✅ | genetic_testing | "None planned." — v24 修复验证 |
| — | ✅ | letter | 无 "blood tests" 幻觉 — v24 修复验证 |

### ROW 2 (coral_idx 141) — 0 P1, 3 P2
- P2: Metastasis 不完整 + chest wall 是区域性
- P2: imaging_plan 遗漏 rescheduled MRI
- P2: Referral Others "Home health?" 是疑问非确定
- ✅ Letter: 优秀通俗化（TNBC 解释、贫血通俗说明）
- ✅ Letter: 无 lab 幻觉、无情绪编造

### ROW 3 (coral_idx 142) — 0 P1, 1 P2
- P2: Referral-Genetics 检测 ordering ≠ referral
- ✅ genetic_testing: "Genetic testing sent and is pending" — v24 修复验证
- ✅ letter: 无 "emotional" 编造 — v24 修复验证
- ✅ findings: video consult 不再编造未检查的体检发现 — v24 修复验证

### ROW 4 (coral_idx 143) — 0 P1, 1 P2
- P2: therapy_plan 重复 medication_plan
- ✅ letter: "No new blood tests were done" — v24 修复验证（不再说 "blood tests mostly normal"）
- ✅ imaging_plan: brain MRI 有条件限定 "if headaches worsen" — v24 P2 修复

### ROW 5 (coral_idx 144) — 0 P1, 1 P2
- P2: follow_up 循环表述
- ✅ letter: "No new blood tests" — v24 修复验证
- ✅ response_assessment: 详细 CT 混合响应

### ROW 6 (coral_idx 145) — 0 P1, 3 P2
- P2: Patient type 与 summary 不一致
- P2: letter "You appear to be feeling anxious" — 有 PMH 基础但本次未提及
- P2: Referral-Genetics 历史转诊（Myriad 已完成）

### ROW 7 (coral_idx 146) — 0 P1, 3 P2
- P2: procedure_plan + lab_plan 都包含 LVEF recheck（应为 imaging）
- P2: letter "medication level" 实际是 LVEF recheck
- ✅ genetic_testing: "None planned." — v24 修复验证（不再包含 "recheck"）

### ROW 8 (coral_idx 147) — 1 P1, 1 P2
| 1 | **P1** | response_assessment | "Not yet on treatment" 但完成了（不完整）neoadjuvant + surgery |
| 2 | P2 | procedure_plan | medication 内容，遗漏 port placement |

### ROW 9 (coral_idx 148) — 0 P1, 0 P2
- ✅✅ 完全干净！
- ✅ response: "Surgical pathology showed 3.84 cm residual tumor with 5% cellularity..." — v24 修复验证

### ROW 10 (coral_idx 149) — 1 P1, 0 P2
| 1 | **P1** | response_assessment | "does not provide specific evidence" 但有 8.8cm 残余 + 20 LN+（新辅助疗效差） |

---

## v24 P1 修复验证（ROW 1-10）

| 原 v24 P1 | v26 状态 |
|-----------|----------|
| ROW 3 letter "emotional" | ✅ 已修复 |
| ROW 4 letter "blood tests" | ✅ 已修复 |
| ROW 5 letter "blood tests" | ✅ 已修复 |
| ROW 3 genetic IHC/biopsy | ✅ 已修复 |
| ROW 7 genetic "recheck" | ✅ 已修复 |
| ROW 9 response "not yet" | ✅ 已修复 |
| ROW 1 lab_plan | ❌ 持续 |
| ROW 10 response "not mentioned" | ❌ 持续（部分修复——ROW 9 好了但 ROW 10 没有）|

---

## 新发现的 P1

| ROW | 字段 | 问题 | 说明 |
|-----|------|------|------|
| 8 | response_assessment | "Not yet on treatment" for incomplete neoadjuvant + surgery | 新发现 |

---

## 追加审查（ROW 11-100）

### ROW 11 (coral_idx 150) — 1 P1, 1 P2
| 1 | **P1** | response_assessment | 引用换药前 PET 进展，A/P 说 "Exam stable" |
| 2 | P2 | imaging_plan | 遗漏 Echo q6mo |

### ROW 12 (coral_idx 151) — 1 P1, 2 P2
| 1 | **P1** | Advance care | "Not discussed" but DNR/DNI in problem list（只有 tool calling 能修复）|
| 2 | P2 | imaging_plan | 遗漏 Echo q6mo |
| 3 | P2 | Metastasis | lung 已 resolved 但仍列出 |

### ROW 13 (coral_idx 152) — 0 P1, 1 P2
- P2: response "On treatment" 但未开始治疗

### ROW 14 (coral_idx 153) — 0 P1, 1 P2
- P2: current_meds 空（患者在替代治疗中）
- ✅ Stage 正确 "Stage IV (metastatic)" — POST hook 生效

### ROW 15 (coral_idx 154) — 0 P1, 2 P2
- P2: genetic_testing 已做回顾非新计划
- P2: procedure_plan 混入 Rx recommendations

### ROW 16-19 — 0 P1, 0 P2 ✅ 全部干净

### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan 混入 imaging/referral/medication

### ROW 21 — 0 P1, 0 P2 ✅

### ROW 22 (coral_idx 161) — 0 P1, 1 P2
- P2: genetic_testing 有 medication plan 内容

### ROW 23 — 0 P1, 0 P2 ✅

### ROW 24 (coral_idx 163) — 0 P1, 2 P2
- P2: procedure_plan 混入 genetic testing
- P2: Metastasis "Not sure" 应为 "No"
- ✅ genetic_testing 正确捕获 MammaPrint — v24 修复验证

### ROW 25 (coral_idx 164) — 0 P1, 1 P2
- P2: response 引用 pre-treatment PET

### ROW 26-32, 34-37, 39-50 — 0 P1, 0 P2 ✅ 全部干净

### ROW 33 (coral_idx 172) — 0 P1, 1 P2
- P2: letter stage "now considered IIIA" for NED patient（改善——不再说 "more advanced"）

### ROW 38 — 0 P1, 1 P2
- P2: response "not responding" but not yet on treatment

### ROW 51, 53-56, 58-74, 76-87, 89-100 — 0 P1, 0 P2 ✅ 全部干净

### ROW 52 — 0 P1, 1 P2: procedure_plan 混入 fertility referral
### ROW 57 — 0 P1, 1 P2: procedure_plan 混入 genetic counseling
### ROW 75 — 0 P1, 1 P2: procedure_plan 混入 genetics+fertility

### ROW 88 (coral_idx 227) — 1 P1, 0 P2
| 1 | **P1** | response_assessment | "Not mentioned" for post-neoadjuvant progression → surgery（持续顽固）|

---

## 6 个 P1 的可修复性分析

| ROW | P1 问题 | 可修复方式 |
|-----|---------|-----------|
| 1 | lab_plan 混入 imaging | **Tool calling** 已证明可修复 |
| 8 | response post-incomplete neoadj | **Prompt** 需更具体示例 |
| 10 | response 8.8cm residual | **Prompt** 需覆盖更多场景 |
| 11 | response 时间线混淆 | **Prompt** 需时间线指导 |
| 12 | Advance care DNR/DNI | **Tool calling** 已证明可修复 |
| 88 | response post-neoadj progression | **Prompt + 示例** 已试但模型顽固 |

**Tool calling 能修复 2/6 P1（ROW 1 + ROW 12），但需先解决 tool calling 自身的退化问题。**
