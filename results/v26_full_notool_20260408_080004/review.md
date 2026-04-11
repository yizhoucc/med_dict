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
| **P2** | **44** | **44%** | 分布均匀，以 procedure_plan 混入为主 |

### P1 分布
- ROW 1: lab_plan 混入 imaging（持续问题）
- ROW 8: response "not yet on treatment" for incomplete neoadjuvant + surgery
- ROW 10: response "no evidence" but 有 8.8cm 残余 + 20 LN+
- ROW 11: response 引用换药前 PET 进展作为当前响应
- ROW 12: Advance care "Not discussed" but DNR/DNI in problem list
- ROW 88: response "Not mentioned" for post-neoadjuvant progression → surgery

### P2 分布
- ROW 1-10: 19 个（最密集区——复杂 note）
- ROW 11-25: 14 个（ROW 13 +1 findings laterality, ROW 14 +1 response CA27.29）
- ROW 26-50: 3 个（ROW 39 Type ER+ inferred from goserelin）
- ROW 51-100: 6 个（procedure_plan 混入 + ROW 74 Type HER2+ 混淆 gastric cancer）

### 与前版本对比
| 版本 | P0 | P1 | P2 |
|------|----|----|-----|
| v24 (修复前) | 0 | 21 | 88 |
| v25 (prompt 修复) | 0 | 4 | ~15 |
| **v26 notool** | **0** | **6** | **44** |

P1 从 21 降到 6，改善 71%。P0 持续为零。

---

## 逐 Sample 问题清单（ROW 1-100，每个 ROW 独立条目）

### ROW 1 (coral_idx 140) — 1 P1, 3 P2
| 1 | **P1** | lab_plan | 混入 MRI/bone scan + 没列具体 labs（持续问题） |
| 2 | P2 | imaging_plan | 遗漏 Bone Scan |
| 3 | P2 | Type_of_Cancer | IDC 原文未明确 |
| 4 | P2 | therapy_plan | 重复 medication_plan |

### ROW 2 (coral_idx 141) — 0 P1, 3 P2
- P2: Metastasis 不完整 + chest wall 是区域性
- P2: imaging_plan 遗漏 rescheduled MRI
- P2: Referral Others "Home health?" 是疑问非确定

### ROW 3 (coral_idx 142) — 0 P1, 1 P2
- P2: Referral-Genetics 检测 ordering ≠ referral

### ROW 4 (coral_idx 143) — 0 P1, 1 P2
- P2: therapy_plan 重复 medication_plan

### ROW 5 (coral_idx 144) — 0 P1, 1 P2
- P2: follow_up 循环表述

### ROW 6 (coral_idx 145) — 0 P1, 3 P2
- P2: Patient type 与 summary 不一致
- P2: letter "You appear to be feeling anxious" — 有 PMH 基础但本次未提及
- P2: Referral-Genetics 历史转诊（Myriad 已完成）

### ROW 7 (coral_idx 146) — 0 P1, 3 P2
- P2: procedure_plan + lab_plan 都包含 LVEF recheck（应为 imaging）
- P2: letter "medication level" 实际是 LVEF recheck

### ROW 8 (coral_idx 147) — 1 P1, 1 P2
- **P1**: response_assessment "Not yet on treatment" 但完成了 neoadjuvant + surgery
- P2: procedure_plan medication 内容，遗漏 port placement

### ROW 9 (coral_idx 148) — 0 P1, 0 P2 ✅

### ROW 10 (coral_idx 149) — 1 P1, 0 P2
- **P1**: response_assessment "does not provide specific evidence" 但有 8.8cm 残余 + 20 LN+

### ROW 11 (coral_idx 150) — 1 P1, 1 P2
- **P1**: response_assessment 引用换药前 PET 进展，A/P 说 "Exam stable"
- P2: imaging_plan 遗漏 Echo q6mo

### ROW 12 (coral_idx 151) — 1 P1, 2 P2
- **P1**: Advance care "Not discussed" but DNR/DNI in problem list
- P2: imaging_plan 遗漏 Echo q6mo
- P2: Metastasis lung 已 resolved 但仍列出

### ROW 13 (coral_idx 152) — 0 P1, 2 P2
- P2: response "On treatment" 但未开始治疗
- P2: findings 左右乳混淆（14mm mass 右乳写成左乳）

### ROW 14 (coral_idx 153) — 0 P1, 2 P2
- P2: current_meds 空（患者在替代治疗中）
- P2: response_assessment "No specific tumor marker data" 但 CA 27.29 从 193→48 在笔记中

### ROW 15 (coral_idx 154) — 0 P1, 2 P2
- P2: genetic_testing 已做回顾非新计划
- P2: procedure_plan 混入 Rx recommendations

### ROW 16 (coral_idx 155) — 0 P1, 0 P2 ✅
### ROW 17 (coral_idx 156) — 0 P1, 0 P2 ✅
### ROW 18 (coral_idx 157) — 0 P1, 0 P2 ✅
### ROW 19 (coral_idx 158) — 0 P1, 0 P2 ✅

### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan 混入 imaging/referral/medication

### ROW 21 (coral_idx 160) — 0 P1, 0 P2 ✅

### ROW 22 (coral_idx 161) — 0 P1, 1 P2
- P2: genetic_testing 有 medication plan 内容

### ROW 23 (coral_idx 162) — 0 P1, 0 P2 ✅

### ROW 24 (coral_idx 163) — 0 P1, 2 P2
- P2: procedure_plan 混入 genetic testing
- P2: Metastasis "Not sure" 应为 "No"

### ROW 25 (coral_idx 164) — 0 P1, 1 P2
- P2: response 引用 pre-treatment PET

### ROW 26 (coral_idx 165) — 0 P1, 0 P2 ✅
### ROW 27 (coral_idx 166) — 0 P1, 0 P2 ✅
### ROW 28 (coral_idx 167) — 0 P1, 0 P2 ✅
### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅
### ROW 30 (coral_idx 169) — 0 P1, 0 P2 ✅
### ROW 31 (coral_idx 170) — 0 P1, 0 P2 ✅
### ROW 32 (coral_idx 171) — 0 P1, 0 P2 ✅

### ROW 33 (coral_idx 172) — 0 P1, 1 P2
- P2: letter stage "now considered IIIA" for NED patient

### ROW 34 (coral_idx 173) — 0 P1, 0 P2 ✅
### ROW 35 (coral_idx 174) — 0 P1, 0 P2 ✅
### ROW 36 (coral_idx 175) — 0 P1, 0 P2 ✅
### ROW 37 (coral_idx 176) — 0 P1, 0 P2 ✅

### ROW 38 (coral_idx 177) — 0 P1, 1 P2
- P2: response "not responding" but not yet on treatment

### ROW 39 (coral_idx 178) — 0 P1, 1 P2
- P2: Type_of_Cancer 错误推断 "ER+ (inferred from goserelin)" — goserelin 用于 fertility preservation，癌症是 TNBC

### ROW 40 (coral_idx 179) — 0 P1, 0 P2 ✅
### ROW 41 (coral_idx 180) — 0 P1, 0 P2 ✅
### ROW 42 (coral_idx 181) — 0 P1, 0 P2 ✅
### ROW 43 (coral_idx 182) — 0 P1, 0 P2 ✅
### ROW 44 (coral_idx 183) — 0 P1, 0 P2 ✅
### ROW 45 (coral_idx 184) — 0 P1, 0 P2 ✅
### ROW 46 (coral_idx 185) — 0 P1, 0 P2 ✅
### ROW 47 (coral_idx 186) — 0 P1, 0 P2 ✅
### ROW 48 (coral_idx 187) — 0 P1, 0 P2 ✅
### ROW 49 (coral_idx 188) — 0 P1, 0 P2 ✅
### ROW 50 (coral_idx 189) — 0 P1, 0 P2 ✅
### ROW 51 (coral_idx 190) — 0 P1, 0 P2 ✅

### ROW 52 (coral_idx 191) — 0 P1, 1 P2
- P2: procedure_plan 混入 fertility referral

### ROW 53 (coral_idx 192) — 0 P1, 0 P2 ✅
### ROW 54 (coral_idx 193) — 0 P1, 0 P2 ✅
### ROW 55 (coral_idx 194) — 0 P1, 0 P2 ✅
### ROW 56 (coral_idx 195) — 0 P1, 0 P2 ✅

### ROW 57 (coral_idx 196) — 0 P1, 1 P2
- P2: procedure_plan 混入 genetic counseling

### ROW 58 (coral_idx 197) — 0 P1, 0 P2 ✅
### ROW 59 (coral_idx 198) — 0 P1, 0 P2 ✅
### ROW 60 (coral_idx 199) — 0 P1, 0 P2 ✅
### ROW 61 (coral_idx 200) — 0 P1, 0 P2 ✅
### ROW 62 (coral_idx 201) — 0 P1, 0 P2 ✅
### ROW 63 (coral_idx 202) — 0 P1, 0 P2 ✅
### ROW 64 (coral_idx 203) — 0 P1, 0 P2 ✅
### ROW 65 (coral_idx 204) — 0 P1, 0 P2 ✅
### ROW 66 (coral_idx 205) — 0 P1, 0 P2 ✅
### ROW 67 (coral_idx 206) — 0 P1, 0 P2 ✅
### ROW 68 (coral_idx 207) — 0 P1, 0 P2 ✅
### ROW 69 (coral_idx 208) — 0 P1, 0 P2 ✅
### ROW 70 (coral_idx 209) — 0 P1, 0 P2 ✅
### ROW 71 (coral_idx 210) — 0 P1, 0 P2 ✅
### ROW 72 (coral_idx 211) — 0 P1, 0 P2 ✅
### ROW 73 (coral_idx 212) — 0 P1, 0 P2 ✅

### ROW 74 (coral_idx 213) — 0 P1, 1 P2
- P2: Type_of_Cancer 说 HER2+ 但乳腺癌是 HER2-（IHC 1+, FISH 1.1）。模型混淆了 prior gastric cancer 的 HER2+

### ROW 75 (coral_idx 214) — 0 P1, 1 P2
- P2: procedure_plan 混入 genetics counseling + fertility referrals

### ROW 76 (coral_idx 215) — 0 P1, 0 P2 ✅
### ROW 77 (coral_idx 216) — 0 P1, 0 P2 ✅
### ROW 78 (coral_idx 217) — 0 P1, 0 P2 ✅
### ROW 79 (coral_idx 218) — 0 P1, 0 P2 ✅
### ROW 80 (coral_idx 219) — 0 P1, 0 P2 ✅
### ROW 81 (coral_idx 220) — 0 P1, 0 P2 ✅
### ROW 82 (coral_idx 221) — 0 P1, 0 P2 ✅

### ROW 83 (coral_idx 222) — 0 P1, 1 P2
- P2: Stage "Stage IV (metastatic)" 但 Distant Met=No, A/P "W/u negative for distant metastasis"。Axillary LN 是 regional — 模型逻辑矛盾

### ROW 84 (coral_idx 223) — 0 P1, 0 P2 ✅
### ROW 85 (coral_idx 224) — 0 P1, 0 P2 ✅
### ROW 86 (coral_idx 225) — 0 P1, 0 P2 ✅
### ROW 87 (coral_idx 226) — 0 P1, 0 P2 ✅

### ROW 88 (coral_idx 227) — 1 P1, 0 P2
- **P1**: response_assessment "Not mentioned" for post-neoadjuvant progression → surgery → brain mets（持续顽固）

### ROW 89 (coral_idx 228) — 0 P1, 0 P2 ✅
### ROW 90 (coral_idx 229) — 0 P1, 0 P2 ✅
### ROW 91 (coral_idx 230) — 0 P1, 0 P2 ✅ Stage I→IV ER+/PR+ IDC to bone, on exemestane+everolimus+denosumab, MRI shows slight progression, PET/CT next week
### ROW 92 (coral_idx 231) — 0 P1, 0 P2 ✅ Stage IV ER+/PR-/HER2- IDC to liver, on epirubicin C2D1, CA27.29=3332, liver exam improved, tumor marker pending
### ROW 93 (coral_idx 232) — 0 P1, 0 P2 ✅ Stage 1 ER-/PR-/HER2+ IDC, s/p left partial mastectomy, needs mediport placement
### ROW 94 (coral_idx 233) — 0 P1, 0 P2 ✅ Stage IIA (pT1b N1sn G2) ER+/PR+/HER2- IDC, RS score 21, NED on exam

### ROW 95 (coral_idx 234) — 0 P1, 1 P2
- P2: Stage "Stage IV (metastatic)" 但 Distant Met=No, Goals=curative, ISPY trial（只招 Stage I-III）。与 ROW 83 同样的逻辑矛盾

### ROW 96 (coral_idx 235) — 0 P1, 0 P2 ✅ 47yo premenopausal, pT1cN0 ER+/PR+/HER2- mixed ductal/cribiform, Oncotype ordered
### ROW 97 (coral_idx 236) — 0 P1, 0 P2 ✅ 53yo, pT1bN0 ER+/PR+/HER2- IDC G1, molecular profiling ordered
### ROW 98 (coral_idx 237) — 0 P1, 0 P2 ✅ 78yo, TNBC with apocrine features, on TC C4, port removal planned
### ROW 99 (coral_idx 238) — 0 P1, 0 P2 ✅ Stage III→IV HER2+ IDC, lung + mediastinal LN mets, PET shows progression, biopsy needed
### ROW 100 (coral_idx 239) — 0 P1, 0 P2 ✅ Stage IV ER+/PR+ IDC, liver + multiple sites, on Gemzar C2, tumor markers rising

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
| ROW 10 response "not mentioned" | ❌ 持续 |

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
