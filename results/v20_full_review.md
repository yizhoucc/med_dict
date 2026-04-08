# v20 Full Review: 61 行全量回归测试

Run: `results/v20_full_20260321_115736/`
Date: 2026-03-21
审查方式: 4 batch 并行 agent 审查（非逐字逐句人工审查，需对重点行补充深入验证）

---

## 总览

| 统计 | 数量 |
|------|------|
| 总行数 | 61 |
| P0 未解决 | 1 |
| P1 问题 | ~15 |
| P2 问题 | ~12 |
| 确认正常 | ~33 |

---

## 🔴 P0 问题（1行）

### Row 13 (ROW 14, coral_idx 153) — 墨西哥自管化疗
- **current_meds**: "Gemcitabine 200mg, Docetaxel 20 mg, pamidronate" — ❌ **P0** 医生不认可的墨西哥自管药物
- **Type_of_Cancer**: "ER+ breast cancer, HER2: status unclear" — P1（原文 FISH negative = HER2-）
- **Stage_of_Cancer**: "Not available (redacted)" — P2（应为 Stage IV）
- **supportive_meds**: "pamidronate once weekly" — P1（与 current_meds 重复）
- **状态**: v19-v20 持续未解决

---

## 🟡 P1 问题（~15行）

### current_meds 类

| Row | ROW | 问题 | 详情 |
|-----|-----|------|------|
| 58 | 59 | 时态错误 | "exemestane, letrozole" — 停 A 换 B 场景，两者都不应算 current |
| 99 | 100 | 遗漏 | current_meds="" — 应有 gemcitabine（当前 regimen，本次 cancelled 但仍在方案中）|

### Type_of_Cancer 类

| Row | ROW | 问题 | 详情 |
|-----|-----|------|------|
| 6 | 7 | 原始受体不明确 | "Originally ER+/PR+/HER2+" — 原文 "Biomarker results unclear" |
| 13 | 14 | 缺 PR+/HER2- | "ER+ breast cancer, HER2: status unclear" — 应为 ER+/PR+/HER2- |
| 40 | 41 | 缺 FISH 信息 | "ER+/PR weakly+/HER2 1+ by IHC" — 应注明 FISH not available |
| 41 | 42 | ER 推断 | "ER+/PR+/HER2-" — ER 来自 tamoxifen 推断（原文 ER 被脱敏）|
| 89 | 90 | 完全缺失 | "ER/PR/HER2 status not mentioned" — P1 |

### Stage_of_Cancer 类

| Row | ROW | 问题 | 详情 |
|-----|-----|------|------|
| 4 | 5 | 空值 | Stage="" — 原文有 "Stage III... now metastatic" |
| 5 | 6 | 推断保守 | Stage="" — 1.5cm node neg 应为 Stage I |
| 10 | 11 | 表述不准 | "Originally Stage IIIC, now metastatic" — 可直接写 "Stage IV"（A/P 有 "St IV"）|
| 86 | 87 | 空值 | Stage="" — 2.2cm, 4/19 LN+ 可推断 Stage IIA-IIB |

### Distant Metastasis 类

| Row | ROW | 问题 | 详情 |
|-----|-----|------|------|
| 10 | 11 | 缺 liver | "Yes, to bone" — 应含 liver（有 liver bx + "3 new liver lesions"）|
| 82 | 83 | 空值 | dmet="" — 应为 "No" |

### 其他

| Row | ROW | 字段 | 问题 |
|-----|-----|------|------|
| 0 | 1 | imaging/lab plan | 遗漏 MRI brain + bone scan + labs（Orders 段，非 A/P）|
| 7 | 8 | procedure_plan | 混入化疗 "AC x 4 cycles"（应为 therapy）|

---

## ✅ 已修复确认

| Row | ROW | 之前问题 | v20 状态 |
|-----|-----|---------|---------|
| 1 | 2 | current_meds 遗漏 irinotecan | ✅ "Irinotecan" |
| 4 | 5 | Referral 文本泄漏 | ✅ 修复 |
| 9 | 10 | Type 缺 HER2 | ✅ "ER+, HER2-" |
| 17 | 18 | Stage 遗漏 | ✅ "pT1b, pNX — approximately Stage I" |
| 40 | 41 | Patient type 串位 | ✅ "Follow up" |
| 56 | 57 | docetaxel FP + 过敏药 + TNBC | ✅ 全部修复 |
| 71 | 72 | 眼药水 latanoprost | ✅ POST-MEDS-FILTER 过滤 |
| 82 | 83 | Referral 幻觉 | ✅ 修复 |
| 84 | 85 | current_meds 空 | ✅ "capecitabine, zolendronic acid" |
| 85 | 86 | in-person 空值 | ✅ 修复 |
| 89 | 90 | current_meds 遗漏 AC | ✅ "ac" |
| 94 | 95 | Distant Met 分类错 | ✅ "No" |

---

## 🟢 确认正常的行（agent 审查无问题）

Row 2, 8, 9, 16, 19, 21, 26, 28, 29, 32, 33, 35, 36, 39, 42, 43, 45, 48, 49, 51, 52, 53, 60, 63, 64, 65, 67, 69, 72, 77, 79, 81, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 96

---

## 📊 vs 错题本对比

### 错题本 Tier 1 必测行结果

| Row | 期望 | v20 实际 | 结论 |
|-----|------|---------|------|
| 1 | irinotecan ✅ | ✅ "Irinotecan" | 修复保持 |
| 13 | P0 墨西哥药 | ❌ P0 仍在 | 未解决 |
| 56 | 三项修复 ✅ | ✅ 全部修复 | 修复保持 |
| 58 | exemestane+letrozole P1 | ❌ P1 仍在 | 未解决 |
| 89 | AC ✅ | ✅ "ac" | 修复保持 |
| 94 | Type PR 状态 | ✅ "ER+/PR+/HER2-" | v20 改善 |

### 错题本 Tier 2 建议测行结果

| Row | 期望 | v20 实际 | 结论 |
|-----|------|---------|------|
| 0 | imaging/lab plan P1 | ❌ P1 仍在 | 结构性问题 |
| 5 | Stage 推断 P1 | ❌ P1 仍空 | 未解决 |
| 7 | procedure_plan P1 | ❌ P1 仍在 | 分类问题 |
| 11 | Stage+liver P1 | ❌ P1 仍在 | 未解决 |
| 33 | Type HER2 | ✅ "ER+/PR+/HER2-" | 修复！|
| 41 | Type ER/HER2 | ⚠️ ER 推断（P2 级别）| 改善 |
| 71 | 眼药水 | ✅ 过滤成功 | 修复！|
| 90 | Type HER2 | ❌ P1 完全缺失 | 未解决 |

### 新发现问题（v20 首次发现）

| Row | 问题 | 级别 |
|-----|------|------|
| 99 | current_meds 遗漏 gemcitabine | P1 |
| 82 (ROW 83) | Distant Met 空值 | P1 |

### 回归检查
- v20 三个改动（Pattern 6 regex、SELF-MANAGED prompt、POST-SUPP-ALLERGY）**未引入新的回归**
- 所有 v19 修复项在 v20 保持稳定

---

## ⚠️ 审查质量说明

本次审查使用 4 个并行 agent，每个 agent 审查 ~15 行。审查方式为"扫描关键字段"，**非逐字逐句对照原文**。以下行建议做人工深入审查：

### 建议人工逐字审查的行
1. **Row 13** — P0，需确认 A/P 中所有不认可语言
2. **Row 10** — Distant Met 缺 liver，需人工确认原文证据
3. **Row 99** — current_meds 遗漏，需确认 gemcitabine 是否仍在 current regimen
4. **Row 6** — Type 原始受体状态，需人工确认 "Biomarker results unclear"
5. **Row 89** — Type 完全缺 ER/PR/HER2，需确认原文是否有受体信息

---

## 📈 v20 全量总结

```
61 行中:
├── P0:  1 行  (Row 13 墨西哥药物 — 唯一)
├── P1: 15 行  (current_meds×2, Type×5, Stage×4, Dist Met×2, 其他×2)
├── P2: 12 行  (Stage推断保守、supportive分类等)
├── OK: 33 行  (54% 无问题)
└── 回归: 0 行  (v20 改动未引入新问题)
```
