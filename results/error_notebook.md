# 错题本 v3：v22 最终状态

## 当前状态 (v22)

```
61 行中:
P0: 1 (Row 13 summary 模板语言 — 无法修)
P1: ~5
P2: ~10
无问题: ~45 行 (74%)
```

---

## 🔴 唯一 P0

### Row 13 (coral_idx 153) — summary 模板语言
- **字段**: summary
- **值**: "currently on faslodex and palbociclib"
- **问题**: 患者 2 个月前已停药，A/P 模板未更新
- **历史**: v14a 起就有，从未修复
- **修复难度**: 极高 — 自由文本，POST hook 难精确修改

**已修复的 Row 13 问题**:
- ✅ current_meds: POST-SELF-MANAGED 清空 (v21)
- ✅ medication_plan: POST-SELF-MANAGED-PLAN 清除墨西哥药句子 (v22)
- ✅ therapy_plan: 同上 (v22)
- ✅ Type HER2: POST-HER2-FISH "unclear"→"HER2-" (v22)

---

## 🟡 P1 残留（~5 个）

| Row | 字段 | 问题 | 修复可能性 |
|-----|------|------|-----------|
| 0 | imaging_plan | 遗漏 MRI brain + bone scan | ❌ 结构性 |
| 0 | lab_plan | 遗漏 CBC/CMP 等 | ❌ 结构性 |
| 5 | Stage | 空（可推断 Stage I）| ⚠️ 可做 POST hook |
| 58 | current_meds | exemestane+letrozole 时态 | ❌ 难 |
| 99 | current_meds | 遗漏 Gemzar（cancelled 但仍在 regimen）| ❌ 边界 |

---

## ✅ 已修复清单（v14→v22 累计）

| 版本 | Fix | 解决 |
|------|-----|------|
| v15a | HER2 三 bug | Row 0,5,9 等 HER2 状态 |
| v16 | POST-REFERRAL regex | Row 4 Referral 泄漏 |
| v17 | POST-REFERRAL-VALIDATE | Row 82 Referral 幻觉 |
| v17 | POST-DISTMET-REGIONAL | Row 94 Distant Met 分类 |
| v17 | POST-TYPE-VERIFY-TNBC | Row 56 TNBC |
| v18 | POST-VISIT-TYPE | Row 85 in-person 空值 |
| v19 | POST-MEDS-IV-CHECK | Row 1 irinotecan, Row 89 AC |
| v19 | POST-PATIENT-TYPE | Row 40 Patient type |
| v20 | Pattern 6 regex | Row 56 docetaxel FP |
| v20 | POST-SUPP-ALLERGY | Row 56 过敏药过滤 |
| v20 | POST-MEDS-FILTER | Row 71 眼药水 |
| v21 | POST-SELF-MANAGED | Row 13 current_meds 清空 |
| v22 | POST-DISTMET-DEFAULT | Row 82 DistMet 空→"No" |
| v22 | POST-SELF-MANAGED-PLAN | Row 13 plan 字段清除 |
| v22 | POST-RECEPTOR-UPDATE | Row 94 PR+→PR- |
| v22 | POST-HER2-FISH | Row 13 HER2 "unclear"→"-" |

---

## 回归测试必测行

| Row | 检查点 |
|-----|--------|
| 1 | current_meds="Irinotecan" (IV-CHECK) |
| 13 | current_meds="" + med/therapy plan 清除 + Type HER2- |
| 56 | current_meds="" + supportive="" + Type TNBC |
| 82 | Distant Met="No" |
| 89 | current_meds="ac" (IV-CHECK) |
| 94 | Type PR- (Addendum 优先) |
