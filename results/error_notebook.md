# 错题本 v4：v22 最终状态 — P0=0

## 当前状态 (v22)

```
61 行中:
P0: 0 🎉
P1: ~5
P2: ~10
无问题: ~45 行 (74%)
```

---

## ✅ 全部 P0 已修复

### Row 13 (coral_idx 153) — 墨西哥自管化疗（5 个 POST hook 协同）
- ✅ current_meds: POST-SELF-MANAGED 清空 (v21)
- ✅ medication_plan: POST-SELF-MANAGED-PLAN 清除 (v22)
- ✅ therapy_plan: POST-SELF-MANAGED-PLAN 清除 (v22)
- ✅ summary: POST-SELF-MANAGED-SUMMARY "currently on"→"previously on" (v22)
- ✅ Type HER2: POST-HER2-FISH "unclear"→"HER2-" (v22)

### Row 94 (coral_idx 234) — 术后受体状态变化
- ✅ Type PR: POST-RECEPTOR-UPDATE PR+→PR- (v22)

### Row 82 (coral_idx 222) — Distant Met 空值
- ✅ Distant Met: POST-DISTMET-DEFAULT ""→"No" (v22)

---

## 🟡 P1 残留（~5 个）

| Row | 字段 | 问题 | 修复可能性 |
|-----|------|------|-----------|
| 0 | imaging_plan | 遗漏 MRI brain + bone scan | ❌ 结构性 |
| 0 | lab_plan | 遗漏 CBC/CMP 等 | ❌ 结构性 |
| 5 | Stage | 空（可推断 Stage I）| ⚠️ 可做 |
| 58 | current_meds | exemestane+letrozole 时态 | ❌ 难 |
| 99 | current_meds | 遗漏 Gemzar | ❌ 边界 |

---

## 已修复清单（v14→v22 累计 17 项）

| 版本 | Fix | 解决 |
|------|-----|------|
| v15a | HER2 三 bug | Row 0,5,9 等 |
| v16 | POST-REFERRAL regex | Row 4 |
| v17 | POST-REFERRAL-VALIDATE | Row 82 |
| v17 | POST-DISTMET-REGIONAL | Row 94 |
| v17 | POST-TYPE-VERIFY-TNBC | Row 56 |
| v18 | POST-VISIT-TYPE | Row 85 |
| v19 | POST-MEDS-IV-CHECK | Row 1, 89 |
| v19 | POST-PATIENT-TYPE | Row 40 |
| v20 | Pattern 6 regex | Row 56 |
| v20 | POST-SUPP-ALLERGY | Row 56 |
| v20 | POST-MEDS-FILTER | Row 71 |
| v21 | POST-SELF-MANAGED | Row 13 |
| v22 | POST-DISTMET-DEFAULT | Row 82, 89 |
| v22 | POST-SELF-MANAGED-PLAN | Row 13 |
| v22 | POST-SELF-MANAGED-SUMMARY | Row 13 |
| v22 | POST-RECEPTOR-UPDATE | Row 94 |
| v22 | POST-HER2-FISH | Row 13 |

---

## 回归测试必测行

| Row | 检查点 |
|-----|--------|
| 1 | current_meds="Irinotecan" |
| 13 | current_meds="" + summary "previously on" + plan 清除 + HER2- |
| 56 | current_meds="" + supportive="" + TNBC |
| 82 | Distant Met="No" |
| 89 | current_meds="ac" + Distant Met="No" |
| 94 | Type PR- |
