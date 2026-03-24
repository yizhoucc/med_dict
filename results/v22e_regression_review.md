# v22e Regression Review: 20 samples 人工审查

Run: v22e_regression (20 samples)
Date: 2026-03-23
审查方式: **人工逐字审查**（关键 sample 全文阅读，其余对比 v22）

---

## Fix 验证（全部通过）

| Fix | Sample | 字段 | 值 | ✅ |
|-----|--------|------|-----|---|
| POST-IMAGING + LAB-SEARCH | Row 0 | imaging_plan | "Brain MRI" | ✅ |
| POST-LAB-SEARCH | Row 0 | lab_plan | "ordered...labs" | ✅ |
| POST-SELF-MANAGED | Row 13 | current_meds | "" | ✅ |
| POST-SELF-MANAGED-PLAN | Row 13 | medication_plan | 只留医生开具的 | ✅ |
| POST-SELF-MANAGED-PLAN | Row 13 | therapy_plan | "" | ✅ |
| POST-SELF-MANAGED-SUMMARY | Row 13 | summary | "previously on" | ✅ |
| POST-HER2-FISH | Row 13 | Type | "HER2-" | ✅ |
| POST-DISTMET-DEFAULT | Row 82 | Distant Met | "No" | ✅ |
| POST-MEDS-IV-CHECK | Row 89 | current_meds | "ac" | ✅ |
| POST-RECEPTOR-UPDATE | Row 94 | Type | "PR-" | ✅ |

## 回归检查（全部通过）

| Sample | 检查点 | 结果 |
|--------|--------|------|
| Row 1 | current_meds="Irinotecan" | ✅ |
| Row 4 | Referral 无泄漏 | ✅ |
| Row 33 | Type HER2- | ✅ |
| Row 40 | Patient type, goals=curative | ✅ |
| Row 56 | current_meds="", Type TNBC | ✅ |
| Row 71 | current_meds="letrozole"（无眼药水）| ✅ |
| Row 85 | in-person 正常 | ✅ |

## P1 残留（已知，不可修）

| Sample | 字段 | 问题 |
|--------|------|------|
| Row 5 | Stage | 空（可推断 Stage I）|
| Row 41 | Stage | "Not mentioned" |
| Row 58 | current_meds | exemestane+letrozole 时态 |
| Row 89 | Stage + DistMet | "Not mentioned" + "Not sure" |
| Row 99 | current_meds | 遗漏 Gemzar |

## LLM 随机差异（14 samples 有微变，无实质影响）

大部分是 summary 措辞、Type 格式、plan 字段细节变化。无 P0 级退化。

---

## 结论

```
20 samples 回归测试:
├── P0: 0
├── Fix 验证: 10/10 全部通过
├── 回归: 0
├── P1 残留: 5（全部已知，不可修）
├── LLM 随机差异: 14 samples（无实质影响）
└── POST hooks: 10 次触发，0 误触发
```
