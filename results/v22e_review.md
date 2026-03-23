# v22e Review: POST-IMAGING/LAB-SEARCH 修复审查

Run: v22e3 (4 rows: Row 0, 1, 13, 89)
Date: 2026-03-23
审查方式: **人工逐字审查**

---

## 修复验证

### Row 0 (ROW 1) — imaging/lab plan ✅ 修复

| 字段 | v22 | v22e | 原文依据 | 判定 |
|------|-----|------|---------|------|
| imaging_plan | "No imaging planned" | **"Brain MRI"** | HPI: "I also ordered a MRI of brain and bone scan" / Orders: "MR Brain with and without Contrast" | ✅ POST-IMAGING 匹配 "MRI of brain" |
| lab_plan | "No labs planned" | **"ordered...labs...Complete Blood Count"** | HPI: "labs to complete her work up" / Orders: "CBC, CMP, CA15-3, CEA, aPTT, PT" | ✅ POST-LAB-SEARCH 匹配 |

**P2 遗漏**: imaging_plan 只有 "Brain MRI"，缺 "Bone scan"。原因：这次 LLM 自己提取了 "Brain MRI"（非空），POST-IMAGING 的 bone scan 搜索因为 search_fullnote=False 跳过了。LLM 随机性导致每次提取不同。

**P2 格式**: lab_plan 包含了 HPI 原句 "ordered a MRI of brain and bone scan as well as labs" — 冗余且混入了 imaging 信息。

### Row 1 (ROW 2) — 回归 ✅
- current_meds = "Irinotecan" ✅
- 所有其他字段与 v22 一致 ✅

### Row 13 (ROW 14) — 全部修复保持 ✅
- summary = "**previously on** faslodex and palbociclib" ✅
- current_meds = "" ✅
- medication_plan = "Continue topical cannabis and sulfur. Rx given for Cymbalta." ✅（墨西哥药物清除）
- therapy_plan = "" ✅
- Type = "ER+ IDC, **HER2-**" ✅

### Row 89 (ROW 90) — 回归 ✅ + 误触发 P2
- current_meds = "ac" ✅ POST-MEDS-IV-CHECK
- **lab_plan = "Comprehensive Metabolic Panel"** ⚠️ **P2 误触发** — POST-LAB-SEARCH 匹配了 Results 段的已有检验结果标题，不是 A/P 中的 lab plan。原文 A/P 没有 lab plan。

---

## POST-LAB-SEARCH 误触发根因

POST-LAB-SEARCH 的 regex 之一:
```python
r'(?:Complete Blood Count|Comprehensive Metabolic Panel|Cancer Antigen)'
```

这直接匹配了 Results 段的标题：
```
Results for orders placed...
Comprehensive Metabolic Panel (BMP, AST, ALT...)
```

**修复方向**: regex 应该只在 "ordered/plan/will" + lab name 的上下文中匹配，不应裸匹配 lab name（会匹配到已有结果）。或者限制搜索范围排除 Results 段。

---

## 总结

```
v22e (4 rows):
├── 修复: Row 0 imaging/lab plan ✅ (从 "No planned" 到有值)
├── 回归: 0 (Row 1, 13, 89 全部保持)
├── P2 遗漏: Row 0 imaging 缺 bone scan (LLM 随机)
├── P2 误触发: Row 89 lab_plan (Results 段标题被误匹配)
└── P2 格式: Row 0 lab_plan 冗余
```

**建议**: 修复 POST-LAB-SEARCH 的裸匹配问题（加 future context 要求），避免误匹配 Results 段的已有检验。
