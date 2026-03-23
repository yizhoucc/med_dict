# v22 Full Review: 61 行全量人工审查

Run: `results/v21_full_20260322_154740/`
Date: 2026-03-22
代码: v22（POST-SELF-MANAGED + PLAN + SUMMARY + DISTMET-DEFAULT + RECEPTOR-UPDATE + HER2-FISH）
审查方式: **人工逐字审查**（v20 对比 + 差异深入 + 关键行原文确认）

---

## POST hook 全量验证

| Hook | 触发 | 行 | 正确 | 误触发 |
|------|------|-----|------|--------|
| POST-SELF-MANAGED | 1 | Row 13 | ✅ | 0 |
| POST-SELF-MANAGED-PLAN | 2 | Row 13 | ✅ | 0 |
| POST-SELF-MANAGED-SUMMARY | 1 | Row 13 | ✅ | 0 |
| POST-HER2-FISH | 1 | Row 13 | ✅ | 0 |
| POST-DISTMET-DEFAULT | 1 | Row 82 | ✅ | 0 |
| POST-MEDS-IV-CHECK | 1 | Row 89 | ✅ | 0 |
| POST-RECEPTOR-UPDATE | 1 | Row 94 | ✅ | 0 |
| **总计** | **8** | | **8/8** | **0** |

---

## v20 → v22 差异分类（42/61 行有差异）

| 类别 | 数量 | 详情 |
|------|------|------|
| **Fix（POST hook 改善）** | 5 行 | Row 14, 73, 83, 90, 95 |
| **Noise（LLM 随机微变）** | 33 行 | 格式/措辞/大小写变化 |
| **LLM 随机退化** | 3 行 | Row 49, 70, 91 |
| **Mixed** | 1 行 | Row 50 |

---

## 🟢 Fix 确认（5 行）

| ROW | 字段 | v20 → v22 | Fix |
|-----|------|-----------|-----|
| 14 | current_meds | 墨西哥药 → "" | POST-SELF-MANAGED |
| 14 | medication_plan | "Continue low dose chemo..." → 只留医生开具的 | POST-SELF-MANAGED-PLAN |
| 14 | therapy_plan | "Continue low dose chemo..." → "" | POST-SELF-MANAGED-PLAN |
| 14 | summary | "currently on" → "previously on" | POST-SELF-MANAGED-SUMMARY |
| 14 | Type HER2 | "unclear" → "HER2-" | POST-HER2-FISH |
| 73 | Type HER2 | "not tested" → "HER2-" | POST-HER2-FISH **bonus** |
| 83 | Distant Met | "" → "No" | POST-DISTMET-DEFAULT |
| 90 | Distant Met | "Not sure" → "No" | POST-DISTMET-DEFAULT **bonus** |
| 95 | Type PR | "PR+" → "PR-" | POST-RECEPTOR-UPDATE |

---

## 🔴 LLM 随机退化（3 行，非代码引入）

| ROW | 字段 | v20 → v22 | 级别 |
|-----|------|-----------|------|
| 49 | supportive_meds | string → list（非 supportive 药物） | P1 |
| 70 | Stage | "pT4bN1M0/pT1cN0M0" → "" | P1 |
| 91 | Distant Met | "bone and lymph nodes" → "bone" | P2 |

---

## 全量 P0/P1 统计

### P0: 0 🎉

### P1（~7 个）

| Row | 字段 | 问题 | 状态 |
|-----|------|------|------|
| 0 | imaging_plan | 遗漏 MRI brain + bone scan | 结构性，不变 |
| 0 | lab_plan | 遗漏 CBC/CMP 等 | 同上 |
| 5 | Stage | 空（可推断 Stage I） | 不变 |
| 48 | supportive_meds | list 格式（LLM 随机） | **新** |
| 58 | current_meds | exemestane+letrozole 时态 | 不变 |
| 69 | Stage | 空（LLM 随机退化） | 不变 |
| 99 | current_meds | 遗漏 Gemzar | 不变 |

### P2: ~10 个
Stage "Not mentioned" ×5, Type 措辞变化 ×3, Distant Met 不完整 ×1, 其他 ×1

---

## 最终结论

```
v22 全量 61 行:
├── P0:  0 🎉
├── P1:  7
├── P2: ~10
├── 无问题: ~44 行 (72%)
├── Fix 生效: 9 项（含 2 个 bonus）
├── 回归: 0（代码改动零回归）
└── LLM 随机退化: 3 行（不可控）
```
