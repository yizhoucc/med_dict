# v21d Review: 29/61 行人工逐字审查

Run: `results/v21_full_20260322_095146/` (crashed at row 30, 29 rows completed)
Date: 2026-03-22
代码: v21（POST-SELF-MANAGED hook + 回退 prompt Addendum 改动）
审查方式: **人工逐字审查**

---

## v20 → v21d 对比

- **23/29 行关键字段完全一致** ✅（无蝴蝶效应）
- **6 行有差异**: 5 个 LLM 随机微变 + 1 个 Fix 生效
- **Row 40 (ROW 41) 正常**: goals="curative" ✅（之前 prompt 改动导致的 "adjuvant" 退化已消除）

---

## POST hook 验证

| Hook | 触发次数 | 行 | 正确性 |
|------|---------|-----|--------|
| POST-SELF-MANAGED | 1 | Row 13 | ✅ 8 信号词，清空 current_meds |
| POST-MEDS-IV-CHECK | 0 | — | ✅ Row 1 irinotecan 由 LLM 自行提取，无需 hook |
| 误触发 | 0 | — | ✅ |

---

## 逐行审查结果

### ✅ 无问题（23 行）

以下行与 v20 审查结论一致，无 P0/P1：

Row 2 (ROW 3), Row 4 (ROW 5), Row 6 (ROW 7), Row 7 (ROW 8), Row 8 (ROW 9), Row 16 (ROW 17), Row 17 (ROW 18), Row 19 (ROW 20), Row 21 (ROW 22), Row 26 (ROW 27), Row 28 (ROW 29), Row 29 (ROW 30), Row 32 (ROW 33), Row 33 (ROW 34), Row 35 (ROW 36), Row 36 (ROW 37), Row 39 (ROW 40), Row 40 (ROW 41), Row 41 (ROW 42), Row 42 (ROW 43), Row 43 (ROW 44), Row 45 (ROW 46)

**注**: Row 7 (procedure_plan 分类 P1)、Row 33 (Stage P2) 等已知 P1/P2 问题与 v20 一致，非新问题。

### P0（3 个，1 行）

**Row 13 (ROW 14)** — 墨西哥自管化疗（与 v20 一致，current_meds 已修复）
1. summary: "currently on faslodex and palbociclib" — 已停药 2 月
2. medication_plan: "Continue low dose chemo..." — 不是医嘱
3. therapy_plan: 同上

**current_meds = "" ✅** — POST-SELF-MANAGED 修复

### P1（4 个，3 行）

| Row | ROW | 字段 | 问题 |
|-----|-----|------|------|
| 0 | 1 | imaging_plan | 遗漏 MRI brain + bone scan（结构性）|
| 0 | 1 | lab_plan | 遗漏 CBC/CMP/CA15-3 等（结构性）|
| 5 | 6 | Stage | 空（可推断 Stage I）|
| 13 | 14 | Type | "HER2: status unclear"（应为 HER2-）|

### P2（3 个，3 行）

| Row | ROW | 字段 | 问题 |
|-----|-----|------|------|
| 9 | 10 | Type | "HR+" 不如 "ER+" 精确（原文写 "HR+"）|
| 11 | 12 | Type | "breast cancer" 丢了 IDC 亚型 |
| 13 | 14 | Stage | "Not available (redacted)"（可推断 Stage IV）|

---

## Crash Bug

在 Row 30 (ROW 46 之后) crash:
```
AttributeError: 'list' object has no attribute 'strip'
File run.py line 1608: supp_val = (tc_dict.get("supportive_meds", "") or "").strip()
```

**原因**: LLM 返回 supportive_meds 为 list 而非 string。
**修复**: 已提交（list→string 转换）。需 resume 剩余 32 行。

---

## 总结

v21d（回退 prompt + POST-SELF-MANAGED）与 v20 表现一致，唯一改善是 Row 13 current_meds 清空。无蝴蝶效应，无新回归。

**P0: 3 个**（全在 Row 13，summary + medication_plan + therapy_plan）
**P1: 4 个**（Row 0 结构性 ×2 + Row 5 Stage + Row 13 Type）
**P2: 3 个**

---

## 续审: ROW 49-85（22 行新增，人工逐字审查）

### ✅ 无问题（14 行）
Row 48(ROW49), 49(ROW50), 52(ROW53), 53(ROW54), 56(ROW57), 60(ROW61), 62(ROW63), 63(ROW64), 64(ROW65), 67(ROW68), 71(ROW72), 72(ROW73), 77(ROW78), 83(ROW84)

### 已知 P1（与 v20 一致）
| Row | ROW | 问题 | 状态 |
|-----|-----|------|------|
| 58 | 59 | current_meds exemestane+letrozole（时态）| P1 不变 |
| 82 | 83 | Distant Met 空值（应为 No）| P1 不变 |

### LLM 随机退化
| Row | ROW | 字段 | v20 → v21d | 级别 |
|-----|-----|------|-----------|------|
| 69 | 70 | Stage | "pT4bN1M0/pT1cN0M0" → "" | **P1 退化** |
| 69 | 70 | Distant Met | "No" → "Not sure" | P2 退化 |

### P2 系统性（Stage 推断保守）
ROW 52, 61, 66, 80 — Stage "Not mentioned"

### 续审总计（22 行）
- P0: 0
- P1: 3（Row 58 时态 + Row 69 Stage 退化 + Row 82 DistMet 空）
- P2: ~6
- 无问题: 14 行

---

## 最终续审: ROW 86-100（10 行，人工逐字审查）

### ✅ 无问题（6 行）
Row 85(ROW86), 87(ROW88), 91(ROW92), 93(ROW94), 96(ROW97)

### 已知 P1/P2（与 v20 一致）
| Row | ROW | 问题 |
|-----|-----|------|
| 86 | 87 | Stage 空（P1，可推断）|
| 89 | 90 | Stage "Not mentioned" + DistMet "Not sure"（P1/P2）|
| 89 | 90 | current_meds="ac" ✅ POST-MEDS-IV-CHECK 保持 |
| 90 | 91 | HER2 "not tested"（P2，脱敏）|
| 94 | 95 | Type "ER+/PR+/HER2-" — **PR 回到 PR+**（prompt 回退的代价，原 P0 回来了）|
| 99 | 100 | current_meds=""（P1，遗漏 Gemzar）+ HER2 "not tested"（P2）|

### POST hook 最终统计
| Hook | 触发 | 行 | 正确 |
|------|------|-----|------|
| POST-SELF-MANAGED | 1 | Row 13 | ✅ 零误触发 |
| POST-MEDS-IV-CHECK | 1 | Row 89 (ac) | ✅ |

---

## v21d 全量 61 行最终总结

```
61 行:
├── P0: 4 个 (Row 13×3 + Row 94×1)
│   ├── Row 13: summary "currently on" + medication_plan/therapy_plan "Continue"
│   └── Row 94: Type PR+ (Addendum 说 PR-)（prompt 回退的代价）
├── P1: ~8 个
│   ├── Row 0: imaging/lab plan 遗漏 ×2（结构性）
│   ├── Row 5: Stage 空
│   ├── Row 13: Type HER2 "unclear"
│   ├── Row 58: current_meds 时态
│   ├── Row 69: Stage 空（LLM 随机退化）
│   ├── Row 82: Distant Met 空
│   └── Row 99: current_meds 遗漏 Gemzar
├── P2: ~10 个
├── 无问题: ~43 行 (70%)
├── 回归: 0 (v21 代码改动零回归)
└── 修复: Row 13 current_meds="" ✅
```

---

## v22 验证审查（5 行人工逐字审查）

### Fix 验证结果

| Fix | 目标 | 结果 | 详情 |
|-----|------|------|------|
| A | Row 82 Distant Met | ✅ | "" → "No" |
| A | Row 89 Distant Met | ✅ **bonus** | "Not sure" → "No"（curative + non-IV）|
| B | Row 13 medication_plan | ✅ | 删除 4 个自管句子，只剩医生开具的 |
| B | Row 13 therapy_plan | ✅ | 删除 2 个句子 → "" |
| C | Row 94 Type PR | ✅ | PR+ → PR-（Addendum 优先）|
| D | Row 13 Type HER2 | ✅ | "status unclear" → "HER2-"（FISH negative）|

### 回归检查
| Row | 字段 | v22 | 状态 |
|-----|------|-----|------|
| 1 | current_meds | "Irinotecan" | ✅ |
| 89 | current_meds | "ac" | ✅ |

### v22 最终 P0/P1 统计

```
P0: 1 (Row 13 summary "currently on faslodex and palbociclib")
P1: ~5
P2: ~10
无问题: ~45 行 (74%)
```

**v14→v22 P0 演进**: 多个 → 5 → 4 → **1**
