# v21 Full Review (Partial): 28 行人工逐字审查

Run: `results/v21_full_20260321_204929/` (partial, 28/61 rows)
Date: 2026-03-21
审查方式: **人工逐字审查**

---

## v20 → v21 差异分析（28 行中 13 行有差异）

### 改善 ✅
| ROW | 字段 | v20 → v21 | 判定 |
|-----|------|-----------|------|
| 6 | Stage | "" → "Approximately Stage I-II (1.5cm, 0/1 nodes neg)" | ✅ **7 版本空值终于填上** |
| 14 | current_meds | "Gemcitabine, Docetaxel, pamidronate" → "" | ✅ **POST-SELF-MANAGED 生效** |
| 17 | Stage | "estimated Stage I-II" → "pT2N0M0" | ✅ 更精确 |

### LLM 随机差异（无实质影响）
| ROW | 字段 | 差异 | 判定 |
|-----|------|------|------|
| 5 | Distant Met | 转移部位表述略变 | ⚠️ P2（v21 少了 sternum, 多了 brachial plexus）|
| 7 | Distant Met | 格式变化 | ✅ |
| 10 | Type | "ER+" → "HR+" | ⚠️ P2（略不精确）|
| 12 | Type | "IDC" → "breast cancer" | ⚠️ P2（丢失组织学亚型）|
| 20 | Distant Met | 格式变化 | ✅ |
| 22 | Type/Distant Met | 格式变化 | ✅ |
| 40 | Type | 截断 | ✅ |
| 44 | Type | 措辞变化 | ✅ |

### 退化 ❌（LLM 随机，非代码引入）
| ROW | 字段 | v20 → v21 | 严重性 |
|-----|------|-----------|--------|
| **34** | Stage | "Originally Stage III, now local recurrence" → "Stage III" | **P2** — 丢失 recurrence 信息 |
| **34** | Distant Met | "No" → "Not sure" | **P2** — PET 有 rib 不确定 uptake，两者均有道理 |
| **41** | **Stage** | "Not mentioned" → **"now metastatic (Stage IV)"** | **❌ P0** — 实际是 Stage II/III（3cm + micromet），非 Stage IV |
| **41** | **goals** | "curative" → **"adjuvant"** | **❌ P0** — "adjuvant" 不是合法值（应 curative/palliative）|
| **41** | **Distant Met** | "No" → "Not sure" | **P1** — 应为 No（无远处转移证据）|

---

## 🔴 新发现 P0: ROW 41 (Row 40) LLM 随机退化

**原文 A/P**: "32 y.o. female ATM mutation carrier...3 cm grade 3 IDC...SLN involved by micrometastasis...decided to proceed with AC-Taxol...After completion of chemotherapy I will recommend ovarian suppression with an aromatase inhibitor"

**这是早期乳腺癌辅助治疗**（curative intent），NOT Stage IV。

v21 LLM 输出了：
1. Stage = "now metastatic (Stage IV)" — 把 SLN micrometastasis 误解为远处转移
2. goals = "adjuvant" — 不是 prompt 定义的合法值（只有 curative/palliative）
3. Distant Met = "Not sure" — 应为 No

**这不是 v21 代码引入的**（v21 只改了 POST-SELF-MANAGED 和 IV-CHECK skip），是 LLM 非确定性行为。v20 在同一行输出了正确结果。

---

## v21 代码改动验证

| 检查项 | 结果 |
|--------|------|
| POST-SELF-MANAGED 仅在 Row 13 触发 | ✅（8 个信号词，零误触发）|
| POST-MEDS-IV-CHECK 在 Row 13 跳过 | ✅（无 faslodex 注入）|
| Row 1 irinotecan 保持 | ✅（LLM 自行提取，不需 IV-CHECK）|
| 其他行 current_meds 无变化 | ✅（15/28 行完全一致）|

**结论**: v21 代码改动零回归。ROW 41 的 P0 是 LLM 随机退化。

---

## 当前 v21 P0 统计（28/61 行）

| Row | 字段 | 问题 | 根因 |
|-----|------|------|------|
| 13 | summary | "currently on faslodex and palbociclib" | A/P 模板未更新 |
| 13 | medication_plan | "Continue low dose chemo" | 曲解医嘱 |
| 13 | therapy_plan | 同上 | 同上 |
| **40** | **Stage** | **"now metastatic (Stage IV)"** | **LLM 随机退化（v20 正确）** |
| **40** | **goals** | **"adjuvant"** | **LLM 随机退化** |
