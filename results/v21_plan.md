# v21 修复计划

## 目标
修复 v20 全量审查发现的 5 个 P0 问题（2 行）。

---

## P0 清单

| # | Row | 字段 | 问题 | 根因 |
|---|-----|------|------|------|
| 1 | 13 | summary | "currently on faslodex and palbociclib"（已停药 2 月）| A/P 模板未更新，模型照抄 |
| 2 | 13 | current_meds | 包含墨西哥自管药物 | 模型无法理解医生不认可态度 |
| 3 | 13 | medication_plan | "Continue low dose chemo"（不是医嘱）| 同 #2，下游传播 |
| 4 | 13 | therapy_plan | 同上 | 同 #2 |
| 5 | 94 | Type_of_Cancer | "ER+/PR+/HER2-"（Addendum 显示 PR-/HER2 equivocal）| 模型跟随 A/P 模板忽略 Addendum |

---

## 修复方案

### Fix 1: POST-SELF-MANAGED hook（解决 Row 13 P0-2）

**位置**: run.py POST hooks 区域（POST-MEDS-FILTER 之后）

**逻辑**:
1. 检测 A/P (assessment_and_plan) 中是否有不认可语言信号:
   - `"apparently"`, `"so-called"`, `"claims to be"`, `"for some reason she believes"`, `"discontinue our medications"`, `"on her own"`, `"self-administered"`, `"mexico"`, `"another country"`, `"outside"`
2. 检查 Current Outpatient Medications 段是否有癌症药:
   - 在 note_text 中找 "Current Outpatient Medications" 或 "MEDICATIONS" 段
   - 检查是否包含 ONCO_WHITELIST 中的任何药物（排除 "not taking" / "discontinued" 标注的）
3. 条件: 信号词 ≥2 个 AND 门诊药物列表无癌症药 → 清空 current_meds
4. 日志: `[POST-SELF-MANAGED] Cleared current_meds (physician disapproval detected)`

**风险评估**:
- 信号词 "apparently" 单独出现可能不够特异（其他笔记也可能用）
- 要求 ≥2 个信号词 降低误报
- 交叉验证 outpatient med list 作为第二重保障
- 61 行中仅 Row 13 有这种模式，误伤风险极低

### Fix 2: POST-SELF-MANAGED 扩展（解决 Row 13 P0-3/4）

**位置**: plan_extraction POST hooks 区域

**逻辑**:
- 如果 POST-SELF-MANAGED 触发清空了 current_meds:
  - 在 medication_plan 和 therapy_plan 中搜索被清空的药物名
  - 将 "Continue [drug]" 替换为空或标注为 "Patient self-administered (not physician-managed)"
  - 或者更简单：如果 medication_plan/therapy_plan 包含被清空的药物，清除相关部分

**替代方案**: 不改代码，接受为已知限制（极端边界案例）

### Fix 3: Prompt 修改（解决 Row 94 P0-5）

**位置**: prompts/extraction.yaml，Cancer_Diagnosis section

**修改**:
在 Type_of_Cancer schema 描述中加入:
```
IMPORTANT: If the note contains a surgical pathology Addendum with updated immunohistochemistry results (ER/PR/HER2), use the MOST RECENT pathology results. Receptor status can change after neoadjuvant treatment. The Addendum results supersede the A/P description if they differ.
```

**风险评估**:
- 仅影响有 Addendum 的笔记（少数）
- 不会影响没有 Addendum 的正常笔记
- 可能让模型花更多时间搜索 Addendum（增加 latency）

### Fix 4: POST-RECEPTOR-UPDATE hook（Fix 3 的 POST 备选）

**位置**: run.py POST hooks 区域（POST-HER2-CHECK 附近）

**逻辑**:
1. 在 note_text 中搜索 "Addendum" 段
2. 在 Addendum 中搜索 receptor 结果:
   - "progesterone receptors is negative/positive"
   - "estrogen receptors is negative/positive"
   - "HER2...equivocal/positive/negative"
3. 与 Type_of_Cancer 中的受体状态对比
4. 如有矛盾，更新为 Addendum 的结果

**风险评估**:
- Addendum 解析可能不够鲁棒（不同格式）
- 但 CORAL 数据集中 Addendum 格式较统一

---

## 实施优先级

| 优先级 | Fix | 解决 P0 | 复杂度 | 风险 |
|--------|-----|---------|--------|------|
| 1 | Fix 1 (POST-SELF-MANAGED) | Row 13 P0-2 | 中 | 低（≥2 信号词 + med list 交叉验证）|
| 2 | Fix 3 (Prompt Addendum) | Row 94 P0-5 | 低 | 低（仅影响有 Addendum 的笔记）|
| 3 | Fix 2 (Plan 扩展) | Row 13 P0-3/4 | 中 | 中（可能过度清除计划内容）|
| 4 | Fix 1 扩展 → summary | Row 13 P0-1 | 高 | 高（summary 是自由文本，难精确修改）|

**建议执行**: Fix 1 + Fix 3 先做（解决 2/5 P0，最可靠）。Fix 2 可选。Fix 4 备用如果 Fix 3 无效。Row 13 P0-1 (summary) 和 P0-3/4 (plan) 可接受为已知限制。

---

## 回测计划

修改后只测 2 行: Row 13 + Row 94
- Row 13: 验证 current_meds 被清空
- Row 94: 验证 Type 使用 Addendum 受体状态

之后跑错题本必测 6 行回归测试: Row 1, 13, 56, 58, 89, 94
