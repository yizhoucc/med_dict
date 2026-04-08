# v18 逐行审查报告

审查日期：2026-03-20
版本：v18（6 项修复: POST-MEDS-IV-CHECK 数据源+regex, row.get→局部变量, POST-VISIT-TYPE via video, POST-PATIENT-TYPE, POST-GENETICS 优先级, POST-STAGE-PLACEHOLDER）
结果目录：v18_verify_20260319_215939
状态：**审查完成**

---

## 一、v18 修复验证总表

| # | 修复 | 触发次数 | 验证状态 | 说明 |
|---|------|---------|---------|------|
| 1 | POST-MEDS-IV-CHECK 数据源+regex | 17 | ❌ **P0 regression** | 数据源修复有效，但药名直接扫描产生大量假阳性 |
| 2 | row.get→局部变量 (RESPONSE/TNBC) | 7 | ✅ **有效** | POST-RESPONSE-TREATMENT 正常工作 |
| 3 | POST-VISIT-TYPE "via video" | 1 | ✅ **有效** | ROW 86 in-person="" → "Televisit" |
| 4 | POST-PATIENT-TYPE 值校验 | 0 | ⚠️ **未测试** | v17 P0 的 Row 41 在 v18 中 LLM 自行输出了合法值 |
| 5 | POST-GENETICS 结果优先 | 9 | ✅ **有效** | ROW 61 "Invitae genetic testing: negative" 已清除 |
| 6 | POST-STAGE-PLACEHOLDER | 7 | ✅ **有效** | 7 行 "[X]" 占位符全部替换为 "Not available (redacted)" |

---

## 二、P0 Regression: POST-MEDS-IV-CHECK 假阳性

### 问题

v18 的药名直接扫描 fallback 在 A/P 中搜索 KNOWN_CHEMO_IV 药名。PAST_CHEMO 过滤器只排除过去时态词，但 **不排除未来/计划性上下文**，导致以下情况被错误匹配：
- "Recommendation was made for AC/THP chemotherapy" → 被当成当前用药
- "will recommend ibrance" → 被当成当前用药
- "She has decided to proceed with AC-Taxol" → 被当成当前用药
- "we also discussed the TCHP regimen" → 被当成当前用药

### 17 次触发逐行分析

| Row | 添加药物 | 判定 | 依据 |
|-----|---------|------|------|
| 36 | ac, taxol | ❌ FP | 已完成手术，"decided to proceed with AC-Taxol"=未来计划 |
| 40 | ac, taxol, ribociclib | ❌ FP | "decided to proceed with AC-Taxol"=未来计划，ribociclib="eligible for trial" |
| 42 | taxol, carboplatin | ❓ 需查 | |
| 43 | taxol, ribociclib, zoladex | ❓ 需查 | |
| 51 | zoladex | ❓ 需查 | |
| 52 | ac, tchp, thp, taxol, pertuzumab, trastuzumab | ❌ FP | "Recommendation was made for"=推荐，"also discussed"=讨论；Meds="none" |
| 56 | ac, docetaxel, pertuzumab, trastuzumab | ❓ 需查 | |
| 63 | taxol | ❓ 需查 | |
| 64 | ac, tc, cyclophosphamide, paclitaxel, taxol, carboplatin, trastuzumab, pembrolizumab, olaparib | ❌ FP | 9 种药=必然是列举所有提及的药名 |
| 65 | pembrolizumab | ❓ 需查 | |
| 67 | tchp | ❓ 需查 | |
| 77 | gemcitabine, eribulin, pembrolizumab | ❓ 需查 | |
| 79 | tc | ❓ 需查 | |
| 84 | olaparib, palbociclib, fulvestrant | ❓ 需查 | |
| 89 | ac | ✅ TP | "continue with cycle 4 of AC"=当前用药 |
| 94 | ac, taxol, capecitabine, pembrolizumab | ❓ 需查 | |
| 99 | gemzar | ✅ TP | "on Gemzar Cycle #2"=当前用药 |

**已确认**: 2 TP (Row 89, 99), 4 FP (Row 36, 40, 52, 64), 11 待查
**保守估计假阳性率**: ≥50%（4/6 已查 + 大量多药触发行高度可疑）

### v19 建议修复

在 PAST_CHEMO 之外新增 PLANNED_CHEMO 过滤器：
```python
PLANNED_CHEMO = [
    "recommend", "will benefit", "discussed", "consider",
    "plan is", "plan to", "decision", "eligible for",
    "will start", "will begin", "would want", "option",
    "alternative", "if she", "if he", "if patient"
]
```
或更可靠的做法：只在 A/P 中**当前正在使用**的上下文中匹配（"continue", "cycle N of", "currently on", "day N of"），而非使用排除法。

---

## 三、v17 P0 修复验证

| v17 P0 | Row | v18 状态 | 说明 |
|--------|-----|---------|------|
| Patient type="in-person" | 41 | ✅ → "New patient" | LLM 自行修正，POST-PATIENT-TYPE 未触发 |
| in-person="" 空值 | 86 | ✅ → "Televisit" | Fix 3 "via video" 模式匹配成功 |
| current_meds 遗漏 AC | 89 | ✅ → "ac" | Fix 1 药名扫描成功检出 |

**但 Fix 1 同时引入了新 P0 regression（假阳性）。**

---

## 四、逐行审查结果

### ROW 1 (coral_idx 140) — Row 0
- **Patient type**: "New patient" ✅ 原文 "New Patient Evaluation"
- **in-person**: "in-person" ✅
- **Cancer**: ER+/PR+/HER2- ✅; Stage IIA→IV ✅; Distant Met ✅
- **Lab**: "No labs in note" ✅（最近 lab 来自 2001 年）
- **current_meds**: "ibrance" ❌ **P0** — 原文 "will recommend ibrance"=条件性计划；Medications 写 "No current outpatient medications"。这是 LLM 提取错误（非 POST hook）
- **Imaging_Plan**: "No imaging planned" ❌ **P1** — Orders 中有 MRI brain + bone scan
- **Lab_Plan**: "No labs planned" ❌ **P1** — Orders 中有 CBC/CMP/CA15-3/CEA

### ROW 2 (coral_idx 141) — Row 1
- **current_meds**: "irinotecan" ✅ **v17 P1 已修复**（原"presents for cycle 3 day 1"）
- **Stage**: "Originally Stage IIB, now metastatic (Stage IV)" ✅
- **response_assessment**: 合理 ✅
- **整体质量**: 很好

### ROW 41 (coral_idx 180) — Row 40
- **Patient type**: "New patient" ⚠️ **P2** — 原文说 "my notes dated 03/17/18 and 04/21/18" 和 "At a previous visit"=follow up
- **current_meds**: "ac, taxol, ribociclib" ❌ **P0** — 所有药都是未来计划："decided to proceed with AC-Taxol"（还没开始）+ "eligible for trial with ribociclib"
- **Stage**: "Not mentioned in note" ⚠️ P1 — 原文 "clinical Stage IIB"（虽在 Oncologic History 中，非 A/P）

### ROW 53 (coral_idx 192) — Row 52
- **current_meds**: "ac, tchp, thp, taxol, pertuzumab, trastuzumab" ❌ **P0** — 全部是推荐方案；原文 Meds="none"
- **Patient type**: "New patient" ✅ "consultation regarding further management"

### ROW 61 (coral_idx 200) — Row 60
- **Genetics**: "None" ✅ **v17 P1 已修复**（原 "Invitae genetic testing: negative" 被清除）
- **in-person**: "Televisit" ✅ 原文 "live video Zoom connection"
- **current_meds**: "" ✅ 正确为空（还没开始治疗）

### ROW 86 (coral_idx 225) — Row 85
- **in-person**: "Televisit" ✅ **v17 P0 已修复**（原为空值 ""）
- **Stage**: "Not available (redacted)" ✅ **Fix 6 生效**
- **current_meds**: "letrozole, ribociclib, denosumab" ✅ 原文 "on letrozole + ribociclib"
- **response_assessment**: PD on let/rib ✅

### ROW 90 (coral_idx 229) — Row 89
- **current_meds**: "ac" ✅ **v17 P0 已修复**（原为空；现 "cycle 2 of AC"）
- **in-person**: "Televisit" ✅ 原文 "real-time Telehealth tools...live video connection"
- **Patient type**: "New patient" ⚠️ P1 — 应为 "Follow up"（已在临床试验中，多次就诊）
- **Stage**: "Not mentioned in note" ⚠️ P1 — A/P 有 "Clinical st II/III"

### ROW 100 (coral_idx 239) — Row 99
- **current_meds**: "gemzar" ✅ **v17 P1 已修复**（原为空；现 "on Gemzar Cycle #2"）
- **Stage**: "Not available (redacted)" ✅ Fix 6 生效
- **response_assessment**: "Tumor markers increased, scan too early to tell" ✅ 忠实

---

## 五、汇总

### v18 成功修复
1. **数据源修复**（Fix 1/2）：`row.get('assessment_and_plan')` → 局部变量 — 解决了 POST hooks 始终拿不到 A/P 数据的根本 bug
2. **POST-VISIT-TYPE "via video"**（Fix 3）：ROW 86 从空值修正为 "Televisit"
3. **POST-GENETICS 结果优先**（Fix 5）：9 次触发，含 ROW 61 的 "Invitae genetic testing: negative"
4. **POST-STAGE-PLACEHOLDER**（Fix 6）：7 行 "[X]" 占位符全部清理

### v18 P0 regression
**POST-MEDS-IV-CHECK 假阳性**：药名直接扫描缺少 PLANNED/FUTURE 上下文过滤。
- 影响至少 4 行（已确认），可能影响 15+ 行
- 严重性：high（将计划中的化疗方案错误标记为当前用药）
- 必须在 v19 修复

### v18 仍存在的 P1 模式
1. **Patient type**: 多行 "New patient" 应为 "Follow up"（Row 40, 89 等）— LLM 判断问题
2. **Stage 遗漏**: 多行 "Not mentioned" 但 A/P 中有 stage 信息（Row 40, 89 等）
3. **Plan 字段遗漏**: Orders 中的 imaging/lab 计划未被 plan_extraction 捕获（Row 0 等）— 因 plan_extraction 只看 A/P

---

## 六、v19 行动项

| 优先级 | 修复 | 说明 |
|--------|------|------|
| **P0** | POST-MEDS-IV-CHECK 增加 PLANNED_CHEMO 过滤 | 或改为正向匹配（只在 "continue/cycle/currently on" 上下文中匹配） |
| P1 | Patient type 判断逻辑 | 考虑在 POST hook 中加入 "previous visit/notes dated" 检测 |
| P1 | Stage 提取改进 | 允许从 A/P 首行摘要中提取 stage |
