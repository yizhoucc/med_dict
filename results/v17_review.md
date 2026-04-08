# v17 逐行审查报告

审查日期：2026-03-19
版本：v17（8 项修复 + POST-VISIT-TYPE 逻辑修正）
结果目录：v17_verify_20260319_080005
状态：**审查完成**

---

## 一、v17 修复验证总表

| # | 修复 | 触发次数 | 目标行 | 验证状态 | 说明 |
|---|------|---------|--------|---------|------|
| 1 | POST-TYPE-VERIFY-TNBC | 1 | Row 56 | ✅ **有效** | HER2+ → HER2-, triple negative 已添加 |
| 2 | POST-DISTMET-REGIONAL | 1 | Row 94 | ✅ **有效** | axillary LN → Distant Met = "No" |
| 3 | POST-REFERRAL-VALIDATE | 2 | Row 60, 82 | ✅ **有效** | Rad Onc 幻觉已清除；Row 94 真实 Rad Onc 保留 |
| 4 | POST-VISIT-TYPE (修正版) | 0 | Row 7 | ⚠️ **v16 P0 为误报** | Row 7 ZOOM=Televisit 本就正确；但 Row 86 空值未处理 |
| 5 | POST-ER-CHECK 格式 | 2 | Row 10, 82 | ✅ **有效** | 无开头逗号；Row 39 ER 95 数值格式正确识别 |
| 6 | POST-GENETICS 扩展 | 8 | Row 1/35/40/51/69/83/85/89 | ✅ **大部分有效** | VUS/negative/基因名清除成功；Row 61/80 漏网 |
| 7 | POST-MEDS-IV-CHECK | **0** | Row 89 | ❌ **完全失败** | 0 触发，3+ 行仍遗漏 IV 化疗 |
| 8 | POST-RESPONSE-TREATMENT | 2 | Row 39, 71 | ✅ **有效** | "Not yet on treatment" 已修正；Row 63 未触发 |

---

## 二、v16 P0 修复验证

| v16 P0 | Row | v17 状态 | 说明 |
|--------|-----|---------|------|
| in-person→Televisit | 7 | ✅ **v16 误报** | 原文 "presents through ZOOM"，Televisit 正确 |
| HER2+/TNBC 回退 | 56 | ✅ **已修复** | POST-TYPE-VERIFY-TNBC：ER-/PR-/HER2- breast cancer, triple negative |
| PR+ vs PR- (手术标本) | 62 | ✅ **已修复** | 模型自行记录双受体状态："later reclassified as ER+/PR-" |
| Rad Onc 幻觉 | 82 | ✅ **已修复** | POST-REFERRAL-VALIDATE：Specialty="" |
| current_meds 遗漏 AC | 89 | ❌ **未修复** | POST-MEDS-IV-CHECK 0 触发 |
| axillary LN=distant | 94 | ✅ **已修复** | POST-DISTMET-REGIONAL：Distant Met="No" |

**修复率: 5/6 (83%)**。Row 89 AC 化疗遗漏是唯一未修复的 v16 P0。

---

## 三、逐行审查结果

### 审查方法
- 10 行详细手动审查（逐字段对照原文）
- 12 行 POST hook 触发行关键字段验证
- 8 行随机抽查
- 61 行全量系统性扫描（Patient type/in-person/Stage/Genetics/Distant Met/response+meds 矛盾）

### 详细审查行

| Row (idx) | coral_idx | P0 | P1 | P2 | 关键问题 |
|-----------|-----------|----|----|-----|---------|
| 0 | 140 | 0 | 2 | 1 | imaging/lab plan 遗漏 Orders 内容 |
| 1 | 141 | 0 | 1 | 2 | current_meds 遗漏 irinotecan |
| 2 | 142 | 0 | 0 | 1 | 非常干净 |
| 4 | 144 | 0 | 0 | 2 | 非常干净 |
| 7 | 147 | 0 | 2 | 1 | Metastasis "No" (3/28 LN+)；Social work referral 幻觉 |
| 40 | 180 | **1** | 1 | 0 | **Patient type="in-person"** (应为 Follow up)；Stage 未提 |
| 56 | 196 | 0 | 1 | 2 | ✅TNBC 修复！过敏列为 supportive_meds |
| 60 | 200 | 0 | 1 | 0 | Genetics "Invitae: negative" 未清除 |
| 62 | 202 | 0 | 1 | 1 | ✅PR 修复！Metastasis "No" (3/4 LN+) |
| 63 | 203 | 0 | 1 | 0 | response "Not yet on treatment" 但 A/P 说 "currently on" |
| 79 | 219 | 0 | 1 | 0 | Genetics 有完成的阴性结果未清除 |
| 82 | 222 | 0 | 2 | 1 | ✅Rad Onc 修复！Stage "[X]"占位符；Distant Met 空 |
| 85 | 225 | **1** | 0 | 0 | **in-person="" 空值** (应为 Televisit) |
| 89 | 229 | **1** | 2 | 3 | **current_meds 空** (AC 化疗)；Patient type 错；Stage 未提 |
| 94 | 234 | 0 | 1 | 3 | ✅Distant Met 修复！PR 状态变化未记录 |
| 99 | 239 | 0 | 1 | 0 | current_meds 空 (Gemzar/gemcitabine 活跃) |

---

## 四、v17 新发现的 P0 问题

| # | Row | 问题 | 原因 |
|---|-----|------|------|
| 1 | 40 (ROW 41) | Patient type = "in-person" | LLM 输出字段串位，非 POST hook 问题 |
| 2 | 85 (ROW 86) | in-person = "" (空) | LLM 未输出；POST-VISIT-TYPE 不处理空值 |
| 3 | 89 (ROW 90) | current_meds 空 (AC 化疗) | POST-MEDS-IV-CHECK 完全失败 (0 触发) |

---

## 五、系统性问题模式

### 1. POST-MEDS-IV-CHECK 完全失败 (0/61 触发)
**影响**: 至少 3 行 (Row 1, 89, 99) 遗漏当前 IV 化疗
**根因推测**: regex 模式 `continue\s+(\w+)` 无法匹配 "continue with cycle 4 of AC"（中间有 "with cycle 4 of"）；`cycle\s+\d+\s+(?:of|day\s+\d+\s+of)\s+(\w+)` 理论上应匹配但实测未触发，需调试
**v18 建议**: 需要在实际 A/P 文本上测试 regex 模式

### 2. Stage "[X]" 占位符 (7 行)
**影响行**: ROW 12, 14, 83, 84, 86, 92, 100
**根因**: 原始 stage 数据被脱敏（*****），模型输出 "[X]" 作为占位符
**v18 建议**: POST-STAGE 检查如果 stage 含 "[X]" 或 "[REDACTED]" 则替换为 "Not available (redacted)"

### 3. Genetics 结果未清除 (2 行)
**影响行**: ROW 61 ("Invitae genetic testing: negative"), ROW 80 ("no mutation in...")
**根因**: POST-GENETICS 的 REFERRAL_WORDS 包含 "testing"，导致 has_referral=True，阻止清除
**v18 建议**: 从 REFERRAL_WORDS 移除 "testing"，或在结果词优先级高于 referral 词时仍清除

### 4. Metastasis "No" 但有 LN+ (至少 3 行)
**影响行**: ROW 8 (3/28 LN+), ROW 44, ROW 63
**根因**: LLM 不将 regional LN metastasis 算作 "Metastasis"
**v18 建议**: 可考虑 POST hook 从 pathology 中检测 LN+ 并修正 Metastasis 字段

### 5. POST-VISIT-TYPE 空值未处理 (1 行)
**影响行**: ROW 86
**根因**: POST-VISIT-TYPE 只在已有值为 Televisit/in-person 时校验，不处理空值
**v18 建议**: 增加空值分支，从 note 中推断 visit type

---

## 六、v17 vs v16 对比

| 指标 | v16 | v17 | 变化 |
|------|-----|-----|------|
| P0 问题数 | 6 | 3 | **-50%** |
| v16 P0 修复 | - | 5/6 (83%) | ✅ |
| POST hook 有效 | 5/5 | 7/8 (88%) | 1 个失败 (MEDS-IV-CHECK) |
| Genetics 扩展 | 2 触发 | 8 触发 | ✅ 显著改善 |
| ER-CHECK 格式 | 有开头逗号 | 无 | ✅ 已修复 |
| RESPONSE-TREATMENT | 未实现 | 2 触发 | ✅ 新增有效 |

---

## 七、v18 优先级建议

### P0 修复 (3 项)
1. **POST-MEDS-IV-CHECK 调试**: 在实际 A/P 文本上测试 regex，修复匹配失败
2. **POST-VISIT-TYPE 空值处理**: 增加 in-person="" 的分支
3. **Patient type 校验**: 增加 POST 检查，Patient type 只能是 "New patient" 或 "Follow up"

### P1 模式修复 (3 项)
4. **Genetics REFERRAL_WORDS**: 移除 "testing" 或改变优先级逻辑
5. **Stage 占位符清理**: "[X]" → "Not available"
6. **Metastasis LN+ 检测**: 从 pathology 推断 regional metastasis

---

## 八、POST Hook 触发明细

| Row (idx) | Hook | Action |
|-----------|------|--------|
| 1 | POST-GENETICS | Cleared mutation result |
| 10 | POST-ER-CHECK | Inferred ER+ from letrozole |
| 17 | POST-GENETICS-RESULT-CHECK | Removed declined test |
| 35 | POST-GENETICS | Cleared VUS |
| 39 | POST-RESPONSE-TREATMENT | Corrected "Not yet on treatment" |
| 40 | POST-GENETICS | Cleared mutation |
| 51 | POST-GENETICS | Cleared VUS/CTNNA1 |
| 53 | POST-GENETICS-RESULT-CHECK | Removed BRCA2 result |
| 56 | POST-TYPE-VERIFY-TNBC | HER2+ → HER2-, triple negative |
| 60 | POST-REFERRAL-VALIDATE | Removed Rad Onc (not in note) |
| 69 | POST-GENETICS | Cleared BRCA1 mutation |
| 71 | POST-RESPONSE-TREATMENT | Corrected "Not yet on treatment" |
| 82 | POST-REFERRAL-VALIDATE | Removed Rad Onc |
| 82 | POST-STAGE-REGIONAL | Corrected Stage IV → regional |
| 82 | POST-ER-CHECK | Inferred ER+ from letrozole |
| 83 | POST-GENETICS | Cleared CHEK2 mutation |
| 85 | POST-GENETICS | Cleared CHEK2 |
| 89 | POST-GENETICS | Cleared BLM mutation carrier |
| 94 | POST-DISTMET-REGIONAL | Axillary LN → Distant Met = No |
| — | POST-VISIT-TYPE | **0 triggers** |
| — | POST-MEDS-IV-CHECK | **0 triggers** |
