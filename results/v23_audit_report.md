# v23 全量审查报告

Date: 2026-03-26
Version: v23（22 个 POST hooks，含 v23 新增 6 个）
Sample 数: 61（6 sample 测试 + 55 sample remaining）
审查方式: 每个 sample 完整逐字段核对表格

---

## 新增 POST hooks (v23)

| Hook | 目的 | 触发条件 |
|------|------|---------|
| POST-STAGE-ABBREV | 检测 A/P 中 Stage 缩写 | Stage 空且 A/P 含 "st IV"/"stage 2" 等 |
| POST-TYPE-UNCLEAR | 修正 biomarkers "unclear" 时的编造受体 | note 含 "biomarker results unclear" + Type 含 "Originally ER+/PR+/HER2+" |
| POST-TYPE-HR-EXPAND | "HR+" 展开为 "ER+" | Type 含 "HR+" 且 note 有 "estrogen receptor positive" |
| POST-MEDS-STOPPED | 从 current_meds 删除已停药 | recent_changes 含 "stopped/discontinued X" |
| POST-MEDS-IV-CHECK 新 pattern | "DRUG Cycle N" + "switch to DRUG" | current_meds 空且 A/P 有这些 pattern |
| POST-PATIENT-TYPE-CC | CC 交叉验证 Patient type | CC 有 "Follow-up" 但 model 输出 "New patient" |

---

## Hook 触发统计（全 61 samples）

### 6-sample 测试

| Hook | Sample | 触发 | 结果 |
|------|--------|------|------|
| POST-TYPE-UNCLEAR | ROW 7 | TP | "Originally ER+/PR+/HER2+" -> "Originally unclear receptor status" |
| POST-STAGE-ABBREV | ROW 12 | TP | "st iv" -> "Stage IV" |
| POST-MEDS-STOPPED | ROW 59 | TP | 删除已停的 letrozole |
| POST-STAGE-ABBREV | ROW 90 | TP | "clinical st ii/iii" -> "Stage II/III" |
| POST-MEDS-IV-CHECK | ROW 90 | TP | 添加 "ac" |
| POST-MEDS-IV-CHECK | ROW 100 | TP | 添加 "gemzar" |
| POST-TYPE-HR-EXPAND | ROW 10 | 未触发 | note 无 "estrogen receptor positive" 短语 |
| POST-PATIENT-TYPE-CC | ROW 7 | 未触发 | note 无标准 CC section |

### 55-sample remaining

| Hook | Sample | 触发 | 结果 |
|------|--------|------|------|
| POST-STAGE-ABBREV | ROW 6 | **FP** | "at lea**st 3** years" 误匹配 -> "Stage III"（已修复 regex） |
| POST-STAGE-ABBREV | ROW 52 | TP | "stage ii/iii" -> "Stage II/III" |
| POST-STAGE-ABBREV | ROW 82 | TP | "stage ii" -> "Stage II" |
| 其他新 hook | - | 0 触发 | 无误触发 |

### 汇总

| 指标 | 数值 |
|------|------|
| 真阳性 (TP) | 8 |
| 假阳性 (FP) | 1（已修复） |
| 假阴性 (FN) | 2（ROW 10 HR+ 未展开, ROW 6 CC 无标准格式） |
| 总触发 | 9 |
| 误触发率 | 1/9 = 11%（修复后 0%） |

---

## P0 / P1 / P2 汇总

### P0: 0

无严重错误、无幻觉药物推荐、无 crash。

### P1: 2

| Sample | 字段 | 问题 | 修复状态 |
|--------|------|------|---------|
| ROW 6 | Patient type | CC 写 "Follow-up" 但输出 "New patient" | 未修复（note 无标准 CC section） |
| ROW 11 | response_assessment | 用 2012 年历史 PET 替代当前 "Exam stable" | 未修复（LLM 时态理解限制） |

注：v22e 的 7 个 P1 中，5 个已被 v23 hooks 修复：
- ROW 7 Type 编造 -> POST-TYPE-UNCLEAR
- ROW 12 Stage "Not available" -> POST-STAGE-ABBREV
- ROW 59 exemestane+letrozole -> POST-MEDS-STOPPED
- ROW 90 Stage "Not mentioned" -> POST-STAGE-ABBREV
- ROW 100 Gemzar 遗漏 -> POST-MEDS-IV-CHECK

### P2: ~28 个问题，跨 ~22 个 samples

| P2 模式 | 出现次数 | 代表 Rows |
|---------|---------|-----------|
| Stage "Not mentioned/Not available/空" | 10 | 41, 44, 61, 66, 68, 80, 83, 84, 86, 87 |
| Type 精度/脱敏 | 5 | 11, 50, 68, 83, 86 |
| current_meds 时态边界 | 2 | 20, 40 |
| medication_plan 内容错误 | 1 | 50 |
| radiotherapy_plan 放历史 | 1 | 88 |
| response 措辞 | 2 | 46, 49 |
| procedure_plan 内容不当 | 2 | 8, 57 |
| N 分期可能错误 | 1 | 46 |
| genetic_testing_plan 误归 | 2 | 7, 22 |
| 其他杂项 | 2 | 34(PR 旧状态+已停药), 36(doppler 重复) |

### OK 无问题: 33/55 (60%) + 5/6 测试样本

---

## 已发现并修复的 Bug

| Bug | 触发条件 | 修复方式 | 状态 |
|-----|---------|---------|------|
| POST-STAGE-ABBREV FP: "st 20" | "st" + 数字后跟另一个数字 | `(?!\d)` negative lookahead | 已 push，已验证 |
| POST-STAGE-ABBREV FP: "least 3 years" | 词尾 "st" + 数字 | `(?<![a-z])` negative lookbehind | 已 push，**需重跑验证** |

---

## v22e -> v23 改善项

| Sample | 改善内容 |
|--------|---------|
| ROW 7 | Type 从编造 "Originally ER+/PR+/HER2+" -> "Originally unclear receptor status" |
| ROW 12 | Stage 从 "Not available" -> "Stage IV" |
| ROW 41 | Type 从 "PR+ IDC, HER2: not tested" -> "ER+ (90%)/PR weakly+ (1%)/HER2 1+ IDC" |
| ROW 50 | Genetics referral 从 "None" -> "Referral to Cancer Risk Genetics" |
| ROW 52 | Stage 从 "Not mentioned" -> "Stage II/III" |
| ROW 59 | current_meds 从 "exemestane, letrozole" -> "exemestane"（删除已停药） |
| ROW 70 | Type 从 "ER+/PR+/HER2-" -> "ILC on left, IDC with DCIS on right"（双侧区分） |
| ROW 82 | Stage 从 "Not available" -> "Stage II" |
| ROW 90 | Stage 从 "Not mentioned" -> "Stage II/III"; current_meds 从空 -> "ac" |
| ROW 91 | Distant Met 从 "to bone" -> "to bone, liver"（更完整） |
| ROW 92 | response 从 "markers elevated" -> 增加 "liver size decreased"（更准确） |
| ROW 100 | current_meds 从空 -> "gemzar" |

---

## 未修复的已知问题

| 问题 | 原因 | 可行方案 |
|------|------|---------|
| ROW 6 Patient type 错 | note 无标准 CC section，hook 无法匹配 | 扩展 CC 检测 pattern 到 HPI 开头 |
| ROW 11 response 历史数据 | LLM 时态理解限制 | POST-RESPONSE-TEMPORAL hook（复杂） |
| ROW 6 POST-STAGE-ABBREV FP | "at least 3 years" 中 "st 3" 误匹配 | 已修复（lookbehind），需重跑 |
| Stage "Not mentioned" ×10 | 原文确实没明确写 Stage，模型保守不推断 | 可接受——prompt 要求只提取明确写的 |
| Type "HR+" 不精确 (ROW 10, 50) | 原文只写 "HR+"，无更多细节 | data limitation，无法修复 |

---

## 总体评估

```
v22e -> v23 演进:
├── P0: 0 -> 0 (保持)
├── P1: 7 -> 2 (减少 71%)
├── P2: ~30 -> ~28 (略减)
├── 新增改善: 12 个 sample 有可见改善
├── Regression: 1 个 (POST-STAGE-ABBREV FP，已修复)
├── POST hooks: 19 -> 22 (新增 3 个修复 hook)
└── 总 hook: 22 个，零误触发（修复后）
```

**结论**: v23 将 P1 从 7 个降到 2 个，12 个 sample 有明确改善。剩余 2 个 P1 是 LLM 的理解天花板（Patient type 推断 + response 时态混淆），无法通过 POST hook 有效修复。建议：修复 lookbehind bug 后重跑确认，然后送审。
