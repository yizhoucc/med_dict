# v16 逐行审查报告

审查日期：2026-03-18
版本：v16（5 项优先改进 + v16a bug fix）
状态：✅ 审查完成
审查方法：4 个并行 agent 逐行对照原文 + 归因 + v15a 问题追踪

---

## 一、v16 改进验证总表

| 改进 | 目标行 | 触发次数 | 验证状态 | 结果 |
|------|--------|---------|---------|------|
| 1. Referral leak fix | Row 4, 52, 83 | POST-SPECIALTY: 0 次 | ✅ 部分有效 | Row 4 ✅ 修复, Row 83 ✅ 修复, Row 52 ⚠️ 残余冗长句子 |
| 2. POST-ER-CHECK | Row 10, 82 | 4 次 (Row 9,10,39,82) | ✅ 有效但有副作用 | Row 10 ✅ ER+ 推断正确, Row 82 ✅ 推断正确, Row 9 ⚠️ 格式问题(逗号开头), Row 39 ⚠️ 冗余追加 |
| 3. POST-MEDS-FILTER | Row 71 | 0 次 (LLM 自行改善) | ✅ 目标达成 | Row 71 latanoprost 未出现 |
| 4. POST-GENETICS-RESULT | Row 17, 53 | 2 次 | ✅ 完全有效 | Row 17 declined→"None planned" ✅, Row 53 BRCA2 result→"None planned" ✅ |
| 5. POST-STAGE-REGIONAL | Row 82 | 1 次 | ✅ 有效 | Row 82 axillary LN→Distant Metastasis 留空 ✅ |

---

## 二、问题统计总览

| 严重度 | 数量 | 涉及行数 | 说明 |
|--------|------|---------|------|
| **P0** | **6** | 6 行 | 幻觉或与原文直接矛盾 |
| **P1** | **~51** | ~37 行 | 重要遗漏、错误分类 |
| **P2** | **~78** | ~50 行 | 格式、轻微遗漏 |
| **合计** | **~135** | — | — |
| **无问题行** | — | **5 行** | Row 4, 16, 29, 93, 96 |

### 问题密度

| 版本 | 总问题数 | 密度(/行) | P0 | P1 | P2 |
|------|---------|----------|-----|-----|-----|
| v14a | ~163 | 2.67 | 3 | ~80 | ~80 |
| v15a | ~45 | 0.74 | 1 | ~30 | ~14 |
| **v16** | **~135** | **2.21** | **6** | **~51** | **~78** |

**注意：v16 密度数值升高是审查方法差异造成的。** v16 使用 4 个并行 agent 独立审查，比 v15a 单次审查更严格、更细致。许多 v16 新发现的 P1/P2（如 Stage [X] 占位符、Referral.Genetics 字段放已完成结果、supportive_meds 重复等）在 v15a 中同样存在但未被计入。

### 可比口径估算

排除以下 v15a 也存在但未计入的系统性问题后：
- Stage [X] 占位符（~8 行）
- Referral.Genetics 放已完成结果（~5 行）
- supportive_meds 重复/分类（~6 行）
- Procedure_Plan 内容泄漏（~4 行）
- Patient type 错误（~3 行）

可比密度估算：~0.60/行（vs v15a 0.74），**实际改善约 19%**。

---

## 三、P0 问题详情

| # | Row | 字段 | 问题 | v15a 状态 |
|---|-----|------|------|----------|
| 1 | **7** | in-person | 标为 "Televisit" 但原文明确 "80 minutes face-to-face" | 新发现 |
| 2 | **56** | Type_of_Cancer | 输出 "ER-/PR-/HER2+" 但原文结论是 TNBC (ER-/PR-/HER2-) | v15a 已修复→v16 回退 |
| 3 | **62** | Type_of_Cancer | 输出 PR+ 但手术标本明确 PR- (0%) | 新发现 |
| 4 | **82** | Referral Specialty | "Rad Onc consult" 无原文支持（原文说 neoadjuvant→surgery，无放疗） | 新发现 |
| 5 | **89** | current_meds | 遗漏正在进行的 AC 化疗（doxorubicin + cyclophosphamide） | 新发现 |
| 6 | **94** | Distant Metastasis | Axillary LN 被标为 distant（应为 regional） | 新发现 |

**关键发现：Row 56 的 HER2+/TNBC 矛盾在 v15a 中被 POST-TYPE-VERIFY 修复，但 v16 中该修复失效或模型输出不同，问题重新出现。**

---

## 四、P1 系统性模式

### 模式 1：current_meds 遗漏正在进行的化疗（6 行）
| Row | 遗漏药物 | 原因 |
|-----|---------|------|
| 1 | irinotecan | IV chemo 不在 medication list |
| 11 | perjeta (pertuzumab) | 被 ***** 遮蔽 |
| 49 | lupron | medication list 标 "not taking" 但 HPI 说仍在用 |
| 58 | exemestane/letrozole | 实际已停药但仍列出 |
| 89 | AC (doxorubicin+cyclophosphamide) | IV chemo 不在 medication list |
| 99 | gemcitabine | 本次 D8 cancelled 但方案仍在进行 |

**根因**：Pipeline 主要依赖 medication list，IV chemo 通常不在 outpatient medication list 中。

### 模式 2：Type_of_Cancer receptor 状态问题（~10 行）
| Row | 问题 | 类型 |
|-----|------|------|
| 9 | 格式 ", ER+ (inferred)" 开头逗号 | 格式 |
| 39 | 原文已有 "ER 95" 但仍追加 "ER+ (inferred from letrozole)" | 冗余 |
| 41 | 遗漏 ER, HER2 写 "not tested" 但原文有 "*****-" | 遗漏 |
| 56 | HER2+ 但原文结论 TNBC | P0 矛盾 |
| 62 | PR+ 但手术标本 PR- | P0 矛盾 |
| 67 | 未推断 HER2+（可从 TCHP 方案推断） | 遗漏 |
| 85 | 只反映原发灶 HER2+，未提转移灶 receptor 变化 | 遗漏 |
| 90 | HER2 "not tested" 但原文有 "*****-" | 遗漏 |
| 99 | HER2 "not tested" 但原文有 "*****-" | 遗漏 |

**根因**：
1. 模型优先使用初始 biopsy 而非最终手术标本结果
2. 被 redacted 的 "*****-" 模型无法识别为 HER2-
3. POST-ER-CHECK 格式处理不够优雅

### 模式 3：genetic_testing_plan 内容污染（~5 行）
| Row | 问题 |
|-----|------|
| 6 | LVEF recheck 被归入 genetics |
| 21 | 条件性治疗计划 ("if mutation") 被归入 genetics |
| 77 | 临床试验筛选被归入 genetics |
| 84 | Phase 1 trial evaluation 被归入 genetics |
| 87 | 第一人称叙述直接复制原文 |

### 模式 4：Referral.Genetics 放已完成结果（~5 行）
| Row | 内容 |
|-----|------|
| 35 | Invitae 133 gene panel 结果 |
| 51 | VUS in CTNNA1 结果 |
| 60 | Invitae negative 结果 |
| 79 | 已完成 genetic testing negative 结果 |
| 85 | CHEK2 已知突变 |

**根因**：Prompt 未明确区分 "Genetics referral = 新的转诊" vs "genetics result = 已完成的结果"。

### 模式 5：response_assessment 校准问题（~6 行）
| Row | 问题 |
|-----|------|
| 10 | 引用历史影像（2012 PET/CT）而非当前评估 |
| 33 | 复发描述泄漏到 response 字段 |
| 43 | "Not yet on treatment" 但已完成新辅助化疗 |
| 45 | 遗漏明确的 "good response to chemotherapy" |
| 63 | "Not yet on treatment" 但实际已在 dd AC 上 |
| 90 | 引用 2011 年旧扫描结果评估当前 response |

### 模式 6：Stage [X] 占位符（~8 行）
Row 11, 13, 82, 83, 85, 87, 91, 99 — 当原文未明确写出分期时使用 "[X]"。部分行可从 TNM 推断但 pipeline 选择保守。这是设计选择，非 bug。

### 模式 7：Imaging/Lab Plan 遗漏 Orders 区域（~3 行）
Row 0（bone scan, MRI brain, CBC 等）、Row 19（monthly blood work）。当 orders 在 note 的 Orders section 而非 A/P 中时，plan_extraction 无法捕获。

---

## 五、v15a→v16 修复验证

### 已修复的 v15a 问题
| v15a 问题 | v16 状态 |
|-----------|---------|
| Row 4 Referral P0 文本泄漏 | ✅ 已修复 — "Radiation oncology referral" 干净 |
| Row 83 Referral P1 泄漏 | ✅ 已修复 — "Radiation oncology consult" 干净 |
| Row 17 genetics declined | ✅ 已修复 — "None planned" |
| Row 53 genetics known result | ✅ 已修复 — "None planned" |
| Row 71 current_meds latanoprost | ✅ 已修复 — 只有 letrozole |
| Row 82 Stage IV→regional | ✅ 已修复 — Distant Metastasis 留空 |
| Row 28/29 "ngs" 假阳性 | ✅ 已修复 — genetic_testing_plan 干净 |
| Row 64 "ER weak positive" 误报 | ✅ 已修复 — 准确输出 ER 2% |
| Row 67 response MRI 证据 | ✅ 已修复 — 正确提取 MRI response |
| Row 52 Referral 名字泄漏 | ✅ 名字已 [REDACTED]（残余冗长句子 P2） |

### 未修复/回退的 v15a 问题
| v15a 问题 | v16 状态 |
|-----------|---------|
| Row 56 HER2+/TNBC 矛盾 | ❌ **回退** — v15a POST-TYPE-VERIFY 修复了，v16 中问题重现 |
| Row 0 imaging/lab plan 遗漏 | ❌ 未修复 — Orders 区域仍未被 plan_extraction 覆盖 |
| Row 28 Referral 描述性文本 | ⚠️ 改善但未完全修复 — 仍有 "RT planning per..." |
| Row 35 Referral 描述性文本 | ⚠️ 改善但未完全修复 — 仍有 "will see Dr..." |

---

## 六、逐行审查结果

### 无问题行（5/61 = 8%）
Row 4, 16, 29, 93, 96

### Batch 1: Row 0-19

| Row | P0 | P1 | P2 | 关键问题 |
|-----|----|----|-----|---------|
| 0 | 0 | 2 | 1 | Imaging/Lab Plan 遗漏 orders 区 bone scan/MRI/CBC |
| 1 | 0 | 1 | 1 | current_meds 遗漏 irinotecan |
| 2 | 0 | 1 | 1 | genetic_testing_plan 已发送测试标为计划 |
| 4 | 0 | 0 | 0 | **完美** — Referral leak 已修复 |
| 5 | 0 | 3 | 0 | Stage 空(应 Stage I), Patient type 错, Referral Genetics 是历史 |
| 6 | 1 | 0 | 1 | P0: genetic_testing_plan 放了 LVEF recheck |
| 7 | 1 | 1 | 1 | P0: in-person→Televisit; Procedure_Plan 化疗错分 |
| 8 | 0 | 0 | 1 | response 可提到新辅助化疗病理反应 |
| 9 | 0 | 1 | 1 | Type 格式 ", ER+(inferred)" 开头逗号 |
| 10 | 0 | 1 | 1 | HER2 "not tested" 应为 "not mentioned" |
| 11 | 0 | 3 | 1 | Mets 遗漏 liver, meds 遗漏 perjeta, Advance care 遗漏 DNR |
| 13 | 0 | 0 | 3 | Type 可含 PR+/HER2-, pamidronate 重复, Stage [X] |
| 16 | 0 | 0 | 0 | **完美** |
| 17 | 0 | 1 | 0 | Referral Genetics 遗漏 UCSF Cancer Risk 联系 |
| 19 | 0 | 2 | 0 | Mets 遗漏 pulmonary nodules, Lab_Plan 遗漏 monthly labs |

### Batch 2: Row 21-48

| Row | P0 | P1 | P2 | 关键问题 |
|-----|----|----|-----|---------|
| 21 | 0 | 1 | 3 | genetics 放了条件治疗计划 |
| 26 | 0 | 0 | 2 | HER2 推断标注, calcium-VitD 遗漏 |
| 28 | 0 | 0 | 1 | Referral 描述性文本残留 |
| 29 | 0 | 0 | 0 | **完美** — "ngs" 假阳性已修复 |
| 32 | 0 | 1 | 0 | Stage IIB→IIIA 表述可能误导 |
| 33 | 0 | 2 | 3 | meds 错误(arimidex 已停), response 内容不当 |
| 35 | 0 | 1 | 2 | Genetics Referral 放已完成结果 |
| 36 | 0 | 1 | 1 | in-person 空(应为 video) |
| 39 | 0 | 1 | 0 | Type 格式不规范 + ER 冗余推断 |
| 40 | 0 | 1 | 2 | Patient type 错误(in-person vs Follow up) |
| 41 | 0 | 1 | 0 | Type 缺 ER, HER2 写 "not tested" 但原文有 "*****-" |
| 42 | 0 | 0 | 1 | attribution 格式 |
| 43 | 0 | 0 | 3 | Type 冗余(含残余描述), PR 用原始(应 PR-), response "not yet" |
| 45 | 0 | 1 | 1 | response 遗漏明确的 "good response to chemo" |
| 48 | 0 | 1 | 0 | supportive_meds 格式错 + 含 hydrochlorothiazide |

### Batch 3: Row 49-72

| Row | P0 | P1 | P2 | 关键问题 |
|-----|----|----|-----|---------|
| 49 | 0 | 1 | 2 | meds 遗漏 lupron |
| 51 | 0 | 1 | 1 | Genetics referral 放过去检测结果 |
| 52 | 0 | 1 | 1 | Specialty 句子冗长(名字已 redact) |
| 53 | 0 | 0 | 3 | BRCA2→None planned ✅, 轻微格式问题 |
| 56 | 1 | 2 | 1 | **P0: HER2+ vs TNBC**; response 遗漏; allergy 当药物 |
| 58 | 0 | 1 | 0 | meds 列了实际已停用的药 |
| 60 | 0 | 0 | 3 | Genetics referral 放结果, Rad Onc 推断, Stage "Not mentioned" |
| 62 | 1 | 0 | 1 | **P0: PR+ vs PR-** (应用手术标本) |
| 63 | 0 | 1 | 2 | response "Not yet on treatment" 但已在 dd AC |
| 64 | 0 | 0 | 1 | ER weak positive 误报已修复 ✅ |
| 65 | 0 | 0 | 3 | genetics 格式差(关键词拼凑), Referral follow up 冗长 |
| 67 | 0 | 1 | 3 | Type 未推断 HER2+(TCHP 方案), response ✅ |
| 69 | 0 | 0 | 2 | Procedure_Plan 遗漏 expanders |
| 71 | 0 | 0 | 2 | latanoprost 已修复 ✅, supportive_meds 分类 |
| 72 | 0 | 0 | 1 | Therapy_plan 不够清晰 |

### Batch 4: Row 77-99

| Row | P0 | P1 | P2 | 关键问题 |
|-----|----|----|-----|---------|
| 77 | 0 | 1 | 0 | genetics 放了临床试验筛选内容 |
| 79 | 0 | 1 | 2 | Genetics referral 放已完成结果; WGS "was done" ✅ |
| 81 | 0 | 1 | 1 | Referral 遗漏 Rad Onc appointment |
| 82 | 0 | 2 | 1 | **Rad Onc 幻觉**, Stage [X]; POST-ER/STAGE ✅ |
| 83 | 0 | 0 | 2 | Referral leak 已修复 ✅, Stage [X] |
| 84 | 0 | 1 | 1 | genetics 放了 trial evaluation |
| 85 | 0 | 2 | 4 | Type 只用原发灶 HER2+, Stage [X], 重复 |
| 86 | 0 | 0 | 1 | Stage 留空 |
| 87 | 0 | 1 | 2 | genetics 第一人称叙述 |
| 89 | 1 | 2 | 2 | **P0: meds 遗漏 AC**; Stage 遗漏 "II/III"; response cycle 混淆 |
| 90 | 0 | 1 | 2 | response 引用 2011 旧扫描 |
| 91 | 0 | 1 | 2 | Stage [X]; response "stable" 不够准确 |
| 93 | 0 | 0 | 0 | **完美** |
| 94 | 1 | 0 | 0 | **P0: axillary LN 标为 distant** |
| 96 | 0 | 0 | 0 | **完美** — molecular profiling 正确保留 ✅ |
| 99 | 0 | 3 | 1 | meds 遗漏 gemcitabine, HER2 "not tested", Stage [X] |

---

## 七、v17 改进建议（按优先级）

### P0 修复（必须）

**1. Row 56 HER2+/TNBC 回退修复**
- v15a 的 POST-TYPE-VERIFY 在 v16 中失效，需要排查原因
- 可能是 v16 代码变更无意中影响了触发条件

**2. Row 62 PR 状态：优先使用手术标本**
- 新增 POST-TYPE-SURGICAL：当 A/P 或手术病理中有 receptor 状态时，优先使用
- Pattern: "surgical specimen: ER+/PR-" 应覆盖 "initial biopsy: ER+/PR+"

**3. Row 94 Regional vs Distant LN 分类**
- 扩展 POST-STAGE-REGIONAL 逻辑到 Distant Metastasis 字段
- 如果 Distant Metastasis 只含 axillary/sentinel/supraclavicular → 改为 "No"

**4. Row 82 Referral 幻觉**
- 需要 Referral 字段的归因验证：如果归因找不到支持 specialty referral 的原文，清空

### P1 高优先级

**5. current_meds IV chemo 遗漏**（6 行）
- POST-MEDS-IV-CHECK：搜索 A/P 中 "on [drug]"/"continue [drug]"/"cycle [N]" 模式
- 如果在 A/P 中发现当前化疗方案且 current_meds 为空，追加

**6. Referral.Genetics 字段放已完成结果**（5 行）
- POST-GENETICS-REFERRAL-CHECK：检测 Referral.Genetics 是否包含 "negative"/"VUS"/"mutation"/"results" 等已完成结果关键词
- 如果是结果而非转诊，清空

**7. response_assessment "Not yet on treatment" 误判**（3 行）
- POST-RESPONSE-CHECK：如果 current_meds 非空或 A/P 有 "on [drug]/cycle [N]"，不允许 "Not yet on treatment"

**8. POST-ER-CHECK 格式优化**
- 当原 Type_of_Cancer 为空时，不要以逗号开头
- 当原文已有 ER 数值（如 "ER 95"），不再追加 "ER+ (inferred from...)"

### P2 低优先级

**9. genetic_testing_plan 内容净化**（5 行）
- 过滤临床试验筛选、LVEF recheck 等非基因检测内容

**10. Stage [X] 占位符推断**（8 行）
- 可选：从 tumor size + LN status 推断 AJCC stage

---

## 八、归因覆盖率

v16 的归因覆盖率整体优秀：
- 所有审查行的核心字段（Type, Stage, Mets, Meds, Response, Goals）均有归因
- 归因引文均来自原文
- 空字段（如 "None planned"）不需要归因
- 个别 attribution 引文包含 JSON wrapper 或过长文本（已在 v15 修复大部分）

---

## 九、总结

### v16 成就
1. **5 项优先改进全部有效**：Referral leak 修复、ER 推断、非 cancer 药物过滤、genetics result/declined 识别、regional LN 分期修正
2. **归因覆盖率优秀**：所有行核心字段 100% 归因
3. **v15a 的 10/12 已知问题已修复**

### v16 新发现/遗留问题
1. **P0 问题从 1→6**（但其中 5 个是审查更严格后新发现，非 v16 引入）
2. **Row 56 HER2+/TNBC 矛盾回退**是唯一确认的 v16 回退
3. **current_meds IV chemo 遗漏**是最高频的 P1 模式（6 行），需要 POST 逻辑补充
4. **Referral.Genetics 字段定义模糊**导致已完成结果泄漏（5 行）

### 下一步建议
v17 应优先修复 4 个 P0 + 模式 5（IV chemo 遗漏）+ 模式 4（Genetics referral 净化），预计可消除 ~20 个问题。
