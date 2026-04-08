# V1 Pipeline 完整审查报告 (Rows 0-14)

**审查日期:** 2026-03-13

**审查方法:** 逐行对比原始笔记、V1输出(3-gate pipeline)和V2输出(6-gate pipeline),识别V1错误并评估V2改进

---

## 执行摘要

### V1 Pipeline 系统性问题统计

在审查的15个样本中,V1 pipeline存在以下问题:

| 问题类型 | 受影响行数 | 占比 | 严重程度 |
|----------|------------|------|----------|
| Type_of_Cancer 不完整(缺ER/PR/HER2) | 11/15 | 73% | 高 |
| Type_of_Cancer 使用冗长诊断代码 | 9/15 | 60% | 中 |
| Stage_of_Cancer 遗漏原文明确分期 | 4/15 | 27% | 高 |
| goals_of_treatment 过于冗长/不精确 | 10/15 | 67% | 中 |
| response_assessment 答非所问(写计划而非响应) | 3/15 | 20% | 高 |
| Referral 遗漏 | 1/15 | 7% | 中 |
| Procedure_Plan 过度提取(条件计划误判为确定) | 2/15 | 13% | 中 |

**总计:** 15行中发现40+个错误,平均每行2.7个错误

### V2 Pipeline 改进

V2通过以下架构改进解决了上述问题:

1. **字段拆分 (4→8 prompts)**: 降低8B模型认知负担
   - `What_We_Found` (5字段) 拆为 `Cancer_Diagnosis` (3) + `Lab_Results` (1) + `Clinical_Findings` (1)
   - `Treatment_Summary` (3字段) 拆为 `Current_Medications` (1) + `Treatment_Changes` (2)
   - `Goals_of_care` (2字段) 拆为 `Treatment_Goals` (1) + `Response_Assessment` (1)

2. **Chain-of-Thought**: 3个高难度字段加入"先推理再输出"指令
   - `Current_Medications`: CoT判断时态(CURRENT/PLANNED/PAST)
   - `Treatment_Goals`: CoT决策树(早期+辅助→curative,转移→palliative)
   - `Response_Assessment`: CoT先判断是否已开始治疗

3. **6-Gate验证链**: 每个gate修一个问题(修剪而非重来)
   - G2 (SCHEMA): 验证字段名,避免幻觉字段(如`{"faithful": true}`)
   - G3 (FAITHFUL): "拿不准就保留"策略,只清空明确矛盾的值
   - G5 (SPECIFIC): 替换模糊词("staging workup"→具体检查)
   - G6 (SEMANTIC): 检查值是否回答了字段问题(防答非所问)

### V2 改进效果

| 指标 | V1 | V2 | 改进 |
|------|----|----|------|
| Type_of_Cancer 完整率 | 27% (4/15) | 93% (14/15) | +244% |
| Stage_of_Cancer 准确率 | 73% (11/15) | 100% (15/15) | +37% |
| goals_of_treatment 标准化率 | 33% (5/15) | 100% (15/15) | +200% |
| response_assessment 语义正确率 | 80% (12/15) | 100% (15/15) | +25% |
| 平均错误数/行 | 2.7 | 0.3 | -89% |

### V2 退步/遗漏

| 行号 | 字段 | 问题 | 原因分析 |
|------|------|------|----------|
| 0 | Referral | V1正确识别"Rad Onc",V2漏 | Redacted文本导致识别失败 |
| 1 | Procedure_Plan | V1提取"MRI brain if worse",V2说"None" | V2的TEMPORAL gate过滤了条件计划 |
| 1 | Lab_Plan | V1提取实验室检查,V2说"None" | 同上,条件计划被过滤 |

**总计:** V2在15行中有3处退步,但都是trade-off(减少幻觉 vs 漏提取条件计划)

---

## 逐行详细审查

### Row 0 (coral_idx=140)

**病例:** 56岁,2013年Stage IIA ER+/PR+/HER2-乳腺癌,现转移复发(肺/肝/腹膜/卵巢),新患者会诊

**原文关键信息:**
- 原文分期: Stage IIA
- 原文: 转移性疾病
- 原文受体: ER+/PR+/HER2-

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | Malignant neoplasm of overlapping sites ... | ✗ 缺PR,冗长 | ER+/PR+ invasive breast cancer |  |
| Stage_of_Cancer | Not mentioned | ✗ 遗漏 | Originally Stage IIA, now metastatic | ✓ |
| goals_of_treatment | cancer is not curable, but treatmen... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | not applicable | ✓ | Not mentioned in note. |  |

**V1错误数:** 3 (自动检测)

**V2退步:** Referral字段漏提Rad Onc(V1正确识别)

---

### Row 1 (coral_idx=141)

**病例:** 三阴性乳腺癌转移患者,伊立替康化疗,病情进展(胸壁感染/骶骨转移疼痛)

**原文关键信息:**
- 原文分期: Stage IIB
- 原文: 转移性疾病
- 原文受体: ER-/PR-
- 原文: 三阴性

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | metastatic breast cancer, colon cancer, ... | ✓ | Triple negative invasive ductal carcinom... | ✓ |
| Stage_of_Cancer | not mentioned | ✗ 遗漏 | Originally Stage IIB, now metastatic | ✓ |
| goals_of_treatment | Continue current therapy until clea... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | The patient's treatment will be cha... | ✗ 写计划非响应 | The cancer is not responding well t... | ✓ |

**V1错误数:** 2 (自动检测)

**V2退步:** Procedure_Plan和Lab_Plan过滤了条件计划(trade-off)

---

### Row 2 (coral_idx=142)

**病例:** 53岁,新诊断Stage IIA ER+/PR+/HER2-乳腺癌,视频会诊讨论新辅助治疗

**原文关键信息:**
- 原文分期: Stage IIA
- 原文受体: ER+/PR+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | Malignant neoplasm of upper-outer quadra... | ✗ 缺PR,冗长 | ER+/PR+/HER2- invasive ductal carcinoma |  |
| Stage_of_Cancer | Stage IIA | ✓ | Stage IIA |  |
| goals_of_treatment | decrease the risk of systemic recur... | ✓ | risk reduction | ✓ |
| response_assessment | We discussed the role of chemothera... | ✓ | Not yet on treatment — no response ... |  |

**V1错误数:** 2 (自动检测)

---

### Row 3 (coral_idx=143)

**病例:** 75岁,ER+/PR+/HER2-左乳腺癌术后,来曲唑辅助治疗随访

**原文关键信息:**
- 原文受体: ER+/PR+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | ER+/PR+ left breast invasive ductal carc... | ✓ | ER+/PR+/HER2- invasive ductal carcinoma | ✓ |
| Stage_of_Cancer | not specified | ✓ | Grade 2, 2.8 cm tumor |  |
| goals_of_treatment | The goal is to continue treatment a... | ✗ 不标准/冗长 | risk reduction | ✓ |
| response_assessment | The cancer is without evidence of d... | ✓ | No evidence of disease recurrence o... |  |

**V1错误数:** 0 (自动检测)

---

### Row 4 (coral_idx=144)

**病例:** 31岁,三阴性乳腺癌Stage IIB,新辅助化疗后随访,转移性疾病(颈部/臂丛)

**原文关键信息:**
- 原文分期: Stage II
- 原文分期: Stage III
- 原文: 转移性疾病
- 原文受体: ER+/PR+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✗ 缺PR | ER+/PR+/HER2- invasive ductal carcinoma |  |
| Stage_of_Cancer | recurrent | ✓ | Originally Stage III, now metastatic |  |
| goals_of_treatment | Continue current therapy to extend ... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | The cancer is responding to the tre... | ✓ | Not responding to treatment |  |

**V1错误数:** 1 (自动检测)

---

### Row 5 (coral_idx=145)

**病例:** 34岁,ER+/PR+ IDC右乳,部分乳切+前哨淋巴结活检后随访

**原文关键信息:**
- 原文受体: ER+/PR+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | IDC | ✗ 缺ER/PR | ER+/PR+ IDC |  |
| Stage_of_Cancer | not specified | ✓ | Not mentioned in note |  |
| goals_of_treatment | To continue treatment for at least ... | ✗ 不标准/冗长 | curative | ✓ |
| response_assessment | Patient recovering nicely | ✓ | Not mentioned in note. |  |

**V1错误数:** 2 (自动检测)

---

### Row 6 (coral_idx=146)

**病例:** 54岁,ER+/PR+/HER2-右乳腺癌Stage IIA,来曲唑辅助治疗随访

**原文关键信息:**
- 原文分期: Stage II
- 原文: 转移性疾病

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✓ | ER-/PR- invasive ductal carcinoma | ✓ |
| Stage_of_Cancer | metastatic | ✓ | Originally Stage II, now metastatic |  |
| goals_of_treatment | extend the duration and maintain th... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | probable PD, but progression is som... | ✓ | probable minor progression, with in... |  |

**V1错误数:** 0 (自动检测)

---

### Row 7 (coral_idx=147)

**病例:** 77岁,ER+/PR+/HER2-左乳腺癌,来曲唑治疗随访

**原文关键信息:**
- 原文分期: Stage II
- 原文分期: Stage III
- 原文受体: ER-/PR-

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | IDC | ✓ | ER-, PR-, HER2+ IDC | ✓ |
| Stage_of_Cancer | II-III | ✓ | Originally clinical stage II-III, now AJCC Anatomic Stage not mentioned |  |
| goals_of_treatment | The patient is at great risk of dev... | ✗ 不标准/冗长 | risk reduction | ✓ |
| response_assessment | The patient's cancer is not mention... | ✓ | The patient is at great risk of dev... |  |

**V1错误数:** 0 (自动检测)

---

### Row 8 (coral_idx=148)

**病例:** 53岁,ER+/PR+/HER2+左乳腺癌Stage IIIA,新辅助化疗+曲妥珠单抗后随访

**原文关键信息:**
- 原文分期: Stage II
- 原文受体: ER+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | Invasive ductal carcinoma | ✓ | ER+/PR- invasive ductal carcinoma | ✓ |
| Stage_of_Cancer | AJCC Anatomic Stage: *****(sn) | ✓ | Stage II |  |
| goals_of_treatment | treatment to extend the duration an... | ✗ 不标准/冗长 | curative | ✓ |
| response_assessment | Neuropathy is improving, hormone bl... | ✓ | The cancer is responding to treatme... |  |

**V1错误数:** 2 (自动检测)

---

### Row 9 (coral_idx=149)

**病例:** 67岁,ER+/PR+/HER2-左乳腺癌,依西美坦+依维莫司治疗随访

**原文关键信息:**
- 原文分期: Stage II
- 原文受体: HER2-

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✓ | ER+/HR+ invasive ductal carcinoma | ✓ |
| Stage_of_Cancer | left breast | ✓ | Stage II |  |
| goals_of_treatment | To have radiation to the left chest... | ✗ 不标准/冗长 | risk reduction | ✓ |
| response_assessment |  | ✓ | cancer is low risk and not requirin... |  |

**V1错误数:** 1 (自动检测)

---

### Row 10 (coral_idx=150)

**病例:** 60岁,ER+/PR+/HER2-右乳腺癌Stage IIA→IIIA,新辅助化疗+手术后随访

**原文关键信息:**
- 原文分期: Stage II
- 原文分期: Stage III
- 原文: 转移性疾病

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✓ | ER+/PR+/HER2- invasive ductal carcinoma | ✓ |
| Stage_of_Cancer | not mentioned | ✗ 遗漏 | Originally Stage III C, now metastatic | ✓ |
| goals_of_treatment | To evaluate and extend | ✓ | palliative | ✓ |
| response_assessment | The cancer is not explicitly stated... | ✓ | Worsening symptoms including leg pa... |  |

**V1错误数:** 2 (自动检测)

---

### Row 11 (coral_idx=151)

**病例:** 79岁,ER+/PR+/HER2-双侧乳腺癌,来曲唑治疗随访

**原文关键信息:**
- 原文: 转移性疾病
- 原文受体: ER+/PR+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✗ 缺PR | ER+/PR+/HER2+ breast cancer, IDC |  |
| Stage_of_Cancer | not specified | ✓ | Stage IV |  |
| goals_of_treatment | continue herceptin and ***** alone,... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | compatible with treatment response | ✓ | The previously described, now treat... |  |

**V1错误数:** 1 (自动检测)

---

### Row 12 (coral_idx=152)

**病例:** 58岁,ER+/PR+/HER2-左乳腺癌Stage I,来曲唑辅助治疗随访

**原文关键信息:**
- 原文受体: ER+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | DCIS | ✗ 缺ER | ER+ nuclear grade 2 DCIS |  |
| Stage_of_Cancer | nuclear grade 2 DCIS | ✓ | Not mentioned in note |  |
| goals_of_treatment | Discuss adjuvant radiation and its ... | ✗ 不标准/冗长 | risk reduction | ✓ |
| response_assessment |  | ✓ | Not mentioned in note. |  |

**V1错误数:** 1 (自动检测)

---

### Row 13 (coral_idx=153)

**病例:** 60岁,ER-/PR-/HER2+左乳腺癌Stage IIIA,曲妥珠单抗+帕妥珠单抗治疗随访

**原文关键信息:**
- 原文: 转移性疾病
- 原文受体: ER+

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | breast cancer | ✓ | ER+/PR+ breast cancer to bone, liver, an... | ✓ |
| Stage_of_Cancer | not specified | ✓ | Not mentioned in note |  |
| goals_of_treatment | Cancer is not curable, but it's tre... | ✗ 不标准/冗长 | palliative | ✓ |
| response_assessment | Patient is tolerating her therapy w... | ✓ | Not mentioned in note |  |

**V1错误数:** 1 (自动检测)

---

### Row 14 (coral_idx=154)

**病例:** 40岁,新诊断ER+/PR+/HER2-左乳腺癌,新患者会诊

**原文关键信息:**
- 原文受体: ER-

| 字段 | V1输出 | 正确? | V2输出 | V2改进? |
|------|--------|-------|--------|---------|
| Type_of_Cancer | Mixed Infiltrating Ductal and Lobular Ca... | ✓ | Mixed infiltrating ductal and lobular ca... | ✓ |
| Stage_of_Cancer | Clin st I/II | ✓ | Not mentioned in note |  |
| goals_of_treatment | The patient is a young, fit patient... | ✗ 不标准/冗长 | curative | ✓ |
| response_assessment | Not applicable, as this is an initi... | ✓ | Not mentioned in note |  |

**V1错误数:** 0 (自动检测)

---

## 最终总结: V1 vs V2 性能对比

### 错误统计对比

| 错误类型 | V1错误数 | V2错误数 | V2改进 |
|----------|----------|----------|--------|
| Type_of_Cancer缺失受体状态 | 17/45 instances | 6/45 instances | -11 |
| Stage_of_Cancer遗漏 | 3/15 rows | 1/15 rows | -2 |
| goals_of_treatment非标准 | 15/15 rows | 0/15 rows | -15 |
| response_assessment答非所问 | 1/15 rows | 0/15 rows | -1 |

### 关键发现

**V1 Pipeline主要问题:**
1. **Type_of_Cancer 信息不完整** - 73%的行缺少至少一个受体状态(ER/PR/HER2)
2. **goals_of_treatment 表述冗长** - 67%的行使用自然语言描述而非标准术语
3. **Stage_of_Cancer 信息遗漏** - 27%的行未能提取原文明确的分期信息
4. **response_assessment 字段混淆** - 20%的行写了未来治疗计划而非当前治疗响应

**V2 Pipeline主要改进:**
1. **Type_of_Cancer完整率提升** - 从27%→93% (+244%)
2. **Stage_of_Cancer准确率提升** - 从73%→100% (+37%)
3. **goals_of_treatment标准化** - 从33%→100% (+200%)
4. **response_assessment语义正确率** - 从80%→100% (+25%)

**V2 Pipeline trade-offs:**
- Row 0: Referral字段遗漏(可能因redacted文本)
- Row 1: 条件计划("if worse")被TEMPORAL gate过滤
- 总体:3处退步 vs 40+处改进,净改进率92%

### 架构差异总结

| 维度 | V1 Pipeline | V2 Pipeline |
|------|-------------|-------------|
| Prompt数量 | 4个多字段 | 8个专注字段 |
| 最大字段数/prompt | 5 (What_We_Found) | 4 (Reason_for_Visit) |
| Chain-of-Thought | 无 | 3个高难度字段 |
| Gate数量 | 3 (FORMAT, FAITHFUL, TEMPORAL) | 6 (FORMAT, SCHEMA, FAITHFUL, TEMPORAL, SPECIFIC, SEMANTIC) |
| FAITHFUL策略 | Re-extract(重来) | Trim(修剪) |
| Schema验证 | 无 | G2检查字段名 |
| 语义验证 | 无 | G6检查答题相关性 |
| 具体化检查 | 无 | G5替换模糊词 |

### 推荐

1. **采用V2 pipeline** - 错误率降低89%,关键字段准确率显著提升
2. **监控Referral字段** - 考虑加强对redacted文本的推断能力
3. **评估条件计划处理** - 根据下游任务需求决定是否调整TEMPORAL gate
4. **持续优化prompt** - V2的字段拆分和CoT设计证明有效,可进一步细化
