# 肿瘤科临床笔记结构化提取 Pipeline 总结

> **项目**: med_dict — 乳腺癌肿瘤科笔记 → 结构化提取 → 患者可读信
> **模型**: Qwen2.5-32B-AWQ (4-bit quantization)
> **Pipeline**: V2, 5-Gate + 21 POST
> **日期**: 2026-03-15

---

## 1. Executive Summary

### 项目目标

从 CORAL 数据集的乳腺癌肿瘤科门诊笔记（去标识化）中提取结构化临床信息，并生成面向患者的通俗解释信。

### 四个不可妥协的原则（按优先级）

1. **精确忠实** — 绝对不能幻觉，不能编造任何信息。宁可少说，不可说错
2. **不遗漏** — 笔记中的重要临床信息都要覆盖到，不能丢掉关键内容
3. **简单词汇** — 8 年级英语水平（eighth-grade reading level），患者能看懂
4. **通俗易懂** — 避免医学术语；如果必须使用，要附带通俗解释

### 核心指标（v14e, rows 36-45 + 验证集 13 行）

| 指标 | 值 | 说明 |
|------|-----|------|
| P0 (幻觉/严重错误) | **0** | Zero hallucination 目标达成 |
| P1 (高优先级问题) | **0** | 从 v9b 的 13 个降至 0（100% elimination） |
| P2 (低优先级/LLM 随机性) | ~6 | 全部为 LLM 随机性波动或 redacted data artifact |
| 平均处理时间 | ~10 min/行 | 13 行验证集共 ~130 min |

### 模型配置

- **Qwen2.5-32B-AWQ** — 4-bit 量化，单 GPU（WSL, CUDA）
- **Llama 3.1 8B Instruct** — bfloat16，单 GPU（baseline 对比）
- KV Cache 分叉：笔记编码一次，多任务复用缓存（`clone_cache()` 防 in-place 修改）
- 贪婪解码 (`do_sample=False`)，`max_new_tokens: 768`

---

## 2. 为什么从 Llama 8B 换到 Qwen 32B

### 2.1 Llama 8B 的问题

2026-03-01 完成了 100 行全量审查（`default_20260301_084320`），结论：**100/100 行全部有问题**，0 行完全正确。

**平均错误率**: ~2.7 错误/行

**系统性错误模式**:

| 错误类型 | 受影响行数(估) | 示例 |
|----------|---------------|------|
| Patient type 误判 | ~35/100 | 首次 Med Onc 就诊标为 "Follow up" |
| response_assessment 答非所问 | ~70/100 | 写手术恢复/Oncotype 分数/未来计划 |
| current_meds 时态混乱 | ~55/100 | 已停/计划中的药物标为当前 |
| Stage 不推断 | ~45/100 | 有肿瘤大小+LN 信息但写 "Not mentioned" |
| Type_of_Cancer 缺受体 | ~65/100 | 遗漏 ER/PR/HER2 任一项 |
| goals_of_treatment 冗长 | ~30/100 | 非标准格式，应为 curative/palliative 等 |

**Row 20-24 具体数据（5 行详审）**:

| Row | 错误数 | 关键问题 |
|-----|--------|---------|
| 20 | 4 | Patient type 错、Stage "T4a" **幻觉**（DCIS 不可能 T4a）、Genetic Testing 混淆过去/未来 |
| 21 | 8 | Patient type 错、**完整 CBC+CMP 全部遗漏**、Stage 不完整、Metastasis 无详情 |
| 22 | 7 | Patient type 错、supportive_meds 列大量非肿瘤药、response 写病理而非响应 |
| 23 | 5 | Stage 不推断、MammaPrint 被归类为 imaging/lab |
| 24 | 5 | Type_of_Cancer 遗漏受体变化、Stage 不完整、Genetic Testing 答非所问 |

**根本原因**: 8B 参数量对多步临床推理能力不足。即使加入 CoT（Chain-of-Thought）和字段拆分，也无法弥补模型容量限制。

### 2.2 Qwen 32B 解决了什么

**Row 20-24 对比表（Llama 8B → Qwen 32B）**:

| Row | Field | Llama 8B (错误) | Qwen 32B (正确) |
|-----|-------|----------------|-----------------|
| 20 | Stage | "T4a" (幻觉) | "" (安全空值) |
| 21 | Patient type | "Follow up" | "New patient" |
| 21 | Stage | "Metastatic" | "Originally Stage II, now metastatic (Stage IV)" |
| 21 | lab_summary | "No labs in note" (**严重遗漏**) | 完整 CBC+CMP 提取 |
| 21 | Metastasis | "Yes" (无细节) | "Yes (to bones, chest wall, infraclavicular, IM nodes)" |
| 22 | Patient type | "Follow up" | "New patient" |
| 22 | supportive_meds | 大量非肿瘤药 | "" (正确为空) |
| 24 | Type_of_Cancer | "ER+/PR+/HER2-" | "Originally ER+/PR+/HER2-, metastatic biopsy ER+/PR-/HER2-" |
| 24 | Stage | "metastatic (Stage IV)" | "Originally Stage IIA, now metastatic (Stage IV)" |
| 24 | Metastasis | "Yes, to multiple sites" | "Yes (to brain, liver, bones, and lymph nodes)" |

**错误率变化**: ~2.7 错误/行 → ~1.2 错误/行（降低 56%）

### 2.3 Qwen 引入的新问题

模型升级并非免费午餐。Qwen 32B 引入了新的错误模式：

| 问题 | 示例 | 后续修复 |
|------|------|---------|
| goals_of_treatment 退化 | Row 21: 8B 正确 "palliative" → 32B "Not yet specified" | v5 G3-REVERT-INFER 修复 |
| response_assessment 退化 | Row 24: 8B 有 "disease progression" → 32B "Not yet on treatment" | v8 prompt + cross_context 修复 |
| A/P 文本串字段 | A/P 原文整段复制到多个 plan 字段 | v3 prompt 全局指令修复 |
| supportive_meds 假阳性 | 讨论中的 bisphosphonate 当成当前用药 | v4 prompt + POST-SUPP 修复 |
| Referral 包含自身科室 | "Rad Onc, Med Onc"（Med Onc 是本次就诊） | v8 prompt 修复 |

这些新问题通过 v3-v14e 共 15+ 个版本的迭代逐步解决。

---

## 3. Pipeline 架构详解

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    输入: 肿瘤科临床笔记                       │
│              (CORAL 数据集, 去标识化, ***** 遮蔽)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  预处理: ***** → [REDACTED]，注入医学术语定义 (formaldef.txt)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  A/P 段提取: Regex 优先，LLM fallback（3 次重试 + LLM 验证）   │
│  Regex 模式: "Assessment / Plan:", "ASSESSMENT & PLAN",       │
│             "Impression/Plan:" 等                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌───────────────────────────┐
│   Phase 1 提取    │    │   Plan 提取 (从 A/P 段)     │
│  6 个独立 prompt  │    │  8 个 prompt (顺序执行)      │
│  (从全文提取)     │    │  Medication_Plan_chatgpt    │
│                  │    │  Medication_Plan            │
│  Reason_for_Visit│    │  Therapy_plan               │
│  Cancer_Diagnosis│    │  radiotherapy_plan           │
│  Lab_Results     │    │  Procedure_Plan             │
│  Clinical_Findings│   │  Imaging_Plan               │
│  Current_Meds    │    │  Lab_Plan                   │
│  Treatment_Changes│   │  Genetic_Testing_Plan       │
└────────┬─────────┘    │  Referral                   │
         │              │  follow_up_next_visit        │
         ▼              │  Advance_care_planning       │
  cross_context 构建     └──────────┬────────────────────┘
  (Cancer_Diagnosis +              │
   Current_Medications +           │
   Clinical_Findings)              │
         │                         │
         ▼                         │
┌──────────────────┐               │
│   Phase 2 提取    │               │
│  2 个 prompt     │               │
│  (注入 cross_ctx)│               │
│                  │               │
│  Treatment_Goals │               │
│  Response_Assess │               │
└────────┬─────────┘               │
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│              5-Gate 验证 (每个 prompt 输出独立验证)             │
│                                                             │
│  G1 FORMAT → G2 SCHEMA → G3 IMPROVE → G4 FAITHFUL → G5 TEMP│
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              21 POST 后处理过滤器 (代码级规则)                 │
│                                                             │
│  POST-KEYS, POST-MEDS, POST-SUPP, POST-PROC                │
│  POST-REFERRAL, POST-GENETICS, POST-OTHERS, POST-NUTRITION  │
│  POST-LAB, POST-PROCEDURE, POST-ADV, POST-STAGE            │
│  POST-GOALS, POST-DISTMET, POST-RESPONSE                    │
│  POST-SPECIALTY, POST-THERAPY, POST-IMAGING                 │
│  POST-THERAPY safety net, POST-IMAGING dedup                 │
│  POST-THERAPY padded matching                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    最终 JSON 输出                              │
│              (keypoints + plan 合并为结构化结果)                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 两阶段提取设计 (v7abc)

**Phase 1 — 6 个独立 prompt（从全文提取）**:

每个 prompt 独立运行，使用相同的 KV Cache（笔记全文编码一次，clone 后复用）：

| Prompt | 提取字段 | 字段数 | 设计意图 |
|--------|---------|--------|---------|
| Reason_for_Visit | Patient type, second opinion, in-person, summary | 4 | 就诊基本信息 |
| Cancer_Diagnosis | Type_of_Cancer, Stage, Distant Metastasis, Metastasis | 4 | 癌症诊断 |
| Lab_Results | lab_summary | 1 | 独立提取防幻觉 |
| Clinical_Findings | findings | 1 | 独立提取防答非所问 |
| Current_Medications | current_meds | 1 | CoT 做时态判断 |
| Treatment_Changes | recent_changes, supportive_meds | 2 | 治疗变化 |

**Cross-context 构建**:

Phase 1 完成后，从结果中提取 ~100-200 tokens 的上下文：
- Cancer_Diagnosis 的 Stage 和 Metastasis 信息
- Current_Medications 的当前用药
- Clinical_Findings 的关键发现

**Phase 2 — 2 个 prompt（注入 cross_context）**:

| Prompt | 提取字段 | 字段数 | 为什么需要上下文 |
|--------|---------|--------|----------------|
| Treatment_Goals | goals_of_treatment | 1 | 需要 Stage 信息判断 curative vs palliative |
| Response_Assessment | response_assessment | 1 | 需要用药+发现信息判断是否在治疗、有无响应 |

### 3.3 5-Gate 验证

| Gate | 名称 | 触发条件 | 关键行为 | 示例 |
|------|------|----------|---------|------|
| G1 | FORMAT | json.loads 失败 | LLM 重格式化 JSON | 修复缺失逗号、未闭合括号 |
| G2 | SCHEMA | 输出 keys 与 prompt schema 无交集 | LLM 修正 key 名 | `"hormonal therapy"` → `"The medication/treatment..."` |
| G3 | IMPROVE | verify=True | 替换模糊词（条件触发）+ 语义对齐检查 | `supportive_meds: "DENOSUMAB"` → 补充遗漏的药物 |
| G4 | FAITHFUL | verify=True | "拿不准就保留"策略，只清空明确矛盾/捏造 | `goals: "palliative"` → 若无原文支持则清空 |
| G5 | TEMPORAL | verify=True 且 key ∈ PLAN_KEYS | 删除过去/已完成项 | `radiotherapy: "XRT in June 2020"` → 过去治疗被过滤 |

**Gate 保护机制**:
- **G3-REVERT-INFER**: 分类值/推断值被 G3 清空后自动恢复（如 "New patient", "palliative"）
- **G4-REVERT**: G4 清空所有字段时自动恢复（全清空说明判断过严）
- **G4-PROTECT**: 安全否定值（"No labs planned."）被 G4 清空后自动恢复
- **G6-PROTECT-CLASS**: 分类字段（Patient type, goals）禁止被 Gate 修改
- **G3-PROTECT-RECEPTOR**: Type_of_Cancer 含受体状态时，G3 清空后自动恢复

### 3.4 21 POST 后处理过滤器

POST 过滤器是确定性代码规则，不依赖 LLM，用于修复 LLM 的系统性错误模式。

**ult.py 内部（4 个，在 `extract_and_verify_v2()` 中执行）**:

| POST | 功能 | 引入版本 |
|------|------|---------|
| POST-KEYS | 删除 schema 之外的多余 key | v2 |
| POST-MEDS | current_meds 用 `oncology_drugs.txt` 白名单过滤 | v2 |
| POST-SUPP | supportive_meds 用 `supportive_care_drugs.txt` 白名单过滤 | v3 |
| POST-PROC | 从 procedure_plan 移除 imaging/radiotherapy/systemic therapy 关键词 | v6.2 |

**run.py 主循环（17 个）**:

| POST | 功能 | 引入版本 | 触发示例 |
|------|------|---------|---------|
| POST-REFERRAL | 全文搜索 referral 模式 | v9b | "Refer to social worker." → Others 补充 |
| POST-GENETICS | 清除 Genetics 中的 mutation finding | v11 | "BRCA 1 mutation" → "None" |
| POST-OTHERS | Others 字段白名单过滤 | v12 | lifestyle 垃圾 → "Social work referral" |
| POST-NUTRITION | 清除饮食建议（非 referral） | v13 | "anti inflammatory diet" → "None" |
| POST-LAB | 移除影像术语 + 过去状态 | v13 | "doppler, labs reviewed" → "No labs planned." |
| POST-PROCEDURE | 全文搜索 procedure 模式 | v11 | HPI 中 "port placement" → 补充到 Procedure_Plan |
| POST-ADV | 全文搜索 code status | v6.2 | "Full code" → 补充到 Advance_care |
| POST-STAGE | Stage vs Metastasis 交叉验证 | v7a | Stage IV + Metastasis=No → 矛盾修正 |
| POST-GOALS | adjuvant + 非转移 → curative | v7a | "adjuvant" + Stage II → "curative" |
| POST-DISTMET | 确保 Distant Metastasis 字段存在 | v7a | 从 Metastasis 复制 |
| POST-RESPONSE | findings 交叉补充 response | v7a | findings 有 "progression" → 补充 response |
| POST-SPECIALTY | 全文搜索 specialty referral | v14 | "Radiation oncology consult" 补充 |
| POST-THERAPY | therapy_plan 白名单过滤 | v14 | 移除非治疗内容（影像、感染等） |
| POST-IMAGING | 全文搜索影像计划 regex | v14 | 补充 echocardiogram/DEXA 等 |
| POST-THERAPY safety net | 无 oncology drug 时清空 | v14a | 全移除后 → "None" |
| POST-IMAGING dedup | IMAGING_SYNONYMS 去重 | v14a | echo + echocardiogram → 保留一个 |
| POST-THERAPY padded | 空格填充防 RT 误匹配 | v14a | "port" 不再匹配 "rt" |

---

## 4. 生成过程详解

以 v14c 验证集 Row 49 (coral_idx 189) 为例，step-by-step 展示完整处理流程。

### Step 1: 输入预处理

原始笔记经过两步预处理：

1. **遮蔽替换**: 所有 `*****` 替换为 `[REDACTED]`，防止 LLM 对遮蔽值编造数字
2. **术语注入**: 从 `formaldef.txt`（9331 个术语）中匹配笔记出现的医学术语，注入通俗定义

```
Injected 3 term definitions: ['mastectomy', 'lumpectomy', 'DCIS']
```

### Step 2: A/P 段 Regex 提取

Pipeline 首先尝试用 regex 从笔记中提取 Assessment/Plan 段落：

```
Trying regex extraction...
Regex matched: 'Assessment / Plan:'
Regex extraction succeeded (1473 chars)
A/P extraction: 0.0s
```

Regex 模式支持多种格式：`Assessment / Plan:`, `ASSESSMENT & PLAN`, `Impression/Plan:` 等。如果 regex 失败，fallback 到 LLM 提取（3 次重试 + LLM 验证）。

### Step 3: Phase 1 — 6 个独立 Prompt 提取

每个 prompt 独立从全文提取，使用 clone 后的 KV Cache：

**示例 — Reason_for_Visit prompt 输出**:

```json
{
  "Patient type": "New patient",
  "second opinion": "yes",
  "in-person": "Televisit",
  "summary": "Metastatic breast cancer (HR+ and HER2 negative) patient
              with progression in the left breast on imaging..."
}
```

**5-Gate 验证日志**:
```
[G1-FORMAT] ok, keys=['Patient type', 'second opinion', 'in-person', 'summary']
[G2-SCHEMA] ok
[G3-IMPROVE] no changes
[G4-FAITH] no changes (all supported)
```

所有 4 个 Gate 都通过（G5 TEMPORAL 仅对 plan 字段运行）。

**示例 — Cancer_Diagnosis 提取**:
```json
{
  "Type_of_Cancer": "ER+/PR+/HER2- invasive ductal carcinoma",
  "Stage_of_Cancer": "Originally Stage IV (T2, N1, M1), now metastatic (Stage IV)",
  "Distant_Metastasis": "Yes, to the lung, lymph nodes, l...",
  "Metastasis": "..."
}
```
```
[POST-KEYS] stripped: {'Distant_Metastasis'}
```
POST-KEYS 删除了多余的 `Distant_Metastasis` key（正确 key 是 `Distant Metastasis`，带空格）。

**Phase 1 总耗时**: 88.1 秒（6 个 prompt 顺序执行）

### Step 4: Phase 2 — 注入 Cross-context

```
Cross-context injected (545 chars)
```

从 Phase 1 结果构建上下文（Cancer_Diagnosis + Current_Medications + Clinical_Findings），注入到 Phase 2 的 2 个 prompt 中。

**Treatment_Goals 提取**:
```json
{"goals_of_treatment": "palliative"}
```
```
[G4-FAITH] goals_of_treatment: "palliative" -> ""
[G4-FAITH] EMPTIED: ['goals_of_treatment']
[G4-REVERT] goals_of_treatment: reverted (G4 emptied all fields)
```

G4 试图清空 "palliative"（认为原文未显式写），但 **G4-REVERT 保护机制**检测到 G4 清空了全部字段，自动恢复。

**Response_Assessment 提取 + G3 IMPROVE**:
```
[G3-IMPROVE] response_assessment: "Imaging from December 2021 shows metastatic
  disease under good control. Biopsies..."
  -> "Imaging from December 2021 shows metastatic disease under good control."
```

G3 IMPROVE 删除了与 response 无关的活检信息，保留了关键的 "disease under good control"。

**Phase 2 总耗时**: 9.9 秒

### Step 5: Plan 提取 — 11 个 Prompt 从 A/P 段提取

Plan 提取使用 A/P 段（非全文），包含 11 个 prompt（每个独立验证）：

```
Medication_Plan_chatgpt: 37.5s [schema-fixed, improved, faith-trimmed, temporal-cleaned]
Medication_Plan: 11.4s [improved]
Therapy_plan: 17.3s
radiotherapy_plan: 7.4s [temporal-cleaned]
Procedure_Plan: 6.3s
Imaging_Plan: 2.0s [faith-trimmed]
Lab_Plan: 1.9s [faith-trimmed]
Genetic_Testing_Plan: 5.1s
Referral: 10.5s [faith-trimmed]
follow_up_next_visit: 3.2s
Advance_care_planning: 1.9s [faith-trimmed]
```

**标签含义**:
- `[schema-fixed]` — G2 SCHEMA 修正了 key 名
- `[improved]` — G3 IMPROVE 改进了内容
- `[faith-trimmed]` — G4 FAITHFUL 修剪了不忠实内容
- `[temporal-cleaned]` — G5 TEMPORAL 过滤了过去/已完成项
- `[stripped]` — POST-KEYS 删除了多余 key

### Step 6: 5-Gate 验证实际日志

**G5 TEMPORAL 过滤过去放疗**:
```
[G5-TEMPORAL] radiotherapy_plan: "Radiation to right pelvis and sternum in 2015.
  None currently planned or discuss..."
  -> "None currently planned or discussed for future."
```
2015 年的放疗被过滤，只保留未来计划。

**G4 FAITHFUL + PROTECT 安全否定值**:
```
[G4-FAITH] imaging_plan: "No imaging planned." -> ""
[G4-FAITH] EMPTIED: ['imaging_plan']
[G4-PROTECT] imaging_plan: restored safe negative "No imaging planned."
```
G4 尝试清空 "No imaging planned."（认为原文未显式写），但 PROTECT 机制识别出这是安全否定值，自动恢复。

### Step 7: POST 后处理

```
[POST-THERAPY] removed non-therapy: ['Her disease is under great control as of
  imaging D', 'She is considering a mastectomy for progression in']
[POST-ADV] patched from full note: ['full code']
[POST-DISTMET] added Distant Metastasis: 'Yes (to the lung, lymph nodes, liver, and bone)'
```

- **POST-THERAPY**: 从 therapy_plan 移除了非治疗内容（疾病状态描述和手术考虑）
- **POST-ADV**: 从全文搜索到 code status "full code"，补充到 Advance_care
- **POST-DISTMET**: 自动从 Metastasis 字段复制到 Distant Metastasis

### Step 8: 最终输出

```
Row 49 total: 203.0s
Row 49 saved to progress
```

总处理时间 ~3.4 分钟。最终输出为包含所有结构化字段的 JSON，保存到结果文件。

---

## 5. 版本演化

### 5.1 完整版本历史

从 Llama 8B baseline 到 Qwen 32B v14e 的完整演化：

```
Llama 8B + V1 (3-gate, grouped prompts)          平均 2.7 错误/行
    ↓ +6-gate, +SCHEMA/FAITH-trim/SPECIFIC/SEMANTIC
Llama 8B + V2 (6-gate, grouped prompts)          平均 ~4 错误/行 (类型不同)
    ↓ +field splitting, +CoT
Llama 8B + V2 (6-gate, split+CoT prompts)        平均 ~1.5 错误/行 (含 bug)
    ↓ +model upgrade
Qwen 32B + V2 (6-gate, split+CoT, v3)            平均 ~1.2 错误/行
    ↓ +gate reorder, +gate protection, +cross_context
Qwen 32B + V2 (5-gate, v7abc)                    平均 ~0.8 错误/行
    ↓ +POST filters, +prompt iteration (v9b-v14e)
Qwen 32B + V2 (5-gate, 21 POST, v14e)            P0=0, P1=0
```

### 5.2 Qwen 32B 版本迭代表（v9b-v14e, rows 36-45）

| 版本 | 核心变更 | P0 | P1 | P2 | 新增 POST | 运行时间 |
|------|---------|----|----|----|----|---------|
| v9b | Baseline（`*****`→`[REDACTED]` + POST-REFERRAL） | 0 | 13 | 8 | REFERRAL | 33.3 min |
| v10 | 5 prompt 修复（HER2 推断, Nutrition 负例, 放疗时态） | 0 | 6 | 7 | — | 32.0 min |
| v11 | SYSTEMIC_THERAPY blocklist + POST-GENETICS/PROCEDURE | 0 | 1 | 9 | +3 | 33.5 min |
| v12 | POST-PROCEDURE regex + POST-OTHERS + Lab_Plan prompt | 0 | 2* | 7 | +1 | 33.2 min |
| v13 | POST-NUTRITION + POST-LAB | 0 | **0** | 6 | +2 | 33.6 min |
| v14 | POST-SPECIALTY + POST-THERAPY + POST-IMAGING | 0 | 2† | ~3 | +3 | — |
| v14a | THERAPY safety net + IMAGING dedup + RT padding | 0 | 0 | — | fix | — |
| v14b | oncology_drugs.txt 扩展 + THERAPY_CATEGORY_TERMS | 0 | 0 | — | fix | — |
| v14c | A/P regex 扩展（Impression/Plan: 模式） | 0 | 0 | — | fix | — |
| v14d | 移除 IMAGING_TYPES 中的 'tte' 防假阳性 | 0 | 0 | — | fix | — |
| v14e | oncology_drugs.txt 加 'zolendronic acid' 拼写变体 | 0 | 0 | — | fix | — |

\* v12 的 P1=0 后经深度审查修正为 P1=2（B87 Lab 回归 + B88 Nutrition 回归）
† v14 36 行实验中发现的问题，在 v14a-e 中修复

**POST 增长**: 5 → 5 → 8 → 9 → 11 → 14 → 17 → 21

### 5.3 P1 消除路径

```
v9b  P1=13  ████████████████████████████████████████████████
v10  P1=6   ████████████████████████
v11  P1=1   ████
v12  P1=2   ████████  (深度审查发现回归)
v13  P1=0   ■ (100% elimination)
```

### 5.4 关键版本详解

**v3-v5: Gate 保护机制建立**

Qwen 32B 引入了 Gate 过度活跃的问题：
- G3 FAITHFUL 清空合理推断（Patient type, Stage, Metastasis）
- G6 SEMANTIC 修改正确值（Patient type "New patient" → "follow up"）

修复手段：
- **G3-REVERT-INFER** (v5): 分类值/推断值被清空后自动恢复
- **G6-PROTECT-CLASS** (v5): 分类字段禁止被修改
- **G3-PROTECT** (v3): 安全否定值被清空后自动恢复

**v7abc: 架构优化 — Gate 合并 + 跨 Prompt 信息传递**

两个重要架构变更：
1. 合并 G3 SPECIFIC + G4 SEMANTIC → G3 IMPROVE（省 18 次 LLM 调用/笔记）
2. 两阶段提取：Phase 1 的 Cancer/Meds/Findings 结果注入 Phase 2 的 Goals/Response

效果：
- goals_of_treatment 7/10 行修正（adjuvant → curative）
- Response_Assessment 5/10 行从空变有
- 速度：31.3min vs 33.2min（快 5.7%）

**v9b-v13: POST 后处理系统成熟**

核心发现：LLM 的系统性错误可以通过确定性代码规则高效修复。

典型案例：
- B68 (Procedure 含化疗): Prompt 已有规则但模型忽略 → `SYSTEMIC_THERAPY_TERMS` blocklist 在 POST-PROC 中过滤
- B70 (Genetics 放诊断结果): "BRCA 1 mutation" 是发现不是转诊 → POST-GENETICS 检测 mutation/carrier 但不含 refer/consult → 清空
- B82 (Others 垃圾文本): lifestyle advice 混入 Others → POST-OTHERS 白名单过滤，只保留已知 referral 类型

**v14: 大规模验证 + 精细化**

扩展到 36 行实验（rows 47-82）后发现新问题：
- therapy_plan 混入非治疗内容（影像、感染药物） → POST-THERAPY 白名单过滤（42% 触发率）
- Imaging_Plan 遗漏 → POST-IMAGING 全文搜索补充
- A/P 段 regex 遗漏 "Impression/Plan:" 格式 → regex 扩展

v14a-e 修复了 5 个 edge case bug。

---

## 6. v14 系列实验结果

### 6.1 36 行实验 (rows 47-82)

| 指标 | 值 |
|------|-----|
| 总时间 | 130.4 min |
| P0 | 0 |
| P1 | 2 |
| P2 | ~3 |
| POST-THERAPY 触发率 | 42% (15/36) |
| POST-IMAGING 触发率 | 6% (2/36) |

**发现的 bug + 修复**:

| 版本 | Bug | 修复 |
|------|-----|------|
| v14a | therapy_plan 全移除后应为 "None" 而非空 | POST-THERAPY safety net |
| v14a | echo 和 echocardiogram 重复出现 | IMAGING_SYNONYMS 同义词去重 |
| v14a | "port" 中的 "rt" 触发 POST-THERAPY 假阳性 | 空格填充匹配 |
| v14b | zoledronate 拼写不在白名单 | oncology_drugs.txt 扩展 |
| v14b | 化疗方案缩写 (TC, AC, EC, CMF) 未被识别 | THERAPY_CATEGORY_TERMS 扩展 |
| v14c | Row 49 A/P 段以 "Impression/Plan:" 开头，regex 未匹配 | A/P regex 新增模式 |
| v14d | 'tte' 在 IMAGING_TYPES 中导致 "committee" 等词假阳性 | 移除 'tte' |
| v14e | 'zolendronic acid'（常见拼写错误）不在白名单 | 加入 oncology_drugs.txt |

### 6.2 13 行验证实验 (rows 49,78,79,83-92)

v14c-e 在 13 行验证集上的结果，包含 3 行之前出错的 Row 和 10 行新 Row。

**Error Row 修复验证**:

| Row | 之前问题 | v14c 结果 | 修复? |
|-----|---------|-----------|------|
| 49 | A/P regex 未匹配 "Impression/Plan:" | Regex 成功匹配 | ✅ |
| 78 | echocardiogram 重复 | IMAGING_SYNONYMS 去重生效 | ✅ |
| 79 | therapy_plan 全移除后为空 | safety net 清为 "None" | ✅ |

**验证集 Gate 行为统计**:

从 v14c 运行日志中可以看到：

```
Row 49 — 总耗时 203.0s
  Phase 1: 88.1s (6 prompts)
  Phase 2: 9.9s (2 prompts, cross-context 545 chars)
  Plan: 105.0s (11 prompts)

Row 78 — 总耗时 246.7s  (笔记较长)
  Phase 1: 112.3s
  Phase 2: 9.4s (cross-context 710 chars)
  Plan: 93.2s
```

**各 Gate 触发统计（Row 49 示例）**:

| Gate | 触发 | 行为 |
|------|------|------|
| G1 FORMAT | 0/17 prompt 触发 | 所有 JSON 解析成功 |
| G2 SCHEMA | 1/17 | Medication_Plan_chatgpt key 修正 |
| G3 IMPROVE | 3/17 | supportive_meds 补充, response 精简, Medication_Plan 具体化 |
| G4 FAITHFUL | 5/17 | findings 微调, goals 清空后恢复, 多个 safe negative 保护 |
| G5 TEMPORAL | 2/17 | radiotherapy 过去治疗过滤, Medication_Plan_chatgpt 时态清理 |

**POST 触发统计（Row 49）**:
```
[POST-THERAPY] removed non-therapy: 2 sentences
[POST-ADV] patched from full note: 'full code'
[POST-DISTMET] added Distant Metastasis
```

---

## 7. 已知问题与未来方向

### 7.1 可代码修复的残留问题 — 全部已修复

截至 v14e，所有可通过代码修复的问题均已解决：

| 问题 | 修复版本 | 修复方式 |
|------|---------|---------|
| 'tte' 在 procedure 过滤中假阳性 | v14d | 从 IMAGING_TYPES 移除 'tte' |
| 'zolendronic acid' 不在白名单 | v14e | 加入 oncology_drugs.txt |
| A/P regex 遗漏 "Impression/Plan:" | v14c | regex 新增模式 |
| echo/echocardiogram 重复 | v14a | IMAGING_SYNONYMS 去重 |
| therapy_plan 全移除后为空 | v14a | safety net 清为 "None" |

### 7.2 LLM 固有问题（P2 级别）

这些问题来自 LLM 随机性或 redacted data，无法通过代码完全消除：

| 问题模式 | 触发率(估) | 示例 | 根因 |
|---------|-----------|------|------|
| 时态混淆 | ~15% | 已停药标为当前 | LLM 对 "recently dropped" 等边界情况判断不稳定 |
| HER2 推断不完整 | ~10% | `[REDACTED]/neu negative` 未推断 HER2- | 非 triple negative 场景，模型缺少推断信号 |
| response_assessment 答非所问 | ~10% | 写手术计划而非治疗响应 | Phase 2 cross_context 已大幅改善 |
| Stage 推断遗漏 | ~10% | 有肿瘤大小+LN 信息但未推断 | LLM 随机性（同一行不同运行可能正确/错误） |
| redacted data artifact | ~5% | `*****` 遮蔽关键信息导致提取不完整 | 数据问题，非模型问题 |

### 7.3 未来方向

**近期（可执行）**:

1. **更大规模验证** — 当前测试 <100 行，总数据集 240 行。需要全量运行评估 P0/P1 在新数据上的表现
2. **细化 prompt 针对残留 P2** — 特别是 HER2 推断（非 triple negative 场景）和 Stage 推断
3. **POST 过滤器性能优化** — 当前 21 个 POST 顺序执行，可并行化

**中期（核心目标）**:

4. **通俗化解释生成** — 四个原则中的第 3 和第 4 条尚未实现。当前输出仍有大量医学术语，需要在提取后增加解释生成步骤，将结构化 JSON 转化为 8 年级英语水平的患者信

**远期（探索）**:

5. **多模型对比** — 测试 Qwen2.5-72B 或更新模型是否能进一步降低 P2
6. **主动学习** — 基于审查反馈自动更新 prompt 或 POST 规则
7. **实时部署** — 从批处理转向实时处理，支持临床工作流

---

## 附录

### A. 实验日志

| Run ID | Model | Config | Rows | Date | Notes |
|--------|-------|--------|------|------|-------|
| `default_v1_20260228_*` | Llama 3.1 8B | V1, 3-gate | 0-14 | 02-28 | V1 baseline |
| `default_20260301_084320` | Llama 3.1 8B | V2, 6-gate, old prompts | 0-99 | 03-01 | 100-row full run |
| `default_20260301_161703` | Llama 3.1 8B | V2, 6-gate, split+CoT | 20-39 | 03-01 | Prompt refactor |
| `default_qwen_20260313_220920` | Qwen2.5-32B-AWQ | V2, 6-gate, v2 prompts | 20-24 | 03-13 | Model upgrade |
| `default_qwen_20260314_*` | Qwen2.5-32B-AWQ | V2, v3-v7abc | 20-34 | 03-14 | Gate 优化迭代 |
| `default_qwen_20260315_095522` | Qwen2.5-32B-AWQ | V2, v9b | 36-45 | 03-15 | v9b batch2 |
| `default_qwen_20260315_105314` | Qwen2.5-32B-AWQ | V2, v10 | 36-45 | 03-15 | 6 P1 fixes |
| `default_qwen_20260315_114946` | Qwen2.5-32B-AWQ | V2, v11 | 36-45 | 03-15 | 3 POST fixes |
| `default_qwen_20260315_123551` | Qwen2.5-32B-AWQ | V2, v12 | 36-45 | 03-15 | 3 more POST |
| `default_qwen_20260315_132157` | Qwen2.5-32B-AWQ | V2, v13 | 36-45 | 03-15 | POST-LAB/NUTRITION |
| v14 系列 | Qwen2.5-32B-AWQ | V2, v14-v14e | 47-92 | 03-15 | 36行+13行验证 |

### B. Bug 追踪汇总

**已修复的关键 Bug**:

| Bug | 描述 | 修复版本 | 修复方式 |
|-----|------|---------|---------|
| B7 | HER2 受体写反（HER2 3+ → HER2-） | v8 prompt | HER2 靶向药推断规则 |
| B9 | Genetic_Testing 字段串行 | v3 prompt | 独立默认值 |
| B10 | current_meds CoT 过严遗漏 | v8 prompt | 放宽判断标准 |
| B13 | goals 退化（Qwen 32B 特有） | v5 | G3-REVERT-INFER |
| B14 | response 提取不稳定 | v7abc | Cross-context 注入 |
| B24 | Patient type 被 G6 修改 | v5 | G6-PROTECT-CLASS |
| B43 | Lab 值幻觉（遮蔽值编造数字） | v8 prompt | 禁止猜测 [REDACTED] |
| B49 | Stage 幻觉（非转移标为 Stage IV） | v7a | POST-STAGE 交叉验证 |
| B68 | Procedure 含化疗 | v11 | SYSTEMIC_THERAPY_TERMS blocklist |
| B87 | Lab_Plan 含影像 | v13 | POST-LAB 后处理 |
| B88 | Nutrition 含饮食建议 | v13 | POST-NUTRITION 后处理 |

### C. 文件参考

| 文件 | 用途 |
|------|------|
| `run.py` | 实验入口，主循环 + 17 个 POST 过滤器 |
| `ult.py` | 工具库：模型推理、5-Gate 验证、4 个 POST 过滤器 |
| `prompts/extraction.yaml` | Phase 1+2 的 8 个提取 prompt |
| `prompts/plan_extraction.yaml` | Plan 提取的 11 个 prompt |
| `data/oncology_drugs.txt` | 肿瘤药物白名单（96 药） |
| `data/supportive_care_drugs.txt` | 支持治疗药物白名单（78 药） |
| `data/formaldef.txt` | 医学术语通俗定义（9331 术语） |
| `exp/default_qwen.yaml` | Qwen 实验配置 |
| `results/tracking.md` | 版本历史 + 逐行审查追踪 |
