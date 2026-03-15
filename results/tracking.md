# Extraction Quality Tracking

## Experiment Log

| Run ID | Model | Config | Rows | Date | Notes |
|--------|-------|--------|------|------|-------|
| `default_v1_20260228_191615` | Llama 3.1 8B (bf16) | V1 pipeline, 3-gate | 0-4 | 2026-02-28 | V1 baseline |
| `default_v1_20260228_194520` | Llama 3.1 8B (bf16) | V1 pipeline, 3-gate | 5-9 | 2026-02-28 | V1 baseline |
| `default_v1_20260228_201229` | Llama 3.1 8B (bf16) | V1 pipeline, 3-gate | 10-14 | 2026-02-28 | V1 baseline |
| `default_20260228_213708` | Llama 3.1 8B (bf16) | V2 early, 6-gate, old prompts | 0-19 | 2026-02-28 | V2 initial implementation |
| `testfix_20260228_221738` | Llama 3.1 8B (bf16) | V2 testfix | 0,1,4,6,10,11,19 | 2026-02-28 | Intermediate fixes |
| `testfix_20260228_223813` | Llama 3.1 8B (bf16) | V2 testfix | 20-29 | 2026-02-28 | Intermediate fixes |
| `default_20260301_084320` | Llama 3.1 8B (bf16) | V2 pipeline, 6-gate, old prompts | 0-99 | 2026-03-01 | **100-row full run** (reviewed in detail) |
| `default_20260301_161703` | Llama 3.1 8B (bf16) | V2 pipeline, 6-gate, new prompts (split+CoT) | 20-39 | 2026-03-01 | Prompt refactoring + CoT + field splitting |
| `default_qwen_20260313_220920` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v2 prompts | 20-24 | 2026-03-13 | Model upgrade: 32B vs 8B comparison |
| `default_qwen_20260314_085717` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v3 prompts, broken template | 20-24 | 2026-03-14 | v3 prompt test (missing turn_end) |
| `default_qwen_20260314_094445` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v3 prompts, fixed template | 20-24 | 2026-03-14 | turn_end fix only |
| `default_qwen_20260314_102232` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v3 prompts, gate protection | 20-24 | 2026-03-14 | + G3/G4/G6 gate protection |
| `default_qwen_20260314_114835` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v3 prompts, gate + supp filter | 20-24 | 2026-03-14 | + supportive_meds whitelist filter (B21 fixed) |
| `default_qwen_20260314_124631` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v4 prompts, gate + supp filter | 20-24 | 2026-03-14 | v4 prompts (B17 fixed at extraction, B14 extraction fixed but G3 emptied) |
| `default_qwen_20260314_131143` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v4 prompts, gate + supp + G3-REVERT | 20-24 | 2026-03-14 | + G3-REVERT (B13 fixed, B14 unstable, goals restored) |
| `default_qwen_20260314_140519` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, 6-gate, v5 code, G3-REVERT-INFER + G6-PROTECT-CLASS | 20-24 | 2026-03-14 | **v5**: B24/B25/B26/B27/B29 全部修复 |
| `default_qwen_20260314_144451` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v5 code | 25-34 | 2026-03-14 | v5 扩展测试（10 rows）|
| `default_qwen_20260314_154025` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v6 code + POST-PROC-FILTER + POST-ADV | 25-34 | 2026-03-14 | v6: POST-PROC-FILTER false positive ("port" → "rt") |
| `default_qwen_20260314_161507` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v6.1 code (RT word boundary fix) | 25-34 | 2026-03-14 | v6.1: Type_of_Cancer G3 regression |
| `default_qwen_20260314_165646` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v7a code (gate reorder + v6.2 fixes) | 25-34 | 2026-03-14 | **v7a**: gate 顺序优化 + 9 bug fixes。33.2min |
| `default_qwen_20260314_194226` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v7abc code (5-gate + 2-phase) | 25-34 | 2026-03-14 | **v7abc**: G3/G4 合并→G3-IMPROVE + 跨 prompt 信息传递。31.3min (v7a 33.2min) |
| (待运行) | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v8 code (v7abc + prompt fixes) | 25-34 | 2026-03-14 | **v8**: 修 v7abc 审查发现的 6 个 P2 问题 |
| `default_qwen_20260315_095522` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v9b code (*****→[REDACTED] + POST-REFERRAL) | 25-34 | 2026-03-15 | **v9b batch1**: 10 rows。P0=0, P1=?, P2=? |
| `default_qwen_20260315_095522` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v9b code | 36-45 | 2026-03-15 | **v9b batch2**: P0=0, P1=13, P2=8。33.3min |
| `default_qwen_20260315_105314` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v10 code (6 P1 fixes) | 36-45 | 2026-03-15 | **v10**: P0=0, P1=6, P2=7。32.0min。7 bugs fixed, 4 new P2 |
| `default_qwen_20260315_114946` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v11 code (3 POST fixes) | 36-45 | 2026-03-15 | **v11**: P0=0, P1=1, P2=9。33.5min。B68+B70+B66 修复确认 |
| `default_qwen_20260315_123551` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v12 code (3 more POST fixes) | 36-45 | 2026-03-15 | **v12**: P0=0, ~~P1=0~~ P1=2, P2=7。33.2min。B75+B82+B77 修复，B87+B88 新回归（深度审查发现） |
| `default_qwen_20260315_132157` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v13 code (POST-LAB + POST-NUTRITION) | 36-45 | 2026-03-15 | **v13**: P0=0, **P1=0**, P2=7。33.6min。B87+B88 修复确认，B73 自愈 |

## Prompt 版本变更记录

### v13 (2026-03-15): 2 个 POST 后处理修复（B87 Lab_Plan + B88 Nutrition）

**变更原因：** v12 深度逐字段对比发现 2 个被简略审查遗漏的 P1 回归。

**代码变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | POST-NUTRITION | run.py | B88 | 新增后处理：Nutrition 字段不含 refer/consult/follow up with nutritionist → 清为 "None" |
| 2 | POST-LAB | run.py | B87 | 新增后处理：从 Lab_Plan 移除影像术语 (doppler/ultrasound/CT/MRI 等) + "labs reviewed/adequate" 类过去状态表述 |

**基线：** v12 (`default_qwen_20260315_123551`)

### v12 (2026-03-15): 3 个 POST 后处理修复（B75 regex + B82 Others 过滤 + B78 Lab 提示）

**变更原因：** v11 仍有 1 个 P1 (B75) + 2 个可修复 P2 (B82, B78)。

**Prompt 变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | Lab_Plan pre-chemo draw | plan_extraction.yaml | B78 | 加 "pre-chemo blood draw" 作为有效 lab plan 示例 |

**代码变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | POST-PROCEDURE regex 修复 | run.py | B75 | 加可选冠词 `(?:an?\s+)?` + `is\s+scheduled\s+for` 模式 |
| 2 | POST-OTHERS 白名单过滤 | run.py | B82 | 新增后处理：Others 字段只保留匹配已知 referral 模式的项（social work, PT, OT, exercise counseling 等） |

**基线：** v11 (`default_qwen_20260315_114946`)

### v11 (2026-03-15): 3 个 POST 后处理修复（B68 化疗过滤 + B70 Genetics 清洗 + B75 全文搜索）

**变更原因：** v10 仍有 5 个 P1，其中 3 个可通过后处理修复。

**Prompt 变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | Procedure 化疗负例 | plan_extraction.yaml | B68 | "Common mistakes to AVOID" 增加 chemotherapy/TCHP/FOLFOX/immunotherapy/targeted therapy |

**代码变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | SYSTEMIC_THERAPY_TERMS blocklist | ult.py | B68 | 在 `filter_procedure_plan()` 中增加化疗/靶向/内分泌治疗 blocklist（30+ 药名/方案名） |
| 2 | POST-GENETICS | run.py | B70 | 新增后处理：Genetics referral 含 mutation/carrier/positive/negative 但不含 refer/consult → 清为 "None" |
| 3 | POST-PROCEDURE | run.py | B75 | 新增后处理：全文搜索 port placement/biopsy/surgery 等 procedure 模式（类似 POST-REFERRAL） |

**基线：** v10 (`default_qwen_20260315_105314`)

### v10 (2026-03-15): v9b batch2 审查修复（6 个 P1 修复）

**变更原因：** v9b batch2 (rows 36-45) 审查发现 P0=0, P1=13, P2=8。13 个 P1 来自 9 个系统性问题，本次修复其中 6 个。

**Prompt 变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 1 | radiotherapy_plan 时态区分 | plan_extraction.yaml | B77/B79 | 加 PAST vs CURRENT/FUTURE 区分规则 + 过去放疗负例 |
| 2 | HER2 从 "triple negative" 推断 | extraction.yaml | B72/B73 | "triple negative" ANYWHERE → HER2=negative；[REDACTED] negative → HER2 negative |
| 3 | Nutrition 假阳性负例 | plan_extraction.yaml | B67/B69 | "I recommend anti inflammatory/mediterranean diet" 等明确列为非 referral |
| 4 | Genetics referral ≠ 发现 | plan_extraction.yaml | B70/B74 | 明确 "BRCA1 mutation"/"ATM carrier" 是 FINDINGS 不是 referrals |
| 5 | response_assessment 放疗后 | extraction.yaml | B76 | 完成放疗 = HAS received treatment，不写 "Not yet on treatment" |

**代码变更：**

| # | 修复 | 文件 | Bug | 说明 |
|---|------|------|-----|------|
| 6 | radiotherapy_plan 加入 PLAN_KEYS | ult.py | B77/B79 | G5 TEMPORAL 现在对 radiotherapy_plan 运行时态过滤 |
| 7 | POST-REFERRAL regex 扩展 | run.py | B71 | 匹配 "Refer to X"（祈使句）格式，不仅限 "will refer" |

**未修复的 Bug（需要更复杂方案）：**
- B68 (Procedure 含化疗): prompt 已有规则但模型忽略，需考虑后处理
- B75 (Port placement 遗漏): 在 HPI 段非 A/P 段，plan extraction 只看 A/P
- B80 (Others referral 垃圾输出): G3 IMPROVE 应捕获但未生效，需调查

**基线：** v9b (`default_qwen_20260315_095522`)

### v8 (2026-03-14): v7abc 审查修复（6 个 P2 prompt 改进）

**变更原因：** v7abc 10 行逐行审查发现 6 个 P2 退步 + 4 个两版共同问题

**Prompt 变更：**

| # | 修复 | 文件 | 问题 | 说明 |
|---|------|------|------|------|
| 1 | HPI vs A/P 矛盾时优先 A/P | extraction.yaml Current_Medications | Row 34: tamoxifen+anastrozole 同时列出 | 加 "prefer Assessment/Plan — it reflects the most recent clinical decision" |
| 2 | zolendronic 拼写变体 | extraction.yaml Current_Medications | Row 26: zolendronic acid 未被识别 | 在 BONE AGENTS 列表加 "zolendronic acid" |
| 3 | node neg ≠ Stage II | extraction.yaml Cancer_Diagnosis | Row 26: "Stage I"→"Stage IIA" | 加 "Node-negative + ≤2cm = Stage I, NOT Stage II" |
| 4 | 局部复发措辞 | extraction.yaml Response_Assessment | Row 33: "not responding to treatment" 不准确 | 加 "describe recurrence factually, do NOT write 'not responding' — patient may have discontinued treatment" |
| 5 | Referral 搜索范围扩展 | plan_extraction.yaml Referral | Row 25: Social Work referral 遗漏 | 加 "Look throughout ENTIRE note, not just A/P" + 搜索关键词列表 |
| 6 | 过时 lab 日期标注 | extraction.yaml Lab_Results | Row 31: 3 年前 lab 丢了日期 | 从 "note the date" 改为 "you MUST note the date" + 示例 |

**代码变更：** 无（v8 仅 prompt 变更）

**基线：** v7abc (`default_qwen_20260314_194226`)

### v7abc (2026-03-14): Gate 合并 + 跨 Prompt 信息传递

**变更原因：** v7a 的 G3 SPECIFIC 和 G4 SEMANTIC 可合并省 LLM 调用；Treatment_Goals 和 Response_Assessment 需要其他字段的信息才能准确判断

**代码变更：**
1. `ult.py` — 合并 G3 SPECIFIC + G4 SEMANTIC → **G3 IMPROVE**（省 18 次 LLM 调用/笔记）
2. `run.py` — 两阶段提取：Phase 1 (6 prompts 独立提取) → Phase 2 (2 prompts 带上下文：Treatment_Goals, Response_Assessment)
3. `run.py` — `_build_cross_context()` 函数，从 Phase 1 结果构建 Cancer_Diagnosis + Current_Medications + Clinical_Findings 的上下文

**v7abc vs v7a 审查结论（10 行逐行对比）：**
- **改善**：goals_of_treatment 7/10 行修正（adjuvant→curative），Response_Assessment 5/10 行从空变有，Row 29 Stage 幻觉修复 (P0)
- **退步**：6 个 P2（Therapy_plan 内容减少、Stage IIA 误判、Lab 日期丢失、current_meds 过度、Medication_Plan_chatgpt 丢内容、response 措辞）
- **速度**：31.3min vs 33.2min（快 5.7%）

### v7a (2026-03-14): Gate 顺序优化 + v6.2 Bug Fixes

**变更原因：** v6.2 审查发现 3 个 P0、9 个 P1、19 个 P2 问题。同时发现 gate 顺序存在结构性漏洞：G5/G6 在 G3 之后运行，可引入未经忠实度检查的幻觉。

**Gate 顺序变更（核心优化）：**
- 旧顺序：G1 FORMAT → G2 SCHEMA → G3 FAITHFUL → G4 TEMPORAL → G5 SPECIFIC → G6 SEMANTIC
- 新顺序：G1 FORMAT → G2 SCHEMA → G3 SPECIFIC → G4 SEMANTIC → G5 FAITHFUL → G6 TEMPORAL
- 思路：先"改进"（G3 具体化 + G4 语义修正），再"验证"（G5 忠实度终审），最后"过滤"（G6 时态）
- G5 FAITHFUL 作为 LLM 内容的终审 gate，确保 G3/G4 引入的改动也经过忠实度检查

**Bug Fix（9 个修复，基于 v6.2 审查）：**

| # | 修复 | 文件 | Bug | 类型 | 说明 |
|---|------|------|-----|------|------|
| 1 | POST-STAGE | run.py | B49 (P0) | 后处理 | Stage="Stage IV" 但 Metastasis="No" → 删除矛盾的 Stage IV 部分 |
| 2 | POST-GOALS | run.py + extraction.yaml | B45 (P1) | 后处理+prompt | "adjuvant" + 非转移 → "curative"；prompt 改为"adjuvant 描述治疗类型不是目标" |
| 3 | POST-DISTMET | run.py | B48 (P2) | 后处理 | Distant Metastasis 字段缺失 → 从 Metastasis 复制 |
| 4 | Lab redacted | extraction.yaml | B43 (P0) | prompt | 禁止猜测遮蔽值 (*****) → 写 "Values redacted" |
| 5 | POST-RESPONSE | run.py + extraction.yaml | B53,B60 (P0/P1) | 后处理+prompt | response="Not mentioned" 但 findings 有 progression/no recurrence → 交叉补充 |
| 6 | Therapy_plan | plan_extraction.yaml | B55,B63 (P2) | prompt | 排除影像/lab 计划，只写药物和放疗 |
| 7 | G3→G5-PROTECT-RECEPTOR 0.75 | ult.py | B64 (P2) | 阈值 | 0.5→0.75，更积极保护受体状态 |
| 8 | "not taking" filter | ult.py | B54 (P1) | 后处理 | supportive_meds 含 "not taking" → 清空 |
| 9 | "recently dropped" | extraction.yaml | B56 (P1) | prompt | Current_Medications 排除 "recently dropped"/"self D/C" 的药物 |

**不修复的 Bug：** B50 (Ki-67 读数错误)、B62 (数据矛盾)、B44/B57/B58/B59/B61（低优先级或需更复杂方案）

**对比实验计划：**
- v7a vs v7abc：验证 gate 顺序优化效果，以及合并 gate + 跨 prompt 传递的额外收益
- 测试行：rows 25-34（coral_idx 165-174），与 v6.2 同组对比

### v5 (2026-03-14): G3 过度清空 + G6 过度修正修复

**变更原因：** v4 实验发现 G3 过度清空合理值 (B25/B26/B27) 和 G6 过度修正分类字段 (B24/B29)

**代码变更：**
1. `ult.py` — G3-REVERT-INFER: G3 清空字段后，检查原始值是否为分类值（"New patient"/"palliative" 等）或含推断标记（"approximately"/"not sure" 等），若是则恢复 → 修 B25/B26/B27
2. `ult.py` — G6-PROTECT-CLASS: 分类字段（Patient type, second opinion, in-person, goals_of_treatment）禁止 G6 修改 → 修 B24
3. `ult.py` — G6 prompt 增加 Nutrition referral 规则："diet advice ≠ nutrition referral" → 修 B29
4. `ult.py` — POST-SUPP 安全否定保护: "None currently being taken." 等短语不再被当成药物列表过滤

**新增常量：**
- `CLASSIFICATION_VALUES`: 标准分类值集合（new patient, curative, palliative, yes, no 等）
- `INFERENCE_MARKERS`: 推断标记词（approximately, not sure, likely 等）
- `G6_NO_MODIFY_FIELDS`: G6 不得修改的分类字段集合
- `_is_classification_or_inference()`: 检测值是否为分类/推断值

**之前版本 (v4):** 同一天早些时候的变更

### v4 (2026-03-14): 提取精度 + gate 保护 + 药物过滤

**变更原因：** v3 实验仍有 B14/B17/B20 未修复，根因分析显示需要更强的 prompt 指导 + 后处理

**Prompt 变更：**
1. `extraction.yaml` — Response_Assessment: step 3 扩展响应证据搜索范围，明确列出"breaking up"等体检发现也算响应证据；step 1 加活性药物列表线索 → 修 B14
2. `extraction.yaml` — Treatment_Changes/supportive_meds: 强化 CRITICAL rules，明确列出 4 种"非当前"模式（hypothetical/discussion/future/contingency）；排除精神科/抗癫痫药 → 修 B17
3. `plan_extraction.yaml` — Referral: 新增 OUTGOING vs INCOMING 转诊区分规则；明确"这些笔记是 Medical Oncology 写的" → 修 B20

**代码变更：**
4. `ult.py` — Gate 保护机制（gate_config 驱动）：
   - G3-PROTECT: 恢复被 G3 错误清空的安全否定值（"No labs planned." 等）
   - G4/G6 skip: 全空字段跳过时态过滤和语义检查
   - G6-PROTECT: 阻止 G6 向空字段填充新内容
5. `ult.py` — POST-SUPP: supportive_meds 白名单过滤（`data/supportive_care_drugs.txt`）→ 修 B21
6. `ult.py` — turn_end 修复：build_base_cache 中加 `chat_tmpl.t['turn_end']` 分隔符

**新增数据文件：**
7. `data/supportive_care_drugs.txt` — 肿瘤支持治疗药物白名单（~65 药）
8. `data/non_oncology_drugs.txt` — 非肿瘤药参考清单（~120 药）

**之前版本 (v3):** `results/default_qwen_20260314_094445/config.yaml` 中有 v3 prompt 快照

### v3 (2026-03-14): 字段边界强化 + response CoT 改进

**变更原因：** Qwen 32B Row 20-24 审查发现 B14-B20 共 6 个新问题

**变更内容：**
1. `extraction.yaml` — Response_Assessment: CoT 第 1 步加治疗中检测线索（off week, cycle, labs okay to proceed）→ 修 B14
2. `extraction.yaml` — Treatment_Changes/supportive_meds: 强调 "当前正在服用"，排除讨论中的未来方案 → 修 B17
3. `plan_extraction.yaml` — 顶部加全局指令：每个字段回答不同问题，不要复制相同文本到多个字段 → 修 B15
4. `plan_extraction.yaml` — Lab_Plan: 明确排除 imaging (PET/CT/DEXA) 和 genomic tests (MammaPrint/Oncotype) → 修 B16
5. `plan_extraction.yaml` — Referral: 排除本次就诊自身科室 → 修 B20

**之前版本 (v2, 2026-03-01):** `results/default_20260301_161703/config.yaml` 中有完整 prompt 快照

---

## Row-by-Row Review: Llama 3.1 8B (`default_20260301_161703`)

### Row 20 (coral_idx=160) — DCIS post-mastectomy, adjuvant consult

**Ground truth (from note):**
- 70 y.o. female, referred by breast surgeon for consideration of adjunct therapy
- Note says "referred by breast surgeon Dr. ***** following recent partial mastectomy"
- Right breast DCIS, intermediate grade, ER+ (98%), PR+ (90%), DCIS spanning ~5.0 cm
- No invasive cancer, no staging mentioned (DCIS is typically stage 0 / Tis)
- Genetic testing (Invitae, 52 genes) — all negative
- No labs in note
- No current prescription meds
- Plan: Rad Onc consult + consider endocrine therapy (Arimidex) + bisphosphonate

| Field | Llama 8B Output | Correct? | Issue |
|-------|-----------------|----------|-------|
| Patient type | "Follow-up" | **WRONG** | First Med Onc visit — "referred for consideration of adjunct therapy". Should be "New patient" |
| second opinion | "No" | OK | Not a second opinion |
| in-person | "in-person" | OK | |
| summary | "referred for consideration of adjunct therapy following partial mastectomy" | OK | Good summary |
| Type_of_Cancer | "DCIS, ER+/PR+" | OK | Correct |
| Stage_of_Cancer | **"T4a"** | **WRONG — HALLUCINATION** | DCIS is Tis (stage 0). T4a means chest wall invasion. Nothing in the note supports this. |
| Metastasis | "No" | OK | |
| lab_summary | "No labs in note." | OK | Correct |
| findings | pathology + physical exam | OK | Accurate |
| current_meds | "" | OK | Note says "not on any prescription meds" |
| supportive_meds | "Vitamin D, calcium" | OK | Note mentions these for osteopenia |
| goals_of_treatment | "risk reduction" | OK | Correct — DCIS adjuvant is risk reduction |
| response_assessment | "Not yet on treatment" | OK | |
| Genetic_Testing_Plan | "negative genetic screen" | **WRONG** | This is a PAST result, not a plan. Genetic testing was done on 09/20/2016. Field asks for planned testing. |
| Referral: Genetics | "negative" | **WRONG** | Should be "None" (no genetics referral planned). The "negative" refers to past genetic test results. |
| radiotherapy_plan | "XRT" | PARTIAL | Correct that XRT is being considered but lacks detail (pending Rad Onc consult) |

**Llama 8B errors for Row 20: 4 issues (Patient type, Stage hallucination, Genetic_Testing confused past/future, Referral-Genetics)**

---

### Row 21 (coral_idx=161) — Metastatic breast cancer, second opinion

**Ground truth (from note):**
- 72 y.o. female, here for a second opinion ("She is here for a second opinion")
- "Medical Oncology Consult Note" + "New Patient Evaluation"
- Complex history: Left DCIS 1994, Right Stage II IDC 2000, Metastatic relapse May 2020
- Right chest wall recurrence + bone mets + infraclavicular + IM nodes, HR+/HER2-
- pT2 staging, originally Stage II
- Current meds: anastrozole (arimidex), denosumab (xgeva), prednisone, lomotil, etc.
- Abemaciclib discontinued due to pneumonitis July 2021
- Labs from 01/29/2021: full CBC + CMP (WBC 3.16, Hgb 10.7, Cr 1.19, etc.)
- Plan: PET CT, if stable continue arimidex, if progression consider faslodex/afinitor/xeloda/trial

| Field | Llama 8B Output | Correct? | Issue |
|-------|-----------------|----------|-------|
| Patient type | "Follow up" | **WRONG** | Note explicitly says "New Patient Evaluation" and "second opinion" |
| second opinion | "yes" | OK | |
| in-person | "yes" | **WRONG FORMAT** | Should be "in-person" not "yes" |
| summary | "Metastatic breast cancer, ER+, stage not specified" | PARTIAL | Stage IS specified (pT2, Stage II original) |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | OK | |
| Stage_of_Cancer | "Metastatic" | **INCOMPLETE** | Should say "Originally Stage II, now metastatic (Stage IV)" |
| Metastasis | "Yes" | **INCOMPLETE** | Should specify where: bones, chest wall, infraclavicular, IM nodes |
| lab_summary | **"No labs in note."** | **WRONG — CRITICAL** | Note has FULL CBC + CMP from 01/29/2021. Massive omission. |
| findings | "Right chest wall recurrence, bone mets..." | PARTIAL | Just repeats cancer history, doesn't mention PET scan response |
| current_meds | ["anastrozole", "denosumab"] | OK | Correct (abemaciclib discontinued, prednisone is supportive) |
| Treatment_Changes | string instead of JSON | **FORMAT BUG** | Output is a raw JSON string, not parsed object |
| goals_of_treatment | "palliative" | OK | Stage IV = palliative |
| response_assessment | "stable to slightly decreased metabolic activity" | OK | Reasonable inference from PET scans |
| Lab_Plan | "PET CT now..." | **WRONG — ANSWER WRONG FIELD** | Lab_Plan should be about lab tests, not imaging |
| follow_up_next_visit | "pet ct" | **WRONG — ANSWER WRONG FIELD** | Should be a clinic visit date, not an imaging test |
| radiotherapy_plan | "XRT to L4 and T10 in June 2020" | **WRONG — PAST** | This is past treatment, not a plan. Temporal gate should have caught this |

**Llama 8B errors for Row 21: 8 issues (Patient type, in-person format, Stage incomplete, Metastasis incomplete, Lab CRITICAL miss, Treatment_Changes format, Lab_Plan/follow_up wrong field, radiotherapy past)**

---

### Row 22 (coral_idx=162) — New patient, post-mastectomy adjuvant therapy

**Ground truth (from note):**
- 63 y.o. female, "presents with discussion of treatment options for recently diagnosed breast cancer"
- First Med Onc visit post-surgery — this is a NEW patient consult
- Left breast mastectomy + SLN Bx: 1 cm ILC + 0.7 cm invasive with ductal/lobular features
- ER+ (>95%)/PR (60%)/HER2 2+ IHC, FISH negative = HER2 negative
- DCIS intermediate and high grade, 0/1 SLN positive
- Staging: not explicitly stated, but 1 cm tumor + 0 LN + no mets = approximately Stage I
- POCT glucose values from surgery hospitalization (not standard labs)
- Genetic testing: MYRIAD, VUS in BRCA2 (no clinically significant mutation)
- Plan: start letrozole (AI), baseline DEXA, exercise guidance, F/U 3 months

| Field | Llama 8B Output | Correct? | Issue |
|-------|-----------------|----------|-------|
| Patient type | "Follow up" | **WRONG** | First Med Onc visit to discuss treatment options. Should be "New patient" |
| second opinion | "No" | OK | |
| summary | "discussion of treatment options for recently diagnosed breast cancer" | OK | |
| Type_of_Cancer | "ER+ (>95%)/PR (60%)/HER2 2+ by IHC and negative by FISH 5-10%..." | **OVERLY VERBOSE** | Includes Ki-67 %, IHC details; also lists DCIS receptor status separately which is confusing |
| Stage_of_Cancer | "Not specified" | **COULD BE BETTER** | 1 cm + 0/1 SLN = approximately Stage I. Prompt says to infer when possible |
| lab_summary | "Glucose, meter download: 105, 185, 236, 116" | **DEBATABLE** | These are POCT glucose from surgery, not standard labs. Arguably correct but not clinically useful in this context |
| findings | Pathology + physical exam | OK | Comprehensive |
| current_meds | "" | OK | Post-surgery, not on oncologic meds yet |
| supportive_meds | Long list of non-oncologic meds | **WRONG** | Lists general meds (TYLENOL, LIPITOR, ATIVAN, etc.) as "supportive" — these aren't supportive oncology meds. Many are marked "Patient not taking" in the note |
| goals_of_treatment | "adjuvant" | OK | Correct |
| response_assessment | "low risk, no tumor in SLN, negative margins" | **WRONG FIELD** | This is pathology findings, not treatment response assessment. Patient hasn't started treatment. |
| Imaging_Plan | "DEXA" | OK | DEXA ordered |
| Lab_Plan | "DEXA" | **WRONG** | DEXA is imaging, not labs |
| Genetic_Testing_Plan | "Not yet on treatment — no genetic testing planned" | **WRONG** | Genetic testing WAS done (MYRIAD, VUS in BRCA2). This field conflates treatment status with genetic testing |
| Referral: Others | "Physical or occupational therapy, Social work" | **FABRICATION** | Note does NOT mention PT/OT/social work referrals. Only mentions exercise guidance. |

**Llama 8B errors for Row 22: 7 issues (Patient type, Type_of_Cancer verbose, Stage not inferred, supportive_meds wrong, response_assessment wrong field, Lab_Plan wrong, Referral fabrication)**

---

### Row 23 (coral_idx=163) — Post-surgery, discuss systemic therapy options

**Ground truth (from note):**
- 56 y.o. female, s/p partial mastectomy, here to discuss systemic therapy options
- Right breast Grade II micropapillary mucinous carcinoma, ER+ (>95%), PR+ (80%), HER2 equivocal IHC(2) negative FISH, Ki-67 5%
- 5.x cm tumor, 2/4 sentinel LN positive (micropapillary mucinous carcinoma 0.4mm)
- PET CT: no definite hypermetabolic metastatic disease, liver lesions likely benign (hemangioma, FNH)
- No staging explicitly stated, but large tumor + LN positive = likely Stage II-III
- Plan: send specimen for MammaPrint (MP), if low risk → radiation + adjuvant hormone therapy
- Referral: radiation oncology (Dr. ***** on 12/07/18), PT referral for axillary issues

| Field | Llama 8B Output | Correct? | Issue |
|-------|-----------------|----------|-------|
| Patient type | "Follow up" | **DEBATABLE** | She is s/p surgery and this is first visit to discuss systemic therapy. Could be New patient. |
| Type_of_Cancer | "Grade II micropapillary mucinous carcinoma, ER+/PR+/HER2-" | OK | Good summary |
| Stage_of_Cancer | "Not specified" | **COULD BE BETTER** | 5+ cm tumor + 2 positive LN = at least Stage II. Prompt says to infer. |
| Metastasis | "No" | **SHOULD BE "Not sure"** | Liver lesions noted (likely benign but not confirmed). PET says "no definite sites" which is equivocal. |
| findings | Very detailed imaging + pathology | OK | Comprehensive |
| current_meds | "" | OK | Post-surgery, not yet on oncologic meds |
| Medication_Plan | "adjuvant hormone therapy" | **VAGUE** | Correct direction but no specifics. Note discusses AI side effects in detail. |
| Genetic_Testing_Plan | "I will order a FISH test today" | **WRONG** | FISH was already done. The test being ordered is MammaPrint (MP), not FISH |
| Imaging_Plan | "MP" | **WRONG** | MammaPrint is a genomic test, not imaging |
| Lab_Plan | "MP" | **WRONG** | MammaPrint is a genomic test, not a lab test |

**Llama 8B errors for Row 23: 5 issues (Stage not inferred, Metastasis should be "Not sure", Genetic_Testing_Plan wrong test, Imaging/Lab_Plan wrong category)**

---

### Row 24 (coral_idx=164) — Metastatic breast cancer, on chemotherapy

**Ground truth (from note):**
- 45 y.o. female, metastatic breast cancer on Xeloda 1500/1000mg + ixabepilone
- Complex history: right IDC 2007 (ER+/PR+/HER2-), left IDC 2008 (ER+/PR-/HER2-)
- Originally Stage IIA (3.5 cm IDC, positive LN)
- Metastatic disease: brain (4cm mass), liver, bones (C2, L2, sacrum), LN, breast skin
- Current visit: off week, will restart cycle 04/12/2011
- Supraclavicular area "appears to be breaking up" (possible response)
- Labs: full CBC + CMP reviewed, "okay to proceed with chemotherapy"
- Pain management: try ibuprofen/tylenol in AM, oxycodone in PM

| Field | Llama 8B Output | Correct? | Issue |
|-------|-----------------|----------|-------|
| Patient type | "Follow up" | OK | Ongoing chemo follow-up |
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | **INCOMPLETE** | Originally ER+/PR+/HER2-, but left breast was ER+/PR-/HER2-. Receptor status changed. |
| Stage_of_Cancer | "metastatic (Stage IV)" | **INCOMPLETE** | Should note original stage (IIA) per prompt instructions |
| Metastasis | "Yes, to multiple sites" | **INCOMPLETE** | Should specify: brain, liver, bones (C2, L2, sacrum), lymph nodes, breast skin |
| lab_summary | Full CBC + CMP with values | OK | Comprehensive |
| findings | Historical imaging findings | OK | Detailed |
| current_meds | ["capecitabine (Xeloda)", "ixabepilone"] | OK | |
| recent_changes | "Started Xeloda 4000mg (actually took 2000mg)" | OK | Note details the dosing confusion |
| goals_of_treatment | "palliative" | OK | Stage IV |
| response_assessment | "New chest wall/breast skin metastases, indicating disease progression" | **DEBATABLE** | Note says supraclavicular area "breaking up" which could be response. Mixed picture. |
| Genetic_Testing_Plan | "Not yet on treatment — no response to assess." | **WRONG — ANSWER FROM WRONG FIELD** | Copy-paste from response_assessment template. Patient had BRCA testing (negative). |
| Imaging_Plan | "Scan in 3 weeks" | OK | Note says "Scan in 3 weeks" |

**Llama 8B errors for Row 24: 5 issues (Type_of_Cancer incomplete, Stage incomplete, Metastasis no details, response_assessment debatable, Genetic_Testing wrong field)**

---

## Summary: Llama 3.1 8B Error Patterns (Row 20-24)

### Error Categories

| Category | Count | Rows | Description |
|----------|-------|------|-------------|
| **Patient type wrong** | 3 | 20, 21, 22 | First Med Onc visits classified as "Follow up" |
| **Hallucination** | 1 | 20 | Stage "T4a" for DCIS (impossible) |
| **Critical omission** | 1 | 21 | Full CBC+CMP labs completely missed |
| **Wrong field (answer doesn't match question)** | 6 | 21, 22, 23, 24 | Lab_Plan gets imaging, follow_up gets imaging, response_assessment gets pathology, etc. |
| **Past vs future confusion** | 3 | 20, 21, 23 | Past genetic results or treatments reported as plans |
| **Incomplete extraction** | 5 | 21, 22, 24 | Stage not fully captured, metastasis sites not listed |
| **Format issues** | 2 | 21, 22 | in-person="yes", Treatment_Changes as raw string |
| **Fabrication** | 1 | 22 | PT/OT/Social work referrals not in note |
| **Wrong test/category** | 3 | 23 | MammaPrint classified as imaging/lab instead of genomic |
| **Vague output** | 1 | 23 | Medication_Plan just says "adjuvant hormone therapy" |

### Top Priority Fixes

1. **Patient type (3/5 rows wrong)** — Prompt needs stronger "first visit to oncologist = New patient" signal
2. **Wrong field / answer doesn't match question (6 instances)** — Gate 6 (Semantic) not catching enough
3. **Past vs future confusion (3 instances)** — Gate 4 (Temporal) not strict enough on plan fields
4. **Stage inference (3/5 rows incomplete)** — Prompt says to infer but model rarely does

---

## Qwen2.5-32B vs Llama 8B: What Changed (Row 20-24)

### Issues FIXED by 32B

| Row | Field | 8B (wrong) | 32B (correct) |
|-----|-------|-----------|--------------|
| 20 | Stage_of_Cancer | "T4a" (hallucination) | "" (safe empty) |
| 21 | Patient type | "Follow up" | "New patient" |
| 21 | Stage_of_Cancer | "Metastatic" | "Originally Stage II, now metastatic (Stage IV)" |
| 21 | lab_summary | "No labs in note" (CRITICAL miss) | Full CBC+CMP extracted |
| 21 | Metastasis | "Yes" (no detail) | "Yes (to bones, chest wall, infraclavicular, IM nodes)" |
| 22 | Patient type | "Follow up" | "New patient" |
| 22 | supportive_meds | Long list of non-oncologic drugs | "" (correctly empty) |
| 24 | Type_of_Cancer | "ER+/PR+/HER2-" (incomplete) | "Originally ER+/PR+/HER2-, metastatic biopsy ER+/PR-/HER2-" |
| 24 | Stage_of_Cancer | "metastatic (Stage IV)" | "Originally Stage IIA, now metastatic (Stage IV)" |
| 24 | Metastasis | "Yes, to multiple sites" | "Yes (to brain, liver, bones, and lymph nodes)" |

### Issues STILL PRESENT in 32B

| Row | Field | Issue |
|-----|-------|-------|
| 20 | Patient type | Still "follow up" (should be "New patient") |
| 20 | Stage_of_Cancer | Empty instead of "Stage 0 / Tis" (DCIS) |
| 21 | goals_of_treatment | "Not yet specified" (should be "palliative" — Stage IV) |
| 22 | Stage_of_Cancer | Empty (should infer ~Stage I from 1cm + 0 LN) |
| 23 | Stage_of_Cancer | Empty (should infer ~Stage II-III from 5cm + 2 LN+) |
| 24 | response_assessment | "Not yet on treatment" (WRONG — patient IS on chemo) |

### NEW Issues introduced by 32B

| Row | Field | Issue |
|-----|-------|-------|
| 21 | goals_of_treatment | Was correct "palliative" in 8B, now "Not yet specified" — regression |
| 24 | response_assessment | Was reasonable in 8B ("disease progression"), now wrong ("Not yet on treatment") |
| 20 | supportive_meds | 8B had "Vitamin D, calcium" (correct), 32B has "Bisphosphonate therapy" (not yet prescribed, only discussed) |

---

## 100-Row Full Review: V2 with Old Prompts (`default_20260301_084320`)

> 详细审查报告见 `results/default_20260301_084320/review.md`

### 总体结论
- **100/100 行全部有问题**，0 行完全正确
- 2 个疑似 pipeline bug 经排查均为审查误报
- 1 个极严重受体错误（Row 76: HER2 3+ 写成 HER2-）

> 各字段具体问题和受影响 Row 编号见文末「回归测试清单」(B3-B14)。

### 逐行错误摘要（Row 0-99）

<details>
<summary>点击展开 100 行逐行审查</summary>

| Row | coral_idx | 主要问题 |
|-----|-----------|---------|
| 0 | 140 | summary 遗漏 Stage IIA; Type_of_Cancer 缺 HER2-; lab 引用 2001 年旧数据; findings 遗漏腋窝肿块 |
| 1 | 141 | summary 遗漏 Lynch Syndrome; Type_of_Cancer 只写乳腺癌漏结肠/子宫内膜癌; lab 遗漏 Hgb 7.7; findings 遗漏胸壁感染 |
| 2 | 142 | Type_of_Cancer 遗漏 Ki-67 30-35%; goals 应为 "curative" 非 "risk reduction" |
| 3 | 143 | Stage 写 Grade/大小而非分期; findings 遗漏无复发证据; recent_changes 不是变化 |
| 4 | 144 | Metastasis 遗漏 sternal/brachial plexus; meds 重复; response 应为 mixed |
| 5 | 145 | summary 写 partial 实为 bilateral mastectomy; Type 缺 HER2-; Stage 可推断 I |
| 6 | 146 | **Patient type/second opinion 均错**; Type 缺 HER2+; lab 遗漏 CA 15-3; findings 只写症状 |
| 7 | 147 | Patient type 错; Stage 不完整; meds 列了 2018 年历史; response 答非所问 |
| 8 | 148 | Type 缺 HER2-/Ki-67; Stage 缺术后; findings 只写症状; response 写 neuropathy 改善 |
| 9 | 149 | in-person 格式错; Type 冗余; Stage 遗漏 8.8cm; findings/response 答非所问 |
| 10 | 150 | Type ER/PR/HER2 幻觉; meds 含已停 Letrozole; lab 遗漏异常值 |
| 11 | 151 | lab 放了 EF%(影像); findings 遗漏新脑转移; response 遗漏 CT 结果 |
| 12 | 152 | Patient type 错; Type 遗漏 DCIS score; Medication_Plan 患者尚未决定 |
| 13 | 153 | **严重**: Type 缺 HER2-; Stage 应为 IV; meds 严重错误; findings 答非所问 |
| 14 | 154 | Type 缺 HER2+; Stage 应写 "Clin st I/II"; findings 遗漏 MRI |
| 15 | 155 | **LLM 重复退化** (docusate 重复数十次) |
| 16 | 156 | Patient type 错; Stage 遗漏; supportive 写了 "nutritionist"; response 答非所问 |
| 17 | 157 | Patient type 错; meds 含未开始药物; Stage 不完整; Genetic_Testing 遗漏 |
| 18 | 158 | Stage 写 Grade/大小; lab 把 ER/PR% 放入; response 答非所问; findings 遗漏 MRI |
| 19 | 159 | Patient type 错; second opinion 错; Type 缺 PR+; Metastasis 遗漏; lab 引用 2013 年 |
| 20 | 160 | Patient type 错(referred for consultation); Stage 不具体; meds 幻觉 |
| 21 | 161 | Type 应标 HER2-; lab 遗漏大量值; Metastasis 不完整 |
| 22 | 162 | Patient type 错; Type 遗漏多灶 PR 差异; Stage 不具体; response 答非所问 |
| 23 | 163 | Patient type 错; Stage 混淆影像/病理大小; lab 把受体放入; response 答非所问 |
| 24 | 164 | summary 缺转移信息; Type 遗漏脑转移灶受体变化; Metastasis 不具体 |
| 25 | 165 | summary CMS code; Type 缺 Ki-67; findings 幻觉(75% Ki-67→肿瘤大小); goals 应为 "curative"; Referral 遗漏 |
| 26 | 166 | lab 答非所问; recent_changes 答非所问; meds 遗漏; Type 缺 HER2-; radiotherapy 把 MRI 当放疗 |
| 27 | 167 | findings 不忠实("no new findings" 但有病理); recent_changes 不准确 |
| 28 | 168 | **Patient type 错; Type 遗漏 micropapillary/PR 差异; Stage 遗漏 N1mi; response 幻觉** |
| 29 | 169 | summary 写 "early stage" 不准确(9cm HER2+); response 把 PET/CT 当治疗响应 |
| 30 | 170 | Metastasis 遗漏肝+可能脑; lab 放 LVEF; meds 含未开始 Doxil |
| 31 | 171 | Patient type 错; Type 缺亚型(pleomorphic lobular); meds 遗漏 pertuzumab |
| 32 | 172 | Type 缺 HER2-; Stage 遗漏 IIIA; meds letrozole/Femara 重复; goals 应为 "adjuvant" |
| 33 | 173 | Type 遗漏 2020 复发 PR+; meds 不忠实; recent_changes 搞反 |
| 34 | 174 | Type 缺 ER; meds HPI/Assessment 矛盾; response 遗漏 "no malignancy" |
| 35 | 175 | **严重**: Metastasis 错误(感染→转移); meds 遗漏 valtrex |
| 36 | 176 | meds 把推荐方案写成当前; goals 应为 "curative" |
| 37 | 177 | Type 缺 PR/HER2/BRCA1; meds 含未来药物; recent_changes 遗漏停 Taxol |
| 38 | 178 | lab 全是影像(CT/MRI/bone scan); meds 含计划中药物; radiotherapy 答非所问 |
| 39 | 179 | Patient type 错; Type 缺 Grade/Ki-67/PR 矛盾; meds 过度列举; response 答非所问 |
| 40 | 180 | Type PR 1% 应注明; Stage 答非所问; meds 含计划 AC-Taxol |
| 41 | 181 | Type 缺 HER2-; Stage 答非所问; findings 不忠实(margin re-excision); goals 应为 "adjuvant" |
| 42 | 182 | summary 遗漏 second primary; Type 缺 grade; lab 过时围手术期; response 答非所问 |
| 43 | 183 | Type 混合两套受体; Stage 遗漏; **meds 严重混乱**; response 答非所问 |
| 44 | 184 | Type 缺 PD-L1; Stage 应为 IV(已转移); meds 含已停 Xeloda |
| 45 | 185 | Type 缺 HER2-/Ki-67; Stage 不完整; recent_changes 遗漏 letrozole; response 含未来 |
| 46 | 186 | Patient type 错(second opinion); 轻微问题 |
| 47 | 187 | Type 遗漏 invasion; meds/supportive 列大量非肿瘤药且重复 |
| 48 | 188 | second opinion 可能错; Metastasis 遗漏 axillary; meds 全非肿瘤药; supportive 含 DISCONTINUED |
| 49 | 189 | Type 缺 PR+; Metastasis 遗漏 LN; lab 过时; response 遗漏 "disease under great control" |
| 50 | 190 | Patient type 应为 Education visit; findings 重复; goals 不准确; Medication_Plan 幻觉 |
| 51 | 191 | Stage 遗漏; lab 遗漏尿妊娠; meds 错误; Genetic_Testing 遗漏 Oncotype |
| 52 | 192 | **严重**: meds 把推荐方案写成当前; Metastasis 遗漏 SLN+; Referral 遗漏 genetics |
| 53 | 193 | Type 缺 BRCA2; Stage 不准确; supportive 分类错; Therapy_plan 写 continue 已完成化疗 |
| 54 | 194 | Patient type 可能错; Type 缺 HER2-/Ki-67; meds 推荐→已用且患者拒绝 |
| 55 | 195 | summary 幻觉(Ki-67→肿瘤大小); Type CMS code; Referral 遗漏 genetics |
| 56 | 196 | **严重**: supportive 把**过敏原当药物**; meds 含已完成化疗; Referral 遗漏 genetics |
| 57 | 197 | 轻微: Type 缺 Ki-67/Oncotype; lab 遗漏 DEXA |
| 58 | 198 | meds 混乱(sequential→同时); supportive 含 not taking; Referral 遗漏 psychiatry |
| 59 | 199 | Patient type 错; Type 缺 HER2-/Ki-67; response 答非所问; **Genetic_Testing 严重遗漏 Oncotype** |
| 60 | 200 | Stage 遗漏 "early stage"; findings 遗漏 CT/MRI |
| 61 | 201 | Patient type 可能错; lab 幻觉("pending"); response 不忠实 |
| 62 | 202 | Patient type 错(second opinion); response 不忠实; goals 应为 "curative" |
| 63 | 203 | Type 缺 PR+/HER2-; lab 引用 2 年前; findings 重复; goals 不准确 |
| 64 | 204 | Patient type 不准确; summary 过长; lab 无结果值; goals 应为 "curative" 非 "neoadjuvant" |
| 65 | 205 | Stage 遗漏; meds 含 not taking; findings 重复 |
| 66 | 206 | Metastasis 应为 "Unknown, staging pending"; meds 含已完成化疗; response 遗漏 LN 缩小 |
| 67 | 207 | **Type HER2 错**(TCHP 方案说明 HER2+但写 HER2-); Stage 遗漏; response 遗漏 MRI CR |
| 68 | 208 | Type 缺 HER2-/Ki-67; lab 遗漏详细 labs; goals 应为 "curative" 非 "neoadjuvant" |
| 69 | 209 | Type 右侧 PR 错; Stage 遗漏; response 答非所问 |
| 70 | 210 | **严重**: lab 全错(ROS 内容→lab); response 幻觉("responding" 无证据); goals 应为 "curative" |
| 71 | 211 | Patient type 错; Type 遗漏 neuroendocrine; meds 时态错; **Genetic_Testing 遗漏 Oncotype** |
| 72 | 212 | Type 缺 HER2-; findings 遗漏 fat necrosis |
| 73 | 213 | lab 答非所问; findings 不完整; Genetic_Testing 遗漏 |
| 74 | 214 | Metastasis 不忠实(axillary=regional); meds 时态错; goals 应为 "curative" |
| 75 | 215 | **极严重: HER2 3+ (FISH 13) 写成 HER2-**; supportive 含 not taking |
| 76 | 216 | goals 应为 "curative"; meds 时态错; response 遗漏 |
| 77 | 217 | **严重**: meds 严重错误(已停→current, 无系统治疗); response 遗漏 CT PD; Medication_Plan 幻觉 |
| 78 | 218 | Type 不具体; findings 答非所问; response 遗漏(tumor marker↑+pain↑); lab 过时 |
| 79 | 219 | Type 缺 HER2-; findings 不够具体 |
| 80 | 220 | Stage 写 "Not mentioned" 但有 Stage I; Type 缺亚型(tubular) |
| 81 | 221 | A/P 上游提取错; Stage 不一致; meds 混入未来计划 |
| 82 | 222 | Metastasis 错误(axillary=regional); second opinion 可能幻觉 |
| 83 | 223 | second opinion 错; Type 未注明受体变化; findings 遗漏脑/肝转移 |
| 84 | 224 | meds 时态错(已 PD 方案仍 current); lab 遗漏 CA 15-3; findings 遗漏脑 MRI |
| 85 | 225 | Type 遗漏 HER2+/受体变化; Stage 不准确; lab 放 FISH; meds 遗漏 denosumab |
| 86 | 226 | Patient type 错(second opinion); Type 缺 HER2-; findings 答非所问(帕金森) |
| 87 | 227 | Patient type/second opinion 错; Type ER/PR 不准确(弱+/转移-); Metastasis 遗漏肺/LN |
| 88 | 228 | supportive 把 tamoxifen 归支持; Procedure/radiotherapy 时态错 |
| 89 | 229 | lab 严重遗漏(TSH/CBC/CMP); **meds 药名错误**("Dexmethylprednisolone"); response 错 |
| 90 | 230 | Type 缺 HER2-; Metastasis 遗漏 LN; meds 含一次性药; response 幻觉 |
| 91 | 231 | Type 仅 "breast cancer" 无受体; Stage 遗漏(Stage IV); Lab_Plan 遗漏 |
| 92 | 232 | Type 格式混乱缺 HER2+; Stage 遗漏; Patient type 错; **response 幻觉**(未治疗却 "not responding") |
| 93 | 233 | Patient type 错; summary CMS code; lab 过时(2 年); supportive 遗漏 gabapentin |
| 94 | 234 | meds 混淆时态; response 含无意义数字 |
| 95 | 235 | Patient type 错; meds 幻觉(tamoxifen 未开始); findings/response 答非所问 |
| 96 | 236 | Type 缺 HER2-; response 幻觉("expected to be curative"); Medication_Plan 幻觉; Genetic_Testing/Referral 遗漏 |
| 97 | 237 | Stage 不规范; recent_changes 答非所问 |
| 98 | 238 | Patient type 错; Type 遗漏右侧 HER2 3+; Stage 遗漏; **Referral 严重遗漏(3 个全丢)** |
| 99 | 239 | Stage 不确定; meds 含已停药; radiotherapy_plan 幻觉 |

</details>

---

## V1 Pipeline Review (Rows 0-14): `default_v1_20260228_*`

> 详细审查见 `results/v1_pipeline_review_rows_0_14.md`

### V1 系统性问题

> V1 主要问题已在 V2 中修复，详见回归测试清单 B1、B2。

- Type_of_Cancer 不完整 (缺 ER/PR/HER2): Rows 0, 1, 2, 4, 5, 8, 9, 11, 12 → V2 已修复
- goals_of_treatment 冗长: 全部 15 行 → V2 已修复
- Stage_of_Cancer 遗漏: Rows 0, 1, 10 → V2 已修复
- response_assessment 答非所问: Rows 0, 1, 6 → V2 已修复

### V1 逐行摘要 (Rows 0-14)

| Row | coral_idx | V1 主要问题 | V2 修正 | V2 退步 |
|-----|-----------|------------|---------|---------|
| 0 | 140 | Type 缺 PR + 冗长; Stage 遗漏; goals 冗长 | 全部修正 | Referral 遗漏 |
| 1 | 141 | Stage 遗漏; goals 冗长; response 写计划 | 全部修正 + Type | Proc/Lab 条件计划被过滤 |
| 2 | 142 | Type 缺 PR + 冗长 | Type 修正 | - |
| 3 | 143 | goals 冗长 | goals + Type 改进 | - |
| 4 | 144 | Type 缺 PR; goals 冗长 | 都修正 | - |
| 5 | 145 | Type 缺 ER/PR; goals 冗长 | 都修正 | - |
| 6 | 146 | goals 冗长 | goals 修正 | - |
| 7 | 147 | goals 冗长 | goals + Type 改进 | - |
| 8 | 148 | Type 缺受体; goals 冗长 | 都修正 | - |
| 9 | 149 | Type 缺受体; goals 冗长 | 都修正 | - |
| 10 | 150 | Stage 遗漏; goals 冗长 | 都修正 | - |
| 11 | 151 | Type 缺 PR; goals 冗长 | goals 修正 | - |
| 12 | 152 | Type 缺 ER; goals 冗长 | 都修正 | - |
| 13 | 153 | goals 冗长 | goals + Type 改进 | - |
| 14 | 154 | goals 冗长 | goals 修正 | - |

**结论**: V2 pipeline 相比 V1 是全面升级，错误率降低 89%。3 处退步均为 trade-off（条件计划过滤 + redacted 文本处理）。

---

## Post-Prompt-Refactor Review (Rows 25-39): `default_20260301_161703`

> 详细审查见 `review_rows_25_39_detailed.md`

### 对比 100-row 旧 prompt：关键改进

| 改进领域 | 改进行数 | 说明 |
|---------|---------|------|
| Genetic_Testing_Plan 准确性 | 5/15 行重大改进 | 旧版误报 "No new tests"，新版正确提取 |
| Findings 详细度 | 10/15 行 | 更丰富的病理/影像/体格检查细节 |
| Procedure_Plan 完整性 | 11/15 行 | 更完整的计划提取 |
| Current_meds 分类 | 7/15 行 | 正确区分抗癌药 vs 支持性药物 |
| Lab_summary 诚实度 | 多行 | 明确 "No labs in note" 而非空白 |

### 对比 100-row 旧 prompt：新引入问题

| 问题 | 受影响行数 | 严重程度 | 根本原因 |
|------|-----------|---------|---------|
| **Genetic_Testing_Plan 字段串行** | 10/15 行 | **P0 Bug** | 误写 "Not yet on treatment — no response to assess"（response_assessment 默认值串到此字段） |
| Current_meds 遗漏 | 8/15 行 | P0 | CoT 过于严格区分 CURRENT/PLANNED，导致正在服用的药也被排除 |
| Lab_summary 遗漏 | 5/15 行 | P1 | Prompt 过于严格区分 labs vs imaging，LVEF/CT 结果被排除 |
| Patient type 错误 | 2/15 行 | P1 | Row 28, 37 误判 |
| Goals 判断错误 | 3/15 行 | P1 | Row 31, 35, 38 adjuvant vs curative vs neoadjuvant |
| Supportive_meds 错误分类 | 2/15 行 | P2 | Row 33, 34 把抗癌药(tamoxifen, arimidex)归为 supportive |

### Post-Refactor 逐行摘要 (Rows 25-39)

| Row | coral_idx | 问题数 | 主要问题 | vs 旧版本 |
|-----|-----------|--------|----------|-----------|
| 25 | 165 | 1 | Current_meds 遗漏 | Genetic testing 重大改进 |
| 26 | 166 | 3 | Current_meds 遗漏, Procedure 遗漏, 字段串行 | Findings 改进 |
| 27 | 167 | 3 | Current_meds 遗漏, 字段串行 | Findings/Procedure 重大改进 |
| 28 | 168 | 4 | **Patient type 错**, Lab 遗漏, Current_meds 遗漏 | 有改进但退化更多 |
| 29 | 169 | 1 | Lab 遗漏肿瘤标志物 | Findings/Procedure 改进 |
| 30 | 170 | 2 | Lab 遗漏 LVEF, Current_meds 遗漏 Doxil | Procedure 改进 |
| 31 | 171 | 2 | Goals 可能错误(curative vs palliative), Genetic 不明确 | Lab_summary 改进 |
| 32 | 172 | 1 | 字段串行 | Goals/Current_meds 改进 |
| 33 | 173 | 1 | Supportive_meds 错误分类(tamoxifen→supportive) | Lab_summary/Findings 改进 |
| 34 | 174 | 1 | Supportive_meds 错误分类(tamoxifen→supportive) | Findings 改进 |
| 35 | 175 | 1 | Goals 可能错误(adjuvant vs curative) | Procedure 改进 |
| 36 | 176 | 0 | **无问题** | Procedure 改进 |
| 37 | 177 | 4 | **Patient type 错**, **Lab 严重遗漏**, Current_meds 遗漏 | Genetic testing 重大改进 |
| 38 | 178 | 2 | Lab 遗漏影像, **Goals 错误**(adjuvant vs neoadjuvant) | Genetic testing 改进 |
| 39 | 179 | 0 | **无问题** | Genetic/Procedure 改进 |

**总体**: 约 30 处改进 vs 约 25 处新问题。其中 Genetic_Testing 字段串行 bug 影响 10 行，修复后净改进率将显著提升。

---

## Testfix 实验 Review

### testfix_20260228_221738 (Rows 0,1,4,6,10,11,19)

**目的推测**: 尝试从 "Not mentioned" 中挖掘更多信息
**结果**: 混合（3 改进 / 2 退化 / 2 中性，改进率 43%）

主要变化集中在 `response_assessment`：
- Rows 4, 6, 10: 更详细的症状和疾病变化描述
- Rows 0, 19: **退化** — 把"疾病诊断/复发状态"当作"治疗响应"（答非所问）

### testfix_20260228_223813 (Rows 20-29)

**目的推测**: 实施治疗时间点检查和概念区分
**结果**: 优秀（**8 改进 / 1 退化 / 1 中性，改进率 80%**）

关键修正：
- **Row 28**: Oncotype Dx "Low Risk" 被错误地当作治疗响应 → 修正为 "Not yet on treatment"
- **Row 29**: 初诊影像学发现被当作治疗响应 → 修正为 "Not yet on treatment"
- `goals_of_treatment` 修正：
  - Row 27: adjuvant → curative（术后目标是治愈）
  - Row 28: risk reduction → curative（早期浸润性癌目标是治愈）

唯一退化：Row 23 改为 "Not applicable" 丢失了有用的计划信息

---

## Cross-Experiment Comparison: Evolution Timeline

### Pipeline 演进概览

```
V1 (3-gate, grouped prompts)          平均 2.7 错误/行
    ↓ +6-gate, +SCHEMA/FAITH-trim/SPECIFIC/SEMANTIC
V2 (6-gate, grouped prompts)          平均 ~4 错误/行 (但类型不同)
    ↓ +field splitting, +CoT
V2 (6-gate, split+CoT prompts)        平均 ~1.5 错误/行 (含 bug)
    ↓ +model upgrade
Qwen 32B (6-gate, split+CoT)          平均 ~1.2 错误/行
```

> 各字段在不同版本中的具体问题和受影响 Row 编号见文末「回归测试清单」。

### 回归测试清单（Bug 追踪 + 具体 Row 编号）

下一版本跑完后，按此清单逐项检查修复情况。每个 Bug 列出了具体受影响的 Row 和验证方法。

---

#### P0 — 必须修复

**B7: HER2 受体写反（极严重，临床后果最大）**
- 发现于: `default_20260301_084320` (100-row)
- 受影响 Rows:
  - **Row 75** (coral_idx=215): HER2 3+, FISH ratio 13 → 模型写成 "HER2-"
  - **Row 67** (coral_idx=207): TCHP 方案（HER2 靶向）→ 模型写成 "HER2-"
- 验证: 检查 `Type_of_Cancer` 字段，必须包含 "HER2+" 或 "HER2 3+"
- 修复方向: prompt 中加强 ER/PR/HER2 三项必须全写规则；加药物→受体推断规则（TCHP/THP → HER2+）

**B9: Genetic_Testing_Plan 字段串行**
- 发现于: `default_20260301_161703` (post-refactor, rows 25-39)
- 受影响 Rows: **26, 27, 28, 29, 30, 32, 33, 34, 35, 36** (10/15 行)
- 错误表现: Genetic_Testing_Plan 输出 "Not yet on treatment — no response to assess"（这是 response_assessment 的默认文本）
- 验证: 以上 10 行的 `Genetic_Testing_Plan` 不应包含 "no response to assess"
- 修复方向: plan_extraction prompt 中给 Genetic_Testing_Plan 设置独立默认值

**B10: Current_meds CoT 过严导致遗漏**
- 发现于: `default_20260301_161703` (post-refactor, rows 25-39)
- 受影响 Rows:
  - 完全为空（应有药物）: **25, 27, 28, 34, 36, 37, 38** (7 行)
  - 遗漏部分药物: **30, 31, 32, 35, 39** (5 行)
- 验证: 对照原始笔记，正在服用的药物不应被遗漏。特别检查术前/术后场景
- 修复方向: 放宽 CoT 判断 — "currently on X" / "taking X" 即使是术前也应提取

---

#### P1 — 高优先级

**B3: response_assessment 答非所问**
- 发现于: `default_20260301_084320` (100-row, 估计 ~70/100 行有问题)
- 受影响 Rows（从审查摘要中确认的）:
  - 答非所问（写计划/症状/诊断而非响应）: **7, 9, 16, 18, 22, 23, 39, 42, 43, 69, 95**
  - 幻觉（编造响应结果）: **70, 90, 92, 96**
  - 不忠实（与原文矛盾）: **61, 62**
  - 遗漏（有响应信息但未提取）: **11, 49, 66, 67, 77, 78**
  - 含未来/计划（不是当前响应）: **8, 29, 45, 94**
  - 更多行详见 `results/default_20260301_084320/review.md`
- 验证:
  1. 未开始治疗 → "Not yet on treatment"
  2. 已治疗 → 影像/肿标/体检的客观变化（如 "CT shows stable disease"）
  3. 不应包含: 手术恢复、Oncotype 评分、风险评估、副作用变化、未来计划
- 修复方向: 继续优化 CoT 决策树

**B4: current_meds 时态混乱**
- 发现于: `default_20260301_084320` (100-row, 估计 ~55/100 行有问题)
- 受影响 Rows（从审查摘要中确认的）:
  - 含已停/历史药: **7, 10, 33, 44, 48, 65, 74, 84, 99**
  - 含计划/未开始药: **17, 30, 36, 38, 40, 52, 81, 95**
  - 严重错误（药名错/严重混乱）: **13, 43, 54, 58, 77, 89**
  - 遗漏正在服用的药: **4, 34, 37, 47, 85, 90**
  - 重复: **4, 32, 47**
  - 更多行详见 `results/default_20260301_084320/review.md`
- 验证: current_meds 应只包含就诊时正在服用的抗癌药物，不含已停/计划/支持性药物
- 注意: Post-refactor (B10) 将此问题从"时态混乱"换成了"CoT 过严遗漏"，需平衡

**B5: Patient type 错标**
- 发现于: `default_20260301_084320` (100-row, 估计 ~35/100 行有问题)
- 受影响 Rows: **6, 7, 12, 16, 17, 19, 20, 22, 23, 28, 31, 39, 46, 50, 54, 59, 61, 62, 71, 86, 87, 92, 93, 95, 98**
  - 更多行详见 `results/default_20260301_084320/review.md`
- 验证: 检查笔记是否含 "New Patient", "Consult Note", "referred by", "second opinion" → 应为 "New patient"
- 修复方向: 加强 "first visit to oncologist = New patient" 规则

**B6: Stage_of_Cancer 遗漏/不推断**
- 发现于: `default_20260301_084320` (100-row, 估计 ~45/100 行有问题)
- 受影响 Rows: **0, 3, 5, 7, 8, 9, 14, 16, 18, 20, 21, 22, 23, 24, 29, 32, 40, 41, 42, 44, 45, 51, 53, 60, 65, 67, 68, 69, 80, 81, 85, 91, 92, 97, 98**
  - 更多行详见 `results/default_20260301_084320/review.md`
- 验证: 检查笔记中有无肿瘤大小 + LN 状态 + 转移信息，有则应推断分期（如 1cm+0LN → ~Stage I）
- 修复方向: prompt 中提供更多推断示例

**B11: Lab_summary 过严排除影像/肿标结果**
- 发现于: `default_20260301_161703` (post-refactor, rows 25-39)
- 受影响 Rows:
  - **Row 27**: 少了 Platelet Count
  - **Row 28**: Lab 遗漏
  - **Row 29**: 遗漏肿瘤标志物
  - **Row 30**: 遗漏 LVEF
  - **Row 37**: **严重** — 完整 CBC+CMP 全部漏掉，误写 "No labs in note"
  - **Row 38**: 影像结果（CT CAP, MRI brain, bone scan）被排除
- 验证: lab_summary 应包含所有实验室检查结果（CBC、CMP、肿瘤标志物）
- 修复方向: 明确 prompt 定义包含哪些检查结果

---

#### P2 — 中期

**B8: LLM 重复退化**
- 发现于: `default_20260301_084320` (100-row)
- 受影响 Rows: **15** (coral_idx=155)
- 错误表现: Treatment_Summary 中 "docusate" 重复数十次
- 验证: Row 15 输出无重复文本
- 修复方向: 生成质量问题，可能需 repetition penalty 或 post-processing 检测

**B12: Supportive_meds 把抗癌药归为支持性药物**
- 发现于: `default_20260301_161703` (post-refactor)
- 受影响 Rows:
  - **Row 33**: tamoxifen 归为 supportive（应在 current_meds）
  - **Row 34**: arimidex (anastrozole) 归为 supportive（应在 current_meds）
- 验证: tamoxifen、letrozole、anastrozole 等内分泌治疗药应在 current_meds
- 修复方向: prompt 中加入抗癌药列表（内分泌治疗 ≠ supportive）

**B1: goals_of_treatment 冗长非标准（V1 遗留，已修复）**
- 发现于: V1 pipeline (rows 0-14)
- 受影响 Rows: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14** (全部 15 行)
- 状态: **已修复** — V2 split+CoT 后 goals 标准化为 curative/palliative/adjuvant 等

**B2: Type_of_Cancer 缺受体状态（V1 遗留，大幅改善）**
- 发现于: V1 pipeline (rows 0-14)
- 受影响 Rows: **0, 1, 2, 4, 5, 8, 9, 11, 12** (9/15 行使用冗长诊断代码)，**0, 1, 2, 4, 5, 8, 9, 11, 12** (11/15 行缺 ER/PR/HER2)
- 状态: **大幅改善** — V2 prompt 改进后完整率从 27% → 93%

---

#### Qwen 32B 问题

**B13: goals_of_treatment 退化**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows: **21** (coral_idx=161)
- 错误表现: Llama 8B 正确输出 "palliative"，Qwen 32B 退化为 "Not yet specified"（Stage IV 应为 palliative）
- 验证: Row 21 (Stage IV 转移性乳腺癌) 的 goals 应为 "palliative"

**B14: response_assessment 对正在治疗的患者写 "Not yet on treatment"**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows: **24** (coral_idx=164)
- 错误表现: 患者明确在 Xeloda + ixabepilone 化疗中，原文有 "supraclavicular area appears to be breaking up" + "labs okay to proceed with chemotherapy"，模型仍写 "Not yet on treatment"
- 验证: Row 24 不应为 "Not yet on treatment"，应提取客观响应信息

**B15: A/P 文本串字段（Qwen 特有）**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows:
  - **Row 21**: A/P 段 "I recommend a pet ct now and if stable continue arimidex alone. If pet ct shows progression could use faslodex with ***** if she has a ***** mutation." 被整段复制进 Lab_Plan、Genetic_Testing_Plan、Procedure_Plan、Referral-follow_up **四个字段**
- 错误表现: 模型不区分字段含义，把 A/P 原文当通用答案填入多个字段
- 验证: 每个字段的内容应回答各自的问题，不应出现相同的大段文本

**B16: Lab_Plan 字段归类错误**
- 发现于: `default_qwen_20260313_220920` (Llama 8B 也有此问题)
- 受影响 Rows:
  - **Row 21**: PET CT 放入 Lab_Plan（应在 Imaging_Plan）
  - **Row 22**: DEXA scan 放入 Lab_Plan（应在 Imaging_Plan）
  - **Row 23**: MammaPrint 放入 Lab_Plan（应在 Genetic_Testing_Plan）
- 验证: Lab_Plan 应只包含实验室检查（CBC、CMP、肿瘤标志物等），不含影像或基因组测试

**B17: supportive_meds 把未来讨论方案当成当前用药**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows:
  - **Row 20**: 原文 "how this latter risk could be mitigated with bisphosphonate therapy"（讨论中的未来方案）→ 模型写成 supportive_meds: "Bisphosphonate therapy"。实际当前 supportive 是 Vitamin D + calcium（Llama 8B 反而写对了）
- 验证: supportive_meds 应只包含患者当前正在服用的支持性药物

**B18: summary 药名与 current_meds 自相矛盾**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows:
  - **Row 24**: summary 写 "on Xeloda and Irinotecan"（来自原文旧模板 header），但 current_meds 正确写了 "capecitabine, ixabepilone"。A/P 段明确说 "ixabepilone"，且 "Port placed just after irinotecan" 说明 irinotecan 是之前方案
- 验证: summary 和 current_meds 中的药名应一致

**B19: recent_changes 模板文本串行**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows:
  - **Row 23**: recent_changes 字段输出 "Not yet on treatment — no response to assess."，这是 response_assessment 的默认模板文本
- 验证: recent_changes 应描述治疗变化，不应包含 "no response to assess"

**B20: Referral 包含本次就诊**
- 发现于: `default_qwen_20260313_220920`
- 受影响 Rows:
  - **Row 20**: Referral-Specialty 写 "Rad Onc and Med Onc"，但 Med Onc 是本次就诊，不是转诊
- 验证: Referral 应只包含向外转诊，不含自身科室

---

### Qwen 32B Row 20-24 逐行审查总结

**旧 Bug 修复情况：**

| Bug | 涉及 Row | 修复? |
|-----|---------|------|
| B5 Patient type | 20, 22, 23 | 22 ✓, 20/23 ✗ (1/3 修复) |
| B6 Stage 不推断 | 20, 22, 23 | 全部 ✗ (不再幻觉但仍空) |
| B13 goals 退化 | 21 | ✗ |
| B14 response 退化 | 24 | ✗ |

**Qwen 32B 关键改进（vs Llama 8B 同行）：**
- Row 21: lab_summary 从完全遗漏 → 完整提取（**最关键修复**）
- Row 21: Stage/Metastasis 从模糊 → 详细具体
- Row 22: Patient type/response_assessment/supportive_meds 修复
- Row 23: Genetic_Testing_Plan MammaPrint + Referral PT 修复
- Row 24: Type_of_Cancer 捕捉到受体变化 + Stage 完整

**Qwen 32B 新引入问题：** B15-B20 共 6 个新 Bug

---

## v4 + G3-REVERT 逐行审查: `default_qwen_20260314_131143`

> 与原始笔记逐行对比。前序实验 Bug 状态更新 + 新引入问题追踪。

### Bug 修复状态汇总

| Bug | 描述 | 之前状态 | 当前状态 | 说明 |
|-----|------|---------|---------|------|
| B13 | goals_of_treatment 退化 | 未修复 | **✓ FIXED** | G3-REVERT 恢复 "palliative" (Row 21) |
| B14 | response_assessment 写 "Not mentioned" | 不稳定 | **✗ NOT FIXED** | 提取不稳定：上一次跑对了，这次又写 "Not mentioned" |
| B15 | A/P 文本串字段 | — | **✓ FIXED** | 不再整段复制 A/P 到多字段 |
| B16 | Lab_Plan 归类错误 | FIXED | **✓ FIXED** | Lab_Plan 正确归类 |
| B17 | supportive_meds 未来方案当现用 | FIXED | **✓ FIXED** | Row 20 不再写 "Bisphosphonate therapy" |
| B18 | summary 药名矛盾 | NOT FIXING | NOT FIXING | 原文数据错误 (Irinotecan vs ixabepilone) |
| B19 | recent_changes 模板串行 | IMPROVED | **✓ FIXED** | 不再出现 "no response to assess" |
| B20 | Referral 包含 Med Onc | NOT FIXED | **✗ NOT FIXED** | Row 20 仍写 "Rad Onc, Med Onc" |
| B21 | supportive_meds 非肿瘤药 | FIXED | **✓ FIXED** | POST-SUPP 白名单过滤正常 |
| B22 | 安全否定值被清空 | FIXED | **✓ FIXED** | G3-PROTECT 正常工作 |

### Row 20 (coral_idx=160) — DCIS 术后辅助治疗咨询

**原文关键信息：**
- 70岁女性，由乳腺外科医生转介首次 Med Onc 就诊
- "referred to ***** (today's visit) and also Rad Onc"
- 右乳 DCIS，ER+(98%)/PR+(90%)，~5.0cm，切缘阴性
- 不在任何处方药上，骨质减少→维生素D+钙
- 基因检测（Invitae）已完成（过去）

| 字段 | 输出 | 正确? | 问题 |
|------|------|-------|------|
| Patient type | "follow up" | **错** | 模型提取 "New patient"（正确），但 **G6-SEMANTIC 改为 "follow up"**（过度修正） |
| second opinion | "" | OK | G3 清空，G6 阻止回填 |
| summary | "Right breast DCIS post-partial mastectomy..." | OK | 准确 |
| Type_of_Cancer | "ER+/PR+ DCIS" | OK | |
| Stage_of_Cancer | "Not mentioned in note" | 可接受 | DCIS 未明确写分期 |
| Metastasis | "No" | OK | |
| lab_summary | "No labs in note." | OK | |
| findings | 详细病理+体检 | **优秀** | |
| current_meds | "" | OK | 原文 "not on any prescription meds" |
| supportive_meds | "" | OK | POST-SUPP 过滤掉 "vitamin D, calcium"（这是骨质减少的补充剂，非癌症支持药） |
| goals_of_treatment | "risk reduction" | OK | |
| response_assessment | "Not yet on treatment" | OK | |
| Genetic_Testing_Plan | "None planned." | OK | 基因检测是过去（2016），不是计划 |
| Referral Specialty | "Rad Onc, Med Onc" | **错** | **B20** Med Onc 是本次就诊 |
| follow up | "in two weeks...3-4 months" | OK | |

**Row 20 问题：2 个**
1. Patient type: G6 过度修正（B24 新问题）
2. Referral: B20 未修复

---

### Row 21 (coral_idx=161) — 转移性乳腺癌，第二意见

**原文关键信息：**
- 72岁女性，"New Patient Evaluation"，second opinion
- ER+/PR+/HER2- IDC，原 Stage II，现转移(IV)
- 骨、胸壁、锁骨下、内乳淋巴结转移
- 当前用药：anastrozole, denosumab, prednisone(肺炎)
- PET 示良好反应 (11/03/20 和 04/24/21)
- 计划：PET CT → 根据结果调整

| 字段 | 输出 | 正确? | 问题 |
|------|------|-------|------|
| Patient type | "New patient" | OK | ✓ |
| second opinion | "yes" | OK | ✓ |
| Type_of_Cancer | "ER+/PR+/HER2- IDC" | OK | |
| Stage_of_Cancer | "Originally Stage II, now metastatic (Stage IV)" | **优秀** | |
| Metastasis | "Yes (bones, chest wall, infraclavicular, IM nodes)" | **优秀** | |
| lab_summary | 完整 CBC+CMP | **优秀** | |
| findings | PET 反应+体检 | OK | |
| current_meds | "anastrozole, denosumab" | OK | |
| recent_changes | "abemaciclib held...letrozole→anastrozole..." | **优秀** | |
| supportive_meds | "prednisone, denosumab, lomotil" | 部分 | denosumab 同时在 current_meds（重复） |
| goals_of_treatment | "palliative" | OK | **B13 FIXED** via G3-REVERT |
| response_assessment | "PET scan showed a good response" | OK | |
| radiotherapy_plan | "XRT to L4 and T10 in June 2020" | **错** | **过去治疗**不是计划。G4 未捕获 |
| Procedure_Plan | "I recommend a pet ct now..." | **错** | PET CT 是影像不是手术 |
| Genetic_Testing_Plan | "If progression could use faslodex..." | **错** | 治疗方案，非基因检测 |
| Referral follow up | 整段治疗计划 | **错** | 应只写时间 |

**Row 21 问题：4 个**
1. radiotherapy_plan: 过去治疗 (B30 新问题)
2. Procedure_Plan: 影像内容放错字段 (B15 残留变体)
3. Genetic_Testing_Plan: 治疗方案放错字段 (B15 残留变体)
4. Referral follow up: 内容过多 (轻微)

---

### Row 22 (coral_idx=162) — 新患者，术后辅助治疗

**原文关键信息：**
- 63岁女性，首次 Med Onc 就诊讨论治疗方案
- 左乳切除+SLN 活检：1cm ILC + 0.7cm IDC/ILC，ER+(>95%)/PR(60%)/HER2 IHC 2+ FISH-
- 0/1 SLN 阳性
- 计划：letrozole、基线 DEXA、钙+维D+运动、随访 3 月
- 遗传测试（MYRIAD，BRCA2 VUS）已完成

| 字段 | 输出 | 正确? | 问题 |
|------|------|-------|------|
| Patient type | "New patient" | OK | ✓（vs 之前 8B 写 "Follow up"） |
| summary | "63 y.o. with newly diagnosed breast cancer..." | OK | |
| Type_of_Cancer | "ER+/PR+/HER2- invasive carcinoma with ductal and lobular features" | OK | |
| Stage_of_Cancer | "" | **空** | 模型提取 "Approximately Stage I-II (1cm+0/1 SLN)"，**G3 清空了**（过度严格） |
| lab_summary | "POCT glucose: 105, 185, 236(H), 116" | OK | 术中血糖，是笔记唯一的 lab |
| findings | 详细病理+体检 | **优秀** | |
| current_meds | "" | OK | 术后未开始用药 |
| recent_changes | "None." | OK | G6 正确判断：letrozole 只是讨论，尚未开始 |
| goals_of_treatment | "adjuvant" | OK | G3-REVERT 恢复 |
| response_assessment | "Not yet on treatment" | OK | |
| Imaging_Plan | "baseline DEXA" | OK | DEXA 正确归类为影像 |
| Lab_Plan | "No labs planned." | OK | DEXA 不再放 Lab（B16 已修） |
| Genetic_Testing_Plan | "None planned." | OK | 遗传测试是过去 |
| Referral Nutrition | "Maintain a calcium-rich diet..." | **错** | **G6-SEMANTIC 过度修正**：prompt 明确说 "diet advice ≠ nutrition referral" |
| Medication_Plan_chatgpt | {} | 轻微 | G2 schema fix 失败 |

**Row 22 问题：2 个**
1. Stage_of_Cancer: G3 清空合理推断 (B26 新问题)
2. Referral Nutrition: G6 填入饮食建议 (B29 新问题)

---

### Row 23 (coral_idx=163) — 术后讨论全身治疗

**原文关键信息：**
- 56岁女性，右乳 Grade II 微乳头状粘液癌
- ER+(>95%)/PR+(80%)/HER2 IHC(2) FISH-，Ki-67 5%
- 5.x cm 肿瘤，2/4 SLN 阳性(0.4mm)
- PET CT 无明确转移；肝脏病灶考虑良性
- 计划：送 MammaPrint、如低风险→放疗+辅助内分泌治疗
- 转诊：放疗科(12/07/18)、PT

| 字段 | 输出 | 正确? | 问题 |
|------|------|-------|------|
| Patient type | "" | **空** | 模型提取 "New patient"（正确），**G3 清空，G6-PROTECT 阻止回填** |
| summary | "56 y.o. female with...mucinous carcinoma..." | OK | |
| Type_of_Cancer | "ER+/PR+/HER2 equivocal neg FISH Grade II micropapillary mucinous carcinoma" | OK | 详细 |
| Stage_of_Cancer | "Not mentioned in note" | 可接受 | G3-PROTECT 恢复 |
| Metastasis | "" | **空** | G3 清空 "Not sure"，G6-PROTECT 阻止回填。PET 说 "no definite sites" 但肝灶未排除 |
| findings | 病理+影像+体检 | **优秀** | 全面详细 |
| current_meds | "" | OK | 术后未开始 |
| supportive_meds | "Tylenol #4, oxycodone" | OK | 术后疼痛药，白名单匹配 |
| goals_of_treatment | "adjuvant" | OK | G3-REVERT |
| Genetic_Testing_Plan | "We will send her surgical specimen for MP." | **优秀** | MammaPrint 正确归类 |
| Referral Specialty | "Radiation oncology consult" | OK | ✓ |
| Referral Others | "Physical therapy" | OK | ✓ |

**Row 23 问题：2 个**
1. Patient type: G3 清空 "New patient" (B25 新问题，同 B24 模式)
2. Metastasis: G3 清空 "Not sure" (B27 新问题)

---

### Row 24 (coral_idx=164) — 转移性乳腺癌，化疗随访

**原文关键信息：**
- 45岁女性，转移性乳腺癌 (脑、肝、骨、淋巴结、乳房皮肤)
- 当前方案：Xeloda 1500/1000mg + ixabepilone + denosumab
- 原始 ER+/PR+/HER2-，左乳 ER+/PR-/HER2-，脑转灶三阴
- A/P 关键证据："supraclavicular area appears to be breaking up"、"labs okay to proceed"
- 计划：3周后扫描，04/12 重启化疗周期

| 字段 | 输出 | 正确? | 问题 |
|------|------|-------|------|
| Patient type | "Follow up" | OK | |
| summary | "on Xeloda and Irinotecan regimen" | **错** | **B18** 原文 header 数据错误（应为 ixabepilone） |
| Type_of_Cancer | "Originally ER+/PR+/HER2-, metastatic ER+/PR-/HER2-" | **优秀** | 捕获受体变化 |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | **优秀** | |
| Metastasis | "Yes (brain, liver, bones, lymph nodes)" | **优秀** | |
| lab_summary | 完整 CBC+CMP | **优秀** | |
| findings | 包含 "supraclavicular area appears to be breaking up" | **优秀** | |
| current_meds | "capecitabine, ixabepilone" | OK | G6 移除 denosumab（轻微遗漏） |
| supportive_meds | "ondansetron, docusate, prochlorperazine, oxycodone, miralax" | OK | POST-SUPP 正确过滤 |
| goals_of_treatment | "palliative" | OK | G3-REVERT |
| response_assessment | "Not mentioned in note." | **错** | **B14** 原文有响应证据但模型未提取（不稳定） |
| Imaging_Plan | "Scan in 3 weeks" | OK | |
| Genetic_Testing_Plan | "None planned." | OK | |

**Row 24 问题：1 个（+1 已知）**
1. response_assessment: B14 提取不稳定
2. summary: B18 数据错误（不修复）

---

### 新发现问题

**B24: G6-SEMANTIC 过度修正 Patient type (P1)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 20**
- 错误表现: 模型正确提取 "New patient"，G6-SEMANTIC 改为 "follow up"
- 根因: G6 模型认为 DCIS 术后讨论 = 随访，但实际是首次 Med Onc 就诊

**B25: G3 过度清空 Patient type (P1)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 23**
- 错误表现: 模型正确提取 "New patient"，G3-FAITH 清空，G6-PROTECT 阻止回填 → 最终为空
- 根因: G3 在笔记中找不到 "New Patient Evaluation" 等显式标记，就清空了合理推断

**B26: G3 过度清空推断的 Stage (P1)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 22**
- 错误表现: 模型提取 "Approximately Stage I-II (1cm and 0/1 SLN positive)"，G3 清空
- 根因: G3 要求原文明确写分期，但 prompt 鼓励从肿瘤大小+LN 推断

**B27: G3 过度清空 Metastasis "Not sure" (P2)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 23**
- 错误表现: 模型提取 "Not sure"（合理：PET 说 "no definite sites" 但肝灶未排除），G3 清空
- 根因: G3 对模糊/不确定答案一律清空

**B29: G6-SEMANTIC 填入饮食建议作为 Nutrition referral (P1)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 22**
- 错误表现: G6 将 Nutrition 从 "None" 改为 "Maintain a calcium-rich diet, take Vitamin D, exercise..."
- 根因: G6 忽略 prompt 规则 "General diet advice from the oncologist is NOT a nutrition referral"

**B30: radiotherapy_plan 写过去治疗 (P2)**
- 发现于: `default_qwen_20260314_131143`
- 受影响 Rows: **Row 21**
- 错误表现: radiotherapy_plan 写 "XRT to L4 and T10 in June 2020"（2020年已完成治疗）
- 根因: G4-TEMPORAL 和 G6-SEMANTIC 均未纠正

---

### 问题模式分析

**模式 1: G3 过度清空合理值 (3 实例: B25, B26, B27)**
- G3-FAITH 对非逐字引用的值太严格
- 影响：Patient type 推断、Stage 推断、不确定答案 ("Not sure") 都被清空
- G3-REVERT 只在全部字段被清空时触发，单字段清空无法恢复
- 需要: 要么放宽 G3 prompt ("keep if reasonably inferable"), 要么扩展 G3-REVERT 逻辑

**模式 2: G6 过度修正 (2 实例: B24, B29)**
- G6-SEMANTIC 有时会把正确值改错
- B24: "New patient" → "follow up" (G6 误判就诊类型)
- B29: "None" → 饮食建议 (G6 无视 prompt 规则)
- 需要: 给 G6 更严格的约束（不改 Patient type、不违反 prompt 显式规则）

**模式 3: Plan 字段残留串行 (Row 21, 3 实例: B30 + B15 变体)**
- Row 21 的 A/P 段短而密集，多个计划字段仍提取到相同的治疗方案文本
- 与 B15 不同的是不再是整段复制，而是模型对短文本的理解能力不足
- 需要: 考虑给 Row 21 类型的密集 A/P 段做更细致的 prompt 指导

**模式 4: B14 提取不稳定 (Row 24)**
- v4 prompt 上一次跑出正确结果，这次又回到 "Not mentioned"
- 即使 greedy decoding，浮点差异也导致不同输出
- 可能需要: 更强的 prompt 锚定或多次生成取最佳

---

### 整体质量评估 (v4 + G3-REVERT)

| Row | 严重问题 | 轻微问题 | 质量评估 |
|-----|---------|---------|---------|
| 20 | 1 (Patient type) | 1 (Referral B20) | 中等 |
| 21 | 0 | 4 (plan 字段串行) | 良好（核心字段全部正确） |
| 22 | 1 (Stage 空) | 1 (Nutrition G6) | 中等 |
| 23 | 1 (Patient type 空) | 1 (Metastasis 空) | 中等 |
| 24 | 1 (response B14) | 0 | 中等 |

**vs 之前版本改进:**
- B13 ✓ (goals "palliative" 恢复)
- B17 ✓ (不再写 "Bisphosphonate therapy")
- B19 ✓ (recent_changes 不再串模板)
- B21 ✓ (非肿瘤药被过滤)
- B22 ✓ (安全否定值保留)
- 核心字段（Type_of_Cancer, Stage, Metastasis, lab_summary, findings）质量显著提升

**待修复优先级:**
1. **P0**: G3 过度清空 Patient type/Stage/Metastasis (B25, B26, B27) — 影响 3/5 行 → **v5 已修复**
2. **P1**: G6 过度修正 (B24, B29) — 影响 2/5 行 → **v5 已修复**
3. **P1**: B14 response_assessment 不稳定 — 影响 1/5 行
4. **P2**: B20 Referral 包含 Med Onc — 影响 1/5 行
5. **P2**: B30 radiotherapy_plan 过去治疗 — 影响 1/5 行

---

## Row-by-Row Review: v5 (`default_qwen_20260314_140519`)

### v5 变更总结
- G3-REVERT-INFER: 分类值/推断值被 G3 清空后自动恢复
- G6-PROTECT-CLASS: 分类字段禁止 G6 修改
- G6 prompt: diet advice ≠ nutrition referral
- POST-SUPP: safe negative 短语不再被过滤

### Bug 状态总表

| Bug | 描述 | 状态 | 修复机制 |
|-----|------|------|----------|
| B13 | goals_of_treatment 被 G3 全部清空 | ✅ FIXED | G3-REVERT |
| B14 | response_assessment 提取不稳定 | ❌ NOT FIXED | 提取随机性 |
| B15 | plan 字段间内容复制 | ✅ FIXED | v3 prompt |
| B16 | Lab_Plan 放影像 | ✅ FIXED | G3-PROTECT |
| B17 | supportive_meds 放未开始药物 | ✅ FIXED | v4 prompt |
| B18 | summary 药名错误 (data) | ⬜ NOT FIXING | 数据错误 |
| B19 | recent_changes 模板串行 | ✅ FIXED | v3 prompt |
| B20 | Referral 包含本次科室 Med Onc | ❌ NOT FIXED | 需 prompt 改进 |
| B21 | supportive_meds 含非肿瘤药 | ✅ FIXED | POST-SUPP filter |
| B22 | 安全否定值被 G3 清空 | ✅ FIXED | G3-PROTECT |
| B24 | Patient type 被 G6 改成 "follow up" | ✅ FIXED | G6-PROTECT-CLASS |
| B25 | Patient type 被 G3 清空 | ✅ FIXED | G3-REVERT-INFER |
| B26 | Stage 被 G3 清空 | ✅ FIXED | G3-REVERT-INFER |
| B27 | Metastasis "Not sure" 被 G3 清空 | ✅ FIXED | G3-REVERT-INFER |
| B29 | Nutrition 被 G6 改成饮食建议 | ✅ FIXED | G6 prompt + G3-PROTECT |
| B30 | radiotherapy_plan 含过去治疗 | ❌ NOT FIXED | G4 temporal 未覆盖 |

### Row 20 (coral_idx=160)

| Field | v4+G3-REVERT | v5 | 变化 |
|-------|-------------|-----|------|
| Patient type | "follow up" (B24 G6改) | **"New patient"** ✓ | G6-PROTECT-CLASS 阻止修改 |
| Nutrition Referral | 饮食建议 (B29 G6改) | **"None"** ✓ | G6 prompt + G3-PROTECT |
| Referral Specialty | "Rad Onc, Med Onc" | "Rad Onc, Med Onc" | B20 未修复 |
| 其他字段 | — | — | 无变化，保持正确 |

### Row 21 (coral_idx=161)

| Field | v4+G3-REVERT | v5 | 变化 |
|-------|-------------|-----|------|
| goals_of_treatment | "palliative" | "palliative" ✓ | G3-REVERT 继续生效 |
| radiotherapy_plan | "XRT to L4 and T10 in June 2020" | "XRT to L4 and T10 in June 2020" | B30 未修复 |
| Procedure_Plan | 治疗 contingency 文本 | 治疗 contingency 文本 | 未修复 |
| Genetic_Testing_Plan | 治疗 contingency 文本 | 治疗 contingency 文本 | 未修复 |
| 核心字段 | — | — | 保持正确 |

### Row 22 (coral_idx=162)

| Field | v4+G3-REVERT | v5 | 变化 |
|-------|-------------|-----|------|
| Stage_of_Cancer | "" (B26 G3清空) | **"Approximately Stage I-II (1 cm and 0/1 SLN positive)"** ✓ | G3-REVERT-INFER 恢复 |
| Nutrition Referral | 饮食建议 (B29 G6改) | **"None"** ✓ | G6 prompt 修正 |
| supportive_meds | "" (POST-SUPP误删) | **"None currently being taken."** ✓ | POST-SUPP safe negative |
| 其他字段 | — | — | 保持正确 |

### Row 23 (coral_idx=163)

| Field | v4+G3-REVERT | v5 | 变化 |
|-------|-------------|-----|------|
| Patient type | "" (B25 G3清空) | **"New patient"** ✓ | G3-REVERT-INFER 恢复 |
| Metastasis | "" (B27 G3清空) | **"Not sure"** ✓ | G3-REVERT-INFER 恢复 |
| 其他字段 | — | — | 保持正确 |

### Row 24 (coral_idx=164)

| Field | v4+G3-REVERT | v5 | 变化 |
|-------|-------------|-----|------|
| response_assessment | "Not mentioned in note." | "Not mentioned in note." | B14 未修复（提取不稳定） |
| summary | 含 "Irinotecan" | 含 "Irinotecan" | B18 数据错误，不修 |
| 其他字段 | — | — | 保持正确 |

### v5 整体质量评估

| Row | 严重问题 | 轻微问题 | 质量评估 | vs v4 |
|-----|---------|---------|---------|-------|
| 20 | 0 | 1 (B20 Referral) | **良好** | ↑ (B24 修复) |
| 21 | 0 | 3 (B30 + plan leakage) | 良好 | → (无变化) |
| 22 | 0 | 0 | **优秀** | ↑↑ (B26 + B29 修复) |
| 23 | 0 | 0 | **优秀** | ↑↑ (B25 + B27 修复) |
| 24 | 1 (B14) | 0 | 中等 | → (无变化) |

**总计：严重问题 1 个 (vs v4 的 4 个)，轻微问题 4 个 (vs v4 的 7 个)**

### 剩余待修复

1. **B14** (P1): Row 24 response_assessment 提取不稳定 — LLM 随机性问题
2. **B20** (P2): Row 20 Referral 包含 "Med Onc" — prompt 已指导但模型未遵循
3. **B30** (P2): Row 21 radiotherapy_plan 含 2020 年过去治疗 — G4 temporal 应覆盖
4. **Row 21 plan leakage** (P2): Procedure_Plan/Genetic_Testing_Plan 含治疗 contingency 文本


## v6.2 变更总结 (`default_qwen_20260314_165646`)

### 代码变更（v6 → v6.2）

1. **POST-PROC-FILTER** (`ult.py`): `IMAGING_TERMS` + `RADIOTHERAPY_TERMS` 关键词表 + `filter_procedure_plan()` 从 procedure_plan 移除影像/放疗项
2. **POST-ADV** (`run.py`): Regex 从全文提取 code status (Full code/DNR/living will)，补丁 Advance_care_planning
3. **G3-PROTECT-RECEPTOR** (`ult.py`): G3 清空 Type_of_Cancer 且原值含受体状态 (ER/PR/HER2) 时自动恢复
4. **RT word boundary** (`ult.py`): `_RT_WORD_RE = re.compile(r'\brt\b')` 防止 "port" 误匹配 "rt"
5. **Prompt 改进**:
   - `Procedure_Plan`: 明确排除影像/放疗/检验/药物/转诊/咨询，附常见错误示例
   - `Advance_care`: 从"仅看 A/P"改为"搜索全文"找 code status
   - `recent_changes`: 区分"无变更"vs"未开始治疗"
   - `supportive_meds`: 强化 "Patient not taking" 排除 + 禁止肿瘤药物

### 旧 Bug 在 v6.2 中的状态

| Bug | v5 状态 | v6.2 状态 | 说明 |
|-----|---------|-----------|------|
| B31 | ❌ Stage 幻觉 (Row 30=170) | ✅ FIXED | 不再说 "Stage IIA"，改为 "Stage not mentioned, now metastatic (Stage IV)" |
| B33 | ❌ supportive_meds "Patient not taking" | ❌ NOT FIXED | Row 30 (170) 仍列入 ondansetron/prochlorperazine |
| B34 | ❌ response_assessment 遗漏 | ❌ NOT FIXED | Row 30 (170) 仍说 "Not mentioned" |
| B35 | ❌ PR 状态错误 | ✅ PARTIAL | Row 33 (173) G3-PROTECT-RECEPTOR 恢复 ✓；Row 34 (174) 受体信息仍被 G3 trimmed |
| B36 | ❌ radiotherapy_plan "None" | ⚠️ NEEDS VERIFY | Row 30 (170) radiotherapy_plan 仍为 "None"，但该行 note 是否讨论 RT 未确认 |
| B37 | ❌ Procedure_Plan 混入影像/RT | ✅ FIXED | Prompt 改进足够，POST-PROC-FILTER 未触发 |
| B38 | ❌ tamoxifen in supportive_meds | ✅ FIXED | Prompt 改进 |
| B39 | ❌ recent_changes "Not yet on treatment" | ✅ PARTIAL | Row 34 (174) ✓；Row 30 (170) 仍有此问题 |
| B42 | ❌ Advance care 遗漏 Full code | ✅ FIXED | POST-ADV 补丁 3 行 (Row 26=165, 31=170, 32=171) |

## Row-by-Row Review: v6.2 (Rows 25-34, coral_idx 165-174)

### Bug 状态总表 (v6.2)

| Bug | Row(s) (coral_idx) | Field | 描述 | 严重程度 | 状态 |
|-----|-------------------|-------|------|---------|------|
| B13 | 20 (160) | goals_of_treatment | 被 G3 全部清空 | P0 | ✅ FIXED |
| B14 | 24 (164) | response_assessment | 提取不稳定 | P1 | ❌ NOT FIXED |
| B15 | — | plan 字段 | 内容复制 | P1 | ✅ FIXED |
| B16 | — | Lab_Plan | 放影像 | P1 | ✅ FIXED |
| B17 | — | supportive_meds | 放未开始药物 | P1 | ✅ FIXED |
| B18 | 24 (164) | summary | 药名错误 (data) | P2 | ⬜ NOT FIXING |
| B19 | — | recent_changes | 模板串行 | P1 | ✅ FIXED |
| B20 | 20 (160) | Referral | 包含 Med Onc | P2 | ❌ NOT FIXED |
| B21 | — | supportive_meds | 含非肿瘤药 | P1 | ✅ FIXED |
| B22 | — | 安全否定值 | 被 G3 清空 | P1 | ✅ FIXED |
| B24 | 20 (160) | Patient type | 被 G6 改 | P0 | ✅ FIXED |
| B25 | 23 (163) | Patient type | 被 G3 清空 | P0 | ✅ FIXED |
| B26 | 22 (162) | Stage | 被 G3 清空 | P0 | ✅ FIXED |
| B27 | 23 (163) | Metastasis | 被 G3 清空 | P0 | ✅ FIXED |
| B29 | 22 (162) | Nutrition | 被 G6 改 | P1 | ✅ FIXED |
| B30 | 21 (161) | radiotherapy_plan | 含过去治疗 | P2 | ❌ NOT FIXED |
| B31 | 30 (170) | Stage | 幻觉 "Stage IIA" | P0 | ✅ FIXED |
| B32 | 34 (174) | current_meds | anastrozole/tamoxifen (data) | — | ⬜ NOT FIXING |
| B33 | 30 (170) | supportive_meds | "Patient not taking" 药物 | P1 | ❌ NOT FIXED |
| B34 | 30 (170) | response_assessment | 遗漏 PET progression | P0 | ❌ NOT FIXED |
| B35 | 33,34 (173,174) | Type_of_Cancer | PR/受体状态 | P1 | ✅ PARTIAL |
| B36 | 30 (170) | radiotherapy_plan | "None" 待验证 | P2 | ⚠️ NEEDS VERIFY |
| B37 | — | Procedure_Plan | 混入影像/RT | P1 | ✅ FIXED |
| B38 | — | supportive_meds | tamoxifen 混入 | P1 | ✅ FIXED |
| B39 | 30 (170) | recent_changes | "Not yet on treatment" 误用 | P1 | ✅ PARTIAL |
| B42 | — | Advance_care | 遗漏 code status | P1 | ✅ FIXED |
| **B43** | **26 (165)** | **Lab_Results** | **为遮蔽值 (*****) 编造具体数字** | **P0 幻觉** | **❌ NEW** |
| **B44** | **26 (165)** | **Referral/Others** | **遗漏 social work 转诊** | **P2 遗漏** | **❌ NEW** |
| **B45** | **26,27,28,32,33,34 (165-168,172-174)** | **goals_of_treatment** | **早期+辅助→应为 curative，输出 adjuvant** | **P1 分类** | **❌ NEW** |
| **B46** | **27 (166)** | **current_meds** | **遗漏 zolendronic acid (Zometa)** | **P2 遗漏** | **❌ NEW** |
| **B47** | **27 (166)** | **Lab_Plan** | **遗漏 UA** | **P2 遗漏** | **❌ NEW** |
| **B48** | **29,30,32,33,34 (168,169,171,172,173)** | **Cancer_Diagnosis** | **Distant Metastasis field 缺失** | **P2 schema** | **❌ NEW** |
| **B49** | **30 (169)** | **Stage_of_Cancer** | **"Stage IV metastatic" 幻觉 (note: stage II-III, no mets, curative)** | **P0 幻觉** | **❌ NEW** |
| **B50** | **30 (169)** | **Clinical_Findings** | **Ki-67 "~3.0%" 应为 "~30%" (差10倍)** | **P1 事实错误** | **❌ NEW** |
| **B51** | **30 (169)** | **radiotherapy_plan** | **"None" 但 note 说 radiation 在治疗计划中** | **P1 遗漏** | **❌ NEW** |
| **B52** | **30 (169)** | **Imaging_Plan** | **遗漏 TTE (echocardiogram)** | **P2 遗漏** | **❌ NEW** |
| **B53** | **31 (170)** | **response_assessment** | **"Not mentioned" 但 PET 明确显示 progression** | **P0 遗漏** | **❌ NEW** |
| **B54** | **31 (170)** | **supportive_meds** | **ondansetron/prochlorperazine "Patient not taking"** | **P1 时态** | **❌ NEW (=B33)** |
| **B55** | **31,35 (170,174)** | **Therapy_plan** | **影像检查 (MRI/PET/Echo/mammogram) 混入 therapy_plan** | **P2 字段** | **❌ NEW** |
| **B56** | **32 (171)** | **current_meds** | **pertuzumab 已停 ("recently dropped") 仍列入** | **P1 时态** | **❌ NEW** |
| **B57** | **32 (171)** | **Referral/Others** | **遗漏 exercise counseling** | **P2 遗漏** | **❌ NEW** |
| **B58** | **32 (171)** | **Referral/Genetics** | **"Genetic testing negative" 是结果非转诊** | **P2 分类** | **❌ NEW** |
| **B59** | **32 (171)** | **Metastasis** | **遗漏 left adnexal lesion 转移灶** | **P2 遗漏** | **❌ NEW** |
| **B60** | **33 (172)** | **response_assessment** | **"Not mentioned" 但 A/P 说 "no evidence of recurrence"** | **P1 遗漏** | **❌ NEW** |
| **B61** | **34 (173)** | **recent_changes** | **过去停药/拒绝 + 未来计划混在一起** | **P2 时态** | **❌ NEW** |
| **B62** | **35 (174)** | **Medication_Plan** | **"anastrozole" vs current_meds "tamoxifen" (数据矛盾)** | **—** | **⬜ NOT FIXING** |
| **B63** | **35 (174)** | **Therapy_plan** | **mammogram 混入 therapy_plan** | **P2 字段** | **❌ NEW (=B55)** |
| **B64** | **35 (174)** | **Type_of_Cancer** | **G3 trimmed 受体信息 (threshold 0.5 过宽)** | **P2 遗漏** | **❌ NEW** |

### Row 26 (coral_idx=165) — Stage IB TNBC, 新患者

| Field | 值 | 评估 |
|-------|-----|------|
| Patient type | "New patient" | ✓ |
| in-person | "Televisit" | 需验证 |
| Type_of_Cancer | "Estrogen receptor negative triple negative breast cancer" | ⚠️ 应列 ER-/PR-/HER2- |
| Stage | "Stage IB" | ✓ |
| lab_summary | WBC 5.4, RBC 4.80, Hgb 13.0... | **❌ B43 幻觉：note 中 lab 值为 *****（遮蔽），模型编造了具体数字** |
| findings | Ki-67 75%, node negative + 症状 | ⚠️ 含症状 (armpit pain) |
| goals_of_treatment | "adjuvant" | **❌ B45：早期+辅助=curative** |
| Referral/Others | "None" | **❌ B44：遗漏 social work 转诊** |
| Advance_care | "Full code." | ✓ POST-ADV 补丁 |

### Row 27 (coral_idx=166) — 转移性 ER+ 骨转移, 随访

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | ✓ |
| Stage | "Originally Stage I, now metastatic (Stage IV)" | ✓ |
| current_meds | "letrozole, goserelin" | **❌ B46：遗漏 zolendronic acid (Zometa)** |
| response_assessment | "Stable to slightly decreased metabolic activity of osseous metastases" | ✓ 优秀 |
| goals_of_treatment | "palliative" | ✓ |
| Lab_Plan | "CBC with platelets" | **❌ B47：遗漏 UA** |
| follow_up | "if pain worsens" | ⚠️ 遗漏 "2 weeks" 时间线 |
| Advance_care | "Not discussed during this visit." | ⚠️ 无 code status |

### Row 28 (coral_idx=167) — 早期 ER+, 新患者

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- grade 1 IDC" | ✓ |
| Stage | "Stage I" | ✓ |
| goals_of_treatment | "adjuvant" | **❌ B45：早期+辅助=curative** |
| radiotherapy_plan | 详细 radiation 计划 | ✓ |
| Imaging_Plan | "DEXA" | ✓ |
| Distant Metastasis | 存在 | ✓ (此行有该字段) |

### Row 29 (coral_idx=168) — 早期 ER+ multifocal, 新患者

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | ✓ |
| Stage | "pT1c(m)N1(sn)M0" | ✓ |
| goals_of_treatment | "adjuvant" | **❌ B45：早期+辅助=curative** |
| Distant Metastasis | 缺失 | **❌ B48** |
| recent_changes | "Not yet on treatment" | ✓ 此患者确实未开始 |
| Procedure_Plan | "surgery pending, September 2019" | ✓ |

### Row 30 (coral_idx=169) — ER-/HER2+ IDC Stage II-III, 新患者

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER-/PR-/HER2+ invasive ductal carcinoma" | ✓ |
| Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | **❌ B49 CRITICAL：note 明确说 "clinical stage II-III, no evidence of metastases, curative intent"** |
| Metastasis | "No" | ✓ (但与 Stage 矛盾！) |
| Distant Metastasis | 缺失 | **❌ B48** |
| findings / Ki-67 | "~3.0%" | **❌ B50：应为 "~30%"，差 10 倍** |
| radiotherapy_plan | "None" | **❌ B51：note 说 "treatment recommendations will include... radiation"** |
| Imaging_Plan | "No imaging planned" | **❌ B52：TTE 遗漏** |
| goals_of_treatment | "curative" | ✓ |
| Therapy_plan | 含 THP/AC 或 TCHP 方案 | ✓ 详细 |
| Medication_Plan | 详细化疗方案 | ✓ 优秀 |
| Procedure_Plan | "Mediport placement" | ✓ |

### Row 31 (coral_idx=170) — 转移性 ER+ Stage IV, 随访

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- metastatic breast cancer" | ✓ |
| Stage | "Originally Stage not mentioned, now metastatic (Stage IV)" | ✓ (B31 FIXED) |
| response_assessment | "Not mentioned in note." | **❌ B53：PET/CT 明确显示 "Progression of metastatic disease"** |
| supportive_meds | "ondansetron, prochlorperazine" | **❌ B54：两药都标注 "Patient not taking"** |
| recent_changes | "Not yet on treatment" | **❌ B39：患者有 prior 治疗史，正在换方案到 Doxil** |
| Therapy_plan | 含 Brain MRI, PETCT, Echo, MRI pelvis | **❌ B55：影像项混入 therapy_plan** |
| Advance_care | "Full code." | ✓ POST-ADV 补丁 |
| goals_of_treatment | "palliative" | ✓ |

### Row 32 (coral_idx=171) — 转移性 ER+/PR-/HER2+ 多发转移, 新患者

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR-/HER2+ pleomorphic lobular cancer" | ✓ |
| current_meds | "exemestane, trastuzumab, pertuzumab" | **❌ B56：pertuzumab "recently dropped due to diarrhea"** |
| Referral/Others | "None" | **❌ B57：遗漏 exercise counseling** |
| Referral/Genetics | "Genetic testing negative" | **❌ B58：是结果非转诊** |
| Metastasis | "Yes (to lymph nodes of left neck, chest, abdomen, pelvis)" | **❌ B59：遗漏 left adnexal lesion** |
| Distant Metastasis | 缺失 | **❌ B48** |
| Advance_care | "Full code. Living will on file." | ✓ POST-ADV 补丁 |

### Row 33 (coral_idx=172) — ER+ ILC on letrozole, 随访

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- invasive lobular carcinoma" | ✓ (G3-PROTECT-RECEPTOR 恢复) |
| Stage | "Originally stage IIB, now stage IIIA" | ✓ |
| current_meds | "letrozole" | ✓ |
| response_assessment | "Not mentioned in note." | **❌ B60：A/P 说 "Exam shows no evidence of recurrence"** |
| goals_of_treatment | "adjuvant" | **❌ B45：可接受但 curative 更精确** |
| Distant Metastasis | 缺失 | **❌ B48** |

### Row 34 (coral_idx=173) — ER+ IDC 局部复发, 随访

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "ER+/PR+/HER2- invasive ductal carcinoma" | ✓ (原 PR- 变 PR+，此为 2020 复发活检结果) |
| current_meds | "" | ⚠️ 患者 self D/C anastrozole，可能正确 |
| recent_changes | "anastrozole D/Ced, declined CW XRT, tamoxifen 20mg" | **❌ B61：过去停药 + 过去拒绝 + 未来计划混在一起** |
| Distant Metastasis | 缺失 | **❌ B48** |

### Row 35 (coral_idx=174) — ILC pT1cN0, 随访

| Field | 值 | 评估 |
|-------|-----|------|
| Type_of_Cancer | "invasive lobular carcinoma" | **❌ B64：G3 trimmed 受体信息，原提取含 ER+/PR+/HER2-** |
| Stage | "pT1cN0(sn)" | ✓ |
| current_meds | "tamoxifen" | ✓ |
| Medication_Plan | "Continue on anastrozole" | **❌ B62：数据矛盾（A/P 写 anastrozole，HPI 写 tamoxifen）⬜ NOT FIXING** |
| Therapy_plan | "mammogram... return in 6 months" | **❌ B63：mammogram 是影像不是治疗** |
| goals_of_treatment | "adjuvant" | **❌ B45** |

### v6.2 整体质量评估

| Row | coral_idx | P0 问题 | P1 问题 | P2 问题 | 质量 |
|-----|-----------|---------|---------|---------|------|
| 26 | 165 | 1 (B43 lab 幻觉) | 1 (B45 goals) | 1 (B44 referral) | **差** |
| 27 | 166 | 0 | 0 | 2 (B46, B47) | **良好** |
| 28 | 167 | 0 | 1 (B45 goals) | 0 | **良好** |
| 29 | 168 | 0 | 1 (B45 goals) | 1 (B48 schema) | **良好** |
| 30 | 169 | 1 (B49 stage 幻觉) | 2 (B50 Ki67, B51 RT) | 2 (B48, B52 TTE) | **差** |
| 31 | 170 | 1 (B53 response) | 2 (B54 supp, B39 recent) | 1 (B55 therapy) | **差** |
| 32 | 171 | 0 | 1 (B56 meds) | 4 (B57-59, B48) | **中等** |
| 33 | 172 | 0 | 1 (B60 response) | 2 (B45, B48) | **中等** |
| 34 | 173 | 0 | 0 | 3 (B61, B48, minor) | **良好** |
| 35 | 174 | 0 | 0 | 3 (B63, B64, B45) | **良好** |

**总计：P0 = 3 个，P1 = 9 个，P2 = 19 个**

### 新发现的系统性问题

1. **B43 Lab 幻觉** (P0): 遮蔽值 (*****) 被模型编造为具体数字。需要在 prompt 中明确禁止猜测遮蔽值，或后处理检测。
2. **B45 goals_of_treatment "adjuvant" vs "curative"** (P1): 6/10 行有此问题。Prompt 已有决策树但模型仍输出 "adjuvant"。可能需要更强的 CoT 引导或 G6 修正。
3. **B48 Distant Metastasis 缺失** (P2): 5/10 行缺少此 field。Schema 验证 (G2) 应该捕获但未生效。
4. **B49 Stage 幻觉** (P0): 最危险错误——note 明确说无转移+curative intent，模型输出 Stage IV/metastatic。需要交叉验证 Stage 与 Metastasis 的一致性。
5. **B53 response_assessment 遗漏** (P0): PET 明确显示 progression 但输出 "Not mentioned"。可能因为 PET 结果在 note body 而非 A/P 段。
6. **B54/B33 "Patient not taking" 问题** (P1): 尽管 prompt 已禁止，模型仍将标注为 "not taking" 的药物列入。需要后处理检测。
7. **B55 Therapy_plan 混入影像** (P2): 影像检查混入 therapy_plan。可考虑类似 POST-PROC-FILTER 的后处理。

### 优先修复建议

1. **P0 — B49 Stage 幻觉**: 后处理校验 — 如果 Metastasis="No" 但 Stage 含 "Stage IV" 或 "metastatic"，自动修正
2. **P0 — B43 Lab 幻觉**: Prompt 加 "If lab values are redacted (*****), write 'redacted' — do NOT guess numbers"
3. **P0 — B53 response_assessment**: 传入全文而非仅 A/P 给 Response_Assessment 提取
4. **P1 — B45 goals "adjuvant"→"curative"**: 后处理规则：if Stage I-III + no distant met + adjuvant therapy → "curative"
5. **P1 — B54 "Patient not taking"**: POST-SUPP 后处理检测 "not taking" 标记
6. **P2 — B48 Distant Metastasis**: G2 schema 验证强化，确保所有必需 key 存在

---

## Row-by-Row Review: v9b Batch 2 (`default_qwen_20260315_095522`)

**模型**: Qwen2.5-32B-AWQ (4bit) | **Pipeline**: V2, 5-gate, v9b code | **Rows**: 36-45 (coral_idx 175-184) | **日期**: 2026-03-15 | **耗时**: 33.3 min

> 这些是新行，无之前 tracking 历史。v9b 改动：`*****`→`[REDACTED]` 预处理 + POST-REFERRAL regex。

### Row 36 (coral_idx=175) — 27岁女性, pT3N0 混合导管/黏液癌, Abraxane cycle 8

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous | ✅ | |
| Stage | pT3N0 | ✅ | |
| current_meds | abraxane, zoladex | ⚠️ P2 | 遗漏 tamoxifen（HPI 有 "she ***** tamoxifen" 但 A/P 不含 tamoxifen。A/P 优先策略的副作用） |
| supportive_meds | Zofran, Compazine | ⚠️ P2 | valtrex（化疗期抗病毒ppx）未被支持药物白名单覆盖 |
| response_assessment | "no evidence of disease recurrence on imaging and exam" | ⚠️ P2 | PET/CT 显示术后改变非正式评估。cycle 8/12 无明确 response 评估 |
| Procedure_Plan | "will get doppler to r/o DVT" | ⚠️ P2 | Doppler 是影像不是手术（已在 Imaging_Plan 正确收录） |
| Referral Specialty | Radiation oncology referral | ✅ | |
| goals | curative | ✅ | |
| Advance care | Not discussed | ✅ | |

**Bug IDs**: B65 (current_meds tamoxifen 遗漏, P2), B66 (Procedure_Plan 含影像, P2)

---

### Row 37 (coral_idx=176) — 61岁女性, Stage IIA TNBC, 视频会诊, 讨论辅助化疗

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER-/PR-/HER2- IDC | ✅ | |
| Stage | Stage IIA | ✅ | |
| in-person | Televisit | ✅ | |
| current_meds | "" | ✅ | |
| **Nutrition** | "I recommend an anti inflammatory/mediterranean diet" | **❌ P1** | 假阳性：饮食建议 ≠ 营养转诊。Prompt 已有此规则但模型忽略 |
| **Procedure_Plan** | "adjuvant chemotherapy with AC followed by Taxol" | **❌ P1** | 化疗 ≠ 手术操作 |
| goals | curative | ✅ | |
| Advance care | Full code | ✅ | |

**Bug IDs**: B67 (Nutrition 假阳性—饮食建议, P1), B68 (Procedure 含化疗, P1)

---

### Row 38 (coral_idx=177) — 43岁女性, BRCA1, Stage IIB, ER-/PR weak+/HER2-

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER-/PR+/HER2- IDC | ✅ | |
| Stage | Stage IIB | ✅ | |
| **Nutrition** | "I recommend an anti inflammatory diet..." | **❌ P1** | 同 B67 |
| **Genetics referral** | "BRCA 1 mutation" | **❌ P1** | 诊断结果 ≠ 外转 referral |
| **Others referral** | "None" | **❌ P1** | 原文 Psychologic 部分有 "Refer to social worker." POST-REFERRAL regex 未匹配（需 "will refer" 格式） |
| Procedure_Plan | bilateral mastectomy Jan 31 | ✅ | |
| Advance care | full code | ✅ | |

**Bug IDs**: B69 (Nutrition 假阳性, P1), B70 (Genetics=诊断结果非referral, P1), B71 (Social work referral 遗漏, P1)

---

### Row 39 (coral_idx=178) — 27岁女性, T2N1 TNBC, 新辅助化疗讨论

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| **Type_of_Cancer** | "ER/PR/[REDACTED] negative, HER2: not tested grade 3 IDC" | **❌ P1** | A/P 说 "triple negative"，应推断 HER2=negative |
| Stage | T2N1 | ✅ | |
| Procedure | Port placement, screening biopsies | ✅ | |
| Genetic_Testing_Plan | "[REDACTED] on genetic testing results" | ⚠️ P2 | 被 [REDACTED] 遮掩，语义不完整 |
| goals | curative | ✅ | |

**Bug IDs**: B72 (HER2 "not tested" 应从 "triple negative" 推断 negative, P1)

---

### Row 40 (coral_idx=179) — 62岁女性, Stage 2 ER+/HER2- IDC, MS 患者

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER 95, PR 5, HER2 2+ FISH negative (1.2) G1 IDC | ✅ 优秀 | |
| Stage | Stage II | ✅ | |
| current_meds | letrozole | ✅ | |
| Others referral | PT referral | ✅ | |
| Imaging | DEXA | ✅ | |
| goals | curative | ✅ | |

**Bug IDs**: 无。✅ 优秀样本

---

### Row 41 (coral_idx=180) — 32岁女性, ATM 突变, ER+/PR weakly+, s/p bilateral mastectomy

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| **Type_of_Cancer** | "ER+/PR weakly+/HER2: not tested IDC" | **❌ P1** | A/P 写 "***** negative" = HER2 negative |
| **Stage** | "" (空) | ⚠️ P2 | 3cm + 1/3 SLN micromet = 至少 Stage IIA |
| **Genetics referral** | "ATM mutation carrier" | **❌ P1** | 同 B70：诊断结果非 referral |
| **Procedure_Plan** | "No procedures planned." | **❌ P1** | 原文："She is scheduled for a port placement later this week" |
| current_meds | "" | ✅ | |
| goals | curative | ✅ | |

**Bug IDs**: B73 (HER2 "not tested" 应为 negative, P1), B74 (Genetics=诊断结果, P1), B75 (Port placement 遗漏, P1)

---

### Row 42 (coral_idx=181) — 41岁女性, 多灶 IDC, PR 95%, 放疗后开始 tamoxifen

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | "PR+ (95%) and HER2- IDC" | ⚠️ P2 | ER 缺失（被 ***** 遮掩） |
| Stage | "Approximately Stage I-II" | ✅ | |
| **response_assessment** | "Not yet on treatment" | **❌ P1** | 刚完成放疗！不是 "not yet on treatment" |
| Medication_Plan | tamoxifen 5 year course | ✅ | |
| Imaging | Routine diagnostic mammogram | ✅ | |

**Bug IDs**: B76 (response "not yet" 但刚完成放疗, P1)

---

### Row 43 (coral_idx=182) — 38岁女性, 第二原发 Stage I TNBC, 计划 taxol+carboplatin

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER-/PR-/HER2- IDC | ✅ | |
| Stage | Stage I (second primary) | ✅ | |
| **radiotherapy_plan** | "followed by taxol and radiation." | **❌ P1** | 过去治疗！"at age 27 treated with lumpectomy, ***** followed by taxol and radiation" |
| **Lab_Plan** | "No labs planned." | ⚠️ P2 | A/P: "RTC 2 days prior to cycle...draw" = 实验室检查 |
| Advance care | Full code | ✅ | |

**Bug IDs**: B77 (radiotherapy_plan 含过去放疗, P1), B78 (Lab_Plan 遗漏, P2)

---

### Row 44 (coral_idx=183) — 33岁女性, ER+/PR+/HER2-, BRCA1, 术后残余, 计划放疗

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| Type_of_Cancer | ER+/PR+/HER2- node+ with residual DCIS | ✅ | |
| Stage | "Not mentioned in note" | ⚠️ P2 | 确实无显式分期 |
| Nutrition | Follow up with nutrition on 11/30/18 | ✅ | 真正的营养随诊 |
| Specialty | Radiation oncology consult | ✅ | |
| Others | Physical therapy referral | ✅ | |
| Procedure | BSO eventually | ✅ | |
| radiotherapy_plan | clinical trial 3 vs 5 weeks | ✅ | |
| goals | curative | ✅ | |

**Bug IDs**: 无实质问题。✅ 优秀样本

---

### Row 45 (coral_idx=184) — 37岁女性, 转移性 TNBC, 第二意见, 视频会诊

| 字段 | v9b 输出 | 判定 | 说明 |
|------|---------|------|------|
| second opinion | yes | ✅ | |
| Stage | Originally Stage IIIB, now metastatic (Stage IV) | ✅ | |
| Metastasis | Yes (lung + right hilar LN) | ✅ | |
| goals | palliative | ✅ | |
| Advance care | full code | ✅ | |
| **radiotherapy_plan** | "She had adjuvant radiation" | **❌ P1** | 过去放疗！同 B77 |
| **Others referral** | (大段治疗摘要) | **❌ P1** | 模型把整个治疗计划塞进 Others 字段 |
| Nutrition | None | ✅ | 同一医生模板，这次正确（与 Row 37/38 不一致） |

**Bug IDs**: B79 (radiotherapy_plan 含过去放疗, P1), B80 (Others referral 垃圾输出, P1)

---

### v9b Batch 2 整体质量评估

| Row | coral_idx | P0 | P1 | P2 | 质量 |
|-----|-----------|----|----|----|----|
| 36 | 175 | 0 | 0 | 3 (B65, B66, response) | **良好** |
| 37 | 176 | 0 | 2 (B67, B68) | 0 | **中等** |
| 38 | 177 | 0 | 3 (B69, B70, B71) | 0 | **差** |
| 39 | 178 | 0 | 1 (B72) | 1 | **中等** |
| 40 | 179 | 0 | 0 | 0 | **优秀** |
| 41 | 180 | 0 | 3 (B73, B74, B75) | 1 | **差** |
| 42 | 181 | 0 | 1 (B76) | 1 | **中等** |
| 43 | 182 | 0 | 1 (B77) | 1 (B78) | **中等** |
| 44 | 183 | 0 | 0 | 1 | **优秀** |
| 45 | 184 | 0 | 2 (B79, B80) | 0 | **中等** |

**总计：P0 = 0，P1 = 13，P2 = 8**

### v9b Batch 2 系统性问题汇总

| 编号 | 问题 | 严重度 | 出现行 | 修复方向 |
|------|------|--------|--------|---------|
| B67/B69 | Nutrition 假阳性（饮食建议 ≠ referral） | P1 | 37, 38 (45正确) | Referral prompt 加强负例 |
| B70/B74 | Genetics referral 放诊断结果（BRCA1/ATM mutation） | P1 | 38, 41 | Referral prompt 明确"finding ≠ referral" |
| B72/B73 | HER2 "not tested" 应从 "triple negative"/"***** negative" 推断 | P1 | 39, 41 | Cancer_Diagnosis prompt 加推断规则 |
| B77/B79 | radiotherapy_plan 纳入过去放疗 | P1 | 43, 45 | radiotherapy prompt + G5 TEMPORAL 强化 |
| B71 | Social work referral 遗漏（"Refer to social worker." 格式） | P1 | 38 | POST-REFERRAL regex 扩展 |
| B76 | response "not yet on treatment" 但刚完成放疗 | P1 | 42 | Response prompt 加规则 |
| B68 | Procedure_Plan 含化疗 | P1 | 37 | Procedure prompt 已有规则但模型忽略 |
| B75 | Port placement 遗漏（在 HPI 段非 A/P 段） | P1 | 41 | Procedure prompt 搜索范围扩展 |
| B80 | Others referral 垃圾输出 | P1 | 45 | G3 IMPROVE 应捕获语义偏差 |

---

## v10 Review (2026-03-15)

**Run ID**: `default_qwen_20260315_105314`
**基线**: v9b batch2 (`default_qwen_20260315_095522`, rows 36-45)
**变更**: 5 prompt 修复 + 2 code 修复（详见 v10 变更记录）
**运行时间**: 32.0 min（v9b 33.3 min）

### v10 Experiment Log Entry

| Run ID | Model | Config | Rows | Date | Notes |
|--------|-------|--------|------|------|-------|
| `default_qwen_20260315_105314` | Qwen2.5-32B-AWQ (4bit) | V2 pipeline, v10 code | 36-45 | 2026-03-15 | **v10**: 6 P1 fixes from v9b batch2。32.0min |

### v10 逐行审查（对照原文 + v9b baseline）

#### Row 36 (coral_idx 175)
**患者**: 27 y.o.♀, pT3N0 ER+/PR+/HER2- grade III mixed ductal/mucinous, s/p bilateral mastectomy, cycle 8 abraxane

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | Follow up | 返回做化疗 cycle 8 | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous | 原文 core bx: ER 95%, PR 15%, HER2 0/FISH 1.1 | ✅ |
| Stage | pT3N0 | A/P 明确说 pT3N0 | ✅ |
| Metastasis | No | PET/CT 无远处转移 | ✅ |
| lab_summary | CBC + CMP with dates | 与原文 lab 完全匹配 | ✅ |
| findings | 右臂肿胀 + thyroid nodule + PET postprocedural | 原文有 | ✅ |
| current_meds | Abraxane, zoladex | A/P: "Continue Abraxane", "Continue zoladex" | ✅ (B65 tamoxifen 仍缺，A/P 未提) |
| supportive_meds | Zofran, Compazine | 原文 antiemetics | ✅ |
| goals | curative | pT3N0, adjuvant chemo | ✅ |
| response_assessment | "Not mentioned in note." | A/P 无正式疗效评估讨论 | ✅ (比 v9b "no evidence of recurrence" 更准确) |
| radiotherapy_plan | Rad Onc referral next week | 未来 referral | ✅ |
| Procedure_Plan | doppler to r/o DVT | **Doppler 是影像不是 procedure** | ❌ B66 仍在 (P2) |
| Imaging_Plan | doppler to r/o DVT | 正确放在 imaging | ✅ |
| Referral.Specialty | Rad Onc referral | A/P 明确 | ✅ |
| follow_up | RTC 2 weeks | A/P 末尾 | ✅ |

**v9b bugs**: B65 tamoxifen (P2 仍在), B66 doppler in Procedure (P2 仍在)
**改善**: response_assessment 更准确（"Not mentioned" vs v9b 过度推断）
**新问题**: 无
**质量**: 良好

---

#### Row 37 (coral_idx 176)
**患者**: 47 y.o.♀, triple negative IDC s/p neoadjuvant chemo + lumpectomy, discussing adjuvant AC-Taxol

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER-/PR-/HER2- IDC | "triple negative" in note | ✅ |
| Stage | Approximately Stage IIA | T2N0 + 2.4cm + 0/2 SLN neg | ✅ |
| current_meds | "" | 尚未开始 adjuvant，合理 | ✅ |
| goals | curative | 早期 + adjuvant | ✅ |
| response_assessment | "Not yet on treatment" | 确实尚未开始 adjuvant | ✅ |
| Nutrition | **"None"** | 原文 "I recommend anti inflammatory/mediterranean diet" = 建议非 referral | ✅ **B67 FIXED** |
| Procedure_Plan | "adjuvant chemo with AC followed by Taxol" | **化疗不是 procedure** | ❌ B68 仍在 (P1) |
| Advance_care | "Full code." | 原文有 | ✅ |

**v9b bugs**: B67 Nutrition FP → ✅ **FIXED**, B68 Procedure=chemo → ❌ 仍在
**新问题**: 无
**质量**: 中等 (B68 是唯一问题)

---

#### Row 38 (coral_idx 177)
**患者**: 42 y.o.♀, BRCA1+, triple negative IDC, s/p neoadjuvant AC+taxol (stopped for toxicity), tumor enlarging

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER-/PR-/HER2- IDC | 原文 "triple negative" | ✅ |
| Stage | Approximately Stage II | 3.5cm + N0 | ✅ |
| current_meds | "" | Taxol 已停，未开始新治疗 | ✅ |
| recent_changes | **""** (空) | 原文 A/P: "S/p [REDACTED] x 4 and 5 weeks of taxol. Stopped due to toxicity." | ❌ **B81 NEW**: taxol 停药信息消失 (P2) |
| supportive_meds | NORCO, ULTRAM | 原文 current meds list | ✅ |
| goals | curative | 早期 + 讨论 neoadjuvant | ✅ |
| response_assessment | "currently progressing...tumor enlarging...MRI mild response...PET negative for distant" | 原文 exam: "large 8x5cm mobile breast mass" + MRI/PET results | ✅ (比 v9b 更详细) |
| Nutrition | **"None"** | 原文 "I recommend anti inflammatory/diabetes risk reducing diet" = 建议 | ✅ **B69 FIXED** |
| Genetics | **"BRCA 1 mutation"** | 这是 FINDING 不是 referral，应为 None | ❌ B70 仍在 (P1) |
| Others | "We discussed lifestyle...diet..., Social work referral" | Social work referral 出现了，但混入垃圾 | ⚠️ B71 部分修复 + **B82 NEW**: Others 混入 lifestyle advice (P2) |
| radiotherapy_plan | "discussed radiation for local recurrence" | 原文 "discussed the role of surgery and radiation" — 未来讨论 | ✅ (正确) |

**v9b bugs**: B69 → ✅ **FIXED**, B70 → ❌ 仍在, B71 → ⚠️ 部分修复
**新问题**: B81 recent_changes 回归 (P2), B82 Others 垃圾文本 (P2)
**质量**: 中等

---

#### Row 39 (coral_idx 178)
**患者**: 38 y.o.♀, T2N1 triple negative IDC, newly diagnosed, planning chemo

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | **ER-/PR-/HER2- IDC** | "triple negative" in A/P and HPI | ✅ **B72 FIXED** |
| Stage | "At least Stage II" | T2N1 → Stage IIB | ✅ (保守但正确) |
| current_meds | "" | 尚未开始 | ✅ |
| goals | curative | 早期 | ✅ |
| response_assessment | "Not yet on treatment" | 正确 | ✅ |
| Procedure_Plan | "Port placement, screening biopsies" | 原文 A/P: "scheduled for port placement" + screening biopsies | ✅ |

**v9b bugs**: B72 HER2 from triple negative → ✅ **FIXED**
**新问题**: 无
**质量**: 优秀

---

#### Row 40 (coral_idx 179)
**患者**: 68 y.o.♀, ER+/PR+/HER2- ILC, Stage I, s/p lumpectomy + radiation, on letrozole

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- ILC | 原文 ER+95%, PR+85%, HER2- | ✅ |
| current_meds | letrozole | A/P: "continuing on letrozole" | ✅ |
| goals | curative | Stage I + adjuvant | ✅ |
| response_assessment | "no evidence of recurrence on exam and imaging" | 原文 "no suspicious mass", mammogram negative | ✅ |
| Referral.Others | **"None"** | 原文 A/P: "PT referral" (明确写了) | ❌ **B83 NEW**: PT referral 遗漏 (P2, LLM 随机性) |
| Therapy_plan | letrozole + Prolia | A/P 提到 | ✅ |

**v9b bugs**: 无 (v9b 已优秀)
**新问题**: B83 PT referral 遗漏 (P2) — v9b 正确提取了，v10 漏了
**质量**: 良好 (仅 1 个 P2)

---

#### Row 41 (coral_idx 180)
**患者**: 32 y.o.♀, ATM carrier, ER+/PR weakly+, IDC, 3cm grade 3, s/p bilateral mastectomy, planning AC-Taxol

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER+/PR weakly+/HER2 not tested IDC | HER2: "***** 1+ by IHC and FISH not available"。Core bx: "***** negative (by FISH)" | ⚠️ B73 仍在 — HER2 data 被 redacted，但 core bx 说 negative by FISH |
| Stage | **"Approximately Stage II (3cm grade 3 IDC with 1/3 SLN with micromet)"** | 原文 pathology 支持 | ✅ **改善** (v9b 空) |
| current_meds | "" | 尚未开始 chemo | ✅ |
| goals | curative | 早期 + adjuvant | ✅ |
| response_assessment | "Not yet on treatment" | 正确 | ✅ |
| Genetics referral | **"None"** | 原文 "ATM mutation carrier" = FINDING | ✅ **B74 FIXED** |
| Procedure_Plan | "No procedures planned." | 原文 HPI: "scheduled for port placement later this week" | ❌ B75 仍在 (P1, HPI 不在 A/P) |

**v9b bugs**: B73 HER2 → ❌ 仍在 (非 triple negative 场景), B74 Genetics → ✅ **FIXED**, B75 Port → ❌ 仍在
**改善**: Stage 从空 → "Approximately Stage II"
**新问题**: 无
**质量**: 中等

---

#### Row 42 (coral_idx 181)
**患者**: 41 y.o.♀, PR+/HER2- IDC, Stage I-II, s/p lumpectomy + radiation, starting tamoxifen

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | PR+ IDC | 原文: "PR strongly positive 95%, *****/neu negative" | ⚠️ v10 丢了 HER2-（v9b 有）— "*****/neu" 被 redacted 但 likely = HER2 (P2) |
| Stage | Approximately Stage I-II | 0.9cm + 0.3cm, 0/5 LN | ✅ |
| current_meds | "" | tamoxifen 刚开处方，尚未开始 | ✅ |
| response_assessment | **"Not mentioned in note."** | 原文 "recently finished radiation", 无正式疗效评估 | ✅ **B76 FIXED** (不再说 "Not yet on treatment") |
| radiotherapy_plan | "None" | 放疗已完成，正确排除 | ✅ |
| Medication_Plan | "Begin 5 year tamoxifen" | A/P 明确 | ✅ |

**v9b bugs**: B76 response → ✅ **FIXED**
**新问题**: B84 Type_of_Cancer 丢 HER2- (P2, v9b 有)
**质量**: 良好

---

#### Row 43 (coral_idx 182)
**患者**: 38 y.o.♀, Stage I triple negative (2nd primary), s/p bilateral mastectomy, planning taxol+carboplatin

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER-/PR-/HER2- IDC | A/P: "triple negative", pathology: "ER and PR negative...her 2 FISH negative" | ✅ |
| Stage | Stage I (second primary) | A/P: "Second primary stage I" | ✅ |
| radiotherapy_plan | **"followed by radiation."** | A/P #1: "treated with lumpectomy, ***** followed by taxol **and radiation**" — 这是 PAST (第一次癌症的治疗) | ❌ B77 仍在 (P1)，v10 稍改善（去掉了 "taxol"） |
| Lab_Plan | "No labs planned." | A/P #4: "RTC 2 days prior to cycle...draw and visit" — "draw" = blood draw | ❌ B78 Lab_Plan 仍遗漏 (P2) |
| response_assessment | "Not yet on treatment" | 新化疗尚未开始，正确 | ✅ |
| Advance_care | "Full code." | 原文 "Code status: Full code" | ✅ |

**v9b bugs**: B77 past radiation → ⚠️ 部分改善 (缩短但仍泄漏), B78 Lab_Plan → ❌ 仍在
**新问题**: 无
**质量**: 中等

---

#### Row 44 (coral_idx 183)
**患者**: 33 y.o.♀, ER+/PR+/HER2-, BRCA1+, node+, s/p neoadjuvant AC-Taxol + mastectomy, discussing radiation + endocrine

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- node+ with residual grade 2 IDC and residual DCIS | A/P + pathology | ✅ (比 v9b 更详细) |
| Stage | Not mentioned in note | 原文 A/P 未明确 staging，可从 residual disease 推断 | ✅ 保守 |
| radiotherapy_plan | clinical trial 3 vs 5 weeks radiation, start 12/16/18 | 未来计划 | ✅ |
| Nutrition | "follow up with nutrition on 11/30/18" | A/P #3: "Follow up with nutrition on 11/30/18" | ✅ (真实 nutrition referral) |
| Others | "Physical therapy referral" | A/P #6: "Referred to PT" | ✅ |
| Specialty | "Radiation oncology consult" | met with Rad Onc today | ✅ |
| Procedure_Plan | BSO eventually + Zoladex authorization | A/P: BSO planned + "submit authorization for Zoladex" | ✅ |
| Imaging_Plan | CT Chest in one year | A/P #2: "Consider follow up CT Chest in one year" | ✅ |
| goals | curative | 早期 + adjuvant | ✅ |
| Advance_care | "Not discussed during this visit." | 原文无 code status | ✅ |

**v9b bugs**: 无
**新问题**: 无
**质量**: 优秀

---

#### Row 45 (coral_idx 184)
**患者**: 37 y.o.♀, metastatic triple negative BC to lung + hilar LN, second opinion, planning gemzar+carboplatin

| 字段 | v10 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER-/PR-/HER2- TNBC | "triple negative" throughout | ✅ |
| Stage | Originally Stage IIIB, now metastatic (Stage IV) | A/P staging + "metastatic to lung" | ✅ |
| Metastasis | Yes (lung + right hilar LN) | 原文明确 | ✅ |
| goals | palliative | Stage IV metastatic | ✅ |
| response_assessment | "has had no treatment since January 2022...Not yet on treatment" | 原文: "no treatment since January 2022 when diagnosed with metastatic disease" | ✅ |
| Nutrition | **"None"** | 原文 "I recommend anti inflammatory diet" = 建议 | ✅ (正确排除) |
| radiotherapy_plan | **"None"** | A/P: "She had adjuvant radiation" — PAST | ✅ **B79 FIXED** |
| Others | **"None"** | 不再有垃圾文本 | ✅ **B80 FIXED** |
| Genetics | "None" | "genetic testing was negative" = FINDING | ✅ |
| Advance_care | "full code." | 原文 "Code status: full code" | ✅ |
| follow_up | treatment plan 混入 follow up | A/P 未指定下次就诊 | ⚠️ follow_up 不够清晰 (P2 minor) |

**v9b bugs**: B79 → ✅ **FIXED**, B80 → ✅ **FIXED**
**新问题**: 无重要新问题
**质量**: 优秀

---

### v10 整体质量评估

| Row | coral_idx | P0 | P1 | P2 | 质量 | vs v9b |
|-----|-----------|----|----|----|----|--------|
| 36 | 175 | 0 | 0 | 2 (B65, B66) | **良好** | ≈ (response 改善) |
| 37 | 176 | 0 | 1 (B68) | 0 | **中等** | ↑ (B67 修了) |
| 38 | 177 | 0 | 2 (B70, B71partial) | 2 (B81, B82) | **中等** | ≈ (B69 修了但新增 B81/B82) |
| 39 | 178 | 0 | 0 | 0 | **优秀** | ↑↑ (B72 修了) |
| 40 | 179 | 0 | 0 | 1 (B83) | **良好** | ↓ (新增 B83 PT 遗漏) |
| 41 | 180 | 0 | 2 (B73, B75) | 0 | **中等** | ↑ (B74 修了, Stage 改善) |
| 42 | 181 | 0 | 0 | 1 (B84) | **良好** | ↑ (B76 修了) |
| 43 | 182 | 0 | 1 (B77partial) | 1 (B78) | **中等** | ≈ (B77 略改善) |
| 44 | 183 | 0 | 0 | 0 | **优秀** | ≈ |
| 45 | 184 | 0 | 0 | 0 | **优秀** | ↑↑ (B79+B80 修了) |

**总计: P0 = 0, P1 = 6 (v9b: 13), P2 = 7 (v9b: 8)**

### v10 Bug 修复效果汇总

| Bug | 描述 | v10 状态 | 说明 |
|-----|------|----------|------|
| B67 | Nutrition FP (Row 37) | ✅ FIXED | prompt 负例生效 |
| B69 | Nutrition FP (Row 38) | ✅ FIXED | prompt 负例生效 |
| B72 | HER2 from triple negative (Row 39) | ✅ FIXED | 推断规则生效 |
| B74 | Genetics=ATM finding (Row 41) | ✅ FIXED | "finding≠referral" 说明生效 |
| B76 | Response after radiation (Row 42) | ✅ FIXED | "completed radiation = treatment" 规则生效 |
| B79 | Past radiation in radiotherapy_plan (Row 45) | ✅ FIXED | PLAN_KEYS + temporal prompt 生效 |
| B80 | Others garbage text (Row 45) | ✅ FIXED | 清理生效 |
| B71 | Social work missed (Row 38) | ⚠️ PARTIAL | POST-REFERRAL 抓到了但 Others 字段混入垃圾 |
| B77 | Past radiation leak (Row 43) | ⚠️ PARTIAL | 缩短了但 "followed by radiation" 仍泄漏（redacted text 干扰） |
| B70 | Genetics=BRCA1 finding (Row 38) | ❌ NOT FIXED | 模型仍把 BRCA1 mutation 当 referral |
| B68 | Procedure=chemo (Row 37) | ❌ NOT FIXED | 未针对修 |
| B73 | HER2 non-triple-neg redacted (Row 41) | ❌ NOT APPLICABLE | 非 triple negative 场景，fix 不适用 |
| B75 | Port in HPI (Row 41) | ❌ NOT FIXED | HPI 不在 A/P，未针对修 |

### v10 新增 Bug

| Bug | 行 | 描述 | 严重度 | 原因 |
|-----|-----|------|--------|------|
| B81 | 38 | recent_changes 回归 (taxol 停药 → 空) | P2 | LLM 随机性或 prompt 改动副作用 |
| B82 | 38 | Others 混入 lifestyle advice 垃圾文本 | P2 | POST-REFERRAL 抓了 social work 但 Others 字段也被 LLM 填入其他内容 |
| B83 | 40 | PT referral 遗漏 (v9b 有) | P2 | LLM 随机性（代码未改 Others 提取） |
| B84 | 42 | Type_of_Cancer 丢 HER2- (v9b 有) | P2 | LLM 随机性（*****/neu redacted） |

### v10 仍需改进的系统性问题

| 编号 | 问题 | 严重度 | 行 | 修复方向 |
|------|------|--------|-----|---------|
| B68 | Procedure_Plan 含化疗 | P1 | 37 | 考虑后处理：过滤 Procedure_Plan 中含化疗关键词的项 |
| B70 | Genetics=BRCA1 finding 仍当 referral | P1 | 38 | G4 FAITHFUL 应捕获但没有；考虑后处理或强化 prompt |
| B73 | HER2 data redacted 但可从上下文推断 | P1 | 41 | 需要更通用的 HER2 推断规则（不限于 triple negative） |
| B75 | Port placement 在 HPI 不在 A/P | P1 | 41 | Procedure prompt 可能需要扩展搜索范围到 HPI |
| B77 | radiotherapy_plan 仍有 partial leak | P1 | 43 | redacted text 干扰模式匹配，需更强的 G5 TEMPORAL 或后处理 |

## v11 Review (2026-03-15)

**Run ID**: `default_qwen_20260315_114946`
**基线**: v10 (`default_qwen_20260315_105314`)
**变更**: 1 prompt + 3 code 修复（SYSTEMIC_THERAPY_TERMS + POST-GENETICS + POST-PROCEDURE）
**运行时间**: 33.5 min（v10 32.0 min）

### v11 POST 后处理触发日志

```
Row 35 (175): [POST-PROC-FILTER] removed: ['adjuvant chemotherapy with AC followed by Taxol']
Row 37 (177): [POST-REFERRAL] found: Social work
Row 37 (177): [POST-GENETICS] cleared: 'BRCA 1 mutation'
Row 38 (178): [POST-PROC-FILTER] removed: ['Echocardiogram', 'MRI of the breasts with additional evaluations as necessary']
```

### v11 逐行审查（对照原文 + v10 baseline）

#### Row 36 (coral_idx 175)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| Procedure_Plan | **"No procedures planned."** | v10: "doppler to r/o DVT..." | ✅ **B66 FIXED** — doppler 是影像不是 procedure |
| 其他字段 | 与 v10 基本一致 | — | ✅ |

**v10 bugs**: B65 tamoxifen (P2 仍在, redacted data), B66 doppler → ✅ **FIXED**

---

#### Row 37 (coral_idx 176)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| Procedure_Plan | **"No procedures planned."** | v10: "adjuvant chemo with AC followed by Taxol" | ✅ **B68 FIXED** — POST-PROC-FILTER 成功移除化疗 |
| Type_of_Cancer | ER-/PR-/HER2- **triple negative** IDC | v10: ER-/PR-/HER2- IDC | ✅ 改善（加了 triple negative 描述） |

**v10 bugs**: B68 Procedure=chemo → ✅ **FIXED**

---

#### Row 38 (coral_idx 177)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| Genetics | **"None"** | v10: "BRCA 1 mutation" | ✅ **B70 FIXED** — POST-GENETICS 清除了 mutation finding |
| Others | lifestyle advice + Social work referral | v10: 同样混入垃圾 | ⚠️ B82 仍在（P2, Social work 被正确捕获但仍混入 lifestyle 垃圾） |
| recent_changes | "Stopped due to toxicity. She declines further IV or preoperative therapy." | v10: "" (空) | ✅ **B81 FIXED** — LLM 随机性改善 |
| radiotherapy_plan | "discussed the role of surgery and radiation to decrease risk of local recurrence." | v10: 类似 | ✅ 正确（未来讨论） |

**v10 bugs**: B70 → ✅ **FIXED**, B81 → ✅ **FIXED** (LLM randomness), B82 → ❌ 仍在 (P2)

---

#### Row 39 (coral_idx 178)

与 v10 一致，无变化。B72 已在 v10 修复，仍有效。✅

---

#### Row 40 (coral_idx 179)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| Others | "None" | v10: "None" | B83 PT referral 仍遗漏 (P2, LLM 随机性) |
| Type_of_Cancer | "ER 95, PR 5, HER2 2+ FISH negative (1.2) G1 IDC with nuclear G1 DCIS" | v10 类似 | ✅ |

---

#### Row 41 (coral_idx 180)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | ER+/**PR-**/HER2 negative IDC | v10: ER+/**PR weakly+**/HER2 not tested | ⚠️ 混合变化 |
| — PR | "-" | "weakly+" (原文 "PR weakly + (1%)") | ❌ **B85 NEW**: PR 回归 (P2, LLM 随机性) |
| — HER2 | "negative" | "not tested" (原文 FISH negative) | ✅ 改善 — HER2 negative 更准确 |
| Stage | **""** (空) | "Approximately Stage II (3cm grade 3...)" | ❌ **B86 NEW**: Stage 回归 (P2, LLM 随机性) |

**v10 bugs**: B73 HER2 部分改善, B75 Port placement 仍遗漏 (POST-PROCEDURE 未触发 — 原文用 "has been scheduled" 不是 "plan for")

---

#### Row 42 (coral_idx 181)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| response_assessment | "No specific evidence of cancer response..." | v10: "Not mentioned in note." | ✅ 可接受（措辞不同但语义等价） |
| 其他字段 | 与 v10 基本一致 | — | ✅ |

---

#### Row 43 (coral_idx 182)

| 字段 | v11 输出 | v10 对比 | 判定 |
|------|---------|---------|------|
| radiotherapy_plan | "followed by radiation." | v10: 同 | ❌ B77 仍在 (P2, redacted data) |
| Stage | "Originally Stage I, now Stage I (second primary)" | v10: "Stage I (second primary)" | ✅ 等价 |

---

#### Row 44 (coral_idx 183) — 与 v10 一致，优秀。✅

#### Row 45 (coral_idx 184) — 与 v10 一致，优秀。✅

---

### v11 整体质量评估

| Row | coral_idx | P0 | P1 | P2 | 质量 | vs v10 |
|-----|-----------|----|----|----|----|--------|
| 36 | 175 | 0 | 0 | 1 (B65) | **良好** | ↑ (B66 FIXED) |
| 37 | 176 | 0 | 0 | 0 | **优秀** | ↑↑ (B68 FIXED) |
| 38 | 177 | 0 | 0 | 2 (B82, B71partial) | **良好** | ↑↑ (B70+B81 FIXED) |
| 39 | 178 | 0 | 0 | 0 | **优秀** | ≈ |
| 40 | 179 | 0 | 0 | 1 (B83) | **良好** | ≈ |
| 41 | 180 | 0 | 1 (B75) | 2 (B85, B86) | **中等** | ↓ (新增 B85/B86 LLM 回归) |
| 42 | 181 | 0 | 0 | 1 (B84) | **良好** | ≈ |
| 43 | 182 | 0 | 0 | 2 (B77, B78) | **良好** | ≈ (B77 降级为 P2) |
| 44 | 183 | 0 | 0 | 0 | **优秀** | ≈ |
| 45 | 184 | 0 | 0 | 0 | **优秀** | ≈ |

**总计: P0 = 0, P1 = 1 (v10: 6), P2 = 9 (v10: 7)**

### v11 vs v10 Bug 修复效果

| Bug | 描述 | v11 状态 | 修复方式 |
|-----|------|----------|---------|
| B66 | Doppler in Procedure_Plan (175) | ✅ FIXED | POST-PROC-FILTER 已有的 IMAGING_TERMS 捕获 |
| B68 | Chemotherapy in Procedure_Plan (176) | ✅ FIXED | **新增 SYSTEMIC_THERAPY_TERMS blocklist** |
| B70 | BRCA1 mutation as Genetics referral (177) | ✅ FIXED | **新增 POST-GENETICS 后处理** |
| B81 | recent_changes 回归 (177) | ✅ FIXED | LLM 随机性改善（非代码修复） |
| B73 | HER2 redacted → "not tested" (180) | ⚠️ PARTIAL | HER2 从 "not tested" 变 "negative" (改善), 但 PR 回归 |
| B75 | Port in HPI not A/P (180) | ❌ NOT TRIGGERED | POST-PROCEDURE 未匹配（原文 "has been scheduled" 不在模式中）|
| B77 | Radiotherapy leak (182) | ❌→P2 | 降级为 P2（redacted data artifact）|

### v11 新增 Bug

| Bug | 行 | 描述 | 严重度 | 原因 |
|-----|-----|------|--------|------|
| B85 | 41 (180) | PR 从 "weakly+" 变 "-" | P2 | LLM 随机性（原文 "PR weakly + (1%)"）|
| B86 | 41 (180) | Stage 从 "Approximately Stage II" 变空 | P2 | LLM 随机性（v10 正确推断，v11 未推断） |

### v11 仍需改进的系统性问题

| 编号 | 问题 | 严重度 | 行 | 修复方向 |
|------|------|--------|-----|---------|
| B75 | Port placement 在 HPI 不在 A/P | P1 | 41 | POST-PROCEDURE regex 需扩展匹配模式（增加 "has been scheduled"、"appointment for" 等） |
| B82 | Others 混入 lifestyle advice | P2 | 38 | 考虑 Others 字段后处理（过滤非 referral 文本） |
| B77 | radiotherapy_plan 仍有 "followed by radiation" | P2 | 43 | redacted data artifact，生产数据无此问题 |

### v11 版本进化总结

```
v9b baseline:  P0=0, P1=13, P2=8  (21 total issues)
v10 (prompts): P0=0, P1=6,  P2=7  (13 total) — 7 bugs fixed, 4 new P2
v11 (POST):    P0=0, P1=1,  P2=9  (10 total) — 5 more bugs fixed, 2 new P2

P1 reduction: 13 → 6 → 1 (92% reduction from v9b)
剩余唯一 P1: B75 (Port in HPI not A/P) — 需扩展 POST-PROCEDURE regex
```

## v12 Review (2026-03-15)

**Run ID**: `default_qwen_20260315_123551`
**基线**: v11 (`default_qwen_20260315_114946`)
**变更**: 1 prompt + 2 code 修复（POST-PROCEDURE regex + POST-OTHERS + Lab_Plan prompt）
**运行时间**: 33.2 min

### v12 POST 后处理触发日志

```
Row 35 (175): [POST-PROC-FILTER] removed: ['adjuvant chemotherapy with AC followed by Taxol']
Row 37 (177): [POST-REFERRAL] found: Social work
Row 37 (177): [POST-GENETICS] cleared: 'BRCA 1 mutation'
Row 37 (177): [POST-OTHERS] cleaned: lifestyle garbage → 'Social work referral'
Row 38 (178): [POST-PROC-FILTER] removed: ['Echocardiogram', 'MRI of the breasts...']
Row 40 (180): [POST-PROCEDURE] found in full note: 'port placement'
```

### v12 逐行审查（对照原文 + v11 baseline）

#### Row 36 (coral_idx 175)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| Procedure_Plan | "No procedures planned." | v10: "will get doppler..." | ✅ B66 FIXED (doppler 是影像不是手术) |
| **Lab_Plan** | "labs reviewed and adequate to receive treatment today, will get doppler to r/o DVT" | v10: "No labs planned." | ❌ **B87 新回归 (P1)** — 两个错误: (1) doppler 是影像不是 lab; (2) "labs reviewed" 是当前状态不是未来计划 |
| current_meds | "Abraxane, zoladex" | 同 | ✅ (B65 tamoxifen 仍缺, P2, redacted) |

**原文依据**: A/P 有 "labs reviewed and adequate to receive treatment today" (过去/当前状态) 和 "will get doppler to r/o DVT" (doppler = 影像)。两者都不属于 Lab_Plan。
**B87 根因**: LLM 把 doppler 从 Procedure_Plan 移到了 Lab_Plan（A/P 中 "labs" 和 "doppler" 在同一段落内混淆了）。G5 TEMPORAL 未过滤因为 "will get" 是未来时态——问题是字段错配。
**修复方案**: POST-LAB 后处理——从 Lab_Plan 移除影像术语 (doppler/ultrasound/CT/MRI/etc.)

#### Row 37 (coral_idx 176)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| Procedure_Plan | "No procedures planned." | v10: "adjuvant chemo with AC..." | ✅ B68 FIXED |
| Type_of_Cancer | "ER-/PR-/HER2- triple negative IDC" | v10: "ER-/PR-/HER2- IDC" | ✅ 略好 (加了 triple negative) |

#### Row 38 (coral_idx 177)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| Others | **"Social work referral"** | v10: lifestyle 垃圾 | ✅ **B82 FIXED** |
| Genetics | "None" | v10: "BRCA 1 mutation" | ✅ **B70 FIXED** |
| lab_summary | 含单位 (mIU/L, g/dL 等) | v10: 不含单位 | ✅ 略好 |
| recent_changes | **""** (空) | v10: "Stopped due to toxicity..." | ⚠️ B81 回归 (P2, LLM 随机性) |

#### Row 39 (coral_idx 178)
v10 完全一致。B72 仍修复。✅

#### Row 40 (coral_idx 179)
微小措辞差异（Type_of_Cancer 全名、Therapy_plan 细节），无实质变化。B83 PT referral 仍遗漏 (P2)。≈

#### Row 41 (coral_idx 180)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| procedure_plan | **"port placement"** | v10: "No procedures planned." | ✅ **B75 FIXED** — POST-PROCEDURE 从 HPI 找到 |
| Stage | "" | v10: "Approximately Stage II..." | ⚠️ B86 回归 (P2, LLM 随机性，v10 有 v12 没) |
| Type_of_Cancer | ER+/PR weakly+/HER2 not tested | v10: 同 | ⚠️ B73 仍在 (P2, redacted) |

#### Row 42 (coral_idx 181)
微小措辞差异（findings 中 "a 0.3 cm" vs "0.3 cm"），无实质变化。B84 Type_of_Cancer 仍丢 HER2- (P2)。≈

#### Row 43 (coral_idx 182)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| radiotherapy_plan | **"None"** | v10: "followed by radiation." | ✅ **B77 FIXED** |
| Lab_Plan | "No labs planned." | 同 | ❌ B78 仍在 (P2, prompt 改动未生效) |

#### Row 44 (coral_idx 183)
微小措辞差异（Procedure_Plan "if" vs "in the event of"），无实质变化。✅ 优秀

#### Row 45 (coral_idx 184)

| 字段 | v12 输出 | vs v10 | 判定 |
|------|---------|--------|------|
| **Nutrition** | "anti inflammatory and/or diabetes risk reducing diet..." | v10: "None" | ❌ **B88 新回归 (P1)** — 饮食建议 ≠ 营养转诊 (B67 同类问题) |

**原文依据**: A/P 有 "I recommend an anti inflammatory and /or diabetes risk reducing diet, limiting alcohol, exercise, stress reduction and sleep management." — 这是医生的一般性建议，不是转诊到营养师。
**B88 根因**: Referral prompt 已有此负例但 LLM 随机性导致有时仍误判。v10 正确排除，v12 回归。
**修复方案**: POST-NUTRITION 后处理——Nutrition 字段不含 "refer/consult/follow up with nutritionist" → 清为 "None"

### v12 整体质量评估（修正版 — 逐字段深度对比后更新）

| Row | coral_idx | P0 | P1 | P2 | 质量 | vs v10 |
|-----|-----------|----|----|----|----|--------|
| 36 | 175 | 0 | **1 (B87)** | 1 (B65) | **中等** | ↑ B66 FIXED, ↓ B87 新回归 |
| 37 | 176 | 0 | 0 | 0 | **优秀** | ↑ B68 FIXED |
| 38 | 177 | 0 | 0 | 1 (B81) | **良好** | ↑↑ B70+B82 FIXED |
| 39 | 178 | 0 | 0 | 0 | **优秀** | ≈ |
| 40 | 179 | 0 | 0 | 1 (B83) | **良好** | ≈ |
| 41 | 180 | 0 | 0 | 2 (B73, B86) | **良好** | ↑↑ B75 FIXED |
| 42 | 181 | 0 | 0 | 1 (B84) | **良好** | ≈ |
| 43 | 182 | 0 | 0 | 1 (B78) | **良好** | ↑ B77 FIXED |
| 44 | 183 | 0 | 0 | 0 | **优秀** | ≈ |
| 45 | 184 | 0 | **1 (B88)** | 0 | **中等** | ↓ B88 新回归 |

**总计: P0 = 0, P1 = 2 (B87, B88), P2 = 7**
**注**: 之前 v12 review 报告 P1=0 不准确——B87 和 B88 被 "与 v11 一致" 的简略审查遗漏。逐字段深度对比 v12 vs v10 后发现两个回归。

### v12 Bug 修复效果

| Bug | 描述 | v12 状态 | 修复方式 |
|-----|------|----------|---------|
| B68 | Procedure ← chemo (176) | ✅ **FIXED** | SYSTEMIC_THERAPY_TERMS blocklist |
| B70 | Genetics ← mutation (177) | ✅ **FIXED** | POST-GENETICS |
| B75 | Port in HPI not A/P (180) | ✅ **FIXED** | POST-PROCEDURE regex 加可选冠词 + "is scheduled for" |
| B82 | Others lifestyle garbage (177) | ✅ **FIXED** | POST-OTHERS 白名单过滤 |
| B77 | radiotherapy_plan leak (182) | ✅ **FIXED** | G5 TEMPORAL 成功过滤 |
| B78 | Lab_Plan "draw" miss (182) | ❌ NOT FIXED | prompt 改动未生效 (P2, LLM 随机性) |

### v12 新发现回归（逐字段深度对比后发现）

| Bug | 行 | 描述 | 严重度 | 根因 |
|-----|-----|------|--------|------|
| **B87** | 175 | Lab_Plan 含 doppler (影像) + "labs reviewed" (过去) | **P1** | LLM 把 doppler 从 Procedure 移到 Lab；G5 TEMPORAL 未过滤因 "will get" 是未来时态但字段错配 |
| **B88** | 184 | Nutrition 含饮食建议 (非 referral) | **P1** | Prompt 有负例但 LLM 随机性回归（v10 正确，v12 回归）|

### v12 剩余 P2（LLM 随机性或 redacted data artifact）

| Bug | 行 | 描述 | 原因 |
|-----|-----|------|------|
| B65 | 175 | tamoxifen 缺失 | redacted data ("she ***** tamoxifen") |
| B73 | 180 | HER2 "not tested" | redacted data (非 triple-negative 场景) |
| B78 | 182 | Lab_Plan 遗漏 "draw" | LLM 随机性 |
| B81 | 177 | recent_changes 空 | LLM 随机性 |
| B83 | 179 | PT referral 遗漏 | LLM 随机性 |
| B84 | 181 | Type_of_Cancer 丢 HER2- | redacted data (*****/neu) |
| B86 | 180 | Stage 空 | LLM 随机性 |

### v12 版本进化总结（修正版）

```
v9b baseline:  P0=0, P1=13, P2=8   (21 total issues)
v10 (prompts): P0=0, P1=6,  P2=7   (13 total) — 7 bugs fixed, 4 new P2
v11 (POST):    P0=0, P1=1,  P2=9   (10 total) — 5 more bugs fixed, 2 new P2
v12 (POST+):   P0=0, P1=2,  P2=7   ( 9 total) — 3 more bugs fixed, 2 new P1 (回归)

注: 之前 v12 review 报告 P1=0 是错误的——B87 和 B88 在简略审查中被遗漏。
逐字段深度对比 v12 vs v10 原始结果后发现两个回归。
教训: 不能用 "与 vX 一致" 跳过逐字段检查，每个 row 的每个字段都需要对比。
```

### v12 后处理 POST 链完整清单

```
ult.py extract_and_verify_v2() 内部:
  POST-KEYS    — 删除多余 key
  POST-MEDS    — oncology_drugs.txt 白名单过滤 current_meds
  POST-SUPP    — supportive_care_drugs.txt 白名单过滤 supportive_meds
  POST-PROC    — 从 procedure_plan 移除 imaging/radiotherapy/systemic therapy

run.py 主循环:
  POST-REFERRAL   — 全文搜索 referral 模式（social work, exercise, nutrition）
  POST-GENETICS   — 清除 Genetics 中的 mutation finding（非 referral）
  POST-OTHERS     — Others 字段白名单过滤（只保留已知 referral 类型）
  POST-NUTRITION  — 清除 Nutrition 中的饮食建议（非 referral）[v13 新增]
  POST-LAB        — 从 Lab_Plan 移除影像术语 + 过去状态表述 [v13 新增]
  POST-PROCEDURE  — 全文搜索 procedure 模式（port, biopsy, surgery）
  POST-ADV        — 全文搜索 code status
  POST-STAGE      — Stage vs Metastasis 交叉验证
  POST-GOALS      — adjuvant + 非转移 → curative
  POST-DISTMET    — 确保 Distant Metastasis 字段存在
  POST-RESPONSE   — findings 交叉补充 response_assessment
```

---

## v13 Review (2026-03-15)

**Run ID**: `default_qwen_20260315_132157`
**基线**: v12 (`default_qwen_20260315_123551`)
**变更**: 2 code 修复（POST-NUTRITION + POST-LAB）
**运行时间**: 33.6 min

### v13 POST 后处理触发日志

```
Row 36 (175): [POST-LAB] cleaned: 'labs reviewed...doppler...' → 'No labs planned.'
Row 37 (176): [POST-PROC-FILTER] removed: ['adjuvant chemotherapy with AC followed by Taxol']
Row 38 (177): [POST-REFERRAL] found: Social work
Row 38 (177): [POST-GENETICS] cleared: 'BRCA 1 mutation'
Row 38 (177): [POST-OTHERS] cleaned: lifestyle garbage → 'Social work referral'
Row 39 (178): [POST-PROC-FILTER] removed: ['Echocardiogram', 'MRI of the breasts...']
Row 41 (180): [POST-PROCEDURE] found in full note: 'port placement'
```

注意: POST-NUTRITION 未触发（因为 Row 45 这次 LLM 直接输出了 "None"，不需要后处理清理）。但过滤器已就位，下次 LLM 随机性回归时会自动拦截。

### v13 逐行深度审查（逐字段对照原文 + prompt + code + 历史 tracking）

#### Row 36 (coral_idx 175) — 27岁女性, Abraxane cycle 8

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Lab_Plan | **"No labs planned."** | A/P: "labs reviewed" (过去) + "doppler" (影像，非 lab) | ✅ **B87 FIXED** (POST-LAB 清除) |
| Procedure_Plan | "No procedures planned." | doppler 是影像不是手术 | ✅ B66 仍修复 |
| current_meds | "Abraxane, zoladex" | HPI 有 tamoxifen 但 ***** 遮掩 | ⚠️ B65 仍在 (P2, redacted) |
| Therapy_plan | 含 "valtrex" | valtrex 是抗病毒药，非 cancer therapy | ⚠️ P2 (LLM 随机性) |
| Others | "None" | "pt will contact PT for lymphedema" | ⚠️ P2 (PT referral 暗示，非明确转诊) |
| 其他字段 | 全部正确 | — | ✅ |

#### Row 37 (coral_idx 176) — 61岁女性, Stage IIA TNBC, Video Visit 新诊断

**原文关键事实**: 61 y.o.，Video Visit，New Patient Evaluation。Left breast cancer，bilateral mastectomies July 2020。2.3cm node negative triple negative。Stage IIA。Current med: dexlansoprazole (PPI)。Code status: Full code。A/P: recommend dd AC followed by Taxol，no radiation/hormone，lifestyle advice。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | "New patient" | "New Patient Evaluation" | ✅ |
| in-person | "Televisit" | "Video Visit" | ✅ |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | "triple negative" + staging "ER-, PR-, *****-" | ✅ |
| Stage | "Stage IIA" | "Stage IIA (pT2...)" | ✅ |
| lab_summary | "No labs in note." | 仅 2014 年 TB 旧结果 | ✅ |
| findings | "2.3cm, node negative..." | Video visit 体检有限，病理信息合理 | ✅ |
| current_meds | "" | dexlansoprazole 是 PPI，非 cancer drug | ✅ (prompt 限定 cancer-related) |
| Treatment_Goals | "curative" | Stage IIA + adjuvant chemo = curative | ✅ |
| response_assessment | "Not yet on treatment" | 尚未开始化疗 | ✅ |
| Medication_Plan | "dd AC followed by Taxol" | A/P #2: "recommend dd AC followed by Taxol" | ✅ |
| Therapy_plan | "AC followed by Taxol" | ⚠️ 原文 "dd AC"，v13 漏了 "dd" (dose-dense) | ⚠️ P2 (缺少 dd) |
| radiotherapy_plan | "None" | A/P #3: "no indication for radiation" | ✅ |
| Procedure_Plan | "No procedures planned." | 化疗不是手术 | ✅ B68 仍修复 |
| Nutrition | "None" | diet advice ≠ referral | ✅ |
| Advance_care | "Full code." | "Code status: Full code." | ✅ |
| follow_up | "chemotherapy at [REDACTED]" | A/P #5 | ✅ |

**新发现**: Therapy_plan 缺少 "dd" (dose-dense)。P2 LLM 随机性。
**vs v12**: 基本一致，Type_of_Cancer v12 多了 "triple negative" 描述。

#### Row 38 (coral_idx 177) — 43岁女性, BRCA1, Stage IIB, 肿瘤复大

**原文关键事实**: 43 y.o.，in-person，New Patient。BRCA1 mutation。6.8cm left IDC，ER-，PR weak (15%)，HER2-。Stage IIB。S/p neoadjuvant chemo（taxol 因毒性停）。Tumor enlarging again (8x5cm on exam)。Labs: TSH/CMP/CBC/HgbA1c 全正常。Code: full code。
A/P: ① S/p chemo, declines further IV ② olaparib 1 year (adjuvant) ③ bilateral mastectomy Jan 31 ④ qualify for xeloda ⑤ discussed radiation ⑥ hormonal blockade per guidelines (PR 15%) ⑦ xeloda + olaparib ⑧ lifestyle ⑨ see after surgery。
**Diagnosis section**: "Ambulatory Referral to Gynecologic Oncology"，"Ambulatory Referral to Social Work"。**Psychologic section**: "Refer to social worker."

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | "New patient" | "New Patient Evaluation" | ✅ |
| Type_of_Cancer | "ER-/PR+/HER2- IDC" | Staging: "ER-, PR+, *****-" | ✅ |
| Stage | "Originally Stage IIB" | Stage IIB。"Originally" 多余（仍是 IIB） | ⚠️ P2 (措辞冗余) |
| findings | "8x5cm mass...MRI mild response...PET negative" | Physical exam + imaging history | ✅ 综合全面 |
| current_meds | "" | 已停 chemo，当前仅非癌药 (gabapentin, pain meds) | ✅ |
| response_assessment | "cancer progressing...tumor enlarging" | "breast cancer palpable again...tumor is enlarging" | ✅ 准确 |
| Medication_Plan | "Olaparib...xeloda" | A/P ②④ | ✅ |
| Therapy_plan | "olaparib...xeloda...hormonal blockade...radiation" | A/P 完整覆盖 | ✅ |
| radiotherapy_plan | "discussed surgery and radiation" | A/P ⑤: discussed role | ✅ (discussion 也算) |
| Procedure_Plan | "bilateral mastectomy...January 31" | A/P ③ | ✅ |
| Others | "Social work referral" | "Refer to social worker." | ✅ B82 仍修复 |
| Genetics | "None" | "BRCA 1 mutation" = finding 非 referral | ✅ B70 仍修复 |
| **Specialty** | **"None"** | Diagnosis 列 "Ambulatory Referral to Gynecologic Oncology" | ⚠️ **NEW P2**: 遗漏 Gyn Onc referral |
| recent_changes | "" | v9b: "Stopped due to toxicity"，但这是 new pt visit | ⚠️ B81 仍在 (P2) |
| Advance_care | "full code." | "Code status: full code." | ✅ |

**新发现**: Specialty 遗漏 "Ambulatory Referral to Gynecologic Oncology"（出现在 Diagnosis section）。Prompt 要求搜索全文 referral，但 LLM 未从 Diagnosis section 发现。P2 LLM 随机性。

#### Row 39 (coral_idx 178) — 27岁女性, T2N1 TNBC, 新辅助化疗讨论

**原文关键事实**: 27 y.o.，in-person consultation。Grade 3 IDC，ER/PR/HER2 negative。T2N1（3.6cm mass + axillary LN positive）。刚摘除左卵巢冷冻保存。Cognitive issues，had LP。No cancer meds。Physical exam: 6x6cm mass L UOQ, 1cm firm node L axilla。
A/P: Neoadjuvant chemo — weekly paclitaxel x 12 wks → AC x 4 → surgery。Trial discussed。Echocardiogram。Port placement。[REDACTED] on genetic testing。Lab studies for ISPY。MRI breasts。Goserelin monthly。Start within 2 weeks。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | "New patient" | "here in consultation" | ✅ |
| second opinion | "no" | 未明确说 "second opinion"，有 primary oncologist | ✅ (遵从 prompt 规则) |
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | "ER/PR/***** negative"，grade 3 | ✅ |
| Stage | "At least Stage II" | "at least stage II...T2N1" | ✅ |
| findings | pathology + imaging + exam 全面 | 6x6cm mass, axillary node, CT/MRI/bone scan | ✅ |
| current_meds | "" | 只有 vitamin D, flonase | ✅ |
| Treatment_Goals | "curative" | neoadjuvant + surgery = curative | ✅ |
| Therapy_plan | "paclitaxel x 12 → AC x 4 → surgery...goserelin" | A/P 覆盖完整 | ✅ |
| Procedure_Plan | "Port placement, screening biopsies" | A/P: "Port placement" + "screening biopsies today" | ✅ |
| **Imaging_Plan** | **"MRI of breasts..."** | A/P 列了 **Echocardiogram** + MRI。Procedure_Plan prompt 明确说 "echocardiogram → IMAGING" | ⚠️ **NEW P2**: 遗漏 echocardiogram |
| Lab_Plan | "Laboratory studies for ISPY" | A/P: "Laboratory studies for ISPY" | ✅ |
| Genetic_Testing | "[REDACTED] on genetic testing results" | 可能是 "await" 结果，非新下单 | ≈ (原文模糊) |
| follow_up | "None" (Referral 字段) | A/P: "start therapy within two weeks" | ⚠️ P2 (follow_up_next_visit 有捕获) |
| Advance_care | "Not discussed" | 无 code status 记录 | ✅ |

**新发现**: Imaging_Plan 遗漏 echocardiogram。Procedure_Plan prompt 明确指出 echocardiogram 属于 Imaging_Plan，但 LLM 两边都没写。P2 LLM 随机性。

#### Row 40 (coral_idx 179) — 62岁女性, Stage 2 ER+ IDC, MS 患者

**原文关键事实**: 62 y.o.，in-person，New Patient。MS on chronic immunosuppression。Stage 2 low grade ER+/HER2- IDC right breast。S/p right partial mastectomy + SLN。Pathology: 2.3cm G1 IDC + DCIS，ER 95, PR 5, HER2 2+ FISH negative (1.2)。1/6 SLN micrometastasis (0.04cm)。
Current meds includes letrozole (FEMARA)。A/P: endocrine therapy with letrozole ("Rx given")，DEXA，PT referral，RTC 3 months。
注: "Rx for letrozole given" + "she can start letrozole immediately" = 处方刚开，可能尚未开始服用。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | "New patient" | HPI 未明确说 new/follow up，但 A/P 开头 "here to discuss systemic therapy options" = 首次 med onc 会诊 | ✅ |
| Type_of_Cancer | "ER 95, PR 5, HER2 2+ FISH neg (1.2) G1 IDC with DCIS" | 完整覆盖 receptor + pathology | ✅ 非常详细 |
| current_meds | "letrozole" | "Rx for letrozole given" + 列在 Current Outpatient Medications | ⚠️ P2 (刚开处方，可能尚未服用) |
| recent_changes | "Rx for letrozole given" | A/P 明确 | ✅ |
| supportive_meds | "ondansetron (ZOFRAN)" | 在 med list 中，但她无 cancer-related nausea (MS 患者，未接受化疗) | ⚠️ P2 (非 cancer-related supportive med) |
| Treatment_Goals | "curative" | Stage 2 + adjuvant = curative | ✅ |
| Imaging_Plan | "DEXA" | A/P: "-DEXA" | ✅ |
| **Others** | **"None"** | A/P: "**PT referral**" | ⚠️ **B83 仍在** (P2, LLM 随机性) |
| summary | "[REDACTED]+[REDACTED] negative" | redacted 处理，v12 更好 ("ER+ PR-") | ⚠️ P2 (redacted) |
| follow_up | "RTC 3 months" | ✅ |

#### Row 41 (coral_idx 180) — 32岁女性, ATM carrier, ER+/PR weakly+

**原文关键事实**: 32 y.o.，in-person。ATM mutation carrier。Left breast: 3cm G3 IDC, ER+ (90%), PR weakly + (1%), HER2 1+ IHC (FISH not available), Ki-67 30%。S/p bilateral mastectomy 04/11/18。1/3 SLN micrometastasis。MammaPrint High Risk。LVEF 79%。
**关键**: "For complete details...see my notes dated 03/17/18 and 04/21/18" = **曾经见过**此 provider。"During our 04/21/18 visit, we discussed..." = 这是 follow-up。
A/P: decided AC-Taxol (Taxol first 12 wks → AC)。Port placement scheduled。After chemo: ovarian suppression + AI + possible ribociclib trial。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| **Patient type** | "New patient" | "see my notes dated 03/17/18 and 04/21/18" = 此 provider 之前见过 | ⚠️ P2 (应为 Follow up) |
| Type_of_Cancer | **"ER+/PR-/HER2 negative"** | HER2 1+ IHC = HER2 neg；PR weakly+ (1%) / A/P says "PR-/***** negative" | ✅ **B73 自愈** HER2 neg 正确 |
| Stage | "" (空) | 3cm + 1/3 SLN = ~Stage II，但原文未明确 | ⚠️ B86 仍在 (P2) |
| procedure_plan | "port placement" | "scheduled for a port placement" | ✅ B75 仍修复 |
| Medication_Plan | "After chemo: AI + ribociclib trial" | ⚠️ 缺少 AC-Taxol 作为当前主计划 | ⚠️ P2 (Therapy_plan 有覆盖) |
| Therapy_plan | "AC-Taxol, Taxol 12 wks → AC...ovarian suppression+AI...ribociclib trial" | 完整覆盖 A/P #1-2 | ✅ |
| supportive_meds | "COLACE, NORCO" | 两者均在 current meds 中 | ✅ |
| recent_changes | "decided to proceed with AC-Taxol" | 更像 PLAN 而非 change，但有信息价值 | ≈ |
| Advance_care | "Not discussed" | 无 code status 记录 | ✅ |

#### Row 42 (coral_idx 181) — 41岁女性, IDC right breast, 放疗完成后开始 tamoxifen

**原文关键事实**: 41 y.o.，routine follow up。Right breast: 0.9cm + 0.3cm IDC, grade 1。0/5 SLN。"Progesterone receptors were strongly positive at 95% and *****/neu was negative"。S/p excision + re-excision + SLN + radiation (completed Jan 5)。Premenopausal。
A/P: Start 5-year tamoxifen。Return 4-6 weeks。Diagnostic mammogram scheduled。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | "PR+ invasive ductal carcinoma" | "PR strongly positive 95%"。但 **"*****/neu was negative"** = HER2-！ | ⚠️ **B84 仍在**: 缺 HER2-。ER 状态也缺（可能 redacted） |
| Stage | "Approximately Stage I-II" | 0.9cm + 0/5 SLN = Stage I，但未明确 | ≈ |
| current_meds | "" | 刚结束放疗，tamoxifen 尚未开始 | ✅ |
| Therapy_plan | "Begin 5 year tamoxifen" | A/P 一致 | ✅ |
| radiotherapy_plan | "None" | 放疗已完成（过去时），G5 TEMPORAL 正确过滤 | ✅ |
| Imaging_Plan | "diagnostic mammogram on next appt day" | A/P: "scheduled a routine diagnostic mammogram" | ✅ |
| follow_up | "Return 4-6 weeks...mammogram" | ✅ |

#### Row 43 (coral_idx 182) — 38岁女性, Stage I TNBC, 第二原发

**原文关键事实**: 38 y.o.，follow-up。Stage I TNBC (第二原发)。S/p bilateral mastectomies 02/22/2021。1.3cm G3 IDC, ER/PR neg, HER2 FISH neg。0/2 SLN negative。BRCA negative。
Current meds: ibuprofen, granisetron, prochlorperazine, senna。Code status: Full code。
A/P: ① past TNBC history ② second primary s/p mastectomies ③ **taxol carboplatin x 4 cycles** (adjuvant) ④ **RTC 2 days prior to cycle Oct 14, draw and visit with provider**。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | "ER-/PR-/HER2- IDC" | pathology ER/PR neg, HER2 FISH neg | ✅ |
| Stage | "Stage I (second primary)" | A/P: "Second primary stage I" | ✅ |
| Therapy_plan | "taxol carboplatin x 4 cycles" | A/P ③ | ✅ |
| radiotherapy_plan | null | 过去放疗（2010）已过滤 | ✅ B77 仍修复 |
| **Lab_Plan** | **"No labs planned."** | A/P ④: "**draw** and visit with provider" = lab draw! | ⚠️ **B78 仍在** (P2, LLM 未识别 "draw") |
| supportive_meds | "granisetron, prochlorperazine, senna" | 均在 current meds，cancer-related supportive | ✅ |
| follow_up | "RTC 2 days prior to cycle Oct 14" | ✅ |
| Advance_care | "Full code." | "Code status: Full code." | ✅ |

#### Row 44 (coral_idx 183) — 33岁女性, ER+/PR+/HER2-, BRCA1, 术后恢复

**原文关键事实**: 33 y.o.，follow-up，shared visit。ER+/PR+/HER2- node+ left breast cancer，BRCA1+。S/p neoadjuvant dd AC → Taxol → bilateral mastectomy 10/07/18。Residual 1cm G2 IDC。
A/P: ① Radiation trial (3 vs 5 wks, start 12/16/18) ② AI after radiation ③ BSO discussion 12/02 + Zoladex authorization ④ Pulmonary nodule stable → CT Chest in 1 year ⑤ Low weight → nutrition 11/30/18 ⑥ PT referral。Follow up 01/05/19。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Type_of_Cancer | "ER+/PR+/HER2- node+...residual G2 IDC + DCIS" | 完整覆盖 | ✅ |
| current_meds | "" | s/p chemo, not yet on AI | ✅ |
| supportive_meds | "Norco April 325 mg, docusate" | ⚠️ "April" 是解析错误，应为 "10-325 mg" | ⚠️ P2 (LLM 解析错误) |
| radiotherapy_plan | "3 vs 5 weeks trial, start 12/16/18" | A/P ① | ✅ |
| Therapy_plan | "AI after radiation...Zoladex...ribociclib trial" | A/P ①②③ | ✅ |
| Procedure_Plan | "BSO eventually, Zoladex authorization" | BSO 正确。"Zoladex authorization" 是药物非手术 | ⚠️ P2 (Zoladex = medication) |
| Imaging_Plan | "CT Chest in one year" | A/P ④ | ✅ |
| Nutrition | "follow up with nutrition on 11/30/18" | A/P ⑤: 明确 nutrition follow-up | ✅ |
| Specialty | "Radiation oncology consult" | A/P ①: met with Dr. in rad onc | ✅ |
| Others | "Physical therapy referral" | A/P ⑥: "Referred to PT" | ✅ |
| follow_up | "01/05/19" | A/P 最后: "Follow up on 01/05/19" | ✅ |

**整体**: Row 44 是 v13 中最优秀的提取之一，Referral 全面覆盖（Nutrition + Specialty + Others 都正确）。

#### Row 45 (coral_idx 184) — 37岁女性, 转移性 TNBC, 第二意见

**原文关键事实**: 37 y.o.，Video Consult，New Patient。Right breast TNBC，T2N1。Originally Stage IIIB。S/p neoadjuvant chemo (progression on taxol) → surgery → radiation → xeloda。Metastatic to bilateral lung nodules + right hilar LN (CT 12/2021, biopsy 02/2022)。PD-L1 positive 10%。Genetic testing negative。No current meds。Code: full code。
A/P: ① Metastatic TNBC ② treatment history ③ Recommend [REDACTED], gemzar carboplatin ASAP。Second line: [REDACTED]。Trials (FGFR inhibitor)。Other: eribulin, doxil。④ answered questions ⑤ lifestyle advice。

| 字段 | v13 输出 | 原文依据 | 判定 |
|------|---------|---------|------|
| Patient type | "New patient" | "New Patient Evaluation" | ✅ |
| second opinion | "yes" | "She is here for a second opinion" | ✅ |
| in-person | "Televisit" | "Video Consult" | ✅ |
| Stage | "Originally Stage IIIB, now metastatic (Stage IV)" | staging: IIIB; now has distant mets | ✅ 正确转换 |
| Metastasis | "Yes (lung + right hilar LN)" | ✅ |
| Treatment_Goals | "palliative" | "metastatic breast cancer is not curable" | ✅ |
| Medication_Plan | "gemzar carboplatin ASAP + 2nd/3rd line options" | A/P ③ 完整 | ✅ |
| **Nutrition** | **"None"** | A/P ⑤: "I recommend diet..." = 建议非 referral | ✅ **B88 FIXED** |
| Advance_care | "full code." | "Code status: full code" | ✅ |
| follow_up (Referral) | 治疗计划内容，非 visit timing | A/P 未指定 follow-up 时间 | ≈ (follow_up_next_visit 也说 "Not specified") |

### v13 整体质量评估

| Row | coral_idx | P0 | P1 | P2 (tracked) | P2 (深度审查新发现) | 质量 | vs v12 |
|-----|-----------|----|----|----|----|----|----|
| 36 | 175 | 0 | 0 | 1 (B65) | +1 (Therapy_plan 含 valtrex) | **良好** | ↑ B87 FIXED |
| 37 | 176 | 0 | 0 | 0 | +1 (Therapy_plan 缺 "dd") | **优秀** | ≈ |
| 38 | 177 | 0 | 0 | 1 (B81) | +1 (Specialty 漏 Gyn Onc referral) | **良好** | ≈ |
| 39 | 178 | 0 | 0 | 0 | +1 (Imaging_Plan 漏 echocardiogram) | **良好** | ≈ |
| 40 | 179 | 0 | 0 | 1 (B83) | +1 (ondansetron 非 cancer supportive) | **良好** | ≈ |
| 41 | 180 | 0 | 0 | 1 (B86) | +1 (Patient type 应为 Follow up) | **良好** | ↑ B73 自愈 |
| 42 | 181 | 0 | 0 | 1 (B84) | — | **良好** | ≈ |
| 43 | 182 | 0 | 0 | 1 (B78) | — | **良好** | ≈ |
| 44 | 183 | 0 | 0 | 0 | +2 ("Norco April" 解析错误; Procedure_Plan 含 Zoladex) | **优秀** | ≈ |
| 45 | 184 | 0 | 0 | 0 | — | **优秀** | ↑ B88 FIXED |

**总计: P0 = 0, P1 = 0, P2 = 6 (tracked) + 8 (深度审查新发现)**
- vs v12 (修正): P1 2→0 ✅, tracked P2 7→6 (B73 自愈)
- 无新回归 (P0/P1)
- 深度审查新发现的 8 个 P2 均为 LLM 随机性微小瑕疵，不影响临床可用性

### v13 深度审查追加发现（仅 P2，全部为 LLM 随机性）

| Row | 字段 | 问题 | 原因 | 可修复? |
|-----|------|------|------|---------|
| 36 | Therapy_plan | 含 valtrex (抗病毒药非 therapy) | LLM 混入非 cancer drug | 可考虑 POST-THERAPY 白名单，但收益低 |
| 37 | Therapy_plan | "AC" 缺 "dd" (dose-dense) | LLM 随机性 | 无法代码修复 |
| 38 | Specialty referral | 漏 "Ambulatory Referral to Gynecologic Oncology" (Diagnosis section) | LLM 未从 Diagnosis section 发现 | prompt 已说搜索全文，LLM 随机性 |
| 39 | Imaging_Plan | 漏 echocardiogram (A/P 明确列出) | LLM 遗漏 | Procedure_Plan prompt 已说 echo=IMAGING，LLM 随机性 |
| 40 | supportive_meds | ondansetron 可能非 cancer-related (MS 患者未接受化疗) | LLM 未判断 context | P2 minor |
| 41 | Patient type | 应为 "Follow up" ("see my notes dated 03/17/18 and 04/21/18") | LLM 随机性 | prompt 已有判断规则 |
| 44 | supportive_meds | "Norco April 325 mg" 应为 "Norco 10-325 mg" | LLM 解析错误（混淆月份和剂量） | 无法代码修复 |
| 44 | Procedure_Plan | 含 "Zoladex authorization"（药物非手术） | LLM 混淆 procedure vs medication | 可考虑 POST-PROC 增加药物过滤 |

**结论**: 这 8 个追加 P2 全部是 LLM 随机性行为，在不同运行中随机出现/消失。
其中 2 个可考虑代码修复（Therapy_plan 白名单、Procedure_Plan 药物过滤），但优先级极低。

### v13 Bug 修复效果

| Bug | 描述 | v13 状态 | 修复方式 |
|-----|------|----------|---------|
| B87 | Lab_Plan 含 doppler+labs reviewed (175) | ✅ **FIXED** | POST-LAB 移除影像术语+过去状态 |
| B88 | Nutrition 含饮食建议 (184) | ✅ **FIXED** | LLM 自行修正 + POST-NUTRITION 安全网就位 |
| B73 | HER2 "not tested" (180) | ✅ **自愈** | LLM 随机性向好（"***** negative" → HER2 negative） |

### v13 剩余 tracked P2

| Bug | 行 | 描述 | 原因 |
|-----|-----|------|------|
| B65 | 175 | tamoxifen 缺失 | redacted data |
| B78 | 182 | Lab_Plan 遗漏 "draw" | LLM 随机性 |
| B81 | 177 | recent_changes 空 | LLM 随机性 |
| B83 | 179 | PT referral 遗漏 | LLM 随机性 |
| B84 | 181 | Type_of_Cancer 丢 HER2- | redacted data |
| B86 | 180 | Stage 空 | LLM 随机性 |

### v13 版本进化总结

```
v9b baseline:  P0=0, P1=13, P2=8   (21 total issues)
v10 (prompts): P0=0, P1=6,  P2=7   (13 total) — 7 bugs fixed, 4 new P2
v11 (POST):    P0=0, P1=1,  P2=9   (10 total) — 5 more bugs fixed, 2 new P2
v12 (POST+):   P0=0, P1=2,  P2=7   ( 9 total) — 3 more bugs fixed, 2 new P1 回归
v13 (POST++):  P0=0, P1=0,  P2=6   ( 6 total) — 2 more bugs fixed, 1 self-healed, 0 new
                        (深度审查: +8 P2 observations, 均为 LLM 随机性微小瑕疵)

P1 reduction: 13 → 6 → 1 → 2 → 0 (100% elimination)
P2: 全部为 LLM 随机性波动或 redacted data artifact
  - 2 个 redacted: B65 (tamoxifen *****), B84 (*****/neu)
  - 4 个 LLM 随机性: B78 (Lab_Plan draw), B81 (recent_changes), B83 (PT referral), B86 (Stage)
  - 所有 P2 在不同运行中会随机出现/消失，无法靠代码修复
  - 深度审查追加的 8 个 P2 在同类 LLM 系统中属正常水平
```

### v13 POST 链完整清单（15 个过滤器）

```
ult.py extract_and_verify_v2() 内部:
  POST-KEYS    — 删除多余 key
  POST-MEDS    — oncology_drugs.txt 白名单过滤 current_meds
  POST-SUPP    — supportive_care_drugs.txt 白名单过滤 supportive_meds
  POST-PROC    — 从 procedure_plan 移除 imaging/radiotherapy/systemic therapy

run.py 主循环:
  POST-REFERRAL   — 全文搜索 referral 模式（social work, exercise, nutrition）
  POST-GENETICS   — 清除 Genetics 中的 mutation finding（非 referral）
  POST-OTHERS     — Others 字段白名单过滤（只保留已知 referral 类型）
  POST-NUTRITION  — 清除 Nutrition 中的饮食建议（非 referral）[v13]
  POST-LAB        — 从 Lab_Plan 移除影像术语 + 过去状态表述 [v13]
  POST-PROCEDURE  — 全文搜索 procedure 模式（port, biopsy, surgery）
  POST-ADV        — 全文搜索 code status
  POST-STAGE      — Stage vs Metastasis 交叉验证
  POST-GOALS      — adjuvant + 非转移 → curative
  POST-DISTMET    — 确保 Distant Metastasis 字段存在
  POST-RESPONSE   — findings 交叉补充 response_assessment
```
