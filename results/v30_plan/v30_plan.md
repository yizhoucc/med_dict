# V30 改进计划

> 基于 v29 full run review (61 samples, P0=0, P1=0, P2=92) 的发现

## 概述

v29 质量已经很好（零幻觉、零重大错误），但有 ~92 个 P2 小问题。分析 P2 模式后，确定 4 个可落地的改进点。

---

## 改进 1: response_assessment — 治疗状态判断

### 问题
v29 中 ~12 个 P2 来自 response_assessment：
- 患者尚未开始治疗时输出 "On treatment"（ROW 40, 49, 65, 72, 80, 82）
- 用旧影像评估当前治疗的反应（ROW 11, 91 — 影像在当前方案开始之前）
- 说 "not responding" 但 A/P 写 "good response"（ROW 46, 63）
- 说 "stable disease" 但影像显示 progression（ROW 84）

### 根因
prompt 已经有 "Has this patient started any cancer treatment yet?" 的 CoT 引导，但模型在以下场景仍犯错：
1. **刚开处方未服用**: 本次就诊开了 letrozole 处方 → 模型认为 "On treatment"
2. **旧影像归因**: 笔记引用了治疗方案切换前的 PET → 模型用旧 PET 评估新方案的反应
3. **A/P "Exam stable" 被忽略**: A/P 明确写了 "Exam stable" 但 response_assessment 只引用了旧影像的 PD 结论

### 修改方案

在 `prompts/extraction.yaml` 的 `Response_Assessment` prompt 中添加：

```
CRITICAL — "刚开处方" ≠ "On treatment":
- If a medication was PRESCRIBED or ORDERED at THIS visit (e.g., "Rx for letrozole given", "prescription ordered", "will start"), the patient has NOT yet started treatment.
- "On treatment" requires the patient to have ACTUALLY BEEN TAKING the medication for some time.
- Clues that treatment has NOT started: "will begin", "Rx given today", "instructed to start", "prescription ordered", "ok to start"

CRITICAL — Always check A/P assessment FIRST:
- If the A/P section contains an explicit assessment like "Exam stable", "doing well", "NED", or "no evidence of recurrence", prioritize THIS over historical imaging data.
- The A/P represents the physician's CURRENT assessment. Historical imaging in the note may be from BEFORE the current treatment regimen.
```

### 预期影响
- 消除 ~8-10 个 P2
- 特别是 pre-treatment consult 和 post-surgery new patient visit 场景

---

## 改进 2: Type_of_Cancer — 受体状态优先级

### 问题
v29 中 ~6 个 P2 来自受体状态新旧混用：
- 用原始 PR+ 但术后病理 PR-（ROW 34, 44）
- 用原始 HER2+ 但转移灶活检 HER2-（ROW 86）
- 转移灶 HER2 未检测但标为 HER2-（ROW 88）
- HER2 在笔记中被 [REDACTED] 但可从上下文推断为阴性，模型写 "not tested"（ROW 73, 91, 100）

### 根因
prompt 已有 "If receptor status changed between original diagnosis and metastatic/recurrent biopsy, note BOTH" 的规则，但：
1. 模型有时只报告一个（最旧或最新），而不是两个
2. 当 HER2 被 [REDACTED] 时，即使 A/P 行文中可以推断（如 "[redacted] negative"），模型仍写 "not tested"

### 修改方案

在 `prompts/extraction.yaml` 的 `Cancer_Diagnosis` prompt 中添加：

```
RECEPTOR STATUS PRIORITY — when multiple biopsies exist:
- If the patient has had MULTIPLE pathology results (core biopsy, surgical specimen, metastatic biopsy), the MOST RECENT pathology takes priority for current receptor status.
- Always report the CURRENT receptor status first, then note the original if different.
- Example: If core biopsy was ER+/PR+/HER2- but surgical specimen after neoadjuvant shows ER+/PR-/HER2-, write: "ER+/PR-/HER2- (post-neoadjuvant; originally PR+)"
- If metastatic biopsy shows different receptors from the primary, the metastatic biopsy is the CURRENT status.

REDACTED HER2 — inference rules:
- If the note says "[REDACTED] negative" in the A/P receptor status line, AND the treatment is endocrine-only (no anti-HER2 drugs), HER2 is NEGATIVE — do NOT write "not tested".
- If the Problem List says "ER+PR+ *****-", the "*****" is HER2 and the "-" means NEGATIVE.
- Only write "HER2: not tested" when there is genuinely NO mention of HER2 testing or results anywhere in the note AND no treatment context to infer from.
```

### 预期影响
- 消除 ~6 个 P2
- 更准确的受体状态报告，尤其是 post-neoadjuvant 和 metastatic biopsy 场景

---

## 改进 3: Letter — [REDACTED] 处理

### 问题
v29 中 ~6 个 P2 来自 letter 的 [REDACTED] garbled text：
- "a chemotherapy regimenaxol"（ROW 41 — "regimen" + "Taxol" 拼接）
- "a test called a medication"（ROW 52 — Oncotype 被 [REDACTED]）
- "results of a medication test"（ROW 97 — Oncotype Dx 被 [REDACTED]）
- "targeted therapy drugs chemotherapy"（ROW 53 — garbled）

另有 ~6 个 P2 来自 letter 的事实性错误：
- "HER2 是药物"（ROW 49）
- "makes mucus" 实为 squamous（ROW 66）
- jaw "grown larger" 但已放疗改善（ROW 11）
- "treatment not working" 用旧影像（ROW 91）

### 根因
letter 生成 prompt 在处理包含大量 [REDACTED] 的 keypoints 值时，无法生成有意义的文本，产出 garbled 内容。

### 修改方案

在 letter 生成 prompt 中添加：

```
HANDLING [REDACTED] CONTENT:
- If a field value is mostly [REDACTED] and you cannot determine what the original content means, SKIP that information rather than generating garbled text.
- Example: If genetic_testing_plan says "Order an [REDACTED] to provide more information", write "You will have a test to learn more about your cancer" — do NOT try to name the test.
- Example: If medication_plan says "[REDACTED] + [REDACTED]", write "You will start new medications as discussed with your doctor" — do NOT generate nonsensical drug names.
- NEVER concatenate partial words from [REDACTED] boundaries. If a word is partially redacted, skip it entirely.

FACTUAL ACCURACY IN PATIENT-FACING LANGUAGE:
- HER2 is a PROTEIN (receptor), NOT a medication. Write "a protein called HER2" not "a medication called HER2".
- "Squamous differentiation" means the cancer cells look like skin cells. It does NOT mean the cancer produces mucus (that would be mucinous).
- When describing imaging results, check if the finding is CURRENT or HISTORICAL. Do not present old imaging as new information if the condition has since been treated.
```

### 预期影响
- 消除 ~10 个 P2（garbled text + factual errors）
- 更安全的 patient letter — 避免 garbled text 误导患者

---

## 改进 4: procedure_plan — 字段定义严格化

### 问题
v29 中 ~10 个 P2 来自字段混入：
- procedure_plan 包含化疗方案（ROW 8: "adjuvant AC x4 → T-DM1"）
- procedure_plan 包含 imaging + medication + dental（ROW 20: "Abdomen, Pelvis, Xgeva"）
- imaging_plan 包含 XRT 治疗（ROW 95: "breast and axilla XRT"）
- imaging_plan fabricated echo 不在 A/P 中（ROW 66, 92）

### 根因
prompt 已有详细的 DO NOT include 列表，但模型仍然混淆。特别是：
1. 化疗方案被放入 procedure_plan
2. 放疗被放入 imaging_plan
3. 模型推测标准监测（如 anthracycline echo）而非只提取 A/P 写的内容

### 修改方案

在 `prompts/plan_extraction.yaml` 的 `Procedure_Plan` prompt 中添加：

```
FINAL CHECK — before outputting procedure_plan, verify EACH item:
1. Is it a physical/surgical intervention? (surgery, biopsy, port placement, drain removal, LP) → YES, include
2. Is it a systemic therapy? (chemo, hormonal, targeted, immunotherapy) → NO, goes in Medication_Plan
3. Is it imaging? (CT, MRI, PET, echo, DEXA, mammogram, ultrasound, bone scan) → NO, goes in Imaging_Plan
4. Is it radiation? (XRT, RT, GK, SRS, proton) → NO, goes in radiotherapy_plan
5. Is it a lab test? → NO, goes in Lab_Plan
6. Is it dental clearance? → NO, this is a prerequisite, not a procedure itself

If procedure_plan contains ANY systemic therapy drug names (AC, TCHP, taxol, letrozole, etc.), you have made an error. Remove them.
```

在 `Imaging_Plan` prompt 中添加：

```
CRITICAL — only include DIAGNOSTIC imaging, not therapeutic radiation:
- XRT, RT, radiation therapy, Gamma Knife, SRS → these go in radiotherapy_plan, NOT here
- Only include: CT, MRI, PET/CT, DEXA, bone scan, mammogram, ultrasound, echocardiogram/TTE, X-ray

CRITICAL — only include imaging ORDERED or PLANNED in the A/P:
- Do NOT add standard-of-care monitoring that is NOT explicitly mentioned in the A/P
- Example: Do NOT add "echocardiogram" just because the patient is on an anthracycline, unless the A/P explicitly orders it
```

### 预期影响
- 消除 ~10 个 P2
- 更清晰的字段分界，减少 downstream letter 中的混淆

---

## 实施优先级

| 改进 | 预期消除 P2 | 难度 | 优先级 |
|------|------------|------|--------|
| 1. response_assessment | ~10 | 低（prompt 文本修改） | **高** |
| 2. Type_of_Cancer 受体 | ~6 | 低（prompt 文本修改） | **高** |
| 3. Letter [REDACTED] | ~10 | 中（letter prompt 修改） | **中** |
| 4. procedure_plan 字段 | ~10 | 低（prompt 文本修改） | **中** |

**总预期**: 消除 ~36 个 P2，从 92 降至 ~56（-39%）

## 执行步骤

1. 修改 `prompts/extraction.yaml` — Response_Assessment 和 Cancer_Diagnosis 两个 prompt
2. 修改 `prompts/plan_extraction.yaml` — Procedure_Plan 和 Imaging_Plan 两个 prompt
3. 修改 letter generation prompt — [REDACTED] handling + factual accuracy rules
4. 创建 `exp/v30_full.yaml` 配置（copy from v29_full.yaml）
5. 在 WSL 上运行 v30 full (61 samples)
6. 下载结果 + 审查
