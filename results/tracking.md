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

## Prompt 版本变更记录

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
1. **P0**: G3 过度清空 Patient type/Stage/Metastasis (B25, B26, B27) — 影响 3/5 行
2. **P1**: G6 过度修正 (B24, B29) — 影响 2/5 行
3. **P1**: B14 response_assessment 不稳定 — 影响 1/5 行
4. **P2**: B20 Referral 包含 Med Onc — 影响 1/5 行
5. **P2**: B30 radiotherapy_plan 过去治疗 — 影响 1/5 行
