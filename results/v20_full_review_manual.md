# v20 Full Review: 人工逐字审查

Run: `results/v20_full_20260321_115736/`
Date: 2026-03-21
审查方式: **人工逐字审查**（非 agent）

---

## ROW 2 (Row 1, coral_idx 141) — ✅ 无问题

**患者**: 44 yo female, Lynch Syndrome + 多原发癌, 转移性 TNBC, irinotecan cycle 3 day 1。

**结论**: 无 P0/P1。current_meds="Irinotecan" 正确（prompt 只要癌症药）。POST-MEDS-IV-CHECK 修复保持。P2: supportive_meds 遗漏 flexeril；Referral Others 过度提取（旧转诊+疑问句）。

---

## ROW 14 (Row 13, coral_idx 153) — ❌ P0×4 + P1×4

**患者**: 58 yo female, de novo metastatic ER+ breast cancer → bone/liver/nodes。已停 faslodex+palbociclib，自行在墨西哥做低剂量化疗在家给药。医生明确不认可。

### P0 问题

**P0-1: summary 时态错误**
- 提取: "currently on faslodex and palbociclib"
- 原文: "She has stopped palbociclib and fulvestrant. Her last dose of [REDACTED] was 01/04/19"
- 判定: 说"currently on"已停 2 个月的药，严重时态错误

**P0-2: current_meds 包含墨西哥自管药物**
- 提取: "Gemcitabine 200mg, Docetaxel 20 mg, pamidronate"
- 原文态度: "Apparently she stopped her [meds], and started multiple chemotherapy combination therapy in mexico, which for some reason she believes is different and better than US based therapy!" / "so-called metabolic therapy to administer on her own at home!" / "[we] discontinue our medications"
- Current Outpatient Medications 段: 只有 DULoxetine (Patient not taking)
- 另外遗漏了 doxorubicin（IMP: "The treatment includes pamidronate, gemcitabine, docetaxel, and doxorubicin"）

**P0-3: medication_plan 曲解医嘱**
- 提取: "Continue low dose chemo combination..."
- 原文: 医生没有说"continue"这些药。A/P 第一位医生只描述 "Following tx protocol from [Mexico]"（描述行为非医嘱）；第二位医生 IMP 表达不认可，说 "When I asked her what she wanted our role to be she says to monitor her."

**P0-4: therapy_plan 同上**
- 提取: "Continue low dose chemo...Continue pamidronate..."
- 同 medication_plan，把患者自行行为当作医生计划

### P1 问题

| 字段 | 问题 | 原文依据 |
|------|------|---------|
| Type_of_Cancer | 缺 PR+，HER2 写 "unclear" | FNA: "ER 99%, PR 25% [IHC] 1+, FISH negative" = ER+/PR+/HER2- |
| Stage | "Not available (redacted)" | metastatic = Stage IV（可推断）|
| findings | 混入治疗方案描述+引用 11/30/18 旧影像 | 当前体检发现被遗漏（axillary node, breast density, ECOG 2, weight）|
| response_assessment | "cancer is currently stable" 基于旧影像 | 当前访视无新影像（CT 被取消），应说明基于旧影像+肿瘤标志物趋势 |

### 跨版本对比（v20 改动是否引入新问题）

| 字段 | v19 | v20b | v20_full | 结论 |
|------|-----|------|---------|------|
| summary | "currently on faslodex and palbociclib" | 同 | 同（多了"including recent discontinuation"但仍以"currently on"开头）| **非 v20 引入** |
| current_meds | "Pamidronate, Gemcitabine, Docetaxel, Doxorubicin" | "pamidronate, gemcitabine, docetaxel, doxorubicin" | "Gemcitabine 200mg, Docetaxel 20 mg, pamidronate" | **非 v20 引入**（v20_full 少了 doxorubicin，LLM 随机差异）|
| medication_plan | "Continue low dose chemo..." | "Continue low dose chemo...Continue insulin..." | "Continue low dose chemo..." | **非 v20 引入** |
| therapy_plan | "Continue low dose chemo..." | 同 | 同 | **非 v20 引入** |
| Type_of_Cancer | "HER2: status unclear" | 同 | 同 | **非 v20 引入** |

**结论: Row 13 的所有 P0/P1 问题都是 v19 就存在的，v20 的三个改动（Pattern 6 regex、SELF-MANAGED prompt、POST-SUPP-ALLERGY）没有引入任何新问题。** SELF-MANAGED prompt 对此行无效果（模型完全忽略）。

### 全版本追溯（v14a → v20_full）

| 字段 | v14a | v15a | v19 | v20b | v20_full | 首次出问题 |
|------|------|------|-----|------|---------|-----------|
| **summary "currently on"** | P1 ❌ | 未查 | 未查 | P0 ❌ | P0 ❌ | **v14a 起** |
| **current_meds 墨西哥药** | ✅ (当时标OK) | ✅ (标OK) | P0 ❌ (重新定义) | P0 ❌ | P0 ❌ | v14a-v15a 标 OK，**v19 升级为 P0** |
| **medication_plan "Continue"** | ✅ (标OK) | 未查 | 未查 | P0 ❌ (本次发现) | P0 ❌ | **v14a 起就有，从未被标注** |
| **therapy_plan "Continue"** | ✅ (标OK) | 未查 | 未查 | P0 ❌ (本次发现) | P0 ❌ | 同上 |
| **Type HER2** | P1 (缺) | ⚠️ "unclear" | P1 "unclear" | P1 "unclear" | P1 "unclear" | **v14a 起** |
| **Stage** | ✅ "Stage IV" | 未查 | "Not available" | ✅ "Stage IV" | "Not available" | v19 起（LLM 随机差异，有时写 Stage IV，有时写 [X] 被替换）|
| **response_assessment** | P1 "responding" | P1 "stable" | 未详查 | P1 旧影像 | P1 旧影像 | **v14a 起** |

**关键发现**:
1. **summary 和 medication_plan/therapy_plan 的 P0 从 v14a 起就存在**，但 v14a 审查时没被标为错误（当时审查标准不同）
2. **current_meds 的"正确性"定义在 v19 发生了变化** — v14a/v15a 认为"患者确实在用" = OK；v19 认为"医生不认可" = P0
3. **没有任何版本是完全没问题的** — v14a 就有 summary P1 + Type P1 + response P1
4. **Stage 在 v14a 是正确的** ("Stage IV")，但 v19/v20_full 变成 "Not available"（LLM 随机输出 "[X]" 触发 POST-STAGE-PLACEHOLDER）。v20b 仍正确。这是随机退化，不是代码引入的。

### 6 版本逐字段对比（v14→v14a→v15a→v16→v20a→v20b→v20_full）

| 字段 | v14 | v14a | v15a | v16 | v20a | v20b | v20_full | 最佳版本 |
|------|-----|------|------|-----|------|------|---------|---------|
| **summary** | "currently on faslodex and palbociclib" ❌ | 同 ❌ | 同 ❌ | 同 ❌ | 同 ❌ | 同 ❌ | 同 ❌ 但多了"including recent discontinuation" | **v20_full 略好**（至少提到了停药，虽然仍以"currently on"开头）|
| **Type** | "ER+ IDC" (缺HER2) | 同 | "+HER2: status unclear" | "+HER2: status unclear" | 同 | 同 | "ER+ breast cancer, HER2: unclear" | **v15a=v16**（加了HER2状态，虽然不准确）|
| **Stage** | "metastatic (Stage IV)" ✅ | 同 ✅ | 同 ✅ | "Stage [X], now metastatic (Stage IV)" | "metastatic (Stage IV)" ✅ | 同 ✅ | "Not available (redacted)" ❌ | **v14/v14a/v15a/v20a/v20b** ✅ |
| **current_meds** | 4 药全 ❌P0 | 同 | 同 | 同 | 3 药（少 doxorubicin）❌P0 | 4 药 ❌P0 | 3 药（少 doxorubicin）❌P0 | **全部 P0**（无最佳）|
| **response** | "currently responding" ❌ | 同 | "currently stable" ⚠️ | 引用旧影像详细描述 | "currently stable...stable disease" ⚠️ | "stable...No new lesions...CA 27.29=48" | "stable...no significant changes" ⚠️ | **v20b**（提到了 CA 27.29 数值）|
| **medication_plan** | "Continue low dose chemo...insulin..." ❌P0 | 同 | 同 | 同 | 同（含 insulin）| 同（含 insulin）| 同（无 insulin）| **全部 P0**（无最佳）|
| **therapy_plan** | "Continue low dose chemo..." ❌P0 | 同 | 同 | 同 | 同 | 同 | 同 | **全部 P0**（无最佳）|

### 综合评价

**没有任何版本是完美的。** 每个版本都有核心 P0（current_meds 墨西哥药物 + medication/therapy plan "Continue"）。

各版本相对优势：
- **v14/v14a**: Stage 正确 ("Stage IV")，但 Type 缺 HER2，response 过度乐观 ("responding")
- **v15a**: 加了 HER2 信息（虽然 "unclear"），response 从 "responding" 降为 "stable"（更准确）
- **v16**: Stage 引入 "[X]" 占位符（退化），其他与 v15a 类似
- **v20a**: Stage 恢复正确。current_meds 少了 doxorubicin
- **v20b**: Stage 正确 + response 提到了 CA 27.29 数值 → **最佳版本**
- **v20_full**: Stage 退化为 "Not available"。summary 略好（提到停药）。current_meds 少了 doxorubicin

**最佳版本: v20b**（Stage 正确 + response 最详细 + HER2 有状态）。但仍有 P0（current_meds + medication_plan）。

---

## ROW 57 (Row 56, coral_idx 196) — ✅ 修复保持

**患者**: 59 yo female, left breast CA, locally advanced, TNBC（初诊 HER2+，术后复核为 TNBC）。S/p TCH+P x6 + 手术 + AC x4。2nd opinion 门诊。

**v20 三个 P0 修复确认**:
1. ✅ Type = "ER-/PR-/HER2- breast cancer, triple negative"（v16-v18 HER2+/TNBC 混淆已解决）
2. ✅ current_meds = ""（v19 docetaxel FP 已解决，Pattern 6 regex 修复有效）
3. ✅ supportive_meds = ""（v19 过敏药 benadryl/codeine 已解决，POST-SUPP-ALLERGY 有效）

**P1×2**（小问题）:
- recent_changes: "Dose reduction 25% after C1" — 2013 年历史事件，非当前变化
- procedure_plan: "which pt is scheduled to receive" — radiotherapy_plan 的残余文本片段

---

## ROW 59 (Row 58, coral_idx 198) — P1: current_meds 时态

**患者**: 52 yo female, Stage I ER+/PR+/HER2- IDC。辅助内分泌治疗药物多次切换（tamoxifen→letrozole→计划 exemestane）。

**核心问题 — current_meds 时态判断**:
- 提取: "exemestane ([REDACTED]), letrozole (FEMARA) 2.5 mg tablet" — ❌ **P1**
- 原文: "she has not tried it yet" + "She was not aware of this recommendation"（exemestane 未开始）
- 正确答案: current_meds = "letrozole"（仅 letrozole，就诊时仍在服用；exemestane 虽有处方但未执行）
- **根因**: EMR med list 同时列出两者，模型按 med list 提取而未结合 A/P 时态

---

## ROW 90 (Row 89, coral_idx 229) — P1×4

**患者**: 51 yo female, right breast adenocarcinoma (IDC), Clinical st II/III, 临床试验中。S/p lumpectomy (2.2cm residual IDC)，AC cycle 3 已完成，准备 cycle 4。Telehealth 副作用管理。

**修复确认**: current_meds="ac" ✅ — POST-MEDS-IV-CHECK 保持

**P1 问题**:
1. **Patient type** = "New patient" — 应为 Follow up（AC cycle 3 已完成，在治疗中）
2. **Stage** = "Not mentioned" — A/P 明确写 "Clinical st II/III"
3. **supportive_meds** = "" — 遗漏 gabapentin + lidocaine patches（A/P: "continue gabapentin", "Continue lidocaine patches"）
4. **Referral Specialty** = "None" — 遗漏 Endocrinology（A/P: "followed by Endo"）

**P2 问题**: summary "cycle 12"误导、Type 应写 IDC 非 adenocarcinoma、Metastasis "Not sure"应推断 No、response_assessment dump 实验室（答非所问）

---

## ROW 95 (Row 94, coral_idx 234) — ❌ P0: Type 受体状态矛盾

**患者**: 49 yo female, left breast cancer, 临床试验中。S/p NAC + lumpectomy。术后病理 Addendum 显示受体状态变化。

**P0: Type_of_Cancer 受体状态与术后病理矛盾**:
- 提取: "ER+/PR+/HER2- IDC"（跟随 A/P 模板语言）
- 术后 Addendum: "PR is **negative** (no nuclear staining in any tumor cells)", "HER2 **equivocal** (2+, FISH pending)"
- 原始 bx (09/16): ER+ 100%, PR+ 40%, HER2 FISH neg
- **受体状态发生了治疗后变化**: PR+ → PR-，HER2- → equivocal
- 模型跟随 A/P 的模板语言而忽略了同一笔记中的 Addendum 最新结果

**P1×2**:
- summary: "presents today to start AC" — AC 已于 01/13/19 完成（A/P 模板复制错误）
- therapy_plan: 同上，含 "Start AC chemotherapy"（已完成）

**确认修复**: v16 Distant Met 分类错 ✅, v19 "缺 tamoxifen" 误判已确认 ✅

---

## Tier 1 审查总结（人工逐字审查）

| Row | ROW | 结果 | P0 | P1 | 关键发现 |
|-----|-----|------|-----|-----|---------|
| 1 | 2 | ✅ | 0 | 0 | current_meds="Irinotecan" 正确，修复保持 |
| 13 | 14 | ❌ | 4 | 4 | 墨西哥药+时态+曲解医嘱（v14a 起就有） |
| 56 | 57 | ✅ | 0 | 2 | 三个 v19 P0 全部修复保持 |
| 58 | 59 | ⚠️ | 0 | 1 | current_meds 时态（exemestane 未开始） |
| 89 | 90 | ⚠️ | 0 | 4 | AC 修复保持；Stage/Patient type/supportive_meds/Referral 遗漏 |
| 94 | 95 | ❌ | 1 | 2 | **新发现 P0**: 术后病理 PR 变阴性但提取仍为 PR+ |

**Tier 1 总计**: P0×5（Row 13×4 + Row 94×1），P1×13

---

# Tier 2 审查（人工逐字审查）

## ROW 1 (Row 0, coral_idx 140) — P1×2（结构性）

**患者**: 56 yo, Stage IIA→Stage IV 转移复发, 新患者初诊。

**P1×2（v14a 起持续 7 版本）**:
- imaging_plan = "No imaging planned" — 遗漏 MRI brain + bone scan（在 HPI: "I also ordered a MRI of brain and bone scan" 和 Orders 段，不在 A/P）
- lab_plan = "No labs planned" — 遗漏 CBC/CMP/CA15-3/CEA（同上）

**根因**: plan_extraction 只看 A/P，A/P 只写 "complete her staging work up" 未列具体检查。**结构性 pipeline 限制。**

**其余字段全部正确**: Type ER+/PR+/HER2- ✅, Stage IIA→IV ✅, findings CT 全覆盖 ✅, Advance care "Full code" ✅

---

## ROW 6 (Row 5, coral_idx 145) — P1×1: Stage 空

**患者**: 34 yo, ER+/PR+/HER2- IDC, 1.5cm N0 Grade 1, Oncotype Low Risk。S/p bilateral mastectomy。Zoladex + letrozole 辅助治疗。

**P1**: Stage="" — A/P: "1.5 cm node neg, grade 1" = pT1c N0 = **Stage I**。v14a 起持续空值。

**其余字段正确**: Type ER+/PR+/HER2- ✅（v15 HER2 反转 P0 已修复），current_meds zoladex+letrozole ✅，goals curative ✅，genetic_testing "None planned" ✅（Myriad 已完成 Negative）。P2: Referral Genetics 引用了过去的 referral（已完成）。

---

## ROW 8 (Row 7, coral_idx 147) — P1×1: procedure_plan 分类

**患者**: 29 yo, Stage III ER-/PR-/HER2+ IDC。S/p 不完整 neoadjuvant TCHP + lumpectomy/ALND。Zoom 会诊讨论 AC→T-DM1→radiation。

**P1**: procedure_plan = "adjuvant AC x 4 cycles, to be followed by T-DM1, plan for port placement" — AC/T-DM1 是化疗（应在 therapy_plan，已正确列出），procedure_plan 应只含 "port placement and echocardiogram"（A/P: "steps that would need to be taken...including port placement and echocardiogram"）。**v14a 起持续。**

**其余字段正确**: Type ER-/PR-/HER2+ (IHC 3+, FISH 5.7) ✅, Televisit ✅, current_meds="" ✅（未开始治疗）, goals curative ✅。

---

## ROW 12 (Row 11, coral_idx 151) — P1×3: Stage + Distant Met 缺 liver

**患者**: 51 yo, de novo Stage IV ER+/PR+/HER2+ IDC。转移到 liver, lung, nodes, brain, bone。用 herceptin/[REDACTED]+letrozole。S/p GK x3 脑转移。

**P1×3**（v14a 起持续）:
1. **Stage** = "Not available (redacted)" — A/P 明确写 "Metastatic breast cancer, **St IV** de [REDACTED]"。Attribution 引用了这句话但模型仍输出 "Not available"
2. **Metastasis** = "Yes (brain, lung, bone)" — 缺 **liver**（CT: "Large hepatic masses likely metastatic"；A/P: "to [REDACTED], lung, nodes, brain and bone"，[REDACTED] 处极可能是 "liver"）
3. **Distant Met** 同样缺 liver

**已修复**: imaging_plan = "CT CAP q4mo, bone scan, MRI brain q4mo" ✅（v19 遗漏已修复）

**其余字段正确**: Type ER+/PR+/HER2+ ✅, current_meds herceptin+letrozole ✅, goals palliative ✅, radiotherapy_plan "await GK/Rad Onc" ✅

---

## ROW 34 (Row 33, coral_idx 173) — ✅ 修复

**患者**: ER+/PR-/HER2- IDC, Stage III → local recurrence, arimidex。

**v14a P1 "Type 缺 HER2" → v20 修复**: Type = "ER positive PR negative [REDACTED] negative IDC, HER2-" ✅

其余字段正确。Stage "Originally Stage III, now local recurrence" ✅, current_meds "arimidex" ✅, goals curative ✅。

---

## ROW 42 (Row 41, coral_idx 181) — P2: Stage 空

**患者**: ER+/PR+/HER2- IDC, tamoxifen 辅助治疗。

**v14a P1 "Type 缺 ER 和 HER2" → v20 修复**: Type = "ER+/PR+/HER2- IDC" ✅

Stage = "Not mentioned in note" — P2（原文可能有信息推断，但不算严重遗漏）。其余字段正确。

---

## ROW 72 (Row 71, coral_idx 211) — ✅ 修复

**患者**: ER+/PR-/HER2- IDC with neuroendocrine differentiation, pT1cN0, letrozole。

**v14a P1 "current_meds 包含眼药水" → v20 修复**: current_meds = "letrozole" ✅（POST-MEDS-FILTER 过滤了 latanoprost）

Stage = "pT1cN0(sn)" ✅, genetic_testing_plan = "Ordered [REDACTED] to evaluate potential benefit of chemotherapy" ✅。

---

## ROW 91 (Row 90, coral_idx 230) — P2: HER2 "not tested"

**患者**: Stage I→IV ER+/PR+/HER2- IDC, bone+LN 转移, everolimus+exemestane+denosumab。

Type = "ER+/PR+ IDC, HER2: not tested" — **P2**: 原文 "ER+PR+ [REDACTED]-"（[REDACTED] 位置是 HER2），加上治疗全是 HR+ 药物（tamoxifen, letrozole, exemestane, everolimus，无 trastuzumab），可推断 HER2-。但脱敏导致模型保守标注 "not tested"。

其余字段正确：Stage "Stage I→Stage IV" ✅, current_meds "everolimus, exemestane, denosumab" ✅, goals palliative ✅。

---

# Tier 2 审查总结

| Row | ROW | 结果 | 关键发现 |
|-----|-----|------|---------|
| 0 | 1 | P1×2 | imaging/lab plan 遗漏 Orders 段（结构性，7 版本持续）|
| 5 | 6 | P1×1 | Stage 空（可推断 Stage I）|
| 7 | 8 | P1×1 | procedure_plan 混入化疗 |
| 11 | 12 | P1×3 | Stage "Not available"（原文有 St IV）+ Metastasis 缺 liver |
| 33 | 34 | ✅ | Type HER2 修复！|
| 41 | 42 | ✅ | Type ER/HER2 修复！P2 Stage 空 |
| 71 | 72 | ✅ | 眼药水过滤修复！|
| 90 | 91 | P2 | HER2 "not tested"（脱敏导致，可接受）|

**Tier 2 总计**: P1×7, P2×2, 修复×3

---

# Tier 3+4 审查（人工逐字审查 + 原文确认）

## 无问题的行（36行）

以下行经人工阅读原文和 keypoints 确认无 P0/P1：

Row 2, 4, 6, 8, 9, 16, 17, 19, 21, 26, 28, 29, 32, 35, 36, 39, 42, 43, 45, 48, 49, 53, 62, 63, 64, 67, 69, 72, 77, 79, 81, 83, 84, 85, 86, 87, 91, 93, 96

## P1 问题行

### ROW 100 (Row 99, coral_idx 239) — P1: current_meds 遗漏 Gemzar
- current_meds = "" — **P1**
- 原文 HPI: "metastatic breast cancer on **Gemzar** Cycle #2 Day 8 **cancelled by patient**"
- A/P: "continue with treatment" + "Did not receive treatment today"
- Gemzar 仍是当前方案（只是本次 day 8 被患者取消），应列入 current_meds
- Type: "ER+(80%)PR+(50%) IDC, HER2: not tested" — P2（原文 "ER+(80%)PR+(50%)[HER2]-"，HER2 被脱敏）

## P2 系统性问题（Stage 推断保守，~8行）

以下行 Stage = "Not mentioned in note"，但原文可能有信息推断：
- ROW 41 (Row 40): A/P 有 3cm N1mi 信息但未推断
- ROW 44 (Row 43): 术后但 Stage 未填
- ROW 52 (Row 51): A/P: "locoregional disease" 可推断 Stage II
- ROW 61 (Row 60): A/P: "early stage breast cancer"
- ROW 66 (Row 65): TNBC metaplastic
- ROW 80 (Row 79): 初诊
- ROW 87 (Row 86): 2.2cm 4/19 LN+ 可推断 Stage IIA-IIB

## P2: HER2 "not tested"（脱敏导致，3行）
- ROW 73 (Row 72): 原文 HER2 被脱敏
- ROW 91 (Row 90): 原文 "[HER2]-" 被脱敏
- ROW 100 (Row 99): 原文 "[HER2]-" 被脱敏

## ROW 83 (Row 82, coral_idx 222) — P1: Distant Met 空
- Distant Metastasis = "" — **P1**（应为 "No"，只有 regional axillary LN）

---

# 全量审查总结

## v20 Full 61行人工逐字审查完成

| 统计 | 数量 |
|------|------|
| 总行数 | 61 |
| **P0** | **2行 5个**: Row 13×4（墨西哥药物）+ Row 94×1（PR 状态矛盾）|
| **P1** | **~10行 ~22个**: Stage×4, DistMet×3, current_meds×2, imaging/lab×2, procedure×1, supportive×1, PatientType×1, Referral×1, Type×3, 其他 |
| **P2** | **~12行**: Stage 推断保守×8, HER2 脱敏×3, 其他 |
| **无问题** | **~39行 (64%)** |
| **回归** | **0行** |
| **修复确认** | **6项**: irinotecan ✅, HER2×2 ✅, docetaxel/过敏/TNBC ✅, 眼药水 ✅, AC ✅ |

**其他字段**: 全部正确。Type、Stage、goals、imaging_plan、medication_plan（正确写了"停 letrozole 换 exemestane"）等无问题。response_assessment 后半段混入副作用描述（P2）。
