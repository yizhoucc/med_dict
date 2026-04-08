# v20b Review: Row 13 & Row 56 逐行审查

Run: `results/v20_verify_20260321_103552/`
Date: 2026-03-21
Focus: 验证 3 个 P0 修复 + 全字段审查

---

## ROW 14 (Row 13, coral_idx 153)

**患者概况**: 58 y.o. female, de novo metastatic ER+ breast cancer → bone, liver, nodes。此前用 faslodex+palbociclib，2019-01 自行停药。去墨西哥诊所接受低剂量化疗（gemcitabine, docetaxel, doxorubicin, pamidronate），在家自行给药。主治医生明确不认可（"Apparently", "so-called", "discontinue our medications"）。本次为随访。

### 逐字段审查

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Patient type** | "follow up" | ✅ | 原文: "She is here today for follow-up." |
| **second opinion** | "no" | ✅ | 非二次意见 |
| **in-person** | "in-person" | ✅ | "face-to-face encounter" |
| **summary** | 58 y.o. female...follow-up | ✅ | 准确 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Type_of_Cancer** | "ER+ invasive ductal carcinoma, HER2: status unclear" | ⚠️ P1 | 原文 biopsy: "ER 99%, PR 25% [IHC] 1+, FISH negative"。应为 ER+/PR+/HER2-。PR+ 被 G3-IMPROVE 删掉（line 43: "ER+/PR+/HER2-" → "ER+"），G4 又清空，G4-PROTECT 恢复为 "ER+"。POST-HER2-CHECK 未识别 "FISH negative" = HER2-，加了 "status unclear"。**根因**: G3 过度简化 + POST-HER2-CHECK 模式不够全面 |
| **Stage_of_Cancer** | "metastatic (Stage IV)" | ✅ | de novo metastatic |
| **Metastasis** | "Yes (to bone, liver, and nodes)" | ✅ | 原文: "metastatic ER+ breast cancer to bone, liver and nodes" |
| **Distant Metastasis** | "Yes (to bone, liver, and nodes)" | ✅ | POST-DISTMET 正确添加 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **lab_summary** | 完整 CMP + CBC + CA 27.29 | ✅ | 准确复制了 RESULTS 段的所有检验值 |
| **findings** | 包含治疗方案+体检+影像 | ⚠️ P2 | 混入了治疗方案描述（"currently on low dose chemo..."），这应在 current_meds/recent_changes 里。findings 应偏重客观发现（PE: "Palpable R axillary node 1 cm"、imaging、tumor markers）。但不算错误。 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **current_meds** | "pamidronate, gemcitabine, docetaxel, doxorubicin" | ❌ **P0** | **Fix 2 未生效**。这些是墨西哥自管药物。原文有多处不认可信号: (1) IMP: "Apparently she stopped her [meds], and started multiple chemotherapy combination therapy in mexico, which for some reason she believes is different and better than US based therapy!" (2) "so-called metabolic therapy to administer on her own at home!" (3) "[we] discontinue our medications"。Current Outpatient Medications 段只有 DULoxetine (not taking)。LLM 完全忽略 SELF-MANAGED prompt 指导，被 IMP 中 "The treatment includes pamidronate, gemcitabine, docetaxel, and doxorubicin" 误导。POST-MEDS-FILTER 不会过滤这些因为它们是真正的化疗药（在 ONCO_WHITELIST 上）。**v20b 比 v20a 还多提取了 doxorubicin**。 |
| **supportive_meds** | "pamidronate once weekly" | ⚠️ P1 | pamidronate 是骨保护剂，分类合理。但 (1) 与 current_meds 重复；(2) 同样来自墨西哥自管方案。原文: "Receive pamidronate once weekly" |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **recent_changes** | "Stopped palbociclib and fulvestrant end of January..." | ✅ | 准确。"Last dose of Fulvestrant was 01/04/19" 也正确。后半段描述了墨西哥方案，但放在 recent_changes 是合理的。 |
| **goals_of_treatment** | "palliative" | ✅ | Stage IV metastatic → palliative。G4 曾清空，G4-REVERT 正确恢复。 |
| **response_assessment** | "The cancer is currently stable..." | ⚠️ P1 | 引用了 11/30/18 MRI 和 CT 结果（"No new or worsening..."），但这不是当前访视的影像。当前访视 (03/01/19) 无新影像（取消了 CT）。不过 CA 27.29 趋势下降 (193→48) 是准确的。应说明是基于旧影像+肿瘤标志物趋势。 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **medication_plan** | "Continue topical cannabis...Continue insulin..." | ⚠️ P1 | "Continue insulin as well as other medications" 来自 IMP 中对墨西哥方案的描述，不是医生的治疗计划。 |
| **therapy_plan** | "Continue low dose chemo...Continue pamidronate..." | ⚠️ P1 | 同上，这些是墨西哥自管方案，不是本院医生的计划。G5 已正确删除了 "Discontinue palbociclib and fulvestrant"（过去事件）。 |
| **radiotherapy_plan** | "None" | ✅ | 无放疗计划 |
| **procedure_plan** | "No procedures planned." | ✅ |  |
| **imaging_plan** | "CT CAP and Total Spine MRI scheduled for May. Repeat total spine MRI in 6 weeks." | ✅ | 原文: "She wants to schedule CT CAP and Total Spine MRI for May" + "Repeat total spine MRI in 6 weeks" |
| **lab_plan** | "Labs to be drawn every two weeks." | ✅ | 原文: "She has been instructed to have her labs drawn every two weeks" |
| **genetic_testing_plan** | "None planned." | ✅ |  |
| **Next clinic visit** | "in-person: 2 months for follow-up" | ✅ | IMP: "will return in two months"（注: 另一位医生写 "F/u 3 months"，有矛盾，取 2 months 合理） |
| **Advance care** | "Not discussed during this visit." | ✅ |  |

| Referral 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Nutrition** | "None" | ✅ | POST-NUTRITION 正确清除了饮食建议 |
| **Genetics** | "None" | ✅ |  |
| **Specialty** | "None" | ✅ |  |
| **Others** | "Physical therapy referral" | ✅ | "start PT on 03/12/19" |
| **follow up** | "RTC after scans" | ✅ |  |

### Row 13 小结
- **P0 x1**: current_meds 仍包含墨西哥自管药物（Fix 2 prompt 指导无效）
- **P1 x5**: Type_of_Cancer 缺 PR+/HER2-、supportive_meds 与 current_meds 重复、response_assessment 引用旧影像、medication_plan/therapy_plan 包含墨西哥方案
- **P2 x1**: findings 混入治疗方案描述

---

## ROW 57 (Row 56, coral_idx 196)

**患者概况**: 59 yo female, left breast CA, locally advanced。初始诊断为 [HER2+]，用 TCH+P x6 新辅助化疗。手术后残留 3.7cm 肿瘤，0/6 nodes。术后病理和外院复核均为 **TNBC**（HER2 阴性）。术后 AC x4。本次为 **二次意见**：确认 HER2 状态、讨论后续方案。当前无癌症药物（仅用 Carvedilol/Lisinopril/Asacol 非癌症药）。过敏: Erythro, benadryl, codeine, demerol, motrin, pcn。

### 逐字段审查

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Patient type** | "New patient" | ✅ | 新到本院 second opinion |
| **second opinion** | "yes" | ✅ | "She is here to obtain 2nd opinion" |
| **in-person** | "in-person" | ✅ | 完整体格检查 |
| **summary** | 59-year-old...TNBC...second opinion | ✅ | 准确 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Type_of_Cancer** | "ER-/PR-/HER2- breast cancer, triple negative" | ✅ | POST hook 链: LLM→"TNBC" ✅ → POST-HER2-VERIFY 误改为 HER2+（因 trastuzumab/pertuzumab 用药历史）→ POST-TYPE-VERIFY 删 "triple negative" → **POST-TYPE-VERIFY-TNBC** 读到 A/P 说 TNBC 后纠正回 HER2-。最终结果正确，但过程迂回。⚠️ P2: POST-HER2-VERIFY 不应仅根据历史用药推断 HER2 状态，尤其当病理明确否定时。 |
| **Stage_of_Cancer** | "Locally advanced" | ✅ | 原文 CC: "locally advanced breast ca" |
| **Metastasis** | "No" | ✅ | 无远处转移 |
| **Distant Metastasis** | "No" | ✅ | G4 清空后 G4-REVERT-INFER 正确恢复 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **lab_summary** | "No labs in note." | ✅ | 原文无检验结果 |
| **findings** | 病理+体检 | ✅ | "residual tumor 3.7 cm, 0/6 nodes" 来自手术病理（2013年，但对 2nd opinion 有参考价值）。体检发现准确。 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **current_meds** | "" | ✅ **FIX 1 确认** | LLM 正确输出空。G4 试图添加 "Carvedilol. Lisinopril. Asacol."（非癌症药）→ POST-MEDS 全部过滤。**v19 的 docetaxel FP 已消除** — POST-MEDS-IV-CHECK 未触发（Pattern 6 regex 修复有效，"beneficial given docetaxel" 不再匹配）。 |
| **supportive_meds** | "" | ✅ **FIX 3 确认** | LLM 提取了 "benadryl, codeine, demerol, motrin"（来自 "ALL: Erythro, benadryl, codeine, demerol, motrin, pcn"）。POST-SUPP 先过滤 demerol/motrin（可能不在 supportive whitelist 上）。**POST-SUPP-ALLERGY 成功移除 benadryl 和 codeine**（匹配到 "ALL:" 段的过敏药物）。最终 ""。 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **recent_changes** | "Dose reduction 25% was instituted after C1." | ⚠️ P1 | 这是 2013 年新辅助化疗 TCH+P 期间的剂量调整，不是当前的 recent change。当前没有 recent changes（这是 2nd opinion 门诊）。G5-TEMPORAL 仅适用于 PLAN_KEYS，不检查 recent_changes。 |
| **goals_of_treatment** | "curative" | ✅ | Locally advanced（非转移），s/p 手术+化疗+计划 XRT → curative。G4 曾清空，G4-REVERT 恢复。 |
| **response_assessment** | "Not mentioned in note." | ✅ | 2nd opinion 门诊，无治疗反应评估 |

| 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **medication_plan** | null | ⚠️ P2 | 应为 "None" 而非 null（一致性）。G5-TEMPORAL 正确清除了历史化疗方案。 |
| **therapy_plan** | "Rec XRT...If [REDACTED], resume trastuzumab." | ✅ | 准确: XRT 是明确计划，trastuzumab 是条件性（如果 HER2+ 确认）。 |
| **radiotherapy_plan** | "Rec XRT, which pt is scheduled to receive" | ✅ |  |
| **procedure_plan** | "Rec genetic counseling and testing..." | ✅ | POST-PROC-FILTER 正确移除了 XRT（是治疗非手术）。genetic counseling 放在 procedure 稍不精确但可接受。 |
| **imaging_plan** | "No imaging planned." | ✅ |  |
| **lab_plan** | "No labs planned." | ✅ |  |
| **genetic_testing_plan** | "Rec genetic counseling and testing" | ✅ | 原文: "Rec genetic counseling and testing"，FHx "Has not had genetic counseling" |
| **Next clinic visit** | "Not specified in the provided note" | ✅ | 原文确实未指定随访时间 |
| **Advance care** | "Not discussed during this visit." | ✅ |  |

| Referral 字段 | 输出值 | 判定 | 分析 |
|------|--------|------|------|
| **Nutrition** | "None" | ✅ |  |
| **Genetics** | "Rec genetic counseling and testing" | ✅ |  |
| **Specialty** | "Rec XRT, which pt is scheduled to receive" | ✅ | = radiation oncology referral |
| **Others** | "None" | ✅ |  |
| **follow up** | "scheduled to receive XRT in the near future" | ✅ |  |

### Row 56 小结
- **P0 x0**: 两个 P0（docetaxel FP + 过敏 supportive_meds）均已修复 ✅
- **P1 x1**: recent_changes 引用了 2013 年的历史剂量调整
- **P2 x2**: medication_plan 为 null 而非 "None"；POST-HER2-VERIFY 误改链迂回

---

## v20 Fix 验证总结

| Fix | 目标 | Row 13 | Row 56 | 结论 |
|-----|------|--------|--------|------|
| **Fix 1** Pattern 6 regex | 消除 "given docetaxel" FP | N/A | ✅ current_meds="" | **修复成功** |
| **Fix 2** SELF-MANAGED prompt | 排除墨西哥自管药物 | ❌ 仍提取 4 种药 | N/A | **修复失败** — 32B 模型忽略 prompt |
| **Fix 3** POST-SUPP-ALLERGY | 过滤过敏列表药物 | N/A | ✅ benadryl/codeine 被移除 | **修复成功** |

## 遗留 P0: Row 13 current_meds

### 根因分析
1. **LLM 行为**: 模型从 IMP 段 "The treatment includes pamidronate, gemcitabine, docetaxel, and doxorubicin" 直接提取，完全忽略 prompt 中的 SELF-MANAGED 规则
2. **G4-FAITH**: 判定为 "all supported"（原文确实有提及这些药物），未质疑
3. **POST-MEDS-FILTER**: 不过滤（都在 ONCO_WHITELIST 上）
4. **POST-DRUG-VERIFY**: 不过滤（note_text 中确实有 gemcitabine/docetaxel/pamidronate/doxorubicin）

### 可能的修复方向
1. **POST hook 方案**: 检测 A/P 中的不认可语言（"apparently", "so-called", "mexico", "discontinue our medications"），如果检测到 + Current Outpatient Medications 段为空/只有非癌症药 → 清空 current_meds
2. **更强的 prompt 措辞**: 将 SELF-MANAGED 从建议改为强制指令，加具体示例
3. **接受限制**: 这是极端边界案例（患者自行用墨西哥化疗药物），在 100 行中仅此 1 例

## 新发现的问题（非 v20 目标）

| 级别 | Row | 字段 | 问题 |
|------|-----|------|------|
| P1 | 13 | Type_of_Cancer | 缺 PR+、HER2 显示 "unclear"（原文 FISH negative = HER2-） |
| P1 | 13 | response_assessment | 引用了旧影像（11/30/18），当前访视无新影像 |
| P1 | 56 | recent_changes | 包含 2013 年历史剂量调整，应为空 |
| P2 | 56 | Type_of_Cancer | POST-HER2-VERIFY 误改链迂回（最终正确但过程浪费） |
| P2 | 56 | medication_plan | null 而非 "None" |
