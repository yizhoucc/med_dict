# v20 Verify2 Review: 7 Previously Problematic Rows

Run: `results/v20_verify2_20260321_105736/`
Date: 2026-03-21
Rows: 0, 7, 11, 17, 58, 84, 94（v19 review 中发现问题的行）

---

## ROW 1 (Row 0, coral_idx 140)

**患者**: 56 yo female, Stage IIA right breast cancer (2013), metastatic relapse → lungs, peritoneum, liver, ovaries. 新患者初诊。无当前癌症药物。

**v19 问题**: Imaging_Plan/Lab_Plan 遗漏 Orders

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Patient type | "New patient" | ✅ | |
| Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma, HER2-" | ✅ | 原文 "ER and PR positive and her 2 neu negative" |
| Stage | "Originally Stage IIA, now metastatic (Stage IV)" | ✅ | |
| Distant Metastasis | "Yes (to lungs, peritoneum, liver, and ovaries)" | ✅ | |
| current_meds | "" | ✅ | 无当前药物 |
| goals_of_treatment | "palliative" | ✅ | |
| response_assessment | "Not yet on treatment" | ✅ | |
| procedure_plan | "biopsy mass in right axilla" | ✅ | |
| **imaging_plan** | "No imaging planned." | ❌ **P1** | 原文 HPI: "I also ordered a MRI of brain and bone scan as well as labs to complete her work up." Orders 段列出了 NM Whole Body Bone Scan + MR Brain。但 A/P 只说 "complete her staging work up" 未具体列出。**根因**: plan_extraction 主要从 A/P 提取，Orders 在笔记头部的独立段落中。 |
| **lab_plan** | "No labs planned." | ❌ **P1** | 同上。Orders 段有 CBC, CMP, CA 15-3, CEA, aPTT, PT。HPI 说 "labs to complete her work up"。 |
| Referral/Specialty | "Integrative Medicine" | ✅ | 原文 Orders 有 "Ambulatory Referral to Integrative Medicine" |

**v19→v20b**: imaging_plan/lab_plan **未修复**（结构性问题，非 v20 目标）。其余字段良好。

---

## ROW 8 (Row 7, coral_idx 147)

**患者**: 29 yo female, Stage III ER-/PR-/HER2+ IDC left breast. 新辅助 TCHP 未完成（仅 3 个不完整 cycle），s/p lumpectomy/ALND（乳房无残留，3/28 LN+）。Zoom 会诊讨论后续方案。

**v19 问题**: current_meds/response 问题

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Patient type | "New patient" | ✅ | 新转来建立 care |
| in-person | "Televisit" | ✅ | "presents through ZOOM" |
| Type_of_Cancer | "ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) IDC" | ✅ | |
| Stage | "Originally Stage II-III, now Stage III" | ✅ | A/P 说 "clinical stage III" |
| current_meds | "" | ✅ | 未开始新治疗（计划 AC → T-DM1） |
| supportive_meds | "oxyCODONE" | ✅ | 药物列表有 oxycodone |
| goals_of_treatment | "curative" | ✅ | |
| response_assessment | "Not yet on treatment" | ✅ | 新就诊，未开始治疗 |
| medication_plan | "adjuvant AC x 4, followed by T-DM1" | ✅ | |
| radiotherapy_plan | "radiation after completing AC" | ✅ | |
| **procedure_plan** | "adjuvant AC x 4 cycles" | ❌ **P1** | AC 是化疗非手术。应在 therapy_plan 中（已正确列出）。procedure_plan 应为 "port placement and echocardiogram"（A/P 中提及）。 |

**v19→v20b**: current_meds 和 response 问题已修复。新发现 procedure_plan 分类错误。

---

## ROW 12 (Row 11, coral_idx 151)

**患者**: 61 yo female, Stage IIIA ER+ invasive lobular carcinoma right breast, 现为 Stage IV 转移到 bone, muscle, liver, brain。当前用 herceptin/[redacted]+letrozole+denosumab。第一线 fulvestrant/palbociclib 已进展。

**v19 问题**: Stage "Not available"、Distant Met 遗漏 liver、current_meds 遗漏 pertuzumab

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Type_of_Cancer | "ER+/PR+/HER2+ IDC" | ⚠️ P1 | 原文活检: "ER+ 95%, [HER2]- [HER2]- (score 0)" 和 liver bx "ER (50%)+/PR-/[HER2] negative"。PR 状态在不同活检中不一致（原始 PR-，但模型写 PR+）。HER2 始终阴性但有 trastuzumab 使用。这是复杂的异质性肿瘤。|
| **Stage** | "Not available (redacted)" | ❌ **P1** | A/P 明确写 "Metastatic breast cancer, St IV"。Attribution 也引用了这句话。但 LLM 仍然输出 "Not available"。G4 可能清空了 LLM 的提取。**与 v19 相同问题未修复。** |
| Distant Metastasis | "Yes (to brain, lung, bone)" | ⚠️ **P1** | **仍遗漏 liver**。A/P: "3 new liver lesions", liver biopsy 结果在笔记中。但 Distant Met attribution 引用了较早的描述。 |
| current_meds | "herceptin, letrozole" | ⚠️ | A/P: "cont herceptin/[redacted]" + "cont letrozole qd"。[redacted] 可能是 pertuzumab，但模型无法知道被脱敏的内容。可接受。 |
| **imaging_plan** | "CT CAP every 4 months, bone scan, MRI brain q4mo" | ✅ | **v19 遗漏已修复！** |
| response_assessment | 详细引用了 MRI/CT 结果 | ✅ | 包含脑转移新发病灶等 |
| radiotherapy_plan | "await GK / Rad Onc input" | ✅ | |

**v19→v20b**: imaging_plan 修复 ✅。Stage 和 Distant Met（缺 liver）未修复。

---

## ROW 18 (Row 17, coral_idx 157)

**患者**: 新诊断 ER+/PR+/HER2- IDC，pT1b pNX。S/p 手术，尚未开始辅助治疗。

**v19 问题**: Stage 遗漏、current_meds 问题

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Type_of_Cancer | "ER+/PR+/HER2- IDC, encapsulated papillary carcinoma" | ✅ | |
| **Stage** | "pT1b, pNX — approximately Stage I" | ✅ | **v19 遗漏已修复！** 从 pT1b 推断 Stage I |
| Distant Metastasis | "No" | ✅ | |
| current_meds | "" | ✅ | 尚未开始治疗。A/P 说 "strongly recommend adjuvant endocrine therapy" = 计划 |
| medication_plan | "adjuvant endocrine therapy, 5-10 yrs" | ✅ | |
| imaging_plan | "DEXA ordered" | ✅ | |

**v19→v20b**: Stage 推断 **修复** ✅。所有字段正确。**本行最佳结果。**

---

## ROW 59 (Row 58, coral_idx)

**患者**: Stage I ER+/PR+/HER2- IDC。正在从 letrozole 转换到 exemestane（关节痛）。

**v19 问题**: current_meds 同时列出 exemestane 和 letrozole（一停一未开始）

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Stage | "Stage I (1.5 cm grade 3 IDC, negative nodes 0/5)" | ✅ | |
| **current_meds** | "exemestane ([REDACTED]), letrozole (FEMARA) 2.5 mg" | ❌ **P1** | **与 v19 相同问题**。A/P: "Discontinue Letrozole and wait 2-3 weeks before starting Exemestane." 即：letrozole 正在停用，exemestane 尚未开始。此刻两者都不应算 current。recent_changes 正确写了 "Stopped letrozole and advised to start exemestane"。 |
| response_assessment | "no evidence of recurrence" | ✅ | |
| medication_plan | "Discontinue Letrozole, wait 2-3 weeks, start Exemestane" | ✅ | |

**v19→v20b**: current_meds 时态问题 **未修复**。

---

## ROW 85 (Row 84, coral_idx)

**患者**: 61 yo female, Stage IIIA→Stage IV ER+ invasive lobular carcinoma。转移到 bone, muscle, liver, brain。fulvestrant/palbociclib 进展后，正在评估 phase 1 trial。

**v19 问题**: current_meds 空（应有 Abraxane）

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Type_of_Cancer | "ER+/PR-/HER2- invasive lobular carcinoma" | ✅ | 与 liver bx 一致 (ER+/PR-/HER2-) |
| Stage | "Originally Stage IIIA, now metastatic (Stage IV)" | ✅ | |
| Distant Metastasis | "Yes (to bone, muscle, liver, and brain)" | ✅ | 包含了 liver！ |
| **current_meds** | "fulvestrant, palbociclib, denosumab" | ⚠️ **P1** | A/P: "she progressed on first line fulvestrant/palbociclib in bone with new liver metastases" → 一线已失败。但 A/P 未明确说 "stop fulvestrant/palbociclib"，EMR 药物列表也未标记为 discontinued。这是边界情况 — 临床上失败的药物通常会停用，但 EMR 记录可能滞后。**v19 说应有 Abraxane，但原文中完全没有 Abraxane/nab-paclitaxel。v19 review 可能误判。** |
| supportive_meds | "morphine 30mg/d, ondansetron 8mg tid, prednisone 50mg/d" | ✅ | |
| goals_of_treatment | "palliative" | ✅ | |
| genetic_testing_plan | "phase 1 trial [REDACTED]+olaparib for [REDACTED] mutations" | ✅ | |
| radiotherapy_plan | "Rad Onc referral, potential GK, 2-week washout for trial" | ✅ | |

**v19→v20b**: 原 v19 的 "应有 Abraxane" 是误判（原文无 Abraxane）。current_meds 列出了进展后可能已停的药物，属于边界问题。其余字段良好。

---

## ROW 95 (Row 94, coral_idx 234)

**患者**: 49 yo female, ER+/PR+/HER2- IDC left breast。S/p 新辅助化疗（[KEYNOTE] trial + T-AC），s/p lumpectomy。计划 AC → XRT → capecitabine → endocrine therapy。

**v19 问题**: current_meds 空（v19 说应有 tamoxifen）

| 字段 | v20b 输出 | 判定 | 分析 |
|------|-----------|------|------|
| Type_of_Cancer | "ER+/PR-/HER2- IDC with DCIS" | ⚠️ P1 | 原文 biopsy: "ER/PR positive [HER2] negative"，但 v20b 写 PR-。可能是后续活检中 PR 状态改变，或模型误读。 |
| Stage | "Not mentioned in note" | ⚠️ P1 | A/P 未明确写分期，但从病理（2.1cm mass, LN+, s/p NAC with residual disease）可以推断至少 Stage II。 |
| current_meds | "" | ✅ | **v19 说应有 tamoxifen 是误判**。原文完全没有 tamoxifen。A/P 说 "presents today to start AC"（尚未开始），"strongly recommend adjuvant endocrine therapy after the above"（未来计划）。当前确实无癌症药物。 |
| response_assessment | "cancer is responding to treatment, MRI shows decrease" | ✅ | 基于 MRI breast 显示肿瘤缩小（16mm→10mm）|
| medication_plan | "Continue prilosec 40mg qd" | ⚠️ P2 | 遗漏了未来计划的 capecitabine 和 endocrine therapy |
| therapy_plan | "Start AC, axilla XRT, capecitabine, endocrine therapy" | ✅ | 完整的治疗路径 |

**v19→v20b**: current_meds="" 实际是正确的（v19 误判）。新发现 Type_of_Cancer PR 状态和 Stage 问题。

---

## 总体总结

### v19→v20b 改善情况

| Row | v19 问题 | v20b 状态 | 说明 |
|-----|---------|-----------|------|
| 0 | imaging/lab plan 遗漏 | ❌ 未修复 | 结构性问题（Orders 不在 A/P 段） |
| 7 | current_meds/response | ✅ 已修复 | current_meds=""正确 |
| 11 | Stage "Not available" | ❌ 未修复 | 尽管 A/P 有 "St IV" |
| 11 | Distant Met 缺 liver | ❌ 未修复 | |
| 11 | imaging_plan 遗漏 | ✅ 已修复 | 正确提取了 CT/MRI/bone scan 计划 |
| 17 | Stage 遗漏 | ✅ 已修复 | 推断为 Stage I |
| 58 | current_meds 时态 | ❌ 未修复 | 仍同时列出 exemestane+letrozole |
| 84 | current_meds 缺 Abraxane | ✅ v19 误判 | 原文无 Abraxane |
| 94 | current_meds 缺 tamoxifen | ✅ v19 误判 | 原文无 tamoxifen |

### 当前问题统计

| 级别 | 数量 | 具体 |
|------|------|------|
| P0 | 0 | 无严重错误 |
| P1 | 7 | Row 0 imaging/lab plan, Row 7 procedure_plan 分类, Row 11 Stage+Distant Met, Row 12 Type PR 状态, Row 58 current_meds 时态, Row 94 PR-/Stage |
| P2 | 1 | Row 94 medication_plan 遗漏未来计划 |

### 模式性问题

1. **Plan 提取 vs Orders 段**（Row 0）: 当 imaging/lab orders 在笔记头部的 Orders 段而非 A/P 中时，plan_extraction 无法捕获
2. **药物时态判断**（Row 58）: "停 A 换 B" 场景下模型同时列出两者
3. **Stage 提取鲁棒性**（Row 11）: A/P 有 "St IV" 缩写但模型未识别
4. **Distant Metastasis 完整性**（Row 11）: liver 转移有明确 biopsy 证据但被遗漏
