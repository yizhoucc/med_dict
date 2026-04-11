# V28 notool Full Run Review

> Run: v28_full_notool_20260411_073832
> Dataset: 28 samples（v26 审查中发现问题的 ROW 子集）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-12 完成（12/28），ROW 13 开始**
> 参照: `results/v27_full_notool_20260410_112141/review.md`（v27 审查）
> Results 文件: `results/v28_full_notool_20260411_073832/results.txt`

### v28 改动摘要
1. response_assessment prompt: 加了 "Not mentioned" 验证检查表 + 时间线归因规则 + 3 个新 BAD 示例
2. Type_of_Cancer prompt: 加了 goserelin fertility exception + multiple cancer HER2 规则
3. POST-ADV hook: 扩展 regex（standalone DNR/DNI + POLST + explicit wishes）
4. POST-STAGE-DISTMET hook: 新增 Stage IV + Distant Met=No → 降级
5. POST-PROCEDURE-FILTER: 扩展 blacklist（fertility/genetics counseling/imaging/medication）

### 恢复审查指南
- 待审查 ROW: **1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 20, 22, 24, 25, 33, 38, 39, 52, 57, 74, 75, 83, 88, 95**（共 28 个）
- 重点：v27 的 5 个 P1 和 29 个 P2 是否被修复？是否引入新问题？
- 每个 ROW 必须：完整读 note_text + keypoints + letter → 逐字段核对 → 与 v27 对比

### v27 已知问题（v28 需要验证修复的）

**P1（5 个）：**
| ROW | v27 问题 | v28 改动针对 |
|-----|---------|-------------|
| 8 | response "Not yet on treatment" — 有 post-neoadjuvant 病理 | prompt: post-neoadjuvant BAD 示例 |
| 10 | response = Oncotype（genomic ≠ pathologic response） | prompt: "Not mentioned" 验证 |
| 11 | response 引用旧 PET 归因到当前治疗 | prompt: 时间线归因规则 |
| 12 | Advance care 遗漏 DNR/DNI（ACTIVE problem list） | POST-ADV: standalone DNR + POLST |
| 88 | response "Not mentioned" 有明确 progression | prompt: "Not mentioned" 验证 |

**主要 P2 模式：**
| 类型 | v27 数量 | v28 改动 |
|------|---------|---------|
| procedure_plan 混入 | 7 | POST-PROCEDURE-FILTER blacklist 扩展 |
| Stage IV + Distant Met=No | 2 (ROW 83, 95) | POST-STAGE-DISTMET hook |
| goserelin→ER+ 推断 | 1 (ROW 39) | prompt: fertility exception |
| gastric HER2+ 混入 | 1 (ROW 74) | prompt: multiple cancer rule |

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 1 | — | ROW 10: response = Oncotype |
| **P2** | 11 | — | ROW 1×2, 6×2, 7×3, 8×1, 11×2, 12×1 |

### 初步 P1 快速评估（完整审查前的预览）
基于 keypoints 快速对比 v27 的 5 个 P1：
| ROW | v27 P1 | v28 response/advance care 值 | 评估 |
|-----|--------|-----|------|
| 8 | "Not yet on treatment" | "completed neoadjuvant...no residual disease in breast but 3/28 LN positive...2.4cm extranodal extension" | **✅ P1 修复** |
| 10 | response = Oncotype | "Low risk [REDACTED]." | **❌ 未修复** |
| 11 | 旧 PET 归因当前治疗 | "PET/CT showed increased metastatic activity...MRI ordered..." | **❌ 未修复** |
| 12 | Advance care 遗漏 DNR/DNI | "POLST on file. Patient has documented wishes against life support." | **✅ P1 修复** |
| 88 | "Not mentioned" | "currently on Xeloda...stable clinically...no evidence of disease progression" | **改善→P2** |

---

## 逐 Sample 问题清单（每个 ROW 独立条目）

### ROW 1 (coral_idx 140) — 0 P1, 2 P2
- P2: lab_plan 仍混入 imaging（MRI + bone scan）。同 v27
- P2: imaging_plan "Brain MRI" 仍遗漏 bone scan。同 v27
- ✅ 其余全部正确。Letter 出色。vs v27: 无变化（2 P2 未修复，但非 v28 改动目标）

### ROW 2 (coral_idx 141) — 0 P1, 0 P2 ✅
- ✅ 44yo, Lynch Syndrome + colon ca + endometrial ca + metastatic TNBC Stage IIB→IV, on irinotecan C3D1
- ✅ Type: TNBC 正确, Stage: IIB→IV 正确, Metastasis: liver/bone/chest wall 正确
- ✅ Lab: 完整记录所有严重异常（Hgb 7.7, Na 124, K 3.1, Albumin 2.1）
- ✅ Findings: 非常全面 — chest wall infection + back pain PD + sacral pain + MRI + electrolytes
- ✅ Medication_plan: comprehensive — doxycycline + morphine + effexor + NS + K + pRBC
- ✅ Response: "No specific imaging or lab results" 技术上正确（irinotecan 后无新影像）
- ✅ Letter: 全面准确 — 所有重要信息覆盖。无编造
- vs v27: 同 0/0 ✅

### ROW 3 (coral_idx 142) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, Stage IIA R breast IDC 1.7cm LN+, ER+/PR+/HER2-(IHC 2+ FISH neg), Ki-67 30-35%
- ✅ All fields correct. Genetic_testing_plan "sent and pending" 正确（v26 P2 保持修复）
- ✅ Letter: "cancer started in the milk ducts" + chemo/surgery/radiation discussed + genetic testing + PET。无编造

### ROW 4 (coral_idx 143) — 0 P1, 0 P2 ✅
- ✅ 75yo postmenopausal, ER+/PR+/HER2- IDC 2.8cm grade 2, s/p L mastectomy, on letrozole since 12/2016
- ✅ Response: "without any evidence of disease recurrence on imaging, exam, and review of systems" 正确
- ✅ Medication_plan: letrozole + calcium/vitamin D + magnesium + conditional Prolia 完整
- ✅ Imaging_plan: mammogram + DEXA + conditional brain MRI 全部捕获
- ✅ Letter: NED + osteopenia解释 + all meds + all imaging plans。通俗准确无编造

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅
- ✅ 31yo premenopausal, Stage III→IV ER+/PR+/HER2- IDC, metastatic recurrence to cervical LN + brachial plexus + possible sternal bone met
- ✅ On leuprolide + anastrozole + palbociclib。current_meds 三个药全部捕获
- ✅ Response: 出色 — 准确传达 mixed response（cervical LN↓, axillary LN↑, sternal lesion, brachial plexus LN↑, brain normal）
- ✅ Radiotherapy_plan: Rad Onc referral for symptomatic brachial plexus 正确
- ✅ Letter: mixed response 通俗化 + Rad Onc + CT/bone scan + labs monthly。无编造

### ROW 6 (coral_idx 145) — 0 P1, 2 P2
- P2: Patient type "New patient" — 应为 "Follow up"（zoladex 06/08 已由该提供者开始）。同 v27
- P2: Referral-Genetics 历史转诊（04/24/2019，Myriad negative）混入当前 referrals。同 v27
- ✅ Type: ER+/PR+/HER2- 正确。current_meds: zoladex + letrozole 正确。Goals curative 正确
- ✅ Lab: Estradiol 172 + Vitamin D 24 + CMP+CBC 完整
- ✅ Medication_plan: letrozole ≥3yr → tamoxifen + gabapentin + estradiol monthly 完整
- ✅ Letter: 无编造情绪词（v26 "anxious" 保持修复）。通俗准确

### ROW 7 (coral_idx 146) — 0 P1, 3 P2
- **P2 NEW**: Stage "Originally Stage II, now Stage III" — 应为 Stage IV。患者有 supraclavicular + mediastinal metastases（Distant Met 字段正确写 "Yes"）。POST-STAGE-DISTMET hook 错误触发（extraction 最初写 Distant Met=No → hook 降级 → 后续 gate 修正为 Yes，但 Stage 未回升）。**v28 regression**
- P2: procedure_plan "Would recheck [REDACTED]" — LVEF/echo 是 imaging 不是 procedure。同 v27
- P2: lab_plan "Would recheck [REDACTED]" — 同上。同 v27
- ✅ Response: 出色 — "probable mild progression...SUV 2.1 (was 1.8)...[REDACTED] 14.8 persistently elevated"
- ✅ Medication_plan: d/c current regimen + recommend [REDACTED] next line
- ✅ Letter: "ejection fraction of 52%" 正确（v26 编造修复保持）。无编造

### ROW 8 (coral_idx 147) — 0 P1, 1 P2 ← **v27 P1 修复！**
- **v27 P1 FIXED**: response_assessment 现在正确描述 post-neoadjuvant pathologic response — breast pCR（no residual carcinoma）+ 3/28 LN positive（2.4cm, extranodal extension）+ PET negative
- P2: procedure_plan "adjuvant AC x 4 cycles, to be followed by T-DM1" — chemo 混入 procedure。同 v27
- ✅ Type: ER-/PR-/HER2+ (IHC 3+, FISH 5.7) 正确
- ✅ Goals curative, Imaging echo before AC, Radiotherapy after AC 正确

### ROW 10 (coral_idx 149) — 1 P1, 0 P2 ← v27 P1 未修复
- **P1**: response_assessment "Low risk [REDACTED]." — 仍是 Oncotype 基因检测结果，不是 pathologic response。患者 neoadjuvant letrozole → 手术后 8.8cm 残留 + LN 受累，response 应描述这些病理发现。**同 v27，未修复**
- ✅ 其余全部正确：Type HR+/HER2- ✅, Stage II ✅, Radiotherapy ✅, DEXA ✅

### ROW 11 (coral_idx 150) — 0 P1, 2 P2 ← v27 P1→P2 改善
- **v27 P1 改善→P2**: response 去掉了 "indicating disease progression on current treatment with Faslodex"（错误归因消除）。但仍引用旧 PET（10/10/12, before Faslodex 10/16/12 开始），遗漏 A/P "Exam stable"，加入 "MRI ordered"（plan 不是 response）
- P2: imaging_plan 只有 PETCT，遗漏 MRI of lumbar/pelvis/femur。同 v27
- ✅ Type IDC ✅, Stage IIIC→IV ✅, current_meds Faslodex+Denosumab ✅, Lab 完整 ✅

### ROW 12 (coral_idx 151) — 0 P1, 1 P2 ← **v27 P1 修复！**
- **v27 P1 FIXED**: Advance care "POLST on file. Patient has documented wishes against life support." — POST-ADV hook 成功捕获 ✅
- **v27 P2 改善**: response 现在包含 "Recent MRI shows new lesions"（brain PD）+ body SD。比 v27 完整 ✅
- P2: imaging_plan 仍遗漏 Echo q6 months（A/P 写 "Echo q6 months"）。同 v27
- ✅ Medication_plan, Radiotherapy_plan, Imaging (CT+bone scan+MRI brain) 正确

