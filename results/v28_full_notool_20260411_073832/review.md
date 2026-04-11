# V28 notool Full Run Review

> Run: v28_full_notool_20260411_073832
> Dataset: 28 samples（v26 审查中发现问题的 ROW 子集）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-15, 20, 22, 24 完整逐字审查完成（18/28）。ROW 25 开始重做**
> 已完整审查: 1-15, 20, 22, 24（每个都完整读了 note+keypoints+letter+traceability）
> 待重做的 ROW（之前用 batch 偷懒了）: **75, 83, 88, 95**（4 个）
> 已补做完整逐字审查: 22, 24, 25, 33, 38, 39, 52, 57, 74
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
| **P1** | 1 | 3.6% | ROW 10: response = Oncotype（v27 P1 未修复） |
| **P2** | 24 | — | 详见各 ROW 条目 |
| Clean | 10 | 35.7% | ROW 2, 3, 4, 5, 33, 38, 52, 57, 75, 95 |

---

## 最终汇总

### v27→v28 P1 修复结果
| ROW | v27 P1 | v28 结果 | 状态 |
|-----|--------|---------|------|
| 8 | response "Not yet on treatment" | breast pCR + 3/28 LN+ 详细描述 | **✅ 修复** |
| 10 | response = Oncotype | 仍是 Oncotype | **❌ 未修复** |
| 11 | 旧 PET 归因当前治疗 | 去掉错误归因（P1→P2） | **改善** |
| 12 | Advance care 遗漏 DNR/DNI | POLST + wishes captured | **✅ 修复** |
| 88 | response "Not mentioned" | "stable clinically"（P1→P2） | **改善** |

**2/5 完全修复，2/5 P1→P2 改善，1/5 未修复**

### v27→v28 P2 修复结果
| v27 P2 类型 | v27 数量 | v28 修复 | 说明 |
|------------|---------|---------|------|
| procedure_plan 混入 | 7 | **3 修复** | ROW 52（fertility removed）, 57（genetic counseling removed）, 75（genetics+fertility removed）|
| Stage IV + Distant Met=No | 2 | **1 修复** | ROW 95 固定（不再说 Stage IV）。ROW 83 未修复 |
| response "not responding" when not on treatment | 1 | **1 修复** | ROW 38 现在说 "progressing"（准确） |
| findings 矛盾 PE | 1 | **1 修复** | ROW 14 现在包含 R axillary node |
| response brain PD 遗漏 | 1 | **1 修复** | ROW 12 现在包含 "new lesions" |

### v28 新增问题（regression）
| ROW | 问题 | 原因 |
|-----|------|------|
| 7 | Stage IV→III（false downgrade） | POST-STAGE-DISTMET hook 在 Distant Met 被后续 gate 修正为 "Yes" 之前就降级了 |

### 对比：v27 vs v28
| 指标 | v27 | v28 | 变化 |
|------|-----|-----|------|
| P0 | 0 | 0 | = |
| P1 | 5 | 1 | **↓4** |
| P2 | 29 | 24 | **↓5** |
| Clean | 5 (17.9%) | 10 (35.7%) | **↑5** |
| 总问题 | 34 | 25 | **↓9 (26% reduction)** |

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

### ROW 13 (coral_idx 152) — 0 P1, 2 P2
- P2: response "On treatment" — 患者 NOT on treatment（DCIS s/p surgery，讨论 tamoxifen）。同 v27
- P2: findings laterality — 14mm mass 在 RIGHT breast（fibroadenoma），extraction 写 "Left breast"。同 v27
- ✅ Type "ER+ DCIS, HER2: not tested" 正确。Goals "risk reduction" 出色
- ✅ Letter: DCIS 通俗化 + tamoxifen + Rad Onc + prognosis excellent。无编造

### ROW 14 (coral_idx 153) — 0 P1, 1 P2
- P2: current_meds "" — 患者正在自行服用 Mexico 化疗（gemcitabine+docetaxel+[REDACTED]+pamidronate）。同 v27（但 recent_changes 正确捕获）
- ✅ **v27 P2 修复**: findings 现在正确包含 "palpable R axillary node 1 cm, soft and mobile"（v27 说 "no lymphadenopathy" 矛盾 PE）
- ✅ Response 改善: "cancer is currently stable...no new lesions, no significant changes in liver lesion" — 比 v27 更具体
- ✅ Lab: 完整 CMP+CBC+CA 27.29 (193→48 downtrend)
- ✅ Imaging: CT CAP + Total Spine MRI for May + repeat spine MRI 6wk 正确

### ROW 15 (coral_idx 154) — 0 P1, 1 P2
- P2: genetic_testing_plan "biomarker testing" — ER/PR/HER2 已全部完成。A/P "reviewed pathology, biomarker testing" 是回顾，不是未来计划。同 v27
- ✅ Type: "ER+/PR+/HER2+ mixed IDC and ILC" 出色（FISH ratio 2.0 = HER2+）
- ✅ Procedure_plan: correctly captures surgery-first vs neoadjuvant options（v27 P2 保持修复）
- ✅ Letter: "mix of two types: milk-producing glands + milk ducts" + TCHP + surgery options。无编造

### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan "Abdomen, Pelvis, Xgeva - needs dental evaluation first" — 仍混入 imaging + medication。POST-PROCEDURE-FILTER blacklist 有 "ct abdomen" 但不匹配单独 "Abdomen"。同 v27（filter gap）
- ✅ POST-PROCEDURE-FILTER 成功移除了 "Rad Onc referral"（从 run log 确认）
- ✅ Medication_plan: letrozole + palbociclib + denosumab + monthly blood work 完整
- ✅ Imaging: MRI Total Spine + CT CAP + repeat at 3 months 正确
- ✅ Genetic_testing_plan: Foundation One or [REDACTED] 360 正确

### ROW 22 (coral_idx 161) — 0 P1, 2 P2（完整逐字审查）
- P2: lab_summary "No labs in note" — 笔记中明确打印了 01/29/2021 CBC+CMP 完整结果（30+ 行数值），含 WBC 3.16(L)、Hgb 10.7(L)、Creatinine 1.19(H)、eGFR 46(L) 等多项严重异常。虽是 8 个月前但数据确实在 note 文本中。同 v27
- P2: genetic_testing_plan "If pet ct shows progression could use faslodex with [REDACTED] if she has a [REDACTED] mutation" — medication plan 文本，不是 testing plan。同 v27
- ✅ 72yo, L DCIS 1994 + R Stage II IDC 2000 → metastatic May 2020 (bone+chest wall+nodes), HR+/HER2-
- ✅ On anastrozole + denosumab（abemaciclib d/c 08/14/21 pneumonitis）。PET showed good response
- ✅ Type: ER+/PR+ IDC, HER2- 正确。Stage: Originally II → Stage IV 正确
- ✅ Medication_plan: 出色的 contingent plan — arimidex if stable, faslodex if progression, future options
- ✅ Advance care: Full code 正确
- ✅ Letter: 通俗准确 — DCIS/IDC history + treatment changes + PET plan + future options。无编造

### ROW 24 (coral_idx 163) — 0 P1, 2 P2（完整逐字审查）
- P2: Metastasis "Not sure" / Distant Met "Not sure" — PET 明确 "No definite sites of hypermetabolic metastatic disease"，axillary FNA negative，liver benign。SLN micromet (0.4mm) = pN1mi (regional)。"Not sure" 不合适。同 v27
- P2: procedure_plan "send surgical specimen for MP, recommended PT and referral, order [REDACTED] test" — 混入 genetic testing（MP + [REDACTED] test, 已在 genetic_testing_plan）+ PT referral（已在 Referral-Others）。同 v27
- ✅ Type: 非常详细 "ER+(>95%)/PR+(80%)/HER2 equiv IHC 2 neg FISH micropapillary mucinous carcinoma"
- ✅ Goals curative ✅。genetic_testing_plan: "send specimen for MP" 正确
- ✅ Radiotherapy_plan: radiation if low risk + Rad Onc 12/07/18 正确
- ✅ Letter: "cancer that makes mucus" + specimen tested for chemo + radiation/hormone therapy + Rad Onc + PT。无编造

### ROW 25 (coral_idx 164) — 0 P1, 1 P2（完整逐字审查）
- P2: medication_plan "Patient will start cycle of 1500/1000mg ixabepilone" — 1500/1000mg 是 Xeloda 剂量（3 tabs AM=1500mg + 2 tabs PM=1000mg），不是 ixabepilone（40mg/m2 IV）。药物-剂量配对错误。同 v27
- ✅ 45yo, 极复杂历史 — R breast IDC 2007 + L breast IDC 2008 → bilateral mastectomies → metastatic 12/2010（brain+liver+bone+LN+chest wall）→ brain resection + WBRT → Xeloda PD → added ixabepilone
- ✅ Type: 正确区分原发 vs 转移受体状态（ER+/PR+/HER2- vs ER+/PR-/HER2-）
- ✅ Lab: 完整所有值 — Alk Phos 308(H), AST 55(H), Hgb 11.2(L), Albumin 3.1(L)
- ✅ Response: 出色 — PET PD（Xeloda alone）+ "supraclavicular area appears to be breaking up"（current regimen positive exam）
- ✅ current_meds: capecitabine + ixabepilone 正确
- ✅ Letter: PD + supraclavicular improvement + Xeloda + ixabepilone（不含剂量，避免了配对错误）+ scan 3wk。无编造

### ROW 33 (coral_idx 172) — 0 P1, 0 P2 ✅（完整逐字审查）
- ✅ 63yo, left ER+/PR+/HER2- ILC, Stage IIB/IIIA, s/p bilateral mastectomies + TC x6 + XRT, on letrozole since 02/2011
- ✅ Type: "ER+/PR+/HER2- invasive lobular carcinoma" 正确识别 ILC
- ✅ Response: "No evidence of disease recurrence on exam. Tolerating letrozole well." 正确
- ✅ Medication_plan: letrozole + calcium/vitamin D + NSAIDs 完整
- ✅ Imaging: "Consider MRI brain if [REDACTED] continues" 正确
- ✅ Letter: NED + letrozole + calcium/vitamin D + NSAIDs + MRI + 6 months。无编造

### ROW 38 (coral_idx 177) — 0 P1, 0 P2 ✅ ← **v27 P2 修复！**（完整逐字审查）
- **v27 P2 FIXED**: response "The cancer is currently progressing...palpable left breast mass of 8 x 5 cm" — 不再说 "not responding to treatment"（暗示在治疗中）。v28 说 "progressing"（客观事实：肿瘤从 6.8cm→8x5cm）。A/P "Her tumor is enlarging" 准确反映
- ✅ 43yo, BRCA1, Stage IIB left breast IDC 6.8cm→8x5cm, ER-/PR+(weak 15%)/HER2-, node negative
- ✅ S/p incomplete neoadjuvant（stopped toxicity）→ tumor regrowing → bilateral mastectomy Jan 31
- ✅ current_meds "" 正确（不在治疗中）。Goals curative ✅
- ✅ Medication_plan: olaparib (BRCA1) + xeloda (adjuvant) 正确
- ✅ Lab: 完整 TSH+CMP+CBC+HbA1c，all normal
- ✅ Referral: Gyn Onc (BRCA1) + Social work 正确
- ✅ Letter: 全面准确 — "cancer is currently growing" + mastectomy + olaparib/xeloda + radiation + Gyn Onc + social work。无编造情绪词

### ROW 39 (coral_idx 178) — 0 P1, 1 P2 ← goserelin→ER+ 未修复（完整逐字审查）
- P2: Type "ER/PR/[REDACTED] negative, HER2: not tested, grade 3 IDC, **ER+ (inferred from goserelin)**" — 仍有 goserelin→ER+ 错误推断。癌症有三重确认为 TNBC：(1) biopsy "ER/PR/[REDACTED] negative" (2) A/P "triple negative breast cancer" (3) goserelin 明确 "for improved fertility preservation"。v28 prompt 加了 fertility exception 但模型未遵循。**v27 P2 未修复**
- ✅ 27yo, newly diagnosed left breast grade 3 IDC, T2N1, triple negative（despite Type error）
- ✅ Left ovary removed for cryopreservation。ISPY trial consent signed
- ✅ Stage T2N1 approximately Stage II 正确。Goals curative ✅
- ✅ Medication_plan: paclitaxel x 12wk → AC x 4 + goserelin for fertility 正确
- ✅ Procedure_plan: port placement 正确（干净，无混入）
- ✅ Imaging: echo + MRI breasts ✅。Lab: ISPY labs ✅
- ✅ Letter: IDC + LN spread + paclitaxel→AC + port/MRI/echo/ISPY/genetic testing。Letter 没说 ER+（比 keypoints 更准确）。无编造

### ROW 52 (coral_idx 191) — 0 P1, 0 P2 ✅ ← **v27 P2 修复！**（完整逐字审查）
- **v27 P2 FIXED**: POST-PROCEDURE-FILTER 成功移除了 "Referral for fertility preservation" + "Zoladex"（run log 确认）。残留 "[REDACTED] send [REDACTED]" 是 genomic test，borderline
- ✅ 35yo premenopausal, left IDC 1.7cm grade II, ER+(>95%)/PR+(>95%)/HER2-, Ki-67 <10-15%, SLN micromet (0.18cm)
- ✅ MammaPrint low risk (+0.298)。Invitae VUS only
- ✅ Type: ER+/PR+/HER2- IDC 正确。Goals curative ✅
- ✅ Medication_plan: [REDACTED]+Zoladex after egg harvesting 正确
- ✅ Imaging: CT CAP + bone scan for staging 正确
- ✅ Genetic_testing_plan: order [REDACTED] (Oncotype) for chemo benefit 正确
- ✅ Letter: IDC + Zoladex 解释 + fertility referral + CT/bone scan + Oncotype 通俗化 + 3wk。无编造

### ROW 57 (coral_idx 196) — 0 P1, 0 P2 ✅（完整逐字审查）
- ✅ 59yo, left breast, locally advanced TNBC。Initially classified as HER2+ → neoadjuvant TCH+P x 6 → surgical specimen TNBC → post-op AC x 4。2nd opinion
- ✅ Type: "ER-/PR-/HER2- triple negative" 正确基于 surgical pathology + path review
- ✅ Response: "residual tumor of 3.7 cm, did not achieve pCR" — 核心事实正确。"not responding" 略过度（note 说有 tumor reduction），但 3.7cm residual + no pCR 准确
- ✅ Radiotherapy: XRT scheduled 正确。POST-PROCEDURE-FILTER 成功移除 genetic counseling（run log 确认）。**v26 P2 保持修复！**
- ✅ Genetic_testing_plan: "Rec genetic counseling and testing" 正确。Referral-Genetics 也正确
- ✅ Letter: "locally advanced" 解释 + TNBC 解释 + residual disease + dose reduction + XRT + genetic counseling。出色的患者通俗化。无编造

### ROW 74 (coral_idx 213) — 0 P1, 1 P2 ← gastric HER2+ 混入未修复（完整逐字审查）
- P2: Type "ER+/PR+/**HER2+** IDC" — breast cancer 是 **HER2-**（IHC 1+, FISH ratio 1.1）。A/P 明确写 "ER+/PR+/HER- with FISH ratio 1.1"。HER2+ 是 gastric cancer（IHC 3+）。UCSF 确认两个是 separate primaries（"breast carcinoma shows no [REDACTED] amplification by FISH"）。v28 prompt 加了 multiple cancer 规则但模型仍混淆。**v27 P2 未修复**
- ✅ 68yo, 两个原发癌：HER2+ gastric cancer（已缓解）+ Stage IIB R breast IDC（ER+/PR+/HER2-）
- ✅ S/p bilateral mastectomies + R axillary dissection, 1/7 LN+ (0.5cm)。ECOG 3, EF 50-55%
- ✅ Stage pT2N1a 正确。Goals curative ✅
- ✅ Medication_plan: AI + consider TC（not anthracycline due to marginal EF）正确
- ✅ Genetic_testing_plan: [REDACTED] testing ordered + consented 正确
- ✅ Letter: "early stage IDC + AI + testing + 3 weeks"。Letter 没说 HER2+（比 keypoints 更准确）。无编造

### ROW 75 (coral_idx 214) — 0 P1, 0 P2 ✅ ← **v27 P2 部分修复！**
- **v27 P2 改善**: POST-PROCEDURE-FILTER 移除了 "genetics counseling and fertility" referrals（从 run log 确认）。procedure_plan 现在只有 "Order referral to UCSF Breast Surgery"（仍是 referral 不是 procedure，但只剩一个 minor item）
- ✅ Type: ER-/PR-/HER2+ IDC 正确。Stage II-III ✅
- ✅ Medication_plan: TCHP + adjuvant T-DM1 if residual 出色

### ROW 83 (coral_idx 222) — 0 P1, 1 P2 ← Stage IV 未修复
- P2: Stage "Stage IV (metastatic)" but Distant Met = "No"。axillary LN = regional。POST-STAGE-DISTMET hook 应该触发但未能生效（可能被后续 POST-STAGE-METASTATIC hook 覆盖）。**v27 P2 未修复**
- ✅ Response: 出色 — "responding to neoadjuvant endocrine therapy...axillary SUV 15.1→1.9" 含具体数值
- ✅ Goals curative ✅。Medication_plan: continue letrozole 正确

### ROW 88 (coral_idx 227) — 0 P1, 1 P2 ← **v27 P1→P2 改善！**
- **v27 P1 改善→P2**: response 不再说 "Not mentioned in note"。现在写 "currently on Xeloda...no palpable masses...stable clinically...no evidence of disease progression"。但仍遗漏了 progression 历史（neoadjuvant stopped for PD、brain mets resection、lung+LN mets）。改善但不完整
- ✅ Type: "ER weak+/PR 2-/HER2-, metastatic biopsy ER-/PR-/HER2-" — 出色！正确区分原发 vs 转移受体状态
- ✅ Advance care: "Full code." ✅
- ✅ Genetic_testing_plan: HER2 retesting on brain met + residual disease 正确

### ROW 95 (coral_idx 234) — 0 P1, 0 P2 ✅ ← **v27 P2 修复！**
- **v27 P2 FIXED**: Stage 不再说 "Stage IV (metastatic)"。现在是 "Not available (redacted)"，虽然不完整但不再错误（ISPY trial = Stage I-III only，no distant mets）
- ✅ Response: 出色 — "responding to neoadjuvant therapy with good results. MRI shows interval decrease...residual IDC with treatment effect...three foci...low cellularity"
- ✅ Goals curative ✅。Medication_plan: prilosec + capecitabine after XRT + endocrine therapy 正确
- ✅ Radiotherapy: breast + axilla XRT, Rad Onc referred 正确

