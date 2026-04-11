# V28 notool Full Run Review

> Run: v28_full_notool_20260411_073832
> Dataset: 28 samples（v26 审查中发现问题的 ROW 子集）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-3 完成（3/28），ROW 4 开始**
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
| **P1** | 0 | — | |
| **P2** | 2 | — | ROW 1×2 |

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

