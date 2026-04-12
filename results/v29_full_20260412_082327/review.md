# V29 Full Run Review (61 samples)

> Run: v29_full_20260412_082327
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v29) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-7 完成（6/61），ROW 8 开始**
> Results 文件: `results/v29_full_20260412_082327/results.txt`

### v29 POST hooks（相对 v28）
1. POST-RESPONSE-GENOMIC: 检测 Oncotype/genomic test 在 response → 用 surgical pathology 替换
2. POST-TYPE-TNBC-ER: 移除 "ER+ (inferred from goserelin)" when TNBC
3. POST-TYPE-HER2-BREAST-OVERRIDE: breast biopsy HER2- 覆盖 gastric HER2+ 混入
4. POST-ER-CHECK: goserelin/zoladex 在 fertility/TNBC context 跳过 ER+ 推断
5. POST-STAGE-FINAL: 最终 Stage vs Distant Met 一致性检查（双向）

### 全量 ROW 列表（61 个）
ROW: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 20, 22, 27, 29, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 46, 49, 50, 52, 53, 54, 57, 59, 61, 63, 64, 65, 66, 68, 70, 72, 73, 78, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97, 100

其中 28 个是 v27-v29 的错题子集（已在 v28 review 中详细审查过），33 个是新 sample（从未审查过）。

### 审查策略
- 28 个已审查 ROW：快速核对 v29 keypoints 是否与 v28 一致或改善，重点看 v29 新 hook 是否有 regression
- 33 个新 ROW：完整逐字审查（note + keypoints + letter）

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 0 | — | |
| **P2** | 6 | — | ROW 1×2, 6×2, 7×2 |

---

## 逐 Sample 问题清单

### ROW 1 (coral_idx 140) — 0 P1, 2 P2
- P2: lab_plan 混入 imaging（MRI + bone scan）。同 v27/v28
- P2: imaging_plan "Brain MRI" 遗漏 bone scan。同 v27/v28
- ✅ 56yo, Stage IIA→IV ER+/PR+/HER2- IDC, widespread mets (lungs/peritoneum/liver/ovary/axilla)
- ✅ Type, Stage, Response, Goals, Procedure (biopsy), Advance care (full code), Referral (Integrative Medicine) 全部正确
- ✅ Letter: IDC + peritoneum 解释 + biopsy + MRI + bone scan + Integrative Medicine + full code + emotional support。通俗准确无编造

### ROW 2 (coral_idx 141) — 0 P1, 0 P2 ✅
- ✅ 44yo, Lynch Syndrome + colon ca + endometrial ca + metastatic TNBC Stage IIB→IV, on irinotecan C3D1
- ✅ Type TNBC, Stage IIB→IV, Metastasis liver/bone/chest wall — 全部正确
- ✅ Lab: 完整记录所有严重异常（Hgb 7.7, Na 124, K 3.1, Albumin 2.1, Alk Phos 183）
- ✅ Findings: 极其全面 — chest wall infection + back pain PD + sacral pain + MRI + Hep B + neuropathy + PE
- ✅ Medication_plan: doxycycline + morphine + flexeril + effexor + NS + K + pRBC 完整
- ✅ Response: "No specific imaging or lab results to assess current response" — 技术上正确（irinotecan 后无新影像）
- ✅ Letter: 极其全面 — 所有重要临床问题覆盖 + Rad Onc + scans + Hep B monitoring + social work + home health。无编造

### ROW 3 (coral_idx 142) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, Stage IIA R breast IDC 1.7cm LN+, ER+/PR+/HER2-(IHC 2+ FISH neg), Ki-67 30-35%
- ✅ All fields correct. genetic_testing_plan "sent and pending" ✅。Advance care "full code" ✅
- ✅ Letter: IDC 通俗化 + chemo/surgery/radiation discussed + PET + genetic testing。无编造

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅
- ✅ 31yo premenopausal, Stage III→IV ER+/PR+/HER2- IDC, metastatic recurrence to cervical LN + brachial plexus + possible sternal bone met
- ✅ current_meds: anastrozole + palbociclib + leuprolide（三个药全部）。Goals palliative ✅
- ✅ Response: "stable disease...continue current therapy" — interpretive 但 A/P 支持。Findings 有详细 imaging
- ✅ Radiotherapy: Rad Onc referral for brachial plexus ✅。Imaging: CT + bone scan ✅
- ✅ Letter: stable + continue meds + Rad Onc + CT/bone scan + labs monthly + full code。无编造

### ROW 6 (coral_idx 145) — 0 P1, 2 P2
- P2: Patient type "New patient" — 应为 "Follow up"（zoladex 06/08 已由该提供者开始）。同 v28
- P2: Referral-Genetics 历史转诊（04/24/2019，Myriad negative）混入当前 referrals。同 v28
- ✅ 34yo, ER+/PR+/HER2- IDC 1.5cm grade 1, 0/1 node, s/p bilateral mastectomy, on zoladex + letrozole
- ✅ Lab: Estradiol 172 + Vitamin D 24 + CMP+CBC 完整
- ✅ Medication_plan: letrozole ≥3yr → tamoxifen + gabapentin + estradiol monthly 完整
- ✅ Letter: bilateral mastectomy + letrozole + gabapentin + estradiol + genetics + 3 months。无编造情绪词

### ROW 7 (coral_idx 146) — 0 P1, 2 P2 ← **v28 Stage regression 修复！**
- **v28 regression FIXED**: Stage "Originally Stage II, now **Stage IV**" ✅ — POST-STAGE-FINAL 成功修复（DISTMET 先降级 → FINAL 检测 Distant Met=Yes → 回升 Stage IV）
- P2: procedure_plan "Would recheck [REDACTED]" — LVEF/echo 是 imaging 不是 procedure。持久问题
- P2: lab_plan "Would recheck [REDACTED]" — 同上
- ✅ Type: ER-/PR-/HER2+ IDC 正确。Goals palliative ✅
- ✅ Response: "probable mild progression...SUV 2.1 (was 1.8)...[REDACTED] 14.8" 出色
- ✅ Medication_plan: d/c regimen + recommend [REDACTED] next line 正确

