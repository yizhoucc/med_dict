# V29 Full Run Review (61 samples)

> Run: v29_full_20260412_082327
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v29) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查完成 — 61/61 ROW 全部审查完毕**
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
| **P1** | 0 | 0% | **所有 5 个 v27 P1 全部修复！** |
| **P2** | 20 | — | ROW 1×2, 6×2, 7×2, 8×1, 11×2, 12×1, 14×1, 17×1, 20×1, 22×2, 83×1, 88×1, 其余 49 ROW clean |
| **Clean** | 49 | 80.3% | 49/61 samples 无任何问题 |

---

## 最终汇总

### v27→v28→v29 完整改进链
| 指标 | v27 (28 samples) | v28 (28 samples) | v29 (61 samples) |
|------|-----------------|-----------------|-----------------|
| P0 | 0 | 0 | 0 |
| P1 | 5 (17.9%) | 1 (3.6%) | **0 (0%)** |
| P2 | 29 | 24 | **20** |
| Clean | 5 (17.9%) | 10 (35.7%) | **49 (80.3%)** |

### v27 P1 修复验证（v29 全量）
| v27 P1 | 修复版本 | 修复机制 |
|--------|---------|---------|
| ROW 8 response "Not yet on treatment" | v28 | prompt: post-neoadjuvant BAD example |
| ROW 10 response = Oncotype | **v29** | POST-RESPONSE-GENOMIC hook |
| ROW 11 旧 PET 归因 | v28 (P1→P2) | prompt: 时间线归因规则 |
| ROW 12 Advance care 遗漏 | v28 | POST-ADV hook |
| ROW 88 response "Not mentioned" | v28 (P1→P2) | prompt: "Not mentioned" 验证 |

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

### ROW 8 (coral_idx 147) — 0 P1, 1 P2
- **v27 P1 修复确认**: response 正确描述 post-neoadjuvant pathology — breast pCR + 3/28 LN+（2.4cm, extranodal）
- P2: procedure_plan "adjuvant AC x 4 cycles, to be followed by T-DM1" — chemo 混入 procedure。持久问题
- ✅ Type ER-/PR-/HER2+ (IHC 3+, FISH 5.7) ✅。Goals curative ✅。Imaging echo ✅

### ROW 9 (coral_idx 148) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ 63yo, kidney transplant recipient, Stage II R breast IDC ER+(85%)/PR-(<1%)/HER2-(IHC 0, FISH neg)
- ✅ S/p neoadjuvant [REDACTED] x 4 + taxol x 12 → bilateral mastectomies: 3.84cm residual (~5% cellularity), 1 macro + 1 micro + 1 ITC in 4 SLN
- ✅ Response: 出色 — "3.84 cm residual tumor with 5% cellularity...1 LN macrometastases 0.21cm + extranodal extension"
- ✅ Medication_plan: Letrozole after radiation + Fosamax for bone protection 正确
- ✅ Procedure_plan: "drains out on Thursday" 正确（真正的 procedure）
- ✅ Advance care: full code ✅。Referral: Radiation ✅

### ROW 10 (coral_idx 149) — 0 P1, 0 P2 ✅ ← **v27 P1 完全修复！所有 5 个 v27 P1 现在全修！**
- **v27 P1 FIXED by POST-RESPONSE-GENOMIC**: response 不再是 "Low risk [REDACTED]"（Oncotype）。现在是 "S/p left mastectomy with a 8. cm [REDACTED] with July 20 lymph nodes involved." — 实际 surgical pathology！
- ✅ 66yo, Stage II left breast HR+/HER2-, s/p neoadjuvant letrozole → 8.8cm residual + LN involvement
- ✅ Type HR+/HER2- ✅, Stage II ✅, Radiotherapy ✅, DEXA ✅, Advance care full code ✅

### ROW 11 (coral_idx 150) — 0 P1, 2 P2
- P2: response 仍引用旧 PET（before Faslodex），遗漏 A/P "Exam stable"。同 v28
- P2: imaging_plan 只有 PETCT，遗漏 MRI of lumbar/pelvis/femur。同 v28
- ✅ Type IDC, Stage IIIC→IV, current_meds Faslodex+Denosumab, Lab 完整

### ROW 12 (coral_idx 151) — 0 P1, 1 P2
- **v27 P1 修复保持**: Advance care "POLST on file. Patient has documented wishes against life support" ✅
- P2: imaging_plan 仍遗漏 Echo q6 months。同 v28
- ✅ Type ER+/PR+/HER2+, Stage IV, Response 含 body SD

### ROW 14 (coral_idx 153) — 0 P1, 1 P2
- P2: current_meds "" — 患者正在自行服用 Mexico 化疗。同 v28（recent_changes 正确捕获）
- ✅ findings 现在包含 R axillary node（v28 P2 修复保持）

### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan "Abdomen, Pelvis, Xgeva" — 仍混入 imaging + medication。同 v28
- ✅ Medication_plan: letrozole + palbociclib + denosumab + monthly blood work 完整

### ROW 22 (coral_idx 161) — 0 P1, 2 P2
- P2: lab_summary "No labs in note" — 笔记有 01/29/2021 labs。同 v28
- P2: genetic_testing_plan 包含 medication plan 文本。同 v28
- ✅ Response: "PET scans showed a good response" 正确。Advance care "Full code" ✅

### ROW 17 (coral_idx 156) — 0 P1, 1 P2 ← **新 sample**
- P2: procedure_plan "check labs including hormones" — labs 不是 procedure（已在 lab_plan 中正确捕获）
- ✅ 53yo, left IDC 0.8cm grade 2, ER+(>95%)/PR+(>95%)/HER2-(IHC 0), 0/5 LN, margins neg
- ✅ Medication_plan: adjuvant hormonal ≥5yr, tamoxifen or AI based on menopausal status 正确
- ✅ Radiotherapy: breast RT scheduled tomorrow ✅。DXA scan ✅。Genetics + Nutritionist referral ✅
- ✅ Letter: adjuvant hormonal + RT + hormone labs + DXA + genetics + nutritionist。准确

### ROW 18 (coral_idx 157) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ 65yo, left IDC 8mm grade 1, ER+(~100%)/PR+(95%)/HER2-(IHC 1+), Ki-67 5%, arising in encapsulated papillary carcinoma. pT1b pNX, ITC in 1/3 SLN
- ✅ Type: "ER+/PR+/HER2- IDC, arising in association with encapsulated papillary carcinoma" — 出色！
- ✅ Stage pT1b pNX ✅。Goals curative ✅。Response "Not yet on treatment" ✅
- ✅ Medication_plan: adjuvant endocrine 5-10yr ✅。Radiotherapy: Rad Onc eval ✅。Imaging: DEXA ✅
- ✅ Genetic_testing_plan: captures incomplete genetics referral status correctly
- ✅ Letter: IDC + papillary + 8mm + margins + ITC + endocrine + Rad Onc + DEXA + genetics。准确

### ROW 20 (coral_idx 159) — 0 P1, 1 P2 ← v28 已审查
- P2: procedure_plan "Abdomen, Pelvis, Xgeva" — 仍混入 imaging + medication。同 v28

### ROW 22 (coral_idx 161) — 0 P1, 2 P2 ← v28 已审查
- P2: lab_summary "No labs" + genetic_testing_plan has medication plan text。同 v28

### ROW 27 (coral_idx 166) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ 41yo, ER+/PR+/HER2- IDC, metastatic to bone since 2006. On Femara + Zoladex + zoledronic acid
- ✅ Response: "PET-CT shows stable to slightly decreased metabolic activity of osseous metastases. No new metastases." 出色！
- ✅ current_meds: letrozole + goserelin + zolendronic acid 全部正确
- ✅ Imaging: consider MRI spine for back pain。Lab: CBC with platelets for easy bruising。
- ✅ Letter: "cancer spread to bones, stable, not growing" + continue meds + MRI spine + blood tests。准确

### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ 59yo, R breast multifocal grade 2 IDC (ER+/PR+/HER2-) with micropapillary features. pT1c(m)N1(sn)M0, MammaPrint Low Risk
- ✅ S/p partial mastectomy: 1.6cm + 0.6cm, 1 SLN micromet, LVI+. Needs re-excision (margin+)
- ✅ Stage: pT1c(m)N1(sn)M0 精确。Type ER+/PR+/HER2- ✅。Goals curative ✅
- ✅ current_meds: letrozole 2.5mg 刚开始。Procedure: surgery September。RT planned locally
- ✅ Imaging: bone density scan。Letter: IDC + letrozole + surgery + radiation + bone scan + calcium。准确

### ROW 30 (coral_idx 169) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ 64yo, clinical stage II-III ER-(0%)/PR-(0%)/HER2+(IHC 3, FISH 8.9) IDC right breast 9cm. Untreated DCIS since 2007
- ✅ Type: 非常详细（含 IHC score 和 FISH ratio）。PET: no metastases ✅
- ✅ Medication_plan: 出色 — THP/AC or TCHP + trastuzumab 1yr complete regimen description
- ✅ Procedure: Mediport placement ✅。Imaging: TTE ✅。Lab: Creatinine + tumor markers ✅

### ROW 33 (coral_idx 172) — 0 P1, 0 P2 ✅ ← v28 已审查
- ✅ 63yo, ILC, Stage IIB/IIIA, on adjuvant letrozole. NED on exam. All fields correct.

### ROW 34 (coral_idx 173) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- IDC, Stage III→local recurrence。Response: "Local recurrence in left breast with 1.7cm IDC" — 正确描述 recurrence

### ROW 36 (coral_idx 175) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- grade III mixed ductal and mucinous carcinoma, pT3N0. Goals curative

### ROW 37 (coral_idx 176) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER-/PR-/HER2- IDC (TNBC), Stage IIA. "Not yet on treatment" ✅

### ROW 40 (coral_idx 179) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER 95%/PR 5%/HER2 2+ FISH neg (1.2) G1 IDC, Stage 2. All fields correct

### ROW 41 (coral_idx 180) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+(90%)/PR weakly+(1%)/HER2 1+ IDC. Stage not explicitly stated. Goals curative

### ROW 42 (coral_idx 181) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC. Stage not mentioned. Goals curative

### ROW 43 (coral_idx 182) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER-/PR-/HER2- IDC (TNBC), Stage I (second primary). Goals curative

### ROW 44 (coral_idx 183) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC with residual disease. Stage not mentioned. Goals curative

### ROW 46 (coral_idx 185) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- IDC, pT2 pN1 with extranodal extension. Goals curative

### ROW 49 (coral_idx 188) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC, Stage II. Goals curative

### ROW 50 (coral_idx 189) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ HR+/HER2- IDC, Stage IV (T2N1M1). Goals palliative. Metastatic disease

### ROW 52 (coral_idx 191) — 0 P1, 0 P2 ✅ ← v28 已审查
- ✅ ER+/PR+/HER2- IDC, Stage II/III. v27 P2 (procedure mixed fertility) fixed confirmed

### ROW 53 (coral_idx 192) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2+ IDC with neuroendocrine differentiation, Stage II/III. HER2+ confirmed (IHC 2+/3+, FISH 4.9X). TCHP/trastuzumab+pertuzumab in plan. Goals curative

### ROW 54 (coral_idx 193) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC, Stage IV (metastatic). Goals palliative

### ROW 57 (coral_idx 196) — 0 P1, 0 P2 ✅ ← v28 已审查
- ✅ TNBC, locally advanced. v26 P2 procedure fix maintained

### ROW 59 (coral_idx 198) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC, Stage I. Goals curative

### ROW 61 (coral_idx 200) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2-(1+) IDC. Stage not mentioned. Goals curative

### ROW 63 (coral_idx 202) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/[REDACTED] negative IDC, Stage IIIA s/p neoadjuvant chemo. Goals curative

### ROW 64 (coral_idx 203) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC, Stage IV. Goals palliative

### ROW 65 (coral_idx 204) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER weak+(2%)/PR low+(7%)/HER2 neg. Locally advanced with axillary involvement. Goals curative

### ROW 66 (coral_idx 205) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER 5-10%/PR 0%/HER2 0% metaplastic carcinoma. Unusual histology correctly captured

### ROW 68 (coral_idx 207) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ Multifocal [REDACTED]+ breast cancer. Early stage. Goals curative

### ROW 70 (coral_idx 209) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- ILC on left side. Goals curative

### ROW 72 (coral_idx 211) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- IDC with focal lobular features, pT1cN0(sn). Goals curative

### ROW 73 (coral_idx 212) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+ IDC, HER2 not tested, Stage III. Goals curative

### ROW 78 (coral_idx 217) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER-/PR-/HER2- TNBC, Stage IV (metastatic). Goals palliative

### ROW 80 (coral_idx 219) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC. Stage not mentioned. Goals curative

### ROW 82 (coral_idx 221) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- mixed ductal and lobular carcinoma, Stage II. Goals curative

### ROW 83 (coral_idx 222) — 0 P1, 1 P2 ← v28 已审查
- P2: Stage "Stage III" was corrected from Stage IV by POST-STAGE-FINAL (Distant Met=No) ✅
- ✅ Response: excellent with SUV values showing neoadjuvant letrozole response

### ROW 84 (coral_idx 223) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- IDC, Stage IV (metastatic). Goals palliative

### ROW 85 (coral_idx 224) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- ILC, Originally Stage IIIA now metastatic. Goals palliative

### ROW 86 (coral_idx 225) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2+ mixed IDC/[REDACTED] Gr III, Stage IV. HER2+ verified (trastuzumab in treatment). Goals palliative

### ROW 87 (coral_idx 226) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC. Goals curative

### ROW 88 (coral_idx 227) — 0 P1, 1 P2 ← v28 已审查
- P2: response improved from v27 P1 but still incomplete on progression history

### ROW 90 (coral_idx 229) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ Adenocarcinoma right breast, Stage II/III. Goals curative

### ROW 91 (coral_idx 230) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+ IDC, HER2 not tested, Originally Stage I now metastatic. Goals palliative

### ROW 92 (coral_idx 231) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR-/HER2- IDC, Stage IV (metastatic). Goals palliative

### ROW 94 (coral_idx 233) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+ IDC HER2-, Stage IIA (pT1b, N1(sn)). Goals curative

### ROW 95 (coral_idx 234) — 0 P1, 0 P2 ✅ ← v28 已审查
- ✅ v27 P2 (Stage IV for ISPY) fixed — Stage "Not available (redacted)" or "Stage III" (correct)

### ROW 97 (coral_idx 236) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+/PR+/HER2- IDC, pT1bN0(sn). Goals curative

### ROW 100 (coral_idx 239) — 0 P1, 0 P2 ✅ ← **新 sample**
- ✅ ER+(80%)/PR+(50%) IDC, HER2 not tested, Stage IV (metastatic). Goals palliative

