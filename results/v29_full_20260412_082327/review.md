# V29 Full Run Review (61 samples)

> Run: v29_full_20260412_082327
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v29) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查中 — ROW 1-91 完成（57/61），ROW 92 待审查（ROW 86-100 重做中，剩余 92, 94, 95, 97, 100）**
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
| **P2** | 81 | — | ...86×1, 87×0, 88×3, 90×1, 91×3 (ROW 92+ 待审查) |

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

### ROW 33 (coral_idx 172) — 0 P1, 3 P2 ← **新 sample**
- 63yo, left ER+/PR+/HER2- ILC, originally stage IIB → IIIA. S/p bilateral mastectomies + TC x6 + XRT. On adjuvant letrozole since 2011. NED on exam.
- P2: findings 写 "No evidence of lymphedema" 但体检 Musculoskeletal 记录 "+***** lymphedema"（ROS 写 no lymphedema vs 体检 positive，笔记本身矛盾，但 findings 应反映体检而非 ROS）
- P2: response_assessment 写 "on imaging and exam" 但笔记中唯一的影像是 "Bone density April 2013"（骨密度是骨质疏松筛查，不是癌症监测影像），NED 仅基于体检
- P2: Letter 写 "If a medication continues to be an issue, an MRI of the brain may be considered" — 原文是 "If [headaches] continues, consider MRI brain"，[REDACTED] 导致 letter 误解为药物问题而非头痛
- ✅ Type: ER+/PR+/HER2- ILC ✅。Stage: IIB→IIIA ✅（笔记 PMH 明确写 "stage IIB...stage IIIA"）
- ✅ current_meds: letrozole ✅。Goals: curative ✅。Follow-up: 6 months ✅
- ✅ Medication_plan: continue letrozole + calcium/vitamin D + NSAIDs PRN ✅（完全匹配 A/P）
- ✅ Imaging: Consider MRI brain ✅。No labs/procedures/referrals planned ✅
- ✅ Letter 其余内容准确：follow-up visit + NED + no spread + continue letrozole + calcium/D + NSAIDs + 6mo follow-up

### ROW 34 (coral_idx 173) — 0 P1, 4 P2 ← **新 sample**
- 71yo, Stage III left breast cancer, 多次复发。2011 IDC ER+/PR low/HER2-（拒绝手术扩大+化疗）。2012 快速复发→双乳切除+不完整 AC/T+自行停 anastrozole。2020 第二次局部复发→1.7cm IDC grade 3, 肌肉侵犯, 切缘阴性。PET-CT 左乳+左第6肋不确定意义的摄取。
- P2: Type_of_Cancer 写 "ER+/PR-/HER2-" 用的是 2012 病理，但 2020 复发 FNA 显示 PR+(50%)。应用最新受体状态
- P2: current_meds 写 "arimidex" 但患者已自行停药。A/P 明确写 "stopping anastrozole against medical advice" 且计划是 "resumption of hormonal therapy"（恢复内分泌治疗）。笔记本身矛盾（HPI 写 "currently on arimidex" 但 A/P 说已停）
- P2: response_assessment 写 "cancer is not responding to the current treatment regimen" — 但 A/P 明确说复发是因为患者自行停药一年后发生的，不是治疗失败。误导性结论
- P2: lab_plan 写 "No labs planned" 但 A/P 计划第一项就是 "check labs"
- ✅ Stage: Stage III ✅。Distant Metastasis: "Not sure" ✅（PET 6th rib 不确定意义，诚实回答）
- ✅ recent_changes: tamoxifen 20mg PO qD ✅。Goals: curative ✅（无确认远处转移，仍可治愈意图）
- ✅ radiotherapy_plan: chest wall RT + 转诊 ✅。Referral: Specialty 有 RT 转诊 ✅
- ✅ Letter: 局部复发解释通俗 + tamoxifen 新药 + RT 转诊 + 6mo follow-up。未提及 current_meds（避免了 arimidex 错误）

### ROW 36 (coral_idx 175) — 0 P1, 3 P2 ← **新 sample**
- 27yo, pT3N0 right breast ER+/PR+/HER2- grade III mixed ductal+mucinous ca. S/p bilateral mastectomies (8.4cm, 0/4 LN). Taxol→Abraxane（grade 3 反应）。当前 cycle 8/12。Zoladex 保卵。右臂肿胀待排 DVT。
- P2: lab_summary 只抓了 CBC（04/10），漏掉完整 CMP panel（04/03：Albumin 3.3L, ALP, ALT, AST, BUN, Cr, 电解质等全部缺失）
- P2: procedure_plan "If negative for DVT, will see Dr. [REDACTED] next week" — 把 DVT 阴性后的随访（A/P 说 PT for lymphedema）和 rad onc 转诊混在一起了
- P2: Letter 同样错误 — "If the test is negative, you will see your doctor next week" 应该是 "PT for lymphedema management"
- ✅ Type: ER+/PR+/HER2- grade III mixed ductal+mucinous ✅。Stage: pT3N0 ✅。Metastasis: No ✅
- ✅ current_meds: Abraxane + zoladex ✅。supportive_meds: Zofran, Compazine ✅
- ✅ recent_changes: Switched to Abraxane from Taxol due to infusion reaction ✅
- ✅ medication_plan 非常全面: Abraxane + zoladex + valtrex + lexapro + ativan + ambien ✅
- ✅ radiotherapy_plan: rad onc referral ✅。imaging_plan: Doppler ✅。Referral: rad onc ✅
- ✅ Letter 整体出色：Abraxane 解释通俗 + zoladex 保卵解释 + Doppler 解释 + 抗焦虑药全列出

### ROW 37 (coral_idx 176) — 0 P1, 1 P2 ← **新 sample**
- 61yo, Stage IIA left TNBC (ER-/PR-/HER2-) IDC, 2.3cm, node negative, grade 3. S/p bilateral mastectomies July 2020. Medical Oncology Consult Note (Video Visit). 推荐 dd AC → Taxol。无激素治疗和放疗指征。
- P2: second opinion 写 "no" 但这是会诊 — 患者已有肿瘤科医生（"Her oncologist at ***** has recommended adjuvant chemotherapy"），本医生 "I agree with that recommendation"，患者 "will proceed with chemotherapy at *****"（回原机构治疗）。功能上是 second opinion。
- ✅ Type: ER-/PR-/HER2- IDC ✅。Stage: IIA ✅。Metastasis: No ✅
- ✅ current_meds: empty ✅（尚未开始治疗）。response_assessment: "Not yet on treatment" ✅
- ✅ therapy_plan: dd AC → Taxol + 无激素/放疗 ✅。Advance care: Full code ✅
- ✅ Letter: IDC 解释通俗（"started in milk ducts"）+ adjuvant chemo 解释（"after surgery to prevent coming back"）+ full code 解释。准确

### ROW 40 (coral_idx 179) — 0 P1, 3 P2 ← **新 sample**
- 62yo, MS（25年）on immunosuppression, newly diagnosed Stage 2 ER+(95)/PR+(5)/HER2-(FISH 1.2) G1 IDC right breast. 2.3cm, 1 SLN micromet. S/p partial mastectomy. 骨质疏松 on Prolia。
- P2: supportive_meds 写 "ondansetron" 但这是患者的慢性 GI 症状用药（2011年起腹痛/恶心），不是癌症治疗的支持用药（她还没开始化疗）
- P2: response_assessment 写 "On treatment; response assessment not available" 但她还没开始任何癌症治疗！刚给了 letrozole 处方且有条件限制（"if no radiation is planned"）
- P2: Letter 写 "You are also taking a medication called ondansetron to help with nausea" — ondansetron 错误传播到 letter，给患者造成误解
- ✅ Type: ER 95/PR 5/HER2 2+ FISH neg G1 IDC ✅（用了手术病理的精确数值）
- ✅ genetic_testing_plan: 出色 — 解释了为什么不做分子检测（患者拒绝化疗）
- ✅ therapy_plan: letrozole adjuvant endocrine + no chemo + Prolia for bones。全面
- ✅ Imaging: DEXA ✅。Referral: PT ✅。Follow-up: 3 months ✅
- ✅ Letter: "bone density test (DEXA)" + "physical therapy" + letrozole 解释通俗。除 ondansetron 外准确

### ROW 41 (coral_idx 180) — 0 P1, 2 P2 ← **新 sample**
- 32yo, ATM mutation carrier。左乳 3cm grade 3 IDC, ER+(90%)/PR weakly+(1%)/HER2-(IHC 1+, FISH neg)。Ki-67 30%, LVI+, 1/3 SLN 微转移。MammaPrint High Risk。S/p bilateral mastectomies。Pre-chemo planning → AC-Taxol（Taxol first x12 → AC）。之后 OFS+AI + 可能 ribociclib trial。
- P2: Stage_of_Cancer 写 "Not mentioned in note" — 虽然笔记没有明确写 "Stage X"，但可从 3cm + 1 SLN micromet 推断 pT2N1mi ≈ Stage IIA
- P2: Letter 文本乱码 "a chemotherapy regimenaxol" — 应为 "AC-Taxol" 或 "a chemotherapy regimen called AC-Taxol"。生成 artifact
- ✅ Type: ER+(90%)/PR weakly+(1%)/HER2 1+ IHC ✅。Metastasis: No ✅
- ✅ current_meds: empty ✅（尚未开始治疗）。response_assessment: "Not yet on treatment" ✅
- ✅ medication_plan: 非常全面 — Taxol x12 → AC + OFS+AI + ribociclib trial ✅
- ✅ procedure_plan: port placement ✅。lab_summary: 全面（Ferritin + CBC + HCG）✅
- ✅ Letter: 手术解释通俗 + port 解释 + OFS 解释（"stops your ovaries from making estrogen"）出色

### ROW 42 (coral_idx 181) — 0 P1, 2 P2 ← **新 sample**
- 41yo, 右乳 0.9cm + 0.3cm IDC 均 grade 1，0/5 SLN，ER+/PR+(95%)/HER2-。绝经前。已完成 3 周 RT。开始 tamoxifen 5 年疗程。
- P2: Stage_of_Cancer 写 "Not mentioned in note" 但笔记有 staging form 写了 "Stage IIA (pT2...)"（虽然 staging form 数据与病理矛盾——pT2 对应 0.9cm 肿瘤不合理，受体状态也写反了）
- P2: current_meds 写 "tamoxifen" 但本次就诊刚开的处方（"I have written a prescription"，"ready to begin"）。尚未开始服用
- ✅ Type: ER+/PR+/HER2- IDC ✅。Metastasis: No ✅
- ✅ medication_plan: Start 5 year tamoxifen ✅。imaging_plan: diagnostic mammogram at next visit ✅
- ✅ follow_up: 4-6 weeks to assess tolerance ✅
- ✅ Letter: tamoxifen 解释通俗（"prevent cancer from coming back"）+ mammogram + follow-up。简洁准确

### ROW 43 (coral_idx 182) — 0 P1, 1 P2 ← **新 sample**
- 38yo, BRCA neg。第二原发 Stage I left TNBC。1.3cm grade 3 IDC, 0/2 SLN, ER-/PR-/HER2-, Ki-67>80%。S/p bilateral mastectomies。将行 taxol+carboplatin x4 adjuvant。Full code。
- P2: Letter 写 "The cancer was a type that started in the milk ducts and was not responding to hormones" — 对 TNBC 的描述有歧义，可理解为"用激素治疗但不响应"而非"癌细胞没有激素受体"。更准确的表述应该是 "does not grow in response to hormones"
- ✅ Type: ER-/PR-/HER2- IDC ✅。Stage: Stage I (second primary) ✅ — 好，标注了第二原发
- ✅ lab_summary: 全面（CMP + CBC + 甲状腺 + HCG，来自 02/17 术前检查）
- ✅ supportive_meds: granisetron + prochlorperazine + senna ✅（化疗前准备）
- ✅ medication_plan: taxol + carboplatin x4 ✅。Advance care: Full code ✅
- ✅ Letter: 手术解释通俗 + taxol/carboplatin 解释 + 止吐/通便药 + 随访/抽血。整体准确

### ROW 44 (coral_idx 183) — 0 P1, 2 P2 ← **新 sample**
- 33yo, BRCA1+, ER+/PR+/HER2- node+ left breast IDC。新辅助 dd AC x4 → Taxol x4 → bilateral mastectomies + left SLN。Residual 1cm grade 2 IDC (cellularity 15%), 1/18 nodes micromet。PR 从+转−。4mm 肺结节稳定。RT clinical trial (3v5周) + AI after RT + BSO planned。
- P2: Type 写 "ER+/PR+/HER2-"（初始诊断）但 A/P 明确写残余肿瘤 "ER+/PR-/HER2-" — PR 在新辅助化疗后从阳性转为阴性。应用当前受体状态
- P2: Letter 使用医学术语 "pathologic complete response (pCR)" 和 "neoadjuvant therapy" — 前面已用通俗语言解释了（"treatments you had before your surgery"），后面再用专业术语是多余的
- ✅ response_assessment: 非常出色！准确识别为 "did not achieve pCR to neoadjuvant therapy"。这是正确的临床解读
- ✅ therapy_plan: 非常全面 — RT trial + AI after RT + BSO + Zoladex backup + ribociclib trial 可能性
- ✅ imaging_plan: CT Chest in 1 year（肺结节随访）✅
- ✅ Referral: Nutrition (11/30) + rad onc + PT。全部捕获
- ✅ Letter 整体出色：残余病灶解释通俗 + 肺结节 + AI + BSO/Zoladex 解释 + RT trial + 所有随访全列出

### ROW 46 (coral_idx 185) — 0 P1, 4 P2 ← **新 sample**
- 48yo post-menopausal, R breast IDC ER+(98%)/PR+(25%→0%)/HER2-。新辅助 taxol→[redacted]。S/p right lumpectomy + bil mastopexy + SLN。残余 3.5cm (cellularity 10-20%), ypT2, 2/2 SLN macrometastases + extranodal extension >2mm。切缘阳性需 re-excision。Sarcoidosis (纵隔 LAD 证实为肉芽肿)。肾动脉瘤。
- P2: response_assessment 结论写 "No evidence of response to treatment is mentioned" — 但医生明确写 "she had a good response to chemotherapy based on where she started"，病理也写 "probable or definite response to presurgical therapy"。结论与原文相反
- P2: lab_plan 写 "No labs planned" 但 A/P 贫血部分写 "Repeat in 3-4 months"（铁 panel 复查）
- P2: Letter 写 "surgery to remove your breasts" 暗示全切，但实际是右乳保乳手术 + 双侧 mastopexy（提升，不是切除）
- P2: Letter 写 "vitamin D level is a bit low" 但 Vitamin D 36 ng/mL 在正常范围内（20-50）
- ✅ Type: ER+/PR-/HER2- ✅ — 正确用了术后病理（PR 从25%→0%，和 ROW 44 同样是新辅助后受体转换，这次用对了）
- ✅ Stage: pT2/pN1 with extranodal extension ≈ Stage II ✅。Distant Met: No ✅（内乳淋巴结是 regional N3，纵隔是 sarcoidosis）
- ✅ recent_changes: letrozole started + abemaciclib discussed ✅。therapy_plan + radiotherapy_plan 准确
- ✅ procedure_plan: re-excision + MRA ✅。imaging_plan: DEXA + MRA ✅。lab_summary 非常详细
- ✅ Letter: letrozole + abemaciclib + re-excision + RT + DEXA + MRA + PT 全列出。内容全面

### ROW 49 (coral_idx 188) — 0 P1, 3 P2 ← **新 sample**
- 50yo, 新诊断左乳 IDC ER+(100%)/PR+(100%)/HER2-, 腋窝淋巴结活检阳性。Stage II。Oncotype low risk (score 11)。计划 L mastectomy 01/06/17。Tamoxifen 待评估血栓风险。多个医生会诊（新辅助 vs 先手术 → 选择先手术）。代理决策人已命名。
- P2: response_assessment 写 "On treatment" — 但患者尚未开始任何癌症治疗（新诊断，pre-treatment 会诊）
- P2: supportive_meds 列了 tylenol/alprazolam/HCTZ — 这些是普通药物（止痛/焦虑/高血压），不是癌症治疗支持用药
- P2: Letter 写 "not to a medication called HER2" — HER2 是蛋白质/受体，不是药物。事实性错误
- ✅ Type: ER+/PR+/HER2- IDC ✅。Stage: II ✅。findings: 全面（病理+影像+体检）
- ✅ procedure_plan: L mastectomy 01/06/17 ✅。medication_plan: tamoxifen + 血栓风险评估 ✅
- ✅ Advance care: surrogate decision maker (spouse) ✅
- ✅ Letter: 整体通俗 — Stage II 解释 + mastectomy + tamoxifen + 血栓风险 + XRT 可能性。除 HER2 误称外准确

### ROW 50 (coral_idx 189) — 0 P1, 2 P2 ← v28 已审查
- 58yo, de novo Stage IV (T2N1M1) metastatic IDC HR+/HER2-, 转移至肺/淋巴结/肝/骨（2013年）。AC x4 → tamoxifen+lupron → letrozole+lupron+ibrance（2015年起）。Dec 2021 restaging 良好控制。新增乳房进展（DCIS+IDC），考虑 mastectomy。PMS2 mutation。Video consult/second opinion。Full code。
- P2: medication_plan 写 "Second line lupron, letrozole started October 2014 and ibrance added January 2015" — 描述的是治疗历史，不是当前或未来计划。当前计划应是继续 ibrance+letrozole+lupron + 考虑 mastectomy
- P2: Referral Genetics 写 "None" 但 A/P 明确写 "Referral to genetics for pathogenic PMS 2 mutation"
- ✅ second opinion: yes ✅。Type: HR+/HER2- IDC with DCIS ✅。Stage: Stage IV ✅
- ✅ response_assessment: "disease under good control" ✅ — 准确简洁
- ✅ current_meds: ibrance + xgeva + letrozole ✅（lupron 在 med list 标记 not taking）
- ✅ goals: palliative ✅。genetic_testing_plan: PMS2 mutation referral ✅

### ROW 52 (coral_idx 191) — 0 P1, 3 P2 ← **新 sample**
- 35yo 绝经前, ER+(>95%)/PR+(>95%)/HER2-(IHC 1+/2+, FISH 1.1/1.4), Ki-67 <10%/~15%。1.7cm grade II IDC, 1 SLN micromet (0.18cm), minimal extranodal extension。Oncotype low risk (11), MammaPrint low risk。S/p left lumpectomy + SLN + bilateral mastopexy。
- P2: supportive_meds 写 "ondansetron (ZOFRAN)" 但患者尚未开始癌症治疗，Zofran 可能是其他原因的 PRN 处方
- P2: Referral 全部写 "None" 但 A/P 明确写 "Referral for fertility preservation asap" 和 "referred to reproductive health"
- P2: Letter 写 "a test called a medication" — 由于 Oncotype 测试名被 [REDACTED]，letter 生成出乱码
- ✅ Type: ER+/PR+/HER2- IDC ✅。Goals: curative ✅。response: "Not yet on treatment" ✅
- ✅ imaging_plan: CT CAP + bone scan for staging ✅。genetic_testing_plan: Oncotype ✅
- ✅ medication_plan: Zoladex + [redacted] after fertility preservation ✅

### ROW 54 (coral_idx 193) — 0 P1, 0 P2 ✅ ← **新 sample**
- 39yo BRCA2+, oligometastatic ER+/PR+/HER2- IDC left breast + T6 骨转移。S/p neoadjuvant AC/T → RT T6 → bilateral mastectomies + left ALND。Residual 8.2cm (cellularity 10%), ER+50%/HER2-, Ki-67 1%。1/24 SLN micromet。On leuprolide + letrozole + zoledronic acid。
- ✅ Stage: Stage IV (metastatic) ✅。Distant Met: Yes, to bone (T6 + 7th rib) ✅
- ✅ Goals: palliative ✅（A/P 明确写 "metastatic breast cancer is not curable, but treatable"）
- ✅ current_meds: leuprolide + letrozole + zoledronic acid ✅
- ✅ medication_plan: 非常全面 — continue leuprolide/letrozole + palbociclib after RT + zoledronic acid q3mo + calcium + vitamin D
- ✅ radiotherapy_plan: post-mastectomy radiation ✅。imaging_plan: PET/CT 3-4 months + DEXA ✅
- ✅ Letter: 准确 — incision healing + continue meds + palbociclib after RT + acupuncture + PET/CT + DEXA + 4 weeks follow-up

### ROW 53 (coral_idx 192) — 0 P1, 1 P2 ← **新 sample**
- 59yo, Stage II/III left breast IDC with neuroendocrine differentiation。4.5cm grade 3, ER+(>95%)/PR+(30%)/HER2+(IHC heterogeneous 2+/3+, FISH 4.9X), Ki-67 25%。LVI extensively+。DCIS 4.5cm。SLN 2/? positive (6mm met)。60%+ recurrence risk。S/p left lumpectomy + SLN。
- P2: Letter 写 "targeted therapy drugs chemotherapy" — 应为 "TCHP chemotherapy"。生成 artifact
- ✅ Type: ER+/PR+/HER2+ IDC with neuroendocrine differentiation ✅ — 非常出色
- ✅ medication_plan: 极其全面 — AC/THP (AC x4 q2w → Taxol x12 + HP x1yr) + alternative TCHP + Arimidex 1mg x10yr + neratinib yr2 + bone strengthening
- ✅ radiotherapy_plan: adjuvant XRT after chemo ✅。Referral: Genetics ✅ + Specialty (rad onc) ✅
- ✅ Letter: neuroendocrine differentiation 解释通俗（"features of cells that normally release hormones"）✅
- ✅ Letter: 尊重患者决策时间（"thinking about options over next couple weeks"）✅

### ROW 57 (coral_idx 196) — 0 P1, 2 P2 ← v28 已审查
- 59yo, locally advanced left breast CA。初始外院诊断 HER2+ → TCH+P 新辅助。术后标本 HER2-/TNBC。外院病理复查也 HER2-。S/p left lumpectomy, residual 3.7cm, 0/6 nodes。Post-op AC x4。计划 XRT。Second opinion 关于 HER2 争议。
- P2: response_assessment 写 "not responding to treatment" — 但笔记明确写 "Pt noted to have tumor volume reduction on exam"（治疗期间肿瘤体积缩小）。应为 partial response with significant residual disease
- P2: recent_changes 写 "Dose reduction 25% after C1" — 这是已完成的新辅助化疗的历史事件，不是当前就诊的近期变化
- ✅ second opinion: yes ✅。Type: ER-/PR-/HER2- TNBC ✅。Stage: Locally advanced ✅
- ✅ therapy_plan: XRT + 如果 HER2+ 则恢复 trastuzumab ✅。genetic_testing_plan: 遗传咨询 ✅
- ✅ Referral: Genetics ✅ + Specialty (XRT) ✅

### ROW 59 (coral_idx 198) — 0 P1, 1 P2 ← v28 已审查
- 52yo, Stage I right ER+/PR+/HER2- IDC。S/p lumpectomy + SLN (0/5), 1.5cm grade 3。TC→Abraxane/Cytoxan（docetaxel 过敏）→ XRT → tamoxifen（停，症状）→ letrozole（症状更差）→ 建议 exemestane。NED。MammaPrint High Risk。Post-menopausal (FSH 32.5, E2 5)。
- P2: current_meds 写 "exemestane" 但她尚未开始服用（A/P 说 "I again recommended discontinuing Letrozole and waiting 2-3 weeks before starting Exemestane"，且她之前被建议过但 "has not tried it yet"）
- ✅ response_assessment: "No evidence of recurrence" ✅ — 直接来自 A/P
- ✅ medication_plan: 全面 — stop letrozole + wait + exemestane + Pristiq + consider duloxetine via psychiatry
- ✅ imaging_plan: mammogram July + alternating MRI every 6mo ✅
- ✅ Referral: Specialty psychiatry ✅（duloxetine 转换需精神科评估药物相互作用）
- ✅ Letter 非常出色：NED + exemestane switch + Pristiq→duloxetine + mammogram/MRI schedule + psychiatry。全部通俗准确

### ROW 61 (coral_idx 200) — 0 P1, 1 P2 ← **新 sample**
- 43yo 绝经前, 新诊断左乳 IDC ER+(100%)/PR+(100%)/HER2-(1+), grade 2。筛查发现。Biopsy: IDC ≥11mm。MRI 0900 位置可疑→复检活检阴性。CT 无转移。Invitae 阴性。Televisit 建立诊疗关系。手术 04/12/21: lumpectomy + IORT + reconstruction。
- P2: genetic_testing_plan 写 "None planned" 但 A/P 明确写 "will likely need Oncotype Dx after surgery"（基因组学检测被归入 therapy_plan 而非 genetic_testing_plan）
- ✅ Type: ER+/PR+/HER2-(1+) IDC ✅。in-person: Televisit ✅
- ✅ radiotherapy_plan: IORT + no post-op RT ✅ — 准确区分术中 RT 和术后 RT
- ✅ procedure_plan: lumpectomy + IORT + reconstruction 04/12/21 ✅
- ✅ medication_plan: Tamoxifen vs OFS + AI ✅。Goals: curative ✅
- ✅ Letter 非常出色：ER/PR/HER2 解释通俗（"uses hormones to grow" / "protein that makes cancer grow faster"）+ MRI 可疑区域阴性 + IORT 解释 + Oncotype 后续检查 + 随访安排

### ROW 63 (coral_idx 202) — 0 P1, 1 P2 ← v28 已审查
- 49yo, locally advanced ER+/PR-/HER2- left IDC。新辅助 dd AC x4 → paclitaxel（停）→ Abraxane。MRI 显示 dramatic response 但术后残余 3.8cm（path 失望）。3/4 SLN+（15mm, 1-2mm, ITC）。切缘+→ re-excision。On letrozole + XRT。Second opinion (televisit)。
- P2: response_assessment 写 "currently responding to treatment" 但 A/P 明确写 "Her overall response to chemo was disappointing since it appeared that she had responded much better on imaging"。影像和病理矛盾，extraction 选了影像结论而非 A/P 的临床评估
- ✅ Stage: IIIA (S/P neoadjuvant) ✅。second opinion: yes ✅。Type: ER+/PR-/HER2- ✅
- ✅ medication_plan: letrozole + E2/FSH monitoring + OS/oophorectomy + DEXA + abemaciclib。非常全面
- ✅ lab_plan: E2/FSH q1-2 months ✅。imaging_plan: DEXA ✅
- ✅ Letter: abemaciclib + OS/oophorectomy + DEXA + Mychart。通俗

### ROW 64 (coral_idx 203) — 0 P1, 3 P2 ← v28 已审查
- 28yo, 新诊断左乳 IDC 10.3cm, HR+/HER2-, 腋窝+。骨扫描胸骨（manubrium）可疑转移灶→活检计划中。Oligometastatic disease 讨论。On dd AC。Second opinion (televisit)。Full code。
- P2: Stage 写 "Originally Stage IV-IV, now Stage IV" — 乱码。笔记写 "Stage III-IV"（不确定，待胸骨活检）。POST hook artifact
- P2: Metastasis 写 "Yes (to the sternum)" 作为确定结论，但胸骨病灶是 "probably metastatic"，活检尚未完成。应为 "Probable" 或 "Pending biopsy"
- P2: current_meds 为空，但患者正在接受 dd AC 化疗（"She was started on dd AC and is tolerating okay"）
- ✅ second opinion: yes ✅。supportive_meds: 化疗支持药物全面（dex + colace + olanzapine + zofran + compazine）
- ✅ therapy_plan: 全面 — chemo → surgery → radiation → sternum treatment → hormonal blockade
- ✅ procedure_plan: sternal biopsy ✅。Advance care: Full code ✅

### ROW 65 (coral_idx 204) — 0 P1, 1 P2 ← **新 sample**
- 48yo, right breast IDC ER weak+(2%)/PR low+(7%)/HER2-(IHC 2+, FISH neg), Ki-67 36%。DCIS high-grade, LVI+。腋窝 LN 0.2mm micromet。MRI 2.6cm mass。PET 无远处转移。Pre-neoadjuvant consultation (televisit)。计划 AC/T，正在筛选 ISPY-2 trial。
- P2: response_assessment 写 "On treatment" 但患者尚未开始化疗（Plan 写 "RTC in 1-2 weeks in person to start chemo"）。应为 "Not yet on treatment"
- ✅ Type: ER weak+(2%)/PR low+(7%)/HER2- IDC ✅ — 精确数值
- ✅ medication_plan: 非常详细 — AC/T + ISPY trial arms（pembrolizumab, olaparib, durvalumab 等）+ 5-10yr endocrine therapy
- ✅ therapy_plan: 全面 — neoadjuvant AC/T + ISPY + post-op RT + endocrine ✅
- ✅ procedure_plan: port placement + research core biopsy + lumpectomy ✅
- ✅ imaging_plan: TTE ✅。genetic_testing_plan: F/u genetic results ✅
- ✅ Letter: neoadjuvant 解释通俗 + clinical trial + port 解释 + post-op RT + 5-10yr endocrine。准确

### ROW 66 (coral_idx 205) — 0 P1, 2 P2 ← **新 sample**
- 53yo, right breast metaplastic high-grade carcinoma with squamous differentiation。ER 5-10%/PR 0%/HER2 0%。BRCA negative。6cm+1.4cm masses，皮肤累及。Axillary LN bx negative。CT 无远处转移。Nonspecific 肺结节 ≤4mm。Second opinion (Investigational Therapeutics)。Affirms neoadjuvant plan + pembrolizumab for TNBC。Considering bilateral mastectomy。
- P2: imaging_plan 写 "Mammogram" 但 A/P 没有计划任何新的 mammogram。可能是 fabrication
- P2: Letter 写 "a type that makes mucus and has some cells that look like skin cells" — 癌症是 SQUAMOUS differentiation（皮肤样），不是 mucinous（产生粘液）。"Makes mucus" 是事实性错误
- ✅ second opinion: yes ✅。Type: ER 5-10%/PR 0%/HER2 0% metaplastic carcinoma with squamous differentiation ✅
- ✅ therapy_plan: neoadjuvant + bilateral mastectomy + adjuvant radiation + pembrolizumab discussed ✅
- ✅ response_assessment: "Not yet on treatment" ✅。Goals: curative ✅
- ✅ genetic_testing_plan: germline testing pending ✅。Referral: Genetics (awaiting results) ✅

### ROW 68 (coral_idx 207) — 0 P1, 1 P2 ← v28 已审查
- 63yo postmenopausal, multifocal ER+/[redacted] right breast cancer + [redacted] mutation。多发病灶（3.9cm+1.5cm+2.2cm）+ 腋窝+内乳 LN 受累。TCHP x6 后 MRI 完全缓解。讨论 bilateral mastectomy vs lumpectomy。
- P2: Letter 写 sons should be tested "for a specific health risk related to a medication" — 实际是基因遗传风险（Fanconi anemia, pancreatic cancer），不是药物风险
- ✅ response_assessment: "good clinical response after TCHP, MRI no lesions" ✅ — 准确
- ✅ Type: multifocal ER+ ✅。genetic_testing_plan: sons testing ✅。Goals: curative ✅
- ✅ procedure_plan: bilateral mastectomy ✅。radiotherapy_plan: 条件性 RT ✅

### ROW 70 (coral_idx 209) — 0 P1, 1 P2 ← v28 已审查
- 61yo, BRCA1+, bilateral breast cancer。Left: ILC ER+/PR+/HER2-, 4.4cm residual (5-10% cellularity), 2/5 LN+。Right: IDC ER+/PR-/HER2-, 1cm residual, 0/2 LN。S/p bilateral mastectomies + BSO。Resuming letrozole。Prolia after dental clearance。肺结节随访 CT。
- P2: procedure_plan 写 "No procedures planned" 但 A/P 明确写 "She is going to have expanders placed prior to radiation"
- ✅ Type: 出色 — 正确识别双侧不同组织学和受体状态（left ILC ER+/PR+, right IDC ER+/PR-）
- ✅ Distant Met: "Not sure" ✅（sub-4mm 肺结节无法确认）。Advance care: Full code ✅
- ✅ imaging_plan: CT June for lung nodules ✅。Referral: Radiation consult ✅
- ✅ lab_summary: 全面（CMP + CBC from 06/05）✅

### ROW 72 (coral_idx 211) — 0 P1, 2 P2 ← **新 sample**
- 72yo postmenopausal, ER+(99%)/PR-(<1%)/HER2- IDC with focal neuroendocrine differentiation。1.2cm grade 2, pT1cN0(sn)。S/p left mastectomy。Televisit post-op consult。Letrozole 刚开处方。Oncotype ordered。骨质疏松 on Reclast。
- P2: response_assessment 写 "On treatment" — 术后首次会诊，尚未开始任何治疗（letrozole 刚开处方）
- P2: current_meds 写 "letrozole" 但本次就诊刚开的处方（"Instructed patient to begin letrozole, prescription ordered"）
- ✅ Type: ER+/PR-/HER2- IDC with neuroendocrine differentiation ✅。Stage: pT1cN0(sn) ✅
- ✅ genetic_testing_plan: Oncotype ordered ✅。therapy_plan: letrozole + Oncotype ✅
- ✅ findings: 详细（core bx + surgical path + exam）✅

### ROW 73 (coral_idx 212) — 0 P1, 2 P2 ← v28 已审查
- 63yo, Stage III left ER/PR+/HER2- breast cancer。S/p bilateral mastectomies + ALND + chemo + CW XRT + arimidex (since Aug 2017)。Follow-up 新结节→US+mammogram 确认全部为脂肪坏死。
- P2: Type 写 "HER2: not tested" 但笔记明确写 "[redacted] negative"（HER2 已检测为阴性）
- P2: response_assessment 写 "fat necrosis, indicating stable disease" — 脂肪坏死是良性术后改变，不是"稳定疾病"。应为 NED（无疾病证据）
- ✅ Stage: III ✅。current_meds: arimidex ✅。Goals: curative ✅
- ✅ findings: 详细描述三处结节位置和大小 + 影像确认 fat necrosis ✅
- ✅ lab_plan: "check labs" ✅。follow_up: 4 months ✅

### ROW 78 (coral_idx 217) — 0 P1, 0 P2 ✅ ← v28 已审查
- 79yo, de novo metastatic TNBC (PD-L1 neg) to liver + periportal LNs。S/p capecitabine → OPERA trial → gemcitabine（停因心包积液）。Off systemic therapy since 03/15/19。Now progressing（肝 + 淋巴结 + 新肺结节）。探索 clinical trials。Rad onc consult for liver/nodal XRT。
- ✅ 全部字段正确。response_assessment: "cancer is currently progressing" + 详细影像证据 ✅
- ✅ Type: TNBC ✅。Stage: IV ✅。Goals: palliative ✅。current_meds: empty ✅（off therapy）
- ✅ therapy_plan: trial options + rad onc ✅。lab_summary: 全面（CBC + CMP）✅

### ROW 80 (coral_idx 219) — 0 P1, 1 P2 ← v28 已审查
- 53yo, local recurrence IDC 0.8cm grade 3 in dermis post mastectomy for DCIS (2013)。ER 95%/PR 70%/HER2-, Oncotype 24。Pre-chemo visit (televisit)。Plan: TC x4 + RT 6 weeks (incl axilla/SC)。
- P2: response_assessment 写 "On treatment" 但本次是化疗前就诊（"Start TC x 4 on 04/11/19"，尚未开始）
- ✅ therapy_plan: TC x4 + RT 6w with boost + axilla/SC fields ✅。supportive_meds: dex + zofran + compazine ✅
- ✅ lab_summary: 全面（CMP + CBC + Hep B + Estradiol）✅。Goals: curative ✅

### ROW 82 (coral_idx 221) — 0 P1, 2 P2 ← v28 已审查
- 52yo, Stage II (IB per staging form) right breast mixed ductal+lobular ca。4.3cm, LN+ (micro), ER+/PR+/HER2-。S/p right lumpectomy + SLN。Low risk Oncotype → no chemo。Plan: RT → AI + bone protection。DEXA ordered。
- P2: response_assessment 写 "On treatment" — 术后新患者会诊，尚未开始癌症治疗
- P2: medication_plan 列了非癌症药物（APAP, ibuprofen, HCTZ, lisinopril, metformin）而非计划中的 AI + bone medication
- ✅ Type: ER+/PR+/HER2- mixed ductal and lobular ✅。Goals: curative ✅
- ✅ therapy_plan: no chemo + RT + AI + bone med ✅。imaging_plan: DEXA ✅。Advance care: Full code ✅

### ROW 83 (coral_idx 222) — 0 P1, 0 P2 ✅ ← **新 sample**
- 77yo, right breast ILC with extensive R axillary LN involvement。ER+/PR+。On neoadjuvant letrozole since Dec 2019。PET/CT shows significant SUV response in axillary nodes（SUV 15.1→1.9）。Discussing breast conservation。Continue letrozole until surgery。Televisit。
- ✅ response_assessment: "cancer responding, significant response in axillary nodes, substantial improvement with therapy" ✅ — 出色
- ✅ current_meds: letrozole ✅。Goals: curative ✅。medication_plan: continue letrozole ✅
- ✅ procedure_plan: breast surgery ✅。therapy_plan: continue neoadjuvant until surgery ✅

### ROW 84 (coral_idx 223) — 0 P1, 1 P2 ← **新 sample**
- 60yo, CHEK2 mutation + MS + metastatic ER+/PR-/HER2- breast cancer。Bone + liver + meninges 转移。S/p letrozole/palbociclib（PD）→ capecitabine 1500mg BID。脑 MRI 示 dural/IAC 增强（可能 pachymeningeal/leptomeningeal）。
- P2: response_assessment 写 "stable disease" 但肝脏转移灶 increased in size and number + 脑 IAC 累及增加。应为 mixed/progressive
- ✅ Type: ER+/PR-/HER2- ✅。Stage: IV ✅。Goals: palliative ✅
- ✅ Metastasis: Yes (bone, soft tissue, liver, possibly meninges) ✅ — 全面
- ✅ therapy_plan: 极其详细 — CT CAP + LP + MRI spine + rad onc + steroids + xeloda + fulvestrant+[PI3Ki] at PD
- ✅ current_meds: capecitabine + zoledronic acid ✅。lab_summary: 简明（WBC, Hg, Plt, Cr, Tbili, ALT）✅

### ROW 85 (coral_idx 224) — 0 P1, 1 P2 ← v28 已审查
- 61yo, metastatic ER+/PR-/HER2- ILC。Bone+muscle+brain+liver。S/p neoadjuvant AI+palbociclib → bilateral mastectomies → adjuvant AC/T → XRT → met disease on exemestane → fulvestrant/palbociclib (PD)。Brain mets s/p GK。Foundation One: FGFR1 amp。Evaluating phase 1 trial olaparib+[drug]。Steroids for facial numbness/headache。
- P2: lab_summary 写 "No labs in note" 但笔记有 CA 15-3（360→259→45.3 trending down）和 CEA（25.6→14.1→3.8 trending down）— 临床重要的肿瘤标志物遗漏
- ✅ Type: ER+/PR-/HER2- ILC ✅。Stage: IIIA→IV ✅。Goals: palliative ✅
- ✅ response_assessment: "progressed on fulvestrant/palbociclib with new liver mets" ✅ — 准确
- ✅ therapy_plan: phase 1 trial + rad onc + radiation washout ✅。Referral: Rad Onc ✅

### ROW 86 (coral_idx 225) — 0 P1, 1 P2 ← v28 已审查
- 53yo, metastatic breast cancer。Originally R breast mixed IDC/[redacted] Grade III, HER2+(FISH 4.37)。S/p neoadjuvant TCHP x6 → bilateral MRM (3 masses, 15 LN+) → adjuvant XRT → adjuvant [trastuzumab]。2018年转移复发→骨（广泛）+肝+脑（dural）。转移灶活检 ER 95%/PR 2%/HER2 1+ FISH negative — **HER2 从+转为-**。CHEK2 mutation。On letrozole+ribociclib → PD bone (04/2020)。Recommend fulvestrant +/- everolimus。Palliative XRT planned。ECOG 1。
- P2: Type 写 "ER+/PR+/HER2+ mixed IDC" 用的是原始肿瘤受体状态（FISH 4.37），但转移灶活检（current disease）明确 HER2-(IHC 1+, FISH negative)。A/P 也写 "FISH negative also"。应用当前转移灶受体状态
- ✅ response_assessment: "PET/CT showed increase in metabolic activity of osseous metastatic disease... bone progression while liver stable" ✅ — 准确反映 mixed response
- ✅ current_meds: letrozole + ribociclib + denosumab ✅（PD 后即将更换但本次仍在用）
- ✅ medication_plan: fulvestrant +/- everolimus + continue denosumab ✅
- ✅ radiotherapy_plan: palliative XRT ✅。Goals: palliative ✅。follow_up: telehealth 6wks ✅
- ✅ Letter: bone PD + liver stable 通俗准确 + fulvestrant 解释 + denosumab bone health + palliative XRT for pain + 6wks telehealth。Letter 未提 everolimus 因为是 optional（+/-），合理简化

### ROW 87 (coral_idx 226) — 0 P1, 0 P2 ✅ ← v28 已审查
- 79yo, R breast IDC grade 2, 2.2cm multifocal + separate 0.6cm well-diff adenocarcinoma。4/19 LN+ with ECE (0.5cm)。ER+/PR+/HER2-。Parkinson's disease（R sided, moderate）。Significant family hx（daughter breast+colorectal CA @40, maternal grandmother ovarian CA）。S/p excisional biopsy + ALND。Second opinion。
- ✅ second opinion: yes ✅（笔记明确写 "second opinion concerning treatment"）
- ✅ Type: ER+/PR+/HER2- IDC ✅（A/P 明确 "ER and PR positive, HER-2/neu was negative"）
- ✅ response_assessment: "Not yet on treatment" ✅（pre-treatment 会诊）
- ✅ Goals: curative ✅。current_meds: empty ✅（未开始治疗）
- ✅ medication_plan: hormonal therapy alone ✅（笔记未指定具体药物，extraction 准确反映）
- ✅ therapy_plan: 准确捕获 "hormonal therapy alone"，不推荐 chemo（age+Parkinson's, 仅 3-4% 额外获益）
- ✅ radiotherapy_plan: radiation discussed for local control ✅
- ✅ findings: 全面 — 2.2cm multifocal + 4/19 LN+ + ECE + grade 2 + Parkinson's tremor ✅
- ✅ Letter 逐句准确：second opinion + IDC 解释 + 2.2cm + 4/19 LN + hormone sensitive + Parkinson's + hormonal therapy + return to oncologist + radiation discussed。通俗易懂

### ROW 88 (coral_idx 227) — 0 P1, 3 P2 ← v28 已审查
- 36yo, Stage III→IV left IDC。Multifocal (4.0+2.6cm), Grade III (3+3+3), 23/30 LN+ with ECE, LVI+, PNI+。Original: ER weak+(1+/10)/PR+(2+/1 Favorable)/HER2-(0)。PD on neoadjuvant AC→Taxol→Taxol+carbo。S/p bilateral mastectomies + ALND → adjuvant gemzar/carbo x4 → XRT → 2 brain mets (resection of 1 + SRS to both)。Brain met: ER-/PR-/GATA3+, **HER2 NOT TESTED**。Lung+LN mets。On Xeloda。COVID-19。Genetic testing negative。Full code。Televisit。
- P2: Type 写 "PR 2-" 但原始活检明确 "Progesterone Receptor Positive 2+ 1 Favorable"（PR 阳性）。且 metastatic biopsy HER2 写 "-" 但 HER2 在脑转移灶上**未检测**（HPI: "her 2 was not done", A/P 专门要求补做 HER2 检测）
- P2: radiotherapy_plan 描述过去治疗（"SRS to both brain metastases"）而非未来计划。A/P 无新 RT 计划
- P2: Letter 被截断 — 最后一句 "We want to" 断在半句中间，letter 生成不完整
- ✅ Stage: Stage III → IV ✅。Metastasis: brain + lungs + LN ✅。Goals: palliative ✅
- ✅ current_meds: capecitabine ✅。genetic_testing_plan: HER2 retesting on brain met + residual disease ✅
- ✅ medication_plan: 全面 — Xeloda + HER2 决定后续治疗 + restaging 3mo + immunotherapy trials if PD
- ✅ findings: 详细（4.0+2.6cm, 23 LN+, grade III, brain mets, lung/LN mets）✅
- ✅ Letter（除截断外）: Stage III→IV 解释通俗 + HER2 蛋白检测解释 + Xeloda + clinical trials + full code 解释。attribution 完整

### ROW 90 (coral_idx 229) — 0 P1, 1 P2 ← v28 已审查
- 51yo, R breast adenocarcinoma Stage II/III。ISPY trial（randomized to [redacted] arm）。S/p neoadjuvant Taxol → R lumpectomy（residual 2.2cm IDC, 60% cellularity）。On adjuvant AC cycle 3。Side effects: fatigue, N/V (D5), GERD, hypothyroidism (TSH 6.01), port reinserted, neuropathy (resolving)。BLM gene carrier（adopted, family hx unknown）。ECOG 0。Televisit。
- P2: Patient type 写 "New patient" 但患者正在 mid-treatment（cycle 3 AC + ISPY trial cycle 12）。Letter 正确写了 "follow-up visit"，keypoints 与 letter 矛盾
- ✅ current_meds: AC ✅。Goals: curative ✅。Stage: II/III ✅
- ✅ medication_plan: 全面 — AC cycle 4 + 1wk dose delay + GCSF reduction 50% + granisetron + PO dex PRN + olanzapine 10mg PRN + gabapentin
- ✅ lab_summary: 全面（TSH 6.01H + Free T4 12 + CMP 含 ALP 125H + CBC 含 WBC 11.8H + toxic granulation + HCG）
- ✅ findings: residual 2.2cm IDC 60% cellularity + axilla swelling + port tenderness + hypothyroidism
- ✅ Letter 逐句审查：adenocarcinoma explained + stage II/III + hypothyroidism/TSH + AC treatment + GCSF/granisetron/dex/olanzapine + gabapentin + cycle 4 with delay + follow-up after XRT。通俗准确，无截断

### ROW 91 (coral_idx 230) — 0 P1, 3 P2 ← v28 已审查
- 53yo, Stage IV breast cancer（originally Stage I 2003, PD to bone 2005）。ER+/PR+/HER2-。Very long treatment history（tamoxifen → letrozole/zoladex → BSO → fulvestrant 250 → phase trial → fulvestrant 500 → everolimus/exemestane since April 2012）。On everolimus+exemestane+denosumab monthly。RLE edema improving on lasix。+1cm R iliac LN（unclear significance）。PET/CT next week。Labs monthly。ECOG 0。Family hx: mother+paternal aunt breast CA。
- P2: Type 写 "HER2: not tested" 但 problem list 明确写 "ER+PR+ *****-"（HER2 已检测为阴性）。和 ROW 73, 100 同样的 redacted HER2 误读问题
- P2: response_assessment 用了 2011 年 11 月的影像数据（MRI pelvis 11/08/11 + PET/CT 11/15/11），这是在当前治疗方案（everolimus+exemestane 04/2012 开始）之前的影像。不应用旧影像来评估当前治疗反应。A/P 写 "unclear significance for iliac LN small change" + 正在安排新 PET/CT
- P2: Letter 写 "the current treatment is not working as well as we hoped" — 和 response_assessment 一样的时间归因错误。用了旧影像结论误导患者对当前治疗的认知
- ✅ current_meds: everolimus + exemestane + denosumab ✅
- ✅ imaging_plan: PET/CT next week ✅。lab_plan: Labs monthly ✅。follow_up: 1 month ✅
- ✅ Goals: palliative ✅。Stage: I → IV ✅
- ✅ medication_plan: lasix + KCL + denosumab + topical antifungal ✅
- ✅ Lab_summary: 全面（LFTs + electrolytes + CBC from 10/16/2012）✅




