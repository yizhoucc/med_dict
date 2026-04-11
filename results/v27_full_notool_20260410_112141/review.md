# V27 notool Full Run Review

> Run: v27_full_notool_20260410_112141
> Dataset: 28 samples（v26 审查中发现问题的 ROW 子集）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查完成 — 28/28 ROW 全部审查完毕**
> 参照: `results/v26_full_notool_20260408_080004/review.md`（v26 notool 审查）和 `results/v26_full_tool_20260408_140605/review.md`（v26 tool 审查）
> Results 文件: `results/v27_full_notool_20260410_112141/results.txt`

### 恢复审查指南
- 待审查 ROW: **1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 20, 22, 24, 25, 33, 38, 39, 52, 57, 74, 75, 83, 88, 95**（共 28 个）
- 这些 ROW 在 v26 中有 P1 或 P2 问题，v27 是修复版本。审查重点：v26 的问题是否被修复？是否引入新问题？
- 每个 ROW 必须：完整读 note_text + keypoints + letter → 逐字段核对 → 写入详细临床描述 → 与 v26 对比
- results.txt 中每个 ROW 的位置可用 `grep "^RESULTS FOR ROW N$"` 查找行号

### v26 已知问题回顾（v27 需要验证修复的）
| ROW | v26 notool 问题 | v26 tool 问题 |
|-----|----------------|--------------|
| 1 | P1: lab_plan 混入 imaging | P2×2: imaging_plan 遗漏 bone scan, therapy_plan 重复 |
| 2 | P2×3 | P2×1: Home health 疑问 |
| 3 | P2: genetic_testing 放错字段 | ✅ |
| 4-5 | ✅ | ✅ |
| 6 | P2×3: Patient type, letter anxious, Referral-Genetics historical | P2×2 |
| 7 | P2×3: procedure/lab LVEF, letter "medication level" | P2×2 |
| 8 | P1: response "Not yet on treatment" | P1+P2×4 |
| 10 | P1: response 遗漏 pathology | P1 |
| 11 | P1: response 引用旧 PET, P2: imaging Echo | P1+P2 |
| 12 | P1: Advance care DNR/DNI 遗漏 | P2: lung mets resolved |
| 13 | P2×2: response "On treatment", findings laterality | P2: response |
| 14 | P2×2: current_meds 空 | P2×3 |
| 15 | P2×2: genetic_testing, procedure_plan mixed | P2: genetic_testing |
| 20 | P2: procedure_plan mixed | P2 |
| 22 | P2: genetic_testing has medication plan | P2: radiotherapy past |
| 24 | P2×2: procedure genetic, Metastasis "Not sure" | P2×2 |
| 25 | P2: response 引用旧 PET | P2 |
| 33 | P2: letter stage confusion | P2 |
| 38 | P2: response "not responding" but not on treatment | P2: letter anxious |
| 39 | P2: goserelin→ER+ 错误推断 | P2: same |
| 52 | P2: procedure mixed fertility | P2 |
| 57 | P2: procedure mixed genetic counseling | ✅ |
| 74 | P2: HER2+ 混淆 gastric cancer | P2: same |
| 75 | P2: procedure mixed genetics+fertility | P2 |
| 83 | P2: Stage IV but Distant Met=No | P2: same |
| 88 | P1: response "Not mentioned" for clear progression | P1: same |
| 95 | P2: Stage IV but ISPY trial (Stage I-III) | ✅ (tool fixed) |

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 5 | 17.9% | ROW 8, 10, 11, 12, 88（全部是 response_assessment 或 advance care 问题）|
| **P2** | 29 | — | 详见下方各 ROW |

---

## 逐 Sample 问题清单（每个 ROW 独立条目）

### ROW 1 (coral_idx 140) — 0 P1, 2 P2
- P2: lab_plan "ordered a MRI of brain and bone scan as well as labs. labs to complete" — 仍混入 imaging（MRI + bone scan），但比 v26 notool P1 改善（v26 lab_plan 完全是 imaging 内容，v27 至少有 "labs to complete"）
- P2: imaging_plan "Brain MRI" — 遗漏了 Bone Scan。A/P 说 "ordered a MRI of brain and bone scan"。与 v26 tool 同样问题，未修复
- ✅ 56yo postmenopausal, Stage IIA→IV ER+/PR+/HER2- IDC right breast, widespread mets（lungs, peritoneum, liver, ovary）+ local recurrence axilla 3cm
- ✅ S/p right mastectomy June 2013（2.4+2.3cm, node negative, G2），declined tamoxifen. No adjuvant therapy
- ✅ Type: "ER+/PR+ invasive ductal carcinoma, HER2-" 正确
- ✅ Response: "Not yet on treatment" 正确（新诊断 met）
- ✅ Goals: palliative 正确
- ✅ Procedure: axilla biopsy Thursday 正确
- ✅ Medication_plan: ibrance + [redacted] if HR+/HER2- confirmed on biopsy
- ✅ Advance care: Full code 正确
- ✅ Referral: Integrative Medicine 正确
- ✅ Letter: 出色 — "cancer that started in the milk ducts" + peritoneum 解释 + emotional support 有原文支持（"She is very scared and appears anxious"）
- **vs v26**: lab_plan P1→P2（改善但未完全修复），imaging bone scan 遗漏未修复

### ROW 2 (coral_idx 141) — 0 P1, 0 P2 ✅
- ✅ 44yo, Lynch Syndrome, Stage IIB→IV TNBC metastatic to liver + bone + chest wall. Also colon cancer Stage I + endometrial cancer
- ✅ On irinotecan cycle 3 day 1, dose changing to q2wk 150mg/m2 due to poor tolerance
- ✅ Type: "ER-/PR-/HER2- triple negative IDC" 正确
- ✅ Response: "No specific imaging or lab results to assess current response" — 正确（no new scans, back pain worse 可能 PD，scans 3 months pending）
- ✅ Lab: 完整 CMP+CBC，正确记录 severe abnormalities（Hgb 7.7, Albumin 2.1, Na 124, K 3.1）
- ✅ Medication_plan: comprehensive — doxycycline + morphine + flexeril + effexor increase + NS + potassium + pRBC transfusion
- ✅ Radiotherapy: urgent rad onc referral for sacral pain 正确
- ✅ Referral: Rad Onc + social work + home health 正确
- ✅ Letter: 出色 — TNBC 通俗解释 + anemia + emotional support 用标准模板（不再具体说 "anxious and depressed"）
- ✅ **vs v26**: v26 notool 3 P2 全部修复！（Patient type 正确，letter 不再编造情绪词，Home health 处理合理）

### ROW 3 (coral_idx 142) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, Stage IIA ER+/PR+/HER2-（IHC 2+, FISH negative）IDC right breast, 1.7cm, node positive（1.5cm axillary LN），Ki-67 30-35%
- ✅ Genetic_testing_plan: "Genetic testing sent and is pending" — 正确放在了 genetic_testing_plan 字段
- ✅ Type: "ER+/PR+/HER2- invasive ductal carcinoma" 正确
- ✅ Response: "Not yet on treatment" 正确（新诊断，pending PET + Oncotype + biopsy）
- ✅ Goals: curative 正确
- ✅ Imaging: PET CT scheduled，follow-up after results
- ✅ Advance care: full code 正确
- ✅ Letter: 通俗准确 — "cancer started in the milk ducts" + "lymph node in your armpit is also affected" + emotional support 用标准模板
- ✅ **vs v26**: v26 notool P2（genetic_testing 放错字段）修复！genetic_testing 现在正确在 genetic_testing_plan 中

### ROW 4 (coral_idx 143) — 0 P1, 0 P2 ✅
- ✅ 75yo postmenopausal, ER+/PR+/HER2-（IHC 2+, FISH negative）IDC left breast, 2.8cm grade 2, s/p left mastectomy + SLN 10/29/16, on letrozole since 12/2016
- ✅ Response: "no evidence of disease recurrence on imaging, exam, and review of systems" — 直接引用 A/P，正确
- ✅ Current_meds: Letrozole 正确
- ✅ Medication_plan: comprehensive — letrozole + calcium/vitamin D + magnesium + conditional prolia
- ✅ Imaging: mammogram July 2019 + DEXA July 2019 + consider brain MRI if headaches worsen — 完整
- ✅ Goals: curative 正确
- ✅ Osteoporosis management: BMD improving（femur neck T-score -2.4, osteopenia range）, continue monitoring
- ✅ **vs v26**: v26 was 0/0, v27 remains 0/0. No regressions.

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅
- ✅ 31yo premenopausal, Stage III→IV ER+/PR+/HER2- IDC left breast, metastatic recurrence to cervical LN + brachial plexus + possible sternal bone met
- ✅ On leuprolide + anastrozole + palbociclib. Tolerating well（occasional zofran）
- ✅ Response: 出色 — "cervical chain LNs decreased, mediastinal LN stable, axillary LN increased, new sternal sclerotic lesion" 准确传达 mixed response
- ✅ MRI brain: normal. C-spine MRI: enlarged left level 5B LN involving brachial plexus
- ✅ Radiotherapy: rad onc referral for symptomatic left neck/brachial plexus — 正确
- ✅ Imaging: CT + bone scan ordered for restaging
- ✅ Lab_plan: labs monthly on lupron day — 正确
- ✅ Goals: palliative, Advance care: full code
- ✅ **vs v26**: v26 notool P2（follow_up 循环表述）修复！v27 follow_up 不再循环

### ROW 6 (coral_idx 145) — 0 P1, 2 P2
- P2: Patient type "New patient" — zoladex 已于 06/08 由该提供者开始（说明已就诊过），应为 Follow up。与 v26 同样的问题，未修复
- P2: Referral-Genetics "Dr. [REDACTED] at [REDACTED]. genetics referral" — 历史转诊（04/24），Myriad 已完成且阴性。与 v26 同样的问题，未修复
- ✅ 34yo, ER+/PR+/HER2-（IHC 2, FISH non-amplified）IDC right breast, 1.5cm grade 1, 0/1 node, s/p bilateral mastectomy 06/21, zoladex + letrozole adjuvant
- ✅ HER2-: IHC 2+ FISH non-amplified = negative，正确
- ✅ Lab: 完整 CMP + CBC + Estradiol 172 + Vitamin D 24
- ✅ Medication_plan: letrozole + zoladex x 3 years → tamoxifen sequencing + gabapentin + estradiol monthly 正确
- ✅ Lab_plan: "Estradiol monthly" 正确
- ✅ Goals: curative 正确
- ✅ **vs v26**: letter 不再有 "You appear to be feeling anxious" 编造（v26 P2 修复！），但 Patient type 和 Referral-Genetics P2 仍在（v26 3 P2 → v27 2 P2）

### ROW 7 (coral_idx 146) — 0 P1, 2 P2
- P2: procedure_plan "Would recheck [REDACTED] prior to starting..." — [REDACTED] 是 LVEF/echo（imaging），不是 procedure。与 v26 同样问题
- P2: lab_plan "Would recheck [REDACTED] prior to above" — 同上，LVEF 是 imaging 不是 lab。imaging_plan 写 "No imaging planned" 是错误的
- ✅ MBC since 2008, originally Stage II T2N1 IDC 1998, met biopsy ER-/PR-/HER2+ by IHC
- ✅ Type: "Originally unclear receptor status, metastatic biopsy ER-/PR-/HER2+ IDC" — 出色，正确区分了原发（不清楚）vs 转移（ER-/PR-/HER2+）
- ✅ Response: "Probable mild progression...equivocal...SUV 2.1 (was 1.8)...tumor marker 14.8 but not increased" — 准确传达不确定性
- ✅ Second opinion: "yes" 正确
- ✅ Goals: palliative 正确
- ✅ Medication_plan: d/c pertuzumab/herceptin/taxotere + recommend [REDACTED] as next line 正确
- ✅ **vs v26**: letter 不再有 "medication level" 误述 → v27 正确写了 "ejection fraction of 52%"（v26 P2 修复！）。但 procedure_plan/lab_plan LVEF 混入未修复（v26 3 P2 → v27 2 P2）

### ROW 8 (coral_idx 147) — 1 P1, 1 P2
- **P1**: response_assessment "Not yet on treatment — no response to assess" — 与 v26 完全相同的 P1！患者完成了 neoadjuvant TCHP（不完整，3 cycles）+ surgery。Breast pCR（无残留侵润癌，cellularity 0%）but 3/28 LN positive（largest 2.4cm with extranodal extension, ER-/PR-/HER2+ IHC 3+, FISH 5.7, Ki-67 75%）。应描述 "mixed pathologic response: breast pCR but residual nodal disease"。**v26 P1 未修复**
- P2: procedure_plan "adjuvant AC x 4 cycles, to be followed by T-DM1" — 化疗方案混入 procedure_plan（应在 medication_plan，已正确有）。procedure_plan 应写 "port placement"
- ✅ 29yo premenopausal, Stage II-III→III ER-/PR-/HER2+（IHC 3+, FISH 5.7）IDC left breast
- ✅ S/p incomplete neoadjuvant TCHP (3 cycles, non-adherent) + lumpectomy/ALND
- ✅ Type: "ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) IDC" 非常详细正确
- ✅ Findings: 极其详细的病理描述 — breast pCR, 3/28 LN+, extranodal extension, necrotizing lymphadenitis
- ✅ Medication_plan: adjuvant AC x 4 → T-DM1 正确
- ✅ Imaging_plan: echocardiogram before AC 正确（anthracycline 心脏毒性）
- ✅ Goals: curative 正确
- ✅ Televisit, ECOG 0
- ❌ **vs v26**: P1 response 未修复（仍说 "Not yet on treatment" 但有 post-neoadjuvant 病理响应数据）


### ROW 10 (coral_idx 149) — 1 P1, 0 P2
- **P1**: response_assessment "Low risk [REDACTED]." — 这是 Oncotype 基因检测结果（"low risk...does not need chemotherapy"），不是治疗响应评估。患者做了 neoadjuvant letrozole（April 2021）→ 左乳切除术 07/24/2021 → 病理：8.8cm 残留肿瘤 + 淋巴结受累。这是 neoadjuvant 后的病理响应，提示有显著残留病灶。response_assessment 应该描述这些病理发现，而非引用 Oncotype 结果。**v26 P1 未修复**
- ✅ 66yo postmenopausal, Stage II left breast cancer HR+/HER2-, s/p neoadjuvant letrozole April 2021
- ✅ S/p left mastectomy 07/24/2021: 8.8cm [REDACTED] + lymph nodes involved; bilateral reductions + re-excision for margins 08/07/2021
- ✅ Patient type "Follow up" 正确（video visit FUP）
- ✅ Type_of_Cancer: "ER+ (inferred from letrozole), HER2-" — 实质正确（A/P 写 "HR + and her 2 negative"），虽说 "inferred from letrozole" 不如直接引 "HR+" 准确，但不影响临床判断
- ✅ Goals: curative 正确（Stage II, adjuvant therapy）
- ✅ Medication_plan: continue letrozole 正确
- ✅ Radiotherapy_plan: radiation to left chest wall + surrounding lymph nodes 正确
- ✅ Imaging_plan: DEXA 正确
- ✅ Advance care: full code 正确
- ✅ Letter: 通俗准确 — 手术+letrozole+radiation+DEXA 全部正确覆盖，无编造。小遗漏："不需要化疗" 这个好消息未提及，但不构成 P2
- ❌ **vs v26**: P1 response 仍未修复（Oncotype 结果 ≠ pathologic response after neoadjuvant）


### ROW 11 (coral_idx 150) — 1 P1, 1 P2
- **P1**: response_assessment "PET/CT showed increased metastatic activity and size of left mandibular mass, indicating disease progression on current treatment with Faslodex and Denosumab" — 时间线错误。PET/CT 是 10/10/12 做的，在 Faslodex+Denosumab 开始（10/16/12）之前。该 PET/CT 显示的是 prior treatment (letrozole) 下的进展，才导致换药。当前 A/P 明确写 "Exam stable"。把旧扫描进展归因到当前治疗，颠倒了治疗响应。**v26 P1（response 引用旧 PET）未修复**
- P2: imaging_plan 只写了 PETCT，遗漏了 A/P 中 "MRI of lumbar, pelvis and right femur"。v26 的 Echo P2 已修复，但引入了新的 MRI 遗漏
- ✅ 68yo, Stage IIIC→IV left breast IDC, bone mets (spine compression fx + mandible), on Faslodex + Denosumab since 10/16/12
- ✅ S/p mastectomy 10/2010, Taxotere/Cytoxan x4, XRT June 2011, XRT to T spine + right femur + jaw
- ✅ Type_of_Cancer: IDC, ER+ (from letrozole/Faslodex), HER2 not tested — 合理推断
- ✅ Stage: "Originally Stage IIIC, now metastatic (Stage IV)" 正确
- ✅ Lab: 完整 CBC + BMP（全部值与原文一致）
- ✅ Goals: palliative 正确
- ✅ Medication_plan: continue Faslodex + Denosumab + Mycelex for thrush 正确
- ✅ recent_changes: started Mycelex 正确
- ✅ supportive_meds: docusate, hydrocodone-acetaminophen, senna-docusate 正确
- ✅ Letter: 通俗准确 — 骨转移 + 下颌增大 + 腿痛 + Mycelex + PETCT，无编造
- **vs v26**: P1 response 未修复（仍引用旧 PET 归因到当前治疗），Echo P2 修复但引入 MRI 遗漏新 P2


### ROW 12 (coral_idx 151) — 1 P1, 2 P2
- **P1**: Advance care "Not discussed during this visit." — 但 ACTIVE problem list 明确记录 DNR/DNI、POLST completed、"would not want life support treatments such as resuscitation, intubation/mechanical ventilation, or artificial nutrition"、"would want trial of medical interventions for potentially reversible conditions"。这是当前有效的预先医嘱。**v26 P1（Advance care DNR/DNI 遗漏）未修复**
- P2: response_assessment 只报告 body SD（CT CAP stable, no evidence of PD），完全遗漏 brain progression（01/31 MRI 显示 2 new enhancing foci）。A/P 写 "However has another episode of brain mets"。实际是 mixed response（body SD + brain PD）
- P2: imaging_plan 遗漏 Echo q6 months（A/P 明确写 "Echo q6 months, recent reviewed and stable/normal, repeat again in April 2019"），herceptin 心脏毒性监测
- ✅ 51yo, de novo Stage IV ER+/PR+/HER2+（IHC 3+, FISH 5.4）breast cancer，bone（广泛 spine/ribs/sternum/pelvis/skull）+ brain（多次 GK）+ lung（now [REDACTED]）+ nodes
- ✅ 极复杂治疗史：XRT T spine → herceptin/[REDACTED]+letrozole → taxotere（sepsis）→ taxol（sepsis）→ off chemo。GK x3（23+19+17 lesions）
- ✅ Type: ER+/PR+/HER2+ 正确
- ✅ Goals: palliative 正确
- ✅ Medication_plan: continue herceptin/[REDACTED]+letrozole, [REDACTED] q12wks, off chemo — 准确
- ✅ Imaging_plan: CT CAP q4mo + bone scan + MRI brain q4mo 正确（但缺 Echo）
- ✅ Radiotherapy_plan: await GK/Rad Onc input 正确
- ✅ Referral: Rad Onc + F/u Dr. [REDACTED] 正确
- ✅ Follow-up: 6 weeks 正确
- ✅ Letter: 出色 — "brain has new small spots but body is stable" 准确传达 mixed response + 通俗化骨药频率调整 + 化疗不耐受。无编造
- **vs v26**: P1 Advance care DNR/DNI 未修复。v26 tool P2（lung mets resolved）仍存在边界模糊（A/P 仍列 lung 但 CT clear）。新增 P2×2（response 遗漏脑进展 + imaging 遗漏 Echo）


### ROW 13 (coral_idx 152) — 0 P1, 2 P2
- P2: response_assessment "On treatment; response assessment not available" — 患者 NOT on treatment。DCIS 已手术切除（margins clear），仅在讨论是否开始 tamoxifen（尚未开始）。**v26 P2（response "On treatment"）未修复**
- P2: findings 侧别混淆 — extraction 写 "Left breast 14 mm solid mass...benign with focal adenosis and PASH"。实际上 RIGHT breast 14mm → fibroadenoma，LEFT breast MRI-guided biopsy → adenosis+PASH。两个活检结果被交叉归错。**v26 P2（findings laterality）未修复**
- ✅ 41yo premenopausal, left ER+ nuclear grade 2 DCIS s/p left partial mastectomy（18mm, G2, margins clear）
- ✅ [REDACTED] DCIS score 60, Invitae 46 gene panel negative
- ✅ Type: "ER+ DCIS, HER2: not tested" 正确（DCIS 不需 HER2 检测）
- ✅ Goals: "risk reduction" 出色 — DCIS s/p 手术后讨论 tamoxifen 确实是 risk reduction
- ✅ Medication_plan: patient will consider tamoxifen 正确
- ✅ Radiotherapy_plan: Rad Onc consult for adjuvant radiation 正确
- ✅ Referral: Radiation oncology 正确
- ✅ Letter: 出色 — "cancer that started in the milk ducts (DCIS)" 通俗化 + tamoxifen risk/benefit + radiation + prognosis excellent + lifestyle factors。无编造
- **vs v26**: 两个 P2（response "On treatment" + findings laterality）均未修复


### ROW 14 (coral_idx 153) — 0 P1, 2 P2
- P2: current_meds "" — 患者正在自行服用 Mexico 处方的低剂量化疗（[REDACTED] 10mg + Gemcitabine 200mg + Docetaxel 20mg weekly）+ pamidronate + immunological vaccines。虽然 recent_changes 字段正确捕获了这些药物，但 current_meds 为空不反映实际用药情况。**v26 P2（current_meds 空）未修复**
- P2: findings 写 "no lymphadenopathy" 但 PE 明确记录 "Palpable R axillary node 1 cm, soft and mobile"；R breast density 1.5x2.0cm 也未反映在 findings 中。findings 与 PE 矛盾
- ✅ 58yo, de novo Stage IV ER+/PR+/HER2-（IHC 1+, FISH neg）breast cancer, mets to bone（extensive spine + pathologic fractures + cord compression）+ liver + nodes
- ✅ 极复杂治疗史：spine fixation + corpectomy → letrozole → Faslodex + palbociclib → extensive spine surgery + XRT → 自行停药 → Mexico 低剂量化疗
- ✅ Lab: 完整 CMP+CBC+CA 27.29，CA 27.29 从 193 下降到 48（持续下降趋势）
- ✅ Goals: palliative 正确
- ✅ recent_changes: 出色 — 完整记录停药 + Mexico 治疗方案
- ✅ Imaging_plan: CT CAP + Total Spine MRI for May + repeat spine MRI in 6 weeks 正确
- ✅ Lab_plan: labs every two weeks 正确
- ✅ Referral: PT 正确
- ✅ Letter: 全面准确 — 停药 + 新方案 + 影像计划 + PT + 骨健康。无编造
- **vs v26**: current_meds P2 未修复（但 recent_changes 现在有了药物信息，部分改善）。新增 findings 与 PE 矛盾的 P2


### ROW 15 (coral_idx 154) — 0 P1, 1 P2
- P2: genetic_testing_plan "biomarker testing" — ER/PR/HER2 biomarker testing 已全部完成且结果在笔记中。A/P "reviewed pathology, biomarker testing" 是回顾已有结果，不是未来计划。应为空或"None planned"。**v26 P2（genetic_testing）类似问题，未完全修复**
- ✅ 46yo premenopausal, newly diagnosed left breast cancer, mixed IDC + ILC, ER+(>95%)/PR+(80-90%)/HER2+（IHC 2+, FISH ratio 2.0）
- ✅ Clinical Stage I/II, no metastasis
- ✅ HER2 borderline but classified HER2+ per ASCO/CAP — physician agrees to treat as HER2+ (young, fit, large benefit)
- ✅ Type: "ER+/PR+/HER2+ mixed infiltrating ductal and lobular carcinoma" 出色
- ✅ Goals: curative 正确
- ✅ Response: "Not yet on treatment" 正确
- ✅ Medication_plan: TCHP if opts for neoadjuvant 正确
- ✅ Procedure_plan: correctly captures surgery-first vs neoadjuvant-first options。**v26 P2（procedure_plan mixed）已修复！**
- ✅ Imaging_plan: ultrasound (R breast second look) 正确
- ✅ Referral: Breast surgery 正确
- ✅ Letter: 出色 — "mix of two types: milk-producing glands + milk ducts" 通俗化 mixed IDC/ILC + "start medication first to make surgery easier" 通俗化 neoadjuvant 概念。无编造
- **vs v26**: procedure_plan P2 修复！genetic_testing P2 仍在（v26 2 P2 → v27 1 P2，改善）


### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan "Abdomen, Pelvis, Rad Onc referral, Xgeva - needs dental evaluation first" — 混入了 imaging（CT CAP 的 Abdomen/Pelvis）+ radiotherapy referral（已在 radiotherapy_plan）+ medication（Xgeva/dental）。procedure_plan 应为 "dental evaluation" 或 "None"。**v26 P2（procedure_plan mixed）未修复**
- ✅ 75yo postmenopausal, early stage ER+/PR+/HER2- IDC (2009) s/p bilateral mastectomy + 5yr tamoxifen → metastatic recurrence to bone（innumerable osseous lesions）+ lymph nodes
- ✅ R iliac crest biopsy: ER+(80%)/PR+(50%)/HER2-(FISH 1.05) — 与原发一致
- ✅ Family Hx: 3 sisters breast cancer + paternal aunt ovarian cancer
- ✅ Type: "ER+/PR+/HER2- IDC" 正确
- ✅ Goals: palliative 正确
- ✅ Medication_plan: letrozole + palbociclib + denosumab (dental clearance) + monthly blood work — 完整准确
- ✅ Imaging_plan: MRI total spine + CT CAP + repeat at 3 months 正确
- ✅ Lab_plan: tumor markers + monthly blood work 正确
- ✅ Genetic_testing_plan: Foundation One or [REDACTED] 360 正确
- ✅ Radiotherapy_plan: Rad Onc consult（migratory pain）正确
- ✅ Follow-up: ~1 month 正确
- ✅ Letter: 出色 — 全面覆盖 metastatic recurrence + 新治疗方案 + denosumab 骨保护 + imaging/lab/follow-up。无编造
- **vs v26**: procedure_plan mixed P2 未修复


### ROW 22 (coral_idx 161) — 0 P1, 2 P2
- P2: lab_summary "No labs in note" — 笔记包含 01/29/2021 CBC+CMP 结果，虽然是 8 个月前的，但有多项临床显著异常（Hgb 10.7(L), Lymph 0.54(L), Creatinine 1.19(H), eGFR 46(L)）。提取完全忽略
- P2: genetic_testing_plan 包含 medication plan 文本（"faslodex with [REDACTED] if she has a [REDACTED] mutation"）而非 testing plan。应为"test for [REDACTED] mutation if progression"。**v26 P2（genetic_testing has medication plan）未修复**
- ✅ 72yo, L DCIS 1994 + R Stage II IDC 2000 → metastatic recurrence May 2020（chest wall + bone + R infraclavicular + IM nodes），HR+/HER2-
- ✅ On abemaciclib + anastrozole → abemaciclib d/c 08/14/21 due to pneumonitis → steroids
- ✅ PET 11/03/20 + 04/24/21: good response
- ✅ Type: "ER+/PR+/HER2- IDC" 正确
- ✅ Goals: palliative 正确
- ✅ Current_meds: anastrozole + denosumab 正确
- ✅ Medication_plan: 出色的 contingent plan — arimidex if stable, faslodex+[REDACTED] if progression, future options
- ✅ Imaging_plan: PET CT now 正确
- ✅ Advance care: Full code 正确
- ✅ Letter: 全面准确 — 历史 + 治疗变化 + contingent plan + future options。无编造
- **vs v26**: genetic_testing P2 未修复。新增 lab_summary P2


### ROW 24 (coral_idx 163) — 0 P1, 2 P2
- P2: Metastasis "Not sure" + Distant Metastasis "Not sure" — PET CT 明确 "No definite sites of hypermetabolic metastatic disease"，liver lesions benign。SLN micromets (0.4mm = pN1mi) 是 regional 不是 distant。Distant Met 应为 "No"。**v26 P2 未修复**
- P2: procedure_plan 包含 genetic testing 内容（send specimen for MP testing, order [REDACTED] test）。这是基因检测不是手术程序，已在 genetic_testing_plan 正确捕获。**v26 P2（procedure has genetic testing）未修复**
- ✅ 56yo, R breast Grade II micropapillary mucinous carcinoma ER+(>95%)/PR+(80%)/HER2-(FISH 1.1), Ki-67 5%
- ✅ S/p R partial mastectomy, 2 SLN with micromet (0.4mm), negative margins
- ✅ PET: no definite metastatic disease; liver lesions benign（hemangioma + FNH）
- ✅ Type: 非常详细正确
- ✅ Goals: curative 正确
- ✅ Medication_plan: adjuvant hormone therapy if genomic low risk 正确
- ✅ Genetic_testing_plan: send for MP (MammaPrint) for chemo benefit 正确
- ✅ Radiotherapy_plan: radiation + Rad Onc 12/07/18 正确
- ✅ Referral: Rad Onc + PT 正确
- ✅ Letter: "cancer that makes mucus" 通俗化 + "test if you need chemo" + radiation。遗漏 hormone therapy 但 extraction 正确
- **vs v26**: 两个 P2 均未修复（Metastasis "Not sure" + procedure has genetic testing）


### ROW 25 (coral_idx 164) — 0 P1, 1 P2
- P2: medication_plan "Patient will start cycle of 1500/1000mg ixabepilone" — 1500/1000mg 是 Xeloda 的剂量（3 tabs AM=1500mg + 2 tabs PM=1000mg），不是 ixabepilone（40mg/m2 IV）。药物-剂量配对错误
- ✅ 45yo, 极复杂历史 — R breast IDC 2007 (ER+/PR+) → L breast IDC 2008 (ER+/PR-/HER2-) → bilateral mastectomies 2009 → metastatic 12/2010 (brain + liver + bone + LN + chest wall)
- ✅ S/p brain met resection + WBRT → Xeloda → PD → added ixabepilone + denosumab
- ✅ Type: 正确区分原发 vs 转移受体状态
- ✅ Goals: palliative 正确
- ✅ Current_meds: capecitabine + ixabepilone 正确
- ✅ Lab: Alk Phos 308(H), AST 55(H), Hgb 11.2(L), Albumin 3.1(L) — 完整
- ✅ Response: 包含 PET PD（Xeloda alone）+ "supraclavicular area appears to be breaking up"（current regimen positive）。**v26 P2（response 引用旧 PET）已改善！**
- ✅ Imaging_plan: scan in 3 weeks 正确
- ✅ Letter: PD + supraclavicular 改善 + Xeloda + pain management + scan。无编造
- **vs v26**: response P2 已改善（现在包含 exam 改善）！新增 medication_plan 药物-剂量配对 P2


### ROW 33 (coral_idx 172) — 0 P1, 0 P2 ✅
- ✅ 63yo, left ER+/PR+/HER2- invasive lobular carcinoma, Stage IIB/IIIA, LN+
- ✅ S/p bilateral mastectomies + TC x 6 + XRT, on adjuvant letrozole since 02/2011
- ✅ Type: "ER+/PR+/HER2- invasive lobular carcinoma" 正确识别 ILC
- ✅ Goals: curative 正确
- ✅ Response: "no evidence of recurrence on exam" 正确 — A/P 明确写 NED
- ✅ Medication_plan: letrozole + calcium/vitamin D + NSAIDs 完整
- ✅ Imaging_plan: conditional MRI brain（if [REDACTED] headaches persist）
- ✅ Follow-up: 6 months 正确
- ✅ Letter: 通俗准确 — NED 解释 + letrozole 功能 + 小 LN 解释 + f/u。无编造
- **vs v26**: P2 letter stage confusion 已修复！（v27 letter 未提及 stage，避免了混淆）


### ROW 38 (coral_idx 177) — 0 P1, 1 P2
- P2: response_assessment "not currently responding to treatment" — 患者 NOT on treatment（neoadjuvant chemo 因毒性停止，等待手术 Jan 31）。肿瘤在 treatment-free hiatus 期间增大（6.8→8x5cm）。应描述为 enlarging during hiatus，而非暗示当前治疗失败。**v26 P2 未修复**
- ✅ 43yo, BRCA1, Stage IIB left breast IDC 6.8cm→8x5cm, ER-/PR+(weak 15%)/HER2-, node negative
- ✅ S/p incomplete neoadjuvant（[REDACTED] x 4 + 5wk taxol, stopped toxicity）→ mild response → tumor regrowing
- ✅ Type: "ER-/PR+/HER2-" 正确
- ✅ Goals: curative 正确（Stage IIB, planning definitive surgery）
- ✅ Medication_plan: olaparib for 1 year (BRCA1) + adjuvant xeloda 正确
- ✅ Procedure_plan: bilateral mastectomy Jan 31 正确
- ✅ Referral: Gyn Onc (BRCA1) + Social work 正确
- ✅ Lab: 完整正常
- ✅ Letter: 全面准确 — surgery plan + adjuvant options + radiation + referrals + lifestyle。**v26 tool P2（letter anxious）已修复！**无编造
- **vs v26**: response P2 未修复（仍说 "not responding to treatment" 但不在治疗中）。letter anxious P2 已修复


### ROW 39 (coral_idx 178) — 0 P1, 1 P2
- P2: Type_of_Cancer 错误添加 "ER+ (inferred from goserelin)" — 癌症是 TRIPLE NEGATIVE（biopsy ER/PR/[REDACTED] negative，A/P "triple negative breast cancer"）。Goserelin 用于化疗期间 fertility preservation（A/P 明确写 "improved fertility preservation with this approach"），不是因为 ER+。这个错误推断改变了整个癌症分类（TNBC→HR+）。**v26 P2 未修复**
- ✅ 27yo, newly diagnosed left breast grade 3 IDC, T2N1, triple negative
- ✅ Left ovary removed for cryopreservation, ISPY trial consent signed
- ✅ Goals: curative 正确
- ✅ Medication_plan: paclitaxel x 12wk → AC x 4 + goserelin for fertility 正确
- ✅ Procedure_plan: port placement 正确
- ✅ Imaging_plan: echo + breast MRI 正确
- ✅ Lab_plan: ISPY labs 正确
- ✅ Letter: 准确 — IDC + LN spread + chemo plan + port/MRI/echo/ISPY。比 keypoints 更准确（letter 没有说 ER+）。无编造
- **vs v26**: goserelin→ER+ 错误推断 P2 未修复


### ROW 52 (coral_idx 191) — 0 P1, 1 P2
- P2: procedure_plan 混入 fertility referral + medication plan（Zoladex prior auth）+ genetic testing。应分别在 Referral、medication_plan、genetic_testing_plan。**v26 P2（procedure mixed fertility）未修复**
- ✅ 35yo premenopausal, left IDC 1.7cm grade II, ER+(>95%)/PR+(>95%)/HER2-, Ki-67 10-15%, SLN micromet (0.18cm)
- ✅ Type, Stage, Goals (curative), Response (not yet on treatment) 正确
- ✅ Medication_plan: [REDACTED]+Zoladex after egg harvesting 正确
- ✅ Imaging_plan: CT CAP + bone scan for staging 正确
- ✅ Genetic_testing_plan: order [REDACTED] for chemo benefit 正确
- ✅ Letter: IDC + Zoladex 解释 + fertility preservation + staging scans。无编造


### ROW 57 (coral_idx 196) — 0 P1, 0 P2 ✅
- ✅ 59yo, left breast, locally advanced TNBC（initially classified as [REDACTED/HER2+?] → surgical specimen + path review confirmed TNBC）
- ✅ S/p neoadjuvant TCH+P → residual 3.7cm, 0/6 LN → post-op AC x 4 → XRT planned
- ✅ Type: "ER-/PR-/HER2- triple negative" 正确
- ✅ Response: 出色 — "neoadjuvant TCH+P → residual 3.7cm, did not achieve pCR"
- ✅ Radiotherapy_plan: XRT scheduled 正确
- ✅ Genetic_testing_plan: genetic counseling + testing 正确
- ✅ Letter: TNBC 解释 + residual disease + XRT + genetic counseling。无编造
- **vs v26**: procedure_plan mixed genetic counseling P2 已修复！

### ROW 74 (coral_idx 213) — 0 P1, 1 P2
- P2: Type_of_Cancer "ER+/PR+/**HER2+** IDC" — breast cancer 是 **HER2-**（IHC 1+, FISH ratio 1.1）。HER2+ 是患者的 **gastric cancer**（IHC 3+, 已确认为 separate primary）。A/P 明确写 "ER+/PR+/HER- with FISH ratio 1.1"。**v26 P2（HER2+ 混淆 gastric cancer）未修复**
- ✅ 68yo, 两个原发癌：HER2+ gastric cancer（2015, 已缓解）+ Stage IIB R breast IDC（ER+/PR+/HER2-, T2N1M0, 2.5cm grade 2）
- ✅ S/p bilateral mastectomies + R axillary dissection, 1/7 LN+ (0.5cm)
- ✅ Goals: curative 正确
- ✅ Medication_plan: AI + consider TC（not anthracycline due to borderline EF + prior chemo）正确
- ✅ Genetic_testing_plan: [REDACTED] testing ordered 正确
- ✅ Letter: "early stage breast cancer" + AI + testing + follow-up。无编造


### ROW 75 (coral_idx 214) — 0 P1, 1 P2
- P2: procedure_plan 混入 genetics counseling + fertility referrals + UCSF Breast Surgery referral。这些是 referrals 不是 procedures。**v26 P2（procedure mixed genetics+fertility）未修复**
- ✅ 33yo premenopausal, R breast high-grade ER-/PR-/HER2+ IDC, grade 3, Ki-67 70-80%, Stage II-III, axillary LN+ to level III
- ✅ Medication_plan: TCHP + adjuvant T-DM1 if residual 出色
- ✅ Referral: genetics + UCSF Breast Surgery + Radiation Oncology 正确
- ✅ Letter: TCHP + port + surgery + genetic counseling + fertility。无编造

### ROW 83 (coral_idx 222) — 0 P1, 1 P2
- P2: Stage "Stage IV (metastatic)" but Distant Metastasis "No"。腋窝 LN = regional，没有远处转移不能是 Stage IV（最多 Stage III）。**v26 P2（Stage IV but Distant Met=No）未修复**
- ✅ 77yo, R breast invasive lobular carcinoma, R axillary LN involvement
- ✅ On neoadjuvant letrozole, PET shows significant response（axillary SUV 15.1→1.9）
- ✅ Response: 出色 — 含具体 PET/SUV 数值变化
- ✅ Goals: curative 正确
- ✅ Medication_plan: continue neoadjuvant letrozole 正确
- ✅ Letter: lobular cancer 通俗化 + letrozole + breast surgery。无编造


### ROW 88 (coral_idx 227) — 1 P1, 0 P2
- **P1**: response_assessment "Not mentioned in note" — 笔记中有丰富的 progression 证据：neoadjuvant "stopped for progression of disease"、on last day of XRT → 2 brain mets、staging shows lung + LN metastases。response 完全不是 "not mentioned"。**v26 P1 未修复**
- ✅ 36yo, Stage III→IV left breast IDC（ER weak/PR weak/HER2-），s/p incomplete neoadjuvant（PD on chemo）→ bilateral mastectomies（4+2.6cm, 23/30 LN+, grade III/III）→ adjuvant gemzar/carbo → XRT → brain mets → resection + SRS → lung + LN mets → Xeloda
- ✅ Type: 正确区分原发（ER weak+/PR weak+/HER2-）vs metastatic（ER-/PR-/HER2 pending）
- ✅ Goals: palliative 正确
- ✅ Medication_plan: Xeloda + anti-HER2 if confirmed + immunotherapy trials 全面
- ✅ Genetic_testing_plan: HER2 retesting on brain met + residual disease 正确
- ✅ Letter: comprehensive — 全面准确。无编造

### ROW 95 (coral_idx 234) — 0 P1, 1 P2
- P2: Stage "Stage IV (metastatic)" — 患者在 ISPY 试验（Stage I-III only），s/p 部分乳房切除，axillary LN = regional，Distant Met = "No"。Goals = curative。Stage IV 与所有其他信息矛盾。**v26 P2 未修复**（v26 tool 已修复但 notool 仍有）
- ✅ 49yo, left ER+/PR+/HER2- IDC, ISPY trial Pembrolizumab arm, s/p neoadjuvant → partial mastectomy
- ✅ Response: "good response to NAC with 3 smallish lesions with low cellularity" 出色
- ✅ Goals: curative 正确
- ✅ Medication_plan: prilosec + capecitabine after XRT（CREATE-X）+ adjuvant endocrine therapy 正确
- ✅ Radiotherapy_plan: breast + axilla XRT, Rad Onc referred 正确
- ✅ Letter: MRI response + capecitabine + XRT + hormone therapy。无编造

---

## 最终汇总

### 总体统计
| 指标 | 数值 |
|------|------|
| 总样本 | 28 |
| P0 | 0 (0%) |
| P1 | 5 (17.9%) |
| P2 | 29 |
| Clean (0/0) | 5 (17.9%): ROW 2, 3, 4, 5, 33 |

### P1 汇总（5 个）
| ROW | 问题 | v26 对比 |
|-----|------|---------|
| 8 | response "Not yet on treatment" — 有 post-neoadjuvant 病理数据 | v26 P1 未修复 |
| 10 | response = Oncotype result（genomic test ≠ pathologic response） | v26 P1 未修复 |
| 11 | response 引用旧 PET 归因到当前治疗 | v26 P1 未修复 |
| 12 | Advance care 遗漏 DNR/DNI（在 ACTIVE problem list 中） | v26 P1 未修复 |
| 88 | response "Not mentioned" 但有明确 progression 证据 | v26 P1 未修复 |

### P1 模式：response_assessment 是最大弱点（4/5 P1），模型对 post-neoadjuvant 病理响应、时间线归因、progression 识别持续困难

### v26→v27 修复统计
| 状态 | 数量 | 示例 |
|------|------|------|
| v26 P2 已修复 | 7 | ROW 2（全修复）, 3（genetic field）, 5（follow_up）, 15（procedure）, 25（response improved）, 33（letter stage）, 57（procedure） |
| v26 P2 未修复 | 22 | 详见各 ROW 条目 |
| v26 P1 未修复 | 5 | ROW 8, 10, 11, 12, 88 |
| 新增 P2 | ~8 | ROW 11（MRI 遗漏）, 12（response brain + imaging Echo）, 14（findings 矛盾 PE）, 22（lab_summary）等 |

### Letter 质量：出色
- 0 P0 letter（无幻觉/编造）
- letter emotional fabrication 问题已大幅改善（ROW 2, 6, 7 的 "anxious" 编造全部修复）
- 通俗化质量高："cancer that started in the milk ducts" + "milk-producing glands" + "medication to prevent cancer from coming back" 等优秀表达

