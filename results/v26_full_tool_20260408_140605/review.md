# V26 tool Full Run Review (tool calling 版审查)

> Run: v26_full_tool_20260408_140605
> Dataset: 100 samples
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + 49 POST hooks + letter generation
> tool_calling: **true**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查完成 — 100/100 ROW 全部完成详细逐字逐句审查**
> 参照: `results/v26_full_notool_20260408_080004/review.md`（notool 版本审查，作为对比基准）
> Results 文件: `results/v26_full_tool_20260408_140605/results.txt`

### 恢复审查指南
- 待审查 ROW: **96, 97, 98, 99, 100**（这些在 notool 版本都是 0/0 ✅）
- 已完成 ROW 85-94 的详细审查：ROW 91 新增 1 P2（response 引用旧数据），其余 clean
- 每个 ROW 必须：完整读 note_text + keypoints + letter → 逐字段核对 → 写入详细临床描述 → 发现问题时更新统计
- 已发现的系统性问题：tool calling 引入了 "On treatment; response assessment not available" 模板化 response（13 个 ROW 受影响）
- results.txt 中每个 ROW 的位置可用 `grep "^RESULTS FOR ROW N$"` 查找行号

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 4 | 4% | ROW 8 response, ROW 10 response, ROW 11 response, ROW 88 response |
| **P2** | 62 | — | 见下方分布 |

### P2 分布
ROW 1×2, 2×1, 6×2, 7×2, 8×4, 9×1, 11×1, 12×1, 13×1, 14×3, 15×1, 16×1, 17×1, 19×1, 20×1, 21×1, 22×1, 23×1, 24×2, 25×1, 27×1, 31×1, 33×1, 34×1, 36×2, 38×1, 39×1, 40×1, 46×1, 48×1, 49×1, 52×1, 65×1, 70×2, 72×1, 74×1, 75×1, 76×1, 77×1, 80×1, 82×2, 83×1, 84×1, 91×1, 99×1

*注：ROW 64 原标 P2 后经重新审查移除（患者确实在 active chemo 中）；ROW 48 新增 P2（letter "anxious" from PMH but ROS negative）*

### 系统性 P2 模式（tool calling 引入）
**"On treatment" 模板化 response**: ROW 13, 23, 31, 40, 46, 49, 64, 65, 70, 72, 77, 80, 82 — 共 13 个 ROW。Tool calling 对 response_assessment 使用了 "On treatment; response assessment not available from current visit" 模板回答，但患者实际上 NOT yet on treatment（刚开处方/术后未开始系统治疗）。这是 tool calling 版本最大的系统性退化

---

## 逐 Sample 问题清单（ROW 1-100，每个 ROW 独立条目）

### ROW 1 (coral_idx 140) — 0 P1, 2 P2
- P2: imaging_plan 遗漏 Bone Scan（A/P 说 "MRI of brain and bone scan"，只提取了 Brain MRI）
- P2: therapy_plan 重复 medication_plan
- ✅ **改善 vs notool**: lab_plan 不再混入 imaging（notool P1 消除）
- ✅ **改善 vs notool**: Type_of_Cancer 正确写了 "invasive ductal carcinoma"
- ✅ Letter: "distressed and anxious" 有原文支持（"She is very scared and appears anxious"）

### ROW 2 (coral_idx 141) — 0 P1, 1 P2
- P2: Referral Others "Home health" — A/P 写的是 "Home health?" 带问号，是疑问非确定
- ✅ **改善 vs notool**: Metastasis 更完整（包含 bone），TNBC 标注更清晰，imaging_plan 捕获 scan plan
- ✅ **改善 vs notool**: notool 3 P2 → tool 1 P2
- ✅ Letter: TNBC 解释通俗，"anxious and depressed" 有原文支持

### ROW 3 (coral_idx 142) — 0 P1, 0 P2 ✅
- ✅ Stage IIA ER+/PR+/HER2- IDC, node positive, PET CT + genetic testing pending
- ✅ **改善 vs notool**: genetic_testing 正确放在 genetic_testing_plan 字段（notool P2 消除）
- ✅ Letter: 通俗准确，无幻觉，无编造 emotional 句子（POST hook 有效）

### ROW 4 (coral_idx 143) — 0 P1, 0 P2 ✅
- ✅ 75yo, ER+/PR+/HER2- IDC, s/p mastectomy, NED on letrozole, osteopenia improving
- ✅ **改善 vs notool**: therapy_plan 包含了 exercise（不再纯重复 medication_plan）
- ✅ Letter: 无 "blood tests" 幻觉，brain MRI 正确标注条件限定（"if headaches get worse"）

### ROW 5 (coral_idx 144) — 0 P1, 0 P2 ✅
- ✅ 31yo, Stage III→IV ER+/PR+/HER2- IDC, metastatic recurrence to cervical LN + brachial plexus + possible sternal bone met
- ✅ Response: mixed (cervical LN↓ but axillary LN↑ + new sternal lesion), 准确传达
- ✅ **改善 vs notool**: notool P2（follow_up 循环表述）消除
- ✅ Letter: "treatment is working well in some areas, but...new areas where the cancer has grown" 准确，无幻觉

### ROW 6 (coral_idx 145) — 0 P1, 2 P2
- P2: Patient type "New patient" 可能不正确 — zoladex 已于 06/08 由该提供者开始，说明此前已就诊过，应为 Follow up；letter 也说 "follow-up visit" 与 keypoints 矛盾
- P2: Referral-Genetics "Dr. [REDACTED] at [REDACTED]. genetics referral" 是历史转诊（04/24），Myriad 已完成且结果阴性（04/25），不是当前计划
- ✅ 34yo, ER+/PR+ HER2-（IHC 2+ FISH non-amplified）IDC right breast, s/p bilateral mastectomy 06/21, zoladex + letrozole 辅助治疗
- ✅ HER2-：IHC 2+ + FISH non-amplified = HER2 negative，正确应用判断规则
- ✅ Stage I-II（1.5cm, node-negative）合理推断
- ✅ Lab 完整（CMP + CBC + Estradiol + Vitamin D），所有值与原文一致
- ✅ **改善 vs notool**: letter 不再有 "You appear to be feeling anxious" 编造（notool P2 消除）
- ✅ **改善 vs notool**: notool 3 P2 → tool 2 P2
- ✅ Letter: 通俗准确，"cancer started in the milk ducts" 解释 IDC，无幻觉

### ROW 7 (coral_idx 146) — 0 P1, 2 P2
- P2: procedure_plan "Would recheck ***** prior to starting..." — 内容是 LVEF recheck（echo），应放 imaging_plan 而非 procedure_plan
- P2: lab_plan "Would recheck labs prior to above" — A/P 原文 "Would recheck *****" 是 LVEF 不是 labs，模型猜测了 "labs"
- ✅ MBC since 2008, originally Stage II T2N1 IDC 1998, ER-/PR-/HER2+, recurrence to supraclavicular LN + mediastinum
- ✅ Second opinion 正确标注（"CC 2nd opinion"）
- ✅ Response: "Probable mild progression" + equivocal evidence 准确传达 A/P 的不确定性
- ✅ Goals: palliative（metastatic）正确
- ✅ Medication_plan: d/c 当前方案 + 推荐新方案，正确
- ✅ **改善 vs notool**: letter 不再有 "medication level" 误述（改为 "some tests redone"，虽模糊但无错误）
- ✅ Letter: 通俗准确，"cancer may have grown a little" 恰当传达 equivocal progression

### ROW 8 (coral_idx 147) — 1 P1, 4 P2
- **P1**: response_assessment "Not yet on treatment — no response to assess" — 患者已完成 neoadjuvant TCHP（虽不完整，3 cycles）+ 手术，有明确病理响应：breast pCR（无残留侵润癌）但 3/28 LN 阳性（extranodal extension, ER-/PR-/HER2+ IHC 3+, FISH 5.7, Ki-67 75%）。应写"mixed response: breast complete response but residual nodal disease"
- P2: procedure_plan 混入 chemotherapy regimen（"adjuvant AC x 4 cycles, to be followed by T-DM1"），应只写 "port placement"
- P2: Referral Others "Social work" + follow up "I will refer to social work" — A/R 未明确提及 social work 转诊（模型从社会背景推断，但原文未陈述）
- P2: follow_up "in-person: this coming week" — A/R 写 "We will aim to speak again this coming week"，是电话/ZOOM 不是 in-person
- P2: letter 重复 social work 两句（"You are referred to social work for support" + "I will refer you to social work"）
- ✅ 29yo, Stage II-III ER-/PR-/HER2+（IHC 3+, FISH 5.7）IDC left breast, s/p incomplete neoadjuvant TCHP + lumpectomy/ALND
- ✅ Type_of_Cancer 完整（包含 IHC + FISH）
- ✅ Goals: curative 正确（无远处转移，adjuvant intent）
- ✅ Imaging_plan: echocardiogram before AC 正确（anthracycline 心脏毒性检查）
- ❌ **未改善 vs notool**: P1 response_assessment 同 notool 一样写 "Not yet on treatment"，tool calling 未修复
- ✅ Letter: 通俗，"port placed and echocardiogram to check your heart" 解释清楚

### ROW 9 (coral_idx 148) — 0 P1, 1 P2
- P2: letter "You appear to be feeling anxious and tearful" — "tearful" 有原文支持（"She is tearful today"），但 "anxious" 与 ROS 矛盾（ROS 明确说 "No depression, or anxiety or trouble sleeping"）
- ✅ 63yo, Stage II ER+/PR-/HER2-（IHC 0, FISH 0.89）IDC right breast, s/p neoadjuvant chemo + bilateral mastectomy, kidney transplant recipient
- ✅ Response: 出色 — "3.84 cm residual tumor with 5% cellularity, 1 LN macrometastasis...did not achieve pCR" 准确详细描述病理响应
- ✅ Type_of_Cancer: ER+/PR-/HER2- 正确（pathology: ER 85%, PR <1%, HER2 IHC 0, FISH non-amplified）
- ✅ Medication_plan: letrozole after radiation + fosamax，正确
- ✅ Advance care: full code 正确标注
- ✅ Referral: Radiation referral 正确
- ✅ Letter: response 描述 "cancer did not completely go away with initial treatment" 通俗准确
- ❌ **退化 vs notool**: notool 版本 0 P2，tool 版本引入 letter "anxious" P2

### ROW 10 (coral_idx 149) — 1 P1, 0 P2
- **P1**: response_assessment "No evidence of disease recurrence on imaging and exam" — 遗漏了关键病理响应信息：neoadjuvant letrozole 后手术仍有 8.8cm residual tumor + LN involvement（可能 2/20 或 20/20，A/P 文本不清），说明 neoadjuvant 响应差。且本次无影像检查，说 "on imaging" 无依据
- ✅ 66yo, Stage II left breast HR+/HER2- IDC, s/p neoadjuvant letrozole + left mastectomy + bilateral reductions + re-excision, Low risk Oncotype
- ✅ Type_of_Cancer: HR+/HER2- IDC 正确
- ✅ Goals: curative 正确
- ✅ Medication_plan: continue letrozole 正确
- ✅ Radiotherapy: left chest wall + surrounding LN 正确
- ✅ Imaging_plan: DEXA 正确（骨密度检查，letrozole 相关）
- ✅ Advance care: full code 正确
- ✅ **改善 vs notool**: response 从 "does not provide specific evidence" → "No evidence of recurrence" 略有改善（at least 有明确判断），但仍是 P1
- ✅ Letter: 通俗准确，"DEXA scan, which is a type of imaging test" 解释清楚

### ROW 11 (coral_idx 150) — 1 P1, 1 P2
- **P1**: response_assessment "PET/CT showed increased metastatic activity and size of left mandibular mass, indicating disease progression" — 这是 10/10/12 的 PET 结果（换药前），当前 A/P 说 "Exam stable"。应描述当前状态：体检稳定但右腿麻木恶化，需 PETCT 重新分期
- P2: letter "cancer has grown in your jaw" — jaw 已经 XRT 治疗完成（"S/p xrt to jaw improved pain with residual numbness"），信中把已治疗的过去进展说成当前情况
- ✅ 68yo, Stage IIIC→IV IDC left breast, ER+（inferred from letrozole), bone mets（spine, ribs, sternum, pelvis, mandible）
- ✅ Current_meds: Fulvestrant + Denosumab 正确
- ✅ Goals: palliative 正确
- ✅ Lab: 完整 CBC + CMP，所有值与原文一致
- ✅ Imaging_plan: PETCT to toes 正确
- ✅ Medication_plan: Faslodex + Denosumab + Mycelex for thrush 正确
- ❌ **未改善 vs notool**: P1 response_assessment 同 notool 一样混淆旧 PET 与当前状态
- ✅ Letter: 除 jaw 问题外，"cancer has spread to...your bones" + "scan to look at your leg and feet" 通俗准确

### ROW 12 (coral_idx 151) — 0 P1, 1 P2
- P2: Metastasis 列出 "lung" 但 A/P 说 "# Lung mets. Now [REDACTED]"（resolved/improved），且最新 CT 说 "No suspicious pulmonary nodules"
- ✅ 51yo, de novo Stage IV ER+/PR+/HER2+ IDC to breast, lung, nodes, brain, bone. On herceptin + [redacted] + letrozole since 08/18/17
- ✅ **改善 vs notool**: Advance care 正确捕获了 DNR/DNI + POLST + goals of care（notool P1 消除！Tool calling 从 Problem List 搜到了关键信息）
- ✅ Response: "SD and possible [REDACTED] status...no evidence of PD" + "2 new foci" brain mets — 混合响应准确传达
- ✅ Current_meds: herceptin + letrozole 正确
- ✅ Goals: palliative 正确
- ✅ Imaging_plan: CT CAP q4mo + bone scan + MRI brain q4mo 完整
- ✅ Radiotherapy: "await GK / Rad Onc input, and potential plan for repeat GK" 正确
- ✅ Letter: "stable in some areas but has new spots in the brain" 准确传达混合响应，"treatment is aimed at controlling the cancer and managing symptoms" 正确解释 palliative intent
- ✅ Letter: "Your advance care status is noted as DNR/DNI, and you want to spend time with family" — 来自 Problem List, 正确且对患者重要
- ✅ **改善 vs notool**: notool 1 P1 + 2 P2 → tool 0 P1 + 1 P2（净改善 2 级）

### ROW 13 (coral_idx 152) — 0 P1, 1 P2
- P2: response_assessment "On treatment; response assessment not available" — 患者尚未开始任何治疗（tamoxifen 在考虑中，radiation 待 consult），应为 "Not on active treatment; post-surgical"
- ✅ 41yo, ER+ nuclear grade 2 DCIS left breast, s/p partial mastectomy, DCIS score 60, Invitae 46 gene panel negative, premenopausal
- ✅ Type: "ER+ DCIS, HER2: not tested" 正确（DCIS 通常不做 HER2）
- ✅ Goals: "risk reduction" 非常准确（DCIS 不是 curative/palliative，是 risk reduction）
- ✅ Medication_plan: 详细描述 tamoxifen 考虑及副作用列表
- ✅ Referral: Radiation oncology consult 正确
- ✅ **改善 vs notool**: findings 不再左右乳混淆（notool 把右乳 14mm mass 写成左乳，tool 版本正确写 "Right breast"）
- ✅ **改善 vs notool**: notool 2 P2 → tool 1 P2
- ✅ Letter: 出色 — "cancer is in the milk ducts and has not spread beyond them" 解释 DCIS 通俗准确，"chances of staying healthy are very good" 传达良好预后

### ROW 14 (coral_idx 153) — 0 P1, 3 P2
- P2: current_meds 空 — 患者虽停了 palbociclib/fulvestrant，但正在自行使用 Mexico 低剂量化疗（doxorubicin 10mg + gemcitabine 200mg + docetaxel 20mg weekly + pamidronate weekly）
- P2: Next clinic visit "2 months" — 但 A/P 第一段说 "F/u 3 months"，模型把 Mexico 的 "return in 2 months" 与 UCSF follow-up 混淆了
- P2: letter 用了第三人称 "The patient reports that she can move better" — 患者信应该用第二人称 "You can move better"
- ✅ 58yo, de novo Stage IV ER+/PR+/HER2-（FISH negative）breast cancer to bone, liver, nodes
- ✅ S/p extensive spine surgery + XRT T1-T10, ECOG 2
- ✅ Lab 完整（CMP + CBC + CA 27-29 = 48, trending down from 193）
- ✅ Recent_changes 完整记录了停药 + Mexico 治疗方案
- ✅ Imaging_plan: CT CAP + Total Spine MRI May 2019 正确
- ✅ Lab_plan: labs every 2 weeks 正确（from Dr.'s note）
- ✅ Referral: PT 正确
- ❌ **退化 vs notool**: notool 2 P2 → tool 3 P2（增加了 letter 第三人称 + follow-up 混淆）

### ROW 15 (coral_idx 154) — 0 P1, 1 P2
- P2: genetic_testing_plan "biomarker testing" — A/P 说 "reviewed pathology, biomarker testing"，是回顾已完成的检测，不是新计划（ER/PR/HER2 已全部完成）
- ✅ 46yo, newly diagnosed left breast mixed IDC/ILC, ER+（>95%）/PR+（80-90%）/HER2+（IHC 2+, FISH ratio 2.0, copy# 4.7 — borderline but ASCO/CAP positive）
- ✅ Second opinion: "yes" 正确
- ✅ Stage: "Clin st I/II" 直接来自 A/P
- ✅ Goals: curative 正确（early stage, fit patient）
- ✅ Procedure_plan: "Plan for breast surgery" 干净（不再混入 Rx recommendations）
- ✅ Medication_plan: TCHP regimen as option 正确
- ✅ **改善 vs notool**: procedure_plan 不再混入 Rx（notool P2 消除）；notool 2 P2 → tool 1 P2
- ✅ Letter: 出色 — "mix of two types: one that started in the milk-producing glands and one that started in the milk ducts" 通俗解释 mixed IDC/ILC 非常准确
- ✅ Letter: "If you choose to start treatment before surgery..." 正确传达 neoadjuvant vs surgery-first 选项

### ROW 16 (coral_idx 155) — 0 P1, 1 P2
- P2: letter "Your doctor noted that you appear anxious and nervous" — 虽有原文支持（ROS: "The patient is nervous/anxious"），但在患者信中提及情绪状态不太适当
- ✅ 54yo postmenopausal, Stage I HR+/HER2-（0+）IDC right breast, 0.3cm grade 1, margins negative, Ki-67 <10%
- ✅ S/p lumpectomy, no DCIS, extremely dense breasts
- ✅ Goals: curative 正确
- ✅ Medication_plan: AI x 5 years after radiation + calcium + vitamin D 正确
- ✅ Radiotherapy: radiation oncology evaluation 正确
- ✅ Imaging_plan: DEXA + surveillance breast MRI consideration 正确
- ✅ Lab_plan: estradiol level 正确
- ✅ Genetic_testing: referral for genetic testing（father colon cancer）正确
- ✅ Letter: 整体出色 — "aromatase inhibitor...helps prevent the cancer from coming back" + "dense breast tissue" 通俗准确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（letter 增加了 "anxious" 句子）

### ROW 17 (coral_idx 156) — 0 P1, 1 P2
- P2: procedure_plan "check labs including hormones" — labs 不是 procedure，应在 lab_plan（已正确放了）；procedure_plan 应为 "No procedures planned"
- ✅ 53yo, ER+/PR+（>95%）/HER2-（IHC 0, FISH 1.1x）IDC left breast, 0.8cm grade 2, margins negative, LN 0/5, no DCIS
- ✅ S/p lumpectomy + SLN biopsy, chest CT negative, menopausal status unclear（s/p hysterectomy）
- ✅ Goals: curative 正确
- ✅ Medication_plan: hormonal therapy x 5 years, tamoxifen or AI based on menopausal status 正确
- ✅ Radiotherapy: breast RT scheduled tomorrow 正确
- ✅ Imaging_plan: baseline DXA 正确
- ✅ Lab_plan: check labs including hormones 正确
- ✅ Genetic_testing: genetics referral（sister ovarian ca, paternal aunt breast ca）正确
- ✅ Referral: Nutrition + Genetics + Specialty（RT consult）all correct
- ✅ Letter: 出色 — "adjuvant hormonal therapy, which is given after surgery to prevent the cancer from coming back" + "The type of medication will depend on whether you are in menopause or not" 通俗解释到位
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（procedure_plan 混入 labs）

### ROW 18 (coral_idx 157) — 0 P1, 0 P2 ✅
- ✅ 65yo, ER+（~100%）/PR+（95%）/HER2-（IHC 1+）IDC left breast, 0.8cm grade 1, arising in association with encapsulated papillary carcinoma, Ki-67 5%
- ✅ Type_of_Cancer 非常详细：包含 IDC + encapsulated papillary carcinoma 关联
- ✅ Stage: pT1b, pNX 正确（ITC in 1/3 LN = pN0(i+), formal staging redacted）
- ✅ Genetic_testing: "UCSF Cancer Risk will reach out to pt today" 准确且有行动性（notool 版本也正确）
- ✅ Medication_plan: adjuvant endocrine therapy 5-10 years 正确
- ✅ Patient 拒绝 chemo → 不做 molecular profiling，正确记录
- ✅ Letter: 通俗准确，"medication for 5 to 10 years to prevent the cancer from coming back" + "UCSF Cancer Risk will reach out to you today"

### ROW 19 (coral_idx 158) — 0 P1, 1 P2
- P2: Referral-Specialty "Radiation oncology consult" — A/P 未提及 radiation referral（讨论的是 neoadjuvant chemotherapy vs surgery），letter 也写了 "referred to radiation oncology" 是错误的
- ✅ 70yo, ER 90%/PR ~15%/HER2 3+（FISH positive, ratio 9.5, 异质性）IDC grade 3 left breast
- ✅ 4-5cm on PE, up to 6cm by imaging, axillary LN FNA positive
- ✅ PET/CT: no distant mets, pulmonary nodules up to 5mm (follow-up per protocol)
- ✅ Goals: curative 正确（no distant mets, neoadjuvant intent）
- ✅ Medication_plan: TCHP + GCSF 正确（避免 anthracycline given CAD history）
- ✅ Procedure_plan: "Port Placement" 正确
- ✅ Imaging_plan: echocardiogram 正确（trastuzumab 前需检查心脏）
- ✅ Findings: 非常详细，包含所有影像和病理数据
- ✅ Letter: "HER2 positive, which means it has a protein that can make the cancer grow faster" + "treatment is given before surgery to shrink the cancer" 通俗解释出色
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（错误 radiation referral）

### ROW 20 (coral_idx 159) — 0 P1, 1 P2
- P2: procedure_plan "Abdomen, Pelvis, Rad Onc referral, Xgeva - needs dental evaluation first" — 混入了影像（CT CAP 片段）、referral（Rad Onc）、药物（Xgeva/denosumab），应只包含实际 procedure 如 dental evaluation
- ✅ 75yo, ER+/PR+/HER2-（original + met biopsy confirmed）IDC left breast 2009, now metastatic recurrence to bone + lymph nodes
- ✅ Original: 0.9cm grade II, 0/2 SLN, 5 years tamoxifen. Met biopsy: ER 80%, PR 50%, HER2 FISH negative（ratio 1.05）
- ✅ Medication_plan: letrozole + palbociclib + denosumab 完整正确
- ✅ Imaging_plan: MRI Total Spine + CT CAP + obtain outside PET/CT 正确
- ✅ Lab_plan: tumor markers + monthly blood work on palbociclib 正确
- ✅ Genetic_testing: Foundation One（or ***** 360）正确
- ✅ Goals: palliative 正确
- ✅ Follow-up: ~1 month 正确
- = **同 notool**: procedure_plan 同样混入内容（notool 1 P2 → tool 1 P2）
- ✅ Letter: 出色 — "denosumab after you get your teeth checked" 通俗解释 dental clearance，"doing more tests to understand your cancer better" 解释 Foundation One

### ROW 21 (coral_idx 160) — 0 P1, 1 P2
- P2: medication_plan 只写了 bisphosphonate 骨保护，遗漏了主要推荐的 aromatase inhibitor（Arimidex）x 5 years after XRT。AI 在 therapy_plan 中有但 medication_plan 应该包含
- ✅ 70yo postmenopausal, ER+（98%）/PR+（90%）intermediate grade DCIS right breast, 5cm span, comedo necrosis, clear margins, s/p partial mastectomy
- ✅ Type: DCIS, HER2 status unclear（DCIS 常不测 HER2）正确
- ✅ Goals: "risk reduction" 非常准确（DCIS recurrence prevention, not curative/palliative）
- ✅ Osteopenia 已存在，AI 需配合 bisphosphonate — 正确讨论
- ✅ Strong family hx（sisters + aunts + niece with breast ca），Invitae 52 genes negative
- ✅ Referral: Rad Onc 正确
- ✅ Follow-up: 3-4 months after XRT 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（medication_plan 遗漏 AI）
- ✅ Letter: "sensitive to estrogen and progesterone" + "help protect your bones if you decide to take a medicine that lowers estrogen" 通俗解释出色

### ROW 22 (coral_idx 161) — 0 P1, 1 P2
- P2: radiotherapy_plan "XRT to L4 and T10 in June 2020" — 这是过去的 radiation（2020 年 6 月），不是当前计划。A/P 未推荐新 radiation
- ✅ 72yo, metastatic ER+/PR+/HER2- IDC right breast, originally Stage II（2000), met recurrence 2020: chest wall + bone + infraclavicular + IM nodes
- ✅ History: left DCIS 1994 + right IDC 2000 + met relapse 2020, complex treatment history
- ✅ Second opinion: "yes" 正确（"She is here for a second opinion"）
- ✅ Current_meds: anastrozole + denosumab 正确（abemaciclib stopped due to pneumonitis）
- ✅ Response: PET scans good response 正确
- ✅ Medication_plan: 详细且正确 — continue arimidex if stable, faslodex + [redacted] if progression, future options afinitor/xeloda/trial
- ✅ Imaging_plan: PET CT now 正确
- ✅ Genetic_testing: mutation testing for treatment guidance 合理推断（PIK3CA for alpelisib）
- ✅ **改善 vs notool**: genetic_testing 不再混入 medication plan 内容（notool P2 消除）；虽有新 P2（radiotherapy past），总数相同

### ROW 23 (coral_idx 162) — 0 P1, 1 P2
- P2: response_assessment "On treatment; response assessment not available" — 但 recent_changes 自己写了 "Not yet on treatment"，两个字段矛盾；letrozole 尚未开始
- ✅ 63yo, multifocal: 1cm grade 2 ILC（ER+/PR-/HER2-）+ 0.7cm grade 2 IDC/ILC（ER+/PR+/HER2-），intermediate/high grade DCIS, 0/1 SLN
- ✅ S/p left mastectomy + SLN, MYRIAD negative（BRCA2 VUS）
- ✅ Goals: curative 正确（low risk, no chemo needed）
- ✅ Medication_plan: letrozole 5+ years adjuvant, comprehensive description of AI vs tamoxifen
- ✅ Imaging: baseline DEXA 正确
- ✅ Follow-up: 3 months 正确
- ✅ Findings: 两个肿瘤的完整病理详情，非常详细
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "On treatment" 矛盾）

### ROW 24 (coral_idx 163) — 0 P1, 2 P2
- P2: Metastasis/Distant Met 都写 "Not sure" — PET/CT 明确说 "No definite sites of hypermetabolic metastatic disease"，应为 "No"（SLN 微转移 0.4mm 是 regional，不算 distant）
- P2: procedure_plan + imaging_plan 都混入了 genetic testing 内容（Oncotype/molecular profiling），应只在 genetic_testing_plan 中
- ✅ 56yo, Grade II micropapillary mucinous carcinoma right breast, ER+（>95%）/PR+（80%）/HER2 equivocal IHC 2+, FISH negative, Ki-67 5%
- ✅ S/p partial mastectomy, 2/4 SLN micro met (0.4mm), negative margins, hepatic lesions benign
- ✅ Goals: curative 正确
- ✅ Genetic_testing: Oncotype/MP for chemo benefit evaluation 正确
- ✅ Radiation: scheduled 12/07/18 with rad onc 正确
- ✅ Referral: PT + radiation oncology 正确
- = **同 notool**: 2 P2（Metastasis "Not sure" + field misplacement）

### ROW 25 (coral_idx 164) — 0 P1, 1 P2
- P2: response_assessment 引用了 12/11/2010 PET/CT 数据（4 个月前），但 A/P 当前说 "Exam supraclavicular area appears to be breaking up"（临床改善）+ "Scan in 3 weeks"（正式重新分期待查）。应描述当前临床响应而非旧 PET
- ✅ 45yo, metastatic ER+/PR+/HER2- IDC, extensive treatment history（right + left breast cancer, brain met resection, WBRT）
- ✅ Type: 捕获了 receptor 变化（original ER+/PR+/HER2- → met biopsy ER-/PR-/HER2-）
- ✅ Metastasis: brain, liver, bones, LN, breast skin — 完整
- ✅ Current_meds: capecitabine + ixabepilone 正确
- ✅ Goals: palliative 正确
- ✅ Imaging: "Scan in 3 weeks" 正确
- ✅ Lab 完整（ALP 308↑, AST 55↑, Albumin 3.1↓, Hgb 11.2↓ — all consistent with advanced disease）
- = **同 notool**: P2 response 引用旧 PET（同样的问题）

### ROW 26 (coral_idx 165) — 0 P1, 0 P2 ✅
- ✅ 56yo, Stage IB TNBC（ER-/PR-/HER2-, Ki-67 75%）right breast, node negative
- ✅ Plan: surgery first（bilateral reductions + lumpectomy + SLN + port），then chemo + radiation
- ✅ Genetics consult + social work referral + advance care full code — all correct
- ✅ Letter: "triple negative, which means it does not respond to hormones or a protein called HER2" 出色解释

### ROW 27 (coral_idx 166) — 0 P1, 1 P2
- P2: procedure_plan "obtain UA, obtain CBC with platelets" — 这些是 lab tests，应在 lab_plan 中。lab_plan 只有 CBC 未包含 UA
- ✅ 41yo, metastatic ER+/PR+/HER2- IDC to bone, on letrozole + zoladex + zoledronic acid
- ✅ Response: "PET-CT shows stable to slightly decreased metabolic activity...No new metastases" 准确
- ✅ Goals: palliative 正确
- ✅ Current_meds: letrozole, zoledronic acid, goserelin 正确
- ✅ Imaging_plan: consider MRI spine for new back pain 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（procedure 混入 labs）

### ROW 28 (coral_idx 167) — 0 P1, 0 P2 ✅
- ✅ 60yo postmenopausal, Stage I HR+/HER2-（IHC 0, FISH 1.1）grade 1 IDC right breast, 0.9cm, 0/3 LN, margins negative
- ✅ Goals: curative, AI x 5 years + radiation planned, DEXA ordered

### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅
- ✅ 59yo, multifocal grade 2 IDC ER+/PR+/HER2-, micropapillary features, 1.6cm + 0.6cm, 1/1 SLN micromet 0.5mm
- ✅ Oncotype Low Risk → no chemo. Start letrozole, re-excision September, RT locally
- ✅ Stage pT1c(m)N1(sn)M0 正确

### ROW 30 (coral_idx 169) — 0 P1, 0 P2 ✅
- ✅ 64yo, clinical stage II-III ER-/PR-/HER2+（IHC 3+, FISH 8.9）IDC right breast, 9cm mass, history of untreated DCIS since 2007
- ✅ Neoadjuvant TCHP or THP→AC detailed plan, TTE + port needed

### ROW 31 (coral_idx 170) — 0 P1, 1 P2
- P2: response_assessment "On treatment; response assessment not available" — 但 April 2021 PET/CT 明确显示 progression（liver + bone enlarging），A/P 专门 review 了 PET 结果。应写 "disease progression on prior therapy"。且患者当前 OFF 治疗（scheduled to start Doxil 07/01）
- ✅ 64yo, de novo metastatic ER+/PR+/HER2- breast cancer to bone + liver + ?brain
- ✅ Multiple prior lines: letrozole → ibrance → fulvestrant → capecitabine → now starting Doxil
- ✅ Imaging_plan: Brain MRI + PET after 3 cycles Doxil + MRI pelvis for hip pain
- ✅ Advance care: Full code 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2

### ROW 32 (coral_idx 171) — 0 P1, 0 P2 ✅
- ✅ 82yo, Stage IIA→IV ER+/PR-/HER2+ pleomorphic lobular cancer, metastatic to LN（neck, chest, abdomen, pelvis）+ left adnexal
- ✅ Response: PET June 2019 = CR ("No abnormal FDG uptake to suggest active metastatic disease") 正确
- ✅ On herceptin + exemestane (pertuzumab recently stopped due to diarrhea)
- ✅ Advance care: Full code + living will 正确
- ✅ Referral: Nutrition + Exercise Counseling 正确

### ROW 33 (coral_idx 172) — 0 P1, 1 P2
- P2: letter "is now considered stage IIIA" — 对 NED 随访患者而言令人困惑。原始分期从 clinical IIB upstage 到 pathologic IIIA（术后 ALND），不是疾病进展。Letter 应强调 "no evidence of recurrence" 而非让患者以为恶化了
- ✅ 63yo, ER+/PR+/HER2- ILC left breast, s/p bilateral mastectomy + TC x 6 + XRT, letrozole since 2011（>5 years, patient prefers to continue）
- ✅ Response: "No evidence of recurrence" 正确
- ✅ Goals: curative 正确
- ✅ Current_meds: letrozole 正确
- ✅ Follow-up: 6 months 正确
- ✅ Imaging: consider MRI brain if headaches continue 正确
- = **同 notool**: P2 letter stage confusion（same issue）

### ROW 34 (coral_idx 173) — 0 P1, 1 P2
- P2: lab_plan "No labs planned" — 但 A/P 明确说 "check labs"
- ✅ 71yo, Stage III ER+/PR-/HER2- IDC left breast, 2nd local recurrence (1.7cm grade 3), s/p bilateral mastectomies
- ✅ Switch from anastrozole to tamoxifen 20mg 正确（AI-resistant concern）
- ✅ Radiotherapy: CW RT referral 正确（patient now accepts after previously declining）
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2

### ROW 35 (coral_idx 174) — 0 P1, 0 P2 ✅
- ✅ 40yo, pT1cN0 ILC right breast, 1.2cm grade 2, 0/4 LN, on adjuvant tamoxifen/anastrozole
- ✅ Response: NED, mammogram negative
- ✅ Follow-up: 6 months routine

### ROW 36 (coral_idx 175) — 0 P1, 2 P2
- P2: procedure_plan 混入了 Abraxane（medication）+ Doppler（imaging）+ RT referral
- P2: current_meds 遗漏 tamoxifen（01/29/21 started，但只列了 Abraxane + zoladex）
- ✅ 27yo premenopausal, pT3N0 ER+/HER2- grade III mixed ductal/mucinous carcinoma right breast, 8.4cm, 0/4 LN
- ✅ On adjuvant abraxane cycle 8/12 + zoladex, radiation referral next week
- ✅ Labs comprehensive, new concern: right arm swelling → doppler for DVT
- ❌ **退化 vs notool**: notool 0 P2 → tool 2 P2

### ROW 37 (coral_idx 176) — 0 P1, 0 P2 ✅
- ✅ 61yo postmenopausal, Stage IIA TNBC（ER-/PR-/HER2-, grade 3）IDC left breast, 2.3cm, node negative（ITC in SLN, pN0(i+)）, s/p bilateral mastectomies July 2020
- ✅ Type: "ER-/PR-/HER2- triple negative IDC" 正确
- ✅ Response: "Not yet on treatment" 正确（尚未开始化疗）
- ✅ Medication_plan: dd AC → Taxol 正确
- ✅ A/P 明确 no radiation, no hormone blockade — 正确不列出
- ✅ Goals: curative 正确
- ✅ Advance care: Full code 正确
- ✅ Letter: "triple negative...cancer did not respond to certain hormones or proteins" 通俗解释准确，无幻觉

### ROW 38 (coral_idx 177) — 0 P1, 1 P2
- P2: letter "We understand you are feeling anxious and depressed" — PMH 有 Anxiety（2020）和 Depression（2021）诊断，但当前 ROS 明确否认 "No depression, or anxiety or trouble sleeping"
- ✅ 43yo, BRCA1 mutation, Stage IIB ER-/PR+（weak 15%）/HER2- IDC left breast, 6.8cm → 8x5cm（post-neoadjuvant regrowth）
- ✅ S/p neoadjuvant AC x 4 + taxol x 5 weeks（stopped due to toxicity），declined further IV therapy
- ✅ **改善 vs notool**: response "Her tumor is enlarging" 正确捕获了肿瘤再增长（notool P2 "not responding but not on treatment" 消除）
- ✅ Medication_plan: olaparib + xeloda adjuvant 正确
- ✅ Procedure: bilateral mastectomy January 31 正确
- ✅ Referral: Gyn Onc + social work 正确
- ✅ Letter: "Your tumor is growing again, which means the treatment you had before is not working as well as we hoped" 通俗准确
- = **同 notool**: 1 P2（不同的 P2 — notool 是 response 问题，tool 是 letter 情绪问题）

### ROW 39 (coral_idx 178) — 0 P1, 1 P2
- P2: Type_of_Cancer 写了 "ER+ (inferred from goserelin)" — 但 goserelin 是用于化疗期间 fertility preservation（A/P 明确说 "improved fertility preservation with this approach"），不是用于 hormonal cancer treatment。癌症是 TNBC（"ER/PR/***** negative"），field 自己也写了 "ER/PR/[REDACTED] negative"，矛盾
- ✅ 27yo, grade 3 TNBC left breast, T2N1, 6x6cm mass, 1cm axillary node FNA+
- ✅ CT CAP/brain MRI/bone scan 均 negative → no distant mets
- ✅ Medication_plan: paclitaxel x 12 wks → AC x 4 + goserelin for fertility 正确
- ✅ Procedure: screening biopsies + port placement 正确
- ✅ Imaging: echo + breast MRI 正确
- ✅ Lab: ISPY studies 正确
- ✅ Letter: 通俗准确，无幻觉
- = **同 notool**: 1 P2（same goserelin→ER+ 推断错误）

### ROW 40 (coral_idx 179) — 0 P1, 1 P2
- P2: response_assessment "On treatment" — letrozole 刚开处方（"Rx for letrozole given"），尚未开始服用
- ✅ 62yo with MS, Stage 2 ER 95%/PR 5%/HER2-（FISH 1.2）G1 IDC right breast, 2.3cm, 1 SLN+（micromet by direct extension）
- ✅ Medication: letrozole adjuvant + Prolia for osteoporosis, no chemo（patient declines, low benefit）
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2

### ROW 41 (coral_idx 180) — 0 P1, 0 P2 ✅
- ✅ 32yo premenopausal, ATM mutation carrier, 3cm grade 3 IDC left breast, ER+ 90%/PR weakly+ 1%/HER2 1+（IHC; core biopsy FISH negative）, Ki-67 30%, LVI+, 1/3 SLN micromet (0.022cm)
- ✅ S/p bilateral mastectomy 04/11/18, Oncotype High Risk, LVEF 79%
- ✅ Medication_plan: Taxol x 12 wks → AC, then ovarian suppression + AI, eligible for ribociclib trial. 非常完整
- ✅ Procedure: port placement 正确
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确
- ✅ Lab: Ferritin + CBC + pregnancy test 完整
- ✅ Letter: "removing both breasts and checking the lymph nodes under your arm" + "medication to suppress your ovaries" 通俗准确

### ROW 42 (coral_idx 181) — 0 P1, 0 P2 ✅
- ✅ 41yo, multifocal G1 IDC right breast, s/p lumpectomy + SLN + XRT, starting tamoxifen
- ✅ Response: "Not mentioned" 正确（post-XRT, starting hormonal therapy, no tumor to assess）

### ROW 43 (coral_idx 182) — 0 P1, 0 P2 ✅
- ✅ 38yo, 2nd primary Stage I TNBC（ER-/PR-/HER2- FISH negative, grade 3, Ki-67 >80%）left breast, 1.3cm, 0/2 SLN negative
- ✅ History: 1st primary TNBC at age 27 s/p lumpectomy + dd AC→T + XRT（2010），BRCA negative
- ✅ S/p bilateral mastectomies 02/22/2021，complicated by severe anemia（Hgb 5.4→transfusion）
- ✅ Medication_plan: taxol + carboplatin x 4 cycles adjuvant 正确
- ✅ Response: "Not yet on treatment" 正确
- ✅ Lab: comprehensive CMP + CBC + thyroid + pregnancy test，all values match
- ✅ Supportive_meds: granisetron + compazine + senna 正确
- ✅ Lab_plan: blood draw prior to cycle 正确

### ROW 44 (coral_idx 183) — 0 P1, 0 P2 ✅
- ✅ Response: detailed pCR assessment — "cancer did not achieve pCR, with 1 cm residual grade 2 IDC" 出色
- ✅ **改善 vs notool**: response 有详细的病理响应数据

### ROW 45 (coral_idx 184) — 0 P1, 0 P2 ✅
- ✅ Response: "cancer progressed on neoadjuvant taxol. Post-surgery, imaging and biopsy showed metastatic breast cancer to the lung" 非常准确详细
- ✅ **改善 vs notool**: 准确捕获了 progression + metastatic recurrence

### ROW 46 (coral_idx 185) — 0 P1, 1 P2
- P2: response_assessment 结尾写 "No evidence of response to treatment is mentioned in the note" — 但医生 A/P 明确说 "She had a good response to chemotherapy based on where she started"，MRI 也显示 "interval decrease in volume, extent and avidity of enhancement"。field 自己矛盾
- ✅ 48yo postmenopausal（s/p BSO），ER+ 95%/PR- 0%/HER2-（IHC 1+）IDC right breast，s/p neoadjuvant taxol + [redacted]
- ✅ Post-op path: ypT2, 3.5cm residual（cellularity 10-20%），2/2 SLN+ with extranodal extension, POSITIVE margins → needs re-excision
- ✅ Complex PMH: sarcoidosis（confirmed by mediastinal node FNA — granulomatous, NOT mets）, renal artery aneurysm, neuropathy, anemia
- ✅ Medication_plan: letrozole started today + abemaciclib after XRT 正确
- ✅ Procedure: re-excision + possible ALND 正确
- ✅ Imaging: DEXA + MRA abdomen in 1 year 正确
- ✅ Invitae: pathogenic variant in [redacted], VUS in WRN — captured in history
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response 自相矛盾）

### ROW 47 (coral_idx 186) — 0 P1, 0 P2 ✅
- ✅ 41yo premenopausal, ER+/PR+ intermediate grade DCIS left breast, 3cm, comedo necrosis, s/p excision + re-excision（margins clear on re-excision）
- ✅ Second opinion: "yes" 正确
- ✅ Goals: "risk reduction" 完美（DCIS = risk reduction, not curative/palliative）
- ✅ Radiotherapy: detailed discussion of XRT to reduce 30% → 3-4% recurrence risk
- ✅ Medication_plan: tamoxifen prophylaxis consideration 正确
- ✅ Response: "Not yet on treatment" 正确
- ✅ Family hx: brother synovial cell sarcoma + grandmothers breast ca → BRCA testing done（results pending）
### ROW 48 (coral_idx 187) — 0 P1, 1 P2
- P2: letter "You appear to be feeling anxious" — PMH 有 Anxiety 诊断（05/21/2017），但当前 ROS "Psychological ROS: negative"，当前就诊无焦虑表现。Letter 不应从 PMH 推断当前情绪
- ✅ 46yo premenopausal, left breast DCIS（at least），intermediate nuclear grade, ER+（90%）/PR+（60%），HER2 not tested
- ✅ 3.8cm complex mass on mammogram（BI-RADS 5），MRI 23mm mass + 31mm NME. Two core biopsies = DCIS only, invasion not established → excision recommended
- ✅ Type: "at least DCIS" 准确反映 uncertainty about invasion
- ✅ Response: "Not yet on treatment" 正确
- ✅ Plan: conditional — DCIS only → XRT +/- endocrine; invasive → LN bx + adj Rx
- ✅ Goals: curative 正确
- ✅ Letter: "If the cancer is only DCIS...If the cancer is more advanced..." 条件性计划解释通俗准确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（letter 从 PMH 推断 anxious）

### ROW 49 (coral_idx 188) — 0 P1, 1 P2
- P2: response_assessment "On treatment" — 但患者尚未开始任何治疗（手术 scheduled 01/06/17，tamoxifen 是术后计划），应为 "Not yet on treatment"
- ✅ 50yo, ER+/PR+/HER2- IDC left breast, biopsy-proven LN+, Oncotype low risk（11）
- ✅ Advance care: surrogate decision maker 正确捕获
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2

### ROW 50 (coral_idx 189) — 0 P1, 0 P2 ✅
- ✅ Response: "Imaging from December 2021 shows metastatic disease under good control" 正确

### ROW 51 (coral_idx 190) — 0 P1, 0 P2 ✅
- ✅ 这是一个 **chemo teaching note**（RN 教育访问），不是 oncology consult。整个 note 是 Gemzar/[redacted]/Carboplatin 副作用教育 + 抗呕吐药物回顾
- ✅ Type/Stage/Response: "Not mentioned" 全部正确（教育访问不讨论诊断）
- ✅ Current_meds: Gemzar + Carboplatin 从 regimen 提取正确
- ✅ Goals: palliative 推断合理（三药化疗方案）
- ✅ 所有 plan 字段合理为空

### ROW 52 (coral_idx 191) — 0 P1, 1 P2
- P2: procedure_plan 混入 fertility referral + Zoladex prior auth（应分别在 Referral 和 medication_plan 中）
- ✅ 35yo premenopausal, ER+/PR+/HER2- IDC left breast, 1.7cm, micromet in SLN
- = **同 notool**: procedure_plan mixed content

### ROW 53 (coral_idx 192) — 0 P1, 0 P2 ✅
- ✅ 59yo, Stage II/III ER+（>95%）/PR+（30%）/HER2+（heterogeneous IHC 2+/3+, FISH 4.9X）IDC with neuroendocrine differentiation left breast, 4.5cm grade 3, 2/SLN positive（6mm met）
- ✅ Type: 非常详细，包含 neuroendocrine differentiation + heterogeneous HER2
- ✅ Medication_plan: 出色 — AC/THP vs TCHP + trastuzumab/pertuzumab x 1 year + neratinib year 2 + Arimidex x 10 years + bone agents
- ✅ Radiotherapy: adjuvant breast XRT after chemo 正确
- ✅ Genetic_testing: counseling referral offered（family hx: aunt breast ca, cousin breast ca）
- ✅ Response: "Not yet on treatment" 正确
### ROW 54 (coral_idx 193) — 0 P1, 0 P2 ✅
- ✅ 39yo premenopausal, BRCA2 mutation, oligometastatic ER+（90%）/PR+（10%）/HER2-（FISH 1.3）IDC left breast with T6 bone met
- ✅ S/p neoadjuvant AC/T → SBRT T6 → bilateral mastectomy + left ALND（8.2cm residual G1, ~10% cellularity, 3/24 SLN+）
- ✅ Currently on leuprolide + letrozole + zoledronic acid q3mo. Palbociclib planned after radiation
- ✅ Response: "cancer is currently stable on treatment" 正确（PET stable, exam NED）
- ✅ Medication_plan: complete plan including OFS + AI + CDK4/6i + bone agents + calcium/vitamin D
- ✅ Imaging: PET/CT in 3-4 months + DEXA ordered
- ✅ Side effects: joint pain（letrozole），neuropathy, hot flashes → starting acupuncture, weaning oxycodone

### ROW 55 (coral_idx 194) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, Stage I T1N0M0 ER+（>95%）/PR+（>95%）/HER2-（IHC 1+, FISH 1.4X）IDC left breast, 5mm grade 2, 0/2 SLN, focal DCIS
- ✅ S/p lumpectomy + SLN + XRT（nearly finished），30-py smoker quit 2015
- ✅ Medication_plan: Arimidex 1mg x 5 years（patient undecided）
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确（low risk, ~10% recurrence risk without adjuvant therapy）

### ROW 56 (coral_idx 195) — 0 P1, 0 P2 ✅
- ✅ 56yo postmenopausal, Stage IB TNBC（ER-/PR-/HER2-, Ki-67 90%）left breast, aggressive tumor
- ✅ Televisit, desires bilateral mastectomy no reconstruction
- ✅ Medication_plan: neoadjuvant dd AC → weekly taxol 正确（urgent — must start within 1 month）
- ✅ Genetics consult recommended（mother + half-sister with breast cancer）
- ✅ Imaging: MRI bilateral breasts recommended
- ✅ Goals: curative, advance care full code 正确

### ROW 57 (coral_idx 196) — 0 P1, 0 P2 ✅
- ✅ Response: 出色 — "did not achieve pCR, with 3.7 cm residual tumor with 0/6 nodes positive" 详细准确
- ✅ **改善 vs notool**: notool 1 P2（procedure mixed genetic counseling）→ tool 0 P2

### ROW 58 (coral_idx 197) — 0 P1, 0 P2 ✅
- ✅ 60yo postmenopausal, Stage IIb T2N1M0 Grade 2 ER+/PR+/HER2- IDC left breast, LVI+, 14 LN+ with extracapsular extension
- ✅ S/p TC x 6 + left partial mastectomy（multiple re-excisions for margins）+ XRT, on letrozole x 1.5 years, tolerating well
- ✅ Osteoporosis spine（T-score -2.5），osteopenia hip — plan for zoledronate authorization + DEXA
- ✅ Medication_plan: continue letrozole + zoledronate + calcium/vitamin D 正确
- ✅ Imaging: DEXA + bilateral mammogram 正确
- ✅ Incidental findings: early Dupuytren's contracture left palm, stitch granuloma chest wall — correctly noted in findings

### ROW 59 (coral_idx 198) — 0 P1, 0 P2 ✅
- ✅ 52yo, Stage I ER+（100%）/PR+（40%）/HER2-（equivocal IHC, negative FISH）IDC right breast, 1.5cm grade 3, high grade DCIS, 0/5 SLN, Oncotype High Risk
- ✅ S/p lumpectomy + SLN + TC x 3 + Abraxane/Cytoxan x 1（allergic to docetaxel + paclitaxel）+ XRT
- ✅ Hormonal therapy: tamoxifen → letrozole → switching to exemestane（intolerance to both prior）
- ✅ Response: "no evidence of disease recurrence" 正确
- ✅ Lab: Vitamin D 32, FSH 32.5, Estradiol 5（postmenopausal confirmed），thyroid normal

### ROW 60 (coral_idx 199) — 0 P1, 0 P2 ✅
- ✅ 65yo postmenopausal, pT1bNX ER+（>95%）/PR+（10%）/HER2-（IHC 1+, FISH 1.3）IDC left breast, 0.7cm grade 2, DCIS low/intermediate
- ✅ S/p partial mastectomy, margins negative, opted for no further surgery（no SLN）
- ✅ Oncotype Dx ordered for risk assessment, adjuvant endocrine therapy 5-10 years recommended
- ✅ Response: "Not yet on treatment" 正确
- ✅ Televisit, ECOG 0, osteopenia

### ROW 61 (coral_idx 200) — 0 P1, 0 P2 ✅
- ✅ 43yo premenopausal, ER+（100%）/PR+（100%）/HER2-（IHC 1+）IDC left breast, at least 11mm（1.5cm on MRI），grade 2
- ✅ MRI: possible 2nd site at 0900 → repeat biopsy negative. CT chest/abdomen negative. Invitae negative
- ✅ Plan: lumpectomy + IORT + reconstruction 04/12/21, Oncotype Dx after surgery, adjuvant endocrine therapy
- ✅ Response: "Not yet on treatment" 正确

### ROW 62 (coral_idx 201) — 0 P1, 0 P2 ✅
- ✅ 44yo premenopausal G0P0, pT1aN0(sn)M0 ER+（>95%）/PR+（>95%）/HER2-（IHC 2+, FISH negative）IDC right breast, 0.2cm grade 1, Ki-67 <5%, low-intermediate grade DCIS 0.5cm
- ✅ S/p lumpectomy + SLN（0/1）08/04/21, negative margins, Myriad 34 gene negative
- ✅ Very low-risk: no chemo recommended, adjuvant endocrine therapy options discussed（tamoxifen vs OS+AI）
- ✅ History of endometriosis → tamoxifen may exacerbate gyn issues, start with [redacted] + close monitoring
- ✅ Stage pT1aN0(sn)M0 非常精确

### ROW 63 (coral_idx 202) — 0 P1, 0 P2 ✅
- ✅ Response: 出色 — "MRI showed a dramatic response to therapy with near total resolution of the lesion"

### ROW 64 (coral_idx 203) — 0 P1, 0 P2 ✅
- ✅ 28yo premenopausal, left breast IDC 10.3cm on MRI, ER+/PR+/HER2-, axillary LN+, suspicious sternal lesion on bone scan（solitary bone met pending biopsy）
- ✅ Currently on neoadjuvant dd AC（已开始），taxol planned. Second opinion for oligometastatic approach
- ✅ Response: "On treatment; response assessment not available" — 正确！患者确实在接受 dd AC 化疗中，response 评估尚早
- ✅ A/P: "Stage III-IV...probably metastatic disease to sternum"，aggressive treatment approach with curative/oligo-met intent
- ✅ Plan: dd AC → taxol → surgery → radiation + sternum treatment if biopsy positive → xgeva
- ✅ Second opinion: "yes" 正确
- ✅ Goals: palliative 正确（Stage IV, but doctor discusses aggressive approach for oligometastatic disease）
- ✅ *前次审查误判为 P2，现更正*：患者确实在 active chemotherapy 中，"On treatment" 是正确的

### ROW 65 (coral_idx 204) — 0 P1, 1 P2
- P2: response_assessment "On treatment" — 但患者尚未开始化疗（A/P 说 "RTC in 1-2 weeks in person to start chemo"），目前在 planning 阶段（screening for ISPY, TTE, port placement）。应为 "Not yet on treatment"
- ✅ 48yo premenopausal RN, right breast IDC, ER weak+（2%）/PR low+（7%）/HER2-（IHC 2+, FISH 1.4），Ki-67 36%, 2.6cm on MRI, LVI+, axillary LN micromet（0.2mm）
- ✅ PET/CT: hypermetabolic breast nodule, NO distant mets
- ✅ Type: 非常详细 — "ER weak positive (2%), PR low positive (7%)" 包含具体百分比
- ✅ Plan: neoadjuvant AC/T（or ISPY trial with multiple arms detailed），then surgery → radiation → endocrine therapy 5-10 years
- ✅ Procedure: TTE + port placement + chemo teaching + research biopsy/MRI for ISPY
- ✅ Genetic testing pending（done locally）
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "On treatment" but not yet started）

### ROW 66 (coral_idx 205) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, right breast metaplastic carcinoma with squamous differentiation, high-grade（3+3+3），ER 5-10%/PR 0%/HER2 0（meets TNBC criteria, ER <10%），GATA3+
- ✅ CT CAP: no distant mets, nonspecific pulmonary nodules ≤4mm. Axillary LN biopsy negative
- ✅ Second opinion: "yes" 正确，discussed pembrolizumab for early-stage TNBC
- ✅ Plan: neoadjuvant + bilateral mastectomy + adjuvant radiation
- ✅ Genetic_testing: Invitae 30 gene panel pending
- ✅ Response: "Not yet on treatment" 正确

### ROW 67 (coral_idx 206) — 0 P1, 0 P2 ✅
- ✅ 54yo postmenopausal, clinical Stage II-III left TNBC（ER-/PR-/HER2-, Ki-67 59%, EGFR+），inflammatory-like presentation（skin thickening, edema）
- ✅ S/p 2 cycles dd AC（severe complications: neutropenic fever, MRSA abscess, pleural effusion, poorly controlled DM with BS >400）
- ✅ Clinical response noted: axillary LN decreased 8→2-3cm, breast mass decreased, chemo-sensitive
- ✅ Second opinion: "yes" 正确
- ✅ Plan: Abraxane 100mg/m2 weekly x 12 wks, then consider 2 more dd AC if tolerated. PET/CT + mammogram/US staging
- ✅ Genetic counseling recommended（TNBC age 53, no family hx）
- ✅ Complex comorbidities: uncontrolled DM, MRSA, possible diastolic dysfunction — correctly captured in findings

### ROW 68 (coral_idx 207) — 0 P1, 0 P2 ✅
- ✅ Response: "good clinical response after 6 cycles of TCHP, no lesions after chemotherapy" 出色

### ROW 69 (coral_idx 208) — 0 P1, 0 P2 ✅
- ✅ 52yo postmenopausal, clinical Stage IIB T2N1 ER+（Allred 8/8）/PR+（5/8）/HER2-（IHC 2+, FISH 1.1）ILC right breast, 5x4cm, FNA+ axillary LN, Ki-67 10%
- ✅ MRI: 3.5x3.5x4.5cm mass with satellite lesions. Left breast lesion biopsied → benign
- ✅ Plan: PET/CT for staging, molecular profiling（Oncotype/MammaPrint）to guide neoadjuvant approach
- ✅ Options: neoadjuvant AI if low risk, neoadjuvant chemo（ISPY trial）if high risk. AI 5-10 years regardless
- ✅ Response: "Not yet on treatment" 正确

### ROW 70 (coral_idx 209) — 0 P1, 2 P2
- P2: response_assessment "On treatment; response assessment not available" — 但 note 有详细的 post-neoadjuvant 病理响应数据：left breast 4.4cm residual（5-10% cellularity），right breast 1cm residual（10% cellularity），2/5 SLN+。应描述 partial response（no pCR, residual disease bilateral）
- P2: letter "What is the plan going forward?" 段落几乎为空（直接跳到 "Thank you"），遗漏了 radiation consult、expanders、prolia/dental clearance、follow-up September 等重要计划内容
- ✅ 61yo postmenopausal, BRCA1 mutation+, bilateral breast cancer:
  - Left: ILC, ER+（>95%）/PR+（90%）/HER2-（IHC 0），residual 4.4cm after neoadjuvant TC, 2/5 SLN+（1 macro + 1 micro）
  - Right: IDC, ER+（95%）/PR-（<1%）/HER2 equivocal（IHC 2+），residual 1cm, 0/2 SLN-, high grade DCIS
- ✅ S/p bilateral mastectomy + preventive BSO（BRCA1）, recovering well
- ✅ Type: 出色 — 正确区分了两侧不同 histology 和 receptor profiles
- ✅ Medication_plan: restart letrozole + prolia after dental clearance 正确
- ✅ Imaging: follow-up CT for lung nodules 正确
- ✅ Letter: "one that started in the milk-producing glands (lobular) and one that started in the milk ducts (ductal)" 出色解释双侧不同类型
- ❌ **退化 vs notool**: notool 0 P2 → tool 2 P2（response + letter plan empty）
### ROW 71 (coral_idx 210) — 0 P1, 0 P2 ✅
- ✅ 45yo premenopausal, clinical Stage IIIB ER+（90-100%）/PR+（1-10%）/HER2-（IHC 1+）IDC left breast grade 3, 5.6cm mass, multiple axillary + subpectoral LN+, equivocal left 5th rib lytic lesion on PET
- ✅ Metastasis "Not sure" — 正确！A/P 明确说 rib finding "may or may not be a site of metastasis, too subtle to biopsy"（与 ROW 83 的错误 "Not sure" 不同，这里 "Not sure" 是临床上正确的判断）
- ✅ S/p neoadjuvant dd AC/taxol started 2 weeks ago + GCSF support
- ✅ Medication_plan: 出色详细 — AC 60/600mg/m2 q2wk → taxol 175 q2wk or 80 weekly, adjuvant hormonal therapy up to 10 years（OFS + AI per SOFT/TEXT trial），bone agents（zoledronate/denosumab q6mo x 3yr），CDK4/6 inhibitor discussion
- ✅ Radiotherapy: post-op RT（chest wall + regional LN）+ possible stereotactic RT to rib lesion
- ✅ Goals: curative 正确（Stage IIIB staging with curative intent）
- ✅ Genetics: 34-gene panel negative（VUS in one gene）
- ✅ Letter: 通俗准确 — "exact extent of the cancer is not fully known yet" 恰当传达 equivocal rib finding
### ROW 72 (coral_idx 211) — 0 P1, 1 P2
- P2: response_assessment "On treatment; response assessment not available" — letrozole 刚在本次就诊开处方（"Instructed patient to begin letrozole, prescription ordered"），尚未开始服用。应为 "Post-surgical, not yet on systemic treatment"
- ✅ 72yo postmenopausal, pT1cN0(sn) ER+（99%）/PR-（<1%）/HER2-（IHC 1+, FISH 1.0）IDC with focal neuroendocrine differentiation left breast, 1.2cm grade 2, 0/2 SLN, Ki-67 20%
- ✅ S/p left mastectomy 03/03/22, margins negative. Grade upgraded from core biopsy G1 → surgical specimen G2
- ✅ Type: 非常详细 — 包含 neuroendocrine differentiation + complete receptor profile
- ✅ Medication_plan: letrozole 5+ years + Oncotype Dx ordered for chemo benefit assessment
- ✅ Genetic_testing: Oncotype Dx ordered 正确
- ✅ Supportive_meds: Reclast for osteoporosis 正确
- ✅ Televisit, ECOG 0, follow-up 3 weeks for Oncotype review
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "On treatment" 模式）
### ROW 73 (coral_idx 212) — 0 P1, 0 P2 ✅
- ✅ 63yo postmenopausal, Stage III ER+/PR+/HER2- left breast cancer, s/p bilateral mastectomies + left ALND + TC/Abraxane + CW XRT + arimidex since August 2017
- ✅ Concern: new nodule over left breast → bilateral US + mammogram today = ALL fat necrosis（no recurrence）
- ✅ Response: "all three areas marked are diagnostic of fat necrosis" — 正确且具体，排除了复发
- ✅ NED on exam, PET-CT negative（2017），labs/markers negative（2018）
- ✅ Medication_plan: continue arimidex + bone agents（next August 2019）
- ✅ Goals: curative 正确
- ✅ Follow-up: 4 months + check labs

### ROW 74 (coral_idx 213) — 0 P1, 1 P2
- P2: Type_of_Cancer 说 "HER2+" — 但乳腺癌是 HER2-（IHC 1+, FISH ratio 1.1）。模型混淆了 prior gastric cancer（HER2+ IHC 3+）和 breast cancer 的 HER2 状态
- ✅ 68yo, Stage IIB pT2N1a ER+/PR+/HER2- IDC right breast, 2.5cm, 1/7 LN+
- = **同 notool**: HER2 confusion（same issue）

### ROW 75 (coral_idx 214) — 0 P1, 1 P2
- P2: procedure_plan 混入 genetics counseling + fertility referrals
- = **同 notool**: procedure mixed content

### ROW 76 (coral_idx 215) — 0 P1, 1 P2
- P2: Metastasis 列出 "left carotid bifurcation" 作为乳腺癌转移灶 — 但这是独立的 carotid body paraganglioma（良性肿瘤），不是乳腺癌转移。A/P 明确说 "Unchanged left carotid body mass, presumed to be a paraganglioma"
- ✅ 55yo, Stage IV ER-/PR-/HER2+（IHC 3+, FISH ratio 13）IDC grade 2 right breast, bone mets（left iliac, bilateral sacral ala — biopsy confirmed）
- ✅ S/p paclitaxel + trastuzumab + pertuzumab → right partial mastectomy（no residual invasive disease, DCIS only）→ breast XRT
- ✅ Currently on maintenance trastuzumab + pertuzumab q3wk
- ✅ Response: "No evidence of recurrent or metastatic hypermetabolic disease...Stable disease" 正确（PET April 2018 = NED）
- ✅ Goals: palliative 正确
- ✅ Imaging_plan: PET/CT up to toes（for new leg pain）+ echo q6mo（trastuzumab monitoring）正确
- ✅ PMH: DM, neuropathy from prior paclitaxel, diarrhea from pertuzumab, anorexia
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（paraganglioma 被错误归为 breast cancer metastasis）
### ROW 77 (coral_idx 216) — 0 P1, 1 P2
- P2: response_assessment "On treatment" — 但 exemestane 尚未开始（pending estradiol confirms menopause），术后未开始系统治疗。应为 "Post-surgical, not yet on systemic treatment"
- ✅ 52yo, Stage IIb pT2N1a(sn) ER+/PR+/HER2-（IHC 0, FISH 1.2）IDC right breast, 2.2cm grade 2, Ki-67 15%, Oncotype 5, 1/1 SLN+（1.1cm）
- ✅ S/p lumpectomy + SLN + IORT 08/27/20. Adjuvant XRT planned September 29
- ✅ Complex comorbidities: morbid obesity BMI 52.5, fibromyalgia, anxiety/PTSD（cannot tolerate MRI/DEXA），IBS, hypothyroidism, fatty liver
- ✅ Duloxetine interaction with tamoxifen → prefer AI（exemestane）正确
- ✅ No chemo recommended（Oncotype 5）
- ✅ Lab_plan: estradiol to confirm menopause + hep serologies for fatty liver 正确
- ✅ Incidental: right axilla fungal infection — correctly noted
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "On treatment" 模式）
### ROW 78 (coral_idx 217) — 0 P1, 0 P2 ✅
- ✅ 79yo, de novo Stage IV TNBC（ER-/PR-/HER2-, PD-L1 negative）metastatic to liver + periportal LNs since July 2017
- ✅ Extensive treatment history: capecitabine x 10 cycles → OPERA trial oral [redacted] → gemcitabine x 4（stopped for fatigue + pericardial effusion）→ off therapy since 03/15/19
- ✅ Response: 出色 — "off systemic therapy since 03/15/19. Imaging shows worsening of metastatic disease with interval enlargement of hepatic and nodal metastases" 准确描述了 off-treatment 期间的进展
- ✅ New lung nodule 7mm suspicious for met — correctly noted in findings
- ✅ Patient exploring Phase 3 ADC trial, not interested in chemo, doxil as backup option
- ✅ Radiotherapy: consult for XRT to liver/LN disease 正确
- ✅ Complex comorbidities: DM, HTN, h/o hemorrhagic brainstem CVA, pericardial effusion, ?IPMN pancreas
- ✅ Lab 完整（CBC + CMP from 08/24），Creatinine 1.27↑ 提示 CKD
- ✅ Letter: 通俗准确 — "cancer has grown since the last visit" + "radiation doctor to discuss if radiation could help"
### ROW 79 (coral_idx 218) — 0 P1, 0 P2 ✅
- ✅ 61yo, Stage IV ER+（inferred from tamoxifen）/HER2+（inferred from Herceptin/Tykerb）metastatic breast cancer to bone + pleural effusion + multiple sites
- ✅ Extensive treatment history（2008-2011）: arimidex → tamoxifen → Herceptin-based combos → faslodex → tykerb+xeloda → all stopped. Off treatment last week（rising markers + bone pain）
- ✅ Response: "cancer is not responding. Increased bone pain and rising tumor markers. Stopped all treatment." 正确描述
- ✅ Complex comorbidities: Type 1 DM with nephropathy/neuropathy，PE on warfarin，pleural effusion
- ✅ Procedure: power port + thoracentesis for markers 正确
- ✅ Imaging: restaging PET/CT 正确
- ✅ Pain management: methadone 10mg TID + oxycodone PRN + consider cymbalta/lyrica
- ✅ Goals: palliative 正确
### ROW 80 (coral_idx 219) — 0 P1, 1 P2
- P2: response_assessment "On treatment" — TC x 4 尚未开始（scheduled 04/11/19），这是 pre-start visit，应为 "Not yet on treatment"
- ✅ 53yo postmenopausal, local skin recurrence of IDC 7 years after mastectomy for DCIS. 0.8cm grade 3 IDC in dermis, ER 95%/PR 70%/HER2-（IHC 1+, FISH not amplified），Ki-67 15-20%, Oncotype 24
- ✅ S/p wide excision（benign margins），PET/CT negative for distant mets
- ✅ Genetic_testing: whole genome sequencing completed — no actionable mutation. Tempus testing: NF1, FGFR1 等
- ✅ Medication_plan: TC x 4 starting 04/11/19 + cold cap + supportive meds（dexamethasone, zofran, compazine）
- ✅ Radiotherapy: 6 weeks（5wk + 1wk boost）including left axilla + SC fields 详细正确
- ✅ Lab 完整（CMP + CBC + Hep B + estradiol 4）
- ✅ Goals: curative 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "On treatment" 模式）
### ROW 81 (coral_idx 220) — 0 P1, 0 P2 ✅
- ✅ 60yo postmenopausal OB/GYN physician, TWO Stage I primary cancers in left breast:
  - IDC tubular type 3mm G1 ER+/PR+/HER2-
  - ILC 2mm G1 ER+/PR+/HER2-
- ✅ 0/1 SLN, margins negative, low grade DCIS + ADH + ALH. S/p lumpectomy + bilateral mastopexy + implant removal
- ✅ Second opinion: "yes" 正确
- ✅ BRCA 1/2 negative. Strong family hx（mother: breast ca + sarcoma, maternal aunt/uncle: pancreatic ca, grandmother: ovarian ca）
- ✅ Response: "Not yet on treatment" 正确
- ✅ Plan: radiation → AI after radiation, no chemo. Baseline DEXA
- ✅ Type: 出色 — 正确分列两个不同 histologic type 的 primary tumors
### ROW 82 (coral_idx 221) — 0 P1, 2 P2
- P2: response_assessment "On treatment" — 但术后未开始 radiation 或 hormonal therapy，应为 "Not yet on systemic treatment"
- P2: medication_plan 直接 dump 了整个 medication list（tylenol, HCTZ, lisinopril, metformin, dental cream 等非癌症药物），应只写癌症治疗计划（AI +/- bone medication after radiation）
- ✅ 52yo postmenopausal, Stage IB→II（note inconsistency）ER+/PR+/HER2- mixed ductal/lobular carcinoma right breast, 4.3cm G2-3, LN positive, Oncotype low risk（Ki-67 15%）
- ✅ S/p right lumpectomy + SLN 11/16/2020, no chemo recommended（low risk Oncotype）
- ✅ Plan: radiation → AI +/- bone medication, DEXA ordered, exercise counseling, lifestyle modifications
- ✅ PMH: DM type 2, HTN, anxiety, GERD, fatty liver, vitamin D deficiency
- ✅ Goals: curative 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 2 P2

### ROW 83 (coral_idx 222) — 0 P1, 1 P2
- P2: Stage "Stage IV (metastatic)" 但 Distant Met = No，A/P 说 "W/u negative for distant metastasis"。Axillary LN 是 regional 不是 distant — 模型逻辑矛盾
- ✅ Response: 出色 — "The cancer is currently responding to neoadjuvant endocrine therapy. Recent imaging shows significant response" 有详细 SUV 数据
- = **同 notool**: Stage IV contradiction

### ROW 84 (coral_idx 223) — 0 P1, 1 P2
- P2: response_assessment 开头说 "stable disease" 但随后描述了 liver lesions "increased in size and number" + brain MRI "increased involvement of the right internal auditory canal" — 自相矛盾。实际上是 progression on letrozole/palbociclib → switched to xeloda，brain disease possibly worsening
- ✅ 60yo, CHEK2 biallelic mutation, MS（wheelchair-bound since 2002），Stage IV ER+（71-80%）/PR-/HER2-（IHC 2+, FISH 1.3）metastatic breast cancer
- ✅ Extensive mets: bone（diffuse lytic/blastic, C2 pathologic fracture）+ liver（enlarging on CT July 2020）+ LN + possible pachymeningeal/leptomeningeal disease
- ✅ Prior treatment: right breast cancer 1999 + 2006（mastectomy, CAF, tamoxifen）→ met recurrence 2019 → letrozole/palbociclib（PD July 2020）→ capecitabine
- ✅ Neurologic progression: right eye droop, hearing loss, dizziness → possible LMD（CSF cytology #1 negative）
- ✅ PIK3CA mutation on Strata → future option fulvestrant + alpelisib
- ✅ Plan: continue xeloda + repeat CT CAP + repeat LP/cytology + MRI spine + rad onc referral + steroids + continue zoledronic acid
- ✅ Goals: palliative 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（response "stable" 但实际 progression）
### ROW 85 (coral_idx 224) — 0 P1, 0 P2 ✅
- ✅ 61yo, Stage IIIA→IV ER+/PR-/HER2- ILC right breast, extremely complex treatment history: neoadjuvant letrozole（minimal effect）→ bilateral mastectomy（82mm residual, 6/6 LN+, grade 3）→ dd AC/T → exemestane → met recurrence within 1 year（bone, muscle, brain）→ fulvestrant/palbociclib（PD: 3 new liver lesions + increased bone）
- ✅ Brain mets s/p GK + Meckel's cave lesion involving trigeminal nerve + leptomeningeal disease. New left facial numbness + headache
- ✅ Response: "Disease progressed on first line fulvestrant/palbociclib in bone with new liver metastases" — 准确描述 PD
- ✅ Molecular: FGFR1 amp, BRCA2 VUS, TMB 14, CCND1 amp 等多个扩增
- ✅ Plan: Phase 1 trial olaparib + [redacted], rad onc referral for trigeminal nerve lesion, steroid taper
- ✅ Goals: palliative 正确
- ✅ Letter: "fulvestrant and palbociclib, did not work as well as we hoped" + "cancer spot near your left trigeminal nerve" 通俗准确
### ROW 86 (coral_idx 225) — 0 P1, 0 P2 ✅
- ✅ 53yo, originally right breast mixed IDC/ILC Grade III, HER2+（FISH 4.37）→ s/p TCHP x 6 → bilateral MRM（3 masses, 4/15 LN+）→ adjuvant XRT
- ✅ Met recurrence Dec 2018: bone + liver + brain dural mets. Met biopsy L pelvis: ER 95%/PR 2%/HER2 1+（FISH negative — receptor discordance from primary）
- ✅ CHEK2 mutation, [redacted] mutation（may have contributed to AI resistance per A/P discussion）
- ✅ On letrozole + ribociclib → PD April 2020（increasing bone mets, liver stable）
- ✅ Response: 出色 — "progression of bone metastases while liver metastases remain stable" 准确区分
- ✅ Medication_plan: fulvestrant +/- everolimus + continue denosumab 正确
- ✅ Radiotherapy: palliative XRT cervical spine + mandible 正确
- ✅ Goals: palliative 正确
- ✅ Letter: "cancer in your bones has grown, but the cancer in your liver has stayed the same" 通俗准确
### ROW 87 (coral_idx 226) — 0 P1, 0 P2 ✅
- ✅ 79yo, multifocal right breast IDC grade 2, 2.2cm + 0.6cm, ER+/PR+/HER2-, 4/19 LN+ with extracapsular extension, clear margins
- ✅ Second opinion: "yes" 正确
- ✅ Parkinson's disease（right-sided, moderate, pill-rolling tremor）— 影响 chemo risk-benefit
- ✅ Family hx: daughter breast+colorectal ca at 40, grandmother ovarian ca
- ✅ Recurrence risk ~40-45%. Chemo benefit only 3-4% additional → patient + doctor agree hormonal therapy alone（no chemo）
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确
- ✅ Medication_plan: hormonal therapy alone（specific drug not yet selected）

### ROW 88 (coral_idx 227) — 1 P1, 0 P2
- **P1**: response_assessment "Not mentioned in note" — 但 note 有大量响应数据：neoadjuvant AC→Taxol→Taxol/Carbo progression → surgery（residual 4cm+2.6cm, 23/30 LN+）→ brain mets → lung/LN mets。应描述持续进展
- ❌ **未改善 vs notool**: 同样的 P1（response "Not mentioned" for clear progression）

### ROW 89 (coral_idx 228) — 0 P1, 0 P2 ✅
- ✅ 53yo perimenopausal, Stage I ER+/PR+/AR+/HER2- IDC left breast, 9mm G2, node negative
- ✅ S/p lumpectomy + SLN March 2020 + radiation April 2020. Genomic profiling low risk → no chemo
- ✅ Current_meds: tamoxifen 正确. Plan to switch to AI midway when postmenopausal
- ✅ Response: "Not mentioned" 正确（post-surgical NED follow-up）
- ✅ Imaging: mammogram February 2021 正确
- ✅ Follow-up: 2-3 months 正确
- ✅ Goals: curative, Advance care: Full code 正确
### ROW 90 (coral_idx 229) — 0 P1, 0 P2 ✅
- ✅ 51yo premenopausal, Clinical Stage II/III right breast adenocarcinoma, on ISPY trial. Oncotype High Risk
- ✅ S/p neoadjuvant taxol → right lumpectomy: 2.2cm residual IDC, ~60% cellularity（partial response, no pCR）
- ✅ Currently on adjuvant AC cycle 3→4（dose delay 1 week for symptom recovery）
- ✅ Response: "2.2 cm residual IDC with ~60% cellularity after neoadjuvant therapy" 准确描述病理响应
- ✅ Side effects well managed: switch granisetron for zofran-induced constipation, add olanzapine + dex, GCSF dose reduced 50%
- ✅ PMH: autoimmune thyroiditis → hypothyroidism（TSH 6.01↑），BLM gene carrier, port extravasation（reinserted）
- ✅ Lab 完整: CMP + CBC + thyroid + pregnancy test. Toxic granulation + immature granulocytes c/w GCSF
- ✅ Goals: curative 正确
### ROW 91 (coral_idx 230) — 0 P1, 1 P2
- P2: response_assessment 引用了 11/2011 MRI/PET 的 progression 数据（pre-everolimus），但患者自 April 2012 起在 everolimus+exemestane 上，当前 response 未知（PET/CT 下周 pending）
- ✅ 53yo postmenopausal, Stage I→IV ER+/PR+/HER2- IDC right breast dx 2003, bone mets since 2005
- ✅ Multiple prior lines → current: everolimus + exemestane + denosumab since April 2012
- ✅ New 1cm right iliac LN, "unclear significance." RLE edema improving on lasix
- ✅ Imaging: PET/CT next week 正确. Goals: palliative 正确
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2

### ROW 92 (coral_idx 231) — 0 P1, 0 P2 ✅
- ✅ 67yo, Stage IV ER+/HER2- breast cancer（dx 1991 ILC, 7/7 LN+）, liver mets（biopsy: ER 60%）
- ✅ 20+ years treatment history: CAF→CMF→XRT→tamoxifen→raloxifene→taxotere/xeloda→abraxane→xeloda→gemcitabine→faslodex→now epirubicin 25mg/m2 weekly C2D1
- ✅ Response: "Liver size decreased, tenderness reduced, indicating some improvement" — 临床改善准确描述
- ✅ Tumor markers extremely elevated: CA 27.29 = 3332, CEA = 380.8 — 正确记录
- ✅ Lab 完整: CMP + CBC + tumor markers. Neutropenia 1.70, low lymphocytes, macrocytic anemia（MCV 102）
- ✅ Goals: palliative 正确. Current_meds: epirubicin + denosumab 正确
### ROW 93 (coral_idx 232) — 0 P1, 0 P2 ✅
- ✅ 53yo postmenopausal, Stage 1 ER-/PR-/HER2+（IHC 3+）IDC left breast, multifocal（0.21cm largest），grade 3, Ki-67 40%, high grade DCIS, 0/1 SLN, margins negative
- ✅ S/p left partial mastectomy + SLN + bilateral reduction mammoplasty. Invitae negative（VUS in APC, CHEK2, DIS3L2）
- ✅ Initially reluctant about chemo → agreed after detailed discussion of rationale
- ✅ Medication_plan: paclitaxel + trastuzumab weekly x 12 wks → trastuzumab q3wk x 9 months 正确
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确（"disease is curable"）
- ✅ Procedure: mediport placement needed. Radiation consult today
### ROW 94 (coral_idx 233) — 0 P1, 0 P2 ✅
- ✅ 75yo postmenopausal, Stage IIA（pT1b N1(sn)）ER+/PR+/HER2- IDC left breast, 1.6cm, 3 LN+, Grade 2, Oncotype RS 21
- ✅ S/p lumpectomy June 2019 → radiation → letrozole. Declined chemo（RS 21）. Genetic testing negative
- ✅ Response: "No evidence of disease recurrence on imaging and exam" 正确（mammogram 12/2020 normal, asymptomatic）
- ✅ Current_meds: letrozole 正确
- ✅ Imaging: mammogram November 2021 + high risk screening MRI 正确
- ✅ Goals: curative 正确
- ✅ Lifestyle excellent: walks 300 min/week, lacto-ovo vegetarian, meditation. Uses CBD for joint pain from AI
- ✅ Follow-up: q6mo alternating NP and MD

### ROW 95 (coral_idx 234) — 0 P1, 0 P2 ✅
- ✅ Stage: "Not mentioned in note"（正确保守）— notool 版本写了 "Stage IV" 但这是 ISPY trial 患者（Stage I-III），tool 版本修复了此 P2
- ✅ **改善 vs notool**: Stage IV P2 消除

### ROW 96 (coral_idx 235) — 0 P1, 0 P2 ✅
- ✅ 47yo premenopausal, pT1cN0(sn) ER+（~60%）/PR+（~50%）/HER2-（IHC 0, FISH 1.2）mixed ductal/cribiform carcinoma left breast, 1.8cm grade I, Ki-67 ~5%, AR+
- ✅ S/p left partial mastectomy 03/11/21, 0/2 SLN. Invitae negative
- ✅ No chemo recommended（low grade, low proliferation）→ Oncotype ordered to confirm
- ✅ Medication_plan: tamoxifen after XRT. If Oncotype high → consider OFS+AI
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确
### ROW 97 (coral_idx 236) — 0 P1, 0 P2 ✅
- ✅ 53yo, pT1bN0(sn) ER+（>95%）/PR+（~60%）/HER2-（IHC 1+）IDC left breast, 0.8cm grade 1, Ki-67 10%, 0/3 SLN, low to intermediate grade DCIS
- ✅ S/p left partial mastectomy 03/26/19. Post-op drain still in place
- ✅ Complex PMH: relapsing-remitting MS on fingolimod（Gilenya）→ coordination needed with MS team
- ✅ Oncotype Dx ordered（"anticipate no need for chemotherapy"）→ adjuvant endocrine therapy planned
- ✅ Response: "Not yet on treatment" 正确
- ✅ Goals: curative 正确
- ✅ No problem anticipated with Gilenya + future endocrine therapy
### ROW 98 (coral_idx 237) — 0 P1, 0 P2 ✅
- ✅ 78yo, TNBC with apocrine features, IDC + high grade DCIS 2.7cm, 0/3 SLN, s/p right partial mastectomy
- ✅ On adjuvant TC cycle 4, complicated by anemia（Hgb 9.7），marked reactive thrombocytosis（plt 1052），scattered rash, low-grade fevers, SOB
- ✅ Type: "ER-/PR-/HER2- IDC with apocrine features" 正确
- ✅ Response: "Not mentioned" 正确（A/P says "Labs reviewed, Exam stable, Okay to proceed"）
- ✅ Medication_plan: continue TC, refer rad onc, port removal planned
- ✅ Goals: curative 正确
### ROW 99 (coral_idx 238) — 0 P1, 1 P2
- P2: Type_of_Cancer "ER+/PR+/HER2+ IDC" — 混合了两个不同 primary cancer 的 receptor：left breast（ER+/PR+/HER2-）和 right breast（ER-/PR-/HER2+）。这是两个独立的原发癌，不能合并 receptor profiles
- ✅ 49yo, TWO primary cancers: left Stage III ER+/PR+/HER2-（2.6cm G3, 5/25 LN+, s/p MRM → dd AC/T → XRT → tamoxifen → letrozole）+ right Stage I ER-/PR-/HER2+（incidental 0.6cm at prophylactic mastectomy → TH x 12wk）
- ✅ Metastatic disease June 2012: lung mass + mediastinal LN. NO biopsy yet — receptor status unknown. On fulvestrant
- ✅ Response: 出色 — "lung mass increased 2.9→3.4cm, new AP window LN with hypermetabolic activity...indicative of disease progression" 准确描述 PD on fulvestrant
- ✅ Plan: biopsy needed first → treatment depends on receptor（ER+ → exemestane+everolimus/xeloda; ER- → chemo）
- ✅ Goals: palliative 正确
- ✅ Symptom management referral placed for severe chest pain
- ❌ **退化 vs notool**: notool 0 P2 → tool 1 P2（receptor profiles merged）
### ROW 100 (coral_idx 239) — 0 P1, 0 P2 ✅
- ✅ 68yo, Stage IV ER+（80%）/PR+（50%）/HER2- IDC, metastatic to liver + multiple sites. 2003 left breast recurrence was ER-/PR-/HER2-
- ✅ Extensive treatment history: lumpectomy → CAF → taxol → XRT → arimidex x 7yr → abraxane+bevacizumab → faslodex → xeloda → now gemcitabine C2D8（cancelled by patient for fatigue）
- ✅ Response: "Tumor markers increased" — A/P correctly captures uncertainty: "Unclear if progressing or tumor flare, scan done too early to tell"
- ✅ Rising markers: CA 15-3 = 118-119, CA 27.29 = 178-181, CEA = 312-320. ALP elevated（172-196）
- ✅ Macrocytic anemia（Hgb 9.9, MCV 104），lymphopenia（0.66）
- ✅ Patient wants treatment break → discussed risks, recommended exercise + Focalin for fatigue
- ✅ Goals: palliative 正确

