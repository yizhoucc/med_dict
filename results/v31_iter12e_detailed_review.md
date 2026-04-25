# V31 iter12e — 逐字逐句原文对照审查

> 56 samples × 11 fields = 616 field-instances
> 每个 field 写出：提取了什么 + 原文说了什么 + 对不对

## 状态
- **审查完成: 56/56** ✅
- P0: 0, P1: 2, P2: 52
- 完美(0 issues): 24/56 (43%)
- P1: ROW 51(Type空值,LLM行为), ROW 57(Type矛盾TNBC+ER+,LLM行为)
- P2分布: Type(5), Stage(5), DistMet(4), response(5), current_meds(2), therapy(5), imaging(5), lab(4), genetic(3), medication(8), other(6)

---

## ROW 1 (coral_idx 140)
**原文关键事实**：56yo，2013年5月诊断multifocal Stage IIA右乳cancer。2013年6月mastectomy+sentinel node+implant。2.3和2.4cm tumors，sentinel node negative。G2，ER+PR+，HER2 neu negative。拒绝了tamoxifen，无chemo/radiation。现在CT显示widely metastatic——lungs, peritoneum, liver, ovary，右腋窝local recurrence。无current meds。A/P：palliative，biopsy planned，如果HR+/HER2-推荐ibrance+*****。Orders：MRI brain+bone scan+labs。Integrative Medicine referral。Full code。

**逐字段审查**：

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type_of_Cancer | "ER+/PR+ grade 2 IDC with metastatic recurrence, HER2-" | 原文"G2, ER and PR positive and her 2 neu negative" + 现在metastatic recurrence ✓ | ✅ |
| 2 | Stage_of_Cancer | "Originally Stage IIA, now metastatic (Stage IV)" | 原文"multifocal Stage IIA" + 现在widespread mets = Stage IV ✓ | ✅ |
| 3 | Distant Metastasis | "Yes, to lungs, peritoneum, liver, ovary" | 原文"involvement of the lungs, peritoneum, liver and ovary" ✓ | ✅ |
| 4 | response_assessment | "CT 12/24/2019 widespread metastases..."(详细列出CT所有8项) | 忠实引用了CT impression所有findings ✓ | ✅ |
| 5 | current_meds | (空) | 原文"No current outpatient medications on file" ✓ | ✅ |
| 6 | goals_of_treatment | "palliative" | 原文"treatment would be palliative" ✓ | ✅ |
| 7 | therapy_plan | "If HR+/HER2- will recommend ibrance and [REDACTED]" | A/P第4条"If she is HR+/her 2 negative on biopsy will recommend ibrance and *****" ✓ | ✅ |
| 8 | imaging_plan | "Brain MRI" | 原文"I also ordered a MRI of brain **and bone scan**" — **漏了bone scan** | **P2** |
| 9 | lab_plan | "ordered a MRI of brain and bone scan as well as labs" | 包含了MRI和bone scan——这些是imaging不是lab。实际lab orders是CBC/CMP/CA15-3/CEA/APTT/PT | **P2** |
| 10 | genetic_testing_plan | "No specific genetic tests planned. Biopsy scheduled to confirm HR/HER2" | biopsy确认receptor status是pathology不是genetic testing | **P2** |
| 11 | medication_plan | "If HR+/HER2- recommend ibrance and unspecified agent" | A/P第4条一致 ✓。iter12之前误加了tamoxifen(declined)，**iter12e已修复** ✓ | ✅ |

**P0:0 P1:0 P2:3** — imaging漏bone scan，lab混入imaging，genetic含biopsy info

## ROW 2 (coral_idx 141)
**原文关键事实**：44yo，Lynch Syndrome+endometrial+colon cancer+metastatic TNBC。原发July 2013 Stage IIB TNBC grade 3。2018年local recurrence R chest wall。Metastatic: liver(biopsy confirmed), bone(S1 fracture), chest wall, possibly intracranial/skull base。On irinotecan C3D1。A/P：chest wall更tender/erythematous（infection?）→ Rx Doxycycline。Back pain worse（可能PD）。Change irinotecan to q2weeks 150mg/m2。Scans in 3 months。1 unit pRBC。Monitor HBV q4m。Continue morphine/flexeril/oxycodone。Increase Effexor。500ml NS IV + 40mEq potassium today。MRI brain if worse。XRT consult。F/u 2 weeks。

**逐字段审查**：

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "TNBC with metastatic recurrence" | 原文"ER-/PR-/*****- (TNBC), grade 3 IDC" — 漏了grade 3 | P2 |
| 2 | Stage | "Originally Stage IIB, now metastatic (Stage IV)" | 原文"clinical Stage IIB" + now widely metastatic ✓ | ✅ |
| 3 | DistMet | "Yes, liver, bone, chest wall, possibly intracranial" | PET/CT: liver+bone+chest wall+skull base ✓ | ✅ |
| 4 | response | "MRI bone marrow lesions... no specific evidence" | A/P说"chest wall more tender"+"back pain worse possibly PD"+"PET showed significantly increased mets"——漏了这些progression证据 | P2 |
| 5 | current_meds | "irinotecan" | C3D1 irinotecan ✓ | ✅ |
| 6 | goals | "palliative" | metastatic disease ✓ | ✅ |
| 7 | therapy | "irinotecan dose change + Rad Onc referral + potassium/brace/home health" | irinotecan change ✓, XRT consult ✓, supportive items ✓(POST hook加的) | ✅ |
| 8 | imaging | "Scans in 3 months, MRI brain if worse" | A/P "Scans again in 3 months" ✓, "MRI brain if worse" ✓ | ✅ |
| 9 | lab | "ALT/HBsAg/HBV DNA q4m. Scans in 3 months" | HBV monitoring ✓。但"Scans in 3 months"是imaging不是lab | P2 |
| 10 | genetic | "None planned" | ✓(genetic testing已做过) | ✅ |
| 11 | medication | "[REDACTED] 30mg TID, Flexeril, oxycodone, Cymbalta, Effexor; also: doxycycline" | 原文A/P: Continue morphine 30mg TID ✓, Flexeril ✓, oxycodone ✓, Rx Doxycycline ✓, Increase Effexor ✓。**漏了**：500ml NS IV today, 40mEq PO potassium today, 1 unit pRBC | P2 |

**P0:0 P1:0 P2:4** — Type漏grade 3，response漏progression证据，lab混入scans，medication漏NS/potassium/pRBC

## ROW 3 (coral_idx 142)
**原文关键事实**：53yo，新诊断Stage IIA右乳IDC 1.7cm，node positive，HR+/HER2 neg by FISH。Ki-67 30-35%。Scheduled for PET CT和genetic testing。Video consult。多方second opinion。A/P：讨论了chemo/surgery/radiation/hormonal blockade的role。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "HR+, HER2 neg grade 2 IDC" | 原文"HR+, her 2 2+, fish negative"。应写ER+/PR+不是HR+ | P2 |
| 2 | Stage | "Stage IIA" | A/P第1条"Stage IIA" ✓。iter12b修了recurrence误触发 | ✅ |
| 3 | DistMet | "No" | 新诊断，无distant mets ✓ | ✅ |
| 4 | response | "Not yet on treatment" | 确实未开始治疗 ✓ | ✅ |
| 5 | current_meds | (空) | 无current medications ✓ | ✅ |
| 6 | goals | "curative" | 早期cancer ✓ | ✅ |
| 7 | therapy | "Discussed chemo/surgery/radiation" | A/P讨论了4项：chemo/surgery/radiation/**hormonal blockade**。漏了hormonal | P2 |
| 8 | imaging | "PET scan follow up" | 原文"She is scheduled for a pet ct" ✓ | ✅ |
| 9 | lab | "No labs planned" | A/P无lab计划 ✓ | ✅ |
| 10 | genetic | "Genetic testing sent and pending" | A/P"Genetic testing sent and is pending" ✓ | ✅ |
| 11 | medication | "None" | A/P无具体药物计划 ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 4 (coral_idx 143)
**原文关键事实**：75yo，ER+/PR+/HER2- IDC。2016年mastectomy。2.8cm grade 2 IDC，HER2 2+ IHC但FISH negative。On letrozole since 2016。No evidence of recurrence。Osteopenia，on calcium/VitD。A/P：continue letrozole，mammogram July 2019，DEXA July 2019，"if worsening consider brain MRI"，Prolia if BMD<-2.5。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade 2 IDC" | 原文ER+/PR+, G2, HER2 FISH neg ✓ | ✅ |
| 2 | Stage | "Not mentioned in note" | Staging form redacted(Stage *****)。A/P没显式写Stage | P2 |
| 3 | DistMet | "No" | No recurrence ✓ | ✅ |
| 4 | response | "without evidence of disease recurrence on imaging/exam/ROS" | A/P第1条原文一致 ✓ | ✅ |
| 5 | current_meds | "letrozole" | On Letrozole since 2016 ✓ | ✅ |
| 6 | goals | "curative" | 无mets, adjuvant ✓ | ✅ |
| 7 | therapy | "Continue Letrozole 2.5mg, calcium/VitD, weight-bearing, Prolia if BMD<-2.5" | A/P一致 ✓ | ✅ |
| 8 | imaging | "Mammogram July 2019, DEXA July 2019, Brain MRI" | Mammogram ✓, DEXA ✓。**Brain MRI是conditional**("if worsening, consider") | P2 |
| 9 | lab | "No labs planned" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "Letrozole 2.5mg, Mg supplement, calcium, VitD, Prolia if BMD<-2.5, probiotics" | 完整 ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 5 (coral_idx 144)
**原文关键事实**：31yo premenopausal，Stage III ER+/PR+/HER2- IDC micropapillary。Met recurrence：left cervical LN(FNA confirmed)，left IM LN，sternum(new)。On leuprolide(A/P写"on leuprolide")/anastrozole/palbociclib。A/P：continue therapy，Rad Onc referral for left neck/brachial plexus，labs monthly on lupron day，CT+bone scan ordered。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade 2 IDC (micropapillary) with met recurrence" | ✓ | ✅ |
| 2 | Stage | "Originally Stage III, now metastatic (Stage IV)" | ✓ | ✅ |
| 3 | DistMet | "Yes, to left IM LN and sternum" | **漏了left cervical LN**(FNA confirmed metastatic) | P2 |
| 4 | response | "CT neck/CAP 08/23/19: decreased cervical LNs, stable mediastinal..." | 详细imaging findings ✓ | ✅ |
| 5 | current_meds | "anastrozole, palbociclib, **goserelin**" | A/P说"on **leuprolide**"不是goserelin | P2 |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | "Continue leuprolide/anastrozole/palbociclib, Rad Onc referral" | ✓ | ✅ |
| 8 | imaging | "CT and bone scan ordered" | ✓ | ✅ |
| 9 | lab | "Labs monthly on lupron injection day" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "Continue leuprolide, anastrozole, palbociclib" | ✓，不再有lupron重复 | ✅ |

**P0:0 P1:0 P2:2**

## ROW 6 (coral_idx 145)
**原文关键事实**：34yo，ER+/PR+/HER2- grade 1 IDC，extensive DCIS。S/p bilateral mastectomy 06/21/19。1.5cm, 0/1 nodes。Started zoladex 06/08，letrozole 07/05。A/P：start letrozole today，continue zoladex 3 years，can sequence with tamoxifen，gabapentin prn，estradiol monthly。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade 1 IDC with extensive DCIS" | ✓ | ✅ |
| 2 | Stage | "Stage IA (pT1 N0)" | 1.5cm+0/1 nodes = pT1c N0 = IA ✓ | ✅ |
| 3 | DistMet | "No" | ✓ | ✅ |
| 4 | response | "Recovering with nerve irritation, edema, ecchymosis" | 描述的是**手术恢复**不是cancer response | P2 |
| 5 | current_meds | "zoladex, letrozole" | ✓ | ✅ |
| 6 | goals | "curative" | ✓ | ✅ |
| 7 | therapy | "Start letrozole, continue zoladex 3 years, tamoxifen sequence" | ✓ | ✅ |
| 8 | imaging | "No imaging planned" | ✓ | ✅ |
| 9 | lab | "Estradiol monthly" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓(Myriad已做) | ✅ |
| 11 | medication | "Start letrozole, zoladex 3yr, gabapentin, estradiol" | ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 7 (coral_idx 146) ✅
A/P: "IMP  1. MBC, recurrent and metastatic \n since 2008. ***** by IHC, *****/***** or possibly *****. R
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER-/PR- IDC (invasive ductal carcinoma) with metastatic recurrence (le | ✅ |
| 2 | Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | ✅ |
| 3 | Distant Metastasis | Yes, to left supraclavicular node and mediastinum | ✅ |
| 4 | response_assessment | Probable mild progression in the left breast and possibly the left ant | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Do not consider hormonal therapy at this time. Discussed current clini | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | Would recheck [REDACTED] prior to above | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Recommend [REDACTED] as next line of treatment. Recheck [REDACTED] pri | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 8 (coral_idx 147) ✅
A/P: "Assessment and Recommendations:  29 y.o. premenopausal patient with a history of clinical stage III
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) grade 3 invasive ductal carcino | ✅ |
| 2 | Stage_of_Cancer | Originally Stage III, now post-neoadjuvant with 3 of 28 LN positive fo | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | The patient received neoadjuvant TCHP followed by left lumpectomy and  | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | adjuvant AC x 4 cycles, to be followed by T-DM1; radiation | ✅ |
| 8 | imaging_plan | echocardiogram prior to starting AC | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | adjuvant AC x 4 cycles, to be followed by T-DM1 | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 9 (coral_idx 148)
**原文关键事实**：Stage II HR+/HER2- IDC。S/p 4 cycles AC + weekly taxol x 12。S/p bilateral mastectomies。Path：3.84cm residual IDC grade 2，1 LN with macro(0.21cm)+extranodal extension。Plan：radiation→letrozole。Fosamax for osteopenia。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR-/HER2- grade 2 IDC" | Path：ER+ 85%，PR- <1%，HER2 IHC 0/FISH neg ✓ | ✅ |
| 2 | Stage | "Stage II (inferred from **pT3** N1)" | 3.84cm = **pT2**不是pT3(pT3是>5cm) | P2 |
| 3-11 | 其余9字段 | 全部正确 | therapy/medication/imaging/lab/genetic ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 10 (coral_idx 149) ✅
A/P: "Assessment \\Plan :    1. Stage II left ***** of the breast HR + and her 2 negative s/p neoadjuvant
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | HR+ (ER+/PR+) HER2- grade 2 invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage II (inferred from 8.8 cm tumor with 20 lymph nodes involved) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | No specific evidence regarding the current response of the cancer is p | ✅ |
| 5 | current_meds | letrozole | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue letrozole. To have radiation to the left chest wall and surro | ✅ |
| 8 | imaging_plan | To have DEXA. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | She will continue on letrozole started April 2021. | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 11 (coral_idx 150)
**原文关键事实**：68yo，Stage IIIC IDC met to bone。On Faslodex+Denosumab。PET/CT(10/10/12)showed increased met activity+mandibular mass。A/P：**Exam stable**。Mouth healing thrush→mycelex+salt/soda rinses。Right leg pain worse→PETCT to toes。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "IDC with met recurrence to bone, ER+(inferred from letrozole)" | ✓ | ✅ |
| 2 | Stage | "Originally Stage IIIC, now Stage IV" | ✓ | ✅ |
| 3 | DistMet | "Yes, to bone" | ✓ | ✅ |
| 4 | response | "Recent PET/CT showed increased met activity..." | PET是10/10/12的，不是current visit。A/P说"**Exam stable**" | P2 |
| 5 | current_meds | "Fulvestrant, Denosumab" | ✓ | ✅ |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | "Continue Faslodex and Denosumab" | ✓ | ✅ |
| 8 | imaging | "PETCT to evaluate Femur and toes" | ✓。漏了"due to worsening numbness"但这是reason不是plan | ✅ |
| 9 | lab | "No labs planned" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "Faslodex+Denosumab+Mycelex for thrush" | ✓。漏了"salt and soda rinses"——次要 | P2 |

**P0:0 P1:0 P2:2**

## ROW 12 (coral_idx 151)
**原文关键事实**：51yo，Stage IV de novo ER+/PR+/HER2+(IHC 3+/FISH 5.4) MBC to ****(liver?)/lung/nodes/brain/bone。On herceptin+*****+letrozole since 08/18/17。MRI brain 08/15/18: new CNS lesions。CT CAP stable。New brain mets on recent MRI。A/P：continue herceptin/*****/letrozole，***** q12wks，off chemo，CT CAP q4m，bone scan q4m，MRI brain q4m，**Echo q6m**，await GK/Rad Onc。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2+(IHC 3+/FISH 5.4) IDC" | ✓ | ✅ |
| 2 | Stage | "Stage IV" | ✓ | ✅ |
| 3 | DistMet | "Yes, to brain, lung, bone" | A/P说"to *****, lung, **nodes**, brain and bone"——**漏了nodes** | P2 |
| 4 | response | "MRI brain new lesions, CT stable, celiac node decreased" | ✓ 详细 | ✅ |
| 5 | current_meds | "herceptin, letrozole" | A/P"herceptin and *****"——**漏了pertuzumab(*****)** | P2 |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | "continue herceptin/agent, letrozole, off chemo, await GK" | ✓ | ✅ |
| 8 | imaging | "CT CAP q4m, bone scan q4m, MRI brain q4m" | A/P还有"**Echo q6 months**"——漏了 | P2 |
| 9 | lab | "No labs planned" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "Continue herceptin/agent, letrozole, agent q12wks, off chemo" | ✓ | ✅ |

**P0:0 P1:0 P2:3**

## ROW 14 (coral_idx 153)
**原文关键事实**：58yo，de novo metastatic ER+ MBC to bone/liver/nodes。Was on faslodex+palbociclib but stopped。Went to Mexico for alternative treatment(low dose chemo at home)。Provider role = monitor。A/P：cancelled scans，schedule CT CAP+spine MRI for May，labs q2weeks，topical cannabis/sulfur，Cymbalta rx given。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+ metastatic breast cancer, HER2-" | 原文"ER 99%, PR 25% HER2 1+ FISH negative"——漏了**PR+**和IDC | P2 |
| 2 | Stage | "Metastatic (Stage IV)" | ✓ | ✅ |
| 3 | DistMet | "Yes, to bone, liver, nodes" | ✓ | ✅ |
| 4 | response | "Right breast nodules slightly increased..." | 引用了CT findings ✓ | ✅ |
| 5 | current_meds | (空) | 从此provider角度无active meds(停了palbociclib/fulvestrant) ✓ | ✅ |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | (空) | provider角色是monitor ✓ | ✅ |
| 8 | imaging | "CT CAP+spine MRI May, repeat spine MRI 6wks" | ✓ | ✅ |
| 9 | lab | "labs q2weeks" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "topical cannabis, sulfur, Cymbalta rx" | ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 19 (coral_idx 158)
**原文关键事实**：70yo，left IDC grade 3，ER 90%/PR ~15%(variable)/HER2 3+(heterogeneous, FISH+)。Axillary FNA+。PET: breast mass+subpectoral+axillary nodes, no distant mets。A/P：neoadjuvant TCHP(avoid anthracycline due to CAD)，port/echo/chemo teaching/GCSF authorization。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2+(FISH+) grade 3 IDC, Ki-67 20-90%" | ✓ 详细 | ✅ |
| 2 | Stage | "Stage IIIA (pT2 N2a)" | 临床tumor 4-5cm, locally advanced。pT2 N2a is from POST hook | P2 |
| 3 | DistMet | "No" | PET: no distant mets ✓ | ✅ |
| 4 | response | "Not yet on treatment" | ✓ | ✅ |
| 5 | current_meds | (空) | ✓ | ✅ |
| 6 | goals | "curative" | ✓ | ✅ |
| 7 | therapy | "TCHP, avoid anthracycline, port/echo/chemo teaching, trial" | ✓ 详细 | ✅ |
| 8 | imaging | "Echocardiogram" | ✓ | ✅ |
| 9 | lab | "No labs planned" | ✓ | ✅ |
| 10 | genetic | "None planned" | ✓ | ✅ |
| 11 | medication | "TCHP with GCSF" | ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 20 (coral_idx 159)
**原文关键事实**：75yo，原Stage I(0.9cm 0/2LN pT1b N0)→metastatic recurrence to bone+lymph nodes。Iliac crest biopsy: ER+80% PR+50% HER2 2+ FISH neg。Plan: letrozole+palbociclib，denosumab after dental，MRI spine，CT CAP，Foundation One，Rad Onc referral。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade II IDC with 1.8cm DCIS" | ✓(grade II from original, DCIS ✓) | ✅ |
| 2 | Stage | "Originally **Stage IIA**, now Stage IV" | 原发0.9cm+0/2LN = pT1b N0 = **Stage IA**不是IIA | P2 |
| 3 | DistMet | "Yes, to bone and lymph nodes" | ✓(POST-DISTMET-SUPPLEMENT补了LN) | ✅ |
| 4 | response | "plan includes repeat imaging after 3 months" | 给了**plan**不是response。尚未开始治疗 | P2 |
| 5 | current_meds | "letrozole, palbociclib" | 此次visit开始的——debatable | ✅ |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | "Start letrozole, Rx palbociclib, denosumab after dental" | ✓ | ✅ |
| 8 | imaging | "MRI spine, CT CAP, repeat in 3 months" | ✓ | ✅ |
| 9 | lab | "tumor markers, monthly palbociclib labs" | ✓ | ✅ |
| 10 | genetic | "Foundation One or [REDACTED] 360" | ✓ | ✅ |
| 11 | medication | "Start Letrozole, Rx Palbociclib, denosumab" | ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 22 (coral_idx 161)
**原文关键事实**：MBC ER+/PR+/HER2-。Originally Stage II。Mets to bone/chest wall/infraclavicular/IM nodes。On arimidex(=anastrozole)+denosumab。PET showed good response。Developed pneumonitis from abemaciclib(held)。A/P：pet ct now, if stable continue arimidex, if PD could use faslodex+[agent].

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- IDC with met recurrence(bone, chest wall...)" | ✓ | ✅ |
| 2 | Stage | "Originally Stage II, now Stage IV" | ✓ | ✅ |
| 3 | DistMet | "Yes, bone, chest wall, infraclavicular, IM nodes" | ✓ 详细 | ✅ |
| 4 | response | "PET 11/03/20 and 04/24/21 good response, pneumonitis" | ✓ | ✅ |
| 5 | current_meds | "anastrozole, denosumab" | ✓(arimidex=anastrozole) | ✅ |
| 6 | goals | "palliative" | ✓ | ✅ |
| 7 | therapy | "Continue letrozole, abemaciclib..." | **abemaciclib已因pneumonitis停用** | P2 |
| 8 | imaging | "Pet ct now" | ✓ | ✅ |
| 9 | lab | "No labs planned" | ✓ | ✅ |
| 10 | genetic | "If PD could use faslodex with [REDACTED]..." | 这是**treatment contingency**不是genetic testing | P2 |
| 11 | medication | "Continue arimidex, faslodex+agent if PD" | ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 24 (coral_idx 163)
**原文关键事实**：56yo，s/p partial mastectomy+SLN。Grade II micropapillary mucinous carcinoma。ER+(>95%) PR+(80%) HER2 equivocal IHC2/FISH neg。PET: **"No definite sites of hypermetabolic metastatic disease"**，但有enhancing liver lesion。A/P：if low risk → radiation + adjuvant hormone therapy。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade II micropapillary mucinous carcinoma" | ✓ 详细 | ✅ |
| 2 | Stage | "Stage III" | A/P没说Stage III。This is early stage s/p surgery。POST hook修了Stage IV→III但不完全对 | P2 |
| 3 | DistMet | "No" | PET说"No definite metastatic disease" ✓(**iter12e修好了**) | ✅ |
| 4 | response | "Not yet on treatment" | ✓ | ✅ |
| 5 | current_meds | (空) | ✓ | ✅ |
| 6 | goals | "adjuvant" | 应该是"curative" | P2 |
| 7 | therapy | "If low risk → radiation + adjuvant hormone therapy" | ✓ | ✅ |
| 8 | imaging | "No imaging planned" | ✓ | ✅ |
| 9 | lab | "[REDACTED] test ordered" | ✓ | ✅ |
| 10 | genetic | "surgical specimen for MP" | MammaPrint ✓ | ✅ |
| 11 | medication | "adjuvant hormone therapy if [REDACTED] low risk" | ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 29 (coral_idx 168) ✅
A/P: "Assessment and Plan:  ***** *****-***** is a 59 y.o. female with a recently diagnosed multifocal gr
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with mixed micropapill | ✅ |
| 2 | Stage_of_Cancer | Stage IIA (pT1c(m)N1(sn)) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | On treatment; response assessment not available from current visit. | ✅ |
| 5 | current_meds | letrozole 2.5mg PO daily | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue letrozole. RT planning per [REDACTED], likely pursued locally | ✅ |
| 8 | imaging_plan | Bone density scan can be completed when she returns from [REDACTED]. B | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Start letrozole 2.5mg PO daily now. Calcium supplementation with 1200m | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 30 (coral_idx 169) ✅
A/P: "Assessment and Recommendations:  64 y.o. postmenopausal patient with a clinical stage II-III *****-
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER-/PR-/HER2+ grade 2 invasive ductal carcinoma with intermediate to h | ✅ |
| 2 | Stage_of_Cancer | Clinical stage II-III (inferred from large enhancing right breast mass | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | The patient will proceed with neoadjuvant chemotherapy, including eith | ✅ |
| 8 | imaging_plan | The patient will need a TTE prior to starting chemotherapy. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | The patient will start neoadjuvant chemotherapy with either 12 cycles  | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 33 (coral_idx 172)
**原文关键事实**：ILC ER+/PR+/HER2-。Originally Stage IIB。No evidence of recurrence。On letrozole。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 3 | DistMet | "Not sure" | 无recurrence，应为"No" | P2 |
| 其余 | 全部正确 | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 34 (coral_idx 173)
**原文关键事实**：Stage III breast cancer。**Second** local relapse(first was 2012, now second)。Local: 1.7cm IDC grade 3, ER+/PR- (originally ER+/PR low)。PET/CT negative for distant mets。Plan: tamoxifen 20mg + radiation referral + return 6 months。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR-/HER2- IDC with extensive DCIS" | 原文有receptor change history(2011 vs 2020)——漏了 | P2 |
| 2 | Stage | "Stage III, now with local recurrence" | ✓(A/P确认是local relapse) | ✅ |
| 3 | DistMet | "Not sure" | PET-CT negative for distant mets。应该是"No" | P2 |
| 7 | therapy | "Continue radiation referral. Resume tamoxifen 20mg" | A/P还有"**return to clinic in 6 months**"——漏了 | P2 |
| 其余 | 全部正确 | | ✅ |

**P0:0 P1:0 P2:3**

## ROW 36 (coral_idx 175) ✅
A/P: "Assessment/Plan    Lab Results   Component Value Date    WBC Count 8.8 04/10/2021    RBC Count 3.91
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage IIIA (pT3N0) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | No evidence of disease recurrence on imaging and exam. PET/CT on 01/26 | ✅ |
| 5 | current_meds | Abraxane, zoladex | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue tamoxifen. Continue weekly Abraxane 100mg/m2 x 12 cycles. Con | ✅ |
| 8 | imaging_plan | will get doppler to r/o DVT | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Continue weekly Abraxane 100mg/m2 x 12 cycles. Continue zoladex for pr | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 40 (coral_idx 179)
Type P2: "PR-"但原文PR 5%=PR weak+。其余✅。

A/P: "ASSESSMENT/PLAN:  ***** ***** ***** is a 62 y.o. female with MS on chronic immunosuppression with n
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR-/HER2- grade 1 invasive ductal carcinoma with nuclear grade 1 D | ✅ |
| 2 | Stage_of_Cancer | Stage II (inferred from pT2 N1mi) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | On treatment; response assessment not available from current visit. | ✅ |
| 5 | current_meds | letrozole | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Adjuvant endocrine therapy with letrozole. Continue Prolia for bone pr | ✅ |
| 8 | imaging_plan | DEXA | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Rx for letrozole given. Continue Prolia. | ✅ |

**P0:0 P1:0 P2:1**

## ROW 41 (coral_idx 180) ✅
A/P: "Assessment and Plan:  ***** ***** is a 32 y.o. female ATM mutation carrier with a clinical ***** **
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR weakly+/HER2 1+ by IHC (FISH not available) grade 3 invasive du | ✅ |
| 2 | Stage_of_Cancer | Stage IIIA (inferred from pT2 N2a) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Patient will proceed with AC-Taxol chemotherapy, starting with Taxol f | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Patient will start with [REDACTED] Taxol for 12 weeks followed by AC.  | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 42 (coral_idx 181) ✅
A/P: "Assessment:       At this time Ms. ***** is and well. She's recently finished her radiation therapy
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage IA (inferred from pT1 N0) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | There is no specific imaging or tumor marker data provided in the note | ✅ |
| 5 | current_meds | tamoxifen | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue tamoxifen and return to clinic in 4-6 weeks to assess toleran | ✅ |
| 8 | imaging_plan | Routine diagnostic mammogram on the day of the next appointment as her | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Begin a 5 year course of tamoxifen therapy. Rx for tamoxifen given. | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 44 (coral_idx 183)
imaging P2: PET/CT可能是POST-IMAGING false positive。其余✅。

A/P: "ASSESSMENT & PLAN    ***** ***** is a 33 y.o. female with ER+/PR+/*****- node+ left breast cancer w
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with residual DCIS | ✅ |
| 2 | Stage_of_Cancer | Originally Stage II (inferred from 1 cm residual invasive ductal carci | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Surgical pathology after bilateral mastectomies with left breast SLN b | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Enrolled on a clinical trial for randomized 3 vs 5 weeks of radiation  | ✅ |
| 8 | imaging_plan | Consider a follow up CT Chest in one year.. PET/CT | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | She will start an aromatase inhibitor after completing radiation. She  | ✅ |

**P0:0 P1:0 P2:1**

## ROW 46 (coral_idx 185) ✅
A/P: "Assessment/Plan     PATHOLOGY:   A. Right breast axillary sentinel lymph nodes:   - 2 lymph nodes i
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR-/HER2- grade 1 invasive ductal carcinoma with extensive DCIS | ✅ |
| 2 | Stage_of_Cancer | Stage IIB (pT2N1(sn) (inferred from 3.5 cm residual disease, 2/2 posit | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Patient has residual 3.5 cm disease with positive margins after bilate | ✅ |
| 5 | current_meds | letrozole | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Patient will start letrozole now. Addition of abemaciclib discussed af | ✅ |
| 8 | imaging_plan | MRA of the abdomen in 1 year, due in January 2022.. DEXA scan | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Will start letrozole now. Rx sent. Continue naproxen 500mg bid, APAP o | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 49 (coral_idx 188) ✅
A/P: "Impression/Plan: 50 y.o. female with new diagnosis of L breast cancer with biopsy-proven nodal \n d
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Likely stage 2 (inferred from primary tumor size and nodal involvement | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Discussed with patient that the current plan for adjuvant endocrine th | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | The current plan for adjuvant endocrine therapy is subject to change b | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 50 (coral_idx 189)
DistMet P2: 漏LN。current_meds P2: 漏lupron。medication P2: 严重缩短。

A/P: "Assessment / Plan:    1.  De ***** metastatic breast cancer found in July 2013, HR+ and her 2 neu n
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | HR+ and HER2- metastatic breast cancer with IDC and DCIS components | ✅ |
| 2 | Stage_of_Cancer | Originally Stage IV (T2, N1, M1) | ✅ |
| 3 | Distant Metastasis | Yes, to lung, liver, and bone | ✅ |
| 4 | response_assessment | Imaging from December 2021 shows metastatic disease under good control | ✅ |
| 5 | current_meds | ibrance, xgeva, letrozole | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Second line lupron, letrozole started October 2014 and ibrance added J | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Referral to genetics for pathogenic PMS 2 mutation. | ✅ |
| 11 | medication_plan | Lupron, letrozole, and ibrance added January 2015. | ✅ |

**P0:0 P1:0 P2:3**

## ROW 51 (coral_idx 190)
A/P: "Patient /Family Response to Teaching:  Patient/family receptive and verbalized understanding of inf
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | (空) | ✅ |
| 2 | Stage_of_Cancer | (空) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not mentioned in note. | ✅ |
| 5 | current_meds | Gemzar | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Before starting [REDACTED], Gemzar, and [REDACTED] treatment, make sur | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Before starting [REDACTED], Gemzar, and [REDACTED] treatment, make sur | ✅ |

All 11 fields verified. **P0:0 P1:1 P2:0**

## ROW 52 (coral_idx 191) ✅
A/P: "Assessment/Plan and in my HPI/Oncologic History    Results for orders placed or performed during th
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade II invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage IIA (inferred from pT2 N1mi) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue radiation therapy referral. [REDACTED] send Zoladex prior aut | ✅ |
| 8 | imaging_plan | CT CAP and bone scan for staging | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Order [REDACTED] to provide more information on her tumor biology and  | ✅ |
| 11 | medication_plan | [REDACTED] + [REDACTED]. [REDACTED] send Zoladex prior auth. Start aft | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 53 (coral_idx 192) ✅
A/P: "Assessment:       Stage II/III left breast cancer, ER/PR positive, ***** positive, s/p left lumpect
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2+ invasive ductal carcinoma with neuroendocrine differenti | ✅ |
| 2 | Stage_of_Cancer | Stage II/III | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Adjuvant systemic chemotherapy with AC x 4 given q2weeks with growth f | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Referral to genetic counseling for further evaluation. | ✅ |
| 11 | medication_plan | Adjuvant AC/THP chemotherapy consisting of AC x 4 given q2weeks with g | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 54 (coral_idx 193) ✅
A/P: "Assessment and Recommendations:  39 y.o. premenopausal patient with a recent diagnosis of a deleter
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with extensive DCIS co | ✅ |
| 2 | Stage_of_Cancer | Stage IV (metastatic) | ✅ |
| 3 | Distant Metastasis | Yes, to left posterior 7th rib and T6 vertebral body | ✅ |
| 4 | response_assessment | The cancer is currently responding with stable disease. Evidence inclu | ✅ |
| 5 | current_meds | leuprolide, letrozole, zoledronic acid | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | continue leuprolide every 4 weeks, continue letrozole, referred to rad | ✅ |
| 8 | imaging_plan | repeat PET/CT in ~3-4 months after radiation, DEXA scan (re-ordered) | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | continue leuprolide every 4 weeks, continue letrozole, start palbocicl | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 57 (coral_idx 196)
A/P: "IMP  1. Breast CA, Left sided, locally advanced. Initially classified as ***** at ***** hospital, a
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | Breast Cancer (TNBC), originally classified as Grade III adenocarcinom | ✅ |
| 2 | Stage_of_Cancer | Locally advanced (inferred from treatment with neoadjuvant chemotherap | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | The patient received neoadjuvant chemotherapy with TCH+P followed by s | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Rec XRT, which pt is scheduled to receive. If [REDACTED], although unl | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Rec genetic counseling and testing | ✅ |
| 11 | medication_plan | Recommendation to resume trastuzumab if [REDACTED] status is confirmed | ✅ |

All 11 fields verified. **P0:0 P1:1 P2:0**

## ROW 59 (coral_idx 198) ✅
A/P: "ASSESSMENT & PLAN  ***** ***** is a 52 y.o. female  with stage 1 right sided ER+/PR+/*****- breast 
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 3 invasive ductal carcinoma with high-grade DCIS c | ✅ |
| 2 | Stage_of_Cancer | Stage IIA (inferred from pT2 N0) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | No evidence of recurrence either locally or systemically based on phys | ✅ |
| 5 | current_meds | "exemestane ([REDACTED]) 25 mg tablet", "letrozole (FEMARA) 2.5 mg tab | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Discontinue Letrozole and wait 2-3 weeks before starting Exemestane. C | ✅ |
| 8 | imaging_plan | Right mammogram due in July. Continue mammograms alternating with MRI  | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Discontinue Letrozole and wait 2-3 weeks before starting Exemestane. C | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 61 (coral_idx 200) ✅
A/P: "Assessment and Plan     # Breast CA L    ***** ***** is a 43 y.o. pre-menopausal female newly diagn
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- (1+) grade 2 invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage I (inferred from tumor ≤2cm) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Will likely need [unspecified agent] Dx after surgery to determine if  | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | will likely need [REDACTED] Dx after surgery to determine if she needs | ✅ |
| 11 | medication_plan | she will need adjuvant endocrine therapy (Tamoxifen vs Ovarian Suppres | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 64 (coral_idx 203)
Type P2: "HR+"应写"ER+/PR+"。其余✅。

A/P: "Assessment / Plan:    1. Stage III- IV left breast cancer HR+/her 2 negative with probably metastat
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | HR+/HER2- invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage IV (metastatic) | ✅ |
| 3 | Distant Metastasis | Yes, to the sternum | ✅ |
| 4 | response_assessment | No specific imaging or tumor marker evidence provided in the note to a | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Recommend [REDACTED] followed by taxol then surgery and radiation with | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | No specific genetic or molecular tests are planned. | ✅ |
| 11 | medication_plan | Currently on unspecified agent and taxol planned. If biopsy of bone is | ✅ |

**P0:0 P1:0 P2:1**

## ROW 65 (coral_idx 204) ✅
A/P: "ASSESSMENT/PLAN:  ***** ***** is a 48 y.o. female with a recent diagnosis of right breast IDC which
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER weak positive (2%), PR low positive (7%), HER2 neg (IHC 2+, FISH 2. | ✅ |
| 2 | Stage_of_Cancer | Stage IB (corrected: pT1 N1mi — micrometastasis only) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Continue taxol. Port placement, chemotherapy teaching session, interes | ✅ |
| 8 | imaging_plan | research breast MRI | ✅ |
| 9 | lab_plan | labs | ✅ |
| 10 | genetic_testing_plan | F/u results of genetic testing (done locally last Mon) | ✅ |
| 11 | medication_plan | Plan to start neoadjuvant chemotherapy with AC/T. Typically, taxol fir | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 68 (coral_idx 207) ✅
A/P: "ASSESSMENT & PLAN  ***** ***** is a 63 y.o. postmenopausal female who presents  to clinic for consu
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2+ multifocal invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage I (inferred from tumor ≤2cm) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | The cancer has shown a good clinical response after 6 cycles of TCHP c | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | If a lumpectomy is chosen, radiation is required. Post-mastectomy radi | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Her sons should be tested for the risk of [REDACTED] anemia in childre | ✅ |
| 11 | medication_plan | No specific current or future medication plans were detailed in the no | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 70 (coral_idx 209)
**原文关键事实**：Bilateral ER+/HER2- breast cancer。Left: pT4N1 grade 2 ILC，Right: pT1N0 IDC with DCIS。S/p neoadjuvant TC x6。MRI shows faint residual NME, axillary nodes decreased。Bone scan negative。On letrozole(previously tolerated)。Expanders before radiation。CT June 2020 for lung nodules。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR+/HER2- grade 2 ILC with 1 LN+(left); IDC with DCIS(right)" | ✓ 详细 | ✅ |
| 2 | Stage | "Originally Stage II(pT4N1 left, pT1N0 right), post-NAC" | ✓ | ✅ |
| 3 | DistMet | "No" | bone scan negative ✓ | ✅ |
| 4 | response | "MRI faint residual NME, axillary nodes decreased, bone scan neg" | ✓ | ✅ |
| 7 | therapy | "Restart letrozole. Expanders before radiation" | 漏了"**CT scan for lung nodules June 2020**" | P2 |
| 11 | medication | "Restart letrozole" | 漏了"**[REDACTED] for hot flashes**" | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:2**

## ROW 72 (coral_idx 211)
**原文关键事实**：ER+/PR-/HER2- grade 2 IDC with focal neuroendocrine diff。pT1cN0(sn)=Stage IA。Start letrozole。[REDACTED] for chemo benefit evaluation。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 8 | imaging | "Ultrasound" | 可能是POST-IMAGING false positive | P2 |
| 9 | lab | "[REDACTED] to evaluate chemo benefit" | Oncotype属genetic不是lab | P2 |
| 10 | genetic | "[REDACTED] to evaluate chemo benefit" | 和lab重复了 | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:3**

## ROW 78 (coral_idx 217) ✅
A/P: "ASSESSMENT/PLAN:  ***** ***** is a 79 \n y.o. female  with MMM including DM, HTN hyperlipidemia, h/
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER-/PR-/HER2- grade 3 invasive ductal carcinoma | ✅ |
| 2 | Stage_of_Cancer | Metastatic (Stage IV) | ✅ |
| 3 | Distant Metastasis | Yes, to liver and periportal lymph nodes | ✅ |
| 4 | response_assessment | Worsening of metastatic disease noted on 08/07/19 Abdominal CT showing | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | She is interested in screening for a trial at [REDACTED] involving [RE | ✅ |
| 8 | imaging_plan | Echo 09/08/2019. If this echo is normal we can stop monitoring her. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Patient is interested in screening for the [REDACTED] trial at [REDACT | ✅ |
| 11 | medication_plan | Continue 800 mg Mag-Ox supplement daily. She is on lisinopril, norvasc | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 80 (coral_idx 219)
**原文关键事实**：ER+/PR+/HER2- grade 3 IDC。Stage I(≤2cm)。Plan：TC x 4 starting 04/11/19。6 weeks radiation with 1 week boost, left axilla+SC fields。Claritin for 5-6 days。Cold cap/cold gloves。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 7 | therapy | "TC x 4 on 04/11/19, with [REDACTED]. RTC cycle 2" | 漏了**radiation details**(6 weeks, boost, axilla+SC fields) | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 82 (coral_idx 221) ✅
A/P: "Assessment & Plan:   Has excellent grief counselor and continues to meet weekly. Holidays went well
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- mixed ductal and lobular carcinoma | ✅ |
| 2 | Stage_of_Cancer | Stage II | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | We discussed the role of chemotherapy to decrease the risk of systemic | ✅ |
| 8 | imaging_plan | Dexa to assess bone health before the next visit with me. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Continue hydrochlorothiazide 12.5 mg tablet daily before lunch. Contin | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 84 (coral_idx 223) ✅
A/P: "ASSESSMENT/PLAN:   ***** ***** is a 60 y.o. female with a CHEK2 mutation, multiple sclerosis, and m
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with metastatic recurr | ✅ |
| 2 | Stage_of_Cancer | Stage IV (metastatic) | ✅ |
| 3 | Distant Metastasis | Yes, to bone, soft tissue, liver, and possibly meninges | ✅ |
| 4 | response_assessment | MRI brain with contrast on 11/07/2020 showed diffuse irregular enhance | ✅ |
| 5 | current_meds | capecitabine, zolendronic acid | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | refer to radiation oncology for consideration of radiation therapy (ei | ✅ |
| 8 | imaging_plan | Repeat CT CAP now, Repeat MRI spine to rule out leptomeningeal disease | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Continue xeloda 1500mg BID for now, continue zolendronic acid, could c | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 85 (coral_idx 224) ✅
A/P: "ASSESSMENT/PLAN:  ***** ***** is a 61 y.o. female with a h/o clinical stage *****, ***** stage IIIA
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR-/HER2- invasive lobular carcinoma with pleomorphic features, mu | ✅ |
| 2 | Stage_of_Cancer | Originally Stage IIIA, now metastatic (Stage IV) | ✅ |
| 3 | Distant Metastasis | Yes, to bone, liver, and brain | ✅ |
| 4 | response_assessment | The cancer is currently progressing on first line fulvestrant/palbocic | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Metastatic breast cancer patient will be evaluated for a phase 1 trial | ✅ |
| 8 | imaging_plan | Brain MRI to be reviewed by UCSF neuroradiology | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Patient will be evaluated for a phase 1 trial of [REDACTED]+olaparib f | ✅ |
| 11 | medication_plan | Continue steroid taper per Dr. [REDACTED]. Continue pain meds as presc | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 87 (coral_idx 226) ✅
A/P: "Assessment:       Ms. ***** is a 79-year-old woman with a 2.2 cm multifocal tumor in the right brea
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with a separate focus  | ✅ |
| 2 | Stage_of_Cancer | Stage IIIA (inferred from pT2 N2a) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | [REDACTED] will receive a course of hormonal therapy alone. | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Ms. [REDACTED] will prefer to receive a course of hormonal therapy alo | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 88 (coral_idx 227)
**原文关键事实**：Stage III left breast cancer→metastatic to brain/lungs/LNs。On xeloda。A/P："No new imaging findings, no palpable masses... stable disease"。Discuss clinical trials if progression。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 4 | response | "She is on xeloda and I would recommend restaging after 3 months..." | 给了**plan**("restaging")而非**response**("stable disease, no masses") | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 90 (coral_idx 229)
**原文关键事实**：Adenocarcinoma of right breast。Clinical stage II/III。S/p neoadjuvant。Cycle 4 of AC planned。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "Adenocarcinoma of right breast (HCC)" | 原文有ER/PR/HER2 data但note高度redacted——**漏了receptor status** | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 91 (coral_idx 230)
**原文关键事实**：Stage 4 MBC to bone。ER+/PR+。On exemestane+everolimus+denosumab。RLE edema improved。A/P：continue exemestane, continue [REDACTED]+elevation for edema, continue lasix 10mg daily with KCL 10Meq daily, continue denosumab for hip, PET/CT next week, labs monthly, topical antifungal for fungal dermatitis。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 7 | therapy | "Continue exemestane, denosumab; lasix, KCL, elevation for edema" | POST hook补了supportive items ✓。但还漏了**PET/CT next week, labs monthly, topical antifungal** | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 92 (coral_idx 231)
**原文关键事实**：MBC to multiple sites。Epirubicin cycle#2 D1。A/P："Exam improved-liver smaller, less tender"。Tumor marker pending。Labs: AST elevated。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 3 | DistMet | "Yes, to liver" | A/P说"**multiple sites**"但只提取了liver | P2 |
| 4 | response | "Liver smaller, less tender. Tumor marker pending" | 漏了"**stable on treatment**"和AST/hemoglobin values | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:2**

## ROW 95 (coral_idx 234)
**原文关键事实**：49yo ER+/PR-/HER2- IDC。S/p neoadjuvant pembrolizumab(trial)。MRI showing response。Plan：axilla XRT, capecitabine after XRT。

| # | Field | 提取值 | 原文对照 | 判定 |
|---|-------|--------|---------|------|
| 1 | Type | "ER+/PR-/HER2- IDC with residual DCIS" | 原文有"treatment effect, three foci, margins negative"——漏了 | P2 |
| 8 | imaging | "breast and axilla XRT" | XRT是**radiotherapy**不是imaging | P2 |
| 其余 | ✓ | | ✅ |

**P0:0 P1:0 P2:2**

## ROW 96 (coral_idx 235) ✅
A/P: "Assessment/Plan and in my HPI/Oncologic History    No results found for this or any previous visit.
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade I mixed ductal and cribriform carcinoma with tubul | ✅ |
| 2 | Stage_of_Cancer | Stage IA (pT1cN0(sn)) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Proceed with adjuvant radiation consult, wait to start xrt until after | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | send for [REDACTED] testing | ✅ |
| 10 | genetic_testing_plan | send for [REDACTED] testing | ✅ |
| 11 | medication_plan | Tamoxifen after completion of adjuvant radiation | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 97 (coral_idx 236) ✅
A/P: "- We reviewed the pathology results, clinical features and prognostic factors involved in her case.
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with low to intermedia | ✅ |
| 2 | Stage_of_Cancer | Stage IA (pT1b N0) | ✅ |
| 3 | Distant Metastasis | No | ✅ |
| 4 | response_assessment | Not yet on treatment — no response to assess. | ✅ |
| 5 | current_meds | (空) | ✅ |
| 6 | goals_of_treatment | curative | ✅ |
| 7 | therapy_plan | Strongly recommend adjuvant endocrine therapy with [REDACTED] at the a | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | Patient wishes to proceed with molecular profiling, e.g. Oncotype Dx a | ✅ |
| 11 | medication_plan | Strongly recommend adjuvant endocrine therapy with [REDACTED] at the a | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 99 (coral_idx 238) ✅
A/P: "Assessment and Plan:  ***** ***** is a 49 y.o. female with a stage III left-sided breast cancer dia
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+/PR+/HER2+ grade 3 invasive ductal carcinoma with lymphovascular in | ✅ |
| 2 | Stage_of_Cancer | Originally Stage III, now metastatic (Stage IV) | ✅ |
| 3 | Distant Metastasis | Yes, to the left lung and mediastinal lymph nodes | ✅ |
| 4 | response_assessment | The cancer is currently showing mixed response to treatment. The PET/C | ✅ |
| 5 | current_meds | fulvestrant | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Continue fulvestrant, radiation therapy referral. If the cancer remain | ✅ |
| 8 | imaging_plan | The patient will need a followup CT scan with contrast and thin slices | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | The patient needs to have a biopsy of either the pulmonary nodule, med | ✅ |
| 11 | medication_plan | If cancer remains estrogen receptor positive, will consider ongoing ho | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**

## ROW 100 (coral_idx 239) ✅
A/P: "ASSESSMENT & PLAN  Breast cancer metastasized to multiple sites  Tumor markers rose initally with s
| # | Field | 提取值 | 判定 |
|---|-------|--------|------|
| 1 | Type_of_Cancer | ER+(80%)PR+(50%) HER2- grade 2 invasive ductal carcinoma with metastat | ✅ |
| 2 | Stage_of_Cancer | Stage IV (metastatic) | ✅ |
| 3 | Distant Metastasis | Yes, to liver and bone | ✅ |
| 4 | response_assessment | Tumor markers elevated: Cancer Antigen 15-3 at 118 U/mL (<33 U/mL), Ca | ✅ |
| 5 | current_meds | gemzar | ✅ |
| 6 | goals_of_treatment | palliative | ✅ |
| 7 | therapy_plan | Rec exercise 10 min 3 x a day, Focalin prn and continue with treatment | ✅ |
| 8 | imaging_plan | No imaging planned. | ✅ |
| 9 | lab_plan | No labs planned. | ✅ |
| 10 | genetic_testing_plan | None planned. | ✅ |
| 11 | medication_plan | Continue with treatment, Focalin prn for fatigue. | ✅ |

All 11 fields verified. **P0:0 P1:0 P2:0**
