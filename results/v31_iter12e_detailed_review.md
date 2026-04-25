# V31 iter12e — 逐字逐句原文对照审查

> 56 samples × 11 fields = 616 field-instances
> 每个 field 写出：提取了什么 + 原文说了什么 + 对不对

## 状态
- 审查中: ROW 3+
- 已完成: 2/56
- P0: 0, P1: 0, P2: 7

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

