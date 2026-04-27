# V31 iter13 — Extraction + Letter 逐字段+逐句审查

> 56 samples，每个 sample 审查：
> 1. Extraction 每个字段 vs 原文
> 2. Letter 每句 vs 原文
> P0=幻觉/编造 | P1=重大错误 | P2=小问题

## 状态
- **✅ 全部完成: 56/56**
- Extraction: P0:0 **P1:1** P2:26
  - ROW1 x2: imaging 漏 bone scan + lab field 混乱
  - ROW2: lab_summary 漏 Na/K
  - ROW4: Brain MRI conditional
  - ROW6 x2: Patient type 错 + genetics 历史
  - ROW7: therapy_plan PT 幻觉
  - ROW8: procedure 漏 port
  - ROW10 x2: LN "20" redacted 误读
  - ROW11: imaging 漏 MRI
  - ROW12: imaging 漏 echo
  - ROW20 x4: Stage IA/IIA + Metastasis 不一致 + old lab + procedure garbled
  - ROW22 x2: POST hook adds stopped meds (letrozole, abemaciclib)
  - ROW33: Stage "now III" for no-recurrence
  - ROW84: PR+ 错 (metastatic biopsy PR-)
  - ROW85 x2: prednisone 剂量缺失 + truncation/garbled
- Letter: P0:0 P1:0 P2:22
  - ROW1: 漏 peritoneum
  - ROW3: "a medication" garbled
  - ROW4: Prolia 漏 "every"
  - ROW6: genetics 已完成
  - ROW7 x3: garbled x2 + PT 幻觉
  - ROW11: jaw "cancer has grown" 误导
  - ROW12: morphine/oxycodone 但患者已不用
  - ROW19: TCHP description garbled
  - ROW20: 漏 denosumab/dental
  - ROW24: SLN micrometastasis 未提及
  - ROW29: MammaPrint/no-chemo 未提及
  - ROW33: MRI brain 原因误解 (headaches→medication)
  - ROW50: medication timing (Oct 2014 vs Jan 2015)
  - ROW72: "a medication" for Oncotype garbled
  - ROW80: cold gloves "hand swelling" (应为 neuropathy)
  - ROW85 x2: truncation + garbled
  - ROW88: truncation
  - ROW91: 漏 everolimus
  - ROW11: jaw "cancer has grown" 误导

---

## ROW 1 (coral_idx 140)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "New Patient Evaluation" ✓ | ✅ |
| second opinion | no | 无 second opinion 提及 ✓ | ✅ |
| in-person | in-person | PE 完整 ✓ | ✅ |
| summary | 56-year-old...initial consult for newly diagnosed metastatic ER+/PR+ breast cancer | HPI: Stage IIA 2013, now metastatic to lungs/peritoneum/liver/ovary ✓ | ✅ |
| Type_of_Cancer | ER+/PR+ grade 2 IDC with metastatic recurrence, HER2- | "G2, ER and PR positive and her 2 neu negative" ✓ | ✅ |
| Stage_of_Cancer | Originally Stage IIA, now metastatic (Stage IV) | "multifocal Stage IIA" + now widespread mets ✓ | ✅ |
| Distant Metastasis | Yes, to lungs, peritoneum, liver, ovary | HPI: "involvement of lungs, peritoneum, liver and ovary" ✓ | ✅ |
| Metastasis | Yes, to lungs, peritoneum, liver, ovary | 同上 ✓ | ✅ |
| lab_summary | No labs in note. | "No visits with results within 1 Month(s)" ✓ | ✅ |
| findings | CT impression + PE findings (detailed) | CT 8 点全部准确覆盖 + PE hepatomegaly/omental masses/axillary mass ✓ | ✅ |
| current_meds | "" | "No current outpatient medications on file" ✓ | ✅ |
| recent_changes | "" | New patient, no changes ✓ | ✅ |
| supportive_meds | "" | None ✓ | ✅ |
| goals_of_treatment | palliative | "treatment would be palliative" ✓ | ✅ |
| response_assessment | CT findings showing widespread metastases | CT 12/24/2019 ✓ | ✅ |
| medication_plan | If HR+/HER2 negative on biopsy, will recommend ibrance and unspecified agent | A/P #4 ✓ | ✅ |
| therapy_plan | ibrance and [REDACTED] | 同上 ✓ | ✅ |
| radiotherapy_plan | None | 无 radiation 讨论 ✓ | ✅ |
| procedure_plan | biopsy mass in right axilla | A/P #3 ✓ | ✅ |
| imaging_plan | Brain MRI | 原文: "ordered a MRI of brain **and bone scan**"——**漏了 bone scan** | P2 |
| lab_plan | ordered a MRI of brain and bone scan as well as labs | 把 bone scan 放到了 lab_plan 里，field 分配不当 | P2 |
| genetic_testing_plan | biopsy to confirm HR and HER2 status | 合理——biopsy 是 receptor confirmation 不是 genetic testing ✓ | ✅ |
| Next clinic visit | after completed work up | A/P #4 "RTC after completed work up" ✓ | ✅ |
| Advance care | Full code | "Full code" ✓ | ✅ |
| Referral: Specialty | Integrative Medicine referral | "Ambulatory Referral to Integrative Medicine" ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:2（imaging_plan 漏 bone scan + lab_plan field 分配混乱）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "first visit regarding your breast cancer treatment" | "New Patient Evaluation" ✓ | ✅ |
| "invasive ductal carcinoma...started in the milk ducts" | IDC ✓ | ✅ |
| "spreading to other parts of your body, including your lungs, liver, and ovaries" | 原文 "lungs, peritoneum, liver and ovary"。**漏了 peritoneum** | P2 |
| "also a new growth near your right armpit" | "local recurrence near the right axilla" ✓ | ✅ |
| "biopsy confirms...HER2...sensitive to hormones...ibrance and another unspecified medication" | A/P #4 ✓ | ✅ |
| "biopsy of the mass in your right armpit" | A/P #3 ✓ | ✅ |
| "brain MRI and a bone scan" | 原文 "ordered a MRI of brain and bone scan" ✓ | ✅ |
| "Integrative Medicine center for additional support" | "Integrative Medicine Referral" ✓ | ✅ |
| "follow-up visit after these tests are completed" | A/P #4 "RTC after completed work up" ✓ | ✅ |
| "full code...all possible life-saving measures" | "Full code" ✓ | ✅ |
| Emotional support sentence | "She is distressed" ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（漏 peritoneum）

### ROW 1 总评
- **Extraction**: P0:0 P1:0 P2:2
- **Letter**: P0:0 P1:0 P2:1
- 无幻觉，无编造。Letter 正确包含了 bone scan（extraction 漏了但 letter 从 lab_plan 补回了）。

## ROW 2 (coral_idx 141)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | C3D1 irinotecan ✓ | ✅ |
| summary | 44 y.o....cycle 3 day 1 of irinotecan | HPI ✓ | ✅ |
| Type_of_Cancer | TNBC with metastatic recurrence | "ER-/PR-/*****- (TNBC), grade 3" ✓ | ✅ |
| Stage_of_Cancer | Originally Stage IIB, now metastatic (Stage IV) | "clinical Stage IIB" + now metastatic ✓ | ✅ |
| Distant Metastasis | Yes, to liver, bone, chest wall, possibly intracranial/skull base | PET/CT 05/31/19 ✓ | ✅ |
| lab_summary | Albumin 2.1, Alk Phos 183, WBC 10.4, Hgb 7.7, Hct 23.6 | ✓ 但**漏了 Sodium 124 (LL), Potassium 3.1 (L)**——这些是临床关键异常，A/P 专门处理了 hyponatremia/hypokalemia | P2 |
| findings | chest wall infection + back pain + MRI spine lesions + Hep B + neuropathy + electrolytes | A/P 各项 ✓（findings 里有 "hyponatremia and hypokalemia" 补偿了 lab_summary 的遗漏） | ✅ |
| current_meds | irinotecan | 当前 cancer treatment ✓ | ✅ |
| recent_changes | irinotecan q2wk increased dose 150mg/m2 | A/P ✓ | ✅ |
| supportive_meds | ondansetron, compazine, imodium, oxycodone | medication list ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | MRI spine findings + no imaging evidence for current treatment response | 合理——scans planned September ✓ | ✅ |
| medication_plan | [REDACTED] 30mg TID, Flexeril TID, oxycodone, Effexor-XR to 75mg, doxycycline | A/P ✓ | ✅ |
| radiotherapy_plan | Radiation Oncology consultation needed | A/P "urgently needs to get in with Rad Onc" ✓ | ✅ |
| imaging_plan | Scans in 3 months + MRI brain if worse | A/P ✓ | ✅ |
| lab_plan | ALT/HBsAg/HBV DNA q4 months | A/P ✓ | ✅ |
| Next clinic visit | 2 weeks | A/P "F/u 2 weeks" ✓ | ✅ |
| Referral: Specialty | Radiation oncology | ✓ | ✅ |
| Referral: Others | Social work | A/P "SMS/ABC referral" ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（lab_summary 漏 Na 124/K 3.1 关键电解质异常）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | C3D1 ✓ | ✅ |
| "chest area is more tender, red, and swollen, might be due to an infection" | A/P "worrisome for infection" ✓ | ✅ |
| "back pain has gotten worse, possibly because the cancer has grown" | A/P "could be due to PD" ✓ | ✅ |
| "MRI...cancer has spread to your bones, especially around your S1 vertebra" | A/P MRI spine ✓ | ✅ |
| "low levels of sodium and potassium" | A/P "Hyponatremia and hypokalemia" ✓ | ✅ |
| "anemia...caused by the cancer and chemotherapy, has become worse" | A/P "Anemia due to malignancy and chemotherapy, worse" ✓ | ✅ |
| "history of exposure to hepatitis B, but...no active infection" | A/P "Positive Hep B Core Ab...HBV DNA is negative" ✓ | ✅ |
| "nerve pain from a previous medication has improved" | A/P "Peripheral neuropathy due to abraxane, improved" ✓ | ✅ |
| "chest pain from cancer and infection has gone away after taking antibiotics" | A/P "resolved s/p ABX" ✓ | ✅ |
| "lower back pain from cancer has gotten worse" | A/P "Sacral pain due to metastases, worse" ✓ | ✅ |
| "irinotecan...changed to every other week...dose increased" | A/P ✓ | ✅ |
| "medications for nausea, pain, and restless leg syndrome" | ondansetron/compazine/oxycodone/cymbalta ✓ | ✅ |
| "Effexor-XR dosage has been increased" | A/P "Increase effexor-XR to 75 mg qd" ✓ | ✅ |
| "doxycycline for 7 days to treat the infection" | A/P "Rx Doxycycline 100mg BID x 7d" ✓ | ✅ |
| "referred to Radiation Oncology" | A/P ✓ | ✅ |
| "scans again in 3 months, due in September 2019" | A/P ✓ | ✅ |
| "MRI of your brain if your symptoms get worse" | A/P "MRI brain if worse" ✓ | ✅ |
| "monitored with ALT, hepatitis B surface antigen, and HBV DNA every 4 months" | A/P ✓——保留了具体 test name ✓ | ✅ |
| "follow-up visit in 2 weeks" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——优秀的 letter，每句有出处，保留了具体 lab test name

### ROW 2 总评
- **Extraction**: P0:0 P1:0 P2:1
- **Letter**: P0:0 P1:0 P2:0

## ROW 3 (coral_idx 142)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "New Patient Evaluation" ✓ | ✅ |
| second opinion | yes | "several opinions and is here for a medical oncology consult" ✓ | ✅ |
| in-person | Televisit | "Video Consult" ✓ | ✅ |
| summary | 53 y.o....neoadjuvant therapy recommendations | HPI ✓ | ✅ |
| Type_of_Cancer | HR+, HER2- grade 2 IDC | "HR+, her 2 2+, fish negative" = HER2- ✓ | ✅ |
| Stage_of_Cancer | Stage IIA | "Clinical: Stage IIA" ✓ | ✅ |
| Distant Metastasis | No | PET pending, no known mets ✓ | ✅ |
| lab_summary | No labs in note | "No results found for any previous visit" ✓ | ✅ |
| findings | 1.7cm tumor, 1.5cm LN+, pending PET/genetics | HPI ✓ | ✅ |
| current_meds | "" | "No current outpatient medications on file" ✓ | ✅ |
| goals_of_treatment | curative | Stage IIA ✓ | ✅ |
| response_assessment | Not yet on treatment | No treatment started ✓ | ✅ |
| therapy_plan | Discussed chemo + surgery + radiation roles | A/P #2-4 ✓ | ✅ |
| imaging_plan | PET scan follow up | PET CT scheduled ✓ | ✅ |
| genetic_testing_plan | Genetic testing sent and is pending | A/P #6 ✓ | ✅ |
| Next clinic visit | after PET and [REDACTED] are back | A/P #7 ✓ | ✅ |
| Advance care | full code | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "medical oncology consult regarding your newly diagnosed breast cancer" | ✓ | ✅ |
| "small cancer...1.7 cm...positive for estrogen receptors...no HER2" | HPI ✓ | ✅ |
| "small lymph node in your armpit that has cancer" | biopsy-proven LN+ ✓ | ✅ |
| "waiting for the results of a PET scan and genetic testing" | ✓ | ✅ |
| "cure the cancer" | curative ✓ | ✅ |
| "chemotherapy to reduce the chance of the cancer spreading" | A/P #2 ✓ | ✅ |
| "surgery and possibly radiation" | A/P #3 ✓ | ✅ |
| "PET scan to get more information...Genetic testing ordered and pending" | A/P ✓ | ✅ |
| "referred to for a follow-up visit after the results of the PET scan and **a medication** are back" | A/P: "follow up after pet and ***** are back"（*****=MammaPrint results）。**"a medication"是 garbled**，应为"a test" | P2 |
| "full code" | ✓ | ✅ |
| Emotional support | "She has good support" ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（"a medication are back" garbled from redaction）

### ROW 3 总评
- **Extraction**: P0:0 P1:0 P2:0
- **Letter**: P0:0 P1:0 P2:1
- A/P 中的 lifestyle modifications（diet, exercise, stress, sleep）未在 extraction 中单独记录，但不属于标准提取字段

## ROW 4 (coral_idx 143)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | on letrozole since 2016 ✓ | ✅ |
| summary | 75 y.o....follow-up care | HPI ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC | pathology: 2.8cm grade 2 IDC, HER2 2+ IHC FISH neg ✓ | ✅ |
| Stage_of_Cancer | Not mentioned in note | staging entry redacted, honest answer ✓ | ✅ |
| Distant Metastasis | No | no recurrence ✓ | ✅ |
| lab_summary | No labs in note | external scanned docs only ✓ | ✅ |
| findings | No recurrence + DEXA results (T-score -2.4 improved) | A/P ✓ | ✅ |
| current_meds | letrozole | cancer treatment med ✓ | ✅ |
| goals_of_treatment | curative | adjuvant ✓ | ✅ |
| response_assessment | No evidence of disease recurrence | A/P #1 ✓ | ✅ |
| medication_plan | letrozole + magnesium + calcium/VitD + Prolia conditional + probiotics | A/P #1-7 ✓ 全面 | ✅ |
| imaging_plan | mammogram July 2019 + DEXA July 2019 + **Brain MRI** | A/P: Brain MRI 是**conditional** "if worsening"，不是 planned | P2 |
| follow_up | 6 months or sooner | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（Brain MRI 是 conditional 不是 planned）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "no sign of the cancer coming back" | A/P "without any evidence of disease recurrence" ✓ | ✅ |
| "bone density has slightly improved in the femur area, but still in the osteopenia range" | A/P #5 T-score -2.4 ✓ | ✅ |
| "right breast looks normal with no lumps or masses" | PE ✓ | ✅ |
| "continue to take Letrozole" | A/P #1 ✓ | ✅ |
| "magnesium supplements for occasional muscle cramps" | A/P #2 ✓ | ✅ |
| "continue taking calcium and vitamin D supplements" | A/P #5 ✓ | ✅ |
| "If your bone density score falls below -2.5, you might start on Prolia **6 months**" | A/P "Prolia 60 mg SQ **every** 6 months"——漏了"every" | P2 |
| "daily probiotics for loose stools" | A/P #7 ✓ | ✅ |
| "annual mammogram...July 2019" | A/P #1 ✓ | ✅ |
| "bone density scan in 1 year...July 2019" | A/P #5 ✓ | ✅ |
| "follow up in 6 months or sooner" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（"Prolia 6 months" 漏 "every"）
- Letter 正确地**没有**提到 Brain MRI（因为它是 conditional，extraction 错误地列为 planned）

### ROW 4 总评
- **Extraction**: P0:0 P1:0 P2:1
- **Letter**: P0:0 P1:0 P2:1
- 非常全面的 letter：涵盖 recurrence status、bone density、all 7 A/P items 的 medication/lifestyle recommendations

## ROW 5 (coral_idx 144)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | on leuprolide/anastrozole/palbociclib ✓ | ✅ |
| in-person | Televisit | "video encounter" ✓ | ✅ |
| summary | Recurrent breast cancer...follow-up | ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC (micropapillary) with metastatic recurrence | pathology + cervical LN FNA ✓ | ✅ |
| Stage_of_Cancer | Originally Stage III, now metastatic (Stage IV) | A/P "Stage III" + metastatic ✓ | ✅ |
| Distant Metastasis | Yes, to left IM LN and sternum | PET/CT ✓ | ✅ |
| Metastasis | Yes, to cervical LN, IM LN, sternum | ✓ | ✅ |
| lab_summary | No labs in note | only old Cr from 08/23/19 ✓ | ✅ |
| findings | cervical LAD + brachial plexus + MRI brain normal + CT mixed response + sternal lesion | 原文 imaging ✓ | ✅ |
| current_meds | anastrozole, palbociclib, goserelin | cancer treatment meds ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | Mixed: cervical LN decreased, breast nodule decreased, axillary LN increased, new sternal lesion | CT 08/23/19 ✓ 完整 | ✅ |
| medication_plan | Continue leuprolide, anastrozole, palbociclib | A/P #1-2 ✓ | ✅ |
| radiotherapy_plan | Radiation referral for symptomatic neck/brachial plexus | A/P #3, #6 ✓ | ✅ |
| imaging_plan | CT and bone scan ordered | A/P #5 ✓ | ✅ |
| lab_plan | Labs monthly on lupron injection day | A/P #4 ✓ | ✅ |
| Advance care | full code | ✓ | ✅ |
| Referral: Specialty | Radiation oncology consult | A/P #3, #6 ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——全部准确

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "recurrent breast cancer...spreading to your left neck and affecting the nerves in your left arm" | cervical LN + brachial plexus involvement ✓ | ✅ |
| "MRI...lymph node in your neck has gotten bigger, which might be causing your arm symptoms" | MRI "interval enlargement of left level 5B...brachial plexus involvement...may account for...left arm symptoms" ✓ | ✅ |
| "MRI of your brain did not show any cancer" | "Normal MRI of the brain...no evidence for intracranial metastatic disease" ✓ | ✅ |
| "some lymph nodes in your neck have gotten smaller" | CT "decreased size of left cervical chain LNs" ✓ | ✅ |
| "lymph node in your chest has stayed about the same size" | "stable borderline enlargement of mediastinal LN" ✓ | ✅ |
| "new spot in your sternum that looks like it might be cancer" | "new focal uptake within sternum, suspicious for osseous metastases" ✓ | ✅ |
| "continue taking leuprolide, anastrozole, and palbociclib" | A/P ✓ | ✅ |
| "ondansetron to help with nausea" | supportive med ✓ | ✅ |
| "referred to radiation oncology for...left neck and arm" | A/P #3, #6 ✓ | ✅ |
| "CT scan and a bone scan done before your next visit" | A/P #5 ✓ | ✅ |
| "blood tests every month...on the day of your lupron injection" | A/P #4 ✓ 保留具体细节 | ✅ |
| "full code" | ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——优秀的 letter。Mixed response 正确描述，brachial plexus 通俗化优秀

### ROW 5 总评
- **Extraction**: P0:0 P1:0 P2:0
- **Letter**: P0:0 P1:0 P2:0
- 无任何问题。特别好的通俗化："affecting the nerves in your left arm"

## ROW 6 (coral_idx 145)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| **Patient type** | **new patient** | Chief Complaint 明确写 "Follow-up"。**应为 follow up** | P2 |
| summary | 34-year-old...post-bilateral mastectomy...adjuvant systemic therapy | ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 1 IDC with extensive DCIS | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage IA (pT1 N0) | 1.5cm, 0/1 LN ✓ | ✅ |
| Distant Metastasis | No | ✓ | ✅ |
| lab_summary | comprehensive (all labs from 06/08/19) | ✓ | ✅ |
| findings | surgery results + PE (nerve irritation, edema) | ✓ | ✅ |
| current_meds | zoladex, letrozole | ✓ | ✅ |
| goals_of_treatment | curative | ✓ | ✅ |
| medication_plan | letrozole + zoladex 3yr + tamoxifen sequence + estradiol + gabapentin | A/P ✓ | ✅ |
| lab_plan | Estradiol monthly | ✓ | ✅ |
| follow_up | 3 months or sooner | ✓ | ✅ |
| **Referral: Genetics** | **genetics referral** | Myriad 基因检测已于 04/25/2019 完成且 **Negative**。这是**历史记录**，不是新 plan | P2 |

**Extraction 小结**: P0:0 P1:0 P2:2（Patient type 错 + genetics referral 是历史记录）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓（正确，虽然 extraction 写了 "new patient"） | ✅ |
| "surgery to remove both breasts (bilateral mastectomy) with expanders on 06/21/2019" | ✓ | ✅ |
| "right breast showed a small cancer (1.5 cm)...removed completely...no cancer in lymph nodes" | pathology: 1.5cm, 0/1 LN, margins clear ✓ | ✅ |
| "left breast was healthy" | "Left breast: benign" ✓ | ✅ |
| "recovering well...nerve irritation and mild swelling" | A/P + PE ✓ | ✅ |
| "start taking letrozole today" | A/P ✓ | ✅ |
| "continue another medication for at least 3 years...switch to tamoxifen later" | zoladex + tamoxifen sequence ✓ | ✅ |
| "monthly estradiol tests" | A/P ✓ | ✅ |
| "Gabapentin is prescribed as needed" | A/P ✓ | ✅ |
| "already on zoladex" | ✓ | ✅ |
| "**referred...for further genetic testing**" | **基因检测已完成且 Negative**——跟了 extraction 的错误 genetics referral | P2 |
| "next visit is in 3 months or sooner if needed" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（genetic testing referral 已完成，不应推荐）

### ROW 6 总评
- **Extraction**: P0:0 P1:0 P2:2
- **Letter**: P0:0 P1:0 P2:1
- 这是 extraction→letter 错误传播的典型案例：extraction 错误捕获历史 genetics referral → letter 不知情地推荐已完成的检测

## ROW 7 (coral_idx 146)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "2nd opinion" ✓ | ✅ |
| second opinion | yes | "CC: 2nd opinion" ✓ | ✅ |
| Type_of_Cancer | ER-/PR- IDC, HER2+ | "ER/PR neg. ***** by IHC" ✓ | ✅ |
| Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | Stage II 1998, metastatic 2008 ✓ | ✅ |
| Distant Metastasis | Yes, to left supraclavicular node and mediastinum | ✓ | ✅ |
| findings | probable mild progression + stable mediastinal + brain MRI normal + LVEF 52% | A/P + imaging ✓ | ✅ |
| current_meds | "" | "Has been off of rx since last wk" ✓ | ✅ |
| recent_changes | d/c current rx (Herceptin/Taxotere) | A/P ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | probable mild progression, equivocal | A/P ✓ | ✅ |
| medication_plan | Recommend new agent + recheck test + no hormonal therapy | A/P ✓ | ✅ |
| **therapy_plan** | "...physical therapy" | **原文 A/P 没有提到 physical therapy——这是 extraction 幻觉** | P2 |

**Extraction 小结**: P0:0 P1:0 P2:1（therapy_plan 幻觉了 physical therapy）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "second opinion on how to manage your metastatic breast cancer" | ✓ | ✅ |
| "cancer seems to have slightly grown in the left breast and possibly in the left chest area" | A/P "probable mild progression" ✓ | ✅ |
| "cancer in your chest area has stayed the same" | "Stable mediastinal adenopathy" ✓ | ✅ |
| "no sign of cancer in your brain" | Brain MRI normal ✓ | ✅ |
| "heart function has slightly decreased" | "LVEF somewhat decreased, 52%" ✓ | ✅ |
| "CT scan shows that a small spot in your left breast has gotten bigger" | CT: SUV 2.1 (was 1.8) ✓ | ✅ |
| "**Another test called a medication is still high**" | [REDACTED] tumor marker persistently elevated——**"a medication"是 garbled** | P2 |
| "current treatment with a medication, Herceptin, and Taxotere has been stopped" | A/P "d/c current rx" ✓ | ✅ |
| "start a new medication as the next step" | A/P "Rec ***** as next line" ✓ | ✅ |
| "**you will need to have a medication test done again**" | A/P "recheck [REDACTED]"——**"medication test"是 garbled** | P2 |
| "not be given any hormones" | A/P "Would not consider hormonal therapy" ✓ | ✅ |
| "clinical trials that might be available" | A/P ✓ | ✅ |
| "**You will have a physical therapy session**" | **原文没有 PT——跟了 extraction 的幻觉** | P2 |

**Letter 小结**: P0:0 P1:0 P2:3（garbled x2 + PT 幻觉传播）

### ROW 7 总评
- **Extraction**: P0:0 P1:0 P2:1
- **Letter**: P0:0 P1:0 P2:3
- extraction 幻觉了 "physical therapy"→ letter 传播了这个错误
- 大量 redaction 导致 letter 多处 garbled

## ROW 8 (coral_idx 147)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | new consult to establish care ✓ | ✅ |
| in-person | Televisit | ZOOM ✓ | ✅ |
| summary | ...incomplete course of neoadjuvant TCHP...discuss treatment options | HPI ✓ | ✅ |
| Type_of_Cancer | ER-/PR-/HER2+ (IHC 3+, FISH 5.7) grade 3 IDC | LN pathology: ER-, HER2 IHC 3, FISH 5.7, Ki-67 75% ✓ | ✅ |
| Stage_of_Cancer | Originally Stage III, now post-neoadjuvant with 3/28 LN+ | A/P ✓ | ✅ |
| Distant Metastasis | No | PET/CT 02 and 06/2019 no mets ✓ | ✅ |
| findings | breast pCR + 3/28 LN positive + extranodal extension + necrotizing lymphadenitis | pathology ✓ 详细 | ✅ |
| current_meds | "" | not on cancer treatment ✓ | ✅ |
| goals_of_treatment | curative | adjuvant intent ✓ | ✅ |
| response_assessment | breast pCR but 3/28 LN+ with HER2+ | pathology + PET ✓ | ✅ |
| medication_plan | adjuvant AC x4 → T-DM1 | A/P ✓ | ✅ |
| radiotherapy_plan | radiation | A/P "importance of radiation after completing AC" ✓ | ✅ |
| imaging_plan | echocardiogram prior to AC | A/P mentions echo ✓ | ✅ |
| **procedure_plan** | **No procedures planned** | A/P: "port placement and echocardiogram"——**漏了 port placement** | P2 |
| follow_up | this coming week | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（漏 port placement）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "consultation to discuss treatment options and to establish care after...surgery and...incomplete course of treatment" | HPI ✓ | ✅ |
| "invasive ductal carcinoma...started in the milk ducts" | IDC ✓ | ✅ |
| "originally at a more advanced stage, but after treatment...less advanced stage...has not spread" | Stage III → post-NAC, no distant mets ✓ | ✅ |
| "**no remaining cancer in the breast tissue, which is a good sign. However, cancer was still found in some of the lymph nodes**" | pathology: breast pCR, 3/28 LN+。**完美使用了 pCR prompt 规则！** ✓ | ✅ |
| "Recent imaging tests show...cancer has not spread" | PET/CT no mets ✓ | ✅ |
| "currently taking oxyCODONE for pain relief" | medication list ✓ | ✅ |
| "AC for four cycles, followed by...T-DM1" | A/P ✓ | ✅ |
| "radiation treatment" | A/P ✓ | ✅ |
| "echocardiogram...heart ultrasound...heart is healthy enough" | A/P ✓ | ✅ |
| "follow-up visit in person this coming week" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0
- **亮点**：pCR 描述完美——"no remaining cancer in the breast tissue...However, cancer was still found in some of the lymph nodes"——这正是 iter13 prompt 修复要实现的效果

### ROW 8 总评
- **Extraction**: P0:0 P1:0 P2:1（漏 port placement）
- **Letter**: P0:0 P1:0 P2:0
- pCR + LN positive 的情况处理得很好

## ROW 9 (coral_idx 148)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | post-surgery follow-up ✓ | ✅ |
| Type_of_Cancer | ER+/PR-/HER2- grade 2 IDC | pathology: ER+ 85%, PR- <1%, HER2- IHC 0, Ki-67 1-2% ✓ | ✅ |
| Stage_of_Cancer | Stage II (pT3 N1) | A/P "Stage II" ✓ | ✅ |
| Distant Metastasis | No | ✓ | ✅ |
| findings | 3.84cm IDC, ~5% cellularity, margins neg, 1 macro + 1 micro + 1 ITC in 4 LN | pathology 详细准确 ✓ | ✅ |
| current_meds | "" | post-surgery, not on cancer treatment ✓ | ✅ |
| goals_of_treatment | curative | adjuvant ✓ | ✅ |
| response_assessment | 3.84cm residual with 5% cellularity, LN status | pathology ✓ | ✅ |
| medication_plan | letrozole after radiation + Fosamax | A/P #5-7 ✓ | ✅ |
| radiotherapy_plan | Radiation referral | A/P #6 ✓ | ✅ |
| procedure_plan | Drains removed Thursday | A/P ✓ | ✅ |
| Advance care | full code | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——全部准确

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "surgery to remove both breasts" | bilateral mastectomies ✓ | ✅ |
| "**edges of the removed tissue are clean, which is a good sign**" | margins negative ✓——**正确使用了 iter13 新 prompt 规则！** | ✅ |
| "**However, cancer was still found in some of the lymph nodes**" | 3 LN involved ✓ | ✅ |
| "invasive ductal carcinoma...milk ducts...estrogen receptor positive" | ✓ | ✅ |
| "No imaging tests were done during this visit" | ✓ | ✅ |
| "recovering well...drains still in place" | HPI ✓ | ✅ |
| "ondansetron and prochlorperazine" for nausea | supportive meds ✓ | ✅ |
| "referred to radiation therapy" | A/P #6 ✓ | ✅ |
| "letrozole" after radiation | A/P #7 ✓ | ✅ |
| "fosamax to protect your bones" | A/P #5 osteopenia ✓ | ✅ |
| "drains will be removed on Thursday" | A/P ✓ | ✅ |
| "full code" | ✓ | ✅ |
| Emotional support sentence | "She is tearful today" ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0
- **亮点**：正确说 "edges are clean...However, cancer was still found in some lymph nodes"——完美处理了 residual cancer + negative margins + LN positive 的情况

### ROW 9 总评
- **Extraction**: P0:0 P1:0 P2:0
- **Letter**: P0:0 P1:0 P2:0
- kidney transplant 患者的复杂背景没有影响提取/letter 质量

## ROW 10 (coral_idx 149)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | post-surgery follow-up ✓ | ✅ |
| Type_of_Cancer | HR+ HER2- grade 2 IDC | A/P "HR+ and her 2 negative" ✓ | ✅ |
| **Stage_of_Cancer** | **Stage II (8.8 cm tumor with 20 lymph nodes involved)** | A/P "with **July 20** lymph nodes involved"——"July 20"是 redacted 的数字（可能是 X/20），**不是 20 个全部受累** | P2 |
| Distant Metastasis | No | ✓ | ✅ |
| **findings** | **8.8 cm tumor and 20 lymph nodes involved** | 同上：**"20 involved" 是 redacted 数字误读** | P2 |
| current_meds | letrozole | ✓ | ✅ |
| goals_of_treatment | curative | adjuvant ✓ | ✅ |
| medication_plan | continue letrozole | A/P #4 ✓ | ✅ |
| radiotherapy_plan | radiation to left chest wall and surrounding lymph nodes | A/P #5 ✓ | ✅ |
| imaging_plan | DEXA | A/P #6 ✓ | ✅ |
| Advance care | full code | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:2（LN count "20 involved" 是 redacted 数字误读，同时出现在 Stage 和 findings）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "surgery to remove a large tumor from your left breast, along with **some** lymph nodes" | A/P ✓——letter 聪明地用了 "some" 而非 extraction 的 "20"，更准确 | ✅ |
| "cancer...starts in the milk ducts...does not have...HER2" | ✓ | ✅ |
| "recovered well...do not need chemotherapy because the risk is low" | "Low risk *****" + HPI "does not need chemotherapy" ✓ | ✅ |
| "continue taking letrozole...April 2021" | A/P #4 ✓ | ✅ |
| "radiation treatment to the left side of your chest and the nearby lymph nodes" | A/P #5 ✓ | ✅ |
| "DEXA scan, which checks the strength of your bones" | A/P #6 ✓ 通俗解释 ✓ | ✅ |
| "full code" | ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——letter 比 extraction 更准确（用 "some lymph nodes" 而非 "20"）

### ROW 10 总评
- **Extraction**: P0:0 P1:0 P2:2
- **Letter**: P0:0 P1:0 P2:0
- letter 自动纠正了 extraction 的 LN count 误读——用 "some" 替代了不确定的 "20"

## ROW 11 (coral_idx 150)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | on Faslodex/Denosumab ✓ | ✅ |
| Type_of_Cancer | IDC, ER+ (inferred from letrozole), HER2: not tested | ✓ | ✅ |
| Stage_of_Cancer | Originally Stage IIIC, now metastatic (Stage IV) | ✓ | ✅ |
| Distant Metastasis | Yes, to bone | bone mets + mandibular mass ✓ | ✅ |
| findings | bone mets, PET increased activity, mandibular mass, worsening RLE | comprehensive ✓ | ✅ |
| current_meds | Fulvestrant, Denosumab | ✓ | ✅ |
| recent_changes | Mycelex for thrush | A/P ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | PET increased activity, worsening RLE numbness | ✓ | ✅ |
| medication_plan | Continue Faslodex + Denosumab + Mycelex + salt/soda | A/P ✓ | ✅ |
| **imaging_plan** | **PETCT to femur/toes** | A/P: PETCT **+ MRI of lumbar, pelvis and right femur**——**漏了 MRI** | P2 |

**Extraction 小结**: P0:0 P1:0 P2:1（imaging_plan 漏 MRI lumbar/pelvis/femur）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "breast cancer has spread to your bones" | bone mets ✓ | ✅ |
| "**cancer has grown in some areas, especially in your jaw and right leg**" | Jaw: A/P "s/p xrt to jaw **improved** pain"——jaw 是改善中的，不是在恶化。Right leg: 确实在恶化。**Letter 把改善中的 jaw 和恶化中的 leg 混在一起说"grown"** | P2 |
| "more numbness in your jaw and right leg" | jaw numbness 是 XRT 后残余（stable），right leg numbness 在恶化——混淆了两者的趋势 | ✅ |
| "Mycelex five times a day for 14 days for thrush" | A/P ✓ | ✅ |
| "salt and soda rinses" | A/P ✓ | ✅ |
| "continue taking Faslodex and Denosumab" | A/P ✓ | ✅ |
| "PETCT scan to check your femur and toes" | A/P ✓（但漏了 MRI order） | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（jaw 的 "cancer has grown" 误导——jaw 已做 XRT 且在改善）

### ROW 11 总评
- **Extraction**: P0:0 P1:0 P2:1
- **Letter**: P0:0 P1:0 P2:1

## ROW 12 (coral_idx 151)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2+ (IHC 3+/FISH 5.4) IDC | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage IV | de novo metastatic ✓ | ✅ |
| Distant Metastasis | Yes, to brain, lung, bone | ✓ | ✅ |
| current_meds | herceptin, letrozole | ✓ (pertuzumab redacted) | ✅ |
| response_assessment | brain new lesions + body stable + celiac node decreased | detailed imaging ✓ | ✅ |
| medication_plan | continue herceptin+[REDACTED], letrozole, off chemo | A/P ✓ | ✅ |
| **imaging_plan** | **CT CAP q4mo + bone scan + MRI brain q4mo** | A/P also says "**Echo q6 months**, repeat April 2019"——**漏了 echo** | P2 |
| radiotherapy_plan | await GK / Rad Onc input | A/P ✓ | ✅ |
| Advance care | POLST on file, against life support | "DNR/DNI" ✓ | ✅ |
| follow_up | 6 weeks | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（漏 echo q6 months）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "MRI brain...new brain lesions" | MRI 08/15/18 ✓ | ✅ |
| "CT scans...cancer in your bones is stable and has not grown" | CT 09/05/18 ✓ | ✅ |
| "slight decrease in the size of a lymph node in your abdomen" | celiac axis node 9→7mm ✓ | ✅ |
| "continue taking herceptin and another medication" | herceptin + pertuzumab ✓ | ✅ |
| "continue taking letrozole every day" | ✓ | ✅ |
| "stay off chemotherapy for now...trouble with side effects" | A/P ✓ | ✅ |
| "**continue taking morphine and oxycodone for pain**" | A/P: neuropathy "improved...able to tolerate **without analgesics**"——**患者已停止用止痛药** | P2 |
| "CT scan every 4 months + MRI brain every 4 months + bone scan" | A/P ✓ | ✅ |
| "6 weeks for follow-up" | ✓ | ✅ |
| "radiation oncology for further consultation" | GK/Rad Onc ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（说 continue morphine/oxycodone 但患者已不用）

### ROW 12 总评: Ext P2:1, Letter P2:1

---

## ROW 14 (coral_idx 153)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | ✓ | ✅ |
| Type_of_Cancer | ER+ metastatic breast cancer, HER2- | FNA: ER 99%, PR 25%, HER2 1+ FISH neg ✓ | ✅ |
| Stage_of_Cancer | Metastatic (Stage IV) | de novo metastatic ✓ | ✅ |
| Distant Metastasis | Yes, to bone, liver, and nodes | ✓ | ✅ |
| current_meds | "" | 停了 palbociclib/fulvestrant，现在做 Mexico alternative therapy ✓ | ✅ |
| recent_changes | stopped palbociclib/fulvestrant, started low-dose chemo in Mexico | HPI ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | breast nodules increased, liver lesion increased, new sclerotic foci | CT 11/30/18 ✓ | ✅ |
| medication_plan | topical cannabis/sulfur, Cymbalta Rx given | A/P #2 ✓ | ✅ |
| imaging_plan | CT CAP + Spine MRI May, Spine MRI in 6 weeks | A/P ✓ | ✅ |
| lab_plan | labs drawn every two weeks | doctor's note ✓ | ✅ |
| follow_up | 2 months (from Mexico return) or 3 months | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——全部准确

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer in your right breast has slightly grown in size" | CT 11/30/18: nodule 10→14mm ✓ | ✅ |
| "no new signs of cancer spreading" | MRI spine: no new disease ✓ | ✅ |
| "stopped palbociclib and fulvestrant" | HPI ✓ | ✅ |
| "started low-dose chemotherapy at home" | Mexico protocol ✓ | ✅ |
| "pamidronate once a week" | ✓ | ✅ |
| "topical cannabis and sulfur" | A/P #2 ✓ | ✅ |
| "Cymbalta, but you haven't tried it yet" | A/P #2 ✓ | ✅ |
| "referred to physical therapy" | PT 03/12/19 ✓ | ✅ |
| "CT scan and total spine MRI in May" | A/P ✓ | ✅ |
| "MRI of your spine in 6 weeks" | A/P #3 ✓ | ✅ |
| "labs checked every two weeks" | doctor's note ✓ | ✅ |
| "in-person visit in 2 months" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——准确覆盖了所有 A/P items

### ROW 14 总评: Ext P2:0, Letter P2:0

## ROW 19 (coral_idx 158)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | new consult ✓ | ✅ |
| summary | newly diagnosed...palpable mass + bloody nipple discharge...neoadjuvant therapy | HPI ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2+ (FISH+) grade 3 IDC with Ki-67 20-90% | pathology: ER 90%, PR 90% (variable), HER2 3+ FISH 9.5 (heterogeneous) ✓ | ✅ |
| Stage_of_Cancer | Stage IIIA (pT2 N2a) | locally advanced by imaging ✓（staging inference 略不精确但合理） | ✅ |
| Distant Metastasis | No | PET/CT: no distant mets ✓ | ✅ |
| findings | core biopsy + PET/CT + PE (4cm mass + nipple inversion) + pulmonary nodules + sinus | 全面准确 ✓ | ✅ |
| goals_of_treatment | curative | neoadjuvant intent ✓ | ✅ |
| medication_plan | TCHP + GCSF | A/P ✓ | ✅ |
| therapy_plan | TCHP, avoid anthracycline due to CAD, port, echo, chemo teaching, clinical trial | A/P 完整捕获 ✓ | ✅ |
| procedure_plan | Port Placement | A/P ✓ | ✅ |
| imaging_plan | Echocardiogram | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——全部准确，"avoid anthracycline due to CAD" 正确捕获

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "first visit...lump in left breast and bloody discharge from nipple" | HPI ✓ | ✅ |
| "invasive ductal carcinoma...grade 3...grows quickly" | pathology ✓ | ✅ |
| "positive for ER, PR, and HER2" | ✓ | ✅ |
| "Ki-67 score of 20-90%" | ✓ | ✅ |
| "stage III...spread to nearby lymph nodes but not to other parts" | ✓ | ✅ |
| "biopsy confirmed...imaging showed mass...enlarged lymph nodes" | ✓ | ✅ |
| "No cancer found in other parts" | PET/CT ✓ | ✅ |
| "**a treatment called a combination of chemotherapy and targeted therapy drugs, which is a combination of chemotherapy drugs**" | A/P "TCHP"——**TCHP 名称被替换后 garbled/redundant** | P2 |
| "port placed...device that helps deliver chemotherapy" | ✓ | ✅ |
| "echocardiogram...check how well your heart is working" | ✓ | ✅ |
| "chemo teaching and authorization for...treatment with GCSF, which helps prevent infections" | ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（TCHP description garbled/redundant from name replacement）

### ROW 19 总评
- **Extraction**: P0:0 P1:0 P2:0
- **Letter**: P0:0 P1:0 P2:1
- 复杂的 HER2 heterogeneous case 处理得很好。extraction 正确捕获了 "avoid anthracycline due to CAD"

## ROW 20 (coral_idx 159)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | consultation for metastatic recurrence ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade II IDC with 1.8cm DCIS | pathology ✓ | ✅ |
| **Stage_of_Cancer** | **Stage IIA** | tumor 0.9cm + 0/2 SLN = **Stage IA**，not IIA | P2 |
| Distant Metastasis | Yes, to bone and lymph nodes | PET/CT ✓ | ✅ |
| **Metastasis** | **Yes, to bone** | 应为 "bone and lymph nodes"——与 Distant Metastasis 不一致 | P2 |
| **lab_summary** | **POCT glucose 104** | 这是 **03/01/2013** 的数据（8 年前），无临床意义 | P2 |
| findings | pathology + PET/CT + iliac biopsy + PE | 全面 ✓ | ✅ |
| current_meds | letrozole, palbociclib | just started ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| medication_plan | letrozole + palbociclib + denosumab pending dental | A/P ✓ | ✅ |
| radiotherapy_plan | consultation, migratory pain, unclear focal site | A/P nuance captured ✓ | ✅ |
| **procedure_plan** | **"Abdomen, Pelvis, if not possible, would send for [REDACTED] 360"** | **garbled**——混淆了 CT CAP 和 Foundation One testing | P2 |
| imaging_plan | MRI Total Spine + CT CAP + Repeat 3 months | A/P ✓ | ✅ |
| lab_plan | tumor markers + monthly palbociclib labs | A/P ✓ | ✅ |
| genetic_testing_plan | Foundation One, if not possible [REDACTED] 360 | A/P ✓ | ✅ |
| follow_up | ~1 month | ✓ | ✅ |
| Referral: Specialty | Rad Onc referral | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:4（Stage IA/IIA + Metastasis 不一致 + old lab + procedure garbled）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "consultation regarding a metastatic recurrence" | ✓ | ✅ |
| "0.9cm grade II IDC...1.8cm DCIS (ductal carcinoma in situ)" | pathology ✓ | ✅ |
| "cancer has now spread to your bones and lymph nodes" | PET/CT + biopsy ✓ | ✅ |
| "Stage IV...spread to other parts of your body" | ✓ | ✅ |
| "left-sided pain in the rib and mid-axillary region" | HPI ✓ | ✅ |
| "start taking Letrozole and Palbociclib" | A/P ✓ | ✅ |
| "monthly blood work to monitor...Palbociclib" | ✓ | ✅ |
| "referred to radiation oncology...rib and back pain" | A/P ✓ | ✅ |
| "MRI of your total spine and a CT scan of your chest, abdomen, and pelvis" | A/P ✓ | ✅ |
| "labs including tumor markers" | ✓ | ✅ |
| "follow-up visit in about 1 month" | ✓ | ✅ |
| "genetic test to learn more about your cancer" | Foundation One ✓ | ✅ |
| **（漏）** | A/P: "initiation of denosumab...dental clearance prior"——**letter 未提及 denosumab 和 dental clearance** | P2 |

**Letter 小结**: P0:0 P1:0 P2:1（漏 denosumab/dental clearance）

### ROW 20 总评
- **Extraction**: P0:0 P1:0 P2:4
- **Letter**: P0:0 P1:0 P2:1
- Extraction 有多个小问题但无 P1。Letter 涵盖了大部分 plan items 但漏了骨保护药

## ROW 22 (coral_idx 161)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | ✓ | ✅ |
| second opinion | yes | "here for a second opinion" ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- IDC with metastatic recurrence | ✓ | ✅ |
| Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | ✓ | ✅ |
| Distant Metastasis | Yes, to bone, chest wall, right infraclavicular and right IM nodes | ✓ | ✅ |
| lab_summary | No labs in note | labs are 8 months old (01/29/2021 vs 09/30/2021 visit), reasonable ✓ | ✅ |
| findings | history + PET good response + pneumonitis + PE normal | ✓ | ✅ |
| current_meds | anastrozole, denosumab | 正确——letrozole 已换，abemaciclib 已停 ✓ | ✅ |
| recent_changes | abemaciclib held + letrozole→anastrozole + dose reduced | comprehensive ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | PET good response + pneumonitis + PE normal | ✓ | ✅ |
| **medication_plan** | "Continue arimidex alone...;**also: letrozole, abemaciclib**" | **letrozole 已换成 anastrozole, abemaciclib 已停——POST hook 错误添加了已停用药物** | P2 |
| **therapy_plan** | "**Continue letrozole, abemaciclib**. Continue arimidex alone." | **错误！letrozole 早已换成 anastrozole, abemaciclib 已因 pneumonitis 停用** | P2 |
| imaging_plan | Pet ct now | A/P #5 ✓ | ✅ |
| Advance care | Full code | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:2（medication_plan + therapy_plan 包含已停用药物——POST hook 错误）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "new patient evaluation and second opinion" | ✓ | ✅ |
| "left breast cancer in 1994 and right breast cancer in 2000" | ✓ | ✅ |
| "May 2020, your cancer returned and spread to your chest wall, bones, and lymph nodes" | ✓ | ✅ |
| "Recent scans show that your treatment has been working well" | PET good response ✓ | ✅ |
| "lung irritation, possibly due to...abemaciclib, which was stopped in August 2021" | pneumonitis ✓ 通俗化 ✓ | ✅ |
| "no new problems were found" on PE | ✓ | ✅ |
| "abemaciclib was stopped because it caused lung irritation" | ✓ | ✅ |
| "letrozole was changed to anastrozole in July 2020 due to a skin rash" | ✓ | ✅ |
| "PET scan now to see how your cancer is doing" | A/P #5 ✓ | ✅ |
| "if stable, continue taking anastrozole" | A/P #5 ✓——**正确说了 anastrozole 而非 extraction 的 letrozole** | ✅ |
| "if cancer has grown...Faslodex...if you have a certain mutation" | A/P #6 ✓ | ✅ |
| "Afinitor or Xeloda or joining a clinical trial" | A/P #7 ✓ | ✅ |
| "full code" | ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——letter 正确忽略了 extraction 的 "Continue letrozole, abemaciclib" 错误

### ROW 22 总评
- **Extraction**: P0:0 P1:0 P2:2（POST hook 错误添加已停用药物）
- **Letter**: P0:0 P1:0 P2:0——**letter 比 extraction 更准确**
- iter12e 的 P2（语法不完整"you have been on since October 2020"）在 iter13 中已消失

## ROW 24 (coral_idx 163)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | discuss systemic therapy ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade II micropapillary mucinous carcinoma | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage III | ~5cm + N1mi, plausible ✓ | ✅ |
| Distant Metastasis | No | PET: no metastatic disease ✓ | ✅ |
| findings | pathology + imaging + PE, 全面 | ✓ | ✅ |
| supportive_meds | TYLENOL #4, oxyCODONE | ✓ | ✅ |
| goals_of_treatment | adjuvant | ✓ | ✅ |
| medication_plan | adjuvant hormone therapy if low risk | A/P ✓ | ✅ |
| therapy_plan | radiation + hormone therapy + PT | A/P ✓ | ✅ |
| radiotherapy_plan | radiation if low risk, appt 12/07/18 | A/P ✓ | ✅ |
| lab_plan | [REDACTED] test to evaluate chemo benefit | MammaPrint ✓ | ✅ |
| genetic_testing_plan | send surgical specimen for MP | A/P ✓ | ✅ |
| Referral: Others | Physical therapy referral | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——全部准确

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "ER+/PR+/HER2- grade II micropapillary mucinous carcinoma (a type of cancer that makes mucus)" | pathology ✓ 通俗解释 ✓ | ✅ |
| "stage III and has not spread to other parts of your body" | ✓ | ✅ |
| "Imaging shows a large mass...some changes in your lymph nodes, but these are considered benign" | FNA axillary LN was negative ✓。**但手术 SLN 有 micrometastasis (2/4, 0.4mm) 未提及** | P2 |
| "goal...prevent the cancer from coming back after surgery" | adjuvant ✓ | ✅ |
| "acetaminophen-codeine (TYLENOL #4) and oxyCODONE (ROXICODONE)" | 保留药名 ✓ | ✅ |
| "hormone therapy if a test shows that you are at low risk" | A/P ✓ | ✅ |
| "**a test to see if you need chemotherapy**" | A/P MammaPrint ✓——**iter12e 的 "medication test" P2 已修复！** | ✅ |
| "radiation oncology...December 7, 2018" | A/P ✓ | ✅ |
| "referred to physical therapy" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（SLN micrometastasis 未提及）
- **改进**: iter12e 的 "medication test" P2 和 "MP" 未解释 P2 都已修复

### ROW 24 总评
- **Extraction**: P0:0 P1:0 P2:0
- **Letter**: P0:0 P1:0 P2:1

## ROW 29 (coral_idx 168)

### Extraction: P0:0 P1:0 P2:0 — 全部准确
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "invasive ductal carcinoma...grade 2...ER+/PR+...no HER2" | pathology ✓ | ✅ |
| "cancer has spread to a nearby lymph node, but not to other parts" | SLN micrometastasis ✓ + no distant mets ✓ | ✅ |
| "start taking letrozole" | A/P ✓ | ✅ |
| "bone density scan when you return from your travels" | A/P ✓ | ✅ |
| "surgery in September 2019" | A/P ✓ | ✅ |
| "long-term oncology follow-up closer to your home" | A/P ✓ | ✅ |
| （漏）MammaPrint Low Risk + no-chemo rationale | A/P 详细讨论了 MINDACT trial + MammaPrint Low Risk → no chemo。**Letter 没提** | P2 |

**Letter 小结**: P0:0 P1:0 P2:1（MammaPrint/no-chemo rationale 未提及）
### ROW 29 总评: Ext P2:0, Letter P2:1

---

## ROW 30 (coral_idx 169) — **iter12e P1 修复验证**

### Extraction: P0:0 P1:0 P2:0 — 全部准确
- Type_of_Cancer: ER-/PR-/HER2+ grade 2 IDC ✓
- Stage: Clinical stage II-III ✓
- Distant Metastasis: No ✓, Metastasis: No ✓

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "invasive ductal carcinoma...HER2 positive...protein called HER2" | ✓ | ✅ |
| "**cancer is in the right breast and has not spread to other parts of your body**" | A/P "PET/CT no evidence of metastases" + "**high-risk node-negative**" ✓ | ✅ |
| "cure the cancer" | curative intent ✓ | ✅ |
| "neoadjuvant chemotherapy...before surgery to try to shrink the cancer" | ✓ | ✅ |
| "12 cycles of weekly paclitaxel plus trastuzumab/pertuzumab...or 6 cycles of TCHP" | A/P: two regimen options ✓ 保留药名 ✓ | ✅ |
| "continue with trastuzumab for a total of one year" | ✓ | ✅ |
| "TTE (echocardiogram)...check how well your heart is working" | ✓ | ✅ |
| "Mediport placed" | ✓ | ✅ |
| "decide after the weekend whether you want to start treatment at your clinic or elsewhere" | ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0
- **✅ iter12e P1 CONFIRMED FIXED** — 不再说 "spread to some nearby lymph nodes"

### ROW 30 总评: Ext P2:0, Letter P2:0 — **P1→P0 修复成功**

---

## ROW 33 (coral_idx 172)

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | on letrozole since 2011 ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 ILC | note says ILC ✓ | ✅ |
| **Stage_of_Cancer** | **Originally Stage IIB, now Stage III** | 患者**无复发**，"now Stage III"对无复发患者毫无意义 | P2 |
| findings | no evidence of recurrence + PE normal | A/P ✓ | ✅ |
| medication_plan | letrozole + calcium/VitD + NSAIDs | A/P #1-2 ✓ | ✅ |
| imaging_plan | Consider MRI brain if [REDACTED] continues | A/P #4 ✓ | ✅ |
| follow_up | 6 months | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（Stage "now Stage III" for no-recurrence patient）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "no signs of the cancer coming back" | A/P "no evidence of recurrence" ✓ | ✅ |
| "small, soft lymph node in your left neck, but it is not concerning" | PE ✓ | ✅ |
| "continue taking letrozole daily" | A/P ✓ | ✅ |
| "calcium and vitamin D daily" | A/P ✓ | ✅ |
| "NSAIDs (pain relievers) with food" | A/P ✓ | ✅ |
| "continue exercising and doing weight-bearing activities" | A/P ✓ | ✅ |
| "**If a certain medication continues to cause issues, an MRI of the brain might be considered**" | A/P 说的是**头痛**持续就考虑 MRI brain。**[REDACTED] 是头痛症状，不是药物** | P2 |
| "return to the clinic in 6 months" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（MRI brain 原因误解——头痛被当成药物问题）

### ROW 33 总评: Ext P2:1, Letter P2:1

## ROW 34 (coral_idx 173)
### Extraction: P0:0 P1:0 P2:0 — 准确（second local relapse, tamoxifen, chest wall RT referral, labs）
### Letter: P0:0 P1:0 P2:0 — "cancer has returned...start taking tamoxifen...referred to radiation oncology...basic blood tests...6 months" 每句有依据
### ROW 34 总评: Ext P2:0, Letter P2:0

---

## ROW 36 (coral_idx 175)
### Extraction: P0:0 P1:0 P2:0 — 准确（Abraxane cycle 8, arm swelling/DVT r/o, radiation referral, RTC 2 weeks）
### Letter: P0:0 P1:0 P2:0 — "swelling...doppler test...no new cancer growth...switched to Abraxane...continue weekly...radiation oncology...2 weeks" 每句有依据
### ROW 36 总评: Ext P2:0, Letter P2:0

---

## ROW 40 (coral_idx 179)
### Extraction: P0:0 P1:0 P2:0 — 准确（MS patient, letrozole, DEXA, PT referral, no chemo benefit small, patient不interested in chemo）
### Letter: P0:0 P1:0 P2:0 — "invasive ductal carcinoma...early stage...not spread...cure...letrozole...Prolia to protect bones...DEXA...physical therapy...3 months" 每句有依据
### ROW 40 总评: Ext P2:0, Letter P2:0

---

## ROW 41 (coral_idx 180)
### Extraction: P0:0 P1:0 P2:0 — 准确（ATM carrier, 3cm grade 3 IDC, SLN micrometastasis, AC→Taxol, ovarian suppression+AI, ribociclib trial）
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "**cancer was removed, and the edges of the tissue are clean**" | margins negative ✓——**正确使用 margins rule** | ✅ |
| "**One of the lymph nodes under your arm had tiny bits of cancer**" | SLN micrometastasis ✓——**正确描述 micrometastasis** | ✅ |
| "Taxol for 12 weeks, followed by another treatment called AC" | A/P ✓ 保留药名 | ✅ |
| "take a medication to stop your ovaries from making estrogen" | ovarian suppression ✓ 通俗化 ✓ | ✅ |
| "join a study with a medication called ribociclib" | monarchE trial ✓ | ✅ |
| "port placed" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——**优秀：margins rule + micrometastasis 描述都完美**
### ROW 41 总评: Ext P2:0, Letter P2:0

---

## ROW 42 (coral_idx 181)
### Extraction: P0:0 P1:0 P2:0 — 准确（post-radiation, start tamoxifen 5 years, mammogram next visit）
### Letter: P0:0 P1:0 P2:0 — "finished radiation therapy on January 5...tamoxifen for five years...return to the clinic in 4-6 weeks...mammogram on the day of your next appointment" 简洁准确
### ROW 42 总评: Ext P2:0, Letter P2:0

## ROW 44 (coral_idx 183)
### Ext P2:0, Letter P2:0 — 全面覆盖：residual cancer, radiation trial, AI, BSO, CT chest 1yr, nutrition, PT

## ROW 46 (coral_idx 185)
### Ext P2:0, Letter P2:0 — 全面覆盖：residual cancer + 2 LN+, sarcoidosis, kidney artery, letrozole+abemaciclib, re-excision, MRA, DEXA

## ROW 49 (coral_idx 188)
### Ext P2:0, Letter P2:0 — new diagnosis IDC ER+/PR+/HER2-, mastectomy planned, tamoxifen + thrombophilia assessment, advance care planning

## ROW 50 (coral_idx 189)
### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:1
- "Lupron, letrozole, and ibrance in January 2015" — A/P: lupron+letrozole started **October 2014**, only ibrance added **January 2015**。**Same P2 as iter12e**

## ROW 51 (coral_idx 190)
### Ext P2:0, Letter P2:0 — education visit: Gemzar teaching, Zofran/Compazine, vaccines, pregnancy, social work, exercise counseling

## ROW 52 (coral_idx 191)
### Ext P2:0, Letter P2:0 — "**edges of the removed tissue are clean**" ✓ margins rule, "**a test** to learn more" ✓ (not "medication test"), Zoladex+AI, CT/bone scan, fertility preservation

## ROW 53 (coral_idx 192)
### Ext P2:0, Letter P2:0 — HER2+ IDC with neuroendocrine differentiation, AC/THP or TCHP options detailed, Arimidex 10yr, radiation, genetic counseling. Drug names preserved.

## ROW 54 (coral_idx 193)
### Ext P2:0, Letter P2:0 — BRCA2 oligometastatic: stable disease, leuprolide+letrozole, palbociclib after radiation, zoledronic acid, DEXA, PET/CT 3-4mo, return 4 weeks

## ROW 57 (coral_idx 196) — **发现新 P1！**

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | 2nd opinion ✓ | ✅ |
| second opinion | yes | ✓ | ✅ |
| **Type_of_Cancer** | **"Breast Cancer (TNBC), originally classified as Grade III adenocarcinoma, ER+/PR+/HER2-"** | 原文: "Pathology showed ***** ***** neg Result (**TNBC**)" = ER-/PR-/HER2-。**Extraction 自相矛盾——TNBC 意味着 ER-/PR-/HER2-，但写了 ER+/PR+** | **P1** |
| Stage | Locally advanced | ✓ | ✅ |
| Distant Metastasis | No | ✓ | ✅ |
| findings | residual tumor 3.7cm, 0/6 nodes, TNBC, PE | ✓ | ✅ |
| goals_of_treatment | curative | ✓ | ✅ |
| response_assessment | residual tumor, not pCR | ✓ | ✅ |
| **medication_plan** | "resume trastuzumab if confirmed; **also: pertuzumab**" | pertuzumab 是原 HER2+ 治疗的药，现在已确认 TNBC，**POST hook 错误添加** | P2 |
| radiotherapy_plan | XRT scheduled | ✓ | ✅ |
| **procedure_plan** | **"which pt is scheduled to receive"** | **garbled fragment** | P2 |
| genetic_testing_plan | genetic counseling and testing | ✓ | ✅ |

**Extraction 小结**: P0:0 **P1:1** P2:2（P1: TNBC 但写 ER+/PR+; P2: POST hook pertuzumab + garbled procedure）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "second opinion regarding your locally advanced breast cancer" | ✓ | ✅ |
| "**triple-negative breast cancer (TNBC), which means the cancer does not have receptors for estrogen, progesterone, or HER2**" | 原文 TNBC ✓——**letter 正确忽略了 extraction 的 ER+/PR+ 错误！** | ✅ |
| "still some cancer left, measuring 3.7 cm...None of the lymph nodes...had cancer" | pathology ✓ | ✅ |
| "dose of your chemotherapy was reduced by 25% after the first cycle" | ✓ | ✅ |
| "scheduled to receive radiation therapy (XRT)" | A/P ✓ | ✅ |
| "genetic counseling and testing" | A/P ✓ | ✅ |
| "If further testing confirms...trastuzumab" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——**letter 比 extraction 更准确**（正确说了 TNBC = no ER/PR/HER2）

### ROW 57 总评
- **Extraction**: P0:0 **P1:1** P2:2——**首个 iter13 extraction P1**
- **Letter**: P0:0 P1:0 P2:0——letter 自动纠正了 extraction 的 receptor status 错误
## ROW 59 (coral_idx 198): Ext P2:0, Letter P2:0 — letrozole→exemestane, Pristiq, mammogram/MRI alternating
## ROW 61 (coral_idx 200): Ext P2:0, Letter P2:0 — lumpectomy with IORT 04/12/21, Oncotype Dx, Tamoxifen vs OS+AI
## ROW 64 (coral_idx 203): Ext P2:0, Letter P2:0 — Stage III-IV, probable sternum met, biopsy planned, TCHP+xgeva
## ROW 65 (coral_idx 204): Ext P2:0, Letter P2:0 — neoadjuvant AC/T or ISPY trial, port, research biopsy/MRI
## ROW 68 (coral_idx 207): Ext P2:0, Letter P2:0 — post-TCHP good response, bilateral mastectomy recommended, sons genetic testing

## ROW 70 (coral_idx 209): Ext P2:0, Letter P2:0 — bilateral cancer, letrozole restart, radiation, expanders, CT for lung nodules

## ROW 72 (coral_idx 211) — **iter12e P1 修复验证**
### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "invasive ductal carcinoma...started in the milk ducts...focal neuroendocrine differentiation" | pathology ✓ | ✅ |
| "**the cancer was removed with surgery, and the edges of the removed tissue are clean**" | margins negative ✓——**iter12e P1 CONFIRMED FIXED** | ✅ |
| "No cancer was found in the lymph nodes" | 0/2 SLN ✓ | ✅ |
| "start taking a medication called letrozole" | ✓ | ✅ |
| "**a medication** to evaluate the potential benefit of chemotherapy" | Oncotype Dx——**"a medication"仍然 garbled** | P2 |
| "results of the **medication**" | 同上 garbled | (同上) |

**Letter 小结**: P0:0 P1:0 P2:1（"a medication" for Oncotype Dx garbled, 但 iter12e P1 "no cancer found" 已修复）
### ROW 72 总评: Ext P2:0, Letter P2:1 — **P1→P0 修复成功**

---

## ROW 78 (coral_idx 218): Ext P2:0, Letter P2:0 — TNBC metastatic, progression, trial interest, echo, radiation consult
## ROW 80 (coral_idx 219)
### Ext P2:0, Letter P2:1
- cold gloves: iter12e "hand-foot syndrome"(错) → iter13 "**hand swelling**"(改善但仍不精确——A/P说for "neuropathy and fingernails")

## ROW 82 (coral_idx 221): Ext P2:0, Letter P2:0 — low risk no chemo, radiation, DEXA, exercise counseling

## ROW 84 (coral_idx 223)
### Ext P2:1, Letter P2:0
- **PR+ 仍然错**——metastatic biopsy pathology: "Progesterone Receptor: NEGATIVE (<1%)"，extraction 仍说 PR+

## ROW 85 (coral_idx 224)
### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:2 — prednisone 剂量可能缺失 + letter 被截断（以 "such" 结尾无 closing）+ 多处 garbled from redaction

## ROW 87 (coral_idx 226): Ext P2:0, Letter P2:0 — 79yo second opinion, 2.2cm 4/19 LN+, hormonal therapy alone

## ROW 88 (coral_idx 227)
### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:1 — letter 截断（以 "stressful" 结尾无 closing）

## ROW 90 (coral_idx 229): Ext P2:0, Letter P2:0 — AC cycle 4, dose delay, GCSF 50%, granisetron+olanzapine

## ROW 91 (coral_idx 230)
### Ext P2:0, Letter P2:1
- extraction 有 everolimus 但 **letter 仍然漏了 everolimus**——只提 exemestane+denosumab（同 iter12e P2）
## ROW 92 (coral_idx 231): Ext P2:0, Letter P2:0 — Epirubicin cycle 2, liver improving, Neupogen
## ROW 95 (coral_idx 234): Ext P2:0, Letter P2:0 — post-NAC good response, AC, capecitabine after XRT

## ROW 96 (coral_idx 235)
### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:0 — **iter12e "medication testing" P2 FIXED: now says "a test to learn more"**

## ROW 97 (coral_idx 236): Ext P2:0, Letter P2:0 — Oncotype Dx (correctly named), MS/Gilenya, drain concerns

## ROW 99 (coral_idx 238): Ext P2:0, Letter P2:0 — **symptom management service** ✓ (医生feedback), biopsy + CT + 2 weeks

## ROW 100 (coral_idx 239): Ext P2:0, Letter P2:0 — **exercise 10 min 3x/day** ✓ (医生feedback), Focalin for fatigue, Gemzar stopped

## ROW 59 (coral_idx 198) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | on hormonal therapy ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 3 IDC with high-grade DCIS | surgical pathology grade 3 ✓ | ✅ |
| **Stage_of_Cancer** | **Stage IIA (pT2 N0)** | tumor 1.5cm = **pT1c** (not pT2), 0/5 LN = N0, A/P says "**stage 1**"。**应为 Stage IA** | P2 |
| findings | no recurrence, weight stable, PE normal | ✓ | ✅ |
| current_meds | exemestane, letrozole | 都在 med list 但实际在 switching ✓ | ✅ |
| recent_changes | Stopped letrozole, advised to start exemestane | A/P ✓ | ✅ |
| medication_plan | discontinue letrozole → wait 2-3wk → exemestane + Pristiq + psychiatry for duloxetine | A/P comprehensive ✓ | ✅ |
| imaging_plan | mammogram July + alternating mammo/MRI + 6mo breast exam | A/P ✓ | ✅ |
| follow_up | 6 months | ✓ | ✅ |
| Referral: Specialty | Psychiatry referral | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（Stage IIA→应为 IA）

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "No new signs of cancer growth" | "without evidence of recurrence" ✓ | ✅ |
| "weight has stayed the same since November/December 2017" | A/P ✓ | ✅ |
| "scar from your surgery looks good" | PE ✓ | ✅ |
| "stopped taking letrozole...start exemestane after a short break" | A/P ✓ | ✅ |
| "also taking Pristiq for depression" | ✓ | ✅ |
| "psychiatrist about changing your antidepressant to...Duloxetine" | A/P #2 ✓ | ✅ |
| "mammogram in July...mammograms and MRIs every six months" | A/P ✓ | ✅ |
| "see us again in six months" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0
### ROW 59 总评: Ext P2:1, Letter P2:0

---

## ROW 61 (coral_idx 200) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | newly diagnosed, here to establish care ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC | biopsy: IDC ≥11mm, ER 100%, PR 100%, HER2- (1+) ✓ | ✅ |
| Stage_of_Cancer | Stage I | 1.5cm, no imaging evidence of mets ✓ | ✅ |
| Distant Metastasis | No | CT chest/abd benign ✓ | ✅ |
| findings | IDC diagnosis + MRI findings + second site biopsy negative + CT findings | ✓ | ✅ |
| goals_of_treatment | curative | ✓ | ✅ |
| medication_plan | Tamoxifen vs Ovarian Suppression + AI | A/P ✓ | ✅ |
| procedure_plan | lumpectomy with IORT and reconstruction 04/12/21 | A/P ✓ | ✅ |
| genetic_testing_plan | Oncotype Dx after surgery | A/P ✓ | ✅ |
| follow_up | after surgery and pathology are complete | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "first visit regarding your newly diagnosed left breast cancer" | ✓ | ✅ |
| "ductal carcinoma...sensitive to hormones (ER+ and PR+)...no HER2" | ✓ | ✅ |
| "early stage (Stage I)...not spread" | ✓ | ✅ |
| "imaging tests show a small area of concern...biopsy...did not find any cancer" | MRI 0900 site → biopsy negative ✓ | ✅ |
| "small cyst in your liver, but it is not related to the cancer" | CT abd: liver cyst ✓ | ✅ |
| "Tamoxifen or another hormone therapy" | A/P ✓ | ✅ |
| "lumpectomy on April 12, 2021...IORT...will not need additional radiation" | A/P "IORT so will not need post op RT" ✓ | ✅ |
| "decide if you need chemotherapy based on the results" | Oncotype Dx after surgery ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0
### ROW 61 总评: Ext P2:0, Letter P2:0

## ROW 64 (coral_idx 203) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | new consult + second opinion ✓ | ✅ |
| second opinion | yes | ✓ | ✅ |
| Type_of_Cancer | HR+/HER2- IDC with probable sternum met | A/P "Stage III-IV...HR+/her 2 negative with probably metastatic disease to the sternum" ✓ | ✅ |
| Stage_of_Cancer | Stage III-IV | A/P ✓ | ✅ |
| goals_of_treatment | curative | oligometastatic treatment intent ✓ | ✅ |
| medication_plan | chemo + taxol + xgeva if bone bx positive | A/P ✓ | ✅ |
| imaging_plan | biopsy of sternal lesion | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter: P0:0 P1:0 P2:0 — "cancer may have spread to your sternum (the middle part of your chest bone)" ✓, biopsy planned ✓, TCHP ✓, xgeva if positive ✓, keep chemo on schedule ✓

---

## ROW 65 (coral_idx 204) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | consultation for neoadjuvant chemo ✓ | ✅ |
| Type_of_Cancer | ER weak+ (2%)/PR low+ (7%)/HER2- IDC | A/P "weakly ER and PR positive, ***** negative" ✓ | ✅ |
| Stage_of_Cancer | locally advanced with LN involvement | A/P ✓ | ✅ |
| Distant Metastasis | No | PET/CT no distant mets ✓ | ✅ |
| medication_plan | neoadjuvant AC/T or ISPY trial | A/P ✓ | ✅ |
| procedure_plan | port placement | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter: P0:0 P1:0 P2:0 — neoadjuvant AC/T ✓, ISPY trial option ✓, port ✓, research biopsy/MRI ✓, 5-10 yr endocrine ✓

---

## ROW 68 (coral_idx 207) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2+ multifocal IDC | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage I | extraction inference ✓ | ✅ |
| response_assessment | good response after 6 cycles TCHP, MRI no visible lesions | ✓ | ✅ |
| procedure_plan | bilateral mastectomy | A/P ✓ | ✅ |
| genetic_testing_plan | sons should be tested | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "ER+/PR+/HER2+ multifocal invasive ductal carcinoma" | ✓ | ✅ |
| "good response to the treatment...6 cycles of TCHP" | ✓ | ✅ |
| "MRI shows the cancer has responded well to treatment" | follow-up MRI no visible lesions ✓ | ✅ |
| "bilateral mastectomy...surgery to remove both breasts" | A/P ✓ | ✅ |
| "if you choose to have a lumpectomy...need radiation" | A/P ✓ | ✅ |
| "sons should be tested for a type of anemia and the risk for pancreatic cancer" | A/P ✓——但 "a type of anemia" 比 iter12e 的 "medication-related anemia" 更准确 | ✅ |
| "healthy diet and regular exercise" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0——**iter12e P2 (sons 遗传筛查 "medication-related anemia") 已改善为 "a type of anemia"**

---

## ROW 70 (coral_idx 209) — 详细审查

### Extraction 逐字段审查

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- ILC (left) + ER+/PR-/HER2- IDC (right) | A/P: left PR+, right PR- ✓——**bilateral cancer 正确区分两侧 receptor** | ✅ |
| Stage_of_Cancer | Stage II-III | locally advanced ✓ | ✅ |
| Distant Metastasis | No | ✓ | ✅ |
| current_meds | letrozole | restart ✓ | ✅ |
| medication_plan | restart letrozole | A/P ✓ | ✅ |
| radiotherapy_plan | radiation consult | A/P ✓ | ✅ |
| imaging_plan | CT due June for lung nodules | A/P ✓ | ✅ |
| follow_up | September | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0——bilateral cancer 处理准确

### Letter: P0:0 P1:0 P2:0 — "recovering well...Two lymph nodes positive...restart letrozole...radiation consult...expanders...CT for lung nodules...follow-up September" 全部准确

## ROW 78 (coral_idx 218) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer has gotten worse...liver and lymph nodes has grown" | imaging progression ✓ | ✅ |
| "small spot in your lung...gotten larger" | ✓ | ✅ |
| "interested in joining a clinical trial" | A/P ✓ | ✅ |
| "not interested in chemotherapy at this time" | A/P ✓ | ✅ |
| "radiation to treat the cancer" | radiation consult ✓ | ✅ |
| "echocardiogram on September 8th" | echo ✓ | ✅ |

### ROW 78 总评: Ext P2:0, Letter P2:0

---

## ROW 82 (coral_idx 221) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "4.3 cm tumor with some lymph nodes involved" | pathology ✓ | ✅ |
| "ER positive, PR positive...no HER2" | ✓ | ✅ |
| "We decided not to start chemotherapy because your risk is low" | A/P "low risk" ✓ | ✅ |
| "appointment...tomorrow to discuss radiation" | A/P ✓ | ✅ |
| "DEXA scan to check your bone health" | A/P ✓ | ✅ |
| "exercise counseling" | A/P ✓ | ✅ |

### ROW 82 总评: Ext P2:0, Letter P2:0

---

## ROW 87 (coral_idx 226) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "second opinion...grade 2, 2.2 cm, 4/19 lymph nodes" | A/P ✓ | ✅ |
| "clear margins...small area of cancer spreading beyond the capsule" | extracapsular extension ✓ | ✅ |
| "ER+, PR+, no HER-2/neu" | ✓ | ✅ |
| "pill-rolling tremor of Parkinson's disease" | PE ✓ | ✅ |
| "hormonal therapy alone" | patient choice due to age+Parkinson's ✓ | ✅ |
| "radiation to prevent local recurrence" | A/P discussed ✓ | ✅ |

### ROW 87 总评: Ext P2:0, Letter P2:0

---

## ROW 90 (coral_idx 229) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "adenocarcinoma...Stage II/III" | A/P ✓ | ✅ |
| "thyroid stimulating hormone level is higher than normal" | TSH 6.01 (H) ✓ | ✅ |
| "low red blood cell count...anemia" | Hgb 11.3 (L) ✓ | ✅ |
| "GCSF dose has been reduced to 50%" | A/P ✓ | ✅ |
| "granisetron and olanzapine" | A/P ✓ | ✅ |
| "continue AC...cycle 4 in about a week" | A/P ✓ | ✅ |
| "visit after radiation therapy, 1-1.5 months" | A/P "RTC after XRT" ✓ | ✅ |

### ROW 90 总评: Ext P2:0, Letter P2:0

---

## ROW 92 (coral_idx 231) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "liver feels smaller and less tender" | A/P "Exam improved of liver" ✓ | ✅ |
| "Epirubicin...Neupogen for 2 days" | A/P ✓ | ✅ |
| "echocardiogram to monitor heart function" | cardiac monitoring for anthracycline ✓ | ✅ |
| "liver function and tumor markers" | A/P ✓ | ✅ |

### ROW 92 总评: Ext P2:0, Letter P2:0

---

## ROW 95 (coral_idx 234) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "MRI...cancer has decreased...10x8x8mm compared to 16x16x15mm" | MRI response ✓ | ✅ |
| "lymph nodes...gotten smaller but not completely gone" | ✓ | ✅ |
| "started AC" | A/P ✓ | ✅ |
| "capecitabine after finishing radiation" | CREATE-X trial ✓ | ✅ |
| "referred to radiation oncologist" | A/P ✓ | ✅ |
| "strongly recommended to take hormone therapy" | A/P ✓ | ✅ |

### ROW 95 总评: Ext P2:0, Letter P2:0

---

## ROW 97 (coral_idx 236) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "invasive ductal carcinoma...early stage...not spread" | ✓ | ✅ |
| "no cancer found in the lymph nodes" | negative SLN ✓ | ✅ |
| "drain in place...concerned about its status" | A/P drain issue ✓ | ✅ |
| "starting a medication to help prevent cancer from coming back" | endocrine therapy ✓ | ✅ |
| "continuing your current medication for multiple sclerosis" | Gilenya ✓ | ✅ |
| "**test called Oncotype Dx**" | A/P ✓——**正确命名** Oncotype Dx | ✅ |
| "referred to a radiation oncologist" | A/P ✓ | ✅ |

### ROW 97 总评: Ext P2:0, Letter P2:0

---

## ROW 99 (coral_idx 238) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer that has spread to...left lung and lymph nodes in the middle of your chest" | mediastinal LAD + lung mass ✓ | ✅ |
| "cancer in your chest area has gotten smaller, but...lung has grown" | A/P mixed response ✓ | ✅ |
| "switched from anastrozole to letrozole because of joint pains" | ✓ | ✅ |
| "Exemestane and Afinitor, or...Xeloda...chemotherapy" | A/P future options ✓ | ✅ |
| "biopsy of the cancer in your lung or chest area" | A/P needs biopsy ✓ | ✅ |
| "new CT scan with contrast" | A/P ✓ | ✅ |
| "**referred to a symptom management service**" | A/P ✓——**医生 feedback 要求的！** | ✅ |
| "return to the clinic in about 2 weeks" | A/P ✓ | ✅ |

### ROW 99 总评: Ext P2:0, Letter P2:0 — **symptom management service 正确包含**

---

## ROW 100 (coral_idx 239) — 详细审查

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句审查

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "tumor markers have gone up...Cancer Antigen 15-3 and Cancer Antigen 27.29" | A/P "Tumor markers rose" ✓ | ✅ |
| "hemoglobin and hematocrit levels are low...anemic" | labs ✓ | ✅ |
| "no new signs of cancer spreading" | A/P "Exam stable" ✓ | ✅ |
| "Gemzar treatment was stopped by you" | patient wants break ✓ | ✅ |
| "Focalin as needed to help with fatigue" | A/P ✓ | ✅ |
| "**exercise for 10 minutes three times a day**" | A/P "Rec exercise 10 min 3 x a day" ✓——**医生 feedback 要求的！** | ✅ |
| "discuss taking a break from your treatment with your doctor" | A/P ✓ | ✅ |

### ROW 100 总评: Ext P2:0, Letter P2:0 — **exercise recommendation 正确包含**

---

# 补充详细审查（之前只有一行摘要的 ROW）

以下 ROW 在之前的审查中只有一行总结，现在补充逐字段 extraction 和逐句 letter 表格。

## ROW 34 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | ✅ |
| Type_of_Cancer | ER+/PR-/HER2- IDC with extensive DCIS | pathology ✓ (2020 recurrence: ER+ 100%, PR+ 50%, HER2-) | ✅ |
| Stage_of_Cancer | Stage III, now with local recurrence | A/P ✓ | ✅ |
| current_meds | arimidex | ✓（self-D/C'd but was on it） | ✅ |
| medication_plan | tamoxifen 20mg PO qD | A/P ✓ | ✅ |
| radiotherapy_plan | chest wall RT recommended | A/P ✓ | ✅ |
| lab_plan | check labs | A/P ✓ | ✅ |
| follow_up | 6 months | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "cancer has returned in your left breast" | second local relapse ✓ | ✅ |
| "small area of cancer that was removed...also saw a small spot on imaging" | pathology + PET 6th rib ✓ | ✅ |
| "start taking...tamoxifen" | A/P ✓ | ✅ |
| "referred to radiation oncology for chest wall radiation" | A/P ✓ | ✅ |
| "basic blood tests" | A/P "check labs" ✓ | ✅ |
| "return to the clinic in 6 months" | A/P ✓ | ✅ |

**ROW 34 总评**: Ext P2:0, Letter P2:0

## ROW 36 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | cycle 8 abraxane ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage IIIA (pT3N0) | ✓ | ✅ |
| findings | arm swelling, doppler for DVT, no new disease | ✓ | ✅ |
| current_meds | Abraxane, zoladex | ✓ | ✅ |
| recent_changes | Switched to Abraxane after grade 3 reaction to Taxol | ✓ | ✅ |
| medication_plan | Continue Abraxane + zoladex + antiemetics; also: tamoxifen | tamoxifen correctly on current meds ✓ | ✅ |
| imaging_plan | doppler to r/o DVT | A/P ✓ | ✅ |
| radiotherapy_plan | radiation referral | A/P ✓ | ✅ |
| follow_up | 2 weeks | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "swelling in your right arm and hand...doppler test...blood clot" | A/P ✓ 通俗化 DVT ✓ | ✅ |
| "No new cancer growth" | ✓ | ✅ |
| "switched to Abraxane after having a bad reaction to Taxol" | ✓ | ✅ |
| "continue weekly Abraxane" | ✓ | ✅ |
| "radiation oncology" | ✓ | ✅ |
| "2 weeks" | ✓ | ✅ |

**ROW 36 总评**: Ext P2:0, Letter P2:0

## ROW 40 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 1 IDC | pathology: ER 95%, PR 5%, HER2 neg ✓ | ✅ |
| Stage_of_Cancer | Stage II (pT2 N1mi) | reasonable ✓ | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| medication_plan | letrozole + Prolia | A/P ✓ | ✅ |
| imaging_plan | DEXA | A/P ✓ | ✅ |
| Referral: Others | PT referral | A/P ✓ | ✅ |
| follow_up | 3 months | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "early stage...not spread...cure" | ✓ | ✅ |
| "letrozole...Prolia to protect bones...DEXA...physical therapy...3 months" | A/P 全覆盖 ✓ | ✅ |

**ROW 40 总评**: Ext P2:0, Letter P2:0

## ROW 42 — 补充表格

### Extraction: 全部准确（post-radiation, start tamoxifen 5yr, mammogram next visit, RTC 4-6 wk）
### Letter: "finished radiation...tamoxifen for five years...return 4-6 weeks...mammogram" 简洁准确

| Letter 句子 | 判定 |
|------------|------|
| "finished radiation therapy on January 5 and it went well" | ✓ | ✅ |
| "tamoxifen for five years" | A/P ✓ | ✅ |
| "return to the clinic in 4-6 weeks" | ✓ | ✅ |
| "routine mammogram to check your breast" at next appt | ✓ | ✅ |

**ROW 42 总评**: Ext P2:0, Letter P2:0

## ROW 44 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC with residual DCIS | pathology ✓ | ✅ |
| Stage | Originally Stage II, post-NAC | ✓ | ✅ |
| medication_plan | AI after radiation, possible BSO | A/P ✓ | ✅ |
| radiotherapy_plan | clinical trial 3 vs 5 weeks | A/P ✓ | ✅ |
| imaging_plan | CT chest in 1 year for lung nodule | A/P ✓ | ✅ |
| Referral | radiation, nutrition, PT | A/P ✓ | ✅ |
| follow_up | 01/05/19 | ✓ | ✅ |

### Letter: 全面覆盖 residual cancer, radiation trial, AI, BSO discussion, CT, nutrition, PT

**ROW 44 总评**: Ext P2:0, Letter P2:0

## ROW 46 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 1 IDC | ER+ 95%, PR- (0%), HER2- IHC 1+ ✓ | ✅ |
| Stage | Stage IIB (pT2N1) | 3.5cm residual + 2/2 SLN+ ✓ | ✅ |
| current_meds | letrozole | started today ✓ | ✅ |
| medication_plan | letrozole + abemaciclib after XRT + naproxen + allegra + iron | A/P ✓ | ✅ |
| imaging_plan | MRA abdomen 1yr + DEXA | A/P ✓ | ✅ |
| procedure_plan | re-excision of margins | positive margins ✓ | ✅ |
| follow_up | 2-3 months | ✓ | ✅ |

### Letter: 准确覆盖 residual cancer, 2 LN+, sarcoidosis, renal artery, letrozole, abemaciclib, re-excision, MRA, DEXA

**ROW 46 总评**: Ext P2:0, Letter P2:0

## ROW 49 — 补充表格

### Extraction: 全部准确（new diagnosis IDC ER+/PR+/HER2-, Stage 2, mastectomy 01/06/17, tamoxifen+thrombophilia, XRT discussed）
### Letter: "mastectomy...plan depends on surgery results...tamoxifen...blood clots...radiation" 准确

**ROW 49 总评**: Ext P2:0, Letter P2:0

## ROW 50 — 补充表格

### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:1（medication timing: lupron+letrozole Oct 2014 vs letter says all three Jan 2015）

**ROW 50 总评**: Ext P2:0, Letter P2:1

## ROW 51 — 补充表格

### Extraction: P0:0 P1:0 P2:0（教育性 note，Type_of_Cancer 空是正确的——note 不含诊断信息）
### Letter: P0:0 P1:0 P2:0（Gemzar teaching, Zofran/Compazine, vaccines, pregnancy, social work, exercise）

**ROW 51 总评**: Ext P2:0, Letter P2:0

## ROW 52 — 补充表格

### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:0（"edges of removed tissue are clean" ✓ margins rule, "a test" ✓ not "medication test", Zoladex+AI, fertility preservation）

**ROW 52 总评**: Ext P2:0, Letter P2:0

## ROW 53 — 补充表格

### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:0（HER2+ IDC, AC/THP or TCHP options detailed, Arimidex 10yr, radiation, genetic counseling）

**ROW 53 总评**: Ext P2:0, Letter P2:0

## ROW 54 — 补充表格

### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:0（BRCA2 oligometastatic: stable disease, leuprolide+letrozole, palbociclib after radiation, zoledronic acid, DEXA, PET/CT）

**ROW 54 总评**: Ext P2:0, Letter P2:0

## ROW 80 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 3 IDC | pathology ✓ | ✅ |
| Stage | Stage I | skin recurrence post mastectomy ✓ | ✅ |
| medication_plan | TC x4, cold gloves, claritin | A/P ✓ | ✅ |
| radiotherapy_plan | 6 weeks (5+1 boost), left axilla and SC fields | A/P ✓ | ✅ |
| follow_up | cycle 2 | ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "TC on April 11, 2019...four times" | ✓ | ✅ |
| "dexamethasone, ondansetron, prochlorperazine" | ✓ | ✅ |
| "**cold gloves...prevent hand swelling**" | A/P says cold gloves for "**neuropathy and fingernails**"——"hand swelling"改善了iter12e的"hand-foot syndrome"但仍不精确 | P2 |
| "six weeks of radiation...five weeks with a one-week boost" | ✓ | ✅ |

**ROW 80 总评**: Ext P2:0, Letter P2:1（cold gloves 用途描述不精确）

## ROW 84 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| **Type_of_Cancer** | **ER+/PR+/HER2-** | metastatic biopsy pathology: "Progesterone Receptor: **NEGATIVE** (<1%)"——**PR 应为阴性** | **P2** |
| Stage | Stage IV (metastatic) | ✓ | ✅ |
| Distant Metastasis | Yes, to bone, soft tissue, liver, meninges | ✓ | ✅ |
| medication_plan | continue Xeloda, consider fulvestrant+inhibitor if progression | A/P ✓ | ✅ |
| imaging_plan | CT CAP + LP + MRI spine | A/P ✓ | ✅ |
| radiotherapy_plan | radiation referral for CNS | A/P ✓ | ✅ |

### Letter: 准确覆盖 brain/liver progression, Xeloda, LP, CT, MRI spine, radiation consult

**ROW 84 总评**: Ext **P2:1**（PR+ 错——metastatic biopsy PR-）, Letter P2:0

## ROW 85 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR-/HER2- ILC with pleomorphic features | pathology ✓ | ✅ |
| Stage | Originally Stage IIIA, now metastatic | ✓ | ✅ |
| recent_changes | prednisone dose decreased | A/P ✓ | ✅ |
| **medication_plan** | "Continue steroid taper...pain meds; **also: palbociclib**" | **palbociclib 已因 progression 停用！** POST hook 错误添加 | **P2** |
| therapy_plan | phase 1 trial +olaparib | A/P ✓ | ✅ |
| radiotherapy_plan | radiation washout for trial | A/P ✓ | ✅ |

### Letter: P2x2（prednisone 剂量缺失 "to /d" + letter 截断 + garbled from redaction）

**ROW 85 总评**: Ext **P2:1**（POST hook adds stopped palbociclib）, Letter P2:2

## ROW 88 — 补充表格

### Extraction: P0:0 P1:0 P2:0
### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "IDC...weakly positive for ER...negative for PR...no HER2" | ✓ | ✅ |
| "cancer has spread to your brain and other parts" | brain + lung + LN mets ✓ | ✅ |
| "capecitabine (Xeloda)" | ✓ | ✅ |
| "clinical trials using immunotherapy" | ✓ | ✅ |
| "If cancer tests positive for HER2...different treatment" | A/P HER2 testing recommended ✓ | ✅ |
| "restaging after three months" | ✓ | ✅ |

**注意**: Letter 截断（以 "stressful" 结尾无 closing）— P2

**ROW 88 总评**: Ext P2:0, Letter P2:1（truncation）

## ROW 91 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR+ IDC with metastatic recurrence to bone | ✓ | ✅ |
| Stage | Originally Stage I, now metastatic | ✓ | ✅ |
| current_meds | everolimus, exemestane, denosumab | ✓ 包含 everolimus | ✅ |
| recent_changes | Started lasix + KCL | ✓ | ✅ |
| medication_plan | continue lasix+KCL+denosumab+antifungal; also: exemestane | exemestane correct ✓ | ✅ |
| imaging_plan | PET/CT next week | ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "cancer in your right hip has grown" | imaging ✓ | ✅ |
| "lasix to help with swelling and potassium" | ✓ | ✅ |
| "continue taking exemestane and denosumab" | ✓ 但**漏了 everolimus**（extraction 有它） | P2 |
| "PET/CT scan next week" | ✓ | ✅ |
| "blood tests every month...next visit in one month" | ✓ | ✅ |

**ROW 91 总评**: Ext P2:0, Letter P2:1（漏 everolimus）

## ROW 96 — 补充表格

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade I mixed ductal and cribriform with tubular | pathology ✓ | ✅ |
| Stage | Stage IA (pT1cN0) | ✓ | ✅ |
| medication_plan | start tamoxifen after radiation | A/P ✓ | ✅ |
| radiotherapy_plan | radiation consult | A/P ✓ | ✅ |
| genetic_testing_plan | Oncotype/MammaPrint (redacted) | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "ER+/PR+/HER2- grade I mixed ductal and cribriform" | ✓ | ✅ |
| "early-stage cancer...not spread" | ✓ | ✅ |
| "**a test** to learn more about your cancer" | ✓——**iter12e "medication testing" P2 已修复！** | ✅ |
| "radiation oncology for a consult" | ✓ | ✅ |
| "start tamoxifen after completing radiation" | ✓ | ✅ |

**ROW 96 总评**: Ext P2:0, Letter P2:0——**"medication testing" → "a test" 修复确认**
