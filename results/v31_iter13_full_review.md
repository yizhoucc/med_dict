# V31 iter13 — Extraction + Letter 逐字段+逐句审查

> 56 samples，每个 sample 审查：
> 1. Extraction 每个字段 vs 原文
> 2. Letter 每句 vs 原文
> P0=幻觉/编造 | P1=重大错误 | P2=小问题

## 状态
- 审查中: ROW 1 开始
- 已完成: 0/56
- Extraction: P0:0 P1:0 P2:0
- Letter: P0:0 P1:0 P2:0

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
