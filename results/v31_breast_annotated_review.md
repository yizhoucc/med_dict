# Auto Review: results.txt

Generated: 2026-04-29 15:55
Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)

## Summary

- **Samples**: 20
- **Clean**: 0/20
- **P0** (hallucination): 0
- **P1** (major error): 6
- **P2** (minor issue): 141

### Critical Issues

- **ROW 1** [P1]: The stage is inaccurate. The patient has Stage IIB (pT2N1a) triple negative breast cancer, not early-stage (Stage I-II).
- **ROW 12** [P1]: Incorrectly listed 'tc' as current medication when the patient is not currently on any cancer-related medications.
- **ROW 16** [P1]: The description 'cancer that started in the milk-producing glands' is not accurate for invasive lobular carcinoma.
- **ROW 17** [P1]: The note does not mention any current cancer-related medications, but the extracted data incorrectly states 'ac'.
- **ROW 17** [P1]: The note mentions uncertainty about the type of genetic testing done, but the extracted data incorrectly states 'brca'.
- **ROW 18** [P1]: The letter does not specify the exact type of breast cancer, which is important for a new patient.

---

## ROW 1 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (HER2 IHC 1; FISH ratio 2.1, but with HER2 sig/nuc 3.0 and Cen17 s
**Stage**: Stage IIB (pT2N1a)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions no cancer-related medications. | No specific medications or future medication plans were detailed in the note. |
| P2 | Treatment_Changes | The field is empty, but the note mentions no recent treatment changes or supportive medications. | No specific medications or future medication plans were detailed in the note. |
| P2 | Treatment_Goals | The goal is listed as 'curative', but the note suggests uncertainty and a need for further evaluation before making final treatment recommendations. | We discussed the likelihood of recurrence which is quite difficult to assess, but certainly notable  |

*Extraction summary*: Most fields are clean, but there are minor issues with treatment goals and the absence of cancer-related medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The stage is inaccurate. The patient has Stage IIB (pT2N1a) triple negative breast cancer, not early-stage (Stage I-II). | You have been diagnosed with an early-stage (Stage I-II) triple negative breast  |
| P2 | The sentence is overly simplified and omits important information about the patient's medical conditions such as congestive heart failure and diabetes. | Your blood tests show low hemoglobin and hematocrit levels, and slightly high gl |

*Letter summary*: The letter contains inaccuracies regarding the stage of cancer and lacks important context about the patient's medical conditions.

---

## ROW 2 — ⚠️ ISSUES

**Type**: ER+/PR-/HER2- grade 1 infiltrating ductal carcinoma
**Stage**: Originally Stage IIA, now metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary does not mention the patient's initial treatment history and the discussion about potential treatment options. | The patient is a 73-year-old woman with locally recurrent, unresectable, hormone-receptor positive b |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is incorrectly stated as 'Originally Stage IIA, now metastatic (Stage IV)', whereas the note does not specify the original stage as IIA. | This patient is a 73-year-old woman with locally recurrent, unresectable, hormone-receptor positive  |
| P2 | Cancer_Diagnosis.Distant_Metastasis | The distant metastasis is incorrectly stated as 'Yes, to the liver and chest wall', whereas the note mentions only the chest wall involvement and a small cyst in the liver. | Her abdomen shows a small, well-demarcated, low-attenuation lesion in the liver consistent with cyst |
| P2 | Treatment_Changes.recent_changes | The recent changes state 'Started on zoledronic acid because of her osteoporosis and need to start her on an aromatase inhibitor.' The note specifies that the patient was started on zoledronic acid on January 9. | On my visit with her on January 09, I started her on zoledronic acid because of her osteoporosis and |
| P2 | Treatment_Goals.goals_of_treatment | The goal is stated as 'palliative', whereas the note suggests a more complex treatment plan aiming for long-term disease control. | I think it would be most appropriate to treat this disease with hormone therapy and follow it closel |
| P2 | Response_Assessment.response_assessment | The response assessment states 'The patient is not yet on treatment — no response to assess.', whereas the note mentions starting zoledronic acid on January 9. | On my visit with her on January 09, I started her on zoledronic acid because of her osteoporosis and |
| P2 | Medication_Plan.medication_plan | The medication plan states 'Will start aromatase inhibitor therapy either alone or on CALGB 4503 randomized to placebo or bevacizumab on January 25.; also: zoledronic'. The note specifies 'on January 25 to begin aromatase inhibitor therapy either alone or on CALGB 4503 randomized to placebo or bevacizumab'. | She will return to my clinic on January 25 to begin aromatase inhibitor therapy either alone or on C |
| P2 | radiotherapy_plan.radiotherapy_plan | The radiotherapy plan states 'Given the limited radiation field that she initially had, it would be possible to radiate this area in order to approach the possibility of long-term disease control.' The note suggests this as a future possibility, not a current plan. | Given the limited radiation field that she initially had, I think it would be possible to shrink the |

*Extraction summary*: Several fields contain minor inaccuracies or imprecisions, but no major errors.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have a type of breast cancer called ER+/PR-/HER2- grade 1 infiltrating ducta |
| P2 | Unexplained medical jargon. | A recent PET/CT scan showed that a nodule in your chest wall has grown, while so |
| P2 | Unexplained medical jargon. | An MRI of your brain showed no signs of cancer. |
| P2 | Unexplained medical jargon. | You started taking zoledronic acid to help with your osteoporosis and to prepare |
| P2 | Unexplained medical jargon. | You will start an aromatase inhibitor therapy either alone or on a clinical tria |

*Letter summary*: The letter contains several instances of unexplained medical jargon that may be confusing to a layperson, but it is otherwise clean.

---

## ROW 3 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (HER2 FISH neg, ratio 1.2) spindle cell metaplastic carcinoma
**Stage**: Locally advanced, multifocal

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate description of the patient's stage. The note mentions 'early stage' but the patient has a locally advanced, multifocal cancer. | Patient with early stage breast cancer here to discuss neoadjuvant therapy |
| P2 | Current_Medications.current_meds | Missing current cancer-related medications. The note does not mention any specific cancer-related medications, but the field should be explicitly stated as empty. | No specific cancer-related medications mentioned. |
| P2 | Treatment_Changes.recent_changes | Field is empty, but the note does not indicate any recent changes in treatment. | No recent changes in treatment mentioned. |
| P2 | Treatment_Changes.supportive_meds | Field is empty, but the note does not indicate any supportive medications related to cancer treatment. | No supportive medications related to cancer treatment mentioned. |
| P2 | Procedure_Plan.procedure_plan | The field only mentions the Medi-port, but the note also discusses the need for a chemo teaching session. | We will arrange for chemo teaching session |
| P2 | Imaging_Plan.imaging_plan | The field mentions Mammaprint and Echocardiogram, but the note specifies that the Mammaprint is pending and the Echocardiogram has already been done. | Mammaprint is pending.. Echocardiogram |

*Extraction summary*: Most fields are accurate, but there are minor issues with the description of the patient's stage, current medications, treatment changes, procedure plan, and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have a rare type of breast cancer called spindle cell metaplastic carcinoma. |
| P2 | Unexplained medical jargon. | Your labs will be checked again to monitor your creatinine and Alk phos levels. |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 4 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (TNBC) grade 3 invasive ductal carcinoma
**Stage**: Stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not specify the stage of cancer, but the extraction infers 'Stage III'. | The note mentions 'R breast cancer' but does not explicitly state the stage. |
| P2 | Treatment_Goals.goals_of_treatment | The goal is listed as 'palliative', but the note suggests an intent to treat with curative intent. | Plan includes chemotherapy with taxol x 12 weekly, suggesting a curative approach. |

*Extraction summary*: Most fields are clean, but there are minor issues with the inferred stage and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'triple-negative' may be too technical for an 8th-grade reading level. | You have a type of breast cancer called triple-negative (TNBC) grade 3 invasive  |
| P2 | The term 'stage III' might be confusing without context. | The cancer is at stage III. |
| P2 | The term 'Taxol' might be unfamiliar to some patients. | You will start chemotherapy with Taxol, which will happen every week for 12 week |

*Letter summary*: Letter contains minor readability issues that could be improved for better clarity.

---

## ROW 5 — ⚠️ ISSUES

**Type**: Left breast: ER+/PR+/HER2- grade 3 invasive ductal carcinoma with suspicion for 
**Stage**: Left breast: Stage III (T3N1); Right breast: Stage I (T1cN0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary is slightly imprecise as it does not mention the patient's concern about the benefits of chemotherapy. | She is concerned that the benefits of chemotherapy will not outweigh its long-term risks. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions that the patient has not yet started any treatment, so this is not a major issue. | Not yet on treatment — no response to assess. |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note discusses potential side effects of chemotherapy, implying supportive care may be needed. | We discussed extensively the typical SE associated with TC chemotherapy, including fatigue, myelosup |
| P2 | Procedure_Plan.procedure_plan | The field is incorrectly labeled as 'AI therapy for at least 5 years', which is a medication plan, not a procedure. | AI therapy for at least 5 years |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, treatment changes, and procedure plan classification.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'invasive ductal carcinoma'. | The left breast cancer is a grade 3 invasive ductal carcinoma (cancer that start |
| P2 | Unexplained jargon 'protein called HER2'. | Both cancers are sensitive to hormones and do not have a protein called HER2. |
| P2 | Unexplained jargon 'aromatase inhibitor'. | You will also take an aromatase inhibitor (a medication that blocks the producti |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 6 — ⚠️ ISSUES

**Type**: ER-neg, PR neg, HER2 3+, FISH ratio 13, Ki67 10-15% invasive ductal carcinoma (I
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but there are no cancer-related medications mentioned in the note. | No cancer-related medications are listed in the note. |
| P2 | Treatment_Changes | Supportive medication 'ondansetron' is listed, but it is not specifically mentioned as being used for cancer treatment. | ondansetron (ZOFRAN) 8 mg tablet To be taken as needed for nausea ( see colonoscopy prep instruction |
| P2 | Treatment_Goals | The goal is listed as 'palliative', but the note suggests anticipation of 'excellent response and possible long term disease control'. This implies a more aggressive treatment approach than purely palliative. | I anticipate excellent response and possible long term disease control. |

*Extraction summary*: Most fields are clean, but there are minor issues with the classification of treatment goals and the inclusion of supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'invasive ductal carcinoma' is not explained further in simple terms. | You have a type of breast cancer called invasive ductal carcinoma (IDC), which m |
| P2 | Terms 'HR-negative' and 'HER2-positive' are not explained. | The cancer is HR-negative and HER2-positive, and it has spread to your bones. |
| P2 | The term 'MRI' is used without context. | You will have an MRI of your pelvis and neck to check for any changes in your ca |

*Letter summary*: Letter is mostly clean but requires minor clarifications for better readability.

---

## ROW 7 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- triple negative invasive ductal carcinoma
**Stage**: Originally Stage IIB, now metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis | The note mentions multiple cancers (breast, colon, endometrial), but the extracted data only focuses on the breast cancer. | Diagnoses: 1) July 2013: R breast invasive ductal carcinoma, ER-/PR-/Her2- (TNBC), grade 3 2) March  |
| P2 | Current_Medications | The extracted data only lists 'pembrolizumab', but the note mentions 'nab paclitaxel' and 'pembrolizumab'. | She started pembrolizumab and abraxane on 03/11/19 and presents today for cycle 1 day 8. |
| P2 | Treatment_Changes | The extracted data mentions restarting Xarelto, but the note indicates it was stopped for port placement. | Restart xarelto, stop for port placement per IR |
| P2 | Imaging_Plan | The extracted data specifies 'PET/CT', but the note does not mention a specific imaging plan beyond 'imaging guided by symptoms or every 3-4 months'. | Imaging guided by symptoms or every 3-4 months, longer intervals if stable. |

*Extraction summary*: Most fields are clean, but there are minor issues related to incomplete cancer diagnoses, medication listing, treatment changes, and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The sentence omits specific details about the location and nature of the new areas of cancer activity, which might be important for completeness. | Imaging studies show that your cancer has progressed. Specifically, a lesion in  |
| P2 | The sentence does not mention that xarelto is stopped temporarily for port placement. | You will continue taking abraxane and pembrolizumab. You will also restart xarel |

*Letter summary*: Letter is mostly clean but could benefit from additional details for completeness.

---

## ROW 8 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma with extensive DCIS
**Stage**: Stage IIA (pT2(m)N1a)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note does not explicitly state that the patient is not on any cancer-related medications. | The note does not mention any current cancer-related medications. |
| P2 | Treatment_Changes | The field is empty, but the note discusses upcoming treatment changes. | Patient agreed to start chemotherapy with AC + paclitaxel, estimated start date 10/18/2017. |
| P2 | follow_up_next_visit | The field states 'Not specified in the note,' but the note mentions an estimated start date for chemotherapy. | Estimated start date 10/18/2017. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related treatment information and follow-up details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'checking the lymph nodes' may be unclear to a layperson. | You recently had surgery to remove your breast cancer, which included removing b |
| P2 | The term 'grade 2' may be unclear to a layperson. | The cancer was a type that started in the milk ducts and was grade 2. |
| P2 | The term 'AC plus paclitaxel' may be unclear to a layperson. | You agreed to start chemotherapy with AC plus paclitaxel, which is scheduled to  |

*Letter summary*: The letter contains minor readability issues that could be improved for better clarity.

---

## ROW 9 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 IDC (micropapillary features) with metastatic recurrence
**Stage**: Originally Stage III (T3N2), now metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medications. | gabapentin (NEURONTIN) 300 mg capsule, hydroCHLOROthiazide (HYDRODIURIL) 25 mg tablet |
| P2 | Treatment_Changes | Supportive medications not mentioned. | States she was prescribed gabapentin for numbness and a deep sharp pain of her elbow that shoots up/ |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' but could be more specific. | if she has HR+/HER2- MBC, it is not curable, but treatable, with primary goals of prolongation of li |
| P2 | Response_Assessment | Mentions no active cancer treatment, but patient is on gabapentin for pain. | The patient has a locally advanced, unresectable recurrence of HR+/HER2- grade 2 IDC with micropapil |
| P2 | Genetic_Testing_Plan | Incorrectly labeled as genetic testing plan. | plan to FNA the mass on left lateral anterior neck |

*Extraction summary*: Most fields are clean, but there are minor issues with missing current cancer-related medications, supportive medications, and mislabeling of the genetic testing plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon (ER+/PR+/HER2- grade 2 IDC with micropapillary features). | You have a type of breast cancer called ER+/PR+/HER2- grade 2 IDC with micropapi |
| P2 | Unexplained medical jargon (left cervical lymph node). | The cancer has now spread to your left breast and left cervical lymph node, maki |
| P2 | Unexplained medical jargon (FNA). | You will have a fine needle aspiration (FNA) of the mass in your left neck to co |

*Letter summary*: Letter contains some unexplained medical jargon that could be simplified further.

---

## ROW 10 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma
**Stage**: T2N1, clinical stage II

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The extracted type of cancer does not fully match the note. The note specifies 'Grade 2 IDC, ER >95%, PR 25%, HER2 neg', whereas the extracted data says 'ER+/PR+/HER2- grade 2 invasive ductal carcinoma'. The PR status is not fully specified. | Grade 2 IDC, ER >95%, PR 25%, HER2 neg by IHC |
| P2 | Current_Medications.current_meds | The extracted data misses the current prescription of tamoxifen, which is a cancer-related medication. | tamoxifen (NOLVADEX) 20 mg tablet |

*Extraction summary*: Most fields are clean, but there are minor issues with the type of cancer description and the omission of tamoxifen in current medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'invasive ductal carcinoma' is not explained in simple terms. | You have been diagnosed with a grade 2 invasive ductal carcinoma (cancer that st |
| P2 | The term 'lymph node' might be unfamiliar to some readers. | The cancer has spread to one lymph node under your right armpit but has not spre |
| P2 | The term 'cold caps' might be unfamiliar to some readers. | You are also considering the use of cold caps during chemotherapy to reduce hair |

*Letter summary*: Letter contains minor readability issues related to unexplained medical terms.

---

## ROW 11 — ⚠️ ISSUES

**Type**: ER+/PR+ ductal carcinoma in situ (DCIS) with intermediate nuclear grade, solid a
**Stage**: pTisNx

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary is slightly imprecise. It should mention the discussion about potential adjuvant anti-hormonal therapy and the pending radiation oncology assessment. | I had a detailed discussion with Mrs. ***** about the nature of ductal carcinoma in situ and our gen |
| P2 | Cancer_Diagnosis.Type_of_Cancer | The type of cancer description is slightly imprecise. It should include 'HER2 status pending' instead of 'not tested'. | >90% estrogen receptor positive, Progesterone receptor pending. |
| P2 | Clinical_Findings.findings | The findings section is slightly imprecise. It should include the absence of microcalcifications in the lumpectomy specimen. | - Microcalcifications: None. |
| P2 | Current_Medications.current_meds | The current medications section is incomplete. It should include tamoxifen, even though the patient is instructed to hold off on taking it. | Prescription for tamoxifen sent to pharmacy but instructed to hold off on taking until after radiati |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, cancer type description, clinical findings, and current medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'ductal carcinoma in situ' is not explained in simple terms. | You have a type of breast cancer called ductal carcinoma in situ (DCIS), which m |
| P2 | This sentence is a bit complex and could be simplified. | The edges of the removed tissue show that some cancer cells might still be prese |

*Letter summary*: Letter is mostly clean but contains minor readability issues that could be improved.

---

## ROW 12 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive mammary carcinoma with mixed ductal and lobular f
**Stage**: Clinical stage II

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | current_meds | Incorrectly listed 'tc' as current medication when the patient is not currently on any cancer-related medications. | The patient is not currently on any cancer-related medications as she has not yet started treatment. |
| P2 | Lab_Results | Inaccurate summary. The note mentions several lab results, but the extraction states 'No labs in note.' | Several lab results are mentioned in the note, including WBC count, hemoglobin, hematocrit, platelet |

*Extraction summary*: Major error in current_meds and minor issue in Lab_Results. Most fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'mixed ductal and lobular features'. | You were diagnosed with an early stage breast cancer. The cancer is a type that  |
| P2 | Grade 2 may be confusing to a layperson. | It measures 1.2 cm and is grade 2. |
| P2 | ER+, PR+, and HER2 are unexplained jargon. | The cancer is sensitive to estrogen and progesterone (ER+ and PR+) but does not  |
| P2 | Technical terms 'neoadjuvant', 'doxorubicin', 'cyclophosphamide', and 'paclitaxel' are unexplained. | You will start neoadjuvant chemotherapy with a chemotherapy regimen (doxorubicin |

*Letter summary*: Letter contains some unexplained medical jargon that could be simplified further for better readability.

---

## ROW 13 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- invasive ductal carcinoma
**Stage**: Stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary does not mention the patient's concern about taking time off from work during treatment. | She is also concerned about whether she needs to take time off from work during her treatment especi |
| P2 | Clinical_Findings.findings | The physical exam finding of no palpable axillary lymph nodes is not supported by the note. | Physical exam shows no palpable axillary lymph nodes. |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions SYNTHROID, which is related to hypothyroidism, not cancer treatment. | SYNTHROID 112 mcg tablet, Take 112 mcg by mouth Daily. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions discussing potential treatment options and plans. | We discussed the toxicities associated with Taxotere and Cytoxan including increased risk of infecti |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note mentions discussing scalp cooling with Dignicap or Penguin Cold Cap. | I also explained that we are proponents of scalp cooling including the Dignicap and Penguin Cold Cap |
| P2 | radiotherapy_plan.radiotherapy_plan | The field states 'None', but the note implies radiotherapy will be part of the treatment plan after surgery. | Patient will start hormonal therapy after radiation. |
| P2 | Procedure_Plan.procedure_plan | The field only mentions surgery, but the note also discusses the possibility of neoadjuvant chemotherapy and the need for a partial mastectomy and ALND. | She is scheduled to undergo a partial mastectomy and ALND with Dr ***** on 11/05/17. |
| P2 | Imaging_Plan.imaging_plan | The field only mentions Brain MRI, but the note also discusses ongoing imaging studies. | MRI brain shows a 5 mm right parafalcine dural-based mass most likely a meningioma. |
| P2 | Genetic_Testing_Plan.genetic_testing_plan | The field mentions Mammaprint, but the note also discusses the possibility of other genetic testing. | We discussed the toxicities associated with Taxotere and Cytoxan including increased risk of infecti |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and classification.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'protein' might be considered jargon for an 8th-grade reading level. | You have a type of breast cancer called invasive ductal carcinoma, which is horm |
| P2 | The term 'biopsy' might be considered jargon for an 8th-grade reading level. | A biopsy of the lymph node confirmed that the cancer had spread there. |
| P2 | The term 'hormonal therapy' might be considered jargon for an 8th-grade reading level. | You will start hormonal therapy after radiation. |
| P2 | The terms 'scalp cooling', 'Dignicap', and 'Penguin Cold Cap' might be considered jargon for an 8th-grade reading level. | To help preserve your hair during chemotherapy, you will use scalp cooling with  |
| P2 | The term 'Mammaprint' might be considered jargon for an 8th-grade reading level. | After surgery, you will have a test called Mammaprint to help decide the best tr |

*Letter summary*: The letter contains minor readability issues related to medical jargon that could be simplified further for an 8th-grade reading level.

---

## ROW 14 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2 equivocal grade 1 invasive ductal carcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis | Missing stage information. | The note mentions 'Grade 1 IDC, ER+(>95%), PR(~90%), HER2 equivocal (IHC 2)', but does not specify a |
| P2 | Current_Medications | Missing current hormonal therapy (Lamictal). | The note mentions 'controlled on lamictal' under past medical history. |
| P2 | Treatment_Changes | Inconsistent mention of tamoxifen. | The note discusses tamoxifen but does not plan to start it. The extracted data incorrectly includes  |
| P2 | Therapy_plan | Incorrectly includes continuing tamoxifen. | The note does not mention continuing tamoxifen, only discussing it as a potential option. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing stage information, inconsistent mention of tamoxifen, and incorrect inclusion of continuing tamoxifen.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'invasive ductal carcinoma' is not explained further, which might be confusing. | You have a type of breast cancer called invasive ductal carcinoma (cancer that s |
| P2 | The terms 'estrogen receptor', 'progesterone receptor', and 'HER2 status' are not explained, which may be confusing. | It is strongly positive for estrogen receptor (ER) and progesterone receptor (PR |
| P2 | The terms 'Effexor', 'Gabapentin', and 'Lamictal' are not explained, which may be confusing. | Your psychiatrist will also review the interactions between Effexor, Gabapentin, |

*Letter summary*: The letter contains minor readability issues related to unexplained medical terms.

---

## ROW 15 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2 equivocal metastatic adenocarcinoma, consistent with breast primary
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note mentions 'de novo MBC' but does not explicitly state 'Stage IV'. The extraction infers 'Stage IV', which is reasonable but not directly stated. | Presumptively, she has a primary breast cancer that has metastasized to axillary, supraclavicular, a |
| P2 | Current_Medications.current_meds | The note does not mention any current cancer-related medications, so the field is correctly empty. However, the note mentions 'PNV NO.122/IRON/FOLIC ACID', which might be relevant for a cancer patient, though not strictly cancer-related. | PNV NO.122/IRON/FOLIC ACID (PRENATAL MULTI ORAL) Take by mouth. |
| P2 | Treatment_Changes.recent_changes | The note does not mention any recent treatment changes, so the field is correctly empty. However, the note implies future treatment plans without mentioning any recent changes. | She has no other evidence of distant metastasis. |
| P2 | Treatment_Goals.goals_of_treatment | The note does not explicitly state the treatment goal as 'palliative', but it is implied given the metastatic nature of the cancer. The extraction is reasonable but not directly stated. | technically, we do not consider MBC to be curable. |
| P2 | Imaging_Plan.imaging_plan | The note suggests 'cervical LN FNA if possible', but the extraction incorrectly classifies this under 'Imaging_Plan' instead of 'Procedure_Plan'. | cervical LN FNA if possible |

*Extraction summary*: Most fields are clean, but there are minor issues related to inferred staging, lack of explicit recent treatment changes, and misclassification of FNA under imaging.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'gland cells'. | The cancer started in gland cells and is positive for estrogen and progesterone  |
| P2 | Unexplained jargon 'fine needle aspiration'. | You will have a biopsy of the breast mass and a fine needle aspiration of the ce |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 16 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2-3 invasive lobular carcinoma
**Stage**: Clinical stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary should include the patient's initial reluctance towards neoadjuvant treatment and preference for upfront mastectomy. | She clearly has surgically accessible disease if she approaches this with a mastectomy. Subsequently |
| P2 | Current_Medications.current_meds | The field should be marked as 'None' instead of being empty. | There are no cancer-related medications mentioned in the note. |
| P2 | Treatment_Changes.recent_changes | The field should be marked as 'None' instead of being empty. | There are no recent treatment changes mentioned in the note. |
| P2 | Treatment_Changes.supportive_meds | The field should be marked as 'None' instead of being empty. | There are no supportive medications mentioned in the note. |
| P2 | Imaging_Plan.imaging_plan | The imaging plan should specify 'PET/CT scan' and 'MRI' instead of just 'CT scan'. | An MRI of her bilateral breasts was performed... PET/CT scan was performed... |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, medication fields, and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The description 'cancer that started in the milk-producing glands' is not accurate for invasive lobular carcinoma. | You were diagnosed with a type of cancer called invasive lobular carcinoma (canc |
| P2 | The term 'enlarged lymph nodes' is not precise enough given the detailed information in the note. | Imaging tests showed a large abnormality in your left breast and some enlarged l |
| P2 | The sentence lacks the context that this is the preferred treatment option after discussion. | You are scheduled to have a mastectomy (surgery to remove the breast). |

*Letter summary*: The letter contains inaccuracies and minor readability issues that need addressing.

---

## ROW 17 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma
**Stage**: Stage IIb (T2N1M0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | current_meds | The note does not mention any current cancer-related medications, but the extracted data incorrectly states 'ac'. | There is no mention of current cancer-related medications in the note. |
| P1 | genetic_testing_plan | The note mentions uncertainty about the type of genetic testing done, but the extracted data incorrectly states 'brca'. | She does not know if she has had expanded panel testing or BRCA testing only. |

*Extraction summary*: Two major errors identified in 'current_meds' and 'genetic_testing_plan'. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'residual cancer' is somewhat technical and could be simplified further for clarity. | Postoperative MRI shows some areas that might have residual cancer. |
| P2 | The phrase 'remaining areas of concern' is vague and could be more specific. | You will have radiation therapy to manage the extent of your cancer and any rema |

*Letter summary*: Letter is mostly clean but contains minor readability issues that could be improved for better clarity.

---

## ROW 18 — ⚠️ ISSUES

**Type**: ER+/PR-/HER2- grade 2 invasive ductal carcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary of the visit reason. It mentions 'newly diagnosed' which is not explicitly stated in the note. | 32 y.o. female here for a discussion of treatment options for recently diagnosed left breast cancer. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it can be reasonably inferred from the note. | Known biopsy proven malignancy (BI-RADS 6). |
| P2 | Clinical_Findings.findings | Missing mention of the previous FNA being benign. | FNA of a left axillary lymph node was attempted in ***** and the path was benign. |
| P2 | Treatment_Changes.recent_changes | Missing mention of the recent FNA of the left axillary LNs being benign. | the patient underwent a second FNA of the left axillary LNs and it again showed no cancer. |
| P2 | Treatment_Goals.goals_of_treatment | Goals of treatment should include both curative and adjuvant aspects. | The patient will receive at least five years of adjuvant hormonal therapy. |
| P2 | Radiotherapy_plan.radiotherapy_plan | Incorrectly states 'None'. The note indicates radiation oncology did not recommend radiation due to the ATM mutation. | were not enthusiastic about offering this patient radiation based on her ATM mutation, and therefore |
| P2 | Genetic_Testing_Plan.genetic_testing_plan | Incorrectly states 'mammaprint'. Mammaprint result is already known and not planned. | Subsequent to the visit, the Mammaprint from her prior core biopsy came back as High Risk (-0.622) |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, staging, clinical findings, treatment goals, radiotherapy plan, and genetic testing plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The letter does not specify the exact type of breast cancer, which is important for a new patient. | regarding your newly diagnosed left breast cancer. |
| P2 | The term 'Mammaprint' may be confusing to a layperson. | The cancer measures 8mm and is considered high risk based on a test called Mamma |
| P2 | The term 'adjuvant hormonal therapy' might be confusing. | You will also receive at least five years of adjuvant hormonal therapy because y |

*Letter summary*: The letter contains minor issues related to clarity and completeness for a new patient.

---

## ROW 19 — ⚠️ ISSUES

**Type**: ER+/PR-/HER2- grade 2 and grade 3 invasive ductal carcinoma (IDC) with DCIS comp
**Stage**: Clinical stage 2-3

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The extracted type of cancer is slightly imprecise. It should include the specific location (left breast) and mention the presence of LCIS. | LEFT 3:30, 5cm FN (1.8cm mass): IDC. Grade 2. No LVI. l0 mm on core. ER 61-70%. PR negative. Ki-6715 |
| P2 | Treatment_Changes.supportive_meds | The extracted supportive medication includes ondansetron, which is not specifically related to cancer treatment. It should only include cancer-related supportive care. | ondansetron (ZOFRAN-ODT) 4 mg rapid dissolve tablet DISSOLVE 1 TAB ON THE TONGUE EVERY 6 HOURS AS NE |
| P2 | Medication_Plan.medication_plan | The extracted medication plan includes general medications (ondansetron, zofran, doxycycline, acetaminophen, hydrocodone) which should be excluded. | Follow up with local psych for ongoing support as needed.; also: ondansetron, zofran, doxycycline, a |
| P2 | Therapy_plan.therapy_plan | The extracted therapy plan includes general medications (ondansetron, zofran, doxycycline, acetaminophen, hydrocodone) which should be excluded. | Follow up with local psych for ongoing support as needed.; also: ondansetron, zofran, doxycycline, a |

*Extraction summary*: Most fields are clean, but there are minor issues with the specificity of the cancer diagnosis and inclusion of non-cancer-related medications in the medication and therapy plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon (NED). | You are NED (no evidence of disease) on physical exam. |
| P2 | Unexplained medical jargon (anxiety and emotional lability). | You switched from letrozole to exemestane in December 2018 due to anxiety and em |
| P2 | Unexplained medical jargon (estradiol level, ovarian suppression). | You will check your estradiol level to ensure ovarian suppression. If suppressed |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 20 — ⚠️ ISSUES

**Type**: Bilateral breast cancer, right breast: ER+/PR+/HER2+ with some lobular different
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate description of HER2 status in left breast. | Left breast: ER+, PR+, HER2 0 |
| P2 | Cancer_Diagnosis.Type_of_Cancer | Inaccurate description of HER2 status in left breast. | Left breast: ER+, PR+, HER2 0 |
| P2 | Clinical_Findings.findings | Inaccurate description of HER2 status in left breast. | Left breast: ER+, PR+, HER2 0 |
| P2 | Lab_Plan.lab_plan | Incomplete description of preparatory studies for chemotherapy. | Once the patient has completed surgery we will initiate all preparatory studies for chemotherapy whi |

*Extraction summary*: Most fields are clean, but there are minor inaccuracies regarding HER2 status in the left breast and incomplete description of preparatory studies for chemotherapy.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon: 'lobular differentiation'. | The right breast tumor measures 7.3 cm and is ER+, PR+, and HER2 positive with s |
| P2 | Unexplained jargon: 'protein called HER2'. | The left breast tumor measures 6 cm and is ER+ and PR+ but does not have a prote |
| P2 | Unexplained jargon: 'anthracycline', 'taxane', 'docetaxel', 'carboplatin'. | Options include either four cycles of an anthracycline followed by four cycles o |
| P2 | Unexplained jargon: 'Pertuzumab', 'Trastuzumab'. | You will also receive Pertuzumab and Trastuzumab for a total of one year. |
| P2 | Unexplained jargon: 'port-a-cath'. | After surgery, preparatory studies for chemotherapy will be initiated, which may |

*Letter summary*: The letter contains minor readability issues due to unexplained medical jargon.

---

