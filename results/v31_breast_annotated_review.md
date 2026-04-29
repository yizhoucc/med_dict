# Auto Review: results.txt

Generated: 2026-04-29 14:37
Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)

## Summary

- **Samples**: 20
- **Clean**: 0/20
- **P0** (hallucination): 0
- **P1** (major error): 4
- **P2** (minor issue): 143

### Critical Issues

- **ROW 1** [P1]: Inaccurate staging information. The patient has a Stage IIB (pT2N1a) triple negative breast cancer, not an early-stage (Stage I-II).
- **ROW 12** [P1]: Incorrectly listed 'tc' as current medication when the patient is not currently on any cancer-related medications.
- **ROW 16** [P1]: The phrase 'cancer that started in the milk-producing glands' is inaccurate. Lobular carcinoma starts in the lobules, not the milk-producing glands.
- **ROW 17** [P1]: The note does not mention any current cancer-related medications, but the extracted data incorrectly lists 'ac'.

---

## ROW 1 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (HER2 IHC 1; FISH ratio 2.1, but with HER2 sig/nuc  3.0 and Cen17 
**Stage**: Stage IIB (pT2N1a (inferred from 3.6 cm tumor and 1 positive sentinel lymph node

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is inferred from the tumor size and lymph node status, but the note explicitly states 'Stage II (T2N1)', so using 'Stage IIB (pT2N1a)' may be slightly imprecise. | 81 y.o. postmenopausal patient with multiple medical comorbidities, including heart failure with LVE |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions no cancer-related medications, so this is technically correct but could be clearer. | No specific medications or future medication plans were detailed in the note. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions no recent treatment changes, so this is technically correct but could be clearer. | No specific medications or future medication plans were detailed in the note. |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note mentions no supportive medications related to cancer treatment, so this is technically correct but could be clearer. | No specific medications or future medication plans were detailed in the note. |

*Extraction summary*: Most fields are clean, but there are minor issues with the precision of the cancer stage and clarity around the absence of cancer-related medications and treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate staging information. The patient has a Stage IIB (pT2N1a) triple negative breast cancer, not an early-stage (Stage I-II). | You have been diagnosed with an early-stage (Stage I-II) triple negative breast  |
| P2 | Unnecessary explanation of 'ductal carcinoma'. | The cancer is a type that started in the milk ducts (ductal carcinoma) and is co |
| P2 | Minor readability issue. It could be clearer. | No cancer was found in the skin or nipple. |

*Letter summary*: The letter contains inaccurate staging information and minor readability issues.

---

## ROW 2 — ⚠️ ISSUES

**Type**: ER+/PR- invasive ductal carcinoma, HER2: not tested
**Stage**: Originally Stage IIA, now metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary does not mention the patient's initial treatment history and the discussion about potential treatment options. | The patient is a 73-year-old woman with locally recurrent, unresectable, hormone-receptor positive b |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is incorrectly stated as 'Originally Stage IIA, now metastatic (Stage IV)', whereas the note suggests the cancer is locally recurrent and unresectable, not necessarily metastatic. | This patient is a 73-year-old woman with locally recurrent, unresectable, hormone-receptor positive  |
| P2 | Cancer_Diagnosis.Distant_Metastasis | The distant metastasis is incorrectly stated as 'Yes, to the right parasternal chest wall and liver', whereas the note indicates that the liver lesion is a cyst and there is no evidence of metastasis in the brain or bone. | Her abdomen shows a small, well-demarcated, low-attenuation lesion in the liver consistent with cyst |
| P2 | Treatment_Changes.recent_changes | The recent changes state 'Started on zoledronic acid because of her osteoporosis and need to start her on an aromatase inhibitor.' The note mentions starting zoledronic acid but does not specify the exact reason. | On my visit with her on January 09, I started her on zoledronic acid because of her osteoporosis and |
| P2 | Treatment_Goals.goals_of_treatment | The goal is stated as 'palliative', whereas the note suggests a more complex treatment plan involving hormone therapy and possibly surgery and radiation. | Based on all of this information, it appears that the patient has a local regional recurrence of a s |
| P2 | radiotherapy_plan.radiotherapy_plan | The radiotherapy plan is speculative and not clearly stated in the note. | Given the limited radiation field that she initially had, I think it would be possible to shrink the |

*Extraction summary*: Several fields contain minor inaccuracies or lack specific details from the note. Overall, the majority of the fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have a type of breast cancer called ER+/PR- invasive ductal carcinoma. |
| P2 | Lacks specificity about the imaging modality used. | Recent imaging showed that the cancer in your chest wall has grown, while the ly |
| P2 | Lacks clarity on the name of the research study. | You will start a medication called an aromatase inhibitor either alone or as par |

*Letter summary*: Letter contains minor readability and clarity issues.

---

## ROW 3 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (HER2 FISH neg, ratio 1.2) spindle cell metaplastic carcinoma
**Stage**: Locally advanced, multifocal (possibly awaiting biopsy confirmation of extent of

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate description of the patient's stage. The note mentions 'early stage' but the patient's cancer is described as 'locally advanced, multifocal'. | Patient with early stage breast cancer here to discuss neoadjuvant therapy |
| P2 | Current_Medications.current_meds | Missing cancer-related medications. The note does not mention any specific cancer-related medications, but the field should be explicitly stated as empty. | No specific cancer-related medications mentioned. |
| P2 | Treatment_Changes.recent_changes | Field is empty, but the note does not indicate any recent changes in treatment. | No recent changes in treatment mentioned. |
| P2 | Treatment_Changes.supportive_meds | Field is empty, but the note does not indicate any supportive medications related to cancer treatment. | No supportive medications related to cancer treatment mentioned. |
| P2 | Therapy_plan.therapy_plan | Redundant information. The therapy plan is repeated in both 'Medication_Plan' and 'Therapy_plan'. | The patient will start neoadjuvant chemotherapy, possibly as part of the I-SPY2 clinical trial. |
| P2 | Procedure_Plan.procedure_plan | Incomplete. The note mentions the patient will benefit from a Medi-port, but the procedure plan should also include the placement of the Medi-port. | The patient will benefit from a Medi-port |
| P2 | Imaging_Plan.imaging_plan | Incomplete. The note mentions that the patient has already done the MRI, research biopsy, and Echo, but the imaging plan only mentions Mammaprint and Echocardiogram. | She has done the MRI, research biopsy, and Echo, mammaprint is pending and the echo looks good. |
| P2 | Advance_care_planning.Advance care | Inaccurate. The note does not mention advance care planning being discussed, but it also does not state that it was not discussed. | No discussion of advance care planning mentioned. |

*Extraction summary*: Most fields are clean, but there are minor issues with the description of the patient's stage, missing cancer-related medications, redundant therapy plan information, incomplete procedure and imaging plans, and inaccurate advance care planning.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (spindle cell metaplastic carcinoma). | You have a rare type of breast cancer called spindle cell metaplastic carcinoma. |
| P2 | Unexplained jargon (ER, PR, HER2). | This cancer does not have proteins called ER, PR, or HER2, which means it doesn' |
| P2 | Unexplained jargon (Medi-port). | You will have a Medi-port placed to make chemotherapy easier. |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 4 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- (TNBC) grade 3 invasive ductal carcinoma
**Stage**: Stage I (inferred from tumor ≤2cm)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is inferred as 'Stage I' based on tumor size, but the note does not specify the stage. | Latest PETCT demonstrates growth of R breast lesion from 1.2 x 0.7 to 2.7 x 1.7 cm |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions no current cancer-related medications. | No current facility-administered medications for this visit. |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note mentions no current cancer-treatment-related supportive care. | No current facility-administered medications for this visit. |

*Extraction summary*: Most fields are clean, but there are minor issues with the inferred stage and the empty supportive medication field.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The explanation of 'triple-negative' is overly detailed and may be confusing at an 8th-grade reading level. | You have a type of breast cancer called triple-negative (TNBC) grade 3 invasive  |
| P2 | This statement is not entirely clear and could be misinterpreted. | No cancer has spread to other parts of your body. |

*Letter summary*: Letter contains minor readability issues that could be improved for clarity.

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
| P2 | Procedure_Plan.procedure_plan | The field incorrectly includes 'AI therapy for at least 5 years', which should be under 'Medication_Plan'. | AI therapy for at least 5 years |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, treatment changes, and procedure plan classification.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'protein called HER2' may be confusing to an 8th-grade reader. | Both cancers are sensitive to hormones and do not have a protein called HER2. |
| P2 | The term 'Oncotype DX RS' may be too technical for an 8th-grade reader. | Your tumor will be tested with Oncotype DX RS to help guide further treatment de |

*Letter summary*: Letter contains minor readability issues that could be improved for better clarity.

---

## ROW 6 — ⚠️ ISSUES

**Type**: ER-neg, PR neg, HER2 3+, FISH ratio 13, Ki67 10-15% invasive ductal carcinoma
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but there are no cancer-related medications mentioned in the note. | No cancer-related medications are listed in the note. |
| P2 | Treatment_Goals | The treatment goal is listed as 'palliative', but the note suggests a more optimistic outlook ('anticipate excellent response and possible long term disease control'). | Given HR-negative HER2+ subtype, I anticipate excellent response and possible long term disease cont |

*Extraction summary*: Most fields are clean, but there are minor issues with the treatment goals and current medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'invasive ductal carcinoma' is not explained in simple terms. | You have a type of breast cancer called invasive ductal carcinoma (cancer that s |
| P2 | The term 'HR-negative' is not explained in simple terms. | It is HR-negative and has a protein called HER2. |
| P2 | The term 'stage IV' is not explained in simple terms. | The cancer has spread to your bones, which means it is now at stage IV. |
| P2 | The term 'ondansetron' is not explained in simple terms. | You will also take ondansetron to help with nausea. |

*Letter summary*: The letter contains minor readability issues due to some unexplained medical terms.

---

## ROW 7 — ⚠️ ISSUES

**Type**: ER-/PR-/HER2- triple negative invasive ductal carcinoma
**Stage**: Originally Stage IIB (cT2 cN1 cM0) -> ypT1c(m) ypN1a (1/22 LN positive), now met

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary does not mention the patient's Lynch syndrome and other cancers. | ***** ***** is a 44 y.o. female with Lynch syndrome with early stage colon cancer and endometrial ca |
| P2 | Cancer_Diagnosis.Type_of_Cancer | The type of cancer is described as 'ER-/PR-/HER2- triple negative invasive ductal carcinoma', but the note mentions 'Metastatic ER negative, HER2 negative breast cancer'. | Impression: Metastatic ER negative, HER2 negative breast cancer on nab paclitaxel and pembrolizumab  |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage of cancer is described as 'Originally Stage IIB (cT2 cN1 cM0) -> ypT1c(m) ypN1a (1/22 LN positive), now metastatic (Stage IV)', but the note does not provide this specific detail. | Early stage breast cancer 2014  clinical Stage IIB (cT2 cN1 cM0) -> ypT1c(m) ypN1a (1/22 LN positive |
| P2 | Current_Medications.current_meds | Only 'pembrolizumab' is listed, but the note mentions 'nab paclitaxel and pembrolizumab'. | She started pembrolizumab and abraxane on 03/11/19 and presents today for cycle 1 day 8. |
| P2 | Treatment_Changes.recent_changes | The note mentions restarting xarelto, but it is not a cancer-related medication. | Restart xarelto, stop for port placement per IR |
| P2 | Response_Assessment.response_assessment | The response assessment repeats information already mentioned under 'Clinical Findings'. | Imaging studies reviewed today show clinical or radiological evidence of progression of metastatic E |
| P2 | Imaging_Plan.imaging_plan | The imaging plan is not clearly specified in the note. | Imaging guided by symptoms or every 3-4 months, longer intervals if stable. |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness and classification of cancer-related information.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'lesion'. | The imaging studies show that your cancer has progressed. Specifically, a lesion |
| P2 | Unexplained jargon 'port'. | You will have a port placed soon to make it easier to receive treatments. |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 8 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma with extensive DCIS
**Stage**: Stage IIA (pT2(m)N1a)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note does not explicitly state that there are no current cancer-related medications. | The note does not mention any current cancer-related medications, but it also does not explicitly st |

*Extraction summary*: Most fields are clean, but the absence of current cancer-related medications is marked as a minor issue.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon | The cancer was a type called invasive ductal carcinoma, which started in the mil |
| P2 | Unexplained jargon | You agreed to start chemotherapy with a combination of drugs called AC plus pacl |
| P2 | Unexplained jargon | You will need a heart ultrasound called a TTE before starting chemotherapy. |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 9 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 IDC (micropapillary features) with metastatic recurrence
**Stage**: Originally Stage III (T3N2), now Stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note mentions the original stage as Stage III (T3N2) but does not specify the current stage as 'now Stage III'. It may be inferred as such, but it is slightly imprecise. | 31 y.o. premenopausal female patient with a history of a Stage III HR+/HER2- grade 2 IDC with microp |
| P2 | Current_Medications.current_meds | The note indicates the patient is currently prescribed gabapentin and hydrochlorothiazide, but these are general medications and thus excluded. However, the note does not mention any current cancer-related medications, so the field is correct but could be misleading if the patient is expected to be on some form of treatment. | Outpatient Encounter Medications as of 01/28/2019: gabapentin (NEURONTIN) 300 mg capsule, hydroCHLOR |
| P2 | Treatment_Changes.recent_changes | The note states the plan is to start goserelin followed by an AI, but the field only mentions goserelin and AI without specifying the sequence. | We will plan to start her on OS as soon as possible and then start an AI. |
| P2 | Genetic_Testing_Plan.genetic_testing_plan | The field incorrectly lists 'plan to FNA the mass on left lateral anterior neck' under genetic testing plan, which is not related to genetic testing. | plan to FNA the mass on left lateral anterior neck |

*Extraction summary*: Most fields are clean, but there are minor issues with the staging, current medications, recent treatment changes, and genetic testing plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have a type of breast cancer called ER+/PR+/HER2- grade 2 IDC with micropapi |
| P2 | Slightly imprecise wording. | The cancer has recurred and may have spread to other parts of your body. |
| P2 | Slightly imprecise wording. | During your physical exam, a mass was found in your left breast and another in y |
| P2 | Slightly imprecise wording. | There is also some discoloration of the skin on your left breast. |
| P2 | Slightly imprecise wording. | You are planning to start a medication called goserelin, which stops the ovaries |
| P2 | Slightly imprecise wording. | This will be followed by an aromatase inhibitor (AI), which blocks the productio |
| P2 | Slightly imprecise wording. | Your next visit to the clinic will be two weeks after you start goserelin. |

*Letter summary*: The letter contains minor readability issues and slightly imprecise wording that could be clarified for better understanding.

---

## ROW 10 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma
**Stage**: Stage II (inferred from T2N1)

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
| P2 | Terms like 'neuropathy', 'cardiac toxicity' are not explained. | Antiemetics and other supportive medications will be used to manage side effects |

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
**Stage**: Stage II (inferred from pT2 N0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | current_meds | Incorrectly listed 'tc' as current medication when the patient is not currently on any cancer-related medications. | The patient is not currently on any cancer-related medications as she has not yet started treatment. |
| P2 | Lab_Results | Inaccurate summary of lab results. The note mentions several lab results, but the summary incorrectly states 'No labs in note.' | Available labs, pathology, and imaging were reviewed and independently interpreted, as described abo |

*Extraction summary*: Major error in current medications and minor issue with lab results summary. Most fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The stage is inaccurately described as 'Stage I-II'. The correct stage is 'Stage II'. | You were diagnosed with an early stage (Stage I-II) breast cancer that started i |
| P2 | The sentence lacks clarity about the specific regimen name (AC/T). | You will start neoadjuvant chemotherapy with a chemotherapy regimen (doxorubicin |

*Letter summary*: There are minor inaccuracies and clarity issues in the letter.

---

## ROW 13 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- invasive ductal carcinoma
**Stage**: Stage IIIB (inferred from 2.2 cm tumor with positive axillary lymph nodes and su

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary does not mention the patient's concern about taking time off from work during treatment. | She is also concerned about whether she needs to take time off from work during her treatment especi |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is inferred as IIIB, but the note does not explicitly state the stage. It might be more accurate to leave it unspecified. | Stage IIIB (inferred from 2.2 cm tumor with positive axillary lymph nodes and suspicious additional  |
| P2 | Clinical_Findings.findings | The physical exam finding of no palpable axillary lymph nodes is not supported by the note. | Physical exam shows no palpable axillary lymph nodes. |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions that the patient is scheduled to start hormonal therapy after radiation. | Patient will start hormonal therapy after radiation. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note discusses potential changes in treatment based on Mammaprint results. | We discussed the toxicities associated with Taxotere and Cytoxan including increased risk of infecti |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note mentions the use of scalp cooling with Dignicap or Penguin Cold Cap. | Scalp cooling with Dignicap or Penguin Cold Cap will be used during chemotherapy. |
| P2 | Treatment_Goals.goals_of_treatment | The goal is listed as 'curative', but the note suggests a more complex approach involving adjuvant chemotherapy and hormonal therapy. | We reviewed her imaging and pathology reports to date. I explained that because of her young age and |
| P2 | Therapy_plan.therapy_plan | The field does not mention the potential use of CDK 4/6 inhibitors in clinical trials. | We briefly discussed participation in clinical trial trials of CDK 4/6 inhibitors such as the PALLAS |
| P2 | Procedure_Plan.procedure_plan | The field does not specify the type of surgery planned. | She is scheduled to undergo a partial mastectomy and ALND with Dr ***** on 11/05/17. |
| P2 | Imaging_Plan.imaging_plan | The field only mentions Brain MRI, but the note also discusses ongoing imaging plans. | MRI brain shows a 5 mm right parafalcine dural-based mass most likely a meningioma. |

*Extraction summary*: Most fields are clean, but there are several minor issues related to completeness and precision of the extracted information.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'lymph nodes'. | The cancer is in the upper central part of your right breast and has spread to s |
| P2 | Unexplained medical jargon 'test of the lymph nodes'. | A test of the lymph nodes confirmed that the cancer had spread there. |
| P2 | Unexplained medical jargon 'hormonal therapy'. | You will start hormonal therapy after your surgery. |
| P2 | Unexplained medical jargon 'scalp cooling cap'. | To help preserve your hair during chemotherapy, you will use a scalp cooling cap |
| P2 | Unexplained medical jargon 'clinical trials'. | We also talked about the possibility of joining clinical trials for new treatmen |
| P2 | Unexplained medical jargon 'Mammaprint'. | After surgery, you will have a test called Mammaprint to learn more about your c |

*Letter summary*: The letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 14 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2 equivocal grade 1 invasive ductal carcinoma
**Stage**: Stage IA (inferred from pT1 N0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'lamictal' which is relevant to her condition. | Past Medical History: Bipolar II disorder (HCC) diagnosed mid 20's, controlled on lamictal |
| P2 | Treatment_Changes | The field mentions 'tamoxifen' but does not specify it as a discussion point only. | We discussed the benefits relatively of shorter duration endocrine therapy, and whether two years wo |
| P2 | Therapy_plan | The field incorrectly includes 'Continue tamoxifen', which is not part of the plan. | She was curious about the benefits relatively of shorter duration endocrine therapy, and whether two |

*Extraction summary*: Most fields are clean, but there are minor issues with current medications, treatment changes, and therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'protein' may be considered jargon for an 8th-grade reading level. | You have early-stage breast cancer (Grade 1 invasive ductal carcinoma) that is E |
| P2 | The term 'stye' may be unfamiliar to some readers. | Additionally, there is a right eyelid stye noted during the physical examination |
| P2 | The terms 'goserelin' and 'letrozole' are medical jargon. | You will start goserelin today and begin letrozole in about two weeks. |

*Letter summary*: Letter contains minor readability issues related to medical jargon.

---

## ROW 15 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2 equivocal invasive ductal carcinoma with metastatic recurrence (rig
**Stage**: Stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note does not specify 'invasive ductal carcinoma', only mentions metastatic adenocarcinoma consistent with breast primary. | US-guided right supraclavicular LN biopsy: metastatic adenocarcinoma, c/w breast primary (CK7, GATA3 |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not explicitly state the stage as 'Stage III'. The stage is inferred based on the metastatic nature of the cancer. | She has no other evidence of distant metastasis. |
| P2 | Cancer_Diagnosis.Distant Metastasis | The note specifies that there is no other evidence of distant metastasis besides lymphadenopathy, so 'Not sure' is imprecise. | She has no other evidence of distant metastasis. |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions several medications, though none are cancer-related. | Outpatient Encounter Prescriptions as of 04/13/2017 |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions ongoing discussions about future treatment plans. | I explained to ***** that if we confirm the above with breast and cervical LN biopsies, then she wou |
| P2 | Treatment_Changes.supportive_meds | Only 'docusate sodium (COLACE)' is listed, but the note mentions several other medications, though none are cancer-related. | Outpatient Encounter Prescriptions as of  04/13/2017 |
| P2 | Imaging_Plan.imaging_plan | The field incorrectly lists 'cervical LN FNA if possible' instead of 'US-guided biopsy of breast mass, cervical LN FNA, core biopsy'. | Recommendations in Brief: - US-guided biopsy of breast mass (possibly calcified nodule in axillary t |

*Extraction summary*: Most fields are clean, but there are minor issues with the cancer diagnosis details, treatment changes, and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'invasive ductal carcinoma' is not explained, which might be confusing for an 8th-grade reading level. | You have a type of breast cancer called invasive ductal carcinoma that has sprea |
| P2 | The terms 'estrogen and progesterone receptors' and 'protein called HER2' are not explained, which might be confusing for an 8th-grade reading level. | The cancer tests positive for estrogen and progesterone receptors, and the resul |
| P2 | The term 'fine needle aspiration' is not explained, which might be confusing for an 8th-grade reading level. | You will have a biopsy of the breast mass and a fine needle aspiration of the ce |

*Letter summary*: Letter contains some terms that may be confusing for an 8th-grade reading level, but overall it is mostly clean.

---

## ROW 16 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2-3 invasive lobular carcinoma
**Stage**: Clinical stage III

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary should include the patient's initial reluctance towards clinical trials. | We discussed this at some length and she was extremely hesitant to consider enrolling in a clinical  |
| P2 | Current_Medications.current_meds | The field should be marked as 'None' instead of being empty. | None |
| P2 | Treatment_Changes.recent_changes | The field should be marked as 'None' instead of being empty. | None |
| P2 | Treatment_Changes.supportive_meds | The field should be marked as 'None' instead of being empty. | None |
| P2 | Imaging_Plan.imaging_plan | The imaging plan should specify the type of CT scan and ultrasound. | Ultrasound revealed a 1.3 x 0.7 x 1.2 cm left axillary lymph node with abnormal morphology, thickene |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, medication fields, and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The phrase 'cancer that started in the milk-producing glands' is inaccurate. Lobular carcinoma starts in the lobules, not the milk-producing glands. | You were diagnosed with a clinical stage III lobular carcinoma (cancer that star |
| P2 | This sentence is somewhat vague and doesn't provide enough detail about the imaging findings. | Imaging tests showed a large abnormality in the left breast and some small lymph |
| P2 | The sentence is missing the word 'a' before 'mastectomy'. | You are scheduled to have a mastectomy (surgery to remove the breast) to treat t |
| P2 | The CT scan and ultrasound are not mentioned in the extracted data as part of the follow-up plan. | After surgery, you will have a CT scan and an ultrasound to monitor your conditi |

*Letter summary*: The letter contains inaccuracies and lacks some important details. It needs corrections and clarifications.

---

## ROW 17 — ⚠️ ISSUES

**Type**: ER+/PR+/HER2- grade 2 invasive ductal carcinoma
**Stage**: Stage IIb (T2N1M0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | current_meds | The note does not mention any current cancer-related medications, but the extracted data incorrectly lists 'ac'. | There is no mention of current cancer-related medications in the note. |
| P2 | genetic_testing_plan | The note mentions uncertainty about the type of genetic testing done, but the extracted data simplifies it to 'brca'. | She does not know if she has had expanded panel testing or BRCA testing only. |

*Extraction summary*: Two issues identified: incorrect listing of current cancer-related medications and oversimplification of genetic testing plan. Most fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (Ki67). | The cancer has a reported Ki67 of 30%, which tells us how fast the cancer cells  |
| P2 | Unexplained jargon (peripheral enhancement). | An MRI showed postoperative changes in your left breast with a seroma (a pocket  |
| P2 | Unexplained jargon (port). | You will have a port placed to make it easier to receive chemotherapy. |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 18 — ⚠️ ISSUES

**Type**: ER+/PR-/HER2- grade 2 invasive ductal carcinoma
**Stage**: Stage IIA (inferred from pT2 N0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary of PR status. | The cancer is ER positive (>95%), PR positive (<5%), HER2 negative (by FISH) |
| P2 | Current_Medications.current_meds | Missing current hormonal therapy. | Because her breast cancer is estrogen receptor positive, she understands that she will receive at le |
| P2 | Treatment_Changes.recent_changes | Missing recent changes in treatment plan. | Because of her ATM mutation, radiation oncology has recommended that she proceed with mastectomy rat |
| P2 | Genetic_Testing_Plan.genetic_testing_plan | Incorrect genetic testing plan. | Subsequent to the visit, the Mammaprint from her prior core biopsy came back as High Risk (-0.622) |

*Extraction summary*: Most fields are accurate, but there are minor issues with the summary of PR status, current medications, recent treatment changes, and genetic testing plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The staging information is imprecise. The note specifies Stage IIA, which should be mentioned. | You have been diagnosed with an early-stage (Stage I-II) estrogen receptor posit |
| P2 | The letter does not mention the chemotherapy plan based on nodal status at the time of surgery. | After surgery, you will have a detailed discussion about the choice of hormonal  |

*Letter summary*: The letter contains minor inaccuracies and omissions that need clarification.

---

## ROW 19 — ⚠️ ISSUES

**Type**: ER+/PR-/HER2- invasive ductal carcinoma (IDC) with extensive DCIS
**Stage**: Clinical stage 2-3

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note mentions 'extensive DCIS', but the extracted diagnosis does not specify this. | There is asymmetric non-mass enhancement extending medially from the two index masses, concerning fo |
| P2 | Treatment_Changes.supportive_meds | The extracted supportive medication includes ondansetron, but it is not specifically related to cancer treatment. | ondansetron (ZOFRAN-ODT) 4 mg rapid dissolve tablet DISSOLVE 1 TAB ON THE TONGUE EVERY 6 HOURS AS NE |
| P2 | Imaging_Plan.imaging_plan | The plan mentions DEXA scans but does not include any mention of annual mammograms, which might be relevant. | needs DEXA scan now and every 2 years. |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness of the cancer diagnosis, the classification of supportive medications, and the imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (BSO, AI) | You are considering bilateral surgery to remove the ovaries and fallopian tubes  |
| P2 | Unexplained jargon (bilateral mastectomies) | You will have 6-monthly exams, and no imaging is needed given the bilateral mast |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 20 — ⚠️ ISSUES

**Type**: Bilateral breast cancer, right breast: ER+/PR+/HER2+ with some lobular different
**Stage**: Not mentioned in note

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
| P2 | Unexplained jargon (lobular differentiation) | The right breast tumor measures 7.3 cm and is ER+, PR+, and HER2 positive with s |
| P2 | Unexplained jargon (ER+, PR+, HER2 negative) | The left breast tumor measures 6 cm and is ER+, PR+, and HER2 negative. |
| P2 | Unexplained jargon (Pertuzumab and Trastuzumab) | You will also receive Pertuzumab and Trastuzumab for a total of 1 year. |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

