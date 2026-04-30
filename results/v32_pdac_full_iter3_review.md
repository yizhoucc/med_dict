# Auto Review: results.txt

Generated: 2026-04-29 21:36
Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)

## Summary

- **Samples**: 100
- **Clean**: 0/100
- **P0** (hallucination): 0
- **P1** (major error): 34
- **P2** (minor issue): 538

### Critical Issues

- **ROW 3** [P1]: The note mentions both 'metastatic adenocarcinoma' and 'metastatic pancreatic adenocarcinoma', but the diagnosis is uncertain and the note suggests it might be metastatic adenocarcinoma of the colon. The extracted data incorrectly lists both diagnoses as definitive.
- **ROW 3** [P1]: The note mentions starting the patient on an unspecified agent, but the extracted data does not include this information.
- **ROW 3** [P1]: The note mentions hospitalization for C. difficile infection and completion of vancomycin treatment, but the extracted data does not include this information.
- **ROW 5** [P1]: Incomplete sentence with missing critical information.
- **ROW 7** [P1]: Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III.
- **ROW 9** [P1]: Incomplete sentence with missing critical information.
- **ROW 12** [P1]: Inaccurate statement. The note mentions that the mesenteric lymph nodes have decreased in size, but the imaging report states that the mesenteric nodule has increased in size.
- **ROW 14** [P1]: The sentence does not specify that the tumor is of duodenal origin, which is important information.
- **ROW 15** [P1]: The plan is to resume an unspecified agent, not specifically oxaliplatin.
- **ROW 16** [P1]: Grammar error ('You has' should be 'You have').
- **ROW 21** [P1]: Incomplete sentence, missing critical information about the specific dose reduction.
- **ROW 24** [P1]: Incorrect goal of treatment. Should be 'surveillance' since the patient has completed adjuvant therapy and is on follow-up.
- **ROW 24** [P1]: This sentence implies that the cancer is not responding well to the prior treatment, which is not supported by the clinical note. The note suggests monitoring the findings closely rather than concluding that the cancer is not responding.
- **ROW 28** [P1]: Inaccurate description of the cancer involvement. The original note mentions 'primary pancreatic head adenocarcinoma with metastatic gallbladder/liver involvement; vs a multifocal biliary tract/GB cancer'.
- **ROW 34** [P1]: Incomplete sentence with missing critical information (specific medication name).
- **ROW 36** [P1]: Incomplete sentence with missing critical information (dose number).
- **ROW 38** [P1]: Incomplete sentence with missing critical information.
- **ROW 39** [P1]: Incomplete sentence with missing critical information.
- **ROW 40** [P1]: Incomplete sentence with missing critical information (lacks the specific dose increase).
- **ROW 45** [P1]: The term 'moderately differentiated' is too technical for an 8th-grade reading level.
- **ROW 47** [P1]: Inaccurate description of imaging findings. The original note mentions an interval increase in nodular thickening of the left adrenal gland, but does not state that there has been no significant change in the pancreatic lesion. It states that the pancreatic lesion is stable.
- **ROW 52** [P1]: Incomplete sentence with missing critical information.
- **ROW 53** [P1]: Missing cancer-related medications (gemcitabine and Abraxane).
- **ROW 53** [P1]: Incomplete sentence with missing critical information.
- **ROW 64** [P1]: Incomplete sentence, missing critical information about the stage of cancer.
- **ROW 67** [P1]: Missing critical lab results such as CA 19-9, which is relevant to the patient's condition.
- **ROW 71** [P1]: Contains technical jargon that may be difficult for an 8th-grade reader to understand.
- **ROW 71** [P1]: Contains technical jargon that may be difficult for an 8th-grade reader to understand.
- **ROW 71** [P1]: Incomplete sentence with missing critical information.
- **ROW 71** [P1]: Incomplete sentence with missing critical information.
- **ROW 74** [P1]: Missing cancer-related medication (rucaparib).
- **ROW 75** [P1]: Incomplete sentence with missing critical information (dose amount).
- **ROW 81** [P1]: Incomplete sentence with missing critical information about the medications.
- **ROW 90** [P1]: Incomplete sentence with missing critical information.

---

## ROW 1 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic adenocarcinoma
**Stage**: Potentially resectable with a regional LN involvement

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (palonosetron). | He initiated ***** on 03/06/2022, once his LFTs had sufficiently normalized post-stent, and is now s |
| P2 | Lab_Results | Missing CA 19-9 value from initial diagnosis. | Baseline *****-9 markedly elevated at 77,736 (in the context of obstructive jaundice) |
| P2 | Treatment_Changes | Missing mention of palonosetron as a supportive med. | Start 2 days after last dose of palonosetron (*****) is given. |
| P2 | Treatment_Goals | Inaccurate goal classification. Should be 'neoadjuvant'. | We reviewed his most recent imaging studies which show, quite encouragingly, resolution of his aorto |
| P2 | Radiotherapy_plan | Uncertainty about radiotherapy role should be mentioned. | While the optimal selection and duration of systemic therapy, as well as the uncertain role of radia |

*Extraction summary*: Most fields are accurate, but there are minor issues with missing cancer-related medications, incomplete lab results, and slight misclassification of treatment goals.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 2 — ⚠️ ISSUES

**Type**: pT3N1 poorly differentiated adenocarcinoma of the pancreas
**Stage**: pT3N1

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note does not explicitly state that there are no current cancer-related medications. | She then started therapy with gemcitabine and capecitabine and has completed six full cycles. |
| P2 | Treatment_Changes | Supportive medications listed are not cancer-treatment-related supportive care. | ondansetron, prochlorperazine |
| P2 | Imaging_Plan | The note mentions a CT scan was done during this visit, but the plan does not mention future imaging plans. | She'll continue on surveillance. Her PCP is managing her diabetes. |

*Extraction summary*: Most fields are clean, but there are minor issues with current medications, supportive medications, and imaging plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 3 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma; Metastatic adenocarcinoma of the colon
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Cancer_Diagnosis.Type_of_Cancer | The note mentions both 'metastatic adenocarcinoma' and 'metastatic pancreatic adenocarcinoma', but the diagnosis is uncertain and the note suggests it might be metastatic adenocarcinoma of the colon. The extracted data incorrectly lists both diagnoses as definitive. | Probable metastatic adenocarcinoma of the colon, responding well to therapy |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not specify a TNM stage, only that the cancer is metastatic. The extracted data should reflect this uncertainty. | Probable metastatic adenocarcinoma of the colon, responding well to therapy |
| P2 | Cancer_Diagnosis.Distant_Metastasis | The note mentions metastasis to the liver and peritoneum but does not confirm metastasis to the lungs. The extracted data incorrectly states possible lung metastasis. | Multiple bilateral noncalcified pulmonary nodules, the largest being a 5 mm left lower lobe nodule. |
| P1 | Current_Medications.current_meds | The note mentions starting the patient on an unspecified agent, but the extracted data does not include this information. | We started her on *****. |
| P1 | Treatment_Changes.recent_changes | The note mentions hospitalization for C. difficile infection and completion of vancomycin treatment, but the extracted data does not include this information. | After completing her first cycle, she was hospitalized for diarrhea and was found to have C. Diffici |

*Extraction summary*: Major issues with cancer diagnosis certainty and current medications. Minor issues with staging and distant metastasis details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon (heterogeneously enhancing mass, necrotic lymphadenopathy) | Additionally, the circumferential heterogeneously enhancing mass in the ascendin |

*Letter summary*: Letter is mostly clean but contains some unexplained medical jargon that could be simplified for better readability.

---

## ROW 4 — ⚠️ ISSUES

**Type**: Metastatic pancreatic ductal adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine-based regimen mentioned as a potential future treatment, but not listed under current medications). | We talked about potential side fx and risks, esp in terms of infectious complications, and adverse i |
| P2 | Treatment_Changes | Supportive medications listed include general pain medications (Fentanyl patch, Dilaudid) without specifying they are for cancer pain. | Symptom-wise, he is followed by our Cancer Center Symptom Management Service; his pain appears reaso |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication and imprecise classification of supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'peritoneal carcinomatosis'. | Additionally, the peritoneal carcinomatosis has slightly worse large volume asci |
| P2 | Unexplained jargon 'pancolitis'. | There is no evidence of bowel obstruction, and pancolitis has nearly resolved wi |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 5 — ⚠️ ISSUES

**Type**: Well-differentiated adenocarcinoma, intestinal type, of ampullary origin
**Stage**: (pT2N1), now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication information. | She is now receiving palliative chemotherapy consisting of dose-modified [REDACTED] (at empirically  |
| P2 | Treatment_Changes | Supportive medications listed are not cancer-related. | The patient is currently receiving palliative chemotherapy consisting of dose-modified [REDACTED] (o |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative'. | She is now receiving palliative chemotherapy. |
| P2 | Response_Assessment | Inaccurate description of response assessment. | She has now completed 11 cycles in total, which has achieved essentially stable disease in the liver |
| P2 | Medication_Plan | Missing specific cancer-related medication details. | She is now receiving palliative chemotherapy consisting of dose-modified [REDACTED] (at empirically  |
| P2 | Therapy_plan | Incorrectly states maintenance therapy with capecitabine is not favored. | I would not favor any form of maintenance therapy (e.g. capecitabine) in her circumstances. |
| P2 | Genetic_Testing_Plan | Molecular profiling results are already known and not a future plan. | ***** molecular profiling results from her liver biopsy are notable for pathogenic mutations in **** |
| P2 | follow_up_next_visit | Lacks specific date range for next visit. | We will proceed with ***** chemotherapy break for the next couple of months and plan on repeat imagi |
| P2 | Referral | Missing referral to Cancer Center Symptom Management Service. | I encouraged her to follow up with our Cancer Center Symptom Management Service(last seen in August) |

*Extraction summary*: Several fields contain minor inaccuracies or omissions regarding cancer-related treatments and plans. Overall, the majority of the fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Repeat imaging/bloodwork is planned between a medication/a medication. |

*Letter summary*: Letter contains an incomplete sentence that needs clarification.

---

## ROW 6 — ⚠️ ISSUES

**Type**: Pancreatic neuroendocrine tumor, grade 2
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note mentions both grade 1 and grade 2, but the extracted data only states grade 2. | Pathology(+) 5.5cm well-differentiated neuroendocrine tumor, grade 1 (ki-67% <2%)... Retroperitoneal |
| P2 | Clinical_Findings.findings | The extracted data does not mention the patient's ECOG PS 0, which is mentioned in the note. | Physical exam shows no palpable masses, jaundice, ascites, or significant weight loss. Patient is EC |

*Extraction summary*: Most fields are clean, but there are minor issues with the cancer grade and ECOG PS status.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'lymph nodes' is used without explanation, which may be jargon for an 8th-grade reading level. | Imaging findings show that the cancer in your liver and lymph nodes has grown si |
| P2 | The term 'MRI' is used without explanation, which may be jargon for an 8th-grade reading level. | MRI scans will be used to monitor the tumor's progress. |

*Letter summary*: Letter is mostly clean but contains minor readability issues with unexplained medical terms.

---

## ROW 7 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic ductal adenocarcinoma
**Stage**: pT3 N1

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. The note mentions metformin, methylphenidate, ondansetron, and oxycodone, but only ondansetron and oxycodone are listed under supportive_meds. | Medications the patient states to be taking prior to today's encounter. METFORMIN HCL (METFORMIN ORA |
| P2 | Treatment_Changes | Incomplete. The note mentions that the patient is currently receiving adjuvant chemotherapy with gemcitabine alone after discontinuing [REDACTED]-paclitaxel due to cumulative peripheral sensory neuropathy, but the current_meds field only lists gemcitabine. | Beginning cycle #4 the [REDACTED]-paclitaxel was discontinued due to cumulative peripheral sensory n |
| P2 | Treatment_Goals | Inaccurate. The note suggests the goal is adjuvant therapy, not curative. The patient is receiving adjuvant chemotherapy after surgery. | He started with the combination of gemcitabine plus [REDACTED]-paclitaxel, but due to cumulative neu |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III. | You have a well-differentiated pancreatic ductal adenocarcinoma (cancer that sta |
| P2 | Unexplained jargon 'labs'. | Your labs are noted to be significantly improved, allowing continuation of gemci |
| P2 | Unexplained jargon 'adjuvant chemotherapy', 'cumulative peripheral sensory neuropathy'. | You are currently receiving adjuvant chemotherapy with gemcitabine alone, having |

*Letter summary*: The letter contains inaccuracies and minor readability issues.

---

## ROW 8 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma with metastatic disease to the lungs
**Stage**: (metastatic to lungs)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (Gemcitabine). | Starting with cycle 3 patient will start to receive Gemcitabine on weeks 1, 2 and 3, followed by 1 w |
| P2 | Treatment_Changes | Missing detail about previous treatment regimen. | Patient is s/p 8 cycles of [REDACTED]. Patient decided to initiate a chemotherapy break following cy |
| P2 | Clinical_Findings | Missing detail about patient's performance status (PS). | Patient's PS is acceptable and he agrees to come in tomorrow for labs and consideration of [REDACTED |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing cancer-related medications and treatment details.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 9 — ⚠️ ISSUES

**Type**: Invasive ductal adenocarcinoma, centered in the head of the pancreas and extendi
**Stage**: Stage III (4 out of 12 lymph nodes positive for carcinoma)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |
| P2 | Treatment_Changes | Missing recent changes regarding gemcitabine. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |
| P2 | Treatment_Goals | Goals should be 'adjuvant' rather than 'curative'. | We previously discussed the role of adjuvant chemotherapy in this context to improve disease-free an |
| P2 | Response_Assessment | Missing mention of imaging findings from 01/23/22. | CT CAP: no evidence of disease |
| P2 | Medication_Plan | Missing mention of gemcitabine. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |
| P2 | Therapy_plan | Missing mention of gemcitabine. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Initially, irinotecan was dose-reduced by 25%, and oxaliplatin was dose-reduced  |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 10 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreatic head
**Stage**: Not staged in note, but with metastatic disease to liver

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Distant Metastasis | The note mentions a liver lesion but does not specify lymph node involvement. | with a liver lesion that is unclear based on imaging. |
| P2 | Lab_Results.lab_summary | CA 19-9 and CEA values are mentioned but not included in the lab summary. | patient does not express CA 19-9 nor does he express ***** |
| P2 | Clinical_Findings.findings | The note mentions elevated LFTs but does not specify the exact cause as bile duct obstruction. | Likely due to bile duct obstruction. |
| P2 | Treatment_Changes.recent_changes | The note mentions holding treatment due to elevated LFTs but does not specify the exact treatment being held. | ***** hold treatment (***** neoadjuvant *****) today because patient is afebrile and has elevated LF |
| P2 | Therapy_plan.therapy_plan | Physical therapy is not mentioned in the note. | Start Cipro 500 mg BID x 7 days for possible intrabdominal infection. |

*Extraction summary*: Most fields are clean, but there are minor issues with distant metastasis, lab summary, clinical findings, treatment changes, and therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'stable' might be too vague for an 8th-grade reading level. | Your recent imaging shows that your cancer has remained stable. |
| P2 | The term 'LFTs' is jargon and might be confusing. | However, you have elevated liver function tests (LFTs), which are likely due to  |
| P2 | The phrase 'status of your cancer' might be unclear. | You will have a CT scan following your next treatment cycle to check on the stat |

*Letter summary*: Letter contains minor readability issues that could be improved for better clarity.

---

## ROW 11 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis | Missing specific stage information | Ms. ***** is a 55 y.o. female who returns to our GI Oncology practice at the UCSF Comprehensive Canc |
| P2 | Lab_Results | Missing some lab values | WBC Count 9.9, Neutrophil Absolute Count 7.66, Hemoglobin 9.3, Hematocrit 29.2, Platelet Count 619,  |
| P2 | Clinical_Findings | Missing details on physical exam findings | Physical exam findings include fatigued and tearful appearance, alopecia, RUQ tenderness, and no sig |
| P2 | Current_Medications | Incomplete listing of current cancer-related medications | She using metoclopramide for its pro-motility properties, and I instructed her today to increase dos |
| P2 | Treatment_Changes | Missing details on recent treatment changes | Based upon cumulative fatigue, I will hold treatment today and ordered restaging CT scans to be perf |
| P2 | Response_Assessment | Missing details on response assessment | The cancer is showing disease progression. This is evidenced by the increase in CA-1[REDACTED]-9 fro |
| P2 | Medication_Plan | Incomplete listing of medication plan | Metoclopramide increased to 10 mg four times daily; megestrol acetate resumed for appetite stimulati |

*Extraction summary*: Most fields are clean, but there are several minor issues related to incomplete or missing cancer-related details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'CA 19-9' and 'tumor marker'. | Recent imaging and lab results show that your cancer is progressing. The CA 19-9 |
| P2 | Inaccurate description of physical exam findings. The original note mentions no signs of jaundice or ascites, but does not mention 'yellowing of your skin'. | Despite this, there is no sign of fluid buildup in your belly or yellowing of yo |
| P2 | Incomplete sentence. The original note specifies the increased dose of metoclopramide. | You are now taking more metoclopramide to help with digestion and have restarted |

*Letter summary*: Letter contains minor readability and accuracy issues.

---

## ROW 12 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic ductal adenocarcinoma with perineural and l
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is described as 'now metastatic ()', which is imprecise and does not specify the exact stage. | The note mentions metastatic disease but does not provide a specific stage beyond metastatic. |
| P2 | Lab_Results.lab_summary | The lab summary includes many non-cancer related blood tests, which are not relevant to the cancer diagnosis or treatment. | The note does not emphasize the need for these specific lab results in the context of cancer treatme |
| P2 | Treatment_Changes.supportive_meds | Gabapentin is mentioned in the therapy plan but not under supportive_meds. | Gabapentin is used for neuropathy, which is a supportive care measure. |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage description, inclusion of non-relevant lab tests, and missing supportive medication.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate statement. The note mentions that the mesenteric lymph nodes have decreased in size, but the imaging report states that the mesenteric nodule has increased in size. | Your disease is stable according to the most recent scans from November 14, 2015 |
| P2 | Unexplained jargon. | Additionally, MMR via IHC or MSI status via PCR will be obtained. |

*Letter summary*: There is one factual error and one readability issue in the letter.

---

## ROW 13 — ⚠️ ISSUES

**Type**: Adenocarcinoma consistent with upper GI/pancreato-biliary primary
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary. The note indicates this is a follow-up visit for evaluation of treatment response and management of the disease, not a new patient visit. | She is now s/p 1 cycle of Gemcitabine + Abraxane on days 1 and 15, started 03/26/18. She presents to |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Inaccurate staging. The note does not specify a stage, only that the cancer is metastatic. | Evaluation of metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Distant Metastasis | Missing information. The note mentions metastasis to the liver and peritoneum. | Additional hypermetabolic hepatic lesions consistent with additional sites of metastatic disease. |
| P2 | Current_Medications.current_meds | Missing cancer-related medications. The note mentions Gemcitabine + Abraxane. | She is now s/p 1 cycle of Gemcitabine + Abraxane on days 1 and 15, started 03/26/18. |
| P2 | Treatment_Changes.supportive_meds | Inaccurate listing of supportive medications. The note mentions ondansetron and prochlorperazine, but does not indicate they are being used. | ondansetron (ZOFRAN), prochlorperazine (COMPAZINE) |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, staging, distant metastasis, current medications, and supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | This sentence contains technical jargon and is overly complex for an 8th-grade reading level. | There is no specific imaging or tumor marker evidence provided in the note to as |

*Letter summary*: Letter is mostly clean but contains some overly complex sentences that could be simplified further.

---

## ROW 14 — ⚠️ ISSUES

**Type**: High-grade neuroendocrine tumor of duodenal/ampullary origin
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. Only 'capecitabine' is listed, but 'temozolomide' is also a cancer-related medication. | She is now s/p 5 cycles of chemotherapy consisting of the combination of capecitabine/temozolomide. |

*Extraction summary*: One minor issue found in Current_Medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The sentence does not specify that the tumor is of duodenal origin, which is important information. | You came in for a follow-up visit regarding your neuroendocrine tumor treatment. |
| P2 | The term 'fluid buildup' might be confusing; 'ascites' is more precise. | During the physical exam, no masses, fluid buildup, or enlarged liver were found |
| P2 | This sentence is incomplete as it doesn't mention the dosing or frequency, though these details are intentionally omitted. | You will continue to take ondansetron and oxycodone to manage side effects. |
| P2 | The sentence lacks the detail about arranging for post-CT IV hydration. | After this cycle, you will have a CT scan with IV contrast to assess your tumor. |

*Letter summary*: The letter contains some inaccuracies and omissions that need addressing.

---

## ROW 15 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is listed as 'Metastatic ()', which is incomplete and could be more precise. | This is a very pleasant 35-year-old Hispanic male who was diagnosed in October 2015 with metastatic  |
| P2 | Treatment_Changes.recent_changes | The note mentions concern about oxaliplatin due to residual neuropathy, but the plan is to resume an unspecified agent, not necessarily oxaliplatin. | He responded initially quite well to ***** but because of his residual neuropathy, I am concerned ab |
| P1 | Therapy_plan.therapy_plan | The plan is to resume an unspecified agent, not specifically oxaliplatin. | We will resume [REDACTED]. He responded initially quite well to [REDACTED] but because of his residu |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage description and major issues with the therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (oxaliplatin). | We will resume a medication. You responded initially quite well to a medication  |
| P2 | Unexplained jargon (omental, peritoneal metastases, ascites). | The cancer is showing progression with extensive peritoneal disease as evidenced |

*Letter summary*: Letter has minor readability issues due to unexplained medical jargon.

---

## ROW 16 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic neuroendocrine tumor, grade 2
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Summary is slightly imprecise, missing key details such as the suspected sporadic condition. | Subjective   ***** ***** is a 34 y.o. female who presents for f/u of pancreatic neuroendocrine tumor |
| P2 | Lab_Results.lab_summary | Lab results are incomplete and contain irrelevant lab values not mentioned in the note. | No specific lab results mentioned in the note. |
| P2 | Current_Medications.current_meds | Field is empty, but the note mentions octreotide and lanreotide. | Off SSA for now-octreotide ***** |
| P2 | Treatment_Changes.recent_changes | Field is empty, but the note mentions stopping octreotide due to intolerance. | June 2021- July 2021: \n octreotide, off for intolerance |
| P2 | Treatment_Goals.goals_of_treatment | Value 'palliative' is imprecise; the note suggests surveillance and management of symptoms. | Continue surveillance q 3 months with abdominal MRI--ordered-- possibly stretch to q 6 mo if scans s |
| P2 | Medication_Plan.medication_plan | Field is incomplete, missing cabergoline for pituitary adenoma. | Continue cabergoline for pituitary adenoma. |
| P2 | Therapy_plan.therapy_plan | Physical therapy is not mentioned in the note. | No mention of physical therapy in the note. |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and precision in several fields.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Grammar error ('You has' should be 'You have'). | You has over 50 liver metastases that are stable and were present back to May 20 |
| P2 | Unexplained medical jargon 'hepatic', 'lymph node', 'metastases'. |  |

*Letter summary*: Letter contains a minor grammatical error and some unexplained medical jargon.

---

## ROW 17 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and capecitabine). | She has now had 4 full cycles and returns today for follow-up imaging. When we last saw her, she had |
| P2 | Treatment_Changes | Missing ongoing chemotherapy (gemcitabine and capecitabine). | She will continue on chemotherapy without dose or schedule modification and we will also start denos |
| P2 | Imaging_Plan | Inaccurate imaging plan, should include both CT Chest and CT Abdomen/Pelvis. | She will continue on chemotherapy without dose or schedule modification and we will also start denos |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing ongoing chemotherapy and an incomplete imaging plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 18 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma, grade 2; Intra-ampullary papillary-tubular neo
**Stage**: Originally Stage IB, now metastatic (pt2n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | Medication Instructions section lists general medications only. |
| P2 | Treatment_Changes | Supportive medications listed are not fully supported by the note. | Supportive medications mentioned are ondansetron, prochlorperazine, and loperamide, but the note doe |
| P2 | Treatment_Goals | Goals of treatment are stated as 'palliative', but the note suggests a mix of palliative and surveillance strategies. | The note discusses ongoing monitoring and potential future treatments, suggesting a mixed approach. |
| P2 | Response_Assessment | The response assessment mentions a recent bump in CA 19-9 levels, but the exact level is not provided. | CA 19-9 levels are mentioned as having increased, but the specific value is not given. |
| P2 | Medication_Plan | The plan mentions unspecified agent and unspecified regimens, which are [REDACTED] in the note. | Specific drug names are [REDACTED] in the note, leading to incomplete medication plan. |
| P2 | Therapy_plan | The therapy plan mentions unspecified agent and unspecified regimens, which are [REDACTED] in the note. | Specific drug names are [REDACTED] in the note, leading to incomplete therapy plan. |
| P2 | Procedure_Plan | The procedure plan mentions unspecified procedure, which is [REDACTED] in the note. | Specific procedure name is [REDACTED] in the note, leading to incomplete procedure plan. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, incomplete supportive medications, and unspecified treatment details.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 19 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma, likely of pancreatic origin
**Stage**: (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'rivaroxaban (XARELTO)' | rivaroxaban (XARELTO) 20 mg tablet Take 1 tablet (20 mg total) by mouth daily with dinner. |
| P2 | Treatment_Changes | Missing supportive medication 'palonosetron' | Start 2 days after last dose of palonosetron (*****) is given. |

*Extraction summary*: Most fields are clean, but there are minor omissions in cancer-related medications and supportive care medications.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 20 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma, T3N1, with a component of intraductal papillary mucin
**Stage**: pT 3N1

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine). | He initiated adjuvant chemotherapy consisting of single-agent gemcitabine on a three-week-on, one-we |

*Extraction summary*: One minor issue identified regarding the omission of a completed cancer-related medication. All other fields are clean.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 21 — ⚠️ ISSUES

**Type**: Metastatic pancreatic cancer
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (5-FU/LV + irinotecan). | On this basis, he initiated 2nd-line chemotherapy consisting of the combination of 5-*****/LV plus * |
| P2 | Treatment_Changes | Missing detail about dose reduction of 5-FU/LV + irinotecan. | Due to obstipation and anorexia following cycle #1, his doses were reduced by 20% beginning with cyc |
| P2 | Treatment_Goals | Incomplete. Should include 'symptom management' along with 'palliative'. | Therefore we had a ***** goals of care discussion in which I recommended that he refocus his goals p |
| P2 | Medication_Plan | Incorrect. The patient is no longer a candidate for further chemotherapy. | At this point, he is no longer a candidate for any further salvage treatment in terms of either SOC  |
| P2 | Therapy_plan | Incorrect. The patient is no longer a candidate for further chemotherapy. | At this point, he is no longer a candidate for any further salvage treatment in terms of either SOC  |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, incomplete treatment goals, and incorrect therapy plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence, missing critical information about the specific dose reduction. | Your chemotherapy doses were reduced by 20% starting with cycle #2 due to side e |

*Letter summary*: Letter contains an incomplete sentence regarding dose reduction.

---

## ROW 22 — ⚠️ ISSUES

**Type**: Pancreatic cancer
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Age discrepancy between '70 y.o.' in summary and '71 y.o.' in impression. | Impression: 71 y.o. female with metastatic pancreatic cancer as summarized above. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage 'Metastatic ()' is vague and could be more specific. | Impression: 71 y.o. female with metastatic pancreatic cancer as summarized above. |
| P2 | Current_Medications.current_meds | Missing cancer-related medication 'liposomal irinotecan'. | 02/04/17 ***** ***** ***** 02/21/17 ***** Dose reduced 5FU 66.7% |
| P2 | Treatment_Changes.supportive_meds | Missing supportive medication 'olanzapine (Zyprexa)' for nausea. | s/e -n/v + relief with addition of olanzapine (Zyprexa) |
| P2 | Treatment_Goals.goals_of_treatment | Goals of treatment should be 'palliative' but could also include 'symptom management'. | She has essentially stable disease by imaging by shows concern for progression by symptoms. |
| P2 | Response_Assessment.response_assessment | Recent scans show no evidence of progression, but tumor markers trending upwards should be highlighted as a concern. | Tumor markers have been trending upwards by  30% over the last |
| P2 | Medication_Plan.medication_plan | Should specify ongoing treatment with 'liposomal irinotecan' and '5FU', with dose reductions. | Dose reduced 5FU 66.7% due to WBC counts |
| P2 | Therapy_plan.therapy_plan | Should mention ongoing chemotherapy with 'liposomal irinotecan' and '5FU'. | Dose reduced 5FU 66.7% due to WBC counts |
| P2 | follow_up_next_visit.Next clinic visit | Next clinic visit date should be specified. | Not specified in the given text |

*Extraction summary*: Most fields are clean, but there are minor issues with age discrepancy, vague staging, missing cancer-related medications, and lack of specificity in treatment plans and next visit dates.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'imaging'. | Recent imaging shows that your cancer is currently stable. |
| P2 | Unexplained jargon 'tumor markers'. | However, tumor markers have increased by 30%, which could indicate a potential c |
| P2 | Unexplained jargon 'physical exam'. | Your physical exam revealed no significant abnormalities, with normal breath sou |
| P2 | Unexplained jargon 'hand-foot rash'. | There is an improved hand-foot rash bilaterally. |
| P2 | Incomplete sentence with missing critical info. | The dose of 5FU was reduced to 66.7% due to low white blood cell counts. |
| P2 | Unexplained jargon 'phase I', 'genetic profile'. | You have been referred to phase I to evaluate your candidacy for early-stage tre |

*Letter summary*: The letter contains minor readability issues and unexplained medical jargon.

---

## ROW 23 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine + [REDACTED]-paclitaxel) | He has recently resumed chemotherapy with gemcitabine/[REDACTED]-paclitaxel on a dosing schedule of  |
| P2 | Treatment_Changes | Missing recent treatment change (resumed chemotherapy with gemcitabine/[REDACTED]-paclitaxel) | He has recently resumed chemotherapy with gemcitabine/[REDACTED]-paclitaxel on a dosing schedule of  |
| P2 | Treatment_Goals | Inaccurate goal (should be 'palliative') | We reviewed his most recent imaging studies which demonstrate a mixed response; while some of his pe |
| P2 | Radiotherapy_plan | Incomplete radiotherapy plan (missing details on controlling the disease with radiation therapy) | Given its central location with mediastinal involvement, I would favor d/c'ing current therapy, and  |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, treatment changes, and incomplete radiotherapy plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 24 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic adenocarcinoma with perineural invasion
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though reasonable inference could be made from the note. | A 2.2 cm well-differentiated adenocarcinoma with extension into the peri-pancreatic soft tissue and  |
| P1 | Treatment_Goals.goals_of_treatment | Incorrect goal of treatment. Should be 'surveillance' since the patient has completed adjuvant therapy and is on follow-up. | He looks and feels well overall, aside from some endocrine and exocrine insufficiency post-surgery/t |
| P2 | Response_Assessment.response_assessment | The response assessment suggests poor response to prior treatment, which is not clearly supported by the note. The note only mentions concern for possible pulmonary metastases. | These findings are concerning for possibly pulmonary metastases, indicating that the cancer may not  |

*Extraction summary*: Major issues with treatment goals and response assessment. Minor issue with missing cancer stage.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | This sentence implies that the cancer is not responding well to the prior treatment, which is not supported by the clinical note. The note suggests monitoring the findings closely rather than concluding that the cancer is not responding. | These findings are concerning for possibly pulmonary metastases, indicating that |

*Letter summary*: Letter contains a potentially misleading statement about the cancer response to prior treatment.

---

## ROW 25 — ⚠️ ISSUES

**Type**: Moderately differentiated Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medications. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |
| P2 | Treatment_Changes | Missing detail about the specific chemotherapy regimen. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |
| P2 | Treatment_Goals | Inaccurate goal classification; should be 'palliative' but note suggests ongoing treatment response monitoring. | Currently showing a treatment response based on the following evidence: CT scans from 11/10/2020 ind |
| P2 | Medication_Plan | Missing detail about the specific chemotherapy regimen. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |
| P2 | Therapy_plan | Missing detail about the specific chemotherapy regimen. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the completeness of cancer-related medications and treatment details.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 26 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it could be inferred from the note. | Evaluation of locoregional extent of disease is limited by surrounding peripancreatic fluid collecti |
| P2 | Lab_Results.lab_summary | Lab results are not fully detailed in the note, and some values are missing. | AST 261, ALT 320, AP 502, Tbili 5.6, CA 19-9: 324, AST 237, ALT 365, AP 495, TBili 7.7 |
| P2 | Current_Medications.current_meds | Only gemcitabine is listed, but the note mentions moxifloxacin which was stopped recently. | Stopped moxifloxacin on 05/15/21 |
| P2 | Treatment_Changes.supportive_meds | Supportive medications are not fully detailed in the note. | No specific supportive medications mentioned except for moxifloxacin. |
| P2 | Treatment_Goals.goals_of_treatment | The goal is listed as 'curative', but the note suggests ongoing management of complications and no mention of curative intent. | Overall, the cancer appears to be stable with ongoing management of complications. |
| P2 | Medication_Plan.medication_plan | The plan mentions continuing moxifloxacin, but it was actually stopped on 05/15/21. | Stopped moxifloxacin on 05/15/21 |
| P2 | Therapy_plan.therapy_plan | The plan mentions continuing moxifloxacin, but it was actually stopped on 05/15/21. | Stopped moxifloxacin on  05/15/21 |

*Extraction summary*: Most fields are clean, but there are minor issues with missing lab details, medication plan, and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'hypermetabolic'. | You have a hypermetabolic pancreatic head mass measuring approximately 2.5 x 2.3 |
| P2 | Unexplained jargon 'pulmonary nodules'. | Scattered small pulmonary nodules are unchanged. |
| P2 | Unexplained jargon 'peripancreatic fluid collection', 'psoas muscle'. | The peripancreatic fluid collection has decreased in size but has a new componen |
| P2 | Unexplained jargon 'tumor markers'. | Tumor markers (CA 19-9) are elevated, suggesting no significant improvement. |

*Letter summary*: Letter contains some unexplained medical jargon that could be clarified for better readability.

---

## ROW 27 — ⚠️ ISSUES

**Type**: Extrahepatic cholangiocarcinoma; Metastatic hepatocellular carcinoma
**Stage**: Unresectable; Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing atezolizumab and bevacizumab, which are cancer-related treatments. | Patient was recommended to start treatment ***** + bevacizumab Q 3 ***** ***** for metastatic HCC, w |
| P2 | Treatment_Changes | Missing mention of starting atezolizumab + bevacizumab. | Patient was recommended to start treatment ***** + bevacizumab Q 3 ***** ***** for metastatic HCC, w |
| P2 | Treatment_Goals | Should specify 'palliative' for metastatic HCC. | In summary, Ms. ***** is a 78 y.o. female who returns to our GI Oncology practice at the UCSF Compre |
| P2 | Medication_Plan | Missing atezolizumab and bevacizumab, which are cancer-related treatments. | Patient continues atezolizumab + bevacizumab Q 3 [REDACTED]. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Use of 'hypodense lesions' might be too technical for an 8th-grade reading level. | Your recent CT scan shows stable disease with unchanged hypodense lesions in hep |
| P2 | Use of 'mesenteric nodes' might be too technical for an 8th-grade reading level. | However, there is some concern about slow progression in the mesenteric nodes. |

*Letter summary*: Letter is mostly clean but contains some technical terms that could be simplified further for an 8th-grade reading level.

---

## ROW 28 — ⚠️ ISSUES

**Type**: Pancreatic head adenocarcinoma; Gallbladder carcinoma
**Stage**: Locally advanced with metastatic disease to liver

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note does not explicitly state that the patient is not currently on any cancer-related medications. | The note does not mention any ongoing cancer-related medications. |

*Extraction summary*: Most fields are clean, but there is a minor issue with the Current_Medications field being empty.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of the cancer involvement. The original note mentions 'primary pancreatic head adenocarcinoma with metastatic gallbladder/liver involvement; vs a multifocal biliary tract/GB cancer'. | You have locally advanced pancreatic cancer with metastatic disease to the liver |
| P2 | Unexplained medical jargon 'area' and 'inflammation'. | Recent imaging shows a 4.8 cm mass in the gallbladder with liver involvement and |
| P2 | Unexplained medical jargon 'hematocrit', 'alkaline phosphatase', 'albumin'. | Your blood tests show elevated white blood cell count, low hemoglobin, low hemat |

*Letter summary*: The letter contains inaccuracies and minor readability issues.

---

## ROW 29 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary mentions 'new diagnosis', which contradicts the note stating 'new diagnosis of metastatic adenocarcinoma of the pancreas'. | This is a very pleasant 61-year-old woman with a new diagnosis of metastatic adenocarcinoma of the p |
| P2 | Current_Medications.current_meds | The field only includes 'Gemcitabine, Abraxane', but the note mentions 'Lovenox' as part of the treatment. | She was started back on Lovenox. |
| P2 | Treatment_Changes.recent_changes | The field only includes 'Restarted Gemcitabine + Abraxane', but the note mentions 'Lovenox' as part of the recent changes. | She was started back on Lovenox. |
| P2 | Medication_Plan.medication_plan | The field only includes 'Gemcitabine, Abraxane, Lasix, spironolactone, Ritalin, Mirtazapine', but the note mentions 'Lovenox' as part of the medication plan. | She was started back on Lovenox. |
| P2 | Therapy_plan.therapy_plan | The field only includes 'Gemcitabine, Abraxane, lasix, home health', but the note mentions 'Lovenox' as part of the therapy plan. | She was started back on Lovenox. |

*Extraction summary*: Most fields are clean, but there are minor issues related to the inclusion of 'Lovenox' in the medication and treatment plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 30 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced; Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note mentions both locally advanced and metastatic disease, but the extraction does not specify the exact stage. | Locally advanced; Metastatic () |
| P2 | Treatment_Changes.recent_changes | The note states that the patient agreed to take a break from chemotherapy, but the extraction does not mention the reason for the break. | We had another discussion during today's visit re: the importance of carefully weighing the risks/be |
| P2 | Treatment_Goals.goals_of_treatment | The note suggests a more nuanced approach to treatment goals, but the extraction simplifies it to 'palliative'. | We reviewed her most recent imaging studies performed during that hospitalization that show essentia |
| P2 | Medication_Plan.medication_plan | The note mentions that the patient is on reduced-dose gemcitabine and [REDACTED]-paclitaxel, but the extraction incorrectly includes 'irinotecan'. | She is now back on chemotherapy consisting of reduced-dose gemcitabine/[REDACTED]-paclitaxel. |
| P2 | Lab_Plan.lab_plan | The note mentions short-term follow-up CT scans and bloodwork, but the extraction states 'No labs planned.' | We will plan on short-term follow up CT scans and bloodwork (right after [REDACTED]). |

*Extraction summary*: Most fields are clean, but there are minor issues with the staging, treatment goals, medication plan, and lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The sentence omits the mention of new liver lesions and biliary issues, which are important clinical details. | Recent imaging shows that the cancer in your pancreas is stable. There are no ne |
| P2 | The sentence is slightly imprecise as it does not specify the timeframe. | Your CA 19-9 (a tumor marker) level has stayed within the normal range. |
| P2 | The sentence is incomplete as it does not specify the duration of the break. | You agreed to take a break from chemotherapy. |
| P2 | The sentence omits the purpose of the ERCPs, which is to manage biliary obstruction. | You will have repeat ERCPs (a procedure to place a small tube to keep the bile d |
| P2 | The sentence is vague and lacks specificity. | Short-term follow-up CT scans and bloodwork will be done soon after your break f |
| P2 | The sentence is slightly imprecise as it does not specify the timing of the liquid biopsy. | A liquid biopsy (a test to look for genetic changes) will be ordered to check fo |

*Letter summary*: The letter contains minor omissions and imprecisions that need clarification for completeness and accuracy.

---

## ROW 31 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it could be inferred as Stage IV due to metastasis. | Metastatic adenocarcinoma the pancreas improving on therapy with resolution of ascites |
| P2 | Current_Medications.current_meds | Missing specific details about the dosing and schedule of gemcitabine. | He is now been through 3 cycles of alternate week fixed dose rate gemcitabine. |
| P2 | Treatment_Changes.recent_changes | Field is empty, but there might be relevant changes in supportive care or other treatments not captured. | He continues on treatment without schedule or dose modification. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing stage information, incomplete medication details, and lack of recent treatment changes.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 32 — ⚠️ ISSUES

**Type**: Metastatic moderately differentiated adenocarcinoma of pancreatic or biliary ori
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine, atorvastatin, etc.) | Medications the patient states to be taking prior to today's encounter. |
| P2 | Treatment_Changes | Supportive medications listed are not exclusively cancer-related | Supportive medications include primary prophylaxis with growth factor support ([REDACTED] or Neupoge |
| P2 | Treatment_Goals | Goals of treatment should be more specific (e.g., palliative with disease control) | The goals of such treatment are to produce disease control and hopefully extend survival, while poss |
| P2 | Medication_Plan | Incomplete listing of chemotherapy agents (missing leucovorin) | The patient is inclined to move ahead with standard of care (SOC) chemotherapy with [REDACTED]. |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and specificity in medication and treatment plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'metastases'. | You have a 6 cm irregular mass in the body of your pancreas with multiple liver  |
| P2 | Unexplained medical jargon 'growth factor support'. | Supportive medications include primary prophylaxis with growth factor support af |
| P2 | Unexplained medical jargon 'Mediport'. | You are scheduled to have a Mediport placed through your local oncologist. |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 33 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma with biopsy-proven liver metastasis
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (irinotecan). | She was started on ***** 06/07/19 and has been tolerating it well with dose reduction of irinotecan  |
| P2 | Treatment_Changes | Inaccurate description of recent treatment changes. Irinotecan was reduced, not switched. | She was started on ***** 06/07/19 and has been tolerating it well with dose reduction of irinotecan  |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' rather than 'palliative'. | my recommendation is to switch her therapy to gemcitabine combined with *****-paclitaxel. |
| P2 | Medication_Plan | Inconsistent mention of irinotecan. It should be removed since it's being switched. | Plan for gemcitabine 1000 mg/m2 + [REDACTED]-paclitaxel 125 mg/m2 on days 1, 8, 15 out of 28-day cyc |
| P2 | Therapy_plan | Inconsistent mention of irinotecan. It should be removed since it's being switched. | Continue irinotecan. Plan for gemcitabine 1000 mg/m2 + [REDACTED]-paclitaxel 125 mg/m2 on days 1, 8, |
| P2 | Lab_Plan | Incomplete lab plan. Should include monitoring of Carcinoembryonic Antigen and Cancer Antigen 19-9 levels. | We will monitor [REDACTED] levels monthly. |

*Extraction summary*: Most fields are clean, but there are minor issues with medication plans, treatment goals, and lab monitoring.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'enhancement' may be confusing to a layperson. | Recent imaging shows stable disease in the liver but progressive disease in the  |
| P2 | While 'Cancer Antigen 19-9' and 'Carcinoembryonic Antigen' are acceptable terms, they might still be confusing. | Your Cancer Antigen 19-9 and Carcinoembryonic Antigen levels remain high. |
| P2 | The term 'gemcitabine and nab-paclitaxel' might be confusing. | You will receive gemcitabine and nab-paclitaxel on days 1, 8, and 15 of a 28-day |

*Letter summary*: Letter contains minor readability issues that could be improved for clarity.

---

## ROW 34 — ⚠️ ISSUES

**Type**: Adenocarcinoma
**Stage**: Borderline resectable

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication. | He is being treated in a neoadjuvant fashion with *****. |
| P2 | Treatment_Changes | Supportive medications listed but not specified as cancer-related. | Also not clear why this ambulatory fit man is getting inpatient *****. |
| P2 | Treatment_Goals | Goal of treatment is not clearly curative or palliative. | We discussed that he should get 4-6 cycles over 2-3 months and then reimage with a quad phase protoc |
| P2 | Medication_Plan | Specific medication name is redacted but not explicitly stated as unknown. | The patient is being treated in a neoadjuvant fashion with unspecified agent ([REDACTED]). |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing specific cancer-related medications and unclear classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (specific medication name). | You are being treated in a neoadjuvant fashion with a medication. |
| P2 | Slightly imprecise wording. It might be better to explain the intent of the treatment in simpler terms. | The goal of treatment is curative. |

*Letter summary*: Letter contains one critical issue and one minor readability concern.

---

## ROW 35 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreatic tail with metastatic disease to liver
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions several medications that are not cancer-related. | Current Outpatient Prescriptions: ERGOCALCIFEROL, VITAMIN D2, escitalopram oxalate, lisinopril, lora |
| P2 | Treatment_Changes | The field is empty, but the note discusses potential future treatment changes based on lab results and clinical trials. | My recommendation would be consideration of the ***** trial to determine if the patient has high ser |

*Extraction summary*: Most fields are clean, but 'Current_Medications' and 'Treatment_Changes' are incomplete.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (EUS). | MRI and EUS tests showed a large pancreatic tumor and four liver tumors, suggest |
| P2 | Unexplained jargon (genetic counselor). | You will be referred to a genetic counselor to check for inherited cancer syndro |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

---

## ROW 36 — ⚠️ ISSUES

**Type**: Nonfunctioning pancreatic neuroendocrine tumor
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. The note mentions everolimus but does not include other cancer-related medications like Sandostatin (mentioned as poorly tolerated previously). | Previous treatment has included a distal pancreatectomy of her primary lesion back in 2002; several  |
| P2 | Treatment_Changes | Incomplete. The note mentions that everolimus was held during partial SBO but does not mention the specific dates or reasons for holding it. | Everolimus held during that period of time, resumed two weeks ago. |
| P2 | Treatment_Goals | Slightly imprecise. The note suggests 'good durable disease control' rather than just 'palliative'. | with evidence of good durable disease control and reasonably good (in fact, improved) tolerance on c |

*Extraction summary*: Most fields are clean, but there are minor issues related to completeness and precision in medication and treatment details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (dose number). | Your dose of everolimus was reduced from 10 . This change was made in April 2012 |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 37 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary mentions 'new patient' which is inconsistent with the note indicating a follow-up visit. | Follow Up    Mr. ***** is a 89 y.o. male whom I am seeing as a video visit at the UCSF Comprehensive |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage of cancer is not provided, although the note does not explicitly state the stage. | Baseline imaging 09/28/19 demonstrated a 2.5cm pancreatic uncinate mass that appeared resectable and |
| P2 | Current_Medications.current_meds | The current medications field is empty, but the note mentions the patient is currently off anticoagulation and has a history of receiving gemcitabine and abraxane. | Currently off anticoagulation. Will consider [REDACTED] if local recurrence only vs 5FU/nal [REDACTE |
| P2 | Treatment_Changes.recent_changes | The recent changes field is empty, but the note indicates the patient has completed 8 cycles of neoadjuvant chemotherapy and is now off anticoagulation. | Completed 8 cycles of ***** (*****=07/06/20) with stable disease on scans. |
| P2 | Treatment_Goals.goals_of_treatment | The treatment goal is listed as 'surveillance', but the note suggests a more complex situation involving potential recurrence and ongoing management. | Most recent CT ***** January 20 and CTA abdomen 01/28/21 do not demonstrate clear recurrence though  |
| P2 | Medication_Plan.medication_plan | The medication plan mentions being off anticoagulation and considering specific treatments, but the exact drugs are [REDACTED], making the plan incomplete. | Currently off anticoagulation. Will consider [REDACTED] if local recurrence only vs 5FU/nal [REDACTE |
| P2 | Therapy_plan.therapy_plan | The therapy plan mentions continuing gemcitabine and starting physical therapy, but the note does not specify starting physical therapy. | Continue/start: gemcitabine; physical therapy |
| P2 | Procedure_Plan.procedure_plan | The procedure plan mentions considering a surgical bypass, but the note does not specify the timing or certainty of this procedure. | will consider presentation at tumor board --d/w Dr. [REDACTED] role and timing of surgical bypass gi |
| P2 | follow_up_next_visit.Next clinic visit | The next clinic visit is not specified, but the note does not provide a specific date for the next visit. | Not specified in the provided text |

*Extraction summary*: Several fields contain minor issues related to completeness and classification, but no major errors or hallucinations.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'adenocarcinoma' is used without explanation. | You have pancreatic adenocarcinoma (cancer that started in gland cells). |
| P2 | The term 'CA 19-9' is used without explanation. | The most recent CA 19-9 value is 8, indicating stable disease. |
| P2 | The term 'anticoagulation' is used without explanation. | You are currently off anticoagulation. |
| P2 | The term 'stenting' is used without explanation. | Your doctors will consider a surgical bypass if you have ongoing symptoms despit |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 38 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma of pancreatic origin with extensive per
**Stage**: Originally pT3N0, now metastatic (pt3n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine). | She states that after the last cycle of chemotherapy, it took her awhile to recover in terms of fati |
| P2 | Treatment_Changes | Missing recent treatment changes (gemcitabine completion). | Her last dose of gemcitabine was at the end of December 2012. |
| P2 | Treatment_Goals | Inaccurate goal description. Should include both palliation and symptom management. | Our recommendation would be try treatment with ***** or ***** and see how well she tolerates it. We  |
| P2 | Medication_Plan | Incomplete. Missing specific supportive medications. | We reviewed some of the potential side effects including but not limited to : bone marrow suppressio |
| P2 | Therapy_plan | Incorrect. No mention of radiation therapy referral in the note. | The decision on the need and timing of re-initiating chemotherapy will depend on the extent and prog |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, treatment changes, and incomplete medication plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | You will be recommended to try treatment with a medication or a medication. |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 39 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma, MMR intact
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'Gemcitabine + Abraxane' as the treatment regimen. | Patient has been recommended to proceed with 1L Gemcitabine + Abraxane |
| P2 | Lab_Plan | The field contains irrelevant text instead of a proper lab plan. | Plan       ***** ***** is a 63 y.o.  male who presents with metastatic pancreati |

*Extraction summary*: Most fields are clean, but there are minor issues with 'Current_Medications' being empty and 'Lab_Plan' containing irrelevant text.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Your treatment (a medication/Ab) was paused for one week because of ongoing fati |
| P2 | Unexplained medical jargon 'cystic mass'. | The cystic mass in your pancreas also hasn't changed. |

*Letter summary*: The letter contains an incomplete sentence and minor readability issues.

---

## ROW 40 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) in the current medications section. | We recommend a course of treatment with gemcitabine and Abraxane. |

*Extraction summary*: One minor issue identified in the Current_Medications field.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (lacks the specific dose increase). | We plan to increase the dose of your pain patch, and added Reglan and when neces |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 41 — ⚠️ ISSUES

**Type**: Invasive adenocarcinoma of pancreatic head, post-treatment tumor appears well- t
**Stage**: Not available (redacted), now with local recurrence

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (chemotherapy). | He initiated chemotherapy with ***** at full doses (minus the 5-***** bolus) beginning 05/15/2018, * |
| P2 | Treatment_Changes | Missing recent treatment changes. | ***** cycle (#12) was administered with 20% dose reductions in both oxaliplatin and irinotecan due t |
| P2 | Imaging_Plan | Missing recommendation for PET/CT. | Further evaluation with PET/CT may be helpful for confirmation. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, treatment changes, and imaging recommendations.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'CT scans' and 'liver lesion'. | Your recent CT scans show a decrease in size of a liver lesion, initially measur |
| P2 | Unexplained medical jargon 'soft tissue' and 'local recurrence'. | However, there is an increase in soft tissue near the surgical site, suggesting  |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

---

## ROW 42 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally recurrent (progressing)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions ongoing use of mirtazapine. | Take 1 tablet (15 mg total) by mouth nightly at bedtime. |
| P2 | Treatment_Changes | Supportive medications listed include non-cancer pain medications (HYDROcodone-acetaminophen, HYDROcodone-ibuprofen). | Take 1 tablet by mouth every 8 (eight) hours as needed for Pain. |
| P2 | Imaging_Plan | The field only mentions 'CT Chest', but the note suggests a comprehensive imaging plan including abdominal and pelvic scans. | Please see a separately reported examination for evaluation of the abdomen. |

*Extraction summary*: Most fields are accurate, but there are minor issues with current medications, supportive medications, and imaging plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 43 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Borderline resectable pancreatic adenocarcinoma due to <180 degrees of vascular 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | The note mentions 'CA 19-9 = 115', 'CA 19-9 = 141', 'CA 19-9 = 80', 'CA 19-9 = 105', and 'CA 19-9 =  |

*Extraction summary*: One minor issue found in Current_Medications, otherwise all fields are clean.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 44 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field only lists 'gemcitabine', but the note mentions '5FU/LV' and 'gemcitabine/abraxane'. | Started on Lovenox for bilateral PE. Switch to gemcitabine/abraxane starting 10/25/20. Held oxalipla |
| P2 | Treatment_Changes | The field mentions 'irinotecan' under 'supportive_meds', but irinotecan is a chemotherapy agent, not a supportive medication. | Stopped irinotecan with C3. |
| P2 | Medication_Plan | The field incorrectly includes 'irinotecan' under the medication plan, while the note indicates irinotecan was stopped due to colitis. | Stopped irinotecan with C3. |

*Extraction summary*: Most fields are clean, but there are minor issues with the inclusion of 'irinotecan' in the medication plan and supportive medications, and the current medications list needs to include '5FU/LV'.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 45 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic adenocarcinoma
**Stage**: Resected (margin-negative surgery)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine). | He was eventually able to start on adjuvant chemotherapy consisting of single-agent gemcitabine on 0 |
| P2 | Treatment_Changes | Supportive medications are missing. | The patient's present review of systems was reviewed and notable for the following: - Some cumulativ |
| P2 | Treatment_Goals | Goals of treatment should be 'adjuvant' rather than 'curative'. | In summary, Mr. ***** is a 64 y.o. male from *****, now 5 months s/p ***** resection of a node-negat |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing current cancer-related medication, supportive medications, and the classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The term 'moderately differentiated' is too technical for an 8th-grade reading level. | You have a moderately differentiated pancreatic adenocarcinoma (cancer that star |
| P2 | The term 'postoperative changes' may be confusing. | Imaging from 08/12/2017 shows postoperative changes, small bilateral pleural eff |

*Letter summary*: Letter contains minor readability issues that need addressing.

---

## ROW 46 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not specify the exact stage, only that it is metastatic. | Ms. ***** is a 72 y.o. female who returns to our GI Oncology practice at the UCSF Comprehensive Canc |
| P2 | Cancer_Diagnosis.Distant Metastasis | The note mentions metastasis to the liver and possibly adrenal gland, but the extracted data states 'Not sure'. | hypodense hepatic segment 8 lesion measuring up to 1.7 cm(previously noted to measure 0.3 cm), and * |
| P2 | Current_Medications.current_meds | The note mentions gemcitabine as part of the chemotherapy regimen, but it is not listed under current medications. | initiated chemotherapy consisting ofthe combination of gemcitabine plus *****-***** ***** dosing sch |
| P2 | Treatment_Changes.recent_changes | The note mentions several recent changes in treatment, but the extracted data does not capture these changes. | Initiated *****, minus the 5FU bolus  - 11/19/2021: Admitted to ***** campus for acute on chronic LU |
| P2 | Treatment_Changes.supportive_meds | The note mentions several supportive medications, but the extracted data only lists a subset. | Continue on PPI therapy and carafate for GI bleeding. Continue anticoagulation with Eliquis without  |
| P2 | Treatment_Goals.goals_of_treatment | The note suggests a palliative approach, but the extracted data does not fully capture the nuances of the discussion. | I do think at some point it will be of great advantage to re-establish home hospice services, when s |
| P2 | Response_Assessment.response_assessment | The note provides more detail on the response assessment, but the extracted data simplifies it. | Compared to 11/19/2021, slight decrease in size of ill-defined pancreatic head mass and adjacent mes |
| P2 | Medication_Plan.medication_plan | The note mentions several specific medications and plans, but the extracted data does not fully capture them. | Continue on PPI therapy and carafate for GI bleeding. Continue anticoagulation with Eliquis without  |
| P2 | Therapy_plan.therapy_plan | The note mentions no current or future chemotherapy, but the extracted data does not fully capture the discussion. | We reviewed her most recent imaging studies which show continued non-progression of her disease, wit |
| P2 | Procedure_Plan.procedure_plan | The note mentions a possible endoscopic evaluation, but the extracted data does not fully capture the context. | We talked about endoscopic evaluation which she is somewhat reluctant to pursue but will take time t |
| P2 | Imaging_Plan.imaging_plan | The note mentions further imaging dictated by the family's request, but the extracted data does not fully capture the context. | Further imaging dictated by the family's request primarily for prognostic purposes. |
| P2 | follow_up_next_visit.Next clinic visit | The note mentions a follow-up in 3-4 weeks, but the extracted data does not fully capture the context. | We will follow up again in another ~3-4 weeks' time, with further imaging dictated by the family's r |

*Extraction summary*: Several fields contain minor issues related to incomplete or slightly imprecise information, but no major errors or hallucinations.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 47 — ⚠️ ISSUES

**Type**: Pancreatic neuroendocrine tumor
**Stage**: Locally advanced (unresectable due to vascular involvement)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Lab_Results | The lab results listed do not match the lab values provided in the note. | Cr 0.77, fasting glucose 66, AST 188, ALT 217, alkaline phosphatase 823, total bilirubin 2.9, gamma  |
| P2 | Current_Medications | Missing other cancer-related medications like octreotide and metformin. | Continue everolimus 5 mg daily. Annual B12 on octreotide. Check cholesterol and TSH (q 3 mo) on ever |
| P2 | Treatment_Changes | Missing recent dose reduction of everolimus. | 12/02/15-present: Everolimus 10 mg QD for hypoglycemia |
| P2 | Response_Assessment | Inaccurate physical exam details, missing mention of abdominal pain being stable. | Physical Exam: Well-developed, well-nourished in no apparent distress. Abdominal: Soft, nontender, n |
| P2 | Therapy_plan | Missing specific plans for monitoring and managing diabetes. | BSL rising --on metformin but Hgb A1C high/rising. Managed by Dr *****. Recently ***** metformin dos |
| P2 | Procedure_Plan | Missing specific plans for tooth extraction and its impact on everolimus. | Would hold everolimus before and after as can delay wound healing-- will need to ***** ***** ***** |
| P2 | Lab_Plan | Missing specific plans for monitoring insulin, proinsulin, and c-peptide levels. | Follow up markers every 3 months: A1c, insulin, proinsulin, c-peptide, fasting blood glucose. |

*Extraction summary*: Most fields are clean, but there are several minor issues with completeness and accuracy of lab results, medication plan, and response assessment.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of imaging findings. The original note mentions an interval increase in nodular thickening of the left adrenal gland, but does not state that there has been no significant change in the pancreatic lesion. It states that the pancreatic lesion is stable. | There has been an increase in nodular thickening of the left adrenal gland, but  |
| P2 | Unexplained medical jargon 'nodule'. | A nodule in your left upper lung has become more solid, while a smaller nodule i |
| P2 | Unexplained medical jargon 'hemoglobin', 'hematocrit', 'fasting glucose'. | Your hemoglobin and hematocrit levels are slightly low, and your fasting glucose |
| P2 | Unexplained medical jargon 'everolimus', 'octreotide', 'metformin', 'entecavir', 'hepatitis B'. | You will continue taking everolimus to manage your condition. You will also cont |

*Letter summary*: The letter contains some inaccuracies and minor readability issues.

---

## ROW 48 — ⚠️ ISSUES

**Type**: Infiltrative pancreatic head and uncinate process adenocarcinoma
**Stage**: At least locally advanced, more likely metastatic

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | Her treatment course to date has included ***** stenting (metallic) on 09/17/2016 due to the develop |
| P2 | Treatment_Changes | Missing mention of 'gemcitabine' in the treatment changes section. | Her treatment course to date has included ***** stenting (metallic) on 09/17/2016 due to the develop |
| P2 | Treatment_Goals | Goals of treatment should include 'symptom management' in addition to 'palliative'. | Otherwise, we will refocus our goals on purely palliative/symptomatic measures with a referral to ho |
| P2 | Medication_Plan | Incorrectly includes 'gemcitabine' under medication plan when it was already administered. | Prescribe low-dose Decadron as an appetite stimulant.; also: gemcitabine |
| P2 | Advance_care_planning | Advance care planning was discussed during the visit. | I had a frank goals of care discussion with her and her family. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, treatment goals, and advance care planning.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'fluid buildup' and 'fluid around your lungs'. | Imaging shows that your cancer has progressed, with increased fluid buildup in y |
| P2 | Mention of 'IV fluids' without context might be confusing. | You were prescribed low-dose Decadron to help improve your appetite. You also re |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

---

## ROW 49 — ⚠️ ISSUES

**Type**: Localized pancreatic cancer
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though not explicitly stated in the note. | The note mentions 'localized pancreatic cancer', but does not specify a stage. |
| P2 | Current_Medications.current_meds | Current medications related to cancer treatment are missing. | The note states the patient is off all therapy for the past 5 months. |
| P2 | Treatment_Changes.recent_changes | Recent treatment changes are missing. | The note mentions the patient has been off all therapy for the past 5 months. |
| P2 | Treatment_Changes.supportive_meds | Supportive medications related to cancer treatment are missing. | The note does not mention any ongoing supportive medications. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the missing stage of cancer, current medications, and recent treatment changes.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 50 — ⚠️ ISSUES

**Type**: Locally advanced pancreatic cancer
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though not explicitly stated in the note. | Locally advanced pancreatic cancer |
| P2 | Current_Medications.current_meds | Missing cancer-related medications (gemcitabine and capecitabine). | chemotherapy with the ***** of gemcitabine and capecitabine |
| P2 | Treatment_Changes.recent_changes | Missing information about the completion of 2 full cycles of therapy. | She has now completed 2 full cycles of therapy |
| P2 | Treatment_Changes.supportive_meds | Missing supportive care medications, though none were specifically mentioned. | denies significant nausea, hand/foot syndrome, mucositis, or fevers related to treatment. |
| P2 | Lab_Plan.lab_plan | Incomplete lab plan, missing specific frequency for monitoring CA 19-9 levels. | We will continue to monitor her *****-9 levels |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing cancer-related medications, treatment changes, and incomplete lab plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 51 — ⚠️ ISSUES

**Type**: Pancreatic head adenocarcinoma
**Stage**: pT2N1 (originally Stage II)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Distant_Metastasis | The note does not provide a clear statement regarding distant metastasis, but the extracted data states 'Not sure'. This is slightly imprecise. | The note mentions hepatic lesions and pulmonary nodules but does not definitively state distant meta |
| P2 | Cancer_Diagnosis.Metastasis | The note does not provide a clear statement regarding metastasis, but the extracted data states 'Not sure'. This is slightly imprecise. | The note mentions hepatic lesions and pulmonary nodules but does not definitively state metastasis. |
| P2 | Current_Medications.current_meds | The field is empty, but the note does not explicitly mention that the patient is not currently taking any cancer-related medications. | The note does not specify ongoing cancer-related medications. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note does not explicitly mention that there were no recent changes in treatment. | The note does not specify recent changes in treatment. |
| P2 | Treatment_Changes.supportive_meds | The field is empty, but the note does not explicitly mention that there were no recent supportive medications. | The note does not specify recent supportive medications. |
| P2 | Medication_Plan.medication_plan | The note does not indicate that gemcitabine is being continued or started, but the extracted data suggests so. | The note indicates a plan for monitoring/expectant management without treatment. |
| P2 | Lab_Plan.lab_plan | The note specifies plans for repeat bloodwork, but the extracted data states 'No labs planned'. | The note mentions plans for repeat bloodwork in another ~3 months' time. |

*Extraction summary*: Most fields are clean, but there are minor issues related to the lack of clarity on metastasis, current medications, treatment changes, supportive medications, medication plan, and lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'soft tissue'. | Recent imaging shows a slight increase in the soft tissue near where your cancer |
| P2 | Unexplained medical jargon 'CA-19-9'. | Your CA-19-9 levels, which can indicate cancer activity, are higher than normal. |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 52 — ⚠️ ISSUES

**Type**: Adenocarcinoma of pancreatic origin
**Stage**: Metastatic adenocarcinoma to pancreas

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | Treatment with either gemcitabine monotherapy or modified ***** was recommended. |
| P2 | Treatment_Changes | Missing recent treatment changes. | Treatment with either gemcitabine monotherapy or modified ***** was recommended. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing cancer-related medications and recent treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | You will be offered a modified medication. |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 53 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane). | We then started gemcitabine and Abraxane. She's had one full cycle. |
| P2 | Treatment_Changes | Mentions increasing fentanyl patch but does not specify the current dose. | We will also increase her fentanyl patch to 125 mcg. |
| P2 | Response_Assessment | Does not mention the initial encouraging clinical response to therapy. | Locally advanced adenocarcinoma of the pancreas with encouraging clinical response to therapy |

*Extraction summary*: Major error in missing cancer-related medications and minor issues in treatment changes and response assessment.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | We increased your fentanyl patch to to better manage your pain. |

*Letter summary*: Letter contains an incomplete sentence that needs correction.

---

## ROW 54 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | He was on a chemotherapy holiday for 8 months, with serial imaging showing relatively stable disease |
| P2 | Treatment_Changes | Incomplete. Does not mention restarting chemotherapy. | On this basis, he restarted chemotherapy with ***** in October 2014 and received 9 cycles in total. |
| P2 | Treatment_Goals | Inaccurate. Should be 'palliative' but context suggests ongoing treatment decisions. | We discussed his various options at this point, not including resuming *****/***** which remain a vi |
| P2 | Response_Assessment | Inaccurate. Should include 'modest disease progression' and 'new pulmonary metastatic disease'. | We appreciate a persistent pancreatic tail mass, lower abdominal peritoneal metastases, and multiple |
| P2 | Medication_Plan | Incomplete. Does not mention ongoing consideration of clinical trial options. | After much consideration and discussion re: the above, including logistics and the randomized trial  |
| P2 | Therapy_plan | Incomplete. Does not mention ongoing consideration of clinical trial options. | After much consideration and discussion re: the above, including logistics and the randomized trial  |
| P2 | Procedure_Plan | Incomplete. Does not mention interventional radiology consult for biopsy. | We have gone ahead and placed an interventional radiology consult to have the IR team evaluate wheth |
| P2 | Imaging_Plan | Incorrect. Imaging was done post-visit. | Most recent imaging: CT C/A/P performed following today's visit, which was personally reviewed and i |

*Extraction summary*: Several fields are incomplete or slightly inaccurate, particularly regarding treatment changes and ongoing considerations for clinical trials.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'pulmonary nodules'. | Additionally, multiple new scattered pulmonary nodules sub-cm in size have appea |
| P2 | Unexplained jargon 'CRS-207', 'mesothelin-expressing Listeria vaccine', 'anti-PD1 mAb nivolumab'. | You are considering several clinical trial options, including a gemcitabine-base |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 55 — ⚠️ ISSUES

**Type**: Pancreatic head mass with invasive adenocarcinoma; Ampullary mass with invasive 
**Stage**: Stage IV (metastatic disease to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note mentions 'pancreatic head mass with invasive adenocarcinoma' and 'ampullary mass with invasive adenocarcinoma', but does not clearly specify which is the primary site. The extracted data should reflect this uncertainty. | Overall no consensus about whether ampullary, biliary, or pancreatic. |
| P2 | Treatment_Changes.recent_changes | The note indicates that oxaliplatin was discontinued due to worsening neuropathy and switched to capecitabine on 05/03/21. The extracted data should include the date of the switch. | Oxaliplatin was discontinued with C10 due to worsening neuropathy. Switched to capecitabine on 05/03 |
| P2 | Treatment_Goals.goals_of_treatment | The note suggests that the treatment goal is palliative, but the extracted data should explicitly state this. | Given his fairly rapid pace of progression despite 5FU, limited therapeutic options, I recommended a |
| P2 | Medication_Plan.medication_plan | The note mentions that the patient is currently on capecitabine, but the extracted data includes 'gemcitabine, abraxane' which are not current medications. | Switched to capecitabine on 05/03/21. |
| P2 | Therapy_plan.therapy_plan | The note mentions stopping capecitabine and switching to UCSF 500 on liver biopsy, but the extracted data should clarify that UCSF 500 is a placeholder for a specific treatment regimen. | Future consideration of revisiting gemcitabine without using abraxane again due to neuropathy. |

*Extraction summary*: Most fields are accurate, but there are minor issues with the clarity of the cancer diagnosis, medication changes, treatment goals, and therapy plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 56 — ⚠️ ISSUES

**Type**: Pancreas adenocarcinoma
**Stage**: (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (Nivolumab, Abraxane, Gemcitabine). | She is currently on a clinical trial involving CD40 agonistic monoclonal antibody, gemcitabine, and  |
| P2 | Treatment_Changes | Supportive medications listed are incorrect (ondansetron, prochlorperazine). Should include only those related to cancer treatment. | Well managed with current ***** ***** |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' but the note suggests ongoing treatment for metastatic disease. | Ok to proceed with cycle 5 day 8 at unchanged doses |
| P2 | Medication_Plan | Missing specific cancer-related medications (Nivolumab, Abraxane, Gemcitabine). | Ok to proceed with cycle 5 day 8 at unchanged doses |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and some inaccuracies in supportive medications and treatment goals.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 57 — ⚠️ ISSUES

**Type**: Node-positive pancreatic cancer
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine). | She has since resumed chemotherapy with single-agent gemcitabine, and has now completed 3 more cycle |
| P2 | Treatment_Changes | Missing mention of resuming single-agent gemcitabine. | She has since resumed chemotherapy with single-agent gemcitabine, and has now completed 3 more cycle |
| P2 | Therapy_plan | Incorrectly includes capecitabine, which the patient could not tolerate. | However, a week after starting she developed fevers, diarrhea, stomatitis, N/V, fatigue, and abdomin |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing current cancer-related medication and incorrect inclusion of capecitabine in the therapy plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 58 — ⚠️ ISSUES

**Type**: Primary pancreatic neuroendocrine tumor
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (Sandostatin). | June 2009: Sandostatin ***** increased to 30 mg |
| P2 | Treatment_Changes | Missing historical change in Sandostatin dosage. | June 2009: Sandostatin ***** increased to 30 mg |
| P2 | Treatment_Goals | Goals should be 'surveillance' rather than 'palliative'. | Given his stable disease currently and lack of symptoms related to malignancy, as well as his desire |
| P2 | Response_Assessment | Inaccurate description of response. Should mention slight decrease in lesion size. | Continued interval decrease in size of enhancing hepatic lesions, including 2.2 cm in the lateral se |
| P2 | Procedure_Plan | Missing potential endoscopy discussion. | Regarding his abdominal heaviness, while his symptoms are subtle, they are new for him. We will cont |

*Extraction summary*: Most fields are accurate, but there are minor omissions and inaccuracies in medication history, treatment goals, response assessment, and procedure planning.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'neuroendocrine tumor' might be confusing for an 8th-grade reading level. | You came in for a follow-up visit regarding your pancreatic neuroendocrine tumor |
| P2 | The term 'lymph nodes' might be confusing for an 8th-grade reading level. | Some lymph nodes near your stomach are unchanged. |
| P2 | The term 'stable' might be confusing for an 8th-grade reading level. | Overall, your disease is stable. |

*Letter summary*: Letter has minor readability issues that could be improved for an 8th-grade reading level.

---

## ROW 59 — ⚠️ ISSUES

**Type**: Well to moderately differentiated adenocarcinoma of the pancreas; Mucinous adeno
**Stage**: Likely advanced stage (likely or higher) due to positive retroperitoneal margin 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions several cancer-related medications such as propylthiouracil (PTU) for hyperthyroidism. | propylthiouracil (PTU) 50 mg tablet Take 25 mg by mouth. |
| P2 | Treatment_Changes | The field is empty, but the note mentions changes in medication dosages, such as lisinopril and omeprazole. | lisinopril (PRINIVIL,ZESTRIL) 40 mg tablet Take 20 mg (one half tablet) by mouth daily. (Patient tak |
| P2 | Imaging_Plan | The field only mentions 'CT Chest', but the note suggests ongoing surveillance which may include other imaging modalities. | She will continue on surveillance. We'll see her again in 6 months. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, treatment changes, and incomplete imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The sentence is vague and does not specify the type of cancer. | You came in for a follow-up visit regarding your cancer treatment. |
| P2 | The term 'groundglass nodule' may be confusing to a layperson. | A 6mm groundglass nodule in the left upper lobe of your lung remains unchanged s |

*Letter summary*: Letter is mostly clean but requires minor clarifications for accuracy and readability.

---

## ROW 60 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: Locally advanced; no distant metastases mentioned

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications such as growth factor used for severe neutropenia. | He later experienced severe neutropenia and was started on growth factor. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete. Missing growth factor used for severe neutropenia. | He later experienced severe neutropenia and was started on growth factor. |
| P2 | Reason_for_Visit | Summary is slightly imprecise. It should mention the discussion about proceeding to surgery after completing chemotherapy. | We will discuss proceeding to surgery as the next step with Dr. *****. |
| P2 | Response_Assessment | The response assessment mentions stable disease but does not explicitly state that the patient has completed 8 cycles of chemotherapy. | He has now had 8 cycles. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the completeness of cancer-related medications and the preciseness of the summary and response assessment.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'adenocarcinoma' is used, which might be too technical for an 8th-grade reading level. | You have localized pancreatic adenocarcinoma (cancer that started in gland cells |
| P2 | The term 'dilation' might be too technical for an 8th-grade reading level. | The CT scan shows increased dilation of the pancreatic duct upstream to the ill- |

*Letter summary*: Letter contains minor readability issues with technical terms that could be simplified further.

---

## ROW 61 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Stage II B (pT3N1)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and [REDACTED]-paclitaxel). | followed by adjuvant chemotherapy consisting of the combination of gemcitabine plus *****-paclitaxel |
| P2 | Treatment_Changes | Supportive medication (Neupogen) is mentioned, but the specific cancer-related medications (gemcitabine and [REDACTED]-paclitaxel) are not listed. | due to recurrent asymptomatic cytopenias, he has had to be dose reduced to dose level -2 of both che |
| P2 | Treatment_Goals | The goal 'surveillance' is correct, but 'adjuvant' might be more appropriate given the context of ongoing chemotherapy. | He has now completed 6 cycles of treatment as of the end of December. |
| P2 | Medication_Plan | Only mentions 'gemcitabine', but [REDACTED]-paclitaxel is also part of the regimen. | adjuvant chemotherapy consisting of the combination of gemcitabine plus *****-paclitaxel |
| P2 | Therapy_plan | Only mentions 'gemcitabine', but [REDACTED]-paclitaxel is also part of the regimen. | adjuvant chemotherapy consisting of the combination of gemcitabine plus *****-paclitaxel |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and slight imprecision in treatment goals and plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The sentence omits the mention of the non-specific pulmonary findings being likely related to a mild drug-related pneumonitis. | Recent imaging shows no signs of the cancer coming back or spreading. |
| P2 | The sentence mentions INR without explaining what it is. | Your blood tests show a slightly low hemoglobin and hematocrit, and an elevated  |

*Letter summary*: The letter is mostly clean but lacks detail on the pulmonary findings and could benefit from clarifying INR.

---

## ROW 62 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: Originally pT3 N1, now metastatic (pt3 n1)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'Continuing abraxane.' | We will cont ***** ***** abraxane. |
| P2 | Treatment_Changes | The field only mentions starting treatment, but the note also states continuing abraxane. | We will cont ***** ***** abraxane. |
| P2 | Lab_Plan | The field states 'No labs planned,' but the note does not explicitly state no future labs are planned. | Labs including the CBC and chemistry profile were reviewed per protocol. |

*Extraction summary*: Most fields are clean, but there are minor issues with medication and lab plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 63 — ⚠️ ISSUES

**Type**: Undifferentiated carcinoma of the tail of the pancreas involving the portal vein
**Stage**: Originally not specified, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) | She has now completed two full cycles of therapy. |
| P2 | Treatment_Changes | Supportive medications missing gemcitabine and Abraxane | She has now completed two full cycles of therapy. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the inclusion of cancer-related medications and supportive treatments.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'tumors'. | The cancer is showing a mixed response to treatment. Some liver tumors have grow |
| P2 | Unexplained medical jargon 'ondansetron', 'hydrocodone-acetaminophen'. | You are currently taking ondansetron (Zofran) for nausea and hydrocodone-acetami |

*Letter summary*: Letter is mostly clean but contains minor readability issues with unexplained medical jargon.

---

## ROW 64 — ⚠️ ISSUES

**Type**: Invasive adenocarcinoma, moderately differentiated, with lymphovascular invasion
**Stage**: Originally pT3N0, now metastatic (pt3n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine). | 02/08/18: Adjuvant Cycle 1 day 1 gemcitabine +***** paclitaxel. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete. | ondansetron (ZOFRAN) 8 mg tablet, prochlorperazine (COMPAZIN) |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative', but the note suggests discussion of multiple treatment options. | Today we discussed treatment options which include 5-***** based therapy |
| P2 | Response_Assessment | Redundant information, already covered in Clinical Findings. | The cancer is showing signs of recurrence and progression. |
| P2 | Medication_Plan | Incomplete, does not specify the exact medication. | The patient will start 5-[REDACTED] based therapy. |
| P2 | Therapy_plan | Vague, should be more specific about the plan. | The plan includes addressing the side effects and utility of the treatment. |
| P2 | follow_up_next_visit | Should be specified as 'not specified in the note'. | Next clinic visit: Not specified in the given note |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and specificity in several fields.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence, missing critical information about the stage of cancer. | Recent imaging and biopsy confirm that the cancer is showing signs of recurrence |
| P2 | Unexplained medical jargon (CA 19-9). | Your blood tests show some abnormalities, including low hemoglobin and hematocri |

*Letter summary*: Letter contains minor issues that need correction for clarity and completeness.

---

## ROW 65 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | The note mentions 'prophylaxis with growth factor support (*****) on day 3 of each cycle.' This shou |
| P2 | Treatment_Changes | Incomplete listing of supportive medications. | The note mentions 'IV fluids and dexamethasone on day 3,' but only dexamethasone is listed. |

*Extraction summary*: Most fields are clean, but there are minor omissions in cancer-related medications and supportive medications.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 66 — ⚠️ ISSUES

**Type**: Metastatic pancreatic cancer, BRCA2-associated
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete - only mentions 'gemcitabine', but the note also mentions '[REDACTED]-paclitaxel'. | He has now completed 3 full cycles of treatment. ... we elected to switch him to the combination of  |

*Extraction summary*: One minor issue identified in Current_Medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'tumor'. | Recent imaging shows a mixed response to your treatment. The size of the pancrea |
| P2 | Unexplained medical jargon 'agent-paclitaxel'. | You were switched to a new combination of chemotherapy drugs called gemcitabine  |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to medical jargon.

---

## ROW 67 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Clinical Stage IIB

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Lab_Results | Missing critical lab results such as CA 19-9, which is relevant to the patient's condition. | I recommended that he start using the levothyroxine. So, I recommended that he start using the levot |
| P2 | Clinical_Findings | Inaccurate measurement of the pancreatic head mass. The note mentions 'approximately 2.5 x 2.3 cm', while the extracted data states '2.5 x 2.3 cm'. | Pancreatic head mass measuring approximately 2.5 x 2.3 cm with ill-defined borders, similar in size  |
| P2 | Current_Medications | Empty field despite the note mentioning ongoing treatment with Eliquis for SMV thrombus. | Currently on Eliquis for SMV thrombus. |
| P2 | Lab_Plan | Incorrectly states 'No labs planned.' when the note indicates obtaining baseline labs including CA 19-9 on the first cycle of chemotherapy. | I recommended obtaining baseline labs including CA 19-9 on the first cycle of chemotherapy. |

*Extraction summary*: Major issues with missing critical lab results and minor inaccuracies in measurements and plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'adenocarcinoma'. | You have pancreatic adenocarcinoma (cancer that started in gland cells), which i |
| P2 | Technical term 'superior mesenteric vein' is unexplained. | No definite involvement of the superior mesenteric vein was found. |
| P2 | Unexplained jargon 'hypothyroidism'. | It was recommended to start Metformin for your diabetes and levothyroxine for yo |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

---

## ROW 68 — ⚠️ ISSUES

**Type**: Adenocarcinoma, pancreaticobiliary type, grade 2, moderately differentiated
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage 'now metastatic ()' is imprecise and lacks clarity. It should specify the current stage or state of the disease. | He now shows evidence of metastatic recurrence involving the abdominal wall, possibly representing s |
| P2 | Response_Assessment.response_assessment | The response assessment mentions that there is no specific imaging, tumor marker, or physical exam findings to assess the current response to treatment, but it incorrectly states that the response to treatment is not mentioned in the note. The note does mention the initiation of chemotherapy. | The patient has initiated chemotherapy with the combination of gemcitabine plus cisplatin on 05/27/2 |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage of cancer and the response assessment.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'retroperitoneal lymph node'. | You have evidence of metastatic recurrence involving the abdominal wall and a ri |
| P2 | Unexplained medical jargon 'Cancer Antigen 19-9'. | Your Cancer Antigen 19-9 level is elevated at 981. |
| P2 | Unexplained medical jargon 'hypermetabolic', 'soft tissue implants', 'conglomerates'. | Imaging shows multiple hypermetabolic soft tissue implants and conglomerates alo |

*Letter summary*: Letter is mostly clean but contains some unexplained medical jargon that could be clarified for better readability.

---

## ROW 69 — ⚠️ ISSUES

**Type**: Adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | This includes the combination of gemcitabine plus [REDACTED]-paclitaxel on a standard three-week-on, |
| P2 | Treatment_Changes | Missing cancer-related medication '[REDACTED]-paclitaxel'. | This includes the combination of gemcitabine plus [REDACTED]-paclitaxel on a standard three-week-on, |
| P2 | Treatment_Goals | The goal 'palliative' may be too broad given the context of a clinical trial. | The patient is currently receiving treatment on the [REDACTED]-002 chemotherapy/immunotherapy clinic |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and the specificity of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (hemoglobin, hematocrit, albumin, sodium). | You have metastatic pancreatic cancer with recent lab results showing high white |
| P2 | Unexplained jargon (Carcinoembryonic Antigen). | Your Carcinoembryonic Antigen level is also high. |
| P2 | Unexplained jargon (tumor biopsy). | You will undergo a repeat tumor biopsy during cycle #2 for research purposes. |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 70 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma with direct invasion into multiple structures, 
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine plus [REDACTED]-paclitaxel) | He has remained on treatment to the present time. ... He has demonstrated a nice CA-19-9 biomarker d |
| P2 | Treatment_Changes | Missing recent treatment changes (chemotherapy holiday for the next few months) | At this point, while indefinite treatment with gemcitabine/[REDACTED]-paclitaxel (or conceivably jus |

*Extraction summary*: Two minor issues identified regarding missing current cancer-related medication and recent treatment changes. All other fields are clean.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 71 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is left blank, though the note suggests metastatic disease. | metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Distant_Metastasis | Distant Metastasis is listed as 'Not sure', though the note indicates metastatic disease. | metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Metastasis | Metastasis is listed as 'Not sure', though the note indicates metastatic disease. | metastatic pancreatic adenocarcinoma |
| P2 | Current_Medications.current_meds | Current medications related to cancer treatment are missing (gemcitabine). | C1D1 Gemcitabine single-agent |
| P2 | Treatment_Changes.recent_changes | Recent changes in treatment are missing (gemcitabine). | C1D1 Gemcitabine single-agent |

*Extraction summary*: Most fields are clean, but there are minor issues with the staging and missing cancer-related medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Contains technical jargon that may be difficult for an 8th-grade reader to understand. | You have a 25.3 X 22.4mm round, hypoechoic, calcified, and solid mass in the hea |
| P1 | Contains technical jargon that may be difficult for an 8th-grade reader to understand. | There is also a 1 cm hypodense lesion within hepatic segment 2, and three develo |
| P1 | Incomplete sentence with missing critical information. | You will continue taking levothyroxine and lovenox /kg BID. |
| P1 | Incomplete sentence with missing critical information. | If you tolerate the current treatment, a medication-paclitaxel will be added wit |

*Letter summary*: The letter contains several sentences with technical jargon and incomplete information that need to be addressed.

---

## ROW 72 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma of the pancreas
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage description 'Originally borderline resectable, now metastatic ()' is slightly imprecise. It should specify the current stage as 'metastatic' without the original stage. | He had an episode of bursitis in his L elbow which is now resolved.     He  completed 4 cycles of th |
| P2 | Treatment_Changes.recent_changes | The recent changes mention switching to reduced dose [REDACTED], but the exact name of the medication is redacted. This could be clearer. | We then elected to resume chemotherapy but switched to reduced dose [REDACTED]. He did not tolerate  |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage description and clarity of recent treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Contains medical jargon that may be difficult for an 8th-grade reader to understand. | Your cancer is currently showing stable disease in the abdomen but progressing p |
| P2 | Mentions 'chemotherapy holiday' which may be confusing. | We decided to resume chemotherapy but switched to a reduced dose of a medication |

*Letter summary*: Letter contains minor readability issues with some medical jargon that could be simplified further.

---

## ROW 73 — ⚠️ ISSUES

**Type**: Adenocarcinoma of pancreatic origin
**Stage**: Stage IB

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'metformin 500mg daily' for diabetes, which is not a cancer-related medication. | On metformin 500mg daily |
| P2 | Imaging_Plan | The imaging plan is listed as 'Ultrasound', but the note suggests MRI/MRCP or endoscopic ultrasound for further workup. | If further imaging workup is desired, MRI/MRCP could be considered. Alternatively, this could be fur |

*Extraction summary*: Most fields are clean, but there are minor issues with the imaging plan and the current medications field.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'gland cells'. | A type of cancer that started in gland cells was identified in a mass in the pan |
| P2 | Unexplained jargon 'superior mesenteric vein'. | Imaging showed moderate dilation of the pancreatic ducts with slight contour fla |
| P2 | Unexplained jargon 'abutment' and 'superior mesenteric vein'. | Consolidative radiation therapy is being considered due to the abutment of the s |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

---

## ROW 74 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing cancer-related medication (rucaparib). | She continues on rucaparib. |

*Extraction summary*: One major error in missing a cancer-related medication. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have metastatic (Stage IV) cancer that has spread to your peritoneum and liv |
| P2 | Unexplained medical jargon. | The cancer is currently in excellent control, with a reduction in the size of th |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 75 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma arising in association with an intraduc
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |
| P2 | Treatment_Changes | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |
| P2 | Medication_Plan | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |
| P2 | Therapy_plan | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |

*Extraction summary*: Most fields are clean, but there are minor omissions related to the inclusion of 'filgrastim-sndz' (Neupogen) as a cancer-related supportive medication.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (dose amount). | Your gemcitabine dose was reduced to on days 1 and 8 due to low ANC. Neupogen fo |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 76 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (Atezolizumab + [REDACTED]) | Patient consented to clinical trial *****# ***** ***** trial and she was randomized to the Atezolizu |

*Extraction summary*: One minor issue identified: missing cancer-related medication in Current_Medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'nodules'. | The CT scan on 10/29/18 shows that the cancer has progressed. There is an increa |
| P2 | Incomplete sentence with missing critical information. | Your C1D1 Atezolizumab + a medication treatment was delayed due to high lab valu |

*Letter summary*: Letter contains minor readability and completeness issues.

---

## ROW 77 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the tail of the pancreas
**Stage**: Resectable at diagnosis, now with negative margins and lymph nodes

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary of the visit reason. It mentions 'follow-up visit for adjuvant therapy with gemcitabine and capecitabine', but the patient has already completed 6 cycles of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage description is imprecise. It should specify the stage at diagnosis rather than the current status. | Because of renal and splenic invasion, the left kidney and left adrenal and spleen were included in  |
| P2 | Treatment_Changes.recent_changes | Missing recent treatment changes. The note indicates the patient has completed 6 cycles of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Response_Assessment.response_assessment | The response assessment does not mention the completion of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Procedure_Plan.procedure_plan | The referral to hepatology is mentioned under Procedure Plan, which is incorrect. It should be under Referral. | We will refer him to hepatology for an evaluation. |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary of the visit reason, stage description, treatment changes, response assessment, and procedure plan classification.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 78 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing relevant cancer-related medication (risedronate). | risedronate (ACTONEL) 150 mg tablet Take 150 mg by mouth every 30 (thirty) days. |

*Extraction summary*: One minor issue identified regarding the inclusion of risedronate under cancer-related medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon (splenic artery, splenic vein) | You have a 3.7 x 3.6 cm mass in the body of your pancreas that encases the splen |
| P2 | Unexplained medical jargon (metastases) | Multiple liver metastases are also present. |
| P2 | Unexplained medical jargon (CK 7, CK 20) | A fine needle aspiration (FNA) of one liver lesion shows metastatic adenocarcino |
| P2 | Unexplained medical jargon (icterus) | On physical examination, you appear frail with possible slight icterus. |
| P2 | Unexplained medical jargon (hepatosplenomegaly, fluid wave) | Your abdomen is soft and non-tender with no hepatosplenomegaly or fluid wave. |
| P2 | Unexplained medical jargon (alkaline phosphatase, bilirubin) | Laboratory results show elevated alkaline phosphatase and total bilirubin levels |

*Letter summary*: The letter contains several instances of unexplained medical jargon that may be confusing to the patient.

---

## ROW 79 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma with metastatic disease to liver
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | current_meds | The field is empty, but the note mentions 'gemcitabine, irinotecan, oxaliplatin, leucovorin'. | phase III evidence supports either ***** (biweekly ***** 5-*****, leucovorin, irinotecan, and oxalip |
| P2 | Treatment_Goals | The field states 'palliative', but the note suggests a goal of achieving a deep and durable remission. | the mainstay of treatment at this juncture should consist of systemic therapy with the goal of achie |
| P2 | Lab_Results | The field includes 'Hemoglobin 14.0, Hematocrit 41.0, Platelet Count 166, Int'l Normaliz Ratio 1.0', which are not mentioned in the note. | Most recent labs notable for the following: Lab Results Component Value Date WBC Count 5.8 05/02/202 |

*Extraction summary*: Most fields are clean, but there are minor issues with current medications, treatment goals, and lab results.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'adenocarcinoma' is used without further explanation, which might be confusing for an 8th-grade reading level. | You have been diagnosed with pancreatic adenocarcinoma (cancer that started in g |
| P2 | While 'CA 19-9' is not flagged for explanation, the term 'tumor marker' might still be too technical for an 8th-grade reading level. | Your CA 19-9 (a tumor marker) level is elevated at 852. |
| P2 | The list of drug names might be overwhelming and confusing for an 8th-grade reading level. | You will start a combination of chemotherapy drugs called a chemotherapy combina |
| P2 | The term 'MRI study' might be too technical for an 8th-grade reading level. | An MRI study will be conducted before starting treatment. |
| P2 | Terms like 'germline testing', 'hereditary predisposition', and 'molecular testing' might be too technical for an 8th-grade reading level. | Germline testing will be done to assess for any hereditary predisposition to can |

*Letter summary*: The letter contains minor readability issues that could be improved for an 8th-grade reading level.

---

## ROW 80 — ⚠️ ISSUES

**Type**: Poorly differentiated pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Slightly imprecise wording. The summary should mention the plan for a diagnostic biopsy. | We plan to do a diagnostic biopsy to better understand this. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Inaccurate stage description. 'Metastatic ()' is unclear and should be 'Metastatic IV'. | Metastatic adenocarcinoma of the pancreas with radiographic evidence for control of pre-existing les |
| P2 | Treatment_Changes.recent_changes | Missing information about the recent change in medication dosage. | f/u after 20% reduction in capecitabine dose. |
| P2 | Imaging_Plan.imaging_plan | Incorrect temporal classification. The patient is returning for a follow-up CT scan evaluation. | He is returning today for a followup CT scan evaluation. |
| P2 | Procedure_Plan.procedure_plan | Missing information about the planned diagnostic biopsy. | We plan to do a diagnostic biopsy to better understand this. |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, stage description, recent changes, imaging plan, and procedure plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Use of 'metastasis' and 'areas' may be slightly confusing without context. | There is an enlarging liver metastasis, now measuring up to 2.4 cm compared to 1 |
| P2 | This sentence is a bit vague and could be clearer. | The cancer is showing signs of disease progression. |
| P2 | This sentence lacks clarity on what the study medication is. | You will be taken off the current study medication due to disease progression. |

*Letter summary*: The letter contains minor readability issues that could be improved for better clarity.

---

## ROW 81 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary. The note mentions a second opinion scheduled with Dr. ***** ***** on 12/14/18. | Patient has appt with Dr. ***** ***** at ***** on 12/14/18 for a second opinion. |
| P2 | Imaging_Plan.imaging_plan | Incorrect imaging plan. The note does not mention a DEXA scan. | Recent Imaging None |

*Extraction summary*: Most fields are accurate, but there are minor issues with the summary and imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information about the medications. | You started a new medication on 12/09/18. You will continue a medication and a m |

*Letter summary*: Letter contains an incomplete sentence with missing critical information about medications.

---

## ROW 82 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Stage IV (metastatic to liver and lung)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication 'abraxane'. | Started ***** abraxane C1D1 08/03/18 |

*Extraction summary*: One minor issue found in Current_Medications.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 83 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane). | He elected to start therapy with gemcitabine and Abraxane. He has now completed 2 cycles. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete. | He is also developed generalized edema and peeling of his skin as well as mouth sores which are resp |
| P2 | Treatment_Goals | Goals of treatment should be more specific (palliative with consideration of pembrolizumab). | I think we need to hold his chemotherapy to see if the symptoms resolve... He is eligible for pembro |
| P2 | Response_Assessment | Missing detail about the patient's self-started prednisone. | He has a history of asthma and felt that he was having problems from mount. He self started predniso |
| P2 | Medication_Plan | Missing detail about the patient's self-started prednisone. | He has a history of asthma and felt that he was having problems from mount. He self started predniso |
| P2 | Therapy_plan | Missing detail about the patient's self-started prednisone. | He has a history of asthma and felt that he was having problems from mount. He self started predniso |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications and incomplete supportive medication details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon (white blood cell count). | Your white blood cell count is elevated, possibly due to prednisone use. |
| P2 | Slightly imprecise wording; 'generalized skin peeling' might be confusing. | You are weak with generalized skin peeling. |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to medical jargon and slightly imprecise wording.

---

## ROW 84 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (acyclovir). | Medication Sig acyclovir (ZOVIRAX) 400 mg tablet |
| P2 | Treatment_Changes | Missing supportive medication (PRBCs). | Due to persistent (rx-related) anemia on screening labs, he received 2u PRBCs at our infusion center |
| P2 | Lab_Plan | Should include regular monitoring of lab results. | Review of his CT scans show essentially stable to slightly enlarging sites of his measurable lesions |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing current cancer-related medications and supportive treatments, and lack of regular lab monitoring.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 85 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions valACYclovir (VALTREX) being prescribed. | valACYclovir (VALTREX) 1 g tablet Take 1,000 mg by mouth every 12 (twelve) hours. |

*Extraction summary*: One minor issue identified in Current_Medications. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'dilated pancreatic ducts'. | You have a 3.3 x 2.9 x 3.3 cm mass in the head of your pancreas with dilated pan |
| P2 | Unexplained medical jargon 'compressing major blood vessels'. | Some of these liver lesions have signs of internal bleeding and are compressing  |
| P2 | Unexplained medical jargon 'clinical trial', 'gemcitabine', 'paclitaxel'. | You have also signed up for a clinical trial that evaluates the combination of g |
| P2 | Unexplained medical jargon 'Mediport', 'growth factor support'. | If you choose a medication, you will need a Mediport for treatment and will rece |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 86 — ⚠️ ISSUES

**Type**: Advanced intrahepatic cholangiocarcinoma; Pancreatic neuroendocrine tumor (PNET)
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete listing of cancer-related medications. Only anastrozole is listed, but the note mentions several other relevant medications such as GM-CSF/pembrolizumab. | She previously received gemcitabine/cisplatin and ***** back in 2013-14, and more recently was enrol |
| P2 | Treatment_Changes | Incomplete listing of supportive medications. Only ondansetron and prochlorperazine are mentioned, but lorazepam and zolpidem are also used for nausea and insomnia. | LORazepam (ATIVAN) 0.5 mg tablet TAKE 1 TABLET BY MOUTH SUBLINGUALLY EVERY 8 HOURS AS NEEDED FOR NAU |
| P2 | Therapy_plan | Inaccurate description of therapy plan. 'Compression stockings' are incorrectly included. | Once liver function tests (LFTs) are trending downward post-procedure, the patient will start chemot |

*Extraction summary*: Most fields are clean, but there are minor issues with incomplete listings of cancer-related medications and supportive medications, and an incorrect inclusion of 'compression stockings' in the therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'phase II trial'. | You came in for a follow-up visit regarding your advanced intrahepatic cholangio |
| P2 | Unexplained medical jargon 'nodules'. | New tiny nodules were found in the right lung. |
| P2 | Unexplained medical jargon 'paracentesis', 'thoracentesis'. | You will have repeat paracentesis and possibly thoracentesis to drain fluid. |
| P2 | Unexplained medical jargon 'gastroenterologist', 'bile duct blockage'. | You will be referred urgently to a gastroenterologist for a procedure to relieve |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 87 — ⚠️ ISSUES

**Type**: Pancreatic neuroendocrine tumor, well-differentiated
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing 'octreotide', which is a cancer-related medication. | octreotide (SANDOSTATIN *****) 30 mg every 30 (thirty) days. |
| P2 | Treatment_Changes | Supportive medications listed incorrectly include 'nystatin (MYCOSTATIN) ointment', which is not cancer-related. | nystatin (MYCOSTATIN) ointment APPLY ON THE SKIN TWICE A DAY UNTIL ***** ***** RED. |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' but the note suggests 'possible slow PD', indicating a more complex situation. | Possible slow PD |
| P2 | Response_Assessment | The response assessment mentions 'some minimal progression' but does not specify the exact nature of the progression. | These findings suggest a slow progression despite ongoing treatment with everolimus. |
| P2 | Medication_Plan | Mentions 'metformin' which is not a cancer-related medication. | Continue Everolimus 10 mg daily. Continue octreotide. If progression, could consider temozolomide, [ |
| P2 | Procedure_Plan | Includes 'continue monitoring triglycerides quarterly' which is not a procedure plan. | Liver directed therapy might be an option, follow up MMA and B12, continue monitoring triglycerides  |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, incorrect classification of supportive medications, and slight imprecision in treatment goals and response assessment.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 88 — ⚠️ ISSUES

**Type**: Locally advanced adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary mentions 'consider treatment options following recent imaging and symptom evaluation,' but the note does not explicitly mention recent imaging. | No recent imaging reviewed |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage of cancer is left blank, though the note provides some information about the extent of the disease. | Infiltrative mass in the distal pancreas, involving arteries and veins, and stable disease noted on  |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions ongoing chemotherapy with Abraxane and another unspecified agent. | Started on 2nd line therapy with [REDACTED] on 02/28/19 after disease progression following 4 cycles |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note indicates a change in treatment due to disease progression. | After disease progression following 4 cycles of [REDACTED]/Abraxane, started on 2nd line therapy wit |
| P2 | Medication_Plan.medication_plan | The field mentions continuing [REDACTED], but it should also include Abraxane. | Started on 2nd line therapy with [REDACTED] on 02/28/19 after disease progression following  4 cycle |

*Extraction summary*: Most fields are clean, but there are minor issues related to the completeness of treatment details and the inclusion of chemotherapy medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'body/tail'. | You have a pancreatic body/tail mass that is causing a blockage in your stomach  |
| P2 | Unexplained medical jargon 'albumin'. | Your albumin and calcium levels are low, and you have anemia with a hemoglobin l |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 89 — ⚠️ ISSUES

**Type**: Well to moderately differentiated ductal adenocarcinoma of pancreatic head with 
**Stage**: Not explicitly stated in note, but described as resected node-positive pancreati

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and nab-paclitaxel). | Switched to the combination of gemcitabine plus nab-paclitaxel and completed six 28 day cycles; |
| P2 | Treatment_Changes | Missing recent treatment change (switch to gemcitabine plus nab-paclitaxel). | Switched to the combination of gemcitabine plus nab-paclitaxel and completed six 28 day cycles; |
| P2 | Treatment_Goals | Inaccurate goal description. Should include 'adjuvant' along with 'surveillance'. | We discussed at length today his risk of disease relapse, including locoregional risk given the (+)  |
| P2 | Response_Assessment | Inaccurate description of response. Should mention locoregional risk due to positive margin. | He looks well at today's visit and has recovered well from last month's partial SBO. We discussed at |
| P2 | Medication_Plan | Incomplete. Should mention possible removal of Mediport and IVC filter. | At his request, we will also arrange for removal of his Mediport and possibly his IVC filter. |
| P2 | Procedure_Plan | Incomplete. Should mention possible removal of Mediport and IVC filter. | At his request, we will also arrange for removal of his Mediport and possibly his IVC filter. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, treatment changes, and incomplete descriptions of treatment goals and plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'perineural invasion'. | You have a well to moderately differentiated ductal adenocarcinoma (cancer that  |
| P2 | Lack of context about what 'positive' means in this case. | One out of twelve lymph nodes was positive. |
| P2 | Unexplained jargon 'radiosensitizer'. | You are considered a reasonable candidate for radiation treatment with concurren |
| P2 | Unexplained jargon 'Mediport' and 'IVC filter'. | We will also arrange for the removal of your Mediport and possibly your IVC filt |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 90 — ⚠️ ISSUES

**Type**: Primary pancreatic neuroendocrine tumor; BRAF-mutant metastatic melanoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medications (dabrafenib and trametinib). | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Treatment_Changes | Missing recent changes related to dabrafenib and trametinib. | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Treatment_Goals | Goals of treatment should include both palliative and symptom management. | Patient has multiple hyperattenuating liver lesions, two of which are new from 05/17/2015, suspiciou |
| P2 | Response_Assessment | Missing response assessment for dabrafenib and trametinib. | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Medication_Plan | Missing medication plan for dabrafenib and trametinib. | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Imaging_Plan | Missing imaging plan for ongoing monitoring of melanoma and pNET. | Patient has multiple hyperattenuating liver lesions, two of which are new from  05/17/2015, suspicio |

*Extraction summary*: Most fields are clean, but there are minor omissions related to dabrafenib and trametinib, and a lack of imaging plan for ongoing monitoring.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Your dose of octreotide was reduced for four days, then increased for four days, |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 91 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas, moderately differentiated
**Stage**: Resected with 8 of 23 lymph nodes positive

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary mentions 'potential recurrence' but does not specify the context of the current coronavirus crisis. | As long as he is asymptomatic, we will watch this as hope that it represents delayed post op changes |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage description is somewhat imprecise. It should include the resection status and lymph node involvement more clearly. | Resected with 8 of 23 lymph nodes positive |
| P2 | Treatment_Changes.recent_changes | There is no mention of recent changes in treatment, but the note suggests ongoing monitoring without immediate changes. | We'll see him again in 3 months for follow-up. |
| P2 | Response_Assessment.response_assessment | The response assessment does not fully capture the uncertainty regarding possible local recurrence due to the current situation. | He may well have local recurrence but with the current coronavirus crisis, it will be difficult to w |

*Extraction summary*: Most fields are accurate and complete, but there are minor issues with the summary and response assessment that could be more precise.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 92 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic adenocarcinoma with perineural and lymphova
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis | The extracted 'Type_of_Cancer' does not fully match the note. The note mentions 'moderately differentiated adenocarcinoma' but does not specify 'perineural and lymphovascular invasion'. | final pathology was notable for residual moderately differentiated adenocarcinoma with (+) marked tr |
| P2 | Lab_Results | The extracted 'lab_summary' includes redacted values for AST/ALT, which should be omitted. | WBC 14.3, Hct 41.2, plts 123, Creat 0.78, AST/ALT [REDACTED]/[REDACTED], tot bili 0.4, alk phos 115, |
| P2 | Current_Medications | The field is empty, but the note mentions ongoing chemotherapy with gemcitabine and [REDACTED]-paclitaxel. | she resumed chemotherapy with the combination of gemcitabine plus *****-paclitaxel |
| P2 | Treatment_Changes | The field is empty, but the note mentions a chemotherapy holiday and potential future treatments. | we will want to continue re-addressing the incorporation of a chemotherapy holiday at some point. |
| P2 | Treatment_Goals | The extracted 'goals_of_treatment' is 'palliative', but the note suggests a mix of palliative and surveillance goals. | assuming continued good disease control, we will want to continue re-addressing the incorporation of |
| P2 | Response_Assessment | The extracted 'response_assessment' is incomplete. It does not mention the stable disease status of the liver and lung lesions. | Postoperative scans have revealed no evidence of active disease in the liver and a slightly growing  |
| P2 | Medication_Plan | The extracted 'medication_plan' mentions trametinib, which is not part of the current plan but a future consideration. | Future considerations include incorporating a chemotherapy holiday at some point and exploring clini |
| P2 | Therapy_plan | The extracted 'therapy_plan' mentions trametinib, which is not part of the current plan but a future consideration. | Future considerations include incorporating a chemotherapy holiday at some point and exploring clini |
| P2 | Imaging_Plan | The extracted 'imaging_plan' is incomplete. It does not mention the specific timing of the next imaging, which is 2-3 months. | repeat imaging in another 2-3 months' time, unless her CA-119-9 markedly rises in the interim. |
| P2 | Lab_Plan | The extracted 'lab_plan' is incomplete. It does not mention the specific timing of the next lab tests, which is 2-3 months. | unless CA 19-9 markedly rises in the interim. |
| P2 | follow_up_next_visit | The extracted 'Next clinic visit' is incomplete. It does not mention the specific timing of the next visit, which is 2-3 months. | in-person: 2-3 months for repeat imaging |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and precision in several fields.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 93 — ⚠️ ISSUES

**Type**: Adenocarcinoma involving the pancreatic head; metastatic adenocarcinoma consiste
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing specific cancer-related medications. | She underwent an U/S-guided LN biopsy on 08/22/2020, which confirmed metastatic adenocarcinoma, cons |
| P2 | Treatment_Changes | Supportive medications listed are not fully accurate. | For pain management: continues with long-acting morphine which is providing good pain control. |
| P2 | Treatment_Goals | Goals of treatment should be more specific. | In summary, Ms. ***** is a 39 y.o. female with pancreatic cancer w/*****-***** *****, developed in t |
| P2 | Medication_Plan | Specific cancer-related medications are not detailed. | Continue chemotherapy with a reduced dose of irinotecan starting from cycle #2. Plan to switch to a  |
| P2 | Therapy_plan | Specific cancer-related medications are not detailed. | Holding current chemotherapy for the time being. Recommending a pivot to a gemcitabine-based regimen |

*Extraction summary*: Most fields are clean, but there are minor issues with missing specific cancer-related medications and incomplete supportive medication details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'hypodense', 'hepatic', and 'hyperemia'. | A new hypodense liver lesion in the right hepatic lobe with surrounding hyperemi |
| P2 | Unexplained jargon 'CA 19-9'. | Elevated CA 19-9 levels (456 U/mL on 11/21/2020) indicate that the cancer is sho |
| P2 | Unexplained jargon 'gemcitabine-based regimen', 'nab-paclitaxel'. | It was recommended to switch to a gemcitabine-based regimen, specifically the co |
| P2 | Inconsistent information. The original note mentions reducing the dose of irinotecan from cycle #2, but the letter states continuing chemotherapy with a reduced dose of irinotecan starting from cycle #2, which implies the dose was already reduced. | You will continue to receive chemotherapy with a reduced dose of irinotecan star |
| P2 | Inconsistent information. The original note specifies 'November 29' of each chemo cycle, but the letter lacks the month. | You will continue prophylactic minocycline on days November 29 of each chemo cyc |

*Letter summary*: The letter contains minor readability issues and inconsistencies that need clarification.

---

## ROW 94 — ⚠️ ISSUES

**Type**: Metastatic ductal adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it could be inferred as Stage IV due to metastatic disease. | Mr. ***** is a 71 y.o. gentleman with a prior history of prostate cancer and a current diagnosis of  |
| P2 | Current_Medications.current_meds | Current cancer-related medications are missing, such as gemcitabine and Abraxane. | He is being treated with gemcitabine and Abraxane. |
| P2 | Treatment_Changes.recent_changes | Recent treatment changes are missing, such as the chemotherapy holiday. | When we saw him in September, we elected to start a chemotherapy holiday. |
| P2 | Treatment_Goals.goals_of_treatment | Goals of treatment should include 'surveillance' but also mention the chemotherapy holiday. | When we saw him in September, we elected to start a chemotherapy holiday. |
| P2 | Response_Assessment.response_assessment | Response assessment mentions a continued decrease in lymph node size but does not specify the exact time frame or recent changes. | Continued interval decrease in size of the left paraaortic lymph node, measuring 1.5 x 2.4 (previous |
| P2 | Medication_Plan.medication_plan | Medication plan only mentions insulin for diabetes management but does not include cancer-related medications. | He is being treated with gemcitabine and Abraxane. |
| P2 | Imaging_Plan.imaging_plan | Imaging plan should indicate ongoing surveillance imaging. | He'll continue on surveillance. I'll see him again in 8 weeks for follow-up. |

*Extraction summary*: Most fields are clean, but there are several minor issues related to missing cancer-related medication details, treatment changes, and imaging plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'para-aortic lymph node'. | There has been a continued decrease in the size of the left para-aortic lymph no |
| P2 | Unexplained medical jargon 'thoracic metastases'. | No new thoracic metastases were detected. |
| P2 | Multiple unexplained medical jargon terms ('pericardial effusion', 'hydronephrosis', 'obstructive uropathy', 'palpable masses', 'jaundice', 'ascites'). | No pericardial effusion, no hydronephrosis, no obstructive uropathy, no suspicio |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 95 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma, probable second primary
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions the patient has been started on a medication. | She has been started on ***** and has received 4 cycles. |
| P2 | Procedure_Plan | The value 'Hopefully' is unclear and does not provide a clear plan. | Hopefully, he will be comfortable with attempting a surgical resection. |

*Extraction summary*: Most fields are clean, but there are minor issues with the current medications and procedure plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (new medication name not specified) | You started on a new medication and have received 4 cycles so far. |

*Letter summary*: Letter is mostly clean but requires clarification on the new medication.

---

## ROW 96 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma with peritoneal carcinomatosis
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine) | She will tentatively be scheduled to receive gemcitabine next week: -gemcitabine 1000 mg/m2 on days  |
| P2 | Treatment_Changes | Inaccurate description of recent treatment change; mentions 'ondansetron, methadone, hydrocodone' which are not cancer-related medications. | Switched to gemcitabine monotherapy from gemcitabine + capecitabine |
| P2 | Therapy_plan | Inconsistent with the note; mentions continuing capecitabine, which contradicts the plan to try gemcitabine monotherapy. | We decided that we would try gemcitabine monotherapy for one more dose and re-evaluate after that. |

*Extraction summary*: Most fields are accurate, but there are minor issues with current medications, treatment changes, and therapy plan descriptions.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (CA 19-9). | Your CA 19-9 (a tumor marker) level has increased to 1301, indicating that your  |
| P2 | Inaccurate dosing information. | You will receive gemcitabine every 7 days for the next few weeks. |

*Letter summary*: Letter contains minor readability and accuracy issues.

---

## ROW 97 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Locally advanced disease

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine). | We started her on an alternate week fixed dose rate gemcitabine with capecitabine. |
| P2 | Treatment_Changes | Missing mention of incomplete dosing with capecitabine during the first cycle. | However between the lip blisters and tenderness in her fingertips, she received incomplete dosing wi |

*Extraction summary*: Most fields are clean, but there are minor omissions regarding cancer-related medications and treatment changes.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 98 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and Abraxane) | We initially saw her about 2 months ago. We recommended therapy with gemcitabine and Abraxane. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete | She was having a lot of difficulty with nausea and vomiting, early satiety and low caloric intake. W |
| P2 | Lab_Plan | Incorrect classification, should be under Therapy_plan | Once we can get her albumin up to 2 or better, we can resume her chemotherapy. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and incomplete supportive medications. Additionally, the lab plan is incorrectly classified.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 99 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced (unresectable); cannot rule out pleural and peritoneal metastas

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'resumed reduced dose [REDACTED]', which should be included. | She resumed reduced dose *****. |
| P2 | Treatment_Changes | The field does not mention the initial chemotherapy regimen that was modified. | She was started on modified *****. She has had 3 cycles of therapy. |
| P2 | Treatment_Goals | The field states 'palliative', but the note suggests a broader goal of symptom control and potential future chemotherapy eligibility. |  |
| P2 | Response_Assessment | The field does not mention the local progression noted in the CT scan. | When we last saw her, her CT scan showed local progression only. |
| P2 | radiotherapy_plan | The field states 'The family is still questioning about whether radiation is the right route to go.' but does not include the scheduling details. | She is scheduled for placement of fiducials tomorrow. Her radiation is not scheduled to begin until  |

*Extraction summary*: Most fields are clean, but there are minor issues related to incomplete information in several fields.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (CA 19-9) | The cancer is showing stable disease based on the latest CT scan and CA 19-9 lev |
| P2 | Unexplained jargon (SUV) | Additionally, the PET/CT scan showed no significant change in the pancreatic nec |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 100 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Stage IV (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and Abraxane) that is being considered for initiation. | We are recommending a change in his management to gemcitabine and Abraxane. |
| P2 | Treatment_Changes | Supportive medications listed include oxyCODONE, which is not specifically cancer-related. | oxyCODONE (ROXICODONE) 5 mg tablet TK 1 T PO Q 6 H PRF MODERATE OR SEVERE PAIN |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication and inclusion of non-specific supportive medication.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'spread more'. | Imaging shows that the cancer in your pancreas has grown slightly since the last |
| P2 | Abraxane is a brand name; consider explaining it as 'a type of chemotherapy'. | We are recommending that you start a new treatment called gemcitabine and Abraxa |

*Letter summary*: Letter is mostly clean with minor readability and jargon issues.

---

