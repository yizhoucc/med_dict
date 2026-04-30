# Auto Review: results.txt

Generated: 2026-04-30 16:49
Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)

## Summary

- **Samples**: 100
- **Clean**: 0/100
- **P0** (hallucination): 0
- **P1** (major error): 48
- **P2** (minor issue): 544

### Critical Issues

- **ROW 3** [P1]: The note mentions both 'metastatic adenocarcinoma' and 'metastatic pancreatic adenocarcinoma', but the diagnosis is uncertain and the note suggests it might be metastatic adenocarcinoma of the colon. The extracted data incorrectly lists both diagnoses as definitive.
- **ROW 4** [P1]: Incorrect verb tense and voice inconsistency.
- **ROW 5** [P1]: Incomplete sentence with missing critical information.
- **ROW 7** [P1]: Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III.
- **ROW 9** [P1]: Incomplete sentence with missing critical information.
- **ROW 14** [P1]: Inaccurate description of the reason for the visit. The patient is being followed up for a high-grade neuroendocrine tumor of duodenal/ampullary origin, not just 'neuroendocrine tumor treatment'.
- **ROW 15** [P1]: The sentence is too complex and uses medical jargon that may be difficult for an 8th-grade reading level patient to understand.
- **ROW 17** [P1]: Incomplete sentence with missing critical information about the specific treatment plan.
- **ROW 18** [P1]: Incorrect goal of treatment stated as 'palliative'.
- **ROW 22** [P1]: Incomplete sentence, missing critical information about the hand-foot rash.
- **ROW 24** [P1]: Incomplete sentence. It mentions 'involvement of 27 lymph nodes' but doesn't specify whether this is positive or negative.
- **ROW 25** [P1]: Incomplete sentence with missing critical information (missing the new dose amount).
- **ROW 26** [P1]: The lab results section contains fabricated information not present in the note.
- **ROW 28** [P1]: Inaccurate description of the cancer involvement. The original note mentions 'primary pancreatic head adenocarcinoma with metastatic gallbladder/liver involvement; vs a multifocal biliary tract/GB cancer'.
- **ROW 33** [P1]: The lab results listed in the extracted data do not match the lab results mentioned in the note.
- **ROW 33** [P1]: The field is empty, but the note mentions several cancer-related medications being used.
- **ROW 33** [P1]: Incomplete sentence with missing critical information (dose reduction of irinotecan).
- **ROW 34** [P1]: Incomplete sentence, missing the name of the medication.
- **ROW 36** [P1]: Incomplete sentence with missing critical information.
- **ROW 38** [P1]: Incomplete sentence with missing critical information.
- **ROW 40** [P1]: Incomplete sentence with missing critical information (lacks the specific dose increase).
- **ROW 43** [P1]: Incorrect goal of treatment.
- **ROW 43** [P1]: Inaccurate description of the reason for the visit. The patient is being treated for borderline resectable pancreatic adenocarcinoma, not general pancreatic cancer.
- **ROW 45** [P1]: The phrase 'cancer that started in gland cells' is not accurate and could be misleading. It should be more precise.
- **ROW 51** [P1]: The note does not mention continuing or starting gemcitabine, but the extracted data states 'Continue/start: gemcitabine'. This contradicts the note.
- **ROW 52** [P1]: Incomplete sentence with missing critical information.
- **ROW 53** [P1]: Missing cancer-related medications (gemcitabine and Abraxane)
- **ROW 53** [P1]: Incomplete sentence: 'to to better manage your pain' is missing critical information.
- **ROW 62** [P1]: Incomplete and unclear sentence with unexplained jargon.
- **ROW 64** [P1]: Incomplete sentence with missing critical info.
- **ROW 67** [P1]: Missing critical lab results such as CA 19-9, which is relevant to the patient's condition.
- **ROW 71** [P1]: Unnecessary detail and jargon that may confuse the patient.
- **ROW 71** [P1]: Incomplete sentence with missing critical information.
- **ROW 71** [P1]: Missing dose details and unclear abbreviation.
- **ROW 71** [P1]: Incomplete sentence with missing critical information.
- **ROW 71** [P1]: Unnecessary detail and jargon that may confuse the patient.
- **ROW 71** [P1]: Missing context about what paracentesis is.
- **ROW 74** [P1]: Missing gemcitabine and oxaliplatin in the past treatment regimen.
- **ROW 75** [P1]: Incomplete sentence with missing critical information (dose amount).
- **ROW 81** [P1]: Incomplete sentence with missing critical information.
- **ROW 86** [P1]: Incomplete sentence with missing critical information.
- **ROW 90** [P1]: Missing current cancer treatments (dabrafenib and trametinib).
- **ROW 90** [P1]: Incomplete sentence with missing critical information.
- **ROW 95** [P1]: Incomplete sentence with missing critical information (specific medication name).
- **ROW 96** [P1]: Missing cancer-related medication (gemcitabine).
- **ROW 96** [P1]: Incorrect statement about recent changes. The patient switched to gemcitabine monotherapy, not from gemcitabine + capecitabine.
- **ROW 96** [P1]: Incorrect statement about continuing capecitabine. The plan is to try gemcitabine monotherapy.
- **ROW 99** [P1]: Grammatical error: 'You is' should be 'You are'.

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
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine and capecitabine'. | She then started therapy with gemcitabine and capecitabine and has completed six full cycles. |
| P2 | Treatment_Changes | Supportive medications listed are not cancer-treatment-related. | Supportive medications should be related to cancer treatment. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and misclassification of supportive medications.

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
| P2 | Current_Medications.current_meds | The note mentions starting a treatment, but the name is redacted. The extracted data should indicate that a cancer-related medication is being used, even if the name is not specified. | We started her on *****. |

*Extraction summary*: Major issues with the definitiveness of the cancer diagnosis and the lack of mention of the ongoing cancer treatment. Other fields are mostly clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | The extensive centrally necrotic lymphadenopathy and circumferential heterogeneo |

*Letter summary*: Letter is mostly clean but contains some unexplained medical jargon.

---

## ROW 4 — ⚠️ ISSUES

**Type**: Metastatic pancreatic ductal adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine-based regimen mentioned as a potential future treatment, but not listed under current medications). | We talked about potential side fx and risks, esp in terms of infectious complications, and adverse i |
| P2 | Treatment_Changes | Supportive medications listed include general pain medications (Fentanyl patch, Dilaudid) without specifying they are for cancer pain. | His pain appears reasonably managed right now with a combination of long- and short acting opioid an |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication and imprecise classification of supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incorrect verb tense and voice inconsistency. | You was discussed the option of transitioning to hospice care with a preference  |

*Letter summary*: Letter contains a minor grammatical error affecting clarity.

---

## ROW 5 — ⚠️ ISSUES

**Type**: Well-differentiated adenocarcinoma, intestinal type, of ampullary origin
**Stage**: (pT2N1), now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication information. | The patient is currently receiving palliative chemotherapy consisting of dose-modified [REDACTED] (o |
| P2 | Treatment_Changes | Supportive medications listed are not cancer-related. | There is a plan for a treatment-free interval for a couple of months. No maintenance therapy (e.g.,  |
| P2 | Lab_Plan | Lab plan should include bloodwork as mentioned in the note. | We plan on repeat imaging/bloodwork between [REDACTED]/[REDACTED]. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication details, non-cancer supportive medications being listed, and lack of mention of bloodwork in the lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Repeat imaging/bloodwork is planned between your medications. |

*Letter summary*: Letter contains an incomplete sentence that needs clarification.

---

## ROW 6 — ⚠️ ISSUES

**Type**: Pancreatic neuroendocrine tumor, grade 2
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage 'now metastatic ()' is incomplete and lacks specificity. | Patient presents for follow up of recurrent grade 1 pancreatic neuroendocrine tumor. |
| P2 | Clinical_Findings.findings | The note mentions imaging findings from 10/05/2015 PET-CT and MRI, but the exact details are not fully captured. | Recent phase III data confirm that everolimus (4.6 vs 11 months; *****, et al. *****, 2011) and suni |
| P2 | Treatment_Changes.recent_changes | There is no mention of recent changes in treatment, but the note does discuss ongoing treatment with lanreotide. | I favor continuing lanreotide for now. |
| P2 | Procedure_Plan.procedure_plan | The plan mentions future options but does not specify any immediate procedures. | Future options include liver directed therapy, everolimus, sunitinib, or chemotherapy. |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness of the stage, clinical findings, treatment changes, and procedure plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'lymph nodes'. | Imaging from October 5, 2015, shows that the cancer in your liver and lymph node |
| P2 | Unexplained medical jargon 'everolimus', 'sunitinib'. | Future options include liver directed therapy, everolimus, sunitinib, or chemoth |

*Letter summary*: Letter contains minor readability issues due to unexplained medical jargon.

---

## ROW 7 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic ductal adenocarcinoma
**Stage**: pT3 N1

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. Current medications should include both gemcitabine and the discontinued [REDACTED]-paclitaxel. | He started with the combination of gemcitabine plus *****-paclitaxel, but due to cumulative neuropat |

*Extraction summary*: Most fields are clean, but the current medications field is incomplete.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III. | You have a well-differentiated pancreatic ductal adenocarcinoma (cancer that sta |
| P2 | Unexplained medical jargon 'stable disease'. | These imaging findings suggest stable disease. |
| P2 | Incomplete sentence with missing critical info ('a medication-paclitaxel'). | You are currently receiving adjuvant chemotherapy with gemcitabine alone. The a  |

*Letter summary*: The letter contains inaccuracies in staging information and an incomplete sentence, along with minor readability issues.

---

## ROW 8 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma with metastatic disease to the lungs
**Stage**: (metastatic to lungs)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (Gemcitabine). | Starting with cycle 3 patient will start to receive Gemcitabine on weeks 1, 2 and 3, followed by 1 w |
| P2 | Treatment_Changes | Missing detail about the switch to Gemcitabine. | He was recommended to restart systemic therapy in the form of single agent Gemcitabine on days 1 and |
| P2 | Clinical_Findings | Missing detail about the patient's performance status (PS). | Patient's PS is acceptable and he agrees to come in tomorrow for labs and consideration of ***** Gem |

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing cancer-related medications and details about treatment changes and performance status.

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
| P2 | Treatment_Changes | Missing mention of gemcitabine in recent treatment changes. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |
| P2 | Treatment_Goals | Goals of treatment should be 'adjuvant', not 'curative'. | We previously discussed the role of adjuvant chemotherapy in this context to improve disease-free an |
| P2 | Medication_Plan | Missing mention of gemcitabine in the medication plan. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |
| P2 | Therapy_plan | Missing mention of gemcitabine in the therapy plan. | Mr. [REDACTED] elected to proceed with irinotecan and oxaliplatin. Irinotecan was initially dose-red |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the inclusion of gemcitabine in the treatment plan and the classification of treatment goals as 'adjuvant'.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | You are continuing with irinotecan and oxaliplatin. Initially, irinotecan was do |
| P2 | Unexplained medical jargon and incomplete sentence. | You were also recommended a medication (a medication co-packaged with a medicati |

*Letter summary*: The letter contains incomplete sentences with missing critical information and some unexplained medical jargon.

---

## ROW 10 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreatic head
**Stage**: Not staged in note, but with metastatic disease to liver

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Distant Metastasis | The note mentions a liver lesion but does not specify lymph node involvement. | with a liver lesion that is unclear based on imaging. |
| P2 | Lab_Results.lab_summary | CA 19-9 and CEA values are mentioned in the note but not included in the lab summary. | patient does not express CA 19-9 nor does he express ***** |
| P2 | Clinical_Findings.findings | The note mentions elevated LFTs and a possible intrabdominal infection, but the extraction does not fully capture the urgency of the situation. | Start Cipro 500 mg BID x 7 days for possible intrabdominal infection |
| P2 | Treatment_Changes.recent_changes | The note mentions holding treatment due to elevated LFTs, but the extraction uses [REDACTED] which is not specific. | ***** hold treatment (***** neoadjuvant *****) today because patient is afebrile and has elevated LF |
| P2 | Therapy_plan.therapy_plan | Physical therapy is not mentioned in the note. | None |

*Extraction summary*: Most fields are clean, but there are minor issues with distant metastasis, lab summary, clinical findings, treatment changes, and therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'LFTs'. | However, you have elevated liver function tests (LFTs), likely due to a blockage |
| P2 | Minor readability issue, could be clearer. | You will also continue to see a social worker for support. |

*Letter summary*: Letter contains minor readability issues and unexplained jargon.

---

## ROW 11 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis | Missing specific stage information | Ms. ***** is a 55 y.o. female who returns to our GI Oncology practice at the UCSF Comprehensive Canc |
| P2 | Current_Medications | Incomplete listing of cancer-related medications | She using metoclopramide for its pro-motility properties, and I instructed her today to increase dos |
| P2 | Treatment_Changes | Incomplete listing of supportive medications | She will continue to take lorazepam and zolpidem as needed for insomnia. |
| P2 | Treatment_Goals | Inaccurate goal classification | Based upon cumulative fatigue, I will hold treatment today and ordered restaging CT scans to be perf |
| P2 | Response_Assessment | Inaccurate response assessment | Restaging imaging demonstrated disease progression with more prominent soft tissue infiltration infe |
| P2 | Medication_Plan | Incomplete listing of medication plan | Metoclopramide increased to 10 mg four times daily; megestrol acetate resumed for appetite stimulati |

*Extraction summary*: Several fields contain minor issues related to incomplete or slightly inaccurate information, but no major errors.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'Gerota's fascia'. | Recent imaging shows that your cancer has progressed, with new growths in your l |
| P2 | Unexplained medical jargon 'CA-19-9'. | This is also reflected in your CA-19-9 levels, which have increased significantl |
| P2 | Incomplete sentence (missing critical info about what treatment was paused). | Due to increasing fatigue, your treatment has been paused temporarily. |

*Letter summary*: Letter contains minor readability issues and unexplained medical jargon.

---

## ROW 12 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic ductal adenocarcinoma with perineural and l
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is described as 'now metastatic ()', which is imprecise and lacks specific staging information. | The note mentions metastatic disease but does not provide a specific stage beyond metastatic. |
| P2 | Current_Medications.current_meds | The current medication is listed as 'Gemcitabine', but the note mentions 'Gemcitabine monotherapy'. | Patient continues to be on gemcitabine monotherapy. |
| P2 | Treatment_Changes.supportive_meds | Gabapentin is used for neuropathy but is not listed under supportive_meds. | He will continue to use gabapentin for his neuropathy. |

*Extraction summary*: Most fields are accurate, but there are minor issues with the stage description, current medication listing, and missing supportive medication.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Use of 'nab-paclitaxel' is jargon that may be confusing to a layperson. | The cancer is currently showing stable disease based on the most recent restagin |
| P2 | Sentence is slightly complex and could be simplified. | Additionally, if you continue to have stable disease with only a few mesenteric  |

*Letter summary*: Letter contains minor readability issues related to medical jargon and sentence complexity.

---

## ROW 13 — ⚠️ ISSUES

**Type**: Adenocarcinoma consistent with upper GI/pancreato-biliary primary
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary. The note mentions 'evaluation of metastatic pancreatic adenocarcinoma', not 'evaluation of treatment response and management of the disease'. | This is an independent visit...for evaluation of metastatic pancreatic adenocarcinoma. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Inaccurate stage description. The note does not specify the exact stage, only that it is metastatic. | metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Distant Metastasis | Missing information. The note mentions metastasis to the liver and peritoneum. | Additional hypermetabolic hepatic lesions consistent with additional sites of metastatic disease. |
| P2 | Current_Medications.current_meds | Missing cancer-related medications. The note mentions Gemcitabine + Abraxane. | C1D1 Gemcitabine + Abraxane |
| P2 | Treatment_Changes.supportive_meds | Incomplete. The note mentions ondansetron and prochlorperazine, but not all supportive medications are listed. | ondansetron (ZOFRAN), prochlorperazine (COMPAZINE) |
| P2 | Response_Assessment.response_assessment | Inaccurate. The note does not provide specific imaging or tumor marker data to assess the response to treatment. | There is no specific imaging or tumor marker data provided in the note to assess the response to thi |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, stage description, distant metastasis, current medications, supportive medications, and response assessment.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon: 'gland cells'. | You have metastatic pancreatic adenocarcinoma (cancer that started in gland cell |
| P2 | Unexplained jargon: 'peritoneum'. | The cancer has spread to your liver and peritoneum. |
| P2 | Minor readability issue: 'ongoing' might be too complex for 8th-grade level. | You report ongoing fatigue and poor appetite. |

*Letter summary*: Letter contains minor readability issues and unexplained jargon.

---

## ROW 14 — ⚠️ ISSUES

**Type**: High-grade neuroendocrine tumor of duodenal/ampullary origin
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete listing of cancer-related medications. | capecitabine (XELODA) 500 mg tablet and temozolomide (*****) 100 mg capsule are listed in the note. |
| P2 | Treatment_Changes | Missing information about dose adjustments. | Because of progressive renal insufficiency early on during her treatment course, I dose reduced her  |

*Extraction summary*: Most fields are clean, but there are minor issues with incomplete listings of cancer-related medications and dose adjustments.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of the reason for the visit. The patient is being followed up for a high-grade neuroendocrine tumor of duodenal/ampullary origin, not just 'neuroendocrine tumor treatment'. | You came in for a follow-up visit regarding your neuroendocrine tumor treatment. |
| P2 | Unexplained jargon 'MRI'. | MRI scans from early June showed that your cancer is stable. |
| P2 | Unexplained jargon 'fluid buildup'. | During the physical exam, no new masses, fluid buildup, or enlarged liver were f |
| P2 | Unexplained jargon 'ondansetron' and 'oxycodone'. | You will continue to take ondansetron and oxycodone to manage side effects. |
| P2 | Unexplained jargon 'CT scan'. | After this cycle, you will have a CT scan to check on your tumor. |

*Letter summary*: The letter contains inaccuracies and minor readability issues.

---

## ROW 15 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Distant Metastasis | Inaccurate specification of distant metastasis sites. Note mentions metastasis to celiac lymph nodes and liver, but the extracted data only lists liver and peritoneum. | At that time, he presented with back pain and a 15 pound weight loss. His metastatic disease at the  |
| P2 | Treatment_Changes.recent_changes | Incomplete description of recent treatment changes. The note mentions discussing clinical trials and cell therapy programs, but these details are missing from the extracted data. | As he was very interested in clinical trials, we decided to explore this with the phase I group as w |
| P2 | Treatment_Changes.supportive_meds | Missing supportive medications. The note mentions the patient takes turmeric, mushroom extract, and cannabis products, though these are not cancer-related medications. | He does take turmeric, mushroom extract and cannabis products |
| P2 | Imaging_Plan.imaging_plan | Inaccurate imaging plan. The note mentions a follow-up CT scan, but the extracted data only specifies CT Chest. | He returns today to discuss that. He's had a follow-up CT scan. |

*Extraction summary*: Most fields are accurate, but there are minor issues with the completeness and specificity of certain details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The sentence is too complex and uses medical jargon that may be difficult for an 8th-grade reading level patient to understand. | The cancer is progressing despite previous treatments. Evidence includes interva |
| P2 | The sentence is unclear and lacks specificity about the medication being resumed. | We will resume a medication. You responded initially quite well to a medication  |

*Letter summary*: The letter contains some complex sentences and unexplained medical jargon that need simplification for better readability at an 8th-grade level.

---

## ROW 16 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic neuroendocrine tumor, grade 2
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Summary is slightly imprecise, missing details about the suspected sporadic condition. | Subjective   ***** ***** is a 34 y.o. female who presents for f/u of pancreatic neuroendocrine tumor |
| P2 | Lab_Results.lab_summary | Lab results are incomplete and contain some [REDACTED] values. | PNET labs (CareEverywhere):  06/21/21 CgA 281 (nl)  06/21/21 glucagon 139 (elevated, but not signifi |
| P2 | Current_Medications.current_meds | Field is empty, but octreotide and lanreotide are mentioned as relevant cancer treatments. | Off SSA for now-octreotide ***** |
| P2 | Treatment_Changes.recent_changes | Field is empty, but there are recent changes in treatment such as stopping SSA and continuing octreotide. | Off SSA for now-octreotide ***** |
| P2 | Treatment_Goals.goals_of_treatment | Goals are labeled as 'palliative', but the note suggests a surveillance approach with no active treatment. | Most recent MRI of the abdomen on 11/01/21 shows stable disease and no treatment recommended at this |
| P2 | Medication_Plan.medication_plan | Plan mentions octreotide and lanreotide, but does not specify the exact dosages or schedule. | Continue home calcium infusions. Continue short acting pain meds plus long acting. If PCP doesn't wa |
| P2 | Therapy_plan.therapy_plan | Physical therapy is mentioned without context or reason. | Off SSA for now - octreotide. Continue home calcium infusions.; physical therapy |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness and precision of lab results, medication plans, and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'well-differentiated' might be confusing for an 8th-grade reading level. | You have a well-differentiated pancreatic neuroendocrine tumor, grade 2, which h |
| P2 | The term 'metastatic' might be confusing for an 8th-grade reading level. | The latest MRI on 11/01/2021 shows that your disease is stable with no new metas |
| P2 | The term 'hypocalcemia' might be confusing for an 8th-grade reading level. | You continue to receive home calcium infusions for severe hypocalcemia. |
| P2 | The term 'surveillance' might be confusing for an 8th-grade reading level. | Surveillance with abdominal MRI is planned every 3 months, which can be extended |

*Letter summary*: The letter contains minor readability issues that could be improved for an 8th-grade reading level.

---

## ROW 17 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and capecitabine). | She was being followed for a pancreatic cyst that was diagnosed in 2014 on a CT scan... She underwen |

*Extraction summary*: One minor issue identified regarding the omission of cancer-related medications in the 'Current_Medications' field. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information about the specific treatment plan. | Please discuss your next steps and treatment plan with your care team at your ne |

*Letter summary*: Letter contains an incomplete sentence with missing critical information about the treatment plan.

---

## ROW 18 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma, grade 2; Intra-ampullary papillary-tubular neo
**Stage**: Originally Stage IB, now metastatic (pt2n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | Medication Instructions section lists general medications only. |
| P2 | Treatment_Changes | Supportive medications listed are not supported by the note. | No mention of ondansetron, prochlorperazine, or loperamide in the note. |
| P1 | Treatment_Goals | Incorrect goal of treatment stated as 'palliative'. | The note indicates ongoing treatment and discussion of future systemic treatments, suggesting a goal |
| P2 | Response_Assessment | Inaccurate CA 19-9 level mentioned. | Note mentions CA 19-9 level of 1,077, but this is not supported by the note. |
| P2 | Medication_Plan | Incomplete medication plan. | Note discusses potential future regimens but does not specify current plan. |
| P2 | Therapy_plan | Incomplete therapy plan. | Note discusses potential future regimens but does not specify current plan. |
| P2 | Procedure_Plan | Incomplete procedure plan. | Note discusses potential future procedures but does not specify current plan. |

*Extraction summary*: Several fields contain incomplete or incorrect information regarding treatment goals, medication plans, and response assessments. Most fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'modest progression' is somewhat technical and may not be fully understood by an 8th-grade reader. | The cancer is showing modest progression in your lung tumors, as seen on the rec |
| P2 | While 'CA 19-9' is acceptable jargon, the phrase 'tumor marker' may be too technical. | Your CA 19-9 (a tumor marker) level has risen to 1,077, indicating that the canc |
| P2 | This sentence is clear but could be more specific about the purpose of the break. | You will take a break from treatment to allow your body to recover from the side |
| P2 | The term 'Radiation Oncology team' may be unfamiliar to an 8th-grade reader. | You will also be referred to the Radiation Oncology team to see if you are a can |

*Letter summary*: The letter contains minor readability issues that could be improved for better clarity at an 8th-grade reading level.

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
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine) that was part of the adjuvant therapy. | He initiated adjuvant chemotherapy consisting of single-agent gemcitabine on a three-week-on, one-we |

*Extraction summary*: One minor issue identified regarding the omission of a cancer-related medication in the 'Current_Medications' field. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'pancreatic cancer' is overly broad and does not specify the type of cancer, which could be confusing for the patient. | You came in for a follow-up visit regarding your pancreatic cancer treatment. |
| P2 | The term 'intraductal papillary mucinous neoplasms' may be too technical for an 8th-grade reading level. | The pancreatic cystic lesions in the body and tail, which are likely intraductal |

*Letter summary*: Letter contains minor readability issues related to medical terminology.

---

## ROW 21 — ⚠️ ISSUES

**Type**: Metastatic pancreatic cancer
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (5-FU/LV + irinotecan). | On this basis, he initiated 2nd-line chemotherapy consisting of the combination of 5-*****/LV plus * |
| P2 | Treatment_Changes | Missing detail about the specific cancer-related medication (irinotecan) dose reduction. | Due to obstipation and anorexia following cycle #1, his doses were reduced by 20% beginning with cyc |
| P2 | Treatment_Goals | Incomplete. Should specify 'palliative/symptomatic measures'. | Therefore we had a ***** goals of care discussion in which I recommended that he refocus his goals p |
| P2 | Medication_Plan | Incorrect. The patient is no longer a candidate for further salvage treatment. | At this point, he is no longer a candidate for any further salvage treatment in terms of either SOC  |
| P2 | Therapy_plan | Incorrect. The patient is no longer a candidate for further salvage treatment. | At this point, he is no longer a candidate for any further salvage treatment in terms of either SOC  |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medication details and incorrect treatment plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 22 — ⚠️ ISSUES

**Type**: Pancreatic cancer
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Age discrepancy between '70 y.o.' in summary and '71 y.o.' in impression. | Impression: 71 y.o. female with metastatic pancreatic cancer as summarized above. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage 'Metastatic ()' is vague and could be more specific. | Impression: 71 y.o. female with metastatic pancreatic cancer as summarized above. |
| P2 | Current_Medications.current_meds | Missing cancer-related medication 'liposomal irinotecan'. | 02/21/17 ***** Dose reduced 5FU 66.7% |
| P2 | Treatment_Changes.supportive_meds | Missing supportive medication 'olanzapine (Zyprexa)' for nausea. | s/e -n/v + relief with addition of olanzapine (Zyprexa) |
| P2 | Treatment_Goals.goals_of_treatment | Goals should be 'palliative' but could be more specific regarding ongoing treatment. |  |
| P2 | Response_Assessment.response_assessment | Recent scans show no evidence of progression, but tumor markers trending upwards should be highlighted. | Tumor markers have been trending upwards by  30% over the last |
| P2 | Genetic_Testing_Plan.genetic_testing_plan | Plan mentions 'phase I', but it should specify 'early-stage treatments based on BRCA2 genotype'. | I sent her to phase I to evaluate her candidacy for early stage treatments based on this genotype |

*Extraction summary*: Most fields are clean, but there are minor issues related to age discrepancy, vague staging, missing cancer-related medications, and incomplete genetic testing plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence, missing critical information about the hand-foot rash. | Your physical exam revealed no significant abnormalities, with normal breath sou |
| P2 | Unexplained jargon 'imaging'. | Recent imaging shows that your cancer is currently stable, with no new disease i |

*Letter summary*: The letter contains an incomplete sentence and minor readability issues.

---

## ROW 23 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage 'now metastatic ()' is incomplete and lacks specificity. | now metastatic () |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions ongoing chemotherapy with gemcitabine and [REDACTED]-paclitaxel. | He has recently resumed chemotherapy with gemcitabine/[REDACTED]-paclitaxel |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions resuming chemotherapy with gemcitabine/[REDACTED]-paclitaxel. | He has recently resumed chemotherapy with gemcitabine/[REDACTED]-paclitaxel |
| P2 | radiotherapy_plan.radiotherapy_plan | The field is vague and does not specify the exact radiotherapy plan. | Given its central location with mediastinal involvement, I would favor d/c'ing current therapy, and  |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness and specificity of the cancer stage, current medications, treatment changes, and radiotherapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'tumor'. | Imaging shows that some areas of your cancer have gotten smaller, but the larges |
| P2 | Missing information about the dosing schedule. | You have recently restarted chemotherapy with gemcitabine and paclitaxel. |

*Letter summary*: Letter is mostly clean but requires minor adjustments for clarity and completeness.

---

## ROW 24 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic adenocarcinoma with perineural invasion
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage of cancer is not specified, though reasonable inference could be made based on the pathology report. | Pathologic specimen: 2.2 cm well-differentiated adenocarcinoma with extension into the peri-pancreat |
| P2 | Cancer_Diagnosis.Distant_Metastasis | The distant metastasis status is marked as 'Not sure', though the note suggests possible pulmonary metastases. | CT-PET 12/22/2012: Small, minimally hypermetabolic pulmonary nodules up to 1.3 cm in size, R>L. |
| P2 | Cancer_Diagnosis.Metastasis | The metastasis status is marked as 'Not sure', though the note suggests possible pulmonary metastases. | CT-PET 12/22/2012: Small, minimally hypermetabolic pulmonary nodules up to 1.3 cm in size, R>L. |
| P2 | Current_Medications.current_meds | The current medications field is empty, though there are no cancer-related medications listed in the note. | No cancer-related medications mentioned. |
| P2 | Treatment_Changes.recent_changes | The recent changes field is empty, though there are no recent changes mentioned in the note. | No recent changes mentioned. |
| P2 | Treatment_Changes.supportive_meds | The supportive medications field is empty, though there are no cancer-treatment-related supportive care medications listed in the note. | No cancer-treatment-related supportive care medications mentioned. |
| P2 | Treatment_Goals.goals_of_treatment | The treatment goal is marked as 'curative', though the patient is currently in a surveillance phase with suspicious findings. | Suspicious findings that need further evaluation. CT-PET on 12/22/2012 shows small, minimally hyperm |

*Extraction summary*: Most fields are clean, but there are minor issues related to the lack of specified cancer stage, metastasis status, and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence. It mentions 'involvement of 27 lymph nodes' but doesn't specify whether this is positive or negative. | The pathology showed a 2.2 cm well-differentiated adenocarcinoma (cancer that st |
| P2 | Unexplained jargon ('hypermetabolic'). | Imaging found small, minimally hypermetabolic pulmonary nodules up to 1.3 cm in  |

*Letter summary*: Letter contains minor issues that need clarification and slight readability improvements.

---

## ROW 25 — ⚠️ ISSUES

**Type**: Moderately differentiated Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it can be reasonably inferred from the note. | Large mixed cystic and solid mass arising from the pancreatic tail which invades the left adrenal gl |
| P2 | Current_Medications.current_meds | Current cancer-related medications are missing. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |
| P2 | Treatment_Changes.recent_changes | Recent treatment changes are incomplete; dose reduction of irinotecan is mentioned but not the specific regimen name. | Currently on ***** *****. C1D1=09/12/20, c/b severe nausea, requiring trt delay, and dose reduced ** |
| P2 | Medication_Plan.medication_plan | Medication plan mentions compazine without specifying its use or discontinuation. | However, recurred *****, so asked to stop compazine and try olanzapine for now |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, incomplete treatment changes, and vague medication plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (missing the new dose amount). | Your dose of irinotecan was reduced by 20% to due to persistent nausea. |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 26 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Lab_Results | The lab results section contains fabricated information not present in the note. | There is no mention of specific lab results such as WBC count, Neutrophil Absolute Count, Hemoglobin |
| P2 | Current_Medications | The current medications section only mentions 'gemcitabine', but the note also mentions 'moxifloxacin'. | Stopped moxifloxacin on 05/15/21 |
| P2 | Treatment_Changes | The treatment changes section mentions 'ondansetron (ZOFRAN), prochlorperazine (COMPAZINE)' which are not mentioned in the note. | None |
| P2 | Treatment_Goals | The treatment goals are listed as 'curative' but the note does not explicitly state the goal as curative. | Plan after initial consultation March 12 was to proceed with neoadjuvant chemotherapy per Dr. *****. |
| P2 | Medication_Plan | The medication plan mentions continuing 'Moxifloxacin' but the note states it was stopped on 05/15/21. | Stopped moxifloxacin on 05/15/21 |
| P2 | Therapy_plan | The therapy plan mentions continuing 'Moxifloxacin' but the note states it was stopped on 05/15/21. | Stopped moxifloxacin on 05/15/21 |

*Extraction summary*: Major issues include fabricated lab results and incomplete current medications. Minor issues involve unspecified treatment goals and outdated medication plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'hypermetabolic'. | You have a hypermetabolic pancreatic head mass measuring approximately 2.5 x 2.3 |
| P2 | Unexplained jargon 'nodal' and 'metastatic'. | No clear nodal or distant metastatic disease was identified. |
| P2 | Unexplained jargon 'pulmonary nodules'. | Scattered small pulmonary nodules are unchanged. |
| P2 | Unexplained jargon 'vertebral body lucent lesion'. | The T5 vertebral body lucent lesion remains unchanged. |
| P2 | Unexplained jargon 'percutaneous drainage'. | The size of the peripancreatic fluid collection has decreased since the percutan |
| P2 | Unexplained jargon 'psoas muscle', 'hydronephrosis', and 'adjacent inflammation'. | However, there is a new component of the collection involving the right psoas mu |

*Letter summary*: The letter contains several instances of unexplained medical jargon that could be clarified for better readability.

---

## ROW 27 — ⚠️ ISSUES

**Type**: Extrahepatic cholangiocarcinoma; Metastatic hepatocellular carcinoma
**Stage**: Unresectable; Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'Lamivudine' for HBV reactivation prevention. | Continue Lamivudine prophylaxis for HBV reactivation prevention |
| P2 | Treatment_Changes | Missing mention of 'Remeron' continuation. | Continue Remeron |
| P2 | Treatment_Changes | Missing mention of 'CBD oil' continuation. | Encouraged her to continue cannabis product (CBD oil) as this has shown to help stimulate appetite a |
| P2 | Medication_Plan | Missing mention of 'Remeron'. | Continue Remeron |
| P2 | Medication_Plan | Missing mention of 'CBD oil'. | Encouraged her to continue cannabis product (CBD oil) as this has shown to help stimulate appetite a |
| P2 | Medication_Plan | Mention of 'oxaliplatin' and 'dexamethasone' is incorrect as they are not part of the current treatment plan. | Continue atezolizumab + bevacizumab Q 3 [REDACTED] |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and incorrect inclusion of non-current treatments.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'chronic lung condition' is vague and could be explained more clearly for an 8th-grade reading level. | Your recent CT scan shows that the cancer is currently stable. There is a slight |
| P2 | The drug names 'atezolizumab' and 'bevacizumab' are complex and might be confusing for an 8th-grade reader. | You will continue your treatment with atezolizumab and bevacizumab every 3 weeks |

*Letter summary*: Letter is mostly clean but contains minor readability issues that could be improved for clarity.

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
| P2 | Current_Medications.current_meds | The field only includes 'Gemcitabine, Abraxane', but the note also mentions 'Lovenox' as part of the treatment. | She was started back on Lovenox. |
| P2 | Treatment_Changes.recent_changes | The field only includes 'Restarted Gemcitabine + Abraxane', but the note also mentions starting 'Lovenox'. | She was started back on Lovenox. |
| P2 | Medication_Plan.medication_plan | The field only includes 'Gemcitabine, Abraxane, Lasix, spironolactone, Ritalin, Mirtazapine', but the note also mentions 'Lovenox'. | She was started back on Lovenox. |
| P2 | Therapy_plan.therapy_plan | The field only includes 'Gemcitabine, Abraxane, lasix, home health', but the note also mentions 'Lovenox'. | She was started back on Lovenox. |

*Extraction summary*: Most fields are clean, but there are minor issues related to the inclusion of 'Lovenox' in the treatment plan and a slight discrepancy in the summary mentioning 'new diagnosis'.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The phrase 'new diagnosis' is inaccurate given the context of a follow-up visit. | You have a new diagnosis of metastatic adenocarcinoma (cancer that started in gl |
| P2 | The term 'lesions' might be confusing; 'spots' or 'areas' would be more understandable. | However, some liver lesions remain unchanged. |
| P2 | Prochlorperazine is not commonly known as 'Prochlorperazine'; it's typically referred to as 'Compazine'. | You are also taking supportive medications such as Zofran and Prochlorperazine f |

*Letter summary*: Letter contains minor inaccuracies and readability issues.

---

## ROW 30 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced; Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not explicitly state the stage of cancer, but infers it as locally advanced and metastatic. The extracted data should reflect this uncertainty. | Locally advanced; Metastatic () |
| P2 | Treatment_Changes.recent_changes | The note indicates that the patient agreed to take a break from chemotherapy, but the extracted data does not specify the duration or future plans. | We agreed that she warrants a break from chemotherapy; we will plan on short-term follow up CT scans |
| P2 | Treatment_Goals.goals_of_treatment | The note suggests a palliative approach, but the extracted data should clarify whether the goal is primarily palliative or includes maintenance therapy. | palliative |
| P2 | Medication_Plan.medication_plan | The note mentions that the patient is on reduced-dose gemcitabine and [REDACTED]-paclitaxel, but the extracted data incorrectly states that she is now back on chemotherapy. | The patient is now back on chemotherapy consisting of reduced-dose gemcitabine and [REDACTED]-paclit |
| P2 | Lab_Plan.lab_plan | The note mentions that short-term follow-up CT scans and bloodwork will be conducted, but the extracted data states no labs are planned. | We will plan on short-term follow up CT scans and bloodwork (right after [REDACTED]). |

*Extraction summary*: Most fields are clean, but there are minor issues with the staging, treatment changes, medication plan, and lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon: 'metastatic disease' | Recent imaging shows that the cancer in your pancreas is stable. There are some  |
| P2 | Unexplained jargon: 'liquid biopsy' | Liquid biopsy analysis will be ordered to look for any actionable mutations. |

*Letter summary*: Letter is mostly clean but contains minor readability issues with unexplained medical jargon.

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
| P1 | Lab_Results | The lab results listed in the extracted data do not match the lab results mentioned in the note. | The note does not provide specific lab results such as Alkaline Phosphatase, Bilirubin, Alanine tran |
| P1 | Current_Medications | The field is empty, but the note mentions several cancer-related medications being used. | The note mentions '*****' (likely a chemotherapy regimen) and 'irinotecan'. |
| P2 | Treatment_Changes | The field incorrectly states 'Switch to gemcitabine combined with nab-paclitaxel' as a recent change, while the note indicates this is a future plan. | The note states 'my recommendation is to switch her therapy to gemcitabine combined with *****-pacli |
| P2 | Treatment_Goals | The field states 'palliative', but the note does not explicitly state the treatment goal. | The note does not explicitly mention the treatment goal as 'palliative'. |
| P2 | Medication_Plan | The field incorrectly includes 'irinotecan' in the medication plan, while the note indicates switching from '*****' (likely a chemotherapy regimen) to 'gemcitabine combined with nab-paclitaxel'. | The note states 'my recommendation is to switch her therapy to gemcitabine combined with *****-pacli |
| P2 | Therapy_plan | The field incorrectly includes 'Continue irinotecan', while the note indicates switching from '*****' (likely a chemotherapy regimen) to 'gemcitabine combined with nab-paclitaxel'. | The note states 'my recommendation is to switch her therapy to gemcitabine combined with *****-pacli |
| P2 | follow_up_next_visit | The field incorrectly states 'in-person: 2 weeks when she is due for [REDACTED]', while the note indicates a televisit. | The note states 'As always, we reminded the patient to call us with any questions or concerns in the |

*Extraction summary*: Major errors in lab results and current medications. Minor issues in treatment changes, treatment goals, medication plan, therapy plan, and follow-up next visit.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (dose reduction of irinotecan). | We switched your treatment to gemcitabine combined with nab-paclitaxel. The dose |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 34 — ⚠️ ISSUES

**Type**: Adenocarcinoma
**Stage**: Borderline resectable

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication. | He is being treated in a neoadjuvant fashion with *****. |
| P2 | Treatment_Changes | Supportive medications listed but not specified as cancer-related. | Also not clear why this ambulatory fit man is getting inpatient *****. |
| P2 | Treatment_Goals | Goal of treatment is unclear, 'curative' might be too specific. | We discussed that he should get 4-6 cycles over 2-3 months and then reimage with a quad phase protoc |
| P2 | Medication_Plan | Specific medication name is [REDACTED]. | He is being treated in a neoadjuvant fashion with *****. |
| P2 | Therapy_plan | Specific medication name is [REDACTED]. | He is being treated in a neoadjuvant fashion with *****. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medication details and specificity of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence, missing the name of the medication. | You are being treated in a neoadjuvant fashion with a medication, which means tr |
| P2 | Unexplained jargon 'EUS FNA'. | Pathology: adenocarcinoma (cancer that started in gland cells) identified via EU |

*Letter summary*: Letter contains one major issue with an incomplete sentence and a minor readability issue with unexplained jargon.

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
| P2 | Current_Medications | Incomplete. The note mentions everolimus but does not include other cancer-related medications like Sandostatin (mentioned previously). | We had also been considering the possibility of liver-directed embolization (***** or Y-90) of her s |
| P2 | Treatment_Changes | Incomplete. The note mentions that everolimus was poorly tolerated initially with significant GI symptoms, but this detail is not captured. | Previous treatment has included a distal pancreatectomy of her primary lesion back in 2002; several  |
| P2 | Treatment_Goals | Inaccurate. The note suggests the goal is 'durable disease control,' which is more specific than just 'palliative.' | In summary, Ms. ***** is a 56 y.o. female with oligometastatic liver disease from a non-functioning  |
| P2 | Response_Assessment | Redundant. The imaging findings are already detailed under 'Clinical Findings.' | My review of her most recent CT scans show essentially stable disease, with minimal growth in some m |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and specificity in a few fields.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Your dose of everolimus was reduced from 10 . |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 37 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary. The note mentions a follow-up visit, not a new patient consultation. | Follow Up    Mr. ***** is a 89 y.o. male whom I am seeing as a video visit at the UCSF Comprehensive |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Missing stage information. The note does not explicitly state the stage, but it implies a locally advanced or metastatic stage based on the treatment history. | Baseline imaging 09/28/19 demonstrated a 2.5cm pancreatic uncinate mass that appeared resectable and |
| P2 | Current_Medications.current_meds | Missing current cancer-related medications. The note mentions the patient is off anticoagulation and has had previous treatments like gemcitabine and abraxane. | Currently off anticoagulation. Will consider [REDACTED] if local recurrence only vs 5FU/nal [REDACTE |
| P2 | Treatment_Changes.recent_changes | Missing recent treatment changes. The note mentions the patient has been off anticoagulation and had previous treatments like gemcitabine and abraxane. | Currently off anticoagulation. Will consider [REDACTED] if local recurrence only vs 5FU/nal [REDACTE |
| P2 | Treatment_Goals.goals_of_treatment | Inaccurate goal. The note suggests a surveillance and management approach, but it does not explicitly state 'surveillance'. | Overall Plan:    # Pancreatic Cancer: History as above.  Most recent CT ***** January 20 and CTA abd |
| P2 | Therapy_plan.therapy_plan | Incomplete therapy plan. The note mentions gemcitabine and abraxane, but the plan is not clearly stated. | Currently off anticoagulation. Will consider [REDACTED] if local recurrence only vs 5FU/nal [REDACTE |
| P2 | Procedure_Plan.procedure_plan | Incomplete procedure plan. The note mentions considering a surgical bypass, but the plan is not clearly stated. | --d/w Dr. ***** role and timing of surgical bypass given ongoing obstructive sx despite stent |
| P2 | follow_up_next_visit.Next clinic visit | Missing next clinic visit date. The note does not specify the next visit date. | Not specified in the provided text |

*Extraction summary*: Several minor issues with the summary, stage, current medications, treatment changes, goals, therapy plan, procedure plan, and follow-up visit date. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'adenocarcinoma'. | You have pancreatic adenocarcinoma (cancer that started in gland cells). |
| P2 | Slightly imprecise wording. The original note mentions 'no radiographic evidence of active gastrointestinal hemorrhage', which is more specific. | There is no sign of bowel problems or active bleeding. |
| P2 | Slightly imprecise wording. The original note suggests specific treatment options based on the extent of the cancer. | You may continue with gemcitabine and abraxane, or consider other treatments if  |

*Letter summary*: Letter contains minor readability and precision issues.

---

## ROW 38 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma of pancreatic origin with extensive per
**Stage**: Originally pT3N0, now metastatic (pt3n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine). | She states that after the last cycle of chemotherapy, it took her awhile to recover in terms of fati |
| P2 | Treatment_Changes | Missing recent treatment changes (gemcitabine completion). | Her last dose of gemcitabine was at the end of December 2012. |
| P2 | Therapy_plan | Incorrectly mentions 'Continue radiation therapy referral'. Radiation therapy is not mentioned in the note. | None |

*Extraction summary*: Most fields are clean, but there are minor issues with missing gemcitabine information and an incorrect mention of radiation therapy.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | You will be recommended to try treatment with a medication or a medication. |

*Letter summary*: Letter contains an incomplete sentence that needs clarification.

---

## ROW 39 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma, MMR intact
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (Gemcitabine + Abraxane) | Patient has been recommended to proceed with 1L Gemcitabine + Abraxane |
| P2 | Treatment_Changes | Inconsistent use of [REDACTED] for the same medication | Treatment ([REDACTED] [REDACTED]/Ab) was held by 1 week due to ongoing fatigue and other ongoing lik |
| P2 | Lab_Plan | Incomplete and irrelevant content | Plan       ***** ***** is a 63 y.o.  male who presents with metastatic pancreati |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication, inconsistent use of [REDACTED], and incomplete lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon: 'Gemcitabine + Abraxane' is not explained. | Your treatment with Gemcitabine + Abraxane was paused for one week due to ongoin |
| P2 | Unexplained jargon: 'restaging CT scan' is not explained. | A restaging CT scan is planned after cycle 4. |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

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
| P2 | Treatment_Changes | Missing recent treatment changes. | ***** cycle (#12)was administered with 20% dose reductions in both oxaliplatin and irinotecan due to |
| P2 | Imaging_Plan | Missing recommendation for PET/CT. | Further evaluation with PET/CT may be helpful for confirmation. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, treatment changes, and imaging recommendations.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'hepatic'. | There is a hepatic lesion in segment 8 that has decreased in size from 1.4 x 0.7 |
| P2 | Unexplained medical jargon 'ill-defined soft tissue'. | There is also an increase in ill-defined soft tissue measuring up to 2.1 x 1.7 c |

*Letter summary*: Letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 42 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally recurrent (progressing)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (***** systemic therapy). | Started ***** systemic therapy March of this year |
| P2 | Treatment_Changes | Supportive medications list is incomplete (missing LORazepam, acetaminophen, lidocaine-prilocaine, simethicone). | Current Outpatient Prescriptions section lists several medications including LORazepam, acetaminophe |
| P2 | Imaging_Plan | Incorrect imaging plan (CT Chest already done). | Most recent imaging was personally reviewed and interpreted in conjunction with formal radiology rep |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, incomplete supportive medications list, and incorrect imaging plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 43 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Borderline resectable pancreatic adenocarcinoma due to <180 degrees of vascular 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | Patient states to be taking prior to today's encounter: clopidogrel (PLAVIX), docusate sodium (COLAC |
| P2 | Treatment_Changes | Supportive medications are incomplete. | Continue compazine on days June 28, then switch to Zofran day 3 in the afternoon for 5 or 7 days. Ca |
| P1 | Treatment_Goals | Incorrect goal of treatment. | Patient has been recommended to undergo neoadjuvant chemotherapy, which he started on 10/03/19. |
| P2 | Medication_Plan | Incomplete medication plan. | Continue [REDACTED] neoadjuvant therapy with Oxaliplatin 49 mg/m2 (25% dose reduction), Irinotecan 1 |
| P2 | Therapy_plan | Incomplete therapy plan. | Patient agrees to proceed with the adjusted neoadjuvant therapy today. [REDACTED] after cycle 10. |
| P2 | Lab_Plan | Incomplete lab plan. | Continue to monitor [REDACTED] July 04 [REDACTED] 4 weeks. |

*Extraction summary*: Several fields contain incomplete information related to cancer treatment and plans, while others are accurate.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of the reason for the visit. The patient is being treated for borderline resectable pancreatic adenocarcinoma, not general pancreatic cancer. | You came in for a follow-up visit regarding your pancreatic cancer treatment. |
| P2 | Unexplained jargon 'CA 19-9'. | Your CA 19-9 levels, a tumor marker, have fluctuated but are slowly declining. O |
| P2 | Unexplained jargon 'CT scan'. | A CT scan on 11/19/19 showed that the cancer is stable. |
| P2 | Unexplained jargon 'Oxaliplatin', 'Irinotecan', 'bolus 5FU'. | Your chemotherapy doses were reduced. Oxaliplatin and Irinotecan were reduced by |
| P2 | Unexplained jargon 'Compazine', 'Zofran'. | You will continue to take Compazine and Zofran for nausea. |
| P2 | Unexplained jargon 'CT scan'. | You will continue with the adjusted chemotherapy treatment. A CT scan will be do |

*Letter summary*: The letter contains inaccuracies and some unexplained medical jargon that need addressing.

---

## ROW 44 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Locally advanced ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field only lists 'gemcitabine', but the note mentions '5FU/LV' and 'gemcitabine/abraxane'. | Started on Lovenox for bilateral PE. Switch to gemcitabine/abraxane starting 10/25/20. Held oxalipla |
| P2 | Treatment_Changes | The field mentions 'irinotecan' being omitted, but does not specify when it was omitted. | Irinotecan was omitted with C3. She has tolerated subsequent cycles of 5FU/LV but continued to have  |
| P2 | Medication_Plan | The field incorrectly includes 'irinotecan' under medication plan, which was omitted. | Irinotecan was omitted with C3. She has tolerated subsequent cycles of 5FU/LV but continued to have  |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness of cancer-related medications and treatment changes.

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
| P2 | Treatment_Changes | Supportive medications are missing. | He has now completed 5 cycles, with his most recent dose in ***** administered on 09/10/2017. |
| P2 | Treatment_Goals | Goals of treatment should be 'adjuvant', not 'curative'. | He is currently in the midst of his course of adjuvant chemotherapy with single-agent gemcitabine. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and misclassification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The phrase 'cancer that started in gland cells' is not accurate and could be misleading. It should be more precise. | You have a moderately differentiated pancreatic adenocarcinoma (cancer that star |
| P2 | This sentence contains medical jargon that may be confusing to a layperson. | Your lab results show WBC 3.2, Hgb 8.7, plts 302, Creat 0.68, Tot bili 0.30, and |

*Letter summary*: There are issues with accuracy and readability in the letter.

---

## ROW 46 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is listed as 'Metastatic ()', which is incomplete. | Ms. ***** is a 72 y.o. female with metastatic pancreatic cancer. |
| P2 | Cancer_Diagnosis.Distant Metastasis | The distant metastasis is listed as 'Yes, to liver and possibly adrenal gland', but the note mentions a hypodense hepatic segment 8 lesion and a right adrenal nodule, which are more specific. | Repeat imaging (CT C/A/P) on 10/24/2020, in addition to the primary pancreatic head mass, demonstrat |
| P2 | Cancer_Diagnosis.Metastasis | The metastasis is listed as 'Not sure', which is vague and not supported by the note. | Repeat imaging (CT C/A/P) on 10/24/2020, in addition to the primary pancreatic head mass, demonstrat |
| P2 | Lab_Results.lab_summary | The lab summary includes values not mentioned in the note. | There is no specific lab summary provided in the note. |
| P2 | Current_Medications.current_meds | The current medications field is empty, but the note mentions gemcitabine and paclitaxel. | The patient ultimately initiated chemotherapy consisting of the combination of gemcitabine plus **** |
| P2 | Treatment_Changes.recent_changes | The recent changes field is empty, but the note mentions changes in treatment. | She continued on the same regimen with a dose reduction in her *****-paclitaxel by 20% for better to |
| P2 | Treatment_Changes.supportive_meds | The supportive medications include general pain medications, which should be excluded. | Aside from her celiac plexus blocks x 2, she has been on a combination of long- and short-acting mor |
| P2 | Treatment_Goals.goals_of_treatment | The treatment goal is listed as 'palliative', but the note suggests a focus on supportive care and monitoring. | I do think at some point it will be of great advantage to re-establish home hospice services, when s |
| P2 | Response_Assessment.response_assessment | The response assessment mentions a size change in the pancreatic mass that is not consistent with the note. | Compared to 11/19/2021, slight decrease in size of ill-defined pancreatic head mass and adjacent mes |
| P2 | Medication_Plan.medication_plan | The medication plan includes general pain medications, which should be excluded. | Refer to Pain management team for further titration and oversight of pain management, including cons |
| P2 | Therapy_plan.therapy_plan | The therapy plan mentions no current or future chemotherapy, but the note does not explicitly state this. | We reviewed her most recent imaging studies which show continued non-progression of her disease, wit |
| P2 | Procedure_Plan.procedure_plan | The procedure plan mentions endoscopic evaluation, but the note states the patient is somewhat reluctant to pursue this. | We talked about endoscopic evaluation which she is somewhat reluctant to pursue but will take time t |
| P2 | Imaging_Plan.imaging_plan | The imaging plan mentions further imaging dictated by the family's request, but the note does not specify this. | We will follow up again in another ~3-4 weeks' time, with further imaging dictated by the family's r |
| P2 | follow_up_next_visit.Next clinic visit | The next clinic visit is listed as 'in-person', but the note mentions a televisit. | I performed this consultation using real-time Telehealth tools, including a live video connection be |

*Extraction summary*: Several fields contain minor issues related to completeness and classification, but no major errors.

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
| P2 | Clinical_Findings | The physical exam findings are not fully detailed as in the note. | Physical Exam: Well-developed, well-nourished in no apparent distress. Abdominal: Soft, nontender, n |
| P2 | Current_Medications | Missing 'metformin' and 'entecavir' in the current medications list. | On entecavir 0.5 mg /d, recently increased metformin dose. |
| P2 | Treatment_Changes | Recent treatment changes are not captured, such as dose reduction of everolimus. | Everolimus 10 mg QD for hypoglycemia, dose-reduced for C4 for Grade 2 ***** Capecitabine 1000/500, c |
| P2 | Treatment_Goals | Goals of treatment should include 'symptom management' in addition to 'palliative'. | Well diff ***** with insulin excess, managed by Dr *****. |
| P2 | Medication_Plan | Missing 'metformin' and 'entecavir' in the medication plan. | On entecavir 0.5 mg/d, recently increased metformin dose. |
| P2 | Therapy_plan | Missing 'metformin' and 'entecavir' in the therapy plan. | On entecavir 0.5 mg/d, recently increased metformin dose. |
| P2 | Lab_Plan | Missing specific lab tests like 'gastrin', 'pancreatic polypeptide', 'VIP', 'chromogranin A', 'glucagon', 'proinsulin', 'C-peptide'. | gastrin level 425 pg per mL, pancreatic polypeptide 260 pg per mL, VIP 45 pg per mL, chromogranin A  |

*Extraction summary*: Several minor issues found in lab results, clinical findings, current medications, treatment changes, treatment goals, medication plan, therapy plan, and lab plan. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'neuroendocrine tumor' may be considered jargon for an 8th-grade reading level. | Recent imaging showed no significant changes in the pancreatic lesion, which is  |
| P2 | The term 'density' might be confusing for an 8th-grade reader. | There was an increase in density of a nodule in your left upper lung, but this i |
| P2 | The term 'octreotide' might be confusing for an 8th-grade reader. | You will continue taking everolimus and octreotide. |
| P2 | The term 'genetics' might be confusing for an 8th-grade reader. | You will return to genetics in 3 years for further evaluation. |

*Letter summary*: The letter contains minor readability issues that could be improved for an 8th-grade reading level.

---

## ROW 48 — ⚠️ ISSUES

**Type**: Infiltrative pancreatic head and uncinate process adenocarcinoma
**Stage**: At least locally advanced, more likely metastatic

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | Her treatment course to date has included ***** stenting (metallic) on 09/17/2016 due to the develop |
| P2 | Treatment_Changes | Missing mention of 'gemcitabine' in the treatment changes. | Her treatment course to date has included ***** stenting (metallic) on 09/17/2016 due to the develop |
| P2 | Treatment_Goals | Goals of treatment should include 'symptom management' in addition to 'palliative'. | Otherwise, we will refocus our goals on purely palliative/symptomatic measures with a referral to ho |
| P2 | Medication_Plan | Incorrectly includes 'gemcitabine', which is not part of the current plan. | Prescribed low-dose Decadron as an appetite stimulant, and received a liter of IV fluids at the infu |
| P2 | Therapy_plan | Incorrectly includes 'gemcitabine', which is not part of the current plan. | Prescribed low-dose Decadron as an appetite stimulant and have her receive a liter of IV fluids at t |
| P2 | Advance_care_planning | Advance care planning was discussed during this visit. | I had a frank goals of care discussion with her and her family. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, incomplete treatment goals, and incorrect inclusion of 'gemcitabine' in the current plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'peritoneum'. | There is more fluid in your belly area (ascites), which suggests the cancer migh |
| P2 | Incomplete sentence, lacks detail about the severity or impact of the symptoms. | You are also having trouble eating and feeling very weak. |

*Letter summary*: Letter is mostly clean but contains minor readability issues and unexplained medical jargon.

---

## ROW 49 — ⚠️ ISSUES

**Type**: Localized pancreatic cancer
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing, though it can be reasonably inferred as localized. | Localized pancreatic cancer, originally diagnosed in May 2013 |
| P2 | Current_Medications.current_meds | Current medications related to cancer are missing, though the patient is currently off treatment. | The patient has now been off all therapy for the past 5 months. |
| P2 | Treatment_Changes.recent_changes | Recent treatment changes are missing, though the patient is currently off treatment. | The patient has now been off all therapy for the past 5 months. |
| P2 | Treatment_Changes.supportive_meds | Supportive medications related to cancer treatment are missing, though the patient is currently off treatment. | The patient has now been off all therapy for the past 5 months. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the missing stage of cancer, and the lack of detail on current medications and treatment changes while the patient is off treatment.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 50 — ⚠️ ISSUES

**Type**: Locally advanced pancreatic cancer
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage of cancer is missing. | The note does not explicitly state the stage of cancer. |
| P2 | Current_Medications.current_meds | Missing cancer-related medications. | The note mentions 'gemcitabine plus capecitabine', but they are not listed in the extracted data. |
| P2 | Treatment_Changes.recent_changes | Missing information about recent treatment changes. | The note indicates the patient has completed 2 full cycles of therapy and is well-tolerated, but thi |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and treatment changes.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 51 — ⚠️ ISSUES

**Type**: Pancreatic head adenocarcinoma
**Stage**: pT2N1 (originally Stage II)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Distant_Metastasis | The note does not provide a clear statement regarding distant metastasis, but the extracted data states 'Not sure'. This is slightly imprecise. | The note mentions 'hepatic metastases' but does not definitively state whether they are distant meta |
| P2 | Cancer_Diagnosis.Metastasis | The note does not provide a clear statement regarding metastasis, but the extracted data states 'Not sure'. This is slightly imprecise. | The note mentions 'hepatic metastases' but does not definitively state whether they are metastasis. |
| P2 | Current_Medications.current_meds | The note does not mention any ongoing cancer-related medications, but the extracted data states ''. This is slightly imprecise as it could imply there might be some. | The note does not mention any ongoing cancer-related medications. |
| P2 | Treatment_Changes.recent_changes | The note does not mention any recent treatment changes, but the extracted data states ''. This is slightly imprecise as it could imply there might be some. | The note does not mention any recent treatment changes. |
| P2 | Treatment_Changes.supportive_meds | The note does not mention any supportive medications related to cancer treatment, but the extracted data states ''. This is slightly imprecise as it could imply there might be some. | The note does not mention any supportive medications related to cancer treatment. |
| P1 | Medication_Plan.medication_plan | The note does not mention continuing or starting gemcitabine, but the extracted data states 'Continue/start: gemcitabine'. This contradicts the note. | The note states 'we agreed to continue with a course of monitoring/expectant management w/o treatmen |

*Extraction summary*: Most fields are clean, but there are minor issues with imprecise wording and a major error regarding the medication plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'soft tissue abnormality'. | Recent imaging shows a slight increase in the extent of the soft tissue abnormal |
| P2 | Unexplained medical jargon 'CA-19-9 levels'. | Additionally, your CA-19-9 levels, which are a tumor marker, are elevated, sugge |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

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
| P1 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) | We then started gemcitabine and Abraxane. She's had one full cycle. |
| P2 | Treatment_Changes | Supportive medications list is incomplete; missing metoclopramide HCl (REGLAN) and lipase-protease-amylase (*****) | metoclopramide HCl (REGLAN) 10 mg tablet... lipase-protease-amylase (*****) 12,000-38,000 -60,000 un |

*Extraction summary*: Major error in missing cancer-related medications and minor issue in incomplete supportive medications list.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence: 'to to better manage your pain' is missing critical information. | We reduced the doses of your medications, which helped improve how you felt. We  |

*Letter summary*: Letter contains an incomplete sentence that needs correction.

---

## ROW 54 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication *****. | He restarted chemotherapy with ***** in October 2014 and received 9 cycles in total. |
| P2 | Treatment_Changes | Missing mention of restarting chemotherapy with *****. | He restarted chemotherapy with ***** in October 2014 and received 9 cycles in total. |
| P2 | Treatment_Goals | Should specify 'palliative' as the goal, but it is implied. | The patient is showing modest disease progression. This is evidenced by the presence of multiple new |
| P2 | Response_Assessment | Could be more precise about the timing of the response assessment. | Most recent CT scans on 03/03/2015 showed minimal change/slight enlargement in his pancreatic tail m |
| P2 | Medication_Plan | Does not explicitly state that the patient is considering clinical trial options involving specific agents. | We discussed his various options at this point, not including resuming *****/***** which remain a vi |
| P2 | Therapy_plan | Does not explicitly state the specific agents being considered in the clinical trial options. | We discussed his various options at this point, not including resuming *****/***** which remain a vi |
| P2 | Procedure_Plan | Does not explicitly state the purpose of the biopsy. |  |
| P2 | Imaging_Plan | Should mention the recent imaging done post-visit. | Most recent CT scans on 03/03/2015 showed minimal change/slight enlargement in his pancreatic tail m |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing specific cancer-related medications and treatment details.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 55 — ⚠️ ISSUES

**Type**: Pancreatic head mass with invasive adenocarcinoma; Ampullary mass with invasive 
**Stage**: Stage IV (metastatic disease to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note mentions 'pancreatic head mass with invasive adenocarcinoma' and 'ampullary mass with invasive adenocarcinoma', but does not clearly specify which is the primary diagnosis. The extracted data should reflect the uncertainty. | Overall no consensus about whether ampullary, biliary, or pancreatic. |
| P2 | Treatment_Changes.recent_changes | The note indicates that the patient switched to capecitabine on 05/03/21, but the extracted data does not mention the date of the switch. | Switched to capecitabine on 05/03/21. |
| P2 | Treatment_Goals.goals_of_treatment | The note suggests that the treatment goal is palliative, but the extracted data does not explicitly state this. | Given his fairly rapid pace of progression despite 5FU, limited therapeutic options, I recommended a |
| P2 | Medication_Plan.medication_plan | The note mentions that the patient is currently on capecitabine, but the extracted data does not include this information. | He completed FOLFOX, and switched to capecitabine on 05/03/21. |
| P2 | Procedure_Plan.procedure_plan | The note mentions arranging for a liver biopsy, but the extracted data does not include this information. | I recommended a liver biopsy to confirm dx, and to send for molecular profiling to guide treatment. |

*Extraction summary*: Most fields are clean, but there are minor issues with the clarity of the cancer diagnosis, the timing of medication changes, and the inclusion of the liver biopsy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'liver tumors'. | Recent imaging shows new and enlarging liver tumors, indicating that the cancer  |
| P2 | Unexplained jargon 'capecitabine'. | The treatment with capecitabine is not working as well as hoped. |
| P2 | Unexplained jargon 'liver biopsy'. | A liver biopsy will be arranged to confirm the diagnosis and guide future treatm |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

---

## ROW 56 — ⚠️ ISSUES

**Type**: Pancreas adenocarcinoma
**Stage**: (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine, nivolumab, and Abraxane). | She is currently on a clinical trial involving CD40 agonistic monoclonal antibody, gemcitabine, and  |
| P2 | Treatment_Changes | Supportive medications listed are incorrect. Should include anti-nausea medications only. | Well managed with current ***** ***** Home antiemetics as needed Good oral hydration |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative', but the note does not explicitly state this. | Ok to proceed with cycle 5 day 8 at unchanged doses |
| P2 | Response_Assessment | The response assessment mentions radiographic improvement but does not specify the exact imaging modality used. | Imaging after 2 cycles showed radiographic improvement. |
| P2 | Medication_Plan | Should include specific cancer-related medications (gemcitabine, nivolumab, and Abraxane). | Ok to proceed with cycle 5 day 8 at unchanged doses |
| P2 | Therapy_plan | Should include specific cancer-related medications (gemcitabine, nivolumab, and Abraxane). | Ok to proceed with cycle  5 day 8 at unchanged doses |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and incomplete supportive care details.

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
| P2 | Response_Assessment | Minor inconsistency in describing the size of hepatic lesions. | Continued interval decrease in size of enhancing hepatic lesions, including 2.2 cm in the lateral se |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, treatment goals, and inconsistent lesion size descriptions.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'neuroendocrine tumor'. | You came in for a follow-up visit regarding your pancreatic neuroendocrine tumor |
| P2 | Unexplained medical jargon 'lymph nodes'. | Lymph nodes near the stomach are unchanged. |
| P2 | Unexplained medical jargon 'stable'. | Overall, your disease is stable. |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

---

## ROW 59 — ⚠️ ISSUES

**Type**: Well to moderately differentiated adenocarcinoma of the pancreas; Mucinous adeno
**Stage**: Likely advanced stage (likely or higher) due to positive retroperitoneal margin 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (methimazole for hyperthyroidism). | She was found to be hyperthyroid and she is now being treated with methimazole. |
| P2 | Treatment_Changes | Missing supportive medication (omeprazole for esophageal reflux). | In February 2022, she had a workup and was found to have esophageal reflux. This is being treated wi |
| P2 | Imaging_Plan | Incomplete. Should include 'CT Abdomen/Pelvis' in addition to 'CT Chest'. | She will continue on surveillance. We'll see her again in 6 months. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and an incomplete imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The letter should specify that the follow-up is for pancreatic cancer. | You came in for a follow-up visit regarding your cancer treatment. |
| P2 | The sentence is missing context about the specific CT scans mentioned. | There is no evidence of recurrent or metastatic disease within the abdomen and p |
| P2 | This sentence is missing context about the stability of the nodule and lymph node. | The 6mm groundglass nodule in the left upper lobe of the lung and the 6mm mesent |

*Letter summary*: The letter is mostly clean but requires minor adjustments for clarity and completeness.

---

## ROW 60 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: Locally advanced; no distant metastases mentioned

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications such as growth factor used for severe neutropenia. | He later experienced severe neutropenia and was started on growth factor. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete. Missing growth factor for neutropenia. | He later experienced severe neutropenia and was started on growth factor. |
| P2 | Treatment_Goals | The goal 'curative' might be misleading given the locally advanced nature of the cancer. 'Adjuvant' or 'risk reduction' might be more appropriate. | His elevated CA-19-9 could be a residual of his bout with cholangitis a couple of months ago but I'm |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the completeness of cancer-related medications and the appropriateness of the treatment goal classification.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 61 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Stage II B (pT3N1)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine and [REDACTED]-paclitaxel). | followed by adjuvant chemotherapy consisting of the combination of gemcitabine plus *****-paclitaxel |
| P2 | Treatment_Changes | Supportive medication (Neupogen) is mentioned, but the specific cancer-related medications (gemcitabine and [REDACTED]-paclitaxel) are not listed. | due to recurrent asymptomatic cytopenias, he has had to be dose reduced to dose level -2 of both che |
| P2 | Treatment_Goals | Goals of treatment should be 'adjuvant' rather than 'surveillance', given the context of ongoing chemotherapy. | He has now completed 6 cycles of treatment as of the end of December. |
| P2 | Medication_Plan | Inaccurate to state 'Continue/start: gemcitabine' when the patient has completed 6 cycles of treatment. | He has now completed 6 cycles of treatment as of the end of December. |
| P2 | Therapy_plan | Inaccurate to state 'Continue/start: gemcitabine' when the patient has completed  6 cycles of treatment. | He has now completed 6 cycles of treatment as of the end of December. |

*Extraction summary*: Most fields are clean, but there are minor issues with the inclusion of cancer-related medications and the classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'imaging'. | Recent imaging shows no signs of cancer returning. |
| P2 | Unexplained jargon 'nodule'. | A small nodule in your lungs is likely due to a mild reaction to your medication |
| P2 | Unexplained jargon 'tumor marker'. | Your CA 19-9 levels, a tumor marker, have stayed stable. |
| P2 | Incomplete sentence (missing the exact dose level). | Your chemotherapy dose was reduced to a lower level due to side effects. |

*Letter summary*: Letter contains minor readability issues and unexplained jargon.

---

## ROW 62 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: Originally pT3 N1, now metastatic (pt3 n1)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions 'Continuing abraxane.' | We will cont ***** ***** abraxane. |
| P2 | Treatment_Changes | The field only mentions starting a treatment, but the note also states continuing abraxane. | We will cont ***** ***** abraxane. |
| P2 | Treatment_Goals | The field states 'palliative,' but the note does not explicitly state the goal of treatment. | Not explicitly stated in the note. |
| P2 | Lab_Plan | The field states 'No labs planned,' but the note mentions reviewing labs per protocol. | Labs including the CBC and chemistry profile were reviewed per protocol. |

*Extraction summary*: Most fields are clean, but there are minor issues with current medications, treatment changes, treatment goals, and lab plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete and unclear sentence with unexplained jargon. | We will ask medical genetics to weigh in on the a medication of the ATM a medica |

*Letter summary*: Letter contains one incomplete and unclear sentence that needs clarification.

---

## ROW 63 — ⚠️ ISSUES

**Type**: Undifferentiated carcinoma of the tail of the pancreas involving the portal vein
**Stage**: Originally not specified, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) | She has now completed two full cycles of therapy. |
| P2 | Treatment_Changes | Supportive medications missing gemcitabine and Abraxane | She has now completed two full cycles of therapy. |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' but is not explicitly stated | Purpose of this visit is to review radiographic response and continue treatment. |
| P2 | Response_Assessment | Mixed response to treatment is mentioned but not explicitly stated as 'palliative' | Some hepatic metastases have increased in size while others have decreased. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the inclusion of cancer-related medications and clarification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'mixed response'. | The cancer is showing a mixed response to treatment. Some liver tumors have grow |
| P2 | Unexplained medical jargon 'nodule'. | You have a small nodule in your left lung that hasn't changed. |

*Letter summary*: Letter is mostly clean but contains minor readability issues with unexplained medical jargon.

---

## ROW 64 — ⚠️ ISSUES

**Type**: Invasive adenocarcinoma, moderately differentiated, with lymphovascular invasion
**Stage**: Originally pT3N0, now metastatic (pt3n0)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | 02/08/2018: Adjuvant Cycle 1 day 1 gemcitabine +***** paclitaxel. |
| P2 | Treatment_Changes | Supportive medications missing 'acetaminophen-codeine (TYLENOL #3)' used for pain. | acetaminophen-codeine (TYLENOL #3) 300-30 mg tablet |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative', but the note suggests discussion of treatment options including 5-[REDACTED] based therapy. | We discussed treatment options which include 5-***** based therapy |
| P2 | Response_Assessment | The response assessment mentions disease progression but does not specify the exact date of the last treatment cycle. | The cancer is not responding to previous treatment. The most recent surveillance scan on 08/15/19 sh |
| P2 | Medication_Plan | The medication plan mentions starting 5-[REDACTED] based therapy but does not specify the exact regimen. | The patient will start 5-[REDACTED] based therapy. |
| P2 | Therapy_plan | The therapy plan mentions addressing side effects and utility of the treatment but lacks specific details. | The plan includes addressing the side effects and utility of the treatment. |
| P2 | follow_up_next_visit | The next clinic visit is not specified in the note. | Not specified in the given note |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications, incomplete treatment plans, and unspecified follow-up visits.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical info. | You will receive information about the end-of-life act and some information to t |

*Letter summary*: Letter contains an incomplete sentence that requires clarification.

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
| P2 | Cancer_Diagnosis | Missing specific stage information. | Mr. ***** is a 63 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer |
| P2 | Current_Medications | Incomplete listing of current cancer-related medications. | He is currently receiving salvage chemotherapy with the combination of gemcitabine plus [REDACTED]-p |
| P2 | Treatment_Changes | Supportive medications are missing. | He continues to use [REDACTED] heparin for a previously diagnosed VTE, with no bleeding complication |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing specific stage information, incomplete listing of current cancer-related medications, and missing supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'hepatic'. | Recent imaging shows a mixed response to your treatment. The size of the pancrea |
| P2 | Slightly imprecise wording. The original note specifies palliative treatment goals. | Your treatment goal is to manage symptoms and improve quality of life. |

*Letter summary*: Letter contains minor readability and precision issues.

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
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage 'now metastatic ()' is imprecise and lacks specific stage information. | He now shows evidence of metastatic recurrence involving the abdominal wall, possibly representing s |
| P2 | Treatment_Changes.recent_changes | The date format is inconsistent with the rest of the note. | Switched to chemotherapy with the combination of gemcitabine plus cisplatin on 05/27/2022. |
| P2 | Treatment_Changes.supportive_meds | Supportive medication details are incomplete. Other supportive medications like pain management drugs are not mentioned. | ondansetron (ZOFRAN) 8 mg tablet Take 1 tab by mouth twice a day for 2 days after each chemo treatme |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage description, date format consistency, and completeness of supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Contains medical jargon that may be difficult for an 8th-grade reader to understand. | Recent imaging (CT-PET scan on 05/18/2022) revealed multiple hypermetabolic soft |
| P2 | Contains medical jargon that may be difficult for an 8th-grade reader to understand. | Radiation therapy (RT) is considered as a less morbid but less definitive option |

*Letter summary*: Letter contains some medical jargon that could be simplified for better readability.

---

## ROW 69 — ⚠️ ISSUES

**Type**: Adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication 'gemcitabine'. | This includes the combination of gemcitabine plus nab-paclitaxel on a standard three-week-on, one-we |
| P2 | Treatment_Changes | Missing cancer-related supportive medication 'nab-paclitaxel'. | This includes the combination of gemcitabine plus nab-paclitaxel on a standard three-week-on, one-we |
| P2 | Treatment_Goals | Goals of treatment should be 'risk reduction' or 'adjuvant' rather than 'palliative', given the context of a clinical trial. | She has been randomized to the arm containing [REDACTED]. This includes the combination of gemcitabi |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and the classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'medication-paclitaxel' is unclear and could be simplified further. | You will continue to receive the combination of gemcitabine plus a medication-pa |

*Letter summary*: Letter is mostly clean with minor readability improvements suggested.

---

## ROW 70 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma with direct invasion into multiple structures, 
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (gemcitabine plus [REDACTED]-paclitaxel) | He has remained on treatment to the present time. ... He has demonstrated a nice *****-9 biomarker d |
| P2 | Treatment_Changes | Missing recent treatment change (planned chemotherapy holiday) | At this point, while indefinite treatment with gemcitabine/*****-paclitaxel (or conceivably just gem |

*Extraction summary*: Most fields are clean, but there are minor issues with missing current cancer-related medication and recent treatment changes.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 71 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The field is empty, but the note suggests metastatic disease. | metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Distant_Metastasis | The field states 'Not sure', but the note implies distant metastasis. | metastatic pancreatic adenocarcinoma |
| P2 | Cancer_Diagnosis.Metastasis | The field states 'Not sure', but the note implies metastasis. | metastatic pancreatic adenocarcinoma |
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions gemcitabine. | gemcitabine |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions starting gemcitabine. | started on gemcitabine monotherapy |

*Extraction summary*: Most fields are clean, but there are minor issues with staging, current medications, and recent treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Unnecessary detail and jargon that may confuse the patient. | You have a 25.3 X 22.4mm round, hypoechoic, calcified, and solid mass in the hea |
| P1 | Incomplete sentence with missing critical information. | Your CA 19-9 levels, a tumor marker, have fluctuated, decreasing from 1,011 to 3 |
| P1 | Missing dose details and unclear abbreviation. | You will continue taking levothyroxine and lovenox /kg BID. |
| P1 | Incomplete sentence with missing critical information. | If you tolerate the current treatment, a medication-paclitaxel will be added wit |
| P1 | Unnecessary detail and jargon that may confuse the patient. | Supportive medications like ondansetron and prochlorperazine are being used to m |
| P1 | Missing context about what paracentesis is. | You are scheduled for a repeat paracentesis tomorrow, June 10. |

*Letter summary*: The letter contains several issues with unnecessary detail, jargon, and incomplete sentences that need to be addressed.

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
| P2 | Contains technical jargon that may be difficult for an 8th-grade reader to understand. | Your cancer is currently showing stable disease in the abdomen but progressing p |

*Letter summary*: Letter is mostly clean but contains some technical jargon that could be simplified further.

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
| P2 | Reason_for_Visit.summary | Inaccurate summary of the patient's treatment history. | She was started on combination therapy with gemcitabine and oxaliplatin plus rucaparib. She had an e |
| P1 | Current_Medications.current_meds | Missing gemcitabine and oxaliplatin in the past treatment regimen. | She was started on combination therapy with gemcitabine and oxaliplatin plus rucaparib. |
| P2 | Treatment_Changes.recent_changes | Should mention discontinuation of chemotherapy in October 2019. | In October 2019, chemotherapy was discontinued and she continues on rucaparib. |
| P2 | Treatment_Goals.goals_of_treatment | Goals of treatment should be more specific to the context of ongoing maintenance therapy. | We reassured them that there are oncologists was doing ***** ***** job in ***** ***** should remain  |

*Extraction summary*: Most fields are accurate, but there are minor issues with the summary of treatment history and goals of treatment, and a major issue with missing past chemotherapy regimens.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon. | You have metastatic (Stage IV) cancer that has spread to your peritoneum and liv |
| P2 | Unexplained medical jargon. | Future therapy may involve rechallenging with the same chemotherapy, potentially |

*Letter summary*: Letter is mostly clean but contains minor readability issues related to unexplained medical jargon.

---

## ROW 75 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma arising in association with an intraduc
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |
| P2 | Treatment_Changes | Missing 'filgrastim-sndz' (Neupogen) which is a cancer-related supportive medication. | filgrastim-sndz (*****) 480 mcg/0.8 mL injection syringe Inject 0.8 mLs (480 mcg total) under the sk |
| P2 | Lab_Plan | Incomplete. Should include 'CA 19-9 levels monthly'. | Monthly CA 19-9 levels have been wnl range. We will obtain CT chest, abdomen and pelvis after the co |

*Extraction summary*: Most fields are clean, but there are minor omissions in current medications and lab plans.

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
| P2 | Unexplained jargon 'nodules'. | Recent imaging shows an increase in the size and number of nodules in your lungs |
| P2 | Incomplete sentence, lacks specificity about the medication. | Your treatment with Atezolizumab + a medication was delayed due to high lab valu |

*Letter summary*: Letter contains minor readability and completeness issues.

---

## ROW 77 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the tail of the pancreas
**Stage**: Resectable at diagnosis, now with negative margins and lymph nodes

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary of the visit. It mentions 'follow-up visit for adjuvant therapy with gemcitabine and capecitabine', but the patient has already completed 6 cycles of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage description is imprecise. It should specify the stage at diagnosis rather than the current status. | Because of renal and splenic invasion, the left kidney and left adrenal and spleen were included in  |
| P2 | Treatment_Changes.recent_changes | Missing information about the completion of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Response_Assessment.response_assessment | The response assessment does not clearly state that the patient has completed adjuvant therapy. | He has had 6 cycles and is on surveillance. |

*Extraction summary*: Most fields are accurate, but there are minor issues with the summary of the visit, the stage description, and the treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained medical jargon 'negative margins'. | You have adenocarcinoma (cancer that started in gland cells) of the tail of the  |
| P2 | Unexplained medical jargon 'transaminitis'. | Your liver function tests show elevated AST (162 U/L) and ALT (274 U/L), indicat |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

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
| P2 | Unexplained medical jargon 'splenic artery' and 'splenic vein'. | You have a 3.7 x 3.6 cm mass in the body of your pancreas that encases the splen |
| P2 | Unexplained medical jargon 'alkaline phosphatase' and 'bilirubin'. | Laboratory results show elevated alkaline phosphatase and total bilirubin levels |

*Letter summary*: Letter is mostly clean but contains minor readability issues due to unexplained medical jargon.

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
| P2 | Current_Medications.current_meds | Incomplete. Should include 'capecitabine'. | capecitabine (XELODA) |
| P2 | Treatment_Changes.recent_changes | Slightly imprecise wording. Should mention the plan for a diagnostic biopsy. | We plan to do a diagnostic biopsy to better understand this. |
| P2 | Imaging_Plan.imaging_plan | Incomplete. Should mention the followup CT scan evaluation. | He is returning today for a followup CT scan evaluation. |
| P2 | Referral.follow up | Slightly imprecise wording. Should mention the plan for a diagnostic biopsy. | We plan to do a diagnostic biopsy to better understand this. |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, current medications, recent treatment changes, imaging plan, and referral follow-up.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Use of 'hepatic' would be more precise, but 'liver' is acceptable. However, 'areas' is vague and could be more specific. | There is an enlarging liver metastasis, now measuring up to 2.4 cm compared to 1 |
| P2 | This is a bit vague. More specific details could improve clarity. | The cancer is showing signs of progression. |
| P2 | It's unclear which study medication is being referred to. | You will be taken off the current study medication due to disease progression. |

*Letter summary*: Letter is mostly clean but could benefit from slight improvements in specificity and clarity.

---

## ROW 81 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary. The note mentions a second opinion scheduled with Dr. ***** ***** on 12/14/18. | Patient has appt with Dr. ***** ***** at ***** on 12/14/18 for a second opinion. |
| P2 | Imaging_Plan.imaging_plan | Incorrect imaging plan. The note does not mention any DEXA scan. | Recent Imaging None |
| P2 | Treatment_Changes.recent_changes | Incomplete. The note mentions the patient was started on [REDACTED] on 12/09/18, but the exact medication is not specified. | She is now s/p C1D1 ***** on 12/09/18 |
| P2 | Treatment_Changes.supportive_meds | Incomplete. The note mentions the patient received Emend and dexamethasone, but also mentions other supportive care measures such as IV hydration and antiemetics. | She received 2 L normal saline, *****, Emend, dexamethasone for nausea and vomiting today. |
| P2 | Medication_Plan.medication_plan | Incomplete. The note mentions recommending a trial of Lexapro, but this is not included in the medication plan. | Recommend a trial of Lexapro. |

*Extraction summary*: Most fields are accurate, but there are minor issues with the summary, imaging plan, treatment changes, supportive medications, and medication plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | You were taken off the clinical trial because the cancer is progressing. |
| P2 | Unexplained medical jargon. | There has been slight enlargement of the pancreatic mass and new lesions in the  |
| P2 | Unexplained medical jargon. | You started a new medication on 12/09/18. |

*Letter summary*: The letter contains some incomplete sentences and unexplained medical jargon that need clarification.

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
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) | He elected to start therapy with gemcitabine and Abraxane. He has now completed 2 cycles. |
| P2 | Treatment_Changes | Supportive medications listed are incomplete; missing ondansetron and prochlorperazine | ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet (8 mg total) by mouth every 8 (eight) hours as needed |
| P2 | Treatment_Goals | Goals of treatment should be more specific (palliative with consideration of pembrolizumab) | I think we need to hold his chemotherapy to see if the symptoms resolve... He is eligible for pembro |
| P2 | Medication_Plan | Inconsistent mention of gemcitabine and Abraxane; should specify holding them | Hold chemotherapy to see if symptoms resolve. Patient is eligible for pembrolizumab and can consider |
| P2 | Therapy_plan | Inconsistent mention of gemcitabine and Abraxane; should specify holding them | Hold chemotherapy to see if symptoms resolve. Eligible for pembrolizumab which can be considered at  |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications and inconsistent medication plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 84 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (atezolizumab and cobimetinib). | Now midway through cycle #2 of treatment on the crossover arm of the same [REDACTED] trial consistin |
| P2 | Treatment_Changes | Missing specific names of the drugs (atezolizumab and cobimetinib). | Switched to the combination of atezolizumab, a PD-L1 inhibitor (administered on a q2 weekly schedule |
| P2 | Lab_Plan | Should mention regular monitoring of lab results given the ongoing treatment. | No labs planned. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing current cancer-related medications and incomplete lab monitoring plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Use of 'hepatic segment 6' might be too technical for an 8th-grade reading level. | The cancer is currently showing stable disease with slightly enlarging measurabl |
| P2 | Use of 'post-pancreatectomy surgical bed' might be too technical. | Additionally, stable disease is observed in the post-pancreatectomy surgical bed |
| P2 | Use of 'PD-L1 inhibitor' and 'MEK inhibitor' might be too technical. | You switched to the combination of atezolizumab (a PD-L1 inhibitor) and cobimeti |
| P2 | Use of 'q2 weekly schedule' and '28-day cycle' might be too technical. | You are continuing on the crossover arm of the trial, consisting of atezolizumab |

*Letter summary*: Letter contains minor readability issues due to technical terms that could be simplified for an 8th-grade reading level.

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
| P2 | Treatment_Changes | Incomplete listing of supportive medications. Only ondansetron and prochlorperazine are listed, but the note mentions several other relevant medications such as lorazepam and zolpidem. | LORazepam (ATIVAN) 0.5 mg tablet TAKE 1 TABLET BY MOUTH SUBLINGUALLY EVERY 8 HOURS AS NEEDED FOR NAU |
| P2 | Therapy_plan | Inaccurate description of therapy plan. The plan includes chemotherapy, but 'compression stockings' is mentioned without context. | We reviewed the logistics, scheduling, need for Mediport placement (for which I will place referral  |

*Extraction summary*: Most fields are clean, but there are minor issues with incomplete listings of cancer-related medications and supportive medications, and an inaccurate description of the therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | There is an anticipated opening of a medication inhibitor trial for a medication |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 87 — ⚠️ ISSUES

**Type**: Pancreatic neuroendocrine tumor, well-differentiated
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing 'cholecalciferol, vitamin D3', 'lipase-protease-amylase', 'metFORMIN', 'multivitamin', 'omega-3 fatty acids-vitamin E', 'zolpidem'. However, these are general medications and not cancer-related. | Medication Sig Dispense Refill section lists several general medications. |
| P2 | Treatment_Changes | Missing specific details about recent changes in treatment, such as the increase in metformin dosage. | Recent metformin and exercise for higher sugars- now fine. Most recent blood sugar numbers averaged  |
| P2 | Treatment_Goals | The goal is stated as 'palliative', but the note mentions 'relatively stable to improved on everolimus'. This could imply a more nuanced goal beyond just palliation. | His disease progressed on octreotide ***** 30 mg/mo and he is relatively stable to improved on evero |
| P2 | Response_Assessment | The response is described as 'stable disease with some minimal progression', but the note also mentions 'slow PD'. This could be more precise. | My interpretation is that there is slow PD v December 2016. |
| P2 | Medication_Plan | Missing specific details about the increase in metformin dosage and monitoring of blood sugar levels. | Most recent blood sugar numbers averaged 117. |
| P2 | Procedure_Plan | Missing specific details about the need for follow-up MMA and B12. | #low B12 On oral replacement- -needs f/u MMA, and B12 |
| P2 | Imaging_Plan | Missing specific details about the need for follow-up H&P and chest CT. | Needs f/u H&P and chest CT 3 mo-- sooner if sx |
| P2 | Lab_Plan | Missing specific details about the need for follow-up MMA and B12. | Needs f/u MMA, and B12 |
| P2 | Referral | Missing specific details about referrals to local dermatology and primary care physician. | F/u local dermatology, Managed by PCP |

*Extraction summary*: Several minor issues related to missing details about treatment changes, medication plans, and follow-up actions. Overall, most fields are clean.

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
| P2 | Current_Medications.current_meds | The field is empty, but the note mentions ongoing chemotherapy with [REDACTED]. | Patient most recently started on 2nd line therapy with [REDACTED] on 02/28/19 after disease progress |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note indicates a change in treatment to [REDACTED] after progression. | Patient most recently started on 2nd line therapy with [REDACTED] on 02/28/19 after disease progress |
| P2 | Treatment_Changes.supportive_meds | The listed medications (ondansetron, prochlorperazine, sennosides) are not mentioned in the note. | None of these medications are mentioned in the note. |
| P2 | Medication_Plan.medication_plan | The plan includes '[REDACTED]' and 'fiber', but the note only mentions continuing Lovenox BID and fiber. | Continue Lovenox BID, Continue [REDACTED] and fiber |

*Extraction summary*: Most fields are clean, but there are minor issues related to the completeness of treatment changes, supportive medications, and the inclusion of non-mentioned medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon (CA 19-9). | Recent imaging shows that the cancer has grown and your CA 19-9 (a tumor marker) |
| P2 | Slightly imprecise wording. | Your physical exam shows that you look thin and sick, with a swollen belly and s |

*Letter summary*: Letter contains minor readability issues related to unexplained jargon and slightly imprecise wording.

---

## ROW 89 — ⚠️ ISSUES

**Type**: Well to moderately differentiated ductal adenocarcinoma of pancreatic head with 
**Stage**: Not explicitly stated in note, but described as resected node-positive pancreati

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine/albumin-bound paclitaxel). | Switched to the combination of gemcitabine plus albumin-bound paclitaxel and completed six 28 day cy |
| P2 | Treatment_Changes | Missing recent treatment change (switch to gemcitabine/albumin-bound paclitaxel). | Switched to the combination of gemcitabine plus albumin-bound paclitaxel and completed six 28 day cy |

*Extraction summary*: Most fields are clean, but there are minor omissions regarding recent cancer-related treatments.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'perineural invasion'. | You have a well to moderately differentiated ductal adenocarcinoma (cancer that  |
| P2 | Unexplained jargon 'acute high-grade small bowel obstruction'. | Imaging showed an acute high-grade small bowel obstruction, clusters of nodules  |
| P2 | Slightly imprecise wording. | Based on the surgical results, the cancer is not responding well to the previous |

*Letter summary*: Letter contains minor readability issues and unexplained jargon.

---

## ROW 90 — ⚠️ ISSUES

**Type**: Primary pancreatic neuroendocrine tumor; BRAF-mutant metastatic melanoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Summary omits mention of ongoing treatment with GSK-436+GSK-212. | currently on treatment with a combined regimen of GSK-436+GSK-212. |
| P2 | Cancer_Diagnosis.Type_of_Cancer | Missing mention of initial right heel melanoma and its progression. | diagnosed with right heel melanoma, status post wide excision and negative sentinel lymph node biops |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | Stage is not specified, though it can be inferred as metastatic. | Metastatic () |
| P1 | Current_Medications.current_meds | Missing current cancer treatments (dabrafenib and trametinib). | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Treatment_Changes.recent_changes | Does not mention recent changes in dabrafenib and trametinib regimen. | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Treatment_Goals.goals_of_treatment | Goals are only listed as 'palliative', missing 'symptom management'. | Per dr ***** |
| P2 | Response_Assessment.response_assessment | Does not mention the complete response to combination therapy. | noted to have complete response to combination therapy |
| P2 | Imaging_Plan.imaging_plan | Does not mention routine PETCT per clinical trial. | Continue to obtain routine PETCT per clinical trial |

*Extraction summary*: Several fields contain minor omissions or inaccuracies related to cancer treatments and goals, but no hallucinations or major contradictions.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Your dose of octreotide was reduced for four days, then increased for four days, |

*Letter summary*: Letter contains an incomplete sentence that needs to be completed with the final dosage details.

---

## ROW 91 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas, moderately differentiated
**Stage**: Resected with 8 of 23 lymph nodes positive

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions the patient has completed neoadjuvant therapy with gemcitabine and Abraxane. | He has completed 6 cycles of therapy. His last 2 cycles were given with alternate week schedule usin |
| P2 | Treatment_Changes | The field is empty, but the note mentions the patient has completed neoadjuvant therapy and had a surgical resection. | He has completed 6 cycles of therapy. He was taken to surgery on 11/27/2018. |
| P2 | Treatment_Goals | The field states 'curative', but the note suggests a more complex situation with possible local recurrence. | He may well have local recurrence but with the current coronavirus crisis, it will be difficult to w |
| P2 | Response_Assessment | The field states 'stable disease', but the note suggests uncertainty about possible local recurrence. | He may well have local recurrence but with the current coronavirus crisis, it will be difficult to w |
| P2 | Imaging_Plan | The field states 'No imaging planned', but the note suggests a need for further evaluation due to suspicious findings. | He may well have local recurrence but with the current coronavirus crisis, it will be difficult to w |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the completeness of treatment details and the clarity of the response assessment.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 92 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic adenocarcinoma with perineural and lymphova
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The extracted type of cancer is overly specific and not directly stated in the note. | The note mentions 'moderately differentiated adenocarcinoma' but does not specify 'perineural and ly |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The extracted stage is not fully accurate. | The note states 'originally borderline resectable, now metastatic', but the extracted stage includes |
| P2 | Lab_Results.lab_summary | The lab summary contains redacted values that are not present in the note. | The note does not provide specific lab values except for CA 19-9. |
| P2 | Clinical_Findings.findings | The extracted findings contain specific measurements and dates not mentioned in the note. | The note does not mention specific measurements like '3.1 x 2.8 cm' or dates like '08/28/2019'. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions ongoing treatment changes. | The note discusses ongoing treatment with gemcitabine and nab-paclitaxel. |
| P2 | Treatment_Goals.goals_of_treatment | The goal is listed as 'palliative', but the note suggests ongoing management with potential for a chemotherapy holiday. | The note indicates ongoing management and consideration for a chemotherapy holiday. |
| P2 | Response_Assessment.response_assessment | The response assessment contains specific measurements and dates not mentioned in the note. | The note does not mention specific measurements like '3.1 x 2.8 cm' or dates like '08/28/2019'. |
| P2 | Medication_Plan.medication_plan | The plan mentions trametinib, which is not a current medication but a future consideration. | Trametinib is mentioned as a future consideration, not a current medication. |
| P2 | Imaging_Plan.imaging_plan | The imaging plan mentions CA-119-9, which is incorrect. | The note mentions CA 19-9, not CA-119-9. |
| P2 | Lab_Plan.lab_plan | The lab plan mentions CA-119-9, which is incorrect. | The note mentions CA 19-9, not CA-119-9. |

*Extraction summary*: Most fields are clean, but there are minor issues with specificity, temporal accuracy, and factual consistency.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'slightly elevated' is somewhat ambiguous and could be clearer. | Your CA 19-9 (a tumor marker) level is 207, which is slightly elevated. |
| P2 | This sentence is vague and does not provide enough detail about what a 'chemotherapy holiday' entails. | Future considerations include incorporating a chemotherapy holiday at some point |

*Letter summary*: Letter is mostly clean but requires minor clarifications for better understanding.

---

## ROW 93 — ⚠️ ISSUES

**Type**: Adenocarcinoma involving the pancreatic head; metastatic adenocarcinoma consiste
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing specific cancer-related medications. | She underwent an U/S-guided LN biopsy on 08/22/2020, which confirmed metastatic adenocarcinoma, cons |
| P2 | Treatment_Changes | Supportive medications listed are not fully aligned with the note. | For pain management: continues with long-acting morphine which is providing good pain control. |
| P2 | Treatment_Goals | Goals of treatment should be more specific to the context of metastatic pancreatic cancer. | In summary, Ms. ***** is a 39 y.o. female with pancreatic cancer w/*****-***** *****, developed in t |
| P2 | Medication_Plan | Specific cancer-related medications are not detailed. | Continue chemotherapy with a reduced dose of irinotecan starting from cycle #2. Plan to switch to a  |

*Extraction summary*: Most fields are clean, but there are minor issues with missing specific cancer-related medications and supportive care details.

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
| P2 | Imaging_Plan.imaging_plan | Imaging plan is missing, though it could be inferred that surveillance imaging will continue. | He'll continue on surveillance. I'll see him again in 8 weeks for follow-up. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication details, treatment changes, and imaging plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 95 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma, probable second primary
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions the patient has been started on a medication (though redacted). | She has been started on ***** and has received 4 cycles. |
| P2 | Treatment_Changes | The field does not specify the name of the medication, only that it has been started and 4 cycles have been given. | She has been started on ***** and has received 4 cycles. |
| P2 | Procedure_Plan | The value 'Hopefully' is vague and does not clearly state the plan for surgical resection. | Hopefully, he will be comfortable with attempting a surgical resection. |

*Extraction summary*: Most fields are clean, but there are minor issues with the current medications, treatment changes, and procedure plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information (specific medication name). | You have started on a new medication and have received 4 cycles. |

*Letter summary*: Letter contains an incomplete sentence with missing critical information.

---

## ROW 96 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma with peritoneal carcinomatosis
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing cancer-related medication (gemcitabine). | She was started on therapy with gemcitabine + capecitabine on 10/20/16. |
| P1 | Treatment_Changes | Incorrect statement about recent changes. The patient switched to gemcitabine monotherapy, not from gemcitabine + capecitabine. | We decided that we would try gemcitabine monotherapy for one more dose and re-evaluate after that. |
| P1 | Therapy_plan | Incorrect statement about continuing capecitabine. The plan is to try gemcitabine monotherapy. | We decided that we would try gemcitabine monotherapy for one more dose and re-evaluate after that. |

*Extraction summary*: Major errors in current medications and treatment changes. Other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'changes in your kidneys and lungs'. | Imaging shows that the cancer has spread further, with fluid buildup in your bel |
| P2 | Unexplained jargon 'CA 19-9'. | Your CA 19-9 (a tumor marker) level has increased, indicating that the cancer is |
| P2 | Unexplained jargon 'ondansetron', 'methadone', 'hydrocodone'. | You will continue to take supportive medications like ondansetron, methadone, an |

*Letter summary*: Letter contains minor readability issues with unexplained medical jargon.

---

## ROW 97 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Locally advanced disease

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine). | We started her on an alternate week fixed dose rate gemcitabine with capecitabine. |

*Extraction summary*: One minor issue identified: missing gemcitabine in current medications.

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
| P2 | Current_Medications | Missing cancer-related medication (modified *****). | She was started on modified *****. She has had 3 cycles of therapy. |
| P2 | Treatment_Changes | Missing detail about initial chemotherapy regimen. | She was started on modified *****. She has had 3 cycles of therapy. |
| P2 | Treatment_Goals | Inaccurate to state only 'palliative'. Should include symptom control. | The family is questioning whether radiation is the right route to go. I really do not feel that she  |
| P2 | Response_Assessment | Missing detail about local progression. | When we last saw her, her CT scan showed local progression only. |
| P2 | radiotherapy_plan | Incomplete. Should mention scheduling of fiducials and start date of radiation. | She is scheduled for placement of fiducials tomorrow. Her radiation is not scheduled to begin until  |

*Extraction summary*: Most fields are clean, but there are minor omissions and slight inaccuracies in several areas.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Grammatical error: 'You is' should be 'You are'. | You is scheduled for placement of fiducials tomorrow and radiation is not schedu |

*Letter summary*: Letter contains a grammatical error that needs correction.

---

## ROW 100 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas
**Stage**: Stage IV (metastatic to liver)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane are recommended but not listed as current medications). | We are recommending gemcitabine and Abraxane. |
| P2 | Treatment_Changes | Supportive medications listed include oxyCODONE, which is not specifically cancer-related. | Zofran, oxyCODONE (ROXICODONE) 5 mg tablet |
| P2 | Treatment_Goals | Goals of treatment are stated as 'palliative', but the note does not explicitly state this goal. | We are recommending gemcitabine and Abraxane. |
| P2 | Response_Assessment | The response assessment mentions that the cancer is not responding to treatment, but the note does not explicitly state this. | Imaging findings show an interval increase in size of the hypoattenuating mass centered in the pancr |
| P2 | Medication_Plan | The medication plan mentions starting gemcitabine and Abraxane, but the patient is still considering this option. | He is recommended to start gemcitabine and Abraxane. He is considering this option or sticking with  |
| P2 | Therapy_plan | The therapy plan mentions starting gemcitabine and Abraxane, but the patient is still considering this option. | He is recommended to start gemcitabine and Abraxane. |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the inclusion of non-cancer-related medications and the clarity of the treatment plan and goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'imaging'. | The cancer is not responding to treatment. Imaging shows the tumor in your pancr |
| P2 | Unexplained jargon 'gemcitabine', 'Abraxane'. | We are recommending a change in your treatment to gemcitabine and Abraxane. You  |

*Letter summary*: Letter is mostly clean but contains minor readability issues with unexplained medical jargon.

---

