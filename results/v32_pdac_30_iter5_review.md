# Auto Review: results.txt

Generated: 2026-04-28 21:55
Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)

## Summary

- **Samples**: 30
- **Clean**: 0/30
- **P0** (hallucination): 0
- **P1** (major error): 25
- **P2** (minor issue): 143

### Critical Issues

- **ROW 4** [P1]: Incomplete sentence, missing information about other medications like dexamethasone, ondansetron, etc.
- **ROW 6** [P1]: Missing current cancer-related medication (lanreotide)
- **ROW 6** [P1]: Missing recent treatment change (start of lanreotide)
- **ROW 7** [P1]: Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III.
- **ROW 14** [P1]: Inaccurate description of the tumor location. The tumor is of duodenal/ampullary origin, not pancreatic.
- **ROW 15** [P1]: Inaccurate description of the progression of cancer. It mentions 'growth of tumors in this region' but does not specify that these are peritoneal metastases.
- **ROW 29** [P1]: Inaccurate statement; the patient is a new patient with a new diagnosis.
- **ROW 31** [P1]: Missing cancer-related medication (gemcitabine).
- **ROW 31** [P1]: This sentence contains medical jargon that may be confusing to an 8th-grade reader.
- **ROW 32** [P1]: This sentence contains medical jargon that is not explained and may be difficult for an 8th-grade reader to understand.
- **ROW 32** [P1]: The brand names 'Zofran', 'morphine', and 'Percocet' may be unfamiliar to the patient.
- **ROW 32** [P1]: The term 'standard of care' is not explained and may be confusing.
- **ROW 32** [P1]: The term 'Mediport' is not explained and may be unfamiliar.
- **ROW 32** [P1]: Terms like 'osseous metastases', 'systemic therapy', and 'spot radiation therapy' are not explained and may be confusing.
- **ROW 33** [P1]: Inaccurate representation of the clinical findings. The original note states that the cancer in the liver is similar in size and appearance, not stable.
- **ROW 35** [P1]: Unexplained jargon 'hepatic lesions'.
- **ROW 36** [P1]: Incomplete sentence with missing critical information.
- **ROW 40** [P1]: The term 'fine-needle aspiration' may be too technical for an 8th-grade reading level.
- **ROW 43** [P1]: Incomplete sentence with missing critical information.
- **ROW 72** [P1]: Missing current cancer-related medication (gemcitabine and Abraxane)
- **ROW 72** [P1]: Contains complex medical jargon that may be difficult for an 8th-grade reader to understand.
- **ROW 87** [P1]: Inaccurate information. Octreotide is not mentioned as a current medication in the extracted data.
- **ROW 90** [P1]: The note mentions both primary pancreatic neuroendocrine tumor and metastatic BRAF mutant melanoma, but the extracted data only lists 'Primary pancreatic neuroendocrine tumor'.
- **ROW 90** [P1]: The note indicates the patient is on intermittent dabrafenib and trametinib for melanoma, but this is not reflected in the extracted data.
- **ROW 90** [P1]: Incomplete sentence with missing critical information.

---

## ROW 1 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic adenocarcinoma
**Stage**: Potentially resectable with a regional LN involvement

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (palonosetron). | He initiated ***** on 03/06/2022, once his LFTs had sufficiently normalized post-stent, and is now s |
| P2 | Lab_Results | Missing initial CA 19-9 value (77,736). | Baseline *****-9 markedly elevated at 77,736 (in the context of obstructive jaundice) |
| P2 | Treatment_Changes | Missing mention of palonosetron. | He initiated ***** on 03/06/2022, once his LFTs had sufficiently normalized post-stent, and is now s |
| P2 | Treatment_Goals | Inaccurate goal description. Should include 'potentially curative' rather than just 'curative'. | We reviewed his most recent imaging studies which show, quite encouragingly, resolution of his aorto |
| P2 | Response_Assessment | Missing mention of the drop in CA 19-9 from 77,736 to 6,105. | Baseline *****-9 markedly elevated at 77,736 (in the context of obstructive jaundice) |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing specific cancer-related medications and lab results.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 4 — ⚠️ ISSUES

**Type**: Metastatic pancreatic ductal adenocarcinoma
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine-based regimen mentioned as a potential future treatment, but not listed under current medications). | We talked about potential side fx and risks, esp in terms of infectious complications, and adverse i |
| P2 | Treatment_Changes | Supportive medications listed are not exclusively cancer-related supportive care. | Supportive medications listed include Fentanyl patch, Dilaudid, DexAMETHasone, ondansetron, which ar |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication and non-exclusive listing of supportive medications.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence, missing information about other medications like dexamethasone, ondansetron, etc. | You are currently on a combination of long- and short-acting opioid analgesics f |

*Letter summary*: Letter has one minor completeness issue.

---

## ROW 6 — ⚠️ ISSUES

**Type**: Well-differentiated neuroendocrine tumor, grade 1, with lymphovascular and perin
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing current cancer-related medication (lanreotide) | continue lanreotide 120 mg/mo |
| P1 | Treatment_Changes | Missing recent treatment change (start of lanreotide) | continue lanreotide 120 mg/mo |
| P2 | Response_Assessment | Inaccurate statement; response assessment is available from imaging findings. | Imaging findings from 10/05/2015 PET-CT and MRI show interval increase in size of arterially enhanci |

*Extraction summary*: Major issues with missing current cancer-related medication and treatment changes. Minor issue with inaccurate response assessment.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 7 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic ductal adenocarcinoma
**Stage**: pT3 N1

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. The note mentions metformin, methylphenidate, ondansetron, and oxycodone, but only ondansetron and oxycodone are listed under supportive_meds. However, only cancer-related medications should be included in current_meds. | Medications the patient states to be taking prior to today's encounter. METFORMIN HCL (METFORMIN ORA |
| P2 | Treatment_Changes | Incomplete. The note mentions that the patient is currently receiving adjuvant chemotherapy with gemcitabine alone after discontinuing [REDACTED]-paclitaxel due to cumulative peripheral sensory neuropathy. The extraction does not mention the initial combination therapy. | He started with the combination of gemcitabine plus [REDACTED]-paclitaxel, but due to cumulative neu |
| P2 | Treatment_Goals | Inaccurate. The goal is adjuvant therapy, not curative. The note mentions that the patient is receiving adjuvant chemotherapy. | He started with the combination of gemcitabine plus [REDACTED]-paclitaxel, but due to cumulative neu |
| P2 | Response_Assessment | Inaccurate. The response assessment should include the patient's current condition and the absence of disease recurrence, not just the improvement in labs. | The patient's labs are noted to be significantly improved, allowing continuation of gemcitabine. The |
| P2 | Medication_Plan | Incomplete. The note mentions that the patient is currently receiving adjuvant chemotherapy with gemcitabine alone after discontinuing [REDACTED]-paclitaxel due to cumulative peripheral sensory neuropathy. The extraction does not mention the initial combination therapy. | He started with the combination of gemcitabine plus [REDACTED]-paclitaxel, but due to cumulative neu |
| P2 | Therapy_plan | Incomplete. The note mentions that the patient is currently receiving adjuvant chemotherapy with gemcitabine alone after discontinuing [REDACTED]-paclitaxel due to cumulative peripheral sensory neuropathy. The extraction does not mention the initial combination therapy. | He started with the combination of gemcitabine plus [REDACTED]-paclitaxel, but due to cumulative neu |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and accuracy in medication and treatment plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate staging information. The correct stage is pT3 N1, not Stage II-III. | You have a well-differentiated pancreatic ductal adenocarcinoma (cancer that sta |
| P2 | Unnecessary detail about the date of the CT scan. | Recent CT scan on 10/21/2015 shows post-operative changes without evidence of re |
| P2 | Lacks clarity on why the previous medication was stopped. | You are currently receiving adjuvant chemotherapy with gemcitabine alone. |

*Letter summary*: The letter contains inaccuracies in staging information and lacks clarity on medication changes. Other minor readability issues are present.

---

## ROW 14 — ⚠️ ISSUES

**Type**: Grade 3 neuroendocrine tumor of duodenal/ampullary origin
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. Only 'capecitabine' is listed, while 'temozolomide' is also a cancer-related medication. | She is now s/p 5 cycles of chemotherapy consisting of the combination of capecitabine/temozolomide. |

*Extraction summary*: One minor issue found in Current_Medications. All other fields are clean.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of the tumor location. The tumor is of duodenal/ampullary origin, not pancreatic. | You came in for a follow-up visit regarding your neuroendocrine tumor of the pan |
| P2 | Minor readability issue. Could be more explicit about what 'stable' means. | MRI scans from early June showed that your disease is stable. |
| P2 | Could be more specific about what was checked during the physical exam. | During your physical exam, no new issues were found. |
| P2 | Unexplained jargon. Ondansetron and oxycodone could be explained briefly. | You will continue to take ondansetron and oxycodone to manage side effects. |
| P2 | Could be more explicit about the purpose of the CT scan. | After this cycle, you will have a CT scan to assess your tumor. |

*Letter summary*: The letter contains inaccuracies and minor readability issues that need addressing.

---

## ROW 15 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is listed as 'now metastatic ()', which is imprecise and could be better specified. | This is a very pleasant 35-year-old Hispanic male who was diagnosed in October 2015 with metastatic  |
| P2 | Treatment_Changes.recent_changes | The recent changes mention resuming an unspecified agent but do not specify what was previously used. | He responded initially quite well to ***** but because of his residual neuropathy, I am concerned ab |
| P2 | Imaging_Plan.imaging_plan | The imaging plan mentions 'CT Chest' but does not specify the purpose or timing. | Report dictated by: ***** ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology  |

*Extraction summary*: Most fields are clean, but there are minor issues with the stage specification, recent treatment changes, and imaging plan details.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate description of the progression of cancer. It mentions 'growth of tumors in this region' but does not specify that these are peritoneal metastases. | There has been a progression of cancer in the lining of your abdomen (belly area |
| P2 | Unexplained jargon 'cancer spread'. | However, there is no sign of cancer spread in your lungs. |
| P2 | Unexplained jargon 'spot'. | A previous spot in your lung seems to be getting better, possibly due to an infe |
| P2 | Vague reference to 'a medication'. | We will resume a medication. You initially responded well to a medication, but w |

*Letter summary*: The letter contains minor inaccuracies and readability issues that need addressing.

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

*Extraction summary*: Most fields are clean, but there are minor issues regarding missing ongoing chemotherapy and an inaccurate imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'cancer spot' is somewhat vague and could be clearer. | The CT scan of your abdomen and pelvis shows that a small cancer spot in your li |
| P2 | The term 'tumor marker' might be unfamiliar to some readers. | Your CA 19-9, a tumor marker, has gone down from 13,468 to 4,187. |

*Letter summary*: Letter is mostly clean with minor readability issues.

---

## ROW 18 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma, 2.1 cm, invading duodenal submucosa, negative 
**Stage**: pT2N0

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | Medication Instructions:  amLODIPine (NORVASC) 5 mg, Oral, Every Morning Before Breakfast Scheduled |
| P2 | Treatment_Changes | Supportive medications listed are not fully supported by the note. | Some cumulative treatment-related peripheral sensory neuropathy, decreased appetite and dysgeusia. S |
| P2 | Lab_Results | Inaccurate representation of lab results. | Recent lab results show elevated CA 19-9 levels, with the most recent value being 1,077 U/mL on 11/1 |
| P2 | Treatment_Goals | Goals of treatment should be 'palliative' but are not explicitly stated. | We discussed the implications of these findings and how they should inform the next steps and timing |
| P2 | Radiotherapy_plan | Incomplete description of radiotherapy plan. | I will refer him to our Rad Onc team here at ***** for this possibility. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, incomplete supportive medication listing, inaccurate lab results, unclear treatment goals, and incomplete radiotherapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The sentence is slightly imprecise. It doesn't specify that the cancer has progressed in the form of pulmonary metastases. | Recent imaging shows that the cancer in your lungs has grown slightly. |
| P2 | The sentence is slightly imprecise. It doesn't mention the specific values or the significance of the increase. | Your CA 19-9 (a tumor marker) level has also increased. |
| P2 | The sentence is slightly imprecise. It doesn't specify the nature of the side effects. | You will take a break from treatment to let your body recover from side effects. |
| P2 | The sentence is slightly imprecise. It doesn't specify the purpose of the radiation therapy. | You will also be referred to the Radiation Oncology team to see if you can get r |

*Letter summary*: The letter contains minor readability issues and slightly imprecise wording.

---

## ROW 21 — ⚠️ ISSUES

**Type**: Metastatic pancreatic cancer
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (5-FU/LV plus irinotecan) | On this basis, he initiated 2nd-line chemotherapy consisting of the combination of 5-*****/LV plus * |
| P2 | Treatment_Changes | Supportive medication (lorazepam) is missing | LORazepam (ATIVAN) 0.5 mg tablet Take 1 tablet (0.5 mg total) by mouth every 6 (six) hours as needed |
| P2 | Therapy_plan | Incorrectly states 'Continue/start: irinotecan'. The patient is no longer a candidate for further treatment. | Therefore we had a ***** goals of care discussion in which I recommended that he refocus his goals p |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing cancer-related medications and incorrect continuation of therapy plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 29 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | Inaccurate summary; mentions 'new diagnosis' instead of 'follow-up'. | This is an independent visit. |
| P2 | Current_Medications.current_meds | Incomplete; only lists Gemcitabine and Abraxane, but does not include other cancer-related medications like Zoledronic acid. | She elected to go on the ***** 301 trial. |
| P2 | Treatment_Changes.recent_changes | Incomplete; only lists Gemcitabine and Abraxane, but does not include other cancer-related medications like Zoledronic acid. | She elected to go on the ***** 301 trial. |
| P2 | Treatment_Changes.supportive_meds | Incomplete; includes only some supportive medications, missing others like Ritalin and Mirtazapine. | Continue Ritalin BID. Followed by SMS. Continue on Mirtazapine 115mg qHS. Followed by SMS. |
| P2 | Medication_Plan.medication_plan | Incomplete; does not include Ritalin and Mirtazapine. | Continue Ritalin BID. Followed by SMS. Continue on Mirtazapine 15mg qHS. Followed by SMS. |
| P2 | Therapy_plan.therapy_plan | Incomplete; does not include Ritalin and Mirtazapine. | Continue Ritalin BID. Followed by SMS. Continue on Mirtazapine 15mg qHS. Followed by SMS. |

*Extraction summary*: Most fields are clean, but there are minor issues with the summary, current medications, and supportive medications lists.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate statement; the patient is a new patient with a new diagnosis. | You came in for a follow-up visit regarding your pancreatic cancer treatment. |
| P2 | Slightly imprecise wording; it doesn't mention the unchanged hepatic hypodensities. | On CT scans, there was a significant decrease in the size of tumors in the pancr |

*Letter summary*: The letter contains inaccuracies and slight imprecision that need addressing.

---

## ROW 31 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing cancer-related medication (gemcitabine). | We initially saw him two months ago. Because of his comorbidities and poor performance status, we re |
| P2 | Treatment_Changes | Supportive medication listed incorrectly as a change. | prochlorperazine (COMPAZINE) 5 mg tablet Take 1 tablet (5 mg total) by mouth every 6 (six) hours as  |

*Extraction summary*: Major error in missing gemcitabine under current medications. Minor issue with listing prochlorperazine as a change.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | This sentence contains medical jargon that may be confusing to an 8th-grade reader. | Your white blood cell count is high, and your red blood cell count, hemoglobin,  |

*Letter summary*: The letter contains minor readability issues that could confuse the patient.

---

## ROW 32 — ⚠️ ISSUES

**Type**: Metastatic moderately differentiated adenocarcinoma of pancreatic or biliary ori
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication (gemcitabine) mentioned in the note. | There are several possible chemotherapy options for metastatic pancreatic cancer, with the choice of |
| P2 | Treatment_Changes | Supportive medications listed are not exclusively cancer-treatment-related. | Supportive medications include primary prophylaxis with growth factor support ([REDACTED] or Neupoge |
| P2 | Treatment_Goals | Goals of treatment are not explicitly stated as palliative in the note. | The mainstay of treatment at this point should consist of systemic therapy, that the goals of such t |
| P2 | Medication_Plan | Missing specific mention of gemcitabine or FOLFIRINOX in the plan. | The patient is inclined to move ahead with standard of care (SOC) chemotherapy with [REDACTED]. |

*Extraction summary*: Most fields are clean, but there are minor issues related to missing specific cancer-related medications and imprecise classification of treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | This sentence contains medical jargon that is not explained and may be difficult for an 8th-grade reader to understand. | You have a 6 cm irregular mass in the body of your pancreas with encasement of t |
| P1 | The brand names 'Zofran', 'morphine', and 'Percocet' may be unfamiliar to the patient. | You are being prescribed supportive medications for nausea and pain, including Z |
| P1 | The term 'standard of care' is not explained and may be confusing. | You are inclined to move ahead with standard of care (standard) chemotherapy. |
| P1 | The term 'Mediport' is not explained and may be unfamiliar. | You are scheduled to have a Mediport placed through your local oncologist to fac |
| P1 | Terms like 'osseous metastases', 'systemic therapy', and 'spot radiation therapy' are not explained and may be confusing. | If the osseous metastases in your sacral area do not respond to systemic therapy |

*Letter summary*: The letter contains several sentences with medical jargon that may be difficult for an 8th-grade reader to understand. Simplifications are suggested to improve readability.

---

## ROW 33 — ⚠️ ISSUES

**Type**: Metastatic pancreatic adenocarcinoma with biopsy-proven liver metastasis
**Stage**: Metastatic (Stage IV)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Lab_Results | The lab results include values not mentioned in the note. | The note does not provide specific lab values such as Alkaline Phosphatase, Bilirubin, Alanine trans |
| P2 | Current_Medications | The field is empty, but the note mentions several cancer-related medications. | The note mentions '*****' (likely a chemotherapy regimen) and 'irinotecan'. |
| P2 | Treatment_Changes | The field incorrectly states 'Switch to gemcitabine combined with nab-paclitaxel', which is part of the future plan, not a past change. | The note states 'my recommendation is to switch her therapy to gemcitabine combined with *****-pacli |
| P2 | Therapy_plan | The field incorrectly states 'Continue irinotecan', which is not part of the future plan. | The note recommends switching to gemcitabine combined with nab-paclitaxel. |

*Extraction summary*: Most fields are clean, but there are minor issues with lab results, current medications, treatment changes, and therapy plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate representation of the clinical findings. The original note states that the cancer in the liver is similar in size and appearance, not stable. | The CT scan of your abdomen and pelvis shows that the cancer in your pancreas ha |
| P2 | Slightly imprecise wording. The original note mentions specific tumor marker levels but does not explicitly state that they indicate active cancer. | Your blood tests show high levels of tumor markers, indicating the cancer is sti |
| P2 | Slightly imprecise wording. The original note specifies the exact dosing schedule (days 1, 8, 15 out of a 28-day cycle). | You will receive gemcitabine and nab-paclitaxel every 28 days. |

*Letter summary*: The letter contains minor inaccuracies and imprecise wording that need correction.

---

## ROW 35 — ⚠️ ISSUES

**Type**: Adenocarcinoma of pancreatic tail
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions a biopsy revealing adenocarcinoma, suggesting the patient might be on initial cancer-related medications. | Biopsy of the pancreatic mass revealed adenocarcinoma. |
| P2 | Treatment_Changes | The field is empty, but the note discusses potential treatment plans and supportive medications. | The patient will receive antiemetics and anti-diarrheal medications prior to the initiation of chemo |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication and treatment changes.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Unexplained jargon 'hepatic lesions'. | You have pancreatic adenocarcinoma (cancer that started in gland cells) with fou |
| P2 | Unexplained jargon 'CA 19-9'. | You will be monitored with monthly CA 19-9 levels and restaging CT scans every 2 |
| P2 | Unexplained jargon 'hereditary pancreatic cancer syndrome'. | We will request assistance from our genetic counselors for evaluation of a possi |

*Letter summary*: The letter contains minor readability issues related to unexplained medical jargon.

---

## ROW 36 — ⚠️ ISSUES

**Type**: Nonfunctioning pancreatic neuroendocrine tumor
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. Everolimus is listed, but the note mentions multiple doses (2.5 mg and 10 mg). | everolimus (AFINITOR) 2.5 mg TAB 7.5 mg (3 tabs) daily  90 tablet  6    everolimus 10 mg TAB Take 1 |
| P2 | Treatment_Changes | Incomplete. The note mentions holding everolimus during partial SBO, but this detail is not fully captured. | Everolimus held during that period of time, resumed two weeks ago. |
| P2 | Treatment_Goals | Slightly imprecise. 'palliative' is correct, but 'durable disease control' might be a more precise term given the context. | with evidence of good durable disease control and reasonably good (in fact, improved) tolerance on c |
| P2 | Medication_Plan | Slightly imprecise. The note mentions starting non-insulin pharmacologic treatment in the future, but this is not clearly stated. | discussed potential plan to start non-insulin pharmacologic rx in the future. |

*Extraction summary*: Most fields are clean, but there are minor issues related to completeness and precision in medication details and treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Some liver lesions have grown slightly, while others have shrunk. |

*Letter summary*: The letter contains an incomplete sentence that requires clarification.

---

## ROW 40 — ⚠️ ISSUES

**Type**: Metastatic adenocarcinoma of the pancreas
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications (gemcitabine and Abraxane) in the medication plan. | We recommend a course of treatment with gemcitabine and Abraxane. |

*Extraction summary*: One minor issue identified regarding the inclusion of cancer-related medications in the medication plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | The term 'fine-needle aspiration' may be too technical for an 8th-grade reading level. | You have a mass in the pancreas and liver lesions. Fine-needle aspiration of one |
| P2 | The term 'CA 19-9' might be confusing for some readers. | Your CA 19-9 level on October 05 was 17,035. |
| P2 | The term 'hepatosplenomegaly' is too technical. | Physical exam shows you are generally tender in the abdomen, with no hepatosplen |
| P2 | The term 'pedal pulses' might be confusing. | There is mild edema in the right lower extremity with full pedal pulses. |

*Letter summary*: The letter contains some technical terms that may be confusing for an 8th-grade reading level audience.

---

## ROW 41 — ⚠️ ISSUES

**Type**: Invasive adenocarcinoma of pancreatic head, well- to moderately differentiated, 
**Stage**: pT3N1 (Isolated tumor cells present in one lymph node out of 36 examined), now w

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medication information. | He is continuing with an insulin regimen and regular use of enzyme supplements with meals. |
| P2 | Treatment_Changes | Missing supportive medication information. | He is continuing with an insulin regimen and regular use of enzyme supplements with meals. |
| P2 | Imaging_Plan | Note suggests further evaluation with PET/CT may be helpful for confirmation, but imaging plan states 'No imaging planned.' | Further evaluation with PET/CT may be helpful for confirmation. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medication information, supportive medication information, and an inconsistent imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | Unexplained jargon 'ill-defined soft tissue' and 'superior mesenteric artery (SMA)' | There is an increase in ill-defined soft tissue measuring up to 2.1 x 1.7 cm sur |

*Letter summary*: Letter is mostly clean but contains some unexplained medical jargon that could be simplified for better readability.

---

## ROW 43 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Borderline resectable pancreatic adenocarcinoma due to <180 degrees of vascular 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing cancer-related medications. | The note mentions 'Continue to monitor [REDACTED] July 04 [REDACTED] 4 weeks.' and 'Continue [REDACT |

*Extraction summary*: Most fields are clean, but the current medications related to cancer treatment are missing.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | one of the chemotherapy drugs was stopped. |

*Letter summary*: Letter is mostly clean but requires clarification on which chemotherapy drug was stopped.

---

## ROW 59 — ⚠️ ISSUES

**Type**: Well to moderately differentiated adenocarcinoma of pancreatic origin with pulmo
**Stage**: Not explicitly mentioned in note, but with pulmonary recurrence documented in De

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note does not explicitly mention the stage of cancer, but the extraction infers a progression based on the pulmonary recurrence. | With pulmonary recurrence documented in December 2017, likely progressed to |
| P2 | Current_Medications.current_meds | The current medications section is empty, but the note mentions several relevant medications such as propylthiouracil (PTU) for hyperthyroidism, which is cancer-related. | propylthiouracil (PTU) 50 mg tablet Take 25 mg by mouth. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note mentions ongoing treatment for hyperthyroidism with propylthiouracil (PTU). | propylthiouracil (PTU) 50 mg tablet Take 25 mg by mouth. |
| P2 | Imaging_Plan.imaging_plan | The plan only mentions 'CT Chest', but the note indicates ongoing surveillance with both CT Chest and CT Abdomen/Pelvis. | We'll see her again in 6 months. |

*Extraction summary*: Most fields are clean, but there are minor issues with the inferred cancer stage, missing current cancer-related medications, and incomplete imaging plan.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 62 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma
**Stage**: pt3 n1, now metastatic (pt3 n1)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (abraxane). | We will cont ***** ***** abraxane. |
| P2 | Treatment_Changes | Missing detail about the specific drug started. | Started [REDACTED] [REDACTED] 11/05/20 |
| P2 | Treatment_Goals | Inaccurate goal description. Should be 'palliative' but context suggests ongoing treatment for metastatic disease. | The patient is advised to call the GI Oncology number *****-*****-***** for all untoward side effect |
| P2 | Lab_Plan | Should mention future lab plans for monitoring. | Labs including the CBC and chemistry profile were reviewed per protocol. |

*Extraction summary*: Most fields are clean, but there are minor issues with missing current cancer-related medications, incomplete treatment changes, inaccurate treatment goals, and lack of future lab plans.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 72 — ⚠️ ISSUES

**Type**: Moderately differentiated adenocarcinoma
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Current_Medications | Missing current cancer-related medication (gemcitabine and Abraxane) | We elected to start gemcitabine and Abraxane. He has now had almost 4 cycles. |
| P2 | Treatment_Changes | Inconsistent mention of chemotherapy holiday and resumption of treatment | We decided to give him a chemotherapy holiday. When we saw him in October 2016, he was having progre |
| P2 | Clinical_Findings | Missing recent physical exam finding of shortness of breath and dyspnea with exertion | Recently, he had an episode of shortness of breath and dyspnea with exertion. |

*Extraction summary*: Major error in missing current cancer-related medication and minor issues in inconsistent treatment changes and missing recent physical exam findings.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Contains complex medical jargon that may be difficult for an 8th-grade reader to understand. | Your cancer is currently showing stable disease in the abdomen but progressing p |
| P2 | Incomplete sentence; it doesn't specify the name of the medication. | We decided to resume chemotherapy but switched to a reduced dose of a medication |

*Letter summary*: The letter contains some complex medical jargon and an incomplete sentence that need addressing.

---

## ROW 77 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the tail of the pancreas
**Stage**: Resectable at diagnosis, now on surveillance

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Reason_for_Visit.summary | The summary mentions 'follow-up visit for adjuvant therapy with gemcitabine and capecitabine', but the note states that he has already completed 6 cycles of adjuvant therapy. | He has had 6 cycles and is on surveillance. |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage is described as 'Resectable at diagnosis, now on surveillance'. While accurate, it could be more precise by specifying the current status as 'NED' (no evidence of disease). | Within this limitation, no evidence of recurrent or metastatic disease in abdomen or pelvis. |
| P2 | Treatment_Changes.recent_changes | The field is empty, but the note indicates that the patient has completed adjuvant therapy and is now on surveillance. This should be reflected. | He has had 6 cycles and is on surveillance. |
| P2 | Response_Assessment.response_assessment | The field states 'Not yet on treatment — no response to assess.' However, the patient has completed adjuvant therapy and is now on surveillance. | He has had 6 cycles and is on surveillance. |
| P2 | Imaging_Plan.imaging_plan | The imaging plan only mentions 'CT Abdomen', but the note suggests a comprehensive surveillance plan which may include both abdomen and chest imaging. | CT Abdomen /Pelvis without Contrast... CT Chest without contrast |

*Extraction summary*: Most fields are clean, but there are minor issues regarding the completion of adjuvant therapy and the specificity of the imaging plan.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'extensive local involvement' may be confusing to an 8th-grade reader. | You have adenocarcinoma (cancer that started in gland cells) of the tail of the  |
| P2 | The phrase 'negative margins and lymph nodes' might be unclear. | It was surgically removed with negative margins and lymph nodes. |
| P2 | The mention of AST and ALT values might be too technical. | However, your liver function tests show elevated levels of AST (162 U/L) and ALT |

*Letter summary*: Letter is mostly clean but contains a few terms that could be simplified for better readability.

---

## ROW 79 — ⚠️ ISSUES

**Type**: Adenocarcinoma of pancreaticobiliary origin
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | The field is empty, but the note mentions no cancer-related medications. | Medications the patient states to be taking prior to today's encounter. |
| P2 | Treatment_Goals | The field states 'palliative', but the note suggests a goal of achieving a deep and durable remission, which aligns more closely with 'curative intent'. | the mainstay of treatment at this juncture should consist of systemic therapy with the goal of achie |
| P2 | Lab_Results | The field includes 'Hemoglobin 14.0, Hematocrit 41.0, Platelet Count 166, Int'l Normaliz Ratio 1.0', which are not mentioned in the note. | Most recent labs notable for the following: WBC Count 5.8 05/02/2020 |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness of medication listing, the specificity of treatment goals, and the inclusion of lab results not mentioned in the note.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'pancreaticobiliary' is medical jargon that is not explained and may be confusing to a layperson. | You were diagnosed with adenocarcinoma (cancer that started in gland cells) of p |

*Letter summary*: Letter is mostly clean but contains minor readability issues.

---

## ROW 82 — ⚠️ ISSUES

**Type**: Pancreatic adenocarcinoma
**Stage**: Stage IV (metastatic to liver and lung)

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing current cancer-related medication (fentanyl patch). | fentaNYL (DURAGESIC) 25 mcg/hr patch Place 1 patch onto the skin every other day. For cancer pain. |

*Extraction summary*: One minor issue found in Current_Medications.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 84 — ⚠️ ISSUES

**Type**: Pancreatic ductal adenocarcinoma with perineural invasion
**Stage**: now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Incomplete. Current medications should include both atezolizumab and cobimetinib. | Now midway through cycle #2 of treatment on the crossover arm of the same [REDACTED] trial consistin |
| P2 | Treatment_Changes | Incomplete. Should mention both atezolizumab and cobimetinib. | Switched to the combination of atezolizumab, a PD-L1 inhibitor (administered on a q2 weekly schedule |
| P2 | Treatment_Goals | Inaccurate. Goals should be 'palliative' rather than 'palliative'. | Goals of treatment are not explicitly stated, but given the context, it should be 'palliative'. |
| P2 | Response_Assessment | Slightly imprecise. Should clarify that the response is stable disease with slight progression. | Review of his CT scans show essentially stable to slightly enlarging sites of his measurable lesions |
| P2 | Medication_Plan | Incomplete. Should include both atezolizumab and cobimetinib. | Mr. [REDACTED] is continuing on the crossover arm of the [REDACTED] trial, consisting of atezolizuma |
| P2 | Therapy_plan | Incomplete. Should include both atezolizumab and cobimetinib. | Continue on the crossover arm of the [REDACTED] trial consisting of the combination of atezolizumab, |
| P2 | radiotherapy_plan | Incomplete. Should specify the nature of the radiotherapy. | At the point he progresses on this urgent regimen, we could have him see Rad Onc for consideration o |
| P2 | Procedure_Plan | Incomplete. Should specify the nature of the procedure. | At the point he progresses on this urgent regimen, we could have him see Rad Onc for consideration o |
| P2 | Genetic_Testing_Plan | Inaccurate. Should be 'pd-l1 expression' or 'not specified' instead of 'pd-l1'. | No specific genetic testing plan mentioned. |

*Extraction summary*: Most fields are clean, but there are minor issues with completeness and precision in medication and treatment plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P2 | The term 'pancreatic cancer' is overly simplified for a new patient, but acceptable here as it's a follow-up. | You came in for a follow-up visit regarding your pancreatic cancer treatment. |
| P2 | The phrase 'not growing or shrinking much' is a simplification of 'stable disease', which may be slightly imprecise. | This means the cancer is not growing or shrinking much. |
| P2 | The term 'less activity' is a simplification of 'decreased associated enhancement'. | The CT scan shows that the mass in your liver has gotten a little bigger but wit |
| P2 | The description of the medicines is simplified but may lack precision. | You switched to a new treatment plan. Now you are taking atezolizumab (a medicin |
| P2 | The term 'radiation treatment' is a simplification of 'referral to Rad Onc for [REDACTED] of limited, discrete sites of hepatic involvement'. | We will also consider radiation treatment if the cancer starts to grow again. |

*Letter summary*: Letter is clean. Some minor simplifications are noted but acceptable given the context.

---

## ROW 87 — ⚠️ ISSUES

**Type**: Well-differentiated pancreatic neuroendocrine tumor (PNET) metastatic to the liv
**Stage**: Metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Current_Medications | Missing octreotide, which is a cancer-related medication. | octreotide (SANDOSTATIN *****) 30 mg every 30 (thirty) days. |
| P2 | Treatment_Changes | Missing mention of increased metformin dosage. | Recent metformin and exercise for higher sugars- now fine. |
| P2 | Treatment_Goals | Should be 'palliative', but the note suggests ongoing management rather than end-of-life care. | Continue Everolimus 10 mg daily. |
| P2 | Response_Assessment | Inaccurate summary of response assessment. It should include the waxing and waning appearance of lung findings. | Left greater than right peripheral lower lobe reticulation and groundglass opacity is increased comp |
| P2 | Medication_Plan | Missing mention of increased metformin dosage. | Recent metformin and exercise for higher sugars- now fine. |
| P2 | Procedure_Plan | Missing mention of follow-up MMA and B12. | #low B12  On oral replacement- -needs f/u MMA, and B12 |
| P2 | Imaging_Plan | Missing mention of follow-up H&P and chest CT. | Needs f/u H&P and chest CT 3 mo-- sooner if sx |
| P2 | Lab_Plan | Missing mention of follow-up MMA and B12. | #low B12  On oral replacement- -needs f/u MMA, and B12 |

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications, treatment changes, and follow-up plans.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Inaccurate information. Octreotide is not mentioned as a current medication in the extracted data. | You will also continue using octreotide. |
| P2 | Unexplained jargon ('bruises'). | You have multiple small bruises noted on physical exam. |

*Letter summary*: There is inaccurate information regarding octreotide and minor readability issues.

---

## ROW 90 — ⚠️ ISSUES

**Type**: Primary pancreatic neuroendocrine tumor
**Stage**: 

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P1 | Cancer_Diagnosis.Type_of_Cancer | The note mentions both primary pancreatic neuroendocrine tumor and metastatic BRAF mutant melanoma, but the extracted data only lists 'Primary pancreatic neuroendocrine tumor'. | Regarding her metastatic ***** mutant melanoma (started on heel, 3 LNs in knee and 5 LNs in groin -> |
| P1 | Current_Medications.current_meds | The note indicates the patient is on intermittent dabrafenib and trametinib for melanoma, but this is not reflected in the extracted data. | continues on clinical trial, taking intermittent dabrafenib +***** ***** then trametinib alone ***** |
| P2 | Treatment_Changes.recent_changes | The note suggests reducing the dose of octreotide, but the extracted data does not clearly indicate the reason for the change. | Given documented ***** to liver-- favor trial of somatostatic analog (based on clarinet data showing |
| P2 | Treatment_Goals.goals_of_treatment | The note implies a mixed goal of managing both the pNET and melanoma, but the extracted data only states 'palliative'. | I'd like to avoid sunitinib or everolimus or chemotherapy given need for ongoing ***** inhibitor. |

*Extraction summary*: Major errors include missing metastatic BRAF mutant melanoma and current melanoma treatments. Minor issues involve unclear reasons for medication changes and incomplete treatment goals.

### Letter

| Severity | Issue | Sentence |
|----------|-------|----------|
| P1 | Incomplete sentence with missing critical information. | Your dose of octreotide was reduced for four days, then increased for another fo |

*Letter summary*: Letter contains an incomplete sentence that needs to be completed with the final dosage details.

---

## ROW 91 — ⚠️ ISSUES

**Type**: Adenocarcinoma of the pancreas, moderately differentiated
**Stage**: now with residual adenocarcinoma and 8 of 23 lymph nodes positive

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The stage description is imprecise and could be clearer. | now with residual adenocarcinoma and 8 of 23 lymph nodes positive |
| P2 | Treatment_Changes.recent_changes | The field should indicate that the patient has completed neoadjuvant therapy and surgical resection. | He has completed 6 cycles of therapy. He was taken to surgery on 11/27/2018. |
| P2 | Treatment_Changes.supportive_meds | Supportive medications used during treatment should be noted if applicable. | He experienced a delay in starting treatment because of tumor ingrowth into his stent which required |
| P2 | Treatment_Goals.goals_of_treatment | The goal should be more specific, reflecting the current situation of monitoring for recurrence. | As long as he is asymptomatic, we will watch this as hope that it represents delayed post op changes |
| P2 | Response_Assessment.response_assessment | The response assessment should include the patient's current status and the plan for monitoring. | He may well have local recurrence but with the current coronavirus crisis, it will be difficult to w |

*Extraction summary*: Most fields are clean, but there are minor issues with the clarity of the stage description, treatment goals, and response assessment.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

## ROW 92 — ⚠️ ISSUES

**Type**: Moderately differentiated pancreatic ductal adenocarcinoma with perineural and l
**Stage**: Originally borderline resectable, now metastatic ()

### Extraction

| Severity | Field | Issue | Note Evidence |
|----------|-------|-------|---------------|
| P2 | Cancer_Diagnosis.Type_of_Cancer | The note does not specify 'perineural and lymphovascular invasion' in the initial diagnosis. | Ms. ***** is a 63 y.o. female whom I am seeing at the UCSF Comprehensive Cancer Center for follow-up |
| P2 | Cancer_Diagnosis.Stage_of_Cancer | The note mentions 'borderline resectable' initially, but the current stage is not explicitly stated as 'metastatic'. | This pleasant woman was originally diagnosed with borderline resectable disease back in November 201 |
| P2 | Lab_Results.lab_summary | The note does not provide specific lab values for WBC, Hct, plts, Creat, AST/ALT, tot bili, alk phos, CA 19-9. | Her *****-9 levels gradually ***** upwards, with a CT scan performed here at ***** on 12/10/2018 sho |
| P2 | Current_Medications.current_meds | The note does not mention any current cancer-related medications, but the patient is on chemotherapy. | She has since opted to resume chemotherapy with gemcitabine/*****-paclitaxel, well-tolerated to date |
| P2 | Treatment_Changes.recent_changes | The note does not mention any recent changes in treatment, but the patient has resumed chemotherapy. | She has since opted to resume chemotherapy with gemcitabine/*****-paclitaxel, well-tolerated to date |
| P2 | Response_Assessment.response_assessment | The note does not provide specific dates for the CT scan and CA 19-9 levels mentioned. | Postoperative scans have revealed no evidence of active disease in the liver and a slightly growing  |
| P2 | Lab_Plan.lab_plan | The note does not specify a plan for labs unless CA 19-9 markedly rises. | We are also keeping in mind future clinical trial possibilities at ***** that may be options for her |

*Extraction summary*: Most fields are clean, but there are minor issues with the completeness and precision of certain fields.

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

*Extraction summary*: Most fields are clean, but there are minor issues with missing cancer-related medications and incomplete supportive medications. Additionally, 'Lab_Plan' is misclassified.

### Letter: ✅ Clean

*Letter summary*: Letter is clean.

---

