# Letter v8 Template Test Review

**Run**: test_v8_template_20260331_212918
**Date**: 2026-03-31
**Samples**: 10 (S0, S5, S7, S11, S19, S29, S35, S43, S64, S87)
**Purpose**: Test template-based letter generation (5-section structure)

## Executive Summary

| Metric | Result |
|--------|--------|
| P0 (hallucination) | 0 |
| P1 (major error) | 2 |
| P2 (minor issue) | 8 |
| Template compliance | 10/10 |
| Avg readability grade | 6.4 (target <8.0) |
| Avg field coverage | 73% |
| POST-LETTER warnings | 0 |

**Overall**: v8 template approach is a clear improvement over v7. Template structure consistently followed. No hallucinations. Receptor explanations removed per doctor feedback. Information repetition resolved. Two P1 issues need prompt/code fixes.

---

## Per-Sample Review

### S0 (ROW 1, coral_idx 140) — Stage IV metastatic ER+/PR+/HER2- IDC
**Grade: 6.4 | Coverage: 80%**

Letter:
- "You came in for an initial consult regarding your breast cancer treatment." ✓ [source:summary]
- "invasive ductal carcinoma, which means the cancer started in the milk ducts" ✓
- "advanced stage (Stage IV) and has spread to other parts of your body, including your lungs, peritoneum, liver, and ovaries." ✓ matches note
- "brain MRI and a bone scan" ✓ matches A/P
- "Ibrance and another medication" ✓ [REDACTED] handled
- "You are referred to the Integrative Medicine center." ✓ passive voice
- "your doctor on Thursday to biopsy the mass in your right axilla" ✓

Issues:
- **P2**: "peritoneum" not explained — should say "peritoneum (the lining of your belly area)". Removed from v8 prompt Rule 15 by accident.
- **P2**: "axilla" not explained — should say "armpit area"
- **P2**: Missing emotional support sentence — note says patient is "very scared", "distressed", "anxious". emotional_context should have been detected and added. Letter omits it.
- **P2**: Missing "Full code" advance care — keypoints extracted it but letter doesn't mention it. Per Rule 12 this is meaningful and should be included.

### S5 (ROW 6, coral_idx 145) — Early stage ER+/PR+/HER2- IDC, post-mastectomy
**Grade: 5.8 | Coverage: 54%**

Letter:
- "You came in for a follow-up visit regarding your breast cancer treatment." ✓
- "invasive ductal carcinoma, which means the cancer started in the milk ducts. It is also ER+/PR+, meaning the cancer cells have receptors for the hormones estrogen and progesterone." ✓ — LLM explained ER+ naturally without old bloated Rule 10 template
- "Your blood tests are mostly normal." ✓ good summarization of extensive lab panel
- "Some specific tests were done, including estradiol and vitamin D levels." ✓
- "You started taking letrozole today." ✓
- "You are referred to your doctor for genetics counseling." ✓
- "We noticed you appear anxious. Please let us know if you need support." ✓ emotional context detected

Issues:
- **P2**: Coverage 54% — many fields not tagged. Lab details appropriately summarized but tagged only as lab_summary, missing coverage of findings, supportive_meds, etc.
- **P2**: Stage missing from letter — keypoints Stage_of_Cancer is empty (extraction issue), but note says "1.5 cm node neg, grade 1" which is approximately Stage I. Letter correctly doesn't fabricate, but could mention "early stage."

### S7 (ROW 8, coral_idx 147) — Stage III ER-/PR-/HER2+ IDC, post-lumpectomy
**Grade: 6.7 | Coverage: 85%**

Letter:
- "You came in for a consultation to discuss treatment options and to establish care after receiving an incomplete course of neoadjuvant TCHP and a lumpectomy/ALND." ✓ accurate
- "It is HER2 positive and both estrogen and progesterone receptor negative." ✓
- "no remaining cancer in the breast tissue but some cancer in the lymph nodes" — ✓ accurately reflects pathology (0 residual in breast, 3/28 LN+)
- "adjuvant AC, followed by T-DM1" ✓
- "You will also need radiation after completing AC." ✓
- "You are referred to social work for additional support." ✓

Issues:
- **P1**: "neoadjuvant TCHP" and "lumpectomy/ALND" are medical jargon in the "Why" section. Should explain: "treatment given before surgery" and "surgery to remove the cancer and check the lymph nodes". This is a significant readability issue for a patient letter.
- **P2**: "oxyCODONE" listed without explanation — should just say "a pain medication"

### S11 (ROW 12, coral_idx 151) — Stage IV ER+/PR+/HER2+ IDC, metastatic to brain/lung/bone
**Grade: 5.3 | Coverage: 86%**

Letter:
- "You have advanced stage breast cancer that has spread to your brain, lungs, and bones." ✓
- "The goal of your treatment is to manage symptoms and keep the cancer from growing." ✓ palliative well-explained
- "Recent imaging shows that the cancer is stable, with no new growth in some areas." — Partially accurate. Note says CT CAP stable but MRI brain shows 2 NEW foci. See P1 below.
- "You will continue taking herceptin, letrozole, and another medication." ✓
- "You are referred to a radiation oncologist for possible repeat radiation treatment." ✓
- "Your next clinic visit is in 6 weeks." ✓

Issues:
- **P1**: Letter says "no new growth in some areas" but omits the 2 new brain lesions found on MRI (01/31 brain MRI). The keypoints note includes this info, and this is clinically significant. The "What was discussed" section should mention BOTH stable body scans AND new brain lesions. This violates Rule 14 (mixed results must mention both).
- **P2**: DNR/DNI advance care not mentioned — the problem list clearly states "DNR/DNI" and "would not want life support treatments". This is meaningful per Rule 12.

### S19 (ROW 20, coral_idx 159) — Stage IV ER+/PR+/HER2- IDC, metastatic recurrence
**Grade: 5.8 | Coverage: 78%**

Letter:
- "You came in for a follow-up visit regarding your breast cancer treatment." — technically this is a New patient consult (keypoints say "New patient"). P2.
- "it has spread to other parts of your body, including your bones, lymph nodes, and lungs." ✓
- "ordered an MRI of your spine and a CT scan of your chest, abdomen, and pelvis." ✓
- "You have started taking Letrozole and Palbociclib." ✓
- "denosumab after a dental check-up to help protect your bones." ✓ good explanation
- "You are referred to a radiation oncologist for a consultation." ✓

Issues:
- **P2**: "follow-up visit" should be "initial consult" — keypoints correctly say New patient, letter got it wrong
- **P2**: Foundation One genetic testing plan mentioned in keypoints but omitted from letter

### S29 (ROW 30, coral_idx 169) — Stage II-III ER-/PR-/HER2+ IDC, neoadjuvant
**Grade: 6.9 | Coverage: 83%**

Letter:
- "You came in for a consultation to discuss treatment options for your recent diagnosis of early stage breast cancer." ✓
- "invasive ductal carcinoma, which means the cancer started in the milk ducts. It is in the early stage, but it is considered stage II-III." ✓
- "The goal of treatment is to cure the cancer." ✓
- "You had a PET/CT scan that showed a large mass in your right breast and some lymph nodes that might be affected." ✓
- "You will start treatment with chemotherapy and biological therapies to shrink the cancer before surgery. This is called neoadjuvant treatment." ✓ explained neoadjuvant
- "Mediport placed to make it easier to give you the chemotherapy" ✓ good explanation
- "echocardiogram (TTE) before starting treatment." ✓

Issues:
- None significant. This is one of the best letters in the batch.

### S35 (ROW 36, coral_idx 175) — pT3N0 ER+/PR+/HER2- mixed ductal/mucinous
**Grade: 5.8 | Coverage: 56%**

Letter:
- "a type of cancer that makes mucus, called ER+/PR+/HER2- grade III mixed ductal and mucinous carcinoma." — ✓ mucinous explained
- "Your cancer is in stage II-III." ✓ TNM converted
- "You have swelling in your right arm and hand, which has improved with elevation." ✓
- "You will get a Doppler test to check for a blood clot in your arm." ✓
- "You switched to a medication called Abraxane on 02/20/21 due to a reaction with your previous medication, Taxol." ✓
- "You are referred to radiation oncology" ✓
- "We understand that you are feeling anxious, and we are here to support you." ✓ emotional support

Issues:
- **P2**: Coverage 56% — abnormal labs (Hgb 11.8 L, Hct 35.3 L, Albumin 3.3 L) not mentioned in letter. These are clinically significant low values.
- **P2**: Thyroid nodule finding (2.1 cm with FDG uptake, recommended for ultrasound evaluation) not mentioned — this is a clinically relevant incidental finding.

### S43 (ROW 44, coral_idx 183) — ER+/PR+/HER2- node+ IDC, BRCA1+, post-chemo
**Grade: 6.3 | Coverage: 67%**

Letter:
- "invasive ductal carcinoma, which means the cancer started in the milk ducts. There is also some cancer in the milk-producing glands (DCIS)." ✓
- "has spread to some nearby lymph nodes." ✓
- "The goal of your treatment is to cure the cancer." ✓
- "imaging tests for neck and back pain were normal." ✓
- "A small nodule in your lung that was found earlier is stable and will be checked again in a year." ✓ good patient-friendly explanation
- "You will start a medicine called an aromatase inhibitor after you finish radiation." ✓
- "surgery to remove both your ovaries and fallopian tubes (BSO)" ✓ BSO explained by pre-translation
- "You are enrolled in a clinical trial for radiation." ✓
- "You are referred to physical therapy" ✓
- "scheduled to see a nutritionist on November 30, 2018." ✓

Issues:
- **P2**: Letter cut off — "Please feel free to contact us if you have any" — truncated at max tokens. Missing "questions." and "Sincerely, Your Care Team". This is a generation length issue.

### S64 (ROW 65, coral_idx 204) — ER weak+/PR low+/HER2- IDC, neoadjuvant + ISPY trial
**Grade: 7.7 | Coverage: 73%**

Letter:
- "weakly positive for estrogen receptors and low positive for progesterone receptors, and it is negative for HER2." ✓ well-explained
- "cancer in the lymph nodes under your arm" ✓ "under your arm" instead of "axillary"
- "neoadjuvant chemotherapy, which is treatment given before surgery to shrink the cancer." ✓ explained
- "doxorubicin, cyclophosphamide, and paclitaxel" — drug names are appropriate to include
- "eligible for a clinical trial that tests new treatments" ✓
- "a port placed for your chemotherapy" ✓
- "You are referred to a genetic counselor to review the results of your genetic testing." ✓

Issues:
- **P2**: "lumpectomy or a mastectomy" — medical jargon. Should explain: "surgery to remove the cancer (either the tumor only or the entire breast)"
- **P2**: TTE/echocardiogram not mentioned in letter — keypoints have it in imaging_plan

### S87 (ROW 88, coral_idx 227) — Stage IV ER+/PR+/HER2- IDC (met biopsy ER-/PR-)
**Grade: 7.1 | Coverage: 71%**

Letter:
- "a new patient evaluation regarding your ongoing management of metastatic breast cancer." ✓
- "breast cancer that started in the milk ducts (invasive ductal carcinoma)" ✓
- "has spread to other parts of your body, including your brain, lungs, and lymph nodes." ✓
- "The cancer in your breast was sensitive to certain hormones, but the cancer that spread to your brain is not." ✓ — excellent plain-language explanation of receptor discordance
- "a medication called Xeloda" ✓
- "If tests show that the cancer is sensitive to a protein called HER2, you may need additional treatment targeting this protein." ✓ explained HER2 testing rationale
- "clinical trials or other treatments that combine immunotherapy with chemotherapy" ✓
- "tests done on the cancer in your brain and the remaining cancer in your breast to check for sensitivity to hormones and the HER2 protein." ✓

Issues:
- **P2**: Missing "Full code" advance care — keypoints have it, letter omits

---

## Pattern Summary

### v8 Improvements over v7
1. **Template compliance**: 10/10 samples follow 5-section structure perfectly
2. **No receptor redundancy**: ER+/PR+ explained naturally in 1-2 words, no bloated "which means it grows in response to hormones (estrogen)" on every letter
3. **No information repetition**: Metastasis sites mentioned once, not 3 times
4. **Passive voice**: "You are referred to" used consistently
5. **Simpler language**: Average readability 6.4 (vs ~7-8 in v7)
6. **Emotional support**: Detected and included where appropriate (S5, S35)
7. **Term pre-translation working**: BSO, mucinous, ductal carcinoma all explained

### Remaining Issues

#### P1 Issues (2 total)
| Sample | Issue | Fix |
|--------|-------|-----|
| S7 | Medical jargon in "Why" section (TCHP, lumpectomy/ALND) | Add prompt rule: "In the 'Why' section, explain all medical terms" |
| S11 | Mixed response omission — said "stable" but omitted 2 new brain lesions | Rule 14 already exists but LLM didn't follow it; may need stronger emphasis or POST check |

#### P2 Patterns (recurring)
| Pattern | Count | Samples | Fix |
|---------|-------|---------|-----|
| "peritoneum"/"axilla" unexplained | 2 | S0 | Re-add peritoneum to Rule 15 term list |
| Full code/DNR omitted from letter | 3 | S0, S11, S87 | Add prompt instruction: "If advance care has meaningful content, always include it" |
| New patient called "follow-up" | 1 | S19 | Letter generation should use Patient type field |
| Low field coverage (<60%) | 2 | S5, S35 | May need to emphasize "cover all important fields" |
| Letter truncated at max tokens | 1 | S43 | Increase max_new_tokens for letter gen |
| Minor jargon leaks | 2 | S7, S64 | Add more terms to explain list |

### Prompt Fixes Needed for v8a
1. **Re-add peritoneum** to Rule 15 medical terms list
2. **Add advance care instruction**: "If Advance care field has meaningful content like 'Full code' or 'DNR/DNI', include it in the 'What is the plan?' section."
3. **Strengthen mixed-response rule**: Add to "What was discussed?" section mapping: "If response data shows mixed results (some stable, some progressing), you MUST mention BOTH."
4. **Increase letter max_new_tokens**: Current limit causes truncation on complex cases (S43)

### Code Fix Needed
- `letter_generation.py` `generate_tagged_letter()`: increase default `max_new_tokens` from 512 to 768 to prevent truncation
