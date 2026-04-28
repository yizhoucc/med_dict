# PDAC v32 Triple Review: Extraction × Letter × Auto-Review

Generated: 2026-04-28
Reviewer: Claude (manual, against original notes)
Scope: 8 PDAC samples — reviewing extraction, letter, AND auto_review accuracy

## Summary

| ROW | Extraction | Letter | Auto-Review Accuracy |
|-----|-----------|--------|---------------------|
| 1   | P2 ×2     | P2 ×2  | Noisy — 4/8 findings valid, 4 false P1 |
| 4   | ✅         | P2 ×2  | 1 valid P1, rest are noise/false positives |
| 15  | P2 ×1     | P1 ×1  | Mixed — caught real issues but also false P1s |
| 36  | ✅         | P1 ×1  | 2 valid P1s (letter dose missing), rest noise |
| 40  | P2 ×1     | P2 ×1  | 1 valid P1 (meds), rest exaggerated |
| 43  | P2 ×1     | P2 ×2  | Mixed — over-flagged current_meds |
| 59  | P1 ×2     | ✅      | 3 valid P1s, 2 false P1s |
| 82  | P2 ×1     | P1 ×1  | 1 valid P1 (meds), letter garbled text caught |

**Real totals**: P0=0, P1=5, P2=12
**Auto-review claimed**: P0=0, P1=26, P2=43
**Auto-review false positive rate**: ~60% of P1s are exaggerated or wrong

---

## ROW 1 (idx 0, coral 40) — Neoadjuvant PDAC, on chemo

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Type_of_Cancer | Moderately differentiated pancreatic adenocarcinoma | ✅ Correct — note says "moderately differentiated pancreatic adenocarcinoma" from FNA |
| Stage_of_Cancer | Originally Stage III (based on CA-19-9...) | P2 — Note never says "Stage III". Says "potentially resectable (with a regional LN)". Inferred staging is reasonable but unsupported |
| Distant Metastasis | Not sure, based on CA-19-9... | ✅ Reasonable — doctor says "highly concerning for micrometastatic disease" but no confirmed distant mets |
| current_meds | (empty) | P2 — Patient IS on chemo (cycle #5 today) but drug name is REDACTED. Empty is defensible since we can't name the drug |
| goals_of_treatment | curative | ✅ Correct — neoadjuvant intent, potentially resectable |
| response_assessment | Positive response, LN resolved, CA 19-9 dropped 77K→6K | ✅ Faithful to note |
| genetic_testing_plan | Foundation One if adequate cellularity | ✅ Exact from note |

### Letter
- P2: "precipitous drop" is not 8th-grade language
- P2: 1855 chars, exceeds 800-1200 target; includes doctor-level language ("Consider a period of up-front chemotherapy")

### Auto-Review Assessment
- **Valid**: Stage P2 (correctly identified), letter readability issues
- **False P1**: "Missing current medications" — marked P1 but drug name is REDACTED, so empty is defensible (P2 at most)
- **False P1**: "Does not specify the type of pancreatic cancer" in letter — the letter says "pancreatic cancer treatment" which is fine for a follow-up patient
- **Overall**: Auto-review over-escalated severity. Real issues are P2, not P1

---

## ROW 4 (idx 3, coral 43) — Late-stage metastatic PDAC, GOC discussion

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Type_of_Cancer | Metastatic pancreatic ductal adenocarcinoma | ✅ |
| Stage_of_Cancer | Stage IV (met to liver and peritoneum) | ✅ |
| current_meds | (empty) | ✅ Correct — patient completed 6 cycles, now s/p front-line, no active chemo. Supportive meds (fentanyl etc.) are in supportive_meds field |
| goals_of_treatment | palliative | ✅ Note explicitly discusses hospice/palliative |
| response_assessment | Progression with recurrent bowel obstruction, some stable imaging | ✅ Faithful |
| Advance care | Hospice discussion, inpatient facility preference | ✅ Good capture |

### Letter
- P2: Medical jargon unexplained ("hepatic segment V/VIII", "peritoneal carcinomatosis", "ascites")
- P2: "The patient was discussed the option" — grammatically awkward, should be "You were told about the option"

### Auto-Review Assessment
- **False P1**: "current medications are missing" — WRONG. Patient has no active cancer treatment. Supportive meds ARE captured. Auto-review confused supportive meds with current_meds
- **False P1**: "supportive medications listed are incomplete" — it lists Fentanyl, Dilaudid, DexAMETHasone, ondansetron. The note has more (acetaminophen, apixaban, bisacodyl, lorazepam, octreotide, pantoprazole, senna, etc.) but those are non-cancer meds. Auto-review doesn't understand our extraction scope
- **Overall**: Auto-review applied wrong standards. The extraction correctly follows our prompt rules

---

## ROW 15 (idx 14, coral 54) — Young patient, peritoneal progression

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Stage_of_Cancer | Originally Stage IIB, now metastatic (Stage IV) | P2 — Note never says "Stage IIB". Originally presented with metastatic disease (celiac LN), so it was Stage IV from the start. "Originally Stage IIB" is hallucinated |
| current_meds | (empty) | ✅ — Note says "He is not taking any prescription medications" |
| response_assessment | Not yet on treatment | ✅ — He's currently off treatment, about to resume |
| therapy_plan | None | P2 — A/P says "We will resume [REDACTED]", that IS a therapy plan |

### Letter
- P1: "We will resume a medication. He responded initially quite well to a medication but because of his residual neuropathy..." — Uses "He" instead of "You". Doctor voice leaked into patient letter
- P2: "Your next visit date is not specified in the given text" — meta-statement should not be in a patient letter

### Auto-Review Assessment
- **Valid P2**: Stage IIB not in note — caught this correctly
- **Valid P1**: Letter voice confusion ("He responded") — auto-review said "confusing and uses medical jargon" which is close
- **False P1**: "omits the specific term 'peritoneal metastases'" — letter actually says "new growths found in the lining of your abdomen" which IS a plain-language version. Auto-review wanted the technical term, opposite of what we want

---

## ROW 36 (idx 35, coral 75) — Neuroendocrine tumor, everolimus

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Type_of_Cancer | Nonfunctioning pancreatic neuroendocrine tumor | ✅ Excellent — correctly identified non-adenocarcinoma subtype |
| current_meds | everolimus | ✅ |
| response_assessment | Stable disease with mixed changes | ✅ Faithful |
| lab_summary | No labs in note | ✅ — Note says "Most recent labs notable for:" but no values follow |

### Letter
- P1: "reduced from 10 to because of some side effects" — dose 7.5mg is MISSING. The extraction has it ("7.5 mg daily") but the letter dropped it. Likely truncation or generation issue
- P2: Says "pancreatic cancer" instead of "neuroendocrine tumor" — should distinguish

### Auto-Review Assessment
- **Valid P1 ×2**: Correctly caught the missing dose in letter ("reduced from 10 to")
- **Valid P1**: Correctly flagged "pancreatic cancer" should be "neuroendocrine tumor"
- **False P2s**: Flagged incomplete lab info, incomplete current_meds — these are actually correct per our extraction rules

---

## ROW 40 (idx 39, coral 79) — New diagnosis, short note

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Stage_of_Cancer | (empty) | P2 — Should be "Stage IV" or "Metastatic". Note says "metastatic adenocarcinoma of the pancreas" in the chief complaint |
| current_meds | (empty) | ✅ — New patient, not yet on cancer treatment. Fentanyl is for pain management (supportive) |
| goals_of_treatment | palliative | ✅ — Metastatic disease |

### Letter
- P2: "increase the dose of your fentanyl patch to and added Reglan" — dose number missing after "to". The extraction has "75 mcg" but letter dropped it

### Auto-Review Assessment
- **False P1**: "current medications are missing despite being listed in the note" — lists fentanyl, insulin, metformin, etc. These are NON-CANCER meds. Our prompt correctly excludes them. Auto-review doesn't understand the extraction scope
- **Valid P1**: Fentanyl dose missing in letter — valid catch but should be P2

---

## ROW 43 (idx 42, coral 82) — Borderline resectable, on neoadjuvant chemo

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Stage_of_Cancer | Borderline resectable due to <180° vascular involvement | ✅ Excellent — captured resectability status |
| current_meds | (empty) | P2 — Patient is on neoadjuvant FOLFIRINOX (drug names are in A/P: Oxaliplatin, Irinotecan, 5FU). Should list these |
| goals_of_treatment | curative | ✅ — Neoadjuvant intent for borderline resectable |

### Letter
- P2: Uses brand names "Compazine" and "Zofran" without explanation
- P2: Chemo dose details ("25% dose reduction") included — prompt says no dosing details

### Auto-Review Assessment
- **False P1**: "current medications are missing" — lists clopidogrel, docusate, empagliflozin as evidence. These are NON-CANCER drugs. The real issue is the chemo drugs (oxaliplatin, irinotecan) are missing, but auto-review cited wrong evidence
- **Overall**: Right conclusion (meds missing) but wrong reasoning

---

## ROW 59 (idx 58, coral 98) — Post-surgical surveillance, NED

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Stage_of_Cancer | Originally Stage IIB, now metastatic (Stage IV) | P1 — Misleading. Lung met was resected in 2017. Currently NED. Should say "Originally Stage IIB, had pulmonary recurrence (resected 2017), currently NED" |
| goals_of_treatment | adjuvant | P1 — Wrong. Adjuvant therapy completed in 2012. Currently on surveillance with NED. Should be "surveillance" or omitted |
| current_meds | (empty) | ✅ — Note says she's on surveillance, no cancer meds. General meds (amlodipine, aspirin, etc.) correctly excluded |
| response_assessment | Stable disease, no evidence of recurrent/metastatic disease | ✅ Correct |

### Letter
- ✅ Actually clean and appropriate. Simple, clear, correct for a surveillance visit

### Auto-Review Assessment
- **Valid P1**: Stage_of_Cancer incorrect — caught correctly
- **Valid P1**: goals "adjuvant" incorrect — caught correctly
- **False P1**: "Missing current medications" — WRONG. Patient is on no cancer meds. Auto-review cited the general medication list (amlodipine, lisinopril, metformin) which are correctly excluded
- **False P1**: "Missing recent treatment changes" / "Missing supportive medications" — WRONG. Patient is on surveillance, there ARE no changes or supportive meds
- **Overall**: Auto-review caught the 2 real P1s but added 3 false P1s by not understanding extraction scope

---

## ROW 82 (idx 81, coral 121) — Progressive disease, starting new chemo

### Extraction
| Field | Value | My Assessment |
|-------|-------|--------------|
| Type_of_Cancer | Pancreatic adenocarcinoma | ✅ |
| Stage_of_Cancer | Stage IV (met to liver and lung) | ✅ |
| current_meds | (empty) | P2 — Patient was on prior treatment that failed. Now starting new regimen. Technically between regimens, so empty is defensible |
| goals_of_treatment | palliative | ✅ |
| response_assessment | Progressive disease, enlarging liver/lung tumors | ✅ Correct |

### Letter
- P1: "Initiation of a modified a chemotherapy regimenIRI regimen including oxaliplatin" — garbled text, words run together. Likely generation artifact

### Auto-Review Assessment
- **Valid P1**: Garbled letter text — correctly caught
- **Over-flagged**: current_meds as "major error" — patient is between regimens, empty is reasonable

---

## Meta-Analysis: Auto-Review Quality

### What auto-review got RIGHT
1. ✅ Caught letter garbled text (ROW 82)
2. ✅ Caught missing dose in letter (ROW 36 "reduced from 10 to")
3. ✅ Caught Stage_of_Cancer issues (ROW 15, 59)
4. ✅ Caught goals_of_treatment "adjuvant" error (ROW 59)
5. ✅ Caught readability issues (medical jargon)

### What auto-review got WRONG (systematic false positives)
1. ❌ **current_meds "missing" (6/8 ROWs)**: Auto-review doesn't understand our extraction rule "only cancer-related medications". It flags non-cancer meds (blood pressure, diabetes, etc.) as missing. In reality, 5 of these 6 are correct — only ROW 43 truly has missing cancer meds (chemo drugs)
2. ❌ **Severity inflation**: Many P2 issues marked as P1. Auto-review lacks calibration on what constitutes "major error" vs "minor issue"
3. ❌ **Letter specificity demands**: Auto-review wants the letter to include specific cancer subtypes and technical terms, but the letter prompt deliberately simplifies. Auto-review and letter prompt have conflicting goals

### Recommendations for auto_review.py
1. Add to review prompt: "current_meds should ONLY contain cancer-related drugs. General medications (blood pressure, diabetes, vitamins) are correctly excluded"
2. Add severity calibration: "P1 = changes clinical meaning or could mislead. Missing non-cancer meds is NOT P1"
3. Add letter context: "The letter is for patients at 8th-grade reading level. Using 'pancreatic cancer' instead of 'pancreatic neuroendocrine tumor' is acceptable simplification for follow-up patients"
