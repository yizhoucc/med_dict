# PDAC 100-Sample — Doctor Feedback (Simulated)

**Reviewer:** Simulated oncologist perspective
**Date:** 2026-04-30
**Dataset:** 100 PDAC unannotated notes, iter4

---

## Feedback Items (8 actionable, prioritized)

### Item 1 — P1: ROW 5 "dose-modified a medication" garbled text
**Quote from letter:** "palliative chemotherapy consisting of dose-modified a medication (oxaliplatin at 80% and omission of the 5-a medication bolus)"

**Problem:** After FOLFOX→"a medication" replacement, the text becomes garbled. "dose-modified a medication" and "5-a medication bolus" are nonsensical to a patient. The original regimen name was replaced but the surrounding dose-modification language references the original components.

**Fix:** When a chemotherapy regimen name is replaced with "a medication"/"a chemotherapy regimen", also clean up surrounding dose-modification references that reference individual drug components. OR: don't replace the regimen name at all for PDAC (PDAC patients often know their regimen names like FOLFIRINOX, unlike breast cancer patients who may not know TCHP).

---

### Item 2 — P1: ROW 4 "You was discussed" grammar
**Quote from letter:** "You was discussed the option of transitioning to hospice care"

**Problem:** The POST-LETTER-VOICE fix changed "The patient" → "You" but didn't fix the surrounding verb tense. "The patient was discussed" → "You was discussed" is grammatically wrong.

**Fix:** Add verb conjugation fix: "You was" → "You were".

---

### Item 3 — P1: ROW 36 dose missing in letter
**Quote from letter:** "Your dose of everolimus was reduced from 10 ." and "at the reduced dose of ."

**Problem:** The extraction correctly has "dose decreased from 10 to 7.5 mg daily" but the letter drops the 7.5 target. The "no dosing details" rule removes it. For dose CHANGES, patients need to know both the old and new dose.

**Fix:** Modify letter prompt: "When describing dose CHANGES (increased, reduced, adjusted), you MAY include the specific doses to help the patient understand the change. Only omit routine maintenance doses."

---

### Item 4 — P1: ROW 40 plan section empty
**Quote from letter:** After "What is the plan going forward?" there is NO content — it jumps straight to "Thank you for trusting us with your care."

**Problem:** The A/P clearly describes a treatment plan (gemcitabine + Abraxane, DVT workup) but the letter's plan section is completely empty. The treatment plan was placed in the "medication changes" section instead.

**Fix:** The letter prompt should enforce that "What is the plan going forward?" is NEVER empty. If the LLM puts plan content in the wrong section, a POST hook should move it.

---

### Item 5 — P2: ROW 3 "unspecified agent regardless of actual origin of your malignancy"
**Quote from letter:** "You will continue the unspecified agent regardless of the actual origin of your malignancy."

**Problem:** This is doctor-to-doctor language ("regardless of the actual origin of her malignancy") that leaked into the patient letter. A patient would be alarmed to read that doctors don't know where their cancer came from. The clinical context (responding well to treatment, will re-image to clarify) should be framed reassuringly.

**Fix:** Letter prompt should flag "regardless of" + "malignancy" patterns. Rephrase as: "Your current treatment is working well, and we will continue it while we gather more information."

---

### Item 6 — P2: ROW 9 "oxaliplatin was dose-reduced ." (missing amount)
**Quote from letter:** "irinotecan was dose-reduced by 25%, and oxaliplatin was dose-reduced ."

**Problem:** Same as Item 3 — dose reduction amount missing for one drug but present for another. Inconsistent within the same sentence.

**Fix:** Same as Item 3 — allow dose change numbers in letter.

---

### Item 7 — P2: ROW 24 "cancer may not be responding well"
**Quote from letter:** "These findings are concerning for possibly pulmonary metastases, indicating that the cancer may not be responding well to the prior treatment."

**Problem:** The note says "small, minimally hypermetabolic pulmonary nodules" which are suspicious but NOT confirmed metastases. The letter's language ("cancer may not be responding well") is too definitive and could cause unnecessary anxiety. The A/P says "closely monitor" and "repeat scan" — indicating the doctor is watching, not concluding progression.

**Fix:** Letter should say "Some small spots were found in the lungs that need to be watched closely. We will repeat the scan in February to check on them." — factual without premature conclusions.

---

### Item 8 — P2: ROW 5 "a medication/a medication" repeated
**Quote from letter:** "Future consideration for an alternative regimen such as a medication may be considered... Repeat imaging/bloodwork is planned between a medication/a medication."

**Problem:** When multiple drug names are [REDACTED] or replaced, the letter says "a medication/a medication" which is meaningless. Multiple sequential replacements create unreadable text.

**Fix:** When 2+ consecutive "[REDACTED]" or replaced drug names appear, collapse them: "between treatment cycles" or "as planned with your care team".

---

## Summary of Fixes Needed

| # | Issue | Type | Fix Approach |
|---|-------|------|-------------|
| 1 | Garbled "dose-modified a medication" | Letter POST | Don't replace regimen names in PDAC, OR clean up dose-modification context |
| 2 | "You was discussed" | Letter POST | Add "You was" → "You were" to grammar fix |
| 3 | Dose changes missing numbers | Letter prompt | Allow dose change amounts (from X to Y) |
| 4 | Empty plan section | Letter POST | Enforce non-empty plan section |
| 5 | Doctor language leaked | Letter POST | Flag "regardless of" + "malignancy" |
| 6 | Inconsistent dose info | Letter prompt | Same as #3 |
| 7 | "Not responding well" premature | Extraction/Letter | Response assessment should distinguish confirmed vs suspected |
| 8 | "a medication/a medication" | Letter POST | Collapse multiple consecutive replacements |
