# Letter v8b Full Run Review

**Run**: full_qwen_20260403_092917
**Date**: 2026-04-03
**Samples**: 100 (full CORAL breastca_unannotated.csv)
**Pipeline**: v2 extraction + v8b letter generation

## Executive Summary

| Metric | Result |
|--------|--------|
| P0 (hallucination) | 0 |
| P1 (systematic) | 1 remaining (emotion PMH false positive, 1 sample) |
| P2 (minor) | 4 patterns |
| Template compliance | 100/100 |
| Letters complete (not truncated) | 98/100 (98%) |
| [REDACTED] leak | 0/100 |
| Dosing leak | 0/100 |
| TCHP jargon leak | 4/100 |
| Emotion false positive | 1/100 (down from ~20+ in v8a) |
| Avg readability grade | 5.8 (target <8.0) |
| Avg field coverage | 71% |

### Comparison with v8a

| Metric | v8a | v8b | Change |
|--------|-----|-----|--------|
| Truncations | 3 | 2 | -1 |
| Emotion false positive | ~20+ | 1 | -95%+ |
| TCHP jargon | 3 | 4 | +1 |
| Avg readability | 5.7 | 5.8 | ~same |
| Avg coverage | 72% | 71% | ~same |
| Port/echo extraction | missed | captured | fixed |

---

## P1: Emotion PMH False Positive (1 sample)

**Affected**: ROW 17

**Problem**: Patient's PMH lists "h/o depression & anxiety" (medical history, not current emotional state). The negation filter checks for "no/not/denies/negative" but doesn't filter PMH context markers like "h/o", "history of", "PMH".

The letter says: "Your care team is aware that you are feeling anxious and depressed."

**Note**: ROW 99 also says "feeling anxious and depressed" but this is CORRECT — patient has active anxiety/depression per PE ("Anxious but alert") and ROS.

**Fix**: Add PMH context markers to the negation filter: "h/o ", "history of ", "past medical", "pmh"

---

## P2 Issues

### P2-1: Letter truncation (2/100)
- ROW 2: complex Stage IV TNBC case, letter cut at "S" (Sincerely)
- ROW 84: complex metastatic case with brain/liver progression, cut at "Thank you"
- Improved from v8a (3→2). Both are extremely information-dense cases.
- **Fix**: Increase max_new_tokens to 1024

### P2-2: TCHP abbreviation leaked (4/100)
- ROW 15, 19, 53, 75: letters contain "TCHP" without explanation
- Slightly worse than v8a (3→4) due to randomness in LLM output
- **Fix**: Add POST check to replace TCHP→"a combination of chemotherapy drugs"

### P2-3: DR-1 metastasis sites not simplified (systemic)
- S0 still lists "lungs, the lining of your abdomen (belly area), liver, and ovaries"
- LLM doesn't consistently follow "3+ sites → use general term" rule
- However, some samples DO follow it (e.g., ROW 12: "spread to other parts of your body, including your brain, lungs, and bones" — 3 sites named, borderline)
- **Fix**: POST check in `_clean_keypoints_for_letter()` to consolidate Metastasis field when 3+ sites

### P2-4: Low field coverage (<60%) — 19 samples
- Most commonly missed source tags: findings, goals_of_treatment, current_meds
- Content usually present in letter but tagged under different field name
- Lower priority — metric issue, not content issue

---

## Doctor Feedback Fixes Verification

### DR-4: Port placement extraction ✅
- Port placement now extracted when mentioned with pre-treatment phrases
- Letters include: "a small device called a port placed" or "a port placed to make it easier to give the chemotherapy"

### DR-5: Echocardiogram extraction ✅
- Echocardiogram/TTE now extracted as pre-chemo imaging
- Letters include: "an imaging test called an echocardiogram before starting the chemotherapy"

### DR-3: pCR misleading statement ✅
- No sample has the misleading "no cancer found in the removed tissue" when LN are positive
- Rule 13 pCR distinction (breast pCR vs complete pCR) is working

### DR-1: Metastasis simplification ⚠️
- Prompt rule exists but LLM inconsistently follows it
- Needs code-level enforcement (POST check)

### Emotion negation filter ✅ (partial)
- Negated ROS ("No depression, no anxiety") no longer triggers false positive
- PMH ("h/o depression & anxiety") still triggers — needs "h/o" added to filter
- Reduced from ~20+ false positives to 1

---

## Metrics Summary

### Readability (Flesch-Kincaid Grade)
- Min: 4.3, Max: 7.6, Mean: 5.8, Median: 5.7
- 100% below 8th grade target
- Significant simplification from notes (mean 11.0 → 5.8)

### Field Coverage
- Min: 45%, Max: 100%, Mean: 71%, Median: 70%
- 19 samples below 60%

### Template Compliance
- 100/100 follow 5-section template structure

### Safety
- 0 [REDACTED] leaks
- 0 dosing leaks
- 0 TNM notation leaks

---

## Recommended Next Steps

1. **Add "h/o"/"history of"/"PMH" to emotion negation filter** — fix last false positive
2. **Increase letter max_new_tokens to 1024** — fix 2 remaining truncations
3. **Add TCHP POST check** — replace abbreviation in generated letters
4. **DR-1 POST check** — consolidate metastasis sites in `_clean_keypoints_for_letter()` when 3+ sites listed
