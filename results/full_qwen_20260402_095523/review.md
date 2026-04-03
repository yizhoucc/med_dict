# Letter v8a Full Run Review

**Run**: full_qwen_20260402_095523
**Date**: 2026-04-02
**Samples**: 100 (full CORAL breastca_unannotated.csv)
**Pipeline**: v2 extraction + v8a letter generation

## Executive Summary

| Metric | Result |
|--------|--------|
| P0 (hallucination) | 0 |
| P1 (systematic) | 1 pattern (emotion false positive, ~20+ samples) |
| P2 (minor) | 3 patterns |
| Template compliance | 100/100 |
| Letters complete (not truncated) | 97/100 (97%) |
| [REDACTED] leak in letters | 0/100 |
| Dosing leak in letters | 0/100 |
| Jargon leak (TCHP) | 3/100 |
| Avg readability grade | 5.7 (target <8.0) |
| Avg field coverage | 72% |

---

## P1: Emotion Detection False Positive (FIXED)

**Problem**: The emotion keyword detector (`_EMOTION_KEYWORDS`) searches for words like "anxiety", "depression", "anxious" using simple substring matching (`kw in note_lower`). This fails on negated ROS entries like:
- "PSYCH: **No** depression, or anxiety or trouble sleeping."
- "Psychiatric: **Negative** for depression, anxiety"

The detector matches "anxiety" and "depression" in these **negated** statements, causing the letter to say things like "It seems you are feeling anxious and depressed" when the patient explicitly denied these symptoms.

**Clinical impact**: Telling a patient "you seem anxious and depressed" when they are NOT is clinically inappropriate and could cause unnecessary worry.

**Affected samples**: All samples whose ROS mentions "No depression/anxiety" (estimated 20+ of 100).

**Fix applied**: Commit `6b6f971d` — Added negation filter. Before accepting a keyword match, checks the preceding 40 characters for negation words (no, not, denies, denied, negative, without, absent). If negation found, keyword is excluded.

**Status**: Fixed in code. Needs re-run of letter generation (extraction unchanged).

---

## P2 Issues

### P2-1: Letter truncation (3/100)

**Affected**: ROW 32, ROW 84, ROW 99

All three are complex cases with many treatment details. The letter reached max_new_tokens (768) before completing.

ROW 32 ends with: `"...Please let us know if you need any help with this. ["`
ROW 84 ends with: `"...Please let us know if you need support.\nThank you"`
ROW 99 ends with: `"...We will also discuss ways to manage your anxiety and depression.\nThank you"`

**Fix**: Increase max_new_tokens to 1024 for letter generation. 768 is sufficient for 97% of cases but complex metastatic cases need more room.

### P2-2: TCHP abbreviation leaked (3/100)

**Affected**: ROW 15, ROW 17, ROW 52

Letters contain "TCHP" (a chemotherapy regimen abbreviation) without explanation. Should be written as "a combination of chemotherapy drugs" or expanded.

**Fix**: Add TCHP/AC/TC to the medical terms list in the prompt, or add a POST check to replace chemo abbreviations.

### P2-3: Low field coverage (<60%)

**Affected**: 14 samples with coverage below 60%

| ROW | Coverage | Key fields missed |
|-----|----------|-------------------|
| 4 | 58% | imaging_plan, lab_plan, supportive_meds |
| 6 | 54% | current_meds, findings, goals_of_treatment |
| 12 | 62% | imaging_plan, radiotherapy_plan |
| 23 | 54% | goals_of_treatment, findings |
| 28 | 58% | current_meds, radiotherapy_plan |
| 34 | 50% | findings, goals_of_treatment, response_assessment |
| 35 | 58% | findings, current_meds |
| 44 | 53% | current_meds, findings |
| 48 | 54% | findings, goals_of_treatment |
| 51 | 55% | goals_of_treatment, findings |
| 69 | 55% | current_meds, findings |
| 72 | 53% | findings, lab_summary |
| 77 | 56% | current_meds, lab_summary |
| 84 | 50% | findings, response_assessment |

Pattern: Most commonly missed fields are `findings` and `goals_of_treatment` in the "What was discussed?" section. These fields contain information but the LLM doesn't always reference them with the correct source tag, reducing measured coverage. The content may still be present but untagged.

**Fix**: Lower priority. Source tag coverage is a tracking metric, not a content issue. Spot-checks show the content is usually present but tagged under a different field name.

---

## Metrics Summary

### Readability (Flesch-Kincaid Grade)
- Min: 3.9, Max: 7.5, Mean: 5.7, Median: 5.6
- 100% below 8th grade target
- Note original mean: 11.0 (significant simplification achieved)

### Field Coverage
- Min: 50%, Max: 100%, Mean: 72%, Median: 71%
- 2 samples at 100% coverage
- 14 samples below 60%

### Template Compliance
- 100/100 samples follow 5-section template structure
- All section headers present (**Why did I come?** / **What was discussed?** / etc.)

### POST Check Performance
- 0 [REDACTED] leaks in any letter
- 0 dosing detail leaks
- 0 TNM staging notation in letters
- POST-LETTER checks had zero warnings (all handling done pre-LLM)

---

## v8a Fixes Verification (from v8 test)

| Fix | Status in full run |
|-----|-------------------|
| peritoneum explained | Verified in S0: "the lining of your abdomen (belly area)" |
| axilla → armpit | Verified: "armpit area" used consistently |
| Advance care included | Verified: "Your advance care status is noted as Full code" appears where applicable |
| Mixed response both sides | Verified in S11: "stable, but...new small areas of cancer in the brain" |
| Why section no jargon | Mostly compliant, 3 exceptions with TCHP |

---

## Recommended Next Steps

1. **Re-run letter-only** with emotion negation fix (already committed)
2. **Increase max_new_tokens** to 1024 for letter generation
3. **Add TCHP/AC/TC** to prompt Rule 15 medical terms list
4. Field coverage improvements are lower priority — content is present, just tagging inconsistency
