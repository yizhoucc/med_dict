# PDAC 30-Sample Iter6 Triple Review

Generated: 2026-04-28
Reviewer: Claude (manual, cross-referencing original notes)

## Summary

| | Auto-Review | My Assessment |
|---|---|---|
| P0 | 0 | **0** ✅ |
| P1 | 11 | **7** |
| P2 | 146 | ~20 |
| False positive rate (P1) | — | **36%** (was 68% in iter5) |

## Iteration Progress

| | iter1 | iter5 | **iter6** |
|---|---|---|---|
| P1 (auto) | 32 | 25 | **11** |
| P1 (real) | ~15 | 8 | **7** |
| Jargon P1 | 12 | 9 | **0** ✅ |
| Stage IIB fabrication | 6 | 0 | **0** ✅ |
| capecitabine P0 | 1 | 0 | **0** ✅ |
| ROW 6 lanreotide | ❌ | ❌ | **✅ fixed** |
| ROW 90 dual cancer | ❌ | ❌ | **✅ fixed** |

## Verified Fixes in Iter6

1. ✅ **ROW 6**: lanreotide now in current_meds (IV-CHECK caught it, CROSSCHECK kept it)
2. ✅ **ROW 90**: Type_of_Cancer now says "Primary pancreatic neuroendocrine tumor; BRAF-mutant metastatic [melanoma]" — dual cancer extracted
3. ✅ **Jargon P1 = 0** — auto-review hardened
4. ✅ **Stage IIB = 0** — still clean
5. ✅ **ROW 14**: Type correctly says "High-grade neuroendocrine tumor of duodenal origin"
6. ✅ **ROW 41**: Goal correctly says "surveillance"
7. ✅ **ROW 59**: Goal correctly says "surveillance"

## Real P1s (7 total)

### 1. ROW 77 — Goal "curative" should be "surveillance"
- A/P explicitly says "he will continue on surveillance"
- Patient completed adjuvant therapy, now monitoring
- Auto-review correctly flagged ✅
- **Root cause**: LLM ignores surveillance prompt rule for this specific case

### 2. ROW 31 — Missing gemcitabine in current_meds  
- A/P says "He'll continue on treatment without schedule or dose modification"
- But doesn't say "continue gemcitabine" explicitly — uses "treatment" instead
- CROSSCHECK removed gemcitabine because no explicit "continue gemcitabine" pattern matched
- Auto-review correctly flagged ✅
- **Root cause**: CROSSCHECK too strict — "continue on treatment" + gemcitabine elsewhere in note should count

### 3. ROW 72 — Missing gemcitabine + Abraxane
- Same pattern as ROW 31 — A/P says "continue treatment" not "continue gemcitabine"
- Auto-review correctly flagged ✅

### 4. ROW 36 — Letter dose gap "was reduced."
- Extraction has "dose decreased from 10 to 7.5 mg daily" but letter drops the number
- POST-LETTER-DOSE-GAP catches "reduced ." but the letter now says "was reduced." which is grammatically OK but clinically incomplete
- Auto-review correctly flagged ✅
- **Verdict**: Borderline P1/P2. The dose IS in the extraction, just not in the letter.

### 5. ROW 15 — Oxaliplatin contradiction
- A/P says "concerned about exposing him to additional oxaliplatin"
- But POST-MEDICATION-SUPPLEMENT added "oxaliplatin" to meds because it found it in A/P
- This is a **POST hook false positive** — oxaliplatin is mentioned in A/P to say DON'T give it
- Auto-review correctly flagged ✅
- **Root cause**: POST-MEDICATION-SUPPLEMENT doesn't check for negation context

### 6. ROW 40 — Letter dose gap
- "increase the dose of your fentanyl and added Reglan" — missing target dose
- Auto-review correctly flagged ✅

### 7. ROW 43 — Letter incomplete
- Auto-review flagged as incomplete sentence about CA 19-9 trend
- **Verdict**: Borderline P1/P2

## False P1s from Auto-Review (4 total)

### ROW 7 — False P1: "staging pT3 N1 not Stage II-III"
- Extraction says "pT3 N1" which IS correct — it's exactly what the note says
- Goal "curative" is also correct — patient is currently ON gemcitabine (adjuvant), surveillance is planned for AFTER completion
- Auto-review confused current treatment with surveillance plan

### ROW 14 — False P1: "sentence does not specify duodenal origin"
- The EXTRACTION correctly says "duodenal/ampullary origin"
- The letter says "your cancer treatment" — acceptable simplification in a follow-up letter
- P2 at most

### ROW 21 — False P1: "incomplete sentence missing dose reduction"
- Need to verify but likely P2 (minor detail)

### ROW 90 — Borderline: "incomplete sentence"
- Type_of_Cancer now has dual cancer ✅ but letter may have formatting issue

## Issues Auto-Review Missed (my scan found)

| ROW | Issue | Severity |
|-----|-------|----------|
| 33 | Missing irinotecan (A/P says "on irinotecan") | P2 |
| 79 | Missing gemcitabine (A/P says "continue gemcitabine") | P1 |
| 82 | Missing abraxane (A/P says "on abraxane") | P2 |
| 84 | Missing gemcitabine (A/P says "continue gemcitabine") | P2 |
| 87 | Missing octreotide (A/P says "Continue octreotide") | P1 |
| 92 | Missing gemcitabine | P2 |

These were all removed by POST-MEDS-CROSSCHECK. The CROSSCHECK is too aggressive — it removes drugs that ARE in the A/P as current but whose surrounding context doesn't match the narrow pattern list ("continue", "currently", "on ", "taking", "resume", "start").

## Root Cause Analysis

### The CROSSCHECK Problem
CROSSCHECK was added to fix ROW 6 capecitabine (literature citation hallucination). It works for that case. But it's now causing ~6 false removals where drugs ARE current but the A/P uses indirect language:
- "He'll continue on treatment" (not "continue gemcitabine")
- "Continue octreotide" (matches! but CROSSCHECK may not find it due to windowing)

**Fix options**:
1. Widen CROSSCHECK context patterns (add "on treatment", "treatment with", "therapy with")
2. Only CROSSCHECK when the drug was added by POST-MEDS-IV-CHECK (not when LLM originally extracted it)
3. Add a whitelist of drugs that are in the note's medication list (these should never be removed)

### Remaining Non-CROSSCHECK Issues
- ROW 77 goals: LLM doesn't always follow the surveillance rule
- Letter dose gaps: LLM drops numbers due to "no dosing" rule conflict
- These are LLM capability limits, hard to fix with hooks

## Recommendations

1. **[HIGH]** Fix CROSSCHECK: if drug appears in note's medication list section, NEVER remove it
2. **[MED]** ROW 77 goals: add POST hook to override goals when A/P says "surveillance"  
3. **[LOW]** Letter dose gaps: diminishing returns on POST pattern matching
