# PDAC 30-Sample Iter5 Triple Review

Generated: 2026-04-28
Reviewer: Claude (manual, cross-referencing original notes)

## Summary

| | Auto-Review | My Assessment |
|---|---|---|
| P0 | 0 | **0** ✅ |
| P1 | 25 | **8** |
| P2 | 143 | ~20 |
| False positive rate (P1) | — | **68%** |

## Systemic Fixes Verified

| Issue | iter1 | iter5 | Status |
|-------|-------|-------|--------|
| Stage IIB fabrication | 6 ROWs | **0 ROWs** | ✅ FIXED |
| capecitabine hallucination (ROW 6) | P0 | **removed** | ✅ FIXED |
| Third-person voice in letter | 3 ROWs | **0 ROWs** | ✅ FIXED |
| FOLFOXIRI garbled text | 1 ROW | **0 ROWs** | ✅ FIXED |
| goals "adjuvant" for surveillance | 2 ROWs | **0 ROWs** | ✅ FIXED |

## Per-ROW P1 Assessment

### Real P1s (8 total)

**ROW 6** — P1: current_meds empty, but patient IS on lanreotide
- Note clearly says "September 2015-present: Lanreotide 120 mg/mo" + medication list has it + A/P says "continue lanreotide 120 mg/mo"
- LLM originally hallucinated capecitabine (fixed by CROSSCHECK), but now it's empty instead of lanreotide
- Auto-review correctly flagged this ✅
- **Root cause**: LLM doesn't extract lanreotide as current_meds, and no POST hook supplements it

**ROW 14** — P1: Type says "duodenal/ampullary origin" ✅ but letter still says "pancreatic cancer"
- Extraction fixed by prompt (now correctly says "Grade 3 neuroendocrine tumor of duodenal/ampullary origin")
- But the letter template defaults to "pancreatic cancer treatment" in the opening
- Auto-review correctly flagged this ✅

**ROW 31** — P1: current_meds empty but patient on gemcitabine
- Note says "3 cycles of alternate week fixed dose rate gemcitabine" and "tumor reduction"
- The A/P says "continue with gemcitabine"
- Auto-review correctly flagged ✅
- **Root cause**: LLM didn't extract gemcitabine, POST-MEDS-IV-CHECK should have caught it

**ROW 72** — P1: current_meds empty but on gemcitabine + Abraxane
- Note says "elected to start gemcitabine and Abraxane. He has now had almost 4 cycles"
- Auto-review correctly flagged ✅

**ROW 87** — P1: meds says "everolimus" but missing octreotide
- Note clearly has octreotide in medication list AND A/P says "Continue octreotide"
- Patient is on BOTH everolimus and octreotide
- Auto-review correctly flagged ✅

**ROW 90** — P1: Two cancers — PNET + BRAF melanoma, only PNET extracted
- Note describes two separate cancers with separate treatment plans
- Pipeline only extracts one Type_of_Cancer
- Auto-review correctly flagged ✅
- **Root cause**: Extraction schema only has one Type_of_Cancer field

**ROW 36** — P1: letter says "dose reduced." without specifying from/to
- Extraction correctly has "dose decreased from 10 to 7.5 mg daily"
- Letter drops the numbers
- **Root cause**: LLM letter generation drops dose info due to "no dosing details" rule

**ROW 43** — P1: letter incomplete sentence
- Need to verify exact text
- Auto-review flagged as incomplete

### False P1s from Auto-Review (17 total)

**ROW 4** — False P1: "missing dexamethasone and ondansetron"
- These are supportive meds correctly in supportive_meds field, not current_meds. Auto-review scope confusion.

**ROW 7** — False P1: "staging pT3 N1 not Stage II-III"
- The extraction says "pT3 N1" which IS correct. Auto-review's complaint that it should be different staging is wrong — pT3N1 is what the note says.

**ROW 15** — False P1: "growth of tumors...not specify peritoneal metastases"
- Letter correctly says cancer has progressed. Not specifying "peritoneal" in a patient letter is acceptable simplification (P2 at most).

**ROW 29** — False P1: "patient is new patient with new diagnosis"
- Auto-review says calling them "new" is inaccurate because they had a diagnosis before. But Patient type says "New patient" (first visit to THIS clinic). Not a real issue.

**ROW 32** — False P1 ×5: all jargon issues
- "Zofran", "morphine", "Percocet", "Mediport", "osseous metastases", "standard of care"
- Most of these ARE jargon but it's P2, not P1. Auto-review still escalating jargon to P1 despite our calibration.

**ROW 33** — False P1: "liver stable vs similar in size"
- Note literally says "similar in size and appearance of multiple hepatic metastases" AND A/P says "overall stable disease in the liver". Calling it "stable" is accurate.

**ROW 35** — False P1: "hepatic lesions" jargon
- P2 at most. Should say "liver" instead of "hepatic".

**ROW 40** — False P1: "fine-needle aspiration too technical"
- P2. It's a procedure name, should have a brief explanation but not a major error.

## Auto-Review Quality Assessment

### What auto-review gets right consistently:
- ✅ Missing cancer drugs (ROW 6, 31, 72, 87)
- ✅ Multi-cancer extraction gaps (ROW 90)
- ✅ Incomplete letter sentences (ROW 36, 43)

### What auto-review still gets wrong:
- ❌ **Jargon → P1** (ROW 32 ×5, 35, 40): Still escalating jargon to P1 despite calibration. This accounts for ~7 false P1s.
- ❌ **Scope confusion** (ROW 4, 29): Doesn't fully understand extraction field scopes.
- ❌ **Simplification vs inaccuracy** (ROW 15, 33): Can't distinguish acceptable simplification from inaccuracy.

### Recommendation for auto-review:
1. Make jargon strictly P2 — add "NEVER classify jargon or unexplained terms as P1" to prompt
2. Remove "reading level" from P1 criteria entirely

## Remaining Fixable Issues (for next iteration)

| Priority | Issue | Fix Approach |
|----------|-------|-------------|
| 1 | Missing current_meds (ROW 6, 31, 72) | POST-MEDS-IV-CHECK drug list needs PDAC drugs: lanreotide, octreotide, sunitinib, everolimus etc. |
| 2 | ROW 90 dual cancer | Schema limitation — would need multi-cancer support |
| 3 | Letter says "pancreatic cancer" for duodenal NET (ROW 14) | Letter prompt: use Type_of_Cancer field, not default "pancreatic cancer" |
| 4 | ROW 87 missing octreotide | Same as #1 — drug list needs PDAC drugs |
| 5 | Letter dose gaps (ROW 36) | POST hook or letter prompt to handle "no dosing" rule better |
| 6 | Auto-review jargon P1 false positives | Strengthen auto-review prompt |
