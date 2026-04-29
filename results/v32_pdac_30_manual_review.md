# PDAC 30-Sample Manual Review

Generated: 2026-04-28
Reviewer: Claude (manual, cross-referencing original notes)

## Summary

- **Samples**: 30
- **P0** (hallucination): **1** (ROW 6 capecitabine → should be lanreotide)
- **P1** (major error): **10** real P1s
- **P2** (minor): ~15
- **Auto-review** reported P1=32, of which ~10 are real → ~68% false positive rate (improved from 60% but still high)

## Systemic Issues (patterns across multiple samples)

### 1. Stage IIB Fabrication — **P1, 6 ROWs** (ROW 6, 18, 41, 62, 84, 90)
LLM systematically invents "Stage IIB" for PDAC patients even when the note never mentions this stage. The model seems to infer Stage IIB from common PDAC pathology (pT3N1), but:
- ROW 18: Note says "Stage IB" and "pT2N0" — LLM wrote "Stage IIB" (completely wrong)
- ROW 6, 41, 84, 90: Note mentions NO stage at all — LLM fabricated "Stage IIB"
- ROW 62: Note has "pT3 N1" but never says "Stage IIB" — inference is reasonable but should say "inferred" not "Originally Stage IIB"

**Fix**: Add to PDAC extraction prompt: "Do NOT write 'Stage IIB' or any specific AJCC stage number unless the note EXPLICITLY states it. If the note only has pTN notation, write the pTN (e.g., 'pT3N1') not the inferred stage name."

### 2. Drug Name Hallucination — **P0, 1 ROW** (ROW 6)
current_meds: "capecitabine" but patient is on lanreotide. capecitabine appears only in a literature citation in the A/P. The letter then says "You continue to take capecitabine."

**Fix**: This is an LLM error, hard to fix via prompt. Could add POST hook to cross-validate current_meds against medication list.

### 3. Incomplete Letter Sentences — **P1, ~6 ROWs** (ROW 15, 32, 33, 36, 40, 62, 87, 90)
Various patterns of missing information in letter: "reduced .", "increased to and", missing drug names. POST hooks partially fix this but some patterns still slip through.

**Fix**: Continue expanding POST-LETTER-DOSE-GAP patterns.

### 4. Goals "curative" for Surveillance Patients — **P1, 2 ROWs** (ROW 41, 77)
Patients who completed treatment and are on surveillance get "curative" instead of "surveillance". The prompt fix from iter2 helps (ROW 59 now correctly says "surveillance") but doesn't catch all cases.

**Fix**: Strengthen the surveillance rule in prompt. Also: ROW 77 is a new patient post-surgery, "curative" may actually be correct there (adjuvant intent).

### 5. Tumor Origin Mislabeling — **P1, 1 ROW** (ROW 14)
Note says "neuroendocrine tumor of duodenal origin" but extraction labels it as pancreatic. The PDAC prompt assumes everything is pancreatic.

**Fix**: Add to extraction prompt: "If the note specifies a non-pancreatic origin (duodenal, ampullary, biliary), include the correct origin in Type_of_Cancer."

### 6. Medical Jargon in Letters — **P2, many ROWs**
Auto-review flags unexplained jargon (IGF-1, CA 19-9, adenocarcinoma) as P1 but these are mostly P2. CA 19-9 is explained in our term map. adenocarcinoma has a plain-language explanation in the letter prompt.

**Status**: Mostly false positives from auto-review. Real readability is acceptable (grade 5-8 range).

---

## Per-ROW Findings (P1+ only)

| ROW | Severity | Issue | Auto-Review Caught? |
|-----|----------|-------|---------------------|
| 6 | **P0** | current_meds "capecitabine" is hallucination — patient is on lanreotide | ✅ Yes |
| 6 | P1 | Stage "IIB" fabricated | ✅ Yes (indirectly) |
| 7 | P2 | Stage "pT3 N1" not translated — acceptable | Auto-review over-flagged as P1 |
| 14 | P1 | Tumor origin: duodenal, not pancreatic | ✅ Yes |
| 18 | P1 | Stage "IIB" when note says "IB" + "pT2N0" — directly contradicts note | ✅ Yes |
| 18 | P1 | Lab values may be fabricated (auto-review flagged) | Need to verify |
| 41 | P1 | Stage "IIB" fabricated + goals "curative" on surveillance | ✅ Partially |
| 62 | P1 | Stage "IIB" fabricated (note has pT3N1 but never says IIB) | Not flagged |
| 77 | P2 | Goals "curative" — debatable, new patient post-surgery | ✅ Flagged |
| 84 | P1 | Stage "IIB" fabricated | Not flagged |
| 90 | P1 | Stage "IIB" fabricated + Type_of_Cancer empty | P2 only |

## Clean ROWs (no real issues found)

ROW 1, 4, 15, 17, 21, 29, 31, 33, 35, 36, 43, 59, 72, 79, 82, 87, 91, 92, 98

## Recommendations (priority order)

1. **[P0 FIX]** POST hook: cross-validate current_meds against medication list in note
2. **[P1 FIX]** Extraction prompt: prohibit fabricated AJCC stage numbers — only extract what note explicitly says
3. **[P1 FIX]** Extraction prompt: respect tumor origin (duodenal/ampullary/biliary)
4. **[P1 FIX]** Strengthen surveillance goal detection
5. **[P2 FIX]** Continue expanding letter dose-gap POST hooks
6. **[META]** Auto-review still has ~68% P1 false positive rate — needs further prompt tuning
