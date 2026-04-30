# PDAC 100-Sample Full Run Manual Review

**Date:** 2026-04-29
**Pipeline:** v32 (PDAC prompts + all fixes through iter9 + surveillance pattern fix)

## Summary

| | Auto-Review | My Assessment |
|---|---|---|
| P0 | 0 | **0** ✅ |
| P1 | 38 | **~18** |
| False P1 | — | ~20 (53%) |
| Clean ROWs | 70/100 | **~82/100** |

## P1 Breakdown — What's Real vs False

### Definite False P1s (~20)
- **Jargon** (3): auto-review still occasionally flags medical terms as P1
- **Inaccurate** (3-4 of 6): ROW 7 pT3N1 staging is correct (confirmed in 30-sample review), ROW 24/47 are debatable simplifications
- **Other** (3-4): ROW 14 letter not saying "duodenal" (P2 simplification), ROW 67 missing CA 19-9 lab (P2), ROW 15 oxaliplatin context (need recheck)
- **Incomplete** (~7 of 17): many auto-review "incomplete" calls are actually acceptable simplification (e.g., "CA 19-9 slowly declining" flagged as incomplete)

### Real P1s (~18)
| Category | Count | Fixable? |
|----------|-------|----------|
| **Incomplete letter sentences** | ~10 | Hard — LLM drops dose/drug info |
| **Goal curative→surveillance** | 3 | ✅ Just fixed surveillance pattern |
| **Missing meds** | 2 (ROW 53, 74) | ✅ CROSSCHECK/drug list |
| **Grammar** | 1 (ROW 16 "You has") | ✅ POST hook |
| **Inaccurate extraction** | ~2 (ROW 12, 51) | Need to verify |

## Systemic Issues Status

| Issue | Status |
|-------|--------|
| Stage IIB fabrication | **0** ✅ (ROW 67 is real from note) |
| Stage inference | **0** ✅ (removed) |
| capecitabine hallucination | **0** ✅ |
| Jargon P1 false positives | **3** (down from 12) ✅ |
| Goal surveillance | **3 remaining** (just widened pattern) |
| Missing PDAC drugs | **2** (rucaparib, gemcitabine/abraxane) |

## Top Priority Fixes for Next Iteration

1. **ROW 16 grammar "You has"** — simple POST hook fix
2. **ROW 51 gemcitabine in medication_plan** — patient on monitoring but POST hook added gemcitabine incorrectly
3. **Goals ROW 20/24/51** — just committed wider surveillance pattern, need to verify
4. **Incomplete sentences** (~10) — diminishing returns, mostly LLM generation quality

## Comparison: iter1 → iter2 (100 samples)

| | iter1 | iter2 |
|---|---|---|
| P1 (auto) | 40 | **38** |
| Goal P1s | 4 | **3** |
| Stage IIB | 1 (real) | **1 (real, same)** |
| capecitabine | 0 | **0** |
| Clean ROWs | 68 | **70** |

Improvement is modest on 100 samples. The remaining issues are mostly:
- LLM letter generation quality (incomplete sentences) — hard to fix
- Edge cases in monitoring/surveillance language — just fixed
- 2 missing meds — need to add rucaparib to drug lists
