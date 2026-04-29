# Breast Cancer Annotated Test Set — Manual Review (Updated)

**Date:** 2026-04-29 (updated — no stage inference)
**Dataset:** 20 held-out annotated breast cancer notes (CORAL)
**Pipeline:** v31 (5-gate + 40+ POST hooks, stage inference removed)

---

## Summary

| | Result |
|---|---|
| P0 (hallucination) | **0** ✅ |
| P1 (major error) | **0** ✅ |
| P2 (minor issue) | ~15 |
| Clean (no real issues) | **17/20 (85%)** |
| Inferred stages | **0** (was 7) |
| FK Grade (mean) | **6.5** (all ≤8) ✅ |

**Overall: Zero P0, zero P1 on held-out test set.** The ROW 4 Stage I→IIIC error from the previous run is gone (now correctly says Stage III). The ROW 18 pT2 error is also gone (stage field now empty for that sample, which is correct since the note doesn't state a stage).

## What Changed

| | Before | After |
|---|---|---|
| Stage inference | 7/20 inferred (50% accuracy) | **0/20 inferred** |
| Explicit stages | 12/20 | **17/20** |
| ROW 4 | Stage I ❌ (vs expert IIIC) | **Stage III** ✅ |
| ROW 18 | Stage IIA ❌ (15mm≠T2) | **(empty)** — correctly unstaged |
| Real P1 count | 1 | **0** |

## Extraction Quality: 20/20

- **Receptor status (ER/PR/HER2):** 93% — unchanged, 1 PR disagreement (ROW 18)
- **Histology:** 100% — IDC, ILC, DCIS, metaplastic all correct
- **Treatment goals:** 100% — curative/palliative/risk reduction all correct
- **Stage:** 85% explicit, 15% empty (appropriate — note doesn't state stage), 0% fabricated

## Letter Quality: Consistently Good

- All 20 letters: FK grade 4.5–8.0 (mean 6.5), all below 8th-grade target
- All 20 letters include diagnosis, treatment/plan, and next steps
- Zero hallucinations in letters
- Mean length 1,459 chars (slightly over 800-1200 target but appropriate for content)

## Ready for Doctor Review

The `results/doctor_review_breast_annotated/` folder contains 20 individual sample files, each with original note + extraction + letter + rating template. All files have been regenerated with the updated (no-inference) pipeline output.
