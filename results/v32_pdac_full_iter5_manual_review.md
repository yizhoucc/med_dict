# PDAC 100-Sample Manual Review — iter5 (No Auto-Review, Pure Manual)

**Date:** 2026-05-01
**Method:** Manual scan + spot-check (no auto_review.py used)

## Summary

| | Result |
|---|---|
| **P0** | **0** ✅ |
| **P1** | **9** |
| **P2** | **6** |
| **Clean** | **85/100** |
| FK Grade (mean) | **7.2** (range 5.0-10.4) |
| Letter length (mean) | **1,346 chars** |

## All Issues Found

### P1 (9 total)

| ROW | Issue | Category |
|-----|-------|----------|
| 4 | "You was discussed" — grammar | Grammar |
| 99 | "You is" — grammar | Grammar |
| 7 | "a medication-paclitaxel" — nab-paclitaxel garbled | REDACTED garble |
| 24 | "a medication-paclitaxel" | REDACTED garble |
| 66 | "unspecified agent-paclitaxel" | REDACTED garble |
| 67 | "a medication-paclitaxel" | REDACTED garble |
| 69 | "a medication-paclitaxel" | REDACTED garble |
| 71 | "a medication-paclitaxel" | REDACTED garble |
| 85 | "a medication-paclitaxel" + "a medication-9" + "a medication-a medication" | REDACTED garble (worst) |

### P2 (6 total)

| ROW | Issue | Category |
|-----|-------|----------|
| 9 | "dose-reduced ." — missing amount | Dose gap |
| 36 | "reduced from 10 ." — missing target | Dose gap |
| 5 | FK 10.2 — complex case | Readability |
| 82 | FK 10.4 — complex case | Readability |
| 26 | Technical imaging language ("hypermetabolic", "psoas muscle") | Readability |
| 85 | "a medication-9 level of 35" — CA 19-9 garbled | REDACTED garble |

## Root Cause Analysis

### REDACTED Garble (7 ROWs — the biggest remaining issue)

The CORAL dataset uses ***** to redact drug names. When notes say "nab-paclitaxel" but "nab" is part of a REDACTED block, the pipeline outputs "[REDACTED]-paclitaxel" which `post_check_letter` converts to "a medication-paclitaxel".

Similarly, "CA 19-9" becomes "a medication-9" when "CA" gets REDACTED.

**Fix needed:** POST hook to clean up:
- "a medication-paclitaxel" → "nab-paclitaxel" (the drug name IS in the note)
- "unspecified agent-paclitaxel" → "nab-paclitaxel"
- "a medication-9" → "CA 19-9" (this is a tumor marker, not a medication)
- "a medication-a medication" → "a treatment combination"

### Grammar (2 ROWs)

POST-LETTER-GRAMMAR has patterns for "You has/was/is" but they didn't fire on this run. Need to verify the hook is executing correctly.

### Dose Gaps (2 ROWs)

Persistent issue — LLM drops the target dose when told "no dosing details". The prompt exception ("include dose changes") helps but doesn't fully solve it.

## Fixes for Next Iteration

1. **[P1]** POST hook: "a medication-paclitaxel" → "nab-paclitaxel"
2. **[P1]** POST hook: "a medication-9" → "CA 19-9"
3. **[P1]** POST hook: "a medication-a medication" → "a treatment combination"
4. **[P1]** Verify grammar hook execution for "You was"/"You is"
