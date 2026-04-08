# Letter Generation Test Review

**Date**: 2026-03-26
**Config**: exp/test_letter.yaml (3 samples: rows 49, 78, 83)
**Model**: Qwen2.5-32B-Instruct-AWQ
**Prompt version**: letter_generation.yaml v1 (rows 49/78) + v2 with "no JSON" fix (row 83)

## Summary

| Row | Coral | Sentences | Attributed | Time   | JSON wrapper? |
|-----|-------|-----------|------------|--------|---------------|
| 49  | 189   | 13        | 11 (85%)   | 20.5s  | Yes           |
| 78  | 218   | 17        | 15 (88%)   | 27.5s  | Yes           |
| 83  | 223   | 14        | 12 (86%)   | 31.6s  | No            |

---

## Row 49 (coral 189) — Metastatic HR+/HER2- IDC, second opinion, televisit

### Letter Quality
- **Readability**: Good. 8th-grade level. Terms like "IDC" and "DCIS" kept as acronyms but explained contextually ("a type of breast cancer called HR+ and HER2- invasive ductal carcinoma (IDC) with DCIS"). Could explain DCIS more explicitly.
- **Accuracy**: Mostly faithful to keypoints.
- **Completeness**: Covers diagnosis, stage, metastasis, labs, findings, meds, goals, procedures, genetics, follow-up, advance care. Skipped: recent_changes (empty), response_assessment (partially merged into findings sentence), supportive_meds, imaging_plan, lab_plan, radiotherapy_plan (all "none/no plan").
- **Tone**: Warm, patient-friendly.

### Issues

| # | Severity | Field | Issue |
|---|----------|-------|-------|
| 1 | P1 | format | Letter wrapped in `{"letter": "..."}` JSON — `_unwrap_json_shell` failed (old prompt, real newlines in JSON string). **Fixed in prompt v2**. |
| 2 | P2 | Type_of_Cancer | "This means the cancer cells have receptors for hormones but not for HER2" — decent explanation but could be simpler for patients unfamiliar with "receptors". |
| 3 | P2 | summary | Letter says "second opinion" correctly but doesn't mention it's a video/televisit — minor omission, low priority. |
| 4 | P2 | response_assessment | "under good control" mentioned in findings sentence but not as a standalone response_assessment sentence — merged, acceptable. |

### Traceability Check
- 11/13 sentences attributed (85%)
- Sentence [0] text polluted with `{"letter":` prefix due to JSON wrapper — will be fixed by `_unwrap_json_shell` regex fallback.
- Sentence [12] is trailing `"` artifact from JSON wrapper — same fix.
- All attributed sentences have correct field mapping and extraction values.
- 9/11 attributed sentences also have note quotes (attribution data).

---

## Row 78 (coral 218) — Metastatic ER+/HER2+ breast cancer, palliative, in-person

### Letter Quality
- **Readability**: Excellent. Explains "pleural effusion" as "fluid around your lungs", "thoracentesis" as "check the fluid around your lungs". "PET/CT scan" left as-is but described functionally ("to see where the cancer is").
- **Accuracy**: Faithful to keypoints. Correctly captures the urgency (stopped all treatment, rising markers, pain management).
- **Completeness**: Covers diagnosis, metastasis, response, findings, labs, meds (current + supportive), changes, goals, medication plan, procedures, imaging, specialty referral, follow-up. Skips: advance care ("not discussed"), radiotherapy/therapy (none), genetics referral (none), lab_plan (none).
- **Tone**: Warm, empathetic. "Our goal is to make you as comfortable as possible" — appropriate for palliative context.

### Issues

| # | Severity | Field | Issue |
|---|----------|-------|-------|
| 1 | P1 | format | Same JSON wrapper issue as Row 49. |
| 2 | P2 | lab_summary | "but we need to keep an eye on your INR, which is stable at 2.1" — INR stable at 2.1 is a clinical detail that may worry patients unnecessarily. Accurate per keypoints though. |
| 3 | P2 | supportive_meds | Letter correctly combines supportive_meds + current_meds into one sentence [6] — good merging. |

### Traceability Check
- 15/17 sentences attributed (88%)
- Same JSON wrapper artifacts in sentence [0] and [16].
- Two medication_plan sentences ([9] and [10]) both point to the same field — correct, as the plan has multiple components.
- All field mappings verified correct.

---

## Row 83 (coral 223) — Metastatic ER+/HER2- IDC, palliative, televisit (NEW PROMPT)

### Letter Quality
- **Readability**: Excellent. Best of the three — no JSON wrapper, clean text. Explains "lumbar puncture" as "check the fluid around your brain and spinal cord", "CT scan of your chest, abdomen, and pelvis" instead of "CT CAP".
- **Accuracy**: Faithful. Correctly notes dose escalation history, disease progression, palliative intent.
- **Completeness**: Covers visit type, diagnosis, metastasis, labs, findings (MRI brain), meds, med changes, goals, response, medication plan, imaging, procedures, radiation referral, follow-up. Skips: advance care (not discussed), supportive_meds (empty), genetic_testing (none), lab_plan (none).
- **Tone**: Warm, honest about disease progression without being alarming. "suggesting the current treatment may not be working as well as we hoped" — well-phrased.

### Issues

| # | Severity | Field | Issue |
|---|----------|-------|-------|
| 1 | P2 | Type_of_Cancer | "ER+ and HER2- invasive ductal carcinoma, which means it has spread to your bones..." — the "which means" clause describes metastasis, not the cancer type itself. The receptor status explanation is missing. Minor readability issue. |
| 2 | P2 | Distant Metastasis | Distant Metastasis not separately tagged — merged into Type_of_Cancer sentence [2] with Metastasis tag. Acceptable. |
| 3 | P2 | Stage_of_Cancer | Stage not mentioned in letter — correct because keypoints says "Not available (redacted)". |
| 4 | INFO | second opinion | Keypoints say "no" for second opinion, but note shows it IS a consultation/referral. This is an extraction issue, not a letter issue. |

### Traceability Check
- 12/14 sentences attributed (86%)
- Sentence [0] "Dear Patient," correctly tagged [source:none].
- Sentence [13] "Sincerely, Your Care Team" correctly tagged [source:none].
- No unattributed sentences (unlike rows 49/78 with trailing `"` artifact).
- All field mappings verified correct.
- 10/12 attributed sentences have both extraction values AND note quotes.

---

## Cross-Row Patterns

### What works well
1. **8th-grade readability**: Medical terms consistently explained in plain language.
2. **Source tagging accuracy**: 85-88% attribution rate. Unattributed sentences are only greetings/closings or JSON artifacts.
3. **Content coverage**: Important fields consistently covered; empty/none fields correctly skipped.
4. **Multi-source tags**: Used correctly (e.g., `[source:Type_of_Cancer,Metastasis,Distant Metastasis]`).
5. **Traceability chain**: sentence -> field value -> note quote works end-to-end.

### Issues to fix
1. **P1: JSON wrapper** — LLM wraps output in `{"letter": "..."}` 2/3 times (old prompt). New prompt fixes this. `_unwrap_json_shell` regex fallback added for robustness.
2. **P2: DCIS/receptor explanation** — Could be more patient-friendly. Consider adding example explanation in prompt.
3. **P2: "which means" clause ambiguity** — Row 83 sentence [2] conflates cancer type with metastasis. Minor.

### Metrics
- Average letter length: ~14.7 sentences
- Average attribution rate: 86%
- Average generation time: ~26.5s per sample
- Total pipeline overhead: ~20-30s per sample (on top of extraction + attribution)
