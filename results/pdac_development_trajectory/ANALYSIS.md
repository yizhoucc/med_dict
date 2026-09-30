# PDAC historical development trajectory

## Scope

This analysis reconstructs the requested sequence:

1. `D1-D9`: 30-sample PDAC development subset, iterations 1-9.
2. `F2-F9`: 100-sample PDAC development set, full iterations 2-9.

`D1` is treated as the first ChatGPT-rubric-assisted PDAC adaptation checkpoint, following the project narrative supplied by the authors. Later checkpoints are treated as LLM-review-guided refinements. The archived review files themselves do not contain a ChatGPT rubric identifier, workflow hash, or prompt hash, so this phase label comes from project history rather than embedded provenance.

The primary trajectory uses only files that explicitly identify the reviewer as `Qwen2.5-32B-Instruct-AWQ (auto_review.py)`. Claude-authored manual reviews, simulated doctor feedback, commit-message scans, and later oncologist ratings are excluded from the plotted curves.

## Artifact audit

### Archived Qwen reviews

Thirteen of the 17 requested checkpoints have an archived `auto_review.py` report:

| Checkpoints | Cohort | Archived Qwen summaries | Result files currently present |
|---|---:|---:|---:|
| D1-D9 | 30 samples | 9/9 | D1, D4-D7 |
| F2-F9 | 100 samples | F2-F5 only | 8/8 |

The D2, D3, D8, and D9 result files are not present in the current working tree. Their review documents are present, contain all 30 expected ROW headings, and their detailed P0/P1/P2 table rows reproduce the summary counts exactly.

F6-F9 have complete 100-sample result files but no archived `auto_review.py` summary. Their automated-review values are therefore `NA`. Later commit messages and Claude-authored scans report operational progress, but those values use a different source and are not inserted into the Qwen trajectory.

The repository also contains `v32_pdac_full_review.md`, corresponding to full iteration 1. It is intentionally excluded because the requested trajectory starts the full-set phase at iteration 2.

### Sample and schema checks

- All nine D-stage review files contain the same 30 ROW identifiers: `1, 4, 6, 7, 14, 15, 17, 18, 21, 29, 31, 32, 33, 35, 36, 40, 41, 43, 59, 62, 72, 77, 79, 82, 84, 87, 90, 91, 92, 98`.
- Every available F-stage result contains ROW 1-100.
- Every available result file exposes the same 29-leaf output schema. Its schema signature in the generated CSV is `9827c47fd0fe`.
- The 30-sample subset is contained within the 100-sample set. The cohort change still alters the case mix, so the plot breaks the connecting line between D9 and F2 and reports rates per 100 samples.

## Automated-review findings

### 30-sample phase

Across D1-D9, Qwen P1 flags decreased from 32 to 8:

- Total P1 flags: 32 to 8, a 75.0% decrease.
- P1 flags per 100 samples: 106.7 to 26.7.
- Extraction P1 flags: 9 to 1, an 88.9% decrease.
- Letter P1 flags: 23 to 7, a 69.6% decrease.
- Diagnosis/stage/metastasis P1 flags: 1 to 0.
- Medication/treatment P1 flags: 4 to 1.
- Goals/response P1 flags: 3 to 0.

The P1 trajectory is not monotonic at every step. It is `32, 28, 29, 30, 25, 11, 12, 10, 8`. The largest recorded reduction occurs between D5 and D6, from 25 to 11.

Minor flags did not decline:

- P2 flags increased from 113 to 138.
- Total P0+P1+P2 flags were 145 at D1 and 146 at D9.
- Every archived Qwen report recorded zero fully clean samples because every sample had at least one P2 flag.

The defensible interpretation is that recorded major flags declined, especially for extraction. The archive does not show uniform removal of all reviewer concerns. Some issues shifted to P2, and the automated reviewer continued to flag many minor completeness and readability items.

### 100-sample phase

The available Qwen P1 counts are:

| Checkpoint | P1 | P1 per 100 samples | Extraction P1 | Letter P1 |
|---|---:|---:|---:|---:|
| F2 | 38 | 38 | 11 | 27 |
| F3 | 34 | 34 | 8 | 26 |
| F4 | 32 | 32 | 6 | 26 |
| F5 | 48 | 48 | 14 | 34 |
| F6-F9 | NA | NA | NA | NA |

F2-F4 show a modest reduction in P1 flags. F5 rises sharply. The associated manual-style document explicitly calls the F5 automated count noisy and reports a different assessment, but that document was written by Claude rather than a physician. It cannot replace the Qwen count or serve as clinical ground truth.

There is no archived automated series showing the claimed final decline through F9. The final F6-F9 outputs exist, but a new fixed-rubric review would be required to extend the automated curve. This task intentionally did not rerun a model.

## Reviewer provenance and comparability

The Qwen review files are internally reproducible: their detailed rows match their headline P0/P1/P2 totals. They are not a fixed-validator benchmark for four reasons:

1. `auto_review.py` changed during the trajectory. Git history records reviewer-prompt changes for medication scope, jargon severity, acceptable simplification, and pTN staging in commits `778c389d`, `29a4d58d`, `02265128`, and `eb39b09d`.
2. The review artifacts do not store the exact reviewer prompt or its hash. Some reports were generated before nearby code changes were committed, so an exact prompt-to-report mapping cannot be reconstructed safely.
3. `auto_review.py` truncated long notes to 12,000 characters, keeping the beginning and end. It could miss evidence in the omitted middle.
4. The reviewer was the same Qwen2.5-32B model family used in the pipeline. These counts carry self-evaluation and rubric-calibration bias.

One concrete calibration discrepancy appears at D1. Qwen recorded `P0=0`, while the later Claude/manual-style review called the capecitabine-for-lanreotide mistake a P0. This confirms that severity counts should be described as `automated reviewer flags`, not true hallucination or error counts.

The combined reports also review extraction and patient letters. The current paper uses extraction as its main endpoint, so the extraction-only P1 curve is the most relevant historical result. The letter curve can remain secondary.

## Recommended Results wording

> We retrospectively reconstructed the PDAC development trajectory from archived outputs of the development-time Qwen2.5-32B reviewer. Across nine iterations on a fixed 30-note subset, major findings flagged by the automated reviewer decreased from 32 to 8, corresponding to 106.7 and 26.7 flags per 100 notes. Extraction-related P1 flags decreased from 9 to 1, while letter-related P1 flags decreased from 23 to 7. Minor flags did not decrease, rising from 113 to 138, and the total number of flags remained similar. After expansion to the 100-note development set, P1 flags decreased from 38 at full iteration 2 to 32 at iteration 4, then increased to 48 at iteration 5. Comparable automated-review summaries were not archived for full iterations 6-9. These findings describe the historical behavior of the development-time LLM reviewer and are not clinician-confirmed estimates of accuracy.

This paragraph can be followed by the existing independent oncologist comparison of the final frozen harness. The two sources should remain separate: the historical trajectory explains development, while the oncologist ratings support the final clinical comparison.

## Recommended figure caption

> **Figure X. Historical PDAC development trajectory based on archived automated reviews.** D1-D9 represent nine checkpoints evaluated on the same 30-note development subset. F2-F9 represent full-set iterations on the same 100-note development set. Counts are normalized as automated reviewer flags per 100 notes, and the line is broken at the cohort transition. F6-F9 are shown as unavailable because no matching `auto_review.py` summaries were archived. The Qwen reviewer rubric changed during early development, and its prompt hash was not stored. The figure therefore describes development-time review signals rather than clinician-confirmed error rates. Claude-authored manual follow-up reviews and simulated clinical feedback are excluded.

## Files

- `build_pdac_trajectory.py`: parses archived reports, verifies summary counts, audits result schemas, and regenerates all outputs without an LLM call.
- `pdac_trajectory.csv`: one row per requested checkpoint, including file availability, hashes, sample sets, schema signatures, raw counts, and normalized rates.
- `pdac_flag_categories.csv`: reproducible aggregates by severity, output domain, and text-rule category.
- `pdac_flag_details.csv`: parsed finding text used to produce the aggregates.
- `pdac_trajectory.svg`: four-panel plot of P1, P2, extraction-versus-letter P1, and P1 output domains.

