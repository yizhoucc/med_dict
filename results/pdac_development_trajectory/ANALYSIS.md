# PDAC extraction development trajectory

## Analysis scope

The main trajectory contains 12 archived checkpoints:

1. `D1-D9`: nine iterations on the same 30-note PDAC development subset.
2. `F2-F4`: three iterations on the same 100-note PDAC development set.

`D1` is labeled as the first ChatGPT-rubric-assisted PDAC adaptation checkpoint, following the authors' project history. Later checkpoints are labeled as LLM-review-guided refinements. The files do not store a ChatGPT rubric identifier or workflow hash, so this phase interpretation does not come from embedded artifact metadata.

The analysis is restricted to extraction P1 findings written in archived reports that explicitly identify the reviewer as `Qwen2.5-32B-Instruct-AWQ (auto_review.py)`. Letter findings, P2 findings, total flags, Claude-authored follow-up reviews, simulated doctor feedback, and physician ratings are outside this trajectory.

F5 is excluded because it was a temporary robustness experiment rather than the final development checkpoint. F6-F9 are also excluded because they did not introduce further intended workflow changes and have no matching `auto_review.py` reports in the working tree or Git history. The only later review artifact is an F7 document written by `Claude (acting as oncologist)`. It is not a physician evaluation and is not comparable to the Qwen series. F4 is therefore treated as the final full-set checkpoint in this trajectory.

## Artifact and comparability audit

- All D1-D9 review files contain the same 30 ROW identifiers.
- All F2-F4 review and result files contain ROW 1-100.
- The 30-note cohort is a subset of the 100-note development set.
- All available result files expose the same 29-leaf extraction schema. The generated schema signature is `9827c47fd0fe`.
- D2, D3, D8, and D9 no longer have their generation result files. Their Qwen review files remain complete. Each contains all 30 ROW sections, and its detailed findings reproduce its summary P0/P1/P2 totals exactly.
- The cohort transition changes both the denominator and case mix. The graph reports flags per 100 samples, draws a dashed D9-to-F2 segment, and labels the expansion from 30 to 100 notes.

## Extraction P1 trajectory

### Fixed 30-note subset

The extraction P1 sequence is:

| Checkpoint | P1 flags | Flags per 100 samples |
|---|---:|---:|
| D1 | 9 | 30.0 |
| D2 | 3 | 10.0 |
| D3 | 5 | 16.7 |
| D4 | 6 | 20.0 |
| D5 | 6 | 20.0 |
| D6 | 4 | 13.3 |
| D7 | 4 | 13.3 |
| D8 | 3 | 10.0 |
| D9 | 1 | 3.3 |

From D1 to D9, extraction P1 flags decreased from 9 to 1, an 88.9% decrease. The path was not monotonic. Counts fell sharply at D2, rose to 6 at D4-D5, and then declined to 1.

A more comparable internal segment is D5-D8. Git history shows no committed `auto_review.py` change between these reviews, although the artifacts still lack an exact reviewer-prompt hash. Extraction P1 flags fell from 6 to 3 across this segment.

The steep D1-to-D2 change should not be used as a clean effect estimate. Both the extraction workflow and the automated-review instructions were being revised during the early D-stage iterations.

### Expanded 100-note set

The extraction P1 sequence is:

| Checkpoint | P1 flags | Flags per 100 samples |
|---|---:|---:|
| F2 | 11 | 11.0 |
| F3 | 8 | 8.0 |
| F4 | 6 | 6.0 |
F2-F4 show a reduction from 11 to 6 flags. F4 is the final full-set checkpoint included in the trajectory.

The D9-to-F2 increase from 3.3 to 11.0 flags per 100 samples should not be interpreted as regression. The development set expanded from a selected 30-note subset to all 100 notes, changing the case mix.

## Clinical problem categories

Categories are assigned only from the extraction field named in each Qwen finding. The script does not reinterpret the note or decide whether the flag is correct. The ten categories are defined before counting, and every category appears for every checkpoint, including zeros:

1. Diagnosis / stage / metastasis
2. Active anticancer medications
3. Treatment changes / therapy plan
4. Treatment goal
5. Response assessment
6. Laboratory results
7. Clinical findings
8. Visit context
9. Follow-up and other plans
10. Other extraction

### Trends that are reasonable to describe

Within the fixed 30-note subset:

- Diagnosis/stage/metastasis flags were 1 at D1-D5 and 0 at D6-D9.
- Active anticancer medication flags decreased from 3 at D1 to 0 at D9, with intermediate fluctuation from 2 to 4.
- Treatment-goal flags decreased from 3 at D1 to 0 at D9. One flag reappeared at D6 before returning to 0.
- Laboratory-result flags decreased from 1 at D1 to 0 for D2-D9.

These are descriptive automated-review trends. The most stable statement is that the fixed 30-note subset ended with fewer extraction P1 flags across diagnosis/staging, active medication, treatment-goal, and laboratory fields.

The strongest within-subset comparison is D5-D8, when Git history shows no committed change to `auto_review.py`. Total extraction P1 flags fell from 6 to 3. Active-medication flags fell from 4 to 2, and diagnosis/stage/metastasis flags fell from 1 to 0. This remains an automated-review comparison because the exact prompt hash was not embedded in the reports.

Within F2-F4, several categories also declined:

- Treatment changes and plans: 3 to 1.
- Treatment goals: 3 to 1.
- Active anticancer medications: 3 to 2.

F2-F4 is the strongest full-set comparison. The same 100 notes were reviewed after the last committed reviewer-prompt update, and extraction P1 flags fell from 11 to 6. This result can be reported as a development-time automated-review trend. It does not identify which workflow change caused the reduction.

### Categories that should not support a strong claim

- Response assessment, clinical findings, visit context, follow-up/other plans, and other extraction have zero P1 flags throughout these archived reports. This may reflect strong performance, low reviewer sensitivity, or the P1 threshold. It does not establish perfect clinical accuracy.
- Diagnosis/stage/metastasis reaches zero during D6-D9, but the reviewer prompt was later updated to clarify that pTN notation should not be called inaccurate. The category trend combines pipeline improvement with reviewer calibration.
- The early treatment-goal drop occurs between D1 and D2 while the reviewer and pipeline prompts were both being changed. Attribution to a single harness component is not possible.
- Active medication findings include true omissions, scope disagreements, and at least one internally odd flag that called an empty field incorrect when the note reportedly contained no medication. The category count describes what Qwen flagged, not a validated medication error rate.
- F2-F4 are more comparable than the early D checkpoints because no later committed `auto_review.py` change appears after the pre-F2 reviewer update. The exact reviewer prompt was still not stored with each report, so this remains a historical development analysis.

## Reviewer limitations

1. The Qwen reviewer rubric changed during the early trajectory. Git history records updates involving medication scope, jargon severity, acceptable simplification, and pTN staging in commits `778c389d`, `29a4d58d`, `02265128`, and `eb39b09d`.
2. The review files do not record an exact prompt hash. Some reviews were produced before nearby changes were committed, which prevents exact prompt-to-report reconstruction.
3. `auto_review.py` truncates long notes to 12,000 characters, retaining the beginning and end. Supporting evidence in the omitted middle can be missed.
4. The reviewer uses the same Qwen2.5-32B model family as the extraction system. Self-evaluation bias is possible.
5. Later Claude/manual-style checks disagreed with Qwen on severity and false positives. At D1, for example, Qwen recorded no P0 while a later manual-style review called the capecitabine-for-lanreotide error P0. Neither review is a physician assessment.

The trajectory is suitable as secondary development evidence. The separate blinded oncologist evaluation of the final frozen system remains the clinical result.

## Recommended Results wording

> We retrospectively reconstructed the PDAC extraction-development trajectory from archived outputs of the development-time Qwen2.5-32B reviewer. Across nine iterations on a fixed 30-note subset, extraction-related P1 findings flagged by the automated reviewer decreased from 9 to 1, corresponding to 30.0 and 3.3 flags per 100 notes. Flagged diagnosis/staging, active-medication, treatment-goal, and laboratory issues were all lower at the final subset checkpoint. After expansion to the 100-note development set, extraction P1 flags decreased from 11 at full iteration 2 to 6 at the final included checkpoint, full iteration 4. The cohort expansion changed the case mix, and the reviewer rubric changed during early development. These counts describe automated development-review signals rather than clinician-confirmed error rates.

## Recommended figure caption

> **Figure X. PDAC extraction development trajectory based on archived automated reviews.** D1-D9 represent nine checkpoints evaluated on the same 30-note development subset. F2-F4 represent three checkpoints evaluated on the same 100-note development set, with F4 treated as the final full-set checkpoint. Values are Qwen2.5-32B reviewer P1 flags normalized per 100 notes. The dashed segment marks expansion from 30 to 100 notes and should not be interpreted as a within-cohort change. Panel B assigns every extraction P1 flag to a predefined clinical field category; all categories, including zero-count categories, are displayed. The reviewer rubric changed during early development, and its exact prompt hash was not stored. Counts are development-time automated-review signals, not clinician-confirmed error rates.

## Generated files

- `build_pdac_trajectory.py`: verifies the archived reports, extracts extraction P1 findings, assigns predefined field categories, and regenerates the CSV and SVG outputs without calling an LLM.
- `pdac_trajectory.csv`: one row per D1-D9 and F2-F4 checkpoint.
- `pdac_flag_categories.csv`: complete 12-by-10 category grid, including zero-count categories.
- `pdac_flag_details.csv`: the 80 extraction P1 findings used in the analysis.
- `pdac_trajectory.svg`: extraction-only trajectory and complete category heatmap.
