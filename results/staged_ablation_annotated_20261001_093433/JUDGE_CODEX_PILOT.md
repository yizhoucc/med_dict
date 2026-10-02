# Codex blinded judge calibration pilot

## Scope and result

Manually reviewed source lines 1, 2, 21, 22, 41, and 42 from each public blinded judge file (12 pairs total). Each selected clinical note and both outputs were read in full. No private mapping was accessed.

Overall verdicts: **A 7, B 5, TIE 0**.

| Cancer | A | B | TIE |
|---|---:|---:|---:|
| Breast | 3 | 3 | 0 |
| PDAC | 4 | 2 | 0 |
| Total | 7 | 5 | 0 |

## Counts by source-line block

| Source lines | Breast | PDAC | Combined |
|---|---|---|---|
| 1–2 | A 2, B 0, TIE 0 | A 2, B 0, TIE 0 | A 4, B 0, TIE 0 |
| 21–22 | A 0, B 2, TIE 0 | A 1, B 1, TIE 0 | A 1, B 3, TIE 0 |
| 41–42 | A 1, B 1, TIE 0 | A 1, B 1, TIE 0 | A 2, B 2, TIE 0 |

## Recurring judgment issues

- **Disease extent was often decisive.** The main errors were treating uncertain lung nodules as confirmed disease, adding unsupported regional nodes, or converting a locoregional breast recurrence into Stage IV disease.
- **Uncertainty was frequently collapsed.** Several outputs changed “staging pending” or “indeterminate but suspicious” into a definite yes/no distant-metastasis answer.
- **Response fields often contained the wrong concept.** Common substitutions were treatment status, physical examination, laboratory eligibility, or future plans. One output inferred “stable disease” from a normal exam without response imaging.
- **Temporal leakage was common.** Completed PET/CT, brain MRI, and laboratory studies were repeatedly listed as future plans, especially in the long breast consultation letter.
- **Field placement errors recurred.** PET/CT appeared under Procedure or Lab Plan; germline results appeared under Referral; surveillance was used as treatment intent; and unrelated medicines were labeled supportive oncology drugs.
- **Hold versus stop mattered.** In the PDAC chemotherapy-break note, “hold/break with surveillance” was more faithful than a definite permanent stop.

## Rubric ambiguities encountered

- **Conflicting HER2 evidence:** The first breast note contains FISH wording suggestive of positivity, an IHC-negative addendum, and a final oncology assessment repeatedly calling the tumor triple-negative. I prioritized the final integrated clinical assessment but recorded the source conflict and lowered confidence where it directly affected the comparison.
- **Treatment intent versus management strategy:** “Surveillance,” “hormonal therapy,” and “chemotherapy” describe strategies, not necessarily curative/palliative intent. I treated this as a semantic mismatch unless the note supported the intent independently.
- **Locoregional unresectable recurrence:** The breast recurrence note discusses both unresectability and a possible sequence of shrinkage, resection, radiation, and long-term control. The exact curative-versus-palliative label is not fully explicit, so this issue alone did not determine a verdict.
- **Inferred stage:** I accepted Stage IV as a reasonable inference when liver metastases were explicitly documented, but not when an output reclassified a parasternal locoregional recurrence as distant disease.
- **Negative future-plan fields:** When surveillance was planned but no scan or laboratory test was explicitly ordered, I did not infer an imaging or lab plan.

No ties were assigned because every selected pair had at least one clinically meaningful difference after prioritizing faithfulness and then completeness.
