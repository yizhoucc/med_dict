# Codex full blinded judging summary

## Scope and validation

- Judged all 120 blinded pairs: 60 breast and 60 PDAC.
- Pilot file: 12 judgments, re-reviewed with the final rubric.
- Remaining file: 108 judgments.
- Both JSONL files passed `jq -e .`; line counts are 12 + 108 = 120.
- Judging used only the public blinded records and did not inspect decoded variant identities or private mappings.

## Overall verdicts

| Verdict | Count | Percent |
|---|---:|---:|
| A | 53 | 44.2% |
| B | 42 | 35.0% |
| TIE | 25 | 20.8% |
| **Total** | **120** | **100%** |

## Verdicts by cancer

| Cancer | A | B | TIE | Total |
|---|---:|---:|---:|---:|
| Breast | 26 | 20 | 14 | 60 |
| PDAC | 27 | 22 | 11 | 60 |

## Verdicts by source-line block

| Cancer | Source lines | A | B | TIE | Total |
|---|---|---:|---:|---:|---:|
| Breast | 1–20 | 10 | 7 | 3 | 20 |
| Breast | 21–40 | 6 | 6 | 8 | 20 |
| Breast | 41–60 | 10 | 7 | 3 | 20 |
| PDAC | 1–20 | 6 | 11 | 3 | 20 |
| PDAC | 21–40 | 8 | 6 | 6 | 20 |
| PDAC | 41–60 | 13 | 5 | 2 | 20 |

## Confidence

| Confidence | Count | Percent |
|---|---:|---:|
| High | 67 | 55.8% |
| Medium | 53 | 44.2% |
| Low | 0 | 0% |

By cancer, breast had 34 high- and 26 medium-confidence judgments; PDAC had 33 high- and 27 medium-confidence judgments.

## Recurring judgment issues

- **Premature certainty about stage or metastasis:** suspicious liver, lung, bone, adrenal, or nodal findings were sometimes converted into confirmed metastatic disease or Stage IV disease despite planned follow-up or biopsy.
- **Anatomic classification errors:** regional nodes or direct local extension were sometimes labeled distant metastases; conversely, documented metastatic sites were occasionally omitted.
- **Medication temporal errors:** outputs frequently omitted active drugs, included medicines explicitly marked “not taking” or on hold, or mixed planned therapy with current medication.
- **Response-assessment errors:** common problems included assessing response before treatment, calling progression during a treatment holiday failure of active therapy, relying on old scans instead of the current clinical state, or describing future plans as response.
- **Plan-versus-history confusion:** completed procedures, prior referrals, past consultations, and historical tests were often presented as future plans. Conditional trial or radiation options were also sometimes phrased as definite plans.
- **Field mismatch:** imaging appeared under procedures, scans appeared under laboratory plans, pathology appeared under genetic results, completed variants appeared under genetics referrals, and surveillance/adjuvant labels were used as treatment intent.
- **Treatment-intent overreach:** palliative or curative intent was sometimes assigned without explicit support, especially when recurrence was suspected but unconfirmed.
- **Completeness versus fidelity tradeoff:** one output might be more complete but include unsupported details, while the other was safer but omitted clinically important medication, metastasis, response, or follow-up information.
- **Formatting and corruption:** malformed or truncated laboratory fields, transcription artifacts, stray redaction tokens, date corruption, and non-schema subfields were recurrent but usually secondary unless they changed clinical meaning.

## Rubric limitations and ambiguities

- Several notes contain internally conflicting evidence, such as a clinician labeling disease metastatic while current imaging describes lesions as suspicious, non-avid, or benign-appearing. Judgments prioritized the most clinically authoritative statement while preserving unresolved uncertainty.
- The schema does not always clarify whether `current_meds` means all outpatient medicines or only active cancer therapy. Omitting the active anticancer regimen was treated as more consequential, but omission of a substantial outpatient list also affected completeness.
- Boundaries among therapy, medication, procedure, imaging, referral, and follow-up fields are not fully specified. Pure field placement without clinical consequence generally supported TIE; temporal or misleading placement could be decisive.
- “Treatment goals” can mean oncologic intent or current management strategy. Labels such as surveillance, adjuvant, neoadjuvant, or follow-up were not treated as substitutes for curative versus palliative intent.
- Conditional future options are difficult to score uniformly. Discussion alone was not treated as a definite plan unless the note committed to an action.
- Overall verdicts necessarily compress many field-level tradeoffs. TIE was used when differences were minor, offsetting, or predominantly formatting/placement rather than clinically meaningful.
- Confidence reflects confidence in the pairwise verdict, not certainty that every individual field has a single indisputable gold interpretation.
