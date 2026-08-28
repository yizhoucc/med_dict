# Matched v2.2 PL vs BL Manual Review

Date: 2026-08-27

## Scope and rules

- Compare the repaired pipeline outputs (`pipeline_*_matched_v22.txt`) with the unchanged semantic-contract single-prompt baseline (`baseline_extract_*_matched_v21.txt`).
- For every sample, read the complete `note_text`, every PL keypoint, PL attribution, and every BL keypoint. Check the relevant extraction and plan prompt definitions.
- Clinical priority: faithful precision > coverage > simple vocabulary > patient-friendly wording.
- Severity: P0 = fabricated/unsupported high-impact clinical fact; P1 = wrong field/direction, major omission, temporal error, or materially misleading wording; P2 = minor incompleteness, imprecision, formatting, or attribution weakness; OK = no material issue.
- Core comparison fields: `current_meds`, `Stage_of_Cancer`, `Distant Metastasis`, `Metastasis`, `response_assessment`, breast `Type_of_Cancer`, and `genetic_testing_results`.
- Empty BL values and explicit negative/unknown PL fallbacks are equivalent when the source truly provides no supported answer.
- v2.1 findings are used only as regression history. Every v2.2 verdict is checked against the complete source note again.

## Status

- Completed: 3/40 (breast 3/20, PDAC 0/20)
- PL findings: P0=1, P1=16, P2=15
- Attribution findings: A0=8, A1=11, A2=29
- Core verdict totals (PL / BL / TIE): 7 / 3 / 11
- Current phase: breast review while the PDAC v2.2 run is in progress

## Results

### Breast sample 1 — coral_idx 20

- Case: untreated right grade-3 IDC after mastectomy; final signed addendum and A/P support TNBC; Stage II / pT2N1a with 1/2 positive regional sentinel nodes; PET/CT pending; no completed molecular result.
- PL P1: `Clinical_Findings.findings` contradicts the final addendum by calling HER2 positive; `Medication_Plan` incorrectly says AC/T, TC, and CMF are not planned, although CMF remains a conditional lower-risk option and the final decision is deferred until PET/CT; `Therapy_plan=None` omits that deferred decision.
- PL P2: Type includes the conflicting FISH ratio without clearly explaining that the final addendum/A&P resolves the cancer as HER2-negative/TNBC; Stage over-converts `Stage II; pT2N1a` to Stage IIB; curative intent is a reasonable but unmarked inference; follow-up mode is not explicit.
- Attribution: A0 second-opinion citation; A1 Type, labs, goals description, medication plan, and therapy plan; A2 summary, Stage, Distant, Metastasis, findings, goals, response, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis PL; response TIE; Type PL; genetic results PL. Total PL 3 / BL 1 / TIE 3.
- v2.1 regression: Type now states TNBC and findings dropped stale labs, but findings newly preserve the superseded `HER2 positive` line. Core totals are unchanged.
- Main verification: read the complete note and confirmed the final signed addendum says `This carcinoma is negative for HER2 oncoprotein over-expression`, A/P says `Stage II (T2N1) triple negative breast cancer`, and treatment remains deferred: `requires additional testing`, `might want to consider a lower risk chemotherapy`, and `resume our conversation ... at that time`.

### Breast sample 2 — coral_idx 21

- Case: historical right grade-1 ER+/PR− IDC with no documented HER2 result; now untreated unresectable locoregional/chest-wall recurrence. Current liver finding is a cyst, brain/bone studies are negative, and axillary nodes are unproven with a prior benign biopsy. Aromatase inhibitor therapy is planned; zoledronic acid is active osteoporosis support.
- PL P0: Type fabricates HER2− for the original and recurrent disease and copies historical PR− onto the current recurrence, for which the source only says `strongly hormone-receptor positive`.
- PL P1: general Metastasis over-corrects to `No` and omits the explicit locoregional/chest-wall recurrence; palliative intent is unsupported; Imaging and Lab plans mislabel already completed PET/CT, brain MRI, and laboratory testing as future.
- PL P2: labs omit other completed normal studies; findings include laboratory results; goals description omits explicit long-term-control wording; response omits documented interval growth; therapy omits supportive bone treatment and the conditional radiation component; one referral fallback is blank.
- Attribution: A0 Metastasis, treatment goal, and lab plan; A1 findings and goals description; A2 visit fields, Type, Stage, response, and imaging plan.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis BL; response PL; Type BL; genetic results TIE. Total PL 3 / BL 2 / TIE 2.
- v2.1 regression: Distant is correctly repaired from obsolete liver/chest-wall uncertainty to `No`; fabricated confirmed axillary disease is removed, but the locoregional recurrence is now omitted. Type is better scoped by timepoint but remains an unsupported receptor fabrication.
- Main verification: read the complete note and confirmed historical ER+/PR− only, no HER2 result, current `strongly hormone-receptor positive` recurrence, benign/unproven axillary nodes, liver `consistent with cyst`, `No other sites of disease`, possible shrinkage→resection/radiation, and explicit `long-term disease control`.

### Breast sample 3 — coral_idx 22

- Case: untreated right multifocal, locally advanced spindle-cell metaplastic TNBC; right axillary FNA negative. PET shows no confirmed distant disease but a 2.3 cm indeterminate right adrenal nodule requiring dedicated follow-up. Invitae panel was sent and MammaPrint is pending.
- PL P1: summary says `early stage` despite final A/P `locally advanced, multifocal`; Distant and general Metastasis both say `No` and omit the indeterminate adrenal nodule; findings omit the negative axillary FNA; therapy mixes port/teaching into systemic treatment; procedure omits chemotherapy teaching; MammaPrint is routed to Imaging; genetic plan changes an already-sent panel into a future test and incompletely represents pending MammaPrint; historical Genetics activity is mislabeled as a new referral.
- PL P2: Type omits laterality and duplicates ER−; lab result is qualitative despite the prompt's numeric rule; trial enrollment wording is slightly too definite; next-visit mode is missing; follow-up attribution is incomplete.
- Attribution: A0 second opinion, summary, imaging, and Genetics referral; A1 Patient type, findings, and response; A2 Type, Stage, both metastasis fields, labs, medication/therapy/lab/genetic plans, next visit, follow-up, and genetic results.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type PL; genetic results TIE. Total PL 1 / BL 0 / TIE 6.
- v2.1 regression: genetic results is correctly repaired from pending tests/pathology to no completed result, but Distant and Metastasis regress from preserving the adrenal uncertainty to `No` because the new generation omitted the site before the conservative hook ran.
- Main verification: read the complete note and confirmed `locally advanced, multifocal`, `Indeterminate right adrenal nodule measuring up to 2.3 cm`, `FNA negative`, `126 InVitae panel sent`, `mammaprint is pending`, Medi-port, and `arrange for chemo teaching session`.
