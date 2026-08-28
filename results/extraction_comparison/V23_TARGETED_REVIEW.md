# Matched v2.3 targeted regression review

## Status

- Scope: 8 samples = 6 previously problematic samples + 2 clean controls.
- Breast row indices: 1, 5, 6 (samples 2, 6, 7).
- PDAC row indices: 3, 7, 8, 14, 19 (samples 4, 8, 9, 15, 20).
- Review method: read each complete `note_text`, every PL keypoint, attribution, the matched BL output, and the field definitions; no scripted clinical judgment.
- Completed: 8/8.
- Final artifacts: `v23_targeted/breast_results_v232.txt` and `v23_targeted/pdac_results_v231.txt`.
- Field-level findings: P0=0, P1=32, P2=33.
- Core PL:BL:TIE: 29:0:22 across 51 applicable comparisons.
- All P0/P1 findings from the parallel initial review were rechecked against the complete source note by the main Codex session.

## Repair targets

1. Breast sample 2: do not borrow historical PR-/HER2- into the current HR+ recurrence.
2. Breast sample 6: suspicious bone disease pending biopsy must not become confirmed Stage IV/M1.
3. PDAC sample 4: current stable/good-control evidence must override unsupported progression language.
4. PDAC sample 8: current stage must remain consistent with genuinely confirmed distant disease.
5. PDAC samples 15 and 20: recover explicit completed MMR-intact/pMMR results.
6. Breast sample 7 and PDAC sample 9: clean controls for regression.

## Per-sample review

### Breast sample 2 — coral_idx 21

- Final repair: historical primary is now `ER+/PR-` with no invented HER2 result; the current recurrence is `HR+ (PR/HER2 not specified)`. General Metastasis now preserves the explicit locoregional recurrence while Distant remains `No`.
- Remaining P1: unsupported `palliative` intent; completed PET/CT and brain MRI remain in Imaging Plan; completed laboratory testing remains in Lab Plan.
- Remaining P2: incomplete labs, findings mix in labs and omit the positive recurrence FNA, response wording is indirect, possible future surgery is omitted, and one empty referral fallback is inconsistent.
- Core: PL 5 / BL 0 / TIE 2.

### Breast sample 6 — coral_idx 25

- Final repair: Stage is `Suspected Stage IV (pending confirmation)`; Distant is suspected bone disease only; the carotid-body/paraganglioma alternative diagnosis is excluded; the positive right axillary LN FNA is represented as pathologically confirmed regional disease.
- Remaining P1: colonoscopy-prep ondansetron is misclassified as oncology supportive care; palliative intent and the medication plan are too definite before Stage IV confirmation; the staging biopsy is omitted from Procedure Plan; Referral follow-up is a plan dump.
- Remaining P2: findings omit the carotid-body differential and Lab Plan is nonspecific.
- Core: PL 4 / BL 0 / TIE 3.

### Breast sample 7 — coral_idx 26

- Final repair: response now uses the current C1D8 observation—improved axillary pain as a possible early response—instead of treating pre-regimen liver growth as current progression.
- Remaining P1: summary/findings still foreground pretreatment progression; Distant omits the suspicious S1 lesion; general Metastasis omits regional nodes and S1; supportive medication coverage is incomplete and includes Xarelto; the explicit 3–4 month imaging schedule is omitted.
- Remaining P2: qualitative labs are reported as absent and response retains a trailing treatment-plan clause.
- Core: PL 6 / BL 0 / TIE 1.

### PDAC sample 4 — coral_idx 3

- Final repair: response now follows the current A/P: `continued good disease control on surveillance`, consistent with stable CT, no suspicious liver lesion, and CA 19-9 non-secretor status.
- Remaining P1: transition from completed Gem/Abraxane to surveillance is omitted from recent changes; goals description is empty.
- Remaining P2: Type/Stage could express the mucinous and longitudinal uncertainty more clearly, supportive medication coverage is incomplete, surveillance is not stated in Therapy Plan, and response retains a harmless trailing `2.` list marker.
- Core: PL 2 / BL 0 / TIE 4.

### PDAC sample 8 — coral_idx 7

- Final repair: Stage now reads `pT2N2, now metastatic (Stage IV)` and general Metastasis combines current confirmed nonregional intra-abdominal nodal recurrence with historical regional N2 disease. Germline ATM and MMR-intact results remain present.
- Remaining P1: Type omits recurrent/metastatic status; recent Gem/nab-paclitaxel start and three completed cycles are omitted from recent changes; incoming consultation and treatment advice remain misrouted into referral fields.
- Remaining P2: one pathology date is malformed, Therapy Plan is overlong, future visit mode is inferred, and genetic results omit the communicated no-actionable-mutation statement.
- Core: PL 5 / BL 0 / TIE 1.

### PDAC sample 9 — coral_idx 8 (control)

- No core regression. Stage IV, lung metastasis, current Gem/Abraxane, response, and completed BRCA/KRAS/CDKN2A/APC results remain correct.
- Remaining P1: findings reverse the CA 19-9 trend; supportive medication coverage is incomplete with a misspelled olanzapine; Medication Plan omits continued Gem/Abraxane and the Abraxane reduction; a historical Phase-I consultation is mislabeled as Genetics referral.
- Remaining P2: summary, recent-change routing, goals detail, Therapy Plan wording, and future visit mode.
- Core: PL 2 / BL 0 / TIE 4.

### PDAC sample 15 — coral_idx 14

- Final repair: completed MMR-intact evidence is present alongside the completed benign-ascites KRAS-negative result; deterministic attribution now points to all four MMR protein lines.
- Remaining P1: anticoagulant hold is misrouted as cancer treatment change; Creon is misrouted as anticancer therapy; next-visit and referral fields contain unsupported or historical routing.
- Remaining P2: findings mix timepoints, supportive medication coverage is incomplete, goals are underspecified, and response should separate poor historical treatment effect from current suspected recurrence.
- Core: PL 2 / BL 0 / TIE 4.

### PDAC sample 20 — coral_idx 19

- Final repair: completed `MMR proteins intact by IHC (pMMR)` is restored without importing pending UCSF500 or future germline testing; attribution now cites the MMR sentence.
- Remaining P1: Type omits metastatic status and pancreatic-tail origin; conditional trial biopsies are promoted to a current Procedure Plan; nontherapeutic-trial eligibility is misrouted as follow-up.
- Remaining P2: findings understate confirmed peritoneal/omental disease, supportive medications omit mainly used Tylenol, treatment options need clearer conditional wording, and the genetics referral is a recommendation rather than a confirmed order.
- Core: PL 3 / BL 0 / TIE 3.

## Final conclusions

- The four v2.2 P0 failures are eliminated in the final targeted artifacts.
- The two initially failed live fixes were corrected and rerun: PDAC sample 4 response and PDAC sample 8 current metastatic stage.
- Both controls retained their core advantages; Breast sample 7 additionally improved after a genuine pre-treatment/current-treatment temporal error was found.
- The remaining problems are mostly plan/referral routing, attribution quality, and completeness. They are useful limitations but do not block the core workshop story.
- This targeted result is not a substitute for a new full-40 run. The primary fully rerun/audited table remains v2.2; v2.3.x establishes that the identified high-impact failure modes are repairable without a detected core regression in the required affected-plus-control subset.
