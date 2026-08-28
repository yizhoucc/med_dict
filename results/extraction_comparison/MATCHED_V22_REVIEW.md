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

- Completed: 12/40 (breast 12/20, PDAC 0/20)
- PL findings: P0=3, P1=64, P2=58
- Attribution findings: A0=33, A1=49, A2=92
- Core verdict totals (PL / BL / TIE): 19 / 12 / 53
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

### Breast sample 4 — coral_idx 23

- Case: two distinct right-breast grade-3 IDC lesions: one ER 1%/PR−/HER2− with Ki-67 70%, the other ER−/PR 1%/HER2− with Ki-67 30–40%. PET/CT shows local breast growth but no regional or distant metastasis. Weekly paclitaxel, port, and teaching are recommended, but the patient remains undecided. BRCA1 carrier status is documented.
- PL P1: summary merges the two receptor/Ki-67 profiles; response treats pretreatment local growth as treatment response; medication and therapy plans omit the patient's undecided status and mix visits/teaching into treatment fields; historical surgical consultations are mislabeled as a new Specialty referral; Referral follow-up copies historical imaging advice instead of the current cycle schedule; genetic results mixes correct BRCA1 carrier status with ovarian-cancer history and an inaccurate TNBC label.
- PL P2: Type now separates two profiles but does not bind them to distinct lesion locations; findings omit the decisive biopsy-level differences; curative intent is an unmarked inference; future visit mode and certainty are over-specified.
- Attribution: A0 second opinion, both metastasis fields, response, and Referral follow-up; A1 Type, Stage, goals, goals description, Specialty, and genetic results; A2 summary, findings, medication/therapy plans, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response BL; Type PL; genetic results BL. Total PL 1 / BL 2 / TIE 4.
- v2.1 regression: Type improves from a merged profile to two profiles and Metastasis no longer treats local breast growth as spread; however, summary still merges profiles, referral follow-up regresses, and genetic results is newly polluted.
- Main verification: read the complete note and confirmed the separate `ER 1% PR negative ... Ki67 70%` and `ER negative PR 1% ... Ki67 30-40%` lesions, no metastatic disease, `Patient unsure about starting chemo`, historical dated surgery consults, current cycle-visit schedule, and that only BRCA1 carrier status belongs in genetic results.

### Breast sample 5 — coral_idx 24

- Case: bilateral resected IDC. Left: Stage III T3N1 grade-3 ER 99%/PR >95%/HER2− with regional micrometastasis and high-risk MammaPrint. Right: Stage I T1cN0 grade-1 ER 99%/PR 90%/HER2− with low-risk MammaPrint. Oncotype is pending; TC×4, radiation, then AI are planned.
- PL P1: Type incorrectly turns the right `MammaPrint low risk` result into a `low risk DCIS component`; genetic results misses both completed MammaPrint results; Referral follow-up contains DEXA/exercise/calcium advice rather than a return appointment.
- PL P2: findings mixes MammaPrint into the objective pathology/imaging field; goals category includes `adjuvant` and its description omits the explicit recurrence-risk rationale; medication and therapy fields incompletely represent the CDK4/6-trial option and treatment sequence.
- Attribution: A0 second opinion, Distant, Referral follow-up, and genetic results; A1 Patient type and response; A2 summary, Type, Metastasis, findings, goals, and medication plan.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response TIE; Type BL; genetic results TIE. Total PL 1 / BL 1 / TIE 5.
- v2.1 regression: laterality/profile separation is substantially better and Metastasis is now internally consistent, but the new `low risk DCIS` error leaves Type behind BL; completed MammaPrint omission remains.
- Main verification: read the complete note and confirmed left/right pathology, `focal high grade DCIS` only on the left, separate MammaPrint scores `−0.614` and `+0.321`, pending Oncotype, and the explicit TC→radiation→AI sequence.

### Breast sample 6 — coral_idx 25

- Case: untreated right grade-2 ER−/PR−, HER2-amplified IDC with FNA-confirmed regional axillary disease. Left iliac and bilateral sacral lesions are suspicious and require biopsy for a definitive Stage IV diagnosis. The longstanding carotid-body paraganglioma is separate. THP is conditional on Stage IV confirmation.
- PL P0: Stage is upgraded to definite Stage IV; Distant is upgraded to definite `Yes`, although the bone lesions remain suspicious and biopsy is required.
- PL P1: general Metastasis collapses confirmed regional nodes and suspected bone sites into a site-free `Yes`; findings omits the carotid-body lesion that resolves a competing metastatic concern; colonoscopy-prep ondansetron is mislabeled as oncology supportive medication; palliative intent is too certain and goals description misses explicit long-term-control wording; medication plan drops `If stage IV`; procedure plan omits bone biopsy and chemotherapy teaching; Referral follow-up is a plan dump rather than a follow-up appointment.
- PL P2: summary omits confirmed regional and unresolved distant status; Type omits laterality; MRI-neck purpose is vague.
- Attribution: A0 second opinion and Referral follow-up; A1 goals description, response, lab plan, next visit, and genetic results; A2 Patient type, Type, Stage, both metastasis fields, findings, goals, and medication plan.
- Core verdicts: current_meds TIE; Stage BL; Distant BL; Metastasis PL; response TIE; Type PL; genetic results TIE. Total PL 2 / BL 2 / TIE 3.
- v2.1 regression: Stage/Distant P0s remain; general Metastasis regresses from a useful regional-plus-suspected-site description to bare `Yes`, while therapy-plan completeness improves.
- Main verification: read the complete note and confirmed `suspicious for bone metastasis`, `biopsy for definitive stage IV diagnosis`, conditional `If stage IV` THP, FNA-confirmed right axillary disease, likely carotid-body paraganglioma, colonoscopy-only ondansetron, and explicit `excellent response and possible long term disease control`.

### Breast sample 7 — coral_idx 26

- Case: recurrent grade-3 TNBC with biopsy-confirmed liver metastasis, historical N1a disease, current right regional nodes, and a hypermetabolic lytic S1 lesion. Pembrolizumab+nab-paclitaxel began 03/11/19; this visit is C1D8, with improved axillary pain as a possible early response. Germline MSH2 and a negative breast-cancer panel are documented.
- PL P1: summary and response misattribute a 01/27 pretreatment PET progression to the current regimen; general Metastasis omits historical/current regional nodes and S1; lab summary misses `Labs are in range for continuation`; recent changes omits the newly started regimen and new supportive prescriptions; supportive meds includes Xarelto while omitting several cancer-care drugs; Imaging plan misses explicit symptom-guided or 3–4-month surveillance.
- PL P2: Type lacks the original grade and timepoint structure; Stage omits original IIB; Distant omits suspected S1; findings does not clearly date the pretreatment PET and misses internal-mammary node/biopsy confirmation; palliative intent is an unmarked inference; medication plan repeats Ativan; next-visit mode is inferred; genetic results should label germline, breast-panel, and colon-tumor results separately.
- Attribution: A0 second opinion and response; A1 labs, findings, goals description, and genetic results; A2 summary, Type, both metastasis fields, goal, medication plan, and next visit.
- Core verdicts: current_meds TIE; Stage PL; Distant PL; Metastasis PL; response BL; Type PL; genetic results PL. Total PL 5 / BL 1 / TIE 1.
- v2.1 regression: Type, findings, genetic results, and removal of incorrect distant-node wording improve; however, response timing remains wrong and general Metastasis is now accurate but materially incomplete.
- Main verification: read the complete note and confirmed the 03/11 regimen start, current C1D8, 01/27 pretreatment PET, current `axillary pain improved`, historical ypN1a, right subpectoral/internal-mammary nodes, S1 lesion, `Labs are in range`, supportive-medication plan, and 3–4-month imaging plan.

### Breast sample 8 — coral_idx 27

- Case: resected multifocal left IDC, largest focus 3.9 cm, mainly grade 2 with a small grade-1 focus, ER/PR strongly positive and HER2−, extensive intermediate-grade DCIS, Stage IIA pT2(m)N1a with 2/12 regional nodes positive. MammaPrint is completed/high risk; AC→paclitaxel, later AI, TTE, port, teaching, and genetics referral are planned.
- PL P1: medication and therapy plans omit the explicit future aromatase inhibitor; therapy misroutes TTE and chemotherapy teaching; procedure omits teaching; genetic results correctly restores MammaPrint but contaminates it with a HER2-FISH centromere technical value.
- PL P2: Type omits multifocality, size, the small grade-1 focus, and Ki-67; findings is overlong and mixes MammaPrint into clinical findings; docusate has a weak oncology-support link; goals description is blank; estimated chemotherapy start is presented as a confirmed visit; one prior appointment is ambiguously placed in follow-up.
- Attribution: A0 second opinion and genetic results; A1 supportive medication; A2 visit mode, summary, Type, both metastasis fields, findings, goals, response, medication/therapy plans, next visit, and follow-up.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type TIE; genetic results PL. Total PL 1 / BL 0 / TIE 6.
- v2.1 regression: Metastasis is now internally consistent, MammaPrint is recovered, and findings restores the defining pathology; AI/teaching routing problems remain, and genetic results still needs stricter HER2-FISH removal.
- Main verification: read the complete note and confirmed multifocal mixed-grade pathology, 2/12 positive nodes, completed high-risk MammaPrint, planned chemotherapy plus AI, and that baseline TTE/port/teaching are preparation rather than systemic therapy.

### Breast sample 9 — coral_idx 28

- Case: original left Stage III T3N2 grade-2 ER+/PR+/HER2− micropapillary IDC treated with surgery, incomplete ddAC, and later endocrine therapy; now biopsy-confirmed locally advanced unresectable recurrence. Current regional nodes are suspicious and a level Vb cervical node may represent distant disease pending FNA. Goserelin then AI are planned but not started.
- PL P1: Stage only reports a converted historical IIIA and omits the current unresectable/possibly metastatic state; general Metastasis omits current suspicious regional nodes; findings omits the recurrence biopsy/biomarkers and mixes symptoms; planned goserelin/AI is mislabeled as a recent completed change; palliative intent loses the clinician's `if MBC` condition; response calls the patient treatment-naive despite prior chemotherapy/endocrine therapy followed by recurrence; the FNA procedure is duplicated as a Specialty referral.
- PL P2: summary omits unresectable and suspected-metastatic status; Type mixes original/current profiles and misses current Ki-67; Distant uses a less precise cervical-node label; next-visit mode is inferred.
- Attribution: A0 second opinion, recent changes, and Specialty; A1 Patient type and genetic results; A2 summary, Type, Stage, Metastasis, findings, goals, response, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis PL; response TIE; Type BL; genetic results TIE. Total PL 1 / BL 2 / TIE 4.
- v2.1 regression: general Metastasis better separates historical confirmed regional disease from suspected distant disease, but Stage now loses the current state; response, planned-change, and conditional-goal errors remain, and the Type lesion/timepoint rule does not fully take effect.
- Main verification: read the complete note and confirmed original Stage III/T3N2, prior ddAC and endocrine therapy, current grade-2 ER/PR >95% recurrence, locally advanced unresectable status, suspicious left axillary/supraclavicular/internal-mammary and level-Vb nodes, pending FNA, conditional MBC goal language, and future goserelin→AI.

### Breast sample 10 — coral_idx 29

- Case: newly diagnosed right grade-2 IDC, ER >95%, PR 25%, clinically HER2 non-amplified, clinical Stage II cT2N1 with FNA-confirmed regional axillary disease and PET-negative distant staging. Egg harvesting is active; no anticancer treatment has started. Paclitaxel→AC is conditional on egg-harvesting timing; genetic testing was sent and MammaPrint is pending.
- PL P1: therapy turns `Hopefully ... assuming egg harvesting` into a definite 05/01 start and mixes trial procedures into treatment; Imaging plan omits echocardiogram; genetic plan omits the already-sent pending panel; echocardiogram/EKG is mislabeled as a clinic visit; sent testing is mislabeled as a Genetics referral.
- PL P2: Type adds a needless and incorrect `ER+ inferred from tamoxifen` despite direct ER pathology; findings does not clearly state positive axillary FNA/PET-negative distant staging; curative intent is unmarked inference; medication plan omits later hormone therapy and has unclear ovarian-suppression timing; cold-cap coordination and follow-up wording are incomplete.
- Attribution: A0 second opinion, in-person, next visit, and follow-up; A1 labs, findings, goals description, response, and Genetics referral; A2 Patient type, summary, Type, Metastasis, goals, medication/therapy plans, and genetic-results fallback.
- Core verdicts: current_meds PL; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type TIE; genetic results TIE. Total PL 1 / BL 0 / TIE 6.
- v2.1 regression: core results are stable and current-meds correctly excludes fertility drugs; conditional-treatment, echo, genetic-plan, visit, and referral errors remain.
- Main verification: read the complete note and confirmed direct ER >95%/PR 25%, clinician-final HER2-negative interpretation, positive axillary FNA, clear PET, fertility drugs only, conditional chemotherapy timing, scheduled echo/EKG, sent genetic test, pending MammaPrint, and cold-cap coordinator plan.

### Breast sample 11 — coral_idx 30

- Case: pure left DCIS after lumpectomy. Final excision pathology is intermediate grade, 1.8 cm, solid/cribriform, no necrosis or invasive cancer, positive/very close posterior margin, pTisNx, ER/PR positive, HER2 untested. Tamoxifen was prescribed but explicitly held until radiation-oncology assessment.
- PL P1: Type prioritizes the earlier core biopsy's PR-pending/micropapillary/focal-necrosis findings over final pathology; recent changes treats held future tamoxifen as already changed treatment; `curative` misses the explicit risk-reduction/prevention goal; therapy overstates radiation as continuing and does not cleanly represent held tamoxifen.
- PL P2: receptor phrasing is awkward; findings does not clearly say there is no invasive cancer and overweights early core morphology; a pre-existing radiation appointment is ambiguously treated as a new referral; therapy mixes in follow-up.
- Attribution: A0 second opinion and therapy; A1 Patient type, Type, Stage, both metastasis fields, findings, and response; A2 in-person, summary, goal category, and goal description.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 0 / TIE 7.
- v2.1 regression: Stage is repaired to pTisNx and goals description is cleaner; final-pathology priority and held-tamoxifen timing remain unresolved.
- Main verification: read the complete note and confirmed final `Invasive tumor type: None`, solid/cribriform architecture, no necrosis, ER/PR positive, pTisNx, risk-reduction language, pending radiation evaluation, and explicit instruction to hold tamoxifen.

### Breast sample 12 — coral_idx 31

- Case: untreated right multifocal clinical Stage II, node-negative grade-2 invasive mammary carcinoma with mixed ductal/lobular features, ER >95%, PR ~70%, HER2−, Ki-67 ~20%, PET-negative distant staging, and completed high-risk MammaPrint. Curative neoadjuvant AC→paclitaxel, later surgery/endocrine treatment and possible radiation are planned.
- PL P1: lab summary misses a dated numeric panel within six months; procedure omits chemotherapy teaching; Referral.Others omits explicit cold-cap CRC contact; genetic results misses completed high-risk MammaPrint.
- PL P2: summary omits stage/laterality/subtype; Type omits laterality and Ki-67; findings misses a third MRI lesion/NME and absent chest-wall involvement; medication plan omits later endocrine therapy and preferred start timing; next-visit mode is inferred.
- Attribution: A0 second opinion; A1 Patient type, in-person, Type, labs, findings, and genetic results; A2 summary, response, medication/therapy plans, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type BL; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- v2.1 regression: Type gains exact ER/PR values and therapy routing improves, but completed MammaPrint, recent labs, teaching, and cold-cap coordination remain missing.
- Main verification: read the complete note and confirmed the 2016-10-22 panel, mixed ductal/lobular pathology with Ki-67, third MRI lesion/NME, PET-negative/bone-island staging, completed `MP high risk`, chemotherapy teaching, cold-cap CRC contact, and chemo→surgery→endocrine sequence.
