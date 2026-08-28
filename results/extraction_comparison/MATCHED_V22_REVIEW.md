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

- Completed: 40/40 (breast 20/20, PDAC 20/20)
- PL findings: P0=4, P1=227, P2=157
- Attribution findings: A0=134, A1=150, A2=258
- Core verdict totals (PL / BL / TIE): 66 / 28 / 166
- Current phase: review complete; failure-pattern synthesis and targeted repair

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

### Breast sample 13 — coral_idx 32

- Case: untreated right IDC, ER/PR 80–90%, HER2−, Ki-67 20–30%, grade unreported; current right axillary FNA confirms regional disease. Lung, abdomen/pelvis, and bone staging are negative. A small falx/dural lesion strongly favors meningioma but retains an unlikely dural-metastasis differential. Surgery is planned; chemotherapy/MammaPrint decisions remain postoperative and conditional.
- PL P1: Distant overstates `No` and deletes the documented low-probability dural uncertainty; general Metastasis labels the current positive axillary FNA as historical and also omits the dural differential; medication plan omits the postoperative chemotherapy decision and mislabels cold-cap devices as supportive medication; radiotherapy invents a Rad Onc consult; genetic plan upgrades `possibly a MammaPrint` to definite testing.
- PL P2: curative intent is an unmarked inference; procedure omits intended second egg-harvest round.
- Attribution: A0 second opinion and Stage; A1 Patient type, Type, Distant, labs, findings, and goals description; A2 summary, Metastasis, goal, response, medication/radiotherapy plans, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant BL; Metastasis BL; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 2 / TIE 5.
- v2.1 regression: Type and negative-organ findings improve, and the regional node is no longer put in a distant clause; Distant still drops the dural differential and the current FNA is newly mislabeled historical.
- Main verification: read the complete note and confirmed the MRI wording `most likely a meningioma, although dural-based metastasis remains an unlikely possibility`, current positive 10/28 axillary FNA, postoperative chemotherapy decision, only-possible MammaPrint, and absence of an explicit Rad Onc consult.

### Breast sample 14 — coral_idx 33

- Case: untreated right grade-1 IDC, ER >95%, PR ~90%, HER2 IHC 2+ but FISH non-amplified and therefore HER2-negative, Ki-67 ~10%; no regional/distant disease. Myriad is negative and MammaPrint low risk. Goserelin starts today; letrozole is planned in about two weeks; surgery timing is unknown.
- PL P1: summary, Type, and findings all retain `HER2 equivocal` instead of resolving linked negative FISH; lab summary uses RPR/HIV results over six months old; recent changes treats future letrozole as already started; medication plan presents Effexor/gabapentin discussion as a definite plan; genetic plan and Genetics referral revive a historical completed referral; Specialty revives historical surgical consultations instead of the current message to an existing team.
- PL P2: Stage fallback wording is nonstandard; curative intent is unmarked and its description is empty; procedure omits that the surgery date remains unknown.
- Attribution: A0 second opinion, Metastasis, labs, and goals; A1 Patient type, Stage, Distant, findings, goals description, genetic plan, Genetics, and Specialty; A2 in-person, summary, Type, medication plan, and genetic results.
- Core verdicts: current_meds PL; Stage TIE; Distant TIE; Metastasis TIE; response PL; Type TIE; genetic results TIE. Total PL 2 / BL 0 / TIE 5.
- v2.1 regression: core verdicts are unchanged; the strengthened Type prompt still fails to apply the FISH-final HER2 rule in this case, while current meds and response remain PL advantages.
- Main verification: read the complete note and confirmed `IHC 2`, `FISH non-amplified`, old 03/17/19 RPR/HIV in a later COVID-era visit, `Start goserelin today`, `Start letrozole in about two weeks`, historical 2019 genetics/surgical visits, completed Myriad/MammaPrint, and unknown surgery date.

### Breast sample 15 — coral_idx 34

- Case: the only pathologically confirmed disease is breast-origin adenocarcinoma in a right supraclavicular node, ER >90%, PR 50%, HER2 IHC 2+ with FISH pending. Right axillary/right cervical/left level-Vb nodes and the breast primary remain presumptive or unbiopsied. Additional biopsies are required before a de novo MBC diagnosis; treatment depends on HER2 and biopsy results.
- PL P1: summary prematurely declares metastatic breast cancer; Stage says only `Not staged` and omits conditional possible de novo MBC; Distant is empty and omits suspected nonregional cervical disease; general Metastasis omits suspected axillary/cervical nodes and mixed certainty; docusate lacks an oncology-support indication; palliative goal and description drop `if confirmed`; genetic testing is mislabeled as a Genetics referral.
- PL P2: Type should label the supraclavicular specimen and unconfirmed breast primary; findings does not clearly separate evidence levels; procedures duplicate biopsy wording and include HER2-result follow-up; next-visit mode is inferred.
- Attribution: A0 second opinion, Genetics referral, and genetic-results fallback; A1 missing Distant support; A2 Patient type, summary, Type, Stage, Metastasis, findings, goals, and response.
- Core verdicts: current_meds TIE; Stage PL; Distant BL; Metastasis PL; response TIE; Type PL; genetic results TIE. Total PL 3 / BL 1 / TIE 3.
- v2.1 regression: fabricated confirmed axillary disease is removed and treatment plans retain HER2-dependent conditions; Distant regresses to empty, while summary/stage/goals still mishandle conditional MBC.
- Main verification: read the complete note and confirmed biopsy-proven right supraclavicular disease only, suspicious right axillary/right cervical/left level-Vb nodes, absent definite breast primary, `if we confirm ... de novo MBC`, conditional goals/treatment, and no explicit Genetics referral.

### Breast sample 16 — coral_idx 35

- Case: untreated left clinical Stage III grade 2–3 invasive lobular carcinoma, ER 96%, PR 35%, HER2−. PET/CT is negative for distant disease but left axillary/subpectoral nodes are radiographically suspicious; axillary FNA is pending. The patient declines neoadjuvant letrozole±bevacizumab and prefers surgical evaluation first.
- PL P1: general Metastasis says `No` and loses the suspicious regional nodes/pending FNA; findings omits the pending status and a clinically benign supraclavicular/anterior-cervical node; medication and therapy fields are dominated by surgery and omit the offered neoadjuvant endocrine trial/decline decision; mastectomy is made definite rather than a preference for consideration; completed ultrasound is called future imaging; Specialty misses the explicit return referral to surgery.
- PL P2: curative intent is unmarked inference; next-visit mode is inferred.
- Attribution: A0 second opinion, Distant, Metastasis, medication plan, therapy, and imaging; A1 Type, goals description, and response; A2 in-person, findings, goal, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- v2.1 regression: fabricated confirmed regional nodes and nonexistent bone lesions are removed, reducing severity from P0 to P1, but the field now over-cleans real suspected regional disease.
- Main verification: read the complete note and confirmed PET-negative distant staging, abnormal left axillary/subpectoral nodes, FNA pending, benign-feeling cervical node, offered letrozole±bevacizumab, trial refusal, preference for upfront mastectomy, completed ultrasound, and explicit surgical referral.

### Breast sample 17 — coral_idx 36

- Case: left Stage IIB T2N1M0 grade-2 IDC, ER >95%, PR 25%, HER2−, with LVI, positive margins, 2/2 positive regional nodes, 1.8 cm nodal deposit/extracapsular extension, and multifocal residual disease including a small positive deep margin after re-excision. TC×6, ONPRO, port, teaching, staging PET/CT, echo, and later radiation are planned; BRCA is reported negative.
- PL P1: general Metastasis says `No` and omits pathologically confirmed regional nodes; findings emphasizes postoperative MRI while omitting decisive pathology burden; therapy mixes teaching/port and omits radiation; Specialty invents definite radiation/surgical referrals; Referral.Others misses chemotherapy teaching; follow-up is a task list rather than an appointment.
- PL P2: Type omits laterality and exact receptor values; radiation certainty is slightly overstated; procedure omits teaching; genetic plan mixes historical uncertainty into a no-plan answer.
- Attribution: A0 second opinion, Distant, Metastasis, next visit, Specialty, and follow-up; A1 Patient type, goals description, and response; A2 summary, Type, Stage, findings, goal, radiation, and genetic results.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type BL; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- v2.1 regression: fabricated bone disease is removed and genetic results now excludes HER2 FISH, but Metastasis over-cleans the real regional node burden; Type still loses to BL.
- Main verification: read the complete note and confirmed T2N1M0, 2/2 positive nodes with 1.8 cm deposit/ECE, LVI, residual multifocal IDC/positive margin, planned TC×6/ONPRO, explicit teaching referral, and BRCA-negative report.

### Breast sample 18 — coral_idx 37

- Case: left grade-2 IDC, ER >95%, PR low-positive <5%, HER2−, Ki-67 44%, cT2NX. MRI shows suspicious axillary nodes, but two FNAs are benign and final nodal status awaits surgery. ATM germline mutation and completed high-risk MammaPrint are documented. Surgery precedes node-dependent TC versus AC-T and long-term endocrine therapy.
- PL P1: Distant asserts `No` without explicit systemic staging/M0 evidence; medication plan omits the clear node-dependent chemotherapy plan; radiotherapy `None` loses the explicit decision to avoid radiation because of ATM and choose mastectomy; procedure omits planned mastectomy.
- PL P2: Metastasis compresses suspicious imaging plus two benign FNAs to `No`; goals description is blank; follow-up is not a real future arrangement; genetic results keeps ATM but omits high-risk MammaPrint.
- Attribution: A0 Distant; A2 Type, Metastasis, and genetic results.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type PL; genetic results TIE. Total PL 1 / BL 0 / TIE 6.
- v2.1 regression: exact low-positive PR and ATM result improve; MammaPrint omission and Distant overconfidence remain, while medication/radiotherapy/procedure omissions are unchanged.
- Main verification: read the complete note and confirmed PR <5%, suspicious axillary MRI, two benign FNAs, absent explicit whole-body staging, ATM mutation, MammaPrint −0.622, radiation-oncology recommendation for mastectomy, and node-dependent chemotherapy plan.

### Breast sample 19 — coral_idx 38

- Case: two distinct left-breast IDC lesions: grade 2 ER 61–70%/PR−/HER2−/Ki-67 15–20%, and grade 3 ER 41–50%/PR 1–10%/HER2−/Ki-67 30–40% with focal LCIS. Original clinical stage is 2–3. After NAHT, bilateral mastectomies, and tamoxifen, she is now on exemestane plus monthly goserelin and is NED on exam. MammaPrint is completed/low risk.
- PL P1: Type still merges both lesions into one ER+/PR− profile; Stage answers with response status `NED on exam` instead of clinical stage 2–3; findings puts future BSO consideration into current findings; supportive meds includes unlinked ondansetron; medication/therapy fields mix estradiol and DEXA and omit active goserelin; procedure misses BSO consideration; genetic results misses completed low-risk MammaPrint.
- PL P2: summary omits active goserelin and distinct lesions; curative is less precise than adjuvant/risk reduction; DEXA plan omits that routine breast imaging is unnecessary after bilateral mastectomy.
- Attribution: A0 second opinion, Stage, and Distant; A1 genetic results; A2 summary, Type, current meds, goal, medication plan, and therapy.
- Core verdicts: current_meds PL; Stage BL; Distant TIE; Metastasis TIE; response PL; Type PL; genetic results TIE. Total PL 3 / BL 1 / TIE 3.
- v2.1 regression: unsupported extensive DCIS is removed and current meds/response remain strong; Stage regresses from the explicit clinical stage to response language, and genetic results loses MammaPrint after cleaning unrelated pathology.
- Main verification: read the complete note and confirmed two separate lesion profiles, `Clinical stage 2-3`, current exemestane plus monthly goserelin, NED exam, BSO consideration, estradiol/DEXA routing, and completed `Mammaprint - low`.

### Breast sample 20 — coral_idx 39

- Case: synchronous very large bilateral breast cancers. Right is ER+/PR+/HER2+ with Ki-67 40%; left is ER+/PR+/HER2 0 with some lobular differentiation. Lung nodules and tiny liver lesions are unbiopsied with no confirmed metastasis. Treatment and surgery remain conditional on final pathology/FISH, size, nodes, and response. Germline panel is pending.
- PL P1: summary makes bilateral disease sound uniformly HER2+ and calls it early stage; Stage adds unsupported `locally advanced`; response says `Not mentioned` despite clearly untreated disease; therapy removes the global conditionality and mixes port, echo, labs, staging, and follow-up; procedure is a sentence fragment and misses conditional port/surgical uncertainty; lab plan copies non-lab preparation content.
- PL P2: Distant calls lesions suspected rather than neutrally unconfirmed; general Metastasis omits unknown regional-node status; redacted labs are called absent; curative intent lacks its dependency on final staging; goals description is empty; medication plan is too definite; next visit and follow-up include unsupported purpose or mixed imaging content.
- Attribution: A0 second opinion and Stage; A1 Patient type, response, and genetic plan; A2 summary, Type, both metastasis fields, findings, goal, medication/procedure/lab plans, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant PL; Metastasis TIE; response BL; Type TIE; genetic results TIE. Total PL 1 / BL 2 / TIE 4.
- v2.1 regression: Type now correctly separates right HER2+ from left HER2 0 and findings restores current tumor sizes; unsupported stage, untreated-response fallback, and conditional-plan routing remain.
- Main verification: read the complete note and confirmed side-specific receptors, lack of metastatic confirmation, recommendations contingent on pathology/FISH/nodes/surgery, no treatment started, surgery determined by response, and only-possible port/echo/lab/staging preparation after surgery.

### PDAC sample 1 — coral_idx 0

- Case: metastatic pancreatic adenocarcinoma after six cycles of gemcitabine/cisplatin; treatment is now paused for a break and surveillance. A prior biliary stent issue belongs to cycle 5. Current imaging raises concern for new pulmonary nodules and possible progression. The patient expresses interest in psycho-oncology support, but no completed referral is documented.
- PL P1: `recent_changes` foregrounds the stale cycle-5 biliary-stent event rather than the current completion of six cycles and treatment break; goals description is materially incomplete; response omits the new suspicious lung nodules and possible progression; Specialty converts a desire for psycho-oncology support into a definite referral.
- PL P2: summary and laboratory coverage are incomplete.
- Attribution: A0 goals and genetic-results fallback; A1 laboratory summary, recent changes, supportive medication, and Specialty; A2 summary, findings, response, next visit, and follow-up.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response PL; genetic results TIE. Total PL 1 / BL 0 / TIE 5.
- Main verification: read the complete note and confirmed six completed cycles followed by a break/surveillance, the historical timing of the cycle-5 stent event, new suspicious pulmonary nodules with possible progression, and interest in psycho-oncology without a documented referral order.

### PDAC sample 2 — coral_idx 1

- Case: resected pancreatic adenocarcinoma with 6/25 regional nodes, later confirmed liver metastases, progression after FOLFIRINOX, and current second-line C2D1 gemcitabine/nab-paclitaxel. Germline testing shows SPINK1 carrier status plus FANCG and NF2 VUS findings.
- PL P1: findings omits the defining resection pathology, liver metastases, prior progression, and the current assessment; Genetic Testing Plan misroutes family SPINK1 screening as testing planned for the patient; Specialty labels the existing GI Oncology clinic as a new referral.
- PL P2: Type is nonspecific; laboratory dates are mixed; supportive medication coverage is incomplete; response misses the treatment-change context; medication plan and next-visit wording are incomplete.
- Attribution: A0 second opinion, Metastasis, findings, genetic plan, and Specialty; A1 goals description; A2 summary, Stage, Distant, labs, response, medication plan, next visit, and genetic results.
- Core verdicts: current_meds TIE; Stage PL; Distant PL; Metastasis TIE; response PL; genetic results TIE. Total PL 2 / BL 0 / TIE 4.
- Main verification: read the complete note and confirmed Whipple pathology with 6/25 positive nodes, subsequent liver metastases, progression on FOLFIRINOX, current C2D1 gemcitabine/nab-paclitaxel, and that the SPINK1 recommendation concerns relatives rather than a new patient test.

### PDAC sample 3 — coral_idx 2

- Case: metastatic PDAC on third-line 5-FU/LV plus nanoliposomal irinotecan. Liver and peritoneal metastases are established. February imaging was mixed/stable, but March imaging and the current assessment show a larger pancreatic-tail mass with direct local invasion, partial gastric-outlet obstruction, ascites, and clinical decline. KRAS G12D and TP53 mutations are documented.
- PL P1: findings is anchored to older February imaging and omits the newer larger mass, direct invasion, gastric-outlet obstruction, and ascites; Procedure contains a stray radiotherapy fragment; Lab Plan copies a chemotherapy condition rather than an actual laboratory plan; Next visit invents a clinic visit from the CT plan; Advance care turns possible future hospice into a completed goals-of-care discussion; Referral follow-up again copies the CT plan rather than a return arrangement.
- PL P2: current medication has broken punctuation; supportive medication coverage is incomplete; goals description and response omit important recent detail, including the clinical decline and infectious-versus-atelectatic lung findings.
- Attribution: A0 second opinion, recent changes, procedure, next visit, advance care, and follow-up; A1 Patient type, findings, and supportive medication; A2 Distant, Metastasis, labs, goals description, response, and lab plan.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis PL; response PL; genetic results TIE. Total PL 4 / BL 0 / TIE 2.
- Main verification: read the complete note and confirmed the March 7.2 cm pancreatic-tail mass, direct spleen/posterior-stomach invasion, possible invasion of adjacent organs, partial gastric-outlet obstruction, ascites, current clinical decline, conditional chemotherapy based on labs, possible—not completed—future hospice transition, and the absence of a specified next clinic appointment.

### PDAC sample 4 — coral_idx 3

- Case: initially locally advanced/unresectable pancreatic adenocarcinoma. A historical segment-7 liver lesion had conflicting interpretations, while current CT describes liver hemangiomas and no suspicious lesions. After FOLFIRINOX and six cycles of gemcitabine/nab-paclitaxel, the patient is on surveillance with stable disease and good control. She is a CA19-9 non-secretor.
- PL P0: response fabricates new/enlarging lesions, a rising tumor marker, and suspected early recurrence, directly contradicting the current imaging and assessment.
- PL P1: findings omits the vascular encasement/narrowing that explains unresectability; recent changes is empty despite completion of six gemcitabine/nab-paclitaxel cycles followed by surveillance; goals description omits explicit continued good disease control.
- PL P2: Type omits the mucinous descriptor found in imaging history; Stage omits unresectable status; supportive-medication coverage is incomplete; Therapy `None` does not convey the intentional chemotherapy break/surveillance state.
- Attribution: A0 second opinion and response; A1 findings, supportive medication, and genetic results; A2 Stage, Distant, and Metastasis.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis TIE; response BL; genetic results PL. Total PL 2 / BL 1 / TIE 3.
- v2.1 change: Distant uncertainty and removal of fabricated regional nodes improve two metastasis fields, but response severely regresses from stable disease to a fabricated recurrence template.
- Main verification: read the complete note and confirmed `no significant change`, `No evidence of metastasis`, `No new suspicious nodule`, `continued good disease control on surveillance`, six completed gemcitabine/nab-paclitaxel cycles, and `She does not express CA-19-9`. The PDAC prompt explicitly includes non-secretor status under completed molecular results.

### PDAC sample 5 — coral_idx 4

- Case: Stage IV pancreatic adenocarcinoma with biopsy-proven oligometastatic abdominal-wall disease. After 12 cycles of FOLFIRINOX, pancreatic and abdominal-wall disease is stable and the patient is on a chemotherapy break. Foundation liquid biopsy reports MSI undetermined and RB1 P26fs*47.
- PL P1: Type omits the biopsy-proven oligometastatic context; recent changes selects an older cycle-10 thrombocytopenia hold instead of the current completion of 12 cycles and chemotherapy break; supportive medication includes ondansetron marked `Patient not taking` while omitting current fluconazole and A/P opioids; Therapy misroutes Creon/opioids as cancer therapy and omits the chemotherapy-break state; existing Nutrition and SMS/tumor-board care are mislabeled as new referrals.
- PL P2: findings is overlong and mixes treatment/laboratory/supportive content; future visit mode is not explicit.
- Attribution: A0 second opinion, Therapy, Nutrition, and Specialty; A1 labs, recent changes, and goals description; A2 summary, findings, supportive medication, response, medication plan, next visit, and advance care.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 0 / TIE 6.
- Main verification: read the complete note and confirmed biopsy-proven abdominal-wall metastasis, stable pancreas/abdominal-wall disease, completed 12-cycle FOLFIRINOX course, current chemotherapy break, `Patient not taking` ondansetron, current fluconazole/opioid plan, existing supportive services, and Foundation liquid-biopsy results.

### PDAC sample 6 — coral_idx 5

- Case: initially borderline-resectable/locally advanced pancreatic adenocarcinoma treated with 12 cycles of modified FOLFIRINOX and resection. Pathology shows a 1.7 cm residual tumor with 51–90% treatment effect, negative margins, and negative nodes. Postoperative CT and CA19-9 raise concern for early liver recurrence, but lesions are too small to confirm; repeat imaging and possible biopsy are planned.
- PL P1: response captures suspected recurrence but omits the documented pathologic treatment response; Procedure incorrectly includes the future scan instead of isolating the conditional biopsy; Next visit invents a definite in-person clinic appointment from the imaging interval.
- PL P2: Stage omits the original borderline-resectable/locally advanced status; lab output violates the string schema, contains an ALT spacing error, and misses the key CA19-9 trend; goals description leaves the explicit surveillance rationale unstated; Referral follow-up mixes imaging/biopsy planning into referral semantics.
- Attribution: A0 Patient type, second opinion, and next visit; A1 Distant, Metastasis, labs, findings, goals description, and genetic fallback; A2 Stage and response.
- Core verdicts: current_meds TIE; Stage PL; Distant PL; Metastasis TIE; response PL; genetic results TIE. Total PL 3 / BL 0 / TIE 3.
- v2.1 change: Distant now names the liver while preserving uncertainty, and general Metastasis no longer fabricates confirmed nodes or confirmed liver disease.
- Main verification: read the complete note and confirmed the 51–90% treatment effect with negative margins/nodes, new and enlarging tiny liver lesions, CA19-9 rise from 44 to 2250, `too small to evaluate further`, two-month repeat scans, conditional biopsy, and no explicit future clinic mode.

### PDAC sample 7 — coral_idx 6

- Case: pancreatic-tail adenocarcinoma with local extension but no distant disease. Gemcitabine/nab-paclitaxel was changed to every-other-week dosing after neutropenia; four cycles are complete and treatment continues. CT shows slight primary-tumor shrinkage. The right adnexal cystic mass is considered likely benign. Germline testing is negative for pathogenic variants with AXIN1, CTC1, ERCC4, and MC1R VUS findings.
- PL P1: Stage upgrades local extension to definite `unresectable` despite continuing discussion of possible surgery; findings calls the adnexal lesion a `cystic pelvic neoplasm` but drops `likely benign`; medication and therapy plans omit the gemcitabine/nab-paclitaxel name and every-other-week schedule; Next visit converts an eight-week imaging interval into a definite telehealth appointment.
- Attribution: A0 second opinion, Distant, and Metastasis; A1 Patient type, labs, current medication, recent changes, goals, goals description, and genetic results; A2 summary, Stage, findings, response, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 5.
- v2.1 change: current medication is repaired from empty to the active gemcitabine/nab-paclitaxel regimen; the remaining stage, certainty, plan-detail, and inferred-visit problems persist.
- Main verification: read the complete note and confirmed local extension/abutment, possible future surgery, no distant metastasis, `likely benign` adnexal mass, every-other-week gemcitabine/nab-paclitaxel, slight tumor shrinkage, and only an eight-week imaging plan—not a scheduled telehealth visit.

### PDAC sample 8 — coral_idx 7

- Case: initially resectable PDAC with Whipple pathology showing poorly differentiated ductal adenocarcinoma, LVI, pT2N2, 11/37 positive regional nodes, negative margins, and intact MMR. A gastrohepatic/mesenteric node later proved metastatic recurrence. The patient is now on every-other-week gemcitabine/nab-paclitaxel with nodal shrinkage and CA19-9 reduction. Germline ATM is documented; formal FoundationOne results remain pending, although no actionable mutation was communicated.
- PL P1: Type omits the current recurrent-metastatic state; Stage reports only historical pT2N2 and loses current metastatic recurrence; general Metastasis keeps only historical regional N2 and omits the current biopsy-proven nonregional nodal recurrence; recent changes is empty despite the new active regimen and three completed cycles; Specialty mislabels this incoming consultation as an outgoing referral; Referral follow-up contains the chemotherapy recommendation rather than the actual as-needed return arrangement.
- PL P2: findings contains an incorrect biopsy date and excessive normal examination detail; Therapy is incomplete in its future-trial list; future visit mode is inferred; genetic results omits the communicated no-actionable-mutation status.
- Attribution: A0 second opinion, Specialty, and follow-up; A1 Patient type, Type, and labs; A2 findings, goal, response, Therapy, next visit, and genetic results.
- Core verdicts: current_meds TIE; Stage BL; Distant PL; Metastasis PL; response PL; genetic results PL. Total PL 4 / BL 1 / TIE 1.
- v2.1 change: current meds, Distant, and response are substantially repaired; historical-only Stage and incomplete metastasis merging remain.
- Main verification: read the complete note and confirmed historical pT2N2/11-of-37 regional disease, later biopsy-proven gastrohepatic/mesenteric nodal recurrence, three cycles of current gemcitabine/nab-paclitaxel, favorable nodal and CA19-9 response, germline ATM/MMR results, pending formal FoundationOne report, and as-requested follow-up.

### PDAC sample 9 — coral_idx 8

- Case: lung-predominant biopsy-proven metastatic pancreatic cancer, currently treated with gemcitabine/nab-paclitaxel. CT shows stable treated lung metastases and a smaller pancreatic primary; CA19-9 fell from 3525 to 2762 to 1109. Abraxane was further dose-reduced today for neuropathy. Completed molecular results include a BRCA VUS and KRAS, CDKN2A, and APC mutations.
- PL P1: findings reverses the dated CA19-9 trend and calls the clear decline an increase; supportive medication omits several current symptom-control drugs and misspells olanzapine; medication plan omits the core gemcitabine/Abraxane continuation and dose reduction while treating the existing GOO stent as a medication action; Genetics misroutes a historical Phase-I trial consultation as a genetics referral.
- PL P2: summary overemphasizes nausea/vomiting despite a negative current review; recent changes mixes the two-week neuropathy follow-up into the dose-change field; goals description is generic; Therapy should explicitly say the reduced-dose combined regimen continues.
- Attribution: A0 second opinion and findings; A1 supportive medication, goals description, Genetics, and genetic results; A2 summary, Distant, Metastasis, current medication, goal, response, and medication plan.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis TIE; response PL; genetic results TIE. Total PL 2 / BL 0 / TIE 4.
- Main verification: read the complete note and confirmed biopsy-proven lung disease, stable treated pulmonary metastases, shrinking pancreatic primary, CA19-9 decline by date, current gemcitabine/Abraxane, same-day Abraxane reduction, active supportive drugs, and that the Phase-I consultation is unrelated to genetics.

### PDAC sample 10 — coral_idx 9

- Case: locally advanced pancreatic ductal adenocarcinoma. Irinotecan was permanently removed after cycle 2 because of colitis and poor tolerance; current C6D1 treatment is FOLFOX. The December CT, obtained when FOLFOX began, shows a stable primary and no confirmed distant disease, but a right-upper-lobe nodule remains indeterminate among scar, lung primary, and metastasis. RECQL4 VUS is documented.
- PL P1: summary simultaneously says current mFOLFIRINOX and FOLFOX; both metastasis fields overstate `No` and drop the explicit indeterminate lung nodule; findings likewise omits that differential; supportive medication misses the documented chemotherapy premedication, hydration, and potassium/magnesium support; response incorrectly uses same-day pre-FOLFOX imaging as response to the current regimen; Therapy contains potassium and monitoring contamination; Imaging mixes CA19-9 monitoring and a cancelled mammogram; Lab Plan mixes imaging and omits the explicit day-3 electrolyte check; Next visit invents an in-person mode; Specialty turns a conditional post-induction surgical reassessment into a current referral.
- PL P2: laboratory summary mixes in older albumin/protein values.
- Attribution: A0 Patient type and second opinion; A1 Televisit, findings, and goals description; A2 summary, both metastasis fields, labs, supportive medication, goals, response, Therapy, Imaging, Lab Plan, next visit, and Specialty.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 0 / TIE 6.
- v2.1 change: active regimen is repaired to FOLFOX and fabricated confirmed nodes/liver-biopsy text is removed, but the indeterminate lung lesion, treatment-response time boundary, and plan routing remain unresolved.
- Main verification: read the complete note and confirmed `FOLFOX only going forward`, the indeterminate RUL nodule differential, the December-10 same-day CT/switch timing, explicit support regimen and electrolyte recheck, cancelled mammogram, conditional surgical reconsideration, and two-week follow-up without a specified mode.

### PDAC sample 11 — coral_idx 10

- Case: new-patient second-opinion telehealth visit for clinically staged Stage IV cT2 cN1 cM1 pancreatic cancer with liver metastasis. No anticancer therapy has started. The patient consented only to Precision Promise screening; FOLFIRINOX and gemcitabine/nab-paclitaxel were discussed as alternative standard options. STRATA is pending, and the BRCA2 mutation belongs to the patient's brother.
- PL P1: medication plan turns trial-screening consent and unselected alternatives into a treatment plan; Therapy upgrades screening consent to trial participation; Imaging copies the lab half of a combined same-day CT/lab order; Lab Plan copies the CT half; existing next-day genetics appointment is treated as a new referral; incoming referral to UCSF is treated as an outgoing Specialty referral; Referral follow-up misuses same-day CT/labs as a return arrangement.
- PL P2: general Metastasis lists liver disease but omits clinically staged regional cN1 involvement.
- Attribution: A0 second opinion, medication plan, Specialty, and follow-up; A1 Patient type, supportive medication, response, genetic plan, next visit, Genetics, and genetic fallback; A2 findings and Therapy.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 0 / TIE 6.
- v2.1 change: general Metastasis no longer upgrades cN1 to pathologically confirmed nodes, and genetic results no longer copies the brother's BRCA2 result or pending STRATA; both become ties.
- Main verification: read the complete note and confirmed exact signed Stage IV cT2 cN1 cM1 staging, liver lesions, no active treatment, screening-only consent, alternative treatment discussion, same-day CT/labs, pre-existing genetics appointment, incoming referral, pending STRATA, and the brother-only BRCA2 result.

### PDAC sample 12 — coral_idx 11

- Case: pancreatic body/tail adenocarcinoma, initially at least T4N1. After 12 cycles of mFOLFIRINOX and a treatment holiday, disease clearly progressed with primary infiltration, new bowel obstruction, peritoneal/omental carcinomatosis, and biliary obstruction. Bilateral adrenal nodules remain suspicious. There is no active anticancer treatment; chemotherapy is judged more harmful than beneficial unless obstruction, bilirubin, and oral intake markedly improve, and hospice transition is anticipated.
- PL P1: visit mode says in-person although only the family attended while the patient was hospitalized; Distant and general Metastasis omit suspicious bilateral adrenal lesions and general Metastasis also omits historical clinical N1 disease; Lab Results is a truncated stringified JSON blob; supportive medication omits several active inpatient symptom-control drugs; goals description leaves explicit palliative/hospice rationale blank; response reports only older stable disease and omits current progression; medication and therapy plans omit the decision not to give chemotherapy plus the narrow conditional exception; Procedure says none despite conditional additional stent placement; Lab Plan mixes oral intake and obstruction with actual bilirubin/liver tests; Advance care upgrades anticipated hospice into completed ACP; Specialty and Referral follow-up invent referral semantics from palliative discussion and direct phone/imaging follow-up.
- PL P2: summary weakens established carcinomatosis/progression to concern; findings mixes historical and current timepoints; recent changes includes a current treatment decision; Imaging describes follow-up of a same-day completed study rather than a new future scan.
- Attribution: A0 second opinion, visit mode, Advance care, Specialty, and follow-up; A1 labs, supportive medication, and genetic fallback; A2 Stage, both metastasis fields, findings, recent changes, and response.
- Core verdicts: current_meds TIE; Stage TIE; Distant PL; Metastasis PL; response BL; genetic results TIE. Total PL 2 / BL 1 / TIE 3.
- v2.1 change: liver certainty and fabricated confirmed-node text are repaired, producing two metastasis wins; stale response and plan-field failures remain.
- Main verification: read the complete note and confirmed family-only attendance, explicit interval progression, peritoneal/omental carcinomatosis, suspicious bilateral adrenal nodules, historical T4N1, active symptom medications, chemotherapy-harm assessment, narrow conditional resumption, possible additional stent, anticipated—not completed—hospice transition, and direct phone follow-up.

### PDAC sample 13 — coral_idx 12

- Case: locally advanced unresectable pancreatic head/uncinate adenocarcinoma with direct duodenal invasion and radiographic regional nodal involvement, but no established distant metastasis. Six cycles of Gem/Cape and December radiotherapy are complete; systemic treatment is now paused. January CT shows stable local disease. The patient is a CA19-9 non-secretor, current genomics is documented as not done, and February bronchoscopy is planned by pulmonary/ID care.
- PL P1: general Metastasis says `No` and drops the radiographic regional nodes; findings mixes an older `increased pneumobilia/biliary dilation` state into the newer scan that says decreased pneumobilia; recent changes is empty despite completed radiation and systemic-therapy pause; response mixes improvement of infectious/drug-related organizing pneumonia into cancer response; medication plan omits continued Creon and the systemic pause; Procedure omits planned bronchoscopy; Lab Plan mixes PRBC transfusion into a blood-test-only field; genetic plan preserves an old `UCSF500 in process` state despite current `Genomics Not done`; Next visit and Referral follow-up mislabel an ID/pulmonary appointment as oncology follow-up; Specialty includes completed tumor-board and radiation consultations as new referrals.
- PL P2: Type omits head/uncinate location and direct invasion; supportive medication omits Norco, dronabinol, and other current support; radiotherapy mixes in the systemic-treatment pause; Imaging points to an already-completed CT.
- Attribution: A0 Patient type, second opinion, Imaging, next visit, Specialty, and follow-up; A2 Stage, Distant, Metastasis, findings, supportive medication, goal, response, medication plan, and genetic plan.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis BL; response BL; genetic results PL. Total PL 3 / BL 2 / TIE 1.
- Main verification: read the complete note and confirmed regional nodal involvement, newer decreased pneumobilia, six completed Gem/Cape cycles, completed December radiation, paused systemic therapy, stable cancer, nonmalignant organizing-pneumonia improvement, continued Creon/Norco, planned bronchoscopy, current `Genomics Not done`, and that the February appointment belongs to another specialty.

### PDAC sample 14 — coral_idx 13

- Case: new-patient telehealth consultation for biopsy-supported Stage IV pancreatic-origin adenocarcinoma with liver metastases and radiographically progressive upper-abdominal/mesenteric nodes. A CBD stent was just placed for malignant obstruction. No chemotherapy has started. The clinician favors gemcitabine/nab-paclitaxel but will reassess after short-term in-person review and LFT normalization; UCSF500 is planned.
- PL P1: Procedure misroutes UCSF500 and upgrades conditional trial biopsy requirements into current procedures, including a trial explicitly paused to enrollment; Specialty labels the incoming UCSF consultation as an outgoing referral.
- PL P2: general Metastasis fails to label nodal disease as radiographic and omits site specificity; medication plan says `planned to start` rather than preserving `favor ... pending in-person reassessment`.
- Attribution: A0 second opinion, Metastasis, Procedure, Specialty, and genetic fallback; A1 Patient type, labs, supportive medication, and response; A2 Distant and findings.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response TIE; genetic results TIE. Total PL 1 / BL 0 / TIE 5.
- Main verification: read the complete note and confirmed liver-biopsy adenocarcinoma interpreted as pancreatic origin, progressive hepatic/nodal disease, just-completed biliary stenting, no active therapy, tentative regimen preference, conditional trial eligibility/biopsies, paused REVOLUTION enrollment, planned UCSF500, and short-term in-person reassessment.

### PDAC sample 15 — coral_idx 14

- Case: resected pancreatic-head ductal adenocarcinoma after neoadjuvant FOLFIRINOX. Final pathology shows a 4.6 cm moderately differentiated tumor, PNI, positive margins, 11/46 regional nodes, ypT3N2, and poor pathologic treatment response. No current distant metastasis is established, but CA19-9 rose from 48 to 4375, prompting repeat marker and CT. Tumor MMR is intact; a benign ascites sample is KRAS-negative.
- PL P1: recent changes incorrectly treats a non-oncology anticoagulant hold as a cancer-treatment change; Therapy misroutes Creon as anticancer therapy; Next visit infers an in-person appointment and copies test timing rather than `RTC after`; Specialty contains historical radiation/systemic consultations while missing the current conditional PCP referral; Others omits that PCP referral; Referral follow-up misroutes oncology testing/return into referral follow-up; genetic results omits intact MMR and retains only benign-ascites KRAS negativity.
- PL P2: findings mixes timepoints and asserts no significant weight loss without support; supportive-medication coverage is incomplete; goals description omits high-risk recurrence surveillance; response should explicitly state Evans grade I/Ryan 3 poor response and separate it from current suspected recurrence.
- Attribution: A0 second opinion, Stage, Distant, Therapy, next visit, and follow-up; A1 labs, goals description, Specialty, and genetic results; A2 Patient type, Type, Metastasis, findings, and response.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response PL; genetic results BL. Total PL 2 / BL 1 / TIE 3.
- v2.1 change: Type is corrected from the wrong grade, Metastasis is repaired, and response now recognizes poor treatment effect; completed MMR remains missing.
- Main verification: read the complete note and confirmed authoritative ypT3N2 pathology despite the inconsistent A/P shorthand, 11/46 positive nodes, positive margins, Evans grade I/Ryan score 3, no confirmed distant disease, steep CA19-9 rise, current CT/marker surveillance, conditional PCP referral, intact MMR, and benign-fluid KRAS result.

### PDAC sample 16 — coral_idx 15

- Case: Stage IIB cT1c cN1 cM0 pancreatic-head adenocarcinoma with imaging-suspicious peripancreatic/periportal nodes but no pathologic node confirmation and no distant disease. The patient is on dose-reduced gemcitabine monotherapy; one dose was delayed for a dental concern, then cleared, and the schedule is changing to every other week. The latest CT shows the pancreatic mass is no longer visible with improved duct dilation.
- PL P1: Next visit adds an unsupported in-person mode; Specialty turns an already-completed dental evaluation into a current referral.
- PL P2: summary omits the C2D15 decision and schedule change; Type omits pancreatic-head location; findings slightly overgeneralizes the abdominal examination; recent changes omits the one-cycle dental delay; supportive medication misses daily senna advice; Imaging omits that the next scan follows this cycle.
- Attribution: A0 second opinion and Specialty; A1 Stage and labs; A2 summary, Metastasis, findings, current medication, supportive medication, medication plan, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response TIE; genetic results TIE. Total PL 1 / BL 0 / TIE 5.
- v2.1 change: general Metastasis is repaired from fabricated confirmed nodes and an invented liver-biopsy pathway to calibrated imaging-suspicious regional nodes with no distant disease.
- Main verification: read the complete note and confirmed exact Stage IIB cT1c cN1 cM0 staging, PET-suspicious regional nodes only, no distant disease, active gemcitabine, completed dental clearance, every-other-week change, marked radiographic response, daily senna recommendation, and dated follow-up without a documented mode.

### PDAC sample 17 — coral_idx 16

- Case: locally advanced pMMR pancreatic body/tail adenocarcinoma with extensive local vascular involvement but no nodal or distant metastasis. Eight FOLFIRINOX cycles are complete, the patient has been on a chemotherapy break since November 2018, and the February 2019 CT remains stable. June oncology follow-up and July CT are planned. Foundation reports MSS, TMB 5, KRAS G12V, TP53 I195F, and listed variants.
- PL P1: lab summary selects an older CEA value while omitting same-date CA19-9 and provides no date; Therapy says `None` instead of representing the ongoing chemotherapy break.
- PL P2: Type omits pMMR; findings mixes isolated older laboratory values; palliative intent is reasonable but not explicit; response adds uncontextualized old labs rather than a true trend; future visit mode is inferred.
- Attribution: A0 second opinion and in-person; A1 Distant, labs, goals, goals description, and genetic results; A2 Patient type, Stage, Metastasis, findings, response, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response PL; genetic results TIE. Total PL 1 / BL 0 / TIE 5.
- v2.1 change: fabricated confirmed nodes and a fabricated liver-biopsy pathway are removed; response continues to use the latest stable scan.
- Main verification: read the complete note and confirmed locally advanced vascular disease, no distant or nodal spread, eight completed FOLFIRINOX cycles, ongoing treatment break, latest stable February CT, June return, July CT, same-date CEA/CA19-9 values, pMMR, and full Foundation results.

### PDAC sample 18 — coral_idx 17

- Case: initially resectable pancreatic-tail moderately differentiated adenocarcinoma. After only two poorly tolerated neoadjuvant Gem/Abrax doses, distal pancreatectomy/splenectomy showed negative margins and 2/29 positive regional nodes. Two postoperative Gem/Cape cycles were stopped for severe hand-foot syndrome and mucositis, with a five-days-on/seven-day-cycle restart planned after recovery. Current CT shows no distant disease; capecitabine is explicitly not being taken. ATM VUS is documented.
- PL P1: Stage is empty and omits resected node-positive status; findings is dominated by laboratory values and omits decisive pathology/toxicity context; supportive medication treats future Lovenox as current and omits active Tylenol/Flexeril/gabapentin; medication plan omits the Gem/Cape regimen name and mixes Doppler imaging into the drug plan; Therapy omits the regimen and restart schedule; Referral follow-up misroutes the chemotherapy hold/restart decision.
- PL P2: goals description omits the adjuvant/restart rationale; response reports no metastatic disease but does not explain that there is no measurable disease with which to assess adjuvant response.
- Attribution: A0 Patient type, Distant, and follow-up; A1 second opinion, labs, findings, goals description, response, next visit, and genetic results; A2 Type, Metastasis, supportive medication, and medication plan.
- Core verdicts: current_meds PL; Stage BL; Distant TIE; Metastasis PL; response PL; genetic results TIE. Total PL 3 / BL 1 / TIE 2.
- v2.1 change: general Metastasis and response improve while current medication remains correctly empty during a full regimen hold; Stage remains unresolved.
- Main verification: read the complete note and confirmed two-dose neoadjuvant exposure, definitive surgery with 2/29 positive nodes, postoperative Gem/Cape, current complete hold, planned five-of-seven-day restart, capecitabine marked not taking, no distant disease, current supportive drugs, future Lovenox/Doppler plan, and ATM VUS.

### PDAC sample 19 — coral_idx 18

- Case: locally advanced unresectable pancreatic head/uncinate adenocarcinoma with prior progression on Gem/Abrax and current dose-modified FOLFIRINOX. Cycle 3 was held once for marked cholestasis. The same-day CT addendum confirms local primary progression with biliary obstruction and possible partial duodenal obstruction; the historical 11 mm liver lesion remains uncharacterized. Same-day leg ultrasound was negative for DVT, and an urgent GI referral for ERCP was placed.
- PL P1: findings omits the negative leg ultrasound and contains a damaged alkaline-phosphatase value; recent changes still describes the already-completed CT as future; Imaging lists the completed CT and ultrasound as future plans; Specialty retains SMS but omits the urgent GI/ERCP referral; Referral follow-up misroutes that GI referral as ordinary follow-up.
- PL P2: summary does not integrate the confirming addendum; Type omits the uncinate location; response retains `possible` after definite interval local growth; Procedure omits the urgent-referral context.
- Attribution: A0 second opinion, Distant, Metastasis, next visit, and follow-up; A1 supportive medication and both goal fields; A2 summary, Type, Stage, labs, findings, current medication, recent changes, Imaging, and Specialty.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis TIE; response PL; genetic results TIE. Total PL 3 / BL 0 / TIE 3.
- v2.1 change: general Metastasis no longer fabricates confirmed regional nodes, and all former BL core wins are eliminated.
- Main verification: read the complete note and confirmed active FOLFIRINOX with a single-cycle hold, addendum-confirmed local progression/obstruction, persistent uncertainty of the old liver lesion, completed CT and negative leg ultrasound, and urgent GI/ERCP referral.

### PDAC sample 20 — coral_idx 19

- Case: newly diagnosed metastatic pancreatic-tail adenocarcinoma. Formal imaging shows definite peritoneal carcinomatosis and omental caking; multiple liver lesions are suspicious but unconfirmed, and retroperitoneal nodes are only prominent. No therapy has started. Standard regimens and trials are under discussion; the patient is merely interested in REVOLUTION and has not consented or screened. UCSF500 is ordered, germline counseling/testing is recommended, and MMR is intact by IHC.
- PL P1: Type omits the explicit metastatic state; Procedure upgrades conditional trial tissue collection/biopsy requirements into current procedures; Referral follow-up treats conditional nontherapeutic-research eligibility as a follow-up arrangement; genetic results incorrectly says none and omits completed intact MMR.
- PL P2: general Metastasis could retain prominent but unconfirmed retroperitoneal nodes; findings weakens definite carcinomatosis and omits omental caking/nodes; supportive medication omits the mainly used Tylenol; medication plan underemphasizes REVOLUTION interest and lack of consent; Genetics wording is a recommendation rather than a clearly placed referral.
- Attribution: A0 second opinion, Procedure, follow-up, and genetic results; A1 labs, supportive medication, response, and next visit; A2 Patient type, Distant, Metastasis, findings, and genetic plan.
- Core verdicts: current_meds TIE; Stage TIE; Distant PL; Metastasis PL; response TIE; genetic results TIE. Total PL 2 / BL 0 / TIE 4.
- v2.1 change: both metastasis fields now preserve mixed certainty and remove fabricated confirmed nodes; completed MMR remains omitted.
- Main verification: read the complete note and confirmed definite peritoneal/omental disease, only-suspicious liver lesions, unconfirmed retroperitoneal nodes, no active therapy, discussion-only systemic options, conditional REVOLUTION interest/screening, planned UCSF500/germline work, and completed intact MMR.
