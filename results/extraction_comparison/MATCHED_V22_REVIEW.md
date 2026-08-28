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

- Completed: 20/40 (breast 20/20, PDAC 0/20)
- PL findings: P0=3, P1=116, P2=89
- Attribution findings: A0=60, A1=74, A2=143
- Core verdict totals (PL / BL / TIE): 29 / 20 / 91
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
