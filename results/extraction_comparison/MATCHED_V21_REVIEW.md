# Matched v2.1 PL vs BL Manual Review

Date: 2026-08-27

## Scope and rules

- Compare the post-fix pipeline outputs (`pipeline_*_matched_v21.txt`) with the semantic-contract single-prompt baseline (`baseline_extract_*_matched_v21.txt`).
- For every sample, read the complete `note_text`, every PL keypoint, PL attribution, and every BL keypoint. Check field definitions in the relevant extraction and plan prompts.
- Clinical priority: faithful precision > coverage > simple vocabulary > patient-friendly wording.
- Severity: P0 = fabricated/unsupported high-impact clinical fact; P1 = wrong field/direction, major omission, temporal error, or materially misleading wording; P2 = minor incompleteness, imprecision, formatting, or attribution weakness; OK = no material issue.
- Core comparison fields: `current_meds`, `Stage_of_Cancer`, `Distant Metastasis`, `Metastasis`, `response_assessment`, breast `Type_of_Cancer`, and `genetic_testing_results`.
- Empty BL values and explicit negative/unknown PL fallbacks are treated as equivalent when the source truly provides no supported answer.

## Status

- Completed: 20/40 (breast 20/20 complete)
- Next: PDAC sample 1
- PL findings: P0=8, P1=117, P2=62
- Attribution findings: A0=45, A1=101, A2=126
- Core verdict totals (PL / BL / TIE): 28 / 28 / 84

## Results

### Breast sample 1 — coral_idx 20

- Case: untreated right grade-3 IDC after mastectomy; final addendum/A&P support TNBC; stated Stage II / pT2N1a with 1/2 positive regional sentinel nodes; PET/CT pending; no completed molecular result.
- PL P1: `Clinical_Findings.findings` mixes in laboratory values, including stale 2015 albumin; `Medication_Plan` incorrectly says the discussed chemotherapy regimens are not planned to be initiated, although the physician deferred the decision until PET/CT and said lower-risk chemotherapy might still be considered; `Therapy_plan=None` omits that deferred treatment decision.
- PL P2: Type does not clearly explain that the final HER2-negative addendum/A&P overrides the conflicting earlier wording; Stage over-converts Stage II/pT2N1a to Stage IIB; curative intent is reasonable but unmarked inference while staging is pending; next-visit mode and one empty referral value are minor presentation gaps.
- Attribution: A0 second-opinion citation points only to PET/CT; A1 for Patient type, Type, labs, findings, goals description, medication plan, and therapy plan; A2 for summary, Stage, Distant, Metastasis, goals, response, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis PL; response TIE; Type PL; genetic results PL. Total PL 3 / BL 1 / TIE 3.
- Main verification: confirmed all three P1s against the complete note. Key source language is `requires additional testing before final recommendations can be made`, `might want to consider a lower risk chemotherapy were she found to have metastasis`, and `obtain a PET/CT and resume our conversation ... at that time`.

### Breast sample 2 — coral_idx 21

- Case: historical ER+/PR− grade-1 IDC, now untreated unresectable locoregional/chest-wall recurrence; current liver finding is a cyst, brain/bone studies are negative, axillary nodes are unproven/previously benign; aromatase inhibitor is planned and zoledronic acid is for osteoporosis.
- PL P0: Type fabricates HER2− although HER2 is never reported; general Metastasis fabricates a `confirmed ipsilateral axillary node` despite benign/not-proven node evidence.
- PL P1: Distant carries forward obsolete liver concern and wrongly treats chest-wall recurrence as distant; palliative intent is unsupported and conflicts with possible shrinkage→surgery/radiation for long-term control; procedure plan omits possible resection; imaging and lab plans report already completed tests as future.
- PL P2: labs omit other completed normal studies; findings include labs; goals description omits explicit long-term-control wording; medication/therapy plan could be more specific; one referral fallback is blank rather than `None`.
- Attribution: A1 for Type, Distant, findings, goals description, and procedure plan; A2 for visit fields, Stage, Metastasis, goals, response, imaging, and lab plan. The Metastasis citation (`No other sites of disease`) conflicts with the fabricated confirmed node.
- Core verdicts: current_meds PL; Stage TIE; Distant BL; Metastasis BL; response PL; Type BL; genetic results TIE. Total PL 2 / BL 3 / TIE 2.
- Main verification: confirmed the P0/P1 findings from `right axillary lymph node biopsy ... benign pathology`, `axillary nodes which have not been proven to be involved with cancer`, liver `consistent with cyst`, `No other sites of disease`, and the explicit possible resection/radiation plan.

### Breast sample 3 — coral_idx 22

- Case: untreated right spindle-cell metaplastic TNBC, locally advanced and multifocal; right axillary FNA negative; PET has no definite distant disease but a 2.3 cm indeterminate right adrenal nodule; InVitae panel and MammaPrint are pending.
- PL P1: visit summary says `early stage` despite final A/P `locally advanced, multifocal`; findings omit the decisive negative axillary FNA; therapy plan includes port/chemo teaching outside the field; procedure plan omits chemo teaching required by its prompt; MammaPrint is misrouted to Imaging; genetic plan omits pending MammaPrint and misstates the already-sent panel; historical Genetics testing is mislabeled as a new referral; pending tests/pathology are placed in completed genetic results.
- PL P2: general Metastasis strengthens `indeterminate` to `suspicious`; labs lack values; next-visit mode is omitted.
- Attribution: A0 for second opinion, summary, imaging, Genetics referral, and genetic results; A1 for Patient type, both metastasis fields, labs, findings, response, medication plan, and therapy plan; A2 for in-person, Type, lab/genetic plans, and follow-up wording.
- Core verdicts: current_meds TIE; Stage TIE; Distant PL; Metastasis PL; response TIE; Type TIE; genetic results BL. Total PL 2 / BL 1 / TIE 4.
- Main verification: confirmed every P1 against the complete note, including `FNA negative`, `locally advanced, multifocal`, `mammaprint is pending`, `126 InVitae panel sent`, `Medi-port`, and `arrange for chemo teaching session`.

### Breast sample 4 — coral_idx 23

- Case: untreated multifocal right grade-3 IDC with two distinct low-HR profiles: one lesion ER 1%/PR−/HER2− and the other ER−/PR 1%/HER2−; local breast disease enlarged to 2.7 × 1.7 cm; PET/CT and axillary imaging show no regional or distant metastasis; weekly paclitaxel, port, and chemotherapy teaching are planned; known BRCA1 carrier.
- PL P1: Type merges the two lesions into one ER+/PR− profile; general Metastasis puts a growing local breast lesion into the metastasis field; `recent_changes` treats future port/teaching/paclitaxel as completed changes; medication plan mixes procedures, teaching, and visit scheduling into the drug-only field; historical surgery consultations are mislabeled as a new Specialty referral.
- PL P2: visit summary omits the multifocal biomarker detail; curative intent is reasonable but not explicitly sourced; next-visit mode is inferred from planned clinic visits rather than directly stated.
- Attribution: A0 for Distant Metastasis, whose growth citation does not support `No`; A1 for Type, Stage, Metastasis, goals, Specialty referral, and BRCA1 result; A2 for second opinion, the much broader findings value, the generated goals fallback, and the partially supported medication-plan value.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response PL; Type BL; genetic results TIE. Total PL 1 / BL 2 / TIE 4.
- Main verification: read the complete note and confirmed that the two biopsies have different ER/PR profiles, PET states `No evidence of hypermetabolic metastatic disease`, and port/teaching/weekly Taxol are future plans rather than recent treatment changes.

### Breast sample 5 — coral_idx 24

- Case: bilateral resected IDC: left Stage III T3N1 grade-3 ER+/PR+/HER2− disease with a confirmed regional micrometastasis and high-risk MammaPrint; right Stage I T1cN0 grade-1 ER+/PR+/HER2− disease with low-risk MammaPrint; Oncotype is pending; planned sequence is TC ×4, radiation, then aromatase inhibitor.
- PL P1: Type lists two profiles without side labels, which is materially ambiguous in bilateral disease; general Metastasis correctly identifies a regional node but then contradicts itself with `distant disease uncertain — no evidence`; Referral follow-up contains DEXA/exercise advice rather than an appointment; genetic results incorrectly say none despite two completed MammaPrint results.
- PL P2: Clinical Findings mixes the two genomic-risk results into a pathology/imaging/exam field.
- Attribution: A0 for second opinion, Distant Metastasis, and Referral follow-up; A1 for Patient type and response; A2 for the partial summary, Type, Metastasis, findings, generated goals fallback, generated next-visit fallback, and genetic-results citation.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response TIE; Type BL; genetic results TIE. Total PL 1 / BL 1 / TIE 5.
- Main verification: confirmed left/right surgical biomarker profiles, Stage III versus Stage I, the prior positive regional node/micrometastasis, MammaPrint scores `−0.614` and `+0.321`, and the still-pending Oncotype result.

### Breast sample 6 — coral_idx 25

- Case: untreated right grade-2 ER−/PR−, HER2-amplified IDC with biopsy-confirmed regional axillary disease; left iliac and bilateral sacral lesions are suspicious but require biopsy for a definitive Stage IV diagnosis; the stable carotid-body paraganglioma is a separate finding; THP is conditional on Stage IV confirmation.
- PL P0: Stage is upgraded from pending confirmation to definite `Stage IV`; Distant Metastasis is upgraded from suspicious bone lesions to definite `Yes`.
- PL P1: colonoscopy-prep ondansetron is mislabeled as oncology supportive medication; palliative intent is too certain while metastatic staging is unresolved; goals description misses the explicit `excellent response and possible long term disease control`; medication plan drops the `If stage IV` condition; procedure plan omits bone biopsy and chemotherapy teaching; Referral follow-up merely dumps the entire plan rather than a follow-up appointment.
- Attribution: A0 for second opinion and Referral follow-up; A1 for the unsupported genetic-results fallback; A2 for Type, definite Stage, definite Distant, the incomplete combined Metastasis support, broad findings, palliative goal, generated goals/response fallbacks, condition-stripped medication plan, and generated next-visit wording.
- Core verdicts: current_meds TIE; Stage BL; Distant BL; Metastasis PL; response TIE; Type PL; genetic results TIE. Total PL 2 / BL 2 / TIE 3.
- Main verification: confirmed the source says `suspicious for bone metastasis`, requires `biopsy for definitive stage IV diagnosis`, and conditions paclitaxel/trastuzumab/pertuzumab on `If stage IV`; the carotid lesion is described separately as a longstanding paraganglioma.

### Breast sample 7 — coral_idx 26

- Case: recurrent grade-3 TNBC with biopsy-confirmed liver metastasis, historical regional N1a disease, current regional nodes and a suspicious lytic S1 lesion; currently cycle 1 day 8 of pembrolizumab plus nab-paclitaxel. Current A/P notes improved axillary pain as a possible early response; the January PET predates this regimen. Germline MSH2 pathogenic variant and a negative breast-cancer gene panel are documented.
- PL P1: general Metastasis mixes historical regional disease and current nodes into an unclear confirmed-distant statement; lab summary omits that labs are adequate for treatment; findings omit the S1, chest-wall/pectoralis, and regional-node findings and present pretreatment imaging as current status; `recent_changes` contains supportive/anticoagulation adjustments while omitting the newly started anticancer regimen; supportive medications omit multiple current A/P drugs; response incorrectly treats pretreatment PET progression as response to the current regimen and misses the explicit early clinical improvement; medication plan omits restart/temporary-stop conditions for rivaroxaban; imaging plan says none despite explicit symptom-guided or 3–4-month surveillance.
- PL P2: summary makes pretreatment progression sound current; Type omits grade/recurrent-metastatic context; Distant omits the suspicious S1 lesion; genetic results should distinguish the colon-tumor MMR finding from germline and breast-panel results.
- Attribution: A0 for second opinion and current response; A1 for summary, Type, labs, findings, goals description, imaging plan, and genetic results; A2 for Patient type, Distant, Metastasis, supportive medications, and medication plan.
- Core verdicts: current_meds TIE; Stage PL; Distant PL; Metastasis TIE; response BL; Type PL; genetic results PL. Total PL 4 / BL 1 / TIE 2.
- Main verification: confirmed `02/25/19 Liver biopsy: triple negative metastatic breast cancer`, current treatment start on `03/11/19`, current `cycle 1 day 8`, `axillary pain improved which is hopeful for early treatment response`, and the explicit imaging/lab/rivaroxaban plans. The enlarging liver lesions belong to the pre-regimen PET, not a current-treatment response scan.

### Breast sample 8 — coral_idx 27

- Case: newly diagnosed multifocal left IDC after bilateral mastectomies, largest focus 3.9 cm, mainly grade 2 with a small grade-1 focus, ER+/PR+/HER2−, extensive DCIS, Stage IIA pT2(m)N1a with 2/12 regional nodes positive; MammaPrint is completed and high risk; AC→paclitaxel, later AI, baseline TTE, chemotherapy teaching, port, and genetics referral are planned.
- PL P1: general Metastasis correctly identifies regional disease but adds a contradictory unsupported `distant disease uncertain — no distant disease confirmed`; findings focus on postoperative exam and stale imaging while omitting the defining current pathology and positive nodes; therapy plan mixes TTE and chemotherapy teaching into treatment and omits the planned AI; genetic results falsely say none despite completed high-risk MammaPrint.
- PL P2: Type omits multifocality, 3.9 cm size, and the small grade-1 focus; goals description is blank; estimated chemotherapy start is presented as a confirmed next clinic visit; Referral follow-up carries a pre-existing outside appointment whose relation to this plan is unclear.
- Attribution: A0 for Referral follow-up; A1 for Distant, findings, and response; A2 for Type, Metastasis, medication plan, therapy plan, genetic results, and indirect second-opinion, in-person, and curative-goal support.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- Main verification: confirmed `Stage IIA pT2(m)N1a`, 2/12 positive regional nodes, multifocal mixed-grade IDC with DCIS, `Mammaprint: High-Risk`, and the separate chemotherapy, TTE, teaching, port, and genetics plans. Both systems miss MammaPrint in genetic results.

### Breast sample 9 — coral_idx 28

- Case: original left Stage III T3N2 grade-2 ER+/PR+/HER2− micropapillary IDC treated with mastectomy, incomplete ddAC, and later endocrine therapy; now biopsy-confirmed locally advanced unresectable breast recurrence. Current regional nodes are suspicious and a level Vb cervical node may represent distant disease, pending FNA. No new systemic regimen has begun; goserelin followed by AI is planned.
- PL P1: Stage gives only suspected Stage IV and omits the original Stage III plus current locally advanced unresectable state; Metastasis fails to mark the confirmed axillary node as historical and omits several current suspicious regional nodes; findings omit the current recurrence biopsy and biomarkers; planned goserelin/AI is mislabeled as a recent completed change; palliative intent drops the clinician's explicit `if ... MBC` condition; response says never treated/no response despite prior treatment followed by biopsy-proven recurrence.
- PL P2: summary omits the current unresectable and suspected-metastatic status; Type omits current Ki-67 30–40%; the FNA clinic could also be represented as a specialty pathway, although the procedure itself is captured.
- Attribution: A0 for second opinion, recent changes, and the unconditional palliative goal; A1 for Patient type, findings, and response; A2 for summary, Type, Stage, Metastasis, and the compound therapy value.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis TIE; response TIE; Type PL; genetic results TIE. Total PL 1 / BL 1 / TIE 5.
- Main verification: confirmed the original `Stage III (T3N2)`, current `locally advanced, unresectable recurrence`, conditional `possibly considered metastatic` level Vb node, planned FNA, future goserelin/AI, and prior systemic/endocrine therapy followed by recurrence. The current palliative framing is conditional until distant disease is confirmed.

### Breast sample 10 — coral_idx 29

- Case: newly diagnosed right grade-2 IDC, ER >95%, PR 25%, clinically HER2 non-amplified, clinical Stage II cT2N1 with FNA-confirmed regional axillary disease and PET-negative distant staging; undergoing egg harvesting, with no current anticancer therapy; conditional plan for neoadjuvant paclitaxel→AC, port, cardiac testing, MammaPrint, and already-sent genetic testing.
- PL P1: medication plan includes port placement, which belongs under procedures; therapy plan turns `hopefully ... assuming egg harvesting` into a definite 05/01 start; imaging plan omits the scheduled echocardiogram/EKG; genetic plan retains MammaPrint but omits the already-sent genetic test; the echocardiogram/EKG date is mislabeled as a clinic visit; already-sent testing is mislabeled as a Genetics referral.
- PL P2: findings do not state the PET-negative distant result directly; fertility-preservation medications are omitted from supportive context; Referral follow-up mixes tests and treatment logistics; cold-cap coordinator contact is not represented cleanly.
- Attribution: A0 for second opinion, in-person, and next clinic visit; A1 for labs, findings, goals description, response, and Genetics referral; A2 for Patient type, summary, Type, Metastasis, goals, medication plan, therapy plan, genetic-results fallback, and mixed follow-up support.
- Core verdicts: current_meds PL; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type TIE; genetic results TIE. Total PL 1 / BL 0 / TIE 6.
- Main verification: confirmed positive axillary FNA, clear PET, no started anticancer regimen, active egg-harvesting drugs, `Hopefully start ... assuming egg harvesting`, echocardiogram/EKG, MammaPrint pending, and `genetic testing sent`. BL incorrectly promotes fertility drugs and unconfirmed tamoxifen to current anticancer treatment.

### Breast sample 11 — coral_idx 30

- Case: pure left-breast DCIS after lumpectomy, final excision pathology intermediate grade, 1.8 cm, solid/cribriform, no necrosis, positive/very close posterior margin, pTisNx, ER/PR positive, HER2 untested. Tamoxifen was prescribed but explicitly held until radiation-oncology assessment.
- PL P1: Type uses the earlier core-biopsy profile (`PR pending`, micropapillary pattern, focal necrosis) instead of prioritizing final excision pathology; Stage is blank despite explicit `pTisNx`; the held future tamoxifen plan is mislabeled as a recent treatment change.
- PL P2: findings should explicitly state no invasive cancer and use final pathology architecture/necrosis; goals description contradicts its explicit content by appending `Not explicitly stated`; therapy wording slightly overstates a conditional decision; future visit mode is not stated; radiation-oncology appointment is not clearly a new outgoing referral from this encounter.
- Attribution: A0 for second opinion; A1 for Patient type, Type, Distant, Metastasis, and response; A2 for in-person, summary, findings, goals description, and therapy plan.
- Core verdicts: current_meds TIE; Stage BL; Distant TIE; Metastasis TIE; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- Main verification: confirmed final pathology `pTisNx`, ER/PR positive, no invasive tumor, solid/cribriform architecture and no necrosis. Both systems retain the earlier `PR pending`; only BL captures stage. The tamoxifen prescription is explicitly on hold pending radiation assessment.

### Breast sample 12 — coral_idx 31

- Case: untreated right clinical Stage II, node-negative grade-2 invasive mammary carcinoma with mixed ductal/lobular features, ER >95%, PR ~70%, HER2−, Ki-67 ~20%, PET-negative distant staging, and completed high-risk MammaPrint; curative neoadjuvant AC/T plus later surgery/endocrine treatment is planned.
- PL P1: lab summary incorrectly says none despite a complete recent numeric panel; therapy plan mixes echo, port, and chemotherapy teaching into treatment; procedure plan omits chemotherapy teaching required by its schema; lab plan misses explicit `labs`; Referral.Others omits cold-cap CRC contact; genetic results misses completed `MP high risk`.
- PL P2: Type omits ER percentage and Ki-67; findings omit a third 0.5 cm MRI lesion/NME and absence of chest-wall involvement; medication plan omits the intended start window.
- Attribution: A0 for second opinion; A1 for Patient type, Type, labs, findings, response, therapy, lab plan, cold-cap coordination, and genetic results; A2 for summary, goals description, medication plan, and procedure plan.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; Type BL; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- Main verification: confirmed the dated laboratory panel, mixed ductal/lobular pathology with receptor percentages and Ki-67, node-negative/PET-negative staging, `MP high risk`, explicit labs, cold-cap coordination, and separate echo/teaching/port preparations. BL is more complete only on Type; both systems miss MammaPrint in genetic results.

### Breast sample 13 — coral_idx 32

- Case: untreated right IDC, ER/PR 80–90%, HER2−, Ki-67 20–30%, grade not reported; right axillary FNA confirms regional disease; lung, abdomen/pelvis, and bone are negative. A 5 mm parafalcine lesion is strongly favored to be a meningioma, while an unlikely dural metastasis remains in the differential. Surgery is planned before the final adjuvant decision.
- PL P1: Distant is overconfidently `No` rather than retaining the low-probability dural uncertainty; Metastasis wrongly places the axillary regional node in its distant-uncertainty phrase and omits the actual dural question; medication plan omits likely postoperative chemotherapy and its pending-decision status; radiotherapy says none despite endocrine therapy being planned after radiation; MammaPrint is upgraded from `possibly` after surgery to a definite test plan.
- PL P2: Type omits receptor percentages and Ki-67; findings omit the explicitly negative lung, abdomen/pelvis, and bone staging.
- Attribution: A0 for second opinion and Stage; A1 for Patient type, Distant, Metastasis, labs, findings, goals/category description, and radiotherapy; A2 for Type, response, and the compound therapy plan.
- Core verdicts: current_meds TIE; Stage TIE; Distant BL; Metastasis BL; response TIE; Type BL; genetic results TIE. Total PL 0 / BL 3 / TIE 4.
- Main verification: confirmed the axillary FNA, otherwise negative PET staging, and MRI conclusion `most likely a meningioma, although dural-based metastasis remains an unlikely possibility`; also confirmed that chemotherapy and MammaPrint remain conditional on postoperative information.

### Breast sample 14 — coral_idx 33

- Case: untreated right grade-1 IDC, ER >95%, PR ~90%, HER2 IHC 2+ but FISH non-amplified and therefore HER2-negative, tumor about 1–2.2 cm with no metastatic evidence; Myriad is negative and MammaPrint low risk. Surgery is delayed; goserelin starts today and letrozole in about two weeks.
- PL P1: summary, Type, and findings preserve `HER2 equivocal` instead of resolving the negative FISH result; lab summary imports old RPR/HIV results from over six months earlier; `recent_changes` treats future letrozole as already started; genetic plan and Genetics referral resurrect a historical completed referral; Specialty invents a new surgical-oncology consultation rather than contact with the existing team.
- PL P2: medication plan includes a tangential psychiatric interaction discussion; bilateral mastectomy/reconstruction should retain its unresolved timing; future visit mode is inferred rather than stated.
- Attribution: A0 for second opinion, curative goal, and the explicitly `NOT_IN_NOTE` Specialty value; A1 for Patient type, Stage, Distant, stale labs, goals description, and genetic plan; A2 for in-person, summary, Type, Metastasis, findings, medication plan, and incomplete genetic-results support.
- Core verdicts: current_meds PL; Stage TIE; Distant TIE; Metastasis TIE; response PL; Type TIE; genetic results TIE. Total PL 2 / BL 0 / TIE 5.
- Main verification: confirmed final HER2 interpretation from IHC 2+ plus FISH non-amplified, same-day goserelin, future letrozole, completed Myriad/MammaPrint results, and absence of a new genetics or surgical-oncology referral.

### Breast sample 15 — coral_idx 34

- Case: untreated breast-origin adenocarcinoma confirmed only in a right supraclavicular node, ER >90%, PR 50%, HER2 IHC 2+ with FISH pending. Right axillary and bilateral/right cervical nodes remain imaging/clinical suspicions; cervical FNA and breast core biopsy are required before calling de novo MBC/Stage IV.
- PL P0: Metastasis falsely labels the right axillary node as confirmed, then also lists that same regional node in the distant-uncertainty clause.
- PL P1: summary prematurely states definite metastatic breast cancer; Stage says merely not staged and omits the explicit conditional de novo-MBC possibility; docusate is treated as oncology supportive medication without a treatment-toxicity link; palliative intent drops the biopsy-confirmation condition; goals description incorrectly says the intent is unstated; medication plan adds anti-HER2 therapy not stated in the note; a test order is mislabeled as a Genetics referral.
- PL P2: second-opinion presentation is weakly supported; Distant omits the left Vb suspicious node; findings contain excess normal examination detail; procedure wording duplicates core/US-guided breast biopsy and weakens `if possible`.
- Attribution: A0 for second opinion, Genetics referral, and genetic-results fallback; A1 for supportive medication, goals description, and response; A2 for Patient type, Type, broad findings, unconditional palliative goal, and the augmented medication plan.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis BL; response TIE; Type PL; genetic results TIE. Total PL 2 / BL 1 / TIE 4.
- Main verification: confirmed that only the right supraclavicular node is biopsy-proven, while breast/right axillary/cervical involvement remains presumptive and the clinician states `if we confirm ... then ... de novo MBC`. BL incorrectly declares Stage IV; PL is safer on Stage but unsafe in its detailed Metastasis value.

### Breast sample 16 — coral_idx 35

- Case: untreated left clinical Stage III grade 2–3 invasive lobular carcinoma, ER 96%, PR 35%, HER2−; PET/CT shows no distant disease. Left axillary/subpectoral nodes are radiographically suspicious, but axillary FNA is pending. The patient declines neoadjuvant endocrine trial participation and prefers surgical evaluation first.
- PL P0: Metastasis upgrades pending regional nodes to confirmed and fabricates `suspicious bone lesions pending biopsy`, which do not exist anywhere in the note.
- PL P1: findings omit the pending FNA status and benign feel of a supraclavicular/anterior-cervical node; medication plan contains only a surgical preference; therapy plan omits the proposed neoadjuvant endocrine approach and the patient's decision; mastectomy is presented as definite rather than a preference/referral for consideration; a completed axillary ultrasound is labeled as future imaging; Specialty misses the explicit return referral to the surgeon.
- PL P2: next visit purpose is right, but future mode is not explicit.
- Attribution: A0 for second opinion and medication plan; A1 for Distant, Metastasis, findings, goals description, response, and Specialty; A2 for in-person, Type, goals, therapy, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; Type TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 6.
- Main verification: confirmed `no evidence of distant metastatic disease`, pending axillary FNA, no bone lesion, proposed neoadjuvant endocrine therapy, patient preference for upfront surgery, and explicit referral back for surgical consideration.

### Breast sample 17 — coral_idx 36

- Case: left Stage IIB T2N1M0 grade-2 IDC, ER >95%, PR 25%, HER2−, Ki-67 ~30%, with LVI, 2/2 positive axillary nodes, 1.8 cm nodal deposit/extracapsular extension, multifocal residual disease, and a small positive deep margin after re-excision; no systemic therapy yet; TC×6, ONPRO, port, PET/CT, baseline echo, teaching, and later radiation are planned; BRCA reported negative.
- PL P0: Metastasis again fabricates `suspicious bone lesions pending biopsy` despite explicit M0 and no bone abnormality in the note.
- PL P1: findings omit LVI, positive-node burden, nodal-deposit size, and extracapsular extension; therapy plan misroutes teaching and port and omits radiation/start date; Referral.Others misses explicit chemotherapy-teaching referral; Referral follow-up is a list of treatment/pathology/cosmetic tasks rather than a visit; genetic results incorrectly mixes standard HER2 FISH pathology into the molecular/genetic field.
- PL P2: Type omits site, receptor percentages, LVI, and multifocal residual disease; radiation is made definite despite `likely/best approach` wording; possible plastic-surgery discussion is omitted.
- Attribution: A0 for second opinion, Distant, Metastasis, next visit, and genetic results; A1 for goals description and response; A2 for summary, Type, Stage, findings, goals, and radiotherapy.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; Type BL; genetic results BL. Total PL 0 / BL 3 / TIE 4.
- Main verification: confirmed T2N1M0, 2/2 positive nodes with extracapsular extension, no bone lesion, TC×6 plan, explicit chemotherapy-teaching referral, and BRCA-negative report. The fabricated bone phrase is identical to sample 16's failure pattern.

### Breast sample 18 — coral_idx 37

- Case: untreated left grade-2 IDC with ER >95%, pathologically low-positive PR <5%, HER2−, Ki-67 44%, clinical cT2NX; axillary imaging is suspicious but two FNAs are benign and final nodal status awaits surgery. Mastectomy, node-dependent chemotherapy, and at least five years of endocrine therapy are planned. Germline ATM mutation and high-risk MammaPrint are complete.
- PL P1: Type converts the exact low-positive PR result to PR− and omits Ki-67; Distant says `No` without explicit distant staging or M0 evidence; medication plan omits the clearly recommended node-dependent TC versus AC-T chemotherapy; radiotherapy says none rather than retaining the explicit decision to avoid radiation because of ATM; next visit and Referral follow-up omit the planned postoperative treatment discussion.
- PL P2: Metastasis preserves nodal uncertainty but should state suspicious imaging, two benign FNAs, and pending surgical determination.
- Attribution: A0 for Distant, response, and next visit; A1 for Patient type, second opinion, findings, and goals description; A2 for summary, Type, Metastasis, goals, medication plan, therapy plan, follow-up, and incomplete genetic-results support.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis PL; response TIE; Type BL; genetic results PL. Total PL 3 / BL 1 / TIE 3.
- Main verification: confirmed exact `PR positive (<5%)`, cT2NX, two benign axillary FNAs despite suspicious imaging, ATM mutation, MammaPrint high risk, node-dependent chemotherapy choice, and radiation-oncology recommendation for mastectomy rather than radiation.

### Breast sample 19 — coral_idx 38

- Case: two distinct left-breast IDC foci (grade 2 ER 61–70%/PR−/HER2−/Ki-67 15–20%, and grade 3 ER 41–50%/PR 1–10%/HER2−/Ki-67 30–40% with focal LCIS), original clinical Stage 2–3 and low-risk MammaPrint; after neoadjuvant endocrine therapy, bilateral mastectomies, and adjuvant tamoxifen, she is currently on exemestane plus monthly goserelin and is NED on exam.
- PL P0: Type fabricates `extensive DCIS` and collapses two biologically distinct foci into one profile; the source has only imaging concern for DCIS and pathology showing focal LCIS.
- PL P1: findings treats a future possible BSO as a current finding; PRN ondansetron without a current oncology/toxicity link is labeled supportive; medication plan mixes estradiol testing, DEXA, psychiatry follow-up, and inactive ondansetron; therapy plan includes estradiol testing; procedure plan resurrects a 2015 cyst-aspiration contingency and misses current possible BSO; Specialty misses explicit local psychiatry follow-up; genetic results correctly includes MammaPrint but is polluted by routine ER/PR/HER2/Ki-67 pathology and LCIS.
- PL P2: goals description leaves explicit adjuvant/risk-reduction context blank.
- Attribution: A0 for second opinion and Distant; A1 for Stage, incomplete current-medication support, goals description, medication plan, Specialty, and genetic results; A2 for Type, findings, goals, and therapy.
- Core verdicts: current_meds PL; Stage TIE; Distant TIE; Metastasis TIE; response PL; Type BL; genetic results PL. Total PL 3 / BL 1 / TIE 3.
- Main verification: confirmed monthly goserelin plus exemestane, `NED on exam`, low-risk MammaPrint, two different tumor profiles, focal LCIS rather than confirmed DCIS, current consideration of BSO, and that the cyst-aspiration language came from a historical embedded `Last Assessment & Plan`.

### Breast sample 20 — coral_idx 39

- Case: large bilateral breast cancers, right ER+/PR+/HER2+ with Ki-67 40%, left ER+/PR+/HER2 0 with some lobular differentiation; current exam measures about 12 × 9 cm right and 10 × 6 cm left. Reported lung nodules and tiny liver lesions are unbiopsied and explicitly unconfirmed. No treatment has started; systemic and surgical decisions remain conditional on pathology/FISH/staging; germline panel is pending.
- PL P1: summary and Type wrongly convert the left HER2 0 tumor to HER2+ and conflict with PL's own summary details; Stage adds unsupported `locally advanced`; findings uses older 7.3/6 cm measurements and omits the much larger current exam; response says merely not mentioned despite clear untreated status; therapy drops the stated contingencies and mixes procedure/imaging/lab/visit logistics; procedure output is an incomplete clause and misses conditional surgery/port; lab plan copies multiple non-lab items; genetic plan misroutes HER2 FISH and misses the pending germline panel.
- PL P2: Metastasis omits unclear nodal involvement; hidden/redacted lab values should be acknowledged rather than called absent; goal should retain pending-staging qualification; medication plan overstates a still-conditional regimen; radiation timing is not fully specified; next-visit purpose is inferred.
- Attribution: A0 for Stage and genetic plan; A1 for Patient type, second opinion, response, goals-description fallback, and genetic-results fallback; A2 for summary, Type, Distant, Metastasis, findings, medication/therapy plans, lab plan, and next visit.
- Core verdicts: current_meds TIE; Stage BL; Distant PL; Metastasis TIE; response BL; Type BL; genetic results TIE. Total PL 1 / BL 3 / TIE 3.
- Main verification: confirmed separate laterality-specific HER2 status, current versus historical sizes, unconfirmed lung/liver findings, no started treatment, explicit dependence on final pathology/FISH/staging, and pending germline panel. The PL's left-HER2 error is a cross-field consistency failure because its own surrounding output contains the correct HER2 0 fact.

## Breast interim summary

- Core verdicts across 20 samples: PL 28 / BL 28 / TIE 84. The matched v2.1 breast comparison is therefore tied, not a demonstrated PL win.
- Repeated high-impact patterns: unsupported bone-metastasis template completion; confirmed/suspected and regional/distant node mixing; conditional Stage IV or palliative language becoming definite; bilateral/multifocal receptor profiles collapsing across lesions; prior-treatment recurrence being replaced by an untreated fallback; completed MammaPrint omitted or standard HER2 pathology misrouted into genetic results.
- Repeated non-core patterns: historical embedded A/P contaminating current plans; drug/procedure/imaging/lab/referral content crossing field boundaries; planned treatment labeled as recent change; proposed or conditional actions upgraded to definite plans.
