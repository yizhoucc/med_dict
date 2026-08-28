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

- Completed: 40/40 (breast 20/20, PDAC 20/20)
- Next: targeted core-field repairs, PL rerun, and regression review
- PL findings: P0=22, P1=226, P2=123
- Attribution findings: A0=103, A1=190, A2=256
- Core verdict totals (PL / BL / TIE): 53 / 53 / 154

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

### PDAC sample 1 — coral_idx 0

- Case: locally advanced pancreatic adenocarcinoma after six cycles of gemcitabine/nab-paclitaxel, now taking a chemotherapy break under surveillance. The primary is stable to minimally larger; abdomen/pelvis has no metastasis, while five new lung nodules are indeterminate but suspicious and the physician calls this possible progression. No regional nodes or completed genetic result are established.
- PL P0: Metastasis fabricates `confirmed regional nodes` and a `pending biopsy`; the only abdominal nodes are small and not enlarged, and no lung biopsy is planned.
- PL P1: recent changes mixes the current chemotherapy break with an older biliary stent episode/cycle interruption; goals description omits the explicit reason for surveillance; response omits the new suspicious lung lesions and clinician's possible-progression judgment; psycho-oncology is listed as a referral although the note records only the patient's wish to meet the service.
- PL P2: lab values omit some abnormal flags; medication plan should name the held regimen; therapy plan could explicitly state chemotherapy break/surveillance.
- Attribution: A0 for second opinion and genetic-results fallback; A1 for Metastasis, labs, findings, recent changes, supportive medication, goals description, and Specialty; A2 for surveillance goal and response.
- Core verdicts: current_meds TIE; Stage TIE; Distant PL; Metastasis BL; response TIE; genetic results TIE. Total PL 1 / BL 1 / TIE 4.
- Main verification: confirmed no positive regional nodes or biopsy plan, `5 new nodules ... indeterminate but ... suspicious`, `possible progression`, maximal benefit, and the explicit decision for surveillance/chemotherapy break.

### PDAC sample 2 — coral_idx 1

- Case: resected pancreatic adenocarcinoma with duodenal invasion and 6/25 regional nodes, followed by confirmed liver metastases; progressed on FOLFIRINOX and is now at cycle 2 day 1 of second-line gemcitabine/nab-paclitaxel. Current-regimen response is not yet established. SPINK1 carrier with FANCG and NF2 VUS.
- PL P1: findings copies extensive normal examination/labs while omitting the essential FOLFIRINOX progression, liver metastasis, and early unassessed status of current therapy; family SPINK1 screening is placed in the patient's genetic-testing plan; the current GI Oncology clinic is mislabeled as an outgoing Specialty referral.
- PL P2: `ductal` is more specific than the source; Stage's `Originally unspecified` is unnecessary; older albumin/calcium are undated in the lab summary; supportive medications omit several active antiemetic/hydration/cannabis measures; response should explicitly separate prior progression from currently unassessed Gem/Abrax; medication plan misses some supportive measures; future visit mode is inferred.
- Attribution: A0 for second opinion, findings, genetic plan, and Specialty; A1 for goals description; A2 for summary, Stage, Distant, Metastasis, labs, response, medication plan, and incomplete molecular-result support.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis PL; response TIE; genetic results TIE. Total PL 2 / BL 0 / TIE 4.
- Main verification: confirmed 6/25 regional nodes, liver metastases, prior FOLFIRINOX progression, current C2D1 Gem/Abrax without response imaging, marker trajectory, and all three molecular findings.

### PDAC sample 3 — coral_idx 2

- Case: metastatic PDAC after FOLFIRINOX, FOLFOX, and gemcitabine/nab-paclitaxel, currently third-line 5-FU/LV plus nanoliposomal irinotecan. Confirmed distant disease includes liver and peritoneum; splenic involvement is confounded by direct invasion from the tail primary. February imaging was mixed/stable, followed by treatment interruption, clinical decline, and resumption of cycle 5. Tumor KRAS G12D and p53 mutations are documented.
- PL P0: Metastasis fabricates `confirmed regional nodes`; the note contains no positive nodal pathology and latest imaging says `Lymphadenopathy: Absent`.
- PL P1: findings prioritizes older February imaging and misses the larger March mass, direct stomach/spleen invasion, gastric outlet obstruction, and ascites; procedure output is a stray radiotherapy-purpose clause rather than a procedure; lab plan is a chemotherapy sentence rather than the required pre-cycle laboratory condition; next visit turns a CT into an in-person appointment and contains `f&u`; Advance Care incorrectly equates possible future hospice refocusing with a documented directive/code-status discussion; Referral follow-up repeats only the CT plan.
- PL P2: current medication has a truncated parenthesis; supportive medications omit lorazepam and possibly pantoprazole; goals description attribution is incomplete; response includes likely infectious lung findings; medication/therapy attribution does not cover exploratory salvage options.
- Attribution: A0 for second opinion, procedure, next visit, Advance Care, and Referral follow-up; A1 for Patient type, findings, and recent changes; A2 for Stage, Distant, Metastasis, labs, goals description, response, medication plan, therapy plan, and lab plan.
- Core verdicts: current_meds TIE; Stage PL; Distant PL; Metastasis BL; response PL; genetic results TIE. Total PL 3 / BL 1 / TIE 2.
- Main verification: confirmed liver/peritoneal disease, lack of regional-node evidence, direct-versus-metastatic splenic ambiguity, mixed February response, subsequent clinical decline, current regimen, short-term CT plan, and KRAS/p53 results.

## Root-cause note: repeated metastasis template hallucination

- The literal example `Yes — confirmed ipsilateral axillary node; distant disease uncertain — suspicious bone lesions pending biopsy` appears in `prompts/extraction.yaml` and is reproduced verbatim in breast samples 6, 16, and 17; the same `confirmed regional nodes` template leaks into PDAC samples 1 and 3.
- G4's whole-value `when in doubt, KEEP` policy preserves the supported half of a compound value together with its unsupported half, while the final reconciliation hook treats mixed regional/distant text as already populated.
- Minimal repair: replace concrete site examples with abstract schema rules, make G4 prune unsupported clauses rather than only whole values, and add a final source-grounded sanitizer that removes any M1 clause when `Distant=No` and no current explicit or uncertain M1 evidence exists, while retaining supported regional disease.

### PDAC sample 4 — coral_idx 3

- Case: pancreatic adenocarcinoma initially considered locally advanced; a segment-7 liver lesion was historically called suspicious/consistent with metastasis by outside review, but was PET non-avid and current CT identifies hemangiomas with no suspicious lesions. The clinician still labels the disease metastatic. After FOLFIRINOX and six cycles of gemcitabine/nab-paclitaxel, she is on surveillance with stable disease; CA19-9 is non-expressed.
- PL P0: Metastasis fabricates `confirmed regional nodes`; porta-hepatis lymphadenopathy is not confirmed malignant, and the liver certainty is also overstated.
- PL P1: Distant states definite liver metastasis despite longitudinally conflicting/now-negative imaging; findings omit key unresectability evidence (SMA stranding, common-hepatic-artery encasement, severe portal-vein/SMV narrowing); recent changes misses completion of six cycles and transition to surveillance; goals description omits the explicit good-disease-control surveillance statement.
- PL P2: supportive medications omit other cancer-care GI agents; response lacks the objective stable-mass/current-negative evidence; medication/therapy plan should explicitly describe ongoing chemotherapy holiday/surveillance.
- Attribution: A0 for second opinion; A1 for Metastasis, findings, supportive medications, and genetic result; A2 for Stage and organ-specific Distant.
- Core verdicts: current_meds TIE; Stage PL; Distant BL; Metastasis BL; response BL; genetic results PL. Total PL 2 / BL 3 / TIE 1.
- Main verification: confirmed historical liver uncertainty, current CT `No evidence of metastasis`, persistent clinician label `metastatic`, no confirmed regional nodes, stable primary, surveillance plan, and explicit CA19-9 non-expression.

### PDAC sample 5 — coral_idx 4

- Case: Stage IV pancreatic adenocarcinoma with biopsy-proven abdominal-wall oligometastasis, stable pancreas and abdominal-wall disease after 12 cycles of FOLFIRINOX, now on chemotherapy break; liquid biopsy shows MSI undetermined and RB1 P26fs*47.
- PL P1: Type omits the biopsy-confirmed abdominal-wall oligometastatic context; recent changes selects an old cycle-10 thrombocytopenia hold rather than the final completed-12-cycles→break transition; supportive medication includes ondansetron marked not taking and omits active opioids/fluconazole; therapy contains Creon/opioids rather than the chemotherapy-break state; existing nutrition and SMS care are mislabeled as new referrals.
- PL P2: findings is overlong and mixes treatment/labs into disease findings; next-visit mode is inferred.
- Attribution: A0 for second opinion, therapy, and Specialty; A1 for labs, recent changes, and goals description; A2 for summary, findings, supportive medication, response, medication plan, and Advance Care.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 0 / TIE 6.
- Main verification: confirmed abdominal-wall biopsy, 12 completed FOLFIRINOX cycles, current chemotherapy break, stable disease, active versus not-taking supportive drugs, complete ACP/POLST content, and liquid-biopsy findings.

### PDAC sample 6 — coral_idx 5

- Case: initially borderline-resectable/locally advanced pancreatic adenocarcinoma treated with 12 cycles modified FOLFIRINOX and resection. Pathology showed 1.7 cm residual tumor with 51–90% destruction, negative margins, and no positive nodes. Postoperative CT has new/enlarging tiny liver lesions suggestive of recurrence, but they are too small for confirmation; repeat scans and possible biopsy are planned. CA19-9 rose from 44 preoperatively to 2250.
- PL P0: Metastasis fabricates confirmed regional nodes and confirmed liver metastasis, contradicting both node-negative pathology and the explicitly unconfirmed tiny liver lesions.
- PL P1: response omits the substantial post-neoadjuvant pathologic treatment effect; procedure plan mixes the repeat scan with conditional biopsy; next visit turns imaging timing into a definite in-person visit.
- PL P2: Stage omits the original borderline/locally advanced state; Distant omits the suspected liver site; lab output violates the string schema, contains an ALT formatting error, and misses the CA19-9 rise.
- Attribution: A0 for second opinion and next visit; A1 for Distant, labs, findings, goals description, and genetic-results fallback; A2 for summary, Stage, Metastasis, surveillance goal, response, and procedure plan.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis BL; response PL; genetic results TIE. Total PL 2 / BL 1 / TIE 3.
- Main verification: confirmed node-negative surgical pathology, unconfirmed suspicious liver lesions, short-interval imaging/conditional biopsy, 51–90% treatment effect, and no active anticancer medication.

### PDAC sample 7 — coral_idx 6

- Case: pancreatic-tail adenocarcinoma with local extension/abutment and no distant metastasis, receiving alternate-week gemcitabine/nab-paclitaxel after neutropenia; after four cycles the primary is slightly smaller and the clinician documents radiographic response. A right adnexal cystic mass is likely benign. Germline testing is negative with four VUS.
- PL P1: Stage upgrades `local extension` to definite unresectable disease despite possible surgery; findings omits the `likely benign` qualification for the adnexal mass; current meds is empty despite active Gem/Abrax; medication and therapy plans omit the drug names and alternate-week schedule; next visit invents an eight-week telehealth clinic appointment from an imaging interval.
- Attribution: A0 for second opinion; A1 for Patient type, unsupported unresectable Stage, labs, recent changes, genetics, and goals; A2 for summary, Distant, Metastasis, broad findings, response, and next visit.
- Core verdicts: current_meds BL; Stage BL; Distant TIE; Metastasis TIE; response TIE; genetic results TIE. Total PL 0 / BL 2 / TIE 4.
- Main verification: confirmed active alternate-week gemcitabine/nab-paclitaxel, slight shrinkage and explicit response, absence of distant disease, likely-benign adnexal mass, and no definite unresectability statement.

### PDAC sample 8 — coral_idx 7

- Case: resected grade-3 pancreatic ductal adenocarcinoma with LVI, pT2N2 and 11/37 regional nodes, later biopsy-confirmed gastrohepatic/mesenteric nodal metastatic recurrence; currently after three cycles of gemcitabine/nab-paclitaxel with shrinking nodes, CA19-9 746→61, and explicit favorable response. Germline ATM mutation and intact MMR are known; FoundationOne is pending, with no actionable finding communicated so far.
- PL P1: Type omits recurrent-metastatic status; Stage gives only historical pT2N2 and misses current Stage IV recurrence; Distant is blank despite confirmed non-regional nodal recurrence; current meds is blank; recent changes omits initiation of current Gem/Abrax; response falsely says not mentioned; current incoming consultation is mislabeled as an outgoing Specialty referral; Referral follow-up contains a treatment recommendation instead of follow-up availability.
- PL P2: Metastasis needs explicit historical/current timing; findings has a malformed biopsy date and excessive normal exam detail; therapy's future-trial list is incomplete; next-visit mode is inferred; genetic results omits the communicated no-actionable-mutation status.
- Attribution: A0 for second opinion, Specialty, and follow-up; A1 for Patient type, Type, labs, supportive medication, and response; A2 for Stage, Metastasis, findings, both goals fields, therapy, and incomplete genetic-result support.
- Core verdicts: current_meds BL; Stage BL; Distant TIE; Metastasis PL; response BL; genetic results PL. Total PL 2 / BL 3 / TIE 1.
- Main verification: confirmed active current regimen, historical pT2N2, biopsy-proven metastatic nodal recurrence, objective radiographic/biochemical favorable response, germline ATM/MMR status, and still-pending formal FoundationOne report.

### PDAC sample 9 — coral_idx 8

- Case: lung-predominant biopsy-confirmed metastatic pancreatic cancer, currently on gemcitabine/nab-paclitaxel with same-day Abraxane dose reduction for neuropathy. CT shows stable treated lung metastases and a smaller pancreatic primary; CA19-9 fell 3525→2762→1109. BRCA VUS and KRAS/CDKN2A/APC mutations are documented.
- PL P1: findings reverses the CA19-9 trend and says it increased while copying excessive normal exam detail; supportive medication omits several active pain/nausea/diarrhea drugs and misspells olanzapine; medication plan treats the existing gastric-outlet stent as a drug action and fails to state continuation of gemcitabine plus reduced-dose Abraxane; historical Phase I consultation is mislabeled as Genetics referral.
- PL P2: summary overemphasizes currently stable nausea/vomiting; recent changes includes a follow-up action; therapy mixes response wording and omits explicit gemcitabine continuation; imaging date wording is unclear; next-visit mode is inferred.
- Attribution: A0 for second opinion and findings; A1 for labs, supportive medication, goals description, Genetics referral, and genetic results; A2 for summary, Distant, Metastasis, goals, response, and medication plan.
- Core verdicts: current_meds TIE; Stage PL; Distant TIE; Metastasis TIE; response PL; genetic results TIE. Total PL 2 / BL 0 / TIE 4.
- Main verification: confirmed biopsy-proven lung progression, active Gem/Abrax, current dose reduction, stable treated lung disease, shrinking pancreatic primary, downward marker trend, and all reported variants.

### PDAC sample 10 — coral_idx 9

- Case: locally advanced unresectable pancreatic ductal adenocarcinoma initially treated with mFOLFIRINOX. Irinotecan was permanently omitted from cycle 3 because of colitis and poor tolerance; the active cycle-6 regimen is FOLFOX only. The December scan predates the switch and shows stable local disease, no definite distant spread, and an indeterminate right-upper-lobe nodule. RECQL4 has a VUS.
- PL P0: general Metastasis fabricates `confirmed regional nodes` and a `suspicious liver lesion pending biopsy`; current medications incorrectly retains FOLFIRINOX alongside active FOLFOX, falsely implying ongoing irinotecan.
- PL P1: summary likewise mixes the old and active regimens; findings omits the unresolved lung-nodule differential; supportive medications omit several explicitly active electrolyte, antiemetic, steroid, anxiolytic, and hydration measures; palliative intent conflicts with induction/downstaging treatment and possible surgery or consolidative chemoradiation; response attributes a pre-switch scan to the current FOLFOX period; therapy mixes potassium and scans into systemic therapy; imaging mixes CA19-9 and a cancelled mammogram into the scan plan; lab plan includes imaging; next-visit mode is invented; a conditional future surgical reassessment is upgraded to a current Specialty referral.
- PL P2: lab summary mixes older albumin/protein values into the current panel without dates.
- Attribution: A0 for Patient type and second opinion; A1 for Televisit, Metastasis, findings, and goals; A2 for summary, Stage, labs, current/supportive medications, response, medication/therapy plans, imaging, next visit, and Specialty.
- Core verdicts: current_meds BL; Stage TIE; Distant TIE; Metastasis BL; response TIE; genetic results TIE. Total PL 0 / BL 2 / TIE 4.
- Main verification: confirmed `Omitted irinotecan since C3`, `Will continue with FOLFOX only going forward`, no liver lesion or biopsy plan, no proven regional node, the unresolved lung differential, the scan-before-regimen-change timing, and the possible post-induction surgery/chemoradiation pathways.

### PDAC sample 11 — coral_idx 10

- Case: new second-opinion telehealth consultation for clinically staged IV pancreatic cancer, cT2 cN1 cM1, documented as metastatic to liver. No anticancer treatment has begun. The patient consented only to Precision Promise screening; FOLFIRINOX and gemcitabine/nab-paclitaxel are treatment options. STRATA is pending, and the BRCA2 mutation belongs to the patient's brother.
- PL P1: Metastasis upgrades clinical cN1 to `confirmed regional nodes`; medication plan blurs trial screening with a selected regimen and mechanically appends alternative drugs; therapy says the patient consented to trial participation rather than screening; imaging and lab fields copy the combined CT-plus-labs sentence without separating modalities; an existing genetics appointment and the incoming UCSF referral are mislabeled as outgoing referrals; follow-up is populated with same-day tests rather than a return visit; genetic results assigns the brother's BRCA2 result and pending STRATA to the patient.
- Attribution: A0 for second opinion, Specialty, and follow-up; A1 for Patient type, supportive medications, response, genetic plan/results, Genetics referral, and next-visit fallback; A2 for summary, Metastasis, findings, goals description, and medication/therapy plans.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; genetic results BL. Total PL 0 / BL 2 / TIE 4.
- Main verification: confirmed the explicit Stage IV/cT2/cN1/cM1 and liver-metastasis labels, absence of started treatment, `consent to proceed with screening procedures`, parallel standard-of-care options, `Strata pending`, and that the BRCA2 carrier is the brother rather than the patient.

### PDAC sample 12 — coral_idx 11

- Case: pancreatic adenocarcinoma initially staged by EUS as at least T4/cN1, after 12 cycles of mFOLFIRINOX and a treatment holiday. The latest course shows clear progression with enlarging/infiltrative primary disease, bowel and biliary obstruction, and peritoneal/omental carcinomatosis. Liver findings remain hard to interpret and bilateral adrenal nodules remain suspicious. No anticancer drug is active; chemotherapy is considered more harmful than beneficial unless the patient's condition improves substantially, and hospice is anticipated.
- PL P0: Distant Metastasis treats the uncertain liver lesions as confirmed while omitting suspicious adrenal disease; general Metastasis also upgrades clinical N1 to `confirmed regional nodes` and the uncertain liver lesions to confirmed metastasis.
- PL P1: Lab Results is a truncated JSON string rather than the required object; supportive medications omit active symptom-control drugs; goals description is blank despite explicit palliative/hospice language; response reports only old stable disease and omits the subsequent, current progression; medication and therapy plans omit the explicit decision against chemotherapy and the narrow condition for possible resumption; procedure plan misses conditional additional stenting; lab plan mixes oral intake and obstruction with blood tests; Advance Care incorrectly maps hospice discussion into a field restricted to directive/proxy/code-status/living-will content; Specialty turns palliative-care discussion into a referral.
- PL P2: summary understates definite current peritoneal carcinomatosis as generalized concern; recent changes includes a current treatment judgment; imaging reports follow-up of an already completed same-day X-ray as future imaging.
- Attribution: A0 for second opinion and Specialty; A1 for Lab Results, findings, supportive medications, and genetic-results fallback; A2 for summary, Stage, both metastasis fields, recent changes, response, and Referral follow-up.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis TIE; response BL; genetic results TIE. Total PL 0 / BL 1 / TIE 5.
- Main verification: confirmed definite peritoneal/omental carcinomatosis, unresolved liver findings, clinical rather than pathologically confirmed N1, suspicious adrenal nodules, explicit progression during the chemotherapy holiday, the current no-chemotherapy judgment, conditional stent consideration, and lack of a formal advance-directive/code-status entry.

### PDAC sample 13 — coral_idx 12

- Case: locally advanced, unresectable pancreatic head/uncinate adenocarcinoma with direct duodenal invasion and no established distant metastasis. Six cycles of gemcitabine/capecitabine and December radiation are complete; systemic therapy is now paused while radiation is reconsidered. The January CT shows stable local disease. The patient is a CA19-9 non-secretor.
- PL P1: Metastasis upgrades radiographic mesenteric/peripancreatic nodal involvement to `confirmed regional nodes`; findings imports an older increase in pneumobilia despite the latest scan reporting decreased pneumobilia; recent changes omits both the systemic-therapy pause and recently completed radiation; response mixes nonmalignant organizing-pneumonia improvement into cancer response; medication plan omits the systemic-therapy pause and Creon; procedure misses the planned February bronchoscopy; imaging gives an underspecified CT plan whose citation is actually about XRT; lab plan mixes PRBC transfusion into a blood-test-only field; genetic plan carries forward an old `UCSF500 in process` state despite the current `Genomics Not done`; next visit treats another specialist's February appointment as the oncology return; Specialty lists historical consultations rather than a new referral.
- PL P2: Type omits head/uncinate location and direct duodenal invasion; supportive medications omit Norco and other active symptom/nutrition support; radiotherapy includes the systemic-therapy pause in the radiation-only field.
- Attribution: A0 for Patient type, second opinion, Imaging, and Specialty; A1 for Metastasis; A2 for Stage, Distant, findings, supportive medications, goals, response, medication plan, next visit, and genetic plan.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis BL; response BL; genetic results PL. Total PL 3 / BL 2 / TIE 1.
- Main verification: confirmed active systemic therapy is paused despite stale capecitabine in the medication list, latest stable pancreatic mass, no definite distant disease, only radiographic nodal involvement, completed radiation, planned bronchoscopy, current `Genomics Not done`, and explicit CA19-9 non-expression. The nodal overstatement is calibrated as P1 rather than P0 because the note does describe imaging-based nodal involvement, but not pathologic confirmation.

### PDAC sample 14 — coral_idx 13

- Case: new telehealth consultation for Stage IV pancreatic-origin adenocarcinoma with biopsy-supported liver metastasis, radiographically metastatic upper-abdominal/mesenteric nodes, malignant biliary obstruction, and a newly placed common-bile-duct stent. No treatment has started; gemcitabine/nab-paclitaxel is favored pending in-person reassessment, and UCSF500 is planned.
- PL P1: procedure plan misroutes UCSF500 into procedures and promotes merely conditional trial-biopsy requirements into current plans; Specialty mislabels the incoming UCSF consultation as an outgoing referral.
- PL P2: Metastasis should name the upper-abdominal/mesenteric nodal sites and their imaging basis; medication plan says gemcitabine/nab-paclitaxel is planned rather than favored pending reassessment.
- Attribution: A0 for second opinion, procedure, and Specialty; A1 for Patient type, lab summary, supportive medication, and response; A2 for Distant, Metastasis, findings, goals description, medication/therapy plans, and genetic-results fallback.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response TIE; genetic results TIE. Total PL 1 / BL 0 / TIE 5.
- Main verification: confirmed liver-biopsy adenocarcinoma, radiology's explicit `hepatic and nodal metastases`, Stage IV status, absence of started treatment, conditional preference for Gem/Abrax, planned UCSF500, and absence of a new outgoing specialty referral.

### PDAC sample 15 — coral_idx 14

- Case: resected pancreatic ductal adenocarcinoma after neoadjuvant FOLFIRINOX and Whipple. Final pathology is a 4.6 cm moderately differentiated pancreatic-head tumor with PNI, positive margins, 11/46 regional nodes, ypT3N2, and poor/no treatment effect. Current imaging has no distant metastasis, but CA19-9 rose from 48 to 4,375; repeat marker and CT are planned. No anticancer drug is active. MMR proteins are intact, and benign ascites fluid was KRAS-negative.
- PL P0: Type says `poorly differentiated`, directly contradicting the final pathology's `moderately-differentiated` grade.
- PL P1: Metastasis fails to label the positive regional nodes as historical/resected and uses contradictory `distant disease uncertain — no evidence` wording; recent changes misroutes a rivaroxaban hold as anticancer treatment; response misses explicit Evans grade I/Ryan score 3 poor-or-no pathologic response and falsely says no marker comparison exists; therapy lists Creon as anticancer therapy; next visit turns test timing into a definite in-person visit; Specialty repeats a historical radiation/systemic-therapy consultation and misses the current PMD referral; genetic results omits intact MLH1/PMS2/MSH2/MSH6 expression.
- Attribution: A0 for second opinion, Stage, Distant, therapy, and next visit; A1 for labs, findings, recent changes, goals-description fallback, response, Specialty, and genetic results; A2 for summary, Type, Metastasis, goals, and Referral follow-up.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis PL; response BL; genetic results BL. Total PL 1 / BL 2 / TIE 3.
- Main verification: confirmed the final moderate grade, ypT3N2 and 11/46 positive nodes, no established distant spread, poor/no neoadjuvant pathologic response, sharply rising CA19-9, current absence of anticancer medication, intact MMR proteins, KRAS-negative benign ascites, and explicit conditional PMD referral.

### PDAC sample 16 — coral_idx 15

- Case: clinical Stage IIB cT1c cN1 cM0 pancreatic-head adenocarcinoma on dose-reduced gemcitabine monotherapy. One dose was delayed for a possible dental abscess; the patient is now cleared and the schedule will change to alternate weeks. The May CT shows the pancreatic mass no longer visible, improved duct dilation, and no distant metastasis. Separate renal and lung lesions are favored to be other primaries rather than pancreatic metastases.
- PL P0: Metastasis upgrades clinical/radiographic regional-node concern to `confirmed regional nodes` and fabricates a suspicious liver lesion with pending biopsy despite cM0 and explicit absence of distant disease.
- PL P1: next visit invents an in-person mode; Specialty turns a completed dentist evaluation into a current referral.
- PL P2: summary omits the C2D15 decision and schedule change; Type omits pancreatic-head location; findings adds `no palpable masses` without a matching source statement; recent changes misses the one-dose dental delay; supportive medications misses recommended daily senna; imaging omits that the next scan is due after this cycle.
- Attribution: A0 for second opinion, Metastasis, and Specialty; A1 for Stage and labs; A2 for summary, findings, current/supportive medications, goals description, medication plan, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 5.
- Main verification: confirmed cN1 is clinical rather than pathologic, PET described regional nodes only as concerning, the current scan reports no metastatic disease and contains no liver-lesion/biopsy plan, gemcitabine is active, and the response/schedule/referral timing stated above.

### PDAC sample 17 — coral_idx 16

- Case: locally advanced pMMR pancreatic body/tail adenocarcinoma with extensive local vascular involvement but no distant spread. After eight FOLFIRINOX cycles the patient entered a chemotherapy break in November 2018; February 2019 CT remained stable. No anticancer drug is active, with June follow-up and July CT CAP planned. Foundation testing reports MSS, TMB 5, KRAS G12V, TP53 I195F, and listed VUS/alterations.
- PL P0: Metastasis fabricates both confirmed regional nodes and a suspicious liver lesion pending biopsy; neither is present anywhere in the note.
- PL P1: lab summary omits CA19-9 289 while retaining an older CEA without its date; therapy says `None` rather than continuing the chemotherapy-break state.
- PL P2: Type omits pMMR; findings mixes laboratory results into the objective-imaging/exam field; palliative intent is reasonable but not explicit; response correctly uses the latest stable CT but appends isolated labs without dates or a supported trend.
- Attribution: A0 for second opinion, in-person, and findings; A1 for Distant, Metastasis, labs, goals, goals-description fallback, and genetic results; A2 for Patient type, Stage, response, and next visit.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response PL; genetic results TIE. Total PL 1 / BL 1 / TIE 4.
- Main verification: confirmed no node or liver-metastasis evidence, the chemotherapy break, the latest February stable scan, no active anticancer drug, and the molecular results. Although `unresectable` is not quoted verbatim, it is accepted here as consistent with the locally advanced label plus SMA/SMV-portal-confluence involvement; Stage is therefore a tie rather than a BL win.

### PDAC sample 18 — coral_idx 17

- Case: initially resectable pancreatic-tail adenocarcinoma after only two poorly tolerated neoadjuvant Gem/Abraxane doses, distal pancreatectomy/splenectomy, and two postoperative Gem/Cape cycles. Pathology shows moderate differentiation, negative margins, and 2/29 positive regional nodes. Chemotherapy is now held for severe hand-foot syndrome/mucositis and is to resume five days of seven after recovery. Current scans show no distant disease; capecitabine is explicitly not being taken. ATM has a VUS.
- PL P1: Stage is blank and loses the initially resectable, now resected node-positive status; Metastasis correctly captures 2/29 positive nodes but omits their historical/resected context and contradicts Distant=`No` with `distant disease uncertain`; findings copies many laboratory values and omits the decisive surgical pathology; medication plan mixes future Doppler imaging into medications and does not name the held Gem/Cape regimen; therapy omits the planned five-days-on/two-days-off resumption and regimen name; Referral follow-up merely repeats the treatment hold/resumption rather than a return instruction.
- PL P2: response reasonably says there is no specific treatment response to assess, but omits the current no-evidence-of-metastatic-disease scan context.
- Attribution: A0 for Patient type, Distant, and Referral follow-up; A1 for second opinion, labs, findings, goals, goals-description fallback, response, next-visit fallback, and genetic results; A2 for summary, Type, Metastasis, supportive medications, and medication plan.
- Core verdicts: current_meds PL; Stage BL; Distant TIE; Metastasis PL; response TIE; genetic results TIE. Total PL 2 / BL 1 / TIE 3.
- Main verification: confirmed capecitabine is marked not taking and chemotherapy is explicitly held, original resectability and 2/29 positive regional nodes, current negative distant imaging, conditional five-of-seven resumption, completed rather than future head CT, planned Dopplers, and ATM VUS.

### PDAC sample 19 — coral_idx 18

- Case: locally advanced unresectable pancreatic-head/uncinate adenocarcinoma that progressed on gemcitabine/nab-paclitaxel and is now treated with modified FOLFIRINOX. Cycle 3 was postponed for severe cholestatic/hepatocellular abnormalities. The same-day CT subsequently confirmed local primary growth with biliary and likely partial duodenal obstruction, but did not establish liver metastasis; an older 11 mm liver lesion remained uncharacterized. FOLFIRINOX remains the active regimen despite the single hold.
- PL P0: Metastasis preserves the supported uncertain liver concern but fabricates `confirmed regional nodes`.
- PL P1: findings omits the same-day negative bilateral-DVT ultrasound; imaging continues to list the CT and leg ultrasound as future even though the addendum contains their results; Specialty captures SMS but omits the urgent outgoing GI/ERCP referral.
- PL P2: summary omits the completed addendum CT and urgent ERCP; response correctly captures growth but calls confirmed local enlargement only `possible progression`; procedure says ERCP without the urgent referral context; Referral follow-up misroutes GI/ERCP into follow-up. The findings field also corrupts alkaline phosphatase 493 as `49[REDACTED]`.
- Attribution: A0 for second opinion, Distant, Metastasis, and next visit; A1 for Metastasis, supportive medications, and palliative goal; A2 for summary, Type, Stage, labs, findings, current medications, response, and Imaging.
- Core verdicts: current_meds PL; Stage TIE; Distant PL; Metastasis BL; response PL; genetic results TIE. Total PL 3 / BL 1 / TIE 2.
- Main verification: confirmed historical Gem/Abraxane progression, active FOLFIRINOX with only the current infusion postponed, definite local progression/obstruction in the addendum, unresolved rather than confirmed liver spread, no regional-node evidence, completed CT/ultrasound, and urgent GI referral.

### PDAC sample 20 — coral_idx 19

- Case: newly diagnosed pancreatic-tail adenocarcinoma with radiographically definite peritoneal carcinomatosis, multiple liver lesions that remain suspicious rather than confirmed, and prominent but unproven retroperitoneal nodes. No treatment has started; FOLFIRINOX or trial participation is being considered, UCSF500 is ordered, and germline counseling/testing is recommended.
- PL P0: Distant Metastasis combines confirmed peritoneal spread with merely suspicious liver lesions as if both were confirmed; general Metastasis repeats that liver-certainty error and additionally fabricates `confirmed regional nodes` from nonspecific prominent retroperitoneal nodes.
- PL P1: Type omits the explicitly metastatic status; procedure promotes trial-dependent tissue collection/biopsy requirements into current scheduled procedures before consent or screening; Referral follow-up substitutes possible nontherapeutic-study eligibility for an actual follow-up plan.
- PL P2: findings omits the prominent retroperitoneal nodes and mixes some symptoms into objective findings; supportive medications omits the primarily used Tylenol; medication plan does not foreground the patient's interest in REVOLUTION screening or the still-unselected regimen; Genetics records a recommendation as though an outgoing referral were definitively placed.
- Attribution: A0 for second opinion, Referral follow-up, and genetic-results fallback; A1 for labs, supportive medications, response, and next-visit fallback; A2 for Patient type, Stage, Distant, Metastasis, findings, goals description, medication plan, therapy plan, and genetic plan.
- Core verdicts: current_meds TIE; Stage TIE; Distant TIE; Metastasis BL; response TIE; genetic results TIE. Total PL 0 / BL 1 / TIE 5.
- Main verification: confirmed definite peritoneal carcinomatosis, only suspicious liver lesions, no malignant confirmation of retroperitoneal nodes, untreated status, conditional treatment/trial choices, ordered UCSF500, recommended germline testing, and intact MMR that both systems omit from genetic results.

## PDAC interim summary

- Core verdicts across 20 samples: PL 25 / BL 25 / TIE 70. Like breast, the matched v2.1 PDAC comparison is tied.
- Repeated high-impact patterns: copied `confirmed regional nodes` and unsupported organ clauses; clinical/radiographic suspicion upgraded to confirmation; old regimens retained as current or genuinely active regimens omitted; old stable scans overriding newer progression; current and historical stage/node status flattened; completed or relative molecular results misclassified as patient genetic results.
- Repeated non-core patterns: plans crossed among medication, therapy, procedure, imaging, lab, referral, and follow-up fields; completed tests persisted as future plans after an addendum; historical/incoming consultations became outgoing referrals; attribution frequently supported only part of a compound value.

## Matched v2.1 conclusion before targeted repairs

- Across 40 held-out samples and 260 manually adjudicated core comparisons, the result is exactly tied: PL 53 / BL 53 / TIE 154.
- Therefore the current matched-v2.1 artifacts do **not** support the paper's desired claim that the pipeline is better than the matched single-prompt baseline on the defined core questions.
- The result does identify a concentrated repair opportunity: many PL losses arise from a few deterministic failure families rather than broad model weakness, especially copied metastasis clauses, current-regimen state, latest-response precedence, and completed/pending genetic-result separation.
