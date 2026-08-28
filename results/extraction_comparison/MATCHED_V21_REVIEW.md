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

- Completed: 12/40
- Next: Breast sample 13
- PL findings: P0=4, P1=64, P2=41
- Attribution findings: A0=23, A1=61, A2=79
- Core verdict totals (PL / BL / TIE): 17 / 15 / 52

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
