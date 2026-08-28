# Matched Baseline Core-Field Review

Updated: 2026-08-27

## Scope

- Systems: final pipeline (PL) versus the corrected matched single-prompt baseline (BL).
- Samples: 20 breast + 20 PDAC annotated notes.
- Review scope: seven prespecified core categories only.
- Categories: `current_meds`, `Stage_of_Cancer`, `Distant Metastasis`, `Metastasis`, `response_assessment`, breast `Type_of_Cancer`, and `genetic_testing_results`.
- Verdict for each applicable sample-field: `PL`, `BL`, or `TIE` against the complete source note.
- Severity: P0 = fabricated/unsafe fact; P1 = material omission, wrong direction, temporal error, or misleading field answer; P2 = minor imprecision/formatting; OK = no substantive problem.
- Review method: natural-language reading of the complete note, complete PL and BL outputs, attribution where available, and field definitions. No script, regex, keyword, or exact-string rule is used to decide correctness.

## Status

- Completed: 40/40 samples.
- Breast: completed 1–20.
- PDAC: completed 1–20.
- Aggregate verdicts: PL 89 / BL 38 / TIE 133 across 260 applicable comparisons.
- Main-review calibration: completed for all 40 samples; every P0/P1 candidate was rechecked against the source note by the main reviewer.

## Aggregate results

| Core category | PL | BL | TIE | Net PL−BL |
|---|---:|---:|---:|---:|
| `current_meds` | 7 | 2 | 31 | +5 |
| `Stage_of_Cancer` | 17 | 4 | 19 | +13 |
| `Distant Metastasis` | 30 | 1 | 9 | +29 |
| `Metastasis` | 14 | 16 | 10 | −2 |
| `response_assessment` | 5 | 6 | 29 | −1 |
| breast `Type_of_Cancer` | 7 | 5 | 8 | +2 |
| `genetic_testing_results` | 9 | 4 | 27 | +5 |
| **Overall** | **89** | **38** | **133** | **+51** |

The matched comparison supports a substantial overall PL advantage and a positive net result in five of seven prespecified categories. It does not support an all-category-win claim: general metastasis and treatment response are slightly negative and need another pipeline revision.

## Cross-sample findings

- Strongest result: `Distant Metastasis` (+29). PL frequently preserves explicit M0/no-distant-disease findings and unresolved distant-site uncertainty that BL leaves empty.
- Next strongest: stage (+13). PL retains stated stage/resectability information, but its automatic `locally advanced → Stage III` conversion violates the current prompt and accounts for several BL wins.
- `current_meds` is positive (+5), mainly through temporal separation of current, prior, planned, and supportive drugs. Remaining PL failures come from active regimens omitted during ongoing therapy or held/completed regimens retained as current.
- General `Metastasis` is negative (−2). The recurring PL failure is collapsing regional nodal disease, direct local invasion, uncertain distant lesions, and confirmed distant sites into a single `Yes/No/Not sure` value.
- `response_assessment` is negative (−1). Recurrent errors include using pretreatment imaging, treatment toxicity, postoperative change, or a previous regimen's response instead of the current disease trajectory.
- The most important integrity issue is certainty preservation by site. Both systems sometimes convert `suspicious`, `indeterminate`, or pending-biopsy findings into confirmed metastases; PL does this in breast sample 15 and PDAC samples 6, 12, and 20.
- Attribution is not yet publication-ready: several PL values are correct but cite unrelated text, omit the decisive evidence, or contradict the final value.
- Contract alignment issue: the production PDAC prompt includes CA 19-9 non-secretor status under genetic results, whereas the matched baseline contract does not and defines the category as completed molecular/genetic testing. The comparison above follows the matched contract and does not award PL for this mismatch.

## Results

### Breast sample 1 — coral_idx 20

- Case basis: untreated right grade-3 IDC after mastectomy, pT2N1a / stated Stage II, 1/2 regional sentinel nodes positive; PET/CT pending for distant staging; final addendum and A/P support triple-negative disease; no completed molecular/genetic testing.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` BL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` PL.
- Main calibration: PL's `Stage IIB` is a P2 over-specific conversion from stated Stage II/pT2N1a while distant staging remained pending. BL has P1 omissions/misclassification for distant-staging uncertainty and genetic results, and P1 current-receptor error by presenting the internally conflicting HER2-positive history as the later status despite the final addendum/A&P supporting HER2-negative TNBC. No P0.
- Sample total: PL 3 / BL 2 / TIE 2.

### Breast sample 2 — coral_idx 21

- Case basis: 1994 ER+/PR− grade-1 IDC with untreated, unresectable locoregional chest-wall recurrence; latest imaging supports liver cyst and no distant disease; axillary nodes are unproven and prior biopsy was benign; zoledronic acid is for osteoporosis; aromatase inhibitor is planned.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` BL; `genetic_testing_results` TIE.
- Main calibration: PL fabricates HER2-negative status (P0) because HER2 is not reported. PL's general `Metastasis=No` is P1 because a locoregional chest-wall recurrence is confirmed. BL incorrectly treats osteoporosis zoledronic acid as current anticancer therapy (P1), carries forward an obsolete suspected liver metastasis despite current cyst/no-other-disease evidence (P1), and asserts unproven axillary-node involvement (P0). Both response values are P1 because the field contract treats recurrence after prior treatment as current disease progression; the verdict remains TIE.
- Sample total: PL 4 / BL 1 / TIE 2.

### Breast sample 3 — coral_idx 22

- Case basis: newly diagnosed ER−/PR−/HER2-FISH-negative spindle-cell metaplastic carcinoma, locally advanced and multifocal; axillary FNA negative and no evidence of distant disease; treatment has not started; InVitae and MammaPrint are pending.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` BL.
- Main calibration: BL's asserted enlarged axillary-node metastasis is P0 because the imaging was nonspecific/non-enlarged and FNA was negative. BL also omits explicit locally advanced status and no-distant-disease conclusion (P1 each). PL puts pending tests and unrelated pathology wording into completed genetic results (P1).
- Sample total: PL 3 / BL 1 / TIE 3.

### Breast sample 4 — coral_idx 23

- Case basis: two right grade-3 IDC lesions with distinct low receptor expression (ER 1%/PR− and ER−/PR 1%), both HER2-negative; local recurrent breast mass enlarged to 2.7 × 1.7 cm; no distant metastasis; chemotherapy not yet started; documented BRCA1 carrier.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` BL; `Type_of_Cancer` BL; `genetic_testing_results` PL.
- Main calibration: empty BL stage and PL `Not staged` are equivalent under the contract, so the initial PL verdict was changed to TIE. PL response is P1 because prior/recurrent disease growth is explicit. PL also collapses two low-positive receptor profiles into ER−/PR− (P1). BL omits explicit negative distant staging and BRCA1 status (P1 each), and places a local breast recurrence in the metastasis field (P1).
- Sample total: PL 3 / BL 2 / TIE 2.

### Breast sample 5 — coral_idx 24

- Case basis: bilateral resected IDC: left Stage III T3N1, grade 3, ER+/PR+/HER2− with micrometastatic axillary disease and MammaPrint high risk −0.614; right Stage I T1cN0, grade 1, ER+/PR+/HER2− with MammaPrint low risk +0.321; adjuvant therapy not yet started.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` TIE.
- Main calibration: BL leaves distant status empty despite supported non-metastatic Stage I/III disease (P1). PL's general regional-node value is correct but BL preserves side and micrometastatic detail. Both systems omit completed bilateral MammaPrint results (P1). PL is more complete on bilateral pathology but both omit the right-sided HER2 change (P2).
- Sample total: PL 2 / BL 1 / TIE 4.

### Breast sample 6 — coral_idx 25

- Case basis: untreated ER−/PR−, HER2-amplified grade-2 IDC; right axillary FNA confirms regional metastasis; left ilium and bilateral sacral lesions remain suspicious pending biopsy for definitive Stage IV diagnosis.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` TIE.
- Main calibration: PL correctly preserves suspected distant disease and its sites but `Metastasis=Not sure` omits confirmed right axillary nodal metastasis (P1). BL correctly combines confirmed regional and suspected distant disease, though it is less specific on distant sites and receptor details (P2).
- Sample total: PL 2 / BL 1 / TIE 4.

### Breast sample 7 — coral_idx 26

- Case basis: biopsy-confirmed TNBC liver metastases plus regional nodes; current pembrolizumab + nab-paclitaxel at cycle 1 day 8; axillary pain has improved; germline MSH2 pathogenic variant/Lynch syndrome with a separate negative breast-cancer panel.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` TIE; `Metastasis` TIE; `response_assessment` BL; `Type_of_Cancer` PL; `genetic_testing_results` PL.
- Main calibration: both systems incorrectly mix regional subpectoral/internal-mammary nodes into the distant-metastasis site list (P1), while liver is the confirmed distant site. PL's response uses a pretreatment liver scan as response to the just-started regimen (P1); BL correctly uses current axillary-pain improvement. BL omits Stage IV, detailed TNBC pathology, and the MSH2/panel results (P1).
- Sample total: PL 3 / BL 1 / TIE 3.

### Breast sample 8 — coral_idx 27

- Case basis: multifocal left Stage IIA pT2(m)N1a ER+/PR+/HER2− IDC with 2/12 macrometastatic axillary nodes; MammaPrint high risk; adjuvant treatment not yet started; germline testing only planned.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` TIE.
- Main calibration: BL omits supported non-distant status (P1). BL is more specific about the regional nodal burden. Both systems omit completed high-risk MammaPrint (P1), while PL explicitly but incorrectly says no result. PL is more complete on IDC/DCIS, though both omit multifocal and mixed-grade detail (P2).
- Sample total: PL 2 / BL 1 / TIE 4.

### Breast sample 9 — coral_idx 28

- Case basis: original Stage III pT3N2 ER+/PR+/HER2− micropapillary IDC with confirmed historical axillary disease; now biopsy-proven unresectable local recurrence with several currently suspicious regional nodes and a level Vb node pending FNA for possible distant disease; new systemic therapy not yet started.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` TIE; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` TIE.
- Main calibration: BL omits the current suspected Stage IV/distant status and gives incomplete current receptor/recurrence detail (P1). PL and BL each capture only part of the general metastasis timeline: PL reports confirmed regional disease without time/site/uncertainty, while BL reports the current suspected level Vb disease but omits confirmed historical regional involvement; the initial BL verdict was changed to TIE.
- Sample total: PL 3 / BL 0 / TIE 4.

### Breast sample 10 — coral_idx 29

- Case basis: untreated clinical Stage II cT2N1 right grade-2 IDC, ER >95%/PR 25%/HER2 clinically not amplified, axillary FNA positive, PET negative for distant disease; fertility preservation is underway and neoadjuvant therapy is planned; genetic/MammaPrint results remain pending.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` TIE.
- Main calibration: BL misclassifies tamoxifen in the fertility-preservation context as current anticancer treatment (P1) and omits the explicit negative distant staging (P1). BL is more specific on the biopsy-positive regional axillary node. Empty BL genetic results and PL's explicit no-result fallback are equivalent because all testing is pending; the initial PL verdict was changed to TIE.
- Sample total: PL 2 / BL 1 / TIE 4.

### Breast sample 11 — coral_idx 30

- Case basis: pure 1.8 cm intermediate-grade ER+/PR+ DCIS after lumpectomy, pTisNx, with no invasive carcinoma or nodal/distant disease; tamoxifen and radiation have not yet started; no genetic result.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` TIE.
- Main calibration: both systems retain an earlier `PR pending` status despite the later lumpectomy pathology reporting ER/PR positive (P1). BL also leaves both metastasis fields blank despite pure pTis disease and explicit absence of invasive/nodal disease (P1).
- Sample total: PL 2 / BL 0 / TIE 5.

### Breast sample 12 — coral_idx 31

- Case basis: untreated clinical Stage II, node-negative right grade-2 invasive mammary carcinoma with mixed ductal/lobular features, ER >95%/PR ~70%/HER2−, PET negative for distant disease; neoadjuvant AC/T planned; no molecular result.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` BL; `genetic_testing_results` TIE.
- Main calibration: BL omits the explicit negative distant/nodal conclusion (P1). Both type values are correct, but BL more completely preserves ER percentage and Ki-67; PL has only a minor completeness gap.
- Sample total: PL 2 / BL 1 / TIE 4.

### Breast sample 13 — coral_idx 32

- Case basis: untreated right IDC, ER/PR 80–90%, HER2-negative, Ki-67 20–30%; FNA-confirmed regional right axillary metastasis; PET negative outside regional nodes; a tiny falx lesion is favored to be meningioma; MammaPrint only planned.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` TIE; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` TIE.
- Main calibration: BL omits the explicit negative distant-metastasis workup (P1). Empty BL stage/genetic-result values are valid equivalents of PL's textual fallbacks, so two initial PL verdicts were changed to TIE.
- Sample total: PL 1 / BL 0 / TIE 6.

### Breast sample 14 — coral_idx 33

- Case basis: right grade-1 ER+/PR+ IDC, HER2 IHC 2+ but FISH non-amplified, no formal stage or explicit distant staging; goserelin started today and letrozole planned; Myriad negative and MammaPrint low risk +0.287.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` TIE; `Metastasis` TIE; `response_assessment` BL; `Type_of_Cancer` TIE; `genetic_testing_results` PL.
- Main calibration: PL's `Not yet on treatment` response contradicts same-day goserelin initiation (P1); BL correctly avoids that assertion. Both systems should normalize HER2 IHC 2+/FISH non-amplified to HER2-negative (P1). PL correctly preserves both completed molecular results; BL omits MammaPrint (P1). In the absence of explicit staging evidence, empty BL metastasis values and PL's inferred negatives are treated as TIE.
- Sample total: PL 1 / BL 1 / TIE 5.

### Breast sample 15 — coral_idx 34

- Case basis: breast-origin metastatic adenocarcinoma confirmed in a right supraclavicular node, ER >90%/PR 50%/HER2 IHC 2+ with FISH pending; additional axillary and cervical nodes remain presumptive or pending biopsy; definitive de novo Stage IV status remains conditional.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` TIE; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` TIE.
- Main calibration: BL turns conditional Stage IV and cervical distant disease into confirmed findings (P0). Both general metastasis outputs erase site-specific evidence status and overstate unconfirmed nodes (P0), so that field is TIE. PL preserves stage uncertainty but should name the suspected cervical site rather than only `Not sure` (P2).
- Sample total: PL 2 / BL 0 / TIE 5.

### Breast sample 16 — coral_idx 35

- Case basis: untreated clinical Stage III left invasive lobular carcinoma, ER 96%/PR 35%/HER2-negative; PET shows no distant disease, while left axillary/subpectoral nodes are suspicious regional findings with FNA still pending; no completed molecular/genetic result.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` TIE.
- Main calibration: BL leaves distant status empty despite explicit PET evidence against distant disease (P1). PL incorrectly gives general `Metastasis=No` despite imaging-supported suspected regional nodal involvement (P1); BL appropriately preserves the regional site and uncertainty. Empty BL genetic results and PL's explicit no-result wording are equivalent under the contract. No P0.
- Sample total: PL 1 / BL 1 / TIE 5.

### Breast sample 17 — coral_idx 36

- Case basis: untreated Stage IIB/pT2N1M0 grade-2 IDC, ER >95%/PR 25%/HER2-negative, with 2/2 confirmed positive regional nodes; germline BRCA testing is negative, while HER2 FISH is receptor testing rather than a hereditary/genomic result.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` TIE; `response_assessment` TIE; `Type_of_Cancer` TIE; `genetic_testing_results` PL.
- Main calibration: BL omits the explicit M0 status (P1). Both systems incorrectly place HER2 FISH in genetic-testing results (P1); PL is still more informative because it preserves the actual negative BRCA result and clearly separates the uncertain panel scope. No P0.
- Sample total: PL 2 / BL 0 / TIE 5.

### Breast sample 18 — coral_idx 37

- Case basis: untreated clinical T2NX left grade-2 IDC, ER >95%/PR low-positive <5%/HER2 FISH-negative; MRI shows abnormal left axillary nodes but two FNAs found no cancer; no explicit systemic distant-staging result; completed Myriad testing found a pathogenic ATM variant and MammaPrint was high risk −0.622.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` TIE; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` BL; `genetic_testing_results` PL.
- Main calibration: BL omits explicit cT2NX staging (P1) and turns imaging-abnormal, twice-FNA-negative axillary nodes into definite regional involvement (P0). PL's unqualified `No` does not preserve the residual imaging/biopsy uncertainty, but it is materially safer than BL's confirmed-metastasis claim. PL type follows the A/P shorthand `PR−` instead of the detailed pathology showing PR low-positive <5% (P1); BL correctly preserves the percentage. PL includes both completed molecular results, whereas BL omits MammaPrint (P1). Because the note provides no explicit systemic distant-staging conclusion, PL's inferred `No` and BL's empty distant field are scored TIE rather than rewarding the textual fallback.
- Sample total: PL 3 / BL 1 / TIE 3.

### Breast sample 19 — coral_idx 38

- Case basis: treated multifocal left IDC with two distinct pathology profiles (grade 2 ER 61–70%/PR−/HER2− and grade 3 ER 41–50%/PR 1–10%/HER2− with focal LCIS), originally clinical Stage 2–3 and MammaPrint low risk; after neoadjuvant hormonal therapy and bilateral mastectomy, the patient is currently NED on exemestane plus monthly goserelin/Zoladex.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `Type_of_Cancer` PL; `genetic_testing_results` PL.
- Main calibration: BL omits active monthly ovarian suppression (P1), explicit prior negative distant staging (P1), the two lesions' grade/receptor distinctions (P1), and completed low-risk MammaPrint (P1). PL is materially more complete, but its type summary collapses the second lesion's PR 1–10% into PR-negative (P2), and its genetic-results field is contaminated with ER/PR/HER2, Ki-67, and LCIS pathology rather than containing only MammaPrint (P1). PL's distant-status attribution is unrelated BSO text even though the extracted value itself is supported (attribution P2). No P0.
- Sample total: PL 5 / BL 0 / TIE 2.

### Breast sample 20 — coral_idx 39

- Case basis: untreated bilateral large breast cancers; right is ER+/PR+/HER2-positive with Ki-67 40%, left is ER+/PR+/HER2 0; an invasive cancer has some lobular differentiation but the note does not bind that feature to a side; regional-node burden is unknown, and lung and liver lesions are suspected but explicitly unconfirmed; germline testing remains pending.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` TIE; `Metastasis` BL; `response_assessment` TIE; `Type_of_Cancer` BL; `genetic_testing_results` TIE.
- Main calibration: no formal AJCC/TNM stage is stated, so PL's textual fallback and BL's empty stage are treated as equivalent; PL's added `locally advanced` is a reasonable clinical inference but not a stated stage and conflicts with the note's own `early stage` shorthand (P2). BL better preserves both unknown regional-node involvement and unconfirmed distant lesions. PL incorrectly attaches the otherwise real `some lobular differentiation` finding specifically to the right breast (P1), while the source gives no side; given the project's precision-first rule, BL's less complete but non-fabricated bilateral receptor summary wins. No P0.
- Sample total: PL 0 / BL 2 / TIE 5.

### PDAC sample 1 — coral_idx 0

- Case basis: locally advanced pancreatic adenocarcinoma after six completed cycles of gemcitabine/nab-paclitaxel; the primary is stable to minimally larger, abdomen/pelvis has no metastatic disease, and five new pulmonary nodules are indeterminate but suspicious; the physician assesses possible progression/maximum regimen benefit and agrees to a chemotherapy break with surveillance; no completed molecular/genetic result.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` BL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: BL temporally mislabels the completed, now-paused gemcitabine/nab-paclitaxel regimen as current (P1) and omits the important uncertain distant status (P1). PL invents the numeric label `Stage III` from `locally advanced`, despite the PDAC prompt's explicit prohibition on converting unstated descriptors into AJCC numbers (P1); BL also omits `locally advanced` (P1), but its empty stage is safer under the precision-first rule. BL's general-metastasis answer is superior because it names the lung site and preserves `possible/indeterminate/suspicious`, whereas PL says only `Not sure`. Both response answers are P1: PL omits the new suspicious lung lesions and overall possible-progression assessment, while BL substitutes the break/plan for a complete objective response and omits the pancreatic-mass finding. Empty BL genetic results and PL's textual no-result fallback are equivalent. No P0.
- Sample total: PL 2 / BL 2 / TIE 2.

### PDAC sample 2 — coral_idx 1

- Case basis: pancreatic adenocarcinoma resected by Whipple with direct duodenal invasion and 6/25 positive regional nodes, followed by confirmed liver metastases; FOLFIRINOX eventually progressed and the patient is now at C2D1 of second-line gemcitabine/nab-paclitaxel without current-regimen restaging imaging; completed testing shows SPINK1 c.101A>G carrier status and FANCG/NF2 VUS.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` TIE; `Metastasis` TIE; `response_assessment` PL; `genetic_testing_results` TIE.
- Main calibration: BL omits definite current Stage IV disease (P1). Both general-metastasis outputs omit the documented 6/25 positive regional nodes (P1). Neither response fully answers the current regimen: PL accurately identifies progression on the prior FOLFIRINOX and the switch but fails to state that Gem/nab-paclitaxel has no restaging assessment yet (P1); BL mixes old FOLFIRINOX-era CT findings into the current regimen and calls a 1,625→634→900 marker sequence `generally decreasing`, obscuring the latest rebound (P1), making PL safer. Both genetic outputs omit the specific SPINK1 c.101A>G variant (P2). No P0.
- Sample total: PL 2 / BL 0 / TIE 4.

### PDAC sample 3 — coral_idx 2

- Case basis: metastatic PDAC after progression through earlier regimens, currently on third-line 5-FU/LV plus nanoliposomal irinotecan; definite distant disease includes liver and peritoneal implants, while splenic involvement is confounded by direct invasion from the pancreatic-tail primary; current imaging is broadly stable/mixed with slightly improved liver lesions and stable-to-minimally larger peritoneal disease despite clinical decline; tumor testing shows KRAS G12D and p53 mutations.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: BL omits explicit metastatic/Stage IV status (P1). BL overstates the spleen as confirmed distant spread even though the current report describes direct extension from the pancreatic-tail mass (P1), and its general-metastasis field also lists the pancreatic primary itself as a metastatic site (P1). PL more safely limits definite distant disease to liver and peritoneum. Both response outputs correctly convey the mixed imaging pattern and clinical decline. PL's metastasis attribution does not directly support the named liver/peritoneal sites (attribution P2). No P0.
- Sample total: PL 3 / BL 0 / TIE 3.

### PDAC sample 4 — coral_idx 3

- Case basis: pancreatic adenocarcinoma initially considered locally advanced; a segment-7 liver lesion was MRI-suspicious and judged consistent with metastasis by an outside center but was not PET-avid, while the latest CT shows only known hemangiomas and no current metastatic evidence; after six cycles of gemcitabine/nab-paclitaxel the patient is on surveillance with stable primary disease; the tumor does not express CA 19-9.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` BL; `Metastasis` BL; `response_assessment` TIE; `genetic_testing_results` BL.
- Main calibration: PL preserves both the locally advanced history and the physician's current metastatic clinical classification, while BL leaves stage empty (P1). For the site-specific metastasis fields, however, PL's definite `Yes, to liver` erases the historical uncertainty and current negative imaging (P1); BL's `suspected segment 7 liver metastasis` is safer, though it too omits the current negative scan (P2). Both response values faithfully express stable/good disease control. PL incorrectly places CA 19-9 non-secretion in completed genetic-testing results (P1); this is tumor-marker behavior, not a molecular/genetic test result, so BL's empty value is correct under the matched contract. No P0.
- Sample total: PL 1 / BL 3 / TIE 2.

### PDAC sample 5 — coral_idx 4

- Case basis: explicitly Stage IV pancreatic adenocarcinoma with biopsy-proven oligometastatic abdominal-wall disease; after 12 completed FOLFIRINOX cycles the patient is on a chemotherapy break, and the latest CT shows stable pancreatic and abdominal-wall lesions; completed molecular results include pancreatic-FNA MSS and Foundation liquid-biopsy MSI-undetermined/RB1 P26fs*47.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` TIE; `Distant Metastasis` TIE; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: PL correctly leaves active anticancer therapy empty during the chemotherapy break; BL incorrectly fills `current_meds` entirely with supportive/non-anticancer agents (P1). BL also calls the biopsy-proven abdominal-wall metastasis `regional`, a directionally wrong classification (P1). Both genetic outputs correctly include the Foundation result but omit the earlier MSS result and testing context (P1). Both response values are accurate, though PL's attribution misses the direct stable-disease sentence (attribution P2). No P0.
- Sample total: PL 2 / BL 0 / TIE 4.

### PDAC sample 6 — coral_idx 5

- Case basis: initially borderline-resectable/locally advanced pancreatic adenocarcinoma treated with 12 cycles of neoadjuvant modified FOLFIRINOX and resection; pathology shows a 1.7 cm residual tumor with 51–90% treatment destruction, negative margins, and negative nodes; postoperative CA 19-9 rose to 2250 and CT shows an enlarging plus a new tiny liver lesion suggestive of recurrence/metastasis but too small to confirm, with short-interval imaging and possible biopsy planned.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` PL; `genetic_testing_results` TIE.
- Main calibration: BL omits the explicit borderline/locally advanced history and early-recurrence status (P1), and omits the important uncertain distant-liver status (P1). PL's general `Metastasis=Yes (to liver)` converts `suspicious/suggestive`, too-small-to-evaluate lesions requiring possible confirmatory biopsy into definite disease (P0), and contradicts its own `Distant Metastasis=Not sure`; BL preserves the uncertainty. PL's response more fully states unconfirmed recurrence and short-interval monitoring, though both outputs omit the earlier neoadjuvant pathologic response (P2). Empty BL genetic results and PL's no-result fallback are equivalent. PL's metastasis attribution is absent and its distant attribution cites its own label rather than source text (attribution A1). 
- Sample total: PL 3 / BL 1 / TIE 2.

### PDAC sample 7 — coral_idx 6

- Case basis: pancreatic-tail adenocarcinoma with local extension/abutment of adjacent structures and splenic-vessel involvement, no formal AJCC stage and no distant disease on current chest/abdomen imaging; after four cycles of gemcitabine/nab-paclitaxel on a modified every-other-week schedule, the mass is slightly smaller and treatment will continue; germline testing is negative for pathogenic variants with AXIN1/CTC1/ERCC4/MC1R VUS.
- Verdicts: `current_meds` BL; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` BL; `genetic_testing_results` TIE.
- Main calibration: PL incorrectly leaves active gemcitabine/nab-paclitaxel empty despite ongoing cycles and an explicit continue instruction (P1). BL omits the available local-extension/resectability information in stage (P1), omits the supported negative distant workup (P1), and puts direct local extension into the metastasis field (P1). PL over-upgrades `slight decrease` and `radiographic evidence of response` into the formal-sounding `partial response` category (P1); BL preserves the measured wording and wins on precision. PL's negative-metastasis and genetic values lack direct attribution (A1/P2). No P0.
- Sample total: PL 3 / BL 2 / TIE 1.

### PDAC sample 8 — coral_idx 7

- Case basis: recurrent metastatic PDAC in a patient with germline ATM mutation; original Whipple pathology was pT2N2 with 11/37 positive nodes and intact MMR, followed by biopsy-confirmed gastrohepatic/intra-abdominal nodal recurrence; the patient is currently receiving every-other-week gemcitabine/nab-paclitaxel with smaller abdominal nodes and CA 19-9 falling 746→433→133→61.
- Verdicts: `current_meds` BL; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `genetic_testing_results` PL.
- Main calibration: PL omits the clearly active gemcitabine/nab-paclitaxel regimen despite ongoing cycles and an explicit continue recommendation (P1). BL omits the physician-explicit recurrent metastatic/Stage IV status and biopsy-confirmed nodal metastatic disease (P1 each). Both metastasis outputs are correct, but BL wins the general field by preserving `intra-abdominal` specificity. Both response summaries are accurate; PL adds a marker trend but stops at 133 rather than the latest 61 (P2), so no advantage is awarded. PL preserves germline ATM and intact MMR, whereas BL's `no actionable mutations identified` omits both (P1); PL in turn omits that reported FoundationOne summary (P2). Several PL attributions support only the original pT2N2 disease rather than current recurrence, and response attribution is absent (A1/P2). No P0.
- Sample total: PL 3 / BL 2 / TIE 1.

### PDAC sample 9 — coral_idx 8

- Case basis: biopsy-confirmed lung-predominant metastatic pancreatic cancer on gemcitabine/nab-paclitaxel with an Abraxane dose reduction for neuropathy; pulmonary treated metastases are stable, the pancreatic primary/infiltrative tissue has decreased, and CA 19-9 fell 3525→2762→1109; molecular findings are BRCA2 VUS plus KRAS, CDKN2A, and APC mutations.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` TIE; `Metastasis` TIE; `response_assessment` PL; `genetic_testing_results` TIE.
- Main calibration: BL omits explicit metastatic/Stage IV status (P1). Both systems correctly capture current therapy and biopsy-confirmed lung disease. PL wins response by adding the objective primary-tumor shrinkage to the correct stable-disease conclusion; it could still include stable pulmonary metastases and the marker decline. Both genetic outputs generalize `BRCA2 VUS` to `BRCA VUS` (P2). PL's metastasis attribution is too generic to support the lung site/biopsy status, and genetic attribution is absent (A1/P2). No P0.
- Sample total: PL 2 / BL 0 / TIE 4.

### PDAC sample 10 — coral_idx 9

- Case basis: locally advanced PDAC with no distant metastasis, currently receiving C6D1 FOLFOX after irinotecan was removed from cycle 3 for colitis/poor tolerance; current CT shows stable local disease and no distant spread; completed genetic testing shows a RECQL4 VUS.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: BL omits the explicit locally advanced status and explicit negative distant workup (P1 each). BL also treats small, stable periportal/peripancreatic nodes as metastatic involvement despite no malignant/suspicious characterization or pathology (P0); PL correctly avoids that conversion. Both response and genetic values are faithful. PL should name the FOLFOX components and keep discontinued irinotecan in treatment changes, and its response attribution should also support the CA 19-9 context (P2). 
- Sample total: PL 3 / BL 0 / TIE 3.

### PDAC sample 11 — coral_idx 10

- Case basis: newly diagnosed metastatic pancreatic cancer with a formally signed Stage IV cT2 cN1 cM1 classification and liver metastases; treatment has not started, STRATA is pending, and the BRCA2 mutation belongs to the patient's brother rather than the patient.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` TIE; `Metastasis` TIE; `response_assessment` TIE; `genetic_testing_results` BL.
- Main calibration: both systems correctly capture the formal stage, liver disease, lack of active treatment, and lack of a response assessment. PL puts a relative's BRCA2 result and the patient's pending STRATA test into completed patient genetic results (P1); BL correctly leaves the completed-results field empty. No P0.
- Sample total: PL 0 / BL 1 / TIE 5.

### PDAC sample 12 — coral_idx 11

- Case basis: pancreatic adenocarcinoma after 12 cycles of modified FOLFIRINOX and a treatment holiday, now with primary progression, bowel/biliary obstruction, and definite peritoneal/omental carcinomatosis; liver lesions remain hard to interpret and shrinking pulmonary nodules only raise concern for involvement; no current anticancer therapy and palliative/hospice transition is being discussed.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: both systems correctly identify Stage IV and current progression, but both commit P0 site-certainty errors. PL correctly includes definite peritoneal disease yet also labels indeterminate liver lesions as confirmed; BL labels both liver and nonspecific lung nodules confirmed and omits the definite peritoneal/omental carcinomatosis, making PL materially better on both metastasis fields. PL's response attribution covers only prior stability rather than current progression (A1/P2). Empty BL genetic results and PL's no-result fallback are equivalent.
- Sample total: PL 2 / BL 0 / TIE 4.

### PDAC sample 13 — coral_idx 12

- Case basis: locally advanced unresectable pancreatic head/uncinate adenocarcinoma with duodenal and vascular involvement and imaging-described mesenteric nodes; after six completed gemcitabine/capecitabine cycles, the current plan is to pause systemic therapy and reconsider radiation; imaging is interpreted as stable, no definite distant metastasis is present, UCSF500 is pending, and the patient does not express CA 19-9.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` TIE; `response_assessment` BL; `genetic_testing_results` BL.
- Main calibration: both systems incorrectly call completed/paused systemic therapy current (P0), although BL lists only capecitabine and PL lists both drugs. BL omits explicit locally advanced/unresectable status (P1) and the supported absence of distant spread. Both general-metastasis answers are P1: PL erases imaging-described nodal involvement, while BL mixes nodal involvement with direct duodenal invasion. BL's concise `stable disease` is cleaner than PL's contamination with biliary findings. Under the matched contract, CA 19-9 non-secretion is not a molecular/genetic result, so PL's placement is P1 and BL's empty value is correct; the production PDAC prompt currently disagrees by explicitly including non-secretor status, which is a contract mismatch to fix before publication.
- Sample total: PL 2 / BL 2 / TIE 2.

### PDAC sample 14 — coral_idx 13

- Case basis: newly diagnosed Stage IV pancreatic-origin adenocarcinoma with biopsy-confirmed liver disease and CT-reported increasing hepatic and nodal metastases; chemotherapy has not started, MMR is pending, and UCSF500 is planned.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` TIE; `Metastasis` BL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: both systems correctly capture stage, liver metastasis, no active therapy, and no response yet. BL wins the general metastasis field by retaining the CT-reported nodal disease that PL omits (P2). No completed genetic result exists, so empty BL and PL's explicit fallback are equivalent. No P0/P1 in the six scored fields.
- Sample total: PL 0 / BL 1 / TIE 5.

### PDAC sample 15 — coral_idx 14

- Case basis: resected PDAC after neoadjuvant FOLFIRINOX; formal surgical pathology shows 4.6 cm moderately differentiated disease with poor/no treatment response, positive margins, 11/46 positive regional nodes, and AJCC ypT3N2; there is no confirmed distant metastasis, but CA 19-9 is rising and recurrence is a concern; completed molecular results include intact MMR and a benign ascites specimen negative for KRAS exons 2/3/4.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` BL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` PL; `genetic_testing_results` PL.
- Main calibration: PL follows the A/P typo-like `pT2N3` rather than the detailed pathology's valid `ypT3N2` (P1). BL omits repeatedly documented negative distant staging (P1). PL incorrectly says no metastasis despite 11/46 positive regional nodes (P1); BL preserves them. BL wrongly says no response can be assessed despite explicit Evans grade I/Ryan score 3 poor response after neoadjuvant therapy (P1), while PL captures the direction but remains vague (P2). PL includes both MMR and KRAS results; BL omits MMR (P1). No P0.
- Sample total: PL 3 / BL 2 / TIE 1.

### PDAC sample 16 — coral_idx 15

- Case basis: clinical Stage IIB cT1c cN1 cM0 pancreatic adenocarcinoma on dose-reduced gemcitabine monotherapy, now resuming C2D15 on an alternate-week schedule; CT shows the pancreatic mass is no longer visible, ductal dilation has improved, and there is no distant disease; prior peripancreatic/periportal nodes were PET-concerning; no completed molecular/genetic result.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` BL; `genetic_testing_results` TIE.
- Main calibration: BL omits explicit cM0/current negative distant staging (P1). PL's general `Metastasis=No` conflicts with cN1 and the described concerning regional nodes (P1); BL preserves site and uncertainty. PL also downgrades the strongest response evidence—mass no longer seen likely due to treatment—to generic stable disease/ductal improvement (P1); BL retains the mass disappearance. Empty BL genetic results and PL's fallback are equivalent. No P0.
- Sample total: PL 1 / BL 2 / TIE 3.

### PDAC sample 17 — coral_idx 16

- Case basis: pMMR locally advanced pancreatic adenocarcinoma with local SMA/mesenteric-root extension and SMV–portal-confluence occlusion but no confirmed distant or nodal metastasis; after eight FOLFIRINOX cycles the patient is on a chemotherapy break with stable disease; Foundation testing reports MSS, TMB 5, KRAS G12V, TP53 I195F, and additional variants.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` BL; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: PL again converts `locally advanced` into an unstated numeric `Stage III`, violating the explicit no-conversion rule (P1); BL omits the descriptive stage (P1) but is safer under the precision-first rule. BL omits the supported negative distant status (P1) and misclassifies continuous local vascular invasion as regional metastasis (P1). Both response and molecular summaries are correct. PL lacks direct attribution for the negative metastasis, response, and genetic results (A1/P2). No P0.
- Sample total: PL 2 / BL 1 / TIE 3.

### PDAC sample 18 — coral_idx 17

- Case basis: resected moderately differentiated pancreatic-tail adenocarcinoma with 2/29 positive regional nodes and no current distant disease; after two postoperative gemcitabine/capecitabine cycles, chemotherapy is temporarily held for severe hand-foot syndrome and mucositis, with a modified restart planned; completed germline testing shows an ATM VUS.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: neither current-medication output satisfies the active-only contract, but PL explicitly marks gemcitabine/capecitabine `currently on hold` (P2) while BL presents capecitabine as active despite both `Patient not taking` and the current hold (P1), so PL is safer. BL omits the available resectable/postoperative node-positive stage context and explicit negative distant imaging (P1 each). PL incorrectly says no metastasis despite 2/29 positive regional nodes (P1); BL correctly preserves them. Both response outputs are P1: PL substitutes postoperative vascular/fluid resolution and BL substitutes treatment toxicity for tumor response. Both capture ATM VUS. No P0.
- Sample total: PL 3 / BL 1 / TIE 2.

### PDAC sample 19 — coral_idx 18

- Case basis: locally advanced unresectable pancreatic-head adenocarcinoma that progressed on gemcitabine/nab-paclitaxel and is now on FOLFIRINOX; cycle 3 is temporarily postponed for severe hepatobiliary abnormalities, and same-day CT confirms primary-tumor growth with biliary and duodenal obstruction; an 11 mm liver lesion remains uncharacterized/unbiopsied, so distant disease is uncertain; no completed molecular/genetic result.
- Verdicts: `current_meds` PL; `Stage_of_Cancer` PL; `Distant Metastasis` PL; `Metastasis` BL; `response_assessment` PL; `genetic_testing_results` TIE.
- Main calibration: BL includes obsolete gemcitabine/nab-paclitaxel alongside the current FOLFIRINOX regimen (P0), omits explicit locally advanced/unresectable status (P1), omits the uncertain distant-liver finding (P1), and fails to incorporate the addendum's confirmed primary progression/obstruction (P1). PL is correct on those questions. BL wins general metastasis because it names the 11 mm liver lesion and preserves its uncharacterized/unbiopsied status, whereas PL says only `Not sure` (P2). Empty BL genetic results and PL's fallback are equivalent. No PL P0/P1.
- Sample total: PL 4 / BL 1 / TIE 1.

### PDAC sample 20 — coral_idx 19

- Case basis: untreated newly diagnosed metastatic pancreatic-tail adenocarcinoma; CT shows imaging-definite peritoneal carcinomatosis and multiple liver lesions described only as suspicious for metastases; MMR proteins are intact, while UCSF500 remains pending.
- Verdicts: `current_meds` TIE; `Stage_of_Cancer` TIE; `Distant Metastasis` PL; `Metastasis` PL; `response_assessment` TIE; `genetic_testing_results` TIE.
- Main calibration: both systems overstate the suspicious liver lesions as confirmed (P0). PL nevertheless wins both metastasis fields because it includes the imaging-definite peritoneal carcinomatosis; BL omits it from distant disease and reverses the evidence strengths by calling liver confirmed but peritoneum only suspected (P0). Both genetic outputs omit completed intact MMR IHC (P1); UCSF500 is pending and should not be reported as a result. No treatment has started, so both response outputs are correct.
- Sample total: PL 2 / BL 0 / TIE 4.
