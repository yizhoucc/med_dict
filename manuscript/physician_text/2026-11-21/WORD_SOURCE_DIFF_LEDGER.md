# Word source diff ledger

## Authority

- Physician archive SHA-256: `81f4fccf08b608aed229408919bfb90bddc9c7f68a0fb056e74a6f953eaedcc4`
- Locked project boundary: `v2.3.0` at `98569a744d2cd40d43419aa1b6c21565b5be1835`
- Locked project TXT SHA-256: `f662c3a769a14ff1ffc5fef083ba6e69758599fe30143fb118abe7cfe0d7b754`
- The physician archive is byte-for-byte unchanged. The Word source applies only the copyedits below and inserts four separate editorial annotations.

## Direct copyedits

| ID | Source line | Before | After | Reason |
|---|---:|---|---|---|
| CE001 | 16 | `concerns[4]` | `concerns [4]` | Add the missing space before citation [4]. |
| CE002 | 16 | `extraction[2]` | `extraction [2]` | Add the missing space before citation [2]. |
| CE003 | 28 | `sinlge prompt` | `single prompt` | Correct the spelling error in the baseline description. |
| CE004 | 38 | `in single prompt` | `in a single prompt` | Add the missing article. |
| CE005 | 20 | `single-prompt` | `single prompt` | Use the physician's preferred unhyphenated term throughout. |
| CE006 | 84 | `single-prompt` | `single prompt` | Use the physician's preferred unhyphenated term throughout. |
| CE007 | 282 | `single-prompt` | `single prompt` | Use the physician's preferred unhyphenated term throughout. |
| CE008 | 167 | `status.Huang` | `status. Huang` | Add the missing sentence space. |
| CE009 | 167 | `have showed` | `have shown` | Correct the verb form. |
| CE010 | 167 | `completeness.[9,10].` | `completeness [9,10].` | Correct citation spacing and duplicate punctuation. |
| CE011 | 171 | `deterministric` | `deterministic` | Correct the spelling error. |
| CE012 | 171 | `post-proccessing` | `post processing` | Correct the spelling and use the requested unhyphenated term. |
| CE013 | 219 | `Verification and postprocessing order` | `Verification and post processing order` | Use the requested spaced term in the Supplement heading. |
| CE014 | 171 | `rules, this suggests` | `rules. This suggests` | Repair the comma splice. |
| CE015 | 171 | `hormone status, which are used to classify breast cancer subtypes but are not applicable to PDAC` | `hormone status, which is used to classify breast cancer subtypes but is not applicable to PDAC` | Correct subject-verb agreement. |
| CE016 | 151 | `In all the 1,040 evaluations` | `Across all 1,040 evaluations` | Correct the sentence opening. |
| CE017 | 131 | `86.21%), the pancreatic cancer estimate` | `86.21%). The pancreatic cancer estimate` | Repair the comma splice between the two estimates. |
| CE018 | 18 | ` ` | `[removed]` | Remove trailing whitespace. |

## Editorial annotations

The physician sentence or paragraph remains in place after direct copyediting. Each item below follows it as a separate parenthetical note.

| ID | Anchor line | Type | Text |
|---|---:|---|---|
| AN001 | 16 | proposed_replacement | (Proposed replacement: "LLMs can extract information from clinical notes, but their outputs can vary. They may omit details, confuse past findings with current findings, or add information that is not supported by the original note [4,5]. In the CORAL benchmark, the commercial models outperformed the open model evaluated in that study [4]. Sending protected health information (PHI) to externally hosted services raises privacy and governance concerns [2]. Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction. Although manual review can identify extraction errors, reviewing every output is impractical at scale. These challenges motivate an LLM-based extraction workflow that improves reliability, preserves data privacy, and reduces manual review.") |
| AN002 | 167 | proposed_replacement | (Proposed replacement: "Prior studies have examined LLM as a judge evaluation in clinical generative AI, with agreement varying by rubric and setting [9,10].") |
| AN003 | 171 | proposed_replacement | (Proposed replacement: "The C versus D comparison did not show a significant directional preference (19 versus 16; P = .736). Condition D added deterministic inline processing and the full POST hook layer to Condition C, so this cumulative comparison cannot isolate the effect of any single component. Some breast cancer rules concerned ER, PR, and HER2 status and were not applicable to PDAC.") |
| AN004 | 205 | reference_recommendation | (Reference recommendation: References 13 and 14 are not cited in the current text and could be removed. Reference 15 could be retained if VALID is cited after "expert reference standards.") |

## Preservation checks

- Physician content blocks retained: 119
- Direct copyedit occurrences: 18
- Separate editorial annotations: 4
- Tables retained: 8
- Figure placeholders retained: 6
- Figure legends retained: 6
- References retained: 15
- TITLE, AUTHORS, and ABSTRACT TODO blocks retained: 3
- All numeric tokens are identical before and after direct copyediting.
- Table structure, row order, figure order, section order, result values, and Supplement content are unchanged except for the explicitly listed single prompt terminology edits.

## Unified diff of direct copyedits

Editorial annotations are excluded from this diff so that every direct change to physician text is visible.

```diff
--- WORKSHOP_PAPER_EDITABLE.physician-original.txt
+++ WORKSHOP_PAPER_WORD_SOURCE.txt (annotations excluded)
@@ -13,11 +13,11 @@
 
 It is estimated that about 80% of healthcare data are unstructured [1], with a large proportion of clinically relevant information encoded in free-text narratives [2]. Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and changes in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real-world practice, laboratory, imaging, and genomic results copied from earlier notes lengthen notes and further complicate the extraction task. A study of outpatient documentation across a decade found that notes became longer and more redundant, with much of the increase attributable to templates or copied and pasted text [3]. LLMs must therefore not only identify the relevant information, but also determine its relationship to the patient’s current diseases.
 
-LLMs can extract information from clinical notes, but their outputs can vary. They may omit details, confuse past findings with current findings, or add information that is not supported by the original note [4,5]. Commercial models perform better on some clinical extraction tasks, but sending protected health information (PHI) to externally hosted services raises privacy and governance concerns[4]. Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction[2]. Although manual review can identify extraction errors, reviewing every output is impractical at scale. These challenges highlight the need for an LLM-based extraction workflow that improves reliability while preserving data privacy and minimizing the need for manual review.
+LLMs can extract information from clinical notes, but their outputs can vary. They may omit details, confuse past findings with current findings, or add information that is not supported by the original note [4,5]. Commercial models perform better on some clinical extraction tasks, but sending protected health information (PHI) to externally hosted services raises privacy and governance concerns [4]. Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction [2]. Although manual review can identify extraction errors, reviewing every output is impractical at scale. These challenges highlight the need for an LLM-based extraction workflow that improves reliability while preserving data privacy and minimizing the need for manual review.
 
 How an LLM is used may be as important as the model itself. Additional training can adapt model weights to a clinical task. When the weights are unchanged, prompt design, task decomposition, verification, and clinical rules can organize the extraction process. Huang et al. iteratively refined prompts on expert curated lung pathology data and later applied the same prompt engineering strategy to osteosarcoma reports. Hein et al. used a human in the loop process to refine prompts, output schemas, and an error ontology, then applied the resulting templates to breast and prostate pathology reports [5,6]. These studies focused on pathology reports and prompt or schema refinement. It remains unclear whether a broader multistep workflow developed from physician feedback can be used to improve note extraction. 
 
-To address this gap, we built a locally deployable inference harness around a frozen LLM for extracting information from real-world unstructured oncology notes. The harness divides extraction into smaller tasks and applies checks and clinical rules without changing the model weights. We developed the breast cancer harness with physician input. We then adapted it to pancreatic ductal adenocarcinoma (PDAC) notes using LLM-assisted review and regression tests without physician review of the PDAC development outputs. Four physicians compared the final harness with a single prompt baseline that used the same model. Our results showed that the output from the harness was favored over that from a single-prompt baseline.
+To address this gap, we built a locally deployable inference harness around a frozen LLM for extracting information from real-world unstructured oncology notes. The harness divides extraction into smaller tasks and applies checks and clinical rules without changing the model weights. We developed the breast cancer harness with physician input. We then adapted it to pancreatic ductal adenocarcinoma (PDAC) notes using LLM-assisted review and regression tests without physician review of the PDAC development outputs. Four physicians compared the final harness with a single prompt baseline that used the same model. Our results showed that the output from the harness was favored over that from a single prompt baseline.
 
 METHODS
 
@@ -25,7 +25,7 @@
 
 The study comprised two development stages followed by an independent physician evaluation. In Stage 1, a physician and an investigator developed the breast cancer inference harness through review of selected outputs. In Stage 2, the investigator adapted the harness to pancreatic ductal adenocarcinoma (PDAC) using LLM review and regression tests, without physician review of the PDAC development outputs (Figure 1A-C). Four physicians then compared the final systems on an untouched holdout set.
 
-The final harness was compared with a sinlge prompt baseline using the same frozen model and field schema on an untouched holdout set of 40 CORAL notes annotated by experts. The set included 20 breast cancer notes and 20 PDAC notes. Four physicians completed evaluations for each field across both cancer types. System identities were not disclosed.
+The final harness was compared with a single prompt baseline using the same frozen model and field schema on an untouched holdout set of 40 CORAL notes annotated by experts. The set included 20 breast cancer notes and 20 PDAC notes. Four physicians completed evaluations for each field across both cancer types. System identities were not disclosed.
 
 We separately conducted an ablation study of four cumulative harness configurations using an OpenAI LLM judge (Figure 1D; Table 2).
 
@@ -35,7 +35,7 @@
 
 Baseline model and model development
 
-We used Qwen2.5-32B-Instruct [8] with AWQ quantization, served locally through vLLM. The baseline used the complete note and all extraction questions in single prompt and requested the full target schema in one response. Each note was processed in one chat completion request. This describes the application-level request and does not make a claim about internal server execution or forward passes. In the physician review, the baseline and harness used the same field definitions and output contract.
+We used Qwen2.5-32B-Instruct [8] with AWQ quantization, served locally through vLLM. The baseline used the complete note and all extraction questions in a single prompt and requested the full target schema in one response. Each note was processed in one chat completion request. This describes the application-level request and does not make a claim about internal server execution or forward passes. In the physician review, the baseline and harness used the same field definitions and output contract.
 
 Inference Harness
 
@@ -81,7 +81,7 @@
 See Table 2 for the 4 conditions.
 
 Table 2. Ablation study conditions
-A	A single-prompt baseline with one inference request per note.
+A	A single prompt baseline with one inference request per note.
 B	Decomposed extraction with separate prompts for each field, routing across fields, bounded Assessment and Plan retry, and the shared format and schema safeguards.
 C	Condition B plus the semantic, faithfulness, and temporal verification gates.
 D	Condition C plus deterministic inline processing and the complete POST hook layer used by the full inference harness.
@@ -128,7 +128,7 @@
 
 Figure 2. Physician preferences across all judgments. Panel A shows harness preferred, tied, and baseline preferred judgments overall and by cancer type. Panel B shows pooled harness preference in blue text and individual physician values as circles among judgments without ties. Panel B values are unadjusted.
 
-The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast cancer estimate was 81.66% (95% CI, 76.02% to 86.21%), the pancreatic cancer estimate was 78.15% (95% CI, 71.47% to 83.62%). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p < 0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p < 0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p < 0.05). See Table 5.
+The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast cancer estimate was 81.66% (95% CI, 76.02% to 86.21%). The pancreatic cancer estimate was 78.15% (95% CI, 71.47% to 83.62%). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p < 0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p < 0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p < 0.05). See Table 5.
 
 Table 5. Adjusted probability of inference harness preference among directional judgments
 Scope	Directional judgments	Harness	Baseline	Note clusters	Adjusted probability (95% CI)	Adjusted odds ratio (95% CI)	P value
@@ -148,7 +148,7 @@
 
 Performance across Clinical Extraction Fields
 
-We examined physician preferences across the seven prespecified clinical extraction fields. In all the 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all 7 fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast cancer type and receptor status, where ties were more common (Figure 4). The adjusted point estimate exceeded 50% for each field; the 95% CI for breast cancer type and receptor status included 50% (62.4%; 95% CI, 38.6% to 81.5%) (Figure 4C).
+We examined physician preferences across the seven prespecified clinical extraction fields. Across all 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all 7 fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast cancer type and receptor status, where ties were more common (Figure 4). The adjusted point estimate exceeded 50% for each field; the 95% CI for breast cancer type and receptor status included 50% (62.4%; 95% CI, 38.6% to 81.5%) (Figure 4C).
 
 [Insert Figure 4 here]
 
@@ -164,11 +164,11 @@
 
 In this study, the inference harness was preferred over a single prompt baseline using the same underlying LLM. This advantage was observed in both breast and pancreatic cancer and remained consistent in sensitivity analyses. These findings suggest that a structured inference harness can improve the clinical quality of extracted information, as reflected in physician preferences, without additional model training. The note level results also indicate that the observed preference was not driven by a small number of clinical notes.
 
-Structured extraction can convert long, complex, free-text notes into a compact set of fields that can be checked, searched, and reused downstream. In oncology, this is useful as clinical notes can be complex and contain multiple diagnoses, prior and future treatments, and changes in disease status.Huang et al. [5] demonstrated that iterative prompt engineering with ChatGPT using an API could improve extraction of free-text pathology notes. Similarly, Hein et al. [6] showed that iterative refinement guided by physicians and systematic error analysis could improve structured extraction and could be adapted to other cancer types. Our study built on these studies by using physician feedback to identify recurring errors and develop clinical rules during breast cancer note development. Rather than producing a large, supervised training set, we used the feedback to develop prompts for each field, verification steps, and clinical rules. This approach incorporated physician knowledge into the inference harness for reuse in subsequent extraction tasks. Instead of asking physicians to perform iterative evaluation, we used ChatGPT to organize the established criteria from breast cancer development, and used Qwen2.5-32B as a judge. Prior studies have showed high concordance between LLM-as-judge evaluations and human assessments on text-comparison criteria such as faithfulness and completeness.[9,10]. The final harness was preferred over the baseline by independent physicians in pancreatic cancer. This finding suggests that criteria established through physician feedback may be transferable to other cancer types and may reduce the need for repeated physician review during development.
+Structured extraction can convert long, complex, free-text notes into a compact set of fields that can be checked, searched, and reused downstream. In oncology, this is useful as clinical notes can be complex and contain multiple diagnoses, prior and future treatments, and changes in disease status. Huang et al. [5] demonstrated that iterative prompt engineering with ChatGPT using an API could improve extraction of free-text pathology notes. Similarly, Hein et al. [6] showed that iterative refinement guided by physicians and systematic error analysis could improve structured extraction and could be adapted to other cancer types. Our study built on these studies by using physician feedback to identify recurring errors and develop clinical rules during breast cancer note development. Rather than producing a large, supervised training set, we used the feedback to develop prompts for each field, verification steps, and clinical rules. This approach incorporated physician knowledge into the inference harness for reuse in subsequent extraction tasks. Instead of asking physicians to perform iterative evaluation, we used ChatGPT to organize the established criteria from breast cancer development, and used Qwen2.5-32B as a judge. Prior studies have shown high concordance between LLM-as-judge evaluations and human assessments on text-comparison criteria such as faithfulness and completeness [9,10]. The final harness was preferred over the baseline by independent physicians in pancreatic cancer. This finding suggests that criteria established through physician feedback may be transferable to other cancer types and may reduce the need for repeated physician review during development.
 
 One recurring challenge we identified was the interpretation of temporal relationships in oncology notes, as reported by prior studies [11,12]. Although the LLM could identify treatment responses documented in the clinical history, it sometimes failed to associate them with the correct disease status. For example, a response to a previous treatment could be incorrectly attributed to the current regimen. This was particularly important for patients who had received multiple lines of therapy, where each treatment response must be associated with treatment history. To reduce this risk, we first revised the prompt to instruct the model to distinguish evidence recorded before and after a regimen change. Beyond the prompt, the response extraction step used the complete note together with structured context on current medications and recent treatment changes. The extracted response then underwent semantic and source support checks. These temporal controls were embedded across the prompt, structured context, and downstream checks, which made their independent contribution difficult to isolate. The physician ratings therefore support the complete workflow but do not determine how much of the observed preference came from the temporal controls alone.
 
-Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure S2). Adding decomposed extraction alone did not produce an overall improvement while adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, the C versus D comparison did not show significant improvement from adding deterministric post-proccessing rules, this suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce tradeoffs, especially when the terminology or treatment differs between cancer types. For example, some rules developed for breast cancer were specific to hormone status, which are used to classify breast cancer subtypes but are not applicable to PDAC.
+Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure S2). Adding decomposed extraction alone did not produce an overall improvement while adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, the C versus D comparison did not show significant improvement from adding deterministic post processing rules. This suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce tradeoffs, especially when the terminology or treatment differs between cancer types. For example, some rules developed for breast cancer were specific to hormone status, which is used to classify breast cancer subtypes but is not applicable to PDAC.
 
 This study has several limitations. First, our primary outcome was physician preference rather than absolute extraction accuracy. A tie could indicate that both systems produced acceptable outputs or that both contained errors. Although physician ratings consistently favored the harness, the evaluation included only four physicians and 40 clinical notes from two cancer types within a dataset from a single source. It is worth noting that these notes were real clinical notes rather than synthetic cases or narratives generated by models. They contain repeated information copied from earlier notes, uncertain findings, treatment plans, or even typographical errors that make the extraction task challenging. Nevertheless, further studies are needed to investigate whether the results can be generalized to other cancer types or language models. The observed preference consistently favored the complete harness output as presented in both cancer types and sensitivity analyses. The current design cannot separate the effects of extraction quality, the surrounding workflow, source attribution, and display position. Second, the exploratory ablation was evaluated by a single LLM judge that was not a physician and compared cumulative stages rather than individual components. Therefore, it could not determine the independent contribution of each verification step or clinical rule. Future studies should evaluate extraction accuracy against expert reference standards, include larger and more diverse clinical datasets, and assess the generalizability of the harness across different open-weight models.
 
@@ -216,7 +216,7 @@
 
 The harness first used regular expressions to locate the Assessment and Plan section. If no section was found, the model made up to three attempts to reproduce the relevant source text. Each candidate was checked against the original note. If all attempts failed, the harness skipped only the plan branch and continued extraction from the complete note.
 
-Verification and postprocessing order
+Verification and post processing order
 
 Each model output for a field passed through the following gates in order.
 
@@ -279,7 +279,7 @@
 
 [Insert Supplementary Figure S2 here]
 
-Supplementary Figure S2. Panel A shows four cumulative configurations. A was the single-prompt baseline with one inference request per note. B added field decomposition. C added semantic refinement, faithfulness review, and temporal verification. D added deterministic inline processing and the complete POST hook layer. Panel B reports three adjacent comparisons across 120 judgments. One blinded nonphysician OpenAI LLM judge evaluated all comparisons. Ties were excluded from two sided exact sign tests. The three unadjusted P values were 1.000, 1.05 × 10⁻⁵, and .736.
+Supplementary Figure S2. Panel A shows four cumulative configurations. A was the single prompt baseline with one inference request per note. B added field decomposition. C added semantic refinement, faithfulness review, and temporal verification. D added deterministic inline processing and the complete POST hook layer. Panel B reports three adjacent comparisons across 120 judgments. One blinded nonphysician OpenAI LLM judge evaluated all comparisons. Ties were excluded from two sided exact sign tests. The three unadjusted P values were 1.000, 1.05 × 10⁻⁵, and .736.
 
 Supplementary Table S3. Sensitivity analysis of the overall directional GEE after excluding each physician
 Excluded physician	Directional judgments	Notes	Adjusted probability (95% CI)	Adjusted odds ratio (95% CI)	P value
```
