# A failure-mode-driven inference harness for oncology note extraction

**Pilot report draft for clinical collaborator review**

Version 0.2, September 2026

Authors: [TODO]

Affiliations: [TODO]

Target venue and format: [TODO]

This draft assumes that the planned multi-oncologist analysis confirms the direction of the current pilot and reaches statistical significance. Bracketed result text is a placeholder, not a completed finding. It must be replaced with the final analysis before submission.

Short version for clinical review: both systems use the same language model. The baseline asks the model to extract everything in one pass. The harness divides the task into smaller clinical questions, checks the answers, applies narrow oncology rules to recurring errors, and links each result to supporting text from the note.

## Questions for the clinical collaborator

Please focus on the following points during this review. These questions will be removed from the submitted manuscript.

1. Are the seven core clinical fields the right ones to call clinically important?
2. Are our interpretations of active therapy, stage, regional versus distant metastasis, treatment response, receptor status, and molecular results clinically sound?
3. Does the Discussion explain the observed strengths and weaknesses without claiming more than the study shows?
4. Which examples would be most persuasive to an oncology audience?
5. Are any clinical statements inaccurate, overstated, or missing necessary context?
6. Is our distinction between clinician-created ground truth and clinician evaluation of final outputs fair and clinically meaningful?

## Abstract

### Background

Large language models can extract structured information from clinical notes, but a single prompt often confuses current and historical treatment, suspected and confirmed disease, and completed findings and future plans. These errors are difficult to detect because the resulting text remains fluent. Oncology notes are a demanding test case because they combine pathology, imaging, treatment history, current therapy, response assessment, and conditional plans across a long clinical timeline.

### Objective

To test whether an inference harness can improve oncology information extraction from a frozen, locally served open-weight language model without fine-tuning.

### Methods

We built a failure-mode-driven inference harness around Qwen2.5-32B-Instruct-AWQ. The harness decomposes extraction into field-specific tasks, passes selected information between dependent tasks, applies a five-stage verification cascade, uses deterministic oncology rules for recurring high-confidence errors, and returns supporting source text. We compared the full harness with a single-prompt baseline using the same model and target field contract on 40 held-out CORAL notes, including 20 breast cancer and 20 pancreatic cancer notes. CORAL contains real, deidentified longitudinal oncology notes with expert annotations rather than synthetic cases or internet vignettes. The planned clinical evaluation uses identity-masked A/B comparisons by [FINAL N] oncologists. The primary analysis will account for repeated judgments by evaluator, note, and field.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. The harness led in six of seven core categories. In the first completed oncologist evaluation of 20 breast cancer notes, the harness was preferred in 84 of 278 required-field comparisons, the baseline in 22, and 172 were ties. Among directional judgments, 79.2% favored the harness. The largest differences involved active anticancer medications and medication planning.

[FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the harness was preferred in FINAL X, the baseline in FINAL Y, and FINAL Z were ties. The adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### Conclusions

An inference harness can improve the reliability of a frozen local model by addressing recurrent clinical failure modes at inference time. If the multi-oncologist result confirms the current pilot, the findings will support inference engineering as a practical alternative to fine-tuning for structured oncology extraction.

## 1. Introduction

Most clinically useful information in oncology is still recorded in free text. A progress note may contain the diagnosis, pathology, receptor status, treatment history, toxicities, response, and next steps, but these facts are spread across sections and timepoints. Manual review is slow, and conventional extraction systems require substantial annotation and task-specific development. The attraction of large language models is straightforward: one model can read many note styles and return structured fields without a new supervised model for every question.

Research activity has moved faster than routine clinical adoption. A 2025 scoping review identified 24 studies of language-model-based oncology information extraction, but also found that external validation and real-world workflow integration remained limited [1]. Clinical deployment has a higher bar than a demonstration on a benchmark. A useful system must preserve uncertainty, distinguish current care from historical events, avoid unsupported facts, and produce results that a clinician can trace back to the note.

Longitudinal oncology notes expose weaknesses that are easy to miss in simpler extraction tasks. A medication list can contain anticancer therapy, supportive treatment, chronic home medications, discontinued drugs, and therapies that are only being discussed. Regional lymph nodes must not be labeled as distant metastases. A suspicious lesion awaiting biopsy must not become confirmed stage IV disease. Tumor growth before treatment is not evidence of failure of a regimen that has just started. A model can mention medically relevant information and still place it in the wrong field or timepoint.

Several strategies have been used to improve clinical extraction. Large proprietary models can perform zero-shot extraction but still omit note-specific details and hallucinate [2]. Local open-weight models reduce dependence on external services, but performance varies with model size and prompt design [3,8]. Fine-tuned hybrid systems can achieve strong performance and external validation, although they require large labeled datasets and model training [5]. Hierarchical prompting, validation layers, and retry mechanisms also improve extraction without conventional fine-tuning [6,7]. These studies establish prior work for the individual techniques used in modern extraction systems.

Human involvement also differs across studies. Clinicians or medical experts often create gold-standard annotations, resolve disagreements, or guide terminology selection. That work establishes whether a model matches a reference label. It does not answer whether a practicing oncologist considers one complete system output more faithful, complete, and clinically useful than another when both contain partly correct information. In the supplemental methods table of the 24-study scoping review, most papers reported automatic performance metrics against labels. Only two explicitly reported Likert-style human ratings of model outputs, and both evaluated radiology tasks [1]. We did not identify a field-level, identity-masked comparison of complete extraction systems by practicing oncologists in that review.

Our approach treats recurring model errors as engineering targets. We use the term inference harness for the software layer that controls how a frozen model is prompted, checked, corrected, and linked to evidence. The model itself is unchanged. When review identifies a repeated failure, such as importing a stopped drug into current therapy or turning suspected metastasis into confirmed disease, the system receives a narrow correction through prompt instructions, verification logic, or a deterministic clinical rule. Each correction can be logged and regression-tested.

The harness is more than a long prompt or retrieval step. Different fields follow different extraction routes. Selected outputs pass into later tasks only when they provide relevant clinical context. Verification stages run after generation, and deterministic hooks enforce narrow oncology constraints across related fields. The system records these interventions and links final values to note evidence. Among the closest studies we reviewed, we did not identify an evaluation of this full combination on real longitudinal oncology notes with a same-model baseline and direct oncologist comparison.

We tested whether this approach improves extraction under a controlled comparison. The harness and baseline use the same Qwen2.5-32B model and the same target schema. The main difference is the inference process surrounding the model. We focus on seven clinical questions that require more than surface entity recognition: active anticancer therapy, stage, distant metastasis, regional or overall metastatic involvement, treatment response, breast cancer type and receptor status, and completed molecular or genetic results.

We hypothesized that the harness would outperform the single-prompt baseline overall and that the largest gains would occur in fields directly addressed by temporal checks, clinical classification rules, and cross-field consistency checks. We also expected many ties because the same base model should answer straightforward questions similarly in both conditions.

## 2. Methods

### 2.1 Study design

This is a pilot evaluation of a structured inference workflow for oncology information extraction. The study includes a same-model technical comparison on 40 held-out notes and a planned multi-oncologist A/B evaluation. At the time of this draft, one oncologist has completed the breast cancer portion of the clinical evaluation.

### 2.2 Dataset

We used CORAL, an expert-curated dataset of deidentified breast and pancreatic cancer progress notes [2]. The dataset contains 200 unannotated development notes and 40 expert-annotated test notes. The held-out comparison used all 40 test notes: 20 breast cancer notes and 20 pancreatic cancer notes.

CORAL is publicly available, but the records are real clinical notes rather than web-derived questions, synthetic cases, or model-generated narratives. They retain the repeated histories, copied-forward content, uncertain findings, and conditional plans that make longitudinal oncology extraction difficult. The value of this dataset is clinical realism and expert annotation, not size. Several related studies use much larger institutional cohorts, while others use narrower procedure or pathology reports or synthetic oncology notes.

Development and evaluation had separate roles. Development notes were used to identify recurrent error patterns and refine the harness. The 40 test notes were used for the matched comparison. No model-weight training or fine-tuning was performed.

### 2.3 Base model and baseline

Both conditions used Qwen2.5-32B-Instruct-AWQ served locally through vLLM [9]. The baseline made one model call per note and returned the complete target schema. It did not use task decomposition, verification gates, retries, dictionaries, or deterministic post-processing. The matched baseline and harness used the same field definitions and output contract.

### 2.4 Inference harness

The harness uses several stages because the target fields do not all require the same context or reasoning.

First, field-specific prompts independently extract visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Dependent prompts then receive selected earlier results. For example, stage and metastatic status provide context for treatment intent, while current therapy and clinical findings provide context for response assessment. Plan fields are extracted primarily from the Assessment and Plan section.

Each model output passes through five verification stages:

1. JSON formatting repair when parsing fails.
2. Schema validation to detect incorrect or leaked keys.
3. Specificity and semantic alignment checks to identify vague or off-task answers.
4. Faithfulness trimming that removes clearly unsupported or contradictory content while retaining supported information.
5. Temporal filtering that removes completed or historical events from future-plan fields.

The deterministic layer addresses recurrent errors with high-confidence clinical rules. These rules distinguish regional nodes from distant disease, suspected from confirmed metastasis, active anticancer therapy from supportive or home medication, and current response from pretreatment change. The system also returns note excerpts that support each extracted value.

| Harness component | Failure mode addressed | Example |
|---|---|---|
| Field-specific extraction | One large prompt drops or mixes fields | Separate active medication extraction from treatment plans |
| Dependency-aware context | Related fields contradict one another | Use metastatic status when interpreting stage and response |
| Semantic verification | A medically related answer does not answer the field | Remove a future treatment plan from current response |
| Faithfulness trimming | The model adds an unsupported conclusion | Preserve a lesion as suspected when biopsy is pending |
| Temporal filtering | Completed results appear as future plans | Remove a completed scan from imaging plan |
| Drug and context rules | Medication names are classified without clinical context | Separate anticancer drugs from home and supportive medications |
| Cross-field clinical rules | Stage, nodes, and distant disease become inconsistent | Keep axillary nodes regional rather than distant |
| Source attribution | A reviewer cannot trace an extracted value | Return the supporting sentence from the note |

### 2.5 Development and regression testing

Development followed a repeated error-review cycle. Reviewers compared each extracted field with the complete source note, classified the error, and identified whether it arose from the prompt, verification stage, or deterministic layer. Corrections were written as general rules rather than sample-specific answers. Each high-impact change was tested on the affected samples and additional previously correct controls.

The pipeline logs the original model output, the action of each verification stage, and deterministic corrections. This makes the behavior more inspectable than a single final model response, although it does not make the language model itself fully explainable.

### 2.6 Prespecified clinical fields

The seven core fields were selected before the matched comparison:

1. Which anticancer drugs is the patient actively receiving?
2. What is the current cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer responding to the current treatment?
6. What is the breast cancer type and ER/PR/HER2 status?
7. What completed molecular or genetic results are documented?

The oncologist instrument also included genetic testing plans, supportive medications, procedure plans, imaging plans, laboratory plans, medication plans, and recent treatment changes. Laboratory summary and general clinical findings were optional and excluded from the primary clinician analysis.

### 2.7 Technical evaluation

The complete matched audit contained 260 applicable note-field comparisons. A source-grounded LLM-assisted review process read the source note and both outputs for each comparison. Each result was classified as harness better, baseline better, or tie. All reported harness losses and contested high-severity findings were rechecked against the source note. This audit was used for development and technical error analysis. It was not treated as a replacement for clinician evaluation.

Four high-impact failures found during the complete v2.2 audit were subsequently repaired. The affected cases and clean controls were rerun as a targeted regression set. These results are reported separately because the targeted set is not a new full-cohort estimate.

### 2.8 Oncologist evaluation

The clinical evaluation presents the source note and two structured outputs through an identity-masked A/B interface. The evaluator selects A better, B better, or tie for each field. The interface does not reveal which output came from the harness.

The first oncologist completed the 20 breast cancer cases. The export contained 278 of 280 required judgments. Two ratings were missing and one extraneous pancreatic entry was excluded. The raw export was preserved. The scoring template retained stale filenames after the displayed results were updated. The project owner confirmed that the oncologist reviewed the newer outputs, but the exact artifact hashes must be inserted before submission: [TODO].

The planned final study will include [TARGET: 5] oncologists. The primary analysis will compare harness and baseline preference among directional ratings with a mixed-effects logistic model that includes evaluator, note, and field as grouping factors. Ties will be reported separately and included in a sensitivity analysis. We will report the effect estimate, 95% confidence interval, two-sided p value, and agreement across evaluators. The final statistical specification will be reviewed before unblinding the aggregate results.

## 3. Results

### 3.1 Final multi-oncologist analysis

[PLACEHOLDER TABLE: Final multi-rater results]

| Outcome | Harness | Baseline | Tie | Effect estimate | 95% CI | p value |
|---|---:|---:|---:|---:|---:|---:|
| All required fields | FINAL | FINAL | FINAL | FINAL | FINAL | FINAL |
| Seven core fields | FINAL | FINAL | FINAL | FINAL | FINAL | FINAL |
| Breast cancer | FINAL | FINAL | FINAL | FINAL | FINAL | FINAL |
| Pancreatic cancer | FINAL | FINAL | FINAL | FINAL | FINAL | FINAL |

[FINAL RESULT TEXT: The harness received significantly more favorable judgments than the single-prompt baseline. The direction of effect was consistent across evaluators and cancer types. Replace this sentence if the final analysis does not support both claims.]

### 3.2 Interim oncologist result

One oncologist has completed 278 required comparisons across the 20 breast cancer notes. The harness was preferred in 84 comparisons, the baseline in 22, and 172 were ties. Among the 106 directional judgments, 79.2% favored the harness. In an exploratory per-note summary, the harness had more field wins in 18 cases, the baseline in one, and one case was tied.

Across the seven core fields, the harness recorded 51 wins, the baseline 12, and 77 ties. Six field categories favored the harness and one was even.

| Core field | Harness | Baseline | Tie | Net advantage |
|---|---:|---:|---:|---:|
| Active anticancer medications | 17 | 1 | 2 | +16 |
| Stage | 8 | 1 | 11 | +7 |
| Distant metastasis | 3 | 1 | 16 | +2 |
| Regional or overall metastasis | 9 | 1 | 10 | +8 |
| Treatment response | 3 | 1 | 16 | +2 |
| Tumor type and receptor status | 5 | 5 | 10 | 0 |
| Completed molecular or genetic results | 6 | 2 | 12 | +4 |
| **Overall** | **51** | **12** | **77** | **+39** |

Medication planning, which was outside the seven core fields, also favored the harness by 13 wins to none, with seven ties. Laboratory planning was the only required field with a negative margin, although the difference was one judgment. Procedure planning and tumor type or receptor status were even.

### 3.3 Complete matched technical audit

Across 260 applicable core comparisons, the harness was preferred 66 times, the baseline 28 times, and 166 were ties. The harness had a positive margin in six of seven categories. Stage was the only category with a negative margin in the complete v2.2 audit.

| Core field | Harness | Baseline | Tie | Net advantage |
|---|---:|---:|---:|---:|
| Active anticancer treatment | 8 | 0 | 32 | +8 |
| Stage | 6 | 8 | 26 | -2 |
| Distant metastasis | 11 | 3 | 26 | +8 |
| Regional or overall metastasis | 14 | 4 | 22 | +10 |
| Treatment response | 13 | 6 | 21 | +7 |
| Breast cancer type and receptors | 8 | 5 | 7 | +3 |
| Completed molecular or genetic results | 6 | 2 | 32 | +4 |
| **Overall** | **66** | **28** | **166** | **+38** |

### 3.4 Targeted repair evaluation

Four high-impact failures identified in the v2.2 audit were repaired with conservative rules. Across 51 applicable comparisons in six affected samples and two controls, the repaired harness recorded 29 wins, no baseline wins, and 22 ties. No P0 error remained in this targeted set, and neither control developed a detected core regression. These targeted results show that the identified errors were repairable. They do not replace the full 40-note result.

### 3.5 Qualitative example

The oncologist provided two written comments. In one imaging-plan comparison, the clinician preferred the harness because the baseline summarized completed findings but did not identify the planned PET/CT. The baseline extracted medically relevant imaging information, but it answered the wrong temporal question. This example illustrates why conventional entity overlap does not fully measure extraction quality in longitudinal notes.

## 4. Discussion

### 4.1 Main interpretation

[FINAL OPENING: The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline.]

The current pilot already shows a clear directional pattern. Most comparisons were ties, which is expected when both systems use the same strong base model. When the oncologist identified a meaningful difference, the harness was preferred nearly four times as often as the baseline. The gains were concentrated in fields that require temporal interpretation and clinical classification rather than simple copying.

This pattern matters more than a broad improvement across every field. The harness was designed to preserve correct base-model answers and intervene when a known failure mode appears. A high tie rate with a strong directional advantage is consistent with that design. It suggests selective correction rather than wholesale rewriting of the model output.

### 4.2 Which questions were difficult for the model?

The first oncologist's ratings separate relatively direct extraction from questions that require clinical context.

Straightforward facts often produced ties. Both systems could usually identify an explicitly stated imaging result, procedure, or receptor value. The harder problems involved deciding what the fact meant in the current clinical context.

Active anticancer medication showed the largest difference. This field requires the model to distinguish treatment from chronic home medications, supportive drugs, discontinued regimens, and future options. Medication planning also strongly favored the harness because future actions must remain separate from current treatment and recent changes.

Stage and metastatic involvement require related decisions. The model must distinguish regional lymph nodes from distant spread, preserve uncertainty for lesions awaiting confirmation, and reconcile metastatic status with stage. The harness was built to check these relationships across fields rather than extract each label in isolation.

Treatment response remained more difficult. A note may include old progression, current symptoms, stable imaging, tumor-marker trends, and a newly started regimen. Determining which evidence reflects response to the current treatment requires a timeline, not keyword recognition. The smaller margin in this field is consistent with that difficulty.

Tumor type and receptor status was the only core category that did not favor either system in the first clinician review. These values are often stated explicitly, so the baseline can perform well. The field also becomes difficult when a note contains bilateral disease, historical and recurrent specimens, or discordant receptor results. This remains an area for clinical review rather than a claimed strength.

### 4.3 How the observed pattern relates to the harness

The current study does not include a complete component ablation, so it cannot assign each improvement to one module with certainty. The field-level pattern is nevertheless consistent with the intended function of several components.

The active-medication result is consistent with the dedicated medication prompt, the oncology drug dictionary, contextual classification of supportive versus home medications, and temporal filtering. The medication-plan result is consistent with extracting plans from the Assessment and Plan section and removing already completed actions. The stage and metastasis results are consistent with cross-field context and deterministic rules that separate regional from distant disease and suspected from confirmed findings.

The high number of ties provides a useful counterpoint. Deterministic rules did not produce an apparent advantage in every field, and laboratory planning slightly favored the baseline. This makes a simple explanation based on output length or a global preference for the harness less convincing. The strongest differences occurred where the system had explicit safeguards.

These associations support the proposed mechanism, but they do not prove it. A future component study should compare a single prompt, decomposed prompts alone, prompts plus verification, and the complete harness. That experiment would show how much each layer contributes.

### 4.4 Comparison with related work

The closest studies use different combinations of data, model adaptation, and human review. Several include clinicians, but their role is usually to create a reference standard or guide model development. Fewer studies ask oncology specialists to compare final system outputs directly.

| Study | Data and task | Role of clinical experts | Main approach | Difference from this study |
|---|---|---|---|---|
| Sushil et al., CORAL [2] | 40 real breast and pancreatic cancer notes; broad oncology schema | Expert annotation and manual evaluation of a subset | Zero-shot GPT-4, GPT-3.5-turbo, and FLAN-UL2 | Same dataset and clinical scope. It established the benchmark and documented omissions and hallucinations. It did not test a failure-mode-driven harness against the same frozen model. |
| Wiest et al. [3] | 500 MIMIC histories; five binary clinical features | Three blinded medical experts created consensus ground truth | Local Llama 2 with constrained JSON and prompt variants | Strong expert validation and local deployment, but the targets were five mostly binary features outside oncology. Experts supplied ground truth rather than comparative preference ratings of two full systems. |
| Bhattarai et al. [4] | 13,646 notes from 63 patients with lung cancer; four longitudinal phenotypes | Two subject-matter experts annotated the cohort and adjudicated disagreements | GPT, open-model, and rule-based comparison | Larger longitudinal corpus with expert labels, but fewer target phenotypes and no same-model comparison isolating the surrounding workflow. |
| Tariq et al. [5] | 26,692 breast cancer patients internally and 162 externally; treatment timelines | Cancer-registry labels supplied the outcome; clinical experts curated treatment concepts and codes | UMLS parser plus a fine-tuned question-answering model | Much stronger scale and external validation. It requires supervised fine-tuning and focuses on five treatment pathways rather than broad field-level extraction. |
| Dao et al. [6] | 220 development and 200 validation right-heart-catheterization notes | One pulmonary vascular disease expert created the validation ground truth and guided development | Local open model, structured preload, validation, and retry | The closest workflow architecture. The task was numerical extraction from procedure notes, and the system did not include oncology-specific field routing and cross-field clinical hooks. |
| Zhang et al., mCODEGPT [7] | 1,000 synthetic oncology notes; 49 mCODE entities | Human reviewers checked automated matches after generation | Hierarchical prompting versus single-step prompting | Direct evidence that prompt hierarchy helps, but the notes were synthetic and the study did not use blinded oncologist comparison of final outputs. |
| Grothey et al. [8] | 579 prostate pathology reports in German and English | A medical doctoral researcher annotated reports under an attending pathologist's supervision | Multiple open and proprietary models, prompt and quantization tests | Larger expert-labeled benchmark, but pathology reports are more regular than longitudinal clinic notes and evaluation relied on reference labels rather than end-user preference. |

These papers show that clinician involvement is not absent from the field. The distinction is where that involvement occurs. Annotation, adjudication, and terminology design provide a reference answer before evaluation. The 24-study scoping review was dominated by precision, recall, F1, AUC, and accuracy comparisons. Its methodology table listed only two studies with explicit Likert-style output ratings, both in radiology [1]. Our planned study adds downstream review by practicing oncologists, who inspect the source note and compare the final outputs field by field without seeing system identity. This captures clinical preference when both outputs are partly correct, when uncertainty matters, or when one answer is more complete without being less faithful.

The data contribution also needs precise wording. CORAL is public, but it contains real deidentified longitudinal oncology notes with expert annotation. This differs from web questions, synthetic notes, and narrow report templates. It does not exceed the scale or external validation of the largest institutional studies. Its advantage for this experiment is that the notes preserve the clinical ambiguity our harness is designed to address.

Our technical claim concerns the combination. Prompt engineering, retrieval, guardrails, retries, and hybrid rules already exist in the literature. In the closest studies reviewed here, we did not find the full evaluated combination used in this project: field-specific routing, selective dependency transfer, five verification stages, oncology drug and terminology resources, deterministic clinical hooks, cross-field consistency checks, action logging, source attribution, and a same-model baseline that removes model capability as the main explanation. The contribution is this integrated inference harness and its evaluation, not any one component in isolation.

Several related studies have larger datasets, more annotators, or external validation. Our pilot should not be presented as the first clinical extraction system or the largest evaluation. It can be presented as a focused test of whether a structured, auditable workflow can make the same frozen model more reliable on difficult oncology fields, with direct review by the clinicians who understand those distinctions.

### 4.5 Clinical and technical implications

The findings suggest that some LLM failures in oncology extraction are repeatable enough to address outside the model weights. This is useful when labeled training data are scarce or when a clinical team needs to change a field definition without retraining. A rule that preserves biopsy-pending disease as suspected can be inspected and tested. A prompt-only system offers less control because a wording change may affect unrelated fields.

The term inference harness is appropriate for this system because it includes more than a prompt sequence. It manages task routing, dependencies, verification, deterministic corrections, logging, and source attribution around a frozen model. Clinical workflow is also understandable, but it can imply integration into routine care, which this pilot has not tested. We therefore use inference harness for the technical contribution and evaluation workflow for the study procedure.

The system is also compatible with local deployment. Local operation does not by itself establish privacy compliance or clinical safety, but it allows an institution to retain control of note processing and system updates. The present work evaluates extraction quality, not readiness for autonomous clinical use.

### 4.6 Pilot status and next steps

This pilot is intended to establish whether the effect is large enough and clinically coherent enough to justify a larger study. The first oncologist's ratings support both conditions. The final multi-rater analysis will determine whether the preference generalizes across evaluators. Pancreatic cancer scoring will test whether the same design transfers beyond breast oncology.

External validation remains important. CORAL notes come from one institution and represent two cancer domains. The most informative next dataset would contain longitudinal oncology notes from a different health system, with independent clinical annotation and a field contract fixed before evaluation.

## 5. Limitations

The current clinical result comes from one oncologist and only the breast cancer subset. It cannot establish inter-rater agreement or generalization across oncology specialties. The final manuscript will replace this interim analysis with the planned multi-rater result.

The A/B interface concealed system identity but used fixed left and right positions. The harness output also included source attribution while the baseline did not. Attribution is part of the system being evaluated, but it may influence preference. A future study should randomize side assignment and separately test the effect of attribution.

The technical audit used LLM-assisted reviewers and repeated error analysis. It is useful for identifying failures but is not independent clinical validation. The complete 40-note table represents v2.2, while later high-impact repairs were tested on affected samples and controls rather than a new full run.

The dataset is small and comes from one institution. The harness contains clinical rules derived from observed errors, and some may capture documentation conventions specific to CORAL. The study does not yet include a complete component ablation, another model family, or external notes.

Finally, this study evaluates structured extraction. It does not test patient understanding, treatment decisions, workflow efficiency, or clinical outcomes.

## 6. Conclusion

[FINAL CONCLUSION: In a multi-oncologist identity-masked evaluation, the failure-mode-driven inference harness significantly outperformed a same-model single-prompt baseline for structured extraction from longitudinal oncology notes.]

The current pilot suggests that the largest gains occur when extraction requires temporal reasoning and clinical classification, especially active therapy, medication planning, and metastatic status. The model weights remained frozen. The improvement came from the inference process around the model: narrower tasks, verification, conservative clinical rules, and source attribution. If confirmed by the planned multi-rater analysis, this approach offers a practical way to improve local clinical LLM systems without task-specific fine-tuning.

## References

1. Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. *JMIR Cancer*. 2025. doi:10.2196/65984.
2. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024. doi:10.1056/AIdbp2300110.
3. Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024. doi:10.1038/s41746-024-01233-2.
4. Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024. doi:10.1093/jamiaopen/ooae060.
5. Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. *JCO Clinical Cancer Informatics*. 2025. doi:10.1200/CCI-25-00002.
6. Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025. doi:10.1093/jamiaopen/ooaf097.
7. Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025. doi:10.1038/s43856-025-01116-x.
8. Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025. doi:10.1038/s43856-025-00808-8.
9. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115.
