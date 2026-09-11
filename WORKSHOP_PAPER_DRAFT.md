# A failure-mode-driven inference harness for oncology note extraction

**Pilot report draft for clinical collaborator review**

Version 0.3, September 2026

Authors: [TODO]

Affiliations: [TODO]

Target venue and format: [TODO]

This draft includes two completed oncologist submissions. Both oncologists evaluated the breast-cancer cases, and the second also evaluated the pancreatic-cancer cases. The planned final multi-oncologist model and bracketed inferential results remain placeholders and must be completed before submission.

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

We built a failure-mode-driven inference harness around Qwen2.5-32B-Instruct-AWQ. The harness decomposes extraction into field-specific tasks, passes selected information between dependent tasks, applies a five-stage verification cascade, uses deterministic oncology rules for recurring high-confidence errors, and returns supporting source text. We compared the full harness with a single-prompt baseline using the same model and target field contract on 40 held-out CORAL notes, including 20 breast cancer and 20 pancreatic cancer notes. CORAL contains real, deidentified longitudinal oncology notes with expert annotations rather than synthetic cases or internet vignettes. Two oncologists have completed identity-masked A/B comparisons to date. Both evaluated breast cancer, and one also evaluated pancreatic cancer. The planned final analysis will account for repeated judgments by evaluator, note, and field.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. Across the currently completed clinical evaluations, the harness was preferred in 249 of 817 required-field judgments, the baseline in 39, and 529 were ties. Among the 288 directional judgments, 86.5% favored the harness. The two breast-cancer evaluations together contributed 163 harness preferences, 30 baseline preferences, and 365 ties; exact agreement between oncologists was 82.7% with Cohen's kappa of 0.645. In the pancreatic-cancer evaluation, the harness was preferred 86 times, the baseline 9 times, and 164 comparisons were ties.

[FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the harness was preferred in FINAL X, the baseline in FINAL Y, and FINAL Z were ties. The adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### Conclusions

Two oncologists independently favored the inference harness on breast-cancer extraction, and the first pancreatic-cancer evaluation showed the same direction. These pilot findings support inference engineering as a practical way to improve a frozen local model, while additional oncologists and a prespecified clustered analysis remain necessary for the final claim.

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

This is a pilot evaluation of a structured inference workflow for oncology information extraction. The study includes a same-model technical comparison on 40 held-out notes and an ongoing multi-oncologist A/B evaluation. At the time of this draft, two oncologists have completed the breast-cancer evaluation, and one of them has also completed the pancreatic-cancer evaluation.

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

The first oncologist completed the 20 breast-cancer cases. The export contained 278 of 280 required judgments. Two ratings were missing and one extraneous pancreatic entry was excluded. The second oncologist completed all 280 required breast-cancer judgments and 259 of 260 pancreatic-cancer judgments; `p7 / lab_plan` was missing. Both raw exports were preserved unchanged. The scoring template retained stale filenames after the displayed results were updated. The project owner confirmed that both clinicians reviewed the newer outputs, but the exact hashes of the displayed PL and BL artifacts must be inserted before submission: [TODO].

For the 278 required breast-cancer judgments available from both oncologists, we calculated exact agreement and Cohen's kappa. We also summarized each clinician-by-cancer evaluation separately, because only one oncologist has completed the pancreatic-cancer set. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The planned final study will include [TARGET: 5] oncologists. The primary analysis will compare harness and baseline preference among directional ratings with a mixed-effects logistic model that includes evaluator, note, and field as grouping factors. Ties will be reported separately and included in a sensitivity analysis. We will report the effect estimate, 95% confidence interval, two-sided p value, and agreement across evaluators. The current two-oncologist summaries are interim descriptive analyses; the final statistical specification should be frozen before the remaining ratings are aggregated.

## 3. Results

### 3.1 Current oncologist evaluation

The available clinical evidence comprises three completed clinician-by-cancer evaluations: breast cancer from both oncologists and pancreatic cancer from the second oncologist. Across 817 required-field judgments, the harness was preferred 249 times, the baseline 39 times, and 529 comparisons were ties. The harness received 86.5% of the 288 directional judgments.

| Completed evaluation | Required judgments | Harness | Baseline | Tie | Harness share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 01, breast cancer | 278 | 84 | 22 | 172 | 79.2% |
| Oncologist 02, breast cancer | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 02, pancreatic cancer | 259 | 86 | 9 | 164 | 90.5% |
| **All completed evaluations** | **817** | **249** | **39** | **529** | **86.5%** |

Both clinicians independently favored the harness on the breast-cancer set. Their pooled breast result was 163 harness preferences, 30 baseline preferences, and 365 ties. The first oncologist's per-note result was 18 harness wins, one baseline win, and one tie. The second oncologist's result was 20 harness wins. When both breast ratings were combined within each note, all 20 notes had a positive harness-minus-baseline margin.

The second oncologist's pancreatic-cancer evaluation produced 86 harness preferences, 9 baseline preferences, and 164 ties. The harness won 18 of 20 notes by within-note field margin, and two notes were tied. One required judgment, `p7 / lab_plan`, was missing.

The final mixed-effects result remains pending:

> [FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### 3.2 Inter-rater agreement and core fields

The two oncologists shared 278 required breast-cancer comparisons. They gave the same verdict on 230, for 82.7% exact agreement and Cohen's kappa of 0.645. Of the 48 disagreements, only nine were direct reversals between harness and baseline: seven changed from a baseline preference by the first oncologist to a harness preference by the second, and two changed in the opposite direction. The remaining disagreements involved a tie from one evaluator.

Across all completed evaluations, the seven prespecified core categories contributed 400 applicable judgments. The harness received 156 preferences, the baseline 18, and 226 were ties. The harness therefore accounted for 89.7% of the 174 directional core judgments, and every core category had a positive aggregate margin.

| Core field | Harness | Baseline | Tie | Net advantage |
|---|---:|---:|---:|---:|
| Active anticancer medications | 50 | 1 | 9 | +49 |
| Stage | 24 | 2 | 34 | +22 |
| Distant metastasis | 8 | 2 | 50 | +6 |
| Regional or overall metastasis | 35 | 1 | 24 | +34 |
| Treatment response | 18 | 1 | 41 | +17 |
| Breast cancer type and receptors | 8 | 6 | 26 | +2 |
| Completed molecular or genetic results | 13 | 5 | 42 | +8 |
| **Overall** | **156** | **18** | **226** | **+138** |

Active anticancer medication and regional or overall metastatic involvement produced the largest and most consistent margins. Medication planning, outside the seven core categories, totaled 31 harness preferences, no baseline preferences, and 29 ties across the completed evaluations. Procedure planning was closer at 8 harness preferences, 5 baseline preferences, and 47 ties. Breast type and receptor status remained the weakest core category, with only a 2-rating net advantage and 60% exact agreement between oncologists.

Exploratory note-level sign tests support the same direction without treating every field as an independent observation. The combined breast margin was positive in all 20 notes (`p=1.9e-6`, two-sided exact sign test). In the pancreatic set, 18 note-level margins were positive and two were tied (`p=7.6e-6` after excluding ties). These tests were not the prespecified final model and should not replace the planned evaluator-note-field analysis.

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

### 3.5 Qualitative comments

The two exports contained 12 written comments. These comments add useful context but also show why preference counts should not be treated as a complete correctness assessment. For one breast imaging-plan item, the first oncologist preferred the harness because the baseline summarized completed findings rather than the planned PET/CT. The second oncologist rated the same comparison as a tie and wrote that neither answer correctly captured the absence of a new imaging plan. This disagreement should be adjudicated before the case is used as a manuscript example.

Other comments identified unsupported receptor status, a regional-node omission, uncertainty about response after a newly started second-line regimen, and cases in which both outputs were inaccurate. These observations support the value of specialist review and identify concrete targets for final error analysis. They also show that a tie can mean either that both outputs are adequate or that both are wrong.

## 4. Discussion

### 4.1 Main interpretation

[FINAL OPENING: The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline.]

The current pilot now shows the same directional pattern in two oncologists and both cancer domains represented in CORAL. Most comparisons were ties, which is expected when both systems use the same strong base model. Across the completed evaluations, when an oncologist identified a meaningful difference, the harness was preferred more than six times as often as the baseline. The gains were concentrated in fields that require temporal interpretation and clinical classification rather than simple copying.

This pattern matters more than a broad improvement across every field. The harness was designed to preserve correct base-model answers and intervene when a known failure mode appears. A high tie rate with a strong directional advantage is consistent with that design. It suggests selective correction rather than wholesale rewriting of the model output.

The second evaluation materially strengthens the evidence. It reproduced the breast-cancer advantage, with 8 baseline preferences compared with 22 in the first evaluation of the same 20 notes, and extended the direction of effect to pancreatic cancer. Agreement between the two oncologists was good enough to show a shared signal, but not so high that the second file appears duplicative or that clinician judgment can be treated as interchangeable. The remaining disagreements are informative targets for adjudication.

### 4.2 Which questions were difficult for the model?

The combined clinician ratings separate relatively direct extraction from questions that require clinical context.

Straightforward facts often produced ties. Both systems could usually identify an explicitly stated imaging result, procedure, or receptor value. The harder problems involved deciding what the fact meant in the current clinical context.

Active anticancer medication showed the largest and most stable difference: 50 harness preferences, one baseline preference, and nine ties across the completed evaluations. This field requires the model to distinguish treatment from chronic home medications, supportive drugs, discontinued regimens, and future options. Medication planning also strongly favored the harness, 31 to zero with 29 ties, because future actions must remain separate from current treatment and recent changes.

Stage and metastatic involvement require related decisions. The model must distinguish regional lymph nodes from distant spread, preserve uncertainty for lesions awaiting confirmation, and reconcile metastatic status with stage. Across the available ratings, regional or overall metastatic involvement favored the harness 35 to one, while stage favored it 24 to two. The harness was built to check these relationships across fields rather than extract each label in isolation.

Treatment response remained conceptually difficult, but the clinical result was directionally consistent: 18 harness preferences, one baseline preference, and 41 ties. A note may include old progression, current symptoms, stable imaging, tumor-marker trends, and a newly started regimen. Determining which evidence reflects response to the current treatment requires a timeline, not keyword recognition. A second-oncologist comment that response was not yet clear after starting second-line therapy illustrates this temporal boundary.

Tumor type and receptor status remains the weakest core category. Across the two breast evaluations it favored the harness only 8 to 6, with 26 ties, and had the lowest inter-rater agreement at 60%. These values are often stated explicitly, so the baseline can perform well. The field also becomes difficult when a note contains bilateral disease, historical and recurrent specimens, or discordant receptor results. This remains an area for clinical review rather than a claimed strength.

### 4.3 How the observed pattern relates to the harness

The current study does not include a complete component ablation, so it cannot assign each improvement to one module with certainty. The field-level pattern is nevertheless consistent with the intended function of several components.

The active-medication result is consistent with the dedicated medication prompt, the oncology drug dictionary, contextual classification of supportive versus home medications, and temporal filtering. The medication-plan result is consistent with extracting plans from the Assessment and Plan section and removing already completed actions. The stage and metastasis results are consistent with cross-field context and deterministic rules that separate regional from distant disease and suspected from confirmed findings.

The high number of ties provides a useful counterpoint. Deterministic rules did not produce a large advantage in every field. Procedure planning was close to even across the completed evaluations, and recent treatment changes were even in the pancreatic-cancer subset. This makes a simple explanation based on output length or a global preference for the harness less convincing. The strongest differences occurred where the system had explicit safeguards.

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

This pilot is intended to establish whether the effect is large enough and clinically coherent enough to justify a larger study. The second oncologist independently reproduced the breast-cancer direction, and the pancreatic-cancer evaluation showed a similar preference pattern. The next step is no longer to establish whether any replication exists. It is to determine how stable the effect remains across additional evaluators and to fit the prespecified clustered analysis with enough clinicians to estimate evaluator variation credibly.

External validation remains important. CORAL notes come from one institution and represent two cancer domains. The most informative next dataset would contain longitudinal oncology notes from a different health system, with independent clinical annotation and a field contract fixed before evaluation.

## 5. Limitations

The current clinical result comes from two oncologists. Both evaluated breast cancer, but only one evaluated pancreatic cancer. The breast subset supports an initial inter-rater estimate, but two evaluators are insufficient to characterize variability across oncologists, and the pancreatic result does not yet have independent replication. The final manuscript should replace the interim descriptive analysis with the planned multi-rater model.

The A/B interface concealed system identity but used fixed left and right positions, with the harness always shown as A. Agreement across two reviewers and the presence of many ties reduce concern about indiscriminate selection of A, but they do not remove possible position bias. The harness output also included source attribution while the baseline did not. Attribution is part of the system being evaluated, but it may influence preference. A future study should randomize side assignment and separately test the effect of attribution.

The preference labels do not distinguish “both correct” from “both incorrect.” Written comments in the second export explicitly identify some ties in which neither output was satisfactory. Final reporting should therefore pair preference counts with adjudicated error categories rather than interpret every tie as success.

The technical audit used LLM-assisted reviewers and repeated error analysis. It is useful for identifying failures but is not independent clinical validation. The complete 40-note table represents v2.2, while later high-impact repairs were tested on affected samples and controls rather than a new full run.

The dataset is small and comes from one institution. The harness contains clinical rules derived from observed errors, and some may capture documentation conventions specific to CORAL. The study does not yet include a complete component ablation, another model family, or external notes.

Finally, this study evaluates structured extraction. It does not test patient understanding, treatment decisions, workflow efficiency, or clinical outcomes.

## 6. Conclusion

[FINAL CONCLUSION: In a multi-oncologist identity-masked evaluation, the failure-mode-driven inference harness significantly outperformed a same-model single-prompt baseline for structured extraction from longitudinal oncology notes.]

The current pilot shows concordant preference for the harness across two oncologists on breast cancer and an additional positive result from one oncologist on pancreatic cancer. The largest gains occur when extraction requires temporal reasoning and clinical classification, especially active therapy, medication planning, and metastatic status. The model weights remained frozen. The improvement came from the inference process around the model: narrower tasks, verification, conservative clinical rules, and source attribution. Additional evaluators and the prespecified clustered analysis are still required before converting this pilot pattern into the final confirmatory claim.

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
