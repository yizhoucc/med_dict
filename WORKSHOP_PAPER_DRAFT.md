<!--
BILINGUAL MAINTENANCE RULE:
1. Keep the complete English manuscript first and the complete Chinese manuscript second in this same file.
2. The two versions must match in scientific claims, numbers, result tables, figure numbering, and references.
3. Keep the English version close to submission prose. The Chinese version may retain collaborator questions, figure-design notes, and explanatory annotations.
4. After changing either version, update the other version in the same edit and run: python3 render_workshop_draft.py
5. Apply the humanizer pass to both versions. Chinese should read as natural academic prose, not as a literal machine translation.
-->

<div id="english-version"></div>

# From clinician-guided error analysis to cross-cancer transfer: an inference harness for oncology note extraction

**Pilot report**

Version 0.9, September 2026

Authors: [TODO]

Affiliations: [TODO]

Target venue and format: [TODO]

## Abstract

### Background

Large language models can extract structured information from clinical notes, but a single prompt often confuses current and historical treatment, suspected and confirmed disease, and completed findings and future plans. These errors are difficult to detect because the resulting text remains fluent. Oncology notes are a demanding test case because they combine pathology, imaging, treatment history, current therapy, response assessment, and conditional plans across a long clinical timeline.

### Objective

To test whether an inference harness can improve oncology information extraction from a frozen, locally served open-weight language model without fine-tuning.

### Methods

We built a failure-mode-driven inference harness around Qwen2.5-32B-Instruct-AWQ in two development stages. During breast-cancer development, an oncologist repeatedly reviewed outputs and identified clinically important errors. The team converted recurring errors into field-specific prompts, verification gates, and deterministic oncology rules. We then transferred this harness to pancreatic cancer. No physician reviewed pancreatic-cancer outputs during development. Instead, a rubric-informed Qwen reviewer identified candidate errors, an external development LLM synthesized the review history and proposed prompt or rule changes, and a human developer accepted, revised, and regression-tested those changes. Model weights remained frozen throughout. We compared the resulting harness with a single-prompt baseline using the same model and field contract on 40 CORAL benchmark notes, including 20 breast and 20 pancreatic cases. Three oncologists completed identity-masked A/B comparisons. All three evaluated breast cancer, and two also evaluated pancreatic cancer.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. The five completed clinician-by-cancer evaluations contributed 1,359 required-field judgments. The harness was preferred in 443, the baseline in 77, and 839 were ties. Among the 520 directional judgments, 85.2% favored the harness. The three breast-cancer evaluations contributed 282 harness preferences, 54 baseline preferences, and 504 ties. Pairwise exact agreement ranged from 73.6% to 82.9%, with Cohen's kappa from 0.511 to 0.646. In pancreatic cancer, where no physician had participated in development, two oncologists together preferred the harness 161 times and the baseline 23 times, with 335 ties. In a separate exploratory patient-letter evaluation by one oncologist, harness-based letters had a higher four-item mean score than ChatGPT letters in 14 of 20 breast cases, tied in 2, and scored lower in 4, but did not clearly outperform the same-model Qwen baseline.

[FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the harness was preferred in FINAL X, the baseline in FINAL Y, and FINAL Z were ties. The adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### Conclusions

The three breast-cancer evaluations support clinician-guided conversion of recurrent model errors into an explicit inference harness. The two pancreatic-cancer evaluations suggest that the resulting rules and evaluation criteria transferred to a second cancer domain without repeated physician involvement during development. This was AI-assisted, human-supervised refinement rather than autonomous self-modification. The clustered analysis remains necessary before making a confirmatory statistical claim.

## 1. Introduction

Most clinically useful information in oncology is still recorded in free text. A progress note may contain the diagnosis, pathology, receptor status, treatment history, toxicities, response, and next steps, but these facts are spread across sections and timepoints. Manual review is slow, and conventional extraction systems require substantial annotation and task-specific development. The attraction of large language models is straightforward: one model can read many note styles and return structured fields without a new supervised model for every question.

Research activity has moved faster than routine clinical adoption. A 2025 scoping review identified 24 studies of language-model-based oncology information extraction, but also found that external validation and real-world workflow integration remained limited [1]. Clinical deployment has a higher bar than a demonstration on a benchmark. A useful system must preserve uncertainty, distinguish current care from historical events, avoid unsupported facts, and produce results that a clinician can trace back to the note.

Longitudinal oncology notes expose weaknesses that are easy to miss in simpler extraction tasks. A medication list can contain anticancer therapy, supportive treatment, chronic home medications, discontinued drugs, and therapies that are only being discussed. Regional lymph nodes must not be labeled as distant metastases. A suspicious lesion awaiting biopsy must not become confirmed stage IV disease. Tumor growth before treatment is not evidence of failure of a regimen that has just started. A model can mention medically relevant information and still place it in the wrong field or timepoint.

Several strategies have been used to improve clinical extraction. Large proprietary models can perform zero-shot extraction but still omit note-specific details and hallucinate [2]. Local open-weight models reduce dependence on external services, but performance varies with model size and prompt design [3,8]. Fine-tuned hybrid systems can achieve strong performance and external validation, although they require large labeled datasets and model training [5]. Hierarchical prompting, validation layers, and retry mechanisms also improve extraction without conventional fine-tuning [6,7]. These studies establish prior work for the individual techniques used in modern extraction systems.

Human involvement also differs across studies. Clinicians or medical experts often create gold-standard annotations, resolve disagreements, or guide terminology selection. That work establishes whether a model matches a reference label. It does not answer whether a practicing oncologist considers one complete system output more faithful, complete, and clinically useful than another when both contain partly correct information. In our audit of the supplemental methods table for the 24-study scoping review, most papers reported automatic performance metrics against labels. Only two entries explicitly described five-point Likert ratings of generated outputs, and both involved radiology reports [1]. We did not identify a field-level, identity-masked comparison of complete extraction systems by practicing oncologists in that review.

Our approach treats recurring model errors as engineering targets. We use the term inference harness for the software layer that controls how a frozen model is prompted, checked, corrected, and linked to evidence. The model itself is unchanged. When review identifies a repeated failure, such as importing a stopped drug into current therapy or turning suspected metastasis into confirmed disease, the system receives a narrow correction through prompt instructions, verification logic, or a deterministic clinical rule. Each correction can be logged and regression-tested.

The harness is more than a long prompt or retrieval step. Different fields follow different extraction routes. Selected outputs pass into later tasks only when they provide relevant clinical context. Verification stages run after generation, and deterministic hooks enforce narrow oncology constraints across related fields. The system records these interventions and links final values to note evidence. Among the closest studies we reviewed, we did not identify an evaluation of this full combination on real longitudinal oncology notes with a same-model baseline and direct oncologist comparison.

We tested whether this approach improves extraction under a controlled comparison. The harness and baseline use the same Qwen2.5-32B model and the same target schema. The main difference is the inference process surrounding the model. We focus on seven clinical questions that require more than surface entity recognition: active anticancer therapy, stage, distant metastasis, regional or overall metastatic involvement, treatment response, breast cancer type and receptor status, and completed molecular or genetic results.

We hypothesized that the harness would outperform the single-prompt baseline overall and that the largest gains would occur in fields directly addressed by temporal checks, clinical classification rules, and cross-field consistency checks. We also expected many ties because the same base model should answer straightforward questions similarly in both conditions.

## 2. Methods

### 2.1 Study design

This pilot has three linked stages. First, we developed the extraction workflow on breast-cancer notes through repeated review with an oncologist collaborator. Second, we adapted the resulting harness to pancreatic cancer without physician review during that development stage, using a model-in-the-loop error-review cycle with human supervision. Third, we compared the final harness with a single-prompt baseline built on the same frozen model. The comparison used 40 expert-annotated CORAL benchmark notes and an identity-masked oncologist A/B evaluation.

The three stages answer different questions. Breast-cancer development asks whether specialist feedback can be converted into reusable prompts and rules. Pancreatic-cancer development asks whether those lessons can guide adaptation to a related but clinically different domain without requiring the specialist to inspect every iteration. The final comparison asks whether the resulting system is preferred to the same model used without the harness.

### 2.2 Dataset

We used CORAL, an expert-curated dataset of deidentified breast and pancreatic cancer progress notes [2]. The release used in this project contains 200 additional notes without expert annotations, 100 per cancer type, and 40 expert-annotated benchmark notes, 20 per cancer type. Documented development iterations covered 56 breast-cancer notes and all 100 pancreatic-cancer notes from the unannotated pool. The 40 annotated notes were initially reserved for comparison.

CORAL is publicly available, but the records are real clinical notes rather than web-derived questions, synthetic cases, or model-generated narratives. They retain the repeated histories, copied-forward content, uncertain findings, and conditional plans that make longitudinal oncology extraction difficult. The value of this dataset is clinical realism and expert annotation, not size. Several related studies use much larger institutional cohorts, while others use narrower procedure or pathology reports or synthetic oncology notes.

Development notes were used to identify recurrent error patterns and refine the harness. The annotated benchmark notes were used for matched technical and clinician comparisons. A later technical audit of the benchmark set also prompted targeted repairs before the clinician-rated version, so the clinician comparison should be read as a pilot benchmark rather than a pristine one-shot external validation. No model-weight training or fine-tuning was performed.

### 2.3 Base model and baseline

Both conditions used Qwen2.5-32B-Instruct-AWQ [9], served locally through vLLM. The baseline made one model call per note and returned the complete target schema. It did not use task decomposition, verification gates, retries, dictionaries, or deterministic post-processing. The matched baseline and harness used the same field definitions and output contract.

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

> **Figure 1 placeholder: Development, transfer, and evaluation pathway.**
>
> **Draft caption:** *Figure 1. Development and evaluation pathway. Recurrent breast-cancer errors identified with an oncologist were converted into explicit harness components. The harness was then adapted to pancreatic cancer through AI-assisted, human-supervised review without physician input during development. Final outputs were compared with a same-model single-prompt baseline in an identity-masked oncologist evaluation.*

### 2.5 Clinician-guided breast-cancer development

Breast-cancer development used approximately 15 documented iterations across 56 unannotated notes. An oncologist collaborator reviewed generated outputs and identified mistakes that mattered clinically, including errors in treatment status, receptor interpretation, staging, metastatic classification, and response. The development team compared each flagged output with the complete note, grouped recurring failures, and converted them into general changes to prompts, verification logic, or deterministic hooks. The oncologist did not annotate a conventional supervised training set, and the model weights were not updated.

This process is best described as clinician-guided rule induction. The durable artifact was not a collection of corrected answers. It was an explicit set of instructions and checks derived from repeated failure patterns. For example, a review finding that axillary nodal disease had been treated as distant metastasis became a general regional-node rule. A finding that a planned drug had been listed as active therapy became a temporal medication rule. Each high-impact change was rerun on the affected cases and on previously correct controls.

### 2.6 Model-in-the-loop pancreatic-cancer refinement

The breast-cancer harness was then adapted to pancreatic cancer through approximately 18 documented development rounds covering all 100 unannotated pancreatic notes. No physician reviewed pancreatic-cancer outputs during this stage. The cycle had four steps: the pipeline generated structured fields; a rubric-informed Qwen reviewer compared those fields with the complete source note; an external general-purpose LLM used in the development environment summarized the accumulated errors and proposed prompt, hook, or workflow changes; and a human developer inspected, accepted, revised, and regression-tested those changes.

The reviewer prompt included the field definitions, severity criteria, and clinical preferences learned during breast-cancer development. This allowed earlier error categories to guide review in the new domain while still exposing pancreatic-specific problems such as regimen names and dose representation. The process did not allow the deployed model to rewrite its own code, and it was not autonomous self-evolution. We refer to it as AI-assisted, human-supervised refinement because models participated in both error detection and change proposal while a human controlled implementation.

Throughout both development stages, the pipeline logged the original model output, each verification action, and deterministic corrections. This record allowed the team to trace a final value back through the harness and to retain or reject proposed changes based on regression results.

### 2.7 Prespecified clinical fields

The seven core fields were selected before the matched comparison:

1. Which anticancer drugs is the patient actively receiving?
2. What is the current cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer responding to the current treatment?
6. What is the breast cancer type and ER/PR/HER2 status?
7. What completed molecular or genetic results are documented?

The oncologist instrument also included genetic testing plans, supportive medications, procedure plans, imaging plans, laboratory plans, medication plans, and recent treatment changes. Laboratory summary and general clinical findings were optional and excluded from the primary clinician analysis.

### 2.8 Technical evaluation

The complete matched audit contained 260 applicable note-field comparisons. A source-grounded LLM-assisted review process read the source note and both outputs for each comparison. Each result was classified as harness better, baseline better, or tie. All reported harness losses and contested high-severity findings were rechecked against the source note. This audit was used for development and technical error analysis. It was not treated as a replacement for clinician evaluation.

Four high-impact failures found during the complete v2.2 audit were subsequently repaired. The affected cases and clean controls were rerun as a targeted regression set. These results are reported separately because the targeted set is not a new full-cohort estimate.

### 2.9 Oncologist evaluation

The clinical evaluation presents the source note and two structured outputs through an identity-masked A/B interface. The evaluator selects A better, B better, or tie for each field. The interface does not reveal which output came from the harness.

[TODO BEFORE SUBMISSION: State whether the oncologist who participated in breast-cancer development was also one of the final evaluators. If so, distinguish masking of system identity from independence from the development process.]

The first oncologist completed all 280 required breast-cancer judgments and all 260 required pancreatic-cancer judgments. The second completed all 280 breast-cancer judgments and 259 of 260 pancreatic-cancer judgments; `p7 / lab_plan` was missing. The third completed all 280 breast-cancer judgments and did not evaluate pancreatic cancer. All raw exports were preserved unchanged. The scoring template retained stale filenames after the displayed results were updated. The project owner confirmed that the clinicians reviewed the newer outputs, but the exact hashes of the displayed PL and BL artifacts must be inserted before submission: [TODO].

All three oncologists completed the same 280 required breast-cancer comparisons. We calculated pairwise exact agreement and Cohen's kappa for each pair. We also summarized all five completed clinician-by-cancer evaluations separately. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The primary analysis will compare harness and baseline preference among directional ratings with a mixed-effects logistic model that includes evaluator, note, and field as grouping factors. Ties will be reported separately and included in a sensitivity analysis. We will report the effect estimate, 95% confidence interval, two-sided p value, and agreement across evaluators. The current three-oncologist summaries remain descriptive until this model and its sensitivity analyses are finalized.

### 2.10 Exploratory patient-letter evaluation

Patient-letter generation preceded the extraction-only study and motivated the shift toward a more controlled task. One oncologist evaluated 20 breast-cancer notes with three patient letters per note. The systems were shown as A, B, and C: a single-prompt GPT-4o letter, a single-prompt Qwen2.5-32B letter, and a Qwen2.5-32B letter generated from the harness output. The evaluator scored accuracy, completeness, comprehensibility, and usefulness on five-point scales, marked possible hallucinations, and rated deployment readiness.

This evaluation was not powered or designed as a primary three-system trial. We therefore report it descriptively as evidence about a downstream use of structured extraction. It should not be combined with the extraction preference counts or used to claim that the complete letter system is superior.

## 3. Results

### 3.1 Development path and cross-cancer transfer

The development record contains approximately 15 breast-cancer iterations across 56 notes and approximately 18 pancreatic-cancer iterations across 100 notes. Breast-cancer revisions were informed by direct oncologist review. Pancreatic-cancer revisions were made without physician review of the pancreatic outputs, using the transferred rubric, model-based error review, external LLM-assisted synthesis, and human-controlled implementation described above.

This sequence produced two types of reuse. Some components transferred unchanged, including the five verification stages, temporal distinctions, source attribution, and rules that separate active treatment from plans or supportive medication. Other components required cancer-specific routing, especially disease terminology, regimen interpretation, and post-processing conditions. The resulting system therefore reused the error-handling framework without assuming that breast and pancreatic cancer were clinically interchangeable.

The pancreatic-cancer clinician result is the most relevant evidence for this transfer. Across two independent evaluations, the oncologists preferred the harness in 161 field comparisons and the baseline in 23, with 335 ties. One oncologist recorded a positive within-note margin in 19 cases and a negative margin in one. The other recorded 18 positive margins and two ties. When their ratings were pooled within notes, all 20 pancreatic cases had a positive harness-minus-baseline margin. Because no physician reviewed pancreatic outputs during development, this pattern is consistent with transfer of the previously codified evaluation criteria and workflow. It does not identify whether the gain came from the transferred rules, pancreatic-specific revisions, the model-based reviewer, or their combination.

### 3.2 Current oncologist evaluation

The available clinical evidence comprises five completed clinician-by-cancer evaluations: breast cancer from all three oncologists and pancreatic cancer from two. Across 1,359 required-field judgments, the harness was preferred 443 times, the baseline 77 times, and 839 comparisons were ties. The harness received 85.2% of the 520 directional judgments.

| Completed evaluation | Required judgments | Harness | Baseline | Tie | Harness share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 01, breast cancer | 280 | 84 | 22 | 174 | 79.2% |
| Oncologist 01, pancreatic cancer | 260 | 75 | 14 | 171 | 84.3% |
| Oncologist 02, breast cancer | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 02, pancreatic cancer | 259 | 86 | 9 | 164 | 90.5% |
| Oncologist 03, breast cancer | 280 | 119 | 24 | 137 | 83.2% |
| **All completed evaluations** | **1,359** | **443** | **77** | **839** | **85.2%** |

> **Figure 2 placeholder: Clinician preference distributions by evaluator and cancer type.**
>
> **Draft caption:** *Figure 2. Distribution of identity-masked clinician preferences across five completed clinician-by-cancer evaluations. Most judgments were ties, as expected for systems using the same base model, but directional judgments consistently favored the inference harness.*

All three clinicians independently favored the harness on the breast-cancer set. Their pooled breast result was 282 harness preferences, 54 baseline preferences, and 504 ties. The first oncologist's per-note result was 18 harness wins, one baseline win, and one tie. The second and third oncologists each recorded a harness win in all 20 notes. When the three breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin.

The two pancreatic-cancer evaluations produced 161 harness preferences, 23 baseline preferences, and 335 ties. One required judgment, `p7 / lab_plan`, was missing. The harness had a positive pooled margin in all 20 pancreatic notes.

The final mixed-effects result remains pending:

> [FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### 3.3 Inter-rater agreement and core fields

All three oncologists rated the same 280 required breast-cancer comparisons. Pairwise exact agreement was 82.9% between oncologists 01 and 02, 80.0% between oncologists 01 and 03, and 73.6% between oncologists 02 and 03. The corresponding Cohen's kappa values were 0.646, 0.644, and 0.511. All three oncologists gave the same verdict on 192 comparisons (68.6%). A simple majority favored the harness in 95 comparisons, the baseline in 15, and a tie in 168. Two comparisons had one vote in each category and therefore no majority.

> **Figure 3 placeholder: Pairwise inter-rater agreement for breast-cancer judgments.**
>
> **Draft caption:** *Figure 3. Pairwise agreement among three oncologists on 280 shared breast-cancer field comparisons. Exact agreement ranged from 73.6% to 82.9%, with Cohen's kappa from 0.511 to 0.646.*

Across all completed evaluations, the seven prespecified core categories contributed 660 applicable judgments. The harness received 276 preferences, the baseline 34, and 350 were ties. The harness therefore accounted for 89.0% of the 310 directional core judgments, and every core category had a positive aggregate margin.

| Core field | Harness | Baseline | Tie | Net advantage |
|---|---:|---:|---:|---:|
| Active anticancer medications | 83 | 2 | 15 | +81 |
| Stage | 41 | 4 | 55 | +37 |
| Distant metastasis | 15 | 2 | 83 | +13 |
| Regional or overall metastasis | 60 | 5 | 35 | +55 |
| Treatment response | 37 | 3 | 60 | +34 |
| Breast cancer type and receptors | 18 | 12 | 30 | +6 |
| Completed molecular or genetic results | 22 | 6 | 72 | +16 |
| **Overall** | **276** | **34** | **350** | **+242** |

> **Figure 4 placeholder: Preference profile across core clinical categories.**
>
> **Draft caption:** *Figure 4. Clinician preference by prespecified core clinical category. The largest harness advantages occurred in active-treatment identification and metastatic-involvement classification. Breast type and receptor status showed the smallest margin.*

Active anticancer medication and regional or overall metastatic involvement produced the largest and most consistent core-field margins. Medication planning, outside the seven core categories, totaled 59 harness preferences, no baseline preferences, and 41 ties. Procedure planning was much closer at 14 harness preferences, 12 baseline preferences, and 74 ties. Breast type and receptor status remained the weakest core category, with a 6-rating net advantage.

Exploratory note-level summaries support the same direction without treating every field as an independent observation. After pooling clinicians within each cancer type, the harness-minus-baseline margin was positive in all 20 breast notes and all 20 pancreatic notes. Each of the five evaluator-by-cancer analyses also favored the harness at the aggregate level. These summaries were not the prespecified final model and should not replace the planned evaluator-note-field analysis.

> **Figure 5 placeholder: Per-note clinician preference margins.**
>
> **Draft caption:** *Figure 5. Distribution of normalized clinician preference margins across individual notes. After pooling the available clinicians within each cancer type, the harness had a positive margin in every breast-cancer and pancreatic-cancer note.*

> **Figure 6 placeholder: Adjusted multi-rater effect estimates.**
>
> **Draft caption:** *Figure 6. Adjusted association between evaluation condition and clinician preference. Odds ratios greater than 1 favor the inference harness. Estimates will come from the prespecified clustered multi-rater analysis.*

### 3.4 Complete matched technical audit

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

> **Supplementary Figure S1 placeholder: Technical-audit and clinician net preference rates.**
>
> **Draft caption:** *Supplementary Figure S1. Category-level net preference rates in the complete technical audit and current clinician evaluation. Differences between series should be interpreted descriptively because the review processes and pipeline versions were not identical.*

### 3.5 Targeted repair evaluation

Four high-impact failures identified in the v2.2 audit were repaired with conservative rules. Across 51 applicable comparisons in six affected samples and two controls, the repaired harness recorded 29 wins, no baseline wins, and 22 ties. No P0 error remained in this targeted set, and neither control developed a detected core regression. These targeted results show that the identified errors were repairable. They do not replace the full 40-note result.

### 3.6 Qualitative comments

Two of the three exports contained 12 written comments. These comments add useful context but also show why preference counts should not be treated as a complete correctness assessment. For one breast imaging-plan item, the first oncologist preferred the harness because the baseline summarized completed findings rather than the planned PET/CT. The second oncologist rated the same comparison as a tie and wrote that neither answer correctly captured the absence of a new imaging plan. This disagreement should be adjudicated before the case is used as a manuscript example.

Other comments identified unsupported receptor status, a regional-node omission, uncertainty about response after a newly started second-line regimen, and cases in which both outputs were inaccurate. These observations support the value of specialist review and identify concrete targets for final error analysis. They also show that a tie can mean either that both outputs are adequate or that both are wrong.

> **Supplementary Figure S2 placeholder: Adjudicated interpretation of tie judgments.**
>
> **Draft caption:** *Supplementary Figure S2. Clinical interpretation of tie judgments after manual adjudication. This analysis distinguishes equivalent correct outputs from comparisons in which both systems are incomplete or incorrect.*

### 3.7 Exploratory patient-letter results

The earlier patient-letter evaluation did not show a uniform advantage for the harness. The same-model Qwen baseline had the highest four-item mean score at 3.80, followed by the harness-based letter at 3.74 and ChatGPT at 3.50. All three systems were rated ready to send without editing in 3 of 20 cases. The clinician marked possible hallucinated content in 4 ChatGPT letters, 1 Qwen baseline letter, and 2 harness-based letters.

| Letter system | Accuracy | Completeness | Comprehensibility | Usefulness | Four-item mean | Hallucination flagged |
|---|---:|---:|---:|---:|---:|---:|
| ChatGPT single-prompt | 3.85 | 3.25 | 3.60 | 3.30 | 3.50 | 4/20 |
| Qwen single-prompt baseline | 4.10 | 3.60 | 3.85 | 3.65 | 3.80 | 1/20 |
| Qwen harness-based letter | 3.95 | 3.60 | 3.85 | 3.55 | 3.74 | 2/20 |

The paired per-note comparison gives a more useful view of the trend. The harness-based letter had a higher four-item mean than the ChatGPT letter in 14 cases, tied in 2, and scored lower in 4, for a mean paired difference of 0.24 points. An unadjusted exact sign test excluding ties gave `p=0.031`; this was exploratory, not prespecified, and several correlated outcomes were examined. Against the same-model Qwen baseline, the harness won 9 cases, tied in 5, and lost 6, with a mean paired difference of -0.06 points. Structured extraction can therefore serve as input to patient communication, but these results do not show that the current letter generator is better than the same model prompted directly.

> **Figure 7 placeholder: Paired patient-letter score differences.**
>
> **Draft caption:** *Figure 7. Exploratory paired differences in oncologist-rated patient-letter quality across 20 breast-cancer notes. Harness-based letters more often outscored ChatGPT, but did not show a clear advantage over the same-model single-prompt baseline.*

## 4. Discussion

### 4.1 Main interpretation

[FINAL OPENING: The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline.]

The main result concerns the development process around the model. A clinician helped identify recurrent errors in breast cancer, those errors were converted into explicit parts of the inference harness, and the resulting process was adapted to pancreatic cancer without physician review during pancreatic development. The final clinician ratings favored the harness in both domains. This sequence connects the development method to the evaluation more directly than a static comparison of two prompts would.

Most field comparisons were ties, which is expected because both systems used the same strong base model. When an oncologist found a meaningful difference, the harness was preferred nearly six times as often as the baseline. The gains were concentrated in fields that need temporal interpretation or clinical classification. This is consistent with a system that leaves straightforward answers alone and intervenes when a known failure pattern appears.

The evidence is still a pilot. The breast and pancreatic stages were not randomized, the refinement process combined several tools, and the benchmark later informed targeted repairs. The results therefore support the whole development strategy, not a causal claim for any single model, prompt, gate, or hook.

### 4.2 Clinician-guided rule induction in breast cancer

The oncologist's role during breast development was different from conventional dataset labeling. The clinician did not produce thousands of field labels for model training. Instead, the clinician reviewed concrete outputs and identified mistakes that would matter in practice. The team then translated repeated mistakes into reusable instructions and checks.

When specialist time is scarce, a clinician can focus on defining the boundary of a difficult concept, such as active therapy, response to the current regimen, or regional versus distant disease, instead of reviewing every future note. Once encoded, the rule can be applied consistently, logged, and regression-tested. Three independent breast evaluations suggest that the resulting harness did not merely reproduce one collaborator's preferences. All three oncologists favored it, although their individual judgments were not identical.

This development pattern also explains why we call the system a harness. The contribution is the accumulation of executable clinical distinctions around a frozen model. Some distinctions live in prompts, some in verification, and some in deterministic code. The clinician supplies the clinical boundary; the engineering process turns that boundary into repeatable behavior.

### 4.3 Transfer and AI-assisted refinement in pancreatic cancer

The pancreatic-cancer stage tests whether the accumulated process can travel beyond the domain in which the clinician gave direct feedback. Pancreatic notes differ in disease course, treatment regimens, staging language, and surgical context. We therefore did not simply reuse every breast-specific rule. We transferred the general workflow, retained rules that expressed shared clinical distinctions, and added pancreatic-specific routing where the notes exposed new failure patterns.

No physician reviewed pancreatic outputs during this development stage. A Qwen reviewer applied the inherited rubric to each output, and an external development LLM synthesized the review history and proposed changes. A human developer decided which changes to implement and ran regression tests. Across two pancreatic evaluations, 161 harness preferences versus 23 baseline preferences with 335 ties are consistent with successful transfer under this model-in-the-loop process.

We do not describe this as autonomous self-evolution. The deployed model did not independently modify its code or approve its own changes. "AI-assisted evolution" is reasonable only in the narrower sense that models helped identify errors and formulate revisions across repeated cycles. Human supervision remained part of the development loop.

### 4.4 Which questions were difficult for the model?

The combined clinician ratings separate relatively direct extraction from questions that require clinical context.

Straightforward facts often produced ties. Both systems could usually identify an explicitly stated imaging result, procedure, or receptor value. The harder problems involved deciding what the fact meant in the current clinical context.

Active anticancer medication showed the largest and most stable core-field difference: 83 harness preferences, two baseline preferences, and 15 ties across the completed evaluations. This field requires the model to distinguish treatment from chronic home medications, supportive drugs, discontinued regimens, and future options. Medication planning also strongly favored the harness, 59 to zero with 41 ties, because future actions must remain separate from current treatment and recent changes.

Stage and metastatic involvement require related decisions. The model must distinguish regional lymph nodes from distant spread, preserve uncertainty for lesions awaiting confirmation, and reconcile metastatic status with stage. Across the available ratings, regional or overall metastatic involvement favored the harness 60 to 5, while stage favored it 41 to 4. The harness was built to check these relationships across fields rather than extract each label in isolation.

Treatment response remained conceptually difficult, but the clinical result was directionally consistent: 37 harness preferences, three baseline preferences, and 60 ties. A note may include old progression, current symptoms, stable imaging, tumor-marker trends, and a newly started regimen. Determining which evidence reflects response to the current treatment requires a timeline, not keyword recognition. A second-oncologist comment that response was not yet clear after starting second-line therapy illustrates this temporal boundary.

Tumor type and receptor status remains the weakest core category. Across the three breast evaluations it favored the harness 18 to 12, with 30 ties. These values are often stated explicitly, so the baseline can perform well. The field also becomes difficult when a note contains bilateral disease, historical and recurrent specimens, or discordant receptor results. This remains an area for clinical review rather than a claimed strength.

### 4.5 How the observed pattern relates to the harness

The current study does not include a complete component ablation, so it cannot assign each improvement to one module with certainty. The field-level pattern is nevertheless consistent with the intended function of several components.

The active-medication result is consistent with the dedicated medication prompt, the oncology drug dictionary, contextual classification of supportive versus home medications, and temporal filtering. The medication-plan result is consistent with extracting plans from the Assessment and Plan section and removing already completed actions. The stage and metastasis results are consistent with cross-field context and deterministic rules that separate regional from distant disease and suspected from confirmed findings.

The high number of ties provides a useful counterpoint. Deterministic rules did not produce a large advantage in every field. Procedure planning was close to even across the completed evaluations, and recent treatment changes were even in the pancreatic-cancer subset. This makes a simple explanation based on output length or a global preference for the harness less convincing. The strongest differences occurred where the system had explicit safeguards.

These associations support the proposed mechanism, but they do not prove it. A future component study should compare a single prompt, decomposed prompts alone, prompts plus verification, and the complete harness. That experiment would show how much each layer contributes.

### 4.6 Comparison with related work

The closest studies use different combinations of data, model adaptation, and human review. Several include clinicians, but their role is usually to create a reference standard or guide model development. Fewer studies ask oncology specialists to compare final system outputs directly.

| Study | Data and task | Role of clinical experts | Main approach | Difference from this study |
|---|---|---|---|---|
| Sushil et al., CORAL [2] | 40 real breast and pancreatic cancer notes; broad oncology schema | Expert annotation; an independent oncologist manually evaluated GPT-4 on 10 notes per cancer type | Zero-shot GPT-4, GPT-3.5-turbo, and FLAN-UL2 | Same dataset and clinical scope. It established the benchmark and documented omissions and hallucinations. It did not test a failure-mode-driven harness against the same frozen model. |
| Wiest et al. [3] | 500 MIMIC histories; five binary clinical features | Three blinded medical experts created consensus ground truth | Local Llama 2 with grammar-constrained JSON and prompt variants | Strong expert validation and local deployment, but the targets were five binary features outside oncology. Experts supplied ground truth rather than comparative preference ratings of two full systems. |
| Bhattarai et al. [4] | 13,646 notes from 63 patients with lung cancer; four longitudinal phenotypes | Two subject-matter experts supplied gold-standard manual annotations | GPT, open-model, and rule-based comparison | Larger longitudinal corpus with expert labels, but fewer target phenotypes and no same-model comparison isolating the surrounding workflow. |
| Tariq et al. [5] | 26,692 breast cancer patients internally and 162 externally; treatment timelines | Cancer-registry data supplied treatment labels; clinical experts curated treatment concepts and codes | UMLS parser plus a fine-tuned question-answering model | Much stronger scale and external validation. It requires supervised fine-tuning and focuses on five treatment categories rather than broad field-level extraction. |
| Dao et al. [6] | 220 development and 200 validation right-heart-catheterization notes | One pulmonary vascular disease expert created the validation ground truth and guided development | Local open model, engineered preload, validation, and retry | The closest workflow architecture. The task was numerical extraction from procedure notes, and the system did not include oncology-specific field routing and cross-field clinical hooks. |
| Zhang et al., mCODEGPT [7] | 1,000 synthetic oncology notes; 49 mCODE entities | Programmatic matching was supplemented by manual validation from human reviewers | Hierarchical prompting versus single-step prompting | Direct evidence that prompt hierarchy helps, but the notes were synthetic and the study did not use blinded oncologist comparison of final outputs. |
| Grothey et al. [8] | 579 prostate pathology reports in German and English | A trained medical doctoral student annotated reports under an attending pathologist's supervision | Multiple open and proprietary models, prompt and quantization tests | Larger expert-labeled benchmark, but it evaluated one report type and 11 predefined parameters rather than heterogeneous longitudinal clinic notes. Evaluation relied on reference labels rather than end-user preference. |

These papers show that clinician involvement is not absent from the field. The distinction is where that involvement occurs. Annotation, adjudication, and terminology design provide a reference answer before evaluation. The 24-study scoping review was dominated by precision, recall, F1, AUC, and accuracy comparisons. In our audit of its supplemental methodology table, only two entries explicitly described five-point Likert ratings of generated outputs, both involving radiology reports [1]. Our planned study adds downstream review by practicing oncologists, who inspect the source note and compare the final outputs field by field without seeing system identity. This captures clinical preference when both outputs are partly correct, when uncertainty matters, or when one answer is more complete without being less faithful.

Our development design adds a second distinction. Specialist feedback was used to discover reusable failure rules in one cancer domain, then withheld during development in the second domain. The pancreatic result therefore examines whether the learned workflow can reduce repeated demand on a scarce specialist, rather than asking the clinician to remain in every iteration. We did not identify this specific development-and-transfer design in the comparison studies above.

The data contribution also needs precise wording. CORAL is public, but it contains real deidentified longitudinal oncology notes with expert annotation. This differs from web questions, synthetic notes, and narrow report templates. It does not exceed the scale or external validation of the largest institutional studies. Its advantage for this experiment is that the notes preserve the clinical ambiguity our harness is designed to address.

Our technical claim concerns the combination. Prompt engineering, retrieval, guardrails, retries, and hybrid rules already exist in the literature. In the closest studies reviewed here, we did not find the full evaluated combination used in this project: field-specific routing, selective dependency transfer, five verification stages, oncology drug and terminology resources, deterministic clinical hooks, cross-field consistency checks, action logging, source attribution, and a same-model baseline that removes model capability as the main explanation. The contribution is this integrated inference harness and its evaluation, not any one component in isolation.

Several related studies have larger datasets, more annotators, or external validation. Our pilot should not be presented as the first clinical extraction system or the largest evaluation. It can be presented as a focused test of whether a structured, auditable workflow can make the same frozen model more reliable on difficult oncology fields, with direct review by the clinicians who understand those distinctions.

### 4.7 From structured extraction to patient communication

The earlier patient-letter result was mixed. Harness-based letters were descriptively better than ChatGPT on the four-item mean in 14 of 20 breast cases and received fewer hallucination flags, but they did not clearly beat the same-model Qwen baseline. Stronger extraction therefore does not automatically produce a better complete letter. Letter quality also depends on selection, organization, wording, explanation of uncertainty, and decisions about which clinical details a patient needs.

This finding supports the revised project order. Structured extraction is the measurable safety layer, and patient communication is a downstream task built on that layer. The extraction study can identify whether stage, treatment, response, and plans are represented faithfully before a generator turns them into prose. Future letter work should test whether specific extraction gains survive that second transformation, ideally with patient readers as well as clinicians.

The letter experiment remains worth reporting as an exploratory application. It shows that the harness can feed a patient-facing output and that the result can compare favorably with a proprietary general model on some dimensions. It does not justify presenting letter generation as the paper's primary success, nor does it show that the current letters are ready for clinical deployment.

### 4.8 Clinical and technical implications

The findings suggest that some LLM failures in oncology extraction are repeatable enough to address outside the model weights. This is useful when labeled training data are scarce or when a clinical team needs to change a field definition without retraining. A rule that preserves biopsy-pending disease as suspected can be inspected and tested. A prompt-only system offers less control because a wording change may affect unrelated fields.

The term inference harness is appropriate for this system because it includes more than a prompt sequence. It manages task routing, dependencies, verification, deterministic corrections, logging, and source attribution around a frozen model. Clinical workflow is also understandable, but it can imply integration into routine care, which this pilot has not tested. We therefore use inference harness for the technical contribution and evaluation workflow for the study procedure.

The system is also compatible with local deployment. Local operation does not by itself establish privacy compliance or clinical safety, but it allows an institution to retain control of note processing and system updates. The present work evaluates extraction quality, not readiness for autonomous clinical use.

### 4.9 Pilot status and next steps

This pilot is intended to establish whether the effect is large enough and clinically coherent enough to justify a larger study. Three oncologists independently reproduced the breast-cancer direction, and two reproduced the pancreatic-cancer direction. The next step is to fit the prespecified clustered analysis and determine how much the effect varies by clinician, note, and field.

External validation remains important. CORAL notes come from one institution and represent two cancer domains. The most informative next dataset would contain longitudinal oncology notes from a different health system, with independent clinical annotation and a field contract fixed before evaluation.

## 5. Limitations

The current clinical result comes from three oncologists. All three evaluated breast cancer, but only two evaluated pancreatic cancer. The data show replication across clinicians, but three evaluators still provide a limited estimate of between-oncologist variation. The final manuscript should replace the interim descriptive analysis with the planned multi-rater model.

The A/B interface concealed system identity but used fixed left and right positions, with the harness always shown as A. Agreement across reviewers and the presence of many ties reduce concern about indiscriminate selection of A, but they do not remove possible position bias. The harness output also included source attribution while the baseline did not. Attribution is part of the system being evaluated, but it may influence preference. A future study should randomize side assignment and separately test the effect of attribution.

The preference labels do not distinguish "both correct" from "both incorrect." Written comments in the second export explicitly identify some ties in which neither output was satisfactory. Final reporting should therefore pair preference counts with adjudicated error categories rather than interpret every tie as success.

The technical audit used LLM-assisted reviewers and repeated error analysis. It is useful for identifying failures but is not independent clinical validation. The complete 40-note table represents v2.2, while later high-impact repairs were tested on affected samples and controls rather than a new full run. Because the benchmark informed these repairs before the clinician-rated artifacts were prepared, the clinical comparison is not a pristine external validation.

The development stages were sequential rather than randomized. Breast development combined oncologist feedback with engineering judgment, while pancreatic development combined an inherited rubric, a model-based reviewer, an external development LLM, and human implementation. The study cannot isolate the contribution of any one element. It also does not evaluate a fully autonomous system because a human approved and regression-tested changes.

The dataset is small and comes from one institution. The harness contains clinical rules derived from observed errors, and some may capture documentation conventions specific to CORAL. The study does not yet include a complete component ablation, another model family, or external notes.

The patient-letter analysis has one oncologist, 20 breast cases, three systems, and several correlated ratings. The exploratory result cannot establish superiority, and it did not show a clear advantage over the same-model Qwen baseline. The study does not test patient understanding, treatment decisions, workflow efficiency, or clinical outcomes.

## 6. Conclusion

[FINAL CONCLUSION: In a multi-oncologist identity-masked evaluation, the failure-mode-driven inference harness significantly outperformed a same-model single-prompt baseline for structured extraction from longitudinal oncology notes.]

The current pilot supports a development strategy in which a clinician identifies clinically important failure patterns, the team converts those patterns into an explicit inference harness, and the harness is adapted to another cancer domain through AI-assisted, human-supervised refinement. All three oncologists favored the harness on breast cancer. Two also favored it on pancreatic cancer, where no physician had participated in development. The largest gains involved active therapy, medication planning, and metastatic status. The model weights remained frozen throughout. A prespecified clustered analysis and external validation are still needed before this pilot pattern becomes a confirmatory claim.

## References

1. Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. *JMIR Cancer*. 2025;11:e65984. [https://doi.org/10.2196/65984](https://doi.org/10.2196/65984).
2. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024;1(4). [https://doi.org/10.1056/AIdbp2300110](https://doi.org/10.1056/AIdbp2300110).
3. Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024;7:257. [https://doi.org/10.1038/s41746-024-01233-2](https://doi.org/10.1038/s41746-024-01233-2).
4. Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024;7(3):ooae060. [https://doi.org/10.1093/jamiaopen/ooae060](https://doi.org/10.1093/jamiaopen/ooae060).
5. Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. *JCO Clinical Cancer Informatics*. 2025;9:e2500002. [https://doi.org/10.1200/CCI-25-00002](https://doi.org/10.1200/CCI-25-00002).
6. Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025;8(5):ooaf097. [https://doi.org/10.1093/jamiaopen/ooaf097](https://doi.org/10.1093/jamiaopen/ooaf097).
7. Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025;5:422. [https://doi.org/10.1038/s43856-025-01116-x](https://doi.org/10.1038/s43856-025-01116-x).
8. Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025;5:96. [https://doi.org/10.1038/s43856-025-00808-8](https://doi.org/10.1038/s43856-025-00808-8).
9. Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025. [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115).

<div class="language-break" id="chinese-version"></div>

# 从临床医生指导的错误分析到跨癌种迁移：用于肿瘤科病历信息提取的推理框架

**供临床合作者审阅的试点报告草稿**

版本 0.9，2026 年 9 月

作者：[TODO]

单位：[TODO]

目标会议及稿件形式：[TODO]

本稿现已纳入三位肿瘤科医生完成的评审。三位医生均评估了乳腺癌病例，其中两位还评估了胰腺癌病例。全文按照实际开发过程展开：首先由临床医生指导乳腺癌任务的错误分析，再将所得经验编码为推理框架；随后在开发阶段没有医生参与的情况下，通过模型参与闭环的方法改进胰腺癌任务；最后由临床医生进行盲法评估。聚类多评审者模型及方括号内的推断性统计结果仍为占位内容，投稿前必须补全。图示占位说明了预期的视觉设计和当前趋势，并非最终图片。

供临床审阅的简要说明：两个系统使用相同的语言模型。基线系统要求模型通过一次调用提取全部信息。推理框架则把任务拆分为较小的临床问题，检查模型答案，针对反复出现的错误应用范围明确的肿瘤学规则，并把每项结果与病历中的支持性原文关联起来。

## 供临床合作者审阅的问题

本轮审阅请重点关注以下问题。投稿时将删除本节。

1. 将这七个核心临床字段定义为具有临床重要性的字段是否合适？
2. 我们对当前抗癌治疗、分期、区域与远处转移、治疗反应、受体状态以及分子检测结果的解释是否符合临床实际？
3. Discussion 是否在不过度扩展研究结论的前提下，准确解释了观察到的优势和不足？
4. 哪些案例最能说服肿瘤学读者？
5. 是否存在不准确、夸大的临床表述，或缺少必要背景的内容？
6. 我们对临床医生建立金标准和临床医生评估最终输出这两种角色的区分，是否公平且具有临床意义？

## 摘要

### 背景

大语言模型可以从临床病历中提取结构化信息，但单一提示经常混淆当前治疗与既往治疗、疑似病变与确诊疾病，以及已经完成的检查与未来计划。这类错误不易发现，因为生成文本通常仍然流畅。肿瘤科病历尤其具有挑战性：一份纵向病历往往同时包含病理、影像、治疗史、当前治疗、疗效评估和条件性计划。

### 目的

评估推理框架能否在不进行微调的情况下，提高冻结参数、本地部署的开放权重语言模型对肿瘤科信息的提取质量。

### 方法

我们围绕 Qwen2.5-32B-Instruct-AWQ 分两个开发阶段构建了一个由失败模式驱动的推理框架。在乳腺癌开发阶段，一位肿瘤科医生反复审阅输出并指出具有临床意义的错误。团队将反复出现的错误转化为字段专用提示、验证门和确定性肿瘤学规则。随后，我们将该框架迁移到胰腺癌。在胰腺癌开发期间，没有医生审阅模型输出。我们使用依据评分准则配置的 Qwen 审查模型识别候选错误，由开发环境中的外部通用 LLM 汇总既往审查记录并提出提示或规则修改方案，再由人工开发者接受、修改并进行回归测试。整个过程中模型权重始终保持冻结。我们在 40 份 CORAL 基准病历上比较了该框架与使用相同模型和字段契约的单提示基线，其中包括 20 例乳腺癌和 20 例胰腺癌。三位肿瘤科医生完成了隐藏系统身份的 A/B 比较。三位医生均评估了乳腺癌病例，其中两位还评估了胰腺癌病例。

### 结果

在已完成的匹配技术审查中，推理框架在 66 项核心比较中更优，基线在 28 项中更优，另有 166 项为平局。五组已完成的评审者与癌种组合共提供 1,359 项必评字段判断，其中 443 项偏好推理框架，77 项偏好基线，839 项为平局。在 520 项非平局判断中，85.2% 偏好推理框架。三次乳腺癌评估合计包括 282 项框架偏好、54 项基线偏好和 504 项平局。两两完全一致率为 73.6% 至 82.9%，Cohen's kappa 为 0.511 至 0.646。在开发阶段没有医生参与的胰腺癌任务中，两位医生合计给予推理框架 161 项偏好，基线 23 项偏好，另有 335 项平局。在一项由一位肿瘤科医生单独完成的探索性患者信件评估中，20 例乳腺癌病例里有 14 例的框架信件四项平均分高于 ChatGPT 信件，2 例持平，4 例较低；但框架信件并未明确优于使用相同模型的 Qwen 基线信件。

[最终多评审者结果：在 FINAL N 位肿瘤科医生完成的 FINAL N 项可评估判断中，推理框架在 FINAL X 项中更优，基线在 FINAL Y 项中更优，FINAL Z 项为平局。校正后分析显示推理框架获得显著偏好，效应估计值为 FINAL，95% CI 为 FINAL，p=FINAL。]

### 结论

三次乳腺癌评估支持将临床医生发现的反复性模型错误转化为明确的推理框架。两次胰腺癌评估提示，由此形成的规则和评估标准可以迁移到第二个癌种，而不需要医生在开发期间反复参与。这一过程属于 AI 辅助、人工监督的改进，并非自主修改。在提出确证性统计结论前，仍需完成预先设定的聚类分析。

## 1. 引言

肿瘤临床工作中，大量有用信息仍记录在自由文本中。一份随访病历可能同时包含诊断、病理、受体状态、治疗史、毒性反应、疗效和后续计划，但这些事实分散在不同章节和时间点。人工审阅耗时较长，传统信息提取系统则需要大量标注和针对具体任务的开发。大语言模型的吸引力很直接：同一个模型可以读取多种病历写法，并针对不同问题返回结构化字段，无须为每个问题重新训练监督模型。

相关研究的发展速度快于临床常规应用。2025 年的一项范围综述纳入了 24 项使用语言模型提取肿瘤学信息的研究，但也发现外部验证和真实工作流程整合仍然有限 [1]。临床部署的要求高于基准数据上的概念验证。一个实用系统必须保留不确定性，区分当前诊疗与历史事件，避免无依据的事实，并让临床医生能够从病历原文追溯每项结果。

纵向肿瘤科病历会暴露一些在简单提取任务中不易发现的问题。药物清单可能同时包含抗癌治疗、支持治疗、长期居家用药、已停用药物，以及仅处于讨论阶段的治疗。区域淋巴结不能被标为远处转移。等待活检的可疑病灶不能被写成已确诊的 IV 期疾病。治疗前的肿瘤增长也不能用来判断刚刚开始的方案无效。模型即使提到了医学相关信息，仍可能把它放入错误的字段或时间点。

已有研究采用了多种方法改进临床信息提取。大型闭源模型可以进行零样本提取，但仍会遗漏病历特有细节并产生幻觉 [2]。本地开放权重模型减少了对外部服务的依赖，但性能会随模型规模和提示设计而变化 [3,8]。经过微调的混合系统可以获得较强性能和外部验证结果，但需要大规模标注数据和模型训练 [5]。分层提示、验证层和重试机制也可以在不采用传统微调的情况下提高提取质量 [6,7]。这些研究分别为现代信息提取系统采用的各项技术提供了依据。

各项研究中人工参与的方式也不相同。临床医生或医学专家通常负责建立金标准标注、解决分歧或指导术语选择。这类工作可以判断模型是否匹配参考标签，但不能回答另一个问题：当两个完整系统的输出都包含部分正确信息时，执业肿瘤科医生是否认为其中一个系统更忠实、更完整，也更有临床用途。在我们对上述 24 项研究范围综述的补充方法表进行审查时，多数论文报告的是相对于标签的自动性能指标。只有两项研究明确使用五点 Likert 量表评估生成结果，而且均针对放射学报告 [1]。在该综述中，我们没有发现由执业肿瘤科医生在隐藏系统身份的条件下，对完整信息提取系统进行字段级直接比较的研究。

我们把反复出现的模型错误视为具体的工程目标。本文所称的推理框架，是指控制冻结模型如何接受提示、检查答案、修正结果并关联证据的软件层。模型本身不发生变化。当审查发现重复性错误，例如把已停用药物列入当前治疗，或把疑似转移写成确诊疾病时，系统通过提示指令、验证逻辑或确定性临床规则加入范围明确的修正。每项修正都可以记录并接受回归测试。

该框架包含字段路由、选择性上下文传递、生成后验证和确定性临床约束，范围超过单一长提示或检索步骤。不同字段采用不同的提取路径。只有在确实提供相关临床背景时，部分输出才会传递给后续任务。系统记录这些干预，并将最终字段值与病历证据关联起来。在我们审阅的最接近本研究的工作中，尚未发现有研究在真实纵向肿瘤科病历上评估这一完整组合，同时采用同模型基线和肿瘤科医生直接比较。

我们在受控条件下检验了这种方法能否提高提取质量。推理框架和基线使用相同的 Qwen2.5-32B 模型及目标 schema，主要差异是模型外围的推理过程。研究聚焦七个不能仅靠表层实体识别解决的临床问题：当前抗癌治疗、癌症分期、远处转移、区域或总体转移受累、治疗反应、乳腺癌类型与受体状态，以及已完成的分子或遗传检测结果。

我们的假设是，推理框架整体上会优于单提示基线，且提升主要出现在由时态检查、临床分类规则和跨字段一致性检查直接处理的字段中。由于两种条件使用相同的基础模型，我们也预期简单问题会出现较多平局。

## 2. 方法

### 2.1 研究设计

本试点研究包含三个相互衔接的阶段。第一阶段通过一位肿瘤科医生合作者的反复审阅，在乳腺癌病历上开发信息提取流程。第二阶段在没有医生参与该阶段审阅的情况下，将所得框架适配到胰腺癌，并采用人工监督的模型参与错误审查循环。第三阶段将最终框架与基于同一冻结模型的单提示基线进行比较。比较使用 40 份经专家标注的 CORAL 基准病历，并由肿瘤科医生完成隐藏系统身份的 A/B 评估。

三个阶段回答不同的问题。乳腺癌开发阶段检验能否把专科医生反馈转化为可复用的提示和规则。胰腺癌开发阶段检验这些经验能否指导框架适配到一个相关但临床特征不同的领域，而不要求专科医生检查每一轮迭代。最终比较则检验加入推理框架后，系统是否优于未使用该框架的同一模型。

### 2.2 数据集

我们使用 CORAL 数据集，其中包含经专家整理和去标识化的乳腺癌及胰腺癌随访病历 [2]。本项目所用版本包含 200 份没有专家标注的附加病历，每个癌种各 100 份；另有 40 份经专家标注的基准病历，每个癌种各 20 份。有记录的开发迭代覆盖了 56 份乳腺癌未标注病历和全部 100 份胰腺癌未标注病历。最初预留 40 份标注病历用于系统比较。

CORAL 可以公开获取，但其中的记录是真实临床病历，并非来自网页的问题、合成病例或模型生成的叙述。这些病历保留了重复病史、复制到后续记录的内容、不确定检查结果和条件性计划，而这些特征正是纵向肿瘤信息提取的难点。该数据集的价值在于临床真实性和专家标注，而非样本规模。部分相关研究使用规模更大的机构队列，另一些研究则使用范围更窄的操作记录、病理报告或合成肿瘤病历。

开发病历用于识别反复出现的错误模式并改进框架。标注基准病历用于匹配技术比较和临床医生比较。之后对基准集进行的一次技术审查还促成了临床评估版本之前的定向修复，因此临床比较应被视为试点性基准评估，而不是完全未接触数据的一次性外部验证。研究未进行模型权重训练或微调。

### 2.3 基础模型与基线

两个实验条件均使用通过 vLLM 在本地部署的 Qwen2.5-32B-Instruct-AWQ [9]。基线系统对每份病历只调用模型一次，并返回完整的目标 schema。它不使用任务拆分、验证门、重试、词典或确定性后处理。匹配基线与推理框架采用相同的字段定义和输出契约。

### 2.4 推理框架

不同目标字段对上下文和推理的需求并不相同，因此框架采用多个处理阶段。

首先，字段专用提示分别提取就诊背景、癌症诊断、实验室结果、客观发现、当前用药和近期治疗变化。依赖型提示随后接收经过选择的前序结果。例如，分期和转移状态为治疗意图提供背景，当前治疗和临床发现则为疗效评估提供背景。计划类字段主要从 Assessment and Plan 章节提取。

每项模型输出依次通过五个验证阶段：

1. JSON 解析失败时修复格式。
2. 验证 schema，发现错误字段名或泄漏的字段名。
3. 检查具体性和语义一致性，识别含糊或答非所问的答案。
4. 进行忠实度修剪，删除明确无依据或相互矛盾的内容，同时保留有支持的信息。
5. 进行时态过滤，从未来计划字段中删除已完成或历史事件。

确定性处理层使用高置信度临床规则解决反复出现的错误。这些规则区分区域淋巴结和远处疾病、疑似和确诊转移、当前抗癌治疗与支持治疗或居家用药，以及当前疗效与治疗前变化。系统还返回支持每个提取值的病历原文片段。

| 框架组件 | 针对的失败模式 | 示例 |
|---|---|---|
| 字段专用提取 | 一个大型提示遗漏或混合字段 | 分开提取当前用药和治疗计划 |
| 依赖感知的上下文 | 相关字段相互矛盾 | 解释分期和疗效时使用转移状态 |
| 语义验证 | 答案与医学相关，但没有回答目标字段 | 从当前疗效中删除未来治疗计划 |
| 忠实度修剪 | 模型增加无依据的结论 | 活检结果待定时，将病灶保留为疑似 |
| 时态过滤 | 已完成的结果出现在未来计划中 | 从影像计划中删除已完成的扫描 |
| 药物与上下文规则 | 仅根据药名分类，忽视临床背景 | 区分抗癌药、居家用药和支持药物 |
| 跨字段临床规则 | 分期、淋巴结和远处疾病不一致 | 将腋窝淋巴结保留为区域受累，而非远处转移 |
| 来源归因 | 审查者无法追溯提取值 | 返回病历中的支持性原句 |

> **图 1 占位：开发、迁移和评估流程。**
>
> **形式：** 三面板横向流程图，附一个较小的受控比较插图。
>
> **面板 A，乳腺癌开发：** 乳腺癌病历 → 框架输出 → 肿瘤科医生审阅 → 反复出现的失败类别 → 修改提示、验证门或确定性规则 → 回归测试。将此阶段标记为 `临床医生指导的规则归纳`。
>
> **面板 B，胰腺癌迁移：** 迁移后的框架 → 胰腺癌病历 → 依据评分准则配置的 Qwen 审查 → 外部开发 LLM 汇总并提出修改方案 → 人工接受或编辑 → 回归测试。明确标注该阶段没有医生审阅胰腺癌输出，且模型权重始终冻结。
>
> **面板 C，最终评估：** 20 份乳腺癌和 20 份胰腺癌基准病历 → 完整推理框架和匹配的单提示基线 → 向肿瘤科医生展示隐藏系统身份的 A/B 输出。
>
> **受控比较插图：** 在两个分支上方显示共享的冻结 Qwen2.5-32B-Instruct-AWQ 模型和目标 schema。框架分支增加字段路由、选择性上下文传递、五个验证门、确定性肿瘤学钩子、跨字段检查、日志记录和来源归因。基线分支仅使用一个与 schema 匹配的提示。
>
> **图注草稿：** *图 1. 开发与评估流程。由肿瘤科医生发现的乳腺癌反复性错误被转化为明确的框架组件。随后，研究通过 AI 辅助、人工监督的审查，将框架适配到胰腺癌；该开发阶段没有医生参与。最终，在隐藏系统身份的肿瘤科医生评估中，将框架输出与使用相同模型的单提示基线进行比较。*

### 2.5 临床医生指导的乳腺癌开发

乳腺癌开发在 56 份未标注病历上进行了约 15 轮有记录的迭代。一位肿瘤科医生合作者审阅生成结果，并指出具有临床意义的错误，包括治疗状态、受体解释、分期、转移分类和疗效判断错误。开发团队将每项标记结果与完整病历对照，对反复出现的失败进行归类，并将其转化为适用于一般情况的提示、验证逻辑或确定性钩子。肿瘤科医生没有建立传统的监督训练集，模型权重也没有更新。

这一过程可称为临床医生指导的规则归纳。最终保留下来的是一套从重复失败模式中总结出的明确指令和检查，而不是逐例修正后的答案。例如，审查发现腋窝淋巴结受累被当作远处转移后，团队将其转化为通用的区域淋巴结规则；发现计划使用的药物被列为当前治疗后，则形成了药物时态规则。每项高影响修改都会在受影响病例和此前正确的对照病例上重新运行。

### 2.6 模型参与闭环的胰腺癌改进

随后，我们通过约 18 轮有记录的开发，将乳腺癌框架适配到全部 100 份未标注胰腺癌病历。在这一阶段，没有医生审阅胰腺癌输出。每轮包含四个步骤：pipeline 生成结构化字段；依据评分准则配置的 Qwen 审查模型将这些字段与完整源病历进行比较；开发环境中的外部通用 LLM 汇总累积错误，并提出提示、钩子或工作流程修改方案；人工开发者检查、接受或修改这些方案，并进行回归测试。

审查提示包含字段定义、严重程度标准，以及乳腺癌开发期间形成的临床偏好。这样既能让已有错误类别指导新癌种的审查，也能发现胰腺癌特有的问题，例如治疗方案名称和剂量表达。该流程不允许部署模型自行改写代码，也不属于自主自我演化。我们将其称为 AI 辅助、人工监督的改进，因为模型同时参与了错误发现和修改建议，而具体实施仍由人工控制。

在两个开发阶段，pipeline 都记录原始模型输出、每项验证操作和确定性修正。这些记录使团队能够追溯最终字段值在框架中的处理过程，并根据回归结果保留或拒绝修改建议。

### 2.7 预先设定的临床字段

匹配比较前预先选定了七个核心字段：

1. 患者当前正在接受哪些抗癌药物？
2. 当前癌症分期是什么？
3. 是否存在远处转移，远处转移是存在、不存在还是不确定，涉及哪些部位？
4. 有哪些区域或总体转移受累得到证据支持？
5. 癌症对当前治疗的反应如何？
6. 乳腺癌类型及 ER/PR/HER2 状态是什么？
7. 病历记录了哪些已经完成的分子或遗传检测结果？

肿瘤科医生评估工具还包括遗传检测计划、支持用药、操作计划、影像计划、实验室检查计划、用药计划和近期治疗变化。实验室结果摘要和一般临床发现为选评字段，不纳入主要临床分析。

### 2.8 技术评估

完整匹配审查包含 260 项适用的病历字段比较。基于源文本的 LLM 辅助审查流程读取每项比较对应的源病历和两个系统输出。每项结果被归类为框架更优、基线更优或平局。所有报告为框架落后的项目，以及存在争议的高严重程度问题，都再次与源病历核对。该审查用于开发和技术错误分析，不能替代临床医生评估。

完整 v2.2 审查发现的四项高影响失败随后得到修复。研究对受影响病例和无问题对照病例重新运行了定向回归集。由于该定向数据集并非对完整队列的新一轮估计，其结果单独报告。

### 2.9 肿瘤科医生评估

临床评估通过隐藏系统身份的 A/B 界面展示源病历和两份结构化输出。评估者针对每个字段选择 A 更优、B 更优或平局。界面不会显示哪份输出来自推理框架。

[投稿前 TODO：说明参与乳腺癌开发的肿瘤科医生是否也是最终评估者之一。如果是，应区分对系统身份实施盲法和评估者独立于开发过程这两个概念。]

第一位肿瘤科医生完成了全部 280 项乳腺癌必评判断和全部 260 项胰腺癌必评判断。第二位医生完成了全部 280 项乳腺癌判断和 260 项胰腺癌判断中的 259 项；缺少的是 `p7 / lab_plan`。第三位医生完成了全部 280 项乳腺癌判断，没有评估胰腺癌。三份原始导出均原样保留。更新界面中显示的结果后，评分模板仍保留了旧文件名。项目负责人确认三位医生评估的都是较新输出，但投稿前仍须填入界面所展示 PL 和 BL 文件的准确哈希值：[TODO]。

三位肿瘤科医生均完成了同样的 280 项乳腺癌必评比较。我们计算了每两位医生之间的完全一致率和 Cohen's kappa，并分别汇总五组已完成的评审者与癌种组合。合并计数仅作描述性统计，不把各字段判断视为相互独立的观测。

主要分析将使用混合效应 logistic 回归模型，在非平局评分中比较框架与基线的偏好，并将评估者、病历和字段作为分组因素。平局将单独报告，并纳入敏感性分析。我们将报告效应估计值、95% 置信区间、双侧 p 值和评估者间一致性。在完成该模型及其敏感性分析前，目前三位医生的汇总结果仍按描述性结果报告。

### 2.10 探索性患者信件评估

患者信件生成研究早于仅评估信息提取的研究，并促使我们转向控制更严格的任务。一位肿瘤科医生评估了 20 份乳腺癌病历，每份病历对应三封患者信件。三个系统分别显示为 A、B 和 C：由单提示 GPT-4o 生成的信件、由单提示 Qwen2.5-32B 生成的信件，以及根据推理框架输出由 Qwen2.5-32B 生成的信件。评估者使用五点量表对准确性、完整性、可理解性和实用性评分，同时标记可能存在的幻觉，并评价部署就绪程度。

该评估的样本量和设计均不足以作为主要的三系统比较试验。因此，我们仅将其作为结构化信息提取下游用途的描述性证据进行报告。该结果不应与信息提取偏好计数合并，也不能用于声称完整患者信件系统具有优势。

## 3. 结果

### 3.1 开发路径与跨癌种迁移

开发记录包括对 56 份乳腺癌病历进行的约 15 轮迭代，以及对 100 份胰腺癌病历进行的约 18 轮迭代。乳腺癌部分的修订直接参考了肿瘤科医生的审查意见。胰腺癌部分的修订则没有医生审查相应输出，而是采用前述迁移后的评估标准、基于模型的错误审查、外部 LLM 辅助综合，以及由人工控制的实施流程。

开发中有一部分组件可以原样迁移，包括五个验证阶段、时态区分、来源归因，以及区分当前治疗、治疗计划和支持性用药的规则。疾病术语、治疗方案解读和后处理条件则需要按癌种分别处理。因此，该系统复用了错误处理框架，但没有假设乳腺癌与胰腺癌在临床上可以互换。

胰腺癌的临床医生评估是检验这种迁移的主要证据。在两次独立评估中，医生有 161 次偏好推理框架、23 次偏好基线，另有 335 次平局。第一位医生按病历计算得到 19 例正向净差和 1 例负向净差，第二位医生得到 18 例正向净差和 2 例平局。合并两位医生的判断后，20 例胰腺癌病例的框架减基线净差均为正值。由于开发期间没有医生审查胰腺癌输出，这一模式与先前编码的评估标准和工作流程具有可迁移性相符。但仅凭该结果，无法判断收益来自迁移规则、胰腺癌特异性修订、基于模型的审查器，还是这些因素的共同作用。

### 3.2 当前肿瘤科医生评估

目前的临床证据包括五组已完成的评审者与癌种组合：三位肿瘤科医生均完成了乳腺癌评估，其中两位还完成了胰腺癌评估。在 1,359 项必评字段判断中，框架获偏好 443 次，基线获偏好 77 次，另有 839 次平局。在 520 项非平局判断中，框架占 85.2%。

| 已完成的评估 | 必评判断数 | 框架 | 基线 | 平局 | 框架在非平局判断中的占比 |
|---|---:|---:|---:|---:|---:|
| 肿瘤科医生 01，乳腺癌 | 280 | 84 | 22 | 174 | 79.2% |
| 肿瘤科医生 01，胰腺癌 | 260 | 75 | 14 | 171 | 84.3% |
| 肿瘤科医生 02，乳腺癌 | 280 | 79 | 8 | 193 | 90.8% |
| 肿瘤科医生 02，胰腺癌 | 259 | 86 | 9 | 164 | 90.5% |
| 肿瘤科医生 03，乳腺癌 | 280 | 119 | 24 | 137 | 83.2% |
| **所有已完成的评估** | **1,359** | **443** | **77** | **839** | **85.2%** |

> **图 2 占位：按评审者和癌种展示临床医生偏好分布。**
>
> **图形：** 五条 100% 堆叠水平条形图。
>
> **x 轴：** 必评字段判断的占比，范围为 0% 至 100%。
>
> **y 轴：** 五组已完成的评审者与癌种组合，包括三组乳腺癌评估和两组胰腺癌评估。
>
> **编码：** 蓝色表示偏好框架，浅灰色表示平局，橙色表示偏好基线。在各区段内标注原始计数，并在每条横条右侧标注框架在非平局判断中的占比。
>
> **观察到的趋势：** 平局在每组评估中均为最大类别。五组评估中，对框架的偏好都明显多于对基线的偏好；框架在非平局判断中的占比为 79.2% 至 90.8%。
>
> **图注草稿：** *图 2. 五组已完成的评审者与癌种组合中，隐藏系统身份后的临床医生偏好分布。由于两个系统使用相同的基础模型，多数判断为平局；在非平局判断中，结果始终偏向推理框架。*

三位肿瘤科医生在乳腺癌数据集上均独立偏好框架。合并三位医生的乳腺癌评估后，框架获偏好 282 次，基线获偏好 54 次，另有 504 次平局。第一位肿瘤科医生按病历评估的结果为框架胜出 18 例、基线胜出 1 例、平局 1 例。第二位和第三位医生均在 20 份病历中全部偏向框架。将三位医生对每份乳腺癌病历的评分合并后，20 份病历的框架减基线净差均为正值。

两次胰腺癌评估合计包括 161 项框架偏好、23 项基线偏好和 335 项平局。有一个必评判断 `p7 / lab_plan` 缺失。合并两位医生的判断后，20 份胰腺癌病历的框架减基线净差均为正值。

最终混合效应分析结果尚待完成：

> [最终多评审者结果：在 FINAL N 位肿瘤科医生和 FINAL N 项可评估判断中，校正分析显示框架获得显著偏好，效应估计值为 FINAL，95% CI FINAL，p=FINAL。]

### 3.3 评审者间一致性与核心字段

三位肿瘤科医生均完成了同样的 280 项乳腺癌必评比较。医生 01 与 02 的完全一致率为 82.9%，医生 01 与 03 为 80.0%，医生 02 与 03 为 73.6%。对应的 Cohen's kappa 分别为 0.646、0.644 和 0.511。三位医生在 192 项比较中给出相同判断，占 68.6%。按简单多数票计算，95 项偏向框架，15 项偏向基线，168 项为平局；另有 2 项分别得到一票框架、一票基线和一票平局，因此没有多数结果。

> **图 3 占位：乳腺癌判断的两两评审者一致性。**
>
> **图形：** 三位肿瘤科医生构成的对称 3 × 3 矩阵。
>
> **上三角：** 两两完全一致率。
>
> **下三角：** 两两 Cohen's kappa。
>
> **编码：** 单元格颜色深浅表示一致性强弱。每个非对角单元格显示百分比或 kappa 值，对角单元格注明每位医生均完成 280 项乳腺癌比较。
>
> **观察到的趋势：** 两两完全一致率为 73.6% 至 82.9%，kappa 为 0.511 至 0.646。三位医生使用平局选项的频率不同，但总体结果方向一致。
>
> **图注草稿：** *图 3. 三位肿瘤科医生对 280 项共同乳腺癌字段比较的两两一致性。完全一致率为 73.6% 至 82.9%，Cohen's kappa 为 0.511 至 0.646。*

在所有已完成的评估中，七个预设核心类别共有 660 项适用判断。框架获偏好 276 次，基线获偏好 34 次，另有 350 次平局。因此，在 310 项核心字段的非平局判断中，框架占 89.0%，且每个核心类别的汇总净差均为正值。

| 核心字段 | 框架 | 基线 | 平局 | 净优势 |
|---|---:|---:|---:|---:|
| 当前抗癌药物 | 83 | 2 | 15 | +81 |
| 分期 | 41 | 4 | 55 | +37 |
| 远处转移 | 15 | 2 | 83 | +13 |
| 区域或总体转移 | 60 | 5 | 35 | +55 |
| 治疗反应 | 37 | 3 | 60 | +34 |
| 乳腺癌类型与受体状态 | 18 | 12 | 30 | +6 |
| 已完成的分子或遗传检测结果 | 22 | 6 | 72 | +16 |
| **总体** | **276** | **34** | **350** | **+242** |

> **图 4 占位：各核心临床类别的偏好分布。**
>
> **图形：** 七条 100% 堆叠水平条形图，按框架净优势排序。
>
> **x 轴：** 适用临床医生判断的占比，范围为 0% 至 100%。
>
> **y 轴：** 当前抗癌药物、区域或总体转移、分期、治疗反应、已完成的分子或遗传检测结果、远处转移，以及乳腺癌类型或受体状态。
>
> **编码：** 蓝色表示偏好框架，浅灰色表示平局，橙色表示偏好基线。在每条横条右侧显示 `PL / BL / TIE` 计数。由于受体状态仅适用于乳腺癌，视觉比较采用百分比，并在标签或图注中保留原始分母。
>
> **观察到的趋势：** 当前抗癌药物以及区域或总体转移的净优势最大。分期和治疗反应也偏向框架。远处转移以平局为主，乳腺癌类型或受体状态则接近持平。
>
> **图注草稿：** *图 4. 各预设核心临床类别中的临床医生偏好。框架在识别当前治疗和判断转移累及方面优势最大。乳腺癌类型与受体状态的净优势最小。*

当前抗癌药物以及区域或总体转移的优势最大，也最为稳定。在七个核心类别之外，药物计划共有 59 次偏好框架、0 次偏好基线和 41 次平局。操作或手术计划的差距较小，共有 14 次偏好框架、12 次偏好基线和 74 次平局。乳腺癌类型与受体状态仍是表现最弱的核心类别，净优势为 6 项判断。

探索性的病历级汇总得出了相同方向的结果，同时避免将每个字段视为相互独立的观察值。按癌种合并医生判断后，20 份乳腺癌病历和 20 份胰腺癌病历的框架减基线净差均为正值。五组评审者与癌种组合的汇总结果也全部偏向框架。这些汇总并非预设的最终模型，不能替代计划中的评审者、病历和字段联合分析。

> **图 5 占位：每份病历的临床医生偏好净差。**
>
> **图形：** 两个对齐的条形图面板，分别展示乳腺癌和胰腺癌，并设置水平零参考线。
>
> **x 轴：** 样本标识符。乳腺癌面板为 `b1` 至 `b20`，胰腺癌面板为 `p1` 至 `p20`。
>
> **y 轴：** 每份病历的标准化偏好净差，计算方式为 `(框架胜出字段数 - 基线胜出字段数) / 该病历已完成的必评字段判断数`，两个面板使用相同刻度。乳腺癌面板合并三位肿瘤科医生的判断，胰腺癌面板合并两位医生的判断。
>
> **编码：** 零线上方的蓝色条形表示偏好框架，零线下方的橙色条形表示偏好基线，位于零线上的灰色标记表示病历级净差为平局。
>
> **观察到的趋势：** 合并后的 20 份乳腺癌病历和 20 份胰腺癌病历净差均为正值。
>
> **图注草稿：** *图 5. 各份病历中标准化临床医生偏好净差的分布。按癌种合并现有医生评分后，框架在每份乳腺癌和胰腺癌病历中均取得正向净差。*

> **图 6 占位：校正后的多评审者效应估计。**
>
> **图形：** 森林图，待计划中的最终临床医生样本完成后填充。
>
> **x 轴：** 非平局判断偏好框架而非基线的校正比值比。采用对数刻度，并在 1.0 处设置垂直参考线。
>
> **y 轴：** 总体效应、乳腺癌、胰腺癌，以及七个预设核心类别。仅纳入最终样本量足以支持的分层结果。
>
> **编码：** 点估计值及其 95% 置信区间。主要校正分析使用实心标记，以不同方式处理平局的敏感性分析使用空心标记。
>
> **预期解释：** 该图将展示考虑评审者、病历和字段重复判断后的估计结果，避免仅依赖合并字段计数。在最终模型拟合前，不应宣称任何趋势。
>
> **图注草稿：** *图 6. 评估条件与临床医生偏好之间的校正关联。比值比大于 1 表示偏好推理框架。估计值将来自预设的聚类多评审者分析。*

### 3.4 完整匹配技术审查

在 260 项适用的核心字段比较中，框架胜出 66 次，基线胜出 28 次，另有 166 次平局。七个类别中有六个类别的框架净差为正。分期是完整 v2.2 审查中唯一净差为负的类别。

| 核心字段 | 框架 | 基线 | 平局 | 净优势 |
|---|---:|---:|---:|---:|
| 当前抗癌治疗 | 8 | 0 | 32 | +8 |
| 分期 | 6 | 8 | 26 | -2 |
| 远处转移 | 11 | 3 | 26 | +8 |
| 区域或总体转移 | 14 | 4 | 22 | +10 |
| 治疗反应 | 13 | 6 | 21 | +7 |
| 乳腺癌类型与受体状态 | 8 | 5 | 7 | +3 |
| 已完成的分子或遗传检测结果 | 6 | 2 | 32 | +4 |
| **总体** | **66** | **28** | **166** | **+38** |

> **补充图 S1 占位：技术审查与临床医生评估的净偏好率。**
>
> **图形：** 水平哑铃图，每个核心类别占一行。
>
> **x 轴：** 净偏好率，计算方式为 `(框架较优数 - 基线较优数) / 适用判断数`，并在零处设置垂直参考线。
>
> **y 轴：** 七个核心临床类别。
>
> **编码：** 一个标记表示完整 v2.2 来源可追溯技术审查，另一个标记表示目前已完成的临床医生评分，并连接每个类别内的两个标记。
>
> **观察到的趋势：** 两类证据总体上都偏向框架。临床医生评分显示，当前抗癌药物以及区域或总体转移的净优势尤其明显。分期从技术审查中的小幅负净差变为临床医生评估中的正净差，而远处转移和受体状态的临床医生评估净差较小。
>
> **解释警示：** 两组结果的评审者和所评估的框架版本不同，因此这里只能描述性比较结果模式，不能视为一致性检验或具有因果意义的前后比较。
>
> **图注草稿：** *补充图 S1. 完整技术审查和当前临床医生评估中各类别的净偏好率。由于两者的审查流程和框架版本并不相同，各系列之间的差异只能作描述性解释。*

### 3.5 定向修复评估

我们采用保守规则修复了 v2.2 审查发现的四个高影响错误。在 6 个受影响样本和 2 个对照样本的 51 项适用比较中，修复后的框架胜出 29 次，基线胜出 0 次，另有 22 次平局。该定向样本集中不再存在 P0 错误，两个对照样本也均未出现可检测到的核心字段回归。这些定向结果表明，已发现的错误可以修复，但不能替代完整的 40 份病历评估结果。

### 3.6 定性评论

三份导出中有两份包含书面评论，共 12 条。这些评论提供了有用的背景，也说明偏好计数不能作为完整的正确性评估。在一项乳腺癌影像计划比较中，第一位肿瘤科医生偏好框架，理由是基线总结了已完成的影像发现，而没有提取计划中的 PET/CT。第二位肿瘤科医生将同一比较评为平局，并指出两个答案都没有正确反映当前并无新影像计划。在将该病例用作论文示例前，应先裁定这一分歧。

其他评论指出了无依据的受体状态、区域淋巴结遗漏、二线方案刚开始后疗效尚不确定，以及两个输出均不准确的病例。这些观察说明专科医生审查具有实际价值，也为最终错误分析提供了具体目标。它们还表明，平局既可能表示两个输出都足够准确，也可能表示两个输出都存在错误。

> **补充图 S2 占位：对平局判断的裁定后解释。**
>
> **图形：** 对预设平局样本完成人工裁定后，绘制堆叠条形图或紧凑型冲积图。
>
> **x 轴：** 平局的解释类别，包括两个输出在临床上均可接受、两个输出部分正确但不完整、两个输出均不正确，以及无法根据病历判断。
>
> **y 轴：** 已裁定平局的数量或百分比。
>
> **编码：** 按癌种或核心字段拆分条形。可使用次级标注说明两个输出是否因相同或不同原因而失败。
>
> **数据状态：** 尚不可用。当前评分界面只记录 `TIE`，因此该图需要人工审查，不能根据现有 CSV 文件推断。
>
> **图注草稿：** *补充图 S2. 人工裁定后对平局判断的临床解释。该分析区分了两个输出均正确且等效的情况，以及两个系统均不完整或不正确的比较。*

### 3.7 探索性患者信件结果

早期患者信件评估没有显示框架具有一致优势。同模型 Qwen 基线的四项平均分最高，为 3.80；框架生成的信件为 3.74，ChatGPT 为 3.50。三个系统均有 3/20 封信件被评为无需编辑即可发送。临床医生在 4 封 ChatGPT 信件、1 封 Qwen 基线信件和 2 封框架信件中标记了可能的幻觉内容。

| 信件系统 | 准确性 | 完整性 | 易理解性 | 实用性 | 四项平均分 | 标记为存在幻觉 |
|---|---:|---:|---:|---:|---:|---:|
| ChatGPT 单提示 | 3.85 | 3.25 | 3.60 | 3.30 | 3.50 | 4/20 |
| Qwen 单提示基线 | 4.10 | 3.60 | 3.85 | 3.65 | 3.80 | 1/20 |
| 基于 Qwen 推理框架的信件 | 3.95 | 3.60 | 3.85 | 3.55 | 3.74 | 2/20 |

按病历配对比较可以更清楚地呈现结果趋势。与 ChatGPT 信件相比，框架信件的四项平均分在 14 例中更高，2 例持平，4 例更低，配对平均差为 0.24 分。排除平局后，未经校正的双侧精确符号检验结果为 `p=0.031`。这项分析属于探索性分析，并非预设分析，而且研究考察了多个相关结局。与同模型 Qwen 基线相比，框架胜出 9 例、持平 5 例、落后 6 例，配对平均差为 -0.06 分。因此，结构化提取可以作为患者沟通内容的输入，但这些结果尚不能证明当前信件生成器优于对同一模型直接使用提示。

> **图 7 占位：患者信件评分的配对差值。**
>
> **图形：** 两个配对差值面板，一个比较框架信件与 ChatGPT，另一个比较框架信件与 Qwen 单提示基线。
>
> **x 轴：** 乳腺癌样本标识符，`b1` 至 `b20`。
>
> **y 轴：** 准确性、完整性、易理解性和实用性四项平均分之差。正值表示框架信件更优。
>
> **编码：** 零线上方使用蓝色点，零线上使用灰色点，零线下方使用橙色点。添加水平零线，并在每个面板中标注胜出、平局和落后的数量。
>
> **观察到的趋势：** 从描述性结果看，框架信件在多数病历中优于 ChatGPT，但与同模型 Qwen 基线的分布接近，既有胜出也有落后。
>
> **图注草稿：** *图 7. 20 份乳腺癌病历中，肿瘤科医生所评患者信件质量的探索性配对差值。框架信件超过 ChatGPT 的情况更多，但相较同模型单提示基线并未显示明确优势。*

## 4. 讨论

### 4.1 主要解读

[最终开头：多位肿瘤科医生参与的评估显示，相较于使用同一模型的单提示基线，临床医生显著更偏好推理框架。]

本文的主要结果来自模型外围的开发流程。临床医生帮助识别乳腺癌场景中反复出现的错误，团队将这些错误转化为推理框架中的明确组件，随后在胰腺癌开发阶段没有医生参与审阅的情况下，将这一流程适配到胰腺癌。最终的临床评分在两个癌种中都更偏向该框架。与静态比较两个提示相比，这一过程更直接地将开发方法与评估结果联系起来。

多数字段比较为平局，这是可以预期的，因为两个系统使用同一个能力较强的基础模型。不过，当肿瘤科医生认为两者存在实质差异时，偏好推理框架的次数接近偏好基线的六倍。优势主要集中在需要时间关系判断或临床分类的字段。这符合系统的设计思路：保留基础模型已经答对的简单问题，只在出现已知失败模式时介入。

这些证据仍属于试点结果。乳腺癌和胰腺癌开发阶段没有采用随机设计，优化过程结合了多种工具，后续的定向修复也参考了基准集上的发现。因此，结果支持的是整体开发策略，不能据此推断某个模型、提示、验证门或钩子单独产生了因果作用。

### 4.2 乳腺癌中由临床医生指导的规则归纳

乳腺癌开发阶段中，肿瘤科医生的角色不同于常规的数据集标注。医生没有为模型训练制作成千上万个字段标签，而是审阅具体输出，指出可能影响实际使用的错误。团队随后将反复出现的错误转化为可复用的指令和检查规则。

当专科医生的时间有限时，他们可以集中界定困难概念的边界，例如何为当前治疗、如何判断当前方案的疗效，以及如何区分区域病变和远处转移，而不必审阅以后生成的每一份病历。规则编码后可以被一致地执行、记录并接受回归测试。三次独立的乳腺癌评估表明，得到的推理框架不只是复现某一位临床合作者的偏好。三位肿瘤科医生都更偏好该框架，但他们的具体判断并不完全相同。

这一开发方式也说明了我们为何将该系统称为推理框架。其贡献在于围绕一个权重冻结的模型，逐步积累可执行的临床区分。有些区分写入提示，有些放在验证阶段，还有一些由确定性代码执行。临床医生界定临床边界，工程流程则将这些边界转化为可重复的系统行为。

### 4.3 胰腺癌中的迁移与 AI 辅助改进

胰腺癌阶段检验的是，已经形成的流程能否迁移到临床医生没有直接提供反馈的领域。胰腺癌病历在疾病进程、治疗方案、分期表述和手术背景方面均有不同。因此，我们没有直接沿用所有乳腺癌专用规则，而是迁移通用工作流程，保留体现共同临床区分的规则，并针对胰腺癌病历暴露出的新失败模式增加专用路由。

这一开发阶段没有医生审阅胰腺癌输出。一个 Qwen 审查模型按照继承的评分准则检查每份输出，另一个外部开发用 LLM 汇总审查记录并提出修改建议。最终由人工开发者决定实施哪些修改，并运行回归测试。两次胰腺癌评估合计包括 161 次框架偏好、23 次基线偏好和 335 次平局。这一结果与模型参与闭环流程下的成功迁移一致。

我们不将这一过程称为自主进化。部署的模型没有独立修改代码，也没有自行批准修改。只有在较窄的意义上，也就是模型在多轮迭代中协助识别错误和形成修改方案时，才适合使用 `AI 辅助演进` 这一说法。人工监督始终是开发闭环的一部分。

### 4.4 哪些问题对模型而言更困难？

综合临床评分可以区分相对直接的提取任务与需要临床语境判断的问题。

直接陈述的事实往往得到平局。两个系统通常都能识别明确写出的影像结果、操作或受体数值。更难的是判断这些事实在当前临床语境中意味着什么。

当前使用的抗癌药物呈现出最大且最稳定的核心字段差异。在已完成的评估中，推理框架获得 83 次偏好，基线获得 2 次偏好，另有 15 次平局。该字段要求模型区分抗癌治疗、长期居家用药、支持治疗药物、已停用方案和未来选项。用药计划也明显偏向该框架，结果为 59 比 0，另有 41 次平局，因为未来行动必须与当前治疗及近期变化分开。

分期与转移累及需要相互关联的判断。模型必须区分区域淋巴结与远处播散，为等待确认的病灶保留不确定性，并使转移状态与分期保持一致。在现有评分中，区域或总体转移累及以 60 比 5 偏向推理框架，分期则以 41 比 4 偏向该框架。该框架会跨字段检查这些关系，而不是孤立地提取每个标签。

疗效评估在概念上仍然困难，但临床结果的方向一致。推理框架获得 37 次偏好，基线获得 3 次偏好，另有 60 次平局。一份病历可能同时包含既往进展、当前症状、稳定影像、肿瘤标志物趋势和刚刚开始的新方案。判断哪些证据反映当前治疗的疗效需要理解时间线，不能只依赖关键词识别。第二位肿瘤科医生指出，开始二线治疗后疗效尚不明确，这条评论正好说明了这一时间边界。

肿瘤类型和受体状态仍是表现最弱的核心类别。在三次乳腺癌评估中，推理框架以 18 比 12 领先基线，另有 30 次平局。这些信息通常在病历中有明确表述，因此基线也能表现良好。当病历包含双侧病变、历史与复发标本，或不一致的受体结果时，该字段会变得困难。因此，这一字段仍需临床复核，不能作为本系统的明确优势。

### 4.5 观察到的模式与推理框架的关系

本研究没有进行完整的组件消融，因此无法确定每项改进来自哪个模块。不过，字段层面的结果与若干组件的预期作用一致。

当前用药的结果与专用药物提示、肿瘤药物词典、支持治疗药物和居家用药的语境分类，以及时态过滤相符。用药计划的结果与从 Assessment and Plan 部分提取计划并移除已完成行动的处理相符。分期和转移结果则与跨字段语境，以及区分区域病变和远处病变、疑似发现和确诊发现的确定性规则相符。

大量平局也提供了反向检验。确定性规则没有在所有字段上形成明显优势。在已完成的评估中，操作计划的结果接近持平，近期治疗变化在胰腺癌子集中也完全持平。因此，仅以输出长度或评分者整体偏好推理框架来解释结果，并不充分。最明显的差异出现在系统设置了明确防护规则的字段。

这些关联支持我们提出的作用机制，但不能证明该机制。未来的组件研究应比较单提示、仅使用任务分解的提示、提示加验证，以及完整推理框架，从而估计每一层的贡献。

### 4.6 与相关工作的比较

最接近的研究采用了不同的数据、模型适配方式和人工审阅方案。其中一些研究有临床医生参与，但医生通常负责建立参考标准或指导模型开发。较少有研究要求肿瘤专科医生直接比较两个系统的最终输出。

| 研究 | 数据与任务 | 临床专家的角色 | 主要方法 | 与本研究的差异 |
|---|---|---|---|---|
| Sushil 等，CORAL [2] | 40 份真实乳腺癌和胰腺癌病历；广泛的肿瘤学 schema | 专家标注；一位独立肿瘤科医生人工评估每个癌种的 10 份 GPT-4 输出 | 零样本 GPT-4、GPT-3.5-turbo 和 FLAN-UL2 | 使用相同的数据集和临床范围。该研究建立了基准，并记录了遗漏和幻觉，但没有在同一冻结模型上比较由失败模式驱动的推理框架。 |
| Wiest 等 [3] | 500 份 MIMIC 病史；五个二分类临床特征 | 三位盲法医学专家建立共识金标准 | 本地 Llama 2、受语法约束的 JSON 和多种提示变体 | 具有严格的专家验证和本地部署，但目标是肿瘤学之外的五个二分类特征。专家提供金标准，而不是对两个完整系统进行比较性偏好评分。 |
| Bhattarai 等 [4] | 63 位肺癌患者的 13,646 份病历；四个纵向表型 | 两位领域专家提供金标准人工标注 | 比较 GPT、开放模型和基于规则的方法 | 纵向语料规模更大，并有专家标签，但目标表型较少，也没有通过同模型比较来隔离外围工作流程的作用。 |
| Tariq 等 [5] | 内部数据包含 26,692 位乳腺癌患者，外部数据包含 162 位患者；治疗时间线 | 癌症登记数据提供治疗标签；临床专家整理治疗概念和编码 | UMLS 解析器加微调问答模型 | 数据规模和外部验证明显更强，但需要有监督微调，并且聚焦五类治疗，而不是广泛的字段级提取。 |
| Dao 等 [6] | 220 份开发用和 200 份验证用右心导管检查病历 | 一位肺血管疾病专家建立验证金标准并指导开发 | 本地开放模型、工程化预加载、验证和重试 | 工作流程架构最为接近。其任务是从操作记录中提取数值，系统不包含肿瘤学专用字段路由和跨字段临床钩子。 |
| Zhang 等，mCODEGPT [7] | 1,000 份合成肿瘤学病历；49 个 mCODE 实体 | 程序化匹配辅以人工审阅者的手动验证 | 分层提示与单步提示比较 | 直接表明提示分层有帮助，但病历是合成数据，也没有采用肿瘤科医生对最终输出的盲法比较。 |
| Grothey 等 [8] | 579 份德语和英语前列腺病理报告 | 一位受过训练的医学博士生在主治病理医师指导下标注报告 | 多个开放和专有模型，以及提示和量化测试 | 专家标注基准更大，但只评估一种报告类型和 11 个预定义参数，而不是异质的纵向门诊病历。评估依赖参考标签，而非最终使用者的偏好。 |

这些论文表明，临床医生并非没有参与这一领域的研究，区别在于他们在哪个环节参与。标注、裁决和术语设计会在评估前提供参考答案。那项涵盖 24 项研究的范围综述主要比较精确率、召回率、F1、AUC 和准确率。在我们对其补充方法学表格的核查中，只有两个条目明确使用五点 Likert 量表评价生成结果，而且两项都涉及放射学报告 [1]。我们计划中的研究增加了执业肿瘤科医生的下游审阅。他们阅读原始病历，在不知道系统身份的情况下逐字段比较最终输出。这种设计能够在两个输出都部分正确、不确定性具有临床意义，或某个答案更完整但并未降低忠实度时，记录临床偏好。

我们的开发设计还有第二个区别。专科医生的反馈用于在一个癌种中发现可复用的失败规则，随后在第二个癌种的开发阶段不再使用医生反馈。因此，胰腺癌结果检验的是已经形成的工作流程能否减少对稀缺专科医生时间的反复占用，而不是要求医生参与每一轮迭代。在上述比较研究中，我们没有发现这种特定的开发和迁移设计。

数据贡献也需要准确表述。CORAL 是公开数据集，但其中包含真实、去标识化的纵向肿瘤学病历和专家标注。这与网络问答、合成病历和范围较窄的报告模板不同。不过，它在数据规模和外部验证方面并未超过规模最大的机构研究。CORAL 对本实验的价值在于，其病历保留了推理框架所要处理的临床歧义。

我们的技术主张针对的是组件组合。提示工程、检索、护栏、重试和混合规则在已有文献中都已出现。在本节所比较的最接近研究中，我们没有找到与本项目完全相同且经过评估的组合，包括字段专用路由、选择性依赖传递、五个验证阶段、肿瘤药物和术语资源、确定性临床钩子、跨字段一致性检查、操作日志、来源归因，以及用于排除模型能力这一主要解释的同模型基线。本文的贡献是这一整合式推理框架及其评估，而不是任何单一组件。

若干相关研究使用了更大的数据集、更多标注者或外部验证。本试点不应被描述为首个临床提取系统或规模最大的评估。更准确的表述是，本研究集中检验结构化、可审计的工作流程能否让同一个冻结模型在困难的肿瘤学字段上表现得更可靠，并由理解这些临床区分的医生直接审阅结果。

### 4.7 从结构化提取到患者沟通

早期患者信件实验的结果并不一致。按四项指标的平均分计算，在 20 例乳腺癌病例中，基于推理框架的信件有 14 例在描述性比较中优于 ChatGPT，幻觉标记也更少，但它们没有明确优于使用同一模型的 Qwen 基线。因此，更好的提取不会自动产生更好的完整信件。信件质量还取决于内容选择、组织方式、措辞、对不确定性的解释，以及应向患者提供哪些临床细节。

这一结果说明，项目应先验证结构化提取，再推进患者沟通。结构化提取是可测量的安全层，患者沟通则是建立在这一层之上的下游任务。提取研究可以先判断分期、治疗、疗效和计划是否得到忠实表达，再由生成器将这些内容写成连贯文字。未来的信件研究应检验具体的提取改进能否经过第二次转换后继续保留，最好同时邀请患者读者和临床医生参与评估。

信件实验仍可作为探索性应用报告。它说明推理框架可以为面向患者的输出提供输入，并且结果在部分维度上可与专有通用模型相比。但这一结果不足以将信件生成作为论文的主要成果，也不能证明当前信件已适合临床部署。

### 4.8 临床与技术意义

研究结果表明，肿瘤学信息提取中的部分 LLM 错误具有足够的重复性，可以在不修改模型权重的情况下处理。当标注训练数据有限，或临床团队需要在不重新训练模型的情况下调整字段定义时，这一点很有价值。例如，将待活检病灶保留为疑似病变的规则可以接受检查和测试。仅使用提示的系统较难提供同等控制，因为一次措辞修改可能影响无关字段。

我们使用推理框架这一名称，是因为系统除提示序列外，还围绕冻结模型管理任务路由、依赖关系、验证、确定性修正、日志和来源归因。临床工作流程也能描述该系统，但可能让人误以为它已经被整合进常规诊疗，而本试点并未检验这一点。因此，本文使用推理框架描述技术贡献，使用评估工作流程描述研究程序。

该系统也支持本地部署。本地运行本身不能证明系统符合隐私要求或具备临床安全性，但能让机构掌控病历处理和系统更新。本研究评估的是提取质量，而不是自主临床使用的准备程度。

### 4.9 试点状态与后续工作

本试点先判断观察到的效应和临床一致性是否足以支持扩大研究规模。三位肿瘤科医生独立复现了乳腺癌评估的总体方向，两位医生也复现了胰腺癌方向。下一步是拟合预先设定的聚类分析模型，并估计不同医生、病历和字段之间的效应差异。

外部验证仍然重要。CORAL 病历来自一个机构，仅覆盖两个癌种。下一步最有信息量的验证数据应来自另一医疗系统，包含纵向肿瘤学病历和独立临床标注，并在评估开始前固定字段契约。

## 5. 局限性

当前临床结果来自三位肿瘤科医生。三人都评估了乳腺癌，但只有两人评估了胰腺癌。现有数据已经显示跨医生复现，但三位评审者仍不足以精确估计肿瘤科医生之间的差异。最终稿应使用计划中的多评审者模型，替代当前的描述性分析。

A/B 界面隐藏了系统身份，但左右位置固定，推理框架始终显示为 A。评审者之间的一致性和大量平局降低了评分者无差别选择 A 的可能性，但不能排除位置偏倚。推理框架的输出还包含来源归因，而基线没有。归因是被评估系统的一部分，但也可能影响偏好。未来研究应随机分配左右位置，并单独检验归因的影响。

偏好标签不能区分 `两者都正确` 和 `两者都错误`。第二份导出中的书面评论明确指出，一些平局实际上是两个输出都不理想。因此，最终报告应将偏好计数与经过裁决的错误类别同时呈现，不能将每个平局都解释为成功。

技术审查使用了 LLM 辅助审阅和反复的错误分析。这一过程有助于识别失败，但不构成独立临床验证。完整的 40 份病历表格对应 v2.2，后续影响较大的修复仅在受影响样本和对照样本上测试，没有重新运行完整数据集。由于基准集在临床评分材料准备完成前已经影响了这些修复，临床比较并不是完全独立的外部验证。

两个开发阶段依次进行，没有采用随机设计。乳腺癌开发结合了肿瘤科医生反馈和工程判断，胰腺癌开发则结合了继承的评分准则、模型审查器、外部开发用 LLM 和人工实施。本研究无法分离其中任何一个要素的贡献。它也没有评估完全自主的系统，因为所有变更都由人工批准并接受回归测试。

数据集规模较小，且来自单一机构。推理框架包含根据已观察错误形成的临床规则，其中一些规则可能反映 CORAL 特有的书写习惯。本研究尚未包含完整的组件消融、其他模型家族或外部病历。

患者信件分析仅涉及一位肿瘤科医生、20 例乳腺癌病例、三个系统和若干相互关联的评分。这一探索性结果不能证明任何系统具有优势，而且没有显示推理框架明确优于使用同一模型的 Qwen 基线。本研究也没有检验患者理解、治疗决策、工作流程效率或临床结局。

## 6. 结论

[最终结论：在一项由多位肿瘤科医生参与、隐藏系统身份的评估中，由失败模式驱动的推理框架在纵向肿瘤学病历结构化提取任务上显著优于使用同一模型的单提示基线。]

当前试点支持一种开发策略：临床医生识别具有临床意义的失败模式，团队将这些模式转化为明确的推理框架，再通过 AI 辅助且由人工监督的改进，将该框架适配到另一个癌种。三位肿瘤科医生在乳腺癌评估中都更偏好该框架，其中两位在胰腺癌评估中也更偏好该框架，而胰腺癌开发阶段没有医生参与。最大的优势出现在当前治疗、用药计划和转移状态字段。模型权重在整个过程中保持冻结。在将这一试点结果上升为确证性结论之前，仍需完成预先设定的聚类分析，并开展外部验证。

## 参考文献

1. Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. *JMIR Cancer*. 2025;11:e65984. [https://doi.org/10.2196/65984](https://doi.org/10.2196/65984).
2. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024;1(4). [https://doi.org/10.1056/AIdbp2300110](https://doi.org/10.1056/AIdbp2300110).
3. Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024;7:257. [https://doi.org/10.1038/s41746-024-01233-2](https://doi.org/10.1038/s41746-024-01233-2).
4. Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024;7(3):ooae060. [https://doi.org/10.1093/jamiaopen/ooae060](https://doi.org/10.1093/jamiaopen/ooae060).
5. Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. *JCO Clinical Cancer Informatics*. 2025;9:e2500002. [https://doi.org/10.1200/CCI-25-00002](https://doi.org/10.1200/CCI-25-00002).
6. Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025;8(5):ooaf097. [https://doi.org/10.1093/jamiaopen/ooaf097](https://doi.org/10.1093/jamiaopen/ooaf097).
7. Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025;5:422. [https://doi.org/10.1038/s43856-025-01116-x](https://doi.org/10.1038/s43856-025-01116-x).
8. Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025;5:96. [https://doi.org/10.1038/s43856-025-00808-8](https://doi.org/10.1038/s43856-025-00808-8).
9. Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025. [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115).
