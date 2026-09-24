<!--
BILINGUAL MAINTENANCE RULE:
1. Keep the complete English manuscript first and the complete Chinese manuscript second in this same file.
2. The two versions must match in scientific claims, numbers, main result tables, figure numbering, and references. The Chinese review version may contain additional annotations and review-only appendices.
3. Keep the English version close to submission prose. The Chinese version may retain collaborator questions, figure-design notes, and explanatory annotations.
4. After changing either version, update the other version in the same edit and run: python3 render_workshop_draft.py
5. Apply the humanizer pass to both versions. Chinese should read as natural academic prose, not as a literal machine translation.
-->

<div id="english-version"></div>

# From clinician-guided error analysis to cross-cancer adaptation: an inference harness for oncology note extraction

**Pilot report**

Version 0.12, September 2026

## Abstract

### Background

Large language models can extract structured information from clinical notes, but a single prompt often confuses current and historical treatment, suspected and confirmed disease, and completed findings and future plans. These errors are difficult to detect because the resulting text remains fluent. Oncology notes are a demanding test case because they combine pathology, imaging, treatment history, current therapy, response assessment, and conditional plans across a long clinical timeline.

### Objective

To test whether an inference harness can improve clinician-assessed oncology information extraction from a frozen, locally served open-weight language model without fine-tuning.

### Methods

We built a failure-mode-driven inference harness around Qwen2.5-32B-Instruct-AWQ in two development stages. During breast-cancer development, a physician coauthor familiar with oncology and the model-development investigators repeatedly reviewed outputs and identified clinically important errors. The team converted recurring errors into field-specific prompts, verification gates, and deterministic oncology rules. We then adapted this harness to pancreatic cancer. No clinician reviewed pancreatic-cancer outputs during development. Instead, a rubric-informed Qwen reviewer identified candidate errors, an external development LLM synthesized the review history and proposed prompt or rule changes, and the investigators selected, implemented, and regression-tested those changes. Model weights remained frozen throughout. We compared the resulting harness with a finalized single-prompt baseline using the same model and field contract on 40 CORAL benchmark notes, including 20 breast and 20 pancreatic cases. Three oncologists completed system-label-masked A/B comparisons. All three evaluated breast cancer, and two also evaluated pancreatic cancer.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. The five completed clinician-by-cancer evaluations contributed 1,359 required-field judgments. The harness was preferred in 443, the baseline in 77, and 839 were ties. Ties accounted for 61.7% of all judgments; among the 520 directional judgments, 85.2% favored the harness. The adjusted directional analysis estimated an odds ratio of 5.70 for harness preference (95% CI 4.24 to 7.66; p<0.001). The three breast-cancer evaluations contributed 282 harness preferences, 54 baseline preferences, and 504 ties. Pairwise exact agreement ranged from 73.6% to 82.9%, with Cohen's kappa from 0.511 to 0.646. In pancreatic cancer, where no physician had participated in development, two oncologists together preferred the harness 161 times and the baseline 23 times, with 335 ties.

### Conclusions

The three breast-cancer evaluations support clinician-informed conversion of recurrent model errors into an explicit inference harness. The two pancreatic-cancer evaluations suggest that the harness can be adapted to a second cancer domain without target-domain clinician review during development. This was model-assisted, investigator-supervised refinement rather than autonomous self-modification. Broader claims still require more oncologists and external validation.

## 1. Introduction

Most clinically useful information in oncology is still recorded in free text. A progress note may contain the diagnosis, pathology, receptor status, treatment history, toxicities, response, and next steps, but these facts are spread across sections and timepoints. Manual review is slow, and conventional extraction systems require substantial annotation and task-specific development. The attraction of large language models is straightforward: one model can read many note styles and return structured fields without a new supervised model for every question.

Research activity has moved faster than routine clinical adoption. A 2025 scoping review identified 24 studies of language-model-based oncology information extraction, but also found that external validation and real-world workflow integration remained limited [1]. Clinical deployment has a higher bar than a demonstration on a benchmark. A useful system must preserve uncertainty, distinguish current care from historical events, avoid unsupported facts, and produce results that a clinician can trace back to the note.

Longitudinal oncology notes expose weaknesses that are easy to miss in simpler extraction tasks. A medication list can contain anticancer therapy, supportive treatment, chronic home medications, discontinued drugs, and therapies that are only being discussed. Regional lymph nodes must not be labeled as distant metastases. A suspicious lesion awaiting biopsy must not become confirmed stage IV disease. Tumor growth before treatment is not evidence of failure of a regimen that has just started. A model can mention medically relevant information and still place it in the wrong field or timepoint.

Several strategies have been used to improve clinical extraction. Large proprietary models can perform zero-shot extraction but still omit note-specific details and hallucinate [2]. Local open-weight models reduce dependence on external services, but performance varies with model size and prompt design [3,8]. Fine-tuned hybrid systems can achieve strong performance and external validation, although they require large labeled datasets and model training [5]. Hierarchical prompting, validation layers, and retry mechanisms also improve extraction without conventional fine-tuning [6,7]. These studies establish prior work for the individual techniques used in modern extraction systems.

Human involvement also differs across studies. Clinicians or medical experts often create gold-standard annotations, resolve disagreements, or guide terminology selection. That work establishes whether a model matches a reference label. It does not answer whether a practicing oncologist considers one complete system output more faithful, complete, and clinically useful than another when both contain partly correct information. In our audit of the supplemental methods table for the 24-study scoping review, most papers reported automatic performance metrics against labels. Only two entries explicitly described five-point Likert ratings of generated outputs, and both involved radiology reports [1]. We did not identify a field-level, system-label-masked comparison of complete extraction systems by practicing oncologists in that review.

Our approach treats recurring model errors as engineering targets. We use the term inference harness for the software layer that controls how a frozen model is prompted, checked, corrected, and linked to evidence. The model itself is unchanged. When review identifies a repeated failure, such as importing a stopped drug into current therapy or turning suspected metastasis into confirmed disease, the system receives a narrow correction through prompt instructions, verification logic, or a deterministic clinical rule. Each correction can be logged and regression-tested.

The harness is more than a long prompt or retrieval step. Different fields follow different extraction routes. Selected outputs pass into later tasks only when they provide relevant clinical context. Verification stages run after generation, and deterministic hooks enforce narrow oncology constraints across related fields. The system records these interventions and links final values to note evidence. Among the closest studies we reviewed, we did not identify an evaluation of this full combination on real longitudinal oncology notes with a same-model baseline and direct oncologist comparison.

We tested whether this approach improves extraction under a controlled comparison. The harness and baseline use the same Qwen2.5-32B model and the same target schema. The main difference is the inference process surrounding the model. We focus on seven clinical questions that require more than surface entity recognition: active anticancer therapy, stage, distant metastasis, regional or overall metastatic involvement, treatment response, breast cancer type and receptor status, and completed molecular or genetic results.

We hypothesized that the harness would outperform the single-prompt baseline overall and that the largest gains would occur in fields directly addressed by temporal checks, clinical classification rules, and cross-field consistency checks. We also expected many ties because the same base model should answer straightforward questions similarly in both conditions.

## 2. Methods

### 2.1 Study design

The study comprised two development stages followed by an independent clinical evaluation. During breast-cancer development, a physician coauthor and the model-development investigators iteratively reviewed extraction outputs, identified recurring error classes, and revised the inference harness. The framework was then adapted to pancreatic cancer without clinician review of pancreatic-cancer outputs. In this second stage, model-assisted review identified candidate errors and proposed revisions, which the investigators assessed and regression-tested. The final harness was compared with a single-prompt baseline using the same frozen model and field schema on 40 expert-annotated CORAL notes. Three oncologists completed system-label-masked A/B evaluations.

The breast-cancer stage examined whether clinically informed error analysis could be converted into reusable system components. The pancreatic-cancer stage examined adaptation to a related but distinct oncology domain without repeated clinician involvement. The final comparison isolated the contribution of the inference harness while holding the base model and target fields constant.

### 2.2 Dataset

We used CORAL, a controlled-access PhysioNet dataset of deidentified medical oncology progress notes from the University of California, San Francisco [2]. Its expert-annotated benchmark contains 20 breast-cancer and 20 pancreatic-cancer notes. The release also includes 100 additional notes for each cancer type with GPT-4-generated labels rather than expert human annotations. Documented development iterations covered 56 of the additional breast-cancer notes and all 100 additional pancreatic-cancer notes. The 40 expert-annotated notes were initially reserved for comparison.

The records are real clinical notes rather than web-derived questions, synthetic cases, or model-generated narratives. They retain repeated histories, copied-forward content, uncertain findings, and conditional plans that make longitudinal oncology extraction difficult. The value of this dataset is clinical realism and expert annotation rather than scale. Access requires the PhysioNet credentialing and data-use process described by the dataset authors.

Development notes were used to identify recurrent error patterns and refine the harness. The annotated benchmark notes were used for matched technical and clinician comparisons. Although these notes were initially reserved for evaluation, a later development-stage audit informed further harness revisions before clinician review. The comparison should therefore be interpreted as a pilot benchmark rather than an untouched external validation. No model-weight training or fine-tuning was performed.

### 2.3 Base model and baseline

Both conditions used Qwen2.5-32B-Instruct-AWQ [9], served locally through vLLM. The baseline made one model call per note and returned the complete target schema. It did not use task decomposition, verification gates, retries, dictionaries, or deterministic post-processing. The finalized baseline used in the clinician review package and the harness used the same field definitions and output contract.

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

<div data-rough-figure="1"></div>

***Figure 1.*** *Development, adaptation, and evaluation pathway. During breast-cancer development, a physician coauthor and model-development investigators converted recurring extraction errors into prompt, verification, and deterministic-rule changes. The harness was then adapted to pancreatic cancer using model-assisted review without clinician review of pancreatic-cancer outputs. Final outputs were compared with a same-model single-prompt baseline in a system-label-masked evaluation by three oncologists.*

### 2.5 Clinician-informed breast-cancer development

Breast-cancer development used approximately 15 documented iterations across 56 additional notes. A physician coauthor familiar with oncology reviewed selected outputs with the model-development investigators. This physician was not an oncology specialist and did not participate in the final oncologist evaluation. The investigators compared flagged outputs with the complete source notes, grouped recurring errors, and translated them into general changes to prompts, verification logic, or deterministic rules. This process did not produce a conventional supervised training set, and the model weights were not updated.

Candidate changes were retained only after testing on affected examples and previously correct controls. The objective was to encode recurring clinical distinctions rather than case-specific corrections. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general regional-node rule. An error in which a planned drug was listed as active therapy motivated a temporal medication rule.

### 2.6 Model-in-the-loop pancreatic-cancer refinement

The breast-cancer harness was then adapted to pancreatic cancer through approximately 18 documented development rounds covering all 100 additional pancreatic notes. No clinician reviewed pancreatic-cancer outputs during this stage. The pipeline generated structured fields, and a rubric-informed Qwen reviewer compared them with the complete source note. A separate general-purpose LLM summarized accumulated review findings and proposed candidate prompt, rule, or workflow changes. The study investigators assessed these proposals, implemented selected changes, and retained them only after regression testing.

The reviewer prompt included the field definitions, severity criteria, and clinical distinctions established during breast-cancer development. This allowed earlier error categories to guide review in the new domain while still exposing pancreatic-specific problems such as regimen names and dose representation. Models contributed to error detection and candidate revision, but the investigators controlled implementation. We therefore describe this stage as model-assisted, investigator-supervised refinement rather than autonomous self-improvement.

Throughout both development stages, the pipeline logged the original model output, each verification action, and deterministic corrections. This record allowed the team to trace a final value back through the harness and to retain or reject proposed changes based on regression results.

### 2.7 Clinician-prioritized clinical fields

Before the matched comparison, the physician coauthor designated seven fields as clinically important for understanding disease status and current management:

1. Which anticancer drugs is the patient actively receiving?
2. What is the current cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer responding to the current treatment?
6. What is the breast cancer type and ER/PR/HER2 status?
7. What completed molecular or genetic results are documented?

The oncologist evaluation instrument also included genetic testing plans, supportive medications, procedure plans, imaging plans, laboratory plans, medication plans, and recent treatment changes. Laboratory summary and general clinical findings were optional and excluded from the primary clinician analysis.

### 2.8 Development-stage LLM-assisted review

LLM-assisted review was used as a development instrument, particularly during adaptation to pancreatic cancer. For each candidate output, the reviewer model compared the extracted fields with the source note using the prespecified field definitions and severity criteria. It flagged possible omissions, unsupported claims, semantic mismatches, and temporal errors. These findings informed candidate revisions that the investigators reviewed and regression-tested.

A matched audit of 260 applicable note-field comparisons was also used for technical error analysis. LLM judgments were not used as final clinical outcome labels and did not replace the independent oncologist evaluation.

### 2.9 Oncologist evaluation

The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were not told which output came from the harness, whether the systems used the same underlying model, or which condition was expected to perform better. The assignment was fixed across evaluations, with the harness displayed as A and the baseline as B. Source attribution appeared with the harness output because it is part of the complete harness output package. The study therefore evaluated the complete presented systems and did not isolate the effect of attribution.

The physician coauthor who participated in breast-cancer development was not one of the final evaluators. Three oncologists independently evaluated the breast-cancer outputs. Two of them also evaluated the pancreatic-cancer outputs; the third did not evaluate pancreatic cancer.

All three oncologists completed the same 280 required breast-cancer comparisons. We calculated pairwise exact agreement and Cohen's kappa for each pair. We also summarized all five completed clinician-by-cancer evaluations separately. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The primary inferential analysis excluded ties and modeled whether a directional judgment favored the harness. We used a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator was included as a fixed effect, and the overall model also included cancer type. This specification accounts for repeated judgments across fields and evaluators within a note. We report adjusted odds ratios, 95% confidence intervals, and two-sided p values. Exact sign tests on note-level harness-minus-baseline margins were used as a sensitivity analysis.

## 3. Results

### 3.1 Development path and cross-cancer adaptation

The development record contains approximately 15 breast-cancer iterations across 56 notes and approximately 18 pancreatic-cancer iterations across 100 notes. Breast-cancer revisions were informed by review from the physician coauthor and model-development investigators. Pancreatic-cancer revisions were made without clinician review of the pancreatic outputs, using the adapted rubric, model-based error review, external LLM-assisted synthesis, and investigator-controlled implementation described above.

This sequence produced two types of reuse. Some components were retained unchanged, including the five verification stages, temporal distinctions, source attribution, and rules that separate active treatment from plans or supportive medication. Other components required cancer-specific routing, especially disease terminology, regimen interpretation, and post-processing conditions. The resulting system therefore reused the error-handling framework without assuming that breast and pancreatic cancer were clinically interchangeable.

The pancreatic-cancer clinician result is the most relevant evidence for this adaptation. Across two independent evaluations, the oncologists preferred the harness in 161 field comparisons and the baseline in 23, with 335 ties. One oncologist recorded a positive within-note margin in 19 cases and a negative margin in one. The other recorded 18 positive margins and two ties. When their ratings were pooled within notes, all 20 pancreatic cases had a positive harness-minus-baseline margin. Because no physician reviewed pancreatic outputs during development, this pattern is consistent with reuse of previously codified evaluation criteria and workflow components in a second cancer domain. It does not identify whether the gain came from shared rules, pancreatic-specific revisions, the model-based reviewer, or their combination.

### 3.2 Current oncologist evaluation

The available clinical evidence comprises five completed clinician-by-cancer evaluations: breast cancer from all three oncologists and pancreatic cancer from two. Across 1,359 required-field judgments, the harness was preferred 443 times, the baseline 77 times, and 839 comparisons were ties. These correspond to 32.6%, 5.7%, and 61.7% of all judgments. Among the 520 directional judgments, the harness received 85.2%.

| Completed evaluation | Required judgments | Harness | Baseline | Tie | Harness share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 01, breast cancer | 280 | 84 | 22 | 174 | 79.2% |
| Oncologist 01, pancreatic cancer | 260 | 75 | 14 | 171 | 84.3% |
| Oncologist 02, breast cancer | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 02, pancreatic cancer | 259 | 86 | 9 | 164 | 90.5% |
| Oncologist 03, breast cancer | 280 | 119 | 24 | 137 | 83.2% |
| **All completed evaluations** | **1,359** | **443** | **77** | **839** | **85.2%** |

<div data-rough-figure="2"></div>

***Figure 2.*** *Distribution of system-label-masked clinician preferences across five completed clinician-by-cancer evaluations. Most judgments were ties, while directional judgments favored the inference harness in every evaluation.*

All three clinicians independently favored the harness on the breast-cancer set. Their pooled breast result was 282 harness preferences, 54 baseline preferences, and 504 ties. The first oncologist's per-note result was 18 harness wins, one baseline win, and one tie. The second and third oncologists each recorded a harness win in all 20 notes. When the three breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin.

The two pancreatic-cancer evaluations produced 161 harness preferences, 23 baseline preferences, and 335 ties. The harness had a positive pooled margin in all 20 pancreatic notes.

In the adjusted directional analysis, the odds of a harness preference were 5.70 times the odds of a baseline preference (95% CI 4.24 to 7.66; p<0.001). Cancer-specific estimates were 5.18 for breast cancer (95% CI 3.61 to 7.46; p<0.001) and 6.94 for pancreatic cancer (95% CI 4.19 to 11.48; p<0.001). As a note-level sensitivity analysis, the pooled harness-minus-baseline margin was positive in all 40 notes (two-sided exact sign test, p<0.001). These estimates describe preference among the observed ratings; three oncologists remain insufficient for a precise estimate of variation across the wider oncologist population.

### 3.3 Inter-rater agreement and core fields

All three oncologists rated the same 280 required breast-cancer comparisons. Pairwise exact agreement was 82.9% between oncologists 01 and 02, 80.0% between oncologists 01 and 03, and 73.6% between oncologists 02 and 03. The corresponding Cohen's kappa values were 0.646, 0.644, and 0.511. All three oncologists gave the same verdict on 192 comparisons (68.6%). A simple majority favored the harness in 95 comparisons, the baseline in 15, and a tie in 168. Two comparisons had one vote in each category and therefore no majority.

<div data-rough-figure="3"></div>

***Figure 3.*** *Pairwise agreement among three oncologists on 280 shared breast-cancer field comparisons. Exact agreement ranged from 73.6% to 82.9%, with Cohen's kappa from 0.511 to 0.646.*

Across all completed evaluations, the seven clinician-prioritized core categories contributed 660 applicable judgments. The harness received 276 preferences, the baseline 34, and 350 were ties. The harness therefore accounted for 89.0% of the 310 directional core judgments, and every core category had a positive aggregate margin.

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

<div data-rough-figure="4"></div>

***Figure 4.*** *Clinician preference by clinician-prioritized core clinical category. The largest harness advantages occurred in active-treatment identification and metastatic-involvement classification. Breast type and receptor status showed the smallest margin.*

Active anticancer medication and regional or overall metastatic involvement produced the largest and most consistent core-field margins. Medication planning, outside the seven core categories, totaled 59 harness preferences, no baseline preferences, and 41 ties. Procedure planning was much closer at 14 harness preferences, 12 baseline preferences, and 74 ties. Breast type and receptor status remained the weakest core category, with a 6-rating net advantage.

Note-level summaries supported the same direction without treating every field as an independent observation. After pooling clinicians within each cancer type, the harness-minus-baseline margin was positive in all 20 breast notes and all 20 pancreatic notes. Each of the five evaluator-by-cancer analyses also favored the harness at the aggregate level. These summaries serve as sensitivity analyses alongside the adjusted model.

<div data-rough-figure="5"></div>

***Figure 5.*** *Distribution of normalized clinician preference margins across individual notes. After pooling the available clinicians within each cancer type, the harness had a positive margin in every breast-cancer and pancreatic-cancer note.*

<div data-rough-figure="6"></div>

***Figure 6.*** *Adjusted odds of clinician preference for the inference harness among non-tie judgments. Estimates come from logistic generalized estimating equations with clustering by note; points show odds ratios and lines show 95% confidence intervals.*

### 3.4 Development-stage LLM audit

A source-grounded LLM reviewer compared the harness and baseline across 260 applicable core note-field pairs during development. It preferred the harness 66 times, the baseline 28 times, and judged 166 comparisons as ties. The harness had a positive margin in six of seven categories. Stage was the only category with a negative margin in this development-stage audit. These model-generated judgments were used for error analysis and are not clinical outcome labels.

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

<div data-rough-figure="S1"></div>

***Supplementary Figure S1.*** *Category-level net preference rates in the development-stage LLM audit and oncologist evaluation. The two series are shown only as a descriptive comparison because they used different reviewers and pipeline versions.*

### 3.5 Interpretation of free-text comments and ties

Two clinician exports contained 12 written comments. These comments clarify an important limitation of the A/B outcome: a tie can mean either that both outputs are acceptable or that both are incomplete or incorrect. In one breast imaging-plan comparison, one oncologist preferred the harness because the baseline summarized completed findings instead of the plan, whereas another oncologist judged both outputs inadequate and selected a tie.

The comments also identified unsupported receptor status, omitted regional nodal disease, uncertainty about response after a newly started regimen, and comparisons in which neither output was satisfactory. They are used here to interpret the rating scale and to identify error-analysis examples, not as a separate quantitative endpoint.

## 4. Discussion

### 4.1 Main interpretation

The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline. In the adjusted directional analysis, the odds ratio for a harness preference was 5.70 (95% CI 4.24 to 7.66; p<0.001).

The main result concerns the development process around the model. A clinician helped identify recurrent errors in breast cancer, those errors were converted into explicit parts of the inference harness, and the resulting process was adapted to pancreatic cancer without physician review during pancreatic development. The final clinician ratings favored the harness in both domains. This sequence connects the development method to the evaluation more directly than a static comparison of two prompts would.

Most field comparisons were ties, which is expected because both systems used the same base model. When an oncologist found a meaningful difference, the harness was preferred nearly six times as often as the baseline. The gains were concentrated in fields that need temporal interpretation or clinical classification. This is consistent with a system that leaves straightforward answers alone and intervenes when a known failure pattern appears.

The two main methodological limits are the absence of a component ablation and the ambiguity of ties, which can represent either two acceptable outputs or two inadequate outputs. The results therefore support preference for the complete presented harness, not absolute correctness or a causal claim for any single prompt, gate, rule, or attribution component.

### 4.2 Clinician-informed extraction as a reusable harness

The physician coauthor's role during breast-cancer development differed from conventional dataset labeling. Rather than producing a large supervised training set, the physician and model-development investigators reviewed concrete failures and defined clinically meaningful distinctions. Those distinctions were encoded as prompts, verification checks, and deterministic rules. The three independent oncologist evaluations suggest that the resulting behavior was not limited to the development physician's preferences.

Structured extraction is useful because it converts a long, internally repetitive note into a compact set of fields that can be checked, searched, and reused downstream. It is particularly valuable when the task is tedious for a clinician but can be bounded precisely, such as reconciling current anticancer therapy across a medication list and treatment history.

We use the term inference harness because the contribution sits around the model rather than in its weights. The routing, schemas, verification logic, deterministic rules, logging, and attribution layer could in principle be attached to another instruction-following model after interface and prompt calibration. This study tested only Qwen2.5-32B, so cross-model portability remains a design property to evaluate rather than an empirical result.

### 4.3 Cross-cancer adaptation and the role of LLM-as-a-judge

The pancreatic-cancer stage tested whether the accumulated workflow could be adapted beyond the domain in which the physician coauthor gave direct feedback. Pancreatic notes differ in disease course, treatment regimens, staging language, and surgical context. We retained shared clinical distinctions, added cancer-specific routing where needed, and did not ask a clinician to review pancreatic outputs during development.

Two model roles supported this adaptation. First, a general-purpose external LLM received the accumulated breast-cancer error history and proposed an initial set of pancreatic-cancer adaptations. Second, a rubric-guided LLM-as-a-judge compared subsequent outputs with their source notes and localized likely omissions, unsupported claims, semantic mismatches, and temporal errors. This reviewer made the iterative process observable by showing where the adapted harness was succeeding or failing.

The judge did not define the clinical endpoint and did not approve its own changes. Investigators selected candidate revisions and regression-tested them, while the final evidence came from independent oncologists. The pancreatic ratings, 161 harness preferences versus 23 baseline preferences with 335 ties, support the usefulness of this model-assisted development loop without establishing autonomous self-improvement.

### 4.4 Model difficulty and human-model complementarity

The field-level results distinguish direct extraction from tasks that require temporal or clinical interpretation. Explicit findings often produced ties because both systems could locate them. The larger differences appeared when the same fact had to be classified in context.

Active anticancer medication showed the largest core-field advantage, with 83 harness preferences, two baseline preferences, and 15 ties. Medication planning also strongly favored the harness, 59 to zero with 41 ties. These tasks are laborious for a person reviewing a long treatment history, but they are well suited to structured assistance when current, historical, supportive, and planned therapies are kept separate.

Stage, metastatic involvement, and treatment response illustrate a different form of complementarity. A clinician may resolve an explicitly documented stage quickly, while a model can still confuse regional nodes with distant disease or treat a suspected lesion as confirmed. Response assessment can be difficult for both because it requires aligning findings with the current regimen. In these fields, the system is better used to organize evidence and flag a proposed answer for rapid verification than to replace clinical judgment.

Breast type and receptor status had the smallest core-field margin, 18 to 12 with 30 ties. This is consistent with a field that is often stated explicitly, leaving less room for the harness to improve on the base model. The useful division of labor is therefore field-dependent: automate repetitive synthesis, surface evidence for judgment-sensitive fields, and allow clinicians to verify compact outputs rather than reread every note from scratch.

<div data-rough-figure="7"></div>

***Figure 7.*** *Conceptual map of human-model complementarity across oncology extraction tasks. Placement is qualitative rather than a measured difficulty score. The map distinguishes repetitive synthesis tasks suited to automation from judgment-sensitive fields that benefit from rapid clinician verification.*

### 4.5 Mechanistic evidence and the need for ablation

The observed field pattern is consistent with the intended function of the harness. The active-medication and medication-plan results align with dedicated prompts, temporal filtering, and drug-context rules. The stage and metastasis results align with cross-field checks and deterministic distinctions between regional and distant disease and between suspected and confirmed findings. These associations support the proposed mechanism but do not isolate the contribution of each component.

Component comparisons are present in related work but are not universal. Wiest et al. compared plain zero-shot, one-shot, definition-enhanced, and grammar-constrained prompting [3]. mCODEGPT directly compared single-step prompting with two hierarchical strategies [7]. Grothey et al. compared five prompting strategies and quantized model configurations [8]. Dao et al. quantified which initial errors were corrected by its validation and retry loop, but did not report a full factorial removal of every pipeline component [6]. Tariq et al. compared the complete hybrid system with zero-shot, structured-code, and rule-based baselines rather than removing each hybrid phase in turn [5].

Our study likewise lacks a complete component ablation. A feasible technical ablation would compare the single-prompt baseline, decomposed prompts alone, prompts plus verification gates, and the complete harness. Repeating the full oncologist evaluation for every variant would impose a disproportionate specialist burden. An LLM-reviewed ablation could help localize mechanisms, but it should be labeled as technical evidence and should not replace the independent oncologist comparison. A smaller clinician review restricted to outputs changed by the ablation would provide a stronger follow-up.

### 4.6 Relation to prior clinical extraction studies

CORAL established the real-note benchmark used here and showed that zero-shot GPT-4 outperformed GPT-3.5-turbo and FLAN-UL2, while an independent oncologist still identified omissions and hallucinations [2]. Our study asks a different question on the same benchmark: whether a structured workflow improves a frozen model relative to that same model under a single prompt.

Several studies support individual parts of this approach. Wiest et al. showed that prompt design and constrained output can improve local-model extraction of binary clinical features [3]. mCODEGPT found that hierarchical prompting outperformed a single-step strategy on synthetic oncology notes [7]. Grothey et al. demonstrated substantial variation across model, prompt, and quantization configurations in prostate pathology extraction [8]. Dao et al. is architecturally close to our work because it combines engineered context, validation, and retries, although its target is numerical extraction from right-heart-catheterization reports [6]. Tariq et al. achieved stronger scale and external validation with a UMLS-plus-fine-tuned-LLM system for breast treatment timelines [5], while Bhattarai et al. compared several model families and a rule-based approach on longitudinal lung-cancer phenotypes [4].

The distinguishing feature of our evaluation is not the invention of prompts, retries, or rules in isolation. It is the combination of field-specific routing, selective context transfer, verification gates, deterministic oncology rules, logging, and source attribution, evaluated against a same-model baseline through direct field-level comparison by oncologists. The breast-to-pancreatic development sequence also tests whether clinically informed failure rules can reduce repeated demand on specialist time in a second cancer domain.

The study should therefore be positioned as a focused, clinically reviewed evaluation of an integrated inference harness, not as the first oncology extraction system or the largest validation study. The detailed study-by-study comparison is retained in the Chinese review appendix rather than in the main manuscript.

### 4.7 From structured extraction to patient communication

The exploratory patient-letter review suggested limited downstream strengths. Relative to ChatGPT, harness-based letters had higher mean ratings across accuracy, completeness, comprehensibility, and usefulness, and received fewer hallucination flags. They did not clearly outperform the same-model Qwen baseline. Stronger extraction therefore does not automatically produce a better complete letter. Letter quality also depends on content selection, organization, wording, explanation of uncertainty, and decisions about which clinical details a patient needs.

This finding supports the revised project order. Structured extraction is the measurable safety layer, and patient communication is a downstream task built on that layer. The extraction study can identify whether stage, treatment, response, and plans are represented faithfully before a generator turns them into prose. Future letter work should test whether specific extraction gains survive that second transformation, ideally with patient readers as well as clinicians.

The letter experiment is best treated as motivation for future work rather than a separate efficacy result. It shows where structured extraction may help patient-facing generation and, equally importantly, where a second generation step can lose the advantages established at the extraction stage.

### 4.8 Clinical and technical implications

The findings suggest that some LLM failures in oncology extraction are repeatable enough to address outside the model weights. This is useful when labeled training data are scarce or when a clinical team needs to change a field definition without retraining. A rule that preserves biopsy-pending disease as suspected can be inspected and tested. A prompt-only system offers less control because a wording change may affect unrelated fields.

The term inference harness is appropriate for this system because it includes more than a prompt sequence. It manages task routing, dependencies, verification, deterministic corrections, logging, and source attribution around a frozen model. The term clinical workflow could imply integration into routine care, which this pilot did not test. We therefore use inference harness for the technical contribution and evaluation workflow for the study procedure.

The system is also compatible with local deployment. Local operation does not by itself establish privacy compliance or clinical safety, but it allows an institution to retain control of note processing and system updates. The present work evaluates extraction quality, not readiness for autonomous clinical use.

### 4.9 Study boundaries and next steps

This pilot was intended to establish whether the observed effect is large and clinically coherent enough to justify a larger study. Three oncologists independently reproduced the breast-cancer direction, and two reproduced the pancreatic-cancer direction. The adjusted clustered analysis confirmed a strong preference among these ratings.

The first main limitation is that the study evaluates the complete harness and does not separate the effects of task decomposition, verification gates, deterministic rules, or source attribution. The second is that the preference scale combines two different meanings of a tie: both outputs may be acceptable, or both may be inadequate. These limitations restrict mechanism and absolute-correctness claims even though the direction of relative preference is consistent.

The broader scope also remains limited. Only three oncologists participated, CORAL represents one institution and two cancer domains, and another model family has not been tested. Technical review of the benchmark informed some later revisions, so the study is a benchmark-informed pilot rather than an untouched external validation. This does not change the narrower pancreatic-cancer observation: no clinician reviewed pancreatic outputs during development.

The next study should adjudicate a sample of ties, evaluate the staged ablation described above, add oncologists, and test a second model and an external health-system dataset. These additions would clarify why the harness helps and support stronger claims about evaluator and institutional generalizability.

## 5. Conclusion

The current pilot supports a development strategy in which a clinician identifies clinically important failure patterns, the team converts those patterns into an explicit inference harness, and the harness is adapted to another cancer domain through model-assisted, investigator-supervised refinement. All three oncologists favored the harness on breast cancer. Two also favored it on pancreatic cancer, where no physician had participated in development. The largest gains involved active therapy, medication planning, and metastatic status. The model weights remained frozen throughout. Component ablation, tie adjudication, more oncologists, and external validation are needed before this pilot pattern becomes a broad confirmatory claim.

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

# 从临床信息指导的错误分析到跨癌种适配：用于肿瘤科病历 extraction 的 inference harness

**供临床合作者审阅的试点报告草稿**

版本 0.12，2026 年 9 月

作者：[TODO]

单位：[TODO]

目标会议及稿件形式：[TODO]

本稿现已纳入三位肿瘤科医生完成的评审。三位医生均评估了乳腺癌病例，其中两位还评估了胰腺癌病例。全文按照实际开发过程展开：首先由医生作者与模型开发作者共同完成乳腺癌任务的错误分析，再将所得经验编码为 inference harness；随后在没有临床医生审阅胰腺癌输出的情况下，通过 model-assisted development loop 改进 PDAC 任务；最后由肿瘤科医生进行隐藏系统标签的 A/B 评估。调整后的 GEE 分析已经完成，Figure 1 至 Figure 7 和 Supplementary Figure S1 已嵌入 HTML。方括号中的内容仅保留在中文版，作为后续协作修改提示。

供临床审阅的简要说明：两个系统使用相同的语言模型。single-prompt baseline 要求模型通过一次调用提取全部信息。inference harness 则把任务拆分为较小的临床问题，检查模型答案，针对反复出现的错误应用范围明确的肿瘤学规则，并把每项结果与病历中的支持性原文关联起来。

> **术语约定：** 中文审阅稿保留 inference harness、workflow、pipeline、baseline、verification gate、hook 和 LLM-as-a-judge 等英文术语，避免翻译后与代码和实验记录中的名称对不上。

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

评估 inference harness 能否在不进行微调的情况下，提高冻结参数、本地部署的开放权重语言模型在肿瘤科医生评审下的结构化提取表现。

### 方法

我们围绕 Qwen2.5-32B-Instruct-AWQ 分两个开发阶段构建了一个由失败模式驱动的 inference harness。在乳腺癌开发阶段，一位熟悉肿瘤学问题的医生作者与模型开发作者反复审阅输出，并指出具有临床意义的错误。团队将反复出现的错误转化为 field-specific prompt、verification gate 和 deterministic oncology rule。随后，我们将 harness 适配到胰腺癌。在胰腺癌开发期间，没有临床医生审阅模型输出。依据 rubric 配置的 Qwen reviewer 识别候选错误，外部通用 LLM 汇总审查记录并提出 prompt 或 rule 修改方案，再由研究人员选择、实施并完成 regression test。整个过程中模型权重始终冻结。我们在 40 份 CORAL benchmark 病历上比较 harness 与使用相同模型和 field contract 的最终 single-prompt baseline，其中包括 20 例乳腺癌和 20 例胰腺癌。三位肿瘤科医生完成了隐藏系统标签的 A/B 比较。三位医生均评估乳腺癌，其中两位还评估 PDAC。

### 结果

在已完成的匹配技术审查中，inference harness 在 66 项核心比较中更优，baseline 在 28 项中更优，另有 166 项为平局。五组已完成的评审者与癌种组合共提供 1,359 项必评字段判断，其中 443 项偏好 inference harness，77 项偏好 baseline，839 项为平局。平局占全部判断的 61.7%；在其余 520 项非平局判断中，85.2% 偏好 inference harness。调整后的方向性分析得到 OR 5.70，95% CI 为 4.24 至 7.66，p<0.001。三次乳腺癌评估合计包括 282 项 harness 偏好、54 项 baseline 偏好和 504 项平局。两两完全一致率为 73.6% 至 82.9%，Cohen's kappa 为 0.511 至 0.646。在开发阶段没有医生参与的胰腺癌任务中，两位医生合计给予 inference harness 161 项偏好，baseline 23 项偏好，另有 335 项平局。

> **协作说明：** Abstract 会在全文定稿后最后压缩。当前统计数字已经更新，不再保留未完成分析的占位内容。

### 结论

三次乳腺癌评估支持将临床医生发现的反复性模型错误转化为明确的 inference harness。两次胰腺癌评估提示，该 harness 可以在没有 target-domain clinician 参与开发审阅的情况下适配到第二个癌种。这一过程属于 AI 辅助、人工监督的改进，并非自主修改。调整后的 clustered analysis 已显示现有评分中存在显著偏好，但更广泛的结论仍需要更多肿瘤科医生和 external validation。

## 1. 引言

肿瘤临床工作中，大量有用信息仍记录在自由文本中。一份随访病历可能同时包含诊断、病理、受体状态、治疗史、毒性反应、疗效和后续计划，但这些事实分散在不同章节和时间点。人工审阅耗时较长，传统信息提取系统则需要大量标注和针对具体任务的开发。大语言模型的吸引力很直接：同一个模型可以读取多种病历写法，并针对不同问题返回结构化字段，无须为每个问题重新训练监督模型。

相关研究的发展速度快于临床常规应用。2025 年的一项范围综述纳入了 24 项使用语言模型提取肿瘤学信息的研究，但也发现外部验证和真实 workflow 整合仍然有限 [1]。临床部署的要求高于基准数据上的概念验证。一个实用系统必须保留不确定性，区分当前诊疗与历史事件，避免无依据的事实，并让临床医生能够从病历原文追溯每项结果。

纵向肿瘤科病历会暴露一些在简单提取任务中不易发现的问题。药物清单可能同时包含抗癌治疗、支持治疗、长期居家用药、已停用药物，以及仅处于讨论阶段的治疗。区域淋巴结不能被标为远处转移。等待活检的可疑病灶不能被写成已确诊的 IV 期疾病。治疗前的肿瘤增长也不能用来判断刚刚开始的方案无效。模型即使提到了医学相关信息，仍可能把它放入错误的字段或时间点。

已有研究采用了多种方法改进临床信息提取。大型闭源模型可以进行零样本提取，但仍会遗漏病历特有细节并产生幻觉 [2]。本地开放权重模型减少了对外部服务的依赖，但性能会随模型规模和提示设计而变化 [3,8]。经过微调的混合系统可以获得较强性能和外部验证结果，但需要大规模标注数据和模型训练 [5]。分层提示、验证层和重试机制也可以在不采用传统微调的情况下提高提取质量 [6,7]。这些研究分别为现代信息提取系统采用的各项技术提供了依据。

各项研究中人工参与的方式也不相同。临床医生或医学专家通常负责建立金标准标注、解决分歧或指导术语选择。这类工作可以判断模型是否匹配参考标签，但不能回答另一个问题：当两个完整系统的输出都包含部分正确信息时，执业肿瘤科医生是否认为其中一个系统更忠实、更完整，也更有临床用途。在我们对上述 24 项研究范围综述的补充方法表进行审查时，多数论文报告的是相对于标签的自动性能指标。只有两项研究明确使用五点 Likert 量表评估生成结果，而且均针对放射学报告 [1]。在该综述中，我们没有发现由执业肿瘤科医生在隐藏系统标签的条件下，对完整信息提取系统进行字段级直接比较的研究。

我们把反复出现的模型错误视为具体的工程目标。本文所称的 inference harness，是指控制冻结模型如何接受提示、检查答案、修正结果并关联证据的软件层。模型本身不发生变化。当审查发现重复性错误，例如把已停用药物列入当前治疗，或把疑似转移写成确诊疾病时，系统通过提示指令、验证逻辑或确定性临床规则加入范围明确的修正。每项修正都可以记录并接受 regression test。

该 inference harness 包含 field routing、selective context transfer、生成后验证和确定性临床约束，范围超过单一长提示或检索步骤。不同字段采用不同的提取路径。只有在确实提供相关临床背景时，部分输出才会传递给后续任务。系统记录这些干预，并将最终字段值与病历证据关联起来。在我们审阅的最接近本研究的工作中，尚未发现有研究在真实纵向肿瘤科病历上评估这一完整组合，同时采用同模型 baseline 和肿瘤科医生直接比较。

我们在受控条件下检验了这种方法能否提高提取质量。inference harness 和 baseline 使用相同的 Qwen2.5-32B 模型及目标 schema，主要差异是模型外围的推理过程。研究聚焦七个不能仅靠表层实体识别解决的临床问题：当前抗癌治疗、癌症分期、远处转移、区域或总体转移受累、治疗反应、乳腺癌类型与受体状态，以及已完成的分子或遗传检测结果。

我们的假设是，inference harness 整体上会优于 single-prompt baseline，且提升主要出现在由时态检查、临床分类规则和跨字段一致性检查直接处理的字段中。由于两种条件使用相同的基础模型，我们也预期简单问题会出现较多平局。

## 2. 方法

### 2.1 研究设计

本研究包括两个开发阶段和一个独立的临床评估阶段。乳腺癌开发期间，一位医生作者与负责模型开发的作者反复审阅提取结果，归纳重复出现的错误，并据此修改 inference harness。随后，研究团队在没有临床医生审阅胰腺癌输出的情况下，将该 inference harness 适配到胰腺癌。第二阶段通过模型辅助审查发现候选问题并提出修改方案，再由研究人员判断、实施和进行 regression test。最后，我们在 40 份经专家标注的 CORAL 病历上，将最终 inference harness 与使用同一冻结模型和相同字段定义的 single-prompt baseline 进行比较，并由三位肿瘤科医生完成隐藏系统标签的 A/B 评估。

这三个环节分别回答不同问题：乳腺癌阶段检验能否把有临床依据的错误分析转化为可复用的系统组件；胰腺癌阶段检验这些组件能否适配到相关但不同的肿瘤领域，而不依赖临床医生持续参与迭代；最终比较则在固定基础模型和目标字段的条件下，评估 inference harness 本身带来的作用。

### 2.2 数据集

我们使用 CORAL 数据集。该数据集通过 PhysioNet 受控开放，包含来自加州大学旧金山分校、经过去标识化处理的肿瘤内科随访病历 [2]。其专家标注基准包括 20 份乳腺癌病历和 20 份胰腺癌病历。公开版本还分别提供每个癌种 100 份附加病历，这些附加病历带有 GPT-4 自动生成的标签，但没有人工专家标注。有记录的开发迭代使用了其中 56 份乳腺癌附加病历和全部 100 份胰腺癌附加病历。40 份专家标注病历最初预留用于系统比较。

这些记录是真实临床病历，而不是网络问答、合成病例或模型生成文本。病历保留了重复病史、复制到后续记录的内容、不确定检查结果和条件性计划，这些特点正是纵向肿瘤信息提取的难点。该数据集对本研究的主要价值是临床真实性和专家标注，而不是规模。数据访问需要完成 PhysioNet 规定的账号认证和数据使用流程。

开发病历用于识别反复出现的错误模式并改进 harness，标注基准病历用于匹配技术比较和临床医生比较。虽然这些基准病历最初预留用于评估，但后续开发阶段的技术审查仍影响了医生评估前的 harness 修改。因此，本研究应被理解为试点性基准评估，而不是完全未接触数据的外部验证。研究未进行模型权重训练或微调。

### 2.3 基础模型与 baseline

两个实验条件均使用通过 vLLM 在本地部署的 Qwen2.5-32B-Instruct-AWQ [9]。baseline 系统对每份病历只调用模型一次，并返回完整的目标 schema。它不使用任务拆分、verification gate、重试、词典或确定性后处理。医生评审包中使用的是最终确定的 baseline 版本，它与 inference harness 采用相同的字段定义和输出契约。

### 2.4 inference harness

不同目标字段对上下文和推理的需求并不相同，因此 harness 采用多个处理阶段。

首先，字段专用提示分别提取就诊背景、癌症诊断、实验室结果、客观发现、当前用药和近期治疗变化。依赖型提示随后接收经过选择的前序结果。例如，分期和转移状态为治疗意图提供背景，当前治疗和临床发现则为疗效评估提供背景。计划类字段主要从 Assessment and Plan 章节提取。

每项模型输出依次通过五个验证阶段：

1. JSON 解析失败时修复格式。
2. 验证 schema，发现错误字段名或泄漏的字段名。
3. 检查具体性和语义一致性，识别含糊或答非所问的答案。
4. 进行忠实度修剪，删除明确无依据或相互矛盾的内容，同时保留有支持的信息。
5. 进行时态过滤，从未来计划字段中删除已完成或历史事件。

确定性处理层使用高置信度临床规则解决反复出现的错误。这些规则区分区域淋巴结和远处疾病、疑似和确诊转移、当前抗癌治疗与支持治疗或居家用药，以及当前疗效与治疗前变化。系统还返回支持每个提取值的病历原文片段。

| inference harness 组件 | 针对的失败模式 | 示例 |
|---|---|---|
| 字段专用提取 | 一个大型提示遗漏或混合字段 | 分开提取当前用药和治疗计划 |
| 依赖感知的上下文 | 相关字段相互矛盾 | 解释分期和疗效时使用转移状态 |
| 语义验证 | 答案与医学相关，但没有回答目标字段 | 从当前疗效中删除未来治疗计划 |
| 忠实度修剪 | 模型增加无依据的结论 | 活检结果待定时，将病灶保留为疑似 |
| 时态过滤 | 已完成的结果出现在未来计划中 | 从影像计划中删除已完成的扫描 |
| 药物与上下文规则 | 仅根据药名分类，忽视临床背景 | 区分抗癌药、居家用药和支持药物 |
| 跨字段临床规则 | 分期、淋巴结和远处疾病不一致 | 将腋窝淋巴结保留为区域受累，而非远处转移 |
| source attribution | 审查者无法追溯提取值 | 返回病历中的支持性原句 |

<div data-rough-figure="1"></div>

***图 1.*** *开发、适配与评估流程。乳腺癌开发期间，医生作者与模型开发作者将反复出现的提取错误转化为提示、验证和确定性规则。随后，研究通过模型辅助审查将 inference harness 适配到胰腺癌，该阶段没有临床医生审阅胰腺癌输出。最终，三位肿瘤科医生在隐藏系统标签的条件下，比较完整 inference harness 与使用相同模型的 single-prompt baseline。*

### 2.5 临床信息指导的乳腺癌开发

乳腺癌开发在 56 份附加病历上进行了约 15 轮有记录的迭代。一位熟悉肿瘤学问题的医生作者与模型开发作者共同审阅部分输出。这位医生不是肿瘤专科医生，也没有参与最终的肿瘤科医生评估。研究团队将发现的问题与完整源病历逐项核对，对反复出现的错误进行归类，并把它们转化为可推广的提示修改、验证逻辑或确定性规则。该过程没有建立传统的监督训练集，也没有更新模型权重。

候选修改只有在受影响样本和此前正确的对照样本上通过测试后才会保留。这里的目标不是修正单个病例，而是编码反复出现的临床区分。例如，腋窝淋巴结受累被误判为远处转移后，团队加入了通用的区域淋巴结规则；计划使用的药物被误列为当前治疗后，则加入了药物时态规则。

> **协作说明：** 2.1 保留为研究设计总览；2.5 单独描述乳腺癌阶段的具体开发程序。两节不是同一层级的信息，因此不建议合并。

### 2.6 model-in-the-loop 的胰腺癌改进

随后，我们通过约 18 轮有记录的开发，将乳腺癌 inference harness 适配到全部 100 份胰腺癌附加病历。在这一阶段，没有临床医生审阅胰腺癌输出。pipeline 先生成结构化字段，再由依据评分准则配置的 Qwen 审查模型将这些字段与完整源病历进行比较。开发环境中的另一个通用 LLM 负责汇总累积的审查发现，并提出提示、规则或 workflow 的候选修改。研究团队审查这些方案，只实施有明确依据的修改，并在 regression test 通过后保留。

审查提示包含字段定义、严重程度标准，以及乳腺癌开发期间确定的临床区分。这样既能用已有错误类别指导新癌种的审查，也能发现胰腺癌特有的问题，例如治疗方案名称和剂量表达。模型参与错误发现和候选修改，但具体实施由研究人员控制。因此，我们将其称为模型辅助、研究人员监督的改进，而不是自主的“自我进化”。

在两个开发阶段，pipeline 都记录原始模型输出、每项验证操作和确定性修正。这些记录使团队能够追溯最终字段值在 harness 中的处理过程，并根据回归结果保留或拒绝修改建议。

### 2.7 由医生确定的核心临床字段

匹配比较开始前，参与开发的医生作者根据理解疾病状态和当前治疗决策的重要性，指定了七个核心临床字段：

1. 患者当前正在接受哪些抗癌药物？
2. 当前癌症分期是什么？
3. 是否存在远处转移，远处转移是存在、不存在还是不确定，涉及哪些部位？
4. 有哪些区域或总体转移受累得到证据支持？
5. 癌症对当前治疗的反应如何？
6. 乳腺癌类型及 ER/PR/HER2 状态是什么？
7. 病历记录了哪些已经完成的分子或遗传检测结果？

最终的肿瘤科医生评估工具还包括遗传检测计划、支持用药、操作计划、影像计划、实验室检查计划、用药计划和近期治疗变化。实验室结果摘要和一般临床发现为选评字段，不纳入主要临床分析。

### 2.8 开发阶段的 LLM 辅助审查

LLM 辅助审查是开发阶段使用的工具，尤其服务于胰腺癌适配。针对每个候选输出，审查模型依据预先设定的字段定义和严重程度标准，将提取结果与源病历进行比较，并标记可能的遗漏、无依据内容、语义错配和时态错误。这些发现只用于形成候选修改，最终是否实施仍由研究人员判断，并通过 regression test 确认。

研究还用同一流程完成了 260 项适用病历字段比较的匹配审查，用于技术错误分析。LLM 的判断不作为最终临床结局标签，也不能替代独立的肿瘤科医生评估。

> **协作说明：** 这里不再罗列“发现了哪些错误、之后修了什么”。这些内容更像开发报告。Methods 只说明 LLM 审查的用途、边界和它与最终医生评估的区别。

### 2.9 肿瘤科医生评估

临床评估通过隐藏系统标签的 A/B 界面展示源病历和两份结构化输出。评估者针对每个字段选择 A 更优、B 更优或平局。他们不知道哪份输出来自 inference harness，也不知道两个系统是否使用同一基础模型或研究预期哪一侧更好。所有评估采用固定映射，inference harness 显示为 A，single-prompt baseline 显示为 B。source attribution 作为完整 harness 输出的一部分随 A 展示。因此，本研究比较的是呈现给医生的完整系统输出，没有单独隔离 attribution 的作用。

参与乳腺癌开发的医生作者不属于最终评估者。三位肿瘤科医生独立评估了乳腺癌输出，其中两位同时评估了胰腺癌输出，第三位没有评估胰腺癌。

三位肿瘤科医生均完成了同样的 280 项乳腺癌必评比较。我们计算了每两位医生之间的完全一致率和 Cohen's kappa，并分别汇总五组已完成的评审者与癌种组合。合并计数仅作描述性统计，不把各字段判断视为相互独立的观测。

主要推断分析排除平局，将非平局判断是否偏好 inference harness 作为二分类结局。我们使用总体平均的 logistic 广义估计方程，在每份病历内采用可交换相关结构。模型将评审者作为固定效应，总体模型同时校正癌种，从而处理同一病历中跨字段、跨评审者的重复判断。结果报告调整后比值比、95% 置信区间和双侧 p 值。敏感性分析将每组字段评分汇总为病历级的 harness 减 baseline 净差，并使用精确符号检验。

## 3. 结果

### 3.1 开发路径与跨癌种适配

开发记录包括对 56 份乳腺癌病历进行的约 15 轮迭代，以及对 100 份胰腺癌病历进行的约 18 轮迭代。乳腺癌部分的修订来自医生作者与模型开发作者的共同审阅。胰腺癌部分的修订没有临床医生审查相应输出，而是采用前述适配后的评估标准、基于模型的错误审查、外部 LLM 辅助综合，以及由研究人员控制的实施流程。

开发中有一部分组件可以原样迁移，包括五个验证阶段、时态区分、source attribution，以及区分当前治疗、治疗计划和支持性用药的规则。疾病术语、治疗方案解读和后处理条件则需要按癌种分别处理。因此，该系统复用了 failure-handling workflow，但没有假设乳腺癌与胰腺癌在临床上可以互换。

胰腺癌的临床医生评估是检验这种适配的主要证据。在两次独立评估中，医生有 161 次偏好 inference harness、23 次偏好 baseline，另有 335 次平局。第一位医生按病历计算得到 19 例正向净差和 1 例负向净差，第二位医生得到 18 例正向净差和 2 例平局。合并两位医生的判断后，20 例胰腺癌病例的 harness 减 baseline 净差均为正值。由于开发期间没有医生审查胰腺癌输出，这一模式说明先前编码的评估标准和 workflow 可以跨癌种复用。但仅凭该结果，无法判断收益来自共享规则、胰腺癌特异性修订、基于模型的审查器，还是这些因素的共同作用。

### 3.2 当前肿瘤科医生评估

目前的临床证据包括五组已完成的评审者与癌种组合：三位肿瘤科医生均完成了乳腺癌评估，其中两位还完成了胰腺癌评估。在 1,359 项必评字段判断中，harness 获偏好 443 次，baseline 获偏好 77 次，另有 839 次平局，分别占全部判断的 32.6%、5.7% 和 61.7%。在 520 项非平局判断中，harness 占 85.2%。

| 已完成的评估 | 必评判断数 | harness | baseline | 平局 | harness 在非平局判断中的占比 |
|---|---:|---:|---:|---:|---:|
| 肿瘤科医生 01，乳腺癌 | 280 | 84 | 22 | 174 | 79.2% |
| 肿瘤科医生 01，胰腺癌 | 260 | 75 | 14 | 171 | 84.3% |
| 肿瘤科医生 02，乳腺癌 | 280 | 79 | 8 | 193 | 90.8% |
| 肿瘤科医生 02，胰腺癌 | 259 | 86 | 9 | 164 | 90.5% |
| 肿瘤科医生 03，乳腺癌 | 280 | 119 | 24 | 137 | 83.2% |
| **所有已完成的评估** | **1,359** | **443** | **77** | **839** | **85.2%** |

<div data-rough-figure="2"></div>

***图 2.*** *五组已完成的评审者与癌种组合中，隐藏系统标签后的临床医生偏好分布。多数判断为平局，而五组评估的非平局判断均偏向 inference harness。*

三位肿瘤科医生在乳腺癌数据集上均独立偏好 harness。合并三位医生的乳腺癌评估后，harness 获偏好 282 次，baseline 获偏好 54 次，另有 504 次平局。第一位肿瘤科医生按病历评估的结果为 harness 胜出 18 例、baseline 胜出 1 例、平局 1 例。第二位和第三位医生均在 20 份病历中全部偏向 harness。将三位医生对每份乳腺癌病历的评分合并后，20 份病历的 harness 减 baseline 净差均为正值。

两次胰腺癌评估合计包括 161 项 harness 偏好、23 项 baseline 偏好和 335 项平局。合并两位医生的判断后，20 份胰腺癌病历的 harness 减 baseline 净差均为正值。

调整后的方向性分析显示，在非平局判断中，临床医生偏好 inference harness 的 odds 是偏好 baseline 的 5.70 倍，95% CI 为 4.24 至 7.66，p<0.001。乳腺癌的调整后 OR 为 5.18，95% CI 为 3.61 至 7.46；胰腺癌为 6.94，95% CI 为 4.19 至 11.48，两者均 p<0.001。作为病历级敏感性分析，合并同一病历的医生评分后，全部 40 份病历的 harness 减 baseline 净差均为正，双侧精确符号检验 p<0.001。该结果说明现有评分中的偏好方向非常稳定，但三位医生仍不足以精确估计更广泛肿瘤科医生群体中的差异。

> **协作说明：** 这就是此前标记为“尚待完成”的分析。现在数据已经足够拟合模型，所以英文稿中的占位内容已删除并替换为实际结果。该模型能支持“现有病历评分中存在显著偏好”，但不能把三位医生直接外推为整个肿瘤科医生群体。

### 3.3 评审者间一致性与核心字段

三位肿瘤科医生均完成了同样的 280 项乳腺癌必评比较。医生 01 与 02 的完全一致率为 82.9%，医生 01 与 03 为 80.0%，医生 02 与 03 为 73.6%。对应的 Cohen's kappa 分别为 0.646、0.644 和 0.511。三位医生在 192 项比较中给出相同判断，占 68.6%。按简单多数票计算，95 项偏向 harness，15 项偏向 baseline，168 项为平局；另有 2 项分别得到一票 harness、一票 baseline 和一票平局，因此没有多数结果。

<div data-rough-figure="3"></div>

***图 3.*** *三位肿瘤科医生对 280 项共同乳腺癌字段比较的两两一致性。完全一致率为 73.6% 至 82.9%，Cohen's kappa 为 0.511 至 0.646。*

> **协作说明：** 这里严格来说不是 correlation，而是 agreement。相关性只能说明两个人的评分变化方向相似，即使其中一人系统性地更偏好某一选项也可能很高；完全一致率和 Cohen's kappa 更适合回答“医生是否给出相同判断”。

在所有已完成的评估中，七个由医生确定的核心类别共有 660 项适用判断。harness 获偏好 276 次，baseline 获偏好 34 次，另有 350 次平局。因此，在 310 项核心字段的非平局判断中，harness 占 89.0%，且每个核心类别的汇总净差均为正值。

| 核心字段 | harness | baseline | 平局 | 净优势 |
|---|---:|---:|---:|---:|
| 当前抗癌药物 | 83 | 2 | 15 | +81 |
| 分期 | 41 | 4 | 55 | +37 |
| 远处转移 | 15 | 2 | 83 | +13 |
| 区域或总体转移 | 60 | 5 | 35 | +55 |
| 治疗反应 | 37 | 3 | 60 | +34 |
| 乳腺癌类型与受体状态 | 18 | 12 | 30 | +6 |
| 已完成的分子或遗传检测结果 | 22 | 6 | 72 | +16 |
| **总体** | **276** | **34** | **350** | **+242** |

<div data-rough-figure="4"></div>

***图 4.*** *各个由医生确定的核心临床类别中的临床医生偏好。harness 在识别当前治疗和判断转移累及方面优势最大，乳腺癌类型与受体状态的净优势最小。*

当前抗癌药物以及区域或总体转移的优势最大，也最为稳定。在七个核心类别之外，药物计划共有 59 次偏好 harness、0 次偏好 baseline 和 41 次平局。操作或手术计划的差距较小，共有 14 次偏好 harness、12 次偏好 baseline 和 74 次平局。乳腺癌类型与受体状态仍是表现最弱的核心类别，净优势为 6 项判断。

病历级汇总也得出了相同方向的结果，同时避免将每个字段视为相互独立的观察值。按癌种合并医生判断后，20 份乳腺癌病历和 20 份胰腺癌病历的 harness 减 baseline 净差均为正值。五组评审者与癌种组合的汇总结果也全部偏向 harness。这些结果作为调整后模型的敏感性分析。

<div data-rough-figure="5"></div>

***图 5.*** *各份病历中标准化临床医生偏好净差的分布。按癌种合并现有医生评分后，harness 在每份乳腺癌和胰腺癌病历中均取得正向净差。*

<div data-rough-figure="6"></div>

***图 6.*** *非平局判断中，临床医生偏好 inference harness 的调整后比值比。模型使用 logistic 广义估计方程，并按病历聚类；点表示比值比，横线表示 95% 置信区间。*

### 3.4 开发阶段的 LLM 审查

开发阶段使用一个基于源病历的 LLM 审查器，对 260 项适用的核心病历字段比较进行判断。该审查器认为 harness 较优 66 次、baseline 较优 28 次，另有 166 次平局。七个类别中有六个类别的 harness 净差为正，分期是唯一净差为负的类别。这些结果用于开发和错误分析，不属于最终临床结局标签。

| 核心字段 | harness | baseline | 平局 | 净优势 |
|---|---:|---:|---:|---:|
| 当前抗癌治疗 | 8 | 0 | 32 | +8 |
| 分期 | 6 | 8 | 26 | -2 |
| 远处转移 | 11 | 3 | 26 | +8 |
| 区域或总体转移 | 14 | 4 | 22 | +10 |
| 治疗反应 | 13 | 6 | 21 | +7 |
| 乳腺癌类型与受体状态 | 8 | 5 | 7 | +3 |
| 已完成的分子或遗传检测结果 | 6 | 2 | 32 | +4 |
| **总体** | **66** | **28** | **166** | **+38** |

<div data-rough-figure="S1"></div>

***补充图 S1.*** *开发阶段 LLM 审查与肿瘤科医生评估中各类别的净偏好率。由于两者使用的评审者和 harness 版本不同，该图只用于描述结果模式。*

### 3.5 自由文本评论与平局的含义

两份医生评分导出共包含 12 条书面评论。这些评论说明 A/B 评估中的“平局”有两种不同含义：两个输出都可以接受，或者两个输出都不完整或不正确。在一项乳腺癌影像计划比较中，一位医生认为 baseline 只总结了已完成的影像发现，因此偏好 harness；另一位医生则认为两个输出都没有正确表达当前并无新影像计划，因此选择平局。

其他评论还指出无依据的受体状态、区域淋巴结遗漏、刚开始新方案时疗效尚不明确，以及两个输出均不理想的情况。这些评论用于解释评分尺度和选择错误分析案例，不作为独立的定量结局。

> **协作说明：** 这就是原来的 3.6。它不是为了再证明一次 harness 胜出，也不是相关性分析，而是提醒读者不能把 839 个平局全部理解为“两个系统都答对了”。目前保留为一个短结果段，并在 4.9 将 `TIE` 含义不唯一列为主要 limitation。下一步可抽取一部分 `TIE` 做人工 adjudication。

## 4. 讨论

### 4.1 主要解读

多位肿瘤科医生参与的评估显示，相较于使用同一模型的 single-prompt baseline，临床医生显著更偏好 inference harness。调整后的方向性分析中，harness 偏好的 OR 为 5.70，95% CI 为 4.24 至 7.66，p<0.001。

本文的主要结果来自模型外围的开发流程。临床医生帮助识别乳腺癌场景中反复出现的错误，团队将这些错误转化为 inference harness 中的明确组件，随后在胰腺癌开发阶段没有医生参与审阅的情况下，将这一流程适配到胰腺癌。最终的临床评分在两个癌种中都更偏向该 inference harness。与静态比较两个提示相比，这一过程更直接地将开发方法与评估结果联系起来。

多数字段比较为平局，这是可以预期的，因为两个系统使用同一个能力较强的基础模型。不过，当肿瘤科医生认为两者存在实质差异时，偏好 inference harness 的次数接近偏好 baseline 的六倍。优势主要集中在需要时间关系判断或临床分类的字段。这符合系统的设计思路：保留基础模型已经答对的简单问题，只在出现已知失败模式时介入。

目前最主要的方法学限制有两个。第一，研究没有 component ablation，无法分别估计 task decomposition、verification gate、deterministic rule 或 source attribution 的贡献。第二，`TIE` 可能表示两个输出都可以接受，也可能表示两个输出都不理想。因此，现有结果支持医生对完整 harness 输出的相对偏好，但不能直接证明绝对正确率，也不能把效果归因于某一个组件。

### 4.2 由临床信息指导、可复用的 inference harness

乳腺癌开发阶段中，医生作者的角色不同于常规的数据集标注。他没有为监督训练制作大量字段标签，而是与模型开发作者一起查看具体错误，并界定有临床意义的区别。团队再把这些区别编码成 prompt、verification check 和 deterministic rule。三位独立肿瘤科医生的最终评估说明，形成的系统行为并不只是在复现开发医生个人的偏好。

结构化 extraction 的价值在于，它可以把一份很长、内部重复较多的病历转换成一组紧凑字段，方便检查、检索和下游使用。当任务对医生而言费时，但边界可以清楚定义时，这种辅助尤其有价值。例如，医生需要在药物清单和治疗史中反复核对当前抗癌治疗，而 harness 可以先完成这部分整理。

我们使用 inference harness 这个名称，是因为贡献位于模型权重之外。routing、schema、verification logic、deterministic rule、logging 和 attribution 都是围绕冻结模型工作的。因此，经过接口和 prompt 校准后，同一套架构原则上可以包在另一个 instruction model 外面。本研究只测试了 Qwen2.5-32B，所以跨模型迁移目前是架构上的可行性，而不是已经验证的结果。

### 4.3 跨癌种适配与 LLM-as-a-judge

胰腺癌阶段检验的是，已经形成的 workflow 能否适配到医生作者没有直接提供反馈的领域。胰腺癌病历在疾病进程、治疗方案、分期表述和手术背景方面均有不同。我们保留共同的临床区分，在需要时增加癌种专用 routing，并且没有让临床医生参与胰腺癌输出的开发审阅。

这次适配中，AI 承担了两个不同角色。首先，我们把乳腺癌阶段累积的错误记录和经验交给外部通用 LLM，由它提出第一版胰腺癌适配方案。随后，依据 rubric 配置的 LLM-as-a-judge 将每轮输出与源病历比较，定位可能的遗漏、无依据内容、semantic mismatch 和 temporal error。这个 judge 的重要作用是让迭代过程变得可观察，使我们知道适配后的 harness 在哪里表现好、在哪里仍然失败。

LLM-as-a-judge 不定义最终临床结局，也不能自行批准代码修改。研究人员选择候选修改并运行 regression test，最终证据仍来自独立肿瘤科医生。胰腺癌评估中，harness 获偏好 161 次，baseline 获偏好 23 次，另有 335 次平局。这支持 model-assisted development loop 的实用性，但不等于自主 self-improvement。

### 4.4 模型难点与 human-model complementarity

字段级结果可以区分直接 extraction 与需要时间关系或临床判断的任务。明确写出的事实往往得到平局，因为两个系统都能找到；更大的差异出现在模型需要判断这些事实在当前语境中意味着什么的时候。

当前抗癌药物是优势最大的核心字段，结果为 83 次偏好 harness、2 次偏好 baseline 和 15 次平局。用药计划也以 59 比 0 明显偏向 harness，另有 41 次平局。这类任务需要在较长的治疗史中区分当前、既往、支持性和计划中的药物，对人工审阅而言费时，但适合由结构化系统先行整理。

分期、转移累及和治疗反应体现了另一种互补关系。病历如果直接给出 stage，医生通常可以很快确认；模型却可能把区域淋巴结当成远处转移，或把疑似病灶当作确诊。治疗反应对双方都更难，因为必须把当前方案与影像、症状和肿瘤标志物的时间线对应起来。对于这些字段，系统更适合整理证据并给出候选答案，再由医生快速复核，而不是替代临床判断。

乳腺癌类型与受体状态的优势最小，结果为 18 比 12，另有 30 次平局。这与该字段经常被直接写在病历中相符，因此 baseline 已经能够完成很多样本。更合理的人机分工应按字段决定：自动化重复而耗时的综合任务，对判断敏感的字段展示证据，并让医生复核紧凑输出，而不是重新通读整份病历。

<div data-rough-figure="7"></div>

***图 7.*** *肿瘤信息 extraction 中 human-model complementarity 的概念图。图中位置是定性解释，不是实测难度分数。它区分了适合自动化的重复综合任务，以及更适合由模型整理证据、医生快速复核的判断敏感字段。*

> **协作说明：** 这张图表达的是你说的“医生很容易、模型反而容易错”和“模型可以替医生处理冗长信息”这两类互补关系。它是 conceptual figure，不会伪装成我们实际测量过医生工作量。

### 4.5 作用机制与 ablation 的必要性

现有字段结果与 inference harness 的设计目标一致。当前用药和用药计划的结果对应专用 prompt、temporal filtering 和 drug-context rule；分期和转移结果对应 cross-field check，以及区分区域与远处疾病、疑似与确诊发现的 deterministic rule。这些关联支持我们提出的机制，但不能单独证明每个组件贡献了多少。

我们核对的相关论文并不是全部都有完整 ablation。Wiest 等比较了 plain zero-shot、one-shot、加入定义和 grammar-constrained prompting 等方案 [3]；mCODEGPT 直接比较 single-step prompting 与两种 hierarchical prompting [7]；Grothey 等比较五种 prompting strategy 和 quantized model configuration [8]。Dao 等单独报告 validation and retry loop 修正了多少初始错误，但没有对所有 pipeline 组件逐一移除 [6]。Tariq 等比较完整 hybrid system 与 zero-shot、structured-code 和 rule-based baseline，也没有逐个拆除 hybrid system 的内部阶段 [5]。

我们的研究同样缺少完整 component ablation。一个成本可控的技术实验可以依次比较 single-prompt baseline、仅 task decomposition、decomposed prompts 加 verification gates，以及 full harness。要求肿瘤科医生对每个版本重复评分并不现实。LLM-reviewed ablation 可以帮助定位机制，但只能作为 technical evidence，不能代替独立医生评估。更强但仍可行的方案，是只让医生复核不同 ablation 版本之间真正发生变化的少量输出。

### 4.6 与既有临床 extraction 研究的关系

CORAL 建立了本研究使用的真实病历 benchmark。原研究比较了 zero-shot GPT-4、GPT-3.5-turbo 和 FLAN-UL2，并由一位独立肿瘤科医生审阅部分 GPT-4 输出，仍发现遗漏和幻觉 [2]。我们的研究在同一 benchmark 上回答另一个问题：当基础模型保持不变时，结构化 workflow 是否优于 single-prompt baseline。

其他研究分别支持本系统中的某些组件。Wiest 等说明 prompt design 和 constrained output 可以改善本地模型对临床二分类字段的 extraction [3]。mCODEGPT 在合成肿瘤病历上发现 hierarchical prompting 优于 single-step strategy [7]。Grothey 等展示了前列腺病理 extraction 在不同 model、prompt 和 quantization configuration 下存在明显差异 [8]。Dao 等的架构与本研究较接近，结合 engineered context、validation 和 retry，但任务是从右心导管报告中提取数值 [6]。Tariq 等使用 UMLS 加 fine-tuned LLM，在乳腺癌治疗时间线上取得更大的数据规模和 external validation [5]；Bhattarai 等则在纵向肺癌表型任务上比较多种模型和 rule-based approach [4]。

本研究的区别不是单独发明 prompt、retry 或 rule，而是把 field-specific routing、selective context transfer、verification gates、deterministic oncology rules、logging 和 source attribution 组合为一个 inference harness，再用 same-model baseline 隔离外围 workflow 的作用，并由肿瘤科医生逐字段直接比较。乳腺癌到胰腺癌的开发顺序还检验了临床信息形成的 failure rules 能否在第二个癌种中减少对稀缺专科医生时间的反复占用。

因此，更准确的定位是：这是一项对整合式 inference harness 的、小规模但由肿瘤科医生直接评估的研究，而不是第一个肿瘤 extraction 系统，也不是规模最大的验证。逐篇对照表移到文末中文审阅附录，正文只保留与论点直接相关的引用。

### 4.7 从结构化提取到患者沟通

探索性的患者信件审阅显示了一些有限的下游优势。与 ChatGPT 相比，harness-based letter 在准确性、完整性、易理解性和实用性的平均评分上均更高，收到的 hallucination 标记也更少；但它没有明确优于 same-model Qwen baseline。因此，更好的 extraction 不会自动产生更好的完整信件。信件质量还取决于内容选择、组织方式、措辞、对不确定性的解释，以及应向患者提供哪些临床细节。

这一结果说明，项目应先验证 structured extraction，再推进患者沟通。Structured extraction 是可测量的 safety layer，患者沟通则是建立在这一层之上的 downstream task。extraction 研究可以先判断分期、治疗、疗效和计划是否得到忠实表达，再由 generator 将这些内容写成连贯文字。未来的信件研究应检验具体的 extraction 改进能否经过第二次转换后继续保留，最好同时邀请患者读者和临床医生参与评估。

信件实验更适合作为 future work 的动机，而不是独立的 efficacy result。它既显示了 structured extraction 可能帮助患者沟通的环节，也说明在第二次文本生成过程中，extraction 阶段已经建立的优势可能再次丢失。

### 4.8 临床与技术意义

研究结果表明，肿瘤学 extraction 中的部分 LLM 错误具有足够的重复性，可以在不修改模型权重的情况下处理。当标注训练数据有限，或临床团队需要在不重新训练模型的情况下调整字段定义时，这一点很有价值。例如，将待活检病灶保留为疑似病变的 rule 可以被直接检查和测试。只依赖 prompt 的系统较难提供同等控制，因为一次措辞修改可能影响无关字段。

我们使用 inference harness 这一名称，是因为系统除 prompt sequence 外，还围绕冻结模型管理 task routing、dependency、verification、deterministic correction、logging 和 source attribution。Clinical workflow 也能描述该系统，但可能让人误以为它已经被整合进常规诊疗，而本 pilot 并未检验这一点。因此，本文用 inference harness 描述技术贡献，用 evaluation workflow 描述研究程序。

该系统也支持本地部署。本地运行本身不能证明系统符合隐私要求或具备临床安全性，但能让机构掌控病历处理和系统更新。本研究评估的是提取质量，而不是自主临床使用的准备程度。

### 4.9 研究边界与 next steps

本 pilot 用于判断观察到的 effect 和临床一致性是否足以支持扩大研究规模。三位肿瘤科医生独立复现了乳腺癌评估的总体方向，两位医生也复现了胰腺癌方向。调整后的 clustered analysis 已经确认这些评分中的偏好很强。

最主要的两个 limitation 是缺少 component ablation，以及 `TIE` 的含义不唯一。前者使我们无法判断收益具体来自 task decomposition、verification gate、deterministic rule 还是 source attribution；后者无法区分“两个输出都对”和“两个输出都错”。这两点限制了机制解释和绝对正确性结论，但不改变现有相对偏好的方向。

研究范围仍然有限。最终临床结果来自三位肿瘤科医生，CORAL 来自单一机构、只覆盖两个癌种，而且我们尚未测试另一个 model family。benchmark 的技术审查影响过部分后续修改，所以本研究属于 benchmark-informed pilot，而不是 untouched external validation。不过，胰腺癌开发期间确实没有临床医生审阅 PDAC 输出，这一较窄的观察不受影响。

下一项研究应优先人工 adjudicate 一部分 `TIE`，完成前述 staged ablation，再增加肿瘤科医生，并在第二种模型和外部医疗系统数据上复现。这样可以更清楚地解释 harness 为什么有效，并支持更强的 evaluator-level 和 institution-level generalization。

## 5. 结论

[最终结论：在一项由多位肿瘤科医生参与、隐藏系统标签的评估中，由失败模式驱动的 inference harness 相较于使用同一模型的 single-prompt baseline 获得了显著的临床医生偏好。]

当前试点支持一种开发策略：临床医生识别具有临床意义的失败模式，团队将这些模式转化为明确的 inference harness，再通过 AI 辅助且由人工监督的改进，将该 inference harness 适配到另一个癌种。三位肿瘤科医生在乳腺癌评估中都更偏好该 inference harness，其中两位在胰腺癌评估中也更偏好该 inference harness，而胰腺癌开发阶段没有医生参与。最大的优势出现在当前治疗、用药计划和转移状态字段。模型权重在整个过程中保持冻结。在提出广泛的确证性结论前，仍需完成 component ablation、区分不同类型的 `TIE`，并增加肿瘤科医生和 external validation。

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

## 附录 A：相关工作对照表（供合作者审阅）

> **说明：** 这张表用于内部核对和讨论，不放入英文主稿。正文 4.6 已改为自然引用。表中的 ablation 描述来自原论文方法与结果部分的逐项核查。

| 研究 | 数据与任务 | 临床专家角色 | 方法与 component comparison | 与本研究的关系 |
|---|---|---|---|---|
| Sushil 等，CORAL [2] | 40 份真实乳腺癌和胰腺癌病历；广泛 oncology schema | 专家标注；一位独立肿瘤科医生审阅每个癌种 10 份 GPT-4 输出 | 比较 zero-shot GPT-4、GPT-3.5-turbo 和 FLAN-UL2；没有 inference harness component ablation | 提供相同 benchmark 和临床范围，但未进行 same-model harness 对 single-prompt baseline 的评估 |
| Wiest 等 [3] | 500 份 MIMIC 病史；5 个二分类字段 | 三位盲法医学专家建立 consensus ground truth | 比较 plain zero-shot、one-shot、加入定义和 grammar-constrained prompting；属于 prompt-level component comparison | 支持 prompt design 和 constrained output 的作用，但任务不是肿瘤学，输出范围较窄 |
| Bhattarai 等 [4] | 63 位肺癌患者的 13,646 份病历；4 个纵向表型 | 两位领域专家提供 gold-standard annotation | 比较 GPT、open model 和 rule-based method；不是单一 pipeline 内部的逐组件 ablation | 数据规模更大，但 target phenotype 较少，也没有 same-model workflow 对照 |
| Tariq 等 [5] | 26,692 位内部乳腺癌患者和 162 位外部患者；治疗时间线 | registry label；专家整理治疗概念和 code | 完整 UMLS + fine-tuned LLM hybrid system 对 zero-shot、structured-code 和 rule-based baseline；没有逐个移除 hybrid phase | external validation 更强，但依赖 supervised fine-tuning，任务集中于 5 类治疗 |
| Dao 等 [6] | 220 份开发和 200 份验证用右心导管病历 | 一位肺血管疾病专家建立 validation ground truth 并指导开发 | engineered preload、validation、retry；单独报告 retry 对初始错误的修正情况，但没有完整 factorial ablation | workflow 架构最接近，但任务是 procedure note 中的数值 extraction |
| Zhang 等，mCODEGPT [7] | 1,000 份合成肿瘤病历；49 个 mCODE entity | 自动匹配结合人工 validation | 直接比较 single-step baseline、BFOP 和 2POP hierarchical prompting；是最接近的 prompt ablation | 证明 hierarchy 有价值，但数据为 synthetic notes，且没有 blinded oncologist comparison |
| Grothey 等 [8] | 579 份德语和英语前列腺病理报告；11 个字段 | 医学博士生在主治病理医生指导下标注 | 比较多个 model、5 种 prompting strategy 和 quantization configuration | component/configuration comparison 较完整，但任务是一类 pathology report，不是纵向门诊病历 |
