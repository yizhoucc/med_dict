<!--
BILINGUAL MAINTENANCE RULE:
1. Keep the complete English manuscript first and the complete Chinese manuscript second in this same file.
2. The two versions must match in scientific claims, numbers, main result tables, figure numbering, and references. The Chinese review version may contain additional annotations and review-only appendices.
3. Keep the English version close to submission prose. The Chinese version may retain collaborator questions, figure-design notes, and explanatory annotations.
4. After changing either version, update the other version in the same edit and run: python3 render_workshop_draft.py
5. Apply the humanizer pass to both versions. Chinese should read as natural academic prose, not as a literal machine translation.
-->

<div id="english-version"></div>

# From clinician-guided extraction to model-assisted cross-cancer adaptation: an inference harness for oncology notes and patient communication

**Pilot report**

Version 0.14, September 2026

## Abstract

### Background

Large language models can extract structured information from clinical notes, but a single prompt often confuses current and historical treatment, suspected and confirmed disease, and completed findings and future plans. These errors are difficult to detect because the resulting text remains fluent. Oncology notes are a demanding test case because they combine pathology, imaging, treatment history, current therapy, response assessment, and conditional plans across a long clinical timeline. Structured extraction can also provide an inspectable intermediate representation before the same information is rewritten for patients.

### Objective

To test whether a clinician-informed inference harness can improve oncology information extraction from a frozen, locally served open-weight language model, adapt to a second cancer domain without target-domain clinician review, and support downstream patient-letter generation.

### Methods

We built an inference harness around Qwen2.5-32B-Instruct-AWQ in two development stages. During breast-cancer development, a physician coauthor and the model-development investigators repeatedly reviewed outputs and converted recurring clinical errors into field-specific prompts, verification gates, and deterministic oncology rules. For pancreatic cancer, ChatGPT helped synthesize the breast-cancer error history into a transferable review rubric and candidate adaptations. The Qwen pipeline generated pancreatic-cancer fields, and a rubric-guided Qwen reviewer compared each output with the source note. Investigators selected, implemented, and regression-tested all retained changes. No clinician reviewed pancreatic-cancer outputs during this stage, and model weights remained frozen. We then compared the harness with a same-model single-prompt baseline on 40 CORAL benchmark notes. Three oncologists evaluated breast-cancer extraction, and two also evaluated pancreatic-cancer extraction. In a separate exploratory analysis, patient letters were generated from the structured extraction plus source-note context and compared descriptively with direct note-to-letter Qwen and GPT-4o baselines.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. The five completed clinician-by-cancer evaluations contributed 1,359 required-field judgments. The harness was preferred in 443, the baseline in 77, and 839 were ties. Ties accounted for 61.7% of all judgments; among the 520 directional judgments, 85.2% favored the harness. The adjusted directional analysis estimated an odds ratio of 5.70 for harness preference (95% CI 4.24 to 7.66; p<0.001). In pancreatic cancer, where no physician had participated in development, two oncologists together preferred the harness 161 times and the baseline 23 times, with 335 ties. In the exploratory letter review, the harness-based route had a mean score difference of +0.24 versus direct GPT-4o generation and -0.06 versus direct same-model Qwen generation.

### Conclusions

The results support a staged development path. Clinician feedback first shaped a breast-cancer inference harness. A model-assisted, investigator-supervised loop then adapted the accumulated rubric and failure rules to pancreatic cancer without target-domain clinician review. Structured extraction can provide a checked intermediate layer for downstream patient communication, although the current letter findings remain exploratory. Broader claims require more oncologists, tie adjudication, component ablation, and external validation.

## 1. Introduction

Most clinically useful information in oncology is still recorded in free text. A progress note may contain the diagnosis, pathology, receptor status, treatment history, toxicities, response, and next steps, but these facts are spread across sections and timepoints. Manual review is slow, and conventional extraction systems require substantial annotation and task-specific development. Large language models can read many note styles and return structured fields without a new supervised model for every question.

Patient communication is a natural downstream use of this information. Direct note-to-letter generation asks one model call to identify the important facts, resolve their timing and certainty, and explain them safely. We instead treat structured extraction as an intermediate layer that can be inspected before prose generation. The letter generator receives both the extracted fields and source-note context, which preserves access to the original record while making the clinical content more explicit.

Research activity has moved faster than routine clinical adoption. A 2025 scoping review identified 24 studies of language-model-based oncology information extraction, but also found that external validation and real-world workflow integration remained limited [1]. Clinical deployment has a higher bar than a demonstration on a benchmark. A useful system must preserve uncertainty, distinguish current care from historical events, avoid unsupported facts, and produce results that a clinician can trace back to the note.

Longitudinal oncology notes expose weaknesses that are easy to miss in simpler extraction tasks. A medication list can contain anticancer therapy, supportive treatment, chronic home medications, discontinued drugs, and therapies that are only being discussed. Regional lymph nodes must not be labeled as distant metastases. A suspicious lesion awaiting biopsy must not become confirmed stage IV disease. Tumor growth before treatment is not evidence of failure of a regimen that has just started. A model can mention medically relevant information and still place it in the wrong field or timepoint.

Several strategies have been used to improve clinical extraction. Large proprietary models can perform zero-shot extraction but still omit note-specific details and hallucinate [2]. Local open-weight models reduce dependence on external services, but performance varies with model size and prompt design [3,8]. Iterative prompt engineering has produced strong results on oncology pathology reports, although staging errors remain a recurring failure mode [10]. Fine-tuned hybrid systems can reach larger scale and external validation, but they require labeled data and model training [5,13]. Hierarchical prompting, validation layers, retry mechanisms, and clinician-informed prompt examples can also improve extraction without conventional fine-tuning [6,7,12,15].

Human involvement also differs across studies. Clinicians or medical experts often create gold-standard annotations, resolve disagreements, or guide terminology selection. That work establishes whether a model matches a reference label. It does not answer whether a practicing oncologist considers one complete system output more faithful, complete, and clinically useful than another when both contain partly correct information. In our audit of the supplemental methods table for the 24-study scoping review, most papers reported automatic performance metrics against labels. Only two entries explicitly described five-point Likert ratings of generated outputs, and both involved radiology reports [1]. More recent studies have used substantial clinical review, including six experts who categorized LLM extraction errors in head and neck oncology and senior-oncologist adjudication of metastasis and response extraction from imaging reports [11,14]. These studies provide stronger reference-based validation than simple automatic scoring. Their endpoints differ from our field-level comparison of two complete, same-model extraction systems.

Our approach treats recurring model errors as engineering targets. We use the term inference harness for the software layer that controls how a frozen model is prompted, checked, corrected, and linked to evidence. The model itself is unchanged. When review identifies a repeated failure, such as importing a stopped drug into current therapy or turning suspected metastasis into confirmed disease, the system receives a narrow correction through prompt instructions, verification logic, or a deterministic clinical rule. Each correction can be logged and regression-tested.

The harness is more than a long prompt or retrieval step. Different fields follow different extraction routes. Selected outputs pass into later tasks only when they provide relevant clinical context. Verification stages run after generation, and deterministic hooks enforce narrow oncology constraints across related fields. The system records these interventions and links final values to note evidence. Among the closest studies we reviewed, we did not identify an evaluation of this full combination on real longitudinal oncology notes with a same-model baseline and direct oncologist comparison.

The study follows this development sequence. We first used clinician feedback to build the breast-cancer harness. We then adapted the accumulated rubric and failure rules to pancreatic cancer through model-assisted review without target-domain clinician feedback. The primary experiment compares extraction from the final harness with a single-prompt baseline using the same Qwen2.5-32B model and target schema. A separate exploratory analysis asks whether the structured output can support patient letters more reliably than direct note-to-letter generation.

The extraction evaluation focuses on seven clinical questions that require more than surface entity recognition: active anticancer therapy, stage, distant metastasis, regional or overall metastatic involvement, treatment response, breast cancer type and receptor status, and completed molecular or genetic results.

We hypothesized that the harness would outperform the single-prompt baseline overall and that the largest gains would occur in fields directly addressed by temporal checks, clinical classification rules, and cross-field consistency checks. We also expected many ties because the same base model should answer straightforward questions similarly in both conditions.

## 2. Methods

### 2.1 Study design

The study followed three linked steps and one downstream exploration. First, a physician coauthor and the model-development investigators used human-in-the-loop review to develop a breast-cancer inference harness. Second, the breast-cancer error history and clinical distinctions were transferred into a model-assisted pancreatic-cancer adaptation loop without clinician review of pancreatic-cancer outputs. Third, the final harness was compared with a single-prompt baseline using the same frozen model and field schema on 40 expert-annotated CORAL notes. Three oncologists completed system-label-masked A/B evaluations. We then explored patient-letter generation using the structured extraction as an intermediate representation.

The breast-cancer stage examined whether clinician feedback could be converted into reusable prompts, gates, and rules. The pancreatic-cancer stage examined whether an AI-assisted review loop could reuse this knowledge without repeated target-domain clinician input. The extraction comparison isolated the complete inference harness while holding the base model and target fields constant. The letter analysis compared the structured route, source note plus extraction plus letter generation, with direct note-to-letter baselines.

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

***Figure 1.*** *Development and downstream-use pathway. Human-in-the-loop review shaped the breast-cancer inference harness. ChatGPT-assisted synthesis and a rubric-guided Qwen reviewer supported pancreatic-cancer adaptation without target-domain clinician review, while investigators controlled all retained changes. Oncologists evaluated the structured extraction, and an exploratory letter study compared the structured route with direct note-to-letter baselines.*

### 2.5 Clinician-informed breast-cancer development

Breast-cancer development used approximately 15 documented iterations across 56 additional notes. A physician coauthor familiar with oncology reviewed selected outputs with the model-development investigators. This physician was not an oncology specialist and did not participate in the final oncologist evaluation. The investigators compared flagged outputs with the complete source notes, grouped recurring errors, and translated them into general changes to prompts, verification logic, or deterministic rules. This process did not produce a conventional supervised training set, and the model weights were not updated.

Candidate changes were retained only after testing on affected examples and previously correct controls. The objective was to encode recurring clinical distinctions rather than case-specific corrections. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general regional-node rule. An error in which a planned drug was listed as active therapy motivated a temporal medication rule.

### 2.6 Model-assisted pancreatic-cancer adaptation

The breast-cancer harness was adapted to pancreatic cancer through approximately 18 documented development rounds covering all 100 additional pancreatic notes. No clinician reviewed pancreatic-cancer outputs during this stage. The team used ChatGPT to synthesize the accumulated breast-cancer error history and clinical distinctions into a transferable review rubric and candidate pancreatic-cancer adaptations. The Qwen pipeline generated structured fields, and a rubric-guided Qwen reviewer compared each output with the complete source note. Investigators assessed the resulting error flags and proposals, implemented selected changes, and retained them only after regression testing.

The reviewer prompt included the field definitions, severity criteria, and clinical distinctions established during breast-cancer development. Earlier error categories could therefore guide review in the new domain while the reviewer still detected pancreatic-specific problems such as regimen names and dose representation. This agent-assisted loop changed prompts, rules, and workflow code rather than model weights. Investigators controlled implementation, so we describe the process as model-assisted and investigator-supervised.

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

LLM-assisted review was used as a development instrument, particularly during adaptation to pancreatic cancer. For each candidate output, the reviewer model compared the extracted fields with the source note using the prespecified field definitions and severity criteria. It flagged possible omissions, unsupported claims, semantic mismatches, and temporal errors. Prior work shows that LLM judges can approximate human preferences but can also exhibit position, verbosity, and self-preference biases [16,17]. The review findings therefore informed candidate revisions that the investigators reviewed and regression-tested rather than serving as clinical outcome labels.

A matched audit of 260 applicable note-field comparisons was also used for technical error analysis. LLM judgments were not used as final clinical outcome labels and did not replace the independent oncologist evaluation.

### 2.9 Oncologist evaluation

The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were not told which output came from the harness, whether the systems used the same underlying model, or which condition was expected to perform better. The assignment was fixed across evaluations, with the harness displayed as A and the baseline as B. Source attribution appeared with the harness output because it is part of the complete harness output package. The study therefore evaluated the complete presented systems and did not isolate the effect of attribution.

The physician coauthor who participated in breast-cancer development was not one of the final evaluators. Three oncologists independently evaluated the breast-cancer outputs. Two of them also evaluated the pancreatic-cancer outputs; the third did not evaluate pancreatic cancer.

All three oncologists completed the same 280 required breast-cancer comparisons. We calculated pairwise exact agreement and Cohen's kappa for each pair. We also summarized all five completed clinician-by-cancer evaluations separately. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The primary inferential analysis excluded ties and modeled whether a directional judgment favored the harness. We used a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator was included as a fixed effect, and the overall model also included cancer type. This specification accounts for repeated judgments across fields and evaluators within a note. We report adjusted odds ratios, 95% confidence intervals, and two-sided p values. Exact sign tests on note-level harness-minus-baseline margins were used as a sensitivity analysis.

### 2.10 Exploratory downstream patient-letter generation

The harness-based letter route first generated structured extraction and then produced a patient letter from the cleaned keypoints together with source-note context, including the original Assessment and Plan section. The generator added field-level source tags, and later checks compared the letter with the note and removed known unsafe formatting or content patterns. The direct Qwen and GPT-4o baselines received the raw note through a single letter prompt without the structured intermediate representation.

One oncologist completed an exploratory review of 20 breast-cancer cases across the three letter conditions. The review covered accuracy, completeness, comprehensibility, usefulness, and hallucination-related concerns. This comparison was descriptive and was not the primary hypothesis test.

## 3. Results

### 3.1 Development path, cross-cancer adaptation, and downstream use

The development record contains approximately 15 breast-cancer iterations across 56 notes and approximately 18 pancreatic-cancer iterations across 100 notes. Breast-cancer revisions were informed by review from the physician coauthor and model-development investigators. Pancreatic-cancer revisions were made without clinician review of the pancreatic outputs, using ChatGPT-assisted rubric synthesis, Qwen-based error review, and investigator-controlled implementation.

This sequence produced two types of reuse. Some components were retained unchanged, including the five verification stages, temporal distinctions, source attribution, and rules that separate active treatment from plans or supportive medication. Other components required cancer-specific routing, especially disease terminology, regimen interpretation, and post-processing conditions. The resulting system therefore reused the error-handling framework without assuming that breast and pancreatic cancer were clinically interchangeable.

The pancreatic-cancer clinician result is the most relevant evidence for this adaptation. Across two independent evaluations, the oncologists preferred the harness in 161 field comparisons and the baseline in 23, with 335 ties. One oncologist recorded a positive within-note margin in 19 cases and a negative margin in one. The other recorded 18 positive margins and two ties. When their ratings were pooled within notes, all 20 pancreatic cases had a positive harness-minus-baseline margin. Because no physician reviewed pancreatic outputs during development, this pattern is consistent with reuse of previously codified evaluation criteria and workflow components in a second cancer domain. It does not identify whether the gain came from shared rules, pancreatic-specific revisions, the model-based reviewer, or their combination.

After extraction, the same structured fields could be passed to the patient-letter generator together with the source note. This created an explicit path from clinician-informed extraction to patient communication. The direct Qwen and GPT-4o comparators skipped the extraction stage and generated letters from the raw note in one prompt.

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

### 3.6 Exploratory patient-letter comparison

One oncologist reviewed 20 breast-cancer cases under three letter-generation conditions: harness-based Qwen letters generated from structured extraction plus source-note context, direct note-to-letter Qwen letters, and direct note-to-letter GPT-4o letters. Against GPT-4o, the harness-based route had 14 higher case-level means, two ties, and four lower means, with an average difference of +0.24 points. Against direct Qwen, it had nine higher means, five ties, and six lower means, with an average difference of -0.06 points. This descriptive result supports the feasibility of using extraction as a checked intermediate layer, while showing that better extraction does not by itself guarantee better patient-facing prose.

<div data-rough-figure="8"></div>

***Figure 8.*** *Exploratory per-case differences in the mean of accuracy, completeness, comprehensibility, and usefulness scores from one oncologist. Positive values favor the harness-based letter. The comparison is descriptive and was not designed as a formal superiority test.*

## 4. Discussion

### 4.1 Main interpretation

The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline. In the adjusted directional analysis, the odds ratio for a harness preference was 5.70 (95% CI 4.24 to 7.66; p<0.001).

The main result concerns the development path around the model. A clinician helped identify recurrent errors in breast cancer. The team converted those errors into an explicit inference harness. ChatGPT-assisted synthesis and a rubric-guided Qwen reviewer then supported adaptation to pancreatic cancer without target-domain clinician review, while investigators controlled each retained change. Final oncologist ratings favored the harness in both domains. The structured output also provided an intermediate layer for exploratory patient-letter generation.

Most field comparisons were ties, which is expected because both systems used the same base model. When an oncologist found a meaningful difference, the harness was preferred nearly six times as often as the baseline. The gains were concentrated in fields that need temporal interpretation or clinical classification. This is consistent with a system that leaves straightforward answers alone and intervenes when a known failure pattern appears.

The two main methodological limits are the absence of a component ablation and the ambiguity of ties, which can represent either two acceptable outputs or two inadequate outputs. The results therefore support preference for the complete presented harness, not absolute correctness or a causal claim for any single prompt, gate, rule, or attribution component.

### 4.2 Clinician-informed extraction as a reusable harness

The physician coauthor's role during breast-cancer development differed from conventional dataset labeling. Rather than producing a large supervised training set, the physician and model-development investigators reviewed concrete failures and defined clinically meaningful distinctions. Those distinctions were encoded as prompts, verification checks, and deterministic rules. The three independent oncologist evaluations suggest that the resulting behavior was not limited to the development physician's preferences.

Structured extraction is useful because it converts a long, internally repetitive note into a compact set of fields that can be checked, searched, and reused downstream. It is particularly valuable when the task is tedious for a clinician but can be bounded precisely, such as reconciling current anticancer therapy across a medication list and treatment history.

We use the term inference harness because the contribution sits around the model rather than in its weights. The routing, schemas, verification logic, deterministic rules, logging, and attribution layer could in principle be attached to another instruction-following model after interface and prompt calibration. This study tested only Qwen2.5-32B, so cross-model portability remains a design property to evaluate rather than an empirical result.

### 4.3 Agent-assisted cross-cancer adaptation and the role of LLM-as-a-judge

The pancreatic-cancer stage tested whether the accumulated workflow could be adapted beyond the domain in which the physician coauthor gave direct feedback. Pancreatic notes differ in disease course, treatment regimens, staging language, and surgical context. We retained shared clinical distinctions, added cancer-specific routing where needed, and did not ask a clinician to review pancreatic outputs during development.

Two model roles supported this adaptation. First, ChatGPT synthesized the accumulated breast-cancer error history and clinician-informed distinctions into a transferable rubric and candidate pancreatic-cancer adaptations. Second, a rubric-guided Qwen LLM-as-a-judge compared subsequent outputs with their source notes and localized likely omissions, unsupported claims, semantic mismatches, and temporal errors. The reviewer showed where the adapted harness was succeeding or failing across development rounds.

The judge did not define the clinical endpoint or directly modify the deployed workflow. Investigators selected candidate revisions and regression-tested them, while independent oncologists supplied the final evidence. General-purpose LLM evaluators have documented order and self-preference biases [16,17], so this separation matters. The pancreatic ratings, 161 harness preferences versus 23 baseline preferences with 335 ties, support the usefulness of the agent-assisted development loop. They do not establish autonomous self-improvement.

### 4.4 Model difficulty and human-model complementarity

The field-level results distinguish direct extraction from tasks that require temporal or clinical interpretation. Explicit findings often produced ties because both systems could locate them. The larger differences appeared when the same fact had to be classified in context.

Active anticancer medication showed the largest core-field advantage, with 83 harness preferences, two baseline preferences, and 15 ties. Medication planning also strongly favored the harness, 59 to zero with 41 ties. These tasks are laborious for a person reviewing a long treatment history, but they are well suited to structured assistance when current, historical, supportive, and planned therapies are kept separate.

Stage, metastatic involvement, and treatment response illustrate a different form of complementarity. A clinician may resolve an explicitly documented stage quickly, while a model can still confuse regional nodes with distant disease or treat a suspected lesion as confirmed. Response assessment can be difficult for both because it requires aligning findings with the current regimen. Passweg et al. similarly found that locally hosted LLMs were noninferior to human accuracy for metastasis classification from oncology imaging reports but remained inferior for treatment-response assessment [14]. In these fields, the system is better used to organize evidence and flag a proposed answer for rapid verification than to replace clinical judgment.

Breast type and receptor status had the smallest core-field margin, 18 to 12 with 30 ties. This is consistent with a field that is often stated explicitly, leaving less room for the harness to improve on the base model. The useful division of labor is therefore field-dependent: automate repetitive synthesis, surface evidence for judgment-sensitive fields, and allow clinicians to verify compact outputs rather than reread every note from scratch.

<div data-rough-figure="7"></div>

***Figure 7.*** *Conceptual map of human-model complementarity across oncology extraction tasks. Placement is qualitative rather than a measured difficulty score. The map distinguishes repetitive synthesis tasks suited to automation from judgment-sensitive fields that benefit from rapid clinician verification.*

### 4.5 Mechanistic evidence and the need for ablation

The observed field pattern is consistent with the intended function of the harness. The active-medication and medication-plan results align with dedicated prompts, temporal filtering, and drug-context rules. The stage and metastasis results align with cross-field checks and deterministic distinctions between regional and distant disease and between suspected and confirmed findings. These associations support the proposed mechanism but do not isolate the contribution of each component.

Component comparisons are present in related work but are not universal. Wiest et al. compared plain zero-shot, one-shot, definition-enhanced, and grammar-constrained prompting [3]. Huang et al. used iterative prompt engineering and error analysis for pathology extraction [10]. mCODEGPT compared single-step prompting with two hierarchical strategies [7]. Corso et al. compared zero-shot, few-shot, and clinician-annotated few-shot prompts across four locally deployable models [12]. Dubey et al. compared simple prompting, chain-of-thought prompting, double filtering, and a rule-based ECOG extractor [15]. Grothey et al. compared five prompting strategies and quantized model configurations [8]. Dao et al. quantified which initial errors were corrected by its validation and retry loop, while Tariq et al. compared a complete hybrid system with zero-shot, structured-code, and rule-based baselines [5,6]. No study in this set reports a factorial removal of every component in a multi-stage oncology harness.

Our study likewise lacks a complete component ablation. A feasible technical ablation would compare the single-prompt baseline, decomposed prompts alone, prompts plus verification gates, and the complete harness. Repeating the full oncologist evaluation for every variant would impose a disproportionate specialist burden. An LLM-reviewed ablation could help localize mechanisms, but it should be labeled as technical evidence and should not replace the independent oncologist comparison. A smaller clinician review restricted to outputs changed by the ablation would provide a stronger follow-up.

### 4.6 Relation to prior clinical extraction studies

CORAL established the real-note benchmark used here and showed that zero-shot GPT-4 outperformed GPT-3.5-turbo and FLAN-UL2, while an independent oncologist still identified omissions and hallucinations [2]. Our study asks a different question on the same benchmark: whether a structured workflow improves a frozen model relative to that same model under a single prompt.

The closest recent oncology studies show three complementary directions. First, prompt design and clinical expertise matter. Huang et al. used 78 lung-cancer pathology reports for prompt development and 774 valid reports for independent testing, followed by an additional test on 191 osteosarcoma reports [10]. Corso et al. found that few-shot examples and clinician expertise improved small-model extraction from Italian oncology records [12], while Dubey et al. reported that chain-of-thought and double-filtering prompts outperformed simple prompting for ECOG extraction [15]. These studies support structured prompting, but their outputs are narrower than the longitudinal field set evaluated here.

Second, other studies provide stronger evidence for scale or absolute correctness. Abhyankar et al. used 700 annotated notes for model development and testing, then applied the resulting staging system to more than two million notes across five cancers [13]. Van Koevorden et al. compared 29 extracted categories from 60 head and neck oncology patients against physician consensus, with six clinical experts assigning error types and impact [11]. Passweg et al. compared five local models with duplicate human extraction and senior-oncologist adjudication on 400 imaging reports [14]. These designs answer whether extraction agrees with a clinical reference. Our A/B study instead estimates clinician preference between a complete harness and a same-model single-prompt baseline, so it does not replace absolute accuracy assessment.

Third, prior systems support individual engineering components. Wiest et al. evaluated local models and constrained output for binary clinical features [3]. mCODEGPT tested hierarchical prompting on synthetic oncology notes [7]. Grothey et al. compared model, prompt, and quantization configurations in prostate pathology extraction [8]. Dao et al. combined engineered context, validation, and retries for numerical extraction from right-heart-catheterization reports [6]. Tariq et al. achieved larger scale and external validation with a UMLS-plus-fine-tuned-LLM system for breast treatment timelines [5], while Bhattarai et al. compared model families and rule-based methods on longitudinal lung-cancer phenotypes [4].

The distinguishing feature of our evaluation is not the invention of prompts, retries, or rules in isolation. It is the combination of field-specific routing, selective context transfer, verification gates, deterministic oncology rules, logging, and source attribution, evaluated against a same-model baseline through direct field-level comparison by oncologists. The breast-to-pancreatic development sequence also tests whether clinically informed failure rules can reduce repeated demand on specialist time in a second cancer domain.

The study should therefore be positioned as a focused, clinically reviewed evaluation of an integrated inference harness, not as the first oncology extraction system or the largest validation study. Its narrower contribution is the complete combination of field-specific routing, selective context transfer, verification gates, deterministic oncology rules, logging, and source attribution, evaluated against a same-model baseline through direct field-level comparison by oncologists. The detailed study-by-study comparison is retained in the Chinese review appendix rather than in the main manuscript.

### 4.7 Structured extraction as the bridge to patient communication

The letter comparison tested two routes to patient communication. The direct route gave the raw note to Qwen or GPT-4o through one prompt. The structured route first ran the note through the inference harness, then generated a letter from the resulting keypoints while retaining source-note context. This design makes the clinical facts visible before they are rewritten as prose.

The exploratory review found a modest descriptive advantage over GPT-4o, with an average four-dimension score difference of +0.24 points. The difference relative to direct same-model Qwen generation was -0.06 points. Letter quality still depends on content selection, organization, wording, uncertainty, and decisions about which details a patient needs.

The project order remains useful even though the initial letter comparison was mixed. Structured extraction is the measurable safety layer. Patient communication follows as a downstream task. The extraction study can identify whether stage, treatment, response, and plans are represented faithfully before a generator turns them into prose. Future letter work should test whether specific extraction gains survive that second transformation, ideally with patient readers as well as clinicians.

The letter experiment remains exploratory. It shows how the inference harness can support an end-to-end clinical use case and where a second generation step can lose advantages established during extraction.

### 4.8 Clinical and technical implications

The findings suggest that some LLM failures in oncology extraction are repeatable enough to address outside the model weights. This is useful when labeled training data are scarce or when a clinical team needs to change a field definition without retraining. A rule that preserves biopsy-pending disease as suspected can be inspected and tested. A prompt-only system offers less control because a wording change may affect unrelated fields.

The term inference harness is appropriate for this system because it includes more than a prompt sequence. It manages task routing, dependencies, verification, deterministic corrections, logging, and source attribution around a frozen model. The term clinical workflow could imply integration into routine care, which this pilot did not test. We therefore use inference harness for the technical contribution and evaluation workflow for the study procedure.

The system is also compatible with local deployment. Local operation does not by itself establish privacy compliance or clinical safety, but it allows an institution to retain control of note processing and system updates. The present work evaluates extraction quality, not readiness for autonomous clinical use.

### 4.9 Study boundaries and next steps

This pilot was intended to establish whether the observed effect is large and clinically coherent enough to justify a larger study. Three oncologists independently reproduced the breast-cancer direction, and two reproduced the pancreatic-cancer direction. The adjusted clustered analysis confirmed a strong preference among these ratings.

The first main limitation is that the study evaluates the complete harness and does not separate the effects of task decomposition, verification gates, deterministic rules, or source attribution. The second is that the preference scale combines two different meanings of a tie: both outputs may be acceptable, or both may be inadequate. These limitations restrict mechanism and absolute-correctness claims even though the direction of relative preference is consistent.

The broader scope also remains limited. Only three oncologists participated, CORAL represents one institution and two cancer domains, and another model family has not been tested. Technical review of the benchmark informed some later revisions, so the study is a benchmark-informed pilot rather than an untouched external validation. This does not change the narrower pancreatic-cancer observation: no clinician reviewed pancreatic outputs during development.

The next study should adjudicate a sample of ties, evaluate the staged ablation described above, add oncologists, and test a second model and an external health-system dataset. QUEST emphasizes advance planning, evaluator selection, multidimensional scoring, reliability, and adjudication in healthcare LLM evaluation [19]. The VALID framework further recommends variable-level comparison with expert reference data, internal consistency and plausibility checks, and replication analyses [20]. These frameworks make the remaining gap precise: our current endpoint measures comparative clinician preference, not absolute acceptability or field-level accuracy. The model version, prompts, evaluator roles, development use of the benchmark, and all human-oversight steps should also remain explicit, consistent with TRIPOD-LLM reporting guidance [18].

## 5. Conclusion

The current pilot supports a staged development strategy. Human-in-the-loop review converted breast-cancer extraction errors into an explicit inference harness. ChatGPT-assisted synthesis and a rubric-guided Qwen reviewer then supported pancreatic-cancer adaptation without target-domain clinician review, while investigators controlled workflow changes and the model weights remained frozen. Independent oncologists favored the harness in both cancers. The structured extraction also provided an inspectable intermediate layer for patient-letter generation, although the current letter evidence is exploratory. Component ablation, tie adjudication, more oncologists, and external validation are needed before making a broad confirmatory claim.

## References

*Draft note: The one-sentence relevance annotations are for collaborator review and should be removed before submission.*

1. Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. *JMIR Cancer*. 2025;11:e65984. [https://doi.org/10.2196/65984](https://doi.org/10.2196/65984). <span class="reference-note"><strong>Draft relevance:</strong> This review maps 24 oncology information-extraction studies and provides the broad literature frame used to limit our novelty claims.</span>
2. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024;1(4). [https://doi.org/10.1056/AIdbp2300110](https://doi.org/10.1056/AIdbp2300110). <span class="reference-note"><strong>Draft relevance:</strong> CORAL supplies our real longitudinal breast and pancreatic oncology notes and establishes the zero-shot benchmark that our harness extends.</span>
3. Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024;7:257. [https://doi.org/10.1038/s41746-024-01233-2](https://doi.org/10.1038/s41746-024-01233-2). <span class="reference-note"><strong>Draft relevance:</strong> This study shows that locally deployed open models, prompt design, and constrained output can support privacy-preserving clinical extraction.</span>
4. Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024;7(3):ooae060. [https://doi.org/10.1093/jamiaopen/ooae060](https://doi.org/10.1093/jamiaopen/ooae060). <span class="reference-note"><strong>Draft relevance:</strong> This longitudinal lung-cancer study compares model families and rule-based methods, while our experiment holds the model fixed and changes the inference workflow.</span>
5. Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. *JCO Clinical Cancer Informatics*. 2025;9:e2500002. [https://doi.org/10.1200/CCI-25-00002](https://doi.org/10.1200/CCI-25-00002). <span class="reference-note"><strong>Draft relevance:</strong> This large hybrid breast-cancer system has external validation and supervised fine-tuning, providing a stronger scale comparator but a different adaptation strategy.</span>
6. Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025;8(5):ooaf097. [https://doi.org/10.1093/jamiaopen/ooaf097](https://doi.org/10.1093/jamiaopen/ooaf097). <span class="reference-note"><strong>Draft relevance:</strong> Its engineered preload, schema and range validation, and retry loop are the closest published architectural precedent for our verification stages.</span>
7. Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025;5:422. [https://doi.org/10.1038/s43856-025-01116-x](https://doi.org/10.1038/s43856-025-01116-x). <span class="reference-note"><strong>Draft relevance:</strong> mCODEGPT shows that hierarchical prompting can outperform a single-step strategy, although it uses synthetic oncology notes and does not evaluate a full verification harness.</span>
8. Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025;5:96. [https://doi.org/10.1038/s43856-025-00808-8](https://doi.org/10.1038/s43856-025-00808-8). <span class="reference-note"><strong>Draft relevance:</strong> This prostate-pathology study documents substantial effects from model, prompt, language, and quantization choices on a narrower report type.</span>
9. Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025. [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115). <span class="reference-note"><strong>Draft relevance:</strong> This report documents the base model family used in both study conditions; the inference harness is our added system layer.</span>
10. Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. *npj Digital Medicine*. 2024;7:106. [https://doi.org/10.1038/s41746-024-01079-8](https://doi.org/10.1038/s41746-024-01079-8). <span class="reference-note"><strong>Draft relevance:</strong> Iterative prompt engineering achieved strong pathology extraction but still exposed specialized terminology and TNM-staging errors that motivate explicit clinical checks.</span>
11. van Koevorden JW, Aben N, Struben V, et al. Validating large language model-assisted data extraction from clinical notes. *ESMO Real World Data and Digital Oncology*. 2026;12:100718. [https://doi.org/10.1016/j.esmorw.2026.100718](https://doi.org/10.1016/j.esmorw.2026.100718). <span class="reference-note"><strong>Draft relevance:</strong> Six experts evaluated 29 extracted categories in head and neck oncology, making this a close clinical-validation comparator with stronger reference-based error classification than our preference study.</span>
12. Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. *Communications Medicine*. 2026. [https://doi.org/10.1038/s43856-026-01790-5](https://doi.org/10.1038/s43856-026-01790-5). <span class="reference-note"><strong>Draft relevance:</strong> This study directly shows that few-shot design and clinician expertise improve locally deployable models, while our work adds multi-stage verification and deterministic cross-field rules.</span>
13. Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. *JCO Clinical Cancer Informatics*. 2026;10:e2500388. [https://doi.org/10.1200/CCI-25-00388](https://doi.org/10.1200/CCI-25-00388). <span class="reference-note"><strong>Draft relevance:</strong> This study used 700 annotated notes for model development and testing, then processed more than two million notes across five cancers; our contribution instead concerns inference-time controls across a broader schema without weight updates.</span>
14. Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. *JCO Clinical Cancer Informatics*. 2026;10:e2600002. [https://doi.org/10.1200/CCI-26-00002](https://doi.org/10.1200/CCI-26-00002). <span class="reference-note"><strong>Draft relevance:</strong> Its oncologist-adjudicated comparison found metastasis classification easier than treatment-response assessment, matching the field-dependent difficulty seen in our results.</span>
15. Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. *JCO Clinical Cancer Informatics*. 2026;10:e2500226. [https://doi.org/10.1200/CCI-25-00226](https://doi.org/10.1200/CCI-25-00226). <span class="reference-note"><strong>Draft relevance:</strong> Its comparison of simple prompting, chain-of-thought, double filtering, and rules provides direct evidence that inference design changes oncology extraction quality even when the target is a single field.</span>
16. Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*. 2023;36:46595-46623. [https://doi.org/10.52202/075280-2020](https://doi.org/10.52202/075280-2020). <span class="reference-note"><strong>Draft relevance:</strong> This work supports LLM judges as scalable development tools but documents position, verbosity, self-enhancement, and reasoning biases that preclude treating them as our clinical endpoint.</span>
17. Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*. 2024:9440-9450. [https://doi.org/10.18653/v1/2024.acl-long.511](https://doi.org/10.18653/v1/2024.acl-long.511). <span class="reference-note"><strong>Draft relevance:</strong> This paper isolates positional bias in LLM comparison judgments and reinforces the need for human final evaluation and cautious interpretation of automated review.</span>
18. Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nature Medicine*. 2025;31:60-69. [https://doi.org/10.1038/s41591-024-03425-5](https://doi.org/10.1038/s41591-024-03425-5). <span class="reference-note"><strong>Draft relevance:</strong> TRIPOD-LLM supports explicit reporting of models, prompts, evaluation procedures, task-specific metrics, and human oversight in this pilot.</span>
19. Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. *npj Digital Medicine*. 2024;7:258. [https://doi.org/10.1038/s41746-024-01258-7](https://doi.org/10.1038/s41746-024-01258-7). <span class="reference-note"><strong>Draft relevance:</strong> Its review of 142 studies and QUEST framework directly inform evaluator selection, rating dimensions, reliability assessment, and adjudication for a stronger follow-up study.</span>
20. Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. *JCO Clinical Cancer Informatics*. 2026;10:e2500215. [https://doi.org/10.1200/CCI-25-00215](https://doi.org/10.1200/CCI-25-00215). <span class="reference-note"><strong>Draft relevance:</strong> VALID requires variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication, clarifying why our preference endpoint is not yet an absolute correctness assessment.</span>

<div class="language-break" id="chinese-version"></div>

# 从医生指导的 extraction 到 AI 辅助的跨癌种适配：面向肿瘤病历与患者沟通的 inference harness

**供临床合作者审阅的试点报告草稿**

版本 0.14，2026 年 9 月

作者：[TODO]

单位：[TODO]

目标会议及稿件形式：[TODO]

本稿现已纳入三位肿瘤科医生完成的评审。三位医生均评估了乳腺癌病例，其中两位还评估了胰腺癌病例。全文按医生建议改为一条连续主线：首先通过 human-in-the-loop 开发乳腺癌 inference harness；随后把乳腺癌阶段积累的错误经验和 clinical rubric 用于 PDAC，在没有 target-domain clinician 审阅输出的情况下完成 AI-assisted adaptation；最后把 structured extraction 作为中间层，用于 patient-letter generation，并与 direct note-to-letter baseline 比较。调整后的 GEE 分析已经完成，Figure 1 至 Figure 7 和 Supplementary Figure S1 已嵌入 HTML。方括号中的内容仅保留在中文版，作为后续协作修改提示。

供临床审阅的简要说明：两个系统使用相同的语言模型。single-prompt baseline 要求模型通过一次调用提取全部信息。inference harness 则把任务拆分为较小的临床问题，检查模型答案，针对反复出现的错误应用范围明确的肿瘤学规则，并把每项结果与病历中的支持性原文关联起来。

> **术语约定：** 中文审阅稿保留 inference harness、workflow、pipeline、baseline、verification gate、hook 和 LLM-as-a-judge 等英文术语，避免翻译后与代码和实验记录中的名称对不上。

## 供临床合作者审阅的问题

本轮审阅请重点关注以下问题。投稿时将删除本节。

1. `breast human-in-the-loop → PDAC agent-assisted adaptation → structured extraction → patient letter` 是否符合你希望向其他医生说明项目的方式？
2. 将这七个核心临床字段定义为具有临床重要性的字段是否合适？
3. 我们对当前抗癌治疗、分期、区域与远处转移、治疗反应、受体状态以及分子检测结果的解释是否符合临床实际？
4. Discussion 是否在不过度扩展研究结论的前提下，准确解释了观察到的优势和不足？
5. 哪些案例最能说服肿瘤学读者？
6. 是否存在不准确、夸大的临床表述，或缺少必要背景的内容？
7. 我们对临床医生建立金标准和临床医生评估最终输出这两种角色的区分，是否公平且具有临床意义？

## 摘要

### 背景

大语言模型可以从临床病历中提取结构化信息，但单一提示经常混淆当前治疗与既往治疗、疑似病变与确诊疾病，以及已经完成的检查与未来计划。这类错误不易发现，因为生成文本通常仍然流畅。肿瘤科病历尤其具有挑战性：一份纵向病历往往同时包含病理、影像、治疗史、当前治疗、疗效评估和条件性计划。Structured extraction 还可以在信息被改写为患者信件前，提供一个可检查的中间表示。

### 目的

评估 clinician-informed inference harness 能否在不进行微调的情况下改善肿瘤信息 extraction，能否在没有 target-domain clinician 审阅的情况下适配到第二个癌种，以及 structured extraction 能否支持下游 patient-letter generation。

### 方法

我们围绕 Qwen2.5-32B-Instruct-AWQ 分两个阶段构建 inference harness。乳腺癌阶段采用 human-in-the-loop：医生作者和模型开发作者反复审阅输出，把重复出现的临床错误转化为 field-specific prompt、verification gate 和 deterministic oncology rule。PDAC 阶段由 ChatGPT 帮助把乳腺癌错误记录整理成可迁移的 review rubric 和候选适配方案。Qwen pipeline 生成 PDAC structured fields，再由 rubric-guided Qwen reviewer 逐项对照源病历。所有最终修改都由研究人员选择、实施并做 regression test。该阶段没有医生审阅 PDAC 输出，模型权重始终冻结。之后，我们在 40 份 CORAL benchmark 病历上比较 inference harness 与同模型 single-prompt baseline。三位肿瘤科医生评估乳腺癌 extraction，其中两位也评估 PDAC extraction。另一个探索性分析使用 structured extraction 和源病历 context 生成 patient letter，并与 direct note-to-letter Qwen 和 GPT-4o baseline 做描述性比较。

### 结果

在已完成的匹配技术审查中，inference harness 在 66 项核心比较中更优，baseline 在 28 项中更优，另有 166 项为平局。五组已完成的评审者与癌种组合共提供 1,359 项必评字段判断，其中 443 项偏好 inference harness，77 项偏好 baseline，839 项为平局。平局占全部判断的 61.7%；在其余 520 项非平局判断中，85.2% 偏好 inference harness。调整后的方向性分析得到 OR 5.70，95% CI 为 4.24 至 7.66，p<0.001。在开发阶段没有医生参与的 PDAC 任务中，两位医生合计给予 inference harness 161 项偏好，baseline 23 项偏好，另有 335 项平局。探索性 letter review 中，structured route 相对 direct GPT-4o generation 的平均分差为 +0.24，相对同模型 direct Qwen generation 为 -0.06。

> **协作说明：** Abstract 会在全文定稿后最后压缩。当前统计数字已经更新，不再保留未完成分析的占位内容。

### 结论

结果支持一条分阶段开发路线。医生反馈先形成乳腺癌 inference harness。随后，AI-assisted 且由研究人员监督的循环把已有 rubric 和 failure rules 适配到 PDAC，期间没有 target-domain clinician 审阅。Structured extraction 还可以作为患者沟通前的可检查中间层，但目前的 letter 结果仍属于探索性证据。更广泛的结论需要更多肿瘤科医生、TIE adjudication、component ablation 和 external validation。

## 1. 引言

肿瘤临床工作中，大量有用信息仍记录在自由文本中。一份随访病历可能同时包含诊断、病理、受体状态、治疗史、毒性反应、疗效和后续计划，但这些事实分散在不同章节和时间点。人工审阅耗时较长，传统信息提取系统则需要大量标注和针对具体任务的开发。大语言模型可以读取多种病历写法，并针对不同问题返回结构化字段，无须为每个问题重新训练监督模型。

患者沟通是这些信息的一个自然下游用途。Direct note-to-letter generation 要求模型在一次调用中同时决定哪些事实重要、如何判断它们的时间和确定性，以及怎样安全地向患者解释。我们的路线先产生可检查的 structured extraction，再让 letter generator 同时使用 extraction 和源病历 context。这样既保留原始记录，也让生成信件前的临床事实更加明确。

相关研究的发展速度快于临床常规应用。2025 年的一项范围综述纳入了 24 项使用语言模型提取肿瘤学信息的研究，但也发现外部验证和真实 workflow 整合仍然有限 [1]。临床部署的要求高于基准数据上的概念验证。一个实用系统必须保留不确定性，区分当前诊疗与历史事件，避免无依据的事实，并让临床医生能够从病历原文追溯每项结果。

纵向肿瘤科病历会暴露一些在简单提取任务中不易发现的问题。药物清单可能同时包含抗癌治疗、支持治疗、长期居家用药、已停用药物，以及仅处于讨论阶段的治疗。区域淋巴结不能被标为远处转移。等待活检的可疑病灶不能被写成已确诊的 IV 期疾病。治疗前的肿瘤增长也不能用来判断刚刚开始的方案无效。模型即使提到了医学相关信息，仍可能把它放入错误的字段或时间点。

已有研究采用了多种方法改进临床信息提取。大型闭源模型可以进行 zero-shot extraction，但仍会遗漏病历特有细节并产生 hallucination [2]。本地开放权重模型减少了对外部服务的依赖，但性能会随模型规模和 prompt design 而变化 [3,8]。在肿瘤病理报告中，迭代式 prompt engineering 可以取得较高准确率，但 stage 解释仍是常见错误来源 [10]。经过 fine-tuning 的 hybrid system 可以达到更大规模并完成 external validation，但需要标注数据和模型训练 [5,13]。hierarchical prompting、validation layer、retry mechanism，以及由医生参与设计的 few-shot example，也可以在不采用传统 fine-tuning 的情况下改善 extraction [6,7,12,15]。

各项研究中人工参与的方式也不相同。临床医生或医学专家通常负责建立 gold-standard annotation、解决分歧或指导术语选择。这类工作可以判断模型是否匹配参考标签，但不能回答另一个问题：当两个完整系统的输出都包含部分正确信息时，执业肿瘤科医生是否认为其中一个系统更忠实、更完整，也更有临床用途。在我们对上述 24 项研究范围综述的补充方法表进行审查时，多数论文报告的是相对于标签的自动性能指标。只有两项研究明确使用五点 Likert 量表评估生成结果，而且均针对放射学报告 [1]。此后出现的研究已经加强了临床审查，例如由六位临床专家对头颈肿瘤 extraction error 分类，以及由高级肿瘤科医生裁决影像报告中的转移和疗效字段 [11,14]。这些研究提供了比自动评分更强的 absolute validation，但它们没有比较同一基础模型上的完整 harness 和 single-prompt baseline。

我们把反复出现的模型错误视为具体的工程目标。本文所称的 inference harness，是指控制冻结模型如何接受提示、检查答案、修正结果并关联证据的软件层。模型本身不发生变化。当审查发现重复性错误，例如把已停用药物列入当前治疗，或把疑似转移写成确诊疾病时，系统通过提示指令、验证逻辑或确定性临床规则加入范围明确的修正。每项修正都可以记录并接受 regression test。

该 inference harness 包含 field routing、selective context transfer、生成后验证和确定性临床约束，范围超过单一长提示或检索步骤。不同字段采用不同的提取路径。只有在确实提供相关临床背景时，部分输出才会传递给后续任务。系统记录这些干预，并将最终字段值与病历证据关联起来。在我们审阅的最接近本研究的工作中，尚未发现有研究在真实纵向肿瘤科病历上评估这一完整组合，同时采用同模型 baseline 和肿瘤科医生直接比较。

本研究沿着实际开发顺序展开。首先利用医生反馈建立乳腺癌 inference harness。随后，通过 model-assisted review，把已有 rubric 和 failure rules 适配到 PDAC，过程中没有 target-domain clinician 反馈。主要实验使用相同的 Qwen2.5-32B 模型和目标 schema，比较最终 inference harness 与 single-prompt baseline。另一个探索性分析检验 structured output 能否比 direct note-to-letter generation 更稳定地支持 patient letter。

Extraction 评估聚焦七个不能仅靠表层实体识别解决的临床问题：当前抗癌治疗、癌症分期、远处转移、区域或总体转移受累、治疗反应、乳腺癌类型与受体状态，以及已完成的分子或遗传检测结果。

我们的假设是，inference harness 整体上会优于 single-prompt baseline，且提升主要出现在由时态检查、临床分类规则和跨字段一致性检查直接处理的字段中。由于两种条件使用相同的基础模型，我们也预期简单问题会出现较多平局。

## 2. 方法

### 2.1 研究设计

本研究包括三个连续环节和一个下游探索。第一步，医生作者与模型开发作者通过 human-in-the-loop review 开发乳腺癌 inference harness。第二步，把乳腺癌错误记录和临床区分用于 model-assisted PDAC adaptation，期间没有临床医生审阅 PDAC 输出。第三步，在 40 份经专家标注的 CORAL 病历上，将最终 inference harness 与使用同一冻结模型和相同字段定义的 single-prompt baseline 进行比较，并由三位肿瘤科医生完成隐藏系统标签的 A/B 评估。最后，我们探索使用 structured extraction 作为中间层生成 patient letter。

乳腺癌阶段检验医生反馈能否转化为可复用的 prompt、gate 和 rule。PDAC 阶段检验 AI-assisted review loop 能否在没有持续 target-domain clinician 输入的情况下复用这些知识。Extraction comparison 在固定基础模型和目标字段的条件下评估完整 inference harness。Letter analysis 则比较 structured route，也就是 source note 加 extraction 再生成 letter，与 direct note-to-letter baseline。

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

***图 1.*** *开发与下游应用流程。乳腺癌阶段通过 human-in-the-loop review 形成 inference harness。PDAC 阶段由 ChatGPT-assisted synthesis 和 rubric-guided Qwen reviewer 支持适配，期间没有 target-domain clinician 审阅，所有保留的修改仍由研究人员控制。之后由肿瘤科医生评估 structured extraction，并在探索性 letter study 中比较 structured route 与 direct note-to-letter baseline。*

### 2.5 临床信息指导的乳腺癌开发

乳腺癌开发在 56 份附加病历上进行了约 15 轮有记录的迭代。一位熟悉肿瘤学问题的医生作者与模型开发作者共同审阅部分输出。这位医生不是肿瘤专科医生，也没有参与最终的肿瘤科医生评估。研究团队将发现的问题与完整源病历逐项核对，对反复出现的错误进行归类，并把它们转化为可推广的提示修改、验证逻辑或确定性规则。该过程没有建立传统的监督训练集，也没有更新模型权重。

候选修改只有在受影响样本和此前正确的对照样本上通过测试后才会保留。这里的目标不是修正单个病例，而是编码反复出现的临床区分。例如，腋窝淋巴结受累被误判为远处转移后，团队加入了通用的区域淋巴结规则；计划使用的药物被误列为当前治疗后，则加入了药物时态规则。

> **协作说明：** 2.1 保留为研究设计总览；2.5 单独描述乳腺癌阶段的具体开发程序。两节不是同一层级的信息，因此不建议合并。

### 2.6 AI-assisted 的胰腺癌适配

随后，我们通过约 18 轮有记录的开发，将乳腺癌 inference harness 适配到全部 100 份 PDAC 附加病历。在这一阶段，没有临床医生审阅 PDAC 输出。团队先用 ChatGPT 汇总乳腺癌阶段累积的错误记录和临床区分，形成可迁移的 review rubric 和候选 PDAC 适配方案。Qwen pipeline 生成 structured fields，再由 rubric-guided Qwen reviewer 将每项输出与完整源病历比较。研究人员审查 error flag 和候选方案，只实施有依据的修改，并在 regression test 通过后保留。

审查提示包含字段定义、严重程度标准，以及乳腺癌开发期间确定的临床区分。已有错误类别因此可以指导新癌种的审查，reviewer 也能发现 PDAC 特有的问题，例如治疗方案名称和剂量表达。这个 agent-assisted loop 修改的是 prompt、rule 和 workflow code，模型权重没有变化。研究人员控制具体实施，所以本文将其称为 model-assisted、investigator-supervised adaptation。

> **协作说明：** 当前按医生口述将外部通用 LLM 写为 ChatGPT。正式投稿前需要补齐当时使用的具体 model/version 和访问日期。

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

LLM 辅助审查是开发阶段使用的工具，尤其服务于胰腺癌适配。针对每个候选输出，审查模型依据预先设定的字段定义和严重程度标准，将提取结果与源病历进行比较，并标记可能的遗漏、无依据内容、语义错配和时态错误。已有研究表明，LLM judge 可以在一定程度上接近人工偏好，但也会受到 position、verbosity 和 self-preference bias 的影响 [16,17]。因此，这些发现只用于形成候选修改，最终是否实施仍由研究人员判断，并通过 regression test 确认，不能作为临床结局标签。

研究还用同一流程完成了 260 项适用病历字段比较的匹配审查，用于技术错误分析。LLM 的判断不作为最终临床结局标签，也不能替代独立的肿瘤科医生评估。

> **协作说明：** 这里不再罗列“发现了哪些错误、之后修了什么”。这些内容更像开发报告。Methods 只说明 LLM 审查的用途、边界和它与最终医生评估的区别。

### 2.9 肿瘤科医生评估

临床评估通过隐藏系统标签的 A/B 界面展示源病历和两份结构化输出。评估者针对每个字段选择 A 更优、B 更优或平局。他们不知道哪份输出来自 inference harness，也不知道两个系统是否使用同一基础模型或研究预期哪一侧更好。所有评估采用固定映射，inference harness 显示为 A，single-prompt baseline 显示为 B。source attribution 作为完整 harness 输出的一部分随 A 展示。因此，本研究比较的是呈现给医生的完整系统输出，没有单独隔离 attribution 的作用。

参与乳腺癌开发的医生作者不属于最终评估者。三位肿瘤科医生独立评估了乳腺癌输出，其中两位同时评估了胰腺癌输出，第三位没有评估胰腺癌。

三位肿瘤科医生均完成了同样的 280 项乳腺癌必评比较。我们计算了每两位医生之间的完全一致率和 Cohen's kappa，并分别汇总五组已完成的评审者与癌种组合。合并计数仅作描述性统计，不把各字段判断视为相互独立的观测。

主要推断分析排除平局，将非平局判断是否偏好 inference harness 作为二分类结局。我们使用总体平均的 logistic 广义估计方程，在每份病历内采用可交换相关结构。模型将评审者作为固定效应，总体模型同时校正癌种，从而处理同一病历中跨字段、跨评审者的重复判断。结果报告调整后比值比、95% 置信区间和双侧 p 值。敏感性分析将每组字段评分汇总为病历级的 harness 减 baseline 净差，并使用精确符号检验。

### 2.10 探索性下游 patient-letter generation

Harness-based letter route 先生成 structured extraction，再使用清理后的 keypoints 和源病历 context 生成 patient letter，其中包括原始 Assessment and Plan。Generator 为句子加入 field-level source tag，后续检查再将信件与病历核对，并处理已知的不安全格式或内容模式。Direct Qwen 和 GPT-4o baseline 则把 raw note 直接交给单一 letter prompt，不经过 structured intermediate representation。

一位肿瘤科医生对 20 份乳腺癌病例的三种 letter condition 进行了探索性评估。评估包括 accuracy、completeness、comprehensibility、usefulness 和 hallucination-related concern。该比较仅作描述，不属于主要假设检验。

## 3. 结果

### 3.1 开发路径、跨癌种适配与下游应用

开发记录包括对 56 份乳腺癌病历进行的约 15 轮迭代，以及对 100 份 PDAC 病历进行的约 18 轮迭代。乳腺癌部分的修订来自医生作者与模型开发作者的共同审阅。PDAC 部分没有临床医生审查相应输出，采用 ChatGPT-assisted rubric synthesis、Qwen-based error review，以及由研究人员控制的实施流程。

开发中有一部分组件可以原样迁移，包括五个验证阶段、时态区分、source attribution，以及区分当前治疗、治疗计划和支持性用药的规则。疾病术语、治疗方案解读和后处理条件则需要按癌种分别处理。因此，该系统复用了 failure-handling workflow，但没有假设乳腺癌与胰腺癌在临床上可以互换。

胰腺癌的临床医生评估是检验这种适配的主要证据。在两次独立评估中，医生有 161 次偏好 inference harness、23 次偏好 baseline，另有 335 次平局。第一位医生按病历计算得到 19 例正向净差和 1 例负向净差，第二位医生得到 18 例正向净差和 2 例平局。合并两位医生的判断后，20 例胰腺癌病例的 harness 减 baseline 净差均为正值。由于开发期间没有医生审查胰腺癌输出，这一模式说明先前编码的评估标准和 workflow 可以跨癌种复用。但仅凭该结果，无法判断收益来自共享规则、胰腺癌特异性修订、基于模型的审查器，还是这些因素的共同作用。

Extraction 完成后，同一组 structured fields 可以与源病历一起传给 patient-letter generator。这样形成从 clinician-informed extraction 到患者沟通的明确路径。Direct Qwen 和 GPT-4o comparator 跳过 extraction stage，直接通过一个 prompt 从 raw note 生成 letter。

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

### 3.6 探索性 patient-letter comparison

一位肿瘤科医生审阅了 20 份乳腺癌病例的三种 letter-generation condition：使用 structured extraction 加源病历 context 生成的 harness-based Qwen letter、direct note-to-letter Qwen letter，以及 direct note-to-letter GPT-4o letter。相对 GPT-4o，harness-based route 有 14 例平均分更高、2 例相同、4 例更低，整体平均分差为 +0.24。相对 direct Qwen，则有 9 例更高、5 例相同、6 例更低，整体平均分差为 -0.06。这个描述性结果支持把 extraction 用作可检查中间层的可行性，也说明 extraction 做得更好，并不会自动保证 patient-facing prose 更好。

<div data-rough-figure="8"></div>

***图 8.*** *一位肿瘤科医生对每个病例的 accuracy、completeness、comprehensibility 和 usefulness 平均分差。正值表示 harness-based letter 得分更高。该比较仅作描述，不属于正式 superiority test。*

## 4. 讨论

### 4.1 主要解读

多位肿瘤科医生参与的评估显示，相较于使用同一模型的 single-prompt baseline，临床医生显著更偏好 inference harness。调整后的方向性分析中，harness 偏好的 OR 为 5.70，95% CI 为 4.24 至 7.66，p<0.001。

本文的主要结果来自模型外围的开发路径。医生先帮助识别乳腺癌中反复出现的错误，团队再把这些错误转成明确的 inference harness。之后，ChatGPT-assisted synthesis 和 rubric-guided Qwen reviewer 支持 PDAC adaptation，期间没有 target-domain clinician 审阅，所有最终修改仍由研究人员控制。两个癌种的最终医生评分都更偏向 inference harness。Structured output 还成为探索性 patient-letter generation 的中间层。

多数字段比较为平局，这是可以预期的，因为两个系统使用同一个能力较强的基础模型。不过，当肿瘤科医生认为两者存在实质差异时，偏好 inference harness 的次数接近偏好 baseline 的六倍。优势主要集中在需要时间关系判断或临床分类的字段。这符合系统的设计思路：保留基础模型已经答对的简单问题，只在出现已知失败模式时介入。

目前最主要的方法学限制有两个。第一，研究没有 component ablation，无法分别估计 task decomposition、verification gate、deterministic rule 或 source attribution 的贡献。第二，`TIE` 可能表示两个输出都可以接受，也可能表示两个输出都不理想。因此，现有结果支持医生对完整 harness 输出的相对偏好，但不能直接证明绝对正确率，也不能把效果归因于某一个组件。

### 4.2 由临床信息指导、可复用的 inference harness

乳腺癌开发阶段中，医生作者的角色不同于常规的数据集标注。他没有为监督训练制作大量字段标签，而是与模型开发作者一起查看具体错误，并界定有临床意义的区别。团队再把这些区别编码成 prompt、verification check 和 deterministic rule。三位独立肿瘤科医生的最终评估说明，形成的系统行为并不只是在复现开发医生个人的偏好。

结构化 extraction 的价值在于，它可以把一份很长、内部重复较多的病历转换成一组紧凑字段，方便检查、检索和下游使用。当任务对医生而言费时，但边界可以清楚定义时，这种辅助尤其有价值。例如，医生需要在药物清单和治疗史中反复核对当前抗癌治疗，而 harness 可以先完成这部分整理。

我们使用 inference harness 这个名称，是因为贡献位于模型权重之外。routing、schema、verification logic、deterministic rule、logging 和 attribution 都是围绕冻结模型工作的。因此，经过接口和 prompt 校准后，同一套架构原则上可以包在另一个 instruction model 外面。本研究只测试了 Qwen2.5-32B，所以跨模型迁移目前是架构上的可行性，而不是已经验证的结果。

### 4.3 Agent-assisted 跨癌种适配与 LLM-as-a-judge

胰腺癌阶段检验的是，已经形成的 workflow 能否适配到医生作者没有直接提供反馈的领域。胰腺癌病历在疾病进程、治疗方案、分期表述和手术背景方面均有不同。我们保留共同的临床区分，在需要时增加癌种专用 routing，并且没有让临床医生参与胰腺癌输出的开发审阅。

这次适配中，AI 承担了两个角色。首先，ChatGPT 把乳腺癌阶段累积的错误记录和 clinician-informed distinction 整理成可迁移的 rubric，并提出候选 PDAC 适配方案。随后，rubric-guided Qwen LLM-as-a-judge 将每轮输出与源病历比较，定位可能的遗漏、无依据内容、semantic mismatch 和 temporal error。这个 reviewer 让我们能看到适配后的 harness 在哪里表现稳定、在哪里仍然失败。

LLM-as-a-judge 不定义最终临床结局，也不直接修改部署中的 workflow。研究人员选择候选修改并运行 regression test，最终证据来自独立肿瘤科医生。通用 LLM evaluator 已被发现存在顺序和 self-preference bias，因此这里的角色边界很重要 [16,17]。PDAC 评估中，harness 获偏好 161 次，baseline 获偏好 23 次，另有 335 次平局。这支持 agent-assisted development loop 的实用性，但不能证明 autonomous self-improvement。

### 4.4 模型难点与 human-model complementarity

字段级结果可以区分直接 extraction 与需要时间关系或临床判断的任务。明确写出的事实往往得到平局，因为两个系统都能找到；更大的差异出现在模型需要判断这些事实在当前语境中意味着什么的时候。

当前抗癌药物是优势最大的核心字段，结果为 83 次偏好 harness、2 次偏好 baseline 和 15 次平局。用药计划也以 59 比 0 明显偏向 harness，另有 41 次平局。这类任务需要在较长的治疗史中区分当前、既往、支持性和计划中的药物，对人工审阅而言费时，但适合由结构化系统先行整理。

分期、转移累及和治疗反应体现了另一种互补关系。病历如果直接给出 stage，医生通常可以很快确认；模型却可能把区域淋巴结当成远处转移，或把疑似病灶当作确诊。治疗反应对双方都更难，因为必须把当前方案与影像、症状和肿瘤标志物的时间线对应起来。Passweg 等在 oncology imaging report 上也发现，本地 LLM 判断 metastasis status 的准确率不劣于人工，但 treatment response 仍明显低于人工 [14]。对于这些字段，系统更适合整理证据并给出候选答案，再由医生快速复核，而不是替代临床判断。

乳腺癌类型与受体状态的优势最小，结果为 18 比 12，另有 30 次平局。这与该字段经常被直接写在病历中相符，因此 baseline 已经能够完成很多样本。更合理的人机分工应按字段决定：自动化重复而耗时的综合任务，对判断敏感的字段展示证据，并让医生复核紧凑输出，而不是重新通读整份病历。

<div data-rough-figure="7"></div>

***图 7.*** *肿瘤信息 extraction 中 human-model complementarity 的概念图。图中位置是定性解释，不是实测难度分数。它区分了适合自动化的重复综合任务，以及更适合由模型整理证据、医生快速复核的判断敏感字段。*

> **协作说明：** 这张图表达的是你说的“医生很容易、模型反而容易错”和“模型可以替医生处理冗长信息”这两类互补关系。它是 conceptual figure，不会伪装成我们实际测量过医生工作量。

### 4.5 作用机制与 ablation 的必要性

现有字段结果与 inference harness 的设计目标一致。当前用药和用药计划的结果对应专用 prompt、temporal filtering 和 drug-context rule；分期和转移结果对应 cross-field check，以及区分区域与远处疾病、疑似与确诊发现的 deterministic rule。这些关联支持我们提出的机制，但不能单独证明每个组件贡献了多少。

我们核对的相关论文并不是全部都有完整 ablation。Wiest 等比较了 plain zero-shot、one-shot、加入定义和 grammar-constrained prompting [3]；Huang 等采用迭代式 prompt engineering 和 error analysis [10]；mCODEGPT 比较 single-step prompting 与两种 hierarchical prompting [7]；Corso 等比较 zero-shot、few-shot 和由医生提供 annotation 的 few-shot prompt [12]；Dubey 等比较 simple prompt、chain-of-thought、double filtering 和 rule-based ECOG extractor [15]；Grothey 等比较五种 prompting strategy 和 quantized model configuration [8]。Dao 等报告 validation and retry loop 修正了多少初始错误，Tariq 等比较完整 hybrid system 与 zero-shot、structured-code 和 rule-based baseline [5,6]。这些工作都没有对一个多阶段 oncology harness 的全部组件进行 factorial removal。

我们的研究同样缺少完整 component ablation。一个成本可控的技术实验可以依次比较 single-prompt baseline、仅 task decomposition、decomposed prompts 加 verification gates，以及 full harness。要求肿瘤科医生对每个版本重复评分并不现实。LLM-reviewed ablation 可以帮助定位机制，但只能作为 technical evidence，不能代替独立医生评估。更强但仍可行的方案，是只让医生复核不同 ablation 版本之间真正发生变化的少量输出。

### 4.6 与既有临床 extraction 研究的关系

CORAL 建立了本研究使用的真实病历 benchmark。原研究比较了 zero-shot GPT-4、GPT-3.5-turbo 和 FLAN-UL2，并由一位独立肿瘤科医生审阅部分 GPT-4 输出，仍发现遗漏和幻觉 [2]。我们的研究在同一 benchmark 上回答另一个问题：当基础模型保持不变时，结构化 workflow 是否优于 single-prompt baseline。

最近的 oncology extraction 研究大致形成了三个方向。第一类研究检验 prompt design 和临床知识。Huang 等用 78 份肺癌病理报告开发 prompt，在 774 份有效报告上做 independent testing，之后又在 191 份骨肉瘤报告上测试 [10]。Corso 等发现 few-shot example 和 clinician expertise 可以改善本地 small model 对意大利语肿瘤病历的 extraction [12]。Dubey 等在 ECOG extraction 中发现 chain-of-thought 和 double filtering 优于 simple prompt [15]。这些结果支持 structured prompting，但它们的目标字段比本研究的纵向 schema 更窄。

第二类研究更重视规模或 absolute correctness。Abhyankar 等用 700 份标注病历完成 model development 和 testing，之后把 staging system 应用于五个癌种、超过两百万份病历 [13]。Van Koevorden 等在 60 位头颈肿瘤患者的 29 类字段上比较 LLM 和医生 consensus，并由六位临床专家判断 error type 与 impact [11]。Passweg 等在 400 份 oncology imaging report 上比较五个本地模型，ground truth 来自双人提取和高级肿瘤科医生 adjudication [14]。这些设计回答 extraction 是否符合临床 reference。我们的 A/B 研究回答的是完整 harness 相对于同模型 single-prompt baseline 是否更受医生偏好，因此不能代替 absolute accuracy evaluation。

第三类研究支持不同的工程组件。Wiest 等说明 prompt design 和 constrained output 可以改善本地模型对临床二分类字段的 extraction [3]。mCODEGPT 在合成肿瘤病历上比较 hierarchical prompting 与 single-step strategy [7]。Grothey 等展示了不同 model、prompt 和 quantization configuration 在前列腺病理 extraction 中的差异 [8]。Dao 等结合 engineered context、validation 和 retry，但任务是右心导管报告中的数值 extraction [6]。Tariq 等使用 UMLS 加 fine-tuned LLM，在乳腺癌治疗时间线上获得更大规模和 external validation [5]；Bhattarai 等则在纵向肺癌表型任务上比较多种模型和 rule-based approach [4]。

本研究的区别不是单独发明 prompt、retry 或 rule，而是把 field-specific routing、selective context transfer、verification gates、deterministic oncology rules、logging 和 source attribution 组合为一个 inference harness，再用 same-model baseline 隔离外围 workflow 的作用，并由肿瘤科医生逐字段直接比较。乳腺癌到胰腺癌的开发顺序还检验了临床信息形成的 failure rules 能否在第二个癌种中减少对稀缺专科医生时间的反复占用。

因此，更准确的定位是：这是一项对整合式 inference harness 的、小规模但由肿瘤科医生直接评估的研究，而不是第一个肿瘤 extraction 系统，也不是规模最大的验证。更具体的贡献是把 field-specific routing、selective context transfer、verification gate、deterministic oncology rule、logging 和 source attribution 组合起来，再用 same-model baseline 和医生逐字段比较评估完整系统。逐篇对照表保留在文末中文审阅附录，正文只放与论点直接相关的引用。

### 4.7 Structured extraction 连接患者沟通

Letter comparison 检验了两条患者沟通路线。Direct route 把 raw note 通过一个 prompt 交给 Qwen 或 GPT-4o。Structured route 先运行 inference harness，再用 structured keypoints 和源病历 context 生成 letter。这样，临床事实可以在被改写为 prose 前单独检查。

探索性评估显示，harness-based route 相对 GPT-4o 的四维平均分差为 +0.24，但相对 direct same-model Qwen generation 的平均分差为 -0.06。Letter quality 仍取决于内容选择、组织、措辞、不确定性表达，以及患者真正需要哪些临床细节。

即使初次 letter comparison 的结果并不完全一致，这个项目顺序仍然合理。Structured extraction 是可测量的 safety layer，患者沟通是它的 downstream task。Extraction 研究可以先判断分期、治疗、疗效和计划是否得到忠实表达，再由 generator 将这些内容写成连贯文字。未来的信件研究应检验具体 extraction 改进能否经过第二次转换后继续保留，最好同时邀请患者读者和临床医生参与评估。

Letter experiment 仍属于 exploratory evidence。它说明 inference harness 如何连接到完整临床应用，也显示第二次文本生成可能丢失 extraction 阶段已经建立的优势。

### 4.8 临床与技术意义

研究结果表明，肿瘤学 extraction 中的部分 LLM 错误具有足够的重复性，可以在不修改模型权重的情况下处理。当标注训练数据有限，或临床团队需要在不重新训练模型的情况下调整字段定义时，这一点很有价值。例如，将待活检病灶保留为疑似病变的 rule 可以被直接检查和测试。只依赖 prompt 的系统较难提供同等控制，因为一次措辞修改可能影响无关字段。

我们使用 inference harness 这一名称，是因为系统除 prompt sequence 外，还围绕冻结模型管理 task routing、dependency、verification、deterministic correction、logging 和 source attribution。Clinical workflow 也能描述该系统，但可能让人误以为它已经被整合进常规诊疗，而本 pilot 并未检验这一点。因此，本文用 inference harness 描述技术贡献，用 evaluation workflow 描述研究程序。

该系统也支持本地部署。本地运行本身不能证明系统符合隐私要求或具备临床安全性，但能让机构掌控病历处理和系统更新。本研究评估的是提取质量，而不是自主临床使用的准备程度。

### 4.9 研究边界与 next steps

本 pilot 用于判断观察到的 effect 和临床一致性是否足以支持扩大研究规模。三位肿瘤科医生独立复现了乳腺癌评估的总体方向，两位医生也复现了胰腺癌方向。调整后的 clustered analysis 已经确认这些评分中的偏好很强。

最主要的两个 limitation 是缺少 component ablation，以及 `TIE` 的含义不唯一。前者使我们无法判断收益具体来自 task decomposition、verification gate、deterministic rule 还是 source attribution；后者无法区分“两个输出都对”和“两个输出都错”。这两点限制了机制解释和绝对正确性结论，但不改变现有相对偏好的方向。

研究范围仍然有限。最终临床结果来自三位肿瘤科医生，CORAL 来自单一机构、只覆盖两个癌种，而且我们尚未测试另一个 model family。benchmark 的技术审查影响过部分后续修改，所以本研究属于 benchmark-informed pilot，而不是 untouched external validation。不过，胰腺癌开发期间确实没有临床医生审阅 PDAC 输出，这一较窄的观察不受影响。

下一项研究应优先人工 adjudicate 一部分 `TIE`，完成前述 staged ablation，再增加肿瘤科医生，并在第二种模型和外部医疗系统数据上复现。QUEST 强调 healthcare LLM evaluation 应提前规划 evaluator selection、multidimensional scoring、reliability 和 adjudication [19]。VALID framework 进一步要求逐变量对照 expert reference，并检查 internal consistency、plausibility 和 replication [20]。这两套 framework 把当前缺口说得更准确：我们的 endpoint 衡量医生的相对偏好，还不是 absolute acceptability 或 field-level accuracy。模型版本、prompt、评审者角色、benchmark 在开发中的使用方式和全部 human-oversight step 也应继续明确报告，这与 TRIPOD-LLM 的透明性要求一致 [18]。

## 5. 结论

[最终结论：在一项由多位肿瘤科医生参与、隐藏系统标签的评估中，由失败模式驱动的 inference harness 相较于使用同一模型的 single-prompt baseline 获得了显著的临床医生偏好。]

当前 pilot 支持一条分阶段开发路线。Human-in-the-loop review 把乳腺癌 extraction error 转化为明确的 inference harness。ChatGPT-assisted synthesis 和 rubric-guided Qwen reviewer 随后支持 PDAC adaptation，期间没有 target-domain clinician 审阅，workflow 修改仍由研究人员控制，模型权重始终冻结。独立肿瘤科医生在两个癌种中都更偏好 inference harness。Structured extraction 也为 patient-letter generation 提供了可检查的中间层，但现有 letter 证据仍属于探索性结果。在提出广泛的确证性结论前，仍需完成 component ablation、区分不同类型的 `TIE`，并增加肿瘤科医生和 external validation。

## 参考文献

*草稿说明：每篇文献后的一句话简评用于合作者审阅，正式投稿前删除。*

1. Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. *JMIR Cancer*. 2025;11:e65984. [https://doi.org/10.2196/65984](https://doi.org/10.2196/65984). <span class="reference-note"><strong>草稿简评：</strong>该综述梳理了 24 项肿瘤信息 extraction 研究，是本文限定 novelty claim 和描述研究版图的主要依据。</span>
2. Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024;1(4). [https://doi.org/10.1056/AIdbp2300110](https://doi.org/10.1056/AIdbp2300110). <span class="reference-note"><strong>草稿简评：</strong>CORAL 提供本文使用的真实乳腺癌和胰腺癌纵向病历，也建立了本文进一步改进的 zero-shot benchmark。</span>
3. Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024;7:257. [https://doi.org/10.1038/s41746-024-01233-2](https://doi.org/10.1038/s41746-024-01233-2). <span class="reference-note"><strong>草稿简评：</strong>该研究说明本地开放模型、prompt design 和 constrained output 可以支持 privacy-preserving clinical extraction。</span>
4. Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024;7(3):ooae060. [https://doi.org/10.1093/jamiaopen/ooae060](https://doi.org/10.1093/jamiaopen/ooae060). <span class="reference-note"><strong>草稿简评：</strong>该纵向肺癌研究比较多个 model family 和 rule-based method，而本文固定基础模型，只改变 inference workflow。</span>
5. Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. *JCO Clinical Cancer Informatics*. 2025;9:e2500002. [https://doi.org/10.1200/CCI-25-00002](https://doi.org/10.1200/CCI-25-00002). <span class="reference-note"><strong>草稿简评：</strong>该 breast-cancer hybrid system 有更大的规模和 external validation，但依赖 supervised fine-tuning，适配路线与本文不同。</span>
6. Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025;8(5):ooaf097. [https://doi.org/10.1093/jamiaopen/ooaf097](https://doi.org/10.1093/jamiaopen/ooaf097). <span class="reference-note"><strong>草稿简评：</strong>其 engineered preload、schema and range validation 和 retry loop 是本文 verification stages 最接近的已发表架构先例。</span>
7. Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025;5:422. [https://doi.org/10.1038/s43856-025-01116-x](https://doi.org/10.1038/s43856-025-01116-x). <span class="reference-note"><strong>草稿简评：</strong>mCODEGPT 证明 hierarchical prompting 可以超过 single-step strategy，但使用 synthetic oncology notes，也没有评估完整 verification harness。</span>
8. Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025;5:96. [https://doi.org/10.1038/s43856-025-00808-8](https://doi.org/10.1038/s43856-025-00808-8). <span class="reference-note"><strong>草稿简评：</strong>该前列腺病理研究展示了 model、prompt、language 和 quantization choice 对较窄报告类型 extraction 的影响。</span>
9. Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025. [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115). <span class="reference-note"><strong>草稿简评：</strong>该报告说明两种实验条件共同使用的 base model family，本文的 inference harness 是模型外部增加的系统层。</span>
10. Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. *npj Digital Medicine*. 2024;7:106. [https://doi.org/10.1038/s41746-024-01079-8](https://doi.org/10.1038/s41746-024-01079-8). <span class="reference-note"><strong>草稿简评：</strong>迭代式 prompt engineering 在病理 extraction 中取得较高准确率，但专业术语和 TNM stage 错误仍说明显式临床检查有必要。</span>
11. van Koevorden JW, Aben N, Struben V, et al. Validating large language model-assisted data extraction from clinical notes. *ESMO Real World Data and Digital Oncology*. 2026;12:100718. [https://doi.org/10.1016/j.esmorw.2026.100718](https://doi.org/10.1016/j.esmorw.2026.100718). <span class="reference-note"><strong>草稿简评：</strong>六位专家评估头颈肿瘤的 29 类 extraction，是非常接近的临床验证工作，并具有本文 A/B preference study 所缺少的 reference-based error classification。</span>
12. Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. *Communications Medicine*. 2026. [https://doi.org/10.1038/s43856-026-01790-5](https://doi.org/10.1038/s43856-026-01790-5). <span class="reference-note"><strong>草稿简评：</strong>该研究直接证明 few-shot design 和 clinician expertise 能改善本地 small model，本文进一步加入多阶段验证和 deterministic cross-field rule。</span>
13. Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. *JCO Clinical Cancer Informatics*. 2026;10:e2500388. [https://doi.org/10.1200/CCI-25-00388](https://doi.org/10.1200/CCI-25-00388). <span class="reference-note"><strong>草稿简评：</strong>该研究用 700 份标注病历完成 model development 和 testing，之后处理五个癌种、超过两百万份病历；本文研究的是不修改权重的多字段 inference control。</span>
14. Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. *JCO Clinical Cancer Informatics*. 2026;10:e2600002. [https://doi.org/10.1200/CCI-26-00002](https://doi.org/10.1200/CCI-26-00002). <span class="reference-note"><strong>草稿简评：</strong>该 oncologist-adjudicated study 发现 metastasis classification 比 treatment-response assessment 更容易，与本文观察到的字段难度差异一致。</span>
15. Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. *JCO Clinical Cancer Informatics*. 2026;10:e2500226. [https://doi.org/10.1200/CCI-25-00226](https://doi.org/10.1200/CCI-25-00226). <span class="reference-note"><strong>草稿简评：</strong>该研究比较 simple prompt、chain-of-thought、double filtering 和 rule-based method，直接说明 inference design 会改变单一肿瘤字段的 extraction quality。</span>
16. Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*. 2023;36:46595-46623. [https://doi.org/10.52202/075280-2020](https://doi.org/10.52202/075280-2020). <span class="reference-note"><strong>草稿简评：</strong>该研究支持把 LLM judge 用作可扩展的开发工具，同时记录 position、verbosity、self-enhancement 和 reasoning bias，因此本文不把它当作临床 endpoint。</span>
17. Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*. 2024:9440-9450. [https://doi.org/10.18653/v1/2024.acl-long.511](https://doi.org/10.18653/v1/2024.acl-long.511). <span class="reference-note"><strong>草稿简评：</strong>该论文单独验证了 LLM comparison 中的 position bias，说明 automated review 需要人工终审和保守解释。</span>
18. Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nature Medicine*. 2025;31:60-69. [https://doi.org/10.1038/s41591-024-03425-5](https://doi.org/10.1038/s41591-024-03425-5). <span class="reference-note"><strong>草稿简评：</strong>TRIPOD-LLM 要求清楚报告 model、prompt、evaluation procedure、task-specific metric 和 human oversight，可用于检查本 pilot 的透明度。</span>
19. Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. *npj Digital Medicine*. 2024;7:258. [https://doi.org/10.1038/s41746-024-01258-7](https://doi.org/10.1038/s41746-024-01258-7). <span class="reference-note"><strong>草稿简评：</strong>该文综述 142 项研究并提出 QUEST framework，可直接指导下一轮 evaluator selection、rating dimension、reliability assessment 和 adjudication。</span>
20. Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. *JCO Clinical Cancer Informatics*. 2026;10:e2500215. [https://doi.org/10.1200/CCI-25-00215](https://doi.org/10.1200/CCI-25-00215). <span class="reference-note"><strong>草稿简评：</strong>VALID 要求逐变量对照 expert reference、检查 internal consistency 和 plausibility，并做 replication，因此能准确说明 preference endpoint 与 absolute correctness 之间的差距。</span>

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
| Huang 等 [10] | 78 份肺癌报告用于 prompt development，774 份有效报告用于 independent testing，另有 191 份骨肉瘤病理报告 | 与专家整理的结构化数据比较 | spiral prompt engineering；分析专业术语和 TNM stage error | 证明 prompt iteration 有效，也说明复杂 staging 仍需要显式临床约束 |
| van Koevorden 等 [11] | 60 位头颈肿瘤患者、1,482 页文档、29 类字段 | 两位医生人工 extraction；六位临床专家判断 match、error type 和 impact | 预训练 open-source LLM 对 physician consensus；报告 absolute accuracy、error taxonomy 和时间节省 | 最接近的 clinician-validated comparator；absolute validation 更强，但不比较 same-model harness 和 baseline |
| Corso 等 [12] | 意大利语 NSCLC EHR；四个本地 small language model | 不同临床经验人员参与 prompt example 和 consistency assessment | zero-shot、few-shot、clinician-annotated few-shot | 直接支持 clinician expertise 和 prompt design，但没有本文的 gates、hooks 和跨字段 consistency control |
| Abhyankar 等 [13] | 700 份标注病历；五个癌种；部署到超过两百万份 notes | 人工 annotation 建立训练、验证和测试数据 | supervised fine-tuning；跨癌种 staging extraction | 规模和 staging validation 更强，但任务较窄且修改 model weights |
| Passweg 等 [14] | 400 份德语 oncology imaging report | 双人 extraction；由高级肿瘤科医生 adjudicate disagreement | 五个本地 LLM 与人工准确率比较 | 显示 metastasis 可接近人工、response 仍更难，支持本文的 field-dependent difficulty 解释 |
| Dubey 等 [15] | 三个癌种病历中的 ECOG status | 使用 adapted QUEST 做 human evaluation | rule-based、simple prompt、CoT 和 double filtering comparison | 是清楚的 prompt-level ablation，但只覆盖一个临床字段 |

## 附录 B：评估方法文献（供合作者审阅）

| 研究 | 主要发现 | 对本文的直接用途 |
|---|---|---|
| Zheng 等 [16] | LLM judge 可以接近人工偏好，但存在 position、verbosity、self-enhancement 和 reasoning bias | 支持将 LLM-as-a-judge 限定为开发工具，不把它作为最终临床结局 |
| Wang 等 [17] | 交换候选答案顺序可以显著改变 LLM evaluator 的判断；校准可减轻但不能消除问题 | 支持对 automated pairwise review 采用保守解释，并保留人工终审 |
| Gallifant 等，TRIPOD-LLM [18] | 提供覆盖 model、prompt、evaluation、human oversight 和 reproducibility 的 LLM 研究报告清单 | 用于检查本文 Methods 与 limitations 是否充分披露开发和评估过程 |
| Tam 等，QUEST [19] | 回顾 142 项 healthcare LLM human-evaluation study，并把流程组织为 planning、implementation、adjudication、scoring 和 review | 说明下一轮评估应预先定义 evaluator、维度、reliability 和争议处理，而不只记录 A/B/TIE |
| Estevez 等，VALID [20] | 要求 extracted variable 与 expert reference 逐项比较，并加入 consistency、plausibility 和 replication 检查 | 说明当前 preference study 不能替代 absolute correctness validation，并给出补强路径 |
