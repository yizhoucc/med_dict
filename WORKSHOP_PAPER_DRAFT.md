<!-- After editing this draft, run: python3 render_workshop_draft.py -->

# From clinician-guided error analysis to cross-cancer transfer: an inference harness for oncology note extraction

**Pilot report draft for clinical collaborator review**

Version 0.6, September 2026

Authors: [TODO]

Affiliations: [TODO]

Target venue and format: [TODO]

This draft includes two completed oncologist submissions. Both oncologists evaluated the breast-cancer cases, and the second also evaluated the pancreatic-cancer cases. The central account now follows the actual development sequence: clinician-guided error analysis in breast cancer, codification of those lessons in an inference harness, model-in-the-loop refinement in pancreatic cancer without physician input during development, and blinded clinician evaluation. The planned final multi-oncologist model and bracketed inferential results remain placeholders and must be completed before submission. Figure placeholders specify the intended visual design and current trend; they are not final artwork.

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

We built a failure-mode-driven inference harness around Qwen2.5-32B-Instruct-AWQ in two development stages. During breast-cancer development, an oncologist repeatedly reviewed outputs and identified clinically important errors. The team converted recurring errors into field-specific prompts, verification gates, and deterministic oncology rules. We then transferred this harness to pancreatic cancer. No physician reviewed pancreatic-cancer outputs during development. Instead, a rubric-informed Qwen reviewer identified candidate errors, an external development LLM synthesized the review history and proposed prompt or rule changes, and a human developer accepted, revised, and regression-tested those changes. Model weights remained frozen throughout. We compared the resulting harness with a single-prompt baseline using the same model and field contract on 40 CORAL benchmark notes, including 20 breast and 20 pancreatic cases. Two oncologists completed identity-masked A/B comparisons. Both evaluated breast cancer, and one also evaluated pancreatic cancer.

### Results

In the completed matched technical audit, the harness was preferred in 66 core comparisons, the baseline in 28, and 166 were ties. Across the completed clinical evaluations, the harness was preferred in 249 of 817 required-field judgments, the baseline in 39, and 529 were ties. Among the 288 directional judgments, 86.5% favored the harness. The two breast-cancer evaluations together contributed 163 harness preferences, 30 baseline preferences, and 365 ties. Exact agreement between oncologists was 82.7%, with Cohen's kappa of 0.645. In pancreatic cancer, where no physician had participated in development, the harness was preferred 86 times, the baseline 9 times, and 164 comparisons were ties. In a separate exploratory patient-letter evaluation by one oncologist, harness-based letters had a higher four-item mean score than ChatGPT letters in 14 of 20 breast cases, tied in 2, and scored lower in 4, but did not clearly outperform the same-model Qwen baseline.

[FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the harness was preferred in FINAL X, the baseline in FINAL Y, and FINAL Z were ties. The adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### Conclusions

The breast-cancer results support clinician-guided conversion of recurrent model errors into an explicit inference harness. The pancreatic-cancer result suggests that the resulting rules and evaluation preferences transferred to a second cancer domain without repeated physician involvement during development. This was AI-assisted, human-supervised refinement rather than autonomous self-modification. Additional oncologists and a prespecified clustered analysis remain necessary for the final claim.

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
> **Format:** A three-panel horizontal flow diagram with a smaller controlled-comparison inset.
>
> **Panel A, breast-cancer development:** breast note → harness output → oncologist review → recurrent failure category → prompt, gate, or deterministic rule revision → regression test. Label this stage "clinician-guided rule induction."
>
> **Panel B, pancreatic-cancer transfer:** transferred harness → pancreatic note → rubric-informed Qwen review → external development LLM synthesis and proposed revision → human acceptance or editing → regression test. Mark clearly that no physician reviewed pancreatic outputs during this stage and that model weights remained frozen.
>
> **Panel C, final evaluation:** 20 breast and 20 pancreatic benchmark notes → full harness and matched single-prompt baseline → identity-masked A/B presentation to oncologists.
>
> **Controlled-comparison inset:** show the shared frozen Qwen2.5-32B-Instruct-AWQ model and target schema above two branches. The harness branch adds field routing, selective context transfer, five verification gates, deterministic oncology hooks, cross-field checks, logging, and source attribution. The baseline branch uses one matched-schema prompt.
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

The first oncologist completed the 20 breast-cancer cases. The export contained 278 of 280 required judgments. Two ratings were missing and one extraneous pancreatic entry was excluded. The second oncologist completed all 280 required breast-cancer judgments and 259 of 260 pancreatic-cancer judgments; `p7 / lab_plan` was missing. Both raw exports were preserved unchanged. The scoring template retained stale filenames after the displayed results were updated. The project owner confirmed that both clinicians reviewed the newer outputs, but the exact hashes of the displayed PL and BL artifacts must be inserted before submission: [TODO].

For the 278 required breast-cancer judgments available from both oncologists, we calculated exact agreement and Cohen's kappa. We also summarized each clinician-by-cancer evaluation separately, because only one oncologist has completed the pancreatic-cancer set. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The planned final study will include [TARGET: 5] oncologists. The primary analysis will compare harness and baseline preference among directional ratings with a mixed-effects logistic model that includes evaluator, note, and field as grouping factors. Ties will be reported separately and included in a sensitivity analysis. We will report the effect estimate, 95% confidence interval, two-sided p value, and agreement across evaluators. The current two-oncologist summaries are interim descriptive analyses; the final statistical specification should be frozen before the remaining ratings are aggregated.

### 2.10 Exploratory patient-letter evaluation

Patient-letter generation preceded the extraction-only study and motivated the shift toward a more controlled task. One oncologist evaluated 20 breast-cancer notes with three patient letters per note. The systems were shown as A, B, and C: a single-prompt GPT-4o letter, a single-prompt Qwen2.5-32B letter, and a Qwen2.5-32B letter generated from the harness output. The evaluator scored accuracy, completeness, comprehensibility, and usefulness on five-point scales, marked possible hallucinations, and rated deployment readiness.

This evaluation was not powered or designed as a primary three-system trial. We therefore report it descriptively as evidence about a downstream use of structured extraction. It should not be combined with the extraction preference counts or used to claim that the complete letter system is superior.

## 3. Results

### 3.1 Development path and cross-cancer transfer

The development record contains approximately 15 breast-cancer iterations across 56 notes and approximately 18 pancreatic-cancer iterations across 100 notes. Breast-cancer revisions were informed by direct oncologist review. Pancreatic-cancer revisions were made without physician review of the pancreatic outputs, using the transferred rubric, model-based error review, external LLM-assisted synthesis, and human-controlled implementation described above.

This sequence produced two types of reuse. Some components transferred unchanged, including the five verification stages, temporal distinctions, source attribution, and rules that separate active treatment from plans or supportive medication. Other components required cancer-specific routing, especially disease terminology, regimen interpretation, and post-processing conditions. The resulting system therefore reused the error-handling framework without assuming that breast and pancreatic cancer were clinically interchangeable.

The pancreatic-cancer clinician result is the most relevant evidence for this transfer. The oncologist preferred the harness in 86 field comparisons and the baseline in 9, with 164 ties. At the note level, the harness had a positive margin in 18 of 20 pancreatic cases and tied in two. Because no physician reviewed pancreatic outputs during development, this result is consistent with transfer of the previously codified evaluation preferences and workflow. It does not by itself identify whether the gain came from the transferred rules, pancreatic-specific revisions, the model-based reviewer, or their combination.

### 3.2 Current oncologist evaluation

The available clinical evidence comprises three completed clinician-by-cancer evaluations: breast cancer from both oncologists and pancreatic cancer from the second oncologist. Across 817 required-field judgments, the harness was preferred 249 times, the baseline 39 times, and 529 comparisons were ties. The harness received 86.5% of the 288 directional judgments.

| Completed evaluation | Required judgments | Harness | Baseline | Tie | Harness share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Oncologist 01, breast cancer | 278 | 84 | 22 | 172 | 79.2% |
| Oncologist 02, breast cancer | 280 | 79 | 8 | 193 | 90.8% |
| Oncologist 02, pancreatic cancer | 259 | 86 | 9 | 164 | 90.5% |
| **All completed evaluations** | **817** | **249** | **39** | **529** | **86.5%** |

> **Figure 2 placeholder: Clinician preference distributions by evaluator and cancer type.**
>
> **Plot:** Three 100% stacked horizontal bars.
>
> **x-axis:** Share of required-field judgments, from 0% to 100%.
>
> **y-axis:** Oncologist 01 breast cancer, Oncologist 02 breast cancer, and Oncologist 02 pancreatic cancer.
>
> **Encoding:** Blue for harness preferred, light gray for tie, and orange for baseline preferred. Print raw counts inside segments and the harness share among directional judgments at the right of each bar.
>
> **Observed trend:** Ties are the largest category in every evaluation, but harness preferences substantially exceed baseline preferences. The baseline share falls from 7.9% for the first breast evaluation to 2.9% for the second breast evaluation and 3.5% for pancreatic cancer.
>
> **Draft caption:** *Figure 2. Distribution of identity-masked clinician preferences across the three completed clinician-by-cancer evaluations. Most judgments were ties, as expected for systems using the same base model, but directional judgments consistently favored the inference harness.*

Both clinicians independently favored the harness on the breast-cancer set. Their pooled breast result was 163 harness preferences, 30 baseline preferences, and 365 ties. The first oncologist's per-note result was 18 harness wins, one baseline win, and one tie. The second oncologist's result was 20 harness wins. When both breast ratings were combined within each note, all 20 notes had a positive harness-minus-baseline margin.

The second oncologist's pancreatic-cancer evaluation produced 86 harness preferences, 9 baseline preferences, and 164 ties. The harness won 18 of 20 notes by within-note field margin, and two notes were tied. One required judgment, `p7 / lab_plan`, was missing.

The final mixed-effects result remains pending:

> [FINAL MULTI-RATER RESULT: Across FINAL N oncologists and FINAL N evaluable judgments, the adjusted analysis showed a significant preference for the harness, effect estimate FINAL, 95% CI FINAL, p=FINAL.]

### 3.3 Inter-rater agreement and core fields

The two oncologists shared 278 required breast-cancer comparisons. They gave the same verdict on 230, for 82.7% exact agreement and Cohen's kappa of 0.645. Of the 48 disagreements, only nine were direct reversals between harness and baseline: seven changed from a baseline preference by the first oncologist to a harness preference by the second, and two changed in the opposite direction. The remaining disagreements involved a tie from one evaluator.

> **Figure 3 placeholder: Inter-rater agreement matrix for breast-cancer judgments.**
>
> **Plot:** A 3 × 3 heatmap or confusion matrix.
>
> **x-axis:** Oncologist 02 verdict: harness preferred, tie, baseline preferred.
>
> **y-axis:** Oncologist 01 verdict in the same order.
>
> **Encoding:** Cell color intensity represents the number of shared judgments; each cell displays its count. Add exact agreement and Cohen's kappa above the matrix.
>
> **Observed trend:** The diagonal contains 230 of 278 judgments, dominated by 162 tie-tie decisions and 62 harness-harness decisions. Direct reversals are uncommon: 7 baseline-to-harness and 2 harness-to-baseline.
>
> **Draft caption:** *Figure 3. Agreement between the two oncologists on 278 shared breast-cancer field comparisons. Agreement was 82.7% with Cohen's kappa of 0.645. Most disagreements involved a tie from one evaluator rather than opposite directional preferences.*

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

> **Figure 4 placeholder: Preference profile across core clinical categories.**
>
> **Plot:** Seven 100% stacked horizontal bars, ordered by net harness advantage.
>
> **x-axis:** Share of applicable clinician judgments, from 0% to 100%.
>
> **y-axis:** Active anticancer medications, regional or overall metastasis, stage, treatment response, completed molecular or genetic results, distant metastasis, and breast type or receptors.
>
> **Encoding:** Blue for harness preferred, light gray for tie, and orange for baseline preferred. Show `PL / BL / TIE` counts at the right of each bar. Because receptor status applies only to breast cancer, use percentages for visual comparison and retain raw denominators in the labels or caption.
>
> **Observed trend:** Active anticancer medications and regional or overall metastatic involvement show the largest margins. Stage and response also favor the harness. Distant metastasis is dominated by ties, and breast type or receptor status is close to even.
>
> **Draft caption:** *Figure 4. Clinician preference by prespecified core clinical category. The largest harness advantages occurred in active-treatment identification and metastatic-involvement classification. Breast type and receptor status showed the smallest margin.*

Active anticancer medication and regional or overall metastatic involvement produced the largest and most consistent margins. Medication planning, outside the seven core categories, totaled 31 harness preferences, no baseline preferences, and 29 ties across the completed evaluations. Procedure planning was closer at 8 harness preferences, 5 baseline preferences, and 47 ties. Breast type and receptor status remained the weakest core category, with only a 2-rating net advantage and 60% exact agreement between oncologists.

Exploratory note-level sign tests support the same direction without treating every field as an independent observation. The combined breast margin was positive in all 20 notes (`p=1.9e-6`, two-sided exact sign test). In the pancreatic set, 18 note-level margins were positive and two were tied (`p=7.6e-6` after excluding ties). These tests were not the prespecified final model and should not replace the planned evaluator-note-field analysis.

> **Figure 5 placeholder: Per-note clinician preference margins.**
>
> **Plot:** Two aligned bar-chart panels, one for breast cancer and one for pancreatic cancer, with a horizontal zero reference line.
>
> **x-axis:** Sample identifier, `b1` through `b20` in the breast panel and `p1` through `p20` in the pancreatic panel.
>
> **y-axis:** Normalized net preference margin per note, calculated as `(harness-preferred fields - baseline-preferred fields) / completed required-field judgments for that note`, on a shared scale. The breast panel pools the two oncologists; the pancreatic panel currently contains Oncologist 02 only.
>
> **Encoding:** Blue bars above zero favor the harness, orange bars below zero favor the baseline, and gray markers at zero indicate tied note-level margins.
>
> **Observed trend:** All 20 pooled breast margins are positive. Eighteen pancreatic margins are positive and two are zero; none are negative.
>
> **Draft caption:** *Figure 5. Distribution of normalized clinician preference margins across individual notes. The harness had a positive pooled margin in every breast-cancer note and in 18 of 20 pancreatic-cancer notes; the remaining two pancreatic notes were tied.*

> **Figure 6 placeholder: Adjusted multi-rater effect estimates.**
>
> **Plot:** Forest plot to be populated after the planned final clinician sample is complete.
>
> **x-axis:** Adjusted odds ratio for a directional judgment favoring the harness rather than the baseline, displayed on a logarithmic scale with a vertical reference line at 1.0.
>
> **y-axis:** Overall effect, breast cancer, pancreatic cancer, and the seven prespecified core categories. Include only strata supported by the final sample size.
>
> **Encoding:** Point estimate with 95% confidence interval. Use a filled marker for the primary adjusted analysis and open markers for sensitivity analyses that handle ties differently.
>
> **Expected interpretation:** This figure will replace reliance on pooled field counts by showing estimates that account for repeated judgments by evaluator, note, and field. No trend should be asserted until the final model is fit.
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
> **Plot:** Horizontal dumbbell plot with one row per core category.
>
> **x-axis:** Net preference rate, calculated as `(harness better - baseline better) / applicable judgments`, with a vertical reference line at zero.
>
> **y-axis:** The seven core clinical categories.
>
> **Encoding:** One marker for the complete v2.2 source-grounded technical audit and one marker for the currently completed clinician ratings; connect the markers within each category.
>
> **Observed trend:** Both evidence sources favor the harness overall. The clinician ratings show especially large margins for active anticancer medication and regional or overall metastatic involvement. Stage changes from a small negative technical-audit margin to a positive clinician margin, while distant metastasis and receptor status show smaller clinician margins.
>
> **Interpretive warning:** The reviewers and evaluated pipeline versions differ, so this is a descriptive comparison of patterns, not an agreement test or causal before-and-after estimate.
>
> **Draft caption:** *Supplementary Figure S1. Category-level net preference rates in the complete technical audit and current clinician evaluation. Differences between series should be interpreted descriptively because the review processes and pipeline versions were not identical.*

### 3.5 Targeted repair evaluation

Four high-impact failures identified in the v2.2 audit were repaired with conservative rules. Across 51 applicable comparisons in six affected samples and two controls, the repaired harness recorded 29 wins, no baseline wins, and 22 ties. No P0 error remained in this targeted set, and neither control developed a detected core regression. These targeted results show that the identified errors were repairable. They do not replace the full 40-note result.

### 3.6 Qualitative comments

The two exports contained 12 written comments. These comments add useful context but also show why preference counts should not be treated as a complete correctness assessment. For one breast imaging-plan item, the first oncologist preferred the harness because the baseline summarized completed findings rather than the planned PET/CT. The second oncologist rated the same comparison as a tie and wrote that neither answer correctly captured the absence of a new imaging plan. This disagreement should be adjudicated before the case is used as a manuscript example.

Other comments identified unsupported receptor status, a regional-node omission, uncertainty about response after a newly started second-line regimen, and cases in which both outputs were inaccurate. These observations support the value of specialist review and identify concrete targets for final error analysis. They also show that a tie can mean either that both outputs are adequate or that both are wrong.

> **Supplementary Figure S2 placeholder: Adjudicated interpretation of tie judgments.**
>
> **Plot:** Stacked bars or a compact alluvial diagram after manual adjudication of a prespecified sample of ties.
>
> **x-axis:** Tie interpretation category: both outputs clinically adequate, both partly correct but incomplete, both incorrect, or indeterminate from the note.
>
> **y-axis:** Number or percentage of adjudicated ties.
>
> **Encoding:** Split bars by cancer type or core field. A secondary annotation may show whether the two outputs failed for the same reason or different reasons.
>
> **Data status:** Not yet available. The current rating interface records only `TIE`, so this figure requires manual review rather than inference from the existing CSV files.
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
> **Plot:** Two paired-difference panels, one comparing the harness-based letter with ChatGPT and one comparing it with the Qwen single-prompt baseline.
>
> **x-axis:** Breast-cancer sample identifier, `b1` through `b20`.
>
> **y-axis:** Difference in the mean of accuracy, completeness, comprehensibility, and usefulness. Positive values favor the harness-based letter.
>
> **Encoding:** Blue points above zero, gray points at zero, and orange points below zero. Add a horizontal zero line and annotate each panel with the win, tie, and loss count.
>
> **Observed trend:** Harness-based letters are descriptively better than ChatGPT for most notes, but the distribution is close to the same-model Qwen baseline and includes both wins and losses.
>
> **Draft caption:** *Figure 7. Exploratory paired differences in oncologist-rated patient-letter quality across 20 breast-cancer notes. Harness-based letters more often outscored ChatGPT, but did not show a clear advantage over the same-model single-prompt baseline.*

## 4. Discussion

### 4.1 Main interpretation

[FINAL OPENING: The multi-oncologist evaluation showed a statistically significant preference for the inference harness over the same-model single-prompt baseline.]

The main result concerns the development process around the model. A clinician helped identify recurrent errors in breast cancer, those errors were converted into explicit parts of the inference harness, and the resulting process was adapted to pancreatic cancer without physician review during pancreatic development. The final clinician ratings favored the harness in both domains. This sequence connects the development method to the evaluation more directly than a static comparison of two prompts would.

Most field comparisons were ties, which is expected because both systems used the same strong base model. When an oncologist found a meaningful difference, however, the harness was preferred more than six times as often as the baseline. The gains were concentrated in fields that need temporal interpretation or clinical classification. This is consistent with a system that leaves straightforward answers alone and intervenes when a known failure pattern appears.

The evidence is still a pilot. The breast and pancreatic stages were not randomized, the refinement process combined several tools, and the benchmark later informed targeted repairs. The results therefore support the whole development strategy, not a causal claim for any single model, prompt, gate, or hook.

### 4.2 Clinician-guided rule induction in breast cancer

The oncologist's role during breast development was different from conventional dataset labeling. The clinician did not produce thousands of field labels for model training. Instead, the clinician reviewed concrete outputs and identified mistakes that would matter in practice. The team then translated repeated mistakes into reusable instructions and checks.

When specialist time is scarce, a clinician can focus on defining the boundary of a difficult concept, such as active therapy, response to the current regimen, or regional versus distant disease, instead of reviewing every future note. Once encoded, the rule can be applied consistently, logged, and regression-tested. The two independent breast evaluations suggest that the resulting harness did not merely reproduce one collaborator's preferences. Both oncologists favored it, although their individual judgments were not identical.

This development pattern also explains why we call the system a harness. The contribution is the accumulation of executable clinical distinctions around a frozen model. Some distinctions live in prompts, some in verification, and some in deterministic code. The clinician supplies the clinical boundary; the engineering process turns that boundary into repeatable behavior.

### 4.3 Transfer and AI-assisted refinement in pancreatic cancer

The pancreatic-cancer stage tests whether the accumulated process can travel beyond the domain in which the clinician gave direct feedback. Pancreatic notes differ in disease course, treatment regimens, staging language, and surgical context. We therefore did not simply reuse every breast-specific rule. We transferred the general workflow, retained rules that expressed shared clinical distinctions, and added pancreatic-specific routing where the notes exposed new failure patterns.

No physician reviewed pancreatic outputs during this development stage. A Qwen reviewer applied the inherited rubric to each output, and an external development LLM synthesized the review history and proposed changes. A human developer decided which changes to implement and ran regression tests. The pancreatic clinician result, 86 harness preferences versus 9 baseline preferences with 164 ties, is consistent with successful transfer under this model-in-the-loop process.

We do not describe this as autonomous self-evolution. The deployed model did not independently modify its code or approve its own changes. "AI-assisted evolution" is reasonable only in the narrower sense that models helped identify errors and formulate revisions across repeated cycles. Human supervision remained part of the development loop.

### 4.4 Which questions were difficult for the model?

The combined clinician ratings separate relatively direct extraction from questions that require clinical context.

Straightforward facts often produced ties. Both systems could usually identify an explicitly stated imaging result, procedure, or receptor value. The harder problems involved deciding what the fact meant in the current clinical context.

Active anticancer medication showed the largest and most stable difference: 50 harness preferences, one baseline preference, and nine ties across the completed evaluations. This field requires the model to distinguish treatment from chronic home medications, supportive drugs, discontinued regimens, and future options. Medication planning also strongly favored the harness, 31 to zero with 29 ties, because future actions must remain separate from current treatment and recent changes.

Stage and metastatic involvement require related decisions. The model must distinguish regional lymph nodes from distant spread, preserve uncertainty for lesions awaiting confirmation, and reconcile metastatic status with stage. Across the available ratings, regional or overall metastatic involvement favored the harness 35 to one, while stage favored it 24 to two. The harness was built to check these relationships across fields rather than extract each label in isolation.

Treatment response remained conceptually difficult, but the clinical result was directionally consistent: 18 harness preferences, one baseline preference, and 41 ties. A note may include old progression, current symptoms, stable imaging, tumor-marker trends, and a newly started regimen. Determining which evidence reflects response to the current treatment requires a timeline, not keyword recognition. A second-oncologist comment that response was not yet clear after starting second-line therapy illustrates this temporal boundary.

Tumor type and receptor status remains the weakest core category. Across the two breast evaluations it favored the harness only 8 to 6, with 26 ties, and had the lowest inter-rater agreement at 60%. These values are often stated explicitly, so the baseline can perform well. The field also becomes difficult when a note contains bilateral disease, historical and recurrent specimens, or discordant receptor results. This remains an area for clinical review rather than a claimed strength.

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

This pilot is intended to establish whether the effect is large enough and clinically coherent enough to justify a larger study. The second oncologist independently reproduced the breast-cancer direction, and the pancreatic-cancer evaluation showed a similar preference pattern. The next step is no longer to establish whether any replication exists. It is to determine how stable the effect remains across additional evaluators and to fit the prespecified clustered analysis with enough clinicians to estimate evaluator variation credibly.

External validation remains important. CORAL notes come from one institution and represent two cancer domains. The most informative next dataset would contain longitudinal oncology notes from a different health system, with independent clinical annotation and a field contract fixed before evaluation.

## 5. Limitations

The current clinical result comes from two oncologists. Both evaluated breast cancer, but only one evaluated pancreatic cancer. The breast subset supports an initial inter-rater estimate, but two evaluators are insufficient to characterize variability across oncologists, and the pancreatic result does not yet have independent replication. The final manuscript should replace the interim descriptive analysis with the planned multi-rater model.

The A/B interface concealed system identity but used fixed left and right positions, with the harness always shown as A. Agreement across two reviewers and the presence of many ties reduce concern about indiscriminate selection of A, but they do not remove possible position bias. The harness output also included source attribution while the baseline did not. Attribution is part of the system being evaluated, but it may influence preference. A future study should randomize side assignment and separately test the effect of attribution.

The preference labels do not distinguish "both correct" from "both incorrect." Written comments in the second export explicitly identify some ties in which neither output was satisfactory. Final reporting should therefore pair preference counts with adjudicated error categories rather than interpret every tie as success.

The technical audit used LLM-assisted reviewers and repeated error analysis. It is useful for identifying failures but is not independent clinical validation. The complete 40-note table represents v2.2, while later high-impact repairs were tested on affected samples and controls rather than a new full run. Because the benchmark informed these repairs before the clinician-rated artifacts were prepared, the clinical comparison is not a pristine external validation.

The development stages were sequential rather than randomized. Breast development combined oncologist feedback with engineering judgment, while pancreatic development combined an inherited rubric, a model-based reviewer, an external development LLM, and human implementation. The study cannot isolate the contribution of any one element. It also does not evaluate a fully autonomous system because a human approved and regression-tested changes.

The dataset is small and comes from one institution. The harness contains clinical rules derived from observed errors, and some may capture documentation conventions specific to CORAL. The study does not yet include a complete component ablation, another model family, or external notes.

The patient-letter analysis has one oncologist, 20 breast cases, three systems, and several correlated ratings. The exploratory result cannot establish superiority, and it did not show a clear advantage over the same-model Qwen baseline. The study does not test patient understanding, treatment decisions, workflow efficiency, or clinical outcomes.

## 6. Conclusion

[FINAL CONCLUSION: In a multi-oncologist identity-masked evaluation, the failure-mode-driven inference harness significantly outperformed a same-model single-prompt baseline for structured extraction from longitudinal oncology notes.]

The current pilot supports a development strategy in which a clinician identifies clinically important failure patterns, the team converts those patterns into an explicit inference harness, and the harness is adapted to another cancer domain through AI-assisted, human-supervised refinement. Two oncologists favored the harness on breast cancer. One oncologist also favored it on pancreatic cancer, where no physician had participated in development. The largest gains involved active therapy, medication planning, and metastatic status. The model weights remained frozen throughout. Additional evaluators, a prespecified clustered analysis, and external validation are still needed before this pilot pattern becomes a confirmatory claim.

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
