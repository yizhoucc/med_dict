<!--
BILINGUAL MAINTENANCE RULE:
1. Keep the complete English manuscript first and the complete Chinese manuscript second in this same file.
2. The two versions must match in scientific claims, numbers, main result tables, figure numbering, and references. The Chinese review version may contain additional annotations and review-only appendices.
3. Keep the English version close to submission prose. The Chinese version may retain collaborator questions, figure-design notes, and explanatory annotations.
4. After changing either version, update the other version in the same edit and run: python3 render_workshop_draft.py
5. Apply the humanizer pass to both versions. Chinese should read as natural academic prose, not as a literal machine translation.
-->

<div id="english-version"></div>

# From physician-guided extraction to model-assisted cross-cancer adaptation: an inference harness for oncology notes and patient communication

**Pilot report**

Version 0.22, October 2026

## Abstract

### Background

Large language models can extract structured information from clinical notes, but a single prompt often confuses current and historical treatment, suspected and confirmed disease, and completed findings and future plans. These errors are difficult to detect because the resulting text remains fluent. Oncology notes are a demanding test case because they combine pathology, imaging, treatment history, current therapy, response assessment, and conditional plans across a long clinical timeline.

### Objective

To test whether a physician-informed inference harness can improve oncology information extraction from a frozen, locally served open-weight language model and adapt to a second cancer domain without target-domain physician review.

### Methods

We built an inference harness around Qwen2.5-32B-Instruct-AWQ in two development stages. During breast-cancer development, a physician author and the model-development authors reviewed outputs and converted recurring clinical errors into field-specific prompts, conditional verification gates, and deterministic oncology rules. For pancreatic cancer, a general-purpose external LLM organized the breast-cancer failure history into a written rubric and suggested candidate adaptations. A source-grounded LLM reviewer then identified possible errors in pancreatic-cancer outputs. The authors decided which changes to implement. Model weights remained frozen, and no physician reviewed pancreatic-cancer outputs during development. We compared the harness with a same-model single-prompt baseline on an untouched holdout test set of 20 breast-cancer and 20 pancreatic-cancer notes. Four physicians evaluated both cancer types. A secondary four-stage technical ablation compared the single-prompt baseline, decomposed extraction with shared orchestration safeguards, added verification gates, and the full deterministic hook layer. One blinded Codex judge evaluated 120 adjacent-stage pairs.

### Results

Eight evaluator-by-cancer evaluations contributed 2,160 required note-field judgments: 699 harness preferences (32.36%), 1,290 ties (59.72%), and 171 baseline preferences (7.92%). Among 870 directional judgments, the crude harness share was 80.34% (note-cluster robust 95% CI 76.14% to 83.96%). The adjusted probability of a harness preference was 79.98% (95% CI 75.75% to 83.63%), an absolute difference of +29.98 percentage points from 50% (95% CI +25.75 to +33.63). The corresponding adjusted probabilities were 81.66% for breast cancer (95% CI 76.02% to 86.21%) and 78.15% for pancreatic cancer (95% CI 71.47% to 83.62%). The secondary odds-scale estimate was 3.995 overall (95% CI 3.124 to 5.110; p=2.67 x 10^-28). The tie-inclusive net preference margin was +24.44 percentage points (note-cluster robust 95% CI +20.42 to +28.47). Thirty-nine of 40 pooled note margins were positive (exact sign-test p=7.46 x 10^-11). In the technical ablation, only the addition of verification gates produced a clear overall preference (24 versus 2, with 14 ties; exact p=0.0000105).

### Conclusions

The results support a staged development path. Physician feedback first shaped a breast-cancer inference harness. A model-assisted, author-supervised loop then adapted the accumulated rubric and failure rules to pancreatic cancer without target-domain physician review. The technical ablation localized the clearest gain to verification gates, while the deterministic hook layer showed cancer-specific tradeoffs. Broader claims require tie adjudication, targeted physician review of component changes, more physicians, and external validation.

## 1. Introduction

Large language models (LLMs) are increasingly used for clinical language tasks, including clinical documentation, information retrieval, and patient communication. Many of these applications depend on first identifying relevant information in the electronic health record. Reliable extraction is therefore a necessary step before oncology notes can be summarized, analyzed, or rewritten for another audience [1].

Although EHRs contain structured fields, much clinically useful information remains in free-text notes [21]. Oncology notes are particularly difficult because they combine diagnoses, pathology, prior and planned treatments, changing disease status, and uncertainty across long timelines. Relevant facts may appear in several sections or be carried forward from earlier encounters. In a study of nearly 3 million outpatient notes, note length and redundancy increased substantially over a decade, with copied and templated text contributing to both trends [22]. An extraction system must determine which statements describe the patient's current disease and treatment, not simply locate medically relevant phrases.

LLMs can assist with extraction, but fluent outputs may still omit facts, confuse historical and current findings, or add unsupported conclusions. Commercially hosted models also raise governance and privacy concerns when protected health information leaves institutional infrastructure. A locally deployed open-weight model can keep note processing within local infrastructure, but local deployment does not by itself prevent extraction errors or establish privacy compliance [3].

Extraction performance depends on how a model is used as well as on the model itself. Fine-tuning can adapt a model to a clinical task but requires labeled data and model updates. Prompt design, task decomposition, validation, and retry logic provide another route for improving a frozen model. Iterative prompting has improved oncology extraction, although staging and specialist terminology remain difficult [10]. Human-in-the-loop studies also show that recurring errors can be organized into an error ontology and converted into general instructions rather than case-specific fixes [23]. The oncology literature contains strong examples of individual components, but evidence remains limited for an integrated inference workflow evaluated against a same-model baseline by practicing physicians [1].

We use inference harness to refer to the execution layer around a frozen LLM: field-specific task routing, selective context transfer, conditional verification, deterministic clinical rules, logging, and model-selected supporting excerpts when available. We first developed the harness for breast-cancer notes through review by a physician author and the model-development authors. We then adapted it to pancreatic-cancer notes without physician review of target-domain outputs, using a general-purpose external LLM to synthesize a written review rubric and a source-grounded LLM reviewer to identify candidate errors. Independent physicians later compared the final harness with a same-model single-prompt baseline. This study evaluates whether the integrated workflow improves structured extraction and whether its development process can be reused across a second cancer domain.

## 2. Methods

### 2.1 Study design

The study followed three linked steps. First, a physician author and the model-development authors used human-in-the-loop review to develop a breast-cancer inference harness. Second, the breast-cancer failure history and clinical distinctions were transferred into a model-assisted pancreatic-cancer adaptation loop without physician review of pancreatic-cancer outputs. Third, the final harness was compared with a single-prompt baseline using the same frozen model and field schema on an untouched holdout test set of 40 expert-annotated CORAL notes. Four physicians completed system-label-masked A/B evaluations for both cancer types.

The breast-cancer stage examined whether physician feedback could be converted into reusable prompts, conditional gates, and rules. The pancreatic-cancer stage examined whether an LLM-assisted review loop could reuse this knowledge without target-domain physician input. The extraction comparison isolated the complete inference harness while holding the base model and target fields constant. A secondary staged technical ablation compared four nested inference conditions using blinded LLM review.

### 2.2 Dataset

We used CORAL, a controlled-access PhysioNet dataset of deidentified medical oncology progress notes from the University of California, San Francisco [2]. The development pool contains 100 breast-cancer and 100 pancreatic-cancer notes without expert annotation. The untouched holdout test set contains 20 expert-annotated notes for each cancer type. The expert annotations were not provided to the extraction model, the development-stage LLM reviewer, or the physicians through the A/B evaluation interface.

The records are real clinical notes rather than web-derived questions, synthetic cases, or model-generated narratives. They retain repeated histories, copied-forward content, uncertain findings, and conditional plans that make longitudinal oncology extraction difficult. The value of this dataset is clinical realism and expert annotation rather than scale. Access requires the PhysioNet credentialing and data-use process described by the dataset authors.

All 100 development notes for each cancer type formed the development data. Small subsets supported faster targeted iterations when errors were still common. Major and final development runs covered all 100 notes to identify less frequent errors and check for regression. The 20 annotated notes per cancer type remained an untouched holdout test set and were used only for the reported evaluation. No model-weight training or fine-tuning was performed.

### 2.3 Base model and baseline

Both conditions used Qwen2.5-32B-Instruct-AWQ [9], served locally through vLLM. The baseline made one model call per note and returned the complete target schema. It did not use task decomposition, verification gates, retries, dictionaries, or deterministic post-processing. The finalized baseline used in the physician review package and the harness used the same field definitions and output contract.

### 2.4 Inference harness

The harness uses several stages because the target fields do not all require the same context or reasoning.

First, field-specific prompts independently extract visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Dependent prompts then receive selected earlier results. For example, stage and metastatic status provide context for treatment intent, while current therapy and clinical findings provide context for response assessment. Plan fields are extracted primarily from the Assessment and Plan section. The harness first searches for this section with deterministic patterns. If that search fails, it makes up to three model-based copy attempts and checks each candidate against the source note. If all attempts fail, the system skips the plan-specific prompts rather than generating a replacement section.

Eligible model outputs pass through the applicable verification stages. A stage runs only when its trigger condition is met:

1. JSON formatting repair when parsing fails.
2. Schema validation to detect incorrect or leaked keys.
3. Specificity and semantic alignment checks to identify vague or off-task answers.
4. Faithfulness trimming that removes clearly unsupported or contradictory content while retaining supported information.
5. Temporal filtering that removes completed or historical events from future-plan fields.

The deterministic layer addresses recurrent errors with high-confidence clinical rules. These rules distinguish regional nodes from distant disease, suspected from confirmed metastasis, active anticancer therapy from supportive or home medication, and current response from pretreatment change. The system also returns model-selected supporting excerpts when available.

| Harness component | Failure mode addressed | Example |
|---|---|---|
| Field-specific extraction | One large prompt drops or mixes fields | Separate active medication extraction from treatment plans |
| Bounded retry and fallback | A note section cannot be recovered reliably | Retry Assessment and Plan extraction up to three times, then skip plan-only prompts |
| Dependency-aware context | Related fields contradict one another | Use metastatic status when interpreting stage and response |
| Semantic verification | A medically related answer does not answer the field | Remove a future treatment plan from current response |
| Faithfulness trimming | The model adds an unsupported conclusion | Preserve a lesion as suspected when biopsy is pending |
| Temporal filtering | Completed results appear as future plans | Remove a completed scan from imaging plan |
| Drug and context rules | Medication names are classified without clinical context | Separate anticancer drugs from home and supportive medications |
| Cross-field clinical rules | Stage, nodes, and distant disease become inconsistent | Keep axillary nodes regional rather than distant |
| Source attribution | A reviewer cannot trace an extracted value | Return a model-selected supporting excerpt when available |

<div data-rough-figure="1"></div>

***Figure 1.*** *Development, transfer, and evaluation pathway. (A) The breast and PDAC development pools were separated from the locked 40-note holdout set. (B) Physician and investigator source review informed breast-cancer prompts, gates, hooks, and regression tests. (C) The breast-cancer failure history, clinical rubric, and reusable harness supported author-controlled PDAC adaptation with a source-grounded LLM reviewer and no target-domain physician review of outputs. (D) Four physicians completed identity-masked field comparisons between the harness and same-model single-prompt baseline on the locked holdout set.*

### 2.5 Physician-informed breast-cancer development

Breast-cancer development used all 100 development notes. Early and targeted iterations used smaller subsets to shorten the feedback cycle, while major and final runs covered all 100 notes to detect less frequent errors and check for regression. A physician author familiar with oncology reviewed selected outputs with the model-development authors. This physician author was not an oncology physician and did not participate in the final evaluation. The authors compared flagged outputs with the source notes, grouped recurring errors, and translated them into general changes to prompts, verification logic, or deterministic rules. This process did not produce a supervised training set, and the model weights were not updated.

Targeted iterations tested whether a change corrected its intended failure. Major and final runs used the full development set to assess the overall version and check for regression. The objective was to encode recurring clinical distinctions rather than case-specific corrections. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general regional-node rule. An error in which a planned drug was listed as active therapy motivated a temporal medication rule.

### 2.6 Model-assisted pancreatic-cancer adaptation

The breast-cancer harness was adapted to pancreatic cancer across 12 archived development checkpoints: nine on a fixed 30-note subset for faster targeted iteration and three on the full 100-note development set. No physician reviewed pancreatic-cancer outputs during this stage. A general-purpose external LLM organized the accumulated breast-cancer failure history and clinical distinctions into a written review rubric and suggested candidate pancreatic-cancer adaptations. The Qwen pipeline generated structured fields, and a source-grounded Qwen reviewer compared each output with the source note. The model-development author assessed the error flags and proposals and decided which changes to implement. Major and final runs covered all 100 development notes.

The reviewer prompt included the field definitions, severity criteria, and clinical distinctions established during breast-cancer development. Earlier error categories could therefore guide review in the new domain while the reviewer still detected pancreatic-specific problems such as regimen names and dose representation. This LLM-assisted loop changed prompts, rules, and workflow code rather than model weights. The author controlled implementation, so we describe the process as model-assisted and author-supervised.

Throughout both development stages, the pipeline logged the original model output, each verification action, and deterministic corrections. This record allowed the team to trace a final value back through the harness and to retain or reject proposed changes based on regression results.

### 2.7 Physician-prioritized clinical fields

Before evaluating the test set, the physician author designated seven fields as clinically important for understanding disease status and current management:

1. Which anticancer drugs is the patient actively receiving?
2. What is the current cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer responding to the current treatment?
6. What is the breast cancer type and ER/PR/HER2 status?
7. What completed molecular or genetic results are documented?

Metastatic involvement and distant metastasis were intentionally evaluated as separate questions. Breast cancer type and receptor status were intentionally combined as a breast-specific question. The physician evaluation instrument also included genetic testing plans, supportive medications, procedure plans, imaging plans, laboratory plans, medication plans, and recent treatment changes. Laboratory summary and general clinical findings were optional and excluded from the primary physician analysis.

### 2.8 Development-stage LLM-assisted review

LLM-assisted review was used only as a development instrument, particularly during adaptation to pancreatic cancer. For each candidate output, the reviewer model compared the extracted fields with the source note using the field definitions and severity criteria. It flagged possible omissions, unsupported claims, semantic mismatches, and temporal errors. Prior work shows that LLM judges can approximate human preferences but can also exhibit position, verbosity, and self-preference biases [16,17]. The model-development author reviewed these flags and decided which revisions to implement. The LLM reviewer did not supply the reported test-set outcomes and did not replace the independent physician evaluation.

### 2.9 Physician evaluation

The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were not told which output came from the harness, whether the systems used the same underlying model, or which condition was expected to perform better. The assignment was fixed across evaluations, with the harness displayed as A and the baseline as B. Model-selected supporting excerpts appeared with the harness output when available. The study therefore evaluated the complete presented systems and did not isolate the effect of attribution.

The physician author who participated in breast-cancer development was not one of the final evaluators. Four physicians independently evaluated both the breast-cancer and pancreatic-cancer outputs.

All four physicians completed the same 280 required breast-cancer comparisons and 260 required pancreatic-cancer comparisons. We calculated pairwise exact agreement and Cohen's kappa for each pair within each cancer type. We also summarized all eight evaluator-by-cancer analyses separately. Pooled counts are descriptive and do not treat field-level judgments as independent observations.

The primary estimand was P(harness preference | directional judgment). Ties were excluded only from this directional analysis. We fitted a population-averaged logistic generalized estimating equation with an exchangeable working correlation within each note. Evaluator was included as a fixed effect, and the overall model also included cancer type. Cancer-specific models adjusted for evaluator. We standardized fitted probabilities over the observed covariate distribution and used the delta method on the logit scale for 95% confidence intervals. The adjusted probability was the primary effect; its difference from 50% was the absolute effect, and the corresponding odds ratio was a secondary model-scale effect. Passweg et al. used a related report-cluster robust logistic and delta-method marginal-effects analysis for oncology extraction [14]. Other physician comparisons have retained an equivalent response category [24], reported preference proportions with confidence intervals [25], or modeled repeated observations across cases and participants [26].

We retained the complete three-category distribution for descriptive analysis. A crude directional proportion received a sandwich confidence interval clustered by note. For a tie-inclusive sensitivity measure, harness, tie, and baseline judgments were coded as 1, 0, and -1; their mean is the net preference margin. Its normal 95% confidence interval used a sandwich standard error clustered by note, not bootstrap resampling. We also pooled field judgments within each note and used an exact two-sided sign test on positive versus negative note margins.

### 2.10 Staged technical ablation

We ran a secondary technical ablation on the same 20 breast-cancer and 20 pancreatic-cancer benchmark notes. All four conditions used Qwen2.5-32B-Instruct-AWQ, greedy decoding, and the same 31-field output contract. Attribution, tool calling, and letter generation were disabled. The four conditions were:

1. A, a single-prompt baseline with one model call per note.
2. B, decomposed field extraction with cross-field routing, bounded Assessment and Plan retry, and the shared format and schema safeguards.
3. C, condition B plus the semantic, faithfulness, and temporal verification gates.
4. D, condition C plus deterministic inline processing and the complete POST-hook layer used by the full inference harness.

The comparison used adjacent stages, A versus B, B versus C, and C versus D, for 120 blinded pairs. A fresh Codex judge was given only the source note and the two outputs. It did not receive the project instructions, manuscript, prior analyses, Git history, or the private condition mapping. The judge prioritized source faithfulness, then completeness, and used TIE when differences were minor or offsetting. The same final rubric was applied to all blinded pairs before decoding. We report exact two-sided sign tests after excluding ties. These model-generated labels are secondary technical evidence and do not replace physician evaluation. The consolidated judge protocol is provided in Appendix S1. The run occurred on October 1, 2026 (PDT). The Codex judge used `gpt-5.6-sol` with `xhigh` reasoning effort and the `default` service tier.

## 3. Results

### 3.1 Development path and cross-cancer adaptation

The development pool contained 100 notes per cancer type. Smaller subsets were used for faster targeted iterations, while major and final runs covered all 100 notes. The pancreatic sequence comprised nine checkpoints on a fixed 30-note subset and three on the full 100-note development set. Breast-cancer revisions were informed by the physician author and model-development authors. Pancreatic-cancer revisions were made without physician review of the target-domain outputs, using an externally synthesized written rubric, source-grounded Qwen review, and author-controlled implementation.

This sequence produced two types of reuse. Some components were retained unchanged, including conditional verification, temporal distinctions, source attribution, and rules that separate active treatment from plans or supportive medication. Other components required cancer-specific routing, especially disease terminology, regimen interpretation, and post-processing conditions. The resulting system reused the error-handling framework without assuming that breast and pancreatic cancer were clinically interchangeable.

The pancreatic-cancer physician result is the most relevant evidence for this adaptation. Across four independent evaluations, the physicians preferred the harness in 336 field comparisons and the baseline in 91, with 613 ties. The four physicians recorded positive within-note margins in 19, 18, 17, and 13 cases. Their remaining results were one negative; two ties; one negative plus two ties; and four negatives plus three ties, respectively. After pooling all four ratings within each note, 19 pancreatic cases had a positive harness-minus-baseline margin and one had a negative margin. Because no physician reviewed pancreatic outputs during development, this pattern is consistent with reuse of previously codified evaluation criteria and workflow components in a second cancer domain. It does not identify whether the gain came from shared rules, pancreatic-specific revisions, the model-based reviewer, or their combination.

### 3.2 Current physician evaluation

The physician evidence comprises eight evaluator-by-cancer analyses. All four physicians evaluated both cancer types. Across 2,160 required note-field judgments, the harness was preferred 699 times, 1,290 comparisons were ties, and the baseline was preferred 171 times. These correspond to 32.36%, 59.72%, and 7.92% of all judgments. Among the 870 directional judgments, the crude harness share was 80.34% (note-cluster robust 95% CI 76.14% to 83.96%).

| Completed evaluation | Required judgments | Harness | Baseline | Tie | Harness share among directional judgments |
|---|---:|---:|---:|---:|---:|
| Physician 01, breast cancer | 280 | 84 | 22 | 174 | 79.2% |
| Physician 01, pancreatic cancer | 260 | 75 | 14 | 171 | 84.3% |
| Physician 02, breast cancer | 280 | 79 | 8 | 193 | 90.8% |
| Physician 02, pancreatic cancer | 260 | 86 | 9 | 165 | 90.5% |
| Physician 03, breast cancer | 280 | 119 | 24 | 137 | 83.2% |
| Physician 03, pancreatic cancer | 260 | 103 | 30 | 127 | 77.4% |
| Physician 04, breast cancer | 280 | 81 | 26 | 173 | 75.7% |
| Physician 04, pancreatic cancer | 260 | 72 | 38 | 150 | 65.5% |
| **All completed evaluations** | **2,160** | **699** | **171** | **1,290** | **80.34%** |

<div data-rough-figure="2"></div>

***Figure 2.*** *Physician preferences across eight evaluator-by-cancer analyses. Panel A shows the raw three-category distribution among all required judgments: 699 harness preferences (32.36%), 1,290 ties (59.72%), and 171 baseline preferences (7.92%). Panel B shows the harness share after restricting to directional judgments. Across all judgments, the tie-inclusive net preference margin was +24.44 percentage points (note-cluster robust 95% CI +20.42 to +28.47).*

All four physicians independently favored the harness on the breast-cancer set. Their pooled breast result was 363 harness preferences, 80 baseline preferences, and 677 ties. The first physician's per-note result was 18 harness wins, one baseline win, and one tie. The second and third physicians each recorded a harness win in all 20 notes. The fourth recorded 16 harness wins, two baseline wins, and two ties. When the four breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin.

The four pancreatic-cancer evaluations produced 336 harness preferences, 91 baseline preferences, and 613 ties. The harness had a positive pooled margin in 19 pancreatic notes and a negative margin in one.

The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). This is an absolute difference of +29.98 percentage points from equal preference (95% CI +25.75 to +33.63). The breast-cancer estimate was 81.66% (95% CI 76.02% to 86.21%), or +31.66 percentage points (95% CI +26.02 to +36.21). The pancreatic-cancer estimate was 78.15% (95% CI 71.47% to 83.62%), or +28.15 percentage points (95% CI +21.47 to +33.62). The same standardized estimates on the odds scale were 3.995 overall (95% CI 3.124 to 5.110; p=2.67 x 10^-28), 4.452 for breast cancer (95% CI 3.169 to 6.253; p=7.04 x 10^-18), and 3.576 for pancreatic cancer (95% CI 2.505 to 5.105; p=2.30 x 10^-12).

| Scope | Adjusted probability (95% CI) | Difference from 50% (95% CI), pp | OR (95% CI) | p | Directional n |
|---|---:|---:|---:|---:|---:|
| Overall | 79.98% (75.75%-83.63%) | +29.98 (+25.75 to +33.63) | 3.995 (3.124-5.110) | 2.67 x 10^-28 | 870 |
| Breast cancer | 81.66% (76.02%-86.21%) | +31.66 (+26.02 to +36.21) | 4.452 (3.169-6.253) | 7.04 x 10^-18 | 443 |
| Pancreatic cancer | 78.15% (71.47%-83.62%) | +28.15 (+21.47 to +33.62) | 3.576 (2.505-5.105) | 2.30 x 10^-12 | 427 |

The tie-inclusive net preference margin was +24.44 percentage points (note-cluster robust 95% CI +20.42 to +28.47). This measure retains all 2,160 judgments and treats a tie as zero contribution to either system. At the note level, pooled harness-minus-baseline margins were positive in 39 notes and negative in one (two-sided exact sign test, p=7.46 x 10^-11). This is a replication and sensitivity summary, not a separate accuracy endpoint. Four physicians remain insufficient for a precise estimate of variation across the wider physician population.

### 3.3 Inter-rater agreement and core fields

All four physicians rated the same 280 required breast-cancer comparisons and 260 required pancreatic-cancer comparisons. In breast cancer, pairwise exact agreement ranged from 73.2% to 82.9% and Cohen's kappa from 0.511 to 0.646. In pancreatic cancer, pairwise exact agreement ranged from 73.1% to 80.0% and kappa from 0.520 to 0.588. All four physicians gave the same verdict on 176 breast comparisons (62.9%) and 155 pancreatic comparisons (59.6%). A strict three-of-four majority favored the harness in 76 breast and 67 pancreatic comparisons, the baseline in 6 and 7, and tie in 155 and 139. The remaining 43 breast and 47 pancreatic comparisons had no three-vote majority.

<div data-rough-figure="3"></div>

***Figure 3.*** *Pairwise agreement among four physicians in breast cancer and PDAC. Each populated cell reports exact agreement and Cohen's kappa for one evaluator pair; color encodes kappa only. Each breast-cancer pair shared 280 field judgments, and each PDAC pair shared 260.*

Across all completed evaluations, the seven physician-prioritized core categories contributed 1,040 applicable judgments. The harness received 418 preferences, the baseline 77, and 545 were ties. The harness therefore accounted for 84.4% of the 495 directional core judgments, and every core category had a positive aggregate margin.

| Core field | Harness | Baseline | Tie | Net advantage |
|---|---:|---:|---:|---:|
| Active anticancer medications | 132 | 3 | 25 | +129 |
| Stage | 69 | 7 | 84 | +62 |
| Distant metastasis | 23 | 5 | 132 | +18 |
| Regional or overall metastasis | 74 | 26 | 60 | +48 |
| Treatment response | 62 | 12 | 86 | +50 |
| Breast cancer type and receptors | 24 | 14 | 42 | +10 |
| Completed molecular or genetic results | 34 | 10 | 116 | +24 |
| **Overall** | **418** | **77** | **545** | **+341** |

Active anticancer medication produced the largest and most consistent core-field margin. Medication planning, outside the seven core categories, totaled 98 harness preferences, 2 baseline preferences, and 60 ties. Procedure planning was nearly balanced at 21 harness preferences and 20 baseline preferences. Laboratory planning had a modest margin at 24 versus 19. Within the core set, distant metastasis had the smallest normalized margin, followed by breast type and receptor status.

Note-level summaries supported the same direction without treating every field as an independent observation. After pooling physicians within each cancer type, the harness-minus-baseline margin was positive in all 20 breast notes and 19 pancreatic notes; the remaining pancreatic note had a negative margin. Each of the eight evaluator-by-cancer analyses also favored the harness at the aggregate level. These summaries serve as sensitivity analyses alongside the adjusted model.

<div data-rough-figure="4"></div>

***Figure 4.*** *Effect profile across fields, cancers, and notes. (A) The seven physician-prioritized core fields are ordered by normalized harness-minus-baseline margin; bars retain harness, baseline, and tie counts, and the receptor category applies only to breast-cancer notes. (B) Adjusted harness-preference probabilities and 95% confidence intervals are shown for all directional judgments and by cancer, with 50% as equal preference. The right side reports each probability and its absolute difference from 50%. Ties were excluded from the GEE, which was clustered by note and standardized over observed evaluator values; the overall model also included cancer type. (C) Each point is one note-level normalized margin after pooling four physicians, without note identifiers. These panels use different analysis units and report preference, not accuracy. Odds ratios, p values, and directional sample sizes remain in the table and text.*

Detailed note-level margins are shown in Supplementary Figure S1.

### 3.4 Staged ablation isolates the gain from verification

The blinded staged ablation tested whether each added layer improved the immediately preceding condition. Only the verification gates produced a clear overall gain. The conditions were cumulative: A was the single-prompt baseline; B was A plus field decomposition and shared orchestration safeguards; C was B plus semantic, faithfulness, and temporal verification; and D was C plus deterministic inline processing and the complete POST-hook layer.

Adding decomposition and orchestration produced no overall gain. The A versus B comparison was evenly divided, with 17 preferences for A, 17 for B, and 6 ties (exact two-sided sign test, p=1.000). Adding verification produced the clear gain: C was preferred over B in 24 comparisons, B was preferred in 2, and 14 were ties (p=0.0000105). This direction appeared in both cancers.

Adding deterministic rules produced no overall gain. C was preferred in 19 cases, D in 16, and 5 were ties (p=0.736). The direction differed by cancer. Breast cancer favored D over C, 11 to 6 with 3 ties, while pancreatic cancer favored C over D, 13 to 5 with 2 ties. The cancer-specific difference among non-tie judgments was nominally significant by Fisher's exact test (odds ratio 4.77, p=0.0437), but this exploratory comparison does not survive broad multiple-testing correction.

| Adjacent comparison | Earlier stage | Later stage | Tie | Exact sign-test p |
|---|---:|---:|---:|---:|
| A single prompt vs B decomposed extraction | 17 | 17 | 6 | 1.000 |
| B decomposed extraction vs C verification | 2 | 24 | 14 | 0.0000105 |
| C verification vs D full deterministic layer | 19 | 16 | 5 | 0.736 |

<div data-rough-figure="5"></div>

***Figure 5.*** *Blinded staged technical ablation across three cumulative transitions. The stage strip defines A as the single prompt, B as A plus decomposition and orchestration, C as B plus verification, and D as C plus deterministic rules. Each panel shows the added layer and its observed change relative to the immediately preceding condition. Only verification produced a clear overall gain. The deterministic layer favored D in breast cancer and C in PDAC. One non-physician Codex judge produced these secondary technical judgments; exact two-sided sign tests exclude ties.*

The judge's explanations suggested a fidelity-completeness tradeoff in the final transition. The D layer often improved caution around stage and metastatic sites, particularly in breast cancer. In pancreatic cancer it more often omitted current medications, response assessments, or therapy plans. These findings identify targets for hook-level audit rather than establishing that any variant is clinically superior.

Free-text physician comments show that a TIE can mean either that both outputs were acceptable or that both were inadequate; examples and cautious interpretation are provided in Appendix S2.

## 4. Discussion

### 4.1 Main interpretation

The multi-physician evaluation favored the inference harness over the same-model single-prompt baseline. Among directional judgments, the adjusted probability of a harness preference was 79.98% (95% CI 75.75% to 83.63%), which is +29.98 percentage points above equal preference (95% CI +25.75 to +33.63). The corresponding odds ratio was 3.995 (95% CI 3.124 to 5.110; p=2.67 x 10^-28).

The main result concerns the development path around the model. A physician author helped identify recurrent errors in breast cancer. The team converted those errors into an explicit inference harness. A general-purpose external LLM organized the failure history into a written rubric, and a source-grounded Qwen reviewer then supported adaptation to pancreatic cancer without target-domain physician review. The model-development author controlled each retained change. Final physician ratings favored the harness in both domains.

Most field comparisons were ties, which is expected because both systems used the same base model. The raw result was therefore reported in two complementary ways. The directional estimand asks which system was preferred when a physician selected a winner. The tie-inclusive net margin was +24.44 percentage points (95% CI +20.42 to +28.47), so the conclusion did not depend on removing ties. The gains were concentrated in fields that need temporal interpretation or clinical classification. This pattern is consistent with a system that leaves straightforward answers alone and intervenes when a known failure pattern appears.

The staged ablation adds mechanistic evidence but does not replace physician component evaluation. It used one non-physician LLM judge, adjacent rather than full-versus-baseline comparisons, and no independent replicate judge. The physician outcome also retains an ambiguity: a tie can represent either two acceptable outputs or two inadequate outputs. The results therefore support physician preference for the complete presented harness and a technical contribution from verification gates, but not absolute correctness or a clinical causal claim for every individual rule.

### 4.2 Physician-informed extraction as a reusable harness

The physician author's role during breast-cancer development differed from conventional dataset labeling. Rather than producing a large supervised training set, the physician and model-development authors reviewed concrete failures and defined medically meaningful distinctions. Those distinctions were encoded as prompts, verification checks, and deterministic rules. The four independent physician evaluations suggest that the resulting behavior was not limited to the development physician's preferences.

Structured extraction is useful because it converts a long, internally repetitive note into a compact set of fields that can be checked, searched, and reused downstream. It is particularly valuable when the task is tedious for a physician but can be bounded precisely, such as reconciling current anticancer therapy across a medication list and treatment history.

We use the term inference harness because the contribution sits around the model rather than in its weights. The routing, schemas, verification logic, deterministic rules, logging, and attribution layer could in principle be attached to another instruction-following model after interface and prompt calibration. This study tested only Qwen2.5-32B, so cross-model portability remains a design property to evaluate rather than an empirical result.

### 4.3 Agent-assisted cross-cancer adaptation and the role of LLM-as-a-judge

The pancreatic-cancer stage tested whether the accumulated workflow could be adapted beyond the domain in which the physician author gave direct feedback. Pancreatic notes differ in disease course, treatment regimens, staging language, and surgical context. We retained shared clinical distinctions, added cancer-specific routing where needed, and did not ask a physician to review pancreatic outputs during development.

Two model roles supported this adaptation. First, a general-purpose external LLM organized the accumulated breast-cancer failure history and physician-informed distinctions into a written rubric and suggested candidate pancreatic-cancer adaptations. Second, a source-grounded Qwen LLM reviewer compared subsequent outputs with their source notes and localized likely omissions, unsupported claims, semantic mismatches, and temporal errors. The reviewer showed where the adapted harness was succeeding or failing across development rounds.

The reviewer did not define the clinical endpoint or directly modify the deployed workflow. The model-development author selected candidate revisions, while independent physicians supplied the final evidence. General-purpose LLM evaluators have documented order and self-preference biases [16,17], so this separation matters. The pancreatic ratings, 336 harness preferences versus 91 baseline preferences with 613 ties, support the usefulness of the LLM-assisted development loop. They do not establish autonomous self-improvement.

### 4.4 Model difficulty and human-model complementarity

The field-level results distinguish direct extraction from tasks that require temporal or clinical interpretation. Explicit findings often produced ties because both systems could locate them. The larger differences appeared when the same fact had to be classified in context.

Active anticancer medication showed the largest core-field advantage, with 132 harness preferences, 3 baseline preferences, and 25 ties. Medication planning also favored the harness, 98 to 2 with 60 ties. These tasks may be laborious for a person reviewing a long treatment history, but the present study did not measure physician time or burden. The results support structured assistance when current, historical, supportive, and planned therapies are kept separate; efficiency remains a hypothesis for future study.

Stage, metastatic involvement, and treatment response illustrate a different form of complementarity. A physician may resolve an explicitly documented stage quickly, while a model can still confuse regional nodes with distant disease or treat a suspected lesion as confirmed. Response assessment can be difficult for both because it requires aligning findings with the current regimen. Passweg et al. similarly found that locally hosted LLMs were noninferior to human accuracy for metastasis classification from oncology imaging reports but remained inferior for treatment-response assessment [14]. In these fields, the system is better used to organize evidence and flag a proposed answer for rapid verification than to replace medical judgment.

Distant metastasis had the smallest normalized core-field margin, 23 to 5 with 132 ties; breast type and receptor status was next, at 24 to 14 with 42 ties. The large tie fractions are consistent with fields that are often stated explicitly, leaving less room for the harness to improve on the base model. The proposed division of labor is field-dependent: automate repetitive synthesis, surface evidence for judgment-sensitive fields, and let physicians verify compact outputs. This is an interpretation of the preference pattern, not a measured workflow benefit.

<div data-rough-figure="6"></div>

***Figure 6.*** *Results-informed conceptual handoff for structured oncology extraction. Source-linked harness outputs can be routed to direct structuring, evidence reconciliation, or escalation for medical interpretation, with physician oversight across all three lanes. The observed core-field margin range provides context only; the study did not measure time, workload, risk, or clinical acceptability.*

### 4.5 Mechanistic evidence from the staged ablation

The staged ablation narrows the mechanism claim. Moving from the single prompt to decomposed extraction with shared orchestration safeguards produced no overall preference difference. The B condition already included field routing, bounded Assessment and Plan retry, and format and schema repair, so this comparison does not isolate prompt decomposition from every other orchestration change. The clear gain came when semantic, faithfulness, and temporal verification were added. This supports verification as the most reproducible component-level contribution in the current experiment.

The final deterministic layer did not improve the combined C versus D comparison. Its effect depended on cancer type. The judge more often preferred D in breast cancer and C in pancreatic cancer. The written reasons suggest that deterministic rules improved caution around stage and metastatic sites but sometimes removed useful pancreatic-cancer medication, response, or therapy-plan content. This is a practical warning about rule accumulation: a narrow correction can improve fidelity in one setting while reducing completeness elsewhere. The next revision should audit the pancreatic hooks implicated in these omissions rather than treating the complete hook set as uniformly beneficial.

Component comparisons are present in related work but are not universal. Wiest et al. compared plain zero-shot, one-shot, definition-enhanced, and grammar-constrained prompting [3]. Huang et al. used iterative prompt engineering and error analysis for pathology extraction [10]. mCODEGPT compared single-step prompting with two hierarchical strategies [7]. Corso et al. compared zero-shot, few-shot, and expert-annotated few-shot prompts across four locally deployable models [12]. Dubey et al. compared simple prompting, chain-of-thought prompting, double filtering, and a rule-based ECOG extractor [15]. Grothey et al. compared five prompting strategies and quantized model configurations [8]. Dao et al. quantified which initial errors were corrected by its validation and retry loop, while Tariq et al. compared a complete hybrid system with zero-shot, structured-code, and rule-based baselines [5,6]. No study in this set reports a factorial removal of every component in a multi-stage oncology harness.

The current ablation is deliberately technical. It does not show that physicians would make the same component-level choices, and adjacent comparisons do not directly estimate D versus A. The full-harness physician comparison and the staged LLM comparison also used different evaluators and endpoints, so their counts should not be combined. A direct randomized A versus D technical comparison and a smaller physician review restricted to outputs changed by C versus D would provide stronger follow-up evidence.

### 4.6 Relation to prior clinical extraction studies

CORAL established the real-note benchmark used here and showed that zero-shot GPT-4 outperformed GPT-3.5-turbo and FLAN-UL2, while an independent physician still identified omissions and hallucinations [2]. Our study asks a different question on the same benchmark: whether a structured workflow improves a frozen model relative to that same model under a single prompt.

The closest recent oncology studies show three complementary directions. First, prompt design and clinical expertise matter. Huang et al. used 78 lung-cancer pathology reports for prompt development and 774 valid reports for independent testing, followed by an additional test on 191 osteosarcoma reports [10]. Corso et al. found that few-shot examples and clinical expertise improved small-model extraction from Italian oncology records [12], while Dubey et al. reported that chain-of-thought and double-filtering prompts outperformed simple prompting for ECOG extraction [15]. These studies support structured prompting, but their outputs are narrower than the longitudinal field set evaluated here.

Second, other studies provide stronger evidence for scale or absolute correctness. Abhyankar et al. used 700 annotated notes for model development and testing, then applied the resulting staging system to more than two million notes across five cancers [13]. Van Koevorden et al. compared 29 extracted categories from 60 head and neck oncology patients against physician consensus, with six clinical experts assigning error types and impact [11]. Passweg et al. compared five local models with duplicate human extraction and senior physician adjudication on 400 imaging reports [14]. These designs answer whether extraction agrees with a medical reference. Our A/B study instead estimates physician preference between a complete harness and a same-model single-prompt baseline, so it does not replace absolute accuracy assessment.

The statistical presentation also follows precedents from clinical AI comparison studies. Passweg et al. used logistic models with report-cluster robust standard errors and delta-method marginal effects [14]. Van Veen et al. retained an equivalent category in blinded clinical summary comparisons rather than forcing a winner [24]. Ayers et al. reported a preference proportion with a confidence interval [25], while Goh et al. modeled repeated cases and participants with mixed effects [26]. These choices support reporting an interpretable adjusted probability, preserving ties in descriptive analyses, and accounting for repeated judgments.

Third, prior systems support individual engineering components. Wiest et al. evaluated local models and constrained output for binary clinical features [3]. mCODEGPT tested hierarchical prompting on synthetic oncology notes [7]. Grothey et al. compared model, prompt, and quantization configurations in prostate pathology extraction [8]. Dao et al. combined engineered context, validation, and retries for numerical extraction from right-heart-catheterization reports [6]. Tariq et al. achieved larger scale and external validation with a UMLS-plus-fine-tuned-LLM system for breast treatment timelines [5], while Bhattarai et al. compared model families and rule-based methods on longitudinal lung-cancer phenotypes [4].

The distinguishing feature of our evaluation is not the invention of prompts, retries, or rules in isolation. It is the combination of field-specific routing, selective context transfer, verification gates, deterministic oncology rules, logging, and source attribution, evaluated against a same-model baseline through direct field-level comparison by physicians. The breast-to-pancreatic development sequence also tests whether medically informed failure rules can reduce repeated demand on specialist time in a second cancer domain.

The study should therefore be positioned as a focused, physician-reviewed evaluation of an integrated inference harness, not as the first oncology extraction system or the largest validation study. Its narrower contribution is the complete combination of field-specific routing, selective context transfer, verification gates, deterministic oncology rules, logging, and source attribution, evaluated against a same-model baseline through direct field-level comparison by physicians. The detailed study-by-study comparison is retained in the Chinese review appendix rather than in the main manuscript.

### 4.7 Exploratory patient communication

We also explored whether structured extraction could support patient-letter generation. The early review did not show that stronger extraction automatically produced a more preferred patient letter. A second generation step introduces separate decisions about content selection, organization, plain-language phrasing, uncertainty, and patient-facing tone. These elements need their own development and evaluation. We therefore treat patient communication as a downstream research direction rather than a conclusion of the extraction study.

### 4.8 Clinical and technical implications

The findings suggest that some LLM failures in oncology extraction are repeatable enough to address outside the model weights. This is useful when labeled training data are scarce or when a clinical team needs to change a field definition without retraining. A rule that preserves biopsy-pending disease as suspected can be inspected and tested. A prompt-only system offers less control because a wording change may affect unrelated fields.

The term inference harness is appropriate for this system because it includes more than a prompt sequence. It manages task routing, dependencies, verification, deterministic corrections, logging, and source attribution around a frozen model. The term clinical workflow could imply integration into routine care, which this pilot did not test. We therefore use inference harness for the technical contribution and evaluation workflow for the study procedure.

The system is also compatible with local deployment. Local operation does not by itself establish privacy compliance or clinical safety, but it allows an institution to retain control of note processing and system updates. The present work evaluates extraction quality, not readiness for autonomous clinical use.

### 4.9 Study boundaries and next steps

This pilot was intended to establish whether the observed effect is large and medically coherent enough to justify a larger study. Four physicians independently reproduced the direction in both breast and pancreatic cancer. The adjusted probability estimate was 79.98%, with its confidence interval remaining well above equal preference.

The technical ablation separates broad stages but remains limited to one non-physician LLM judge. It does not isolate every component within B, C, or D, does not estimate source attribution, and compares adjacent stages rather than D directly with A. The physician preference scale also combines two meanings of a tie: both outputs may be acceptable, or both may be inadequate. These limitations restrict mechanism and absolute-correctness claims even though the direction of the primary physician result is consistent.

The broader scope also remains limited. Only four physicians participated, CORAL represents one institution and two cancer domains, and another model family has not been tested. The 40 expert-annotated notes formed an untouched holdout test set, but this remains an internal benchmark rather than external validation on a second institution. No physician reviewed pancreatic outputs during development.

The next study should adjudicate a sample of physician ties, run a direct randomized A versus D comparison, audit pancreatic-cancer hooks that may over-prune medications and treatment summaries, add physicians, and test a second model and an external health-system dataset. QUEST emphasizes advance planning, evaluator selection, multidimensional scoring, reliability, and adjudication in healthcare LLM evaluation [19]. The VALID framework further recommends variable-level comparison with expert reference data, internal consistency and plausibility checks, and replication analyses [20]. These frameworks make the remaining gap precise: our current endpoint measures comparative physician preference, not absolute acceptability or field-level accuracy. The model version, prompts, evaluator roles, development use of the benchmark, and all human-oversight steps should also remain explicit, consistent with TRIPOD-LLM reporting guidance [18].

## 5. Conclusion

The current pilot supports a staged development strategy. Human-in-the-loop review converted breast-cancer extraction errors into an explicit inference harness. A general-purpose external LLM and a source-grounded Qwen reviewer then supported pancreatic-cancer adaptation without target-domain physician review, while the model-development author controlled workflow changes and the model weights remained frozen. Independent physicians favored the harness in both cancers. The technical ablation identified verification gates as the clearest component-level gain and found cancer-specific tradeoffs in the deterministic hook layer. Tie adjudication, targeted physician review of component changes, more physicians, and external validation are needed before making a broad confirmatory claim.

## Version history

- **v0.22:** Consolidated the core-field, adjusted-effect, and compact note-distribution results into one main figure, reduced the main figure set to six, and moved the fully labeled note-level plot to Supplementary Figure S1.
- **v0.21:** Standardized study personnel as physicians, moved qualitative comments to Appendix S2, clarified that verification was the only staged addition with a clear overall gain, and aligned all figures to fixed color semantics with a probability-only adjusted-effect graphic.
- **v0.20:** Made the standardized adjusted preference probability the primary effect, added absolute and tie-inclusive effects with note-cluster robust confidence intervals, and redrew the adjusted-effect graphic on probability and odds scales.
- **v0.19:** Folded the Chinese review metadata, consolidated Appendix S1 into one general judge protocol, and redrew the complete figure set with one clinical-AI visual system.
- **v0.18:** Replaced the Introduction with the physician-led medical framing; clarified the development and untouched holdout cohorts; updated all extraction results and figures for four physicians across both cancers; removed the final test-set LLM audit and quantitative letter results; and added the blinded Codex judge instructions.
- **v0.17:** Added the first four-physician breast analysis, three-physician pancreatic analysis, staged technical ablation, and Chinese author-confirmation annotations.

## Appendix S1. General blinded Codex judge protocol

The ablation judge ran on October 1, 2026 (PDT) with model `gpt-5.6-sol`, `xhigh` reasoning effort, and the `default` service tier. The consolidated protocol below records the instructions applied across the full judging run. It preserves the blinding rules, clinical rubric, and output schema while omitting operational file-management language.

### General blinded judging protocol

```text
Act as an independent blinded judge for a medical information-extraction ablation. Do NOT read or use AGENTS.md, CLAUDE.md, any session log, manuscript, analysis file, git history, or any `judge_pairs.private_mapping.jsonl`. Do not try to identify the systems. Use only the public blinded input files listed below and the clinical note contained in each record.

Inputs:
- `<REPO_ROOT>/results/staged_ablation_annotated_20261001_093433/breast/judge/judge_pairs.blinded.jsonl`
- `<REPO_ROOT>/results/staged_ablation_annotated_20261001_093433/pdac/judge/judge_pairs.blinded.jsonl`

Review every record in the blinded input files. Read each note and both outputs completely, one pair at a time. Do not use scripts, regex, automated scoring, or another model to make the clinical judgments. Shell commands may be used only to display one record at a time and to validate or write the output.

For every pair, compare Output A and Output B field by field against the note. Evaluate unsupported claims or hallucination, important omissions, semantic or field mismatch, temporal errors, and clinically misleading wording. Prioritize faithfulness, then completeness. Choose `A` or `B` only when one output has a clinically meaningful overall advantage. Use `TIE` when differences are minor, offsetting, limited to formatting or field placement without clinical consequence, or when neither output is meaningfully better. Do not select a winner merely because the outputs differ.

Write one valid JSON object per pair with keys:
`cancer`, `source_line`, `pair_id`, `verdict`, `confidence` (`high|medium|low`), `a_errors`, `b_errors`, `decisive_fields`, `reason`.
Errors should be short arrays of concrete field-level findings grounded in the note. Preserve blinding throughout the review and do not access the private mapping after judging. After all records have been reviewed, validate the JSON and summarize counts by cancer and comparison, confidence levels, recurring errors, and limitations of the rubric. Do not commit or push.
```

Audit artifact: [full blinded summary](results/staged_ablation_annotated_20261001_093433/JUDGE_CODEX_FULL_BLINDED_SUMMARY.md).

## Appendix S2. Qualitative physician comments and interpretation of ties

Three physician exports contained 14 written comments. These comments clarify a limitation of the A/B outcome: a TIE can mean either that both outputs are acceptable or that both are incomplete or incorrect. In one breast imaging-plan comparison, one physician preferred the harness because the baseline summarized completed findings instead of the plan, whereas another physician judged both outputs inadequate and selected a TIE. A fourth physician marked one medication comparison as a TIE because both outputs were poor and preferred the baseline in one procedure-plan comparison because the harness omitted part of the source sentence.

The comments also identified unsupported receptor status, omitted regional nodal disease, uncertainty about response after a newly started regimen, and comparisons in which neither output was satisfactory. They help interpret the rating scale and provide examples for error analysis. They are not a separate quantitative endpoint, and the small, non-systematic set of comments cannot estimate how often either meaning of TIE occurred.

## Appendix S3. Note-level sensitivity analysis

<div data-supplementary-figure="S1"></div>

***Supplementary Figure S1.*** *Detailed note-level normalized preference margins after pooling all four physicians within each cancer type. Each label identifies one held-out note. The margin is (harness preferences minus baseline preferences) divided by completed judgments for that note. All 20 breast-cancer notes and 19 of 20 PDAC notes had positive margins; one PDAC note had a negative margin. This is a supporting sensitivity analysis, not an accuracy estimate.*

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
12. Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. *Communications Medicine*. 2026. [https://doi.org/10.1038/s43856-026-01790-5](https://doi.org/10.1038/s43856-026-01790-5). <span class="reference-note"><strong>Draft relevance:</strong> This study shows that few-shot design and clinical expertise improve locally deployable models, while our work adds multi-stage verification and deterministic cross-field rules.</span>
13. Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. *JCO Clinical Cancer Informatics*. 2026;10:e2500388. [https://doi.org/10.1200/CCI-25-00388](https://doi.org/10.1200/CCI-25-00388). <span class="reference-note"><strong>Draft relevance:</strong> This study used 700 annotated notes for model development and testing, then processed more than two million notes across five cancers; our contribution instead concerns inference-time controls across a broader schema without weight updates.</span>
14. Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. *JCO Clinical Cancer Informatics*. 2026;10:e2600002. [https://doi.org/10.1200/CCI-26-00002](https://doi.org/10.1200/CCI-26-00002). <span class="reference-note"><strong>Draft relevance:</strong> Its senior-physician-adjudicated comparison found metastasis classification easier than treatment-response assessment, matching the field-dependent difficulty seen in our results.</span>
15. Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. *JCO Clinical Cancer Informatics*. 2026;10:e2500226. [https://doi.org/10.1200/CCI-25-00226](https://doi.org/10.1200/CCI-25-00226). <span class="reference-note"><strong>Draft relevance:</strong> Its comparison of simple prompting, chain-of-thought, double filtering, and rules provides direct evidence that inference design changes oncology extraction quality even when the target is a single field.</span>
16. Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*. 2023;36:46595-46623. [https://doi.org/10.52202/075280-2020](https://doi.org/10.52202/075280-2020). <span class="reference-note"><strong>Draft relevance:</strong> This work supports LLM judges as scalable development tools but documents position, verbosity, self-enhancement, and reasoning biases that preclude treating them as our clinical endpoint.</span>
17. Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*. 2024:9440-9450. [https://doi.org/10.18653/v1/2024.acl-long.511](https://doi.org/10.18653/v1/2024.acl-long.511). <span class="reference-note"><strong>Draft relevance:</strong> This paper isolates positional bias in LLM comparison judgments and reinforces the need for human final evaluation and cautious interpretation of automated review.</span>
18. Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nature Medicine*. 2025;31:60-69. [https://doi.org/10.1038/s41591-024-03425-5](https://doi.org/10.1038/s41591-024-03425-5). <span class="reference-note"><strong>Draft relevance:</strong> TRIPOD-LLM supports explicit reporting of models, prompts, evaluation procedures, task-specific metrics, and human oversight in this pilot.</span>
19. Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. *npj Digital Medicine*. 2024;7:258. [https://doi.org/10.1038/s41746-024-01258-7](https://doi.org/10.1038/s41746-024-01258-7). <span class="reference-note"><strong>Draft relevance:</strong> Its review of 142 studies and QUEST framework directly inform evaluator selection, rating dimensions, reliability assessment, and adjudication for a stronger follow-up study.</span>
20. Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. *JCO Clinical Cancer Informatics*. 2026;10:e2500215. [https://doi.org/10.1200/CCI-25-00215](https://doi.org/10.1200/CCI-25-00215). <span class="reference-note"><strong>Draft relevance:</strong> VALID requires variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication, clarifying why our preference endpoint is not yet an absolute correctness assessment.</span>
21. Kong HJ. Managing Unstructured Big Data in Healthcare System. *Healthcare Informatics Research*. 2019;25(1):1. [https://doi.org/10.4258/hir.2019.25.1.1](https://doi.org/10.4258/hir.2019.25.1.1). <span class="reference-note"><strong>Draft relevance:</strong> This article supports the broad estimate that much healthcare information remains unstructured; the estimate is not oncology-specific.</span>
22. Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. *JAMA Network Open*. 2021;4(7):e2115334. [https://doi.org/10.1001/jamanetworkopen.2021.15334](https://doi.org/10.1001/jamanetworkopen.2021.15334). <span class="reference-note"><strong>Draft relevance:</strong> This study of nearly 3 million outpatient notes documents increasing note length and redundancy and the contribution of copied and templated text.</span>
23. Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. *npj Digital Medicine*. 2025;8:301. [https://doi.org/10.1038/s41746-025-01686-z](https://doi.org/10.1038/s41746-025-01686-z). <span class="reference-note"><strong>Draft relevance:</strong> This study provides a close precedent for human-in-the-loop refinement, error ontology development, and general instructions rather than case-specific fixes.</span>
24. Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. *Nature Medicine*. 2024;30:1134-1142. [https://doi.org/10.1038/s41591-024-02855-5](https://doi.org/10.1038/s41591-024-02855-5). <span class="reference-note"><strong>Draft relevance:</strong> Its blinded reader study used a scale with an equivalent midpoint, supporting retention of ties in comparative clinical evaluation.</span>
25. Ayers JW, Poliak A, Dredze M, et al. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum. *JAMA Internal Medicine*. 2023;183(6):589-596. [https://doi.org/10.1001/jamainternmed.2023.1838](https://doi.org/10.1001/jamainternmed.2023.1838). <span class="reference-note"><strong>Draft relevance:</strong> This physician-rated comparison reported the preference proportion with a confidence interval, providing a clear precedent for probability-scale reporting.</span>
26. Goh E, Gallo R, Hom J, et al. Large Language Model Influence on Diagnostic Reasoning. *JAMA Network Open*. 2024;7(10):e2440969. [https://doi.org/10.1001/jamanetworkopen.2024.40969](https://doi.org/10.1001/jamanetworkopen.2024.40969). <span class="reference-note"><strong>Draft relevance:</strong> Its mixed-effects analysis included participant and case effects, illustrating how repeated clinical observations should be modeled rather than treated as independent.</span>

<div class="language-break" id="chinese-version"></div>

# 从医生指导的 extraction 到 AI 辅助的跨癌种适配：面向肿瘤病历与患者沟通的 inference harness

**供临床合作者审阅的试点报告草稿**

版本 0.22，2026 年 10 月

作者：[TODO]

单位：[TODO]

目标会议及稿件形式：[TODO]

本稿现已纳入四位医生对两个癌种的完整评审。全文按医生建议采用一条主线：先通过 human-in-the-loop 开发乳腺癌 inference harness，再把乳腺癌阶段的 failure history 和 clinical rubric 用于 PDAC，并在没有 target-domain physician 审阅输出的情况下完成 LLM-assisted adaptation。最终由四位医生比较 inference harness 与同模型 single-prompt baseline。调整后的 GEE、四级 staged technical ablation、Figure 1 至 Figure 6、Appendix S1 至 Appendix S3 已纳入 HTML。方括号内容仅用于中文版协作审阅。

供临床审阅的简要说明：两个系统使用相同的语言模型。single-prompt baseline 要求模型通过一次调用提取全部信息。inference harness 则把任务拆分为较小的临床问题，检查模型答案，针对反复出现的错误应用范围明确的肿瘤学规则，并把每项结果与病历中的支持性原文关联起来。

> **术语约定：** 中文审阅稿保留 inference harness、workflow、pipeline、baseline、verification gate、hook 和 LLM-as-a-judge 等英文术语，避免翻译后与代码和实验记录中的名称对不上。

<section class="author-confirmation-index" id="author-confirmation-index">
<div class="author-confirmation-title">作者确认清单</div>
<p><span class="status-badge badge-suggested">最终复核</span><strong>Clinical accuracy 与医学措辞：</strong>投稿前请 physician author 复核临床准确性、临床术语和 Discussion 中的医学 interpretation。</p>
</section>

## 摘要

### 背景

大语言模型可以从临床病历中提取结构化信息，但 single prompt 经常混淆当前治疗与既往治疗、疑似病变与确诊疾病，以及已经完成的检查与未来计划。这类错误不易发现，因为生成文本通常仍然流畅。肿瘤科病历尤其具有挑战性：一份纵向病历往往同时包含病理、影像、治疗史、当前治疗、疗效评估和条件性计划。

### 目的

评估 physician-informed inference harness 能否在不进行微调的情况下改善肿瘤信息 extraction，以及能否在没有 target-domain physician 审阅的情况下适配到第二个癌种。

### 方法

我们围绕 Qwen2.5-32B-Instruct-AWQ 分两个阶段构建 inference harness。乳腺癌阶段采用 human-in-the-loop：physician author 和模型开发作者审阅输出，把重复出现的临床错误转化为 field-specific prompt、conditional verification gate 和 deterministic oncology rule。PDAC 阶段由 general-purpose external LLM 把 failure history 整理成书面 rubric，并提出候选适配方案；source-grounded Qwen reviewer 再识别可能的问题，具体修改由作者决定。该阶段没有 physician 审阅 PDAC 输出，模型权重始终冻结。之后，我们在每个癌种 20 份 untouched holdout test notes 上比较 inference harness 与同模型 single-prompt baseline。四位医生均评估两个癌种。Secondary technical ablation 依次比较 single-prompt baseline、带共享 orchestration safeguards 的 decomposed extraction、加入 verification gates 的版本，以及包含完整 deterministic hooks 的版本。一位保持盲态的 Codex judge 评估了 120 对相邻阶段输出。

### 结果

八组 evaluator-by-cancer 分析共提供 2,160 项必评 note-field judgments：699 项偏好 inference harness（32.36%），1,290 项为 TIE（59.72%），171 项偏好 baseline（7.92%）。在 870 项非平局判断中，偏好 inference harness 的粗比例为 80.34%，note-cluster robust 95% CI 为 76.14% 至 83.96%。调整后的 harness 偏好概率为 79.98%（95% CI 75.75% 至 83.63%），相对 50% 的 absolute difference 为 +29.98 个百分点（95% CI +25.75 至 +33.63）。Breast 的调整后概率为 81.66%（95% CI 76.02% 至 86.21%），PDAC 为 78.15%（95% CI 71.47% 至 83.62%）。次要的 odds-scale estimate 为 3.995（95% CI 3.124 至 5.110；p=2.67 x 10^-28）。保留 TIE 的 net preference margin 为 +24.44 个百分点，note-cluster robust 95% CI 为 +20.42 至 +28.47。合并医生评分后，40 份病历中 39 份的净差为正（exact sign-test p=7.46 x 10^-11）。Technical ablation 中，只有加入 verification gates 后形成清楚的整体偏好，结果为 24 比 2，另有 14 个 TIE，exact p=0.0000105。

> **协作说明：** Abstract 会在全文定稿后最后压缩。当前统计数字已经更新，不再保留未完成分析的占位内容。

### 结论

结果支持一条分阶段开发路线。医生反馈先形成乳腺癌 inference harness。随后，LLM-assisted 且由作者监督的循环把已有 rubric 和 failure rules 适配到 PDAC，期间没有 target-domain physician 审阅。Technical ablation 将最清楚的增益定位在 verification gates，deterministic hook layer 则表现出癌种相关的取舍。更广泛的结论需要 TIE adjudication、针对 component changes 的医生复核、更多医生和 external validation。

## 1. 引言

大语言模型（LLM）越来越多地用于临床文本任务，包括临床文书、信息检索和患者沟通。很多应用都依赖一个前置步骤：先从电子健康记录中找出与当前任务有关的信息。只有完成可靠的 extraction，肿瘤病历才能进一步用于总结、分析或面向其他读者改写 [1]。

虽然 EHR 包含结构化字段，大量有临床价值的信息仍保留在 free-text note 中 [21]。肿瘤病历尤其困难，因为同一份病历会把诊断、病理、既往治疗、未来计划、疾病状态变化和不确定信息放在一条很长的时间线上。相关事实可能分散在多个 section，也可能从早期就诊记录 copy forward 到当前病历。一项覆盖近 300 万份门诊病历的研究发现，十年间病历长度和冗余度明显增加，复制内容和模板内容都与这种增长有关 [22]。因此，extraction system 不能只找到医学相关词句，还要判断哪些内容描述患者当前的疾病和治疗。

LLM 可以辅助 extraction，但流畅的输出仍可能遗漏事实、混淆历史与当前发现，或加入原文不支持的结论。若 protected health information 被发送到外部托管服务，commercial model 还会带来数据治理和隐私问题。本地部署的 open-weight model 可以让病历处理留在机构控制的基础设施内，但本地部署本身不能消除 extraction error，也不能自动证明系统符合隐私要求 [3]。

Extraction performance 取决于模型本身，也取决于模型的使用方式。Fine-tuning 可以让模型适应临床任务，但需要标注数据并修改模型权重。Prompt design、task decomposition、validation 和 retry logic 则可以在冻结权重的前提下改进模型。迭代式 prompting 已经改善了 oncology extraction，但 staging 和专科术语仍是常见难点 [10]。Human-in-the-loop 研究也表明，研究者可以把重复错误整理成 error ontology，再转化为通用规则，而不是只修某一个病例 [23]。现有肿瘤研究已经验证了不少单独组件，但对于 integrated inference workflow，以及 practicing physician 对 integrated workflow 和 same-model baseline 的直接比较，现有证据仍然有限 [1]。

本文用 inference harness 指围绕冻结 LLM 的执行层，包括 field-specific task routing、selective context transfer、conditional verification、deterministic medical rules、logging，以及在可用时由模型选择 supporting excerpts。我们先由 physician author 和模型开发作者共同审阅乳腺癌输出并开发 harness。随后，在没有 physician 审阅目标癌种输出的情况下，我们使用 general-purpose external LLM 把既往 failure history 整理成书面 rubric，再用 source-grounded LLM reviewer 识别 PDAC 输出中的候选问题。最终由独立医生比较完整 harness 与同模型 single-prompt baseline。本研究评估 integrated workflow 能否改善 structured extraction，以及这套开发方式能否复用于第二个癌种。

## 2. 方法

### 2.1 研究设计

本研究包括三个连续环节。第一步，physician author 与模型开发作者通过 human-in-the-loop review 开发乳腺癌 inference harness。第二步，把乳腺癌 failure history 和医学区分用于 model-assisted PDAC adaptation，期间没有 physician 审阅 PDAC 输出。第三步，在 40 份经专家标注的 CORAL untouched holdout test notes 上，将最终 inference harness 与使用同一冻结模型和相同字段定义的 single-prompt baseline 进行比较，并由四位医生对两个癌种完成隐藏系统标签的 A/B 评估。

乳腺癌阶段检验医生反馈能否转化为可复用的 prompt、conditional gate 和 rule。PDAC 阶段检验 LLM-assisted review loop 能否在没有 target-domain physician 输入的情况下复用这些知识。Extraction comparison 在固定基础模型和目标字段的条件下评估完整 inference harness。Secondary staged technical ablation 使用 blinded LLM review 比较四个累积的 inference conditions。

### 2.2 数据集

我们使用 CORAL 数据集。该数据集通过 PhysioNet 受控开放，包含来自加州大学旧金山分校、经过去标识化处理的肿瘤内科随访病历 [2]。Development pool 包括每个癌种 100 份没有专家 annotation 的病历。Untouched holdout test set 包括每个癌种 20 份 expert-annotated notes。Expert annotation 没有输入 extraction model、开发阶段的 LLM reviewer，也没有通过 A/B 界面提供给最终评估者。

这些记录是真实临床病历，而不是网络问答、合成病例或模型生成文本。病历保留了重复病史、复制到后续记录的内容、不确定检查结果和条件性计划，这些特点正是纵向肿瘤信息提取的难点。该数据集对本研究的主要价值是临床真实性和专家标注，而不是规模。数据访问需要完成 PhysioNet 规定的账号认证和数据使用流程。

每个癌种的 100 份 development notes 都属于开发数据。问题仍较常见时，小 subset 用于加快 targeted iteration。Major 和 final development runs 覆盖全部 100 份，用于发现低频问题和检查 regression。每个癌种的 20 份 annotated notes 一直保留为 untouched holdout test set，只用于本文报告的 evaluation。研究未进行模型权重训练或 fine-tuning。

### 2.3 基础模型与 baseline

两个实验条件均使用通过 vLLM 在本地部署的 Qwen2.5-32B-Instruct-AWQ [9]。baseline 系统对每份病历只调用模型一次，并返回完整的目标 schema。它不使用任务拆分、verification gate、重试、词典或确定性后处理。医生评审包中使用的是最终确定的 baseline 版本，它与 inference harness 采用相同的字段定义和输出契约。

### 2.4 inference harness

不同目标字段对上下文和推理的需求并不相同，因此 harness 采用多个处理阶段。

首先，字段专用提示分别提取就诊背景、癌症诊断、实验室结果、客观发现、当前用药和近期治疗变化。依赖型提示随后接收经过选择的前序结果。例如，分期和转移状态为治疗意图提供背景，当前治疗和临床发现则为疗效评估提供背景。计划类字段主要从 Assessment and Plan 章节提取。Harness 先使用 deterministic pattern 查找该章节；如果失败，则最多进行三次 model-based copy attempt，并逐次将候选结果与源病历核对。三次均失败时，系统跳过依赖该章节的 plan prompts，不生成替代文本。

只有符合触发条件的输出才进入相应 verification stage。各 stage 按需执行：

1. JSON 解析失败时修复格式。
2. 验证 schema，发现错误字段名或泄漏的字段名。
3. 检查具体性和语义一致性，识别含糊或答非所问的答案。
4. 进行忠实度修剪，删除明确无依据或相互矛盾的内容，同时保留有支持的信息。
5. 进行时态过滤，从未来计划字段中删除已完成或历史事件。

确定性处理层使用高置信度临床规则解决反复出现的错误。这些规则区分区域淋巴结和远处疾病、疑似和确诊转移、当前抗癌治疗与支持治疗或居家用药，以及当前疗效与治疗前变化。系统还在可用时返回 model-selected supporting excerpts。

| inference harness 组件 | 针对的失败模式 | 示例 |
|---|---|---|
| 字段专用提取 | 一个大型提示遗漏或混合字段 | 分开提取当前用药和治疗计划 |
| Bounded retry 与 fallback | 无法可靠恢复目标病历章节 | 最多重试三次 Assessment and Plan extraction，失败后跳过 plan-only prompts |
| 依赖感知的上下文 | 相关字段相互矛盾 | 解释分期和疗效时使用转移状态 |
| 语义验证 | 答案与医学相关，但没有回答目标字段 | 从当前疗效中删除未来治疗计划 |
| 忠实度修剪 | 模型增加无依据的结论 | 活检结果待定时，将病灶保留为疑似 |
| 时态过滤 | 已完成的结果出现在未来计划中 | 从影像计划中删除已完成的扫描 |
| 药物与上下文规则 | 仅根据药名分类，忽视临床背景 | 区分抗癌药、居家用药和支持药物 |
| 跨字段临床规则 | 分期、淋巴结和远处疾病不一致 | 将腋窝淋巴结保留为区域受累，而非远处转移 |
| source attribution | 审查者无法追溯提取值 | 在可用时返回 model-selected supporting excerpt |

<div data-rough-figure="1"></div>

***图 1.*** *开发、迁移与评估流程。（A）breast 和 PDAC development pool 与锁定的 40 份 holdout notes 分开。（B）physician author 和模型开发作者对乳腺癌输出进行 source review，并据此调整 prompt、gate、hook 和 regression test。（C）乳腺癌 failure history、clinical rubric 和 reusable harness 用于 PDAC；source-grounded LLM reviewer 提供候选问题，修改由作者决定，期间没有 target-domain physician 审阅输出。（D）四位医生在锁定 holdout set 上，对 inference harness 与同模型 single-prompt baseline 完成隐藏系统身份的逐字段比较。*

### 2.5 医生指导的乳腺癌开发

乳腺癌开发使用全部 100 份 development notes。早期和 targeted iterations 使用较小 subset 缩短反馈周期，major 和 final runs 则覆盖全部 100 份，用于发现低频问题并检查 regression。一位熟悉肿瘤学的 physician author 与模型开发作者共同审阅部分输出。这位 physician author 不是 oncology physician，也没有参与最终评估。作者将发现的问题与源病历核对，对重复错误进行归类，再转化为通用的 prompt、verification logic 或 deterministic rule。该过程没有建立 supervised training set，也没有更新模型权重。

Targeted iterations 检查某项修改能否解决目标问题。Major 和 final runs 使用完整 development set 评估整体版本并检查 regression。这里的目标是编码重复出现的临床区分，而不是修正单个病例。例如，腋窝淋巴结受累被误判为远处转移后，团队加入通用的区域淋巴结规则；计划使用的药物被误列为当前治疗后，则加入药物时态规则。

> **协作说明：** 2.1 保留为研究设计总览；2.5 单独描述乳腺癌阶段的具体开发程序。两节不是同一层级的信息，因此不建议合并。

### 2.6 AI-assisted 的胰腺癌适配

随后，我们通过 12 个已归档的 development checkpoints 将乳腺癌 inference harness 适配到 PDAC。其中 9 个 checkpoints 使用固定的 30 份病历子集，以便更快完成 targeted iteration；另外 3 个使用完整的 100 份 development notes。在这一阶段，没有 physician 审阅 PDAC 输出。General-purpose external LLM 把乳腺癌阶段累积的 failure history 和临床区分整理成书面 rubric，并提出候选 PDAC adaptation。Qwen pipeline 生成 structured fields，再由 source-grounded Qwen reviewer 将输出与源病历比较。模型开发作者审查 error flags 和候选方案，并决定具体实施哪些修改。Major 和 final runs 覆盖全部 100 份 development notes。

审查提示包含字段定义、严重程度标准，以及乳腺癌开发期间确定的临床区分。已有错误类别可以指导新癌种的审查，reviewer 也能发现 PDAC 特有的问题，例如治疗方案名称和剂量表达。这个 LLM-assisted loop 修改的是 prompt、rule 和 workflow code，模型权重没有变化。作者控制具体实施，所以本文将其称为 model-assisted、author-supervised adaptation。

在两个开发阶段，pipeline 都记录原始模型输出、每项验证操作和确定性修正。这些记录使团队能够追溯最终字段值在 harness 中的处理过程，并根据测试结果保留或拒绝修改建议。

### 2.7 由医生确定的核心临床字段

在评估 test set 之前，参与开发的 physician author 根据理解疾病状态和当前治疗决策的重要性，指定了七个核心临床字段：

1. 患者当前正在接受哪些抗癌药物？
2. 当前癌症分期是什么？
3. 是否存在远处转移，远处转移是存在、不存在还是不确定，涉及哪些部位？
4. 有哪些区域或总体转移受累得到证据支持？
5. 癌症对当前治疗的反应如何？
6. 乳腺癌类型及 ER/PR/HER2 状态是什么？
7. 病历记录了哪些已经完成的分子或遗传检测结果？

Metastatic involvement 与 distant metastasis 被有意设计为两个独立问题。Breast cancer type 与 receptor status 则被有意合并为一个 breast-specific question。最终的医生评估工具还包括遗传检测计划、支持用药、操作计划、影像计划、实验室检查计划、用药计划和近期治疗变化。实验室结果摘要和一般临床发现为选评字段，不纳入主要医生分析。

### 2.8 开发阶段的 LLM 辅助审查

LLM 辅助审查只用于开发阶段，尤其服务于 PDAC adaptation。针对每个候选输出，reviewer model 依据字段定义和严重程度标准，将提取结果与源病历比较，并标记可能的遗漏、无依据内容、semantic mismatch 和 temporal error。已有研究表明，LLM judge 可以在一定程度上接近人工偏好，但也会受到 position、verbosity 和 self-preference bias 的影响 [16,17]。模型开发作者审阅这些 flags 并决定是否修改。LLM reviewer 不提供本文报告的 test-set outcomes，也不能替代独立的医生评估。

> **协作说明：** 这里不再罗列“发现了哪些错误、之后修了什么”。这些内容更像开发报告。Methods 只说明 LLM 审查的用途、边界和它与最终医生评估的区别。

### 2.9 医生评估

临床评估通过隐藏系统标签的 A/B 界面展示源病历和两份结构化输出。评估者针对每个字段选择 A 更优、B 更优或平局。他们不知道哪份输出来自 inference harness，也不知道两个系统是否使用同一基础模型或研究预期哪一侧更好。所有评估采用固定映射，inference harness 显示为 A，single-prompt baseline 显示为 B。当 model-selected supporting excerpts 可用时，source attribution 作为完整 harness 输出的一部分随 A 展示。因此，本研究比较的是呈现给医生的完整系统输出，没有单独隔离 attribution 的作用。

参与乳腺癌开发的 physician author 不属于最终评估者。四位医生均独立评估了乳腺癌和 PDAC 输出。

四位医生均完成了同样的 280 项乳腺癌必评比较和 260 项 PDAC 必评比较。我们按癌种计算每两位医生之间的完全一致率和 Cohen's kappa，并分别汇总八组 evaluator-by-cancer analyses。合并计数仅作描述性统计，不把各字段判断视为相互独立的观测。

Primary estimand 定义为 P(harness preference | directional judgment)。只有该方向性分析排除 TIE。我们使用总体平均的 logistic GEE，并在每份病历内采用 exchangeable working correlation。模型将 evaluator 作为固定效应，总体模型同时校正 cancer type；癌种分层模型只校正 evaluator。我们在观测到的 covariate distribution 上标准化 fitted probabilities，并在 logit scale 上用 delta method 计算 95% CI。调整后偏好概率是 primary effect；它相对 50% 的差值是 absolute effect，对应的 OR 是次要 model-scale effect。Passweg 等在 oncology extraction 中使用了相近的 report-cluster robust logistic model 和 delta-method marginal effects [14]。其他临床比较研究保留了 equivalent response category [24]、报告带 CI 的 preference proportion [25]，或用 mixed-effects model 处理 case 和 participant 的重复观测 [26]。

描述性分析保留完整的三分类结果。粗方向性比例使用按病历聚类的 sandwich CI。Tie-inclusive sensitivity analysis 将 harness、TIE 和 baseline 分别编码为 1、0 和 -1，其均值即 net preference margin。该指标的 normal 95% CI 使用按病历聚类的 sandwich standard error，不使用 bootstrap。我们还把字段判断合并到病历层面，并对正向与负向 note margin 使用双侧 exact sign test。

### 2.10 Staged technical ablation

我们在同一组 20 份乳腺癌和 20 份 PDAC benchmark 病历上完成 secondary technical ablation。四个条件均使用 Qwen2.5-32B-Instruct-AWQ、greedy decoding 和相同的 31-field output contract。实验关闭 attribution、tool calling 和 letter generation。四个条件为：

1. A：single-prompt baseline，每份病历只调用一次模型。
2. B：decomposed field extraction，包含 cross-field routing、bounded Assessment and Plan retry，以及共享的 format 和 schema safeguards。
3. C：在 B 上增加 semantic、faithfulness 和 temporal verification gates。
4. D：在 C 上增加 deterministic inline processing 和 full inference harness 使用的完整 POST-hook layer。

实验比较相邻阶段 A vs B、B vs C 和 C vs D，共形成 120 个 blinded pairs。一个全新 Codex judge 只能看到源病历与两份隐藏条件标签的输出，不能读取项目说明、稿件、既有分析、Git history 或 private mapping。判断优先考虑 source faithfulness，其次考虑 completeness；差异较小或相互抵消时判为 TIE。解盲前，全部 blinded pairs 均使用同一版最终 rubric。统计使用排除 TIE 后的双侧 exact sign test。该结果属于 secondary technical evidence，不能替代医生评估。合并后的 judge protocol 收录于 Appendix S1。运行日期为 2026 年 10 月 1 日（PDT）。Codex judge 使用 `gpt-5.6-sol`，reasoning effort 为 `xhigh`，service tier 为 `default`。

> **协作说明：** B 不能严格写成纯粹的 decomposition-only。它已经包含 A/P 的三次 bounded retry、cross-field routing 和 G1/G2 format/schema safeguards。因此，A vs B 衡量的是 decomposition 加基础 orchestration 的组合，不是单独一个 prompt 拆分动作。

## 3. 结果

### 3.1 开发路径与跨癌种适配

Development pool 包括每个癌种 100 份病历。较小 subset 用于加快 targeted iterations，major 和 final runs 则覆盖全部 100 份。PDAC 序列包括固定 30 份病历子集上的 9 个 checkpoints，以及完整 100 份 development notes 上的 3 个 checkpoints。乳腺癌部分的修订来自 physician author 与模型开发作者的共同审阅。PDAC 部分没有 physician 审查相应输出，采用 externally synthesized written rubric、source-grounded Qwen review，以及由作者控制的实施流程。

开发中有一部分组件可以原样迁移，包括 conditional verification、时态区分、source attribution，以及区分当前治疗、治疗计划和支持性用药的规则。疾病术语、治疗方案解读和后处理条件则需要按癌种分别处理。因此，该系统复用了 failure-handling workflow，但没有假设乳腺癌与胰腺癌在临床上可以互换。

PDAC 的医生评估是检验这种适配的主要证据。在四次独立评估中，医生有 336 次偏好 inference harness、91 次偏好 baseline，另有 613 次平局。四位医生按病历计算分别得到 19、18、17 和 13 例正向净差；其余结果分别为 1 例负向、2 例平局、1 例负向加 2 例平局，以及 4 例负向加 3 例平局。合并四位医生的判断后，19 例 PDAC 病例的 harness 减 baseline 净差为正值，1 例为负值。由于开发期间没有 physician 审查 PDAC 输出，这一模式与此前编码的评估标准和 workflow 可以跨癌种复用相一致。但仅凭该结果，无法判断收益来自共享规则、PDAC-specific revisions、model-based reviewer，还是这些因素的共同作用。

### 3.2 当前医生评估

目前的医生证据包括八组 evaluator-by-cancer analyses，四位医生均完成两个癌种的评估。在 2,160 项必评 note-field judgments 中，harness 获偏好 699 次，1,290 次为 TIE，baseline 获偏好 171 次，分别占全部判断的 32.36%、59.72% 和 7.92%。在 870 项非平局判断中，偏好 harness 的粗比例为 80.34%，note-cluster robust 95% CI 为 76.14% 至 83.96%。

| 已完成的评估 | 必评判断数 | harness | baseline | 平局 | harness 在非平局判断中的占比 |
|---|---:|---:|---:|---:|---:|
| 医生 01，乳腺癌 | 280 | 84 | 22 | 174 | 79.2% |
| 医生 01，胰腺癌 | 260 | 75 | 14 | 171 | 84.3% |
| 医生 02，乳腺癌 | 280 | 79 | 8 | 193 | 90.8% |
| 医生 02，胰腺癌 | 260 | 86 | 9 | 165 | 90.5% |
| 医生 03，乳腺癌 | 280 | 119 | 24 | 137 | 83.2% |
| 医生 03，胰腺癌 | 260 | 103 | 30 | 127 | 77.4% |
| 医生 04，乳腺癌 | 280 | 81 | 26 | 173 | 75.7% |
| 医生 04，胰腺癌 | 260 | 72 | 38 | 150 | 65.5% |
| **所有已完成的评估** | **2,160** | **699** | **171** | **1,290** | **80.34%** |

<div data-rough-figure="2"></div>

***图 2.*** *八组 evaluator-by-cancer analyses 的医生偏好。（A）全部必评 judgments 中的原始三分类分布：699 项偏好 inference harness（32.36%），1,290 项为 TIE（59.72%），171 项偏好 baseline（7.92%）。（B）只保留非平局 judgments 后，偏好 inference harness 的比例。保留全部判断时，tie-inclusive net preference margin 为 +24.44 个百分点，note-cluster robust 95% CI 为 +20.42 至 +28.47。*

四位医生在乳腺癌数据集上均独立偏好 harness。合并四位医生的乳腺癌评估后，harness 获偏好 363 次，baseline 获偏好 80 次，另有 677 次平局。第一位医生按病历评估的结果为 harness 胜出 18 例、baseline 胜出 1 例、平局 1 例。第二位和第三位医生均在 20 份病历中全部偏向 harness。第四位医生为 harness 胜出 16 例、baseline 胜出 2 例、平局 2 例。将四位医生对每份乳腺癌病历的评分合并后，20 份病历的 harness 减 baseline 净差均为正值。

四次 PDAC 评估合计包括 336 项 harness 偏好、91 项 baseline 偏好和 613 项平局。合并四位医生的判断后，19 份 PDAC 病历的 harness 减 baseline 净差为正值，1 份为负值。

在非平局判断中，调整后 harness 偏好概率为 79.98%（95% CI 75.75% 至 83.63%）。它相对 equal preference 的 absolute difference 为 +29.98 个百分点（95% CI +25.75 至 +33.63）。Breast 的调整后概率为 81.66%（95% CI 76.02% 至 86.21%），absolute difference 为 +31.66 个百分点（95% CI +26.02 至 +36.21）。PDAC 的调整后概率为 78.15%（95% CI 71.47% 至 83.62%），absolute difference 为 +28.15 个百分点（95% CI +21.47 至 +33.62）。同一组标准化结果在 odds scale 上分别为总体 OR 3.995（95% CI 3.124 至 5.110；p=2.67 x 10^-28）、breast OR 4.452（95% CI 3.169 至 6.253；p=7.04 x 10^-18）和 PDAC OR 3.576（95% CI 2.505 至 5.105；p=2.30 x 10^-12）。

| 范围 | 调整后概率（95% CI） | 相对 50% 的差值（95% CI），百分点 | OR（95% CI） | p | 非平局 n |
|---|---:|---:|---:|---:|---:|
| 总体 | 79.98%（75.75%-83.63%） | +29.98（+25.75 至 +33.63） | 3.995（3.124-5.110） | 2.67 x 10^-28 | 870 |
| Breast | 81.66%（76.02%-86.21%） | +31.66（+26.02 至 +36.21） | 4.452（3.169-6.253） | 7.04 x 10^-18 | 443 |
| PDAC | 78.15%（71.47%-83.62%） | +28.15（+21.47 至 +33.62） | 3.576（2.505-5.105） | 2.30 x 10^-12 | 427 |

保留全部 2,160 项判断时，tie-inclusive net preference margin 为 +24.44 个百分点，note-cluster robust 95% CI 为 +20.42 至 +28.47；TIE 对两个系统的贡献均记为零。病历级敏感性分析中，合并医生评分后 39 份病历的 harness-minus-baseline margin 为正，1 份为负，双侧 exact sign test p=7.46 x 10^-11。这是 replication 和 sensitivity summary，不是独立的 accuracy endpoint。四位医生仍不足以精确估计更广泛医生群体中的差异。

### 3.3 评审者间一致性与核心字段

四位医生均完成了同样的 280 项乳腺癌必评比较和 260 项 PDAC 必评比较。Breast 的六组两两完全一致率为 73.2% 至 82.9%，Cohen's kappa 为 0.511 至 0.646；PDAC 的一致率为 73.1% 至 80.0%，kappa 为 0.520 至 0.588。四位医生在 176 项 breast comparisons 中给出相同判断，占 62.9%；在 155 项 PDAC comparisons 中给出相同判断，占 59.6%。采用至少三票的严格多数标准，breast 中 76 项偏向 harness、6 项偏向 baseline、155 项为平局，43 项没有三票多数；PDAC 中相应为 67、7、139 和 47 项。

<div data-rough-figure="3"></div>

***图 3.*** *四位医生在 breast 和 PDAC 中的两两一致性。每个有数值的 cell 对应一对医生，同时报告 exact agreement 和 Cohen's kappa；颜色只编码 kappa。每对医生共同评估 280 项 breast field judgments 和 260 项 PDAC field judgments。*

> **协作说明：** 这里严格来说不是 correlation，而是 agreement。相关性只能说明两个人的评分变化方向相似，即使其中一人系统性地更偏好某一选项也可能很高；完全一致率和 Cohen's kappa 更适合回答“医生是否给出相同判断”。

在所有已完成的评估中，七个由医生确定的核心类别共有 1,040 项适用判断。harness 获偏好 418 次，baseline 获偏好 77 次，另有 545 次平局。因此，在 495 项核心字段的非平局判断中，harness 占 84.4%，且每个核心类别的汇总净差均为正值。

| 核心字段 | harness | baseline | 平局 | 净优势 |
|---|---:|---:|---:|---:|
| 当前抗癌药物 | 132 | 3 | 25 | +129 |
| 分期 | 69 | 7 | 84 | +62 |
| 远处转移 | 23 | 5 | 132 | +18 |
| 区域或总体转移 | 74 | 26 | 60 | +48 |
| 治疗反应 | 62 | 12 | 86 | +50 |
| 乳腺癌类型与受体状态 | 24 | 14 | 42 | +10 |
| 已完成的分子或遗传检测结果 | 34 | 10 | 116 | +24 |
| **总体** | **418** | **77** | **545** | **+341** |

当前抗癌药物的优势最大，也最为稳定。在七个核心类别之外，药物计划共有 98 次偏好 harness、2 次偏好 baseline 和 60 次平局。操作或手术计划接近平衡，为 21 比 20。检验计划的差距也较小，为 24 比 19。七个核心类别中，distant metastasis 的标准化净差最小，其次是乳腺癌类型与受体状态。

病历级汇总也得出了相同方向的结果，同时避免将每个字段视为相互独立的观察值。按癌种合并医生判断后，20 份乳腺癌病历和 19 份 PDAC 病历的 harness 减 baseline 净差为正值，另有 1 份 PDAC 病历为负值。八组 evaluator-by-cancer analyses 的汇总结果也全部偏向 harness。这些结果作为调整后模型的敏感性分析。

<div data-rough-figure="4"></div>

***图 4.*** *字段、癌种与病历三个层面的 effect profile。（A）七个由医生确定的核心字段按标准化 harness-minus-baseline margin 排序；bar 保留 harness、baseline 和 TIE 的计数，受体类别只适用于 breast notes。（B）总体及分癌种的非平局判断显示调整后 harness 偏好概率及 95% CI，50% 表示 equal preference；右侧报告各概率及其相对 50% 的 absolute difference。GEE 按病历聚类，并在观测到的 evaluator values 上标准化；总体模型还包括 cancer type。（C）每个点表示合并四位医生评分后的一份 note-level normalized margin，不显示 note ID。三个 panel 的分析单位不同，结果均表示偏好而不是 accuracy。OR、p value 和 directional n 保留在相邻表格与正文。*

完整的 note-level margins 见 Supplementary Figure S1。

### 3.4 Staged ablation 将清楚增益定位在 verification

Blinded staged ablation 检验每一项新增层相对于紧邻前一条件是否改善。只有 verification gates 产生清楚的总体增益。四个条件逐层累积：A 是 single-prompt baseline；B 是 A 加 field decomposition 和共享 orchestration safeguards；C 是 B 加 semantic、faithfulness 和 temporal verification；D 是 C 加 deterministic inline processing 和完整 POST-hook layer。

增加 decomposition 和 orchestration 没有形成总体增益。A vs B 完全持平：A 获偏好 17 次，B 获偏好 17 次，另有 6 个 TIE，双侧 exact sign test 的 p=1.000。增加 verification 后出现清楚增益：C 获偏好 24 次，B 获偏好 2 次，另有 14 个 TIE，p=0.0000105。breast 和 PDAC 的方向一致。

增加 deterministic rules 也没有形成总体增益。C 获偏好 19 次，D 获偏好 16 次，另有 5 个 TIE，p=0.736。两个癌种的方向不同：breast 以 11 比 6 偏向 D，另有 3 个 TIE；PDAC 则以 13 比 5 偏向 C，另有 2 个 TIE。排除 TIE 后，癌种之间的方向差异在 Fisher exact test 中得到 nominal p=0.0437，OR=4.77，但这一 exploratory comparison 无法通过更广泛的 multiple-testing correction。

| 相邻阶段比较 | 前一阶段更优 | 后一阶段更优 | TIE | Exact sign-test p |
|---|---:|---:|---:|---:|
| A single prompt vs B decomposed extraction | 17 | 17 | 6 | 1.000 |
| B decomposed extraction vs C verification | 2 | 24 | 14 | 0.0000105 |
| C verification vs D full deterministic layer | 19 | 16 | 5 | 0.736 |

<div data-rough-figure="5"></div>

***图 5.*** *三个累积 transition 的 blinded staged technical ablation。顶部 stage strip 定义 A 为 single prompt，B 为 A 加 decomposition 和 orchestration，C 为 B 加 verification，D 为 C 加 deterministic rules。每个 panel 都显示新增层及其相对于紧邻前一条件的变化。Verification 是唯一具有清楚总体优势的 transition。Deterministic layer 在 breast 中偏向 D，在 PDAC 中偏向 C。全部判断来自一位非医生 Codex judge，只能作为 secondary technical evidence；双侧 exact sign test 排除 TIE。*

Judge 的文字理由提示，最后一步存在 fidelity 与 completeness 的取舍。D 在 breast 中更常改善 stage 和 metastatic site 的谨慎表达；在 PDAC 中，它更常遗漏 current medications、response assessment 或 therapy plan。这个结果适合用于定位需要审计的 hooks，不能证明任何 variant 在临床上更优。

> **协作说明：** 这张图最重要的不是证明 D 一定最好，而是把故事收紧：真正稳定的 component gain 来自 verification gates。完整 hooks 在 breast 和 PDAC 上方向相反，说明 rules 需要按癌种审计，不能把 rule 越多直接等同于系统越好。

医生自由文本评论显示，TIE 既可能表示两个输出都可接受，也可能表示两个输出都不充分；具体案例和谨慎解释见 Appendix S2。

## 4. 讨论

### 4.1 主要解读

多位医生参与的评估中，inference harness 相对于 same-model single-prompt baseline 获得更高偏好。在非平局判断中，调整后 harness 偏好概率为 79.98%（95% CI 75.75% 至 83.63%），相对 equal preference 高 +29.98 个百分点（95% CI +25.75 至 +33.63）。对应 OR 为 3.995（95% CI 3.124 至 5.110；p=2.67 x 10^-28）。

本文的主要结果来自模型外围的开发路径。Physician author 先帮助识别乳腺癌中反复出现的错误，团队再把这些错误转成明确的 inference harness。之后，general-purpose external LLM 把 failure history 整理成书面 rubric，source-grounded Qwen reviewer 支持 PDAC adaptation，期间没有 target-domain physician 审阅，具体修改由模型开发作者控制。两个癌种的最终医生评分都更偏向 inference harness。

多数字段比较为平局，这是可以预期的，因为两个系统使用同一个能力较强的基础模型。因此，我们用两种互补口径报告原始结果。Directional estimand 回答医生明确选出一侧时更偏好哪个系统；保留所有判断的 net preference margin 为 +24.44 个百分点（95% CI +20.42 至 +28.47），说明结论不依赖删除 TIE。优势主要集中在需要时间关系判断或临床分类的字段。这符合系统的设计思路：保留基础模型已经答对的简单问题，只在出现已知失败模式时介入。

Staged ablation 增加了机制证据，但不能代替医生对各组件的临床评估。该实验只有一位非临床 LLM judge，比较的是相邻阶段，也没有独立的第二位 judge。医生评估中的 `TIE` 仍有两种含义：两个输出都可以接受，或者两个输出都不理想。因此，现有结果支持医生对完整 harness 输出的相对偏好，也支持 verification gates 的 technical contribution，但不能直接证明绝对正确率或每条 rule 的临床因果作用。

### 4.2 由临床信息指导、可复用的 inference harness

乳腺癌开发阶段中，physician author 的角色不同于常规的数据集标注。他没有为监督训练制作大量字段标签，而是与模型开发作者一起查看具体错误，并界定有医学意义的区别。团队再把这些区别编码成 prompt、verification check 和 deterministic rule。四位独立医生的最终评估说明，形成的系统行为并不只是在复现开发医生个人的偏好。

结构化 extraction 的价值在于，它可以把一份很长、内部重复较多的病历转换成一组紧凑字段，方便检查、检索和下游使用。当任务对医生而言费时，但边界可以清楚定义时，这种辅助尤其有价值。例如，医生需要在药物清单和治疗史中反复核对当前抗癌治疗，而 harness 可以先完成这部分整理。

我们使用 inference harness 这个名称，是因为贡献位于模型权重之外。routing、schema、verification logic、deterministic rule、logging 和 attribution 都是围绕冻结模型工作的。因此，经过接口和 prompt 校准后，同一套架构原则上可以包在另一个 instruction model 外面。本研究只测试了 Qwen2.5-32B，所以跨模型迁移目前是架构上的可行性，而不是已经验证的结果。

### 4.3 Agent-assisted 跨癌种适配与 LLM-as-a-judge

PDAC 阶段检验的是，已经形成的 workflow 能否适配到 physician author 没有直接提供反馈的领域。PDAC 病历在疾病进程、治疗方案、分期表述和手术背景方面均有不同。我们保留共同的临床区分，在需要时增加癌种专用 routing，并且没有让 physician 参与 PDAC 输出的开发审阅。

这次适配中，LLM 承担了两个角色。首先，general-purpose external LLM 把乳腺癌阶段累积的 failure history 和 physician-informed distinctions 整理成书面 rubric，并提出候选 PDAC adaptation。随后，source-grounded Qwen LLM reviewer 将每轮输出与源病历比较，定位可能的遗漏、无依据内容、semantic mismatch 和 temporal error。这个 reviewer 让我们能看到适配后的 harness 在哪里表现稳定、在哪里仍然失败。

LLM reviewer 不定义最终医学结局，也不直接修改部署中的 workflow。模型开发作者选择候选修改，最终证据来自独立医生。通用 LLM evaluator 已被发现存在顺序和 self-preference bias，因此这里的角色边界很重要 [16,17]。PDAC 评估中，harness 获偏好 336 次，baseline 获偏好 91 次，另有 613 次平局。这支持 LLM-assisted development loop 的实用性，但不能证明 autonomous self-improvement。

### 4.4 模型难点与 human-model complementarity

字段级结果可以区分直接 extraction 与需要时间关系或临床判断的任务。明确写出的事实往往得到平局，因为两个系统都能找到；更大的差异出现在模型需要判断这些事实在当前语境中意味着什么的时候。

当前抗癌药物是优势最大的核心字段，结果为 132 次偏好 harness、3 次偏好 baseline 和 25 次平局。用药计划也以 98 比 2 偏向 harness，另有 60 次平局。这类任务需要在较长的治疗史中区分当前、既往、支持性和计划中的药物，可能对人工审阅而言费时，但本研究没有测量 physician time 或 burden。现有结果支持由结构化系统先行整理，效率改善仍属于 future hypothesis。

分期、转移累及和治疗反应体现了另一种互补关系。病历如果直接给出 stage，医生通常可以很快确认；模型却可能把区域淋巴结当成远处转移，或把疑似病灶当作确诊。治疗反应对双方都更难，因为必须把当前方案与影像、症状和肿瘤标志物的时间线对应起来。Passweg 等在 oncology imaging report 上也发现，本地 LLM 判断 metastasis status 的准确率不劣于人工，但 treatment response 仍明显低于人工 [14]。对于这些字段，系统更适合整理证据并给出候选答案，再由医生快速复核，而不是替代临床判断。

Distant metastasis 的标准化核心字段净差最小，结果为 23 比 5，另有 132 次平局；乳腺癌类型与受体状态紧随其后，为 24 比 14，另有 42 次平局。较高的 TIE 比例与这些字段经常被直接写在病历中相符，因此 baseline 已经能够完成很多样本。我们据此提出一种 field-dependent 分工假设：系统处理重复综合任务，对判断敏感的字段展示证据，再由医生复核紧凑输出。这是对结果的 interpretation，不是已经测量的 workflow benefit。

<div data-rough-figure="6"></div>

***图 6.*** *根据结果提出的 structured oncology extraction 医学 handoff 概念图。带 source evidence 的 harness 输出可进入直接结构化、证据核对或医学解释升级三条路径，三者都保留医生监督。图中引用的核心字段净差只提供结果背景；本研究没有测量时间、工作负担、风险或医学可接受性。*

> **协作说明：** 图 6 使用三条 clinical handoff lane，不采用未测量的 workload 或 risk 坐标。它表达的互补关系是：系统整理明确或冗长的信息，对需要判断的字段展示证据，再由医生复核或解释。

### 4.5 Staged ablation 提供的机制证据

Staged ablation 使机制结论更具体。Single prompt 到带共享 orchestration safeguards 的 decomposed extraction 没有形成整体偏好差异。B 已经包含 field routing、bounded Assessment and Plan retry，以及 format/schema repair，所以 A vs B 不能把 prompt decomposition 与所有基础 orchestration 完全拆开。最清楚的增益来自 semantic、faithfulness 和 temporal verification gates。这是目前最稳定的 component-level technical result。

最后的 deterministic layer 没有改善合并后的 C vs D 结果，而且在两个癌种中方向相反。Judge 的文字理由显示，deterministic rules 可以改善 stage 和 metastatic site 的谨慎表达，但在 PDAC 中有时会删除有用的 current medications、response assessment 或 therapy-plan content。这个结果提醒我们，rule accumulation 会产生 fidelity 与 completeness 的取舍。下一轮应优先审计造成 PDAC 信息缺失的 hooks，而不是把完整 hook set 当作统一增益。

我们核对的相关论文并不是全部都有完整 ablation。Wiest 等比较了 plain zero-shot、one-shot、加入定义和 grammar-constrained prompting [3]；Huang 等采用迭代式 prompt engineering 和 error analysis [10]；mCODEGPT 比较 single-step prompting 与两种 hierarchical prompting [7]；Corso 等比较 zero-shot、few-shot 和由医生提供 annotation 的 few-shot prompt [12]；Dubey 等比较 simple prompt、chain-of-thought、double filtering 和 rule-based ECOG extractor [15]；Grothey 等比较五种 prompting strategy 和 quantized model configuration [8]。Dao 等报告 validation and retry loop 修正了多少初始错误，Tariq 等比较完整 hybrid system 与 zero-shot、structured-code 和 rule-based baseline [5,6]。这些工作都没有对一个多阶段 oncology harness 的全部组件进行 factorial removal。

当前 ablation 明确属于 technical evidence。它不能说明医生是否会做出同样的 component-level 判断，相邻阶段比较也不能直接估计 D vs A。完整 harness 的医生比较与 staged LLM comparison 使用不同评审者和 endpoint，因此不能合并计数。下一步更有价值的是直接进行 randomized blinded A vs D technical comparison，并让医生只复核 C vs D 之间真正改变的少量输出。

### 4.6 与既有临床 extraction 研究的关系

CORAL 建立了本研究使用的真实病历 benchmark。原研究比较了 zero-shot GPT-4、GPT-3.5-turbo 和 FLAN-UL2，并由一位独立医生审阅部分 GPT-4 输出，仍发现遗漏和幻觉 [2]。我们的研究在同一 benchmark 上回答另一个问题：当基础模型保持不变时，结构化 workflow 是否优于 single-prompt baseline。

最近的 oncology extraction 研究大致形成了三个方向。第一类研究检验 prompt design 和医学知识。Huang 等用 78 份肺癌病理报告开发 prompt，在 774 份有效报告上做 independent testing，之后又在 191 份骨肉瘤报告上测试 [10]。Corso 等发现 few-shot example 和 clinical expertise 可以改善本地 small model 对意大利语肿瘤病历的 extraction [12]。Dubey 等在 ECOG extraction 中发现 chain-of-thought 和 double filtering 优于 simple prompt [15]。这些结果支持 structured prompting，但它们的目标字段比本研究的纵向 schema 更窄。

第二类研究更重视规模或 absolute correctness。Abhyankar 等用 700 份标注病历完成 model development 和 testing，之后把 staging system 应用于五个癌种、超过两百万份病历 [13]。Van Koevorden 等在 60 位头颈肿瘤患者的 29 类字段上比较 LLM 和医生 consensus，并由六位医学专家判断 error type 与 impact [11]。Passweg 等在 400 份 oncology imaging report 上比较五个本地模型，ground truth 来自双人提取和高级医生 adjudication [14]。这些设计回答 extraction 是否符合医学 reference。我们的 A/B 研究回答的是完整 harness 相对于同模型 single-prompt baseline 是否更受医生偏好，因此不能代替 absolute accuracy evaluation。

统计呈现也有相近的临床 AI 研究先例。Passweg 等使用 logistic model、按 report 聚类的 robust standard error 和 delta-method marginal effects [14]。Van Veen 等在 blinded clinical summary comparison 中保留 equivalent category，没有强迫评审者选出一侧 [24]。Ayers 等报告带 CI 的 preference proportion [25]；Goh 等则用 mixed-effects model 同时处理 case 和 participant 的重复观测 [26]。这些方法支持本文报告直观的 adjusted probability、在描述性分析中保留 TIE，并处理同一病历内的重复判断。

第三类研究支持不同的工程组件。Wiest 等说明 prompt design 和 constrained output 可以改善本地模型对临床二分类字段的 extraction [3]。mCODEGPT 在合成肿瘤病历上比较 hierarchical prompting 与 single-step strategy [7]。Grothey 等展示了不同 model、prompt 和 quantization configuration 在前列腺病理 extraction 中的差异 [8]。Dao 等结合 engineered context、validation 和 retry，但任务是右心导管报告中的数值 extraction [6]。Tariq 等使用 UMLS 加 fine-tuned LLM，在乳腺癌治疗时间线上获得更大规模和 external validation [5]；Bhattarai 等则在纵向肺癌表型任务上比较多种模型和 rule-based approach [4]。

本研究的区别不是单独发明 prompt、retry 或 rule，而是把 field-specific routing、selective context transfer、verification gates、deterministic oncology rules、logging 和 source attribution 组合为一个 inference harness，再用 same-model baseline 隔离外围 workflow 的作用，并由医生逐字段直接比较。乳腺癌到胰腺癌的开发顺序还检验了医学信息形成的 failure rules 能否在第二个癌种中减少对稀缺专科医生时间的反复占用。

因此，更准确的定位是：这是一项对整合式 inference harness 的、小规模但由医生直接评估的研究，而不是第一个肿瘤 extraction 系统，也不是规模最大的验证。更具体的贡献是把 field-specific routing、selective context transfer、verification gate、deterministic oncology rule、logging 和 source attribution 组合起来，再用 same-model baseline 和医生逐字段比较评估完整系统。逐篇对照表保留在文末中文审阅附录，正文只放与论点直接相关的引用。

### 4.7 探索性的患者沟通方向

我们还探索了 structured extraction 能否支持 patient-letter generation。早期 review 没有显示更强的 extraction 会自动产生更受偏好的 patient letter。第二次 generation 会引入新的问题，包括 content selection、organization、plain-language phrasing、不确定性表达和 patient-facing tone。这些环节需要单独开发和评估。因此，患者沟通在本稿中只作为 downstream research direction，不属于 extraction study 的主结论。

### 4.8 临床与技术意义

研究结果表明，肿瘤学 extraction 中的部分 LLM 错误具有足够的重复性，可以在不修改模型权重的情况下处理。当标注训练数据有限，或临床团队需要在不重新训练模型的情况下调整字段定义时，这一点很有价值。例如，将待活检病灶保留为疑似病变的 rule 可以被直接检查和测试。只依赖 prompt 的系统较难提供同等控制，因为一次措辞修改可能影响无关字段。

我们使用 inference harness 这一名称，是因为系统除 prompt sequence 外，还围绕冻结模型管理 task routing、dependency、verification、deterministic correction、logging 和 source attribution。Clinical workflow 也能描述该系统，但可能让人误以为它已经被整合进常规诊疗，而本 pilot 并未检验这一点。因此，本文用 inference harness 描述技术贡献，用 evaluation workflow 描述研究程序。

该系统也支持本地部署。本地运行本身不能证明系统符合隐私要求或具备临床安全性，但能让机构掌控病历处理和系统更新。本研究评估的是提取质量，而不是自主临床使用的准备程度。

### 4.9 研究边界与 next steps

本 pilot 用于判断观察到的 effect 和医学一致性是否足以支持扩大研究规模。四位医生在 breast 和 PDAC 中都独立复现了总体方向。调整后偏好概率为 79.98%，其 CI 整体高于 equal preference。

Technical ablation 已经区分了几个 broad stages，但仍只有一位非医生 LLM judge。它没有继续拆开 B、C 或 D 内部的全部组件，也没有估计 source attribution 的作用；相邻阶段比较还不能直接回答 D vs A。医生评分中的 `TIE` 也仍然无法区分“两个输出都对”和“两个输出都错”。这些限制约束了机制解释和绝对正确性结论，但不改变 primary physician result 的方向。

研究范围仍然有限。最终医学结果来自四位医生，CORAL 来自单一机构、只覆盖两个癌种，而且我们尚未测试另一个 model family。40 份 expert-annotated notes 构成 untouched holdout test set，但它仍是内部 benchmark，不是第二家机构的 external validation。PDAC 开发期间没有 physician 审阅输出。

下一项研究应优先人工 adjudicate 一部分 physician `TIE`，直接进行 randomized blinded A vs D comparison，审计可能过度裁剪 medication 和 treatment summary 的 PDAC hooks，再增加医生，并在第二种模型和外部医疗系统数据上复现。QUEST 强调 healthcare LLM evaluation 应提前规划 evaluator selection、multidimensional scoring、reliability 和 adjudication [19]。VALID framework 进一步要求逐变量对照 expert reference，并检查 internal consistency、plausibility 和 replication [20]。这两套 framework 把当前缺口说得更准确：我们的 endpoint 衡量医生的相对偏好，还不是 absolute acceptability 或 field-level accuracy。模型版本、prompt、评审者角色、benchmark 在开发中的使用方式和全部 human-oversight step 也应继续明确报告，这与 TRIPOD-LLM 的透明性要求一致 [18]。

## 5. 结论

[最终结论：在一项由多位医生参与、隐藏系统标签的评估中，由失败模式驱动的 inference harness 相较于使用同一模型的 single-prompt baseline 获得了显著的医生偏好。]

当前 pilot 支持一条分阶段开发路线。Human-in-the-loop review 把乳腺癌 extraction error 转化为明确的 inference harness。General-purpose external LLM 和 source-grounded Qwen reviewer 随后支持 PDAC adaptation，期间没有 target-domain physician 审阅，workflow 修改仍由模型开发作者控制，模型权重始终冻结。独立医生在两个癌种中都更偏好 inference harness。Technical ablation 将最清楚的 component-level gain 定位在 verification gates，并发现 deterministic hook layer 存在癌种相关的取舍。在提出广泛的确证性结论前，仍需区分不同类型的 `TIE`，针对 component changes 做医生复核，并增加医生和 external validation。

## 版本历史

- **v0.22：** 将核心字段、调整后 effect 和紧凑的病历级分布整合为一张主图，把主文图组收敛为六张，并将带完整病历标签的图移至 Supplementary Figure S1。
- **v0.21：** 统一用 physician 描述本研究人员，将定性评论移到 Appendix S2，明确 verification 是唯一产生清楚总体增益的 staged addition，并统一图中颜色语义，将 adjusted-effect graphic 简化为 probability-only presentation。
- **v0.20：** 将标准化调整后偏好概率作为 primary effect，补充 absolute effect 和保留 TIE 的 net preference margin 及其 note-cluster robust CI，并将 adjusted-effect graphic 重绘为 probability 与 odds 双尺度图。
- **v0.19：** 折叠中文版草稿信息与作者确认，将 Appendix S1 合并为一个 general judge protocol，并以统一的临床 AI 视觉语言重绘完整图组。
- **v0.18：** 使用 physician-led clinical framing 重写 Introduction；明确 development data 和 untouched holdout test set；用四位医生对两个癌种的完整结果和新图统一更新全文；删除 final test-set LLM audit 和定量 letter results；加入 blinded Codex judge instructions。
- **v0.17：** 加入四位医生的首版 breast 分析、三位医生的 PDAC 分析、staged technical ablation，以及中文版作者确认批注。

## Appendix S1. General blinded Codex judge protocol

Ablation judge 于 2026 年 10 月 1 日（PDT）运行，model 为 `gpt-5.6-sol`，reasoning effort 为 `xhigh`，service tier 为 `default`。下面的 general protocol 汇总了整次评审采用的指令，保留 blinding rules、clinical rubric 和 output schema，省略运行过程中的文件管理文字。

### General blinded judging protocol

```text
Act as an independent blinded judge for a medical information-extraction ablation. Do NOT read or use AGENTS.md, CLAUDE.md, any session log, manuscript, analysis file, git history, or any `judge_pairs.private_mapping.jsonl`. Do not try to identify the systems. Use only the public blinded input files listed below and the clinical note contained in each record.

Inputs:
- `<REPO_ROOT>/results/staged_ablation_annotated_20261001_093433/breast/judge/judge_pairs.blinded.jsonl`
- `<REPO_ROOT>/results/staged_ablation_annotated_20261001_093433/pdac/judge/judge_pairs.blinded.jsonl`

Review every record in the blinded input files. Read each note and both outputs completely, one pair at a time. Do not use scripts, regex, automated scoring, or another model to make the clinical judgments. Shell commands may be used only to display one record at a time and to validate or write the output.

For every pair, compare Output A and Output B field by field against the note. Evaluate unsupported claims or hallucination, important omissions, semantic or field mismatch, temporal errors, and clinically misleading wording. Prioritize faithfulness, then completeness. Choose `A` or `B` only when one output has a clinically meaningful overall advantage. Use `TIE` when differences are minor, offsetting, limited to formatting or field placement without clinical consequence, or when neither output is meaningfully better. Do not select a winner merely because the outputs differ.

Write one valid JSON object per pair with keys:
`cancer`, `source_line`, `pair_id`, `verdict`, `confidence` (`high|medium|low`), `a_errors`, `b_errors`, `decisive_fields`, `reason`.
Errors should be short arrays of concrete field-level findings grounded in the note. Preserve blinding throughout the review and do not access the private mapping after judging. After all records have been reviewed, validate the JSON and summarize counts by cancer and comparison, confidence levels, recurring errors, and limitations of the rubric. Do not commit or push.
```

审计 artifact：[full blinded summary](results/staged_ablation_annotated_20261001_093433/JUDGE_CODEX_FULL_BLINDED_SUMMARY.md)。

## Appendix S2. 医生自由文本评论与 TIE 的解释

三份医生评分导出共包含 14 条书面评论。这些评论说明 A/B outcome 的一个限制：`TIE` 既可能表示两个输出都可接受，也可能表示两个输出都不完整或不正确。在一项乳腺影像计划比较中，一位医生偏好 harness，因为 baseline 总结了已经完成的检查结果，而不是后续计划；另一位医生则认为两个输出都不充分，因此选择 `TIE`。第四位医生在一项用药比较中选择 `TIE`，理由是两个输出都很差；在一项操作计划比较中，该医生偏好 baseline，因为 harness 漏掉了原句的一部分。

这些评论还指出了缺少支持证据的 receptor status、遗漏的区域淋巴结病变、刚开始新方案后无法确定 response，以及两个输出都不理想的比较。它们有助于解释评分量表，并为 error analysis 提供案例，但不构成单独的 quantitative endpoint。评论数量少，且并非系统收集，因此不能估计两种 `TIE` 含义各自出现的频率。

## Appendix S3. 病历级敏感性分析

<div data-supplementary-figure="S1"></div>

***Supplementary Figure S1.*** *按癌种合并四位医生评分后的完整 note-level normalized preference margins。每个标签对应一份 held-out note。Margin 定义为该病历的 harness preferences 减 baseline preferences，再除以 completed judgments。20 份 breast notes 的 margin 全部为正；20 份 PDAC notes 中 19 份为正，1 份为负。该图属于 supporting sensitivity analysis，不是 accuracy estimate。*

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
12. Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. *Communications Medicine*. 2026. [https://doi.org/10.1038/s43856-026-01790-5](https://doi.org/10.1038/s43856-026-01790-5). <span class="reference-note"><strong>草稿简评：</strong>该研究直接证明 few-shot design 和 clinical expertise 能改善本地 small model，本文进一步加入多阶段验证和 deterministic cross-field rule。</span>
13. Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. *JCO Clinical Cancer Informatics*. 2026;10:e2500388. [https://doi.org/10.1200/CCI-25-00388](https://doi.org/10.1200/CCI-25-00388). <span class="reference-note"><strong>草稿简评：</strong>该研究用 700 份标注病历完成 model development 和 testing，之后处理五个癌种、超过两百万份病历；本文研究的是不修改权重的多字段 inference control。</span>
14. Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. *JCO Clinical Cancer Informatics*. 2026;10:e2600002. [https://doi.org/10.1200/CCI-26-00002](https://doi.org/10.1200/CCI-26-00002). <span class="reference-note"><strong>草稿简评：</strong>该 senior-physician-adjudicated study 发现 metastasis classification 比 treatment-response assessment 更容易，与本文观察到的字段难度差异一致。</span>
15. Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. *JCO Clinical Cancer Informatics*. 2026;10:e2500226. [https://doi.org/10.1200/CCI-25-00226](https://doi.org/10.1200/CCI-25-00226). <span class="reference-note"><strong>草稿简评：</strong>该研究比较 simple prompt、chain-of-thought、double filtering 和 rule-based method，直接说明 inference design 会改变单一肿瘤字段的 extraction quality。</span>
16. Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*. 2023;36:46595-46623. [https://doi.org/10.52202/075280-2020](https://doi.org/10.52202/075280-2020). <span class="reference-note"><strong>草稿简评：</strong>该研究支持把 LLM judge 用作可扩展的开发工具，同时记录 position、verbosity、self-enhancement 和 reasoning bias，因此本文不把它当作临床 endpoint。</span>
17. Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics*. 2024:9440-9450. [https://doi.org/10.18653/v1/2024.acl-long.511](https://doi.org/10.18653/v1/2024.acl-long.511). <span class="reference-note"><strong>草稿简评：</strong>该论文单独验证了 LLM comparison 中的 position bias，说明 automated review 需要人工终审和保守解释。</span>
18. Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. *Nature Medicine*. 2025;31:60-69. [https://doi.org/10.1038/s41591-024-03425-5](https://doi.org/10.1038/s41591-024-03425-5). <span class="reference-note"><strong>草稿简评：</strong>TRIPOD-LLM 要求清楚报告 model、prompt、evaluation procedure、task-specific metric 和 human oversight，可用于检查本 pilot 的透明度。</span>
19. Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. *npj Digital Medicine*. 2024;7:258. [https://doi.org/10.1038/s41746-024-01258-7](https://doi.org/10.1038/s41746-024-01258-7). <span class="reference-note"><strong>草稿简评：</strong>该文综述 142 项研究并提出 QUEST framework，可直接指导下一轮 evaluator selection、rating dimension、reliability assessment 和 adjudication。</span>
20. Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. *JCO Clinical Cancer Informatics*. 2026;10:e2500215. [https://doi.org/10.1200/CCI-25-00215](https://doi.org/10.1200/CCI-25-00215). <span class="reference-note"><strong>草稿简评：</strong>VALID 要求逐变量对照 expert reference、检查 internal consistency 和 plausibility，并做 replication，因此能准确说明 preference endpoint 与 absolute correctness 之间的差距。</span>
21. Kong HJ. Managing Unstructured Big Data in Healthcare System. *Healthcare Informatics Research*. 2019;25(1):1. [https://doi.org/10.4258/hir.2019.25.1.1](https://doi.org/10.4258/hir.2019.25.1.1). <span class="reference-note"><strong>草稿简评：</strong>该文支持医疗数据中大量信息仍为非结构化数据的宽泛估计；这个估计不是肿瘤专科测量。</span>
22. Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. *JAMA Network Open*. 2021;4(7):e2115334. [https://doi.org/10.1001/jamanetworkopen.2021.15334](https://doi.org/10.1001/jamanetworkopen.2021.15334). <span class="reference-note"><strong>草稿简评：</strong>该研究覆盖近 300 万份门诊病历，记录了病历长度和冗余增加，以及复制内容和模板内容的影响。</span>
23. Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. *npj Digital Medicine*. 2025;8:301. [https://doi.org/10.1038/s41746-025-01686-z](https://doi.org/10.1038/s41746-025-01686-z). <span class="reference-note"><strong>草稿简评：</strong>该研究是 human-in-the-loop refinement、error ontology 和通用规则替代病例特异性修补的近似先例。</span>
24. Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. *Nature Medicine*. 2024;30:1134-1142. [https://doi.org/10.1038/s41591-024-02855-5](https://doi.org/10.1038/s41591-024-02855-5). <span class="reference-note"><strong>草稿简评：</strong>该 blinded reader study 的评分保留 equivalent midpoint，支持 comparative clinical evaluation 不强迫评审者选出一侧。</span>
25. Ayers JW, Poliak A, Dredze M, et al. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum. *JAMA Internal Medicine*. 2023;183(6):589-596. [https://doi.org/10.1001/jamainternmed.2023.1838](https://doi.org/10.1001/jamainternmed.2023.1838). <span class="reference-note"><strong>草稿简评：</strong>该 physician-rated comparison 报告 preference proportion 及其 CI，为 probability-scale presentation 提供直接先例。</span>
26. Goh E, Gallo R, Hom J, et al. Large Language Model Influence on Diagnostic Reasoning. *JAMA Network Open*. 2024;7(10):e2440969. [https://doi.org/10.1001/jamanetworkopen.2024.40969](https://doi.org/10.1001/jamanetworkopen.2024.40969). <span class="reference-note"><strong>草稿简评：</strong>其 mixed-effects analysis 同时加入 participant 和 case effect，说明临床研究中的重复观测不应作为独立数据处理。</span>

## 附录 A：相关工作对照表（供合作者审阅）

> **说明：** 这张表用于内部核对和讨论，不放入英文主稿。正文 4.6 已改为自然引用。表中的 ablation 描述来自原论文方法与结果部分的逐项核查。

| 研究 | 数据与任务 | 临床专家角色 | 方法与 component comparison | 与本研究的关系 |
|---|---|---|---|---|
| Sushil 等，CORAL [2] | 40 份真实乳腺癌和胰腺癌病历；广泛 oncology schema | 专家标注；一位独立医生审阅每个癌种 10 份 GPT-4 输出 | 比较 zero-shot GPT-4、GPT-3.5-turbo 和 FLAN-UL2；没有 inference harness component ablation | 提供相同 benchmark 和临床范围，但未进行 same-model harness 对 single-prompt baseline 的评估 |
| Wiest 等 [3] | 500 份 MIMIC 病史；5 个二分类字段 | 三位盲法医学专家建立 consensus ground truth | 比较 plain zero-shot、one-shot、加入定义和 grammar-constrained prompting；属于 prompt-level component comparison | 支持 prompt design 和 constrained output 的作用，但任务不是肿瘤学，输出范围较窄 |
| Bhattarai 等 [4] | 63 位肺癌患者的 13,646 份病历；4 个纵向表型 | 两位领域专家提供 gold-standard annotation | 比较 GPT、open model 和 rule-based method；不是单一 pipeline 内部的逐组件 ablation | 数据规模更大，但 target phenotype 较少，也没有 same-model workflow 对照 |
| Tariq 等 [5] | 26,692 位内部乳腺癌患者和 162 位外部患者；治疗时间线 | registry label；专家整理治疗概念和 code | 完整 UMLS + fine-tuned LLM hybrid system 对 zero-shot、structured-code 和 rule-based baseline；没有逐个移除 hybrid phase | external validation 更强，但依赖 supervised fine-tuning，任务集中于 5 类治疗 |
| Dao 等 [6] | 220 份开发和 200 份验证用右心导管病历 | 一位肺血管疾病专家建立 validation ground truth 并指导开发 | engineered preload、validation、retry；单独报告 retry 对初始错误的修正情况，但没有完整 factorial ablation | workflow 架构最接近，但任务是 procedure note 中的数值 extraction |
| Zhang 等，mCODEGPT [7] | 1,000 份合成肿瘤病历；49 个 mCODE entity | 自动匹配结合人工 validation | 直接比较 single-step baseline、BFOP 和 2POP hierarchical prompting；是最接近的 prompt ablation | 证明 hierarchy 有价值，但数据为 synthetic notes，且没有 blinded physician comparison |
| Grothey 等 [8] | 579 份德语和英语前列腺病理报告；11 个字段 | 医学博士生在主治病理医生指导下标注 | 比较多个 model、5 种 prompting strategy 和 quantization configuration | component/configuration comparison 较完整，但任务是一类 pathology report，不是纵向门诊病历 |
| Huang 等 [10] | 78 份肺癌报告用于 prompt development，774 份有效报告用于 independent testing，另有 191 份骨肉瘤病理报告 | 与专家整理的结构化数据比较 | spiral prompt engineering；分析专业术语和 TNM stage error | 证明 prompt iteration 有效，也说明复杂 staging 仍需要显式临床约束 |
| van Koevorden 等 [11] | 60 位头颈肿瘤患者、1,482 页文档、29 类字段 | 两位医生人工 extraction；六位临床专家判断 match、error type 和 impact | 预训练 open-source LLM 对 physician consensus；报告 absolute accuracy、error taxonomy 和时间节省 | 最接近的 physician-validated comparator；absolute validation 更强，但不比较 same-model harness 和 baseline |
| Corso 等 [12] | 意大利语 NSCLC EHR；四个本地 small language model | 不同临床经验人员参与 prompt example 和 consistency assessment | zero-shot、few-shot、expert-annotated few-shot | 直接支持 clinical expertise 和 prompt design，但没有本文的 gates、hooks 和跨字段 consistency control |
| Abhyankar 等 [13] | 700 份标注病历；五个癌种；部署到超过两百万份 notes | 人工 annotation 建立训练、验证和测试数据 | supervised fine-tuning；跨癌种 staging extraction | 规模和 staging validation 更强，但任务较窄且修改 model weights |
| Passweg 等 [14] | 400 份德语 oncology imaging report | 双人 extraction；由高级医生 adjudicate disagreement | 五个本地 LLM 与人工准确率比较 | 显示 metastasis 可接近人工、response 仍更难，支持本文的 field-dependent difficulty 解释 |
| Dubey 等 [15] | 三个癌种病历中的 ECOG status | 使用 adapted QUEST 做 human evaluation | rule-based、simple prompt、CoT 和 double filtering comparison | 是清楚的 prompt-level ablation，但只覆盖一个临床字段 |

## 附录 B：评估方法文献（供合作者审阅）

| 研究 | 主要发现 | 对本文的直接用途 |
|---|---|---|
| Zheng 等 [16] | LLM judge 可以接近人工偏好，但存在 position、verbosity、self-enhancement 和 reasoning bias | 支持将 LLM-as-a-judge 限定为开发工具，不把它作为最终临床结局 |
| Wang 等 [17] | 交换候选答案顺序可以显著改变 LLM evaluator 的判断；校准可减轻但不能消除问题 | 支持对 automated pairwise review 采用保守解释，并保留人工终审 |
| Gallifant 等，TRIPOD-LLM [18] | 提供覆盖 model、prompt、evaluation、human oversight 和 reproducibility 的 LLM 研究报告清单 | 用于检查本文 Methods 与 limitations 是否充分披露开发和评估过程 |
| Tam 等，QUEST [19] | 回顾 142 项 healthcare LLM human-evaluation study，并把流程组织为 planning、implementation、adjudication、scoring 和 review | 说明下一轮评估应预先定义 evaluator、维度、reliability 和争议处理，而不只记录 A/B/TIE |
| Estevez 等，VALID [20] | 要求 extracted variable 与 expert reference 逐项比较，并加入 consistency、plausibility 和 replication 检查 | 说明当前 preference study 不能替代 absolute correctness validation，并给出补强路径 |
