<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 2.0, November 8, 2026.

Current change: the latest doctor Word draft is the prose authority. The HTML retains the current technical objects. Underlined phrases require author review.

The canonical physician blocks below mirror `202601102_Manuscript_JAMA.docx` for audit purposes. The clean HTML applies only the approved cross-reference and reporting edits while preserving all source blocks in its hidden audit representation.

<!-- physician:P000 -->
<!-- P000 empty -->

<!-- physician:P001 -->
<!-- P001 empty -->

## Introduction

<!-- physician:P002 -->
Introduction

<!-- physician:P003 -->
<!-- P003 empty -->

<!-- physician:P004 -->
Large language models (LLMs) are increasingly used for clinical documentation, patient communication, and administrative tasks. Many of these applications depend on on identifying relevant information in electronic health record (EHR) notes before those facts can be summarized, analyzed, or used in another task. Reliable extraction from those notes is therefore an important step for many uses of LLMs in oncology care and research. This study focuses on that extraction task.

<!-- physician:P005 -->
<!-- P005 empty -->

<!-- physician:P006 -->
It is estimated that 80% of clinical data in EHR exist in unstructured form, with a large proportion of clinically relevant information is encoded in free-text narratives. Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and changes in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real-world practice, laboratory, imaging, and genomic results copied from earlier notes lengthen notes and further complicating the extraction task. A study of outpatient documentation across a decade found that notes became longer and more redundant, with much of the increase attributable to templates or copied and pasted text . LLMs must therefore not only identify the relevant information, but also determine its relationship to patient’s current diseases.&#32;

<!-- physician:P007 -->
<!-- P007 empty -->

<!-- physician:P008 -->
LLMs can extract information from clinical notes, but their outputs can vary. They may omit details, confuse past findings with current findings, or add information that is not supported by the original note (Sushil et al., 2024; Huang et al., 2024). A reviewer may be able to identify these errors, but checking every field in every output can approach the cost of manual extraction. Commercial models perform better on some clinical extraction tasks, but sending protected health information (PHI) to externally hosted services raises privacy and governance concerns, including HIPAA compliance (Huang et al., 2024; Wiest et al., 2024). Local deployment avoids external transfer of PHI, but local models can still produce unstable or incorrect extractions (Wiest et al., 2024). A useful clinical workflow must therefore improve reliability without depending on external processing or case by case manual review.

<!-- physician:P009 -->
<!-- P009 empty -->

<!-- physician:P010 -->
How an LLM is used may be as important as the model itself. Additional training can adapt model weights to a clinical task. When the weights are unchanged, prompt design, task decomposition, verification, and clinical rules can organize the extraction process. Huang et al. iteratively refined prompts on expert curated lung pathology data and later applied the same prompt engineering strategy to osteosarcoma reports. Hein et al. used a human in the loop process to refine prompts, output schemas, and an error ontology, then applied the resulting templates to breast and prostate pathology reports (Huang et al., 2024; Hein et al., 2025). These studies focused on pathology reports and prompt or schema refinement. It remains unclear whether a broader multistep workflow developed from physician feedback can be used to improve note extraction.&#32;

<!-- physician:P011 -->
<!-- P011 empty -->

<!-- physician:P012 -->
To address this gap, we built a locally deployable inference harness around a frozen LLM for extracting information from real-world unstructured oncology notes. The harness divides extraction into smaller tasks and applies checks and clinical rules without changing the model weights. We developed the breast cancer harness with physician input, we then adapted it to pancreatic ductal adenocarcinoma (PDAC) note using LLM-assist review and regression tests without physician review of the PDAC development outputs. Four physicians compared the final harness with a one prompt baseline that used the same model. Our results showed that, the output from the harness is favored compared to a single-prompt baseline.

<!-- physician:P013 -->
<!-- P013 empty -->

## Methods

<!-- physician:P014 -->
Methods

<!-- physician:P015 -->
<!-- P015 empty -->

<!-- physician:P016 -->
Study Design

<!-- physician:P017 -->
The study comprised two development stages followed by an independent physician evaluation. In Stage 1, a physician and an investigator developed the breast cancer inference harness through review of selected outputs. In Stage 2, the investigator adapted the harness to pancreatic ductal adenocarcinoma (PDAC) using LLM review and regression tests, without physician review of the PDAC development outputs. Four physicians then compared the final systems on an untouched holdout set.

<!-- physician:P018 -->
<!-- P018 empty -->

<!-- physician:P019 -->
The final harness was compared with a one prompt baseline using the same frozen model and field schema on an untouched holdout set of 40 CORAL notes annotated by experts. The set included 20 breast cancer notes and 20 PDAC notes. Four physicians completed evaluations for each field across both cancer types. System identities were not disclosed.

<!-- physician:P020 -->
<!-- P020 empty -->

<!-- physician:P021 -->
We separately conducted an ablation study of four cumulative harness configurations using an OpenAI LLM judge. The study workflow is shown in Figure 1.

<!-- physician:P022 -->
<!-- P022 empty -->

<!-- physician:P023 -->
Dataset

<!-- physician:P024 -->
We used CORAL, a PhysioNet dataset of deidentified medical oncology notes obtained from the University of California, San Francisco. The development process used 100 unannotated breast cancer notes and 100 unannotated pancreatic cancer notes. A separate untouched holdout set of 40 notes annotated by experts consisted of 20 breast cancer notes and 20 PDAC notes. The holdout note texts were not used for development. The annotations were not provided to the model or used during development, and they were not used to calculate the primary endpoint based on physician preference. They were not shown to the physicians in the evaluation interface.

<!-- physician:P025 -->
<!-- P025 empty -->

<!-- physician:P026 -->
Baseline model and model development

<!-- physician:P027 -->
We used Qwen2.5-32B-Instruct-AWQ served locally through vLLM. The baseline used the complete note and all extraction questions in one prompt and requested the full target schema in one response. Each note was processed in one chat completion request. This describes the application-level request and does not make a claim about internal server execution or forward passes. In the physician review, the baseline and harness used the same field definitions and output contract.

<!-- physician:P028 -->
<!-- P028 empty -->

<!-- physician:P029 -->
Inference Harness

<!-- physician:P030 -->
The harness extracted clinical information in multiple steps because different fields required different clinical context, reasoning, processing, and validation. It first used regular expressions to identify the Assessment and Plan section for fields related to the plan. If this step failed, the model made up to three attempts to locate and reproduce the relevant text, and each attempt was checked against the original note. If all attempts failed, the harness skipped only the plan branch and continued extraction from the complete note. Separate prompts for each field identified the visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Information extracted in earlier steps then guided dependent assessments. For example, cancer stage and metastatic status provided context for treatment intent, while current therapy and clinical findings provided context for treatment response. The plan branch used text from the Assessment and Plan section that was supported by the source note. Prompts instructed the model to leave a field empty when the note did not provide sufficient evidence.

<!-- physician:P031 -->
<!-- P031 empty -->

<!-- physician:P032 -->
Outputs passed through structural checks for JSON and schema validity, followed by checks for specificity, semantic alignment, accuracy and support from the original note, and temporal consistency. Deterministic clinical rules and processing across fields were then applied to repeated extraction errors. Source attribution was added when enabled. See Table 1. Operational definitions and order are provided in Supplementary Methods.

<!-- physician:P033 -->
<!-- P033 empty -->

<!-- physician:P034 -->
We included clinical rules to address repeated extraction errors. For example, these rules helped the system distinguish regional lymph node involvement from distant metastases, suspected metastatic disease from confirmed metastatic disease, and active anticancer treatment from supportive or home medications. They also prevented disease changes documented before treatment initiation from being misclassified as a treatment response. The system returned supporting excerpts from the original clinical note when available.

<!-- physician:P035 -->
<!-- P035 empty -->

<!-- physician native table:T1; cells preserved in canonical JSON -->

<!-- physician:P036 -->
<!-- P036 empty -->

<!-- physician:P037 -->
<!-- P037 empty -->

<!-- physician:P038 -->
Figure 1. Development, transfer, and evaluation pathway. Physician review of breast cancer outputs informed the inference harness. The failure history and clinical rubric then supported investigator controlled PDAC adaptation without physician review of PDAC outputs. Four physicians compared the final harness with the same model one prompt baseline on the holdout set of 40 notes.

<!-- technical insert after P038 -->

<!-- physician:P039 -->
<!-- P039 empty -->

<!-- physician:P040 -->
<!-- P040 empty -->

<!-- physician:P041 -->
Breast cancer development with physician review

<!-- physician:P042 -->
Breast cancer development used all 100 development notes. Early and targeted iterations used smaller subsets to shorten the feedback cycle, while major and final runs covered all 100 notes to detect less frequent errors and check for regression. A physician reviewed selected outputs, and the investigators compared flagged outputs with the complete source note, grouped repeated errors, and translated them into general changes to prompts, verification logic, or deterministic rules. These changes were retained only after testing on affected examples. The objective was to encode repeated clinical distinctions rather than corrections for individual cases. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general rule for regional nodes. An error in which a planned drug was listed as active therapy motivated a temporal medication rule. This process did not produce a conventional supervised training set, and the model weights were not updated.

<!-- physician:P043 -->
<!-- P043 empty -->

<!-- physician:P044 -->
PDAC adaptation with feedback from language models

<!-- physician:P045 -->
<!-- P045 empty -->

<!-- physician:P046 -->
The harness developed from breast cancer notes was further adapted to pancreatic ductal adenocarcinoma using 100 additional pancreatic cancer notes. No physician reviewed pancreatic cancer outputs during this stage. ChatGPT organized the clinical error categories and clinical distinctions established during breast cancer development into a structured review rubric. The Qwen pipeline generated structured outputs, and a Qwen reviewer guided by the rubric compared them with the complete pancreatic source notes and flagged potential errors. Investigators assessed the flags, implemented selected revisions to prompts, clinical rules, or workflow logic, and retained changes only after regression testing.

<!-- physician:P047 -->
The adaptation retained the core framework while introducing modifications specific to pancreatic cancer, particularly for disease terminology, treatment regimens, and rules applied after extraction. The framework preserved its core approach to error detection and correction while accommodating differences between the two cancer types.

<!-- physician:P048 -->
<!-- P048 empty -->

<!-- physician:P049 -->
Clinical evaluation data collection and scoring rubric

<!-- physician:P050 -->
<!-- P050 empty -->

<!-- physician:P051 -->
The clinical evaluation presented the source note and two structured outputs through an A/B test interface. Four physicians selected A better, B better, or tie for each field. The physician who participated in Stage 1 development was not one of the four physicians. System identities were not disclosed. Source attribution was available only with the harness output when present, so physicians compared the full output packages rather than source attribution alone. Before evaluating the holdout set, the authors designated seven core fields as clinically important for understanding disease status and current management. Details of the interface are provided in Supplementary Methods. See Supplementary Table S1.

<!-- physician:P052 -->
<!-- P052 empty -->

<!-- physician:P053 -->
<!-- P053 empty -->

<!-- physician:P054 -->
Ablation study

<!-- physician:P055 -->
We conducted an ablation study on the same 20 breast cancer and 20 pancreatic cancer holdout notes to examine how output quality changed across successive harness configurations. All four conditions used Qwen2.5-32B-Instruct-AWQ, greedy decoding, and the same output contract with 31 fields. Attribution, tool calling, and letter generation were disabled.

<!-- physician:P056 -->
See Table 2 for the 4 conditions.

<!-- physician:P057 -->
<!-- P057 empty -->

<!-- physician:P058 -->
Table 2. Ablation study conditions

<!-- physician native table:T2; cells preserved in canonical JSON -->

<!-- physician:P059 -->
<!-- P059 empty -->

<!-- physician:P060 -->
We compared adjacent configurations (A vs B, B vs C, and C vs D), yielding 120 paired comparisons across 40 notes. A single OpenAI LLM judge received only the source note and the two outputs for each adjacent configuration pair. The judge had no access to the project code, development history, prior evaluations, or configuration identities. It was instructed to prioritize accuracy and support from the original note, then completeness, and to use TIE when differences were minor or offsetting. We compared adjacent configurations using two sided exact sign tests, excluding ties. Details of the judge model and settings are provided in Supplementary Methods.

<!-- physician:P061 -->
<!-- P061 empty -->

<!-- physician:P062 -->
Statistical analysis

<!-- physician:P063 -->
For the physician evaluation, interrater agreement was assessed pairwise using exact agreement and Cohen’s κ for each evaluator pair. Results from the 8 completed clinician-by-cancer evaluations were summarized separately. Pooled counts were reported descriptively and were not treated as independent field-level observations.

<!-- physician:P064 -->
<!-- P064 empty -->

<!-- physician:P065 -->
For the primary inferential analysis, tied comparisons were excluded, and each remaining directional judgment was coded according to whether it favored the harness or the baseline. We fit a population-averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level to account for clustering of repeated field-level judgments within the same note. Evaluator was included as a fixed effect, and cancer type was additionally included in the overall model. Cancer-specific models adjusted for evaluator.&#32;

<!-- physician:P066 -->
<!-- P066 empty -->

<!-- physician:P067 -->
The primary outcome was the adjusted probability of harness preference among directional judgments. Predicted probabilities were standardized over the observed covariate distribution, and 95% confidence intervals (CIs) were calculated using the delta method on the logit scale. The difference from 50% was reported as the absolute preference effect, while adjusted odds ratios (ORs) were reported as secondary measures.

<!-- physician:P068 -->
<!-- P068 empty -->

<!-- physician:P069 -->
For sensitivity analyses, we refitted the overall model four times after excluding one physician at a time. We also retained tied judgments by coding harness, tie, and baseline preferences as 1, 0, and −1, respectively, and calculated the mean net preference margin with 95% CIs using sandwich standard errors clustered by note. Finally, we aggregated preferences for each field within each note and performed an exact two-sided sign test comparing positive and negative margins by note.

<!-- physician:P070 -->
<!-- P070 empty -->

<!-- physician:P071 -->
Exploratory analysis of core fields used one joint population averaged logistic GEE on 495 directional judgments. Field, physician, and cancer type were fixed effects. Repeated judgments were clustered by note with an exchangeable working correlation. Probabilities for each field were standardized over the physician and cancer distribution observed for that field. Breast type and receptor status applied only to breast cancer notes. Ties were excluded. These estimates describe preference rather than accuracy, and no pairwise field tests or multiplicity claims were made.

<!-- physician:P072 -->
<!-- P072 empty -->

## Results

<!-- physician:P073 -->
Results

<!-- physician:P074 -->
All four physicians evaluated both cancer types. Across 2,160 required judgments for each note and field, the harness was preferred 699 times, 1,290 comparisons were ties, and the baseline was preferred 171 times. See Table 3, Table 4, and Figure 2. These correspond to 32.36%, 59.72%, and 7.92% of all judgments. Among the 870 directional judgments, the crude harness share was 80.34% (note cluster robust 95% CI 76.14% to 83.96%).

<!-- technical insert after P074 -->

<!-- physician:P075 -->
Table 3. Physician preferences for breast cancer extraction

<!-- physician native table:T3; cells preserved in canonical JSON -->

<!-- physician:P076 -->
Table 4. Physician preferences for PDAC extraction

<!-- physician native table:T4; cells preserved in canonical JSON -->

<!-- physician:P077 -->
For breast cancer note, all four physicians independently favored the harness on the breast cancer set. Their pooled breast result was 363 harness preferences, 80 baseline preferences, and 677 ties. The first physician's result by note was 18 harness wins, one baseline win, and one tie. The second and third physicians each recorded a harness win in all 20 notes. The fourth recorded 16 harness wins, two baseline wins, and two ties. When the four breast evaluations were combined within each note, all 20 notes had a positive difference between harness and baseline preferences (Supplementary Figure S1).

<!-- physician:P078 -->
For pancreatic cancer notes, the evaluations produced 336 harness preferences, 91 baseline preferences, and 613 ties. The harness had a positive pooled margin in 19 pancreatic notes and a negative margin in one.

<!-- physician:P079 -->
<!-- P079 empty -->

<!-- physician:P080 -->
Table 2

<!-- physician:P081 -->
<!-- P081 empty -->

<!-- physician:P082 -->
Figure 2 Physician preferences across eight combinations of physician and cancer type. Panel A retains harness, baseline, and tie judgments. Panel B shows the harness share among directional judgments.

<!-- technical insert after P082 -->

<!-- physician:P083 -->
<!-- P083 empty -->

<!-- physician:P084 -->
The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast cancer estimate was 81.66% (95% CI, 76.02% to 86.21%), the pancreatic cancer estimate was 78.15% (95% CI, 71.47% to 83.62%). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p < 0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p < 0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p < 0.05). See Table 5.

<!-- technical insert after P084 -->

<!-- physician:P085 -->
<!-- P085 empty -->

<!-- physician:P086 -->
Table 5. Adjusted probability of inference harness preference among directional judgments

<!-- physician:P087 -->
<!-- P087 empty -->

<!-- physician:P088 -->
Sensitivity and physician agreement

<!-- physician:P089 -->
Sensitivity analyses supported the primary findings. When ties were included, the net preference margin favored the harness (+24.4 percentage points; 95% CI, +20.4 to +28.5). At the note level, the harness was preferred in 39 of 40 cases (97.5%). When each physician was excluded individually, adjusted probabilities of harness preference ranged from 77.08% to 83.53% (Supplementary Table S3).

<!-- physician:P090 -->
Pairwise exact agreement ranged from 73.1% to 82.9%, with Cohen's kappa ranging from 0.511 to 0.646 (Figure 3 and Supplementary Table S2).

<!-- physician:P091 -->
<!-- P091 empty -->

<!-- physician:P092 -->
Figure 3. Pairwise exact agreement and Cohen's kappa among four physicians in breast cancer and PDAC. Each breast cancer pair shared 280 judgments, and each PDAC pair shared 260 judgments.

<!-- technical insert after P092 -->

<!-- physician:P093 -->
Performance across Clinical Extraction Fields

<!-- physician:P094 -->
We examined physician preferences across the seven prespecified clinical extraction fields. In all the 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all 7 fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast cancer type and receptor status, where ties were more common (Figure 4).

<!-- physician:P095 -->
<!-- P095 empty -->

<!-- physician:P096 -->
Figure 4. Preference results for core fields. Panel A retains the raw distribution with ties for seven fields prioritized by physicians. Panel B excluded ties and shows adjusted probabilities of harness preference for each core field with 95% confidence intervals from one joint exploratory GEE fitted to 495 directional judgments. Breast type and receptor status applied only to breast cancer notes.&#32;

<!-- technical insert after P096 -->

<!-- physician:P097 -->
We also analyzed physician preferences for each individual note to determine whether the overall advantage was consistent across cases. After combining physician ratings for each note, the harness was preferred in all 20 breast cancer notes and 19 of 20 pancreatic cancer notes. The overall ratings also favored the harness in each evaluator-by-cancer group. These findings were consistent with the primary analysis and suggest that the observed advantage was not driven by a small number of clinical notes (Figure 4).

<!-- physician:P098 -->
Technical ablation

<!-- physician:P099 -->
A separate technical ablation comprised 120 adjacent comparisons across the same 40 notes. A single nonphysician LLM judge, blinded to configuration identity, evaluated the comparisons. The four configurations are defined in Table 2. The one prompt versus decomposed comparison yielded 17 preferences for each, 6 ties, and P = 1.000. With verification added, the result was 24 preferences for verified extraction, 2 for decomposed extraction, 14 ties, with a P <0.05, this suggests the verified extraction step significantly affected the LLM judge’s preference. After full inference harness was added, the result was 19 (Verified extraction) vs 16 (full harness), with 5 ties, and a P value of 0.736. The three unadjusted two-sided exact sign tests excluded ties. See Supplementary Figure S2 for the full roadmap.

<!-- physician:P100 -->
<!-- P100 empty -->

<!-- physician:P101 -->
<!-- P101 empty -->

## Discussion

<!-- physician:P102 -->
Discussion

<!-- physician:P103 -->
<!-- P103 empty -->

<!-- physician:P104 -->
In this study, the inference harness was preferred over a one prompt baseline using the same underlying LLM. This advantage was observed in both breast and pancreatic cancer and remains consistent in both sensitivity analysis. These findings suggest that a structured inference harness can improve the clinical quality of extracted information without additional model training. The note level results also indicate that the observed preference was not driven by a small number of clinical notes.

<!-- physician:P105 -->
<!-- P105 empty -->

<!-- physician:P106 -->
Structured extraction can convert long, complex, free-text note into a compact set of fields that can be checked, searched, and reused downstream. In oncology, this is useful as clinical notes can be complex and contain multiple diagnoses, prior and future treatments, and change in disease status. Prior studies have demonstrated the potential of LLMs for extracting structured information from clinical notes. However, the CORAL study showed that advanced, commercialized models could have omission or generate unsupported findings. Huang et al demonstrated that iterative prompt engineering with ChatGPT using API could improve extraction of free-text pathology notes. Similarly, Hein et al. showed that iterative refinement guided by clinicians and systematic error analysis could improve structured extraction and can be adapted to other cancer types.

<!-- physician:P107 -->
<!-- P107 empty -->

<!-- physician:P108 -->
Our study built on these studies by using physician feedback to identify recurring errors and develop clinical rules during breast cancer note development. Rather than producing a large, supervised training set, we used the feedback to develop prompts for each field, verification steps, and clinical rules. This approach incorporated physician knowledge into the inference harness and reused in subsequent extraction tasks. Instead of asking physicians to perform iterative evaluation, we used ChatGPT to organize the established criteria from breast cancer development, and used Qwen2.5-32B as a judge. LLM-as-a-judge has been reported and studies before, prior studies showed that LLM judges showed high concordance on text-comparison rubrics such as faithfulness and completeness. The final harness was preferred over the baseline by independent physician evaluators in pancreatic cancer. This finding suggests that criteria established through physician feedback may be transferable to other cancer types and may reduce the need for repeated physician review during development.

<!-- physician:P109 -->
<!-- P109 empty -->

<!-- physician:P110 -->
One recurring challenge we identified was the interpretation of temporal relationships in oncology notes, as reported by prior studies (Kruse et al., 2025; Kougia et al., 2024). Although the LLM could identify treatment responses documented in the clinical history, it sometimes failed to associate it with the correct disease status. For example, a response to a previous treatment could be incorrectly attributed to current regimen. This was particularly important for patients who had received multiple lines of therapy, where each treatment response must be associated with treatment history. To reduce this risk, we first revised the prompt to instruct the model to distinguish evidence recorded before and after a regimen change. Beyond the prompt, the response extraction step used the complete note together with structured context on current medications and recent treatment changes. The extracted response then underwent semantic and source support checks. These temporal controls were embedded across the prompt, structured context, and downstream checks, which made their independent contribution difficult to isolate. The physician ratings therefore support the complete workflow but do not determine how much of the observed preference came from the temporal controls alone.

<!-- physician:P111 -->
<!-- P111 empty -->

<!-- physician:P112 -->
Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure S2). Adding decomposed extraction alone did not produce an overall improvement while adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, we found that the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer, this suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce tradeoffs, especially when the terminology or treatment differ between cancer types. For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC.

<!-- physician:P113 -->
<!-- P113 empty -->

<!-- physician:P114 -->
This study has several limitations. First, our primary outcome was physician preference rather than absolute extraction accuracy. A tie could indicate that both systems produced acceptable outputs or that both contained errors. Although physician ratings consistently favored the harness, the evaluation included only four physicians and 40 clinical notes from two cancer types within a dataset from a single source. It is worth noting that these notes were real clinical notes rather than synthetic cases or narratives generated by models. They contain repeated information copied from earlier notes, uncertain findings, treatment plans, or even typo or errors that make the extraction task challenging, nevertheless, further studies are needed to investigate if the results can be generalized to other cancer types or language models. The observed preference consistently favored the complete harness output as presented in both cancer types and sensitivity analyses. The current design cannot separate the effects of extraction quality, the surrounding workflow, source attribution, and display position. Second, the exploratory ablation was evaluated by a single LLM judge that was not a physician and compared cumulative stages rather than individual components. Therefore, it could not determine the independent contribution of each verification step or clinical rule. Future studies should evaluate extraction accuracy against expert reference standards, include larger and more diverse clinical datasets, and assess the generalizability of the harness across different open-weight models.

<!-- physician:P115 -->
<!-- P115 empty -->

<!-- physician:P116 -->
<!-- P116 empty -->

<!-- physician:P117 -->
<!-- P117 empty -->

<!-- physician:P118 -->
<!-- P118 empty -->

<!-- physician:P119 -->
<!-- P119 empty -->

<!-- physician:P120 -->
<!-- P120 empty -->

<!-- physician:P121 -->
<!-- P121 empty -->


## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Outstanding physician comments

The canonical extraction retains all 9 Word comments. The HTML keeps all 9 source comments in its hidden audit representation.

## Author literature-review workspace

The HTML includes 30 annotated literature-review entries as a default-closed author workspace outside the manuscript body.
