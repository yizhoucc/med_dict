<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 1.2.1, October 27, 2026.

Current change: bounded print output now contains only the clean manuscript and formal supplement; screen comments and author workspaces remain available but are excluded from print.

The canonical physician blocks below mirror `202601023_Manuscript_JAMA.docx` for audit purposes. The clean HTML applies only the approved cross-reference and reporting edits while preserving all source blocks in its hidden audit representation.

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
Recent years have witnessed increased use of Large language models (LLMs) in clinical language tasks, including patient education, clinical documentation, and administrative work. Many of these applications relies on identifying relevant information from electronic health record (EHR) data before it can be summarized, analyzed, or used in another task. Reliable extraction from those notes is therefore an important step in applying LLMs to oncology care and research.

<!-- physician:P005 -->
<!-- P005 empty -->

<!-- physician:P006 -->
Although EHRs contains structured fields, it is estimated that 80% of clinical data exist in unstructured form, and a large proportion of clinically relevant information is encoded in free-text narratives. Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and change in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real-world practice, copy-forward laboratory, imaging, and genomic results further length notes, further complicating the extraction task. In a study of outpatient documentations across a decade, notes became longer and more redundant, and most of it through templates or copy-paste (citation). LLMs must therefore not only identify the relevant information, but also determine its relationship to patient’s current diseases.&#32;

<!-- physician:P007 -->
<!-- P007 empty -->

<!-- physician:P008 -->
Currently, LLMs can assist with this task, but the output does not ensure that the extracted information is fully loyal to the original note. Models may generated false information, omit details, confused historical findings, or infer a plausible content where the note does not provide one. These errors can be difficult to spot by human eyes. Commercial LLMs have been evaluated for clinical note extraction, but performance remains variable. The use of commercial models also raised privacy and governance concerns, the potential of transmission patient protected health information (PHI) to externally hosted services can introduce institutional and patient concerns related to data processing, retention, and HIPPA compliance. A locally deployed open-weight model can process notes without sending PHI to outside source, however local models alone does not resolve extractions errors.&#32;

<!-- physician:P009 -->
<!-- P009 empty -->

<!-- physician:P010 -->
Local open-weight models are smaller, and their performances varies with model size. Recent studies suggest that the performance of LLMs depends not only on the model but also on  how the model is used. Several strategies have been used to improve the task of clinical extraction. Fine-tuning adapts the model to medical task by continue to train the model on domain specific knowledge, it is usually time consuming and requires large datasets. Alternatively, prompt design, task decomposition, and verification can improve extraction accuracy without retraining the model. Expert review can guide the process by identifying recurring errors and refine the extraction work. Less is known on how to turn the expert feedback into a workflow for note extraction.

<!-- physician:P011 -->
<!-- P011 empty -->

<!-- physician:P012 -->
We use inference harness to describe the execution workflow surrounding an LLM, including the sequence of extraction tasks, checks, and clinical rules applied without changing the model weights. We developed a locally deployable framework for extracting information from real-world unstructured oncology notes. The framework combines a structured inference harness engineering with a physician-preference-guided development loop. Our results showed that, the output from the harness is favored compared to a single-prompt baseline, we are able to adapt this harness to a second cancer type with model-assist development.

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
The study comprised two development stages followed by an independent clinical evaluation. During breast-cancer development, a physician coauthor and the model-development investigators iteratively reviewed extraction outputs, identified recurring errors, and revised the inference harness. The framework was then adapted to pancreatic cancer using model-assisted review, without clinician review of pancreatic-cancer outputs. The model-assisted review used the error categories and clinical distinctions established during breast-cancer development to identify potential errors and propose revisions, which the investigators assessed and regression-tested.

<!-- physician:P018 -->
<!-- P018 empty -->

<!-- physician:P019 -->
The final harness was compared with a single-prompt baseline using the same frozen model and field schema on 40 expert-annotated CORAL notes. Four physicians completed system-label-masked A/B evaluations across both cancer types. A secondary technical ablation compared four stages of inference-harness development using blinded LLM review.

<!-- technical insert after P019 -->

<!-- physician:P020 -->
<!-- P020 empty -->

<!-- physician:P021 -->
Dataset

<!-- physician:P022 -->
We used CORAL dataset, a PhysioNet dataset of 40 deidentified medical oncology notes obtain from the University of California, San Francisco. The development pool contains 100 breast-cancer and 100 pancreatic-cancer notes without expert annotation. The untouched holdout test set contains 20 expert-annotated notes for each cancer type. Access requires the PhysioNet credentialing and data-use process described by the dataset authors.

<!-- physician:P023 -->
<!-- P023 empty -->

<!-- physician:P024 -->
Baseline model and model development

<!-- physician:P025 -->
We used Qwen2.5-32B-Instruct-AWQ served locally through vLLM. The baseline model made one model call per note and returned the complete target schema. In the clinician review, the baseline model and the harness model used the same field definitions and output contract.

<!-- physician:P026 -->
<!-- P026 empty -->

<!-- physician:P027 -->
Inference Harness

<!-- physician:P028 -->
The harness extracted clinical information in multiple steps because different fields required different clinical context and reasoning. First, field-specific prompts independently identified the visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Information extracted in earlier steps was then used to guide subsequent assessments. For example, cancer stage and metastatic status provide context for treatment intent, while current therapy and clinical findings provide context for treatment response. Plan fields were extracted primarily from the Assessment and Plan section. The harness first used deterministic patterns to identify this section. If the section could not be identified, the model made up to three attempts to locate and reproduce the relevant text, with each attempt checked against the original note. If all attempts failed, plan-specific extraction was omitted rather than generating information not present in the source note.

<!-- physician:P029 -->
<!-- P029 empty -->

<!-- physician:P030 -->
Each model output passes through five verification stages. The first two address structural errors though JSON formatting repair and schema validation. The remaining three assess clinical content: Semantic alignment, faithfulness check, and temporal filtering. These verification stages ensure that the extracted information was complete, clinically relevant, and consistent with the source note. See table 1.

<!-- physician:P031 -->
<!-- P031 empty -->

<!-- physician:P032 -->
JSON formatting repair when parsing failed.

<!-- physician:P033 -->
Schema validation to detect incorrect or leaked keys.

<!-- physician:P034 -->
Specificity and semantic alignment checks to identify vague or off-task answers.

<!-- physician:P035 -->
Faithfulness trimming to remove clearly unsupported or contradictory content while retaining supported information.

<!-- physician:P036 -->
Temporal filtering to remove completed or historical events from future-plan fields.

<!-- physician:P037 -->
<!-- P037 empty -->

<!-- physician:P038 -->
We included clinical rules to address recurring extraction errors. For example, these rules help the system distinguish regional lymph node involvement from distant metastases, suspected from confirmed metastatic disease, and active anticancer treatment from supportive or home medications. They also prevented disease changes documented before treatment initiation from being misclassified as treatment response. The system returned supporting excerpts from the original clinical note when available.

<!-- technical insert after P038 -->

<!-- physician:P039 -->
<!-- P039 empty -->

<!-- physician:P040 -->
<!-- P040 empty -->

<!-- physician:P041 -->
<!-- P041 empty -->

<!-- physician:P042 -->
<!-- P042 empty -->

<!-- physician:P043 -->
<!-- P043 empty -->

<!-- physician:P044 -->
<!-- P044 empty -->

<!-- physician:P045 -->
<!-- P045 empty -->

<!-- physician:P046 -->
<!-- P046 empty -->

<!-- physician:P047 -->
<!-- P047 empty -->

<!-- physician:P048 -->
<!-- P048 empty -->

<!-- physician:P049 -->
<!-- P049 empty -->

<!-- physician:P050 -->
Table 1

<!-- physician:P051 -->
<!-- P051 empty -->

<!-- physician native table:T1; cells preserved in canonical JSON -->

<!-- physician:P052 -->
<!-- P052 empty -->

<!-- physician:P053 -->
<!-- P053 empty -->

<!-- physician:P054 -->
<!-- P054 empty -->

<!-- physician:P055 -->
Clinician-in-the-loop breast-cancer refinement

<!-- physician:P056 -->
<!-- P056 empty -->

<!-- physician:P057 -->
Breast-cancer development used all 100 development notes. Early and targeted iterations used smaller subsets to shorten the feedback cycle, while major and final runs covered all 100 notes to detect less frequent errors and check for regression. A physician reviewed selected outputs, and the investigator compared flagged outputs with the complete source note, grouped recurring errors, and translated them into general changes to prompts, verification logic, or deterministic rules. This physician author did not participate in the final evaluation. These changes were retained only after testing on affected examples. The objective was to encode recurring clinical distinctions rather than case-specific corrections. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general regional-node rule. An error in which a planned drug was listed as active therapy motivated a temporal medication rule. This process did not produce a conventional supervised training set, and the model weights were not updated.

<!-- physician:P058 -->
<!-- P058 empty -->

<!-- physician:P059 -->
Model-assist pancreatic-cancer refinement

<!-- physician:P060 -->
The harness developed from breast cancer notes was further adapted to pancreatic using 100 additional pancreatic cancer notes. No clinician reviewed pancreatic-cancer outputs during this stage. Instead, an external LLM (ChatGPT) was used to organize the clinical error categories that were found during breast cancer development into a structured review rubric. The Qwen pipeline generated structured outputs, and a rubric-guided reviewer Qwen model served as reviewer, compared these outputs with the full pancreatic notes to identify potential errors and areas for improvement. Investigators reviewed the errors and implemented selected revisions to prompts, such clinical rules, or workflow logic, and keep those changes only after regression testing.&#32;

<!-- physician:P061 -->
The adaptation retained the core framework while introducing pancreatic cancer-specific modifications, particularly for disease terminology, treatment regimens, and post-processing rules. The framework thus preserved its core approach to error detection and correction while accommodating differences between the two cancer types.

<!-- physician:P062 -->
Clinical evaluation data collection and scoring rubric

<!-- physician:P063 -->
The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were blinded from the harness and models. The physician participated in the human-in-the-loop development was not one of the final evaluators. Before evaluating the test set, the authors designed 7 core-fields as clinically important for understanding disease status and current management, see Supplement 1

<!-- physician:P064 -->
<!-- P064 empty -->

<!-- physician:P065 -->
<!-- P065 empty -->

<!-- physician:P066 -->
Staged Technical ablation

<!-- physician:P067 -->
We ran a secondary technical ablation on the same 20 breast-cancer and 20 pancreatic-cancer benchmark notes. All four conditions used Qwen2.5-32B-Instruct-AWQ, greedy decoding, and the same 31-field output contract. Attribution, tool calling, and letter generation were disabled.&#32;

<!-- physician:P068 -->
See Table 2 for the 4 conditions.

<!-- physician:P069 -->
<!-- P069 empty -->

<!-- physician:P070 -->
<!-- P070 empty -->

<!-- physician native table:T2; cells preserved in canonical JSON -->

<!-- physician:P071 -->
Table 2

<!-- physician:P072 -->
<!-- P072 empty -->

<!-- physician:P073 -->
We compared adjacent configurations (A vs B, B vs C, and C vs D), yielding 120 paired comparisons across 40 notes.  An AI judge (Codex) was given only the source note and the two outputs. The judge had no access to the project code, development history, prior evaluations, or system identities. The judge was instructed prioritized source faithfulness, then completeness, and used TIE when differences were minor or offsetting. We compared adjacent configurations using two-sided exact sign tests, excluding ties.

<!-- physician:P074 -->
<!-- P074 empty -->

<!-- physician:P075 -->
Statistical analysis

<!-- physician:P076 -->
For the physician evaluation, interrater agreement was assessed pairwise using exact agreement and Cohen’s κ for each evaluator pair. Results from the 8 completed clinician-by-cancer evaluations were summarized separately. Pooled counts were reported descriptively and were not treated as independent field-level observations.

<!-- physician:P077 -->
<!-- P077 empty -->

<!-- physician:P078 -->
For the primary inferential analysis, tied comparisons were excluded, and each remaining directional judgment was coded according to whether it favored the harness or the baseline. We fit a population-averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level to account for clustering of repeated field-level judgments within the same note. Evaluator was included as a fixed effect, and cancer type was additionally included in the overall model. Cancer-specific models adjusted for evaluator.&#32;

<!-- physician:P079 -->
<!-- P079 empty -->

<!-- physician:P080 -->
The primary outcome was the adjusted probability of harness preference among directional judgments. Predicted probabilities were standardized over the observed covariate distribution, and 95% confidence intervals (CIs) were calculated using the delta method on the logit scale. The difference from 50% was reported as the absolute preference effect, while adjusted odds ratios (ORs) were reported as secondary measures.

<!-- physician:P081 -->
<!-- P081 empty -->

<!-- physician:P082 -->
For sensitivity analysis, we refitted the overall model four times after excluding one physician at a time. We also retained tied judgments by coding harness, tie, and baseline preferences as 1, 0, and −1, respectively, and calculated the mean net preference margin with 95% CIs using note-clustered sandwich standard errors. Finally, we aggregated field-level preferences within each note and performed an exact two-sided sign test comparing positive and negative note-level margins.

<!-- technical insert after P082 -->

<!-- physician:P083 -->
<!-- P083 empty -->

## Results

<!-- physician:P084 -->
Results

<!-- physician:P085 -->
All 4 physicians evaluated both cancer types and completed 2160 comparisons. Across 2,160 required note-field judgments, the harness was preferred 699 times, 1,290 comparisons were ties, and the baseline was preferred 171 times, see Table1. These correspond to 32.36%, 59.72%, and 7.92% of all judgments. Among the 870 directional judgments, the crude harness share was 80.34% (note-cluster robust 95% CI 76.14% to 83.96%).&#32;

<!-- technical insert after P085 -->

<!-- physician:P086 -->
For breast cancer note, all four physicians independently favored the harness on the breast-cancer set. Their pooled breast result was 363 harness preferences, 80 baseline preferences, and 677 ties. The first physician's per-note result was 18 harness wins, one baseline win, and one tie. The second and third physicians each recorded a harness win in all 20 notes. The fourth recorded 16 harness wins, two baseline wins, and two ties. When the four breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin (Figure 2).&#32;

<!-- physician:P087 -->
For pancreatic cancer notes, the evaluations produced 336 harness preferences, 91 baseline preferences, and 613 ties. The harness had a positive pooled margin in 19 pancreatic notes and a negative margin in one.

<!-- technical insert after P087 -->

<!-- physician:P088 -->
<!-- P088 empty -->

<!-- physician:P089 -->
Table 2

<!-- physician:P090 -->
<!-- P090 empty -->

<!-- physician:P091 -->
Figure 2&#32;

<!-- physician:P092 -->
The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast-cancer estimate was 81.66% (95% CI, 76.02% to 86.21%), the pancreatic-cancer estimate was 78.15% (95% CI, 71.47% to 83.62). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p<0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p<0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p <0.05), see Table 3.&#32;

<!-- technical insert after P092 -->

<!-- physician:P093 -->
<!-- P093 empty -->

<!-- physician:P094 -->
Table 3&#32;

<!-- physician:P095 -->
Sensitivity analyses supported the primary findings. When ties were included, the net preference margin favored the harness (+24.4 percentage points; 95% CI, +20.4 to +28.5). At the note level, the harness was preferred in 39 of 40 cases (97.5%). Results remained consistent when each physician was excluded individually.

<!-- technical insert after P095 -->

<!-- physician:P096 -->
Performance across Clinical Extraction Fields

<!-- physician:P097 -->
Next, we analyzed which questions have the most influence on the results. In all the 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all 7 fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast cancer type and receptor status, where ties were more common (Figure 3).

<!-- technical insert after P097 -->

<!-- physician:P098 -->
<!-- P098 empty -->

<!-- physician:P099 -->
Figure 3

<!-- physician:P100 -->
We also analyzed physician preferences for each individual note to determine whether the overall advantage was consistent across cases. After combining physician ratings for each note, the harness was preferred in all 20 breast cancer notes and 19 of 20 pancreatic cancer notes. The overall ratings also favored the harness in each evaluator-by-cancer group. These findings were consistent with the primary analysis and suggest that the observed advantage was not driven by a small number of clinical notes (Figure 4).

<!-- physician:P101 -->
<!-- P101 empty -->

<!-- physician:P102 -->
<!-- P102 empty -->

<!-- physician:P103 -->
<!-- P103 empty -->

<!-- physician:P104 -->
<!-- P104 empty -->

<!-- physician:P105 -->
Figure 4 删掉4b,4c放在Supplement.

<!-- physician:P106 -->
<!-- P106 empty -->

## Discussion

<!-- physician:P107 -->
Discussion

<!-- physician:P108 -->
<!-- P108 empty -->

<!-- physician:P109 -->
In this study, the inference harness was preferred over a single-prompt baseline using the same underlying LLM. This advantage was observed in both breast and pancreatic cancer and remains consistent in both sensitivity analysis. These findings suggest that a structured inference harness can improve the clinical quality of extracted information without additional model training.&#32;

<!-- physician:P110 -->
<!-- P110 empty -->

<!-- physician:P111 -->
Structured extraction can convert long, complex, free-text note into a compact set of fields that can be checked, searched, and reused downstream. In oncology, this is useful as clinical notes can be complex and contain multiple diagnoses, prior and future treatments, and change in disease status. Prior studies have demonstrated the potential of LLMs for extracting structured information from clinical notes. However, the CORAL study showed that advanced, commercialized models could have omission or generate unsupported findings. Huang et al demonstrated that iterative prompt engineering with ChatGPT using API could improve extraction of free-text pathology notes. Similarly, Hein et al. showed that iterative refinement guided by clinicians and systematic error analysis could improve structured extraction and can be adapted to other cancer types.

<!-- physician:P112 -->
<!-- P112 empty -->

<!-- physician:P113 -->
Our study built on these studies by using physician feedback to identify recurring errors and develop clinical rules during breast cancer note development. Rather than producing a large, supervised training set, we used the feedback to develop field-specific prompts, verification steps, and clinical rules. This approach incorporated clinician knowledge into the inference harness and reused in subsequent extraction tasks. Instead of asking physicians to perform iterative evaluation, we used ChatGPT to organize the established criteria from breast cancer development, and used Qwen2.5-32B as a judge. LLM-as-judge has been reported and studies before, prior studies showed that LLM judges showed high concordance on text-comparison rubrics such as faithfulness and completeness. The final harness was preferred over the baseline by independent physician evaluators in pancreatic cancer. This finding suggests that criteria established through physician feedback may be transferable to other cancer types and may reduce the need for repeated physician review during development.

<!-- physician:P114 -->
<!-- P114 empty -->

<!-- physician:P115 -->
One recurring challenge we identified was the interpretation of temporal relationships in oncology notes, as reported by prior studies. Although the LLM could identify treatment responses documented in the clinical history, it sometimes failed to associate it with the correct disease status. For example, a response to a previous treatment could be incorrectly attributed to current regimen. This was particularly important for patients who had received multiple lines of therapy, where each treatment response must be associated with treatment history. We addressed this issue by incorporating treatment history and temporal context into the extraction process and adding verification steps to align treatment responses with the appropriate regimen. The physician evaluators preferred the harness than the baseline, suggesting that temporal checks could help reduce these errors.&#32;

<!-- physician:P116 -->
<!-- P116 empty -->

<!-- physician:P117 -->
Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure X).  Adding decomposed extraction alone did not produce an overall improvement while adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, we found that the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer, this suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce trade-offs, especially when the terminology or treatment differ between cancer types. For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC.

<!-- physician:P118 -->
<!-- P118 empty -->

<!-- physician:P119 -->
This study has several limitations. First, our primary outcome was physician preference rather than absolute extraction accuracy. A tie could indicate that both systems produced acceptable outputs or that both contained errors. Although physician ratings consistently favored the harness, the evaluation included only four physicians and 40 clinical notes from two cancer types within a single-source dataset. However, these were real clinical notes rather than synthetic cases or model-generated narratives. They contain repeated, copied-forward information, uncertain findings, treatment plans, or even typo or errors that make the extraction task challenging, nevertheless, further studies are needed to investigate if the results can be generalized to other cancer types or language models. Second, the exploratory ablation was evaluated by a single non-physician LLM judge and compared cumulative stages rather than individual components. Therefore, it could not determine the independent contribution of each verification step or clinical rule. Future studies should evaluate extraction accuracy against expert reference standards, include larger and more diverse clinical datasets, and assess whether the harness improves efficiency and reliability in clinical workflows.&#32;

<!-- physician:P120 -->
<!-- P120 empty -->

<!-- physician:P121 -->
<!-- P121 empty -->

<!-- physician:P122 -->
<!-- P122 empty -->

<!-- physician:P123 -->
<!-- P123 empty -->

<!-- physician:P124 -->
<!-- P124 empty -->

<!-- physician:P125 -->
<!-- P125 empty -->

<!-- physician:P126 -->
<!-- P126 empty -->


## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Physician comments

All 12 Word comments, including anchor text and paragraph context, are rendered in the HTML from the canonical JSON extraction.

## Author literature-review workspace

The HTML restores 26 annotated literature-review entries as a default-closed author workspace outside the manuscript body.
