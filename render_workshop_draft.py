#!/usr/bin/env python3
"""Build the English-only physician-authored manuscript HTML (version 1.3).

The canonical Word extraction is the sole prose source. Technical tables,
figures, captions, navigation, and supplementary analyses are inserted around
the physician blocks and are explicitly marked with data-origin attributes.
"""

from __future__ import annotations

import base64
import csv
import html
import importlib.util
import json
import re
import struct
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DOCX = ROOT / "manuscript/doctor_drafts/202601023_Manuscript_JAMA.docx"
CANONICAL_JSON = ROOT / "manuscript/doctor_drafts/202601023_Manuscript_JAMA.canonical.json"
EXTRACTOR = ROOT / "manuscript/doctor_drafts/extract_physician_source.py"
SOURCE_MD = ROOT / "WORKSHOP_PAPER_DRAFT.md"
OUTPUT_HTML = ROOT / "WORKSHOP_PAPER_DRAFT.html"
RATINGS_DIR = ROOT / "results/extraction_comparison/clinician_ratings"
FIGURE_DIR = RATINGS_DIR / "draft_figures"
EXPECTED_DOCX_SHA256 = "6a1ffc6809e66ad3861284274bdf001d110b4997bbdb053cdf666d4ef27e7066"
EXPECTED_VISIBLE_BLOCK_SHA256 = "5adf3d6434992babecc631e584ff5ba925983bd5036a28995e90b48474c577e3"

MAIN_FIGURES = {
    "1": FIGURE_DIR / "figure1_development_pathway_rough.svg",
    "2": FIGURE_DIR / "figure2_evaluator_distribution_rough.svg",
    "3": FIGURE_DIR / "figure3_interrater_matrix_rough.svg",
    "4": FIGURE_DIR / "figure4_effect_profile.svg",
}
SUPPLEMENTARY_FIGURES = {
    "S1": FIGURE_DIR / "supplementary_figure_s1_note_margins.svg",
    "S2": FIGURE_DIR / "supplementary_figure_s2_staged_ablation.svg",
}
PRINT_FIGURES = {
    "1": FIGURE_DIR / "figure1_development_pathway_print.png",
    "2": FIGURE_DIR / "figure2_evaluator_distribution_print.png",
    "3": FIGURE_DIR / "figure3_interrater_matrix_print.png",
    "4": FIGURE_DIR / "figure4_effect_profile_print.png",
    "S1": FIGURE_DIR / "supplementary_figure_s1_note_margins_print.png",
    "S2": FIGURE_DIR / "supplementary_figure_s2_staged_ablation_print.png",
}
PRINT_FIGURE_DIMENSIONS = {
    "1": (1500, 1180), "2": (1500, 720), "3": (1500, 760),
    "4": (1500, 940), "S1": (1500, 360), "S2": (1500, 760),
}

SECTION_HEADINGS = {
    "P002": ("introduction", "Introduction"),
    "P014": ("methods", "Methods"),
    "P084": ("results", "Results"),
    "P107": ("discussion", "Discussion"),
}
SUBHEADINGS = {
    "Study Design",
    "Dataset",
    "Baseline model and model development",
    "Inference Harness",
    "Clinician-in-the-loop breast-cancer refinement",
    "Model-assist pancreatic-cancer refinement",
    "Clinical evaluation data collection and scoring rubric",
    "Staged Technical ablation",
    "Statistical analysis",
    "Performance across Clinical Extraction Fields",
}
HIDDEN_SOURCE_BLOCKS = {"P050", "P071", "P089", "P091", "P094", "P099", "P105"}
OUTSTANDING_COMMENT_IDS = {"13", "14"}
RESOLVED_COMMENT_LITERATURE_MAP = {
    "0": 21, "1": 3, "2": 22, "3": 10, "4": 23,
    "8": 2, "9": 27, "10": 28, "11": 29, "12": 30,
}

REVISED_HEADINGS = {
    "P059": "Model-assisted pancreatic-cancer refinement",
    "P066": "Staged technical ablation",
    "P096": "Performance across clinical extraction fields",
}

REVISED_PARAGRAPHS = {
    "P004": "Recent years have witnessed increased use of large language models (LLMs) in clinical language tasks, including patient education, clinical documentation, and administrative work. Many of these applications rely on identifying relevant information from electronic health record (EHR) data before it can be summarized, analyzed, or used in another task. Reliable extraction from those notes is therefore an important step in applying LLMs to oncology care and research.",
    "P006": "Although EHRs contain structured fields, it is estimated that 80% of clinical data exist in unstructured form ([[cite:kong]]), and a large proportion of clinically relevant information is encoded in free-text narratives ([[cite:wiest]]). Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and changes in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real-world practice, copied-forward laboratory, imaging, and genomic results lengthen notes and further complicate the extraction task. A decade-long study of outpatient documentation found that notes became longer and more redundant, with much of the increase attributable to templates or copied and pasted text ([[cite:rule]]). LLMs must therefore not only identify the relevant information, but also determine its relationship to the patient's current diseases. ",
    "P008": "Currently, LLMs can assist with this task, but the output does not ensure that the extracted information is fully faithful to the original note. Models may generate false information, omit details, confuse historical findings, or infer plausible content where the note does not provide it. These errors can be difficult for human reviewers to detect. Commercial LLMs have been evaluated for clinical note extraction, but performance remains variable. The use of commercial models also raises privacy and governance concerns. The potential transmission of patients' protected health information (PHI) to externally hosted services can introduce institutional and patient concerns related to data processing, retention, and HIPAA compliance. A locally deployed open-weight model can process notes without sending PHI to outside sources. However, local models alone do not resolve extraction errors. ",
    "P010": "Local open-weight models are smaller, and their performance varies with model size. Recent studies suggest that the performance of LLMs depends not only on the model but also on how the model is used. Several strategies have been used to improve clinical extraction. Fine-tuning adapts the model to a medical task by continuing to train the model on domain-specific knowledge. This process is usually time-consuming and requires large datasets. Alternatively, prompt design, task decomposition, and verification can improve extraction accuracy without retraining the model. Expert review can guide the process by identifying recurring errors and refining the extraction workflow ([[cite:huang]]; [[cite:hein]]). Less is known about how to turn expert feedback into a workflow for note extraction.",
    "P012": "We use the term inference harness to describe the execution workflow surrounding an LLM, including the sequence of extraction tasks, checks, and clinical rules applied without changing the model weights. We developed a locally deployable framework for extracting information from real-world unstructured oncology notes. The framework combines structured inference-harness engineering with a physician-preference-guided development loop. Our results showed that the harness output was favored over a single-prompt baseline and that we could adapt the harness to a second cancer type through model-assisted development.",
    "P022": "We used the CORAL dataset, a PhysioNet dataset of 40 deidentified medical oncology notes obtained from the University of California, San Francisco. The development pool contains 100 breast-cancer and 100 pancreatic-cancer notes without expert annotation. The untouched holdout test set contains 20 expert-annotated notes for each cancer type. Access requires the PhysioNet credentialing and data-use process described by the dataset authors.",
    "P028": "The harness extracted clinical information in multiple steps because different fields required different clinical context and reasoning. First, field-specific prompts independently identified the visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Information extracted in earlier steps was then used to guide subsequent assessments. For example, cancer stage and metastatic status provide context for treatment intent, while current therapy and clinical findings provide context for treatment response. Plan fields were extracted primarily from the Assessment and Plan section. The harness first used deterministic patterns to identify this section. If the section could not be identified, the model made up to three attempts to locate and reproduce the relevant text, with each attempt checked against the original note. If all attempts failed, the system omitted plan-specific extraction rather than generate information not present in the source note.",
    "P030": "Each model output passes through five verification stages. The first two address structural errors through JSON formatting repair and schema validation. The remaining three assess clinical content: semantic alignment, a faithfulness check, and temporal filtering. These verification stages ensure that the extracted information is complete, clinically relevant, and consistent with the source note. See Table 1.",
    "P032": "JSON formatting repair when parsing fails.",
    "P038": "We included clinical rules to address recurring extraction errors. For example, these rules helped the system distinguish regional lymph node involvement from distant metastases, suspected metastatic disease from confirmed metastatic disease, and active anticancer treatment from supportive or home medications. They also prevented disease changes documented before treatment initiation from being misclassified as a treatment response. The system returned supporting excerpts from the original clinical note when available.",
    "P060": "The harness developed from breast-cancer notes was further adapted to pancreatic cancer using 100 additional pancreatic-cancer notes. No clinician reviewed pancreatic-cancer outputs during this stage. Instead, an external LLM (ChatGPT) was used to organize the clinical error categories that were found during breast-cancer development into a structured review rubric. The Qwen pipeline generated structured outputs, and a rubric-guided Qwen reviewer compared these outputs with the full pancreatic notes to identify potential errors and areas for improvement. Investigators reviewed the errors, implemented selected revisions to prompts, clinical rules, or workflow logic, and kept those changes only after regression testing. ",
    "P061": "The adaptation retained the core framework while introducing pancreatic-cancer-specific modifications, particularly for disease terminology, treatment regimens, and post-processing rules. The framework thus preserved its core approach to error detection and correction while accommodating differences between the two cancer types.",
    "P063": "The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were blinded to the identities of the systems. The physician who participated in the human-in-the-loop development was not one of the final evaluators. Before evaluating the test set, the authors designated seven core fields as clinically important for understanding disease status and current management. See Supplementary Table S1.",
    "P068": "See Table 2 for the 4 conditions.",
    "P073": "We compared adjacent configurations (A vs B, B vs C, and C vs D), yielding 120 paired comparisons across 40 notes. An AI judge (Codex) was given only the source note and the two outputs. The judge had no access to the project code, development history, prior evaluations, or system identities. The judge was instructed to prioritize source faithfulness, then completeness, and to use TIE when differences were minor or offsetting. We compared adjacent configurations using two-sided exact sign tests, excluding ties.",
    "P076": "For the physician evaluation, interrater agreement was assessed pairwise using exact agreement and Cohen's κ for each evaluator pair. Results from the 8 completed clinician-by-cancer evaluations were summarized separately. Pooled counts were reported descriptively and were not treated as independent field-level observations.",
    "P080": "The primary outcome was the adjusted probability of harness preference among directional judgments. Predicted probabilities were standardized over the observed covariate distribution, and 95% confidence intervals (CIs) were calculated using the delta method on the logit scale. Adjusted probabilities and adjusted odds ratios (ORs) were reported.",
    "P082": "For sensitivity analyses, we refitted the overall model four times after excluding one physician at a time. We also retained tied judgments by coding harness, tie, and baseline preferences as 1, 0, and −1, respectively, and calculated the mean net preference margin with 95% CIs using note-clustered sandwich standard errors. Finally, we aggregated field-level preferences within each note and performed an exact two-sided sign test comparing positive and negative note-level margins.",
    "P085": "All 4 physicians evaluated both cancer types and completed 2,160 comparisons. Across 2,160 required note-field judgments, the harness was preferred 699 times, 1,290 comparisons were ties, and the baseline was preferred 171 times. See Table 3, Table 4, and Figure 2. These correspond to 32.36%, 59.72%, and 7.92% of all judgments. Among the 870 directional judgments, the crude harness share was 80.34% (note-cluster-robust 95% CI 76.14% to 83.96%). ",
    "P086": "For breast-cancer notes, all four physicians independently favored the harness. Their pooled breast result was 363 harness preferences, 80 baseline preferences, and 677 ties. The first physician's per-note result was 18 harness wins, one baseline win, and one tie. The second and third physicians each recorded a harness win in all 20 notes. The fourth recorded 16 harness wins, two baseline wins, and two ties. When the four breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin (Supplementary Figure S1). ",
    "P092": "The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast-cancer estimate was 81.66% (95% CI, 76.02% to 86.21%), and the pancreatic-cancer estimate was 78.15% (95% CI, 71.47% to 83.62%). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p < 0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p < 0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p < 0.05). See Table 5. ",
    "P095": "Sensitivity analyses supported the primary findings. When ties were included, the net preference margin favored the harness (+24.4 percentage points; 95% CI, +20.4 to +28.5). At the note level, the harness was preferred in 39 of 40 cases (97.5%). Results remained consistent when each physician was excluded individually; adjusted harness-preference probabilities ranged from 77.08% to 83.53% (Supplementary Table S3).",
    "P097": "Next, we analyzed which questions had the greatest influence on the results. Across the 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all seven fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast-cancer type and receptor status, where ties were more common (Figure 4).",
    "P100": "We also analyzed physician preferences for each individual note to determine whether the overall advantage was consistent across cases. After combining physician ratings for each note, the harness was preferred in all 20 breast-cancer notes and 19 of 20 pancreatic-cancer notes. The overall ratings also favored the harness in each evaluator-by-cancer group. These findings were consistent with the primary analysis and suggest that the observed advantage was not driven by a small number of clinical notes (Supplementary Figure S1).",
    "P109": "In this study, the inference harness was preferred over a single-prompt baseline using the same underlying LLM. This advantage was observed in both breast and pancreatic cancer and remained consistent across the sensitivity analyses. These findings suggest that a structured inference harness can improve the clinical quality of extracted information without additional model training. ",
    "P111": "Structured extraction can convert long, complex, free-text notes into a compact set of fields that can be checked, searched, and reused downstream. In oncology, this is useful because clinical notes can be complex and contain multiple diagnoses, prior and future treatments, and changes in disease status. Prior studies have demonstrated the potential of LLMs for extracting structured information from clinical notes. However, the CORAL study showed that advanced commercial models could omit information or generate unsupported findings ([[cite:sushil]]). [[cite:huang_named]] demonstrated that iterative prompt engineering using ChatGPT via an API could improve extraction of free-text pathology notes. Similarly, [[cite:hein_named]] showed that iterative refinement guided by clinicians and systematic error analysis could improve structured extraction and be adapted to other cancer types.",
    "P113": "Our study built on these studies by using physician feedback to identify recurring errors and develop clinical rules during breast-cancer note development. Rather than producing a large, supervised training set, we used the feedback to develop field-specific prompts, verification steps, and clinical rules. This approach incorporated clinician knowledge into the inference harness and was reused in subsequent extraction tasks. Instead of asking physicians to perform iterative evaluation, we used ChatGPT to organize the established criteria from breast-cancer development, and used Qwen2.5-32B as a judge. LLM-as-a-judge approaches have been reported in prior studies, which showed that LLM judges had high concordance on text-comparison rubrics such as faithfulness and completeness ([[cite:gershon]]). The final harness was preferred over the baseline by independent physician evaluators for pancreatic cancer. This finding suggests that criteria established through physician feedback may be transferable to other cancer types and may reduce the need for repeated physician review during development.",
    "P115": "One recurring challenge we identified was the interpretation of temporal relationships in oncology notes. Prior studies have reported broader temporal reasoning challenges in clinical notes ([[cite:kruse]]; [[cite:kougia]]). Although the LLM could identify treatment responses documented in the clinical history, it sometimes failed to associate them with the correct disease status. For example, a response to a previous treatment could be incorrectly attributed to the current regimen. This was particularly important for patients who had received multiple lines of therapy, where each treatment response must be associated with the corresponding treatment history. We addressed this issue by incorporating treatment history and temporal context into the extraction process and adding verification steps to align treatment responses with the appropriate regimen. The physician evaluators preferred the harness over the baseline, suggesting that temporal checks could help reduce these errors. ",
    "P117": "Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure S2). Adding decomposed extraction alone did not produce an overall improvement, whereas adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, we found that the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer. This suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce trade-offs, especially when the terminology or treatment differs between cancer types. For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC.",
    "P119": "This study has several limitations. First, our primary outcome was physician preference rather than absolute extraction accuracy. A tie could indicate that both systems produced acceptable outputs or that both contained errors. Although physician ratings consistently favored the harness, the evaluation included only four physicians and 40 clinical notes from two cancer types within a single-source dataset. However, these were real clinical notes rather than synthetic cases or model-generated narratives. They contain repeated, copied-forward information, uncertain findings, treatment plans, or even typographical errors that make the extraction task challenging. Nevertheless, further studies are needed to investigate whether the results can be generalized to other cancer types or language models. Second, the exploratory ablation was evaluated by a single non-physician LLM judge and compared cumulative stages rather than individual components ([[cite:williams]]). Therefore, it could not determine the independent contribution of each verification step or clinical rule. Future studies should evaluate extraction accuracy against expert reference standards, include larger and more diverse clinical datasets, and assess whether the harness improves efficiency and reliability in clinical workflows. ",
}

GRAMMAR_REVISED_IDS = {
    "P004", "P006", "P008", "P010", "P012", "P022", "P028", "P030", "P032", "P038",
    "P060", "P061", "P063", "P073", "P076", "P082", "P085", "P086", "P092", "P097",
    "P100", "P109", "P111", "P113", "P115", "P117", "P119",
}

REVISED_TABLE_CELLS = {
    "T1R4C2": "The model returns medically related answers that do not address the target question",
    "T1R5C2": "The model infers a conclusion that is not supported by the note ",
    "T1R5C3": "Keep a lesion classified as suspected rather than label it as confirmed when a biopsy is pending",
    "T1R6C2": "Confusion among past, current, and future information.",
    "T1R6C3": "Remove a completed scan from the imaging plan",
}

XREF_TARGETS = {
    "Table 1": "table-1",
    "Table 2": "table-2",
    "Table 3": "table-3",
    "Table 4": "table-4",
    "Table 5": "table-5",
    "Supplementary Table S1": "supplementary-table-s1",
    "Supplementary Table S2": "supplementary-table-s2",
    "Supplementary Table S3": "supplementary-table-s3",
    "Figure 1": "figure-1",
    "Figure 2": "figure-2",
    "Figure 3": "figure-3",
    "Figure 4": "figure-4",
    "Supplementary Figure S1": "figure-s1",
    "Supplementary Figure S2": "figure-s2",
}

EXTERNAL_CITATIONS = {
    "kong": ("Kong, 2019", "https://e-hir.org/journal/view.php?id=10.4258/hir.2019.25.1.1", "Kong HJ. Managing Unstructured Big Data in Healthcare System. Healthcare Informatics Research. 2019;25(1):1-2."),
    "wiest": ("Wiest et al., 2024", "https://www.nature.com/articles/s41746-024-01233-2", "Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. npj Digital Medicine. 2024;7:257."),
    "rule": ("Rule et al., 2021", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8290305/", "Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. JAMA Network Open. 2021;4(7):e2115334."),
    "huang": ("Huang et al., 2024", "https://www.nature.com/articles/s41746-024-01079-8", "Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. npj Digital Medicine. 2024;7:106."),
    "huang_named": ("Huang et al. (2024)", "https://www.nature.com/articles/s41746-024-01079-8", "Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. npj Digital Medicine. 2024;7:106."),
    "hein": ("Hein et al., 2025", "https://www.nature.com/articles/s41746-025-01686-z", "Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. npj Digital Medicine. 2025;8:301."),
    "hein_named": ("Hein et al. (2025)", "https://www.nature.com/articles/s41746-025-01686-z", "Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. npj Digital Medicine. 2025;8:301."),
    "sushil": ("Sushil et al., 2024", "https://doi.org/10.1056/AIdbp2300110", "Sushil M, Kennedy VE, Mandair D, et al. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. NEJM AI. 2024;1(4)."),
    "gershon": ("Gershon et al., 2026", "https://www.nature.com/articles/s41746-026-03248-3", "Gershon R, Ben-Shlomo Y, Yitzhaki S, et al. Large language models as judges for clinical generative AI evaluation. npj Digital Medicine. Published online September 25, 2026."),
    "williams": ("Williams et al., 2026", "https://www.nature.com/articles/s41746-026-02992-w", "Williams G, Rutunda S, Nzabakira F, Mateen BA. Human evaluators vs. LLM-as-a-Judge: toward scalable evaluation of GenAI in global health. npj Digital Medicine. Published online July 20, 2026."),
    "kruse": ("Kruse et al., 2025", "https://aclanthology.org/2025.findings-emnlp.1128/", "Kruse M, Hu S, Derby N, et al. Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction. Findings of ACL: EMNLP 2025. 2025:20715-20735."),
    "kougia": ("Kougia et al., 2024", "https://aclanthology.org/2024.bionlp-1.6/", "Kougia V, Sedova A, Stephan AJ, Zaporojets K, Roth B. Analysing zero-shot temporal relation extraction on clinical notes using temporal consistency. Proceedings of BioNLP 2024. 2024:72-84."),
}


CSS = r"""
:root{--ink:#1e2832;--muted:#65717c;--navy:#173f5f;--blue:#0072b2;--orange:#d55e00;--tie:#b8b8b8;--green:#009e73;--paper:#fff;--page:#eef2f5;--line:#d5dde4;--soft:#f6f8fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--page);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;line-height:1.62}
.layout{display:grid;grid-template-columns:270px minmax(0,940px);gap:28px;width:min(1260px,calc(100% - 36px));margin:24px auto 70px;align-items:start}.toc{position:sticky;top:18px;max-height:calc(100vh - 36px);overflow:auto;padding:18px 15px;border:1px solid var(--line);border-radius:12px;background:#fffffff5}.toc strong{display:block;margin-bottom:9px;color:var(--navy)}.toc a{display:block;padding:4px 6px;border-radius:5px;color:#3e5160;text-decoration:none;font-size:12.5px}.toc a:hover{background:#eaf4fa}.toc .sub{padding-left:18px;font-size:11.5px;color:var(--muted)}
article{min-width:0;padding:45px 54px 68px;border:1px solid var(--line);border-radius:14px;background:var(--paper);box-shadow:0 12px 34px #142b4012}h1{margin:0;color:var(--navy);font-family:Georgia,"Times New Roman",serif;font-size:36px;line-height:1.18}h2,h3{color:var(--navy)}h3{margin:27px 0 9px;font-size:19px}.subtitle{margin:8px 0 20px;color:var(--muted)}
.revision-legend{margin:10px 0 18px;color:#55636e;font-size:12.5px}.revision-swatch{display:inline-block;width:.85em;height:.85em;margin-right:.35em;border:1px solid #e1c86c;border-radius:2px;background:#fff2b3;vertical-align:-.05em}mark.revision-highlight{background:#fff2b3;color:inherit;padding:.08em .12em;border-radius:2px;-webkit-box-decoration-break:clone;box-decoration-break:clone}.revision-paragraph{margin-top:9px}
details{border:0}summary{cursor:pointer}.fold>summary,.fold-toggle{padding:13px 16px;color:var(--navy);font-weight:750;list-style:none}.fold>summary::-webkit-details-marker,.version-entry>summary::-webkit-details-marker{display:none}.fold>summary::before,.fold-toggle::before,.version-entry>summary::before{content:"▸";display:inline-block;width:1.2em}.fold[open]>summary::before,.custom-fold.is-expanded>.fold-toggle::before,.version-entry[open]>summary::before{content:"▾"}.current-change{margin:14px 0 20px;padding:10px 13px;border-left:4px solid var(--blue);background:#eef7fb;color:#264b61;font-size:13px}.version-entry{margin:8px 0;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.version-entry>summary{padding:9px 11px;color:#30485a;font-weight:700;list-style:none}.version-entry-body{padding:0 13px 10px;color:#4b5964;font-size:13px}.version-entry-body p{margin:5px 0}
.fold{margin:22px 0;border-top:1px solid var(--line)}.fold>summary,.fold-toggle{margin:0;padding:20px 0 10px;font-family:Georgia,"Times New Roman",serif;font-size:25px}.fold-toggle{display:block;width:100%;border:0;background:transparent;text-align:left;cursor:pointer}.custom-fold.is-collapsed>.fold-body{display:none}.custom-fold.is-expanded>.fold-body{display:block}.fold-body{padding:0 0 8px}.source-empty,.source-audit{display:none!important}.physician-paragraph,.manuscript-paragraph{margin:11px 0}.physician-subheading{scroll-margin-top:15px}.xref,.citation,.comment-anchor a{color:var(--blue);text-decoration:none}.xref:hover,.citation:hover,.comment-anchor a:hover{text-decoration:underline}.comment-anchor{margin-left:.18em;font-size:.72em;vertical-align:super}.comment-anchor a{font-weight:650}.comment-backlink{display:inline-block;margin-top:7px;font-size:12px}.print-page-spacer{display:none}
.table-wrap{margin:20px 0 28px;overflow-x:auto}.table-caption,.figure-caption{margin:0 0 8px;color:#314452;font-size:13px}.table-caption strong,.figure-caption strong{color:var(--navy)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 9px;border:1px solid #d6dde3;vertical-align:top}th{background:#e9f1f6;color:#173f5f;text-align:left}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#fafbfc}.physician-native-table th{background:#eaf4ef;color:#18533f}
.figure{margin:24px 0 30px}.figure img{display:block;width:100%;height:auto;border:1px solid #e0e5e9;background:white}.figure-print{display:none!important}.figure-caption{margin-top:9px}.comment{margin:13px 0;padding:12px 14px;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.comment-meta{font-size:12px;color:var(--muted)}.comment-text{margin:7px 0;white-space:pre-wrap}.comment-context{margin-top:7px;color:#4e5c67;font-size:12px}.literature-workspace li{margin:11px 0}.reference-note{display:block;margin-top:3px;color:#4e5c67;font-size:12px}.workspace-note{color:#4e5c67;font-size:13px}.version-list li{margin:6px 0}.source-hash{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
@media(max-width:900px){.layout{display:block;width:min(100% - 18px,940px);margin:10px auto 40px}.toc{position:relative;top:auto;max-height:none;margin-bottom:12px}.toc a.sub{display:none}article{padding:30px 24px 48px}h1{font-size:30px}.metadata-grid{grid-template-columns:1fr}.table-wrap{margin-left:-8px;margin-right:-8px}.figure{margin-left:-8px;margin-right:-8px}}
@media(max-width:440px){article{padding:24px 15px 40px}h1{font-size:27px}.fold>summary,.fold-toggle{font-size:22px}table{font-size:11px}th,td{padding:6px 5px}.figure-caption,.table-caption{padding:0 3px}}
@media print{@page{margin:.62in}html{scroll-behavior:auto}body{background:#fff;font-family:Arial,sans-serif;font-size:10pt;line-height:1.42}.layout{display:block;width:auto;margin:0}.toc,.screen-only,#physician-comments,#version-history,#literature-workspace,.source-audit,.source-empty,.figure-screen{display:none!important}article{padding:0;border:0;box-shadow:none}.main-section,.supplement{display:block!important;border:0;margin:0}.main-section:not([open])>.fold-body,.supplement>.fold-body{display:block!important}.main-section>summary,.supplement>.fold-toggle{display:block!important;padding:16px 0 7px;list-style:none;break-after:avoid;font-size:19pt}.main-section>summary::before,.supplement>.fold-toggle::before{display:none!important}.physician-subheading{break-after:avoid}.figure,.table-wrap{break-inside:avoid;page-break-inside:avoid}.figure{margin:14px 0 18px}.figure .figure-print{display:block!important;width:100%;max-height:8.4in;object-fit:contain;border:0}.table-wrap{margin:12px 0 18px;overflow:visible}table{font-size:8.5pt}th,td{padding:5px 6px}.table-caption,.figure-caption{font-size:9pt}.comment-anchor{display:none!important}mark.revision-highlight{background:transparent!important;color:inherit;padding:0}}
html.print-export{scroll-behavior:auto}html.print-export body{background:#fff;font-family:Arial,sans-serif;font-size:13.333px;line-height:1.42;margin:0}html.print-export .layout{display:block;width:720px;margin:0}html.print-export .toc,html.print-export .screen-only,html.print-export #physician-comments,html.print-export #version-history,html.print-export #literature-workspace,html.print-export .source-audit,html.print-export .source-empty,html.print-export .figure-screen{display:none!important}html.print-export article{width:720px;padding:45px;box-sizing:border-box;border:0;box-shadow:none}html.print-export .main-section,html.print-export .supplement{display:block!important;border:0;margin:0}html.print-export .main-section>.fold-body,html.print-export .supplement>.fold-body{display:block!important}html.print-export .main-section>summary,html.print-export .supplement>.fold-toggle{display:block!important;padding:16px 0 7px;list-style:none;font-size:25px}html.print-export .main-section>summary::before,html.print-export .supplement>.fold-toggle::before{display:none!important}html.print-export .figure,html.print-export .table-wrap{break-inside:avoid}html.print-export .figure{margin:14px 0 18px}html.print-export .figure .figure-print{display:block!important;width:100%;max-height:806px;object-fit:contain;border:0}html.print-export .table-wrap{margin:12px 0 18px;overflow:visible}html.print-export table{font-size:11.333px}html.print-export th,html.print-export td{padding:5px 6px}html.print-export .table-caption,html.print-export .figure-caption{font-size:12px}html.print-export .comment-anchor{display:none!important}html.print-export mark.revision-highlight{background:transparent!important;color:inherit;padding:0}html.print-export .print-page-spacer{display:block!important;width:1px;visibility:hidden}
"""

SCRIPT = r"""
function setCustomFold(fold,open){if(!fold||!fold.classList.contains('custom-fold'))return;fold.classList.toggle('is-expanded',open);fold.classList.toggle('is-collapsed',!open);const button=fold.querySelector(':scope>.fold-toggle');if(button)button.setAttribute('aria-expanded',open?'true':'false')}
function revealTarget(target){if(!target)return;if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true;if(target.classList&&target.classList.contains('custom-fold'))setCustomFold(target,true);let p=target.parentElement;const chain=[];while(p){if(p.tagName==='DETAILS'||p.classList.contains('custom-fold'))chain.push(p);p=p.parentElement;}chain.reverse().forEach(d=>{if(d.tagName==='DETAILS')d.open=true;else setCustomFold(d,true)});if(!target.hasAttribute('tabindex'))target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start'})}
document.addEventListener('click',e=>{const toggle=e.target.closest('.fold-toggle');if(toggle){const fold=toggle.closest('.custom-fold');setCustomFold(fold,!fold.classList.contains('is-expanded'));return}const a=e.target.closest('a[href^="#"]');if(!a)return;const id=decodeURIComponent(a.getAttribute('href').slice(1));revealTarget(document.getElementById(id))});
window.addEventListener('hashchange',()=>revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1)))));window.addEventListener('DOMContentLoaded',()=>{if(location.hash)revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1))))});
window.preparePrintExport=function(pageHeight=960){if(!Number.isFinite(pageHeight)||pageHeight<400)throw new Error('Invalid page height');document.querySelectorAll('.print-page-spacer').forEach(node=>node.remove());document.documentElement.classList.add('print-export');void document.documentElement.offsetHeight;const candidates=[...document.querySelectorAll('.figure,.table-wrap')].filter(node=>getComputedStyle(node).display!=='none');for(const node of candidates){const rect=node.getBoundingClientRect();const top=rect.top+window.scrollY;const offset=((top%pageHeight)+pageHeight)%pageHeight;if(rect.height<pageHeight&&offset>0&&offset+rect.height>pageHeight-.5){const spacer=document.createElement('div');spacer.className='print-page-spacer';spacer.setAttribute('aria-hidden','true');spacer.style.height=`${Math.ceil(pageHeight-offset)+1}px`;node.before(spacer);void spacer.offsetHeight}}const items=candidates.map(node=>{const rect=node.getBoundingClientRect();const top=rect.top+window.scrollY;const bottom=rect.bottom+window.scrollY;return{id:node.id,kind:node.classList.contains('figure')?'figure':'table',top,bottom,height:rect.height,crosses:Math.floor((top+.5)/pageHeight)!==Math.floor((bottom-.5)/pageHeight)}});const printable=[...document.querySelectorAll('article>.main-section>.fold-body>*,#supplement-body>*')].filter(node=>!node.classList.contains('print-page-spacer')&&getComputedStyle(node).display!=='none');const contentBottom=Math.ceil(Math.max(0,...printable.map(node=>node.getBoundingClientRect().bottom+window.scrollY)));const revisionHighlights=[...document.querySelectorAll('mark.revision-highlight')];return{pageHeight,documentHeight:document.documentElement.scrollHeight,contentBottom,supplementVisible:getComputedStyle(document.getElementById('supplement-body')).display!=='none',revisionHighlightCount:revisionHighlights.length,revisionHighlightBackgrounds:revisionHighlights.map(node=>getComputedStyle(node).backgroundColor),items,offenders:items.filter(item=>item.crosses).map(item=>item.id)}};
window.resetPrintExport=function(){document.querySelectorAll('.print-page-spacer').forEach(node=>node.remove());document.documentElement.classList.remove('print-export')};
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def markdown_source_text(value: str) -> str:
    """Keep source trailing spaces visible without Git whitespace errors."""

    trailing = len(value) - len(value.rstrip(" "))
    return value.rstrip(" ") + "&#32;" * trailing


def load_canonical() -> dict[str, object]:
    subprocess.run(
        [sys.executable, str(EXTRACTOR), str(SOURCE_DOCX), "--output", str(CANONICAL_JSON), "--check"],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
    assert data["source_docx_sha256"] == EXPECTED_DOCX_SHA256
    assert data["canonical_visible_block_sha256"] == EXPECTED_VISIBLE_BLOCK_SHA256
    assert data["counts"] == {
        "body_paragraphs": 127,
        "nonempty_body_paragraphs": 62,
        "native_tables": 2,
        "images": 5,
        "comments": 12,
        "tracked_changes": 0,
        "visible_words": 3335,
    }
    return data


def svg_data_uri(path: Path) -> str:
    data = path.read_bytes()
    if len(data) >= 209_715_200:
        raise ValueError(f"Figure exceeds 200 MiB limit: {path}")
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


def validate_print_png(number: str, path: Path) -> None:
    data = path.read_bytes()
    if len(data) >= 5_242_880:
        raise ValueError(f"Print figure exceeds 5 MiB limit: {path}")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Print figure is not PNG: {path}")
    dimensions = struct.unpack(">II", data[16:24])
    if dimensions != PRINT_FIGURE_DIMENSIONS[number]:
        raise ValueError(f"Unexpected print figure dimensions for {number}: {dimensions}")


def link_crossrefs(text: str) -> str:
    xref_pattern = "|".join(re.escape(label) for label in sorted(XREF_TARGETS, key=len, reverse=True))
    pattern = re.compile(r"\[\[cite:([a-z_]+)\]\]" + ("|" + xref_pattern if xref_pattern else ""))
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(esc(text[cursor:match.start()]))
        label = match.group(0)
        citation_key = match.group(1)
        if citation_key:
            if citation_key not in EXTERNAL_CITATIONS:
                raise KeyError(f"Unknown external citation token: {citation_key}")
            citation_label, url, title = EXTERNAL_CITATIONS[citation_key]
            parts.append(
                f'<a class="citation external-citation" href="{esc(url)}" target="_blank" rel="noopener" '
                f'title="{esc(title)}" aria-label="Open source: {esc(title)}">{esc(citation_label)}</a>'
            )
        else:
            parts.append(f'<a class="xref" href="#{XREF_TARGETS[label]}">{esc(label)}</a>')
        cursor = match.end()
    parts.append(esc(text[cursor:]))
    return "".join(parts)


def source_audit(block_id: str, text: str) -> str:
    return (
        f'<span class="source-audit" hidden aria-hidden="true" data-origin="physician" '
        f'data-source-block="{esc(block_id)}">{esc(text)}</span>'
    )


def hidden_comment_audit_text(text: str) -> str:
    """Preserve canonical text while keeping resolved tracking URLs inert."""

    return esc(text).replace("utm_source", "utm&#95;source").replace("fromPaywallRec", "fromPaywall&#82;ec")


def comment_anchors(comment_ids: list[str]) -> str:
    return "".join(
        f'<sup class="comment-anchor" id="comment-anchor-{esc(comment_id)}">'
        f'<a href="#physician-comment-{esc(comment_id)}" aria-label="Comment {esc(comment_id)}">{esc(comment_id)}</a></sup>'
        for comment_id in comment_ids if comment_id in OUTSTANDING_COMMENT_IDS
    )


def manuscript_paragraph(text: str, *, paragraph_id: str | None = None) -> str:
    id_attr = f' id="{esc(paragraph_id)}"' if paragraph_id else ""
    return f'<p class="manuscript-paragraph" data-origin="technical-insert"{id_attr}>{link_crossrefs(text)}</p>'


def revision_paragraph(text: str, *, paragraph_id: str) -> str:
    return (
        f'<p class="manuscript-paragraph revision-paragraph" data-origin="technical-insert" id="{esc(paragraph_id)}">'
        f'<mark class="revision-highlight">{link_crossrefs(text)}</mark></p>'
    )


def render_physician_paragraph(paragraph: dict[str, object], comment_ids: list[str]) -> str:
    block_id = str(paragraph["id"])
    text = str(paragraph["raw_visible_text"])
    anchors = comment_anchors(comment_ids)
    if block_id in HIDDEN_SOURCE_BLOCKS:
        return source_audit(block_id, text) + anchors
    if block_id in REVISED_PARAGRAPHS:
        return (
            source_audit(block_id, text)
            + f'<p class="manuscript-paragraph" id="{esc(block_id)}" data-origin="manuscript-edit">'
            + link_crossrefs(REVISED_PARAGRAPHS[block_id])
            + anchors
            + "</p>"
        )
    if block_id in REVISED_HEADINGS:
        revised = REVISED_HEADINGS[block_id]
        slug = re.sub(r"[^a-z0-9]+", "-", revised.lower()).strip("-")
        return source_audit(block_id, text) + f'<h3 class="physician-subheading" id="{slug}" data-origin="manuscript-edit">{esc(revised)}{anchors}</h3>'
    source_span = (
        f'<span data-origin="physician" data-source-block="{esc(block_id)}">{esc(text)}</span>'
    )
    if not text:
        return f'<div class="source-empty" id="{esc(block_id)}">{source_span}</div>{anchors}'
    if text in SUBHEADINGS:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return f'<h3 class="physician-subheading" id="{slug}">{source_span}{anchors}</h3>'
    return f'<p class="physician-paragraph" id="{esc(block_id)}">{source_span}{anchors}</p>'


def render_native_table(table: dict[str, object], label: str, title: str) -> str:
    rows = table["rows"]
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{esc(label.lower().replace(" ", "-"))}">',
        f'<p class="table-caption"><strong>{esc(label)}.</strong> {esc(title)}</p>',
        '<table class="physician-native-table">',
    ]
    for row_index, row in enumerate(rows):
        parts.append("<thead><tr>" if row_index == 0 else "<tbody><tr>")
        tag = "th" if row_index == 0 else "td"
        for cell in row:
            cell_id = str(cell["id"])
            raw_text = str(cell["raw_visible_text"])
            if cell_id in REVISED_TABLE_CELLS:
                parts.append(
                    f'<{tag} data-origin="manuscript-edit">{source_audit(cell_id, raw_text)}'
                    f'{esc(REVISED_TABLE_CELLS[cell_id])}</{tag}>'
                )
            else:
                parts.append(
                    f'<{tag} data-origin="physician" data-source-block="{esc(cell_id)}">'
                    f'{esc(raw_text)}</{tag}>'
                )
        parts.append("</tr></thead>" if row_index == 0 else "</tr></tbody>")
    parts.extend(["</table>", "</div>"])
    return "".join(parts)


def result_table(label: str, title: str, headers: list[str], rows: list[list[str]], *, table_id: str) -> str:
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{table_id}">',
        f'<p class="table-caption"><strong>{esc(label)}.</strong> {esc(title)}</p>',
        "<table><thead><tr>",
    ]
    for index, header in enumerate(headers):
        parts.append(f'<th class="{"num" if index else ""}">{esc(header)}</th>')
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, value in enumerate(row):
            parts.append(f'<td class="{"num" if index else ""}">{esc(value)}</td>')
        parts.append("</tr>")
    parts.extend(["</tbody></table></div>"])
    return "".join(parts)


def figure(number: str, title: str, caption: str, *, supplementary: bool = False) -> str:
    path = SUPPLEMENTARY_FIGURES[number] if supplementary else MAIN_FIGURES[number]
    print_path = PRINT_FIGURES[number]
    validate_print_png(number, print_path)
    print_src = print_path.relative_to(ROOT).as_posix()
    label = f"Supplementary Figure {number}" if supplementary else f"Figure {number}"
    attr = "data-supplementary-figure" if supplementary else "data-main-figure"
    return (
        f'<figure class="figure" id="figure-{number.lower()}" data-origin="technical-insert" {attr}="{number}">'
        f'<img class="figure-screen" src="{svg_data_uri(path)}" alt="{esc(title)}">'
        f'<img class="figure-print" src="{esc(print_src)}" alt="" aria-hidden="true">'
        f'<figcaption class="figure-caption"><strong>{esc(label)}.</strong> {esc(caption)}</figcaption></figure>'
    )


def table_3() -> str:
    return result_table(
        "Table 3", "Physician preferences for breast-cancer extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "280", "84", "22", "174", "79.2%"], ["02", "280", "79", "8", "193", "90.8%"], ["03", "280", "119", "24", "137", "83.2%"], ["04", "280", "81", "26", "173", "75.7%"], ["All", "1,120", "363", "80", "677", "81.9%"]],
        table_id="table-3",
    )


def table_4() -> str:
    return result_table(
        "Table 4", "Physician preferences for PDAC extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "260", "75", "14", "171", "84.3%"], ["02", "260", "86", "9", "165", "90.5%"], ["03", "260", "103", "30", "127", "77.4%"], ["04", "260", "72", "38", "150", "65.5%"], ["All", "1,040", "336", "91", "613", "78.7%"]],
        table_id="table-4",
    )


def table_5() -> str:
    return result_table(
        "Table 5", "Adjusted probability of inference-harness preference among directional judgments",
        ["Scope", "Directional judgments", "Harness", "Baseline", "Note clusters", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        [["Overall", "870", "699", "171", "40", "79.98% (75.75%-83.63%)", "3.995 (3.124-5.110)", "2.67e-28"], ["Breast cancer", "443", "363", "80", "20", "81.66% (76.02%-86.21%)", "4.452 (3.169-6.253)", "7.04e-18"], ["PDAC", "427", "336", "91", "20", "78.15% (71.47%-83.62%)", "3.576 (2.505-5.105)", "2.30e-12"]],
        table_id="table-5",
    )


def load_agreement_rows() -> list[list[str]]:
    module_path = RATINGS_DIR / "plot_manuscript_figures.py"
    spec = importlib.util.spec_from_file_location("manuscript_plots", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raters = module.load_raters()
    names = list(raters)
    rows: list[list[str]] = []
    for cancer, prefix, fields in [("Breast cancer", "b", module.BREAST_FIELDS), ("PDAC", "p", module.PDAC_FIELDS)]:
        for left in range(4):
            for right in range(left + 1, 4):
                agreement, kappa, count = module.pairwise_agreement(raters[names[left]], raters[names[right]], prefix, fields)
                rows.append([cancer, f"Physician {left + 1} vs Physician {right + 1}", str(count), f"{agreement:.1%}", f"{kappa:.3f}"])
    return rows


def supplementary_table_1() -> str:
    return result_table(
        "Supplementary Table S1", "Core fields used in the primary clinical comparison",
        ["Core field", "Applicability", "Rating field", "Evaluation definition"],
        [
            ["Active anticancer medications", "Breast cancer and PDAC", "Current_Medications.current_meds", "Anticancer drugs currently being taken or administered; supportive or home medications, completed or stopped drugs, and planned-only drugs are excluded."],
            ["Stage", "Breast cancer and PDAC", "Cancer_Diagnosis.Stage_of_Cancer", "Cancer stage, TNM, or resectability status, retaining original and current time context when both are documented."],
            ["Distant metastasis", "Breast cancer and PDAC", "Cancer_Diagnosis.Distant Metastasis", "Presence, absence, sites, and certainty of distant spread; regional nodes are not counted as distant metastasis."],
            ["Regional or overall metastasis", "Breast cancer and PDAC", "Cancer_Diagnosis.Metastasis", "Regional nodal and distant involvement, with the site and certainty of each supported finding."],
            ["Treatment response", "Breast cancer and PDAC", "Response_Assessment.response_assessment", "Response to the current or most recently completed anticancer treatment, linked to the applicable regimen and supporting evidence."],
            ["Breast cancer type and receptors", "Breast cancer only", "Cancer_Diagnosis.Type_of_Cancer", "Breast histology and the individual ER, PR, and HER2 results when documented."],
            ["Completed molecular or genetic results", "Breast cancer and PDAC", "Genetic_Testing_Results.genetic_testing_results", "Completed germline, somatic, or other molecular results; pending or merely ordered tests are excluded."],
        ],
        table_id="supplementary-table-s1",
    )


def supplementary_table_2() -> str:
    return result_table(
        "Supplementary Table S2", "Pairwise exact agreement and Cohen's kappa",
        ["Cancer", "Physician pair", "Shared judgments", "Exact agreement", "Cohen's kappa"],
        load_agreement_rows(), table_id="supplementary-table-s2",
    )


def supplementary_table_3() -> str:
    rows: list[list[str]] = []
    with (RATINGS_DIR / "leave_one_physician_out_gee.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append([row["excluded_physician"], row["directional_judgments"], row["clusters"], f'{float(row["adjusted_probability"]):.2%} ({float(row["probability_ci_low"]):.2%}-{float(row["probability_ci_high"]):.2%})', f'{float(row["odds_ratio"]):.3f} ({float(row["ci_low"]):.3f}-{float(row["ci_high"]):.3f})', f'{float(row["p_value"]):.3g}'])
    return result_table(
        "Supplementary Table S3", "Leave-one-physician-out sensitivity analysis of the overall directional GEE",
        ["Excluded physician", "Directional judgments", "Notes", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        rows, table_id="supplementary-table-s3",
    )


def render_comments(data: dict[str, object]) -> str:
    parts = []
    for comment in data["comments"]:
        comment_id = str(comment["id"])
        if comment_id not in OUTSTANDING_COMMENT_IDS:
            literature_entry = RESOLVED_COMMENT_LITERATURE_MAP[comment_id]
            parts.append(
                f'<span class="source-audit" hidden aria-hidden="true" data-origin="physician-comment" '
                f'data-comment-text-id="{esc(comment_id)}" data-status="resolved" '
                f'data-resolved-literature-entry="{literature_entry}">{hidden_comment_audit_text(str(comment["raw_visible_text"]))}</span>'
            )
            continue
        parts.append(
            f'<div class="comment" id="physician-comment-{esc(comment_id)}" data-origin="physician-comment" data-comment-id="{esc(comment_id)}">'
            f'<div class="comment-meta">Comment {esc(comment_id)} · {esc(comment["author"])} · {esc(comment["date"])}</div>'
            f'<div class="comment-text" data-origin="physician" data-comment-text-id="{esc(comment_id)}">{esc(comment["raw_visible_text"])}</div>'
            f'<div class="comment-context"><strong>Anchor:</strong> <a href="#{esc(comment["anchor_paragraph_id"])}">{esc(comment["anchor_paragraph_id"])}</a><br>'
            f'<strong>Selected text:</strong> {esc(comment["anchor_selected_text"])}<br>'
            f'<strong>Paragraph context:</strong> {esc(comment["anchor_paragraph_context"])}<br>'
            f'<a class="comment-backlink" href="#comment-anchor-{esc(comment_id)}">Back to comment anchor</a></div></div>'
        )
    return "".join(parts)


LITERATURE_ENTRIES = [
    ("Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. JMIR Cancer. 2025;11:e65984.", "https://doi.org/10.2196/65984", "This review maps 24 oncology information-extraction studies and provides the broad literature frame used to limit our novelty claims."),
    ("Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. NEJM AI. 2024;1(4).", "https://doi.org/10.1056/AIdbp2300110", "CORAL supplies our real longitudinal breast and pancreatic oncology notes and establishes the zero-shot benchmark that our harness extends."),
    ("Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. npj Digital Medicine. 2024;7:257.", "https://www.nature.com/articles/s41746-024-01233-2", "This study shows that locally deployed open models, prompt design, and constrained output can support privacy-preserving clinical extraction."),
    ("Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. JAMIA Open. 2024;7(3):ooae060.", "https://doi.org/10.1093/jamiaopen/ooae060", "This longitudinal lung-cancer study compares model families and rule-based methods, while our experiment holds the model fixed and changes the inference workflow."),
    ("Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. JCO Clinical Cancer Informatics. 2025;9:e2500002.", "https://doi.org/10.1200/CCI-25-00002", "This large hybrid breast-cancer system has external validation and supervised fine-tuning, providing a stronger scale comparator but a different adaptation strategy."),
    ("Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. JAMIA Open. 2025;8(5):ooaf097.", "https://doi.org/10.1093/jamiaopen/ooaf097", "Its engineered preload, schema and range validation, and retry loop are the closest published architectural precedent for our verification stages."),
    ("Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. Communications Medicine. 2025;5:422.", "https://doi.org/10.1038/s43856-025-01116-x", "mCODEGPT shows that hierarchical prompting can outperform a single-step strategy, although it uses synthetic oncology notes and does not evaluate a full verification harness."),
    ("Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. Communications Medicine. 2025;5:96.", "https://doi.org/10.1038/s43856-025-00808-8", "This prostate-pathology study documents substantial effects from model, prompt, language, and quantization choices on a narrower report type."),
    ("Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025.", "https://arxiv.org/abs/2412.15115", "This report documents the base model family used in both study conditions; the inference harness is our added system layer."),
    ("Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. npj Digital Medicine. 2024;7:106.", "https://www.nature.com/articles/s41746-024-01079-8", "Iterative prompt engineering achieved strong pathology extraction but still exposed specialized terminology and TNM-staging errors that motivate explicit clinical checks."),
    ("van Koevorden JW, Aben N, Struben V, et al. Validating large language model-assisted data extraction from clinical notes. ESMO Real World Data and Digital Oncology. 2026;12:100718.", "https://doi.org/10.1016/j.esmorw.2026.100718", "Six experts evaluated 29 extracted categories in head and neck oncology, making this a close clinical-validation comparator with stronger reference-based error classification than our preference study."),
    ("Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. Communications Medicine. 2026.", "https://doi.org/10.1038/s43856-026-01790-5", "This study shows that few-shot design and clinical expertise improve locally deployable models, while our work adds multi-stage verification and deterministic cross-field rules."),
    ("Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. JCO Clinical Cancer Informatics. 2026;10:e2500388.", "https://doi.org/10.1200/CCI-25-00388", "This study used 700 annotated notes for model development and testing, then processed more than two million notes across five cancers; our contribution instead concerns inference-time controls across a broader schema without weight updates."),
    ("Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. JCO Clinical Cancer Informatics. 2026;10:e2600002.", "https://doi.org/10.1200/CCI-26-00002", "Its senior-physician-adjudicated comparison found metastasis classification easier than treatment-response assessment, matching the field-dependent difficulty seen in our results."),
    ("Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. JCO Clinical Cancer Informatics. 2026;10:e2500226.", "https://doi.org/10.1200/CCI-25-00226", "Its comparison of simple prompting, chain-of-thought, double filtering, and rules provides direct evidence that inference design changes oncology extraction quality even when the target is a single field."),
    ("Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems. 2023;36:46595-46623.", "https://doi.org/10.52202/075280-2020", "This work supports LLM judges as scalable development tools but documents position, verbosity, self-enhancement, and reasoning biases that preclude treating them as our clinical endpoint."),
    ("Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024:9440-9450.", "https://doi.org/10.18653/v1/2024.acl-long.511", "This paper isolates positional bias in LLM comparison judgments and reinforces the need for human final evaluation and cautious interpretation of automated review."),
    ("Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. Nature Medicine. 2025;31:60-69.", "https://doi.org/10.1038/s41591-024-03425-5", "TRIPOD-LLM supports explicit reporting of models, prompts, evaluation procedures, task-specific metrics, and human oversight in this pilot."),
    ("Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. npj Digital Medicine. 2024;7:258.", "https://doi.org/10.1038/s41746-024-01258-7", "Its review of 142 studies and QUEST framework directly inform evaluator selection, rating dimensions, reliability assessment, and adjudication for a stronger follow-up study."),
    ("Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. JCO Clinical Cancer Informatics. 2026;10:e2500215.", "https://doi.org/10.1200/CCI-25-00215", "VALID requires variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication, clarifying why our preference endpoint is not yet an absolute correctness assessment."),
    ("Kong HJ. Managing Unstructured Big Data in Healthcare System. Healthcare Informatics Research. 2019;25(1):1-2.", "https://e-hir.org/journal/view.php?id=10.4258/hir.2019.25.1.1", "This article supports the broad estimate that much healthcare information remains unstructured; the estimate is not oncology-specific."),
    ("Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. JAMA Network Open. 2021;4(7):e2115334.", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8290305/", "This study of nearly 3 million outpatient notes documents increasing note length and redundancy and the contribution of copied and templated text."),
    ("Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. npj Digital Medicine. 2025;8:301.", "https://www.nature.com/articles/s41746-025-01686-z", "This study provides a close precedent for human-in-the-loop refinement, error ontology development, and general instructions rather than case-specific fixes."),
    ("Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. Nature Medicine. 2024;30:1134-1142.", "https://doi.org/10.1038/s41591-024-02855-5", "Its blinded reader study used a scale with an equivalent midpoint, supporting retention of ties in comparative clinical evaluation."),
    ("Ayers JW, Poliak A, Dredze M, et al. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum. JAMA Internal Medicine. 2023;183(6):589-596.", "https://doi.org/10.1001/jamainternmed.2023.1838", "This physician-rated comparison reported the preference proportion with a confidence interval, providing a clear precedent for probability-scale reporting."),
    ("Goh E, Gallo R, Hom J, et al. Large Language Model Influence on Diagnostic Reasoning. JAMA Network Open. 2024;7(10):e2440969.", "https://doi.org/10.1001/jamanetworkopen.2024.40969", "Its mixed-effects analysis included participant and case effects, illustrating how repeated clinical observations should be modeled rather than treated as independent."),
    ("Gershon R, Ben-Shlomo Y, Yitzhaki S, et al. Large language models as judges for clinical generative AI evaluation. npj Digital Medicine. Published online September 25, 2026. doi:10.1038/s41746-026-03248-3.", "https://www.nature.com/articles/s41746-026-03248-3", "This study supports rubric-specific LLM judging for faithfulness and completeness while showing weaker agreement for reasoning and safety."),
    ("Williams G, Rutunda S, Nzabakira F, Mateen BA. Human evaluators vs. LLM-as-a-Judge: toward scalable evaluation of GenAI in global health. npj Digital Medicine. Published online July 20, 2026. doi:10.1038/s41746-026-02992-w.", "https://www.nature.com/articles/s41746-026-02992-w", "This study shows that LLM-judge agreement depends on the model, rubric, language, and clinical setting and does not replace final physician evaluation."),
    ("Kruse M, Hu S, Derby N, et al. Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction. Findings of ACL: EMNLP 2025. 2025:20715-20735. doi:10.18653/v1/2025.findings-emnlp.1128.", "https://aclanthology.org/2025.findings-emnlp.1128/", "This study supports the statement that temporal progression across long clinical records remains difficult for LLMs."),
    ("Kougia V, Sedova A, Stephan AJ, Zaporojets K, Roth B. Analysing zero-shot temporal relation extraction on clinical notes using temporal consistency. Proceedings of BioNLP 2024. 2024:72-84. doi:10.18653/v1/2024.bionlp-1.6.", "https://aclanthology.org/2024.bionlp-1.6/", "This study documents difficulty in clinical temporal-relation extraction and shows that temporal consistency alone does not establish accuracy."),
]


def literature_workspace() -> str:
    entries = "".join(
        f'<li id="lit-{index}" data-literature-entry="{index}">{esc(citation)} '
        f'<a href="{esc(url)}" target="_blank" rel="noopener" title="Open verified source for literature entry {index}" '
        f'aria-label="Open verified source for literature entry {index}">{esc(url)}</a>'
        f'<span class="reference-note"><strong>Draft relevance:</strong> {esc(relevance)}</span></li>'
        for index, (citation, url, relevance) in enumerate(LITERATURE_ENTRIES, 1)
    )
    return (
        '<details class="fold literature-workspace" id="literature-workspace">'
        '<summary>Author literature-review workspace (not manuscript body)</summary><div class="fold-body">'
        '<p class="workspace-note">These historical entries support drafting and discussion development; every citation and link requires pre-submission verification.</p>'
        f'<ol>{entries}</ol></div></details>'
    )


def technical_inserts_after(block_id: str) -> str:
    inserts = {
        "P019": manuscript_paragraph("The study workflow is shown in Figure 1.", paragraph_id="workflow-cross-reference"),
        "P022": revision_paragraph("The expert annotations for the holdout notes were not provided to either system, were not used to develop the inference harness, and were not used to calculate the primary physician-preference endpoint.", paragraph_id="revision-annotation-use"),
        "P038": figure("1", "Development, adaptation, and evaluation workflow", "Development, transfer, and evaluation pathway. Breast-cancer physician review informed the inference harness. The failure history and clinical rubric then supported author-controlled PDAC adaptation without target-domain physician review of outputs. Four physicians compared the final harness with the same-model single-prompt baseline on the 40-note holdout set."),
        "P063": revision_paragraph("Output A was always the inference-harness output, and output B was always the single-prompt baseline output. Physicians were blinded to these system identities. Source attribution was available only in the inference-harness output, so the comparison evaluated the full system output packages rather than any isolated component.", paragraph_id="revision-evaluation-package"),
        "P082": manuscript_paragraph("Exploratory core-field analysis used one joint population-averaged logistic GEE on 495 directional core judgments. Field, physician, and cancer type were fixed effects; repeated judgments were clustered by note with an exchangeable working correlation. Field-specific probabilities were standardized over the physician and cancer distribution observed for that field. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded. These estimates describe preference rather than accuracy, and no pairwise field tests or multiplicity claims were made.", paragraph_id="core-field-statistical-analysis"),
        "P085": table_3() + table_4(),
        "P087": figure("2", "Physician preference overview", "Physician preferences across eight physician-by-cancer evaluations. Panel A retains harness, baseline, and tie judgments. Panel B shows the harness share among directional judgments."),
        "P092": table_5(),
        "P095": manuscript_paragraph("Pairwise exact agreement ranged from 73.1% to 82.9%, with Cohen's kappa ranging from 0.511 to 0.646 (Figure 3 and Supplementary Table S2).", paragraph_id="agreement-results") + figure("3", "Inter-physician agreement", "Pairwise exact agreement and Cohen's kappa among four physicians in breast cancer and PDAC. Each breast-cancer pair shared 280 judgments, and each PDAC pair shared 260 judgments."),
        "P097": figure("4", "Core-field raw and adjusted preference estimates", "Core-field preference results. Panel A retains the tie-inclusive raw distribution for seven physician-prioritized fields. Panel B shows core-field adjusted harness-preference probabilities with 95% confidence intervals from one joint exploratory GEE fitted to 495 directional core judgments. The model included field, physician, and cancer fixed effects, clustered repeated judgments by note with an exchangeable working correlation, and standardized each field over its observed physician and cancer distribution. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded from Panel B. The estimates report preference, not accuracy. No pairwise field tests or multiplicity claims were made."),
        "P119": revision_paragraph("Both systems used the same base-model family, so this study did not evaluate whether the findings generalize across model families.", paragraph_id="revision-model-family-limitation"),
    }
    return inserts.get(block_id, "")


def build_scientific_sections(data: dict[str, object]) -> tuple[str, str]:
    paragraphs = {item["id"]: item for item in data["paragraphs"]}
    tables = {item["id"]: item for item in data["tables"]}
    comment_ids_by_paragraph: dict[str, list[str]] = {}
    for comment in data["comments"]:
        comment_ids_by_paragraph.setdefault(str(comment["anchor_paragraph_id"]), []).append(str(comment["id"]))
    front_blocks: list[str] = []
    sections: list[str] = []
    markdown: list[str] = []
    current_parts: list[str] = []
    current_md: list[str] = []
    current_key = ""
    current_title = ""

    def flush() -> None:
        nonlocal current_parts, current_md
        if not current_key:
            return
        heading_id = next(key for key, value in SECTION_HEADINGS.items() if value[0] == current_key)
        heading = paragraphs[heading_id]
        heading_span = f'<span id="{heading_id}" data-origin="physician" data-source-block="{heading_id}">{esc(heading["raw_visible_text"])}</span>'
        sections.append(f'<details class="fold main-section" id="section-{current_key}" open><summary>{heading_span}</summary><div class="fold-body">{"".join(current_parts)}</div></details>')
        markdown.extend([f"## {current_title}", "", f'<!-- physician:{heading_id} -->', current_title, ""])
        markdown.extend(current_md)
        current_parts = []
        current_md = []

    for entry in data["body_sequence"]:
        if entry["kind"] == "paragraph":
            block_id = entry["id"]
            paragraph = paragraphs[block_id]
            if block_id in SECTION_HEADINGS:
                flush()
                current_key, current_title = SECTION_HEADINGS[block_id]
                continue
            rendered = render_physician_paragraph(paragraph, comment_ids_by_paragraph.get(block_id, []))
            if not current_key:
                front_blocks.append(rendered)
                markdown.extend([f'<!-- physician:{block_id} -->', markdown_source_text(str(paragraph["raw_visible_text"])) if paragraph["raw_visible_text"] else f'<!-- {block_id} empty -->', ""])
                continue
            current_parts.append(rendered)
            text = str(paragraph["raw_visible_text"])
            current_md.extend([f'<!-- physician:{block_id} -->', markdown_source_text(text) if text else f'<!-- {block_id} empty -->', ""])
            insert = technical_inserts_after(block_id)
            if insert:
                current_parts.append(insert)
                current_md.extend([f'<!-- technical insert after {block_id} -->', ""])
        elif entry["kind"] == "table":
            table_id = entry["id"]
            if table_id == "T1":
                current_parts.append(render_native_table(tables[table_id], "Table 1", "Inference harness components"))
            elif table_id == "T2":
                current_parts.append(render_native_table(tables[table_id], "Table 2", "Staged technical ablation conditions"))
            current_md.extend([f'<!-- physician native table:{table_id}; cells preserved in canonical JSON -->', ""])
    flush()
    return "".join(front_blocks + sections), "\n".join(markdown)


def build_supplement() -> str:
    return (
        '<section class="fold supplement custom-fold is-collapsed" id="supplementary-material" data-origin="technical-insert">'
        '<button class="fold-toggle" type="button" aria-expanded="false" aria-controls="supplement-body">Supplementary material</button>'
        '<div class="fold-body" id="supplement-body">'
        + supplementary_table_1()
        + supplementary_table_2()
        + figure("S1", "Note-level preference-margin distribution", "This figure shows the note-level distribution after pooling all four physicians within each cancer type, previously drafted as Panel C of the main figure. The margin is (harness preferences minus baseline preferences) divided by completed judgments for that note. All 20 breast-cancer notes and 19 of 20 PDAC notes had positive margins; one PDAC note had a negative margin. The prior detailed breast and PDAC note panels are not retained. This is a supporting preference sensitivity analysis, not an accuracy estimate.", supplementary=True)
        + figure("S2", "Staged technical ablation", "Blinded staged technical ablation across three cumulative transitions. A was the single-prompt baseline; B added decomposition and orchestration; C added semantic, faithfulness, and temporal verification; D added deterministic inline processing and the complete POST-hook layer. One non-physician Codex judge produced these secondary technical judgments. Exact two-sided sign tests exclude ties.", supplementary=True)
        + supplementary_table_3()
        + "</div></section>"
    )


def version_history() -> str:
    entries = [
        ("v1.3", "2026-10-30", "Applied conservative grammar and punctuation edits, added direct verified author-year citations, retained two outstanding physician comments, and highlighted three new methods or limitation clarifications for review."),
        ("v1.2.3", "2026-10-27", "Removed the empty trailing WebKit export page by ending capture at the last printable manuscript element, and added explicit failure cleanup with a post-write fault-injection test."),
        ("v1.2.2", "2026-10-27", "Replaced the supplementary native disclosure with an accessible custom fold so print media always contains the complete supplement, and added safe-boundary pagination preparation for bounded WebKit PDF verification."),
        ("v1.2.1", "2026-10-27", "Restricted print output to the clean manuscript and formal supplement, excluded author-only workspaces and comments from print, and added bounded WebKit print regression coverage."),
        ("v1.2", "2026-10-26", "Converted the review draft into a clean manuscript view, made the main tables sequential, corrected all approved table and figure cross-references, retained the 12 Word comments in a neutral end section, and restored the 26-entry author literature-review workspace."),
        ("v1.1", "2026-10-26", "Removed the draft-information panel, consolidated six open author queries with bidirectional review links, retained only the approved note-distribution panel in Supplementary Figure S1, and made each version-history entry independently collapsible."),
        ("v1.0", "2026-10-26", "Replaced the prior bilingual draft with an English-only rendering of the physician Word manuscript as the sole prose source, added exact source-preservation checks, split the cancer-specific result tables, and rebuilt the main and supplementary figure set."),
        ("v0.25", "2026-10-13", "Added the leave-one-physician-out GEE sensitivity analysis and its supplementary table."),
        ("v0.24", "2026-10-12", "Restored the physician-authored point that fluent outputs can conceal extraction errors."),
        ("v0.23", "2026-10-12", "Made manuscript sections collapsible, added anchor-driven disclosure opening, and preserved complete print output."),
        ("v0.22", "2026-10-11", "Consolidated the main result figures and moved the fully labeled note-level result to the supplement."),
        ("v0.21", "2026-10-11", "Clarified the staged ablation, standardized figure color semantics, and changed the adjusted-effect display to probability scale."),
        ("v0.20", "2026-10-11", "Added probability-scale adjusted effects alongside the odds-ratio results."),
        ("v0.19", "2026-10-11", "Redrew the figure set in a common visual system and consolidated the judge protocol."),
        ("v0.18", "2026-10-10", "Integrated four physicians across two cancer types and revised the study framing around extraction rather than letter generation."),
        ("v0.17", "Date not recorded", "Established the first tracked multi-physician analysis and staged technical ablation draft."),
    ]
    return "".join(
        f'<details class="version-entry" id="version-{esc(version[1:].replace(".", "-"))}" data-origin="technical-insert">'
        f'<summary>{esc(version)} · {esc(date)}</summary><div class="version-entry-body"><p>{esc(note)}</p></div></details>'
        for version, date, note in entries
    )


def toc() -> str:
    links = [("#section-introduction", "Introduction", False), ("#section-methods", "Methods", False)]
    for heading in ["Study Design", "Dataset", "Baseline model and model development", "Inference Harness", "Clinician-in-the-loop breast-cancer refinement", "Model-assisted pancreatic-cancer refinement", "Clinical evaluation data collection and scoring rubric", "Staged technical ablation", "Statistical analysis"]:
        links.append(("#" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-"), heading, True))
    links.extend([("#section-results", "Results", False), ("#performance-across-clinical-extraction-fields", "Performance across clinical extraction fields", True), ("#section-discussion", "Discussion", False), ("#supplementary-material", "Supplementary material", False), ("#physician-comments", "Outstanding physician comments (2)", False), ("#version-history", "Version history", False), ("#literature-workspace", "Literature-review workspace", False)])
    return '<nav class="toc" data-origin="technical-insert"><strong>Contents</strong>' + "".join(f'<a class="{"sub" if sub else ""}" href="{href}" data-origin="technical-insert">{esc(label)}</a>' for href, label, sub in links) + "</nav>"


def build_html(data: dict[str, object]) -> tuple[str, str]:
    sections, markdown_body = build_scientific_sections(data)
    article = f"""
<article>
  <h1 class="screen-only">Physician-authored manuscript draft</h1>
  <p class="subtitle screen-only">English working manuscript · Version 1.3 · October 30, 2026</p>
  <p class="current-change screen-only"><strong>Current change:</strong> conservative grammar edits, verified author-year citations, and three highlighted methods or limitation clarifications.</p>
  <p class="revision-legend screen-only"><span class="revision-swatch" aria-hidden="true"></span>Highlighted text marks new methods or limitation clarifications for physician review.</p>
  {sections}
  {build_supplement()}
  <details class="fold" id="physician-comments"><summary>Outstanding physician comments (2)</summary><div class="fold-body">{render_comments(data)}</div></details>
  <details class="fold" id="version-history"><summary>Version history</summary><div class="fold-body">{version_history()}</div></details>
  {literature_workspace()}
</article>
"""
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physician-authored manuscript draft v1.3</title><style>{CSS}</style></head><body><div class="layout">{toc()}{article}</div><script>{SCRIPT}</script></body></html>'
    markdown = f"""<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 1.3, October 30, 2026.

Current change: conservative grammar edits, verified author-year citations, and three highlighted methods or limitation clarifications.

The canonical physician blocks below mirror `{SOURCE_DOCX.name}` for audit purposes. The clean HTML applies only the approved cross-reference and reporting edits while preserving all source blocks in its hidden audit representation.

{markdown_body}

## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Outstanding physician comments

The canonical extraction retains all 12 Word comments. The HTML displays the 2 unresolved comments and keeps the 10 resolved literature comments only in its hidden audit representation.

## Author literature-review workspace

The HTML includes 30 annotated literature-review entries as a default-closed author workspace outside the manuscript body.
"""
    return document, markdown


class PreservationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str | None, str | None, list[str], bool, bool]] = []
        self.paragraphs: dict[str, str] = {}
        self.paragraph_order: list[str] = []
        self.cells: dict[str, str] = {}
        self.cell_order: list[str] = []
        self.comments: dict[str, str] = {}
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.main_figures: list[str] = []
        self.supplementary_figures: list[str] = []
        self.technical_table_containers: list[str] = []
        self.literature_entries: list[str] = []
        self.comment_anchor_ids: list[str] = []
        self.comment_backlinks: list[str] = []
        self.visible_comment_ids: list[str] = []
        self.external_links: list[dict[str, str]] = []
        self.revision_highlights = 0
        self.hidden_depth = 0
        self.non_manuscript_depth = 0
        self.visible_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"): self.ids.append(str(values["id"]))
        if values.get("href", "").startswith("#"): self.hrefs.append(str(values["href"])[1:])
        if values.get("href", "").startswith(("https://", "http://")):
            self.external_links.append({key: str(value) for key, value in values.items() if value is not None})
        if values.get("data-main-figure"): self.main_figures.append(str(values["data-main-figure"]))
        if values.get("data-supplementary-figure"): self.supplementary_figures.append(str(values["data-supplementary-figure"]))
        if values.get("data-literature-entry"): self.literature_entries.append(str(values["data-literature-entry"]))
        if str(values.get("id", "")).startswith("comment-anchor-"): self.comment_anchor_ids.append(str(values["id"]))
        if "comment-backlink" in str(values.get("class", "")).split(): self.comment_backlinks.append(str(values.get("href", "")))
        if "comment" in str(values.get("class", "")).split() and values.get("data-comment-id") is not None:
            self.visible_comment_ids.append(str(values["data-comment-id"]))
        if "revision-highlight" in str(values.get("class", "")).split(): self.revision_highlights += 1
        if "table-wrap" in str(values.get("class", "")).split():
            if values.get("data-origin") != "technical-insert": raise AssertionError("Generated table container lacks technical-insert provenance")
            self.technical_table_containers.append(str(values.get("id", "")))
        capture_type = None
        capture_id = None
        source_block = values.get("data-source-block")
        if values.get("data-origin") == "physician" and source_block:
            capture_type = "cell" if str(source_block).startswith("T") else "paragraph"
            capture_id = str(source_block)
        elif values.get("data-comment-text-id") is not None:
            capture_type = "comment-text"
            capture_id = str(values["data-comment-text-id"])
        starts_hidden = "hidden" in values or "source-audit" in str(values.get("class", "")).split()
        if starts_hidden: self.hidden_depth += 1
        starts_non_manuscript = values.get("id") in {"physician-comments", "version-history", "literature-workspace"}
        if starts_non_manuscript: self.non_manuscript_depth += 1
        self.stack.append((tag, capture_type, capture_id, [], starts_hidden, starts_non_manuscript))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs); self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0 and self.non_manuscript_depth == 0: self.visible_parts.append(data)
        for _, capture_type, _, parts, _, _ in self.stack:
            if capture_type in {"paragraph", "cell", "comment-text"}: parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack: return
        open_tag, capture_type, capture_id, parts, starts_hidden, starts_non_manuscript = self.stack.pop()
        if open_tag != tag: return
        if capture_type == "paragraph" and capture_id:
            self.paragraphs[capture_id] = "".join(parts); self.paragraph_order.append(capture_id)
        elif capture_type == "cell" and capture_id:
            self.cells[capture_id] = "".join(parts); self.cell_order.append(capture_id)
        elif capture_type == "comment-text" and capture_id: self.comments[capture_id] = "".join(parts)
        if starts_hidden: self.hidden_depth -= 1
        if starts_non_manuscript: self.non_manuscript_depth -= 1

    @property
    def visible_text(self) -> str:
        return " ".join("".join(self.visible_parts).split())


def validate(document: str, markdown: str, data: dict[str, object]) -> None:
    parser = PreservationParser(); parser.feed(document)
    expected_paragraphs = {item["id"]: item["raw_visible_text"] for item in data["paragraphs"]}
    expected_cells = {cell["id"]: cell["raw_visible_text"] for table in data["tables"] for row in table["rows"] for cell in row}
    expected_comments = {item["id"]: item["raw_visible_text"] for item in data["comments"]}
    if parser.paragraphs != expected_paragraphs:
        missing = sorted(set(expected_paragraphs) - set(parser.paragraphs)); changed = sorted(key for key in expected_paragraphs if parser.paragraphs.get(key) != expected_paragraphs[key])
        raise AssertionError(f"Physician paragraphs changed: missing={missing}, changed={changed}")
    if parser.paragraph_order != list(expected_paragraphs): raise AssertionError("Physician paragraph sequence changed")
    if parser.cells != expected_cells: raise AssertionError("Physician native table cells changed")
    if parser.cell_order != list(expected_cells): raise AssertionError("Physician native table-cell sequence changed")
    if parser.comments != expected_comments: raise AssertionError("Physician comments changed")
    markdown_physician_ids = re.findall(r"<!-- physician:(P\d{3}) -->", markdown)
    expected_markdown_ids = list(expected_paragraphs)
    if markdown_physician_ids != expected_markdown_ids:
        raise AssertionError(f"Generated Markdown physician sequence changed: expected {len(expected_markdown_ids)}, found {len(markdown_physician_ids)}")
    if len(parser.ids) != len(set(parser.ids)): raise AssertionError("Duplicate HTML IDs")
    missing_anchors = sorted(set(parser.hrefs) - set(parser.ids))
    if missing_anchors: raise AssertionError(f"Missing internal anchors: {missing_anchors}")
    if parser.main_figures != ["1", "2", "3", "4"]: raise AssertionError(f"Unexpected main figures: {parser.main_figures}")
    if parser.supplementary_figures != ["S1", "S2"]: raise AssertionError(f"Unexpected supplementary figures: {parser.supplementary_figures}")
    expected_table_containers = ["table-1", "table-2", "table-3", "table-4", "table-5", "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3"]
    if parser.technical_table_containers != expected_table_containers: raise AssertionError(f"Unexpected generated table containers: {parser.technical_table_containers}")
    for label in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3"]:
        if document.count(f"<strong>{label}.</strong>") != 1: raise AssertionError(f"Table label missing or duplicated: {label}")
    if parser.literature_entries != [str(index) for index in range(1, 31)]: raise AssertionError("The 30-entry literature workspace is incomplete")
    if '<details class="fold literature-workspace" id="literature-workspace">' not in document: raise AssertionError("Literature workspace is missing")
    if "if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true" not in document or "target.focus({preventScroll:true})" not in document:
        raise AssertionError("Closed-details target reveal behavior is missing")
    print_requirements = [
        "@media print", ".main-section:not([open])>.fold-body,.supplement>.fold-body{display:block!important}",
        ".toc,.screen-only,#physician-comments,#version-history,#literature-workspace,.source-audit,.source-empty,.figure-screen{display:none!important}",
        ".figure .figure-print{display:block!important;width:100%;max-height:8.4in;object-fit:contain;border:0}",
        "window.preparePrintExport=function(pageHeight=960)", "contentBottom=Math.ceil", "offenders:items.filter(item=>item.crosses)",
        "mark.revision-highlight{background:transparent!important;color:inherit;padding:0}",
    ]
    if any(rule not in document for rule in print_requirements):
        raise AssertionError("Print view does not isolate and constrain the clean manuscript")
    if document.count('class="figure-print"') != 6 or document.count('class="figure-screen"') != 6:
        raise AssertionError("Print-safe figure fallbacks are incomplete")
    for print_path in PRINT_FIGURES.values():
        source = print_path.relative_to(ROOT).as_posix()
        if document.count(f'src="{source}"') != 1:
            raise AssertionError(f"Print-safe figure is not linked exactly once: {source}")
    if '<section class="fold supplement custom-fold is-collapsed" id="supplementary-material" data-origin="technical-insert">' not in document:
        raise AssertionError("Accessible default-closed supplementary fold is missing")
    if '<button class="fold-toggle" type="button" aria-expanded="false" aria-controls="supplement-body">' not in document:
        raise AssertionError("Supplementary fold control is incomplete")
    for closed_id in ["physician-comments", "version-history", "literature-workspace"]:
        opening = re.search(rf'<details[^>]*id="{closed_id}"[^>]*>', document)
        if not opening or re.search(r"\sopen(?:\s|=|>)", opening.group(0)):
            raise AssertionError(f"Screen-only fold is not default-closed: {closed_id}")
    expected_comment_anchors = [f"comment-anchor-{item['id']}" for item in data["comments"] if str(item["id"]) in OUTSTANDING_COMMENT_IDS]
    if parser.comment_anchor_ids != expected_comment_anchors: raise AssertionError("Comment anchors are incomplete")
    if parser.comment_backlinks != [f"#comment-anchor-{item['id']}" for item in data["comments"] if str(item["id"]) in OUTSTANDING_COMMENT_IDS]: raise AssertionError("Comment backlinks are incomplete")
    if parser.visible_comment_ids != ["13", "14"]: raise AssertionError(f"Unexpected visible physician comments: {parser.visible_comment_ids}")
    if "Outstanding physician comments (2)" not in document: raise AssertionError("Outstanding-comment heading is missing")
    for comment_id, literature_entry in RESOLVED_COMMENT_LITERATURE_MAP.items():
        marker = f'data-comment-text-id="{comment_id}" data-status="resolved" data-resolved-literature-entry="{literature_entry}"'
        if marker not in document: raise AssertionError(f"Resolved literature comment mapping is missing: {comment_id}")
    if parser.revision_highlights != 3: raise AssertionError(f"Expected 3 revision highlights, found {parser.revision_highlights}")
    for revision_id in ["revision-annotation-use", "revision-evaluation-package", "revision-model-family-limitation"]:
        if f'id="{revision_id}"' not in document: raise AssertionError(f"Highlighted clarification is missing: {revision_id}")
    if 'id="draft-metadata"' in document or "Draft information and source provenance" in document: raise AssertionError("Removed draft-information block remains")
    if "Version 1.3 · October 30, 2026" not in document or "draft v1.3" not in document: raise AssertionError("Version 1.3 metadata is missing")
    if document.count('class="version-entry"') != 16 or any(f'id="version-{item}"' not in document for item in ["1-3", "1-2-3", "1-2-2", "1-2-1", "1-2", "1-1", "1-0", "0-25", "0-24", "0-23", "0-22", "0-21", "0-20", "0-19", "0-18", "0-17"]):
        raise AssertionError("Nested version history is incomplete")
    if 'lang="zh"' in document or 'id="chinese-version"' in document: raise AssertionError("Chinese manuscript shell remains")
    visible_text_lower = parser.visible_text.lower()
    if "—" in parser.visible_text or "–" in parser.visible_text: raise AssertionError("Visible manuscript contains an em or en dash")
    if "[22]" in parser.visible_text: raise AssertionError("Orphaned numeric citation remains")
    if "83.62)." in parser.visible_text or "83.62)" in REVISED_PARAGRAPHS["P092"]: raise AssertionError("P092 confidence interval is missing its percent sign")
    if re.search(r"[\u4e00-\u9fff]", parser.visible_text): raise AssertionError("Chinese text remains in the visible manuscript")
    for forbidden in ["technical insert", "physician source note", "review marker", "[todo]", "figure 4 删掉", "difference from 50%", "(citation)"]:
        if forbidden in visible_text_lower: raise AssertionError(f"Clean manuscript contains review artifact: {forbidden}")
    for obsolete in ["author-queries", "review-marker", "source-highlight", "physician-editorial", "technical-note", "class=\"badge\""]:
        if obsolete in document: raise AssertionError(f"Removed review UI remains: {obsolete}")
    for source_block in HIDDEN_SOURCE_BLOCKS:
        if f'data-source-block="{source_block}"' not in document or f'id="{source_block}"' in document:
            raise AssertionError(f"Hidden source placeholder audit is incomplete: {source_block}")
    if not GRAMMAR_REVISED_IDS.issubset(REVISED_PARAGRAPHS): raise AssertionError("Grammar revision registry is incomplete")
    citation_token = re.compile(r"\s*\[\[cite:[a-z_]+\]\]")
    for paragraph_id in sorted(GRAMMAR_REVISED_IDS):
        original = expected_paragraphs[paragraph_id]
        revised = citation_token.sub("", REVISED_PARAGRAPHS[paragraph_id])
        if original == revised: raise AssertionError(f"Grammar revision is unchanged: {paragraph_id}")
        original_words = re.findall(r"[A-Za-z0-9]+", original.lower())
        revised_words = re.findall(r"[A-Za-z0-9]+", revised.lower())
        if SequenceMatcher(None, original_words, revised_words).ratio() < 0.75:
            raise AssertionError(f"Grammar revision exceeded conservative edit distance: {paragraph_id}")
    if set(REVISED_HEADINGS) != {"P059", "P066", "P096"}: raise AssertionError("Revised heading registry changed")
    if set(REVISED_TABLE_CELLS) != {"T1R4C2", "T1R5C2", "T1R5C3", "T1R6C2", "T1R6C3"}: raise AssertionError("Table 1 grammar revision registry changed")
    for revised_cell in REVISED_TABLE_CELLS.values():
        if esc(revised_cell) not in document: raise AssertionError(f"Revised Table 1 text is missing: {revised_cell}")
    expected_inline_urls = []
    for paragraph in data["paragraphs"]:
        revised = REVISED_PARAGRAPHS.get(str(paragraph["id"]), "")
        for citation_key in re.findall(r"\[\[cite:([a-z_]+)\]\]", revised):
            expected_inline_urls.append(EXTERNAL_CITATIONS[citation_key][1])
    actual_inline_urls = re.findall(r'<a class="citation external-citation" href="([^"]+)"', document)
    if actual_inline_urls != expected_inline_urls: raise AssertionError("External author-year citation links are incomplete or out of order")
    expected_external_urls = expected_inline_urls + [entry[1] for entry in LITERATURE_ENTRIES]
    actual_external_urls = [link["href"] for link in parser.external_links]
    if Counter(actual_external_urls) != Counter(expected_external_urls): raise AssertionError("External links differ from the verified URL registry")
    if "utm_source=" in document or "fromPaywallRec=" in document: raise AssertionError("Tracking parameters remain in rendered HTML")
    for link in parser.external_links:
        if link.get("target") != "_blank" or link.get("rel") != "noopener" or not link.get("title") or not link.get("aria-label"):
            raise AssertionError(f"External link lacks safe accessible attributes: {link.get('href')}")
        if "utm_" in link["href"].lower() or "frompaywall" in link["href"].lower():
            raise AssertionError(f"Tracking parameter remains in external URL: {link['href']}")
    required_crossrefs = {
        "P030": "#table-1", "P063": "#supplementary-table-s1", "P068": "#table-2",
        "P085": "#table-3", "P086": "#figure-s1", "P092": "#table-5",
        "P095": "#supplementary-table-s3", "P097": "#figure-4", "P100": "#figure-s1", "P117": "#figure-s2",
    }
    for paragraph_id, href in required_crossrefs.items():
        match = re.search(rf'id="{paragraph_id}"[^>]*>(.*?)</p>', document, flags=re.S)
        if not match or f'href="{href}"' not in match.group(1): raise AssertionError(f"Cross-reference missing from {paragraph_id}: {href}")
    if 'href="#table-4"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1) or 'href="#figure-2"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1):
        raise AssertionError("P085 does not cite Tables 3-4 and Figure 2")
    if 'href="#figure-3"' not in document or 'href="#supplementary-table-s2"' not in document: raise AssertionError("Agreement Results cross-references are missing")
    for stale in ["figure5_staged_ablation.svg", "figure6_clinical_handoff.svg", 'data-main-figure="5"', 'data-main-figure="6"']:
        if stale in document: raise AssertionError(f"Retired main figure remains: {stale}")
    if len(markdown.split()) < 3000: raise AssertionError("Generated Markdown is unexpectedly short")
    if len(document.encode("utf-8")) >= 209_715_200: raise AssertionError("HTML exceeds 200 MiB")


def main() -> None:
    data = load_canonical()
    document, markdown = build_html(data)
    validate(document, markdown, data)
    SOURCE_MD.write_text(markdown, encoding="utf-8")
    OUTPUT_HTML.write_text(document, encoding="utf-8")
    print(f"Wrote {SOURCE_MD}")
    print(f"Wrote {OUTPUT_HTML}")
    print("Preserved 162/162 physician text blocks (127 paragraphs and 35 native-table cells) and 12/12 comments; rendered 4 main and 2 supplementary figures.")


if __name__ == "__main__":
    main()
