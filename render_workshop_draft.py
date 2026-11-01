#!/usr/bin/env python3
"""Build the English-only physician-authored manuscript HTML (version 1.6.7).

The canonical Word extraction is the sole prose source. Technical tables,
figures, captions, navigation, and supplementary analyses are inserted around
the physician blocks and are explicitly marked with data-origin attributes.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import importlib.util
import json
import re
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
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
REVIEW_BASELINE_PATH = ROOT / "manuscript/review_baselines/v1.3_visible_blocks.json"
REVIEW_BASELINE_VERSION = "v1.3"
EXPECTED_REVIEW_BASELINE_COMMIT = "de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8"
EXPECTED_REVIEW_BASELINE_HTML_SHA256 = "db7b8fb330cce451383bc9741e616aa17f6bc87d5b3da3afdf39a309d6ecbb47"

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
EXPECTED_FIGURE_SHA256 = {
    "figure1_development_pathway_rough.svg": "597ef83181cbb4c080c236f8c2568ad11320cf2bc7627766f43a95f146349b15",
    "figure2_evaluator_distribution_rough.svg": "0a4e013e7d7d8c4f9f358de3104b2bb8b3dfc43a7943f0c16ff2e4c60d756682",
    "figure3_interrater_matrix_rough.svg": "55d0e996a3d4622e44a6a4581c56983b8fbfe4828f4f081e4f079a7779806a54",
    "figure4_effect_profile.svg": "1245d80c1f06333233fded624f3c6fa573b0c1bf574faee903a82bdda84cfa03",
    "supplementary_figure_s1_note_margins.svg": "ac9998447b8fb77c9bf6a53cbfbc459542fe07a6f10d53e3041d40648dad84d0",
    "supplementary_figure_s2_staged_ablation.svg": "13333a831115c39a2571cf16d3f3dce0234d1b6dbe08bbb96fe06d915f7850df",
    "figure1_development_pathway_print.png": "8ca61a861af3a78f870a58defa94bb2d54d8d09bd5c7c84802d780daed473b2b",
    "figure2_evaluator_distribution_print.png": "7bdc48f3eff05f26d1f272a5afbb454fdad9c5d8b1679407945592af0027e372",
    "figure3_interrater_matrix_print.png": "f09f4a215ce9699358b785603f1ca33a87333b6b28178acd7acda07f84d12fcb",
    "figure4_effect_profile_print.png": "afabeb331e0189f21b685112201ce491030adb0e7023205d1a460171c1321183",
    "supplementary_figure_s1_note_margins_print.png": "85b0096242ef36193477048fc46cd85637807b29cdb65a34d18e49f63133cd5b",
    "supplementary_figure_s2_staged_ablation_print.png": "5e0545aa353427368c51a86083bda7766e108de010617e47ec9603c2d3630691",
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
    "Breast cancer development with physician review",
    "PDAC adaptation with feedback from language models",
    "Clinical evaluation data collection and scoring rubric",
    "Staged Technical ablation",
    "Statistical analysis",
    "Performance across Clinical Extraction Fields",
}
HIDDEN_SOURCE_BLOCKS = {"P032", "P033", "P034", "P035", "P036", "P050", "P071", "P089", "P091", "P094", "P099", "P105"}
OUTSTANDING_COMMENT_IDS = {"13"}
RESOLVED_COMMENT_LITERATURE_MAP = {
    "0": 21, "1": 3, "2": 22, "3": 10, "4": 23,
    "8": 2, "9": 27, "10": 28, "11": 29, "12": 30,
}
COMMENT_14_RESOLUTION_BASIS = "Breast ER/PR/HER2 rules verified in prompt, breast-only routing, hooks, and tests."
COMMENT_RESOLUTION_REGISTRY: dict[str, dict[str, object]] = {
    comment_id: {"status": "resolved", "literature_entry": literature_entry}
    for comment_id, literature_entry in RESOLVED_COMMENT_LITERATURE_MAP.items()
}
COMMENT_RESOLUTION_REGISTRY["14"] = {
    "status": "resolved",
    "basis": COMMENT_14_RESOLUTION_BASIS,
}

REVISED_HEADINGS = {
    "P055": "Breast cancer development with physician review",
    "P059": "PDAC adaptation with feedback from language models",
    "P066": "Ablation study",
}

REVISED_PARAGRAPHS = {
    "P004": "Large language models (LLMs) are increasingly used for clinical documentation, patient communication, and administrative tasks. Many clinical applications depend on identifying relevant facts in electronic health record (EHR) notes before those facts can be summarized, analyzed, or used in another task. Reliable extraction is therefore an important step for many uses of LLMs in oncology care and research. This study focuses on that extraction task.",
    "P006": "Although EHRs contain structured fields, it is estimated that 80% of clinical data exist in unstructured form ([[cite:kong]]), with a large proportion of clinically relevant information encoded in free text narratives ([[cite:wiest]]). Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and changes in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real world practice, laboratory, imaging, and genomic results copied from earlier notes lengthen notes and further complicate the extraction task. A study of outpatient documentation across a decade found that notes became longer and more redundant, with much of the increase attributable to templates or copied and pasted text ([[cite:rule]]). LLMs must therefore identify the relevant information and determine its relationship to the patient's current disease status, treatment, and timeline.",
    "P008": "LLMs can extract information from clinical notes, but their outputs can vary. They may omit details, confuse past findings with current findings, or add information that is not supported by the original note ([[cite:sushil]]; [[cite:huang]]). A reviewer may be able to identify these errors, but checking every field in every output can approach the cost of manual extraction. Commercial models perform better on some clinical extraction tasks, but sending protected health information (PHI) to externally hosted services raises privacy and governance concerns, including HIPAA compliance ([[cite:huang]]; [[cite:wiest]]). Local deployment avoids external transfer of PHI, but local models can still produce unstable or incorrect extractions ([[cite:wiest]]). A useful clinical workflow must therefore improve reliability without depending on external processing or case by case manual review.",
    "P010": "How an LLM is used may be as important as the model itself. Additional training can adapt model weights to a clinical task. When the weights are unchanged, prompt design, task decomposition, verification, and clinical rules can organize the extraction process. Huang et al. iteratively refined prompts on expert curated lung pathology data and later applied the same prompt engineering strategy to osteosarcoma reports. Hein et al. used a human in the loop process to refine prompts, output schemas, and an error ontology, then applied the resulting templates to breast and prostate pathology reports ([[cite:huang]]; [[cite:hein]]). These studies focused on pathology reports and prompt or schema refinement. It remains unclear whether a broader multistep workflow developed from physician feedback on breast cancer notes can transfer to longitudinal PDAC notes with multiple fields when no physician reviews the PDAC development outputs.",
    "P012": "To address this gap, we built a locally deployable inference harness around a frozen LLM. The harness divides extraction into smaller tasks and applies checks and clinical rules without changing the model weights. We developed the breast cancer harness with physician input. We then adapted it to PDAC using LLM review and regression tests without physician review of the PDAC development outputs. Four physicians compared the final harness with a one prompt baseline that used the same model. We also conducted a technical ablation to examine changes across successive harness configurations.",
    "P017": "The study comprised two development stages followed by an independent physician evaluation. In Stage 1, a physician and an investigator developed the breast cancer inference harness through review of selected outputs. In Stage 2, the investigator adapted the harness to pancreatic ductal adenocarcinoma (PDAC) using LLM review and regression tests, without physician review of the PDAC development outputs. Four physicians then compared the final systems on an untouched holdout set.",
    "P019": "The final harness was compared with a one prompt baseline using the same frozen model and field schema on an untouched holdout set of 40 CORAL notes annotated by experts. The set included 20 breast cancer notes and 20 PDAC notes. Four physicians completed evaluations for each field across both cancer types. System identities were not disclosed.",
    "P022": "We used CORAL, a PhysioNet dataset of deidentified medical oncology notes obtained from the University of California, San Francisco. The development process used 100 unannotated breast cancer notes and 100 unannotated pancreatic cancer notes. A separate untouched holdout set of 40 notes annotated by experts consisted of 20 breast cancer notes and 20 PDAC notes. The holdout note texts were not used for development. The annotations were not provided to the model or used during development, and they were not used to calculate the primary endpoint based on physician preference. They were not shown to the physicians in the evaluation interface.",
    "P025": "We used Qwen2.5-32B-Instruct-AWQ served locally through vLLM. The baseline used the complete note and all extraction questions in one prompt and requested the full target schema in one response. Each note was processed in one chat completion request. This describes the application level request and does not make a claim about internal server execution or forward passes. In the physician review, the baseline and harness used the same field definitions and output contract.",
    "P028": "The harness extracted clinical information in multiple steps because different fields required different clinical context, reasoning, processing, and validation. It first used regular expressions to identify the Assessment and Plan section for fields related to the plan. If this step failed, the model made up to three attempts to locate and reproduce the relevant text, and each attempt was checked against the original note. If all attempts failed, the harness skipped only the plan branch and continued extraction from the complete note. Separate prompts for each field identified the visit context, cancer diagnosis, laboratory results, objective findings, active medications, and recent treatment changes. Information extracted in earlier steps then guided dependent assessments. For example, cancer stage and metastatic status provided context for treatment intent, while current therapy and clinical findings provided context for treatment response. The plan branch used text from the Assessment and Plan section that was supported by the source note. Prompts instructed the model to leave a field empty when the note did not provide sufficient evidence.",
    "P030": "Outputs passed through structural checks for JSON and schema validity, followed by checks for specificity, semantic alignment, accuracy and support from the original note, and temporal consistency. Deterministic clinical rules and processing across fields were then applied to repeated extraction errors. Source attribution was added when enabled. See Table 1. Operational definitions and order are provided in Supplementary Methods.",
    "P038": "We included clinical rules to address repeated extraction errors. For example, these rules helped the system distinguish regional lymph node involvement from distant metastases, suspected metastatic disease from confirmed metastatic disease, and active anticancer treatment from supportive or home medications. They also prevented disease changes documented before treatment initiation from being misclassified as a treatment response. The system returned supporting excerpts from the original clinical note when available.",
    "P057": "Breast cancer development used all 100 development notes. Early and targeted iterations used smaller subsets to shorten the feedback cycle, while major and final runs covered all 100 notes to detect less frequent errors and check for regression. A physician reviewed selected outputs, and the investigators compared flagged outputs with the complete source note, grouped repeated errors, and translated them into general changes to prompts, verification logic, or deterministic rules. These changes were retained only after testing on affected examples. The objective was to encode repeated clinical distinctions rather than corrections for individual cases. For example, an error in which axillary nodal disease was classified as distant metastasis motivated a general rule for regional nodes. An error in which a planned drug was listed as active therapy motivated a temporal medication rule. This process did not produce a conventional supervised training set, and the model weights were not updated.",
    "P060": "The harness developed from breast cancer notes was further adapted to pancreatic ductal adenocarcinoma using 100 additional pancreatic cancer notes. No physician reviewed pancreatic cancer outputs during this stage. ChatGPT organized the clinical error categories and clinical distinctions established during breast cancer development into a structured review rubric. The Qwen pipeline generated structured outputs, and a Qwen reviewer guided by the rubric compared them with the complete pancreatic source notes and flagged potential errors. Investigators assessed the flags, implemented selected revisions to prompts, clinical rules, or workflow logic, and retained changes only after regression testing.",
    "P061": "The adaptation retained the core framework while introducing modifications specific to pancreatic cancer, particularly for disease terminology, treatment regimens, and rules applied after extraction. The framework preserved its core approach to error detection and correction while accommodating differences between the two cancer types.",
    "P063": "The clinical evaluation presented the source note and two structured outputs through an A/B test interface. Four physicians selected A better, B better, or tie for each field. The physician who participated in Stage 1 development was not one of the four physicians. System identities were not disclosed. A always displayed the harness output, and B always displayed the baseline output. Source attribution was available only with the harness output when present, so physicians compared the full output packages rather than source attribution alone. Before evaluating the holdout set, the authors designated seven core fields as clinically important for understanding disease status and current management. Details of the interface are provided in Supplementary Methods. See Supplementary Table S1.",
    "P067": "We conducted an ablation study on the same 20 breast cancer and 20 pancreatic cancer holdout notes to examine how output quality changed across successive harness configurations. All four conditions used Qwen2.5-32B-Instruct-AWQ, greedy decoding, and the same output contract with 31 fields. Attribution, tool calling, and letter generation were disabled.",
    "P068": "See Table 2 for the 4 conditions.",
    "P073": "We compared adjacent configurations (A vs B, B vs C, and C vs D), yielding 120 paired comparisons across 40 notes. A single OpenAI LLM judge received only the source note and the two outputs for each adjacent configuration pair. The judge had no access to the project code, development history, prior evaluations, or configuration identities. It was instructed to prioritize accuracy and support from the original note, then completeness, and to use TIE when differences were minor or offsetting. We compared adjacent configurations using two sided exact sign tests, excluding ties. Details of the judge model and settings are provided in Supplementary Methods.",
    "P076": "For the physician evaluation, interrater agreement was assessed pairwise using exact agreement and Cohen's κ for each pair of physicians. Results from the eight completed combinations of physician and cancer type were summarized separately. Pooled counts were reported descriptively and were not treated as independent observations for each field.",
    "P078": "For the primary inferential analysis, tied comparisons were excluded, and each remaining directional judgment was coded according to whether it favored the harness or the baseline. We fit a population averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level to account for repeated judgments for each field within the same note. Physician was included as a fixed effect, and cancer type was additionally included in the overall model. Models for each cancer type adjusted for physician.",
    "P080": "The primary outcome was the adjusted probability of harness preference among directional judgments. Predicted probabilities were standardized over the observed covariate distribution, and 95% confidence intervals (CIs) were calculated using the delta method on the logit scale. Adjusted probabilities and adjusted odds ratios (ORs) were reported.",
    "P082": "For sensitivity analyses, we refitted the overall model four times after excluding one physician at a time. We also retained tied judgments by coding harness, tie, and baseline preferences as 1, 0, and −1, respectively, and calculated the mean net preference margin with 95% CIs using sandwich standard errors clustered by note. Finally, we aggregated preferences for each field within each note and performed an exact two sided sign test comparing positive and negative margins by note.",
}


@dataclass(frozen=True)
class SectionOverlay:
    category: str
    old: str
    new: str
    count: int = 1


SECTION_OVERLAY_CATEGORIES = {
    "terminology", "hyphen-only", "number-format", "cross-reference", "citation",
    "user-approved-limitation",
}
RESULTS_DISCUSSION_OVERLAYS: dict[str, tuple[SectionOverlay, ...]] = {
    "P085": (
        SectionOverlay("number-format", "completed 2160 comparisons", "completed 2,160 comparisons"),
        SectionOverlay("terminology", "required note-field judgments", "required judgments for each note and field"),
        SectionOverlay("cross-reference", ", see Table1.", ". See Table 3, Table 4, and Figure 2."),
        SectionOverlay("hyphen-only", "note-cluster robust 95% CI", "note cluster robust 95% CI"),
    ),
    "P086": (
        SectionOverlay("terminology", "breast-cancer", "breast cancer"),
        SectionOverlay("terminology", "first physician's per-note result", "first physician's result by note"),
        SectionOverlay("terminology", "harness-minus-baseline margin", "difference between harness and baseline preferences"),
        SectionOverlay("cross-reference", "(Figure 2).", "(Supplementary Figure S1)."),
    ),
    "P092": (
        SectionOverlay("terminology", "breast-cancer", "breast cancer"),
        SectionOverlay("terminology", "pancreatic-cancer", "pancreatic cancer"),
        SectionOverlay("number-format", "83.62)", "83.62%)"),
        SectionOverlay("number-format", "p<0.05", "p < 0.05", 2),
        SectionOverlay("number-format", "p <0.05", "p < 0.05"),
        SectionOverlay("cross-reference", ", see Table 3.", ". See Table 5."),
    ),
    "P097": (
        SectionOverlay("cross-reference", "(Figure 3).", "(Figure 4)."),
    ),
    "P100": (
        SectionOverlay("terminology", "evaluator-by-cancer", "physician and cancer type"),
        SectionOverlay("cross-reference", "(Figure 4).", "(Supplementary Figure S1)."),
    ),
    "P109": (
        SectionOverlay("terminology", "single-prompt", "one prompt"),
    ),
    "P111": (
        SectionOverlay("terminology", "free-text", "free text", 2),
        SectionOverlay("citation", "unsupported findings.", "unsupported findings ([[cite:sushil]])."),
        SectionOverlay("citation", "Huang et al demonstrated", "[[cite:huang_named]] demonstrated"),
        SectionOverlay("citation", "Hein et al. showed", "[[cite:hein_named]] showed"),
    ),
    "P113": (
        SectionOverlay("terminology", "develop field-specific prompts", "develop prompts for each field"),
        SectionOverlay("terminology", "clinician knowledge", "physician knowledge"),
        SectionOverlay("terminology", "LLM-as-judge", "LLM as a judge"),
        SectionOverlay("terminology", "text-comparison", "text comparison"),
        SectionOverlay(
            "citation",
            "rubrics such as faithfulness and completeness.",
            "rubrics such as faithfulness and completeness ([[cite:gershon]]; [[cite:williams]]).",
        ),
        SectionOverlay("terminology", "independent physician evaluators", "independent physicians"),
    ),
    "P115": (
        SectionOverlay(
            "citation",
            "as reported by prior studies.",
            "as reported by prior studies ([[cite:kruse]]; [[cite:kougia]]).",
        ),
        SectionOverlay("terminology", "physician evaluators", "physicians"),
    ),
    "P117": (
        SectionOverlay("cross-reference", "(Supplementary Figure X).", "(Supplementary Figure S2)."),
        SectionOverlay("terminology", "trade-offs", "tradeoffs"),
    ),
    "P119": (
        SectionOverlay("terminology", "single-source dataset", "dataset from a single source"),
        SectionOverlay("terminology", "model-generated narratives", "narratives generated by models"),
        SectionOverlay("terminology", "repeated, copied-forward information", "repeated information copied from earlier notes"),
        SectionOverlay("terminology", "non-physician LLM judge", "LLM judge that was not a physician"),
        SectionOverlay(
            "user-approved-limitation",
            "language models. Second,",
            "language models. The fixed A/B display order and the availability of source attribution only in the harness output may have influenced physician ratings. As a result, the direction and magnitude of any effect from this asymmetry could not be isolated. Second,",
        ),
        SectionOverlay(
            "user-approved-limitation",
            "clinical rule. Future studies",
            "clinical rule. Both systems used the same base model family, so generalization across model families was not assessed. Future studies",
        ),
    ),
}

POSITIONED_COMMENT_ANCHORS = {
    ("P115", "13"): "We addressed this issue by incorporating treatment history and temporal context into the extraction process and adding verification steps to align treatment responses with the appropriate regimen.",
    ("P117", "14"): "For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC.",
}

F7_SECTION_BASELINE = {
    "introduction": {
        "markup_sha256": "482dc09b37bc4b893b8afee63958816c54f1d25a976a4db23636278ac8b68074",
        "text_sha256": "765a1c0e59c3c6b883d9b89df95fbb737758eddef30a470fc5b421b979119e78",
    },
    "methods": {
        "markup_sha256": "65c54db21c365bd475a834e33f51e8ace65332a81ed72e9912d8a7fa5be7b0af",
        "text_sha256": "628d07037caf487000bb27b068fccc0cbd398d1d9cae74422daba48635aa31a0",
    },
}
EXPECTED_V164_RESULTS_SHA256 = "5aceff6ac9b1f9a78abf8bc322c19556a32cc0ba722359b7656e987b226965b3"
EXPECTED_V164_RESULTS_NUMERIC_SHA256 = "2966552230c8eb68c13797418d36abf7abdd1e8269f753f5f49bf91a74aec5b4"
EXPECTED_F7_V164_SEMANTIC_RESULTS_NUMERIC_SHA256 = "c63c2e9083b71313e519062257189f27373673692ddce5d1bab0e4a3e743ddd5"
EXPECTED_V166_VISIBLE_BLOCK_TEXT_EXCLUDING_COMMENT_UI_SHA256 = "2f4621fc82992f179bf1b437ac6ede273f084d5c797993b4aa6d1669b6146f7f"
EXPECTED_V167_P119_TEXT_SHA256 = "0b6b9d289adf6a6dd13d9dffd8a67d0686e684fd207d88cb5d0342436dab28fb"


def apply_results_discussion_overlay(block_id: str, canonical_text: str) -> str:
    revised = canonical_text
    for overlay in RESULTS_DISCUSSION_OVERLAYS.get(block_id, ()):
        if overlay.category not in SECTION_OVERLAY_CATEGORIES:
            raise AssertionError(f"Unknown section overlay category for {block_id}: {overlay.category}")
        if revised.count(overlay.old) != overlay.count:
            raise AssertionError(
                f"Section overlay source mismatch for {block_id}: {overlay.old!r} "
                f"expected {overlay.count}, found {revised.count(overlay.old)}"
            )
        revised = revised.replace(overlay.old, overlay.new, overlay.count)
    return revised

GRAMMAR_REVISED_IDS = {
    "P004", "P006", "P008", "P010", "P012", "P017", "P019", "P022", "P025", "P028", "P030", "P038",
    "P057", "P060", "P061", "P063", "P067", "P073", "P076", "P078", "P082",
}

# The visible page uses a small set of manually selected review focuses. The
# complete v1.3 comparison remains available in block metadata and hidden
# source audit text. Simple grammar, punctuation, and linking edits are not
# colored.
MANUAL_FOCUS_PHRASES = {
    "P004": ["focuses on that extraction task"],
    "P008": [
        "outputs can vary",
        "checking every field in every output can approach the cost of manual extraction",
        "privacy and governance concerns",
        "local models can still produce unstable or incorrect extractions",
    ],
    "P010": [
        "Additional training can adapt model weights",
        "prompt design, task decomposition, verification, and clinical rules",
        "no physician reviews the PDAC development outputs",
    ],
    "P012": [
        "To address this gap",
        "locally deployable inference harness",
        "without changing the model weights",
        "without physician review of the PDAC development outputs",
        "technical ablation",
    ],
    "P017": ["without physician review"],
    "P022": [
        "100 unannotated breast cancer notes and 100 unannotated pancreatic cancer notes",
        "separate untouched holdout set",
        "not provided to the model or used during development",
        "not used to calculate the primary endpoint based on physician preference",
    ],
    "P028": [
        "Separate prompts for each field",
        "skipped only the plan branch and continued extraction from the complete note",
        "leave a field empty when the note did not provide sufficient evidence",
    ],
    "P060": ["No physician reviewed pancreatic cancer outputs"],
    "P061": ["modifications specific to pancreatic cancer"],
    "P063": ["System identities"],
    "P073": ["judge model and settings"],
    "P119": [
        "fixed A/B display order",
        "source attribution only in the harness output",
        "direction and magnitude",
        "same base model family",
        "generalization across model families",
    ],
}

SUBSTANTIVE_REVISED_IDS = {
    "P004", "P008", "P010", "P012", "P017", "P019", "P022", "P025", "P028", "P030", "P057", "P060",
    "P063", "P067", "P073",
}

INTRODUCTION_IDS = ("P004", "P006", "P008", "P010", "P012")
CONTENT_WORD_STOPLIST = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "must",
    "no", "not", "of", "on", "one", "only", "or", "other", "our", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "you", "your",
}

REVISED_TABLE_CELLS = {
    "T1R2C1": "Extraction with separate prompts for each field",
    "T1R3C1": "Context that accounts for dependencies",
    "T1R4C2": "The model returns medically related answers that do not address the target question",
    "T1R5C2": "The model infers a conclusion that is not supported by the note ",
    "T1R5C3": "Keep a lesion classified as suspected rather than label it as confirmed when a biopsy is pending",
    "T1R6C2": "Confusion among past, current, and future information.",
    "T1R6C3": "Remove a completed scan from the imaging plan",
    "T1R8C1": "Clinical rules across fields",
    "T2R1C2": "A one prompt baseline with one inference request per note.",
    "T2R2C2": "Decomposed extraction with separate prompts for each field, routing across fields, bounded Assessment and Plan retry, and the shared format and schema safeguards.",
    "T2R4C2": "Condition C plus deterministic inline processing and the complete POST hook layer used by the full inference harness.",
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
.focus-highlight{color:inherit;background:rgba(255,193,7,.28);border-radius:2px;-webkit-box-decoration-break:clone;box-decoration-break:clone;cursor:help}.review-block{background:transparent}.citation.external-citation{background:transparent}.revision-paragraph{margin-top:9px}.supplement-methods h4{margin:18px 0 5px;color:#30485a;font-size:15px}.supplement-methods ol{margin:7px 0 14px;padding-left:24px}.supplement-methods li{margin:5px 0}
details{border:0}summary{cursor:pointer}.fold>summary,.fold-toggle{padding:13px 16px;color:var(--navy);font-weight:750;list-style:none}.fold>summary::-webkit-details-marker,.version-entry>summary::-webkit-details-marker{display:none}.fold>summary::before,.fold-toggle::before,.version-entry>summary::before{content:"▸";display:inline-block;width:1.2em}.fold[open]>summary::before,.custom-fold.is-expanded>.fold-toggle::before,.version-entry[open]>summary::before{content:"▾"}.current-change{margin:14px 0 20px;padding:10px 13px;border-left:4px solid var(--blue);background:#eef7fb;color:#264b61;font-size:13px}.version-entry{margin:8px 0;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.version-entry>summary{padding:9px 11px;color:#30485a;font-weight:700;list-style:none}.version-entry-body{padding:0 13px 10px;color:#4b5964;font-size:13px}.version-entry-body p{margin:5px 0}
.fold{margin:22px 0;border-top:1px solid var(--line)}.fold>summary,.fold-toggle{margin:0;padding:20px 0 10px;font-family:Georgia,"Times New Roman",serif;font-size:25px}.fold-toggle{display:block;width:100%;border:0;background:transparent;text-align:left;cursor:pointer}.custom-fold.is-collapsed>.fold-body{display:none}.custom-fold.is-expanded>.fold-body{display:block}.fold-body{padding:0 0 8px}.source-empty,.source-audit{display:none!important}.physician-paragraph,.manuscript-paragraph{margin:11px 0}.physician-subheading{scroll-margin-top:15px}.xref,.citation,.comment-anchor a{color:var(--blue);text-decoration:none}.xref:hover,.citation:hover,.comment-anchor a:hover{text-decoration:underline}.comment-anchor{margin-left:.18em;font-size:.72em;vertical-align:super}.comment-anchor a{font-weight:650}.comment-backlink{display:inline-block;margin-top:7px;font-size:12px}.print-page-spacer{display:none}
.table-wrap{margin:20px 0 28px;overflow-x:auto}.table-caption,.figure-caption{margin:0 0 8px;color:#314452;font-size:13px}.table-caption strong,.figure-caption strong{color:var(--navy)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 9px;border:1px solid #d6dde3;vertical-align:top}th{background:#e9f1f6;color:#173f5f;text-align:left}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#fafbfc}.physician-native-table th{background:#eaf4ef;color:#18533f}
.figure{margin:24px 0 30px}.figure img{display:block;width:100%;height:auto;border:1px solid #e0e5e9;background:white}.figure-print{display:none!important}.figure-caption{margin-top:9px}.comment{margin:13px 0;padding:12px 14px;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.comment-meta{font-size:12px;color:var(--muted)}.comment-text{margin:7px 0;white-space:pre-wrap}.comment-context{margin-top:7px;color:#4e5c67;font-size:12px}.literature-workspace li{margin:11px 0}.reference-note{display:block;margin-top:3px;color:#4e5c67;font-size:12px}.workspace-note{color:#4e5c67;font-size:13px}.version-list li{margin:6px 0}.source-hash{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
@media(max-width:900px){.layout{display:block;width:min(100% - 18px,940px);margin:10px auto 40px}.toc{position:relative;top:auto;max-height:none;margin-bottom:12px}.toc a.sub{display:none}article{padding:30px 24px 48px}h1{font-size:30px}.metadata-grid{grid-template-columns:1fr}.table-wrap{margin-left:-8px;margin-right:-8px}.figure{margin-left:-8px;margin-right:-8px}}
@media(max-width:440px){article{padding:24px 15px 40px}h1{font-size:27px}.fold>summary,.fold-toggle{font-size:22px}table{font-size:11px}th,td{padding:6px 5px}.figure-caption,.table-caption{padding:0 3px}}
@media print{@page{margin:.62in}html{scroll-behavior:auto}body{background:#fff;font-family:Arial,sans-serif;font-size:10pt;line-height:1.42}.layout{display:block;width:auto;margin:0}.toc,.screen-only,#physician-comments,#version-history,#literature-workspace,.source-audit,.source-empty,.figure-screen{display:none!important}article{padding:0;border:0;box-shadow:none}.main-section,.supplement{display:block!important;border:0;margin:0}.main-section:not([open])>.fold-body,.supplement>.fold-body{display:block!important}.main-section>summary,.supplement>.fold-toggle{display:block!important;padding:16px 0 7px;list-style:none;break-after:avoid;font-size:19pt}.main-section>summary::before,.supplement>.fold-toggle::before{display:none!important}.physician-subheading{break-after:avoid}.figure,.table-wrap{break-inside:avoid;page-break-inside:avoid}.figure{margin:14px 0 18px}.figure .figure-print{display:block!important;width:100%;max-height:8.4in;object-fit:contain;border:0}.table-wrap{margin:12px 0 18px;overflow:visible}table{font-size:8.5pt}th,td{padding:5px 6px}.table-caption,.figure-caption{font-size:9pt}.comment-anchor{display:none!important}.focus-highlight{background:transparent!important;outline:0!important;color:inherit}}
html.print-export{scroll-behavior:auto}html.print-export body{background:#fff;font-family:Arial,sans-serif;font-size:13.333px;line-height:1.42;margin:0}html.print-export .layout{display:block;width:720px;margin:0}html.print-export .toc,html.print-export .screen-only,html.print-export #physician-comments,html.print-export #version-history,html.print-export #literature-workspace,html.print-export .source-audit,html.print-export .source-empty,html.print-export .figure-screen{display:none!important}html.print-export article{width:720px;padding:45px;box-sizing:border-box;border:0;box-shadow:none}html.print-export .main-section,html.print-export .supplement{display:block!important;border:0;margin:0}html.print-export .main-section>.fold-body,html.print-export .supplement>.fold-body{display:block!important}html.print-export .main-section>summary,html.print-export .supplement>.fold-toggle{display:block!important;padding:16px 0 7px;list-style:none;font-size:25px}html.print-export .main-section>summary::before,html.print-export .supplement>.fold-toggle::before{display:none!important}html.print-export .figure,html.print-export .table-wrap{break-inside:avoid}html.print-export .figure{margin:14px 0 18px}html.print-export .figure .figure-print{display:block!important;width:100%;max-height:806px;object-fit:contain;border:0}html.print-export .table-wrap{margin:12px 0 18px;overflow:visible}html.print-export table{font-size:11.333px}html.print-export th,html.print-export td{padding:5px 6px}html.print-export .table-caption,html.print-export .figure-caption{font-size:12px}html.print-export .comment-anchor{display:none!important}html.print-export .focus-highlight{background:transparent!important;outline:0!important;color:inherit}html.print-export .print-page-spacer{display:block!important;width:1px;visibility:hidden}
"""

SCRIPT = r"""
function setCustomFold(fold,open){if(!fold||!fold.classList.contains('custom-fold'))return;fold.classList.toggle('is-expanded',open);fold.classList.toggle('is-collapsed',!open);const button=fold.querySelector(':scope>.fold-toggle');if(button)button.setAttribute('aria-expanded',open?'true':'false')}
function revealTarget(target){if(!target)return;if(target.matches&&target.matches('#version-history .version-entry'))document.querySelectorAll('#version-history .version-entry').forEach(entry=>entry.open=entry===target);else if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true;if(target.classList&&target.classList.contains('custom-fold'))setCustomFold(target,true);let p=target.parentElement;const chain=[];while(p){if(p.tagName==='DETAILS'||p.classList.contains('custom-fold'))chain.push(p);p=p.parentElement;}chain.reverse().forEach(d=>{if(d.tagName==='DETAILS')d.open=true;else setCustomFold(d,true)});if(!target.hasAttribute('tabindex'))target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start'})}
document.addEventListener('click',e=>{const toggle=e.target.closest('.fold-toggle');if(toggle){const fold=toggle.closest('.custom-fold');setCustomFold(fold,!fold.classList.contains('is-expanded'));return}const a=e.target.closest('a[href^="#"]');if(!a)return;const id=decodeURIComponent(a.getAttribute('href').slice(1));revealTarget(document.getElementById(id))});
window.addEventListener('hashchange',()=>revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1)))));window.addEventListener('DOMContentLoaded',()=>{if(location.hash)revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1))))});
window.preparePrintExport=function(pageHeight=960){if(!Number.isFinite(pageHeight)||pageHeight<400)throw new Error('Invalid page height');document.querySelectorAll('.print-page-spacer').forEach(node=>node.remove());document.documentElement.classList.add('print-export');void document.documentElement.offsetHeight;const candidates=[...document.querySelectorAll('.figure,.table-wrap')].filter(node=>getComputedStyle(node).display!=='none');for(const node of candidates){const rect=node.getBoundingClientRect();const top=rect.top+window.scrollY;const offset=((top%pageHeight)+pageHeight)%pageHeight;if(rect.height<pageHeight&&offset>0&&offset+rect.height>pageHeight-.5){const spacer=document.createElement('div');spacer.className='print-page-spacer';spacer.setAttribute('aria-hidden','true');spacer.style.height=`${Math.ceil(pageHeight-offset)+1}px`;node.before(spacer);void spacer.offsetHeight}}const items=candidates.map(node=>{const rect=node.getBoundingClientRect();const top=rect.top+window.scrollY;const bottom=rect.bottom+window.scrollY;return{id:node.id,kind:node.classList.contains('figure')?'figure':'table',top,bottom,height:rect.height,crosses:Math.floor((top+.5)/pageHeight)!==Math.floor((bottom-.5)/pageHeight)}});const printable=[...document.querySelectorAll('article>.main-section>.fold-body>*,#supplement-body>*')].filter(node=>!node.classList.contains('print-page-spacer')&&getComputedStyle(node).display!=='none');const contentBottom=Math.ceil(Math.max(0,...printable.map(node=>node.getBoundingClientRect().bottom+window.scrollY)));const focusHighlights=[...document.querySelectorAll('.focus-highlight')];return{pageHeight,documentHeight:document.documentElement.scrollHeight,contentBottom,supplementVisible:getComputedStyle(document.getElementById('supplement-body')).display!=='none',revisionHighlightCount:focusHighlights.length,revisionHighlightBackgrounds:focusHighlights.map(node=>getComputedStyle(node).backgroundColor),items,offenders:items.filter(item=>item.crosses).map(item=>item.id)}};
window.resetPrintExport=function(){document.querySelectorAll('.print-page-spacer').forEach(node=>node.remove());document.documentElement.classList.remove('print-export')};
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def markdown_source_text(value: str) -> str:
    """Keep source trailing spaces visible without Git whitespace errors."""

    trailing = len(value) - len(value.rstrip(" "))
    return value.rstrip(" ") + "&#32;" * trailing


def introduction_content_word_reuse(data: dict[str, object]) -> tuple[int, int, float]:
    source_by_id = {str(item["id"]): str(item["raw_visible_text"]) for item in data["paragraphs"]}
    citation_token = re.compile(r"\s*\[\[cite:[a-z_]+\]\]")

    def tokens(text: str) -> list[str]:
        return [
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in CONTENT_WORD_STOPLIST and not token.isdigit()
        ]

    source_counts = Counter(tokens(" ".join(source_by_id[block_id] for block_id in INTRODUCTION_IDS)))
    revised_counts = Counter(tokens(" ".join(citation_token.sub("", REVISED_PARAGRAPHS[block_id]) for block_id in INTRODUCTION_IDS)))
    reused = sum(min(count, source_counts[token]) for token, count in revised_counts.items())
    total = sum(revised_counts.values())
    return reused, total, reused / total


@dataclass
class ReviewBlock:
    block_id: str
    text: str
    kind: str
    section: str
    start: int
    start_tag_end: int
    tag: str
    changed_from: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class _ReviewNode:
    tag: str
    attrs: dict[str, str | None]
    parent: "_ReviewNode | None"
    start: int
    start_tag_end: int
    excluded: bool
    parts: list[str] = field(default_factory=list)
    children: list["_ReviewNode"] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(str(self.attrs.get("class", "")).split())

    @property
    def text(self) -> str:
        return " ".join("".join(self.parts).split())


class ReviewBlockParser(HTMLParser):
    """Collect stable visible manuscript blocks and their source offsets."""

    BLOCK_TAGS = {"p", "h3", "h4", "li", "th", "td", "figcaption", "summary"}
    NON_MANUSCRIPT_IDS = {"physician-comments", "version-history", "literature-workspace"}

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_offsets: list[int] = []
        cursor = 0
        for line in source.splitlines(keepends=True):
            self.line_offsets.append(cursor)
            cursor += len(line)
        if not self.line_offsets:
            self.line_offsets.append(0)
        self.root = _ReviewNode("document", {}, None, 0, 0, False)
        self.stack: list[_ReviewNode] = [self.root]

    def _absolute_offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        raw = self.get_starttag_text() or ""
        start = self._absolute_offset()
        parent = self.stack[-1]
        classes = set(str(values.get("class", "")).split())
        excluded = (
            parent.excluded
            or "hidden" in values
            or "source-audit" in classes
            or "source-empty" in classes
            or "screen-only" in classes
            or values.get("id") in self.NON_MANUSCRIPT_IDS
        )
        node = _ReviewNode(tag, values, parent, start, start + len(raw), excluded)
        parent.children.append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.stack[-1].excluded:
            return
        for node in self.stack:
            if not node.excluded:
                node.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def nodes(self) -> list[_ReviewNode]:
        ordered: list[_ReviewNode] = []

        def visit(node: _ReviewNode) -> None:
            ordered.append(node)
            for child in node.children:
                visit(child)

        visit(self.root)
        return ordered


def _ancestor(node: _ReviewNode, predicate) -> _ReviewNode | None:
    current: _ReviewNode | None = node
    while current:
        if predicate(current):
            return current
        current = current.parent
    return None


def _descendants(node: _ReviewNode) -> list[_ReviewNode]:
    output: list[_ReviewNode] = []
    for child in node.children:
        output.append(child)
        output.extend(_descendants(child))
    return output


def _source_block_for(node: _ReviewNode) -> str | None:
    direct = node.attrs.get("data-source-block")
    if direct:
        return str(direct)
    for descendant in _descendants(node):
        source_block = descendant.attrs.get("data-source-block")
        if source_block:
            return str(source_block)
    if node.parent:
        index = node.parent.children.index(node)
        for sibling in reversed(node.parent.children[:index]):
            source_block = sibling.attrs.get("data-source-block")
            if source_block:
                return str(source_block)
            if sibling.tag not in {"span"}:
                break
    return None


def _review_section(node: _ReviewNode) -> str:
    if _ancestor(node, lambda item: item.attrs.get("id") == "supplementary-methods"):
        return "supplementary-methods"
    section = _ancestor(node, lambda item: str(item.attrs.get("id", "")).startswith("section-"))
    if section:
        return str(section.attrs["id"])[len("section-"):]
    if _ancestor(node, lambda item: item.attrs.get("id") == "supplementary-material"):
        return "supplementary-material"
    return "front"


def _review_kind(node: _ReviewNode) -> str:
    if node.tag in {"h3", "h4", "summary"}:
        return "heading"
    if node.tag in {"th", "td"}:
        return "cell"
    if node.tag == "figcaption" or "table-caption" in node.classes:
        return "caption"
    if node.tag == "li":
        return "list-item"
    return "paragraph"


def _table_cell_id(node: _ReviewNode) -> str | None:
    row = _ancestor(node, lambda item: item.tag == "tr")
    table = _ancestor(node, lambda item: item.tag == "table")
    wrapper = _ancestor(node, lambda item: "table-wrap" in item.classes and bool(item.attrs.get("id")))
    if not row or not table or not wrapper:
        return None
    rows = [item for item in _descendants(table) if item.tag == "tr"]
    cells = [item for item in row.children if item.tag in {"th", "td"}]
    return f'{wrapper.attrs["id"]}::r{rows.index(row)}c{cells.index(node)}'


def _stable_review_block_id(node: _ReviewNode, ordered_candidates: list[_ReviewNode]) -> str:
    explicit = node.attrs.get("data-review-block-id")
    if explicit:
        return str(explicit)
    source_block = _source_block_for(node)
    if source_block:
        return source_block
    html_id = node.attrs.get("id")
    if html_id:
        return str(html_id)
    if node.tag in {"th", "td"}:
        cell_id = _table_cell_id(node)
        if cell_id:
            return cell_id
    if node.tag == "figcaption":
        figure_node = _ancestor(node, lambda item: item.tag == "figure" and bool(item.attrs.get("id")))
        if figure_node:
            return f'{figure_node.attrs["id"]}::caption'
    if "table-caption" in node.classes:
        wrapper = _ancestor(node, lambda item: "table-wrap" in item.classes and bool(item.attrs.get("id")))
        if wrapper:
            return f'{wrapper.attrs["id"]}::caption'
    context = _ancestor(
        node,
        lambda item: item.attrs.get("id") == "supplementary-methods"
        or str(item.attrs.get("id", "")).startswith("section-"),
    )
    context_id = str(context.attrs["id"]) if context else "manuscript"
    peers = [
        item for item in ordered_candidates
        if item.tag == node.tag
        and _ancestor(
            item,
            lambda candidate: candidate.attrs.get("id") == "supplementary-methods"
            or str(candidate.attrs.get("id", "")).startswith("section-"),
        ) is context
    ]
    return f"{context_id}::{node.tag}:{peers.index(node)}"


def collect_review_blocks(document: str) -> dict[str, ReviewBlock]:
    parser = ReviewBlockParser(document)
    parser.feed(document)
    article = next((node for node in parser.nodes() if node.tag == "article"), None)
    candidates = [
        node for node in parser.nodes()
        if node.tag in parser.BLOCK_TAGS
        and not node.excluded
        and node.text
        and article is not None
        and _ancestor(node, lambda item: item is article) is not None
    ]
    blocks: dict[str, ReviewBlock] = {}
    for node in candidates:
        block_id = _stable_review_block_id(node, candidates)
        if block_id in blocks:
            raise AssertionError(f"Duplicate stable review block ID: {block_id}")
        blocks[block_id] = ReviewBlock(
            block_id=block_id,
            text=node.text,
            kind=_review_kind(node),
            section=_review_section(node),
            start=node.start,
            start_tag_end=node.start_tag_end,
            tag=node.tag,
            changed_from=str(node.attrs.get("data-review-changed-from")) if node.attrs.get("data-review-changed-from") else None,
        )
    return blocks


def load_review_baseline() -> dict[str, object]:
    baseline = json.loads(REVIEW_BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 1:
        raise AssertionError("Unexpected review-baseline schema")
    if baseline.get("baseline_version") != REVIEW_BASELINE_VERSION:
        raise AssertionError("Unexpected review-baseline version")
    if baseline.get("baseline_commit") != EXPECTED_REVIEW_BASELINE_COMMIT:
        raise AssertionError("Review-baseline commit changed")
    if baseline.get("baseline_html_sha256") != EXPECTED_REVIEW_BASELINE_HTML_SHA256:
        raise AssertionError("Review-baseline HTML hash changed")
    blocks = baseline.get("blocks")
    if not isinstance(blocks, dict) or baseline.get("block_count") != len(blocks):
        raise AssertionError("Review-baseline block count is inconsistent")
    for block_id, block in blocks.items():
        if not isinstance(block, dict) or hashlib.sha256(str(block.get("text", "")).encode("utf-8")).hexdigest() != block.get("sha256"):
            raise AssertionError(f"Review-baseline text hash changed: {block_id}")
    return baseline


def review_block_in_scope(block: ReviewBlock | dict[str, object]) -> bool:
    section = str(block.section if isinstance(block, ReviewBlock) else block.get("section", ""))
    kind = str(block.kind if isinstance(block, ReviewBlock) else block.get("kind", ""))
    block_id = str(block.block_id if isinstance(block, ReviewBlock) else block.get("block_id", ""))
    return (
        section in {"introduction", "methods", "supplementary-methods"}
        or block_id in {"P117", "P119", "revision-model-family-limitation"}
        or kind in {"heading", "cell", "caption"}
    )


def review_redline_state(document: str) -> dict[str, object]:
    baseline = load_review_baseline()
    baseline_blocks = baseline["blocks"]
    current = collect_review_blocks(document)
    scoped_current = {block_id: block for block_id, block in current.items() if review_block_in_scope(block)}
    changed = {
        block_id for block_id, block in scoped_current.items()
        if block_id not in baseline_blocks or block.text != baseline_blocks[block_id]["text"]
    }
    highlighted = {block_id for block_id, block in scoped_current.items() if block.changed_from == REVIEW_BASELINE_VERSION}
    deleted = {
        block_id for block_id, block in baseline_blocks.items()
        if review_block_in_scope({"block_id": block_id, **block}) and block_id not in current
    }
    section_counts = Counter(current[block_id].section for block_id in changed)
    return {
        "baseline": baseline,
        "current": current,
        "scoped_current": scoped_current,
        "changed": changed,
        "highlighted": highlighted,
        "deleted": deleted,
        "section_counts": dict(sorted(section_counts.items())),
    }


def _review_start_tag(raw: str, block_id: str, changed: bool) -> str:
    attributes = f' data-review-block-id="{esc(block_id)}"'
    if changed:
        attributes += f' data-review-changed-from="{REVIEW_BASELINE_VERSION}"'
        class_match = re.search(r'class="([^"]*)"', raw)
        if class_match:
            classes = class_match.group(1).split()
            for class_name in ["review-block"]:
                if class_name not in classes:
                    classes.append(class_name)
            raw = raw[:class_match.start(1)] + " ".join(classes) + raw[class_match.end(1):]
        else:
            attributes += ' class="review-block"'
    return raw[:-1] + attributes + ">"


def apply_review_block_metadata(document: str) -> str:
    """Attach stable change metadata without coloring whole blocks.

    The visible review marks are a separate, manually selected set of focus
    phrases. Block metadata records the complete v1.3 comparison but has no
    yellow style.
    """

    state = review_redline_state(document)
    scoped_current: dict[str, ReviewBlock] = state["scoped_current"]
    changed: set[str] = state["changed"]
    replacements: list[tuple[int, int, str]] = []
    for block_id, block in scoped_current.items():
        raw = document[block.start:block.start_tag_end]
        replacements.append((block.start, block.start_tag_end, _review_start_tag(raw, block_id, block_id in changed)))
    for start, end, replacement in sorted(replacements, reverse=True):
        document = document[:start] + replacement + document[end:]
    return document


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


def place_comment_anchors(rendered: str, block_id: str, comment_ids: list[str]) -> tuple[str, str]:
    trailing_ids: list[str] = []
    for comment_id in comment_ids:
        target_sentence = POSITIONED_COMMENT_ANCHORS.get((block_id, comment_id))
        if not target_sentence:
            trailing_ids.append(comment_id)
            continue
        rendered_target = link_crossrefs(target_sentence)
        if rendered.count(rendered_target) != 1:
            raise AssertionError(f"Positioned comment target is missing or ambiguous: {block_id}/{comment_id}")
        rendered = rendered.replace(rendered_target, rendered_target + comment_anchors([comment_id]), 1)
    return rendered, comment_anchors(trailing_ids)


def manuscript_paragraph(text: str, *, paragraph_id: str | None = None) -> str:
    id_attr = f' id="{esc(paragraph_id)}"' if paragraph_id else ""
    return f'<p class="manuscript-paragraph" data-origin="technical-insert"{id_attr}>{link_crossrefs(text)}</p>'


def revision_paragraph(text: str, *, paragraph_id: str) -> str:
    return (
        f'<p class="manuscript-paragraph revision-paragraph" data-origin="technical-insert" id="{esc(paragraph_id)}">'
        f'{link_crossrefs(text)}</p>'
    )


_CITATION_GROUP_RE = re.compile(
    r"\(\s*\[\[cite:[a-z_]+\]\](?:\s*;\s*\[\[cite:[a-z_]+\]\])*\s*\)|\[\[cite:[a-z_]+\]\]"
)
MAX_REVIEW_SPAN_CHARS = 180


def _review_tokens(text: str) -> list[dict[str, object]]:
    xrefs = "|".join(re.escape(label) for label in sorted(XREF_TARGETS, key=len, reverse=True))
    pattern = re.compile(
        rf"\[\[cite:[a-z_]+\]\]|{xrefs}|\s+|\w+(?:[-'’]\w+)*|[^\w\s]"
    )
    citation_ranges = [match.span() for match in _CITATION_GROUP_RE.finditer(text)]
    tokens: list[dict[str, object]] = []
    for match in pattern.finditer(text):
        start, end = match.span()
        raw = match.group(0)
        tokens.append({
            "raw": raw,
            "start": start,
            "end": end,
            "space": raw.isspace(),
            "citation": any(start >= left and end <= right for left, right in citation_ranges),
            "highlight": False,
            "title": "",
            "change": "",
        })
    if "".join(str(token["raw"]) for token in tokens) != text:
        raise AssertionError(f"Review tokenizer did not preserve text: {text[:80]!r}")
    return tokens


def _short_review_text(text: str, limit: int = 150) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _mark_focus_phrases(
    tokens: list[dict[str, object]],
    text: str,
    phrases: list[str],
) -> None:
    for phrase in phrases:
        cursor = 0
        found = False
        while True:
            start = text.find(phrase, cursor)
            if start < 0:
                break
            found = True
            end = start + len(phrase)
            for token in tokens:
                if bool(token["space"]) or bool(token["citation"]):
                    continue
                if int(token["start"]) < end and int(token["end"]) > start:
                    token["highlight"] = True
                    token["title"] = f'Focus for review: “{phrase}”'
                    token["change"] = "focus"
            cursor = end
        if not found:
            raise AssertionError(f"Review focus phrase is missing: {phrase!r}")


def _render_review_tokens(tokens: list[dict[str, object]], block_id: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not bool(token["highlight"]):
            parts.append(link_crossrefs(str(token["raw"])))
            index += 1
            continue
        title = str(token["title"])
        change = str(token["change"])
        run = [token]
        index += 1
        while index < len(tokens):
            candidate = tokens[index]
            if bool(candidate["highlight"]) and str(candidate["title"]) == title and str(candidate["change"]) == change:
                run.append(candidate)
                index += 1
                continue
            if bool(candidate["space"]) and index + 1 < len(tokens):
                following = tokens[index + 1]
                if bool(following["highlight"]) and str(following["title"]) == title and str(following["change"]) == change:
                    run.extend([candidate, following])
                    index += 2
                    continue
            break
        remaining = "".join(str(item["raw"]) for item in run)
        while remaining:
            if len(remaining.strip()) <= MAX_REVIEW_SPAN_CHARS:
                fragment, remaining = remaining, ""
            else:
                cut = remaining.rfind(" ", 0, MAX_REVIEW_SPAN_CHARS + 1)
                if cut <= 0:
                    cut = MAX_REVIEW_SPAN_CHARS
                fragment, remaining = remaining[:cut], remaining[cut:]
            leading = fragment[: len(fragment) - len(fragment.lstrip())]
            trailing = fragment[len(fragment.rstrip()):]
            core = fragment.strip()
            parts.append(link_crossrefs(leading))
            attrs = (
                f'class="focus-highlight" data-review-for="{esc(block_id)}" '
                f'data-review-change="{esc(change)}" title="{esc(title)}" '
                f'aria-label="{esc(title)}"'
            )
            parts.append(f'<span {attrs}>{link_crossrefs(core)}</span>')
            parts.append(link_crossrefs(trailing))
    return "".join(parts)


def word_diff_text(original: str, revised: str, *, block_id: str) -> tuple[str, list[str]]:
    """Render revised text with only manually selected focus phrases.

    Citations remain ordinary blue links. Deletions are retained as review
    metadata on the containing block instead of occupying visible body space.
    """

    old_tokens = _review_tokens(original)
    new_tokens = _review_tokens(revised)
    old_indices = [
        index for index, token in enumerate(old_tokens)
        if not bool(token["space"]) and not bool(token["citation"])
    ]
    new_indices = [
        index for index, token in enumerate(new_tokens)
        if not bool(token["space"]) and not bool(token["citation"])
    ]
    old_values = [str(old_tokens[index]["raw"]) for index in old_indices]
    new_values = [str(new_tokens[index]["raw"]) for index in new_indices]
    deleted: list[str] = []
    focus_phrases = MANUAL_FOCUS_PHRASES.get(block_id, [])
    matcher = SequenceMatcher(None, old_values, new_values, autojunk=False)
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        old_text = " ".join(old_values[old_start:old_end])
        new_text = " ".join(new_values[new_start:new_end])
        if operation in {"delete", "replace"} and old_text:
            deleted.append(old_text)
    _mark_focus_phrases(new_tokens, revised, focus_phrases)
    return _render_review_tokens(new_tokens, block_id), deleted


def highlighted_text(text: str, phrases: list[str], *, block_id: str) -> str:
    """Highlight only selected key phrases in wholly generated text."""

    tokens = _review_tokens(text)
    _mark_focus_phrases(tokens, text, phrases)
    return _render_review_tokens(tokens, block_id)


def revision_inline_paragraph(text: str, *, paragraph_id: str, phrases: list[str]) -> str:
    return (
        f'<p class="manuscript-paragraph revision-paragraph" data-origin="technical-insert" id="{esc(paragraph_id)}">'
        f'{highlighted_text(text, phrases, block_id=paragraph_id)}</p>'
    )


def render_physician_paragraph(paragraph: dict[str, object], comment_ids: list[str]) -> str:
    block_id = str(paragraph["id"])
    text = str(paragraph["raw_visible_text"])
    if block_id in HIDDEN_SOURCE_BLOCKS:
        return source_audit(block_id, text) + comment_anchors(comment_ids)
    if block_id in RESULTS_DISCUSSION_OVERLAYS:
        revised = apply_results_discussion_overlay(block_id, text)
        rendered, deleted = word_diff_text(text, revised, block_id=block_id)
        rendered, trailing_anchors = place_comment_anchors(rendered, block_id, comment_ids)
        deleted_attr = f' data-review-deleted="{esc(" | ".join(deleted))}"' if deleted else ""
        if block_id == "P096":
            slug = re.sub(r"[^a-z0-9]+", "-", revised.lower()).strip("-")
            return source_audit(block_id, text) + f'<h3 class="physician-subheading" id="{slug}" data-origin="manuscript-edit"{deleted_attr}>{rendered}{trailing_anchors}</h3>'
        return (
            source_audit(block_id, text)
            + f'<p class="manuscript-paragraph" id="{esc(block_id)}" data-origin="manuscript-edit"{deleted_attr}>'
            + rendered
            + trailing_anchors
            + "</p>"
        )
    if block_id in REVISED_PARAGRAPHS:
        rendered, deleted = word_diff_text(text, REVISED_PARAGRAPHS[block_id], block_id=block_id)
        rendered, trailing_anchors = place_comment_anchors(rendered, block_id, comment_ids)
        deleted_attr = f' data-review-deleted="{esc(" | ".join(deleted))}"' if deleted else ""
        return (
            source_audit(block_id, text)
            + f'<p class="manuscript-paragraph" id="{esc(block_id)}" data-origin="manuscript-edit"{deleted_attr}>'
            + rendered
            + trailing_anchors
            + "</p>"
        )
    if block_id in REVISED_HEADINGS:
        revised = REVISED_HEADINGS[block_id]
        slug = re.sub(r"[^a-z0-9]+", "-", revised.lower()).strip("-")
        rendered, deleted = word_diff_text(text, revised, block_id=block_id)
        deleted_attr = f' data-review-deleted="{esc(" | ".join(deleted))}"' if deleted else ""
        rendered, trailing_anchors = place_comment_anchors(rendered, block_id, comment_ids)
        return source_audit(block_id, text) + f'<h3 class="physician-subheading" id="{slug}" data-origin="manuscript-edit"{deleted_attr}>{rendered}{trailing_anchors}</h3>'
    source_span = (
        f'<span data-origin="physician" data-source-block="{esc(block_id)}">{esc(text)}</span>'
    )
    if not text:
        return f'<div class="source-empty" id="{esc(block_id)}">{source_span}</div>{comment_anchors(comment_ids)}'
    if text in SUBHEADINGS:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return f'<h3 class="physician-subheading" id="{slug}">{source_span}{comment_anchors(comment_ids)}</h3>'
    rendered_source, trailing_anchors = place_comment_anchors(source_span, block_id, comment_ids)
    return f'<p class="physician-paragraph" id="{esc(block_id)}">{rendered_source}{trailing_anchors}</p>'


def render_native_table(
    table: dict[str, object],
    label: str,
    title: str,
    *,
    caption_review_phrases: list[str] | None = None,
) -> str:
    rows = table["rows"]
    table_id = label.lower().replace(" ", "-")
    rendered_title = (
        highlighted_text(title, caption_review_phrases, block_id=f"{table_id}::caption")
        if caption_review_phrases else esc(title)
    )
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{esc(table_id)}">',
        f'<p class="table-caption"><strong>{esc(label)}.</strong> {rendered_title}</p>',
        '<table class="physician-native-table">',
    ]
    for row_index, row in enumerate(rows):
        parts.append("<thead><tr>" if row_index == 0 else "<tbody><tr>")
        tag = "th" if row_index == 0 else "td"
        for cell in row:
            cell_id = str(cell["id"])
            raw_text = str(cell["raw_visible_text"])
            if cell_id in REVISED_TABLE_CELLS:
                revised_text, deleted = word_diff_text(raw_text, REVISED_TABLE_CELLS[cell_id], block_id=cell_id)
                deleted_attr = f' data-review-deleted="{esc(" | ".join(deleted))}"' if deleted else ""
                parts.append(
                    f'<{tag} data-origin="manuscript-edit"{deleted_attr}>{source_audit(cell_id, raw_text)}'
                    f'{revised_text}</{tag}>'
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


def figure(
    number: str,
    title: str,
    caption: str,
    *,
    supplementary: bool = False,
    caption_review_phrases: list[str] | None = None,
) -> str:
    path = SUPPLEMENTARY_FIGURES[number] if supplementary else MAIN_FIGURES[number]
    print_path = PRINT_FIGURES[number]
    validate_print_png(number, print_path)
    print_src = print_path.relative_to(ROOT).as_posix()
    label = f"Supplementary Figure {number}" if supplementary else f"Figure {number}"
    attr = "data-supplementary-figure" if supplementary else "data-main-figure"
    rendered_caption = (
        highlighted_text(caption, caption_review_phrases, block_id=f"figure-{number.lower()}::caption")
        if caption_review_phrases else esc(caption)
    )
    return (
        f'<figure class="figure" id="figure-{number.lower()}" data-origin="technical-insert" {attr}="{number}">'
        f'<img class="figure-screen" src="{svg_data_uri(path)}" alt="{esc(title)}">'
        f'<img class="figure-print" src="{esc(print_src)}" alt="" aria-hidden="true">'
        f'<figcaption class="figure-caption"><strong>{esc(label)}.</strong> {rendered_caption}</figcaption></figure>'
    )


def table_3() -> str:
    return result_table(
        "Table 3", "Physician preferences for breast cancer extraction",
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
        "Table 5", "Adjusted probability of inference harness preference among directional judgments",
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
            ["Active anticancer medications", "Breast cancer and PDAC", "Current_Medications.current_meds", "Anticancer drugs currently being taken or administered. Supportive or home medications, completed or stopped drugs, and drugs that are only planned are excluded."],
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
        "Supplementary Table S3", "Sensitivity analysis of the overall directional GEE after excluding each physician",
        ["Excluded physician", "Directional judgments", "Notes", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        rows, table_id="supplementary-table-s3",
    )


def render_comments(data: dict[str, object]) -> str:
    parts = []
    for comment in data["comments"]:
        comment_id = str(comment["id"])
        if comment_id not in OUTSTANDING_COMMENT_IDS:
            resolution = COMMENT_RESOLUTION_REGISTRY.get(comment_id)
            if not resolution or resolution.get("status") != "resolved":
                raise AssertionError(f"Resolved comment is missing from the resolution registry: {comment_id}")
            resolution_metadata = ""
            if resolution.get("literature_entry") is not None:
                resolution_metadata += f' data-resolved-literature-entry="{resolution["literature_entry"]}"'
            if resolution.get("basis") is not None:
                resolution_metadata += f' data-resolution-basis="{esc(str(resolution["basis"]))}"'
            parts.append(
                f'<span class="source-audit" hidden aria-hidden="true" data-origin="physician-comment" '
                f'data-comment-text-id="{esc(comment_id)}" data-status="resolved" '
                f'{resolution_metadata.strip()}>{hidden_comment_audit_text(str(comment["raw_visible_text"]))}</span>'
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
        "P019": revision_inline_paragraph(
            "We separately conducted an ablation study of four cumulative harness configurations using an OpenAI LLM judge. The study workflow is shown in Figure 1.",
            paragraph_id="study-design-ablation",
            phrases=[],
        ),
        "P038": figure("1", "Development, adaptation, and evaluation workflow", "Development, transfer, and evaluation pathway. Physician review of breast cancer outputs informed the inference harness. The failure history and clinical rubric then supported investigator controlled PDAC adaptation without physician review of PDAC outputs. Four physicians compared the final harness with the same model one prompt baseline on the holdout set of 40 notes."),
        "P082": manuscript_paragraph("Exploratory analysis of core fields used one joint population averaged logistic GEE on 495 directional judgments. Field, physician, and cancer type were fixed effects. Repeated judgments were clustered by note with an exchangeable working correlation. Probabilities for each field were standardized over the physician and cancer distribution observed for that field. Breast type and receptor status applied only to breast cancer notes. Ties were excluded. These estimates describe preference rather than accuracy, and no pairwise field tests or multiplicity claims were made.", paragraph_id="core-field-statistical-analysis"),
        "P085": table_3() + table_4(),
        "P087": figure("2", "Physician preference overview", "Physician preferences across eight combinations of physician and cancer type. Panel A retains harness, baseline, and tie judgments. Panel B shows the harness share among directional judgments."),
        "P092": table_5(),
        "P095": manuscript_paragraph("In the leave one physician out sensitivity analysis, adjusted probabilities of harness preference ranged from 77.08% to 83.53% (Supplementary Table S3).", paragraph_id="loo-results") + manuscript_paragraph("Pairwise exact agreement ranged from 73.1% to 82.9%, with Cohen's kappa ranging from 0.511 to 0.646 (Figure 3 and Supplementary Table S2).", paragraph_id="agreement-results") + figure("3", "Agreement among physicians", "Pairwise exact agreement and Cohen's kappa among four physicians in breast cancer and PDAC. Each breast cancer pair shared 280 judgments, and each PDAC pair shared 260 judgments."),
        "P097": figure("4", "Raw and adjusted preference estimates for core fields", "Preference results for core fields. Panel A retains the raw distribution with ties for seven fields prioritized by physicians. Panel B shows adjusted probabilities of harness preference for each core field with 95% confidence intervals from one joint exploratory GEE fitted to 495 directional judgments. The model included field, physician, and cancer fixed effects, clustered repeated judgments by note with an exchangeable working correlation, and standardized each field over its observed physician and cancer distribution. Breast type and receptor status applied only to breast cancer notes. Ties were excluded from Panel B. The estimates report preference, not accuracy. No pairwise field tests or multiplicity claims were made."),
    }
    return inserts.get(block_id, "")


def technical_inserts_before(block_id: str) -> str:
    if block_id == "P119":
        return '<h3 id="limitations" class="physician-subheading" data-origin="technical-insert">Limitations</h3>'
    return ""


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
            insert_before = technical_inserts_before(block_id)
            if insert_before and current_key:
                current_parts.append(insert_before)
                current_md.extend(["### Limitations", ""])
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
                current_parts.append(render_native_table(
                    tables[table_id],
                    "Table 2",
                    "Ablation study conditions",
                ))
            current_md.extend([f'<!-- physician native table:{table_id}; cells preserved in canonical JSON -->', ""])
    flush()
    return "".join(front_blocks + sections), "\n".join(markdown)


def supplementary_methods() -> str:
    gate_items = [
        "Gate 1 repaired JSON formatting only when parsing failed.",
        "Gate 2 checked the output schema and repaired incorrect or leaked keys.",
        "Gate 3 checked specificity and semantic alignment for vague answers or answers that did not address the task.",
        "Gate 4 trimmed clearly unsupported or contradictory content while retaining supported information.",
        "Gate 5 removed completed or historical events from fields for future plans.",
    ]
    gates = "".join(
        f'<li>{esc(item)}</li>' for item in gate_items
    )
    headings = [
        "Data access and holdout definition",
        "Assessment and Plan identification",
        "Verification and postprocessing order",
        "Insufficient source information",
        "Model roles during PDAC adaptation",
        "Physician evaluation interface",
        "Ablation study judge",
    ]
    rendered_headings = [esc(text) for text in headings]
    return (
        '<div class="supplement-methods" id="supplementary-methods">'
        '<h3>Supplementary Methods</h3>'
        + f'<h4>{rendered_headings[0]}</h4>'
        + revision_inline_paragraph(
            "CORAL access is governed by the PhysioNet credentialing and data use process specified by the dataset authors. The 40 holdout note texts and their existing expert annotations were excluded from development, error analysis, prompt revision, rule development, and regression testing before the final harness was frozen. The annotations were not provided to the models, not used during harness development, and not used to calculate the primary physician preference endpoint. They were not shown in the evaluation interface.",
            paragraph_id="supplement-data-access",
            phrases=[
                "not provided to the models",
                "not used during harness development",
                "not used to calculate the primary physician preference endpoint",
            ],
        )
        + f'<h4>{rendered_headings[1]}</h4>'
        + revision_inline_paragraph(
            "The harness first used regular expressions to locate the Assessment and Plan section. If no section was found, the model made up to three attempts to reproduce the relevant source text. Each candidate was checked against the original note. If all attempts failed, the harness skipped only the plan branch and continued extraction from the complete note.",
            paragraph_id="supplement-plan-identification",
            phrases=[],
        )
        + f'<h4>{rendered_headings[2]}</h4>'
        + '<p class="manuscript-paragraph" data-origin="technical-insert">'
        + esc("Each model output for a field passed through the following gates in order.")
        + '</p>'
        + f'<ol data-origin="technical-insert">{gates}</ol>'
        + revision_inline_paragraph(
            "After the gates, deterministic filters for each field and clinical hooks across fields addressed repeated errors. Source attribution followed processing after extraction when enabled.",
            paragraph_id="supplement-postprocessing-order",
            phrases=[],
        )
        + f'<h4>{rendered_headings[3]}</h4>'
        + revision_inline_paragraph(
            "A hypothetical note may contain imaging findings or other clinical details associated with advanced disease but still lack enough confirmed information supported by the source to assign a formal cancer stage under the extraction rules. In that case, the system returns not stated or leaves the stage field empty. This example does not describe a CORAL patient.",
            paragraph_id="supplement-stage-example",
            phrases=[],
        )
        + f'<h4>{rendered_headings[4]}</h4>'
        + revision_inline_paragraph(
            "ChatGPT organized the error categories and clinical distinctions from breast cancer development into a structured review rubric. A local Qwen reviewer used that rubric to compare pipeline outputs with complete pancreatic source notes and flag potential errors. Investigators decided which changes to implement and retained them only after regression testing. Neither ChatGPT nor the Qwen reviewer served as the ablation judge.",
            paragraph_id="supplement-pdac-model-roles",
            phrases=[],
        )
        + f'<h4>{rendered_headings[5]}</h4>'
        + revision_inline_paragraph(
            "The interface used a fixed position mapping. A displayed the inference harness output, and B displayed the one prompt baseline. Physicians were blinded to system identities. Source attribution was available only with the harness output when present. Physicians reviewed the complete output packages.",
            paragraph_id="supplement-evaluation-interface",
            phrases=[
                "A displayed the inference harness output",
                "B displayed the one prompt baseline",
                "blinded to system identities",
                "Source attribution was available only with the harness output",
                "complete output packages",
            ],
        )
        + f'<h4>{rendered_headings[6]}</h4>'
        + revision_inline_paragraph(
            "The ablation judge was Codex using model gpt-5.6-sol, reasoning effort xhigh, and service tier default. It reviewed adjacent configuration pairs and received only the source note and two outputs. ChatGPT was used only for the PDAC review rubric role described above.",
            paragraph_id="supplement-ablation-judge",
            phrases=[],
        )
        + "</div>"
    )


def build_supplement() -> str:
    return (
        '<section class="fold supplement custom-fold is-collapsed" id="supplementary-material" data-origin="technical-insert">'
        '<button class="fold-toggle" type="button" aria-expanded="false" aria-controls="supplement-body">Supplementary material</button>'
        '<div class="fold-body" id="supplement-body">'
        + supplementary_methods()
        + supplementary_table_1()
        + supplementary_table_2()
        + figure("S1", "Distribution of preference margins by note", "This figure shows the distribution by note after pooling all four physicians within each cancer type, previously drafted as Panel C of the main figure. The margin is the difference between harness and baseline preferences divided by completed judgments for that note. All 20 breast cancer notes and 19 of 20 PDAC notes had positive margins. One PDAC note had a negative margin. The prior detailed breast and PDAC note panels are not retained. This is a supporting preference sensitivity analysis, not an accuracy estimate.", supplementary=True)
        + figure(
            "S2",
            "Ablation study",
            "Comparisons across three transitions between four cumulative configurations. A was the one prompt baseline. B added decomposition and orchestration. C added semantic, accuracy, source support, and temporal verification. D added deterministic inline processing and the complete POST hook layer. One OpenAI LLM judge produced the judgments. Exact two sided sign tests exclude ties.",
            supplementary=True,
        )
        + supplementary_table_3()
        + "</div></section>"
    )


def version_history() -> str:
    entries = [
        ("v1.6.7", "2026-11-01", "Current change", "Added a main manuscript Limitations heading and highlighted only five approved P119 design limitation phrases for focused review. The limitation text is unchanged."),
        ("v1.6.6", "2026-11-01", "Comment 14 resolution", "Closed physician Comment 14 after verifying breast ER, PR, and HER2 rules in the prompt, breast-only routing, hooks, and tests. Comment 13 remains open."),
        ("v1.6.5", "2026-11-01", "Approved evaluation limitations", "Restored five user approved evaluation design limitations in P119 without changing the remaining physician prose or adding review highlights."),
        ("v1.6.4", "2026-11-01", "Canonical Results wording", "Restored the canonical Results confidence interval wording with only hyphen normalization and restored the canonical capitalization of the clinical extraction fields heading."),
        ("v1.6.3", "2026-11-01", "Results and Discussion section revert", "Reset Results and Discussion physician prose to the canonical Word draft for physician review. Retained only approved terminology, numbers, cross references, citations, comment anchors, and independently labeled technical result inserts."),
        ("v1.6.2", "2026-11-01", "Adjacent focus correction", "Removed the adjacent second focus phrase in the system identity sentence and added a DOM aware check that rejects adjacent focus spans when their combined visual coverage forms a complete sentence."),
        ("v1.6.1", "2026-10-31", "Literature framing", "Separated the Huang and Hein study descriptions to preserve their distinct methods, narrowed five complete sentence highlights to short review phrases, and retained the 24 browser comment revisions without changing Results."),
        ("v1.6", "2026-10-31", "Browser comments", "Resolved 24 browser review comments. Reworked the Introduction and Study Design, clarified the holdout, baseline, plan branch, evaluation interface, and ablation judge, replaced broad revision coloring with a small set of manual focus phrases, and completed a language audit for unnecessary hyphens and colons."),
        ("v1.5.2", "2026-10-31", "Word level review", "Replaced whole-block yellow review backgrounds with token-level highlights. Only inserted or replaced words and selected key phrases in wholly new passages are colored; manuscript wording, statistics, links, numbering, and comments are unchanged."),
        ("v1.5.1", "2026-10-31", "Added light, medium, and strong review highlights to show the importance of the 44 wording changes already identified against v1.3. Manuscript wording and Results are unchanged; deletions remain in the version diff."),
        ("v1.5", "2026-10-31", "Clarified the five-paragraph Introduction and added a complete block-level redline against the v1.3 visible manuscript, including the revised ablation discussion. Every visible manuscript block with changed wording is highlighted; deletions are recorded in the version diff."),
        ("v1.4.1", "2026-10-31", "Simplified the A/B interface and ablation-study wording, defined the ablation purpose as examining output-quality changes across successive harness configurations, and aligned the baseline condition with the application-level inference-request definition."),
        ("v1.4", "2026-10-31", "Reorganized the five-paragraph Introduction while retaining the physician's language, clarified the two-stage Methods and holdout design, moved gate and interface details to Supplementary Methods, and added the corresponding evaluation limitation."),
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
        ("v0.17", "Date not recorded", "Established the first tracked multi-physician analysis and ablation-study draft."),
    ]
    rendered = []
    for entry in entries:
        if len(entry) == 4:
            version, date, label, note = entry
        else:
            version, date, note = entry
            label = ""
        current = version == "v1.6.7"
        open_attr = " open" if current else ""
        summary = f"{version} · {label} · {date}" if label else f"{version} · {date}"
        rendered.append(
            f'<details class="version-entry" id="version-{esc(version[1:].replace(".", "-"))}" data-origin="technical-insert"{open_attr}>'
            f'<summary>{esc(summary)}</summary><div class="version-entry-body"><p>{esc(note)}</p></div></details>'
        )
    return "".join(rendered)


def toc() -> str:
    links = [("#section-introduction", "Introduction", False), ("#section-methods", "Methods", False)]
    for heading in ["Study Design", "Dataset", "Baseline model and model development", "Inference Harness", "Breast cancer development with physician review", "PDAC adaptation with feedback from language models", "Clinical evaluation data collection and scoring rubric", "Ablation study", "Statistical analysis"]:
        links.append(("#" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-"), heading, True))
    links.extend([("#section-results", "Results", False), ("#performance-across-clinical-extraction-fields", "Performance across clinical extraction fields", True), ("#section-discussion", "Discussion", False), ("#limitations", "Limitations", True), ("#supplementary-material", "Supplementary material", False), ("#physician-comments", "Outstanding physician comments (1)", False), ("#version-history", "Version history", False), ("#literature-workspace", "Literature-review workspace", False)])
    return '<nav class="toc" data-origin="technical-insert"><strong>Contents</strong>' + "".join(f'<a class="{"sub" if sub else ""}" href="{href}" data-origin="technical-insert">{esc(label)}</a>' for href, label, sub in links) + "</nav>"


def build_html(data: dict[str, object]) -> tuple[str, str]:
    sections, markdown_body = build_scientific_sections(data)
    article = f"""
<article>
  <h1 class="screen-only">Physician-authored manuscript draft</h1>
  <p class="subtitle screen-only">English working manuscript · Version 1.6.7 · November 1, 2026</p>
  <details class="fold" id="version-history"><summary>Version history</summary><div class="fold-body">{version_history()}</div></details>
  {sections}
  {build_supplement()}
  <details class="fold" id="physician-comments"><summary>Outstanding physician comments (1)</summary><div class="fold-body">{render_comments(data)}</div></details>
  {literature_workspace()}
</article>
"""
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physician-authored manuscript draft v1.6.7</title><style>{CSS}</style></head><body><div class="layout">{toc()}{article}</div><script>{SCRIPT}</script></body></html>'
    document = apply_review_block_metadata(document)
    markdown = f"""<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 1.6.7, November 1, 2026.

Current change: added a main manuscript Limitations heading and highlighted only five approved P119 design limitation phrases for focused review. The limitation text is unchanged.

The canonical physician blocks below mirror `{SOURCE_DOCX.name}` for audit purposes. The clean HTML applies only the approved cross-reference and reporting edits while preserving all source blocks in its hidden audit representation.

{markdown_body}

## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Outstanding physician comments

The canonical extraction retains all 12 Word comments. The HTML displays the 1 unresolved comment and keeps the 11 resolved comments only in its hidden audit representation.

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
        if "focus-highlight" in str(values.get("class", "")).split(): self.revision_highlights += 1
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


class FocusCoverageParser(HTMLParser):
    """Map focus spans to their DOM text offsets inside each review block."""

    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active_id: str | None = None
        self.parts: list[str] = []
        self.length = 0
        self.ranges: list[tuple[int, int]] = []
        self.stack: list[tuple[str, bool, int | None]] = []
        self.blocks: dict[str, tuple[str, list[tuple[int, int]]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        starts_block = values.get("data-review-block-id") is not None
        if starts_block:
            if self.active_id is not None:
                raise AssertionError("Nested review blocks are not supported by the focus coverage audit")
            self.active_id = str(values["data-review-block-id"])
            self.parts = []
            self.length = 0
            self.ranges = []
        starts_focus = self.active_id is not None and "focus-highlight" in str(values.get("class", "")).split()
        focus_start = self.length if starts_focus else None
        if tag not in self.VOID_TAGS:
            self.stack.append((tag, starts_block, focus_start))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.active_id is not None:
            self.parts.append(data)
            self.length += len(data)

    def handle_endtag(self, tag: str) -> None:
        if self.active_id is None:
            return
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] != tag:
                continue
            closing = self.stack[index:]
            del self.stack[index:]
            for _, starts_block, focus_start in reversed(closing):
                if focus_start is not None:
                    self.ranges.append((focus_start, self.length))
                if starts_block:
                    if self.active_id in self.blocks:
                        raise AssertionError(f"Duplicate focus coverage block: {self.active_id}")
                    self.blocks[self.active_id] = ("".join(self.parts), sorted(self.ranges))
                    self.active_id = None
                    self.parts = []
                    self.length = 0
                    self.ranges = []
            return


def sentence_sized_focus_runs(document: str) -> list[str]:
    """Return blocks where adjacent focus spans visually cover a full sentence."""

    parser = FocusCoverageParser()
    parser.feed(document)
    offenders: list[str] = []
    for block_id, (text, ranges) in parser.blocks.items():
        if not ranges:
            continue
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and not any(character.isalnum() for character in text[merged[-1][1]:start]):
                merged[-1][1] = end
            else:
                merged.append([start, end])
        sentence_start = 0
        sentence_cores: list[tuple[int, int]] = []
        for boundary in re.finditer(r"[.!?](?=\s|$)", text):
            core_start = sentence_start
            core_end = boundary.start()
            while core_start < core_end and text[core_start].isspace():
                core_start += 1
            while core_end > core_start and text[core_end - 1].isspace():
                core_end -= 1
            if core_start < core_end:
                sentence_cores.append((core_start, core_end))
            sentence_start = boundary.end()
        core_start = sentence_start
        core_end = len(text)
        while core_start < core_end and text[core_start].isspace():
            core_start += 1
        while core_end > core_start and text[core_end - 1].isspace():
            core_end -= 1
        if core_start < core_end:
            sentence_cores.append((core_start, core_end))
        if any(run_start == core_start and run_end == core_end for run_start, run_end in merged for core_start, core_end in sentence_cores):
            offenders.append(block_id)
    return offenders


def section_markup(document: str, section: str, next_section: str) -> str:
    section_marker = document.index(f'id="section-{section}"')
    start = document.rfind("<details", 0, section_marker)
    next_marker = document.index(f'id="section-{next_section}"', section_marker)
    end = document.rfind("<details", section_marker, next_marker)
    if start < 0 or end <= start:
        raise AssertionError(f"Unable to isolate section markup: {section}")
    return document[start:end]


def section_block_text(document: str, section: str) -> str:
    blocks = collect_review_blocks(document)
    return "\n".join(f"{block_id}\t{block.text}" for block_id, block in blocks.items() if block.section == section)


def rendered_physician_prose(document: str, block_id: str) -> str:
    match = re.search(
        rf'<(?:p|h3|summary)[^>]*(?:\bid|\bdata-review-block-id)="{re.escape(block_id)}"[^>]*>(.*?)</(?:p|h3|summary)>',
        document,
        flags=re.S,
    )
    if not match:
        raise AssertionError(f"Visible physician block is missing: {block_id}")
    body = re.sub(r'<sup class="comment-anchor".*?</sup>', "", match.group(1), flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", body)).split())


def overlay_display_text(text: str) -> str:
    return " ".join(
        re.sub(
            r"\[\[cite:([a-z_]+)\]\]",
            lambda match: EXTERNAL_CITATIONS[match.group(1)][0],
            text,
        ).split()
    )


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
        ".focus-highlight{background:transparent!important;outline:0!important;color:inherit}",
    ]
    if any(rule not in document for rule in print_requirements):
        raise AssertionError("Print view does not isolate and constrain the clean manuscript")
    if document.count('class="figure-print"') != 6 or document.count('class="figure-screen"') != 6:
        raise AssertionError("Print-safe figure fallbacks are incomplete")
    for print_path in PRINT_FIGURES.values():
        source = print_path.relative_to(ROOT).as_posix()
        if document.count(f'src="{source}"') != 1:
            raise AssertionError(f"Print-safe figure is not linked exactly once: {source}")
    figure_paths = list(MAIN_FIGURES.values()) + list(SUPPLEMENTARY_FIGURES.values()) + list(PRINT_FIGURES.values())
    for figure_path in figure_paths:
        expected_hash = EXPECTED_FIGURE_SHA256.get(figure_path.name)
        actual_hash = hashlib.sha256(figure_path.read_bytes()).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise AssertionError(f"Figure resource changed: {figure_path.name}")
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
    if parser.visible_comment_ids != ["13"]: raise AssertionError(f"Unexpected visible physician comments: {parser.visible_comment_ids}")
    if "Outstanding physician comments (1)" not in document: raise AssertionError("Outstanding-comment heading is missing")
    if set(COMMENT_RESOLUTION_REGISTRY) != {
        str(item["id"]) for item in data["comments"] if str(item["id"]) not in OUTSTANDING_COMMENT_IDS
    }:
        raise AssertionError("Comment resolution registry does not match the resolved canonical comments")
    for comment_id, literature_entry in RESOLVED_COMMENT_LITERATURE_MAP.items():
        marker = f'data-comment-text-id="{comment_id}" data-status="resolved" data-resolved-literature-entry="{literature_entry}"'
        if marker not in document: raise AssertionError(f"Resolved literature comment mapping is missing: {comment_id}")
    comment_14_marker = (
        'data-comment-text-id="14" data-status="resolved" '
        f'data-resolution-basis="{esc(COMMENT_14_RESOLUTION_BASIS)}"'
    )
    if comment_14_marker not in document:
        raise AssertionError("Resolved Comment 14 evidence is missing from the hidden audit")
    redline = review_redline_state(document)
    changed = redline["changed"]
    highlighted = redline["highlighted"]
    scoped_current = redline["scoped_current"]
    if changed != highlighted:
        raise AssertionError(
            f"Changed and highlighted block sets differ: missing={sorted(changed - highlighted)}, "
            f"orphan={sorted(highlighted - changed)}"
        )
    unchanged_highlighted = {block_id for block_id in highlighted if block_id in scoped_current and block_id not in changed}
    if unchanged_highlighted:
        raise AssertionError(f"Unchanged review blocks are highlighted: {sorted(unchanged_highlighted)}")
    if any(block.changed_from and block.changed_from != REVIEW_BASELINE_VERSION for block in redline["current"].values()):
        raise AssertionError("An orphan review baseline marker is present")
    if redline["baseline"]["block_count"] != 341:
        raise AssertionError("The v1.3 review-baseline block count changed")
    if "revision-legend" in document or "revision-swatch" in document or "revision-light" in document or "revision-medium" in document or "revision-strong" in document:
        raise AssertionError("Obsolete review legend or importance styling remains")
    if "<mark" in document:
        raise AssertionError("Review markup must use accessible span metadata")
    block_level_highlights = re.findall(
        r'<(?:p|h3|h4|li|td|th|figcaption)[^>]*class="[^"]*\bfocus-highlight\b',
        document,
    )
    if block_level_highlights:
        raise AssertionError("A whole paragraph, heading, cell, list item, or caption is yellow")
    revision_spans = re.findall(
        r'<span (?P<attrs>[^>]*class="[^"]*\bfocus-highlight\b[^"]*"[^>]*)>(?P<body>.*?)</span>',
        document,
        flags=re.S,
    )
    if len(revision_spans) != parser.revision_highlights:
        raise AssertionError("Revision-span parser counts differ")
    span_blocks: set[str] = set()
    span_texts: set[str] = set()
    span_records: list[tuple[str, str]] = []
    for attrs, body in revision_spans:
        block_match = re.search(r'data-review-for="([^"]+)"', attrs)
        if not block_match:
            raise AssertionError("A word-level review span lacks its block ID")
        block_id = html.unescape(block_match.group(1))
        text = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        if not text or len(text) > MAX_REVIEW_SPAN_CHARS:
            raise AssertionError(f"Invalid review-span length in {block_id}: {len(text)}")
        if "external-citation" in body or "comment-anchor" in body:
            raise AssertionError(f"Citation or comment marker is inside a yellow span: {block_id}")
        if not re.search(r'title="[^"]+"', attrs) or not re.search(r'aria-label="[^"]+"', attrs):
            raise AssertionError(f"A review span lacks hover or accessibility metadata: {block_id}")
        span_blocks.add(block_id)
        span_texts.add(text)
        span_records.append((block_id, text))
    expected_span_blocks = set(MANUAL_FOCUS_PHRASES) | {"supplement-data-access", "supplement-evaluation-interface"}
    if span_blocks != expected_span_blocks:
        raise AssertionError(
            f"Manual focus coverage differs: missing={sorted(expected_span_blocks - span_blocks)}, "
            f"orphan={sorted(span_blocks - expected_span_blocks)}"
        )
    for required_phrase in [
        "focuses on that extraction task",
        "outputs can vary", "checking every field in every output can approach the cost of manual extraction",
        "privacy and governance concerns", "local models can still produce unstable or incorrect extractions",
        "without physician review", "No physician reviewed pancreatic cancer outputs",
        "System identities", "judge model and settings",
        "not provided to the models", "not used during harness development",
        "not used to calculate the primary physician preference endpoint",
        "A displayed the inference harness output", "B displayed the one prompt baseline",
        "blinded to system identities", "Source attribution was available only with the harness output",
        "complete output packages",
    ]:
        if required_phrase not in span_texts:
            raise AssertionError(f"Required focused review phrase is missing: {required_phrase}")
    for paragraph_id in ["P004", "P008", "P010", "P012"]:
        paragraph = re.search(rf'<p[^>]*id="{paragraph_id}"[^>]*>(.*?)</p>', document, flags=re.S)
        if not paragraph:
            raise AssertionError(f"Introduction paragraph missing: {paragraph_id}")
        plain = html.unescape(re.sub(r"<[^>]+>", "", paragraph.group(1))).strip()
        focus_chars = sum(len(html.unescape(re.sub(r"<[^>]+>", "", body)).strip()) for attrs, body in revision_spans if f'data-review-for="{paragraph_id}"' in attrs)
        if focus_chars >= len(plain) * 0.45:
            raise AssertionError(f"Introduction focus density is too high: {paragraph_id}")
    source_highlighted_paragraphs = {
        str(item["id"]) for item in data["paragraphs"]
        if any(bool(segment.get("highlighted")) for segment in item.get("segments", []))
    }
    if source_highlighted_paragraphs != {"P060", "P062", "P063", "P115"}:
        raise AssertionError("Canonical Word-highlight audit changed")
    for revision_id in [
        "study-design-ablation", "supplement-data-access", "supplement-plan-identification",
        "supplement-postprocessing-order", "supplement-stage-example", "supplement-pdac-model-roles",
        "supplement-evaluation-interface", "supplement-ablation-judge",
    ]:
        if f'id="{revision_id}"' not in document: raise AssertionError(f"Clarification is missing: {revision_id}")
    if 'id="draft-metadata"' in document or "Draft information and source provenance" in document: raise AssertionError("Removed draft-information block remains")
    if "Version 1.6.7 · November 1, 2026" not in document or "draft v1.6.7" not in document: raise AssertionError("Version 1.6.7 metadata is missing")
    if document.count('class="version-entry"') != 29 or any(f'id="version-{item}"' not in document for item in ["1-6-7", "1-6-6", "1-6-5", "1-6-4", "1-6-3", "1-6-2", "1-6-1", "1-6", "1-5-2", "1-5-1", "1-5", "1-4-1", "1-4", "1-3", "1-2-3", "1-2-2", "1-2-1", "1-2", "1-1", "1-0", "0-25", "0-24", "0-23", "0-22", "0-21", "0-20", "0-19", "0-18", "0-17"]):
        raise AssertionError("Nested version history is incomplete")
    if not re.search(r'<details class="version-entry" id="version-1-6-7"[^>]* open>', document):
        raise AssertionError("Current v1.6.7 history entry is not expanded inside the closed parent")
    if any(re.search(rf'<details class="version-entry" id="version-{item}"[^>]* open>', document) for item in ["1-6-6", "1-6-5", "1-6-4", "1-6-3", "1-6-2", "1-6-1", "1-6", "1-5-2", "1-5-1", "1-5", "1-4-1", "1-4", "1-3"]):
        raise AssertionError("Prior version-history entries must be closed")
    if "v1.6.7 · Current change · 2026-11-01" not in document:
        raise AssertionError("Current version-history label is missing")
    if "querySelectorAll('#version-history .version-entry').forEach(entry=>entry.open=entry===target)" not in document:
        raise AssertionError("Version-history deep links do not isolate the targeted entry")
    if 'lang="zh"' in document or 'id="chinese-version"' in document: raise AssertionError("Chinese manuscript shell remains")
    manuscript_blocks = collect_review_blocks(document)
    document_without_comment_ui = re.sub(
        r'<sup class="comment-anchor".*?</sup>', "", document, flags=re.S
    )
    blocks_without_comment_ui = collect_review_blocks(document_without_comment_ui)
    visible_text_without_comment_ui = "\n".join(
        f"{block_id}\t{block.text}"
        for block_id, block in blocks_without_comment_ui.items()
        if block_id != "limitations"
    )
    if (
        hashlib.sha256(visible_text_without_comment_ui.encode("utf-8")).hexdigest()
        != EXPECTED_V166_VISIBLE_BLOCK_TEXT_EXCLUDING_COMMENT_UI_SHA256
    ):
        raise AssertionError("Visible manuscript block text changed outside comment UI")
    for section, next_section in [("introduction", "methods"), ("methods", "results")]:
        markup_hash = hashlib.sha256(section_markup(document, section, next_section).encode("utf-8")).hexdigest()
        text_hash = hashlib.sha256(section_block_text(document, section).encode("utf-8")).hexdigest()
        if markup_hash != F7_SECTION_BASELINE[section]["markup_sha256"] or text_hash != F7_SECTION_BASELINE[section]["text_sha256"]:
            raise AssertionError(f"{section.title()} changed from commit f7b0f34d")
    results_discussion_focus = {
        block_id for block_id, _ in span_records
        if re.fullmatch(r"P\d{3}", block_id) and 84 <= int(block_id[1:]) <= 119
    }
    p119_focus = [text for block_id, text in span_records if block_id == "P119"]
    if results_discussion_focus != {"P119"} or p119_focus != MANUAL_FOCUS_PHRASES["P119"]:
        raise AssertionError("Results or Discussion focus differs from the five approved P119 phrases")
    if not re.search(r'<h3[^>]*id="limitations"[^>]*>Limitations</h3>', document):
        raise AssertionError("Main manuscript Limitations heading is missing")
    if '<a class="sub" href="#limitations" data-origin="technical-insert">Limitations</a>' not in document:
        raise AssertionError("Limitations TOC link is missing")
    if hashlib.sha256(rendered_physician_prose(document, "P119").encode("utf-8")).hexdigest() != EXPECTED_V167_P119_TEXT_SHA256:
        raise AssertionError("P119 wording changed while adding the Limitations presentation")
    for paragraph in data["paragraphs"]:
        block_id = str(paragraph["id"])
        if not re.fullmatch(r"P\d{3}", block_id) or not 84 <= int(block_id[1:]) <= 119:
            continue
        canonical_text = str(paragraph["raw_visible_text"])
        if not canonical_text or block_id in HIDDEN_SOURCE_BLOCKS or block_id in {"P084", "P107"}:
            continue
        expected_visible = overlay_display_text(apply_results_discussion_overlay(block_id, canonical_text))
        actual_visible = rendered_physician_prose(document, block_id)
        if actual_visible != expected_visible:
            raise AssertionError(f"Results or Discussion differs from its canonical base and whitelist: {block_id}")
    results_text = "\n".join(
        f"{block_id}\t{block.text}" for block_id, block in manuscript_blocks.items() if block.section == "results"
    )
    if hashlib.sha256(results_text.encode("utf-8")).hexdigest() != EXPECTED_V164_RESULTS_SHA256:
        raise AssertionError("Visible v1.6.4 Results hash changed")
    result_numbers = re.findall(
        r"(?<![A-Za-z])(?:[+-]?\d[\d,]*(?:\.\d+)?%?|[+-]?\d+(?:\.\d+)?e-\d+)",
        results_text,
        flags=re.I,
    )
    if hashlib.sha256("\n".join(result_numbers).encode("utf-8")).hexdigest() != EXPECTED_V164_RESULTS_NUMERIC_SHA256:
        raise AssertionError("Authoritative Results number sequence changed")
    semantic_results_text = re.sub(r"\bseven\b", "7", results_text, flags=re.I)
    semantic_result_numbers = re.findall(
        r"(?<![A-Za-z])(?:[+-]?\d[\d,]*(?:\.\d+)?%?|[+-]?\d+(?:\.\d+)?e-\d+)",
        semantic_results_text,
        flags=re.I,
    )
    if (
        hashlib.sha256("\n".join(semantic_result_numbers).encode("utf-8")).hexdigest()
        != EXPECTED_F7_V164_SEMANTIC_RESULTS_NUMERIC_SHA256
    ):
        raise AssertionError("Semantic Results number sequence differs from commit f7b0f34d")
    for block_id, focused_text in span_records:
        block = manuscript_blocks.get(block_id)
        if not block:
            raise AssertionError(f"Focused review block is missing: {block_id}")
        sentences = {
            sentence.strip().rstrip(".!?").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block.text)
            if sentence.strip()
        }
        if focused_text.strip().rstrip(".!?").strip() in sentences:
            raise AssertionError(f"A focus highlight covers a complete sentence: {block_id}")
    combined_sentence_offenders = sentence_sized_focus_runs(document)
    if combined_sentence_offenders:
        raise AssertionError(f"Adjacent focus spans visually cover a complete sentence: {combined_sentence_offenders}")
    manuscript_text = " ".join(block.text for block in manuscript_blocks.values())
    visible_text_lower = manuscript_text.lower()
    if "—" in manuscript_text or "–" in manuscript_text: raise AssertionError("Visible manuscript contains an em or en dash")
    remaining_hyphens = set(re.findall(r"[A-Za-z0-9.]+(?:-[A-Za-z0-9.]+)+", manuscript_text))
    allowed_model_ids = {"Qwen2.5-32B-Instruct-AWQ", "Qwen2.5-32B", "gpt-5.6-sol"}
    unjustified_hyphens = {
        token for token in remaining_hyphens
        if token not in allowed_model_ids
        and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?-[+-]?\d+(?:\.\d+)?", token)
        and not re.fullmatch(r"\d+(?:\.\d+)?e-\d+", token, flags=re.I)
    }
    if unjustified_hyphens:
        raise AssertionError(f"Unnecessary visible hyphens remain: {sorted(unjustified_hyphens)}")
    if ":" in manuscript_text:
        raise AssertionError("An avoidable colon remains in the visible manuscript")
    if "[22]" in parser.visible_text: raise AssertionError("Orphaned numeric citation remains")
    if "83.62)." in parser.visible_text or "83.62)" in rendered_physician_prose(document, "P092"): raise AssertionError("P092 confidence interval is missing its percent sign")
    if re.search(r"[\u4e00-\u9fff]", manuscript_text): raise AssertionError("Chinese text remains in the visible manuscript")
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
        threshold = 0.10 if paragraph_id in SUBSTANTIVE_REVISED_IDS else 0.58
        if SequenceMatcher(None, original_words, revised_words).ratio() < threshold:
            raise AssertionError(f"Revision exceeded allowed edit distance: {paragraph_id}")
    reused_words, revised_words, reuse_ratio = introduction_content_word_reuse(data)
    if reuse_ratio < 0.48:
        raise AssertionError(f"Introduction physician-language reuse fell below 48%: {reuse_ratio:.2%}")
    if set(REVISED_HEADINGS) != {"P055", "P059", "P066"}: raise AssertionError("Revised heading registry changed")
    if set(REVISED_TABLE_CELLS) != {"T1R2C1", "T1R3C1", "T1R4C2", "T1R5C2", "T1R5C3", "T1R6C2", "T1R6C3", "T1R8C1", "T2R1C2", "T2R2C2", "T2R4C2"}: raise AssertionError("Native-table revision registry changed")
    for cell_id, revised_cell in REVISED_TABLE_CELLS.items():
        rendered_cell, _ = word_diff_text(expected_cells[cell_id], revised_cell, block_id=cell_id)
        if rendered_cell not in document: raise AssertionError(f"Revised native-table text is missing: {revised_cell}")
    expected_inline_urls = []
    for paragraph in data["paragraphs"]:
        block_id = str(paragraph["id"])
        revised = (
            apply_results_discussion_overlay(block_id, str(paragraph["raw_visible_text"]))
            if block_id in RESULTS_DISCUSSION_OVERLAYS
            else REVISED_PARAGRAPHS.get(block_id, "")
        )
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
        "P097": "#figure-4", "P100": "#figure-s1", "P117": "#figure-s2",
    }
    for paragraph_id, href in required_crossrefs.items():
        match = re.search(rf'id="{paragraph_id}"[^>]*>(.*?)</p>', document, flags=re.S)
        if not match or f'href="{href}"' not in match.group(1): raise AssertionError(f"Cross-reference missing from {paragraph_id}: {href}")
    if 'href="#table-4"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1) or 'href="#figure-2"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1):
        raise AssertionError("P085 does not cite Tables 3-4 and Figure 2")
    loo_match = re.search(r'id="loo-results"[^>]*>(.*?)</p>', document, flags=re.S)
    if not loo_match or 'href="#supplementary-table-s3"' not in loo_match.group(1):
        raise AssertionError("Independent LOO result or its Supplementary Table S3 link is missing")
    p095_match = re.search(r'id="P095"[^>]*>(.*?)</p>', document, flags=re.S)
    if not p095_match or "77.08%" in p095_match.group(1) or "83.53%" in p095_match.group(1):
        raise AssertionError("LOO range remains mixed into physician P095")
    if 'href="#figure-3"' not in document or 'href="#supplementary-table-s2"' not in document: raise AssertionError("Agreement Results cross-references are missing")
    p115_html = re.search(r'id="P115"[^>]*>(.*?)</p>', document, flags=re.S)
    p117_html = re.search(r'id="P117"[^>]*>(.*?)</p>', document, flags=re.S)
    if not p115_html or not p117_html:
        raise AssertionError("Discussion comment anchor paragraphs are missing")
    regimen_sentence = "We addressed this issue by incorporating treatment history and temporal context into the extraction process and adding verification steps to align treatment responses with the appropriate regimen."
    if f'{esc(regimen_sentence)}<sup class="comment-anchor" id="comment-anchor-13">' not in p115_html.group(1):
        raise AssertionError("Comment 13 is not anchored to the appropriate regimen sentence")
    er_sentence = "For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC."
    if esc(er_sentence) not in p117_html.group(1):
        raise AssertionError("The P117 ER, PR, and HER2 sentence changed")
    if 'comment-anchor-14' in p117_html.group(1) or 'id="physician-comment-14"' in document:
        raise AssertionError("Resolved Comment 14 remains visible")
    p113_html = re.search(r'id="P113"[^>]*>(.*?)</p>', document, flags=re.S)
    p119_html = re.search(r'id="P119"[^>]*>(.*?)</p>', document, flags=re.S)
    if not p113_html or EXTERNAL_CITATIONS["williams"][1] not in p113_html.group(1):
        raise AssertionError("Williams citation is not attached to P113")
    if not p119_html or EXTERNAL_CITATIONS["williams"][1] in p119_html.group(1):
        raise AssertionError("Williams citation remains attached to P119")
    approved_added_limitations = [
        "fixed A/B display order",
        "source attribution only in the harness output",
        "direction and magnitude of any effect from this asymmetry could not be isolated",
        "same base model family",
        "generalization across model families was not assessed",
    ]
    p119_text = rendered_physician_prose(document, "P119")
    if any(phrase.lower() not in p119_text.lower() for phrase in approved_added_limitations):
        raise AssertionError("P119 is missing a user approved evaluation design limitation")
    for stale in ["figure5_staged_ablation.svg", "figure6_clinical_handoff.svg", 'data-main-figure="5"', 'data-main-figure="6"']:
        if stale in document: raise AssertionError(f"Retired main figure remains: {stale}")
    for obsolete_wording in [
        "system-identity-masked a/b interface",
        "exploratory staged technical ablation",
        "staged technical ablation",
        "one model call per note",
        "to examine how successive harness components contributed to the observed improvement",
    ]:
        if obsolete_wording in visible_text_lower:
            raise AssertionError(f"Obsolete visible wording remains: {obsolete_wording}")
    revised_ablation_purpose = "to examine how output quality changed across successive harness configurations"
    p067_match = re.search(r'<p[^>]*id="P067"[^>]*>(.*?)</p>', document, flags=re.S)
    if not p067_match or revised_ablation_purpose not in re.sub(r"<[^>]+>", "", p067_match.group(1)) or "P067" not in highlighted:
        raise AssertionError("Revised ablation purpose is missing or its block is not highlighted")
    required_plain_language = {
        "P004": "This study focuses on that extraction task.",
        "P008": "checking every field in every output can approach the cost of manual extraction",
        "P010": "no physician reviews the PDAC development outputs",
        "P012": "To address this gap",
        "P017": "without physician review of the PDAC development outputs",
        "P022": "A separate untouched holdout set of 40 notes annotated by experts consisted of 20 breast cancer notes and 20 PDAC notes.",
        "P025": "Each note was processed in one chat completion request.",
        "P028": "If this step failed",
        "P060": "No physician reviewed pancreatic cancer outputs during this stage.",
        "P061": "modifications specific to pancreatic cancer",
        "P063": "System identities were not disclosed.",
        "P073": "Details of the judge model and settings are provided in Supplementary Methods.",
    }
    for paragraph_id, required_text in required_plain_language.items():
        if required_text not in REVISED_PARAGRAPHS[paragraph_id]:
            raise AssertionError(f"Browser comment wording is missing from {paragraph_id}: {required_text}")
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
    reused_words, revised_words, reuse_ratio = introduction_content_word_reuse(data)
    print(f"Introduction physician-language content-word reuse: {reused_words}/{revised_words} ({reuse_ratio:.2%}).")
    print("Preserved 162/162 physician text blocks (127 paragraphs and 35 native-table cells) and 12/12 comments; rendered 4 main and 2 supplementary figures.")


if __name__ == "__main__":
    main()
