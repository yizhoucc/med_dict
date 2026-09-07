# Failure-Mode-Driven Inference Improves a Frozen Local LLM for Oncology Note Extraction

**Working workshop-paper draft — Version 0.1, September 2026**

**Authors:** [TODO]

**Affiliations:** [TODO]

> Draft status: The technical results and first oncologist ratings are included. Before submission, add the target venue and format, exact clinician-evaluated artifact identifiers, additional clinician results if available, final statistical analysis, and verified bibliography metadata.

## Abstract

Large language models can extract structured information from clinical notes, but single-pass prompting remains vulnerable to temporal confusion, unsupported inference, and clinically important category errors. These failures are especially consequential in longitudinal oncology notes, where current therapy must be separated from historical and planned treatment, regional nodal disease from distant metastasis, and suspected disease from confirmed disease. We developed a failure-mode-driven inference harness around a frozen, locally served Qwen2.5-32B-Instruct-AWQ model. The harness uses task decomposition, dependency-aware extraction, a five-stage verification cascade, deterministic oncology-specific corrections, and source attribution; model weights are not modified. We compared the harness with a single-prompt baseline using the same base model and target field contract on 40 held-out CORAL oncology notes (20 breast cancer and 20 pancreatic cancer). In a source-grounded, LLM-assisted audit of seven prespecified core categories, the harness was preferred in 66 comparisons, the baseline in 28, and 166 were ties. The harness led in six of seven categories. A subsequent targeted regression evaluation of repaired high-impact failure modes yielded 29 harness wins, no baseline wins, and 22 ties across 51 applicable comparisons. In a preliminary identity-masked evaluation of newer outputs by one oncologist on 20 breast cancer notes, the harness was preferred in 84 of 278 completed required-field comparisons, the baseline in 22, and 172 were ties. Among decisive comparisons, 79.2% favored the harness; the harness won more fields in 18 of 20 cases. The largest clinician-rated gains involved active anticancer medications and medication planning. These findings suggest that explicit inference-time controls can improve the clinical reliability of a frozen local model without fine-tuning. Additional oncologist ratings, pancreatic-cancer evaluation, and external validation are needed.

## 1. Introduction

Clinical information extraction is a promising use of large language models (LLMs), but a fluent answer is not necessarily a faithful one. Oncology notes are particularly difficult because they contain long treatment histories, repeated pathology and imaging results, uncertain findings, and plans that may be conditional or no longer active. A single note may mention several anticancer regimens, only one of which is current; both regional lymph-node involvement and possible distant disease; or a radiographic change that predates the current therapy. A model that recognizes the correct entities but assigns them to the wrong timepoint or clinical category can produce a plausible yet misleading structured record.

Fine-tuning is one way to adapt an LLM to these distinctions, but it requires labeled data, computational resources, and repeated retraining as clinical requirements change. An alternative is to keep the model weights frozen and place the model inside an explicit inference harness. In this approach, complex extraction is decomposed into narrower tasks, intermediate outputs are verified, recurring clinical failure modes are encoded as auditable rules, and extracted values remain linked to source evidence.

We developed such a harness for structured extraction from longitudinal breast and pancreatic cancer notes. The system was designed around four priorities: factual fidelity, coverage of important information, simple language, and understandable downstream communication. The present study focuses on extraction, because reliable structured facts are a prerequisite for safe patient-facing summaries and other downstream uses.

This work makes three contributions:

1. It describes a failure-mode-driven inference architecture for oncology extraction using a frozen, locally deployed open-weight model.
2. It compares the full harness with a same-model, single-prompt baseline using prespecified clinically important fields.
3. It reports preliminary identity-masked oncologist evaluation showing that the observed gains are clinically recognizable and are concentrated in the intended failure modes.

## 2. Methods

### 2.1 Dataset and evaluation subsets

We used the CORAL dataset of de-identified oncology notes from breast and pancreatic cancer care. System development used the unannotated development collection. The principal technical comparison used 40 held-out notes: 20 breast cancer notes and 20 pancreatic cancer notes. No model-weight training or fine-tuning was performed on these notes.

The prespecified core extraction categories were:

1. active anticancer medications;
2. cancer stage;
3. distant metastatic disease and involved sites;
4. regional or overall metastatic involvement;
5. current treatment response;
6. tumor type and ER/PR/HER2 receptor status for breast cancer; and
7. completed molecular or genetic results.

The clinician scoring instrument additionally included treatment and testing plans, supportive medications, procedures, imaging, laboratory plans, and recent treatment changes. Low-priority laboratory-summary and general-findings questions were optional and excluded from the primary clinician result.

### 2.2 Base model

Both systems used Qwen2.5-32B-Instruct-AWQ served locally through vLLM. The model weights were frozen. Local inference avoids sending note content to a third-party model endpoint and permits the complete extraction and verification workflow to run within an institution-controlled environment.

### 2.3 Single-prompt baseline

The baseline used one model call to extract the complete target schema from each note. It did not use task decomposition, verification gates, retries, dictionaries, deterministic clinical corrections, or post-processing. For the primary technical comparison, the baseline and harness used the same target field contract.

### 2.4 Inference harness

The harness decomposes extraction into independent and dependency-aware stages. Initial prompts extract visit context, cancer diagnosis, laboratory results, clinical findings, active medications, and recent treatment changes. Subsequent prompts receive selected earlier outputs when the target requires cross-field reasoning, such as treatment intent or response assessment. Plan fields are extracted primarily from the Assessment and Plan section.

Each model output then passes through a five-stage verification cascade:

1. repair invalid JSON formatting;
2. validate output keys against the requested schema;
3. improve specificity and semantic alignment;
4. remove clearly unsupported or contradictory values while conservatively retaining supported information; and
5. remove historical or completed events from future-plan fields.

A deterministic post-processing layer addresses recurrent, high-confidence clinical failure modes. Examples include separating regional nodes from distant metastases, preserving uncertainty for suspected disease, reconciling stage with metastatic status, distinguishing active anticancer therapy from supportive or historical medications, and preventing pretreatment progression from being reported as response to a newly started regimen. The system also returns source-attribution excerpts for extracted values.

### 2.5 Development process

Development followed an iterative error-analysis cycle. Candidate outputs were reviewed against the complete source note, errors were classified by severity and failure mode, and generalizable corrections were added through prompt changes, verification logic, or deterministic rules. High-impact changes were tested against the affected examples and additional previously correct controls. This process was intended to convert recurring errors into explicit, inspectable system behavior rather than case-specific answer memorization.

### 2.6 Technical evaluation

The complete matched comparison included 260 applicable sample-field decisions across the seven core categories. A source-grounded LLM-assisted audit read the complete source note and both outputs for every decision. Judgments were recorded as harness better, baseline better, or tie. Each reported harness loss and contested high-severity judgment was rechecked against the note. This evaluation served as a systematic development and error-analysis measure rather than a substitute for physician review.

After the complete matched evaluation, four high-impact errors were repaired. The affected samples and two clean controls were rerun, producing a targeted 51-comparison regression set. Because this was a targeted subset, it was analyzed separately and did not replace the complete 40-note result.

### 2.7 Preliminary oncologist evaluation

One oncologist evaluated newer harness and baseline outputs for all 20 breast cancer samples through an identity-masked A/B interface. The evaluator viewed the source note and both structured outputs and selected A better, B better, or tie for each field. System identity was not disclosed. Fourteen clinically important fields were required; two lower-priority fields were optional.

The export contained 278 of 280 expected required judgments. Two judgments were missing (`b12/genetic_plan` and `b19/recent_changes`). One extraneous `p8/supportive_meds` entry was excluded because the completed batch otherwise contained breast samples only. The raw export was retained unchanged.

The scoring template retained stale filenames after the displayed outputs were updated. The project owner confirmed that the clinician reviewed newer outputs. **Before submission, the exact PL and BL artifact identifiers or hashes must be inserted here: [TODO].**

### 2.8 Outcomes and analysis

The primary descriptive outcomes were counts of harness wins, baseline wins, and ties. We report all judgments and separately describe the proportion of harness preferences among non-tied comparisons. We also summarize, exploratorily, which system obtained more field-level wins within each sample. Because only one clinician has completed the current evaluation, we do not report inter-rater reliability or inferential statistics.

## 3. Results

### 3.1 Complete matched technical comparison

Across 260 applicable core comparisons, the harness was preferred 66 times, the baseline 28 times, and 166 comparisons were tied. The harness had a positive net advantage in six of seven categories. Stage was the only category with a negative margin in this frozen complete run.

| Core category | Harness | Baseline | Tie | Net harness advantage |
|---|---:|---:|---:|---:|
| Active anticancer treatment | 8 | 0 | 32 | +8 |
| Stage | 6 | 8 | 26 | −2 |
| Distant metastasis | 11 | 3 | 26 | +8 |
| Overall/regional metastasis | 14 | 4 | 22 | +10 |
| Treatment response | 13 | 6 | 21 | +7 |
| Breast cancer type/receptors | 8 | 5 | 7 | +3 |
| Completed molecular/genetic results | 6 | 2 | 32 | +4 |
| **Overall** | **66** | **28** | **166** | **+38** |

### 3.2 Targeted regression after high-impact repairs

Four high-impact failures identified in the complete comparison were addressed with conservative deterministic corrections. Across 51 applicable core comparisons in six affected samples and two controls, the repaired harness recorded 29 wins, no baseline wins, and 22 ties. No critical P0 error remained in this targeted set, and the controls did not develop a detected core regression. These targeted results demonstrate repairability but should not be interpreted as a new full-dataset estimate.

### 3.3 Preliminary oncologist preference

The oncologist completed 278 required field comparisons. The harness was preferred in 84 comparisons (30.2%), the baseline in 22 (7.9%), and 172 were ties (61.9%). Among the 106 comparisons with a directional preference, 84 (79.2%) favored the harness. In an exploratory sample-level aggregation, the harness had more field wins in 18 of 20 samples; one sample favored the baseline and one was tied.

Across the seven core clinical categories, the harness recorded 51 wins, the baseline 12, and 77 ties. Six categories favored the harness and one was even.

| Core category | Harness | Baseline | Tie | Net harness advantage |
|---|---:|---:|---:|---:|
| Active anticancer medications | 17 | 1 | 2 | +16 |
| Stage | 8 | 1 | 11 | +7 |
| Distant metastasis | 3 | 1 | 16 | +2 |
| Regional/overall metastasis | 9 | 1 | 10 | +8 |
| Treatment response | 3 | 1 | 16 | +2 |
| Tumor type and receptor status | 5 | 5 | 10 | 0 |
| Completed molecular/genetic results | 6 | 2 | 12 | +4 |
| **Overall** | **51** | **12** | **77** | **+39** |

The largest core advantage occurred in active anticancer medications. Among additional required fields, medication planning also showed a large separation (13 harness wins, no baseline wins, and seven ties). Laboratory planning was the only required field with a small negative margin (one harness win, two baseline wins, and 17 ties). Procedure planning and tumor type/receptor status were even.

### 3.4 Qualitative error pattern

The clinician supplied limited free-text comments, but one example illustrates the intended effect of the harness. For an imaging-plan field, the clinician preferred the harness because the baseline described completed results while failing to identify the planned PET/CT. This is a temporal and semantic-routing error: the extracted content may be medically related to imaging, yet it does not answer the question of what imaging remains planned.

More broadly, the distribution of preferences supports a selective-correction interpretation. Most field comparisons were ties because both systems shared the same base model and many notes did not trigger a difficult failure mode. When a difference was clinically meaningful, the direction favored the harness, particularly for active therapy, medication planning, stage, and metastatic involvement.

## 4. Discussion

### 4.1 Principal finding

A structured inference harness improved the clinical usefulness of extractions produced by a frozen local LLM. The improvement was observed in a complete same-model technical comparison and was also recognizable in preliminary oncologist scoring. The result does not depend on changing model weights. Instead, it arises from decomposing the task, checking intermediate outputs, and enforcing a limited set of clinically motivated invariants.

The oncologist result is especially informative because its strongest separation occurred in the categories the harness was designed to protect. Identifying active anticancer therapy requires more than medication-name recognition: the system must distinguish anticancer drugs from ordinary home medications, identify whether treatment is current rather than planned or stopped, and preserve regimen context. Similarly, stage and metastasis extraction require the model to distinguish regional nodes from distant spread and suspected findings from confirmed disease. These are precisely the distinctions that single-pass generation often compresses into plausible but incorrect summaries.

### 4.2 Why the high tie rate is expected

Ties constituted 61.9% of the oncologist's required-field judgments. This does not negate the harness effect. Both systems use the same capable base model, and many fields are straightforward enough that a single prompt succeeds. The intended role of the harness is not to rewrite every correct answer; it is to intervene when known failure modes arise. The combination of many ties with a nearly four-to-one harness advantage among directional preferences is therefore consistent with a selective safety and reliability layer.

### 4.3 Contribution relative to prior work

Prior studies have demonstrated zero-shot oncology extraction, local open-model deployment, hierarchical prompting, and hybrid rule-plus-LLM systems. The contribution here is narrower. We focus on longitudinal clinic notes and explicitly convert observed clinical failure modes into auditable inference-time controls around one frozen model. The resulting system exposes where corrections occur, retains source attribution, and permits rule-level regression testing without retraining.

This framing avoids claiming that local LLM extraction, staged prompting, or hybrid systems are themselves new. Instead, the novelty lies in the operational method: identify recurring clinically meaningful errors, encode conservative safeguards, test affected and clean cases, and preserve a traceable path from source note to structured output.

### 4.4 Implications

Reliable extraction is useful beyond patient letters. Structured oncology fields can support cohort identification, registry abstraction, prior-authorization preparation, clinical research workflows, and patient-facing explanation systems. Local deployment may also simplify institutional data-governance requirements, although deployment within a local environment does not by itself establish regulatory compliance or clinical safety.

The results also suggest a practical alternative when labeled data or retraining resources are limited. As the underlying model improves, a well-specified inference harness can preserve task definitions, verification logic, and clinical invariants while allowing the base model to be replaced. That portability remains a hypothesis until the harness is evaluated with additional model families.

## 5. Limitations

The study has several limitations. First, the clinician result currently comes from one oncologist and only the breast cancer subset. It demonstrates preliminary clinical preference but cannot establish inter-rater reliability or generalization to pancreatic cancer. Second, the clinician interface concealed system identity but used fixed A/B positions rather than randomized side assignment. The harness also displayed source attribution while the baseline did not; attribution is part of the deployed harness output, but this asymmetry may influence preference and should be isolated in a future evaluation. Third, two required clinician ratings were missing, although assigning both missing ratings to the baseline would still leave the harness ahead 84 to 24.

Fourth, the complete 40-note matched table represents v2.2, whereas the subsequent corrections were evaluated only on affected samples plus controls. A complete rerun is required before reporting full-dataset v2.3.x category totals. Fifth, development and technical review used repeated error inspection on a single public dataset, creating a risk that some rules capture dataset-specific conventions. External notes from another institution are needed. Finally, the current study evaluates extraction quality rather than whether patients understand or act appropriately on generated explanations.

## 6. Conclusion

A failure-mode-driven inference harness improved structured oncology-note extraction from a frozen, locally served open-weight LLM. In a matched 40-note technical comparison, the harness outperformed a single-prompt baseline overall and in six of seven core categories. A preliminary identity-masked oncologist evaluation of newer breast-cancer outputs showed a larger directional preference for the harness, including 84 versus 22 required-field wins and a sample-level advantage in 18 of 20 cases. The largest gains involved active anticancer therapy and medication planning, supporting the hypothesis that explicit temporal and semantic safeguards address clinically important weaknesses of single-pass prompting. Multi-rater, pancreatic-cancer, and external-dataset evaluation are the next steps.

## References — provisional

1. Sushil M, et al. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. *NEJM AI*. 2024. doi:10.1056/AIdbp2300110.
2. Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115.
3. Wiest IC, et al. Privacy-preserving large language models for structured medical information retrieval. *npj Digital Medicine*. 2024. doi:10.1038/s41746-024-01233-2.
4. Bhattarai S, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. *JAMIA Open*. 2024. doi:10.1093/jamiaopen/ooae060.
5. Dao [initials TODO], et al. Generative artificial intelligence for automated data extraction from unstructured medical text. *JAMIA Open*. 2025. doi:10.1093/jamiaopen/ooaf097.
6. Zhang [initials TODO], et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. *Communications Medicine*. 2025. doi:10.1038/s43856-025-01116-x.
7. Grothey [initials TODO], et al. Comprehensive testing of large language models for extraction of structured data in pathology. *Communications Medicine*. 2025. doi:10.1038/s43856-025-00808-8.
