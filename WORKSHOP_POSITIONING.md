# Workshop / Poster Positioning Memo

Updated: 2026-08-27

## Working conclusion

The project already has enough technical and experimental context for a workshop paper or poster. The strongest story is not that oncology information extraction has no competitors. It is that recurring errors in longitudinal oncology notes can be converted into explicit, auditable inference-time controls around a frozen local model.

Recommended framing:

> A failure-mode-driven inference harness improves a frozen local open-weight model on prespecified, clinically important extraction categories from longitudinal oncology notes, especially active-treatment temporal disambiguation and stage/metastasis consistency.

The current physician review remains the confirmatory evaluation. LLM-assisted review is an internal quality screen before material is sent to physicians.

## Matched-baseline result

The 40-sample matched single-prompt baseline rerun is complete and has been manually reviewed against the source notes. Across 260 applicable sample-field comparisons:

| Core category | PL | BL | Tie | Net PL−BL |
|---|---:|---:|---:|---:|
| Active anticancer treatment | 7 | 2 | 31 | +5 |
| Stage | 17 | 4 | 19 | +13 |
| Distant metastasis | 30 | 1 | 9 | +29 |
| Overall/regional metastasis | 14 | 16 | 10 | −2 |
| Treatment response | 5 | 6 | 29 | −1 |
| Breast cancer type/receptors | 7 | 5 | 8 | +2 |
| Completed molecular/genetic results | 9 | 4 | 27 | +5 |
| **Overall** | **89** | **38** | **133** | **+51** |

The corrected result supports a strong overall advantage and positive net advantage in five of seven categories. It does **not** yet support the claim that PL wins every core category: the general `Metastasis` field and `response_assessment` remain slightly behind BL. The detailed audit is in `results/extraction_comparison/MATCHED_BASELINE_CORE_REVIEW.md`.

The review also exposed one contract inconsistency: the production PDAC prompt currently treats CA 19-9 non-secretor status as a genetic result, while the matched baseline contract limits that field to completed molecular/genetic tests. This should be aligned before the final ablation is reported.

## Decisions frozen for the matched-baseline rerun

1. The primary comparison is the full pipeline (PL) versus a single-call baseline (BL) using the same Qwen2.5-32B-Instruct-AWQ model and the same target field contract.
2. BL remains a true baseline: one prompt and one model call, with no task decomposition, gates, hooks, dictionaries, retries, or post-processing.
3. The original baseline results are exploratory because its prompt omitted the general `Metastasis` field and defined `current_meds` more broadly than the evaluator.
4. The corrected baseline is implemented as `baseline_extraction.py --matched`, with its auditable contract in `prompts/matched_baseline_contract.yaml`.
5. The primary claim is category-level: PL should have a positive net advantage in each prespecified core category. This does not mean that BL cannot win an individual sample-field comparison.
6. The matched run uses all 20 breast and 20 PDAC held-out samples.

## Prespecified core questions

These categories are fixed before inspecting the matched-baseline results:

1. What anticancer treatment is the patient actively receiving now?
2. What is the cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer currently responding to treatment?
6. What is the cancer type and receptor status? (Breast cancer only.)
7. What completed molecular or genetic results are documented?

The old, non-matched comparison had a positive aggregate PL advantage in all seven categories, but those numbers must not be presented as the final matched ablation. The matched audit currently shows two categories with small negative margins, which are concrete targets for the next pipeline revision.

## Where this project is strongest

### 1. Active-treatment temporal disambiguation

The pipeline distinguishes active anticancer therapy from ordinary home medications, supportive medications, stopped/completed treatments, and future plans. This is more clinically meaningful than simply recognizing drug names.

### 2. Oncology-specific semantic consistency

The harness explicitly handles distinctions that single-pass extraction often collapses:

- regional lymph nodes versus distant metastasis;
- suspected or pending disease versus confirmed disease;
- current treatment versus historical or planned treatment;
- stage, metastatic status, and treatment response across related fields.

### 3. Failure-mode-driven inference engineering

The model weights remain frozen. Adaptation occurs through task decomposition, verification, trimming, and narrow deterministic rules derived from recurring error patterns. This is the central engineering contribution.

### 4. Auditable corrections

Gate decisions and POST-hook triggers are logged, and important rules have CPU smoke tests. The contribution is the traceable conversion of clinical failure modes into regression-tested controls, not the raw number of hooks.

### 5. Real longitudinal oncology notes across two cancer domains

The proof of concept uses breast and pancreatic oncology progress notes, which require integration of pathology, imaging, treatment history, current therapy, and future plans. The defensible scope is “two oncology domains,” not broad clinical generalization.

## Closest related work

| Work | What it establishes | Remaining distinction for this project |
|---|---|---|
| Sushil et al., CORAL (2024), DOI: [10.1056/AIdbp2300110](https://doi.org/10.1056/AIdbp2300110) | Introduces the same expert-curated oncology-note dataset and evaluates zero-shot LLM inference. | This project studies inference-time workflow engineering on top of a frozen local model. |
| Chen et al. (2025), DOI: [10.2196/65984](https://doi.org/10.2196/65984) | Reviews 24 oncology information-extraction studies and shows that prompt-based methods are already an active area. | Supports the importance of the task but rules out a “no competitors” claim. |
| Wiest et al. (2024), DOI: [10.1038/s41746-024-01233-2](https://doi.org/10.1038/s41746-024-01233-2) | Demonstrates privacy-preserving structured extraction with locally deployed Llama models. | Local/open-weight deployment alone is not novel; oncology-specific controls are the differentiator. |
| Bhattarai et al. (2024), DOI: [10.1093/jamiaopen/ooae060](https://doi.org/10.1093/jamiaopen/ooae060) | Extracts cancer phenotype, stage, treatment, and progression using LLM and rule-based approaches. | The present work emphasizes a single frozen model plus explicit temporal and cross-field safeguards. |
| Tariq et al. (2025), DOI: [10.1200/CCI-25-00002](https://doi.org/10.1200/CCI-25-00002) | Uses a hybrid UMLS and fine-tuned LLM system for longitudinal breast cancer treatment pathways with external validation. | The present work requires no fine-tuning or labeled training set and covers a broader field contract. |
| Dao et al. (2025), DOI: [10.1093/jamiaopen/ooaf097](https://doi.org/10.1093/jamiaopen/ooaf097) | Uses local LLM inference with task serialization, schema validation, source validation, and retries. | The closest architectural comparison; the narrower distinction is oncology-specific failure-mode rules for longitudinal notes. |
| Zhang et al., mCODEGPT (2025), DOI: [10.1038/s43856-025-01116-x](https://doi.org/10.1038/s43856-025-01116-x) | Shows hierarchical prompting for zero-shot cancer information extraction. | Multi-stage prompting is not itself novel; real-note error control and deterministic clinical invariants are the useful distinction. |
| Grothey et al. (2025), DOI: [10.1038/s43856-025-00808-8](https://doi.org/10.1038/s43856-025-00808-8) | Compares open and proprietary models for structured oncology pathology extraction. | The present task uses heterogeneous longitudinal clinic notes rather than a narrow pathology template. |

These citations were checked against Crossref/Europe PMC/arXiv by a read-only Codex research pass; titles and claims should be rechecked when the manuscript bibliography is assembled.

## Claims to avoid

- “There are no competitors.”
- “This is the first local/open-weight oncology extraction system.”
- “This is the first multi-stage, hybrid, or verified LLM pipeline.”
- “The hooks guarantee safety” or “guarantee zero hallucinations.”
- “The harness is model-agnostic” before another model is tested.
- “The harness is the only variable” when referring to the legacy baseline.
- “There were no core-field losses.”
- “Pipeline exceeds GPT-4o” based on the older, unmatched letter comparison.

## Minimal remaining work before writing

Required:

1. Fix the general-metastasis evidence/state representation and rerun affected samples plus the required clean-sample regression set.
2. Fix response temporal/evidence selection and rerun affected samples plus the required clean-sample regression set.
3. Align the production and matched-baseline field semantics, especially current anticancer medication and CA 19-9 non-secretor handling.
4. Regenerate the PL-versus-BL figure and replace all legacy counts after the corrected rerun.
5. Incorporate the real physician scores when available.

Useful if time permits:

1. Show three representative before/after cases: active medication, stage/metastasis, and response.
2. Report errors as hallucination, omission, temporal error, and semantic misalignment, rather than only PL/BL preference.
3. Run a small component ablation: single prompt; decomposed prompts; plus gates; plus hooks. This is helpful but not required for a workshop poster.

## Suggested title

**Failure-Mode-Driven Inference for Structured Oncology Note Extraction**

Alternative:

**From Single Prompt to Auditable Extraction: A Local Inference Harness for Oncology Notes**

## Result wording templates

Before the rerun:

> Preliminary review against the original, non-matched baseline showed a positive aggregate advantage for the harness in each of seven prespecified core clinical categories. Because the original baseline did not use an identical field contract, the primary comparison is being repeated with a matched single-prompt baseline.

Current accurate wording:

> Using the same frozen Qwen2.5-32B model and target output schema, the inference harness achieved an overall 89–38 advantage over a single-pass baseline across 260 manually reviewed core-field comparisons, with 133 ties. The harness led in five of seven categories, most strongly in distant-metastasis status (+29) and stage (+13), while general metastasis (−2) and treatment response (−1) remained targets for refinement.
