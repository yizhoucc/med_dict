# Workshop / Poster Positioning Memo

Updated: 2026-08-27

## Working conclusion

The project already has enough technical and experimental context for a workshop paper or poster. The strongest story is not that oncology information extraction has no competitors. It is that recurring errors in longitudinal oncology notes can be converted into explicit, auditable inference-time controls around a frozen local model.

Recommended framing:

> A failure-mode-driven inference harness improves a frozen local open-weight model on prespecified, clinically important extraction categories from longitudinal oncology notes, especially active-treatment temporal disambiguation and stage/metastasis consistency.

The current physician review remains the confirmatory evaluation. LLM-assisted review is an internal quality screen before material is sent to physicians.

## Decisions frozen before the matched-baseline rerun

1. The primary comparison is the full pipeline (PL) versus a single-call baseline (BL) using the same Qwen2.5-32B-Instruct-AWQ model and the same target field contract.
2. BL remains a true baseline: one prompt and one model call, with no task decomposition, gates, hooks, dictionaries, retries, or post-processing.
3. The original baseline results are exploratory because its prompt omitted the general `Metastasis` field and defined `current_meds` more broadly than the evaluator.
4. The corrected baseline is implemented as `baseline_extraction.py --matched`, with its auditable contract in `prompts/matched_baseline_contract.yaml`.
5. The primary claim is category-level: PL should have a positive net advantage in each prespecified core category. This does not mean that BL cannot win an individual sample-field comparison.
6. WSL rerunning is pending and will begin only when the user confirms that the machine is available.

## Prespecified core questions

These categories are fixed before inspecting the matched-baseline results:

1. What anticancer treatment is the patient actively receiving now?
2. What is the cancer stage?
3. Is distant metastatic disease present, absent, or uncertain, and where?
4. What regional or overall metastatic involvement is supported?
5. How is the cancer currently responding to treatment?
6. What is the cancer type and receptor status? (Breast cancer only.)
7. What completed molecular or genetic results are documented?

The old, non-matched comparison had a positive aggregate PL advantage in all seven categories, but it included individual BL wins in response and metastasis. Those numbers must not be presented as the final matched ablation.

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

1. Run the 20 breast and 20 PDAC matched baselines on WSL.
2. Apply the same field-level comparison rubric to the new outputs.
3. Regenerate the PL-versus-BL figure and replace all legacy counts.
4. Incorporate the real physician scores when available.

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

After the rerun, if supported:

> Using the same frozen Qwen2.5-32B model and an identical target field contract, the inference harness achieved a positive net advantage over the single-pass baseline in all seven prespecified core clinical categories, although the baseline remained better on some individual sample-field comparisons.
