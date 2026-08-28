# Workshop / Poster Positioning Memo

Updated: 2026-08-28

## Working conclusion

The project already has enough technical and experimental context for a workshop paper or poster. The strongest story is not that oncology information extraction has no competitors. It is that recurring errors in longitudinal oncology notes can be converted into explicit, auditable inference-time controls around a frozen local model.

Recommended framing:

> A failure-mode-driven inference harness improves a frozen local open-weight model on prespecified, clinically important extraction categories from longitudinal oncology notes, especially active-treatment temporal disambiguation and stage/metastasis consistency.

The current physician review remains the confirmatory evaluation. LLM-assisted review is an internal quality screen before material is sent to physicians.

## Matched-baseline result

The current fully rerun and manually audited comparison is matched v2.2. Across 260 applicable sample-field comparisons:

| Core category | PL | BL | Tie | Net PL−BL |
|---|---:|---:|---:|---:|
| Active anticancer treatment | 8 | 0 | 32 | +8 |
| Stage | 6 | 8 | 26 | −2 |
| Distant metastasis | 11 | 3 | 26 | +8 |
| Overall/regional metastasis | 14 | 4 | 22 | +10 |
| Treatment response | 13 | 6 | 21 | +7 |
| Breast cancer type/receptors | 8 | 5 | 7 | +3 |
| Completed molecular/genetic results | 6 | 2 | 32 | +4 |
| **Overall** | **66** | **28** | **166** | **+38** |

The fully rerun v2.2 result supports a clear overall advantage and positive net advantage in six of seven categories. Stage is the only category still behind in that frozen table. The detailed sample-level audit is in `results/extraction_comparison/MATCHED_V22_REVIEW.md`.

The production and matched-baseline contracts now both include explicit CA 19-9 non-secretor status in completed molecular/biologic results.

## Targeted v2.3.x repair check

The four P0 failures found in v2.2 were repaired and rerun together with the required clean controls. Across the 51 applicable core comparisons in this 8-sample targeted set, PL scored **29 / 0 / 22** (PL / BL / tie), with **P0=0**. The rerun specifically verified:

- historical and recurrent breast receptor profiles no longer borrow unsupported HER2/PR values;
- suspicious bone disease pending biopsy remains suspected rather than confirmed Stage IV/M1;
- an explicit stable/good-control assessment overrides unsupported progression language;
- confirmed nonregional abdominal nodal recurrence remains consistent with current metastatic stage;
- completed MMR-intact/pMMR results are recovered with source attribution;
- the two controls did not develop a core regression.

This is targeted validation, not a replacement full-40 run. Use 66/28/166 as the formal complete-run table until the revised pipeline is rerun on all 40 samples. The targeted evidence supports the claim that the remaining high-impact errors are narrow, auditable, and repairable.

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

The old, non-matched comparison had a positive aggregate PL advantage in all seven categories, but those numbers must not be presented as the final matched ablation. The complete v2.2 audit has one negative category, Stage. The targeted v2.3.x repair resolves the reviewed Stage failures, but a full-40 rerun is still required before claiming every category is net-positive in the final table.

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

1. Rerun the revised pipeline on all 40 matched samples if the final poster will claim per-category v2.3.x totals; otherwise report the complete v2.2 table plus the targeted repair check separately.
2. Regenerate the PL-versus-BL figure using the chosen frozen result table and remove legacy 89/38 numbers.
3. Incorporate the real physician scores when available.

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

> Using the same frozen Qwen2.5-32B model and target output schema, the inference harness achieved an overall 66–28 advantage over a single-pass baseline across 260 manually reviewed core-field comparisons, with 166 ties. The harness led in six of seven categories; Stage remained slightly behind in the complete v2.2 run. A subsequent affected-sample-plus-control regression eliminated all four identified P0 failures and scored 29–0–22 on 51 applicable core comparisons, but has not yet been repeated across all 40 samples.
