# Workshop draft reference audit

Audit date: 2026-09-14

Scope: all nine references in `WORKSHOP_PAPER_DRAFT.md`, all DOI links in `WORKSHOP_POSITIONING.md`, and the factual descriptions attached to those citations.

## Bottom line

- All eight journal articles and the Qwen2.5 arXiv report are real.
- All eight DOI records resolve to the cited articles. The arXiv identifier resolves to the cited technical report.
- Crossref and Europe PMC agree on the journal titles, years, authors, and DOI metadata. Europe PMC provides a PubMed or PMC record for all eight journal articles.
- No cited article was flagged as retracted in the Europe PMC records checked on the audit date.
- The central literature claims are supported, including the 24-study count in Chen et al. and the two five-point Likert evaluations in its supplemental methods table.
- Three manuscript descriptions were tightened because the previous wording went beyond what the source explicitly stated: Bhattarai et al. did not explicitly describe disagreement adjudication; mCODEGPT used automated matching plus human validation rather than simply having reviewers check all matches; and the Grothey comparison now describes the narrower report type and 11 parameters without asserting that pathology reports are inherently more regular.

## Verification sources

For each journal article, bibliographic metadata were checked through the Crossref DOI API and the corresponding Europe PMC record. Methods, sample sizes, expert roles, and reported evaluation procedures were checked in the PMC full text or author manuscript. The claim about Likert ratings was checked directly in Chen et al., Multimedia Appendix 2. Qwen2.5 metadata were checked on the arXiv abstract page.

| Ref. | Identity and link | Claims checked against the source | Audit result |
|---:|---|---|---|
| 1 | Chen et al., *JMIR Cancer* 2025. [DOI](https://doi.org/10.2196/65984) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970800/) | 24 included studies; need for validation beyond the original domain and real-world integration; supplemental evaluation methods | Supported. Multimedia Appendix 2 contains 24 study entries. Only Fink 2023 and Lyu 2023 explicitly describe five-point Likert ratings, and both use radiology reports. The manuscript now identifies this as our audit of the supplemental table rather than wording it as a direct conclusion of the review authors. |
| 2 | Sushil et al., *NEJM AI* 2024. [DOI](https://doi.org/10.1056/AIdbp2300110) · [PMC author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007910/) | 40 expert-annotated notes; 20 breast and 20 pancreatic; GPT-4, GPT-3.5-turbo, and FLAN-UL2 zero-shot comparison; independent oncologist review on 10 notes per cancer type; omissions and hallucinations | Supported. The paper also reports an additional 100 notes per cancer type labeled automatically by GPT-4. The local CORAL release used by this project contains those 200 notes without expert annotation, so the draft now distinguishes the release contents from the paper's expert-annotated benchmark. |
| 3 | Wiest et al., *npj Digital Medicine* 2024. [DOI](https://doi.org/10.1038/s41746-024-01233-2) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11415382/) | 500 MIMIC-IV histories; five presence/absence features; three blinded medical experts with consensus ground truth; Llama 2 model-size and prompt comparisons; grammar-constrained JSON | Supported. “Mostly binary” was tightened to “five binary features.” |
| 4 | Bhattarai et al., *JAMIA Open* 2024. [DOI](https://doi.org/10.1093/jamiaopen/ooae060) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11221943/) | 13,646 clinical texts from 63 patients with NSCLC; four phenotype groups; two subject-matter experts; GPT, Flan-T5, Llama-3-8B, medspaCy, and scispaCy comparison | Sample, task, and model claims are supported. The paper states that two experts supplied gold-standard manual annotations, but the accessible text does not explicitly state that disagreements were adjudicated. That phrase was removed. |
| 5 | Tariq et al., *JCO Clinical Cancer Informatics* 2025. [DOI](https://doi.org/10.1200/CCI-25-00002) · [PMC author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12208650/) | 26,692 Mayo Clinic patients; 162 Stanford patients; UMLS parser plus fine-tuned question-answering model; cancer-registry labels; expert-curated concepts and codes; five treatment targets; external validation | Supported. “Five treatment pathways” was tightened to the paper's wording, “five treatment categories.” |
| 6 | Dao et al., *JAMIA Open* 2025. [DOI](https://doi.org/10.1093/jamiaopen/ooaf097) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12410982/) | 220 development and 200 validation right-heart-catheterization notes; one pulmonary vascular disease expert; local Llama models; engineered preload, JSON/type/source-text validation, and retries | Supported. The related-work table now uses the paper's term “engineered preload.” |
| 7 | Zhang et al., *Communications Medicine* 2025. [DOI](https://doi.org/10.1038/s43856-025-01116-x) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12528503/) | 1,000 synthetic oncology notes; 49 mCODE entities; hierarchical BFOP/2POP prompting versus a one-round baseline; automated matching and human validation | Supported after wording correction. The source says evaluation combined programmatic automation with human validation; it does not establish that reviewers manually checked every automated match. |
| 8 | Grothey et al., *Communications Medicine* 2025. [DOI](https://doi.org/10.1038/s43856-025-00808-8) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11958830/) | 579 prostatectomy pathology reports from 340 patients; German originals and English translations; 11 extraction parameters; trained doctoral student under attending-pathologist supervision; open/proprietary models, prompting, and quantization | Supported. The previous claim that pathology reports are “more regular” than longitudinal notes was an inference, not a directly measured result, and was replaced by the narrower factual comparison. |
| 9 | Qwen et al. Qwen2.5 Technical Report. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) | Report title, identifier, current-version year, Qwen2.5 family, instruction-tuned and quantized variants | Supported. The bibliography author was changed from the informal “Qwen Team” to the arXiv-listed group/lead-author form “Qwen, Yang A, Yang B, et al.” The current v2 citation record is dated 2025. The report supports the model-family citation; local AWQ serving through vLLM is a study implementation detail. |

## Link behavior

All nine DOI/arXiv links resolved to the intended record during this audit. Automated requests to the NEJM AI, Oxford Academic, and ASCO publisher landing pages returned HTTP 403 after DOI redirection because those sites block automated clients. This does not indicate a broken DOI: Crossref returned the matching record, and the corresponding PubMed/PMC pages were accessible with HTTP 200.

## Ablation and component-comparison audit, 2026-09-23

- **Wiest et al. [3]:** compared plain zero-shot, one-shot, definition-enhanced, and grammar-constrained prompting. This is a prompt-level component comparison, although not an ablation of a multi-stage oncology harness.
- **Tariq et al. [5]:** compared the complete UMLS-plus-fine-tuned-LLM system with zero-shot LLM, structured-code, and rule-based baselines. It did not remove the two hybrid phases one at a time.
- **Dao et al. [6]:** reported the errors detected and corrected by its validation and retry loop, including correction of 6 of 15 notes with initially detected errors. It did not report a full factorial ablation of the engineered preload, model, validation, and retry components.
- **Zhang et al., mCODEGPT [7]:** directly compared a single-step baseline with BFOP and 2POP hierarchical prompting. This is the closest explicit prompting ablation among the reviewed oncology studies.
- **Grothey et al. [8]:** compared five prompting strategies as well as model and quantization configurations. This is a broad configuration comparison rather than a removal study of one integrated pipeline.
- **CORAL [2] and Bhattarai et al. [4]:** primarily compared model families or complete methods and did not report a component ablation comparable to the one proposed for the present harness.

Conclusion: component comparisons are common enough that a staged technical ablation would strengthen the paper, but a full clinician-rated ablation is not standard across all related work. An LLM-reviewed ablation should be labeled as mechanistic technical evidence rather than clinical validation.

## Remaining boundary

The statement that we did not identify a field-level, identity-masked, practicing-oncologist comparison is limited to the 24 studies included in Chen et al.'s review and the closest additional studies listed in the draft. It is not a claim that no such study exists anywhere in the entire literature. The manuscript keeps this bounded wording.
