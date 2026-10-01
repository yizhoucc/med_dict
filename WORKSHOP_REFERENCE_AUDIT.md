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
| 6 | Dao et al., *JAMIA Open* 2025. [DOI](https://doi.org/10.1093/jamiaopen/ooaf097) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12410982/) | 220 development and 200 validation right-heart-catheterization notes; one pulmonary vascular disease expert; local Llama models; engineered preload, JSON/type/range validation, and retries | Supported. The related-work table now uses the paper's term “engineered preload.” |
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

## Literature update, 2026-09-26

The search was extended through 2026 using Europe PMC, Crossref, official publisher pages, ACL Anthology, NeurIPS proceedings, and arXiv metadata. Eleven references were added to the draft. The full synthesis and screening notes are in `WORKSHOP_LITERATURE_REVIEW.md`.

| Ref. | Identity and source | Main claim checked | Audit result |
|---:|---|---|---|
| 10 | Huang et al., *npj Digital Medicine* 2024. [DOI](https://doi.org/10.1038/s41746-024-01079-8) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063058/) | 1,026 acquired lung-cancer pathology reports, with 78 used for prompt development and 774 valid reports used for independent testing, plus 191 osteosarcoma reports; spiral prompt engineering; TNM and specialized terminology were error sources | Supported by Europe PMC metadata and full text |
| 11 | van Koevorden et al., *ESMO Real World Data and Digital Oncology* 2026. [DOI](https://doi.org/10.1016/j.esmorw.2026.100718) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13195338/) | 60 patients, 1,482 pages, 29 categories, two physician extractors, six clinical experts, 2,555 reviewed values, explicit error taxonomy and impact assessment | Supported by Europe PMC abstract and full text |
| 12 | Corso et al., *Communications Medicine* 2026. [DOI](https://doi.org/10.1038/s43856-026-01790-5) | Four local small language models; zero-shot, few-shot, and clinician-annotated few-shot prompting; clinical expertise improved consistency; multiclass TNM, PD-L1, and ECOG variables were difficult | Supported by Crossref and Europe PMC abstract; publisher DOI resolved with HTTP 200 |
| 13 | Abhyankar et al., *JCO Clinical Cancer Informatics* 2026. [DOI](https://doi.org/10.1200/CCI-25-00388) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42441924/) | 700 annotated notes used across development, validation, targeted error-pattern training, and testing; staging extraction across five cancers; more than two million notes and 217,768 patients in the deployment analysis | Supported by Crossref and PubMed/Europe PMC metadata and abstract |
| 14 | Passweg et al., *JCO Clinical Cancer Informatics* 2026. [DOI](https://doi.org/10.1200/CCI-26-00002) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13465685/) | 400 reports; five local models; duplicate human extraction and senior-oncologist adjudication; noninferiority for metastasis but inferiority for treatment response | Supported by Europe PMC abstract and full text |
| 15 | Dubey et al., *JCO Clinical Cancer Informatics* 2026. [DOI](https://doi.org/10.1200/CCI-25-00226) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/41678773/) | Rule-based, simple-prompt, chain-of-thought, and double-filtering ECOG extraction; advanced prompts reached 94% binary accuracy; all methods sometimes hallucinated | Supported by Europe PMC metadata and abstract |
| 16 | Zheng et al., NeurIPS 2023. [DOI](https://doi.org/10.52202/075280-2020) · [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) | Strong LLM judges showed more than 80% agreement with reported human preferences but also position, verbosity, self-enhancement, and reasoning biases | Supported by official proceedings BibTeX and arXiv abstract |
| 17 | Wang et al., ACL 2024. [DOI](https://doi.org/10.18653/v1/2024.acl-long.511) · [ACL Anthology](https://aclanthology.org/2024.acl-long.511/) | Candidate answer order changed LLM evaluator rankings; the paper proposed positional-bias calibration | Supported by ACL Anthology metadata and abstract |
| 18 | Gallifant et al., *Nature Medicine* 2025. [DOI](https://doi.org/10.1038/s41591-024-03425-5) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12104976/) | TRIPOD-LLM contains 19 main items and 50 subitems and emphasizes transparent model, evaluation, task-specific performance, and human-oversight reporting | Supported by Nature metadata and Europe PMC abstract/full text |
| 19 | Tam et al., *npj Digital Medicine* 2024. [DOI](https://doi.org/10.1038/s41746-024-01258-7) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11437138/) | Literature review of 142 healthcare LLM human-evaluation studies; QUEST framework covering planning, implementation and adjudication, scoring, and review | Supported by Europe PMC metadata, abstract, and full text |
| 20 | Estevez et al., *JCO Clinical Cancer Informatics* 2026. [DOI](https://doi.org/10.1200/CCI-25-00215) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13001894/) | VALID framework recommending variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication analyses | Supported by Europe PMC metadata, abstract, and full text |

All eleven new DOI links resolved to the intended publisher record on 2026-09-26. The ASCO landing pages returned HTTP 403 to the automated client after correct DOI redirection; Crossref and PubMed/Europe PMC independently confirmed the title, authors, year, journal, and DOI for each.

## Clinician-provided sources, 2026-09-30

Six sources embedded in the clinician coauthor's Word comments were checked against Crossref, DataCite, Europe PMC, or the PMC full text.

| Source | Verification result | Manuscript role |
|---|---|---|
| Kong HJ. *Healthcare Informatics Research*. 2019;25(1):1. DOI `10.4258/hir.2019.25.1.1` | Crossref confirms the citation. PMC `PMC6372467` contains the statement that about 80% of medical data remain unstructured and untapped. | Candidate Introduction citation for the broad unstructured-data estimate. The underlying estimate is broad and not oncology specific. |
| Wiest IC, Ferber D, Zhu J, et al. *npj Digital Medicine*. 2024;7:257. DOI `10.1038/s41746-024-01233-2` | Already verified as working reference 3. | Supports local open-weight clinical extraction and inference-time constraints. |
| Rule A, Bedrick S, Chiang MF, Hribar MR. *JAMA Network Open*. 2021;4(7):e2115334. DOI `10.1001/jamanetworkopen.2021.15334` | Crossref and Europe PMC confirm the identity. PMC `PMC8290305` reports nearly 3 million notes, a 60.1% increase in median length, and increased redundancy from 2009 to 2018. | Strong source for note length, redundancy, templates, and copy-forward content. |
| Huang J, Yang DM, Rong R, et al. *npj Digital Medicine*. 2024;7:106. DOI `10.1038/s41746-024-01079-8` | Already verified as working reference 10. | Supports iterative prompt refinement and oncology-specific extraction errors. |
| Hein D, Christie A, Holcomb M, et al. *npj Digital Medicine*. 2025;8:301. DOI `10.1038/s41746-025-01686-z` | Crossref and Europe PMC confirm the title, authors, journal, article number, DOI, PMID `40410408`, and PMC `PMC12102345`. | New close comparator for human-in-the-loop refinement, error ontology development, and avoiding case-specific rules. |
| Sushil M, Kennedy VE, Mandair D, et al. CORAL dataset, PhysioNet version 1.0. DOI `10.13026/v69y-xa45` | DataCite confirms the dataset title, six creators, PhysioNet publisher, 2024 publication year, and dataset type. | Dataset citation to accompany the existing CORAL article citation. |
