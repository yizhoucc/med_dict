# Workshop literature review

Review date: 2026-09-26

## Scope and method

This review updates the literature base for `WORKSHOP_PAPER_DRAFT.md`. It covers:

- LLM extraction from oncology notes and reports;
- local and open-weight models;
- prompt decomposition, validation, retries, rules, and hybrid systems;
- clinician involvement in development and final evaluation;
- LLM-as-a-judge limitations;
- reporting guidance relevant to a clinical LLM pilot.

The search used Europe PMC, Crossref, official publisher pages, ACL Anthology, NeurIPS proceedings, and arXiv metadata. Bibliographic records and factual claims were checked against abstracts or full text rather than search snippets. The nine references already in the draft had previously been audited in `WORKSHOP_REFERENCE_AUDIT.md`; this update rechecked their role and added eleven references, bringing the working bibliography to twenty.

## Main synthesis

### 1. The direct oncology extraction literature is now substantial

The manuscript should not imply that oncology LLM extraction lacks competitors or clinician involvement. Recent studies include large-scale staging extraction, local-model comparisons, prompt-strategy comparisons, and prospective or expert-reviewed validation. The defensible distinction is narrower: this study evaluates an integrated inference harness against a same-model single-prompt baseline on longitudinal oncology notes, with field-level oncologist preference judgments.

### 2. Prompt design and clinician knowledge have direct empirical support

Huang et al. used iterative prompt engineering for lung-cancer and osteosarcoma pathology extraction. Corso et al. compared zero-shot, few-shot, and clinician-annotated few-shot prompts across locally deployable models. Dubey et al. compared simple prompting, chain-of-thought, double filtering, and a rule-based ECOG extractor. These studies support the premise that inference design matters, but none evaluates the complete combination of routing, dependent prompts, verification gates, deterministic oncology rules, logging, and source attribution used here.

### 3. Some recent studies provide stronger absolute validation or scale

Van Koevorden et al. used two physician extractors and six clinical experts to evaluate 29 categories in head and neck oncology. Passweg et al. used duplicate human extraction and senior-oncologist adjudication for metastasis and response fields. Abhyankar et al. used 700 annotated notes across model development and testing, then applied the staging system to more than two million notes across five cancers. These studies exceed the present pilot in absolute correctness measurement, cohort size, or both. The current paper should treat its A/B outcome as relative clinician preference, not a substitute for accuracy or safety estimates.

### 4. Field difficulty is clinically meaningful

Passweg et al. found local LLMs noninferior to human accuracy for metastasis classification but inferior for treatment-response assessment. Corso et al. found multiclass TNM, PD-L1, and ECOG variables harder than binary features. Huang et al. identified TNM interpretation as a recurrent error source. These results support our finding that explicitly stated facts are often easy, while temporal and clinical interpretation remain difficult.

### 5. LLM-as-a-judge is appropriate for development, not as the clinical endpoint

Zheng et al. reported useful agreement between strong LLM judges and human preferences, alongside position, verbosity, self-enhancement, and reasoning biases. Wang et al. isolated positional bias and showed that changing answer order can change the evaluator's ranking. These findings support our current design choice: the LLM reviewer finds candidate errors during development, investigators decide what to change, and oncologists provide the final study outcome.

### 6. Human evaluation needs a stronger absolute-validation layer

Tam et al. reviewed 142 healthcare LLM human-evaluation studies and proposed QUEST, which organizes planning, implementation, adjudication, scoring, and review. Estevez et al. proposed VALID for LLM- or machine-learning-derived EHR data, emphasizing variable-level comparison with expert abstraction, internal consistency and plausibility checks, and replication. Together, these frameworks clarify the main limitation of our three-choice design: it measures relative preference, but does not determine whether either output is independently acceptable or correct.

### 7. The paper's reporting should remain explicit

TRIPOD-LLM calls for transparent reporting of model identity, prompt and evaluation procedures, human oversight, and task-specific performance. The current draft should preserve the details about the frozen model, benchmark-informed development, evaluator roles, fixed A/B mapping, source attribution, tie handling, and the limits of the directional odds ratio.

## References added to the main draft

| Ref. | Study | Verified contribution | Relation to this paper |
|---:|---|---|---|
| 10 | Huang et al., *npj Digital Medicine* 2024, [DOI](https://doi.org/10.1038/s41746-024-01079-8) | 1,026 lung-cancer pathology reports, including 78 for prompt development and 774 for independent testing, plus 191 pediatric osteosarcoma reports; iterative prompt engineering; TNM and specialist terminology remained error sources | Supports failure-mode-driven prompt refinement and the need for clinical safeguards |
| 11 | van Koevorden et al., *ESMO Real World Data and Digital Oncology* 2026, [DOI](https://doi.org/10.1016/j.esmorw.2026.100718) | 60 patients, 1,482 pages, 29 categories; two physician extractors; six clinical experts reviewed 2,555 values and classified errors; only 4 values were classified as hallucinations, but 68 errors were rated high impact | Closest clinical-validation comparator; stronger reference-based error classification, but no same-model workflow comparison |
| 12 | Corso et al., *Communications Medicine* 2026, [DOI](https://doi.org/10.1038/s43856-026-01790-5) | Four small local models; zero-shot, few-shot, and clinician-annotated few-shot prompts; Italian oncology EHRs; clinical expertise improved consistency | Direct support for clinician-informed inference design; our harness adds gates, hooks, and cross-field controls |
| 13 | Abhyankar et al., *JCO Clinical Cancer Informatics* 2026, [DOI](https://doi.org/10.1200/CCI-25-00388) | 700 annotated notes used across development, validation, targeted error-pattern training, and testing; five cancer types; more than two million notes and 217,768 patients in deployment analysis | Provides the scale comparator for staging extraction; narrower task and different training strategy |
| 14 | Passweg et al., *JCO Clinical Cancer Informatics* 2026, [DOI](https://doi.org/10.1200/CCI-26-00002) | 400 German oncology imaging reports; five local LLMs; duplicate human extraction and senior-oncologist adjudication; metastasis easier than response | Supports our field-difficulty interpretation and shows the value of absolute clinical reference labels |
| 15 | Dubey et al., *JCO Clinical Cancer Informatics* 2026, [DOI](https://doi.org/10.1200/CCI-25-00226) | Rule-based, simple-prompt, chain-of-thought, and double-filtering ECOG extraction; advanced prompts reached 94% accuracy but still hallucinated | Direct prompt-level ablation relevant to our missing component ablation |
| 16 | Zheng et al., NeurIPS 2023, [DOI](https://doi.org/10.52202/075280-2020) | LLM judges exceeded 80% agreement with human preferences in the reported benchmarks but showed position, verbosity, self-enhancement, and reasoning biases | Supports the LLM reviewer as a scalable development tool, not a final clinical evaluator |
| 17 | Wang et al., ACL 2024, [DOI](https://doi.org/10.18653/v1/2024.acl-long.511) | Demonstrated that candidate answer order can materially change LLM comparison judgments and proposed calibration | Supports conservative interpretation of automated pairwise review |
| 18 | Gallifant et al., *Nature Medicine* 2025, [DOI](https://doi.org/10.1038/s41591-024-03425-5) | TRIPOD-LLM checklist with 19 main items and 50 subitems for transparent LLM study reporting | Provides the reporting standard used to audit model, prompt, evaluator, and oversight descriptions |
| 19 | Tam et al., *npj Digital Medicine* 2024, [DOI](https://doi.org/10.1038/s41746-024-01258-7) | Review of 142 healthcare LLM human-evaluation studies; proposed QUEST across planning, implementation and adjudication, scoring, and review | Directly informs evaluator selection, rating dimensions, reliability assessment, and adjudication for follow-up evaluation |
| 20 | Estevez et al., *JCO Clinical Cancer Informatics* 2026, [DOI](https://doi.org/10.1200/CCI-25-00215) | VALID framework for variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication | Defines the absolute-validation layer that is missing from the present relative-preference endpoint |

## Additional studies screened but not added to the main reference list

These papers are relevant but overlap with stronger or more direct citations already selected:

- Teterycz et al. 2026, [DOI](https://doi.org/10.1016/j.esmorw.2026.100705): four small models and prompt-sensitive extraction from 302 Polish bone-sarcoma records; ensemble voting improved accuracy. Useful for an ensemble discussion, which is not central to this paper.
- Nachbar et al. 2026, [DOI](https://doi.org/10.1016/j.ctro.2026.101143): Llama-3.1-8B, chain-of-thought prompting, and post-processing populated radiation-oncology case-report forms; only 10 patients were used for independent testing.
- Guillot et al. 2026, [DOI](https://doi.org/10.1371/journal.pdig.0001426): zero-shot GPT-4 extracted CAR-T adverse events, dates, grades, and interventions from longitudinal notes; performance varied by attribute.
- Zou et al. 2026, [DOI](https://doi.org/10.1038/s41698-026-01521-y): local LLM extraction of 30 prognostic variables from head and neck cancer reports with three radiation oncologists forming a majority-vote reference.
- Kang et al. 2025, [DOI](https://doi.org/10.2196/73605): multicenter breast-cancer curation compared LLM processing with manual physician review and reported large reductions in physician time.
- Luo et al. 2024, [DOI](https://doi.org/10.1200/CCI.23.00258): fine-tuned open-source models extracted patient-centered outcomes across three institutions, while zero-shot and few-shot models performed poorly.

## Recommended Discussion structure

1. Start with CORAL and explain that this paper changes the inference process around a frozen model rather than comparing model families.
2. Use Huang, Corso, and Dubey to establish that prompt and workflow design change oncology extraction performance.
3. Use van Koevorden, Abhyankar, and Passweg to acknowledge stronger absolute validation and larger-scale studies.
4. State the narrower contribution: an integrated multi-field harness, a same-model baseline, and direct field-level oncologist preference.
5. Use Passweg and Corso to support the field-specific difficulty and human-model complementarity discussion.
6. Use Zheng and Wang to explain why LLM-as-a-judge remains a development tool.
7. Use QUEST and VALID to distinguish relative preference from absolute human evaluation and to specify the next evaluation design.
8. Use TRIPOD-LLM to justify transparent reporting and the next-study plan.

## Claim boundaries

- Do not claim that most oncology LLM papers lack physician involvement. Several recent studies use extensive clinician annotation or adjudication.
- Do not claim that prompting, verification, retries, local deployment, or hybrid rules are individually novel.
- Do not equate clinician preference with absolute correctness.
- Do not describe the three-choice A/B form as a complete human-evaluation framework; it omits absolute acceptability, adjudication, and formal reliability analysis.
- Do not describe pancreatic-cancer adaptation as zero-modification transfer.
- The defensible contribution is the evaluated integration of multiple inference-time controls around a frozen model, with direct oncologist comparison against a same-model single-prompt baseline.
