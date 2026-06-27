# Progress Report — Faithful Structured Extraction from Oncology Notes with a Local LLM

## Abstract

We are building a system that turns free-text oncology clinic notes into faithful, structured clinical fields (current anticancer medications, stage, metastasis, treatment response, receptor/molecular status, and care plans) so the information can be summarized for patients without hallucination or privacy leakage. Rather than fine-tuning, our approach is an **inference harness around a frozen, locally served open-weight model** (Qwen2.5-32B-Instruct-AWQ via vLLM): multi-stage prompting, a five-gate verification cascade, oncology drug/term dictionaries, and ~135 deterministic clinical post-processing rules — all on-premises and HIPAA-relevant. On 40 expert-annotated held-out notes (20 breast, 20 pancreatic), we ran a controlled ablation holding the base model and field schema fixed so the harness is the only variable. The harness (PL) beats the same model run from a single prompt (BL) on **131 of 618** field comparisons with **12** losses and the rest ties; among the fields where the two systems disagree, **PL wins 90%** and never loses a core diagnostic field. Physician review of an earlier patient-letter version flagged over-simplification, which led us to refocus the reporting period on the objectively-measurable extraction task and to prepare a blinded clinician scoring study.

## Accomplishments (this reporting period)

**Methods built.** A complete, reproducible extraction pipeline ("PL") over a frozen open model, with **no weight training by design** (keeps data local, needs no labeled corpus, stays reproducible). Adaptation is entirely inference-time: (i) multi-stage extraction (8 field prompts + a dependency-aware second pass + a plan-extraction stage); (ii) a five-gate per-field verification cascade (format → schema → specificity/semantic-alignment → faithfulness trim → temporal filter); (iii) two deterministic dictionaries (~158 oncology drugs; 18,739 medical terms at an 8th-grade reading level); (iv) ~135 clinical post-processing rules. Development used an evaluation-driven loop on 200 unannotated notes (~15 breast / ~18 pancreatic rounds), preferring generalizable clinical rules over test-set hard-coding.

**Key outcomes.**
- **Controlled ablation (40 held-out notes), PL vs single-prompt baseline (BL), same base model + schema:** PL better **131** / tie **475** / BL better **12** out of 618 field comparisons; among differing fields, **PL 90%** and **zero core-field losses** (Figure 1). The strongest separation is current-medication "anticancer vs non-cancer home-med" identification (**33 wins : 0 losses**); other PL-dominant high-value fields are cancer stage, distant metastasis, and molecular/genetic results.
- **Deployment safety (earlier 3-way letter comparison, 40 letters):** the naked model was **0/40 sendable** (45% leaked de-identification placeholders, 12.5% hallucination); inside the harness the same model reached **zero hallucination, zero leaked redaction, ~97.5% sendable, fully on-prem**, exceeding a cloud GPT-4o baseline on deployment metrics.
- **Held-out extraction accuracy (breast):** P0 (hallucination) = 0, P1 = 0; receptor status 93%, treatment-goal direction 100%.

**Impact.** The result shows that a frozen, open, *locally hosted* model — unusable for this task when prompted naively — becomes clinically reliable purely through an inference harness. This is directly relevant to clinical deployment, where data cannot leave the institution and model fine-tuning is often infeasible; the same recipe transfers to other note types without retraining.

**Cancer-relevant impact.** Every high-value field the harness fixes is oncology-specific and decision-relevant: correctly separating active anticancer regimens from home medications, assigning AJCC stage, distinguishing regional nodal involvement from distant metastasis, and surfacing molecular results (e.g., BRCA, MMR/MSI, CA19-9 non-secretor). Accurate, faithful extraction of these is the prerequisite for safe patient communication and for downstream cohort/registry use in breast and pancreatic cancer.

## Goals for the next reporting period

1. **Blinded clinician scoring:** run the prepared A/B-blinded scoring instrument (system identity hidden) with one or more oncologists to obtain an independent, quantitative PL-vs-BL win rate.
2. **Generalization audit:** confirm the ~135 rules encode general clinical logic (not test-set artifacts) and validate on fresh notes / additional cancer types.
3. **Determinism:** reduce greedy-decode run-to-run variation so final metrics are stable.
4. **Accuracy-first summarization:** reintroduce patient-facing output under an explicit "accuracy over simplification" constraint.
5. **Write-up:** consolidate the ablation + deployment results into a publishable "inference-harness for reliable extraction on a frozen local LLM" report.

## Figure

![Figure 1](results/extraction_comparison/figs/report_fig.png)

**Figure 1. Pipeline (PL) vs single-prompt baseline (BL) on 40 held-out oncology notes; the base model and field schema are identical, so the harness is the only variable.** **(a)** Per-field outcome (number of samples judged PL better / tie / BL better), sorted by PL advantage; counts are shown for PL wins (left) and BL wins (right). PL leads on every clinically deep field, most strongly on current-medication classification (33:0). **(b)** Restricting to the field-comparisons where the two systems actually disagree (n = 145), PL is better in 90% of cases, tie in 2, and BL better in 12; the 12 BL "wins" are all non-hallucination omissions in secondary plan/summary fields, with no core diagnostic loss. Field-by-field verdicts came from clinically-briefed reviewers reading each source note (natural-language judgment, no automated scoring).

## Discussion

Most fields tie because both systems share the same strong base model; the harness's measurable value concentrates precisely in the high-knowledge fields where a naive prompt is systematically wrong — exactly the fields a clinician cares about. The few BL "wins" are minor omissions, not correctness failures, and several were further reduced this period by a prompt change that pushes the model to *extract concrete facts rather than infer tidy summaries*. The main limitation is run-to-run nondeterminism from greedy decoding, which produces long-tail variation we now control with deterministic rules. The pivot from patient letters to extraction was driven by physician feedback and by the need for an objective, controlled metric; with extraction now demonstrably reliable, accuracy-constrained summarization is the natural next step.
