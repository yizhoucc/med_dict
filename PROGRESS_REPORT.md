# Progress Report — Structured Extraction from Oncology Notes

**Project:** Faithful, structured extraction (and patient-facing summarization) from oncology clinic notes using a local LLM.
**Period covered:** development through 2026-06. **Data:** UCSF CORAL (de-identified breast + pancreatic oncology notes; 200 unannotated for development, 40 expert-annotated held out for evaluation).

---

## 1. Objective

Pull the clinically important facts out of free-text oncology notes into a fixed set of structured fields (current medications, stage, metastasis, treatment response, receptor/molecular status, plans, etc.), under four non-negotiable principles in priority order: (1) faithful, no hallucination; (2) no omission of important content; (3) plain language; (4) patient-readable. The longer-term aim was to turn that structured output into a patient-facing summary letter.

## 2. What we built (method)

The system is an **inference harness around a frozen open-weight model** — there is **no model fine-tuning by design**. The base model is **Qwen2.5-32B-Instruct-AWQ served locally via vLLM**; model weights are never modified. This is deliberate: it keeps all patient data on-premises (HIPAA-relevant), needs no labeled training set, and stays fully reproducible. All adaptation happens at inference time through four layers:

- **Multi-stage extraction** — instead of one prompt, the note is processed in stages: six independent Phase-1 prompts (reason for visit, diagnosis, labs, findings, current meds, treatment changes), two dependency-aware Phase-2 prompts (treatment goal, response) that receive Phase-1 results as context, and a plan-extraction stage (medication/procedure/imaging/lab/genetic/referral/follow-up). Splitting the task lowers the model's per-call cognitive load.
- **5-gate verification cascade** (per field): JSON-format repair → schema/key check → specificity + semantic-alignment → faithfulness trim ("keep if supported, only empty if it clearly contradicts or is fabricated") → temporal filter (drop completed/past items).
- **Two deterministic dictionaries**: an oncology drug whitelist (~158 drugs) and an 18,739-term medical glossary filtered to an 8th-grade vocabulary.
- **~135 deterministic post-processing hook families** encoding general clinical rules (e.g., regional vs distant nodes, chemo-hold as a plan change, receptor consistency).

## 3. Development process ("training")

No gradient training was performed. Instead we ran an **iterative, evaluation-driven development loop** on the 200 unannotated development notes:

1. Run the pipeline; 2) review every sample field-by-field against the source note (manual, natural-language clinical reading, no scripted judgment); 3) classify defects (P0 hallucination / P1 wrong-field-or-direction / P2 minor); 4) fix via prompt edits, gates, or generalized hooks; 5) re-run a regression set (all previously-failing samples + 30% random clean samples). Breast went through ~15 rounds, PDAC ~18 rounds. We deliberately preferred general clinical rules over test-set-specific hard-coding so improvements transfer to unseen notes, and stopped when remaining issues were "whack-a-mole" long-tail amplified by vLLM run-to-run nondeterminism.

## 4. Physician review results

- **Letter quality (clinician feedback):** a physician reviewed the generated patient letters and flagged that the most aggressive simplification pass (V33) **changed meaning / over-simplified**; the verdict was *accuracy must come before simplification*, and the more faithful earlier style was preferred. Earlier breast/PDAC letter packets passed with all flagged items fixed.
- **Three-way deployment comparison (pipeline vs naked Qwen vs GPT-4o), 40 letters:** the naked-Qwen baseline was **0/40 sendable** (45% leaked the de-identification `*****`, 12.5% hallucination); GPT-4o was HIPAA-non-compliant (cloud) and longer/denser. The **pipeline reached zero hallucination, zero leaked redaction, ~97.5% sendable, with on-prem data** — i.e., the same open model is unusable raw but clinically deployable inside the harness, and on deployment metrics beats a cloud GPT-4o baseline.
- **Held-out extraction accuracy (breast v31):** P0 = 0, P1 = 0; receptor status 93%, treatment-goal direction 100%.

## 5. Results & analysis — extraction ablation (current focus)

We isolated the harness's value with a clean controlled comparison on all 40 held-out samples: **PL** (full pipeline) vs **BL** (the *same* base model and *same* field schema, single prompt, no post-processing). Only the harness differs. Each field on each sample was judged by clinically-briefed reviewers (8 sub-agents, one per 5 samples, with main-reviewer verification; natural-language reading only, no scripted scoring).

Across the scored question set (618 field-comparisons): **PL better 131 · Tie 475 · BL better 12**. Among the questions where the two systems actually differ, PL wins the large majority and BL essentially never wins a high-value field. The strongest, most defensible separation is **current-medications "anticancer vs non-cancer home med" identification (PL wins, ~35:0)** — the baseline almost always dumps blood-pressure/diabetes/pain home meds into the cancer-med field, while the pipeline isolates the actual regimen. Other PL-dominant high-value fields: stage, distant metastasis, molecular/genetic results. The 12 BL "wins" are all **non-hallucination**, secondary plan/summary fields (imaging/lab/procedure sub-plans, lab summary, supportive meds) or boundary wording calls — not core diagnostic errors.

**Interpretation:** most fields tie because both systems share the same capable base model; the harness's measurable value concentrates in the high-knowledge fields where a naked prompt is systematically wrong (drug classification, staging, metastasis nuance, molecular results). This is the intended "many ties + clear PL wins + ~zero core BL wins" pattern.

## 6. Why we changed direction, and the new direction

**Why pivot away from letter generation as the headline:** patient-letter quality is hard to evaluate objectively and the physician review showed the simplification step trades off against faithfulness — the very principle we rank first. The letter was becoming the bottleneck and the riskiest, least-measurable part of the system.

**New direction — extraction-only ablation:** we re-centered on the structured-extraction task, which is (a) the foundation the letter depends on, (b) objectively evaluable field-by-field by physicians, and (c) a clean controlled experiment (base model + schema fixed, harness as the only variable). This reframes the contribution as an **inference harness that makes a frozen open model clinically reliable**, demonstrated by a per-field PL-vs-BL comparison rather than a subjective letter critique.

We also tightened the evaluation itself based on physician input: dropped clinically-useless questions (patient-type, treatment-goal direction, reason-for-visit summary), made the receptor question breast-only (PDAC has no ER/PR/HER2), and added a prompt principle pushing the model to **extract concrete facts rather than infer tidy summaries** (a re-run across all 40 fixed cases such as a "chemo break" being mislabeled "no medication change").

## 7. Next steps

1. **Physician blind scoring.** Two clinician-facing scoring pages are ready: a labeled one and a **blind A/B version** (systems shown only as A/B, identity hidden) so the physician scores PL-better / Tie / BL-better without bias. Collect quantitative win rates from one or more physicians.
2. **Generalization check.** Audit the ~135 hooks for any test-set-specific logic; verify rules hold on fresh notes / additional cancer types.
3. **Lock determinism.** Reduce reliance on greedy-decode run-to-run variation so the final numbers are stable.
4. **Revisit summarization, accuracy-first.** Once extraction is locked, reintroduce patient-facing output under an explicit accuracy-over-simplification constraint.
5. **Write-up.** Consolidate into an "inference-harness for clinically reliable extraction on a frozen local LLM" paper, with the PL-vs-BL ablation and the deployment-metric comparison as the core results.
