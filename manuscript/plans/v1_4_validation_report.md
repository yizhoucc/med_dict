# v1.4 validation report

Date: 2026-10-31

## Scope

Implemented the approved v1.4 changes in the Introduction, Methods, Supplementary Methods, the related Limitations disclosure, and the screen-only version-history interface. Results text, numerical results, figures, statistical analyses, and other Discussion claims were not revised.

## Introduction language reuse

The Introduction remains five paragraphs.

1. Paragraph 1 retains the physician's clinical-LLM framing and makes EHR note extraction the upstream task.
2. Paragraph 2 retains the physician's description of unstructured, longitudinal, copied-forward oncology notes.
3. Paragraph 3 keeps extraction reliability and externally hosted privacy concerns as separate constraints. It retains the point that local open-weight deployment can avoid sending PHI to outside services and does not by itself solve extraction errors.
4. Paragraph 4 removes the unsupported implication that local or open-weight models are inherently smaller. It retains fine-tuning, prompt design, task decomposition, verification, and expert review, then narrows the gap to converting recurring feedback into a reproducible oncology-note workflow.
5. Paragraph 5 retains the physician's inference-harness definition and adds the approved development and evaluation summary. It states that the harness operates outside frozen model weights and is technically separable from the base model. It makes no model-swapping performance claim.

Reuse was measured over the five canonical physician paragraphs and their five revised counterparts. The check lowercases alphanumeric word tokens, excludes a fixed English function-word stoplist and numeric-only tokens, and counts multiset overlap with the revised text as the denominator. Citations are excluded.

- Reused physician content words: 309
- Revised Introduction content words: 387
- Reuse rate: 79.84%
- Required minimum: 70%
- Result: pass

The renderer enforces the 70% minimum on every build.

## Methods and supplement checks

- Study Design names both stages before their details and separates the exploratory ablation.
- Dataset names breast cancer and PDAC directly, records 100 development notes for each, and records a 20 plus 20 untouched holdout based on the author's 2026-10-31 confirmation.
- Baseline wording defines one application-level chat-completion request per note and makes no claim about internal forward passes.
- Inference Harness distinguishes Assessment and Plan locator failure from field-level evidence insufficiency. Locator failure skips only the plan branch. Full-note extraction continues.
- The main five-gate list is compressed. Ordered definitions, postprocessing, and attribution order appear in Supplementary Methods.
- The supplement labels the stage-insufficiency example as hypothetical and states that it is not a CORAL patient.
- Evaluator independence appears once in Clinical evaluation.
- Fixed A/B mapping and the attribution asymmetry are disclosed in Supplementary Methods and discussed in Limitations.
- ChatGPT, the local Qwen reviewer, and the Codex ablation judge have separate roles. The Codex configuration is `gpt-5.6-sol`, reasoning effort `xhigh`, service tier `default`.
- The clinical evaluation reports four physician evaluators.

## Renderer and interface checks

- Canonical audit representation preserved: 127 physician paragraphs, 35 native-table cells, and 12 Word comments.
- Visible physician comments remain 13 and 14.
- Verified citation registry and internal cross-references passed renderer validation.
- Version history is closed by default. When opened normally, v1.4 is open and older entries are closed. A deep link closes sibling entries and opens only its target.
- Version history remains excluded from print.
- Revision highlighting is limited to short new or substantively revised phrases rather than full paragraphs.

## Test results

- `python3 render_workshop_draft.py`: pass
- `python3 -m py_compile render_workshop_draft.py`: pass
- `git diff --check`: pass
- Bounded WebKit screen and print regression: pass
  - 14 pages
  - 1,272,666-byte temporary PDF
  - 6 figures and 8 tables
  - no page-boundary offenders
  - post-write fault cleanup passed and left no PDF

The WebKit verifier now constructs a self-contained in-memory page for testing. This avoids local-file sandbox failures while preserving the same rendered HTML and all six print images.

## v1.4.1 wording follow-up

Date: 2026-10-31

The follow-up simplified the main evaluation sentence to `A/B test interface`, standardized the current scientific text on `Ablation study`, added the ablation purpose to Methods, and changed Table 2 condition A to `one inference request per note`. The Supplementary Methods and P119 disclosures about fixed ordering, source-attribution asymmetry, cumulative configurations, and the single non-physician LLM judge remain intact. Results, numerical values, and conclusions were not changed.

The revision legend now says: `Selected highlights call attention to key new or substantively revised passages; they are not a complete redline.` Version history remains closed by default; v1.4.1 is the open current entry inside it, and v1.4 and older entries are closed.

Validation passed with 127 physician paragraphs, 35 native-table cells, 12 Word comments, 2 visible comments, 30 literature entries, 6 figures, and 8 tables. The renderer, Python compilation, 9 ablation unit tests, internal cross-references, unique IDs, real hash navigation, bounded WebKit print regression, and post-write cleanup checks passed. The temporary regression PDF had 14 pages and 1,266,508 bytes and was deleted after validation.
