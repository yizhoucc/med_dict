# v1.4 manuscript revision plan

Status: **approved by the author and implemented on 2026-10-31**.

The approved scope covers the Introduction, Methods, Supplementary Methods, the corresponding Limitations disclosure, the screen-only version-history interface, and their renderer and tests. Results, figures, statistics, and other Discussion claims remain outside this revision. The companion file `v1_4_proposed_text.diff` remains a pre-implementation review aid and is not an executable patch.

Author confirmations recorded on 2026-10-31:

- The 40 expert-annotated note texts formed an untouched holdout for final development. Earlier generation or evaluation on those notes was performed while preparing the paper and did not inform workflow iteration.
- Breast-cancer development used all 100 development notes. The absence of a complete run manifest in the reviewed records does not change the author's study record.

The earlier conflicting project records remain documented below as an internal provenance note. The manuscript follows the author's confirmed account.

## Editorial position

The user's overall reading of the Introduction is close, but the current argument combines several distinct claims. The revision should separate the clinical task, the note-level difficulty, the deployment constraints, the available engineering approaches, and this study's contribution.

The current v1.3 visible text is anchored to the physician-authored Word draft, but it is not a word-for-word copy. It already contains conservative grammar edits, verified citations, and technical insertions. A v1.4 restructuring would add substantive editorial input from the model-development authors, so the physician author should approve the revised scientific framing.

## Response to the 13 browser comments

1. **Version history interface.** Replace the always-visible current-change box with one top-level `Version history` disclosure that is closed by default. When opened, the current v1.4 entry is expanded and labeled `Current change`. Every older entry remains closed until selected. Each entry contains its date and a concise list of changes. Deep links may open the requested entry, but a normal page load should preserve the default-closed state.

2. **Authorship of the Introduction.** The present prose comes from the physician draft, with grammar, citation, and reporting edits in the visible manuscript. It therefore reflects physician authorship plus editorial input. The proposed v1.4 Introduction would reorganize the logic and should be shown as a proposed author revision, not as untouched physician text.

3. **Study Design.** Start with a one-sentence map of the two development stages. Define Stage 1 as physician-guided breast-cancer development. Define Stage 2 as pancreatic ductal adenocarcinoma (PDAC) adaptation with no physician review of pancreatic outputs. Then describe the independent physician evaluation. Move the exploratory ablation to a separate paragraph because it answers a technical question rather than describing development or clinical evaluation.

4. **Dataset.** Name the two cancer types directly instead of writing `each cancer type`. Remove PhysioNet credentialing from the main Methods because it is a data-access detail. Place it in Data availability or Supplementary Methods. Retain a concise statement that the existing CORAL annotations were not inputs to the model, were not used to calculate the primary physician-preference endpoint, and were not shown in the evaluation interface. The author confirmed on 2026-10-31 that the 40 note texts were an untouched holdout for final development.

5. **One model call per note.** Replace this phrase with an application-level description: the baseline submitted the complete note once through one prompt and requested the full schema in one response. State that the study did not measure or make claims about lower-level server execution.

6. **Inference Harness.** Explain that decomposition supports both field-specific reasoning and field-specific processing or checks. Begin with note segmentation and Assessment and Plan identification. Then describe field extraction, dependency-aware downstream fields, plan extraction, and verification. Replace `First` unless the paragraph also supplies an explicit sequence. Explain omission as a source-support rule: when the system cannot identify sufficient note evidence after the allowed attempts, it leaves the relevant field empty rather than inferring a value.

7. **Five-gate list.** Keep one compact sentence in the main Methods. Move the five item definitions and operational details to Supplementary Methods. Table 1 may summarize the sequence if the journal format benefits from a compact table, but the prose list should not interrupt the main narrative.

8. **Insufficient stage information example.** Add a clearly labeled hypothetical example in Supplementary Methods. A note may contain findings associated with advanced disease but still lack enough source-supported information to assign a formal stage. In that case, the system should return `not stated` or an empty value under the field contract instead of guessing a stage. The example must be hypothetical and must not be presented as a CORAL patient.

9. **Evaluator independence.** State evaluator independence once in the Clinical evaluation subsection. Delete the duplicate sentence from the breast-cancer refinement subsection.

10. **Fixed A/B positions.** Use one neutral disclosure in Methods: the implementation fixed A as the harness and B as the baseline, while evaluators were not told the system identities. Put interface details in the supplement. Discuss the possible effect in Limitations.

11. **Source attribution asymmetry.** Mention in Methods that source excerpts were available only as part of the harness output when present, so the study compared complete system outputs. Put the display details in the supplement. In Limitations, state that fixed ordering and unequal source-attribution features may have permitted pattern inference or influenced ratings. Do not assert the direction or size of the effect.

12. **`Secondary` ablation.** Rename the subsection and study component `Exploratory staged technical ablation`. This phrase states the evidentiary role and the staged design without an unexplained hierarchy label.

13. **ChatGPT versus Codex.** Keep the systems distinct. ChatGPT was used during PDAC development to organize the breast-derived error categories into a review rubric. The ablation used a separate blinded OpenAI LLM judge. The main text should say `blinded OpenAI LLM judge`. Supplementary Methods should report `Codex`, model `gpt-5.6-sol`, reasoning effort `xhigh`, and service tier `default`. Calling the ablation judge ChatGPT would be factually inaccurate.

## Introduction: interpretation and revision

### What the user's reading gets right

- Paragraph 1 introduces clinical LLM use and narrows the paper to extraction from EHR notes.
- Paragraph 2 describes why oncology notes are difficult inputs.
- Paragraph 3 identifies unreliable extraction and privacy or governance concerns.
- Paragraph 4 contrasts model retraining with inference-time workflow design and identifies the expert-feedback gap.
- Paragraph 5 defines the inference harness and states what the study did.

### What needs correction or sharper framing

- The argument is not that EHRs and LLMs must always be used together, or that LLMs would have no clinical role without EHRs. The paper studies one important application in which EHR note extraction is an upstream task.
- `Poor data quality` is too broad. The relevant problems are free text, dispersed information, temporal ambiguity, uncertainty, copied-forward content, and redundancy. The text should not imply that LLMs simply cannot handle EHRs.
- Model errors and external-service privacy are separate constraints. The current third paragraph repeats the performance point before switching to governance.
- Open-weight or locally deployable models are not inherently small. Model size may affect performance, but the draft does not need an unsupported comparison with closed models whose parameter counts may be undisclosed.
- Fine-tuning, prompt design, task decomposition, verification, and deterministic rules are different adaptation strategies. The gap is narrower: there is limited evidence on converting recurring expert feedback into a reproducible inference workflow for oncology-note extraction.

### Recommended five-paragraph structure

1. Clinical use case and the upstream extraction task.
2. Why longitudinal oncology notes are hard to extract reliably.
3. Fidelity and deployment constraints, with privacy treated separately from extraction quality.
4. Available adaptation strategies and the specific evidence gap.
5. Definition of the harness, the two-stage development path, and the two evaluations.

### Complete proposed English text

**Paragraph 1**

Large language models (LLMs) are increasingly used for clinical documentation, patient communication, and other language tasks. Many such applications begin by extracting relevant facts from electronic health record (EHR) notes before those facts can be summarized or reused. Reliable note extraction is therefore a practical requirement for applying LLMs to oncology care and research.

**Paragraph 2**

Although EHRs contain structured fields, much clinically relevant information appears in free-text narratives (Kong, 2019; Wiest et al., 2024). Oncology notes often combine diagnoses, prior and planned treatments, test results, and changing disease status across sections and time points. Uncertainty, copied-forward text, and redundant documentation make the current clinical state difficult to separate from historical information. Outpatient notes have also become longer and more redundant over time, in part because of templates and copied text (Rule et al., 2021).

**Paragraph 3**

LLMs can extract information from these notes, but fluent output may omit facts, confuse timelines, or add plausible statements that the note does not support. Commercial models have shown variable performance on clinical extraction tasks. Their use can also raise privacy and governance questions when protected health information is sent to externally hosted services. Local deployment keeps note processing within the institution, but it does not by itself prevent extraction errors.

**Paragraph 4**

Extraction performance depends on both the model and the workflow used at inference. Fine-tuning can adapt a model to a clinical task, but it requires suitable training data and additional development. Prompt design, task decomposition, verification, and deterministic clinical rules offer another route that does not change model weights. Prior work has used iterative prompt refinement and clinical review to improve structured extraction (Huang et al., 2024; Hein et al., 2025). Less is known about how to convert recurring expert feedback into a reproducible inference workflow for complex oncology notes.

**Paragraph 5**

We use the term inference harness for the workflow that surrounds a frozen LLM: the sequence of extraction steps, verification checks, and clinical rules applied at inference. We developed a locally deployable harness for oncology-note extraction, refined it first on breast-cancer notes with physician input, and then adapted it to pancreatic ductal adenocarcinoma notes through model-assisted review without physician review of pancreatic outputs. We compared the final harness with a single-prompt baseline using the same underlying model and field definitions, with system identities masked during physician evaluation. We also conducted an exploratory staged technical ablation to examine how successive harness components changed output quality.

## Methods revision map

### Study Design

Use four short paragraphs:

1. A map of Stage 1, Stage 2, and the independent evaluation.
2. Stage 1: physician-guided breast-cancer error review and regression-tested harness revisions.
3. Stage 2: PDAC adaptation with no physician review of pancreatic outputs. ChatGPT organized the prior error categories into a rubric, a rubric-guided Qwen reviewer compared outputs with source notes, and investigators selected and tested revisions.
4. Clinical evaluation, followed by a separate sentence that introduces the exploratory staged technical ablation.

Recommended opening:

> The study had two development stages, followed by an independent clinical evaluation. Stage 1 used physician-guided review of breast-cancer outputs. Stage 2 adapted the harness to pancreatic ductal adenocarcinoma without physician review of pancreatic outputs, using model-assisted review and investigator-directed regression testing.

### Dataset

Use the author-confirmed holdout description. On 2026-10-31, the author confirmed that the 40 note texts and their annotations were untouched during final development and that earlier paper-preparation runs did not inform iteration.

Move this sentence to Data availability or Supplementary Methods:

> CORAL access is governed by the PhysioNet credentialing and data-use process specified by the dataset authors.

### Baseline model and model development

Define the unit of comparison at the application layer:

> For each note, the baseline sent the complete note to the model in a single prompt and requested the full target schema in one response. This definition concerns the study's model request and does not make a claim about lower-level server execution.

Keep the shared model, decoding, field definitions, and output contract adjacent to this sentence so the fairness claim is easy to audit.

### Inference Harness

Use the following sequence:

1. Identify relevant note sections, including Assessment and Plan.
2. Extract independent clinical fields with field-specific prompts.
3. Pass selected earlier results into dependent fields such as treatment intent and response.
4. Extract plan fields from the source-supported plan section.
5. Apply structural repair, content verification, temporal filtering, and field-specific clinical rules.
6. Return source excerpts when available and omit a field when the source evidence remains insufficient.

Recommended main-text summary of the gates:

> Outputs then passed through structural checks for JSON and schema validity, followed by checks for semantic alignment, source faithfulness, and temporal consistency. Operational definitions for all five gates are provided in Supplementary Methods.

### Clinical evaluation

Describe the source note, the two outputs, the field-level choices, and the seven prespecified fields. Keep evaluator independence here and nowhere else. Use neutral interface disclosure:

> The implementation used a fixed display order, with the harness shown as A and the baseline as B, while evaluators were not told the system identities. Source excerpts were available only as part of the harness output when present, so the evaluation compared the complete outputs of the two systems. Full interface details are provided in Supplementary Methods.

### Exploratory staged technical ablation

Change the subsection title from `Staged Technical ablation` to `Exploratory staged technical ablation`. Replace `secondary technical ablation` with the same phrase throughout the manuscript.

Main text:

> We conducted an exploratory staged technical ablation on 20 breast-cancer and 20 pancreatic-cancer benchmark notes. A single blinded OpenAI LLM judge compared adjacent configurations using the source note and paired outputs.

Supplementary Methods:

> The ablation judge was Codex using `gpt-5.6-sol`, reasoning effort `xhigh`, and service tier `default`.

Do not call this judge ChatGPT. Reserve `ChatGPT` for its PDAC rubric role.

### Supplementary Methods

Move or add the following material:

- Definitions and order of the five verification gates.
- Assessment and Plan identification, the allowed attempts, source checking, and the omission rule.
- A hypothetical stage-insufficiency example:

  > Hypothetical example: a note may describe suspicious distant lesions or palliative treatment without documenting enough confirmed information to assign a formal cancer stage under the extraction rules. The system should return `not stated` or leave the stage field empty rather than infer a stage from incomplete evidence.

- The fixed A/B implementation and the source-attribution display difference.
- Separate descriptions of ChatGPT's PDAC rubric role, the Qwen PDAC reviewer, and the Codex ablation judge.
- Exact Codex configuration and the blinded comparison rubric.
- PhysioNet access requirements and the statement that the CORAL expert annotations were not shown in the primary evaluation interface.

### Limitations

Add four explicit boundaries:

1. The primary endpoint was preference, not absolute extraction accuracy.
2. The fixed A/B order and source-attribution asymmetry may have allowed pattern inference or influenced ratings. The study cannot determine the direction or magnitude of that effect.
3. The ablation used one non-physician LLM judge and cumulative stages, so it cannot isolate each component's causal contribution.
4. Both systems used the same base-model family. The results estimate the effect of this harness within that setting and do not establish generalization to other model families.

The author confirmed the untouched-holdout description on 2026-10-31. The internal provenance note below records why that wording required explicit confirmation.

## Version-history UI proposal

The HTML implementation should follow this disclosure hierarchy when the author approves v1.4:

```text
Version history                         [closed by default]
  v1.4, Current change                  [open after parent opens]
    Introduction logic reorganized
    Methods roles and evaluation design clarified
    Gate details moved to Supplementary Methods
  v1.3, 2026-10-30                      [closed]
  v1.2.3, 2026-10-27                    [closed]
  ...
```

Implementation requirements for a later execution task:

- Use semantic nested `details` and `summary` elements.
- Keep the top-level disclosure closed on ordinary page load.
- When the top-level disclosure is opened, show the current version expanded and all older versions collapsed.
- Allow each older version to open independently.
- Preserve anchor behavior by opening only the entry targeted by the URL fragment.
- Keep the author-only history out of the print manuscript, as in v1.3.
- Summarize substantive manuscript changes. Do not mix renderer maintenance with scientific revision unless the maintenance affected what authors reviewed.

## Author confirmations and internal provenance note

Both conflicts were resolved by author confirmation on 2026-10-31. The following record is retained internally so the decision remains auditable.

### 1. Forty benchmark note texts and the word `untouched` (author-confirmed)

The current manuscript calls the 40 expert-annotated notes an `untouched holdout test set`. Project records dated August 27, 2026 state that the same 40 note texts had already been used in multiple rounds of error analysis, hook or prompt revision, and full reruns. Even if the expert annotation files were never used, use of the note texts for iterative development conflicts with `untouched` and `held out from development`.

Earlier provenance questions retained for audit:

- Were any of the 40 benchmark note texts viewed during error analysis before the final harness was frozen?
- Were changes to prompts, gates, or deterministic rules made after viewing outputs on those notes?
- What were the freeze date and commit for the harness used in the physician evaluation?
- Were the physician-evaluation outputs generated before or after any benchmark-driven revisions?

Author decision, 2026-10-31: the 40 note texts were an untouched holdout for final development. Earlier generation or evaluation was done to prepare the paper and did not inform prompt, gate, hook, or workflow iteration. The manuscript therefore uses `untouched holdout`. The conflicting project-record summary above remains an internal provenance note.

### 2. Fifty-six selected breast-cancer notes versus the claim of all 100 (author-confirmed)

The current Methods says that breast-cancer development used all 100 development notes and that major and final runs covered all 100. Project records for the v31 development cycle document detailed runs and review on 56 selected breast-cancer notes. The available evidence does not establish the current `all 100` claim for the final development cycle.

Earlier provenance questions retained for audit:

- Which breast-cancer note IDs were used in each development iteration?
- Is there a manifest or result artifact showing a complete 100-note major or final run for the harness described in the manuscript?
- Did the physician review all 100 outputs, 56 selected outputs, or another subset?
- Should Methods distinguish notes processed by the system from notes reviewed by the physician?

Author decision, 2026-10-31: breast-cancer development used all 100 development notes. The manuscript retains 100. The incomplete manifest noted above remains an internal provenance note.

## Approval gates and completion status

1. Completed: the author approved the five-paragraph Introduction and requested minimal reuse-focused editing.
2. Completed: the author resolved both provenance conflicts by explicit confirmation dated 2026-10-31.
3. Completed: the author selected the untouched-holdout description.
4. Completed: fixed ordering and attribution asymmetry are disclosed briefly in main Methods, fully in Supplementary Methods, and discussed in Limitations.
5. Completed: ChatGPT, the local Qwen reviewer, and the Codex ablation judge are reported as separate roles.
6. Completed: the renderer, generated manuscript, supplement, tests, and version history were updated within the approved scope.
