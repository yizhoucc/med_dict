# v1.5 review diff against visible v1.3

Baseline: visible manuscript in commit `de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8`.

The HTML shows current changed and new blocks in yellow. It does not restore deleted text. The following nine v1.3 blocks are absent as standalone blocks in v1.5 because their content was removed, consolidated, or moved.

| Stable block ID | Visible v1.3 text | v1.5 disposition |
|---|---|---|
| `P032` | JSON formatting repair when parsing fails. | Consolidated into the main Methods summary and the ordered gate definitions in Supplementary Methods. |
| `P033` | Schema validation to detect incorrect or leaked keys. | Consolidated into the main Methods summary and the ordered gate definitions in Supplementary Methods. |
| `P034` | Specificity and semantic alignment checks to identify vague or off-task answers. | Consolidated into the main Methods summary and the ordered gate definitions in Supplementary Methods. |
| `P035` | Faithfulness trimming to remove clearly unsupported or contradictory content while retaining supported information. | Consolidated into the main Methods summary and the ordered gate definitions in Supplementary Methods. |
| `P036` | Temporal filtering to remove completed or historical events from future-plan fields. | Consolidated into the main Methods summary and the ordered gate definitions in Supplementary Methods. |
| `revision-annotation-use` | The expert annotations for the holdout notes were not provided to either system, were not used to develop the inference harness, and were not used to calculate the primary physician-preference endpoint. | Merged into P022 and the Supplementary Methods holdout definition. |
| `revision-evaluation-package` | Output A was always the inference-harness output, and output B was always the single-prompt baseline output. Physicians were blinded to these system identities. Source attribution was available only in the inference-harness output, so the comparison evaluated the full system output packages rather than any isolated component. | Merged into P063, Supplementary Methods, and the P119 limitation. |
| `revision-model-family-limitation` | Both systems used the same base-model family, so this study did not evaluate whether the findings generalize across model families. | Merged into P119. |
| `workflow-cross-reference` | The study workflow is shown in Figure 1. | Consolidated into the current study-design ablation paragraph. |
