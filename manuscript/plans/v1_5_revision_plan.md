# v1.5 manuscript revision plan

Status: **implemented on 2026-10-31**.

## Scope implemented

- Clarified the five Introduction paragraphs while retaining the physician draft's structure and language.
- Added the study focus to P004; consolidated note complexity and current clinical context in P006; separated reliability from privacy and governance in P008; corrected the descriptions of fine-tuning and inference-time methods in P010; and stated the study objective once in P012.
- Implemented a complete block-level review redline against the visible v1.3 manuscript at commit `de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8`.
- Applied the comparison to Introduction, Methods, Supplementary Methods, all changed manuscript or supplement headings, table cells and captions, figure captions, and the P119 limitation. Results wording, data, figures, and statistics were not changed.
- Recorded deleted or consolidated v1.3 blocks in `v1_5_version_diff.md`; deleted wording is not inserted back into the manuscript.
- Updated the screen-only version interface to v1.5. The outer history stays closed, v1.5 is open after the parent is opened, and v1.4.1 and older entries stay closed.

## Review-baseline design

`manuscript/review_baselines/v1.3_visible_blocks.json` is the review authority. It was independently extracted from the committed v1.3 HTML, not from the canonical Word file. The file records the baseline commit, HTML Git blob, HTML SHA-256, and stable block ID to visible plain text and text SHA-256 mappings.

The renderer compares the current visible block text with this baseline. A changed or new in-scope block receives `data-review-changed-from="v1.3"` and the `revision-highlight` class. An unchanged block receives no yellow review class. A block moved without wording changes remains unhighlighted because comparison uses stable IDs rather than position.

## Author-confirmed study record retained

The following confirmations from 2026-10-31 remain the manuscript authority:

- The 40 expert-annotated note texts formed an untouched holdout for final development. Earlier generation or evaluation on those notes was performed while preparing the paper and did not inform workflow iteration.
- Breast-cancer development used all 100 development notes. The absence of a complete run manifest in the reviewed records does not change the author's study record.

Earlier conflicting project records remain documented in the v1.4 plan as an internal provenance note. This revision does not alter those confirmations.

## Exclusions

- No Results text, numerical result, statistical analysis, figure data, or figure-generating code was changed.
- The canonical Word source and its 127 paragraphs, 35 native-table cells, and 12 comments remain unchanged.
- The v1.5 change does not add an ablation claim or a model-swapping claim to the Introduction.
