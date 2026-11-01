# v1.5 validation report

Validation date: 2026-10-31.

## Introduction

- Canonical physician-language content-word reuse: **302/355 (85.07%)**. The renderer uses a lowercase alphanumeric multiset comparison after removing citation tokens and common function words.
- Relative to v1.4.1 commit `17e09d5d`, the approved five-paragraph text contains **29 inserted words and 42 replacement-side words, for 71 inserted or replacement words**. This count uses case-normalized word tokens, removes citation tokens, and applies `difflib.SequenceMatcher` across the complete five-paragraph text. The earlier estimate of about 60 words used a different content-word or renderer-oriented counting convention; the final approved wording was kept unchanged.
- P004, P006, P008, P010, and P012 match the approved final text. Their external citation links remain clickable.

## Complete block-level redline

- Baseline: v1.3 commit `de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8`.
- Baseline HTML Git blob: `2599f8938f4af20c203ae394cc675760c3767824`.
- Baseline HTML SHA-256: `db7b8fb330cce451383bc9741e616aa17f6bc87d5b3da3afdf39a309d6ecbb47`.
- Baseline visible blocks: **341**.
- Current visible blocks: **354**.
- In-scope current blocks: **341**.
- Changed or new current blocks: **43**.
- Highlighted current blocks: **43**.
- Deleted standalone baseline blocks: **9**.
- Assertion result: changed block set equals highlighted block set. No unchanged block is highlighted, and no orphan review marker exists.

Changed and highlighted block counts by current section:

| Section | Blocks |
|---|---:|
| Introduction | 5 |
| Methods | 15 |
| Supplementary Methods | 21 |
| Supplementary material figure caption | 1 |
| Discussion limitation | 1 |

The nine deleted standalone blocks and their current disposition are listed in `v1_5_version_diff.md`.

## Preservation and regression results

- Canonical source preservation: **127 paragraphs + 35 native-table cells + 12 comments**.
- Visible physician comments: **2**. Literature workspace entries: **30**.
- Figures: **6**. Tables: **8**.
- Results visible-block count: **120**.
- Results visible-block SHA-256 at v1.4.1 and v1.5: `8c05de43c1eaa3fda1f1429af90a1cf181d9a54330ddba3fc8ff62888659eb0c` in both versions.
- Cross-references, unique IDs, external citation safety attributes, and baseline metadata assertions pass in the renderer.
- `tests/validate_workshop_v15.py` checks the baseline, changed/highlighted equality, Introduction diff metric, and exact Results hash.
- `tests/verify_workshop_print.swift` samples ten changed blocks and five unchanged blocks in a real WKWebView, checks the legend and version-history hash behavior, and verifies that print retains the text while removing the yellow background.
- Real WKWebView result: **14 pages, 1,266,224 bytes**, with all 6 figures and 8 tables bounded within pages. Post-write fault injection left no PDF behind.
