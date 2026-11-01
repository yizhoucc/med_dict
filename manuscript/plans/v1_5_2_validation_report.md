# v1.5.2 token-level review validation

## Scope

This revision changes only review markup, review metadata, the version label,
and their regression tests. The 354 visible manuscript blocks are textually
identical to v1.5 commit `744986d9`.

## Why the Introduction looked fully yellow

The v1.5.1 renderer compared visible blocks with v1.3 and assigned the
`revision-highlight` class to the block element itself. There were 44 changed
blocks: 5 Introduction blocks, 15 Methods blocks, 2 Discussion blocks, 1 other
supplement block, and 21 Supplementary Methods blocks. Therefore, a small
grammar change in P004 or P006 colored the background of the entire paragraph.

The pre-change highlight sources were:

- 4 highlighted paragraphs inherited from the physician Word source (P060,
  P062, P063, and P115). These were retained only in the canonical audit and
  did not need a screen color.
- 44 v1.3-to-v1.5.1 changed block wrappers. These caused the large yellow
  rectangles.
- 31 registered grammar-revised paragraphs, plus 3 revised headings and 6
  revised native-table cells. Their prior block wrapper did not identify the
  changed words.
- 12 inline citation links. Their links were blue but inherited the yellow
  background of a highlighted paragraph.
- 2 visible physician-comment superscripts. They were neutral blue and remain
  outside review highlights.

## v1.5.2 rendering

- The renderer tokenizes the canonical physician text and the approved visible
  wording, then uses a sequence diff to mark only insertion-side and
  replacement-side tokens.
- Citations and their surrounding citation punctuation are excluded from the
  diff and remain ordinary blue links.
- Deleted source wording is stored in `data-review-deleted`; it does not create
  visible placeholders.
- Hover text records original and revised wording for automatic replacements.
- Wholly new passages use focused phrases so they do not become solid yellow
  sentences. This is used for the annotation, evaluation-interface, and
  base-model-family clarifications.
- The 44 changed blocks keep their v1.3 audit metadata but have transparent
  backgrounds.
- The result contains 242 review spans across 66 blocks: 143 replacements, 37
  insertions, and 62 focused phrases. The longest span is 79 characters.
- Supplementary Figure S1 remains unhighlighted. Supplementary Figure S2 uses
  two short highlighted phrases in its revised caption.
- Print and `print-export` modes remove every yellow background.

## Regression results

- Canonical renderer: 127/127 paragraphs, 35/35 Word table cells, and 12/12
  Word comments preserved.
- Deterministic review test: 354 visible blocks unchanged from `744986d9`; 120
  Results blocks retain SHA-256
  `8c05de43c1eaa3fda1f1429af90a1cf181d9a54330ddba3fc8ff62888659eb0c`.
- WebKit screen test: token spans are visible; P004 and P006 have sparse
  word-level marks; citations and comment markers are outside yellow spans.
- WebKit print test: 14 pages, 1,273,698 bytes, 6 figures, 8 tables, no retained
  review background, and no temporary PDF after the injected failure path.

