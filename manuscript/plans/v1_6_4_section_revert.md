# v1.6.4 Results and Discussion section revert

Date: 2026-11-01

## Scope

Results and Discussion physician prose now starts from the canonical Word
extraction. `RESULTS_DISCUSSION_OVERLAYS` is the machine readable whitelist for
the remaining differences. The renderer fails if a source phrase is absent,
appears an unexpected number of times, or if a visible physician block differs
from its canonical base plus the listed overlays.

Introduction and Methods are unchanged from commit `f7b0f34d`. The renderer
locks both their markup and ordered DOM text hashes.

## Results blocks

- P084 uses the canonical section heading with no overlay.
- P085 uses the canonical paragraph. Overlays format `2,160`, replace the
  hyphenated note and field phrase, retain the current Table 3, Table 4, and
  Figure 2 links, and apply only hyphen removal to the canonical
  `note-cluster robust 95% CI` phrase, yielding `note cluster robust 95% CI`.
- P086 uses the canonical paragraph. Overlays remove ordinary prose hyphens
  and retain the Supplementary Figure S1 link.
- P087 uses the canonical paragraph with no overlay.
- P092 uses the canonical paragraph. Overlays remove cancer name hyphens,
  restore the missing percent sign in `83.62%`, normalize P value spacing, and
  retain the Table 5 link.
- P095 uses the canonical paragraph with no prose overlay. The 77.08% to
  83.53% leave one physician out range is a separate technical insert named
  `loo-results`, linked to Supplementary Table S3. `agreement-results` and
  Figure 3 remain separate technical artifacts.
- P096 uses the exact canonical heading capitalization,
  `Performance across Clinical Extraction Fields`, with no overlay.
- P097 uses the canonical paragraph and retains the Figure 4 link.
- P100 uses the canonical paragraph. `evaluator-by-cancer` becomes `physician
  and cancer type`, and the paragraph retains the Supplementary Figure S1 link.

Tables 3 to 5 and Figures 2 to 4, including their captions, remain unchanged.

## Discussion blocks

- P107 uses the canonical section heading with no overlay.
- P109 uses the canonical paragraph. `single-prompt` becomes `one prompt`.
- P111 uses the canonical paragraph. `free-text` becomes `free text`. The
  Sushil, Huang, and Hein citations remain attached to their claims. The Hein
  description retains the external study's term `clinicians`.
- P113 uses the canonical paragraph. The overlays use physician for this
  study's roles, remove ordinary prose hyphens, and attach the Gershon and
  Williams citations to the LLM judge statement.
- P115 uses the canonical paragraph. It retains the Kruse and Kougia citations
  and uses physician for this study's reviewers. Comment 13 is anchored
  immediately after the sentence ending `appropriate regimen.`
- P117 uses the canonical paragraph. It retains the Supplementary Figure S2
  link, writes `tradeoffs` without a hyphen, and anchors comment 14 to the ER,
  PR, and HER2 sentence.
- P119 uses the canonical limitation paragraph. Only ordinary hyphen removal
  is applied. The five added limitations from the prior draft are absent, and
  the Williams citation is located in P113 rather than P119.

## Review display

Results and Discussion contain no manual focus highlights. Introduction,
Methods, and Supplementary Methods retain the 33 existing focus spans across
13 blocks.

## Validation targets

- Canonical audit remains 127 paragraphs, 35 native table cells, and 12 Word
  comments.
- Introduction markup SHA-256 is
  `482dc09b37bc4b893b8afee63958816c54f1d25a976a4db23636278ac8b68074`;
  ordered text SHA-256 is
  `765a1c0e59c3c6b883d9b89df95fbb737758eddef30a470fc5b421b979119e78`.
- Methods markup SHA-256 is
  `65c54db21c365bd475a834e33f51e8ace65332a81ed72e9912d8a7fa5be7b0af`;
  ordered text SHA-256 is
  `628d07037caf487000bb27b068fccc0cbd398d1d9cae74422daba48635aa31a0`.
- v1.6.4 Results text SHA-256 is
  `5aceff6ac9b1f9a78abf8bc322c19556a32cc0ba722359b7656e987b226965b3`;
  the current ordered numeric token SHA-256 is
  `2966552230c8eb68c13797418d36abf7abdd1e8269f753f5f49bf91a74aec5b4`.
  After normalizing the canonical word `seven` to `7`, the semantic ordered
  numeric token SHA-256 is
  `c63c2e9083b71313e519062257189f27373673692ddce5d1bab0e4a3e743ddd5`
  in both commit `f7b0f34d` and v1.6.4. This confirms that the authoritative
  Results numbers did not change during the prose reset.
- The renderer locks the SHA-256 values of all six SVG figures and all six
  print PNG fallbacks.
- Citation URLs, cross references, comment backlinks, 30 literature entries,
  and the print layout remain regression tested.

## Completed checks

- Renderer byte exact generation and the v1.6.4 deterministic validator pass.
- All 9 staged ablation unit tests pass.
- Native WKWebView screen and print checks pass. The bounded print output is
  14 pages and 1,277,334 bytes, with 6 figures and 8 tables kept within page
  boundaries. Fault injection leaves no temporary PDF.
- Results and Discussion contain zero focus highlights. Introduction and
  Methods match commit `f7b0f34d` at both markup and ordered DOM text level.
