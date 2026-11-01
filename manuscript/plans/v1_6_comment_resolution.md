# v1.6 browser comment resolution and validation

Date: 2026-10-31

## Comment resolution

1. Removed the revision legend, its swatches, and the light, medium, and strong
   review styles. The page now uses 44 manually selected focus spans in 14
   blocks. Complete change metadata remains separate from visible color.
2. Replaced P004 with the requested plain statement of the extraction task.
   Only `Reliable extraction` and `This study focuses on that extraction task`
   receive visible focus.
3. Replaced faithfulness language in the manuscript with plain descriptions of
   accuracy and support from the original note.
4. Rebuilt P008 around variable output, the cost of checking every field,
   external PHI processing, and errors that can remain in local models. The
   paragraph no longer says that humans have difficulty detecting errors.
5. Retained privacy, governance, HIPAA, and PHI as substantive concepts from
   the physician draft. HIPAA is not colored as a new concept.
6. Removed `recurring` and `recurrent` from the visible manuscript. Repeated
   errors are described directly where needed.
7. Rebuilt P010 to distinguish additional training from methods that keep model
   weights unchanged. Huang and Hein are described as pathology studies that
   used physician review to refine prompts and schemas and then applied the
   strategies or templates to other cancer domains. The gap is limited to the
   broader breast cancer to PDAC workflow tested here.
8. P022 now states that annotations were not provided to the model or used
   during development, did not determine the physician preference endpoint,
   and were not shown to physicians.
9. P025 now describes one application level chat completion request per note.
   It does not make a claim about internal forward passes.
10. Replaced `breast-cancer` with `breast cancer` in P060 and elsewhere in the
    manuscript narrative.
11. P061 now says `modifications specific to pancreatic cancer` and describes
    rules applied after extraction.
12. P063 uses `four physicians` rather than `evaluators`.
13. P073 now points to details of the judge model and settings in Supplementary
    Methods and does not call them exact.
14. Kept ablation configuration labels separate from the physician A/B
    interface. P073 refers to adjacent configuration pairs and configuration
    identities.
15. Completed a manual hyphen audit of narrative text, headings, tables,
    captions, and Supplementary Methods.
16. Removed avoidable prose colons. No colon remains in the visible manuscript
    blocks.
17. P063 states that system identities were not disclosed. It does not claim
    complete blinding. The fixed A and B mapping, harness only source
    attribution, and comparison of complete output packages remain disclosed.
18. P017 now gives the two development stages and independent physician
    evaluation in plain language. It states that no physician reviewed PDAC
    development outputs.
19. The Study Design paragraph stays concise. P057 to P061 retain the detailed
    breast cancer development and PDAC adaptation process.
20. P022 describes the holdout as a separate untouched set of 40 notes, with 20
    breast cancer notes and 20 PDAC notes, without a colon.
21. P022 uses `The development process used` for the 100 notes from each cancer
    type.
22. P028 uses `fields related to the plan`, `separate prompts for each field`,
    and `plan branch`.
23. P028 now states the complete fallback sequence. Regular expressions locate
    Assessment and Plan, the model can try up to three times, and failure skips
    only the plan branch while extraction from the complete note continues.
    Prompts instruct the model to leave unsupported fields empty.
24. The Introduction does not name Qwen. The full Qwen model identifier remains
    in Methods.

## Hyphen audit

No ordinary prose compound with a hyphen remains in the visible manuscript.
The remaining hyphens are limited to these justified forms:

- model identifiers `Qwen2.5-32B-Instruct-AWQ`, `Qwen2.5-32B`, and
  `gpt-5.6-sol`;
- numeric confidence interval ranges in tables;
- negative exponents in scientific notation.

Formal reference titles and URLs remain unchanged in the author literature
workspace, which is outside the manuscript body.

## Validation

- Canonical physician audit: 127 paragraphs, 35 native table cells, and 12 Word
  comments preserved exactly.
- Visible page: 354 stable blocks, 66 blocks changed from the v1.3 audit
  baseline, and 44 manual focus spans across 14 blocks.
- Comments and literature: 12 canonical comments, 2 visible unresolved
  comments, and 30 literature entries preserved.
- Introduction: P004, P006, P008, P010, and P012 reviewed individually. No
  paragraph level yellow wrapper is present. P006 has no visible focus span.
- Renderer: HTML and Markdown are byte exact after a second render.
- Experiment tooling: all 9 staged ablation unit tests pass in the project
  `.venv`.
- Results regression: 120 Results blocks with SHA-256
  `e8a0a9844ffa51292752c4aff018e489a31acd1027ae630f70024694efe1d5e6`.
- WebKit screen and print: 14 pages, 1,271,706 bytes, 6 figures, 8 tables, no
  figure or table split across a page boundary, no focus background in print,
  and no temporary PDF after the injected failure path.
