# Workshop manuscript figures

These are the canonical generated figures for manuscript version 2.2.0. The November 17 physician SVGs are the authoritative visual masters. `plot_manuscript_figures.py` invokes `physician_svg_normalizer.py`, which preserves their DOM order, text, geometry, markers, line styles, rounded nodes, arrows, and spatial layout. It changes only shared semantic colors, font-family declarations, and root accessibility or export metadata. The archived physician SVGs remain read-only.

Figures 2-4 use the four physician datasets. Supplementary Figure S2 reads the locked counts in `../../../staged_ablation_annotated_20261001_093433/JUDGE_CODEX_DECODED_ANALYSIS.md`.

Generate the physician-derived figures with:

```bash
.venv/bin/python results/extraction_comparison/clinician_ratings/analyze_clinician_ratings.py
.venv/bin/python results/extraction_comparison/clinician_ratings/plot_manuscript_figures.py
```

The analysis command also writes `adjusted_gee_results.csv`,
`leave_one_physician_out_gee.csv`, `core_field_adjusted_gee.csv`, and
`ADJUSTED_ANALYSIS.md`. The core-field CSV contains the standardized estimates
from the joint exploratory GEE used in Figure 4C. The leave-one-physician-out
CSV supplies Supplementary Table S3.

Files:

- `figure1_development_pathway_rough.svg`: the physician four-panel study datasets, breast development, PDAC adaptation, and cumulative technical ablation schematic.
- `figure2_evaluator_distribution_rough.svg`: physician solid stacked distributions plus pooled blue text and individual physician circles.
- `figure3_interrater_matrix_rough.svg`: paired exact agreement and Cohen's kappa for all four physicians, shown separately for breast cancer and PDAC.
- `figure4_effect_profile.svg`: raw count table, complete field distributions, and seven field-specific adjusted probabilities from one joint exploratory GEE.
- `supplementary_figure_s1_note_margins.svg`: compact note-level harness minus baseline margins with redundant position, marker, and color encodings.
- `supplementary_figure_s2_staged_ablation.svg`: cumulative A through D configurations and the three pooled adjacent comparisons across the same 40 notes.

The `rough` suffix remains on three legacy paths for compatibility. Across all figures, blue identifies the harness or focal result, orange identifies the baseline, gray identifies ties or neutral elements, green identifies physicians, and purple identifies LLM review. The normalization process never introduces hatching or new markers.

`PHYSICIAN_STRUCTURE_PRESERVATION_LEDGER.md` records the preserved structure and
the allowed normalization for each figure.

Each SVG has a correspondingly named `_print.png` fallback rendered from the
normalized SVG at 300 DPI and 2250-pixel width while preserving the physician
viewBox ratio. The HTML uses the SVG on
screen and the PNG only under `@media print`; this avoids Safari's excessive
inline-SVG print spooling without reducing screen clarity. The six PNGs total
less than 2 MiB, and the normalizer rejects any fallback larger than 5 MiB. SVG text remains editable,
and every SVG root includes explicit dimensions, a viewBox, and accessible title
and description metadata.

Normal Safari printing relies on `break-inside: avoid` for each figure and
table. The bounded WebKit regression in `tests/verify_workshop_print.sh` uses
fixed-height `createPDF` rectangles instead of Safari's print compositor. The
HTML's `preparePrintExport()` function therefore inserts transient, print-only
spacers before any figure or table that would cross one of those test
boundaries. It does not alter the screen document or normal Safari print
layout, and the verifier requires all six figures and eight tables to remain
within a single test page before it creates the PDF. Capture ends at the last
visible manuscript or supplementary element rather than the document box, so
trailing article padding cannot create an empty final page. A post-write fault
injection also verifies that every failure path removes its temporary PDF.
