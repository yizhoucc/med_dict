# Workshop manuscript figures

These vector figures use one publication layout and fixed semantic color system across the English-only version 1.2.3 draft. Figures 2-4 use the four current physician datasets; Bolun's breast and PDAC ratings are stored in separate source workbooks. Patient-letter findings and the final test-set LLM audit are not reported as manuscript results.

Supplementary Figure S2 reads the locked counts in `../../../staged_ablation_annotated_20261001_093433/JUDGE_CODEX_DECODED_ANALYSIS.md`. The same script generates four main figures and two supplementary figures.

Generate the physician-derived figures with:

```bash
python3 results/extraction_comparison/clinician_ratings/analyze_clinician_ratings.py
python3 results/extraction_comparison/clinician_ratings/plot_manuscript_figures.py
```

The analysis command also writes `adjusted_gee_results.csv`,
`leave_one_physician_out_gee.csv`, `core_field_adjusted_gee.csv`, and
`ADJUSTED_ANALYSIS.md`. The core-field CSV contains the standardized estimates
from the joint exploratory GEE used in Figure 4B. The leave-one-physician-out
CSV supplies Supplementary Table S3.

Files:

- `figure1_development_pathway_rough.svg`: four-panel data-isolation, breast development, PDAC adaptation, and holdout-evaluation schematic.
- `figure2_evaluator_distribution_rough.svg`: raw outcome composition and directional harness share for all eight physician-by-cancer analyses.
- `figure3_interrater_matrix_rough.svg`: paired exact agreement and Cohen's kappa for all four physicians, shown separately for breast cancer and PDAC.
- `figure4_effect_profile.svg`: two-panel core-field result. Panel A retains raw harness, baseline, and tie counts. Panel B shows field-specific adjusted probabilities from one joint exploratory GEE.
- `supplementary_figure_s1_note_margins.svg`: the compact note-level harness-minus-baseline margin distribution previously drafted as main-figure Panel C. The prior detailed breast and PDAC note panels are not retained.
- `supplementary_figure_s2_staged_ablation.svg`: A-D stage strip and adjacent-stage ablation results overall and by cancer.

The `rough` suffix remains only on the three unchanged legacy paths. The content is regenerated from source data and the locked ablation summary whenever physician exports or analysis specifications change. Across all figures, blue identifies the harness, orange the baseline, gray ties or neutral elements, green physicians, and purple LLM review.

Each SVG has a correspondingly named `_print.png` fallback rendered from that
SVG at the same 1500-pixel width and aspect ratio. The HTML uses the SVG on
screen and the PNG only under `@media print`; this avoids Safari's excessive
inline-SVG print spooling without reducing screen clarity. The six PNGs total
less than 1 MiB, and the renderer rejects any fallback larger than 5 MiB or
with dimensions that do not match its source figure.

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
