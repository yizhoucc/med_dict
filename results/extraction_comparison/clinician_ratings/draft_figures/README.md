# Workshop manuscript figures

These vector figures use one publication layout and fixed semantic color system across the v0.22 draft. Figures 2-4 use the four current physician datasets; Bolun's breast and PDAC ratings are stored in separate source workbooks. Patient-letter findings remain a qualitative Discussion topic, so no letter plot is embedded. The final test-set LLM audit is not a reported result and is also excluded.

Figure 5 reads the locked counts in `../../../staged_ablation_annotated_20261001_093433/JUDGE_CODEX_DECODED_ANALYSIS.md`. The same script generates six main figures and one supplementary figure.

Generate the physician-derived figures with:

```bash
python3 results/extraction_comparison/clinician_ratings/analyze_clinician_ratings.py
python3 results/extraction_comparison/clinician_ratings/plot_manuscript_figures.py
```

Files:

- `figure1_development_pathway_rough.svg`: four-panel data-isolation, breast development, PDAC adaptation, and holdout-evaluation schematic.
- `figure2_evaluator_distribution_rough.svg`: raw outcome composition and directional harness share for all eight physician-by-cancer analyses.
- `figure3_interrater_matrix_rough.svg`: paired exact agreement and Cohen's kappa for all four physicians, shown separately for breast cancer and PDAC.
- `figure4_effect_profile.svg`: unified three-panel effect profile. Panel A shows seven core clinical categories, panel B shows adjusted probability estimates, and panel C shows the distribution of pooled margins across 40 notes without note identifiers.
- `figure5_staged_ablation.svg`: A-D stage strip and adjacent-stage ablation results overall and by cancer.
- `figure6_clinical_handoff.svg`: results-informed three-lane clinical handoff concept.
- `supplementary_figure_s1_note_margins.svg`: fully labeled note-level harness-minus-baseline margins, pooled across four physicians within cancer type.

The `rough` suffix remains only on the three unchanged legacy paths. The content is regenerated from source data and the locked ablation summary whenever physician exports or analysis specifications change. Across all figures, blue identifies the harness, orange the baseline, gray ties or neutral elements, green physicians, and purple LLM review.
