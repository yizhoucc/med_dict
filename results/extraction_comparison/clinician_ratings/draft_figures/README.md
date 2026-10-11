# Draft clinician-rating figures

These SVG files are internal trend-check and conceptual plots used by the current draft. Figures 2-6 use the four current clinician datasets; Bolun's breast and PDAC ratings are stored in separate source workbooks. The figures are intentionally plain and are not publication-ready artwork. The v0.18 manuscript embeds the eight figures listed below. Patient-letter findings remain a qualitative Discussion topic, so no letter plot is embedded. The former technical-audit comparison is also excluded because the final test-set LLM audit is not a reported result.

The staged-ablation plot uses the locked counts in `../../../staged_ablation_annotated_20261001_093433/JUDGE_CODEX_DECODED_ANALYSIS.md`. It is separate from the clinician-rating plotting scripts below.

Generate the clinician-derived figures with:

```bash
python3 results/extraction_comparison/clinician_ratings/analyze_clinician_ratings.py
python3 results/extraction_comparison/clinician_ratings/plot_clinician_ratings.py
```

Files:

- `figure1_development_pathway_rough.svg`: breast-cancer development, pancreatic-cancer transfer, and independent clinical evaluation pathway.
- `figure2_evaluator_distribution_rough.svg`: outcome distribution for the eight completed clinician-by-cancer evaluations.
- `figure3_interrater_matrix_rough.svg`: pairwise exact agreement and Cohen's kappa for all four oncologists, shown separately for breast cancer and PDAC.
- `figure4_core_fields_rough.svg`: pooled preference distribution across the seven core clinical categories.
- `figure5_note_margins_rough.svg`: note-level PL-minus-BL field margins for breast cancer and PDAC, each pooled across four oncologists.
- `figure6_adjusted_effects_rough.svg`: adjusted overall and cancer-specific odds ratios from the directional-rating GEE.
- `figure7_staged_ablation_rough.svg`: blinded A-B, B-C, and C-D technical-ablation judgments overall and by cancer.
- `figure7_human_model_complementarity_rough.svg`: conceptual map of human review burden and model error risk across extraction tasks, embedded as Figure 8.

The current manuscript embeds these SVGs as draft figures with formal captions. Regenerate them whenever another clinician export is added or the analysis specification changes.
