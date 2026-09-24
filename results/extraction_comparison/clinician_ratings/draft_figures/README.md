# Draft clinician-rating figures

These SVG files are internal trend-check plots generated from the three current clinician-rating exports. They are intentionally plain and are not publication-ready figures.

The exploratory letter plot uses `../patient_letter_scores_oncologist_01_summary.csv`, a deidentified per-note summary derived from the original oncologist scoring workbook.

Generate them with:

```bash
python3 results/extraction_comparison/clinician_ratings/analyze_clinician_ratings.py
python3 results/extraction_comparison/clinician_ratings/plot_clinician_ratings.py
```

Files:

- `figure1_development_pathway_rough.svg`: breast-cancer development, pancreatic-cancer transfer, and independent clinical evaluation pathway.
- `figure2_evaluator_distribution_rough.svg`: outcome distribution for the five completed clinician-by-cancer evaluations.
- `figure3_interrater_matrix_rough.svg`: pairwise breast-cancer exact agreement and Cohen's kappa for the three oncologists.
- `figure4_core_fields_rough.svg`: pooled preference distribution across the seven core clinical categories.
- `figure5_note_margins_rough.svg`: note-level PL-minus-BL field margins for breast cancer and PDAC.
- `figure6_adjusted_effects_rough.svg`: adjusted overall and cancer-specific odds ratios from the directional-rating GEE.
- `figure7_human_model_complementarity_rough.svg`: conceptual map of human review burden and model error risk across extraction tasks.
- `figure7_letter_differences_rough.svg`: retained internal exploratory patient-letter plot; no longer embedded in the main manuscript.
- `supplementary_figure1_technical_vs_clinician_rough.svg`: descriptive comparison of technical-audit and clinician net preference rates.

The current manuscript embeds these SVGs as draft figures with formal captions. Regenerate them whenever another clinician export is added or the analysis specification changes.
