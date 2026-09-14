# Draft clinician-rating figures

These SVG files are internal trend-check plots generated from the two preserved clinician-rating CSV exports. They are intentionally plain and are not publication-ready figures.

Generate them with:

```bash
python3 results/extraction_comparison/clinician_ratings/plot_clinician_ratings.py
```

Files:

- `figure2_evaluator_distribution_rough.svg`: outcome distribution for each completed clinician-by-cancer evaluation.
- `figure3_interrater_matrix_rough.svg`: breast-cancer agreement matrix for the two oncologists.
- `figure4_core_fields_rough.svg`: pooled preference distribution across the seven core clinical categories.
- `figure5_note_margins_rough.svg`: note-level PL-minus-BL field margins for breast cancer and PDAC.
- `supplementary_figure1_technical_vs_clinician_rough.svg`: descriptive comparison of technical-audit and clinician net preference rates.

The manuscript contains detailed placeholders for the final versions. Final figures should be regenerated after the remaining oncologist ratings and the prespecified clustered analysis are complete.
