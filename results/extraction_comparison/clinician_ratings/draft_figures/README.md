# Draft clinician-rating figures

These SVG files are internal trend-check plots generated from the three current clinician-rating exports. They are intentionally plain and are not publication-ready figures.

The exploratory letter plot uses `../patient_letter_scores_oncologist_01_summary.csv`, a deidentified per-note summary derived from the original oncologist scoring workbook.

Generate them with:

```bash
python3 results/extraction_comparison/clinician_ratings/plot_clinician_ratings.py
```

Files:

- `figure1_development_pathway_rough.svg`: breast-cancer development, pancreatic-cancer transfer, and independent clinical evaluation pathway.
- `figure2_evaluator_distribution_rough.svg`: outcome distribution for the five completed clinician-by-cancer evaluations.
- `figure3_interrater_matrix_rough.svg`: pairwise breast-cancer exact agreement and Cohen's kappa for the three oncologists.
- `figure4_core_fields_rough.svg`: pooled preference distribution across the seven core clinical categories.
- `figure5_note_margins_rough.svg`: note-level PL-minus-BL field margins for breast cancer and PDAC.
- `figure7_letter_differences_rough.svg`: exploratory per-note patient-letter score differences versus ChatGPT and the same-model Qwen baseline.
- `supplementary_figure1_technical_vs_clinician_rough.svg`: descriptive comparison of technical-audit and clinician net preference rates.

The manuscript contains detailed placeholders for the final versions. Final figures should be regenerated after the clustered analysis is complete and whenever another clinician export is added.
