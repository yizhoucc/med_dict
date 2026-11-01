# v1.5.1 highlight-importance plan

Status: **implemented on 2026-10-31**.

## Scope

- Keep all manuscript wording, Results, figures, tables, and citations identical to v1.5 commit `744986d9`.
- Retain the complete block-level redline against v1.3. Every one of the 44 changed blocks remains highlighted.
- Assign each changed block to exactly one review-importance level: light for minor edits, medium for clarifications, and strong for substantive changes.
- Use one yellow hue with three alpha levels: 0.12 for light, 0.27 for medium, and 0.50 for strong. Do not use bold text as an importance cue.
- Keep deletions in `v1_5_version_diff.md`; do not restore deleted wording to the manuscript.
- Remove all review-highlight backgrounds and outlines from normal print and bounded print-export output.

## Importance registry

The renderer owns the explicit stable-block registry. It assigns 20 light, 12 medium, and 12 strong blocks. Rendering fails if the registry does not exactly cover the 44 blocks changed from the frozen v1.3 baseline, if a block appears in more than one level, or if an unchanged block receives a level.

## Version interface

The outer version history remains closed. The v1.5.1 entry opens when the parent is opened; v1.5 and all older entries remain closed. The current entry records this as a visual review change only.
