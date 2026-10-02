# Codex-judged staged ablation: decoded analysis

## Bottom line

The blinded technical evaluation supports the verification gates, but not a monotonic improvement across all four stages.

- A to B, field decomposition alone: no overall difference.
- B to C, adding verification: large and consistent advantage for C.
- C to D, adding deterministic filters and POST hooks: no overall advantage, with opposite directions in breast and PDAC.

This is secondary technical evidence from one Codex judge. It is not clinician validation.

## Blinded judging protocol

- 120 pairs: 40 A vs B, 40 B vs C, and 40 C vs D.
- Each comparison contains the same 20 breast and 20 PDAC held-out notes.
- The judge used only the note and two blinded outputs. It was explicitly barred from project instructions, manuscripts, prior results, Git history, and private mappings.
- Faithfulness was prioritized, followed by completeness. TIE was used when differences were minor, offsetting, or not clinically meaningful.
- The initial 12-pair calibration set was re-read under the final rubric before decoding. Three verdicts changed to TIE.
- Final blinded totals were output A 53, output B 42, and TIE 25. Confidence was high for 67 and medium for 53 judgments.

## Decoded results

| Comparison | Earlier wins | Later wins | Ties | Later win rate among non-ties | Exact sign-test p |
|---|---:|---:|---:|---:|---:|
| A baseline vs B decomposition | 17 | 17 | 6 | 50.0% | 1.000 |
| B decomposition vs C verification | 2 | 24 | 14 | 92.3% | 0.0000105 |
| C verification vs D full harness | 19 | 16 | 5 | 45.7% | 0.736 |

The exact two-sided sign tests exclude ties. P values are exploratory and unadjusted. The B vs C result remains significant after a Bonferroni correction across the three overall comparisons.

## Results by cancer

| Comparison | Cancer | Earlier wins | Later wins | Ties | Exact sign-test p |
|---|---|---:|---:|---:|---:|
| A vs B | Breast | 11 | 6 | 3 | 0.332 |
| A vs B | PDAC | 6 | 11 | 3 | 0.332 |
| B vs C | Breast | 0 | 12 | 8 | 0.000488 |
| B vs C | PDAC | 2 | 12 | 6 | 0.0129 |
| C vs D | Breast | 6 | 11 | 3 | 0.332 |
| C vs D | PDAC | 13 | 5 | 2 | 0.0963 |

For C vs D, breast favored D while PDAC favored C. A two-sided Fisher exact test on non-tie verdicts gives an exploratory cancer-by-stage difference of OR 4.77 and p = 0.0437. This nominal result does not survive broad multiple-testing correction and should be treated as a diagnostic signal, not a confirmatory interaction.

## What the judge found

The strong B to C gain was driven by better handling of source support, uncertainty, temporal status, response assessment, and plan-versus-history distinctions. The benefit appeared in both cancers.

The D layer showed a tradeoff. In breast, deterministic hooks more often improved stage, metastasis, and other faithfulness problems. In PDAC, D sometimes improved metastatic-site caution but frequently lost clinically relevant completeness, especially current medications, response assessment, and therapy plans. This explains why D helped breast but did not improve the combined result.

## Evidence boundary and next experiment

- One LLM judge cannot establish clinical correctness, absolute acceptability, or inter-rater reliability.
- Adjacent comparisons do not directly estimate D versus A and are not guaranteed to be transitive.
- The most informative next comparison is a separately randomized, blinded A versus D evaluation on the same 40 notes.
- Before rerunning D, the PDAC deterministic-hook layer should be audited for over-pruning of current medications, response assessment, and therapy-plan content.
