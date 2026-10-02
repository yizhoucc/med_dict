# Codex judge pilot: decoded result

## Scope

The independent Codex judge reviewed 12 blinded pairs: two breast and two PDAC samples from each adjacent ablation comparison. It did not read project instructions, prior analyses, manuscripts, or the private mapping. The mapping was opened only after all verdicts had been written.

## Decoded verdicts

| Comparison | Earlier variant wins | Later variant wins | Ties |
|---|---:|---:|---:|
| A single-prompt baseline vs B decomposition only | 2 | 1 | 1 |
| B decomposition only vs C decomposition plus verification | 0 | 3 | 1 |
| C decomposition plus verification vs D full inference harness | 0 | 3 | 1 |

By cancer:

| Cancer | A vs B | B vs C | C vs D |
|---|---|---|---|
| Breast | 1:1:0 | 0:2:0 | 0:2:0 |
| PDAC | 1:0:1 | 0:1:1 | 0:1:1 |

The ratios list earlier-variant wins, later-variant wins, and ties.

## Interpretation boundary

This calibration pilot suggested that field decomposition alone did not have a stable advantage, while verification and the full deterministic-hook layer might add value. It was used only to test the judging protocol.

These are not final ablation results. Each comparison contains only four judged pairs, selected as the first two samples from each cancer block. The complete blinded set contains 120 pairs. No statistical claim or manuscript result should be based on this pilot alone.

After the full-run rubric was finalized, the judge re-read all 12 pilot pairs while still blinded and changed three PDAC judgments to TIE. The final analysis therefore uses one consistent TIE standard across all 120 pairs.
