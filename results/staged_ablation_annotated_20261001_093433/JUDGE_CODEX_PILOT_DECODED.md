# Codex judge pilot: decoded result

## Scope

The independent Codex judge reviewed 12 blinded pairs: two breast and two PDAC samples from each adjacent ablation comparison. It did not read project instructions, prior analyses, manuscripts, or the private mapping. The mapping was opened only after all verdicts had been written.

## Decoded verdicts

| Comparison | Earlier variant wins | Later variant wins | Ties |
|---|---:|---:|---:|
| A single-prompt baseline vs B decomposition only | 2 | 2 | 0 |
| B decomposition only vs C decomposition plus verification | 1 | 3 | 0 |
| C decomposition plus verification vs D full inference harness | 0 | 4 | 0 |

By cancer:

| Cancer | A vs B | B vs C | C vs D |
|---|---|---|---|
| Breast | 1:1 | 0:2 | 0:2 |
| PDAC | 1:1 | 1:1 | 0:2 |

The ratios list earlier-variant wins first and later-variant wins second.

## Interpretation boundary

This calibration pilot suggests that field decomposition alone does not have a stable advantage, while verification and especially the full deterministic-hook layer may add value. The strongest pilot signal is D over C at 4:0.

These are not final ablation results. Each comparison contains only four judged pairs, selected as the first two samples from each cancer block. The complete blinded set contains 120 pairs. No statistical claim or manuscript result should be based on this pilot alone.

The judge returned no ties and used medium confidence for 8 of 12 verdicts. Before the full run, the same rubric should retain an explicit TIE option when differences are minor, offsetting, or not clinically meaningful.
