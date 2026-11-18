# Physician figure structure preservation ledger

Version 2.2.1 uses the six archived physician SVGs as the layout masters. The
normalizer verifies each source SHA-256 before writing an output. It preserves
the source DOM order, visible text, geometry, transforms, markers, line styles,
rounded corners, nodes, and arrows. Only semantic paint, font-family
declarations, and root accessibility or export metadata may change.

## Figure 1

- Preserved the physician four-panel A through D layout and every source node.
- Preserved the dataset sequence, the breast cancer review loop, the PDAC
  adaptation path, and the ablation arrows.
- Mapped the full harness in Condition D to the focal harness blue so green
  remains reserved for physician review. No geometry or text changed.

## Figure 2

- Preserved both panels, the three solid stacked bars, pooled blue text, and
  the eight individual physician circles.
- Added no diamonds, lollipops, patterns, or hatching.

## Figure 3

- Preserved both matrices, all cells, labels, values, and the color scale.
- Retained the continuous light green to physician green scale.

## Figure 4

- Preserved the count table, seven solid stacked bars, and seven adjusted
  probability intervals.
- Added no patterns or hatching.

## Supplementary Figure S1

- Preserved the complete note margin layout, all 40 points, reference line,
  direct labels, and positional encoding.

## Supplementary Figure S2

- Preserved the cumulative A through D chain, three solid comparison bars,
  direct counts, and reported P values.
- Added no patterns or hatching.

The strict validator compares a source and output invariant hash that excludes
only the allowed paint, font-family, and root metadata attributes. It also
checks the original text sequence and the Figure 1 node count.

## Version 2.2.1 palette map

- Harness: `#2F6690`
- Baseline: `#B85C45`
- Tie: `#C2C5C9`
- Physician: `#4C8872`
- LLM review: `#766A91`
- Neutral stage: `#788692`
- Text: `#293741`
- Grid: `#E7EAEC`
- Figure 3 sequential scale: `#EDF5F1` to `#4C8872`

Light semantic fills use a fixed 90% white mix. The resulting colors are
`#EAF0F4`, `#F8EFEC`, `#EDF3F1`, `#F1F0F4`, and `#F2F3F4` for harness,
baseline, physician, LLM review, and neutral elements. Panel fill uses a fixed
96% white mix with neutral stage, which gives `#FAFAFB`.
