# Physician figure structure preservation ledger

Version 2.2.0 uses the six archived physician SVGs as the layout masters. The
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
