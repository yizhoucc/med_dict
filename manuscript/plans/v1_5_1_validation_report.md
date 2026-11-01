# v1.5.1 validation report

Validation date: 2026-10-31.

## Redline integrity

- Frozen baseline: v1.3 commit `de4fff87eae0dc00f4c4eb65ec2274a3c7108cd8`.
- Baseline visible blocks: **341**.
- Changed blocks: **44**. Highlighted blocks: **44**.
- Importance levels: **20 light, 12 medium, 12 strong**.
- The changed set equals the union of the three mutually exclusive importance sets.
- No unchanged block has a review-importance attribute or class.

## Content preservation

- All **354** visible manuscript blocks match v1.5 commit `744986d9` exactly.
- Results contain **120** visible blocks with SHA-256 `8c05de43c1eaa3fda1f1429af90a1cf181d9a54330ddba3fc8ff62888659eb0c`.
- Canonical preservation remains **127 physician paragraphs, 35 native-table cells, and 12 Word comments**.
- The rendered draft retains 2 visible physician comments, 30 literature entries, 6 figures, and 8 tables.

## Browser and print regression

- WKWebView checks the three level counts, three distinct screen colors, stable-block samples from every level, unchanged-block samples, version-history behavior, supplement behavior, and hash navigation.
- Print and print-export remove backgrounds and outlines from all three levels.
- The bounded PDF regression retains all manuscript and supplement content, checks six figures and eight tables for page-boundary violations, enforces the 190 MiB process cap, and removes the temporary PDF on success and injected failure.
- Real WKWebView result: **14 pages, 1,266,224 bytes**. All six figures and eight tables remained within page boundaries, and the injected post-write failure left no PDF behind.
