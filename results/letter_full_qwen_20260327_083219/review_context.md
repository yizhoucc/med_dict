# Letter Review Context (for context recovery after compaction)

## What we're doing
逐个审查 61 封 patient letter 的质量。每封信必须：
1. 用 Read 工具读 results.txt 中的原文 (note_text)、keypoints、attribution、letter、traceability
2. 逐句对照原文和 keypoints，检查：
   - **准确性**: letter 内容是否忠实于 keypoints 和原文
   - **归因**: [source:field] 标签是否指向正确的字段
   - **通俗性**: 8th-grade English，医学术语是否解释
   - **完整性**: 重要字段是否覆盖
   - **冗余**: 是否有重复句子
3. 记录问题到 review.md，分 P0/P1/P2

## Severity
- P0: 幻觉 — letter 说了 keypoints/原文没有的东西
- P1: 显著错误 — 错误归因、重复句子、误导信息、不准确的疾病描述
- P2: 小问题 — 术语没解释、轻微冗余、遗漏次要信息

## Files
- results.txt: `/Users/yizhoucc/repo/med_dict/results/letter_full_qwen_20260327_083219/results.txt`
- review.md: `/Users/yizhoucc/repo/med_dict/results/letter_full_qwen_20260327_083219/review.md`
- 本文件: review_context.md

## Progress
- 61 rows total, indices: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 19, 21, 26, 28, 29, 32, 33, 35, 36, 39, 40, 41, 42, 43, 45, 48, 49, 51, 52, 53, 56, 58, 60, 62, 63, 64, 65, 67, 69, 71, 72, 77, 79, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 93, 94, 96, 99]
- ROW N in results.txt = row_index N-1 (ROW 1 = index 0, ROW 2 = index 1, etc.)
- Reviewed: ALL 61 rows (61/61 DONE)
- Review complete!

## How to find rows in results.txt
Use Grep to find "RESULTS FOR ROW N" where N = row_index + 1. Then Read from that line.
Example: row_index 6 → "RESULTS FOR ROW 7" in results.txt.

## Running tally
- P0: 0
- P1: 4 (Row 1 repeated sentence, Row 1 wrong source tag, Row 4 inaccurate disease status, Row 10 lab value direction wrong)
- P1: 5
- P2: 46
- Perfect letters (0 issues): Rows 9, 13
- New pattern: [REDACTED] leaks into patient letter (Row 29)

## Additional patterns found (rows 6-19)
6. LLM sometimes uses "which means" to connect Type_of_Cancer with Metastasis, creating false causal link
7. When recent_changes and current_meds both mention the same medication → duplication
8. Old/stale lab data (e.g., blood sugar from years ago) sometimes included
9. Radiation referral sometimes described as "second opinion" instead of "evaluation/consult"

## Common patterns found so far
1. "ER+/PR+/HER2- invasive ductal carcinoma" often not explained in plain language
2. Stage progression info (e.g., "Originally IIA, now IV") often omitted
3. recent_changes and therapy_plan can contain same info → duplicate sentences
4. LLM sometimes tags with wrong field name (e.g., Others content tagged as Specialty)
5. response_assessment details sometimes oversimplified to just "we will monitor"
