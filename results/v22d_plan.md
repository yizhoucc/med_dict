# v22d 实验计划：plan_extraction 用全文

## 改动
把 plan_extraction prompts 的输入从 A/P 段改为整篇 note_text。

## 风险
- 笔记很长（有的 16000 字），可能超过 context window 或导致幻觉
- 蝴蝶效应：改了输入可能影响所有 plan 字段（不只是 imaging/lab）
- 可能引入新问题（从历史影像/检验结果中提取过去的 plan）

## 测试行
Row 0 (ROW 1) — 已知 imaging/lab plan 遗漏
Row 6 (ROW 7) — imaging empty
Row 48 (ROW 49) — imaging+lab empty

3 行针对性测试，对比 v22 结果看是否改善。
