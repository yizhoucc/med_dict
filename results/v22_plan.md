# v22 修复计划

## 目标
修复 v21d 审查发现的 4 个 P0 和 2 个 P1。

## 修复清单

| Fix | 目标 | 方法 | 解决 |
|-----|------|------|------|
| A | Row 82 Distant Met 空 | POST-DISTMET-DEFAULT | P1→修复 |
| B | Row 13 medication_plan/therapy_plan | POST-SELF-MANAGED 扩展 | P0×2→降级 |
| C | Row 94 Type PR+ (应 PR-) | POST-RECEPTOR-UPDATE | P0→修复 |
| D | Row 13 Type HER2 "unclear" | POST-HER2-CHECK 扩展 | P1→修复 |

## 测试行
Row 1, 13, 82, 89, 94（5 行针对性测试）

## 预期效果
P0: 4→1 | P1: 8→6
