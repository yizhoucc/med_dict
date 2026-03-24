# v22e 最终审查: 全量 61 samples

Date: 2026-03-23
版本: v22e（19 个 POST hooks）
审查方式: 20 samples 人工逐字审查 + 41 samples 对比扫描
数据: 分两批运行（20 regression + 41 remaining）

---

## POST hook 全量统计

| Hook | 触发次数 | Samples | 误触发 |
|------|---------|---------|--------|
| POST-SELF-MANAGED | 1 | Row 13 | 0 |
| POST-SELF-MANAGED-PLAN | 2 | Row 13 | 0 |
| POST-SELF-MANAGED-SUMMARY | 1 | Row 13 | 0 |
| POST-HER2-FISH | 1 | Row 13 | 0 |
| POST-DISTMET-DEFAULT | 1 | Row 82 | 0 |
| POST-MEDS-IV-CHECK | 1 | Row 89 | 0 |
| POST-RECEPTOR-UPDATE | 1 | Row 94 | 0 |
| POST-IMAGING (full note search) | 3 | Row 0 + others | 0 |
| POST-LAB-SEARCH | 1 | Row 0 | 0 |
| **总计** | **12** | | **0** |

---

## 61 samples 结果

```
P0: 0
P1: 4 (Row 5 Stage, Row 58 时态, Row 69 Stage随机, Row 99 Gemzar)
P2: ~8 (Stage "Not mentioned" ×5, Type "not tested" ×2, 其他 ×1)
无问题: ~49 samples (80%)
回归: 0
错误/crash: 0
```

## 已修复的 P0 确认（全部保持）

| Sample | 问题 | Fix | 状态 |
|--------|------|-----|------|
| Row 0 | imaging/lab plan 遗漏 | POST-IMAGING + POST-LAB-SEARCH | ✅ |
| Row 1 | current_meds 遗漏 irinotecan | POST-MEDS-IV-CHECK | ✅ |
| Row 13 | 墨西哥自管药物（5 个字段）| POST-SELF-MANAGED 系列 | ✅ |
| Row 56 | docetaxel FP + 过敏药 + TNBC | Pattern 6 + POST-SUPP-ALLERGY + POST-TYPE-VERIFY-TNBC | ✅ |
| Row 82 | Distant Met 空值 | POST-DISTMET-DEFAULT | ✅ |
| Row 89 | current_meds 遗漏 AC | POST-MEDS-IV-CHECK | ✅ |
| Row 94 | Type PR 状态矛盾 | POST-RECEPTOR-UPDATE | ✅ |

## v14→v22e 演进总结

| 版本 | P0 | 主要改进 |
|------|-----|---------|
| v14 | 数十个 | 初始版本 |
| v15a | ~10 | HER2 三 bug 修复 |
| v17 | ~6 | Referral 修复, TNBC, regional LN |
| v19 | ~5 | IV-CHECK, Patient type |
| v20 | ~5 | docetaxel FP, 过敏药, 眼药水 |
| v21 | 4 | POST-SELF-MANAGED (current_meds) |
| v22 | 1 | DISTMET-DEFAULT, RECEPTOR-UPDATE, HER2-FISH, SELF-MANAGED-PLAN/SUMMARY |
| **v22e** | **0** | **POST-IMAGING/LAB-SEARCH** |
