# 错题本: v26 notool 审查 — 28 个有问题的 Sample

> 来源: v26_full_notool_20260408_080004/review.md
> 用途: v27 验证用。v27 只改了 letter prompt，extraction 不变
> 总计: 6 P1 + 44 P2（100 samples 中 28 个有问题）

---

## P1 问题汇总（6 个）

| ROW | coral_idx | 字段 | 问题 | 类型 | v27 能修吗 |
|-----|-----------|------|------|------|-----------|
| 1 | 140 | lab_plan | 混入 MRI/bone scan，没列具体 labs | extraction | ❌ 需 tool calling |
| 8 | 147 | response_assessment | "Not yet on treatment" 但完成了不完整 neoadjuvant + surgery | extraction | ❌ 需 prompt 改进 |
| 10 | 149 | response_assessment | "does not provide evidence" 但有 8.8cm 残余+20LN+ | extraction | ❌ 需 prompt 改进 |
| 11 | 150 | response_assessment | 引用换药前 PET 进展，A/P 说 "Exam stable" | extraction | ❌ 需 prompt 改进 |
| 12 | 151 | Advance care | "Not discussed" 但 DNR/DNI in problem list | extraction | ❌ 需 tool calling |
| 88 | 227 | response_assessment | "Not mentioned" for post-neoadjuvant progression→surgery | extraction | ❌ 模型顽固 |

**v27 letter 改动不影响任何 P1（全是 extraction 问题）。**

---

## P2 问题逐 Sample 清单

### ROW 1 (coral_idx 140) — 3 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | imaging_plan | 遗漏 Bone Scan |
| 2 | Type_of_Cancer | IDC 原文未明确 |
| 3 | therapy_plan | 重复 medication_plan |

### ROW 2 (coral_idx 141) — 3 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Metastasis | 不完整 + chest wall 是区域性 |
| 2 | imaging_plan | 遗漏 rescheduled MRI |
| 3 | Referral Others | "Home health?" 是疑问非确定 |

### ROW 3 (coral_idx 142) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Referral-Genetics | 检测 ordering ≠ referral |

### ROW 4 (coral_idx 143) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | therapy_plan | 重复 medication_plan |

### ROW 5 (coral_idx 144) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | follow_up | 循环表述 |

### ROW 6 (coral_idx 145) — 3 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Patient type | 与 summary 不一致 |
| 2 | **letter** | **"You appear to be feeling anxious" — v27 应修复** |
| 3 | Referral-Genetics | 历史转诊（Myriad 已完成） |

### ROW 7 (coral_idx 146) — 3 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 LVEF recheck（应为 imaging） |
| 2 | lab_plan | 混入 LVEF recheck（应为 imaging） |
| 3 | **letter** | **"medication level" 实际是 LVEF recheck — v27 可能改善** |

### ROW 8 (coral_idx 147) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | medication 内容，遗漏 port placement |

### ROW 10 (coral_idx 149) — 0 P2（仅 P1）

### ROW 11 (coral_idx 150) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | imaging_plan | 遗漏 Echo q6mo |

### ROW 12 (coral_idx 151) — 2 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | imaging_plan | 遗漏 Echo q6mo |
| 2 | Metastasis | lung 已 resolved 但仍列出 |

### ROW 13 (coral_idx 152) — 2 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | response | "On treatment" 但未开始（手术完成，tamoxifen 未决定） |
| 2 | findings | 左右乳混淆 |

### ROW 14 (coral_idx 153) — 2 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | current_meds | 空值（患者在替代治疗中） |
| 2 | response_assessment | 漏了 CA 27.29 从 193→48 |

### ROW 15 (coral_idx 154) — 2 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | genetic_testing | 已做回顾非新计划 |
| 2 | procedure_plan | 混入 Rx recommendations |

### ROW 20 (coral_idx 159) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 imaging/referral/medication |

### ROW 22 (coral_idx 161) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | genetic_testing | 有 medication plan 内容 |

### ROW 24 (coral_idx 163) — 2 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 genetic testing |
| 2 | Metastasis | "Not sure" 应为 "No" |

### ROW 25 (coral_idx 164) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | response | 引用 pre-treatment PET |

### ROW 33 (coral_idx 172) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | **letter** | **stage "now considered IIIA" for NED patient — v27 follow-up 逻辑可能改善** |

### ROW 38 — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | response | "not responding" but not yet on treatment |

### ROW 39 (coral_idx 178) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Type_of_Cancer | 错误推断 ER+（goserelin 用于 fertility，非 ER+） |

### ROW 52 — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 fertility referral |

### ROW 57 — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 genetic counseling |

### ROW 74 (coral_idx 213) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Type_of_Cancer | HER2+ 混淆了 gastric cancer 的 HER2 状态 |

### ROW 75 — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | procedure_plan | 混入 genetics+fertility |

### ROW 83 (coral_idx 222) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Stage | "Stage IV" 但 Distant Met=No，axillary LN 是 regional |

### ROW 88 (coral_idx 227) — 0 P2（仅 P1）

### ROW 95 (coral_idx 234) — 1 P2
| # | 字段 | 问题 |
|---|------|------|
| 1 | Stage | "Stage IV" 但 Distant Met=No, ISPY trial (Stage I-III only) |

---

## v27 Letter 改动可能修复的 P2

| ROW | 原问题 | v27 改动 | 预期效果 |
|-----|--------|---------|---------|
| 6 | letter "You appear to be feeling anxious" | 情感支持改为标准句 | ✅ 应修复 |
| 7 | letter "medication level" 误述 | letter 结构重组 | 可能改善 |
| 33 | letter stage "now considered IIIA" | follow-up 不重复诊断 | 可能改善 |

---

## v27 验证用 row_indices（0-based）

```
[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 19, 21, 23, 24, 32, 37, 38, 51, 56, 73, 74, 82, 87, 94]
```

28 个 sample，对应 ROW 1-8, 10-15, 20, 22, 24-25, 33, 38-39, 52, 57, 74-75, 83, 88, 95
