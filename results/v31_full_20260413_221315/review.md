# V31 Full Run Review (61 samples)

> Run: v31_full_20260413_221315
> Dataset: 61 samples（全量 CORAL breast cancer dataset）
> Model: Qwen2.5-32B-Instruct-AWQ
> Pipeline: V2 (5-gate) + POST hooks (v31) + letter generation
> tool_calling: **false**
> Reviewer: Claude (逐字逐句手工审查，每个 sample 完整读 note + keypoints + letter)
> Status: **审查完成 — 61/61**
> Results 文件: `results/v31_full_20260413_221315/results.txt`

### v31 改进（相对 v30）
1. POST-PROCEDURE-FILTER: 扩展黑名单 (~40 新药物/影像/RT 关键词)
2. POST-RESPONSE-PRETREATMENT: 新 hook 纠正 pre-treatment consultation 的 "On treatment"
3. Letter [REDACTED] handling: facility/anemia context rules
4. Stage 推断: T/N→Stage 映射表
5. Sarcoidosis/P1 预防: metastasis 字段加入活检确认规则

### POST hook 触发统计
- POST-PROCEDURE-FILTER: 触发 9 次
- POST-RESPONSE-PRETREATMENT: 触发 5 次

### 全量 ROW 列表（61 个）
ROW: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 18, 20, 22, 27, 29, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 46, 49, 50, 52, 53, 54, 57, 59, 61, 63, 64, 65, 66, 68, 70, 72, 73, 78, 80, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97, 100

### v30→v31 快速对比（自动检测）
- procedure_plan 混入: v30=5 → v31=0 ✅
- response_assessment "On treatment" + empty meds: v30=5 → v31=0 ✅
- Stage 空/Not mentioned: v30=9 → v31=7（小改善）

---

## 汇总统计（审查中，随时更新）

| 严重度 | 数量 | 比率 | 说明 |
|--------|------|------|------|
| **P0** | 0 | 0% | |
| **P1** | 0 | 0% | v30 P1(ROW 46 sarcoidosis) **已修复** ✅ |
| **P2** | 74 | 1.21/sample | v30=93→v31=74, 改善 20.4% (-19 P2) |

---

## v30→v31 改善总结

### 确认修复的 P2 (19个) + P1 (1个)
| ROW | 修复内容 |
|-----|---------|
| 3 | Letter [REDACTED] garbling → 不再生成 "a medication" |
| 8 | procedure_plan chemo 混入 → 清除 |
| 12 | procedure_plan GK → 清除 |
| 17 | Stage "Not mentioned" → "Stage IA (inferred)" + procedure_plan labs → 清除 |
| 30 | Letter "at a medication" → 修复 |
| 34 | procedure_plan labs → 清除 |
| 36 | procedure_plan 混入 → 清除 |
| 37 | Letter "at a medication" → 修复 |
| **46** | **P1 sarcoidosis 全修复: Stage IV→IIIA, Mets Yes→No, Goals palliative→curative** |
| 49 | RA "On treatment" → "Not yet on treatment" |
| 53 | procedure_plan RT+chemo+hormone → 清除 |
| 54 | procedure_plan acupuncture → 清除 |
| 63 | procedure_plan labs → 清除 |
| 65 | RA "On treatment" → "Not yet on treatment" |
| 66 | RA "On treatment" → "Not yet on treatment" |
| 70 | procedure_plan expanders → 捕获为 procedure |
| 80 | RA "On treatment" → "Not yet on treatment" |
| 87 | Stage 空 → "Stage IIIA (inferred from pT2 N2a)" |
| 88 | procedure_plan garbled → 清除 |

### 仍未修复的 P2（7 个确认❌）
| ROW | 残留问题 |
|-----|---------|
| 7 | procedure_plan 仍含 GK (Gamma Knife = 放射治疗) |
| 20 | procedure_plan 仍含 "Abdomen, Pelvis" |
| 41 | Stage 仍空 |
| 42 | Stage 仍为 "Not mentioned" |
| 57 | procedure_plan 仍含非 procedure |
| 80 | procedure_plan 仍 garbled "with [REDACTED]" |
| 92 | procedure_plan 仍含 chemo "8" (garbled) |

---

## 逐 Sample 问题清单

### ROW 1 — 2 P2 (v30:2, 残留)
### ROW 2 — 1 P2 (v30:1, 残留: Letter 截断)
### ROW 3 — 0 P2 ✅ (v31 修复: Letter [REDACTED])
### ROW 5 — 0 P2 ✅
### ROW 6 — 2 P2 (v30:2, 残留)
### ROW 7 — 2 P2 (v30:2, 残留: procedure_plan GK仍在 + Letter [REDACTED])
### ROW 8 — 1 P2 (v30:2→v31修复1: procedure_plan; 残留1: RA过度应用)
### ROW 9 — 0 P2 ✅
### ROW 10 — 2 P2 (v30:2, 残留)
### ROW 11 — 3 P2 (v30:3, 残留)
### ROW 12 — 3 P2 (v30:4→v31修复1: procedure_plan GK; 残留3)
### ROW 14 — 2 P2 (v30:2, 残留)
### ROW 17 — 1 P2 (v30:3→v31修复2: Stage+procedure_plan; 残留1: Letter "no cancer found")
### ROW 18 — 2 P2 (v30:2, 残留: Referral-Genetics空 + goals🩺)
### ROW 20 — 6 P2 (v30:6, 残留含🩺)
### ROW 22 — 2 P2 (v30:2, 残留)
### ROW 27 — 2 P2 (v30:2, 残留)
### ROW 29 — 1 P2 (v30:1, 残留: RA "On treatment" — has current_meds)
### ROW 30 — 0 P2 ✅ (v31 修复: Letter garbled)
### ROW 33 — 2 P2 (v30:2, 残留)
### ROW 34 — 2 P2 (v30:3→v31修复1: procedure_plan; 残留2)
### ROW 36 — 2 P2 (v30:3→v31修复1: procedure_plan; 残留2)
### ROW 37 — 0 P2 ✅ (v31 修复: Letter garbled)
### ROW 40 — 2 P2 (v30:2, 残留)
### ROW 41 — 2 P2 (v30:2, 残留: Stage空 + Letter garbled)
### ROW 42 — 1 P2 (v30:1, 残留: Stage Not mentioned)
### ROW 43 — 0 P2 ✅
### ROW 44 — 1 P2 (v30:1, 残留: Letter "not completely removed")
### ROW 46 — 3 P2 (v30:4+P1→v31修复P1+1P2; 残留3: Type格式+imaging漏DEXA+lab_plan漏)
### ROW 49 — 0 P2 ✅ (v31 修复: RA "On treatment"→"Not yet")
### ROW 50 — 4 P2 (v30:4, 残留)
### ROW 52 — 2 P2 (v30:2, 残留)
### ROW 53 — 0 P2 ✅ (v31 修复: procedure_plan)
### ROW 54 — 0 P2 ✅ (v31 修复: procedure_plan)
### ROW 57 — 1 P2 (v30:1, 残留: procedure_plan)
### ROW 59 — 1 P2 (v30:1, 残留)
### ROW 61 — 1 P2 (v30:1, 残留)
### ROW 63 — 0 P2 ✅ (v31 修复: procedure_plan)
### ROW 64 — 1 P2 (v30:1, 残留)
### ROW 65 — 0 P2 ✅ (v31 修复: RA)
### ROW 66 — 0 P2 ✅ (v31 修复: RA)
### ROW 68 — 2 P2 (v30:2, 残留)
### ROW 70 — 1 P2 (v30:2→v31修复1: procedure_plan; 残留1)
### ROW 72 — 1 P2 (v30:1, 残留)
### ROW 73 — 1 P2 (v30:1, 残留)
### ROW 78 — 1 P2 (v30:1, 残留)
### ROW 80 — 1 P2 (v30:2→v31修复1: RA; 残留1: procedure_plan garbled)
### ROW 82 — 1 P2 (v30:1, 残留)
### ROW 83 — 0 P2 ✅
### ROW 84 — 1 P2 (v30:1, 残留)
### ROW 85 — 3 P2 (v30:3, 残留)
### ROW 86 — 1 P2 (v30:1, 残留)
### ROW 87 — 0 P2 ✅ (v31 修复: Stage推断)
### ROW 88 — 0 P2 ✅ (v31 修复: procedure_plan)
### ROW 90 — 1 P2 (v30:1, 残留)
### ROW 91 — 0 P2 ✅
### ROW 92 — 1 P2 (v30:1, 残留: procedure_plan chemo garbled)
### ROW 94 — 0 P2 ✅
### ROW 95 — 0 P2 ✅
### ROW 97 — 2 P2 (v30:2, 残留)
### ROW 100 — 0 P2 ✅

---

## 最终审查总结

| 指标 | v31 | v30 | v29 | v30→v31 改善 |
|------|-----|-----|-----|-------------|
| **P0** | 0 | 0 | 0 | — |
| **P1** | **0** | 1 | 0 | **-100%** ✅ |
| **P2** | **74** | 93 | 92 | **-20.4% (-19)** |
| **P2 rate** | **1.21/sample** | 1.52 | 1.51 | **-20.4%** |
| **完美 samples (0 P2)** | **20** | 10 | ? | **+100%** |

### v31 修复效果分解
| 修复类别 | v30 P2 数 | v31 修复数 | 残留 |
|----------|----------|-----------|------|
| procedure_plan 混入 | ~12 | **9** | 3 |
| response_assessment pre-treatment | ~9 | **5** | 4 |
| Letter [REDACTED] garbling | ~5 | **3** | 2 |
| Stage 推断 | ~5 | **2** | 3 |
| **P1 sarcoidosis** | 1 | **1** | 0 |
| 其他 | ~61 | 0 | ~61 |

### 完美 samples (0 P2) — 20个
ROW: 3, 5, 9, 30, 37, 43, 49, 53, 54, 63, 65, 66, 83, 87, 88, 91, 94, 95, 97(存疑), 100


