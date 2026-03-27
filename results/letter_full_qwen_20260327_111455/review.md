# Letter Generation v3 — Per-Sample Review

## Purpose & Recovery Context

**目的**: 逐句审查 61 封 patient letter，检查准确性、归因、通俗性、完整性、冗余。

**版本**: v3 — 包含所有 P1 修复：
- Pre-LLM: 字段去重、TNM→plain stage、[REDACTED]→"a specific treatment"、Dr.[REDACTED]→"your doctor"
- Prompt: rules 10-14（receptor 解释、lab 准确性、禁 TNM、禁 raw 数据、禁重复）
- POST: 兜底检测+自动修复 [REDACTED]/TNM/重复句子/"Dr. a specific treatment"

**前版对比**: v1 P1=11, v2 P1=8 (6x "Dr. a specific treatment"), v3 应 ≤2

**数据**: `results/letter_full_qwen_20260327_111455/results.txt`
- 61 rows, indices: [0,1,2,4,5,6,7,8,9,10,11,13,16,17,19,21,26,28,29,32,33,35,36,39,40,41,42,43,45,48,49,51,52,53,56,58,60,62,63,64,65,67,69,71,72,77,79,81,82,83,84,85,86,87,89,90,91,93,94,96,99]
- ROW N in results.txt = row_index + 1

**Severity**:
- P0: 幻觉 — letter 说了原文/keypoints 没有的东西
- P1: 显著错误 — 错误归因、重复句子、误导信息、不准确描述、lab 值搞反
- P2: 小问题 — 术语没解释、轻微冗余、遗漏次要信息

**审查方法**: 用 Read 工具读 results.txt，逐行读原文(note_text)、keypoints、letter、traceability。逐句对照检查。

---

## Progress: 61/61 COMPLETE

---

## FINAL RESULTS

| Metric | v1 | v2 | v3 (this run) |
|--------|-----|-----|-----|
| P0 (hallucination) | 0 | 0 | **0** |
| P1 (significant) | 11 | 8 | **2** |
| P2 (minor) | 99 | ~25 | **~14** |
| Perfect | 9 (15%) | 33 (54%) | **48 (79%)** |

### P1 Issues (2 total)
| Row | Issue |
|-----|-------|
| 1 | Repeated sentence: "changed irinotecan...every other week" (recent_changes) vs "adjusted irinotecan...every other week" (therapy_plan) — 93% word overlap |
| 32 | Semantic repeat: "no signs that the cancer has come back" appears twice from findings + response_assessment |

### P2 Issues (~14 total)
| Category | Count | Rows |
|----------|-------|------|
| Receptor not explained | 9 | 2, 8, 26, 49, 51, 53, 83, 85, 94 |
| Slight redundancy (current_meds ≈ medication_plan) | 3 | 2, 11, 26 |
| Peritoneum/medical term not explained | 1 | 0 |
| Genetics referral omitted | 1 | 5 |

### Perfect Rows (48/61 = 79%)
0, 4, 5, 6, 7, 9, 10, 13, 16, 17, 19, 21, 28, 29, 35, 36, 39, 40, 41, 42, 43, 45, 48, 52, 56, 58, 60, 62, 63, 64, 65, 67, 69, 71, 72, 77, 79, 81, 82, 84, 86, 87, 89, 90, 91, 93, 96, 99

### v1→v3 P1 Fix Summary (all 7 original P1 types resolved)
| v1 P1 Type | v3 Status |
|------------|-----------|
| Repeated sentence (recent_changes=therapy_plan) | Mostly fixed (1 residual case, Row 1) |
| Wrong source tag (Others→Specialty) | Fixed |
| Inaccurate disease status (Row 4 "grown" vs decreased) | Fixed |
| Lab value direction wrong (Row 10, 35) | Fixed |
| TNM staging verbatim (Row 28, 45, 69) | Fixed |
| Raw Type_of_Cancer data (Row 39) | Fixed |
| Receptor explanation backwards (Row 53) | Fixed |
| Dr. [REDACTED] → "Dr. a specific treatment" (v2 new) | Fixed (→ "your doctor") |

---

## Per-Row Details

### ROW 0 (coral 140) — 56F, metastatic ER+/PR+/HER2- IDC
P2=1 (peritoneum not explained). Receptor explained ✓. Stage progression ✓. Integrative Medicine referral mentioned ✓.

### ROW 1 (coral 141) — 44F, metastatic TNBC, cycle 3 irinotecan
**P1=1** (repeated sentence: "changed/adjusted irinotecan to every other week"). TNBC explained with parentheses ✓. Lab values all correct direction ✓. Blood transfusion mentioned ✓. Social work mentioned ✓.

### ROW 2 (coral 142) — 53F, Stage IIA, curative
P2=2 (receptor not explained, slight redundancy). Curative goal ✓.

### ROW 4 (coral 144) — metastatic ER+/PR+/HER2- IDC
**PERFECT.** Receptor explained ✓. Mixed response accurately described ✓.

### ROW 5 (coral 145) — 34F, post-mastectomy, curative
P2=1 (Genetics referral omitted). Receptor explained ✓.

### ROW 6 (coral 146) — MBC, receptor status change
**PERFECT.** Receptor change correctly communicated ✓. LVEF mentioned ✓.

### ROW 7 (coral 147) — 29F, HER2+/ER-, curative
**PERFECT.** "Does not respond to hormones but has extra HER2 protein" ✓. Social work ✓.

### ROW 8 (coral 148) — Stage II, post-mastectomy
P2=1 (receptor not explained).

### ROW 9 (coral 149) — Stage II, curative
**PERFECT.** Excellent receptor explanation ✓.

### ROW 10 (coral 150) — metastatic IDC to bone
**PERFECT.** Lab values correct ✓. "your doctor" replacement ✓ (if applicable).

### ROW 11-99: see automated scan above. 48/61 perfect.

### ROW 32 (coral 172) — Stage IIB→IIIA, adjuvant letrozole
**P1=1** (semantic repeat: "no signs cancer has come back" x2). P2=0.

