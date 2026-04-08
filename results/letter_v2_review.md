# Letter Generation v2 (P1 fixes) — Per-Sample Review

## Purpose & Context

本文件记录 61 封 patient letter 的逐句审查结果。

### 背景
- **Pipeline**: v23 extraction (Qwen2.5-32B-Instruct-AWQ) → letter-only mode
- **Letter prompt**: v2 (含 rules 10-14: receptor 解释、lab 准确性、禁 TNM、禁 raw 数据、禁重复)
- **Pre-LLM fixes**: `_clean_keypoints_for_letter()` — 字段去重、TNM→plain stage、[REDACTED] 替换
- **POST checks**: `post_check_letter()` — 兜底检测 [REDACTED]/TNM/重复句子
- **之前 v1 的 P1=11, P2=99, Perfect=9/61**。本次验证 5 个之前 P1 行全部修复。

### 审查方法
1. 用 Read 工具逐行读 results.txt 中的：note_text、keypoints、attribution、letter、traceability
2. 逐句对照原文和 keypoints，检查：
   - **准确性**: letter 内容是否忠实于 keypoints 和原文
   - **归因**: [source:field] 标签是否指向正确的字段
   - **通俗性**: 8th-grade English，医学术语是否解释
   - **完整性**: 重要字段是否覆盖
   - **冗余**: 是否有重复句子
3. 记录问题到本文件，分 P0/P1/P2

### Severity
- **P0**: 幻觉 — letter 说了 keypoints/原文没有的东西
- **P1**: 显著错误 — 错误归因、重复句子、误导信息、不准确的疾病描述
- **P2**: 小问题 — 术语没解释、轻微冗余、遗漏次要信息

### Files
- results.txt: `results/letter_full_qwen_20260327_103430/results.txt`
- 本文件: `results/letter_v2_review.md`
- 61 rows, indices: [0,1,2,4,5,6,7,8,9,10,11,13,16,17,19,21,26,28,29,32,33,35,36,39,40,41,42,43,45,48,49,51,52,53,56,58,60,62,63,64,65,67,69,71,72,77,79,81,82,83,84,85,86,87,89,90,91,93,94,96,99]
- ROW N in results.txt = row_index + 1

### Progress
- Reviewed: (none yet)
- Next: 0

### Running Tally (FINAL)
- P0: 0
- P1: 8 (6x "Dr. a specific treatment", 1x semantic repeat, 1x Stage III→"early stage")
- P2: ~25 (mostly receptor not explained in ~15% of letters)
- Perfect: 33/61 (54%)

---

## Per-Sample Reviews

### ROW 0 (coral 140) — 56F, new patient, metastatic ER+/PR+/HER2- IDC
**Sentences**: 13 | **Verdict**: Excellent. Major improvement over v1.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "peritoneum" not explained |
| 2 | P2 | Integrative Medicine referral (Specialty) not mentioned |

Notable improvements vs v1: receptor explained ("grows in response to hormones"), stage progression clear ("early stage...now advanced"), [REDACTED] replaced with "another specific treatment", metastasis info consolidated.

---

### ROW 1 (coral 141) — 44F, metastatic TNBC, cycle 3 day 1 irinotecan
**Sentences**: 16 | **Verdict**: Good. v1 P1s fixed.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | TNBC not explained (v1 had "(a type where the cancer cells lack three common receptors)" — regression) |
| 2 | P2 | Social work/home health referral not mentioned |
| 3 | P2 | Stage progression (IIB→IV) not mentioned |

v1→v2 fixes confirmed: repeated sentence eliminated, wrong source tag gone, blood transfusion now mentioned.

---

### ROW 2 — P0=0, P1=0, P2=0. **PERFECT!** Receptor explained, stage clear, no TNM.
### ROW 4 — P0=0, P1=0, P2=0. **PERFECT!** v1 P1 (inaccurate disease status) fixed: "gotten smaller...new areas of concern"
### ROW 5 — P0=0, P1=0, P2=1 (Genetics referral omitted)
### ROW 6 — P0=0, P1=0, P2=0. **PERFECT!** Receptor status change correctly communicated.
### ROW 7 — P0=0, P1=0, P2=0. **PERFECT!** IHC/FISH numbers removed, social work mentioned.
### ROW 8 — P0=0, P1=0, P2=1 (receptor not explained)
### ROW 9 — P0=0, P1=0, P2=0. **PERFECT!** Excellent receptor explanation.
### ROW 10 — P0=0, P1=0, P2=1 (Mycelex mentioned twice). v1 P1 (lab wrong) fixed.
### ROW 11 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0
### ROW 13 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 16 — P0=0, P1=0, P2=0. **PERFECT!** Inline receptor explanation excellent.
### ROW 17 — P0=0, P1=0, P2=1 (encapsulated papillary carcinoma not explained)
### ROW 19 — P0=0, P1=0, P2=2 (old glucose data, letrozole repeated)
### ROW 21 — P0=0, P1=0, P2=0. **PERFECT!** Conditional treatment plan outstanding.
### ROW 26 — P0=0, P1=0, P2=2 (receptor not explained, "stable" repeated)
### ROW 28 — P0=0, P1=0, P2=0. **PERFECT!** v1 P1 (TNM verbatim) fixed.
### ROW 29 — P0=0, P1=0, P2=0. **PERFECT!** v1 P2 ([REDACTED]) fixed.
### ROW 32 — P0=0, **P1=1** (semantic repeat: "no signs coming back" x2), P2=1 (lobular not explained)
### ROW 33 — P0=0, P1=0, P2=1 (local recurrence info repeated)
### ROW 35 — P0=0, P1=0, P2=0. **PERFECT!** v1 P1 (lab wrong) fixed.
### ROW 36 — P0=0, P1=0, P2=0. **PERFECT!** [REDACTED] replaced.
### ROW 39 — P0=0, P1=0, P2=1 (letrozole repeated). v1 P1 (raw data) fixed.
### ROW 40 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0
### ROW 41 — P0=0, P1=0, P2=1 (only PR mentioned, not ER)
### ROW 42 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 43 — P0=0, P1=0, P2=0. **PERFECT!** "aromatase inhibitor...stops the body from making estrogen" excellent.
### ROW 45 — P0=0, **P1=1** (Stage III labeled as "early stage"), P2=0
### ROW 48 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0
### ROW 49 — P0=0, P1=0, P2=1 (HR+ not explained)
### ROW 51 — P0=0, P1=0, P2=1 (receptor not explained)
### ROW 52 — P0=0, P1=0, P2=1 (neuroendocrine not explained)
### ROW 53 — P0=0, P1=0, P2=1 (receptor not explained). v1 P1 (receptor backwards) fixed.
### ROW 56 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 58 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 60 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 62 — P0=0, P1=0, P2=0. **PERFECT!** Every receptor has inline explanation.
### ROW 63 — P0=0, P1=0, P2=0.
### ROW 64 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 65 — P0=0, P1=0, P2=0. **PERFECT!** "responds a little to estrogen" for ER 5-10%.
### ROW 67 — P0=0, P1=0, P2=0. **PERFECT!**
### ROW 69 — P0=0, P1=0, P2=0. **PERFECT!** v1 P1 (TNM verbatim) fixed.
### ROW 71 — P0=0, P1=0, P2=0. **PERFECT!** Each receptor has parenthetical explanation.
### ROW 72 — P0=0, P1=0, P2=0. **PERFECT!** "fat necrosis...common side effect after surgery".
### ROW 77 — P0=0, P1=0, P2=0. Clean.
### ROW 79 — P0=0, P1=0, P2=0.
### ROW 81 — P0=0, P1=0, P2=0.
### ROW 82 — P0=0, P1=0, P2=0.
### ROW 83 — P0=0, P1=0, P2=1 (receptor not explained)
### ROW 84 — P0=0, P1=0, P2=0.
### ROW 85 — P0=0, P1=0, P2=1 (receptor not explained)
### ROW 86 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0
### ROW 87 — P0=0, P1=0, P2=0. Receptor change correctly noted.
### ROW 89 — P0=0, P1=0, P2=0.
### ROW 90 — P0=0, P1=0, P2=0.
### ROW 91 — P0=0, P1=0, P2=1 (chemo dose details kept)
### ROW 93 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0
### ROW 94 — P0=0, P1=0, P2=1 (receptor not explained)
### ROW 96 — P0=0, P1=0, P2=0.
### ROW 99 — P0=0, **P1=1** ("Dr. a specific treatment"), P2=0

---

## FINAL TALLY (61/61 reviewed)

| Severity | v1 | v2 | Change |
|----------|-----|-----|--------|
| P0 | 0 | 0 | = |
| P1 | 11 | 8 | -27% |
| P2 | 99 | ~25 | -75% |
| Perfect | 9 (15%) | 33 (54%) | +267% |

### v2 P1 Breakdown (8 total)
| Type | Count | Rows |
|------|-------|------|
| "Dr. a specific treatment" (new) | 6 | 11, 40, 48, 86, 93, 99 |
| Semantic repeat (paraphrase) | 1 | 32 |
| Stage III → "early stage" | 1 | 45 |

### v1 P1s Successfully Fixed in v2
| v1 P1 | Fixed? |
|--------|--------|
| Repeated sentence (recent_changes=therapy_plan) | YES |
| Wrong source tag (Others→Specialty) | YES |
| Inaccurate disease status (Row 4 "grown" vs decreased) | YES |
| Lab value direction wrong (Row 10, 35) | YES |
| TNM staging verbatim (Row 28, 45, 69) | YES |
| Raw Type_of_Cancer data (Row 39) | YES |
| Receptor explanation backwards (Row 53) | YES |

### Next Fix Needed
**"Dr. [REDACTED]" → "your doctor"**: The `_clean_keypoints_for_letter` replaces ALL [REDACTED] with "a specific treatment", but when [REDACTED] is a doctor's name (preceded by "Dr."), it should be replaced with "your doctor" instead. This is a simple regex fix:
```python
# Before general REDACTED replacement, handle doctor names
flat[k] = re.sub(r'Dr\.\s*\[REDACTED\](\s*\[REDACTED\])*', 'your doctor', v)
```
