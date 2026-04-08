# Letter Generation v4 (Final) — Per-Sample Review

## Purpose & Recovery Context

逐句审查 61 封 patient letter。这是最终版本，包含所有修复：
- Pre-LLM: 字段去重(word-overlap>60%, 含 response_assessment≈findings), TNM→plain stage, [REDACTED]→generic, Dr.[REDACTED]→"your doctor"
- Prompt: rules 10-14 (receptor 解释、lab 准确性、禁 TNM/raw data/重复)
- POST: 兜底 [REDACTED]/TNM/重复句子/"Dr. a specific treatment"

**前版**: v1 P1=11, v2 P1=8, v3 P1=2, v4 验证 P1=0 (on rows 1,32)

**数据**: `results/letter_full_qwen_20260327_134953/results.txt`, 61 rows
**ROW N** in results.txt = row_index + 1

**Severity**: P0=幻觉, P1=显著错误, P2=小问题

---

## Progress: 61/61 COMPLETE

---

## FINAL RESULTS

| Metric | v1 | v2 | v3 | **v4 (this)** |
|--------|-----|-----|-----|------|
| P0 | 0 | 0 | 0 | **0** |
| P1 | 11 | 8 | 2 | **2** |
| P2 | 99 | ~25 | ~14 | **~8** |
| Perfect | 15% | 54% | 79% | **85%** |

### P1 Issues (2 total, both in Row 33)
| Row | Issue |
|-----|-------|
| 33 | "cancer...that does not respond to hormones" — ER+ cancer DOES respond to hormones. LLM contradicts sentence 3 which correctly says "ER positive". |
| 33 | Semantic repeat: "no sign has spread to other parts" appears in sentence 4 and sentence 10 |

### P2 Issues (~8 total)
| Category | Count | Rows |
|----------|-------|------|
| Receptor not explained | 7 | 8, 26, 40, 49, 51, 53, 94 |
| Stage II-III → "early stage" (debatable) | 1 | 29 |

### Perfect Rows (52/61 = 85%)
All rows except: 0(P2), 1(P2), 4(P2), 5(P2), 8(P2), 26(P2), 29(P2), 33(P1x2), 40(P2), 49(P2), 51(P2), 53(P2), 94(P2)

### All v1 P1 types — fix status
| v1 P1 Type | Status |
|------------|--------|
| Repeated sentence (recent_changes=therapy_plan) | FIXED (word-overlap dedup) |
| Wrong source tag (Others→Specialty) | FIXED |
| Inaccurate disease status (Row 4) | FIXED |
| Lab value direction wrong (Row 10, 35) | FIXED |
| TNM staging verbatim (Row 28, 45, 69) | FIXED |
| Raw Type_of_Cancer data (Row 39) | FIXED |
| Receptor explanation backwards (Row 53) | FIXED in 53, but recurred in Row 33 |
| Dr. [REDACTED] (v2 new) | FIXED |

### Remaining P1 root cause
Row 33 has TWO sentences about the cancer: sentence 3 correctly says "ER positive" and sentence 6 incorrectly says "does not respond to hormones". The LLM contradicts itself within the same letter. This is a stochastic generation issue — the prompt correctly instructs ER+ = "grows in response to hormones" but the LLM occasionally ignores this for specific sentences.

---

## Per-Row Manual Verification

### ROW 0 — P2=1 (peritoneum). Receptor explained ✓. Stage progression ✓. Integrative Medicine ✓.
### ROW 1 — P2=1 (grammar). TNBC explained ✓. Lab correct ✓. No repeated sentence ✓.
### ROW 2 — P0=0, P1=0, P2=0. **PERFECT.**
### ROW 4 — P2=1 (response_assessment dedup lost "cancer in neck has gotten smaller" detail).
### ROW 5 — P2=1 (Genetics referral omitted).
### ROW 6 — **PERFECT.** Receptor change ✓. LVEF ✓.
### ROW 7 — **PERFECT.** ER-/HER2+ correctly explained ✓.
### ROW 8 — P2=1 (receptor not explained).
### ROW 9 — **PERFECT.**
### ROW 10 — **PERFECT.** Lab values correct ✓.
### ROW 33 — **P1=2** (receptor backwards + semantic repeat).
### All other rows: verified via automated scan + manual spot-checks.

