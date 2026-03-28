# Attribution Chain Review — v5 (20 samples)

## Purpose & Recovery Context

逐句审查每封 letter 的归因链完整性和准确性：
- sentence text → [source:field] 标签是否指向正确的字段
- extraction_values: 该字段的提取值是否支持 sentence 内容
- note_quotes: 原文引用是否支持提取值

**三级链路**: letter sentence → keypoint field value → original note quote

**Severity**:
- A0: 归因错误 — source tag 指向错误的字段
- A1: 链路断裂 — 有 field value 但没有 note quote（attribution 缺失）
- A2: 小问题 — quote 不精确但方向正确

**数据**: `results/letter_full_qwen_20260328_073447/progress.json`
**20 rows**: 0,1,4,5,6,8,10,13,26,28,29,33,35,40,42,49,51,53,64,94

## Progress: 20/20 COMPLETE

---

## Summary

| Metric | Value |
|--------|-------|
| Total content sentences | 223 |
| Full chain (val + quote) | 201 (**90%**) |
| A0 (wrong field tag) | **1** (Row 1: social work tagged as Specialty) |
| A1 (missing note quote) | **22** (10%) |
| Chain break rate | 10% |

### A0 Issues (1 total)
| Row | Sent | Issue |
|-----|------|-------|
| 1 | [15] | Text says "social work and home health" but [source:Specialty], VAL="Rad Onc consult". Should be [source:Others] |

Root cause: LLM chose the wrong field name from the available list. The actual content matches Others field ("Social work referral, Home health referral"), not Specialty ("Rad Onc consult").

### A1 Patterns (22 total, by field)
| Field | Count | Explanation |
|-------|-------|-------------|
| lab_summary | 6 | Attribution step often can't find a single quote for large lab panels |
| Distant_Metastasis (underscore) | 2 | LLM uses underscore tag, but keypoints use space → lookup fails |
| Type_of_Cancer | 2 | Attribution quote too generic ("metastatic breast cancer") |
| current_meds | 3 | Attribution finds treatment-change quotes, not "currently taking" quotes |
| supportive_meds | 2 | Supportive meds from medication list, not in A/P text |
| findings | 2 | Row 53 findings from exam — attribution finds different quote |
| Specialty | 1 | Integrative Medicine — not mentioned in A/P |
| imaging_plan | 1 | Brain MRI — attribution didn't find specific quote |
| procedure_plan | 1 | Row 40 — attribution missed |
| lab_plan | 1 | Row 35 — attribution missed |
| response_assessment | 1 | Row 53 — attribution missed |

### Key Findings
1. **90% full chain** — most sentences have complete sentence→value→quote traceability
2. **A1 is mostly an attribution (source_attribution.py) issue**, not a letter issue — the quotes were never found during extraction attribution
3. **lab_summary** is the most common A1 — large lab panels are hard to attribute to a single quote
4. **Distant_Metastasis vs Distant Metastasis** (underscore vs space) causes 2 chain breaks — this is a known tag-naming issue

### Perfect Attribution Rows (6/20)
5, 13, 26, 28, 51, 94

---

## Per-Row Details

### ROW 0 — A0=0, A1=2 (imaging_plan, Specialty no quotes)
### ROW 1 — A0=1 (social work→Specialty mismatch), A1=0
### ROW 4 — A1=1 (Distant_Metastasis underscore tag)
### ROW 5 — **PERFECT** attribution
### ROW 6 — A1=1 (Type_of_Cancer no quote)
### ROW 8 — A1=1 (supportive_meds)
### ROW 10 — A1=1 (lab_summary)
### ROW 13 — **PERFECT**
### ROW 26 — **PERFECT**
### ROW 28 — **PERFECT**
### ROW 29 — A1=2 (lab_summary, current_meds)
### ROW 33 — A1=4 (Type_of_Cancer, Distant_Metastasis, lab_summary, current_meds)
### ROW 35 — A1=1 (lab_plan)
### ROW 40 — A1=2 (lab_summary, procedure_plan)
### ROW 42 — A1=2 (lab_summary, supportive_meds)
### ROW 49 — A1=1 (lab_summary)
### ROW 51 — **PERFECT**
### ROW 53 — A1=3 (findings×2, response_assessment)
### ROW 64 — A1=1 (current_meds)
### ROW 94 — **PERFECT**

