# V1 Pipeline Review Summary (Rows 0-14)

**Date:** March 13, 2026
**Reviewer:** Automated analysis + manual verification
**Scope:** First 15 rows (coral_idx 140-154) from V1 pipeline results

---

## Executive Summary

Reviewed 15 samples from V1 pipeline (3-gate: FORMAT, FAITHFUL re-extract, TEMPORAL) and compared against V2 pipeline (6-gate: FORMAT, SCHEMA, FAITHFUL trim, TEMPORAL, SPECIFIC, SEMANTIC).

**Key Findings:**
- V1 had **40+ errors across 15 rows** (2.7 errors/row average)
- V2 reduced errors by **89%** (0.3 errors/row average)
- V2 showed **3 minor regressions** (trade-offs in conditional plan extraction)

---

## V1 Pipeline Systematic Issues

| Issue Type | Affected Rows | Percentage | Severity |
|------------|---------------|------------|----------|
| Type_of_Cancer incomplete (missing ER/PR/HER2) | 11/15 | 73% | High |
| Type_of_Cancer uses verbose diagnostic codes | 9/15 | 60% | Medium |
| Stage_of_Cancer missing explicit staging info | 4/15 | 27% | High |
| goals_of_treatment too verbose/imprecise | 10/15 | 67% | Medium |
| response_assessment answers wrong question (plans vs response) | 3/15 | 20% | High |
| Referral omission | 1/15 | 7% | Medium |

**Examples:**

### Row 0 (coral_idx=140)
- **Ground truth:** Stage IIA breast cancer, ER+/PR+/HER2-, now metastatic
- **V1 Type_of_Cancer:** "Malignant neoplasm of overlapping sites of right breast in female, estrogen receptor positive" (missing PR/HER2)
- **V1 Stage_of_Cancer:** "Not mentioned" (despite note saying Stage IIA)
- **V1 goals_of_treatment:** "cancer is not curable, but treatment can improve quality of life and extend duration of life" (should be "palliative")

### Row 1 (coral_idx=141)
- **Ground truth:** Triple negative breast cancer, Stage IIB now metastatic
- **V1 Type_of_Cancer:** "metastatic breast cancer, colon cancer, endometrial cancer" (incorrect - patient only has breast cancer)
- **V1 Stage_of_Cancer:** "not mentioned" (missing Stage IIB)
- **V1 response_assessment:** "The patient's treatment will be changed due to poor tolerance" (this is a PLAN, not a response assessment)

### Row 2 (coral_idx=142)
- **Ground truth:** Stage IIA ER+/PR+/HER2- breast cancer
- **V1 Type_of_Cancer:** "Malignant neoplasm of upper-outer quadrant of right breast in female, estrogen receptor positive" (missing PR/HER2)

---

## V2 Pipeline Improvements

### Error Reduction

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Type_of_Cancer completeness | 27% (4/15) | 93% (14/15) | +244% |
| Stage_of_Cancer accuracy | 73% (11/15) | 100% (15/15) | +37% |
| goals_of_treatment standardization | 33% (5/15) | 100% (15/15) | +200% |
| response_assessment semantic correctness | 80% (12/15) | 100% (15/15) | +25% |
| Average errors per row | 2.7 | 0.3 | -89% |

### Architectural Improvements

**1. Field Splitting (4→8 prompts):**
- Reduces cognitive load on 8B model
- `What_We_Found` (5 fields) → `Cancer_Diagnosis` (3) + `Lab_Results` (1) + `Clinical_Findings` (1)
- `Treatment_Summary` (3 fields) → `Current_Medications` (1) + `Treatment_Changes` (2)
- `Goals_of_care` (2 fields) → `Treatment_Goals` (1) + `Response_Assessment` (1)

**2. Chain-of-Thought for Hard Fields:**
- `Current_Medications`: CoT to distinguish CURRENT/PLANNED/PAST
- `Treatment_Goals`: CoT decision tree (early+adjuvant→curative, metastatic→palliative)
- `Response_Assessment`: CoT to first check if treatment started

**3. 6-Gate Verification Chain:**
- G2 (SCHEMA): Validates field names, prevents hallucinated fields
- G3 (FAITHFUL): "Keep if uncertain" strategy, only empties contradictory values
- G5 (SPECIFIC): Replaces vague terms ("staging workup" → specific tests)
- G6 (SEMANTIC): Checks if value answers the field's question

---

## V2 Regressions (Trade-offs)

| Row | Field | Issue | Analysis |
|-----|-------|-------|----------|
| 0 | Referral | V1 correctly identified "Rad Onc", V2 missed | Redacted text caused recognition failure |
| 1 | Procedure_Plan | V1 extracted "MRI brain if worse", V2 said "None" | V2's TEMPORAL gate filtered conditional plans |
| 1 | Lab_Plan | V1 extracted lab orders, V2 said "None" | Same - conditional plans filtered |

**Total:** 3 regressions vs 40+ improvements → **92% net improvement rate**

---

## Architecture Comparison

| Dimension | V1 Pipeline | V2 Pipeline |
|-----------|-------------|-------------|
| Number of prompts | 4 multi-field | 8 focused |
| Max fields per prompt | 5 (What_We_Found) | 4 (Reason_for_Visit) |
| Chain-of-Thought | None | 3 hard fields |
| Number of gates | 3 (FORMAT, FAITHFUL, TEMPORAL) | 6 (FORMAT, SCHEMA, FAITHFUL, TEMPORAL, SPECIFIC, SEMANTIC) |
| FAITHFUL strategy | Re-extract (redo) | Trim (fix) |
| Schema validation | None | G2 checks field names |
| Semantic validation | None | G6 checks answer relevance |
| Specificity check | None | G5 replaces vague terms |

---

## Recommendations

1. **Adopt V2 pipeline** - 89% error reduction, significant accuracy gains on key fields
2. **Monitor Referral field** - Consider enhancing inference capability for redacted text
3. **Evaluate conditional plan handling** - Adjust TEMPORAL gate based on downstream task needs
4. **Continue prompt optimization** - V2's field splitting and CoT design proven effective

---

## Detailed Row-by-Row Analysis

See full Chinese report: `v1_pipeline_review_rows_0_14.md` (411 lines)

**Report includes:**
- Original note key facts for each row
- Field-by-field comparison (V1 vs V2)
- Error identification and categorization
- V2 improvement assessment

---

**Files:**
- Full report: `/Users/yizhoucc/repo/med_dict/results/v1_pipeline_review_rows_0_14.md`
- This summary: `/Users/yizhoucc/repo/med_dict/results/v1_pipeline_review_summary.md`
