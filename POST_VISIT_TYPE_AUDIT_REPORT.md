# POST-VISIT-TYPE Rule Audit Report

## Rule Description
**Trigger**: Correct `"Televisit"` → `"in-person"` when note contains `"face-to-face"` text

**Triggered Rows**: 2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85

---

## Audit Summary

| Row | In Results? | Visit Type | Evidence in Note | Verdict | Concern |
|-----|-------------|------------|------------------|---------|---------|
| 2   | ✓ | in-person | **STRONG**: "I performed a face-to-face encounter with the patient" | ✓ **CORRECT** | None |
| 7   | ✓ | in-person | No clear visit type evidence found | ⚠ **UNCERTAIN** | Attribution shows unrelated text ("Pt appears to have minor degree of progression...") |
| 8   | ✓ | in-person | "I spent a total of 80 minutes face-to-face with the patient" (billing template) | ⚠ **AMBIGUOUS** | Only billing template, no clear confirmation |
| 9   | ✓ | in-person | **CONFLICTING**: Chief Complaint: "Video Visit" + "I performed this evaluation using real-time telehealth tools, including a live video Zoom connection" | ✗ **FALSE POSITIVE** | Clear telehealth visit, "face-to-face" is just billing template boilerplate |
| 49  | ✓ | in-person | No visit type evidence found | ⚠ **UNCERTAIN** | No attribution data |
| 60  | ✗ | - | Row not in results | **SKIPPED** | - |
| 62  | ✗ | - | Row not in results | **SKIPPED** | - |
| 63  | ✓ | in-person | **CONFLICTING**: "I performed this consultation using real-time Telehealth tools, including a live video connection" + "I spent 60 minutes face-to-face" (billing template) | ✗ **FALSE POSITIVE** | Clear telehealth visit, "face-to-face" is billing boilerplate |
| 79  | ✗ | - | Row not in results | **SKIPPED** | - |
| 82  | ✓ | in-person | "This time includes face-to-face time with the patient" (billing template) | ⚠ **AMBIGUOUS** | Only billing template, no contradicting telehealth evidence |
| 85  | ✓ | in-person | "Total face to face time: 45 min" (billing template) | ⚠ **AMBIGUOUS** | Only billing template, no contradicting telehealth evidence |

---

## Verdict Breakdown

- **Audited**: 8 rows (3 skipped - not in results)
- ✓ **Correct**: 1 (Row 2)
- ✗ **False Positives**: 2 (Rows 9, 63) — **25% error rate**
- ⚠ **Ambiguous** (billing template only, no telehealth evidence): 4 (Rows 8, 82, 85, + Row 7 unclear)

**Accuracy: 12.5%** (1/8 clearly correct)
**False Positive Rate: 25%** (2/8 明确误报)

---

## Key Findings

### 🚨 Critical Problem: Billing Template Confusion

**Many notes contain "face-to-face" in billing/documentation templates even for telehealth visits.**

#### Example (Row 9):
- **Chief Complaint**: "Video Visit"
- **Note states**: "I performed this evaluation using real-time telehealth tools, including a live video Zoom connection"
- **BUT also contains**: "This time includes face-to-face time with the patient" (standard billing template)
- **Result**: FALSE POSITIVE — Televisit incorrectly changed to in-person

#### Example (Row 63):
- **Note opens with**: "I performed this consultation using real-time Telehealth tools, including a live video connection"
- **Note ends with**: "I spent a total of 60 minutes face-to-face with the patient" (billing template)
- **Result**: FALSE POSITIVE

---

## Recommendation

### ❌ Do NOT apply this rule blindly

The current rule has a **25% false positive rate** because:
1. **"Face-to-face" appears in billing templates** even for telehealth visits
2. The phrase "This time includes face-to-face time with the patient" is **boilerplate text** used regardless of visit type

### ✅ Suggested Fix

**Option 1: Add Exclusion Filter**
```
IF note contains ANY of:
  - "Video Visit" (in Chief Complaint)
  - "I performed this evaluation using real-time telehealth tools"
  - "live video connection" / "Zoom connection"
  - "telehealth tools"
  - "televisit"

THEN: Do NOT apply this rule (it's a telehealth visit)
```

**Option 2: Require Stronger Evidence**
```
Only correct to "in-person" if note contains SPECIFIC phrases like:
  - "I performed a face-to-face encounter with the patient"
  - "shared visit for services provided by me"
  - "saw [patient] in clinic"

Do NOT correct based on billing templates like:
  - "This time includes face-to-face time"
  - "I spent X minutes face-to-face with the patient"
```

**Option 3: Remove This Rule Entirely**
- Let the LLM handle visit type extraction from structured note fields
- Use Chief Complaint ("Video Visit" vs "Follow-up") and explicit telehealth statements

---

## Rows Requiring Attention

### False Positives (Must Fix)
- **Row 9**: Change back to "Televisit" — clear video visit
- **Row 63**: Change back to "Televisit" — explicit telehealth consultation

### Ambiguous (Review Recommended)
- **Row 7**: Attribution evidence is unrelated to visit type, unclear how correction was triggered
- **Rows 8, 82, 85**: Only billing template evidence, but no contradicting telehealth indicators

---

## Next Steps

1. **Disable POST-VISIT-TYPE rule** until exclusion filter is added
2. **Manually review and correct Rows 9 and 63** in current results
3. **Implement Option 1 (exclusion filter)** as the safest approach
4. **Re-run verification** on the corrected rule

