# V1 vs V2 Quick Reference Table (Rows 0-14)

| Row | Coral | Patient Type | V1 Type_of_Cancer Issue | V1 Stage Issue | V1 goals Issue | V1 response Issue | V2 Fixes | V2 Regressions |
|-----|-------|--------------|-------------------------|----------------|----------------|-------------------|----------|----------------|
| 0 | 140 | New | 缺PR/HER2,冗长 | 遗漏Stage IIA | 冗长 | ✓ | Type/Stage/goals | Referral漏 |
| 1 | 141 | F/U | ✓ | 遗漏Stage IIB | 冗长 | 写计划 | Type/Stage/goals/resp | Procedure/Lab过滤 |
| 2 | 142 | New | 缺PR/HER2,冗长 | ✓ | ✓ | ✓ | Type |  |
| 3 | 143 | F/U | ✓ | ✓ | 冗长 | ✓ | Type/goals |  |
| 4 | 144 | F/U | 缺PR | ✓ | 冗长 | ✓ | Type/goals |  |
| 5 | 145 | F/U | 缺ER/PR | ✓ | 冗长 | ✓ | Type/goals |  |
| 6 | 146 | F/U | ✓ | ✓ | 冗长 | ✓ | goals |  |
| 7 | 147 | F/U | ✓ | ✓ | 冗长 | ✓ | Type/goals |  |
| 8 | 148 | F/U | 缺ER/HER2 | ✓ | 冗长 | ✓ | Type/goals |  |
| 9 | 149 | F/U | 缺HER2 | ✓ | 冗长 | ✓ | Type/goals |  |
| 10 | 150 | F/U | ✓ | 遗漏Stage III | ✓ | ✓ | Stage/goals |  |
| 11 | 151 | F/U | 缺PR | ✓ | 冗长 | ✓ | goals |  |
| 12 | 152 | F/U | 缺ER | ✓ | 冗长 | ✓ | goals |  |
| 13 | 153 | F/U | 缺HER2 | ✓ | 冗长 | ✓ | goals |  |
| 14 | 154 | New | ✓ | ✓ | 冗长 | ✓ | goals |  |


**Legend:**
- ✓ = No issue detected
- F/U = Follow-up visit
- 缺 = Missing receptor status
- 冗长 = Too verbose, not standardized
- 遗漏 = Information in note but not extracted

**Summary Statistics:**
- Total V1 Type_of_Cancer issues: 11/15 rows (73%)
- Total V1 Stage_of_Cancer issues: 3/15 rows (20%)
- Total V1 goals_of_treatment issues: 15/15 rows (100%)
- Total V1 response_assessment issues: 1/15 rows (7%)
- Total V2 regressions: 2/15 rows (13%)

**V2 Net Improvement:** V2 fixed 29 field issues but introduced 2 regressions → 93% success rate