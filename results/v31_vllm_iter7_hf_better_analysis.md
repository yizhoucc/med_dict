# Iter7 HF-better 59 案例人工分类

## 分类标准
- **TIE**: 实质相同，只是措辞/格式不同
- **vLLM-BETTER**: vLLM 实际更正确或更具体
- **REAL-MISS**: vLLM 真的漏了 HF 有的重要信息
- **HF-WRONG**: HF 混入了不属于本字段的内容

---

## Distant Metastasis (5)

| ROW | 判定 | 理由 |
|-----|------|------|
| 5 | TIE | HF有cervical LN, vLLM有sternum — 各有不同信息 |
| 20 | REAL-MISS | vLLM漏了"lymph nodes" |
| 50 | REAL-MISS | vLLM漏了"lymph nodes" |
| 92 | vLLM-BETTER | HF写"multiple sites"(模糊), vLLM写"liver"(具体) |
| 100 | TIE | HF"liver and multiple sites" vs vLLM"liver and bone" — vLLM更具体 |

**REAL: 2, TIE: 2, vLLM-BETTER: 1**

## Medication_Plan (9)

| ROW | 判定 | 理由 |
|-----|------|------|
| 2 | REAL-MISS | vLLM漏了Doxycycline, NS IV, potassium, pRBC |
| 11 | TIE | vLLM漏了"salt and soda rinses" — 极次要 |
| 46 | REAL-MISS | vLLM漏了gabapentin |
| 49 | TIE | 实质相同, HF多了"No specific doses"说明 |
| 50 | TIE | 不同措辞描述同一组药物 |
| 52 | TIE | 实质相同 |
| 68 | TIE | 两个都说没有计划, 措辞不同 |
| 88 | TIE | 两个都有xeloda+clinical trials, 顺序/措辞不同 |
| 95 | TIE | vLLM更简洁但信息一致 |

**REAL: 2, TIE: 7**

## Stage_of_Cancer (7)

| ROW | 判定 | 理由 |
|-----|------|------|
| 34 | REAL-MISS | vLLM漏了"now local recurrence (not metastatic)" |
| 36 | TIE | "Stage IIIA (inferred from pT3 N0)" vs "Stage IIIA (pT3N0)" — 只差"inferred from" |
| 50 | TIE | HF多了冗余的"now metastatic (Stage IV)" |
| 57 | TIE | 措辞不同, 信息一致 |
| 59 | vLLM-BETTER | vLLM"Stage IIA (pT2 N0)"比HF"Stage I (T1-2 N0)"更准确 |
| 65 | TIE | vLLM"Stage IB (pT1 N1mi)"比HF"Stage II (pT2 N1)"更精确 |
| 68 | vLLM-BETTER | vLLM"Stage I (≤2cm)"比HF"Early stage"更具体 |

**REAL: 1, TIE: 4, vLLM-BETTER: 2**

## Type_of_Cancer (6)

| ROW | 判定 | 理由 |
|-----|------|------|
| 20 | TIE | HF有receptor change history, vLLM有grade+DCIS — 各有不同信息 |
| 34 | TIE | HF有receptor change, vLLM有DCIS — 不同信息 |
| 40 | REAL-MISS | HF有精确百分比(ER 95, PR 5, HER2 2+ FISH 1.2), vLLM只有+/- |
| 44 | TIE | HF有"node+" vLLM没有, 但这属于Stage不属于Type |
| 90 | TIE | HF多了"ER/PR/HER2 not explicitly stated"说明 |
| 95 | REAL-MISS | HF有"treatment effect, three foci, margins negative" |

**REAL: 2, TIE: 4**

## genetic_testing_plan (2)

| ROW | 判定 | 理由 |
|-----|------|------|
| 68 | TIE | vLLM漏了"(if the spouse also carries a mutation)" — 次要条件 |
| 78 | TIE | 不同措辞描述同一个trial |

**REAL: 0, TIE: 2**

## imaging_plan (4)

| ROW | 判定 | 理由 |
|-----|------|------|
| 11 | TIE | vLLM漏了"due to worsening numbness" — 这是reason不是imaging plan |
| 22 | HF-WRONG | HF混入了treatment contingency"if stable continue arimidex" — 不属于imaging_plan |
| 72 | REAL-MISS | vLLM添加了"Ultrasound"但HF说"No imaging planned" — 需验证 |
| 95 | TIE | XRT是radiotherapy不是imaging, 两个都放错了 |

**REAL: 1(?), TIE: 2, HF-WRONG: 1**

## lab_plan (2)

| ROW | 判定 | 理由 |
|-----|------|------|
| 34 | REAL-MISS | vLLM说"check labs"但HF说"No labs planned" — 需验证谁对 |
| 65 | TIE | HF多了"F/u pending" |

**REAL: 1(?), TIE: 1**

## response_assessment (12)

| ROW | 判定 | 理由 |
|-----|------|------|
| 2 | TIE | HF加了"progressing"总结词, vLLM列了imaging细节 — 相似信息 |
| 5 | TIE | 非常相似内容 |
| 7 | TIE | 几乎完全一致 |
| 8 | TIE | vLLM有更多PET/CT细节, HF有更多手术背景 |
| 11 | REAL-MISS | vLLM给了未来计划(MRI ordered)而非response(PET/CT showing progression) |
| 12 | TIE | 两个都有imaging细节, 角度不同 |
| 59 | TIE | 几乎完全一致 |
| 70 | TIE | HF总结为"responding", vLLM列具体imaging — 实质相同 |
| 84 | TIE | HF加了"stable"总结, vLLM有更多imaging细节 |
| 85 | TIE | 两个都描述progression, 不同细节 |
| 88 | REAL-MISS | vLLM给了treatment plan而非exam/response findings |
| 92 | TIE | HF更全面但vLLM有AST具体值 |

**REAL: 2, TIE: 10**

## therapy_plan (12)

| ROW | 判定 | 理由 |
|-----|------|------|
| 6 | TIE | vLLM说zoladex不是letrozole — 可能不同但是同个疗程 |
| 10 | TIE | vLLM"Continue letrozole"vs HF"continue on letrozole started April 2021" |
| 34 | REAL-MISS | vLLM漏了radiation referral和return to clinic |
| 64 | TIE | vLLM有xgeva, HF有"currently on" — 不同信息 |
| 70 | TIE | HF有CT scan detail, vLLM有letrozole — 不同信息 |
| 72 | TIE | "Instructed to begin" vs "Continue" — 措辞不同 |
| 80 | REAL-MISS | vLLM漏了radiation details(6 weeks, boost, axilla) |
| 84 | TIE | 两个都有substantial content, 不同focus |
| 85 | TIE | vLLM漏了"2-week radiation washout" — 次要细节 |
| 87 | TIE | HF多了"schedule were not detailed" — 这是disclaimer不是信息 |
| 88 | TIE | 非常相似 |
| 91 | REAL-MISS | vLLM漏了lasix, KCL, elevation等项目 |

**REAL: 3, TIE: 9**

---

## 总结

| 分类 | 数量 | 占比 |
|------|------|------|
| TIE (实质相同) | **41** | 69% |
| REAL-MISS (vLLM真的漏了) | **14** | 24% |
| vLLM-BETTER (vLLM实际更好) | **3** | 5% |
| HF-WRONG (HF放错字段) | **1** | 2% |
| **总计** | **59** | 100% |

## 真正的 vLLM 遗漏 (14个)

### 可通过 POST hook 修复的 (3):
- ROW 34 therapy_plan: 漏 radiation referral
- ROW 20 Distant Met: 漏 lymph nodes
- ROW 50 Distant Met: 漏 lymph nodes

### 模型行为差异,难以系统性修复 (11):
- ROW 2 Medication_Plan: 漏多个supportive meds
- ROW 46 Medication_Plan: 漏 gabapentin
- ROW 40 Type_of_Cancer: 漏精确receptor百分比
- ROW 95 Type_of_Cancer: 漏 treatment effect details
- ROW 34 Stage_of_Cancer: 漏 local recurrence info
- ROW 80 therapy_plan: 漏 radiation details
- ROW 91 therapy_plan: 漏多个continuation items
- ROW 11 response_assessment: 给了plan而非response
- ROW 88 response_assessment: 给了plan而非response
- ROW 72 imaging_plan: false positive Ultrasound(?)
- ROW 34 lab_plan: "check labs" vs "No labs planned"(?)
