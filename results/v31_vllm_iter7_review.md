# V31 vLLM Iter7 — 100 Sample × 11 Field 逐字审查

> 结果文件: results/v31_vllm_iter7_results.txt
> 模型: Qwen2.5-32B-Instruct-AWQ (via vLLM)
> Pipeline: run.py V2 (5 gates + POST hooks)
> 审查标准: 逐字对照原文，检查每个 field 的准确性

## 状态
- 待审查: ROW 11-100
- 已完成: 10/100
- P0: 0, P1: 2, P2: 13
- P1: response_assessment(1:ROW2误报no evidence), Stage(1:ROW4误读"5cm from nipple"为tumor size)
- P2汇总:
  - response_assessment(1): ROW6手术恢复非cancer response
  - Stage_of_Cancer(1): ROW9 pT3应为pT2(3.84cm<5cm)
  - therapy_plan(1): ROW7漏next-line treatment
  - imaging_plan(2): ROW1漏bone scan, ROW4 conditional brain MRI
  - lab_plan(2): ROW1+2混入imaging内容
  - genetic_testing_plan(1): ROW1含biopsy内容
  - Medication_Plan(1): ROW2漏多个supportive meds
  - Type_of_Cancer(1): ROW3 "HR+"应写"ER+/PR+"
  - therapy_plan(1): ROW3漏hormonal blockade讨论
  - Distant_Met(1): ROW5漏cervical LN
  - current_meds(1): ROW5 goserelin应为leuprolide

## 11 个审查字段
1. Type_of_Cancer
2. Stage_of_Cancer
3. Distant Metastasis
4. response_assessment
5. current_meds
6. goals_of_treatment
7. therapy_plan
8. imaging_plan
9. lab_plan
10. genetic_testing_plan
11. Medication_Plan

---

## ROW 1 (coral_idx 140) ✅
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+ grade 2 IDC with metastatic recurrence, HER2-" — 原文G2, ER/PR+, her2 neg, metastatic ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage IIA, now metastatic (Stage IV)" — 原文Stage IIA, 现widespread mets ✓ |
| Distant Metastasis | ✅ | "Yes, to lungs, peritoneum, liver, ovary" — 原文involvement of lungs/peritoneum/liver/ovary ✓ |
| response_assessment | ✅ | CT showing widespread mets, cancer progressing — 合理 |
| current_meds | ✅ | 空 — 原文"No current outpatient medications" ✓ |
| goals_of_treatment | ✅ | "palliative" — 原文"treatment would be palliative" ✓ |
| therapy_plan | ✅ | "ibrance and [REDACTED]" — 原文一致 ✓ |
| imaging_plan | P2 | "Brain MRI" — 漏了bone scan（bone scan错放到lab_plan里了） |
| lab_plan | P2 | 混入了"MRI of brain and bone scan" — 这些是imaging不是lab |
| genetic_testing_plan | P2 | 含biopsy内容 — biopsy是procedure不是genetic testing |
| Medication_Plan | ✅ | "ibrance and unspecified agent" — 原文*****被正确处理 ✓ |

**P0: 0 | P1: 0 | P2: 3**

## ROW 2 (coral_idx 141)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "TNBC with metastatic recurrence" — 漏了grade 3（原文有"grade 3"） |
| Stage_of_Cancer | ✅ | "Originally Stage IIB, now metastatic (Stage IV)" — 原文一致 ✓ |
| Distant Metastasis | ✅ | "liver, bone, chest wall, possibly intracranial/skull base" — PET/CT一致 ✓ |
| response_assessment | P1 | 说"No specific evidence"但05/31 PET/CT明确显示progression(significantly increased metastases)，A/P说"back pain worse, could be due to PD" |
| current_meds | ✅ | "irinotecan" — cycle 3 day 1 ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | irinotecan dose change + Rad Onc referral ✓ |
| imaging_plan | ✅ | "Scans in 3 months, MRI brain if worse" ✓ |
| lab_plan | P2 | 混入了"Scans again in 3 months"（imaging不是lab） |
| genetic_testing_plan | ✅ | "None planned" — 已有genetic testing结果，无新计划 ✓ |
| Medication_Plan | P2 | 漏了Doxycycline 100mg BID, 500ml NS IV, 40mEq potassium, 1 unit pRBC |

**P0: 0 | P1: 1 | P2: 3**

## ROW 3 (coral_idx 142) ✅
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "HR+" — 应该写"ER+/PR+"更具体；grade 2来源不明确（原文grade被redacted） |
| Stage_of_Cancer | ✅ | "Stage IIA" — staging form一致 ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" — 新诊断未治疗 ✓ |
| current_meds | ✅ | 空 — "No current outpatient medications" ✓ |
| goals_of_treatment | ✅ | "curative" — Stage IIA, no mets ✓ |
| therapy_plan | P2 | 有chemo和surgery/radiation讨论但漏了"hormonal blockade"讨论 |
| imaging_plan | ✅ | "PET scan follow up" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "Genetic testing sent and is pending" ✓ |
| Medication_Plan | ✅ | "None" — A/P无具体药物计划 ✓ |

**P0: 0 | P1: 0 | P2: 2**

## ROW 4 (coral_idx 143)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC" — 手术病理grade 2, FISH neg=HER2- ✓ |
| Stage_of_Cancer | P1 | "Stage II (inferred from 5.0cm tumor)" — 错误！"5 cm from the nipple"是位置不是大小。实际肿瘤2.8cm(pT2) |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "without any evidence of disease recurrence" — A/P一致 ✓ |
| current_meds | ✅ | "letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue Letrozole, Prolia if BMD<-2.5" ✓ |
| imaging_plan | P2 | "Brain MRI" — A/P说"If worsening, consider brain MRI"是conditional, 不是确定计划 |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | 完整列出letrozole, magnesium, calcium/VitD, Prolia contingency ✓ |

**P0: 0 | P1: 1 | P2: 1**

## ROW 5 (coral_idx 144)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC (micropapillary) with metastatic recurrence" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage III, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | P2 | 漏了left cervical LN（FNA证实转移），只列了internal mammary LN和sternum |
| response_assessment | ✅ | CT 08/23/19详细imaging findings, 尺寸变化 ✓ |
| current_meds | P2 | "goserelin"应该是"leuprolide"——A/P明确说"on leuprolide"，goserelin是早期用的 |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | 继续当前治疗+radiation referral+labs monthly ✓ |
| imaging_plan | ✅ | "CT and bone scan ordered" ✓ |
| lab_plan | ✅ | "Labs monthly" ✓ |
| genetic_testing_plan | ✅ | "None planned" — BRCA已做 ✓ |
| Medication_Plan | ✅ | "Continue leuprolide, anastrozole, palbociclib" ✓ |

**P0: 0 | P1: 0 | P2: 2**

## ROW 6 (coral_idx 145)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 1 IDC with extensive DCIS" — 手术病理一致 ✓ |
| Stage_of_Cancer | ✅ | "Stage IA (pT1 N0)" — 1.5cm, 0/1 nodes ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | P2 | 描述了手术恢复("recovering nicely, mild edema")而非癌症治疗反应 |
| current_meds | ✅ | "zoladex, letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" — Stage IA, adjuvant ✓ |
| therapy_plan | ✅ | "Start letrozole, continue zoladex 3 years, can sequence with tamoxifen" ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "Estradiol monthly" ✓ |
| genetic_testing_plan | ✅ | "None planned" — Myriad已完成(negative) ✓ |
| Medication_Plan | ✅ | 完整列出letrozole, zoladex, gabapentin, estradiol ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 7 (coral_idx 146)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR- IDC with metastatic recurrence, HER2+" — 治疗含Herceptin推断HER2+ ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage II, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "left supraclavicular node and mediastinum" ✓ |
| response_assessment | ✅ | PET-CT probable PD, SUV 2.1(was 1.8), tumor marker elevated ✓ |
| current_meds | ✅ | 空 — "has been off rx since last wk" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | P2 | 有"no hormonal therapy"和trial讨论, 但漏了next-line treatment推荐(在MedPlan里) |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "recheck [REDACTED]" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "Recommend unspecified agent as next line" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 8 (coral_idx 147) ✅
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR-/HER2+ (IHC 3+, FISH 5.7) grade 3 IDC" — 手术病理一致 ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage III, now post-NAC 3/28 LN+" — reasonable ✓ |
| Distant Metastasis | ✅ | "No" — PET/CT neg ✓ |
| response_assessment | ✅ | 手术病理: no residual breast carcinoma, 3/28 LN+, PET neg ✓ |
| current_meds | ✅ | 空 — 未在治疗中 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "adjuvant AC x 4, T-DM1, radiation" — A/P一致 ✓ |
| imaging_plan | ✅ | "echocardiogram prior to starting AC" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "adjuvant AC x 4 cycles, to be followed by T-DM1" ✓ |

**P0: 0 | P1: 0 | P2: 0** ✅ 完美

## ROW 9 (coral_idx 148)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR-/HER2- grade 2 IDC" — 手术病理一致 ✓ |
| Stage_of_Cancer | P2 | "Stage II (inferred from pT3 N1)" — pT3错误，3.84cm=pT2（但Stage II结论正确） |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | 手术病理: 3.84cm residual IDC, 5% cellularity, 1 node macro — NAC后response ✓ |
| current_meds | ✅ | 空 — 手术后未开始新治疗 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Refer for radiation. Start letrozole after radiation" ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "Letrozole after radiation. Fosamax for bone protection" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 10 (coral_idx 149) ✅
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "HR+ (ER+/PR+) HER2- grade 2 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage II" — A/P一致 ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | 正确指出无具体imaging response evidence ✓ |
| current_meds | ✅ | "letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue letrozole. Radiation to left chest wall" ✓ |
| imaging_plan | ✅ | "DEXA" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "Continue letrozole" ✓ |

**P0: 0 | P1: 0 | P2: 0** ✅ 完美

