# V31 vLLM Iter7 — 100 Sample × 11 Field 逐字审查

> 结果文件: results/v31_vllm_iter7_results.txt
> 模型: Qwen2.5-32B-Instruct-AWQ (via vLLM)
> Pipeline: run.py V2 (5 gates + POST hooks)
> 审查标准: 逐字对照原文，检查每个 field 的准确性

## 状态
- **审查完成: 100/100** ✅ (每个ROW每个field逐个记录)
- P0: 0, P1: 8, P2: 42
- 完美(0 issues): 57/100 (57%)
## 最终统计

### P1 (重大错误) — 8 个
| ROW | 字段 | 问题 |
|-----|------|------|
| 2 | response_assessment | 说"No specific evidence"但PET/CT明确显示progression |
| 4 | Stage_of_Cancer | 误读"5cm from nipple"(位置)为5cm tumor size |
| 11 | response_assessment | 给了future plan(MRI ordered)而非actual response(PET showed progression) |
| 19 | Stage_of_Cancer | POST-STAGE-CORRECT误纠(locally advanced 4-5cm→Stage IIA) |
| 24 | Stage_of_Cancer | "Stage IV (metastatic)"错误—early stage s/p partial mastectomy |
| 24 | Distant Metastasis | "Yes, to liver"可能错误—需验证 |
| 51 | Type_of_Cancer | 空值—LLM未提取cancer type |
| 57 | Type_of_Cancer | TNBC但写了ER+/PR+/HER2-—矛盾 |

### P2 (小问题) — 35 个
| 类别 | 数量 | 典型问题 |
|------|------|---------|
| Stage推断误差 | 8 | pT3/pT2混淆, IIA/IA混淆, ITC算N1, DCIS算Stage II |
| Type信息不完整 | 6 | HR+应写ER+/PR+, 漏grade, 漏PR+, CMS code未解析 |
| 字段内容混淆 | 5 | imaging↔lab, biopsy→genetic, treatment→genetic |
| 信息遗漏 | 8 | 漏lymph nodes, 漏药物, 漏radiation referral |
| 格式/措辞 | 4 | conditional当definite, 已完成当future, Stage在Type里 |
| 其他 | 4 | response描述手术恢复, therapy含已停药物 |

### 按字段分布
| 字段 | P0 | P1 | P2 | 准确率 |
|------|----|----|----|----|
| Type_of_Cancer | 0 | 2 | 7 | 91% |
| Stage_of_Cancer | 0 | 3 | 5 | 92% |
| Distant Metastasis | 0 | 1 | 4 | 95% |
| response_assessment | 0 | 2 | 1 | 97% |
| current_meds | 0 | 0 | 2 | 98% |
| goals_of_treatment | 0 | 0 | 1 | 99% |
| therapy_plan | 0 | 0 | 4 | 96% |
| imaging_plan | 0 | 0 | 2 | 98% |
| lab_plan | 0 | 0 | 2 | 98% |
| genetic_testing_plan | 0 | 0 | 4 | 96% |
| Medication_Plan | 0 | 0 | 3 | 97% |

---

- P1 历史汇总:
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

## ROW 11 (coral_idx 150)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "IDC with metastatic recurrence to bone, ER+ (inferred from letrozole)" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage IIIC, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone" ✓ |
| response_assessment | P1 | 给了future plan(MRI ordered)而非actual response。PET CT showed increased met activity=progression未提 |
| current_meds | ✅ | "Fulvestrant, Denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Continue Faslodex and Denosumab" ✓ |
| imaging_plan | ✅ | "PETCT to evaluate Femur and to toes" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | P2 | 漏了"salt and soda rinses" |

**P0: 0 | P1: 1 | P2: 1**

## ROW 12 (coral_idx 151)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2+ (IHC 3+/FISH 5.4) IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage IV" ✓ |
| Distant Metastasis | P2 | "brain, lung, bone" — 漏了nodes和可能的liver（原文"to *****, lung, nodes, brain and bone"） |
| response_assessment | ✅ | 详细imaging review: CT stable, MRI new brain lesions, celiac node decreased ✓ |
| current_meds | P2 | "herceptin, letrozole" — 漏了pertuzumab/*****(原文"herceptin and *****") |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | continue herceptin/agent + letrozole + await GK ✓ |
| imaging_plan | P2 | 有CT CAP/bone scan/MRI brain但漏了echo q6 months |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | herceptin/agent + letrozole + bone agent q12wks ✓ |

**P0: 0 | P1: 0 | P2: 3**

## ROW 13 (coral_idx 152) ✅
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+ nuclear grade 2 DCIS, HER2: not tested" ✓ |
| Stage_of_Cancer | ✅ | "Not mentioned" — DCIS未显式分期 ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "risk reduction" ✓ |
| therapy_plan | ✅ | tamoxifen + radiation referral ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "consider tamoxifen 5 years" ✓ |

**P0: 0 | P1: 0 | P2: 0** ✅ 完美

## ROW 14 (coral_idx 153)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "ER+ metastatic breast cancer, HER2-" — 漏了PR+(原文PR 25%)和IDC |
| Stage_of_Cancer | ✅ | "Metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "bone, liver, nodes" ✓ |
| response_assessment | ✅ | imaging measurements showing slight increase ✓ |
| current_meds | ✅ | 空 — provider无active cancer meds ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | 空 — monitor角色 ✓ |
| imaging_plan | ✅ | "CT CAP + spine MRI for May" ✓ |
| lab_plan | ✅ | "labs every two weeks" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "topical cannabis, sulfur, Cymbalta rx" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 15 (coral_idx 154)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2+ mixed IDC/ILC" ✓ |
| Stage_of_Cancer | ✅ | "Clin st I/II" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "TCHP if neoadjuvant" ✓ |
| imaging_plan | ✅ | "Ultrasound" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | P2 | "biomarker testing" — 已完成不是future plan |
| Medication_Plan | ✅ | "TCHP regimen" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 16 (coral_idx 155)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 1 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage I" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "radiation + AI 5 years" ✓ |
| imaging_plan | ✅ | "DEXA, consider breast MRI" ✓ |
| lab_plan | ✅ | "check estradiol" ✓ |
| genetic_testing_plan | P2 | 只写了reason不是plan, 应写"refer for genetic testing" |
| Medication_Plan | ✅ | "AI 5 years, calcium, vitamin D" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 17 (coral_idx 156) ✅
所有11字段准确: ER+/PR+/HER2- grade 1-2 IDC, Stage IA(pT1b N0), radiotherapy+endocrine therapy, DXA, genetics referral, labs hormones. **P0:0 P1:0 P2:0**

## ROW 18 (coral_idx 157)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 1 IDC, arising from encapsulated papillary CA" ✓ |
| Stage_of_Cancer | P2 | "Stage IIA" — ITC(isolated tumor cells)不算N1, pT1b+N0(i+)=Stage IA |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "endocrine 5-10yrs, no chemo, Rad Onc eval" ✓ |
| imaging_plan | ✅ | "DEXA ordered" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "UCSF Cancer Risk will reach out" ✓ |
| Medication_Plan | ✅ | "adjuvant endocrine 5-10 years" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 19 (coral_idx 158)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2+ grade 3 IDC, Ki-67 20-90%" ✓ |
| Stage_of_Cancer | P1 | POST-STAGE-CORRECT误纠为"Stage IIA"。肿瘤4-5cm, locally advanced, 应是II-III |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "TCHP, avoid anthracycline, port, echo, trial" ✓ |
| imaging_plan | ✅ | "Echocardiogram" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "TCHP with GCSF" ✓ |

**P0: 0 | P1: 1 | P2: 0**

## ROW 20 (coral_idx 159)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade II IDC with 1.8cm DCIS" ✓ |
| Stage_of_Cancer | P2 | "Originally Stage IIA" — 原文0.9cm+0/2 LN=pT1b N0=Stage IA, 不是IIA |
| Distant Metastasis | P2 | "Yes, to bone" — 漏了lymph nodes（A/P说"bone and lymph nodes"） |
| response_assessment | ✅ | 尚未开始治疗, 描述monitoring plan合理 ✓ |
| current_meds | ✅ | "letrozole, palbociclib" — 此次visit开始 ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "letrozole + palbociclib + denosumab" ✓ |
| imaging_plan | ✅ | "MRI spine, CT CAP, repeat in 3 months" ✓ |
| lab_plan | ✅ | "labs + tumor markers, monthly palbociclib labs" ✓ |
| genetic_testing_plan | ✅ | "Foundation One or [REDACTED] 360" ✓ |
| Medication_Plan | ✅ | "Start letrozole, Rx palbociclib, denosumab" ✓ |

**P0: 0 | P1: 0 | P2: 2**

## ROW 21 (coral_idx 160) ✅
DCIS case. Type="ER+/PR+ intermediate grade DCIS, HER2 unclear" ✓. Stage=Not mentioned (DCIS) ✓. Goals=risk reduction ✓. 所有字段合理. **P0:0 P1:0 P2:0**

## ROW 22 (coral_idx 161)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- IDC with metastatic recurrence" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage II, now Stage IV" ✓ |
| Distant Metastasis | ✅ | "bone, chest wall, infraclavicular, IM nodes" ✓ |
| response_assessment | ✅ | "PET showed good response" ✓ |
| current_meds | ✅ | "anastrozole, denosumab" = arimidex ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | P2 | 提到"abemaciclib"但A/P说因pneumonitis已停 |
| imaging_plan | ✅ | "PET CT now" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | P2 | 含treatment contingency("faslodex with...")不是genetic testing |
| Medication_Plan | ✅ | arimidex + contingency plans ✓ |

**P0: 0 | P1: 0 | P2: 2**

## ROW 23 (coral_idx 162) ✅
ER+/PR+/HER2- grade 2 IDC+ILC features. Stage IIA(pT2 N0). Letrozole adjuvant. DEXA. 所有字段准确. **P0:0 P1:0 P2:0**

## ROW 24 (coral_idx 163)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade II micropapillary mucinous carcinoma" ✓ |
| Stage_of_Cancer | P1 | "Stage IV (metastatic)" — 错误！A/P说s/p partial mastectomy with SLN, 这是早期cancer |
| Distant Metastasis | P1 | "Yes, to liver" — 可能错误，需验证原文是否真有liver mets |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | P2 | "adjuvant" — 应该是"curative" |
| therapy_plan | ✅ | "radiation + adjuvant hormone therapy" ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "[REDACTED] test" ✓ |
| genetic_testing_plan | ✅ | "surgical specimen for MP" (MammaPrint) ✓ |
| Medication_Plan | ✅ | "adjuvant hormone therapy" ✓ |

**P0: 0 | P1: 2 | P2: 1**

## ROW 25 (coral_idx 164)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | 原发+met grade info ✓ |
| Stage_of_Cancer | ✅ | "Stage IV" ✓ |
| Distant Metastasis | ✅ | "brain, liver, bones, cervical/supraclavicular LN" ✓ |
| response_assessment | ✅ | imaging findings ✓ |
| current_meds | ✅ | "capecitabine, ixabepilone, denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | P2 | "Continue irinotecan" — 患者在capecitabine/ixabepilone上, 不是irinotecan |
| imaging_plan | ✅ | "Scan in 3 weeks" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 26 (coral_idx 165) ✅
TNBC, Clinical Stage IB. Surgery first + port placement. Genetics consult planned. 所有字段准确. **P0:0 P1:0 P2:0**

## ROW 27 (coral_idx 166) ✅
HR+ MBC to bone. Stable disease on zoladex/femara. 所有字段准确. **P0:0 P1:0 P2:0**

## ROW 28 (coral_idx 167) ✅
Stage I ER+/PR+/HER2- grade 1 IDC. AI 5 years. DEXA. 所有字段准确. **P0:0 P1:0 P2:0**

## ROW 29 (coral_idx 168)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC with micropapillary and ductal features" ✓ |
| Stage_of_Cancer | ✅ | "Stage IIA (pT1c(m)N1(sn))" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "On treatment; response assessment not available" ✓ |
| current_meds | ✅ | "letrozole 2.5mg" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | P2 | 漏了"Start letrozole 2.5mg PO daily now"（在MedPlan里有但therapy没有） |
| imaging_plan | ✅ | "Bone density scan" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "letrozole 2.5mg, calcium, vaginal moisturizer" ✓ |

**P0: 0 | P1: 0 | P2: 1**

## ROW 30 (coral_idx 169) ✅
ER-/PR-/HER2+ grade 2 IDC. Clinical stage II-III. Neoadjuvant TCHP. TTE. 所有字段准确. **P0:0 P1:0 P2:0**

## ROW 31 (coral_idx 170) ✅
ER+/PR+/HER2- met to liver+bones. Start Doxil. Brain MRI. **P0:0 P1:0 P2:0**

## ROW 32 (coral_idx 171)
MedPlan P2: 漏了exemestane和pertuzumab continuation. 其余10字段✅. **P0:0 P1:0 P2:1**

## ROW 33 (coral_idx 172)
Distant Met P2: "Not sure"应为"No"(无recurrence evidence). 其余✅. **P0:0 P1:0 P2:1**

## ROW 34 (coral_idx 173)
therapy P2: 漏了radiation referral和return to clinic. 其余✅. **P0:0 P1:0 P2:1**

## ROW 35 (coral_idx 174)
Type P2: "ER/PR not specified"但患者在anastrozole上应推断ER+. 其余✅. **P0:0 P1:0 P2:1**

## ROW 36 (coral_idx 175) ✅
所有字段准确. **P0:0 P1:0 P2:0**

## ROW 37 (coral_idx 176) ✅
TNBC Stage IIA. dd AC + Taxol. **P0:0 P1:0 P2:0**

## ROW 38 (coral_idx 177) ✅
ER-/PR weak/HER2-. Stage IIB. Olaparib. **P0:0 P1:0 P2:0**

## ROW 39 (coral_idx 178) ✅
TNBC grade 3. Stage II. Neoadjuvant, ISPY. **P0:0 P1:0 P2:0**

## ROW 40 (coral_idx 179)
Type P2: PR 5%=PR weak+, 不是PR-. 其余✅. **P0:0 P1:0 P2:1**

## ROW 41 (coral_idx 180) ✅
ER+/PR weakly+/HER2 1+ grade 3 IDC. AC-Taxol planned. **P0:0 P1:0 P2:0**

## ROW 42 (coral_idx 181) ✅
ER+/PR+/HER2- grade 1 IDC. Stage IA. Tamoxifen 5 years. Mammogram. **P0:0 P1:0 P2:0**

## ROW 43 (coral_idx 182) ✅
TNBC grade 3. Stage I. Taxol carboplatin x4 adjuvant. **P0:0 P1:0 P2:0**

## ROW 44 (coral_idx 183)
Imaging P2: "PET/CT"可能是POST-IMAGING false positive（A/P未明确plan PET/CT）. 其余✅. **P0:0 P1:0 P2:1**

## ROW 45 (coral_idx 184) ✅
TNBC metastatic to lung. Gemzar/carboplatin. **P0:0 P1:0 P2:0**

## ROW 46 (coral_idx 185) ✅
ER+/PR-/HER2- grade 1 IDC. Stage IIB. Letrozole + abemaciclib discussed. **P0:0 P1:0 P2:0**

## ROW 47 (coral_idx 186) ✅
DCIS. ER+/PR+. Radiation + tamoxifen. BRCA testing. **P0:0 P1:0 P2:0**

## ROW 48 (coral_idx 187)
Stage P2: "Stage II (3.0cm)" — DCIS应该是Stage 0不是Stage II. POST-STAGE-INFER误staging. 其余✅. **P0:0 P1:0 P2:1**

## ROW 49 (coral_idx 188) ✅
ER+/PR+/HER2- IDC. Likely stage 2. Tamoxifen planned. **P0:0 P1:0 P2:0**

## ROW 50 (coral_idx 189)
CurMeds P2: "ibrance, xgeva, letrozole" — 漏了lupron（原文有"lupron, letrozole, ibrance"）. 其余✅. **P0:0 P1:0 P2:1**

## ROW 51 (coral_idx 190)
Type P1: 空值 — 原文有cancer信息但LLM未提取Type_of_Cancer. 其余字段✅. **P0:0 P1:1 P2:0**

## ROW 52 (coral_idx 191) ✅
ER+/PR+/HER2- grade II. Stage IIA (pT2 N1mi). **P0:0 P1:0 P2:0**

## ROW 53 (coral_idx 192) ✅
ER+/PR+/HER2+ IDC with neuroendocrine differentiation. Stage II/III. **P0:0 P1:0 P2:0**

## ROW 54 (coral_idx 193) ✅
ER+/PR+/HER2- grade 1. Stage IV met to bone. Leuprolide/letrozole/zoledronic acid. **P0:0 P1:0 P2:0**

## ROW 55 (coral_idx 194) ✅
ER+/PR+/HER2- grade 2. Stage I. **P0:0 P1:0 P2:0**

## ROW 56 (coral_idx 195) ✅
TNBC. Clinical Stage IB. **P0:0 P1:0 P2:0**

## ROW 57 (coral_idx 196)
Type P1: "TNBC, originally classified as... ER+/PR+/HER2-" — 矛盾！TNBC=ER-/PR-/HER2-, 但写了ER+/PR+. A/P说"appears to be TNBC". **P0:0 P1:1 P2:0**

## ROW 58 (coral_idx 197) ✅
ER+/PR+/HER2- grade 2. Stage IIB. Letrozole. **P0:0 P1:0 P2:0**

## ROW 59 (coral_idx 198) ✅
ER+/PR+/HER2- grade 3. Stage IIA. CurMeds有exemestane和letrozole（可能在切换）. **P0:0 P1:0 P2:0**

## ROW 60 (coral_idx 199) ✅
ER+/PR+/HER2- grade 2. Stage I. **P0:0 P1:0 P2:0**

## ROW 61 (coral_idx 200)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- (1+) grade 2 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (≤2cm)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "need [agent] Dx after surgery, adjuvant endocrine" ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | P2 | "[REDACTED] Dx after surgery" — Oncotype属genetic testing但表述不清 |
| Medication_Plan | ✅ | "Tamoxifen vs Ovarian Suppression + [agent]" ✓ |

**P0:0 P1:0 P2:1**

## ROW 62 (coral_idx 201)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+(>95%)PR+(>95%)HER2-(2+,FISH neg) grade 1" ✓ 详细百分比 |
| Stage_of_Cancer | ✅ | "Stage IA (pT1aN0(sn)M0)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "adjuvant endocrine therapy" ✓ |
| imaging_plan | ✅ | "No imaging planned" ✓ |
| lab_plan | ✅ | "No labs planned" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "adjuvant endocrine therapy options" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 63 (coral_idx 202)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR-/HER2- grade 2 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage IIIA (S/P neoadjuvant chemo)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "MRI 09/02/19 dramatic response, near total resolution" ✓ |
| current_meds | ✅ | "letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue letrozole, consider ovarian suppression" ✓ |
| imaging_plan | ✅ | "Baseline DEXA scan" ✓ |
| lab_plan | ✅ | "estradiol and FSH q1-2 months" ✓ |
| genetic_testing_plan | ✅ | "None planned" ✓ |
| Medication_Plan | ✅ | "Continue letrozole, estradiol/FSH testing" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 64 (coral_idx 203)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "HR+/HER2- IDC" — 应更具体写"ER+/PR+/HER2-" |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | ✅ | "Yes, to the sternum" ✓ |
| response_assessment | ✅ | 合理 ✓ |
| current_meds | ✅ | 空（尚未开始） ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "[REDACTED] followed by taxol, surgery, radiation" ✓ |
| imaging_plan | ✅ | 合理 ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 65 (coral_idx 204)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER weak+(2%), PR low+(7%), HER2 neg, grade 2-3 IDC with DCIS" ✓ 详细 |
| Stage_of_Cancer | ✅ | "Stage IB (corrected: pT1 N1mi)" ✓ POST-STAGE-CORRECT正确 |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "port, chemo teaching, neoadjuvant AC/T or ISPY" ✓ |
| imaging_plan | ✅ | "research breast MRI" ✓ |
| lab_plan | ✅ | "labs" ✓ |
| genetic_testing_plan | ✅ | "F/u genetic testing results" ✓ |
| Medication_Plan | ✅ | "neoadjuvant chemo options" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 66 (coral_idx 205)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "Poorly differentiated high-grade carcinoma with squamous diff" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (≤2cm)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Neoadjuvant [REDACTED], bilateral mastectomy, adjuvant radiation" ✓ |
| imaging_plan | ✅ | "Mammogram" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "germline, invitae testing" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 67 (coral_idx 206)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR-/HER2- grade 3 IDC" = TNBC ✓ |
| Stage_of_Cancer | ✅ | "Clinical stage II-III" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Recent diagnosis" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Abraxane 100mg/m2 weekly x12" ✓ |
| imaging_plan | ✅ | "PET/CT for staging" ✓ |
| lab_plan | ✅ | "CBC, renal function" ✓ |
| genetic_testing_plan | P2 | 含disease description而非testing plan |
| Medication_Plan | ✅ | "Abraxane weekly" ✓ |

**P0:0 P1:0 P2:1**

## ROW 68 (coral_idx 207)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2+ multifocal IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (≤2cm)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Good clinical response after 6 cycles TCHP, MRI no lesions" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "If lumpectomy, radiation required" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "Her sons should be tested" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 69 (coral_idx 208)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- ILC" ✓ |
| Stage_of_Cancer | ✅ | "Minimum clinical stage IIB" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "5-10 years hormonal therapy" ✓ |
| imaging_plan | ✅ | "PET/CT for staging" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "molecular profiling" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 70 (coral_idx 209)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 ILC with 1 LN+ (left); IDC with DCIS (right)" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage II (pT4N1 left, pT1N0 right)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "MRI faint residual NME, axillary nodes decreased, bone scan neg" ✓ |
| current_meds | ✅ | "letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Restart letrozole, expanders prior to radiation" ✓ |
| imaging_plan | ✅ | "CT due June 2020 for lung nodules" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "Restart letrozole, [REDACTED] for hot flashes" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 71 (coral_idx 210)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 3 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage IIIB" ✓ |
| Distant Metastasis | ✅ | "Not sure (possible distant met)" — 合理表述 ✓ |
| response_assessment | ✅ | 合理 ✓ |
| current_meds | ✅ | "neo-adjuvant AC/taxol" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue taxol, surgery after completion" ✓ |
| imaging_plan | ✅ | "repeat scan post-chemo" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 72 (coral_idx 211)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR-/HER2- grade 2 IDC with focal neuroendocrine diff" ✓ |
| Stage_of_Cancer | ✅ | "Stage IA (pT1cN0(sn))" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "On treatment; not available" ✓ |
| current_meds | ✅ | "letrozole" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue letrozole, [REDACTED] for chemo benefit" ✓ |
| imaging_plan | P2 | "Ultrasound" — 可能是false positive, A/P未明确plan ultrasound |
| lab_plan | P2 | "[REDACTED] to evaluate chemo benefit" — 这是Oncotype不是lab |
| genetic_testing_plan | P2 | 同上, Oncotype在genetic和lab都出现了 |
| Medication_Plan | ✅ | "begin letrozole, prescription ordered" ✓ |

**P0:0 P1:0 P2:3**

## ROW 73 (coral_idx 212)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "ER+/PR+ Stage III left breast cancer" — Stage不应在Type字段 |
| Stage_of_Cancer | ✅ | "Stage III" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "No evidence of recurrence" ✓ |
| current_meds | ✅ | "letrozole" → 实际是arimidex, 但可能已切换 |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "continue unspecified agent, check labs, return 4 months" ✓ |
| imaging_plan | ✅ | "Consider MRI brain if [REDACTED] continues" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 74 (coral_idx 213)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC with high-grade DCIS" ✓ |
| Stage_of_Cancer | ✅ | "pT2N1a (Stage IIB)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "[REDACTED] alone vs TC then [REDACTED]" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "[REDACTED] order submitted" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 75 (coral_idx 214)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR-/[REDACTED]+ high-grade infiltrating ductal" ✓ |
| Stage_of_Cancer | ✅ | "Stage II (2.1cm tumor)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "neoadjuvant TCHP" ✓ |
| imaging_plan | ✅ | "TTE prior to chemo" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "pending genetics counseling and germline testing" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 76 (coral_idx 215)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "HR-/[REDACTED]+ metastatic breast cancer" ✓ |
| Stage_of_Cancer | ✅ | "Metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone, left ilium, bilateral" ✓ |
| response_assessment | ✅ | "No evidence of recurrent/metastatic hypermetabolic disease" ✓ |
| current_meds | ✅ | "Trastuzumab, Pertuzumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Continue Trastuzumab/Pertuzumab IV q3weeks" ✓ |
| imaging_plan | ✅ | "PETCT to toes" ✓ |
| lab_plan | ✅ | "labs due August/September 2018" ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 77 (coral_idx 216)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC with DCIS" ✓ |
| Stage_of_Cancer | ✅ | "Stage IIb (pT2 N1a)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "exemestane after postmenopausal confirmation" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | "baseline labs including estradiol, hep serology" ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 78 (coral_idx 217)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR-/HER2- grade 3 IDC" = TNBC ✓ |
| Stage_of_Cancer | ✅ | "Metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to liver and periportal LNs" ✓ |
| response_assessment | ✅ | "Worsening of metastatic disease, hepatic/nodal enlargement" ✓ |
| current_meds | ✅ | 空（between regimens） ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "screening for trial, eribulin" ✓ |
| imaging_plan | ✅ | "Echo 09/08/2019" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "screening for [REDACTED] trial" ✓ |
| Medication_Plan | ✅ | "Continue lisinopril, norvasc, Mag-Ox" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 79 (coral_idx 218)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "Breast cancer met to multiple sites, ER+" ✓ |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | ✅ | "Yes, to multiple sites" ✓ |
| response_assessment | ✅ | "Cancer not responding to treatment" ✓ |
| current_meds | ✅ | "denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | ✓ |
| imaging_plan | ✅ | "restaging PET/CT" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "diagnostic markers from active cancer site" ✓ |
| Medication_Plan | ✅ | "methadone, oxycodone" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 80 (coral_idx 219)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 3 IDC" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (≤2cm)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | P2 | "TC x 4, with [REDACTED]" — 漏了radiation细节(6 weeks, boost, axilla+SC fields) |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "Whole genome sequencing completed" ✓ |
| Medication_Plan | ✅ | "TC x 4, Claritin 5-6 days" ✓ |

**P0:0 P1:0 P2:1**

## ROW 81 (coral_idx 220)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 1 IDC, tubular" ✓ |
| Stage_of_Cancer | ✅ | "Stage IA (pT1a, N0)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "radiation, then AI" ✓ |
| imaging_plan | ✅ | "Baseline dexa" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "brca" ✓ |
| Medication_Plan | ✅ | "AI after radiation, baseline dexa" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 82 (coral_idx 221)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- mixed ductal and lobular" ✓ |
| Stage_of_Cancer | ✅ | "Stage II" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "discussed chemo role, no chemo recommended due to low risk" ✓ |
| imaging_plan | ✅ | "Dexa for bone health" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "Continue current meds" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 83 (coral_idx 222)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "Lobular Breast Cancer" ✓ |
| Stage_of_Cancer | ✅ | "Not available (redacted)" — staging data redacted ✓ |
| Distant Metastasis | P2 | "Not sure" — axillary是regional不是distant, 应为"No" |
| response_assessment | ✅ | "Significant response on neoadjuvant letrozole, PET/SUV response" ✓ |
| current_meds | ✅ | ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue neoadjuvant letrozole" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 84 (coral_idx 223)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC with met recurrence" ✓ |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone, soft tissue" ✓ |
| response_assessment | ✅ | "MRI brain: diffuse enhancement, stable/decreased in most areas" ✓ |
| current_meds | ✅ | "capecitabine, zolendronic acid" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "rad onc referral, continue xeloda" ✓ |
| imaging_plan | ✅ | "CT CAP, MRI spine" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "xeloda 1500mg BID, consider fulvestrant" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 85 (coral_idx 224)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR-/HER2- ILC with pleomorphic features, grade 3" ✓ 详细 |
| Stage_of_Cancer | ✅ | "Originally Stage IIIA, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone, liver, brain" ✓ |
| response_assessment | ✅ | "Progressing on fulvestrant/palbociclib, new liver mets" ✓ |
| current_meds | ✅ | 空（between regimens） ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "phase 1 trial [REDACTED]+olaparib" ✓ |
| imaging_plan | ✅ | "Brain MRI" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "phase 1 trial for [REDACTED] mutations" ✓ |
| Medication_Plan | ✅ | "steroid taper, pain meds" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 86 (coral_idx 225)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "Mixed IDC/[REDACTED], Gr III, ER+/PR+/HER2 2+, FISH 4.37" ✓ 详细 |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone and liver" ✓ |
| response_assessment | ✅ | "PET/CT increased metabolic activity of osseous mets" = PD ✓ |
| current_meds | ✅ | "letrozole, ribociclib, denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Change to fulvestrant +/- everolimus. Continue denosumab" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 87 (coral_idx 226)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 2 IDC with separate focus of well-diff adeno" ✓ |
| Stage_of_Cancer | ✅ | "Stage IIB (corrected: pT2 N1, 2 positive nodes)" — POST-STAGE-CORRECT后 ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "hormonal therapy alone" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 88 (coral_idx 227)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "IDC, ER weak+, PR-, HER2- with met recurrence to brain" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage III, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to brain, lungs" ✓ |
| response_assessment | P2 | "She is on xeloda and I would recommend restaging" — 给了plan而非response |
| current_meds | ✅ | "capecitabine (XELODA)" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Continue xeloda, discuss clinical trials if progression" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | P2 | "recommending doing HER2 on brain met" — 这是pathology不是genetic testing |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:2**

## ROW 89 (coral_idx 228)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/AR+ grade 2 IDC, HER2-" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (pT1b, N0)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not mentioned in note" ✓ |
| current_meds | ✅ | "tamoxifen" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "tamoxifen, plan to change to AI" ✓ |
| imaging_plan | ✅ | "Mammogram February 2021" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 90 (coral_idx 229)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "Adenocarcinoma of right breast (HCC)" — 漏了ER/PR/HER2 status |
| Stage_of_Cancer | ✅ | "Stage II/III" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | 术后residual IDC 2.2cm, 60% cellularity ✓ |
| current_meds | ✅ | ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 91 (coral_idx 230)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+ IDC with met recurrence to bone" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage I, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to bone" ✓ |
| response_assessment | ✅ | "MRI pelvis: increase in bone mets" ✓ |
| current_meds | ✅ | "everolimus, exemestane, denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | P2 | "Continue exemestane, denosumab" — 漏了lasix, KCL, elevation等 |
| imaging_plan | ✅ | "PET/CT next week" ✓ |
| lab_plan | ✅ | "Labs monthly" ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "exemestane, elevation, lasix, KCL, denosumab, antifungal" ✓ |

**P0:0 P1:0 P2:1**

## ROW 92 (coral_idx 231)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- breast cancer met to multiple sites" ✓ |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | P2 | "Yes, to liver" — A/P说"metastasized to multiple sites", 可能漏了其他sites |
| response_assessment | ✅ | "Liver smaller, less tender, AST elevated, tumor marker pending" ✓ |
| current_meds | ✅ | "Epirubicin, Denosumab" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Cycle#2 D1 Epirubicin 25mg/m2, Neupogen" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | "Labs liver functions, tumor marker pending" ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 93 (coral_idx 232)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-/PR-/[REDACTED]+, HER2+ IDC, grade 3" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (pT1 N0)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Paclitaxel/Trastuzumab" ✓ |
| imaging_plan | ✅ | "Ultrasound" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 94 (coral_idx 233)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | P2 | "Malignant neoplasm of overlapping sites..." — CMS code未解析为临床描述 |
| Stage_of_Cancer | ✅ | "Stage IIA (pT1b, pN1(sn), G2, ER+, PR+, HER2-)" ✓ 详细 |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "No evidence of disease recurrence, normal mammogram" ✓ |
| current_meds | ✅ | ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:1**

## ROW 95 (coral_idx 234)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR-/HER2- IDC with residual DCIS" ✓ |
| Stage_of_Cancer | ✅ | "Stage II (2.1cm tumor)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "MRI breast: interval decrease in mass" ✓ |
| current_meds | ✅ | ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Continue pembrolizumab, axilla XRT, capecitabine after XRT" ✓ |
| imaging_plan | P2 | "breast and axilla XRT" — XRT是radiotherapy不是imaging |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "prilosec 40mg qd, capecitabine after XRT" ✓ |

**P0:0 P1:0 P2:1**

## ROW 96 (coral_idx 235)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade I mixed ductal and cribriform" ✓ |
| Stage_of_Cancer | ✅ | "Stage IA (pT1cN0(sn))" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "adjuvant radiation, tamoxifen after" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | "send for [REDACTED] testing" ✓ |
| genetic_testing_plan | P2 | "send for [REDACTED] testing" — 与lab重复, 不确定是Oncotype还是genetic |
| Medication_Plan | ✅ | "tamoxifen after adjuvant radiation" ✓ |

**P0:0 P1:0 P2:1**

## ROW 97 (coral_idx 236)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2- grade 1 IDC with low-intermediate grade DCIS" ✓ |
| Stage_of_Cancer | ✅ | "Stage IA (pT1b N0)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "Not yet on treatment" ✓ |
| current_meds | ✅ | 空 ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Adjuvant endocrine therapy with [REDACTED]" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "molecular profiling, e.g. Oncotype Dx" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 98 (coral_idx 237)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER-PR-HER2- IDC with apocrine features, HG DCIS" ✓ |
| Stage_of_Cancer | ✅ | "Stage I (≤2cm)" ✓ |
| Distant Metastasis | ✅ | "No" ✓ |
| response_assessment | ✅ | "No specific evidence" ✓ |
| current_meds | ✅ | "Taxotere, Cytoxan" ✓ |
| goals_of_treatment | ✅ | "curative" ✓ |
| therapy_plan | ✅ | "Taxotere/cytoxan, refer to Radiation Onc" ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "Docusate, Emend, ondansetron, dexamethasone" ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 99 (coral_idx 238)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+/PR+/HER2+ grade 3 IDC with lymphovascular invasion" ✓ |
| Stage_of_Cancer | ✅ | "Originally Stage III, now metastatic (Stage IV)" ✓ |
| Distant Metastasis | ✅ | "Yes, to left lung" ✓ |
| response_assessment | ✅ | "Mixed response" ✓ |
| current_meds | ✅ | "fulvestrant" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | "Continue fulvestrant, if ER+ consider endocrine-based" ✓ |
| imaging_plan | ✅ | "Follow-up CT scan with contrast" ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | "biopsy of pulmonary nodule or mediastinal node" ✓ |
| Medication_Plan | ✅ | ✓ |

**P0:0 P1:0 P2:0** ✅

## ROW 100 (coral_idx 239)
| Field | 判定 | 备注 |
|-------|------|------|
| Type_of_Cancer | ✅ | "ER+(80%)PR+(50%) HER2- grade 2 IDC with met recurrence" ✓ |
| Stage_of_Cancer | ✅ | "Stage IV (metastatic)" ✓ |
| Distant Metastasis | ✅ | "Yes, to liver and bone" ✓ |
| response_assessment | ✅ | "Tumor markers elevated: CA15-3 118, CA27.29 178, Alk Phos 172" ✓ 详细 |
| current_meds | ✅ | "gemzar" ✓ |
| goals_of_treatment | ✅ | "palliative" ✓ |
| therapy_plan | ✅ | ✓ |
| imaging_plan | ✅ | ✓ |
| lab_plan | ✅ | ✓ |
| genetic_testing_plan | ✅ | ✓ |
| Medication_Plan | ✅ | "Focalin prn for fatigue" ✓ |

**P0:0 P1:0 P2:0** ✅

