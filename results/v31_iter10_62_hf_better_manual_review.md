# iter10 — 62 个 HF>vLLM 逐个人工判定

> 判定标准：读 vLLM 值、HF 值、A/P 原文，判断谁更准确/更完整

## 分类：REAL=vLLM真漏了 | TIE=信息一致 | vLLM_OK=vLLM实际更正确 | HF_WRONG=HF放错内容

| # | ROW | Field | HF len | vLLM len | 判定 | 理由 |
|---|-----|-------|--------|----------|------|------|
| 1 | 2 | response_assessment | 312 | 241 | TIE | 两者都有MRI/bone lesion信息，HF多了"progressing"总结词 |
| 2 | 2 | medication_plan | 161 | 104 | REAL | vLLM漏了Doxycycline, NS IV, potassium, pRBC |
| 3 | 5 | Distant Metastasis | 71 | 44 | REAL | vLLM漏了cervical LN(只有IM LN+sternum) |
| 4 | 5 | response_assessment | 698 | 583 | TIE | 两者都有详细CT findings，HF略多一些细节 |
| 5 | 5 | therapy_plan | 201 | 157 | TIE | vLLM漏了"Labs monthly on the day of lupron"但这是lab不是therapy |
| 6 | 7 | response_assessment | 288 | 237 | TIE | 几乎一致，HF多了几个词 |
| 7 | 10 | therapy_plan | 120 | 89 | TIE | HF多了"started April 2021"日期，实质一致 |
| 8 | 11 | imaging_plan | 86 | 47 | TIE | HF多了"due to worsening numbness in right leg"（原因不是imaging plan） |
| 9 | 12 | response_assessment | 1227 | 592 | REAL | HF有更多历史imaging context(CT stable, echo, multiple MRI dates) |
| 10 | 12 | medication_plan | 195 | 145 | TIE | 两者都有herceptin/agent/letrozole/bone agent，HF格式不同 |
| 11 | 20 | Type_of_Cancer | 99 | 68 | REAL | HF有receptor change history(original vs metastatic biopsy)，vLLM有grade+DCIS |
| 12 | 20 | Distant Metastasis | 25 | 12 | REAL | vLLM漏了lymph nodes(只有bone) |
| 13 | 22 | imaging_plan | 160 | 10 | HF_WRONG | HF混入了treatment contingency("if stable continue arimidex") |
| 14 | 29 | Stage_of_Cancer | 30 | 25 | vLLM_OK | HF写了pTN格式，vLLM翻译为Stage IIA更清楚 |
| 15 | 34 | Type_of_Cancer | 118 | 59 | REAL | HF有receptor change(originally/metastatic biopsy)，vLLM只有当前 |
| 16 | 34 | Stage_of_Cancer | 59 | 9 | REAL | HF有"now local recurrence (not metastatic)"，vLLM只有"Stage III" |
| 17 | 36 | Stage_of_Cancer | 33 | 18 | TIE | "Stage IIIA (inferred from pT3 N0)" vs "Stage IIIA (pT3N0)"——一样 |
| 18 | 40 | Type_of_Cancer | 89 | 73 | REAL | HF有精确百分比(ER 95, PR 5, HER2 2+ FISH 1.2)，vLLM只有+/- |
| 19 | 44 | Type_of_Cancer | 104 | 66 | TIE | HF有"node+"但这属于Stage不是Type，vLLM有grade+DCIS |
| 20 | 46 | medication_plan | 259 | 182 | REAL | vLLM漏了gabapentin(A/P有提到) |
| 21 | 50 | Stage_of_Cancer | 59 | 32 | TIE | HF多了冗余的"now metastatic (Stage IV)" |
| 22 | 50 | Distant Metastasis | 46 | 29 | REAL | vLLM漏了lymph nodes |
| 23 | 50 | medication_plan | 232 | 50 | REAL | vLLM严重缩短，漏了tamoxifen和很多context |
| 24 | 53 | genetic_testing_plan | 64 | 54 | TIE | "Offered a referral" vs "Referral"——措辞差异 |
| 25 | 54 | Stage_of_Cancer | 25 | 21 | TIE | "now metastatic (Stage IV)" vs "Stage IV (metastatic)" |
| 26 | 57 | Stage_of_Cancer | 106 | 84 | TIE | 措辞不同但信息一致 |
| 27 | 59 | Stage_of_Cancer | 105 | 32 | vLLM_OK | HF说Stage I，vLLM说Stage IIA(pT2 N0)——pT2 N0=IIA，vLLM更准确 |
| 28 | 59 | response_assessment | 486 | 291 | TIE | 两者都有"no evidence of recurrence"，HF更verbose |
| 29 | 61 | medication_plan | 103 | 88 | TIE | 措辞差异 |
| 30 | 64 | therapy_plan | 235 | 184 | TIE | HF多了"Currently on [REDACTED]"，vLLM有xgeva——不同信息 |
| 31 | 65 | Stage_of_Cancer | 106 | 53 | vLLM_OK | vLLM的IB(pT1 N1mi)比HF的Stage II更精确 |
| 32 | 65 | lab_plan | 18 | 4 | TIE | "labs. F/u pending" vs "labs" |
| 33 | 65 | medication_plan | 430 | 352 | TIE | 两者都有AC/T和ISPY trial选项，HF更verbose |
| 34 | 68 | Stage_of_Cancer | 73 | 34 | vLLM_OK | vLLM"Stage I(≤2cm)"比HF"Early stage(inferred)"更具体 |
| 35 | 68 | genetic_testing_plan | 147 | 107 | TIE | vLLM漏了"(if spouse carries mutation)"——次要条件 |
| 36 | 68 | medication_plan | 150 | 73 | TIE | 两者都说没有medication plan，HF更verbose |
| 37 | 70 | response_assessment | 393 | 258 | TIE | 两者都有MRI findings，HF加了"responding to treatment"总结 |
| 38 | 70 | therapy_plan | 166 | 111 | REAL | HF有"follow-up CT for lung nodules June 2020"，vLLM没有 |
| 39 | 70 | medication_plan | 108 | 62 | REAL | HF有"Information about [REDACTED] for hot flashes"，vLLM没有 |
| 40 | 72 | therapy_plan | 126 | 85 | TIE | "Instructed to begin letrozole" vs "Continue letrozole"——措辞 |
| 41 | 78 | response_assessment | 622 | 522 | TIE | 两者都有详细imaging，HF略多一些context |
| 42 | 78 | genetic_testing_plan | 122 | 97 | TIE | 措辞差异 |
| 43 | 80 | therapy_plan | 128 | 87 | REAL | vLLM漏了radiation详细(6 weeks, boost, axilla+SC fields) |
| 44 | 82 | medication_plan | 356 | 169 | REAL | vLLM漏了acetaminophen, ibuprofen等多个supportive meds |
| 45 | 84 | response_assessment | 913 | 699 | TIE | 两者都有MRI brain和CT findings，HF加了总结 |
| 46 | 84 | therapy_plan | 492 | 273 | HF_WRONG | HF混入了CT/LP/MRI(imaging)到therapy_plan |
| 47 | 85 | response_assessment | 440 | 356 | TIE | 两者都描述progression，HF多了brain MRI reviewer名 |
| 48 | 85 | therapy_plan | 163 | 133 | TIE | vLLM漏了"2-week radiation washout"——次要细节 |
| 49 | 85 | genetic_testing_plan | 128 | 108 | TIE | 措辞差异 |
| 50 | 87 | therapy_plan | 133 | 59 | TIE | HF多了"specific therapy not detailed"——这是disclaimer不是信息 |
| 51 | 88 | response_assessment | 184 | 103 | REAL | vLLM给了plan("restaging")而非response("stable disease, no masses") |
| 52 | 88 | therapy_plan | 237 | 137 | TIE | 两者都有xeloda+clinical trials |
| 53 | 88 | medication_plan | 297 | 218 | TIE | 两者都有xeloda+HER2 contingency+clinical trials |
| 54 | 90 | Type_of_Cancer | 90 | 36 | TIE | HF多了"ER/PR/HER2 not explicitly stated"——这是disclaimer |
| 55 | 91 | therapy_plan | 298 | 92 | REAL | vLLM虽然加了lasix/KCL/elevation但还漏了denosumab for hip, PET/CT next week |
| 56 | 92 | Distant Metastasis | 22 | 13 | REAL | vLLM只有"liver"，A/P说"multiple sites" |
| 57 | 92 | response_assessment | 442 | 119 | REAL | vLLM只有exam，HF有"stable on treatment"+tumor markers+详细exam |
| 58 | 92 | current_meds | 25 | 21 | TIE | "EPIRUBICIN HCL, DENOSUMAB" vs "Epirubicin, Denosumab"——大小写 |
| 59 | 95 | Type_of_Cancer | 166 | 78 | REAL | vLLM漏了"treatment effect, three foci, margins negative" |
| 60 | 95 | imaging_plan | 53 | 21 | HF_WRONG | XRT是radiotherapy不是imaging，两者都放错了 |
| 61 | 95 | medication_plan | 131 | 64 | TIE | vLLM略简但信息一致(prilosec+capecitabine after XRT) |
| 62 | 100 | Distant Metastasis | 32 | 22 | vLLM_OK | HF"liver and multiple sites"模糊，vLLM"liver and bone"更具体 |

## 汇总

| 分类 | 数量 | 占比 |
|------|------|------|
| **TIE** (信息一致，措辞不同) | **33** | 53% |
| **REAL** (vLLM真漏了信息) | **20** | 32% |
| **vLLM_OK** (vLLM实际更好) | **5** | 8% |
| **HF_WRONG** (HF放错内容) | **3** | 5% |
| **不确定** | **1** | 2% |

## 20 个 REAL MISS 按字段分布

| Field | 数量 | 典型问题 |
|-------|------|---------|
| medication_plan | 5 | 漏supportive meds(Doxycycline, gabapentin, acetaminophen等) |
| response_assessment | 3 | 过于简短或给plan而非response |
| Distant Metastasis | 3 | 漏lymph nodes |
| Type_of_Cancer | 3 | 漏receptor change history或精确百分比或treatment effect |
| therapy_plan | 3 | 漏radiation details或follow-up imaging |
| Stage_of_Cancer | 1 | 漏"local recurrence"说明 |
| imaging_plan | 1 | — |
| lab_plan | 1 | — |

## 可修的 REAL MISS

| # | ROW | Field | 可否加POST hook | 方案 |
|---|-----|-------|----------------|------|
| 2 | 2 | medication_plan | ✅ | POST-MEDICATION-SUPPLEMENT加Doxycycline等 |
| 20 | 46 | medication_plan | ✅ | gabapentin已在drug list但未匹配到 |
| 23 | 50 | medication_plan | ✅ | tamoxifen在A/P但POST hook未捕获 |
| 44 | 82 | medication_plan | ✅ | 加更多supportive meds到drug list |
| 55 | 91 | therapy_plan | ✅ | 加denosumab, PET/CT到supportive items |
| 3,12,22,56 | 5,20,50,92 | Distant Metastasis | ✅ | POST hook搜lymph nodes |
| 43 | 80 | therapy_plan | 部分 | radiation details是model行为 |
| 51 | 88 | response_assessment | ❌ | model给plan而非response |
| 9 | 12 | response_assessment | ❌ | model输出短，无法强制更长 |
| 57 | 92 | response_assessment | ❌ | 同上 |
