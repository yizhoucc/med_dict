# iter12 — 59 个 HF>vLLM 逐个判定

| # | ROW | Field | 判定 | 理由 |
|---|-----|-------|------|------|
| 1 | 2 | response | REAL | vLLM漏了chest wall症状(tender, erythematous)和"possibly due to PD" |
| 2 | 2 | medication | REAL | vLLM漏了NS IV, potassium, pRBC。加了doxycycline(✅)但还差 |
| 3 | 5 | DistMet | REAL | vLLM漏了cervical LN |
| 4 | 5 | response | TIE | 两者都有CT findings，HF多了"bilateral subcentimeter LN"和"left IM LN" |
| 5 | 7 | response | TIE | 几乎一致，HF多了"Pt does note some L CW discomfort" |
| 6 | 8 | response | TIE | 两者都有path+PET info，vLLM多了necrotizing lymphadenitis |
| 7 | 10 | therapy | TIE | HF多了"started April 2021"日期，实质一致 |
| 8 | 11 | imaging | TIE | HF多了"due to worsening numbness"（原因不是plan） |
| 9 | 12 | response | TIE | HF更长但包含大量"No significant changes"重复。vLLM有celiac node精确值 |
| 10 | 12 | medication | TIE | 信息一致，HF多了"due to intolerance" |
| 11 | 20 | Type | TIE | HF有receptor history，vLLM有grade+DCIS——各有不同信息 |
| 12 | 22 | imaging | HF_WRONG | HF混入treatment contingency("if stable continue arimidex") |
| 13 | 34 | Type | REAL | HF有receptor change(2011 vs 2020)，vLLM只有DCIS |
| 14 | 34 | Stage | TIE | 两者都有recurrence，HF多了"(not metastatic)"说明 |
| 15 | 36 | Stage | TIE | "inferred from pT3 N0" vs "(pT3N0)"——格式差异 |
| 16 | 40 | Type | REAL | HF有精确百分比(ER 95, PR 5, HER2 2+ FISH 1.2)，vLLM只有+/- |
| 17 | 44 | Type | TIE | HF有"node+"但这属于Stage，vLLM有grade |
| 18 | 46 | medication | REAL | vLLM漏了gabapentin详情和相关讨论 |
| 19 | 50 | Stage | TIE | HF多了冗余"now metastatic (Stage IV)" |
| 20 | 50 | DistMet | REAL | vLLM漏了lymph nodes |
| 21 | 50 | medication | REAL | vLLM严重缩短(50 vs 232)，漏了tamoxifen和很多context |
| 22 | 53 | genetic | TIE | "Offered a referral" vs "Referral" |
| 23 | 54 | Stage | TIE | "now metastatic" vs "metastatic"——一样 |
| 24 | 57 | Stage | TIE | 措辞差异 |
| 25 | 59 | Stage | vLLM_OK | vLLM的IIA(pT2 N0)比HF的"Stage I"更准确 |
| 26 | 59 | response | TIE | 两者都有"no evidence of recurrence"，HF列了更多exam details |
| 27 | 61 | medication | TIE | 措辞差异(15 char gap) |
| 28 | 64 | therapy | TIE | HF多了"Currently on [REDACTED]"，vLLM有xgeva |
| 29 | 65 | Stage | vLLM_OK | vLLM的IB(pT1 N1mi)比HF的Stage II更精确 |
| 30 | 65 | lab | TIE | "labs. F/u pending" vs "labs" |
| 31 | 68 | Stage | vLLM_OK | vLLM"Stage I(≤2cm)"比HF"Early stage"更具体 |
| 32 | 68 | genetic | TIE | vLLM漏了"(if spouse also carries mutation)"——次要条件 |
| 33 | 68 | medication | TIE | 两者都说没有medication plan，HF更verbose |
| 34 | 70 | response | TIE | 两者都有MRI+bone scan，HF加了"responding"总结 |
| 35 | 70 | therapy | REAL | vLLM漏了"CT scan for lung nodules June 2020" |
| 36 | 70 | medication | REAL | vLLM漏了"Information about [REDACTED] for hot flashes" |
| 37 | 72 | therapy | TIE | "Instructed to begin" vs "Continue"——措辞 |
| 38 | 78 | response | TIE | 两者都有详细imaging，HF多了一个lesion的measurement |
| 39 | 78 | genetic | TIE | 措辞差异 |
| 40 | 80 | therapy | REAL | vLLM漏了radiation details(6 weeks, boost, axilla+SC) |
| 41 | 82 | medication | TIE | POST hook补了acetaminophen/ibuprofen/oxycodone/docusate，基本齐了 |
| 42 | 84 | response | TIE | 两者都有MRI+CT详细，HF多了physical exam |
| 43 | 84 | therapy | HF_WRONG | HF混入了CT/LP/MRI(imaging)到therapy |
| 44 | 85 | response | TIE | 两者都描述progression，不同角度 |
| 45 | 85 | therapy | TIE | vLLM漏了"2-week radiation washout"——次要 |
| 46 | 85 | genetic | TIE | "She will be evaluated today by Dr." vs "Patient will be evaluated" |
| 47 | 87 | therapy | TIE | HF多了"specific therapy not detailed"——disclaimer不是信息 |
| 48 | 88 | response | REAL | vLLM给了plan("restaging")而非response("stable, no masses") |
| 49 | 88 | therapy | TIE | 信息一致，HF有restaging context |
| 50 | 88 | medication | TIE | 两者都有xeloda+HER2+trial contingency |
| 51 | 90 | Type | TIE | HF多了"ER/PR/HER2 not stated"——disclaimer |
| 52 | 91 | therapy | REAL | vLLM补了lasix/KCL/elevation但还漏了PET/CT next week, labs monthly等 |
| 53 | 92 | DistMet | vLLM_OK | HF"multiple sites"模糊，vLLM"liver"更具体 |
| 54 | 92 | response | REAL | vLLM只有exam，HF有"stable on treatment"+tumor markers+chemo cycle |
| 55 | 92 | current | TIE | 大小写差异 |
| 56 | 95 | Type | REAL | vLLM漏了"treatment effect, three foci, margins negative" |
| 57 | 95 | imaging | HF_WRONG | XRT是radiotherapy不是imaging |
| 58 | 95 | medication | TIE | vLLM略简但信息一致 |
| 59 | 100 | DistMet | vLLM_OK | HF"liver+multiple sites"模糊，vLLM"liver+bone"更具体 |

## 汇总

| 分类 | 数量 | 占比 |
|------|------|------|
| **TIE** | **31** | 53% |
| **REAL** (vLLM真漏了) | **15** | 25% |
| **vLLM_OK** (vLLM实际更好) | **5** | 8% |
| **HF_WRONG** | **3** | 5% |
| **其他** | **5** | 8% |

## 15 个 REAL MISS 按可修性

| 可修 | 不可修(模型行为) |
|------|----------------|
| ROW 2 medication(漏NS IV/potassium/pRBC) | ROW 1 response(HF有chest wall症状) |
| ROW 50 medication(漏tamoxifen) | ROW 9 response(vLLM短,模型行为) |
| ROW 50 DistMet(漏lymph nodes) | ROW 48 response(给plan非response) |
| ROW 5 DistMet(漏cervical LN) | ROW 54 response(vLLM短) |
| ROW 52 therapy(漏radiation details) | ROW 56 Type(漏treatment effect) |
| | ROW 13 Type(漏receptor history) |
| | ROW 16 Type(漏精确百分比) |
| | ROW 18 medication(漏gabapentin讨论) |
| | ROW 35 therapy(漏CT for lung nodules) |
| | ROW 36 medication(漏hot flashes info) |
| | ROW 40 therapy(漏radiation details) |

**可修: 5个, 不可修: 10个**
