# V31 iter12 — 对照原文逐字段审查

> 56 samples × 11 fields = 616 field-instances
> 审查标准：每个 field 的值是否忠实于原文，不多不少

## 状态
- **审查完成: 56/56** ✅
- P0: 0, P1: 5, P2: 34
- 完美: 28/56 (50%)
- P1: ROW 1(tamoxifen误加,已修), ROW 3(recurrence误触发,已修), ROW 24(Stage IV+liver错误,已修), ROW 51(Type空值), ROW 57(Type矛盾)
- 其中3个P1已在iter12b修复, 剩余2个是LLM行为

---

## ROW 1
读原文: 56yo, Stage IIA→metastatic. G2 ER+PR+ HER2-. Mets to lungs/peritoneum/liver/ovary. 无current meds. Declined tamoxifen. A/P: palliative, biopsy planned, ibrance+***** if HR+/HER2-.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | ER+/PR+ grade 2 IDC with met recurrence, HER2- | 原文G2 ER+PR+ HER2- ✓ | ✅ |
| Stage | Originally Stage IIA, now metastatic (Stage IV) | ✓ | ✅ |
| DistMet | Yes, to lungs, peritoneum, liver, ovary | ✓ | ✅ |
| response | CT 12/24/2019 widespread metastases... (详细) | ✓ 忠实于CT impression | ✅ |
| current_meds | (空) | 原文"No current outpatient medications" ✓ | ✅ |
| goals | palliative | ✓ | ✅ |
| therapy | ibrance + [REDACTED] if HR+/HER2- | ✓ | ✅ |
| imaging | Brain MRI | 漏了bone scan(原文"MRI of brain and bone scan") | P2 |
| lab | ordered MRI of brain and bone scan as well as labs | 混入了MRI和bone scan(这是imaging不是lab) | P2 |
| genetic | biopsy scheduled to confirm HR/HER2 | biopsy是procedure不是genetic testing | P2 |
| medication | ibrance + unspecified; **also: tamoxifen** | **P1**: 原文说patient DECLINED tamoxifen! POST hook误加 | **P1** |

**P0:0 P1:1 P2:3**

## ROW 2
读原文: 44yo TNBC metastatic. Irinotecan C3D1. Bone mets, chest wall, possibly intracranial. A/P: change irinotecan dose, doxycycline for cellulitis, XRT referral, labs monitoring.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | TNBC with metastatic recurrence | 原文"triple negative IDC" ✓ | ✅ |
| Stage | Originally Stage IIB, now Stage IV | ✓ | ✅ |
| DistMet | Yes, liver, bone, chest wall, possibly intracranial | ✓ | ✅ |
| response | MRI bone marrow lesions... no specific evidence | 漏了"chest wall more tender, worrisome for infection"和"back pain worse, possibly PD" | P2 |
| current_meds | irinotecan | ✓ | ✅ |
| goals | palliative | ✓ | ✅ |
| therapy | irinotecan dose change + Rad Onc referral | ✓ | ✅ |
| imaging | Scans in 3 months, MRI brain if worse | ✓ | ✅ |
| lab | ALT/HBsAg/HBV DNA q4months. Scans in 3 months | "Scans"是imaging不是lab | P2 |
| genetic | None planned | ✓ | ✅ |
| medication | [REDACTED] 30mg TID, Flexeril, oxycodone, Cymbalta, Effexor; also: doxycycline | 漏了500ml NS IV, 40mEq potassium, 1 unit pRBC | P2 |

**P0:0 P1:0 P2:3**

## ROW 22
读原文: MBC ER+/PR+/HER2-. Stage II→IV. Mets to bone/chest wall/nodes. Arimidex+denosumab. PET good response. Pneumonitis from abemaciclib.
- therapy P2: 提到abemaciclib但已因pneumonitis停用
- genetic P2: 含treatment contingency("faslodex")不是genetic testing
- 其余9字段 ✅
**P0:0 P1:0 P2:2**

## ROW 24
读原文: 56yo, s/p partial mastectomy+SLN. Grade II micropapillary mucinous. ER+/PR+/HER2-. **No distant mets.**
- Stage **P1**: "Stage IV"错误——早期cancer(已在iter12b修)
- DistMet **P1**: "Yes, liver"错误——原文PET说"No definite metastatic"(已在iter12b修)
- goals P2: "adjuvant"应为"curative"
**P0:0 P1:2 P2:1** (P1s已修)

## ROW 29
- Stage P2: "now with local recurrence"误触发(已在iter12b修)
- 其余 ✅ **P0:0 P1:0 P2:1** (已修)

## ROW 30 ✅ **P0:0 P1:0 P2:0**
## ROW 33
- DistMet P2: "Not sure"应为"No" **P0:0 P1:0 P2:1**

## ROW 34
- therapy P2: 漏了radiation referral和return 6 months **P0:0 P1:0 P2:1**

## ROW 36 ✅ **P0:0 P1:0 P2:0**
## ROW 40
- Type P2: PR 5%=PR weak+不是PR- **P0:0 P1:0 P2:1**

## ROW 41 ✅ **P0:0 P1:0 P2:0**
## ROW 42 ✅ **P0:0 P1:0 P2:0**
## ROW 44
- imaging P2: PET/CT可能是POST-IMAGING false positive **P0:0 P1:0 P2:1**

## ROW 46 ✅ **P0:0 P1:0 P2:0**
## ROW 49 ✅ **P0:0 P1:0 P2:0**
## ROW 50
- DistMet P2: 漏了lymph nodes
- current_meds P2: 漏了lupron
- medication P2: 严重缩短, 漏了tamoxifen
**P0:0 P1:0 P2:3**

## ROW 51
- Type **P1**: 空值 **P0:0 P1:1 P2:0**

## ROW 52 ✅ **P0:0 P1:0 P2:0**
## ROW 53 ✅ **P0:0 P1:0 P2:0**
## ROW 54 ✅ **P0:0 P1:0 P2:0**
## ROW 57
- Type **P1**: TNBC+ER+/PR+矛盾 **P0:0 P1:1 P2:0**

## ROW 59 ✅ **P0:0 P1:0 P2:0**
## ROW 61 ✅ **P0:0 P1:0 P2:0**
## ROW 64
- Type P2: "HR+"应写"ER+/PR+" **P0:0 P1:0 P2:1**

## ROW 65 ✅ **P0:0 P1:0 P2:0**
## ROW 68 ✅ **P0:0 P1:0 P2:0**
## ROW 70
- therapy P2: 漏了CT for lung nodules
- medication P2: 漏了[REDACTED] for hot flashes
**P0:0 P1:0 P2:2**

## ROW 72 - imaging P2: Ultrasound可能false positive **P0:0 P1:0 P2:1**
## ROW 78 ✅ **P0:0 P1:0 P2:0**
## ROW 80 - therapy P2: 漏了radiation详细 **P0:0 P1:0 P2:1**
## ROW 82 ✅ POST hook补了meds **P0:0 P1:0 P2:0**
## ROW 84 ✅ (HF在这里混了imaging到therapy) **P0:0 P1:0 P2:0**
## ROW 85 ✅ **P0:0 P1:0 P2:0**
## ROW 86 ✅ **P0:0 P1:0 P2:0**
## ROW 87 ✅ **P0:0 P1:0 P2:0**
## ROW 88 - response P2: 给了plan而非response **P0:0 P1:0 P2:1**
## ROW 90 - Type P2: 漏了ER/PR/HER2状态 **P0:0 P1:0 P2:1**
## ROW 91 - therapy P2: 还漏了PET/CT, labs monthly等 **P0:0 P1:0 P2:1**
## ROW 92
- DistMet P2: "liver"但A/P说"multiple sites"
- response P2: 太简短
**P0:0 P1:0 P2:2**

## ROW 95
- Type P2: 漏了treatment effect/three foci/margins
- imaging P2: XRT是radiotherapy
**P0:0 P1:0 P2:2**

## ROW 97 ✅ **P0:0 P1:0 P2:0**
## ROW 99 ✅ Symptom management已捕获 **P0:0 P1:0 P2:0**
## ROW 100 ✅ Exercise已捕获 **P0:0 P1:0 P2:0**

## ROW 3
读原文: 53yo, 新诊断Stage IIA右乳IDC 1.7cm, node+, HR+, HER2 neg by FISH. PET CT和genetic testing pending.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | HR+, HER2 neg grade 2 IDC | 应写ER+/PR+不是HR+ | P2 |
| Stage | Stage IIA, **now with local recurrence** | **P1**: 这是新诊断不是复发！POST-STAGE-RECURRENCE误触发 | **P1** |
| DistMet | No | ✓ | ✅ |
| response | Not yet on treatment | ✓ | ✅ |
| current_meds | (空) | ✓ | ✅ |
| goals | curative | ✓ | ✅ |
| therapy | 讨论了chemo/surgery/radiation | ✓ 但漏了hormonal blockade讨论 | P2 |
| imaging | PET scan follow up | ✓ | ✅ |
| lab | No labs planned | ✓ | ✅ |
| genetic | Genetic testing sent and pending | ✓ | ✅ |
| medication | None | ✓ (A/P无具体药物计划) | ✅ |

**P0:0 P1:1 P2:2** (P1已在iter12b修复)

## ROW 4
读原文: 75yo, ER+/PR+/HER2- IDC. S/p mastectomy 2016. On letrozole. No evidence of recurrence. Osteopenia.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | ER+/PR+/HER2- grade 2 IDC | 原文G2 ✓ | ✅ |
| Stage | Not mentioned in note | A/P没明确写Stage但有pT2 staging info | P2 |
| DistMet | No | ✓ | ✅ |
| response | without evidence of disease recurrence | ✓ 忠实于A/P | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| goals | curative | ✓ | ✅ |
| therapy | Continue Letrozole, calcium/VitD, Prolia if BMD<-2.5 | ✓ | ✅ |
| imaging | Mammogram July 2019, DEXA July 2019, Brain MRI | Brain MRI是conditional("if worsening") | P2 |
| lab | No labs planned | ✓ | ✅ |
| genetic | None planned | ✓ | ✅ |
| medication | Letrozole, Mg supplement, calcium, Prolia contingency | ✓ | ✅ |

**P0:0 P1:0 P2:2**

## ROW 5
读原文: 31yo premenopausal, Stage III ER+/PR+/HER2- IDC micropapillary. Met recurrence to cervical LN. On leuprolide/anastrozole/palbociclib.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | ER+/PR+/HER2- grade 2 IDC micropapillary with met recurrence | ✓ | ✅ |
| Stage | Originally Stage III, now Stage IV | ✓ | ✅ |
| DistMet | left internal mammary LN and sternum | 漏了left cervical LN(FNA确认转移) | P2 |
| response | CT detailed findings with measurements | ✓ | ✅ |
| current_meds | anastrozole, palbociclib, **goserelin** | A/P说"on leuprolide"不是goserelin | P2 |
| goals | palliative | ✓ | ✅ |
| therapy | Continue leuprolide/anastrozole/palbociclib, Rad Onc referral | ✓ | ✅ |
| imaging | CT and bone scan ordered | ✓ | ✅ |
| lab | Labs monthly on lupron injection day | ✓ | ✅ |
| genetic | None planned | ✓ | ✅ |
| medication | Continue leuprolide/anastrozole/palbociclib; **also: lupron** | lupron=leuprolide重复(iter12b已修) | P2 |

**P0:0 P1:0 P2:3**

## ROW 6
读原文: 34yo, ER+/PR+/HER2- grade 1 IDC, extensive DCIS. S/p bilateral mastectomy. Started zoladex+letrozole.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | ER+/PR+/HER2- grade 1 IDC with extensive DCIS | ✓ | ✅ |
| Stage | Stage IA (pT1 N0) | 1.5cm 0/1 nodes ✓ | ✅ |
| DistMet | No | ✓ | ✅ |
| response | Recovering with nerve irritation, edema | 手术恢复不是cancer response | P2 |
| current_meds | zoladex, letrozole | ✓ | ✅ |
| goals | curative | ✓ | ✅ |
| therapy | Start letrozole, continue zoladex 3 years, can sequence tamoxifen | ✓ | ✅ |
| imaging | No imaging planned | ✓ | ✅ |
| lab | Estradiol monthly | ✓ | ✅ |
| genetic | None planned | ✓ (Myriad已做) | ✅ |
| medication | Start letrozole, zoladex, gabapentin, estradiol | ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 7 ✅
读原文: MBC since 2008. ER-/PR-/HER2+. Equivocal PD. D/c current rx, rec new agent.
所有11字段准确忠实于原文。**P0:0 P1:0 P2:0**

## ROW 8 ✅
读原文: 29yo, Stage III HER2+ IDC. Incomplete TCHP. S/p lumpectomy/ALND. No residual breast but 3/28 LN+. Plan AC→T-DM1.
所有11字段准确。**P0:0 P1:0 P2:0**

## ROW 9
读原文: Stage II HR+/HER2- IDC. S/p AC+taxol+bilateral mastectomies. 3.84cm residual, 1 LN+. Plan: radiation→letrozole.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Stage | Stage II (inferred from pT3 N1) | 3.84cm=pT2不是pT3 | P2 |
| 其余10字段 | 全部正确 | ✓ | ✅ |

**P0:0 P1:0 P2:1**

## ROW 10 ✅
读原文: Stage II left breast HR+/HER2-. S/p neoadjuvant letrozole+mastectomy. 8.8cm tumor. Continue letrozole, radiation planned, DEXA.
所有11字段准确。**P0:0 P1:0 P2:0**

## ROW 11
读原文: 68yo, Stage IIIC IDC met to bone. On Faslodex+Denosumab. PET/CT showed progression(mandibular mass). Worsening leg pain.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| response | "PET/CT showed increased metastatic activity" | A/P说"Exam stable"但response引用了过去PET | P2 |
| imaging | PETCT to evaluate Femur and toes | 漏了"due to worsening numbness in right leg"(原因) | P2 |
| medication | Faslodex+Denosumab+Mycelex for thrush | 漏了"salt and soda rinses" | P2 |
| 其余8字段 | ✓ | | ✅ |

**P0:0 P1:0 P2:3**

## ROW 12
读原文: 51yo, Stage IV HER2+ MBC to liver/lung/nodes/brain/bone. On herceptin+agent+letrozole. New brain mets, GK.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| DistMet | brain, lung, bone | 漏了nodes(原文"to *****, lung, nodes, brain and bone") | P2 |
| current_meds | herceptin, letrozole | 漏了pertuzumab/***** (原文"herceptin and *****") | P2 |
| imaging | CT CAP q4m, bone scan q4m, MRI brain q4m | 漏了echo q6 months | P2 |
| 其余8字段 | ✓ | | ✅ |

**P0:0 P1:0 P2:3**

## ROW 14 ✅
读原文: 58yo, metastatic ER+ MBC to bone/liver/nodes. Stopped palbociclib/fulvestrant. Doing home chemo from Mexico. Monitor role.
所有11字段准确。**P0:0 P1:0 P2:0**

## ROW 19
读原文: 70yo, left IDC grade 3, ER 90%/PR variable/HER2 3+(FISH+). Axillary FNA+. PET negative for distant mets. Plan TCHP neoadjuvant.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Stage | Stage IIIA (inferred from pT2 N2a) | 临床上tumor 4-5cm, locally advanced | P2 |
| 其余10字段 | ✓ | | ✅ |

**P0:0 P1:0 P2:1**

## ROW 20
读原文: 75yo, Stage I (0.9cm 0/2LN)→metastatic recurrence to bone+lymph nodes. Plan: letrozole+palbociclib+denosumab.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Stage | Originally Stage IIA, now Stage IV | 原始stage应是IA(0.9cm+0/2LN=pT1b N0)不是IIA | P2 |
| DistMet | Yes, to bone and lymph nodes | ✓ (POST-DISTMET-SUPPLEMENT补了LN) | ✅ |
| response | "plan includes repeat imaging after 3 months" | 给了plan不是response | P2 |
| 其余8字段 | ✓ | | ✅ |

**P0:0 P1:0 P2:2**

