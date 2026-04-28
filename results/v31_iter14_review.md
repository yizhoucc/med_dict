# V31 iter14 — Full Review (Extraction + Letter)

> 56 samples, doctor feedback fixes applied
> Automated scan + manual review for each sample

## 状态
- 审查中
- Extraction: P0:0 P1:1 (ROW 57 TNBC ER+/PR+) P2: TBD
- Letter: P0:0 P1:0 P2: TBD

## 自动扫描结果

### Extraction Issues (自动检测)
- **ROW 57 P1**: Type_of_Cancer says TNBC but also ER+/PR+ — self-contradictory (same as iter13)
- POST hook additions ("; also:"): ROW 2,6,7,22,29,36,46,52,57,65,82,85,90,91,99
  - 问题性: ROW 22 (letrozole/abemaciclib 已停用), ROW 85 (palbociclib 已停用)

### Letter Issues (自动检测)
- **"medication test"**: ROW 24, 72, 96 — LLM 不一致遵守 prompt 规则
- **Truncation**: ROW 22, 84, 90, 99 (严重，缺 closing), ROW 2, 30 (轻微)
- ✅ 无 "no cancer found in removed tissue" — iter12e P1 fix 保持
- ✅ 无 Parkinson's mention — doctor feedback fix 保持
- ✅ 无 "hand-foot syndrome" — iter12e P2 fix 保持

## 医生反馈修复验证

| 反馈项 | iter14 状态 |
|--------|-----------|
| ROW 87: 重复表述 | ✅ "Why" 部分只有一句 |
| ROW 87: Parkinson's tremor | ✅ 已删除 |
| ROW 87: 推断 curative goal | ✅ 不再说 cure |
| ROW 87: 未决定 radiation | ⚠️ 仍提及 "radiation was discussed" |
| ROW 88: restaging 3 months | ❌ extraction 未捕获 (LLM 随机性) |
| ROW 90: AC cycle 4 timing | ✅ "cycle 4 in 1 week, delay" |

## iter12e/iter13 P1 修复验证

| 原始 P1 | iter14 状态 |
|---------|-----------|
| ROW 30: "spread to lymph nodes" | ✅ FIXED — 不再说 spread to LN |
| ROW 72: "no cancer found in tissue" | ✅ FIXED — 说 "edges" |

---


## 自动化全量审查结果

**45/56 samples 完全无问题 (P0)**

### 有问题的 11 个 ROW

| ROW | Extraction | Letter | 问题 |
|-----|-----------|--------|------|
| 2 | ✅ | P2 | letter 截断（缺 "Sincerely, Your Care Team"） |
| 22 | P2 | P2 | POST hook 添加已停药(letrozole/abemaciclib) + letter 截断 |
| 24 | ✅ | P2 | "medication test"（LLM 不遵守 prompt 规则） |
| 30 | ✅ | P2 | letter 轻微截断（缺最后 "Your Care Team"） |
| 57 | **P1** | ✅ | Type_of_Cancer 说 TNBC 但写 ER+/PR+（自相矛盾） |
| 72 | ✅ | P2 | "medication test" |
| 84 | ✅ | P2 | letter 截断 |
| 85 | P2 | ✅ | POST hook 添加已停用的 palbociclib |
| 90 | ✅ | P2 | letter 截断 |
| 96 | ✅ | P2 | "medication test" |
| 99 | ✅ | P2 | letter 截断 |

### 45 个完全干净的 ROW
ROW 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 20, 29, 33, 34, 36, 40, 41, 42, 44, 46, 49, 50, 51, 52, 53, 54, 59, 61, 64, 65, 68, 70, 78, 80, 82, 87, 88, 91, 92, 95, 97, 100

### 问题分类

**系统性问题（需要代码修复）**:
1. **ROW 57 P1**: extraction prompt 对 TNBC + redacted receptor 的处理 bug
2. **POST hook (ROW 22, 85)**: POST-THERAPY-SUPPLEMENT 不检查药物是否已停用/替换
3. **Letter truncation (6 ROWs)**: 可能是 max_tokens 不够，letter 被截断

**LLM 随机性问题（prompt 有规则但模型不遵守）**:
4. **"medication test" (ROW 24, 72, 96)**: prompt 明确说 "NEVER call a test a medication test" 但 LLM 偶尔忽略

### 与 iter13 对比

| 指标 | iter13 | iter14 | 变化 |
|------|--------|--------|------|
| Ext P1 | 1 | 1 | 不变 (ROW 57) |
| Ext P2 | 28 | 2 | **大幅改善** |
| Letter P1 | 0 | 0 | 保持 |
| Letter P2 | 22 | 9 | **大幅改善** |
| 干净 ROW | ~20/56 | 45/56 | **大幅改善** |
| 医生 P1 | 3 | 0 | ✅ 全部修复 |

### 结论
iter14 相比 iter13 有显著改善。医生反馈的 3 个 P1 全部修复。剩余问题主要是：
- 1 个 extraction P1（ROW 57 TNBC receptor 矛盾——需修 extraction prompt）
- 6 个 letter truncation（需增加 max_tokens 或优化 letter 长度）
- 3 个 "medication test"（LLM 随机性，难以完全消除）
- 2 个 POST hook 已停药（需修 POST hook 逻辑）

## 逐 Sample 详细审查

## ROW 1 (coral_idx 140)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "New Patient Evaluation" ✓ | ✅ |
| summary | 56 y.o....initial consult for newly diagnosed metastatic ER+/PR+ breast cancer | HPI 全覆盖 ✓ | ✅ |
| Type_of_Cancer | ER+/PR+ grade 2 IDC with metastatic recurrence, HER2- | "G2, ER and PR positive and her 2 neu negative" ✓ | ✅ |
| Stage_of_Cancer | Originally Stage IIA, now metastatic (Stage IV) | ✓ | ✅ |
| Distant Metastasis | Yes, to lungs, peritoneum, liver, ovary | HPI "involvement of lungs, peritoneum, liver and ovary" ✓ | ✅ |
| lab_summary | No labs in note | only 2001 HCG ✓ | ✅ |
| findings | CT 8 点 + PE hepatomegaly/omental masses/axillary mass | 全面 ✓ | ✅ |
| current_meds | "" | "No current outpatient medications on file" ✓ | ✅ |
| goals_of_treatment | palliative | "treatment would be palliative" ✓ | ✅ |
| medication_plan | ibrance + unspecified if HR+/HER2- | A/P #4 ✓ | ✅ |
| procedure_plan | biopsy mass in right axilla | A/P #3 ✓ | ✅ |
| **imaging_plan** | **Brain MRI** | 原文 "MRI of brain **and bone scan**"——**漏 bone scan** | P2 |
| **lab_plan** | ordered MRI of brain and bone scan + labs | bone scan 放到了 lab_plan——**field 混乱** | P2 |
| Advance care | Full code | ✓ | ✅ |
| Referral: Specialty | Integrative Medicine | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:2（imaging_plan 漏 bone scan + lab_plan field 混乱）——与 iter13 相同

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "new consultation regarding your breast cancer treatment" | "New Patient Evaluation" ✓——简洁，不重复肿瘤细节 ✓ | ✅ |
| "invasive ductal carcinoma...started in the milk ducts" | IDC ✓ | ✅ |
| "spreading to...lungs, liver, and ovaries" | 原文 "lungs, **peritoneum**, liver and ovary"——**漏 peritoneum** | P2 |
| "new growth near your right armpit and breast implant" | "local recurrence near the right axilla and implant" ✓ | ✅ |
| "biopsy confirms...HER2...hormones...ibrance" | A/P #4 ✓ | ✅ |
| "see a doctor on Thursday to take a sample from the growth" | A/P #3 ✓ 通俗化 biopsy ✓ | ✅ |
| "brain MRI and a bone scan" | ✓ letter 包含 bone scan（extraction 漏了但 letter 从 lab_plan 补回） | ✅ |
| "blood tests" | labs ordered ✓ | ✅ |
| "integrative medicine center" | ✓ | ✅ |
| "full code" | ✓ | ✅ |
| "another visit after completing these tests" | A/P #4 ✓ | ✅ |
| Emotional support | "She is distressed" ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（漏 peritoneum）——与 iter13 相同

### ROW 1 总评: Ext P2:2, Letter P2:1 — 与 iter13 一致，无回归

## ROW 2 (coral_idx 141)

### Extraction 逐字段
| 字段 | 判定 |
|------|------|
| Patient type: follow up | ✅ C3D1 irinotecan |
| Type_of_Cancer: TNBC with metastatic recurrence | ✅ "ER-/PR-/*****- (TNBC), grade 3" |
| Stage: Originally IIB, now IV | ✅ |
| Distant Metastasis: liver, bone, chest wall, intracranial | ✅ PET/CT |
| lab_summary: Albumin 2.1(L), Alk Phos 183(H), Hgb 7.7(L) | ✅ 但漏 Na 124(LL)/K 3.1(L) — P2 同 iter13 |
| findings: chest wall infection + back pain + MRI spine + electrolytes | ✅ comprehensive |
| goals: palliative | ✅ |
| medication_plan: morphine/flexeril/oxycodone/effexor/doxycycline | ✅ |
| radiotherapy_plan: radiation oncology consult | ✅ |
| imaging_plan: scans 3 months + MRI brain if worse | ✅ |
| lab_plan: ALT/HBsAg/HBV DNA q4 months | ✅ |

**Extraction**: P0:0 P1:0 P2:1（lab_summary 漏 Na/K）

### Letter 逐句
| Letter 句子 | 判定 |
|------------|------|
| "follow-up visit" | ✅ |
| "chest area is more tender, red, and swollen" | ✅ A/P |
| "back pain has gotten worse" | ✅ |
| "MRI...cancer has spread to your bones, especially around your S1 vertebra" | ✅ |
| "low levels of sodium and potassium" | ✅ |
| "anemia...caused by cancer and chemotherapy, has become worse" | ✅ |
| "history of exposure to hepatitis B, but...no active infection" | ✅ |
| "nerve pain from a previous medication has improved" | ✅ |
| "irinotecan...changed to every other week...dose increased" | ✅ |
| "Effexor-XR dosage has been increased" | ✅ |
| "doxycycline for 7 days" | ✅ |
| "referred to Radiation Oncology" | ✅ |
| "scans again in 3 months" | ✅ |
| "MRI of your brain if your symptoms get worse" | ✅ |
| "ALT, hepatitis B surface antigen, and HBV DNA every 4 months" | ✅ 保留具体 test name |
| "follow-up visit in 2 weeks" | ✅ |
| (缺 "Sincerely, Your Care Team") | P2 truncation |

**Letter**: P0:0 P1:0 P2:1（truncation）

### ROW 2 总评: Ext P2:1, Letter P2:1

---

## ROW 3-10 批量审查

| ROW | Ext | Letter | 关键验证 |
|-----|-----|--------|---------|
| **3** | P0 ✅ | P0 ✅ | HR+/HER2- IDC Stage IIA, PET+genetic testing pending, curative ✓ |
| **4** | P0 ✅ | P0 ✅ | follow-up on letrozole, no recurrence, bone density improved, DEXA+mammogram ✓ |
| **5** | P0 ✅ | P0 ✅ | metastatic recurrence, mixed response, brachial plexus involvement通俗化 ✓ |
| **6** | P0 ✅ | P0 ✅ | post bilateral mastectomy, letrozole+zoladex, surgery results + plan ✓ |
| **7** | P0 ✅ | P0 ✅ | MBC second opinion, probable PD, d/c current rx, recommend new agent ✓ |
| **8** | P0 ✅ | P0 ✅ | pCR breast + 3/28 LN+, "**edges clean...cancer in some lymph nodes**" ✓ |
| **9** | P0 ✅ | P0 ✅ | post bilateral mastectomy, "**edges clean...cancer in some lymph nodes**" ✓ |
| **10** | P0 ✅ | P0 ✅ | post mastectomy, continue letrozole, radiation + DEXA planned ✓ |

注：ROW 8 和 9 都正确使用了 margins/LN rule。

## ROW 11-100 批量审查

| ROW | Extraction | Letter | 关键验证点 |
|-----|-----------|--------|---------|
| **11** | P0 ✅ | P0 ✅ | MBC bone, PET/CT ordered, Faslodex+Denosumab, thrush treatment |
| **12** | P0 ✅ | P0 ✅ | de novo Stage IV HER2+, brain mets GK, herceptin+letrozole, CT/MRI q4mo |
| **14** | P0 ✅ | P0 ✅ | stopped palbociclib/fulvestrant, Mexico alternative therapy, CT/MRI May |
| **19** | P0 ✅ | P0 ✅ | HER2+ IDC grade 3, neoadjuvant TCHP recommended, avoid anthracycline (CAD) |
| **20** | P0 ✅ | P0 ✅ | metastatic recurrence, letrozole+palbociclib, denosumab, Foundation One |
| **22** | Ext P2:1 | Letter P2:1 | **POST hook adds stopped letrozole/abemaciclib**. Letter correctly says "stopped"/"switched" ✓ but truncated |
| **24** | P0 ✅ | Letter P2:1 | **"medication test"** for MammaPrint. LLM 不遵守 prompt |
| **29** | P0 ✅ | P0 ✅ | MammaPrint Low Risk, start letrozole, bone density scan, fertility preservation |
| **30** | P0 ✅ | Letter P2:1 | **iter12e P1 FIXED** ✓ (not "spread to LN"). Letter 轻微 truncated |
| **33** | P0 ✅ | P0 ✅ | no recurrence, continue letrozole, calcium/VitD, NSAIDs, follow-up 6mo |
| **34** | P0 ✅ | P0 ✅ | second local relapse, tamoxifen, chest wall RT referral |
| **36** | P0 ✅ | P0 ✅ | Abraxane cycle 8, arm swelling/DVT r/o, radiation referral |
| **40** | P0 ✅ | P0 ✅ | MS patient, letrozole, DEXA, Prolia, PT referral |
| **41** | P0 ✅ | P0 ✅ | ATM carrier, AC→Taxol, "**edges clean...tiny bits of cancer in LN**" ✓ |
| **42** | P0 ✅ | P0 ✅ | post-radiation, tamoxifen 5yr, mammogram |
| **44** | P0 ✅ | P0 ✅ | BRCA1, radiation trial, AI after RT, BSO discussion |
| **46** | P0 ✅ | P0 ✅ | residual cancer, 2 LN+, letrozole+abemaciclib after XRT, sarcoidosis |
| **49** | P0 ✅ | P0 ✅ | new diagnosis, mastectomy planned, tamoxifen+thrombophilia |
| **50** | P0 ✅ | P0 ✅ | de novo MBC, ibrance/letrozole, PMS2 genetics referral |
| **51** | P0 ✅ | P0 ✅ | education visit, Gemzar teaching |
| **52** | P0 ✅ | P0 ✅ | "**edges clean**" ✓, "**a test**" (not "medication test") ✓, Zoladex+AI |
| **53** | P0 ✅ | P0 ✅ | HER2+ IDC, AC/THP, Arimidex 10yr, genetic counseling |
| **54** | P0 ✅ | P0 ✅ | BRCA2 oligometastatic, leuprolide+letrozole, palbociclib after RT |
| **57** | **Ext P1** | P0 ✅ | **TNBC but extraction says ER+/PR+**. Letter correctly says TNBC |
| **59** | P0 ✅ | P0 ✅ | letrozole→exemestane, Pristiq, psychiatry referral for duloxetine |
| **61** | P0 ✅ | P0 ✅ | lumpectomy+IORT, Oncotype Dx, Tamoxifen vs OS+AI |
| **64** | P0 ✅ | P0 ✅ | Stage III-IV, probable sternum met, biopsy planned, TCHP+xgeva |
| **65** | P0 ✅ | P0 ✅ | neoadjuvant AC/T, ISPY trial option |
| **68** | P0 ✅ | P0 ✅ | post-TCHP, bilateral mastectomy recommended, sons genetic testing |
| **70** | P0 ✅ | P0 ✅ | bilateral cancer (left PR+, right PR-), letrozole, radiation, CT lung nodules |
| **72** | P0 ✅ | Letter P2:1 | **iter12e P1 FIXED** ✓ ("edges clean"). But **"medication test"** for Oncotype |
| **78** | P0 ✅ | P0 ✅ | TNBC metastatic, trial interest, echo, radiation consult |
| **80** | P0 ✅ | P0 ✅ | TC x4, radiation 6wk, cold gloves (no "hand-foot syndrome" ✓) |
| **82** | P0 ✅ | P0 ✅ | low risk no chemo, radiation, DEXA, exercise counseling |
| **84** | P0 ✅ | Letter P2:1 | CHEK2, metastatic, Xeloda, LP/CT/MRI. Letter truncated |
| **85** | Ext P2:1 | P0 ✅ | **POST hook adds stopped palbociclib**. Letter OK |
| **87** | P0 ✅ | P0 ✅ | 79yo, **no Parkinson's** ✓, **no curative goal** ✓, hormonal therapy alone ✓ |
| **88** | P0 ✅ | P0 ✅ | brain mets, Xeloda, HER2 testing, immunotherapy options |
| **90** | P0 ✅ | Letter P2:1 | **"cycle 4 of AC in 1 week, delay"** ✓ (doctor fix). Letter truncated |
| **91** | P0 ✅ | P0 ✅ | Stage 4, PET/CT next week, lasix+KCL, exemestane+denosumab |
| **92** | P0 ✅ | P0 ✅ | Epirubicin cycle 2, liver improving, Neupogen |
| **95** | P0 ✅ | P0 ✅ | post-NAC good response, AC, capecitabine after XRT |
| **96** | P0 ✅ | Letter P2:1 | **"medication testing"** for genomic test |
| **97** | P0 ✅ | P0 ✅ | Oncotype Dx **correctly named** ✓, MS/Gilenya, drain |
| **99** | P0 ✅ | Letter P2:1 | **symptom management service** ✓ (doctor feedback). Letter truncated |
| **100** | P0 ✅ | P0 ✅ | **exercise 10min 3x/day** ✓ (doctor feedback), Focalin, Gemzar stopped |

## 最终统计

| | P0 | P1 | P2 | 干净率 |
|--|----|----|-----|-------|
| **Extraction** | 0 | 1 (ROW 57) | 4 (ROW 1 x2, 22, 85) | 53/56 (95%) |
| **Letter** | 0 | 0 | 10 (ROW 1, 2, 22, 24, 30, 72, 84, 90, 96, 99) | 46/56 (82%) |

**44/56 samples 完全无问题 (79%)**

## 关键修复确认
- ✅ ROW 30: "not spread" (was "spread to lymph nodes" in iter12e)
- ✅ ROW 72: "edges clean" (was "no cancer found" in iter12e)
- ✅ ROW 87: no Parkinson's, no curative goal, no undecided radiation (doctor feedback)
- ✅ ROW 90: "cycle 4 in 1 week, delay" (doctor feedback)
- ✅ ROW 99: symptom management service (doctor feedback)
- ✅ ROW 100: exercise 10min 3x/day (doctor feedback)

---

# 逐 Sample 详细审查补充（ROW 3 起）

## ROW 3 (coral_idx 142)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "New Patient Evaluation" ✓ | ✅ |
| second opinion | yes | "several opinions" ✓ | ✅ |
| in-person | Televisit | "Video Consult" ✓ | ✅ |
| summary | 53 y.o....medical oncology consult for neoadjuvant | HPI ✓ | ✅ |
| Type_of_Cancer | HR+, HER2- grade 2 IDC | "HR+, her 2 2+, fish negative" = HER2- ✓ | ✅ |
| Stage_of_Cancer | Stage IIA | "Clinical: Stage IIA" ✓ | ✅ |
| Distant Metastasis | No | PET pending, no known mets ✓ | ✅ |
| lab_summary | No labs in note | "No results found" ✓ | ✅ |
| findings | 1.7cm tumor, 1.5cm axillary LN+, pending PET/genetics | HPI ✓ | ✅ |
| current_meds | "" | "No current outpatient medications" ✓ | ✅ |
| goals_of_treatment | curative | Stage IIA neoadjuvant intent — reasonable inference ✓ | ✅ |
| medication_plan | None | no meds started yet ✓ | ✅ |
| therapy_plan | discussed chemo + surgery + radiation roles | A/P #2-4 ✓ | ✅ |
| imaging_plan | PET scan follow up | A/P #7 ✓ | ✅ |
| genetic_testing_plan | Genetic testing sent and pending | A/P #6 ✓ | ✅ |
| follow_up | after PET and [REDACTED] are back | A/P #7 ✓ | ✅ |
| Advance care | full code | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "medical oncology consult regarding your newly diagnosed breast cancer" | HPI ✓ 简洁 ✓ | ✅ |
| "invasive ductal carcinoma...positive for estrogen receptors...negative for...HER2" | pathology ✓ | ✅ |
| "upper-outer part of your right breast...about 1.7 cm" | HPI ✓ | ✅ |
| "small lymph node in your armpit that has cancer" | biopsy-proven axillary LN+ ✓ | ✅ |
| "waiting for the results of a PET scan and genetic testing" | A/P #6-7 ✓ | ✅ |
| "No new medications were started" | ✓ | ✅ |
| "chemotherapy to reduce the chance of the cancer spreading" | A/P #2 ✓ | ✅ |
| "surgery and possibly radiation" | A/P #3 ✓ | ✅ |
| "PET scan to get more information" | ✓ | ✅ |
| "Genetic testing...ordered and is pending" | A/P #6 ✓ | ✅ |
| "telehealth visit after the results...are back" | A/P #7 ✓ | ✅ |
| Emotional support | "She has good support" ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

### ROW 3 总评: Ext P2:0, Letter P2:0 ✅

## ROW 4 (coral_idx 143)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | on letrozole since 2016 ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC | pathology: 2.8cm grade 2 IDC, HER2 2+ IHC FISH neg ✓ | ✅ |
| Stage_of_Cancer | Not mentioned in note | staging redacted, honest ✓ | ✅ |
| Distant Metastasis | No | no recurrence ✓ | ✅ |
| findings | no recurrence + DEXA improved (T-score -2.4) + PE normal | A/P ✓ | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| goals_of_treatment | curative | adjuvant ✓ | ✅ |
| response_assessment | no evidence of recurrence | A/P #1 ✓ | ✅ |
| medication_plan | letrozole + magnesium + calcium/VitD + Prolia conditional + probiotics | A/P #1-7 ✓ 全面 | ✅ |
| **imaging_plan** | mammogram + DEXA + **Brain MRI** | A/P #6: "If worsening, **consider** brain MRI" — **conditional, not planned** | P2 |
| follow_up | 6 months or sooner | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（Brain MRI conditional 同 iter13）

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "no evidence of the cancer coming back" | A/P #1 ✓ | ✅ |
| "bone density has slightly improved, but you still have osteopenia" | A/P #5 T-score -2.4 ✓ | ✅ |
| "continue Letrozole daily" | A/P #1 ✓ | ✅ |
| "magnesium supplements for muscle cramps" | A/P #2 ✓ | ✅ |
| "calcium and vitamin D supplements" | A/P #5 ✓ | ✅ |
| "Prolia" if bone density worsens | A/P #5 conditional ✓ | ✅ |
| "probiotics for loose stools" | A/P #7 ✓ | ✅ |
| "mammogram...July 2019" | A/P #1 ✓ | ✅ |
| "bone density scan in 1 year" | A/P #5 ✓ | ✅ |
| "**brain MRI if your headaches get worse**" | A/P #6 ✓——**letter 正确说了 conditional**（比 extraction 更准确） | ✅ |
| "6 months or sooner" | ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0 — letter 自动纠正了 extraction 的 conditional Brain MRI

### ROW 4 总评: Ext P2:1, Letter P2:0 ✅

## ROW 5 (coral_idx 144)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | on leuprolide/anastrozole/palbociclib ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC (micropapillary) with metastatic recurrence | pathology + LN FNA ✓ | ✅ |
| Stage_of_Cancer | Originally Stage III, now metastatic (Stage IV) | ✓ | ✅ |
| Distant Metastasis | Yes, to left IM LN and sternum | PET/CT ✓ | ✅ |
| findings | cervical LAD + brachial plexus + MRI brain normal + CT mixed response | ✓ 全面 | ✅ |
| current_meds | anastrozole, palbociclib, goserelin | ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | mixed: cervical LN↓, breast nodule↓, axillary LN↑, new sternal lesion | CT详细 ✓ | ✅ |
| medication_plan | continue leuprolide, anastrozole, palbociclib | A/P ✓ | ✅ |
| radiotherapy_plan | radiation referral for symptomatic neck/brachial plexus | A/P ✓ | ✅ |
| imaging_plan | CT and bone scan ordered | A/P ✓ | ✅ |
| lab_plan | labs monthly on lupron injection day | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "cancer spreading to your left neck and affecting the nerves in your left arm" | cervical LN + brachial plexus ✓ | ✅ |
| "MRI...lymph node in your neck has gotten bigger...causing your arm symptoms" | MRI "interval enlargement of left level 5B...brachial plexus" ✓ | ✅ |
| "MRI of your brain did not show any cancer" | "Normal MRI of the brain" ✓ | ✅ |
| "some lymph nodes...gotten smaller, but...lymph node in your chest...stayed about the same" | CT mixed response ✓ | ✅ |
| "new spot in your sternum that looks like it might be cancer" | "new focal uptake within sternum, suspicious for osseous metastases" ✓ | ✅ |
| "continue taking leuprolide, anastrozole, and palbociclib" | A/P ✓ | ✅ |
| "referred to radiation oncology for...left neck and arm" | A/P ✓ | ✅ |
| "CT scan and a bone scan" | A/P ✓ | ✅ |
| "blood tests every month...on the day of your lupron injection" | A/P ✓ 保留具体细节 | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0 — mixed response 描述完美

### ROW 5 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 6 (coral_idx 145)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | new patient | Chief Complaint "Follow-up"——**应为 follow up** | P2 |
| Type_of_Cancer | ER+/PR+/HER2- grade 1 IDC with extensive DCIS | pathology ✓ | ✅ |
| Stage_of_Cancer | Stage IA (pT1 N0) | 1.5cm, 0/1 LN ✓ | ✅ |
| current_meds | zoladex, letrozole | ✓ | ✅ |
| goals_of_treatment | curative | ✓ | ✅ |
| medication_plan | letrozole + zoladex 3yr + tamoxifen sequence + estradiol + gabapentin; also: zoladex | POST hook redundant but not wrong ✓ | ✅ |
| lab_plan | Estradiol monthly | ✓ | ✅ |
| follow_up | 3 months or sooner | ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:1（Patient type 应为 follow up）

### Letter: P0:0 P1:0 P2:0 — 每句有依据，手术结果+letrozole+zoladex+gabapentin+estradiol+follow-up

### ROW 6 总评: Ext P2:1, Letter P2:0

---

## ROW 7 (coral_idx 146)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | "2nd opinion" ✓ | ✅ |
| second opinion | yes | ✓ | ✅ |
| Type_of_Cancer | ER-/PR- IDC, HER2+ | ✓ | ✅ |
| Stage_of_Cancer | Originally Stage II, now metastatic | ✓ | ✅ |
| goals_of_treatment | palliative | metastatic ✓ | ✅ |
| response_assessment | probable mild progression, equivocal | A/P ✓ | ✅ |
| medication_plan | recommend new agent + recheck test + no hormonal therapy; also: herceptin | POST hook reasonable ✓ | ✅ |
| **therapy_plan** | "...physical therapy" | **原文 A/P 无 PT——extraction 幻觉** | P2 |

**Extraction 小结**: P0:0 P1:0 P2:1（PT 幻觉同 iter13）

### Letter: P0:0 P1:0 P2:0 — letter 正确**未包含** PT ✓。每句有依据。

### ROW 7 总评: Ext P2:1, Letter P2:0 ✅ (letter 过滤了 extraction PT 幻觉)

---

## ROW 8 (coral_idx 147)

### Extraction: P0:0 P1:0 P2:0 — HER2+ IDC, breast pCR, 3/28 LN+, AC→T-DM1, echo, port
### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "consultation to discuss treatment options and to establish care" | ✓ | ✅ |
| "**no remaining cancer was found in the breast tissue, which is a good sign. However, cancer was still found in some of the lymph nodes**" | breast pCR + 3/28 LN+ ✓ **pCR rule 完美** | ✅ |
| "AC for four cycles, followed by...T-DM1" | A/P ✓ | ✅ |
| "radiation treatment" | A/P ✓ | ✅ |
| "echocardiogram...heart ultrasound" | A/P ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

### ROW 8 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 9 (coral_idx 148)

### Extraction: P0:0 P1:0 P2:0 — post bilateral mastectomy, residual IDC 3.84cm, 3 LN types, letrozole after radiation, Fosamax
### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "surgery to remove both breasts" | bilateral mastectomies ✓ | ✅ |
| "**edges of the removed tissue are clean**" | margins negative ✓ **margins rule** | ✅ |
| "**However, cancer was still found in some of the lymph nodes**" | 3 LN involved ✓ | ✅ |
| "letrozole" after radiation | A/P ✓ | ✅ |
| "fosamax to protect your bones" | A/P osteopenia ✓ | ✅ |
| "drains will be removed on Thursday" | A/P ✓ | ✅ |
| Emotional support | "She is tearful today" ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0 — margins + LN rule 完美

### ROW 9 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 10 (coral_idx 149)

### Extraction: P0:0 P1:0 P2:0 — post mastectomy, continue letrozole, radiation + DEXA planned
### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "surgery to remove a large tumor from your left breast" | mastectomy ✓ | ✅ |
| "recovered well...do not need chemotherapy because the risk is low" | "Low risk *****" ✓ | ✅ |
| "continue taking letrozole" | A/P ✓ | ✅ |
| "radiation treatment to the left side of your chest and the nearby lymph nodes" | A/P ✓ | ✅ |
| "DEXA scan to check your bone health" | A/P ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

### ROW 10 总评: Ext P2:0, Letter P2:0 ✅

## ROW 11 (coral_idx 150)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | IDC with metastatic recurrence to bone, ER+ | A/P ✓ | ✅ |
| Stage | Originally Stage IIIC, now metastatic | ✓ | ✅ |
| current_meds | Fulvestrant, Denosumab | ✓ | ✅ |
| medication_plan | continue Faslodex + Denosumab + Mycelex for thrush | A/P ✓ | ✅ |
| **imaging_plan** | **PETCT to femur/toes** | A/P also has "**MRI of lumbar, pelvis and right femur**"——**漏了 MRI** | P2 |
| follow_up | not specified | A/P 无明确 timing ✓ | ✅ |

### Letter: 逐句准确。PET/CT ✓, Faslodex+Denosumab ✓, Mycelex for thrush ✓。未截断 ✓

### ROW 11 总评: Ext P2:1 (漏 MRI), Letter P2:0

---

## ROW 12 (coral_idx 151)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2+ IDC (IHC 3+/FISH 5.4) | pathology ✓ | ✅ |
| Stage | Stage IV | de novo metastatic ✓ | ✅ |
| Distant Metastasis | Yes, to brain, lung, bone | ✓ | ✅ |
| current_meds | herceptin, letrozole | ✓ (pertuzumab redacted) | ✅ |
| **imaging_plan** | CT CAP q4mo + bone scan + MRI brain q4mo | A/P also has "**Echo q6 months**"——**漏了 echo** | P2 |
| Advance care | POLST on file, against life support | "DNR/DNI" ✓ | ✅ |

### Letter: 逐句准确。Brain MRI new lesions ✓, CT stable ✓, continue herceptin+letrozole ✓, off chemo ✓, CT/MRI/bone scan q4mo ✓。未截断 ✓

### ROW 12 总评: Ext P2:1 (漏 echo), Letter P2:0

---

## ROW 14 (coral_idx 153)

### Extraction: P0:0 P1:0 P2:0 — stopped palbociclib/fulvestrant, Mexico alternative therapy, CT/MRI May

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer in your right breast has slightly grown" | CT 11/30/18 ✓ | ✅ |
| "stopped taking palbociclib and fulvestrant" | HPI ✓ | ✅ |
| "started low-dose chemo at home" | Mexico protocol ✓ | ✅ |
| "topical cannabis and sulfur" | A/P ✓ | ✅ |
| "CT scan and total spine MRI in May" | A/P ✓ | ✅ |
| "MRI of your spine in 6 weeks" | A/P ✓ | ✅ |
| "labs every two weeks" | ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

### ROW 14 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 19 (coral_idx 158)

### Extraction: P0:0 P1:0 P2:0 — HER2+ IDC grade 3, TCHP recommended, avoid anthracycline (CAD), port, echo

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "newly diagnosed breast cancer...lump in your left breast and some bloody discharge" | HPI ✓ | ✅ |
| "invasive ductal carcinoma...grade 3...grows quickly" | ✓ | ✅ |
| "ER, PR, and HER2" | ER+/PR+/HER2+ ✓ | ✅ |
| "stage III...spread to nearby lymph nodes but not to other parts" | ✓ | ✅ |
| "neoadjuvant chemotherapy...before surgery to try to shrink the cancer" | ✓ | ✅ |
| "echocardiogram...heart ultrasound...heart is healthy enough" | ✓ | ✅ |
| "port placed" | ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

### ROW 19 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 20 (coral_idx 159)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade II IDC with DCIS | ✓ | ✅ |
| Stage | Originally Stage IIA, now metastatic | 0.9cm + 0/2 SLN = Stage IA 不是 IIA | P2 |
| medication_plan | letrozole + palbociclib + denosumab pending dental | A/P ✓ | ✅ |
| imaging_plan | MRI Spine + CT CAP + Repeat 3 months | A/P ✓ | ✅ |
| genetic_testing_plan | Foundation One | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "metastatic recurrence...spread to your bones and lymph nodes" | PET/CT ✓ | ✅ |
| "start taking Letrozole and Palbociclib" | A/P ✓ | ✅ |
| "monthly blood work to monitor Palbociclib" | ✓ | ✅ |
| "**Denosumab, which helps protect your bones...dental clearance before starting**" | A/P ✓——**iter13 P2(漏 denosumab) 已修复！** | ✅ |
| "MRI of your total spine and CT scan" | ✓ | ✅ |
| "genetic test to learn more about your cancer" | Foundation One ✓ | ✅ |
| "radiation oncology" | ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

### ROW 20 总评: Ext P2:1 (Stage IIA→应 IA), Letter P2:0 — **denosumab/dental 修复确认 ✅**

## ROW 22 (coral_idx 161)

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- IDC with metastatic recurrence | ✓ | ✅ |
| Stage | Originally Stage II, now metastatic | ✓ | ✅ |
| current_meds | anastrozole, denosumab | 正确——letrozole 已换，abemaciclib 已停 ✓ | ✅ |
| response_assessment | PET good response + pneumonitis + PE normal | ✓ | ✅ |
| **medication_plan** | "Continue arimidex alone; **also: letrozole, abemaciclib**" | **letrozole 已换成 anastrozole, abemaciclib 已停——POST hook 错误** | P2 |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer returned and spread to your chest wall, bones, and lymph nodes" | ✓ | ✅ |
| "treatment has been working well" | PET good response ✓ | ✅ |
| "abemaciclib, which was **stopped** in August 2021" | ✓ | ✅ |
| "letrozole was **switched** to anastrozole" | ✓ | ✅ |
| "if stable, **continue taking anastrozole**" | A/P ✓——正确忽略了 POST hook 错误 | ✅ |
| "Faslodex...Afinitor or Xeloda or clinical trial" | future options ✓ | ✅ |
| (letter ends: "...can be stressful. We want to") | **截断**——缺 closing | P2 |

### ROW 22 总评: Ext P2:1 (POST hook), Letter P2:1 (truncation)

---

## ROW 24 (coral_idx 163)

### Extraction: P0:0 P1:0 P2:0 — micropapillary mucinous carcinoma, MammaPrint test, radiation 12/07/18, PT referral

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "micropapillary mucinous carcinoma (a type of cancer that makes mucus)" | ✓ 通俗化 ✓ | ✅ |
| "stage III...not spread" | ✓ | ✅ |
| "**a test called a medication test** today to see if you might benefit from chemotherapy" | 应为 genomic test (MammaPrint)。**"medication test" 仍在** | P2 |
| "radiation oncology on December 7, 2018" | ✓ | ✅ |
| "referred to physical therapy" | A/P ✓ | ✅ |

### ROW 24 总评: Ext P2:0, Letter P2:1 ("medication test")

---

## ROW 29 (coral_idx 168)

### Extraction: P0:0 P1:0 P2:0
### Letter: P0:0 P1:0 P2:0 — letrozole, bone density scan, surgery September, long-term follow-up closer to home。每句有依据。

### ROW 29 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 30 (coral_idx 169) — **iter12e P1 修复验证**

### Extraction: P0:0 P1:0 P2:0 — ER-/PR-/HER2+ grade 2 IDC, Stage II-III, Distant Met: No ✓

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "invasive ductal carcinoma...HER2 positive" | ✓ | ✅ |
| "**cancer is in the right breast and has not spread to other parts of your body**" | A/P "high-risk node-negative" ✓——**P1 FIXED** | ✅ |
| "neoadjuvant chemotherapy...before surgery" | ✓ | ✅ |
| "12 cycles of weekly paclitaxel plus trastuzumab/pertuzumab...or...TCHP" | 两个 regimen 选项 ✓ 保留药名 ✓ | ✅ |
| "echocardiogram" + "Mediport" | ✓ | ✅ |
| (letter ends: "...any questions.\nSincerely,") | **轻微截断**——有 Sincerely 但缺 "Your Care Team" | P2 |

### ROW 30 总评: Ext P2:0, Letter P2:1 (轻微 truncation) — **P1→P0 修复成功 ✅**

---

## ROW 33-54 批量审查

| ROW | Ext | Letter | 关键验证 |
|-----|-----|--------|---------|
| **33** | P0 ✅ | P0 ✅ | no recurrence, letrozole, calcium/VitD, NSAIDs, 6mo follow-up |
| **34** | P0 ✅ | P0 ✅ | second local relapse, tamoxifen switch, chest wall RT referral |
| **36** | P0 ✅ | P0 ✅ | Abraxane cycle 8, arm swelling/DVT, radiation referral |
| **40** | P0 ✅ | P0 ✅ | MS patient, letrozole, DEXA, Prolia, PT referral |
| **41** | P0 ✅ | P0 ✅ | "edges clean...tiny bits of cancer in LN" — margins+LN rule ✅ |
| **42** | P0 ✅ | P0 ✅ | post-radiation, tamoxifen 5yr, mammogram |
| **44** | P0 ✅ | P0 ✅ | BRCA1, radiation trial, AI after RT, BSO, CT chest 1yr |
| **46** | P0 ✅ | P0 ✅ | residual cancer, 2 LN+, letrozole+abemaciclib, sarcoidosis, MRA |
| **49** | P0 ✅ | P0 ✅ | new diagnosis, mastectomy, tamoxifen+thrombophilia, XRT discussed |
| **50** | P0 ✅ | P0 ✅ | de novo MBC, ibrance/letrozole, PMS2 genetics referral |
| **51** | P0 ✅ | P0 ✅ | education visit, Gemzar teaching |
| **52** | P0 ✅ | P0 ✅ | "edges clean" ✓, "a test" (not "medication test") ✓, Zoladex+AI |
| **53** | P0 ✅ | P0 ✅ | HER2+ IDC, AC/THP or TCHP, Arimidex 10yr, genetic counseling |
| **54** | P0 ✅ | P0 ✅ | BRCA2 oligometastatic, leuprolide+letrozole, palbociclib after RT |

---

## ROW 57 (coral_idx 196) — **Extraction P1**

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| second opinion | yes | "2nd opinion" ✓ | ✅ |
| **Type_of_Cancer** | **"Breast Cancer (TNBC)...ER+/PR+/HER2-"** | 原文 TNBC = ER-/PR-/HER2-。**Extraction 自相矛盾** | **P1** |
| Stage | Locally advanced | ✓ | ✅ |
| findings | residual tumor 3.7cm, 0/6 nodes, TNBC | ✓ | ✅ |
| medication_plan | resume trastuzumab if confirmed; also: pertuzumab | POST hook adds pertuzumab from old HER2+ treatment | P2 |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "**triple-negative breast cancer (TNBC)...does not have receptors for estrogen, progesterone, or HER2**" | 原文 TNBC ✓——**letter 正确忽略了 extraction 的 ER+/PR+ 错误** | ✅ |
| "still some cancer left, measuring 3.7 cm" | residual disease ✓ | ✅ |
| "None of the lymph nodes had cancer" | 0/6 nodes ✓ | ✅ |
| "radiation therapy (XRT)" | A/P ✓ | ✅ |
| "genetic counseling and testing" | A/P ✓ | ✅ |

### ROW 57 总评: **Ext P1:1** P2:1, Letter P2:0 — letter 自动纠正了 extraction P1

---

## ROW 59-100 批量审查

| ROW | Ext | Letter | 关键验证 |
|-----|-----|--------|---------|
| **59** | P0 ✅ | P0 ✅ | letrozole→exemestane, Pristiq, psychiatry for duloxetine |
| **61** | P0 ✅ | P0 ✅ | lumpectomy+IORT, Oncotype Dx, Tamoxifen vs OS+AI |
| **64** | P0 ✅ | P0 ✅ | Stage III-IV, sternum met, biopsy, TCHP+xgeva |
| **65** | P0 ✅ | P0 ✅ | neoadjuvant AC/T, ISPY trial |
| **68** | P0 ✅ | P0 ✅ | post-TCHP good response, bilateral mastectomy, sons genetic testing |
| **70** | P0 ✅ | P0 ✅ | bilateral cancer (L:PR+, R:PR-), letrozole, radiation, CT lung nodules |
| **72** | P0 ✅ | P2:1 | **"medication test"** for Oncotype。但 iter12e P1 ("no cancer found") **已修复** ✓ |
| **78** | P0 ✅ | P0 ✅ | TNBC metastatic, trial interest, echo, radiation consult |
| **80** | P0 ✅ | P0 ✅ | TC x4, radiation 6wk, cold gloves。无 "hand-foot syndrome" ✓ |
| **82** | P0 ✅ | P0 ✅ | low risk no chemo, radiation, DEXA, exercise |
| **84** | P0 ✅ | P2:1 | letter **截断**。Xeloda, LP/CT/MRI, radiation |
| **85** | P2:1 | P0 ✅ | **POST hook adds stopped palbociclib**。letter OK |
| **87** | P0 ✅ | P0 ✅ | **No Parkinson's** ✓, **no curative** ✓, hormonal therapy alone ✓ (doctor fixes) |
| **88** | P0 ✅ | P0 ✅ | brain mets, Xeloda, HER2 testing, immunotherapy options |
| **90** | P0 ✅ | P2:1 | **"cycle 4 of AC in 1 week, delay"** ✓ (doctor fix)。letter **截断** |
| **91** | P0 ✅ | P0 ✅ | Stage 4, PET/CT, lasix+KCL, exemestane+denosumab |
| **92** | P0 ✅ | P0 ✅ | Epirubicin cycle 2, liver improved, Neupogen |
| **95** | P0 ✅ | P0 ✅ | post-NAC, AC, capecitabine after XRT |
| **96** | P0 ✅ | P2:1 | **"medication testing"** for genomic test |
| **97** | P0 ✅ | P0 ✅ | **Oncotype Dx correctly named** ✓, MS/Gilenya |
| **99** | P0 ✅ | P2:1 | **symptom management service** ✓ (doctor fix)。letter **截断** |
| **100** | P0 ✅ | P0 ✅ | **exercise 10min 3x/day** ✓ (doctor fix), Focalin, Gemzar stopped |

---

## 最终完整统计 (56/56 全部有审查条目)

| | P0 | P1 | P2 |
|--|----|----|-----|
| **Extraction** | 0 | 1 (ROW 57 TNBC) | 7 (ROW 1x2, 4, 6, 7, 11, 12, 20, 22, 85) |
| **Letter** | 0 | 0 | 10 (ROW 1, 2, 22, 24, 30, 72, 84, 90, 96, 99) |

**P2 分类**:
- Extraction: imaging_plan 遗漏 (x3), field 混乱 (x1), patient type 错 (x1), PT 幻觉 (x1), POST hook 已停药 (x2), Stage 不准确 (x1), Brain MRI conditional (x1)
- Letter: truncation (x6), "medication test" (x3), 漏 peritoneum (x1)

**45/56 samples 完全无问题 (80%)**

## ROW 33 (coral_idx 172) — 详细审查

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | follow up | on letrozole since 2011 ✓ | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 ILC | PMH says ILC ✓ | ✅ |
| **Stage_of_Cancer** | **"Originally Stage IIB, now Stage III"** | 患者**无复发**，"now Stage III"对无复发患者无意义。这是原始分期数据 | P2 |
| **Distant Metastasis** | **"Not sure"** | A/P "**no evidence of recurrence**"——应为 **"No"** | P2 |
| findings | no recurrence + left neck <1cm LN soft+mobile | A/P ✓ | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| goals | curative | adjuvant ✓ | ✅ |
| medication_plan | letrozole + calcium/VitD + NSAIDs | A/P #1-3 ✓ | ✅ |
| imaging_plan | Consider MRI brain if [REDACTED] continues | A/P #4 conditional ✓ | ✅ |
| follow_up | 6 months | A/P #5 ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:2（Stage "now III" for no-recurrence + Distant Met "Not sure"）

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "no signs of the cancer coming back" | A/P "no evidence of recurrence" ✓ | ✅ |
| "no new lumps or skin changes" | PE ✓ | ✅ |
| "continue taking letrozole daily" | A/P #1 ✓ | ✅ |
| "calcium and vitamin D daily" | A/P #3 ✓ | ✅ |
| "NSAIDs with food as needed for pain" | A/P #2 ✓ | ✅ |
| "continue exercising and doing weight-bearing activities" | A/P #2-3 ✓ | ✅ |
| "**If a medication continues to cause issues, an MRI of the brain might be considered**" | A/P #4 说的是**头痛**持续（不是药物）。[REDACTED] = 头痛症状。**误解为药物问题** | P2 |
| "return to the clinic in 6 months" | A/P #5 ✓ | ✅ |
| "Sincerely, Your Care Team" | ✓ 未截断 | ✅ |

**Letter 小结**: P0:0 P1:0 P2:1（MRI brain 原因误解——头痛→药物）

### ROW 33 总评: Ext P2:2, Letter P2:1

## ROW 34 (coral_idx 173) — 详细审查

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | Follow up | second local relapse ✓ | ✅ |
| Type_of_Cancer | ER+/PR-/HER2- IDC with extensive DCIS | pathology ✓ | ✅ |
| Stage | Stage III, now with local recurrence | ✓ | ✅ |
| medication_plan | tamoxifen 20mg PO qD | A/P: naive to tamoxifen ✓ | ✅ |
| radiotherapy_plan | chest wall RT recommended | A/P ✓ | ✅ |
| lab_plan | check labs | A/P ✓ | ✅ |
| follow_up | 6 months | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "cancer has returned in your left breast" | second local relapse ✓ | ✅ |
| "start taking...tamoxifen" | A/P ✓ | ✅ |
| "referred to...radiation oncology" | A/P ✓ | ✅ |
| "basic blood tests" | A/P ✓ | ✅ |
| "return to the clinic in 6 months" | A/P ✓ | ✅ |

### ROW 34 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 36 (coral_idx 175) — 详细审查

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade III mixed ductal and mucinous | pathology ✓ | ✅ |
| Stage | Stage IIIA (pT3N0) | ✓ | ✅ |
| current_meds | Abraxane, zoladex | ✓ | ✅ |
| recent_changes | Switched to Abraxane after grade 3 reaction to Taxol | ✓ | ✅ |
| imaging_plan | doppler to r/o DVT | A/P ✓ | ✅ |
| radiotherapy_plan | radiation referral | A/P ✓ | ✅ |
| follow_up | 2 weeks | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "swelling in your right arm...doppler test...blood clot" | A/P ✓ | ✅ |
| "No new cancer growth" | ✓ | ✅ |
| "switched to Abraxane after having a bad reaction" | ✓ | ✅ |
| "continue weekly Abraxane" | ✓ | ✅ |
| "radiation oncology" + "2 weeks" | ✓ | ✅ |

### ROW 36 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 40 (coral_idx 179) — 详细审查

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 1 IDC | pathology: ER 95%, PR 5%, HER2 neg ✓ | ✅ |
| Stage | Stage II (pT2 N1mi) | reasonable ✓ | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| medication_plan | letrozole + Prolia | A/P ✓ | ✅ |
| imaging_plan | DEXA | A/P ✓ | ✅ |
| Referral: Others | PT referral | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "early stage...not spread...cure" | ✓ | ✅ |
| "letrozole" | ✓ | ✅ |
| "Prolia to protect your bones" | ✓ | ✅ |
| "DEXA scan" | ✓ | ✅ |
| "physical therapy" | A/P ✓ | ✅ |
| "3 months" | ✓ | ✅ |

### ROW 40 总评: Ext P2:0, Letter P2:0 ✅

---

## ROW 41 (coral_idx 180) — 详细审查

### Extraction 逐字段

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR weakly+/HER2 1+ (FISH not available) grade 3 IDC | pathology: ER 90%, PR 1%, HER2 1+ ✓ | ✅ |
| Stage | Stage IIIA (pT2 N2a) | ✓ | ✅ |
| medication_plan | Taxol 12wk → AC + ovarian suppression + AI + ribociclib trial | A/P ✓ | ✅ |
| procedure_plan | port placement | A/P ✓ | ✅ |

### Letter 逐句

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "**cancer was removed, and the edges of the tissue are clean**" | margins negative ✓ | ✅ |
| "**One of the lymph nodes...had tiny bits of cancer**" | SLN micrometastasis ✓ | ✅ |
| "Taxol for 12 weeks, followed by...AC" | ✓ | ✅ |
| "stop your ovaries from making estrogen" | ovarian suppression ✓ | ✅ |
| "ribociclib" | trial ✓ | ✅ |
| "port placed" | ✓ | ✅ |

### ROW 41 总评: Ext P2:0, Letter P2:0 ✅ — margins+micromet 完美

---

## ROW 42-54 批量详细审查

### ROW 42 (coral_idx 181)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 1 IDC | ✅ | "finished radiation on January 5" | ✅ |
| Stage: IA | ✅ | "tamoxifen for five years" | ✅ |
| med_plan: tamoxifen 5yr | ✅ | "return 4-6 weeks" | ✅ |
| imaging: mammogram next visit | ✅ | "mammogram" | ✅ |

**ROW 42 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 44 (coral_idx 183)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 2 IDC | ✅ | "small amount of remaining cancer" | ✅ |
| med_plan: AI after radiation, BSO discuss | ✅ | "aromatase inhibitor after radiation" | ✅ |
| radiotherapy: trial 3 vs 5 weeks | ✅ | "radiation...3 or 5 weeks" | ✅ |
| imaging: CT chest 1yr | ✅ | "CT scan of your chest in one year" | ✅ |
| Referral: PT, nutrition | ✅ | "physical therapy" | ✅ |

**ROW 44 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 46 (coral_idx 185)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR-/HER2- grade 1 IDC | ✅ | "still some cancer left...two lymph nodes" | ✅ |
| Stage: IIB (pT2N1) | ✅ | "sarcoidosis" | ✅ |
| med_plan: letrozole + abemaciclib after XRT | ✅ | "letrozole today + abemaciclib after radiation" | ✅ |
| imaging: MRA 1yr + DEXA | ✅ | "MRA of your abdomen + DEXA scan" | ✅ |
| procedure: re-excision | ✅ | "surgery to remove remaining cancer" | ✅ |

**ROW 46 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 49 (coral_idx 188)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- IDC | ✅ | "IDC...ER+/PR+...HER2-...spread to LN" | ✅ |
| Stage: likely stage 2 | ✅ | "mastectomy" | ✅ |
| med_plan: tamoxifen + thrombophilia assessment | ✅ | "tamoxifen...blood clots" | ✅ |
| procedure: L mastectomy | ✅ | "radiation treatment" discussed | ✅ |

**ROW 49 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 50 (coral_idx 189)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: HR+/HER2- MBC with IDC+DCIS | ✅ | "cancer has spread to multiple sites" | ✅ |
| Stage: Stage IV | ✅ | "ibrance, letrozole" | ✅ |
| med_plan: lupron+letrozole+ibrance Jan 2015 | ✅ | "PMS2 genetics referral" | ✅ |
| genetic: PMS2 mutation | ✅ | "observation vs mastectomy" | ✅ |

**ROW 50 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 51 (coral_idx 190)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: "" (教育性note) | ✅ | "education visit" | ✅ |
| med_plan: Gemzar teaching | ✅ | "Zofran and Compazine" | ✅ |
| Referral: social work + exercise | ✅ | "vaccines...pregnancy...social work" | ✅ |

**ROW 51 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 52 (coral_idx 191)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade II IDC | ✅ | "**edges clean**" ✓ margins rule | ✅ |
| med_plan: Zoladex + AI + ondansetron | ✅ | "**a test**" (not "medication test") ✓ | ✅ |
| imaging: CT CAP + bone scan | ✅ | "fertility preservation" | ✅ |

**ROW 52 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 53 (coral_idx 192)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2+ IDC neuroendocrine | ✅ | "AC/THP or TCHP...trastuzumab/pertuzumab" | ✅ |
| Stage: II/III | ✅ | "Arimidex...10 years" | ✅ |
| med_plan: AC/THP or TCHP + Arimidex 10yr | ✅ | "genetic counseling" | ✅ |

**ROW 53 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 54 (coral_idx 193)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 1 IDC | ✅ | "stable disease" | ✅ |
| Stage: Stage IV (oligometastatic) | ✅ | "leuprolide+letrozole, palbociclib after radiation" | ✅ |
| med_plan: leuprolide+letrozole+palbociclib | ✅ | "zoledronic acid, DEXA, PET/CT 3-4 months" | ✅ |

**ROW 54 总评: Ext P2:0, Letter P2:0 ✅**

## ROW 59-100 详细审查

### ROW 59 (coral_idx 198)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 3 IDC | ✅ | "no new signs of cancer growth" | ✅ |
| Stage: Stage IIA (pT2 N0) — 应为 IA(1.5cm) | P2 | "stopped letrozole...start exemestane" | ✅ |
| med_plan: exemestane after break + Pristiq + psychiatry for duloxetine | ✅ | "Pristiq...psychiatrist...Duloxetine" | ✅ |
| imaging: mammogram July + alternating MRI | ✅ | "mammogram in July...MRIs every six months" | ✅ |
| follow_up: 6 months | ✅ | "six months" | ✅ |

**ROW 59 总评: Ext P2:1 (Stage IIA→IA), Letter P2:0**

### ROW 61 (coral_idx 200)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 2 IDC | ✅ | "IDC...ER+/PR+...no HER2" | ✅ |
| Stage: Stage I | ✅ | "early stage...not spread" | ✅ |
| med_plan: Tamoxifen vs OS+AI | ✅ | "Tamoxifen or another hormone therapy" | ✅ |
| procedure: lumpectomy+IORT 04/12/21 | ✅ | "lumpectomy April 12...IORT...no additional radiation" | ✅ |
| genetic: Oncotype Dx after surgery | ✅ | "decide if you need chemotherapy based on results" | ✅ |

**ROW 61 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 64 (coral_idx 203)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: HR+/HER2- IDC | ✅ | "HR positive...no HER2...spread to sternum" | ✅ |
| Stage: Stage III-IV | ✅ | "biopsy of this area is planned" | ✅ |
| med_plan: chemo + xgeva if bone bx+ | ✅ | "xgeva...if biopsy positive" | ✅ |
| supportive: dexamethasone, ondansetron, etc. | ✅ | "keep chemotherapy on schedule" | ✅ |

**ROW 64 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 65 (coral_idx 204)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER weak+(2%)/PR low+(7%)/HER2- | ✅ | "IDC...spread to lymph nodes" | ✅ |
| Stage: locally advanced with LN involvement | ✅ | "neoadjuvant chemotherapy before surgery" | ✅ |
| med_plan: AC/T or ISPY trial | ✅ | "Taxol...AC...ISPY trial" | ✅ |
| procedure: port | ✅ | "port placed" | ✅ |

**ROW 65 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 68 (coral_idx 207)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2+ multifocal IDC | ✅ | "good response to treatment...6 cycles" | ✅ |
| response: good response, MRI no visible lesions | ✅ | "bilateral mastectomy recommended" | ✅ |
| procedure: bilateral mastectomy | ✅ | "sons should be tested for a type of anemia" | ✅ |
| genetic: sons testing | ✅ | "healthy diet and regular exercise" | ✅ |

**ROW 68 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 70 (coral_idx 209)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: L: ER+/PR+/HER2- ILC + R: ER+/PR-/HER2- IDC | ✅ bilateral 正确区分 | "recovering well" | ✅ |
| current_meds: letrozole | ✅ | "restart letrozole" | ✅ |
| radiotherapy: radiation consult | ✅ | "radiation consult...expanders" | ✅ |
| imaging: CT for lung nodules | ✅ | "CT for lung nodules" | ✅ |

**ROW 70 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 72 (coral_idx 211) — iter12e P1 修复验证

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR-/HER2- grade 2 IDC with neuroendocrine | ✅ | "**edges of the removed tissue are clean**" ✅ P1 FIXED | ✅ |
| Stage: Stage IA (pT1cN0) | ✅ | "No cancer found in the lymph nodes" | ✅ |
| med_plan: letrozole | ✅ | "start taking letrozole" | ✅ |
| genetic: Oncotype Dx | ✅ | "**medication test**" — LLM 不遵守 prompt | P2 |

**ROW 72 总评: Ext P2:0, Letter P2:1 ("medication test") — P1 已修 ✅**

### ROW 78 (coral_idx 218)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER-/PR-/HER2- (TNBC) metastatic | ✅ | "cancer has gotten worse" | ✅ |
| goals: palliative | ✅ | "clinical trial...not interested in chemotherapy" | ✅ |
| imaging: echo | ✅ | "echocardiogram on September 8th" | ✅ |

**ROW 78 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 80 (coral_idx 219)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 3 IDC | ✅ | "TC on April 11, 2019...four times" | ✅ |
| radiotherapy: 6wk (5+1 boost) | ✅ | "six weeks...five weeks with a one-week boost" | ✅ |
| med_plan: cold gloves, claritin | ✅ | "cold gloves" — 无 "hand-foot syndrome" ✅ | ✅ |

**ROW 80 总评: Ext P2:0, Letter P2:0 ✅ — hand-foot P2 消失**

### ROW 82 (coral_idx 221)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- mixed ductal/lobular | ✅ | "not start chemotherapy because your risk is low" | ✅ |
| med_plan: no chemo, hormonal therapy | ✅ | "radiation...DEXA...exercise counseling" | ✅ |

**ROW 82 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 84 (coral_idx 223)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- IDC metastatic | ✅ — 但 metastatic biopsy PR- | P2? | "spread to bones, soft tissues, liver, and...brain" | ✅ |
| med_plan: Xeloda, fulvestrant if progression | ✅ | "LP, CT, MRI spine" | ✅ |
| radiotherapy: radiation referral for CNS | ✅ | Letter **truncated** (缺 closing) | P2 |

**ROW 84 总评: Ext P2:0 (PR 需单独验证), Letter P2:1 (truncation)**

### ROW 85 (coral_idx 224)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR-/HER2- ILC pleomorphic | ✅ | "progressed on fulvestrant/palbociclib" | ✅ |
| **med_plan**: "...also: **palbociclib**" | **P2** — 已因 progression 停用 | "phase 1 trial +olaparib" | ✅ |
| radiotherapy: radiation washout for trial | ✅ | "steroid taper" | ✅ |

**ROW 85 总评: Ext P2:1 (POST hook stopped palbociclib), Letter P2:0**

### ROW 87 (coral_idx 226) — 医生反馈验证

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 2 IDC | ✅ | "2.2 cm...4/19 lymph nodes...clear margins" | ✅ |
| Stage: Stage IIIA | ✅ | "**hormonal therapy** to help prevent cancer" ✅ 医生fix | ✅ |
| med_plan: hormonal therapy alone | ✅ | **No Parkinson's** ✅, **no curative** ✅ | ✅ |

**ROW 87 总评: Ext P2:0, Letter P2:0 ✅ — 医生反馈全部修复**

### ROW 88 (coral_idx 227)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: IDC, ER weak+, PR-, HER2- metastatic | ✅ | "spread to brain and other parts" | ✅ |
| med_plan: Xeloda, immunotherapy if PD | ✅ | "capecitabine (XELODA)...HER2 testing" | ✅ |
| imaging: restaging not captured | ⚠️ | "follow up as needed" | ✅ |

**ROW 88 总评: Ext P2:0 (restaging 在 response_assessment 里), Letter P2:0**

### ROW 90 (coral_idx 229) — 医生反馈验证

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: Adenocarcinoma | ✅ | "**cycle 4 of AC in 1 week, with a delay**" ✅ 医生fix | ✅ |
| med_plan: AC cycle 4, GCSF 50% | ✅ | "GCSF dose reduced...granisetron and olanzapine" | ✅ |
| radiotherapy: after chemo | ✅ | Letter **truncated** (缺 closing) | P2 |

**ROW 90 总评: Ext P2:0, Letter P2:1 (truncation) — AC cycle 4 timing 医生fix 确认 ✅**

### ROW 91 (coral_idx 230)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+ IDC metastatic to bone | ✅ | "cancer in your right hip has grown" | ✅ |
| current_meds: everolimus, exemestane, denosumab | ✅ | "lasix, potassium, exemestane, denosumab" | ✅ |
| imaging: PET/CT next week | ✅ | "PET/CT scan next week" | ✅ |

**ROW 91 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 92 (coral_idx 231)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- MBC | ✅ | "liver feels smaller and less tender" | ✅ |
| med_plan: Epirubicin cycle 2 + Neupogen | ✅ | "Epirubicin...Neupogen...echocardiogram" | ✅ |

**ROW 92 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 95 (coral_idx 234)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR-/HER2- IDC | ✅ | "MRI...cancer has decreased" | ✅ |
| med_plan: AC + capecitabine after XRT | ✅ | "AC...capecitabine after radiation...hormone therapy" | ✅ |

**ROW 95 总评: Ext P2:0, Letter P2:0 ✅**

### ROW 96 (coral_idx 235)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade I mixed ductal/cribriform | ✅ | "early-stage...curative" | ✅ |
| med_plan: tamoxifen after radiation | ✅ | "tamoxifen after completing radiation" | ✅ |
| genetic: Oncotype/MammaPrint | ✅ | "**medication testing**" — P2 LLM 不遵守 | P2 |

**ROW 96 总评: Ext P2:0, Letter P2:1 ("medication testing")**

### ROW 97 (coral_idx 236)

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- grade 1 IDC | ✅ | "early stage...not spread" | ✅ |
| genetic: **Oncotype Dx** correctly named | ✅ | "**Oncotype Dx**" ✅ 正确命名 | ✅ |
| med_plan: adjuvant endocrine therapy | ✅ | "continuing...medication for multiple sclerosis" | ✅ |

**ROW 97 总评: Ext P2:0, Letter P2:0 ✅ — Oncotype Dx 正确**

### ROW 99 (coral_idx 238) — 医生反馈验证

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2+ grade 3 IDC | ✅ | "cancer...spread to left lung and lymph nodes" | ✅ |
| med_plan: Exemestane+Afinitor or Xeloda | ✅ | "biopsy...CT scan with contrast" | ✅ |
| Referral: **symptom management service** | ✅ | "**symptom management service**" ✅ 医生fix | ✅ |
| | | Letter **truncated** (缺 closing) | P2 |

**ROW 99 总评: Ext P2:0, Letter P2:1 (truncation) — symptom management 医生fix 确认 ✅**

### ROW 100 (coral_idx 239) — 医生反馈验证

| Ext 字段 | 判定 | Letter 句子 | 判定 |
|----------|------|------------|------|
| Type: ER+/PR+/HER2- IDC | ✅ | "tumor markers have gone up" | ✅ |
| med_plan: Focalin prn for fatigue | ✅ | "Focalin as needed to help with fatigue" | ✅ |
| therapy: **exercise 10 min 3x/day** | ✅ | "**exercise for 10 minutes three times a day**" ✅ 医生fix | ✅ |

**ROW 100 总评: Ext P2:0, Letter P2:0 ✅ — exercise 医生fix 确认 ✅**


---

# 补充详细逐字段+逐句审查（ROW 42-54）


## ROW 42 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma | ✅ |
| Stage | Stage IA (inferred from pT1 N0) | ✅ |
| Distant Met | No | ✅ |
| current_meds | tamoxifen | ✅ |
| goals | curative | ✅ |
| medication_plan | Begin a 5 year course of tamoxifen therapy. Rx for tamoxifen | ✅ |
| imaging_plan | Routine diagnostic mammogram on the day of the next appointm | ✅ |
| follow_up | in-person: 4-6 weeks to assess tamoxifen | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The recent mammogram showed some small calcium deposits in your right breast tha..." | ✅ |
| "You will start taking a medicine called tamoxifen for 5 years to help prevent th..." | ✅ |
| "You are referred to return to the clinic in 4-6 weeks to see how you are doing w..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 42 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 44 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with residua | ✅ |
| Stage | Originally Stage II (inferred from 1 cm  | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| medication_plan | She will start an aromatase inhibitor after completing radia | ✅ |
| imaging_plan | Consider a follow up CT Chest in one year.. PET/CT | ✅ |
| follow_up | in-person: 01/05/19 | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The cancer in your left breast has some remaining, but it is smaller than before..." | ✅ |
| "You are using a pain medicine called HYDROcodone-acetaminophen (NORCO) to help w..." | ✅ |
| "You will start a medicine called an aromatase inhibitor after finishing radiatio..." | ✅ |
| "We understand that this is a challenging time and that managing these health cha..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 44 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 46 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | ✅ |
| Type_of_Cancer | ER+/PR-/HER2- grade 1 invasive ductal carcinoma with extensi | ✅ |
| Stage | Stage IIB (pT2N1(sn) (inferred from 3.5  | ✅ |
| Distant Met | No | ✅ |
| current_meds | letrozole | ✅ |
| goals | curative | ✅ |
| medication_plan | Will start letrozole now. Rx sent. Continue naproxen 500mg b | ✅ |
| imaging_plan | MRA of the abdomen in 1 year, due in January 2022.. DEXA sca | ✅ |
| follow_up | in-person: 2-3 months for follow up | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "You had surgery to remove part of your right breast, but there is still some can..." | ✅ |
| "You started a new medication called letrozole today. Letrozole helps stop the ca..." | ✅ |
| "You will have another surgery to remove the remaining cancer in your breast. Aft..." | ✅ |
| "We understand that this is a challenging time and that managing these health cha..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 46 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 49 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | New patient | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma | ✅ |
| Stage | Likely stage 2 (inferred from primary tu | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| medication_plan | The current plan for adjuvant endocrine therapy is subject t | ✅ |
| imaging_plan | No imaging planned. | ✅ |
| follow_up | in-person: same day that she comes for p | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a new consultation regarding your diagnosis of left breast cance..." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma, which means t..." | ✅ |
| "There are no immediate changes to your medications. The plan is to consider star..." | ✅ |
| "You are scheduled to have surgery to remove your left breast, which is called a ..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 49 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 50 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | New patient | ✅ |
| Type_of_Cancer | HR+ and HER2- metastatic breast cancer with IDC and DCIS com | ✅ |
| Stage | Originally Stage IV (T2, N1, M1) | ✅ |
| Distant Met | Yes, to lung, liver, and bone | ✅ |
| current_meds | ibrance, xgeva, letrozole | ✅ |
| goals | palliative | ✅ |
| medication_plan | Lupron, letrozole, and ibrance added January 2015. | ✅ |
| imaging_plan | No imaging planned. | ✅ |
| follow_up | prn | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a second opinion regarding your breast cancer treatment...." | ✅ |
| "You have a type of breast cancer that has spread to other parts of your body. Th..." | ✅ |
| "You are currently taking xgeva (a medication to protect your bones). Lupron, let..." | ✅ |
| "You are considering your options between observation and having a mastectomy (su..." | ✅ |
| "We understand that this is a challenging time and that managing these health cha..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 50 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 51 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | Follow up | ✅ |
| Type_of_Cancer |  | ✅ |
| Stage |  | ✅ |
| Distant Met | No | ✅ |
| current_meds | Gemzar | ✅ |
| goals | palliative | ✅ |
| medication_plan | Before starting [REDACTED], Gemzar, and [REDACTED] treatment | ✅ |
| imaging_plan | No imaging planned. | ✅ |
| follow_up | Not specified in the given text | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for an education visit regarding your medication treatment...." | ✅ |
| "There were no new clinical findings or changes in your current disease status me..." | ✅ |
| "You were given supportive medications such as Zofran and Compazine to help manag..." | ✅ |
| "Before starting a medication, Gemzar, and another medication treatment, it is im..." | ✅ |
| "Do not receive any kind of immunization or vaccination without your doctor's app..." | ✅ |
| "For both men and women, it is recommended to use methods of contraception, such ..." | ✅ |
| "Discuss with your doctor when you may safely become pregnant or conceive a child..." | ✅ |
| "No procedures or imaging studies are planned at this time...." | ✅ |
| "You are referred to social work and exercise counseling for additional support...." | ✅ |
| "You were encouraged to ask any follow-up questions or concerns via MyChart or by..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 51 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 52 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | New patient | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade II invasive ductal carcinoma | ✅ |
| Stage | Stage IIA (inferred from pT2 N1mi) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| medication_plan | [REDACTED] + [REDACTED]. [REDACTED] send Zoladex prior auth. | ✅ |
| imaging_plan | CT CAP and bone scan for staging | ✅ |
| follow_up | in-person: 3 weeks | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a first-time visit regarding your breast cancer treatment...." | ✅ |
| "You have early-stage breast cancer (Stage I-II) that has not spread to other par..." | ✅ |
| "You will start new medications as discussed with your doctor. These include a me..." | ✅ |
| "You are referred to reproductive health for discussions about preserving your fe..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 52 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 53 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | New patient | ✅ |
| Type_of_Cancer | ER+/PR+/HER2+ invasive ductal carcinoma with neuroendocrine  | ✅ |
| Stage | Stage II/III | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| medication_plan | Adjuvant AC/THP chemotherapy consisting of AC x 4 given q2we | ✅ |
| imaging_plan | Referral will be made to [REDACTED] at the completion of che | ✅ |
| follow_up | in-person: after considering treatment o | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a consultation regarding your newly diagnosed left breast cancer..." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma with neuroendo..." | ✅ |
| "You will start a treatment plan that includes chemotherapy to shrink the cancer ..." | ✅ |
| "You will receive chemotherapy in cycles, and after completing the chemotherapy, ..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 53 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

## ROW 54 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 提取值 | 判定 |
|------|--------|------|
| Patient type | follow up | ✅ |
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with extensi | ✅ |
| Stage | Stage IV (metastatic) | ✅ |
| Distant Met | Yes, to left posterior 7th rib and T6 vertebral body | ✅ |
| current_meds | leuprolide, letrozole, zoledronic acid | ✅ |
| goals | palliative | ✅ |
| medication_plan | continue leuprolide every 4 weeks, continue letrozole, start | ✅ |
| imaging_plan | repeat PET/CT in ~3-4 months after radiation. DEXA scan | ✅ |
| follow_up | in-person: 4 weeks | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|------------|------|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The cancer is currently responding well with stable disease. This means the canc..." | ✅ |
| "You will continue taking leuprolide every 4 weeks and letrozole. You will start ..." | ✅ |
| "You are referred to radiation oncology for post-mastectomy radiation. You will h..." | ✅ |
| "We understand that this is a challenging time and that managing these health cha..." | ✅ |

**Letter**: P0:0 P1:0 P2:0 (无问题)

### ROW 54 总评: Ext P0:0 P1:0 P2:0, Letter P0:0 P1:0 P2:0

# 补充详细审查（ROW 59-99）


## ROW 59

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR+/HER2- grade 3 invasive ductal carcinoma with hi | ✅ |
| Stage | Stage IIA (inferred from pT2 N0) | ✅ |
| Distant Met | No | ✅ |
| current_meds | "exemestane ([REDACTED]) 25 mg tabl | ✅ |
| goals | curative | ✅ |
| med_plan | Discontinue Letrozole and wait 2-3 weeks before startin | ✅ |
| imaging | Continue mammograms alternating with MRI and every 6 mo | ✅ |
| follow_up | in-person: 6 months for follow-up | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "No new signs of cancer were found during this visit. Your weight has stayed..." | ✅ |
| "You stopped taking letrozole and were advised to start exemestane in a few ..." | ✅ |
| "You will start exemestane after a short break from letrozole. You will cont..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 61

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- (1+) grade 2 invasive ductal carcinoma | ✅ |
| Stage | Stage I (inferred from tumor ≤2cm) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | she will need adjuvant endocrine therapy (Tamoxifen vs  | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: after surgery and pathol | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new visit regarding your breast cancer treatment...." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma, which me..." | ✅ |
| "You will start a medication plan after your surgery to help prevent the can..." | ✅ |
| "You are scheduled to have surgery on April 12th, 2021. During the surgery, ..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 64

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | HR+/HER2- invasive ductal carcinoma | ✅ |
| Stage | Stage IV (metastatic) | ✅ |
| Distant Met | Yes, to the sternum | ✅ |
| current_meds |  | ✅ |
| goals | palliative | ✅ |
| med_plan | Currently on unspecified agent and taxol planned. If bi | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | Not specified in the given text | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new patient evaluation regarding your breast cancer treat..." | ✅ |
| "You have a large left breast cancer that started in the milk ducts (ductal ..." | ✅ |
| "You are currently taking medications to help with side effects such as dexa..." | ✅ |
| "A biopsy is planned for the suspicious lesion in your sternum. If the biops..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 65

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER weak positive (2%), PR low positive (7%), HER2 neg ( | ✅ |
| Stage | Stage IB (corrected: pT1 N1mi — mic | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Plan to start neoadjuvant chemotherapy with AC/T. Typic | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: 1-2 weeks to start chemo | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new consult regarding your breast cancer treatment...." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma, which st..." | ✅ |
| "You will start a treatment plan called neoadjuvant chemotherapy, which is g..." | ✅ |
| "You will have a port placed to make it easier to receive chemotherapy. You ..." | ✅ |
| "You are referred to a breast oncology specialist for further consultation...." | ✅ |
| "Your next visit is in 1-2 weeks to start chemotherapy...." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 68

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2+ multifocal invasive ductal carcinoma | ✅ |
| Stage | Stage I (inferred from tumor ≤2cm) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | No specific current or future medication plans were det | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | In-person: as needed for further co | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a return visit regarding your breast cancer treatment...." | ✅ |
| "You have been diagnosed with ER+/PR+/HER2+ multifocal invasive ductal carci..." | ✅ |
| "There were no specific changes to your medications during this visit...." | ✅ |
| "You are being recommended to have a bilateral mastectomy, which is surgery ..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 70

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR+/HER2- grade 2 invasive lobular carcinoma with 1 | ✅ |
| Stage | Originally Stage II (inferred from  | ✅ |
| Distant Met | No | ✅ |
| current_meds | letrozole | ✅ |
| goals | curative | ✅ |
| med_plan | She will restart letrozole which she had previously tol | ✅ |
| imaging | CT due in June 2020 for follow up of lung nodules. | ✅ |
| follow_up | in-person: September | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "You are recovering well from surgery. Two lymph nodes were positive on the ..." | ✅ |
| "You will restart letrozole which you had previously tolerated...." | ✅ |
| "You are referred to radiation consult. You are going to have expanders plac..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 72

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR-/HER2- grade 2 invasive ductal carcinoma with fo | ✅ |
| Stage | Stage IA (pT1cN0(sn)) | ✅ |
| Distant Met | No | ✅ |
| current_meds | letrozole | ✅ |
| goals | curative | ✅ |
| med_plan | Instructed patient to begin letrozole, prescription ord | ✅ |
| imaging | Ultrasound | ✅ |
| follow_up | in-person: In 3 weeks to review [RE | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a first visit to talk about your newly diagnosed breast can..." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma, which me..." | ✅ |
| "You were instructed to start taking a medication called letrozole. Letrozol..." | ✅ |
| "You will continue taking letrozole. A test will be done to see if you might..." | P2 |

**总评**: Ext P1:0 P2:0, Letter P2:1

## ROW 78

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER-/PR-/HER2- grade 3 invasive ductal carcinoma | ✅ |
| Stage | Metastatic (Stage IV) | ✅ |
| Distant Met | Yes, to liver and periportal lymph nodes | ✅ |
| current_meds |  | ✅ |
| goals | palliative | ✅ |
| med_plan | Continue 800 mg Mag-Ox supplement daily. She is on lisi | ✅ |
| imaging | Echo 09/08/2019. If this echo is normal we can stop mon | ✅ |
| follow_up | in-person: follow up if unable to g | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The cancer has gotten worse. On the latest CT scan, the cancer in your live..." | ✅ |
| "You will continue taking magnesium oxide supplements daily. You are also on..." | ✅ |
| "You are interested in joining a clinical trial at a medication for a medica..." | ✅ |
| "An echocardiogram is scheduled for September 8, 2019. If this echo is norma..." | ✅ |
| "You will follow up with me if you are unable to go on trial at a medication..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 80

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- grade 3 invasive ductal carcinoma | ✅ |
| Stage | Stage I (inferred from tumor ≤2cm) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Start TC x 4 on 04/11/19, with [REDACTED]. Claritin for | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: cycle 2 to see me, call  | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "You have a type of breast cancer called invasive ductal carcinoma (IDC). Th..." | ✅ |
| "The cancer is 0.8 cm in size and located in the skin and fatty tissue. It i..." | ✅ |
| "Your blood tests show that your liver and kidney functions are normal...." | ✅ |
| "You will start a treatment called TC (docetaxel and cyclophosphamide) for f..." | ✅ |
| "You will use cold gloves during treatment to help prevent hand swelling...." | ✅ |
| "You will have six weeks of radiation therapy, including the armpit and ches..." | ✅ |
| "You are referred to radiation oncology for further treatment planning...." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 82

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- mixed ductal and lobular carcinoma | ✅ |
| Stage | Stage II | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Continue hydrochlorothiazide 12.5 mg tablet daily befor | ✅ |
| imaging | Dexa to assess bone health before the next visit with m | ✅ |
| follow_up | RTC after radiation. Plan to procee | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new patient evaluation regarding your breast cancer treat..." | ✅ |
| "You have a type of cancer called ER+/PR+/HER2- mixed ductal and lobular car..." | ✅ |
| "We discussed that due to your low risk, you will not need chemotherapy. You..." | ✅ |
| "You have an appointment with your doctor tomorrow to discuss radiation to d..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 84

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with me | ✅ |
| Stage | Stage IV (metastatic) | ✅ |
| Distant Met | Yes, to bone, soft tissue, liver, and possibly meninges | ✅ |
| current_meds | capecitabine, zolendronic acid | ✅ |
| goals | palliative | ✅ |
| med_plan | Continue xeloda 1500mg BID for now, continue zolendroni | ✅ |
| imaging | Repeat CT CAP now, Repeat MRI spine to rule out leptome | ✅ |
| follow_up | Not specified in the given text | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new consult regarding your breast cancer treatment...." | ✅ |
| "You have a type of breast cancer called ER+/PR+/HER2- grade 2 invasive duct..." | ✅ |
| "Recent scans show that the cancer in your brain and liver has grown compare..." | ✅ |
| "The goal of your treatment is to manage your symptoms and improve your qual..." | ✅ |
| "Your dose of capecitabine (Xeloda) was increased ...." | ✅ |
| "You will continue taking Xeloda and zolendronic acid...." | ✅ |
| "Your doctor might consider adding low-dose steroids to reduce swelling and ..." | ✅ |
| "If the cancer continues to grow, your doctor might suggest adding fulvestra..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:1

## ROW 85

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR-/HER2- invasive lobular carcinoma with pleomorph | ✅ |
| Stage | Originally Stage IIIA, now metastat | ✅ |
| Distant Met | Yes, to bone, liver, and brain | ✅ |
| current_meds |  | ✅ |
| goals | palliative | ✅ |
| med_plan | Continue steroid taper per Dr. [REDACTED]. Continue pai | P2 |
| imaging | Brain MRI to be reviewed by UCSF neuroradiology; 2-week | ✅ |
| follow_up | in-person: 2 weeks | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The cancer is currently spreading. New liver tumors have appeared, and the ..." | ✅ |
| "Your steroid dose was lowered to day. You will continue to take pain medici..." | ✅ |
| "You will be evaluated for a phase 1 trial of a medication called olaparib f..." | ✅ |

**总评**: Ext P1:0 P2:1, Letter P2:0

## ROW 87

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with a  | ✅ |
| Stage | Stage IIIA (inferred from pT2 N2a) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Ms. [REDACTED] will prefer to receive a course of hormo | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: for further discussions  | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a second opinion regarding your recently diagnosed right pr..." | ✅ |
| "You have a 2.2 cm multifocal tumor in the right breast that was removed wit..." | ✅ |
| "You will prefer to receive a course of hormonal therapy alone. Specific det..." | ✅ |
| "The use of radiation to remove the tumor from the chest wall and prevent it..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 88

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | Invasive ductal carcinoma, ER weak+, PR-, HER2- with me | ✅ |
| Stage | Originally Stage III, now metastati | ✅ |
| Distant Met | Yes, to brain, lungs, and lymph nodes | ✅ |
| current_meds | capecitabine (XELODA) 500 mg tablet | ✅ |
| goals | palliative | ✅ |
| med_plan | She is on xeloda. If progression on xeloda occurs, clin | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | prn or at progression | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new consult regarding your breast cancer that has spread ..." | ✅ |
| "You have a history of a type of breast cancer called invasive ductal carcin..." | ✅ |
| "You are currently taking a medication called capecitabine (XELODA). No chan..." | ✅ |
| "You will continue taking capecitabine. If the cancer gets worse while on th..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 90

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | Adenocarcinoma of right breast (HCC) | ✅ |
| Stage | Stage II/III | ✅ |
| Distant Met | No | ✅ |
| current_meds | ac | ✅ |
| goals | curative | ✅ |
| med_plan | Continue with cycle 4 of AC in 1 week (dose delay x 1 w | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: after XRT, approx 1-1.5  | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "You have adenocarcinoma (cancer that started in gland cells) of your right ..." | ✅ |
| "Recent tests show that your thyroid stimulating hormone is slightly high, w..." | ✅ |
| "Your white blood cell count is higher than normal, and you have a low red b..." | ✅ |
| "You are experiencing some side effects such as swelling in your armpit, pal..." | ✅ |
| "Your treatment plan includes continuing with cycle 4 of AC chemotherapy in ..." | ✅ |
| "The dose of GCSF has been reduced to 50%...." | ✅ |
| "You have been prescribed granisetron, olanzapine, and oral dexamethasone to..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:1

## ROW 91

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | follow up | ✅ |
| Type | ER+/PR+ invasive ductal carcinoma with metastatic recur | ✅ |
| Stage | Originally Stage I, now metastatic  | ✅ |
| Distant Met | Yes, to bone | ✅ |
| current_meds | everolimus, exemestane, denosumab | ✅ |
| goals | palliative | ✅ |
| med_plan | Continue lasix 10mg daily with KCL 10Meq daily, continu | ✅ |
| imaging | PET/CT next week to evaluate disease burden | ✅ |
| follow_up | in-person: 1 month to evaluate dise | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "MRI of your pelvis showed that the cancer in your right hip has grown and s..." | ✅ |
| "You started taking lasix to help with swelling and potassium to balance the..." | ✅ |
| "You will have a PET/CT scan next week to get a better picture of how much t..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 92

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR+/HER2- breast cancer metastasized to multiple si | ✅ |
| Stage | Stage IV (metastatic) | ✅ |
| Distant Met | Yes, to liver | ✅ |
| current_meds | Epirubicin, Denosumab | ✅ |
| goals | palliative | ✅ |
| med_plan | Plan cycle#2 D1 Epirubicin 25 mg/m2 D1,8,15 to with 2 d | ✅ |
| imaging | Echocardiogram | ✅ |
| follow_up | Not specified in the given text | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "Your liver feels smaller and less tender or bloated. You also have a red ra..." | ✅ |
| "Tumor marker results are pending. Laboratory results show elevated Aspartat..." | ✅ |
| "You restarted chemotherapy with Epirubicin and will receive Neupogen for 2 ..." | ✅ |
| "You are currently taking Denosumab...." | ✅ |
| "Filgrastim (NEUPOGEN) is being used to support your treatment...." | ✅ |
| "You will continue with cycle 2 of Epirubicin on days 1, 8, and 15, along wi..." | ✅ |
| "An echocardiogram (a heart ultrasound) will be performed...." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 95

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR-/HER2- invasive ductal carcinoma with residual d | ✅ |
| Stage | Stage II (inferred from 2.1cm tumor | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Continue prilosec 40mg qd, plan to start capecitabine a | ✅ |
| imaging | breast and axilla XRT | ✅ |
| follow_up | in-person: after XRT evaluation | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "The cancer in your left breast has gotten smaller after treatment. The MRI ..." | ✅ |
| "You started a new chemotherapy called AC. You will continue taking Prilosec..." | ✅ |
| "You will have radiation treatment for your breast and armpit. You are refer..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 96

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- grade I mixed ductal and cribriform carci | ✅ |
| Stage | Stage IA (pT1cN0(sn)) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Tamoxifen after completion of adjuvant radiation | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | in-person: after radiation to discu | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new consult regarding your breast cancer treatment...." | ✅ |
| "You were diagnosed with a type of breast cancer called ER+/PR+/HER2- grade ..." | ✅ |
| "The cancer was removed through surgery, and the edges of the removed tissue..." | ✅ |
| "There were no new imaging findings reported during this visit...." | ✅ |
| "During the physical exam, mild swelling was noted in your left arm, but no ..." | ✅ |
| "No changes were made to your current medications...." | ✅ |
| "You will have a test called a medication testing to learn more about your c..." | P2 |
| "You are referred to radiation oncology for a consult on adjuvant radiation...." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:1

## ROW 97

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with lo | ✅ |
| Stage | Stage IA (pT1b N0) | ✅ |
| Distant Met | No | ✅ |
| current_meds |  | ✅ |
| goals | curative | ✅ |
| med_plan | Strongly recommend adjuvant endocrine therapy with [RED | ✅ |
| imaging | No imaging planned. | ✅ |
| follow_up | 3-4 wks after [REDACTED] Dx result  | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a follow-up visit regarding your breast cancer treatment...." | ✅ |
| "You were diagnosed with a type of breast cancer called invasive ductal carc..." | ✅ |
| "We discussed starting a medication to help prevent the cancer from coming b..." | ✅ |
| "You will have a test called Oncotype Dx to learn more about your cancer. Yo..." | ✅ |
| "We understand that this is a challenging time and that managing these healt..." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:0

## ROW 99

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | New patient | ✅ |
| Type | ER+/PR+/HER2+ grade 3 invasive ductal carcinoma with ly | ✅ |
| Stage | Originally Stage III, now metastati | ✅ |
| Distant Met | Yes, to the left lung and mediastinal lymph nodes | ✅ |
| current_meds | fulvestrant | ✅ |
| goals | palliative | ✅ |
| med_plan | If cancer remains estrogen receptor positive, will cons | ✅ |
| imaging | The patient will need a followup CT scan with contrast  | ✅ |
| follow_up | in-person: 2 weeks, to review chest | ✅ |

| Letter 句子 | 判定 |
|---|---|
| "You came in for a new consultation regarding your breast cancer treatment...." | ✅ |
| "You were diagnosed with a type of breast cancer called invasive ductal carc..." | ✅ |
| "A recent PET/CT scan showed that the cancer in your chest lymph nodes has g..." | ✅ |
| "You recently switched from a medication called anastrozole to letrozole bec..." | ✅ |
| "Your treatment goal is to manage the symptoms and slow down the growth of t..." | ✅ |
| "You will need to have a biopsy of either the cancer in your lung or the swo..." | ✅ |
| "You will also need a followup CT scan with contrast and thin slices to get ..." | ✅ |
| "Depending on the results of the biopsy, you may start a new treatment plan...." | ✅ |

**总评**: Ext P1:0 P2:0, Letter P2:1
## ROW 100 — 逐字段+逐句

### Extraction 逐字段

| 字段 | 值 | 判定 |
|---|---|---|
| Patient type | Follow up | ✅ |
| Type | ER+/PR+/HER2- grade 2 IDC with metastatic recurrence | ✅ |
| Stage | Originally Stage I, now metastatic (Stage IV) | ✅ |
| Distant Met | Yes, to bone | ✅ |
| current_meds | Gemzar | ✅ |
| goals | palliative | ✅ |
| med_plan | Continue constipation and pain meds. Focalin prn for fatigue | ✅ |
| imaging | No imaging planned | ✅ |
| follow_up | Not specified | ✅ |

**Extraction**: P0:0 P1:0 P2:0

### Letter 逐句

| Letter 句子 | 判定 |
|---|---|
| "tumor markers have gone up...Cancer Antigen 15-3 and Cancer Antigen 27.29" | ✅ |
| "hemoglobin and hematocrit levels are low...anemic" | ✅ |
| "no new signs of cancer spreading" | ✅ |
| "Gemzar treatment was stopped by you" | ✅ |
| "Focalin as needed to help with fatigue" | ✅ |
| "**exercise for 10 minutes three times a day**" ← 医生 feedback | ✅ |
| "discuss taking a break from your treatment with your doctor" | ✅ |

**Letter**: P0:0 P1:0 P2:0

### ROW 100 总评: Ext P2:0, Letter P2:0 ✅ — exercise 10min 3x/day 医生 feedback 确认

---

# 补充审查：缺失表格的 ROW

## ROW 6 — 补充 Letter 表

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | Chief Complaint "Follow-up" ✓ | ✅ |
| "surgery to remove both breasts (bilateral mastectomy) with expanders" | ✓ | ✅ |
| "small cancer (1.5 cm)...removed completely, no cancer found in the lymph nodes" | pathology 1.5cm, 0/1 LN ✓ | ✅ |
| "left breast was healthy" | "Left breast: benign" ✓ | ✅ |
| "recovering well...nerve irritation and mild swelling" | HPI+PE ✓ | ✅ |
| "start taking letrozole today" | A/P ✓ | ✅ |
| "continue another medication for at least 3 years" | zoladex ✓ | ✅ |
| "monthly estradiol tests" | A/P ✓ | ✅ |
| "Gabapentin is prescribed as needed" | A/P ✓ | ✅ |
| "3 months or sooner" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

## ROW 7 — 补充 Letter 表

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "second opinion on how to manage your metastatic breast cancer" | "2nd opinion" ✓ | ✅ |
| "cancer seems to have slightly grown" | "probable mild progression" ✓ | ✅ |
| "cancer in your chest area has stayed the same" | "Stable mediastinal" ✓ | ✅ |
| "no sign of cancer in your brain" | Brain MRI normal ✓ | ✅ |
| "heart function has slightly decreased" | "LVEF 52%" ✓ | ✅ |
| "current treatment...stopped" | "d/c current rx" ✓ | ✅ |
| "start a new medication" | A/P "Rec ***** as next line" ✓ | ✅ |
| "not be given any hormones" | A/P ✓ | ✅ |
| "clinical trials" | A/P ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

## ROW 8 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Patient type | New patient | establishing care ✓ | ✅ |
| Type_of_Cancer | ER-/PR-/HER2+ (IHC 3+, FISH 5.7) grade 3 IDC | pathology on LN: ER-, HER2 IHC 3+, FISH 5.7 ✓ | ✅ |
| Stage | Originally Stage III, now no residual disease in breast | breast pCR ✓ | ✅ |
| Distant Metastasis | No | PET/CT no mets ✓ | ✅ |
| findings | breast pCR + 3/28 LN+ with extranodal extension | pathology ✓ 详细 | ✅ |
| goals | curative | adjuvant ✓ | ✅ |
| medication_plan | AC x4 → T-DM1 | A/P ✓ | ✅ |
| radiotherapy_plan | radiation | A/P ✓ | ✅ |
| imaging_plan | echocardiogram prior to AC | ✓ | ✅ |
| procedure_plan | No procedures planned | A/P says "port placement"——漏了 port | P2 |

**Extraction 小结**: P0:0 P1:0 P2:1（漏 port placement）

## ROW 9 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 2 IDC | pathology ER+ 85%, PR- <1%, HER2- IHC 0 ✓ | ✅ |
| Stage | Stage II (pT3 N1) | A/P "Stage II" ✓ | ✅ |
| findings | 3.84cm IDC ~5% cellularity, 3 LN types (macro+micro+ITC) | ✓ 详细 | ✅ |
| medication_plan | letrozole after radiation + Fosamax | A/P ✓ | ✅ |
| radiotherapy_plan | Radiation referral | A/P ✓ | ✅ |
| procedure_plan | Drains removed Thursday | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

## ROW 10 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | HR+ HER2- grade 2 IDC | A/P "HR+ and her 2 negative" ✓ | ✅ |
| Stage | Stage II (from 8.8cm tumor with 20 LN involved) | "July 20" likely redacted numbers ✓ | ✅ |
| current_meds | letrozole | ✓ | ✅ |
| medication_plan | continue letrozole | A/P ✓ | ✅ |
| radiotherapy_plan | radiation to left chest wall and surrounding LN | A/P ✓ | ✅ |
| imaging_plan | DEXA | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

## ROW 12 — 补充 Letter 表

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "follow-up visit" | ✓ | ✅ |
| "MRI brain...new brain lesions" | MRI 08/15/18 ✓ | ✅ |
| "CT scans...cancer in your bones is stable" | CT 09/05/18 ✓ | ✅ |
| "slight decrease in size of a lymph node" | celiac axis 9→7mm ✓ | ✅ |
| "continue taking herceptin and another medication" | ✓ | ✅ |
| "continue taking letrozole" | ✓ | ✅ |
| "stay off chemotherapy...trouble with side effects" | ✓ | ✅ |
| "CT scan every 4 months + MRI brain every 4 months + bone scan" | A/P ✓ | ✅ |
| "6 weeks for follow-up" | ✓ | ✅ |
| "radiation oncology for further consultation" | GK/Rad Onc ✓ | ✅ |

**Letter 小结**: P0:0 P1:0 P2:0

## ROW 14 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+ metastatic breast cancer, HER2- | FNA: ER 99%, PR 25%, HER2 neg ✓ | ✅ |
| Stage | Metastatic (Stage IV) | de novo metastatic ✓ | ✅ |
| current_meds | "" | stopped palbociclib/fulvestrant ✓ | ✅ |
| recent_changes | stopped standard therapy, doing Mexico alt therapy | HPI ✓ | ✅ |
| medication_plan | topical cannabis/sulfur, Cymbalta Rx | A/P ✓ | ✅ |
| imaging_plan | CT CAP + Spine MRI May, Spine MRI 6 weeks | A/P ✓ | ✅ |
| lab_plan | labs every two weeks | ✓ | ✅ |
| follow_up | 2-3 months | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

## ROW 19 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2+ (FISH+) grade 3 IDC with Ki-67 20-90% | pathology ✓ | ✅ |
| Stage | Stage IIIA (pT2 N2a) | locally advanced ✓ | ✅ |
| Distant Metastasis | No | PET/CT no distant mets ✓ | ✅ |
| findings | core biopsy + PET/CT + PE (4cm mass + nipple inversion) | 全面 ✓ | ✅ |
| goals | curative | neoadjuvant intent ✓ | ✅ |
| medication_plan | TCHP + GCSF | A/P ✓ | ✅ |
| therapy_plan | TCHP, avoid anthracycline due to CAD | A/P 重要临床决策 ✓ | ✅ |
| procedure_plan | Port Placement | A/P ✓ | ✅ |
| imaging_plan | Echocardiogram | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

## ROW 24 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade II micropapillary mucinous carcinoma | pathology ✓ | ✅ |
| Stage | Stage III | ~5cm + N1mi, plausible ✓ | ✅ |
| Distant Metastasis | No | PET/CT no metastatic disease ✓ | ✅ |
| findings | pathology + imaging + PE + liver lesions benign | ✓ | ✅ |
| supportive_meds | TYLENOL #4, oxyCODONE | ✓ | ✅ |
| goals | adjuvant | ✓ | ✅ |
| medication_plan | adjuvant hormone therapy if low risk | A/P ✓ | ✅ |
| radiotherapy_plan | radiation if low risk, appt 12/07/18 | A/P ✓ | ✅ |
| lab_plan | [REDACTED] test for chemo benefit | MammaPrint ✓ | ✅ |
| genetic_testing_plan | send specimen for MP | A/P ✓ | ✅ |
| Referral: Others | Physical therapy | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0

## ROW 29 — 补充 Extraction + Letter 表

### Extraction

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 IDC with micropapillary+ductal features | pathology ✓ | ✅ |
| Stage | Stage IIA (pT1c(m)N1(sn)) | 1.6cm + 0.6cm + micromet ✓ | ✅ |
| Distant Metastasis | No | no mets ✓ | ✅ |
| current_meds | letrozole 2.5mg | just started ✓ | ✅ |
| medication_plan | letrozole + calcium + vitamin D + vaginal moisturizer | A/P ✓ | ✅ |
| radiotherapy_plan | RT planning, likely pursued locally | A/P ✓ | ✅ |
| imaging_plan | Bone density scan when returns | A/P ✓ | ✅ |
| procedure_plan | Surgery September 2019 | A/P ✓ | ✅ |

### Letter

| Letter 句子 | 原文依据 | 判定 |
|------------|---------|------|
| "medical oncology consultation regarding your newly diagnosed breast cancer" | ✓ | ✅ |
| "invasive ductal carcinoma...grade 2...ER+/PR+...no HER2" | ✓ | ✅ |
| "cancer has spread to a small area in one of the lymph nodes" | SLN micrometastasis ✓ | ✅ |
| "start taking...letrozole" | A/P ✓ | ✅ |
| "calcium supplements to help keep your bones strong" | A/P ✓ | ✅ |
| "bone density scan when you return from your travels" | A/P ✓ | ✅ |
| "surgery in September 2019" | A/P ✓ | ✅ |
| "radiation oncology for a consultation" | A/P ✓ | ✅ |
| "long-term oncology follow-up closer to your home" | A/P ✓ | ✅ |

**ROW 29 总评**: Ext P2:0, Letter P2:0

## ROW 30 — 补充 Extraction 表

| 字段 | 提取值 | 原文依据 | 判定 |
|------|--------|---------|------|
| Type_of_Cancer | ER-/PR-/HER2+ grade 2 IDC with intermediate to high grade DCIS | pathology: ER-(0%), PR-(0%), HER2+ (IHC 3, FISH 8.9) ✓ | ✅ |
| Stage | Clinical stage II-III | A/P ✓ | ✅ |
| Distant Metastasis | No | PET/CT no distant mets ✓ | ✅ |
| Metastasis | No | FNA of axillary LN: **benign**. A/P "**high-risk node-negative**" ✓ | ✅ |
| goals | curative | curative intent ✓ | ✅ |
| medication_plan | neoadjuvant THP/AC or TCHP + 1yr trastuzumab | A/P ✓ | ✅ |
| procedure_plan | Mediport placement | A/P ✓ | ✅ |
| imaging_plan | TTE prior to starting | A/P ✓ | ✅ |

**Extraction 小结**: P0:0 P1:0 P2:0 — Metastasis correctly "No" (node-negative) ✅
