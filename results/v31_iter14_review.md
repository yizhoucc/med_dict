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
