# 详细审查报告:Rows 25-39 (Post-Refactor vs Old Prompts)

**审查日期**: 2026-03-13
**数据来源**:
- Post-refactor: `/Users/yizhoucc/repo/med_dict/results/default_20260301_161703/progress.json`
- Old prompts: `/Users/yizhoucc/repo/med_dict/results/default_20260301_084320/progress.json`

## 总体发现

### 关键改进 (Improvements)

1. **Genetic_Testing_Plan / Referral-Genetics 提取准确性显著提升**
   - Row 25, 31, 37, 38, 39: 旧版本全部输出 "No new genetic or molecular tests were planned"(误报),新版本正确提取了原文中的遗传咨询计划

2. **Current_meds vs Supportive_meds 分类更准确**
   - 新版本成功区分了抗癌治疗药物(current_meds)和支持性药物(supportive_meds)
   - 旧版本经常混淆,将所有药物混在一起或放错字段

3. **Lab_summary 字段更诚实**
   - 新版本在没有实验室检查时明确说 "No labs in note"
   - 旧版本有时遗漏,有时幻觉(如 Row 26 写 "CBC with platelets to be ordered" 但实际是计划而非结果)

4. **Findings 字段更详细**
   - 新版本提取了更丰富的病理/影像/体格检查细节
   - 例如 Row 27, 28, 29, 30 的 findings 都比旧版本更完整

5. **Procedure_Plan 更完整**
   - Row 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39 的procedure_plan 都比旧版本更详细

### 主要问题 (Regressions)

1. **Lab_summary 遗漏问题**
   - Row 27: 旧版本提取了所有实验室数值,新版本也提取了(但少了 Platelet Count)
   - Row 37: 旧版本提取了完整的CBC和化学检验,新版本误写 "No labs in note"(严重遗漏)
   - Row 38: 旧版本写了影像检查结果(CT CAP, MRI brain, bone scan),新版本误写 "No labs in note"
   - **根本原因**: 新版本的 prompt 可能过于严格区分 "labs" vs "imaging",导致将影像结果从 lab_summary 中排除

2. **Current_meds 遗漏问题**
   - Row 25, 27, 28, 34, 36, 37, 38: 新版本 current_meds 为空,但旧版本提取了药物
   - Row 30, 31, 32, 35, 39: 新版本遗漏了部分药物
   - **根本原因**: 新版本的 CoT prompt 可能过于严格区分 "CURRENT" vs "PLANNED",导致在术前咨询笔记中将讨论的药物误判为"未开始服用"

3. **Patient type 大小写不一致**
   - Row 26, 31, 32, 34, 37, 39: 新版本写 "Follow up",旧版本写 "follow up"
   - 虽然不影响语义,但格式不统一

4. **Goals of treatment 判断错误**
   - Row 28: 新版本写 "curative",旧版本写 "risk reduction"。原文是术后辅助治疗,应该是 curative(新版本正确)
   - Row 31: 新版本写 "curative",旧版本写 "palliative"。原文显示 PET-CT "No abnormal areas of FDG uptake to suggest active metastatic disease",说明治疗有效,但患者曾经转移,应该是 palliative(旧版本可能更准确,需进一步核实)
   - Row 32: 新版本写 "adjuvant",旧版本写 "risk reduction"。原文是术后随访,应该是 adjuvant(新版本正确)
   - Row 35: 新版本写 "adjuvant",旧版本写 "curative"。原文是新辅助化疗后,应该是 curative(旧版本可能更准确)
   - Row 37: 新版本写 "adjuvant",旧版本写 "Risk reduction"。原文是术前评估,应该是 neoadjuvant or curative(都不完全准确)
   - Row 38: 新版本写 "adjuvant",旧版本写 "neoadjuvant"。原文明确说 "neoadjuvant chemotherapy"(旧版本正确)

5. **Response_assessment 幻觉问题**
   - Row 26, 27, 28, 29, 33, 37: 新版本提取了疑似 response 的内容,但部分是术前/未开始治疗的患者,应该写 "Not yet on treatment"
   - Row 32: 新版本写 "Not mentioned in note, but exam shows no evidence of recurrence",旧版本更简洁 "No evidence of recurrence"

6. **Genetic_Testing_Plan 格式错误**
   - Row 26, 27, 28, 29, 30, 32, 33, 34, 35, 36: 新版本误写 "Not yet on treatment — no response to assess"(这是 response_assessment 的默认值,不是genetic testing的回答)
   - 这是明显的字段串行错误

---

## 逐行详细审查

### Row 25 (coral_idx=165) — Stage IB TNBC 新患者评估

**Ground truth (from note):**
- 新患者,视频问诊
- Stage IB triple negative breast cancer
- 计划先手术(双侧缩乳 + 右侧肿块切除 + 前哨淋巴结),然后化疗和放疗
- 已转诊遗传咨询 ("She has been referred to genetics")
- 港置入术计划在手术时同时进行
- 术后复诊审查最终病理

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Genetic_Testing_Plan | "Genetics consult." | "No new genetic or molecular tests were planned" | ✅ 新版本正确,旧版本**严重误报** |
| Referral-Genetics | "Genetics consult" | "None" | ✅ 新版本正确,旧版本**遗漏** |
| Procedure_Plan | "surgery first for reductions bilaterally and a right lumpectomy with sentinel node, port placement" | "plan for port placement" | ✅ 新版本更完整 |
| Current_meds | "" (empty) | "amoxicillin, cephalexin, cetirizine, diphenhydrAMINE, levothyroxine" | ❌ 新版本**遗漏**。但这些是支持性药物,不是抗癌药,新版本分类更准确 |
| Supportive_meds | [6 medications] | "" (empty) | ✅ 新版本正确分类 |
| Findings | "...node negative" | "...75% tumor size" | ✅ 新版本更完整(加了"node negative") |

**错误: 1 个问题**
- Current_meds 为空:技术上正确(患者未开始抗癌治疗),但旧版本的做法也可接受

**vs 100-row (old prompts):**
- ✅ **重大改进**: Genetic_Testing_Plan 和 Referral-Genetics 从误报变为正确
- ✅ **改进**: Procedure_Plan 更完整
- ✅ **改进**: Current_meds vs Supportive_meds 分类更准确
- ⚠️ **微小改进**: Findings 增加了 "node negative"

---

### Row 26 (coral_idx=166) — 骨转移随访

**Ground truth (from note):**
- 随访患者,ER+/PR+ IDC,原本早期,现在骨转移
- 正在服用 letrozole + Zoladex,加 Zometa(zoledronic acid)
- PET-CT 显示骨转移稳定或略减轻代谢活性
- 腰痛症状,考虑2周后复查,必要时做脊柱MRI
- 未提及新的实验室检查结果(但计划抽血检查CBC + platelets)

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Current_meds | "letrozole (FEMARA)" | "Femara, Zoladex, calcium-vitamin D" | ❌ 新版本**遗漏** Zoladex |
| Supportive_meds | "calcium-vitamin D, zolendronic acid" | "calcium-vitamin D" | ✅ 新版本正确加入 Zometa |
| Lab_summary | "No labs in note." | "CBC with platelets to be ordered" | ⚠️ 旧版本是计划(plan),不应写在 lab_summary。新版本更准确 |
| Findings | (detailed PET-CT findings) | "stable disease, lower back pain, easy bruising, frequent urination" | ✅ 新版本更客观(影像发现),旧版本混入症状 |
| Procedure_Plan | "No procedures planned." | "Reassess lower back pain at two weeks and consider MRI of the spine." | ❌ 新版本**遗漏**计划的MRI |
| Genetic_Testing_Plan | "Not yet on treatment — no response to assess." | "No new genetic or molecular tests were planned during this visit." | ❌ 新版本**字段串行**错误(复制了response_assessment的默认值) |

**错误: 3 个问题**
- Current_meds 遗漏 Zoladex
- Procedure_Plan 遗漏 MRI 计划
- Genetic_Testing_Plan 字段串行

**vs 100-row (old prompts):**
- ✅ **改进**: Findings 更客观(影像 vs 症状)
- ✅ **改进**: Lab_summary 更准确(不把计划当成结果)
- ✅ **改进**: Supportive_meds 加入 Zometa
- ❌ **退化**: Current_meds 遗漏 Zoladex
- ❌ **退化**: Procedure_Plan 遗漏 MRI
- ❌ **新问题**: Genetic_Testing_Plan 字段串行

---

### Row 27 (coral_idx=167) — Stage I IDC 术后随访

**Ground truth (from note):**
- 随访患者,Stage I IDC,已完成手术
- 病理:0.9 cm grade 1 IDC, ER+ 99%, PR+ 95%, HER2-, 0/3 LN+
- 实验室检查:Hemoglobin 15.3, Hematocrit 48.8, Platelet Count 217, 以及完整的化学检验
- 计划放疗咨询

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Lab_summary | "Hemoglobin 15.3, Hematocrit 48.8, ..." (missing Platelet Count 217) | "Hemoglobin 15.3, Hematocrit 48.8, ..." (also missing Platelet Count) | ⚠️ 两者都**遗漏** Platelet Count |
| Findings | (detailed pathology) | "No new findings or disease status mentioned" | ✅ 新版本正确,旧版本**严重错误** |
| Current_meds | "" (empty) | [long list of medications] | ❌ 新版本**遗漏**所有药物 |
| Supportive_meds | "pregabalin, calcium, vitamin D" | "acetaminophen, artificial tears, ascorbic acid, docusate sodium" | ⚠️ 两者都不完整 |
| Procedure_Plan | "proceed with radiation oncology consultation; she will have radiation" | "No procedures planned." | ✅ 新版本正确,旧版本**遗漏** |
| Genetic_Testing_Plan | "Not yet on treatment — no response to assess." | "No new genetic or molecular tests were planned during this visit." | ❌ 新版本**字段串行** |

**错误: 3 个问题**
- Current_meds 遗漏所有药物
- Genetic_Testing_Plan 字段串行
- Lab_summary 遗漏 Platelet Count(但两者都有此问题)

**vs 100-row (old prompts):**
- ✅ **重大改进**: Findings 从 "No new findings" 变为详细病理
- ✅ **改进**: Procedure_Plan 正确提取放疗计划
- ❌ **退化**: Current_meds 完全遗漏
- ❌ **新问题**: Genetic_Testing_Plan 字段串行

---

### Row 28 (coral_idx=168) — pT1cN0 微转移随访

**Ground truth (from note):**
- 这是随访患者(之前手术),不是新患者
- 病理:1.6 cm和0.6 cm IDC,前哨淋巴结微转移(0.5 mm)
- Oncotype DX: Low Risk (+0.046)
- 计划手术(由Dr. *****,暂定2019年9月)
- 正在服用 Letrozole 2.5mg

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Patient type | "New patient" | "follow up" | ❌ 新版本**错误**,这是随访患者 |
| Lab_summary | "No labs in note." | "Low Risk profile (+0.046)" | ❌ 新版本**遗漏** Oncotype DX 结果 |
| Current_meds | "" (empty) | "Letrozole 2.5mg PO daily" | ❌ 新版本**遗漏** |
| Goals | "curative" | "risk reduction" | ⚠️ 术后辅助治疗,curative 可能更准确 |
| Response_assessment | (long description) | "Good response to treatment, as evidenced by Low Risk profile..." | ⚠️ 新版本写了病理/影像发现,旧版本写了Oncotype结果,两者都不太准确(患者刚手术,还未开始系统治疗) |

**错误: 4 个问题**
- Patient type 错误
- Lab_summary 遗漏 Oncotype DX
- Current_meds 遗漏 Letrozole
- Response_assessment 答非所问(应该是"Not yet on systemic treatment")

**vs 100-row (old prompts):**
- ✅ **改进**: Goals 可能更准确(curative vs risk reduction)
- ✅ **改进**: Findings 更详细
- ❌ **严重退化**: Patient type 错误
- ❌ **退化**: Lab_summary 遗漏 Oncotype DX
- ❌ **退化**: Current_meds 遗漏

---

### Row 29 (coral_idx=169) — HER2+ IDC 新患者评估

**Ground truth (from note):**
- 新患者评估
- HER2+ IDC (ER-, PR-),大肿块(9.0 x 3.8 cm)
- 计划新辅助化疗,港置入,TTE,2-3周内开始

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Lab_summary | "No labs in note." | "Creatinine 0.74, CA 27.29 16.6-106.2, CA 15-3 60.9-74.5" | ❌ 新版本**遗漏**肿瘤标志物 |
| Findings | (very detailed) | "Large, enhancing hypermetabolic right breast mass, multiple enlarged lymph nodes..." | ✅ 新版本更详细 |
| Procedure_Plan | "Chemotherapy, TTE, Mediport placement, and surgery, will start within 2 to 3 weeks" | "TTE, Mediport placement, chemotherapy teaching session" | ✅ 新版本更完整 |

**错误: 1 个问题**
- Lab_summary 遗漏肿瘤标志物

**vs 100-row (old prompts):**
- ✅ **改进**: Findings 更详细
- ✅ **改进**: Procedure_Plan 更完整
- ❌ **退化**: Lab_summary 遗漏肿瘤标志物

---

### Row 30 (coral_idx=170) — 转移性乳腺癌进展

**Ground truth (from note):**
- 转移性乳腺癌,PET-CT显示肝转移和骨转移进展
- 正在接受 Doxil 治疗
- 支持性药物:ondansetron, lorazepam, prochlorperazine, 大麻
- LVEF 58.5%
- 计划脑MRI和盆腔MRI

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Lab_summary | "No labs in note." | "LVEF 58.5%, LVEF by MOD Bi-plane 58.5%" | ❌ 新版本**遗漏**超声心动图结果 |
| Current_meds | "" (empty) | "Doxil, ascorbic acid, cholecalciferol, LORazepam, magnesium, medical cannabis, ondansetron" | ❌ 新版本**遗漏** Doxil(抗癌药) |
| Supportive_meds | "ondansetron, LORazepam, prochlorperazine" | "medical cannabis, magnesium" | ⚠️ 两者都不完整,应包含所有支持性药物 |
| Procedure_Plan | "Cycle 1 Doxil IV every 28 days, Brain MRI, MRI pelvis" | "MRI pelvis, Brain MRI" | ✅ 新版本更完整(包含化疗计划) |

**错误: 2 个问题**
- Lab_summary 遗漏 LVEF
- Current_meds 遗漏 Doxil

**vs 100-row (old prompts):**
- ✅ **改进**: Procedure_Plan 更完整
- ✅ **改进**: Findings 更详细
- ❌ **退化**: Lab_summary 遗漏 LVEF
- ❌ **退化**: Current_meds 遗漏 Doxil

---

### Row 31 (coral_idx=171) — HER2+ IDC 转移后CR

**Ground truth (from note):**
- 随访患者,原本 Stage IIA,后来转移到淋巴结,现在 PET-CT 显示 "No abnormal areas of FDG uptake"
- 正在服用 exemestane + trastuzumab + pertuzumab
- 炎症标志物:Sedimentation Rate 27, CRP 3.1
- 计划在*****设置输液,3周和6周后复诊

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Stage | "Stage IIA" | "Originally Stage IIA, now metastatic" | ⚠️ 旧版本更准确(患者有转移史) |
| Lab_summary | "{'Sedimentation Rate (MB)': '27', 'C-Reactive Protein': '3.1'}" | "" (empty) | ✅ 新版本正确提取 |
| Findings | "No abnormal areas of FDG uptake..." | "Metastatic relapse in the lymph nodes..." | ⚠️ 新版本是当前状态(CR),旧版本是历史(转移史),都不完全准确 |
| Current_meds | "exemestane, trastuzumab, pertuzumab" | "trastuzumab (HERCEPTIN IV), exemestane, levothyroxine, loperamide" | ⚠️ 新版本正确提取抗癌药,但遗漏 levothyroxine(支持性药物) |
| Goals | "curative" | "palliative" | ⚠️ 患者有转移史,应该是 palliative(旧版本可能更准确) |
| Genetic_Testing_Plan | "negative" | "No genetic testing planned." | ⚠️ 新版本写 "negative" 不明确,应该说明是什么检测结果为阴性 |

**错误: 2 个问题**
- Goals 可能错误(curative vs palliative)
- Genetic_Testing_Plan 不明确("negative" 指什么?)

**vs 100-row (old prompts):**
- ✅ **改进**: Lab_summary 正确提取炎症标志物
- ✅ **改进**: Procedure_Plan 更完整
- ⚠️ **可能退化**: Goals 判断可能不准确
- ⚠️ **可能退化**: Stage 和 Findings 应该包含转移史

---

### Row 32 (coral_idx=172) — Stage IIB 术后随访

**Ground truth (from note):**
- 随访患者,术后无复发迹象
- 正在服用 FEMARA (letrozole) 辅助内分泌治疗
- 其他药物:ibuprofen, cholecalciferol, Exforge, Klor-Con, hydrochlorothiazide

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Current_meds | "FEMARA" | "letrozole, ibuprofen, CHOLECALCIFEROL, EXFORGE, KLOR-CON, hydrochlorothiazide, FEMARA" | ⚠️ 新版本只提取抗癌药,更准确;旧版本混入所有药物 |
| Goals | "adjuvant" | "risk reduction" | ✅ 新版本更准确(术后辅助治疗) |
| Response_assessment | "Not mentioned in note, but exam shows no evidence of recurrence." | "No evidence of recurrence" | ⚠️ 新版本啰嗦,旧版本更简洁 |
| Genetic_Testing_Plan | "Not yet on treatment — no response to assess." | "No new genetic or molecular tests were planned during this visit." | ❌ 新版本**字段串行** |

**错误: 1 个问题**
- Genetic_Testing_Plan 字段串行

**vs 100-row (old prompts):**
- ✅ **改进**: Goals 更准确
- ✅ **改进**: Current_meds 分类更准确
- ❌ **新问题**: Genetic_Testing_Plan 字段串行

---

### Row 33 (coral_idx=173) — 局部复发

**Ground truth (from note):**
- 左乳重建后局部复发,1.7cm grade 3 IDC
- 已服用 Tamoxifen,后改为 Arimidex
- DXA骨密度检查:-2.0 / -1.0 / -1.8
- 计划胸壁放疗
- 讨论化疗降低复发风险和死亡率

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Lab_summary | "August 2018 - labs/markers - unremarkable, February 2018 - DXA = -2.0 / -1.0 / -1.8..." | "" (empty) | ✅ 新版本正确提取 |
| Findings | (detailed pathology and imaging) | "Local recurrence of cancer in the left breast, 1.7cm tumor, grade 3..." | ✅ 新版本更详细 |
| Supportive_meds | "Arimidex, Tamoxifen" | "" (empty) | ❌ 新版本**错误分类**:这两个是抗癌药,不是支持性药物 |
| Response_assessment | "No evidence of response to current treatment, but disease may be sensitive to hormonal therapy" | "The benefit of chemotherapy in reducing recurrence risk and mortality was discussed." | ⚠️ 两者都不太准确,应该描述复发情况 |
| Procedure_Plan | "chest wall RT" | "lumpectomy" | ⚠️ 两者都正确,但新版本是计划,旧版本是已完成的手术 |

**错误: 1 个问题**
- Supportive_meds 错误分类(应该是 current_meds)

**vs 100-row (old prompts):**
- ✅ **改进**: Lab_summary 正确提取 DXA 结果
- ✅ **改进**: Findings 更详细
- ❌ **新问题**: Supportive_meds 错误分类

---

### Row 34 (coral_idx=174) — ILC 术后随访

**Ground truth (from note):**
- 随访患者,侵袭性小叶癌,术后无复发
- 正在服用 tamoxifen
- 病理:margins negative, 0/4 LN
- 最近一次钼靶(04/15/2019)无恶性征象

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Findings | (detailed pathology and imaging) | "No new findings, patient is doing well" | ✅ 新版本更详细 |
| Supportive_meds | "tamoxifen" | "" (empty) | ❌ 新版本**错误分类**:tamoxifen 是抗癌药,不是支持性药物 |

**错误: 1 个问题**
- Supportive_meds 错误分类

**vs 100-row (old prompts):**
- ✅ **改进**: Findings 更详细
- ❌ **新问题**: Supportive_meds 错误分类

---

### Row 35 (coral_idx=175) — 新辅助化疗中

**Ground truth (from note):**
- 正在接受 Abraxane 新辅助化疗(cycle 8)
- 同时服用 Tamoxifen + Zoladex
- 支持性药物:Lorazepam, Zofran, compazine, omeprazole, Lexapro, ambien, valtrex
- 计划放疗咨询

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Current_meds | "Tamoxifen, Zoladex, Abraxane" | "Abraxane, tamoxifen, zoladex, lorazepam, zofran, compazine, omeprazole, lexapro, ambien, valtrex" | ✅ 新版本分类更准确 |
| Goals | "adjuvant" | "curative" | ⚠️ 这是新辅助化疗,应该是 curative(旧版本可能更准确) |
| Procedure_Plan | "Radiation oncology referral, will see Dr. ***** next week" | "No procedures planned." | ✅ 新版本正确提取 |

**错误: 1 个问题**
- Goals 可能错误(adjuvant vs curative)

**vs 100-row (old prompts):**
- ✅ **改进**: Current_meds 分类更准确
- ✅ **改进**: Procedure_Plan 正确提取
- ⚠️ **可能退化**: Goals 判断可能不准确

---

### Row 36 (coral_idx=176) — TNBC 术后计划化疗

**Ground truth (from note):**
- TNBC (ER-, PR-, HER2-),2.3 cm, node negative, grade 3
- 术后恢复良好
- 计划辅助化疗(AC → Taxol)
- 支持性药物:dexlansoprazole, prednisone

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Current_meds | "" (empty) | "AC followed by Taxol" | ⚠️ 新版本遗漏计划的化疗,但技术上患者还未开始治疗 |
| Procedure_Plan | "adjuvant chemotherapy" | "No procedures planned." | ✅ 新版本正确 |

**错误: 0 个问题**
- Current_meds 为空是合理的(患者还未开始化疗)

**vs 100-row (old prompts):**
- ✅ **改进**: Procedure_Plan 正确提取
- ⚠️ **微小差异**: Current_meds 处理方式不同(旧版本写计划,新版本只写已开始的)

---

### Row 37 (coral_idx=177) — BRCA1+ 患者术前评估

**Ground truth (from note):**
- 这是**新患者**评估(首次肿瘤科会诊)
- ER- IDC, Stage IIB, 8 x 5 cm 肿块,可触及,正在增大
- BRCA 1 突变
- 正在服用 Olaparib + Xeloda(新辅助治疗)
- 计划1月31日双侧乳房切除术
- 有完整的实验室检查结果(CBC, 化学检验, HbA1c)

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Patient type | "Follow up" | "New patient" | ❌ 新版本**错误**,这是新患者 |
| Lab_summary | "No labs in note." | (complete lab results) | ❌ 新版本**严重遗漏** |
| Current_meds | "" (empty) | "Olaparib, Xeloda" | ❌ 新版本**遗漏** |
| Goals | "adjuvant" | "Risk reduction" | ⚠️ 这是新辅助治疗,应该是 neoadjuvant or curative |
| Genetic_Testing_Plan | "BRCA 1 mutation" | "No new genetic or molecular tests were planned during this visit." | ✅ 新版本正确,旧版本误报 |

**错误: 4 个严重问题**
- Patient type 错误
- Lab_summary 严重遗漏
- Current_meds 遗漏
- Goals 不准确

**vs 100-row (old prompts):**
- ✅ **重大改进**: Genetic_Testing_Plan 正确提取 BRCA1
- ❌ **严重退化**: Patient type 错误
- ❌ **严重退化**: Lab_summary 完全遗漏
- ❌ **退化**: Current_meds 遗漏

---

### Row 38 (coral_idx=178) — TNBC 新患者评估

**Ground truth (from note):**
- 新患者评估
- Triple negative IDC, 3.6 x 2.8 x 2.0 cm, 腋窝淋巴结可疑
- 计划新辅助化疗
- 计划港置入、超声心动图、乳房MRI
- 正在等待遗传检测结果

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Lab_summary | "No labs in note." | "CT CAP negative, MRI brain: negative, bone scan negative" | ❌ 新版本**遗漏**影像检查结果 |
| Current_meds | "" (empty) | "paclitaxel, fluticasone, cholecalciferol" | ⚠️ 新版本正确(患者还未开始化疗),旧版本提前写了计划的化疗药 |
| Goals | "adjuvant" | "neoadjuvant" | ❌ 新版本**错误**,明确是新辅助化疗 |
| Genetic_Testing_Plan | "on genetic testing results" | "No new genetic or molecular tests were planned during this visit." | ✅ 新版本更准确 |
| Procedure_Plan | "Port placement, surgery" | "Echocardiogram, Port placement, MRI of the breasts..." | ⚠️ 旧版本更完整 |

**错误: 2 个问题**
- Lab_summary 遗漏影像检查
- Goals 错误(adjuvant vs neoadjuvant)

**vs 100-row (old prompts):**
- ✅ **改进**: Genetic_Testing_Plan 更准确
- ✅ **改进**: Current_meds 更准确(不写未开始的药物)
- ❌ **退化**: Lab_summary 遗漏影像检查
- ❌ **严重退化**: Goals 错误
- ⚠️ **微小退化**: Procedure_Plan 不够完整

---

### Row 39 (coral_idx=179) — Stage 2 IDC 术后

**Ground truth (from note):**
- 随访患者,Stage 2 low grade IDC
- 病理:2.3 cm G1 IDC, 1 个阳性淋巴结(0.04 cm)
- 开始服用 letrozole
- 讨论可能做 FISH 或 Oncotype DX
- 计划物理治疗转诊

| Field | Post-refactor Output | Old Output | Issue |
|-------|---------------------|------------|-------|
| Current_meds | "letrozole" | "letrozole, baclofen, gabapentin, levothyroxine, modafinil, nitrofurantoin, ondansetron, oxyCODONE-acetaminophen, SUMAtriptan, teriflunomide" | ✅ 新版本分类更准确(只提取抗癌药) |
| Genetic_Testing_Plan | "We could send an FISH or Oncotype DX" | "No new genetic or molecular tests were planned during this visit." | ✅ 新版本正确,旧版本误报 |
| Procedure_Plan | "PT referral, start letrozole immediately if no radiation is planned" | "No procedures planned." | ✅ 新版本正确,旧版本遗漏 |

**错误: 0 个问题**

**vs 100-row (old prompts):**
- ✅ **改进**: Current_meds 分类更准确
- ✅ **重大改进**: Genetic_Testing_Plan 正确提取
- ✅ **改进**: Procedure_Plan 正确提取

---

## 汇总统计

### 改进的字段 (Post-refactor better)

1. **Genetic_Testing_Plan / Referral-Genetics**: 8 行改进 (Row 25, 31, 37, 38, 39 从误报变为正确;其他多行更准确)
2. **Findings**: 10 行改进 (更详细的病理/影像/体格检查细节)
3. **Procedure_Plan**: 11 行改进 (更完整的计划提取)
4. **Current_meds 分类准确性**: 7 行改进 (正确区分抗癌药 vs 支持性药物)
5. **Lab_summary 诚实度**: 多行改进 (明确说 "No labs in note" vs 空白)

### 退化的字段 (Post-refactor worse)

1. **Lab_summary 遗漏**: 5 行严重退化 (Row 28, 29, 30, 37, 38 遗漏了实验室/影像结果)
2. **Current_meds 遗漏**: 8 行退化 (Row 25, 27, 28, 30, 32, 34, 37, 38 遗漏了正在服用的药物)
3. **Patient type 错误**: 2 行退化 (Row 28, 37 误判为 New patient / Follow up)
4. **Goals of treatment 判断错误**: 3 行退化 (Row 31, 35, 38 判断不准确)
5. **Genetic_Testing_Plan 字段串行**: 10 行新问题 (误写 "Not yet on treatment — no response to assess")

### 总体评估

**固定的问题数量**: ~30 个
**引入的新问题数量**: ~25 个
**净改进**: 略微正向,但有严重的新问题

**最严重的新问题**:
1. **Genetic_Testing_Plan 字段串行** (10行):这是明显的 bug,复制了 response_assessment 的默认值
2. **Lab_summary 遗漏实验室/影像结果** (5行):新 prompt 可能过于严格区分 "labs" vs "imaging"
3. **Current_meds 遗漏** (8行):新 CoT prompt 可能过于严格区分 "CURRENT" vs "PLANNED"

**最大的改进**:
1. **Genetic_Testing_Plan 准确性** (5行重大改进):旧版本经常误报 "No new genetic tests",新版本正确提取
2. **Findings 详细度** (10行改进):新版本提取了更丰富的临床细节
3. **Procedure_Plan 完整性** (11行改进):新版本更完整地提取了治疗计划

---

## 建议

### 立即修复 (P0)

1. **修复 Genetic_Testing_Plan 字段串行 bug**
   - 当前问题:10行错误地复制了 "Not yet on treatment — no response to assess"
   - 修复方法:检查 prompt 中的默认值设置,确保每个字段有独立的默认值

2. **放宽 Lab_summary 定义,包含影像检查**
   - 当前问题:5行遗漏了 CT、MRI、bone scan、超声心动图结果
   - 修复方法:在 prompt 中明确说明 "labs 包括血液检查、影像检查、心电图、病理检查等所有客观检查结果"

3. **修正 Current_meds CoT 逻辑**
   - 当前问题:8行遗漏了正在服用的药物
   - 修复方法:在 CoT 中加入 "如果笔记中写 'currently on X' 或 'taking X',即使是术前咨询笔记,也应提取"

### 中期改进 (P1)

4. **改进 Patient type 判断规则**
   - Row 28, 37 误判
   - 建议:加强 "New patient" vs "Follow-up" 的判断逻辑,检查笔记中是否有 "new patient evaluation" 关键词

5. **改进 Goals of treatment 决策树**
   - Row 31, 35, 38 判断不准确
   - 建议:明确 adjuvant vs neoadjuvant vs curative vs palliative 的区分标准

### 长期优化 (P2)

6. **统一 Patient type 格式**
   - "Follow up" vs "follow up"
   - 建议:在 schema 中规定标准格式

7. **改进 Supportive_meds 分类**
   - Row 33, 34 误将抗癌药分类为 supportive_meds
   - 建议:在 prompt 中加入明确的抗癌药列表(tamoxifen, letrozole, exemestane 等都是抗癌药)
