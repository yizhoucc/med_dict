# 逐行审查报告 - default_20260301_084320

审查日期: 2026-03-01
审查范围: 100行结果（全部逐行审查，无抽查）
审查标准: 4个核心提取任务 × 4个质量维度（忠实、完整、答对字段、具体）

---

## 总体结论

- **100行中，0行完全正确，100行全部有不同程度的问题**
- ~~发现 2 个 pipeline 级别 bug~~ **经排查均为审查误报**（Row 16 keypoints 与原文一致、Row 89 coral_idx 不同于 Row 88）
- 1 个极严重的受体状态错误（Row 76 HER2+ 写成 HER2-）

---

## 一、错误频率统计

### 按字段分类（出现问题的行数/100）

| 字段 | 有问题行数 | 最常见错误类型 |
|------|-----------|---------------|
| **response_assessment** | ~70/100 | 答非所问（混入计划/症状/预后）、遗漏原文有的证据、幻觉 |
| **Type_of_Cancer** | ~65/100 | 不完整（遗漏HER2状态最常见、缺亚型、缺Ki-67） |
| **current_meds** | ~55/100 | 时态混乱（计划/已停药写成当前）、混入非肿瘤药 |
| **findings** | ~55/100 | 不完整（遗漏病理/影像发现）、答非所问（写症状而非发现） |
| **Stage_of_Cancer** | ~45/100 | 遗漏（原文有分期信息却写"Not mentioned"）、答非所问（写Grade/大小代替分期） |
| **Patient type** | ~35/100 | 首次肿瘤内科会诊被错标为"follow up" |
| **lab_summary** | ~35/100 | 答非所问（影像/病理/受体放入lab）、过时数据、遗漏异常值 |
| **goals_of_treatment** | ~30/100 | 不够精确（"risk reduction"代替"curative/adjuvant"、"neoadjuvant"代替"curative"） |
| **recent_changes** | ~30/100 | 时态错误（历史事件当近期变化）、遗漏重要变化 |
| **supportive_meds** | ~25/100 | 分类错误（抗癌药归为supportive、过敏原当药物） |
| **Distant Metastasis** | ~15/100 | 区域淋巴结错标为远处转移、遗漏转移部位 |
| **Referral** | ~15/100 | 遗漏原文明确的转诊 |
| **Genetic_Testing_Plan** | ~15/100 | 遗漏原文明确的基因/分子检测计划 |
| **second opinion** | ~10/100 | 原文明确写second opinion却标no，或反之 |

### 按错误类型分类

| 错误类型 | 估计出现次数 | 说明 |
|---------|-------------|------|
| **不完整** | ~200+ | 最普遍的问题，遍布所有字段 |
| **答非所问** | ~120+ | 字段间内容混淆，回答的不是该字段的问题 |
| **不忠实/幻觉** | ~60+ | 编造原文没有的信息，或与原文矛盾 |
| **时态错误** | ~50+ | 过去/已停/计划中的内容写成当前 |
| **不具体** | ~40+ | 使用CMS code措辞、模糊描述 |

---

## 二、最常见问题模式（Top 10）

### 1. response_assessment 普遍答非所问（~70行）
**问题**: 模型不理解"治疗响应评估"的含义，经常填入：
- 未来计划（"will start radiation", "plan to begin chemo"）
- 症状描述（"patient reports nausea"）
- 预后/风险评估（"10-15% risk of recurrence"）
- 手术恢复状态（"healing well from surgery"）
- 病理特征描述（"0.9cm grade 1 IDC, margins negative"）
- Oncotype结果（"Low Risk profile"）

**正确理解**: response_assessment 应该是**当前治疗对癌症的效果**，需要具体证据（影像、肿瘤标志物、体检变化）。未开始治疗的应写"Not yet on treatment"。

**代表行**: Row 8, 9, 10, 23, 24, 29, 34, 42, 43, 62, 67, 68, 70, 78, 91, 93

### 2. Type_of_Cancer 遗漏受体状态（~65行）
**问题**: 最常见的是遗漏 HER2 状态，其次是 PR 状态、Ki-67。
- 只写"ER+/PR+ IDC"，不写HER2-
- 只写"breast cancer"，不写亚型和受体
- Row 76: **HER2 3+ (FISH ratio 13) 完全写反为 HER2-**（极严重）

**代表行**: Row 1, 3, 6, 7, 14, 20, 22, 27, 32, 38, 42, 46, 50, 64, 69, 72, 76, 80, 87, 91, 97

### 3. current_meds 时态混乱（~55行）
**问题**: 模型无法区分"当前正在用的药"vs"计划用的药"vs"已经停了的药"：
- 推荐/讨论中的药物写成 current（Row 21, 37, 38, 39, 41, 53, 55, 72, 75, 82, 96）
- 已完成/停用的药物写成 current（Row 14, 31, 44, 45, 57, 67, 78, 85, 100）
- Row 57: **把过敏原当成supportive_meds**（Benadryl, Codeine, Demerol等是ALL: allergies）

**代表行**: Row 7, 8, 14, 21, 31, 37, 38, 39, 41, 44, 45, 53, 55, 57, 67, 72, 75, 78, 82, 85, 96, 100

### 4. Patient type 错标（~35行）
**问题**: 首次肿瘤内科会诊（术后初次讨论辅助治疗、second opinion、转诊来的新患者）被错标为"follow up"。
- 原因：患者已有癌症诊断/手术史，模型误认为是"随访"
- 实际：对于这个provider/clinic来说是"新患者"

**代表行**: Row 7, 8, 13, 21, 23, 24, 29, 32, 40, 47, 55, 60, 62, 63, 65, 71, 72, 87, 88, 93, 94, 96, 99

### 5. Stage_of_Cancer 遗漏或答非所问（~45行）
**问题**:
- 原文有明确分期信息，模型却写"Not mentioned in note"（Row 6, 15, 28, 44, 61, 68, 81, 83, 92, 93）
- 用肿瘤大小/Grade代替正式分期（Row 4, 19, 41, 42）
- 转移性复发只写原始分期，不更新为Stage IV（Row 45）

**代表行**: Row 4, 6, 15, 19, 28, 41, 42, 44, 45, 61, 68, 81, 83, 92, 93, 98

### 6. lab_summary 混入非实验室内容（~35行）
**问题**:
- 影像结果放入lab（超声心动图EF%, CT/MRI结果）（Row 12, 31, 39）
- 病理受体状态放入lab（ER%, PR%）（Row 19, 24）
- 基因组检测结果放入lab（Oncotype score, FISH ratio）（Row 29, 86）
- 使用过时数据（>1年前的结果）（Row 1, 20, 49, 50, 64, 79, 94）

### 7. findings 不完整或答非所问（~55行）
**问题**:
- 写症状（"patient reports pain"）而非检查发现（Row 14, 20, 79）
- 遗漏关键病理/影像发现（Row 1, 2, 5, 9, 10, 30, 39, 68, 73, 74, 82）
- 术前旧发现混入术后当前状态（Row 24）
- findings重复（Row 64, 66）

### 8. goals_of_treatment 不够精确（~30行）
**问题**:
- "risk reduction" vs "curative/adjuvant": 早期乳腺癌术后的辅助治疗应该是"curative"或"adjuvant"，而非"risk reduction"（后者太笼统）
- "neoadjuvant" 是治疗策略，不是治疗目标（Row 65, 69）
- Stage IV metastatic的"curative"应该是"palliative"（反之亦然）

**代表行**: Row 3, 26, 29, 33, 37, 42, 52, 62, 63, 64, 65, 69, 71, 75, 77

### 9. Referral/Genetic_Testing_Plan 遗漏（各~15行）
**问题**: 原文A/P中明确的转诊和基因检测计划被遗漏。
- Genetics consult遗漏（Row 26, 53, 56, 57）
- Oncotype Dx遗漏（Row 52, 60, 72, 74, 97）
- Specialty referral遗漏（Row 59, 97, 99）

### 10. ~~Pipeline Bug~~（审查误报，已排查澄清）
- **Row 16**: ~~keypoints 完全来自 Row 15 的患者数据~~ **误报**：经 run.log 和原文逐字段比对，keypoints 与 Row 16 自身的 note_text 一致。审查 agent 对齐错误。Row 16 的真实问题是 Treatment_Summary 中 LLM 重复退化（"docusate" 重复数十次）。
- **Row 89**: ~~与 Row 88 数据完全重复~~ **误报**：Row 88 coral_idx=227, Row 89 coral_idx=228，患者数据不同。审查 agent 对齐错误。

---

## 三、按字段的改进建议

### Reason_for_Visit
1. **Patient type**: 加规则——如果笔记标题含"Consult"/"New Patient Evaluation"/"second opinion"/"establish care"/"referred by"，则标为"New patient"
2. **summary**: 强制要求包含：癌症类型+受体+分期+本次就诊具体原因。禁止CMS code措辞。

### What_We_Found
3. **Type_of_Cancer**: Prompt 中加 checklist——必须包含 ER/PR/HER2 三项状态+组织学亚型。如果原文有但模型漏了，Gate 5 (SPECIFIC) 应该能捕获。
4. **Stage_of_Cancer**: 明确指示——如果原文有 pT/pN/M 信息，必须推断分期。不能只写 "Not mentioned" 除非真的没有任何分期线索。
5. **lab_summary**: 在 prompt 中加负面示例——"LVEF, CT results, ER/PR%, FISH ratio, Oncotype score 都**不是**实验室值"。只接受血液检查值（CBC, CMP, tumor markers）。加时效性要求（>6个月的标注为 "historical"）。
6. **Distant Metastasis**: 明确区分 regional (axillary LN) vs distant (bone, liver, brain, lung)。

### Treatment_Summary
7. **current_meds**: 这是最需要改进的字段之一。在 prompt 中加明确的时态规则：
   - "recommended"/"discussed"/"plan to start" = **不是** current_meds
   - "completed"/"s/p"/"stopped"/"discontinued" = **不是** current_meds
   - "continue"/"currently on"/"taking daily" = **是** current_meds
   - 非肿瘤药物（降压、精神科、过敏药）不应放入 current_meds
8. **recent_changes**: 明确定义"近期"= 与本次就诊相关的变化，不是几年前的历史。
9. **supportive_meds**: 加负面示例——"过敏原(ALL:)不是药物"、"抗癌药不是supportive med"。

### Goals_of_care
10. **goals_of_treatment**: 在 prompt 中加决策树：
    - 早期乳腺癌 + 术后辅助治疗 → "curative" 或 "adjuvant"
    - 转移性乳腺癌 → "palliative"
    - DCIS → "risk reduction"
    - 新辅助化疗 → "curative"（不要写"neoadjuvant"，那是策略不是目标）
11. **response_assessment**: 这是问题最严重的字段。改进建议：
    - 在 prompt 中加更多负面示例（当前的负面示例已经很多，但模型仍然犯错）
    - 考虑用 Gate 6 (SEMANTIC) 专门检查此字段
    - 加决策树：未开始治疗→"Not yet on treatment"；有影像→引用影像结论；有肿瘤标志物→引用趋势；都没有→"Not mentioned in note"

### 其他字段
12. **Referral**: 在 A/P 中搜索 "refer"/"referral"/"consult" 关键词，确保不遗漏。
13. **Genetic_Testing_Plan**: 在 A/P 中搜索 "Oncotype"/"genetic"/"molecular"/"BRCA"/"germline" 关键词。

---

## 四、Pipeline Bug 排查结论（2026-03-01 已排查）

### ~~Bug 1: Row 16 数据串行~~ → 审查误报
- **原报告**: keypoints 完全来自 Row 15 的患者数据
- **排查结论**: 经 run.log、results.txt、progress.json 逐字段比对，Row 16 的 keypoints 与其自身的 note_text 一致。审查 agent 在批量审查时对齐错误。
- **Row 16 的真实问题**: Treatment_Summary 中 LLM 出现重复退化（"docusate" 重复数十次），属于生成质量问题，非数据串行。

### ~~Bug 2: Row 89 数据重复~~ → 审查误报
- **原报告**: Row 89 和 Row 88 的 coral_idx 相同（228），数据完全相同
- **排查结论**: Row 88 coral_idx=227, Row 89 coral_idx=228，两行是不同的患者数据。审查 agent 对齐错误。

### 潜在风险：KV Cache in-place 修改
- **问题**: 如果 transformers >= 4.36 使用 DynamicCache，`base_cache` 可能在多次 prompt 推理间被污染
- **当前影响**: 100 行结果未观察到严重后果，但属于隐患
- **建议修复**: 在 `ult.py` 的 `run_model_with_cache_manual()` 调用前对 `base_cache` 做 `copy.deepcopy()`

---

## 五、优先级排序

### P0 - 必须立即修复
1. ~~Pipeline Bug（Row 16 数据串行）~~ 已排查为审查误报，无需修复
2. Type_of_Cancer HER2 状态遗漏/错误——受体状态决定治疗方案，错误后果严重
3. current_meds 时态混乱——直接影响患者用药安全

### P1 - 高优先级
4. response_assessment 答非所问——影响治疗效果评估
5. Patient type 错标——影响患者分类
6. Stage_of_Cancer 遗漏——影响预后判断

### P2 - 中优先级
7. lab_summary 混入非实验室内容
8. findings 不完整
9. goals_of_treatment 不够精确
10. Referral/Genetic_Testing_Plan 遗漏

### P3 - 低优先级
11. supportive_meds 分类
12. recent_changes 时态
13. summary 不够具体

---

## 六、逐行审查记录

### Rows 1-10

## Row 1
- 有问题
- summary不具体（"stage not specified"但原文有Stage IIA）; Type_of_Cancer缺HER2-; lab_summary引用2001年过时HCG; findings遗漏右腋窝3cm肿块和阑尾积液

## Row 2
- 有问题
- summary遗漏Lynch Syndrome和多癌种; Type_of_Cancer只写了乳腺癌漏了结肠癌和子宫内膜癌; lab_summary遗漏Hgb 7.7; findings遗漏胸壁感染和精神状态改变; response_assessment"not responding well"过于笼统且缺乏具体证据

## Row 3
- 有问题
- Type_of_Cancer遗漏Ki-67 30-35%; goals_of_treatment写"risk reduction"应为"curative"(新辅助治疗意图治愈)

## Row 4
- 有问题
- Stage_of_Cancer写Grade/大小而非分期; findings遗漏无复发证据的重要发现; recent_changes写"continues letrozole"不是变化

## Row 5
- 有问题
- Distant Metastasis遗漏sternal lesion和brachial plexus involvement; findings遗漏MRI新发现; current_meds anastrozole重复两次; recent_changes时态错误; supportive_meds把anastrozole错归; response_assessment"not responding"应为mixed response

## Row 6
- 有问题
- summary写partial mastectomy但实际是bilateral mastectomy; Type_of_Cancer缺HER2-; Stage写"Not mentioned"但可推断Stage I; response_assessment遗漏手术结果

## Row 7
- 有问题
- Patient type写"follow up"但原文CC是"2nd opinion"; second opinion写"no"但原文明确是2nd opinion; Type_of_Cancer缺HER2+; lab_summary遗漏CA 15-3; findings只写症状不写PET/CT发现; supportive_meds把过敏药Clindamycin列入; response_assessment中CA 15-3趋势描述不准确

## Row 8
- 有问题
- Patient type应为New patient; Stage不完整; current_meds列了2018年历史治疗; recent_changes列了2018年事件; supportive_meds列了2018年一次性抗生素; response_assessment答非所问（写风险评估而非手术病理结果）

## Row 9
- 有问题
- Type_of_Cancer缺HER2-和Ki-67; Stage只写"Stage II"缺术后详情; findings只写症状遗漏关键病理; current_meds遗漏lorazepam等; recent_changes遗漏手术; response_assessment把neuropathy改善当cancer response

## Row 10
- 有问题
- in-person写"Televisit"但实际是telephone(视频失败转电话); Type_of_Cancer格式冗余(ER+/HR+); Stage遗漏8.8cm肿瘤和淋巴结信息; findings写"no new findings"但有术后病理和Oncotype结果; recent_changes写"stopped smoking"不是treatment change; response_assessment把Oncotype结果当treatment response

### Rows 11-20

## Row 11
- 有问题
- Type_of_Cancer ER/PR/HER2受体状态幻觉（原文无明确记录）; current_meds包含已停的Letrozole; recent_changes答非所问; response_assessment写症状而非治疗反应; lab_summary遗漏多个异常值

## Row 12
- 有问题
- lab_summary放了EF%(影像不是lab); findings遗漏2个新脑转移灶细节和CT CAP结果; response_assessment遗漏体部CT和新脑转移信息

## Row 13
- 有问题
- Patient type应为New patient(术后首次med onc); Type_of_Cancer遗漏DCIS score 60; Medication_Plan写"will start"但患者尚未决定

## Row 14
- 有问题（严重）
- Type_of_Cancer缺HER2-; Stage写"Not mentioned"但明显Stage IV; current_meds严重错误（已停药写成当前）; supportive_meds列了患者未服用的Cymbalta; findings答非所问（写症状不写检查发现）; lab_summary遗漏CA 27.29下降趋势

## Row 15
- 有问题
- Type_of_Cancer缺HER2+(FISH positive ratio 2.0); Stage写"Not mentioned"但有"Clin st I/II"; findings遗漏MRI发现

## Row 16
- ~~**致命问题 - Pipeline Bug**~~ **审查误报**（经排查 keypoints 与自身 note_text 一致）
- 真实问题：Treatment_Summary 中 LLM 重复退化（"docusate" 重复数十次）

## Row 17
- 有问题
- Patient type应为New patient(consultation); Stage写"Not mentioned"但原文有; supportive_meds写了"nutritionist"(referral不是药物); response_assessment答非所问（写预后估计）

## Row 18
- 有问题（轻微）
- Patient type应为New patient; current_meds写了尚未开始的adjuvant endocrine therapy; Stage不完整(只写pT1b); Genetic_Testing_Plan遗漏genetic counseling讨论

## Row 19
- 有问题
- Stage写Grade/大小而非分期; lab_summary把ER/PR%放入(非lab值); response_assessment答非所问（写治疗计划）; findings遗漏MRI详细发现

## Row 20
- 有问题
- Patient type应为New patient; second opinion标"yes"但原文无此信息; Type_of_Cancer缺PR+; Distant Metastasis遗漏淋巴结转移; findings写症状而非检查发现; lab_summary引用2013年数据

### Rows 21-30

## Row 21
- 有问题
- Patient type应为New patient(referred for consultation); Stage不具体; current_meds幻觉（Arimidex尚未使用）; lab_summary遗漏osteopenia; findings不完整

## Row 22
- 有问题
- Type_of_Cancer应标HER2-; lab_summary遗漏大量检验值; Distant Metastasis不完整; recent_changes遗漏letrozole switched to anastrozole

## Row 23
- 有问题
- Patient type应为New patient; Type_of_Cancer遗漏多灶性PR状态差异; Stage不具体; response_assessment答非所问（写治疗建议）

## Row 24
- 有问题
- Patient type应为New patient; Stage混淆影像大小和病理大小; lab_summary把受体状态放入; findings时态错误; response_assessment答非所问（写计划）

## Row 25
- 有问题
- summary缺具体转移信息; Type_of_Cancer遗漏受体状态变化（脑转移灶ER-/PR-/HER2-）; Distant Metastasis不具体("multiple sites"); findings不完整; Imaging_Plan遗漏"Scan in 3 weeks"

## Row 26
- 有问题
- summary用CMS code措辞; Type_of_Cancer缺Ki-67和亚型; findings幻觉（75%是Ki-67不是肿瘤大小）; goals_of_treatment应为"curative/adjuvant"而非"risk reduction"; Referral遗漏genetics和social work

## Row 27
- 有问题
- lab_summary答非所问（计划中的检查不是结果）; recent_changes答非所问; current_meds遗漏zolendronic acid; Type_of_Cancer缺HER2-; radiotherapy_plan把MRI当放疗

## Row 28
- 有问题
- findings不忠实（写"no new findings"但有术后病理结果）; recent_changes不准确; Stage不完整

## Row 29
- 有问题
- Patient type应为New patient; Type_of_Cancer遗漏micropapillary特征和PR状态差异; Stage遗漏微转移(N1mi); goals_of_treatment应为"adjuvant"; response_assessment幻觉（Oncotype Low Risk不是治疗反应）; lab_summary把基因组检测放入

## Row 30
- 有问题
- summary写"early stage"不准确(9cm HER2+肿瘤); response_assessment把baseline PET/CT当治疗响应; lab_summary的tumor markers趋势标注不清

### Rows 31-40

## Row 31
- 有问题
- Distant Metastasis遗漏肝转移和可能脑转移; lab_summary把LVEF放入(影像非lab); current_meds把尚未开始的Doxil列为current; recent_changes答非所问; response_assessment混入不确定影像发现

## Row 32
- 有问题
- Patient type应为New patient; Type_of_Cancer缺亚型(pleomorphic lobular); Stage可更具体(T2,N0(i+)); current_meds遗漏pertuzumab; recent_changes原因不准确（写infusion reaction但实际是diarrhea）

## Row 33
- 有问题
- Type_of_Cancer缺HER2-; Stage不完整(遗漏Stage IIIA); current_meds letrozole和Femara重复(同一药物); current_meds混入高血压药; goals_of_treatment "risk reduction"应为"adjuvant"

## Row 34
- 有问题
- Type_of_Cancer遗漏2020年复发活检的PR+; current_meds不忠实（anastrozole已自行停用、tamoxifen是新计划）; recent_changes搞反了（tamoxifen和anastrozole互换）; response_assessment答非所问（写讨论内容）; Procedure_Plan时态错误

## Row 35
- 有问题
- Type_of_Cancer缺ER状态; current_meds HPI和Assessment矛盾(tamoxifen vs anastrozole); response_assessment遗漏（原文有"no signs of malignancy"）; Imaging_Plan遗漏scheduled mammogram

## Row 36
- 有问题（严重）
- Distant Metastasis完全错误：把术后感染的hypermetabolism误判为左乳转移; Metastasis同上错误; current_meds遗漏valtrex

## Row 37
- 有问题
- current_meds把推荐方案(AC/Taxol)写成当前用药; recent_changes答非所问（写治疗决策不是变化）; goals_of_treatment应为"curative"; findings不够完整

## Row 38
- 有问题
- Type_of_Cancer缺PR weak+(15%)和HER2-和BRCA1; current_meds把未来药物(olaparib/xeloda)写成当前; recent_changes遗漏停Taxol; response_assessment混入计划

## Row 39
- 有问题
- lab_summary全是影像结果（CT/MRI/bone scan不是lab值）; current_meds把计划中的paclitaxel写成当前; recent_changes不忠实（新诊断无既往方案切换）; radiotherapy_plan答非所问; findings不完整

## Row 40
- 有问题
- Patient type应为New patient; Type_of_Cancer缺Grade/Ki-67且PR有矛盾; Stage不够精确; findings不忠实（SLN中的ITC写成DCIS）; current_meds过度列举非肿瘤药; response_assessment答非所问; Imaging_Plan遗漏DEXA

### Rows 41-50

## Row 41
- 有问题
- Type_of_Cancer PR仅1%应注明且HER2 FISH未做不能写HER2-; Stage答非所问(写大小/Grade非分期); current_meds把计划中的AC-Taxol写成当前; response_assessment可改进

## Row 42
- 有问题
- Type_of_Cancer缺HER2-; Stage答非所问; findings不忠实（deep margin re-excision后已clear）; goals_of_treatment应为"adjuvant"; response_assessment不忠实

## Row 43
- 有问题
- summary遗漏second primary; Type_of_Cancer缺grade; lab_summary混入过时围手术期数据; findings不够具体; response_assessment答非所问; recent_changes遗漏手术

## Row 44
- 有问题
- Type_of_Cancer混合两套受体数据; Stage遗漏; current_meds严重混乱（已完成化疗+标注not taking的药+拼写错误）; recent_changes不忠实; response_assessment答非所问

## Row 45
- 有问题
- Type_of_Cancer缺PD-L1; Stage只写IIIB但已转移应为Stage IV; current_meds把已停的Xeloda写为当前; response_assessment答非所问（写治疗目标讨论）; Metastasis遗漏hilar LN

## Row 46
- 有问题
- Type_of_Cancer缺HER2-和Ki-67; Stage不完整; recent_changes遗漏新开的letrozole; supportive_meds遗漏多个药物和包含患者拒绝的gabapentin; response_assessment含未来计划

## Row 47
- 有问题（轻微）
- Patient type应为New patient(second opinion)

## Row 48
- 有问题
- Type_of_Cancer遗漏可能的invasion; current_meds/supportive_meds列了大量非肿瘤药且两者重复; findings不够具体

## Row 49
- 有问题
- second opinion可能错标; Metastasis遗漏axillary node; current_meds全是非肿瘤药; supportive_meds列了DISCONTINUED的药; findings不完整（遗漏MRI和Oncotype）; lab_summary过时

## Row 50
- 有问题
- Type_of_Cancer缺PR+; Distant Metastasis遗漏lymph nodes; lab_summary过时(>1年); current_meds包含标注not taking的lupron; response_assessment遗漏（原文说"disease under great control"）

### Rows 51-60

## Row 51
- 有问题
- Patient type应为Education visit而非follow up; findings重复"in stable condition"; goals_of_treatment不准确; Medication_Plan幻觉（"starting today"无原文支持）

## Row 52
- 有问题
- Stage遗漏; lab_summary遗漏尿妊娠试验; current_meds错误（PRN术后药不是癌症治疗药）; Referral遗漏reproductive health; Genetic_Testing_Plan遗漏Oncotype

## Row 53
- 有问题（严重）
- current_meds严重错误（推荐方案写成当前用药）; Metastasis遗漏SLN阳性; Referral遗漏genetics counseling

## Row 54
- 有问题
- Type_of_Cancer缺BRCA2; Stage不准确; supportive_meds分类错误; response_assessment遗漏; Therapy_plan错误（已完成化疗写成continue）; Medication_Plan bowel regimen分类错误

## Row 55
- 有问题
- Patient type可能应为New patient; Type_of_Cancer缺HER2-和Ki-67; current_meds错误（推荐药写为已用药且患者拒绝用药）

## Row 56
- 有问题
- summary幻觉（"90% tumor size"实际是Ki-67）; Type_of_Cancer用CMS code措辞; findings同样幻觉; Referral遗漏genetics; Genetic_Testing_Plan遗漏

## Row 57
- 有问题（严重）
- **supportive_meds把过敏原当药物**（ALL: Benadryl, Codeine, Demerol, Motrin, PCN）; current_meds时态错误（已完成化疗写为current）; response_assessment遗漏化疗后残留; Referral遗漏genetic counseling

## Row 58
- 有问题（轻微）
- Type_of_Cancer缺Ki-67和Oncotype; lab_summary遗漏DEXA结果; response_assessment遗漏

## Row 59
- 有问题
- current_meds混乱（sequential药物看起来像同时在用）; supportive_meds包含标注not taking的药; Referral遗漏psychiatry; Therapy_plan不准确

## Row 60
- 有问题
- Patient type应为New patient; Type_of_Cancer缺HER2-和Ki-67; findings不完整; response_assessment答非所问; Genetic_Testing_Plan严重遗漏（Oncotype Dx）; Procedure_Plan与原文矛盾（患者拒绝手术）

### Rows 61-70

## Row 61
- 有问题
- Stage遗漏（原文有"early stage"）; lab_summary空但应注明无结果; findings不完整（遗漏CT和MRI发现）

## Row 62
- 有问题
- Patient type可能应为New patient; lab_summary幻觉（"lab results pending"原文无此说法）; response_assessment不忠实; follow_up_next_visit答非所问

## Row 63
- 有问题
- Patient type应为New patient(second opinion); response_assessment不忠实（"not responding as expected"过于笼统）; goals_of_treatment应为"curative"

## Row 64
- 有问题
- Type_of_Cancer缺PR+和HER2-; lab_summary引用2年前旧数据; findings重复; goals_of_treatment不准确（probable oligomet应为aggressive curative intent）; response_assessment遗漏

## Row 65
- 有问题
- Patient type不准确; summary过长; lab_summary列出检查名称无结果值; goals_of_treatment应为"curative"而非"neoadjuvant"; current_meds答非所问

## Row 66
- 有问题（轻微）
- Stage遗漏（可推断locally advanced）; current_meds列了标注not taking的药; findings重复; Genetic_Testing_Plan遗漏

## Row 67
- 有问题
- Distant Metastasis应为"Unknown, staging pending"; current_meds列了已完成的化疗; recent_changes遗漏化疗停止; response_assessment遗漏（原文有axillary LN缩小证据）; Genetic_Testing_Plan遗漏

## Row 68
- 有问题
- Type_of_Cancer HER2状态错（TCHP方案说明HER2+但写成HER2-）; Stage遗漏; response_assessment遗漏（原文有"good clinical response" + MRI CR）; findings不完整

## Row 69
- 有问题
- Type_of_Cancer缺HER2-和Ki-67; lab_summary遗漏（原文有详细labs）; goals_of_treatment应为"curative"而非"neoadjuvant"

## Row 70
- 有问题
- summary用CMS code措辞; Type_of_Cancer右侧PR状态错误; Stage遗漏; response_assessment答非所问（应描述neoadjuvant pathologic response）; Procedure_Plan时态错误; current_meds遗漏supplement

### Rows 71-80

## Row 71
- 有问题（严重）
- lab_summary全错（ROS内容放入lab，且"negative for nausea"与原文ROS矛盾）; response_assessment幻觉（"responding to chemotherapy"但仅开始2周无证据）; goals_of_treatment应为"curative"; Type_of_Cancer缺HER2-

## Row 72
- 有问题
- Patient type应为New patient; Type_of_Cancer遗漏neuroendocrine differentiation; findings答非所问; current_meds时态错误; Genetic_Testing_Plan遗漏Oncotype; supportive_meds列了非癌症药

## Row 73
- 有问题
- Type_of_Cancer缺HER2-; findings遗漏关键发现（fat necrosis排除复发）; Lab_Plan遗漏

## Row 74
- 有问题
- lab_summary答非所问; findings不完整; Genetic_Testing_Plan遗漏; follow_up遗漏

## Row 75
- 有问题
- Distant Metastasis不忠实（axillary LN是regional不是distant）; current_meds时态错误; goals_of_treatment应为"curative"; findings不完整; Genetic_Testing_Plan遗漏

## Row 76
- **有问题（极严重）**
- **Type_of_Cancer HER2状态完全写反：原文HER2 3+ (FISH ratio 13)写成HER2-**; supportive_meds列了标注not taking的药

## Row 77
- 有问题（轻微）
- goals_of_treatment应为"curative"; current_meds时态错误; response_assessment遗漏; recent_changes不忠实

## Row 78
- 有问题（严重）
- current_meds严重错误（已停药写为current，患者无系统治疗）; response_assessment遗漏（CT明确显示PD）; Medication_Plan幻觉（患者拒绝化疗但写"plan to start Doxil"）; Procedure_Plan不忠实

## Row 79
- 有问题
- Type_of_Cancer不具体（缺受体状态）; findings答非所问; response_assessment遗漏（tumor marker rising + bone pain增加）; lab_summary过时(>1.5年); follow_up遗漏; Referral遗漏pain management

## Row 80
- 有问题（轻微）
- Type_of_Cancer缺HER2-; findings不够具体（缺病理细节）; lab_summary过长未筛选

### Rows 81-90

## Row 81
- 有问题
- Stage写"Not mentioned"但原文有Stage I; Type_of_Cancer缺亚型(tubular type); findings遗漏DCIS/ADH/ALH

## Row 82
- 有问题
- assessment_and_plan上游提取错误（抓到旧A/P）; Stage IB vs Stage II不一致; current_meds混入未来计划; findings不够完整

## Row 83
- 有问题
- Distant Metastasis错误（axillary LN是regional不是distant）; second opinion可能幻觉; Type_of_Cancer遗漏受体状态

## Row 84
- 有问题
- second opinion错标为no; Type_of_Cancer未注明受体变化; Stage原始分期不确定; findings遗漏脑/肝转移; Lab_Plan遗漏LP

## Row 85
- 有问题
- current_meds时态错误（已PD的方案仍列为current）; lab_summary遗漏CA 15-3; response_assessment不准确; Therapy_plan与A/P矛盾; findings遗漏脑MRI

## Row 86
- 有问题
- Type_of_Cancer遗漏HER2+且受体变化未注明; Stage不准确（原始非Stage IV）; lab_summary放了FISH(非lab值); current_meds遗漏denosumab

## Row 87
- 有问题
- Patient type应为New patient(second opinion); Type_of_Cancer缺HER2-; findings答非所问（帕金森症状与癌症无关）

## Row 88
- 有问题
- Patient type/second opinion错误; summary用CMS code; Type_of_Cancer ER/PR状态不准确（弱阳性且转移灶阴性）; Distant Metastasis遗漏肺和淋巴结; recent_changes时态错误; response_assessment遗漏; Genetic_Testing_Plan遗漏

## Row 89
- ~~**Pipeline Bug - 数据重复**~~ **审查误报**（Row 88 coral_idx=227, Row 89 coral_idx=228，不同患者）
- 真实问题：supportive_meds 把 tamoxifen 归为支持药物; Procedure_Plan/radiotherapy_plan 时态错误

## Row 90
- 有问题
- lab_summary严重遗漏(TSH/CBC/CMP全漏); current_meds药名错误("Dexmethylprednisolone"不存在); response_assessment错误（写"Not yet on treatment"但已完成neoadjuvant+手术+正在adjuvant AC）; findings混杂无关症状

### Rows 91-100

## Row 91
- 有问题
- Type_of_Cancer缺HER2-; Distant Metastasis遗漏淋巴结; lab_summary遗漏多项; current_meds把一次性药物列为常规; response_assessment幻觉（"likely responding"无支持）

## Row 92
- 有问题
- Type_of_Cancer仅写"breast cancer"缺受体; Stage遗漏(明显Stage IV); recent_changes答错字段; Lab_Plan遗漏; supportive_meds分类不当

## Row 93
- 有问题
- Type_of_Cancer格式混乱且缺HER2+; Stage遗漏(原文有Stage 1); Patient type应为New patient; response_assessment幻觉（"not responding well"但还没开始治疗）; Imaging_Plan/Genetic_Testing_Plan/Procedure_Plan都是过去完成的而非计划

## Row 94
- 有问题
- Patient type应为New patient; summary用CMS code; lab_summary过时(2年前); findings不具体; supportive_meds遗漏gabapentin

## Row 95
- 有问题
- current_meds混淆时态; summary不准确; response_assessment含无意义数字; findings过长

## Row 96
- 有问题
- Patient type应为New patient; current_meds幻觉(tamoxifen未开始); findings答错字段; response_assessment答错字段

## Row 97
- 有问题
- Type_of_Cancer缺HER2-; response_assessment幻觉("treatment expected to be curative"); Medication_Plan幻觉; Genetic_Testing_Plan遗漏Oncotype; Referral遗漏Rad Onc; follow_up遗漏

## Row 98
- 有问题（轻微）
- Stage不规范; recent_changes答错字段; response_assessment不够精确

## Row 99
- 有问题
- Patient type应为New patient; Type_of_Cancer遗漏右侧ER-/PR-/HER2 3+; Stage遗漏右侧; Referral严重遗漏（3个转诊全丢）

## Row 100
- 有问题
- Stage不确定（原文无"Stage IIA"）; current_meds列了已停药(Xeloda, Faslodex); recent_changes时态; radiotherapy_plan幻觉
