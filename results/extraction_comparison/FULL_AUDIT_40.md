# 全字段全量人工审查 — 最终输出 (breast run4 + pdac run2)

> 触发：修复 9 个 stage/met/response hook + 重跑多次后，最终 FINAL 输出的**所有字段**（非仅诊断四维）从未经完整审查。本 doc 按 CLAUDE.md 严格全字段审查。
> 方法：subagent 并行初审（用户授权）→ 主 Claude 亲自复核全部 P0/P1 原文后才采纳。subagent 与主 Claude 均禁止用脚本/grep/regex 做判断，逐字逐句读 note 原文找依据。
> 数据：PL = pipeline_breast_FINAL.txt (run4) / pipeline_pdac_FINAL.txt (run2)。BL = baseline_extract_*_json.txt。每个 sample 含 note_text 全文 + assessment_and_plan + keypoints(JSON) + attribution。

## 🎯 总目标：让 PL 全方位碾压 BL
每个字段审查三件事，并且**始终带着"如何让 PL 在这个点上碾压 BL"思考**：
1. **PL 正确性**：P0/P1/P2/OK（对照 note 原文 + attribution + prompt 字段定义）
2. **PL vs BL 胜负**：PL / BL / TIE / NA（谁更忠实+准确+恰当）
3. **PL 改进机会**：我们**可以自由操作 PL 侧**——加更多提取 stage、加更多 POST hook、改多层级 prompt（extraction.yaml 字段定义 / CoT / 示例）。凡 PL 有错、或仅平 BL、或输 BL 的点，都要给出**具体可执行**的 PL 改进建议（加什么 hook / 改哪个 prompt 字段 / 加什么 stage），目标=该字段既对又碾压 BL。
→ 改进机会汇总到本 doc 末尾「## PL 改进机会清单」，作为下一轮迭代输入。

## 分级定义
- **P0**：幻觉/编造（note 中不存在的药/诊断/站点/数值/分期），或把疑似说成确诊。绝对不可发。
- **P1**：重大错误（方向错、关键遗漏致误导、时态混淆把过去当现在/计划当现状、答非所问）。
- **P2**：小问题（模糊词、不够具体、可推断却写"未提及"、格式瑕疵）。
- **OK**：该字段忠实且恰当。

## 审查字段清单（每个 sample 都要逐个过）
Reason_for_Visit(Patient type/second opinion/in-person/summary) · Cancer_Diagnosis(Type/Stage/Distant Met/Metastasis) · Lab_Results(lab_summary) · Clinical_Findings(findings) · Current_Medications(current_meds) · Treatment_Changes(recent_changes/supportive_meds) · Treatment_Goals(goals) · Response_Assessment · Medication_Plan · Therapy_plan · radiotherapy_plan · Procedure_Plan · Imaging_Plan · Lab_Plan · Genetic_Testing_Plan · follow_up · Advance_care · Referral · Genetic_Testing_Results

## STATUS（上下文满了从这里恢复）
- [进行中] breast 1-10 ✓ | breast 11-20 待 | pdac 1-20 待
- 进度：10/40 | 下一步：派 subagent 审 breast 11-15(PL行2182/2322/2474/2626/2781; BL行768/844/920/996/1072; coral_idx 30-34)
- 派 subagent 方式：general-purpose，每个审1 sample，给PL行范围+BL行范围+prompt路径+三重任务(正确性/vs BL/改进)。主Claude复核P0/P1。
- PL行范围(breast): R1 620,R2 760,R3 921,R4 1091,R5 1237,R6 1392,R7 1547,R8 1711,R9 1869,R10 2024,R11 2182,R12 2322,R13 2474,R14 2626,R15 2781,R16 2942,R17 3097,R18 3261,R19 3413,R20 3574(end3715)
- PL行范围(pdac): R1 445,R2 591,R3 761,R4 931,R5 1074,R6 1247,R7 1413,R8 1559,R9 1711,R10 1869,R11 2033,R12 2185,R13 2340,R14 2510,R15 2668,R16 2823,R17 2990,R18 3127,R19 3273,R20 3428(end3584)
- BL行范围(both, 每76行): Rn = 8+(n-1)*76 起，即 R1 8,R2 84,R3 160,R4 236,R5 312,R6 388,R7 464,R8 540,R9 616,R10 692,R11 768,R12 844,R13 920,R14 996,R15 1072,R16 1148,R17 1224,R18 1300,R19 1376,R20 1452
- prompt: breast=prompts/extraction.yaml+prompts/plan_extraction.yaml；pdac=prompts/pdac/extraction.yaml+prompts/pdac/plan_extraction.yaml

## ★ 汇总 tally（实时更新, 主复核后）
| 严重级 | 计数 | 涉及 sample |
|---|---|---|
| P0 | 1 | b4(Stage III幻觉) |
| P1 | ~18 | b1(Type) b2(goals) b3(therapy待复核) b4(Type/goals) b5(GeneticRes/Procedure/Metastasis) b6(findings) b7(current_meds漏abraxane/Lab"No labs") b8(Lab_Plan TTE/follow_up漏) b9(Type/DistMet regional/FNA错配) b10(Imaging漏echo/Referral germline) |
| P2 | 多 | 各sample均有 |

### 反复出现的模式(改进重点, 10/40 已很清晰)
1. **plan类时态混淆**：已完成(已做影像/已抽labs)当未来计划(b2,b3) → plan时态过滤hook强化
2. **stage幻觉**：note无依据却给数字分期(b4 Stage III) → "数字stage须note有stage/pT/pN依据否则Not staged"hook ⭐P0
3. **goals方向错**：无转移/疑似IV却palliative(b2,b4,b6) → goals=palliative回查DistMet,无确诊转移→curative/条件化 hook
4. **字段错配(高频)**：内容真实放错字段 → 字段净化hook。实例:b5病理→GeneticRes、b5/b9 AI/procedure、b8 TTE→Lab_Plan、b9 FNA→Imaging/Genetic、b10 germline→Referral
5. **Type措辞**：疑似当确诊"metastatic recurrence"(b4,b9)、夹带矛盾受体(b1,b10丢FISH) → Type清洗+FAITHFUL疑似门
6. **regional vs distant**：节点(cervical/supraclav/level V/internal mammary/axillary)误当distant(b9) → 节点regional hook
7. **current_meds漏药**：doublet"X and Y"只取一个(b7漏abraxane) → doublet补全+白名单
8. **Lab_Results "No labs"**：实为"labs in range"(b7) → in-range忠实表述
9. **plan字段遗漏**：Imaging漏baseline echo(b10)、Procedure漏诊断biopsy(b6)、follow_up漏appointment(b8) → 必收清单+全文搜索
10. **Referral堆砌**：follow up子键塞重复plan项(b1,b5,b6,b10) → Referral去重hook

### ⚠️ 阶段性结论(10/40)
全字段审查证实了用户的担忧：**重跑后非诊断字段(plan类/Type措辞/字段归位/完整性)有大量 P1**，是四维重评完全没覆盖的。诊断四维(stage/met/response/distmet)PL确实强,但plan字段错配+疑似当确诊+完整性遗漏让多个sample被BL在个别字段反超。要"全方位碾压"需基于本审查的改进清单做新一轮 hook+prompt 迭代→重跑→重审。本doc持久化,可跨上下文继续。

---

## 逐 sample 审查结果
（subagent 初审 → 主 Claude 复核后写入；每条问题标 [subagent报告] / [主复核确认/驳回]）
（P0/P1 校准：纯编造 note 中不存在的事实=P0；内容真实但放错字段/方向错/误导措辞=P1）

### breast ROW1 (coral_idx=20) — coral_idx对齐✓
- **P1** Type_of_Cancer: 最终定性 TNBC 正确，但括号夹带被 10/31 addendum 推翻的 "FISH ratio 2.1" 矛盾数据（addendum 明写 HER2 negative）→ 误导读者以为 HER2 borderline。[subagent报告,引用addendum可信]
- P2 Stage: "Stage IIB (pT2N1a)" 自行把 pTN 转 AJCC（prompt 禁止）；P2 findings 夹带 CHF/LVEF 合并症。
- vs BL: PL胜12 / BL胜3(Type夹FISH、Stage转换、therapy_plan写None漏化疗讨论) / TIE9。
- 改进: ①TNBC一致性清洗hook(剥离最终诊断里的矛盾FISH/HER2+片段) ②pTN不转AJCC hook ③therapy_plan延迟话术(treatment deferred pending staging)。

### breast ROW2 (coral_idx=21) — coral_idx对齐✓
- **P1** goals_of_treatment: "palliative" 方向错——局部复发(local regional recurrence)、无远处转移、追求 "long-term disease control"+手术切除企图→应 curative/局部根治。[subagent引用"long-term disease control"可信]
- P2 Imaging_Plan/Lab_Plan: 把已完成(PET/CT/brain MRI已出结果、labs已抽)的过去项当未来计划(时态混淆)。P2 Type HER2-属弱推断(1994老诊断无HER2检测)。
- vs BL: PL胜11 / BL胜1(Imaging_Plan) / TIE12。
- 改进: ①Treatment_Goals决策树加"局部复发+无远处+根治企图→curative"+POST hook(stage含locally recurrent且DistMet=No且goals=palliative→改curative) ②plan类时态过滤强化(已出结果=过去)。

### breast ROW3 (coral_idx=22) — coral_idx对齐✓
- **P1?** Therapy_plan 尾部 "; physical therapy"——note 全文疑无 PT，可能幻觉。[待主复核note]
- P2 Imaging_Plan "Echocardiogram"(echo已完成"looks good"当计划)；P2 lab_summary漏"dehydration"限定;P2 findings重复受体状态。
- vs BL: PL胜11 / BL胜2(lab_summary含dehydration、Imaging_Plan如实标已完成) / TIE12。
- 改进: ①imaging已完成("done/looks good")清空hook ②lab保留临床限定语 ③therapy_plan忠实门(每项须A/P可溯源,剔除PT)。

### breast ROW4 (coral_idx=23) — coral_idx对齐✓ ⚠️重点
- **P0 [主复核确认]** Stage_of_Cancer "Originally unspecified, now Stage III"——note全文无任何Stage III依据(PET无淋巴结/无远处转移/腋窝正常/肿块~1.4cm=T1)，纯编造。上轮四维审查漏判(只记Q10 TIE)。
- **P1 [主复核确认]** Type_of_Cancer "...metastatic recurrence"——PET明写"No evidence of metastatic disease"，是局部复发非转移，且与DistMet=No自相矛盾。
- **P1 [主复核确认]** goals "palliative"——无转移+taxol根治化疗→应curative。
- P2 Imaging_Plan "Mammogram"(A/P无)；P2 Referral.Specialty=incoming/past surgery consult(应只保留outgoing)。
- vs BL: PL胜7 / BL胜4(Stage幻觉/Type/goals/Imaging—BL更保守忠实) / TIE14。**本sample PL因幻觉反输BL多项。**
- 改进: ①**stage含I/II/III/IV数字但note无"stage/pT/pN"依据→强制"Not staged in note" hook**(关键) ②"recurrent disease"在乳腺内病灶≠metastatic(Type区分+CoT) ③goals=palliative回查DistMet,无转移则降curative hook ④imaging_plan须A/P-grounding ⑤Referral剔除incoming/past consult。

### breast ROW5 (coral_idx=24) — coral_idx对齐✓ ⚠️重点
- **P1 [主复核:字段错配非幻觉,下调P0→P1]** Genetic_Testing_Results 污染：塞入手术病理(LVI/margins/DCIS/0-8 LN+)+pending Oncotype，而真正该填的 MammaPrint(左MP-0.614/右MP+0.321)只抓到右侧且语境错。内容真实但放错字段。
- **P1 [确认]** Procedure_Plan "AI therapy 5yr"——AI是药物非手术(prompt明令procedure不得含系统药)。
- **P1 [确认]** Metastasis "No"——漏左腋窝LN核穿证实转移(N1,regional);"Distant Met:No"对但"Metastasis:No"遗漏区域转移。
- P2 goals_description "Not explicitly stated"(原文有"reduce distant recurrence risk")。
- vs BL: PL胜9 / BL胜2(goals_description、Genetic_Testing_Results) / TIE14。Stage双侧/Type/findings PL碾压。
- 改进: ①Genetic_Testing_Results剔除标准病理+pending,保留MammaPrint分数hook(+prompt示例) ②Procedure_Plan药物白名单剔除hook ③Metastasis区分regional/distant,区域转移须写"Regional: <部位>LN(N1)" ④goals_description扫描intent短语。

### breast ROW6 (coral_idx=25, 疑似骨转Stage IV pending) — 对齐✓
- P1 findings: PET "concerning for"骨灶被转述为既成转移+重复受体诊断信息(冗长)。P2 Metastasis "Not sure"(腋窝LN已确诊区域转移+骨疑似,应分述)。P2 goals "palliative"(疑似IV未确诊,应条件化)。
- vs BL: PL胜8/BL胜1/TIE12。BL胜=Procedure_Plan(PL漏"biopsy for definitive stage IV diagnosis",BL抓到)。PL碾压点:current_meds排除10种非癌药、Distant Met用Suspected(BL"Yes bone met"疑似当确诊)、Response不臆测。
- 改进: ①Procedure_Plan加"诊断性/分期biopsy必收"示例 ②findings去重受体+良性灶(paraganglioma)标注+PET"concerning for"不转确诊 ③Metastasis分regional确诊/distant疑似。

### breast ROW7 (coral_idx=26, 确诊转移liver+nodes肝活检证实) — 对齐✓
- **P1** current_meds: 漏 abraxane(nab-paclitaxel)——note"started pembrolizumab **and** abraxane",只列了pembrolizumab(doublet漏药)。
- **P1** Lab_Results "No labs in note": note明写"Labs are in range for continuation"(答非所问/遗漏)。
- P2 findings/Response混入治疗前(01/27)PET当今日证据(时态);P2 imaging_plan臆造"PET/CT"模态(note只说"guided by symptoms/every 3-4mo")。
- vs BL: PL胜11/BL胜3(Lab_Results、Imaging_Plan、current_meds各有问题)/TIE8。PL碾压:Stage(IV vs BL"Not specified")、Type、基因结果。
- 改进: ①current_meds: "started X and Y"doublet必全收+abraxane同义词入白名单 ②Lab "in range/within limits"→"Labs reviewed,in range"不写No labs ③imaging模态禁臆造hook ④治疗起始日时态门(早于起始的影像不作今日证据)。

### breast ROW8 (coral_idx=27, stage IIA pT2(m)N1a术后adjuvant) — 对齐✓
- **P1** Lab_Plan "ordered for pre-chemotherapy TTE": TTE是心超(影像)非化验,放错字段(应Imaging_Plan,Lab_Plan应"No labs")。
- **P1** follow_up_next_visit "Not specified": 漏note明写"appointment with [REDACTED] on 10/10/2017"。
- P2 goals_description空(有intent语言可填);P2 Referral.follow up有10/10但未同步到follow_up_next_visit。
- vs BL: PL胜7/BL胜2(Current_Medications信息量、Lab_Plan BL=None反而对)/TIE12。PL碾压:Patient type(BL误判Follow up)、Stage、Response(BL违规"Not applicable")、schema完整。
- 改进: ①Lab_Plan含TTE/echo→移Imaging_Plan+清Lab_Plan hook ②follow_up全文搜"appointment with...on date"(不止A/P) ③goals_description扫intent。

### breast ROW9 (coral_idx=28, 疑似转移possibly metastatic FNA待) — 对齐✓ ⚠️
- **P1** Type_of_Cancer "with metastatic recurrence": 疑似当确诊(note"possibly considered metastatic"+"FNA to confirm")。[同b4模式]
- **P1** Distant Metastasis "Suspected, to left cervical LN": 实为level Vb/supraclavicular=乳腺**regional**非distant(prompt明文),且定位错。BL"Not sure"反而更稳。
- **P1** Genetic_Testing_Plan="FNA"+Imaging_Plan="FNA": FNA是穿刺(procedure),答非所问(应None/在Procedure_Plan)。
- vs BL: PL胜9/BL胜2(Type、Distant Met——BL更保守正确)/TIE11。
- 改进: ①Distant Met节点词(cervical/supraclav/level V/internal mammary)→regional不计distant hook ②FAITHFUL"疑似当确诊"门(possibly/suspected+待FNA→Type不写definite metastatic) ③plan字段biopsy/FNA清洗(→Procedure_Plan)。

### breast ROW10 (coral_idx=29, 新患初诊neoadj待启生育保存中) — 对齐✓
- **P1** Imaging_Plan: 漏"04/23 echocardiogram+EKG"(蒽环前baseline echo临床刚需);日期错(Mammi/PET note说04/29,PL写05/01)。
- **P1** Referral.Genetics="genetic testing sent": 已完成检测发送≠转诊到genetics clinic(错字段)。germline panel sent应入Genetic_Testing_Results。
- P2 Type HER2-丢失FISH 2.0边界细节;P2 Referral.follow up堆砌重复plan项。
- vs BL: PL胜11/BL胜2(Imaging_Plan漏echo、Genetic_Testing_Results)/TIE8。PL碾压(强论据):current_meds正确清空,BL把IVF促排药Cetrorelix/Menotropins当抗癌药(P0级BL错)。
- 改进: ①Imaging_Plan: A/P含echo/EKG/MUGA/TTE必收hook+日期忠实 ②区分germline检测(→Results"sent pending")vs转诊(→Referral) ③Referral禁塞已完成动作/plan内容。

