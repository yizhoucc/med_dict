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
- [进行中] breast 1-20 ✓ | pdac 1-5 ✓ | pdac 6-20 待
- 进度：25/40 | 下一步：派 subagent 审 pdac 6-20。PL行(pdac):R6 1247,R7 1413,R8 1559,R9 1711,R10 1869,R11 2033,R12 2185,R13 2340,R14 2510,R15 2668,R16 2823,R17 2990,R18 3127,R19 3273,R20 3428(end3584)。BL行(pdac):Rn=8+(n-1)*76 (R6 388,R7 464,R8 540,R9 616,R10 692,R11 768,R12 844,R13 920,R14 996,R15 1072,R16 1148,R17 1224,R18 1300,R19 1376,R20 1452)。pdac coral_idx=n-1。prompt=prompts/pdac/extraction.yaml+prompts/pdac/plan_extraction.yaml。subagent模板见上方pdac1-5。
- 模板见上方各 subagent prompt(三重任务:正确性/vs BL/改进; 铁律禁脚本); 主复核P0(亲读原文),P1抽验,P2信任subagent
- 注意pdac特有:碳水化合物癌(carcinomatosis)、CA19-9、FOLFIRINOX/gem-nab、surveillance、Whipple等
- 派 subagent 方式：general-purpose，每个审1 sample，给PL行范围+BL行范围+prompt路径+三重任务(正确性/vs BL/改进)。主Claude复核P0/P1。
- PL行范围(breast): R1 620,R2 760,R3 921,R4 1091,R5 1237,R6 1392,R7 1547,R8 1711,R9 1869,R10 2024,R11 2182,R12 2322,R13 2474,R14 2626,R15 2781,R16 2942,R17 3097,R18 3261,R19 3413,R20 3574(end3715)
- PL行范围(pdac): R1 445,R2 591,R3 761,R4 931,R5 1074,R6 1247,R7 1413,R8 1559,R9 1711,R10 1869,R11 2033,R12 2185,R13 2340,R14 2510,R15 2668,R16 2823,R17 2990,R18 3127,R19 3273,R20 3428(end3584)
- BL行范围(both, 每76行): Rn = 8+(n-1)*76 起，即 R1 8,R2 84,R3 160,R4 236,R5 312,R6 388,R7 464,R8 540,R9 616,R10 692,R11 768,R12 844,R13 920,R14 996,R15 1072,R16 1148,R17 1224,R18 1300,R19 1376,R20 1452
- prompt: breast=prompts/extraction.yaml+prompts/plan_extraction.yaml；pdac=prompts/pdac/extraction.yaml+prompts/pdac/plan_extraction.yaml

## ★ 汇总 tally（实时更新, 主复核后）
| 严重级 | 计数 | 涉及 sample |
|---|---|---|
| P0 | 2 | b4(Stage III幻觉), b17(current_meds="ac"编造现用化疗) |
| P1 | ~40 | breast 20样本累计(见各sample)。高危类:current_meds化疗方案幻觉(b12 tc/b17 ac)、否决药当现药(b14)、Type亚型幻觉(b20 IDC)、findings塞plan(b19)、Medication_Plan hook-bug追加非癌药(b19) |
| P2 | 多 | 各sample均有 |

### breast 20样本 vs BL 总览(主复核校准后)
逐sample PL整体均领先BL(PL胜6-14 vs BL胜0-4),但**没有一个sample做到字段级零失分**——每个都有1-4个字段被BL反超(多为plan字段错配/遗漏、current_meds时态、Type措辞)。即"逐sample PL赢"成立,但"字段级全方位碾压"尚未达到。BL的胜场几乎全来自:①plan字段PL放错/遗漏(FNA/echo/surgery/biopsy/radiotherapy) ②current_meds PL编造化疗(b12/b17,比BL倒灌非癌药更危险) ③Type/goals措辞。BL自身普遍犯:current_meds倒灌非癌药(几乎每个breast)、goals写疗法类型、schema缺second opinion/in-person/Metastasis子键、漏stage/grade/MammaPrint。

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
11. **current_meds时态/拒绝药**(高危)：把讨论/拒绝/未来方案当现用药(b12 tc被拒、b14 continue tamoxifen否决) → current_meds命中discussed/alternative/prefer-instead/concerned-about上下文清空 + Response="not yet on treatment"时强制空 + "continue X"须有current依据
12. **suspected-as-confirmed stage/Type**(b4,b9,b13,b15)：推断/疑似当确诊或武断 → suspected-stage门 + inferred标注 + FAITHFUL疑似门(已有POST-STAGE-SUSPECTED但仅管stage已IV的情形,需扩到Type字段+stage<IV的"now Stage III"推断)
13. **plan字段遗漏**：radiotherapy漏隐含"after radiation"(b13)、Procedure泛化丢术式(b13)、findings漏切缘阳性(b11) → 隐含放疗推断+术式全称保留+margin必收
14. **文本截断bug**(b11 findings "(t.")：排查生成max_tokens/字符串处理
15. **current_meds化疗方案幻觉**(高危P0类,b12 tc/b17 ac)：讨论/拒绝/未启动的化疗regimen名(AC/TC/AC-T/FOLFIRINOX)进current_meds → G4 FAITHFUL拦"明确矛盾"(note含"not receiving X"/"options include"/"prefer...instead"+无cycle/currently on→清空)
16. **Medication_Plan hook-bug**(b19)：我方"; also:"拼接逻辑把非癌药(doxycycline)/PRN镇痛/重复药(ondansetron=zofran)塞进plan → 修该hook:仅追加癌相关+去重
17. **completed-test当plan**(b17 brca/b18 mammaprint)+**germline pending漏**(b20)：已出结果→Results;pending→Plan"sent pending";已完成→不进Plan
18. **radiotherapy "not recommended"丢失**(b13,b18)：RadOnc不推荐放疗(due to X)写成None → 写"Radiation NOT recommended (reason)"
19. **Patient type模板表头误导**(b20)：表头"New patient Visit"vs正文"return/follow up"冲突→以正文为准
20. **Type亚型幻觉**(b20)：病理仅"lobular differentiation"却断言IDC → 仅pathology明写才标IDC/ILC
21. **Procedure_Plan遗漏**(b16 FNA/b20 surgery+port)：A/P含surgery/port/pending biopsy/FNA→必收
22. ⚠️**supportive_meds hook回归**(高优先,pdac2/4/5)：commit f0699e8c"supportive faithfulness check(drop not-in-note)"**误删note中确有的Creon/Xarelto/pantoprazole**(A/P"Continue Creon"却被drop)。PDAC胰酶遗漏严重 → 修hook:A/P"continue X"的supportive药必须保留;勿因匹配不到outpatient list原样就drop
23. **current_meds化疗doublet漏成分**(pdac3 nal-IRI、b7 abraxane)：多药方案只取一个 → 从summary/therapy回填
24. **Response stable vs decline**(pdac1/3)：只报单部位/旧CT stable,漏A/P医生整体"progression/decline"判断 → 医生整体判断优先
25. **recent_changes 未来当已发生/漏真实事件**(pdac1/4)：chemo break(未来)当recent;漏完成化疗→surveillance、本次ERCP/stent → 区分已发生事件
26. **Clinical_Findings堆lab/裸"no met"/造负性**(pdac2造no-weight-loss、pdac3/5堆lab、pdac4裸"No evidence of metastasis"误导) → findings去lab+负性结论须note支撑+裸no-met加上下文
27. **non-secretor CA19-9状态漏**(pdac4)：prompt明确应收 → hook
28. **locally advanced漏stage**(pdac1) → "locally advanced"→Stage III兜底(同breast的"early stage"→取)

### ⚠️ breast 20/40 阶段结论
- 2个真P0(b4 stage幻觉/b17 current_meds化疗幻觉)均为我上轮四维审查漏掉的(四维只看诊断字段,没看current_meds/全字段)。
- 21类改进模式已饱和(后续pdac预计主要印证+少量pdac特异)。
- **核心修复优先级**(P0/安全优先): A)current_meds化疗方案幻觉门(G4) B)stage数字须note依据否则Not staged C)Medication_Plan hook-bug D)plan字段路由(echo/FNA/biopsy/port/TTE各归位)+时态(已完成不当未来) E)Type疑似/亚型不臆断 F)plan字段必收清单(surgery/port/baseline echo/诊断biopsy/radiotherapy-not-recommended/germline pending) G)Referral去重+只outgoing。
- 这是新一轮 hook+prompt 迭代的输入。建议:先修A-C(P0+hook-bug)→小样本验证→再批量修D-G→重跑→重审。

---

## pdac 逐 sample (主复核P0/抽验P1)

### pdac ROW1 (coral_idx=0, locally advanced,肺新结节suspicious) — 对齐✓
- **P1** Distant Met "Not sure": 可写客观——CT chest 5新肺结节"suspicious for mets"+CT腹盆"no met"。应分述非笼统Not sure。
- **P1** recent_changes "chemo break"(未来plan当已发生),漏真实事件(5th cycle因胆道梗阻中断+换stent+抗生素)。
- **P1** Response漏核心:只写"胰腺mass stable",漏A/P"possible progression"+新肺结节(误导偏稳定)。
- P2 Stage空("locally advanced"应=Stage III)。
- vs BL: PL胜6/BL胜2(Stage占位、Distant Met"No"更可用)/TIE~11。PL碾压:Lab(CA19-9具体)、current_meds(BL错"See intake")。
- 改进:①suspicious met分述非Not sure ②locally advanced→Stage III兜底 ③recent_changes区分已发生事件vs未来break ④Response纳入"possible progression"。

### pdac ROW2 (coral_idx=1, s/p Whipple now liver met) — 对齐✓
- **P1** Clinical_Findings编造"no significant weight loss",与note"+Thin"/"weight loss"矛盾。
- **P1** supportive_meds漏**Creon**(胰酶,note"Continue Creon",BL抓到)。
- P1 Lab混用不同日期面板。
- vs BL: PL胜14/BL胜1(supportive_meds漏Creon)/TIE6。PL碾压:current_meds(BL倒灌14种非癌药)、findings、response、Stage IV。
- 改进:①PDAC supportive强制查Creon/pancrelipase ②findings PE阴性结论须note支撑(禁造"no weight loss") ③Lab值带日期。

### pdac ROW3 (coral_idx=2, metastatic liver/spleen/peritoneal) — 对齐✓
- **P1** Distant Met/Metastasis漏**脾**(note"hypodense splenic lesions",同bug10)。
- **P1** current_meds漏 nal-IRI/Onivyde(只写"5-FU/LV",doublet漏irinotecan;summary/therapy却写全)。
- **P1** Response"stable disease"(引旧02/16 CT)vs医生整体"continued clinical/symptomatic decline"(CA19-9 15→142升、GOO、贫血)。
- **P1** Procedure_Plan=RT句残片("which would be primarily for palliative...")放错字段。
- vs BL: PL胜14/BL胜3(Distant Met脾、current_meds完整、Response decline)/TIE5。
- 改进:①met站点并集补全(splenic/hepatic/peritoneal) ②current_meds从summary/therapy回填多药方案 ③Response"医生整体判断(decline/progression)优先"于单次旧CT ④Procedure_Plan残片检测→"No procedures planned"。

### pdac ROW4 (coral_idx=3, locally advanced→liver met, surveillance) — 对齐✓
- **P1** recent_changes空:漏"完成6cycle gem/Abraxane→surveillance"+本周EGD/ERCP换stent(BL抓到)。
- **P1** Clinical_Findings裸"No evidence of metastasis"(与Stage IV矛盾,误导)+漏EGD/ERCP(tumor ingrowth/stone/clot/new stent)。
- P2 Genetic_Testing_Results漏non-secretor("does not express CA-19-9",prompt明确应收)。
- vs BL: PL胜9/BL胜4(recent_changes、findings EGD、supportive pantoprazole)/TIE9。PL碾压:Stage IV、met schema、in-person(BL漏Televisit)。
- 改进:①recent_changes:完成化疗→surveillance+本次procedure(stent)算change ②findings剥裸"No evidence of metastasis"加上下文+收EGD/ERCP ③non-secretor状态hook。

### pdac ROW5 (coral_idx=4, Stage IV oligomet abdominal wall) — 对齐✓ ⚠️hook回归
- **P1 [hook回归]** supportive_meds漏**CREON+XARELTO**(note明确"Continue Creon"/"Continue xarelto")。**subagent定位=commit f0699e8c "supportive-med faithfulness check(drop not-in-note)"误删在note中的药**。BL反而抓到。PDAC胰酶遗漏严重。
- P2 Clinical_Findings堆砌lab值(与Lab_Results重复)。P2 Therapy_plan attribution错配(指向GI bleed句)。
- vs BL: PL胜14/BL胜2(supportive_meds Creon/Xarelto、findings简洁)/TIE6。PL碾压:Distant Met双键、Medication_Plan剂量、radiotherapy、Advance_care。
- 改进:①**查修f0699e8c supportive faithfulness hook:A/P"continue X"的supportive药必须保留,勿误删** ②PDAC supportive强制清单(Creon/抗凝/止吐/止泻/癌痛) ③findings去重lab。

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

### breast ROW11 (coral_idx=30, DCIS pTisNx风险降低) — 对齐✓
- **P1** Clinical_Findings: 漏最关键的"posterior margin POSITIVE(tumor at ink)",且文本在切缘处**截断**("margins are negative (t.")——疑似生成截断/字符串处理bug。
- P2 findings重复诊断信息;P2 Type把"PR pending"静默省成not tested;P2 second opinion attribution错配。
- vs BL: PL胜6/BL胜1(findings:BL更干净无截断)/TIE14。PL碾压:Patient type(BL误Follow up)、current_meds(BL倒灌非癌药)、HER2 not tested。
- 改进: ①findings优先保留margin status(positive/close必标)+查截断bug ②PR pending显式写 ③去重诊断信息。

### breast ROW12 (coral_idx=31, Clinical Stage II PET无mets) — 对齐✓ ⚠️
- **P1** current_meds "tc": 三重错——TC是**被患者拒绝**的备选方案(选了AC/T)、是PLANNED非current、患者尚未开始治疗。把"讨论/拒绝的方案"当现用药(危险)。
- **P1** Lab_Results "No labs in note": note有完整CBC+代谢(10/22/2016,距就诊~4.5月<6月)应报告却漏。
- vs BL: PL胜8/BL胜3(Lab_Results、supportive_meds cold caps、current_meds——BL的tc错更轻)/TIE10。
- 改进: ①current_meds命中"discussed/alternative/prefers...instead"上下文药名→清空+Response="not yet on treatment"时强制current_meds空(治疗时序交叉校验stage) ②Lab "No labs"时回扫note表格,≤6月强制填回+标日期。

### breast ROW13 (coral_idx=32, node+;脑falx灶meningioma良性) — 对齐✓
- **P1** Stage "now Stage III": 从node+推断(attribution自曝inferred),"now"暗示升级误导。应"Stage III (cN+, inferred; not explicitly staged)"。
- **P1** Clinical_Findings: 列了falx灶"or dural metastasis"鉴别但丢失医生结论(放射科"suspicion LOW"/favored meningioma)→下游过度警示风险。
- vs BL: PL胜9/BL胜2(radiotherapy_plan PL漏"after radiation"暗示放疗、Procedure_Plan PL泛化"surgery"丢ALND/术式)/TIE12(brain MRI今日已做,双方都误当计划)。
- 改进: ①radiotherapy隐含短语("after radiation"/"hormonal after RT")→推断放疗在计划 ②Procedure_Plan保留术式全称(partial mastectomy/ALND/SLNB)不泛化 ③推断stage标注+去"now" ④含糊病灶必带医生结论判断。

### breast ROW14 (coral_idx=33, 新诊ER+/PR+/HER2-) — 对齐✓ ⚠️
- **P1** Therapy_plan "Continue tamoxifen": 患者新诊断从未用tamoxifen、医生明确**反对**用(患者选AI)。"Continue"=时态+事实双幻觉。
- **P1** Medication_Plan "; also: tamoxifen": 被否决的药当方案。
- P2 Type HER2 equivocal(IHC2)+FISH non-amplified未按表收敛为HER2-;P2 Lab陈旧(>1年syphilis/HIV)未滤。
- vs BL: PL胜8/BL胜2(Medication_Plan、Therapy_plan——BL没抄tamoxifen更干净)/TIE11。PL碾压:Type/findings/current_meds(BL倒灌kava/zolpidem等)/recent_changes/Referral genetics。
- 改进: ①药物否定语境("concerned about using X"/"prefer X over Y"/"advised against")→移出plan ②"continue [drug]"须Current_Medications或"currently on"依据否则降级start/discussed ③HER2 IHC2+FISH non-amp→收敛HER2-。

### breast ROW15 (coral_idx=34, presumptive MBC,supraclav FNA证实,cervical待FNA) — 对齐✓
- **P1** Stage "Metastatic (Stage IV)": 应"Suspected Stage IV (pending confirmation)"(note"if we confirm...de novo MBC"+"requires additional work-up")。
- **P1** Distant Metastasis含"axillary"(=regional非distant,违schema);真正distant的cervical已正确标Suspected。
- P2 Type受体块在findings未进Type_of_Cancer;P2 goals palliative(疑似未确诊可条件化)。
- vs BL: **PL胜13/BL胜0/TIE9**(BL: current_meds倒灌全部非癌药=P0级、goals"curative"方向错、Stage"Not specified"漏)。PL本sample全胜但自身2个P1可修。
- 改进: ①Distant Met清洗regional节点词(axillary/supraclav/infraclav/sentinel/internal mammary),仅留真distant(cervical/肝肺骨脑/对侧);若只剩regional→"No confirmed distant(regional only)" ②suspected-stage门:A/P含"if we confirm/presumptively/pending/requires workup"+stage=IV→"Suspected Stage IV (pending confirmation)"。

### breast ROW16 (coral_idx=35, Clinical stage III ILC) — 对齐✓
- **P1** Imaging_Plan "CT scan": PET/CT(07/06)和超声均已完成,A/P无未来影像→应"No imaging planned"(已完成当未来)。
- **P1** Procedure_Plan: 漏pending左腋窝FNA(应收)。
- P2 findings重复受体/分期。
- vs BL: PL胜7/BL胜1(Lab_Plan:BL抓到pending FNA)/TIE13。PL碾压:current_meds(BL倒灌降压药HCTZ/Exforge)、findings全、schema全。
- 改进: ①imaging已完成(performed/accomplished)清空hook ②Procedure_Plan收pending FNA/biopsy ③findings去重受体。

### breast ROW17 (coral_idx=36, Stage IIb T2N1M0 s/p lumpectomy推荐adjuvant) — 对齐✓ ⚠️P0
- **P0 [主复核确认]** current_meds "ac": note imaging_plan归因明写"**not receiving AC as planned so far**",患者选TC未启动,current_meds自身归因句讲的是TC讨论——纯编造现用化疗(且是被放弃方案)。同b12模式,确认P0。
- **P1** goals: 丢失note显式"recommend adjuvant chemotherapy"(goals_description="Not explicitly stated"遗漏)。
- **P1** Genetic_Testing_Plan "brca": BRCA已做完阴性(属results非plan)。
- **P1** Lab_Plan "Baseline echo": echo是影像非化验(真实lab plan="Path review Ki67")。
- **P1** Referral.Specialty空: 漏"chemo teaching/port placement/plastic surgeon"。
- vs BL: PL胜9/BL胜4(current_meds、goals adjuvant、Lab_Plan、Referral)/TIE9。
- 改进: ①current_meds: 化疗方案名(AC/TC/AC-T)无"continue/currently on/cycle N"证据→清空(G4 FAITHFUL应拦"明确矛盾")+"not receiving X"检测 ②goals显式adjuvant→写入+description回填 ③已完成检测(BRCA done)→Results不进Plan ④Lab_Plan剔echo ⑤Referral收chemo teaching/port/plastic。

### breast ROW18 (coral_idx=37, cT2NX ATM携带者) — 对齐✓
- **P1** Genetic_Testing_Plan "mammaprint": MammaPrint已出结果(High Risk -0.622,已在Results)非未来计划。
- **P1** radiotherapy_plan "None": 漏"RadOnc因ATM突变不推荐放疗"(重要临床决策,BL抓到"Not recommended due to ATM")。
- P2 PR跨字段不一致(Type用A/P"PR-",Genetic_Results用path"PR+<5%")。
- vs BL: PL胜14/BL胜2(radiotherapy、Genetic_Plan)/TIE6。PL碾压:second opinion(BL漏)、grade、节点依赖化疗逻辑、findings。
- 改进: ①已出结果的genomic test(MammaPrint/Oncotype/BRCA)→剔出Plan置"None planned" ②radiotherapy "not recommended/not offering (due to X)"→写"Radiation NOT recommended (reason)"非None ③PR跨字段一致化(A/P优先+加注)。

### breast ROW19 (coral_idx=38, s/p mastectomy NED随访) — 对齐✓ ⚠️hook-bug
- **P1** current_meds: 漏zoladex/goserelin(卵巢抑制作内分泌治疗,note"cont zoladex locally monthly"),只写exemestane。
- **P1** Clinical_Findings: 几乎全是治疗PLAN(continue exemestane/DEXA/estradiol/BSO/q6mo)塞进findings,非客观查体/影像。
- **P1 [hook-bug]** Medication_Plan尾部"; also: ondansetron, zofran, doxycycline, acetaminophen, hydrocodone": **我们自己的supportive/med hook误追加非癌药(doxycycline抗生素)+重复(ondansetron=zofran)+PRN镇痛**。需查该hook。
- P2 Genetic_Testing_Results含ER/PR/Ki67 IHC(应只留FISH/MammaPrint)。
- vs BL: PL胜6/BL胜1(Clinical_Findings:BL方向对PL全塞plan)/TIE14。PL碾压:Stage(BL漏"Clinical stage 2-3")、MammaPrint(BL漏)、[REDACTED]处理。
- 改进: ①current_meds收zoladex/goserelin/leuprolide(endocrine上下文) ②findings剔计划动词句(continue/recommend/needs/consider/check) ③**修Medication_Plan "; also:"拼接hook:过滤抗生素/PRN镇痛+同药去重** ④Genetic_Results剔IHC受体%。

### breast ROW20 (coral_idx=39, 双侧HER2+,liver/lung nodules疑似) — 对齐✓
- **P1** Patient type "New patient": note正文"presents for a return visit"/"returns for follow up"(模板表头"New patient Visit"误导)→应Follow up。
- **P1** Type "invasive ductal carcinoma": note病理只说"invasive cancer with some lobular differentiation"+ecadherin+,无ductal→编造IDC亚型。
- **P1** Procedure_Plan "No procedures planned": 漏手术("surgery will be determined by response")+port-a-cath。
- **P1** Genetic_Testing_Plan/Results: 漏"Germ line panel pending"(BL抓到)。
- P2 Stage空(note"early stage"应取);P2 Lab_Plan混入port(procedure)。
- vs BL: PL胜11/BL胜3(Procedure_Plan手术、Genetic germline pending、Stage)/TIE7。PL碾压:Distant Met(Suspected vs BL答非所问)、goals、方案完整度。
- 改进: ①Patient type表头vs正文冲突→以正文return/follow up为准 ②Type病理仅"lobular differentiation"禁断言IDC ③Procedure_Plan: A/P含surgery/port→必收 ④germline panel pending→Genetic_Plan"sent pending" ⑤Stage取"early stage" ⑥Lab_Plan剔port。

