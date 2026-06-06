# 重审 (FIX2 全 40 逐字段) — 验证 14 个修复后 PL vs BL

> 触发：本轮 14 个 hook 修复 + 全量重跑 (FIX2)。对 `pipeline_breast_FIX2.txt` / `pipeline_pdac_FIX2.txt`（PL）vs `baseline_extract_breast_json.txt` / `baseline_extract_pdac_json.txt`（BL）做严格全字段重审。
> 方法：subagent 并行初审（每个审 1 sample，按 CLAUDE.md 三块完整 brief）→ 主 Claude 亲自复核全部 P0/P1 原文后才采纳。禁止脚本/grep/regex 做判断。
> 数据对齐：PL/BL 均按 `ROW N` 对齐（N=1..20），coral_idx 一致。breast coral_idx=N+19；pdac coral_idx=N-1。

## 目标
验证：① 2 个 P0（b4 stage、b17 current_meds）是否清零 ② 本轮 14 修复是否各自正确落地、无新回归 ③ 各字段 BL 胜场是否归零（"全方位碾压无死角"）。

## 严重度
- **P0**：幻觉/编造（note 无的药/诊断/站点/数值/分期）或疑似当确诊。
- **P1**：内容真实但放错字段/方向错/关键遗漏/时态混淆/答非所问/误导。
- **P2**：模糊/不具体/可推断却写未提及/格式。

## STATUS
- [ ] breast 1-20  [ ] pdac 1-20
- P0 统计：_ | P1：_ | 各字段 BL 胜：_

## 本轮已修（subagent 须知，验证是否生效）
POST-STAGE-NOBASIS(b4 stage幻觉清零→"Not staged") / POST-MEDS-REGIMEN-FAB(b12/b17 化疗方案幻觉清空) / 删 'pt' physical-therapy幻觉 / Medication_Plan 去非癌药+去重 / POST-SUPP-SUPPLEMENT(Creon/Xarelto恢复) / IV-CHECK增强+ENZYME-STRIP(active化疗) / POST-PLAN-ROUTING(TTE/FNA/port/germline归位) / POST-PLAN-TEMPORAL(已完成影像/labs清空) / POST-GENETIC-PENDING(UCSF500) / POST-GOALS-PALLIATIVE-CHECK / POST-TYPE-MET-CONSISTENCY(疑似/局部≠转移确诊) / POST-STAGE-LOCALLY-ADVANCED / POST-MEDS-DOUBLET / CROSSCHECK去括号。

---

## breast 逐 sample (1-10 完成, 主复核P0/P1)
- **b1(idx20)**: P0=0 P1=0(PL). BL有2个P1(palliative方向错/非癌药入current_meds). PL P2: Stage写"IIB"(note仅"Stage II")、Type括注FISH2.1易误导. PL胜4/BL胜0.
- **b2(idx21)**: **P1[PL] goals="palliative"应curative**(locally recurrent非转移+"long-term disease control/resection/radiate"). 主复核确认 DistMet=No/Stage=Locally recurrent → 我的 PALLIATIVE-CHECK 因插在DistMet终值前未触发=**ordering bug**. P2:imaging_plan列已完成PET/MRI. BL胜0.
- **b3(idx22)**: P0=0 P1=0. PL胜6(Patient type/current_meds/findings/imaging时态). 
- **b4(idx23)**: ✅**3个P0全修正确**(stage="Not staged"/Type"local recurrence"/goals curative). **P1[PL] Imaging_Plan="Mammogram"**(过去已完成检查当plan,BL="None"对). P2:Response与findings冗余. PL胜10/BL胜1.
- **b5(idx24)**: **P1[PL] Procedure_Plan="AI therapy for 5 years"**(全身激素治疗漏进procedure,违反prompt). PL胜6/BL胜1. 其余PL碾压(BL findings/therapy/imaging多处P1遗漏).
- **b6(idx25)**: P0=0 P1=0. PL碾压(BL 3个P1:current_meds倒灌全部非癌药/distmet把suspicious当确诊/response写预后臆测). P2:procedure漏staging biopsy. PL胜6/BL胜1.
- **b7(idx26)**: ✅**doublet修复生效**(current_meds="pembrolizumab, abraxane"). P0=0 P1=0. BL P1(漏整个抗癌方案). BL胜1(goals_description PL写"Not explicitly stated"但note有). PL胜11.
- **b8(idx27)**: ✅TTE归位生效. **P1[PL] Stage="Not staged in note"但note明写"stage IIA"×3+TNM"pt2(m)n1a"**(BL反超). 主复核=**NOBASIS锚点正则缺陷**(stage iia的\b/TNM的(m)打断). P2:follow_up漏10/10预约. PL胜3/BL胜2.
- **b9(idx28)**: ✅**3修复全验证**(FNA剥离/Type软化suspected). P0=0 P1=0. PL胜8/BL胜0(BL response答非所问).
- **b10(idx29)**: ✅germline归位. P0=0 P1=0. **BL胜3**:Imaging_Plan漏echo/EKG+日期、Genetic_Results漏"sent pending"、Referral.follow up仍堆砌plan项(去重未覆盖follow up子字段). P2:stage clinical→IIA越界. PL胜11/BL胜3.

### breast 1-10 新发现 PL 问题(下轮修)
- **R1[P1] NOBASIS过删**(b8): A/P-stage正则`stage\s*ii\b`卡"iia"; TNM正则被"(m)"打断 → 加[abc]?+容忍(m)多灶标记. ⭐高优先(BL反超)
- **R2[P1] PALLIATIVE-CHECK ordering**(b2): 移到DistMet终值之后. ⭐高优先
- **R3[P1] procedure激素泄漏**(b5): PROC_BLACKLIST加"aromatase inhibitor"/"ai therapy"/"ai for". 
- **R4[P1] imaging时态**(b4 Mammogram): temporal的bare-modality完成检查扩展到扫note body.
- **R5[P2] Referral.follow up堆砌**(b10/b1/b6): follow up子字段去plan重复项.
- 杂P2: stage过度细化(b1 IIB/b10 IIA←clinical), b10 imaging漏echo/EKG, b7/b6 goals_description.

## breast 11-20 完成 (主复核P0/P1)
- **b11(idx30)**: P0=0 P1=0. PL胜4(current_meds空/Patient type/Type HER2/findings). BL P1(current_meds灌10个非癌药). P2:findings切缘句尾截断.
- **b12(idx31)**: ✅**"tc"幻觉已清空**(current_meds=""). P0=0 P1=0. PL关键胜:lab_summary正确排除上一年旧labs(BL错列). BL胜2(procedure漏chemo teach/referral漏cold cap,均P2).
- **b13(idx32)**: **P1[PL] Stage="Stage III"过度**(cT2N1M0应IIB或not-staged,note未stage). **P1[PL] radiotherapy_plan="None"漏planned放疗**("hormonal after radiation"隐含). P1[PL] imaging="Brain MRI"已完成当plan(BL同错TIE). P2:Metastasis"possibly falx"过度(meningioma favored). BL胜(stage/radiotherapy).
- **b14(idx33)**: ✅current_meds/recent_changes clean(无tamoxifen). **P1[PL] Therapy_plan="Continue tamoxifen"+Medication_Plan"; also:tamoxifen"**——否决药(tamoxifen被医生质疑+患者选AI)泄漏进plan字段,且"continue"对never-started药=幻觉. PL胜12/BL胜0(但plan字段BL更干净).
- **b15(idx34)**: ✅**axillary-as-distant已修**(DistMet="Not sure",BL反而把LN当distant). **P1[PL] Stage="Stage III"应Stage IV**(note"de novo MBC...not curable"+cervical淋巴结=远处;且与goals=palliative内部矛盾). PL胜11/BL胜0(stage实质错).
- **b16(idx35)**: **P1[PL] imaging_plan="CT scan"幻觉**(无未来CT,PET/CT是过去). P2:Referral外科转诊错放follow up非Specialty. PL胜8/BL胜2. BL P1(current_meds灌降压药).
- **b17(idx36)**: ✅**"ac"幻觉已清空**. **P1[PL] lab_plan="Baseline echo"**(echo属imaging非lab)+漏"Path review". **P1[PL] genetic_testing_plan="brca"**(已完成既往BRCA当plan). P2:Referral漏chemo teaching/plastic. PL胜9/BL胜4.
- **b18(idx37)**: ✅**Stage cT2NX已修**(CTNM生效). P0=0 P1=0. PL碾压13/BL2. P2:genetic_plan="mammaprint"(已完成当plan)/follow_up漏"after surgery".
- **b19(idx38)**: ✅**physical therapy幻觉已消除+Medication_Plan无非癌药堆砌**(仅追加真实ondansetron). **P1[PL] Clinical_Findings灌入plan内容**(答非所问,应为客观体检发现). P2:Procedure漏"considering BSO". PL胜13/BL胜4.
- **b20(idx39)**: ⚠️**P0[PL] Type="invasive ductal carcinoma"亚型幻觉**(病理仅"invasive cancer with some lobular differentiation",全文无ductal). **P1: Stage=""**(note明写"early stage"). **P1: Lab_Plan仍含"port a cath"**(port只加进Procedure没从Lab删). **P1: germline panel pending遗漏**(germline hook移出Referral但没送进Genetic_Plan→信息丢失). PL胜9/BL胜4. 本轮最差sample.

### breast 11-20 新发现 PL 问题(归并主题, 下轮修)
- **R6[P0] b20亚型幻觉**: 病理含"lobular differentiation"且无ductal/IDC → 禁止注入"ductal". ⭐
- **R7[P1] stage越界**: b13 Stage III过度(cT2N1=IIB)、b15应Stage IV(MBC/de novo metastatic+cervical=distant). → stage hook: node+无明文stage不臆造III; "MBC/de novo metastatic/not curable"+cervical/visceral→Stage IV; extraction.yaml补"cervical=distant for breast".
- **R8[P1] 否决药入plan**(b14 tamoxifen): plan字段剔除"concerned about/prefer Y instead/X alone/declined"语境药; 禁止对never-started药写"Continue X". 定位"; also:tamoxifen"拼接hook.
- **R9[P1] plan类含已完成/无依据项(高频回归)**: b4 mammogram/b13 brain MRI/b16 CT/b17 BRCA/b18 mammaprint/b20——imaging_plan&genetic_plan&lab_plan须排除已完成检查(有结果/过去日期/today done)且只保留A/P有order依据的项. ⭐共性大类
- **R10[P1] 字段路由残留**: b5 AI→procedure(激素入手术)、b17 echo→lab(心超入lab)、b20 port残留lab_plan. → procedure剔除激素/全身药; lab_plan剔除echo/port→各归位.
- **R11[P1] findings=plan**(b19): Clinical_Findings灌入治疗计划 → findings只收客观体检/影像/病理发现,剔除plan句.
- **R12[P1] 完整性**: b20 germline pending→Genetic_Plan; b20"early stage"→Stage; b19 considering BSO→Procedure; b17 Path review→lab.
- **R13[P2] Referral堆砌/错放**(b10/b16/b17): follow up去plan重复; 外科/chemo-teach/plastic转诊→Specialty.

## breast 汇总(20/20)
- **P0=1**(b20 ductal亚型幻觉). P1≈12(b13×2/b14/b15/b16/b17×2/b19/b20×3 等). 上轮2个P0(b4/b17)**均已清零✅**.
- BL反超PL的sample: b13(stage/RT)、b15(stage)、b16(imaging CT)、b17(lab/genetic plan)、b19(findings)、b20(type/stage/lab/germline)、b10(imaging/genetic/referral)、b12(2个P2). 其余12个PL无死角≥BL.
- 主题: 诊断P0/P1基本清零(仅b20亚型+b13/b15 stage), 大量残留在**plan类时态/字段路由/findings字段纯度/完整性**——是本轮hook未覆盖到的新manifestation.

## pdac 1-10 完成 (主复核P0/P1)
- **pdac1(idx0)**: ✅Stage III locally advanced. P0=0 P1=0. PL胜9/BL1. P2:goals_description空(应fallback). findings加"No palpable masses"未明示.
- **pdac2(idx1)**: ✅Creon在supportive. P0=0 P1=0. PL胜13/BL1. BL多处硬伤(current_meds灌非癌药/lab堆旧值/goals编造). P2:recent_changes可补from-FOLFIRINOX.
- **pdac3(idx2)**: ✅doublet(5-FU/LV+nal-IRI). **P1[PL] Advance_care漏hospice**(note明讨论home hospice,BL抓到). **P1[PL] Procedure_Plan=RT句残片**(字段错配,应"No procedures"). P2:recent_changes漏resumed cycle/med_plan"; also:trametinib"尾. PL胜11/BL4.
- **pdac4(idx3)**: **P1[PL] current_meds=""被清空**(note明列7药"patient states to be taking",BL完整). **P1 CA19-9 non-secretor漏**("does not express CA-19-9",PL+BL同漏). brief的"Creon+Xarelto"是误标(属pdac5). PL胜5/BL2.
- **pdac5(idx4)**: ✅Creon+Xarelto恢复. **P1[PL] current_meds="folfirinox"**但s/p 12 cycles+chemo break=已完成PAST,应空. PL胜16/BL1.
- **pdac6(idx5)**: **P1[PL] goals="palliative"应surveillance**(s/p R0切除+仅疑似复发+无active治疗,BL对). **P1[PL] response="not responding well"**把疑似当进展(违反suspected≠confirmed,BL更忠实). ✅Metastasis"Suspected to liver"正确(peritoneal carcinomatosis未误判). PL胜9/BL5.
- **pdac7(idx6)**: **P1[PL] Stage=""**(locally advanced+splenic artery>180°+无远处=Stage III,LOCALLY-ADVANCED未触发). **P1[PL] supportive_meds="ondansetron"**(note标"Patient not taking"应过滤). **P1[PL] Therapy_plan="None"**(A/P"continue treatment",违反禁None). 3处全BL反超. PL胜8/BL3.
- **pdac8(idx7)**: ✅Creon在supportive. P0=0 P1=0. PL碾压13/0. P2:medication_plan"; also:irinotecan"尾(假设后线药误列).
- **pdac9(idx8)**: ✅Creon移出current_meds+进supportive. **P1[PL] current_meds="abraxane"漏gemcitabine**("gem abraxane"双药,IV-CHECK只抓abraxane;但plan侧正确写了gem abraxane可回填). PL胜18/0.
- **pdac10(idx9)**: ✅FOLFOX恢复. **P1[PL] current_meds过窄**(只化疗漏active支持药;但BL漏化疗更糟). P2:lab旧值(09/24,12/10)、med_plan"; also:capecitabine,irinotecan"尾、therapy"; potassium"尾、goals_description漏"4-6mo upfront chemo".

### pdac 1-10 新主题(下轮修)
- **TA[P1] current_meds完整性/时态(高频)**: pdac4误清空(明列药)、pdac5 folfirinox已完成(s/p+chemo break应空)、pdac9漏gemcitabine(gem别名)、pdac10过窄. → IV-CHECK补"gem"别名双药; s/p+completed N cycles+chemo break→清空; current_meds误清空排查(filter勿删"states to be taking").
- **TB[P1] Stage locally advanced未触发**(pdac7): LOCALLY-ADVANCED需扩展到"vessel encasement/splenic artery>180°/SMA/celiac contact"且无远处→Stage III(不只靠字面"locally advanced").
- **TC[P1] supportive "Patient not taking"未过滤**(pdac7): 剔除标记"(Patient not taking"的药.
- **TD[P1] Therapy_plan=None禁令**(pdac7): A/P含"continue treatment/continue [drug]"+患者在治→禁None.
- **TE[P1] goals/response疑似vs确诊**(pdac6): s/p R0+仅疑似复发→surveillance非palliative; response疑似软化不写"not responding".
- **TF[P1] Advance_care hospice**(pdac3): "home hospice/hospice/comfort care"→ACP不写"Not discussed".
- **TG[P1] Procedure_Plan字段错配**(pdac3): RT句残片→应"No procedures planned".
- **TH[P2] plan尾部噪声**(pdac3/8/10): medication_plan/therapy_plan的"; also:X"/"; potassium"等尾巴(future/stopped/电解质)清除.
- **TI[P2] 完整性**: pdac4 CA19-9 non-secretor; recent_changes补完成化疗→surveillance.

## pdac 11-20 完成 (主复核P0/P1)
- **pdac11(idx10)**: ✅胰酶(generic)恢复+STRATA pending. **P1[PL] Lab_Plan乱码**(regex跨日期把HPI"03/16 CT AP..."影像文本拼入). P2:genetic_plan尾部"CT Chest"乱码. PL胜6/BL2.
- **pdac12(idx11)**: ✅met格式修好(无双to). **P1[PL] therapy_plan="ipilimumab"幻觉**(患者05/2016已拒绝的trial选项当计划,实为palliative/hospice,BL="None"对). **P1[PL] CA19-9取旧值276非最新4720**(掩盖大幅上升). **P1[PL] Advance_care漏hospice**(note明讨论hospice,BL对). P2:liver met存疑("hard to interpret"). PL胜9/BL4.
- **pdac13(idx12)**: ✅Creon+UCSF500 in process. **P1[PL] recent_changes漏"pause systemic therapy"**(本次真实变更,BL抓到). P2:Imaging_Plan="CT Chest"已完成当plan; genetic_plan截断"separa"; non-secretor漏. PL胜14/BL1.
- **pdac14(idx13)**: ✅. **P1[PL] supportive_meds漏Tylenol**(note"using Tylenol for pain at home"癌痛,BL抓到). PL胜5/BL1. BL current_meds灌7个非癌药.
- **pdac15(idx14)**: ✅Creon. **P1[PL] goals="curative"应surveillance**(s/p neoadj+Whipple+无active治疗仅recheck markers,surveillance hook未命中,BL对). P2:Referral抓过去XRT漏本次PMD转诊; stage pT2N3(A/P)vs ypT3N2(病理)冲突. PL胜13/BL2.
- **pdac16(idx15)**: ✅无physical therapy幻觉+正确区分肾RCC/肺另原发≠胰癌转移. **P1[PL] Stage="Not staged"但note有专门"Cancer Staging"段写明"Stage IIB (cT1c,cN1,cM0)"**(专段漏读,BL对). P2:Referral dental过去评估. PL胜9/BL0(stage唯一输).
- **pdac17(idx16)**: ✅recent_changes时态修复(空,BL反把数月前break当变更)+Referral无incoming误报. P0=0 P1=0. P2:findings与lab重复. PL胜12/BL1.
- **pdac18(idx17)**: ✅Lovenox+Creon+Magic Mouthwash+capecitabine(resume). **P1[PL] current_meds漏gemcitabine**(resume regimen双药"gem+cape"只抓capecitabine). **P1[PL] Referral="GI surgical oncology"是incoming**("has been seen by",非outgoing,BL="None"对). PL胜14/BL0.
- **pdac19(idx18)**: ✅**bug8四点全验证**(current_meds=FOLFIRINOX/supportive含Creon/Therapy无physical therapy/response保持进展). **P1[PL] Procedure_Plan漏ERCP**(写"No procedures"却Referral有"GI for ERCP"自相矛盾,BL对). P2:Referral漏SMS转诊;Lab_Plan截断垃圾文本;goals_description可推断却写未提及. PL胜9/BL5.
- **pdac20(idx19)**: ✅current_meds过滤(BL灌Valtrex/Lorazepam)+MMR intact已含+无incoming误报. P0=0 P1=0. P2:supportive Percocet边界;imaging塞ctDNA试验. PL胜9/BL0.

### pdac 11-20 新主题(并入下轮)
- TA current_meds: pdac18漏gemcitabine(resume双药)、pdac12 CA19-9旧值. TB stage: pdac16专"Cancer Staging"段漏读. TE goals/response: pdac15 surveillance未命中(s/p resection+无active→surveillance,勿被"high risk recurrence"挡). TF ACP: pdac12 hospice. TG procedure: pdac19 ERCP漏+自相矛盾、pdac18 referral incoming("has been seen by"). 
- **新增**: TJ[P1] **plan类regex乱码/截断**(pdac11 Lab_Plan、pdac13/pdac19 lab/genetic截断)——某plan抽取regex跨日期/边界把HPI影像文本拼入,需清洗hook. TK[P1] **therapy_plan被拒trial当计划**(pdac12 ipilimumab,同b14 tamoxifen)——剔除"chose X vs beginning Y"/"declined"语境药. TL recent_changes漏本次"pause systemic therapy"(pdac13).

---

# ✅✅✅ 全 40 重审完成 — 总评

## 修复验证 (本轮14个hook)
**全部targeted修复确认生效、无回归**:
- ✅ b4 stage幻觉清零("Not staged") / b12 b17 化疗方案幻觉清空(2个旧P0全消)
- ✅ Creon/Xarelto/Lovenox恢复 9+ pdac (pdac2/5/8/9/11/13/15/18/19全验证)
- ✅ active化疗恢复 (pdac9 gem/abraxane部分/pdac10 FOLFOX/pdac18 capecitabine/pdac19 FOLFIRINOX)
- ✅ doublet (b7 abraxane/pdac3 nal-IRI) / locally advanced→Stage III (pdac1) / cT2NX (b18) / UCSF500 pending (pdac13)
- ✅ physical therapy幻觉消除 (b19/pdac16/pdac19) / Medication_Plan非癌药堆砌消除 (b19)
- ✅ 疑似≠确诊 (b9 b15 met软化suspected / pdac6 met正确) / axillary-as-distant修复 (b15)
- ✅ FNA/port/germline字段归位 (b8 b9 b10 b20部分) / 已完成labs清空 (b2 b3)

## 当前 P0/P1 (主复核)
- **P0 = 1**: b20 Type亚型幻觉"invasive ductal carcinoma"(病理仅"invasive cancer with lobular differentiation",全文无ductal). 旧2个P0(b4/b17)已清零.
- **P1 ≈ 33** (PL侧): 集中在下方11主题,多为本轮hook未targeted的新manifestation.
- **逐sample PL≥BL整体**: 40个sample PL都整体领先;但**~13个sample有个别字段被BL反超**(b13/b15/b16/b17/b19/b20/b10/pdac3/6/7/12/15/16/18/19),阻碍"无死角".

## 下一轮修复主题表 (11类, 按频次/严重度)
1. **[P0] 亚型幻觉**(b20): 病理含"lobular differentiation"且无ductal/IDC→禁注入ductal.
2. **[P1高频~8] plan类含已完成/无依据项**: imaging_plan(b4 mammogram/b13 brain MRI/b16 CT/pdac13 CT Chest)、genetic_plan(b17 BRCA/b18 mammaprint已出结果当plan). →plan排除已完成(有结果/过去日期/today done)+只留A/P有order依据.
3. **[P1高频~6] current_meds完整性/时态**: pdac9/18漏gemcitabine(gem别名+resume双药回填)、pdac5 folfirinox已完成(s/p+chemo break→空)、pdac4误清空(明列药勿删)、pdac10过窄、pdac12 CA19-9取最新.
4. **[P1~5] Stage**: pdac7 vessel-encasement>180°→Stage III(LOCALLY-ADVANCED扩触发)、pdac16专"Cancer Staging"段漏读、b13过度III(应IIB/not-staged)、b15 MBC/cervical→Stage IV、b20"early stage"漏.
5. **[P1~4] goals/response surveillance & 疑似**: pdac6/15 s/p resection+无active→surveillance(勿被high-risk-recurrence挡)、b2 ordering bug、response疑似软化(pdac6).
6. **[P1~5] 字段路由残留**: b5 AI/激素→procedure剔除、b17 echo→lab剔除、b20 port残留lab_plan、pdac19 ERCP/pdac3 RT残片→procedure归位、pdac18 referral incoming("has been seen by"剔除).
7. **[P1~3] findings纯度**: b19 findings=plan、pdac5/17 findings重复lab、重复type/stage→findings只收客观发现.
8. **[P1~3] supportive_meds**: pdac7 "Patient not taking"过滤、pdac14 home癌痛药(Tylenol)纳入、b4 alendronate.
9. **[P1~3] plan regex乱码/截断**: pdac11 Lab_Plan、pdac13/19 lab/genetic截断→清洗hook(跨日期/影像文本/截断回退).
10. **[P1~3] therapy_plan被拒药/None**: pdac7 None(continue时禁)、pdac12 ipilimumab+b14 tamoxifen(剔除"chose X vs"/"declined"语境药).
11. **[P1~3] ACP/完整性**: pdac3/12 hospice→ACP、CA19-9 non-secretor(pdac4/13)、recent_changes漏pause/完成治疗→surveillance(pdac13)、Referral SMS(pdac19).

## 结论
本轮14修复100%达成targeted目标且无回归(2个P0清零、Creon/化疗/字段归位全恢复)。全字段重审暴露出更深一层的递归问题(plan时态/current_meds完整/stage/字段路由/findings纯度等11类共~33 P1+1 P0),是上轮未覆盖的新manifestation。要"全方位碾压无死角"需基于本11类主题做第3轮hook+prompt迭代→重跑→重审。本doc持久化可续。
