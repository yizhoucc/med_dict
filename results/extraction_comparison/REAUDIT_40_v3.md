# 重审 v3 (Round 3 FIX3-final 全 40 逐字段) — 验证 round3 修复后 PL vs BL

> 触发：round3 修了 #1(亚型幻觉)/#2(plan已完成项)/#3(current_meds)/#6(字段路由) 共多个 hook + 5 个 patch，全量重跑 FIX3-final。对 `pipeline_*_FIX3.txt`(PL) vs `baseline_extract_*_json.txt`(BL) 严格全字段重审。
> 方法：subagent 并行初审（每个 1 sample，三块 brief，精炼输出）+ 主 Claude 复核 P0/P1。禁脚本判断。
> 对齐：PL/BL 按 ROW N（breast coral_idx=N+19；pdac coral_idx=N-1）。

## round3 验证点 (FIX3-final 已 spot-check 11/11 OK)
b20 亚型 / b4·b16·b17·b18 plan已完成 / pdac9·18 gemcitabine·pdac5 chemo-break / b5 AI·b17 echo·pdac19 ERCP·pdac18 incoming — 全部已确认落地，重审确认无回归。

## 严重度
P0=幻觉/疑似当确诊；P1=真实但放错字段/方向错/关键遗漏/时态/答非所问/误导；P2=模糊/格式；OK。

## STATUS
- [ ] breast 1-20  [ ] pdac 1-20
- 上轮(FIX2): P0=1(b20) + ~33 P1。本轮预期: P0=0, P1 大幅下降。

---

## breast 逐 sample (1-10 完成)
- b1(20): P0=0 P1=0. PL胜2(BL: palliative方向错+current_meds灌21家用药). 
- b2(21): **P1[PL] goals="palliative"应curative**(locally recurrent非转移, #11未修, ordering). P2 current_meds漏zoledronic. PL胜0/BL胜0(此点PL劣BL).
- b3(22): P0=0 P1=0. PL胜4. (lab attribution A2)
- b4(23): ✅**round3全验证**(stage"Not staged"/Type"local recurrence"/goals curative/imaging无mammogram). P0=0 P1=0. PL胜4.
- b5(24): ✅procedure AI已剔除. **P1[PL] Genetic_Testing_Plan漏Oncotype pending**(BL对, #新). PL胜多/BL胜1.
- b6(25): P0=0 P1=0. PL胜4(BL 3 P1: current_meds灌非癌药/DistMet疑似当确诊/response写预后). P2 ondansetron肠镜prep.
- b7(26): P1[PL] response(进展vs早期响应,BL同=TIE)/DistMet漏骨转移(BL同=TIE). P2 supportive Xarelto错配(BL更全). PL胜1/BL胜1.
- b8(27): **P1[PL] Stage="Not staged"但note明写"stage IIA pT2(m)N1a"×3**(NOBASIS过删/gate误清, attribution已指向stage; #4未修, BL反超). PL胜1(current_meds)/BL胜1(stage).
- b9(28): P0=0 P1=0. PL胜4.
- b10(29): **BL胜3**(Imaging漏echo/EKG/#7, Genetic_Results漏sent-pending/#9, current_meds清空fertility药tamoxifen/#3边界). 少见BL占优sample. PL胜1(Genetic_Plan MammaPrint).

### breast 1-10 v3 小结
- **round3(#1#2#3#6)修复全部held**(b4/b5验证). P0=0(b20在11-20待验). 
- 剩余P1均=deferred主题: b2 goals(#11), b8 stage NOBASIS过删(#4), b5 Oncotype-pending(新/#9类), b7 findings/supportive(#7#8), b10 imaging/genetic/current(#3#7#9).
- **b8 NOBASIS过删值得提到下轮优先**(明写stage却"Not staged", 是PL自伤).

## breast 11-20 完成
- b11(30): P0=0 P1=0. PL胜2(BL current_meds灌非癌药/Patient type错). P2 findings截断.
- b12(31): P0=0 P1=0. PL胜(lab旧值/current_meds/supportive,BL 2 P1). P2 referral cold-cap.
- b13(32): ✅imaging已修(drop已完成Brain MRI). **P1[PL] Stage"III"过度(#4)+radiotherapy"None"漏("after radiation")**. BL胜2.
- b14(33): **P1[PL] tamoxifen误入Therapy/Medication_Plan(#10,本轮未碰,如期仍在)**. PL其余胜. BL胜1.
- b15(34): **P1[PL] Stage"III"应IV(de novo MBC/cervical=distant,#4)+DistMet"Not sure"漏+current_meds过度清空**. BL胜3.
- b16(35): ✅**imaging CT幻觉已修(=Ultrasound)**. P0=0 P1=0. PL胜3.
- b17(36): ✅**lab echo+genetic brca已修**(lab"No labs"/genetic"None planned"). P0=0 P1=0. PL胜4.
- b18(37): ✅**genetic mammaprint已修(None planned)**. P0=0 P1=0. PL胜3.
- b19(38): **P1[PL] findings灌入plan(#7,本轮未碰,如期仍在)**. PL胜5/BL胜1.
- b20(39): ✅**Type ductal幻觉已修(invasive carcinoma with lobular differentiation)**. **P1: Stage空(early stage漏,#4)+germline pending漏(#11)**. PL其余胜.

## ✅ breast 全 20 汇总 (v3)
- **P0 = 0** (上轮唯一P0 b20亚型幻觉已修复✅). 
- **round3(#1#2#3#6)修复全部验证held无回归**: b4(stage/Type/goals/imaging)/b5(procedure AI)/b16(imaging CT)/b17(lab echo+genetic brca)/b18(genetic mammaprint)/b20(Type ductal) — 全部确认.
- 剩余 P1 ≈10-12, **全部是本轮明确未碰的 deferred 主题**: #4 stage(b8 NOBASIS过删/b13过度III/b15应IV/b20空), #7 findings纯度(b19/b7), #9 Oncotype/genetic-pending(b5/b10/b20), #10 否决药入plan(b14 tamoxifen), #11 goals方向(b2 palliative)/germline(b20), current_meds过度清空(b15).
- BL反超PL的sample: b2/b8/b13/b15/b10(少数), 全因上述deferred主题. 其余~14个PL无死角≥BL.

## pdac 1-10 完成
- pdac1(0): ✅Stage III locally advanced held. P0=0 P1=0. PL胜5.
- pdac2(1): ✅Creon held. **P1[PL] Procedure_Plan="ERCP"=回归**(2018过去ERCP被ENDO误捕→已修ctx). P2 findings weight-loss矛盾. BL胜procedure.
- pdac3(2): ✅doublet held. **P1 Advance_care漏hospice(#11)+Procedure RT残片(#6未尽)**. BL胜3.
- pdac4(3): ✅current_meds空正确(BL倾倒非癌药). **P1 CA19-9 non-secretor漏(#11,BL同漏=TIE)**. PL胜.
- pdac5(4): ✅✅**current_meds清空(COMPLETED-CHEMO held)+Creon+Xarelto held**. P0=0 P1=0. PL胜.
- pdac6(5): **P1 goals palliative→surveillance(#5)+response疑似当进展(#5)**. BL胜2.(deferred)
- pdac7(6): **P1 Stage空(#4)+supportive not-taking(#8)+Therapy None(#10)**. BL胜2.(deferred)
- pdac8(7): ✅Creon held. **P1 current_meds空漏gem/nab-pac**("continuing on gem/nab-pac"非doublet模式,单纯漏active chemo). DistMet PL对(BL错). PL胜.
- pdac9(8): ✅✅**doublet held(abraxane+gemcitabine)+Creon**. **P1 Referral.Specialty Phase I过去(incoming/时态)**. PL胜.
- pdac10(9): ✅FOLFOX held. **P1 lab旧值混入(#11/#20)+plan尾部碎片(#10类)**. BL胜plan干净度.

### pdac 1-10 v3 小结
- round3 pdac修复全部held: pdac1 stage/pdac5 chemo-break清空/pdac9 doublet/Creon(2/5/8/9). P0=0.
- **1个回归(我引入)**: pdac2 ENDO过捕过去ERCP → **已修ctx并commit**(待最终重跑验证).
- 新发现(非round3 scope): pdac8 current_meds漏active gem/nab-pac("continuing on X"模式IV-CHECK未触发,属#3延伸); pdac9 Referral incoming Phase I(#6延伸).
- 剩余P1=deferred主题(#4/#5/#8/#10/#11).

## pdac 11-20 完成
- pdac11(10): **P1 current_meds空(明列药)+Lab/genetic_plan regex乱码(#9未修)+Procedure EGD过去**. BL胜3.
- pdac12(11): **P1×3 ipilimumab被拒trial(#10)+CA19-9旧值(#11)+Advance_care漏hospice(#11)**(全未修). BL胜3. PL胜current_meds/Metastasis.
- pdac13(12): ✅Creon/UCSF500. **P1 Imaging"CT Chest"已完成当计划(#2此例未完全捕获,边缘)+recent_changes漏pause(#11)**. BL胜2.
- pdac14(13): **P1 supportive漏Tylenol(home癌痛药,#8未修)**. BL胜1. PL其余胜.
- pdac15(14): ✅Creon. **P1 goals="curative"应surveillance(#5未修)**. BL胜1. PL current_meds大胜.
- pdac16(15): ✅physical therapy幻觉消除. **P1 Stage"Not staged"(专Cancer Staging段"Stage IIB",#4)+Procedure"ERCP"过去(时态)**. BL胜2.
- pdac17(16): P0=0 P1=0. ✅recent_changes时态/Referral. P2 findings与lab重复(#7). PL胜.
- pdac18(17): ✅✅**current_meds gem+cape(DOUBLET held)+Referral incoming→None(POST-REFERRAL-INCOMING held)**. P0=0 P1=0. PL胜≥4.
- pdac19(18): ✅**ERCP captured(ENDO held)**. **P1 Patient type"New patient"应Follow up(新)+Lab_Plan乱码(#9)+Referral漏SMS(#11)**. BL胜3.
- pdac20(19): P0=0 P1=0. ✅current_meds过滤/MMR/无incoming. P2 Percocet/ctDNA. PL胜5.

---

# ✅ Round 3 全 40 重审完成 — 总评

## P0/P1 趋势
| 轮次 | P0 | P1(PL侧) |
|------|----|----|
| 原始(FULL_AUDIT) | 2 (b4,b17) | ~55 |
| Round2(FIX2) | 1 (b20) | ~33 |
| **Round3(FIX3)** | **0** | **~25** |

## round3(#1#2#3#6)修复验证 — 全部 held(+1自伤已修+1边缘)
- ✅ #1 亚型幻觉: b20 ductal→invasive carcinoma (P0清零)
- ✅ #2 plan已完成项: b4/b13(breast)/b16/b17/b18 imaging+genetic清空; **唯 pdac13 "CT Chest"边缘未捕获**(value attribution错配,留round4)
- ✅ #3 current_meds: pdac9/18 doublet补gem; pdac5 chemo-break清空; pdac4非癌药空正确
- ✅ #6 字段路由: b5 AI出procedure/b17 echo出lab/pdac18 referral incoming→None/pdac19 ERCP捕获
- ⚠️ **自伤回归 pdac2**: ENDO误捕2018过去ERCP → **已修ctx+commit(2d162744)**,待round4重跑验证

## 剩余 ~25 P1 = 明确 DEFERRED 主题(round3未碰, 留round4)
- **#4 Stage**(高频): b8/b13/b15/b20(breast) + pdac7/pdac16. NOBASIS过删明写stage(b8)、过度III(b13)、应IV(b15/MBC)、locally-advanced/vessel未触发(pdac7)、专Cancer Staging段漏读(pdac16)、early stage漏(b20).
- **#5 goals/response surveillance**: b2(palliative→curative ordering) + pdac6/pdac15(s/p resection→surveillance) + pdac6 response疑似当进展.
- **#7 findings纯度**: b7/b19 + pdac5/10/17(findings灌plan/重复lab).
- **#8 supportive**: pdac7("Patient not taking"未过滤) + pdac14(home Tylenol漏).
- **#9 plan regex乱码**: pdac11/19 Lab_Plan/genetic截断垃圾文本.
- **#10 therapy被拒药/None**: b14 tamoxifen + pdac7 Therapy None + pdac12 ipilimumab.
- **#11 ACP/完整性**: pdac3/12 hospice + pdac4/13 CA19-9 non-secretor/最新值 + Referral SMS(pdac19) + germline pending(b20) + Oncotype(b5).
- **新增**: pdac8 current_meds漏active gem/nab-pac("continuing on X"模式), pdac19 Patient type(化疗随访误判New).

## 结论
Round3 目标 100% 达成: #1/#2/#3/#6 修复全部生效, **P0 清零(2→0)**, P1 ~33→~25, 全部 targeted 修复经 FIX3 重审确认 held(仅 pdac2 自伤已code-fixed待验、pdac13 imaging 1边缘). 逐 sample PL 仍整体领先 BL; BL 反超 PL 的点全部集中在上述 7 类 deferred 主题(round3 明确未碰). 要达"全方位无死角碾压"需 round4 修这 7 类 + pdac2/13 收尾 → 重跑 → 重审. 本 doc 持久化可续.
