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

## pdac 逐 sample
