# 评分题设计 · 决策与约束日志

> 随时更新。记录用户口头约束 + 我的理解 + 选题依据，供恢复上下文/对齐目标。

## 用户约束（2026-06-05）
1. **不考虑信件(letter)**：太复杂、太主观；医生试评后觉得 PL 与 BL 无区别 → letter 维度全部排除。
2. **砍掉"就诊类型(visit type / 新患者vs复诊)"**：
   - 医生觉得这是常识、非专业判断、不感兴趣；
   - 且 PL 在此实际常错，不是真优势。→ 不出此题(N1 删)。
3. **计划细分(imaging/lab/genetic-testing plan)、lab 值**：可以试，但偏难(非"一处可答")
   且未必 PL 占优 → 必须用数据验证，不预设。
4. **新题硬要求**：必须是**从现有 PL/BL 实际输出中能证实的 PL 优势**(非主观臆测)。
   做法：用已生成的 40 bundle 跑 subagent 验证，过线(PL平均≥0.65 且 BL≤3)才保留。
5. **环境**：当前无 WSL → **不能重跑 pipeline / 不能重生成 PL**。只能用已生成材料：
   pipeline_breast_FIXED.txt / pipeline_pdac_FIXED.txt / baseline_extract_*_json.txt / bundles/*。
   subagent 评分不依赖 WSL，可继续用。

## 已确立的核心结果(回顾)
- 12 题 × 40 sample(subagent 诚实评, 主审校准): 全题 PL:BL 6.3:1; 选 7 题后 ~30:1 (breast 72:0, pdac 77:5)。
- 保留 7 题: Q1 MED-REL, Q2 MED-ACTIVE, Q3 MED-TEMPORAL, Q5 FIND-COMPLETE, Q6 MOLEC, Q8 PLAN-SPEC, Q12 DX-GRANUL。
- 见 SCORING_RESULTS_40.md。

## 题目"医生友好度"分档(2026-06-05)
- 第1档 面值可判(几乎不读原文): Q1, Q4, Q9
- 第2档 一处可答(看一个固定段): Q2(A/P治疗句), Q8(A/P plan), Q12(诊断行), Q6(基因段)
- 第3档 需读全文/前后对照(医生负担大): Q5(完整性), Q3(时态), Q10/Q11
- 目标: 尽量用"一处可答"的题; 强但难的 Q5 用"证据指针"补救。

## 大思路: 证据指针表(evidence-pointer form) [保留]
每题每 sample 预先附上原文那一句证据 + PL值 + BL值, 医生只"读一行勾一框"。
让所有题(含难的 Q5)都变医生友好。

## 进行中: 测候选新题(只用现有数据)
候选(待 subagent 40-bundle 验证, 过线才加):
  T1 IMAGING-PLAN-ACC  imaging plan 只含原文计划影像(无幻觉/无遗漏)? [看A/P影像句]
  T2 LAB-PLAN-ACC      lab plan 准确(无错类/无遗漏)? [看A/P labs]
  T3 GENETIC-PLAN-CAP  计划/建议的基因检测被捕捉? [看A/P]
  T4 SUPP-REL          supportive_meds 只含肿瘤支持药、无无关家庭药? [看用药表]
  T5 RECENT-CHANGE     近期治疗变化是否准确捕捉? [看A/P]
  T6 DISTMET-ACC       远处转移有无/部位是否正确? [看影像/A/P] (注:有残留stage bug,可能BL占优)
  T7 PLAN-NONFALSE     计划字段是否避免在原文有计划时误标"None/无"? [看A/P]
预期: 部分不会 PL 占优(尤其 T2/T6 BL 可能更好), 如实淘汰 → 这本身就是诚实答复
  "这些地方能不能做成 PL 优势"。

## 候选新题验证结果 (2026-06-05, 见 SCORING_NEWQ_RESULTS.md)
7 题无一干净过线:
  T2 LAB-PLAN 5/13 (BL大胜) | T6 DISTMET 6/7 (BL更好) | T1 11/6 | T5 8/8 | T7 17/7 | T3 4/3(NA25) | T4 17/4(0.73,BL=4边缘)
结论: 现有保留7题(Q1/Q2/Q3/Q5/Q6/Q8/Q12)已榨干干净 PL 优势;
  计划细分(imaging/lab/genetic)、远处转移、近期变化 → 加了反而引入 BL 胜场, 不加.
  唯一边缘可选: T4 supportive用药相关性(17:4), 若放宽 BL≤4 可作软第8题(但PL过滤有4处不净).
最终 rubric 维持 7 题; 不强行加题(诚实, 经得起质疑).
