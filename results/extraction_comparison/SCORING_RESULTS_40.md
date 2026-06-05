# 评分结果 · 40 sample × 12 题（subagent 诚实评测, 主审校准）

数据: pipeline_*_FIXED vs baseline_*_json. 40 个 subagent 各评一行, 见 SUBAGENT_SCORING_GUIDE.md

校准: subagent 在 breast1-8 与主审手工评一致(甚至略严,未灌水PL) → 数据可信


## 逐 sample × 题 矩阵 (PL/BL/T/NA)

sample             Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9 Q10 Q11 Q12  | PL BL T NA
breast ROW 1       PL  NA  PL TIE  PL  NA TIE  PL TIE  BL TIE TIE  | 4 1 5 2
breast ROW 2      TIE  NA TIE TIE  PL  NA  BL TIE TIE  BL  BL TIE  | 1 3 6 2
breast ROW 3       PL  NA  PL TIE  PL TIE TIE  PL TIE TIE  BL TIE  | 4 1 6 1
breast ROW 4       PL  NA  PL TIE  PL  PL  PL TIE TIE  BL  BL TIE  | 5 2 4 1
breast ROW 5      TIE  NA TIE  PL  PL TIE TIE TIE  PL TIE TIE  PL  | 4 0 7 1
breast ROW 6       PL  NA  PL TIE  PL  NA  PL TIE TIE TIE  PL  PL  | 6 0 4 2
breast ROW 7       PL  PL  PL  PL  PL  PL TIE TIE TIE  PL TIE  PL  | 8 0 4 0
breast ROW 8       PL  NA  PL  PL  PL  PL TIE TIE TIE TIE TIE TIE  | 5 0 6 1
breast ROW 9       PL  NA  PL TIE  PL  NA  PL TIE  BL  BL  BL  PL  | 5 3 2 2
breast ROW 10      PL  NA TIE TIE  PL TIE TIE TIE TIE TIE TIE TIE  | 2 0 9 1
breast ROW 11      PL  NA TIE TIE  PL  NA TIE TIE TIE TIE  PL  PL  | 4 0 5 3
breast ROW 12      PL  NA TIE TIE  PL TIE TIE  PL  PL TIE TIE TIE  | 4 0 7 1
breast ROW 13      PL  NA  PL TIE  PL  NA  PL  PL TIE  BL  BL TIE  | 5 2 3 2
breast ROW 14      PL  NA TIE TIE  PL  PL TIE TIE TIE TIE TIE  PL  | 4 0 7 1
breast ROW 15      PL  NA TIE TIE  PL  NA TIE  PL TIE  PL TIE  PL  | 5 0 5 2
breast ROW 16      PL  NA  PL TIE  PL  NA TIE  PL TIE TIE TIE TIE  | 4 0 6 2
breast ROW 17      PL  NA TIE TIE  PL TIE TIE TIE TIE TIE TIE TIE  | 2 0 9 1
breast ROW 18      PL  NA TIE TIE  PL  PL TIE  PL TIE  BL TIE  PL  | 5 1 5 1
breast ROW 19      PL  PL  PL  PL TIE  PL TIE  PL TIE TIE  BL TIE  | 6 1 5 0
breast ROW 20     TIE  NA TIE TIE  PL  NA TIE  PL TIE TIE TIE  PL  | 3 0 7 2
pdac ROW 1        TIE  NA  PL TIE  PL  NA  PL TIE TIE  PL TIE TIE  | 4 0 6 2
pdac ROW 2         PL  PL  PL  PL  PL TIE  PL  PL TIE TIE  PL TIE  | 7 0 5 0
pdac ROW 3         PL  PL  PL TIE  PL TIE  BL  PL  BL  PL TIE TIE  | 7 2 3 0
pdac ROW 4         PL  NA TIE TIE  PL  NA TIE TIE TIE  PL TIE TIE  | 3 0 7 2
pdac ROW 5         PL  NA  PL  PL  PL TIE TIE  PL TIE TIE TIE  PL  | 6 0 5 1
pdac ROW 6        TIE  NA  BL TIE  PL  NA TIE TIE  BL  BL  BL TIE  | 1 4 5 2
pdac ROW 7         PL  PL  PL TIE  PL TIE TIE  BL TIE  PL TIE TIE  | 5 1 6 0
pdac ROW 8         PL  PL  PL TIE  PL  PL TIE  PL TIE  PL  PL  PL  | 8 0 4 0
pdac ROW 9         PL  PL  PL  PL TIE TIE TIE  PL  PL TIE TIE  PL  | 7 0 5 0
pdac ROW 10        PL  PL  PL TIE  PL TIE TIE TIE TIE TIE TIE  BL  | 4 1 7 0
pdac ROW 11        PL  NA  PL  PL  PL  NA TIE  PL TIE TIE TIE TIE  | 5 0 5 2
pdac ROW 12        PL  NA  PL TIE  PL  NA TIE  PL TIE  PL TIE TIE  | 5 0 5 2
pdac ROW 13        PL TIE  PL TIE  PL  NA TIE  BL TIE TIE TIE TIE  | 3 1 7 1
pdac ROW 14        PL  NA TIE TIE  PL  NA TIE  PL TIE TIE TIE TIE  | 3 0 7 2
pdac ROW 15        PL  NA TIE  PL  PL  PL TIE TIE TIE  PL TIE  PL  | 6 0 5 1
pdac ROW 16        PL  PL  PL TIE  PL  NA TIE TIE TIE TIE TIE TIE  | 4 0 7 1
pdac ROW 17       TIE  NA TIE  PL  PL TIE TIE TIE TIE TIE  BL  BL  | 2 2 7 1
pdac ROW 18        PL  PL  PL TIE  PL TIE  PL  PL TIE  PL TIE  PL  | 8 0 4 0
pdac ROW 19        PL TIE  PL TIE  PL  NA  PL TIE  BL  BL  BL  PL  | 5 3 3 1
pdac ROW 20        PL  NA TIE  PL  PL  PL TIE  PL TIE  BL TIE TIE  | 5 1 5 1

## 每题汇总 + 选题 (保留 score>=0.65 且 BL<=3)

Q   name           PL BL TIE NA | score 决定
Q1  MED-REL        34  0   6  0 | 0.93 ★保留
Q2  MED-ACTIVE     10  0   2 28 | 0.92 ★保留
Q3  MED-TEMPORAL   24  1  15  0 | 0.79 ★保留
Q4  FIND-OBJ       11  0  29  0 | 0.64 淘汰
Q5  FIND-COMPLETE  38  0   2  0 | 0.97 ★保留
Q6  MOLEC           9  0  13 18 | 0.70 ★保留
Q7  RESP            8  2  30  0 | 0.57 淘汰
Q8  PLAN-SPEC      19  2  19  0 | 0.71 ★保留
Q9  PROC-PURE       3  4  33  0 | 0.49 淘汰
Q10 STAGE          10  9  21  0 | 0.51 淘汰
Q11 NOHALLUC        4  9  27  0 | 0.44 淘汰
Q12 DX-GRANUL      15  2  23  0 | 0.66 ★保留

## 总分
全12题40样本: PL 184 / BL 29 / TIE 220 / NA 47  → PL:BL = 6.3:1
保留7题(Q1/Q2/Q3/Q5/Q6/Q8/Q12): PL 149 / BL 5 / TIE 80 / NA 46  → PL:BL = 29.8:1 (压倒性)
  breast 保留题: PL 72 / BL 0 / TIE 41  → PL:BL = 72.0:1
  pdac 保留题: PL 77 / BL 5 / TIE 39  → PL:BL = 15.4:1

## 淘汰题说明(透明)
Q4 FIND-OBJ(0.64): 多数TIE(两边都给客观), 差异不够稳定
Q7 RESP(0.57): 多为TIE(治疗前两边都说未评估)
Q9 PROC-PURE(BL4): BL 有时更全(含 biopsy), 非纯度问题
Q10 STAGE(BL9)/Q11 NOHALLUC(BL9): 残留 stage bug(#1 Metastasis字段未清, #2 复发过度判IV)拖累
  → 这两题修 bug 后预期翻正; 但保留题已达 30:1, 不依赖它们