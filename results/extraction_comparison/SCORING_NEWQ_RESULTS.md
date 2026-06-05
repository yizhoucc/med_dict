# 候选新题验证结果 (40 sample subagent, 2026-06-05)

目的: 测 7 个候选维度能否做成干净的 PL 优势。结论: 均不过线(见下)。

数据: 已生成的 bundles (无 WSL). guide=SUBAGENT_GUIDE_V2.md


## 逐 sample × 候选题 矩阵

sample             T1  T2  T3  T4  T5  T6  T7  |PL BL T NA
breast ROW 1      TIE  NA  BL  NA  NA  BL TIE  |0 2 2 3
breast ROW 2       NA  NA  NA  PL TIE  BL  PL  |2 1 1 3
breast ROW 3       BL TIE  PL  BL  NA TIE  PL  |2 2 2 1
breast ROW 4       BL TIE  NA  NA  BL  BL TIE  |0 3 2 2
breast ROW 5       PL  NA TIE  NA  NA TIE  PL  |2 0 2 3
breast ROW 6       PL TIE  NA TIE  NA  PL TIE  |2 0 3 2
breast ROW 7      TIE  BL  NA TIE  PL TIE  PL  |2 1 3 1
breast ROW 8      TIE  BL TIE  BL TIE TIE TIE  |0 2 5 0
breast ROW 9       BL  NA  NA  PL  BL  BL TIE  |1 3 1 2
breast ROW 10      BL TIE  PL  NA  NA TIE TIE  |1 1 3 2
breast ROW 11      NA  NA  NA  NA  PL TIE  PL  |2 0 1 4
breast ROW 12     TIE  BL  NA  NA  BL TIE  PL  |1 2 2 2
breast ROW 13      PL  NA  BL  NA  NA TIE  BL  |1 2 1 3
breast ROW 14      NA  PL  NA  PL  BL TIE TIE  |2 1 2 2
breast ROW 15      NA  NA TIE  NA  NA TIE  PL  |1 0 2 4
breast ROW 16     TIE  PL  NA  NA  NA TIE  PL  |2 0 2 3
breast ROW 17      BL  NA  PL  BL  NA TIE  PL  |2 2 1 2
breast ROW 18      NA  NA  NA  NA  NA TIE TIE  |0 0 2 5
breast ROW 19      PL TIE  NA  PL TIE TIE  PL  |3 0 3 1
breast ROW 20     TIE  BL  PL  NA  NA  PL  PL  |4 1 1 2
pdac ROW 1         PL  NA  NA  PL  PL  PL TIE  |4 0 1 2
pdac ROW 2        TIE  BL TIE TIE TIE TIE  BL  |0 2 5 0
pdac ROW 3         BL  BL  NA TIE TIE  PL TIE  |1 2 3 1
pdac ROW 4        TIE  NA  NA TIE  BL TIE  PL  |1 1 3 2
pdac ROW 5        TIE  BL  NA TIE  NA TIE  BL  |0 2 3 2
pdac ROW 6        TIE  BL  NA  NA  BL  BL TIE  |0 3 2 2
pdac ROW 7        TIE  NA  NA  PL TIE TIE TIE  |1 0 4 2
pdac ROW 8         PL  PL TIE  PL TIE  PL  PL  |5 0 2 0
pdac ROW 9        TIE  BL  NA  PL  PL TIE  PL  |3 1 2 1
pdac ROW 10       TIE  BL  BL  PL TIE TIE  PL  |2 2 3 0
pdac ROW 11        PL  BL TIE  PL  NA TIE  PL  |3 1 2 1
pdac ROW 12       TIE  BL  NA  PL  PL  PL  BL  |3 2 1 1
pdac ROW 13        PL TIE  NA  PL  BL TIE  BL  |2 2 2 1
pdac ROW 14        NA  BL TIE TIE  NA TIE  BL  |0 2 3 2
pdac ROW 15       TIE TIE  NA  PL  PL TIE  BL  |2 1 3 1
pdac ROW 16        PL  PL  NA  PL  PL TIE  PL  |5 0 1 1
pdac ROW 17       TIE TIE  NA  BL  BL TIE TIE  |0 2 4 1
pdac ROW 18        PL  PL  NA  PL TIE TIE TIE  |4 0 2 1
pdac ROW 19        PL TIE  NA  PL  PL  BL TIE  |3 1 2 1
pdac ROW 20        NA  NA TIE  PL  NA  BL TIE  |1 1 2 3

## 每候选题汇总 + 决定 (保留: score>=0.65 且 BL<=3)

T   name          PL BL TIE NA | score 决定
T1  IMAGING-PLAN  11  6  16  7 | 0.58 淘汰
T2  LAB-PLAN       5 13   9 13 | 0.35 淘汰
T3  GENETIC-PLAN   4  3   8 25 | 0.53 淘汰
T4  SUPP-REL      17  4   7 12 | 0.73 淘汰
T5  RECENT-CHANGE  8  8   9 15 | 0.50 淘汰
T6  DISTMET-ACC    6  7  27  0 | 0.49 淘汰
T7  PLAN-NONFALSE 17  7  16  0 | 0.62 淘汰

## 结论
7 个候选维度无一干净过线:
- T2 LAB-PLAN: BL 大胜(5:13) — lab 计划是 BL 强项, 不可加
- T6 DISTMET-ACC: BL 更好(6:7) — BL 保守更安全 + PL 残留 stage bug
- T1/T5/T7: 太多 BL 胜场(6/8/7), 不稳定
- T3 GENETIC-PLAN: 大多 NA(25), 区分度不足
- T4 SUPP-REL: 最接近(17:4, score0.73), 但 BL=4>3; PL 支持药过滤有4处不净 → 边缘, 可选(放宽阈值才进)
=> 现有 7 题已榨干干净的 PL 优势; 新维度只会引入 BL 胜场/稀释. 不加(除非接受 T4 作软第8题).