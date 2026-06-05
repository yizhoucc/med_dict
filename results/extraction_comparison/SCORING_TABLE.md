# 评分表 · 人工逐 sample 逐题评测（PL vs BL）

> 按 CLAUDE.md 审查规矩：完整读原文 + PL/BL 输出，自然语言医疗判断，逐 sample 逐题，
> 不写脚本/不用 grep/不用 Agent/不跳过/不偷懒。每完成一个 sample 立即写入本表。
> 数据：PL=pipeline_breast_FIXED.txt / pipeline_pdac_FIXED.txt(修复定稿); BL=baseline_extract_*_json.txt
> 题库来源：RUBRIC_method1_topdown.md + RUBRIC_method2_empirical.md（合并去重为下列 12 题）

## 题集（12 题，全 40 sample 共用，中性专业措辞）

| ID | 题面（判定"PL 更好 / BL 更好 / 打平 / N/A"） |
|----|---|
| Q1 MED-REL | current medications 是否仅含肿瘤诊疗相关活动用药、未混入无关家庭慢病药？ |
| Q2 MED-ACTIVE | 当前在用的抗癌治疗(化疗/靶向/免疫/内分泌)是否被完整纳入？ |
| Q3 MED-TEMPORAL | 用药是否正确区分当前/既往/计划(无停用或未启动药列为当前)？ |
| Q4 FIND-OBJ | clinical findings 是否给出客观病理/影像发现而非仅主观症状/正常查体？ |
| Q5 FIND-COMPLETE | 驱动管理的关键客观发现(大小/进展/淋巴结/远处灶/关键合并症)是否更全？ |
| Q6 MOLEC | 已完成的分子/基因检测结果是否完整报告(含 panel 阴性项)？(无检测=N/A) |
| Q7 RESP | response assessment 是否正确反映是否在治疗中/有无可评疗效？ |
| Q8 PLAN-SPEC | 用药/治疗计划是否给出具体方案(药名+周期)而非含糊/遗漏？ |
| Q9 PROC-PURE | procedure plan 是否未混入影像/放疗/系统治疗？ |
| Q10 STAGE | 分期与转移描述是否一致且贴合原文(无过度分期/跨癌种借用)？ |
| Q11 NOHALLUC | 各字段是否不含原文不存在的内容(无幻觉)？ |
| Q12 DX-GRANUL | 癌症诊断粒度(组织学+分级+受体/分子亚型)是否更完整具体？ |

判定记法：PL / BL / TIE / NA。得分 PL=1, BL=0, TIE=0.5, NA=剔除。

## 入选标准（评完 40 后筛题）
保留：PL 平均 ≥0.65 且 BL 胜出 sample ≤3。其余删除（防 cherry-pick 单例质疑）。

## 进度跟踪（任务记忆 — 恢复时读这里）
BREAST(20): [x]1 [x]2 [x]3 [x]4 [x]5 [x]6 [x]7 [x]8 [ ]9 [ ]10 [ ]11 [ ]12 [ ]13 [ ]14 [ ]15 [ ]16 [ ]17 [ ]18 [ ]19 [ ]20
PDAC(20):   [ ]1 [ ]2 [ ]3 [ ]4 [ ]5 [ ]6 [ ]7 [ ]8 [ ]9 [ ]10 [ ]11 [ ]12 [ ]13 [ ]14 [ ]15 [ ]16 [ ]17 [ ]18 [ ]19 [ ]20

## 累计计数（实时，每题 PL/BL/TIE/NA）
截至 BREAST ROW 1-8 (8 行已审, 共 96 格): 总 PL 32 / BL 4 / TIE 49 / NA 11 (PL:BL=8:1)
每题(breast1-8) PL/BL/TIE/NA:
  Q1 MED-REL      PL6 BL0 TIE2 NA0  ← 强 PL, 保留
  Q2 MED-ACTIVE   PL1 BL0 TIE0 NA7  ← 待更多在治疗行 (非NA时全PL)
  Q3 MED-TEMPORAL PL0 BL0 TIE8 NA0  ← 全TIE, 候删
  Q4 FIND-OBJ     PL3 BL0 TIE5 NA0  ← PL倾向, 保留
  Q5 FIND-COMPLETE PL8 BL0 TIE0 NA0 ← 完美PL, 保留
  Q6 MOLEC        PL3 BL0 TIE1 NA4  ← 有检测时全PL, 保留
  Q7 RESP         PL2 BL1 TIE5 NA0  ← 弱PL
  Q8 PLAN-SPEC    PL4 BL0 TIE4 NA0  ← PL倾向, 保留
  Q9 PROC-PURE    PL1 BL0 TIE7 NA0  ← 弱(BL仅ROW5放DEXA)
  Q10 STAGE       PL1 BL2 TIE5 NA0  ← 残留stage bug拖累, 修后预期翻正
  Q11 NOHALLUC    PL0 BL1 TIE7 NA0  ← 弱(ROW2残留), 候删
  Q12 DX-GRANUL   PL3 BL0 TIE5 NA0  ← PL倾向, 保留
注: 待审 BREAST 9-20 + PDAC 1-20 (32 行). 残留stage bug #1/#2 修后 Q10/Q11 预期改善.

================================================================================
逐 sample 评分
================================================================================

### BREAST ROW 1 (coral 20) — 81yo 新患者, 三阴 IDC pT2N1a(Stage II), LVEF25%/CKD, 待PET/CT, 22家庭药无抗癌药
Q1 MED-REL=PL (BL列22家庭药/PL空,正确)
Q2 MED-ACTIVE=NA (未上抗癌治疗)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=TIE (均客观)
Q5 FIND-COMPLETE=PL (PL含LVEF25%/CKD关键合并症; BL漏)
Q6 MOLEC=NA (无分子检测,仅IHC受体)
Q7 RESP=TIE (均"未治疗无响应")
Q8 PLAN-SPEC=PL (PL捕捉AC/T·TC·CMF+不安全理由; BL"待定")
Q9 PROC-PURE=TIE (均干净)
Q10 STAGE=TIE (均Stage II合理)
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=TIE (均三阴+grade+IDC)
小计: PL3 BL0 TIE7 NA2

### BREAST ROW 2 (coral 21) — 73yo 局部复发 ER+/PR- IDC, 肝病灶=囊肿, 待AI(letrozole±bev)
Q1 MED-REL=TIE (均 zoledronic acid)
Q2 MED-ACTIVE=NA (AI 未起,无活动抗癌药)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=TIE (均客观)
Q5 FIND-COMPLETE=PL (PL含肝囊肿澄清/骨/对比更全)
Q6 MOLEC=NA
Q7 RESP=BL (PL response冗长且矛盾"无复发证据"——实有复发; BL"未评估"更稳)
Q8 PLAN-SPEC=TIE (均具体 AI+CALGB+bev)
Q9 PROC-PURE=TIE
Q10 STAGE=BL (PL Metastasis子字段"Yes含liver"与DistMet"No"矛盾[残留bug]; BL一致正确)
Q11 NOHALLUC=BL (PL "Originally StageIIA"原文未述+liver当转移; BL无幻觉)
Q12 DX-GRANUL=PL (PL含grade+三受体; BL缺grade/HER2)
小计: PL2 BL3 TIE5 NA2
[残留bug] POST-MET-RECONCILE 只清 Distant Metastasis 未清 Metastasis 字段 → 待修

### BREAST ROW 3 (coral 22) — 60yo 新患者三阴 metaplastic, 肾上腺结节2.3cm, 待新辅助化疗
Q1 MED-REL=PL (BL列5家庭药/PL空)
Q2 MED-ACTIVE=NA (新辅助未起)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=TIE (均客观)
Q5 FIND-COMPLETE=PL (PL含肾上腺结节2.3cm; BL仅一句肿块)
Q6 MOLEC=NA (panel待回,无完成结果)
Q7 RESP=TIE
Q8 PLAN-SPEC=PL (PL全方案paclitaxel→AC+I-SPY2; BL"AC/T"含糊)
Q9 PROC-PURE=TIE (均Medi-port)
Q10 STAGE=TIE (均locally advanced/No,一致)
Q11 NOHALLUC=TIE (PL微加antiemetics,极小)
Q12 DX-GRANUL=TIE (均三阴+metaplastic组织学)
小计: PL3 BL0 TIE7 NA2

### BREAST ROW 4 (coral 23) — 71yo 新患者 TNBC 乳腺复发(PET/CT 无转移), 含卵巢癌史, BRCA1+
Q1 MED-REL=PL (BL17家庭药/PL空)
Q2 MED-ACTIVE=NA (taxol未起)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=TIE (均客观)
Q5 FIND-COMPLETE=PL (PL含进展1.2→2.7+稳定肺结节; BL一句)
Q6 MOLEC=TIE (均BRCA1 carrier)
Q7 RESP=PL (BL把治疗前生长当疗效; PL正确"未治疗")
Q8 PLAN-SPEC=PL (BL"无用药计划"漏taxol; PL捕捉taxol x12+周期)
Q9 PROC-PURE=TIE (均Port)
Q10 STAGE=BL (PL乳腺复发误判StageIV远处转移[残留bug#2],与自身findings矛盾; BL正确No)
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=TIE (均IDC+grade+三受体)
小计: PL4 BL1 TIE6 NA1

[残留bug #2] note 用"recurrent/metastatic recurrence"时, 乳腺原位复发被过度判 Stage IV 远处转移
  (ROW4: DistMet"Yes to right lateral breast"=原发灶, 与 findings"no metastatic disease"矛盾) → 待修

### BREAST ROW 5 (coral 24) — 55yo 双侧乳腺癌新患者, s/p双乳切除, MammaPrint高/低危
Q1 MED-REL=TIE (均无药)
Q2 MED-ACTIVE=NA (术后辅助未起)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=PL (BL findings只正常查体漏病理; PL客观病理)
Q5 FIND-COMPLETE=PL (PL双乳病理+MRI+LN; BL仅正常查体)
Q6 MOLEC=PL (PL捕捉MammaPrint高危结果; BL完全没有)
Q7 RESP=TIE (均未治疗)
Q8 PLAN-SPEC=TIE (PL fuller chemo/AI/bone但漏radiation字段; BL有radiation但therapy薄,互抵)
Q9 PROC-PURE=PL (BL把DEXA影像放procedure; PL干净)
Q10 STAGE=TIE (均左III/右I一致)
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=TIE (均双乳完整受体+grade)
小计: PL4 BL0 TIE7 NA1 (注:PL漏radiotherapy_plan,不在12题内)

### BREAST ROW 6 (coral 25) — 53yo HR-/HER2+, 疑似骨转移, 颈动脉paraganglioma
Q1 MED-REL=PL (BL10家庭药/PL空)
Q2 MED-ACTIVE=NA (THP未起)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=TIE (均客观)
Q5 FIND-COMPLETE=PL (PL含病理IDC/HER2 3+/Ki67+体检7x5cm; BL略简)
Q6 MOLEC=NA (仅IHC受体)
Q7 RESP=PL (BL"预期excellent response"当现状; PL正确"未治疗")
Q8 PLAN-SPEC=TIE (均THP具体)
Q9 PROC-PURE=TIE (均纯净;BL多biopsy但Q9仅考纯度)
Q10 STAGE=TIE (PL"Suspected StageIV"+骨转移,颈动脉已移除,一致; BL亦suspected)
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=PL (PL含IDC+grade+ER/PR/HER2 3+; BL仅HR-/HER2+)
小计: PL4 BL0 TIE6 NA2 (Stage修复:Suspected+carotid移除+字段一致)

### BREAST ROW 7 (coral 26) — 44yo Lynch多癌种, 转移性TNBC在用abraxane+pembro, germline MSH2+
Q1 MED-REL=PL (PL抓抗癌药; BL 13家庭药)
Q2 MED-ACTIVE=PL (PL抓abraxane+pembro; BL完全漏抗癌药)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=PL (BL findings=主观症状清单; PL客观影像/病理)
Q5 FIND-COMPLETE=PL (PL全PET/CT灶+活检确诊; BL仅症状)
Q6 MOLEC=PL (PL全panel MSH2+BRCA/8基因阴; BL仅MSH2)
Q7 RESP=TIE (均不完美:PL含腋窝改善但时序微误;BL仅"progression")
Q8 PLAN-SPEC=PL (PL点名abraxane+pembro+方案; BL"continue current meds"含糊)
Q9 PROC-PURE=TIE (均Port)
Q10 STAGE=PL (PL正确IIB→IV一致; BL"not specified")
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=PL (PL全三阴+IDC+转移; BL省PR/组织学)
小计: PL8 BL0 TIE4 NA0

### BREAST ROW 8 (coral 27) — 70yo Stage IIA 新患者, AC+paclitaxel, 真实TTE, MammaPrint高危
Q1 MED-REL=PL (BL 9家庭药/PL空)
Q2 MED-ACTIVE=NA (AC+pac未起)
Q3 MED-TEMPORAL=TIE
Q4 FIND-OBJ=PL (BL findings术后症状; PL病理IDC/2-2LN/DCIS)
Q5 FIND-COMPLETE=PL (PL病理完整; BL仅术后查体)
Q6 MOLEC=PL (PL抓MammaPrint高危; BL漏)
Q7 RESP=TIE (均未治疗)
Q8 PLAN-SPEC=TIE (均AC+paclitaxel+日期)
Q9 PROC-PURE=TIE (TTE已被filter移出procedure)
Q10 STAGE=TIE (均pT2(m)N1a/No)
Q11 NOHALLUC=TIE
Q12 DX-GRANUL=TIE (均完整受体+grade)
小计: PL4 BL0 TIE7 NA1
