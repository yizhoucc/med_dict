# 重评 Q10/Q11/Q7/T6 vs BL（PL=pipeline_*_FINAL.txt 含7-fix+scope修复）

> 目的：fix 后 PL 在 stage/halluc/resp/distmet 是否已能干净碾压 BL → 能否加回这 4 类评分题。
> 方法：CLAUDE.md 人工逐样本，读 note A/P 原文 + PL 字段 + BL 字段，自然语言判断，不写脚本/不用 Agent。
> PL = pipeline_breast_FINAL.txt / pipeline_pdac_FINAL.txt（2026-06-05 final2，含 A2/A3 A/P-scope 修复）
> BL = baseline_extract_*_json.txt（单 prompt 裸模型，单 Distant_Metastasis 字段，无 Metastasis）

## 评分题定义
- **Q10 STAGE**：Stage_of_Cancer 是否正确 + 恰当具体 + 与 met 自洽（confirmed/suspected 区分）。
- **Q11 NOHALLUC**：诊断字段是否无幻觉（编造 stage/Originally/met 站点/确诊化疑似）。
- **Q7 RESP**：response_assessment 是否正确 + 简洁 + 答"如何响应"（非 plan/非整段 CT 复制）。
- **T6 DISTMET**：远处转移有无/部位是否正确（含疑似 vs 确诊）。
- 每题每样本：**PL** / **BL** / **TIE** / **NA**。判据 = 谁更忠实+准确+自洽。

## STATUS（上下文满了从这里恢复）
- 决策：**先评完全部 40，再统一修**（用户定）
- [x] **breast 1-20 全部已评**（1-9 高置信读全文；10-20 中置信 BL block+PL字段）
- [ ] **pdac 1-20（待评，下一批）**
- 已发现 5 类 PL bug（见"已发现的 PL bug 类"），评完 pdac 后统一修。
- 恢复方法：读本 header → 读 pipeline_pdac_FINAL.txt 下一未评 ROW 的 note + baseline_extract_pdac_json.txt 对应 ROW block → 判 4 题写入 → 更新 tally。
- BL 字段名：单 "Distant_Metastasis"（无 Metastasis）。

## running tally（breast 全 20）
| 题 | PL win | BL win | TIE | 说明 |
|---|---|---|---|---|
| Q10 STAGE | 6 (1,2,6,7,8,9) | 2 (5,18) | 12 | PL 明显领先 |
| Q11 NOHALLUC | 1 (6) | 1 (5) | 18 | 已扳平（fix前BL9） |
| Q7 RESP | 3 (4,6,9) | 1 (10) | 16 | PL 领先 |
| T6 DISTMET | 3 (6,9,20) | 4 (1,13,15) | 13 | 仍微输，需修3类distmet bug |

---

## 逐样本（breast）

> 已读全文核对 ground truth 的样本：2,4,5,7,9（高置信）。1,3,6,8,10-20 待读。

### breast ROW2（locally recurrent, 肝=cyst, 未治疗）
- PL: Stage "Locally recurrent" / DistMet No / Met No / resp "Not yet on treatment"
- BL: Stage "Not specified in the note" / DistMet "No distant met mentioned" / resp "Not assessed"
- Q10 **PL**（"Locally recurrent" 正确且有信息量 vs BL 放弃）| Q11 TIE（都干净，PL 这轮无 IIA 幻觉）| Q7 TIE | T6 TIE

### breast ROW4（右乳 recurrent, "No evidence of metastatic disease", 将启 taxol）
- PL: Stage "Originally unspecified, now Stage III" / DistMet No / Met No / resp "Not yet on treatment"
- BL: Stage "Not specified" / DistMet No / resp "cancer grown 1.2x0.7→2.7x1.7 cm"
- Q10 TIE（PL "now Stage III" 依据不足偏险；BL 安全；都不完美）| Q11 TIE | Q7 **PL**（未治疗→"not on treatment" 对；BL 把肿块增长当 response，答非所问）| T6 TIE

### breast ROW5（双侧；A/P 明写 Left Stage III T3N1 + Right Stage I T1cN0）⚠️PL输
- PL: Stage "Stage IIB (corrected: pT2 N1, 1 positive node)" ❌（6.2cm=T3 非 T2；应 III；且漏双侧）/ DistMet No / Met No
- BL: Stage "Left: Stage III (T3N1); Right: Stage I (T1cN0)" ✅（与原文逐字一致 + 双侧全）/ DistMet No
- Q10 **BL**（PL stage 错且漏双侧；BL 完全正确）| Q11 **BL**（PL "corrected pT2/IIB" 与原文 T3 矛盾＝过度推断/幻觉；BL 用原文真值）| Q7 TIE | T6 TIE
- **根因**：PL 的 stage-inference hook（"corrected: ..."）对双侧 + 微转移病例过度处理，把医生明写的 III 改错成 IIB。BL 裸抄反而对。**这是 fix 没覆盖的另一类 PL stage bug。**

### breast ROW7（确诊转移 metastatic to liver+nodes, 肝活检证实）✓回归已修
- PL: Stage "Originally Stage IIB, now metastatic (Stage IV)" / DistMet+Met "Yes, to liver and nodes"
- BL: Stage "Not specified" / DistMet "Yes (liver and nodes)" / resp "Evidence of progression"
- Q10 **PL**（正确 Stage IV + Originally IIB 有原文依据 "clinical Stage IIB"；BL "Not specified" 漏）| Q11 TIE | Q7 TIE（都对：progression）| T6 TIE（都 Yes liver+nodes）

### breast ROW9（疑似转移 "possibly metastatic", FNA 待确诊）✓回归已修
- PL: Stage "Originally Stage III (T3N2), Suspected Stage IV (pending confirmation)" / DistMet "Suspected, to left cervical LN" / Met "Not sure"
- BL: Stage "Stage III (T3N2)" / DistMet "Not sure" / resp "Unresectable recurrence, possibly metastatic"
- Q10 **PL**（捕捉 original III + 当前 suspected IV 的双重状态；BL 只给 III 漏了 suspected 进展）| Q11 TIE | Q7 **PL**（PL "not on treatment" 对；BL 把诊断陈述塞进 response）| T6 **PL**（"Suspected, to left cervical LN" 具体且恰当 hedge；BL "Not sure" 更含糊）

### breast ROW1（新患者, 术后 pT2N1a, PET/CT 待做评估转移）
- PL: Stage "Stage IIB (pT2N1a)"（正确+具体, pT2N1a 在 path 报告）/ DistMet **No** / Met No / resp "Not yet on treatment"
- BL: Stage "pT2N1a"（原始码）/ DistMet **Not sure** / resp "Not applicable"
- Q10 **PL**（IIB 是 pT2N1a 的正确分组, 更有用且有原文依据）| Q11 TIE | Q7 TIE | T6 **BL**（A/P 明说 "obtain PET/CT to assess metastasis" 即分期未做完 → BL "Not sure" 对；PL "No" 过早断定）
- **根因 bug 类2**：POST-DISTMET-DEFAULT 在"分期影像待做"时仍填 "No"。应在 A/P 含 "PET/CT to assess mets"/"staging imaging" 时给 "Not sure / pending staging"。

### breast ROW3（locally advanced multifocal, A/P 明说 "no evidence of metastatic disease", curative neoadjuvant）
- PL: Stage "Locally advanced, multifocal" / DistMet No / Met No / resp "Not yet on treatment"
- BL: Stage "Locally advanced, multifocal" / DistMet No / resp "Not yet assessed, awaiting neoadjuvant"
- Q10 TIE | Q11 TIE | Q7 TIE | T6 TIE（此处 PL "No" 正确——A/P 明写 no mets，非分期待做）

### breast ROW6（A/P "suspicious for bone met", "biopsy for definitive stage IV", "if stage IV"=疑似）✓PL横扫
- PL: Stage "Suspected Stage IV (pending confirmation)" / DistMet "Suspected to left ilium and bilateral sacral ala" / Met "Not sure" / resp "Not yet on treatment"
- BL: Stage "Not definitively staged yet, suspected Stage IV due to bone metastasis" / DistMet "Yes (bone metastasis)" / resp "Anticipated excellent response..."
- Q10 **PL**（PL 干净 hedge；BL "due to bone metastasis" 把疑似当因果）| Q11 **PL**（BL "Yes (bone metastasis)" 把疑似说成确诊＝overstatement；PL 正确 hedge）| Q7 **PL**（PL "not on treatment" 对；BL 把预后 "anticipated excellent response" 当 response，未治疗就臆测）| T6 **PL**（PL 具体+hedge "Suspected to ilium/sacral ala"；BL "Yes" 过度断定）
- A3 疑似处理在此完胜 BL。

### breast ROW8（A/P 明写 "stage IIA pT2(m)N1a", 术后 adjuvant）
- PL: Stage "Stage IIA (pT2(m)N1a)"（与原文逐字一致）/ DistMet No / Met No / resp "Not yet on treatment"
- BL: Stage "pT2(m)N1a"（原始码）/ DistMet No / resp "Not applicable"
- Q10 **PL**（给出 stage group 且=原文 IIA；BL 只给码）| Q11 TIE | Q7 TIE | T6 TIE
- 注：此为 stage 处理的**正例**（原文 IIA → PL IIA）；对比 breast5 反例（原文 III → PL 错成 IIB）。区别：breast5 是双侧+微转移把 hook 搞晕。

### breast ROW10-20（中置信：读 BL block + PL 字段 + 关键 note 段；多为早期病例）
- ROW10（新患, neoadj 待启, 生育保存中）：Q10 TIE | Q11 TIE | Q7 **BL**（PL "On treatment" 存疑——尚未开始抗癌治疗；BL "Not applicable yet" 更稳）| T6 TIE
- ROW11（DCIS pTisNx 风险降低）：全 TIE（PL/BL 都 pTisNx + No）
- ROW12（Clinical Stage II, PET 无 mets）：全 TIE
- ROW13（node+；脑 parafalcine 灶=meningioma 良性；axillary 区域）：Q10 TIE | Q11 TIE | Q7 TIE | T6 **BL**（真值无远转；BL "No" 准；PL "Not sure" 过度 hedge，被良性 meningioma 吓到）
- ROW14（cT 未明, no distant met detected）：全 TIE（都 punt stage + No）
- ROW15（"metastatic ... involving lymph nodes", cervical+axillary, FNA 待）：Q10 TIE | Q11 TIE | Q7 TIE | T6 **BL**（PL DistMet **空**＝bug；BL "Yes (to lymph nodes)" 给了可用答案）
- ROW16（Clinical stage III ILC）：全 TIE（都 "Clinical stage III" + No）
- ROW17（Stage IIb T2N1M0）：全 TIE（都给 IIb T2N1M0 + No）
- ROW18（cT2NX, ATM mut）：Q10 **BL**（原文 cT2NX，BL 捕到，PL "Not mentioned" 漏）| Q11 TIE | Q7 TIE | T6 TIE
- ROW19（s/p mastectomy, NED, 随访）：全 TIE（PL/BL resp 都 "NED on exam"；stage 都弱）
- ROW20（双侧 HER2+, "liver and lung nodules present" 未活检）：Q10 TIE（PL stage **空** bug；BL "Not specified" 都不给）| Q11 TIE | Q7 TIE | T6 **PL**（PL "Suspected (liver and lung nodules pending confirmation)" 具体+hedge；BL 冗长 "no confirmation"）

### breast 全 20 汇总
- **Q10 STAGE：PL 6 : BL 2 : TIE 12** —— PL 明显领先（metastatic/suspected/IIA-IIB 正确分组）。BL 2 胜=breast5(过度推断)+breast18(PL漏cT2NX)
- **Q11 NOHALLUC：PL 1 : BL 1 : TIE 18** —— 已扳平（fix 前 BL 9）。唯一 BL 胜=breast5
- **Q7 RESP：PL 3 : BL 1 : TIE 16** —— PL 领先。BL 1 胜=breast10(PL "On treatment" 存疑)
- **T6 DISTMET：PL 3 : BL 4 : TIE 13** —— **仍微输 BL**。BL 胜=breast1(分期待做误 No)、13(良性灶过度 hedge)、15(空 distmet bug)；PL 胜=6,9,20(疑似 hedge 漂亮)
- 结论：Q10/Q7 可作 PL 评分题（领先）；Q11 已扳平（修 breast5 类后可转 PL）；**T6 仍需修 3 类 distmet bug 才能转正**。

### ⚠️ 已发现的 PL stage/met bug 类（待全部评完后统一修）
1. **stage 过度推断**（breast5）：原文已明写 stage（含双侧/Stage X）时，inference hook 不该改写。规则：原文有明确 stage → 忠实采用。
2. **distmet 过早 No**（breast1）：分期影像待做（PET/CT to assess mets）时 POST-DISTMET-DEFAULT 不该填 No，应 "Not sure"。
3. **distmet 过度 hedge**（breast13）：原文明说某灶"most likely meningioma"等良性时，不该因它把 DistMet 写 "Not sure"，应 "No"。
4. **distmet 空**（breast15）：Met 含 cervical/nodal 但 DistMet 留空。"cervical/supraclavicular" 未进 distant 站点表 → reconcile R2 没触发同步。补站点表 + 同步。
5. **stage 漏取**（breast18）：原文有 cT2NX 但 PL 写 "Not mentioned"。提取/hook 把 cTxNx 也算作有效 stage，别漏。
（注：1 和 5 相反——5 是漏、1 是过度改；修时都归到"原文 stage 优先忠实采用"原则）

### 小结（breast 6 个高置信样本：1,2,4,5,7,9）
更新 tally：Q10 PL4/BL1/TIE1 · Q11 PL0/BL1/TIE5 · Q7 PL2/TIE4 · T6 PL1/BL1/TIE4
（原文5个小结见下）
### 小结（旧·breast 5 个高置信样本）
- Q10: PL 3（2,7,9） / BL 1（5） / TIE 1（4）
- Q11: PL 0 / BL 1（5） / TIE 4
- Q7:  PL 2（4,9） / BL 0 / TIE 3
- T6:  PL 1（9） / BL 0 / TIE 4
- **关键发现**：fix 后 metastatic/suspected/recurrent 类 PL 明显赢；但 breast5 暴露**另一类 PL stage 过度推断 bug**（"corrected pT2/IIB" 改错医生明写的 III + 漏双侧），导致 Q10/Q11 各送 BL 一分。要让 Q10/Q11 干净碾压，需再修这类 stage-inference 过度处理。

