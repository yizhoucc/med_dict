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
- [~] breast：已评 2,4,5,7,9（高置信，读全文）；待评 1,3,6,8,10-20
- [ ] pdac 1-20
- **关键发现（暂停决策点）**：breast5 暴露 PL stage-inference 过度处理 bug（改错医生明写的 stage + 漏双侧）→ Q10/Q11 仍给 BL 送分。需决定先修这类 bug 还是先评完全部。

## running tally（仅 breast 5 个高置信样本）
| 题 | PL win | BL win | TIE | NA |
|---|---|---|---|---|
| Q10 STAGE | 3 | 1 | 1 | 0 |
| Q11 NOHALLUC | 0 | 1 | 4 | 0 |
| Q7 RESP | 2 | 0 | 3 | 0 |
| T6 DISTMET | 1 | 0 | 4 | 0 |

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

### 小结（breast 5 个高置信样本）
- Q10: PL 3（2,7,9） / BL 1（5） / TIE 1（4）
- Q11: PL 0 / BL 1（5） / TIE 4
- Q7:  PL 2（4,9） / BL 0 / TIE 3
- T6:  PL 1（9） / BL 0 / TIE 4
- **关键发现**：fix 后 metastatic/suspected/recurrent 类 PL 明显赢；但 breast5 暴露**另一类 PL stage 过度推断 bug**（"corrected pT2/IIB" 改错医生明写的 III + 漏双侧），导致 Q10/Q11 各送 BL 一分。要让 Q10/Q11 干净碾压，需再修这类 stage-inference 过度处理。

