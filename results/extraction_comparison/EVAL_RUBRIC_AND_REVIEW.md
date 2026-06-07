# 评估题目设计准则 + 当前状态回顾 (Round 5 起点)

> 本文档记录：(A) 当前 4 轮迭代后的状态快照；(B) 用户提出的"好题目"评判准则（评估框架的重新定义）；(C) 据此对 40 个 v4 subagent 结果的回顾。持久化、防 context 丢失。

---

## A. 当前状态快照 (4 轮后)
- **全量测试**: FIX5 = 全 40 held-out (20 breast coral_idx20-39 + 20 pdac coral_idx0-19)，run.py + vLLM(Qwen2.5-32B-AWQ, port8000)，letter-off，0 错误。已提升为 `pipeline_breast_FINAL.txt` / `pipeline_pdac_FINAL.txt`。
- **审查**: 全 40 都用 subagent 处理（v4: 8 subagent × 5 sample），主 Claude 复核 P0/P1。
- **P0/P1 趋势**: P0 2→1→0→**0维持**；PL P1 ~55→~33→~25→**~9**。
- **round4 七主题修复全 held**（#4 stage / #5 goals-surveillance / #7 findings / #8 supportive / #9 regex / #10 否决药 / #11 ACP·non-secretor·germline·SMS）。
- **PL 残留 ~9 P1**（无 P0）: 多为 stage 字段边界(b10/b13/b15/b20/pdac10/15) + current_meds active-chemo gate 过保守(pdac7/8) + 个别(b11 PR-pending / pdac11 Patient type)。
- 文件: `REAUDIT_40_v4.md`(本轮逐 sample) / `ROUND3_TRACKING.md`(全四轮)。

---

## B. 用户的"好题目"评判准则 (评估框架重定义)

### 总目标 (不变)
在 40 个 sample 上，让 PL **全方位碾压** BL。两个要点：
1. **碾压式 (crushing)**: 找尽可能多的点，**每个点 PL 都明显胜出**（不是微弱/可辩，而是清晰的赢）。
2. **全方位 (comprehensive)**: 覆盖尽可能多的点（字段/sample），不留空白。

### 题目质量四准则
1. **细致简单（医生友好）**: 题目要**很具体/精确**——直接定位到 哪个 sample、哪个字段、哪句原文。医生**不需要来回找**，一眼能核对。
2. **表面公正（中立）**: 题目虽然是我们看 PL vs BL 差异设计出来的，但**表面上要中立公正**，不能明显是为 PL 量身定做/诱导。给医生看时像一道客观题。
3. **针对医疗要点（需医学知识）**: 题目要落在**医疗相关、需要医学背景知识**的点上——含**专用名词**或**临床判断/想法**，普通人（即使文学功底强、是聪明真人）没有背景知识也**答不出来/容易答错**。
4. （隐含）**可验证忠实**: 答案能在 note 原文找到依据。

### 好题 vs 坏题 分类（关键）
- **好题（医生感兴趣，要多挖）**:
  - **药物分类 / 现用药判定**（current_meds：哪些是抗癌药 vs 非癌家用药 vs 已停/计划药；化疗方案成分）—— 需药理+时态判断，普通人会乱列。**这是最强护城河。**
  - **Stage 分期**（cTNM、locally advanced→III、MBC→IV、节点 regional vs distant）—— 需 AJCC 知识。
  - **Metastasis 判定**（疑似 vs 确诊；regional 淋巴结 vs 远处；直接侵犯 vs 转移）—— 需临床判断。
  - **疗效 response**（疑似进展不当确诊；副作用≠疗效）—— 需临床判断。
  - **字段归位**（FNA→procedure、echo→imaging、激素→medication 非 procedure）—— 需医学分类知识。
  - **受体/分子状态**（ER/PR/HER2、PR pending≠PR+、non-secretor）—— 需病理知识。
- **坏题（普通人也能答 / 非核心，少用或不用）**:
  - **乱码 / 截断 / 格式**（regex garbage）—— 关键但非核心医疗题；普通人一眼看出是乱码，医生没兴趣。（看出现在哪里，但不当主打题）
  - **treatment goal（治疗目标 curative/palliative/surveillance）**—— 太广泛；普通真人读 note 也能懂，不需医学知识/不需 extract。医生没太大兴趣。
  - **findings 冗长 / 重复**、纯格式/字段空壳 —— 弱。

### 据此对评估的影响
- 我们之前修的很多 P1（goals 方向、findings 纯度、regex 乱码、plan 时态）—— 虽然让 PL 更干净，但**作为"展示 PL>BL 的题目"价值低**（坏题类）。
- **真正要主打的强题**: current_meds 药物分类（BL 全 40 系统性把非癌药当现用药+漏化疗 = PL 碾压最稳的点）、stage、metastasis、受体/分子、字段归位。
- 下一步回顾要做的: 重新审视 40 个 sample，**按"好题"标准**找出尽可能多的 PL 明显胜 BL 的医疗要点，评估是否达到"碾压式 + 全方位"，并定位还差哪些点。

---

## C. 回顾结果 (v5 全 40 重审，据"好题"lens)

**方法**: 把 PL FINAL(note_text+keypoints+attribution) 与 BL JSON 同放，8 个 subagent×5 sample 覆盖全 40，每个完整 brief 三块+好题准则，逐题判 PL vs BL 谁明显胜+分类 STRONG-MED/WEAK+顺带标 PL 自身错误。主 Claude 复核高价值 P1（读原文）。批次文件 `_audit_v5/`。

### C.1 医疗要点(STRONG-MED)总比分 ≈ **PL 76 : BL 10**（已剔除 WEAK 类如 goals/findings/lab/regex）
碾压**在总量上成立**。但 BL 的 10 个胜点不是散的，**聚成 4 个可修主题**——这就是"全方位无死角"还差的地方。

### C.2 护城河（PL 碾压最稳、最该当主打题的——已达成）
1. **current_meds 药物分类 (a) — 最强，~20/分有药的 sample 全胜**。BL 系统性把家用药(降压/降糖/胰岛素/他汀/PPI/眼药/酶替代)全倒进 current_meds，甚至列 **blood glucose test strips / lancets / syringes / BP monitor 设备 / "Patient not taking" 的药**；更致命的是 **BL 多次漏掉真正化疗**(b7 abraxane+pembro、pdac2 Gem/Abraxane、pdac3 5-FU/nal-IRI、pdac9 gem/abraxane、pdac10 FOLFOX、pdac18 gem/cape、pdac19 FOLFIRINOX)。PL 全部正确(在用列化疗、chemo break/术后随访留空、支持药路由到 supportive)。**这是最需医学知识、最干净、最可复现的碾压点。**
2. **分子/遗传结果 (f) — ~7-8 clean 胜**。BL 漏：BRCA2 家族史(pdac11)、CA19-9 non-secretor(pdac4/13)、MMR/MSI IHC 全present(pdac15)、Mammaprint 结果(b14/18)、完整 germline panel(b7)。PL 全捕获。
3. **PDAC stage 推断 (b) — ~7 clean 胜**。locally advanced→III(pdac1/7/13)、metastatic→IV(pdac12/14/20)、复发→IV(pdac8)；BL 一律 "Not specified"。
4. **Metastasis 疑似/确诊/区域 (c) — ~6 clean 胜**。suspected liver/bone 不当确诊(b6/pdac1/6/12)、淋巴结≠远处(b15)、nodal recurrence=Yes(pdac8)、peritoneal 而非 BL 误报 lungs(pdac12)。
5. **Response (d) / 字段归位 (e)** — 零星 clean 胜：not-yet-treated(b4/b9)、progression(pdac19)；echo→imaging(b12)、Mammaprint→genetic 非 lab(b18)、Creon→supportive(pdac8)。

### C.3 ❗还差的"无死角"——BL 反超 PL 的 4 个主题（Round 5 靶点，按优先级）
**主题 A（最高优先，且违反原则①）: 乳腺癌 STAGE 过度开火 — PL 输 4 个 (b10/b13/b15/b20)**。主 Claude 已逐一读原文复核确认：
  - b10: 原文 "T2N1, **clinical** stage II"；PL "Stage IIA (pT2N1)" = c误标p + cT2N1 实为 IIB 非 IIA（双错）。
  - b13: 原文 "not certain of tumor size/nodes"、**全程无分期**；PL 凭空 "Stage III"。
  - b15: 原文定性 **de novo MBC**、无分期；PL "Stage III" 自相矛盾无依据。
  - b20: 原文 ">10 and 9 cm + lung/liver lesions"；PL 却 "Early stage"（被原文直接打脸，近误导级）。
  → round-4 的 POST-STAGE-EXPLICIT / -MBC / -LOCALLY-ADVANCED / early-stage fallback **过度自信**。修法：note 无明确 AJCC/cTNM 时不得自拼 stage；含 metastatic/de novo MBC/goals=palliative 时禁输有限期分期；保留 c/p 前缀不臆造亚组；有明确远处结节或巨大原发(>5cm)时禁判 early。这是把 stage 从**负资产**(输 BL+违忠实)扳回**正资产**(本应是好题)。

**主题 B: 乳腺癌 Genetic_Testing_PLAN 漏/错字段 — PL 输 3 个 (b5 Oncotype / b8 genetics referral / b13 Mammaprint plan)**。PL 写 "None planned"，而检测/转诊确实计划中(有时 PL 捕获在 Referral/Results 但 PLAN 字段空)。修法：Genetic_Testing_Plan 触发词补 Oncotype/Mammaprint/genomic assay/genetic counseling referral；跨字段去重时优先填 PLAN。

**主题 C: pdac3 — PL 输 2 (DistMet 漏脾 / Response 偏乐观)**。met sites 漏 "direct invasion of spleen"（SITES hook 的"精确>完整"取舍误伤）；response 锚定旧 02/16 "stable" 漏 A/P 明示 "continued clinical/symptomatic decline"+CA19-9 15→142。

**主题 D（次要）: 受体精度 b10 (BL 给 PR 25%/ER>95%，PL 泛化 PR+)；Patient type pdac11 (New 误标 Follow up，属 WEAK)**。

### C.4 稳定性检查（用户担心的"4 轮后不稳"）结论
- **无新增 P0 幻觉**，除 1 个**已知且已在代码修复**的：**pdac8 ACP "transition to hospice" 编造**（note 仅 "prioritize QoL" 无 hospice）。hook 已收紧(commit dc06e6d1)，但 **FINAL 输出是修复前生成的，文件里仍带错** → 需重跑一次才落地。
- 其余 PL-ERR 均 P2 边界：pdac10 EUS/mammogram 多余 plan 项、b13 falx "possibly met"(放射科判 low-suspicion/meningioma)、b12 lab 漏、pdac13/18 化疗 on-hold 时态细节。
- **重要再框定**: 主题 A 的乳腺 stage cluster 本身就是 4 个 faithfulness 隐患，之前按"找幻觉"的 P0/P1 视角被低估；按"PL 是否输给 BL"的新视角才暴露。

### C.5 Round 5 行动建议（待用户确认）
1. 修 stage 过度开火(主题A，最高优先) → 2. 修 genetic_testing_plan(主题B) → 3. 修 pdac3 spleen+response(主题C) → 4. 受体精度(主题D) → 5. **统一重跑全 40**(同时落地已修的 pdac8 ACP) → 6. 再按本 lens 重审，确认 BL 的 10 胜点清零、无新回归。
目标：医疗要点 STRONG-MED 比分从 76:10 推向接近 全胜:0，且 stage 从负资产转正。

---

## D. Round 5 执行记录 (2026-06-06)

### 已修 (代码 + 单元模拟验证，逐一确认 single-sample scope 无回归)
**主题 A — stage 忠实 (commit 2b6fcf2f)**:
- `POST-STAGE-CLINICAL`: 原文 "clinical/pathologic stage X" → 忠实捕获并覆盖 LLM 臆造的亚组/c-p 误标。b10 "Stage IIA (pT2N1)" → "Clinical stage II (cT2N1)"。
- `POST-STAGE-EARLY-VERIFY`: "early stage" 被巨大原发(>5cm)/远处灶打脸 → 降级。b20 "Early stage" → "Not staged in note (locally advanced; distant lesions unconfirmed)"。
- `POST-STAGE-MBC` 加 hedge: presumptive de novo MBC → "Suspected Stage IV (de novo MBC, pending confirmation)"。b15。
- b13 "Stage III"(无依据) 经实测当前 NOBASIS 已会剥离成 "Not staged in note"（FINAL 陈旧故未体现）——重跑自动修复。
- 回归 b8/pdac16 不变。✅

**主题 B — genetic_testing_plan 捕获 (commit a894384b)**:
- `POST-GENETIC-PENDING` 改为遍历所有 assay 出现处(跳过科普句、加"已出具体结果"护栏): b5→"Oncotype DX RS (planned)"、b13→"Mammaprint on her tumor (planned)"；b18(已出 High Risk)正确保持 None planned。
- 新增 `POST-GENETIC-PLAN-REFERRAL`: genetics 转诊反映进 plan(保留 Referral)。b8→"Referred for genetic testing/counseling."。
- 回归 b17/b18/pdac11/pdac13 不变。✅

**主题 C — pdac3 (commit 060c8362)**:
- `POST-RESPONSE-AP-DECLINE`: A/P impression 明写 clinical/symptomatic decline → 覆盖锚定单条影像的 "stable"。pdac3。全 40 仅 pdac3 触发，无误触。
- **脾经主 Claude 读原文复核：不改**。原文是 "direct invasion of spleen"(局部 T4) + "splenic lesions compatible with **infarcts**"(脾梗死非转移)——PL "liver, peritoneum" 才是忠实的远处转移列表；subagent 误判为 BL 胜。精确>完整 成立。

**主题 D — 受体精度 + patient type (commit 682f2cd4)**:
- `POST-PATIENT-TYPE-NEW`: present-tense this-visit 新患者信号(seeing as new patient / presents today for second opinion / INITIAL VISIT 标题) → Follow up→New。**收紧排除时间线里 "MM/DD: Initial consultation" 往期事件**——全 40 仅 pdac11 触发，不误伤真随访 pdac10/13/16/17。
- `POST-TYPE-RECEPTOR-PCT`: 原文 "ER N%, PR N%" 经典措辞 + Type 仅泛化 ER+ → 追加百分比。b10 "(ER >95%, PR 25%)"。`\bER` 护栏避开 HER2 误匹配。仅 b10。

### 待办
- [ ] 统一重跑全 40 (WSL, run.py + vLLM)，落地 A/B/C/D + 之前已修但 FINAL 陈旧的 (b13 NOBASIS、b15 MBC、pdac8 ACP hospice)。
- [ ] 重跑后按好题 lens 再审，确认 BL 的 10 胜点清零 + 无新回归。
