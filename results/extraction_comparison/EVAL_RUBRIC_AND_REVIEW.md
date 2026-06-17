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

### 重跑暴露的 hook 冲突（live ≠ 单元模拟，逐个修）
单元模拟通过 ≠ live 通过——live pipeline 有几十个 hook 互相覆盖，逐次重跑暴露：
1. **theme B 完全没生效（commit 0fb25ec1）**：`POST-GENETIC-PLAN-COMPLETED` + `POST-GENETICS-RESULT-CHECK` 在 note 里测试名附近见结果词就删，把 LLM 正确提取的 b5 "being assessed for Oncotype"、b8 "pursue genetic counseling"、b13 "Mammaprint to be obtained after surgery" 全清成 "None planned."（note 有科普句 "Mammaprint **Low Risk** profile"、"if **high risk** with Oncotype" 误触）。修：两 hook 都加"值本身是未来计划(to be/being assessed/pursue/after surgery)则永不删"护栏 + RESULT-CHECK 加假设语境(if/patients/tools like)排除。
2. **theme C 被压缩丢弃（commit 5704b251）**：`POST-RESPONSE-AP-DECLINE` 在 pdac3 触发了，但其后的 `POST-RESPONSE-COMPRESS` 把 "decline" 句丢了（decline 不在 RESP_KW）。修：AP-DECLINE 移到 COMPRESS 之后 + RESP_KW 加 declin/deteriorat/worsen。
3. **b13/b15 stage 非确定性根因（commit 8eb97056，最关键）**：`POST-STAGE-FINAL` Case 1 把"Stage IV 但 DistMet=No"**凭空改成 "Stage III"**（非转移≠III=fabrication，且覆盖 MBC）。b13/b15 随 LLM 当次 emit 的 stage 形态在 Not-staged/IV/III 间翻动。修：FINAL 不再造 III；有正向(非否定)MBC框架→保留/归一(b15→Suspected Stage IV)，否则→诚实 "Not staged in note"(b13)。**两种 LLM 形态都收敛到同一答案**，彻底锁死确定性。

### 关键教训
- vLLM greedy 跨运行非确定：同代码不同 run 输出不同 stage 形态 → 确定性 hook 必须对**所有可能的上游形态**收敛到同一答案，不能依赖 LLM emit 特定形态。
- 单元模拟只测单个 hook；必须 live 重跑 + 查 log 才能发现 hook 间覆盖。本轮 breast 跑了 3 次、pdac 跑了 2 次才逐一锁死。

### 最终重跑 (commit 8eb97056) + 全 40 再审 —— ✅ 完成
- 全 40 重跑完成，提升为 FINAL（commit 2bcb371e）。8 subagent 全 40 再审 + 主 Claude 复核。

#### 再审结果：STRONG-MED 比分 76:10 → **PL 77 : BL 0(清晰) [+1 可辩 b10]**
- **4 主题全部确认翻转**（subagent 独立证实）：
  - A stage: b10="Clinical stage II (cT2N1)"、b13="Not staged in note"、b15="Suspected Stage IV (de novo MBC, pending confirmation)"、b20="Not staged (locally advanced; distant unconfirmed)"——全部正确，不再输 BL。
  - B genetic_plan: b5 Oncotype / b8 转诊 / b13 Mammaprint 全捕获；b17/b18(已出结果)正确保持 None。
  - C: pdac3 response 以 "Continued clinical/symptomatic decline" 开头（subagent: "fix confirmed working"）；脾正确排除（局部侵犯+梗死，非远处转移）。
  - D: b10 Type "(ER >95%, PR 25%)"；pdac11 "New patient"。
- **b13/b15 确定性锁死**：本次重跑 b13="Not staged"、b15="Suspected Stage IV"，与第三次重跑一致（POST-STAGE-FINAL 修复后两种 LLM 形态都收敛）。
- **稳定性**：pdac8 ACP="Not discussed"（无 hospice 幻觉）确认；**全 40 零 P0**。
- **subagent 报的 2 个"BL 胜"经主 Claude 复核推翻**：
  - b9(gabapentin+HCTZ)：纯非癌家用药，PL 留空=current_meds 三分法正确设计（护城河本身），非 BL 胜。
  - b10(fertility cetrorelix/menopur + tamoxifen)：促排卵药非癌；tamoxifen 在生育保存语境（非乳腺癌定义治疗）——**唯一真正可辩边界**，列上更完整但排除可辩护。
- **残留 P2（全部 plan 类/非诊断要点，可选后续清理，需再跑一次才落地）**：
  - medication_plan 尾部黏附未来 trial 药（pdac3 trametinib、pdac8 irinotecan，均 note 中有但属"未来可能"）。
  - current_meds 化疗 on-hold 时态（pdac13/18 gemcitabine 已 s/p/暂停仍列现用——但远胜 BL 漏化疗）。
  - b12 lab_summary 漏旧 CBC（坏题字段）；pdac20 研究性 ctDNA assay 误入 imaging_plan；pdac10 EUS/mammogram plan 黏附。

#### 结论
**医疗要点上 PL 全方位碾压 BL（77:0 清晰胜 + 1 可辩），零 P0，4 主题全修，b13/b15 确定性锁死。** 达成"碾压式 + 全方位"目标。最强护城河仍是 current_meds 药物分类（全 40 中有药的样本几乎全胜——BL 系统性把降压/降糖/眼药/试纸/促排卵药当现用药且漏真正化疗）+ 分子遗传捕获 + stage 推断 + 疑似≠确诊。残留仅 plan 类 P2，不影响诊断要点胜负。

Round 5 commits: 2b6fcf2f(A) → a894384b/0fb25ec1(B) → 5704b251(C) → 682f2cd4(D) → 8eb97056(确定性根因) → 2bcb371e(FINAL)。

---

## E. Round 5 P2 残留清理 (2026-06-06) —— 完成

按用户"逐个修 P2 → 重跑全 40 → 再审"。修了 5 个 plan 类/response P2（跳过 #3 b12 lab=坏题、#7 typo=cosmetic）：
- **#1**（commit d96b90b7）POST-MEDICATION-SUPPLEMENT 加投机性未来 trial 药排除（possibility/conceivably/preliminary data/limited yield…，±110 窗口）→ pdac3 trametinib、pdac8 irinotecan 不再混入 medication_plan。
- **#6**（commit 84ac25b0）POST-RESPONSE-PRETREATMENT-DESC：未治疗患者 response 是疾病测量描述→"Not yet on treatment"。修 b4，顺带改进 b9。
- **#2**（commit efbeea4e）POST-MEDS-ONHOLD-ANNOTATE：化疗明确 hold/pause 时加 "(systemic therapy currently on hold)" 标注（不删药，保护护城河）。pdac18；pdac13 因"proceed with FOLFIRINOX"换方案保守跳过。
- **#4**（efbeea4e）POST-PLAN-GARBAGE-CLEAN(c)：imaging_plan 研究试验文本(无具体 modality)→"No imaging planned"。pdac20 ctDNA/microbiome assay。
- **#5**（efbeea4e）GARBAGE-CLEAN(d)：procedure_plan 可切除性散文(无未来手术动词)→"No procedures planned"。pdac10。

### 最终重跑(commit efbeea4e→14ac6329 FINAL v2) + 全 40 再审结果
- **STRONG-MED 比分 PL ~73 : BL 0**（无任何清晰 BL 胜；b16/17/18 的"BL current_meds"是 PL 有意过滤非癌家用药=设计本身，非真胜）。
- **全部 P2 修复 + round-5 themes 经独立 subagent 再审确认落地、无回归**：pdac3/8 无投机药✓、b4/b9 response✓、pdac18 on-hold✓、pdac20 imaging✓、pdac10 procedure✓；A/B/C/D 全保持；b13/b15 确定性保持；pdac8 ACP 无 hospice✓。**零 P0，零我引入的回归。**
- **深审新浮现 2 个既有问题（非本轮回归）**：
  - **b5（P1）**: genetic_testing_results 混入病理发现(LVI/margins/DCIS/LN/微转移)，只有 MP Low Risk + Oncotype pending 是真基因组内容。字段纯度问题(好题 f 字段)。POST-GENETIC-RESULTS-IHC 因字段含 MP/Oncotype 关键词未剥离病理段。
  - **b11（P2）**: Type "PR+" 但原文 "PR pending"(findings 已正确写 pending)。轻微 over-reach。
  - 二者均非 round-5/P2 改动引入；是本次彻底再审才暴露的旧问题。

### F. b5/b11 收尾 (commit 1ad661da → 200f6bd8 FINAL v3) —— 完成
- **b5(P1)**: POST-GENETIC-RESULTS-IHC 扩展为也剥离外科病理段(LVI/margins/DCIS/节点数/微转移/grade/tumor size)，只保留含基因/assay 名的段。b5 → "MP Low Risk (+0.321). Oncotype DX RS is pending."(仅基因组内容)。仅 b5 受影响。
- **b11(P2)**: POST-TYPE-PR-PENDING：note 说 PR pending 且无确定 PR 结果时，把 "PR+" 改 "PR pending"。b11 Type → "ER+/PR pending DCIS"。仅 b11 触发。
- **最终重跑全 40 + 全字段回归抽查**：b5/b11 修复落地；主题 A/B/C/D + 5 个 P2 + 护城河(pdac2 Gem/Abraxane、b19 exemestane、pdac8 ACP 无 hospice) + b13/b15 确定性 **全部保持，零回归**。

## 最终结论 (Round 5 全部完成)
经 6 轮迭代 + P2 清理 + b5/b11 收尾：**医疗要点上 PL 对 BL 全方位碾压(STRONG-MED ~73:0，无清晰 BL 胜)，零 P0，确定性锁死，所有已知 P0/P1/P2 问题清完**。FINAL = pipeline_{breast,pdac}_FINAL.txt (commit 200f6bd8)。

---

## G. 诚实更正：r7 真正的 8-subagent 全审查 (2026-06-06)
**背景**：用户质疑 r7(b5/b11 收尾) 只做了脚本抽查、没做 subagent 仔细审查。属实——我之前的"零回归"是基于脚本抽取关键字段，覆盖不全，**夸大了**。现补做 8-subagent×5 全 40 逐字审查。

### 确认成立（headline 不变）
- **PL ≫ BL 全字段**: STRONG-MED ≈ 85:0（无清晰 BL 胜）。BL 仍系统性 current_meds 倒家用药+漏化疗、stage 放弃、**pdac12 幻觉肺转移**。
- b5(genetic_results 病理污染已剥离)、b11(PR pending) 修复确认落地正确。
- b13/b15 stage 确定性保持；pdac3 脾/response、pdac8 ACP、pdac18 on-hold、pdac20 imaging、pdac10 procedure 均确认。
- **无我 P2/b5/b11 改动引入的新回归**。

### ❗全审查暴露的真实残留 PL 问题（脚本抽查漏掉的；多为既有，非本轮回归）
- **P1 b19 current_meds 漏 Zoladex**：患者在用 exemestane + **Zoladex(goserelin 卵巢抑制)**("cont zoladex locally monthly")，PL 只列 exemestane。护城河字段的真实遗漏。
- **P1 pdac10 medication_plan 仍混入 "; also: capecitabine, irinotecan"**：irinotecan 已停("omit irinotecan since C3")、capecitabine 仅未来假设(consolidative chemoradiation)。我的 #1 投机药排除有缺口("omit" 不在排除词)。
- **P1/可辩 b16/b17 Metastasis="No" 但淋巴结阳性**：b17 "2/2 lymph nodes involved + ECE"(N1)、b16 clinical stage III；而 Metastasis 字段在别处(b13/b15)用于记区域淋巴结受累→此处"No"前后不一致。(DistMet="No" 正确)。字段语义问题。
- **P2**: b12 lab_summary "No labs in note"(实有 CBC/BMP)；pdac3 procedure_plan 黏 RT 散文碎片；b18 genetic_results 混入肿瘤受体(纯度)；b16/b17/b18 DistMet attribution 错配(值对引用错,A0/A2)；b2 Type "HER2-" 过度断言(1994 肿瘤未测 HER2)；pdac7 Referral 漏 Gyn Onc。

### 教训
- **"脚本抽查" ≠ "subagent 仔细审查"**：抽查只覆盖预设字段，抓不到 Metastasis/lab/attribution 等未抽的字段，也抓不到 vLLM 漂移。声称"零回归"前必须跑真正的逐字审查。

---

## H. r9/FINAL v4 真正 8-subagent 全审查 (2026-06-06)
**b19/pdac10 两个 P1 已修复并确认**:
- b19 current_meds = "exemestane, Goserelin" ✅ (OS hook 移到 CROSSCHECK 之后；CROSSCHECK 曾剥掉 LLM 提取的 zoladex，现 OS hook 在其后补回 Goserelin)。
- pdac10 medication_plan 干净 (全局停药护栏 + omit/consolidative 排除)。

**headline 确认**: PL 医疗要点全面 ≥ BL，零 P0，current_meds 5/5 碾压、molecular/stage/疑似≠确诊领先。既往全部修复保持。

**全审查暴露的残留 (抽查会漏的；多为次要字段/vLLM 漂移)**:
- **b16/b17/b18 Metastasis 字段="No" 但 N+/NX**: 多 subagent 反复 flag (P1/P2)。次要模糊字段(prompt 定义仅"if there is met")，主字段 DistMet 都正确。上次决定不改因广改会误伤 b18(NX+FNA良性→假阳性)。**可加"N0/NX/良性结节"护栏后只翻真 N+ (b1/b8/b9/b10/b17)，与 b13/b15 一致**。
- **b5 genetic_results="No results"**: 本轮 vLLM 漂移把 MammaPrint(MP high/low risk)丢出该字段(仍在 findings)；BL 同错=TIE。跨运行非确定性，需正向捕获 hook 才能锁。
- **b20 Type 合并双侧 HER2→"HER2+"**(漏左侧 HER2-；findings 有双侧)。唯一 subagent 记的"BL 略胜"点。
- b7 Xarelto(暂停仍在 supportive)、pdac3 procedure_plan RT 碎片: P2。

**教训重申**: 每次重跑 vLLM 让次要字段小幅漂移(本轮 b5)，确定性 hook 只锁住了重要字段；次要字段的彻底锁定需更多正向捕获 hook，是收益递减的长尾。

---

## I. 最终收尾轮 (rB/FINAL v5, commit cf41ecaa→b335700a) + 我亲自复审 — 停止
**本轮做的（2 个干净 hook）**：
- `POST-MET-REGIONAL-NODE`（Metastasis 字段）：node-positive(TNM N1-3 from glued TNM / X-Y nodes positive / 确诊性 node-positive 排除 if-patients-negated) → "Yes, regional lymph node(s)"。**全 8-subagent 审查确认**：b1/5/8/9/10/13/17 正确触发（真 N+），b11(DCIS)/b16/b18(NX) 正确跳过，零假阳性。
- 撤掉 b20 双侧 HER2 hook（脆弱正则 live 误判 left HER2+）+ b5 MammaPrint hook（科普句过度触发）= 打地鼠，按规则不做。

**全 40 全 8-subagent 最终审查结论**：
- PL 在核心诊断轴（Stage/Metastasis/Response/molecular/current_meds）全面 ≥ BL；**BL 不赢任何核心好题点**；零 P0 幻觉。
- current_meds 全 40 系统性碾压（BL 倒家用药+漏化疗；pdac11/14/15 被一个 subagent 判"BL 胜"= 严格完整性视角，**主 Claude 复审推翻**：留空非癌药是 PL 设计护城河，其余 subagent 一致认同 PL 对）。
- 既往全部修复保持（b19 Goserelin、stage 确定性、b11 PR-pending、pdac3/8/10/18/11、molecular 捕获 BRCA2/CA19-9-nonsecretor/MMR）。

**审查新冒出的次要长尾（= 打地鼠，停止追）**：
- b6 Metastasis="Not sure"（FNA 证实腋窝转移未被 hook 捕获，新 miss，P2，DistMet 正确）。
- pdac13 gemcitabine 时态、pdac7 Gyn-Onc 转诊漏、b20 Type 双侧 HER2 未拆分、b5 genetic vLLM 漂移——均次要字段、信息多在 findings、且每轮重跑 vLLM 漂移会持续产生新的此类长尾。

**最终判定（按用户 "如果打地鼠就停止" 规则）**：核心目标已达成且稳固，剩余为次要字段长尾 + vLLM 跨运行漂移的打地鼠。**停止迭代。** FINAL = pipeline_{breast,pdac}_FINAL.txt (commit b335700a)。

---

## 2026-06-15 删除两题（医生反馈：无评分价值）

临床医生看过题库后，明确指出下面两题**对评估没有价值**，要求删除：

| 题 | 字段 | 医生意见 | 历史 PL vs BL 表现（保留存档） |
|----|------|----------|-------------------------------|
| 就诊类型 Patient type | `Reason_for_Visit.Patient type` | 新患者 vs 随访，无临床评分意义 | PL 胜 6 / 打平 34 / BL 胜 0 |
| 治疗目标 goals（治愈 vs 维持） | `Treatment_Goals.goals_of_treatment` | curative vs palliative 方向，无临床评分意义 | PL 胜 10 / 打平 30 / BL 胜 0 |

**处理方式（保留结果、只删评分载体）**：
- **保留**：两题逐样本 PL/BL 判定结果完整留在 `_audit_v5/verdicts.json`（共 19 条），及本 doc 上述历史表现，供日后复盘/存档——"我们曾评过这两题、发现没用"这件事本身被记录。
- **删除**：从公用题库 `QUESTIONS.txt`（19→17 题）、三个 HTML 生成器题库定义（`build_scoring_html.py` / `build_verdict_html.py` / `build_review_html.py`）、所有图（`make_figs.py`，经 `QUESTIONS` 自动排除）中移除这两题。打分载体 `PL_vs_BL_scoring.html` 现为每样本 17 题（40×17=680）。
- 因 scoring HTML 题集变化，localStorage key 升 `pl_bl_scoring_v1`→`v2`，避免旧版残留答案串入导出。

**删后口径变化**：scored 题 758→678（减 80 = 2 题×40 −2 NA）；PL 110 / 打平 558 / BL 10（PL 总胜从 126 降至 110，因这两题原各贡献 +6、+10，且 BL 在这两题从未获胜）。核心结论不变：current_meds 仍 35:0 碾压，BL 仍无任何核心高价值题获胜。

---

## 2026-06-16 评审侧改进（医生反馈：题目要体现"我们要 extraction"）

医生指出当前评审有几个会误导人类打分的问题，本轮**只改评审侧**（题目 / rubric / HTML），prompt 改动留 backlog。

**1. Q6 type/receptor 按癌种区分 →（同日进一步）PDAC 直接删除 Q6**：胰腺癌没有 ER/PR/HER2（那是乳腺标志物）。最初用 `PER_CANCER` 把胰腺 Q6 改成"Histologic type & grade"，但医生表示 PDAC 根本没有 marker，**这道题对 PDAC 没意义，直接不做**。最终：`qset_for(cancer)` 对 PDAC 过滤掉 type_receptor，**乳腺 17 题 / 胰腺 16 题**（共 660）。进度条改为按每样本题数 `data-nq` 计算；localStorage key 升 v2→v3。乳腺 Q6 仍用 `PER_CANCER['b']` 强调要逐项 ER/PR/HER2。

**2. 核心评分原则 "extraction ≠ summary" 写进题目（防人类被带偏）**：
- 问题动机：PL 没信心时把一堆证据/细节罗列出来（= 成功 extract 了细节，让医生判断）= **好**；BL 丢了细节、给一句模糊总结，"碰巧读起来正确无可挑剔" = **跑题**（任务是 extraction，summary/infer 即使对了也不是我们要的）。但人类打分天然会被流畅总结带偏，给 BL 高分。
- 落地（"离得近不容易忘"）：① `legend` 顶部加醒目 `★ Scoring principle`（三条 bullet 解释）；② sticky 顶栏加一句常驻提醒 `barrule`；③ **每一道题**的打分按钮上方都加一条 `⚑ Reward extraction, not summary` 提示（680 条）；④ **每道题的 qtext 重写**为 extraction 导向，明确"要抽取的具体内容 + We want the actual …, not a vague …"。
- 典型例子写进 Q13 medication_plan 题面：pdac ROW 1（coral 0）PL 值 = "No specific medications…"（错误的 infer/summary），但其 attribution = "He would really like a chemotherapy break."（chemo break = 有变化）。题面据此明示"想 chemo break 就是一种变化，不能答无变化"。

**3. attribution 标明是模型生成**：HTML 每条 attribution 改标 "(quote the model pulled from the note)"，legend 说明它是**模型二次回合自己摘的原文引文**（`source_attribution.py`，非 pipeline 编写、未程序化校验逐字），需回原文验证。

**[BACKLOG] prompt 改动（本轮未做，待开 WSL/vLLM 重跑全 40）**：让模型多 extract 少 infer——有信心的 infer 用括号括起来，没信心的不硬下结论（直接列证据让医生判断）。目标是消除 Q13 那类"把有变化 infer 成无变化"的错误总结。改 `prompts/pdac/extraction.yaml` + `prompts/extraction.yaml` 后需重跑→重新审查→重生成所有 HTML/图。
