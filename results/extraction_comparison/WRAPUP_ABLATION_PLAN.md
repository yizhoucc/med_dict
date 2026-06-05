# Wrap-up 计划：Pipeline vs Baseline 消融评估（讨论稿）

> 状态：**讨论稿 v0.1**（2026-06-04）。这是用来和用户对齐方法论的底稿，不是最终结论。
> 目标：用一个**公平但聚焦**的对比，清晰证明 inference harness（多阶段 + gates + hooks）相对裸模型单 prompt 的增益。

---

## 0. 一句话论点（claim）

> **同一个基座模型（Qwen2.5-32B-Instruct-AWQ）、同一套字段 schema、同样的提取要求，唯一的变量是「有没有 harness」。Harness（多阶段提取 + 5 道 gate + 40+ POST hook）系统性地修掉了裸模型单 prompt 会犯的一类一类错误。**

这是一个**消融实验（ablation）**：自变量 = harness，因变量 = 每个字段的对错。基座模型、schema、解码参数全部对齐。

---

## 1. 实验设计：控制变量

| 维度 | Pipeline (PL) | Baseline (BL) | 是否对齐 |
|------|---------------|---------------|----------|
| 基座模型 | Qwen2.5-32B-Instruct-AWQ | 同 | ✅ 对齐 |
| 推理后端 | vLLM (greedy) | 同 | ✅ 对齐 |
| 输出格式 | JSON (18 字段) | JSON (同 schema) | ✅ 对齐 |
| 字段 schema | 18 字段 | 同 18 字段 | ✅ 对齐 |
| 测试集 | 40 annotated (20 breast + 20 PDAC) | 同 | ✅ 对齐 |
| **多阶段提取** | Phase1→Phase2→Plan，跨 prompt 上下文注入 | ❌ 一次性 single prompt | ⬅ **变量** |
| **5 道 gate** | FORMAT/SCHEMA/IMPROVE/FAITHFUL/TEMPORAL | ❌ 无 | ⬅ **变量** |
| **40+ POST hook** | 全部 | ❌ 无 | ⬅ **变量** |
| **医学知识/规则注入** | 药物白名单、术语词典、分期推断规则等 | ❌ 不告诉模型这些规则 | ⬅ **变量** |

### 1.1 关于「同 prompt」的论证（用户的核心 trick）

> 用户原话："我们给 BL 同样的 prompt，但不让 BL 知道那些信息和规则。"

- BL 拿到的是**同一份字段 schema**（`BASELINE_PROMPT_JSON`，`baseline_extraction.py:37`），逐字段定义和 PL 用的字段定义一致。
- BL 拿不到的是 harness 里**编码进 hook/gate/stage 的领域知识和纠错逻辑**——比如"axillary LN 是 regional 不是 distant""计划字段要剔除已完成项""current_meds 只保留肿瘤药白名单内的"等。
- **论证逻辑**：这些规则不是 prompt 能一句话塞进去并让 8B/32B 一次性正确执行的。把它们做成多阶段 + 后处理，正是 harness 的价值所在。BL 代表"只靠一个好 prompt 能做到的上限"，PL 代表"加上工程化纠错后的结果"。
- **诚实边界**：我们不偷偷给 PL 更多原始信息（两者看的 note 原文完全一样）。差异只来自 harness 的处理。这一点必须在 doc 里写明，否则就不是公平消融而是注水。

---

## 2. 评审方法：二元选择（binary）

每个 sample 的每个字段，评审员只做一个二元判断：

> **对该字段，PL 和 BL 哪个更忠实/更完整/更对？**
> 选项：`PL 更好` / `BL 更好` / `打平`（两者都对或都错且程度相同）

- 不打 1–5 分（减少评审主观噪声，也避免"都还行"的中庸分淹没差异）。
- 评审员**盲审**：A/B 标签随机化，不告诉评审哪个是 pipeline。
- 评审依据：原文 note 是 ground truth，对照 annotated 标注。

### 2.1 为什么二元
- 信号清晰：直接数 PL 赢多少字段、BL 赢多少字段。
- 抗噪：评审员之间对"3 分还是 4 分"分歧大，但对"哪个对"分歧小。
- 好讲故事：最终一张表——"PL 在 N 个字段类别上胜出，BL 仅在 M 个胜出"。

---

## 3. 聚焦的错误类别（评审维度 = hook 的设计目标）

这是 trick 的第二层：**评审维度直接对齐 harness 设计要解决的错误类别**。每一类都是 BL 单 prompt 结构上做不到、而 PL 用某个 gate/hook 专门解决的。

| # | 错误类别 | BL 为什么会犯 | PL 用什么解决 | 代码位置 |
|---|----------|---------------|---------------|----------|
| 1 | **JSON / schema 失败** | 一次性生成 18 字段易格式崩 | G1 FORMAT + G2 SCHEMA | ult.py:1173, 1202 |
| 2 | **答非所问**（值没回答字段的问题） | 单 prompt 字段多，模型混淆 | G3 IMPROVE 语义对齐 | ult.py:1234 |
| 3 | **幻觉 / 套话**（"tolerating well"等原文没有的） | 无核查机制 | G4 FAITHFUL + POST-DRUG-VERIFY + POST-MEDS-CROSSCHECK + POST-REFERRAL-VALIDATE + POST-STAGE-VERIFY-NOTE | ult.py:1327; run.py:2673, 3003, 1321, 2208 |
| 4 | **时态泄漏**（已完成项混进 plan） | 不区分过去/计划 | G5 TEMPORAL + POST-MEDS-STOPPED + plan 过滤 | ult.py:1437; run.py:2975 |
| 5 | **字段串扰**（影像混进 procedure，基因混进 referral 等） | 单 prompt 无字段隔离 | filter_procedure_plan + POST-*-FILTER 系列 | ult.py:1049; run.py:1957, 1898 |
| 6 | **遗漏**（信息只在非 A/P 段，BL 漏掉） | 单 prompt 注意力分散 | 全文二次检索 hook：POST-*-SEARCH / SUPPLEMENT + Referral/Genetic_Results 全文 pass | run.py:1416, 1481, 1771, 1979 |
| 7 | **临床一致性错误**（Stage IV 但 met=No；HER2 矛盾等） | 无交叉校验 | POST-STAGE/DISTMET/GOALS/RESPONSE/HER2/ER 交叉校验 | run.py:2131–3351 |
| 8 | **受体状态不全**（breast ER/PR/HER2 缺项） | 不强制三项全写 | POST-ER/HER2/RECEPTOR 系列 | run.py:3067–3351 |

> 评审时按这 8 类逐字段判定。预期：类别 1–8 上 PL 系统性胜出；少数字段（如曾经的 Referral）打平——因为我们已把 Referral prompt 简化到和 BL 一样。

---

## 4. 已知会打平/BL 不输的地方（诚实披露）

为了让对比可信，主动写明 PL 不占优或会输的点：

- **Referral**：已把 PL 的 Referral prompt 简化成和 BL 一样的一句话（`prompts/plan_extraction.yaml`），所以这个字段两者应基本打平。我们**不**把它包装成 PL 胜出。
- **Breast ROW 2 肝囊肿误判为转移**（Stage IV）：这是 PL 的一个已知错误，BL 在此处反而对。如实记录，算 BL 赢 1 个。
- **Breast Genetic_Testing_Results 的 IHC 污染**：模型仍会把 ER/PR/HER2 病理混进来，hook 尚未清。如实记录。

> 这些"BL 赢"的点恰恰增强了消融的可信度——证明我们不是在选择性报喜。

---

## 5. 交付物（5 个文件，已在 `results/extraction_comparison/`）

1. `baseline_extract_breast_json.txt` — BL 乳腺 JSON 提取（40/40 valid JSON）
2. `baseline_extract_pdac_json.txt` — BL 胰腺 JSON 提取
3. `v31_breast_referral_fix.txt` — PL 乳腺提取（最新，Referral 简化后）
4. `v32_pdac_referral_fix.txt` — PL 胰腺提取（最新）
5. `pipeline_vs_json_baseline_comparison.txt` — 字段级对比分析

> ⚠️ **待办**：第 5 个对比文件是基于 Referral 简化**之前**的 PL 跑的。要不要用最新结果重生成，需要在讨论里定。

---

## 6. 最终呈现（建议）

1. **一张总表**：8 个错误类别 × {PL 赢 / BL 赢 / 打平} 字段计数。
2. **JSON 可靠性**：BL 40/40 valid（说明 32B 上 format gate 价值低——诚实写出来，把功劳归给内容类 hook 而非格式类 gate）。
3. **典型案例对照**：每个错误类别挑 1–2 个 sample，并排展示 BL 错 / PL 对，附原文出处。
4. **消融结论**：harness 的增益主要来自内容类 hook（类别 3/5/6/7/8），而非格式类 gate（类别 1）。

---

## 7. 决策（2026-06-04 已和用户对齐）

- [x] **A. 评审单位** → **字段 × sample**（最细粒度，40 sample × 18 字段，每格判 PL/BL/打平）。最有说服力。
- [x] **B. 谁评审** → **主 Claude 手动盲审**，按 CLAUDE.md 规则逐 sample 逐字段读原文判定，不写脚本、不用 Agent、不用 grep。
- [x] **C. 第 5 个对比文件** → **用最新结果重生成**（基于 v31/v32 referral_fix 最新 PL 输出）。
- [x] **D. 半消融中间档** → **不做**，只保留纯 BL vs 纯 PL 两端。
- [x] **E. PDAC vs Breast** → **分开报 + 合并总表**都给。
- [x] **F. limitation** → **写**，见第 8 节 threats-to-validity。

## 8. Threats to Validity（回应审稿人）

1. **"同 prompt 不同 harness 是否公平？"**
   - BL 和 PL 看到的 note 原文**完全一样**，字段 schema 定义**逐字一致**。差异**只**来自 harness 的多阶段处理与后处理，没有给 PL 喂额外原始信息。
   - harness 编码的规则（白名单、分期推断、时态剔除）不是一句 prompt 能让模型一次性可靠执行的——这正是工程化纠错的价值，也是消融要测的东西。BL 代表"单 prompt 上限"，不是 strawman。

2. **"是不是只挑 PL 赢的维度？"**
   - 评审是**全 18 字段全覆盖**，不是只看 8 个 hook 类别。BL 赢的字段（Referral 打平、Breast ROW2 肝囊肿、基因结果 IHC 污染）如实计入。

3. **"格式类 gate 算不算注水？"**
   - 不算。BL 在 32B 上 40/40 valid JSON，format gate（类别1）在本模型上**无价值**，已明确披露，增益归内容类 hook。

4. **"40 sample 够不够？"**
   - 这是 CORAL 全部 annotated held-out 测试集（带专家 BRAT 标注），不是抽样。Dev set（200 unannotated）只用于迭代，未进评估。

## 9. 执行进度（review doc 任务记忆）

> 实际逐 sample 评审写入 `pipeline_vs_json_baseline_comparison.txt`（重生成版）。本节只记里程碑。

- [ ] 重生成对比文件 header + 方法论
- [ ] Breast 20 sample 字段级盲审
- [ ] PDAC 20 sample 字段级盲审
- [ ] 合并总表（8 类别 + 全字段胜负计数）
- [ ] 典型案例对照
