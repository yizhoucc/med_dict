# med_dict - 医学肿瘤笔记结构化提取与患者解释生成

## 核心目标
从肿瘤科医生的临床笔记中提取结构化信息，生成**面向患者**的通俗总结。

四个不可妥协的原则（按优先级）：
1. **精确忠实** — 绝对不能幻觉，不能编造任何信息。宁可少说，不可说错。
2. **不遗漏** — 笔记中的重要临床信息都要覆盖到，不能丢掉关键内容。
3. **简单词汇** — 8 年级英语水平（eighth-grade reading level），患者能看懂。
4. **通俗易懂** — 避免医学术语；如果必须使用，要附带通俗解释。

这四个原则贯穿整个 pipeline 设计：
- Gate 4 (FAITHFUL) 的"拿不准就保留"策略 = 平衡原则 1 和 2（防幻觉 vs 防遗漏）
- Gate 3 (IMPROVE) 的具体化+语义对齐 = 服务原则 2（不用模糊词敷衍 + 不答非所问）
- 解释生成步骤 = 服务原则 3 和 4（通俗化）
- 提取 prompt 中的具体示例和边界情况处理 = 服务原则 1（精确引导模型）

## 项目概述
使用 LLM (Llama 3.1 8B Instruct) 从肿瘤科临床笔记中提取结构化信息，并生成面向患者的通俗解释。

## 数据
- **CORAL 数据集**：乳腺癌肿瘤科门诊笔记（去标识化）
- **医学词典**：`data/formaldef.txt`（9369 术语），经 8 年级词汇过滤
- HuggingFace token 存于 `hf.token`（勿提交）

## 核心文件
- `run.py` - 实验入口，根据 config 的 `extraction.pipeline` 选择 v1 或 v2 pipeline
- `ult.py` - 工具库：模型推理（含 KV Cache 分叉）、JSON 修复、两个 pipeline 实现
  - `build_base_cache()` - 构建 KV Cache
  - `extract_and_verify()` - V1 pipeline（3-gate: format, faithfulness 重来, temporal）
  - `extract_and_verify_v2()` - V2 pipeline（5-gate: format, schema, improve 合并, faithfulness 修剪, temporal）
  - `extract_schema_keys()` - 从 prompt 的 JSON schema 解析期望 keys
  - `run_model_with_cache_manual()` - 手动 token-by-token 生成，支持 eos_token_id 列表
- `notebooks/` - 早期工作：医学术语简化 + 可读性评估
- `plots/` - 可读性指标对比图（SVG）

## Pipeline 架构

### 整体流程（v7abc 两阶段提取）
1. **Assessment/Plan 提取** - regex 优先，LLM fallback（3 次重试 + LLM 验证）
2. **Phase 1 关键点提取** (6 prompts) - 从全文独立提取：就诊原因、癌症诊断、检验、检查发现、当前用药、治疗变化
3. **Phase 2 依赖推理提取** (2 prompts) - 注入 Phase 1 结果作为上下文，提取：治疗目标、疗效评估
4. **计划提取** (plan_extraction_prompts) - 从 A/P 段提取：用药/手术/影像/检验/基因检测/转诊/随访/预前计划
5. **解释生成**（可选）- 从 keypoints 生成 8 年级英语水平的患者信

### V2 Pipeline（默认，5 Gate — v7abc 合并版）
每个 gate 独立修一个问题，修剪而非重来：

| Gate | 功能 | 触发条件 | 关键行为 |
|------|------|----------|----------|
| 1 FORMAT | JSON 解析修复 | json.loads 失败 | LLM 重格式化 |
| 2 SCHEMA | Key 名验证 | 输出 keys 与 prompt schema 无交集 | LLM 修正 key 名 |
| 3 IMPROVE | 具体化+语义对齐（合并） | verify=True | 替换模糊词（条件触发）+ 检查每个值是否回答了字段定义的问题 |
| 4 FAITHFUL | 忠实度修剪 | verify=True | "拿不准就保留"策略，只清空明确矛盾/捏造的值 |
| 5 TEMPORAL | 时态过滤 | verify=True 且 key ∈ PLAN_KEYS | 删除过去/已完成项 |

Gate 3 IMPROVE 设计要点（合并自原 G3 SPECIFIC + G4 SEMANTIC）：
- 合并为一次 LLM 调用，省 18 次 LLM 调用/笔记
- 先修复模糊词（条件触发：仅当检测到 vague terms 时）
- 再检查语义对齐：值是否真的回答了字段定义的问题
- 例如：goals_of_treatment 写了"follow-up"（就诊目的）而非"palliative"（治疗意图）
- 例如：response_assessment 写了"will have radiation"（未来计划）而非"CT stable"（当前响应）

Gate 4 FAITHFUL 设计要点：
- Prompt 指示 "KEEP if supported or reasonably inferable, ONLY empty if clearly CONTRADICTS or fabricated"
- 之前用 "is it explicitly stated?" 太严格，导致 response_assessment 等合理推断被清空
- 所有 gate 都验证 key overlap 防止 schema 泄漏（如 `{"faithful": true}`）

### 跨 Prompt 信息传递（v7abc）
- Phase 1 的 Cancer_Diagnosis（stage/metastasis）、Current_Medications、Clinical_Findings 结果注入 Phase 2
- Treatment_Goals 可利用 stage 信息正确判断 curative vs palliative
- Response_Assessment 可利用用药+发现信息判断是否在治疗、有无响应
- 上下文 ~100-200 tokens，不影响生成质量

### V1 Pipeline（对比用，3 Gate）
- faithfulness 失败后整个重来（可能引入新错误）
- 无 schema 验证、无 specificity 检查

### 15 样本对比结论
- V2 速度与 V1 持平
- V2 内容更丰富（+7%），准确性更高（减少 "No X planned" 误报）
- V2 的 faith-trimmed 替代了 V1 的 re-extracted（修剪 vs 重做）

## Logging
V2 在 `run.log` 中记录每个 gate 的详细行为：
- `[EXTRACT]` 原始提取
- `[G1-FORMAT]` `[G2-SCHEMA]` `[G3-IMPROVE]` `[G4-FAITH]` `[G5-TEMPORAL]`
- 每个 gate 记录：ok / 字段级 before→after / EMPTIED / FAILED / REJECTED

## 已知问题与待改进

### 2026-03-01 百行审查结论（100/100行有问题）
审查报告：`results/default_20260301_084320/review.md`

**~~Pipeline Bug（P0）~~ → 审查误报（2026-03-01 已排查）**：
- Row 16 ~~数据串行~~ 误报：keypoints 与自身 note_text 一致，审查 agent 对齐错误。真实问题是 LLM 重复退化（Treatment_Summary 中 "docusate" 重复）
- Row 89 ~~数据重复~~ 误报：coral_idx 分别为 227 和 228，不同患者。审查 agent 对齐错误
- ~~潜在风险：KV Cache in-place 修改（DynamicCache）~~ → 已修复：所有调用前 `clone_cache(base_cache)`

**已通过 prompt 改进的问题（2026-03-01 更新 `prompts/extraction.yaml`）**：
- `Patient type`：~35行错误。新增详细判断规则（首次 med onc 会诊 = New patient）
- `Type_of_Cancer`：~65行不完整。强制要求 ER/PR/HER2 三项全写；加 HER2 靶向药推断规则
- `Stage_of_Cancer`：~45行遗漏。改为允许从肿瘤大小+LN推断分期，不轻易写"Not mentioned"
- `current_meds`：~55行时态混乱。新增 CURRENT/PLANNED/PAST 三分法，禁止计划药物和已停药物
- `supportive_meds`：~25行分类错误。明确禁止把过敏原当药物
- `lab_summary`：~35行答非所问。明确排除影像/病理/基因组检测结果
- `response_assessment`：~70行答非所问。新增更多 BAD 示例（手术恢复、Oncotype、风险评估、副作用变化）
- `goals_of_treatment`：~30行不精确。新增决策树（早期+辅助→curative，新辅助→curative，Stage IV→palliative）

**仍需进一步改进的问题**：
- `findings`：~55行不完整/答非所问（写症状而非客观发现），可能需要在 prompt 中加更多指导
- `Referral`：~15行遗漏原文明确的转诊。可在 plan_extraction prompt 中加关键词搜索指导
- `Genetic_Testing_Plan`：~15行遗漏。可在 plan_extraction prompt 中加更多触发关键词（Oncotype, BRCA, germline, molecular profiling）
- `Distant Metastasis`：~15行错误（axillary LN 被标为 distant）。已在 schema 描述中加注"axillary LN = REGIONAL, not distant"
- 通俗化（8 年级英语水平）：暂未实现，当前输出仍有大量医学术语，后续在解释生成步骤处理

## 模型配置
- 模型：`meta-llama/Llama-3.1-8B-Instruct`，bfloat16，单 GPU
- Chat Template：通过 `ChatTemplate` 类抽象（`ult.py`），支持 llama3 和 mistral，通过 `exp/default.yaml` 的 `model.chat_template` 配置
- KV Cache 分叉：笔记编码一次，多任务复用缓存。所有 gate 调用前 `clone_cache()` 防 DynamicCache in-place 修改
- 提取用贪婪解码，`max_new_tokens: 768`（CoT 需要更多空间）

### 2026-03-01 架构重构
1. **ChatTemplate 抽象**：`ult.py` 和 `run.py` 中所有硬编码的 Llama chat token 替换为 `ChatTemplate` 类。切换模型只需改 config 的 `chat_template` 字段。
2. **字段拆分**：`prompts/extraction.yaml` 从 4 个 prompt（多字段）拆为 8 个 prompt（1-4 字段/个），降低 8B 模型的认知负担：
   - `Reason_for_Visit`（4 字段，保持不变）
   - `Cancer_Diagnosis`（Type + Stage + Metastasis，原 What_We_Found 的一部分）
   - `Lab_Results`（lab_summary，独立出来防幻觉）
   - `Clinical_Findings`（findings，独立出来防答非所问）
   - `Current_Medications`（current_meds，加 CoT 做时态判断）
   - `Treatment_Changes`（recent_changes + supportive_meds）
   - `Treatment_Goals`（goals_of_treatment，加 CoT 做决策树推理）
   - `Response_Assessment`（response_assessment，加 CoT 先判断是否已开始治疗）
3. **Chain-of-Thought**：3 个最易出错的字段（Current_Medications、Treatment_Goals、Response_Assessment）加入 "Think step by step" + "Write reasoning first, then JSON" 指令。`try_parse_json()` 已支持从混合文本中提取 JSON。
4. **KV Cache 防御**：所有 `run_model_with_cache_manual()` 调用前 `clone_cache(base_cache)`，防止 DynamicCache 被 in-place 修改。

## 运行环境
- WSL (Ubuntu 22.04)，通过 `ssh wsl` 从 Mac 访问
- Conda 环境：medllm

## 术语规范
- **sample**：一条临床笔记及其对应的提取结果。不要叫"行"或"row"（容易和 CSV 行号、results.txt 行号混淆）。
- **row_index**：CSV 数据集中的行索引（从 0 开始），对应 `row_indices` 配置中的值。
- **ROW N**：results.txt 中的 "RESULTS FOR ROW N"，N = row_index + 1。
- **coral_idx**：CORAL 数据集的原始编号。
- **当前版本**：v23（第 23 个大版本）。使用 Qwen2.5-32B-Instruct-AWQ。22 个 POST hooks。
- **数据集**：61 个 sample（CORAL breastca_unannotated.csv 的子集）。
- **质量**：P0=0, P1=2 (3.3%), POST hooks 零误触发。详见 `results/v23_audit_report.md`。
- **架构文档**：`PIPELINE_OVERVIEW.md`（给非技术人员看的完整架构说明）。

## 代码规范
- results.txt 保持当前自定义格式，不改为 JSON
- prompt 中的 JSON schema 必须是合法 JSON（注意逗号）
- `exp/` 中的 yaml 是实验配置入口，通过 `extraction.pipeline` 切换 v1/v2

## 工作流程规则

### 生成结果后的质量审查（每次必做）
每次 pipeline 在 WSL 上跑完并下载结果后，必须对每个 sample **逐行手动审查**，不写脚本：
1. **从生成结果出发**：逐个字段读 keypoints 的每个值
2. **往原文找**：到 note_text 中找对应的原文依据，确认是否忠实
3. **结合归因**：如果有 attribution 数据，检查模型给出的归因句子是否真的支持该值
4. **结合 prompt 规则**：对照 extraction.yaml / plan_extraction.yaml 中该字段的 schema 定义和要求
5. **查历史问题**：参考 tracking.md / 之前的 review doc，检查该行该字段之前报过的问题是否已修复
6. **检查新问题**：是否引入了新的错误（幻觉、遗漏、答非所问、时态混淆等）
7. **分类记录**：将发现的问题按 P0/P1/P2 分类，建立 review doc 记录每行每字段的审查结果
8. **汇总模式**：审查完所有行后，汇总反复出现的模式性问题

**关键原则**：
- 不偷懒、不跳过、不用脚本批处理。真正逐行、逐字段地读原文和生成结果，像人类审查员一样。
- **必须通过自然语言方式判断**，不能写代码来辅助审查。用自己的理解去读原文、理解语义、做判断，不要用 python/grep/regex 来做批量检查或自动分类。
- **必须逐字逐句审查**：审查时必须完整阅读 note_text 原文（不能只看关键字段或 grep 关键词），完整阅读 keypoints 每个字段的值，然后在原文中找到对应的依据。不能"扫一眼"就下结论。
- **必须亲自审查，禁止用 Agent**：所有审查必须由主 Claude 亲自完成，不得委托给 Agent。Agent 不理解 prompt 字段定义，会产生误判（如把正确的 current_meds 标为 P0）。每次审查一个 sample row，完整读完原文和所有字段后再进入下一行。
- **触发规则**：只要用户提到"审查"、"review"、"检查结果"，就自动按上述全套流程执行：下载结果 → 逐行逐字段读原文 → 对照 prompt/code → 查归因 → 找问题 → 写 review doc。不需要用户重复说明规则。

### 远程任务监控与通知
当 WSL 上的 pipeline 运行时：
- **每 3 分钟检查一次进度**（不论任务大小）
- **Bark 通知策略**：
  - 小任务（≤5 个 sample）：每 3 分钟 Bark 一次进度
  - 大任务（>5 个 sample）：每 3 分钟检查，自行判断是否 Bark（有重要进展、发现问题、完成时才发）
  - 完成时必须立即 Bark
  - 发现错误/crash 时立即 Bark
- 完成后立即下载结果+开始审查
- Bark URL: `https://api.day.app/mWBcxqxVNZRUzECXxiLxs5/{标题}/{内容}`
