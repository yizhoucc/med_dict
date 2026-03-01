# med_dict - 医学肿瘤笔记结构化提取与患者解释生成

## 核心目标
从肿瘤科医生的临床笔记中提取结构化信息，生成**面向患者**的通俗总结。

四个不可妥协的原则（按优先级）：
1. **精确忠实** — 绝对不能幻觉，不能编造任何信息。宁可少说，不可说错。
2. **不遗漏** — 笔记中的重要临床信息都要覆盖到，不能丢掉关键内容。
3. **简单词汇** — 8 年级英语水平（eighth-grade reading level），患者能看懂。
4. **通俗易懂** — 避免医学术语；如果必须使用，要附带通俗解释。

这四个原则贯穿整个 pipeline 设计：
- Gate 3 (FAITHFUL) 的"拿不准就保留"策略 = 平衡原则 1 和 2（防幻觉 vs 防遗漏）
- Gate 5 (SPECIFIC) 的具体化 = 服务原则 2（不用模糊词敷衍）
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
  - `extract_and_verify_v2()` - V2 pipeline（6-gate: format, schema, faithfulness 修剪, temporal, specificity, semantic）
  - `extract_schema_keys()` - 从 prompt 的 JSON schema 解析期望 keys
  - `run_model_with_cache_manual()` - 手动 token-by-token 生成，支持 eos_token_id 列表
- `notebooks/` - 早期工作：医学术语简化 + 可读性评估
- `plots/` - 可读性指标对比图（SVG）

## Pipeline 架构

### 整体流程
1. **Assessment/Plan 提取** - regex 优先，LLM fallback（3 次重试 + LLM 验证）
2. **关键点提取** (extraction_prompts) - 从全文提取：就诊原因、检查发现、治疗摘要、治疗目标
3. **计划提取** (plan_extraction_prompts) - 从 A/P 段提取：用药/手术/影像/检验/基因检测/转诊/随访/预前计划
4. **解释生成**（可选）- 从 keypoints 生成 8 年级英语水平的患者信

### V2 Pipeline（默认，6 Gate）
每个 gate 独立修一个问题，修剪而非重来：

| Gate | 功能 | 触发条件 | 关键行为 |
|------|------|----------|----------|
| 1 FORMAT | JSON 解析修复 | json.loads 失败 | LLM 重格式化 |
| 2 SCHEMA | Key 名验证 | 输出 keys 与 prompt schema 无交集 | LLM 修正 key 名 |
| 3 FAITHFUL | 忠实度修剪 | verify=True | "拿不准就保留"策略，只清空明确矛盾/捏造的值。丢失的 key 自动从原始提取恢复 |
| 4 TEMPORAL | 时态过滤 | verify=True 且 key ∈ PLAN_KEYS | 删除过去/已完成项 |
| 5 SPECIFIC | 具体化 | verify=True 且含模糊词 | 替换 "staging workup" 等为具体内容 |
| 6 SEMANTIC | 语义相关性 | verify=True | 检查每个值是否真的回答了字段定义的问题，修正"答非所问" |

Gate 6 设计要点：
- 解决的是 Gate 3 无法覆盖的问题：值在原文中有支持（忠实），但放错了字段（答非所问）
- 例如：goals_of_treatment 写了"follow-up"（就诊目的）而非"palliative"（治疗意图）
- 例如：response_assessment 写了"will have radiation"（未来计划）而非"CT stable"（当前响应）
- 把 prompt schema 中的字段定义传给模型，让模型对照定义检查每个值

Gate 3 设计要点：
- Prompt 指示 "KEEP if supported or reasonably inferable, ONLY empty if clearly CONTRADICTS or fabricated"
- 之前用 "is it explicitly stated?" 太严格，导致 response_assessment 等合理推断被清空
- 所有 gate 都验证 key overlap 防止 schema 泄漏（如 `{"faithful": true}`）

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
- `[G1-FORMAT]` `[G2-SCHEMA]` `[G3-FAITH]` `[G4-TEMPORAL]` `[G5-SPECIFIC]` `[G6-SEMANTIC]`
- 每个 gate 记录：ok / 字段级 before→after / EMPTIED / FAILED / REJECTED

## 已知问题与待改进
- `goals_of_treatment`：之前模型常写就诊目的（"follow-up"）而非治疗意图（"palliative"），已通过 prompt 改进 + Gate 6 修复
- `response_assessment`：之前模型常混入未来计划（"will have radiation"）而非当前响应（"CT stable"），已通过 prompt 改进 + Gate 6 修复
- `Type_of_Cancer`：部分行仍抄 CMS code（"Malignant neoplasm of..."），已通过 prompt 要求写具体亚型+受体状态
- `Stage_of_Cancer`：已通过 prompt 要求处理"原始分期+转移复发"场景
- `recent_changes`：模型有时将新变化放入 current_meds 而非 recent_changes
- `Referral`：空值率高，但原文确实很少有明确转诊语句（数据特征非模型问题）
- `Genetic_Testing_Plan`：14/15 空值，属数据特征
- 通俗化（8 年级英语水平）：暂未实现，当前输出仍有大量医学术语，后续在解释生成步骤处理

## 模型配置
- 模型：`meta-llama/Llama-3.1-8B-Instruct`，bfloat16，单 GPU
- KV Cache 分叉：笔记编码一次，多任务复用缓存
- 提取用贪婪解码，生成用采样（temperature=0.3）

## 运行环境
- WSL (Ubuntu 22.04)，通过 `ssh wsl` 从 Mac 访问
- Conda 环境：medllm

## 代码规范
- results.txt 保持当前自定义格式，不改为 JSON
- prompt 中的 JSON schema 必须是合法 JSON（注意逗号）
- `exp/` 中的 yaml 是实验配置入口，通过 `extraction.pipeline` 切换 v1/v2
