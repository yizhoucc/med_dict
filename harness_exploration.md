# Harness Exploration: med_dict Pipeline 能从 Claude Code Harness 学到什么

> 日期: 2026-04-05
> 目标: 从 pipeline 质量角度（extraction + letter），看 harness 的设计模式能否提升 pipeline 效果

---

## 背景：两个系统的本质区别

Claude Code harness 是交互式 chatbot 的扩展框架（plugin/hook/command）。
med_dict 是固定流程的 LLM pipeline（note → extraction → letter）。

不能直接搬 harness 的交互功能（slash command、SessionStart 注入等），但 harness 里有一些**设计模式**值得研究是否能改进 pipeline 的提取质量。

---

## Harness 设计模式 → Pipeline 映射分析

### 模式 1：POST hooks 外部化

**harness 怎么做的**：
hookify plugin 把验证规则放在独立的 `.local.md` 文件里，每个规则有 name/enabled/event/pattern/action 字段。规则引擎 (`rule_engine.py`) 读取规则文件，逐条匹配执行。添加规则不需要改主代码。

**我们现在**：
47 个 POST hook 全部硬编码在 `run.py` 的约 900 行代码里。每个 hook 是一段 if/regex/白名单逻辑，直接操作 `keypoints` dict。

**能搬到 pipeline 吗**：
能。把 47 个 POST hook 抽出来做成独立的规则配置。好处：
- **实验可控**：在 `exp/*.yaml` 里指定 `post_hooks: [stage-verify, drug-verify, meds-filter]`，不同实验用不同规则组合，做消融实验看哪些规则贡献了质量提升
- **版本追踪更清晰**：每个规则改动是独立 commit，现在改 run.py 的 diff 很难看出改了哪个 hook
- **enable/disable 开关**：临时禁用某个 hook 不用注释代码

**成本**：需要重构 run.py，把 47 个 hook 抽成独立函数 + 配置加载器。工作量中等。

**值不值得做**：**可能值得，但不紧急**。当前 47 个 hook 工作正常（P0=0），主要收益在可维护性和实验便利性，不直接提升质量。如果将来 hook 数量继续增长，或者需要做消融实验，再做。

---

### 模式 2：置信度评分（只对高置信度结果执行操作）

**harness 怎么做的**：
code-review plugin 的 reviewer agent 给每个发现打 0-100 置信度分数。只报告 ≥80 的问题。低于 80 的视为可能的误报，不报告。

**我们现在**：
Gate 4 FAITHFUL 让 LLM 做二值判断——"supported" 或 "contradicted/fabricated"。如果 LLM 判断 "fabricated"，直接清空该字段。没有中间状态。

之前出过问题：Gate 4 把 response_assessment 的合理推断（如从用药+影像推断治疗响应）清空了，因为它"不是 note 里逐字写的"。后来改了 prompt（"拿不准就保留"），但这是用 prompt 措辞来模拟一个 soft 决策，不如直接输出置信度。

**能搬到 pipeline 吗**：
能。改 Gate 4 的 prompt，让 LLM 对每个字段输出三级判断：

```json
{
  "Type_of_Cancer": {"action": "keep", "confidence": 95},
  "response_assessment": {"action": "empty", "confidence": 40},
  "goals_of_treatment": {"action": "keep", "confidence": 85}
}
```

然后用阈值控制：
- confidence ≥ 80 → 执行 action（keep 或 empty）
- confidence < 80 → 保留原值（不敢确定就不动）

**好处**：
- 解决 Gate 4 "太严格清空合理推断" 的历史问题
- 阈值可调（`exp/*.yaml` 里配 `faith_threshold: 80`），不同实验用不同阈值
- 审查时可以看置信度：低置信度的字段优先人工检查

**成本**：改 Gate 4 的 prompt + 解析逻辑。改动不大，但需要验证 Qwen 32B 能不能稳定输出这个格式。

**值不值得做**：**值得尝试**。直接解决一个真实问题（Gate 4 误修剪），而且实现成本不高。可以先在 8 个 sample 上测试格式稳定性。

---

### 模式 3：交叉验证（独立二次确认）

**harness 怎么做的**：
code-review plugin 的步骤 5：对每个 agent 发现的问题，启一个独立 subagent 做二次验证。只有两个 agent 都同意的问题才最终报告。

**我们现在**：
Gate 3 IMPROVE 和 Gate 4 FAITHFUL 串行执行。Gate 3 修改了字段值后，Gate 4 再验证忠实度。但 Gate 4 看到的是 Gate 3 修改后的值，不知道原始值是什么——如果 Gate 3 改错了，Gate 4 可能认为新值"看起来合理"就放过了。

**能搬到 pipeline 吗**：
能。两种方式：

**方式 A：Gate 4 同时看原始值和 Gate 3 修改后的值**：
```
Gate 4 prompt: "原始提取是 X，改进后是 Y。哪个更忠实于原文？"
```
让 Gate 4 做仲裁而不是单方面判断。

**方式 B：独立的二次提取对比**：
同一个字段用不同的 prompt 提取两次，只保留两次结果一致的部分。但这会翻倍 LLM 调用次数。

**值不值得做**：**方式 A 成本低，值得尝试**。方式 B 太贵。

---

### 模式 4：自动重试修复（失败字段单独重跑）

**harness 怎么做的**：
ralph-wiggum 用 Stop hook 拦截退出，把同一个 prompt 重新喂入，让 agent 看到上一轮的结果继续改进。循环直到满足完成条件或达到 max_iterations。

**我们现在**：
- V1 pipeline 的 faithfulness 重来就是一种粗糙版本：整个 prompt 重跑，但可能引入新错误
- V2 改成修剪（不重跑），避免了新错误，但也放弃了"重新提取可能更好"的机会
- 47 个 POST hook 是补丁式修复，不涉及重新调用 LLM

**能搬到 pipeline 吗**：
能。**针对性重试**：当 POST hook 发现特定错误模式时，只重跑该字段，且在 prompt 里加入错误提示：

```
POST-STAGE-VERIFY 发现 "Originally Stage IIA" 是幻觉
  → 重新调用 Cancer_Diagnosis prompt，附加指令：
    "DO NOT use 'Originally Stage X' unless the note explicitly states the original stage number."
  → 只重跑这一个字段，不影响其他字段
  → max_retries = 2，防止无限循环
```

**V1 的"整个重做"vs 这个"单字段重试 + 错误提示"的区别**：
- V1：整个 prompt 重跑 → 可能修好 A 但搞坏 B
- 这里：只重跑出错的字段 → 不影响已经正确的字段
- 附加错误提示 → 针对性引导，避免再犯同一个错

**成本**：每次触发重试多 1 次 LLM 调用。但只有检测到已知错误模式才触发，不会每个 sample 都多跑。

**值不值得做**：**值得在高频错误模式上试**。比如 POST-STAGE-VERIFY（Stage 幻觉）、POST-DRUG-VERIFY（幻觉药物）这两个 hook 检测到问题后，与其直接 regex 修补，不如让 LLM 在提示下重新提取，可能给出更完整准确的结果。

---

### 模式 5：多维度并行（不同角度的独立提取，对比合并）

**harness 怎么做的**：
code-review 启 4 个并行 agent，各关注不同维度（CLAUDE.md 合规 × 2 + bug 检测 × 2），最后合并结果。

**我们现在**：
Phase 1 的 6 个 prompt 已经是"各关注不同维度的独立提取"。但每个维度只有一次提取机会。

**能搬到 pipeline 吗**：
可以但不太值得。同一个字段用两个不同 prompt 提取，对比取交集或并集。理论上能提高准确率，但：
- LLM 调用次数翻倍（目前 61 个 sample × 10 个 prompt = 610 次调用）
- 两次提取结果的合并策略不好定义（取并集会增加幻觉，取交集会增加遗漏）

**值不值得做**：**不值得**。ROI 太低。

---

## 总结：值得尝试的改进

| # | 改进 | 来源模式 | 预期收益 | 成本 | 建议 |
|---|------|----------|----------|------|------|
| 1 | Gate 4 加置信度评分 | code-review 置信度过滤 | 减少 G4 误修剪 | 低 | **先做** |
| 2 | 检测到已知错误模式后，单字段重试+错误提示 | ralph-wiggum 自动迭代 | 减少 Stage 幻觉/药物幻觉 | 低 | **先做** |
| 3 | Gate 4 同时看原始值和 G3 修改值做仲裁 | code-review 交叉验证 | 减少 G3→G4 串行误传播 | 低 | 可以试 |
| 4 | POST hooks 外部化为规则配置 | hookify 规则引擎 | 可维护性、消融实验 | 中 | 不紧急 |
| 5 | 多 prompt 对比合并 | code-review 多 agent | 理论上提高准确率 | 高 | 不做 |

**优先尝试 #1 和 #2**，因为它们直接解决已知质量问题（G4 误修剪、Stage/药物幻觉），成本低，可以在 8 个 sample 上快速验证。
