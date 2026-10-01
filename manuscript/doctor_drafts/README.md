# 医生版论文草稿归档与对比审查

## 文件状态

- 原稿：[`20260929_Manuscript_JAMA.docx`](20260929_Manuscript_JAMA.docx)
- 来源：`/Users/yizhoucc/Downloads/20260929_Manuscript_JAMA.docx`
- 归档方式：原文件已移动到本目录，Downloads 中的副本已删除。原稿内容未修改。
- 文档作者元数据：Yuqing Wang。文档包含 11 条批注，没有 tracked changes。
- 渲染结果：7 页，约 2,327 个英文单词。正文有 1 个表格，没有正式插图。

## 总体判断

[已查证] 医生稿和我们现有稿件的核心故事基本一致：先通过 physician-guided human-in-the-loop 开发乳腺癌 extraction harness，再把乳腺癌阶段积累的规则迁移到 PDAC，由 ChatGPT 帮助整理 review rubric，Qwen reviewer 支持开发阶段审查，最后由独立肿瘤科医生评价冻结后的系统。

两份稿子的主要区别在写作视角。医生稿从临床问题出发，强调长病历、copy-forward、时间线混乱、隐私和本地部署。我们的 [`WORKSHOP_PAPER_DRAFT.md`](../../WORKSHOP_PAPER_DRAFT.md) 更强调 inference harness 的技术组成、跨癌种适配、统计分析、字段级结果和研究边界。

医生稿适合提供临床叙事和简洁语气。现有稿件更适合作为事实、数字和方法细节的主版本。建议合并，不建议用医生稿整体替换现有稿件。

## 医生稿写得好的地方

1. 临床问题讲得更自然。Introduction 对 oncology note 的困难描述很具体，包括历史与当前治疗混在一起、copy-forward 结果、复杂时间线、怀疑性发现和当前疾病状态之间的关系。这部分比我们现在偏文献综述式的开头更像临床作者写的论文。
2. 故事线短而清楚。breast human-in-the-loop、PDAC model-assisted adaptation、独立 oncologist A/B evaluation、patient letter exploration 四步已经出现，没有按项目实际走过的弯路来叙述。
3. `inference harness` 的定义直接。医生稿把它解释为围绕冻结模型的任务顺序、检查和 clinical rules，读者容易理解。
4. harness component 表格有用。它把 field-specific extraction、dependency-aware context、verification、temporal filtering、clinical rules 和 source attribution 分别对应到 failure mode。这比连续技术段落更容易读。
5. Results 已经抓住主结果。乳腺癌 282:54:504、总体 443:77:839、85.2% directional preference 和 adjusted OR 5.70 都写进去了。

## 医生稿与现有稿件的强调差异

| 主题 | 医生稿 | 现有稿件 | 建议 |
|---|---|---|---|
| 论文入口 | 临床病历难读、非结构化数据、隐私和本地部署 | inference harness、跨癌种迁移和同模型消融 | Introduction 以医生稿为主，再补我们的 literature gap |
| 主贡献 | 把 expert feedback 变成 extraction workflow | 集成 routing、gates、hooks、logging、attribution，并展示跨癌种适配 | 保留两者，临床问题在前，技术定义随后 |
| patient letter | 标题和封面仍明显强调 patient education | 明确作为 downstream exploratory analysis | 以 extraction 为主标题，letter 放次要结果和 Discussion |
| PDAC 故事 | model-assisted，但有时写成较宽泛的 "agent" 或模型自适应 | 区分 ChatGPT rubric synthesis、Qwen reviewer 和 investigator-controlled implementation | 使用我们的精确定义，避免 "self-learning" 或 autonomous adaptation |
| 结果 | 主要报告 aggregate preference 和 OR | 另有 inter-rater agreement、core fields、note-level sensitivity、tie 解释 | Results 以我们的完整结果为准，再按医生稿的简洁程度压缩 |
| 研究边界 | 尚未写 Discussion，因此几乎没有 limitation | 明确写出无 component ablation、tie 含义不明、只有 3 位医生、benchmark-informed development | 这些边界必须保留 |
| 相关工作 | 只有批注中的几篇参考文献 | 已有 20 篇核查后的 references 和逐项定位 | 使用现有 reference set，吸收医生新增的临床文献线索 |

## 必须修正的事实问题

1. 标题与主要证据不匹配。`Simplifying oncology terminology: Leveraging Large Language Models for patient education` 把 patient education 放成主任务，但本研究最强的证据是 structured extraction。letter 只有 1 位医生、20 个乳腺癌病例的探索性评价。
2. 首页摘要写 `aims to train open-source LLMs`。项目没有训练或 fine-tune 模型，权重始终冻结。应写成 `engineer an inference harness around a locally deployed open-weight LLM`。
3. Dataset 段把 40 个 annotated notes 写成开发与保留测试的全部数据，并称 breast notes 用于开发、PDAC notes 用于评估。实际开发使用额外的 56 个乳腺癌 notes 和 100 个 PDAC notes；40 个 expert-annotated notes 用于最终 comparison。现有稿件还披露了后期 benchmark-informed revision，因此不能写成完全 untouched external validation。
4. Oncologist evaluation 写成 3 位医生都评价 breast 和 PDAC，并完成相同的 280 个 comparison。实际是 3 位医生评价 breast，2 位评价 PDAC；Kevin 的 PDAC 缺 `p7 / lab_plan`，所以五份评价共 1,359 项。
5. PDAC Results 写成 `3 physicians` 给出 161:23:335。该 subtotal 来自 2 位医生。
6. 小标题 `Auto Model-assisted development of breast cancer notes` 应为 pancreatic cancer notes。
7. Abstract 仍是占位数字，JAMA structured abstract 的 Objective、Design、Main outcome 等栏位为空。
8. Discussion 和 Conclusion 基本为空。当前文档还没有完成结果解释、相关工作比较和 limitation。
9. References 尚未形成 reference list。11 条批注中有文献链接和待确认问题，但没有正式编号和正文引用。

以上数字已与 [`THREE_CLINICIAN_ANALYSIS.md`](../../results/extraction_comparison/clinician_ratings/THREE_CLINICIAN_ANALYSIS.md) 和 [`ADJUSTED_ANALYSIS.md`](../../results/extraction_comparison/clinician_ratings/ADJUSTED_ANALYSIS.md) 对照。

## 对 11 条批注的处理建议

- 关于 56 个 breast development notes：可以保留。项目记录为约 15 个 cycles、56 个 dev samples。
- 关于 18 个 PDAC rounds：项目历史记录为 18 个 development cycles，但可比较的 trajectory 目前只保留 D1-D9 和 F2-F4，共 12 个 checkpoints。正文可以写项目开发记录中的 18 cycles；如果配 trajectory 图，需要明确图只展示有同口径 archived review 的 12 个 checkpoints。依据见 [`results/pdac_development_trajectory/ANALYSIS.md`](../../results/pdac_development_trajectory/ANALYSIS.md)。
- 关于 `260`：这是 260 个 applicable note-field comparisons，不是 260 个 notes。应始终写完整单位。
- 关于 plan fields 是否只看 Assessment and Plan：大多数 plan prompts 以提取出的 A/P 为主要输入。启用 tool calling 时可以搜索完整 note；Referral 和 Genetic_Testing_Results 固定从完整 note 提取。因此 `primarily from the Assessment and Plan section` 是准确说法，不应改成绝对限定。
- 关于 statistical analysis：共有 5 份 clinician-by-cancer evaluations。Pooled counts 只作描述。主要推断分析排除 ties，使用按 note 聚类的 logistic GEE，并调整 evaluator 和 cancer type。OR 表示 non-tie judgment 中偏好 harness 的 odds，不表示临床正确性的 odds。
- 批注中的文献适合作为候选来源。正式合并时应去掉 `utm_source=chatgpt.com`，使用 DOI、PubMed 或期刊正式链接，并与现有 reference list 去重。

## 排版和完成度

[已查证] 当前 DOCX 仍有明显工作稿痕迹：

- 第 1 页重复标题，并保留 `Harness workflow`、`Agent PDAC`、`JAMA version` 等零散笔记。
- 第 2 页的 structured abstract 只有栏目名。
- 第 4 页有一处可见删除线，表格被分页拆开，第一行在第 5 页续接。
- 第 5 页和第 6 页仍有黄色高亮。
- 第 7 页 Discussion 和 Conclusion 为空，只留一条 clinical trial 链接。
- 英文有较多语法和术语问题，例如 `HIPPA`、`fully loyal`、`models may generated`、`3 physicians preferer`。内容可以保留，但需要完整 line edit。

## 推荐的合并方案

1. 标题继续以 extraction 和 inference harness 为主。可用：`A Clinician-Guided Inference Harness for Structured Oncology Note Extraction With Model-Assisted Cross-Cancer Adaptation`。
2. Introduction 采用医生稿的临床问题顺序，保留我们的 literature positioning 和谨慎 novelty statement。
3. Methods 使用现有稿件作为事实底稿。保留医生稿的 harness component 表格，并压缩重复解释。
4. Results 使用现有稿件的完整数字。主文至少保留 overall result、breast/PDAC 分层、inter-rater agreement、core-field pattern 和 adjusted analysis。PDAC trajectory 可作为 secondary development evidence。
5. Discussion 以现有稿件为起点，但缩短。重点讨论 harness 为什么在 active medications、metastatic involvement 和 response 上更有优势，同时明确 tie ambiguity 和 component ablation 缺失。
6. patient letter 继续作为 exploratory downstream use。除非后续获得更强的 letter evaluation，不应把它放进主标题或主结论中心。

## 最终评价

[推断] 这份医生稿最大的价值是帮我们校准论文的临床入口。医生没有否定现有故事，反而采用了同一条主线。差别在于他更想先让临床读者明白为什么 oncology note extraction 难、为什么本地 open-weight model 有意义，再介绍 harness。这个顺序值得采用。

[已查证] 目前它还不能作为新的 master draft。关键数字有几处对象写错，Discussion、Conclusion、structured abstract 和 references 尚未完成。最稳妥的做法是保留现有稿件为 master，把医生稿的临床叙事、简洁表达和 component table 合并进去。
