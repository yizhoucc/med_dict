# 评分表 · 方法二（自下而上：从 PL/BL 实际输出的真实差异出题）

> 状态：question bank v1（2026-06-05）。基于已审 8 行 + 全 40 扫描的实际输出观察。
> 配套：`RUBRIC_method1_topdown.md`（自上而下）、`pipeline_vs_json_baseline_comparison.txt`。

## 0. 方法与原则

**思路**：不从"设计期待"出发，而是**直接读 PL 和 BL 的实际最终输出**，把当前输出当成定稿，
逐字段比对找出**反复出现的真实差异**，再针对这些已观察到的差异出二元题。
—— 与方法一互补：方法一可能"期待的优势没出现"，方法二只写"实际看到的优势"。

**评测单位 / 二元值 / 入选标准 / 措辞原则**：同方法一（每题全 40 sample 共用，取平均，
只保留 PL 平均 ≥0.65 且 BL 胜 ≤3 的题；题面中性专业；只考真实医疗维度）。

**数据来源**：PL = `pipeline_breast_FIXED.txt` / `pipeline_pdac_FIXED.txt`（修复后定稿）；
BL = `baseline_extract_breast_json.txt` / `baseline_extract_pdac_json.txt`。

---

## 1. 已观察到的反复差异 → 题（附实际案例锚点）

| # | 评测题（中性专业措辞） | 二元判定定义 | 实际观察锚点（已审行） | 出现频度(8行) |
|---|---|---|---|---|
| E1 | **Current-medication signal-to-noise** — 当前用药字段中肿瘤相关用药 vs 无关家庭用药的比例，哪个更聚焦于肿瘤诊疗？ | 无关家庭药更少者判优 | ROW1 BL列22种家庭药/PL空; ROW4 BL17种/PL空; ROW3/6/8 同 | 高(≥6/8) |
| E2 | **Anticancer-drug presence** — 当前在用的抗癌药物是否出现在用药字段中？ | 漏在用抗癌药者判劣 | ROW7 BL漏abraxane+pembro/PL有 | 中(出现即强证据) |
| E3 | **Findings: objective vs subjective** — clinical findings 给出的是客观病理/影像发现，还是主观症状清单？ | 仅主观症状者判劣 | ROW7 BL列"GERD/nausea/fatigue"症状/PL列影像病理; ROW1/3/5/8 BL残缺 | 高(≥5/8) |
| E4 | **Findings completeness** — 关键客观发现（肿瘤大小/进展/受体/淋巴结/远处灶/驱动治疗的合并症）覆盖更全者？ | 漏关键发现者判劣 | ROW3 BL漏肾上腺结节; ROW1 BL漏LVEF; ROW5 BL漏全部病理 | 高 |
| E5 | **Molecular-result reporting** — 已完成的分子/基因检测结果，哪个报告得更完整（含 panel 阴性项）？ | 仅"无结果"或仅一条者判劣 | ROW7 BL仅MSH2/PL全panel; ROW8 BL"无结果"/PL MammaPrint; PDAC PL富KRAS/SPINK1/ATM | 高(有检测时) |
| E6 | **Response-status correctness** — response assessment 是否正确反映"是否在治疗中/有无可评疗效"？ | 误判治疗前或预期为当前者判劣 | ROW4 BL把治疗前生长当疗效; ROW6 BL把"预期excellent"当现状 | 中 |
| E7 | **Plan concreteness** — 治疗/用药计划是否给出具体药名与方案，而非含糊概括或遗漏？ | 含糊/遗漏方案者判劣 | ROW4 BL"无用药计划"漏taxol; 多行PL给具体方案 | 中-高 |
| E8 | **Field-boundary cleanliness** — procedure / genetic / supportive 字段是否未被错类内容污染（影像入procedure、受体入genetic、家庭药入supportive）？ | 含串扰者判劣 | 修复后PL干净; BL procedure偶混影像 | 中(评测确认) |
| E9 | **Stage–metastasis coherence** — 分期与转移描述是否相互一致且贴合原文（无过度分期/无跨癌种借用）？ | 含矛盾者判劣 | 修复后PL ROW2"locally recurrent"/ROW4不借卵巢期 | 中 |
| E10 | **Diagnostic granularity** — 癌症诊断的组织学+分级+受体/分子亚型是否更具体完整？ | 更含糊者判劣 | ROW7 BL省PR/组织学; 多行PL更全 | 中 |

---

## 2. 方法二相对方法一的差异点

- 方法一的 Q7/Q12（genetic 纯度、imaging 忠实）在**修复后两侧都干净**，方法二据实合并进 E8/不单列。
- 方法一的 Q15（referral）已对齐 BL，方法二**不出此题**（实际无稳定差异）。
- 方法二新增 E10（诊断粒度）——实际输出中反复看到 PL 受体/组织学更全，方法一未单列。

## 3. 评分与汇总（同方法一）

- 逐题 × 40 sample 评测 → 每题 PL 胜率 → 筛选 → 总平均 + 图。
- 两个 doc 各自产出一套入选题与总分；最终可取**两法交集**（最稳健、最不可质疑）
  或**并集**（覆盖最广）呈现。

## 4. 执行状态
- [ ] E1–E10 × 40 sample 评测
- [ ] 按入选标准筛题
- [ ] 与方法一结果对照（交集/并集）
- [ ] 出总表 + 图
