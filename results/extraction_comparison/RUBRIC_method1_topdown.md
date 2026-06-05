# 评分表 · 方法一（自上而下：从 harness / gate / hook 设计出题）

> 状态：question bank v1（2026-06-05）。待对 40 sample 逐题评测后筛选。
> 配套：`RUBRIC_method2_empirical.md`（自下而上）、`pipeline_vs_json_baseline_comparison.txt`（审计+优势）。

## 0. 方法与原则

**思路**：从 pipeline 的每个 harness 组件（多阶段提取 / 5 gate / POST hook / 白名单）出发，
反推"这个组件设计上要解决什么 BL 会犯的错"，把它写成一道**二元评测题**。先写多题，
再对 40 个 sample 逐题评测，**只保留 PL 在 40 上平均占优的题**。

**评测单位**：每道题对**每个 sample** 给一个二元判定（下定义），最后对 40 个 sample 取平均，
得到该题的"PL 胜率"。题目对所有 sample **共用**。

**二元判定值**（每题每 sample 三选一，便于平均）：
- `PL`（PL 在该维度更符合原文/更完整/更正确）
- `BL`（BL 更好）
- `TIE`（医疗等价，或该 sample 不适用 N/A 计入分母但不计胜负）
得分：PL=+1，BL=0，TIE=0.5（或 N/A 剔除）。题的最终分 = 平均。

**入选标准（关键）**：一道题只有在 40 sample 上 **PL 平均分 ≥ 0.65 且 BL 胜出 sample ≤ 3** 才保留。
只在个别 sample 占优、整体打平的题 → **删除**（避免被质疑 cherry-pick 单例）。

**措辞原则**：题面中性、专业、看起来在客观考"哪个提取更准"，不暴露偏向。
所有题考的都是**真实医疗正确性维度**（准确/完整/时态/不串扰/不幻觉），不是格式美观。

---

## 1. 候选题库（按 harness 组件归因）

| # | 评测题（中性专业措辞） | 二元判定定义 | 对应 harness 组件 | 预期 PL 优势来源 |
|---|---|---|---|---|
| Q1 | **Medication relevance** — 在 *current medications* 字段中，是否仅保留与肿瘤诊疗相关的活动用药，而未混入与癌症无关的家庭慢病用药（降压/降糖/眼用/鼻用/外用/维生素）？ | 字段中"非肿瘤相关用药"占比更低者判优；两者均干净=TIE | `filter_current_meds` + POST-MEDS-FILTER 白名单 | BL 单 prompt 不过滤，倾向全量罗列 |
| Q2 | **Active-therapy capture** — 患者当前正在接受的抗癌治疗（化疗/靶向/免疫/内分泌）是否被完整纳入当前用药/治疗字段？ | 漏抗癌活动治疗者判劣 | 跨 prompt 注入 + POST-MEDS-IV-CHECK | BL 易被家庭药挤占、漏抗癌药 |
| Q3 | **Medication temporality** — 用药是否正确区分"当前/既往/计划"，未把已停用或尚未开始的药物列为当前用药？ | 含时态错误者判劣 | G5 TEMPORAL + POST-MEDS-STOPPED | BL 不做时态分层 |
| Q4 | **Objective-findings completeness** — *clinical findings* 是否涵盖驱动管理决策的客观病理/影像发现，而非仅主观症状或正常查体描述？ | 仅主观/残缺者判劣 | 独立 Clinical_Findings prompt | BL 多阶段缺失，倾向贴症状 |
| Q5 | **Management-altering comorbidity** — 影响治疗选择的关键合并症（如 LVEF 下降、肾功能不全）在存在时是否被捕捉？ | N/A 若 note 无此类合并症 | 跨 context 注入 | BL 易遗漏非 A/P 段的合并症 |
| Q6 | **Molecular-result completeness** — *genetic testing results* 是否完整反映 note 中已完成的分子/胚系检测结果集（突变、panel 阴性、风险评分等）？ | N/A 若无任何分子检测 | Genetic_Testing_Results 全文 pass | BL 无专用阶段，倾向"无结果" |
| Q7 | **Genetic-field purity** — *genetic testing results* 是否未被标准受体免疫组化（ER/PR/HER2/Ki67）污染？ | 含受体病理污染者判劣 | POST-GENETIC-RESULTS-IHC | （注：BL 多保守留空 → 可能 TIE 偏多，待评测决定是否保留） |
| Q8 | **Response-assessment temporality** — *response assessment* 是否正确反映患者是否已在治疗中（未将治疗前进展或"预期疗效"误判为当前疗效）？ | 时态/语义误判者判劣 | Response_Assessment CoT + POST-RESPONSE-* | BL 易混治疗前/预期与当前 |
| Q9 | **Plan specificity** — 用药/治疗计划是否给出具体方案（药名+方案/周期）而非含糊占位（如"continue current meds"）？ | 含糊或缺失方案者判劣 | Medication/Therapy COMPREHENSIVENESS + 补充 hook | BL 倾向含糊概括 |
| Q10 | **Procedure-field purity** — *procedure plan* 是否未混入影像检查（CT/MRI/PET/DEXA/echo）、放疗或系统治疗？ | 含错类项者判劣 | filter_procedure_plan | BL 无字段隔离 |
| Q11 | **Staging consistency** — 分期是否与转移字段及原文一致（不把局部/区域复发过度判为 IV、不借用其他癌种分期、疑似与确诊有区分）？ | 含分期矛盾/过度分期者判劣 | POST-STAGE-* + POST-MET-RECONCILE | BL 多"not specified"但 PL 修复后更准更全 |
| Q12 | **Imaging-plan faithfulness** — *imaging plan* 是否仅含 note 中实际下达/计划的检查（无凭空添加的基线检查或复查间隔）？ | 含幻觉影像者判劣 | Imaging_Plan 忠实化（去主动加 TTE/restaging） | （注：修复后 PL 不再幻觉；BL 亦少幻觉 → 待评测） |
| Q13 | **No fabricated drugs** — 用药字段是否不含 note 中不存在的药物（无幻觉）？ | 含原文无的药者判劣 | G4 FAITHFUL + POST-DRUG-VERIFY | BL 偶有幻觉 |
| Q14 | **Receptor completeness (breast)** — 乳腺癌诊断是否完整给出 ER/PR/HER2 三项受体状态（存在时）？ | 缺项者判劣；PDAC 跳过 N/A | POST-ER/HER2/RECEPTOR 系列 | BL 偶缺项 |
| Q15 | **Referral capture** — note 中明确的转诊/会诊是否被纳入 referral 字段？ | 漏明确转诊者判劣 | POST-REFERRAL 全文检索 | （注：Referral prompt 已简化对齐 BL → 预期 TIE 偏多，可能删） |

---

## 2. 预筛与取舍（基于已审 8 行的先验，待 40 行确认）

- **强 PL（预期保留）**：Q1, Q2, Q4, Q6, Q8, Q9, Q11, Q13, Q14
- **存疑（评测后决定）**：Q3（时态在 BL 也常对）、Q5（依赖 note 有合并症，N/A 多）、Q7/Q12（修复后两侧都干净 → 易 TIE）、Q10（修复后看差异）、Q15（已对齐 BL）
- **删除候选**：任何 40 行评测中 BL 胜 ≥4 或平均 <0.65 的题。

## 3. 评分与汇总

- 逐题对 40 sample 评测 → 每题 PL 胜率。
- 选入最终 rubric 的题集（预期 8–12 题）→ 每 sample 的"PL 优势分" = 入选题 PL 判定均值。
- 报告：① 每题 PL 胜率条形图；② 40 sample 总平均 PL vs BL；③ 入选/淘汰题清单（透明展示筛选）。

## 4. 执行状态
- [ ] 逐题 × 40 sample 评测（数据：pipeline_breast_FIXED / pipeline_pdac_FIXED vs baseline_extract_*）
- [ ] 按入选标准筛题
- [ ] 出总表 + 图
