# TODO: Pipeline 改进（基于 v24 全量审查结果）

> 创建日期: 2026-04-05
> 更新日期: 2026-04-05（审查完成后修正优先级）
> 来源: harness_exploration.md + results/full_qwen_20260405_073716/review.md
> 审查结果: P0=0, P1=20, P2=70 (100 samples)

---

## 优先级修正说明

原来计划的两个 harness 改进（Gate 4 置信度、单字段重试）在实际审查中**没有对应的 P1 问题**：
- Gate 4 的"拿不准就保留"策略在 v23 prompt 改进后已经工作良好
- POST-STAGE-VERIFY / POST-DRUG-VERIFY 的 regex 修补在这次跑中未触发 P1

实际 P1 集中在 3 个主要来源，全部是 **prompt 层面** 的问题：

| P1 来源 | 数量 | 占比 |
|---------|------|------|
| Letter 生成幻觉/编造 | 3 | 15% |
| response_assessment 误判 | 3 | 15% |
| genetic_testing_plan 误分类 | 3 | 15% |
| 其他分散问题 | 11 | 55% |

---

## 改进 1 (最高优先级): Letter 生成 prompt 修复

**目标**: 消除 letter 中的幻觉/编造（3 P1）

**问题**:
- ROW 4, 5: lab_summary = "No labs in note" 但 letter 写 "Your blood tests are mostly normal"
- ROW 3: ROS 写 "No anxiety/depression" 但 letter 写 "You appear to be emotional"

**改动**:

### 1.1 修改 letter_generation.yaml
- [ ] 加规则: "If lab_summary says 'No labs in note', do NOT mention blood test results in the letter. Instead say 'No new tests were done during this visit' or omit entirely."
- [ ] 加规则: "Do NOT describe the patient's emotional state unless the note EXPLICITLY states the patient is emotional, tearful, anxious, or distressed. 'She has good support' does NOT mean the patient is emotional."

### 1.2 测试
- [ ] 在包含 "No labs in note" 的 sample 上跑（ROW 4, 5, 及其他类似），验证 letter 不再编造 lab 结果

---

## 改进 2 (高优先级): response_assessment prompt 修复

**目标**: 消除 response_assessment 误判（3 P1）

**问题**:
- ROW 9, 10: 完成新辅助化疗+手术后，写 "Not yet on treatment"——忽略了术后病理就是 response assessment
- ROW 11: 把换药前的 PET 进展归因于当前药物（A/P 说 "exam stable" 但 response 写 "not responding"）

**改动**:

### 2.1 修改 extraction.yaml Response_Assessment prompt
- [ ] 加指导: "If the patient completed neoadjuvant therapy and had surgery, describe the pathologic response from surgical pathology (e.g., residual tumor size, cellularity, LN status). This IS the response assessment."
- [ ] 加指导: "Carefully distinguish which treatment the imaging findings are assessing. If the A/P says 'exam stable' on current therapy, do NOT describe prior therapy's progression as current response."
- [ ] 加 BAD 示例: "BAD: 'Not yet on treatment' when patient completed neoadjuvant chemo and had surgery with pathologic partial response"

### 2.2 测试
- [ ] 在 post-neoadjuvant sample 上跑（ROW 9, 10），验证 response_assessment 描述病理响应

---

## 改进 3 (高优先级): genetic_testing_plan prompt 修复

**目标**: 消除 genetic_testing_plan 误分类（3 P1）

**问题**:
- ROW 3: 把 "biopsy to confirm HR and HER2 status" 放入 genetic_testing_plan（IHC 不是 genetic testing）
- ROW 7: 把 "recheck [LVEF/tumor marker]" 放入 genetic_testing_plan
- ROW 24: Oncotype/MammaPrint 明确计划中但写 "None planned"

**改动**:

### 3.1 修改 plan_extraction.yaml Genetic_Testing_Plan prompt
- [ ] 加定义: "Genetic testing includes: Oncotype DX, MammaPrint, Foundation One, Guardant360, BRCA testing, germline panels, tumor mutational profiling. Does NOT include: IHC (ER/PR/HER2), FISH, LVEF/echo, tumor markers (CA 27.29), routine blood work."
- [ ] 加 BAD 示例: "BAD: 'Biopsy to confirm HR/HER2 status' — this is IHC, not genetic testing"
- [ ] 加 GOOD 示例: "GOOD: 'Oncotype DX ordered' / 'Foundation One testing planned'"

### 3.2 测试
- [ ] 在 ROW 7 (LVEF recheck) 和 ROW 24 (Oncotype planned) 上验证

---

## 改进 4 (中优先级): POST-STAGE-METASTATIC hook

**目标**: 消除 Stage "Not available (redacted)" for metastatic disease（7 P2）

**改动**:

### 4.1 新增 POST hook in run.py
- [ ] 当 Metastasis = "Yes" 或 Distant Metastasis = "Yes" 且 Stage_of_Cancer 包含 "Not available" 或 "Not mentioned" 时，自动设为 "Stage IV (metastatic)"

### 4.2 测试
- [ ] 在 ROW 76, 79, 83, 84, 86, 92, 100 上验证

---

## 改进 5 (中优先级): procedure_plan 字段定义收紧

**目标**: 消除 procedure_plan 混入非 procedure 内容（5 P2 + 2 P1）

**改动**:

### 5.1 修改 plan_extraction.yaml Procedure_Plan prompt
- [ ] 加排除规则: "Procedure does NOT include: imaging orders (CT, MRI, DEXA, PET, echo), lab tests, referrals, genetic testing, medication plans. Procedure IS: port placement, biopsy, surgery, drain removal, transfusion."

---

## ~~改进 A: Gate 4 置信度评分~~ → 降级为观察

**原因**: v24 审查 100 sample 中 Gate 4 没有造成 P1 问题。"拿不准就保留"策略已通过 prompt 改进解决。
**行动**: 暂不实施。如果未来版本出现 Gate 4 误修剪的 P1，再启动。

## ~~改进 B: 单字段重试~~ → 降级为观察

**原因**: POST-STAGE-VERIFY 和 POST-DRUG-VERIFY 在 v24 中没有触发 P1。现有 regex 修补工作正常。
**行动**: 暂不实施。保留设计方案，待需要时启用。

---

## 执行顺序

1. **改进 1**（Letter prompt）— 最高 ROI，改 1 个 yaml 文件消除 3 P1
2. **改进 2**（response_assessment prompt）— 改 1 个 yaml 文件消除 3 P1
3. **改进 3**（genetic_testing_plan prompt）— 改 1 个 yaml 文件消除 3 P1
4. **改进 4**（POST hook）— 加几行 Python 消除 7 P2
5. **改进 5**（procedure_plan prompt）— 改 1 个 yaml 文件消除 5+ P2
6. 跑 8 sample 验证 → 全量跑 → 审查对比

---

## 进度

| 步骤 | 状态 | 日期 |
|------|------|------|
| 1.1 修改 letter_generation.yaml | 🔲 待做 | |
| 1.2 测试 letter 幻觉修复 | 🔲 待做 | |
| 2.1 修改 response_assessment prompt | 🔲 待做 | |
| 2.2 测试 post-neoadjuvant response | 🔲 待做 | |
| 3.1 修改 genetic_testing_plan prompt | 🔲 待做 | |
| 3.2 测试 genetic vs biomarker 区分 | 🔲 待做 | |
| 4.1 新增 POST-STAGE-METASTATIC hook | 🔲 待做 | |
| 5.1 修改 procedure_plan prompt | 🔲 待做 | |
| 全量跑 + 审查 | 🔲 待做 | |
