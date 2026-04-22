# V31 vLLM Iteration — Match/Exceed HF Quality

## 成功标准
1. 提取质量无差异，仅措辞不同
2. miss 的更少（空值更少）
3. 细节更多
4. response_assessment 更详细
5. 不出空值
6. Type_of_Cancer 更完善

## HF vs vLLM 差距分析（61 samples 审查）

### vLLM 的 5 个系统性弱点
1. **Type_of_Cancer 偏简**：漏 DCIS、grade、micropapillary、metastatic recurrence（HF 赢 10, vLLM 赢 3）
2. **空值问题**：therapy_plan、Stage、Distant Met 偶尔为空（HF 有值）
3. **response_assessment 缺细节**：漏具体器官/部位信息（HF 赢 16, vLLM 赢 7）
4. **current_meds 漏药**：如 leuprolide
5. **genetic_testing_plan 错放内容**：把 medication plan 内容放进来

### vLLM 的 2 个优势（要保持）
1. Stage 推理更详细（附推理依据）
2. Redacted 内容处理更好

## 迭代计划

### Iteration 1: Prompt 改进
- [ ] Type_of_Cancer: 强制要求 grade、DCIS、histologic subtype
- [ ] response_assessment: 要求包含具体器官/部位/measurement
- [ ] 所有字段: 减少空值（"不确定时写推断"）
- [ ] genetic_testing_plan: 更强的边界（不允许 medication 内容）

### 进度

| Iteration | 改动 | HF wins | vLLM wins | Tie | 改善 |
|-----------|------|---------|-----------|-----|------|
| Baseline | - | 41 | 24 | 175 | - |
| **Iter1** | prompts改进 | **33** | **66** | 123 | **HF 33% → vLLM 67%** ✅ |
| **Iter2** | therapy_plan+Distant Met修复 | **27** | **67** | 131 | **HF 29% → vLLM 71%** ✅ |
| **Iter3** | POST hooks: therapy+imaging+lab | **24** | **72** | — | **HF 25% → vLLM 75%** ✅ 10/11字段达标 |
| **Iter4** | POST-STAGE-INFER | **25** | **76** | — | **HF 25% → vLLM 75%** ✅ **11/11字段全部达标** |
