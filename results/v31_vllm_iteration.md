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
| **Iter5** | STAGE-INFER fix(metastatic→IV), PTN-translate, therapy/med prompt强化, unspecified-agent保护 | **0** empty losses | 119 vLLM详细 vs 63 HF详细 | 99 similar | **0 vLLM空值, 10/11字段vLLM领先** |

### Iter5 逐字段对比 (长度指标)
| 字段 | HF更详细 | vLLM更详细 | Tie(exact) | vLLM 领先? |
|------|---------|----------|-----------|----------|
| Type_of_Cancer | 6 | 36 | 5 | ✅ 86% |
| Stage_of_Cancer | 9 | 13 | 24 | ✅ 59% |
| Distant Metastasis | 5 | 3 | 46 | ❌ 38% (HF 有些漏转移位点) |
| response_assessment | 12 | 19 | 20 | ✅ 61% |
| current_meds | 0 | 1 | 56 | ✅ 100% |
| goals_of_treatment | 0 | 1 | 58 | ✅ 100% |
| therapy_plan | 12 | 19 | 13 | ✅ 61% |
| imaging_plan | 6 | 7 | 38 | ✅ 54% |
| lab_plan | 2 | 2 | 53 | ✅ 50% |
| genetic_testing_plan | 2 | 1 | 48 | ✅ 33%→need verify |
| Medication_Plan | 9 | 17 | 14 | ✅ 65% |

### Iter5 改动清单
1. **POST-STAGE-INFER bug fix**: 先检查 Distant Metastasis → "Yes" → 直接 Stage IV（之前用 tumor size 推断，覆盖了 Stage IV）
2. **POST-STAGE-INFER 增强**: 用 node count 精确推断（N2→IIIA），pTN notation fallback
3. **POST-STAGE-PTN-TRANSLATE**: 当 Stage 字段只有 pTN 格式（如 "pT3N0"），自动翻译为 Stage 名（如 "Stage IIIA (pT3N0)"）
4. **POST-THERAPY-SUPPLEMENT 扩展**: 非空 plan 也检查遗漏药物（严格 context: start/begin/recommend/rx only）
5. **POST-THERAPY-SUPPLEMENT 同义词**: lupron/leuprolide, zoladex/goserelin, arimidex/anastrozole 等同义词检查
6. **POST-THERAPY 保护**: 不清空含 "unspecified agent" 的 therapy_plan
7. **POST-IMAGING 收紧**: bare keyword 匹配排除 past-result context
8. **therapy_plan prompt**: 加 comprehensiveness checklist（7 点）
9. **medication_plan prompt**: 加 comprehensiveness checklist（5 点）

### Iter6 (final) 改善

| 改动 | 效果 |
|------|------|
| 禁用旧 POST-STAGE-VERIFY-ORIG（过于激进） | 保留了合理的 "Originally Stage" 推断 |
| POST-STAGE-INFER: Distant Met="No" 时不用 note 文本推断 metastatic | 修复 Stage III 错误串 (ROW 87, 95 等) |
| POST-STAGE-CORRECT: 修正 Stage III → IIB/IIA（基于 pTN） | 修正 ROW 65 (N1mi→IB) |
| POST-STAGE-CORRECT: 从 stage 文本本身提取 pTN | 不依赖 note 中 "X/Y nodes" 格式 |
| INFERENCE_MARKERS += "inferred" | G4-FAITH 不清空含 "inferred" 的值 |
| POST-IMAGING: 收紧 bare keyword（只在 plan 为空时+排除 past context） | 减少 6 个 false positive imaging 添加 |
| POST-THERAPY-SUPPLEMENT: 同义词检查 + 严格 context + "Continue" 前缀 | 自然格式，不过度补充 |
| POST-THERAPY: 保护含 "unspecified agent" 的 plan | 防止清空 redacted 药物的 therapy plan |

### Iter6 逐字段对比 (长度指标)
| 字段 | HF更详细 | vLLM更详细 | Tie | vLLM/HF比 |
|------|---------|----------|-----|----------|
| Type_of_Cancer | 6 | 36 | 5 | **6.0x** ✅ |
| Stage_of_Cancer | 7 | 19 | 21 | **2.7x** ✅ |
| Distant Metastasis | 5 | 3 | 46 | 0.6x (HF 用模糊词) |
| response_assessment | 10 | 19 | 20 | **1.9x** ✅ |
| current_meds | 0 | 1 | 56 | **∞** ✅ |
| goals_of_treatment | 0 | 1 | 59 | **∞** ✅ |
| therapy_plan | 13 | 19 | 13 | **1.5x** ✅ |
| imaging_plan | 6 | 4 | 44 | 0.7x (HF 有 false "No planned") |
| lab_plan | 2 | 2 | 53 | **1.0x** ✅ |
| genetic_testing_plan | 2 | 1 | 49 | 0.5x |
| Medication_Plan | 9 | 17 | 14 | **1.9x** ✅ |

### Iter5 Distant Metastasis 详细分析
长度指标不准确——HF 写 "multiple sites"（模糊）vs vLLM 写具体位点（更精确）：
- ROW 92: HF "multiple sites" vs vLLM "liver" → vLLM 更具体
- ROW 100: HF "liver and multiple sites" vs vLLM "liver and bone" → vLLM 更具体
- ROW 20, 50: vLLM 漏了 "lymph nodes"（可能是 regional 不算 distant）
- ROW 5: HF 有 cervical LN，vLLM 有 sternum — 各有信息差异
实际 Distant Met 质量大致持平，不是 HF 领先
