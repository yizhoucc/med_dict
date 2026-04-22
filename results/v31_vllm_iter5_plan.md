# V31 vLLM Iter5+ 迭代计划 — 每个字段×每个sample 全面领先

## 目标
不仅是字段级聚合领先，而是 **每个字段×每个sample** 都不输 HF。

## Iter4 vs HF 对比分析 (61 matched samples, 11 fields = 671 field-instances)

### 统计
- Tie (完全相同): 375 (55.9%)
- vLLM empty / HF has content: **0** ✅
- HF empty / vLLM has content: 17 (2.5%) — vLLM wins
- Content differences (both have content): **279** (41.6%) — 需要逐个分析

### 按字段分类的关键问题

#### 1. Stage_of_Cancer — POST-STAGE-INFER BUG (P0)
**POST-STAGE-INFER 用 tumor size 推断 Stage，但忽略了已有的 Distant Metastasis 信息**
- ROW 54: HF "now metastatic (Stage IV)" → vLLM "Stage II (inferred from 3.0cm tumor)" ❌
- ROW 84: HF "Stage IV (metastatic)" → vLLM "Stage II (inferred from 2.3cm tumor)" ❌
- ROW 86: HF "Stage IV (metastatic)" → vLLM "Stage II (inferred from 3.3cm tumor)" ❌
- ROW 92: HF "Stage IV (metastatic)" → vLLM "Stage II (inferred from 2.0cm tumor)" ❌
- ROW 87: HF "Stage IIIA (inferred from pT2 N2a)" → vLLM "Stage II (inferred from 2.2cm tumor)" ❌ (忽略了 node info)
- ROW 17: HF "Stage IA (inferred from pT1b N0)" → vLLM "Stage II (inferred from 0.8cm tumor)" ❌ (0.8cm+N0=IA 不是 II)
- ROW 95: HF "Stage IIA (inferred from pT2 N1a)" → vLLM "Stage II (inferred from 2.1cm tumor)" — 不够精确
- ROW 83: HF "Stage III" → vLLM "Not available (redacted)" — LLM 没提取到

**修复方案**:
1. POST-STAGE-INFER 先检查 Distant Metastasis 字段，如果 "Yes" → 直接设为 Stage IV
2. POST-STAGE-INFER 同时搜索 node status，用 tumor size + node 联合推断
3. POST-STAGE-INFER 不应覆盖已有的更详细的 stage

#### 2. therapy_plan — vLLM 在多个 sample 中细节不足
HF 更详细的 cases:
- ROW 6, 12, 29, 34, 46, 66, 70, 72, 84, 85, 87, 88, 91, 97

**可能原因**: 
- vLLM 的 max_tokens 或 prompt 限制导致输出截断
- therapy_plan prompt 不够强调"完整列出所有计划"

#### 3. Medication_Plan — 类似 therapy_plan
HF 更详细: ROW 7, 40, 44, 46, 49, 50, 59, 87, 88, 95

#### 4. imaging_plan — POST hook 可能过度添加
vLLM 添加了 HF 没有的 imaging:
- ROW 7: vLLM "PET/CT" vs HF "No imaging planned"
- ROW 53: vLLM "DEXA scan" vs HF "No imaging planned"
- ROW 72: vLLM "DEXA scan. Ultrasound" vs HF "No imaging planned"
- ROW 83: vLLM "Bone scan" vs HF "No imaging planned"
需要验证这些是否在原文中

#### 5. response_assessment — 大部分 vLLM 更好，少数 HF 更详细
HF 更详细: ROW 2, 7, 11, 12, 59, 70, 84, 85, 88, 92

#### 6. Distant Metastasis — vLLM 偶尔漏信息
- ROW 20, 50, 92, 100: vLLM 比 HF 漏了一些转移位点

#### 7. Type_of_Cancer — vLLM 大幅领先
极少数 HF 更好:
- ROW 57: vLLM 写了 "ER+/PR+/HER2-" 但 HF 写 "triple negative" — 需要验证
- ROW 94: vLLM 写了 CMS code 而不是解析的信息

## 执行计划

### Phase 1: 修 POST-STAGE-INFER bug (最高优先级)
- [ ] 先检查 Distant Met → 有 → Stage IV
- [ ] 搜索 node status 联合推断
- [ ] 不覆盖已有的更详细 stage

### Phase 2: 验证 imaging_plan POST hook (可能的 false positive)
- [ ] 检查 ROW 7, 53, 72, 83 的原文

### Phase 3: therapy_plan / medication_plan 不够详细
- [ ] 分析是否是 max_tokens 问题
- [ ] 考虑 prompt 改进

### Phase 4: response_assessment 少数 HF 更好的 case
- [ ] 分析原因

### Phase 5: 重新全量运行并对比
