# V31 vLLM 迭代循环计划

## 成功条件
**所有 sample 的 extraction 全方位击败或等同 HF 版本。**
- 每个 field × 每个 sample：vLLM ≥ HF（内容更多或一致，不能更少）
- 零空值（HF有内容的地方 vLLM 不能为空）
- 零 P0（不能幻觉/编造）

## 迭代流程
1. WSL 跑完 → 下载结果
2. 逐 sample 逐 field 审查（对照原文 + 对照 HF）
3. 找出所有 vLLM 不如 HF 的 field-instance
4. 分析根因 → 修改 code/prompt
5. 选择测试集：所有未全胜的 sample + 10 个随机回归
6. 重新跑 → 回到步骤 1

## 当前迭代状态

### Iter8（当前）
- 配置: exp/v31_vllm_iter8.yaml
- 样本数: 56（46 must-test + 10 regression）
- 修复内容:
  - POST-STAGE-INFER: 排除 "5cm from nipple"
  - POST-STAGE-CORRECT: N classification ≠ node count
  - POST-STAGE-INFER: "no definite metastatic" 排除
  - POST-THERAPY: exercise 加入 whitelist
  - Referral prompt: "symptom management" 关键词
  - 全局 prompt: "不添加/不假设/不泛化" 原则
- 状态: **运行中**
- 预计完成: ~48 分钟

### 历史
| Iter | Samples | P0 | P1 | P2 | 全胜率 | 关键改动 |
|------|---------|----|----|----|----|---------|
| iter4 | 100 | 0 | — | — | — | baseline POST hooks |
| iter5 | 100 | 0 | — | — | — | STAGE-INFER fix, therapy/med prompt |
| iter6 | 100 | 0 | — | — | — | STAGE-CORRECT, imaging tighten |
| iter7 | 100 | 0 | 8 | 42 | 21/61(34%) | 详细100×11审查完成 |
| iter8 | 56 | 0 | — | — | — | 3 P1 fix + doctor feedback (8a/8b/8c) |
| iter8c | 56 | 0 | 0 vLLM空值 | — | — | DISTMET-NOMET搜A/P only + exercise + 5个P1修复 |
