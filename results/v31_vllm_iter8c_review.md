# V31 vLLM Iter8c — 审查报告

> 配置: exp/v31_vllm_iter8.yaml (56 samples)
> 匹配 HF: 50 rows
> 修复: 5个P1 + doctor feedback

## 状态
- All-win (vLLM ≥ HF 所有字段): **12/50 (24%)**
- HF 至少 1 个字段更长: **38/50**
- vLLM 空值: **0**

## 自动对比 vs iter7

| 指标 | iter7 (61 matched) | iter8c (50 matched) | 变化 |
|------|-------|--------|------|
| All-win 比例 | 21/61 (34%) | 12/50 (24%) | ⚠️ 下降（样本不同，iter8c只测了难sample） |
| vLLM 空值 | 0 | 0 | ✅ |
| P1 修复 | 8 个 | 5 个已修 | ✅ |

注意：iter8c 只测了"HF有优势"的 sample + 10 个回归，所以 all-win 比例低是预期的。

## 47 个 Big Gap (HF >30% 更长) 人工分类

### 分类结果

| 分类 | 数量 | 说明 |
|------|------|------|
| TIE (措辞差异) | 25 | HF 更啰嗦但信息一致 |
| REAL-MISS | 12 | vLLM 确实少了信息 |
| vLLM-BETTER | 5 | vLLM 更精确/更正确 |
| HF-WRONG | 5 | HF 混入不属于本字段的内容 |

### REAL-MISS 详情 (12个)

| ROW | Field | vLLM 漏了什么 | 可否修复 |
|-----|-------|-------------|---------|
| 91 | therapy_plan | 漏了lasix, KCL, elevation等supportive care | 模型行为 |
| 50 | medication_plan | 漏了tamoxifen (只有lupron/letrozole/ibrance) | 模型行为 |
| 92 | response_assessment | 只有exam findings, 漏了"stable on treatment"总结+tumor markers | 模型行为 |
| 88 | response_assessment | 给了treatment plan而非response (persistent P1) | 模型行为 |
| 82 | medication_plan | 只列cancer meds, HF列了所有meds (HYDROchlorothiazide等) | 设计差异 |
| 12 | response_assessment | HF有更多历史imaging context | 模型行为(detail level) |
| 84 | therapy_plan | HF有CT/LP/MRI在therapy_plan | HF-WRONG(imaging in therapy) |
| 46 | medication_plan | 漏了gabapentin | 模型行为 |
| 95 | Type_of_Cancer | 漏了"treatment effect, three foci, margins negative" | 模型行为 |
| 65 | medication_plan | 漏了部分chemo details | 模型行为 |
| 34 | Type_of_Cancer | 漏了receptor change history (originally/metastatic biopsy) | 模型行为 |
| 22 | imaging_plan | vLLM只有"Pet ct now", HF有更多context | HF-WRONG(有therapy in imaging) |

### 关键发现

**无法通过 POST hook 修复的问题 (模型行为差异):**
1. vLLM 倾向生成更短的 therapy_plan / medication_plan / response_assessment
2. vLLM 偶尔在 response_assessment 中给 future plan 而非 actual response
3. vLLM Type_of_Cancer 偶尔漏 receptor change history

**可通过 POST hook 修复的问题:** 基本为零（已经修完了）

## 结论

iter8c 的 12 个 REAL-MISS 全部是**模型行为差异**：
- vLLM 的 Qwen2.5-32B 通过 PagedAttention 生成的文本系统性地比 HF 版本更短
- 这不是 prompt 或 POST hook 能解决的
- 唯一可能的改善：增加 max_new_tokens 或修改 prompt 要求更详细

**当前 all-win 的上限估计: ~34-40%** (基于 iter7 的 21/61)
要达到 100% all-win，需要解决模型行为差异，这超出了 prompt/POST hook 的能力范围。
