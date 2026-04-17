# V32 vLLM Re-run Review (8 samples, post-fix)

> Run: v32_vllm_test_20260416_214405
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (独立 pipeline)
> Fixes applied: (1) strip `<think>` tags (2) Advance_care_planning uses full note
> Previous run: v32_vllm_test_20260416_211909 (P2=10, 1.25/sample)
> Status: **审查完成 — 全部 8/8 samples 已审查**

### 修复验证

| 问题 | 状态 | 说明 |
|------|------|------|
| `<think>` 标签污染 | ✅ 已修复 | 0 occurrences (之前所有输出都有) |
| Advance care "Full code" 提取 | ✅ 已修复 | ROW 1 + ROW 64 都正确提取了 "Full code" |
| ROW 2 Response_Assessment | ❌ 未修复 | 非 `<think>` 问题，模型生成叙事文本而非 JSON |

## 汇总统计

| 严重度 | 数量 | 比率 | 对比前次 |
|--------|------|------|----------|
| **P0** | 0 | 0% | = |
| **P1** | 0 | 0% | = |
| **P2** | 7 | 0.88/sample | ↓ from 10 (1.25/sample) |

---

### ROW 1 (coral_idx 140) — 0 P1, 3 P2 (was 5)
- ✅ **Advance care**: "Code status: Full code." ✅ (was "Not discussed")
- ✅ **Letter**: 不再说 "No imaging/labs planned" (之前有这句错误)
- P2: **imaging_plan "No imaging planned"** — A/P 说 "complete staging work up"，MRI brain + bone scan orders 在笔记顶部
- P2: **lab_plan "No labs planned"** — CBC, CMP, CA 15-3, CEA, PT/APTT orders 在笔记顶部
- P2: **Referral 全 "None"** — A/P #6 "[redacted] Referral asap" (integrative medicine) 但名字被 redact

### ROW 2 (coral_idx 141) — 0 P1, 1 P2 (unchanged)
- P2: **Response_Assessment 仍然 error** — 模型生成叙事文本而非 JSON, JSON 修复也失败
- ✅ `<think>` 已清除，但底层模型行为导致此字段持续失败

### ROW 8 (coral_idx 147) — 0 P1, 1 P2 (unchanged)
- P2: **Stage "pT0N2"** — 3/28 LN+ (2 macro + 1 micro) = ypN1a, not N2。ypT0 N1a = Stage IIA

### ROW 17 (coral_idx 156) — 0 P1, 0 P2 ✅

### ROW 29 (coral_idx 168) — 0 P1, 0 P2 ✅

### ROW 46 (coral_idx 185) — 0 P1, 0 P2 ✅
- ✅ 关键 sarcoidosis 测试通过: Metastasis "No", Goals "curative"

### ROW 64 (coral_idx 203) — 0 P1, 0 P2 ✅ (was 1)
- ✅ **Advance care**: "Full code." ✅ (was "Not discussed")

### ROW 100 (coral_idx 239) — 0 P1, 2 P2 (unchanged)
- P2: **lab_summary "Values redacted"** — Labs 未被 redact（CA 15-3, CA 27.29, CEA, ALP, AST, Hgb 等全在）
- P2: **current_meds empty** — Gemzar (gemcitabine) 是当前活跃治疗

---

## 完整统计

| ROW | coral_idx | P0 | P1 | P2 | 主要问题 |
|-----|-----------|----|----|-----|---------|
| 1 | 140 | 0 | 0 | 3 | imaging/lab/referral orders 在 A/P 之外 |
| 2 | 141 | 0 | 0 | 1 | Response_Assessment 模型生成叙事文本 |
| 8 | 147 | 0 | 0 | 1 | Stage N2 vs N1a |
| 17 | 156 | 0 | 0 | 0 | ✅ |
| 29 | 168 | 0 | 0 | 0 | ✅ |
| 46 | 185 | 0 | 0 | 0 | ✅ 关键 sarcoidosis 测试通过 |
| 64 | 203 | 0 | 0 | 0 | ✅ (Advance care fixed) |
| 100 | 239 | 0 | 0 | 2 | lab_summary 误判, current_meds 漏 Gemzar |
| **Total** | | **0** | **0** | **7** | **0.88/sample** |

## V31 vs V32 最终对比

| 指标 | V31 (Qwen2.5-32B-AWQ) | V32 (Qwen3.5-35B-A3B-GPTQ) |
|------|------------------------|----------------------------|
| 模型 | Qwen2.5-32B-Instruct-AWQ | Qwen3.5-35B-A3B-GPTQ-Int4 |
| Pipeline | run.py (HF 直接加载, KV Cache 分叉) | vllm_pipeline (vLLM HTTP API, Prefix Caching) |
| 样本数 | 61 | 8 (test) |
| P0 | 0 (0%) | 0 (0%) |
| P1 | 0 (0%) | 0 (0%) |
| P2 | 112 (1.84/sample) | 7 (0.88/sample) |
| 速度 | ~15-20min/8 samples | 2.0min/8 samples (8-10x faster) |
| Letter 质量 | 良好 | 出色（通俗化更好, source tags 正确） |
| 关键测试 | — | Sarcoidosis 正确 ✅ |

## 剩余 P2 模式分析（模型行为问题，非代码 bug）

1. **Orders outside A/P** (ROW 1, ×3): 笔记顶部的 orders 不在 A/P 段中，plan extraction 只看 A/P
2. **Response_Assessment JSON 失败** (ROW 2): 模型输出叙事文本而非 JSON
3. **Stage N classification error** (ROW 8): N1a vs N2 分错
4. **lab_summary 误判 "Values redacted"** (ROW 100): 实际有大量 lab 数据
5. **current_meds 遗漏** (ROW 100): 当前活跃治疗 Gemzar 未被提取
