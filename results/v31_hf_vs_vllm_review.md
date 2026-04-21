# V31 HF vs vLLM Extraction Comparison Review

> Model: Qwen2.5-32B-Instruct-AWQ
> HF: results/v31_full_20260413_221315 (61 samples)
> vLLM: results/v31_vllm_full_20260421_091554 (100 samples, 61 matched)
> Pipeline: run.py V2 (5 gates, 40+ POST hooks)
> Status: **审查中 — 8/61 逐字审查完成**

## 汇总

| 指标 | HF 更好 | vLLM 更好 | 平手 |
|------|---------|----------|------|
| 总计 | 9 | 4 | 15 |

## 审查记录

| ROW | HF wins | vLLM wins | Tie | 备注 |
|-----|---------|-----------|-----|------|
| 1 | 1 | 0 | 7 | HF: genetic_testing_plan 错放 medication 内容 (逐字审查) |
| 2 | 2 | 0 | 2 | HF: therapy_plan+lab_plan; vLLM therapy_plan="None"! (逐字审查) |
| 3 | 0 | 0 | 0 | 无内容差异 ✅ (逐字审查) |
| 5 | 2 | 1 | 0 | HF: micropapillary+leuprolide; vLLM: Distant Met更准确(sternum) (逐字审查) |
| 6 | 1 | 1 | 2 | HF: Type详细(DCIS/grade); vLLM: Stage有值(HF空) (逐字审查) |
| 7 | 2 | 0 | 1 | HF: Distant Met+Herceptin; vLLM空 (逐字审查) |
| 8 | 0 | 1 | 1 | vLLM: Stage更详细 (逐字审查) |
| 9 | 1 | 1 | 2 | HF: bone protection; vLLM: Stage更详细 (逐字审查) |
