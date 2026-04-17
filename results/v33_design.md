# V33 Design Doc — Fix 17 P2s from V32 Review

## 成功条件
- 61/61 samples 全部 0 P2
- 不引入新 P2（回归测试）

## V32 P2 根因分析

**核心问题**：`run_vllm.py` 只有 ~10 个 POST hooks，而 `run.py` 有 40+ 个经过 v14-v31 验证的 hooks。大多数 P2 是 run.py 中已有 hook 能修的，但没有移植到 run_vllm.py。

## 修复计划

### Phase 1: 移植 POST hooks（高影响，低风险）

| Hook | 来源 run.py | 修复 P2 | 说明 |
|------|------------|---------|------|
| POST-DISTMET-REGIONAL | ~L1781 | ROW 66,68,83 | 将 regional LN 从 distant mets 中移除 |
| POST-STAGE-DISTMET | ~L1642 | ROW 10,66,68,83 | Stage IV + Distant Met=No → 降级 |
| POST-STAGE-REGIONAL | ~L1665 | ROW 66,68,83 | Stage IV 但只有 regional nodes → Stage III |
| POST-HER2-VERIFY | ~L2321 | ROW 68 | HER2- 但用了 TCHP → 修正为 HER2+ |
| POST-GENETICS-SEARCH | ~L1472 | ROW 57,61,97 | genetic_testing_plan "None" 时搜索关键词 |
| POST-IMAGING (full) | ~L1287 | ROW 40,63 | 搜索 DEXA 等遗漏的影像计划 |
| POST-LAB-SEARCH | ~L1227 | ROW 63 | 搜索遗漏的 lab 计划 |
| POST-MEDS-SP | 新 | ROW 9 | medication_plan 中 s/p → 移除已完成治疗 |
| POST-MEDS-NOT-STARTED | 新 | ROW 59 | current_meds 中 "has not tried it yet" → 移除 |
| POST-INDETERMINATE-MET | 新 | ROW 27 | liver "cysts/indeterminate" → 不算 distant met |

### Phase 2: Prompt 改进

| Prompt | 改动 | 修复 P2 |
|--------|------|---------|
| Cancer_Diagnosis | regional vs distant LN 明确说明 | ROW 66,68,83 |
| Cancer_Diagnosis | lobular mass ≠ lobular carcinoma | ROW 57 |
| Distant Metastasis | indeterminate lesions 不算 confirmed met | ROW 27 |
| Medication_Plan | s/p = completed, 不列入 | ROW 9 |
| Current_Medications | "has not tried" = 不列入 | ROW 59 |
| Genetic_Testing_Plan | 包含 counseling+testing, Oncotype after surgery | ROW 57,61,97 |
| Lab_Plan | 包含 monitoring labs (estradiol/FSH) | ROW 63 |

### Phase 3: 其他

- 增加 `Genetic_Testing_Plan` 到 `full_note_keys`
- 增加关键词到 `data/genetic_tests.txt`
- 改进 lab_summary "Values redacted" hook

## 执行步骤

1. ✅ 创建 design doc
2. [ ] 移植 POST hooks 到 run_vllm.py
3. [ ] 改进 prompts
4. [ ] 创建 v33 config
5. [ ] 在 WSL 测试（8 sample test run）
6. [ ] 审查测试结果
7. [ ] 全量运行 61 samples
8. [ ] 逐字审查 → iterate 直到 0 P2

## 进度追踪

| 日期 | 事件 | P2 |
|------|------|-----|
| 2026-04-17 | V32 审查完成 | 17 |
| 2026-04-17 | V33 开始 | - |
| 2026-04-17 | V33 iter 1: 10 POST hooks + prompts | 17→1 (ROW 1 brain) |
| 2026-04-17 | V33 iter 2: TNBC排除 + lab/genetics搜索 | 1→0 (test) |
| 2026-04-17 | V33 iter 3: 全量审查发现 ROW 1 + ROW 86 回归 | 0→2 |
| 2026-04-17 | V33 iter 4-5: brain staging + HER2 met biopsy | 2→0 (待全量验证) |
