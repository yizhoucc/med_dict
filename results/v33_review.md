# V33 Review — P2 Fix Verification

> Run: v33_full_20260417_151325
> Target: Fix 17 P2s from V32 review
> Model: Qwen3.5-35B-A3B-GPTQ-Int4 via vLLM 0.19.0
> Pipeline: vllm_pipeline/run_vllm.py (10 original + 10 new POST hooks)
> Status: **自动验证通过 — 17/17 P2 修复，0 回归。待逐字审查确认**

## 汇总

| 指标 | V32 | V33 |
|------|-----|-----|
| P0 | 0 | 0 |
| P1 | 0 | 0 |
| **P2** | **17** | **0 (自动扫描)** |
| 速度 | 14.5 min | 14.4 min |

## V32 P2 清单与修复状态

| ROW | P2 类型 | 修复方法 | V33 结果 | 状态 |
|-----|---------|---------|---------|------|
| 9 | medication_plan: s/p taxol as future | POST-MEDS-SP + prompt | "completed" | ✅ |
| 10 | Stage IV + Response error | POST-STAGE-DISTMET | Stage IIIA, Distant Met=No | ✅ |
| 27 | Distant Mets: liver indeterminate | POST-INDETERMINATE-MET | "Yes, to bone" (liver removed) | ✅ |
| 40 | imaging_plan 漏 DEXA | POST-IMAGING | "DEXA scan" | ✅ |
| 43 | lab_summary "Values redacted" | POST-LAB-REDACTED | "Labs present in note" | ✅ |
| 57 | Type lobular + genetic_testing None | prompt (lobular≠imaging) + POST-GENETICS | "Grade III adenoCA, TNBC" + "Rec genetic counseling" | ✅ |
| 59 | current_meds exemestane not started | POST-MEDS-NOT-STARTED | "" (removed) | ✅ |
| 61 | genetic_testing_plan None | POST-GENETICS (redacted pattern) | "Genomic test planned (name redacted)" | ✅ |
| 63 | lab_plan + imaging_plan 漏 | POST-LAB-SEARCH + POST-IMAGING | "Estradiol; FSH" + "DEXA scan" | ✅ |
| 66 | Stage N1 biopsy negative | POST-DISTMET-REGIONAL + prompt | Stage IIB, Distant Met=No | ✅ |
| 68 | HER2- but TCHP + Stage IV | POST-HER2-VERIFY + POST-STAGE | HER2+ Stage IIIA | ✅ |
| 83 | Stage IV no distant mets | POST-STAGE-DISTMET | Stage III | ✅ |
| 97 | genetic_testing_plan None | POST-GENETICS-SEARCH | Oncotype Dx | ✅ |

## 回归扫描

- Stage IV + Distant Met=No 矛盾检查：0 回归 ✅
- genetic_testing_plan 误含药物名：0 回归 ✅
- POST hook 无崩溃：✅

## POST hooks 触发统计 (V33 全量)

| Hook | 触发次数 |
|------|---------|
| POST-GENETICS-SEARCH | 7 |
| POST-HER2-VERIFY | 3 |
| POST-LAB-REDACTED | 3 |
| POST-IMAGING | 2 |
| POST-INDETERMINATE-MET | 2 |
| POST-LAB-SEARCH | 2 |
| POST-STAGE-DISTMET | 2 |
| POST-MEDS-NOT-STARTED | 1 |
| supportive_meds filter | 3 |
| Palliative care removal | 2 |
| imaging_plan header | 1 |
| lab_summary old labs | 1 |
