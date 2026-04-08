# V26 Tool Calling Test Review

> Run: v26_test_tools_20260407_210517
> 5 samples: ROW 1 (coral 140), 12 (151), 14 (153), 33 (172), 88 (227)
> Focus: 验证 tool calling (SEARCH_NOTE/DEFINE) + Gate 3/4 跳过 + 持续 P1 检查

---

## 结果

| ROW | P0 | P1 | P2 | 关键发现 |
|-----|----|----|----|----|
| 1 | 0 | 1 | 3 | lab_plan 持续混入 imaging（A/P 措辞导致）。genetic_testing ✅ 修复。letter ✅ 无幻觉 |
| 12 | 0 | 0 | 4 | **Advance care ✅ DNR/DNI 成功捕获！** Tool calling 核心验证成功 |
| 14 | 0 | 0 | 1 | response 改善（"cancer stable"）。Stage ✅ POST hook 修复。current_meds 空（borderline） |
| 33 | 0 | 1 | 0 | letter stage "IIB→IIIA" 仍然误导 NED 患者（但加了 "no recurrence"） |
| 88 | 0 | 1 | 0 | response "Not mentioned" 持续（neoadjuvant progression→surgery 场景） |

**总计: 0 P0, 3 P1, 8 P2**

---

## Tool Calling 验证

### ✅ 成功: ROW 12 Advance Care
```
v24: "Not discussed during this visit."
v26: "Patient has completed POLST and is DNR/DNI. Goal to spend time with family 
     and maintain independence as possible, would not want life support treatments..."
```
- 模型主动搜索: SEARCH_NOTE("Code status"), SEARCH_NOTE("DNR"), SEARCH_NOTE("DNI") 等 6 次
- 从 problem list 成功找到 advance care planning 完整信息
- Gate 3/4 跳过 tool-enriched 字段，信息得以保留

### 部分成功: ROW 1 Lab_Plan
- 模型搜索了 SEARCH_NOTE("CBC"), SEARCH_NOTE("CMP") 等 7 次
- 但 A/P 原文 "ordered a MRI of brain and bone scan as well as labs" 把 imaging 和 labs 混写
- Tool calling 无法解决 A/P 文本本身的混合问题——需要 prompt 层面教模型分离

### 未触发: ROW 88 Response Assessment  
- response_assessment 不是 plan extraction 字段，不走 tool calling 路径
- 这是 Phase 2 extraction（用全文 cache），tool calling 目前只对 plan extraction 启用

---

## 持续 P1（3 个）

| ROW | 字段 | 问题 | 修复方向 |
|-----|------|------|----------|
| 1 | lab_plan | A/P 混写 imaging+labs | 需 lab_plan prompt 加分离指导 |
| 33 | letter | "IIB→IIIA" 误导 NED 患者 | 需 letter prompt 区分 staging update vs progression |
| 88 | response | neoadjuvant progression→surgery | 需 response prompt 覆盖 progression 场景 |
