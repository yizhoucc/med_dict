# V17 Targeted Fix Verification Report

**验证时间**: 2026-03-19 11:23 (运行 3小时)
**环境**: Qwen2.5-32B-Instruct-AWQ, WSL
**样本数**: 61 行 (从 v16_tracking 中选取的问题行)

---

## Fix 5 - POST-ER-CHECK Format

**目标**: 避免 `Type_of_Cancer` 出现 leading comma (当 Type 为空时)，避免重复 ER 推断

| CSV Row | coral_idx | Type_of_Cancer | 状态 | 备注 |
|---------|-----------|----------------|------|------|
| 8 | 148 | ER+/PR-/HER2- invasive ductal carcinoma | ✅ | 格式正确，无 leading comma |

**结论**: ✅ **通过** (v16 Row 9 对应 CSV row 8)

---

## Fix 6 - POST-GENETICS Referral Expansion

**目标**: `Referral.Genetics` 应排除已完成的测试结果，只保留 referral plan

| CSV Row | coral_idx | Referral.Genetics | 状态 | v16 原问题 |
|---------|-----------|-------------------|------|-----------|
| 84 | 224 | None | ✅ | "CHEK2 known mutation" |

**结论**: ✅ **通过** (v16 Row 85 对应 CSV row 84)

---

## Fix 7 - POST-MEDS-IV-CHECK

**目标**: `current_meds` 应包含 A/P 中提到的 IV 化疗药物 (之前 regex 只匹配口服药)

| CSV Row | coral_idx | 预期药物 | current_meds | 状态 | A/P 证据 |
|---------|-----------|----------|--------------|------|----------|
| 1 | 141 | irinotecan | (empty) | ❌ | "started irinotecan on 06/30/19", "will change her irinotecan to every other week" |
| 49 | 189 | lupron | ibrance, xgeva, letrozole | ❌ | "tamoxifen lupron until progression" |
| 63 | 203 | AC | (empty) | ❌ | "started on dd AC" |

**结论**: ❌ **完全失败** - 所有 3 个目标行的 IV 药物均未提取

**根因分析**:
- A/P 中明确提到这些药物 (如 "started irinotecan", "remains on lupron", "started on dd AC")
- 但 `current_meds` 仍为空或缺失这些药物
- 说明 POST-MEDS-IV-CHECK hook 未被执行，或执行后未修改 `current_meds`

---

## Fix 8 - POST-RESPONSE-TREATMENT

**目标**: `response_assessment` 不应在患者已完成/正在接受治疗时说 "Not yet on treatment"

| CSV Row | coral_idx | response_assessment | 状态 | Note 证据 |
|---------|-----------|---------------------|------|-----------|
| 42 | 182 | Not yet on treatment — no response to assess. | ❌ | "S/p dose dense AC x 4" (已完成新辅助化疗) |
| 62 | 202 | The cancer is currently stable with no evidence of disease recurrence... | ✅ | "started on dd AC" |

**结论**: ⚠️ **部分通过** (1/2) - Row 63 已修复，但 Row 43 仍有 false negative

**Row 43 详情**:
- Note: "S/p dose dense AC x 4" (已完成 4 个周期新辅助化疗)
- response_assessment 仍说 "Not yet on treatment"
- 说明 POST-RESPONSE-TREATMENT hook 的治疗关键词检测未覆盖 "S/p dose dense AC" 模式

---

## 总结

| Fix | 状态 | 通过率 | 关键问题 |
|-----|------|--------|----------|
| Fix 5 (POST-ER-CHECK) | ✅ | 1/1 (100%) | 无 |
| Fix 6 (POST-GENETICS) | ✅ | 1/1 (100%) | 无 |
| Fix 7 (POST-MEDS-IV) | ❌ | 0/3 (0%) | **hook 未生效或未写回** |
| Fix 8 (POST-RESPONSE) | ⚠️ | 1/2 (50%) | "S/p AC" 模式未匹配 |

### 关键发现

1. **✅ Fix 5 和 Fix 6 工作正常** - 格式修复和 genetics 过滤均生效

2. **❌ Fix 7 完全失败** - 最严重的问题
   - 所有 3 个测试用例均未提取到 IV 药物
   - A/P 中明确提到 "started irinotecan", "remains on lupron", "started on dd AC"
   - 怀疑 `run.py` 中的 POST-MEDS-IV-CHECK hook 未执行或未正确写回 `current_meds`
   - **需要紧急检查 `run.py:POST-MEDS-IV-CHECK` 实现**

3. **⚠️ Fix 8 部分失效**
   - Row 63 (coral 202) 正确识别了 "started on dd AC" 为治疗中
   - Row 43 (coral 182) 未识别 "S/p dose dense AC x 4" (已完成新辅助化疗)
   - 需要扩展治疗模式: `S/p` (status post = 已完成), `dose dense`, `neoadjuvant`

4. **⚠️ results.txt 写入 bug**
   - CSV row 62 和 84 在 `progress.json` 中有数据，但未写入 `results.txt`
   - 导致 OUTPUT ROW 38 和 51 缺失
   - 可能是 `ult.py` 中的 `write_results_to_file()` 逻辑问题

### 下一步行动

**P0 紧急修复**:
1. 检查 `run.py` 中 POST-MEDS-IV-CHECK 的实现和调用位置
2. 确认 hook 返回的修改是否正确写回到 `keypoints['Current_Medications']['current_meds']`
3. 检查 `write_results_to_file()` 为什么漏掉某些行

**P1 改进**:
1. Fix 8: 扩展 POST-RESPONSE-TREATMENT 的治疗模式检测:
   - 添加 `S/p` (status post)
   - 添加 `dose dense`
   - 添加 `completed [N] cycles`
2. 在 v18 中重新测试这 3 个失败的 Fix 7 行

---

**生成时间**: 2026-03-19
**审查者**: ClaudeCode Agent
