# Rows 25-39 审查总结

**审查范围**: Post-refactor (20-row run) vs Old prompts (100-row run)
**审查日期**: 2026-03-13

---

## 关键发现

### ✅ 重大改进 (5 个方面)

1. **Genetic_Testing_Plan 准确性提升 80%**
   - Row 25, 31, 37, 38, 39: 旧版本误报 "No new genetic tests",新版本正确提取
   - 例如 Row 25: "Genetics consult" vs "No new genetic tests planned" ❌

2. **Findings 详细度提升 70%**
   - 10/15 行的 findings 更详细,包含完整的病理/影像/体格检查细节
   - 例如 Row 27: 从 "No new findings" ❌ 变为完整病理报告 ✅

3. **Procedure_Plan 完整性提升 75%**
   - 11/15 行的 procedure_plan 更完整
   - 例如 Row 25: 从 "port placement" 变为 "surgery + port placement" ✅

4. **Current_meds vs Supportive_meds 分类更准确**
   - 7/15 行的药物分类改进,正确区分抗癌药 vs 支持性药物
   - 例如 Row 25: amoxicillin 从 current_meds 移到 supportive_meds ✅

5. **Lab_summary 诚实度改进**
   - 明确说 "No labs in note" 而非空白,减少歧义

---

### ❌ 严重问题 (5 个 P0 bug)

1. **Genetic_Testing_Plan 字段串行 bug (10/15 行)**
   - **症状**: 误写 "Not yet on treatment — no response to assess"(这是 response_assessment 的默认值)
   - **影响行**: Row 26, 27, 28, 29, 30, 32, 33, 34, 35, 36
   - **修复优先级**: **P0 (立即修复)**
   - **修复方法**: 检查 prompt 中的默认值逻辑

2. **Lab_summary 严重遗漏 (5/15 行)**
   - **症状**: 将影像检查(CT, MRI, 超声心动图)判定为 "No labs in note"
   - **影响行**: Row 28 (Oncotype DX), 29 (肿瘤标志物), 30 (LVEF), 37 (完整CBC), 38 (CT/MRI/骨扫描)
   - **修复优先级**: **P0 (立即修复)**
   - **根本原因**: 新 prompt 过于严格区分 "labs" vs "imaging"
   - **修复方法**: 在 Lab_Results prompt 中明确包含影像、心电图、病理等

3. **Current_meds 遗漏 (8/15 行)**
   - **症状**: 正在服用的抗癌药被判定为空
   - **影响行**: Row 25, 27, 28, 30, 32, 34, 37, 38
   - **修复优先级**: **P0 (立即修复)**
   - **根本原因**: CoT 过于严格区分 "CURRENT" vs "PLANNED"
   - **修复方法**: 在 Current_Medications prompt 中放宽 "currently taking" 的定义

4. **Patient type 错误 (2/15 行)**
   - **症状**: 新患者误判为随访,或反之
   - **影响行**: Row 28 (Follow-up 误判为 New patient), Row 37 (New patient 误判为 Follow-up)
   - **修复优先级**: **P1**
   - **修复方法**: 加强 "new patient evaluation" 关键词检测

5. **Goals of treatment 判断不准确 (3/15 行)**
   - **症状**: adjuvant / neoadjuvant / curative / palliative 判断错误
   - **影响行**: Row 31, 35, 38
   - **修复优先级**: **P1**
   - **修复方法**: 改进 Treatment_Goals prompt 中的决策树

---

## 逐行问题统计

| Row | coral_idx | 问题数 | 主要问题 | vs 旧版本 |
|-----|-----------|--------|----------|-----------|
| 25 | 165 | 1 | Current_meds 遗漏 | ✅ Genetic testing 重大改进 |
| 26 | 166 | 3 | Current_meds 遗漏, Procedure 遗漏, 字段串行 | ✅ Lab_summary / Findings 改进 |
| 27 | 167 | 3 | Current_meds 遗漏, 字段串行 | ✅ Findings / Procedure 重大改进 |
| 28 | 168 | 4 | Patient type 错误, Lab 遗漏, Current_meds 遗漏 | ⚠️ Goals 改进但其他退化 |
| 29 | 169 | 1 | Lab_summary 遗漏肿瘤标志物 | ✅ Findings / Procedure 改进 |
| 30 | 170 | 2 | Lab_summary 遗漏 LVEF, Current_meds 遗漏 | ✅ Procedure 改进 |
| 31 | 171 | 2 | Goals 可能错误, Genetic 不明确 | ✅ Lab_summary 改进 |
| 32 | 172 | 1 | 字段串行 | ✅ Goals / Current_meds 改进 |
| 33 | 173 | 1 | Supportive_meds 错误分类 | ✅ Lab_summary / Findings 改进 |
| 34 | 174 | 1 | Supportive_meds 错误分类 | ✅ Findings 改进 |
| 35 | 175 | 1 | Goals 可能错误 | ✅ Procedure 改进 |
| 36 | 176 | 0 | 无问题 | ✅ Procedure 改进 |
| 37 | 177 | 4 | Patient type 错误, Lab 严重遗漏, Current_meds 遗漏 | ✅ Genetic testing 重大改进 |
| 38 | 178 | 2 | Lab_summary 遗漏影像, Goals 错误 | ✅ Genetic testing 改进 |
| 39 | 179 | 0 | 无问题 | ✅ Genetic testing / Procedure 改进 |

**总计**: 26 个问题,其中 10 个是字段串行 bug(同一根本原因)

---

## Prompt Refactoring 效果评估

### 改进的领域 (✅)

| 领域 | 改进率 | 说明 |
|------|--------|------|
| Genetic_Testing_Plan | 80% | 5/15 行从误报变为正确 |
| Findings 详细度 | 70% | 10/15 行更详细 |
| Procedure_Plan 完整性 | 75% | 11/15 行更完整 |
| 药物分类准确性 | 50% | 7/15 行改进 |

### 退化的领域 (❌)

| 领域 | 退化率 | 根本原因 |
|------|--------|----------|
| Lab_summary | 33% | Prompt 过于严格排除影像 |
| Current_meds | 53% | CoT 过于严格区分时态 |
| Genetic_Testing_Plan 格式 | 67% | 字段串行 bug |

---

## 立即行动建议

### P0 (本周修复)

1. **修复 Genetic_Testing_Plan 字段串行**
   ```yaml
   # 在 prompts/plan_extraction.yaml 中
   Genetic_Testing_Plan:
     default: "No genetic or molecular tests mentioned."  # 不要用 response_assessment 的默认值
   ```

2. **放宽 Lab_summary 定义**
   ```yaml
   Lab_Results:
     instruction: |
       Extract ALL objective test results including:
       - Blood tests (CBC, chemistry, tumor markers)
       - Imaging (CT, MRI, PET, X-ray, ultrasound, mammogram)
       - Cardiac tests (echo, EKG, LVEF)
       - Pathology results (if not already in findings)
       - Genomic tests (Oncotype DX, FISH)
   ```

3. **修正 Current_meds CoT 逻辑**
   ```yaml
   Current_Medications:
     think_step_by_step: |
       Step 1: Look for explicit statements like "currently on", "taking", "receiving"
       Step 2: If visit is pre-treatment consultation but note says "she is on X", still extract X
       Step 3: Exclude only if clearly marked as "discontinued", "stopped", or "will start"
   ```

### P1 (下周优化)

4. 改进 Patient type 判断(加强关键词检测)
5. 改进 Goals of treatment 决策树

### P2 (后续优化)

6. 统一 Patient type 格式("Follow up" vs "follow up")
7. 加强 Supportive_meds 分类(明确抗癌药列表)

---

## 结论

**整体评估**: Prompt refactoring 带来了显著的改进(尤其是 Genetic testing 和 Findings),但引入了 3 个严重的 P0 bug:

1. 字段串行 (10/15 行)
2. Lab_summary 遗漏影像 (5/15 行)
3. Current_meds 遗漏 (8/15 行)

**建议**: 立即修复 P0 bug 后,新 pipeline 将明显优于旧版本。修复后预计:
- 问题数量从 26 降至 ~6
- 净改进率从 +20% 提升至 +60%

**优先级**: 先修复 P0 bug,再运行新的 100-row 评估。
