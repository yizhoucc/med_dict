# Pipeline Version Tracking

## v14 (2026-03-16) — 基于 v13a 审查结果的 8 项改进

v13a 审查（100 行）发现 392 个问题（2 P0, 172 P1, 218 P2）。本版本针对系统性问题逐项修复。

### 改进清单

| # | 改进 | 修改文件 | 预期效果 |
|---|------|---------|---------|
| 1 | 删除 Medication_Plan_chatgpt prompt | plan_extraction.yaml, ult.py | ~45 个 P1 消除（字段不再存在） |
| 2 | 修复 supportive_meds 过滤（不再合并 oncology_whitelist） | ult.py | letrozole/tamoxifen 不再出现在 supportive_meds |
| 3 | Stage 幻觉修复（"Originally Stage IIA"） | extraction.yaml, run.py | 幻觉分期应降为 ≤2 |
| 4 | Cross-field 白名单过滤 | data/lab_tests.txt (新建), run.py | 跨字段分类错误显著减少 |
| 5 | therapy_plan "continue" 不再输出 None | plan_extraction.yaml | therapy_plan "None" 应降为 ≤2 |
| 6 | genetic_testing_plan 白名单搜索 | data/genetic_tests.txt (新建), run.py | 遗漏 Oncotype/BRCA 等应减少 |
| 7 | Referral 使用全文提取（不只 A/P） | run.py | Referral 遗漏应减少 |
| 8 | current_meds cross-reference 药物列表 | extraction.yaml | 时态混乱应减少 |

### 详细改动

#### 改进 1: 删除 Medication_Plan_chatgpt
- **文件**: `prompts/plan_extraction.yaml` — 删除 Medication_Plan_chatgpt 整个 prompt（原第 1-20 行）
- **文件**: `ult.py` — 从 `PLAN_KEYS` 集合中移除 `'Medication_Plan_chatgpt'`
- **原因**: ~45% 空率，且 0 行有 chatgpt 独有的有用内容（medication_plan 严格更好）

#### 改进 2: 修复 supportive_meds 过滤
- **文件**: `ult.py` `filter_supportive_meds()` 函数
- **改动**: `combined = supportive_whitelist | oncology_whitelist` → 只用 `supportive_whitelist`
- **原因**: 合并导致 letrozole/tamoxifen 等 oncology 主药通过 supportive_meds 过滤
- **验证**: denosumab/xgeva/zometa 已在 `supportive_care_drugs.txt` 的 BONE-PROTECTING 分类中

#### 改进 3: Stage 幻觉修复
- **Step A**: `prompts/extraction.yaml` Cancer_Diagnosis
  - 示例从 "Originally Stage IIA" 改为 "Originally Stage [X]"
  - 新增说明：只有原文明确写出原始分期才能用 Originally
- **Step B**: `run.py` 新增 POST-STAGE-VERIFY
  - 检测 "Originally Stage [X]" → 在原文中搜索该分期号
  - 找不到 → 去掉 "Originally Stage X" 部分

#### 改进 4: Cross-field 白名单过滤
- **4A**: 创建 `data/lab_tests.txt` (71 个术语)，新增 POST-LAB-WHITELIST
  - 排除 lumbar puncture, biopsy, imaging, genomic 等非 lab 项
- **4B**: 新增 POST-PROCEDURE-FILTER
  - 排除 IHC, FISH, Oncotype, BRCA, genomic 等属于 genetic_testing_plan 的项
- **4C**: 新增 POST-IMAGING-FILTER
  - 排除 biopsy, thoracentesis, lumbar puncture 等属于 procedure 的项

#### 改进 5: therapy_plan prompt 修复
- **文件**: `prompts/plan_extraction.yaml` Therapy_plan prompt
- **新增**: "continue [drug]" IS a valid therapy plan. Do NOT write "None" when patient is actively on treatment.

#### 改进 6: genetic_testing_plan 白名单搜索
- **新建**: `data/genetic_tests.txt` (38 个关键词)
- **新增**: `run.py` POST-GENETICS-SEARCH
  - 搜索全文（不只 A/P），匹配 genetic_tests.txt 中的关键词
  - 只有带 future context（will order, send for, plan to 等）时才添加

#### 改进 7: Referral 使用全文提取
- **文件**: `run.py`
- **改动**: Referral 从 plan_extraction_prompts 中 pop 出来，单独用 full-note cache 提取
- **原因**: 很多 referral 写在 HPI / Diagnosis section，不在 A/P 段
- **保留**: POST-REFERRAL 规则继续作为补充验证

#### 改进 8: current_meds 时态 prompt 改进
- **文件**: `prompts/extraction.yaml` Current_Medications prompt
- **新增**: CROSS-REFERENCE 指示 — 如果存在 "Current Outpatient Medications" 段，以其为主要参考

### 新增文件
- `data/lab_tests.txt` — lab 测试白名单（71 术语）
- `data/genetic_tests.txt` — 基因检测关键词列表（38 术语）

### 待验证行
需要在 WSL 上重新跑以下有问题的行来验证改进效果。

---
