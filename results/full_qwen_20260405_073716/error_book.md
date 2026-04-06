# 错题本: v24 Full Run P1 Issues

> 来源: full_qwen_20260405_073716/review.md
> 用途: 跟踪每个 P1 问题的修复状态，验证修复效果

---

## Letter 幻觉/编造 (3 P1)

| ROW | coral_idx | 问题 | 修复方式 | 验证状态 |
|-----|-----------|------|----------|----------|
| 3 | 142 | Letter "You appear to be emotional" 但 ROS 说 no anxiety/depression | letter_generation.yaml 规则 16 修改 | 🔲 待验证 |
| 4 | 143 | Letter "blood tests mostly normal" 但 lab_summary="No labs in note" | letter_generation.yaml 规则 11 修改 | 🔲 待验证 |
| 5 | 144 | Letter "blood tests mostly normal" 但 lab_summary="No labs in note" | 同上 | 🔲 待验证 |

**验证 sample**: ROW 3 (coral_idx 142), ROW 4 (coral_idx 143), ROW 5 (coral_idx 144)
**需要 letter**: ✅

### 验证结果 (2026-04-05)
- ROW 3 "emotional": ✅ **已修复** (prompt + POST-LETTER-EMOTIONAL hook)
- ROW 4 "blood tests": ✅ **已修复** (prompt 规则 11)
- ROW 5 "blood tests": ✅ **已修复** (prompt 规则 11)

---

## response_assessment 误判 (3 P1)

| ROW | coral_idx | 问题 | 修复方式 | 验证状态 |
|-----|-----------|------|----------|----------|
| 9 | 148 | "Not yet on treatment" 但完成新辅助化疗+手术（部分病理响应） | extraction.yaml Response_Assessment prompt 修改 | 🔲 待验证 |
| 10 | 149 | "Not mentioned" 但 8.8cm 残余+20LN+ = 新辅助疗效差 | 同上 | 🔲 待验证 |
| 11 | 150 | 把 letrozole 上的 PET 进展归因于 Faslodex（A/P 说 stable） | 同上（时间线指导） | 🔲 待验证 |

**验证 sample**: ROW 9 (coral_idx 148), ROW 10 (coral_idx 149), ROW 11 (coral_idx 150)
**需要 letter**: ✅ (ROW 11 letter 也有 P1 — "not responding well")

### 验证结果 (2026-04-05)
- ROW 9: ✅ **已修复** — 正确描述 post-neoadjuvant 病理响应
- ROW 10: ✅ **已修复** — 正确描述新辅助后手术结果
- ROW 11: ✅ **已修复** — 不再说 "not responding"，时间线正确

---

## genetic_testing_plan 误分类 (3 P1)

| ROW | coral_idx | 问题 | 修复方式 | 验证状态 |
|-----|-----------|------|----------|----------|
| 3 | 142 | biopsy/IHC for HR/HER2 错归为 genetic testing | plan_extraction.yaml 排除 IHC/FISH | 🔲 待验证 |
| 7 | 146 | "recheck [LVEF/tumor marker]" 错归为 genetic testing | plan_extraction.yaml 排除 LVEF/markers | 🔲 待验证 |
| 24 | 163 | Oncotype/MammaPrint 明确计划中但写 "None planned" | plan_extraction.yaml 明确包含 Oncotype | 🔲 待验证 |

**验证 sample**: ROW 3 (coral_idx 142), ROW 7 (coral_idx 146), ROW 24 (coral_idx 163)
**需要 letter**: ✅

### 验证结果 (2026-04-05)
- ROW 3: ✅ **已修复** — "Genetic testing sent and is pending" (不再包含 IHC/biopsy)
- ROW 7: ✅ **已修复** — "None planned." (POST-GENETICS-RECHECK hook 清除了 "Would recheck")
- ROW 24: ✅ **已修复** — "We will send her surgical specimen for MP" (正确捕获分子检测)

---

## 其他 P1 (11 个)

| ROW | coral_idx | 字段 | 问题 | 修复方式 | 验证状态 |
|-----|-----------|------|------|----------|----------|
| 1 | 140 | lab_plan | 混入 MRI/bone scan + 没列具体化验 | prompt 改进 | 🔲 待验证 |
| 6 | 145 | Patient type | CC 说 Follow-up 提取为 New patient | POST hook 已覆盖? | 🔲 待验证 |
| 11 | 150 | letter | "not responding well" 但 A/P 说 stable | response_assessment 修复应连带解决 | 🔲 待验证 |
| 12 | 151 | Advance care | "Not discussed" 但 problem list 明确 DNR/DNI | 需考虑搜索 problem list | 🔲 待验证 |
| 14 | 153 | current_meds | 空值，患者正在服用 gem/doc/dox/pamidronate | prompt 需更积极提取 | 🔲 待验证 |
| 14 | 153 | response_assessment | 漏了 CA 27.29 下降趋势 + CT 缩小 | prompt 已加指导 | 🔲 待验证 |
| 20 | 159 | procedure_plan | 混入 CT/referral/genetic testing | procedure prompt 已修复 | 🔲 待验证 |
| 22 | 161 | lab_summary | "No labs" 但有临床显著 labs (贫血/肾功能不全) | 需定义 old labs 策略 | 🔲 待验证 |
| 25 | 164 | medication_plan | Xeloda 剂量错归给 ixabepilone | 需更精确的剂量归因 | 🔲 待验证 |

---

## 验证计划

### 需要重跑的 sample (去重后)

Letter P1 验证: coral_idx 142, 143, 144
Response assessment P1 验证: coral_idx 148, 149, 150
Genetic testing P1 验证: coral_idx 142, 146, 163
其他 P1 验证: coral_idx 140, 145, 150, 151, 153, 159, 161, 164

**去重合并**: coral_idx = 140, 142, 143, 144, 145, 146, 148, 149, 150, 151, 153, 159, 161, 163, 164

**共 15 个 sample** 需要重跑验证。

### row_indices 配置 (0-based)
对应 CSV row_indices (coral_idx → row_index 需要查映射):
需要查 breastca_unannotated.csv 中这些 coral_idx 对应的行号。

### 验证步骤
1. 创建 exp/v24_verify.yaml，row_indices = [上述 15 个 sample 的 row_index]
2. SSH 到 WSL 跑 pipeline
3. 下载结果，逐个对比：
   - Letter: 是否消除了 "blood tests normal" / "emotional" 幻觉？
   - response_assessment: post-neoadjuvant 是否描述病理响应？
   - genetic_testing_plan: IHC/LVEF 是否排除？Oncotype 是否捕获？
   - POST-STAGE-METASTATIC: Stage "Not available" 是否自动变为 Stage IV？
4. 记录结果到此错题本
