# v16 改进追踪

创建日期：2026-03-18
基于：v15a 审查（61 行，问题密度 0.74/行）
目标：进一步降低问题密度，从 0.74 → < 0.40

---

## 优先改进（P0/P1）

### 改进 1：Referral Specialty 文本泄漏修复
- **优先级**: P0 (1 行) + P1 (3 行)
- **受影响行**: Row 4 (P0), Row 52 (P1), Row 83 (P1)
- **问题描述**: POST-REFERRAL regex 捕获组过宽，导致非 specialty 文本串入
  - Row 4: "Radiation Oncology    CT Abdomen /Pelvis" — CT orders 串入
  - Row 83: "radiation oncology for consideration of, radiation oncology to consider either fo" — 同一 referral 重复匹配
  - Row 52: "Referral will be made to [REDACTED] at the completion of chemotherapy..." — 整句串入
- **根因**: `run.py:799` `match_short = match[:40]` 截断太长；regex 捕获组 `[^,.\n:;()\-–—]+` 仍过宽
- **修复方案**:
  1. 截断长度从 40 → 25 字符
  2. 新增短语黑名单: `"for consideration of"`, `"at the completion"`, `"CT "`, `"MRI"`, `"will be made"`
  3. 去重: 如果 Specialty 已含 "radiation oncology" 则不再追加
  4. 只保留 SPECIALTY_KEYWORDS 匹配部分，不取整个 capture group
- **修改文件**: `run.py:746-802`
- **验证**: Row 4, 52, 83 的 Specialty 应只含科室名（≤25 字符）
- **状态**: ✅ 已实现 (v16 commit 28f4251)

---

### 改进 2：POST-ER-CHECK — 从药物推断 ER 状态
- **优先级**: P1
- **受影响行**: Row 10, Row 82 (共 2 行)
- **问题描述**: 当 LLM 完全不提取受体状态时（如 "infiltrating ductal carcinoma", "Lobular Breast Cancer"），POST-HER2-CHECK 不触发（需要已有 ER/PR 关键词）
- **根因**: `run.py:1297` 触发条件 `re.search(r'(?i)\b(?:ER|HR|PR)[+-]...'` — 必须已有 ER 才查 HER2
- **修复方案**: 在 POST-HER2-CHECK 之前新增 POST-ER-CHECK：
  1. 检查 Type_of_Cancer 是否缺 ER/PR
  2. 从 `current_meds` 交叉推断:
     - Faslodex/fulvestrant/tamoxifen/letrozole/anastrozole/exemestane/arimidex → ER+
     - Zoladex/goserelin/leuprolide (+ 上述药物) → ER+
  3. 追加推断的 ER 状态到 Type_of_Cancer → 触发 POST-HER2-CHECK
  4. 也检查 note 全文中的 receptor 关键词（ER positive, ER+, estrogen receptor positive 等）
- **修改文件**: `run.py` — POST-HER2-CHECK 之前新增 ~30 行
- **验证**:
  - Row 10: Faslodex → ER+ 追加 → HER2 检查触发
  - Row 82: letrozole → ER+ 追加 → HER2 检查触发（但 HER2 脱敏，结果为 "not tested"）
- **风险**: 低。推断方向明确（这些药物只用于 ER+ 患者）
- **状态**: ✅ 已实现 (v16 commit 28f4251)

---

### 改进 3：POST-MEDS-FILTER — 黑名单过滤非 cancer 药物
- **优先级**: P1
- **受影响行**: Row 71 (至少 1 行)
- **问题描述**: Prompt 已要求 "Do NOT include: eye drops, blood pressure meds..." (extraction.yaml:133) 但 Qwen 32B 不遵守。Row 71 current_meds 含 latanoprost (XALATAN) 眼药水
- **根因**: 纯靠 prompt 约束不可靠，缺少后处理兜底
- **修复方案**: 在 POST-DRUG-VERIFY (run.py:1260) 之后新增 POST-MEDS-FILTER:
  1. **明确非 cancer 药物黑名单**:
     - 眼科: `ophthalmic`, `eye drop`, `latanoprost`, `timolol`, `brimonidine`
     - 血压: `lisinopril`, `amlodipine`, `losartan`
     - 糖尿病: `metformin`, `insulin`, `glipizide`
     - 维生素: `vitamin`, `fish oil`, `calcium carbonate`（但 calcium+vitamin D 如果是骨保护相关，保留）
     - 过敏: `allegra`, `zyrtec`, `cetirizine`, `fexofenadine`
     - 其他: `buspirone`, `citalopram`（精神科用药）
  2. **格式清理**: 去除 "Every 6 Hours;" 等用药频率前缀（Row 71 特有问题）
  3. **安全策略**: 只过滤 current_meds 中的非 cancer 药物。对 PPI (omeprazole) 等边界药物保留（可能是化疗 supportive）
  4. **白名单保护**: 如果药物在 oncology_whitelist 中，即使也在黑名单中，保留
- **修改文件**: `run.py` — POST-DRUG-VERIFY 之后新增 ~25 行
- **验证**: Row 71 的 current_meds 应只含 zoledronic acid (Reclast)
- **风险**: 中。需要仔细选择黑名单，避免误删 supportive care 药物
- **状态**: ✅ 已实现 (v16 commit 28f4251)

---

### 改进 4：genetic_testing_plan 时态/拒绝检测
- **优先级**: P1
- **受影响行**: Row 17, Row 53, Row 79 (共 3 行)
- **问题描述**: 三种混淆模式：
  1. Row 17: "molecular profiling" — 患者明确说 "will not pursue molecular profiling"
  2. Row 53: "brca2" — 这是已知突变结果 (Problem List)，非未来 plan
  3. Row 79: "Whole genome sequencing was done" — 已完成测试
- **根因**:
  - LLM 提取时没区分 plan vs result vs declined
  - POST-GENETICS-SEARCH (run.py:1103) 的 PAST_CONTEXT 没覆盖 "will not pursue"/"declined" 等
  - POST-GENETICS-SEARCH 只处理 "None planned" 的情况，不处理 LLM 已提取但错误的情况
- **修复方案**:
  1. **扩展 POST-GENETICS-SEARCH 的 PAST_CONTEXT** (run.py:1117): 新增
     - `"will not pursue"`, `"declined"`, `"not interested"`, `"deferred"`, `"refuses"`, `"does not wish"`, `"patient does not want"`
  2. **新增 POST-GENETICS-RESULT-CHECK**: 在 POST-GENETICS-SEARCH 之后
     - 检查 LLM 提取的值本身:
       - 如果值是纯突变名 (brca1, brca2, pms2, chek2, palb2, atm, tp53) 且无 "test"/"order"/"send" → 标记为结果，清空或改为 "Known [mutation] mutation (not a future plan)"
       - 如果值包含 "was done"/"results reviewed"/"results show"/"completed" → 清空
     - 检查原文上下文:
       - 在 term 附近 100 字符内搜索 decline/refuse/not pursue → 清空
  3. **保护正确提取**: Row 96 "molecular profiling"（患者同意 "wishes to proceed"）不受影响
- **修改文件**: `run.py:1103-1153` (扩展) + 新增 POST-GENETICS-RESULT-CHECK ~20 行
- **验证**:
  - Row 17: genetic_testing_plan 应为 "None planned" 或 "Discussed molecular profiling but patient declined"
  - Row 53: genetic_testing_plan 应为 "None planned" (BRCA2 是已知结果)
  - Row 79: genetic_testing_plan 应为 "None planned" (WGS 已完成)
  - Row 96: genetic_testing_plan 应保持 "molecular profiling" (患者同意)
- **状态**: ✅ 已实现 (v16 commit 28f4251)

---

### 改进 5：Stage IV vs Regional LN 校验
- **优先级**: P1
- **受影响行**: Row 82 (1 行)
- **问题描述**: Row 82 Stage = "Stage IV" 但原文 "W/u negative for distant metastasis"。Axillary LN = regional, 非 distant → 应为 Stage III
- **根因**: POST-STAGE (run.py:1173) 检查 Metastasis="No" + Stage IV 的矛盾。但 Row 82 Metastasis="Yes, to right axilla LN"，不是 "No"，所以不触发
- **修复方案**: 扩展 POST-STAGE 逻辑：
  1. **检查 Metastasis 内容是否仅含 regional sites**:
     - Regional: `axillary`, `axilla`, `sentinel`, `supraclavicular`, `infraclavicular`, `internal mammary`, `chest wall`
     - Distant: `liver`, `lung`, `bone`, `brain`, `pleural`, `peritoneal`, `ovary`, `skin`
     - 如果 Metastasis 仅含 regional sites → 不应是 Stage IV
  2. **搜索原文**: 如果笔记中有 `"negative for distant metastasis"` / `"no distant metastasis"` / `"no evidence of distant"` → 移除 Stage IV 标记
  3. **修改行为**: 如果检测到矛盾 → Stage 改为 "Originally Stage [X], regional disease" 或直接移除 "Stage IV" 部分
- **修改文件**: `run.py:1173-1189` (扩展)
- **验证**: Row 82 Stage 不应包含 "Stage IV"（axillary LN = regional）
- **风险**: 低。逻辑清晰 — 仅 regional LN + 原文说 no distant mets 时触发
- **状态**: ✅ 已实现 (v16 commit 28f4251)

---

## 非优先改进（P2/观察）

### 改进 6：current_meds 归因覆盖率
- **优先级**: P2（观察）
- **问题**: current_meds 归因覆盖率仅 42%（50 行中 29 行缺失）。其他字段 ≥72%
- **根因**: 药物列表通常是结构化格式（medication list section），LLM 归因时难以找到支持引文
- **可能方案**: 修改 attribution prompt，对 current_meds 允许引用 medication list section 的条目
- **状态**: ⬜ 低优先级，暂不实现

---

### 改进 7：current_meds clinic-administered chemo 遗漏
- **优先级**: P2
- **受影响行**: Row 1 (遗漏 irinotecan), Row 6 (遗漏 Herceptin/Taxotere)
- **问题**: 诊所内给药的化疗（IV chemo）不在 "outpatient medication list" 中，虽然 prompt 有 EXCEPTION (extraction.yaml:137) 提到 "if A/P says patient is CURRENTLY ON a drug, include it"，但模型仍遗漏
- **可能方案**: 增强 EXCEPTION 部分的示例，或新增 POST-MEDS-IV-CHECK 搜索 A/P 中 "on [drug]"/"continue [drug]" 模式
- **状态**: ⬜ 低优先级

---

### 改进 8：imaging/lab plan orders 区域遗漏
- **优先级**: P2
- **受影响行**: Row 0 (遗漏 Bone Scan, MRI Brain, CBC 等 orders)
- **问题**: Orders 在 note 的 Order 区域，不在 A/P 段中。plan_extraction 只搜索 A/P
- **可能方案**: 让 plan_extraction 也搜索 note 中 "Orders" section，或新增 POST-IMAGING-SEARCH 类似 POST-GENETICS-SEARCH
- **备注**: Row 0 是极端情况 — 大多数 note 的 orders 在 A/P 中
- **状态**: ⬜ 低优先级

---

### 改进 9：genetic_testing_plan 格式（第一人称叙述）
- **优先级**: P2
- **受影响行**: Row 87
- **问题**: genetic_testing_plan 输出为医生第一人称叙述 "I recommending doing her 2 on the brain metastasis..."，应简化为 "Recheck HER2 and hormone receptors on brain metastasis"
- **可能方案**: 在 prompt 中加示例强调简洁格式，或后处理截断
- **状态**: ⬜ 低优先级

---

### 改进 10：response_assessment 描述治疗状态而非疗效
- **优先级**: P2
- **受影响行**: Row 89, Row 91
- **问题**:
  - Row 89: "Patient is currently on cycle 2 of AC treatment" — 这是治疗进度，非 response assessment
  - Row 91: "Labs and exam show stable disease. Liver size decreased" — 基于 exam/labs 的 "stable"，缺乏影像证据
- **可能方案**: 已有 Fix 6 prompt 改进，但效果有限。可考虑后处理检测 "on cycle" 等治疗进度描述
- **状态**: ⬜ 低优先级

---

### 改进 11：Type_of_Cancer 数据脱敏限制
- **优先级**: P2（系统限制，无法完全解决）
- **受影响行**: Row 41 (*****/neu 脱敏), Row 82 (biopsy 受体脱敏), Row 13 (HER2 脱敏)
- **问题**: 数据集中 HER2/ER 被 `*****` 替代，模型无法识别
- **可能方案**: 无法在 pipeline 层面解决。POST-ER-CHECK (改进 2) 可部分缓解
- **状态**: ⬜ 接受为数据集限制

---

### 改进 12：Stage [X] 占位符改进
- **优先级**: P2
- **受影响行**: Row 82, 85, 87, 91, 99 等 (约 6 行)
- **问题**: 原始分期未知时用 "Originally Stage [X]" 占位。一些情况可从 tumor size + LN status 推断
- **可能方案**: POST-STAGE 中加推断逻辑（如 T1N0 = Stage I），但风险较高（推断可能不准确）
- **状态**: ⬜ 低优先级

---

### 改进 13：Referral Specialty 含描述性文本 (P2)
- **优先级**: P2
- **受影响行**: Row 28 ("RT planning per [REDACTED], likely pursued locally"), Row 35 ("will see Dr. [REDACTED] next week")
- **问题**: Specialty 字段包含时间/地点信息，而非纯科室名。区别于 P0/P1 的文本泄漏
- **可能方案**: 改进 1 的长度截断 (25 字符) 会自然解决大部分
- **状态**: ⬜ 改进 1 覆盖

---

### 改进 14：Row 64 FISH 数据格式
- **优先级**: P2
- **受影响行**: Row 64
- **问题**: Type = "HER2 neg (IHC 2+, FISH 2.March 09.8=1.4)" — "2.March 09.8" 是原文中的格式问题（可能是日期和 FISH ratio 混在一起）
- **根因**: 原文数据格式异常，非 pipeline 问题
- **状态**: ⬜ 接受为数据格式问题

---

### 改进 15：procedure_plan vs therapy_plan 分类
- **优先级**: P2
- **受影响行**: Row 7 (化疗方案出现在 procedure_plan)
- **问题**: "adjuvant AC x 4 cycles" 应在 therapy_plan 而非 procedure_plan
- **可能方案**: POST-PROCEDURE-FILTER 已有化疗 blacklist，但 "AC x 4 cycles" 可能未匹配（无完整药名）
- **状态**: ⬜ 低优先级

---

## 改善追踪统计

| 版本 | 问题密度 | P0 | 改善 |
|------|---------|-----|------|
| v13a | 3.92 | 2 | — |
| v14a | 2.67 | 3 | -32% |
| v15a | 0.74 | 1 | -72% |
| v16 (目标) | < 0.40 | 0 | — |

### 预期改善

| 改进 | 预计消除问题数 | 密度影响 |
|------|--------------|---------|
| 改进 1 | 4 (1 P0 + 3 P1) | -0.07 |
| 改进 2 | 2 (2 P1) | -0.03 |
| 改进 3 | 1 (1 P1) | -0.02 |
| 改进 4 | 3 (3 P1) | -0.05 |
| 改进 5 | 1 (1 P1) | -0.02 |
| **合计** | **11** | **-0.19** |
| **预期 v16 密度** | | **~0.55** |

注：预期值保守。部分改进可能在新数据上暴露新的边缘情况。
