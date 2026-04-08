# v15 审查报告（进行中）

审查日期：2026-03-17
版本：v15（基于 v14a 审查的 7 项修复）
状态：Pipeline 运行中，前 34/61 行已审查
审查方法：逐行对照原文 + 归因 + v14a 问题追踪

---

## 一、v15 修复效果验证

### Fix 1: Referral 文本泄漏 [v14a P0, 2 行]
| 行 | v14a | v15 | 状态 |
|----|------|-----|------|
| Row 0 | "Integrative Medicine, Integrative Medicine History of Present Illness: 56" | "Integrative Medicine" | ✅ 修复 |
| Row 4 | "Radiation oncology referral, Radiation Oncology CT Abdomen /Pelvis with Contrast CT" | "Radiation oncology referral, Radiation Oncology    CT Abdomen /Pelvis" | ⚠️ 部分修复 — "CT Abdomen /Pelvis" 仍然泄漏 |

**分析**: Row 0 完美修复。Row 4 的泄漏来自 LLM 提取输出本身（A/P 段落中 orders 列表紧跟 referral 文本），POST-REFERRAL regex 修复无法解决 LLM 层面的问题。需要在 plan_extraction prompt 中加强 Specialty 字段的约束。

### Fix 2: HER2+/triple-negative 矛盾 [v14a P0, 1 行]
| 行 | v14a | v15 | 状态 |
|----|------|-----|------|
| Row 56 | "ER-/PR-/HER2+ triple negative breast cancer" | "ER-/PR-/HER2+ breast cancer" | ✅ 修复 |

**分析**: POST-TYPE-VERIFY 正确移除了矛盾的 "triple negative"。保留 HER2+ 是合理的（与 TCH+ 治疗方案一致，虽然术后病理有不一致）。

### Fix 3: "ngs" 假阳性 [v14a P1, 4 行]
| 行 | v14a | v15 | 状态 |
|----|------|-----|------|
| Row 28 | genetic_testing_plan 含 "ngs" | "None planned." | ✅ 修复 |
| Row 29 | genetic_testing_plan 含 "ngs" | "None planned." | ✅ 修复 |
| Row 62 | genetic_testing_plan 含 "ngs" | ⏳ 待完成 | — |
| Row 93 | genetic_testing_plan 含 "ngs" | ⏳ 待完成 | — |

**分析**: 已验证的 2 行完全修复。word boundary `\b` 匹配和删除 genetic_tests.txt 中的 "ngs" 短词均生效。

### Fix 4: Type_of_Cancer 缺 HER2 [v14a P1, 11 行]

**POST-HER2-CHECK 代码存在 3 个 BUG:**

#### Bug 4a: "her 2" 格式不匹配 (P0 级)
关键词列表 `["her2", "her-2", "her2neu", ...]` 不包含 "her 2"（带空格）。大量临床笔记使用 "her 2" 格式。
- **Row 0**: 笔记 3 处写 "her 2 neu negative" / "her 2 negative" → 代码未匹配 → 错误标为 "HER2: not tested"
- **Row 9**: 笔记写 "her 2 negative" → 代码未匹配 → HER2 完全缺失（因为 "HR+" 也不匹配 `\bER[+-]`）
- **影响**: 所有使用 "her 2" 格式的笔记都受影响

#### Bug 4b: ER 格式匹配不完整 (P1 级)
正则 `\bER[+-]` 仅匹配 "ER+", "ER-"，不匹配：
- "ER positive" / "ER negative"（Row 33 = "ER positive PR negative"）
- "HR+"（Row 9 = "HR+ invasive ductal carcinoma"）
- 仅有 PR 无 ER 的情况（Row 41 = "PR+ invasive ductal carcinoma"）

#### Bug 4c: "amplified" vs "non-amplified" 判断顺序错误 (P0 级)
代码先检查 `"amplified" in ctx` 再检查 `"not amplified" in ctx`。但 "non-amplified" 包含 "amplified" → 错误判定为 HER2+。
- **Row 5**: 笔记 "IHC 2, FISH non-amplified" = HER2 阴性 → 代码输出 "HER2+" **错误！引入了比 v14a 更严重的问题**

| 行 | v14a | v15 | 状态 |
|----|------|-----|------|
| Row 0 | 缺 HER2 | "HER2: not tested" ← 错误（应为 HER2-） | ❌ Bug 4a |
| Row 5 | 缺 HER2 | "HER2+" ← **严重错误**（应为 HER2-） | ❌ Bug 4c (P0) |
| Row 9 | 缺 HER2 | 仍缺 HER2 | ❌ Bug 4b |
| Row 10 | 缺 HER2 | 仍缺 HER2（无 ER/PR 触发条件） | ❌ 无法修复 |
| Row 13 | 缺 HER2 | "HER2: not tested" ← 正确（笔记无 HER2 信息） | ✅ 修复 |
| Row 33 | 缺 HER2 | 仍缺 HER2（"ER positive" 格式不匹配） | ❌ Bug 4b |
| Row 41 | 缺 HER2 | 仍缺 HER2（仅 PR+ 无 ER） | ❌ Bug 4b |
| Row 82 | 缺 HER2 | ⏳ 待完成 | — |
| Row 90 | 缺 HER2 | ⏳ 待完成 | — |
| Row 96 | 缺 HER2 | ⏳ 待完成 | — |
| Row 99 | 缺 HER2 | ⏳ 待完成 | — |

**结论**: Fix 4 仅修复了 1/7 已验证的行（Row 13），且引入了 1 个 P0 新 bug（Row 5 HER2 反转）。需要紧急修补。

### Fix 5: Attribution JSON wrapper 泄漏 [v14a P1, 11 行]
| 指标 | v14a | v15 | 状态 |
|------|------|-----|------|
| JSON wrapper 泄漏次数 | 17 次 / 11 行 | 0 次 | ✅ 完美修复 |
| 归因覆盖率 | ~70% | 72.9% | ✅ 略有提升 |

**分析**: JSON 解包逻辑完美生效。零泄漏。

### Fix 6: response_assessment 过度解读 [v14a P1, ≥3 行]
| 行 | v14a | v15 | 状态 |
|----|------|-----|------|
| Row 10 | 把治疗前 PET 误解读为治疗后进展 | "PET/CT showed increased metastatic activity...not responding to Faslodex" — 仍用 PET 数据但说 "not responding"，PET 在开始 Faslodex 之前拍的 | ⚠️ 部分改善 |
| Row 13 | "currently responding" 无足够证据 | "No evidence of new or worsening...on MRI Total Spine" — 用具体影像语言 | ✅ 修复 |
| Row 45 | "currently responding" | "currently responding to treatment. Imaging shows 3.5 cm residual disease..." — 仍用 "currently responding"，但有影像证据 | ⚠️ 有证据但解读有争议 |
| Row 82 | "currently responding" | ⏳ 待完成 | — |

**分析**: Row 13 改善明显。Row 45 仍然使用 "currently responding"，虽然引用了影像证据但将术后残余病灶和阳性切缘解读为 "responding" 值得商榷。prompt 修改有效果但不够彻底。

### Fix 7: current_meds 非 cancer 药物 [v14a P1, ≥1 行]
| 指标 | v14a | v15 | 状态 |
|------|------|-----|------|
| 非 cancer 药物 | Row 71 有 latanoprost | ⏳ Row 71 待完成 | — |
| 前 34 行检查 | — | 未发现非 cancer 药物 | ✅ 初步良好 |

---

## 二、v15 新引入的问题

### 新 P0：Row 5 HER2 状态反转
**严重**: POST-HER2-CHECK 将 IHC 2+/FISH non-amplified（= HER2-）错误标为 HER2+。
- 原因: 代码中 `"amplified" in ctx` 在 `"not amplified" in ctx` 之前检查，"non-amplified" 包含 "amplified" 子串
- 影响: 可能导致患者被建议使用 HER2 靶向治疗（错误的治疗方案）
- 修复: 将 negative 条件检查（含 "not amplified", "non-amplified"）移到 positive 检查之前

### 新 P0：Row 0 HER2 "not tested" 标注错误
**严重**: 笔记明确写 "her 2 neu negative" 三次，但 POST-HER2-CHECK 未匹配（"her 2" 格式不在关键词列表中），默认标为 "HER2: not tested"
- 修复: 在 HER2_SEARCH_KEYWORDS 中添加 "her 2", "her 2 neu"

---

## 三、v15 修复效果总结（34 行）

| Fix | 目标 | 修复率 | 新问题 | 评级 |
|-----|------|--------|--------|------|
| Fix 1 Referral 泄漏 | 2 行 P0 | 1/2 = 50% | 无 | ⚠️ |
| Fix 2 HER2/TN 矛盾 | 1 行 P0 | 待验证 | — | ⏳ |
| Fix 3 ngs 假阳性 | 4 行 P1 | 2/2 = 100%（2 待验证） | 无 | ✅ |
| Fix 4 HER2 缺失 | 11 行 P1 | 1/7 = 14%（4 待验证） | 1 个 P0 反转 | ❌ 需修补 |
| Fix 5 JSON wrapper | 11 行 P1 | 11/11 = 100% | 无 | ✅ |
| Fix 6 response_assessment | ≥3 行 P1 | 1/3 = 33%（1 待验证） | 无 | ⚠️ |
| Fix 7 non-cancer meds | ≥1 行 P1 | 待验证 | 无 | ⏳ |

---

## 四、逐行详细审查（34 行）

### Row 0 (coral_idx=140) — 新患者，转移性 ER+/PR+ IDC

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Patient type | New patient | 不变 | OK |
| second opinion | no | 不变 | OK |
| in-person | in-person | 不变 | OK |
| summary | 56-year-old female with newly diagnosed metastatic ER+ breast cancer... | 不变 | OK |
| Type_of_Cancer | ER+/PR+ invasive ductal carcinoma, HER2: not tested | v14a "ER+/PR+ IDC" → v15 加了 "HER2: not tested" | **P0** 错误！笔记有 "her 2 negative" |
| Stage | Originally Stage IIA, now metastatic (Stage IV) | 不变 | OK |
| Metastasis | Yes (to lungs, peritoneum, liver, ovaries) | 不变 | OK |
| lab_summary | No labs in note. | 不变 | OK |
| findings | Widespread metastases... | 不变 | OK |
| current_meds | "" | 不变 | OK |
| goals_of_treatment | palliative | 不变 | OK |
| response_assessment | Not yet on treatment — no response to assess. | v14a "Metastatic relapse..." → v15 更准确 | ✅ 改善 |
| imaging_plan | No imaging planned. | 不变 | **P1** 笔记有 bone scan + MRI brain orders |
| lab_plan | No labs planned. | 不变 | **P1** 笔记有 CBC/CMP/CA15-3/CEA orders |
| Referral: Specialty | Integrative Medicine | v14a 有文本泄漏 → v15 干净 | ✅ 修复 |

### Row 1 (coral_idx=141) — 转移性 TNBC

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER-/PR-/HER2- triple negative IDC | 不变 | OK（完整） |
| current_meds | "" | 不变 | **P1** 应有 irinotecan |
| response_assessment | PET/CT 05/31/19 showed significantly increased metastases... | 不变，改善 | OK |

### Row 2 (coral_idx=142) — 新患者，Stage IIA

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- IDC | 不变 | OK（完整） |
| response_assessment | Not yet on treatment — no response to assess. | 不变 | OK |

### Row 4 (coral_idx=144) — 转移性 ER+/HER2- IDC

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER+/PR+/HER2- IDC | 不变 | OK |
| current_meds | anastrozole, palbociclib, leuprolide | 不变（改进 8 修复持续生效） | OK |
| Referral: Specialty | "Radiation oncology referral, Radiation Oncology    CT Abdomen /Pelvis" | v14a 更长 → v15 略短但仍有泄漏 | **P0** 仍有泄漏 |

### Row 5 (coral_idx=145) — 早期 ER+/PR+ IDC，adjuvant

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER+/PR+ IDC, HER2+ | v14a 缺 HER2 → v15 加了 HER2+ | **P0** 错误！IHC 2/FISH non-amplified = HER2- |
| Stage | "" | 不变 | **P1** 仍然为空 |
| current_meds | zoladex, letrozole | 不变 | OK |

### Row 9 (coral_idx=149) — HR+ IDC

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | HR+ invasive ductal carcinoma | 不变 | **P1** 仍缺 HER2（Bug 4b） |

### Row 10 (coral_idx=150) — infiltrating ductal carcinoma

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | infiltrating ductal carcinoma | 不变 | **P1** 完全缺 ER/PR/HER2 |
| response_assessment | PET/CT showed increased metastatic activity...not responding to Faslodex | 改善（不再说 "currently responding"） | **P1** 时间线错误：PET 在开始 Faslodex 之前 |

### Row 13 (coral_idx=153) — ER+ IDC with spine mets

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER+ IDC, HER2: not tested | v14a 缺 HER2 → v15 加了 "not tested" | ✅ 正确（笔记无 HER2） |
| response_assessment | No evidence of new or worsening osseous or epidural... | v14a "currently responding" → v15 用 MRI 语言 | ✅ 显著改善 |

### Row 28 (coral_idx=168)

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| genetic_testing_plan | None planned. | v14a 有 "ngs" 假阳性 → v15 修复 | ✅ 修复 |

### Row 29 (coral_idx=169)

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| genetic_testing_plan | None planned. | v14a 有 "ngs" 假阳性 → v15 修复 | ✅ 修复 |

### Row 33 (coral_idx=173)

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | ER positive PR negative IDC | 不变 | **P1** 缺 HER2（Bug 4b） |

### Row 41 (coral_idx=181)

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| Type_of_Cancer | PR+ invasive ductal carcinoma | 不变 | **P1** 缺 HER2 + 缺 ER（Bug 4b） |

### Row 45 (coral_idx=185)

| 字段 | 值 | v14a→v15 | 判定 |
|------|-----|---------|------|
| response_assessment | "currently responding to treatment. Imaging shows 3.5 cm residual disease..." | 不变 | **P1** 仍用 "currently responding"，且将残余病灶/阳性切缘解读为 "responding" |

---

## 五、v15→v16 修复建议

### 紧急 (P0)

1. **POST-HER2-CHECK: 修复 "amplified" 判断逻辑**
   - 将 negative 条件检查（含 "non-amplified", "non amplified", "not amplified"）移到 positive 条件之前
   - 或者在 positive 条件中排除 "non-amplified" 前缀

2. **POST-HER2-CHECK: 添加 "her 2" 格式关键词**
   - 在 HER2_SEARCH_KEYWORDS 中添加 "her 2" 和 "her 2 neu"
   - 很多临床笔记使用 "her 2" 而非 "her2"

3. **POST-HER2-CHECK: 扩展 ER 格式匹配**
   - 正则改为: `re.search(r'(?i)\b(?:ER|HR|PR)[+-]|\b(?:ER|HR|PR)\s+(?:positive|negative)', type_val)`
   - 或更简单: 只要 Type_of_Cancer 非空且无 HER2 信息就触发

### 重要 (P1)

4. **Row 4 Referral 泄漏**: 来自 LLM 输出本身，需在 plan_extraction prompt 中加约束

5. **response_assessment "currently responding"**: 需进一步限制 prompt，或在 POST 处理中检测并修正

---

## 六、统计对比

| 指标 | v14a | v15 (34行) | 变化 |
|------|------|-----------|------|
| P0 问题 | 3 | 3 | 0（修了 1 个 Referral，引入 2 个 HER2 错误）|
| JSON wrapper 泄漏 | 17/11行 | 0 | -100% ✅ |
| ngs 假阳性 | 4 行 | 0/2验证 | -100% ✅ |
| HER2 缺失 | 11 行 | ~5/7验证行仍缺（4待验证） | -29%（仅 2 行改善） |
| "currently responding" 误用 | ≥3 行 | 1/3验证行仍有 | -33% |

---

*（剩余 27 行待 pipeline 完成后补充）*
