# v22d Full Review: plan_extraction 用全文实验

Run: `results/v21_full_20260322_203659/`
Date: 2026-03-23
改动: plan_extraction 输入从 A/P 改为 note_text（全文）
审查方式: **人工逐字审查**

---

## 稳定性

| 指标 | v22 (A/P) | v22d (全文) |
|------|-----------|------------|
| 与 v22 完全一致 | — | **3/61** (5%) |
| 关键字段变化 | — | **20 处** |
| Plan 字段变化 | — | **146 处** |

**极不稳定。** 只有 3 行完全没变，几乎每行的 plan 字段都被影响。

---

## 改善（5 行）

| ROW | 字段 | v22 → v22d | 真改善? |
|-----|------|-----------|---------|
| 1 | imaging_plan | "No imaging planned" → "MRI of brain and bone scan" | ✅ 真改善 |
| 1 | lab_plan | "No labs planned" → "CBC, CMP, CA15-3, CEA, aPTT, PT" | ✅ 真改善 |
| 34 | lab_plan | "No labs planned" → "check labs" | ✅ 模糊但有值 |
| 43 | lab_plan | "No labs planned" → "blood draw prior to cycle" | ✅ |
| 66 | imaging_plan | "No imaging planned" → "Clinical correlation for focal skin..." | ⚠️ 这是放射科建议，不是肿瘤科 imaging plan |

---

## 退化（8 行）— 全部是 A/P 中有的信息被全文输入淹没

| ROW | 字段 | v22 原有值 | v22d 变成 | 原文确认 |
|-----|------|----------|----------|---------|
| 20 | lab_plan | "tumor markers, monthly blood work" | "No labs planned" | A/P 有 "monthly blood work" ✅ |
| 30 | imaging_plan | "TTE needed before treatment" | "No imaging planned" | A/P 有 "TTE" ✅ |
| 37 | medication_plan | "dd AC followed by Taxol" | dexlansoprazole（无关药物）| A/P 有 AC/Taxol ✅ |
| 46 | lab_plan | "iron panel in 3-4 months" | "No labs planned" | A/P 有 "iron panel" ✅ |
| 49 | medication_plan | "tamoxifen, thrombophilia" | "None" | A/P 有 tamoxifen ✅ |
| 61 | medication_plan | "Tamoxifen or OFS+AI" | "None" | A/P 有 endocrine therapy ✅ |
| 65 | lab_plan | "labs, follow-up pending" | "No labs planned" | A/P 有 "labs" ✅ |
| 92 | lab_plan | "liver functions, tumor markers" | "No labs planned" | A/P 有 "liver functions" ✅ |

**根因**: 全文包含大量历史影像/检验结果（过去的 CT、MRI、PET、CBC 结果等），LLM 被这些历史数据淹没，反而看不清 A/P 中的简短 plan 语句。

---

## 关键字段退化（2 个 P1 级别）

| ROW | 字段 | v22 | v22d | 评估 |
|-----|------|-----|------|------|
| 42 | Type | "ER+/PR+/HER2- IDC" | "PR+ IDC, HER2: not tested" | **P1 退化** — 丢了 ER+，HER2 变 "not tested" |
| 54 | Type | "ER+/PR-/HER2- IDC" | "ER+/PR+/HER2- IDC" | **实际 v22d 更正确** — 活检 PR+10%/20%，v22 的 PR- 可能是脱敏误判 |

---

## 根因分析

**为什么全文输入会让 plan 字段退化？**

1. **Context 稀释**：A/P 段通常 500-2000 tokens。全文可达 5000-16000 tokens。plan 相关信息在全文中占比极小，容易被稀释。

2. **历史数据干扰**：笔记包含大量历史影像结果（"11/30/18 CT CAP: ..."）和检验结果（"Hemoglobin 12.9"）。LLM 容易把这些历史结果当作 plan 提取，或者被这些细节分散注意力。

3. **plan_extraction prompt 设计**：prompt 说 "Extract from the note"，但没有明确指导 "focus on the Assessment & Plan section for future plans"。当输入是全文时，LLM 不知道该重点看哪个段落。

4. **KV Cache 影响**：全文构建的 base cache 完全不同于 A/P 的 cache，导致所有 plan 字段的 greedy decoding 路径改变。

---

## 速度影响

| 版本 | 3 行测试 | 61 行全量 | 每行平均 |
|------|---------|----------|---------|
| v22 (A/P) | ~10 min | ~200 min | ~3.3 min |
| v22d (全文) | ~17 min | ~268 min | ~4.4 min |

**慢了 33%**。

---

## 结论

**全文输入方案失败。**

- 改善 5 行但退化 8 行 = **净负收益**
- 146 处噪声 = **极不稳定**
- 2 个关键字段退化
- 慢了 33%

**建议**：回退到 v22（A/P 输入），Row 0 的 imaging/lab plan 用 POST hook（方案 A）修。

POST hook 方案的优势：
1. 不改变其他行的输出（零蝴蝶效应）
2. 只在 plan 为空时触发（不会丢失已有信息）
3. 搜索范围可精确控制（只搜 Orders 段和 HPI，不搜历史结果段）
