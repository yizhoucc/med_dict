# V26 Full Comparison Review: notool vs tool

> Date: 2026-04-10
> notool: v26_full_notool_20260408_080004 (100 samples, tool_calling=false)
> tool: v26_full_tool_20260408_140605 (100 samples, tool_calling=true)
> Reviewer: Claude (系统性对比 + 逐字审查关键差异)

---

## 整体结果

| 指标 | notool | tool | 差异 |
|------|--------|------|------|
| Letter "blood tests" 幻觉 | 0 | 0 | = |
| Letter "emotional" 编造 | 0 | 0 | = |
| 实质性改善 (tool 更好) | — | 13 个 sample | +13 |
| 实质性退化 (notool 更好) | — | 6 个 sample | -6 |
| 中性差异 | — | 5 个 sample | 0 |
| 相同/措辞差异 | — | 76 个 sample | = |

**净收益: +7 个 sample 改善，但有 2 个 P0 级别的退化（垃圾数据）**

---

## 改善 (13 个 sample — tool 更好)

### 高价值改善
| ROW | 字段 | notool | tool | 改善级别 |
|-----|------|--------|------|----------|
| **12** | **Advance care** | "Not discussed" | "DNR/DNI + POLST details" | **P1→✅** |
| **1** | **lab_plan** | "MRI + bone scan + labs" (混入 imaging) | "labs to complete her work up" (纯 lab) | **P1→✅** |
| **45** | **response** | "no specific evidence" | "progression on neoadjuvant taxol + mets" | **P1→✅** |
| **90** | **response** | "cycle 2 AC" (当前状态) | "residual 2.2cm after neoadjuvant" (病理响应) | **P1→✅** |

### 中等改善
| ROW | 字段 | 改善内容 |
|-----|------|----------|
| 10 | response | 从 "no specific evidence" → "no evidence of recurrence" |
| 17 | genetic_testing | 增加了家族史上下文 |
| 18 | genetic_testing | 从过去式 → "UCSF Cancer Risk will reach out today" |
| 22 | response | 增加了体检详情 |
| 33 | response | 从简略 → 详细体检发现 |
| 44 | response | 明确说了 "no pCR" |
| 54 | response | 增加了体检详情 |
| 80 | genetic_testing | 从 "None planned" → "whole genome completed" |
| 92 | response | 从 "marker pending" → "stable + liver details" |

---

## 退化 (6 个 sample — notool 更好) ⚠️

### P0 级别退化（垃圾数据）
| ROW | 字段 | notool | tool |
|-----|------|--------|------|
| **73** | lab_plan | "check labs" | **"plant reconstruction. Right breast - 1cm hard nodule 7OClock"** — 物理检查段落混入 lab_plan |
| **65** | lab_plan | "labs follow-up pending" | **"Plan: -- order TTE -- Port placement -- Chemotherapy teaching session"** — 非 lab 内容混入 |

**根因**: SEARCH_NOTE("labs") 返回了 note 中包含 "labs" 字样但不是 lab 相关的段落（如 "labs reviewed" 附近的其他计划内容），模型把工具返回的无关段落当成了 lab plan 的内容。

### P1 级别退化（信息丢失）
| ROW | 字段 | notool | tool |
|-----|------|--------|------|
| 75 | Stage | "Stage II-III" | "Not available (redacted)" |
| 94 | Stage | "Stage IIA (pT1b, N1, G2, ER+, PR+, HER2-)" | "Not available (redacted)" |
| 95 | Stage | "Stage IV (metastatic)" | "Not mentioned in note" |
| 70 | response | 详细 MRI 发现 | "On treatment; not available" |

**根因**: prompt 末尾追加的 TOOLS 说明文本改变了模型的注意力分配，导致：
- Stage 提取对 redacted 内容更保守（原来能推断的现在不推断了）
- response_assessment 从详细描述退化为笼统判断

---

## 诊断分析

### Tool calling 有效的场景
模型在这些场景下正确使用了 SEARCH_NOTE 并获得了改善：
1. **A/P 中信息不全**（Advance care DNR/DNI 在 problem list）→ 搜索全文补全
2. **A/P 混写不同类型信息**（lab_plan 中 imaging+labs）→ 搜索全文分离
3. **response_assessment 需要更多上下文**（病理结果在 oncologic history）→ 搜索获取详情

### Tool calling 有害的场景
1. **SEARCH_NOTE 返回错误段落**（ROW 73, 65）→ 模型把无关文本当 lab plan
2. **TOOLS 说明文本干扰正常提取**（ROW 75, 94, 95）→ Stage 提取退化

### 根本问题
Tool calling 是**双刃剑**：
- 当模型正确使用工具时，提供了 A/P 之外的关键信息（+13 改善）
- 当工具返回错误结果或 prompt 过长时，引入了新问题（-6 退化）
- **净效果为正（+7），但 P0 退化不可接受**

---

## 修复建议

### 必须修复（P0 退化）
1. `execute_tool_calls()` 中 SEARCH_NOTE 返回结果需要过滤：
   - 检查返回段落是否真的与请求相关
   - 限制返回长度，避免注入大段无关文本
   - 对 lab_plan 的 SEARCH_NOTE，只返回包含实际 lab 名称（CBC, CMP 等）的段落

### 应该修复（P1 退化）
2. TOOLS 说明文本不应影响 Phase 1/2 的 Stage 提取：
   - 方案 A: Phase 1/2 不加 TOOLS 说明（只对 plan extraction 加）
   - 方案 B: TOOLS 说明放在 prompt 最后一行之后（减少注意力干扰）

---

## 最终推荐

**目前不建议对全量跑开启 tool_calling=true**。虽然净改善为正，但 2 个 P0 退化和 4 个 P1 退化意味着 tool calling 还需要优化：

1. 先修复 SEARCH_NOTE 返回结果过滤（消除 P0）
2. 只对 plan extraction 开启 tool calling（避免 Phase 1/2 的 Stage 退化）
3. 重新跑验证确认无退化后再考虑全量部署

**v26_full_notool 是当前最佳版本**（P1≈3-4），建议作为正式 v26 baseline。

---

## notool 版本残余 P1（手工抽检发现）

| ROW | 字段 | 问题 | 能否用 tool 修复 |
|-----|------|------|-----------------|
| 1 | lab_plan | imaging 混入（A/P 措辞混写） | ✅ tool 版本已修复 |
| 10 | response | "does not provide evidence" 但有术后病理 | 否（prompt 问题） |
| 12 | Advance care | DNR/DNI 在 problem list 未捕获 | ✅ tool 版本已修复 |
| 88 | response | "Not mentioned" post-neoadjuvant progression | 否（prompt 问题） |

**Tool calling 能修复 2/4 个 notool 残余 P1**（ROW 1 lab_plan + ROW 12 Advance care），但同时引入了 2 P0 + 4 P1 退化。需要先消除退化才能开启 tool calling。

---

## 版本对比总结

| 版本 | P0 | P1 | 说明 |
|------|----|----|------|
| v24 (修复前) | 0 | 21 | baseline |
| v25 (prompt 修复) | 0 | 4 | -81% |
| v26 notool (prompt 修复+新 POST hooks) | 0 | 3-4 | 最佳稳定版 |
| v26 tool (全量 tool calling) | **2** | ~3 | 有改善但引入 P0 退化 |

**结论**: prompt 工程是主力（P1: 21→4），tool calling 有价值但需要更多打磨才能安全部署。
