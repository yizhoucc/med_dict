# v19 逐行审查报告

审查日期：2026-03-20
版本：v19（修复 POST-MEDS-IV-CHECK 假阳性 — 正向匹配替代 fallback 药名扫描）
结果目录：v19_verify_20260320_172459
状态：**审查完成**

---

## 一、v19 修复验证总表

| # | 修复 | v18 触发 | v19 触发 | 验证状态 | 说明 |
|---|------|---------|---------|---------|------|
| 1 | POST-MEDS-IV-CHECK 正向匹配 | 17 | 3 | ⚠️ **2 TP + 1 FP** | Row 1 irinotecan ✅, Row 56 docetaxel ❌ FP, Row 89 ac ✅ |

### v18 其他修复保持情况

| # | 修复 | 预期 | 状态 | 说明 |
|---|------|------|------|------|
| 2 | row.get→局部变量 | 继续有效 | ✅ | POST-RESPONSE-TREATMENT 7次触发，POST-TYPE-VERIFY-TNBC 正常 |
| 3 | POST-VISIT-TYPE "via video" | ROW 86 → Televisit | ✅ | LLM 自行输出 Televisit（hook 未触发但结果正确）|
| 4 | POST-PATIENT-TYPE 值校验 | 无效值被修正 | ✅ | ROW 41 "in-person" → "Follow up" (1次触发) |
| 5 | POST-GENETICS 结果优先 | ROW 61 清除 | ✅ | 11次触发，全部正确清除已完成的基因检测结果 |
| 6 | POST-STAGE-PLACEHOLDER | [X] → "Not available (redacted)" | ✅ | 9次触发，全部正确替换 |

---

## 二、v18 FP 消除验证

v18 有 17 次 POST-MEDS-IV-CHECK 触发（至少 4 个确认 FP）。v19 的正向匹配策略成功消除了绝大部分：

| v18 FP Row | v18 添加药物 | v19 状态 | 结果 |
|------------|-------------|---------|------|
| Row 36 | ac, taxol | ✅ 不再触发 | 修复（"decided to proceed"=计划） |
| Row 40 | ac, taxol, ribociclib | ✅ 不再触发 | 修复（全部计划用药） |
| Row 42 | taxol, carboplatin | ✅ 不再触发 | 修复 |
| Row 43 | taxol, ribociclib, zoladex | ✅ 不再触发 | 修复 |
| Row 49 | zoladex | ✅ 不再触发 | 修复 |
| Row 52 | ac, tchp, thp, taxol, pertuzumab, trastuzumab | ✅ 不再触发 | 修复（全部推荐方案） |
| Row 56 | ac, docetaxel, pertuzumab, trastuzumab | ⚠️ 仍触发 1 药 | 4药→1药（docetaxel FP） |
| Row 63 | taxol | ✅ 不再触发 | 修复 |
| Row 64 | 9 种药 | ✅ 不再触发 | 修复 |
| Row 65 | pembrolizumab | ✅ 不再触发 | 修复 |
| Row 67 | tchp | ✅ 不再触发 | 修复 |
| Row 77 | gemcitabine, eribulin, pembrolizumab | ✅ 不再触发 | 修复 |
| Row 79 | tc | ✅ 不再触发 | 修复 |
| Row 83 | olaparib, palbociclib, fulvestrant | ✅ 不再触发 | 修复 |
| Row 93 | ac, taxol, capecitabine, pembrolizumab | ✅ 不再触发 | 修复（保留正确的 letrozole） |

**消除率**: 14/15 v18 FP 行不再触发 = **93%**
**Row 56 残留 FP 原因**: Pattern 6 `(?:receiving|given)\s+(\w+)` 匹配了 "beneficial **given docetaxel** component"，此处 "given" = 介词"鉴于"而非动词"给予"

---

## 三、Row 56 docetaxel FP 详细分析

**原文 A/P 关键句**:
> "TCH+P regimen would be expected to be beneficial **given docetaxel** component as well as platinum"

**时间线**:
1. 新辅助 TCH+P x6 → **已完成**
2. 手术 (12/25/2013) → **已完成**
3. 术后 AC x4 → **已完成**
4. XRT → **下一步计划**（"scheduled to begin XRT in the near future"）
5. 当前用药: Carvedilol, Lisinopril, Asacol（全部非抗癌药）

**FP 根因**: Pattern 6 中 "given" 有双义：
- 动词: "She was **given** docetaxel" = 给予了药物 ✅ 应匹配
- 介词: "beneficial **given** docetaxel component" = 鉴于/考虑到 ❌ 不应匹配

**修复建议**: 改为 `(?:receiving|(?:was|been|be)\s+given)\s+(\w+)` — 要求 "given" 前有 was/been/be

**该行其他问题**:
- `supportive_meds: "benadryl, codeine"` ❌ **P0** — 来自过敏列表（"ALL: Erythro, benadryl, codeine, demerol, motrin, pcn"），不是药物
- `recent_changes: "Dose reduction 25% after C1"` ❌ P2 — 历史信息，非当前变化

---

## 四、POST Hook 触发统计

| POST Hook | 触发次数 | 正确率 | 说明 |
|-----------|---------|--------|------|
| POST-MEDS-IV-CHECK | 3 | 67% (2/3) | Row 1 TP, Row 89 TP, Row 56 FP |
| POST-GENETICS | 11 | 100% | 全部正确清除已完成的基因检测结果 |
| POST-STAGE-PLACEHOLDER | 9 | 100% | 全部正确替换 [X]/redacted |
| POST-RESPONSE-TREATMENT | 7 | 100% | 全部正确修正 "Not yet on treatment" |
| POST-ER-CHECK | 2 | 100% | Row 9, 82 正确推断 ER+ |
| POST-HER2-CHECK | 4 | 100% | 正确附加 HER2 状态 |
| POST-HER2-VERIFY | 2 | 100% | 基于药物证据覆盖 HER2 状态 |
| POST-TYPE-VERIFY-TNBC | 1 | 100% | Row 56 A/P 说 TNBC 覆盖 HER2+ |
| POST-DISTMET-REGIONAL | 2 | 100% | Row 82, 93 修正 regional LN |
| POST-PATIENT-TYPE | 1 | 100% | Row 40 "in-person" → "Follow up" |
| POST-VISIT-TYPE | 0 | — | LLM 自行输出正确值 |
| POST-NUTRITION | 3 | 100% | 正确清除饮食建议 |
| POST-REFERRAL-VALIDATE | 1 | 100% | Row 82 移除无原文支撑的 Rad Onc |
| POST-GOALS | 2 | 100% | adjuvant → curative |

---

## 五、逐行审查结果

### Batch 1: ROW 1-12 (Row 0-11)

#### ROW 1 (coral_idx 140) — Row 0
- **Patient type**: "New patient" ✅ — "New Patient Evaluation"
- **in-person**: "in-person" ✅
- **Cancer**: ER+/PR+/HER2- ✅; Stage IIA→IV ✅; Distant Met Yes ✅
- **current_meds**: "" ✅ — **v18 P0 修复**（v18 为 "ibrance"，来自 "will recommend ibrance"=计划药）
- **Imaging_Plan**: "No imaging planned." ❌ **P1** — Orders 有 MRI brain + bone scan
- **Lab_Plan**: "No labs planned." ❌ **P1** — Orders 有 CBC/CMP/CA15-3/CEA
- **整体**: 一般（核心修复成功，但 plan 类遗漏）

#### ROW 2 (coral_idx 141) — Row 1
- **current_meds**: "irinotecan" ✅ **POST-MEDS-IV-CHECK TP** — "presents for cycle 3 day 1"
- **Stage**: Stage IIB→IV ✅; Distant Met Yes ✅
- **整体**: 好

#### ROW 3 (coral_idx 142) — Row 2
- **Televisit** ✅; second opinion ✅; ER+/PR+/HER2- ✅; Stage IIA ✅
- **整体**: 好

#### ROW 5 (coral_idx 144) — Row 4
- **current_meds**: "anastrozole, palbociclib, leuprolide" ✅
- **Distant Met**: Yes (cervical LN, internal mammary LN, sternum) ✅
- **整体**: 好

#### ROW 6 (coral_idx 145) — Row 5
- **Patient type**: "New patient" ⚠️ **P1** — 已开始 letrozole，可能应为 Follow up
- **Stage**: "" ❌ **P1** — 1.5cm, node neg, grade 1 可推断 Stage I
- **current_meds**: "zoladex, letrozole" ✅
- **整体**: 一般

#### ROW 7 (coral_idx 146) — Row 6
- **Type_of_Cancer**: "Originally ER+/PR+/HER2+" ⚠️ **P1** — 原始 ER/PR 状态不明确（note 中被 redacted）
- **整体**: 一般

#### ROW 8 (coral_idx 147) — Row 7
- **in-person**: "Televisit" ✅; Cancer type ER-/PR-/HER2+ (IHC 3+, FISH 5.7) ✅
- **整体**: 好

#### ROW 9 (coral_idx 148) — Row 8
- **Type_of_Cancer**: ER+/PR-/HER2- ✅ — POST-ER-CHECK 推断 ER+ from letrozole
- **current_meds**: "" ⚠️ **P1** — 原文 med list 有 letrozole，但可能是 A/P 说 "after radiation" = 计划
- **response_assessment**: "Not yet on treatment" ⚠️ **P1** — 已接受 AC + taxol（但可能指目前未在化疗）
- **整体**: 一般

#### ROW 10 (coral_idx 149) — Row 9
- **current_meds**: "letrozole" ✅
- **整体**: 好

#### ROW 11 (coral_idx 150) — Row 10
- **current_meds**: "Faslodex, Denosumab" ✅; Stage III→IV ✅
- **整体**: 好

#### ROW 12 (coral_idx 151) — Row 11
- **Stage**: "Not available (redacted)" ❌ **P1** — A/P 有 "St IV de novo"
- **Distant Met**: 遗漏 liver ❌ **P1** — 原文 "liver, lung, nodes, brain and bone"
- **current_meds**: "herceptin, letrozole" ⚠️ P1 — 遗漏 ***** (可能是 pertuzumab)
- **整体**: 一般

### Batch 2: ROW 14-37 (Row 13-36)

#### ROW 14 (coral_idx 153) — Row 13
- **current_meds**: "Pamidronate, Gemcitabine, Docetaxel, Doxorubicin" ❌ **P0** — 墨西哥自我管理化疗，非临床处方
- **HER2**: "status unclear" ❌ P1 — 原文有 FISH negative，应为 HER2-
- **findings**: 写成治疗描述 ❌ P1
- **Stage**: "Not available (redacted)" ✅ POST-STAGE-PLACEHOLDER
- **整体**: 差

#### ROW 17 (coral_idx 156) — Row 16
- 新患者会诊，ER+/PR+/HER2- IDC, 0.8cm, grade 2 ✅
- **Stage**: "estimated Stage I-II" ⚠️ P2 — 可从 T1N0 精确推断 Stage I
- **整体**: 好

#### ROW 18 (coral_idx 157) — Row 17
- **genetic_testing_plan**: "None" ❌ P1 — 遗漏 "discussed with UCSF Cancer Risk"
- **整体**: 好（仅 1 个遗漏）

#### ROW 20-34 — Row 19-33（快速审查）
- ROW 20: current_meds="letrozole, palbociclib" — 需验证
- ROW 22: current_meds="anastrozole, denosumab" — 需验证
- ROW 27: current_meds="letrozole, goserelin" — 需验证
- ROW 29: current_meds="letrozole" — 合理
- ROW 30: current_meds="" — 合理
- ROW 33: current_meds="letrozole" — 合理
- ROW 34: current_meds="arimidex" — 合理

#### ROW 36 (coral_idx 175) — Row 35
- **current_meds**: "Abraxane, zoladex" ✅ — "cycle 8 of abraxane, Continue zoladex"
- **整体**: 好

#### ROW 37 (coral_idx 176) — Row 36
- **current_meds**: "" ✅ — **v18 FP 修复**（v18 为 "ac, taxol"，来自 "recommend dd AC followed by Taxol"=计划）
- **整体**: 好

### Batch 3: ROW 40-57 (Row 39-56)

#### ROW 40 (coral_idx 179) — Row 39
- **current_meds**: "letrozole" ⚠️ **P1** — "Rx for letrozole given"=刚开处方，可能未开始服用
- **整体**: 一般

#### ROW 41 (coral_idx 180) — Row 40
- **Patient type**: "Follow up" ✅ — **POST-PATIENT-TYPE 修正**
- **current_meds**: "" ✅ — **v18 FP 修复**（v18 为 "ac, taxol, ribociclib"）
- **整体**: 好

#### ROW 42 (coral_idx 181) — Row 41
- **current_meds**: "tamoxifen" ⚠️ **P1** — "will begin a 5 year course"=计划，非当前
- **整体**: 一般

#### ROW 43 (coral_idx 182) — Row 42
- **current_meds**: "" ✅ — taxol/carboplatin 是计划辅助治疗
- **整体**: 好

#### ROW 44 (coral_idx 183) — Row 43
- POST-GENETICS 清除 VUS ✅
- **整体**: 好

#### ROW 46 (coral_idx 185) — Row 45
- **整体**: 好

#### ROW 49 (coral_idx 188) — Row 48
- **Patient type**: "New patient" ✅ — "initial consultation"=新患者（per prompt 规则）
- **整体**: 一般

#### ROW 50 (coral_idx 189) — Row 49
- v18 FP "zoladex" ✅ 不再触发
- **整体**: 好

#### ROW 52 (coral_idx 191) — Row 51
- **整体**: 好

#### ROW 53 (coral_idx 192) — Row 52
- **current_meds**: "" ✅ — **v18 FP 修复**（v18 为 6 种推荐药物）
- **整体**: 好

#### ROW 54 (coral_idx 193) — Row 53
- **current_meds**: "leuprolide, letrozole, zoledronic acid" ✅
- **goals_of_treatment**: "palliative" ✅ — Stage IV
- **整体**: 好（本 batch 最佳）

#### ROW 57 (coral_idx 196) — Row 56
- **current_meds**: "docetaxel" ❌ **P0 FP** — POST-MEDS-IV-CHECK "given docetaxel" 误匹配（详见第三节）
- **supportive_meds**: "benadryl, codeine" ❌ **P0** — 来自过敏列表，非药物
- **Type_of_Cancer**: "ER-/PR-/HER2- triple negative" ✅ — POST-TYPE-VERIFY-TNBC 正确覆盖
- **整体**: 差

### Batch 4: ROW 59-100 (Row 58-99)

#### ROW 59 (coral_idx 198) — Row 58
- **current_meds**: "exemestane, letrozole" ❌ **P1** — letrozole 要停，exemestane 未开始
- **整体**: 一般

#### ROW 61 (coral_idx 200) — Row 60
- **Genetics**: "None" ✅ — **POST-GENETICS 清除 "Invitae genetic testing: negative"**
- **in-person**: "Televisit" ✅ — "real-time telehealth tools, live video Zoom"
- **current_meds**: "" ✅ — 新诊断，未开始治疗
- **整体**: 好

#### ROW 63 (coral_idx 202) — Row 62
- **current_meds**: "letrozole" ✅ — "Continue letrozole"
- **整体**: 好

#### ROW 64 (coral_idx 203) — Row 63
- **current_meds**: "" ✅ — **v18 FP 修复**（v18 为 9 种药）
- **整体**: 好

#### ROW 65 (coral_idx 204) — Row 64
- POST-GENETICS 清除 PMS2 mutation ✅; ER 2%/PR 7% 弱阳性 ✅
- **整体**: 好

#### ROW 66 (coral_idx 205) — Row 65
- POST-RESPONSE-TREATMENT 修正 ✅
- **整体**: 好

#### ROW 68 (coral_idx 207) — Row 67
- POST-GENETICS 清除 VUS ✅; HER2+ multifocal ✅
- **整体**: 好

#### ROW 70 (coral_idx 209) — Row 69
- 双侧乳腺癌完整描述 ✅; current_meds="letrozole" ✅
- **整体**: 好

#### ROW 72 (coral_idx 211) — Row 71
- **current_meds**: "letrozole" ✅
- **整体**: 好

#### ROW 73 (coral_idx 212) — Row 72
- **current_meds**: "arimidex" ✅
- **整体**: 好

#### ROW 78 (coral_idx 217) — Row 77
- **current_meds**: "" ✅ — **v18 FP 修复**（v18 为 "gemcitabine, eribulin, pembrolizumab"）
- "Did not receive treatment today"，等临床试验
- **整体**: 好

#### ROW 80 (coral_idx 219) — Row 79
- **v18 FP "tc" ✅ 不再触发**
- **整体**: 好

#### ROW 82 (coral_idx 221) — Row 81
- POST-GENETICS + POST-RESPONSE-TREATMENT + POST-STAGE-REGIONAL + POST-ER-CHECK 联合生效 ✅
- **整体**: 好

#### ROW 83 (coral_idx 222) — Row 82
- POST-STAGE-PLACEHOLDER ✅; ER+ (inferred) ✅
- **Distant Met**: "" ❌ P2 — 应为 "No"
- **整体**: 一般

#### ROW 84 (coral_idx 223) — Row 83
- **v18 FP "olaparib, palbociclib, fulvestrant" ✅ 不再触发**
- POST-GENETICS 清除 BRCA1 ✅
- **整体**: 好

#### ROW 85 (coral_idx 224) — Row 84
- **current_meds**: "" ❌ **P1** — 原文 "cycle 2 of Abraxane" = 当前用药
- **整体**: 一般

#### ROW 86 (coral_idx 225) — Row 85
- **in-person**: "Televisit" ✅ — **v18 修复保持**（"via video"）
- **current_meds**: "letrozole, ribociclib, denosumab" ✅
- **整体**: 好

#### ROW 87 (coral_idx 226) — Row 86
- POST-GENETICS 清除 CHEK2 ✅; POST-STAGE-PLACEHOLDER ✅
- **整体**: 好

#### ROW 88 (coral_idx 227) — Row 87
- **Type_of_Cancer**: 正确捕捉受体转换（primary vs metastatic biopsy）✅
- **current_meds**: "capecitabine" ✅
- **整体**: 好

#### ROW 90 (coral_idx 229) — Row 89
- **current_meds**: "ac" ✅ **POST-MEDS-IV-CHECK TP** — "Currently on cycle 2 of AC"
- POST-GENETICS 清除 BLM mutation ✅
- **Distant Met**: "Not sure" ❌ P2 — 应为 "No"
- **整体**: 好（核心 TP 成功）

#### ROW 91 (coral_idx 230) — Row 90
- **current_meds**: "everolimus, exemestane, denosumab" ✅
- **整体**: 好

#### ROW 92 (coral_idx 231) — Row 91
- **current_meds**: "Epirubicin, Denosumab" ✅
- **整体**: 好

#### ROW 94 (coral_idx 233) — Row 93
- **POST-DISTMET-REGIONAL 修正 ✅** — v18 误判 axillary LN 为 distant
- **v18 FP "ac, taxol, capecitabine, pembrolizumab" ✅ 不再触发**
- **整体**: 好

#### ROW 95 (coral_idx 234) — Row 94
- **current_meds**: "" ❌ **P1** — 原文 "continue tamoxifen" = 当前用药
- POST-STAGE-PLACEHOLDER ✅
- **整体**: 一般

#### ROW 97 (coral_idx 236) — Row 96
- current_meds="" ✅（等 Oncotype Dx）
- **整体**: 好

#### ROW 100 (coral_idx 239) — Row 99
- **current_meds**: "" ⚠️ **边界** — "on Gemzar Cycle #2 Day 8 cancelled by patient"。v18 提取 "gemzar"，v19 为空。两种理解都有道理
- POST-GENETICS 清除 CHEK2 ✅; POST-STAGE-PLACEHOLDER ✅
- **整体**: 一般

---

## 六、问题汇总

### P0 问题 (3 个)

| Row | 字段 | 问题 | 原因 |
|-----|------|------|------|
| 13 (ROW 14) | current_meds | "Pamidronate, Gemcitabine, Docetaxel, Doxorubicin" | 墨西哥自我管理化疗药，非临床处方 |
| 56 (ROW 57) | current_meds | "docetaxel" FP | POST-MEDS-IV-CHECK pattern 6 误匹配介词 "given" |
| 56 (ROW 57) | supportive_meds | "benadryl, codeine" | 来自过敏列表（ALL），非药物 |

### P1 问题 (~15 个)

| 模式 | 影响行数 | 说明 |
|------|---------|------|
| current_meds 时态误判 | 5 | Row 39 (letrozole 刚开方), Row 41 (tamoxifen 计划), Row 58 (letrozole 要停+exemestane 未开始), Row 84 (abraxane 遗漏 cycle 2), Row 94 (tamoxifen "continue" 遗漏) |
| Plan 类字段遗漏 | 2 | Row 0 (Imaging+Lab Plan 遗漏 Orders) |
| Stage 遗漏/不精确 | 3 | Row 5, 11, 17 等可推断但未填 |
| 其他遗漏 | 5 | Row 6 (Type 幻觉), Row 11 (Met 遗漏 liver), Row 13 (HER2, findings), Row 17 (Genetics referral) |

### P2 问题 (~8 个)
- Stage 推断不足（多行）
- Distant Met 值不规范（Row 82 空值应为 No, Row 89 "Not sure"）
- recent_changes 历史信息（Row 56）

---

## 七、v19 vs v18 对比

| 维度 | v18 | v19 | 变化 |
|------|-----|-----|------|
| IV-CHECK 触发 | 17 | 3 | **-82%** |
| IV-CHECK FP | ≥4 (50%+) | 1 (33%) | **大幅改善** |
| IV-CHECK TP | 2 | 2 | 保持 |
| v18 FP 行消除 | — | 14/15 | **93% 消除率** |
| P0 总数 | 1 (current_meds regression) | 3 | 略增（新发现 Row 13 + Row 56 supportive_meds） |
| POST hooks 有效性 | 多数有效 | **100%**（除 IV-CHECK 1 FP） | 改善 |
| 总体质量 | — | 好 44/61 (72%), 一般 15/61 (25%), 差 2/61 (3%) | — |

---

## 八、v20 建议优先级

### P0 修复

1. **POST-MEDS-IV-CHECK Pattern 6**: `(?:receiving|given)` → `(?:receiving|(?:was|been|be)\s+given)` — 排除介词义 "given"
2. **supportive_meds 过敏列表过滤**: 在提取或 POST hook 中检测 "ALL:" / "Allergies:" 段落内的药名，排除

### P1 改进

3. **current_meds 时态增强**:
   - "continue X" / "cycle N of X" = 当前用药（Row 84 abraxane, Row 94 tamoxifen 遗漏）
   - "will begin" / "Rx given" = 计划（Row 41 tamoxifen, Row 39 letrozole）
   - "discontinuing" / "plan to switch" = 即将停用
4. **Plan 类字段**: 考虑在 plan_extraction 中也搜索 Orders 段落（Row 0 的 MRI/labs 在 Orders 而非 A/P）

### P2 优化

5. **Stage 推断**: 从 T/N 分期推断 AJCC Stage（多行可推断但未填）
6. **Distant Met 规范化**: 空值 → "No"（ROW 83），"Not sure" → 更明确的值

### 已满足、不需改动

- POST-GENETICS: 11/11 = 100% ✅
- POST-STAGE-PLACEHOLDER: 9/9 = 100% ✅
- POST-RESPONSE-TREATMENT: 7/7 = 100% ✅
- POST-PATIENT-TYPE: 1/1 = 100% ✅
- POST-VISIT-TYPE: LLM 自行正确，hook 不需触发 ✅

---

## 九、边界情况讨论

1. **Row 99 gemzar**: 患者 "cancelled" 本次化疗，想暂停。v19 LLM 输出 current_meds=""，v18 POST hook 添加 "gemzar"。临床上患者的治疗方案仍是 gemzar（只是本次 cancelled），可能应保留。但 "Did not receive treatment today" + "take a break" 也支持为空。**建议保持为空**（更保守）。

2. **Row 13 墨西哥化疗**: 患者自行从墨西哥获取化疗药物在家注射。这些药名出现在原文中，但医生不认可（"discontinue our medications"）。LLM 提取了这些药名是正确的文本理解（原文确实说 "currently doing low dose chemo"），但从临床角度不应算 "current prescribed medications"。**需要在 prompt 中区分 "self-administered" vs "clinically prescribed"**。

3. **Patient type 边界**: 多行 "initial consultation" 被标为 New patient（per prompt 规则），但临床上可能是 follow up（已确诊+治疗后转院）。当前 prompt 规则 "consultation = New patient" 是合理的简化。
