# v19 计划 — 修复 POST-MEDS-IV-CHECK 假阳性

## 背景

v18 审查发现 POST-MEDS-IV-CHECK 的药名直接扫描（v18 新增的 fallback）产生严重假阳性：
- 17 次触发中，已确认 4 FP + 2 TP，11 待查
- 假阳性根因：扫描 A/P 中所有药名，只排除过去时态，不排除计划/推荐/讨论上下文
- 例：Row 52 Meds="none"，但 POST hook 添加了 6 种推荐方案药物

## 修复策略

**改 fallback 为正向匹配**（排除法不可靠，改为"只在明确当前用药上下文中匹配"）：

### 原策略（v18，问题代码）
```
for drug in KNOWN_CHEMO_IV:
    在 A/P 全文中搜索 drug
    如果前 40 字符没有 PAST_CHEMO 词 → 添加为当前用药
```
问题：没有排除 "recommend/discussed/will benefit/plan/option" 等计划性上下文

### 新策略（v19）
```
ACTIVE_CHEMO_PATTERNS:
  1. "continue/continuing [with] [cycle N of] DRUG" — 当前正在用
  2. "currently on/on DRUG" — 注意要排除 "started on X in 2013"（过去）
  3. "cycle N [day N] of DRUG" — 正在某个周期
  4. "DRUG day N" — 正在某天
  5. "started DRUG on DATE" — 如果日期在近期（原始 pattern 保留）

fallback 药名直接扫描：删除。用正向 pattern 替代。
```

核心改动：**删除 v18 的 `if not found_chemo:` fallback 块**，改为增强 IV_CHEMO_PATTERNS 的覆盖面。

### 具体 Pattern 改进

v18 的 IV_CHEMO_PATTERNS 有 3 个 pattern，但太宽泛。v19 改为：

```python
IV_CHEMO_PATTERNS = [
    # Pattern 1: "continue/continuing [with] [cycle N of] DRUG"
    r'(?:continue|continuing)\s+(?:with\s+)?(?:cycle\s+\d+\s+(?:of\s+)?)?(\w+(?:\s*/\s*\w+)?)',
    # Pattern 2: "currently on DRUG" / "on cycle N of DRUG"
    r'(?:currently\s+on|still\s+on)\s+(\w+(?:\s*/\s*\w+)?)',
    # Pattern 3: "cycle N [day N] of DRUG"
    r'cycle\s+\d+\s+(?:day\s+\d+\s+)?(?:of\s+)?(\w+)',
    # Pattern 4: "started DRUG on DATE" (近期) — 保留原始
    r'started\s+(\w+)\s+on\s+\d',
    # Pattern 5: "on DRUG cycle" / "on DRUG day"
    r'\bon\s+(\w+)\s+(?:cycle|day)\b',
    # Pattern 6: "receiving DRUG"
    r'(?:receiving|given)\s+(\w+)',
]
```

同时保留 PAST_CHEMO 过滤。

### Row 89 验证

Row 89 A/P: "will continue with cycle 4 of AC in 1 week"
- Pattern 1 匹配 "continue with cycle 4 of AC" → 捕获 "ac" ✅

### Row 99 验证

Row 99 A/P: "on Gemzar Cycle #2"
- Pattern 5 匹配 "on Gemzar cycle" → 捕获 "gemzar" ✅
- 但 "Gemzar" 后面是 "Cycle #2"，pattern 需要支持 `#\d+`

Row 99 HPI: "started irinotecan on 06/30/19" (这是 Row 1 不是 99)
Row 1 A/P: "started irinotecan on 06/30/19" + "cycle 3 day 1"
- Pattern 3 "cycle 3 day 1" 后面没有 "of DRUG"... 实际是 "She presents today for cycle 3 day 1"
- 需要 Pattern 4 匹配 "started irinotecan on 06/30/19" → "irinotecan" ✅
- 或 "will change her irinotecan to every other week" → Pattern 需要"change DRUG"？不，这是计划变更，irinotecan 确实是当前药物

让我再看 Row 1 A/P 文本中的关键句：
- "started irinotecan on 06/30/19" → Pattern 4 ✅
- "will continue with cycle 4 of AC" → 这不对，这是 Row 89。

Row 1 实际 A/P：
- "metastatic breast cancer who started irinotecan on 06/30/19"
- "will change her irinotecan to every other week"

Pattern 4 "started irinotecan on 06/30/19" → 捕获 "irinotecan" ✅

### 假阳性验证

Row 52 A/P: "Recommendation was made for adjuvant AC/THP chemotherapy"
- 无 Pattern 匹配（没有 continue/currently on/cycle/started/receiving）✅ 正确不触发

Row 40 A/P: "She has decided to proceed with AC-Taxol"
- 无 Pattern 匹配 ✅ 正确不触发

Row 64: 9 种药
- 需查原文确认无 active pattern

## 修改内容

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | run.py | 增强 IV_CHEMO_PATTERNS，删除 fallback 药名直接扫描块 |
| 2 | exp/v19_verify.yaml | 创建验证配置 |

## 修改记录

### run.py 行 1570-1622

**删除**：v18 的 fallback 药名直接扫描块（`for drug in KNOWN_CHEMO_IV: ... re.finditer(re.escape(drug), ap_lower_iv)`）

**替换为**：7 个正向匹配 pattern，只匹配明确的当前用药上下文：
1. `continue/continuing [with] [cycle N of] DRUG`
2. `currently on / still on DRUG`
3. `cycle N [day N] of DRUG`
4. `started DRUG on DATE`
5. `on DRUG cycle/day`
6. `receiving/given DRUG`
7. `DRUG day N`

保留 PAST_CHEMO 过滤（30 字符前文检查）。
保留 KNOWN_CHEMO_IV 药名白名单验证（pattern 捕获的词必须在白名单中才添加）。

### exp/v19_verify.yaml

创建验证配置，与 v18_verify.yaml 相同（61 行验证集）。
