# Letter v5 Enhancement Plan

## Changes (4 items)

### A. Receptor Pre-Translation (方案 B)
**文件**: `letter_generation.py` → `_clean_keypoints_for_letter()`
**目的**: 消除 P2 "receptor not explained" (7/61)

在发送给 LLM 之前，把 Type_of_Cancer 中的 receptor 标记预翻译为通俗语言：

```python
# In _clean_keypoints_for_letter(), after dedup, before [REDACTED] replacement:
toc = flat.get("Type_of_Cancer", "")
if toc:
    # Pre-translate receptor status into plain language
    explanations = []
    if re.search(r'ER\+|ER positive|HR\+', toc, re.IGNORECASE):
        explanations.append("grows in response to hormones (estrogen)")
    elif re.search(r'ER-|ER negative', toc, re.IGNORECASE):
        explanations.append("does not respond to hormones")
    if re.search(r'HER2\+|HER2 positive', toc, re.IGNORECASE):
        explanations.append("has extra HER2 protein")
    elif re.search(r'HER2-|HER2 negative', toc, re.IGNORECASE):
        explanations.append("does not have extra HER2 protein")
    if 'triple negative' in toc.lower() or 'TNBC' in toc:
        explanations = ["the cancer cells lack three common receptors (ER, PR, HER2)"]
    if explanations:
        flat["Type_of_Cancer"] = toc + " — in plain language: " + "; ".join(explanations)
```

这样 LLM 收到的 input 已经包含通俗解释，它只需要引用即可。

### B. Emotional Support
**文件**: `letter_generation.py` → `generate_tagged_letter()` + `prompts/letter_generation.yaml`
**目的**: 检测原文中的情绪信号，在 letter 中加入支持性语言

**步骤 1**: 在 `generate_tagged_letter()` 中，扫描 note_text 的情绪关键词：
```python
EMOTION_KEYWORDS = [
    "distressed", "anxious", "anxiety", "scared", "fearful", "crying", "tearful",
    "depressed", "depression", "worried", "overwhelmed", "upset", "emotional",
    "frightened", "nervous", "stressed",
]

def _detect_emotions(note_text):
    """Scan note for emotional state mentions."""
    note_lower = note_text.lower()
    found = [kw for kw in EMOTION_KEYWORDS if kw in note_lower]
    return found
```

**步骤 2**: 在 prompt 中注入情绪上下文（如有），并加 prompt rule：
```
  15. EMOTIONAL SUPPORT — If the input includes an "emotional_context" field,
      add 1-2 warm, supportive sentences acknowledging the patient's feelings.
      Example: "We understand this is a difficult time, and we are here to support you."
      Do NOT be dismissive. Do NOT say "don't worry". Use empathetic language.
```

**步骤 3**: 在 `_clean_keypoints_for_letter()` 中，如果检测到情绪关键词，加入 flat 字段：
```python
flat["emotional_context"] = "Patient appears anxious and scared."
```

### C. Medical Term Definitions (字典注入)
**文件**: `letter_generation.py` → `generate_tagged_letter()`
**目的**: 为 letter 中可能出现的少见术语提供字典定义

复用 extraction 的 `find_relevant_definitions()` 机制，但用于 letter 生成：

**步骤 1**: 在 `generate_tagged_letter()` 中，扫描 flattened keypoints 找术语：
```python
from ult import load_medical_dictionary, find_relevant_definitions, format_definitions_context

# Scan keypoints values for medical terms
kp_text = " ".join(v for v in flat.values() if isinstance(v, str))
med_dict = load_medical_dictionary()
definitions = find_relevant_definitions(kp_text, med_dict, max_terms=5)
```

**步骤 2**: 把定义注入 prompt（与 extraction 相同方式）：
```python
if definitions:
    defs_text = format_definitions_context(definitions)
    prompt_text = defs_text + "\n\n" + prompt_text
```

需要扩展 `INJECT_PRIORITY_TERMS` 加入 letter 专用术语：
- peritoneum, peritoneal
- lobular carcinoma, invasive lobular carcinoma
- mucinous, mucinous carcinoma
- metaplastic carcinoma
- seroma
- papillary carcinoma
- adenocarcinoma

### D. Dedup 修复 (response_assessment 保留独有信息)
**文件**: `letter_generation.py` → `_clean_keypoints_for_letter()`
**目的**: 修复 Row 4 的 P2（response_assessment dedup 太激进丢失好消息）

当前逻辑：如果 response_assessment ≈ findings（word overlap > 60%），清空 response_assessment。
问题：response_assessment 可能包含 findings 没有的信息（如 "decreased size"）。

**修复**: 不清空，而是只保留 response_assessment 中 findings 没有的部分：
```python
# Instead of clearing response_assessment entirely:
if ra and fi and _similar(ra, fi):
    # Keep only sentences in ra that are NOT in fi
    ra_sents = [s.strip() for s in re.split(r'[.;]', ra) if s.strip()]
    fi_lower = fi.lower()
    unique = [s for s in ra_sents if not any(w in fi_lower for w in s.lower().split() if len(w) > 4)[:3]]
    # Actually simpler: just keep ra sentences containing "decrease/improve/stable/respond/better/worse"
    response_words = {'decrease', 'decreased', 'increase', 'increased', 'improve', 'improved',
                      'stable', 'respond', 'responding', 'progression', 'better', 'worse', 'growing', 'shrink'}
    unique = [s for s in ra_sents if any(w in s.lower().split() for w in response_words)]
    flat["response_assessment"] = ". ".join(unique) if unique else ""
```

---

## Implementation Order
1. A (receptor pre-translate) — 改 `letter_generation.py`
2. D (dedup fix) — 改 `letter_generation.py`
3. B (emotional support) — 改 `letter_generation.py` + `prompts/letter_generation.yaml`
4. C (medical dict) — 改 `letter_generation.py`, 扩展 `INJECT_PRIORITY_TERMS`

## Test Rows (20 samples, prioritize problematic)
之前有问题的 rows: 0, 1, 4, 5, 8, 26, 29, 33, 40, 49, 51, 53, 94
补充到 20 个: + 6, 10, 13, 28, 35, 42, 64
```
row_indices: [0, 1, 4, 5, 6, 8, 10, 13, 26, 28, 29, 33, 35, 40, 42, 49, 51, 53, 64, 94]
```

## 验证标准
- P2 "receptor not explained" → 0
- Row 4 response_assessment 好消息保留
- 情绪相关 row 有 1-2 句支持性语言
- 少见术语解释出现
- 之前 P1 不回归
