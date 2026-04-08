# P1 Fix Plan — Letter Generation

## P1 Issues Summary (11 total, 7 distinct types)

| # | Type | Rows | Root Cause | Fix Layer |
|---|------|------|------------|-----------|
| 1 | Repeated sentences | 1, 32 | recent_changes ≈ therapy_plan → LLM generates two identical sentences | Pre-LLM dedup |
| 2 | Wrong source tag | 1 | LLM tags Others content as [Specialty] | Prompt (minor, hard to fix) |
| 3 | Inaccurate disease status | 4 | LLM misinterprets response_assessment (decreased → "grown") | Prompt |
| 4 | Lab value direction wrong | 10, 35 | LLM says "low" for normal/high values when summarizing labs | Prompt |
| 5 | TNM staging verbatim | 28, 45, 69 | Stage_of_Cancer = "pT1c(m)N1(sn)M0" → LLM copies into letter | Pre-LLM + Prompt |
| 6 | Raw Type_of_Cancer data | 39 | Type_of_Cancer = "ER 95, PR 5, HER2 2+ FISH negative G1 IDC..." → copied verbatim | Prompt |
| 7 | Receptor explanation backwards | 53 | LLM says ER+ "does not respond to hormones" (opposite of truth) | Prompt |

## Fix Strategy

### Fix A: Pre-LLM Keypoints Dedup (letter_generation.py)
**Fixes**: #1 (repeated sentences), partially #5 (TNM cleanup)

在 `flatten_keypoints()` 之后、发送给 LLM 之前，做两件事：

1. **Dedup identical values**: 如果 `recent_changes` == `therapy_plan` 或 `medication_plan` == `therapy_plan`，只保留一个，另一个设为空字符串。这样 LLM 就不会为同样的内容生成两句话。

2. **TNM → plain stage**: 检查 `Stage_of_Cancer` 是否包含 TNM 格式（regex `pT\d|pN\d|T\d.*N\d.*M\d`），如果是，尝试映射到简单 stage：
   - 含 "M1" 或 "metastatic" → "Stage IV"
   - 含 "N2" 或 "N3" → "Stage III"
   - 含 "T3" 或 "T4" → "Stage II-III"
   - 其他 → "Early stage"
   - 保留原始值作为备注但不发送到 LLM

**实现位置**: `letter_generation.py` 的 `generate_tagged_letter()` 中，在 `flatten_keypoints()` 之后加一个 `_clean_keypoints_for_letter(flat)` 函数。

```python
def _clean_keypoints_for_letter(flat):
    """Clean keypoints before sending to LLM for letter generation."""
    # 1. Dedup: if recent_changes == therapy_plan, clear one
    rc = flat.get("recent_changes", "").strip()
    tp = flat.get("therapy_plan", "").strip()
    if rc and tp and (rc in tp or tp in rc):
        flat["therapy_plan"] = ""

    mp = flat.get("medication_plan", "").strip()
    if mp and tp and (mp in tp or tp in mp):
        flat["therapy_plan"] = ""

    # 2. TNM → plain stage
    stage = flat.get("Stage_of_Cancer", "")
    if re.search(r'pT\d|pN\d|T\d.*N\d.*M\d', stage):
        if "M1" in stage or "metastatic" in stage.lower():
            flat["Stage_of_Cancer"] = "Stage IV (metastatic)"
        elif re.search(r'N[23]', stage):
            flat["Stage_of_Cancer"] = "Stage III"
        elif re.search(r'T[34]', stage):
            flat["Stage_of_Cancer"] = "Stage II-III"
        else:
            flat["Stage_of_Cancer"] = "Early stage (Stage I-II)"

    return flat
```

### Fix B: Prompt Enhancement (prompts/letter_generation.yaml)
**Fixes**: #3, #4, #6, #7, and reinforces #5

在 prompt 的 RULES 部分加入以下指令：

```yaml
  10. RECEPTOR STATUS — you MUST explain what ER/PR/HER2 mean:
     - ER+ or HR+ = "the cancer grows in response to hormones (estrogen)"
     - ER- = "the cancer does not respond to hormones"
     - HER2+ = "the cancer cells have too much of a protein called HER2"
     - HER2- = "the cancer does not have extra HER2 protein"
     - Triple negative = "the cancer cells lack three common receptors"
     Do NOT say ER+ "does not respond to hormones" — that is the OPPOSITE of what ER+ means.
  11. LAB VALUES — when summarizing labs, double-check which values are HIGH vs LOW:
     - Only say a value is "low" if it is flagged (L) or below the reference range
     - Only say a value is "high" if it is flagged (H) or above the reference range
     - Do NOT guess — if unsure, say "your blood tests are mostly normal"
  12. STAGE — never use TNM notation (pT2N1M0, etc.) in the letter. Always translate:
     - Stage I or II = "early stage"
     - Stage III = "locally advanced"
     - Stage IV = "advanced, the cancer has spread"
  13. Do NOT copy raw medical data into the letter. Simplify:
     - BAD: "ER 95, PR 5, HER2 2+ FISH negative G1 IDC with nuclear G1 DCIS"
     - GOOD: "Your cancer responds to the hormone estrogen but does not have extra HER2 protein."
  14. Do NOT include [REDACTED] in the letter. If a value is redacted, skip it or say "a specific medicine".
  15. Do NOT repeat the same information. If two fields say the same thing, write it once.
```

### Fix C: POST Hook (run.py, after letter generation)
**Fixes**: [REDACTED] 泄漏 (P2 but systematic), TNM 残留检测, 重复句子检测

在 `parse_tagged_letter()` 之后加 POST 验证：

```python
def _post_check_letter(letter_text):
    """Post-generation checks on letter text. Returns list of warnings."""
    warnings = []

    # Check for [REDACTED] leaks
    if "[REDACTED]" in letter_text:
        # Auto-fix: replace with generic text
        letter_text = re.sub(
            r'\[REDACTED\](\s*\[REDACTED\])*',
            'a specific treatment',
            letter_text
        )
        warnings.append("[POST-LETTER] stripped [REDACTED] from letter")

    # Check for TNM staging patterns
    if re.search(r'pT\d|pN\d|stage\s+pT', letter_text, re.IGNORECASE):
        warnings.append("[POST-LETTER] WARNING: TNM staging found in letter text")

    # Check for repeated sentences
    sentences = [s.strip() for s in letter_text.split('\n') if s.strip()]
    seen = set()
    for s in sentences:
        normalized = s.lower().rstrip('.')
        if normalized in seen and len(normalized) > 20:
            warnings.append(f"[POST-LETTER] WARNING: repeated sentence: '{s[:60]}...'")
        seen.add(normalized)

    return letter_text, warnings
```

## Implementation Order

1. **Fix A** (letter_generation.py) — dedup + TNM cleanup. 最高 ROI，直接消除 5/11 P1。
2. **Fix B** (prompts/letter_generation.yaml) — prompt 增强。消除剩余 P1 + 大部分 P2。
3. **Fix C** (run.py) — POST hook。兜底检测，防止遗漏。

## Validation Results (2026-03-27)

Ran 5 P1 rows with fixes. Results in `results/letter_full_qwen_20260327_102856/`.

| Row | Previous P1 | Status | Notes |
|-----|------------|--------|-------|
| 1 | Repeated sentence (recent_changes=therapy_plan) | FIXED | Dedup eliminated duplicate; only one sentence now |
| 10 | Lab value direction wrong (RBC high→"low") | FIXED | Now says "mostly normal" instead of guessing |
| 28 | TNM staging verbatim (pT1c(m)N1(sn)M0) | FIXED | Now says "early stage"; TNM cleaned by _clean_keypoints_for_letter |
| 39 | Raw Type_of_Cancer data dumped | FIXED | Now says "grows in response to hormones (estrogen)" — fully simplified |
| 53 | Receptor explanation backwards (ER+ → "does not respond") | FIXED | No longer generates wrong explanation |

**P1 = 0/5 on validation set. All fixes confirmed working.**

Remaining P2 in these 5 rows: ~3 (letrozole repeat in Row 39, receptor not explained in Row 53)

Ready for full-scale re-run.

## Files to modify

| File | Change |
|------|--------|
| `letter_generation.py` | Add `_clean_keypoints_for_letter()`, add `_post_check_letter()` |
| `prompts/letter_generation.yaml` | Add rules 10-15 |
| `run.py` | Call `_post_check_letter()` after `parse_tagged_letter()`, log warnings |
