# Letter v8a Full Run Review

**Run**: full_qwen_20260402_095523
**Date**: 2026-04-02
**Samples**: 100 (full CORAL breastca_unannotated.csv)
**Pipeline**: v2 extraction + v8a letter generation

## Executive Summary

| Metric | Result |
|--------|--------|
| P0 (hallucination) | 0 |
| P1 (systematic) | 3 patterns: emotion false positive (fixed), pCR misleading (DR-3), extraction miss prereqs (DR-4/5) |
| P2 (minor) | 5 patterns: truncation, TCHP leak, low coverage, met sites too detailed (DR-1), televisit (DR-2) |
| Template compliance | 100/100 |
| Letters complete (not truncated) | 97/100 (97%) |
| [REDACTED] leak in letters | 0/100 |
| Dosing leak in letters | 0/100 |
| Jargon leak (TCHP) | 3/100 |
| Avg readability grade | 5.7 (target <8.0) |
| Avg field coverage | 72% |

---

## P1: Emotion Detection False Positive (FIXED)

**Problem**: The emotion keyword detector (`_EMOTION_KEYWORDS`) searches for words like "anxiety", "depression", "anxious" using simple substring matching (`kw in note_lower`). This fails on negated ROS entries like:
- "PSYCH: **No** depression, or anxiety or trouble sleeping."
- "Psychiatric: **Negative** for depression, anxiety"

The detector matches "anxiety" and "depression" in these **negated** statements, causing the letter to say things like "It seems you are feeling anxious and depressed" when the patient explicitly denied these symptoms.

**Clinical impact**: Telling a patient "you seem anxious and depressed" when they are NOT is clinically inappropriate and could cause unnecessary worry.

**Affected samples**: All samples whose ROS mentions "No depression/anxiety" (estimated 20+ of 100).

**Fix applied**: Commit `6b6f971d` — Added negation filter. Before accepting a keyword match, checks the preceding 40 characters for negation words (no, not, denies, denied, negative, without, absent). If negation found, keyword is excluded.

**Status**: Fixed in code. Needs re-run of letter generation (extraction unchanged).

---

## P2 Issues

### P2-1: Letter truncation (3/100)

**Affected**: ROW 32, ROW 84, ROW 99

All three are complex cases with many treatment details. The letter reached max_new_tokens (768) before completing.

ROW 32 ends with: `"...Please let us know if you need any help with this. ["`
ROW 84 ends with: `"...Please let us know if you need support.\nThank you"`
ROW 99 ends with: `"...We will also discuss ways to manage your anxiety and depression.\nThank you"`

**Fix**: Increase max_new_tokens to 1024 for letter generation. 768 is sufficient for 97% of cases but complex metastatic cases need more room.

### P2-2: TCHP abbreviation leaked (3/100)

**Affected**: ROW 15, ROW 17, ROW 52

Letters contain "TCHP" (a chemotherapy regimen abbreviation) without explanation. Should be written as "a combination of chemotherapy drugs" or expanded.

**Fix**: Add TCHP/AC/TC to the medical terms list in the prompt, or add a POST check to replace chemo abbreviations.

### P2-3: Low field coverage (<60%)

**Affected**: 14 samples with coverage below 60%

| ROW | Coverage | Key fields missed |
|-----|----------|-------------------|
| 4 | 58% | imaging_plan, lab_plan, supportive_meds |
| 6 | 54% | current_meds, findings, goals_of_treatment |
| 12 | 62% | imaging_plan, radiotherapy_plan |
| 23 | 54% | goals_of_treatment, findings |
| 28 | 58% | current_meds, radiotherapy_plan |
| 34 | 50% | findings, goals_of_treatment, response_assessment |
| 35 | 58% | findings, current_meds |
| 44 | 53% | current_meds, findings |
| 48 | 54% | findings, goals_of_treatment |
| 51 | 55% | goals_of_treatment, findings |
| 69 | 55% | current_meds, findings |
| 72 | 53% | findings, lab_summary |
| 77 | 56% | current_meds, lab_summary |
| 84 | 50% | findings, response_assessment |

Pattern: Most commonly missed fields are `findings` and `goals_of_treatment` in the "What was discussed?" section. These fields contain information but the LLM doesn't always reference them with the correct source tag, reducing measured coverage. The content may still be present but untagged.

**Fix**: Lower priority. Source tag coverage is a tracking metric, not a content issue. Spot-checks show the content is usually present but tagged under a different field name.

---

## Metrics Summary

### Readability (Flesch-Kincaid Grade)
- Min: 3.9, Max: 7.5, Mean: 5.7, Median: 5.6
- 100% below 8th grade target
- Note original mean: 11.0 (significant simplification achieved)

### Field Coverage
- Min: 50%, Max: 100%, Mean: 72%, Median: 71%
- 2 samples at 100% coverage
- 14 samples below 60%

### Template Compliance
- 100/100 samples follow 5-section template structure
- All section headers present (**Why did I come?** / **What was discussed?** / etc.)

### POST Check Performance
- 0 [REDACTED] leaks in any letter
- 0 dosing detail leaks
- 0 TNM staging notation in letters
- POST-LETTER checks had zero warnings (all handling done pre-LLM)

---

## v8a Fixes Verification (from v8 test)

| Fix | Status in full run |
|-----|-------------------|
| peritoneum explained | Verified in S0: "the lining of your abdomen (belly area)" |
| axilla → armpit | Verified: "armpit area" used consistently |
| Advance care included | Verified: "Your advance care status is noted as Full code" appears where applicable |
| Mixed response both sides | Verified in S11: "stable, but...new small areas of cancer in the brain" |
| Why section no jargon | Mostly compliant, 3 exceptions with TCHP |

---

## Doctor Review Feedback (2026-04-02)

医生朋友对 v8a 10-sample 测试结果 (`test_v8_template_20260402_091236`) 的真人审查意见：

### DR-1: 转移部位不要逐个列举 (S0, Letter)
**问题**: Letter 写 "spread to...your lungs, the lining of your abdomen (belly area), your liver, and your ovaries" — 医生认为太详细，对患者来说信息量过大。
**建议**: 简化为 "has spread to other parts of your body"，不需要列出每个器官。
**分类**: P2 (letter prompt 层面)
**修复**: 在 prompt "What was discussed?" section 加指令：对转移部位，说 "spread to other parts of your body" 即可，除非只有 1-2 个部位才具体说。

### DR-2: Televisit 分类争议 (S7, Extraction)
**问题**: S7 keypoints 提取为 `"in-person": "Televisit"`。原文 HPI 写 "presents through ZOOM"，但医生认为这不算 televisit。
**分类**: P2 (extraction 层面，视角差异)
**备注**: 原文 physical exam "Vital Signs - None taken" 也支持远程就诊。可能是医生对该 case 的理解不同。暂不改。

### DR-3: pCR 表述误导 (S7, Letter) — P1
**问题**: Letter 写 "After treatment and surgery, no cancer was found in the removed tissue. This is a very good sign." 但实际情况是**乳房**无残留癌（breast pCR），但 **3/28 淋巴结仍有癌**。Letter 紧接着说 "However, 3 of 28 lymph nodes under your arm had cancer"，但前一句 "no cancer was found in the removed tissue" 已经造成了误导 — 患者会先理解为完全没有癌，再被 "however" 打击。
**正确表述**: "After treatment and surgery, no cancer was found in the breast tissue itself, which is a good sign. However, cancer was still found in 3 of the 28 lymph nodes that were removed."
**分类**: P1 (临床误导)
**修复**: 修改 prompt Rule 13 的 pCR 规则：必须区分 breast pCR 和 complete pCR。如果 LN 仍有癌，不能笼统说 "no cancer was found in the removed tissue"。

### DR-4 & DR-5: Extraction 遗漏 — port placement 和 echocardiogram (S7)
**问题**: S7 的 A/P 明确写了 "the steps that would need to be taken **in order to start**, including **port placement** and **echocardiogram**"。但提取结果：
- `procedure_plan`: "adjuvant AC x 4 cycles" （错误 — 这是 therapy，不是 procedure）
- `imaging_plan`: "No imaging planned." （遗漏 echocardiogram/TTE）

**根因**: 模型把 "AC x 4 cycles" 当作 procedure 提取，但漏掉了紧随其后的 "port placement" 和 "echocardiogram"。这些是化疗前置步骤，用关键短语描述：
- "prior to starting"
- "before initiating"
- "needs to undergo"
- "baseline"
- "steps required"
- "in order to start"

**临床背景**（医生解释）:
- **Port placement**: IV 化疗需要植入式给药装置（port），是常规前置操作
- **Echocardiogram (TTE/Echo)**: AC 类药物（蒽环类）有心脏毒性，开始前必须检查心功能
- 这些是临床常识性的关联：AC → 需要 echo；IV chemo → 需要 port
- 理想情况下，模型应该既能从文本直接提取这些步骤，也能通过临床知识推断

**分类**: P1 (extraction 遗漏关键信息)
**修复方向**:
1. **直接提取改进**: 在 `procedure_plan` 和 `imaging_plan` 的 prompt 中加入关键短语提示："Look for pre-treatment prerequisites mentioned with phrases like 'prior to starting', 'before initiating', 'in order to start', 'steps required', 'baseline'. These include port placement, echocardiogram (TTE/Echo), chemotherapy teaching, dental clearance."
2. **临床知识推断（可选）**: 如果 therapy_plan 包含 AC/anthracycline，检查 imaging_plan 是否包含 echo/TTE；如果 therapy_plan 包含 IV chemo，检查 procedure_plan 是否包含 port placement。如果缺失且 note 中提到了，补充。

---

## Recommended Next Steps

1. **Re-run letter-only** with emotion negation fix (already committed)
2. **Increase max_new_tokens** to 1024 for letter generation
3. **Add TCHP/AC/TC** to prompt Rule 15 medical terms list
4. **DR-1**: 简化转移部位列举 — prompt 改为 "keep metastasis sites general unless only 1-2 sites"
5. **DR-3**: 修改 pCR Rule 13 — 区分 breast pCR 和 complete pCR
6. **DR-4/5**: 改进 extraction prompt — 加入化疗前置步骤关键短语提示
7. Field coverage improvements are lower priority — content is present, just tagging inconsistency
