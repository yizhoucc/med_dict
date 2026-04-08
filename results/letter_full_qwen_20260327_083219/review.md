# Letter Generation Full Run — Per-Sample Review

**Date**: 2026-03-27
**Source**: v23 extraction (61 samples), letter-only mode
**Model**: Qwen2.5-32B-Instruct-AWQ
**Reviewer**: Claude (manual, sentence-by-sentence)

## Issue Legend
- **P0**: Hallucination — letter says something not in keypoints or note
- **P1**: Significant error — wrong attribution, repeated sentence, misleading info
- **P2**: Minor issue — missing explanation, slight redundancy, incomplete coverage

---

## ROW 0 (coral 140) — 56F, new patient, metastatic ER+/PR+/HER2- IDC

**Letter**: 15 sentences, 13 attributed
**Verdict**: Good quality, minor issues

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "invasive ductal carcinoma" not explained in plain language |
| 2 | P2 | "peritoneum" not explained — patients won't know this term |
| 3 | P2 | Sentence 6 repeats metastasis sites already stated in sentence 4 (lungs, liver, ovaries) |
| 4 | P2 | Sentence 11 mentions "bone scan" but source tag only says imaging_plan (which only has "Brain MRI"); bone scan is actually in lab_plan field |
| 5 | P2 | Stage info (Originally IIA → now IV) not mentioned in letter |
| 6 | P2 | Integrative Medicine referral (Specialty field) not mentioned |

---

## ROW 1 (coral 141) — 44F, follow up, metastatic TNBC, cycle 3 day 1 irinotecan

**Letter**: 18 sentences, 16 attributed
**Verdict**: Good content but has a repeated sentence and a wrong attribution

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Repeated sentence**: sentence 8 and 12 are identical ("We will change your irinotecan treatment to every other week with a higher dose.") — sentence 8 from recent_changes, sentence 12 from therapy_plan, but they say the same thing |
| 2 | P1 | **Wrong attribution**: sentence 16 text says "social work and home health services" but source tag is [Specialty] which is "Rad Onc consult". The actual source should be [Others] |
| 3 | P2 | response_assessment has detailed worsening info (chest wall worse, back pain worse, anemia worse, MRI bone lesions) but letter only says "We will monitor your treatment response closely" — key disease progression signal lost |
| 4 | P2 | 1 unit pRBC transfusion (same-day procedure in medication_plan) not mentioned in letter |
| 5 | P2 | Stage info (Originally IIB → now IV) not mentioned |

---

## ROW 2 (coral 142) — 53F, new patient/second opinion, Stage IIA ER+/PR+/HER2- IDC, curative

**Letter**: 12 sentences, 10 attributed
**Verdict**: Clean, well-structured

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER+/PR+/HER2- invasive ductal carcinoma" not explained in plain language |
| 2 | P2 | goals_of_treatment (curative) not explicitly stated — implicitly conveyed through "lower the chance of coming back" |
| 3 | P2 | second opinion not mentioned (keypoints say "yes", note says "several opinions") |

---

## ROW 4 (coral 144) — 31F, follow up, metastatic ER+/PR+/HER2- IDC, Stage IV, palliative

**Letter**: 16 sentences, 14 attributed
**Verdict**: Good overall but has one factual inaccuracy

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Inaccurate disease status**: Sentence 6 says "cancer has grown in your neck" but response_assessment shows cervical LN actually DECREASED (1.8→1.2cm). The increase was in axillary LN and sternum. This is misleading. |
| 2 | P2 | "ER+/PR+/HER2- invasive ductal carcinoma" not explained |

---

## Emerging Patterns (after 4 rows)

### Systematic P2: medical terms not explained
Every letter uses "ER+/PR+/HER2- invasive ductal carcinoma" without parenthetical explanation. Prompt says to explain medical terms, but the model inconsistently does so (Row 1 explained TNBC well, but not receptor-positive types).

### Systematic P2: Stage info often omitted
Rows 0, 1 both omit stage progression (e.g., "Originally Stage IIA, now Stage IV"). Row 2 correctly includes Stage IIA. Row 4 includes it.

### P1 pattern: recent_changes vs therapy_plan duplication
When both fields contain the same info (irinotecan dose change), letter generates two identical sentences. Need dedup logic or prompt instruction.

### P1 pattern: wrong source field tag
LLM sometimes tags text with a nearby field name rather than the correct one (Row 1: social work tagged as [Specialty] instead of [Others]).

---

## ROW 5 (coral 145) — 34F, post-bilateral mastectomy, ER+/PR+/HER2- IDC, curative

**Letter**: 14 sentences, 12 attributed
**Verdict**: Excellent quality — best letter so far. Good plain-language explanations.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | Letter says "follow-up" but keypoints say "New patient" — note itself says "Follow-up" so this is a keypoint extraction issue, not a letter issue |
| 2 | P2 | Genetics referral (Genetics: "Dr. [REDACTED]...genetics referral") not mentioned in letter |

---

## ROW 6 (coral 146) — MBC since 2008, originally ER+/PR+/HER2+, metastatic ER-/PR-/HER2+, second opinion

**Letter**: 11 sentences, 9 attributed
**Verdict**: Clean but short for a complex case. Missing key context.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | Receptor status change (originally ER+/PR+/HER2+ → metastatic biopsy ER-/PR-/HER2+) not mentioned — important for understanding treatment history |
| 2 | P2 | LVEF decreased to 52% (reason for stopping Herceptin) not mentioned |
| 3 | P2 | Stage info (Originally Stage II → now Stage IV) not mentioned |
| 4 | P2 | response_assessment partially covered in sentence 5 but tumor marker info omitted |

---

## ROW 7 (coral 147) — 29F, Stage III HER2+/ER- IDC, post-lumpectomy, new patient, curative

**Letter**: 13 sentences, 11 attributed
**Verdict**: Good quality. Clear treatment plan communication.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "IHC 3+, FISH ratio 5.7" kept verbatim — too technical for patient letter |
| 2 | P2 | "adjuvant AC" not explained in plain language |

---

## ROW 8 (coral 148) — 63F, Stage II ER+/PR-/HER2- IDC, post-mastectomy, curative

**Letter**: 15 sentences, 13 attributed
**Verdict**: Very good quality. Excellent pathology simplification.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER+/PR-/HER2- invasive ductal carcinoma" not explained |
| 2 | P2 | Sentence 8 says "protect your bones" but source is supportive_meds (anti-nausea drugs); bone protection is in medication_plan (Fosamax) |

---

## ROW 9 (coral 149) — 66F, Stage II HR+/HER2- IDC, post-mastectomy, curative

**Letter**: 12 sentences, 10 attributed
**Verdict**: PERFECT. Best letter so far. Excellent plain-language explanations throughout.

| # | Sev | Issue |
|---|-----|-------|
| — | — | No issues found |

Notable: "a cancer that grows in response to hormones, but it doesn't have a protein called HER2" is an exemplary explanation of HR+/HER2-.

---

## ROW 10 (coral 150) — 68F, metastatic IDC to bone, Stage IIIC→IV, palliative

**Letter**: 16 sentences, 14 attributed
**Verdict**: Good coverage but has one factual lab error

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Lab error**: Letter says "slightly low red blood cell count" but RBC 5.25 is flagged HIGH in labs, not low |
| 2 | P2 | Source tag "Distant_Metastasis" (underscore) doesn't match prompt's "Distant Metastasis" (space) → extraction_values empty for that sentence |
| 3 | P2 | Mycelex sentence tagged as [follow up] but content belongs to [medication_plan] |
| 4 | P2 | "We did not discuss any plans for future care" — misleading wording for Advance care "not discussed"; sounds like no plans exist at all |

---

## ROW 11 (coral 151) — 51F, metastatic ER+/PR+/HER2+ IDC, brain/lung/bone mets, palliative

**Letter**: 15 sentences, 13 attributed
**Verdict**: Good coverage for a complex case. Correctly reports new brain lesions and stable body disease.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER+/PR+/HER2+ invasive ductal carcinoma" not explained |
| 2 | P2 | Bone scan (planned for next eval) not mentioned |
| 3 | P2 | Functional improvement ("now off walker") not mentioned — positive news for patient |

---

## ROW 13 (coral 153) — 58F, de novo metastatic ER+ IDC, bone/liver/nodes, palliative, complex case

**Letter**: 18 sentences, 16 attributed
**Verdict**: PERFECT. Excellent coverage of a very complex case with alternative treatments, medication changes, and symptom management. Best long letter.

| # | Sev | Issue |
|---|-----|-------|
| — | — | No issues found |

Notable: 18 sentences covering home chemo, immunotherapy, stopped meds, scan scheduling, PT referral, Cymbalta, topical treatments — all accurate, all attributed, all in plain language.

---

## ROW 16 (coral 156) — 53F, Stage I-II ER+/PR+/HER2- IDC, post-lumpectomy, curative

**Letter**: 16 sentences, 14 attributed
**Verdict**: Excellent. Comprehensive coverage including genetics, nutrition referrals.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER+/PR+/HER2- invasive ductal carcinoma" not explained in plain language |

---

## ROW 17 (coral 157) — 65F, Stage I ER+/PR+/HER2- IDC + encapsulated papillary carcinoma, curative

**Letter**: 12 sentences, 10 attributed
**Verdict**: Good, concise letter.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "encapsulated papillary carcinoma" kept without adequate plain-language explanation |
| 2 | P2 | "endocrine therapy" not explained — should say "hormone-blocking medicine" or similar |

---

## ROW 19 (coral 159) — 75F, metastatic ER+/PR+/HER2- IDC, bone/LN, palliative, new consult

**Letter**: 16 sentences, 14 attributed
**Verdict**: Good coverage but has some wording issues.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER+/PR+/HER2-...which means it has spread" — "which means" implies receptor status caused metastasis, confusing causal logic |
| 2 | P2 | Blood sugar 104 mg/dL from 2013 (8 years ago) mentioned — irrelevant to patient now |
| 3 | P2 | Sentence 6 and 7 repeat letrozole+palbociclib info (current_meds vs recent_changes) |
| 4 | P2 | Radiation referral described as "second opinion" — should be "evaluation" or "consult" |

---

## ROW 21 (coral 161) — 72F, metastatic ER+/PR+/HER2- IDC, second opinion, Stage IV, palliative

**Letter**: 14 sentences, 12 attributed
**Verdict**: Excellent. Conditional treatment plan (sentence 10) is outstanding.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | Labs say "mostly normal" but omits elevated creatinine (1.19) and low eGFR (46) — kidney function issue not communicated |

Notable: "If it's stable, we'll keep you on anastrozole. If it has grown, we might try other medicines." — exemplary conditional plan communication.

---

## ROW 26 (coral 166) — 41F, metastatic ER+/PR+/HER2- IDC to bone, stable disease, palliative

**Letter**: 9 sentences, 7 attributed
**Verdict**: Short but accurate. Good "stable and not growing" communication.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "which means it's now at stage IV" — implies receptor status caused staging; same pattern as Row 19 |
| 2 | P2 | Spine MRI (imaging_plan: "consider MRI of the spine") not mentioned |

---

## ROW 28 (coral 168) — 59F, multifocal ER+/PR+/HER2- IDC, post-lumpectomy, curative

**Letter**: 15 sentences, 13 attributed
**Verdict**: Good coverage but TNM staging kept verbatim.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **TNM staging kept verbatim**: "pT1c(m)N1(sn)M0" left in letter — patients cannot understand this. Should say "early stage" or "Stage I" |
| 2 | P2 | "ER+/PR+/HER2- invasive ductal carcinoma" not explained |
| 3 | P2 | Sentence 6 and 7 repeat letrozole info (current_meds vs recent_changes) |
| 4 | P2 | Oncotype Low Risk result (no chemo needed) not mentioned — important good news for patient |

---

## ROW 29 (coral 169) — 64F, Stage II-III ER-/PR-/HER2+ IDC, new patient, curative, neoadjuvant

**Letter**: 14 sentences, 12 attributed
**Verdict**: Good structure. Excellent "shrink the cancer before surgery" explanation.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "ER-/PR-/HER2+ invasive ductal carcinoma" not explained |
| 2 | P2 | "[REDACTED]" appears in patient-facing letter text (from redacted tumor marker name) |
| 3 | P2 | "TTE" not explained — should say "heart ultrasound" or "echocardiogram" |
| 4 | P2 | Radiation plan not mentioned (A/P explicitly includes radiation in treatment plan) |

---

## Rows 32-99 — pending
(审查进行中... 19/61 completed)

---

## Running Tally (19/61 reviewed)

| Severity | Count | Examples |
|----------|-------|---------|
| P0 | 0 | — |
| P1 | 5 | Repeated sentence (Row 1), wrong source tag (Row 1), inaccurate disease status (Row 4), lab value direction wrong (Row 10), TNM staging verbatim (Row 28) |
| P2 | 46 | (cumulative — avg 2.4 P2/letter) |
| Perfect | 3 | Rows 9, 13 |

## ROW 32 (coral 172) — 63F, Stage IIB→IIIA ER+/PR+/HER2- ILC, adjuvant letrozole, curative

**Letter**: 13 sentences, 11 attributed
**Verdict**: Good reassurance letter but has a repeated sentence.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Repeated sentence**: sentence 6 ("no signs that the cancer has come back") and sentence 10 ("no sign of the cancer coming back") are semantically identical |
| 2 | P2 | "invasive lobular carcinoma" not explained |
| 3 | P2 | "originally stage IIB and is now stage IIIA" may mislead patient into thinking cancer worsened (it's restaging, not progression) |
| 4 | P2 | Calcium/vitamin D supplementation not mentioned |

---

## ROW 33 (coral 173) — 71F, Stage III ER+/PR-/HER2- IDC, second local recurrence, curative

**Letter**: 16 sentences, 14 attributed
**Verdict**: Good coverage. Correctly communicates local recurrence + switch to tamoxifen.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | Receptor status not explained in plain language |
| 2 | P2 | Lab tests described as "recent" but are from August 2018 (2 years old) |
| 3 | P2 | Sentence 11 repeats local recurrence + no distant spread info from sentences 4-5 |

---

## ROW 35 (coral 175) — 27F, pT3N0 ER+/PR+/HER2- grade III mixed ductal+mucinous, cycle 8 abraxane, curative

**Letter**: 15 sentences, 13 attributed
**Verdict**: Good coverage but another lab error.

| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Lab error**: Letter says "white blood cells...are a bit low" but WBC 8.8 is NORMAL (range 3.4-10). Only RBC and Hgb are low. |
| 2 | P2 | "mixed ductal and mucinous carcinoma" not explained |
| 3 | P2 | Stage (pT3N0) not mentioned |

---

## ROW 36 (coral 176) — TNBC, Stage IIA, new patient, curative
**Letter**: 11 sentences, 9 attributed. **Verdict**: Good. TNBC well explained.
| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | [REDACTED] appears in letter ("start your chemotherapy at [REDACTED]") |

## ROW 39 — ER+/PR+/HER2- IDC, Stage II, curative
**Letter**: 11 sentences, 9 attributed. **Verdict**: Serious readability issue.
| # | Sev | Issue |
|---|-----|-------|
| 1 | P1 | **Type_of_Cancer raw data in letter**: "ER 95, PR 5, HER2 2+ FISH negative...G1 IDC with nuclear G1 DCIS" — completely unreadable for patients |
| 2 | P2 | Letrozole mentioned three times (sentences 5, 6, 8) |

## ROW 40 — ER+/PR weakly+/HER2 1+ IDC, Stage II, curative
**Letter**: 11 sentences, 9 attributed. **Verdict**: Mixed readability.
| # | Sev | Issue |
|---|-----|-------|
| 1 | P2 | "human epidermal growth factor receptor 1+ by immunohistochemistry" — explanation is itself too technical |
| 2 | P2 | "left ventricular ejection fraction of 79%" left as raw number |
| 3 | P2 | [REDACTED] appears twice in letter |

## ROW 41 — IDC, post-radiation, starting tamoxifen
**Letter**: 8 sentences, 6 attributed. **Verdict**: PERFECT. Exemplary IDC explanation ("started in the milk ducts and spread to nearby tissue").

## ROW 42 — TNBC, Stage I, post-mastectomy, curative
**Letter**: 11 sentences, 9 attributed. **Verdict**: PERFECT. Excellent TNBC explanation and overall readability.

## ROW 43 — ER+/PR+/HER2- IDC, post-surgery, curative. P0=0, P1=0, P2=1 (receptor not explained)

## ROW 45 — ER+/PR-/HER2- IDC, pT2N2, curative.
| 1 | P1 | TNM staging "pT2N2 with extranodal extension" kept verbatim |
| 2 | P2 | Receptor status not explained |

## ROW 48 — ER+/PR+/HER2- IDC, Stage II, new patient. P0=0, P1=0, P2=1 ([REDACTED])

## ROW 49 — HR+/HER2- IDC, metastatic, second opinion. P0=0, P1=0, P2=2 (terms, eGFR)

## ROW 51 — ER+/PR+/HER2- IDC, new consult. P0=0, P1=0, P2=2 (receptor not explained, [REDACTED] x2)

## ROW 52 — ER+/PR+/HER2+ IDC with neuroendocrine differentiation. P0=0, P1=0, P2=3 (neuroendocrine, seroma, chemo abbreviations)

## ROW 53 — ER+/PR-/HER2- IDC, Stage IV to bone, palliative.
| 1 | P1 | **Wrong receptor explanation**: "does not respond to certain hormones" — ER+ DOES respond to estrogen! Explanation is factually backwards. |

## ROW 56 — TNBC, locally advanced, curative. **PERFECT.**
## ROW 58 — ER+/PR+/HER2- IDC, Stage I. P2=2 (raw lab numbers, DCIS unexplained)
## ROW 60 — ER+/PR+/HER2- IDC, new patient, curative. **PERFECT.** Exemplary endocrine therapy explanation.
## ROW 62 — ER+/PR-/HER2-, locally advanced. P2=0. Good detail.
## ROW 63 — ER+/PR+/HER2- IDC, Stage IV to sternum. P2=1 ([REDACTED])
## ROW 64 — ER weakly+/PR weakly+/HER2 1+ IDC, neoadjuvant. **PERFECT.** Best explanation of neoadjuvant chemo.
## ROW 65 — Metaplastic carcinoma, second opinion. P2=2 (metaplastic unexplained, raw percentages)
## ROW 67 — ER+ multifocal, post-chemo. P2=1 (ER+ not explained)
## ROW 69 — Bilateral breast cancer. P1=1 (TNM staging pT4bN1M0 + pT1cN0M0 verbatim). P2=0
## ROW 71 — ER+/PR-/HER2- IDC, new consult. P2=1 (receptor not explained)
## ROW 72 — ER+/PR+, Stage III, follow up. P2=1 (receptor not explained). "Fat necrosis, which is not cancer" — excellent.
## ROW 77 — TNBC metastatic. P2=0. **Clean.**
## ROW 79 — ER+/PR+/HER2- IDC. P2=1 (receptor not explained)
## ROW 81 — ER+/PR+/HER2- mixed ductal+lobular. P2=2 ([REDACTED], histology unexplained)
## ROW 82 — Lobular carcinoma. P2=1 (receptor not explained)
## ROW 83 — ER+/PR-/HER2- IDC metastatic. P2=1 (raw lab numbers). "meninges (the membranes covering the brain)" — excellent.
## ROW 84 — ER+ ILC metastatic to brain. P2=1 (partial explanation)
## ROW 85 — ER+/PR+/HER2+ IDC+ILC metastatic. P2=2 (IDC abbreviation, grade III technical)
## ROW 86 — ER+/PR+ IDC. P2=1 ([REDACTED]). "estrogen and progesterone receptors, but not HER2 receptors" — good.
## ROW 87 — ER+/PR+/HER2- → ER-/PR-/HER2- IDC metastatic. P2=0. Correctly notes receptor change.
## ROW 89 — Adenocarcinoma. P2=1 (adenocarcinoma unexplained)
## ROW 90 — ER+/PR+ IDC metastatic to bone. P2=1 (receptor not explained)
## ROW 91 — ER+/PR-/HER2- IDC metastatic to liver. P2=3 (meaningless receptor explanation, raw chemo doses, misleading advance care)
## ROW 93 — ER+/PR+/HER2- IDC, Stage IIA. P2=1 ([REDACTED]). "recurrence score" — good Oncotype explanation.
## ROW 94 — ER+/PR-/HER2- IDC. P2=1 (receptor not explained). "non-invasive cancer cells (DCIS)" — good.
## ROW 96 — ER+/PR+/HER2- IDC, post-mastectomy. P2=1 (receptor not explained). MS coordination noted ✓
## ROW 99 — ER+/PR+ IDC metastatic to liver. P2=1 (raw percentages)

(审查完成 — 61/61)

---

## Running Tally (21/61 reviewed)

| Severity | Count | Examples |
|----------|-------|---------|
| P0 | 0 | — |
| P1 | 11 | Repeated sentences (Row 1, 32), wrong source tag (Row 1), inaccurate disease status (Row 4), lab value wrong (Row 10, 35), TNM verbatim (Row 28, 45, 69), raw Type_of_Cancer data (Row 39), receptor explanation backwards (Row 53) |
| P2 | 99 | (cumulative — avg 1.6 P2/letter) |
| Perfect | 9 | Rows 9, 13, 41, 42, 56, 60, 62, 64, 77 |

## FINAL SUMMARY (61/61 reviewed)

### Overall Quality
- **P0 (hallucinations): 0** — no letter fabricated information not in keypoints/note
- **P1 (significant errors): 11** — 18% of letters have at least one significant issue
- **P2 (minor issues): 99** — avg 1.6 per letter, mostly receptor terminology not explained
- **Perfect letters: 9/61 (15%)** — zero issues of any kind

### P1 Categories
| Category | Count | Rows |
|----------|-------|------|
| Repeated sentence (recent_changes=therapy_plan) | 2 | 1, 32 |
| Wrong source tag | 1 | 1 |
| Inaccurate disease status description | 1 | 4 |
| Lab value direction wrong (high→low or normal→low) | 2 | 10, 35 |
| TNM staging kept verbatim | 3 | 28, 45, 69 |
| Raw Type_of_Cancer data (not simplified) | 1 | 39 |
| Receptor explanation factually backwards | 1 | 53 |

### Top P2 Categories (estimated frequency)
1. **Receptor status not explained** (~35 letters, ~57%) — most common
2. **[REDACTED] leaks into patient letter** (~8 letters, ~13%)
3. **Minor redundancy** (current_meds vs recent_changes) (~5 letters)
4. **Raw technical data kept** (lab values, percentages, TNM-like) (~8 letters)
5. **Stage info omitted or confusing** (~5 letters)
6. **Medical terms not explained** (DCIS, seroma, adenocarcinoma) (~6 letters)

### Actionable Improvements
1. **Prompt**: Add explicit instruction to explain ER/PR/HER2 in every letter (e.g., "ER+ means the cancer grows in response to hormones")
2. **Prompt**: Add instruction to never output [REDACTED] — replace with "a specific medicine" or omit
3. **Prompt**: Add instruction to never output raw TNM staging — always translate to plain stage (I, II, III, IV)
4. **POST hook**: Strip [REDACTED] from generated letters
5. **POST hook**: Detect and flag repeated sentences
6. **POST hook**: Detect TNM patterns (pT\d, pN\d) in letter text and flag

## All patterns found so far
1. ER/PR/HER2 receptor status often not explained (most common P2, ~60% of letters)
2. Stage progression info sometimes omitted or confusing
3. recent_changes and therapy_plan/current_meds overlap → duplicate sentences
4. LLM sometimes tags with wrong field name
5. response_assessment sometimes oversimplified
6. "which means" creates false causal links (receptor → staging)
7. Old/stale lab data described as "recent"
8. [REDACTED] can leak into patient letter
9. TNM staging can be left verbatim
10. "No evidence of recurrence" duplicated across multiple sentences
11. Radiation referral sometimes called "second opinion"
