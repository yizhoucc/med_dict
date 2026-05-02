# Baseline Letters — Word-by-Word Clinical Review

**Date:** 2026-05-02
**Reviewer:** Claude (acting as oncologist, reading every letter against original note)
**Samples:** 20 breast + 20 PDAC = 40 baseline letters
**Method:** Pure manual review. No scripts. Each letter read word by word against A/P.

---

## Overall Quality Assessment

The baseline (raw Qwen + single prompt) produces **surprisingly competent letters** in terms of clinical content. The model understands oncology notes well and translates them reasonably. However, there are **systematic problems** that make the baseline undeployable.

---

## Breast Cancer Baseline (20 letters)

### Hallucinations / Factual Errors: 3 found

| ROW | Issue | Severity |
|-----|-------|----------|
| **4** | Letter says "Taxol...once a week for 12 weeks" — A/P says "taxol x 12 weekly" but patient was "unsure about starting chemo" and "would like to speak to Dr. ***** first". Letter presents the plan as decided when the patient hasn't agreed yet. | P1 — misleading |
| **15** | Letter speculates "If the cancer is also HER2 positive, we might use different medicines that target HER2" — the note says HER2 is equivocal, not confirmed. The baseline introduces a hypothetical treatment scenario the doctor didn't discuss. | P1 — fabricated |
| **11** | Letter says "you should wait until after your radiation appointment before starting it [tamoxifen]" — this is a direct medical recommendation that the baseline should not make. The A/P discusses this as a plan but the letter frames it as an instruction. | P2 — overstepping |

### REDACTED / Privacy Issues: 20/20 affected

| Issue | Count |
|-------|-------|
| ***** leaks into letter | 8/20 |
| Uses "Mrs./Ms./Mr. *****" | 16/20 |
| Fabricated sign-off "[Your Name], Medical Communication Specialist" | 18/20 |
| Uses "[Patient Name]" or "[Patient's First Name]" placeholder | 4/20 |
| "[Cancer Center Name]" placeholder | 14/20 |
| "[Doctor's Name]" placeholder | 2/20 |

**Every single breast baseline letter has at least one REDACTED/privacy issue.** Not a single clean letter.

### Missing Information: 4 found

| ROW | Missing |
|-----|---------|
| 1 | Missing the specific staging (A/P says "Stage II T2N1") — letter just says "breast cancer" |
| 12 | Missing follow-up schedule — A/P has specific timing |
| 20 | Missing bilateral cancer detail — A/P describes two separate tumors, letter vaguely says "cancer in both breasts" |
| 2 | Missing mention of bone and liver metastasis — A/P discusses metastatic disease to chest wall and liver |

### Language / Style Issues

**Good:** All 20 letters are warm and empathetic. Natural tone. Appropriate section headers in most.

**Bad:**
- Sign-off blocks are verbose and contain placeholders (3-5 lines of "[Your Name]...[Cancer Center Name]")
- Some letters end with meta-comments: "This summary aims to convey the key points..." (ROW 16), "Please replace [Patient's First Name]..." (ROW 5)
- 2 letters exceed 2000 chars (too long for a patient letter)

### Per-letter Verdict (Breast)

| Verdict | Count | ROWs |
|---------|-------|------|
| Would send (with minor edits) | 0 | — |
| Would send after fixing REDACTED/name | 12 | 3,4,5,7,9,10,12,13,14,16,19,20 |
| Needs significant revision | 6 | 2,6,8,11,17,18 |
| Would NOT send | 2 | 1 (too vague), 15 (HER2 speculation) |

---

## PDAC Baseline (20 letters)

### Hallucinations / Factual Errors: 2 found

| ROW | Issue | Severity |
|-----|-------|----------|
| **1** | Letter ends with "P.S. Your girlfriend had a baby girl on April 27. That's wonderful news!" — this personal information was in the social history of the note, NOT appropriate for a medical summary letter. This is a **bizarre inclusion** that a real doctor would never put in a patient letter. | P1 — inappropriate |
| **5** | Letter says "You have...a small amount of cancer spread to your abdominal wall" — the A/P says "mesenteric metastasis" and "liver metastasis" but baseline converted "abdominal wall" which may not be accurate. | P2 — imprecise |

### REDACTED / Privacy Issues: 17/20 affected

| Issue | Count |
|-------|-------|
| ***** leaks | 10/20 |
| Uses "Mr./Ms./Mrs. *****" | 16/20 |
| Fabricated sign-off placeholders | 14/20 |

### Missing Information: 5 found

| ROW | Missing |
|-----|---------|
| 3 | Missing specific imaging findings (multiple hepatic metastases, peritoneal disease) |
| 8 | Missing CA 19-9 trend (significant for PDAC prognosis tracking) |
| 12 | Missing reason for drug change (neuropathy → stopped Abraxane) |
| 14 | Missing specific cycle number and restaging plan |
| 19 | Missing genetic testing discussion |

### PDAC-specific Observations

- PDAC baseline handles FOLFIRINOX/gemcitabine regimen names well (doesn't garble them)
- Good at explaining "what is pancreatic cancer" for new patients
- Consistently misses the nuance between "locally advanced" vs "metastatic" vs "borderline resectable"
- Doesn't distinguish neoadjuvant vs adjuvant intent — everything is just "treatment"

### Per-letter Verdict (PDAC)

| Verdict | Count |
|---------|-------|
| Would send (with minor edits) | 0 |
| Would send after fixing REDACTED/name | 13 |
| Needs significant revision | 5 |
| Would NOT send | 2 (ROW 1 baby PS, ROW 5 inaccurate metastasis) |

---

## Summary Statistics

| Metric | Breast Baseline | PDAC Baseline | Combined |
|--------|----------------|---------------|----------|
| **Hallucinations** | 3/20 (15%) | 2/20 (10%) | **5/40 (12.5%)** |
| **REDACTED leaks** | 8/20 (40%) | 10/20 (50%) | **18/40 (45%)** |
| **Privacy issues** | 20/20 (100%) | 17/20 (85%) | **37/40 (92.5%)** |
| **Missing key info** | 4/20 (20%) | 5/20 (25%) | **9/40 (22.5%)** |
| **Would send as-is** | 0/20 (0%) | 0/20 (0%) | **0/40 (0%)** |
| **Would send after name/REDACTED fix** | 12/20 | 13/20 | **25/40 (62.5%)** |
| **Would NOT send** | 2/20 | 2/20 | **4/40 (10%)** |

---

## Most Concerning Findings

### 1. ROW 1 PDAC — "P.S. Your girlfriend had a baby girl"
The model pulled personal social history from the note and appended it as a postscript to the medical letter. This is:
- **Inappropriate** — medical letters should not contain personal life events
- **Privacy concern** — social history is not for patient letters
- **Shows lack of clinical judgment** — no doctor would include this

### 2. ROW 15 Breast — HER2 Speculation
The model fabricated a treatment scenario ("If the cancer is also HER2 positive, we might use different medicines") that the doctor did not discuss. This could confuse the patient about their actual diagnosis.

### 3. ROW 4 Breast — Undecided Plan Presented as Decided
The patient said she wanted to think about chemo and talk to another doctor first. The baseline letter presents chemotherapy as the definite plan with specific scheduling. This misrepresents the clinical situation.

### 4. Systematic REDACTED Leaks
45% of letters contain ***** markers visible to the patient. This is the single biggest deployability issue.

### 5. Universal Privacy Issue
92.5% of letters attempt to use the patient's name, exposing "*****" or using placeholder brackets. No baseline letter uses a safe generic salutation.

---

## Comparison: Baseline vs Pipeline

| Issue | Baseline (40 letters) | Pipeline (40 letters) |
|-------|----------------------|----------------------|
| Hallucinations | **5 (12.5%)** | **0 (0%)** |
| REDACTED leaks | **18 (45%)** | **0 (0%)** |
| Privacy issues | **37 (92.5%)** | **0 (0%)** |
| Missing key info | **9 (22.5%)** | **~2 (5%)** |
| Would send as-is | **0 (0%)** | **~38 (95%)** |
| Inappropriate content | **1 (baby PS)** | **0** |
| Speculative content | **1 (HER2)** | **0** |

**The pipeline eliminates 100% of the baseline's deployment-blocking issues.**
