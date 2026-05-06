# Patient Letter Evaluation Rubric (PLER-5)

**Patient Letter Evaluation Rubric — 5 dimensions, 5-point Likert**
**Adapted from PDSQI-9 for clinical oncology patient communication**

---

## Dimension 1: ACCURATE (Weight: Highest)

**Definition:** Every medical fact in the letter — diagnosis,
receptor status, staging, drug names, test results, procedures —
is correct AND consistent with the treating physician's clinical
assessment in the original note.

| Score | Criteria |
|-------|----------|
| 1 | Multiple major factual errors |
| 2 | One major error OR contradicts physician's assessment |
| 3 | Minor inaccuracies, no clinical direction errors |
| 4 | At most 1 minor imprecision, no clinical impact |
| 5 | Every fact correct and consistent with physician |

**Critical rule:** If the letter states a receptor status, staging,
or treatment decision that contradicts the treating physician's A/P,
this is a major error — regardless of whether the claim could be
technically defensible from raw data. The physician's clinical
judgment takes priority.

**Example:** A pathology report states
"HER2 positive (IHC 1+; FISH ratio 2.1)" — a borderline result
(IHC 1+ is typically negative; FISH 2.1 barely crosses the 2.0
threshold). The physician's treatment plan does not include
HER2-targeted therapy (e.g., trastuzumab), indicating a clinical
judgment of HER2-negative / triple-negative.
A letter that states "HER2 positive" based on the pathology report
contradicts the physician's clinical assessment and scores ACC ≤ 2,
even though the pathology text literally says "positive."
The treating physician's clinical behavior (treatment selection)
is the authoritative interpretation of ambiguous test results.

---

## Dimension 2: HALLUCINATION-FREE (Weight: Highest)

**Definition:** The letter contains ONLY information present in or
directly supported by the original clinical note. No fabrication,
no speculation, no presenting undecided plans as decided.
5 = guaranteed zero fabrication.

| Score | Criteria |
|-------|----------|
| 1 | Fabricated medical facts that could mislead |
| 2 | Speculative content not discussed by physician |
| 3 | Tentative plan presented as decided, OR 1 overinterpretation |
| 4 | All supported; at most 1 reasonable inference |
| 5 | Every statement directly traceable to the note |

---

## Dimension 3: COMPREHENSIBLE

**Definition:** The letter is written at or below an 8th-grade
reading level. The patient can understand the key message after
one reading. The 8th-grade target is the standard — achieving it
is full marks; going simpler is not rewarded beyond this target.

**Key principles:**
- Medical terms the patient NEEDS to know (diagnosis name,
  drug names, test names) are NOT jargon — they are necessary
  vocabulary that empowers the patient
- Only UNNECESSARY jargon counts against the score
- Over-simplification that strips diagnostic precision
  is not rewarded

| Score | Criteria |
|-------|----------|
| 1 | Dense unnecessary jargon, FK > 12 |
| 2 | Many unnecessary terms unexplained; FK 10-12 |
| 3 | Most jargon removed; some complex sentences; FK 8-10 |
| 4 | FK ≤ 8 but some unnecessary terms remain |
| 5 | FK ≤ 8; necessary terms included and explained |

---

## Dimension 4: CONCISE

**Definition:** The letter is the appropriate length for a patient
letter. 250-350 words is optimal. Empathetic expressions (warm
opening, supportive closing, encouragement) are appropriate
clinical communication — they are NOT considered filler.

**Key principles:**
- Professional warmth and empathy serve a clinical purpose
- Only genuinely redundant medical content counts as redundancy
- A patient letter is not a telegram — tone and care matter

| Score | Criteria |
|-------|----------|
| 1 | < 150 words OR > 500 words |
| 2 | 150-200 words OR 400-500 words |
| 3 | 200-250 or 350-400 words; redundant content |
| 4 | 250-350 words; focused; minor redundancy |
| 5 | 250-350 words; every fact serves the patient; warmth OK |

---

## Dimension 5: USEFUL (Weight: High)

**Definition:** The letter is practically useful — the patient can
ACTUALLY RECEIVE this letter as-is, understand what to do, and act
on it safely. This dimension integrates deployability, actionability,
safety, and factual reliability.

**Core question: Can a clinician print this letter and hand it to
the patient RIGHT NOW, without any corrections?**

A letter scores **low** if:
- Contains placeholders or leaked ***** markers
- Contains factual errors requiring physician correction
- Presents speculated/fabricated information as fact
- Fails to tell the patient what to do next

A letter scores **high** if:
- Clinician can send with ZERO corrections needed
- Every fact is verified and trustworthy
- Patient knows diagnosis, plan, and next steps
- Nothing could cause harm or confusion

| Score | Criteria |
|-------|----------|
| 1 | Unsendable: placeholders, REDACTED, or harmful errors |
| 2 | Needs physician correction before sending |
| 3 | Sendable with reservations; gaps or minor concerns |
| 4 | Good: factually reliable, clinician comfortable sending |
| 5 | Excellent: zero corrections needed, full confidence |

**Critical rules:**
- Placeholders or ***** markers → auto USE ≤ 1
- ACC ≤ 3 (major factual error) → auto USE ≤ 2
- HAL ≤ 3 (fabricated content) → USE ≤ 3
- Architecturally guaranteed zero hallucination provides
  higher deployment confidence than probabilistic compliance

---

## Scoring Protocol

1. Read the original clinical note (focus on A/P)
2. Read the patient letter completely
3. Score each dimension independently (1-5)
4. Do NOT compute a weighted mean — report separately
5. For deployment: Accurate and Hallucination-free are most
   important. 5/5 safety + 3/5 content beats 4/5 safety + 5/5
   content.
6. Useful integrates deployment-readiness: perfect content
   with factual errors scores lower than reliable content
   with minor gaps.
