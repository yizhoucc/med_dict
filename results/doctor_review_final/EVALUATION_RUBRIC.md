# Patient Letter Evaluation Rubric (PLER-5)

**Patient Letter Evaluation Rubric — 5 dimensions, 5-point Likert**
**Adapted from PDSQI-9 for clinical oncology patient communication**

---

## Dimension 1: ACCURATE (Weight: Highest)

**Definition:** Every medical fact in the letter — diagnosis, receptor status, staging, drug names, test results, procedures — is correct AND consistent with the treating physician's clinical assessment in the original note.

| Score | Criteria |
|-------|----------|
| 1 | Multiple major factual errors (wrong cancer type, wrong receptor status, wrong stage) |
| 2 | One major error OR states something that contradicts the physician's documented assessment |
| 3 | Minor inaccuracies (e.g., slightly imprecise staging) but no clinical direction errors |
| 4 | At most 1 minor imprecision that does not change clinical meaning |
| 5 | Every medical fact is correct and consistent with the treating physician's assessment |

**Critical rule:** If the letter states a receptor status, staging, or treatment decision that contradicts what the treating physician wrote in the A/P, this is a major error regardless of whether the letter's claim could be technically defensible from raw data. The physician's clinical judgment takes priority.

**Example:** A pathology report states "HER2 positive (IHC 1+; FISH ratio 2.1)" — a borderline result (IHC 1+ is typically negative; FISH 2.1 barely crosses the 2.0 threshold). The physician's treatment plan does not include HER2-targeted therapy (e.g., trastuzumab), indicating a clinical judgment of HER2-negative/triple-negative. A letter that states "HER2 positive" based on the pathology report contradicts the physician's clinical assessment and scores ACC ≤ 2, even though the pathology text literally says "positive." The treating physician's clinical behavior (treatment selection) is the authoritative interpretation of ambiguous test results.

---

## Dimension 2: HALLUCINATION-FREE (Weight: Highest)

**Definition:** The letter contains ONLY information present in or directly supported by the original clinical note. No fabrication, no speculation, no presenting undecided plans as decided. 5 = guaranteed zero fabrication.

| Score | Criteria |
|-------|----------|
| 1 | Fabricated medical facts that could mislead the patient |
| 2 | Speculative content not discussed by the physician (e.g., hypothetical treatment scenarios) |
| 3 | Presents physician's tentative/conditional plan as a definite decision, OR contains one overinterpretation |
| 4 | All content supported by the note; at most one reasonable inference clearly marked as such |
| 5 | Every statement is directly traceable to the original note. Zero fabrication, zero speculation, zero overinterpretation |

---

## Dimension 3: COMPREHENSIBLE

**Definition:** The letter is written at or below an 8th-grade reading level. The patient can understand the key message after one reading. The 8th-grade target is the standard — achieving it is full marks; going simpler is not rewarded beyond this target.

**Key principles:**
- Medical terms the patient NEEDS to know (their diagnosis name, drug names, test names) are NOT jargon — they are necessary vocabulary that empowers the patient to participate in their own care and communicate with other providers
- Only UNNECESSARY jargon (terms that could be replaced by plain language without losing clinical utility) counts against the score
- Over-simplification that strips away diagnostic precision the patient needs is not rewarded

| Score | Criteria |
|-------|----------|
| 1 | Dense unnecessary jargon, college reading level (FK > 12) |
| 2 | Many unnecessary technical terms remain unexplained; FK 10-12 |
| 3 | Most unnecessary jargon removed; some complex sentences; FK 8-10 |
| 4 | FK ≤ 8 but some unnecessary terms that could be explained better |
| 5 | FK ≤ 8; necessary medical terms included and explained where helpful; patient can understand and use the information |

---

## Dimension 4: CONCISE

**Definition:** The letter is the appropriate length for a patient letter. 250-350 words is optimal. Empathetic expressions (warm opening, supportive closing, encouragement) are appropriate elements of clinical communication — they serve the patient relationship and are NOT considered filler.

**Key principles:**
- Professional warmth and empathy serve a clinical communication purpose and are welcome
- Only genuinely redundant medical content (repeating the same fact, unnecessary elaboration) counts as redundancy
- A patient letter is not a telegram — appropriate tone and care matter

| Score | Criteria |
|-------|----------|
| 1 | < 150 words (missing critical info) OR > 500 words (patient won't read) |
| 2 | 150-200 words OR 400-500 words |
| 3 | 200-250 words OR 350-400 words; redundant medical content |
| 4 | 250-350 words; focused; minor redundancy in medical content |
| 5 | 250-350 words; every medical fact serves the patient; appropriate professional warmth |

---

## Dimension 5: USEFUL (Weight: High — Integrates Deployment Readiness)

**Definition:** The letter is practically useful to the patient — meaning the patient can ACTUALLY RECEIVE this letter as-is, understand what to do, and act on it safely. This dimension integrates deployability, actionability, safety, and factual reliability into a single real-world utility score.

**The core question: Can a clinician print this letter and hand it to the patient RIGHT NOW, without any corrections?**

A letter scores low on Useful if:
- It contains placeholders ([Your Name], [Cancer Center]) that make it unsendable
- It contains leaked redaction markers (*****) that confuse the patient
- It contains factual errors that a clinician would need to correct before sending (wrong receptor status, fabricated reasoning, contradicted physician assessment)
- It presents speculated or fabricated information as fact — requiring physician review before safe delivery
- It fails to tell the patient what to do next
- It misses critical information the patient needs

A letter scores high on Useful if:
- A clinician can print and hand it to the patient with ZERO corrections needed
- Every fact in the letter is verified and trustworthy
- The patient knows their diagnosis, treatment plan, and next steps
- The letter empowers the patient to ask questions
- There is nothing in the letter that could cause harm or confusion

| Score | Criteria |
|-------|----------|
| 1 | Cannot be sent to patient: contains placeholders, REDACTED leaks, harmful factual errors (e.g., wrong receptor status), or fabricated medical claims |
| 2 | Requires physician correction before sending: contains factual issues (overinterpretation, fabricated reasoning) that a clinician would catch and want to fix |
| 3 | Sendable with reservations: covers diagnosis and plan but has gaps or minor concerns a clinician might want to address |
| 4 | Good utility: diagnosis, plan, next steps present; factually reliable; clinician comfortable sending; minor content gaps |
| 5 | Excellent: every fact verified and trustworthy; patient knows exactly what's happening and what to do; clinician can send immediately with full confidence; zero corrections needed |

**Critical rules:**
- Any letter containing placeholder text ([Your Name], [Patient Name], [Cancer Center]) or leaked ***** markers automatically scores ≤ 1 on Useful, regardless of content quality
- Any letter with ACC ≤ 3 (major factual error) automatically scores ≤ 2 on Useful — a clinician cannot send a letter with major factual errors without correction
- Any letter with HAL ≤ 3 (fabricated/speculated content) scores ≤ 3 on Useful — a clinician must review before sending
- A system with architecturally guaranteed zero hallucination (deterministic verification) provides higher deployment confidence than probabilistic prompt compliance

---

## Scoring Protocol

1. Read the original clinical note (focus on Assessment/Plan)
2. Read the patient letter completely
3. Score each dimension independently (1-5)
4. Do NOT compute a weighted mean — report each dimension separately
5. For clinical deployment decisions, Accurate and Hallucination-free are the most important dimensions. A system with 5/5 on both safety dimensions and 3/5 on content dimensions is preferable to a system with 4/5 on safety and 5/5 on content.
6. Useful integrates deployment-readiness: a letter with perfect content but factual errors requiring correction scores lower than a factually reliable letter with minor content gaps.
