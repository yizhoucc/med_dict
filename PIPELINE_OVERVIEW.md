# Pipeline Overview: Structured Extraction from Oncology Clinical Notes

## What This System Does

This pipeline takes **oncology clinical notes** (free-text physician notes from breast cancer patient visits) and extracts **structured, machine-readable information** covering diagnosis, treatment, medications, and plans.

---

## 1. Input: What Goes In

A single clinical note from the CORAL dataset — a de-identified breast cancer oncology visit note. A typical note is 2,000-8,000 words and includes:

```
History of Present Illness (HPI)
  - Patient demographics, cancer history, treatment timeline
Oncologic History (problem list)
  - Prior diagnoses, surgeries, pathology, imaging
Current Medications
Allergies
Review of Systems
Physical Exam
Lab Results / Imaging / Pathology reports
Assessment & Plan (A/P)
  - Physician's clinical impression and treatment plan
```

**Example input** (truncated):

> 61 year old female who was diagnosed with a left breast cancer in May and had
> primary surgery in late July. She had bilateral mastectomies and a left sentinel
> node procedure and had a 2.3 cm, node negative, triple negative breast cancer.
> She is healing well from surgery...
>
> Biopsy: INVASIVE DUCTAL CARCINOMA, GRADE 3, ER-, PR-, HER2-...
>
> Assessment/Plan:
> 1. Stage IIA left triple negative IDC, 2.3cm, node negative, s/p bilateral mastectomies
> 2. I recommend dd AC followed by Taxol for adjuvant chemotherapy
> 3. No indication for hormone blockade or radiation therapy
> 4. Lifestyle modifications: anti-inflammatory diet, exercise, stress reduction

---

## 2. Output: What Comes Out

A structured JSON with 15 sections, each containing specific clinical fields:

```json
{
  "Reason_for_Visit": {
    "Patient type": "New patient",
    "second opinion": "no",
    "in-person": "Televisit",
    "summary": "61 yo female with Stage IIA left TNBC, 2.3cm, node negative,
                post bilateral mastectomies, initial consult for adjuvant chemotherapy."
  },
  "Cancer_Diagnosis": {
    "Type_of_Cancer": "ER-/PR-/HER2- triple negative invasive ductal carcinoma",
    "Stage_of_Cancer": "Stage IIA",
    "Metastasis": "No",
    "Distant Metastasis": "No"
  },
  "Lab_Results": { "lab_summary": "No labs in note." },
  "Clinical_Findings": { "findings": "2.3 cm, node negative, TNBC..." },
  "Current_Medications": { "current_meds": "" },
  "Treatment_Changes": { "recent_changes": "", "supportive_meds": "" },
  "Treatment_Goals": { "goals_of_treatment": "curative" },
  "Response_Assessment": { "response_assessment": "Not yet on treatment." },
  "Medication_Plan": { "medication_plan": "dd AC followed by Taxol." },
  "Therapy_plan": { "therapy_plan": "Adjuvant chemo, no hormone blockade or RT." },
  "radiotherapy_plan": { "radiotherapy_plan": "None" },
  "Procedure_Plan": { "procedure_plan": "No procedures planned." },
  "Imaging_Plan": { "imaging_plan": "No imaging planned." },
  "Lab_Plan": { "lab_plan": "No labs planned." },
  "Genetic_Testing_Plan": { "genetic_testing_plan": "None planned." },
  "follow_up_next_visit": { "Next clinic visit": "Not specified" },
  "Advance_care_planning": { "Advance care": "Full code." },
  "Referral": { "Nutrition": "None", "Genetics": "None", "Specialty": "None" }
}
```

Each field also has **source attribution** — the exact quote from the note that supports the extracted value.

---

## 3. How It Works: Pipeline Architecture

```
                        CLINICAL NOTE (free text, 2000-8000 words)
                                        |
                                        v
                    +-----------------------------------+
                    |   Assessment/Plan Extraction      |
                    |   (regex first, LLM fallback)     |
                    +-----------------------------------+
                                        |
                        ________________|________________
                       |                                 |
                       v                                 v
        +-------------------------+       +---------------------------+
        |   PHASE 1: Independent  |       |   PLAN EXTRACTION         |
        |   Extraction (6 prompts)|       |   (from A/P only)         |
        |                         |       |   8 prompts:              |
        |   From FULL note:       |       |   - Medication Plan       |
        |   - Reason for Visit    |       |   - Therapy Plan          |
        |   - Cancer Diagnosis    |       |   - Radiotherapy Plan     |
        |   - Lab Results         |       |   - Procedure Plan        |
        |   - Clinical Findings   |       |   - Imaging Plan          |
        |   - Current Medications |       |   - Lab Plan              |
        |   - Treatment Changes   |       |   - Genetic Testing Plan  |
        |                         |       |   - Follow-up / Advance   |
        +-------------------------+       |     Care / Referral       |
                       |                  +---------------------------+
                       v                                 |
        +-------------------------+                      |
        |   PHASE 2: Contextual   |                      |
        |   Inference (2 prompts) |                      |
        |                         |                      |
        |   Receives Phase 1      |                      |
        |   results as context:   |                      |
        |   - Treatment Goals     |                      |
        |   - Response Assessment |                      |
        +-------------------------+                      |
                       |                                 |
                       +----------------+----------------+
                                        |
                                        v
                    +-----------------------------------+
                    |   5-GATE VERIFICATION             |
                    |   (per prompt, per field)          |
                    |                                    |
                    |   G1: JSON Format Fix              |
                    |   G2: Schema Key Validation        |
                    |   G3: Improve (specificity +       |
                    |       semantic alignment)          |
                    |   G4: Faithfulness                 |
                    |       ("keep unless clearly wrong")|
                    |   G5: Temporal Filter              |
                    |       (remove past from plans)     |
                    +-----------------------------------+
                                        |
                                        v
                    +-----------------------------------+
                    |   22 POST HOOKS                   |
                    |   (rule-based corrections)         |
                    |                                    |
                    |   Fix known LLM failure patterns   |
                    |   using regex + cross-field logic   |
                    |   (see POST Hook Reference below)  |
                    +-----------------------------------+
                                        |
                                        v
                    +-----------------------------------+
                    |   SOURCE ATTRIBUTION              |
                    |   Find evidence quotes for each   |
                    |   extracted value                  |
                    +-----------------------------------+
                                        |
                                        v
                              STRUCTURED OUTPUT (JSON)
                              + Attribution quotes
```

### Key Design Decisions

- **Phase 1 vs Phase 2 split**: Cancer type/stage must be extracted first so Treatment Goals can use it (Stage IV = palliative, early stage = curative).
- **Plan extraction from A/P only**: Treatment plans are in the Assessment/Plan section, not scattered throughout the note.
- **Gate 4 "keep unless clearly wrong"**: Conservative approach — we keep reasonable clinical inferences rather than deleting anything not explicitly quoted.
- **POST hooks instead of prompt changes**: Changing prompts alters the LLM's KV cache, causing unpredictable output shifts across all samples ("butterfly effect"). POST hooks apply deterministic corrections without affecting LLM behavior.

---

## 4. Quality: How Good Is It

### Dataset
- **61 breast cancer clinical notes** from CORAL (de-identified UCSF oncology notes)
- **Model**: Qwen2.5-32B-Instruct-AWQ (32 billion parameters, 4-bit quantized)

### Accuracy (v23, latest)

| Severity | Count | Definition |
|----------|-------|------------|
| P0 (critical) | **0** | Dangerous errors: hallucinated drugs, wrong dosages, fabricated diagnoses |
| P1 (significant) | **2** / 61 samples (3.3%) | Factually wrong but not dangerous: wrong Patient type, historical data used as current |
| P2 (minor) | **28** / 61 samples | Precision issues: Stage "Not mentioned" when inferable, "HR+" instead of "ER+", timing edge cases |
| OK (no issues) | **31** / 61 samples (51%) | All fields correct |

### Version History (improvement over time)

| Version | P0 | P1 | Key Changes |
|---------|----|----|-------------|
| v14 | dozens | - | Initial version |
| v17 | ~6 | - | TNBC fix, regional LN, referral |
| v20 | ~5 | - | Docetaxel FP, allergy drugs, eye drops |
| v22e | 0 | 7 | 19 POST hooks, full audit |
| **v23** | **0** | **2** | 22 POST hooks, 71% P1 reduction |

### What Gets Extracted Well
- Cancer type with receptor status (ER/PR/HER2) — including receptor discordance between primary and metastatic tumors
- Precise staging (e.g., "pT1c(m)N1(sn)M0", "Stage IIA")
- Current medications vs. stopped medications
- Treatment plans with specific regimens (e.g., "AC x4 -> Taxol x12 -> trastuzumab 1yr")
- Distant metastasis sites (e.g., "bone (left 7th rib and T6 pedicle)")

### Known Limitations
- Stage not extracted when the note doesn't explicitly state it (model is conservative)
- Redacted fields (*****/[REDACTED]) lose information
- Rare temporal confusion (listing historical imaging as current response)

---

## 4b. Patient Letter Generation

After extraction and attribution, the pipeline can generate a **plain-language patient letter** summarizing the visit. Each sentence in the letter is traced back to its source field and the original note.

### How It Works

```
Structured Keypoints (JSON)
        |
        v
+---------------------------+
| Pre-LLM Cleanup           |
| - Dedup overlapping fields |
| - TNM → plain stage       |
| - [REDACTED] → generic    |
| - Dr. [REDACTED] → "your  |
|   doctor"                  |
+---------------------------+
        |
        v
+---------------------------+
| LLM Letter Generation     |
| (Qwen2.5-32B-AWQ)         |
|                            |
| Prompt rules:              |
| - 8th-grade reading level  |
| - Explain medical terms    |
| - [source:field] per sent. |
| - No TNM, no raw data     |
| - ER+ = "grows in response|
|   to hormones"             |
+---------------------------+
        |
        v
+---------------------------+
| POST Checks               |
| - Strip residual [REDACTED]|
| - Detect TNM patterns     |
| - Detect repeated sentences|
+---------------------------+
        |
        v
+---------------------------+
| Parse & Trace              |
| - Extract [source:field]   |
| - Link to keypoint values  |
| - Link to note quotes      |
+---------------------------+
        |
        v
  Patient Letter + Traceability JSON
```

### Traceability Chain

Every sentence in the letter can be traced through two levels back to the original note:

```
Letter sentence
  └── [source:field_name] ──> keypoints[field] (extracted value)
                                    └── [attribution] ──> original note quote
```

**Example**:

| Letter sentence | Source field | Extraction value | Note quote |
|----------------|-------------|-----------------|------------|
| "Your cancer grows in response to hormones (estrogen)." | Type_of_Cancer | "ER+/PR+ invasive ductal carcinoma, HER2-" | "ER and PR positive and her 2 negative." |
| "The cancer has spread to your bones." | Distant Metastasis | "Yes (to bone)" | "metastatic breast cancer with bone mets" |

### Quality (v4, latest — 61 samples)

| Severity | Count | Definition |
|----------|-------|------------|
| P0 (hallucination) | **0** | Letter says something not in keypoints or note |
| P1 (significant) | **2** in 1 sample (1.6%) | Receptor explanation contradicts itself within same letter (Row 33) |
| P2 (minor) | **8** in 7 samples | Receptor status not explained; minor terminology |
| Perfect | **52 / 61 (85%)** | Zero issues |

### Improvement Over Iterations

| Version | P0 | P1 | Perfect | Key Fix |
|---------|----|----|---------|---------|
| v1 | 0 | 11 | 15% | Initial implementation |
| v2 | 0 | 8 | 54% | Dedup, TNM cleanup, prompt rules |
| v3 | 0 | 2 | 79% | Dr. [REDACTED] fix |
| **v4** | **0** | **2** | **85%** | Word-overlap dedup, response_assessment dedup |

### Example Letter Output

> Dear Patient,
>
> You visited us for your first appointment to discuss your breast cancer. Your cancer is ER+ and PR+ invasive ductal carcinoma, which means it grows in response to hormones (estrogen). It does not have extra HER2 protein.
>
> The cancer was originally at an early stage (Stage IIA) but has now spread to other parts of your body, including your lungs, liver, and ovaries. This is now considered advanced stage (Stage IV).
>
> You are not currently taking any medications for your cancer. The goal of treatment is to make you feel better and improve your quality of life.
>
> We will need to take a small sample (biopsy) of the mass in your right armpit to confirm the type of cancer and plan the best treatment. We will also do more tests, including a brain MRI and a bone scan.
>
> You are currently set to be treated as a full code, which means you want all possible life-saving measures if needed. We will see you again after we complete all the tests to decide on the best treatment plan.
>
> Sincerely, Your Care Team

### Usage

```bash
# Generate letters from existing extraction results (no re-extraction)
python run.py exp/full_qwen.yaml --letter-only results/run1/progress.json results/run2/progress.json

# Or enable letter generation in the full pipeline
# In exp config: extraction.letter: true
```

---

## 5. POST Hook Reference

POST hooks are rule-based corrections that run after LLM extraction. They fix known patterns the LLM gets wrong.

| # | Hook | What It Does |
|---|------|-------------|
| 1 | POST-VISIT-TYPE | Detects "video visit"/"telehealth" keywords to correct in-person field |
| 2 | POST-PATIENT-TYPE | Validates Patient type is "New patient" or "Follow up" |
| 3 | POST-PATIENT-TYPE-CC | Cross-checks Patient type against Chief Complaint section |
| 4 | POST-REFERRAL | Searches full note for referral patterns missed in A/P extraction |
| 5 | POST-GENETICS | Clears genetic test results incorrectly placed in Referral |
| 6 | POST-STAGE | Cross-validates Stage vs Metastasis for contradictions |
| 7 | POST-STAGE-REGIONAL | Corrects Stage IV when only regional LN (not distant) metastasis |
| 8 | POST-STAGE-VERIFY | Removes unsupported "Originally Stage X" claims |
| 9 | POST-STAGE-PLACEHOLDER | Cleans up [X]/[REDACTED] placeholders in Stage |
| 10 | POST-STAGE-ABBREV | Detects Stage abbreviations in A/P ("St IV", "st II/III") |
| 11 | POST-GOALS | Converts "adjuvant" to "curative" for non-metastatic cases |
| 12 | POST-DISTMET | Ensures Distant Metastasis field exists |
| 13 | POST-DISTMET-DEFAULT | Fills empty Distant Met with "No" when goals=curative + non-Stage IV |
| 14 | POST-SUPP | Removes oncologic drugs from supportive_meds |
| 15 | POST-MEDS-IV-CHECK | Detects active IV chemo in A/P when current_meds is empty |
| 16 | POST-MEDS-STOPPED | Removes stopped/discontinued drugs from current_meds |
| 17 | POST-ER-CHECK | Infers ER+ from hormonal medications (letrozole, tamoxifen) |
| 18 | POST-SELF-MANAGED | Detects physician-disapproved self-administered drugs (Mexico clinics) |
| 19 | POST-RECEPTOR-UPDATE | Updates receptor status from surgical pathology Addendum |
| 20 | POST-TYPE-VERIFY-TNBC | Overrides HER2+ when A/P confirms TNBC |
| 21 | POST-TYPE-UNCLEAR | Corrects fabricated receptor status when note says "biomarker results unclear" |
| 22 | POST-TYPE-HR-EXPAND | Expands "HR+" to specific "ER+/PR-" using note receptor details |

---

## 6. Files Quick Guide

| File | What It Is |
|------|-----------|
| `run.py` | Main pipeline code — extraction, letter generation, `--letter-only` mode |
| `ult.py` | Utility library: model inference, KV cache, JSON repair, gate logic |
| `letter_generation.py` | Patient letter generation: tagged output, traceability, POST checks |
| `source_attribution.py` | Per-field source attribution (LLM second-pass with KV cache) |
| `prompts/extraction.yaml` | LLM prompts for Phase 1+2 extraction |
| `prompts/plan_extraction.yaml` | LLM prompts for plan extraction from A/P |
| `prompts/letter_generation.yaml` | LLM prompt for patient letter (8th-grade, [source:field] tags) |
| `exp/full_qwen.yaml` | Experiment config (dataset, model, parameters) |
| `results/v23_audit_report.md` | Extraction quality audit: 61 samples, every P1/P2 documented |
| `results/letter_full_qwen_20260327_134953/review.md` | Letter generation quality audit: 61 samples reviewed |
| `CLAUDE.md` | Development rules and project conventions |
