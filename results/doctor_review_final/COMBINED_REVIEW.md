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

---

# Part I: Breast Cancer (20 Samples × 3 Systems)

# Breast Cancer — Sample 1

**Cancer Type:** ER-/PR-/HER2- (HER2 IHC 1; FISH ratio 2.1, but with HER2 sig/nuc 3.0 and Cen17 sig/nuc 1.5) grade 3 invasive ductal carcinoma

## Original Clinical Note

```
ID: ***** ***** is a 81 y.o. postmenopausal patient with a recent diagnosis of breast cancer, who presents in consultation to discuss treatment options and to establish care.    HPI: The patient last had a screening mammogram on 11/06/11 (at 76 years) which was notable for heterogeneously dense breast tissue, but no features of malignancy.  The patient felt a painful mass in her right breast in August 2016.  She presented to her PCP (Dr. *****) of 30+ years on 09/11/16 who noted a firm mobile mass (6 cm) in the right upper outer quadrant of her breast with dimpling of overlying skin.      A diagnostic mammogram on 09/17/16 demonstrated heterogeneously dense breast tissue and a round, indistinct, high density mass (4.1 x 3.3 cm) at the site of the palpable lump, which was in the upper outer right breast, posterior depth.  A targeted ultrasound at that time was notable for a hypoechoic irregular solid mass (2.7 x 2.9 x 3.8 cm) in the upper outer right breast, at the 10:00 position, 9 cm from the nipple.  She underwent a right breast fine needle aspiration at the 10:00 position on 09/17/16.  The pathology was consistent with adenocarcinoma with ER negative, PR negative, and HER2 negative (IHC 0; FISH ratio 1.7).    She established care with Dr. ***** ***** on 09/25/16.  She was offered breast conserving surgery, but preferred mastectomy in an effort to avoid radiation.  ***** underwent a right breast simple mastectomy and sentinel lymph node biopsy on 10/25/16 with Dr. *****.  The pathology was notable for 3.6 cm of grade 3 IDC.  The mitotic count was quite elevated at 74/10 hpf.  The final margins were negative.  Features suggestive of lymphovascular invasion were present.  No DCIS was present.  One of the 2 sentinel lymph nodes was positive for adenocarcinoma (1.2 cm, but no extranodal extension).  Biomarkers were notable for estrogen receptor negative, progesterone receptor negative, HER2 positive (IHC 1; FISH ratio 2.1, \n but with HER2 sig/nuc 3.0 and Cen17 sig/nuc 1.5), and variable Ki67 expression (<5% to 25-30%; average ~15%).      ***** developed dyspnea on the evening of 11/20/16 and presented to ***** ***** ***** in the context of her known congestive hart failure with left venrticular dysfunction (LVEF 25%).  She was managed with Lasix in the CDU and discharged earlier today.  She presents to clinic with her daughter (*****) and granddaughter (*****).  She feels much better today, back to her baseline with regards to energy and dyspnea.  She has no pain at the site of her breast surgery.      Past Medical History:   Diagnosis Date   \u0007 Cardiac pacemaker in situ    \u0007 Cataract    \u0007 CHF (congestive heart failure)    \u0007 CKD (chronic kidney disease)     Chronic kidney disease (CKD) stage G3b/A1, moderately decreased glomerular filtration rate (GFR) between 30-44 mL/min/1.73 square meter and albuminuria creatinine ratio less than 30 mg/g   \u0007 Coronary atherosclerosis of unspecified type of vessel, native or graft    \u0007 Diabetes mellitus     115-180s   \u0007 GERD (gastroesophageal reflux disease)    \u0007 Glaucoma     suspect   \u0007 HTN (hypertension)    \u0007 Hypertension    \u0007 Other and unspecified hyperlipidemia    \u0007 Pacemaker    \u0007 SBO (small bowel obstruction)     Partial sbo without history of previous abdominal surgeries. Unclear etiology. DDx includes small occult neoplasm vs focal enteritis. Adhesion less likely with lack of abdominal surgeries. EGD/Colo neg in October 2012   \u0007 Sick sinus syndrome        Past Surgical History:   Procedure Laterality Date   \u0007 BIOPSY / EXCISION BREAST     \u0007 CHOLECYSTECTOMY     \u0007 MASTECTOMY Right 10/25/2016   \u0007 PACEMAKER INSERTION  1998   \u0007 SENTINEL LYMPH NODE BIOPSY Right 10/25/2016        Family History   Problem Relation Age of Onset   \u0007 Stroke Mother    \u0007 Uterine cancer Mother 40     died at 43   \u0007 Stroke Father    \u0007 Diabetes Other    \u0007 Arthritis Other    \u0007 Stroke Sister 68     March 2013   \u0007 Stomach cancer Maternal \n Grandfather    \u0007 Uterine cancer Maternal Aunt 37     died at 39   \u0007 Blindness Neg Hx    \u0007 Amblyopia Neg Hx    \u0007 Cataracts Neg Hx    \u0007 Glaucoma Neg Hx    \u0007 Macular degen Neg Hx    \u0007 Retinal detachment Neg Hx    \u0007 Strabismus Neg Hx        Social History     Social History   \u0007 Marital status: Married     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A     Occupational History   \u0007 Not on file.     Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   \u0007 Alcohol use No   \u0007 Drug use: No   \u0007 Sexual activity: Not on file     Other Topics Concern   \u0007 Not on file     Social History Narrative    Married; husband still alive. Lives with her daughter.  Originally from *****.     The patient underwent menarche at 11 years of age.  She never took OCP.  She is G7P5.  She underwent menopause at 55.  She has no history of HRT.        Outpatient Encounter Prescriptions as of 11/21/2016   Medication Sig Dispense Refill   \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 1 tablet (500 mg total) by mouth every 6 (six) hours as needed (MILD PAIN).     \u0007 amLODIPine (NORVASC) 5 mg tablet Take 0.5 tablets (2.5 mg total) by mouth Daily. 90 tablet 3   \u0007 aspirin 81 mg EC tablet TAKE 1 TABLET BY MOUTH DAILY 90 tablet 3   \u0007 atorvastatin (LIPITOR) 40 mg tablet TAKE 1 TABLET BY MOUTH EVERY DAY 90 tablet 3   \u0007 bisoprolol (ZEBETA) 5 mg tablet TAKE 1/2 TABLET BY MOUTH DAILY 50 tablet 3   \u0007 blood glucose (BLOOD GLUCOSE) test strip Check blood sugar twice daily as directed. 180 each 3   \u0007 blood glucose monitoring kit Check blood sugar as directed 1 each 0   \u0007 brimonidine (ALPHAGAN) 0.1 % ophthalmic solution Place 1 drop into both eyes 2 (two) times daily. 10 mL 11   \u0007 calcium carbonate-vitamin D (OYSCO 500/D) 1,250 mg (500 mg elemental)-200 unit tablet TAKE 1 TABLET BY MOUTH EVERY DAY 90 tablet 3   \u0007 glipiZIDE (GLUCOTROL) 10 mg tablet TAKE 1 TABLET BY MOUTH TWICE DAILY BEFORE MEALS 180 \n tablet 1   \u0007 insulin glargine (LANTUS) 100 unit/mL injection INJECT 5 UNITS INTO THE SKIN EVERY NIGHT AT BEDTIME 10 mL 5   \u0007 insulin syringe-needle U-100 0.3 mL 29 SYRINGE USE DAILY AS DIRECTED 100 Syringe 3   \u0007 lancets lancets Use twice daily as directed. 180 each 3   \u0007 latanoprost (XALATAN) 0.005 % ophthalmic solution Place 1 drop into both eyes nightly at bedtime. 2.5 mL 11   \u0007 metFORMIN (GLUCOPHAGE) 1,000 mg tablet TAKE 1 TABLET BY MOUTH TWICE DAILY TAKE WITH MEALS 200 tablet 3   \u0007 valsartan (DIOVAN) 80 mg tablet TAKE 1 TABLET BY MOUTH DAILY 90 tablet 2   \u0007 ALPHAGAN P 0.1 % ophthalmic solution INSTILL 1 DROP IN BOTH EYES TWICE DAILY 5 mL 0   \u0007 docusate sodium (COLACE) 100 mg capsule Take 1 capsule (100 mg total) by mouth Twice a day. 60 capsule 0   \u0007 loratadine (CLARITIN) 10 mg tablet Take 1 tablet (10 mg total) by mouth Daily. 90 tablet 1   \u0007 mometasone (NASONEX) 50 mcg/actuation spray 2 sprays by Nasal route Daily. 17 g 6   \u0007 nitroGLYCERIN (NITROSTAT) 0.4 mg SL tablet Place 1 tablet (0.4 mg total) under the tongue every 5 (five) minutes as needed for Chest pain. 100 tablet 3   \u0007 olopatadine (PATANOL) 0.1 % ophthalmic solution Place 1 drop into both eyes daily as needed (1 drop as needed). 5 mL 6   \u0007 [DISCONTINUED] doxycycline (MONODOX) 100 mg capsule      \u0007 [DISCONTINUED] furosemide (LASIX) 40 mg tablet Take 0.5 tablets (20 mg total) by mouth Daily. 3 tablet 0   \u0007 [DISCONTINUED] oxyCODONE (ROXICODONE) 5 mg tablet Take 1 tablet (5 mg total) by mouth every 4 (four) hours as needed for Pain. 30 tablet 0     Facility-Administered Encounter Medications as of 11/21/2016   Medication Dose Route Frequency Provider Last Rate Last Dose   \u0007 [COMPLETED] furosemide (LASIX) injection 20 mg  20 mg Intravenous Once ***** *****, MD   20 mg at 11/20/16 1632   \u0007 [DISCONTINUED] 0.9 % sodium chloride flush injection syringe  3 mL Intravenous Q8H SCH ***** ***** *****, *****-C   3 mL at 11/20/16 2138   \u0007 [DISCONTINUED] 0.9 % sodium \n chloride flush injection syringe  3 mL Intravenous PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] acetaminophen (TYLENOL) tablet 500 mg  500 mg Oral Q6H PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] acetaminophen (TYLENOL) tablet 500 mg  500 mg Oral Q6H PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] amLODIPine (NORVASC) tablet 2.5 mg  2.5 mg Oral Daily (AM) ***** ***** *****, *****-C       \u0007 [DISCONTINUED] aspirin EC tablet 81 mg  81 mg Oral Daily (AM) ***** ***** *****, *****-C       \u0007 [DISCONTINUED] atorvastatin (LIPITOR) tablet 40 mg  40 mg Oral Q PM ***** ***** *****, *****-C   40 mg at 11/20/16 2128   \u0007 [DISCONTINUED] brimonidine (ALPHAGAN) 0.1 % ophthalmic solution 1 drop  1 drop Both Eyes BID ***** ***** *****, *****-C   1 drop at 11/20/16 2128   \u0007 [DISCONTINUED] dextrose 50% injection syringe 12.5 g  25 mL Intravenous Q15 Min PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] docusate sodium (COLACE) capsule 100 mg  100 mg Oral BID ***** ***** *****, *****-C   100 mg at 11/20/16 2128   \u0007 [DISCONTINUED] glipiZIDE (GLUCOTROL) tablet 10 mg  10 mg Oral Daily with Breakfast ***** ***** *****, *****-C       \u0007 [DISCONTINUED] glucose chewable tablet 20 g  20 g Oral Q15 Min PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] insulin aspart (NovoLOG) injection 100 units/mL pen  0-20 Units Subcutaneous TID ***** ***** ***** *****, *****-C   0 Units at 11/20/16 1843   \u0007 [DISCONTINUED] insulin aspart (NovoLOG) injection 100 units/mL pen  0-3 Units Subcutaneous Bedtime and early am ***** ***** *****, *****-C   0 Units at 11/20/16 2134   \u0007 [DISCONTINUED] insulin glargine (LANTUS, BASAGLAR) injection 100 units/mL pen  5 Units Subcutaneous Bedtime ***** ***** *****, *****-C   5 Units at 11/20/16 2136   \u0007 [DISCONTINUED] latanoprost (XALATAN) 0.005 % ophthalmic solution 1 drop  1 drop Both Eyes Bedtime ***** ***** *****, *****-C   1 drop at 11/20/16 2128   \u0007 [DISCONTINUED] loratadine (CLARITIN) tablet 10 mg  10 mg Oral Daily (AM) ***** ***** \n *****, *****-C       \u0007 [DISCONTINUED] metFORMIN (GLUCOPHAGE) tablet 1,000 mg  1,000 mg Oral BID ***** ***** ***** *****, *****-C   1,000 mg at 11/20/16 1845   \u0007 [DISCONTINUED] nitroGLYCERIN (NITROSTAT) SL tablet 0.4 mg  0.4 mg Sublingual Q5 Min PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] ondansetron (ZOFRAN) injection 4 mg  4 mg Intravenous Q8H PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] ondansetron (ZOFRAN) tablet 4 mg  4 mg Oral Q8H PRN ***** ***** *****, *****-C       \u0007 [DISCONTINUED] valsartan (DIOVAN) tablet 80 mg  80 mg Oral Daily (AM) ***** ***** *****, *****-C           No Known Allergies       Review of Systems:   General - some fatigue at baseline; stable weight   Eyes - no vision changes   HENT - stable hearing, no nasal discharge or sinus tenderness, no difficulty swallowing, no mouth sores, no sore throat   Breast - no breast pain  Respiratory - dyspnea at baseline; no cough, no wheezing   Cardiovascular - no chest pain, no palpitations, no lower extremity edema   GI - constipation; no heartburn, no nausea, no emesis, no abdominal pain, no diarrhea, no melena, no hematochezia   GU - no suprapubic pain, no dysuria, no vaginal bleeding   Musculoskeletal - joint stiffness; no muscle pain, no bone pain   Endocrine - no heat/cold intolerance, no hot flashes  Heme/Lymph - no easy bruising/bleeding, no lymphedema   Neurological - no headaches, no dizziness, no numbness/tingling, no falls   Psychological - anxious; no depression   Skin - no rashes/lesions, no diaphoresis       Physical Exam:   ECOG 2   Vital Signs - BP 118/67 | Pulse 87 | Temp 36.3 C (97.3 F) (Oral)  | Resp 16 | Ht 150 cm (4' 11.06\") Comment: 11/21/2016 ***** | Wt 53.5 kg (118 lb) | SpO2 98% | BMI 23.79 kg/m2   Constitutional - WDWN, NAD  Eyes - sclera anicteric, PERRL, EOMI  HENT - sinuses nontender, nasal mucosa intact, pharynx without erythema, stomatitis, or thrush   Lymph Nodes - no cervical, supraclavicular, or axillary lymphadenopathy. \n   Respiratory - resonant to percussion throughout, CTA bilaterally; no wheezes, rhonchi, or crackles  Cardiovascular - Normal heart rate, normal rhythm, no murmurs, no edema  Breast - right simple mastectomy; no mass in left breast  GI - Bowel sounds normal, soft; no tenderness, no distention, no HSM  Musculoskeletal - No tenderness over bones or joints.   Neurologic - Alert & oriented x 3, ambulates w/o difficulty, good strength throughout  Psychiatric - Mood stable; no HI, SI, hallucinations, paranoia, or delusions  Skin - warm without rashes or lesions       Studies:  Available labs, pathology, and imaging were reviewed and independently interpreted, as described above in the HPI.  Lab Results   Component Value Date    WBC Count 9.9 11/20/2016    Hemoglobin 9.6 (L) 11/20/2016    Hematocrit 28.9 (L) 11/20/2016    Platelet Count 222 11/20/2016     Lab Results   Component Value Date    Sodium, Serum / Plasma 133 (L) 11/20/2016    Potassium, Serum / Plasma 4.2 11/20/2016    Chloride, Serum / Plasma 102 11/20/2016    Carbon Dioxide, Total 22 11/20/2016    Urea Nitrogen, Serum / Plasma 18 11/20/2016    Creatinine 1.01 (H) 11/20/2016    Glucose, non-fasting 201 (H) 11/20/2016     Lab Results   Component Value Date    Calcium, total, Serum / Plasma 9.4 11/20/2016     Lab Results   Component Value Date    Aspartate transaminase 27 11/20/2016    Alanine transaminase 19 11/20/2016    Alkaline Phosphatase 60 11/20/2016    Bilirubin, Total 0.9 11/20/2016    Albumin, Serum / Plasma 3.0 (L) 06/01/2015     10/25/16 SURGICAL PATHOLOGY REPORT    ***** WITH ADDENDUM *****    Patient Name: *****, *****  *****. Rec.#: *****  DOB: 04/22/1935 (Age: 81)  Sex: Female  Accession #: *****-*****  Visit #: *****  Service Date: 10/25/2016  Received: 10/25/2016  Location: PPE  Client:*****   Physician(s): ***** *****. ***** ((*****) *****-*****)    FINAL PATHOLOGIC DIAGNOSIS    A. Right breast, simple mastectomy:   1. Invasive ductal carcinoma, SBR   grade \n 3, 3.6 cm, negative margins;  see comment.  2. Fibroadenomas.  3. Cystic dilatation of ducts and apocrine metaplasia.  4. Calcifications associated with apocrine metaplasia and benign ducts.    5. Unremarkable skin and nipple.    B. Right axillary sentinel lymph node, #1- ex vivo count 5000, biopsy:   Metastatic carcinoma in one lymph node, 1.2 cm, no extranodal extension  (May 19); see comment.    C. Right axillary palpable non-sentinel lymph node #1, biopsy: No tumor  in one lymph node (0/1).    D. Right new anterior lateral margin, excision: Benign fibroadipose  tissue, no carcinoma.    E. Right breast skin, excision: Benign skin, no carcinoma.    COMMENT:  Breast Tumor Synoptic Comment    - Laterality: Right.  - Tumor site: Upper outer quadrant.    - Position: 10 o'clock.  - Inv
```

---

## Letter A

Dear Patient,

I am writing to explain your recent test results and next steps in your care.

You have been diagnosed with breast cancer in your right breast. The type is called invasive ductal carcinoma, which means the cancer started in the milk ducts and has grown into nearby breast tissue. The tumor was about 3.6 cm in size. It is a grade 3 cancer, which means the cancer cells are growing faster than normal.

Your cancer was tested for certain markers. It is estrogen receptor (ER) negative and progesterone receptor (PR) negative, meaning it does not use these hormones to grow. It is HER2 positive, which means the cancer has a protein that can make it grow more quickly.

You had surgery on October 25, 2016, called a mastectomy (removal of the breast). The surgeon also checked nearby lymph nodes (small glands that help fight infection). Out of 2 lymph nodes, 1 had cancer cells. The edges of the removed tissue were clear, which means no cancer was seen at the margins.

This information suggests the cancer has spread to a nearby lymph node but was fully removed with surgery. Your care team will use this information to decide the stage and the best treatment plan.

Your doctors will now discuss additional treatment options with you. These may include systemic therapy, which is treatment that travels through the body to lower the chance of the cancer coming back.

You also have other health conditions, including heart failure (your heart does not pump as well as it should), and you recently had shortness of breath but are now feeling better. This is important when planning treatment.

Next steps:

Meet with your cancer team to review treatment options
Continue follow-up for your heart condition
Monitor your recovery from surgery

Please remember, you are not alone. Your care team is here to support you. Bring any questions to your next visit so we can review them together.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's Name],

Thank you for coming to our clinic with your family. We want to make sure you understand everything about your health and the next steps.

**Diagnosis:**
You have breast cancer. It is called "triple negative" because it doesn't have three things that some cancers have: estrogen receptors, progesterone receptors, and HER2 protein. This means we need to use different treatments.

**Stage:**
The cancer is at Stage II. This means it has grown a bit and has spread to one lymph node. But it hasn’t spread far.

**Treatment Plan:**
We talked about using chemotherapy to treat the cancer. Chemotherapy uses strong medicines to kill cancer cells. However, because you have other health problems like heart failure and diabetes, these medicines might make you very sick. So, we need to be careful.

**Next Steps:**
To find out if the cancer has spread to other parts of your body, we will do a special scan called a PET/CT. This will help us decide the best way to treat you.

**What to Watch For:**
Please let us know if you notice any new lumps or changes in how you feel. Also, keep an eye on your energy levels and breathing. If you feel very tired or have trouble breathing, call us right away.

**Medicines:**
You are taking several medicines to manage your other health conditions. Please continue to take them as prescribed by your doctor.

**Questions:**
It’s important to talk to your care team if you have any questions or concerns. We are here to help you understand everything and support you during this time.

Take care and we look forward to seeing you soon.

Sincerely,

[Your Name]
Medical Communication Specialist
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation regarding your recent diagnosis of breast cancer to discuss treatment options and establish care.
**What's new or changed since your last visit?**
You have been diagnosed with an early-stage (Stage I-II) triple negative breast cancer, which means the cancer does not have receptors for estrogen, progesterone, or the protein HER2. The cancer is a type called invasive ductal carcinoma, which started in the milk ducts. The tumor is 3.6 cm in size and was removed with a simple mastectomy. The edges of the removed tissue are clean, but one of the lymph nodes under your arm had cancer. No cancer was found in the skin or nipple. Your blood tests show low hemoglobin and hematocrit levels, and slightly high glucose and creatinine levels.
**What treatment or medication changes were made?**
No specific medication changes were made during this visit.
**What is the plan going forward?**
You will have a PET/CT scan to check if the cancer has spread to other parts of your body. After the results of the PET/CT, we will discuss the next steps in your treatment plan.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 2

**Cancer Type:** ER+/PR-/HER2- grade 1 infiltrating ductal carcinoma

## Original Clinical Note

```
                        UCSF ***** ***** FAMILY                        COMPREHENSIVE CANCER CENTER                      ***** ***** ***** ***** Care Center                             ***** ***** *****                           Second *****, ***** *****                    ***** *****, ***** *****-*****               Phone: (*****) *****-*****          Fax: (*****) *****-*****    01/24/2009    ***** *****, M.D.  ***** ***** *****, ***** *****  *****, *****  *****    ***** *****, M.D.  ***** ***** ***** #*****  *****, *****  *****    ***** *****, M.D.  ***** ***** *****  *****, *****  *****    ***** *****, M.D.  ***** ***** *****, ***** *****  *****, *****  *****    RE: *****, ***** *****  U#: *****  DATE OF SERVICE: 01/24/09    Dear *****:    I had the pleasure of seeing this patient in consultation regarding the  treatment of her locally recurrent breast cancer on January 09.  As you  know, she is a 73-year-old woman who initially had a right breast mass  removed in February 1994 with an axillary lymph node dissection.  That  surgical procedure revealed a 1 cm, grade 1, infiltrating ductal  carcinoma with clear surgical margins and 21 axillary lymph nodes were  negative for metastatic carcinoma.  S-phase was low at 3.6%.  The tumor  was found to be estrogen-receptor positive and progesterone-receptor  negative.  She received interstitial radiation, which comprised 4500  centigray over a 3 cm diameter.  She received tamoxifen from 1994 to  1996 and stopped due to concern regarding side effects.  At that time,  she was seeing Dr. ***** *****.  In January 1996, a node was  palpated in the left supraclavicular area.  A fine-needle aspiration was  performed and was unremarkable.  A dense area at the 12 o'clock position  in her right breast was tender in 2006 and was felt to be postsurgical  scarring.  This was noted to increase over time in size and density.  Initially, the workup included a PET scan in November 2005 \n that  revealed a 5 x 7 cm area of density consistent with inflammation.  It is  not clear whether she had a CT-guided biopsy at that time or not.  In  February 2008, she noted discomfort in the right anterior chest wall and  again thought that this area might be slightly larger.  She also thought  that she had a new right breast mass.  Workup of that breast mass was  unremarkable.  However, a breast MRI was performed on March 24 that  revealed a bulky irregular mass in the right 8 o'clock posterior breast,  which measured 1.9 x 1.5 cm with heterogenous enhancement.  Concordant  with an area of metabolic uptake on PET/CT scan, there was a second mass  abutting the pectoralis muscle with similar enhancing characteristics  measuring 1.7 x 1.2 cm.  A third nodule was seen along the right lateral  breast measuring 0.6 x 0.3 cm.  The left breast was unremarkable.  The  PET/CT scan had been performed on 02/22/2008, and was compared to a  PET/CT scan in December 2005.  This revealed a 2 cm right axillary  lymph node and a 1.4 cm hypermetabolic soft tissue nodule with an SUV of  4.1 in the right mid anterior chest wall deep in the subcutaneous fat.  A second right axillary lymph node was also noted, which was  hypermetabolic.  Subsequently, Dr. ***** performed a fine-needle  aspiration of the upper medial area, which was positive for carcinoma,  and a core biopsy of the right breast mass, which revealed a reactive  lymph node.    The patient states that at that time there was much discussion about  surgical management of this tumor.  It was felt that she would require  muscle resection and the patient initially decided to move forward with  this, but then did not have the surgery.  A follow-up PET/CT scan in  August 2008 revealed a 2 cm subcutaneous hypermetabolic focus in  the right upper parasternal level with an SUV of 7.4 and an ill-defined  1.3 cm hypermetabolic focus in the posterior segment of the medial \n right  lobe of the liver with an SUV of 4.8.  These findings were felt to be  consistent with metastatic disease.  In November 2008, a follow-up  PET/CT scan was performed.  This revealed a soft tissue mass inferior  and anterior to the right aspect of the manubrium sternum, measuring 1.9  x 1.5 cm, slightly larger than the scan in August and diffuse  heterogenous uptake throughout the liver without the focal lesions seen  in August.  Apparently a right axillary lymph node biopsy had been  performed in February 2008 and revealed benign pathology.  In any case,  she comes here today for a discussion of potential treatment options  having as yet not pursued treatment for her locally recurrent and  unresectable disease.    PAST MEDICAL HISTORY: Her past medical history is remarkable for  osteoarthritis in the left hand, trace emphysema by x-ray,  diverticulosis on colonoscopy, osteopenia, and a benign thyroid nodule.  Her most significant symptom is neuropathy of the right hand associated  with a right brachial plexopathy and loss of muscle mass in her  intrinsic hand muscles.  This history dates back to 1987 when a scalene  rib was removed on the right due to neuropathy thought to be causing a  right thoracic outlet syndrome.  Since that time, her right hand has  been weak.  It occasionally freezes on her and overall her symptoms have  gotten worse with increasing loss of muscle mass.  She has a mild distal  polyneuropathy with tingling and numbness in the feet and hands and  milder in the left hand.  This is improved some with CoQ10.  She notes  that she has short-term memory loss.  She was seen in ***** and Aging  Clinic in January 2008 and was diagnosed with symptoms consistent  with aging and depression.  A right frontal CVA was discovered on  imaging, which is thought embolic, and was very small in 2003.  She has  a history of ocular migraines associated with opaque vision the last 20  minutes.  \n They are episodic, but come in batches.  She has been to the  emergency room approximately five times for dizziness and  lightheadedness.  The last was in 2007.  She started episodic  lightheadedness and dizziness with moving around and falling spells,  which she finds to be disabling.  She has had a thyroid nodule for two  years thought to be benign.  She had surgery on her feet for hammertoes  in the early 1980s and other than being hospitalized for her  pregnancies, she has been hospitalized for migraines in the past as  well.  From her falling spells, she had hairline fractures in her wrists  and elbows about four years ago.  She still has pain in her wrists and  elbows.  She fractured her ankle with a fall more recently requiring a  brace after hitting a hole in the street.    GYN HISTORY: She underwent menarche at age 12.  Her menses stopped in  her mid-50s.  She took oral contraceptives at age 27 for six years and  hormone replacement therapy for 24 months at age 56.  She is G2, P2, and  was 22 at the birth of her first child.    FAMILY HISTORY: She has one maternal aunt with breast cancer at the age  of 70 and she has two maternal cousins with a history of cancer, one  small intestine and one prostate.    SOCIAL HISTORY: She is divorced and lives with her 42-year-old son who  is unemployed.  She worked with the military.  She does not drink  alcohol.  She smoked cigarettes for 75-pack-years stopping more than 12  months ago.    REVIEW OF SYSTEMS: She notes that she has lack of energy and pain in her  right hand.  She has some discomfort associated with the breast mass.  She has pain in her wrists and elbows from where she fractured them.  She is not eating well.  Her appetite has decreased.  She has lost five  pounds over the last few months.  She notes that she has tight breathing  when she has dizziness and this is episodic and it is associated with  panic-like attacks, flushing, and then dizziness.  If \n she is carrying  something heavy, she cannot go up the stairs without stopping.  She  sleeps well.  She has had fatigue since the time of her radiation in  1994, unable to work, which has led to her retirement.  She feels a  pulling sensation in her anterior chest wall down the center of her  right breast with occasional shooting pain and pain occasionally in her  left elbow, which she was told was tennis elbow.  The rest of her review  of systems is in the history of present illness.    PHYSICAL EXAMINATION:  GENERAL: On physical examination, she is a pleasant woman in no acute  distress who appears to have very good recall.  VITAL SIGNS: Her height is 5'6\", weight is 152 pounds, blood pressure  160/82, pulse 82, respirations 18, and her saturation is 98% on room  air.  SKIN: Her skin reveals no lesions.  HEENT: Her oropharynx is clear.  She is anicteric.  LYMPH NODES: She has no cervical or axillary adenopathy.  There are  soft, mobile nodes in the left supraclavicular space.  LUNGS: Her lungs are clear bilaterally to auscultation and percussion.  CARDIAC: Her cardiac exam is without murmur or gallop.  BREASTS: Her left breast is without masses.  Her right breast reveals a  well-healed incision in the 12 o'clock aspect of her right breast with  some density just below the incision.  She has a 2.5 x 2.5 cm, hard,  fixed mass just lateral to the sternal edge in the left superior chest  wall above her breast.  ABDOMEN: Her abdomen is soft and nontender without hepatosplenomegaly.  EXTREMITIES: Her extremities are without edema.  NEUROLOGIC: Neurologic exam is grossly nonfocal, although I did not  evaluate for brachial plexopathy in her right arm.    IMPRESSION: In summary, this patient is a complicated 73-year-old woman  with locally recurrent, unresectable, hormone-receptor positive breast  cancer, which was initially treated at diagnosis in 1994 with surgery  and interstitial radiation.  She has never taken hormone \n therapy for her  recurrence, although she was exposed to tamoxifen for a little under two  years back in 1994.  During our discussion, I went over in detail her  treatment options, including hormonal therapy with an aromatase  inhibitor on or off a clinical trial and certainly we would offer her  letrozole with alendronate as indicated for her bone density.  The  patient states that her bone density in February 2008 revealed a T-score  of -2.8 in her spine, which is consistent with osteoporosis, and -1.9 in  her femur consistent with osteopenia.  Therefore, I think it would be  appropriate even without bone mets to start her on zoledronic acid if  she agrees to start on a course with an aromatase inhibitor.  I also  pointed out to her that she is eligible to participate in CALGB 40503,  which would treat her with letrozole and then randomize to receive  bevacizumab or placebo.  In order to further evaluate her candidacy for  additional treatment, I obtained a PET/CT scan, which was performed on  January 16.  This showed increased FDG avidity within a right  parasternal chest wall nodule as well as in a lymph node along the  inferior right axillary chain.  The right parasternal soft tissue nodule  measured 1.9 cm and was intricately involved with the underlying chest  wall.  The lesion was adjacent to a soft tissue density deep to three  surgical clips in the right upper inner breast without FDG avidity  consistent with scar.  There was a right inferior axillary lymph node  measuring 1.9 cm with an SUV of 2 and an adjacent 0.5 cm lymph node.  Compared to the PET/CT scan first done in February 2008, the parasternal  nodule had increased in size, although interestingly the axillary nodes  which have not been proven to be involved with cancer, have decreased in  size.  Her abdomen shows a small, well-demarcated, low-attenuation  lesion in the liver consistent with cyst and her bones reveal no  abnormalities.    In \n summary, this patient is a 73-year-old woman with a remarkable  history dating back initially to a low-stage cancer in 1994 and now  presenting 14+ years later with a local regional recurrence that is not  resectable and has not yet been treated with hormone therapy due to the  patient's concerns about her treatment options.  I had a long talk with  her today and suggested that she consider treatment with an aromatase  inhibitor either on or off study and that we obtain laboratory studies  as well as PET/CT scan and MRI of her brain because of her falling  spells.  The MRI of her brain showed no evidence of malignancy either in  brain or bone.  She did have a right middle frontal gyrus  encephalomalacia and gliosis consistent with a prior insult.  We  obtained laboratory studies on the day of her visit as well.  These  revealed a normal CBC, creatinine, electrolytes, liver function tests,  and calcium.  Interestingly, a CA27-29 was only 15 and an LDH was normal  as well.  Based on all of this information, it appears that the patient  has a local regional recurrence of a strongly hormone-receptor positive  indolent breast cancer.  No other sites of disease appear to be present.  I think it would be most appropriate to treat this disease with hormone  therapy and follow it closely.  Given the limited radiation field that  she initially had, I think it would be possible to shrink the tumor,  potentially proceed with a surgical resection, and radiate this area in  order to at least approach the possibility of long-term disease control.  On my visit with her on January 09, I started her on zoledronic acid  because of her osteoporosis and need to start her on an aromatase  inhibitor.  She will return to my clinic on January 25 to begin  aromatase inhibitor therapy either alone or on CALGB 4503 randomized to  placebo or bevacizumab.    Thank you very much for allowing me to participate in the care of this  very \n interesting and lovely woman.  Please feel free to contact me if  you have any questions regarding my recommendations or evaluation.    *****,    ***** *****. *****, M.D.  CLINICAL PROFESSOR OF MEDICINE  UCSF COMPREHENSIVE CANCER CENTER  ***** ***** ***** BREAST CARE CENTER    EXTRA COPIES:    CARBON COPIES:    DICTATED BY:                 ***** *****. *****, MD *****                                 Electronically Signed by                               ***** *****. *****, MD *****/*****/***** *****:*****              ______________________________  ATTENDING PHYSICIAN:         ***** *****. *****, MD *****    D:    *****/*****/*****  *****:***** *****  T:    *****/*****/***** *****:***** ***** *****  *****#: *****  \n
```

---

## Letter A

Dear Patient,

I am writing to explain your current health condition and your care plan.

You have a history of breast cancer that was first treated in 1994. At that time, the cancer was small and had not spread to the lymph nodes.

Now, the cancer has come back in the same area, which is called a local recurrence (the cancer returns near where it first started). The current cancer is hormone-receptor positive, which means it uses natural hormones in your body to grow.

Recent scans show a mass (lump) in the chest wall near your right breast and a nearby lymph node. These areas show activity on imaging, which means cancer is likely present. The cancer cannot be safely removed with surgery right now.

The good news is that there is no clear spread to other organs based on your recent scans.

Treatment Plan

Your doctors recommend starting hormone therapy, also called endocrine therapy. This treatment lowers or blocks hormones that help the cancer grow. One option discussed is an aromatase inhibitor, which reduces estrogen levels in the body.

You may also have the option to join a clinical trial (a research study) that tests adding another medicine to this treatment.

Because your bone density is low (osteoporosis, meaning weaker bones), you have started a medicine to help strengthen your bones.

Next Steps
Start hormone therapy as planned
Follow up regularly with your cancer team
Repeat imaging tests to see how the tumor responds

If the tumor shrinks, surgery and radiation may be possible later.

You are not alone in this. Your care team will guide you at every step. Please bring any questions or concerns to your next visit so they can support you fully.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to see us. I want to share what we learned about your health and the next steps we will take.

**Diagnosis and Stage:**
You have breast cancer that has come back in the same area where it was before. This is called locally recurrent breast cancer. It is not able to be removed completely with surgery.

**Treatment Plan:**
We talked about using medicine to help control the cancer. One option is to use a type of drug called an aromatase inhibitor. This medicine stops the body from making too much estrogen, which can help slow down the growth of the cancer. We also discussed joining a research study to try a new treatment.

**Key Test Results:**
- **PET/CT Scan:** This test shows that the cancer is growing in your chest wall. It also shows that you have a small spot in your liver, but this is not cancer.
- **MRI of Brain:** This test shows no signs of cancer in your brain.
- **Blood Tests:** Your blood tests show that you are healthy and ready to start treatment.

**Next Steps:**
- You will start taking an aromatase inhibitor. This medicine will help control the cancer.
- We will keep watching the cancer closely with regular tests.
- We may also do more surgery or radiation later if needed.

**What to Watch For:**
- Keep an eye on any new pain or changes in your chest or breasts.
- Let us know if you have any new symptoms like feeling dizzy or having trouble breathing.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Doctor’s Name]  
[Doctor’s Title]  
UCSF Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consult regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You have a type of breast cancer called ER+/PR-/HER2- grade 1 infiltrating ductal carcinoma. The cancer has spread to your liver and chest wall. A recent PET/CT scan showed that a nodule in your chest wall has grown, while some lymph nodes have decreased in size. An MRI of your brain showed no signs of cancer.
**What treatment or medication changes were made?**
You started taking zoledronic acid to help with your osteoporosis and to prepare for starting an aromatase inhibitor.
**What is the plan going forward?**
You will start an aromatase inhibitor therapy either alone or on a clinical trial called CALGB 4503 randomized to placebo or bevacizumab on January 25. You will also have a PET/CT scan and an MRI of your brain to evaluate your falling spells. You will return to the clinic on January 25 to begin your new treatment.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 3

**Cancer Type:** ER-/PR-/HER2- (HER2 FISH neg, ratio 1.2) spindle cell metaplastic carcinoma

## Original Clinical Note

```
*****  60 yo F    Chief complaint: Patient with early stage breast cancer here to discuss neoadjuvant therapy    History of Present Illness:  ***** ***** is a 60 y.o. female with a recently diagnosed right sided invasive  spindle cell metaplastic carcinoma breast cancer.    The patient's history began in October 2016 when she felt a mass in her right breast, she sought care at ***** ***** with a breast surgeon 11/30/16, who felt a 3cm mass at 9:00 6cm FN and in the office an US showed a complex cystic mass, she performed a CNB in her clinic    11/30/16: US guided CNB performed: malignant spindle cell carcinoma of the breast, ER neg 0%, PR neg 0%, Her2 FISH neg (ratio 1.2)  12/17/16 Bilateral diagnostic mammo:  L: benign, R: 2.7 cm focal rounded nodular density in the lateral aspect of the right breast, BIRADS 0  12/17/16: US R breast: rounded 2.8 x 1.9 cm area of decreased echogenicity with somewhat variegated internal echotexture. The overall appearance is consistent with a probable postbiopsy hematoma  12/17/16: CXR - nml  12/30/16:MRI breasts performed in *****: irregular mass with heterogenous enhancement and central necrosis (with clip) in 9:00 6cmFN measures 2.9x2.5x3.3. Prominent nodes in Right axillary tail   Left breast with lobulated mass in upper outer quadrant, 7mm    01/15/17: Genetics at ***** - 126 InVitae panel sent  01/15/17: Visit with Dr *****  01/18/17: MRI at *****: 3.5 x 3.2 x 2.6 cm round mass with irregular, slightly spiculated margins and rim enhancement in the outer central right breast, middle depth. Approximately 15 mm anterior to the index mass there is a 12 mm irregular mass enhancing mass. In addition there is another similar 8 mm enhancing mass located 18 mm anterosuperior to the index lesion at the 10 o'clock position.  Taken together the index right breast mass and associated satellite masses span approximately 5.5 cm in AP extent.    IMPRESSION:   Right breast: Known biopsy proven \n malignancy (BI-RADS 6).  Left breast: Negative (BI-RADS 1).Recommendations:  1) Right outer central breast index malignancy measures 3.5 cm. Two small satellite masses are located 12-15 mm anterior/superior to the index mass.  2) Mildly prominent right axillary lymph nodes. Recommend further evaluation with ultrasound.    02/02/17: PETCT 1. Heterogenous hypermetabolic right breast mass consistent with patient's known breast carcinoma. Associated minimally hypermetabolic non-enlarged bilateral axillary lymph nodes, nonspecific and better evaluated on same day breast MRI. 2. Indeterminate right adrenal nodule measuring up to 2.3 cm. Further evaluation can be obtained with adrenal protocol CT/MRI.     02/02/17: UCSF review of outside path agreed spindle cell metaplastic carcinoma, did not repeat markers, no grade    02/03/17: R US with research cores and placement of clip, FNA axillary node with clip    02/03/17: FNA negative    02/11/17 MRI guided CNB for extent of disease      Cr 1.0 in 2011      The patient has been referred to discuss neoadjuvant treatment.          Past Medical History:   Diagnosis Date   \u0007 Abnormal liver enzymes 2012    Hyperbilirubinemia   \u0007 Common bile duct (CBD) stricture 2012    post-chole CBD stricture, presented with acholic stool and dark urine   \u0007 GERD (gastroesophageal reflux disease)     mild, tums occasionally   \u0007 HTN (hypertension)    \u0007 Plantar fasciitis         Gynecologic History:   Menarche was at age 12. Patient is gravida 0 para 0. Patient used oral contraceptives for 0 years. Menopause was at age 57 (3 years of hot flashes). Never used hormone replacement therapy           Past Surgical History:   Procedure Laterality Date   \u0007 CHOLECYSTECTOMY, LAPAROSCOPIC  2007   \u0007 Common bile duct stenting  2012    post- Lap chole, placed August 2011, removed 2012   \u0007 HYSTERECTOMY  1987    for a large fibroid         Current Medications           Current Outpatient \n Prescriptions   Medication Sig Dispense Refill   \u0007 clotrimazole-betamethasone (LOTRISONE) 1-0.05 % cream 2 (two) times a day.     \u0007 ibuprofen (ADVIL,MOTRIN) 800 mg tablet Take 800 mg by mouth once as needed for Pain.     \u0007 omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth As needed after meals.     \u0007 triamterene-hydrochlorothiazide (DYAZIDE) 37.5-25 mg capsule Take 1 capsule by mouth Daily.     \u0007 predniSONE (DELTASONE) 5 mg tablet        No current facility-administered medications for this visit.            Social History:  Social History           Social History   \u0007 Marital status: Married     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A           Occupational History   \u0007 ***** ***** ***** *****     *****         Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: None   \u0007 Alcohol use No   \u0007 Drug use: No   \u0007 Sexual activity: No          Other Topics Concern   \u0007 None         Social History Narrative    Recently divorced    Raised step son from age 3          family history includes Breast cancer in her sister; Cancer (age of onset: 40) in her father; Lung cancer in her paternal grandfather; Stomach cancer in her paternal grandmother.     Review of Systems:   Negative   ROS      Physical Examination:   General appearance - alert, well appearing, and in no distress  Mental status - alert, oriented to person, place, and time  Eyes - sclera anicteric  Neck - supple, no significant adenopathy  Lymphatics - no palpable lymphadenopathy, no hepatosplenomegaly  Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate, regular rhythm, normal S1, S2, no murmurs, rubs, clicks or gallops  Abdomen - soft, nontender, nondistended, no masses or organomegaly  lap chole scars  Breasts - left breast normal without mass, skin or nipple changes or axillary nodes, abnormal mass palpable in \n the right breast at 9:00 6cm FN that measures 4x4cm  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema, no clubbing or cyanosis  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted    Assessment / Plan:   60 year old female with ER Negative, PR Negative, HER2 Negative right sided breast cancer that is locally advanced, multifocal (possibly awaiting bippsy confirmation of extent of disease). Based on the tumor size and multifocal nature, the patient was referred to discuss neoadjuvant chemotherapy.     We discussed that the metaplastic aspect of her cancer is rare and portends a more aggressive biology - therefore we strongly encouraged neoadjuvant therapy and consideration of the trial with the potential benefit of receiving a promising experimental agent.    We discussed with ***** in great detail the natural history and treatment of locally advanced triple negative breast cancer.  We informed the patient that she currently has no evidence of metastatic disease and that we can therefore treat her with curative intent.  We reviewed the pathology report and discussed the multimodality approach of breast cancer treatment.  The patient understands that treatment will include chemotherapy, surgery, hormonal therapy, and potentially radiation therapy depending on her surgical choice and final pathology.  We discussed the advantages of offering chemotherapy in the neoadjuvant setting in order to assess response to treatment, and to increase the chances of breast conserving surgery.     We discussed the chemotherapy regimens that would be recommended as the standard of care, and discussed how these regimens would be different in the context of the I-SPY2 clinical trial.  We explained that in the context of her higher risk triple \n negative metaplastic disease we would typically recommend that she receive either 12 cycles of weekly paclitaxel followed by 4 cycles of doxorubicin plus cyclophosphamide (AC) every 2 weeks.  We explained that in the context of I-SPY2, patients are randomized to a control or to receive an experimental agent in addition to paclitaxel.   If she is randomized to an experimental therapy, then we will discuss that drug in more detail at that time.    The patient is very interested in the I-SPY2 clinical trial and has already consented. She has done the MRI, research biopsy, and Echo, mammaprint is pending and the echo looks good.  Her labs revealed an elevated creatinine and Alk phos, and she may have been dehydrated.  We will check these again    We explained that if her tumor has a Mammaprint low risk profile, she would not be eligible to participate in the therapeutic portion of the trial, but given the metaplastic aspect I am fairly certain it will be high risk.  If for some reason it is not, we will need to re-assess the diagnosis of metaplastic breast cancer and we could present her case to our multidisciplinary tumor Board.    We specifically discussed the typical and less common side-effects associated with AC/T, such as fatigue, neuropathy, skin and nail changes, myalgias, alopecia, myelosuppression, nausea.      She will benefit from a Medi-port, which was explained to her and to which she has agreed.  We will arrange for chemo teaching session, and while we could start therapy as early as next week the patient is needing a little more time to arrange things with work and would prefer to start the first week in February which we will plan for.     The patient asked about her genetic predisposition to this cancer - and we will discuss this further with her when we have the results of the gene panel that has been sent.     Pt visit was 55 min, with >50% of the time spent on counseling.     \n
```

---

## Letter A

Dear Patient,

I am writing to explain your diagnosis and treatment plan in a clear way.

You have breast cancer in your right breast. The type is called metaplastic carcinoma, which is a rare form of breast cancer. It can grow more aggressively than common types.

Your tumor is about 3.5–4 cm in size. There are also a few smaller nearby spots in the same breast. This is called multifocal disease, meaning more than one area of cancer in the breast.

Your cancer is triple negative. This means it does not use estrogen, progesterone, or HER2 to grow. Because of this, hormone therapy is not helpful, and chemotherapy is the main treatment.

The good news is that your scans show no clear spread to other parts of the body. This means your doctors can treat the cancer with the goal of cure.

Treatment Plan

Your doctors recommend starting neoadjuvant chemotherapy. This means chemotherapy is given before surgery to shrink the tumor. This can make surgery easier and may help save more breast tissue.

The plan includes:

Chemotherapy medicines given over several cycles
Then surgery to remove the cancer
Possible radiation therapy depending on results after surgery

You have also joined the I-SPY2 clinical trial. This is a research study that may give you a new treatment along with standard chemotherapy.

A port (small device under the skin) will be placed to make giving treatment easier.

Next Steps
Start chemotherapy soon
Attend a teaching session about treatment
Repeat blood tests to monitor your health
Review genetic test results when ready
You are not alone. Your care team is here to support you through each step. Please bring any questions to your visits so we can help you feel informed and confident.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

I hope this letter finds you well. I wanted to share some important information about your health in a way that is easy to understand.

**Diagnosis and Stage:**
You have a type of breast cancer called spindle cell metaplastic carcinoma. This is a rare kind of cancer that can grow quickly. It is found in your right breast and is considered locally advanced, which means it is larger and might be in more than one spot. However, it has not spread to other parts of your body.

**Treatment Plan:**
We recommend starting with a special type of treatment called neoadjuvant chemotherapy. This means you will take medicine before having surgery to remove the cancer. The goal is to make the cancer smaller so that surgery can be easier and more successful. 

The chemotherapy will help us see how well the cancer responds to the treatment. Depending on how you respond, we might also consider surgery, radiation, and possibly other treatments later.

**Key Test Results:**
- A biopsy showed that the cancer cells are not affected by hormones (ER and PR negative) and do not have too much of a protein called HER2.
- An MRI scan showed that the cancer is about 3.5 cm in size and there are a few smaller spots nearby.
- Your blood tests showed that your kidneys and liver are working well, but we will check them again soon.

**Next Steps:**
- You will start chemotherapy soon. We will put a special tube called a Medi-port in your chest to make it easier to give you the medicine.
- We will teach you more about the treatment and its possible side effects, like feeling tired, hair loss, and skin changes.
- You will have regular check-ups to see how the treatment is working.

**What to Watch For:**
- Let us know if you feel very tired, have pain, or notice any new lumps or changes in your skin.
- Keep track of any side effects and tell us about them during your visits.

**Important Reminders:**
- Please talk to your care team if you have any questions or concerns.
- We will keep you updated on your progress and adjust your treatment plan as needed.

Thank you for trusting us with your care. We are here to support you every step of the way.

Warm regards,

[Your Name]  
Medical Communication Specialist  
[Cancer Center Name]

Please remember to discuss any questions or concerns with your care team. They are here to help you through this journey.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consultation regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You have a rare type of breast cancer called spindle cell metaplastic carcinoma. The cancer does not have proteins called ER, PR, or HER2. The tumor is 3.5 x 3.2 x 2.6 cm and has two smaller masses nearby. Imaging shows a cancerous mass in your right breast, but no signs of the cancer spreading to other parts of your body. Your physical exam shows a mass in your right breast.
**What treatment or medication changes were made?**
You will start neoadjuvant chemotherapy, which may include 12 cycles of weekly paclitaxel followed by 4 cycles of doxorubicin plus cyclophosphamide (AC) every 2 weeks. If you join the I-SPY 2 clinical trial, you may also receive an experimental agent along with paclitaxel. You have agreed to have a Medi-port placed for easier chemotherapy administration.
**What is the plan going forward?**
You will start chemotherapy in the first week of February. You will have a follow-up after the initial chemotherapy session and further assessments based on how you respond to the treatment. You will also have genetic testing with a gene panel to learn more about your cancer. Your labs will be checked again to monitor your creatinine and Alk phos levels. Depending on your surgical choice and final pathology, you may need radiation therapy.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 4

**Cancer Type:** ER-/PR-/HER2- (TNBC) grade 3 invasive ductal carcinoma

## Original Clinical Note

```
ID: 71 year old female    CC:  HR low, HER2 negative cancer of the right breast    HPI:    ***** ***** ***** is a 71 y.o. female with R breast cancer here for discussion for a new patient visit.     10/04/2017- Bilateral Breast MRI .  Found 1.2 x 0.7 x 0.7 cm mass in the upper outer right breast.  A 2nd mass is seen posterior and lateral to the index mass measuring 0.6 x 0.6 x 0.7 cm. A 3rd 9 mm mass is also seen posterior and superior to the larger mass.   10/04/2017 R Breast Ultrasound: At 10 o'clock 7 cm from the nipple a 0.7 x 0.9 x 0.9 cm mass. A mass is also seen at 10 o'clock 10 cm from the nipple measuring 1 x 0.5 x 0.6 cm.   10/11/2017-   Core needle biopsy:  #1mass at 10o'clock 10 cm from nipple x 0.6 cm grade 3 ER 1% PR negative Her2 1+ IHC/FISH negative IDC with a Ki67 of 70%.   Biopsy of the mass at 10 o'clock 7 cm from the nipple 0.9cm of IDC showed a grade 3 ER negative PR 1% Her2 0IHC/FISH negative with a Ki67 of 30-40%.   10/19/2017- Initial consult with surgery team  12/27/2017- Follow up consult with surgery team  01/10/2018- PET CT Limited Whole Body- right lateral breast mass measures 2.7 x 1.7 cm compatible with recurrent disease. No evidence of hypermetabolic metastatic disease    ROS  Constitutional: Negative for fever, chills, diaphoresis, appetite change, fatigue and unexpected weight change.  + chronic headaches  HENT: Negative for hearing loss, nosebleeds, congestion, sore throat, mouth sores, neck pain, sinus pressure, or tinnitus.    Eyes: Negative for photophobia, pain, discharge, redness, itching and visual disturbance.   Respiratory: Negative for apnea, chest tightness, shortness of breath, wheezing. + allergic cough  Cardiovascular: Negative for chest pain, palpitations and leg swelling.   Gastrointestinal: Negative for nausea, vomiting, abdominal pain, diarrhea, constipation, blood in stool, or abdominal distention.   Genitourinary: Negative for dysuria, urgency, frequency, hematuria, vaginal \n bleeding or discharge or difficulty urinating.   Musculoskeletal: Negative for myalgias,  joint swelling, or arthralgias. + chronic back and shoulder pain x 2 years.  Skin: Negative for itching, rash  Breast: + tenderness at biopsy site and around axilla  Neurological: Negative for neuropathy, dizziness, tremors, seizures, syncope, speech difficulty, weakness, light-headedness, numbness or  headaches.   Hematological: Negative for adenopathy. Does not bruise/bleed easily.   Psychiatric/Behavioral: Negative for suicidal ideas, sleep disturbance, decreased concentration. + anxiety r/t disease and potential for treatment complications.    MEDICATIONS  Current Outpatient Prescriptions   Medication Sig Dispense Refill   \u0007 acetaminophen (TYLENOL) 325 mg tablet Take 1 tablet (325 mg total) by mouth every 6 (six) hours as needed for Pain. Dose, strength, route or frequency not available. 180 tablet 3   \u0007 alendronate (FOSAMAX) 70 mg tablet TAKE 1 TABLET BY MOUTH EVERY 7 DAYS 12 tablet 3   \u0007 ascorbate calcium (VITAMIN C ORAL) Take by mouth.     \u0007 baclofen (LIORESAL) 10 mg tablet Take 1 tablet (10 mg total) by mouth nightly as needed (Muscle pain, use at night only). 90 tablet 3   \u0007 benzonatate (TESSALON) 100 mg capsule Take 1 capsule (100 mg total) by mouth 3 (three) times daily as needed for Cough. 90 capsule 3   \u0007 clotrimazole (LOTRIMIN) 1 % cream APPLY EXTERNALLY TO THE AFFECTED AREA TWICE DAILY 60 g 0   \u0007 cyanocobalamin, vitamin B-12, (VITAMIN B-12 ORAL) Take by mouth.     \u0007 dextran 70-hypromellose (NATURAL BALANCE TEARS) 0.1-0.3 % DROPSOLN ophthalmic solution Place 2 drops into both eyes 2 (two) times daily. 30 mL 3   \u0007 ergocalciferol, vitamin D2, (VITAMIN D ORAL) Take by mouth.     \u0007 loratadine (CLARITIN) 10 mg tablet Take 1 tablet (10 mg total) by mouth Daily. 90 tablet 3   \u0007 melatonin 3 mg TAB tablet Take 1 tablet (3 mg total) by mouth nightly at bedtime. 90 tablet 3   \u0007 mometasone (NASONEX) 50 mcg/actuation spray 2 \n sprays by Nasal route Daily. 17 g 1   \u0007 multivitamin tablet Take 1 tablet by mouth Daily.       \u0007 omeprazole (PRILOSEC) 20 mg capsule TAKE 1 CAPSULE BY MOUTH DAILY 90 capsule 1   \u0007 promethazine-dextromethorphan 6.25-15 mg/5 mL syrup Take 5 mLs by mouth 4 (four) times daily as needed for Cough. 473 mL 3   \u0007 blood pressure monitor KIT Please check your blood pressure daily and record (Patient not taking: Reported on 01/24/2018) 1 kit 0   \u0007 menthol (BENGAY COLD THERAPY) 5 % GEL Apply 1 Application topically 2 (two) times daily. (Patient not taking: Reported on 01/24/2018) 85 g prn   \u0007 VENTOLIN HFA 90 mcg/actuation metered dose inhaler INHALE 1 PUFF INTO THE LUNGS EVERY 4 HOURS AS NEEDED FOR WHEEZING OR SHORTNESS OF BREATH (Patient not taking: Reported on 10/24/2017) 54 g 0     No current facility-administered medications for this visit.          ALLERGIES  Allergies/Contraindications   Allergen Reactions   \u0007 Iodinated Contrast- Oral And Iv Dye Itching and Rash     Patient takes pre-meds prior to CT Scans. She will continue to need the pre-meds for future scans   \u0007 Contrast [Gadolinium-Containing Contrast Media] Itching     Patient not sure if she is allergic to GAD.   Will add to allergy list: She is allergic to NON-Ionic Contrast (CT,etc.) She takes pre - meds for this.    \u0007 Hydrocodone-Acetaminophen Nausea Only   \u0007 Penicillins Unknown   \u0007 Pollen Extracts      Runny nose and dizziness           PAST, FAMILY, and SOCIAL HISTORY  Past Medical History:   Diagnosis Date   \u0007 Cataract    \u0007 Elevated CA-125    \u0007 GERD (gastroesophageal reflux disease)    \u0007 Kidney stones    \u0007 Metastatic papillary carcinoma to lymph node     involving left pelvic lymph nodes   \u0007 NSVD (normal spontaneous vaginal delivery)     x2   \u0007 Ovarian cancer 1998    recurrent, stage IIIC; with platinum-sensitive disease       Family History   Problem Relation Age of Onset   \u0007 Cancer Mother 35     breast cancer   \u0007 Hypertension Father    \u0007 Stroke Father 88   \u0007 \n Diabetes Other    \u0007 Cancer Sister      liver cancer   \u0007 Diabetes Brother    \u0007 Hypertension Brother    \u0007 Glaucoma Brother    \u0007 Hypertension Sister    \u0007 Diabetes Sister    \u0007 Glaucoma Sister    \u0007 Cancer Sister 62     Breast cancer, bilateral mastectomy   \u0007 Amblyopia Neg Hx    \u0007 Blindness Neg Hx    \u0007 Cataracts Neg Hx    \u0007 Keratoconus Neg Hx    \u0007 Macular degen Neg Hx    \u0007 Retinal detachment Neg Hx    \u0007 Strabismus Neg Hx    \u0007 Anesth problems Neg Hx    \u0007 Bleeding disorder Neg Hx    \u0007 Anesthesia problems Neg Hx    \u0007 Autoimmune disease Neg Hx    \u0007 Congenital Anomalies Neg Hx    \u0007 Hyperthyroidism Neg Hx    \u0007 Hypothyroidism Neg Hx    \u0007 Melanoma Neg Hx    \u0007 Neurofibromatosis Neg Hx    \u0007 Ptosis Neg Hx        Social History     Social History   \u0007 Marital status: Single     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A     Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   \u0007 Alcohol use No   \u0007 Drug use: No   \u0007 Sexual activity: Not on file     Other Topics Concern   \u0007 Not on file     Social History Narrative    2 sons 51, 46 yrs old, 2 miscarriages, 2 abortion    Divorced    Living alone, lives in ***** *****, ***** *****.     2 sons live in ***** *****       PHYSICAL EXAM  ECOG 0  Vital Signs:  BP 127/66  | Pulse 69  | Temp 36.3 C (97.4 F) (Oral)  | Resp 16  | Ht 158 cm (5' 2.21\") Comment: 01/24/2018 @ ***** | Wt 59.1 kg (130 lb 6.4 oz)  | LMP 01/09/1998  | SpO2 98%  | Breastfeeding? No  | BMI 23.69 kg/m     Constitutional:  WDWN, NAD.  Skin: warm, moist. No rashes or lesions.  Eyes: sclera anicteric, PERRL, EOMI.   HENT: sinuses nontender, nasal mucosa intact, pharynx without erythema, no thrush.   Lymph Nodes: no cervical, supraclavicular, or axillary lymphadenopathy.    Respiratory: BL lungs CTA, resonant to percussion, no wheezes, rhonchi, or crackles  Cardiovascular: normal heart rate, normal rhythm, no murmurs, no edema  Breast:: firm palpable \n nodule felt at 11 o'clock between axilla and areola in R breast.   GI: bowel sounds normal, soft, nontender, no distention, no organomegaly   Musculoskeletal: no point tenderness over bones or joints   Neurologic: alert & oriented x 3, no focal motor or sensory deficits  Psychiatric: mood stable    RESULTS  I have personally reviewed and interpreted the following:     PETCT Limited Whole Body Vertex to Mid Thigh 01/10/2018  COMPARISON (including at least the prior year or since last change in therapy):   08/18/2014    CLINICAL HISTORY:   71 years old Female with history of BRCA1 genetic carrier with a history of ovarian cancer and triple negative right breast IDC with new masses on mammogram 12/22/2017.    FINDINGS:  Lymph nodes: No hypermetabolic lymphadenopathy.  Lungs: No metastases.  Abdominal and pelvic organs: No metastases.  Bones: No metastases.  Other: Right lateral breast dumbbell shaped enhancing hypermetabolic mass measures 2.7 x 1.7 cm (SUV Max 10.5 for largest component).    Non-oncologic findings:  Brain:   Unremarkable, symmetric, FDG uptake is seen throughout the cortical gray matter, basal ganglia and the cerebellum.  No mass effect.  While these images appear within normal limits, MRI is recommended to rule out intracranial and/or skull base metastases if clinically indicated.  Neck:   Otherwise unremarkable.  Chest:   Unchanged right upper lobe pleural-based mass measuring 2.4 cm x 1.0 cm, dating back to 08/18/2014, without significant FDG avidity. Unchanged left upper and lower lobe pulmonary nodules measuring up to 6 mm in the left lower lobe (series 3, image 121 and image 32).  Abdomen/Pelvis:   Otherwise unremarkable.  Musculoskeletal:   Scattered bone islands in the pelvis are unchanged compared to 08/18/2014. Multilevel degenerative changes of the cervical and lumbar spine. Unchanged 3.4 cm lipomatous lesion in the right tensor fascia lata. Midline abdominal staples.    IMPRESSION: \n   Hypermetabolic  (FDG avid) right lateral breast mass measures 2.7 x 1.7 cm compatible with recurrent disease. No evidence of hypermetabolic metastatic disease.     Right breast diagnostic mammography, Limited R breast US 12/22/2017  Comparison:  Prior examinations dated 10/11/2017, 10/11/2017, 10/04/2017 (MRI) and 01/23/2017 were reviewed for comparison.    Mammographic Findings:  There are scattered areas of fibroglandular density.  BB marker on the upper outer right breast indicating patient-palpated lump corresponds to two masses, both biopsy-proven IDC (ribbon clip and heart clip). The irregular mass containing the ribbon clip now measures approximately 14 mm, previously 11 mm, and the more posterior obscured mass containing the heart clip now measures approximately 15 mm, previously 9 mm.  BB marker on the lower central/slightly outer right breast indicates second patient-palpated lump, for which there is no mammographic correlate.  Ultrasound Findings:  There is no sonographic abnormality (no correlate) to patient-palpated lump in the lower central/slightly outer right breast, at approximately 7 o'clock, 4 cm from nipple.  Limited ultrasound of the right axilla demonstrates lymph nodes with normal cortical thickness and morphology.    IMPRESSION:   Right breast: Known biopsy proven malignancy (BI-RADS 6).    Recommendations:  1. Recommend continued surgical/oncologic management of two-site biopsy-proven right breast malignancy. Of note, right axillary lymph nodes appear sonographically normal at today's examination.    2. Recommend clinical follow up for patient-palpated lump in the lower central/slightly outer right breast, for which there is no mammographic nor sonographic correlate. If indicated, palpation-guided FNA could be pursued.    Bilateral MRI of Breasts 10/04/2017  Comparison:  Prior examinations dated 01/23/2017 (mammogram), 07/08/2016 and 11/10/2015 were reviewed for comparison.  MRI \n Findings:  This bilateral breast MRI exam demonstrates minimum background enhancement.  There are scattered areas of fibroglandular tissue.    RIGHT Breast:  There is a new 12 x 7 x 7 mm (ML x AP x CC) irregularly shaped mass with spiculated margins in the upper outer quadrant of the right breast. Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics.  This lesion is best seen on axial postcontrast image # 90 and sagittal reformation #341.  More posteriorly and laterally in the right breast, there is a second new 6 x 6 x 7 mm (ML x AP x CC) irregularly shaped mass with irregular margins. Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics.  This lesion is best seen on image # 90 and sagittal reformation #363.    LEFT Breast:  The previously described area of non-mass enhancement with linear distribution and clumped internal enhancement in the lower outer left breast is significantly less conspicuous than on MRI dated 07/08/2016, and can no longer be accurately measured. No new abnormal areas of enhancement or other MRI features of malignancy are identified.    IMPRESSION:   Right breast: Suspicious (BI-RADS 4).  Left breast: Benign (BI-RADS 2).    Recommendations:  1. Two new irregular masses demonstrating rapid enhancement with washout are noted in the upper outer quadrant of the right breast. Recommend diagnostic right breast mammogram with targeted ultrasound for further evaluation and biopsy planning, with possible stereotactic or ultrasound-guided biopsy to follow. If no correlate is identified by mammogram or ultrasound, then recommend MR-guided core biopsy.    2. Significant interval decrease in conspicuity of the previously described area of non-mass enhancement in the lower outer quadrant of the left breast, compatible with a benign etiology. Annual mammography in addition to MRI is recommended in patients undergoing \n high-risk screening.        ASSESSMENT & PLAN  ***** ***** ***** is a 71 y.o. female with R breast cancer here for discussion for a new patient visit.       Breast Cancer  - Latest PETCT demonstrates growth of R breast lesion from 1.2 x 0.7 to 2.7 x 1.7 cm  - Patient unsure about starting chemo as we recommended today she would like to speak to Dr. ***** first.     Plan:  Port placement ASAP  Chemo teach ASAP for taxol x 12 weekly  Can start chemo as early as next Wednesday, but the following week also ok. She will need clinic visits with cycles 1, 2, 5, 8, and 12.        I have reviewed and updated the patient's past medical history, medications, allergies, family and social histories. I spent a total of 60 minutes face-to-face with the patient and 45 minutes of that time was spent counseling regarding the treatment plan, symptoms and therapeutic options. Patient expresses understanding and agrees with the plan.          \n
```

---

## Letter A

Dear Patient,

I am writing to explain your diagnosis and next steps in your care.

You have breast cancer in your right breast. The type is called invasive ductal carcinoma, which means the cancer started in the milk ducts and has grown into nearby tissue.

Your cancer is hormone receptor low/negative and HER2 negative. This means the cancer does not strongly use hormones or HER2 protein to grow.

Imaging tests showed more than one tumor in the right breast. The main tumor has grown to about 2.7 cm, and there are other smaller areas nearby.

The good news is that your scan shows no spread to other parts of the body.

Treatment Plan

Your doctors recommend starting chemotherapy, which is medicine that travels through the body to kill cancer cells.

The plan is:

Weekly chemotherapy (Taxol) for about 12 treatments
Placement of a port (a small device under the skin) to make treatment easier
Regular clinic visits during treatment

Chemotherapy is recommended because your cancer has grown and involves more than one area of the breast.

Next Steps
Schedule port placement soon
Attend a chemotherapy teaching session
Decide on the exact start date for treatment
Follow up with your oncology team regularly
It is normal to feel unsure about starting treatment. Please take time to speak with your doctors and ask any questions you have.

You are not alone. Your care team is here to support you every step of the way. Please bring any concerns to your next visit so we can help you feel confident and cared for.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,

Thank you for coming to our cancer center. We want to make sure you understand everything about your health and treatment plan.

**Diagnosis and Stage:**
You have breast cancer in your right breast. The cancer has grown from a small size to a bigger one. We call this "recurrent" because it came back after being treated before.

**Treatment Plan:**
We recommend chemotherapy to help fight the cancer. Chemotherapy uses strong medicines to kill cancer cells. We will place a special tube called a "port" to give you the medicine. The first treatment can start as early as next Wednesday or the following week.

**Key Test Results:**
Tests show that the cancer in your right breast has grown. We did not find any cancer in other parts of your body.

**Next Steps:**
You will come to the clinic for check-ups during your treatment. You will need to visit us for five different times during your treatment.

**What to Watch For:**
Please let us know if you feel very tired, have a fever, or notice any new lumps or pain. Also, tell us if you have any trouble breathing or chest pain.

**Medications:**
You take several medicines to manage pain, allergies, and other conditions. Make sure to take them as directed.

**Questions:**
If you have any questions or concerns, please talk to your doctor or nurse. They are here to help you understand everything about your treatment.

We hope this helps you understand your situation better. Remember, you are not alone in this journey.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new patient visit regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You have a type of breast cancer called triple-negative (TNBC) grade 3 invasive ductal carcinoma, which means the cancer started in the milk ducts and does not have proteins that respond to hormones. The cancer is at stage III. The latest imaging shows that a lump in your right breast has grown from 1.2 x 0.7 cm to 2.7 x 1.7 cm. No cancer has spread to other parts of your body. During the physical exam, a firm lump was found in your right breast.
**What treatment or medication changes were made?**
A port will be placed soon to make chemotherapy easier. You will start chemotherapy with Taxol, which will happen every week for 12 weeks.
**What is the plan going forward?**
You will have a port placed soon to prepare for chemotherapy. You will start chemotherapy with Taxol as early as next Wednesday or the following week. You will need to visit the clinic for cycles 1, 2, 5, 8, and 12. You will have a mammogram to check the progress of your treatment.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 5

**Cancer Type:** Left breast: ER+/PR+/HER2- grade 3 invasive ductal carcinoma with suspicion for LVI and focal high-grade DCIS component; Right breast: ER+/PR+/HER2- grade 1 invasive ductal carcinoma with intermediate nuclear grade DCIS

## Original Clinical Note

```
ID: ***** ***** is a 55 y.o. ***** with a recent diagnosis of early stage breast cancer, who presents in consultation to discuss treatment options and to establish care.    Date of Service: 10/21/17    HPI: The patient has not had screening mammograms.  She felt a lump in her left breast in February 2017.  She did not immediately seek care, but did start an evaluation (through ***** *****) in June 2017.  Her work-up and treatment to date are as follows:    February 2017 Felt lump in her left breast    06/14/17 Diagnostic mammogram: irregular spiculated mass in left upper outer quadrant corresponding to palpable abnormality; irregular mass in posterior, inferior, subareolar region of right breast  Targeted left breast ultrasound: irregular hypoechoic mass (1.8 x 2.3 x 2.7 cm) at 1:30 position, 10 cm from nipple  Targeted right breast ultrasound: lobulated hypoechoic mass (1 x 1.1 x 1.2 cm) at 8:00 position, 6 cm from nipple    06/17/17 US-guided left breast core biopsy at 8:00 position, 6 cm from nipple: 0.75 cm grade 3 IDC with ER+ 100%, PR+ 75%, HER2- (IHC 1+; FISH ratio 1.1)   US-guided right breast core biopsy at 1:30 position, 10 cm from nipple: 0.6 cm grade 1 IDC with ER+ 100%, PR+ 75%, HER2+ (IHC 2+; FISH ratio 2.0 with HER2 copy # per cell 4.4); intermediate nuclear grade DCIS     07/02/17 Breast MRI: right - irregular homgeneously enhancing mass (2.3 x 1.4 x 1.1 cm) corresponding to right breast index mass (at 8:00 position, 13 cm from nipple); left - irregular homgeneously enhancing mass (2.2 x 3.3 x 2.6 cm) corresponding to left breast index mass (at 2:00 position, 10 cm from nipple); multiple prominent low level 1 left axillary LN with thickened cortex, asymmetric compared to right    07/11/17 US-guided left axillary LN core biopsy (3:00 position, 19 cm from nipple): multiple foci of metastatic adenocarcinoma (largest 1.5 mm) with tumor cells present in adipose tissue adjacent to fibrous tissue, unclear whether this \n represents extracapsular invasion; HER2- (IHC 1+; FISH ratio 1.2)    08/13/17 Left breast partial mastectomy and axillary lymph node dissection: 6.2 cm grade 3 IDC; ER+ 99%, PR+ >95%, HER2- (1+; FISH ratio 1.0); suspicion for LVI, margins negative, focal high grade DCIS within tumor; 0/8 LN+ (0.7 mm micrometastasis with no extracapsular extension); MP high risk (-0.614)  Right breast partial mastectomy and axillary lymph node dissection: 2.0 cm grade 1 IDC with ER+ 99%, PR+ 90%, HER2- (IHC 0; FISH ratio 1.5); 0/8 LN+; MP Low Risk (+0.321)  (Surgeon: Dr. ***** *****)    Presents to clinic with her husband and sister (*****).  She reports that she has felt a lot healthier since having her breast surgery.  She has some mld joint stiffness, in particular in her knees.      ***** has met with Dr. ***** ***** and Dr. ***** ***** who have both recommended adjuvant chemotherapy (she states that they have specifically recommended TC).  She states that when she saw Dr. ***** he recommended that her tumor (presumptively left) be sent for Oncotype.  She is not yet aware of a result.    She is concerned that the benefits of chemotherapy will not outweigh its long-term risks.      Past Medical History:   Diagnosis Date   \u0007 Breast cancer    \u0007 Diabetes 08/09/2017    HbA1c of October 20 on 08/09/17    \u0007 Obesity        Past Surgical History:   Procedure Laterality Date   \u0007 AXILLARY LYMPH NODE DISSECTION Bilateral 10/09/2017    8 LN ON RIGHT; 12 LN ON LEFT   \u0007 MASTECTOMY, PARTIAL Bilateral 08/13/2017    WITH BILATERAL REDUCTION        Family History   Problem Relation Name Age of Onset   \u0007 Breast cancer Maternal Grandmother  58       Social History     Social History   \u0007 Marital status: Married     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A     Occupational History   \u0007 Not on file.     Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   \u0007 Alcohol use No   \u0007 Drug \n use: No   \u0007 Sexual activity: Yes     Other Topics Concern   \u0007 Not on file     Social History Narrative    The patient underwent menarche at 12 years of age.  She took OCP for < 1 year.  She is G0P0.  She states she underwent menopause at 51 years of age.  No HRT.    States she worked at ***** ***** ***** for a long time.    She walks 4-5 miles a day.       No outpatient encounter prescriptions on file as of 10/21/2017.     No facility-administered encounter medications on file as of 10/21/2017.    No Medications    Allergies/Contraindications  Allergies not on file      ROS     Physical Exam:   Vital Signs - Pulse 51  | Temp 36.1 C (97 F) (Oral)  | Resp 16  | Ht 165.5 cm (5' 5.16\") Comment: 10/21/2017 @ ***** | Wt (!) 105.7 kg (233 lb)  | SpO2 97%  | BMI 38.59 kg/m    Constitutional - WDWN, NAD  Eyes - sclera anicteric, PERRL, EOMI  HENT - sinuses nontender, nasal mucosa intact, pharynx without erythema, stomatitis, or thrush   Lymph Nodes - no cervical, supraclavicular, or axillary lymphadenopathy  Respiratory - resonant to percussion throughout, CTA bilaterally; no wheezes, rhonchi, or crackles  Cardiovascular - Normal heart rate, normal rhythm, no murmurs, no edema  Breast - no masses bilaterally  GI - Bowel sounds normal, soft; no tenderness, no distention, no HSM  Musculoskeletal - No tenderness over bones or joints.   Neurologic - Alert & oriented x 3, ambulates w/o difficulty, good strength throughout  Psychiatric - Mood stable; no HI, SI, hallucinations, paranoia, or delusions  Skin - warm without rashes or lesions     Studies:  Available labs, pathology, and imaging were reviewed and independently interpreted, as described above in the HPI.    Assessment and Recommendations:  55 y.o. postmenopausal patient with a recent diagnosis of bilateral breast cancer who should proceed with adjuvant chemotherapy in the context of clinically and genomically high risk disease in the left breast.       I discussed with \n the patient in great detail the natural history and treatment of early stage HR+/HER2- breast cancer.  We reviewed the available imaging and pathology reports.  I noted that she had 2 breast cancers with very different features.  Specifically, she has a Stage III (T3N1) HR+/HER2- grade 3 IDC of the left breast (MP high risk -0.614) and a Stage I (T1cN0) HR+/HER2- grade 1 IDC of the right breast (MP low risk).    With regards to her left breast cancer, I noted the high clinical risk features, including tumor size, lymph node status, and grade.  I explained that with the above clinical characteristics, patients would historically be recommended chemotherapy. I then introduced the utility of genomic profiling of the tumor (with tools like MammaPrint and Oncotype) to predict recurrence risk and benefit from chemotherapy. I noted that in the MINDACT clinical trial all patients with both high clinical risk and high genomic risk were recommended chemotherapy and that patients with such characteristics have not been studied with randomization to receive or not receive chemotherapy.  I stated that the addition of chemotherapy was important to reduce distant recurrence risk and that I would expect that benefit to be an absolute reduction in DFS of ~8%.  Her tumor is apparently being assessed for Oncotype DX RS.  I stated that I have seen some discordance between these two tests, and that were she found to have intermediate or high risk disease with Oncotype I would still recommend chemotherapy, but if she had low risk disease it would be both unexpected and difficult to reconcile - I would want to discuss the result at that time.  I noted that TAILORx does not apply to her given her positive LN, even as a micrometastasis.    I stated that with regards to chemotherapy regimens, I would recommend TC x 4 cycles over AC/T in the context of the ABC clinical trials and that she had only 1 positive LN.  We discussed extensively the \n typical SE associated with TC chemotherapy, including fatigue, myelosuppression, joint pain, myalgias, fluid retention, finger nail changes, alopecia, and cognitive dysfunction.    With regards to her right breast cancer, I noted the low clinical risk features, including tumor size, clear LN status, grade, and biomarker status.  I noted that chemotherapy would not be recommended for this tumor as it would not provide benefit.    I explained in the context of HR+/HER2- breast cancers, regardless of other therapy decisions, anti-estrogen therapy was the most important part of treatment her moving forward.  I explained that the standard, first-line recommendation for adjuvant anti-estrogen therapy in a post-menopausal patient with HR+/HER2- breast cancer is an aromatase inhibitor (AI), which blocks the conversion of androgen to estrogen, reducing the total body estrogen supply. I discussed the typical side-effects, including hot flashes, arthralgias, vaginal dryness, and bone density loss. I recommend that she take an AI for at least 5 years and that this would not start until after chemotherapy, and radiation.    I stated she would need a DEXA to assess her current bone health. We discussed ways to improve her bone health, such as weight bearing exercise. She will also take calcium and vitamin D. We also discussed the importance of aerobic activity (150 minutes a week), which she does appear to currently achieve through daily walking.      Recommendations in Brief:  - docetaxel/cyclophosphamide x 4 cycles  - radiation  - AI therapy for at least 5 years  - consider adjuvant trial of a CDK4/6 inhibitor    I spent a total of 120 minutes face-to-face with the patient and 115 minutes of that time was spent counseling regarding the diagnosis, the treatment plan, the prognosis, medication risks, lifestyle modification, symptoms and therapeutic options.  \n
```

---

## Letter A

Dear Patient,

We are writing to explain your recent diagnosis and the next steps in your care. We know this is a lot of information, and we are here to support you.

You have breast cancer in both breasts. In the left breast, you have Stage III invasive ductal carcinoma (a common type of breast cancer that starts in the milk ducts and grows into nearby tissue). This cancer is hormone receptor positive (HR+), which means it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein. It is higher risk because of its larger size and spread to a lymph node (small glands that help fight infection).

In the right breast, you have Stage I invasive ductal carcinoma. This cancer is also HR+ and HER2 negative, but it is smaller and lower risk, with no lymph node spread.

You have already had surgery to remove both tumors. The next step is adjuvant therapy, which means treatment after surgery to lower the chance of the cancer coming back.

For the left breast cancer, we recommend:

Chemotherapy (medicine that kills cancer cells) to reduce the risk of the cancer spreading
Radiation therapy (high-energy rays to kill any remaining cancer cells)

For both cancers, we recommend:

Hormone (anti-estrogen) therapy, such as an aromatase inhibitor. This medicine lowers estrogen in the body and helps prevent cancer from returning. It is usually taken for at least 5 years.

Before starting hormone therapy, you will have a DEXA scan (a test to check bone strength), since this treatment can affect your bones.

Next steps:

Finalize chemotherapy plan
Start treatment, followed by radiation
Begin hormone therapy after these treatments

You are not alone in this. Please bring any questions or concerns to your care team—we are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's Name],

Thank you for coming to our clinic today. I wanted to share with you the details of your recent diagnosis and treatment plan in a way that is easy to understand.

**Diagnosis and Stage:**
You have been diagnosed with early-stage breast cancer. This means there are two tumors, one in each breast. The left breast tumor is more serious and is considered Stage III. The right breast tumor is less serious and is considered Stage I.

**Treatment Plan:**
We recommend a few treatments to help manage your condition:

1. **Chemotherapy:** This is a type of medicine that helps kill cancer cells. We will give you four rounds of a treatment called TC. It may cause some side effects like feeling tired, joint pain, and hair loss. But it is important to reduce the chance of the cancer coming back.

2. **Radiation Therapy:** After chemotherapy, you will have radiation therapy. This uses special energy beams to kill any remaining cancer cells.

3. **Anti-Estrogen Therapy:** Since your cancer is sensitive to hormones, we will give you a medicine called an aromatase inhibitor (AI). This medicine stops your body from making too much estrogen, which can help prevent the cancer from growing. You will take this medicine for at least five years.

**Key Test Results:**
- Your left breast has a larger tumor that is more aggressive.
- Your right breast has a smaller tumor that is less aggressive.
- Some lymph nodes in your left armpit show signs of cancer, but not in your right armpit.

**Next Steps:**
- You will start chemotherapy soon.
- After chemotherapy, you will begin radiation therapy.
- Once you finish chemotherapy and radiation, you will start taking the AI medicine.

**What to Watch For:**
- If you feel very tired or have joint pain, let us know.
- Keep track of any new lumps or changes in your breasts.
- Continue to walk every day, as this is good for your health.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Your Name]  
Medical Communication Specialist  
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation to discuss treatment options for your recent diagnosis of bilateral breast cancer.
**What's new or changed since your last visit?**
You were diagnosed with two types of breast cancer: a more aggressive type in your left breast and a less aggressive type in your right breast. The left breast cancer is a grade 3 invasive ductal carcinoma (cancer that started in the milk ducts) with some features suggesting it might have spread slightly beyond the original tumor. The right breast cancer is a grade 1 invasive ductal carcinoma with some abnormal cells around the milk ducts. Both cancers are sensitive to hormones and do not have a protein called HER2.
**What treatment or medication changes were made?**
You will start a treatment plan that includes chemotherapy with docetaxel and cyclophosphamide for four cycles. This is to help reduce the chance of the cancer coming back. You will also take an aromatase inhibitor (a medication that blocks the production of estrogen) for at least five years to further reduce the risk of the cancer returning. Additionally, you will take calcium and vitamin D supplements to help maintain your bone health.
**What is the plan going forward?**
You will undergo radiation therapy to treat the areas where the cancer was removed. You will also continue taking an aromatase inhibitor for at least five years. A DEXA scan (a test to measure bone density) will be done to check your current bone health. Your tumor will be tested using a tool called Oncotype DX RS to help guide your treatment decisions.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 6

**Cancer Type:** ER-neg, PR neg, HER2 3+, FISH ratio 13, Ki67 10-15% invasive ductal carcinoma (IDC) grade 2

## Original Clinical Note

```
***** ***** is a 53 y.o. female with newly diagnosed HR-negative HER2+ breast cancer here to discuss systemic therapy and management    HISTORY OF PRESENT ILLNESS:    06/09/16 right diagnostic mammo with 3D Tomo- spiculated mass measuring 2.2 cm at   10-11:00    in right breast with adjacent architectural distortion and nodularity.   Ultrasound showed 1.6 cm mass at 10:00 3CFN, another 1 cm mass and 0.5 adjacent mass. Suspicious right axillary node    06/30/16 right breast core biopsy - IDC  G2, no LVI, ER-neg, PR neg, HER2 3+, FISH ratio 13,  Ki67 10-15%, FISH   Right axillary LN FNA: +metastatic breast adenoca    07/22/16 Breast MRI -   RIGHT - multiple right breast masses. 2.4 x 1.7 cm with clip, similar adjacent mass 2.4x1.6 cm also with clip. Multiple additional smaller masses. Enlarged level 1 and level 2 axillary LNs, the largest level 1 node is 3.5 x 1.7 cm  LEFT : negative      07/24/16 PET CT : . Solitary small focus of hypermetabolism in left ilium without definite CT correlate is concerning for metastatic disease.  Right breast mass compatible the patient's known primary breast cancer and multiple right axillary and subpectoral lymph nodes consistent with lymph node metastases.  3.4 x 2.6 cm hyperattenuating and mildly hypermetabolic lesion seen around the left carotid artery bifurcation may represent a carotid body tumor.     07/31/16 MRI Pelvis:  At least 3 rounded foci of signal abnormality and enhancement involving the left iliac bone and bilateral sacral ala, measuring up to 12 mm, suspicious for osseous metastatic disease.       *****/*****/*****: MR Face/Neck :  Avidly enhancing left carotid bifurcation mass with flow voids and splaying of the proximal internal and external carotid arteries. The imaging characteristics are suggestive of a carotid body paraganglioma.        The patient is accompanied by her husband today. She has no bone pain or any other systemic concerns. She says her neck mass is longstanding and \n stable.       PAST MEDICAL HISTORY:  Past Medical History:   Diagnosis Date   \u0007 Breast cancer    \u0007 Diabetes mellitus        PAST SURGICAL HISTORY:  No past surgical history on file.    GYN HISTORY:  OB History     Gravida Para Term Preterm AB TAB SAB Ectopic Multiple Living    7 6                Obstetric Comments    Perimenopausal, LMP 1-2 months ago, G7P6, first delivery age ~17, no HRT          MEDICATIONS:  Current Outpatient Prescriptions   Medication Sig Dispense Refill   \u0007 ergocalciferol, vitamin D2, (ERGOCALCIFEROL) 50,000 unit capsule      \u0007 metFORMIN (GLUCOPHAGE) 500 mg tablet      \u0007 omega-3 fatty acids-fish oil 300-1,000 mg capsule      \u0007 GAVILYTE-G 236-22.74-6.74 -5.86 gram solution      \u0007 JOLIVETTE 0.35 mg tablet      \u0007 ondansetron (ZOFRAN) 8 mg tablet To be taken as needed for nausea ( see colonoscopy prep instructions for details) 1 tablet 1   \u0007 ondansetron (ZOFRAN) 8 mg tablet To be taken as needed for nausea ( see colonoscopy prep instructions for details) 1 tablet 1   \u0007 prednisoLONE acetate (PRED FORTE) 1 % ophthalmic suspension      \u0007 PROLENSA 0.07 % DROPSOLN      \u0007 VIGAMOX 0.5 % ophthalmic solution        No current facility-administered medications for this visit.        ALLERGIES:  No Known Allergies    FAMILY HISTORY:  Family History   Problem Relation Age of Onset   \u0007 Family history unknown: Yes   no h/o malignancy. FA says he and wife are related.    SOCIAL HISTORY:  Social History     Social History Narrative    Lives in ***** with husband and son who is 25. 1 other estranged son in *****,  1 son in *****, 2 daughters live in *****    She moved from ***** 1 year ago    Husband was living in ***** for 33 years before coming to ***** 1 year ago with wife    She does not work, husband works cleaning buildings           REVIEW OF SYSTEMS:  Review of Systems - 14 point review of systems negative except problems noted above.        PHYSICAL EXAM:  General appearance - alert, well appearing, and in no \n distress  HEENT: anicteric, OP clear  Mental status - alert, oriented to person, place, and time  Lymph nodes - no palpable cervical or supraclavicular lymphadenopathy   Axilla - multiple right axillary nodes, largest 2.5 cm and mobile  Breasts - 7x5 cm mass in RUOQ, mobile, inverted right nipple, left breast without dominant mass   Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate, regular rhythm  Abdomen - soft,ND/NT, no HSM  Neurological - A&Ox3, grossly non-focal  Musculoskeletal - No point tenderness in spine or other ***** tenderness  Extremities - no edema  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted    LABS:    IMAGING/PATHOLOGY:  As noted above in HPI    ASSESSMENT/PLAN:    ***** ***** is a 53 y.o. female with newly diagnosed HR-negative HER2+ breast cancer here to discuss systemic therapy and management.  Her PET CT is suspicious for bone metastasis. To complete her staging work up we will need MRI of pelvis and biopsy for definitive stage IV diagnosis.  If stage IV will initiate systemic therapy with weekly paclitaxel, q3wk trastuzumab, q3wk pertuzumab. Given HR-negative HER2+ subtype, I anticipate excellent response and possible long term disease control. She is asymptomatic from bone lesion but we can radiate bone lesions as well.     Plan:  MRI Pelvis  MRI neck - like cartoid body, longstanding  Chemoteach for weekly paclitaxel, q3wk trastuzumab, q3wk pertuzumab  Baseline echocardiogram  Port placement  labs      1. The patient concerns were discussed in detail. Patient verbalizes understanding of these issues and agrees with the plan.   2. Reviewed treatment options, treatment side effects in detail.   3. I reviewed the patient's medical information and medical history.   4. I have reviewed the past medical, family, and social history sections including the medications and allergies listed in the above medical \n record.   5. Total face to face time: 60 ***** counseling time: 50    \n
```

---

## Letter A

Dear Patient,

We are writing to explain your recent diagnosis and the next steps in your care. We know this is a lot to take in, and we are here to support you.

You have breast cancer in the right breast. The type is called invasive ductal carcinoma, which means the cancer started in the milk ducts and spread into nearby tissue. Your cancer is HER2-positive (HER2+), which means it has a protein that helps cancer grow. It is hormone receptor negative (HR-negative), meaning it does not use hormones like estrogen to grow.

Tests show that the cancer has spread to nearby lymph nodes (small glands that help fight infection). There are also areas in the bone that may be cancer, based on imaging scans. We need more tests to confirm this.

Key test results:

Breast imaging showed multiple masses in the right breast
Biopsy confirmed HER2+ breast cancer
Lymph node biopsy showed cancer spread
Scans showed possible spots in the bone

Next steps:

MRI of the pelvis and a biopsy to confirm if the cancer has spread to the bone
MRI of the neck to check a long-standing neck mass
Echocardiogram (heart test) before starting treatment
Placement of a port (a small device under the skin to give medicine)

Treatment plan:
If the cancer is confirmed to have spread, we will start systemic therapy (treatment that goes through the whole body), including:

Chemotherapy (medicine to kill cancer cells)
Targeted therapy (drugs like trastuzumab and pertuzumab that specifically target HER2)

Radiation therapy may also be used if there are bone spots causing symptoms.

We understand this is overwhelming. Please ask questions and share any concerns with your care team. We are here to help you through every step.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's Name],

I hope this letter finds you well. I wanted to share some important information about your health with you in a way that is easy to understand.

**Diagnosis:**
You have been diagnosed with a type of breast cancer called HER2-positive breast cancer. This means the cancer cells have a protein called HER2, which helps them grow faster. Your cancer is also hormone receptor-negative, meaning it doesn’t respond to hormones like estrogen.

**Stage:**
The tests show that your cancer might have spread to your bones. We call this stage IV cancer. To be sure, we need to do more tests, such as an MRI of your pelvis and a biopsy.

**Treatment Plan:**
We will start you on a treatment plan that includes three different medicines:
- Paclitaxel: This medicine helps stop the growth of cancer cells.
- Trastuzumab and Pertuzumab: These medicines target the HER2 protein to help slow down the cancer.

We will also check how well your heart is working with a test called an echocardiogram. We will put in a port to make it easier to give you the medicines.

**Next Steps:**
- We will do an MRI of your pelvis to look closely at your bones.
- We will also do a biopsy to confirm if the cancer has spread.
- We will start the treatment plan as soon as we have all the information.

**What to Watch For:**
Please let us know if you notice any new symptoms or changes in how you feel. It’s important to keep track of any pain or discomfort.

**Questions:**
If you have any questions or concerns, please talk to your care team. We are here to help you understand everything and support you through this journey.

Warm regards,

[Your Name]
Medical Communication Specialist
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consultation regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You have a type of breast cancer called invasive ductal carcinoma (IDC), which means the cancer started in the milk ducts. The cancer is HR-negative and HER2-positive, and it has spread to your bones.
**What treatment or medication changes were made?**
You will start new medications: weekly paclitaxel, trastuzumab every three weeks, and pertuzumab every three weeks. These medications are used to treat HER2-positive breast cancer.
**What is the plan going forward?**
You will have an MRI of your pelvis and neck to check for any changes in your cancer. You will also have an echocardiogram (a heart ultrasound) to monitor your heart function. A port will be placed to make it easier to give you medications. You will also have some routine blood tests.
You are referred to have chemotherapy teaching sessions to prepare you for your treatment.
If needed, you can receive radiation therapy to treat the cancer in your bones.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 7

**Cancer Type:** ER-/PR-/HER2- triple negative invasive ductal carcinoma

## Original Clinical Note

```
This is a shared visit for services provided by me, ***** ***** *****, MD. I performed a face-to-face encounter with the patient and the following portion of the note is my own.      Impression: Metastatic ER negative, HER2 negative breast cancer on nab paclitaxel and pembrolizumab for     ***** ***** presents today for follow up ,she is is doing well and but has clinical or radiological evidence of progression for which she is to be treated with chemo-immunotherapy for her MSI high  metastatic disease.  o    The imaging studies were reviewed today and compared to prior scans.   Labs are in range for continuation of treatment     Plan:   Continue  with therapy until progression or intolerable side effects  Imaging guided by symptoms or every 3-4 months, longer intervals if stable.       .      This is a shared visit for services provided by Dr. ***** and myself.    SUBJECTIVE    ***** ***** is a 44 y.o. female with Lynch syndrome with early stage colon cancer and endometrial cancer and newly diagnosed metastatic breast cancer to liver and nodes. She started pembrolizumab and abraxane on 03/11/19 and presents today for cycle 1 day 8.    At today's visit she reports worsening GERD. Prilosec helps. Taking prn. Also reports nausea for which ativan works well.     She reports fatigue for 3 days after treatment.    She reports right axillary pain to shoulder blade on the right, much better since last week! Also right lumbosacral pain is stable. She reports new aching in her arms and legs. Continues flexeril and MS contin as well as oxycodone prn.    She reports numbness/tingling in toes and fingertips that is transient. She also reports leg cramping at night.    Off xarelto still since recent biopsy- was on it for non-occlusive jugular clot.     ROS  Constitutional: Negative for fever, chills, diaphoresis, appetite change, fatigue and unexpected weight change.   HENT: Negative for hearing loss, nosebleeds, \n congestion, sore throat, mouth sores, neck pain, sinus pressure, or tinnitus.    Eyes: Negative for photophobia, pain, discharge, redness, itching and visual disturbance.   Respiratory: Negative for apnea, cough, chest tightness, shortness of breath, wheezing.   Cardiovascular: Negative for chest pain, palpitations and leg swelling.   Gastrointestinal: Negative for nausea, vomiting, abdominal pain, diarrhea, constipation, blood in stool, or abdominal distention.   Genitourinary: Negative for dysuria, urgency, frequency, hematuria, vaginal bleeding or discharge or difficulty urinating.   Musculoskeletal: Negative for myalgias, back pain, joint swelling, or arthralgias.   Skin: Negative for itching or rash.   Neurological: Negative for neuropathy, dizziness, tremors, seizures, syncope, speech difficulty, weakness, light-headedness, numbness or  headaches.   Hematological: Negative for adenopathy. Does not bruise/bleed easily.   Psychiatric/Behavioral: Negative for suicidal ideas, sleep disturbance, decreased concentration.   Remaining systems were reviewed and are negative.    Patient Active Problem List    Diagnosis Date Noted   \u0007 Lynch syndrome 02/23/2019   \u0007 Colon cancer (HCC) 02/23/2019   \u0007 Metastatic breast cancer (HCC) 02/22/2019     Early stage breast cancer 2014  clinical Stage IIB (cT2 cN1 cM0) -> ypT1c(m) ypN1a (1/22 LN positive) triple negative invasive ductal carcinoma   s/p neoadjuvant ddAC + 5/12 weekly cycles of carboplatin/paclitaxel (discontinued due to poor tolerance) with disease progression while on carboplatin/paclitaxel   s/p partial mastectomy in February 2014   S/p adjuvant RT *****/*****-*****/*****,    Breast cancer local recurrence 2018  March 2018 biopsy proven recurrence of TNBC in the R chest wall   Genetic testing performed prior to the colon biopsy was negative for BRCA 1/2.     She had a pelvic US for pain 04/01/18 showing: 11mm EMS 2.4cm countor abnormality in the myometrium, 1.4cm complex right \n ovarian cyst, left 4.7cm septated cyst in left adnexa, poss hydrosalpinx.      Colon cancer  PET/CT for complete restaging showed no evidence of distant metastases but revealed a hypermetabolic focus at the splenic flexure, which was biopsied on 05/25/18 and showed a primary colon adenocarcinoma with mucinous differentiation and nuclear loss of MSH2 and MSH6 by IHC.   Germline mutation testing positive for MSH2 mutation. Lynch syndrome panel (blood): 5 genes were analyzed: EPCAM, MLH1, MSH2, MSH6, and PMS2. Positive for: MSH2 c.1906G>C (p.Ala636Pro)  05/25/18 Colonoscopy performed which showed a 3cm descending colon polyp s/p polypectomy: primary colon adenocarcinoma with mucinous features with loss of MSH2, MSH6 -> s/p transverse colectomy in June 2018 -> Stage I (pT2 pN0), no chemotherapy or radiation   --Brain MRI negative for metastases    Uterine cancer  PET/CT also showed some hypermetabolic activity in the uterus and workup is ongoing.   TVUS/pelvic US: No uterine masses to correspond to hypermetabolic activity (likely physiological), cyst c/f hydrosalpinx identified. + irregular vaginal bleeding  s/p endometrial biopsy on 06/30/18 with pathology showing well-differentiated endometrioid adenocarcinoma       Overall diagnosis currently:  Diagnoses:  1) July 2013: R breast invasive ductal carcinoma, ER-/PR-/Her2- (TNBC), grade 3  *****BRCA testing not performed as insurance did not approve  2) March 2018: Recurrence of ER-/PR-/Her2- breast cancer in R chest wall   *****Genetic testing 05/19/18: Negative for BRCA 1/2, ATM, CDH1,CHEK2, PALB2, PTEN, STK11, TP53  3) May 2018: Splenic flexure moderately differentiated colon adenocarcinoma showing mucinous differentiation; *****loss of MSH2, MSH6  --> Final Surgical Pathology showed poorly differentiated signet ring cell carcinoma, TVA, negative for high-grade dysplasia   *****Lynch syndrome panel (blood): 5 genes were analyzed: EPCAM, MLH1, MSH2, MSH6, and PMS2. Positive for: MSH2 c.1906G>C (p.Ala636Pro)  4) Endometrioid adenocarcinoma, well-differentiated, FIGO 1      April 2018 - November 2018 gemcitabine + carboplatin: good clinical response in pectoral lymph nodes    Metastatic breast cancer  PET/CT 01/27/19:   1. Ill-defined right pectoralis soft tissue mass with internal calcifications: 1.2 x 2.4 cm (series 4 1 image 64) with a maximum SUV of 4.6.  2. Right subpectoral lymph node: 0.7 cm in short axis (series 4 1 image 69) with maximum SUV of 4.8.  3. Hypoenhancing caudate lobe lesion: 1.9 x 1.5 cm (series 4 1 image 29) with a maximum SUV of 16.1.   4. Lytic S1 vertebral body lesion: maximum SUV of 6.6.  Lymph nodes: Index nodes as described above. Additional hypermetabolic right internal mammary node is 0.6 x 0.8 cm (series 4 1 image 75 with a maximum SUV of 4.3.  The hypermetabolic caudate lobe lesion is markedly increased in size when compared to prior CT abdomen/pelvis from 09/29/2018. Additional hypoattenuating, hypermetabolic segment VII lesion is 1.3 x 1.7 x 1.5 cm  with a maximum SUV of 11.2  02/25/19 Liver biopsy: triple negative metastatic breast cancer     \u0007 Vaginal bleeding 09/28/2018   \u0007 DVT (deep venous thrombosis) (HCC) 08/31/2018     Left Subclavian DVT diagnosed 05/16/18. Dye study demonstrated complete occlusion of the brachiocephalic vein - now s/p Mediport removal on 08/04/18 on rivaroxaban 20mg PO daily.       \u0007 Endometrial carcinoma (HCC) 08/26/2018     Added automatically from request for surgery *****           MEDICATIONS  Current Outpatient Medications   Medication Sig Dispense Refill   \u0007 albuterol 90 mcg/actuation metered dose inhaler Inhale 2 puffs into the lungs every 4 (four) hours as needed.     \u0007 AMBIEN CR 12.5 mg ER tablet nightly at bedtime.     \u0007 brimonidine (MIRVASO) 0.33 % GEL Apply topically Daily.      \u0007 cyclobenzaprine (FLEXERIL) 5 mg tablet Take 5 mg by mouth 3 (three) times daily as needed.     \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 tablet (1 mg total) by \n mouth every 6 (six) hours as needed for Anxiety. 30 tablet 3   \u0007 morphine (MS CONTIN) 15 mg 12 hr ER tablet Take 15 mg by mouth Twice a day.     \u0007 oxyCODONE (ROXICODONE) 5 mg tablet Take 1 tablet (5 mg total) by mouth every 8 (eight) hours as needed for Pain. 20 tablet 0   \u0007 traMADol (ULTRAM) 50 mg tablet Take 1 tablet every 8 hours as needed for pain     \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 2 tablets (1,000 mg total) by mouth every 6 (six) hours. (Patient not taking: Reported on 02/22/2019) 60 tablet 0   \u0007 esomeprazole (NEXIUM) 20 mg capsule Take 20 mg by mouth daily as needed.      \u0007 LORazepam (ATIVAN) 0.5 mg tablet Take 0.5 mg by mouth every 12 (twelve) hours as needed.     \u0007 naloxone 4 mg/actuation SPRAYNAERO Spray entire dose into 1 nostril for suspected overdose. If no response in 2-3 minutes, use new nasal spray for 2nd dose in other nostril.     \u0007 polyethylene glycol (MIRALAX) 17 gram packet Take 1 packet (17 g total) by mouth daily as needed (constipation). (Patient not taking: Reported on 02/22/2019)     \u0007 polyethylene glycol (MIRALAX) 17 gram packet Take 1 packet (17 g total) by mouth Daily. (Patient not taking: Reported on 03/17/2019) 510 g 11   \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 10 mg by mouth every 6 (six) hours as needed.     \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting). (Patient not taking: Reported on 03/17/2019) 30 tablet 5   \u0007 rivaroxaban (XARELTO) 20 mg tablet Take 1 tablet (20 mg total) by mouth every morning. Starting in 1 week, 10/07/18. DO NOT TAKE THIS MEDICATION FROM *****/*****-*****/***** (Patient not taking: Reported on 03/17/2019)     \u0007 simethicone (MYLICON) 80 mg chewable tablet Take 80 mg by mouth every 6 (six) hours as needed.     \u0007 simethicone (MYLICON) 80 mg chewable tablet Take 1 tablet (80 mg total) by mouth every 4 (four) hours as needed for Flatulence. (Patient not taking: Reported on \n 03/17/2019) 100 tablet 2     No current facility-administered medications for this visit.        ALLERGIES  Allergies/Contraindications   Allergen Reactions   \u0007 Penicillins Anaphylaxis     Puffy and itching,       Puffy and itching,          \u0007 Adhesive Tape-Silicones      Other reaction(s): Rash   \u0007 Haemophilus Influenzae      Other reaction(s): Unknown H1N1  Get puffy and injection site itchiness     Get puffy and injection site itchiness        \u0007 Hydrochlorothiazide      Other reaction(s): Swelling   \u0007 Hydroxychloroquine      Other reaction(s): Rash  Pt states she got a total body skin rash with hives needing prednisone to resolve and was told to stop taking it      Pt states she got a total body skin rash with hives needing prednisone to resolve and was told to stop taking it         \u0007 Ibuprofen      Other reaction(s): Swelling  As per patient elevated BP     As per patient elevated BP            PAST, FAMILY, and SOCIAL HISTORY  Past Medical History:   Diagnosis Date   \u0007 Asthma     mild, rarely uses her inhaler.     \u0007 Bipolar disorder (HCC)    \u0007 Breast cancer (HCC)    \u0007 Depression    \u0007 DVT (deep venous thrombosis) (HCC)     L subclavian   \u0007 Hepatitis C     treated *****-*****.   \u0007 History of chemotherapy    \u0007 Hypertension     not on any meds at the moment, bp controlled.   \u0007 Personal history of irradiation, presenting hazards to health      Family History   Problem Relation Name Age of Onset   \u0007 Leukemia Mother          as a child, in remission   \u0007 Other (Other) Mother          hysterectomy ?ovaries, 40s or 50s, s/p lacerated liver and liver transplant    \u0007 No Known Problems Father          no health information known about paternal family   \u0007 No Known Problems Brother          maternal half-brother, no health information known   \u0007 No Known Problems Maternal Uncle     \u0007 Colon cancer Maternal Grandmother          diagnosed 40s, mets to lung? deceased 70s   \u0007 Uterine cancer Maternal Grandmother     \u0007 Breast cancer \n Maternal Grandmother     \u0007 No Known Problems Brother          maternal half-brother, no health information known   \u0007 No Known Problems Son          22   \u0007 No Known Problems Son          18   \u0007 No Known Problems Daughter 16     Social History     Tobacco Use   \u0007 Smoking status: Former Smoker     Packs/day: 1.00     Years: 33.00     Pack years: 33.00     Types: Cigarettes     Last attempt to quit: 12/01/2017     Years since quitting: 1.2   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Not Currently   \u0007 Drug use: Yes     Types: Marijuana       PHYSICAL EXAM  ECOG 0  Vital Signs:  BP 116/75  | Pulse 75  | Temp 36.8 C (98.2 F) (Oral)  | Resp 16  | Ht 166 cm (5' 5.35\") Comment: December 2018 | Wt 86.5 kg (190 lb 12.8 oz)  | LMP 06/22/2018  | SpO2 98%  | BMI 31.41 kg/m   Constitutional:  WDWN, NAD.  Skin: warm, moist. No rashes or lesions.  Eyes: sclera anicteric, PERRL, EOMI.   HENT: sinuses nontender, nasal mucosa intact, pharynx without erythema, no thrush.   Lymph Nodes: no cervical, supraclavicular, or axillary lymphadenopathy.    Respiratory: BL lungs CTA, resonant to percussion, no wheezes, rhonchi, or crackles  Cardiovascular: normal heart rate, normal rhythm, no murmurs, no edema  Breast: no masses  GI: bowel sounds normal, soft, nontender, no distention, no organomegaly   Musculoskeletal: no point tenderness over bones or joints   Neurologic: alert & oriented x 3, no focal motor or sensory deficits  Psychiatric: mood stable    RESULTS  I have personally reviewed and interpreted the following:         ASSESSMENT & PLAN  ***** ***** is a 44 y.o. female with Lynch syndrome with early stage colon cancer and endometrial cancer and newly diagnosed metastatic breast cancer to liver and nodes. She started pembrolizumab and abraxane on 03/11/19 and presents today for cycle 1 day 8.    Metastatic breast cancer  - axillary pain improved which is hopeful for early treatment \n response  - Continue abraxane every week (continuous) and Pembrolizumab every 3 weeks. Monitor TSH  - Due to poor peripheral access she will need a port, hopefully she can get it next week    Jugular clot  - restart xarelto, stop for port placement per IR    Colon cancer    Endometrial cancer    Nausea/GERD due to chemo  - Continue ativan and compazine prn. Allergic to zofran  - Rx prilosec 40mg qd    Peripheral neuropathy due to Taxane, new and mild  - Monitor    Mylagias due to Taxane  - Rx mag oxide 400mg qhs    Pain  - Continue work with pain mgmt  - Continue MS Contin 15mg BID, oxycodone prn, flexeril, and ativan  - continue stool softener to prevent opioid induced constipation    F/u 2 weeks then every 3 weeks      I have reviewed and updated the patient's past medical history, medications, allergies, family and social histories. I spent a total of 45 minutes face-to-face with the patient and 40 minutes of that time was spent counseling regarding the treatment plan, symptoms and therapeutic options. Patient exp
```

---

## Letter A

Dear Patient,

We are writing to explain your current health and treatment plan. We know this is a lot, and we are here to support you.

You have metastatic breast cancer, which means the cancer has spread to other parts of the body, including the liver and lymph nodes (small glands that help fight infection). Your breast cancer is triple negative, meaning it does not use estrogen, progesterone, or HER2 to grow.

You also have a condition called Lynch syndrome, which is a genetic condition that increases the risk of certain cancers.

Current treatment:
You are receiving:

Chemotherapy (nab-paclitaxel) – medicine that kills cancer cells
Immunotherapy (pembrolizumab) – a treatment that helps your immune system fight cancer

Your care team reviewed your recent scans. These show that the cancer has grown or progressed, so your current treatment will continue as long as it is helping and side effects are manageable.

How you are feeling:
You have reported:

Heartburn (GERD), helped by medication
Nausea, helped by medication
Fatigue for a few days after treatment
Some pain, which has improved in places
Tingling in fingers and toes (nerve irritation from treatment)

Next steps:

Continue current treatment
Repeat scans every 3–4 months, or sooner if needed
Place a port (a small device under the skin to help give treatment)

We understand this is challenging. Please talk with your care team about any questions or concerns. We are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's Name],

Thank you for coming to your appointment today. Here is a summary of what we discussed:

**Diagnosis:**
You have metastatic breast cancer. This means the cancer has spread to other parts of your body, like your liver and lymph nodes. It is also called triple-negative breast cancer because it does not respond to certain hormones or proteins.

**Treatment Plan:**
You are receiving two medicines: abraxane and pembrolizumab. These help fight the cancer. You will keep getting these treatments as long as they are working and you do not have bad side effects.

**Key Test Results:**
We looked at your latest scans and lab tests. They show that you are doing okay, but there are signs that the cancer is growing a little bit. We will keep watching closely.

**Next Steps:**
You will come back in 2 weeks, and then every 3 weeks after that. We will check how you are doing with more scans and tests.

**What to Watch For:**
Please let us know if you have any new symptoms or if your current ones get worse. Some things to watch out for include:
- Feeling very tired
- Pain in your arms, legs, or back
- Tingling in your fingers or toes
- Trouble sleeping

**Medicines:**
You are taking several medicines to help with pain, nausea, and other symptoms. Please take them as directed.

**Important Reminders:**
- Keep all your appointments.
- Call us if you have any questions or concerns.
- Remember to talk to your care team about any worries you might have.

We are here to support you through this journey. Please feel free to reach out anytime.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your breast cancer treatment.
**What's new or changed since your last visit?**
Imaging studies show that your cancer has progressed. Specifically, a lesion in your liver has grown larger, and there are new areas of cancer activity in other parts of your body.
**What treatment or medication changes were made?**
You will continue taking abraxane and pembrolizumab. You will also restart xarelto, a blood thinner, after your port placement. You will take medications to manage nausea, pain, and to prevent constipation.
**What is the plan going forward?**
You will have a port placed soon to make it easier to receive your treatments. You will have a PET/CT scan to monitor your cancer. Your TSH levels will be checked. You will return to the clinic in 2 weeks, then every 3 weeks.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 8

**Cancer Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma with extensive DCIS

## Original Clinical Note

```
  Chief complaint: Patient with breast cancer here to discuss adjuvant therapy    History of Present Illness:  ***** ***** is a 70 y.o. female with a h/o HTN, HLD, and a recent diagnosis of stage IIA pT2(m)N1a L breast IDC (3.9 cm, grade 2, ER+/PR+/HER- with FISH ratio 1.1, high-risk mammaprint) s/p 08/18/17 bilateral mastectomies with left axillary LND with 2/12 LN+, who presents to establish Oncology care.    Ms. *****'s history is notable for a screening mammogram performed 01/25/17, which was read as unremarkable with BI-RADS 1.  Several months later, she noticed a left breast mass.  She underwent another mammogram on 05/27/17 that showed several solid masses (solid mass in the left breast at 3 o'clock; a solid mass in the left breast at 1 o'clock; prominent lymph node in the left axilla).  Ultrasound performed at ***** showed a 11 x 12 x 12 mm irregular, heterogeneous hypoechoic solid mass in the outer central left breast at ~3 o'clock and a 15 x 12 x 14 mm irregular solid mass in the upper outer left breast at ~2 o'clock likely within the large area of multiple small masses seen at mammography, and at least two abnormal L axillary lymph nodules.  A 06/20/17 ultrasound-guided core biopsy of the L breast mass at  3 o'clock showed invasive ductal carcinoma, as well as ductal carcinoma in situ, intermediate nuclear grade, and microcalcification associated with ductal carcinoma in situ, with strong ER positive (100%), strong PR positive (>95%), and HER-2 negative (1+ IHC).She also underwent a FNA of a left axillary lymph node that revealed metastatic carcinoma of the breast.      She was evaluated by Dr. ***** of breast surgical oncology and the patient opted to pursue bilateral double mastectomies.  She was admitted from *****/*****/*****-*****/*****/***** for bilateral mastectomies, left axillary lymph node dissection, levels 1 and 2, and bilateral plastics closure.  Her pathology showed IDC multifocal, SBR grade 2 and 1, \n largest focus 3.9 cm, margins negative; Ductal carcinoma in situ, intermediate nuclear grade, metastatic carcinoma to two lymph nodes (2/2), 1.5 cm largest deposit, microcalcifications in association with ductal carcinoma in situ and benign ducts.  She was last seen by Dr. ***** in clinic on 08/31/17 where her surgical sites were well-healed and her JP drains were removed. She presents today to establish medical oncology care.  She has an appointment with ***** ***** on 10/10/2017.    Has been feeling well since her surgery, has been healing well.  Has slight more arm stiffness on her left arm but it is improving.  Has some associated numbness near her left axilla and left elbow post-operatively.    Of note is that her paternal grandmother passed away from breast cancer in her 60s.  Sister had breast cancer s/p double mastectomy.   No ovarian or pancreatic cancer history.  Her father has ***** Jewish heritage.  She has not undergone Genetic testing previously.         Past Medical History:   Diagnosis Date   \u0007 Arthritis    \u0007 Breast cancer     left side   \u0007 Cataract     right eye   \u0007 GERD (gastroesophageal reflux disease)     on PPI   \u0007 Glaucoma     both eyes   \u0007 Heart murmur    \u0007 Hyperlipidemia    \u0007 Hypertension    \u0007 Migraine headache    \u0007 MVP (mitral valve prolapse)     reports resolved- r/t scarlet fever   \u0007 Osteopenia    \u0007 PONV (postoperative nausea and vomiting)     with foot surgery- for 3 days post op and post op headaches   \u0007 Thyroid disease     hyperparathyroid        Past Surgical History:   Procedure Laterality Date   \u0007 APPENDECTOMY      with hysterectomy   \u0007 BREAST LUMPECTOMY Left     benign pathology   \u0007 HAMMER TOE SURGERY      left foot (and bunion)   \u0007 HYSTERECTOMY      for fibroids   \u0007 LIPOMA RESECTION      left side neck-    \u0007 R eye - Phaco w/ IOL      right eye   \u0007 TONSILLECTOMY     \u0007 vein/clot thermal ablation      right leg        Gynecologic History:   Menarche was at age 13. Patient is \n gravida 1 para 1 with first full term pregnancy at age 29. Patient used oral contraceptives for 6 months at age 23.   Hysterectomy performed at age 29 (due to fibroid).  Menopause was at age 50. Patient did not use hormone replacement therapy.    Family History:  Paternal grandmother passed away from breast cancer in her 60s.  Sister had breast cancer s/p double mastectomy.     No ovarian or pancreatic cancer.  Maternal grandmother had stomach cancer.  Maternal uncle had leukemia, lung cancer, HCC and laryngeal cancer.         Current Outpatient Prescriptions   Medication Sig Dispense Refill   \u0007 acetaminophen (TYLENOL) 325 mg tablet Take 2 tablets (650 mg total) by mouth every 4 (four) hours as needed for Pain or temp > 38.5 C. 60 tablet 0   \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 1 tablet (500 mg total) by mouth every 6 (six) hours as needed for Pain. 60 tablet 0   \u0007 ascorbic acid (VITAMIN C) 500 mg tablet Take 500 mg by mouth every other day.      \u0007 atenolol (TENORMIN) 25 mg tablet Take 25 mg by mouth every morning.      \u0007 docusate sodium (COLACE) 100 mg capsule Take 1 capsule (100 mg total) by mouth Twice a day. 60 capsule 0   \u0007 gabapentin (NEURONTIN) 300 mg capsule Take 1 capsule (300 mg total) by mouth 3 (three) times daily. 90 capsule 0   \u0007 ibuprofen (ADVIL,MOTRIN) 600 mg tablet Take 1 tablet (600 mg total) by mouth every 6 (six) hours as needed for Pain or temp > 38.5 C. 60 tablet 0   \u0007 multivitamin tablet Take 1 tablet by mouth Daily. Daily packet multi vitamin     \u0007 omeprazole (PRILOSEC) 20 mg capsule Take 20 mg by mouth every morning.      \u0007 simvastatin (ZOCOR) 20 mg tablet Take 20 mg by mouth nightly at bedtime.        No current facility-administered medications for this visit.         Social History:  She lives in ***** ***** in ***** *****, although is currently living in ***** with her daughter and grandchildren.  Works as an elementary school counselor.  Former smoker, smoked for a total \n of 10 years, quit decades ago.         Family History   Problem Relation Age of Onset   \u0007 Diabetes Father    \u0007 Breast cancer Paternal Grandmother    \u0007 Anesth problems Neg Hx    \u0007 Bleeding disorder Neg Hx    \u0007 PONV Neg Hx         Review of Systems:   Review of Systems   Constitutional: Negative for malaise/fatigue and weight loss.   HENT: Negative for congestion and sinus pain.    Eyes: Negative for double vision.   Respiratory: Negative for cough, sputum production and shortness of breath.    Cardiovascular: Negative for chest pain, palpitations and leg swelling.   Gastrointestinal: Negative for abdominal pain, constipation, diarrhea and heartburn.   Musculoskeletal: Negative for back pain and myalgias.   Skin: Negative for rash.        Post-mastectomy scars healing well   Neurological: Positive for sensory change (numbness of L axilla and L elbow). Negative for dizziness, focal weakness, weakness and headaches.   Psychiatric/Behavioral: Negative.        Physical Examination:   Physical Exam   Vitals reviewed.  Constitutional: She is oriented to person, place, and time. She appears well-developed and well-nourished. No distress.   Older woman, non-toxic, accompanied by daughter   HENT:   Head: Normocephalic and atraumatic.   Mouth/Throat: Oropharynx is clear and moist. No oropharyngeal exudate.   Eyes: EOM are normal. Right eye exhibits no discharge. Left eye exhibits no discharge. No scleral icterus.   Neck: Normal range of motion. Neck supple.   Cardiovascular: Normal rate, regular rhythm, normal heart sounds and intact distal pulses.  Exam reveals no gallop and no friction rub.    No murmur heard.  Pulmonary/Chest: Effort normal and breath sounds normal. No respiratory distress. She has no wheezes. She has no rales.   Abdominal: Soft. Bowel sounds are normal. She exhibits no distension. There is no tenderness. There is no rebound and no guarding.   Musculoskeletal: Normal range of motion. She exhibits no edema, \n tenderness or deformity.   Neurological: She is alert and oriented to person, place, and time. No cranial nerve deficit. Coordination normal.   Skin: Skin is warm.   Well-healing post-mastectomy scars, no erythema or fluctuance, not TTP   Psychiatric: She has a normal mood and affect. Her behavior is normal. Thought content normal.         Radiology:  06/20/17 Ultrasound, Left  Mammographic Findings:  There are scattered areas of fibroglandular density.    At the site of the palpable lump in the outer central left breast, middle depth, marked by BB  there is a 15 mm irregular obscured mass with associated architectural distortion.    No single mammographic correlate for site of palpable concern in the left breast marked by more anterior BB.  However, there are innumerable small masses centered in the outer central left breast with some extension into upper outer quadrant.  There is also associated nipple retraction.    Ultrasound Findings:  There is a 11 x 12 x 12 mm irregular, not-parallel, heterogeneous hypoechoic solid mass with angular margins, in the outer central left breast, at approximately 3 o'clock, 8 cm from nipple, corresponding to the mammographic mass with associated architectural distortion and one of the palpable lumps. Vascularity is present.    There is a 15 x 12 x 14 mm irregular solid mass with angular margins, in the upper outer left breast, at approximately 2 o'clock, 3 cm from nipple, which contains calcifications, likely within the large area of multiple small masses seen at mammography.    There are at least two abnormal left axillary lymph nodes at 2:00, 20cm from the nipple, with coritcal thickness measuring up to 9mm.    IMPRESSION:   Left breast: Highly suggestive of malignancy (BI-RADS 5).    Recommendations:  Recommend ultrasound-guided core biopsy of left breast mass, at 3:00, 8cm from the nipple as the more suspicious of the two ultrasound masses.    Recommend fine needle \n aspiration of abnormal left axillary lymph node.      06/20/17 Mammogram, Left  Mammographic Findings:  There are scattered areas of fibroglandular density.    At the site of the palpable lump in the outer central left breast, middle depth, marked by BB  there is a 15 mm irregular obscured mass with associated architectural distortion.    No single mammographic correlate for site of palpable concern in the left breast marked by more anterior BB.  However, there are innumerable small masses centered in the outer central left breast with some extension into upper outer quadrant.  There is also associated nipple retraction.    Ultrasound Findings:  There is a 11 x 12 x 12 mm irregular, not-parallel, heterogeneous hypoechoic solid mass with angular margins, in the outer central left breast, at approximately 3 o'clock, 8 cm from nipple, corresponding to the mammographic mass with associated architectural distortion and one of the palpable lumps. Vascularity is present.    There is a 15 x 12 x 14 mm irregular solid mass with angular margins, in the upper outer left breast, at approximately 2 o'clock, 3 cm from nipple, which contains calcifications, likely within the large area of multiple small masses seen at mammography.    There are at least two abnormal left axillary lymph nodes at 2:00, 20cm from the nipple, with coritcal thickness measuring up to 9mm.    IMPRESSION:   Left breast: Highly suggestive of malignancy (BI-RADS 5).  Recommendations:  Recommend ultrasound-guided core biopsy of left breast mass, at 3:00, 8cm from the nipple as the more suspicious of the two ultrasound masses.  Recommend fine needle aspiration of abnormal left axillary lymph node.        05/27/2017 Mammogram  IMPRESSION  Finding 1:Solid mass in the left breast at 3 o'clock is suspicious.  Ultrasound-guided core biopsy is recommended.     Finding 2:Solid mass in the left breast at 1 o'clock is suspicious.  Ultrasound-guided core biopsy \n is recommended.     Finding 3:Prominent lymph node in the left axilla is suspicious.  Ultrasound-guided core biopsy is recommended. The referring physician was  notified of these results by fax.    BI-RADS Category 4 : Suspicious  4C - Moderate Suspicion      01/25/2017 Mammogram  IMPRESSION  No mammographic evidence of malignancy.    Annual screening mammography in 1 year and monthly breast self-examination  are recommended for this patient.    The patient will be notified of these results by mail. Your office may  receive these results before the patient has been made aware.    BI-RADS Category 1 : Negative      Pathology:  Mammaprint: High-Risk    08/18/2017 FINAL PATHOLOGIC DIAGNOSIS  A. Right breast, simple mastectomy:   1. Atypical ductal hyperplasia.  2. Lobular carcinoma in situ and atypical lobular hyperplasia.  3. Papilloma, usual ductal hyperplasia, apocrine metaplasia, and duct  ectasia.  4. Micocalcifications in association with benign ducts.  5. Benign skin and nipple.    B. Left breast, simple mastectomy:  1. Invasive ductal carcinoma, multifocal, SBR grade 2 and 1, largest  focus 3.9 cm, margins negative; see comment.  2. Ductal carcinoma in situ, intermediate nuclear grade, mixed  cribriform, micropapillary, solid and papillary patterns with comedo  necrosis, margins negative.  3. Papilloma, usual ductal hyperplasia, radial scar, apocrine  metaplasia, cysts, and duct ectasia.  4. Metastatic carcinoma to two lymph nodes (2/2), 1.5 cm largest  deposit.  5. Biopsy site changes.  6. Microcalcifications in association with ductal carcinoma in situ and  benign ducts.  7. Skin and nipple with no tumor.    C. Left axillary level 1 and level 2 contents, dissection: No tumor in  ten lymph nodes (0/10).    D. Left breast skin, excision: Benign skin.    E. Right breast skin, excision: Benign skin.    COMMENT:  For the left mastectomy specimen (part B), two lesions are \n identified  grossly. The first lesion (designated as lesion #1 in the gross  description) is in the lower central breast, measures 3.0 cm and is  associated with a biopsy clip. This lesion shows a single continuous  invasive ductal carcinoma microscopically, spans slices 9 to 11 and  measures 3.0 cm (tumor #1 in the synoptic comment). The second lesion  (lesion #2) is superior to lesion #1 and appears multinodular grossly in  slices 7 to 11. Extensive and sequential sampling of lesion #1 shows  extensive DCIS with three separate foci of invasive carcinoma:  designated as tumors #2 to 4 as in the synoptic comment. Tumor #2 is in  slice 7 and tumor #3 in slices 9 to 11, separated by slice 8, which  shows DCIS without invasion (slides B7 and B26). Tumor #1 and tumor #3  are separated by 1.8 cm of DCIS as in slide B9. Tumor #4 is in ***** 9  and is 0.8 cm away from tumor #3 as in slide B10.     Tumors #1 to 3 have overall similar morphology and are of SBR grade 2,  while tumor #4 has different morphology and is of SBR grade 1.     Breast Tumor Synoptic Comment   - Laterality: Left.   - Invasive tumor type: Invasive ductal carcinoma, four foci, summarized  as follows:  Tumor #1: 3.0 cm (lower central, correspon
```

---

## Letter A

Dear Patient,

We are writing to explain your recent diagnosis and your treatment plan. We know this can feel overwhelming, and we are here to support you.

You have breast cancer in the left breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and spread into nearby breast tissue. Your cancer is Stage IIA, which means it is larger and has spread to nearby lymph nodes (small glands that help fight infection).

Your tumor is hormone receptor positive (ER+/PR+), which means it grows with hormones like estrogen and progesterone. It is HER2 negative, meaning it does not have extra HER2 protein.

You already had surgery (bilateral mastectomies) to remove the breast tissue and lymph nodes. The largest tumor was 3.9 cm, and 2 lymph nodes had cancer. The surgery margins were clear, meaning no cancer was seen at the edges.

A special test called MammaPrint shows your cancer is high risk, meaning there is a higher chance the cancer could come back.

Treatment plan (adjuvant therapy):
This means treatment after surgery to lower the risk of the cancer returning. Your care team will discuss:

Chemotherapy (medicine to kill cancer cells)
Hormone therapy (medicine to block estrogen and slow cancer growth)

Next steps:

Meet with your oncology team to finalize your treatment plan
Begin therapy as recommended

It is good to hear you are healing well from surgery. Some arm stiffness and numbness can happen after lymph node surgery and often improve with time.

You are not alone in this. Please ask your care team any questions you have. We are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Ms. *****,

I hope this letter finds you well. I am writing to summarize your recent visit to our cancer center and to explain your diagnosis and treatment plan in simple terms.

**Diagnosis:**
You have been diagnosed with stage IIA breast cancer. This means the cancer has spread to some nearby lymph nodes but not to distant parts of your body. The cancer is 3.9 centimeters in size and is called invasive ductal carcinoma (IDC).

**Treatment Plan:**
You recently had surgery to remove both breasts (bilateral mastectomies) and some lymph nodes from your left armpit. The cancer was found in two lymph nodes out of twelve that were checked.

After reviewing your case, we believe you would benefit from additional treatments to help prevent the cancer from coming back. These treatments include:

1. **Chemotherapy:** This is medicine that helps kill any remaining cancer cells. We plan to start this treatment around October 18, 2017. Chemotherapy can cause side effects like feeling tired, being sick to your stomach, and losing your hair. We will teach you more about these before you start.

2. **Medicine to block hormones:** Since your cancer uses hormones to grow, you will take a medicine called an aromatase inhibitor. This helps stop the cancer from growing.

**Next Steps:**
Before starting chemotherapy, you will have a special heart check-up called a TTE. We will also place a small device called a port under your skin to make it easier to give you the chemotherapy medicines.

**What to Watch For:**
Please let us know if you notice any new symptoms or if you feel very tired or sick. It's important to keep your follow-up appointments so we can monitor your progress.

**Genetic Testing:**
Because of your family history, we think it’s a good idea for you to have genetic testing. This can help us understand more about your risk for cancer and guide future care.

**Questions:**
If you have any questions or concerns, please talk to your care team. We are here to support you through this journey.

Warm regards,

[Your Name]  
Medical Communication Specialist  
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consultation regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You recently had surgery to remove your breast cancer, which included removing both breasts and checking the lymph nodes under your left arm. The cancer was a type that started in the milk ducts and was grade 2. The doctors found that the cancer had spread to two lymph nodes. No new imaging findings were noted. Your surgical wounds are healing well, and you have some numbness near your left armpit and elbow.
**What treatment or medication changes were made?**
You agreed to start chemotherapy with AC plus paclitaxel, which is scheduled to begin on October 18, 2017.
**What is the plan going forward?**
Before starting chemotherapy, you will have a heart ultrasound (TTE) to check your heart function. A port will be placed for IV access. You are referred to genetics for counseling and testing due to your family history of breast cancer. Your next step is to start chemotherapy on October 18, 2017.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 9

**Cancer Type:** ER+/PR+/HER2- grade 2 IDC (micropapillary features) with metastatic recurrence

## Original Clinical Note

```
ID: ***** ***** is a 31 y.o. premenopausal female patient with a history of a Stage III HR+/HER2- IDC of the left breast, now with biopsy proven recurrence, now presenting in consultation to discuss treatment options and to establish care.     Date of Service: 01/28/19    HPI: The patient was originally diagnosed with breast cancer in late 2013 after palpating a mass in her left breast while breastfeeding.  Imaging in August 2013 demonstrated a hypoechoic focus in her left breast and subsequent evaluations led to a diagnosis of a HR+/HER2- IDC of the left breast. She was treated with bilateral skin-sparing mastectomy and started on adjuvant ddAC, but stopped after 3 cycles due to intolerance of side-effects including several hospitalizations.  She eventually started leuprolide plus exemestane, later switching to leuprolide plus tamoxifen, but this therapy was d/c after 1-2 years in the context of loss of her health insurance coverage. In late June 2019, she felt a left breast mass and a subsequent US and biopsy demonstrated a HR+/HER2- IDC of the left breast. Her diagnostic history and evaluation are as listed below:     08/03/13 Bilateral breast US: Irregular somewhat scalloped appearning hypoechoic focus (2 cm) in left axillary tail  US-guided core biopsy of 2.2 cm palpable nodule in upper-outer left breast: IDC with micropapillary features  ER+ (H score 290, or >75%), PR+ (H score 290, or >75%), HER2- (FISH ratio 1.0), Ki-67 5%     08/10/13 Bilateral diagnostic digital mammogram: Extensive, segmentally distributed, clustered calcifications (13 cm in extent) in upper outer quadrant of left breast; associated focal, poorly defined, dense and irregular mass (2.3 cm) present superolaterally in upper outer quadrant of left breast  Left axillary US: Two adjacent LN demonstrating focal, nodular thickening of parenchyma (1.8 cm and 1.9 cm)  US-guided left axillary LN biopsy: Metastatic carcinoma with ER+ (H score 290, or \n >75%), PR+ (H score 290, or >75%), HER2- (FISH ratio 1.02), Ki-67 5%     09/08/13 Bilateral breast skin-sparing mastectomy + left axillary SLNB with DIEP: left breast - 8.0 cm grade 2 IDC micropapillary type, LVI present, final margins clear; intermediate nuclear grade DCIS present, 7/16 LN+ (1.2 cm with extracapsular extension) pT3pN2; right breast - no carcinoma    09/17/13 Whole body bone scan: Usual symmetric radiotracer uptake in axial and appendicular skeleton is observed, no findings suggesting metastatic disease     09/18/13 CT CAP: no evidence of metastases     11/12/13 started dose dense Adriamycin/cytoxan - decided to stop active treatment after 3 cycles.  Had severe nausea, vomiting, fevers, chills as well as diffused body pains - per pt.     12/18/13 CT CAP: Soft tissue swelling of bilateral breast reconstructions without obvious loculated fluid collection in visualized lower breast reconstructions, region of abdominoplasty wound in midline there is a sinus tract extending to open would on skin filled with fluid (4.1 x 2.2 x 1.3 cm), induration and soft tissue swelling about this region   Bilateral breast US: Moderate bilateral subcutaneous soft tissue swelling; no evidence of loculated fluid collection     03/10/15 CT CAP: increased size of mildly hypermetabolic right deep pectoral and axillary LN (1.4 cm short axis MAX suv 4.9), mildly hypermetabolic right axillary LN (1.1 short axis MAX suv 4.9) are larger, area of right outer lower breast central fat density (4.3 x 4.9 cm SUV max 4.1) most compatible with fat necrosis, similar area is seen in upper and lower medial left breast    03/21/15 US-guided right breast LN biopsy: Benign axillary node hyperplasia    2015 Started on 4 week Lupron x 1-2 years plus exemestane and latera tamoxifen by Dr. *****    08/18/15 CT chest: No pulmonary embolus or other acute abnormality     09/13/15 Bilateral US complete: Previously examined right axillary LN decreased in size \n with decreased cortical thickening, consistent with benign path results     Recurrence     June 2018 Patient noticed a left breast lump increasing in size and darkening of the associated skin      10/26/18 Left breast US (at *****):  Irregular lobulated solid mass (3.1 x 1.8 x 3.0 cm) with a satellite mass (1.2 x 0.9 x 0.9 cm) at 1:00 position, 7 cm from nipple; mass appearing to involve overlying skin  US-guided biopsy of mass at 2:00 position of left breast: grade 2 IDC ER+(>95%), PR+(>95%), HER2- (IHC 1+) Ki-67 30%-40%    01/27/19 PET/CT: Left breast subcutaneous mass (2.1 x 3.2 cm SUV 8.4), left axillary node (1.3 x 1.4 cm SUV 6.0), left level 5B LN conglomerate (1.8 x 2.5 cm SUV 9.23), left internal mammary LN (1.2 x 2.0 cm SUV 6.8); left axillary, left supraclavicular, left level VB and left internal mammary hypermetabolic LAD are suspicious for metastatic diease; hypermetabolic subcutaneous masses overlying the left breast     The patient's gynecologic history is as follows.  She underwent menarche at 11 years of age.  She took OCP for less than a year.  She is G5P3; had two abortions. Her children are 12, 10 and 7 yo.  Her first pregnancy was at 19 years of age.  She has irregular menstrual periods.     She presents to clinic with her husband who has multiple medical problems.  She is currently unemployed - prior to her diagnosis she was involved in a home health agency.     States she was prescribed gapabentin for numbness and a deep sharp pain of her elbow that shoots up/down her entire left arm. The medication helps \"take the edge off\" - takes it as needed (~2x a week) when the pain is severe. Currently the pain is a 5-6/10.    Additionally, she has chronic back pain and occasional aches of her bilateral knees and ankles.     She believes that she tolerated Lupron injections well - was happy that her menstrual periods stopped during that time. Had to stop the Lupron injections once her insurance policy changed. \n She also recalls taking tamoxifen and exemestane and experiencing nausea and body pain.     States that her way to cope with her disease has been through getting tattoos.    She feels that the mass on her left breast is growing rapidly. She has noticed a some discoloration above the mass as well as a lump on her left neck area - initially assumed that this was an injury from straining her neck.     Her weight is stable. Admits to having a sweet tooth. She has been drinking soursop fruit juice.       Past Medical History:   Diagnosis Date   \u0007 Breast cancer (HCC) 2014   \u0007 HTN (hypertension)    \u0007 Hyperemesis gravidarum        Past Surgical History:   Procedure Laterality Date   \u0007 MASTECTOMY Bilateral 2014        Family History   Problem Relation Name Age of Onset   \u0007 Lymphoma Father     \u0007 Breast cancer Neg Hx         Social History     Socioeconomic History   \u0007 Marital status: Unknown/Declined     Spouse name: Not on file   \u0007 Number of children: Not on file   \u0007 Years of education: Not on file   \u0007 Highest education level: Not on file   Occupational History   \u0007 Not on file   Social Needs   \u0007 Financial resource strain: Not on file   \u0007 Food insecurity:     Worry: Not on file     Inability: Not on file   \u0007 Transportation needs:     Medical: Not on file     Non-medical: Not on file   Tobacco Use   \u0007 Smoking status: Current Every Day Smoker   \u0007 Tobacco comment: 3 cig/day   Substance and Sexual Activity   \u0007 Alcohol use: Yes     Frequency: 2-4 times a month   \u0007 Drug use: Not Currently   \u0007 Sexual activity: Not on file   Lifestyle   \u0007 Physical activity:     Days per week: Not on file     Minutes per session: Not on file   \u0007 Stress: Not on file   Relationships   \u0007 Social connections:     Talks on phone: Not on file     Gets together: Not on file     Attends religious service: Not on file     Active member of club or organization: Not on file     Attends meetings of clubs or organizations: Not on file     Relationship status: Not \n on file   \u0007 Intimate partner violence:     Fear of current or ex partner: Not on file     Emotionally abused: Not on file     Physically abused: Not on file     Forced sexual activity: Not on file   Other Topics Concern   \u0007 Not on file   Social History Narrative    The patient underwent menarche at 11 years of age.  She took OCP for <1 year.  She is G5P3.  Her first pregnancy was at 19 years of age.   Has regular periods, sometimes 3 week cycle, sometimes lasts over a week, sometimes spotting.       Outpatient Encounter Medications as of 01/28/2019   Medication Sig Dispense Refill   \u0007 gabapentin (NEURONTIN) 300 mg capsule      \u0007 hydroCHLOROthiazide (HYDRODIURIL) 25 mg tablet Take 25 mg by mouth.     \u0007 [DISCONTINUED] polymyxin b-trimethoprim (POLYTRIM) ophthalmic solution Affected eye for 5 days (Patient not taking: Reported on 12/28/2018) 10 mL 0   \u0007 [EXPIRED] iohexol (OMNIPAQUE) 350 mg iodine/mL solution 150 mL        No facility-administered encounter medications on file as of 01/28/2019.        Allergies/Contraindications   Allergen Reactions   \u0007 Hydrocodone-Acetaminophen    \u0007 Ibuprofen          Review of Systems   Constitutional: Negative for weight loss.   HENT: Negative.    Eyes: Negative.    Respiratory: Negative.    Cardiovascular: Negative.    Gastrointestinal: Negative.    Genitourinary:        Irregular menstrual periods   Musculoskeletal: Positive for joint pain.        Bilateral knee and ankle ache   Skin:        Discoloration of skin on left breast   Neurological:        Numbness on her entire left arm    Endo/Heme/Allergies:        Hot flashes        Physical Exam:    Vital Signs - BP 140/81  | Pulse 62  | Temp 36.7 C (98 F) (Temporal)  | Resp 18  | Ht 169.2 cm (5' 6.61\") Comment: December 19 @ ***** | Wt (!) 107.4 kg (236 lb 12.8 oz)  | SpO2 99%  | BMI 37.52 kg/m    Constitutional - WDWN, NAD  Eyes - sclera anicteric, PERRL, EOMI  HENT - sinuses nontender, nasal mucosa intact, pharynx without erythema, \n stomatitis, or thrush   Lymph Nodes -  No cervical, supraclavicular or axillary lymphadenopathy  Respiratory - resonant to percussion throughout, CTA bilaterally; no wheezes, rhonchi, or crackles  Cardiovascular - Normal heart rate, normal rhythm, no murmurs, no edema  Breast - Sitting upright, there is a  6.5 x 5 cm mass located on her left upper outer quadrant at 2:00 position 7cm from center of nipple; while lying down the same mass: 4.5 x 7 cm; 3.5 x 3.5 cm mass in left lateral anterior neck   GI - Bowel sounds normal, soft; no tenderness, no distention, no HSM  Musculoskeletal - No tenderness over bones or joints.   Neurologic - Alert & oriented x 3, ambulates w/o difficulty, good strength throughout  Psychiatric - Mood stable; no HI, SI, hallucinations, paranoia, or delusions  Skin - warm without rashes or lesions     Studies:  Available labs, pathology, and imaging were reviewed and independently interpreted, as described above in the HPI.    Assessment and Recommendations:  31 y.o. premenopausal female patient with a history of a Stage III (T3N2) HR+/HER2- grade 2 IDC with micropapillary features of the left breast, now with a locally advanced, unresectable recurrence (possibly considered metastatic in context of a level Vb LN), who requires additional evaluation prior to starting systemic therapy.    I discussed with ***** and her husband in great detail the natural history and treatment of locally advanced, and recurrent HR+/HER2- breast cancer.  We reviewed the pathology report and discussed the multidisciplinary approach to breast cancer treatment.  Her PET/CT demonstrated uptake in a level Vb lymph nodes, which is concerning for metastatic disease; regardless, her disease is currently unresectable and she needs to start systemic therapy.  We will get an FNA through the palpable FNA clinic as soon as possible to confirm diagnosis.     I explained that if she has HR+/HER2- MBC, it is not curable, but \n treatable, with primary goals of prolongation of life, while maintaining a good quality of life without excessive toxicities.  I stated that the use of chemotherapy upfront does not lead to better outcomes, and we therefore prefer to avoid chemotherapy as long as possible to limit toxicity.      I stated that the standard first line endocrine-based treatment for premenopausal patients with HR+ MBC is ovarian suppression plus an aromatase inhibitor and more recently, the addition of a CDK 4/6 inhibitor.     We will plan to start her on OS as soon as possible and then start an AI.  She has had poor tolerance of therapy in the past and we will need to take this process of adding on additional medications slowly.        Recommendations in Brief:  - plan to FNA the mass on left lateral anterior neck  - plan to start goserelin, followed by AI  - encouraged to reach out with any concerns    The patient will return to clinic two weeks after her first OS.     I spent a total of 70 minutes face-to-face with the patient and 65 minutes of that time was spent counseling regarding the diagnosis, the treatment plan, the prognosis, medication risks, lifestyle modification, the risks and benefits of surgery, symptoms and therapeutic options.    *****, ***** *****, am acting as a scribe for services provided by ***** ***** *****, MD on *****/*****/***** *****:***** PM.    The above scribed documentation as annotated by me accurately reflects the services I have provided.   ***** ***** *****, MD  *****/*****/***** 7:37 AM      \n
```

---

## Letter A

Dear Patient,

We are writing to explain your current condition and the next steps in your care. We know this is a lot to process, and we are here to support you.

You have breast cancer in the left breast called invasive ductal carcinoma (a cancer that starts in the milk ducts and spreads into nearby tissue). Your cancer is hormone receptor positive (HR+), meaning it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

You were first treated for Stage III breast cancer in 2013. Now, the cancer has come back (recurrence) in the left breast and nearby areas. Imaging shows cancer in the breast and lymph nodes (small glands that help fight infection), including areas near the neck. This disease is considered locally advanced and not removable with surgery at this time.

Next steps:

A needle biopsy (FNA) of the lump in your neck to confirm the diagnosis
Start systemic therapy (treatment that works throughout the body)

Treatment plan:
Your care team recommends:

Ovarian suppression (medicine to stop the ovaries from making estrogen)
Followed by hormone therapy with an aromatase inhibitor (a drug that lowers estrogen levels)
Additional medicines may be added slowly based on how you tolerate treatment

At this time, chemotherapy is not planned first, as hormone-based treatments can work well and may have fewer side effects.

Your care team’s goal is to control the cancer, help you live longer, and maintain your quality of life.

We understand this is difficult. Please ask questions and share any concerns. We are here for you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear *****,

Thank you for coming to our clinic today. I want to share with you the important details about your health and the next steps in your care.

**Diagnosis and Stage:**
You have breast cancer that has come back. It is called Stage III breast cancer. This means the cancer has spread a little bit beyond where it started.

**Treatment Plan:**
We need to start new treatments to help control the cancer. First, we will do a small test called an FNA to check the lump on your neck. This will help us know more about the cancer.

After that, we will start you on a medicine called goserelin. This medicine stops your ovaries from making hormones that can feed the cancer. Then, we will add another medicine called an aromatase inhibitor. These medicines work together to slow down the growth of the cancer.

**Key Test Results:**
Your recent tests show that the cancer has grown again in your left breast and some nearby lymph nodes. We found this using special scans like a PET/CT.

**Next Steps:**
- We will do the FNA test on your neck lump soon.
- We will start you on goserelin and then add the aromatase inhibitor.
- You will come back to see us two weeks after starting the goserelin.

**What to Watch For:**
Please let us know if you notice any changes in how you feel or if the lumps grow bigger. Also, tell us if you have any new pain or discomfort.

**Important Reminders:**
- Keep all your appointments.
- Talk to your care team if you have any questions or concerns.
- Remember, we are here to support you every step of the way.

Take care and stay strong.

Sincerely,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation to discuss treatment options for your breast cancer and to establish care.
**What's new or changed since your last visit?**
You have a type of breast cancer called ER+/PR+/HER2- grade 2 IDC with micropapillary features. The cancer has now spread to your left breast and left cervical lymph node, making it a stage IV cancer. Imaging showed a mass in your left breast and lymph nodes with increased metabolic activity. You also have a palpable mass in your left breast and another in your left neck. There is discoloration of the skin on your left breast.
**What treatment or medication changes were made?**
You will start on goserelin, which suppresses the ovaries, followed by an aromatase inhibitor (AI). These medications help control hormone levels that can fuel the growth of certain types of breast cancer.
**What is the plan going forward?**
You will have a fine needle aspiration (FNA) of the mass in your left neck to confirm the diagnosis. You will return to the clinic two weeks after starting goserelin.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 10

**Cancer Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma

## Original Clinical Note

```
HPI:  ***** ***** is a 32 y.o. female here in consultation to discuss treatment options for her newly diagnosed breast cancer whose history is as follows:    Patient Active Problem List    Diagnosis Date Noted   \u0007 Breast cancer, right breast 04/11/2017     Found mass in her right breast while in ***** on vacation and pain under her right arm.  03/19/17 diagnostic mammogram and US, core biopsy + for carcinoma  04/11/17 UCSF US right breast: 28 x 16 x 30 mm hypoechoic, irregular mass , upper central breast, at 12 o'clock, 3 cm from nipple.   Right axilla: lymph node with asymmetrically thickened cortex measuring up to 4mm maximum thickness, 11 o'clock, 9cm from the right nipple.  04/11/17 right breast core biopsy and axillary node FNA:  Grade 2 IDC, ER >95%, PR 25%, HER2 neg by IHC, 2.9/1.4 by FISH for average ratio of 2.0, axillary node positive for carcinoma  04/11/17 genetic testing sent  04/15/17 MRI: right breast with 34 x 20 x 25 mm irregularly shaped mass with spiculated margins and heterogeneous enhancement in the upper central to slightly outer quadrant. Enlarged right axillary lymph node with a cortex measuring up to 6 mm, likely corresponding to the recently sample axillary lymph node.  04/16/17 start egg harvesting process  04/17/17 PET/CT: Prominent right axillary lymph node measures up to 9 mm a maximum SUV of 2.1 (series 4, image 66). Hypermetabolic right breast mass is better characterized on recent MRI breast and measures approximately 3.2 cm the maximum SUV of 8.4.       Patient states that she overall feeling well, except for some mild bloating. She has a good support system through her family. ***** has questions regarding her ability to continue exercise including, biking, boogie boarding, and surfing while undergoing treatment. She is trying to improve her diet and eat more healthy.     Past Medical History:   Diagnosis Date   \u0007 Breast cancer, right breast March 2017   \u0007 Closed left ankle fracture 2010   \u0007 Viral \n meningitis 1994     Past Surgical History:   Procedure Laterality Date   \u0007 TRANSFER / TRANSPLANT ANKLE TENDON SUPERFICIAL / DEEP Left February 2010    snow board injury     Family History   Problem Relation Age of Onset   \u0007 Lung cancer Father 68     Heavy smoker, surgery January 2017   \u0007 Melanoma Maternal Grandfather 76     on eyelid   \u0007 Colon cancer Maternal Uncle 85     great uncle   \u0007 Melanoma Paternal Uncle 68     Allergies   Allergen Reactions   \u0007 Sulfa (Sulfonamide Antibiotics)        Social History     Social History Narrative    Single, lesbian without a partner    Marketing director for a ***** ***** ***** company    G0P0    Mother splits her time between ***** and *****    Sister also lives in ***** but has been staying with ***** since her diagnosis with her cat           Current Outpatient Prescriptions:   \u0007  CETRORELIX ACETATE (CETROTIDE SUBCUT), Inject under the skin., Disp: , Rfl:   \u0007  MENOTROPINS (MENOPUR SUBCUT), Inject under the skin., Disp: , Rfl:   \u0007  tamoxifen (NOLVADEX) 20 mg tablet, , Disp: , Rfl:   \u0007  UNABLE TO FIND, Powder vials SQ (Mix with 1cc diluent), Disp: , Rfl:     Review of Systems - All other systems were reviewed and are negative except that outlined above.    BP 120/61 | Pulse 61 | Temp 37.3 C (99.1 F) (Oral)  | Resp 16 | Ht 170.2 cm (5' 7.01\") | Wt 71 kg (156 lb 9.6 oz) | LMP 04/02/2017 (Exact Date) | SpO2 96% | BMI 24.52 kg/m2  Physical Examination: General appearance - alert, well appearing, and in no distress  Mental status - alert, oriented to person, place, and time  Eyes - pupils equal and reactive, extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - Right axillary node measuring 1 cm, no other adenopathy   Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate, regular rhythm, normal S1, S2, no murmurs, rubs, clicks or \n gallops  Abdomen - soft, nontender, nondistended, no masses or organomegaly  Breasts - There is a 7 x 3 cm mass, centered from 3:00 to 4:00 in the upper outer quadrant of the right breast. Right nipple is slightly pushed upward. Dimpling to the upper outer quadrant of the right breast.   Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema, no clubbing or cyanosis  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted    I personally reviewed and interpreted the following results:  Results for orders placed or performed during the hospital encounter of 04/17/17   POCT glucose, Fingerstick   Result Value Ref Range    Glucose, meter download 94 70 - 199 mg/dL     Petct Limited Whole Body Vertex To Mid Thigh    Result Date: 04/18/2017  PET/CT Whole Body (vertex to thighs)    *****/*****/***** *****:***** PM COMPARISON:   None REASON FOR THE STUDY:   Staging. CLINICAL HISTORY:   32 years old Female with right breast carcinoma, evaluate for metastatic disease.  TECHNIQUE:   Patient's fasting time was > 6 hours.  Following intravenous administration of 8.1 mCi of F18-FDG, a contrast enhanced, diagnostic CT was performed. This was followed by an emission PET scan started 52 minutes after FDG injection.  PET images were corrected for attenuation using the CT transmission data.  PET/CT was acquired from vertex to thighs. A rotating 3D MIP, as well as axial, coronal, and sagittal PET images with and without attenuation correction was interpreted.  Acquired and fused PET/CT images were reviewed alongside the PET images.  Patient's random blood glucose at the time of FDG injection was 94 mg/dL.  RADIATION DOSE INDICATORS: ***** ***** received 2 exposure event(s) (excluding scout) \n during this CT study. The range of CTDIvol was 9.04 to 9.04 (mGy) and the TOTAL DLP was 923.24 (mGy-cm). The dose indicators for CT are the volume Computed Tomography Dose Index (CTDIvol) and the Dose Length Product (DLP), and are measured in units of mGy and mGy-cm, respectively. These indicators are not patient dose, but values generated from the CT scanner acquisition factors.  Individual patient doses calculated from these values will depend on patient size and can substantially underestimate or overestimate the actual dose. For additional information on radiation dose, safety, and protection in the UCSF Department of Radiology and Biomedical Imaging, please refer to one of our FAQ leaflets on \"Computed Tomography (CT)\" or go to *****://*****.*****.*****.*****/*****/*****. FINDINGS: Brain:   Unremarkable, symmetric, FDG uptake is seen throughout the cortical gray matter, basal ganglia and the cerebellum.  No mass effect.  While these images appear within normal limits, MRI is recommended to rule out intracranial and/or skull base metastases if clinically indicated. Neck:   No hypermetabolic neck masses or lymphadenopathy. Chest:   Lungs are clear. Prominent right axillary lymph node measures up to 9 mm a maximum SUV of 2.1 (series 4, image 66). Hypermetabolic right breast mass is better characterized on recent MRI breast and measures approximately 3.2 cm the maximum SUV of 8.4. Abdomen/Pelvis:   Unremarkable abdomino-pelvic organs.  No hypermetabolic lymphadenopathy. Musculoskeletal:   No metastatic disease.     1. Hypermetabolic breast mass better characterized on recent breast MRI. 2. Prominent right axillary lymph node with low-level hypermetabolism, corresponding to previously sampled lymph node. No additional hypermetabolic lymph nodes. This study was reviewed by attendings Dr. ***** ***** from ***** ***** and  Dr. ***** *****. ***** of Nuclear Medicine. END OF IMPRESSION:     Breast Ultrasound Core \n Biopsy, Clip/specimen, Right    Addendum Date: 04/18/2017    ===ADDENDUM===Pathology is now available for review, and demonstrates invasive ductal carcinoma. Please see pathology report for full details. These pathology results are concordant with imaging findings. Recommend continued surgical/oncological management.    Result Date: 04/18/2017  Ultrasound Guided Breast Core Biopsy: Spring Loaded with Post Biopsy Clip Placement:   *****/*****/***** *****:***** PM COMPARISON:    Targeted ultrasound performed the same date CLINICAL HISTORY:    Biopsy proven invasive ductal carcinoma of the right breast. Core biopsy for clinical and research purposes. TECHNIQUE: Procedure was explained in detail to patient.  Risks (including risks of pain, infection, bleeding, non-recovery of targeted lesion, possible need for further intervention for diagnostic purposes), benefits, and alternatives discussed.  All questions were answered.  Written and verbal informed consent obtained.  This procedure immediately followed fine-needle aspiration.  Under sterile conditions, the lesion was re-identified at the 12 o'clock position, 3 cm from the nipple in the right breast. Following administration of a 1% Lidocaine with epinephrine, a small dermatotomy was made in the skin with a scalpel and a 14-16 gauge needle was placed under direct sonographic guidance with its tip at the lesion. Pre and post fire images demonstrated transgression of the needle through the lesion. A total of 8 biopsy samples were obtained (4 clinical, 4 research). Sterile Steri-Strips were placed at the incision site. The patient tolerated the procedure well and left the department in satisfactory condition with instructions that if excessive bleeding or oozing should occur, she should contact her health care provider or visit an emergency room. She may obtain the results from her referring physician within 4 working days and if this is not possible, she may contact this department. \n     1.   Ultrasound guided core right breast biopsy for clinical and research purposes. 2.   Pathology pending. 3.   Fine needle aspiration of a right axillary lymph node was performed on the same date. Please see separate report. END OF IMPRESSION:     Mr Breast With And Without Contrast, Bilateral    Result Date: 04/15/2017  Exam Date: 04/15/2017 Exam(s): MR breast bilateral en + un Clinical History: 32-year-old woman with biopsy proven right breast cancer (outside hospital). Patient underwent repeat core biopsy here at ***** (with research cores) and fine needle aspiration of a suspicious right axillary node (results pending at the time of this MRI). Comparison: Prior examinations dated 04/11/2017 (ultrasound) and 03/19/2017 were reviewed for comparison. MRI Technique: Bilateral 1.5T breast MRI without and with contrast was performed with a dedicated breast coil using T1 and T2-weighted sequences including:  axial T2-weighted with fat saturation, high resolution T1 fat-suppressed 3D gradient echo pre-, and dynamic post-contrast sequences. Gadolinium based contrast was injected intravenously at a dose of 0.1 millimole per kilogram body weight without adverse reaction. The study was processed with a Hologic Aegis Sentinelle system to optimize interpretation by generating multiplanar reconstructions (MPRs) and maximum intensity projections (MIPs) in 3 dimensions using raw and generated subtracted images from the dynamic contrast enhanced data set. MRI Findings: This bilateral breast MRI exam demonstrates moderate background enhancement. The breasts are composed of heterogeneous fibroglandular tissue and fat. RIGHT Breast: There is a 34 x 20 x 25 mm irregularly shaped mass with spiculated margins and heterogeneous enhancement in the upper central to slightly outer right breast (axial image 59, sag image 413). Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics. Susceptibility \n artifact within the mass corresponds to the clip within the mass. No other abnormal areas of enhancement or other MRI features of malignancy are identified in the right breast. LEFT Breast: No abnormal areas of enhancement or other MRI features of malignancy are identified. There is an enlarged right axillary lymph node with a cortex measuring up to 6 mm, likely corresponding to the recently sample axillary lymph node. IMPRESSION: Right breast: Known biopsy proven malignancy (BI-RADS 6). Left breast: Negative (BI-RADS 1). Recommendations: Recommend continued surgical and oncologic follow up of the known right breast cancer. Additionally, recommend correlation with cytology results of the recently sampled suspicious right axillary node. Recommend continued annual screening mammography for the left breast, according to American Cancer Society guidelines. Radiologist ***** *****. ***** <This report was electronically signed by ***** *****. ***** at *****/*****/***** *****:*****:53 PM>    Breast Ultrasound Complete, Right    Result Date: 04/11/2017  Exam Date: 04/11/2017 Exam(s): Right breast ultrasound Clinical History: 32-year-old woman for sonographic exam targeted to biopsy proven invasive ductal carcinoma located at the 12 o'clock position, 3 cm from the right nipple prior to ultrasound guided biopsy for clinical and research purposes.  New palpable lump of the right axillary region. Comparison: Prior examination dated 03/19/2017 was reviewed for comparison. Ultrasound Findings: Targeted US of the right breast redemonstrated at 28 x 16 x 30 mm hypoechoic, irregular mass with indistinct margins, in the upper central right breast, at approximately 12 o'clock, 3 cm from nipple. Vascularity is not present. Findings are compatible with biopsy proven malignancy. Targeted US of the right axillary region demonstrated a lymph node with asymmetrically thickened cortex measuring up to 4mm maximum thickness, located at the 11 o'clock position, \n 9cm from the right nipple. IMPRESSION: Right breast: Known biopsy proven malignancy (BI-RADS 6). Recommendations: 1.  Core biopsy will be performed today for the known malignancy in the upper central right breast. 2.  Fine needle aspiration will be performed today for the right axillary lymph node. Radiologist ***** *****, M.D. <This report was electronically signed by ***** *****, M.D. at *****/*****/***** *****:*****:47 PM>    Breast Ultrasound Fna, Right    Addendum Date: 04/18/2017    ===ADDENDUM===Cytology is now available for review, and demonstrates metastatic ductal carcinoma of breast. Please see pathology report for full details. These cytology results are concordant with imaging findings. Recommend continued surgical/oncological management.    Result Date: 04/18/2017  Ultrasound Guided Fine Needle Aspiration: *****/*****/***** *****:***** PM COMPARISON:    Right breast and right axillary ultrasound 04/11/17 and right breast ultrasound 03/19/17 CLINICAL HISTORY: 32 y.o. female with biopsy proven right breast cancer (diagnosed at an outside facility) presents for FNA of an abnormal right axillary lymph node with thickened cortex. TECHNIQUE: Procedure was explained in detail to patient. 
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment options. We know this is a lot to take in, and we are here to support you.

You have breast cancer in the right breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and has spread into nearby breast tissue. Your cancer is hormone receptor positive (ER+ and PR+), which means it grows with hormones like estrogen. It is HER2 negative, meaning it does not have extra HER2 protein.

Tests show a tumor about 3 cm in size in your right breast. A nearby lymph node (a small gland that helps fight infection) also has cancer cells.

Good news: your PET/CT scan did not show cancer spread to other parts of the body.

What this means:
Your cancer is at an early but higher-risk stage because it involves a lymph node.

Next steps:
Your care team will discuss a treatment plan with you. This may include:

Surgery to remove the tumor
Systemic therapy (treatment that works throughout the body), such as:
Hormone therapy (to block estrogen and slow cancer growth)
Chemotherapy (medicine to kill cancer cells)

You have already started egg harvesting, which is a way to help preserve fertility before treatment.

It is great that you are staying active and focusing on healthy habits. Please talk with your care team about exercise during treatment.

You are not alone. Please ask questions and share any concerns. We are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to our clinic. I want to share with you the important information about your health.

**Diagnosis:**
You have breast cancer in your right breast. It is called a grade 2 invasive ductal carcinoma. This means the cancer cells are growing and spreading into nearby tissues. The cancer is in the early stages, which is good news.

**Tests:**
We did several tests to learn more about your cancer. We found a lump in your right breast and a swollen lymph node in your armpit. These tests help us know how to treat you best.

**Treatment Plan:**
We will start with chemotherapy to shrink the tumor before surgery. Chemotherapy uses special medicines to kill cancer cells. We will also use a device called a port to give you the medicine easily.

Here’s what we will do:
- **Echocardiogram and EKG:** These tests check your heart.
- **PET Scan:** This test checks if there is any cancer spread.
- **Cold Cap:** We talked about using a cold cap to try to keep your hair during chemotherapy.
- **Chemotherapy:** We will start chemotherapy soon. We will use a mix of medicines to fight the cancer.

**Next Steps:**
- We will do more tests on Thursday.
- We hope to start chemotherapy next week.
- We will put in a port to make giving you medicine easier.

**What to Watch For:**
During treatment, you might feel tired, lose your hair, or have other side effects. If you have any problems, please let us know.

**Support:**
You have a strong support system with your family. Keep doing things that make you happy, like exercising and eating healthy food.

If you have any questions, please talk to your care team. We are here to help you.

Best wishes,

[Your Name]  
Medical Communication Specialist  
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for an initial consultation regarding your newly diagnosed right breast cancer.
**What's new or changed since your last visit?**
You have been diagnosed with a grade 2 invasive ductal carcinoma (cancer that started in the milk ducts) in your right breast. The cancer is estrogen receptor positive (ER+), progesterone receptor positive (PR+), and does not have a protein called HER2. The cancer has spread to one lymph node under your right armpit but has not spread to other parts of your body.
**What treatment or medication changes were made?**
You will start neoadjuvant chemotherapy (treatment given before surgery to shrink the cancer) on May 1st, 2017. This will include weekly paclitaxel for 12 weeks followed by AC every 2-3 weeks with Neulasta support. After surgery, you will need hormone therapy, which includes suppressing your ovaries. You are also considering the use of cold caps during chemotherapy to reduce hair loss.
**What is the plan going forward?**
You will start neoadjuvant chemotherapy on May 1st, 2017. A port will be placed either on the start of therapy or one week later. You will have a mammogram and PET scan on May 1st, 2017. Labs will be done on Thursday. Mammoprint results are expected on Thursday of this week.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 11

**Cancer Type:** ER+/PR+ ductal carcinoma in situ (DCIS) with intermediate nuclear grade, solid and cribriform patterns, HER2: not tested

## Original Clinical Note

```
Subjective:       Patient ID: ***** ***** ***** ***** 51 y.o. female.    HPI  ***** ***** is a 51 y.o. female with a biopsy proven left breast DCIS comes to our clinic today for a discussion of potential adjuvant anti-hormonal therapy. Ms. ***** presented on 12/05/2016 for routine screening mammography at which time indeterminate calcifications in the lower inner left breast were identified and a recommendation of additional imaging was made. On 12/11/2016 left breast diagnostic mammography was performed and showed a mammogram that was extremely dense with 3 mm group of amorphous punctate round calcifications in the left inner breast middle depth visible only on the cc view and these were new since 2011 and were deemed suspicious and a stereotactic core biopsy was recommended and performed on 12/27/2016. The pathology from this is appended below:    Left breast December 28/  FINAL PATHOLOGIC DIAGNOSIS  Left breast, stereotactic core needle biopsy:  1. Ductal carcinoma in situ, intermediate nuclear grade, cribriform and  micropapillary patterns, with focal necrosis; see comment.  2. Microcalcifications in association with ductal carcinoma in situ.  >90% estrogen receptor positive, Progesterone receptor pending.      Based on this finding a lumpectomy was recommended and performed on 01/29/2017 the pathology of which is appended below:  Laterality: Left.  - Tumor site: Lower inner quadrant.    - Position: 7 o'clock.  - Invasive tumor type: None.  - Ductal carcinoma in situ (DCIS): Present.  - DCIS nuclear grade: Intermediate.  - DCIS architectural patterns: Solid and cribriform.  - Necrosis in DCIS: None.  - Ductal carcinoma in situ size: DCIS present in 5 contiguous slices  (December 26), spanning 1.8 cm.  - Microcalcifications: None.  - Resection margins for DCIS:   For Part A:     - Posterior margin: Positive (tumor is at ink, on slide A8).  Additional margin submitted in part B is <0.1 from the new margin.    - \n Anterior medial margin: Negative (tumor is <0.1 cm away, on  slide A7, A9, A10).    - Anterior lateral margin: Negative (tumor is >1 cm away, on slide  A6).    - Superior margin: Negative (tumor is >1 cm away).    - Inferior margin: Negative (tumor is >1 cm away).    Additional margin was submitted separately (Part B) and shows DCIS <0.1  cm from the new margin (black ink). Level sections were performed on  parts *****-***** to confirm the above diagnosis.    - Lobular carcinoma in situ: None.  - Non-neoplastic breast: Apocrine metaplasia, usual ductal hyperplasia,  fat necrosis.  - Lymph node status: N/A.  - AJCC/UICC stage: pTisNx.    - Tumor biomarker (ER/PR) status was positive.    At this time Ms. ***** is at her baseline. Her left breast is still somewhat sore subsequent to her surgery but she is in no severe pain. His weight is continuing to menstruate although she is having significant hot flashes and night sweats which have been going on about 9 months and perhaps are worsening. She has continues to have chronic low back pain but this is unchanged for her. Her appetite is good her activities are full her weight is stable and her bowel and bladder function are within normal limits               Patient Active Problem List   Diagnosis   \u0007 Migraine   \u0007 Chronic low back pain   \u0007 Right-sided low back pain without sciatica   \u0007 Current smoker   \u0007 High blood pressure   \u0007 Abnormal mammogram of left breast   \u0007 DCIS (ductal carcinoma in situ)            Past Medical History:   Diagnosis Date   \u0007 Anemia    \u0007 High blood pressure 11/27/2016             Past Surgical History:   Procedure Laterality Date   \u0007 OVARIAN CYST REMOVAL Bilateral     Age 27             Allergies   Allergen Reactions   \u0007 Penicillins Hives   \u0007 Hydrocodone-Acetaminophen Hives        Current Medications           Current Outpatient Prescriptions   Medication Sig Dispense Refill   \u0007 amLODIPine (NORVASC) 5 mg tablet Take 1 \n tablet (5 mg total) by mouth Daily. 30 tablet 3   \u0007 gabapentin (NEURONTIN) 300 mg capsule Take 1 capsule (300 mg total) by mouth 3 (three) times daily. 90 capsule 2   \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 1 tablet (500 mg total) by mouth every 6 (six) hours as needed for Pain or temp > 38.5 C. 120 tablet 6   \u0007 blood pressure monitor KIT Check blood pressure once per day 1 kit 0   \u0007 cetirizine (ZYRTEC) 10 mg tablet Take 1 tablet (10 mg total) by mouth Daily. 90 tablet prn   \u0007 cyclobenzaprine (FLEXERIL) 5 mg tablet TAKE 1 TABLET BY MOUTH EVERY 8 HOURS AS NEEDED FOR MUSCLE SPASMS 90 tablet 3   \u0007 ferrous sulfate 325 mg (65 mg elemental) tablet Take 1 tablet (325 mg total) by mouth daily with breakfast. 90 tablet 0   \u0007 heating pads PADS 1 Units by Misc.(Non-Drug; Combo Route) route 3 (three) times daily. 3 each 3   \u0007 lidocaine (XYLOCAINE) 5 % OINT ointment Apply thin layer (dime-sized amount) to right lower back once or twice daily as needed for pain 50 g 3   \u0007 naproxen (NAPROSYN) 500 mg tablet TAKE 1 TABLET BY MOUTH TWICE DAILY WITH MEALS 100 tablet 0     No current facility-administered medications for this visit.                   Family History   Problem Relation Age of Onset   \u0007 Hypertension Mother    \u0007 Arthritis Mother    \u0007 Depression Mother    \u0007 Migraines Mother    \u0007 Diabetes Father    \u0007 Hypertension Father                   Review of Systems  Constitutional: Negative. Negative for chills, diaphoresis, fever, malaise/fatigue and weight loss.   HENT: Negative for congestion, ear discharge, ear pain, hearing loss, nosebleeds, sore throat and tinnitus.   Eyes: Negative for blurred vision, double vision, photophobia, pain, discharge and redness.   Respiratory: Negative for cough, hemoptysis, sputum production, shortness of breath, wheezing and stridor.   Cardiovascular: Negative for chest pain, palpitations, orthopnea, claudication, leg swelling and PND.   Gastrointestinal: Negative. Negative \n for abdominal pain, blood in stool, constipation, diarrhea, heartburn, melena, nausea and vomiting.   Genitourinary: Negative for dysuria, flank pain, frequency, hematuria and urgency.   Musculoskeletal: Positive for back pain. Negative for falls, joint pain, myalgias and neck pain.   Ambulate with a cane. Right lower extremity weakness and chronic back pain after an injury.   Skin: Negative for itching and rash.   Neurological: Negative for dizziness, tingling, tremors, sensory change, speech change, focal weakness, seizures, loss of consciousness, weakness and headaches.   Endo/Heme/Allergies: Negative for environmental allergies and polydipsia. Does not bruise/bleed easily.   Psychiatric/Behavioral: Negative for depression, hallucinations, memory loss, substance abuse and suicidal ideas. The patient is not nervous/anxious and does not have insomnia.      Objective:      Physical Exam   Nursing note and vitals reviewed.  Constitutional: She is oriented to person, place, and time. She appears well-developed and well-nourished.   HENT:   Head: Normocephalic and atraumatic.   Eyes: EOM are normal. Pupils are equal, round, and reactive to light.   Neck: Normal range of motion. Neck supple.   Cardiovascular: Normal rate and regular rhythm.    Pulmonary/Chest: Effort normal and breath sounds normal.   Chest wall examination is significant for a left breast which is bandaged in the dependent portion and very tender to palpation. To gentle palpation there is no suspicious mass other than the expected postsurgical firmness. The left axilla is clear as is the left supraclavicular and left infraclavicular area. The right breast is without suspicious mass as is the right axilla right supraclavicular and right infraclavicular area   Abdominal: Soft.   Neurological: She is alert and oriented to person, place, and time.   Skin: Skin is warm and dry.   Psychiatric: She has a normal mood and affect. Her behavior is normal. \n Judgment and thought content normal.           Assessment:       I had a detailed discussion with Mrs. ***** about the nature of ductal carcinoma in situ and our general approach to treatment of ductal carcinoma in situ including the use of surgery and/or radiation to remove the growth from the breast so that it does not recur locally and the use of adjuvant systemic anti-hormonal therapy which also would help to decrease the risk of local recurrence. In addition I spoke about its primary preventative effect as well with the decrease in new breast cancer events of 50%. Since Ms. ***** is continuing to menstruate, although because she is having symptomatology of hot flashes and night sweats she may be in the perimenopausal period, I suggested that the drug of choice would be a tamoxifen if she were to contemplate anti-hormonal therapy. We went the mechanism of tamoxifen with both its pro-as well as antiestrogenic effects and its various side effects including exacerbation of postmenopausal symptomatology increases in hot flashes and night sweats, increase in clotting risk with increased risk of deep venous thrombosis as well as pulmonary embolus and stroke, and the increased risk of endometrial carcinoma. Ms. ***** seemed to understand all of these issues fairly well. She is currently pending an appointment with ***** ***** to assess whether or not she will be a candidate for radiation therapy at this time. I will send a prescription for tamoxifen into her pharmacy but have told her to hold off on taking this drug until after she has her radiation oncology assessment.      Plan:       return to clinic 4-6 weeks after beginning tamoxifen to assess therapy tolerance        \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment options. We know this can feel overwhelming, and we are here to support you.

You were diagnosed with ductal carcinoma in situ (DCIS) in your left breast. DCIS means abnormal cells are inside the milk ducts but have not spread into nearby breast tissue. This is an early form of breast cancer.

Your DCIS is estrogen receptor positive (ER+), which means the cells can grow with estrogen.

You had surgery (lumpectomy) to remove the area of DCIS. The size of the DCIS was about 1.8 cm.

Next steps in treatment:
Your care team discussed ways to lower the chance of the DCIS coming back:

Radiation therapy: uses high-energy rays to treat the breast after surgery
Hormone (anti-estrogen) therapy, such as tamoxifen. This medicine blocks estrogen and can:
Lower the chance of the same cancer coming back
Lower the chance of a new breast cancer in the future

Because you are still having menstrual periods, tamoxifen is the recommended option if you choose hormone therapy.

You will meet with a radiation specialist to decide if radiation is right for you. You should wait to start tamoxifen until after this visit.

It is normal to have some soreness after surgery, and this should improve with time.

Please talk with your care team about any questions or concerns. We are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Ms. *****,

I hope this letter finds you well. I wanted to share some important information about your recent health check-up and the next steps we will take together.

**Diagnosis and Stage:**
You have been diagnosed with ductal carcinoma in situ (DCIS). This means there are abnormal cells in your breast that haven't spread beyond the milk ducts. It's like a small growth inside your breast that needs to be treated to prevent it from getting bigger.

**Treatment Plan:**
We have already done a lumpectomy, which means we removed the growth from your left breast. Now, we need to decide on the best way to make sure it doesn’t come back. One option is to use medicine called tamoxifen. This medicine helps stop the growth from coming back by changing how your body uses hormones.

**Key Test Results:**
The tests show that the growth is in the lower part of your left breast. The doctors found that the growth is sensitive to hormones, which is why we think tamoxifen might work well for you.

**Next Steps:**
We will set up an appointment with a radiation doctor to see if you need radiation therapy. This is another way to make sure the growth doesn’t come back. We will give you a prescription for tamoxifen, but you should wait to start taking it until after you talk with the radiation doctor.

**What to Watch For:**
Please keep an eye out for any changes in how you feel. If you notice anything unusual, let us know right away. Also, continue to manage your chronic back pain and any other symptoms you have.

**Important Reminders:**
Remember to follow up with us in 4 to 6 weeks after starting tamoxifen. This will help us make sure the medicine is working well for you.

If you have any questions or concerns, please don’t hesitate to reach out to your care team. We are here to support you every step of the way.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a discussion about your left breast cancer treatment.
**What's new or changed since your last visit?**
You have a type of breast cancer called ductal carcinoma in situ (DCIS), which means the cancer is confined to the milk ducts and has not spread beyond them. The cancer is ER+ and PR+, meaning it responds to hormones. The cancer is located in the left breast and measures 1.8 cm. The edges of the removed tissue show that some cancer cells might still be present, so further treatment is needed.
**What treatment or medication changes were made?**
A prescription for tamoxifen has been sent to your pharmacy. Tamoxifen is a medication used to reduce the risk of the cancer coming back. You have been instructed to wait until after your radiation oncology assessment before starting the medication.
**What is the plan going forward?**
You are referred to radiation oncology to assess whether you will need radiation therapy. You will return to the clinic 4-6 weeks after starting tamoxifen to check how you are tolerating the medication.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 12

**Cancer Type:** ER+/PR+/HER2- grade 2 invasive mammary carcinoma with mixed ductal and lobular features

## Original Clinical Note

```
ID: ***** ***** ***** is a 67 y.o. postmenopausal patient with a recent diagnosis of early stage breast cancer, who presents in consultation to discuss treatment options and to establish care.    Date of Service: 03/08/17    HPI: The patient underwent a right breast partial mastectomy on 07/06/14 to remove a lipoma; she had no evidence of malignancy at that time. She also underwent an FNA in the right breast at the 9:00 position, 11 cm from the nipple on 11/09/14 which revealed benign breast tissue in association with scar and surgical site changes.  She had normal mammogram in June 2015 and July 2016.  She felt a mass in her right breast in late January and a subsequent evaluation led to a diagnosis of breast cancer.  Evaluation to date is as follows:    07/12/15 screening mammogram: heterogeneously dense breast tissue; no features of malignancy; right breast post-surgical scarring    07/16/16 screening mammogram: heterogeneously dense breast tissue; no mammographic features of malignancy     02/15/17 right diagnostic mammogram: irregular mass with spiculated margins in lower slightly outer right breast, posterior depth; heterogeneously dense breast tissue  Right breast ultrasound: irregular solid mass (2.4 x 1.5 x 1.5 cm) with spiculated margins in lower outer right breast at 7:00 position, 5 cm from nipple (at site of palpable lump); irregular solid mass (1.0 x 1.0 x 0.7 cm) with spiculated margins in lower outer right breast at 8:00 position, 5 cm from nipple (a 2nd smaller palpable lump); non-enlarged morphologically normal appearing LN in right axilla  right breast FNA at 7:00 position, 4 cm from nipple: adenocarcinoma    02/22/17  US-guided core biopsy of right breast mass at 7:00 position, 5 cm from nipple: 1.2 cm grade 2 invasive mammary carcinoma with mixed ductal and lobular features, ER+ >95%, PR+ ~70%, HER2- (IHC 0; FISH ratio 0.65), Ki-67 ~20%  US-guided right breast FNA at 8:00 position, 5 cm from   nipple: adenocarcinoma  US-guided right breast FNA at 7:00 position, 7 cm from nipple: adenocarcinoma    02/19/17 Breast MRI: irregularly shaped mass (2.2 x 2.3 x 2.0 cm) with spiculated margins in lower outer right breast (corresponding to 7:00 position mass on US and palpable-FNA guided biopsy proven carcinoma); in the lower outer right breast, slightly superior and lateral to biopsy proven carcinoma, an irregular enhancing mass (1.5 x 1.2 x 1.8 cm) (corresponding to 8:00 position mass on US); enhancing oval circumscribed mass (0.5 cm) located 1.5 cm directly posterior to 7:00 position mass with faint NME extending from posterior aspect of 7:00 position mass to this mass; no adenopathy or chest wall involvement; left breast unremarkable    02/20/17 PET/CT: no evidence of metastatic disease; 2 hypermetabolic enhancing nodules within right breast; sclerotic foci within left ileum, right pubic symphysis, and left ischium likely representing bone islands; mild uptake along distal esophagus (likely esophagitis)    The patient presents to clinic today with her husband.  She feels well and states that she is not surprised by her diagnosis as her paternal great aunts and her mother all suffered from breast cancer and that she has therefore always expected such a diagnosis.  She is active: swims a mile many times a week in the summer, walks a lot on a daily basis, and rides horses.    The patient's gynecologic history is as follows.  She underwent menarche at 12-13 years of age.  She started taking OCP at 16 years of age for very irregular periods at 16.  She is G3P2.  She used Clomid in 30s.  Her first pregnancy was at 30 years of age.  She believes she underwent menopause at 57.  Had nightsweats for years and hot flashes.  She has no history of HRT.      Past Medical History:   Diagnosis Date   \u0007 Breast cancer    \u0007 Fibrocystic breast disease    \u0007 Lipoma     (R) breast lipoma   \u0007 Melanoma in situ 2016       Past Surgical \n History:   Procedure Laterality Date   \u0007 Excisional Biopsy of Breast      (R) breast   \u0007 KNEE ARTHROSCOPY Left     plica syndrome, 12 years ago        Family History   Problem Relation Age of Onset   \u0007 Breast cancer Mother      early 50s, had unilateral mastectomy, no other treatment known   \u0007 Heart failure Mother      d. 85   \u0007 Breast cancer Other      Paternal Great Aunt, suspected   \u0007 Lymphoma Father      dxed 70s, required chemotherapy at 86, d. complications chemotherapy   \u0007 Lung cancer Brother      d. 56, owned dry cleaning business, 20 year pack history.       Social History     Social History   \u0007 Marital status: Married     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A     Occupational History   \u0007 Not on file.     Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   \u0007 Alcohol use 1.2 oz/week     2 Glasses of wine per week      Comment:      2-3 gl wine per week   \u0007 Drug use: No   \u0007 Sexual activity: Not on file     Other Topics Concern   \u0007 Not on file     Social History Narrative    Newspaper editor, plans on retiring soon.    Lives with her husband in *****       Outpatient Encounter Prescriptions as of 03/08/2017   Medication Sig Dispense Refill   \u0007 simvastatin (ZOCOR) 20 mg tablet      \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 tab po/sl 1 hr prior to scans.  July repeat x 1 at time of scan. (Patient not taking: Reported on 03/08/2017) 10 tablet 0     No facility-administered encounter medications on file as of 03/08/2017.        No Known Allergies      Review of Systems   Constitutional: Negative.    HENT: Negative.    Eyes: Negative.    Respiratory: Negative.    Cardiovascular: Negative.    Gastrointestinal: Negative.    Genitourinary: Negative.    Musculoskeletal: Negative.    Skin: Negative.    Neurological: Negative.    Endo/Heme/Allergies: Negative.    Psychiatric/Behavioral: Negative.         Physical Exam:   Vital Signs - BP 105/54  | \n Pulse 66  | Temp 36.1 C (97 F) (Oral)  | Resp 18  | Ht 156.4 cm (5' 1.58\") Comment: March 03@***** | Wt 54.5 kg (120 lb 1.6 oz)  | LMP 04/06/2008  | SpO2 96%  | BMI 22.27 kg/m    Constitutional - WDWN, NAD  Eyes - sclera anicteric, PERRL, EOMI  HENT - sinuses nontender, nasal mucosa intact, pharynx without erythema, stomatitis, or thrush   Lymph Nodes - no cervical, supraclavicular, or axillary lymphadenopathy  Respiratory - resonant to percussion throughout, CTA bilaterally; no wheezes, rhonchi, or crackles  Cardiovascular - Normal heart rate, normal rhythm, no murmurs, no edema  Breast - right breast with mass (3 x 4 cm) in lower outer quadrant; left breast without palpable masses  GI - Bowel sounds normal, soft; no tenderness, no distention, no HSM  Musculoskeletal - No tenderness over bones or joints.   Neurologic - Alert & oriented x 3, ambulates w/o difficulty, good strength throughout  Psychiatric - Mood stable; no HI, SI, hallucinations, paranoia, or delusions  Skin - warm without rashes or lesions     Studies:  Available labs, pathology, and imaging were reviewed and independently interpreted, as described above in the HPI.  Lab Results   Component Value Date    WBC Count 5.6 10/22/2016    Hemoglobin 12.6 10/22/2016    Hematocrit 38.4 10/22/2016    Platelet Count 202 10/22/2016     Lab Results   Component Value Date    Sodium, Serum / Plasma 139 10/22/2016    Potassium, Serum / Plasma 4.2 10/22/2016    Chloride, Serum / Plasma 106 10/22/2016    Carbon Dioxide, Total 25 10/22/2016    Urea Nitrogen, Serum / Plasma 18 10/22/2016    Creatinine 0.76 10/22/2016    Glucose, non-fasting 120 10/22/2016     Lab Results   Component Value Date    Calcium, total, Serum / Plasma 9.2 10/22/2016       Assessment and Recommendations:  67 y.o. postmenopausal patient with a clinical stage II HR+/HER2- IDC of the right breast (MP high risk), who should receive neoadjuvant or adjuvant chemotherapy.    I discussed with the patient \n and her husband in great detail the natural history and treatment of a MP high risk, clinically high risk, node-negative HR positive breast cancer.  We reviewed the pathology report and discussed the multimodality approach of breast cancer treatment.  I encouraged her that she currently has no evidence of metastatic disease and that we can therefore treat her with curative intent.  The patient understands that her treatment will include chemotherapy, surgery, hormonal therapy, and possibly radiation.  We discussed the advantages of offering chemotherapy in the neoadjuvant setting in order to assess response to treatment, and to increase the chances of breast conserving surgery.  I discussed the chemotherapy regimens that would be recommended as the standard of care, and discussed how these regimens would be different in the context of the I-SPY2 clinical trial.   The patient was screened for ISPY2 but has already decided that she does not want to participate.      I discussed the regimens typically used for the treatment of HR+ breast cancer when chemotherapy is recommended.  We discussed treatment with AC/T (or T/AC) with 4 cycles of doxorubicin plus cyclophosphamide (AC) every 2 weeks followed by 12 cycles of weekly paclitaxel.  We also discussed the regimen of docetaxel plus cyclophosphamide (TC), given every 3 weeks for 4 to 6 cycles.  I referred to data from the Anthracyclines in Breast Cancer clinical trial in which patients with node-negative HR+/HER2- breast cancer did not gain additional benefit from taxane/anthracyline regimens over 6 cycles of TC.  I then explained that we do not generally give TC as a neoadjuvant regimen there is no certainty of node-negativity until after surgery and if a patient were to have evidence of positive LN after receiving TC there would be concern that they had not received an anthracycline.    The patient, however, prefers to receive chemotherapy upfront (in light that chemo \n is recommended) and therefore prefers to proceed with AC/T.  She is aiming for a lumpectomy and at this time would only be able to have a mastectomy.  I specifically discussed with the patient the typical and less common side-effects associated with AC/T, such as fatigue, neuropathy, skin and nail changes, myalgias, alopecia, myelosuppression, nausea).  I stated that patients who receive AC, are at increased risk of cardiotoxicity (acute and distant, 1-3%), which can result in asymptomatic or symptomatic heart failure with dyspnea and chest pain.  I also explained that AC is associated with an increased risk (1-3%) of secondary leukemias.    After an extensive discussion, the patient states that she understands the rationale for neoadjuvant treatment and will proceed with neoadjuvant chemotherapy, specifically AC/T. In order to start treatment, she will first require chemotherapy teaching as well as an echocardiogram.  She also stated interest in cold caps and we will have the CRC contact her to share specific details.   We will additionally have a port placed, which was explained to her and to which she has agreed.    Recommendations in Brief:  - echocardiogram  - chemo teach  - port placement  - discussion with cold cap CRC  - prior auth  - plan to start with AC, ideally first week of March (March 19 or March 20)    The patient will return to clinic in 2 weeks.    I spent a total of 100 minutes face-to-face with the patient and 95 minutes of that time was spent counseling regarding the diagnosis, the treatment plan, the prognosis, medication risks, lifestyle modification, the risks and benefits of surgery, symptoms and therapeutic options.  \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment plan. We know this is a lot of information, and we are here to support you.

You have breast cancer in the right breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and has spread into nearby breast tissue. Your cancer is hormone receptor positive (ER+/PR+), meaning it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

Your cancer is early stage (Stage II) and has not spread to other parts of the body, which is good news.

Treatment plan:
Your care team recommends a multi-step (combined) treatment approach:

Chemotherapy (medicine to kill cancer cells), given first before surgery
Surgery to remove the tumor
Hormone therapy (medicine to block estrogen and reduce the risk of cancer coming back)
Possible radiation therapy, depending on surgical results

You have chosen to start with chemotherapy before surgery (called neoadjuvant therapy). This can help shrink the tumor and may allow for less extensive surgery.

Next steps before treatment:

Echocardiogram (heart test)
Chemotherapy teaching session
Placement of a port (a small device under the skin to give treatment)

We understand this is a difficult time, but your cancer is being treated with the goal of cure.

Please ask your care team any questions you have. We are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to our clinic today. I am writing to summarize what we discussed about your health.

**Diagnosis:**
You have been diagnosed with early-stage breast cancer. This means the cancer is still in the breast and has not spread to other parts of your body. Your cancer is called "stage II" because it is a bit bigger than very early cancers.

**Treatment Plan:**
We talked about several steps to treat your cancer:

1. **Chemotherapy:** This is medicine that helps kill cancer cells. We will start with a type of chemotherapy called AC/T. It will help make the tumor smaller so that a lumpectomy (removing just the tumor) might be possible instead of removing the whole breast.
   
2. **Surgery:** After the chemotherapy, we will do surgery to remove the tumor. We hope to do a lumpectomy, but we will decide based on how well the chemotherapy works.

3. **Hormonal Therapy:** Since your cancer is sensitive to hormones, you will take medicine to block these hormones after the surgery.

4. **Radiation:** Depending on the results of your surgery, you may need radiation to make sure all the cancer is gone.

**Key Test Results:**
- A special X-ray (mammogram) showed a lump in your right breast.
- An ultrasound found two lumps in your right breast.
- A small sample (biopsy) confirmed that the lumps are cancerous.

**Next Steps:**
- You will have an echocardiogram (heart test) to check your heart before starting chemotherapy.
- We will place a port (a small tube under your skin) to make it easier to give you the chemotherapy.
- We will talk more about using cold caps to try to prevent hair loss during chemotherapy.

**What to Watch For:**
- Let us know if you feel very tired, have any pain, or notice any changes in your skin or nails.
- If you have any questions or concerns, please talk to your care team.

Remember, you are not alone in this journey. We are here to support you every step of the way.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

Please discuss any questions you have with your care team.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation to discuss treatment options for your recently diagnosed breast cancer.
**What's new or changed since your last visit?**
You were diagnosed with an early stage breast cancer. The cancer is a type that started in the milk-producing glands and milk ducts (mixed ductal and lobular features). It measures 1.2 cm and is grade 2. The cancer is sensitive to estrogen and progesterone (ER+ and PR+) but does not have a protein called HER2. Imaging shows two small masses in your right breast, but there is no evidence of the cancer spreading to other parts of your body.
**What treatment or medication changes were made?**
You will start neoadjuvant chemotherapy with a chemotherapy regimen (doxorubicin plus cyclophosphamide every 2 weeks for 4 cycles followed by weekly paclitaxel for 12 cycles).
**What is the plan going forward?**
Before starting chemotherapy, you will have an echocardiogram (a heart ultrasound) to check your heart function. You will also have a port placed to make it easier to receive chemotherapy. You will return to clinic in 2 weeks for a follow-up visit.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 13

**Cancer Type:** ER+/PR+/HER2- invasive ductal carcinoma

## Original Clinical Note

```
HPI:  ***** ***** is a 36 y.o. female with a recently diagnosed hormone receptor positive/HER2 negative node positive right breast cancer. She is scheduled to proceed to surgery on 11/05/17 and is here to discuss treatment options and recent imaging findings.    ***** underwent a mammogram at an outside facility (*****) on 09/24/17 which showed an 1.8 cm irregular hyperdense mass in the right breast. There was no mammographic evidence of malignancy in left breast. US revealed a 1.7 cm irregular hypoechoic mass in the 11:00 position along with 2 prominent axillary LNs. US guided core biopsy of the breast on 09/30/17 revealed invasive ductal carcinoma (ER+ (80-90%), PR+ (80-90%), HER2 by IHC & FISH negative;  Ki67% 20-30%) Histologic grading was not reported.     She sought a surgical opinion at ***** and saw Dr. ***** on 10/02/17. An MRI of breast performed on 10/03/17 at the ***** showed a 2.2 cm mass in the upper central right breast and a 0.4 cm anterior satellite lesion. There were also findings suspicious for additional ductal disease, disease within the nipple and prominent right axillary LNs.    There was some delay in the outside MRI becoming available for Dr. *****'s review but eventually the enlarged lymph nodes were noted to be suspicious and an axillary LN biopsy was scheduled. In the meantime, the patient was seen by Dr. ***** in ***** ***** and the patient underwent an egg harvest. This resulted in only 3 fertilized embryos, so ***** has interest in trying another egg harvest at some point.    ***** underwent an axillary FNA on 10/28/17 and the node was positive for metastatic cancer. PET/CT performed on 10/31/17 revealed a hypermetabolic right breast mass (2.1 x 1.6 cm) and an enlarged right axillary lymph node. There were other subcentimeter, asymmetric right subpectoral lymph nodes with low-level uptake. There was also a hyperdense 0.7 cm falx lesion that could be a partially calcified \n falcine meningioma or dural metastasis. She underwent an MRI brain today to exclude a dural metastasis. The MRI showed a 0.5 cm right parafalcine dural-based mass most likely a meningioma. I spoke with the radiologist to find out if this lesion was amenable to biopsy and the neuroradiologist thought a biopsy could be difficult and morbid and their level of suspicion for this lesion being metastatic disease is low.    She is scheduled to undergo a partial mastectomy and ALND with Dr ***** on 11/05/17. She is here for planning of subsequent steps of treatment and to discuss her treatment options. She is also concerned about whether she needs to take time off from work during her treatment especially during chemotherapy.     Patient Active Problem List    Diagnosis Date Noted   \u0007 Breast cancer 11/03/2017     09/24/17: Mammogram at outside facility (*****): 1.8 cm irregular hyperdense mass in the the right breast at 11:00 with indeterminate prominent axillary LN. No mammographic evidence of malignancy in left breast.   09/24/17: US breast: Right breast US shows 1.8x1.4x1.7 cm irregular hypoechoic mass in the 11:00 position. 2 non enlarged prominent axillary LN.   09/30/17: US guided core biopsy: IDC, ER+ (80-90%), PR+ (80-90%), HER2 by IHC & FISH negative. Ki67% 20-30%.   10/03/17: MRI breast: Dominant biopsy-proven malignancy, upper central right breast measuring 2.2 cm with 0.4 cm anterior satellite lesion. Also, findings suspicious for additional ductal  disease and disease within the nipple. Prominent right axillary LN.  Left breast: Negative  10/28/17: US axilla right: There is an enlarged LN measuring 0.7 x 0.7 cm in the right axilla, at approximately 10:00, 14 cm from nipple. Biopsy findings: Metastatic carcinoma  10/31/17: PET/CT: Hypermetabolic right breast mass and axillary lymph node consistent with known metastatic IDC. Other subcentimeter, asymmetric right subpectoral lymph nodes with low-level uptake could represent \n additional sites of metastatic disease. Hyperdense 0.7 cm falx lesion  may represent a partially calcified falcine meningioma, however dural metastasis is a differential consideration. Recommend brain MRI for further evaluation.Hypermetabolic right breast mass measures 2.1 x 1.6 cm.Mildly hypermetabolic right axillary LN measures 2.8 cm.  11/03/17: MRI brain: 5 mm right parafalcine dural-based mass is most likely a meningioma, although dural-based metastasis remains an unlikely possibility.  11/05/17: Partial mastectomy planned: Dr ***** and Dr *****.      \u0007 Encounter for fertility preservation procedure 10/20/2017     Added automatically from request for surgery *****         Past Medical History:   Diagnosis Date   \u0007 Breast cancer    \u0007 Hypothyroidism      Past Surgical History:   Procedure Laterality Date   \u0007 CESAREAN SECTION  2015   \u0007 egg retrieval         Allergies/Contraindications  No Known Allergies      Current Outpatient Prescriptions:   \u0007  cholecalciferol, vitamin D3, (VITAMIN D3 ORAL), Take by mouth., Disp: , Rfl:   \u0007  multivitamin tablet, Take 1 tablet by mouth Daily., Disp: , Rfl:   \u0007  SYNTHROID 112 mcg tablet, Take 112 mcg by mouth Daily. , Disp: , Rfl:   \u0007  acetaminophen (TYLENOL) 500 mg tablet, Take 2 tablets (1,000 mg total) by mouth every 8 (eight) hours as needed for Pain. (Patient not taking: Reported on 11/06/2017), Disp: 100 tablet, Rfl: 1  \u0007  docusate sodium (COLACE) 100 mg capsule, Take 1 capsule (100 mg total) by mouth Twice a day. While taking oxycodone to prevent constipation (Patient not taking: Reported on 11/06/2017), Disp: 30 capsule, Rfl: 1  \u0007  HYDROcodone-acetaminophen (NORCO) 5-325 mg tablet, Take 1 tablet by mouth every 4 (four) hours as needed for Pain. (Patient not taking: Reported on 11/06/2017), Disp: 26 tablet, Rfl: 0  \u0007  ibuprofen (ADVIL,MOTRIN) 600 mg tablet, Take 1 tablet (600 mg total) by mouth every 8 (eight) hours as needed for Pain. (Patient not taking: Reported on \n 11/06/2017), Disp: 60 tablet, Rfl: 1  \u0007  LORazepam (ATIVAN) 0.5 mg tablet, Take 0.5 mg by mouth every 6 (six) hours as needed for Anxiety., Disp: , Rfl:   \u0007  oxyCODONE (ROXICODONE) 5 mg tablet, Take 1-2 tablets (5-10 mg total) by mouth every 6 (six) hours as needed for Pain. (Patient not taking: Reported on 11/06/2017), Disp: 20 tablet, Rfl: 0  No current facility-administered medications for this visit.     Gynecologic history: Patient is G1P1. She has 4 yrs use of oral contraceptives. Menses were regular up to the diagnosis of breast cancer.    Social History     Social History Narrative    She lives with her husband, child and extended family in ***** *****. She work in tech.       History   Smoking Status   \u0007 Never Smoker   Smokeless Tobacco   \u0007 Never Used       History   Alcohol use Not on file       Family History   Problem Relation Name Age of Onset   \u0007 Breast cancer Cousin           Review of Systems -  Patient denies sore throat, nasal congestion, vision changes, cough, chest pain, dyspnea, nausea, vomiting, abdominal pain, headaches, fevers, and change in bladder or bowel function. Remaining systems reviewed and are negative except that outlined above    Physical Examination: General appearance - alert, well appearing, and in no distress  BP 99/68  | Pulse 65  | Temp 36.1 C (96.9 F) (Oral)  | Resp 18  | Ht 157.5 cm (5' 2\")  | Wt 77 kg (169 lb 11.2 oz)  | LMP 10/13/2017  | SpO2 96%  | BMI 31.04 kg/m   Mental status - alert, oriented to person, place, and time  Eyes - pupils equal and reactive, extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - no palpable lymphadenopathy, no hepatosplenomegaly  Chest - clear to auscultation, no wheezes, symmetric air entry  Heart - normal rate, regular rhythm, no murmurs  Abdomen - soft, nontender, nondistended, no masses or organomegaly  Breasts - breast exam not done, \n no axillary LN palpable.   Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema  Skin - no rashes, no suspicious skin lesions noted    Labs reviewed and discussed with patient:  Results for orders placed or performed during the hospital encounter of 10/31/17   POCT glucose, Fingerstick   Result Value Ref Range    Glucose, meter download 93 70 - 199 mg/dL       Imaging results reviewed and discussed with patient:  Mr Brain With And Without Contrast    Result Date: 11/03/2017  MR BRAIN WITH AND WITHOUT CONTRAST : *****/*****/***** *****:***** PM CLINICAL HISTORY: Invasive ductal breast carcinoma with axillary lymph node metastasis. COMPARISON: PET/CT 10/31/2017. TECHNIQUE: Multiple sequences through the brain were acquired at 3 Tesla. CONTRAST MEDIA: 7.5 ml gadobutrol (Gadavist) IV. FINDINGS: Homogenously-enhancing, partially calcified 5 mm right parafalcine extra-axial mass with dural attachment. Anteriorly, a dural calcification of the anterior falx does not demonstrate abnormal enhancement. Normal brain parenchyma and signal intensity. No abnormal enhancement of the brain. Heterogenous skull base and calvarial bone marrow without evidence of focal lesion. No abnormal reduced diffusion within the calvarium. Normal flow-related signal voids. Left maxillary sinus mucus retention cyst.     5 mm right parafalcine dural-based mass is most likely a meningioma, although dural-based metastasis remains an unlikely possibility; follow-up imaging in 6 months may be helpful.      Petct Limited Whole Body Vertex To Mid Thigh    Result Date: 10/31/2017  PET/CT Whole Body (vertex to thighs)    *****/*****/***** *****:***** PM COMPARISON (including at least the prior year or since last change in therapy):   None REASON FOR THE \n STUDY:   Staging. CLINICAL HISTORY:   36 years old Female with right invasive ductal carcinoma metastatic to axillary lymph nodes, evaluate for distant metastatic disease.  CURRENT THERAPY AND DATE INITIATED: None TECHNIQUE:   Patient's fasting time was > 6 hours.  Following intravenous administration of 9.1 mCi of F18-FDG, a contrast enhanced, diagnostic CT was performed. This was followed by an emission PET scan started 60 minutes after FDG injection.  PET images were corrected for attenuation using the CT transmission data.  PET/CT was acquired from vertex to thighs. A rotating 3D MIP, as well as axial, coronal, and sagittal PET images with and without attenuation correction was interpreted.  Acquired and fused PET/CT images were reviewed alongside the PET images.  Patient's random blood glucose at the time of FDG injection was 93 mg/dL.  RADIATION DOSE INDICATORS: 3 exposure event(s), CTDIvol:  2.9 - 10.3 mGy. DLP: 1016 mGy-cm. FINDINGS: For reference, the cerebellar SUVmax is 13.2. Oncologic Findings: Index lesions: 1. Hypermetabolic right breast mass measures 2.1 x 1.6 cm, SUV max 7.7 (series 201, image 81) 2. Mildly hypermetabolic right axillary lymph node measures 2.8 cm, SUV max 2.9 (series 201, image 82) Lymph nodes: Additional prominent, asymmetric right subpectoral lymph nodes with low-level radiotracer uptake. Lungs: No metastases. Abdominal and pelvic organs: No metastases. Bones: Hyperdense 7 mm lesion along the falx projecting rightward with possible dural tail (series 201, image 13). Inferior and anterior to this there is an 8 mm calcified falcine meningioma. Other: No metastases. Non-oncologic findings: Brain:   Unremarkable, symmetric, FDG uptake is seen throughout the cortical gray matter, basal ganglia and the cerebellum.  No mass effect.  While these images appear within normal limits, MRI is recommended to rule out intracranial and/or skull base metastases if clinically indicated. Neck:   \n Otherwise unremarkable. Chest:   Otherwise unremarkable. Abdomen/Pelvis: Trace amount of physiologic pelvic free fluid. Small left ovarian cysts, likely physiologic. Otherwise unremarkable. Musculoskeletal:   Otherwise unremarkable.     1. Hypermetabolic right breast mass and axillary lymph node consistent with known metastatic invasive ductal carcinoma. Other subcentimeter, asymmetric right subpectoral lymph nodes with low-level uptake could represent additional sites of metastatic disease. 2. Hyperdense 7 mm falx lesion is favored to represent a partially calcified falcine meningioma, however dural metastasis is a differential consideration. Recommend brain MRI for further evaluation. This study was reviewed by Dr. ***** ***** of Nuclear Medicine. //ALERT// 2 Report dictated by: ***** *****, MD MS, signed by: ***** ***** *****, MD PhD Department of Radiology and Biomedical Imaging    Place ***** Clip ***** Core, Rt    Addendum Date: 10/30/2017    ===ADDENDUM===Cytology is now available for review of the right axillary lymph node FNA, and demonstrates metastatic carcinoma. Please see pathology report for full details. These malignant results are concordant with imaging findings. Recommend continued surgical/oncologic management for known right breast cancer. Per patient request, Radiology Dr ***** ***** notified the patient of these malignant results at 1535 on 10/29/17. Report dictated by: ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging    Result Date: 10/30/2017  Ultrasound Guided Fine Needle Aspiration: *****/*****/***** 11:53 AM COMPARISON:    Diagnostic right ultrasound 10/28/17 and outside hospital 09/24/17 CLINICAL HISTORY: 36 year old woman with recent diagnosis right breast IDC from biopsy at an outside hospital. Enlarged right axillary lymph node on ultrasound. TECHNIQUE: Procedure was explained in detail to patient.  Risks (including risks of pain, infection, bleeding, non-recovery of \n targeted lesion, possible need for further intervention for diagnostic purposes), intent to place post-biopsy marking clip, benefits, and alternatives discussed.  All questions were answered.   Written and verbal informed consent obtained.  Under sterile conditions, the enlarged lymph node was re-identified at the 10 o'clock location, 14 cm from the nipple in the posterior right depth  right axilla. Following administration of 1% Xylocaine without epinephrine, a 20G needle was placed under sonographic guidance with its tip just within the lesion. The cytopathologist then inserted a 25G  needle through this first needle with multiple to-and-fro motions. This insertion was repeated 2 more times. The  material was preliminarily reviewed with a microscope by the cytopathologist on site who deemed the specimen adequate.  Material will be further studied. A post biopsy marking clip was placed with confirmation of its deployment. Subsequent mammogram demonstrates the coil shaped Hydromark clip to be in the right axilla. Also noted is the previously biopsied right breast malignant mass with a clip. The patient tolerated the procedure well and left the department in satisfactory conditio
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and the next steps in your care. We know this is a lot to take in, and we are here to support you.

You have breast cancer in the right breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and spread into nearby tissue. Your cancer is hormone receptor positive (ER+/PR+), which means it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

Tests show a tumor about 2 cm in size in your right breast. The cancer has also spread to nearby lymph nodes (small glands that help fight infection).

Good news: your scans did not show cancer spread to other organs. A small spot in the brain was seen, but it is most likely a meningioma, which is usually a benign (non-cancer) growth.

Next steps:

You are scheduled for surgery (partial mastectomy and lymph node removal)
After surgery, your care team will plan further treatment

Treatment plan after surgery may include:

Chemotherapy (medicine to kill cancer cells)
Hormone therapy (medicine to block estrogen and reduce the chance of cancer coming back)
Possibly radiation therapy, depending on final results

You have also completed egg harvesting to help preserve fertility, which is an important step before treatment.

We understand you may be concerned about work and daily life during treatment. Please discuss this with your care team so we can help you plan ahead.

You are not alone. Please ask any questions you have—we are here to help you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's First Name],

Thank you for coming to our clinic. I want to share with you the important information about your health and the next steps in your treatment.

**Diagnosis:**
You have been diagnosed with breast cancer. It is a type called hormone receptor positive and HER2 negative. This means the cancer cells grow in response to hormones like estrogen and progesterone, but they do not have too much of a protein called HER2.

The cancer is in your right breast and has spread to some lymph nodes in your armpit. This is called node-positive breast cancer.

**Treatment Plan:**
On November 5th, you will have surgery to remove part of your breast and some lymph nodes. This is called a partial mastectomy and axillary lymph node dissection (ALND).

After surgery, we will decide on the best treatment for you. This might include chemotherapy and hormone therapy. Chemotherapy uses medicines to kill cancer cells. Hormone therapy stops the hormones that make the cancer grow.

We will talk more about these treatments after your surgery. We will also look at your test results to decide the best plan for you.

**Key Test Results:**
- A mammogram showed a small lump in your right breast.
- An ultrasound found a small lump and some swollen lymph nodes.
- A biopsy confirmed that the lump is cancer.
- A PET/CT scan showed the cancer in your breast and lymph nodes.
- An MRI of your brain showed a small growth that is likely not cancer.

**Next Steps:**
- You will have surgery on November 5th.
- After surgery, we will review your test results and decide on the next steps.
- If you need chemotherapy, we will talk about how to manage side effects like feeling tired and losing hair.

**What to Watch For:**
- Any new lumps or changes in your breasts.
- Feeling very tired or having other unusual symptoms.

Please remember to ask your care team any questions you have. They are here to help you understand everything about your treatment.

Take care and stay strong.

Sincerely,

[Your Name]
Medical Communication Specialist
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consultation regarding your recently diagnosed breast cancer.
**What's new or changed since your last visit?**
You have a type of breast cancer called invasive ductal carcinoma, which is hormone receptor positive and does not have a protein called HER2. The cancer has spread to the lymph nodes in your right armpit but has not spread to other parts of your body. Imaging tests show a tumor in your right breast and enlarged lymph nodes. A biopsy of the lymph node confirmed that the cancer had spread there.
**What treatment or medication changes were made?**
You will start hormonal therapy after radiation. To help preserve your hair during chemotherapy, you will use scalp cooling with a Dignicap or Penguin Cold Cap.
**What is the plan going forward?**
You are scheduled to undergo surgery on November 5th. After surgery, you will have a test called Mammaprint to help decide the best treatment plan. You will start hormonal therapy after radiation. We also discussed the possibility of participating in clinical trials of CDK 4/6 inhibitors such as PALLAS and JPBE. You are referred to have a brain MRI. Your next visit will be after surgery.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 14

**Cancer Type:** ER+/PR+/HER2 equivocal grade 1 invasive ductal carcinoma

## Original Clinical Note

```
HPI:  ***** ***** is a 34 y.o. female here in consultation to discuss treatment options for her newly diagnosed breast cancer whose history is as follows:    Patient Active Problem List    Diagnosis Date Noted   \u0007 Breast Cancer Right 04/25/2019     December 2018 self-palpated right breast mass  03/05/2019 Bilateral breast US (***** ***** Hospital in *****): 6x7 mm mass with ductal extension at 1 o'clock, likely DCIS.   03/17/2019 Gynecology at ***** for lower abdomen swelling, urethral discomfort and breast mass in the breast US (*****). On exam 1 cm firm, mobile, nontender mass approximately 5 cm from nipple at 1 o'clock.  03/17/2019: Right breast FNA at 1 o'clock (*****): Benign and could be seen in the setting of fibrocystic change or fibroadenoma.  03/17/2018 Abdomen-pelvic CT: No acute findings (for abdominal pain).  03/28/2019 Bilateral breast diagnostic mammogram (*****): A focal asymmetry is suggested in the region. Triangular marker denotes site of focal pain in the lower slightly outer right breast, with no underlying mammographic correlate. Scattered and occasionally grouped benign-appearing microcalcifications, both breasts, predominantly coarse dystrophic, likely the sequelae of reported fat grafting.  03/28/2019 Bilateral breast diagnostic US(*****): 22 x 10 x 15 mm oval complex solid and cystic mass with peripheral echogenicity and an irregular, antiparallel hypoechoic central component in the right upper inner quadrant, 1:00, 8 cm from nipple. The central hypoechoic component measures up to 10 mm. No sonographic correlate to site of focal pain in the lower outer right breast at 7 o'clock, 5 cm from nipple. No sonographic correlate to clinician's site of concern in the left axilla. A few normal-appearing nodes are identified. Left breast: benign  03/30/2019 Right breast US guided core biopsy at 1 o'clock 8 cm from nipple (UCSF): Grade 1 IDC, ER+(>95%), PR(~90% ranging from weak to strong nuclear staining), HER2 \n equivocal(IHC 2), FISH non-amplified (0.55 cm largest dimension in a core; multiple core involved), Ki-67: ~10%. Focal atypical ductal hyperplasia. Pseudoangiomatous stromal hyperplasia. DCIS not present.  04/24/2019 Dr. ***** at *****. genetics referral   04/24/2019 Dr. ***** at *****: discussed bilateral TSSM with expander/implant reconstruction  04/25/2019 Dr. ***** (Surgeon) discussed surgical options lumpectomy vs bilateral mastectomy, recommended genetic testing.  04/25/2019 Myriad Genetic Result: Negative  05/08/2019 Dr. ***** (***** *****) planned double mastectomy with reconstruction with Dr. ***** ***** (*****)  05/24/2019 Mammaprint (Agendia): Low risk: +0.287     \u0007 Family history of Alzheimer's disease 04/25/2019   \u0007 Bipolar 2 disorder (HCC) 04/04/2019   \u0007 Insomnia 04/04/2019     Pathology report from 03/30/19 shows she has ER+/PR+ and HER2 equivocal IDC of the right breast. Recently had a bilateral breast MRI on April 24.    Notes she has been drinking more wine recently due to the Shelter-in-Place orders, 1-3 glasses 5x a week.     She is planning on attending ***** ***** at ***** *****.      Past Medical History:   Diagnosis Date   \u0007 Allergy 2010    Seasonal   \u0007 Anxiety    \u0007 Bilateral ovarian cysts    \u0007 Bipolar II disorder (HCC)     diagnosed mid 20's, controlled on lamictal   \u0007 Hemorrhoids 2012   \u0007 Invasive ductal carcinoma of right breast in female (HCC) 04/04/2019    Invasive Ductal carcinoma     Past Surgical History:   Procedure Laterality Date   \u0007 BLEPHAROPLASTY Bilateral 11/14/2018   \u0007 BREAST BIOPSY     \u0007 Fat transfer  2019    breast and face   \u0007 labialplasty  2015   \u0007 LIPOSUCTION  2015    abd, thighs, flanks, buttocks, trunk - fat transfer to breast/fat and sculpting   \u0007 REFRACTIVE SURGERY Bilateral      Family History   Problem Relation Name Age of Onset   \u0007 Rashes / Skin problems Maternal Aunt     \u0007 Hypertension Mother     \u0007 Other (smoking) Mother     \u0007 Hypertension Maternal Uncle     \u0007 Diabetes \n Paternal Uncle     \u0007 Hypertension Paternal Uncle     \u0007 Rheum arthritis Maternal Grandmother     \u0007 Diabetes Maternal Grandmother     \u0007 Hypertension Maternal Grandmother     \u0007 Colon cancer Maternal Grandmother          dx L60's to E70's   \u0007 Diabetes Maternal Grandfather     \u0007 Heart disease Maternal Grandfather     \u0007 Hypertension Maternal Grandfather     \u0007 Migraines Paternal Grandmother     \u0007 Diabetes Paternal Grandfather     \u0007 Alzheimer's disease Paternal Grandfather     \u0007 Alzheimer's disease Paternal Great-Grandfather     \u0007 Breast cancer Maternal Aunt ***** ***** 40        Lobular carcinoma, 40 bilateral (in situ)   \u0007 Breast cancer Maternal Aunt ***** ***** 55        DCIS, 55   \u0007 Other (smoker) Father     \u0007 Melanoma Neg Hx     \u0007 Basal cell carcinoma Neg Hx     \u0007 Squamous cell carcinoma Neg Hx     \u0007 Skin ca. unk/oth Neg Hx       Allergies/Contraindications   Allergen Reactions   \u0007 Latex Itching     IRRITATION  Sensitivity- causes tender skin and skin irritation   \u0007 Aripiprazole      Suicidal ideation  depression  Suicidal ideation   \u0007 Lurasidone      Suicidal ideation       Social History     Patient does not qualify to have social determinant information on file (likely too young).   Social History Narrative    Travelled for work for ***** as a ***** and ***** *****, then came back to ***** with the COVID pandemic    Had been planning to go to ***** school in *****    Now plans to complete her undergraduate school in the ***** *****    Also has worked as a ***** ***** ***** and *****  *****    Lives with her aunt and uncle in ***** ***** ***** right now    OCPs for 5 years, stopped with diagnosis    G1TAB1         Current Outpatient Medications:   \u0007  ascorbic acid, vitamin C, 500 mg TABSR, Take by mouth daily., Disp: , Rfl:   \u0007  B-complex with vitamin C tablet, Take by mouth daily., Disp: , Rfl:   \u0007  clonazePAM (KLONOPIN) 1 mg tablet, Take 1 tablet (1 mg total) by mouth nightly as needed for Anxiety \n (Insomnia)., Disp: 5 tablet, Rfl: 3  \u0007  KAVA ROOT ORAL, Take by mouth. Kava tincture-  Use 1/2 -1 teaspoon by mouth Kava for anxiety, Disp: , Rfl:   \u0007  Lactobacillus acidophilus (PROBIOTIC ORAL), Take 1 capsule by mouth. PROBIOTIC-10PO, Disp: , Rfl:   \u0007  lamoTRIgine (LAMICTAL XR) 200 mg 24 hr ER tablet, Take 1 tablet (200 mg total) by mouth daily., Disp: 30 tablet, Rfl: 0  \u0007  multivitamin capsule, Take 1 capsule by mouth daily. Multiple Vitamin, Disp: , Rfl:   \u0007  multivitamin-Ca-iron-minerals TAB, Take 1 tablet by mouth., Disp: , Rfl:   \u0007  tretinoin microspheres (RETIN-A MICRO PUMP) 0.08 % GlwP, Apply 1 Application topically Daily., Disp: 50 g, Rfl: 5  \u0007  TURMERIC ORAL, Take 1 capsule by mouth. TURMERIC PO take 1 cap by mouth daily, Disp: , Rfl:   \u0007  UNABLE TO FIND, Med Name: ELDERBERRY PO take by mouth daily., Disp: , Rfl:   \u0007  zolpidem (AMBIEN) 5 mg tablet, Take 1 tablet (5 mg total) by mouth nightly as needed for Sleep, Disp: 5 tablet, Rfl: 0    Review of Systems - All other systems were reviewed and are negative except that outlined above.    BP 110/75 (BP Location: Left upper arm, Patient Position: Sitting, Cuff Size: Adult) Comment: vitals after visit | Pulse 52  | Temp 36 C (96.8 F) (Temporal)  | Resp 16  | Wt 61.5 kg (135 lb 9.6 oz) Comment: w/shoes | SpO2 100%  | BMI 23.61 kg/m   Physical Examination: General appearance - alert, well appearing, and in no distress  Mental status - alert, oriented to person, place, and time  Eyes - Right eyelid stye, upper lid pupils equal and reactive, extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - no palpable lymphadenopathy  Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate, regular rhythm, normal S1, S2, no murmurs, rubs, clicks or gallops  Abdomen - soft, nontender, nondistended, no masses or \n organomegaly  Breasts - 2 cm R breast mass 8 cm from the nipple   Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema, no clubbing or cyanosis  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted  Tatoo left upper shoulder and left abdomen     I personally reviewed and interpreted the following results:  Results for orders placed or performed in visit on 03/17/19   Syphilis Screen by RPR with Reflex to Treponemal Antibody   Result Value Ref Range    Syphilis Screen by RPR with Reflex to Treponemal Antibody Nonreactive Nonreactive   HIV Ag/Ab Combo   Result Value Ref Range    HIV Ag/Ab Combo NEG NEG       Assessment and Plan:    Patient Active Problem List   Diagnosis   \u0007 Bipolar 2 disorder (HCC)   \u0007 Insomnia   \u0007 Breast Cancer Right   \u0007 Family history of Alzheimer's disease   IMP  34 year old premenopausal woman with 1 to 2.2 cm grade 1 IDC, strongly ER and PR positive, and mammaprint low risk.  Diagnosis made in March and surgery has been delayed due to COVID19.  Unclear when Dr. ***** and ***** will be able to do surgery at *****.  I have talked with Dr. ***** and as of today, the date is unknown.  She has been offered surgery at *****, but would prefer to stick with this team of surgeons.    We discussed the biology of her cancer, and the prevalence.  We reviewed the imaging and her pathology in detail.  I reviewed the data that supports the use of ovarian suppression in woman with HR+ breast cancer under the age of 35, as the general duration of endocrine therapy which at present would be 5 years.  She was curious about the benefits relatively of shorter duration endocrine therapy , and whether two years would be good enough - and we \n reviewed that the current studies are based on the knowledge that 5 years is not good enough for many cancers like this due to late recurrence risks, so we would not advise shorter durations.    We then talked about different endocrine therapy approaches.  I would be concerned about using tamoxifen alone, both for efficacy which what appears to be at least a 2 cm cancer on exam and for the risks of depression that could be difficult given her underlying bipolar II disorder.    We discussed to toxicities of OS with monthly goserelin, as well as the toxicities of tamoxifen versus an AI.  She would prefer to start with an AI, and I have told her that we will wait two weeks to at least have OS start working.  I also advised her about the menses that often occur two weeks after starting OS    Plan  Start goserelin today (no auth required!)  Start letrozole in about two weeks  Message sent to *****. ***** to discuss surgery timing  Discussed lumpectomy versus mastectomy; she is certain about bilateral mastectomy with reconstruction  Hot soaks to stye  Labs today including estradiol    We talked about treatment of hot flashes and she will ask her psychiatrist about the interaction of effexor and gabapentin and her lamictal and underlying bipolar II disorder.    RTC one month depending on plans for surgery.    Patient ready and able to be educated, verbalizes understanding of information/instructions given.  Treatment options, side effects and nutrition reviewed in detail.  Total face to face time: 75  Total counseling time: 60    *****, ***** ***** am acting as a scribe for services provided by ***** *****. *****, MD on *****/*****/***** *****:***** PM    The above scribed documentation accurately reflects the services I have provided.    ***** *****. *****, MD   *****/*****/***** *****:***** PM          \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment plan. We know this is a lot of information, and we are here to support you.

You have breast cancer in the right breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and spread into nearby tissue. Your cancer is hormone receptor positive (ER+/PR+), meaning it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

Your tumor is about 1–2 cm in size and is considered low risk based on a special test (MammaPrint).

Treatment plan:
Because your cancer depends on hormones, the main treatment is hormone (endocrine) therapy:

Ovarian suppression (medicine like goserelin to stop the ovaries from making estrogen)
Followed by an aromatase inhibitor (medicine like letrozole that lowers estrogen levels)

These treatments help slow or stop cancer growth.

You are also planning to have surgery (bilateral mastectomy with reconstruction).

Next steps:

Start ovarian suppression now
Begin hormone therapy about 2 weeks later
Finalize timing of surgery with your surgical team

Your care team also discussed that hormone therapy is usually taken for about 5 years, as shorter treatment may not be as effective.

We understand this is a stressful time. Please talk with your care team about any concerns, including how treatment may affect your mood and daily life.

You are not alone. We are here to support you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's First Name],

Thank you for coming to see us. Here’s a summary of what we discussed during your visit:

**Diagnosis:**
You have been diagnosed with breast cancer in your right breast. This type of cancer is called invasive ductal carcinoma (IDC). It is a small tumor, about 1 to 2.2 centimeters in size. The cancer is strongly estrogen receptor (ER) and progesterone receptor (PR) positive, which means it grows in response to these hormones. This is important because it helps us choose the best treatment for you.

**Treatment Plan:**
We have decided on a treatment plan that includes surgery and medicine to help control the cancer. 

1. **Surgery:** You will have a bilateral mastectomy, which means both breasts will be removed. This will be followed by reconstruction to rebuild your breasts.
   
2. **Medicine:** You will start taking a medicine called goserelin to stop your ovaries from making hormones. This will begin today. About two weeks later, you will start another medicine called letrozole to further control the cancer.

**Key Test Results:**
- Your recent tests show that the cancer is low risk, which is good news.
- You had a test called Mammaprint, which showed a low risk score.

**Next Steps:**
- We will schedule your surgery as soon as possible. Due to the current situation with COVID-19, there might be some delays.
- You will need to come back in one month for a follow-up appointment. We will check how you are doing and discuss any questions you may have.

**What to Watch For:**
- Keep an eye out for any new symptoms or changes in how you feel.
- If you notice anything unusual, please call us right away.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Your Name]
Medical Communication Specialist
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for an initial consultation regarding your newly diagnosed breast cancer.
**What's new or changed since your last visit?**
You have a type of breast cancer called invasive ductal carcinoma (cancer that started in the milk ducts). It is strongly positive for estrogen receptor (ER) and progesterone receptor (PR), and the HER2 status is uncertain. The cancer is small, measuring about 2 cm, and located 8 cm from the nipple in your right breast. No cancer has spread to other parts of your body.
**What treatment or medication changes were made?**
You will start a medication called goserelin today. In about two weeks, you will begin another medication called letrozole. These medications are used to treat breast cancer. Your psychiatrist will also review the interactions between Effexor, Gabapentin, and Lamictal, which you are taking for your bipolar II disorder.
**What is the plan going forward?**
You are scheduled to have a bilateral mastectomy with reconstruction. This means both breasts will be removed and rebuilt. You will have blood tests today to check your estradiol levels. Your next visit will be in one month, depending on the plans for your surgery.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 15

**Cancer Type:** ER+/PR+/HER2 equivocal metastatic adenocarcinoma, consistent with breast primary

## Original Clinical Note

```
ID: ***** ***** is a 33 y.o. premenopausal patient with a recent diagnosis of a HR+ metastatic breast cancer involving lymph nodes, who presents in consultation to discuss treatment options and to establish care.    HPI: The patient was evaluated by her PCP on 10/30/16 at which time she reported a new right lateral neck lump.  Her exam was notable for a hard, mobile mass (1 cm) on her right lateral neck.  She was recommended to return in 2 weeks if not resolved.  She returned on 11/09/16 with concern that the lymph nodes were increasing in size (0.5 cm mobile mass in right anterior chain; 2 x 1 cm mass and subcentimeter masses of right lateral neck).  She was recommended to get a CBC and return in 2 weeks at which time, if persistent, they would obtain imaging.  She returned on 12/27/16 at which time she reported the LN were smaller and sore with palpation.  Her exam at that time was noted to be supple with a 0.5 cm soft, mobile LN in the right mid-cervical chain.  She was reassured the nodes were shotty and to continue to monitor them.  Finally, she returned on 03/26/17 with concern the nodes were grainy and stable in size, but had started to cause pain.  Her exam at that time demonstrated fullness (1.5 to 2 cm) along the right anterior and superior cervical chain as well as a firm round mass (2 x 2 cm) in the right lateral neck.  The following work-up was then initiated.    03/30/17 Neck US: diffuse bilateral submandibular, anterior, and posterior cervical adenopathy (R submandibular 3.6 x 1.7 cm; R anterior cervical chain 1.0 x 0.8 cm; L anterior cervical 2.7 x 1.0 cm; L posterior lateral neck inferiorly 0.9 x 0.7 cm); thyroid unremarkable    04/03/17 US-guided right supraclavicular LN biopsy: metastatic adenocarcinoma, c/w breast primary (CK7, GATA3, ER - positive; CK20, TTF1, thyroglobulin, napsin, mammaglobin - negative); ER+ (>90%), PR+ (50%), HER2 equivocal (IHC 2; FISH pending)    04/10/17 bilateral diagnostic \n mammogram: extremely dense breast tissue; no features of primary malignancy in either breast; right axillary lymphadenopathy including an enlarged calcified LN  right breast ultrasound: multiple axillary LN with the cortex (0.8 cm); additionally, enlarged right axillary LN with calcifications    04/11/17 PET/CT: multiple FDG-avid right cervical LN (2.2 x 1.8 cm, SUV 6.0) in jugular chain; left Vb LN (0.8 cm, SUV 2.4); scattered subcentimeter thyroid nodules (non-FDG avid); multiple FDG-avid right axillary LN (2.7 x 1.7 cm, SUV 4.5); right supraclavicular LN (0.9 cm, SUV 3.5); heterogeneous breast tissue bilaterally with heterogeneous FDG-avidity, with slightly nodular area of uptake in upper outer right breast    04/12/17 MRI breast: marked background enhancement with the most confluent area of enhancement (3.5 cm) in right upper outer breast, posterior depth (which is slightly more prominent on right than left); superficial and smaller right axillary tail mass (1.5 x 0.5 x 1.0 cm) that may correspond to area of calcifications seen on mammogram; markedly enlarged right axillary LN with cortical thickness (up to 1 cm)    ***** presents to clinic today by herself.  She states she is aware that she now has a cancer diagnosis.  She asks many questions about how long she will live, repeatedly asking what the average prognosis for a patient with metastatic breast cancer.  The patient underwent menarche at 10 years of age.  She took OCP for 4 years.  She is G1P1.  Her first pregnancy was at 31 years of age.          Past Medical History:   Diagnosis Date   \u0007 Back pain    \u0007 Breast cancer    \u0007 Depression    \u0007 Hypertension    \u0007 Obesity    \u0007 Seasonal allergies        History reviewed. No pertinent surgical history.       Family History   Problem Relation Age of Onset   \u0007 Cerebral aneurysm Father    \u0007 Melanoma Neg Hx    \u0007 Basal cell carcinoma Neg Hx    \u0007 Squamous cell carcinoma Neg Hx    \u0007 Skin ca. unk/oth Neg Hx    \u0007 Cancer Neg \n Hx        Social History     Social History   \u0007 Marital status: Single     Spouse name: N/A   \u0007 Number of children: N/A   \u0007 Years of education: N/A     Occupational History   \u0007 ***** *****      Social History Main Topics   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Not on file   \u0007 Alcohol use 0.6 oz/week     1 Standard drinks or equivalent per week      Comment: 1 per month   \u0007 Drug use: No   \u0007 Sexual activity: Not on file     Other Topics Concern   \u0007 Not on file     Social History Narrative    Grew up in ***** *****, *****.  Family is from ***** *****.    She is a single mother; her male infant was born in April 2016.       Outpatient Encounter Prescriptions as of 04/13/2017   Medication Sig Dispense Refill   \u0007 buPROPion (WELLBUTRIN SR) 150 mg 12 hr tablet Take 150 mg by mouth 2 (two) times daily.     \u0007 cetirizine (ZYRTEC) 10 mg tablet Take 1 tablet (10 mg total) by mouth Daily. 30 tablet 6   \u0007 docusate sodium (COLACE) 250 mg capsule Take 1 capsule (250 mg total) by mouth 2 (two) times daily as needed for Constipation. 60 capsule 1   \u0007 hydrOXYzine (ATARAX) 10 mg tablet Take 10 mg by mouth 3 (three) times daily as needed for Itching.     \u0007 ibuprofen (ADVIL,MOTRIN) 600 mg tablet Take 1 tablet (600 mg total) by mouth every 6 (six) hours as needed for Pain. 60 tablet 1   \u0007 iron 18 mg TAB tablet Take by mouth.     \u0007 labetalol (NORMODYNE) 100 mg tablet Take 100 mg by mouth 2 (two) times daily.     \u0007 PNV NO.122/IRON/FOLIC ACID (PRENATAL MULTI ORAL) Take by mouth.     \u0007 triamcinolone (KENALOG) 0.1 % cream Apply topically 2 (two) times daily. Use as instructed     \u0007 UNABLE TO FIND Med Name: Fluoxetine     \u0007 zolpidem (AMBIEN) 5 mg tablet Take 5 mg by mouth nightly as needed for Sleep.       No facility-administered encounter medications on file as of 04/13/2017.        No Known Allergies       Review of Systems:   General - no fatigue, no weight loss   Eyes - no vision changes   HENT - no change in hearing, no \n nasal discharge or sinus tenderness, no difficulty swallowing, no mouth sores, no sore throat   Breast - no breast pain  Respiratory - no cough, no dyspnea, no wheezing   Cardiovascular - no chest pain, no palpitations, no lower extremity edema   GI - no heartburn, no nausea, no emesis, no abdominal pain, no diarrhea, no constipation, no melena, no hematochezia   GU - no suprapubic pain, no dysuria, no vaginal bleeding   Musculoskeletal - no joint aches, no joint stiffness, no muscle pain, no bone pain   Endocrine - no heat/cold intolerance, no hot flashes  Heme/Lymph - swollen lymph nodes; no easy bruising/bleeding, no lymphedema   Neurological - no headaches, no dizziness, no numbness/tingling, no falls   Psychological - no anxiety, no depression   Skin - no rashes/lesions, no diaphoresis       Physical Exam:   ECOG 0   Vital Signs - BP 134/82 | Pulse 69 | Temp 35.7 C (96.2 F) (Oral)  | Resp 18 | Ht 157.5 cm (5' 2.01\") | Wt 80.8 kg (178 lb 3.2 oz) | LMP 09/03/2015 (Exact Date) | SpO2 98% | BMI 32.58 kg/m2   Constitutional - WDWN, NAD  Eyes - sclera anicteric, PERRL, EOMI  HENT - sinuses nontender, nasal mucosa intact, pharynx without erythema, stomatitis, or thrush   Lymph Nodes -   Respiratory - resonant to percussion throughout, CTA bilaterally; no wheezes, rhonchi, or crackles  Cardiovascular - Normal heart rate, normal rhythm, no murmurs, no edema  Breast - no masses bilaterally  GI - Bowel sounds normal, soft; no tenderness, no distention, no HSM  Musculoskeletal - No tenderness over bones or joints.   Neurologic - Alert & oriented x 3, ambulates w/o difficulty, good strength throughout  Psychiatric - Mood stable; no HI, SI, hallucinations, paranoia, or delusions  Skin - warm without rashes or lesions       Studies:  Available labs, pathology, and imaging were reviewed and independently interpreted, as described above in the HPI.    04/03/17 CYTOPATHOLOGY REPORT  ***** WITH ADDENDUM *****    Patient Name: *****, \n *****  Med. Rec.#: *****  DOB: 08/20/1983 (Age: 33)  Sex: Female  Accession #: *****-*****  Visit #: *****  Service Date: 04/03/2017  Received: 04/03/2017  Location: *****  Client: *****   Physician(s): ***** *****. ***** ((*****) *****-*****)    Source of Specimen:  Right supraclavicular lymph node, US guided fine needle aspiration    FINAL CYTOLOGIC DIAGNOSIS:  Right supraclavicular lymph node, US guided fine needle aspiration:   Metastatic adenoccarcinoma, consistent with breast primary; see comment.       COMMENTS:   The aspirates are moderately cellular and demonstrate clusters of  crowded, malignant epithelial cells arranged in cohesive aggregates and  papillary-like groups. The cells contain enlarged and angulated  hyperchromatic nuclei with coarse chromatin, nuclear grooves, and one to  multiple nucleoli. The cytoplasm is dense and eosinophilic, imparting  and apocrine appearnce. A rare intranuclear pseudoinclusion and  scattered mitotic figures are identified.    To further evaluate the neoplastic cells immunohistochemical stains were  performed and evaluated on the cell block and show the following  results:  - CK7: Positive  - CK20: Negative  - Thyroglobulin: Negative  - TTF1: Negative  - Napsin: Negative  - p40: Negative  - p63: Negative  - Mammaglobin: Negative  - GATA3: Positive  - GCDFP15: Negative  - PAX8: Negative  - ER: Positive     The cytopathologic and immunohistochemical findings are most compatible  with metastatic carcinoma of breast primary. Correlation with clinical  and radiologic findings is recommended.    ***** *****, NP, at ***** ***** was contacted with the diagnosis on  04/06/2017. She relayed an understanding of the diagnosis.    Immunohistochemical tests for estrogen and progesterone receptors, HER2  and Ki-67 were performed by manual morphometry on the cell block.    The test for estrogen receptors is positive. There is 3+ nuclear  staining in >90% of tumor cells. Internal positive \n control is present,  and external positive control is appropriate.    The test for progesterone receptors is positive. There is variable  (weak to strong, November 28+) nuclear staining in 50% of tumor cells. Internal  positive control is present, and external positive control is  appropriate.    Result of HER2 test: This carcinoma is equivocal for HER2 oncoprotein  over-expression. The staining intensity of this carcinoma was 2 on a  scale of 0-3.     ***** *****/Pathology Resident   ***** ***** *****/Pathologist  Electronically signed out on *****/*****/***** *****:*****:19    Addendum   Date Ordered:   04/06/2017   Status:  Signed Out    Date Complete:   04/10/2017   By: ***** *****    Date Reported:   04/10/2017     Addendum Diagnosis    - CA125: Weak patchy staining  - CAIX: Negative   - RCC: Negative  - Calcitonin: Negative    The additional immunohistochemical stains support the morphologic  findings. The previously reported diagnosis remains unchanged.     Exam Date: 04/10/2017  Exam(s): Bilateral diagnostic mammography, right breast ultrasound    Clinical History:  33-year-old woman having diagnostic mammography to evaluate for primary breast malignancy. Patient had recent cervical lymph node FNA positive for breast adenocarcinoma. Receptors are pending per APEX.    Comparison:  There are no prior examinations available for comparison.    Mammographic Findings:  The breasts are extremely dense, which lowers the sensitivity of mammography.  There is right axillary lymphadenopathy including an enlarged calcified lymph node.  No mammographic features of primary malignancy in the right or left breast.    Ultrasound Findings:  There are multiple axillary lymph nodes with the cortex measuring up to 8mm. Additionally there is an enlarged right axillary lymph node with calcifications, which corresponds to the right axillary calcified enlarged lymph node on mammogram.    IMPRESSION:  Right breast: \n Known biopsy proven malignancy (BI-RADS 6).  Left breast: Negative (BI-RADS 1).    Recommendations:  Breast MRI to evaluate for underlying breast primary malignancy which is scheduled for 04/12/17. Patient is also scheduled for PET/CT tomorrow to evaluate for additional sites of disease.    Findings and management plan discussed with patient during the exam by Dr. *****.    A letter explaining this interpretation in lay language has been sent to the patient.    Radiologist  ***** *****. *****, M.D., Ph.D.    <This report was electronically signed by ***** *****. *****, M.D., Ph.D. at *****/*****/***** *****:23:13 AM>    Exam Date: 04/12/2017  Exam(s): MR breast bilateral en + un    Clinical History:  33-year-old woman status post palpation-guided FNA of cervical lymph node demonstrating metastatic adenocarcinoma consistent with breast carcinoma, no definite primary breast lesion identified on mammogram and ultrasound, with PET/CT demonstrating possible increased FDG avidity in the outer right breast, here for evaluation for a primary breast lesion.    Comparison:  Prior examinations dated 04/10/2017 (ultrasound) and 04/10/2017 (mammogram) were reviewed for comparison.    MRI Findings:  This bilateral breast MRI exam demonstrates marked background enhancement.  The breasts are composed of heterogeneous fibroglandular tissue and fat.    RIGHT Breast:  Marked background enhancement is seen, with the most confluent area of enhancement occurring in the right upper outer breast, posterior depth (best seen on post-gadolinium axial image #101, sagittal image #363), measuring up to 3.5 cm in AP dimension. This is only slightly more prominent on the right compared with the left side, and appears to correspond with the area noted on recent PET CT.  A more superficial and smaller right axillary tail mass is seen measuring approximately 1.5 x 0.5 x 1.0 cm (best seen on post-gadolinium axial image #104, sagittal image #369), which \n demonstrates suggestion of susceptibility artifact on T1 imaging and may correspond to the area of calcifications seen on mammogram. This appears to enhance with slightly greater intensity than the breast tissue described above.    LEFT Breast:  Marked background enhancement is seen. No abnormal areas of enhancement or other MRI features of malignancy are identified.    OTHER:  Markedly enlarged right axillary lymph nodes are seen with cortical thickness measuring up to 1 cm. Left axillary lymph nodes are unremarkable. Visualized abdomen and remainder of the chest is unremarkable.    IMPRESSION:  Right breast: Suspicious (BI-RADS 4).  Left breast: Negative (BI-RADS 1).    Recommendations:  1. Recommend second look ultrasound to evaluate 2 areas: (1) the area of more confluent enhancement in the upper outer right breast and (2) the right axillary tail mass which may correspond to the calcification-containing lesion on mammogram, with plan for ultrasound-guided core biopsy of these areas if correlates can be found, else recommend MRI-guided biopsy.    2. Abnormally enlarged right axillary lymph nodes are again seen, compatible with known metastatic disease.    
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment options. We know this is a lot to take in, and we are here to support you.

You have metastatic breast cancer, which means the cancer has spread beyond the breast. In your case, it has spread to the lymph nodes (small glands that help fight infection), including nodes in your neck and under your arm.

Your cancer is hormone receptor positive (HR+), meaning it grows with hormones like estrogen. The HER2 result is not clearly positive or negative yet.

Even though the main tumor in the breast is hard to see on some tests, imaging suggests a likely area in the right breast.

What this means:
Metastatic breast cancer is not curable, but it is treatable. The goal of treatment is to control the cancer, help you live longer, and maintain your quality of life.

Treatment plan:
Because your cancer is hormone-sensitive, treatment usually starts with:

Hormone (endocrine) therapy to block estrogen
Other medicines may be added depending on test results (such as HER2 testing)

Next steps:

Complete any remaining testing (such as HER2 results if pending)
Finalize and start your treatment plan

It is normal to have many questions, including about the future. Please talk openly with your care team—they are here to support you and guide you through your care.

You are not alone. We will walk through this together, step by step.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear *****,

Thank you for coming to our clinic. I want to share some important information about your health with you.

**Diagnosis:**
You have been diagnosed with a type of breast cancer called hormone receptor-positive (HR+) metastatic breast cancer. This means the cancer has spread to your lymph nodes. Lymph nodes are small glands in your body that help fight infections.

**Tests and Results:**
We did several tests to understand your condition better. These included:

- **Ultrasound:** This showed enlarged lymph nodes in your neck and underarm.
- **Biopsy:** A small sample from your lymph node was taken. The test showed cancer cells that likely came from the breast.
- **Mammogram and Ultrasound:** These tests looked at your breasts and found no signs of cancer in the breast tissue itself.
- **MRI and PET/CT scans:** These detailed pictures showed areas where the cancer might be growing.

**Treatment Plan:**
To make sure we choose the best treatment for you, we need to do a few more tests. These include:

- **Breast Biopsy:** We will take a small piece of tissue from your breast to check for cancer.
- **Lymph Node Test:** We will also take a small sample from your neck lymph nodes.
- **Genetic Testing:** This helps us understand more about your cancer and choose the best medicines.

**Next Steps:**
Once we have all the test results, we will decide on the best treatment plan for you. This might include medicines that stop the cancer from growing or other treatments.

**What to Watch For:**
Please let us know if you notice any new lumps or changes in your body. Also, tell us if you feel very tired or have any pain.

**Questions:**
It’s important to talk to your doctors if you have any questions or concerns. We are here to help you understand everything about your health.

Take care,
[Your Name]
Medical Communication Specialist
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation regarding your recent diagnosis of metastatic breast cancer and to discuss treatment options.
**What's new or changed since your last visit?**
You have a diagnosis of metastatic breast cancer (Stage IV) that has spread to your cervical lymph nodes and possibly to the right breast. The cancer started in gland cells and is positive for estrogen and progesterone receptors, with uncertain HER2 status.
**What treatment or medication changes were made?**
You will start treatment with antiestrogen therapy if the cancer does not have the protein called HER2. If the cancer does have this protein, you will receive anti-HER2 therapy along with chemotherapy.
**What is the plan going forward?**
You will have a biopsy of the breast mass and a fine needle aspiration of the cervical lymph node to confirm the diagnosis. Genetic testing is also recommended. You will return to the clinic once these tests are completed.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 16

**Cancer Type:** ER+/PR+/HER2- grade 2-3 invasive lobular carcinoma

## Original Clinical Note

```
                            UCSF ***** ***** FAMILY                        COMPREHENSIVE CANCER CENTER                      ***** ***** ***** ***** Care Center                             ***** ***** *****                           Second *****, ***** *****                    ***** *****, ***** *****-*****               Phone: (*****) *****-*****          Fax: (*****) *****-*****    ***** AREA: *****    07/12/2010    *****. ***** (*****-*****) *****, M.D.  ***** *****    ***** *****. *****, M.D.  ***** ***** *****  *****, *****  *****    RE: *****, ***** *****  U#: *****  DATE OF SERVICE: 07/07/10    Dear *****:    I had the pleasure of seeing the above named patient in consultation  regarding treatment of her newly diagnosed breast cancer on July 08,  2011.    As you know, she is a 59-year-old woman who had four benign axillary  lymph nodes resected from her left axilla 10 years ago at ***** *****.  Subsequent to that time, she has had yearly mammograms which were  unremarkable.  In 2007 she had a fine needle aspiration of her right  breast that was read as benign.  In March 2010, an exam was felt to  represent no change in her breast.  In May she had planned on a  screening mammogram; however, two weeks before this mammogram, the  patient felt that her left breast had an area of increased density.  The  mammogram was unremarkable, but an ultrasound performed on June 07  revealed a focal shadowing measuring 2 x 1.5 cm in 10 o'clock that was  suspicious.  A core biopsy on the same day revealed invasive lobular  carcinoma, grade 2-3, measuring at least 1.5 cm.  This lobular cancer  was ER positive with 96% of the cells staining, PR positive with 35% of  cells staining and HER-2/neu negative with 0 cells staining.    An MRI of her bilateral breasts was performed in *****, although the  patient lives in *****; this revealed a large abnormality in her left  breast, over 5.3 cm in AP extent by 1.8 x 3.6 cm.    On \n June 29 the patient saw Dr. ***** ***** at *****.  Dr. *****  palpated a large mass in the left breast and was concerned about the  extent of her disease.  She ordered a PET/CT scan which was performed on  July 06; this revealed hypermetabolic enhancing left breast mass, with  associated left axillary and left subpectoral nodes.  There was no  evidence of distant metastatic disease.  Specifically, the enhancing  lesion in the left breast measured 3.8 x 1.3 cm, but had a low SUV of  1.9; there were subcentimeter left axillary and left subpectoral nodes;  the largest left axillary node measured 0.6 cm and the largest  subpectoral node measured 0.7 cm.  In addition to these findings there  was a 7 mm hypoattenuating nodule on the right lobe of the thyroid gland  that was hypermetabolic with an SUV of 2.4.  Because of this thyroid  nodule, an ultrasound and fine needle aspiration of this lesion was  performed on July 08; this revealed a benign thyroid nodule, colloid  type.    In addition, to the pathology, the core biopsy that was performed in  ***** was reviewed by our pathologist, they agreed with the outside  pathologist that this represented invasive lobular carcinoma, with the  maximum dimension of 1.1 cm; they felt that this was intermediate grade.  There was no lymphovascular invasion.  ER and PR were strongly positive  and HER-2 was negative.    She came to see me to discuss treatment options.    PAST MEDICAL HISTORY:  Her past medical history is remarkable for  hypertension for 30 years, on medications.  She states that her blood  pressure has been well controlled over these last 30 years, but her  hypertension did develop in her late 20s.  She had a hysterectomy and  bilateral salpingo-oophorectomy at age 35 for fibroids with bleeding.  She had a tonsillectomy and she had a right wrist fracture as a child.  Two years ago she slipped and fell and herniated her disk at L3-4; she  was treated with physical \n therapy and ice and has no residual pain at  this time.    GYN HISTORY:  She underwent menarche at 11.  She is G0, P0.  She took  hormone replacement therapy very briefly after menopause and then  stopped.  Her last menstrual period was of course, before her  hysterectomy as noted above.  She took soy and black cohosh for hot  flashes in the past, but these are much improved now.    MEDICATIONS INCLUDE:  Hydrochlorothiazide, potassium and Exforge which  she takes for her blood pressure.    ALLERGIES:  She is allergic to erythromycin with chest pain and sulfa,  with a rash.    SOCIAL HISTORY:  She has been divorced for many years.  She works full  time as a ***** for ***** ***** and ***** ***** in  administration.  She exercises a little bit.  She does not smoke  cigarettes or drink alcohol.    FAMILY HISTORY:  Her mother had Parkinson's disease in her older years  and she does not know anything about her father's history.    REVIEW OF SYSTEMS:  She notes that she has a high tolerance for pain.  She has noted right breast aching that is gradually increasing with a  dull throbbing, this has gotten significantly worse over the last couple  of weeks and has made her very anxious.  She has been using ice packs.  She notes dry eyes.  She has no back pain.  Her energy level is good,  although as noted, she is not very physically active.  She is not  shortness of breath.  She has no cough or abdominal pain and no problems  with her balance.  The rest of her review of systems is negative in  detail.    PHYSICAL EXAMINATION:  GENERAL:  On physical examination, she is a healthy appearing woman in  no acute distress.  VITAL SIGNS:  Her weight is 77 kg, her blood pressure is 123/86, pulse  91, respirations 14 and she is saturating at 99% on room air and is  afebrile.  SKIN:  Her skin reveals no lesions.  HEENT : Her oropharynx is clear.  She is anicteric.  She has no cervical  adenopathy.  She has a 1 cm left \n supraclavicular or anterior cervical  node that feels benign.  LUNGS:  Her lungs are clear to auscultation bilaterally to auscultation  and percussion and she has no bony tenderness.  BREASTS:  Examination of her right breast reveals no masses.  She has a  0.5 cm soft and mobile right axillary lymph node.  Examination of her  left breast reveals a 5 x 5.5 cm mass that extends to underneath the  nipple, but originates at 12 o'clock.  She has a 1 cm palpable left  axillary lymph node.  CARDIAC:  Her cardiac exam is without murmur or gallops.  ABDOMEN:  Her abdomen is soft and nontender, without  hepatosplenomegaly.  EXTREMITIES:  Her extremities reveal no edema.    In summary, ***** ***** is a 59-year-old woman with a new diagnosis of  a clinical stage III lobular carcinoma of the left breast.  On PET/CT  scan there is evidence of local regional node involvement; therefore, we  ordered an ultrasound on the day of her visit, which she accomplished  the following week.  The ultrasound revealed a 1.3 x 0.7 x 1.2 cm left  axillary lymph node which was abnormal in morphology, with a thickened  cortex and a fine needle aspiration was performed on the same day and is  pending.    I had a long discussion with ***** and the friend that accompanied her  to her visit.  We spoke for approximately 60 minutes about her treatment  options.  I had recommended that she consider pursuing neoadjuvant  hormone therapy to try and shrink this tumor before approaching it with  surgery and that this would give us a better idea of the hormone  responsiveness of this very large lobular cancer.  I also discussed with  her the hormone receptors and how they can indicate to us that these  tumors will be very sensitive to hormone therapy.    We talked about the availability of a randomized phase II clinical trial  that randomizes patients to the aromatase inhibitor letrozole, with or  without bevacizumab, the antiangiogenic antibody.  The main \n side effect  of the antibody for Ms. ***** would be hypertension.  We discussed this  at some length and she was extremely hesitant to consider enrolling in a  clinical trial and after receiving additional information and thinking  about it over the weekend, reiterated the fact that although she  appreciated the information that we provided to her, she was not  interested in participating in a clinical trial.  We talked abut  neoadjuvant hormone therapy versus taking an adjuvant approach.  She  clearly has surgically accessible disease if she approaches this with a  mastectomy.  Subsequently, she has responded to us that she would most  like to pursue a mastectomy up front for this cancer and not pursue  neoadjuvant treatment approach.  Therefore, I have referred Ms. *****  back to *****. ***** for consideration for surgery and then I will meet  with her after surgery to discuss adjuvant treatment options.    Thank you very much for allowing me to participate in the care of this  lovely woman.  Please feel free to contact me if you have any questions  regarding my evaluation or recommendations and we will make a final  decision regarding her adjuvant therapy following receipt of her  pathology from her upcoming surgery.    *****,    ***** *****. *****, M.D.  CLINICAL PROFESSOR OF MEDICINE  UCSF COMPREHENSIVE CANCER CENTER  ***** ***** ***** BREAST CARE CENTER    CARBON COPIES:              ***** ***** *****, MD                              ***** ***** *****,***** *****                              ***** ***** *****                                *****-***** *****, MD                              ***** *****          DICTATED BY:                 ***** *****. *****, MD *****                                       ____________________________    ATTENDING PHYSICIAN         ***** *****. *****, MD *****                                   Electronically Signed by                               ***** *****. *****, MD *****/*****/***** *****:***** *****                            \n    ____________________________    D:    *****/*****/***** *****:***** *****  T:    *****/*****/***** *****:***** ***** *****  *****#: *****  \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment options. We know this is a lot to take in, and we are here to support you.

You have breast cancer in the left breast, called invasive lobular carcinoma. This means the cancer started in the breast and has spread into nearby tissue. Your cancer is hormone receptor positive (ER+/PR+), which means it grows with hormones like estrogen. It is HER2 negative, meaning it does not have extra HER2 protein.

Your cancer is Stage III, which means it is larger and has spread to nearby lymph nodes (small glands that help fight infection).

Good news: your scans did not show cancer spread to distant organs.

Treatment options:
Your care team discussed two main approaches:

Surgery first (mastectomy) – to remove the breast tumor
Hormone therapy before surgery (called neoadjuvant therapy) – to shrink the tumor

You have chosen to move forward with surgery first.

After surgery, your care team will review the results and discuss additional treatment (adjuvant therapy). This may include:

Hormone therapy to block estrogen and lower the chance of cancer coming back
Other treatments based on final results

We understand this is a difficult time. Please ask your care team any questions you have. We are here to support you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to see me about your breast cancer. I want to share what we found and what the next steps are.

**Diagnosis and Stage:**
You have a type of breast cancer called invasive lobular carcinoma. It is in the left breast and is at stage III. This means the cancer has spread to nearby lymph nodes.

**Key Test Results:**
- A biopsy showed the cancer is ER positive and PR positive, which means it might respond well to hormone treatments.
- The cancer is HER-2 negative, so we won’t use drugs that target HER-2.
- Tests also showed a large lump in your left breast and some small lumps in the lymph nodes under your arm.

**Treatment Plan:**
We talked about different ways to treat your cancer. You decided you would like to have a mastectomy first. This means removing the entire left breast. After the surgery, we will look at the tissue removed to decide the best next steps.

**Next Steps:**
- You will have a mastectomy soon. Dr. [Doctor’s Name] will perform the surgery.
- After the surgery, we will meet again to talk about further treatment options based on the results from the surgery.

**What to Watch For:**
Please watch for any changes in your breast or any new symptoms. If you notice anything unusual, please call us right away.

**Questions:**
If you have any questions or need more information, please talk to your care team. They are here to help you understand everything and support you through this journey.

Warm regards,

[Doctor’s Name], M.D.
UCSF Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consult regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You were diagnosed with a type of cancer called invasive lobular carcinoma (cancer that started in the milk-producing glands). The cancer is ER positive with 96% of cells staining, PR positive with 35% of cells staining, and HER-2/neu negative with 0 cells staining. Imaging tests showed a large abnormality in your left breast and some enlarged lymph nodes. A fine needle aspiration of a lymph node was done and is pending.
**What treatment or medication changes were made?**
No changes were made to your current medications.
**What is the plan going forward?**
You are scheduled to have a mastectomy (surgery to remove the breast). After your surgery, you will have a CT scan and an ultrasound. You will also have a follow-up visit after surgery to discuss further treatment options.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 17

**Cancer Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma

## Original Clinical Note

```
HPI:  ***** ***** ***** is a 58 y.o. female here in consultation to discuss treatment options for her newly diagnosed breast cancer whose history is as follows:          Patient Active Problem List    Diagnosis Date Noted   \u0007 Breast cancer, left breast 06/24/2017     Stage IIb (T2N1M0) Gr 2 IDC (ER pos/PR pos/Her-2-neu neg)  March 2017 patient found a left breast mass measuring 2cm after she felt a pinch in her left breast  Saw primary care doctor  04/10/17 BL Dx mammogram: L irregular mass 20 mm with spiculated margins at 12 o'clock  04/10/17 US L 12 o'clock 2 cm from the nipple with solid mass 17 mm   04/21/17: US Guided Needle Core biopsy Gr 2 IDC ER+ >95% PR+ 25%,HER-2/neg by IHC 1+, Ki-67 >30%  FISH not amplified  04/30/17 Left lumpectomy and SLNBx by Dr. ***** ***** *****: gr 2 IDC, 3.6cm, + LVI, margins positive, 2/2 lymph nodes involved (largest 1.8cm with extracapsular extension).   Postoperative MRI 05/06/17 enhancement surrounding the surgical site suspicious for residual disease and a 4mm focus of enhancement in the L breast at 6 o'clock.   MRI guided biopsy 05/20/17 of this additional area revealed proliferative fibrocystic change without atypia.   05/19/17 BRCA test negative by report (Ambry - not sure whether panel or only BRCA)  06/04/17 Surgical re-excision left breast, lumpectomy re-excision:1.)Residual IDC spans 2cm  2.)Invasive carcinoma at deep margin in one small focus  Comment:The description of tumor 'spanning 2cm' in this specimen is probably naturally going to be interpreted as a much larger lesional  volume than the aggregate of the small residual tumor deposits actually warrant.The largest of these foci is only 6mm x 4mm in cross section.  A much smaller residual deposit (2 x 1mm) is at the deep resection margin (Block *****).       Patient reports mild left breast pain status post lumpectomy re-excision on 06/04/2107. It is intermittent and described as \"pinching\" sensation which is \n tolerable. ***** notes that she has very supportive family/friends that live nearby.   She is very upset about her diagnosis and need for treatment.      She is going to ***** in February for her niece's *****.          Past Medical History:   Diagnosis Date   \u0007 Breast cancer, left breast March 2017           Past Surgical History:   Procedure Laterality Date   \u0007 BIOPSY / EXCISION / DISSECTION AXILLARY NODE Left    \u0007 MASTECTOMY PARTIAL / LUMPECTOMY W/ AXILLARY LYMPHADENECTOMY Left 04/30/2017     Family History   Problem Relation Age of Onset   \u0007 Breast cancer Paternal Grandmother 35   \u0007 Breast cancer Paternal Aunt 70   \u0007 Breast cancer Cousin 37   \u0007 Breast cancer Cousin 42     the two daughters of the aunt with cancer      Breast cancer Paternal Grandmother 35   \u0007 Breast cancer Paternal Aunt   diagnosed in her late 70s, mother of the 2 sisters below   \u0007 Breast cancer Other   Paternal Cousin x 2, they are sisters, diagnosed at 37 and 42yo     G2 P2  Age at first pregnancy: 22  Age at menarche: 14  Age at menopause: 42  Previous hormone replacement therapy: NO         Allergies   Allergen Reactions   \u0007 Penicillins        Social History         Social History Narrative    Divorced, lives alone    G2P2, 36 and 25 year old sons    Works as a ***** ***** in the ***** *****    LMP age 42 at the time of her divorce    No HRT         Current Outpatient Prescriptions:   \u0007  cholecalciferol, vitamin D3, (VITAMIN D3) 2,000 unit CAP capsule, Take by mouth., Disp: , Rfl:     Review of Systems - All other systems were reviewed and are negative except that outlined above.    BP 132/85 | Pulse 69 | Temp 36.5 C (97.7 F) (Oral)  | Resp 16 | Ht 160 cm (5' 3\") Comment: June 2017 @ ***** | Wt 57.9 kg (127 lb 9.6 oz) | SpO2 98% | BMI 22.6 kg/m2  Physical Examination: General appearance - alert, well appearing, and in no distress  Mental status - alert, oriented to person, place, and time  Eyes - pupils equal and reactive, \n extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - no palpable lymphadenopathy   Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry,  Heart - normal rate, regular rhythm, normal S1, S2, no murmurs, rubs, clicks or gallops  Abdomen - soft, nontender, nondistended, no masses or organomegaly  Breasts - Left breast with periareolar scar and distortion  Right breast with no masses  Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema, no clubbing or cyanosis  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted    I personally reviewed and interpreted the following results:  No results found for this or any previous visit.     05/20/2017 MRI BREAST BX VACCUUM ASSIST LT  FINDINGS: Benefits, risks, and alternatives of the procedure were discussed with the patient and informed consent was obtained. A time out was held. The patient's left breast was prepped in the usual sterile fashion. Local anesthesia was achieved with 1% lidocaine, along with 1% lidocaine with diluted epinephrine Gadolinium based contrast (5 cc of Gadavist) was administered intravenously without difficulty and the finding in question at the posterior left 6:00 position was visualized. Using MRI guidance and grid compression, a 9 gauge vacuum assisted core biopsy device was used to obtain tissue samples. A titanium marker was placed after the biopsy. The patient tolerated procedure well with no immediate Complications. Pathology showed proliferative fibrocystic changes without atypia.   IMPRESSIONS:  MRI-guided core biopsy of a 4 mm focus of \n contrastenhancement in the posterior, lower (6:00) region of the left breast.Pathology showed fibrocystic changes.This is concordant with the imaging findings.Recommendation would be for breast MRI in 6 months as per the usual imaging surveillance protocol for benign concordant diagnosis.  05/09/2017 Left Breast Ultrasound Limited  LEFT ULTRASOUND FINDINGS: A targeted ultrasound was performed of the left breast at 6 o'clock located 3 centimeters from the nipple. There is no sonographic correlate to the MRI finding in the left lower central breast  IMPRESSION  There is no sonographic finding to the MRI finding of a 4 mm focus of enhancement in the left lower central breast. Therefore, recommend MRI-guided biopsy of the focus of enhancement in the left lower central breast, as per MRI report. The results of this examination were reviewed directly with the patient by a member of the Breast Health Center, and a written summary report was provided to her. BI-RADS Category 1 : Negative    05/06/2017 MRI BREAST BILATERAL WWO CONTRAST  FINDINGS: The breasts demonstrate heterogeneous fibroglandular tissue. There is minimal background enhancement that is symmetric.  Right breast: There are no abnormal areas of contrast enhancement.  Left breast: Postoperative changes in the left breast with a seroma in the left breast at 12:00 extending from the nipple areolar complex to the pectoralis muscle. The seroma measures 8.6 x 4.6 cm (AP, transverse). The craniocaudal extent of the anterior portion of the seroma measures 5.1 cm in the posterior portion measures 2.9 cm. The posterior portion of the seroma measures 3.4 x 1.1 x 2.9 cm (AP, transverse, CC). There is peripheral enhancement surrounding the seroma. There is irregular peripheral enhancement along the anterolateral superior margin of the anterior portion of the seroma (postcontrast axial image 54 and 49 and sagittal image 32) and along the posterolateral aspect of the \n anterior portion of the seroma (postcontrast axial image 42 and sagittal 39). The nodular peripheral enhancement extends to the nipple areolar complex. There is also nodular peripheral enhancement along the medial margin of the anterior portion of the seroma(postcontrast axial image 45 and sagittal image 53). The peripheral enhancement demonstrates type I kinetics. No fat plane between the peripheral enhancement of the seroma and the left pectoralis muscle. 4 x 2 x 4 mm focus of enhancement in the posterior left breast at 6:00 (postcontrast axial image 72 and sagittal image 44). Kinetics curve assessment demonstrates type I enhancement There is diffuse skin thickening, likely related to postoperative changes.  Other: Postoperative seroma in the left axilla from recent sentinel lymph node dissection measuring 6.6 x 4.0 x at least 5.5 cm (AP, transverse, CC). The full craniocaudad extent of this seroma is not imaged. There is thin peripheral enhancement surrounding the seroma, which is due to postoperative changes. There is no axillary or internal mammary lymphadenopathy.  IMPRESSION:  Right breast: BI-RADS 1: Negative. Recommend annual mammogramthe right breast.  Left breast: BI-RADS 4C: High suspicion for malignancy. Irregular peripheral enhancement along the anterior, anterolateral, medial, and posterolateral aspect of the anterior portion of the seroma in the left breast at 12:00 is suspicious for residual disease. Differential diagnosis includes postoperative inflammation given recent lumpectomy. Recommend surgical consultation. 4 mm focus of enhancement in the left breast at 6:00 is mildly suspicious (4A). Recommend targeted ultrasound. If there is no sonographic correlate, recommend MRI guided biopsy.  04/30/2017 NM Sentinel node imaging of the left breast  FINDINGS:  Expected intense activity is noted at the injection site at the  left breast. 3 foci of tracer uptake in the left axilla,  compatible with \n sentinel lymph nodes.    04/21/2017 Left Breast US Guided Biopsy, Left Post-Biopsy Mammogram  Pathology Results: Infiltrating Ductal Carcinoma (see final pathology report). This was concordant with the imaging impression. Recommend surgical excision. Results, recommendations and plans for management were discussed with the patient by the Breast Health Center clinical nurse specialist. Patient to follow up with Dr ***** *****.    04/10/2017 Bilateral Digital Diagnostic Mammogram w/CAD, Left Breast Ultrasound Limited   BILATERAL MAMMOGRAM FINDINGS  The breasts are heterogeneously dense, which may obscure small masses. There is an irregular mass measuring 20 millimeters with spiculated margin in the left breast at 12 o'clock.This is best seen on the tomosynthesis views. The right breast appears unremarkable.  LEFT ULTRASOUND FINDINGS  A targeted ultrasound was performed of the left breast at 12 o'clock  located 2 centimeters from the nipple. Ultrasound is suggestive of a solid mass measuring 17 millimeters in the left breast at 12 o'clock located 2 centimeters from the nipple.This corresponds to the palpable finding of clinical concern.  IMPRESSION  Solid mass in the left breast is highly suggestive of malignancy.  Ultrasound-guided core biopsy is recommended. The referring physician was notified of these results by fax. The left axilla will be evaluated with ultrasound when the patient returns for biopsy. If there adenopathy, US guided core biopsy will be performed.    Assessment and Plan:        Patient Active Problem List   Diagnosis   \u0007 Breast cancer, left breast   58 year old postmenopausal woman with a T2N1 grade 2 IDC with a reported Ki67 of 30% on her core biopsy.  HER2 was retested on the surgical sample and showed no gene amplification by FISH.  Long discussion of the pathology and the goal of therapy.  She did not understand the relationship between the cancer she currently has, her genetic testing \n and the risk of distant recurrence. We discussed this at length.  She does not know if she has had expanded panel testing or BRCA testing only.    We also discussed her pathology report which shows a possible small area of positivity at the deep margin on the re-resection sample, as well as additional multifocal disease.  I believe that the fascia was removed, so this will be managed by radiation therapy.    Given the extent of her cancer, I would recommend adjuvant chemotherapy.  We will need to review her pathology and also discuss whether any additional node sampling is required or whether she can proceed to radiation without further surgery.  LIkely give the fact that she needs radiation already, the best approach will be radiation without further node sampling.    We discussed types of chemotherapy, her options include AC/weekly T or TC x 4 chemotherapy. Given recent data suggesting similar outcome in hormone receptor positive disease with these two regimens, and less toxicity with TC as well better hair preservation with scalp cooling - which is very important to her in particular.  I think it is reasonable to go with TC x 6 based on these recent studies.    We discussed scalp cooling and the risk of hair loss, which she also discussed with our scalp cooling coordinator.  She understands that this is not a guarantee for success.    She is most concerned about having a port, but understands that getting chemotherapy in her right arm with poor baseline access is not feasible.  We reviewed the port placement and what it will look like.    She is unhappy with her cosmesis in the left breast.  I explained that we have to wait for the area to be fully healed and that she could see a plastic surgeon before radiation to consider whether additional surgery would be useful.    Plan  Submit authorization for TC x 6 with ONPRO support  Referral for chemotherapy teaching, port placement  Staging with \n PET/CT  Baseline echo although not receiving AC as planned so far  Start 07/09/17 with dignicap.    Path review (review Ki67)    Discuss cosmetics when the left breast is better healed.    Patient ready and able to be educated, verbalizes understanding of information/instructions given.  Treatment options, side effects and nutrition reviewed in detail.  Total face to face time: 70  Total counseling time: 55      \n
```

---

## Letter A

Dear Patient,
We are writing to explain your diagnosis and your treatment plan. We know this is a lot of information, and we are here to support you.
You have breast cancer in the left breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and spread into nearby tissue. Your cancer is Stage IIb, which means the tumor is larger and has spread to nearby lymph nodes (small glands that help fight infection). 
Your cancer is hormone receptor positive (ER+/PR+), meaning it grows with hormones like estrogen, and HER2 negative, meaning it does not have extra HER2 protein. 
You have already had surgery (lumpectomy and re-excision). The tumor was about 3.6 cm, and 2 lymph nodes had cancer. Some cancer cells were still close to the edge (margin), so further treatment is needed. 
Treatment plan (after surgery):
Your care team recommends:


Chemotherapy (medicine to kill cancer cells)


Radiation therapy (high-energy rays to treat the breast area)


Hormone therapy (medicine to block estrogen and reduce the risk of cancer coming back) 


Chemotherapy options were discussed, and a plan using TC (docetaxel and cyclophosphamide) is being considered. 
Next steps:


Schedule chemotherapy teaching and start treatment


Place a port (a small device under the skin to give medicine)


Complete staging scans and heart testing before treatment 


We understand this is a difficult time, and it is normal to feel upset. You have support around you, and your care team is here to help.
Please ask any questions you have. We are here for you every step of the way.
Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear *****,

I hope this letter finds you well. I want to share some important information about your health in a way that is easy to understand.

**Diagnosis:**
You have been diagnosed with breast cancer in your left breast. The doctors call this "Stage IIb." This means the cancer is growing but hasn't spread to other parts of your body yet.

**Treatment Plan:**
We will use a combination of treatments to fight the cancer. First, you will have chemotherapy. Chemotherapy uses special medicines to kill cancer cells. We will use a treatment called "TC x 6," which means you will get six rounds of chemotherapy. 

To help keep your hair during treatment, we will use a special cooling cap called "Dignicap." This cap helps protect your hair from the effects of chemotherapy.

We will also use a small device called a "port" to give you the medicine. This makes it easier to give you the treatment without needing to find a vein each time.

After chemotherapy, you will have radiation therapy. Radiation uses high-energy beams to kill any remaining cancer cells.

**Key Test Results:**
Tests show that the cancer is sensitive to hormones. This means certain treatments can work well against it. The tests also show that the cancer is not aggressive and doesn’t spread easily.

**Next Steps:**
Starting July 9, 2017, you will begin chemotherapy using the Dignicap. Before starting, we will teach you more about the treatment and how to use the port.

Once the chemotherapy is done, you will start radiation therapy. We will also check your heart to make sure it stays healthy during treatment.

**What to Watch For:**
Please watch for any new lumps or changes in your breast. Also, let us know if you feel very tired or have any other unusual symptoms.

**Support:**
You mentioned that you have a strong support system with family and friends nearby. This is very important during your treatment journey.

If you have any questions or concerns, please talk to your care team. They are here to help you every step of the way.

Take care and stay strong.

Best regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new consult regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You have a stage IIb (T2N1M0) grade 2 invasive ductal carcinoma (IDC) in your left breast. The cancer is estrogen receptor positive (ER+) and progesterone receptor positive (PR+), but does not have a protein called HER2. The tumor measures 3.6 cm and has spread to two lymph nodes. Postoperative MRI shows some areas that might have residual cancer.
**What treatment or medication changes were made?**
You will start a new treatment plan with TC x 6, which involves chemotherapy. You will also use ONPRO support and dignicap to help manage side effects.
**What is the plan going forward?**
You will have radiation therapy to manage the extent of your cancer and any remaining areas of concern. A port will be placed to make it easier to receive chemotherapy. You will have a staging PET/CT scan and a baseline echo before starting chemotherapy. You will also undergo genetic testing for BRCA.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 18

**Cancer Type:** ER+/PR-/HER2- grade 2 invasive ductal carcinoma

## Original Clinical Note

```
HPI:  ***** ***** is a 32 y.o. female here for a discussion of treatment options for recently diagnosed left breast cancer. Her history is as follows:    ***** underwent genetic testing in early 2018 due to family history of breast cancer diagnosed around age 30. The Myriad myRisk panel identified a mutation in the ATM gene (c1109dup (p.Tyr370*) heterozygous). In December 2017, she palpated a left breast lump and presented to PCP. On 01/10/18, a left breast US at ***** ***** ***** demonstrated an irregularly marginated mass with internal vascularity at 5:00, 4-5cm from the nipple. On 01/12/18 left diagnostic mammogram at ***** ***** ***** found an irregular 15 mm mass at 5 o'clock, 5cm from the nipple.    On  01/24/18 an US-guided core biopsy of the left breast mass at ***** ***** ***** revealed a grade 2 invasive ductal carcinoma measuring 8mm. The cancer is ER positive (>95%), PR positive (<5%), HER2 negative (by FISH), Ki67 44%.    On 02/06/18, an US of left axilla noted multiple lymph nodes, largest measuring 1.2 x 1.5 x 0.5cm and smallest 0.5 x 0.5 x 0.4cm. The small node was echogenic throughout. The two slightly larger nodes showed normal hilar echogenicity with a hypoechoic rim. No architectural distortion was noted. On 02/14/18, an FNA of a left axillary lymph node was attempted in ***** and the path was benign. The report described a few benign mesenchymal cells admixed with lymphocytes and peripheral blood cell elements, and no malignant cells identified.    On 02/17/18, a diagnostic mammogram of right breast showed no evidence of malignancy. On 02/22/18, the patient came to ***** for a second opinion with Dr. *****.     She underwent bilateral breast MRI on 03/16/18. In the right breast, there is a 9 mm oval mass with smooth margins and dark internal septation in the upper outer right breast. Findings are classic for fibroadenoma. There was no abnormal areas of enhancement or \n other MRI features of malignancy  on the right. In the LEFT breast, there is a 2.5 cm x 2.2 cm  x 2.3 cm oval mass with irregular margins and heterogeneous enhancement in the lower outer left breast. There is also at least two abnormal lymph nodes in the left axilla level I and I/II with cortical thickening, measuring up to 6 mm.     She was seen by Radiation Oncology at ***** yesterday who were not enthusiastic about offering this patient radiation based on her ATM mutation, and therefore recommended she proceed with mastectomy. She was seen in ***** today and Dr. ***** has recommended a repeat FNA of the left axilla.     Subsequent to the visit, the Mammaprint from her prior core biopsy came back as High Risk (-0.622)    History reviewed. No pertinent past medical history.     Past Surgical History:   Procedure Laterality Date   \u0007 Colposcopy         Allergies/Contraindications   Allergen Reactions   \u0007 Nut (All) Anaphylaxis     Almonds    \u0007 Corn    \u0007 Soy    \u0007 Sulfa (Sulfonamide Antibiotics)          Current Outpatient Prescriptions:   \u0007  ALPRAZolam (XANAX) 0.25 mg tablet, , Disp: , Rfl:   \u0007  ibuprofen (ADVIL) 200 mg tablet, Take 200 mg by mouth every 6 (six) hours., Disp: , Rfl:   \u0007  trazodone HCl (TRAZODONE ORAL), Take 0.5 tablets by mouth., Disp: , Rfl:   No current facility-administered medications for this visit.     Gynecologic history: Menarche at around February 16. Patient is G2P2. She has used Depo-Provera and oral contraceptives. Premenopausal    Social History     Social History Narrative    Manager of jewelry store in *****       History   Smoking Status   \u0007 Former Smoker   \u0007 Packs/day: 0.20   \u0007 Years: 10.00   \u0007 Start date: 03/07/2002   \u0007 Quit date: 08/05/2012   Smokeless Tobacco   \u0007 Never Used       History   Alcohol Use No     Comment: Previously drank 4 a week       Family History   Problem Relation Name Age of Onset   \u0007 Breast cancer Mother     \u0007 Breast cancer Other     \u0007 Breast cancer Other           Review \n of Systems  Answers for HPI/ROS submitted by the patient on 03/16/2018   Nervousness or Anxiety: Yes  Sleep disturbance: Yes  Patient denies sore throat, nasal congestion, vision changes, cough, chest pain, dyspnea, nausea, vomiting, abdominal pain, headaches, fevers, and change in bladder or bowel function. Remaining systems reviewed and are negative except that outlined above    Physical Examination: General appearance - alert, well appearing, and in no distress  BP 128/72  | Pulse 92  | Temp 36.6 C (97.9 F) (Oral)  | Resp 18  | Ht 157.5 cm (5' 2.01\") Comment: March 2018 @ ***** | Wt 57.9 kg (127 lb 11.2 oz)  | SpO2 100%  | BMI 23.35 kg/m     Mental status - normal mood, behavior, speech, dress, motor activity, and thought processes  Eyes - pupils equal and reactive, extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - no palpable lymphadenopathy  Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate and regular rhythm, no murmurs noted  Abdomen - soft, nontender, nondistended, no masses or organomegaly  Breasts - Deferred today by me since she has had multiple exams today already  Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - cranial nerves II through XII intact, motor and sensory grossly normal bilaterally  Musculoskeletal - no joint tenderness, deformity or swelling, no muscular tenderness noted  Extremities - no pedal edema noted  Skin - No rashes      Imaging results reviewed and discussed with patient:    Exam Date: 03/16/2018    Exam(s): MR breast bilateral en + un    Clinical History:  32-year-old woman history of ATM mutation and left IDC presenint for evaluation of extent of disease.    Comparison:  Prior examination dated 01/12/2018 was reviewed for comparison.    MRI Technique:  Bilateral 1.5T breast MRI \n without and with contrast was performed with a dedicated breast coil using T1 and T2-weighted sequences including:  axial T2-weighted with fat saturation, high resolution T1 fat-suppressed 3D gradient echo pre-, and dynamic post-contrast sequences. Gadolinium based contrast was injected intravenously at a dose of 0.1 millimole per kilogram body weight without adverse reaction.    The study was processed with a Hologic Aegis Sentinelle system to optimize interpretation by generating multiplanar reconstructions (MPRs) and maximum intensity projections (MIPs) in 3 dimensions using raw and generated subtracted images from the dynamic contrast enhanced data set.    MRI Findings:  This bilateral breast MRI exam demonstrates mild background enhancement.  The breasts are composed of heterogeneous fibroglandular tissue and fat.    RIGHT Breast:  There is a 9 mm oval mass with smooth margins and dark internal septation in the upper outer right breast. Signal intensity/time curve demonstrates slow initial rise and persistent enhancement kinetics.All findings classic for fibroadenoma.  No abnormal areas of enhancement or other MRI features of malignancy are identified. Findings are best seen on post-contrast axial image #66 and sagittal image #603.    LEFT Breast:  There is a 2.5 cm (AP) x 2.2 cm (TV) x 2.3 cm (CC) oval mass with irregular margins and heterogeneous enhancement in the lower outer left breast. Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics. Findings are best seen on post-contrast axial image #87 and sagittal image #118.    Other: At least two abnormal lymph nodes in the left axilla level I and I/II with cortical thickening, measuring up to 6 mm. Findings are best seen on post-contrast axial images #24 and 55.    IMPRESSION:   Left breast: Known biopsy proven malignancy (BI-RADS 6).  Right breast: Benign (BI-RADS 2).    Recommendations:  1. Recommend continued \n surgical/oncological management for knwon left breast malignancy.    Pathology results reviewed and discussed with the patient:    Assessment and Plan:  ***** ***** is a 32 y.o. female ATM mutation carrier with a clinical T2NX ER+/PR-/HER2 negative left breast. Subsequent to our clinic visit, the patient underwent a second FNA of the left axillary LNs and it again showed no cancer. Because of her ATM mutation, radiation oncology has recommended that she proceed with mastectomy rather than breast conserving surgery.     Despite the fact that the patient's tumor is Mammaprint High Risk and I will recommend chemotherapy, I explained to the patient that if she is truly node negative at the time of surgery, we will consider Taxotere and Cytoxan, whereas if she is node positive, she should get AC-Taxol. Therefore in order to make the most appropriate recommendation for chemotherapy, I favor her going to surgery first.    I explained my rationale for this recommendation in detail with the patient and her family and they voiced understanding. I began to explain the differences in the schedule and toxicities for various chemotherapies but will defer and comprehensive conversation until after surgery.    Because her breast cancer is estrogen receptor positive, she understands that she will receive at least five years of adjuvant hormonal therapy.   I briefly introduced the updated data from the SOFT and TEXT trials published in the New England Journal of Medicine in September 2017. Previously published data from this trial demonstrated that for patients who were premenopausal at diagnosis and recovered ovarian function after completion of chemotherapy, the combination of ovarian suppression and an aromatase inhibitor was superior to tamoxifen in terms of preventing distant recurrence. The more recently published data from the SOFT trial shows the 8-year disease-free survival rate was 78.9% with tamoxifen alone, 83.2% with \n tamoxifen plus ovarian suppression, and 85.9% with exemestane plus ovarian suppression (P=0.009 for tamoxifen alone vs. tamoxifen plus ovarian suppression). Among the women with HER2 negative cancers who received chemotherapy, the 8-year rate of distant recurrence with exemestane plus ovarian suppression was lower than the rate with tamoxifen plus ovarian suppression (by 7.0 percentage points in ***** and by 5.0 percentage points in TEXT).     W will have a more thorough discussion regarding choice of hormonal therapy after surgery as well.    Patient ready and able to be educated, verbalizes understanding of information/instructions given.  Treatment options, side effects, clinical trials, and follow up plan were reviewed  in detail.  Total face to face time: 60 minutes  Total counseling time: 45 minutes          \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment plan. We know this is a lot of information, and we are here to support you.

You have breast cancer in the left breast, called invasive ductal carcinoma. This means the cancer started in the milk ducts and spread into nearby tissue. Your cancer is hormone receptor positive (ER+), which means it grows with estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

Your tumor is about 2–2.5 cm in size. Tests show possible lymph node involvement, but biopsies of the lymph nodes have not found cancer so far.

You also have a change in a gene called ATM, which can affect treatment choices. Because of this, your doctors recommend mastectomy (removal of the breast) instead of breast-conserving surgery with radiation.

Next steps:

Proceed with surgery (mastectomy)
After surgery, your care team will review results to guide further treatment

Treatment plan after surgery may include:

Chemotherapy (medicine to kill cancer cells), especially because your cancer is considered higher risk
Hormone therapy (medicine to block estrogen), usually for at least 5 years

The exact chemotherapy plan will depend on whether cancer is found in the lymph nodes at surgery.

We understand this is stressful, and it is normal to feel anxious. Please share your concerns with your care team. We are here to support you and help guide you through each step.

You are not alone in this.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient's First Name],

I hope this letter finds you well. I wanted to share some important information about your recent diagnosis and treatment plan in a way that is easy to understand.

**Diagnosis and Stage:**
You have been diagnosed with left breast cancer. The cancer is at an early stage, which means it hasn't spread far yet. It is called "stage T2N0" because it is a bit bigger than 2 centimeters but has not spread to the lymph nodes.

**Treatment Plan:**
Based on your situation, the doctors recommend a mastectomy. This means removing the entire left breast to get rid of the cancer. After the surgery, you will likely need chemotherapy to kill any remaining cancer cells. Since your cancer is sensitive to estrogen, you will also need to take medicine to block estrogen for at least five years to help prevent the cancer from coming back.

**Key Test Results:**
- **Biopsy:** The cancer is ER positive, which means it uses estrogen to grow. It is PR negative and HER2 negative, which means it doesn’t use progesterone or another protein to grow.
- **MRI:** The MRI showed a mass in your left breast and some enlarged lymph nodes, but no cancer was found in the lymph nodes.
- **Genetic Testing:** You have a mutation in the ATM gene, which makes radiation less effective for you.

**Next Steps:**
- **Surgery:** You will have a mastectomy to remove the cancer.
- **Chemotherapy:** After surgery, you will start chemotherapy to kill any remaining cancer cells.
- **Hormonal Therapy:** You will take medicine to block estrogen for at least five years.

**What to Watch For:**
After surgery, watch for signs of infection like redness, swelling, or fever. Also, let us know if you have any unusual pain or discomfort.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a second opinion regarding your newly diagnosed left breast cancer.
**What's new or changed since your last visit?**
You have been diagnosed with a type of breast cancer called invasive ductal carcinoma, which is ER positive, PR negative, and does not have a protein called HER2. The cancer measures 8mm and is considered high risk based on a test called Mammaprint. Imaging shows a 2.5 cm mass in the left breast and some enlarged lymph nodes in the armpit area. No cancer was found in the right breast.
**What treatment or medication changes were made?**
No specific medications were changed during this visit.
**What is the plan going forward?**
You will proceed with surgery to remove the breast (mastectomy). After surgery, you will start chemotherapy based on whether the cancer has spread to the lymph nodes. If the cancer has not spread to the lymph nodes, you will receive Taxotere and Cytoxan. If it has spread, you will receive a chemotherapy regimenaxol. You will also receive at least five years of adjuvant hormonal therapy because your cancer is ER positive. The specific type of hormonal therapy will be discussed after surgery.
You are referred to radiation oncology for a consultation.
A more thorough discussion regarding the choice of hormonal therapy will occur after surgery.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 19

**Cancer Type:** ER+/PR-/HER2- grade 2 and grade 3 invasive ductal carcinoma (IDC) with DCIS component

## Original Clinical Note

```
SUBJECTIVE    ***** ***** is a 49 y.o. female with ER+/PR-/HER2- left breast cancer, mammaprint low risk IDC of the left breast s/p NAHT, s/p bilateral mastectomies, s/p adjuvant tamoxifen *****/*****-*****/*****. She was switched to OS/AI in November 2018. Switched from letrozole to exemestane December 2018 due to anxiety and emotional lability.      Interval History  Attempted letrozole for August 12.5 months but switched to exemestane after experiencing anxiety and mood swings. Tolerating exemestane better.   +minimal joint aches  +mild hot flashes and insomnia  +bone pain  +vaginal dryness    Recently graduated from master's program so has more time now to exercise.    No new or concerning bone pain, sob, cough, abd pain, n/v, headache, visual changes    ROS  All systems reviewed and negative except for as outlined above in the HPI.        Past Medical History:   Diagnosis Date   \u0007 Breast cancer (HCC)    \u0007 Endometriosis    \u0007 Malignant neoplasm of left female breast (HCC) 01/10/2017   \u0007 PONV (postoperative nausea and vomiting)        Patient Active Problem List    Diagnosis Date Noted   \u0007 Malignant neoplasm of female breast (HCC) 11/12/2017     Added automatically from request for surgery *****     \u0007 Status post bilateral mastectomy 07/18/2017   \u0007 S/P mastectomy, bilateral 05/27/2017     Added automatically from request for surgery *****     \u0007 Postoperative visit 05/27/2017     Added automatically from request for surgery *****     \u0007 Wound disruption, post-op, skin, sequela 05/27/2017     Added automatically from request for surgery *****     \u0007 Breast cancer, left (HCC) 04/09/2017   \u0007 PONV (postoperative nausea and vomiting) 03/27/2017   \u0007 Malignant neoplasm of left female breast (HCC) 01/10/2017     Clinical stage 2-3 HR+ HER2 negative IDC on NAHT     -12/05/16: Screening mammo - Extremely dense. LEFT 3:30, 5 cm FN: 1.8cm irregular spiculated mass with overlying skin retraction.  LEFT 3:30, 2.5 cm FN: 1.3cm irregular spiculated mass. LEFT \n inner retroareolar: Small benign appearing oval circumscribed masses measuring up to 1.3 cm. RIGHT inner retroareolar: Multiple oval circumscribed masses measuring up to 3cm.  -12/05/16: Bilateral US (converted from screening to diagnostic):  LEFT 3:30, 5cm FN: 1.8 x 1.1 x 0.9cm irregular spiculated hypoechoic mass, highly suspicious. LEFT 3:30, 2.5cm FN: 0.8 x 0.9 x 1.3cm irregular spiculated mass, highly suspicious. Few small cysts. No left axillary adenopathy. RIGHT 11:00, 4cm FN: 0.9 x 0.6 x 0.5cm oval mass with microlobulated margins (originally better seen on ABUS but confirmed on focused US). RIGHT inner retroareolar additional benign appearing oval circumscribed masses including cysts measuring up to 3cm.  Pathology:  LEFT 3:30, 5cm FN (1.8cm mass): IDC. Grade 2. No LVI. l0 mm on core. ER 61-70%. PR negative. Ki-6715-20%. Her-2 negative by FISH  HER2 negative by FISH.  LEFT 3:30, 2.5cm FN (1.3cm mass): IDC. Grade 3. No LVI. 15 mm on core. ER 41-50%. PR 1-10%. Ki-67 30-40%.  HER2 negative by FISH. +LCIS focally.  RIGHT 11:00. 4cm FN (0.9cm mass): Sclerosing adenosis, negative for CA. Felt to be concordant.  -12/22/16: MRI breast - LEFT posterior lateral mass measures 2.4 x 2.5 x 2.9cm, abuts but does not invade the pectoralis major. Extends to within 2mm of overlying skin which is abnormally thickened and retracted, though no abnormal skin enhancement. LEFT anterior-medial smaller mass measures 1.6 x 1.3x 1.5cm. Highly suspicious lower outer (3:30-5:30) segmental nonmass enhancement inferior, medial and between both masses spanning a region 3.6 x 4.2 x 3cm. Few small inner left breast cysts. No axillary adenopathy. RIGHT breast with multiple cysts measuring up to 2.5cm in the outer breast and 3.lcm in the inner breast.  -01/06/17: Mammaprint - low  -01/08/17: CT CAP -  Enhancing mass in the left breast, no axillary, hilar, mammary LAD.  No distant mets  -01/10/17: MRI  LEFT Breast:  1) Index tumor: There are \n two adjacent irregular masses with spiculated margins in the lower outer left breast, posterior depth, best seen on post gad axial image 77/***** and sagittal *****/*****, with a biopsy clip centered in each mass. Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics.    2) Additionally, there is asymmetric non-mass enhancement extending medially from the two index masses, concerning for DCIS. The overall span of the two adjacent masses and NME measures 34 x 32 x 28 mm.    3) There is a 5 mm oval mass in the outer central left breast, anterior depth, best seen on post gad axial image 66/*****. Signal intensity/time curve demonstrates rapid initial rise and washout enhancement kinetics. This mass is approximately 2 cm anterosuperior to the index masses.    RIGHT Breast:  1) There are 2 adjacent foci in the upper outer right breast, anterior depth. One is 4mm and is best seen on post gad axial image 57/*****. Signal intensity/time curve demonstrates rapid initial rise and plateau enhancement kinetics. The other is 3 mm and is best seen on post gad axial image 55/*****. Signal intensity/time curve demonstrates medium initial rise and persistent enhancement kinetics.    2) There is a post biopsy clip in the outer central right breast, middle depth, site of biopsy proven sclerosing adenosis, which was sampled at an outside facility. There is non-mass enhancement just medial to the biopsy clip, best seen on post gad axial image 69/*****, which is likely represents the sclerosing adenosis.  01/11/17: start tamoxifen    01/18/17: start zoladex injection    February 2017: switch from tamoxifen to letrozole  March 2017: d/c letrozole and switch back to tamoxifen September 13 depression  04/09/17:  bilateral mastectomies and left slnb: path pending  November 2018: stopped tamoxifen, switch to OS/letrozole    December 2018: switched from letrozle to exemestane due to anxiety and emotional lability, cont zoladex locally monthly       \u0007 \n Polycystic ovaries 12/26/2016     Last Assessment & Plan:   Long history of amenorrhea, in the last year, they have increased to almost monthly.  Now, increased pain with ovulation and periods, they are heavier and clotting.      \u0007 Cyst of breast 06/16/2015     Last Assessment & Plan:   HISTORY/REASON FOR EXAM: 2 left breast lumps noted on self breast examination.  TECHNIQUE/EXAM DESCRIPTION: Unilateral LEFT digital diagnostic mammography was performed and interpreted with CAD, limited left breast ultrasound.  COMPARISON: 11/26/14 and 06/01/09.  FINDINGS:  3 views of the left breast were obtained.  The breast tissue is extremely dense.  There are is a large rounded benign-appearing nodule in the inferior medial periareolar region of the left breast at the site of one lump.  There is dense breast parenchyma in the lateral mid anterior portion of the left breast at the site of a second lump which was marked.  No spiculated mass or suspicious calcification is identified.  Limited left breast ultrasound: There is a large cyst containing one thin partial septation located at the 10:30 position 1 cm from the nipple measuring 3.9 x 3.0 x 2.1 cm. This accounts for one of the 2 palpable nodules.  There is a small simple cyst at the 3:00 position 3 cm from the nipple measuring 4.2 x 3.8 x 3.3 mm. This accounts for the second palpable nodule.  No solid nodule is present.  IMPRESSION:  1. 2 simple cysts in the left breast accounting for the palpable abnormalities.  2. Cyst aspiration of the larger cyst could be performed if pain persists.  3. If symptoms resolve routine bilateral followup mammography in November 2015 is recommended.  The results were given to the patient at the time of visit.     \u0007 Mass of breast 05/12/2015     Last Assessment & Plan:   mammo done November 2014.      1. BREASTS ARE EXTREMELY DENSE, WHICH LOWERS SENSITIVITY OF MAMMOGRAPHY. NO GROSS EVIDENCE OF MALIGNANCY AND NO INTERVAL CHANGE.    2. SCREENING MAMMOGRAM \n IN ONE YEAR IS RECOMMENDED.    Recently found a couple of lumps left breast.  No nipple discharge.  Some discomfort left breast     \u0007 Neck pain 05/12/2015     Last Assessment & Plan:   Had a MVA last  *****.  Went to ***** ***** ***** for workup.  She recalls imaging as negative.  She continues to have neck pain radiating to left shoulder.  Underwent PT, this made it worse.  She is undergoing rolfing and would like to try some muscle relaxers to take prior to the sessions.  Numbness and tingling in left arm.  Her hand will shake with weight.  Her arm symptoms predated the MVA.  She was seen by the ***** for this complaint just before the MVA.      \u0007 Anxiety disorder 01/12/2015       MEDICATIONS  Current Outpatient Medications   Medication Sig Dispense Refill   \u0007 amoxicillin (AMOXIL) 875 mg tablet TAKE 1 TABLET BY MOUTH TWICE A DAY UNTIL GONE  0   \u0007 amoxicillin-clavulanate (AUGMENTIN) 875-125 mg tablet Take by mouth.     \u0007 benzonatate (TESSALON) 200 mg capsule Take 200 mg by mouth.     \u0007 buPROPion (WELLBUTRIN XL) 150 mg 24 hr tablet Take 150 mg by mouth.     \u0007 doxycycline (VIBRA-TABS) 100 mg tablet TAKE 1 TABLET BY MOUTH TWICE A DAY FOR 10 DAYS  0   \u0007 exemestane (AROMASIN) 25 mg tablet TAKE 1 TABLET BY MOUTH EVERY DAY FOR 30 DAYS  4   \u0007 HYDROcodone-acetaminophen (NORCO) 5-325 mg tablet TAKE 1 TO 2 TABS BY MOUTH EVERY 6 HOURS AS NEEDED FOR PAIN *G89*  0   \u0007 IBU 800 mg tablet TAKE 1 TABLET BY MOUTH EVERY 8 HOURS AS NEEDED  3   \u0007 metroNIDAZOLE (METROGEL) 0.75 % gel Apply thin film to affected area BID     \u0007 ondansetron (ZOFRAN-ODT) 4 mg rapid dissolve tablet DISSOLVE 1 TAB ON THE TONGUE EVERY 6 HOURS AS NEEDED FOR NAUSEA/VOMITING  0     No current facility-administered medications for this visit.        ALLERGIES    Allergies/Contraindications   Allergen Reactions   \u0007 Latex Hives   \u0007 Venom-Honey Bee Anaphylaxis   \u0007 Penicillins Hives   \u0007 Codeine Nausea Only       PHYSICAL EXAM  Vital signs:   Vitals:    01/21/19 1252 \n   BP: 117/66   Pulse: 75   Resp: 18   Temp: 37 C (98.6 F)   TempSrc: Temporal   SpO2: 100%   Weight: 64.2 kg (141 lb 9.6 oz)   Height: 162.9 cm (5' 4.13\")     ECOG: 0  Physical Examination: General appearance - alert, well appearing, and in no distress  Mental status - alert, oriented to person, place, and time  Eyes - pupils equal and reactive, extraocular eye movements intact  Mouth - mucous membranes moist, pharynx normal without lesions  Neck - supple, no significant adenopathy  Lymphatics - no suspicious cervical, SC, axillary lymphadenopathy  Chest - clear to auscultation, no wheezes, rales or rhonchi, symmetric air entry  Heart - normal rate, regular rhythm, normal S1, S2, no murmurs, rubs, clicks or gallops  Abdomen - soft, nontender, nondistended, no masses or organomegaly  Breasts -s/p bilateral mastectomies with implants. No suspicious masses, nodules, rashes bilaterally   Back exam - full range of motion, no tenderness, palpable spasm or pain on motion  Neurological - alert, oriented, normal speech, no focal findings or movement disorder noted  Musculoskeletal - no joint tenderness, deformity or swelling  Extremities - peripheral pulses normal, no pedal edema, no clubbing or cyanosis  Skin - normal coloration and turgor, no rashes, no suspicious skin lesions noted      RESULTS  None to review    ASSESSMENT & PLAN  ***** ***** is a 49 y.o. female with  ER+/PR-/HER2- left breast cancer, mammaprint low risk IDC of the left breast s/p NAHT with tamoxifen, s/p bilateral mastectomies, s/p adjuvant tamoxifen *****/*****-*****/*****, switched to OS/letrozole in November 2018 but switched from letrozole to exemestane in December 2018 due to anxiety and emotional lability.       1. Breast Cancer  -NED on exam  -Continue exemestane 25mg daily. We discussed continuing AI for at least 5 years.   -Recommend checking estradiol level to ensure ovarian suppression, if suppressed can check q3months  -needs DEXA scan now and every 2 years. If \n osteopenia can consider adding bisphosphonate. Reviewed adjuvant bisphosphonate data showing reduced distant mets, improved BCSS, and OS for postmenopausal women and bisphosphonate use.   -she is considering BSO which is reasonable since she will likely be in natural menopause after 5 years of OS/AI.  -continue q6months exams, no imaging given bilateral mastectomies    2. Depression/Anxiety, improved  -Follow up with local psych for ongoing support as needed.      Follow-up in 6 months or sooner if she has new or worsening symptoms.     I have reviewed and updated the patient's past medical history, medications, allergies, family and social histories. I spent a total of 40 minutes face-to-face with the patient and 30 minutes of that time was spent counseling regarding the treatment plan, symptoms and therapeutic options. Patient expresses understanding and agrees with the plan.    *****, ***** *****, am acting as a scribe for services provided by ***** ***** *****, MD on *****/*****/***** 11:46 AM      The above scribed documentation as annotated by me accurately reflects the services I have provided.   ***** ***** *****, MD  *****/*****/***** 11:51 AM      \n
```

---

## Letter A

Dear Patient,

We are writing to summarize your current health and treatment plan. We are glad to see you are doing well, and we are here to support you.

You have a history of breast cancer in the left breast, called invasive ductal carcinoma. Your cancer is hormone receptor positive (ER+), meaning it grows with estrogen, and HER2 negative, meaning it does not have extra HER2 protein.

You have already completed:

Surgery (bilateral mastectomies)
Hormone therapy (tamoxifen) in the past
You are now on ovarian suppression plus an aromatase inhibitor (exemestane)

At your recent visit, there is no evidence of active cancer (NED) on exam, which is very good news.

Current treatment plan:

Continue exemestane (hormone therapy that lowers estrogen)
Continue ovarian suppression
Plan to stay on hormone therapy for at least 5 years

Monitoring and next steps:

Check estradiol levels (to make sure estrogen is low)
Have a DEXA scan (a test to check bone strength) now and every 2 years
Clinic visits every 6 months for follow-up exams

How you are feeling:
You reported:

Mild joint aches
Mild hot flashes and trouble sleeping
Some bone pain and vaginal dryness

These can be side effects of hormone therapy. Please let your care team know if they get worse.

We are happy to hear your mood symptoms have improved. Please continue to follow up for emotional support as needed.

You are doing well, and we will continue to monitor you closely. Please reach out with any questions or concerns.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to see us today. Here’s a summary of what we talked about:

**Diagnosis:**
You have breast cancer. It started in your left breast. The type is called IDC, which means invasive ductal carcinoma. It is at a stage where it has spread a little bit but not far. Your cancer is ER+ (estrogen receptor positive), PR- (progesterone receptor negative), and HER2- (human epidermal growth factor receptor 2 negative).

**Treatment:**
You had surgery to remove both breasts (bilateral mastectomies). After the surgery, you took a medicine called tamoxifen. In November 2018, you started taking another medicine called letrozole. But because you felt anxious and had mood swings, you switched to a different medicine called exemestane in December 2018. You will keep taking exemestane every day.

**Tests:**
We did tests to check how well the treatment is working. Everything looks good so far. We also checked your bones and they look fine too.

**Next Steps:**
- Keep taking exemestane every day.
- We will check your hormone levels to make sure the medicine is working.
- You need to get a special X-ray of your bones (DEXA scan) now and every 2 years.
- You should see your doctor every 6 months for a check-up. We won’t need to do any more scans because you had both breasts removed.

**What to Watch For:**
- If you feel very worried or sad, talk to your local mental health provider.
- Let us know if you have new or worsening symptoms.

Remember, you can always ask your care team any questions you have. We are here to help you.

Best wishes,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your breast cancer treatment.
**What's new or changed since your last visit?**
You are NED (no evidence of disease) on physical exam.
**What treatment or medication changes were made?**
You switched from letrozole to exemestane in December 2018 due to anxiety and emotional lability.
You will continue taking exemestane.
You will also continue taking ondansetron (a medication to prevent nausea) and other supportive medications as needed.
**What is the plan going forward?**
You will continue taking exemestane.
You will have a DEXA scan now and every 2 years to check for bone density issues.
You will check your estradiol level to ensure ovarian suppression. If suppressed, you can check every 3 months.
You will continue having 6-monthly exams, with no imaging needed due to previous bilateral mastectomies.
You are referred to local psychology for ongoing support as needed.
You will have your next visit in 6 months or sooner if you have new or worsening symptoms.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Breast Cancer — Sample 20

**Cancer Type:** Bilateral breast cancer, right breast: ER+/PR+/HER2+ with some lobular differentiation, left breast: ER+/PR+/HER2 0

## Original Clinical Note

```
New patient Visit  Breast Cancer *****    ***** ***** ***** 07/01/2019        NAME: ***** ***** ***** Date of Birth: 01/26/1975   MRN: *****        ***** ***** *****  07/01/2019        Patient Name: ***** ***** *****  *****: 01/26/1975   Medical Record: *****  New Patient Coordination          SUBJECTIVE  Subjective   ***** ***** ***** is a 44 y.o. female who presents for a return visit to the breast cancer center for her  ER+ HER2 positive  early stage breast cancer with the following medical concerns.      HISTORY OF PRESENT ILLNESS:    The patient is here with a recent onset of bilateral of breast cancer that began to grow in March 2019    Relevant medical history:     Patient Active Problem List    Diagnosis Date Noted   \u0007 Breast cancer (HCC) 07/01/2019     44 yo Female felt a lump in the right breast, two episodes of nipple discharge, no redness  She sought medical attention and was scheduled for an ultrasound which happened in *****     *****/*****/***** bilateral ultrasound;  Large bilateral masses right greater than left     Lungs with multiple discrete nodules   No pleural effusion     CT 06/30/19 multiple liver lesions too small to characterize    The patient has lived in ***** until age 11, then moved to ***** *****, moved to ***** ***** at age 27  She has been on birth control since 2014    She has been pregnant once in 2017, one 12 yo son    Otherwise no medical problems    FH: father with colon cancer and liver diagnosed age in late 40's died of presumed liver metastases: age 55, associated with Hep C   MH: no history of cancer  Cousin from father's side: breast cancer in her early 50's     Pathology:   Bilateral breast cancer   Right breast:   ER+, PR+, HER2 positive Ki67 40%      Left breast:   ecadherin +  ER+, PR+   HER2 0    Germ line panel pending: *****     Invasive cancer with some lobular differentiation     Ultrasound:   Right breast 7.3 cm  Left breast: 6 cm                    FAMILY HISTORY:   Family History   Problem \n Relation Name Age of Onset   \u0007 Colon cancer Father       SOCIAL HISTORY:    Social History     Socioeconomic History   \u0007 Marital status: Married     Spouse name: *****   \u0007 Number of children: None   \u0007 Years of education: None   \u0007 Highest education level: None   Occupational History   \u0007 None   Social Needs   \u0007 Financial resource strain: None   \u0007 Food insecurity:     Worry: None     Inability: None   \u0007 Transportation needs:     Medical: None     Non-medical: None   Tobacco Use   \u0007 Smoking status: Never Smoker   Substance and Sexual Activity   \u0007 Alcohol use: None   \u0007 Drug use: None   \u0007 Sexual activity: None   Lifestyle   \u0007 Physical activity:     Days per week: None     Minutes per session: None   \u0007 Stress: None   Relationships   \u0007 Social connections:     Talks on phone: None     Gets together: None     Attends religious service: None     Active member of club or organization: None     Attends meetings of clubs or organizations: None     Relationship status: None   \u0007 Intimate partner violence:     Fear of current or ex partner: None     Emotionally abused: None     Physically abused: None     Forced sexual activity: None   Other Topics Concern   \u0007 None   Social History Narrative   \u0007 None     unchanged and non contributory to this visit      MEDICAL  HISTORY  Past Medical History:   Diagnosis Date   \u0007 Hepatitis B      Medical problems were reviewed, those pertinent today's visit were addressed individually        ROS  Review of Systems   Constitutional: Negative for malaise/fatigue and weight loss. Negative for chills and fever.   Eyes: Negative for blurred vision and double vision.   Respiratory: Negative for cough. Good effort  Cardiovascular: Negative for chest pain and palpitations.   Gastrointestinal: Negative for constipation, diarrhea,  and nausea.   Genitourinary: Negative for dysuria.   Musculoskeletal: Negative for myalgias or weakness.   Skin: Negative for rash.   Neurological: Negative for dizziness and \n headaches.   Endo/Heme/Allergies: Negative for environmental allergies. Does not bruise/bleed easily.   Psychiatric/Behavioral: Negative for depression and suicidal ideas.  '        PHYSICAL EXAM  ECOG1    Vital Signs:  BP 125/79  | Pulse 97  | Temp 37 C (98.6 F) (Temporal)  | Resp 16  | Ht 153.2 cm (5' 0.32\")  | Wt 50.8 kg (112 lb)  | LMP 06/11/2019  | SpO2 98%  | BMI 21.65 kg/m   Wt Readings from Last 3 Encounters:   07/01/19 50.8 kg (112 lb)     Constitutional:  Well developed, well nourished, NAD  HENT:  Normocephalic, Atraumatic,  Eyes: Conjunctiva normal,   Skin:   No rashes  Lymphatics:  No cervical, supraclavicular, axillary, inguinal adaenopathy  Cardiovascular:  Normal heart rate,  .   Respiratory:  Normal breath sounds,   GI:   No masses, No organomegaly.   Musculoskeletal:  No edema, No tenderness,   Neurologic:  Alert & oriented x 3, No focal deficits noted.   Psych:  Linear goal-directed thought    Pertinent findings:   Right breast 12 x 9 cm slightly red, NO LONGER MOBILE   Left breast: 10 X 6 CM SLIGHTLY MOBILE         RESULTS    Labs: reviewed today    Labs today:   @*****@        Imaging reviewed today    @*****@      No scans available for review only reports:    Per report, she has liver and lung nodules   No biopsies and there is no confirmation of metastatic disease     I have individually visualized and interpreted the following studies above:   Based on my review of recent scans, I have concluded the following:  clear evidence of bilateral breast cancer              ASSESSMENT & PLAN  ***** ***** ***** is a 44 y.o. premenopausal female who returns to clinic for follow up of her breast cancer and the following specific problems.        IMPRESSION/PLAN:    Impression: This is a premenopausal patient with a recent diagnosis of a bilateral breast cancer measuring >10 and 9 cam  cm and involving unknown number lymph nodes: The tumor is ER-+ on both sides and HER2 positive on the right. \n       .    The discussions today were based on the assumption that the review of outside records and pathology confirms the current diagnosis. All recommendation will be adjusted after final pathology diagnosis, review of path from her original tumor, FISH assessment and final size and lymph node involvement as well as the ultimate surgical approach.       Plan: Based on the tumor characteristics outline above, the patient a candidate for adjuvant therapy.      A HER2 targeting approach with a chemotherapy back bone. Options include an anthracycline followed by a taxane for 4 cycles each or with the current data preferable six cycles of docetaxel and carboplatin. The latter has a reduced cardiac risk.  This would also include Pertuzumab and  Trastuzumab would be given for a total of 1 year. The side effects were discussed but will be further discussed in detail upon final decision.  Hormonal therapy would not start until completion of chemotherapy and radiation therapy.   Once the patient has completed surgery we will initiate all preparatory studies for chemotherapy which may include a port a cath, ECHO cardiogram and appropriate laboratory and staging studies.     Side effects of this regimens were explained and the patient is made aware of the short term and long term risks, including fevers, myeosuppression, cardiac risks and death.     The patient should have a follow up PET/CT         Surgery will be determined by the response over the course  Of the therapy   radiation to both breast     Follow up on lung and liver lesions    Pain and distress: we discussed the level of distress and pain that the diagnosis brings to the patient. No new intervention is necessary.     Diet and exercise: the importance of diet and exercise with regard to breast cancer outcome was re-itinerated   Follow up: after 2 cycles  Reviewed treatment options, treatment side effects in detail.  I reviewed the patient's medical information \n and medical history. I have reviewed the past medical, family, and social history sections including the medications and allergies listed in the above medical record.    The patient and accompanying family members had ample opportunity to ask questions about the recommendations and follow up and is agreeable to the proposed care plan. The patient may contact our office for further questions or the need for visits sooner than those outlined.     PATIENT COUNSELED REGARDING: risk of recurrence, treatment side effects, length of therapy, imaging schedule  TIME SPENT FACE TO FACE WITH PATIENT: >80 min  Greater than 50% of the total face-to-face time was spent counseling the patient per above    Electronically signed by: ***** ***** *****, *****/*****/***** *****:***** PM                \n
```

---

## Letter A

Dear Patient,

We are writing to explain your diagnosis and your treatment plan. We know this is a lot to take in, and we are here to support you.

You have breast cancer in both breasts (bilateral breast cancer). This means cancer is present in the right and left breasts.

In the right breast, the cancer is hormone receptor positive (ER+/PR+) and HER2 positive.
In the left breast, the cancer is hormone receptor positive (ER+/PR+) and HER2 negative.

The tumors are large, measuring about 7 cm on the right and 6 cm on the left.

Scans have shown small spots in the lungs and liver, but it is not yet confirmed if these are cancer. More testing is needed.

Treatment plan:
Your care team recommends starting with systemic therapy (treatment that goes through the whole body):

Chemotherapy (medicine to kill cancer cells)
Targeted therapy (drugs like trastuzumab and pertuzumab that specifically target HER2 on the right-sided cancer)

These treatments are usually given before surgery to shrink the tumors.

After this, your plan may include:

Surgery, depending on how the cancer responds
Radiation therapy
Hormone therapy (to block estrogen), which will start after chemotherapy and radiation

Next steps:

Repeat imaging (such as PET/CT)
Heart test (echocardiogram) before treatment
Possible placement of a port (a small device under the skin to give medicine)

We understand this is overwhelming. Please ask any questions you have. We are here to support you every step of the way.

Warmly,
Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear [Patient Name],

Thank you for coming to our clinic. We want to make sure you understand your health situation clearly.

**Diagnosis:**
You have breast cancer. It is in both breasts. The cancer is called "early stage," which means it hasn't spread very far yet.

**Tests:**
We did some tests like ultrasounds and CT scans. These tests showed that you have lumps in both breasts. There are also some small spots in your lungs and liver, but we don’t know if they are cancer yet.

**Treatment Plan:**
The doctors think you will need several types of treatments:
- **Surgery:** To remove the cancer from your breasts.
- **Chemotherapy:** Medicine to kill cancer cells.
- **Targeted Therapy:** Special medicine to fight the HER2 type of cancer.
- **Radiation Therapy:** Strong light to kill any remaining cancer cells.

**Next Steps:**
After surgery, we will start preparing for chemotherapy. This might include getting a special tube called a "port" to give you medicine. We will also do more tests to check how well the treatment is working.

**What to Watch For:**
Please let us know if you feel very tired, have a fever, or if you notice any new lumps or changes in your body.

**Important Reminders:**
- Keep all your appointments.
- Eat healthy food and try to stay active.
- If you have any questions, please talk to your doctor or nurse.

We are here to support you every step of the way. Please feel free to reach out to us anytime.

Warm regards,

[Your Name]  
Medical Communication Specialist  
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter C

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a first visit to discuss your newly diagnosed bilateral breast cancer and the treatment options.
**What's new or changed since your last visit?**
You have bilateral breast cancer. The right breast tumor measures 7.3 cm and is ER+, PR+, and HER2 positive with some lobular differentiation (cancer that started in the milk-producing glands). The left breast tumor measures 6 cm and is ER+ and PR+ but does not have a protein called HER2. Imaging shows nodules in the liver and lungs, but it’s not clear if these are cancerous.
**What treatment or medication changes were made?**
You are a candidate for adjuvant therapy (treatment given after surgery to prevent the cancer from coming back). The plan includes a HER2 targeting approach with chemotherapy. Options include either four cycles of an anthracycline followed by four cycles of a taxane or six cycles of docetaxel and carboplatin. You will also receive Pertuzumab and Trastuzumab for a total of one year. Hormonal therapy will start after completing chemotherapy and radiation therapy.
**What is the plan going forward?**
You will have radiation to both breasts. A follow-up PET/CT scan is planned. After surgery, preparatory studies for chemotherapy will be initiated, which may include placing a port-a-cath. Your next clinic visit is after two cycles of treatment.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Part II: Pancreatic Cancer (20 Samples × 2 Systems)

# Pancreatic Cancer — Sample 1

**Cancer Type:** Locally advanced adenocarcinoma of the pancreas

## Original Clinical Note

```
SUBJECTIVE:    This is a very pleasant 47-year-old African-American gentleman with a diagnosis of locally advanced adenocarcinoma of the pancreas. He presented with several months of vague abdominal discomfort and had some gross hematuria in September. He went to ***** Hospital and a CT scan was done which revealed a pancreatic mass. His hematuria was felt to be secondary to renal lithiasis. This problem resolved.    He was noted to have biliary obstruction and underwent an ERCP with an EUS at ***** on October 24. A plastic stent was placed. Fine-needle aspiration of the mass revealed adenocarcinoma.    When I initially saw him, he had lost 43 pounds and was having a lot of pain. We elected to start gemcitabine and Abraxane. He's completed 6 full cycles. At the end of 2 cycles, his CT scan suggested progression. However, since he was doing so well clinically, we felt that this represented pseudo-progression.    His 5th cycle treatment was interrupted because of biliary ductal obstruction. He had a plastic stent was replaced with an uncovered metal stent. Apparently there was a lot of pus and material that drained at the time of that procedure and he was placed on IV and oral antibiotics. Since the procedure, he has had intermittent abdominal discomfort, usually at night, requiring oxycodone.     He feels that his appetite has returned. He's gained kilogram last month. He notices some numbness and tingling in his hands but it doesn't interfere with his function. He is having regular bowel movements.    He would like to meet with psycho oncology as he is having some difficulty coping with his disease and with some social and economic issues.    His girlfriend gave birth to a baby girl on April 27.     PAST MEDICAL HISTORY:     MEDICAL ILLNESSES:  Hypertension, voluntarily stopped his antihypertensive    PRIOR SURGERIES:  Drainage of a perirectal abscess, age 17    INJURIES:  None reported    CURRENT MEDICATIONS:  See \n intake    ALLERGIES:  Allergies   Allergen Reactions   \u0007 Penicillins Hives       FAMILY HISTORY:  There is no cancer among his first-degree relatives. He has 2 siblings. His maternal grandmother had colorectal cancer diagnosed at age 55. A paternal grandmother may have colorectal cancer, age of diagnosis uncertain.    SOCIAL HISTORY:  He smoked about a half a pack of cigarettes per day for about 4 years, quitting in 2006. There is a prior history of alcohol abuse where he drank up to a pint of tequila day, but he has been drinking only socially more recently. He has a past history of heroin abuse, but has not used for several years.   He is accompanied by his mother ***** who works as a *****'***** ***** and his sister ***** who is an administrative *****.      PE:    Vitals:   Vitals: BP 131/90 | Pulse 64 | Temp 36.9 C (98.5 F) (Oral)  | Resp 20 | Ht 182 cm (5' 11.65\") Comment: October 2015 | Wt 94.3 kg (208 lb) | SpO2 100% | BMI 28.48 kg/m2   In general, he appears comfortable.  HEENT no icterus      LABORATORY RESULTS:  Results for orders placed or performed during the hospital encounter of 05/28/16   Complete Blood Count with 5-part Differential   Result Value Ref Range    WBC Count 6.7 3.4 - 10 x10E9/L    RBC Count 3.89 (L) 4.4 - 5.9 x10E12/L    Hemoglobin 11.1 (L) 13.6 - 17.5 g/dL    Hematocrit 33.5 (L) 41 - 53 %    MCV 86 80 - 100 fL    MCH 28.5 26 - 34 pg    MCHC 33.1 31 - 36 g/dL    Platelet Count 203 140 - 450 x10E9/L    Neutrophil Absolute Count 4.06 1.8 - 6.8 x10E9/L    Lymphocyte Abs Cnt 1.49 1.0 - 3.4 x10E9/L    Monocyte Abs Count 0.84 (H) 0.2 - 0.8 x10E9/L    Eosinophil Abs Ct 0.23 0.0 - 0.4 x10E9/L    Basophil Abs Count 0.02 0.0 - 0.1 x10E9/L    Imm Gran, Left Shift 0.02 <0.1 x10E9/L   Urea Nitrogen, Serum / Plasma   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 13 6 - 22 mg/dL   Creatinine, Serum / Plasma   Result Value Ref Range    Creatinine 1.06 0.61 - 1.24 mg/dL    eGFR if non-African \n American 83 >60 mL/min    eGFR if African Amer 96 >60 mL/min   Aspartate Transaminase   Result Value Ref Range    Aspartate transaminase 20 17 - 42 U/L   Alanine Transaminase   Result Value Ref Range    Alanine transaminase 25 12 - 60 U/L   Bilirubin, Total   Result Value Ref Range    Bilirubin, Total 0.9 0.2 - 1.3 mg/dL   Alkaline Phosphatase   Result Value Ref Range    Alkaline Phosphatase 60 31 - 95 U/L   Cancer Antigen 19-9   Result Value Ref Range    Cancer Antigen 19-9 3 <36 U/mL       IMAGING:  Ct Abdomen /pelvis With Contrast    Result Date: 06/10/2016  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** *****:***** PM CLINICAL HISTORY: 48-year-old male with pancreatic cancer COMPARISON:  01/28/2016 CT abdomen pelvis, 03/18/2016 CT RADIATION DOSE INDICATORS: ***** ***** received 2 exposure event(s) (excluding scout) during this CT study. The CTDIvol (mGy) and DLP (mGy-cm) radiation dose values for each event are: Event: 1;  anatomic area: Chest;  phantom: body;  CTDIvol: 14.2;  DLP: 533.3 Event: 2;  anatomic area: Chest;  phantom: body;  CTDIvol: 18.8;  DLP: 888.6 The following accession numbers are related to this dose report [*****]: ***** The dose indicators for CT are the volume Computed Tomography Dose Index (CTDIvol) and the Dose Length Product (DLP), and are measured in units of mGy and mGy-cm, respectively. These indicators are not patient dose, but values generated from the CT scanner acquisition factors.  Individual patient doses calculated from these values will depend on patient size and can substantially underestimate or overestimate the actual dose. For additional information on radiation dose, safety, and protection in the UCSF Department of Radiology and Biomedical Imaging, please refer to one of our FAQ leaflets on \"Computed Tomography (CT)\" or go to *****://*****.*****.*****.*****/*****/*****. TECHNIQUE: Following the administration of 150 cc of Omnipaque 350, contiguous 1.25-mm collimation axial \n images were obtained through the abdomen and pelvis. The gastrointestinal tract was opacified with oral and rectal  contrast. Coronal and sagittal reformats were also obtained. FINDINGS: Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Bones:  No suspicious lesions Liver:  6 mm hypervascular lesion in hepatic segment 8 which may reflect flash filling hemangioma or portosystemic shunt, unchanged from prior No significant intrahepatic biliary duct dilatation. Gallbladder:  Unremarkable Spleen:  Unremarkable Pancreas:  Redemonstration of ill-defined hypoattenuating mass centered in the pancreatic head, extending to the pancreatic tail, and involving the portosplenic confluence and superior mesenteric vein. Although difficult to measure this appears minimally increased in size and appears prior. Minimally decreased pancreatic duct dilatation to 7 mm in the pancreatic tail. The stent has been changed in courses from the common hepatic duct into the duodenum persistent pneumobilia suggests patency. Adrenal Glands:  Unremarkable Kidneys:  Unremarkable GI Tract:  No bowel obstruction. Pancreatic mass continues to abut the duodenum, unchanged from prior. Vasculature: No significant change in vascular involvement. Lymphadenopathy: Small mesenteric and retroperitoneal nodes, none of which are enlarged by size criteria. Ascites: Absent Bladder: Unremarkable Reproductive organs: Unremarkable Extraperitoneal soft tissues: Unremarkable     1.  Stable to minimally increased size of pancreatic mass compatible with known pancreatic adenocarcinoma. 2.  No evidence of metastatic disease in the abdomen/pelvis. END OF IMPRESSION:     Ct Chest With Contrast    Result Date: 06/10/2016  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Panceatic cancer COMPARISON: 03/18/2016 and earlier TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the \n administration of intravenous contrast. RADIATION DOSE INDICATORS: ***** ***** received 2 exposure event(s) (excluding scout) during this CT study. The CTDIvol (mGy) and DLP (mGy-cm) radiation dose values for each event are: Event: 1;  anatomic area: Chest;  phantom: body;  CTDIvol: 14.2;  DLP: 533.3 Event: 2;  anatomic area: Chest;  phantom: body;  CTDIvol: 18.8;  DLP: 888.6 The following accession numbers are related to this dose report [*****]: ***** The dose indicators for CT are the volume Computed Tomography Dose Index (CTDIvol) and the Dose Length Product (DLP), and are measured in units of mGy and mGy-cm, respectively. These indicators are not patient dose, but values generated from the CT scanner acquisition factors.  Individual patient doses calculated from these values will depend on patient size and can substantially underestimate or overestimate the actual dose. For additional information on radiation dose, safety, and protection in the UCSF Department of Radiology and Biomedical Imaging, please refer to one of our FAQ leaflets on \"Computed Tomography (CT)\" or go to *****://*****.*****.*****.*****/*****/*****. FINDINGS: LUNGS: Few new small nodules in the left lower lobe measuring up to 3 mm (series 2, images 121, 138, 166, 170) and a 7 mm new nodule in the lingula. Previously noted tiny nodule in the peripheral right upper lobe (prior study series 2, image 90) is not visualized on this exam. Right middle lobe nodule is unchanged (series 2, image 129). PLEURA: No pleural effusion. MEDIASTINUM: 8 mm lymph node adjacent to the inferior vena cava (previously 7 mm; series 2, image 179). No lymphadenopathy by size criteria. HEART/GREAT VESSELS: Normal heart size. No pericardial effusion. BONES/SOFT TISSUES: No suspicious lesions. VISIBLE ABDOMEN: Please see dedicated Abdomen and Pelvis CT report.     1.  5 new nodules measuring up to 7 mm in the lingula and left lower lobe. These are indeterminate \n but are suspicious for metastases in the setting of known pancreatic adenocarcinoma. Attention on follow-up. 2.  No other definite evidence of metastatic disease in the chest. END OF IMPRESSION:       ASSESSMENT:  1. Locally advanced adenocarcinoma of the pancreas, with possible progression  2. Hypertension  3. Possible cancer family syndrome  4. ECOG 0     PLAN:    The new lung nodules are concerning but these are unlikely to become a problem for him. He would really like a chemotherapy break. This seems reasonable since he appears to have received maximum benefit on this regimen. Given the sites of disease involvement and his lack of symptoms, I think it would be safe to put him on surveillance. We'll see him again in 2 months for follow-up. I spent 20 minutes in face-to-face consultation with the patient and his mother today going over all aspects of his care and management.    \n
```

---

## Letter A

Dear Patient,

I hope this letter finds you well. I wanted to share some important information about your health and the next steps in your care.

**Diagnosis and Stage:**
You have a type of cancer called locally advanced adenocarcinoma of the pancreas. This means the cancer started in your pancreas and has grown quite large.

**Treatment Plan:**
You have been receiving two medicines called gemcitabine and Abraxane. You have finished six full treatments. Recently, you had a small tube called a stent placed in your bile duct to help with blockage. 

**Key Test Results:**
The latest scans show that the tumor in your pancreas has gotten a little bigger. However, there are no signs of the cancer spreading to other parts of your body. Some small spots were found in your lungs, but they are not causing any problems right now.

**Next Steps:**
We think it’s okay for you to take a break from chemotherapy. We will keep watching you closely. You will come back to see us in two months for another check-up.

**What to Watch For:**
Please let us know if you have any new pain, changes in how you feel, or any other concerns. It’s good to know that your appetite has returned and you’ve gained some weight. Keep an eye out for any new symptoms.

**Additional Support:**
You mentioned wanting to talk to someone about coping with your illness. We can set up an appointment with a psychologist who specializes in helping people with cancer.

If you have any questions or need more information, please talk to your care team. We are here to support you.

Best wishes,
[Your Name]
Medical Communication Specialist

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer.
**What's new or changed since your last visit?**
There are new nodules in your lungs that might be cancer spread. Your pancreatic tumor has stayed about the same size. You've gained some weight, which is good. You still have some nighttime stomach pain that needs oxycodone.
**What treatment or medication changes were made?**
You asked for and were given a break from chemotherapy because you've had the best response possible from the treatment.
**What is the plan going forward?**
We will watch your condition closely without giving you any more chemotherapy for now. You will come back to the clinic in 2 months for a follow-up visit.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 2

**Cancer Type:** Adenocarcinoma of the pancreas with direct invasion of the duodenum

## Original Clinical Note

```
This is an independent visit    ATTENDING: ***** *****, MD     PATIENT IDENTIFICATION: Mr. ***** is a 71 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer Center for follow-up of his diagnosis of Adenocarcinoma of the pancreas, status post resection, now with new metastatic disease.    ONCOLOGIC HISTORY:   He presented with abdominal discomfort and biliary ductal obstruction. In March 2018, and an ERCP and stent was placed.    04/01/2018: Underwent a Whipple for pancreatic adenocarcinoma in *****. Experienced postoperative complications of internal bleeding that required an additional laparotomy. He had direct invasion of the tumor into the duodenum and 6 of 25 lymph nodes were positive. Postoperatively, he also had a small bowel obstruction which was relieved by nasogastric suction. He had a port placed but this was removed as it became infected and he also experienced sepsis with klebsiella. This was treated with Cipro.    05/25/18: New baseline CT CAP reveals evidence of new liver metastases    06/05/18: C1D1 FOLFIRINOX.  CA 19-9 was 17,090.  06/12/18: Presented to UCSF ED with fever of 38.3 C. CT was negative for any acute findings, but did reveal worsening metastatic disease to liver. CXR clear and UA negative.   06/21/18: C2D1 FOLFIRINOX   07/05/18: C3D1. CA 19-9 was 21,247.  07/19/18: C4D1  08/01/18: CT CAP reveals decrease in size of hepatic mets, interval increase in pancreatic ductal dilatation.   08/02/18: C5D1. CA 19-9 dropped to 6,040.  08/16/18: C6D1  08/30/18: C7D1. CA 19-9 was 1,909.  09/13/18: C8D1  09/26/18: CT CAP reveals slight increase in the size of liver mets. However, given that his CA 19-9 continued to drop and patient is asymptomatic, Dr. ***** decided to continue him on FOLFIRINOX.    10/03/18: C9D1. CA 19-9 further dropped to 789.   10/18/18: C10D1  11/01/18: C11D1. CA 19-9 = 839  11/15/18: C12D1  11/24/18:  CT CAP indicates disease progression. CA 19-9 = \n 1,625.  11/29/18: C1D1 Gemcitabine + Abraxane   12/13/18: C1D15    INTERVAL HISTORY:   Patient's daughter interpreted for today's visit    Patient presents for C2D1. Patient once again developed intermittent fevers for 4-5 days following treatment. His numbness in his hands is \"less intense\" but has spread to the palms of his hands, also the soles of bilateral feet also feel more numb. Denies any problems with the use of his hands and denies feeling unsteady on his feet. He is also experiencing more fatigue.     He plans to travel to ***** to visit his sister and brother-in-law from December 30 - January 11.     All other systems were reviewed and are negative.    Medications the patient states to be taking prior to today's encounter.   Medication Sig   \u0007 ALPRAZolam (XANAX) 0.25 mg tablet Take 1 tablet (0.25 mg total) by mouth nightly as needed for Sleep.   \u0007 atorvastatin (LIPITOR) 40 mg tablet Take 1 tablet (40 mg total) by mouth Daily.   \u0007 diclofenac (VOLTAREN) 1 % GEL gel APPLY TO AFFECTED AREAS TOPICALLY PRN FOR PAIN   \u0007 DOMPERIDONE, BULK, MISC Take 10 mg by mouth Twice a day.    \u0007 granisetron (SANCUSO) 3.1 mg/24 hour patch Apply patch at least 24 hrs prior to chemo. Remove minimum of 24 hrs after chemo completion. Patch may be worn up to 7 days   \u0007 INSULIN ASPART 100 UNIT/ML SUBCUTANEOUS PEN, CONTINUOUS TPN/TF/NPO,    \u0007 lidocaine-diphenhydramine-maalox (MAGIC MOUTHWASH) 200-25-400-40 mg/30 mL MWSH mouthwash Take 15 mLs by mouth 4 (four) times daily as needed (painful mouth sores & throat pain).   \u0007 lipase-protease-amylase (CREON) 36,000-114,000- 180,000 unit capsule TAKE 2 CAPSULES BY MOUTH WITH EACH MEAL AND 1 CAPSULE BY MOUTH WITH EACH SNACK. TAKE WITH OR DURING MEAL   \u0007 multivit with min-folic acid (OMNICAP) 0.4 mg TAB    \u0007 nitroGLYCERIN 6.5 mg CR capsule Take 6.5 mg by mouth Daily.    \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet three times a day as needed for nausea and vomiting. Start 2 days after last dose of \n palonosetron (Aloxi) is given.   \u0007 pantoprazole (PROTONIX) 40 mg EC tablet Take 40 mg by mouth Daily.   \u0007 pantoprazole 40 mg GrPS    \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting).   \u0007 tamsulosin (FLOMAX) 0.4 mg 24 hr capsule Take 1 capsule (0.4 mg total) by mouth Daily.     ALLERGIES:  Allergies/Contraindications  No Known Allergies     Medical History     Diagnosis Date Comment Source    COPD (chronic obstructive pulmonary disease) (HCC)   Provider    Diabetes mellitus (HCC)   Provider    Hypertension   Provider    Pancreatic adenocarcinoma (HCC)   Provider        Surgical History     Procedure Laterality Date Comment Source    IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY)  05/29/2018 IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY) 05/29/2018 ***** *****. *****, MD RAD IR MZ Provider    PANCREAS SURGERY  2018  Provider        Family History     Relation Problem Comments    Father Pancreatic cancer        Maternal Uncle Pancreatic cancer        Paternal Aunt Pancreatic cancer        Paternal Aunt Pancreatic cancer            Social History     Socioeconomic History   \u0007 Marital status: Married     Spouse name: *****   \u0007 Number of children: None   \u0007 Years of education: None   \u0007 Highest education level: None   Occupational History   \u0007 None   Social Needs   \u0007 Financial resource strain: None   \u0007 Food insecurity:     Worry: None     Inability: None   \u0007 Transportation needs:     Medical: None     Non-medical: None   Tobacco Use   \u0007 Smoking status: Former Smoker   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: No   \u0007 Drug use: No   \u0007 Sexual activity: None   Lifestyle   \u0007 Physical activity:     Days per week: None     Minutes per session: None   \u0007 Stress: None   Relationships   \u0007 Social connections:     Talks on phone: None     Gets together: None     Attends religious service: None     \n Active member of club or organization: None     Attends meetings of clubs or organizations: None     Relationship status: None   \u0007 Intimate partner violence:     Fear of current or ex partner: None     Emotionally abused: None     Physically abused: None     Forced sexual activity: None   Other Topics Concern   \u0007 None   Social History Narrative    He is a commercial ***** but is now retired. He and his wife are now living with her daughter in ***** *****. Before he got sick, he liked to travel and manage his garden. He has a 50-pack-year history of smoking and just stopped 3 months ago. He is not using nicotine replacement. He does not drink alcohol.     PHYSICAL EXAM:  Objective   Vital Signs:   Vitals:    12/27/18 1058   BP: 133/69  Comment: ALL VST FROM IC   Pulse: 79   Resp: 18   Temp: 36.9 C (98.4 F)   TempSrc: Temporal   SpO2: 100%   Weight: 55.8 kg (123 lb)   Height: 166.6 cm (5' 5.59\")  Comment: September 2018 @ *****   PainSc:  0      Pain: 1  ECOG: 0  Constitutional:  WDWN, NAD. +Thin.   Skin: warm, moist. No rashes or lesions  Eyes: sclera anicteric, PERRL, EOMI   HENT: No open sores in his buccal mucosa, no erythema/edema. sinuses nontender, nasal mucosa intact, pharynx without erythema, no thrush   Lymph Nodes: no cervical, supraclavicular, or axillary lymphadenopathy    Respiratory: resonant to percussion throughout, CTA bilaterally, no wheezes, rhonchi, or crackles  Cardiovascular: Normal heart rate, Normal rhythm, No murmurs, No rubs, No gallops. No LE edema   GI: Bowel sounds normal, Soft, No tenderness or distention. No HSM   Musculoskeletal: No point tenderness at cervical, thoracic or lumbar spine, muscular strength equal throughout   Neurologic: +Right arm with slight resting tremor. Alert & oriented x 3, ambulates w/o difficulty, CN II-XII intact.  Psychiatric: Mood stable. No HI, SI, hallucinations, paranoia, or delusions.    DIAGNOSTIC STUDIES:   Laboratory data:   Lab Results   Component \n Value Date    WBC Count 7.4 12/27/2018    Neutrophil Absolute Count 4.62 12/27/2018    Hemoglobin 9.4 (L) 12/27/2018    Hematocrit 30.1 (L) 12/27/2018    Platelet Count 351 12/27/2018    Creatinine 0.58 (L) 12/27/2018    Bilirubin, Total 0.4 12/27/2018    Alkaline Phosphatase 305 (H) 12/27/2018    Aspartate transaminase 33 12/27/2018    Alanine transaminase 31 12/27/2018    Albumin, Serum / Plasma 3.0 (L) 11/24/2018    Int'l Normaliz Ratio 1.1 05/29/2018    Sodium, Serum / Plasma 137 11/24/2018    Potassium, Serum / Plasma 3.9 11/24/2018    Calcium, total, Serum / Plasma 8.7 (L) 11/24/2018     Lab Results   Component Value Date    Cancer Antigen 19-9 900 (H) 12/13/2018    Cancer Antigen 19-9 634 (H) 12/07/2018    Cancer Antigen 19-9 1,625 (H) 11/24/2018    Carcinoembryonic Antigen 8.3 (H) 12/13/2018    Carcinoembryonic Antigen 8.5 (H) 12/07/2018    Carcinoembryonic Antigen 10.6 (H) 11/24/2018     Germline and Molecular testing:  - Patient is a SPINK1 carrier.   - He has variant of unknown significance in FANCG and NF2.    Radiographic data:   No recent imaging reviewed    ASSESSMENT AND PLAN:   In summary, Mr. ***** is a 71 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer Center for follow-up of his diagnosis of Adenocarcinoma of the pancreas, status post resection, now with new metastatic disease.    #pancreatic adenocarcinoma  He was most recently progressed on FOLFIRINOX, and has since started 2nd line therapy with Gemcitabine + Abraxane on 11/29/18.    His labs and PS are acceptable and he agrees to proceed with C2D1 Gem/Abrax. We discussed the possibility of having to reduce his Abraxane dose in the future if his peripheral neuropathy worsens. Given that he has grade 1 neuropathy at this point, no dose modifications are indicated. His labs and PS are acceptable and he will proceed with C2D1 today. CT scan due after he completes cycle 2.     #Nausea/vomiting: Improved  - \n Likely June 24 chemotherapy   - Premed with Aloxi & Emend.   - On day 3 patient will apply granisetron patch 3.1 mg/24 hours, and take compazine PRN.   - IV hydration on additional days 3 & 11 PRN   - Patient will also continue to use cannabis product.     #pancreatic enzyme insufficiency   - Continue Creon 2 tab/meal and 1 tab/snacks    #SPINK1 gene mutation: SPINK1: c.101A>G  - Patient is a carrier, but does not have SPINK1-related risks that are due to having TWO mutations. His family needs to be screened for SPINK1 which is classically a gene and some familial pancreatitis but also leads to pancreatic cancer.    #R UE tremor & bilateral foot drop L>R: resolved  - EMG on 09/15/18 to evaluate foot drop revealed the following:   \"evidence for bilateral, predominantly demyelinating peroneal neuropathies across the fibular head...Demyelinating peroneal neuropathies at the fibular head can be seen in patients who frequently cross their legs, particularly after a period of weight loss (as is the case in this patient)\"  - Followed by his PCP, Dr. *****.   - Continue to monitor    RETURN TO CLINIC: 2 weeks    As always, the patient was counseled to contact the clinic or the after-hours physician on call for any new symptoms or complications in the interim between now and her follow up visit with me.    I spent a total of 25 minutes face-to-face with the patient and >50% of that time was spent counseling regarding the symptoms, treatment plan, risks and/or therapeutic options for the diagnoses above.    ***** *****, NP    \n
```

---

## Letter A

Dear Mr. *****,

Thank you for coming to see us today. Here is a summary of your visit:

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. The cancer has spread to other parts of your body, which we call metastatic disease.

**Treatment Plan:**
You are now receiving a new treatment called Gemcitabine plus Abraxane. This treatment helps control the cancer. If you start to feel more numbness in your hands and feet, we might need to adjust the dose of one of these medicines.

**Key Test Results:**
We checked your blood and found that your levels of a protein called CA 19-9 are high. This protein can show how well the treatment is working. Your levels have gone down, which is good news.

**Next Steps:**
You will come back to see us in two weeks. We will do another CT scan after you finish this round of treatment.

**What to Watch For:**
Please watch for any new symptoms like pain, fever, or feeling very tired. If you notice anything new or different, please call us right away.

**Medications:**
You are taking several medications to help with pain, nausea, and other symptoms. Please keep taking them as directed.

**Travel Plans:**
It’s great that you’re planning to visit your sister and brother-in-law. Make sure to take all your medications with you and stay in touch with us if you have any concerns.

If you have any questions or need more information, please talk to your care team. We are here to support you.

Warm regards,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Your cancer is showing signs of disease progression. This is seen in the increase in the size of liver metastases and the rise in CA 19-9 levels.
**What treatment or medication changes were made?**
You switched to Gemcitabine + Abraxane on 11/29/18. You will continue to use supportive medications such as granisetron patches and ondansetron tablets to manage side effects.
**What is the plan going forward?**
You will continue with C2D1 Gemcitabine + Abraxane. If your peripheral neuropathy worsens, the dose of Abraxane may need to be reduced. A CT scan is due after you complete cycle 2. Your next visit is in 2 weeks for follow up.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 3

**Cancer Type:** Metastatic pancreatic ductal adenocarcinoma

## Original Clinical Note

```
Mr. ***** is a 74 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer Center for follow-up of his metastatic pancreatic cancer, originally diagnosed 2018.    Treatment summary to date:  -He initially received 4 cycles of FOLFIRINOX, through early November 2017; this was quite poorly tolerated overall, with progressive anorexia and fatigue. He then held off on further treatment for the next couple of months.  -After our initial meeting on 01/14/2018, he was agreeable to try additional chemotherapy with FOLFOX, and has now completed 13 cycles through 08/19/2018. The 5-FU bolus was d/c'ed beginning cycle #4 due to mild thrombocytopenia (plts = 88K);beginning cycle #9, his dosing interval was spaced out from q2 to q3 weeks; and beginning cycle #11, the oxaliplatin was dropped altogether. He completed 13 cycles of treatment altogether, through 08/17/2018.  - On chemotherapy holiday over the next 2  months so he could enjoy his daughter's upcoming ***** in mid-September; repeat imaging thereafter (10/14/2018) showed evidence of both locoregional and metastatic disease progression  - Started on 2L treatment with the combination of gemcitabine plus nab-paclitaxel on a QOW dosing schedule (days 1 and 15 of a 28-day cycle) beginning 10/26/2018. After 2 treatment cycles, was noted to have further disease progression in pancreas, liver, and spleen, with waxing and waning size of multiple peritoneal implants  - Switched to 3L treatment with the combination of 5-FU/LV plus nanoliposomal irinotecan (nal-IRI, or Onivyde), beginning 12/28/2018; received 4 treatment cycles through the end of January. Well-tolerated overall, with only mild fatigue (not preventing him from playing tennis) and minimal GI side fx. Repeat imaging (02/15/2019) demonstrated overall stable disease, with slightly improving hepatic metastases and simultaneously stable to minimally enlarging peritoneal metastases  - Interrupted rx to \n take a17-day trip to *****; did well for most of the trip, but toward the end and since returning home on 03/06/2019, began noting increasing abdominal/mid-back pain. Was noted to be anemic (Hgb 7.0) without evident GI bleed, so underwent PRBC tx x 2 units and underwent EGD (03/16/2019) notable for partial gastric outlet obstruction, through which a 25 mm by 8 cm enteric stent was placed.  - Resumed chemotherapy (cycle #5 of nal-IRI/5-FU/LV) on 03/29/2019.    Interval history/review of systems  - Tolerating POs adequately since stent placement, w/o postprandial vomiting; however, appetite generally poor  - Occasional bilateral abdominal pain; uses Norco twice-daily  - Generalized fatigue, with waxing/waning energy level  - No melena or suggestion of GI bleeding  - No fevers/chills      Medications the patient states to be taking prior to today's encounter.   Medication Sig   \u0007 acetaminophen (TYLENOL ORAL) Take 200 mg by mouth As Needed.    \u0007 aspirin 81 mg EC tablet aspirin 81 mg tablet,delayed release   Take 1 tablet every day by oral route.   \u0007 carvedilol (COREG) 3.125 mg tablet    \u0007 HYDROcodone-acetaminophen (NORCO) 5-325 mg tablet 1-2 tabs q6 hours prn pain   \u0007 ibuprofen (ADVIL,MOTRIN) 200 mg tablet Take 200 mg by mouth As Needed.    \u0007 loperamide (IMODIUM) 2 mg capsule Take two capsules by mouth at the onset of diarrhea, then take one capsule every 2 hours until diarrhea-free for 12 hours.   \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 tablet (1 mg total) by mouth every 12 (twelve) hours as needed for Anxiety (nausea).   \u0007 pantoprazole (PROTONIX) 20 mg tablet Take 1 tablet (20 mg total) by mouth Daily.   \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting).   \u0007 rosuvastatin (CRESTOR) 20 mg tablet          Allergies/Contraindications  No Known Allergies      Comprehensive past medical, family, and social history which was performed during a \n previous encounter was re-examined and reviewed with the patient. There is nothing new to add at today's visit. For details, please refer to my original consultation note in this chart.      Physical Exam:    Blood pressure 117/73, pulse 75, temperature 36.7 C (98 F), temperature source Temporal, resp. rate 16, height 164.9 cm (5' 4.92\"), weight 54.5 kg (120 lb 1.6 oz), SpO2 99 %.    Constitutional: Non-toxic-appearing, moderately cachectic, in no acute distress.  Skin: Warm and dry; non-jaundiced.  Eyes: No sclericterus.  Abdomen: Soft, non-tender, tympanitic to percussion. No hepatosplenomegaly.  Extrems: No LE edema  Neuro: Normal gait. No focal motor deficits.  Psych:  Appropriately interactive. No evidence of depression/psychosis.        Relevant Diagnostic Studies:    Most recent labs notable for the following:    Lab Results   Component Value Date    WBC Count 7.4 03/29/2019    Neutrophil Absolute Count 5.37 03/29/2019    Hemoglobin 7.6 (L) 03/29/2019    Hematocrit 25.1 (L) 03/29/2019    Platelet Count 227 03/29/2019    Creatinine 0.66 03/29/2019    Bilirubin, Total 0.4 03/29/2019    Alkaline Phosphatase 207 (H) 03/29/2019    Aspartate transaminase 24 03/29/2019    Alanine transaminase 18 03/29/2019    Albumin, Serum / Plasma 2.7 (L) 03/14/2019    Int'l Normaliz Ratio 1.2 03/13/2019    Sodium, Serum / Plasma 137 03/17/2019    Potassium, Serum / Plasma 2.8 (LL) 03/17/2019     Lab Results   Component Value Date    Cancer Antigen 19-9 142 (H) 03/29/2019    Cancer Antigen 19-9 15 03/08/2019    Cancer Antigen 19-9 20 01/25/2019    Carcinoembryonic Antigen 2.3 05/18/2018    Carcinoembryonic Antigen 2.2 04/20/2018    Carcinoembryonic Antigen 3.0 03/02/2018         Most recent imaging was personally reviewed and interpreted in conjunction with formal radiology reports, as follows:     Result Date: 02/16/2019  CT ABDOMEN/PELVIS WITH CONTRAST    FINDINGS: Pancreatic tumor: 1)  Location:  Tail of the pancreas 2)  Size: \n  The multicystic, ill-defined mass measures 4.7 x 5.6 cm, previously measuring 4.6 x 5.9 cm when ***** in a similar fashion. 3)  Enhancement relative to pancreas: Hypoenhancing with a hypodense multicystic component. 4)  Biliary obstruction: yes, there is increased intrahepatic biliary dilation in the left hepatic lobe compared to prior without a clear obstructing mass identified. 5)  Pancreatic duct obstruction: no Vascular-arterial:        1) Celiac axis is not involved        2)  SMA is not involved.        3)  CHA is not involved.        4)  Arterial variant: None.        5) other: The splenic artery is encased by tumor, unchanged. Vascular-venous: 1)  MPV is not involved. 2)  SMV is not involved. 3)  Thrombus in vein:  absent 4)  Venous collateral: None. 5)  Other: No change in chronic occlusion of the splenic vein. Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Liver:  Increased, now moderate to severe intrahepatic biliary ductal dilatation in the left hepatic lobe. Interval slightly decreased size of hypodense 0.7 x 0.7 cm lesion in segment 6 (series 7 image 49), previously 1.1 x 0.8 cm. Interval slightly decreased size of ill-defined nodule along the inferior hepatic margin, now measuring 1.3 x 0.9 cm compared to 1.5 x 1.2 cm previously with new central hypodensity (series 7 image 56). Gallbladder: Unremarkable Spleen:  No significant change in hypodense splenic lesions. Adrenal Glands:  Unremarkable Kidneys:  Hypodense renal mass(es) too small to characterize, either benign or clinically insignificant, unchanged. GI Tract:  Small paraesophageal hernia. Vasculature:  Unremarkable Lymphadenopathy: Absent Peritoneum: Unchanged 1.4 x 1.2 cm nodule in the right paracolic gutter (series 7 image 72). Slightly increased size of left paracolic gutter nodule measuring 1.3 x 1.0 cm, previously 1.1 x 0.9 cm (series 7 image 64). \n Similar anterior left upper quadrant nodule measuring 1.1 x 0.9 cm, previously 1.1 x 1.0 cm (series 7 image 43). Previously described peritoneal implant in the lesser curvature the stomach is less conspicuous, but likely unchanged in size. Bladder: Unremarkable Reproductive organs: Unremarkable Bones:  No suspicious lesions Extraperitoneal soft tissues: Unremarkable Lines/drains/medical devices: None     1.  Compared to 12/18/2018, no significant change in multicystic distal pancreas mass with slightly decreased size of hepatic lesions, no significant change in splenic lesions, and stable to slightly increased peritoneal implants. 2.  Increase in now moderate intrahepatic biliary dilation in the left hepatic lobe without a clear obstructing lesion. 3.  Unchanged chronic occlusion of the splenic vein and tumor encasement of the splenic artery.     Result Date: 02/15/2019  CT CHEST WITH CONTRAST   FINDINGS: LUNGS: Unchanged left lower lobe nodule measuring 3 mm (series 2 ***** 123); on the most remote prior in 2018 this nodule measured 6 mm. New patchy peribronchial vascular groundglass opacities in the left lower lobe (series 2 ***** 167). Unchanged right middle lobe scarring and calcified granulomata. PLEURA: No pleural effusion. MEDIASTINUM: No lymphadenopathy. HEART/GREAT VESSELS: Unchanged mild cardiomegaly and postsurgical changes status post CABG BONES/SOFT TISSUES: No suspicious osseous lesions.     1. Unchanged left lower lobe nodule, decrease in size compared to most remote prior, possibly inflammatory versus treated metastasis. 2. New patchy groundglass left lower lobe mild opacities, likely infectious.       Result Date: 03/11/2019 (outside)  CT ABDOMEN/PELVIS WITH CONTRAST  FINDINGS:  Inferior Chest: The heart is enlarged in size. No pericardial  effusion. There is a small left pleural effusion. Patchy ground-glass  and airspace opacities in the inferior right middle lobe and left lung  base, may \n represent atelectasis versus infection. The lung bases are  clear.    Liver: Liver is normal in size and smooth in contour. Circumscribed  subcentimeter hyperdensities in the caudate lobe and inferior right  hepatic lobe, too small to accurately characterize. There is an ill  defined contour deforming hypodense lesion in the inferior right  hepatic lobe measuring approximately 1.4 cm in diameter (axial image  55). Portal and hepatic veins are patent.    Gallbladder and biliary tree: The gallbladder appears unremarkable.  There is prominent intrahepatic biliary ductal dilatation predominantly  involving the left hepatic lobe. The common bile duct appears normal in  caliber.    Pancreas/spleen: There is an ill defined mixed solid and cystic mass  involving the tail of the pancreas associated with few coarse  calcifications, measuring approximately 7.2 cm AP by 5.4 cm transverse  by 5.6 cm craniocaudal. This mass abuts the splenic hilum with  multiple cystic spaces within the spleen and loss of fat plane,  suggestive of direct invasion. Large wedge-shaped areas of  hypoenhancement within the peripheral aspects of spleen suggestive of  multiple splenic infarcts. There is loss of fat plane between the  pancreatic mass in the posterior wall of the stomach suggestive of  possible direct invasion.    Adrenals: The right adrenal gland appears unremarkable. There is mild  diffuse thickening of the left adrenal gland without definite focal  nodularity. Subtle loss of fat plane between the left adrenal gland  and pancreatic tail mass (axial image 55).    Kidneys: Bilateral subcentimeter hypodensities, too small to  accurately characterize but statistically likely to represent benign  cysts.The left kidney appears slightly inferior displaced. There is a  loss of fat plane between the anterior aspect of the left kidney and a  pancreatic tail mass (axial image 64). No nephrolithiasis or  hydronephrosis.    GI tract: There \n is mucosal hyperenhancement involving the distal aspect  of the stomach adjacent to likely direct invasion from pancreatic mass.  There is associated wall thickening/ edema. Proximal stomach is  distended with fluid and food material. Findings concerning for at  least partial gastric outlet obstruction. There is mild wall edema of  the visualized distal esophagus. Small bowel loops are grossly normal  in course in caliber with no significant wall thickening or dilatation.  There is wall thickening involving the transverse colon, and descending  colon, and sigmoid colon. In the region of the splenic flexure, there  is loss of fat plane between the ill defined pancreatic tail mass and  the colon.    Bladder: Urinary bladder appears unremarkable.    Reproductive organs: The prostate gland is mildly prominent in size  measuring 4.6 x 4.2 x 4.5 cm.    Peritoneal Cavity: There is a small amount of free fluid noted in the  abdomen in the perihepatic space, perisplenic space, bilateral  paracolic gutters, and pelvis. No free air is identified. No  retroperitoneal or mesenteric lymphadenopathy.    Vasculature: Scattered atherosclerotic calcification of the abdominal  aorta and its main branches with no hemodynamically significant  stenosis or occlusion.    Bones and Soft tissues: No acute osseous abnormality. Multilevel  degenerative changes of the thoracolumbar spine and bilateral  sacroiliac joints. Very small fat containing left inguinal hernia.  There is mild anasarca of the subcutaneous tissues. Few foci of air in  the right anterior abdominal wall, likely representing recent injection.    IMPRESSION:  1. Ill defined mixed solid and cystic pancreatic tail mass with few  coarse calcifications, measuring at least 7.2 x 5.4 x 5.6 cm. There is  direct invasion of the spleen and posterior stomach. There is also  questionable direct invasion of the splenic flexure of colon common the  left adrenal gland, and the \n left kidney.  2. Proximal gastric distension. Diffuse gastric wall edema with  mucosal hyperenhancement of the distal stomach, compatible with direct  invasion from adjacent pancreatic tail mass causing at least partial  gastric outlet obstruction.  3. Multiple peripheral wedge-shaped areas of hypoenhancement throughout  the spleen compatible with multiple splenic infarcts.  4. Diffuse wall edema of the transverse, descending, and sigmoid colon.  Possible direct invasion from pancreatic tail mass at the level of the  splenic flexure. Other possibilities include colitis of infectious or  inflammatory etiologies.  5. Small left pleural effusion. Patchy ground-glass and airspace  opacities in the inferior right middle lobe and left lung base,  possibly representing atelectasis versus infiltrate/infection.  6. Ill defined contour deforming hypodense lesion within the inferior  right hepatic lobe measuring 1.4 cm, concerning for metastatic disease  in this clinical setting.  7. Small volume diffuse abdominal ascites.      Impression and Recommendations:  In summary, Mr. ***** is a 74 y.o. male with metastatic PDAC, currently on 3L treatment consisting of the combination of 5-FU/LV plus nanoliposomal irinotecan (nal-IRI, or Onivyde), but with some evidence concerning of continued clinical/symptomatic dec
```

---

## Letter A

Dear Mr. *****,

Thank you for coming to our clinic for your follow-up visit. Here is a summary of your current situation and what we discussed:

**Diagnosis and Stage:**
You have been diagnosed with metastatic pancreatic cancer. This means the cancer has spread beyond the pancreas to other parts of your body.

**Treatment Plan:**
You have been receiving chemotherapy to help control the cancer. Recently, you started a new treatment called 5-FU/LV plus nanoliposomal irinotecan (Onivyde). This treatment has helped keep the cancer stable, but you are experiencing some symptoms like pain and low energy.

**Key Test Results:**
Recent tests show that the cancer is still growing in some places, but it has not changed much overall. Your blood tests show that you are anemic, which means you don’t have enough red blood cells. We gave you a transfusion to help with this.

**Next Steps:**
We will continue with one more cycle of the current chemotherapy if your blood tests are okay. We will do another scan soon to check how the cancer is doing.

If the cancer continues to grow, we might consider other treatments like joining a clinical trial or using radiation therapy to help with pain.

**What to Watch For:**
Please watch for any new or worsening symptoms such as pain, tiredness, or trouble eating. Let us know if you notice anything different.

**Important Reminders:**
Remember to take your medications as prescribed. If you have any questions or concerns, please talk to your care team. We are here to support you.

Sincerely,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

Please feel free to reach out to us anytime. We are here to help you through this journey.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Your cancer is showing stable disease based on imaging and tumor marker evidence. The CT scan shows no significant change in the pancreatic mass, slightly decreased size of liver lesions, and stable to slightly increased cancer growths in the abdomen lining. Your CA 19-9 levels have decreased, indicating a positive trend. However, there are some suspicious findings that need further evaluation.
**What treatment or medication changes were made?**
You will continue with 1 additional cycle of the same chemotherapy (5-FU/LV plus nanoliposomal irinotecan (Onivyde)), contingent upon acceptable laboratory parameters. Supportive medications for pain and nausea have been prescribed.
**What is the plan going forward?**
Short-term follow-up CT scans are planned to assess for any further disease progression. Radiation therapy to your pancreatic primary may be considered for palliative purposes. Your next visit will be in-person to review the results of the follow-up scans.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 4

**Cancer Type:** Adenocarcinoma of the pancreas

## Original Clinical Note

```
SUBJECTIVE:    This is a very pleasant 64-year-old woman with a diagnosis of adenocarcinoma of the pancreas, extent of disease uncertain. She presented in June 2020 and was thought to have locally advanced disease. She initiated treatment with FOLFIRINOX and after 4 cycles of therapy, her CT scan showed a new liver lesion. This was confirmed with MRI as suspicious for metastasis. If subsequent PET/CT scan did not show that the liver lesions were PET avid.    At that point, in October we initiated gemcitabine and Abraxane. She has now completed six 28 day cycles and is on surveillance. it should be noted that she did have an opinion at ***** *****. Their feeling was that while some of her liver lesions were hemangiomas there was a lesion in segment 7 that was consistent with a liver metastasis.    Earlier this week, she had an EGD and ERCP following complaints of persistent nausea. She was found to have tumor ingrowth and her stent along with a biliary stone and blood clots. These were removed and a covered metal stent was deployed within the existing stent.    She does not express CA-19-9.    She feels much better after the stent revision. She has a good energy level. She can walk up to 4 miles a day. She does complain of brain fog which she attributes to her chemotherapy. The numbness and tingling in her fingers and hands have resolved. She still has some numbness and tingling in her feet. It does not affect her gait. Her appetite is improving following the ERCP. She has a bowel movement every 3 days or so with the help of fiber and MiraLAX. Her hair is growing back.      MED:  Medications the patient states to be taking prior to today's encounter.   Medication Sig   \u0007 buPROPion (WELLBUTRIN SR) 100 mg 12 hr tablet Take 100 mg by mouth daily   \u0007 estradiol twice weekly patch 0.05 mg/24 hr estradiol 0.05 mg/24 hr semiweekly transdermal patch   \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 mg by mouth every 8 (eight) hours \n as needed for Anxiety (nausea and vomiting)   \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet three times a day as needed for nausea and vomiting. Start 2 days after last dose of palonosetron (Aloxi) is given.   \u0007 pantoprazole (PROTONIX) 40 mg EC tablet Take 1 tablet (40 mg total) by mouth daily Take 30 minutes before breakfast   \u0007 PRISTIQ 100 mg 24 hr tablet Take 100 mg by mouth daily      \u0007 progesterone (PROMETRIUM) 100 mg capsule TAKE 1 CAPSULE BY MOUTH DAILY AT BEDTIME         OBJECTIVE:    Vital signs were not obtained for this visit.  She appears comfortable.    Imaging:    CT Abdomen /Pelvis with Contrast    Result Date: 06/18/2021  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** *****:***** PM CLINICAL HISTORY: restaging scan. Pancreatic cancer unresectable. Pt currently on chemo holiday. COMPARISON:  CT abdomen pelvis dated 04/15/2021 TECHNIQUE: CT of the abdomen and pelvis was performed. MEDICATIONS: Omnipaque - 500 mL - Oral Omnipaque - 500 mL - Oral Iohexol 350 - 150 mL - Intravenous FINDINGS: Visualized lung bases:  Please to same-day CT chest for pulmonary findings. Liver:  Similar pneumobilia when compared to 04/15/2021. Known hemangiomas in segment 3 and 7. No suspicious lesions. Gallbladder: Cholelithiasis. Metallic stent in the CBD with debris within the lumen similar to recent study. Spleen:  Similar splenomegaly. Pancreas:  Similar ill-defined pancreatic head mass which is difficult to accurately measure and appears relatively similar in size when compared to recent MRI measuring 4.2 x 3.6 cm. The cystic component is also unchanged. Similar ductal dilatation of the main pancreatic duct. Adrenal Glands:  Unremarkable Kidneys:  Unremarkable GI Tract:  Unremarkable Vasculature:  Similar fat stranding surrounding the SMA and encasement of the common hepatic artery as well as severe narrowing of the portal vein/SMV confluence. Lymphadenopathy: Similar porta hepatis lymphadenopathy measuring up to 1.3 \n cm. Peritoneum: No ascites Bladder: Unremarkable Reproductive organs: Unremarkable Bones:  Unchanged postsurgical changes of the lumbar spine. Extraperitoneal soft tissues: Unremarkable Lines/drains/medical devices: None RADIATION DOSE INDICATORS: Exposure Events: 3 , CTDIvol Min: 16.9 mGy, CTDIvol Max: 16.9 mGy, DLP: 1100.1 mGy.cm. The following accession numbers are related to this dose report *****,*****     Compared to recent MRI dated 04/23/2021, no significant change in size of ill-defined pancreatic head/neck mass. No evidence of metastasis. Report dictated by: ***** *****, DO, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging    CT Chest with Contrast    Result Date: 06/18/2021  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Pancreatic cancer, primary restaging scan. Pancreatic cancer unresectable. Pt currently on chemo holiday. Additional history: Mucinous pancreatic adenocarcinoma. COMPARISON: Thoracic CT from 04/15/2021 and more remote priors, concurrent abdominal CT TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the administration of intravenous contrast. RADIATION DOSE INDICATORS: Exposure Events: 3 , CTDIvol Min: 16.9 mGy, CTDIvol Max: 16.9 mGy, DLP: 1100.1 mGy.cm. The following accession numbers are related to this dose report *****,***** FINDINGS: Support device(s): Right chest port catheter terminates at the superior cavoatrial junction. LUNGS: No new suspicious nodule. A few tiny groundglass nodules in the upper lobes are unchanged. For reference, there is a 3 mm right apical groundglass nodule is unchanged (series 4, image 64). PLEURA: No pleural effusion. MEDIASTINUM: No adenopathy. HEART/GREAT VESSELS: The heart and major thoracic vascular structures are normal in size. No pericardial effusion. BONES/SOFT TISSUES: No suspicious lesion. VISIBLE ABDOMEN: Please see dedicated abdomen and pelvis CT report. \n     Unchanged tiny groundglass nodules in the apices, most likely inflammation or scar. However, attention on follow-up is advised. Report dictated by: ***** *****, MD, signed by: ***** *****, MD Department of Radiology and Biomedical Imaging    XR ERCP Cholangiogram Pancreatography    Result Date: 06/25/2021  FL ERCP CHOLANGIOGRAM PANCREATOGRAPHY:   *****/*****/***** *****:***** PM CLINICAL HISTORY: pancreatic cancer COMPARISON: CT abdomen/pelvis 06/18/2021 TECHNIQUE: Image(s) of an ERCP were submitted for evaluation.     Endoscope advanced to the duodenum with subsequent wire cannulization of the metallic common bile duct stent. Wire was passed beyond the stent into the intrahepatic bile ducts with subsequent cholangiogram demonstrating irregular moderate intrahepatic biliary ductal dilation with filling defects related to sludge or stones. Irregular narrowing of the common bile duct in the region of the stent with subsequent balloon sweeping related to known history of pancreatic cancer. Sweeping of the right intrahepatic bile duct was also performed. A likely second metal stent was deployed within the existing metallic common bile duct stent. Please refer to a separate report by the physician who performed the procedure for further details. Lines/drains/medical devices: Metallic common bile duct stents Report dictated by: ***** *****-***** *****, MD, signed by: ***** *****-***** *****, MD Department of Radiology and Biomedical Imaging    ASSESSMENT:    1. Metastatic adenocarcinoma of the pancreas with continued good disease control on surveillance  2. ECOG 1    PLAN:    She'll continue on surveillance and we will see her back in 2 months with a follow-up CT scan. She had many questions about future investigational therapies. I explained that while there are many things in progress, the menu varies constantly and we should talk specifics when the need arises.    I performed this evaluation using real-time telehealth tools, including a live \n video Zoom connection between my location and the patient's location. Prior to initiating, the patient consented to perform this evaluation using telehealth tools.     I spent a total of 40 minutes on this patient's care on the day of their visit excluding time spent related to any billed procedures. This time includes time spent with the patient as well as time spent documenting in the medical record, reviewing patient's records and tests, obtaining history, placing orders, communicating with other healthcare professionals, counseling the patient, family, or caregiver, and/or care coordination for the diagnoses above.      \n
```

---

## Letter A

Dear [Patient’s Name],

I hope this letter finds you well. I wanted to share some important information about your health in a way that is easy to understand.

**Diagnosis and Stage:**
You have a type of cancer called adenocarcinoma of the pancreas. This means the cancer started in the pancreas. We are not sure how far the cancer has spread yet.

**Treatment Plan:**
You have been receiving treatments to fight the cancer. Recently, you had a procedure where a special tube (stent) was placed to help with some blockages in your body. This helped improve your symptoms like nausea.

**Key Test Results:**
We did some scans to check how your body is doing. The latest scans show that the cancer is under control, but we need to keep watching closely.

**Next Steps:**
You will continue to be watched carefully. We will see you again in two months for another scan to make sure everything is okay.

**What to Watch For:**
If you notice any new symptoms or if something feels different, please let us know right away. Keep an eye out for any changes in how you feel.

**Medications:**
You are taking several medications to help manage your symptoms and side effects. Make sure to take them as directed by your doctor.

**How You Are Doing:**
It’s great to hear that you are feeling much better after the stent procedure. Your energy levels are good, and you can walk up to 4 miles a day. Your appetite is improving, and your hair is growing back. Some of the side effects from the chemotherapy, like numbness in your hands and feet, have gotten better.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Your Name]  
Medical Communication Specialist  
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer.
**What's new or changed since your last visit?**
The cancer is currently showing stable disease based on imaging findings.
CT scans show that the tumor in your pancreas has stayed the same size.
There is no new evidence of the cancer spreading to other parts of your body.
**What treatment or medication changes were made?**
You will continue to take ondansetron (Zofran) to manage nausea.
**What is the plan going forward?**
You will continue on surveillance and we will see you back in 2 months with a follow-up CT scan.
CT scan in 2 months for follow-up.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 5

**Cancer Type:** Adenocarcinoma of the pancreas

## Original Clinical Note

```
This is an independent service.  The available consultant for this service is ***** ***** *****, MD.       PATIENT IDENTIFICATION: Mr. ***** is a 53 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer Center for follow-up of his diagnosis of adenocarcinoma of the pancreas with biopsy-proven oligo metastatic disease in the abdominal wall.    ONCOLOGIC HISTORY:   He first presented with abdominal pain and weight loss. On 01/10/2019, he had a CT of the abdomen which showed a mass in the pancreas which encased the celiac axis along with a concerning soft tissue nodule in the abdominal wall at midline.      02/06/19: CT PET showed a hypermetabolically active large pancreatic tail mass abutting adjacent structures, although the lesion in the abdominal wall was slightly PET avid (SUV 1.2).     01/22/19: EUS directed FNA of the pancreatic lesion revealed a diagnosis of adenocarcinoma, MSS.    02/14/19: Biopsy of abdominal wall nodule showed metastatic adenocarcinoma.    03/08/19: Patient presented for C1D1 FOLFIRINOX and labs revealed hyperbilirubinemia (t bili 1.7). Per patient report, he has h/o hyperbilirubinemia and diagnosed ~2 years ago with possible Gilbert syndrome. Irinotecan was held and he proceeded with C1D1 FOLFOX.    03/19/19: C2D1 FOLFIRINOX. T bili was 0.8. Dr. ***** recommended that patient proceed with Irinotecan and closely follow patient for possible irinotecan toxicities.   04/02/19: C3D1 FOLFIRINOX    04/30/19: C4D1 FOLFIRINOX    05/15/19: CT CAP showed stable pancreatic mass and slight decreased size of anterior abdominal wall lesion  05/16/19: C5D1 FOLFIRINOX    05/18/19: US with doppler LE revealed segmental occlusive thrombi within the superior and mid duplicated left peroneal veins. Patient started on xarelto.   05/28/19: C6D1 FOLFIRINOX. CA 19-9 = 2,528    06/11/19: C7D1 FOLFIRINOX    06/25/19: C8D1 FOLFIRINOX. CA 19-9 = 995    07/06/19: CT CAP shows continued \n response to treatment   07/09/19: C9D1 FOLFIRINOX    07/11/19: CA 19-9 = 588  07/25/19: Treatment held (Cycle 10) d/t thrombocytopenia (PLT 95)  08/01/19: C10D1 FOLFIRINOX. CA 19-9 = 826  08/13/19: C11D1 FOLFIRINOX    09/03/19: C12D1 FOLFIRINOX  (Last cycle)  09/10/19: CT CAP stable disease in the pancreas and abdominal wall lesion. Patient was recommended to start chemotherapy break.     10/03/19: Per patient's request, his case was presented at Pancreas Tumor Board to see if he is a candidate for surgical resection. His pancreas mass is not resectable for multiple reasons, primarily because of extensive vessel involvement and encasement. Dr. ***** ***** recommended consolidation radiation due to the fact that his pancreas tumor is abutting the stomach and concerns over the complications that this will likely lead to. However, Dr. ***** does not recommend radiation therapy at this time since patient is asymptomatic.     INTERVAL HISTORY:   ***** has been feeling generally well. Continues to have some oral thrush. Started fluconazole today. Was also seen by dermatologist recently and started on cicloprox cream for tinea cruris in groin area.     REVIEW OF SYSTEMS:   14 point ROS completed and all other systems were reviewed and are negative.    Current Outpatient Medications   Medication Sig Dispense Refill   \u0007 butenafine (LOTRIMIN ULTRA) 1 % cream      \u0007 cholecalciferol, vitamin D3, 1000 UNITS TAB tablet Take 4 tablets by mouth daily        \u0007 clotrimazole (MYCELEX) 10 mg troche 1 tablet (10 mg total) by Other route 3 (three) times daily Take by buccal route. Take for 14 days and during chemotherapy cycles after initial treatment 90 Troche 1   \u0007 fluconazole (DIFLUCAN) 100 mg tablet Take 200mg PO qd on day one followed by 100mg PO qd on subsequent days.  Take for 1 week and then stop. 8 tablet 0   \u0007 hyoscyamine (LEVSIN/SL) 0.125 MG SL tablet Place 1 tablet (0.125 mg total) under the tongue every 4 (four) hours as needed \n (abdominal cramping). (Patient not taking: Reported on 09/17/2019  ) 60 tablet 3   \u0007 KERYDIN 5 % SolA      \u0007 L.acidoph/B.long/L.plant/B.lac (PROBIOTIC ACIDOPHILUS BEADS ORAL) Take by mouth.     \u0007 lidocaine-prilocaine (EMLA) 2.5-2.5 % cream Apply cream one hour before port access as needed for pain. 30 g 2   \u0007 lipase-protease-amylase (CREON) 24,000-76,000 -120,000 unit capsule Take 2 tabs with meals and 1 tab with snacks. 300 capsule 5   \u0007 loperamide (IMODIUM) 2 mg capsule Take two capsules by mouth at the onset of diarrhea, then take one capsule every 2 hours until diarrhea-free for 12 hours. 30 capsule 5   \u0007 medical cannabis      \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet three times a day as needed for nausea and vomiting. Start 2 days after last dose of palonosetron (Aloxi) is given. (Patient not taking: Reported on 07/25/2019  ) 30 tablet 5   \u0007 oxyCODONE (OXYCONTIN) 10 mg 12 hr ER tablet Take 1 tablet (10 mg total) by mouth every 12 (twelve) hours (Patient not taking: Reported on 09/17/2019  ) 60 tablet 0   \u0007 oxyCODONE (ROXICODONE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 4 (four) hours as needed for Pain (for moderate to severe pain). (Patient not taking: Reported on 09/17/2019  ) 90 tablet 0   \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting). (Patient not taking: Reported on 09/17/2019  ) 30 tablet 5   \u0007 rivaroxaban (XARELTO) 20 mg tablet Take 1 tablet (20 mg total) by mouth daily 30 tablet 5   \u0007 terbinafine HCl (LAMISIL) 1 % cream Apply topically Daily. Use as instructed     \u0007 TURMERIC ORAL Take by mouth     \u0007 UNABLE TO FIND Take 1 tablet by mouth Host Defense brand \"Stamets 7\" Daily Immune Support, one tablet every other day during weeks of no infusion.   Each tablet contains 71.5 mg of royal sun blazei, cordyceps, reishi, maitake, lion's mane, chaga, and mesima.       \u0007 UNABLE TO FIND Take by mouth Host \n Defense brand Turkey Tail extract, ~ 0.5 ml every other day during weeks of no infusion       No current facility-administered medications for this visit.        ALLERGIES:  Allergies/Contraindications  No Known Allergies    Past medical, social, and family history were reviewed at today's visit and are unchanged from prior visit.      PHYSICAL EXAM:  Objective   Vital Signs:   There were no vitals filed for this visit.  ECOG Performance Status: 1 - Symptomatic but completely ambulatory  Constitutional:  WDWN, NAD, pleasant and conversant   Eyes: sclera anicteric  Respiratory: no respiratory distress  Neurologic: Alert & oriented x 3  Psychiatric: Mood stable. No HI, SI, hallucinations, paranoia, or delusions.    DIAGNOSTIC STUDIES:   Laboratory data:   Lab Results   Component Value Date    WBC Count 6.4 09/03/2019    Neutrophil Absolute Count 4.18 09/03/2019    Hemoglobin 11.4 (L) 09/03/2019    Hematocrit 35.4 (L) 09/03/2019    Platelet Count 126 (L) 09/03/2019    Creatinine 0.74 09/03/2019    Bilirubin, Total 0.4 09/03/2019    Alkaline Phosphatase 160 (H) 09/03/2019    Aspartate transaminase 31 09/03/2019    Alanine transaminase 30 09/03/2019    Albumin, Serum / Plasma 3.7 04/26/2019    Int'l Normaliz Ratio 1.1 03/05/2019    Sodium, Serum / Plasma 134 (L) 04/26/2019    Potassium, Serum / Plasma 3.9 04/26/2019    Calcium, total, Serum / Plasma 9.1 04/26/2019     Lab Results   Component Value Date    Cancer Antigen 19-9 576 (H) 08/27/2019    Cancer Antigen 19-9 826 (H) 08/01/2019    Cancer Antigen 19-9 588 (H) 07/11/2019    Carcinoembryonic Antigen 16.2 (H) 08/27/2019    Carcinoembryonic Antigen 11.7 (H) 08/01/2019    Carcinoembryonic Antigen 17.0 (H) 07/11/2019     Foundation Medicine liquid biopsy results:   MSI Status Undetermined  RB1 P26fs*47      ASSESSMENT AND PLAN:   In summary, Mr. ***** is a 53 y.o. male who returns to our GI Oncology practice at the UCSF Comprehensive Cancer Center for follow-up of his \n diagnosis of adenocarcinoma of the pancreas with biopsy-proven oligo metastatic disease in the abdominal wall. He is now s/p 12 cycles of FOLFIRINOX.    #Stage IV pancreatic adenocarcinoma  Patient has completed 12 cycles of FOLFIRINOX from February 2019 - August 2019, and is currently on chemotherapy break. He has restaging CT planned for 11/19/19.    During today's visit I reviewed with ***** the Pancreas Tumor Board's recommendations. I explained why surgical resection is not recommended. Also discussed the role of palliative radiation therapy to his pancreas mass and why Dr. ***** has not recommended radiation at this time. I also reviewed signs and symptoms of a GI bleed and other complications that could arise if his pancreas tumor where to invade into the stomach.     Foundation Medicine liquid biopsy results:   MSI Status Undetermined  RB1 P26fs*47    #Thrush: Ongoing  - SMS recommended fluconazole.   - Will also have patient complete labs to rule out diabetes (he has h/o prediabetes): blood glucose and Hemboglobin A1C    #occlusive thrombi within the superior and mid duplicated left peroneal veins  -  Continue xarelto QD    #Abdominal pain/cramping: Improved  - Followed by SMS, Dr. ***** *****   - Continue Creon to 2 or 3 tab/meal and 1 tab/snack   - Cont Oxycontin 10mg PO bid  - Oxycodone 5 to 10mg PO q4h prn BTP (can cut 10mg tabs in half)    #Weight loss: Improved  - Followed by Cancer Center Nutritionist ***** *****, *****    #Supportive care   - followed by  SMS, Dr. *****    Advance Care Planning   Discussed with: ***** *****, NP     Surrogate decision maker: Identified and documented: ***** ***** (1st); ***** ***** (2nd)    Life sustaining treatment preferences (i.e. Code): DNR and Unknown    Additional preferences of care (eg: Goals; Fears and worries; Sources of strength; Critical abilities; Trade-offs; Family): Relevant values/priorities: Mental capacity, free of pain    Changes to treatment \n plan as a result of this conversation: Preferences for life sustaining treatment clarified and POLST completed     RTC: 11/21/19 to see Dr. ***** or sooner if indicated    As always, the patient was counseled to contact the clinic or the after-hours physician on call for any new symptoms or complications in the interim between now and her follow up visit with me. Thank you for involving me in the care of this very kind patient.  Please do not hesitate to contact me with any questions or concerns.    I spent a total of 25 minutes face-to-face with the patient and >50% of that time was spent counseling regarding the symptoms, treatment plan, risks and/or therapeutic options for the diagnoses above.    I performed this consultation using real-time Telehealth tools, including a live video connection between my location and the patient's location. Prior to initiating the consultation, I obtained informed verbal consent to perform this consultation using Telehealth tools and answered all the questions about the Telehealth interaction.    ***** *****, NP    \n
```

---

## Letter A

Dear Mr. *****,

Thank you for visiting us today. Here is a summary of your recent health check-up and treatment plan:

**Diagnosis:** You have a type of cancer called adenocarcinoma of the pancreas. This means there is a tumor in your pancreas. It has spread a little bit to your abdominal wall, but not much.

**Treatment Plan:** You have finished 12 rounds of a medicine called FOLFIRINOX. Now, you are taking a break from this treatment. We will do another scan on November 19, 2019, to see how things are going.

**Key Test Results:**
- Your blood tests show that your body is handling the treatment well.
- The levels of a protein called CA 19-9 have gone down, which is good news. This protein is often higher when there is cancer.

**Next Steps:**
- You will have another scan on November 19, 2019.
- We will keep watching for any new symptoms or problems.

**What to Watch For:**
- If you have any new pain, bleeding, or other unusual symptoms, please let us know right away.
- Keep taking your medicines as prescribed.

**Other Health Issues:**
- You have a condition called thrush, which is a yeast infection in your mouth. You are starting a new medicine called fluconazole to treat it.
- You have a blood clot in your leg. You are taking a medicine called Xarelto to help prevent more clots.
- You have some pain in your abdomen, but it has improved. You are taking a medicine called Creon to help with digestion.

**Important Reminders:**
- If you have any questions or concerns, please talk to your care team.
- Remember to come back for your next appointment on November 21, 2019.

We hope you feel better soon and we look forward to seeing you again soon.

Warm regards,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

Please remember to discuss any questions or concerns with your care team. They are here to help you.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Your cancer is currently stable in the pancreas and abdominal wall based on recent imaging. The CA 19-9 levels, a tumor marker, have remained high, indicating stable disease.
**What treatment or medication changes were made?**
You will continue taking xarelto daily for blood clots. You will also continue taking Creon to help with digestion. Your pain medications, Oxycontin and Oxycodone, will remain the same. Fluconazole was added to treat thrush.
**What is the plan going forward?**
You will have a restaging CT scan on November 19th to monitor your cancer. You will also have blood tests to check for diabetes. Your next visit is scheduled for November 21st. Palliative radiation therapy was discussed but is not currently recommended.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 6

**Cancer Type:** Adenocarcinoma of the pancreas

## Original Clinical Note

```
Subjective:    This is a very pleasant 63-year-old African American woman from ***** ***** with the aforementioned diagnosis.  She presented in February with abdominal discomfort, bloating and eventually jaundice.  A metal stent was placed on May 29 and since then, she has been relatively asymptomatic. CT scan at that time revealed a mass in the head of the pancreas and a 2 cm adrenal mass.  A followup PET/CT in June suggested that only the mass in the pancreas was PET avid.  An ultrasound of the pelvis done on June 05 shows a 3.2 cm fibroid in the fundus of the uterus and thickened endometrium.  No adnexal masses were seen. EUS done on May 29 suggested encasement of the hepatic artery and invasion into the portal vein.  FNA of the pancreas revealed adenocarcinoma    When we initially saw her we repeated her pancreas protocol CT scan. We were concerned about the possibility of peritoneal carcinomatosis.  She underwent a laparoscopy which was unremarkable.  There was a small amount of ascites in the pelvic region but washings were negative.  A small lesion in the liver was biopsied and this recovered elements of a cyst.    After her laparoscopy, she developed pleuritic chest pain and some shortness of breath.  She was evaluated for a PE in the emergency room and CT scan was unremarkable although she was noted to have some interlobular septal thickening and a ground glass opacities.    We then started her on treatment with modified FOLFIRINOX. She completed 12 cycles. At that point, she is really evaluated for surgical resection. On March 22, she underwent a pylorus-preserving pancreatectomy with open cholecystectomy and portal vein and hepatic artery resection and interposition graft. Surgical pathology revealed a 1.7 cm residual tumor mass within estimation of 51-90% of tumor cells destroyed. Margins were negative. There were no positive lymph nodes.    Postoperatively, she had no unexpected complications. She \n feels that she is continuing to recover from her surgery. However, she notes continued abdominal discomfort particularly on the right side. She does have a good appetite. She's having normal bowel movements. She is lost 15 pounds since her surgery.    Objective:    Vitals: BP 148/79 | Pulse 84 | Temp(Src) 36.7 C (98.1 F) (Oral) | Resp 16 | Ht 173 cm (5' 8.11\") | Wt 79.561 kg (175 lb 6.4 oz) | BMI 26.58 kg/m2 | SpO2 100% | LMP 10/08/1998   In general, she appears comfortable.  HEENT no icterus  Nodes no clinical adenopathy  Neck without thyromegaly  Lungs clear to A and P  Heart RR without murmur or gallop  Abd somewhat distended. Right side is tender to palpation. No fluid wave.  GU not done  Skin without rash or petechiae  Extrem without edema and pedal pulses are full  Neuro without gross focal deficit    Results for orders placed in visit on 04/26/15   COMPLETE BLOOD COUNT WITH 5-PART DIFFERENTIAL       Result Value Ref Range    WBC Count 5.7  3.4 - 10 x10E9/L    RBC Count 3.57 (*****) 4.0 - 5.2 x10E12/L    Hemoglobin 9.5 (*****) 12.0 - 15.5 g/dL    Hematocrit 30.5 (*****) 36 - 46 %    MCV 85  80 - 100 fL    MCH 26.6  26 - 34 pg    MCHC 31.1  31 - 36 g/dL    Platelet Count 171  140 - 450 x10E9/L    Neutrophil Absolute Count 3.19  1.8 - 6.8 x10E9/L    Lymphocyte Abs Cnt 1.60  1.0 - 3.4 x10E9/L    Monocyte Abs Count 0.44  0.2 - 0.8 x10E9/L    Eosinophil Abs Ct 0.48 (*****) 0.0 - 0.4 x10E9/L    Basophil Abs Count 0.03  0.0 - 0.1 x10E9/L   CANCER ANTIGEN 19-9       Result Value Ref Range    Cancer Antigen 19-9 2250 (*****) <36 U/mL   BILIRUBIN, TOTAL       Result Value Ref Range    Bilirubin, Total 0.4  0.2 - 1.3 mg/dL   CREATININE, SERUM / PLASMA       Result Value Ref Range    Creatinine 0.86  0.52 - 1.06 mg/dL    eGFR if non-African American 72  >60 mL/min    eGFR if African Amer 83  >60 mL/min   ASPARTATE TRANSAMINASE       Result Value Ref Range    Aspartate transaminase 23  17 - 42 U/L   ALANINE TRANSAMINASE       Result Value \n Ref Range    Alanine transaminase 17  11 - 50 U/L   ALKALINE PHOSPHATASE       Result Value Ref Range    Alkaline Phosphatase 89  31 - 95 U/L     CA 19-9 dated April 26 was 2250. Preoperative value in January was 44.  The following imaging studies are reviewed and interpreted:    Ct Abdomen /pelvis With Contrast    04/26/2015   CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** 8:37 AM  CLINICAL HISTORY: Pt s/p resection for panc ca; r/o recurrence  COMPARISON:  Comparison CT from 01/12/2015.  TECHNIQUE: Following the administration of 150 cc of Omnipaque 350, contiguous 1.25-mm collimation axial images were obtained through the abdomen and pelvis.  Coronal and sagittal reformats were also obtained.  RADIATION DOSE INDICATORS:  ***** *****-***** received 1 exposure event(s) (excluding scout) during this CT study. The CTDIvol (mGy) and DLP (mGy-cm) radiation dose values for each event are:  Event: 1;  anatomic area: Chest/Abdomen/Pelvis;  phantom: body;  CTDIvol: 11.1;  DLP: 699.1  The following accession numbers are related to this dose report [*****]: *****  The dose indicators for CT are the volume Computed Tomography Dose Index (CTDIvol) and the Dose Length Product (DLP), and are measured in units of mGy and mGy-cm, respectively. These indicators are not patient dose, but values generated from the CT scanner acquisition factors.  Individual patient doses calculated from these values will depend on patient size and can substantially underestimate or overestimate the actual dose. For additional information on radiation dose, safety, and protection in the UCSF Department of Radiology and Biomedical Imaging, please refer to one of our FAQ leaflets on \"Computed Tomography (CT)\" or go to *****://*****.*****.*****.*****/*****/*****.   FINDINGS:  Visualized lung bases:  Please see separately dictated report from same day for chest findings.   Liver:    Interval development of periportal edema involving the lateral segment \n of the left hepatic lobe with associated mild biliary dilatation. This may be postsurgical in nature. However, tumor infiltration is not entirely excluded. Attention on follow-up.  Small pneumobilia in the left lobe is noted. Patient is status post hepaticojejunostomy.  Interval increase in size of a 14 mm hypodense lesion in segment March 14 (previously 9 mm). New 5 mm hypodense lesion in segment 8/4A is noted and is suspicious for metastatic disease. Other hypodense liver lesions are not significantly changed.  Gallbladder:  Surgically absent.   Spleen:  Unremarkable  Pancreas:  Patient is status post Whipple procedure . There is moderate dilatation of the main pancreatic duct, stable to slightly increased from prior exam.  Adrenal Glands:  Unremarkable  Kidneys:  Unremarkable  Vasculature:  Unremarkable  GI Tract:  Status post Whipple procedure. Otherwise, unremarkable.  Pelvis:  Enhancing mass in the endometrial canal is noted. This may be secondary to an intracavitary fibroid. However, recommend pelvic ultrasound to exclude malignancy.  Lymphadenopathy: Multiple reactive mesenteric lymph nodes are noted.  Ascites: Small ascites. There is small amount of fluid and stranding around the central mesentery.  Bones:  No suspicious lesions     04/26/2015   IMPRESSION: //ALERT//   1.Interval increase in size of a 14 mm hypodense lesion in segment March 14 and new 5 mm hypodense lesion in segment 8/4A, suggestive of worsening metastatic disease.   2. Interval development of periportal edema involving the lateral segment of the left hepatic lobe with associated mild biliary dilatation. This may be postsurgical in nature. However, tumor infiltration is not entirely excluded. Attention on follow-up.  3. Enhancing mass in the endometrial canal is noted. This may be secondary to an intracavitary fibroid. However, recommend pelvic ultrasound to exclude malignancy.  END OF IMPRESSION:     Ct Chest With Contrast    04/26/2015   CT CHEST \n WITH CONTRAST   *****/*****/***** 8:37 AM  CLINICAL HISTORY:  Pt s/p resection for pancreatic cancer; r/o recurrence  COMPARISON: 03/22/2015 chest radiograph; 01/12/2015 CT chest; 06/15/2014 PET/CT from an outside hospital.  TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the administration of intravenous contrast.  RADIATION DOSE INDICATORS:  ***** *****-***** received 1 exposure event(s) (excluding scout) during this CT study. The CTDIvol (mGy) and DLP (mGy-cm) radiation dose values for each event are:  Event: 1;  anatomic area: Chest/Abdomen/Pelvis;  phantom: body;  CTDIvol: 11.1;  DLP: 699.1  The following accession numbers are related to this dose report [*****]: *****  The dose indicators for CT are the volume Computed Tomography Dose Index (CTDIvol) and the Dose Length Product (DLP), and are measured in units of mGy and mGy-cm, respectively. These indicators are not patient dose, but values generated from the CT scanner acquisition factors.  Individual patient doses calculated from these values will depend on patient size and can substantially underestimate or overestimate the actual dose. For additional information on radiation dose, safety, and protection in the UCSF Department of Radiology and Biomedical Imaging, please refer to one of our FAQ leaflets on \"Computed Tomography (CT)\" or go to *****://*****.*****.*****.*****/*****/*****.  FINDINGS:  Bilateral nonspecific thyroid nodules again noted, including an 8 mm hypodense nodule in the right thyroid lobe.  Scattered lower lobe subsegmental atelectasis is new from previous. No consolidation or pleural effusion. Unchanged calcified granuloma in the right upper lobe. Few scattered pulmonary nodules measuring up to 3 mm in the subpleural aspect of the right upper lobe are unchanged, series 301 image 58.  No lymphadenopathy. Scattered calcified mediastinal and hilar lymph nodes are unchanged and suggestive of old \n granulomatous disease.  The heart and great vessels show no significant abnormality. Right chest wall Port-A-Cath tip extends through the tricuspid valve into the right ventricle, previously terminating in the right atrium.  No suspicious bone lesions are identified. Mild degenerative changes of the thoracic spine.  Please see a separately reported examination for evaluation of the abdomen.     04/26/2015   IMPRESSION:  Few scattered tiny pulmonary nodules measuring up to 3 mm, unchanged since 09/16/2014. Continued followup per clinical oncologic protocol recommended.  END OF IMPRESSION:      Assessment:    1. Borderline resectable/locally advanced adenocarcinoma the pancreas, s/p neoadjuvant chemotherapy followed by resection, with early evidence of disease recurrence.  2. ECOG 0    Plan:    I'm very concerned about her imaging studies. The lesions are too small to evaluate further at this point. We will repeat her scans in 2 months. If indicated, we will seek a confirmatory biopsy. I spent 20 minutes in face-to-face consultation with the patient and her husband today going over all aspects of her care and management.  \n
```

---

## Letter A

Dear Patient,

Thank you for trusting us with your care. Here’s a summary of your recent health updates:

**Diagnosis:** You have a type of cancer called adenocarcinoma in your pancreas. This is a serious condition, but we are working hard to help you.

**Treatment:** You had surgery to remove the cancer. The surgery went well, and there were no signs of cancer left behind. After the surgery, you felt better, but you still had some discomfort in your belly.

**Tests:** We did several tests to check how you are doing. One test showed that a part of your liver might have some swelling. We also found a small spot in your liver that could be cancer coming back. We need to keep watching this closely.

**Next Steps:** We will do more tests in two months to see if the spots in your liver are growing. If they are, we might need to take a small piece of tissue to look at it more closely.

**What to Watch For:** Keep an eye on any new pain or discomfort in your belly. Let us know if you notice anything different.

**General Health:** Your blood tests show that you are doing okay overall. You have lost some weight since your surgery, but this is normal.

Please talk to your care team if you have any questions or concerns. We are here to support you every step of the way.

Warm regards,

[Your Medical Communication Specialist]  
[Cancer Center]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Imaging shows early signs of the cancer coming back. There is a growing spot in your liver and a new small spot, which might mean the cancer is getting worse. Also, there is swelling around some blood vessels in your liver, which could be due to the surgery or the cancer. A small growth in your uterus needs more checking to make sure it’s not cancer. You’ve lost weight since your surgery.
**What treatment or medication changes were made?**
No changes were made to your medications or treatments.
**What is the plan going forward?**
We will repeat your scans in 2 months to check on the spots in your liver and the growth in your uterus. If needed, we will take a sample to confirm if it’s cancer. Your next visit will be in 2 months for these scans.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 7

**Cancer Type:** Adenocarcinoma of the tail of the pancreas

## Original Clinical Note

```
SUBJECTIVE:    This is a very pleasant 68-year-old woman who presented with symptoms of small bowel obstruction in early June 2020. She was hospitalized and treated conservatively. CT scan of the abdomen during her hospitalization showed a mass in the tail of the pancreas measuring 4.8 x 3.6 x 2.9 cm. She was also noted to have a complex cyst in her right adnexa. An EUS was done but biopsy was not attempted as there was a large blood vessel running through the mass. CT PET was done which showed that the mass in the tail of the pancreas was PET avid. The cystic pelvic mass was not PET avid.    She was then seen and evaluated by Dr. ***** at *****. A pancreas protocol CT scan was done which revealed extension of the mass to the posterior gastric wall as well as abutment and possible invasion of the left adrenal gland and occlusion of the splenic vein. Dr. ***** recommended neoadjuvant chemotherapy.    We started her on standard schedule gemcitabine and Abraxane. She did not tolerate this well with neutropenia and we switched her to alternate week and she has tolerated that much better. She's now had 4 full cycles and returns for CT evaluation.    She is doing well. She has no cancer pain but take oxycodone for chronic back and shoulder pain. Appetite waxes and wanes but weight is stable. Denies symptoms of neuropathy. Energy level recovers after chemotherapy. She has a facil rash which has come on recently. It's tolerable.    She has been seen by Gyn Onc regarding the right adnexal mass. Assuming she does go to surgery, they would like to coordinate and remove this.    Germ line testing showed no pathologic variants. VUS seen in AXIN1, CTC1, ERCC4, MC1R.      PAST MEDICAL HISTORY:       MEDICAL ILLNESSES:  Chronic opioid use  arthritis  chronic back pain  hepatitis C, treated in 2010 with good viable clearance  hypertension  history of renal lithiasis    PRIOR SURGERIES:  Ankle ligament \n reconstruction  bilateral bunionectomy  correction of hammertoe  bilateral knee replacement, 2003  repeat right knee replacement, 2017, lumbar fusion 2000  posterior laminectomy/decompression lumbar spine, 1999  excision of breast cysts, age 18 and 21    INJURIES:  None reported    CURRENT MEDICATIONS:    Current Outpatient Medications on File Prior to Visit   Medication Sig Dispense Refill   \u0007 bisacodyL (DULCOLAX) 5 mg EC tablet Take 5 mg by mouth daily as needed for Constipation     \u0007 cholecalciferol, vitamin D3, 1000 UNITS tablet Take 1,000 Units by mouth daily     \u0007 cloNIDine (CATAPRES) 0.1 mg/24 hr patch Place 1 patch onto the skin every 7 (seven) days Use as instructed     \u0007 cyanocobalamin, Vitamin B12, (VITAMIN B-12) 500 mcg tablet Take 500 mcg by mouth daily     \u0007 diclofenac (VOLTAREN) 1 % gel Apply topically 4 (four) times daily     \u0007 docusate sodium (COLACE) 100 mg capsule Take 100 mg by mouth every morning     \u0007 fexofenadine (ALLEGRA) 60 mg tablet Take 60 mg by mouth every morning     \u0007 fluocinonide (LIDEX) 0.05 % ointment Apply topically Twice a day Use as instructed     \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 mg by mouth every 8 (eight) hours as needed for Anxiety     \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet (8 mg total) by mouth every 8 (eight) hours as needed (nausea and vomiting) (Patient not taking: Reported on 12/07/2020  ) 30 tablet 5   \u0007 oxybutynin (DITROPAN XL) 5 mg 24 hr tablet Take 5 mg by mouth daily     \u0007 oxyCODONE (OXY-IR) 5 mg capsule Take 5 mg by mouth every 4 (four) hours as needed     \u0007 pantoprazole (PROTONIX) 40 mg EC tablet Take 40 mg by mouth every morning     \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting) (Patient not taking: Reported on 12/07/2020  ) 30 tablet 5   \u0007 senna (SENOKOT) 8.6 mg tablet Take 1 tablet by mouth 2 (two) times daily     \u0007 UNABLE TO FIND Med Name: Equi-Block topical \n ointment; apply to right shoulder prn       Current Facility-Administered Medications on File Prior to Visit   Medication Dose Route Frequency Provider Last Rate Last Admin   \u0007 [COMPLETED] iohexoL (OMNIPAQUE) 350 mg iodine/mL solution 150 mL  150 mL Intravenous ***** ***** *****, DO   130 mL at 12/08/20 1145     ALLERGIES:  Allergies/Contraindications   Allergen Reactions   \u0007 Penicillins Rash       FAMILY HISTORY:  The only cancer in her family is in maternal aunt with leukemia.    SOCIAL HISTORY:  Her first husband died at  *****. She is married to ***** and has been married to him for ***** ***** yrs. They have one child together. Each has a child from a prior relationship. Both are retired. She previously worked for ***** ***** a   *****  *****. She has not used alcohol since 2010 coincident with her diagnosis and treatment of hepatitis C. She is a former tobacco user and last smoked about 20 years ago.      PE:    Vitals: There were no vitals taken for this visit.   In general, she appears comfortable.      LABORATORY RESULTS:  Results for orders placed or performed during the hospital encounter of 11/25/20   Alkaline Phosphatase   Result Value Ref Range    Alkaline Phosphatase 71 38 - 108 U/L   Bilirubin, Total   Result Value Ref Range    Bilirubin, Total 0.4 0.2 - 1.2 mg/dL   Alanine Transaminase   Result Value Ref Range    Alanine transaminase 31 10 - 61 U/L   Aspartate Transaminase   Result Value Ref Range    Aspartate transaminase 30 5 - 44 U/L   Creatinine, Serum / Plasma   Result Value Ref Range    Creatinine 0.56 0.55 - 1.02 mg/dL    eGFR - low estimate 96 >59 mL/min    eGFR - high estimate 111 >59 mL/min   Urea Nitrogen, Serum / Plasma   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 11 7 - 25 mg/dL   Complete Blood Count with Differential   Result Value Ref Range    WBC Count 5.6 3.4 - 10.0 x10E9/L    RBC Count 4.11 4.00 - 5.20 x10E12/L    Hemoglobin 12.1 12.0 - 15.5 g/dL    Hematocrit 38.3 \n 36.0 - 46.0 %    MCV 93 80 - 100 fL    MCH 29.4 26.0 - 34.0 pg    MCHC 31.6 31.0 - 36.0 g/dL    Platelet Count 138 (L) 140 - 450 x10E9/L    Neutrophil Absolute Count 3.22 1.80 - 6.80 x10E9/L    Lymphocyte Abs Cnt 1.54 1.00 - 3.40 x10E9/L    Monocyte Abs Count 0.58 0.20 - 0.80 x10E9/L    Eosinophil Abs Ct 0.17 0.00 - 0.40 x10E9/L    Basophil Abs Count 0.04 0.00 - 0.10 x10E9/L    Imm Gran, Left Shift 0.01 <0.10 x10E9/L   Cancer Antigen 19-9   Result Value Ref Range    Cancer Antigen 19-9 8 <38 U/mL       IMAGING:  Ct Abdomen /pelvis With And Without Contrast    Result Date: 12/08/2020  CT ABDOMEN/PELVIS WITH AND WITHOUT CONTRAST  *****/*****/***** 11:29 AM CLINICAL HISTORY: Pt with pancreatic cancer and pelvic mass receiving chemo needs restaging scan. needs pancreas protocol imaging COMPARISON:  10/13/2020 Techniques: Contiguous 1.25 mm collimation images were obtained through the abdomen without intravenous contrast. Subsequently, 1.25 mm axial images were acquired through the abdomen during the arterial phase followed by 1.25 mm axial images through the abdomen and pelvis during portal venous phase, and through the abdomen at 5 minutes delay. CONTRAST MEDIA: Intravenous contrast Findings: Pancreatic tumor: 1)  Location:  Pancreatic tail 2)  Size:  Hypoenhancing mass 4.2 x 4 cm (series 304, image 52) 3)  Enhancement relative to pancreas: Hypoenhancing, no definite necrosis. Areas of calcifications. 4)  Biliary obstruction: no 5)  Pancreatic duct obstruction: no The mass abuts the hilum of the spleen and the left adrenal gland. Vascular-arterial:        1) Celiac axis is not involved        2)  SMA is not involved.        3)  CHA is not involved.        4)  Arterial variant: None Splenic artery is involved. Solid tumor contact greater than 180 degrees. Focal narrowing of the splenic artery. Vascular-venous: 1)  MPV is not involved. 2)  SMV is not involved. 3)  Thrombus in vein:  present 4)  Venous collateral:  present. \n Perigastric varices. Splenic vein is involved with occlusion propagation of nonocclusive thrombus towards the central splenic vein. Liver: Unremarkable Gallbladder: Cholelithiasis without evidence of cholecystitis. Peritoneal or omental nodules:  None Ascites: absent Lymph nodes:  No pathologically enlarged nodes Spleen: See above Adrenal: See above Kidney: Pancreatic tail mass abuts the anterior margin of the left upper renal pole. Subcentimeter hypodense lesions in the bilateral kidneys, too small to characterize. GI tract: The pancreatic tail mass abuts the gastric fundus. No bowel obstruction. Visualized lung bases:  Unremarkable Bones:  No suspicious lesions L3-L5 PLIF. Extraperitoneal soft tissues: Unremarkable Reproductive: Cystic pelvic neoplasm measures up to 12 cm, similar compared to prior examination. Fibroid uterus. RADIATION DOSE INDICATORS:Exposure Events: 8 , CTDIvol Max: 23.1 mGy, DLP: 3638.5 mGy.cm. The following accession numbers are related to this dose report *****,*****     1.  Compared to 10/13/2020, slight decrease in size of pancreatic tail mass with persistent extension/abutment of the left adrenal gland, splenic hilum, gastric fundus, and anterior left upper renal pole. 2.  Pancreatic mass encases and narrows the splenic artery. Pancreatic tail mass encases and occludes the distal splenic vein with propagation of nonocclusive thrombus towards the central splenic vein. 3.  Cystic pelvic neoplasm appears to arise from the right adnexa measures up to 12 cm. Report dictated by: ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging    Ct Chest With Contrast    Result Date: 12/08/2020  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Pancreatic cancer, adenocarcinoma Pt with pancreatic cancer and pelvic mass receiving chemo needs restaging scan. COMPARISON: CT chest 3 01/10/2021 TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained \n after the administration of intravenous contrast. RADIATION DOSE INDICATORS: Exposure Events: 8 , CTDIvol Max: 23.1 mGy, DLP: 3638.5 mGy.cm. The following accession numbers are related to this dose report *****,***** FINDINGS: LUNGS: Clear lungs. No suspicious pulmonary nodules. PLEURA: No pleural effusion or pneumothorax. MEDIASTINUM: No lymphadenopathy by CT size criteria. HEART/GREAT VESSELS: Normal heart size. Normal caliber thoracic aorta and main pulmonary artery. BONES/SOFT TISSUES: No suspicious soft tissue or osseous lesions. VISIBLE ABDOMEN: Please see dedicated Abdomen and Pelvis CT report.     No evidence of metastatic disease in the chest. Report dictated by: ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging      ASSESSMENT:  1. Adenocarcinoma the tail of the pancreas with local extension with radiographic evidence of response to treatment  2. Right cystic adnexal mass, likely benign  3. Chronic opioid use  4. History of hepatitis C, status post successful treatment  5. hypertension  6. history of renal lithiasis  7. ECOG 1    PLAN:    Will continue on with treatment without dose or schedule modification. We will repeat her imaging again in 8 weeks.    I performed this consultation using real-time Telehealth tools, including a live video connection between my location and the patient's location. Prior to initiating the consultation, I obtained informed verbal consent to perform this consultation using Telehealth tools and answered all the questions about the Telehealth interaction.    I spent a total of 40 minutes in face-to-face time with the patient and in non-face-to-face activities conducted today 12/09/2020 directly related to this video visit, including reviewing records and tests, obtaining history and exam, placing orders, communicating with other healthcare professionals, counseling the patient, family or caregiver, documenting in the \n medical record, and/or care coordination for the diagnoses above.              \n
```

---

## Letter A

Dear Patient,

Thank you for your trust in our care team. Here’s a summary of your recent medical information:

**Diagnosis:** You have a type of cancer called adenocarcinoma in the tail of your pancreas. This means the cancer started in the cells lining the pancreas. The cancer is in Stage 4, which means it has spread beyond the pancreas.

**Treatment Plan:** You are currently receiving chemotherapy to help shrink the cancer. The medicines you are taking are called gemcitabine and Abraxane. You are having these treatments every two weeks. So far, you have completed four full cycles.

**Key Test Results:** Recent scans show that the cancer in your pancreas has gotten smaller. This is good news and shows that the treatment is working. You also have a cyst in your right ovary, but it looks like it is not cancerous.

**Next Steps:** Your next scan is scheduled in eight weeks to check how the cancer is responding to the treatment. You should continue taking your medications as prescribed.

**What to Watch For:** If you notice any new symptoms such as severe pain, nausea, or changes in appetite, please let us know. You may also experience some side effects from the chemotherapy, such as a rash. If the rash becomes uncomfortable, talk to your doctor.

**Medications:** You are taking several medications to manage different health issues. These include pain medicine, vitamins, and other drugs to help with your overall health.

**Past Medical Issues:** You have a history of chronic back pain, arthritis, and high blood pressure. You were also treated for hepatitis C, and the treatment was successful.

**Social History:** You are married and have a supportive family. You do not drink alcohol and quit smoking many years ago.

Please remember to discuss any questions or concerns with your care team. We are here to support you throughout your treatment journey.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your cancer treatment.
**What's new or changed since your last visit?**
The cancer is showing signs of getting smaller. Specifically, the tumor in your pancreas has slightly decreased in size compared to previous imaging. However, it still touches the left adrenal gland, spleen, and stomach. You also have a rash that is tolerable.
**What treatment or medication changes were made?**
Your treatment was switched from a standard schedule of gemcitabine and Abraxane to an alternate week schedule because you experienced neutropenia (low white blood cell count). You are also taking ondansetron to manage nausea.
**What is the plan going forward?**
You will continue with the same treatment without changing the dose or schedule. Your imaging will be repeated in 8 weeks. Your next visit will be a telehealth appointment in 8 weeks.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 8

**Cancer Type:** Poorly differentiated pancreatic ductal adenocarcinoma with lymphovascular invasion

## Original Clinical Note

```
Ms. ***** is a 69 y.o. female who comes to our GI Oncology practice at the UCSF Comprehensive Cancer Center in consultation for her recurrent metastatic pancreatic adenocarcinoma. The history was obtained from the patient, along with my review of all her outside records.    History of Present Illness  This pleasant woman with a known germline ATM mutation was initially diagnosed in May 2019 with resectable pancreatic adenocarcinoma after presenting with painless jaundice. CT A/P at the time (05/16/19) showed a 2x2.7x2.9 cm pancreatic head mass with associated pancreatic ductal and biliary ductal dilatation. EUS-guided FNA (05/18/19) was consistent with pancreatic adenocarcinoma. She underwent several ERCPs for biliary stent placement and exchange, and ultimately required a percutaneous biliary drain (06/05/19) but w/o complete resolution of her hyperbilirubinemia.     She received 1 cycle of FOLFOX chemotherapy on 06/15/19 (with irinotecan on hold due to elevated bilirubin), and subsequently was taken for a Whipple resection with Dr. ***** ***** (*****) on 07/09/19. Pathology showed a pT2N2, 3.3 cm poorly differentiated adenocarcinoma, LVI present, PNI negative, 11/37 lymph nodes, negative margins, MMR intact.     Postoperative CT scan (08/02/19) showed a new soft tissue area inferior to the pancreas, c/w either postoperative changes or recurrent/residual disease. She then started back on chemotherapy with FOLFIRINOX on 08/07/19 and completed 11 cycles in total, through 01/09/20.     She underwent incisional hernia repair in June 2020, after which time she noted persistent mid-back pain. Surveillance CT C/A/P (09/10/20) showed enlarging aortocaval and gastrohepatic lymph nodes, new subcentimeter mesenteric lymph nodes, and ongoing hazy attenuation/stranding encasing the peripancreatic vasculature. EUS-guided sampling of an 18 mm lymph node in the gastrohepatic ligament region confirmed metastatic adenocarcinoma \n (09/12/20).    She started on the combination of gemcitabine and nab-paclitaxel with an every other week dosing schedule beginning on 09/19/20. Recent restaging CT (10/24/20) after approximately 6 weeks on treatment showed mildly decreased size of intra-abdominal lymph nodes which have become less defined. No new intra-abdominal or intrathoracic disease. New subtle groundglass opacities throughout the lungs in the upper lobes may be related to an infectious/inflammatory etiology.     The patient's present review of systems was reviewed. She denies any pain at present (currently on once-daily long-acting morphine), and reports tolerating current chemotherapy very well, aside from mild fatigue and transient nausea. Appetite is good and weight is stable/improving. She has minimal non-debilitating peripheral sensory neuropathy. Bowel movements are regular and well-formed. She continues to be very active, with an ECOG PS of 0.      PAST MEDICAL AND SURGICAL HISTORY  Hypertension  Hyperlipidemia  Adenomatous polyp (2011)  History of cryptogenic organizing pneumonia (2013)    ALLERGIES: Erythromycin causing rash    MEDICATIONS:   \u0007 acetaminophen (TYLENOL PO) take by mouth every day   \u0007 amylase-lipase-protease (Creon 24,000) 120,000-24,000-76,000 delayed release capsule take 2 Caps by mouth 3 times a day with meals (Patient taking differently: take 2 Caps by mouth 2 times a day ) 270 Cap 3   \u0007 cholecalciferol (vitamin D3) (VITAMIN D3 PO) take 5,000 Int'l Units by mouth every day   \u0007 morphine (MS Contin) 15 mg controlled release tablet take 1 Tablet (15 mg total) by mouth daily 30 Tablet 0   \u0007 polyethylene glycol (MIRAlax) 17 gram packet take 17 g by mouth One Time   \u0007 potassium chloride (Klor-Con) 20 mEq sustained release tablet take 2 Tabs (40 mEq total) by mouth 2 times a day (Patient taking differently: take 20 mEq by mouth every morning ) 60 Tab 2    PERSONAL AND SOCIAL HISTORY  Married, husband recently diagnosed with \n prostate cancer  Two adult children live in *****  Retired  Previous smoker, 10 cigarettes/day x 30 years, quit at age 49      FAMILY HISTORY   Paternal grandmother with esophageal cancer   Maternal aunt with melanoma diagnosed in her 40s, breast cancer diagnosed in her 70s, died at 78y   Maternal aunt with brain tumor, died four months following diagnosis.   Maternal first cousin, son to above, with uveal melanoma, died at 63y.   Maternal aunt with ovarian cancer diagnosed in her 70s, died at 78y.    No ***** Jewish ancestry        Physical Exam:  Blood pressure 126/65, pulse 52, temperature 36.9 C (98.4 F), temperature source Temporal, resp. rate 16, height 164 cm (5' 4.57\"), weight 70 kg (154 lb 4.8 oz), SpO2 98 %. ECOG PS 0.  General appearance: No acute distress, conversant, appropriately interactive. Non-toxic, well-appearing.   Eyes: anicteric sclerae, moist conjunctivae; no lid lag.   HEENT: Atraumatic; oropharynx clear with moist mucous membranes and no mucosal ulcerations; normal hard and soft palate  Neck: Full range of motion, supple, no lymphadenopathy  Lungs/respiratory: Clear to auscultation, with normal respiratory effort and no intercostal retractions  Cardiovascular: RRR, normal S1 S2, no murmurs, gallops, or rubs.  Abdomen/GI: Soft, non-tender, nondistended; no masses, ascites, or hepatomegaly. Tympanitic to percussion.  Back: No point spinal tenderness.  Extremities: No peripheral edema or discoloration palpated or visualized.  Skin: Normal temperature, turgor and texture; no rash, ulcers or subcutaneous nodules visualized or palpated.  Neuro: Normal bulk, tone, and strength in upper and lower extremities. Intact gait. No focal sensorimotor deficits.  Psych: Appropriate affect, alert and oriented.      RELEVANT DIAGNOSTIC STUDIES    Most recent labs notable for the following:   11/21/20  WBC 5, Hb 10.7, plt 137  Cr 0.67  ALT 70, AST 66, ALP 522  Total bili 0.5, alb 4    CA 19-9 \n levels  11/21/20=61  10/24/20=133  10/10/20=433  09/19/20=746        FNA 09/12/20  CYTOLOGIC DIAGNOSIS:   lymph node, gastrohepatic, endoscopic us-guided fine needle   aspiration        --   metastatic adenocarcinoma (see comment)     COMMENT:  The aspirate smear and cell block contain a malignant   population of cells with increased nuclear size, increased nuclear   to cytoplasmic ratio, and irregular nuclear contours. The malignant   cells are arranged in loosely cohesive clusters within a background   of lymphoid tissue. Collectively, the features are diagnostic of   metastatic adenocarcinoma. Suggest clinical and radiologic   correlation.    Whipple resection 07/09/19  DIAGNOSIS (MICROSCOPIC):   A. PANCREAS, NECK MARGIN, BIOPSY (FSA)   --   NEGATIVE FOR CARCINOMA (SEE COMMENT)   B. PANCREAS, BILE DUCT MARGIN, BIOPSY (FSB)   --   NEGATIVE FOR CARCINOMA   C. PANCREAS, SMA MARGIN, BIOPSY (FSC)   --   NEGATIVE FOR CARCINOMA   D. LYMPH NODE, ADJACENT TO D4, BIOPSY   --   ONE LYMPH NODE NEGATIVE FOR CARCINOMA (0/1)   E. LYMPH NODE, HEPATIC ARTERY, BIOPSY   --   ONE LYMPH NODE NEGATIVE FOR CARCINOMA (0/1)   F. LYMPH NODE, CALOT, BIOPSY   --   ONE LYMPH NODE NEGATIVE FOR CARCINOMA (0/1)   G. PANCREAS, REMAINDER OF CYSTIC DUCT, BIOPSY   --   NEGATIVE FOR CARCINOMA   H. PANCREAS AND DUODENUM AND GALLBLADDER, PANCREATICODUODENECTOMY   AND CHOLECYSTECTOMY        PANCREAS AND DUODENUM   --   INVASIVE, MODERATELY TO POORLY-DIFFERENTIATED ADENOCARCINOMA   3.3 CM, EXCISED (SEE COMMENT AND SUMMARY TABLE)   --   METASTATIC ADENOCARCINOMA IN ELEVEN OF THIRTY FOUR LYMPH NODES   (11/34)        GALLBLADDER             --   NO SIGNIFICANT ABNORMALITY     COMMENT:  Histologic sections of the Whipple specimen (part H)   demonstrate the adenocarcinoma is located 0.1 cm from the vascular   groove.  Multiple lymph nodes are positive for metastatic carcinoma   (April 34) and there is extensive lymphovascular invasion and small   vein invasion.  We \n have reviewed the frozen section slides and   concur with the intraoperative diagnoses. Immunohistochemistry for   mismatch repair proteins is being performed on block ***** with   results to be reported in an addendum.  Additional prognostic   details are provided in the summary table below.     PANCREAS (EXOCRINE)   SPECIMEN   Procedure:     Pancreaticoduodenectomy (Whipple resection), partial   pancreatectomy   TUMOR   Tumor Site:    Specified   Pancreatic head   .   Histologic Type:    Ductal adenocarcinoma   Histologic Grade:   G3: Poorly differentiated   Tumor Size:    3.3cm   Tumor Extent   Tumor Extension:    Evidence of primary tumor   Can be assessed   Tumor invades duodenal wall   Accessory Findings   Treatment Effect:   Cannot be determined   No tumor regression however only one cycle of neoadjuvant   chemotherapy is given.   Lymphovascular Invasion: Present   Perineural Invasion:     Not identified   MARGINS   All margins are uninvolved by invasive carcinoma and high-grade   intraepithelial neoplasia   Margins Examined:   Pancreatic neck / parenchymal   Uncinate (retroperitoneal / superior mesenteric artery)   Bile duct   Proximal (gastric or duodenal)   Distal (duodenal or jejunal)   Distance of Invasive Carcinoma from Closest Margin:    0.7cm   Closest Margin:     Uncinate (retroperitoneal / superior mesenteric   artery)   LYMPH NODES   Regional Lymph Nodes:    Lymph nodes submitted or found   Number of Lymph Nodes Involved:    11   Number of Lymph Nodes Examined:    37   PATHOLOGIC STAGE CLASSIFICATION  (pTNM, AJCC 8th Edition)   TNM Descriptors:    Not applicable   .   Primary Tumor (pT): pT2   Regional Lymph Nodes (pN):    pN2   Distant Metastasis (pM):   Not applicable - pM cannot be determined from the submitted   specimen(s)    MMR proteins all intact by IHC.      I personally reviewed and interpreted the patient's relevant imaging studies in conjunction with the corresponding formal radiology \n reports, as follows:    CT C/A/P 10/24/20  FINDINGS:     CHEST:   Medical devices: Right-sided port with tip in the cavoatrial junction     Thyroid: Normal.       Lymph nodes: No supraclavicular, axillary, mediastinal, or hilar lymphadenopathy.     Vasculature: Aorta and main pulmonary artery diameters are within normal range.     Heart: Mild coronary artery calcification. No pericardial effusion.     Other mediastinal structures: Normal.     Lung parenchyma and pleura: No new or enlarging nodules. Unchanged 3 mm nodule in the apical right upper lobe (January 70). Subtle areas of groundglass throughout both lungs (8/80, 96, 106). No pleural effusions.     Airways: Normal.     Chest wall: Normal.     ABDOMEN/PELVIS:   Liver and biliary tree: Unchanged hypoattenuating lesion in segment IVb measuring 0.7 cm (7/83) which may represent a hemangioma (MR 05/30/2019). No new or enlarging lesions. Mild intrahepatic duct dilation with left-sided pneumobilia, unchanged. Mild periportal edema.     Gallbladder: Surgically absent.     Spleen: Enlarged measuring up to 13.3 cm.     Pancreas: Status post Whipple's. Remaining parenchyma is atrophic, unchanged. No focal lesions.     No sign of any change of hazy stranding surrounding the pancreas with greater than 180 degrees involvement of the celiac axis and common hepatic artery, gastroduodenal artery as well as the main portal and splenic veins..     Adrenal glands: Unchanged left adrenal adenoma and thickening. Right adrenal is normal.     Kidneys and ureters: Normal.     Gastrointestinal tract: Post Whipple changes. Anastomosis appears intact. No evidence of obstruction. Diverticulosis without evidence of diverticulitis. Appendix is normal.     Peritoneal cavity: No free air or fluid. Similar mesenteric edema.     Bladder: Normal.     Uterus: Normal.     Ovaries: Normal.     Vasculature: Atherosclerosis without aneurysm.     Lymph nodes: Several lymph nodes have mildly decreased in \n size and overall appear less defined. For reference:   -Portacaval node measuring 2.4 x 1.2 cm (December 94) is unchanged   -Gastrohepatic measuring 1.8 x 1.2 cm (December 76) (previously 2.0 x 1.5 cm)   -Aortocaval measuring 1.5 x 0.7 cm (December 236) (previously 2.2 x 1.6 cm)     No new enlarged nodes.     Abdominal wall: Post surgical changes in the ventral wall..     Musculoskeletal: Degenerative change of the spine. No aggressive osseous lesions    IMPRESSION:   1.  Postsurgical changes from Whipple's. No significant change of peripancreatic stranding, for attention on follow-up.   2.  Mildly decreased size of intra-abdominal lymph nodes which have become less defined which may be posttreatment related. No new intra-abdominal disease.   3.   No evidence of intrathoracic disease. New subtle groundglass opacities throughout the lungs particularly in the upper lobes may related to infectious/inflammatory or drug related etiologies.      Impression and Recommendations:  In summary, Ms. ***** is a 69 y.o. female with a known germline ATM mutation and recurrent metastatic pancreatic adenocarcinoma. She was initially diagnosed in May 2019 with resectable disease and received 1 cycle of FOLFOX prior to Whipple resection (07/09/19) with pT2N2 disease (11/37 lymph nodes), followed by 11 cycles of adjuvant FOLFIRINOX, completed in 01/09/20. She was subsequently diagnosed with biopsy-proven mesenteric nodal recurrence on 09/12/20, and has since received 3 cycles of gemcitabine/nab-paclitaxel with evidence of favorable treatment response to date. She feels entirely well and is holding her own excellently from a functional and nutritional standpoint    We reviewed her diagnosis, pathology, investigations, and treatment, reinforcing why the mainstay of treatment should consist of systemic therapy with the goal of achieving as deep and durable a remission as possible, hopefully translating into significant prolongation of life, but that treatment \n was not necessarily expected to be curative. As such, it would be important to carefully weigh the risks/benefits of therapy and to prioritize QoL considerations in the process.    We affirmed her treatment thus far, and would recommend continuing on gemcitabine/nab-paclitaxel with an every other week dosing schedule while she is tolerating it adequately, aiming to treat to maximal radiographic/biochemical response or at least 6 cycles in total before considering building in a chemotherapy holiday. She did also inquire about the role for radiation treatment to address any residual disease; I explained that while this can be offered for palliative purposes or to treat isolated recurrences, this modality would not be associated with any clear benefit in terms of prolonging survival.    In terms of future treatment possibilities at a point when her disease progresses, aside from rechallenging with SoC chemotherapy (the combination of 5-FU/LV plus nanoliposomal irinotecan (nal-IRI, or Onivyde) could conceivably be tried, but would likely be of limited yield), we talked about exploring clinical trials which we would be happy to avail her of at *****, although this would very much depend on her overall functional status and goals of care. Possibilities in the future could include
```

---

## Letter A

Dear Ms. *****,

Thank you for coming to our cancer center. Here is a summary of your recent visit and treatment plan:

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. It has come back and spread to other parts of your body. This is called recurrent metastatic pancreatic adenocarcinoma.

**Treatment Plan:**
You are currently receiving chemotherapy with two medicines: gemcitabine and nab-paclitaxel. These medicines help fight the cancer. You will continue this treatment as long as it works well and you feel okay.

**Key Test Results:**
Recent tests show that the cancer is getting smaller. This is good news. Your blood tests also show that you are doing well.

**Next Steps:**
We will keep giving you these medicines every other week. We will check how you are doing with more tests in the future. If the cancer starts growing again, we will talk about other treatments like joining a research study.

**What to Watch For:**
Please let us know if you have any new symptoms or if you feel very tired. Also, tell us if you have any pain or trouble eating.

**Questions:**
If you have any questions, please talk to your doctor or nurse. They are here to help you.

Take care and stay strong.

Best regards,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation regarding your pancreatic cancer.
**What's new or changed since your last visit?**
You have a poorly differentiated pancreatic ductal adenocarcinoma (cancer that started in gland cells) with lymphovascular invasion. The cancer is now metastatic (pt2n2) and has spread to lymph nodes.
Recent imaging shows that the size of the lymph nodes has decreased, indicating a positive response to treatment. Additionally, your CA 19-9 levels, a tumor marker, have decreased, suggesting the treatment is working well.
**What treatment or medication changes were made?**
You are receiving supportive medication, including morphine (MS Contin) for pain relief.
Your treatment plan includes continuing on gemcitabine/nab-paclitaxel (a combination of chemotherapy drugs) with an every other week dosing schedule.
**What is the plan going forward?**
The goal of your treatment is to manage symptoms and improve quality of life (palliative care).
You will continue on gemcitabine/nab-paclitaxel until you have completed at least 6 cycles or until the maximum response is achieved.
A genetic test called FoundationOne has been ordered to learn more about your tumor.
You will be seen again in person at your request or as needed for further evaluation.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 9

**Cancer Type:** Metastatic pancreatic cancer

## Original Clinical Note

```
REASON FOR VISIT/CC:   Chief Complaint   Patient presents with   \u0007 Pancreatic Cancer   \u0007 Follow-up     71 y.o. female with metastatic pancreatic cancer, BRCA VUS, Lung predominant disease.       HISTORY OF PRESENT ILLNESS:   ***** ***** ***** presents today for evaluation of her metastatic pancreatic cancer. Her oncologic history is reviewed here:    1st line treatment with FOLFIRINOX *****/*****/*****-*****/*****/***** x 12 cycles; initially treated with 10 cycles- XRT x 2 fractions d/c d/t SLE and then completed 12 full cycles of FOLFIRINOX; s/e -peripheral neuropathy -improved off of treatment    2ndline treatment (PD in lungs biopsy proven) 5FU &liposomal irinotecan *****/*****/***** -*****/*****/***** x 12 cycles; s/e -n/v + relief with addition of olanzapine (Zyprexa)    EUS: Celiac block October 2016 with relief  11/15/16: Restarted 5FU +liposomal irinotecan  12/31/16: Scan showed slight increase in primary, but CA 19-9 was dropping   01/23/2017: C19 5FU +liposomal irinotecan  02/04/17 C20 5FU +liposomal irinotecan   02/21/17 C21 Dose reduced 5FU 66.7%   neutropenia limited therapy in late 2017    04/30/2017: CT scan showed interval progression in the lungs with no disease in the liver  05/14/17: restarted 5FU +liposomal irinotecan    10/18/17: CT showed essentially stable disease with interval progression of a catheter-related clot in the right SVC    10/29/17: IR removal of indwelling venous catheter  And placement of midline venous catheter    10/29/17: cycle 33 5FU +liposomal irinotecan  11/01/17: We removed  and proceeded with single-agent liposomal irinotecan  12/10/17: Cycle 35 single-agent liposomal irinotecan    12/18/17: CT showed stable disease. However, tumor markers had been trending upwards by 30%. Patient was also symptomatic with ongoing nausea/vomiting, which was concerning for disease progression. Liposomal irinotecan was discontinued and she was referred to Phase I for consideration of a trial based on a variant of unknown \n significance found in the BRCA2 gene.    01/21/18: Consultation with Dr. *****, Phase I, and was placed on waitlist for nab-paclitaxel + CORT125134.    *****/*****/***** - *****/*****/*****: Patient presented to ED and later admitted for hyperbilirubinemia and gastric outlet obstruction. ERCP revealed evidence of high grade post-bulbar obstruction and unable to advance the endoscope and the procedure was aborted. Underwent uncomplicated PTBD w IR on February 27. HPB placed a duodenal stent with the proximal end in the gastric antrum and the proximal end in the 2nd portion of the duodenum on February 29.    03/19/18: C1D1 Gemcitabine + Abraxane   04/02/18:  C1D15 Gemcitabine + Abraxane    04/17/18: C2D1 Gemcitabine + Abraxane    05/01/18: C2D15 Gemcitabine + Abraxane    05/20/18: C3D1 Gemcitabine + Abraxane    06/03/18: C3D15 Gemcitabine + Abraxane   06/17/18: C4D1 Gemcitabine + Abraxane      Interim History  Worsening neuropathy.  Improved abd pain.   Trouble sleeping.       Review of Systems   Constitutional: Negative for chills, diaphoresis, fever, malaise/fatigue and weight loss.   HENT: Negative for nosebleeds.    Eyes: Negative.    Respiratory: Negative.  Negative for cough, hemoptysis and shortness of breath.    Cardiovascular: Negative for chest pain and leg swelling.   Gastrointestinal: Positive for abdominal pain. Negative for constipation, diarrhea, nausea and vomiting.   Genitourinary: Negative for flank pain and urgency.   Musculoskeletal: Negative for falls, joint pain and neck pain.   Skin: Negative.  Negative for itching.   Neurological: Positive for sensory change. Negative for dizziness, tremors, speech change, focal weakness and loss of consciousness.   Endo/Heme/Allergies: Negative.    Psychiatric/Behavioral: Negative for depression, memory loss and substance abuse. Suicidal ideas:    The patient is not nervous/anxious and does not have insomnia.    All other systems reviewed and are negative.    Past Medical History:   Diagnosis \n Date   \u0007 Asthma    \u0007 Breast cancer (HCC)    \u0007 Difficult intubation     02/26/2018 - per notes difficult intubation Mac3. easy intubation Glidescope (Glidescope LoPro T4 vs T3 not charted .    \u0007 History of continuous positive airway pressure (CPAP) therapy    \u0007 Lupus    \u0007 Nausea 11/14/2016   \u0007 Obstructive sleep apnea       Past Surgical History:   Procedure Laterality Date   \u0007 BREAST LUMPECTOMY Bilateral 2007   \u0007 Chest port placement  2014, 2015   \u0007 ENDO ADULT EGD WITH PEG/TUBE PLACEMENT  02/28/2018   \u0007 IR PERCUTANEOUS CHOLECYSTOSTOMY DRAIN (ORDERABLE BY IR ONLY)  02/26/2018    IR PERCUTANEOUS CHOLECYSTOSTOMY DRAIN (ORDERABLE BY IR ONLY) 02/26/2018 ***** ***** *****, MD RAD IR PARN   \u0007 IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY)  01/30/2018    IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY) 01/30/2018 ***** *****-***** *****, MD RAD IR MZ   \u0007 LUNG BIOPSY  2016   \u0007 TUBAL LIGATION        Current Medications       Dosage    albuterol 90 mcg/actuation metered dose inhaler Inhale 1-2 puffs into the lungs every 6 (six) hours as needed.     cetirizine (ZYRTEC) 10 mg tablet Take 10 mg by mouth daily as needed.     diphenoxylate-atropine (LOMOTIL) 2.5-0.025 mg tablet Take 2 tabs after each loose bowel movement. You can take up to 8 tabs a day. FOR DIARRHEA.    fluticasone propion-salmeterol (ADVAIR DISKUS) 100-50 mcg/dose diskus inhaler Inhale 1 puff into the lungs Twice a day.    lipase-protease-amylase (CREON) 24,000-76,000 -120,000 unit CAPDREC capsule take 3-4 caps with each meal and April 23 caps with each snack.    LORazepam (ATIVAN) 1 mg tablet Take 1 tablet (1 mg total) by mouth every 8 (eight) hours as needed (nausea/vomiting).    metoclopramide HCl (REGLAN) 10 mg tablet Take 1 tab 30 minutes prior to every meal. Take 3 times a day. For NAUSEA.    mirtazapine (REMERON) 15 mg tablet Take 1 tablet (15 mg total) by mouth nightly at bedtime.    OLANZapine (ZYPREXA) 5 mg tablet Take 1 or 2 tabs on the \n day of chemotherapy followed by 1 or 2 tabs once daily for 3 days following chemotherapy infusion. FOR NAUSEA.    olopatadine (PATANOL) 0.1 % ophthalmic solution Apply 1 drop to eye.    oxyCODONE (ROXICODONE) 5 mg tablet Take 1-2 tablets (5-10 mg total) by mouth every 4 (four) hours as needed for Pain.    ranitidine (ZANTAC) 150 mg tablet Take 150 mg by mouth Twice a day.    TRIAMCINOLONE ACETONIDE (NASACORT NASAL) by Nasal route daily as needed.       Hospital Medications       Dosage    heparin flush 100 unit/mL injection syringe 500 Units Inject 5 mLs (500 Units total) into the vein As needed (line flush after each use).         Allergies/Contraindications   Allergen Reactions   \u0007 Levofloxacin Rash   \u0007 Shellfish Containing Products Hives and Itching   \u0007 Alendronate Sodium Rash   \u0007 Anastrozole Rash   \u0007 Terbinafine Rash      Family History   Problem Relation Name Age of Onset   \u0007 Stomach cancer Mother  71   \u0007 Colon cancer Sister  59   \u0007 Rectal cancer Sister  64   \u0007 Breast cancer Paternal Aunt  45   \u0007 Cancer Paternal Uncle     \u0007 Breast cancer Maternal Grandmother     \u0007 Brain cancer Cousin          Maternal first cousin   \u0007 Breast cancer Paternal Aunt     \u0007 Cancer Paternal Aunt     \u0007 Breast cancer Daughter  42   \u0007 Anesth problems Neg Hx     \u0007 Bleeding disorder Neg Hx       Social History     Socioeconomic History   \u0007 Marital status: Married     Spouse name: Not on file   \u0007 Number of children: Not on file   \u0007 Years of education: Not on file   \u0007 Highest education level: Not on file   Social Needs   \u0007 Financial resource strain: Not on file   \u0007 Food insecurity - worry: Not on file   \u0007 Food insecurity - inability: Not on file   \u0007 Transportation needs - medical: Not on file   \u0007 Transportation needs - non-medical: Not on file   Occupational History   \u0007 Not on file   Tobacco Use   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Yes     Comment: 1x per month \n   \u0007 Drug use: Yes     Types: Marijuana   \u0007 Sexual activity: Not on file   Other Topics Concern   \u0007 Not on file   Social History Narrative   \u0007 Not on file        OBJECTIVE ASSESSMENT  BP 105/59  | Pulse 80  | Temp 36.8 C (98.2 F) (Oral)  | Resp 18  | Ht 154.9 cm (5' 0.98\") Comment: April 2018 | Wt 53.9 kg (118 lb 14.4 oz)  | SpO2 100%  | BMI 22.48 kg/m     ECOG: 1    Physical Exam   Nursing note and vitals reviewed.  Constitutional: She is oriented to person, place, and time. She appears well-developed and well-nourished.   HENT:   Head: Normocephalic.   Right Ear: External ear normal.   Left Ear: External ear normal.   Eyes: Pupils are equal, round, and reactive to light. EOM are normal. No scleral icterus.   Neck: Neck supple. No tracheal deviation present. No thyromegaly present.   Cardiovascular: Normal rate, regular rhythm and intact distal pulses. Exam reveals no friction rub.   Pulmonary/Chest: Breath sounds normal. No respiratory distress. She exhibits no tenderness.   Abdominal: Soft. Bowel sounds are normal. There is no tenderness. There is no rebound.   Musculoskeletal: Normal range of motion. She exhibits no edema, tenderness or deformity.   Neurological: She is alert and oriented to person, place, and time. No cranial nerve deficit. She exhibits normal muscle tone.   Skin: Skin is warm and dry. No rash noted.   Psychiatric: She has a normal mood and affect. Her behavior is normal. Thought content normal.     LABORATORY RESULTS  I have reviewed the following labs in clinic today and discussed them with the patient.  Lab Results   Component Value Date    WBC Count 5.3 08/12/2018    Neutrophil Absolute Count 3.56 08/12/2018    Hemoglobin 10.1 (L) 08/12/2018    Hematocrit 32.1 (L) 08/12/2018    Platelet Count 159 08/12/2018     Lab Results   Component Value Date    Creatinine 0.51 08/12/2018    Sodium, Serum / Plasma 136 07/29/2018    Potassium, Serum / Plasma 4.2 07/29/2018    Calcium, total, Serum / Plasma \n 8.3 (L) 07/29/2018     Lab Results   Component Value Date    Bilirubin, Total 0.4 08/12/2018    Alkaline Phosphatase 61 08/12/2018    Aspartate transaminase 28 08/12/2018    Alanine transaminase 25 08/12/2018    Albumin, Serum / Plasma 3.1 (L) 07/29/2018    Int'l Normaliz Ratio 1.1 02/28/2018     Lab Results   Component Value Date    Cancer Antigen 19-9 1,109 (H) 07/15/2018    Cancer Antigen 19-9 2,762 (H) 06/17/2018    Cancer Antigen 19-9 3,525 (H) 05/20/2018    Carcinoembryonic Antigen 3.0 08/04/2015    Carcinoembryonic Antigen 2.7 06/07/2015    Carcinoembryonic Antigen 2.5 04/04/2015     Molecular: BRCA variant of unknown significance in the setting of a family history of cancer and a personal history of breast and pancreatic cancer.  K-ras and CDKN2a and APC mutations as well.     RADIOGRAPHIC RESULTS   CT Chest abd pelvis 07/14/2018:  Unchanged size and number of bilateral pulmonary nodules consistent with treated metastases.  Compared to 05/09/2018, interval decrease in size of hypoattenuating/hypoenhancing proximal pancreatic body mass with interval decrease in associated infiltrative soft tissue. No significant interval change in associated vascular encasement and occlusion as characterized above.    Assessment    Impression: 71 y.o. female with metastatic pancreatic  cancer as summarized above.    Plan:  1. Metastatic pancreatic cancer. Responding to gem abraxane with stable disease but neuropathy.  Further reduced abraxane today.   Recheck neuropathy in 2 weeks.     CT scan on next end of September 22.new dperession  : remeron started today.     3. Nausea/vomiting  - GOO stented and stable for now.     RTC: in 2 weeks with ***** for neuropathy check    The above plan was reviewed with the patient and all questions and issues were addressed to the patient's satisfaction.      I spent a total of 25 minutes face-to-face with the patient and 25 minutes of that time was spent counseling regarding the treatment \n plan, medication risks, lifestyle modification and symptoms        \n
```

---

## Letter A

Dear [Patient’s Name],

I hope this letter finds you well. I wanted to share some important information from your recent visit with us.

**Diagnosis and Stage:**
You have metastatic pancreatic cancer. This means the cancer has spread beyond the pancreas to other parts of your body.

**Treatment Plan:**
We have been treating your cancer with different medicines. Recently, we started using gemcitabine and Abraxane. These medicines help control the cancer, but they can cause side effects like neuropathy, which is a tingling feeling in your hands and feet.

**Key Test Results:**
The latest tests show that the cancer is stable, meaning it hasn’t grown much. However, your tumor markers, which are special proteins in your blood, have gone up a bit. This is something we need to keep an eye on.

**Next Steps:**
- We will check your neuropathy again in two weeks.
- You will have a CT scan at the end of September to see how things are going.
- We started a new medicine called Remeron to help with any feelings of sadness or low mood.

**What to Watch For:**
Please let us know if you notice any changes in how you feel, especially if you have more pain, nausea, or vomiting. Also, keep an eye out for any new symptoms or if existing ones get worse.

Remember, it’s very important to talk to your care team about any questions or concerns you might have. We are here to support you through this journey.

Warm regards,

[Your Name]  
Medical Communication Specialist  
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your metastatic pancreatic cancer.
**What's new or changed since your last visit?**
Your cancer is currently stable. Imaging shows that the size of the tumors in your lungs has not changed, and the tumor in your pancreas has gotten smaller. However, your blood test for Cancer Antigen 19-9 is still high, indicating that the cancer is still active.
**What treatment or medication changes were made?**
Your dose of abraxane was reduced today due to neuropathy. You also started taking Remeron for depression.
**What is the plan going forward?**
You will continue with gemcitabine and abraxane, but with a reduced dose of abraxane. You will have a CT scan at the end of September to check on the size of the tumors. Your next visit is in 2 weeks to check on your neuropathy.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 10

**Cancer Type:** Pancreatic ductal adenocarcinoma

## Original Clinical Note

```
FOLLOW UP VISIT    Ms. ***** is a 73 y.o. female whom I am seeing as a video visit at the UCSF Comprehensive Cancer Center for follow-up of her locally advanced pancreatic cancer.    Oncologic History:  07/08/19: Presented to UCSF ED with decreased PO intake, epigastric abdominal discomfort, new dx diabetes with A1C 10.  Workup included a CT AP with contrast with showed: 3.1x2.1cm ill defined mass in pancreatic body with upstream PD dilation, tail atrophy, encasement of celiac axis, CHA, proximal splenic        artery and at least abutment of the portal splenic confluence which is moderately narrowed.  Mildly enlarged gastrohepatic and periportal LN up to 1.5cm, hepatic steatosis  08/02/19: Initial consultation with Dr. ***** who recommended and EUS with biopsy and staging CT panc protocol  09/09/19: EUS (Dr. *****, UCSF): Pancreatic tail mass measuring at least 23.6 mm by 18.2 mm with FNB results show malignancy no preliminary cytology. The mass invaded into the main pancreatic duct causing pancreatic duct dilation and splenic vein. The mass abutted the celiac artery without clear invasion and encased the SMV.   09/24/19: CT AP Pancreatic Protocol: 4.1x3.4cm hypoenhancing mass involving pancreatic body measuring 4.1x3.4cm, completely encasing celiac axis, SMA, SMV, splenic artery and splenic vein. Portacaval and pericaval LAD. No ascites.    09/24/19: CT chest: Ill defined 10x5mm nodule in RUL is nonspecific and may represent a focus of infection, inflammation or metastatic disease.  1.7 x 2.2 cm right breast soft tissue mass  09/27/19: Follow up with Dr. *****: recommended systemic therapy for locally advanced pancreatic cancer.   10/01/19: Established care with GI medical oncology: recommended FOLFIRINOX for locally adv PDAC  10/22/19: C1D1 FOLFIRINOX ( 5FU 2400mg/m2 over 48 hours, oxaliplatin 85 mg/m2,  irinotecan 180 mg/m2 on day 1, Fulphila and IVF on D3  10/27/19: Pt reported to RN that she'd been having somewhat \n bloody bm (only after eating), decreased/minimal PO intake, diarrhea with every PO intake since receiving chemotherapy.  She also reported fatigue.  RN advised that pt take Zofran ATC, pick up imodium that had been prescribed, and to increase intake of food, and to report if bloody diarrhea worsened.  10/29/19: Reported continued diarrhea, improved bleeding.  Potassium 2.8, Mg <2, Cr 1.35 and ANC 900.  Received 60 mEQ potassium with IVF, 1g Mg and improvement in K to 3.8, Cr to 0.98. She was sent an Rx for 20mEq PO potassium as well. Advised to take ATC loperamide until no diarrhea x12h  11/05/19: Infusion center for labs notable for K 3.0, Mg 1.3, Cr 1.32. WBC 35. Received 60mEq IV KCL, 1g Magnesium, 1L NS.  Diarrhea had resolved, but still with fatigue, loss of appetite  11/12/19: C2D1 FOLFIRINOX held as pt still recovering--diarrhea had resolved, but she still had poor PO intake, fatigue and preferred to wait an additional week so that she could have support at home.    11/19/19: C2D1 FOLFIRINOX (had been delayed by 1 week for pt recovery per her preference)  12/03/19: C3 FOLFIRINOX delayed per pt preference to have an extra week to recover.  She experienced continued fatigue, loss of appetite diarrhea with C2.    12/10/19: CT CAP: Stable pancreatic mass with vessel encasement, No evidence of distant metastatic disease. Colitis on scans   12/10/19: C3D1 FOLFOX (irinotecan discontinued given evidence of colitis on scans, poor tolerability)  12/24/19: C4D1 FOLFOX (omit irinotecan since C3)    Interval History:  01/07/20: C5D1 FOLFOX  (omit irinotecan since C3)    Today, she presents for follow up and same treatment with C6D1 FOLFOX  -Feels \"almost normal\"  -Notes constant saltiness in mouth which limits her appetite, but trying to find foods she likes.  Despite my caution against losing weight, she is have she has lost additional weight and aims to drop to 140lb.  -Denies diarrhea except with fruit  -Notes sensation of \n throat constriction for D1-7: when drinking room temperature fluids. Not with warm fluids or foods.  Acknowledges that her room is cold and that room temp may still not be warm  -Trying to eat potassium rich foods, would like Klor Con for potassium repletion as she tolerates the size and taste of these pills  -Continued darkening of hands, no neuropathy    Review of Systems:  Systemic: Negative for fevers, chills, or sweats.   fatigue improving, decreased appetite improving, (+) weight loss  Cardiac: Negative for chest pain or palpitations.    Respiratory: Negative for shortness of breath, dyspnea, or cough.    Abdominal: Negative for distension, pain, nausea, vomitin. Resolved bloody diarrhea   Skin: Negative for rash or jaundice.    Musculoskeletal: Negative for bone pain or swelling.   Neurologic: Negative for neuropathy, headaches, blurred vision, or confusion.  Hematologic: Negative for bruising or bleeding.    Psychiatric: Mood is stable.  Patient denies symptoms of anxiety or depression.      All other systems were reviewed and are negative.    Current Medications:  Current Outpatient Medications   Medication Sig Dispense Refill   \u0007 diphenoxylate-atropine (LOMOTIL) 2.5-0.025 mg tablet Take 1 tablet by mouth 4 (four) times daily as needed for Diarrhea 60 tablet 1   \u0007 EYEBRIGHT ORAL Take by mouth.     \u0007 gabapentin (NEURONTIN) 300 mg capsule Take 1 capsule (300 mg total) by mouth 3 (three) times daily. 90 capsule 5   \u0007 glucosam/chond/collagen/hyalur (JOINT SUPPORT ORAL) Take by mouth.     \u0007 Lactobacillus acidophilus (PROBIOTIC ORAL) Take by mouth.     \u0007 loperamide (IMODIUM) 2 mg capsule Take two capsules by mouth at the onset of diarrhea, then take one capsule every 2 hours until diarrhea-free for 12 hours. 30 capsule 5   \u0007 LORazepam (ATIVAN) 0.5 mg tablet Take 1 tablet (0.5 mg total) by mouth every 8 (eight) hours as needed (nausea) 30 tablet 3   \u0007 multivitamin tablet Take 1 tablet by mouth daily.     \u0007 \n ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet three times a day as needed for nausea and vomiting. Start 2 days after last dose of palonosetron (Aloxi) is given. 30 tablet 5   \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting) 30 tablet 5   \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 500 mg by mouth every 6 (six) hours as needed for Pain.     \u0007 lidocaine-prilocaine (EMLA) 2.5-2.5 % cream Apply cream one hour before port access as needed for pain. (Patient not taking: Reported on 10/22/2019  ) 30 g 2   \u0007 oxyCODONE (ROXICODONE) 5 mg tablet Take 1 tablet (5 mg total) by mouth every 8 (eight) hours as needed for Pain. (Patient not taking: Reported on 10/01/2019  ) 20 tablet 0   \u0007 potassium chloride (KLOR-CON 10) 10 mEq ER tablet Take 2 tablets (20 mEq total) by mouth daily 60 tablet 3     No current facility-administered medications for this visit.      Facility-Administered Medications Ordered in Other Visits   Medication Dose Route Frequency Provider Last Rate Last Dose   \u0007 0.9 % sodium chloride infusion  10-30 mL/hr Intravenous Continuous ***** *****, MD       \u0007 albuterol (PROVENTIL HFA;VENTOLIN HFA) 90 mcg/actuation inhaler 2 puff  2 puff Inhalation Once PRN ***** *****, MD       \u0007 dextrose 5% infusion  10-30 mL/hr Intravenous Continuous ***** *****, MD 30 mL/hr at 01/21/20 1220 30 mL/hr at 01/21/20 1220   \u0007 diphenhydrAMINE (BENADRYL) injection 50 mg  50 mg Intravenous Once PRN ***** *****, MD       \u0007 EPINEPHrine (EPIPEN) injection 0.3 mg  0.3 mg Intramuscular Once PRN ***** *****, MD       \u0007 fluorouraciL (ADRUCIL) 4,032 mg in sodium chloride 0.9 % 92 mL chemo infusion - for home use  2,400 mg/m2 (Treatment Plan Recorded) Intravenous Once Q46H (Pump) ***** ***** *****, *****       \u0007 heparin flush 100 unit/mL injection syringe 300 Units  300 Units Intravenous PRN ***** *****, MD       \u0007 heparin flush 100 \n unit/mL injection syringe 500 Units  500 Units Intravenous PRN ***** ***** *****, *****   500 Units at 01/21/20 0941   \u0007 heparin flush 100 unit/mL injection syringe 500 Units  500 Units Intravenous PRN ***** *****, MD       \u0007 hydrocortisone sodium succinate (SOLU-CORTEF) 100 mg/2 mL injection 100 mg  100 mg Intravenous Once PRN ***** *****, MD       \u0007 leucovorin (WELLCOVORIN) 672 mg in dextrose 5% 308 mL infusion  400 mg/m2 (Treatment Plan Recorded) Intravenous Once ***** ***** *****, *****       \u0007 LORazepam (ATIVAN) tablet 1 mg  1 mg Oral Once PRN ***** *****, MD       \u0007 oxaliplatin (ELOXATIN) 142.8 mg in dextrose 5% 579 mL chemo infusion  85 mg/m2 (Treatment Plan Recorded) Intravenous Once ***** ***** *****, *****           Allergies:  Allergies/Contraindications  No Known Allergies    Past medical, social, and family history were reviewed at today's visit and are unchanged from prior visit.      PHYSICAL EXAM  Vitals:  Vitals    01/21/20 1106   BP: 137/70   Pulse: 94   Resp: 18   Temp: 37.1 C (98.8 F)   SpO2: 97%   Weight: 64 kg (141 lb)   Height: 159 cm (5' 2.6\")   PainSc:  0         Constitutional: Normal appearance. No acute distress. Not ill-appearing. Not diaphoretic.  Head: Normocephalic, atraumatic. External ears normal. Nose normal.    Eyes: Extraocular movements intact. Conjunctivae normal. No eye discharge. No scleral icterus.  Neck: Normal range of motion.   Pulmonary/Chest: Normal effort. No respiratory distress. No cough.  GI: No abdominal distention.   Musculoskeletal: No apparent edema.   Skin: Not jaundiced. Not pale. No visible erythema. No visible rash.  Neurological: Alert and oriented to person, place and time. Able to stand from sitting and walk.   Psychiatric: Normal mood and affect. Behavior is normal. Judgment and thought content normal.   ECOG Performance Status: 1 on 01/21/20    Laboratory Results-reviewed with patient today:    Key elements of latest CBC values... Please see Chart \n Review for additional result details.  Lab Results   Component Value Date    WBC Count 30.6 (H) 01/21/2020    Hemoglobin 8.7 (L) 01/21/2020    Hematocrit 27.1 (L) 01/21/2020    MCV 99 01/21/2020    Platelet Count 192 01/21/2020     Neutrophil Absolute Count (x10E9/L)   Date Value   01/21/2020 24.06 (H)       Lab Results   Component Value Date    Albumin, Serum / Plasma 4.4 09/24/2019    Alkaline Phosphatase 204 (H) 01/21/2020    Alanine transaminase 22 01/21/2020    Aspartate transaminase 39 01/21/2020    Bilirubin, Total 0.4 01/21/2020    Urea Nitrogen, Serum / Plasma 13 01/21/2020    Calcium, total, Serum / Plasma 8.9 01/21/2020    Chloride, Serum / Plasma 104 01/21/2020    Carbon Dioxide, Total 25 01/21/2020    Anion Gap 12 01/21/2020    Creatinine 1.09 (H) 01/21/2020    eGFR if non-African American 50 01/21/2020    eGFR if African Amer 58 01/21/2020    Glucose, non-fasting 107 01/21/2020    Potassium, Serum / Plasma 2.9 (LL) 01/21/2020    Sodium, Serum / Plasma 141 01/21/2020    Protein, Total, Serum / Plasma 7.7 09/24/2019       Lab Results   Component Value Date    Int'l Normaliz Ratio 1.2 10/15/2019    PT 14.4 10/15/2019       No results found for: HCV, HCVEXL, HCVEXQ, HCVG, HCVRNAQUAL, HCVRNAQUANT, HCVEXT  Lab Results   Component Value Date    HBCAB NEG 10/15/2019    HBSAG NEG 10/15/2019     Cancer Antigen 19-9   Date Value Ref Range Status   01/07/2020 8,274 (H) <38 U/mL Final     Comment:     Performed using the ***** Architect Chemiluminescent Microparticle Immunoassay (*****).     Carcinoembryonic Antigen   Date Value Ref Range Status   12/10/2019 4.2 <5.1 ug/L Final     Comment:     Performed using the ***** Architect Chemiluminescent Microparticle Immunoassay (*****).         Imaging Results:    I personally reviewed and interpreted the patient's relevant imaging studies in conjunction with the corresponding formal radiology reports, as follows:  Ct Abdomen /pelvis With Contrast    Result Date: \n 12/10/2019  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** 8:35 AM CLINICAL HISTORY: Restaging scans. Pancreatic cancer COMPARISON:  CT abdomen/pelvis from 09/24/2019. TECHNIQUE: Following the administration of 150 cc of Omnipaque 350, contiguous 1.25-mm collimation axial images were obtained through the abdomen and pelvis. Coronal and sagittal reformats were also obtained. CONTRAST MEDIA: Intravenous and oral iodinated contrasts were administered. FINDINGS: Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Liver:  Diffuse hepatic steatosis. Heterogeneous and diminished enhancement. Mild intrahepatic biliary ductal dilatation. Gallbladder: Unremarkable Spleen:  Unremarkable Pancreas:  4.4 x 3.5 cm mass in pancreatic body (series 5, image 48), with upstream pancreatic ductal dilatation and parenchymal atrophy, similar to prior study from 09/24/2019. Adrenal Glands:  Unremarkable Kidneys:  Unremarkable GI Tract: Small hiatal hernia. Diffuse wall edema involving the ascending and transverse colon. No obstruction. Vasculature:  Encasement and obstruction of the portal confluence by the pancreatic mass, with further attenuation of right and left portal veins. Encasement of celiac axis and superior mesenteric artery, with severe stenosis of hepatic artery, splenic artery, as well as superior mesenteric artery. Lymphadenopathy: Several periportal/peripancreatic lymph nodes measuring up to 1 cm, grossly similar to prior study. Peritoneum: Mild edema and trace ascites in upper abdomen. Bladder: Unremarkable Reproductive organs: Unremarkable Bones:  Multilevel degenerative changes in thoracic and lumbar spine, with grade 1 anterolisthesis of L3 on L4 and L4 on L5. Extraperitoneal soft tissues: Substantial interval weight loss. Lines/drains/medical devices: None RADIATION DOSE INDICATORS: Exposure Events: 4 , CTDIvol Min: 12 mGy, CTDIvol Max: 13.4 mGy, \n DLP: 1012.8 mGy.cm. The following accession numbers are related to this dose report *****,*****     1. Pancreatic mass as detailed above, with encasement and severe narrowing of celiac and superior mesenteric arteries. 2. Ascending and transverse colitis. Report dictated by: ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging    Ct Chest With Contrast    Result Date: 12/10/2019  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Cancer metastatic to lung suspected No iodinated contrast contraindication Scan request unrelated to lung cancer Restaging scans. COMPARISON: 09/24/2019 TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the administration of intravenous contrast. RADIATION DOSE INDICATORS: Exposure Events: 4 , CTDIvol Min: 12 mGy, CTDIvol Max: 13.4 mGy, DLP: 1012.8 mGy.cm. The following accession numbers are related to this dose report *****,***** FINDINGS: LUNGS: Unchanged right upper lobe mixed solid groundglass nodule measuring 1.0 x 0.5 cm (series 14 image 56). Unchanged nodular scar or atelectasis adjacent to the osteophyte in the medial right lower lobe. No new nodules. PLEURA: The pleura is normal. MEDIASTINUM: No intrathoracic lymph
```

---

## Letter A

Dear Ms. *****,

Thank you for your visit today. Here is a summary of our discussion:

**Diagnosis and Stage:**
You have locally advanced pancreatic cancer. This means the cancer is growing in the pancreas and has spread to nearby blood vessels and lymph nodes.

**Treatment Plan:**
We are treating you with a chemotherapy called FOLFOX. This helps control the cancer. We stopped using a part of the treatment called irinotecan because it caused some side effects.

**Key Test Results:**
- Your blood tests show that your white blood cells are high, and your red blood cells are low. This is common with chemotherapy.
- Your potassium levels are low, so we need to make sure you get enough potassium.
- Your CA 19-9, a marker for pancreatic cancer, has gone up. This can happen even when the cancer is being controlled.

**Next Steps:**
- You will continue to receive FOLFOX chemotherapy.
- We will check your potassium levels and give you potassium supplements if needed.
- We will do scans after your eighth treatment to see how the cancer is responding.
- We will keep checking your CA 19-9 levels each month.

**What to Watch For:**
- Keep an eye on your appetite and try to eat foods you enjoy.
- If you have any new symptoms or if your diarrhea gets worse, let us know.
- Try to avoid losing more weight. Eating potassium-rich foods like bananas and potatoes can help.

**Important Reminders:**
- Continue to take your medications as prescribed.
- If you have any questions or concerns, please talk to your care team.

We are here to support you through this journey. Thank you for trusting us with your care.

Warm regards,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

Office Phone Number: [Phone Number]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
The CT scan on 12/10/19 shows a stable pancreatic mass measuring with encasement and severe narrowing of celiac and superior mesenteric arteries, without evidence of distant metastases. Additionally, you's clinical status allows continuation of treatment with a chemotherapy regimen, indicating stable disease. However, the CA 19-9 level has increased from 5000 to 11,000 U/mL, which may suggest disease activity despite stable imaging findings.
**What treatment or medication changes were made?**
You will continue with a chemotherapy regimen only going forward: oxaliplaton and 5FU. Premeds include Aloxi, aprepitant, and dexamethasone. On day 3, you will receive 1L NS, dexamethasone, prn Zofran, and ativan. D3 Fulphila is omitted given ANC >20 today. Potassium management includes + (with 4x Klor Con pills) on D1. Repeat K and Mg on D3: give + (with 4x Klor Con pills) only if K <=3.3. Hold if >3.3. Magnesium: 1g today. Rx for Klor Con: advised to take . Reviewed potassium-rich foods. Consider renal evaluation for RTA if no improvement. Loperamide available prn for diarrhea. Counseled on BRAT diet and avoiding dairy.
**What is the plan going forward?**
After 4-6 months of upfront chemotherapy, consider either (1) readdressing surgical prospects with your doctor or (2) consolidative chemoradiation, either conventionally fractionated therapy with concurrent capecitabine as a radiosensitizer, or SBRT. Plan for scans again after C8. Monthly CA 19-9 monitoring. Next visit is in-person in 2 weeks.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 11

**Cancer Type:** Pancreatic cancer

## Original Clinical Note

```
INITIAL GI MEDICAL ONCOLOGY VISIT     Patient name  ***** ***** *****      DOB 12/24/1968      Medical record number *****   Date of service 04/18/2021    Referring Provider: Dr. ***** ***** *****    Mr. ***** is a 52 y.o. male whom I am seeing as a new patient at the UCSF Comprehensive Cancer Center for his pancreatic cancer . The history was obtained from the patient, along with my review of all his available  records.    History of Present Illness     03/16/21: ***** ER with abdominal pain x1mo, 35lb weight loss, and emesis with PO intake. (Pt had been dx with hpylori in January 2021 and treated with omeprazole without improvement, outpt EGD/colo planned)  03/16/21: CT AP: Pancreatic cystic lesion >/= 3 cm without suspicious features. Additional 1 cm cystic lesion in the pancreatic body inferior to the larger cystic mass. Recommend GI consult for further management. (#pan3a) Left adrenal nodule versus retroperitoneal lymph node  03/19/21: EGD/EUS: 38mm mass in the pancreatic body, lymphadenopathy noted in body region  FNA/FNB Path:   Strata pending  03/28/21: CT Chest: a 3.7cm ill defined low density pancreatic mass and 1.4cm hypoattenuating liver mass in the inferior anterior right lobe.   03/29/21: CA 19-9: 13,737  03/29/21: Cs with surgery, *****: Recommended CT panc protocol, potential trt of borderline resectable disease   04/01/21: CT AP: 2.8x4.2cm pancreatic neck mass with PD dilatation, compresses SMV and MPV by <180 degrees, encompasses splenic vein near confluence by <180 degrees.  Completely encompasses GDA which remains patent.  1.3cm R hepatic lobe lesion and additional subtle peripherally enhancing inferior right hepatic lobe lesions (7mm and 5mm).   04/03/21: Referred by Dr. *****  to UCSF for consideration of Precision Promise trial vs consideration of mFOLFIRINOX if not trial candidate    He presents today for a second opinion and consideration of clinical trials.  The patient's present review of systems was reviewed and notable for the following:  --early satiety, nausea post prandial, loss of appetite, continued weight loss, now down 35lb  --was constipated before, now needs stool softener  --has CT AP to r/o biliary obstruction today, and will ask for repeat labs  --has noticed yellowing of eyes in the past two weeks  --energy is ok, still going to work   --no itching of skin  --recently, lower back started hurting , especially when sitting on it  --has appt with genetics at ***** tomorrow.  Of note, he has a family history of prostate, breast, stomach ca, leukemia. His brother was found to have a BRCA2 mutation in 2020.     The remainder of a full review of systems was negative.      PAST MEDICAL AND SURGICAL HISTORY    Past Medical History:   Diagnosis Date   \u0007 Borderline hypertension    \u0007 Diabetes (CMS code) March 2021    on metformin      History reviewed. No pertinent surgical history.       Medications the patient states to be taking prior to today's encounter.   Medication Sig   \u0007 acetaminophen (TYLENOL) 500 mg tablet Take 1,000 mg by mouth daily as needed for Pain   \u0007 ibuprofen (ADVIL,MOTRIN) 200 mg tablet Take 200 mg by mouth daily as needed for Pain   \u0007 lipase-protease-amylase 5,000-17,000- 24,000 unit CAPDREC Take 1 capsule by mouth 3 (three) times daily with meals   \u0007 metFORMIN (GLUCOPHAGE) 500 mg tablet Take 500 mg by mouth daily   \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 8 mg by mouth every 8 (eight) hours as needed for Nausea   \u0007 simvastatin (ZOCOR) 40 mg tablet Take 40 mg by mouth daily         Allergies/Contraindications  No Known Allergies      FAMILY HISTORY:  Family History   Problem Relation Name Age of Onset   \u0007 Breast cancer Mother     \u0007 Prostate cancer Father     \u0007 Stomach cancer Paternal Uncle     \u0007 Leukemia Other     \u0007 Prostate cancer Brother         Reviewed; no hereditary pattern of cancer noted in the family.      PERSONAL AND SOCIAL HISTORY  Social History     Socioeconomic \n History   \u0007 Marital status: Married     Spouse name: Not on file   \u0007 Number of children: Not on file   \u0007 Years of education: Not on file   \u0007 Highest education level: Not on file   Occupational History   \u0007 Not on file   Tobacco Use   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Never   \u0007 Drug use: Not on file   \u0007 Sexual activity: Not on file   Other Topics Concern   \u0007 Not on file   Social History Narrative    Works as a ***** in *****    Has two twin ***** kids.        Social Determinants of Health     Financial Resource Strain:    \u0007 Difficulty of Paying Living Expenses: Not on file   Food Insecurity:    \u0007 Worried About Running Out of Food in the Last Year: Not on file   \u0007 Ran Out of Food in the Last Year: Not on file   Transportation Needs:    \u0007 Lack of Transportation (Medical): Not on file   \u0007 Lack of Transportation (Non-Medical): Not on file       ECOG performance status 0 - Asymptomatic    PHYSICAL EXAM via video observation:   Vitals: Not performed as today's visit was conducted via video  Constitutional: Normal appearance. No acute distress. Not ill-appearing. Not diaphoretic.  Head: Normocephalic, atraumatic. External ears normal. Nose normal.    Eyes: Extraocular movements intact. Conjunctivae normal. No eye discharge. No scleral icterus.  Neck: Normal range of motion.   Pulmonary/Chest: Normal effort. No respiratory distress. No cough.  GI: No abdominal distention.   Musculoskeletal: No apparent edema.   Skin: Not jaundiced. Not pale. No visible erythema. No visible rash.  Neurological: Alert and oriented to person, place and time. Able to stand from sitting and walk.   Psychiatric: Normal mood and affect. Behavior is normal. Judgment and thought content normal.     Relevant Diagnostic Studies:    Lab results: I personally reviewed and interpreted each of the patient's relevant lab tests as outlined     03/30/21  Na 130  K 3.8   Cl \n 92  Cr 0.92  AST 28  ALT 29  Tbili 1.0  Alk phos 67    WBC 8.8  Hgb 14.4  Plt 286    03/29/21: CA 19-9: 13737    Radiology: I personally reviewed and interpreted each of the patient's relevant imaging studies in conjunction with the corresponding formal radiology reports, as outlined in HPI    03/29/21 CT AP (*****)  CT PANCREAS WITHOUT AND WITH CONTRAST     ***** HISTORY *****:   52 years old, history of pancreatic neck cancer. Evaluate for vascular involvement.     ***** TECHNIQUE *****:   CT images of the abdomen acquired without and with 130 mL Omnipaque 300 intravenous contrast. Multiphasic post-contrast images of the pancreas acquired.     CTDI: 10.15,10.88,12.69,12.68 mGy   DLP: 278.61,10.88,339.05,338.71 mGy-cm     COMPARISON: CT 03/16/2021     ***** FINDINGS *****:   5 mm right lower lobe nodule (image 16) is unchanged. There is a 2.8x 4.2 cm pancreatic neck hypodense mass with associated upstream pancreatic atrophy and pancreatic ductal dilatation.     This mass encompasses the superior mesenteric vein and main portal vein by <180 degrees. There is secondary moderate narrowing of these vessels which remain patent.      This mass encompasses the splenic vein near the confluence of the SMV by <180 degrees. There is severe secondary narrowing with the splenic vein remaining thinly patent.     This mass completely encompasses the gastroduodenal artery which remains fully patent.     The common and proper hepatic arteries are remote and from tumor. Superior mesenteric artery is also remote from tumor.     Hepatic steatosis. There is a 1.3 cm peripherally enhancing anterior right hepatic lobe lesion (image 57). There are additional subtle peripherally enhancing inferior right hepatic lobe lesions including a 7 mm lesion (image 74) and 5 mm lesion (image 70).     Bilateral adrenal thickening. The gallbladder and spleen are unremarkable.     The kidneys show symmetric enhancement. There is no hydronephrosis. No suspicious renal \n mass.     Visualized loops of small and large bowel are not dilated. No suspicious abdominal nodes.     Aortic atherosclerosis. Normal abdominal aortic diameter (<3cm).      03/28/21  CT Chest  Normal chest CT.     Abdominal findings noted incidentally on described above, including ill-defined pancreatic mass.        Pathology:    03/19/21 EUS FNA/FNB  FINAL PATHOLOGIC DIAGNOSIS   PANCREAS, BODY, MASS, FINE NEEDLE BIOPSY:   -   RARE ATYPICAL GLANDS (SEE COMMENT).       Genomics  Pending STRATA    Impression and Recommendations:  In summary, Mr. ***** is a 52 y.o. male with newly diagnosed metastatic pancreatic cancer.    We discussed the following at today's visit:  I reviewed his imaging, pathology, and staging studies to date, explaining why the mainstay of treatment should consist of systemic therapy (rather than locoregional approaches, such as surgery or radiation), with the goal of achieving as deep and durable a remission as possible, hopefully translating into significant prolongation of life, but that treatment was not expected to be curative.      There are several possible chemotherapy options for metastatic pancreatic cancer, with the choice of therapy depending on a patient's age, performance status, co-morbid conditions, and personal preference. The two current gold standards for front-line treatment, based on phase III data, include FOLFIRINOX (biweekly infusional 5-FU, leucovorin, irinotecan, and oxaliplatin) and the combination of gemcitabine plus nab-paclitaxel. Given his age and performance status, FOLFIRINOX is a reasonable first choice standard of care option.    I also discussed potential clinical trials, notably our ongoing *****#*****: Precision Promise Platform Trial For Metastatic Pancreatic Cancer. Prior to this appointment, he reviewed the sample ICF for this study.      Today, he provided consent to proceed with screening procedures:  The UCSF Experimental Subjects Bill of rights has been \n reviewed with the patient prior to signing the consent form and a copy has been given to the patient.  Alternative treatment to the patient's disease in lieu of clinical trial participation has been discussed with the patient.  Possible risks and possible side effects of participation in this clinical trial have been discussed.  The patient has been given opportunity to ask questions.  The patient has acknowledged that his/her questions were satisfactorily answered.  The requirement for the use of birth control while participating in this clinical trial was discussed.  The financial responsibilities while participating in this clinical trial were reviewed.  The patient has been given a copy of the signed consent form and signed HIPAA Authorization form.  A  Copy of the prohibited medication list has been reviewed with the  Patient.  The patient has been instructed to provide this list to any health care providers that may prescribe medication for the patient outside this clinical trials    # ? Jaundice  -will get CT AP and labs locally at ***** today      # Cancer stage at diagnosis  Cancer Staging  Pancreatic cancer metastasized to liver (CMS code)  Staging form: Exocrine Pancreas, AJCC 8th Edition  - Clinical stage from 04/18/2021: Stage IV (cT2, cN1, cM1) - Signed by ***** *****, MD on 04/18/2021            ***** ***** MD, MPH  Assistant Professor of Clinical Medicine   Gastrointestinal Oncology and ***** *****  ***** of *****, ***** *****  Office Phone Number: *****-*****-*****         Below for Billing purposes only              I spent a total of 70 minutes on this patient's care on the day of their visit excluding time spent related to any billed procedures. This time includes time spent with the patient as well as time spent documenting in the medical record, reviewing patient's records and tests, obtaining history, placing orders, communicating with other healthcare professionals, \n counseling the patient, family, or caregiver, and/or care coordination for the diagnoses above.        I performed this evaluation using real-time telehealth tools, including a live video Zoom connection between my location and the patient's location. Prior to initiating, the patient consented to perform this evaluation using telehealth tools.                     \n
```

---

## Letter A

Dear Mr. *****,

Thank you for visiting us at the UCSF Comprehensive Cancer Center. I want to share some important information about your health with you.

**Diagnosis and Stage:**
You have been diagnosed with pancreatic cancer that has spread to your liver. This is called stage IV cancer. It means the cancer has moved beyond where it started.

**Treatment Plan:**
We talked about different ways to treat your cancer. One option is a special medicine called FOLFIRINOX. This is a mix of four different medicines that can help fight the cancer. Another option is a combination of two medicines: gemcitabine and nab-paclitaxel.

We also talked about joining a research study called the Precision Promise trial. This study might offer new treatments that could help you. You agreed to join this study and we will do some tests to see if you qualify.

**Key Test Results:**
- **Imaging Tests:** We did scans of your belly and chest. These showed a growth in your pancreas and some small spots in your liver.
- **Blood Tests:** Your blood tests showed high levels of a protein called CA 19-9, which can be a sign of pancreatic cancer.

**Next Steps:**
Today, you will have another scan and some blood tests done locally. We will use these results to decide the best way to move forward with your treatment.

**What to Watch For:**
Please watch for any changes in how you feel. Some things to look out for include:
- Yellowing of your eyes (jaundice)
- Feeling very tired
- Changes in your appetite or weight

If you notice any of these, please let us know right away.

Remember, you can always talk to your care team if you have any questions or concerns. We are here to support you.

Best regards,

[Your Doctor’s Name]
Assistant Professor of Clinical Medicine
Gastrointestinal Oncology and [Specialty]

Office Phone Number: [Phone Number]

Please feel free to call us if you have any questions or need more information.

Warm regards,

[Your Doctor’s Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a second opinion regarding your newly diagnosed metastatic pancreatic cancer.
**What's new or changed since your last visit?**
You have a pancreatic neck mass measuring with associated pancreatic atrophy and pancreatic ductal dilatation. The mass affects several blood vessels but remains patent. Multiple small lesions in the right hepatic lobe suggest metastatic disease. Your CA 19-9 level is very high at 13,737, indicating advanced disease.
**What treatment or medication changes were made?**
You have agreed to participate in a clinical trial called the Precision Promise Platform Trial For Metastatic Pancreatic Cancer. The trial will involve a combination of chemotherapy drugs including a chemotherapy combination, which consists of gemcitabine, nab-paclitaxel, irinotecan, oxaliplatin, and leucovorin.
**What is the plan going forward?**
You will have a CT scan and blood tests done locally today to check for signs of jaundice. You also have an appointment scheduled with genetics at your treatment center tomorrow.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 12

**Cancer Type:** Pancreatic adenocarcinoma

## Original Clinical Note

```
I had the pleasure of seeing ***** ***** for follow up of pancreatic adenocarcinoma at our GI Medical Oncology practice at the UCSF ***** ***** Family Comprehensive Cancer Center.    Patient Identification and Oncology History  ***** ***** is a 74 y.o. male who initially presented to Dr. ***** on 10/30/15 with weight loss (8 lbs over past month), decline in appetite and difficulty in controlling his blood glucose levels.     On 11/02/15, a CT abdomen was obtained which revealed a 7.5x3.5x3.7cm mass in the tail of the pancreas with thrombosis of splenic vein and marked narrowing and encasement of the splenic artery. There was also mass effect on the proximal peroneal vein at the level of the portal confluence. Portal vein and splenic vein were noted to be narrowed but patent. Radiologist report notes that the mass abuts the superior mesenteric artery along its superior superficial margin as it traverses the pancreas. The loss of fat plane between mass and the antral region of the stomach was suspicious for Direction extension.     Subsequently, an EUS was performed on 11/08/15 which revealed the mass extending from the neck to the tail of the pancreas. Pancreatic ductal dilatation was noted. The mass noted to invade the muscularis propria of the stomach (T4 disease) and was noted to encase the splenic vessels. Due to difficulty in advancing the scope into the duodenum, further evaluation of portal vein, lymph nodes and uncinate process was not possible. However, a large lymph node measuring 2.7 cm was noted adjacent to the celiac axis. EUS staging was at least T4, N1. Cytopathology from the FNA of the pancreatic mass was consistent with adenocarcinoma.   He presented to a local urgent care and then ED on 11/12/15 with bilateral lower extremity weakness and slurred speech which lasted for approximately 20 minutes. It was noted that due to the abdominal and back pain he experienced after the biopsy, patient took marijuana  (smoked/ate) after which he was noted to have the symptoms of weakness and slurred speech.     He saw Dr. ***** on 11/14/15 who recommended systemic treatment with gemcitabine and nab-paclitaxel and referred the patient to Dr. ***** for potential surgical evaluation. During that visit on November 17, a diagnostic laparoscopy was recommended.     He was seen in our clinic on 11/21/15 when we recommended initiation of systemic therapy with mFOLFIRINOX.    12/05/15: Cycle 1, Day 1 of mFOLFIRINOX (20% dose reduction in irinotecan)    01/03/16: BLE Doppler Ultrasound- Negative for DVT    01/31/16: C5D1 - Admitted from the infusion center after developing dyspnea and chest tightness while receiving irinotecan. In the ED, he was noted to be in afib, CTA neg for PE. Discharged on 02/01/16.     05/14/16: Cycle 12 of mFOLFIRINOX     05/28/2016: Began treatment holiday    Interval History:  *****/*****/*****-*****/*****/***** Presented to the ED with worsening abdominal pain: CT scan showing tumor in pancreatic body/tail causing mass effect on the small intestine in the region of the ligament of Treitz, resulting in partial bowel obstruction. In addition, the CT imaging demonstrates carcinomatosis with peritoneal and omental implants. During this admission, hepatobiliary performed EGD with 5 overlapping stents placed across the malignant duodenal obstruction, with evidence of carcinomatosis of the jejunum as well. A CT scan demonstrated resolution of the intestinal obstruction s/p stent placement, lack of pneumobilia in left biliary system, and moderate ascites from prior. Hepatobiliary does not believe that his discomfort is related to the stents, though he may have free flow of fluid and gas from the more distal small bowel retrograde through the stented intestine as the stents cause lack of normal peristalsis. In addition, it is possible he may have some discomfort from the stent causing back pain (part of the duodenum is retroperitoneal).     08/25/2016: \n Patient had an NG tube placed    Patient's family presents today with patient simultaneously undergoing xray imaging of his small bowel as an inpatient at *****. His recent CT imaging is most notable for small bowel obstruction. His bilirubin was 3.5mg/dL on 08/24/16. His family voiced some concerns about transitioning the patient to palliative care as he reportedly is still very alert and sentient. They expressed interest in trying to transition him from an NG tube to a blended diet and attempt one final treatment regimen.     Review of Systems  General: no recent fevers or chills.  (+) weight loss  Skin: no rashes, lesions or jaundice  Eyes: no vision changes  HENT: no change in hearing, no nasal discharge or sinus tenderness. No mouth sores or sore throat.   Respiratory: no cough or shortness of breath  Cardiovascular: no chest pain, or palpitations. Positive for intermittent leg swelling  GI: no nausea, vomiting, diarrhea or constipation, no melena or BRBPR. (+) bloating  GU: no dysuria, no hematuria (+) bilirubinuria   Heme: no recent bleeding or bruising  Endocrine:  normal hair texture, no change in cold or heat intolerance  Musculoskeletal: no joint pain or back pain  Neurologic: no headaches, dizziness, falls. Positive for mild numbness or tingling in his fingertips; not interfering with fine motor activity.  Psychological: mood stable  14 systems were assessed in detail and are negative, except for those noted.    Past medical, social, and family history: Unchanged from prior visit and reviewed above in electronic record.    Allergies  No Known Allergies    Current Medications  No current facility-administered medications for this visit.   No current outpatient prescriptions on file.    Facility-Administered Medications Ordered in ***** *****:   \u0007  0.9 % sodium chloride flush injection syringe, 3 mL, Intravenous, Q8H SCH, ***** ***** *****, MD, 3 mL at 08/27/16 0540  \u0007  0.9 % sodium chloride flush \n injection syringe, 3 mL, Intravenous, PRN, ***** ***** *****, MD  \u0007  0.9 % sodium chloride infusion, 50 mL/hr, Intravenous, Continuous, ***** ***** *****, MD, Last Rate: 50 mL/hr at 08/28/16 0437, 50 mL/hr at 08/28/16 0437  \u0007  baclofen (LIORESAL) tablet 5 mg, 5 mg, Oral, TID *****, ***** *****, MD, 5 mg at 08/28/16 0629  \u0007  carvedilol (COREG) tablet 12.5 mg, 12.5 mg, Oral, *****, ***** *****, MD, 12.5 mg at 08/28/16 0921  \u0007  diltiazem (CARDIZEM CD) 24 hr capsule 120 mg, 120 mg, Oral, Daily (AM), ***** ***** *****, MD, 120 mg at 08/28/16 0921  \u0007  diphenhydrAMINE (BENADRYL) injection 12.5 mg, 12.5 mg, Intravenous, Bedtime *****, ***** *****, MD, 12.5 mg at 08/28/16 0021  \u0007  lidocaine (XYLOCAINE) 2 % viscous solution 5 mL, 5 mL, Mouth/Throat, Q3H *****, ***** ***** *****, MD  \u0007  methyl salicylate-menthol (BENGAY) ointment, , Topical, PRN, ***** ***** *****, MD  \u0007  ondansetron (ZOFRAN) injection 4 mg, 4 mg, Intravenous, Q30 Min PRN, ***** ***** *****, MD, 4 mg at 08/25/16 0143  \u0007  phenol (PHENASEPTIC) 1.4 % spray 1-3 spray, 1-3 spray, Mouth/Throat, Q2H PRN, ***** ***** *****, MD, 3 spray at 08/25/16 0640  \u0007  traMADol (ULTRAM) tablet 50 mg, 50 mg, Oral, Q6H *****, ***** *****, MD, 50 mg at 08/28/16 0309  \u0007  warfarin (COUMADIN) tablet 3 mg, 3 mg, Oral, Q PM, ***** ***** *****, MD, 3 mg at 08/27/16 2104    Physical Exam  Vital Signs:    There were no vitals taken for this visit. I have reviewed all vital signs at the time of exam.   Constitutional:  Well-developed, no acute distress, ECOG 1  Skin: No rash, lesions, diaphoresis, or jaundice.  Eyes: PERRL, EOMI, anicteric sclera.  HENT: Moist mucous membranes, oropharynx clear without exudates or thrush.   Lymph Nodes: no cervical, supraclavicular lymphadenopathy  Respiratory:  Good effort, clear to auscultation bilaterally; no wheezes, rales or rhonchi.  Cardiovascular:  Distant heart sounds. Irregular rate/rhythm but rate controlled; no murmurs, rubs, or gallops.  GI:  \n Non-distended, bowel sounds normal, soft, tenderness to palpation in the LLQ; palpable hepatic edge.  Musculoskeletal: Point tenderness on the right deltoid ligament.   Extremities: trace edema in BLE; no cyanosis  Neurologic:  Alert & oriented x 3, CN III-XII intact, ambulates w/o difficulty, December 28 BUE/BLE strength    Diagnostic Data  Labs  Results for orders placed or performed during the hospital encounter of 08/24/16   Complete Blood Count with Differential   Result Value Ref Range    WBC Count 9.6 3.4 - 10 x10E9/L    RBC Count 3.09 (L) 4.4 - 5.9 x10E12/L    Hemoglobin 9.6 (L) 13.6 - 17.5 g/dL    Hematocrit 28.5 (L) 41 - 53 %    MCV 92 80 - 100 fL    MCH 31.1 26 - 34 pg    MCHC 33.7 31 - 36 g/dL    Platelet Count 172 140 - 450 x10E9/L    Neutrophil Absolute Count 8.55 (H) 1.8 - 6.8 x10E9/L    Lymphocyte Abs Cnt 0.59 (L) 1.0 - 3.4 x10E9/L    Monocyte Abs Count 0.38 0.2 - 0.8 x10E9/L    Eosinophil Abs Ct 0.01 0.0 - 0.4 x10E9/L    Basophil Abs Count 0.01 0.0 - 0.1 x10E9/L    Imm Gran, Left Shift 0.05 <0.1 x10E9/L   Basic Metabolic Panel (Na, K, Cl, CO2, BUN, Cr, Glu, Ca)   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 10 6 - 22 mg/dL    Calcium, total, Serum / Plasma 8.1 (L) 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 101 97 - 108 mmol/L    Creatinine 0.56 (L) 0.61 - 1.24 mg/dL    eGFR if non-African American 102 mL/min    eGFR if African Amer 118 mL/min    Glucose, fasting 187 (H) 70 - 99 mg/dL    Potassium, Serum / Plasma 3.4 (L) 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma ***** ***** - 145 mmol/L    Carbon Dioxide, Total 26 22 - 32 mmol/L    Anion Gap 9 4 - 14   Venous Blood gas w/ Lactate (***** & *****) (includes Na+, K+, Ca++, Cl-, Glu, Hct, tHb)   Result Value Ref Range    Sample Type Venous     pH, Blood 7.39 7.35 - 7.45    PCO2 47 32 - 48 mm Hg    PO2 32 (LL) 83 - 108 mm Hg    Base excess 3 mmol/L    Bicarbonate 28 (H) 22 - 27 mmol/L    Oxygen Saturation 55 (L) 95 - 99 %    FIO2 Not specified 20 - 100 \n %    Sodium, whole blood ***** ***** - 146 mmol/L    Potassium, whole blood 3.4 3.4 - 4.5 mmol/L    Calcium, Ionized, whole blood 1.22 1.15 - 1.29 mmol/L    Chloride, whole blood 102 98 - 106 mmol/L    Hemoglobin, Whole Blood 9.9 (L) 13.6 - 17.5 g/dL    Hematocrit from ***** 31 (L) 45 - 65 %    Lactate, whole blood 1.1 0.0 - 2.0 mmol/L    Glucose, whole blood 190 70 - 199 mg/dL    Comments (1) no FiO2 on label or req    Aspartate Transaminase   Result Value Ref Range    Aspartate transaminase 30 17 - 42 U/L   Alanine Transaminase   Result Value Ref Range    Alanine transaminase 38 12 - 60 U/L   Bilirubin, Total   Result Value Ref Range    Bilirubin, Total 3.5 (H) 0.2 - 1.3 mg/dL   Alkaline Phosphatase   Result Value Ref Range    Alkaline Phosphatase 265 (H) 31 - 95 U/L   Lipase   Result Value Ref Range    Lipase 20 19 - 56 U/L   Magnesium, Serum / Plasma   Result Value Ref Range    Magnesium, Serum / Plasma 1.6 (L) 1.8 - 2.4 mg/dL   Comprehensive Metabolic Panel (BMP, AST, ALT, T.Bili, AlkP, TProt, Alb)   Result Value Ref Range    Albumin, Serum / Plasma 2.1 (L) 3.5 - 4.8 g/dL    Alkaline Phosphatase 246 (H) 31 - 95 U/L    Alanine transaminase 33 12 - 60 U/L    Aspartate transaminase 27 17 - 42 U/L    Bilirubin, Total 3.5 (H) 0.2 - 1.3 mg/dL    Urea Nitrogen, Serum / Plasma 9 6 - 22 mg/dL    Calcium, total, Serum / Plasma 8.2 (L) 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 102 97 - 108 mmol/L    Creatinine 0.62 0.61 - 1.24 mg/dL    eGFR if non-African American 98 mL/min    eGFR if African Amer 113 mL/min    Glucose, fasting 153 (H) 70 - 99 mg/dL    Potassium, Serum / Plasma 4.2 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma 138 135 - 145 mmol/L    Protein, Total, Serum / Plasma 5.1 (L) 6.0 - 8.4 g/dL    Carbon Dioxide, Total 28 22 - 32 mmol/L    Anion Gap 8 4 - 14   Complete Blood Count (includes Platelet Count)   Result Value Ref Range    WBC Count 7.9 3.4 - 10 x10E9/L    RBC Count 3.11 (L) 4.4 - 5.9 x10E12/L    Hemoglobin \n 9.5 (L) 13.6 - 17.5 g/dL    Hematocrit 29.2 (L) 41 - 53 %    MCV 94 80 - 100 fL    MCH 30.5 26 - 34 pg    MCHC 32.5 31 - 36 g/dL    Platelet Count 190 140 - 450 x10E9/L   Basic Metabolic Panel (Na, K, Cl, CO2, BUN, Cr, Glu, Ca)   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 7 6 - 22 mg/dL    Calcium, total, Serum / Plasma 7.8 (L) 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 104 97 - 108 mmol/L    Creatinine 0.53 (L) 0.61 - 1.24 mg/dL    eGFR if non-African American 104 mL/min    eGFR if African Amer >120 mL/min    Glucose, fasting 130 (H) 70 - 99 mg/dL    Potassium, Serum / Plasma 3.3 (L) 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma 138 135 - 145 mmol/L    Carbon Dioxide, Total 25 22 - 32 mmol/L    Anion Gap 9 4 - 14   Prothrombin Time   Result Value Ref Range    PT 30.8 (H) 11.6 - 15.8 s    Int'l Normaliz Ratio 3.0 (H) 0.9 - 1.3   Complete Blood Count (includes Platelet Count)   Result Value Ref Range    WBC Count 9.4 3.4 - 10 x10E9/L    RBC Count 3.28 (L) 4.4 - 5.9 x10E12/L    Hemoglobin 10.3 (L) 13.6 - 17.5 g/dL    Hematocrit 30.5 (L) 41 - 53 %    MCV 93 80 - 100 fL    MCH 31.4 26 - 34 pg    MCHC 33.8 31 - 36 g/dL    Platelet Count 224 140 - 450 x10E9/L   Basic Metabolic Panel (Na, K, Cl, CO2, BUN, Cr, Glu, Ca)   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 7 6 - 22 mg/dL    Calcium, total, Serum / Plasma 8.0 (L) 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 101 97 - 108 mmol/L    Creatinine 0.56 (L) 0.61 - 1.24 mg/dL    eGFR if non-African American 102 mL/min    eGFR if African Amer 118 mL/min    Glucose, fasting 136 (H) 70 - 99 mg/dL    Potassium, Serum / Plasma 2.9 (LL) 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma 140 135 - 145 mmol/L    Carbon Dioxide, Total 27 22 - 32 mmol/L    Anion Gap 12 4 - 14   Prothrombin Time   Result Value Ref Range    PT 30.0 (H) 11.6 - 15.8 s    Int'l Normaliz Ratio 2.9 (H) 0.9 - 1.3   Bilirubin, Total   Result Value Ref Range    Bilirubin, Total 3.7 (H) 0.2 - 1.3 \n mg/dL   Complete Blood Count (includes Platelet Count)   Result Value Ref Range    WBC Count 7.4 3.4 - 10 x10E9/L    RBC Count 3.17 (L) 4.4 - 5.9 x10E12/L    Hemoglobin 9.9 (L) 13.6 - 17.5 g/dL    Hematocrit 29.4 (L) 41 - 53 %    MCV 93 80 - 100 fL    MCH 31.2 26 - 34 pg    MCHC 33.7 31 - 36 g/dL    Platelet Count 185 140 - 450 x10E9/L   Basic Metabolic Panel (Na, K, Cl, CO2, BUN, Cr, Glu, Ca)   Result Value Ref Range    Urea Nitrogen, Serum / Plasma 7 6 - 22 mg/dL    Calcium, total, Serum / Plasma 7.9 (L) 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 99 97 - 108 mmol/L    Creatinine 0.53 (L) 0.61 - 1.24 mg/dL    eGFR if non-African American 104 mL/min    eGFR if African ***** >120 mL/min    Glucose, fasting 140 (H) 70 - 99 mg/dL    Potassium, Serum / Plasma 3.1 (L) 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma 138 135 - 145 mmol/L    Carbon Dioxide, Total 29 22 - 32 mmol/L    Anion Gap 10 4 - 14   Prothrombin Time   Result Value Ref Range    PT 25.5 (H) 11.6 - 15.8 s    Int'l Normaliz Ratio 2.3 (H) 0.9 - 1.3       Lab Results   Component Value Date    CANCER ANTIGEN 19-April 4720 (H) 07/30/2016    CANCER ANTIGEN 19-9 276 (H) 06/25/2016    CANCER ANTIGEN 19-9 89 (H) 05/27/2016    CANCER ANTIGEN 19-9 71 (H) 05/16/2
```

---

## Letter A

Dear Mr. *****,

I hope this letter finds you well. I wanted to share some important information about your health in a way that’s easy to understand.

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. This means there is a tumor in your pancreas. The cancer has spread to other parts of your body, which is why we call it metastatic.

**Treatment Plan:**
We have been treating your cancer with a special medicine called mFOLFIRINOX. Recently, we stopped the treatment because the cancer grew bigger and caused some problems. Now, we are focusing on making you comfortable and managing any pain or other symptoms you might have.

**Key Test Results:**
The latest tests show that the tumor in your pancreas is causing a blockage in your small intestine. This makes it hard for food to pass through normally. We put in special tubes called stents to help open the blockage, but it hasn’t worked as well as we hoped.

**Next Steps:**
Right now, we are watching closely to see how you are doing. We will check your bilirubin levels and how well you can eat. If things don’t get better, we will talk about moving to hospice care. Hospice helps people feel as comfortable as possible when they are very sick.

**What to Watch For:**
Please let us know if you have any new pain, trouble eating, or if you feel very tired. These could be signs that we need to adjust your care.

Remember, you can always ask me or any member of your care team any questions you have. We are here to support you and your family.

Warm regards,

[Your Name]
Medical Communication Specialist
UCSF ***** ***** Family Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer.
**What's new or changed since your last visit?**
Recent imaging shows that the cancer in your pancreas has grown and spread, causing a blockage in your small intestine and affecting your bile duct. Your bilirubin level is also higher, indicating liver issues.
**What treatment or medication changes were made?**
No specific medication changes were made during this visit. You continue to receive supportive medications like ondansetron and tramadol.
**What is the plan going forward?**
The focus is now on palliative care to manage symptoms and improve your quality of life. We will monitor your condition closely, including checking your bilirubin and liver function. We will also review your small bowel X-ray to see if additional stent placement is possible. You will be followed up with a phone call.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 13

**Cancer Type:** Pancreatic adenocarcinoma

## Original Clinical Note

```
  FOLLOW UP GASTROINTESTINAL MEDICAL ONCOLOGY VISIT     Patient name  ***** *****    DOB 04/05/1943    Medical record number *****   Date of service 01/31/2022    Referring Provider: Dr. ***** ***** *****        Subjective      ***** ***** is a 78 y.o. male referred by ***** ***** *****, ***** for pancreatic adenocarcinoma located in pancreatic head/unicinate       The history was obtained from the patient, along with my review of all his outside records    History of Present Illness      This pleasant man, hx of aspergillosis, treated with vori now stopped. His recent medical history includes imaging for aspergillosis which revealed a pancreatic mass.    Staging CT scan of the abdomen at that time showed nodal involvement and invasion of the duodenum and abutment of the superior mesenteric vein. partial obstruction is also noted.    *****/*****/*****-*****/*****/*****: UCSF admission for acute symptomatic anemia (s/p transfusion and iron) 2/2 presumed bleed 2/2 pancreatic uncinate adenocarconima with invasion to duodenal wall s/p EGD/EUS with biopsies that showed evidence of adenocarcinoma. After intervention he developed fever with leukocytosis that was attributed to procedure with resolution in absence of abx or further investigation.    *****/*****/*****-*****/*****/*****: Admitted at ***** after presenting to ED with sudden onset respiratory distress, found to have sepsis 2/2 acute on chronic hypoxemic respiratory failure 2/2 R-sided PNA.     04/29/21: Initial consultation with Dr. *****. Patient recommended to initiate FOLFIRINOX. But first patient was recommended to complete C Diff treatment (Vanco), and document neg c diff stool antigen, and then proceed with FOLFIRINOX.    *****/*****/*****-*****/*****/*****: Admitted at ***** for CBD obstruction, gastric outlet obstruction, duodenal obstruction s/p stents to duodenum and CBD with poor PO intake, with course now c/b pancreatitis. Pt underwent EGD/EUS/ERCP 05/06/21 with placement of biliary and duodenal \n metal stents. Newly identified development of nonocclusive thrombi within the superior mesenteric vein and its branches.     06/03/21: Patient presented for C1D1 Gem/Cape and treatment was held d/t HGB 7.5. Patient refused blood transfusion that day.   06/06/21: Patient received 1 unit PRBC    06/08/21: CT CAP completed d/t increased RUQ pain. IMPRESSION:  - Compared to 05/12/2021, unchanged size of large pancreatic head/unicinate mass. No new suspicious lesions in the abdomen or pelvis.   - Interval resolution of pancreatic inflammatory changes.  - Unchanged position of duodenal and common bile duct stents with further decrease in biliary ductal dilation.   - Suggestion of new rectal wall thickening and hyperemia in the setting of large stool burden and distention that may reflect stercoral colitis.   - Compared to 03/15/2021, slight interval increase in size of mediastinal and bilateral hilar lymph nodes which are borderline enlarged by CT size criteria. Additional apparent increase in patchy groundglass opacities and centrilobular nodules although direct comparison is somewhat limited due to differences in slice thickness between studies. These findings are favored to represent acute infection including viral infection or drug reaction and are similar to although less conspicuous than on more remote prior study from 11/10/2020.    06/09/21: C1D1 Gemcitabine + Capecitabine   06/15/21: Patient fell outside of LabCorp while walking down the stairs  06/23/21: C1D15 Gem/Cape. Labs: HGB 7.9  06/25/21: transfused 1 unit PRBC  07/07/21: C2D1 Gem/Cape   07/21/21: C2D15  08/01/21: CT CAP stable disease   08/04/21: C3D1  08/10/21: Dr.***** presented case at Pancreas tumor board. Dr ***** and Dr. ***** were present. We reviewed images. Not resectable now as SMA involved and tumor low in mesentery. Plan was to evaluate for for Rad Onc for bleeding control.     08/11/21: Consultation with Dr. *****.   Impression: locally advanced pancreas \n cancer, he is currently unresectable due to extensive vascular involvement at the root of the mesentery.  followup with Dr. ***** regarding continuing systemic regimen vs starting different regimen. Consider radiation.    08/18/21: C3D15 Gem/Cape  08/25/21: Consultation with Dr. *****, Rad Onc. Recommendations:  If they decide to proceed with radiotherapy, we discussed either 5 fraction low dose radiation purely to stop his duodenal bleeding, 5 fraction SBRT for tumor control, or 25 fraction capecitabine-radiotherapy for tumor control.  If SBRT, treatment will be given in 5 treatment sessions over the course of 2 weeks.   We discussed the potential short and long term side effects of radiotherapy, as well as the logistics of radiation, including CT simulation, radiation planning, and actual treatments.  We discussed that with fractionated therapy, we can deliver higher radiation dose to the tumor for potentially better durability of local control, but that no randomized data proves its superiority of SBRT.  At this point, ***** ***** and his daughter did not want to schedule any appointment, preferring first to discuss with Dr. *****.    Plan:  1) Patient uncertain about SBRT vs. Fractionated chemoRT vs new multiagent systemic therapy at this point.  2) No fiducial needed due to involvement of nearby visualizable vessels    09/01/21: C4D1 Gem/Cape   Had one episode of vomiting around 11/08/21.  No hematemesis.   November 22 changed the jejunal stent.   11/10/21 C6D15  Went to ED 12/02/21   XRT  12/08/21--> 12/22/21  12/22/21 ***** (ID) visit: recommended repeat CT scan at some point.   Was given Abx by a pulmonologist recently (~12/23/21)     INTERIM HISTORY  Patient presents for consideration  Fevers episodically around 12/20/21 for a few nights.   12/22/21 pulmonologist gave abx and that seems to have helped.     PAST MEDICAL AND SURGICAL HISTORY  Past Medical History:   Diagnosis Date   \u0007 Cancer (CMS code)     prostate \n cancer   \u0007 Cardiac arrest (CMS code) 2017   \u0007 Constipation     Mild   \u0007 COPD (chronic obstructive pulmonary disease) (CMS code)    \u0007 Diabetes mellitus (CMS code) October 2020    Mild   \u0007 GERD (gastroesophageal reflux disease) 2000   \u0007 Hearing loss    \u0007 Hypertension    \u0007 Osteoporosis 2018    Moderate   \u0007 Prostate cancer (CMS code)    \u0007 Psoriasis        Past Surgical History:   Procedure Laterality Date   \u0007 EYE SURGERY     \u0007 IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY)  06/01/2021    IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY) 06/01/2021 ***** ***** *****, MD RAD IR MZ   \u0007 prostate cancer surgery     \u0007 PROSTATE SURGERY         Allergies/Contraindications  No Known Allergies     Current Outpatient Medications   Medication Instructions   \u0007 acetaminophen (TYLENOL) 325 mg, Oral, Once PRN   \u0007 bisacodyL (DULCOLAX) 5 mg, Oral, Once PRN   \u0007 budesonide-glycopyr-formoterol (BREZTRI AEROSPHERE) 160-9-4.8 mcg/actuation INH 2 puffs, Inhalation, 2 times daily   \u0007 calcium carbonate-vitamin D (OSCAL D) 1,250 mg (500 mg elemental)-200 unit tablet 1 Tbsp, Oral, Every 4 Hours PRN   \u0007 CALCIUM CITRATE ORAL 1 tablet, Oral, Daily Scheduled   \u0007 capecitabine (XELODA) 500 mg tablet Take 3 tabs (1500 mg) in the morning with food, take 2 tabs (1000 mg) in the evening with food. Take for 7 days, and then do not take for 7 days.   \u0007 CREON 6,000-19,000 -30,000 unit capsule TAKE 1 CAPSULE BY MOUTH 3 TIMES DAILY WITH MEALS.   \u0007 denosumab (PROLIA) 60 mg, Subcutaneous, See Admin Instructions, Inject 60 mg subcutaneously as directed*****Clinic administered every 6 mos   \u0007 dronabinoL (MARINOL) 2.5 mg, Oral, 2 Times Daily Before Meals Scheduled, Take 1 hour before breakfast and one hour before dinner.   \u0007 famotidine (PEPCID ORAL) Oral   \u0007 GLIMEPIRIDE ORAL 5 mg, Oral, Daily At Bedtime Scheduled   \u0007 lipase-protease-amylase (CREON) 6,000-19,000 -30,000 unit capsule TAKE 1 CAPSULE BY MOUTH 3 TIMES DAILY WITH MEALS.   \u0007 melatonin 1 mg \n tablet Oral, Daily At Bedtime Scheduled   \u0007 mirtazapine (REMERON) 15 mg tablet TAKE 1 TABLET BY MOUTH EVERYDAY AT BEDTIME   \u0007 multivitamin tablet 1 tablet, Oral, Daily Scheduled, Brand:  Centrum   \u0007 ondansetron (ZOFRAN) 8 mg, Oral, Every 8 Hours PRN   \u0007 pantoprazole (PROTONIX) 40 mg EC tablet Take Pantoprazole 40 ***** tablets, twice daily. Pantoprazole 40 mg tablet 30 minutes before your first meal of the day and 30 minutes before bedtime.   \u0007 pantoprazole (PROTONIX) 40 mg, Oral, 2 times daily   \u0007 polyethylene glycol (MIRALAX) 17 g, Oral, Daily Scheduled   \u0007 prochlorperazine (COMPAZINE) 10 mg, Oral, Every 6 Hours PRN   \u0007 senna (SENOKOT) 17.2 mg, Oral, Daily At Bedtime Scheduled   \u0007 sucralfate (CARAFATE) 1 g, Oral, 4 Times Daily Before Meals and Bedtime Scheduled       FAMILY HISTORY  Family History   Problem Relation Name Age of Onset   \u0007 Hearing loss Mother     \u0007 Heart disease Mother     \u0007 High blood pressure Mother     \u0007 Hearing loss Father     \u0007 Hearing loss Sister     \u0007 Hearing loss Brother     \u0007 Pulmonary fibrosis Neg Hx       Reviewed; no hereditary pattern of cancer noted in the family    PERSONAL AND SOCIAL HISTORY  Social History     Tobacco Use   \u0007 Smoking status: Former Smoker     Packs/day: 0.50     Years: 28.00     Pack years: 14.00     Types: Cigarettes     Start date: 1990     Quit date: 2008     Years since quitting: 14.8   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Never   \u0007 Drug use: Never   \u0007 Sexual activity: Not Currently   Social History Narrative    Immigrated from ***** in 2021    Used to work in business            Objective      Vitals    Flowsheet Row Most Recent Value   Pain Score 0          Wt Readings from Last 3 Encounters:   12/22/21 54 kg (119 lb)   11/29/21 55.3 kg (122 lb)   11/28/21 55.6 kg (122 lb 9.6 oz)      ECOG performance status 2 - Symptomatic, <50% confined to bed  Gen NAD  Able to ambulate in the room without assistance   Respiratory: No \n respiratory distress, no cough, no use of accessory muscles   LE: No edema  Psych:  appropriate A&O X 3  Neck: full ROM    Review of Prior Testing  Relevant Diagnostic Studies    Lab results   I personally reviewed and interpreted each of the patient's relevant lab tests as outlined in my Assessment/Plan    Lab Results   Component Value Date    WBC Count 8.3 01/31/2022    Abs Neutrophils 3.18 01/31/2022    Hemoglobin 9.7 (L) 01/31/2022    Hematocrit 30.8 (L) 01/31/2022    Platelet Count 241 01/31/2022    Creatinine 0.73 11/24/2021    Bilirubin, Total 0.3 11/24/2021    Alkaline Phosphatase 80 11/24/2021    AST 18 11/24/2021    Alanine transaminase 11 11/24/2021    Albumin, Serum / Plasma 3.0 (L) 11/24/2021    INR 1.1 06/01/2021    Sodium, Serum / Plasma 135 11/24/2021    Potassium, Serum / Plasma 4.0 11/24/2021    Calcium, total, Serum / Plasma 9.4 11/24/2021     Lab Results   Component Value Date    Cancer Antigen 19-9 <2 11/18/2021    Cancer Antigen 19-9 <2 10/27/2021    Cancer Antigen 19-9 <2 06/06/2021       Radiology  I personally reviewed and interpreted each of the patient's relevant imaging studies in conjunction with the corresponding formal radiology reports, as outlined in my Assessment/Plan     CT Abdomen /Pelvis with and without Contrast    CT Abdomen /Pelvis with Contrast    Result Date: 01/31/2022  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** *****:***** PM CLINICAL HISTORY: Pancreatic adenocarcinoma Restaging scans. COMPARISON:  CT abdomen pelvis 11/18/2021 TECHNIQUE: CT of the abdomen and pelvis was performed. MEDICATIONS: Iohexol 350 - 100 mL - Intravenous FINDINGS: Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Liver:  Persistent mild intrahepatic and extrahepatic biliary dilation with decreased pneumobilia. Similar position of common bile duct stent. Gallbladder: Prior cholecystectomy. Similar fluid in the gallbladder fossa. \n Spleen:  Unremarkable Pancreas:  Similar size of ill-defined pancreatic mass in the uncinate process, measuring approximately 5.2 x 2.0 cm (series 5, image 72) compared to 5.2 x 2.2 cm on 11/18/2021. Adrenal Glands:  Unremarkable Kidneys:  Contrast in the bilateral collecting systems and ureters. Benign cyst in the right kidney lower pole. GI Tract:  Persistent encasement of the third portion of the duodenum by the pancreatic mass. Similar enhancing soft tissue within the stent, suspicious for tumor (series 5, image 76). Unchanged position of duodenal stent. Interval gastrojejunostomy with axios stent in place. Nondistended stomach. Diverticulosis without findings of acute diverticulitis. Vasculature:  Moderate atherosclerotic disease of the abdominal aorta and its branches. Unchanged obliteration of the superior mesenteric vein at the level of the pancreatic mass. The mass demonstrate similar abutment of the distal superior mesenteric artery. Lymphadenopathy: Similar mesenteric lymph nodes adjacent to the mass, the largest measuring 1.0 cm (series 5, image 68). Peritoneum: No ascites Bladder: Contrast layering dependently in the bladder. Reproductive organs: Unremarkable Bones:  No suspicious lesions Extraperitoneal soft tissues: Similar medium fat-containing left inguinal hernia. Lines/drains/medical devices: As above. RADIATION DOSE INDICATORS: Exposure Events: 3 , CTDIvol Min: 6.8 mGy, CTDIvol Max: 6.8 mGy, DLP: 449.1 mGy.cm. The following accession numbers are related to this dose report *****,*****     1.  Similar size of pancreatic mass centered in the uncinate process and invading the duodenum. Similar peripancreatic lymph nodes. 2.  Interval placement of gastrojejunal stent. No findings of proximal obstruction.     CT Chest with Contrast    Result Date: 01/31/2022  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Initial staging COMPARISON: CT chest dated 01/05/2022. TECHNIQUE: Serial 1.25 mm axial images \n through the chest were obtained after the administration of intravenous contrast. Iohexol 350 - 100 mL - Intravenous RADIATION DOSE INDICATORS: Exposure Events: 3 , CTDIvol Min: 6.8 mGy, CTDIvol Max: 6.8 mGy, DLP: 449.1 mGy.cm FINDINGS: Right chest wall port with tip terminating in the superior cavoatrial junction. LUNGS: Overall, slight improvement in waxing and waning multifocal peribronchovascular groundglass, now worse in the right upper lobe and improved in the left upper lobe. Bronchial wall thickening, most notably within the right lower lobe. PLEURA: No pleural effusion or pneumothorax. MEDIASTINUM: Unchanged mediastinal lymph nodes. HEART/GREAT VESSELS: The heart is normal in size without evidence of pericardial effusion. Severe coronary artery atherosclerotic calcifications. Normal caliber thoracic aorta and main pulmonary artery. BONES/SOFT TISSUES: No suspicious osseous lesions. VISIBLE ABDOMEN: Please see dedicated Abdomen and Pelvis CT report.     1.  Slight improvement in waxin
```

---

## Letter A

Dear Mr. *****,

Thank you for coming to our clinic. Here is a summary of your recent visit:

**Diagnosis:** You have a type of cancer called pancreatic adenocarcinoma. It is in a part of your pancreas called the head and has grown into the duodenum (the first part of your small intestine).

**Treatment Plan:** You have been receiving a treatment called Gem/Cape, which is a combination of two medicines. We have paused this treatment for now. We are considering another treatment called radiation therapy to help control the growth of the cancer.

**Key Test Results:**
- Your recent CT scans show that the size of the cancer has stayed the same.
- You have a stent in your duodenum and bile duct to help with blockages.
- Your blood tests show that your red blood cell count is low, so you might need a blood transfusion if it gets too low.

**Next Steps:**
- We will keep monitoring your condition closely.
- You should continue to take your medications as prescribed.
- You will need to have regular blood tests to check your red blood cell count.
- You should follow a low-residue diet to help with digestion.

**What to Watch For:**
- If you feel very tired or weak, let us know.
- If you have any pain or discomfort, tell your doctor.
- Keep track of your appetite and weight.

Please remember to talk to your care team if you have any questions or concerns. We are here to support you.

Best regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
The size of your pancreatic tumor has stayed the same, and there are no new signs of cancer spread. However, there has been an increase in bile flow into the lungs and dilation of the bile ducts.
**What treatment or medication changes were made?**
You will continue taking capecitabine for your cancer. Additionally, you will use ondansetron as needed for nausea. You will also continue taking Remeron and melatonin for insomnia and Creon for pancreatic enzyme replacement.
**What is the plan going forward?**
We will consider radiation therapy at this time. We will pause your current systemic therapy and reconsider radiation based on your preference. You will have a CT scan of your chest to monitor your condition. You will continue to have weekly blood tests to check your red blood cell count and receive a blood transfusion if your hemoglobin level drops below 8.0. A nutrition consultation is needed to address your nutritional needs. Your doctor will see you in person on February 17 and will perform a bronchoscopy in February after following cultures.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 14

**Cancer Type:** Metastatic pancreatic adenocarcinoma

## Original Clinical Note

```
VIDEO VISIT  I performed this consultation using real-time Telehealth tools, including a live video connection between my location and the patient's location. Prior to initiating the consultation, I obtained informed verbal consent to perform this consultation using Telehealth tools and answered all the questions about the Telehealth interaction.    Ms. ***** is a 69 y.o. female whom I am seeing as a video visit new patient at the UCSF Comprehensive Cancer Center in consultation for her metastatic pancreatic adenocarcinoma. The history was obtained from the patient, along with my review of all her available  records.    History of Present Illness  This pleasant Cantonese-speaking woman presented to the ***** ***** on 06/26/2021 with 2 months of intermittent epigastric and left sided abdominal pain, along with 15 lbs wt loss, bloating and nausea, acutely worsening on the day of admission. Diagnostic workup at the time included a CT scan of the abdomen/pelvis notable for a mass in the pancreaticoduodenal groove, along with multiple hepatic hypoattenuating lesions suspicious for metastases. She was discharged on oxycodone with arrangement for follow up liver biopsy by I.R., performed on 07/06/2021. This demonstrated adenocarcinoma (MMR by IHC pending).    The patient has since been re-admitted to the hospital on 07/18/2020 due to increasing abdominal pain as well as elevated LFTs/mild jaundice symptoms inc sclericterus. She underwent an ERCP this morning (07/20/2020) with placement of a 10x80 mm Evolution uncovered bare metal stent through a high-grade stricture of the distal CBD. Simultaneous EUS confirmed a 45.8x44.4mm well circumscribed hypoechoic mass in the head of the pancreas, with additional biopsies obtained.    On review of systems, the patient's family notes that she was previously independent with her ADLs. Appetite has been sluggish in the past few weeks, with a weight loss of > 20 lbs. No fevers/chills. She was \n using primarily Tylenol for pain relief at home, as the oxycodone she had been prescribed was causing some constipation.      PAST MEDICAL AND SURGICAL HISTORY  HTN  Dyslipidemia  Pre-diabetes  Lumbar stenosis/chronic low back pain with bilateral sciatica and neurogenic claudication  Cervical myelopathy      Current Outpatient Medications   Medication Instructions   \u0007 amLODIPine (NORVASC) 5 mg, Oral, Daily Scheduled   \u0007 atorvastatin (LIPITOR) 20 mg, Oral, Daily Scheduled   \u0007 famotidine (PEPCID) 20 mg, Oral, 2 Times Daily PRN   \u0007 losartan (COZAAR) 50 mg, Oral, Daily Scheduled   \u0007 metFORMIN (GLUCOPHAGE) 500 mg, Oral, Daily With Breakfast Scheduled   \u0007 OYSTER SHELL CALCIUM-VIT D3 500 mg(1,250mg) -200 unit tablet 1 tablet, Oral, 2 times daily   \u0007 polyethylene glycol (MIRALAX) 17 g, Oral, Daily PRN         Allergies/Contraindications  No Known Allergies      FAMILY HISTORY:  Reviewed; no hereditary pattern of cancer noted in the family.      PERSONAL AND SOCIAL HISTORY  Tobacco Use   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Not Currently   \u0007 Drug use: Never     Lives with husband at home in *****; 3 children.        ECOG Performance Status: 2    PHYSICAL EXAM via video observation:   Vitals: Not performed as today's visit was conducted via video  Constitutional: Ill-appearing. Not diaphoretic.  Head: Normocephalic, atraumatic. External ears normal. Nose normal.    Eyes: Extraocular movements intact. Conjunctivae normal. No eye discharge. Cannot appreciate scleral icterus.  Neck: Normal range of motion.   Pulmonary/Chest: Normal effort. No respiratory distress. No cough.  GI: No abdominal distention.   Musculoskeletal: No apparent edema.   Skin: Not jaundiced. Not pale. No visible erythema. No visible rash.  Neurological: Alert and oriented to person, place and time. Able to stand from sitting and walk.   Psychiatric: Normal mood and affect. Behavior is \n normal. Judgment and thought content normal.         Relevant Diagnostic Studies:    Most recent labs notable for the following:    Lab Results   Component Value Date    WBC Count 10.8 (H) 07/20/2021    Neutrophil Absolute Count 8.26 (H) 07/20/2021    Hemoglobin 9.9 (L) 07/20/2021    Hematocrit 32.4 (L) 07/20/2021    Platelet Count 315 07/20/2021    Creatinine 0.54 (L) 07/20/2021    Bilirubin, Total 4.5 (H) 07/20/2021    Alkaline Phosphatase 787 (H) 07/20/2021    AST 66 (H) 07/20/2021    Alanine transaminase 83 (H) 07/20/2021    Albumin, Serum / Plasma 2.6 (L) 07/20/2021    Int'l Normaliz Ratio 1.3 (H) 07/20/2021    Sodium, Serum / Plasma 136 07/20/2021    Potassium, Serum / Plasma 3.6 07/20/2021    Calcium, total, Serum / Plasma 9.3 07/20/2021       Lab Results   Component Value Date    Cancer Antigen 19-9 1,041 (H) 06/27/2021         I personally reviewed and interpreted the patient's relevant imaging studies in conjunction with the corresponding formal radiology reports, as follows:    Result Date: 07/18/2021  CT ABDOMEN/PELVIS  FINDINGS:  Visualized lung bases:  Bibasilar atelectasis    Liver:  Slightly increased size of some of the hypoattenuating/hypoenhancing lesions in the liver. Other lesions are not significant changed. Index lesions as below.    INDEX LESIONS (Restaging):   AI1:  Hepatic segment 5 lesion : 1.7 x 1.3 cm (Se/Im September 43), previously 1.7 x 1.3 cm   AI2:  Hepatic segment 6 lesion : 0.9 x 0.7 cm (Se/Im September 53), previously 0.4 x 0.3 cm   AI3:  Hepatic segment December 23 lesion : 1.0 x 0.9 cm (Se/Im September 39), previously 0.7 x 0.4 cm     Additionally, a few new small lesions, for example the lesion in hepatic segment 5 measuring up to 0.8 cm (series 3 image 47).    Gallbladder/bile ducts: Distended gallbladder measuring up to 4.7 cm in diameter, but without significant associated inflammatory changes.    New mild intrahepatic/extra hepatic biliary ductal dilatation with abrupt caliber change at the mass \n centered in the pancreaticoduodenal groove.    Spleen:  Unremarkable    Pancreas:  Increased size of the heterogeneously enhancing mass centered in the pancreaticoduodenal groove, now measuring approximately 4.5 cm x 3.7 cm x 7.5 cm (series 3 image 71, series 602 image 55). Mass extends into the distal aspects of the common bile duct (series 602 image 55). Increased associated mild pancreatic ductal dilatation.    New mild peripancreatic fat stranding and mild pancreatic parenchymal edema throughout the remaining pancreas.     Adrenal Glands:  Similar small left adrenal nodule measuring up to 1.1 cm (series 3 image 41).    Kidneys:  Similar subcentimeter hypoattenuating foci, too small to fully characterize, but likely benign.     GI Tract:  Unremarkable    Vasculature:  Similar atherosclerosis.    Lymphadenopathy: Increased size of multiple upper abdominal and mesenteric lymph nodes, some of which are centrally cystic/necrotic, for example the superior mesenteric lymph node now measuring approximately 1.5 cm x 1.2 cm) series 3 image 78) previously 1.3 cm x 0.9 cm, and the portal caval lymph node now measuring approximately 2.2 cm x 1.2 cm (series 3 image 57) previously 1.4 cm x 0.9 cm.    Peritoneum: No ascites    Bladder: Distended    Reproductive organs: Unremarkable    Bones:  Similar multilevel spondylosis. Similar degenerative changes of the hips, sacroiliac joints, and pelvis.    Extraperitoneal soft tissues: Unremarkable    Lines/drains/medical devices: None    IMPRESSION:   1. Compared to 06/26/2021, increased size of the primary malignancy centered at the pancreaticoduodenal groove. Increased size and number of hepatic and nodal metastases.    2. New mild biliary ductal and gallbladder dilatation, likely due to worsening biliary obstruction.    3. New mild pancreatic ductal dilatation, mild peripancreatic fat stranding, and mild pancreatic parenchymal edema. Overall, findings are most \n compatible with acute interstitial edematous pancreatitis from pancreatic ductal obstruction.    4. Similar small left adrenal nodule.        Result Date: 06/27/2021  CT ABDOMEN/PELVIS WITH CONTRAST   FINDINGS: Visualized lung bases:  Unremarkable Liver:  At least 6 hepatic metastatic lesions with largest in the gallbladder fossa 2.4 cm (series 8, image 38). Gallbladder: Unremarkable Spleen:  Unremarkable Pancreas: Heterogeneous relatively well-circumscribed mass in the pancreaticoduodenal groove measuring 3 3.7 x 3.4 cm (series 8, image 59). The mass likely represent exophytic pancreatic or exophytic duodenal tumor. No associated pancreatic or biliary ductal dilation. Prominent lymph nodes adjacent to the mass measuring up to 9 mm (series 8, image 57) which is nonspecific but could represent disease involvement. Adrenal Glands:  Indeterminate left adrenal nodule measuring up to 1.1 cm (series 8, image 34). Kidneys:  Likely benign renal cyst in the left inferior pole.    GI Tract:  Mass in the pancreaticoduodenal groove as described above. Nonobstructive bowel. No free air. Vasculature:  Mild to moderate atherosclerotic calcification of the abdominal aorta and its branches. Lymphadenopathy: Prominent mesenteric lymph nodes as described above. Peritoneum: No ascites Bladder: Unremarkable Reproductive organs: Unremarkable Bones:  No suspicious lesion Extraperitoneal soft tissues: Unremarkable Lines/drains/medical devices: None    1.  Heterogeneous relatively well-circumscribed mass in the pancreaticoduodenal groove without associated biliary or pancreatic ductal dilation. Differential diagnoses include neuroendocrine tumor and gastrointestinal stromal tumor. 2.  Aleast 6 hepatic hypoattenuating lesions, with imaging findings suspicious for metastases. 3.  Indeterminate small left adrenal nodule.       Result Date: 06/28/2021  CT CHEST WITH CONTRAST   FINDINGS: LUNGS: 3 mm lateral left upper lobe nodule (series \n August 270). A few scattered patchy perihilar bronchovascular groundglass opacities in both upper lobes. PLEURA: No pleural effusion. MEDIASTINUM: No mediastinal lymphadenopathy by CT criteria. HEART/GREAT VESSELS: Normal heart size. No pericardial effusion. Normal caliber of the ascending thoracic aorta. Mild coronary calcifications. BONES/SOFT TISSUES: No suspicious osseous lesion.     1. Nonspecific solid 3 mmlateral left upper lobe nodule (series 2 image 100), indeterminate. 2. Few nonspecific patchy peribronchovascular groundglass opacities in the upper lobes, possibly infectious/inflammatory.       Impression and Recommendations:  In summary, Ms. ***** is a 69 y.o. female with a newly diagnosed metastatic adenocarcinoma of pancreatic origin, causing malignant biliary obstruction for which she has just undergone endobiliary stenting. I will also request somatic tumor testing of the patient's tumor specimen, if there is adequate cellularity, through our in-house UCSF500 platform to assess for the presence of any actionable mutations or other genetic alterations that may inform future treatment options, either using already-approved agents or in the context of clinical trials.     We discussed the details of her cancer diagnosis and why the mainstay of treatment in the setting of stage IV (metastatic) disease should consist of systemic therapy (chemotherapy) with the goal of achieving as deep and durable a remission as possible, hopefully translating into significant prolongation of life, but that treatment was not expected to be curative. As such, it would be important to carefully weigh the risks/benefits of therapy and to prioritize QoL considerations in the process.     In terms of standard of care options, there are several possible chemotherapy regimens to select from for metastatic pancreatic cancer, with the choice of therapy depending on the patient's age, performance status, co-morbid conditions, and personal \n preference; phase III evidence supports either FOLFIRINOX (biweekly infusional 5-FU, leucovorin, irinotecan, and oxaliplatin) or the combination of gemcitabine plus nab-paclitaxel as appropriate front-line choices. In *****. *****'s case, at least based on assessment at today's video visit I would favor going with the combination of gemcitabine and nab-paclitaxel, although I am willing to re-evaluate/reconsider when I have the opportunity to meet her in person.    From the standpoint of clinical trial options for the first-line treatment of metastatic pancreatic cancer, we have the following here at ***** for which the patient could be considered, if interested and eligible (however, these would delay her being able to start on treatment immediately, and thus we are more likely to proceed with SoC chemotherapy):    - Our Precision Promise clinical trial platform, supported by the Pancreatic Cancer Action Network (PanCAN). In the first-line setting, this consists of randomization to either SOC chemotherapy (gemcitabine/nab-paclitaxel or FOLFIRINOX); or an experimental therapy arm that presently consists of gemcitabine/nab-paclitaxel plus pamrevlumab, an anti-CTGF (connective tissue growth factor) monoclonal antibody). Of note, this trial mandates fresh tumor tissue collection both pre- and on-treatment.    - The REVOLUTION (PICI0044) trial, sponsored by the Parker Institute for Cancer Immunotherapy (PICI). This trial also employs a platform design intended to test multiple different chemotherapy/immunotherapy combinations; at present, the one cohort open at ***** consists of the combination of gemcitabine/nab-paclitaxel together with hydroxychloroquine (an anti-autophagy agent) and two doses of low-dose ipilimumab. This trial is non-randomized; and also requires either an adequate amount of archived tumor material or a fresh pre-treatment biopsy for correlative analyses. [CURRENTLY ON HOLD TO ENROLLMENT FOR SAFETY \n ,PAUSE].    - A phase 1/1b study of TTX-030 (an anti-CD39 antibody) that is being run through our Cancer Immunotherapy Program; this includes a 1st-line pancreas-specific cohort where this agent is being evaluated in combination with gemcitabine/nab-paclitaxel.    All of the patient's questions were answered to her and her family's satisfaction. We will arrange short-term in-person follow up for both clinical and laboratory assessment (including making sure her LFTs have sufficiently normalized) to get her started on chemotherapy ASAP.      Medical decision-making:  Problems: Her active cancer represents a life-threatening illness.  Risk of complications, morbidity/mortality of patient management: High; the patient's systemic cancer therapy requires regular and intensive monitoring for potential major/life-threatening toxicities.      \n
```

---

## Letter A

Dear Ms. *****,

Thank you for joining me for your video visit today. I want to make sure you understand everything we talked about and what comes next.

**Diagnosis and Stage:**
You have a type of cancer called metastatic pancreatic adenocarcinoma. This means the cancer started in your pancreas and has spread to other parts of your body.

**Treatment Plan:**
The main treatment for your cancer will be chemotherapy. Chemotherapy uses medicines to kill cancer cells. We will start with a combination of two drugs: gemcitabine and nab-paclitaxel. These drugs help slow down the growth of cancer and can improve how you feel.

**Key Test Results:**
We looked at some tests that show how your body is working. Some of your blood tests showed that your liver is having a hard time. We also saw that you have a tumor in your pancreas and some spots in your liver that might be cancer too.

**Next Steps:**
We will set up a time for you to come to the clinic so we can check on you in person. We need to make sure your liver is working better before we start the chemotherapy. Once we see that your liver is okay, we will start the treatment right away.

**What to Watch For:**
Please let us know if you have any new pain, nausea, or if you feel very tired. Also, tell us if you notice any yellowing of your skin or eyes, which is called jaundice.

**Questions:**
If you have any questions or concerns, please talk to your care team. We are here to help you and answer anything you want to know.

Take care and I look forward to seeing you soon.

Sincerely,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a consultation regarding your metastatic pancreatic cancer (cancer that started in gland cells).
**What's new or changed since your last visit?**
Your cancer has spread to your liver. Imaging shows the primary cancer in your pancreas has grown larger, and there are new small tumors in your liver. There is also some swelling around your pancreas and bile ducts, likely due to blockage. You have lost over 20 pounds recently.
**What treatment or medication changes were made?**
You will start on a combination of gemcitabine and nab-paclitaxel, which are chemotherapy drugs. If you are interested, you could also consider joining a clinical trial that adds extra treatments to this combination.
**What is the plan going forward?**
You will have a short-term in-person follow-up visit to check your liver function tests and make sure they are normal before starting chemotherapy. We will also collect a sample of your tumor tissue for genetic testing to see if there are any specific mutations that can guide your treatment.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 15

**Cancer Type:** Moderately-differentiated pancreatic ductal adenocarcinoma with perineural invasion

## Original Clinical Note

```
REASON FOR VISIT/CC:   Chief Complaint   Patient presents with   \u0007 Follow-up     IDENTIFICATION:   64 y.o. male with resected pancreatic cancer    HISTORY OF PRESENT ILLNESS:   ***** ***** ***** presents today for evaluation of his resected pancreatic cancer. His oncologic history is reviewed here:     HISTORY OF PRESENT ILLNESS:  ***** ***** is a 62 year old male,   History notable for type II DM, w  03/19/17 he presented to ***** with 3 weeks of painless jaundice and acholic stools. He was found to have transaminitis, hyperbilirubinemia, and a mass at the pancreatic head.   03/26/17, the patient had an endoscopic US, stent placement in the stricture at the CBD, and a biopsy/brushings taken.   \"-A mass was identified in the pancreatic head. Fine needle aspiration performed with atypical cells seen but otherwise official results pending.  -Hyperechoic material consistent with sludge was visualized ***** in the gallbladder.  -There was no evidence of significant pathology in the visualized portion of the liver.\"  03/30/17, the stent was replaced in the setting of upward trending total bilirubin. Biopsy from 03/26/17 revealed adenocarcinoma, and a CT abd/pelvis and chest shows stage 3 local advancement without evidence of distant metastasis.   04/13/17: ***** ***** Surgical Eval.    04/20/17: ***** surgical eval.     06/03/17: C1D1 neoadjuvant FOLFIRINOX   10/20/17: CT scan: Stable, no e/o progression   11/05/17: Cycle 12 FOLFIRINOX (Final cycle)  01/25/18: Whipple with Dr. *****. Completed pancreatico-duodenectomy with pancreatico-jejunostomy, and per op report, mass was adherent to the portal vein. Pathology revealed 4.6cm pancreatic head adenocarcinoma with +PNI, +SM at the SMA margin with 11/46 lymph nodes positive. His postoperative course was complicated by a wound infection in his left neck incision and he was discharged to a SNF where he required 1 month of rehab and was discharged home on 03/24/18.  04/15/18: \n CT CAP reports ill defined stranding in surgical resection bed consistent with recurrence or post-surgical changes with significant portal vein anastamosis narrowing. No evidence of metastatic disease.  04/21/18: Consultation with Dr. *****. He presented TB's recommendations: Given patient's high risk of recurrence with + margins at the SMA and many LNs involved, it was recommended that patient receive XRT and then systemic therapy   05/04/18: Consultation with Dr. *****. Recommendations: Repeat PET/CT prior to start of radiotherapy. IMRT auth submitted, anticipate 28 fractions - plan for CT sim in ~2 weeks after resolution of C dificile infection. Return to clinic in 2 weeks for further discussion with same day sim.    May 2018: Patient developed ascites and has required weekly paracentesis. Ascites fluid is negative for malignancy and not KRAS mutated. Ascites likley related to CT finding of portal vein stenosis, which is likely from surgery; no evidence of thrombus.   June 2018: Portal vein stented. Improvement in Ascites.   September 08: Neg PET CT  October 08: neg abd U/s, no ascites.     accompanied by Daughter ***** ***** today.  He's doing pretty weel..   HgBA1C running >12  No pain. Wounds improved.        Review of Systems   Constitutional: Negative for chills, fever and malaise/fatigue.   HENT: Negative.    Eyes: Negative.  Negative for double vision and pain.   Respiratory: Negative for cough and shortness of breath.    Cardiovascular: Negative for palpitations, claudication, leg swelling and PND.   Gastrointestinal: Negative.  Negative for abdominal pain, blood in stool, diarrhea and vomiting.   Genitourinary: Negative.  Negative for dysuria, hematuria and urgency.   Musculoskeletal: Negative.  Negative for joint pain and neck pain.   Skin: Negative.  Negative for rash.   Neurological: Negative for tingling, speech change, focal weakness and headaches.   Psychiatric/Behavioral: Negative.  Negative for depression, \n hallucinations and substance abuse. The patient is not nervous/anxious and does not have insomnia.    All other systems reviewed and are negative.    Past Medical History:   Diagnosis Date   \u0007 Arthritis    \u0007 History of continuous positive airway pressure (CPAP) therapy 2011    pt weighed ~25 lbs more at that time; didn't tolerate CPAP   \u0007 Hyperlipidemia    \u0007 Pancreatic mass     adenocarcinoma   \u0007 Positive PPD 2015    no history of active disease; neg CXR   \u0007 Type II diabetes mellitus (HCC)       Past Surgical History:   Procedure Laterality Date   \u0007 APPENDECTOMY      May 2010   \u0007 ERCP  2018    with stent placement for CBD stricture 03/26/17, replaced stent 03/30/17   \u0007 IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY)  04/28/2017    IR PORT PLACEMENT (SITE NONSPECIFIC - ORDERABLE BY IR ONLY) 04/28/2017 ***** *****. *****, MD RAD IR MZ   \u0007 LAPAROSCOPIC APPENDECTOMY  06/08/2010    SFGH   \u0007 STENT PLACEMENT      Bile duct in pancreas: March 2017   \u0007 TONSILLECTOMY      ca. 1960-62      Current Medications       Dosage    acetaminophen (TYLENOL) 500 mg tablet Take 1,000 mg by mouth 2 (two) times daily as needed for Pain.     alpha-D-galactosidase (BEANO ORAL) Take by mouth. not taking, patient do not have supplies    blood glucose (BLOOD GLUCOSE) test strip Check blood sugar 4x daily, as directed    blood-glucose monitoring kit Check glucose daily as directed by provider    calcium carbonate (TUMS ORAL) Take 1,000 mg by mouth once as needed.    calcium citrate/vitamin D3 (CALCIUM CITRATE + D ORAL) Take by mouth Daily.    cholecalciferol, vitamin D3, 1000 UNITS TAB tablet Take 1,000 Units by mouth Daily.    clotrimazole 1 mg/mL SUSP suspension Take by mouth daily as needed. buccal    empty container (SHARPS CONTAINER) MISC Sharps Container: Use as directed to discard insulin needles and lancets with each use.    enoxaparin (LOVENOX) 80 mg/0.8 mL injection Inject 60 mg under the skin every 12 (twelve) hours.     famotidine \n (PEPCID) 10 mg tablet Take 10 mg by mouth 2 (two) times daily as needed (indigestion).    folic acid/multivit-min/lutein (CENTRUM SILVER ORAL) Take by mouth Daily.    furosemide (LASIX) 20 mg tablet Take 20 mg by mouth Daily.     glucose 4 gram chewable tablet When Blood sugar <70mg/dL, chew and swallow 20grams (5 tablets) and restest blood sugar in 15 minutes to goal blood sugar >100mg/dL    insulin aspart U-100 (NOVOLOG) 100 unit/mL injection Inject under the skin three times daily with meals and at bedtime according to your insulin sliding scale    insulin glargine (LANTUS) 100 unit/mL injection Inject 14 Units under the skin Daily. Or as directed. Note that your dose may change.    insulin syringe-needle U-100 (BD INSULIN SYRINGE ULTRA-FINE) 0.3 mL 31 gauge x August 05\" SYRINGE syringe Use four times daily as directed.    lancets lancets Check blood sugar 4x per day as directed.    lipase-protease-amylase (CREON) 24,000-76,000 -120,000 unit CAPDREC capsule Take 6 capsules by mouth 3 (three) times daily. 3 capsules with snacks    multivitamin complete chewable (FLINTSTONES COMPLETE) chewable tablet Take 1 tablet by mouth Daily.    pantoprazole (PROTONIX) 40 mg EC tablet Take 40 mg by mouth 2 (two) times daily.    rifAXIMin (XIFAXAN) 550 mg tablet Take 550 mg by mouth 2 (two) times daily.    rivaroxaban (XARELTO) 20 mg tablet Take 20 mg by mouth Daily. On hold by *****. ***** ***** / pt is on Lovenox - SFGH    Saccharomyces boulardii (FLORASTOR) 250 mg capsule Take 250 mg by mouth 2 (two) times daily.    vancomycin (VANCOCIN) 125 mg capsule Take 125 mg by mouth 4 (four) times daily.          Allergies/Contraindications   Allergen Reactions   \u0007 Ondansetron Wheezing     Felt wheezing, throat felt like it was closing up. Some coarseness in throat. Occurred in the hospital   \u0007 Oxaliplatin Other (See Comments)     40min into infusion, pt c/o numbness on the tip of his tongue. Redness on forehead noted.  Benadryl IV and \n hydrocortisone given, remaining chemo irinotecan and leucovorin completed without complication. Patient reports heaviness of tongue and numbness at back of throat.   \u0007 Sodium Phosphates Other (See Comments) and Wheezing     Patient states that 3rd dosage of saline fleet enema results in wheezing and chest tightness   \u0007 Acesulfame Other (See Comments)     Exhaustion, slept for over a day   \u0007 Monosodium Glutamate Other (See Comments)     Patient states that MSG results in redness on his lips   \u0007 Penicillins Other (See Comments)     Numbness on right thigh,minor reaction per patient. Patient reports heaviness of tongue, numbness at back of throat.  Was given ertapenem and fleet enema around this time and thinks it may be caused by this.         Social History     Socioeconomic History   \u0007 Marital status: Married     Spouse name: *****   \u0007 Number of children: None   \u0007 Years of education: None   \u0007 Highest education level: None   Occupational History   \u0007 None   Social Needs   \u0007 Financial resource strain: None   \u0007 Food insecurity:     Worry: None     Inability: None   \u0007 Transportation needs:     Medical: None     Non-medical: None   Tobacco Use   \u0007 Smoking status: Never Smoker   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: No     Comment: reports he had small amount of ETOH 04/19/17   \u0007 Drug use: No   \u0007 Sexual activity: None   Lifestyle   \u0007 Physical activity:     Days per week: None     Minutes per session: None   \u0007 Stress: None   Relationships   \u0007 Social connections:     Talks on phone: None     Gets together: None     Attends religious service: None     Active member of club or organization: None     Attends meetings of clubs or organizations: None     Relationship status: None   \u0007 Intimate partner violence:     Fear of current or ex partner: None     Emotionally abused: None     Physically abused: None     Forced sexual activity: None   Other Topics Concern   \u0007 None   Social History \n Narrative   \u0007 None        OBJECTIVE ASSESSMENT  BP 146/70  | Pulse 88  | Temp 37.3 C (99.2 F) (Temporal)  | Resp 16  | Ht 175.5 cm (5' 9.09\")  | Wt 70.9 kg (156 lb 3.2 oz)  | SpO2 99%  | BMI 23.00 kg/m     ECOG: 1   Physical Exam   Nursing note and vitals reviewed.  Constitutional: He is oriented to person, place, and time. He appears well-developed and well-nourished.   HENT:   Head: Normocephalic and atraumatic.   Nose: Nose normal.   Eyes: Conjunctivae and EOM are normal. Left eye exhibits no discharge. No scleral icterus.   Neck: Neck supple. No JVD present. No tracheal deviation present. No thyromegaly present.   Cardiovascular: Normal rate, regular rhythm and normal heart sounds.   No murmur heard.  Pulmonary/Chest: Breath sounds normal. No respiratory distress. He has no wheezes.   Abdominal: Bowel sounds are normal. He exhibits distension. He exhibits no mass. There is no tenderness. There is no rebound.   Musculoskeletal: Normal range of motion. He exhibits no edema, tenderness or deformity.   Neurological: He is alert and oriented to person, place, and time. He has normal reflexes. Coordination normal.   Skin: Skin is warm and dry. No rash noted. No pallor.   Psychiatric: He has a normal mood and affect. His behavior is normal. Thought content normal.     LABORATORY RESULTS  I have reviewed the following labs in clinic today and discussed them with the patient.  Lab Results   Component Value Date    WBC Count 6.4 09/05/2018    Neutrophil Absolute Count 4.43 09/05/2018    Hemoglobin 9.5 (L) 09/05/2018    Hematocrit 29.0 (L) 09/05/2018    Platelet Count 412 09/05/2018     Lab Results   Component Value Date    Creatinine 0.67 09/05/2018    Sodium, Serum / Plasma 123 (LL) 09/05/2018    Potassium, Serum / Plasma 4.7 09/05/2018    Calcium, total, Serum / Plasma 8.7 (L) 09/05/2018     Lab Results   Component Value Date    Bilirubin, Total 0.5 09/05/2018    Alkaline Phosphatase 665 (H) 09/05/2018    Aspartate \n transaminase 101 (H) 09/05/2018    Alanine transaminase 58 09/05/2018    Albumin, Serum / Plasma 3.3 (L) 09/05/2018    Int'l Normaliz Ratio 1.0 05/24/2018     Lab Results   Component Value Date    Cancer Antigen 19-9 4,375 (H) 08/30/2018    Cancer Antigen 19-9 48 (H) 04/14/2018    Cancer Antigen 19-9 182 (H) 10/21/2017    Carcinoembryonic Antigen 1.0 10/21/2017    Carcinoembryonic Antigen 1.2 09/23/2017    Carcinoembryonic Antigen 1.9 08/26/2017     Rads:  U/S Abd: 10/07/18    IMPRESSION:     Insufficient fluid for paracentesis.    Report dictated by: ***** *****, MD, signed by: ***** ***** *****, MD  Department of Radiology and Biomedical Imaging    Path from surgery :  Service Date: 01/25/2018  FINAL PATHOLOGIC DIAGNOSIS  A. Soft tissue, \"superior mesenteric artery margin,\" biopsy:   Adenocarcinoma; see comment.   B. Celiac lymph node, biopsy: Metastatic adenocarcinoma in one lymph  node (March 22).  C. Gallbladder, cholecystectomy: Mild chronic cholecystitis.  D. Spleen, splenectomy:   1. Spleen (564 grams) with no significant pathologic abnormality.  2. No tumor in two lymph nodes (0/2).    E. Pancreas, tail and body, distal pancreatectomy:  1. Ductal adenocarcinoma; see comment.   2. Chronic pancreatitis, extensive.   3. No tumor in twenty-six lymph nodes (0/26).       F. Pancreas, duodenum, portal vein, pancreaticoduodenectomy:  1. Ductal adenocarcinoma, moderately-differentiated, 4.6 cm, involving  duodenal wall; see comment  2. Chronic pancreatitis, extensive.   3. Pancreatic intraepithelial neoplasia, high-grade.   4. Adenocarcinoma in ten lymph nodes (January 06).    COMMENT:  PANCREAS SYNOPTIC COMMENT  - Tumor Location: Pancreatic head.  - Procedure: Pancreaticoduodenectomy.  - Size: Unifocal, 4.6 cm.  - Histologic Type: Ductal adenocarcinoma (slides F1, F4, F5, and  F8-F15).  - Histologic Grade of Invasive Carcinoma: Moderately-differentiated  (status post treatment).  - Microscopic \n Tumor Extension: Tumor invades duodenum and tunica media  of portal vein.     - Lymphovascular Invasion: Not identified.  - Perineural Invasion: Present.  - Margins:    Superior mesenteric artery margin: In the separately submitted \"SMA  margin\" (part A) invasive adenocarcinoma is present on permanent and  deeper level sections performed on block A1. Adenocarcinoma was not  present on the slides evaluated for frozen section diagnosis.     Uncinate (Retroperitoneal) Margin: For the pancreaticoduodenectomy  specimen (part F), invasive carcinoma is less than 0.1 cm from  cauterized margin on slide F5.   Bile Duct Margin: Negative for invasive carcinoma and high-grade  dysplasia, invasive carcinoma is 0.5 cm from margin.  Portal Vein Margin: Both the superior and inferior portal vein margins  are negative for invasive carcinoma. Invasive carcinoma is 0.8 cm from  the superior portal vein margin and 2.3 cm from the inferior portal vein  margin.   Pancreatic Neck Margin: For the pancreaticoduodenectomy specimen (part  F), the pancreatic neck margin is positive for invasive adenocarcinoma  on slide F1. For the distal pancreatectom
```

---

## Letter A

Dear [Patient Name],

I hope this letter finds you well. I wanted to share some important information about your recent visit and your health.

**Diagnosis and Stage:**
You have a type of cancer called pancreatic cancer. It started in the head part of your pancreas. Your doctors say it is at a stage called pT2N3. This means the cancer has spread to some nearby lymph nodes.

**Treatment Plan:**
We are watching closely because there is a chance the cancer could come back. We will check your blood tests and do imaging tests to keep track of how you are doing.

**Key Test Results:**
Some of your blood tests show that certain numbers are higher than they should be. These numbers help us know if the cancer might be coming back. We will check these again soon.

**Next Steps:**
- We will check your blood tests again this week.
- You will have a CT scan in two weeks.
- Keep taking your medicine for digestion and diabetes.
- If you notice any new symptoms or feel worried, please call us.

**What to Watch For:**
Please let us know if you have any new pain, swelling, or if you feel very tired. Also, keep an eye on your blood sugar levels and make sure they stay within the right range.

Remember, you can always talk to your care team if you have any questions or concerns. We are here to support you.

Best wishes,

[Your Name]  
Medical Communication Specialist  
[Cancer Center Name]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Your recent blood tests showed an increase in a tumor marker called CA 19-9, which suggests the possibility of cancer returning or growing. However, your latest imaging tests did not show any signs of cancer spread or growth.
**What treatment or medication changes were made?**
Your medication XARELTO (used to prevent blood clots) has been put on hold.
You will continue to take Creon to help with digestion.
**What is the plan going forward?**
You will have your CA 19-9 levels checked again this week. You will also have a CT scan in 2 weeks to monitor your condition. Your next visit will be in 2 weeks to review these results.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 16

**Cancer Type:** Pancreatic adenocarcinoma

## Original Clinical Note

```
 UCSF Cancer Center GI Medical Oncology Program  ***** ***** *****, ***** *****  ***** ***** ***** *****-*****  Phone: *****-*****-***** | Fax: *****-*****-*****     Follow-up *****  ***** : ***** *****, MD, *****    05/12/2019    I had the pleasure of seeing ***** ***** for follow up of pancreatic adenocarcinoma at our GI Medical Oncology practice at the UCSF ***** ***** Family Comprehensive Cancer Center.    Patient Identification and Oncology History  ***** ***** is a 86 y.o.  female who presents to our GI Oncology practice at ***** ***** ***** Family Comprehensive Cancer Center for follow-up of pancreatic adenocarcinoma.     -November 2018: Presented to her primary care doctor with painless jaundice and 15lb weight loss since *****.    -*****/*****/*****: labs notable for a Tbili of 15.8, AST 110, ALT 133, Alk phos 199.   -12/02/18: CT abdomen/pelvis: intra/extrahepatic biliary dilatation (CBD 22mm) with transition point in the pancreatic head.  9x7mm mass in the pancreatic head, 8mm periportal lymph node  -12/12/18: syncopized while in waiting room for planned ERCP.  Admitted for:       -CTA chest: negative for PE       -ERCP 12/13/18: (Tbili at that time 14.7) high grade obstruction in the distal bile duct.  Placement of a pigtail stent.  Brushings; pancreatic adenocarcinoma (no MMR testing)      -discharged December 16 with Tbili 12.9,  s/p IVF for orthostatic hypotension  -12/22/18: presented to ED again with syncopal episode due to orthostatic hypotension, again received IVF and was discharged home.  Labs notable for: T bili 8.0  -01/13/19: PET/CT: 1. Focal area of hypermetabolism in the pancreatic head/neck and upstream pancreatic ductal dilation with parenchymal atrophy suggestive of primary pancreatic malignancy. Primary mass is not definitely visualized, though a possible hypodense lesion corresponding to this area of hypermetabolism measures 1.1 cm.  2. Hypermetabolic peripancreatic and periportal lymph nodes concerning for metastatic \n disease. 3. Interval placement of common bile duct stent with interval improvement in moderate intrahepatic biliary ductal dilation, previously severe. Mild hypermetabolism surrounding the common bile duct may be postprocedural. Attention on follow-up. 4. Left mid kidney 1.3 cm intermediate density lesion is indeterminate and may represent a hemorrhagic cyst versus solid renal mass. Attention on follow-up.    01/31/19: Initial consultation in our clinic    03/08/19: ERCP reveals high grade obstruction of the intrapancreatic portion of the CBD with proximal dilation. A metal stent was placed across the stricture and was fully expanded across the stricture with proximal pneumobilia.    03/09/19: C1D1 Gemcitabine   03/13/19: CT CAP reveals stable disease in the liver and no evidence of metastatic disease  03/28/19: C1D8  04/04/19:  C1D15  04/18/19: C2D1 04/25/19: C2D8 held due to concern about potential dental abscess  04/26/19: Patient evaluated by her dentist; found to have periodontal inflammation and a deep pocket associated with her last left lower molar, which has been a chronic problem for her. Tooth was irrigated and rinsed with chlorhexidine. Patient to continue to use chlorhexidine rinse BID. No infection noted and ok'ed to continue on chemotherapy.    05/08/19: CT CAP reports ill-defined pancreatic mass is no longer seen. Dr. ***** recommends switching regimen to be on alternating weeks for better tolerability.     Interim History  No more gum bleeding. Her energy level is low and her appetite is low. She has not been eating as well and losing weight. She also has not been walking outside as much as before. She is trying to eat small frequent meals.     Review of Systems  As listed above per interim history. 14 systems were assessed/reviewed in detail and are negative, except for those noted.    Past medical, social, and family history: Unchanged from prior visit and reviewed above in electronic \n record.    Allergies  Allergies as of 05/12/2019  Review Complete On: 05/12/2019 By: ***** ***** *****   No Known Allergies       Current Medications  Current Outpatient Medications   Medication Sig Dispense Refill   \u0007 acetaminophen (TYLENOL EXTRA STRENGTH) 500 mg tablet Take 500 mg by mouth every 6 (six) hours as needed for Pain.     \u0007 bimatoprost (LUMIGAN) 0.01 % DROPSOLN Place 1 drop into both eyes Daily.       \u0007 calcium-vitamin D 500-125 mg-unit tablet Take 1 tablet (500 mg total) by mouth Daily.     \u0007 cholecalciferol, vitamin D3, (VITAMIN D3) 2,000 unit TAB Take 1 tablet by mouth Daily.       \u0007 dorzolamide (TRUSOPT) 2 % ophthalmic solution Place 1 drop into both eyes nightly at bedtime.      \u0007 multivitamin (THERAGRAN) per tablet Take 1 tablet by mouth.       \u0007 polyethylene glycol (MIRALAX) 17 gram packet Take 1 packet (17 g total) by mouth daily as needed (constipation). 527 g 0   \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 5 mg by mouth every 6 (six) hours as needed for Nausea.     \u0007 senna (SENOKOT) 8.6 mg tablet Take 1 tablet by mouth Daily.     \u0007 traMADol (ULTRAM) 50 mg tablet Take 1 tablet (50 mg total) by mouth Every 4 hours. 40 tablet 1   \u0007 omeprazole (PRILOSEC) 40 mg capsule Take 1 capsule (40 mg total) by mouth every morning before breakfast. (Patient not taking: Reported on 03/17/2019) 90 capsule 1     Current Facility-Administered Medications   Medication Dose Route Frequency Provider Last Rate Last Dose   \u0007 influenza vaccine (age >/= 6 months) (PF) (FLUVAVAL, FLUZONE, FLUARIX) injection 0.5 mL  0.5 mL Intramuscular Once ***** *****, MD         Physical Exam  Vital Signs:    Vitals:    05/12/19 0927   BP: 150/90  Comment: no blurred vision, headache or dizzyness.   Pulse: 94   Resp: 16   Temp: 36.8 C (98.2 F)   TempSrc: Temporal   SpO2: 99%   Weight: 40.5 kg (89 lb 3.2 oz)   Height: 155.9 cm (5' 1.38\")  Comment: March 2019@*****   PainSc:  0   Repeat BP 136/68  ECOG Performance Status: 2 - Symptomatic, \n <50% confined to bed    Constitutional:  Well-developed, no acute distress  Skin: No rash, lesions, diaphoresis, or jaundice.  Eyes: PERRL, EOMI, anicteric sclera.  HENT: Poor dentition but no gum bleeding. Moist mucous membranes, oropharynx clear without exudates or thrush  Lymph Nodes: no cervical, supraclavicular lymphadenopathy  Respiratory:  Good effort, clear to auscultation bilaterally; no wheezes, rales or rhonchi.  Cardiovascular:  Regular rate and rhythm; no murmurs, rubs, or gallops.  GI:  Non-distended, bowel sounds normal, soft, no tenderness to palpation; no hepatomegaly appreciated.  Musculoskeletal: No point tenderness at cervical, thoracic or lumbar spine  Extremities: No BLE edema or cyanosis  Neurologic:  Alert & oriented x 3, CN III-XII intact, ambulates w/o difficulty, December 12 BUE/BLE strength    Diagnostic Data  Labs  Results for orders placed or performed during the hospital encounter of 05/02/19   Alkaline Phosphatase   Result Value Ref Range    Alkaline Phosphatase 68 31 - 95 U/L   Bilirubin, Total   Result Value Ref Range    Bilirubin, Total 0.9 0.2 - 1.3 mg/dL   Alanine Transaminase   Result Value Ref Range    Alanine transaminase 20 11 - 50 U/L   Aspartate Transaminase   Result Value Ref Range    Aspartate transaminase 25 17 - 42 U/L   Creatinine, Serum / Plasma   Result Value Ref Range    Creatinine 0.46 0.44 - 1.00 mg/dL    eGFR if non-African American 90 mL/min    eGFR if African Amer 104 mL/min   Complete Blood Count with Differential   Result Value Ref Range    WBC Count 9.1 3.4 - 10 x10E9/L    RBC Count 4.15 4.0 - 5.2 x10E12/L    Hemoglobin 13.0 12.0 - 15.5 g/dL    Hematocrit 37.5 36 - 46 %    MCV 90 80 - 100 fL    MCH 31.3 26 - 34 pg    MCHC 34.7 31 - 36 g/dL    Platelet Count 248 140 - 450 x10E9/L    Neutrophil Absolute Count 3.79 1.8 - 6.8 x10E9/L    Lymphocyte Abs Cnt 4.36 (H) 1.0 - 3.4 x10E9/L    Monocyte Abs Count 0.88 (H) 0.2 - 0.8 x10E9/L    Eosinophil Abs Ct 0.03 0.0 - 0.4 \n x10E9/L    Basophil Abs Count 0.03 0.0 - 0.1 x10E9/L    Imm Gran, Left Shift 0.02 <0.1 x10E9/L     Imaging  Ct Abdomen /pelvis With Contrast    Result Date: 05/08/2019  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** 9:00 AM CLINICAL HISTORY: pt with pancreatic cancer currently on treatment. needs restaging scan COMPARISON:  CT abdomen pelvis 03/13/2019 TECHNIQUE: Following the administration of 150 cc of Omnipaque 350, contiguous 1.25-mm collimation axial images were obtained through the abdomen and pelvis. Coronal and sagittal reformats were also obtained. FINDINGS: Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Liver:  Mild intrahepatic biliary ductal dilatation and pneumobilia, stable. Gallbladder: Unchanged position of biliary stent. Gallbladder is unremarkable. Spleen:  Unremarkable Pancreas: Slight improvement in pancreatic ductal dilatation, currently measuring 9 mm (series 14 image 37), previously measuring 12 mm. Ill-defined pancreatic mass not clearly seen. Unchanged appearance of bilobed hypoattenuating cystic lesion in the anterior pancreatic head measuring 8 mm which can represent a sidebranch IPMN (series 14 image 50). Adrenal Glands:  Unremarkable Kidneys:  Similar appearance of left kidney interpolar exophytic 1.1 x 1 cm lesion which is hypodense on noncontrast imaging and hyperenhancing on postcontrast images (series 14 image 31). Right kidney interpole simple cyst measuring 1.5 x 1.1 cm, unchanged. GI Tract:  Diverticulosis of the sigmoid colon. Vasculature:  Stable in size mildly prominent subcentimeter periportal and peripancreatic lymph nodes measuring up to 0.7 cm in short axis. Lymphadenopathy: Absent Peritoneum: Unchanged pelvic free fluid. Bladder: Unremarkable Reproductive organs: Unremarkable Bones:  No suspicious lesions Extraperitoneal soft tissues: Unremarkable Lines/drains/medical devices: Biliary stent as \n above. RADIATION DOSE INDICATORS: Exposure Events: 7 , CTDIvol Max: 5.1 mGy, DLP: 628.9 mGy.cm. The following accession numbers are related to this dose report *****,*****,*****     1. Compared to 03/13/2019, the ill-defined pancreatic mass is no longer seen, likely related to treatment. Interval decrease in pancreatic duct dilation. No evidence of metastatic disease. 2. 1.1 cm left renal enhancing lesion concerning for renal cell carcinoma. Report dictated by: ***** *****, MD, signed by: ***** *****-***** *****, MD Department of Radiology and Biomedical Imaging    Ct Chest With Contrast    Result Date: 05/08/2019  CT CHEST WITH CONTRAST CLINICAL HISTORY:  pt with pancreatic cancer currently on treatment. needs restaging scan pt with pancreatic cancer currently on treatment. needs restaging scan COMPARISON: Multiple priors, most recent of 12/26/2018 TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the administration of intravenous contrast. RADIATION DOSE INDICATORS: Exposure Events: 7 , CTDIvol Min: 3.5 mGy, CTDIvol Max: 9.3 mGy, DLP: 628.9 mGy.cm. The following accession numbers are related to this dose report *****,*****,***** *****: LUNGS: No new lung nodules. A few scattered sub-3 mm solid nodules are unchanged compared to multiple remote priors. 16mm mixed solid and groundglass nodule in the left upper lobe (image 84, series 20) with a 3 mm soft tissue component, and adjacent 7 mm groundglass nodule in the left upper lobe (image 81) are unchanged. PLEURA: The pleura is normal. MEDIASTINUM: No intrathoracic lymphadenopathy. HEART/GREAT VESSELS: Unchanged heart size. No pericardial effusion. Thoracic aorta and pulmonary artery are unchanged caliber. BONES/SOFT TISSUES: No new bone or soft tissue lesions in the chest. VISIBLE ABDOMEN: Please see dedicated Abdomen and Pelvis CT report.     1. No new findings to suggest intrathoracic metastatic disease. 2. 2 subsolid left \n upper lobe nodules likely represent primary lung adenocarcinoma lesions as previously described. The largest nodule is 16 mm with a 3 mm solid component. Report dictated by: ***** ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging    Cancer Staging  Pancreatic cancer (HCC)  Staging form: Exocrine Pancreas, AJCC 8th Edition  - Clinical stage from 02/02/2019: Stage IIB (cT1c, cN1, cM0) - Signed by ***** *****, MD on 02/02/2019    ASSESSMENT & PLAN  In summary, ***** ***** is a 86 y.o. female who presents to our GI Oncology  practice at the UCSF ***** ***** Family Comprehensive Cancer Center for a follow-up visit of pancreatic adenocarcinoma.     Given her age, comorbidities and preference, she is not a candidate for surgical resection.  As such, the mainstay of treatment for her would be systemic therapy, given with a goal of palliating any symptoms she may have, and achieving as deep and durable a remission as possible, hopefully translating into significant prolongation of life, but this treatment was not expected to be curative.     Patient started dose reduced gemcitabine monotherapy on 03/09/19. She presents for consideration of C2D15 today. This was delayed from last week due to concern for potential dental abscess. Since then, she has been evaluated by her dentist and cleared for chemotherapy today.      Her scans obtained on 05/08/19 showed that the ill-defined pancreatic mass is no longer seen, likely related to treatment. Interval decrease in pancreatic duct dilation. No evidence of metastatic disease. Of note, there was a 1.1 cm left renal enhancing lesion concerning for renal cell carcinoma. 2 subsolid left upper lobe nodules likely represent primary lung adenocarcinoma lesions. The largest nodule is 16 mm with a 3 mm solid component. I reviewed the results of these scans with the patient and her family today. I informed them that due to the agressiveness \n of the pancreatic cancer, our focus will continue to be treating that. We will monitor the renal lesion and pulmonary nodules closely.    We will continue with gemcitabine monotherapy but we will change to alternating week schedule for better tolerability.    - Plan to continue with monthly CA 19-9 levels and restaging CT cap with contrast every 2 months to monitor treatment response. Her next set of scans will be due after this cycle.     #Left gum pain and bleeding: cleared by her dentist as no sign of infection.   - We will defer to her dentist regarding continued monitoring; pt to continue to chlorhexidine rinses bid.    #Supportive care: Influenza vaccine ordered today (to be given at infusion center). We will refer to nutritionist. She is taking sennokot 4 times per week. I recommended that she take on a daily basis. She has supply of antiemetics and takes as needed. I recommended maintaining her weight and activity level.     #Follow-up: on 05/16/19 with ***** *****. As always, the patient was counseled to contact the clinic or the after-hours physician on call for any new symptoms or complications in the interim between now and her follow up visit with me.    I spent a total of 40minutes face-to-face with the patient and >50% of
```

---

## Letter A

Dear [Patient's Name],

Thank you for coming to our clinic for your follow-up visit. I want to share some important information about your health and the next steps in your care.

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. It is in stage IIB, which means it has spread to nearby lymph nodes but not to other parts of the body.

**Treatment Plan:**
Because of your age and other health conditions, surgery is not a good option for you. Instead, we are using a medicine called gemcitabine to help control the cancer and make you feel better. We recently changed the schedule to give you the medicine every two weeks instead of weekly to make it easier on your body.

**Key Test Results:**
Recent tests show that the tumor in your pancreas has gotten smaller, which is a good sign. There are also some small growths in your lungs and kidney that we will keep an eye on, but right now, they do not seem to be causing any problems.

**Next Steps:**
We will continue giving you the gemcitabine medicine. We will also check your blood and do scans every two months to see how the treatment is working. Your next set of scans will be after this cycle of treatment.

**What to Watch For:**
Please let us know if you notice any new symptoms or if anything changes. It’s important to stay in touch with your dentist and keep using the special mouth rinse to prevent infections.

**Supportive Care:**
We ordered a flu shot for you to protect against the flu. We also recommend that you talk to a nutritionist to help you eat better and maintain your weight. Keep taking your medicine for nausea as needed.

**Follow-Up:**
Your next appointment is on May 16th with Dr. [Doctor's Name]. If you have any questions or concerns before then, please don’t hesitate to call the clinic.

Remember, you are not alone in this journey. Our team is here to support you every step of the way.

Warm regards,

[Doctor's Name], MD  
UCSF [Family Name] Family Comprehensive Cancer Center

Please discuss any questions you have with your care team.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Recent imaging shows slight improvement in the size of your pancreatic duct. The ill-defined pancreatic mass is no longer visible, likely due to treatment. There is a 1.1 cm left renal lesion that may be cancerous. No new signs of cancer spread were found.
**What treatment or medication changes were made?**
The plan is to continue with gemcitabine but switch to an alternating week schedule for better tolerability. You will continue to use medications for nausea and pain as needed.
**What is the plan going forward?**
You will have monthly CA 19-9 blood tests and restaging CT scans every 2 months to monitor how well the treatment is working. You are being referred to a nutritionist for dietary advice. Your next visit is scheduled for May 16th.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 17

**Cancer Type:** Pancreatic adenocarcinoma

## Original Clinical Note

```
This is an independent visit     Patient ID:  ***** ***** is a 58 y.o. male with locally advanced pancreatic adenocarcinoma.    ATTENDING: ***** *****, MD     Primary Care Provider:  ***** ***** *****, MD    HISTORY OF PRESENT ILLNESS:   October 2017: abd pain. Seen in ED sent home.     A first endoscopy showed no abnormalities in the stomach or duodenum.    Ultrasound showed pancreatic tail lesion.     6//18: CT scan performed in December 2017 showed locally advanced pancreatic cancer.Pancreatic body/tail mass measuring 4.8 x 2.5 x 2.8 cm.    He had an endoscopic ultrasound guided biopsy of the pancreatic tail which showed adenocarcinoma. pMMR.    01/02/18: Initial consultation with Dr. *****. Patient recommended to start chemotherapy but patient decided to avoid any active cancer treatment instead chose to focus on nutritional optimization first.    04/17/18: CT C/A/P. Slight interval increase in size of locally advanced pancreatic adenocarcinoma. Tracks further along SMA and mesenteric root, causing new focal occlusion of SMV-portal venous confluence.  04/17/18 CA 19-9 72,710    07/03/18: CT C/A/P: Preliminary read: Questionable slightly increased size of a locally advanced pancreatic mass with similar vascular involvement and unchanged thrombosis of the SMV-portal vein confluence and splenic vein. No intrathoracic metastatic disease.     Following the 07/03/18 CT scan patiently initially agreed to start chemotherapy but later decided to postpone treatment for some time. He eventually decided to start treatment.     07/31/18: C1D1 FOLFIRINOX   08/14/18: C2D1  08/28/18: C3D1  09/11/18: C4D1  09/22/18: CT CAP slight interval decrease in size of mass centered in the pancreatic body/tail  09/25/18: C5D1- Irinotecan dose reduced d/t fatigue  10/09/18: C6D1  10/23/18: C7D1  11/06/18 C8D1 held irinotecan   11/17/18: CT showed stable disease. Patient decided to initiate a chemotherapy break.   02/02/19: CT reveals stable disease. \n     Interval history   Patient has felt very well. He is exercising and eating well. Continues to have intermittent upper abdominal pain, not increasing.    Review of Systems   Constitutional: Negative for chills, diaphoresis, fatigue, fever and unexpected weight change.   HENT:   Negative for lump/mass, sore throat and trouble swallowing.    Eyes: Negative for icterus.   Respiratory: Negative for hemoptysis and wheezing.    Cardiovascular: Negative for leg swelling.   Gastrointestinal: Positive for abdominal pain. Negative for blood in stool, constipation, diarrhea, nausea and vomiting.   Genitourinary: Negative for frequency, hematuria and pelvic pain.    Musculoskeletal: Negative.  Negative for back pain, myalgias and neck stiffness.   Skin: Negative for itching and rash.   Neurological: Negative for dizziness, extremity weakness and numbness.   Hematological: Negative.  Does not bruise/bleed easily.   Psychiatric/Behavioral: Negative.  Negative for depression and sleep disturbance.   All other systems reviewed and are negative.     No history exists.     Patient's allergies, medications, past medical, surgical, family and social histories were reviewed and no interval updates necessary or noted.      Objective   Physical Exam:  Vital Signs for this encounter:  BSA: 1.95 meters squared  Vitals:    05/07/19 0939   BP: 143/71  Comment: ALL VST FROM *****   Pulse: (!) 49  Comment: NO ABNORMAL SYMPTOMS   Resp: 18   Temp: 37.6 C (99.7 F)   TempSrc: Temporal   SpO2: 100%   Weight: 77.8 kg (171 lb 9.6 oz)   Height: 175.4 cm (5' 9.06\")  Comment: December 01@*****   PainSc:   5   PainLoc: Abdomen  Comment: STOMACH, PANCREAS AREA       Performance Status:  ECOG: 0     Physical Exam  Vitals signs and nursing note reviewed.   Constitutional:       General: He is not in acute distress.     Appearance: He is well-developed. He is not diaphoretic.      Comments: +Thin   HENT:      Head: Normocephalic and atraumatic.      Right Ear: \n External ear normal.      Left Ear: External ear normal.   Eyes:      General: No scleral icterus.        Left eye: No discharge.      Pupils: Pupils are equal, round, and reactive to light.   Neck:      Musculoskeletal: Neck supple.      Vascular: No JVD.      Trachea: No tracheal deviation.   Cardiovascular:      Rate and Rhythm: Normal rate and regular rhythm.      Heart sounds: Normal heart sounds. No murmur. No friction rub.   Pulmonary:      Effort: Pulmonary effort is normal. No respiratory distress.      Breath sounds: Normal breath sounds. No stridor. No wheezing or rales.   Abdominal:      General: Bowel sounds are normal.      Palpations: Abdomen is soft. There is no mass.      Tenderness: There is no tenderness. There is no guarding.      Hernia: No hernia is present.   Genitourinary:     Comments: Deferred per patient preference  Musculoskeletal: Normal range of motion.         General: No tenderness or deformity.   Skin:     Findings: No rash.   Neurological:      Mental Status: He is alert and oriented to person, place, and time.      Cranial Nerves: No cranial nerve deficit.      Motor: No abnormal muscle tone.      Deep Tendon Reflexes: Reflexes normal.         Lab Results   Component Value Date    WBC Count 6.0 05/07/2019    Hemoglobin 13.8 05/07/2019    Hematocrit 40.4 (L) 05/07/2019    Platelet Count 151 05/07/2019    Neutrophil Absolute Count 3.74 05/07/2019    Alanine transaminase 39 05/07/2019    Aspartate transaminase 44 (H) 05/07/2019    Alkaline Phosphatase 80 05/07/2019    Bilirubin, Total 1.2 05/07/2019    Int'l Normaliz Ratio 1.1 07/25/2018    Carcinoembryonic Antigen 15.1 (H) 12/23/2018    Carcinoembryonic Antigen 33.9 (H) 11/06/2018    Carcinoembryonic Antigen 37.6 (H) 10/23/2018    Cancer Antigen 19-9 289 (H) 12/23/2018    Cancer Antigen 19-9 2,878 (H) 11/06/2018    Cancer Antigen 19-9 6,428 (H) 10/23/2018     Molecular testing:  Foundation Medicine results:   MS-Stable  Tumor \n mutational burden = 5 Muts/Mb  KRAS G12V  TP53 I195F  VUS:   CBL P782L  CDKN2A/B loss  MTOR  R2109  NKX2-1 G239_G241del  PDGFRB  V523M  PIK3C2G  T488A  TNFAIP3  P752S    Imaging Results:  No recent imaging reviewed     Assessment/Plan    ***** ***** is a 58 y.o. male with locally advanced pMMR pancreatic cancer.    Patient is s/p 8 cycles of FOLFIRINOX and now on chemotherapy break since 11/06/18. He continues to do clinically well. Patient prefers to hold off on any additional imaging for now and agrees to return to see me in June. Agrees to complete a new CT CAP in July 2019.    RTC: June 2019    ***** spent a total of 25 minutes face-to-face with the patient and >50% of that time was spent counseling regarding the symptoms, treatment plan, and therapeutic options for the diagnoses above.    ***** *****, NP                \n
```

---

## Letter A

Dear [Patient Name],

I hope this letter finds you well. I wanted to share some important information from your recent visit with us.

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. It is in a stage we call "locally advanced." This means the cancer is growing in the pancreas but has not spread to other parts of your body yet.

**Treatment Plan:**
You have completed 8 cycles of a treatment called FOLFIRINOX. This is a mix of different medicines used to fight cancer. You are now taking a break from chemotherapy. We will check on how you are doing in the future.

**Key Test Results:**
The tests show that the cancer is not growing bigger right now. Your blood tests also look good. You have some mild pain in your upper belly, but it is not getting worse.

**Next Steps:**
We want to see you again in June. At that time, we will do another scan to check on the cancer. If you have any questions or concerns before then, please let us know.

**What to Watch For:**
Keep an eye on any changes in your health. If you notice anything new or if your pain gets worse, please tell us right away.

Remember, you can always talk to your care team about any questions or worries you might have. We are here to help you through this journey.

Warm regards,

[Your Name]  
Medical Communication Specialist  
[Contact Information]

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a follow-up visit regarding your pancreatic cancer treatment.
**What's new or changed since your last visit?**
Imaging results from February 2nd show that your cancer has stayed the same size. Recent lab results show that your Carcinoembryonic Antigen (a tumor marker) and Cancer Antigen 19-9 (another tumor marker) levels are still high but have decreased compared to previous readings. Physically, you are stable with no new issues.
**What treatment or medication changes were made?**
No changes were made to your treatment or medications.
**What is the plan going forward?**
You will have a new CT scan in July 2019 to check on your cancer again. Your next visit is scheduled for June 2019.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 18

**Cancer Type:** Moderately differentiated adenocarcinoma of the tail of the pancreas

## Original Clinical Note

```
  SUBJECTIVE:  This is a very pleasant 65-year-old woman from ***** who was admitted to the hospital recently for pneumonia. An abdominal CT scan was done during her admission with an incidental finding of a pancreatic mass in the tail of the pancreas. Fine needle aspiration of the mass by EUS showed adenocarcinoma. She has subsequently had a pancreas protocol CT scan at *****. This did not show any evidence of metastatic disease. The mass in the distal necrotic body measures 3.9 cm and is localized to the pancreas. She's been seen by GI surgical oncology at *****.    She has been presented at tumor Board. Although she was  technically resectable, it was felt that she might benefit from neoadjuvant therapy. She was started on gemcitabine and Abraxane. She only received 2 doses. With each administration, she developed severe adverse events including metabolic abnormalities with hypercalcemia and hyponatremia as well as total body aches. She was hospitalized each time. With that reason, we elected to stop preoperative therapy and refer her for surgery.    On 03/01/22, she underwent a distal pancreatectomy and splenectomy. She was found to have a moderately differentiated adenocarcinoma with negative margins and 2 of 29 lymph nodes were positive. Her postoperative course was complicated by an infection and abscess which required ERCP for drainage.    We recommended postoperative gemcitabine and capecitabine using an alternate week schedule. She has had 2 cycles. She has tolerated it quite poorly. She's had mucositis and severe hand-foot syndrome with cracking and blistering. She has refused dose modification. She's also been falling and is not quite sure why. She does not describe vertigo but seemed to lose her balance. She has now using a walker.    She is very fatigued and sleeps about 12 hours in a 24 hour timeframe. She feels that she can only handle one event a day. She can walk a few blocks. She has \n musculoskeletal pain and takes Tylenol, Flexeril and gabapentin. Fortunately, her appetite is good and she has not lost any weight. She is having one to 2 loose bowel movements a day.    She has had some genetic testing in the past and is known to carry an ATM VUS.    PAST MEDICAL HISTORY:       MEDICAL ILLNESSES:  Hypertension  history of hypothyroidism  asthma, requiring frequent intervention  depression  psoriatic arthritis  history of alcohol abuse  restless leg syndrome  GERD  history of Barrett's esophagus  possible stroke, 2021  MRSA infection right elbow, 2010    PRIOR SURGERIES:  Metatarsal fracture requiring surgery and placement of hardware, 2010  right total knee replacement complicated by MRSA infection  left clavicle fracture requiring surgery, 2001  C-section, 1995    INJURIES:  Broken clavicle following accident with ***** bike    CURRENT MEDICATIONS:  Current Outpatient Medications on File Prior to Visit   Medication Sig Dispense Refill   \u0007 albuterol 90 mcg/actuation metered dose inhaler Inhale 2 puffs into the lungs every 4 (four) hours as needed    (Patient not taking: Reported on 06/20/2022)     \u0007 buPROPion 450 mg TABSR24H Take 450 mg by mouth daily     \u0007 capecitabine (XELODA) 500 mg tablet Take 3 tabs with food in the morning, and take 3 tabs with food in the evening. Take for 7 days and then off for 7 days. (Patient not taking: Reported on 06/20/2022) 84 tablet 5   \u0007 clonazePAM (KLONOPIN) 1 mg tablet Take 1 mg by mouth 2 (two) times daily as needed     \u0007 clotrimazole (LOTRIMIN) 1 % cream Apply topically Twice a day Use as instructed apply to sides of mouth/lips     \u0007 cyanocobalamin 2,500 mcg SL tablet 2,500 mcg     \u0007 cyclobenzaprine (FLEXERIL) 10 mg tablet   1 Tab, PO TID, PRN AS NEEDED, Qty: 90 Tab, Refills: 2, ***** ***** ***** Electronically, ***** ***** *****, 30     \u0007 diclofenac (VOLTAREN) 1 % gel Apply topically 4 (four) times daily as needed     \u0007 etodolac (*****) 400 mg tablet Take 400 \n mg by mouth 2 (two) times daily     \u0007 famotidine (PEPCID) 20 mg tablet Take 1 tablet (20 mg total) by mouth in the morning and 1 tablet (20 mg total) in the evening. (Patient taking differently: Take 40 mg by mouth in the morning and 40 mg at noon and 40 mg in the evening.) 180 tablet 2   \u0007 FLUoxetine (PROZAC) 40 mg capsule Take 40 mg by mouth daily     \u0007 fluticasone propionate (FLOVENT DISKUS INH) Inhale into the lungs (Patient not taking: Reported on 06/20/2022)     \u0007 gabapentin (NEURONTIN) 300 mg capsule Take 1,200 mg by mouth 2 (two) times daily     \u0007 hydrocortisone 2.5 % ointment Apply topically 4 (four) times daily     \u0007 levocarnitine HCl (ACETYL-L-CARNITINE MISC) Take 1 tablet by mouth daily     \u0007 lidocaine-diphenhydrAMINE-maalox (MAGIC MOUTHWASH) mouthwash Use as directed 10 mL in the mouth or throat in the morning and 10 mL at noon and 10 mL in the evening and 10 mL before bedtime. For mouth sores.. 480 mL 1   \u0007 lipase-protease-amylase (CREON) 24,000-76,000 -120,000 unit capsule Take 1 to 2 caps with meals, take 1 cap with snacks 180 capsule 11   \u0007 loperamide (IMODIUM) 2 mg capsule Take two capsules by mouth at the onset of diarrhea, then take one capsule every 2 hours until diarrhea-free for 12 hours. (Patient not taking: Reported on 06/20/2022) 30 capsule 5   \u0007 LORazepam (ATIVAN) 1 mg tablet Take 1 mg by mouth nightly as needed for Anxiety        \u0007 losartan (COZAAR) 50 mg tablet Take 50 mg by mouth every evening (Patient not taking: Reported on 06/20/2022)     \u0007 montelukast (SINGULAIR) 10 mg tablet Take 10 mg by mouth daily     \u0007 multivitamin tablet Take 1 tablet by mouth daily     \u0007 naloxone 4 mg/actuation SPRAYNAERO 1 spray by Nasal route once as needed (suspected overdose) for up to 1 dose Call 911. Repeat if needed (Patient not taking: Reported on 06/20/2022) 2 each 0   \u0007 ondansetron (ZOFRAN) 8 mg tablet Take 1 tablet (8 mg total) by mouth every 8 (eight) hours as needed (nausea and vomiting) (Patient not \n taking: Reported on 06/20/2022) 30 tablet 5   \u0007 oxyCODONE (ROXICODONE) 5 mg tablet Take 1 tablet (5 mg total) by mouth every 8 (eight) hours as needed for Pain (Pain not relieved by gabapentin, tylenol or ibuprofen) (Patient not taking: Reported on 06/20/2022) 10 tablet 0   \u0007 polyethylene glycol (MIRALAX) 17 gram packet Take 17 g by mouth as needed in the morning (constipation). (Patient not taking: Reported on 06/20/2022)     \u0007 prochlorperazine (COMPAZINE) 10 mg tablet Take 1 tablet (10 mg total) by mouth every 6 (six) hours as needed (nausea and vomiting) (Patient not taking: Reported on 06/20/2022) 30 tablet 5   \u0007 rOPINIRole (REQUIP) 1 mg tablet Take 1 mg by mouth nightly at bedtime     \u0007 tamsulosin (FLOMAX) 0.4 mg 24 hr capsule Take 1 capsule (0.4 mg total) by mouth daily 30 capsule 1   \u0007 thyroid, porcine, (ARMOUR) 120 mg tablet 120 mg daily     \u0007 UNABLE TO FIND 28 mg daily Med Name: ferrous  bis-glycinate     \u0007 VITAMIN B-1, MONONITRATE, 100 mg tablet Take 100 mg by mouth daily (Patient not taking: Reported on 06/20/2022)     \u0007 zolpidem (AMBIEN) 10 mg tablet Take 10 mg by mouth nightly as needed       No current facility-administered medications on file prior to visit.     ALLERGIES:  Allergies/Contraindications   Allergen Reactions   \u0007 Sulfa (Sulfonamide Antibiotics) Swelling       FAMILY HISTORY:  Her father had prostate cancer. She has 3 paternal aunts with breast cancer all diagnosed in their 80s.    SOCIAL HISTORY:  She lives with her sister currently. She has 2 sons. She is separated from her second husband. She used to teach *****. She has a history of alcohol abuse. She used chewing tobacco when she was younger for about 10 years. Most of her life, she was very active including skiing and competitive squash but she is now limited by arthritis.      PE:    Vitals: There were no vitals taken for this visit.   In general, she appears comfortable.      LABORATORY RESULTS:  Results for orders placed or \n performed in visit on 06/21/22   Comprehensive Metabolic Panel, Random (BMP, AST, ALT, T.BILI, ALKP, TP, ALB)   Result Value Ref Range    Albumin, Serum / Plasma 3.4 3.4 - 4.8 g/dL    Alkaline Phosphatase 151 (H) 38 - 108 U/L    Alanine transaminase 29 10 - 61 U/L    AST 24 5 - 44 U/L    Bilirubin, Total 0.3 0.2 - 1.2 mg/dL    Urea Nitrogen, Serum / Plasma 12 7 - 25 mg/dL    Calcium, total, Serum / Plasma 9.3 8.4 - 10.5 mg/dL    Chloride, Serum / Plasma 99 (L) 101 - 110 mmol/L    Creatinine 0.82 0.55 - 1.02 mg/dL    ***** 79 >59 mL/min/1.73m2    Potassium, Serum / Plasma 5.3 (H) 3.5 - 5.0 mmol/L    Sodium, Serum / Plasma 129 (L) 135 - 145 mmol/L    Protein, Total, Serum / Plasma 6.8 6.3 - 8.6 g/dL    Carbon Dioxide, Total 25 22 - 29 mmol/L    Anion Gap 5 4 - 14    Glucose, non-fasting 109 70 - 199 mg/dL   Complete Blood Count with 5-part Differential   Result Value Ref Range    WBC Count 11.7 (H) 3.4 - 10.0 x10E9/L    RBC Count 3.34 (L) 4.00 - 5.20 x10E12/L    Hemoglobin 11.5 (L) 12.0 - 15.5 g/dL    Hematocrit 34.3 (L) 36.0 - 46.0 %    MCV 103 (H) 80 - 100 fL    MCH 34.4 (H) 26.0 - 34.0 pg    MCHC 33.5 31.0 - 36.0 g/dL    Platelet Count 485 (H) 140 - 450 x10E9/L    Abs Neutrophils 5.24 1.80 - 6.80 x10E9/L    Abs Lymphocytes 2.75 1.00 - 3.40 x10E9/L    Abs Monocytes 2.59 (H) 0.20 - 0.80 x10E9/L    Abs Eosinophils 0.93 (H) 0.00 - 0.40 x10E9/L    Abs Basophils 0.04 0.00 - 0.10 x10E9/L    Abs Imm Granulocytes 0.13 (H) <0.10 x10E9/L       IMAGING:  CT Abdomen /Pelvis with Contrast    Result Date: 06/19/2022  CT ABDOMEN/PELVIS WITH CONTRAST  *****/*****/***** *****:***** PM CLINICAL HISTORY: surveillance scan h/o pancreatic cancer COMPARISON:  CT abdomen/pelvis 04/11/2022 TECHNIQUE: CT of the abdomen and pelvis was performed. MEDICATIONS: Iohexol 350 - 110 mL - Intravenous RADIATION DOSE INDICATORS: ***** *****: 3 , CTDIvol Min: 7.9 mGy, CTDIvol Max: 7.9 mGy, DLP: 504.5 mGy.cm. The following accession numbers are related to this dose \n report *****,***** FINDINGS: Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date. Liver:  Unremarkable Gallbladder: Unremarkable Spleen:  Status post splenectomy. Pancreas:  Expected postsurgical changes related to prior distal pancreatectomy of the body and tail. Adrenal Glands:  Unremarkable Kidneys: Subcentimeter bilateral hypodensities too small to definitively characterize, likely cysts, unchanged. GI Tract:  Small hiatal hernia, unchanged. Vasculature:  - Decrease in stranding surrounding the celiac axis and branches, reflecting expected postoperative changes. - Interval resolution of previously noted narrowing of the junction of the portal vein and superior mesenteric vein. - Mild atherosclerotic disease of the abdominal aorta and its branches. Lymphadenopathy: Unchanged gastrohepatic lymph node measuring 9 x 7 mm, may be reactive secondary to prior surgery. Peritoneum: No ascites. Interval resolution of previously seen trace free fluid adjacent to the cut edge of pancreas. Bladder: Unremarkable Reproductive organs: Intrauterine device in place. Leiomyoma. Bones:  No suspicious lesions. Redemonstrated bilateral pars defects at L5, and degenerative disc disease at L5/S1 with grade 1 anterolisthesis. Extraperitoneal soft tissues: Unremarkable Lines/drains/medical devices: As above     1. Expected postsurgical changes related to prior distal pancreatectomy of the body and tail. 2. No evidence of metastatic disease in the abdomen or pelvis. Report dictated by: ***** *****, MD, signed by: ***** *****-***** *****, MD Department of Radiology and Biomedical Imaging    CT Chest with Contrast    Addendum Date: 06/19/2022    //Segmental/subsegmental right lower lobe pulmonary embolism discussed with Ong, RN by ***** ***** *****, MD (Radiology) on *****/*****/***** *****:***** PM.// Report dictated by: ***** ***** *****, MD, signed by: ***** ***** *****, MD \n Department of Radiology and Biomedical Imaging    Result Date: 06/19/2022  CT CHEST WITH CONTRAST CLINICAL HISTORY:  Pancreatic cancer, primary surveillance scan COMPARISON: CT chest dated 02/10/2022 TECHNIQUE: Serial 1.25 mm axial images through the chest were obtained after the administration of intravenous contrast. Iohexol 350 - 110 mL - Intravenous RADIATION DOSE INDICATORS: Exposure Events: 3 , CTDIvol Min: 7.9 mGy, CTDIvol Max: 7.9 mGy, DLP: 504.5 mGy.cm. The following accession numbers are related to this dose report *****,***** FINDINGS: LUNGS: Unchanged left lower lobe pulmonary nodule (December 301). No new or enlarging pulmonary nodules. PLEURA: No pleural effusion or pneumothorax. MEDIASTINUM: No suspicious lymphadenopathy. HEART/GREAT VESSELS: The heart is normal in size without evidence of pericardial effusion. No significant coronary artery atherosclerotic calcifications. Normal caliber thoracic aorta and main pulmonary artery. Pulmonary embolism within the segmental and subsegmental right lower lobe pulmonary arteries. BONES/SOFT TISSUES: No suspicious osseous lesions. VISIBLE ABDOMEN: Please see dedicated Abdomen and Pelvis CT report.     Tiny left lower lobe pulmonary nodule. No new or enlarging pulmonary nodules. Report dictated by: ***** ***** *****, MD, signed by: ***** ***** *****, MD Department of Radiology and Biomedical Imaging       ASSESSMENT:  1. Adenocarcinoma of the tail of the pancreas, status post distal pancreatectomy and splenectomy with positive lymph nodes.  2. Psoriatic arthritis  3. Chronic asthma  4. Hypertension  5. history of MRSA infection x2  6. ECOG 2  7. Hyponatremia, etiology uncertain  8. Incidental PE   9. History of frequent falls associated with loss of balance  10. Severe hand-foot syndrome and mucositis secondary to chemotherapy          PLAN:    We will hold her chemotherapy until she has resolution of her hand-foot syndrome. We will then resume treatment, treating \n 5 days out of 7.    With regard to her falls, we have checked a CT of the head which is unremarkable. It is possible that the capecitabine is contributing to a balance disorder and we will see how she does off treatment. Fortunately, she is very agreeable to continuing to use a walker.    With regard to her PE, we will start her on Lovenox. We will also get Dopplers to understand whether she has DVT. She is very agreeable to doing Lovenox as she has done it before.  .  I performed this evaluation using real-time telehealth tools, including a live video Zoom connection between my location and the patient's location. Prior to initiating, the patient consented to perform this evaluation using telehealth tools.    I spent a total of 40 minutes on this patient's care on the day of their visit excluding time spent related to any billed procedures. This time includes time spent with the patient as well as time spent documenting in the medical record, reviewing patient's records and tests, obtaining history, placing orders, communicating with other healthcare professionals, counseling the patient, family, or caregiver, and/or care coordinatio
```

---

## Letter A

Dear Patient,

I hope this letter finds you well. I want to share some important information about your health in a way that is easy to understand.

**Diagnosis and Stage:**
You have a type of cancer called adenocarcinoma. This cancer was found in the tail part of your pancreas. The cancer has spread to two lymph nodes, but it hasn't spread to other parts of your body.

**Treatment Plan:**
You had surgery to remove the cancer from your pancreas. After the surgery, you started chemotherapy to help prevent the cancer from coming back. However, the chemotherapy made you feel very sick, so we stopped it for now.

**Key Test Results:**
Recent tests show that you have a low sodium level in your blood. We are watching this closely. Also, there was a small blood clot found in your lung. We will give you medicine to help prevent more clots.

**Next Steps:**
We will wait until your skin problems from the chemotherapy get better before starting again. We will also check if the medicine is causing you to fall. You are using a walker to stay safe.

**What to Watch For:**
Please watch for any new symptoms like dizziness or trouble walking. If you notice anything different, let us know right away.

Remember, you can always talk to your care team if you have any questions or concerns. We are here to support you.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new visit regarding your cancer treatment.
**What's new or changed since your last visit?**
You have a history of adenocarcinoma (cancer that started in gland cells) of the tail of the pancreas, which was treated with surgery. Imaging shows no new signs of cancer spread. However, you have a pulmonary embolism (a blockage in the lung's blood vessels) and low sodium levels.
**What treatment or medication changes were made?**
We will pause your chemotherapy due to side effects like hand-foot syndrome and mucositis. We will also start you on Lovenox for the pulmonary embolism.
**What is the plan going forward?**
We will restart your chemotherapy once your hand-foot syndrome improves. We will treat you 5 days out of 7. We will also check for deep vein thrombosis (DVT) with Doppler tests.
We understand that this is a challenging time and that managing these health changes can be stressful. We want to reinforce that you are not alone in this process.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 19

**Cancer Type:** Adenocarcinoma of the head of the pancreas

## Original Clinical Note

```
  This is an independent visit    SUBJECTIVE  This is a very pleasant 69-year-old African-American woman from *****, *****. She has recently relocated to *****. She has a diagnosis of locally advanced and unresectable adenocarcinoma of the head of the pancreas.    She presented with abdominal pain and weight loss as early as November and December 2016. Eventually, she went to the emergency room in February 2017 and received pain medications and a workup was initiated. CT scan of the abdomen done on 03/08/2017 showed a mass measuring 3.6 x 4.3 x 4.3 cm. Vascular involvement was not described. An 11 mm lesion in the liver was seen but not characterized or biopsied. EUS done on February 21 showed a mass in the uncinate process and involvement of the SMA. FNA of the mass revealed an adenocarcinoma.    She was initially started on treatment with gemcitabine and Abraxane but had demonstrated radiographic progression on 06/14/2017. At that point, she was started on dose modified FOLFIRINOX (C1D1 on 07/31/17).     08/15/2017: C2D1 FOLFIRINOX     Interim History:  Patient reports increase fatigue over the past 3 weeks. She is sitting most of the day but can perform all of her ADLs. She most recently moved in with her daughter and is living with her in ***** and is able to get up and go for short walks.     Has also experienced increased emesis, nausea, and burping. Has 2 or 3 episodes of emesis a day, which are not associated with eating. Has only tried Zofran twice, states that medication did help, but she is trying to avoid taking too many meds and so has been hesitant to take zofran more often.    Continues to take Creon 3 tabs with meals/2 tabs with snacks. Her upper abdominal pain continues but is helped by taking Methadone BID and Oxycodone PRN (usually twice a day as needed).     Reports 1 wk history of bilateral LE edema. Elevating her legs up does help to decrease swelling. Denies leg pain, SOB, chest \n pain.     Also endorses continued poor appetite, but has gained some weight.     Reports recent onset of dark colored urine x 3 days. Denies fevers, chills, pruritis.    PAST MEDICAL HISTORY:    MEDICAL ILLNESSES:  She was diagnosed with diabetes in 2007 and was on an oral hypoglycemic. Her blood sugars are now in the normal range.    PRIOR SURGERIES:  IUD removal, 1977   arthroscopic surgery left knee, about 8 years ago.    INJURIES:  None reported    CURRENT MEDICATIONS:  Medications the patient states to be taking prior to today's encounter.   Medication Sig   \u0007 aspirin 81 mg EC tablet Take 81 mg by mouth Daily.   \u0007 atorvastatin (LIPITOR) 40 mg tablet TK 1 T PO QD   \u0007 coenzyme Q10 100 mg capsule Take by mouth Daily.   \u0007 CREON 36,000-114,000- 180,000 unit capsule Take 3 tabs with meals and 2 tabs with snacks.   \u0007 metFORMIN (GLUCOPHAGE) 850 mg tablet Take 850 mg by mouth 2 (two) times daily with meals.   \u0007 methadone (DOLOPHINE) 10 mg tablet Take 10 mg by mouth every 12 (twelve) hours.   \u0007 metoclopramide HCl (REGLAN) 10 mg tablet    \u0007 multivitamin with iron-minerals-folic acid (MULTIVITAMIN-IRON-MINERALS-FOLIC ACID) 3,500-18-0.4 unit-mg-mg chewable tablet Take 1 tablet by mouth Daily.   \u0007 omeprazole (PRILOSEC) 20 mg capsule TK 1 C PO QD   \u0007 ondansetron (ZOFRAN) 8 mg tablet TK 1 T PO Q 8 H PRN NV   \u0007 oxyCODONE (ROXICODONE) 5 mg tablet Take 5 mg by mouth every 4 (four) hours as needed for Pain. 1-2 tabs po q 3-4 hours   \u0007 polyethylene glycol (MIRALAX) 17 gram packet    \u0007 [DISCONTINUED] CREON 36,000-114,000- 180,000 unit capsule Take 2-3 caps po q snack and August 07 caps po QAC     ALLERGIES: None    FAMILY HISTORY:  She has 4 siblings. One sister was diagnosed with pancreatic cancer at the age 57. A maternal aunt had breast cancer.    SOCIAL HISTORY:  Used tobacco between the ages of 17-34 years. There is no history of alcohol abuse. She is retired. She previously worked for   ***** ***** ***** ***** ***** ***** *****.    She \n is a widow and her husband died of colorectal cancer. She has 1 daughter ***** who lives in ***** *****     has a *****-year-old child.    PE:  Vitals: BP 132/85  | Pulse 76  | Temp 36.5 C (97.7 F) (Oral)  | Resp 18  | Ht 163.1 cm (5' 4.21\") Comment: April 2017 | Wt 45.4 kg (100 lb)  | SpO2 100%  | BMI 17.05 kg/m    In general, she appears comfortable. ECOG 1  Head: Conjunctiva with slight jaundice. Tongue with white/yellowish thick coating.   Nodes no clinical adenopathy  Neck without thyromegaly  Lungs clear to A and P  Heart RR without murmur or gallop  Abd soft and nontender. No hepatosplenomegaly. No fluid wave. No masses  GU not done  Skin without rash or petechiae  Extremities: bilateral +1 LE edema from feet to ankles  Neuro without gross focal deficit    LABORATORY RESULTS:  Results for orders placed or performed during the hospital encounter of 09/05/17   Comprehensive Metabolic Panel - UCSF/LabCorp/Quest (BMP, AST, ALT, T.BILI, ALKP, TP, ALB)   Result Value Ref Range    Albumin, Serum / Plasma 3.3 (L) 3.5 - 4.8 g/dL    Alkaline Phosphatase 493 (H) 31 - 95 U/L    Alanine transaminase 387 (H) 11 - 50 U/L    Aspartate transaminase 357 (H) 17 - 42 U/L    Bilirubin, Total 7.4 (H) 0.2 - 1.3 mg/dL    Urea Nitrogen, Serum / Plasma 15 6 - 22 mg/dL    Calcium, total, Serum / Plasma 9.2 8.8 - 10.3 mg/dL    Chloride, Serum / Plasma 94 (L) 97 - 108 mmol/L    Creatinine 0.67 0.44 - 1.00 mg/dL    eGFR if non-African American 90 >60 mL/min    eGFR if African Amer 104 >60 mL/min    Potassium, Serum / Plasma 3.7 3.5 - 5.1 mmol/L    Sodium, Serum / Plasma 133 (L) 135 - 145 mmol/L    Protein, Total, Serum / Plasma 7.5 6.0 - 8.4 g/dL    Carbon Dioxide, Total 27 22 - 32 mmol/L    Anion Gap 12 4 - 14    Glucose, non-fasting 131 70 - 199 mg/dL   Complete Blood Count with 5-part Differential   Result Value Ref Range    WBC Count 15.3 (H) 3.4 - 10 x10E9/L    RBC Count 3.29 (L) 4.0 - 5.2 x10E12/L    Hemoglobin 10.4 (L) 12.0 - 15.5 \n g/dL    Hematocrit 30.8 (L) 36 - 46 %    MCV 94 80 - 100 fL    MCH 31.6 26 - 34 pg    MCHC 33.8 31 - 36 g/dL    Platelet Count 203 140 - 450 x10E9/L    Neutrophil Absolute Count 12.04 (H) 1.8 - 6.8 x10E9/L    Lymphocyte Abs Cnt 1.75 1.0 - 3.4 x10E9/L    Monocyte Abs Count 1.20 (H) 0.2 - 0.8 x10E9/L    Eosinophil Abs Ct 0.01 0.0 - 0.4 x10E9/L    Basophil Abs Count 0.04 0.0 - 0.1 x10E9/L    Imm Gran, Left Shift 0.23 (H) <0.1 x10E9/L     IMAGING:  No images are attached to the encounter.    ASSESSMENT / PLAN:  1. Localized adenocarcinoma of the pancreas, on therapy with uncertain disease control  Patient is now s/p 2 cycles of FOLFIRINOX. She presents for C3 today but labs are worrisome for disease progression, biliary obstruction, or ascending cholangitis: T bili 7.4, Alk Phos 493, AST 357, ALT 387    Given that her most recent CT was performed almost 3 months ago (outside CT from 06/14/17) plus today's evidence of hyperbilirubinemia, I have recommended that we postpone today's infusion and have her undergo urgent imaging to rule out biliary ductal dilatation versus disease progression in the liver. ED precautions were reviewed with patient and her daughter. As she may be at risk for cholangitis, she was advised to present to local ED ASAP if she develops fevers/chills. Furthermore, as she presents with daily emesis, which could indicate partial small bowel obstruction, she was also advised to go to ED if she is unable to tolerate any PO intake.     2. Bilateral LE edema  Will have pt undergo bilateral US today to r/o DVT.     3. Multiple symptoms  Encouraged referral to SMS to help with management of her pain, fatigue, and anorexia. Pt agrees to this consultation.     I spent a total of 40 minutes face-to-face with the patient and 40 minutes of that time was spent counseling regarding the diagnosis, the treatment plan, the prognosis and symptoms    ***** *****, NP    ADDENDUM:   Today's CT scan indicates the \n following:  1.  In comparison to 06/14/2017, interval increase in size of ill-defined pancreatic head/uncinate process mass, resulting in obstruction of the biliary system distally with proximal intrahepatic and extrahepatic biliary ductal dilatation. Increased pancreatic ductal dilatation.    2.  Increased distention of the stomach with narrowing of the third portion of duodenum, likely due to mass effect from adjacent pancreatic mass, concerning for developing or partial obstruction.    An urgent referral to GI for an ERCP was placed.     Pt was also called and informed of these findings. ED precautions were reviewed.     Lower extremity US did not identify any DVTs.    ***** *****, NP   \n
```

---

## Letter A

Dear Patient,

I hope this letter finds you well. I want to share some important information about your health and the next steps in your care.

**Diagnosis and Stage:**
You have a type of cancer called adenocarcinoma in the head of your pancreas. The cancer is advanced and cannot be removed by surgery.

**Treatment Plan:**
You are currently receiving a treatment called FOLFIRINOX. This is a combination of medicines to fight the cancer. We need to check how well the treatment is working.

**Key Test Results:**
Some of your recent tests show that the cancer might be growing again. Your blood tests show high levels of certain substances, which can mean the cancer is causing problems with your bile ducts.

**Next Steps:**
We need to do more tests to see exactly what is happening. You will have a special picture taken of your pancreas and bile ducts. If there are any blockages, we will refer you to a doctor who specializes in digestive issues to help open them up.

**What to Watch For:**
Please watch for any new symptoms like fever, chills, or trouble eating. If you have any of these, please go to the emergency room right away.

**Other Symptoms:**
You have been feeling tired and having some swelling in your legs. We will look into these issues and try to make you feel better.

**Medicines:**
You are taking several medicines to manage your pain and other symptoms. Please keep taking them as directed.

**Important Reminders:**
Remember to talk to your care team if you have any questions or concerns. We are here to support you.

Take care and stay strong.

Warm regards,

[Your Name]
Medical Communication Specialist
Cancer Center

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for a new visit regarding your localized adenocarcinoma (cancer that started in gland cells) of the head of the pancreas.
**What's new or changed since your last visit?**
The cancer is not responding to treatment. Recent imaging shows the tumor has grown and is causing blockages in your bile ducts. Lab results also show signs of liver issues, including high levels of bilirubin and enzymes.
**What treatment or medication changes were made?**
Today’s treatment cycle of a chemotherapy combination was postponed because of concerns about disease progression, biliary obstruction, or ascending cholangitis.
**What is the plan going forward?**
You will need urgent imaging to check for biliary ductal dilatation or disease progression in the liver. A bilateral ultrasound will also be done to rule out deep vein thrombosis (DVT). You will be referred to a gastroenterologist for an ERCP (a procedure to unblock the bile ducts).
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

# Pancreatic Cancer — Sample 20

**Cancer Type:** Pancreatic adenocarcinoma

## Original Clinical Note

```
Mr. ***** is a 51 y.o. male who comes to our GI Oncology practice at the UCSF Comprehensive Cancer Center in consultation for his . The history was obtained from the patient, along with my review of all his outside records.    History of Present Illness  This pleasant gentleman, previously quite healthy overall, began developing intermittent mid/LUQ abdominal discomfort, bloating, and gassiness starting in ***** ***** after returning from some travels to ***** ***** *****. These symptoms increased in frequency prompted the following diagnostic tests over the succeeding months:    - 10/22/2020: EGD (WNL); colonoscopy with 2 tubular adenomas in the rectum and cecum  - 10/22/2020: Stool studies negative for infectious organisms  - December 2020: Trial of metronidazole and rifaximin followed by probiotics, with dietary modifications; minimal relief  - 03/25/2021: Abdominal U/S notable for prominent hepatic steatosis and splenomegaly  - 05/21/2021: Abdominal/pelvic CT scan: pancreatic tail cystic lesion (5 cm) with multiple hepatic lesions concerning for metastases, as well as suggestion of peritoneal carcinomatosis (see formal report below)  - 05/28/2021: EUS (at *****): confirmed pancreatic tail mass (39.4 mm by 22.6 mm) with cystic and necrotic components; fine needle bx c/w adenocarcinoma. MMR proteins all intact by IHC.    The patient's present review of systems was reviewed and notable for the following:  - Appetite fair; some early satiety and postprandial distention/discomfort; 8 lb weight loss  - Uses primarily Tylenol for abdominal discomfort, rarely Percocet  - Semi-regular BMs, no bleeding  - Normally very active, working out 5x/weekly  - Denies fevers/chills/night sweats  - No pre-existing peripheral sensory neuropathy  - No SOB, cough, or respiratory symptoms.  - No jaundice sxs  - (+) modest anxiety         PAST MEDICAL AND SURGICAL HISTORY    Past Medical History:   Diagnosis Date   \u0007 Anxiety     \"sometimes\"   \u0007 \n Environmental allergies    HIV negative.    Medications:  \u0007 oxyCODONE-acetaminophen (Percocet) 5-325 mg tablet Take 2 tablets by mouth every 6 (six) hours if needed for severe pain for up to 10 days. 30 tablet 0   \u0007 valACYclovir (Valtrex) 1 gram tablet prn cold sores  \u0007 Lorazepam PRN      Allergies/Contraindications  No Known Allergies      FAMILY HISTORY:  Family History   Problem Relation Name Age of Onset   \u0007 Ovarian cancer Mother     \u0007 Lung cancer Maternal grandmother     \u0007 Breast cancer Paternal grandmother           PERSONAL AND SOCIAL HISTORY  Married (husband), no children.  Occupational History   \u0007 Previously in *****, now  ***** *****   Tobacco Use   \u0007 Smoking status: Former Smoker     Packs/day: 0.00     Quit date: 1992     Years since quitting: 30.3   \u0007 Smokeless tobacco: Never Used   Substance and Sexual Activity   \u0007 Alcohol use: Yes     Alcohol/week: 5.0 - 8.0 standard drinks     Types: 5 - 8 Standard drinks or equivalent per week     Comment:     5-8/wk   \u0007 Drug use: Yes   \u0007 Sexual activity: Not on file       Physical Exam:  ECOG PS 0  Blood pressure (!) 172/103, pulse 83, temperature 36.4 C (97.6 F), temperature source Temporal, resp. rate 16, height 172.5 cm (5' 7.91\"), weight 82.1 kg (181 lb), SpO2 97 %.  General appearance: No acute distress, conversant, appropriately interactive. Non-toxic, well-appearing.   Eyes: anicteric sclerae, moist conjunctivae; no lid lag.   HEENT: Atraumatic; oropharynx clear with moist mucous membranes and no mucosal ulcerations; normal hard and soft palate  Neck: Full range of motion, supple, no lymphadenopathy  Lungs/respiratory: Clear to auscultation, with normal respiratory effort and no intercostal retractions  Cardiovascular: RRR, normal S1 S2, no murmurs, gallops, or rubs.  Abdomen/GI: Soft, non-tender, nondistended; no masses, ascites, or hepatomegaly. Tympanitic to percussion.  Back: No point spinal tenderness.  Extremities: No peripheral edema or discoloration \n palpated or visualized.  Skin: Normal temperature, turgor and texture; no rash, ulcers or subcutaneous nodules visualized or palpated.  Neuro: Normal bulk, tone, and strength in upper and lower extremities. Intact gait. No focal sensorimotor deficits.  Psych: Appropriate affect, alert and oriented.        Relevant Diagnostic Studies:    Most recent labs notable for the following:    03/25/2021  WBC 6.0, Hct 46.5, plts 207  Creat 1.16  Alb 4.3, AST/ALT 26/61, tot bili 0.6, alk phos 63      I personally reviewed and interpreted the patient's relevant imaging studies in conjunction with the corresponding formal radiology reports, as follows:    CT Abdomen /Pelvis with Contrast   Result Date: 05/21/2021  FINDINGS:    Lines / tubes: None.     Lower Thorax: Normal.      Hepatobiliary:    Liver: Measures 14.9 cm in craniocaudal dimension. Multiple varying sized hypodense lesions   throughout the right and left lobes of the liver are present measuring up to 1.7 x 1.4 cm in   segment March 13 (axial image 19, series 2) and 0.9 x 0.8 cm in segments August 15 (axial image 16,   series 2).    Gallbladder: Normal.    Bile Ducts: Normal.    Hepatic vasculature: Normal.     Spleen: A wedge-shaped hypodense area is noted in the lower part of the spleen most likely   related to an infarct. Spleen is upper limits of normal size measuring 12.6 cm.     Pancreas: A 5 x 4.1 cm cystic lesion with irregular rim enhancement is noted involving the   pancreatic tail and abutting the splenic hilum. No pancreatic duct dilatation is noted.     Adrenals: Normal.     Kidneys/Ureters: Normal.     GI Tract:    GE junction and stomach: Normal.    Small bowel: Normal in diameter    Appendix: Not visualized.    Colon and rectum: Diffuse wall thickening of the descending and sigmoid colon is present.     Peritoneum / Retroperitoneum: Nodular foci of enhancement involving the greater omentum is   present. Small peritoneal free fluid is \n present. Irregular foci of nodular peritoneal   enhancement are present. Example in the left lateral peritoneal line (axial image 42, series   2).     Diffuse mesenteric edema with interspersed fluid is noted.     Lymph Nodes: Multiple retroperitoneal lymph nodes are present in an example left periaortic   lymph node (axial image 42, series 2) measuring 0.9 x 1.1 cm.     Vessels: Moderate atherosclerotic ossification of aorta.     Pelvis:    Prostate: Normal.    Seminal vesicles: Normal.    Bladder: Normal.     Bones and Soft Tissues: Normal.     IMPRESSION:  1. A lobulated cystic lesion with rim enhancement is noted involving the pancreatic tail   measuring 5 x 4.1 cm in size. This may represent a cystic neoplastic process, necrotic   metastatic disease or sequelae of prior inflammation.   2. Multiple hypodense lesions throughout the liver are present suspicious for   metastatic/neoplastic process.   3. Diffuse omental caking and multiple foci of nodular peritoneal enhancement are noted   representing peritoneal carcinomatosis. Small peritoneal free fluid and diffuse mesenteric   edema is present.   4. Wall thickening of the descending and sigmoid colon may be reactive to peritonitis or   related to colitis.   5. Prominent retroperitoneal lymph nodes are present.       Impression and Recommendations:  In summary, Mr. ***** is a 51 y.o. male with newly diagnosed metastatic pancreatic adenocarcinoma, still in robust condition overall.    At today's visit, we discussed his diagnosis, pathology, and investigations to date. We reviewed the details of his diagnosis and why the mainstay of treatment should consist of systemic therapy with the goal of achieving as deep and durable a remission as possible, hopefully translating into significant prolongation of life, but that treatment was not expected to be curative. As such, it would be important to carefully weigh the risks/benefits of therapy and to prioritize \n QoL considerations in the process. In terms of standard of care treatment options, there are several possible chemotherapy regimens to select from for metastatic pancreatic cancer, with the choice of therapy depending on the patient's age, performance status, co-morbid conditions, and personal preference; phase III evidence supports either FOLFIRINOX (biweekly infusional 5-FU, leucovorin, irinotecan, and oxaliplatin) or the combination of gemcitabine plus nab-paclitaxel as appropriate front-line choices. We discussed the logistics, scheduling, and some of the potential side effects of these respective regimens including but not limited to nausea/vomiting, alopecia, fatigue, cytopenias with resultant immunosuppression and/or anemia, and peripheral sensory neuropathy/cold sensitivity. If he opts for SoC treatment, I generally start with FOLFIRINOX for fit and otherwise healthy individuals like Mr. *****.    From the standpoint of clinical trial options for the first-line treatment of metastatic pancreatic cancer, we have the following trials here at ***** for which the patient could be considered:    - Our Precision Promise clinical trial platform, supported by the Pancreatic Cancer Action Network (PanCAN). In the first-line setting, treatment would entail randomization to either SOC chemotherapy (gemcitabine/nab-paclitaxel or FOLFIRINOX); or to one or more experimental therapy arms (presently, the only open experimental arm consists of gemcitabine/nab-paclitaxel plus pamrevlumab, an anti-CTGF (connective tissue growth factor) monoclonal antibody). Of note,this trial mandates fresh tumor tissue collection both pre- and on-treatment.    - The REVOLUTION (PICI0044) trial, sponsored by the Parker Institute for Cancer Immunotherapy (PICI). This trial also employs a platform design intended to test multiple different chemotherapy/immunotherapy combinations; at present, the one cohort open at ***** (cohort B) consists of \n the combination of gemcitabine/nab-paclitaxel together with hydroxychloroquine (an anti-autophagy agent) and two doses of low-dose ipilimumab. This trial is non-randomized; and also requires either an adequate amount of archived tumor material or a fresh pre-treatment biopsy for correlative analyses.     Irrespective of trial participation, we also talked about the role of somatic tumor testing of the patient's tumor specimen, if there is adequate cellularity, to assess for the presence of any actionable mutations or other genetic alterations that may inform future treatment options, either using already-approved agents or in the context of clinical trials. This has already been ordered using our UCSF500 in-house platform. I did explain that while this information is always good to have, it will not necessarily yield useful clinical information for treatment purposes.  Additionally, the patient should undergo genetic counseling with germline testing to assess for any hereditary predisposition to cancer, as is now recommended for all individuals with a pancreatic cancer diagnosis, irrespective of family history or known risk factors, as this would have screening implications for family members as well as potentially therapeutic implications for the patient.    From a symptom management perspective:  - No present opioid requirements  - Referral made to our Cancer Center nutritionist    Finally, he is also potentially eligible for several non-therapeutic trials we have here at ***** involving novel imaging and blood- and stool-based assays (for ctDNA and microbiome studies, respectively), if he is interested and chooses to receive his care here at *****.    All of the patient's questions were answered to his and his family's satisfaction. As a starting point, he expressed interest in the REVOLUTION trial, and so we will get him an ICF and move in this direction if he wishes to consent and be screened for this \n study.      Medical decision-making:  Problems: His active cancer represents a life-threatening illness.  Risk of complications, morbidity/mortality of patient management: High; the patient's systemic cancer therapy requires regular and intensive monitoring for potential major/life-threatening toxicities.  We discussed the following at today's visit:        Time spent:  I spent 75 minutes reviewing the patient's history and medical records in advance, in F2F time during his visit, and in post-visit charting..    Medical decision-making:  Problems: His active cancer represents a life-threatening illness.  Risk of complications, morbidity/mortality of patient management: High; the patient's systemic cancer therapy requires regular and intensive monitoring for potential major/life-threatening toxicities.    \n
```

---

## Letter A

Dear Mr. *****,

Thank you for coming to our cancer center. I want to share with you what we found and what we plan to do next.

**Diagnosis and Stage:**
You have a type of cancer called pancreatic adenocarcinoma. It has spread to other parts of your body, which means it is at an advanced stage.

**Key Test Results:**
- We found a growth in your pancreas and some spots in your liver.
- Tests showed that the growth is cancer.
- Other tests showed that you have some swelling in your belly and some fluid around your organs.

**Treatment Plan:**
We talked about different ways to treat your cancer. The main goal is to make the cancer smaller and help you feel better for as long as possible. Here are your options:

1. **Standard Treatment:** There are two main types of medicine we can use. One is called FOLFIRINOX, and the other is a mix of gemcitabine and nab-paclitaxel. Both can help slow down the cancer. They might cause some side effects like feeling sick, being tired, or losing hair.

2. **Clinical Trials:** We also talked about joining a research study. These studies test new medicines to see if they work better than the ones we already have. You mentioned you might want to join a study called REVOLUTION. We can talk more about this if you're interested.

**Next Steps:**
- We will order some tests to look closely at your cancer cells. This helps us know if there are any special treatments that might work best for you.
- We will also talk to a doctor who knows about genes to check if you have any genes that might make you more likely to get cancer.
- We will set up an appointment with a dietitian to help you eat well.

**What to Watch For:**
- Keep an eye on how you feel. Let us know if you have any new symptoms or if your current symptoms get worse.
- If you have any questions or concerns, please talk to your care team.

Remember, you are not alone. We are here to support you every step of the way.

Sincerely,

[Your Name]
Medical Communication Specialist
UCSF Comprehensive Cancer Center

Please feel free to ask any questions you may have. Your care team is here to help you understand everything clearly.

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

## Letter B

Dear Patient,
We hope you are doing well. We are writing this letter to help you understand what happened during your recent visit.
**Why did you come to the clinic?**
You came in for an initial consult regarding your newly diagnosed metastatic pancreatic adenocarcinoma (cancer that started in gland cells).
**What's new or changed since your last visit?**
Your cancer has spread to your liver and peritoneum. A biopsy of a mass in your pancreas confirmed adenocarcinoma (cancer that started in gland cells). Imaging showed a 5 cm cystic lesion in your pancreatic tail with multiple liver lesions and signs of cancer spread in the abdomen lining. Your blood pressure is 172/103, and you've lost 8 pounds. There is no sign of jaundice or ascites.
**What treatment or medication changes were made?**
Supportive medications for pain control, such as oxyCODONE-acetaminophen (Percocet), were discussed.
**What is the plan going forward?**
We discussed standard of care treatment options, including a chemotherapy combination or gemcitabine plus nab-paclitaxel. Clinical trial options were also discussed, such as the Precision Promise trial or the REVOLUTION trial. Fresh tumor tissue collection for the Precision Promise trial and potential pre-treatment biopsy for the REVOLUTION trial were mentioned. You will undergo genetic counseling and germline testing to assess for any hereditary predisposition to cancer. A referral was made to our Cancer Center nutritionist. If you wish to consent and be screened for the study, further steps will be taken.
Thank you for trusting us with your care.
Please feel free to contact us if you have any questions.
Sincerely, Your Care Team

### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

---

