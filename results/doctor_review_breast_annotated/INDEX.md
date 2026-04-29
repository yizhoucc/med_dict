# Breast Cancer Annotated Test Set — Doctor Review

**Date:** 2026-04-29

**Model:** Qwen2.5-32B-Instruct-AWQ via vLLM

**Pipeline:** v31 (5-gate verification + 40+ POST hooks)

**Samples:** 20 annotated breast cancer notes (held-out test set, never used during development)

**Auto-review:** P0=0, P1=4, P2=143


---


## Instructions for Reviewers


For each sample below, please review:

1. **Original Note** — the clinical note as written by the oncologist

2. **Extracted Data** — structured fields extracted by the system

3. **Patient Letter** — the letter generated for the patient


For each letter, please rate (1-5):

- **Accuracy**: Are all facts correct?

- **Completeness**: Is important information included?

- **Safety**: Could anything cause harm?

- **Simplification**: Is it understandable at an 8th-grade level?

- **Overall Quality**: Would you send this to a patient?


---


### [Sample 1](sample_01.md)

- **Type:** ER-/PR-/HER2- (HER2 IHC 1; FISH ratio 2.1, but with HER2 sig

- **Stage:** Stage IIB (pT2N1a (inferred from 3.6 cm tumor and 1 positive


### [Sample 2](sample_02.md)

- **Type:** ER+/PR- invasive ductal carcinoma, HER2: not tested

- **Stage:** Originally Stage IIA, now metastatic (Stage IV)


### [Sample 3](sample_03.md)

- **Type:** ER-/PR-/HER2- (HER2 FISH neg, ratio 1.2) spindle cell metapl

- **Stage:** Locally advanced, multifocal (possibly awaiting biopsy confi


### [Sample 4](sample_04.md)

- **Type:** ER-/PR-/HER2- (TNBC) grade 3 invasive ductal carcinoma

- **Stage:** Stage I (inferred from tumor ≤2cm)


### [Sample 5](sample_05.md)

- **Type:** Left breast: ER+/PR+/HER2- grade 3 invasive ductal carcinoma

- **Stage:** Left breast: Stage III (T3N1); Right breast: Stage I (T1cN0)


### [Sample 6](sample_06.md)

- **Type:** ER-neg, PR neg, HER2 3+, FISH ratio 13, Ki67 10-15% invasive

- **Stage:** Metastatic (Stage IV)


### [Sample 7](sample_07.md)

- **Type:** ER-/PR-/HER2- triple negative invasive ductal carcinoma

- **Stage:** Originally Stage IIB (cT2 cN1 cM0) -> ypT1c(m) ypN1a (1/22 L


### [Sample 8](sample_08.md)

- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma with extensi

- **Stage:** Stage IIA (pT2(m)N1a)


### [Sample 9](sample_09.md)

- **Type:** ER+/PR+/HER2- grade 2 IDC (micropapillary features) with met

- **Stage:** Originally Stage III (T3N2), now Stage III


### [Sample 10](sample_10.md)

- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma

- **Stage:** Stage II (inferred from T2N1)


### [Sample 11](sample_11.md)

- **Type:** ER+/PR+ ductal carcinoma in situ (DCIS) with intermediate nu

- **Stage:** pTisNx


### [Sample 12](sample_12.md)

- **Type:** ER+/PR+/HER2- grade 2 invasive mammary carcinoma with mixed 

- **Stage:** Stage II (inferred from pT2 N0)


### [Sample 13](sample_13.md)

- **Type:** ER+/PR+/HER2- invasive ductal carcinoma

- **Stage:** Stage IIIB (inferred from 2.2 cm tumor with positive axillar


### [Sample 14](sample_14.md)

- **Type:** ER+/PR+/HER2 equivocal grade 1 invasive ductal carcinoma

- **Stage:** Stage IA (inferred from pT1 N0)


### [Sample 15](sample_15.md)

- **Type:** ER+/PR+/HER2 equivocal invasive ductal carcinoma with metast

- **Stage:** Stage III


### [Sample 16](sample_16.md)

- **Type:** ER+/PR+/HER2- grade 2-3 invasive lobular carcinoma

- **Stage:** Clinical stage III


### [Sample 17](sample_17.md)

- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma

- **Stage:** Stage IIb (T2N1M0)


### [Sample 18](sample_18.md)

- **Type:** ER+/PR-/HER2- grade 2 invasive ductal carcinoma

- **Stage:** Stage IIA (inferred from pT2 N0)


### [Sample 19](sample_19.md)

- **Type:** ER+/PR-/HER2- invasive ductal carcinoma (IDC) with extensive

- **Stage:** Clinical stage 2-3


### [Sample 20](sample_20.md)

- **Type:** Bilateral breast cancer, right breast: ER+/PR+/HER2+ with so

- **Stage:** Not mentioned in note

