# Breast Cancer Annotated Test Set — Doctor Review

**Date:** 2026-04-29 (updated — no stage inference)

**Model:** Qwen2.5-32B-Instruct-AWQ via vLLM

**Pipeline:** v31 (5-gate verification + 40+ POST hooks, no stage inference)

**Samples:** 20 annotated breast cancer notes (held-out test set, never used during development)


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
- **Stage:** Stage IIB (pT2N1a)


### [Sample 2](sample_02.md)
- **Type:** ER+/PR-/HER2- grade 1 infiltrating ductal carcinoma
- **Stage:** Originally Stage IIA, now metastatic (Stage IV)


### [Sample 3](sample_03.md)
- **Type:** ER-/PR-/HER2- (HER2 FISH neg, ratio 1.2) spindle cell metapl
- **Stage:** Locally advanced, multifocal


### [Sample 4](sample_04.md)
- **Type:** ER-/PR-/HER2- (TNBC) grade 3 invasive ductal carcinoma
- **Stage:** Stage III


### [Sample 5](sample_05.md)
- **Type:** Left breast: ER+/PR+/HER2- grade 3 invasive ductal carcinoma
- **Stage:** Left breast: Stage III (T3N1); Right breast: Stage I (T1cN0)


### [Sample 6](sample_06.md)
- **Type:** ER-neg, PR neg, HER2 3+, FISH ratio 13, Ki67 10-15% invasive
- **Stage:** Metastatic (Stage IV)


### [Sample 7](sample_07.md)
- **Type:** ER-/PR-/HER2- triple negative invasive ductal carcinoma
- **Stage:** Originally Stage IIB, now metastatic (Stage IV)


### [Sample 8](sample_08.md)
- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma with extensi
- **Stage:** Stage IIA (pT2(m)N1a)


### [Sample 9](sample_09.md)
- **Type:** ER+/PR+/HER2- grade 2 IDC (micropapillary features) with met
- **Stage:** Originally Stage III (T3N2), now metastatic (Stage IV)


### [Sample 10](sample_10.md)
- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma
- **Stage:** T2N1, clinical stage II


### [Sample 11](sample_11.md)
- **Type:** ER+/PR+ ductal carcinoma in situ (DCIS) with intermediate nu
- **Stage:** pTisNx


### [Sample 12](sample_12.md)
- **Type:** ER+/PR+/HER2- grade 2 invasive mammary carcinoma with mixed 
- **Stage:** Clinical stage II


### [Sample 13](sample_13.md)
- **Type:** ER+/PR+/HER2- invasive ductal carcinoma
- **Stage:** Stage III


### [Sample 14](sample_14.md)
- **Type:** ER+/PR+/HER2 equivocal grade 1 invasive ductal carcinoma
- **Stage:** 


### [Sample 15](sample_15.md)
- **Type:** ER+/PR+/HER2 equivocal metastatic adenocarcinoma, consistent
- **Stage:** Metastatic (Stage IV)


### [Sample 16](sample_16.md)
- **Type:** ER+/PR+/HER2- grade 2-3 invasive lobular carcinoma
- **Stage:** Clinical stage III


### [Sample 17](sample_17.md)
- **Type:** ER+/PR+/HER2- grade 2 invasive ductal carcinoma
- **Stage:** Stage IIb (T2N1M0)


### [Sample 18](sample_18.md)
- **Type:** ER+/PR-/HER2- grade 2 invasive ductal carcinoma
- **Stage:** 


### [Sample 19](sample_19.md)
- **Type:** ER+/PR-/HER2- grade 2 and grade 3 invasive ductal carcinoma 
- **Stage:** Clinical stage 2-3


### [Sample 20](sample_20.md)
- **Type:** Bilateral breast cancer, right breast: ER+/PR+/HER2+ with so
- **Stage:** 

