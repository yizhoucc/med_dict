# V31 iter10 — 逐 sample 逐 field 详细对比审查

> 匹配样本: 50 | 审查: 所有 field 的 vLLM vs HF 值

## 总结

| 指标 | 数量 |
|------|------|
| 总字段 | 550 |
| vLLM ≥ HF | 152 (27%) |
| HF > vLLM | 62 (11%) |
| 相同/近似 | 336 (61%) |
| All-win ROWs | 13/50 (26%) |
| All-win: [1, 3, 6, 8, 9, 14, 30, 33, 41, 42, 49, 52, 97] |


## ROW 1
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+ grade 2 invasive ductal carcinoma with metastatic re | ER+/PR+ invasive ductal carcinoma, HER2- | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to lungs, peritoneum, liver, ovary | Yes, to lungs, peritoneum, liver, and ovary | ≈ |
| response_assessment | The cancer is currently progressing. The CT chest, abdomen,  | The cancer is currently progressing. Evidence includes wides | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | If the patient is HR+/her 2 negative on biopsy will recommen | If the patient is HR+/HER2 negative on biopsy, will recommen | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | No specific genetic or molecular tests are planned. An appoi | (空) | **vLLM** |
| medication_plan | If HR+/HER2 negative on biopsy, will recommend ibrance and u | If HR+/HER2 negative on biopsy, will recommend Ibrance and [ | vLLM≥ |
**✅ ALL-WIN**

## ROW 2
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | Triple negative invasive ductal carcinoma (TNBC) with metast | ER-/PR-/HER2- triple negative invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to liver, bone, chest wall, and possibly intracranial/s | Yes, to liver, bone, and chest wall | vLLM≥ |
| response_assessment | MRI lumbar spine shows numerous bone marrow replacing lesion | Metastatic breast cancer is progressing. Scans show numerous | HF长 ⚠️ |
| current_meds | irinotecan | Irinotecan | vLLM≥ |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Due to her poor tolerance, irinotecan will be changed to eve | Due to poor tolerance, irinotecan will be changed to every o | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | Monitor with ALT, hepatitis B surface antigen and HBV DNA ev | Monitor with ALT, hepatitis B surface antigen and HBV DNA ev | ≈ |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | [REDACTED] 30mg TID, Flexeril TID, oxycodone 5mg prn, Cymbal | Doxycycline 100mg BID x 7d, [REDACTED] 30mg TID, Flexeril TI | HF长 ⚠️ |

## ROW 3
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | HR+, HER2 neu negative grade 2 invasive ductal carcinoma | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | PET scan follow up after PET results are back. | PET imaging follow up after results are back. | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 5
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma (micropapill | ER+/PR+/HER2- invasive ductal carcinoma (micropapillary feat | ≈ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to left internal mammary LN and sternum | Yes, to left cervical lymph nodes and left internal mammary  | HF长 ⚠️ |
| response_assessment | CT neck and CAP on 08/23/19 showed decreased size of left ce | CT neck and CAP: decreased size of left cervical chain LNs ( | HF长 ⚠️ |
| current_meds | anastrozole, palbociclib, goserelin | anastrozole, palbociclib, leuprolide | ≈ |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue current therapy including leuprolide, anastrozole,  | Continue current therapy with leuprolide, anastrozole, and p | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |

## ROW 6
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with extensi | ER+/PR+/HER2- grade 1 IDC (invasive ductal carcinoma) with e | ≈ |
| Stage_of_Cancer | Stage IA (inferred from pT1 N0) | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Recovering nicely with some residual nerve irritation, mild  | On treatment; response assessment not available from current | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Discussed starting letrozole today. [REDACTED] try to contin | Discussed starting letrozole today. Plan to continue letrozo | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Start letrozole today. Continue [REDACTED] for at least 3 ye | Start letrozole today. Continue [REDACTED] for at least 3 ye | vLLM≥ |
**✅ ALL-WIN**

## ROW 7
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER-/PR- IDC (invasive ductal carcinoma) with metastatic recu | ER-/PR-/HER2+ invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Originally Stage II, now metastatic (Stage IV) | Originally Stage II, now Stage IV | vLLM≥ |
| Distant Metastasis | Yes, to left supraclavicular node and mediastinum | Yes, to the left supraclavicular node and mediastinum | ≈ |
| response_assessment | Probable mild progression in the left breast and possibly th | Probable mild progression in the left breast and possibly th | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Do not consider hormonal therapy at this time. Discussed cur | Do not consider hormonal therapy at this time. | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Recommend [REDACTED] as next line of treatment. Recheck [RED | Discontinue current regimen including [REDACTED]/Herceptin/T | vLLM≥ |

## ROW 8
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) grade 3 invasive duct | ER-/PR-/HER2+ (IHC 3+, FISH ratio 5.7) invasive ductal carci | vLLM≥ |
| Stage_of_Cancer | Originally Stage III, now post-neoadjuvant with 3 of 28 LN p | Stage III | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | The patient received neoadjuvant TCHP followed by left lumpe | The patient has completed neoadjuvant therapy and had a left | ≈ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | adjuvant AC x 4 cycles, to be followed by T-DM1; radiation | adjuvant AC x 4 cycles, to be followed by T-DM1 and radiatio | ≈ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 9
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 2 invasive ductal carcinoma | ER+/PR-/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage II (inferred from pT3 N1) | Stage II | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | The surgical pathology showed a 3.84 cm residual invasive du | The cancer has responded to neoadjuvant chemotherapy as evid | ≈ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Plan to start letrozole after radiation. | Radiation referral. Letrozole after radiation. | ≈ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 10
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | HR+ (ER+/PR+) HER2- grade 2 invasive ductal carcinoma | HR+ invasive ductal carcinoma, HER2- | vLLM≥ |
| Stage_of_Cancer | Stage II (inferred from 8.8 cm tumor with 20 lymph nodes inv | Stage II | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | No specific evidence regarding the current response of the c | Low risk [REDACTED]. S/p left mastectomy with a 8.8.cm [REDA | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue letrozole. To have radiation to the left chest wall | She will continue on letrozole started April 2021. To have r | HF长 ⚠️ |
| imaging_plan | To have DEXA. | DEXA. | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |

## ROW 11
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | infiltrating ductal carcinoma with metastatic recurrence to  | ER+/PR+/HER2- infiltrating ductal Carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Recent PET/CT showed increased metastatic activity and a lef | PET/CT showed increased metastatic activity and size of left | vLLM≥ |
| current_meds | Fulvestrant, Denosumab | Faslodex, Denosumab | vLLM≥ |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | Will order PETCT to evaluate Femur and to toes. | Will order PETCT to evaluate Femur and to toes due to worsen | HF长 ⚠️ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue on Faslodex and Denosumab. [REDACTED]-Mycelex [REDA | Continue on Faslodex and Denosumab. [REDACTED]-Mycelex [REDA | ≈ |

## ROW 12
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2+ (IHC 3+/FISH 15.April 10=5.4) invasive ductal  | ER+/PR+/HER2+ invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | MRI brain on 08/15/18 showed multiple very small new CNS les | CT stable, MRI brain on 08/15/18 showed multiple very small  | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | continue herceptin and unspecified agent; continue letrozole | continue herceptin/[REDACTED] [REDACTED], continue letrozole | ≈ |
| imaging_plan | CT CAP every 4 months; will add bone scan with next eval in  | CT CAP every 4 months, will add bone scan with next eval in  | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue herceptin and [REDACTED]; continue letrozole qd; co | continue herceptin/[REDACTED] [REDACTED], continue letrozole | HF长 ⚠️ |

## ROW 14
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+ metastatic breast cancer, HER2- | ER+ breast cancer, HER2- | vLLM≥ |
| Stage_of_Cancer | Metastatic (Stage IV) | Stage IV (metastatic) | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Right breast nodules have slightly increased in size, with t | The cancer is currently stable. No significant changes in im | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | CT CAP and Total Spine MRI scheduled for May; Repeat total s | CT CAP and Total Spine MRI scheduled for May. Repeat total s | vLLM≥ |
| lab_plan | She has been instructed to have her labs drawn every two wee | Labs to be drawn every two weeks. | vLLM≥ |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 20
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade II invasive ductal carcinoma with 1.8 cm | ER+/PR+/HER2- invasive ductal carcinoma (originally ER+/PR+/ | HF长 ⚠️ |
| Stage_of_Cancer | Originally Stage IIA, now metastatic (Stage IV) | now metastatic (Stage IV) | vLLM≥ |
| Distant Metastasis | Yes, to bone | Yes, to bone, lymph nodes | HF长 ⚠️ |
| response_assessment | The plan includes repeat imaging after 3 months on treatment | Not mentioned in note. | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Start letrozole, Rx given. Rx for Palbociclib. She will need | Start Letrozole, Rx given. Rx for Palbociclib. She will need | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | labs including tumor markers. Monthly blood work for Palboci | labs including tumor markers. She will need monthly blood wo | ≈ |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Start Letrozole, Rx given. Rx for Palbociclib. OK to start.  | Start Letrozole, Rx given. Rx for Palbociclib. OK to start.  | ≈ |

## ROW 22
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- invasive ductal carcinoma with metastatic recu | ER+/PR+/HER2- invasive ductal carcinoma (originally ER+/PR+/ | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to bone, chest wall, right infraclavicular and right IM | Yes, to bones, chest wall | vLLM≥ |
| response_assessment | PET scans on 11/03/20 and 04/24/21 showed a good response. T | PET/CT scans on 11/03/20 and 04/24/21 showed a good response | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue letrozole, abemaciclib. Continue arimidex alone. If | If PET CT shows stability, continue arimidex alone. If PET C | vLLM≥ |
| imaging_plan | Pet ct now | I recommend a pet ct now and if stable continue arimidex alo | HF长 ⚠️ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue arimidex alone. If PET CT shows progression, could  | Continue arimidex alone if PET CT is stable. If PET CT shows | vLLM≥ |

## ROW 29
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with mixed m | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IIA (pT1c(m)N1(sn)) | pT1c(m)sn, G2, ER+, PR+, HER2- | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue letrozole. RT planning per [REDACTED], likely pursu | Start letrozole 2.5mg PO daily now. Plan to begin radiation  | ≈ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Start letrozole 2.5mg PO daily now. Calcium supplementation  | Start letrozole 2.5mg PO daily now. Calcium supplementation  | vLLM≥ |

## ROW 30
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER-/PR-/HER2+ grade 2 invasive ductal carcinoma with interme | ER-/PR-/HER2+ invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Clinical stage II-III (inferred from large enhancing right b | Stage II-III (inferred from large enhancing right breast mas | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | The patient will proceed with neoadjuvant chemotherapy, incl | The patient will proceed with neoadjuvant chemotherapy, incl | vLLM≥ |
| imaging_plan | The patient will need a TTE prior to starting chemotherapy. | The patient will need a TTE prior to starting treatment. | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 33
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive lobular carcinoma | ER+/PR+/HER2- invasive lobular carcinoma | vLLM≥ |
| Stage_of_Cancer | Originally Stage IIB, now Stage III | Originally Stage IIB, now Stage IIIA | ≈ |
| Distant Metastasis | Not sure | (空) | **vLLM** |
| response_assessment | No evidence of recurrence on imaging and exam. No palpable m | No evidence of recurrence on exam. No specific imaging or la | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue letrozole daily. Encouraged weight bearing exercise | Continue letrozole daily. | vLLM≥ |
| imaging_plan | Consider MRI brain if [REDACTED] continues. | If continues, consider MRI brain | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 34
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- invasive ductal carcinoma with extensive DCIS | ER+/PR-/HER2- invasive ductal carcinoma (originally ER+/PR l | HF长 ⚠️ |
| Stage_of_Cancer | Stage III | Originally Stage III, now local recurrence (not metastatic) | HF长 ⚠️ |
| Distant Metastasis | Not sure | (空) | **vLLM** |
| response_assessment | Local recurrence in left breast with a 1.7cm invasive ductal | Local recurrence in left breast, 1.7cm IDC on excisional bio | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | curative | adjuvant | vLLM≥ |
| therapy_plan | Continue radiation therapy referral. Resume hormonal therapy | tamoxifen 20mg PO qD, referral to [REDACTED] for chest wall  | ≈ |
| imaging_plan | (同) | (同) | = |
| lab_plan | check labs | (空) | **vLLM** |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |

## ROW 36
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | (同) | (同) | = |
| Stage_of_Cancer | Stage IIIA (pT3N0) | Stage IIIA (inferred from pT3 N0) | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | No evidence of disease recurrence on imaging and exam. PET/C | Not mentioned in note. | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue tamoxifen. Continue weekly Abraxane 100mg/m2 x 12 c | Continue weekly Abraxane 100mg/m2 x 12 cycles. Continue zola | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue weekly Abraxane 100mg/m2 x 12 cycles. Continue zola | Continue weekly Abraxane 100mg/m2 x 12 cycles. Continue zola | vLLM≥ |

## ROW 40
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 1 invasive ductal carcinoma with nuclear | ER 95, PR 5, HER2 2+ FISH negative (1.2) G1 invasive ductal  | HF长 ⚠️ |
| Stage_of_Cancer | Stage II (inferred from pT2 N1mi) | Stage II | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Adjuvant endocrine therapy with letrozole. Continue Prolia f | The patient will derive the greatest benefit from adjuvant e | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Rx for letrozole given. Continue Prolia. | Rx for letrozole given. She is on Prolia now. | ≈ |

## ROW 41
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR weakly+/HER2 1+ by IHC (FISH not available) grade 3 i | ER+/PR-/HER2 negative invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IIIA (inferred from pT2 N2a) | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Patient will start with [REDACTED] Taxol for 12 weeks follow | After completion of chemotherapy, the patient will be recomm | vLLM≥ |
**✅ ALL-WIN**

## ROW 42
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma | PR+ invasive ductal carcinoma, HER2: not tested | vLLM≥ |
| Stage_of_Cancer | Stage IA (inferred from pT1 N0) | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | There is no specific imaging or tumor marker data provided i | On treatment; response assessment not available from current | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue tamoxifen and return to clinic in 4-6 weeks to asse | Begin a 5 year course of tamoxifen therapy. | vLLM≥ |
| imaging_plan | Routine diagnostic mammogram on the day of the next appointm | Routine diagnostic mammogram on the day of the next appointm | ≈ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Begin a 5 year course of tamoxifen therapy. Rx for tamoxifen | Begin a 5 year course of tamoxifen therapy. | vLLM≥ |
**✅ ALL-WIN**

## ROW 44
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with residua | ER+/PR+/HER2- node+ left breast cancer with residual grade 2 | HF长 ⚠️ |
| Stage_of_Cancer | Originally Stage II (inferred from 1 cm residual invasive du | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Surgical pathology after bilateral mastectomies with left br | The patient has residual disease after neoadjuvant therapy a | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Enrolled on a clinical trial for randomized 3 vs 5 weeks of  | She will start an aromatase inhibitor after she completes ra | vLLM≥ |
| imaging_plan | Consider a follow up CT Chest in one year.. PET/CT | Consider a follow up CT Chest in one year. | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | She will start an aromatase inhibitor after completing radia | Patient will start an aromatase inhibitor after completing r | vLLM≥ |

## ROW 46
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 1 invasive ductal carcinoma with extensi | ER+/PR- HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IIB (pT2N1(sn) (inferred from 3.5 cm residual disease, | Stage IIIA (inferred from pT2 N2a) | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Patient has residual 3.5 cm disease with positive margins af | The cancer is currently responding to treatment. The patient | ≈ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Patient will start letrozole now. Addition of abemaciclib di | The patient will start letrozole now. Discussed addition of  | vLLM≥ |
| imaging_plan | MRA of the abdomen in 1 year, due in January 2022.. DEXA sca | Baseline dexa ordered. MRA of the abdomen due in January 202 | ≈ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Will start letrozole now. Rx sent. Continue naproxen 500mg b | Start letrozole now. Continue naproxen 500mg bid, APAP ok if | HF长 ⚠️ |

## ROW 49
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | (同) | (同) | = |
| Stage_of_Cancer | Likely stage 2 (inferred from primary tumor size and nodal i | Likely stage 2 | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | The current plan for adjuvant endocrine therapy is subject t | Patient to be put on tamoxifen, with a request to obtain mor | vLLM≥ |
**✅ ALL-WIN**

## ROW 50
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | HR+ and HER2- metastatic breast cancer with IDC and DCIS com | ER+/PR+/HER2- invasive ductal carcinoma (IDC) and DCIS | vLLM≥ |
| Stage_of_Cancer | Originally Stage IV (T2, N1, M1) | Originally Stage IV (T2, N1, M1), now metastatic (Stage IV) | HF长 ⚠️ |
| Distant Metastasis | Yes, to lung, liver, and bone | Yes, to the lung, lymph nodes, liver, and bone | HF长 ⚠️ |
| response_assessment | Imaging from December 2021 shows metastatic disease under go | Imaging from December 2021 shows metastatic disease under go | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Lupron, letrozole, and ibrance added January 2015. | Current medications include tamoxifen, lupron, letrozole, an | HF长 ⚠️ |

## ROW 52
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade II invasive ductal carcinoma | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IIA (inferred from pT2 N1mi) | Stage II/III | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue radiation therapy referral. [REDACTED] send Zoladex | [REDACTED] send Zoladex prior auth. | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Order [REDACTED] to provide more information on her tumor bi | Order an [REDACTED] to provide more information on her tumor | ≈ |
| medication_plan | (同) | (同) | = |
**✅ ALL-WIN**

## ROW 53
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2+ invasive ductal carcinoma with neuroendocrine  | ER+/PR+/HER2+ invasive ductal carcinoma with neuroendocrine  | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Adjuvant systemic chemotherapy with AC x 4 given q2weeks wit | Adjuvant AC/THP chemotherapy consisting of AC x 4 given q2we | ≈ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Referral to genetic counseling for further evaluation. | Offered a referral to genetic counseling for further evaluat | HF长 ⚠️ |
| medication_plan | Adjuvant AC/THP chemotherapy consisting of AC x 4 given q2we | Adjuvant AC/THP chemotherapy: AC x 4 given q2weeks with grow | vLLM≥ |

## ROW 54
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with extensi | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IV (metastatic) | now metastatic (Stage IV) | HF长 ⚠️ |
| Distant Metastasis | Yes, to left posterior 7th rib and T6 vertebral body | Yes, to bone (left posterior 7th rib and T6 vertebral body) | ≈ |
| response_assessment | The cancer is currently responding with stable disease. Evid | Surgical pathology after neoadjuvant therapy shows 8.2 cm re | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | continue leuprolide every 4 weeks, continue letrozole, refer | continue leuprolide every 4 weeks, continue letrozole, refer | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | continue leuprolide every 4 weeks, continue letrozole, start | Continue leuprolide every 4 weeks, continue letrozole, start | ≈ |

## ROW 57
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | Breast Cancer (TNBC), originally classified as Grade III ade | ER-/PR-/HER2- breast cancer, triple negative | vLLM≥ |
| Stage_of_Cancer | Locally advanced (inferred from treatment with neoadjuvant c | Locally advanced (inferred from description of residual tumo | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | The patient received neoadjuvant chemotherapy with TCH+P fol | The patient received neoadjuvant chemotherapy with TCH+P, fo | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Recommendation to resume trastuzumab if [REDACTED] status is | (空) | **vLLM** |

## ROW 59
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 3 invasive ductal carcinoma with high-gr | ER+/PR+/HER2- invasive ductal carcinoma, high grade DCIS | vLLM≥ |
| Stage_of_Cancer | Stage IIA (inferred from pT2 N0) | Not explicitly stated, inferred as Stage I (T1-2 N0 M0 based | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | No evidence of recurrence either locally or systemically bas | Patient is currently without evidence of recurrence either l | HF长 ⚠️ |
| current_meds | "exemestane ([REDACTED]) 25 mg tablet", "letrozole (FEMARA)  | exemestane ([REDACTED]) | vLLM≥ |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | (同) | (同) | = |
| imaging_plan | Right mammogram due in July. Continue mammograms alternating | Right mammogram due in July, continue mammograms alternating | ≈ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Discontinue Letrozole and wait 2-3 weeks before starting Exe | Discontinue Letrozole and wait 2-3 weeks before starting Exe | vLLM≥ |

## ROW 61
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- (1+) grade 2 invasive ductal carcinoma | ER+/PR+/HER2- (1+) invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage I (inferred from tumor ≤2cm) | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Will likely need [unspecified agent] Dx after surgery to det | Will likely need [REDACTED] Dx after surgery to determine if | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | will likely need [REDACTED] Dx after surgery to determine if | (空) | **vLLM** |
| medication_plan | she will need adjuvant endocrine therapy (Tamoxifen vs Ovari | Patient will need adjuvant endocrine therapy with either Tam | HF长 ⚠️ |

## ROW 64
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | HR+/HER2- invasive ductal carcinoma | ER+/PR+/HER2- invasive ductal carcinoma | ≈ |
| Stage_of_Cancer | Stage IV (metastatic) | Stage IV- IV | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | No specific imaging or tumor marker evidence provided in the | Not yet on treatment — no response to assess. | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | palliative | curative | vLLM≥ |
| therapy_plan | Recommend [REDACTED] followed by taxol then surgery and radi | Currently on [REDACTED] and taxol planned. Recommended [REDA | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | No specific genetic or molecular tests are planned. | (空) | **vLLM** |
| medication_plan | Currently on unspecified agent and taxol planned. If biopsy  | [REDACTED] and taxol planned. | vLLM≥ |

## ROW 65
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER weak positive (2%), PR low positive (7%), HER2 neg (IHC 2 | ER weak positive (2%), PR low positive (7%), HER2 neg (IHC 2 | vLLM≥ |
| Stage_of_Cancer | Stage IB (corrected: pT1 N1mi — micrometastasis only) | Stage II (inferred from pT2 N1 with 0.2mm focus of micrometa | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue taxol. Port placement, chemotherapy teaching sessio | Port placement, chemotherapy teaching session, will start ne | ≈ |
| imaging_plan | research breast MRI | Order TTE | vLLM≥ |
| lab_plan | labs | labs. F/u pending. | HF长 ⚠️ |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Plan to start neoadjuvant chemotherapy with AC/T. Typically, | Neoadjuvant chemotherapy with options including AC/T (doxoru | HF长 ⚠️ |

## ROW 68
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2+ multifocal invasive ductal carcinoma | ER+/PR+/HER2+ multifocal breast cancer | vLLM≥ |
| Stage_of_Cancer | Stage I (inferred from tumor ≤2cm) | Early stage (inferred from multifocal disease with no distan | HF长 ⚠️ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | The cancer has shown a good clinical response after 6 cycles | The cancer is currently responding to treatment. Follow-up M | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | If a lumpectomy is chosen, radiation is required. Post-maste | If she chooses to have a lumpectomy, she should have radiati | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Her sons should be tested for the risk of [REDACTED] anemia  | Her sons should be tested for the risk of [REDACTED] anemia  | HF长 ⚠️ |
| medication_plan | No specific current or future medication plans were detailed | No specific future medication plans, changes to current medi | HF长 ⚠️ |

## ROW 70
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive lobular carcinoma with 1 lymp | ER+/PR+/HER2- invasive lobular carcinoma (left breast); ER+/ | vLLM≥ |
| Stage_of_Cancer | Originally Stage II (inferred from pT4N1 on left side and pT | (inferred from pT4 N1 on left side, pT1 N0 on right side), n | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | MRI shows faint residual non-mass enhancement in both breast | The cancer is currently responding to treatment as evidenced | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Restart letrozole which she had previously tolerated. She is | She is going to have expanders placed prior to radiation. Sh | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | She will restart letrozole which she had previously tolerate | Restart letrozole which she had previously tolerated. Inform | HF长 ⚠️ |

## ROW 72
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- grade 2 invasive ductal carcinoma with focal n | ER+/PR-/HER2- invasive ductal carcinoma with focal neuroendo | vLLM≥ |
| Stage_of_Cancer | Stage IA (pT1cN0(sn)) | pT1cN0(sn) | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue letrozole. Ordered [REDACTED] to evaluate potential | Instructed patient to begin letrozole, prescription ordered. | HF长 ⚠️ |
| imaging_plan | Ultrasound | (空) | **vLLM** |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | (同) | (同) | = |

## ROW 78
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER-/PR-/HER2- grade 3 invasive ductal carcinoma | ER-/PR-/HER2- triple negative breast cancer | vLLM≥ |
| Stage_of_Cancer | Metastatic (Stage IV) | Stage IV (metastatic) | vLLM≥ |
| Distant Metastasis | Yes, to liver and periportal lymph nodes | Yes, to liver and periportal LNs | vLLM≥ |
| response_assessment | Worsening of metastatic disease noted on 08/07/19 Abdominal  | Worsening of metastatic disease with interval enlargement of | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | She is interested in screening for a trial at [REDACTED] inv | She is interested in screening for a trial at [REDACTED] inc | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Patient is interested in screening for the [REDACTED] trial  | Patient is interested in screening for the [REDACTED] trial  | HF长 ⚠️ |
| medication_plan | (同) | (同) | = |

## ROW 80
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 3 invasive ductal carcinoma | ER+/PR+ invasive ductal carcinoma, HER2: not tested | ≈ |
| Stage_of_Cancer | Stage I (inferred from tumor ≤2cm) | (空) | **vLLM** |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Start TC x 4 on 04/11/19, with [REDACTED]. RTC cycle 2 to se | Start TC x 4 on 04/11/19, with 6 weeks of radiation therapy, | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Whole genome sequencing was completed and there are no actio | (空) | **vLLM** |
| medication_plan | Start TC x 4 on 04/11/19, with [REDACTED]. Claritin for 5-6  | Start TC x 4 on 04/11/19, with Claritin for 5-6 days. Discus | ≈ |

## ROW 82
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | (同) | (同) | = |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | We discussed the role of chemotherapy to decrease the risk o | We discussed the role of chemotherapy to decrease the risk o | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue hydrochlorothiazide 12.5 mg tablet daily before lun | Continue current medications including acetaminophen 1000 mg | HF长 ⚠️ |

## ROW 84
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with metasta | ER+/PR- (71-80%, <1%) invasive ductal carcinoma, HER2 equivo | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | MRI brain with contrast on 11/07/2020 showed diffuse irregul | The cancer is currently stable on imaging and exam. MRI brai | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | refer to radiation oncology for consideration of radiation t | Repeat CT CAP now, Repeat LP for CSF cytology, Repeat MRI sp | HF长 ⚠️ |
| imaging_plan | Repeat CT CAP now, Repeat MRI spine to rule out leptomeninge | Repeat CT CAP now, Repeat MRI spine to r/o LMD in the spine | vLLM≥ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue xeloda 1500mg BID for now, continue zolendronic aci | Continue xeloda 1500mg BID for now, could consider starting  | vLLM≥ |

## ROW 85
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- invasive lobular carcinoma with pleomorphic fe | ER+/PR-/HER2- invasive lobular carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to bone, liver, and brain | Yes, to bone, muscle, and brain | ≈ |
| response_assessment | The cancer is currently progressing on first line fulvestran | Disease progressed on first line fulvestrant/palbociclib in  | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Metastatic breast cancer patient will be evaluated for a pha | The patient will be evaluated for a phase 1 trial of [REDACT | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Patient will be evaluated for a phase 1 trial of [REDACTED]+ | She will be evaluated today by Dr. [REDACTED] for a phase 1  | HF长 ⚠️ |
| medication_plan | (同) | (同) | = |

## ROW 87
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 2 invasive ductal carcinoma with a separ | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | [REDACTED] will receive a course of hormonal therapy alone. | [REDACTED] will receive a course of hormonal therapy alone.  | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Ms. [REDACTED] will prefer to receive a course of hormonal t | Ms. [REDACTED] will receive a course of hormonal therapy alo | vLLM≥ |

## ROW 88
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | Invasive ductal carcinoma, ER weak+, PR-, HER2- with metasta | ER+/PR+/HER2- invasive ductal carcinoma, metastatic biopsy E | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | She is on xeloda and I would recommend restaging after 3 mon | No new imaging findings are reported in this visit, and the  | HF长 ⚠️ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue xeloda. Discuss clinical trials using immunotherapy | She is on xeloda and I would recommend restaging after 3 mon | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | She is on xeloda. If progression on xeloda occurs, clinical  | She is on xeloda. If she has HER2 positive disease, she will | HF长 ⚠️ |

## ROW 90
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | Adenocarcinoma of right breast (HCC) | Adenocarcinoma of right breast (HCC) - ER/PR/HER2 status not | HF长 ⚠️ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | Physical exam via video observation shows slight tissue swel | Pathology from initial imaging showed a 3 cm tumor. After tr | ≈ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | continue with cycle 4 of AC in 1 week (dose delay x 1 wk) to | Continue with cycle 4 of AC in 1 week (dose delay x 1 wk) to | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue with cycle 4 of AC in 1 week (dose delay x 1 wk). R | Continue cycle 4 of AC in 1 week, reduce GCSF dose to 50%. A | vLLM≥ |

## ROW 91
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+ invasive ductal carcinoma with metastatic recurrence | ER+/PR+ invasive ductal carcinoma, HER2: not tested | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | MRI pelvis on 11/08/2011 shows moderate increase in size and | MRI pelvis and PET/CT scans show an increase in bone metasta | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue exemestane daily. Continue denosumab for hip pain.; | Continue exemestane daily, continue [REDACTED] [REDACTED] an | HF长 ⚠️ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue lasix 10mg daily with KCL 10Meq daily, continue den | Continue lasix 10mg daily with KCL 10Meq daily, continue den | vLLM≥ |

## ROW 92
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | (同) | (同) | = |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to liver | Yes, to multiple sites | HF长 ⚠️ |
| response_assessment | Liver smaller and feels less tender or bloated. No significa | The cancer is currently stable on treatment. The patient is  | HF长 ⚠️ |
| current_meds | Epirubicin, Denosumab | EPIRUBICIN HCL, DENOSUMAB | HF长 ⚠️ |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Plan cycle#2 D1 Epirubicin 25 mg/m2 D1,8,15 to with 2 days o | Plan cycle#2 D1 Epirubicin 25 mg/m2 D1,8,15 to with 2 days o | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | Labs liver functions, Tumor marker pending | Labs liver functions, tumor marker pending | vLLM≥ |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Plan cycle#2 D1 Epirubicin 25 mg/m2 D1,8,15 to with 2 days o | Cycle #2: D1 Epirubicin 25 mg/m2, D1,8,15 with 2 days of Neu | vLLM≥ |

## ROW 95
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR-/HER2- invasive ductal carcinoma with residual ductal | ER+/PR-/HER2- invasive ductal carcinoma with treatment effec | HF长 ⚠️ |
| Stage_of_Cancer | Stage II (inferred from 2.1cm tumor) | Stage IIA (inferred from pT2 N1a) | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | MRI breast: Interval decrease in irregularly shaped mass wit | The cancer is responding to neoadjuvant therapy with a good  | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Continue pembrolizumab. axilla XRT; patient prefers breast a | Start AC chemotherapy. Axilla XRT; patient is interested in  | vLLM≥ |
| imaging_plan | breast and axilla XRT | Pt wishes to proceed with breast and axilla XRT next. | HF长 ⚠️ |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue prilosec 40mg qd, plan to start capecitabine after  | Continue prilosec 40mg qd. Patient is interested in starting | HF长 ⚠️ |

## ROW 97
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+/PR+/HER2- grade 1 invasive ductal carcinoma with low to  | ER+/PR+/HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | Stage IA (pT1b N0) | pT1bN0(sn) | vLLM≥ |
| Distant Metastasis | (同) | (同) | = |
| response_assessment | (同) | (同) | = |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Strongly recommend adjuvant endocrine therapy with [REDACTED | do not anticipate any need for chemotherapy; no problem with | vLLM≥ |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | Patient wishes to proceed with molecular profiling, e.g. Onc | molecular profiling | vLLM≥ |
| medication_plan | Strongly recommend adjuvant endocrine therapy with [REDACTED | Continue with ongoing GILENYA regimen. No changes to MS medi | vLLM≥ |
**✅ ALL-WIN**

## ROW 100
| Field | vLLM | HF | 判定 |
|-------|------|-----|------|
| Type_of_Cancer | ER+(80%)PR+(50%) HER2- grade 2 invasive ductal carcinoma wit | ER+(80%)PR+(50%)HER2- invasive ductal carcinoma | vLLM≥ |
| Stage_of_Cancer | (同) | (同) | = |
| Distant Metastasis | Yes, to liver and bone | Yes, to liver and multiple sites | HF长 ⚠️ |
| response_assessment | Tumor markers elevated: Cancer Antigen 15-3 at 118 U/mL (<33 | Tumor markers (Cancer Antigen 15-3 and Cancer Antigen 27.29) | vLLM≥ |
| current_meds | (同) | (同) | = |
| goals_of_treatment | (同) | (同) | = |
| therapy_plan | Rec exercise 10 min 3 x a day, Focalin prn and continue with | (空) | **vLLM** |
| imaging_plan | (同) | (同) | = |
| lab_plan | (同) | (同) | = |
| genetic_testing_plan | (同) | (同) | = |
| medication_plan | Continue with treatment, Focalin prn for fatigue. | Focalin prn to address fatigue. | vLLM≥ |
