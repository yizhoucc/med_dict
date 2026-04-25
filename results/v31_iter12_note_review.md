# V31 iter12 — 对照原文逐字段审查

> 56 samples × 11 fields = 616 field-instances
> 审查标准：每个 field 的值是否忠实于原文，不多不少

## 状态
- 审查中: ROW 1 开始
- P0: 0, P1: 0, P2: 0

---

## ROW 1
读原文: 56yo, Stage IIA→metastatic. G2 ER+PR+ HER2-. Mets to lungs/peritoneum/liver/ovary. 无current meds. Declined tamoxifen. A/P: palliative, biopsy planned, ibrance+***** if HR+/HER2-.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | ER+/PR+ grade 2 IDC with met recurrence, HER2- | 原文G2 ER+PR+ HER2- ✓ | ✅ |
| Stage | Originally Stage IIA, now metastatic (Stage IV) | ✓ | ✅ |
| DistMet | Yes, to lungs, peritoneum, liver, ovary | ✓ | ✅ |
| response | CT 12/24/2019 widespread metastases... (详细) | ✓ 忠实于CT impression | ✅ |
| current_meds | (空) | 原文"No current outpatient medications" ✓ | ✅ |
| goals | palliative | ✓ | ✅ |
| therapy | ibrance + [REDACTED] if HR+/HER2- | ✓ | ✅ |
| imaging | Brain MRI | 漏了bone scan(原文"MRI of brain and bone scan") | P2 |
| lab | ordered MRI of brain and bone scan as well as labs | 混入了MRI和bone scan(这是imaging不是lab) | P2 |
| genetic | biopsy scheduled to confirm HR/HER2 | biopsy是procedure不是genetic testing | P2 |
| medication | ibrance + unspecified; **also: tamoxifen** | **P1**: 原文说patient DECLINED tamoxifen! POST hook误加 | **P1** |

**P0:0 P1:1 P2:3**

## ROW 2
读原文: 44yo TNBC metastatic. Irinotecan C3D1. Bone mets, chest wall, possibly intracranial. A/P: change irinotecan dose, doxycycline for cellulitis, XRT referral, labs monitoring.

| Field | 值 | vs 原文 | 判定 |
|-------|-----|---------|------|
| Type | TNBC with metastatic recurrence | 原文"triple negative IDC" ✓ | ✅ |
| Stage | Originally Stage IIB, now Stage IV | ✓ | ✅ |
| DistMet | Yes, liver, bone, chest wall, possibly intracranial | ✓ | ✅ |
| response | MRI bone marrow lesions... no specific evidence | 漏了"chest wall more tender, worrisome for infection"和"back pain worse, possibly PD" | P2 |
| current_meds | irinotecan | ✓ | ✅ |
| goals | palliative | ✓ | ✅ |
| therapy | irinotecan dose change + Rad Onc referral | ✓ | ✅ |
| imaging | Scans in 3 months, MRI brain if worse | ✓ | ✅ |
| lab | ALT/HBsAg/HBV DNA q4months. Scans in 3 months | "Scans"是imaging不是lab | P2 |
| genetic | None planned | ✓ | ✅ |
| medication | [REDACTED] 30mg TID, Flexeril, oxycodone, Cymbalta, Effexor; also: doxycycline | 漏了500ml NS IV, 40mEq potassium, 1 unit pRBC | P2 |

**P0:0 P1:0 P2:3**

