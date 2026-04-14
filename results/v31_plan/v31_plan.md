# V31 改进计划

## 基于 v30 审查 (93 P2, 1 P1) 的改进方案

### 目标
- 减少 P2 数量 ~30%（93 → ~65）
- 消除 P1（sarcoidosis 误分类）

---

## 改进项目（按影响大小排序）

### 1. POST-PROCEDURE-CLEAN hook（新增）— 预计减少 ~12 P2
**问题**: procedure_plan 字段混入 labs/imaging/chemo/RT/pathology review 等非手术项目（~12 ROWs）
**影响 ROWs**: 7, 8, 17, 20, 34, 36, 53, 54, 57, 63, 80, 88, 92

**方案**: 新增 POST hook，在 procedure_plan 提取后自动清除非 procedure 内容
```python
# POST-PROCEDURE-CLEAN
# 检测 procedure_plan 中包含的非 procedure 关键词
NON_PROCEDURE_PATTERNS = [
    # Labs
    r'\bcheck labs\b', r'\bblood work\b', r'\blab draw\b', r'\bestradiol\b', r'\bFSH\b',
    r'\bhormone studies\b', r'\btumor markers?\b',
    # Imaging
    r'\b(MRI|CT|PET|DEXA|bone scan|ultrasound|mammogram|echocardiogram|TTE|doppler|X-ray)\b',
    # Chemo/Systemic therapy
    r'\b(chemotherapy|Abraxane|taxol|paclitaxel|carboplatin|AC|TCHP|THP|letrozole|tamoxifen|'
    r'Epirubicin|Neupogen|capecitabine|Xeloda|doxorubicin|gemcitabine|docetaxel|palbociclib|'
    r'ibrance|ribociclib|everolimus|fulvestrant|exemestane|anastrozole|arimidex|Herceptin|'
    r'pertuzumab|trastuzumab|T-DM1|abemaciclib|olaparib|pembrolizumab|durvalumab)\b',
    # RT
    r'\b(radiation|XRT|RT|GK|SRS|radiosurgery|gamma knife)\b',
    # Non-procedure
    r'\bacupuncture\b', r'\breferral\b', r'\bconsult\b', r'\bpathology review\b',
    r'\bgenetic testing\b', r'\bFoundation One\b', r'\bOncotype\b',
]
# 如果 procedure_plan 中所有内容都匹配非 procedure 模式，则替换为 "No procedures planned."
# 如果部分匹配，保留未匹配的部分（如 "port placement"）
```

### 2. POST-RESPONSE-PRETREATMENT hook（新增）— 预计减少 ~9 P2
**问题**: response_assessment 对 pre-treatment 患者写 "On treatment" 而非 "Not yet on treatment"（~9 ROWs）
**影响 ROWs**: 20, 29, 40, 49, 65, 66, 72, 80 等

**方案**: 新增 POST hook，检测常见 pre-treatment 线索
```python
# POST-RESPONSE-PRETREATMENT
# 如果 response_assessment 包含 "On treatment" 但:
# 1. current_meds 为空或只有刚开的处方
# 2. summary 包含 "consultation", "new patient", "initial visit"
# 3. recent_changes 包含 "Rx given", "prescription ordered", "will start"
# → 则替换为 "Not yet on treatment — no response to assess."
```

### 3. POST-LETTER-REDACTED hook（改进）— 预计减少 ~5 P2
**问题**: Letter 中 [REDACTED] 被解读为 "a medication"（~5 ROWs: 3, 7, 10, 30, 37）
**影响**: [REDACTED] 出现在机构名/地名/药名位置时被误解

**方案**: 改进 letter_generation prompt + POST hook
```
# 在 letter_generation.yaml rule 15 中添加:
- If [REDACTED] appears where a FACILITY or LOCATION name should be (e.g., "proceed with treatment at [REDACTED]", "treated at [REDACTED]"), write "your clinic" or "your treatment center" — do NOT write "a medication".
- Common patterns: "at [REDACTED]" (facility), "in [REDACTED]" (location), "[REDACTED] Hospital" (facility)
```

### 4. Stage 推断增强 — 预计减少 ~5 P2
**问题**: Stage "Not mentioned" 当可从 pathology 推断时（~5 ROWs: 17, 42, 52, 87 等）

**方案**: 在 extraction.yaml Cancer_Diagnosis prompt 中增强 stage 推断规则
```
# 添加到 Stage_of_Cancer 定义:
- If stage is not explicitly stated but you have BOTH tumor size (T) AND lymph node status (N):
  - T1 (≤2cm) + N0 = Stage I
  - T1 + N1mi (micromet) = Stage IIA
  - T2 (2-5cm) + N0 = Stage IIA
  - T2 + N1 = Stage IIB
  - T3 (>5cm) or N2 (4-9 LN+) = Stage III
  - M1 (distant metastasis) = Stage IV
  Do NOT leave stage empty if these components are available.
```

### 5. Lab_plan 完整性 — 预计减少 ~3 P2
**问题**: lab_plan 漏掉 A/P 中明确提到的 labs（~5 ROWs: 27, 34 等）

**方案**: 在 plan_extraction.yaml Lab_Plan prompt 中添加:
```
# 添加到 Lab_Plan:
- IMPORTANT: Search the ENTIRE A/P section for ANY mention of labs or blood tests:
  - "check labs", "obtain UA", "CBC", "tumor markers", "labs including", "blood work"
  - "repeat [lab test] in [timeframe]"
  If ANY lab is mentioned in the A/P, it MUST be in lab_plan.
```

### 6. P1 预防: Sarcoidosis/非癌症性 LAD 识别
**问题**: ROW 46 — sarcoidosis（endobronchial biopsy 确认）被误分类为癌症转移
**影响**: 1 P1（Stage IV + palliative 全部错误）

**方案**: 在 extraction.yaml Cancer_Diagnosis 中添加:
```
# 添加到 Metastasis/Distant Metastasis:
- CRITICAL: If the note mentions that a lymph node biopsy showed NON-MALIGNANT findings (e.g., "granulomatous inflammation", "sarcoidosis", "reactive changes", "benign"), that lymph node is NOT a cancer metastasis — even if PET/CT showed it as hypermetabolic.
- Sarcoidosis, granulomatous disease, and reactive lymphadenopathy are NOT cancer metastases.
- Always check if enlarged/hypermetabolic lymph nodes were BIOPSIED and what the biopsy showed.
```

---

## 不修改的项目（ROI 低或 prompt 已足够详细）
- **current_meds 遗漏** (~4 P2): 涉及复杂判断（外院用药、刚开处方），prompt 已有规则
- **medication_plan 内容不当** (~3 P2): 边缘案例，难以用规则解决
- **Referral 遗漏** (~4 P2): 已有详细 prompt，多为 genetic_testing_plan 和 Referral 字段重叠

## 实施顺序
1. POST-PROCEDURE-CLEAN hook（代码修改 run.py）
2. POST-RESPONSE-PRETREATMENT hook（代码修改 run.py）
3. Letter [REDACTED] prompt + POST hook（prompt + 代码）
4. Stage 推断增强（prompt 修改）
5. Lab_plan 完整性（prompt 修改）
6. Sarcoidosis/P1 预防（prompt 修改）

## 预期结果
- P2: 93 → ~60（减少 ~35%）
- P1: 1 → 0
- P2 rate: 1.52 → ~1.0/sample
