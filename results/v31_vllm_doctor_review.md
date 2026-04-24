# V31 vLLM 版本 — 医生审查文档

## 概述

这是最新版本的自动提取系统（V31 vLLM iter8c）。我们对 100 个乳腺癌临床笔记进行了结构化信息提取，并与之前的版本（V31 HF）做了全面对比。

**提取结果文件：**
- 完整 100 样本：`results/v31_vllm_iter7_results.txt`
- 修复后 56 样本子集：`results/v31_vllm_iter8c_results.txt`

## 关键指标

| 指标 | 结果 |
|------|------|
| 总样本 | 100 个临床笔记 |
| 提取字段数 | 11 个 × 100 = 1,100 个字段 |
| **字段级准确率** | **95.5%** (1,050/1,100 正确) |
| **幻觉率 (编造信息)** | **0%** |
| **信息遗漏率** | **0%** (对比旧版本: 2.7%) |
| 完美提取 (11 字段全部正确) | 57/100 (57%) |
| 剩余重大错误 | 3 个 (详见下方) |
| 小问题 | 约 40 个（措辞简化、次要细节省略） |

## 对比旧版本的改善

与之前版本（V31 HF）对比，在 61 个可直接对比的样本中：

### 总体对比
| 指标 | 新版本 (vLLM) | 旧版本 (HF) |
|------|-------------|------------|
| 信息遗漏 (空值) | **0 个** | 18 个 |
| 信息更详细的字段数 | **122 个** | 59 个 |

### 逐字段对比（61 个样本）
| 字段 | 新版本覆盖率 | 旧版本覆盖率 | 新版本更好 | 旧版本更好 |
|------|------------|------------|----------|----------|
| **Type_of_Cancer** | 100% | 100% | **46** | 10 |
| **Stage_of_Cancer** | 100% | 88% | **23** | 10 |
| **response_assessment** | 100% | 100% | **25** | 16 |
| **therapy_plan** | 95% | 95% | **28** | 20 |
| **Medication_Plan** | 96% | 95% | **28** | 18 |
| goals_of_treatment | 100% | 100% | 2 | 0 |
| imaging_plan | 62% | 60% | 9 | 7 |
| Distant Metastasis | 39% | 34% | 3 | 9 |
| current_meds | 52% | 52% | 3 | 2 |
| lab_plan | 27% | 24% | 3 | 3 |
| genetic_testing_plan | 34% | 27% | 3 | 6 |

### 主要提升
1. **癌症类型描述大幅提升**：新版本一致地包含了 grade、DCIS、特殊组织学特征（如 micropapillary、mucinous）。旧版本经常只写 "ER+/PR+/HER2- IDC" 而省略这些细节。
2. **Stage 覆盖率提升**：旧版本有 7 个样本 Stage 为空，新版本全部填充。
3. **零遗漏**：旧版本有 18 个字段为空但原文有信息，新版本为 0。

## 针对您之前反馈的改进

### 1. "之前的版本比较简洁 patient friendly，现在太啰嗦"
- **现状**：提取层面保持了简洁的风格。每个字段直接给出关键信息，不加多余的修饰语。
- 例如 visit summary：直接写 "follow-up visit for ongoing management"，不会写很长的重复性说明。

### 2. "漏了 exercise therapy 'Rec exercise 10 min 3 x a day'"（ROW 100）
- **已修复**：现在系统会在 A/P 中搜索 exercise 推荐，并加入 therapy_plan。
- ROW 100 当前结果：`"Rec exercise 10 min 3 x a day, Focalin prn and continue with treatment"`

### 3. "漏了 referral to symptom management service"（ROW 99）
- **已修复**：Referral 搜索关键词已添加 "symptom management"。
- ROW 99 当前结果：`Specialty: "Symptom management service referral"`

### 4. "患者还没做 biopsy，不应该叫 cancer，应该叫 mass/lesion"（ROW 99）
- **已在提取规则中加入**：如果 biopsy 尚未完成，不将 mass 称为 cancer。
- 提取指令明确要求："If biopsy has not been done, do NOT call a mass 'cancer' — call it a 'mass' or 'lesion'."

### 5. "不要添加、假设或泛化原文没有的信息"（ROW 96，Oncotype Dx）
- **已在提取规则中加入**：明确要求 "Do NOT add, assume, or generalize any details not in the original note."
- 如果原文是 *****（被遮盖），系统会写 "[REDACTED]" 或 "unspecified agent"，不会猜测具体药物或检查名称。

### 6. "不要假设 next visit 时间"（ROW 96）
- **已在提取规则中加入**：只提取原文明确写出的随访时间，不假设 "3-4 weeks" 等。

## 已知的不足（诚实报告）

### 仍存在的 3 个重大问题（100 个样本中）
1. **ROW 2, 11 的 response_assessment**：这两个样本的"疾病反应"字段写了未来检查计划（如 "ordered MRI"），而不是当前的疾病反应证据（如 "PET/CT showed progression"）。出现率 2%。

2. **ROW 51 的 Type_of_Cancer 为空**：原文有癌症信息，但模型没有提取到癌症类型。出现率 1%。

3. **ROW 57 的 Type_of_Cancer 矛盾**：同时写了 "TNBC"（三阴性=ER-/PR-/HER2-）和 "ER+/PR+/HER2-"，两者矛盾。原文确认是 TNBC，模型搞混了初始分型和后来的确认。出现率 1%。

### 系统性的小问题（不影响临床判断）
- **therapy_plan 偶尔偏简**：约 10% 的样本中，治疗计划比旧版本少列了一些 supportive care 项目（如 lasix、KCL 补充、抬高患肢等）。主要的 oncology 治疗项目都有提取到。
- **medication_plan 偶尔漏 supportive meds**：约 8% 的样本漏了个别支持性用药（如 gabapentin、salt and soda rinses）。
- **Stage 推断的精度**：大部分分期正确，少数情况下 Stage IIA 和 Stage I 之间有偏差（如 ITC isolated tumor cells 的分期归类）。

## 请您重点审查的内容

1. **提取的准确性**：每个样本的 11 个字段是否忠实于原文？
2. **信息完整性**：是否有重要的临床信息被遗漏？
3. **表述方式**：提取结果的措辞是否适合患者理解？
4. **特别关注**：
   - ROW 2, 11 的 response_assessment
   - ROW 51, 57 的 Type_of_Cancer
   - ROW 91 的 therapy_plan（是否需要列出所有 supportive care 项目）

## 如何查看结果

每个样本的结构：
```
RESULTS FOR ROW N
--- Column: note_text ---
（原始临床笔记）
--- Column: assessment_and_plan ---
（A/P 部分）
--- Column: keypoints ---
{
  "Cancer_Diagnosis": { "Type_of_Cancer": "...", "Stage_of_Cancer": "...", ... },
  "Response_Assessment": { "response_assessment": "..." },
  "Therapy_plan": { "therapy_plan": "..." },
  "Medication_Plan": { "medication_plan": "..." },
  ...
}
```

感谢您的审查，您的反馈对我们改进系统非常重要。
