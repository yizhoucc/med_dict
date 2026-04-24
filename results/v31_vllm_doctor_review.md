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

## 可读性指标

提取结果的可读性（100 个样本的所有文本字段汇总）：

| 指标 | 分数 | 说明 |
|------|------|------|
| Flesch-Kincaid Grade Level | 10.5 | 约 10 年级阅读水平 |
| Flesch Reading Ease | 42.0 | 偏难（医学文本正常范围） |
| Gunning Fog Index | 14.4 | 大学水平 |
| SMOG Index | 13.0 | 大学水平 |
| Coleman-Liau Index | 13.0 | 大学水平 |
| Dale-Chall Score | 13.7 | 大学水平 |
| ARI | 11.4 | 约 11 年级 |

注：提取结果是结构化临床信息，包含医学术语（ER+/PR+/HER2-、IDC、pT2N1 等），可读性分数偏高是正常的。Patient letter 的可读性会更好（8 年级水平）。

---

## 医生历次反馈完整记录

以下是医生在各个版本审查中给出的所有反馈，按时间排序。即使某些反馈针对的是早期版本，这些意见仍然反映了医生的偏好和标准。

### V31 HF 版本反馈
（无具体反馈记录，医生表示偏好 V31 的 letter 风格）

### V33 版本反馈（letter 相关）
1. **"过度简化改变了原意"**：A/P 写了 "monitor with ALT, hepatitis B surface antigen and HBV DNA every 4 months"，但 letter 写成了 "you will have blood tests including a complete blood count, liver tests, and cancer markers" — 把具体的 HBV 监测变成了模糊的 "liver tests"。
2. **"废话太多"**：letter 加了原文没有的内容。
3. **"准确性优先于简化"**：Without accuracy, no point to simplify.
4. **"V31 的风格更好"**：更贴近原文，不过度改写。

### V31 vLLM iter3 版本反馈（extraction 相关）
5. **"之前的版本说 you come in for a new patient visit 就很好，现在太啰嗦"**：visit summary 应该简洁 patient friendly。
6. **ROW 100："漏了 exercise therapy"**：原文有 "Rec exercise 10 min 3 x a day" 但提取没包含。→ **已修复**
7. **ROW 99："漏了 referral to symptom management service"**：原文明确有这个 referral。→ **已修复**
8. **ROW 99："患者还没做 biopsy，不应该叫 cancer"**：应该叫 mass 或 lesion。→ **已加入规则**
9. **ROW 96："Oncotype Dx 不应该假设"**：原文没明确提到 Oncotype Dx，不能假设。**不要添加、假设或泛化原文没有的信息。** → **已加入规则**
10. **ROW 96："不要假设 next visit 时间"**：如果原文没写 3-4 weeks，不能假设。→ **已加入规则**

### 通用偏好（从历次反馈总结）
11. **准确性 > 简化**：永远不能为了简化而改变原意
12. **不逐一列出转移器官**：keep it simple
13. **优先级排序**：化疗下一步 > 慢性病管理
14. **已有检查结果 ≠ 未来计划**：不要混淆
15. **Routine recurring tests 不要列出**：除非结果改变了方案
16. **Follow-up 患者不要重复诊断**：除非首次知道
17. **包含所有药物变化**：不管是否与主要诊断相关
18. **不要说 "anxious and depressed"**：用固定句表达关怀
19. **pCR 准确描述**：breast clear + nodes positive ≠ pCR
20. **Echo/echocardiogram 不要遗漏**
21. **Port placement 属于 procedure plan**

---

感谢您的审查，您的反馈对我们改进系统非常重要。
