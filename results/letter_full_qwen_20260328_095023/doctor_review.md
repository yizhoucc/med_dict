# Letter v6 医生审查报告 (61 samples)

## 审查目标
以临床医生视角逐个审查每封患者信件：
1. **内容准确**：原文 → extraction → letter 三级链条，确保无编造
2. **通俗易懂**：适合患者阅读，重点覆盖，无多余细节

## 严重等级
- P0: 幻觉/编造 | P1: 重大错误 | P2: 小问题

## 数据
`results/letter_full_qwen_20260328_095023/progress.json` (letter v6, 61 samples)

## 对比基线
letter v5 审查 (`results/letter_full_qwen_20260328_075437/doctor_review.md`): P0=0, P1=2, P2~25

---

## 逐 Sample 审查

### SAMPLE 0 (coral 140) — 新诊转移性ER+乳腺癌
**原文**：56岁，ER+/PR+/HER2-，Stage IIA(2013)→现转移（肺/腹膜/肝/卵巢）。活检右腋窝。脑MRI+骨扫描。Integrative Medicine转诊。Full code。目标palliative。患者very scared/distressed。

**信件逐句**：
1. "first consultation" ✓
2. "grows in response to hormones (estrogen)" ✓ — 但未单独说HER2-（句7中提到"does not have extra HER2 protein"作为条件，部分覆盖）
3. "originally at an early stage (Stage IIA) but has now spread...making it an advanced stage (Stage IV)" ✓ — **语法自然流畅，v5的断裂句修复成功**
4. "lungs, the lining of your abdomen (belly area), liver, ovaries" ✓ — peritoneum通俗解释 ✓
5. "widespread cancer...mass near your right armpit" — 与句4有重叠（都说了扩散位置）P2
6. "not currently taking any medications" ✓
7. "If biopsy shows...hormone-sensitive and does not have extra HER2 protein...ibrance and another medication" ✓ — 条件性计划正确。"another medication"比v5的"a specific treatment"自然
8-9. 活检计划和随访 ✓
10. 情绪支持 ✓

**遗漏**：Advance care (Full code)、Specialty (Integrative Medicine)、脑MRI+骨扫描（提取遗漏，非信件问题）
**总评**：P0=0, P1=0, P2=1（句4-5轻微重叠）。Stage IV语法改善显著。

---

### SAMPLE 1 (coral 141) — 转移性TNBC，irinotecan第3周期
**原文**：44岁，TNBC，Stage IV（肝/骨/胸壁）。Irinotecan改为每两周。胸壁感染可能。背痛加重（可能PD）。严重贫血(Hgb 7.7)。低钠/低钾。输血+NS+KCl。Rad Onc急诊。Effexor增量。社工/家庭健康。2周随访。

**v5→v6改善**：
- Stage IV语法 ✓："advanced stage, **meaning** it has spread"（v5是"making it advanced stage, the cancer has spread"）
- 剂量简化 ✓："change your irinotecan to every other week and increase the dose"（v5是"150mg/m2 days 1 and 15 every 28 days"）
- 2周随访补上 ✓
- Coverage 74%→84%

**逐句**：TNBC解释 ✓。转移部位 ✓。Lab异常值方向正确 ✓。胸壁感染+背痛 ✓。Doxycycline+输血 ✓。Rad Onc ✓。3月扫描+脑MRI ✓。Hep B监测 ✓。社工+家庭健康 ✓。情绪支持 ✓。
P2=1："cancer is progressing"说得比原文更肯定（原文说"possibly due to PD"）

**总评**：P0=0, P1=0, P2=1。比v5显著改善。

---

### SAMPLE 2 (coral 142) — Stage IIA ER+/HER2-，新会诊
受体 ✓。"early stage" ✓。化疗/手术/放疗讨论 ✓。基因检测 ✓。PET随访 ✓。
**总评**：P0=0, P1=0, P2=0。

---

### SAMPLE 4 (coral 144) — 复发转移ER+/HER2-随访
"brain is normal, but there are some changes in your neck and arm" — 好坏都报 ✓（Rule 17生效）。v5的无关肌酐值消失 ✓（Rule 19生效）。用药/Rad Onc/CT+骨扫描/月度lab ✓。
**总评**：P0=0, P1=0, P2=0。**v5改善**。

---

### SAMPLE 5 (coral 145) — 早期ER+/HER2-，letrozole启动
受体 ✓。手术 ✓。Letrozole+zoladex ✓。"cure the cancer" ✓。Estradiol monthly ✓。3个月随访 ✓。
**总评**：P0=0, P1=0, P2=0。

---

### SAMPLE 6 (coral 146) — 转移性HER2+，second opinion
"does not respond to hormones and has extra HER2 protein" ✓。"a medication, Herceptin, and Taxotere" — 占位符比v5的"a specific treatment"自然 ✓。"slightly grown...However, mediastinum is stable" — 好坏并报 ✓。"no signs of cancer in your brain" ✓。
P2=1："We will recheck **a medication** before starting" — [REDACTED]替换后句意不通，应该是"recheck test results"。
**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 7 (coral 147) — Stage III HER2+/ER-，术后辅助治疗
受体 ✓。"locally advanced" ✓。AC→T-DM1→放疗 ✓。Port ✓。
**P1=1**："Tests show that the cancer is no longer present in your breast, **but it was found in some lymph nodes**" — 原文术后病理显示**无残留肿瘤**（pCR），包括淋巴结也是阴性的。信件说"found in some lymph nodes"与病理结果矛盾，给患者一个淋巴结还有癌细胞的错误印象。
**总评**：P0=0, **P1=1**, P2=0。

---

### SAMPLE 8 (coral 148) — Stage II HR+/HER2-，术后恢复
受体 ✓。手术结果 ✓。"radiation therapy and then letrozole" ✓。"fosamax for bone protection" ✓。"full code, which means you want all possible life-saving measures" — Advance care有意义且解释清楚 ✓。
P2=1：列出4个止吐/支持药物的通用名+品牌名过于详细。
**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 9 (coral 149) — Stage II HR+/HER2-，术后随访
"grows in response to hormones (estrogen); does not have extra HER2 protein" ✓ — **无"— in plain language:"标记泄漏** ✓。手术恢复 ✓。放疗计划 ✓。DEXA ✓。"full code...all possible life-saving measures" ✓。
**总评**：P0=0, P1=0, P2=0。v5标记泄漏修复确认。

---

### SAMPLE 10 (coral 150) — 转移性骨转移，Faslodex治疗中
"infiltrating ductal carcinoma, which means it started in the milk ducts" ✓ — **医学词汇表生效**。颌骨/腿部症状 ✓。Faslodex+Denosumab ✓。鹅口疮药物 ✓。PETCT计划 ✓。Coverage 85% ✓。
**总评**：P0=0, P1=0, P2=0。

---

### SAMPLE 11 (coral 151) — 转移性ER+/PR+/HER2+，脑/肺/骨转移 ★
**v5 P1修复确认**："cancer is stable, **but a brain scan showed two small new spots**" ✓✓✓ — v5说"stable, no new problems"遗漏了脑部进展。现在好坏并报，准确传达。
Herceptin+letrozole ✓。Gamma Knife ✓。CT/MRI/骨扫描计划 ✓。6周随访 ✓。100%归因链 ✓。
**总评**：P0=0, P1=0, P2=0。**v5 P1完全修复。**

---

### SAMPLE 13 (coral 153) — 转移性ER+，低剂量化疗 [重新审查]
**原文对照**：58岁，de novo转移性ER+(99%)/PR+(25%)/HER2-(FISH阴)，骨/肝/LN。已停palbociclib/fulvestrant。在墨西哥接受低剂量化疗([药]+Gem+Docetaxel)+免疫疫苗+pamidronate。骨药后寒战减量。背部僵硬改善。CA 27.29从193→48持续下降。Labs: Ca 8.7(L), Cl 109(H)。Plan: CT+脊柱MRI 5月, 脊柱MRI 6周, labs每2周, PT 3/12, RTC after scans。
**逐句**：受体 ✓。转移部位 ✓。Ca低/Cl高正确 ✓。化疗方案+疫苗+pamidronate ✓。停palbociclib/fulvestrant ✓。骨药寒战 ✓。背部改善 ✓。影像/lab/随访计划 ✓。
P2=1：CA 27.29下降趋势（193→48好消息）未告知。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 16 (coral 156) — 早期ER+/PR+/HER2-，辅助治疗讨论 [重新审查]
**原文对照**：53岁，左乳IDC 0.8cm，Grade 2，ER+>95%/PR+>95%/HER2-（IHC 0），LN 0/5。10-15%复发风险。Tamoxifen vs AI（取决于绝经状态）。放疗。DXA。遗传学。营养师。
**逐句**：受体完整解释 ✓。左乳+未扩散 ✓。Tamoxifen vs AI ✓。放疗 ✓。DXA ✓。遗传学 ✓。营养师 ✓。随访 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 17 (coral 157) — Stage I ER+/HER2-，辅助内分泌5-10年 [重新审查]
**原文对照**：左乳IDC 8mm + 包裹性乳头状癌，ER+/HER2-。强烈建议辅助内分泌5-10年。患者不愿化疗→不做分子分析。DEXA。Rad Onc±XRT。
**逐句**：受体 ✓。"early stage" ✓。"adjuvant endocrine therapy...5 to 10 years" ✓。DEXA ✓。Rad Onc ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 19 (coral 159) — 转移性ER+/HER2-，letrozole+palbociclib [重新审查]
**原文对照**：75岁，2009年Stage I ER+/HER2- IDC→双侧乳房切除+5年tamoxifen。现转移复发至骨/LN（肺结节待确认）。Plan: letrozole+palbociclib, denosumab牙检后, Rad Onc, MRI脊柱+CT CAP, Foundation One, 月度血检, ~1月随访。
**逐句**：受体 ✓。Denosumab牙检 ✓。Rad Onc ✓。影像 ✓。Foundation One ✓。月度血检 ✓。
P2=2：（1）"blood tests from 2013 show a normal glucose level" — 2013年旧值（Rule 19未遵守，根因是extraction）；（2）letrozole句6/10重复。
**总评**：P0=0, P1=0, P2=2。

### SAMPLE 21 (coral 161) — 转移性ER+/HER2-，abemaciclib停药后 [重新审查]
**原文对照**：72岁，多段病史。Abemaciclib因肺炎停用。Letrozole→anastrozole因皮疹。Plan: PET CT→稳定继续arimidex→进展faslodex+[突变药]。
**逐句**：转移部位 ✓。Abemaciclib停药+原因 ✓。条件性计划 ✓。未来选项 ✓。
P2=1：占位符"a medication if you have a specific mutation"。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 26 (coral 166) — 骨转移稳定 [重新审查]
**原文对照**：HR+ MBC骨转移稳定。Continue [药]+zoladex+femara。背痛2周评估→MRI。UTI→UA。瘀伤→CBC。
**逐句**：稳定 ✓。用药 ✓。背痛MRI ✓。瘀伤血检 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 28 (coral 168) — 早期ER+/HER2-，Oncotype低风险 [重新审查]
**原文对照**：59岁，多灶Grade 2 IDC，SLN微转移，[基因检测]低风险→不推荐化疗。Letrozole。Re-excision。放疗。9月手术。
**逐句**：受体 ✓。微转移通俗化 ✓。letrozole ✓。钙VitD ✓。骨密度 ✓。放疗 ✓。9月手术 ✓。
P2=1：letrozole句5/6重复。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 29 (coral 169) — Stage II-III HER2+，新辅助计划 [重新审查]
**原文对照**：64岁，ER-/HER2+ IDC。PET无转移。新辅助化疗。Mediport。TTE。
**逐句**：受体 ✓。"locally advanced" ✓。"chemotherapy and biological therapies" ✓。Mediport ✓。Echo ✓。Coverage 92% ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 32 (coral 172) — ER+ ILC，letrozole随访 [重新审查]
**原文对照**：63岁，ER+ ILC，S/p双侧乳切+化疗+letrozole。无复发。Continue letrozole>5年。
**逐句**：无复发 ✓。Letrozole ✓。钙VitD ✓。NSAIDs ✓。6个月 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 33 (coral 173) — ER+/PR-/HER2-，局部复发 [重新审查]
**原文对照**：ER+/PR-，局部复发。Arimidex→tamoxifen。胸壁RT。Labs。6个月。
**逐句**：受体 ✓。复发 ✓。药物转换 ✓。RT ✓。Labs ✓。6个月 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 35 (coral 175) — ER+/HER2- Grade III，Abraxane化疗 [重新审查]
**原文对照**：27岁，pT3N0 Grade III mixed ductal+mucinous。Taxol过敏→Abraxane。Doppler DVT。Rad Onc。
**逐句**：Abraxane转换 ✓。Doppler ✓。低RBC/Hgb ✓。Rad Onc ✓。2周 ✓。
P2=1："mucinous carcinoma"未解释（prompt要求"a type of cancer that makes mucus"）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 36 (coral 176) — TNBC Stage IIA [重新审查]
**原文对照**：Stage IIA TNBC。dd AC+Taxol。无激素/放疗。生活方式。Full code。
**逐句**：TNBC ✓。"adjuvant chemotherapy to help prevent coming back" ✓。Full code ✓。生活方式 ✓。
P2=1："a medication center"（[REDACTED]替换不当，应为"a treatment center"）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 39 (coral 179) — Stage II ER+/HER2-，letrozole [重新审查]
**原文对照**：62岁+MS，Stage 2 ER+/HER2-。Letrozole。Prolia。DEXA。副作用讨论。
**逐句**：受体简化 ✓。Letrozole**只提一次** ✓（v5修复确认）。副作用 ✓。Prolia ✓。DEXA ✓。3个月 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 40 (coral 180) — ATM突变，AC-Taxol [重新审查]
**原文对照**：32岁ATM，ER+/HER2-，3cm+1LN微转移。AC-Taxol。Port。卵巢抑制。
**逐句**：全部要点 ✓。P0=0, P1=0, P2=0。

### SAMPLE 41 (coral 181) — PR+ IDC，tamoxifen [重新审查]
**原文对照**：PR+ IDC。S/p放疗。Tamoxifen 5年。乳腺X线。
**逐句**："cancer started in the milk ducts" ✓。Tamoxifen ✓。乳腺X线 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 42 (coral 182) — TNBC Stage I [重新审查]
**原文对照**：TNBC Stage I。双侧乳切+ALND。Taxol+carboplatin。
**逐句**：TNBC ✓。化疗 ✓。手术恢复 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 43 (coral 183) — ER+/HER2-，BRCA1，残留病变 [重新审查]
**原文对照**：33岁，BRCA1，残留IDC。临床试验（3vs5周放疗）。AI after radiation。BSO。
P2=2：（1）"bilateral salpingo-oophorectomy (BSO)"未用通俗语言解释；（2）"Norco April 325 mg"原始药名泄漏。
**总评**：P0=0, P1=0, P2=2。

### SAMPLE 45 (coral 185) — ER+/PR-/HER2-，margin阳性 [重新审查]
**原文对照**：48岁，pT2N2+extranodal extension。Positive margins。Letrozole。再切。Abemaciclib after radiation。
P2=1：letrozole句5/7重复。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 48 (coral 188) — ER+/PR+/HER2-，术前讨论 [重新审查]
**原文对照**：ER+/PR+/HER2-，Stage II。1月6日手术。Tamoxifen。放疗。配偶代理决策人。
P2=1："does not have extra HER2 protein"在同一句中重复两遍。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 49 (coral 189) — HR+/HER2-，Stage IV [重新审查]
**原文对照**：de novo Stage IV。Ibrance/xgeva/letrozole。"Under great control" but左乳进展。
**逐句**：平衡报道 ✓。用药 ✓。Mastectomy考虑 ✓。遗传学 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 51 (coral 191) — 35岁ER+/HER2-，初诊 [重新审查]
**原文对照**：35岁，ER+/PR+/HER2-，S/p partial mastectomy+SLN。生育力保存。CT+骨扫描。基因检测。
**逐句**：生育力保存 ✓。影像 ✓。基因检测 ✓。3周 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 52 (coral 192) — ER+/PR+/HER2+，Stage II/III [重新审查]
**原文对照**：HER2+。Seroma。AC→taxol→trastuzumab/pertuzumab→Arimidex 10yr。放疗。遗传咨询。
P2=1：化疗方案细节过多（"AC x 4 q2wk + taxol 12wk + trastuzumab 1yr"）+ readability 10.1。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 53 (coral 193) — ER+/PR-/HER2-，骨转移 [重新审查]
**原文对照**：骨转移Stage IV。Leuprolide/letrozole/zoledronic acid。Palbociclib after radiation。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 56 (coral 196) — TNBC局部晚期 [重新审查]
**原文对照**：TNBC（重新分类）。S/p mastectomy，3.7cm残留。AC x 4。放疗。遗传咨询。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 58 (coral 198) — ER+/PR+/HER2- Stage I，换药 [重新审查]
**原文对照**：Stage I。Tamoxifen→letrozole→exemestane转换。无复发。
P2=1："currently taking letrozole and exemestane" — 应为转换，非同时服用。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 60 (coral 200) — ER+/PR+/HER2-，新诊 [重新审查]
**原文对照**：43岁，新诊。Lumpectomy+IORT+reconstruction 4/12。Oncotype待查。内分泌治疗。
**逐句**："using radiation during the surgery" — IORT通俗解释出色 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 62 (coral 202) — ER+/PR-/HER2- Stage IIIA [重新审查]
**原文对照**：近完全缓解。Letrozole。激素监测。Abemaciclib讨论。DEXA。
P2=1：阅读等级10.3偏高。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 63 (coral 203) — ER+/PR+/HER2-，寡转移 [重新审查]
**原文对照**：Stage III-IV，可能胸骨转移。活检。寡转移治疗讨论。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 64 (coral 204) — ER弱阳(2%)，新辅助 [重新审查]
**原文对照**：ER 2%/PR 7%/HER2-。新辅助化疗。临床试验。Port。
"weakly positive for estrogen...neoadjuvant chemotherapy, which is treatment given before surgery to shrink the cancer" ✓。
**总评**：P0=0, P1=0, P2=0。**信件质量优秀。**

### SAMPLE 65 (coral 205) — 化生性癌 [重新审查]
**原文对照**：Metaplastic carcinoma，ER 5-10%，HER2/PR neg。新辅助→双侧乳切。
"rare type of breast cancer called metaplastic carcinoma" ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 67 (coral 207) — 多灶ER+，TCHP后 [重新审查]
**原文对照**：ER+/PR+/HER2-，6周期TCHP后近完全缓解。Mastectomy推荐。儿子遗传检测。
"ER+/PR+/HER2-...grows in response to hormones (estrogen) and does not have extra HER2 protein" ✓（**v5 P2修复确认**）。
P2=1：readability 10.0偏高。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 69 (coral 209) — 双侧乳腺癌 [重新审查]
**原文对照**：左ILC+右IDC。双侧乳切。Letrozole。放疗。CT肺结节。
P2=1：letrozole提及两次。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 71 (coral 211) — ER+/PR-/HER2- Stage I [重新审查]
**原文对照**：72岁，1.2cm IDC，margin-，LN-。Letrozole。Oncotype待查。3周。
**逐句**：手术结果 ✓。Letrozole**只提一次** ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 72 (coral 212) — Stage III，脂肪坏死 [重新审查]
**原文对照**：乳腺肿块→fat necrosis非复发。Arimidex。4个月。
"fat necrosis, not cancer recurrence" ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 77 (coral 217) — TNBC Stage IV [重新审查]
**原文对照**：79岁，TNBC Stage IV肝+LN。多线后疾病进展。临床试验。Rad Onc。
**逐句**："cancer is growing in your liver and lymph nodes" ✓。临床试验 ✓。**无"Advance care not discussed"** ✓（Rule 20生效）。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 79 (coral 219) — 局部复发 [重新审查]
**原文对照**：ER+/PR+/HER2-，DCIS术后7年局部复发。化疗4/11。放疗6周。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 81 (coral 221) — ER+/HER2-，混合癌 ★ [重新审查]
**原文对照**：A/P明确写"Stage II...November 24 lymph nodes positive"。"November 24"是脱敏文本。信件说"involved 11 lymph nodes"。11个LN在Stage II仍不合理。LN POST check regex已push但本次运行用旧代码。
**P1=1保留**。P2=1（"blood sugar levels...readings of 94"无意义旧值）。
**总评**：P0=0, **P1=1**, P2=1。

### SAMPLE 82 (coral 222) — 小叶癌，letrozole反应好 [重新审查]
**原文对照**：Lobular，LN involvement。Letrozole反应好（PET示显著缩小）。Continue until surgery。
P2=1：letrozole提及3次。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 83 (coral 223) — ER+/PR-/HER2-，脑/肝/骨 [重新审查]
**原文对照**：60岁，CHEK2，MS。Metastatic到骨/软组织/肝/可能脑膜。On capecitabine+zolendronic acid。脑/肝生长。
P2=1："capecitabine to 1500mg twice a day" — 剂量细节（Rule 18）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 84 (coral 224) — ER+/PR-/HER2- ILC，脑转移 [重新审查]
**原文对照**：Stage IIIA→IV（骨/肌肉/肝/脑）。多线。脑病灶。Steroid减量。临床试验。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 85 (coral 225) — ER+/PR+/HER2+，骨/肝 [重新审查]
**原文对照**：骨活动增加但肝稳定。Letrozole/ribociclib→fulvestrant+/-everolimus。
"bones has gotten worse, but liver has not changed" ✓。100%链 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 86 (coral 226) — ER+/PR+/HER2-，辅助 [重新审查]
**原文对照**：2.2cm，4/19 LN+。Curative。Hormonal therapy。100%链 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 87 (coral 227) — Stage IIIB→IV，脑/肺 ★ [重新审查]
**原文对照**：ER+/PR+/HER2-原发→脑转移变ER-/PR-/HER2-。Xeloda。Foundation One neg ERBB2。
"originally sensitive to hormones...However, the cancer that spread to your brain is now not sensitive" ✓✓ — **受体变化完美传达**（v5缺失）。Full code ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 89 (coral 229) — 腺癌，AC化疗 [重新审查]
**原文对照**：右乳腺癌。AC化疗。GCSF减量。放疗后。甲状腺问题。
P2=1："adenocarcinoma"未解释（应为"cancer that started in gland cells"）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 90 (coral 230) — ER+/PR+ Stage IV骨 [重新审查]
**原文对照**：骨转移进展。Everolimus/exemestane/denosumab。Lasix+KCl。PET/CT下周。
P2=1："lasix 10mg once a day with KCL 10Meq once daily" — 剂量细节。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 91 (coral 231) — ER+/PR-/HER2-，肝转移Epirubicin ★ [重新审查]
**原文对照**：A/P明确写"Epirubicin 25 mg/m2 D1,8,15"。
信件说"restarted your chemotherapy with Epirubicin and are using Neupogen" — **无mg/m2！** ✓✓。**无"Advance care not discussed"** ✓✓。肝改善 ✓。100%链 ✓。
**v5所有P2全部修复确认。**
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 93 (coral 233) — ER+/PR+/HER2- Stage IIA [重新审查]
**原文对照**：Oncotype RS 21→no chemo。Letrozole。Mammogram+MRI。Full code。
P2=1：letrozole提及两次（轻微重复）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 94 (coral 234) — ER+/HER2-，临床试验 [重新审查]
**原文对照**：Pembrolizumab试验→AC→radiation→capecitabine→hormone therapy。
**逐句**：治疗反应 ✓。治疗序列 ✓。100%链 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 96 (coral 236) — ER+/PR+/HER2- Stage I，MS [重新审查]
**原文对照**：0.8cm Grade 1。GILENYA for MS。Oncotype待查。
"GILENYA for your multiple sclerosis" ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 99 (coral 239) — ER+/PR+ 肝转移 [重新审查]
**原文对照**：Gemzar（因疲劳取消）。Tumor markers上升。Focalin prn。Exercise 10min x 3/day。
"cancelled your Gemzar treatment due to feeling very tired" ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 21 (coral 161) — 转移性ER+/HER2-，abemaciclib停药后
条件性计划（PET稳定vs进展）✓。"stopped abemaciclib due to pneumonitis...switched letrozole to anastrozole...because of skin rash" ✓。P2=1（"specific mutation"占位符）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 26 (coral 166) — 骨转移稳定
"stable, with no new spots found" ✓。背痛+瘀伤检查计划 ✓。简洁。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 28 (coral 168) — 早期ER+/HER2-，Oncotype低风险
TNM→"early stage" ✓。手术/放疗/letrozole/钙+VitD ✓。P2=1（letrozole句5/6重复）。
**总评**：P0=0, P1=0, P2=1。

### SAMPLE 29 (coral 169) — Stage II-III HER2+，新辅助计划
Coverage 92%！受体 ✓。"chemotherapy and biological therapies" ✓。Mediport ✓。Echo ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 32 (coral 172) — ER+/HER2- ILC，letrozole随访
无复发 ✓。Letrozole+钙+VitD ✓。NSAIDs ✓。6个月随访 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 33 (coral 173) — ER+/PR-/HER2-，局部复发
"grows in response to hormones" ✓。Arimidex→tamoxifen ✓。胸壁放疗 ✓。
**总评**：P0=0, P1=0, P2=0。

### SAMPLE 35 (coral 175) — ER+/HER2- Grade III，Abraxane化疗
Taxol过敏→Abraxane ✓。Doppler DVT ✓。Rad Onc ✓。P2=1（"mucinous carcinoma"未解释）。

### SAMPLE 36 (coral 176) — TNBC Stage IIA，辅助化疗
TNBC ✓。"adjuvant chemotherapy to help prevent the cancer from coming back" — adjuvant有解释 ✓。Full code解释 ✓。生活方式 ✓。P0=0, P1=0, P2=0。

### SAMPLE 39 (coral 179) — Stage II ER+/HER2-，letrozole
受体从原始数据(ER 95,PR 5)正确简化 ✓。副作用讨论 ✓。Prolia ✓。**Letrozole只提一次** ✓（v5重复两次的问题修复）。P0=0, P1=0, P2=0。

### SAMPLE 40 (coral 180) — ER+/HER2-，AC-Taxol计划
手术结果 ✓。AC-Taxol ✓。Port ✓。P0=0, P1=0, P2=0。

### SAMPLE 41 (coral 181) — PR+ IDC，tamoxifen启动
"cancer started in the milk ducts" ✓（词汇表）。Tamoxifen 5年 ✓。P0=0, P1=0, P2=0。

### SAMPLE 42 (coral 182) — TNBC Stage I，taxol+carboplatin
TNBC解释 ✓。手术恢复 ✓。化疗方案 ✓。P0=0, P1=0, P2=0。

### SAMPLE 43 (coral 183) — ER+/HER2-，BRCA1，残留病变
残留病变描述 ✓。临床试验（3vs5周放疗）✓。P2=1（"bilateral salpingo-oophorectomy (BSO)"未用通俗语言解释）。

### SAMPLE 45 (coral 185) — ER+/PR-/HER2-，margin阳性需再切
再切+可能腋窝清扫 ✓。letrozole ✓。DEXA+MRA ✓。P2=1（letrozole句5/7重复）。

### SAMPLE 48 (coral 188) — ER+/PR+/HER2-，术前讨论
手术日期 ✓。Tamoxifen ✓。放疗 ✓。"named your spouse as a person who can make decisions" ✓。P0=0, P1=0, P2=0。

### SAMPLE 49 (coral 189) — HR+/HER2-，de novo Stage IV
"cancer is under good control. However, biopsies...show both invasive and non-invasive cancer" — 平衡 ✓。P0=0, P1=0, P2=0。

### SAMPLE 51 (coral 191) — 35岁ER+/HER2-，初诊+生育力保存
生育力保存 ✓。CT+骨扫描 ✓。基因检测 ✓。P0=0, P1=0, P2=0。

### SAMPLE 52 (coral 192) — ER+/PR+/HER2+，Stage II/III
"pocket of clear fluid (seroma)" ✓。P2=1（化疗方案细节过多：AC x 4 q2wk + taxol 12wk + trastuzumab 1yr）。阅读等级10.1偏高。

### SAMPLE 53 (coral 193) — ER+/PR-/HER2-，骨转移Stage IV
稳定 ✓。palbociclib+放疗序列 ✓。P0=0, P1=0, P2=0。

### SAMPLE 56 (coral 196) — TNBC局部晚期
TNBC ✓。放疗 ✓。遗传咨询 ✓。P0=0, P1=0, P2=0。

### SAMPLE 58 (coral 198) — ER+/PR+/HER2- Stage I，换药
P2=1："currently taking letrozole and exemestane" — 应该是只在服letrozole，计划2-3周后换exemestane，不是同时服用。

### SAMPLE 60 (coral 200) — ER+/PR+/HER2-，新诊+手术
"surgery...using radiation during the surgery" — IORT通俗解释 ✓。P0=0, P1=0, P2=0。

### SAMPLE 62 (coral 202) — ER+/PR-/HER2- Stage IIIA
治疗反应好 ✓。Abemaciclib讨论 ✓。阅读等级10.3偏高。P2=1。

### SAMPLE 63 (coral 203) — ER+/PR+/HER2-，寡转移至胸骨
"a medication and taxol" 占位符 ✓。Full code ✓。焦虑情绪支持 ✓。P0=0, P1=0, P2=0。

### SAMPLE 64 (coral 204) — ER弱阳(2%)，局部晚期
"weakly positive for estrogen (ER) and low positive for progesterone (PR)" — **ER 2%的细致表达** ✓。"neoadjuvant chemotherapy, which is treatment given before surgery to shrink the cancer" ✓（词汇表）。P0=0, P1=0, P2=0。

### SAMPLE 65 (coral 205) — 化生性癌
"rare type of breast cancer called metaplastic carcinoma" ✓。P0=0, P1=0, P2=0。

### SAMPLE 67 (coral 207) — 多灶ER+，TCHP后近完全缓解
**v5 P2修复**："ER+/PR+/HER2-...grows in response to hormones (estrogen) and does not have extra HER2 protein" ✓（v5说"ER+ breast cancer"未解释）。阅读等级10.0偏高。P2=1。

### SAMPLE 69 (coral 209) — 双侧乳腺癌
双侧病理详情准确但偏技术化。Full code ✓。P2=1。

### SAMPLE 71 (coral 211) — ER+/PR-/HER2- Stage I
手术结果清晰 ✓。**Letrozole只提一次** ✓。P0=0, P1=0, P2=0。

### SAMPLE 72 (coral 212) — Stage III，脂肪坏死
"fat necrosis, not cancer recurrence" ✓。Arimidex ✓。P0=0, P1=0, P2=0。

### SAMPLE 77 (coral 217) — TNBC Stage IV，疾病进展
"cancer is growing in your liver and lymph nodes" ✓。临床试验 ✓。**无"Advance care not discussed"** ✓（Rule 20生效）。P0=0, P1=0, P2=0。

### SAMPLE 79 (coral 219) — ER+/PR+/HER2-，化疗+放疗
化疗日期 ✓。6周放疗 ✓。全基因组测序结果 ✓。P0=0, P1=0, P2=0。

### SAMPLE 81 (coral 221) — ER+/HER2-，混合导管/小叶癌 ★
**v5 P1部分改善**：从"24 lymph nodes"变为"11 lymph nodes"。但11个LN在Stage II仍不合理（脱敏数据问题）。LN POST check regex已修复但本次运行使用旧代码。**P1=1保留**。P2=1（旧血糖值"readings of 94"无意义）。

### SAMPLE 82 (coral 222) — 小叶癌，letrozole反应良好
"responding well...decreased significantly" ✓。P0=0, P1=0, P2=0。

### SAMPLE 83 (coral 223) — ER+/PR-/HER2-，脑/肝/骨转移
"meninges (the covering of the brain)" ✓。腰穿 ✓。P2=1（capecitabine 1500mg剂量出现）。

### SAMPLE 84 (coral 224) — ER+/PR-/HER2- ILC，多线后脑转移
脑部病灶 ✓。临床试验筛选 ✓。Prednisone减量 ✓。P0=0, P1=0, P2=0。

### SAMPLE 85 (coral 225) — ER+/PR+/HER2+，骨/肝转移
"bones has gotten worse, but...liver has not changed" — 混合反应 ✓。治疗更换 ✓。100%链 ✓。P0=0, P1=0, P2=0。

### SAMPLE 86 (coral 226) — ER+/PR+/HER2-，辅助治疗
"2.2 cm tumor and some spread to nearby lymph nodes" ✓。100%链 ✓。P0=0, P1=0, P2=0。

### SAMPLE 87 (coral 227) — ER+/PR+/HER2- Stage IIIB→IV ★
**受体变化通俗传达**："was originally sensitive to hormones...However, the cancer that spread to your brain is now not sensitive to hormones" ✓✓ — v5未能传达受体变化，v6完美表达。Full code ✓。P0=0, P1=0, P2=0。

### SAMPLE 89 (coral 229) — 腺癌，AC化疗中
P2=1（"adenocarcinoma"仍未解释为"cancer that started in gland cells"）。

### SAMPLE 90 (coral 230) — ER+/PR+ Stage IV骨转移
"cancer has progressed despite the current treatment" ✓。**语法修复**："advanced stage, meaning it has spread" ✓。P2=1（lasix 10mg/KCL 10Meq剂量细节）。

### SAMPLE 91 (coral 231) — ER+/PR-/HER2-，肝转移Epirubicin ★
**v5所有P2修复**：（1）无mg/m2剂量 ✓（2）无"Advance care not discussed" ✓（3）肝改善报告 ✓。100%链 ✓。P0=0, P1=0, P2=0。

### SAMPLE 93 (coral 233) — ER+/PR+/HER2- Stage IIA
P2=1（"blood tests from 2019" — 旧lab值仍出现）。

### SAMPLE 94 (coral 234) — ER+/HER2-，临床试验→AC→capecitabine
治疗反应好 ✓。治疗序列 ✓。100%链 ✓。P0=0, P1=0, P2=0。

### SAMPLE 96 (coral 236) — ER+/PR+/HER2- Stage I，MS合并
"GILENYA for your multiple sclerosis" ✓。Oncotype待查 ✓。P0=0, P1=0, P2=0。

### SAMPLE 99 (coral 239) — ER+/PR+ 肝转移，Gemzar
"cancelled your Gemzar treatment due to feeling very tired" ✓。P0=0, P1=0, P2=0。

---

## 全局总结（重新审查后修正）

### v6 vs v5 对比

| 指标 | v5 | v6 | 变化 |
|------|----|----|------|
| **P0 (幻觉)** | 0 | 0 | = |
| **P1 (重大错误)** | 2 | **2** | 1修复 + 1新增 + 1遗留 |
| **P2 (小问题)** | ~25 | **~22** | ↓12% |

### P1 详情

| Sample | v5 | v6 | 状态 |
|--------|----|----|------|
| 11 | 脑MRI新病灶遗漏 | **"but a brain scan showed two small new spots"** | ✅ 修复 |
| 81 | "24 lymph nodes"误读 | "11 lymph nodes"（仍不合理） | ⚠️ 部分改善，regex fix已push待下次运行 |
| 7 (新) | — | "found in some lymph nodes"与pCR矛盾 | ❌ 新问题：术后无残留肿瘤但信件说LN有发现 |

### v5已知P2模式修复确认

| 模式 | v5 | v6 | 确认 |
|------|----|----|------|
| "advanced stage, the cancer has spread"语法 | 4x | **0x** | ✅ 全修 |
| "a specific treatment"占位符 | 4x | **0x** | ✅ →"a medication" |
| "— in plain language:"标记泄漏 | 1x | **0x** | ✅ |
| "Advance care was not discussed"措辞 | 2x | **0x** | ✅ 全修 |
| Epirubicin mg/m2剂量 | 2x | **0x** | ✅ 全修 |
| ER+未解释 (S67) | 1x | **0x** | ✅ 现有完整解释 |
| 受体变化未传达 (S87) | 1x | **0x** | ✅ 现清楚传达 |

### v6残余P2模式（重新统计）

| 模式 | 次数 | Samples |
|------|------|---------|
| Letrozole/药物重复提及 | 6x | 19, 28, 45, 69, 82, 93 |
| 术语未解释(adenocarcinoma, mucinous, BSO) | 3x | 35, 43, 89 |
| 阅读等级偏高(>9.5) | 3x | 52, 62, 67 |
| 剂量细节残留 | 2x | 83, 90 |
| 旧/无关lab值 | 2x | 19, 93 |
| [REDACTED]替换不当("a medication center") | 1x | 36 |
| 用药时态混淆(letrozole+exemestane同时) | 1x | 58 |
| HER2-同句重复 | 1x | 48 |
| BSO医学术语未解释 | 1x | 43 |
| "cancer is progressing"过于肯定 | 1x | 1 |
| 原始药名泄漏(Norco April) | 1x | 43 |

### v6 每个 Sample 评分汇总

| Sample | P0 | P1 | P2 | Sample | P0 | P1 | P2 |
|--------|----|----|-----|--------|----|----|-----|
| 0 | 0 | 0 | 1 | 49 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 51 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 52 | 0 | 0 | 1 |
| 4 | 0 | 0 | 0 | 53 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 56 | 0 | 0 | 0 |
| 6 | 0 | 0 | 1 | 58 | 0 | 0 | 1 |
| **7** | **0** | **1** | **0** | 60 | 0 | 0 | 0 |
| 8 | 0 | 0 | 1 | 62 | 0 | 0 | 1 |
| 9 | 0 | 0 | 0 | 63 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 | 64 | 0 | 0 | 0 |
| 11 | 0 | 0 | 0 | 65 | 0 | 0 | 0 |
| 13 | 0 | 0 | 1 | 67 | 0 | 0 | 1 |
| 16 | 0 | 0 | 0 | 69 | 0 | 0 | 1 |
| 17 | 0 | 0 | 0 | 71 | 0 | 0 | 0 |
| 19 | 0 | 0 | 2 | 72 | 0 | 0 | 0 |
| 21 | 0 | 0 | 1 | 77 | 0 | 0 | 0 |
| 26 | 0 | 0 | 0 | 79 | 0 | 0 | 0 |
| 28 | 0 | 0 | 1 | **81** | **0** | **1** | **1** |
| 29 | 0 | 0 | 0 | 82 | 0 | 0 | 1 |
| 32 | 0 | 0 | 0 | 83 | 0 | 0 | 1 |
| 33 | 0 | 0 | 0 | 84 | 0 | 0 | 0 |
| 35 | 0 | 0 | 1 | 85 | 0 | 0 | 0 |
| 36 | 0 | 0 | 1 | 86 | 0 | 0 | 0 |
| 39 | 0 | 0 | 0 | 87 | 0 | 0 | 0 |
| 40 | 0 | 0 | 0 | 89 | 0 | 0 | 1 |
| 41 | 0 | 0 | 0 | 90 | 0 | 0 | 1 |
| 42 | 0 | 0 | 0 | 91 | 0 | 0 | 0 |
| 43 | 0 | 0 | 2 | 93 | 0 | 0 | 1 |
| 45 | 0 | 0 | 1 | 94 | 0 | 0 | 0 |
| 48 | 0 | 0 | 1 | 96 | 0 | 0 | 0 |
|  |  |  |  | 99 | 0 | 0 | 0 |

**汇总**：P0=0/61, P1=2/61 (3.3%), P2有问题的Sample=22/61, 无问题Sample=39/61 (64%)

### 医生整体评价

v6在v5基础上有明显改善。最关键的修复是：
1. **Sample 11 脑MRI进展现在被准确报告** — v5最严重的临床安全问题已解决
2. **Stage IV语法全部修复** — 从"making it advanced stage, the cancer has spread"到"advanced stage, meaning it has spread"
3. **Sample 87受体变化传达** — "originally sensitive to hormones...However, the cancer that spread to your brain is now not sensitive"
4. **Sample 91所有P2清零** — Epirubicin剂量、Advance care措辞全部修正
5. **"a specific treatment"→"a medication"** — 占位符更自然

需要下一轮修复：
- **Sample 7 P1**：pCR但信件说"found in some lymph nodes" — 需要在letter prompt中加入pCR处理规则
- **Sample 81 P1**：LN数字误读 — regex fix已push，下次运行会生效
- **Letrozole重复**（6x）：最常见的P2，可考虑POST check加药物名去重

