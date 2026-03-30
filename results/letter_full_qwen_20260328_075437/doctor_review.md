# 医生审查报告 — Patient Letter Generation (61 samples)

## 审查目标
以临床医生视角逐个审查每封患者信件：
1. **内容准确性**：原文 → 提取 → 信件，每一步归因链条是否可靠，有无编造
2. **通俗易懂性**：信件是否适合患者阅读，重点是否覆盖，细节是否恰当

## 审查标准
- **P0**: 幻觉/编造 — 信件中出现原文没有的信息
- **P1**: 重大错误 — 信息方向错误（如把"好转"写成"恶化"）
- **P2**: 小问题 — 遗漏、措辞不当、但不影响安全

## 数据来源
`results/letter_full_qwen_20260328_075437/progress.json` (61 samples)

---

## 逐 Sample 审查

### SAMPLE 0 (coral 140) — 新诊转移性乳腺癌

**原文关键事实**：56岁女性，2013年Stage IIA右乳ER+/PR+/HER2-乳腺癌，行乳房切除术。未做放化疗，拒绝他莫昔芬。现广泛转移（肺、腹膜、肝、卵巢、右腋窝局部复发）。CT无骨转移。情绪：非常害怕、焦虑，丈夫支持。计划：活检右腋窝→确认表型→如HR+/HER2-推荐ibrance。脑MRI+骨扫描+实验室检查。转诊Integrative Medicine。Advance care: Full code。目标：姑息。

**提取审查**：
- Type_of_Cancer: "ER+/PR+ invasive ductal carcinoma, HER2-" — 原文"ER and PR positive and her 2 neu negative" ✓
- Stage: "Originally Stage IIA, now metastatic (Stage IV)" ✓
- Metastasis: 肺、腹膜、肝、卵巢 ✓（与CT一致）
- goals_of_treatment: "palliative" ✓（原文"treatment would be palliative"）
- medication_plan: "If HR+/HER2 negative on biopsy, will recommend ibrance" ✓（条件性计划）
- procedure_plan: 活检右腋窝 ✓
- imaging_plan: "Brain MRI" — 只提了脑MRI，骨扫描放到了lab_plan里。轻微分类问题。
- lab_plan: "ordered a MRI of brain and bone scan as well as labs" — 把影像和实验室混在一起，且未列出具体实验室项目（CBC, CMP, CA 15-3, CEA, aPTT, PT）。P2。
- Advance care: Full code ✓
- Specialty: Integrative Medicine ✓
- 情绪检测："distressed", "very scared", "anxious" ✓

**信件逐句审查**：
1. "You came to us for your first visit because you have breast cancer that has spread." — 准确，新患者+转移 ✓
2. "Your cancer grows in response to hormones (estrogen) and does not have extra HER2 protein." — ER+解释正确，HER2-解释正确 ✓
3. "...originally at an early stage but has now spread...lungs, the lining of your abdomen (belly area), liver, and ovaries." — Stage IIA→"early stage" ✓，peritoneum→"lining of your abdomen (belly area)" 通俗解释优秀 ✓，转移部位完整 ✓
4. "...small nodules in your lungs and large masses in your liver...signs of cancer in your ovaries and a mass near your right armpit." — 与CT所见一致 ✓
5. "You are not currently taking any medications for your cancer." — 原文"No current outpatient medications on file" ✓
6. "Our goal is to make you feel better and improve your quality of life." — palliative目标的通俗表达 ✓
7. "We will be doing a biopsy of the mass in your right armpit to confirm the type of cancer..." — ✓
8. "We have ordered a brain MRI and bone scan to get a full picture..." — ✓
9. "We will meet again after we have all the test results..." — 与follow up一致 ✓
10. "We understand this is a difficult time, and our team is here to support you every step of the way." — 情绪支持恰当 ✓。原文记录她"very scared"、"distressed"，这句话温暖但不说教 ✓

**信件遗漏（P2）**：
- Advance care (Full code) 未提及
- Specialty (Integrative Medicine) 转诊未提及
- medication_plan (ibrance条件计划) 未提及 — 但因为是条件性的（等活检结果），不提也合理
- CT显示无骨转移 — 这是好消息，可以考虑告知患者

**归因链审查**：
- 句8 (imaging_plan, lab_plan): 有提取值但无原文引用 → A1
- 句10 (emotional_context): 自动检测，无提取值 — 可接受
- 其余句子归因链完整 ✓

**总评**：P0=0, P1=0, P2=2（Specialty和Advance care遗漏）。信件通俗、准确、温暖。受体解释正确。peritoneum解释出色。作为医生，我对这封信整体满意——患者读后能理解自己的病情和接下来的计划。

---

### SAMPLE 1 (coral 141) — 转移性TNBC，irinotecan第3周期

**原文关键事实**：44岁女性，Lynch综合征，合并早期子宫内膜癌和结肠癌。2013年右乳TNBC (ER-/PR-/HER2-)，原Stage IIB。现转移（肝、骨、胸壁）。正接受irinotecan化疗（第3周期第1天），因转氨酶升高和腹泻每周期都错过第8天。症状：混乱加重、胸壁疼痛/红肿（疑感染）、背痛加重（可能疾病进展）、焦虑/抑郁加重。实验室：严重贫血（Hgb 7.7）、低钠（124）、低钾（3.1）、低白蛋白（2.1）、碱性磷酸酶高（183）。计划：换irinotecan为每两周+增加剂量、多西环素治疗感染、输血1单位、补液补钾、加Effexor剂量、Rad Onc急诊转诊、社工+家庭健康转诊、3个月后复查CT。

**提取审查**：
- Type_of_Cancer: "ER-/PR-/HER2- triple negative IDC" ✓（原文明确"triple negative"）
- Stage: "Originally Stage IIB, now metastatic (Stage IV)" ✓
- Metastasis: "liver, bone, chest wall" ✓（PET/CT证实）
- lab_summary: 完整列出异常值，方向正确（L/H标记一致）✓
- findings: 详尽覆盖胸壁症状、背痛、贫血、MRI骨病变 ✓
- current_meds: "Irinotecan" ✓（当前化疗药）
- recent_changes: irinotecan改为每两周+增加剂量 ✓
- supportive_meds: ondansetron, prochlorperazine, loperamide, morphine, oxycodone ✓
- medication_plan: 多西环素、输血、补液补钾 ✓
- radiotherapy_plan: Rad Onc急诊转诊 ✓
- Others: "Social work, Home health" ✓（原文"Home health?"和之前的social work referral）
- 归因链：14/15句完整链 — 非常好

**信件逐句审查**：
1. "You visited us for a follow-up about your breast cancer." ✓
2. "triple negative...cancer cells lack three common receptors (ER, PR, HER2)" — TNBC解释准确 ✓
3. "The cancer has spread to the liver, bone, and chest wall, making it advanced stage, the cancer has spread." — **语法问题**：后半句"making it advanced stage, the cancer has spread"读起来断裂不流畅。P2
4. "blood tests show low levels of albumin, calcium, chloride, creatinine, potassium, and sodium. Your hemoglobin is also low, indicating anemia." — 低值方向全部正确（全部有L标记）✓。提到贫血 ✓。但未提碱性磷酸酶升高 — 轻微遗漏
5. "chest pain and swelling that might be due to an infection. Your back pain has worsened, possibly due to the cancer spreading." ✓ 与原文一致
6. "currently on irinotecan" ✓
7. "changed your irinotecan treatment to every other week and increased the dose to 150mg/m2 on days 1 and 15 every 28 days" — **剂量信息太详细**（"150mg/m2 days 1 and 15 every 28 days"），患者不太需要知道mg/m2。P2
8. "manage your symptoms and improve your quality of life" — palliative的通俗表达 ✓
9. "supportive medications like ondansetron, prochlorperazine, loperamide, morphine, and oxycodone" — 列出药名，患者可能不熟悉。但"supportive medications"提供了语境 ✓
10. "doxycycline for 7 days...blood transfusion to help with your anemia" ✓（原文"Rx Doxycycline 100mg BID x 7d"和"1 unit pRBC today"）
11. "send a message to your doctor...radiation oncology consultation" ✓
12. "scans again in 3 months...MRI of your brain if your symptoms get worse" ✓
13. "monitor your liver function and hepatitis B status every 4 months" ✓
14. "refer you to social work and home health services" ✓ — 正确标记为[source:Others]
15. 情绪支持句 ✓

**遗漏但可能重要**：
- 患者的混乱/意识改变（原文开头特别提到"patient still appears more confused"）— 这对患者来说可能重要
- 焦虑/抑郁加重和Effexor增量（从37.5→75mg）— 心理健康方面
- 补液补钾的具体措施（NS 500ml, 40mEq KCl）
- Lynch综合征和其他癌症史 — 但这次访问聚焦乳腺癌随访，不提也合理

**总评**：P0=0, P1=0, P2=2（句3语法、句7剂量过详）。信件信息量大且准确，覆盖了治疗变更、感染、贫血、转诊等要点。TNBC解释正确。归因链近乎完美（93%）。作为医生，这封信基本能让患者了解这次复诊的重要信息。

---

### SAMPLE 2 (coral 142) — 新诊Stage IIA HR+/HER2-乳腺癌，视频会诊

**原文关键事实**：53岁女性，乳腺X线发现右乳癌。1.7cm肿瘤，腋窝淋巴结阳性，HR+/HER2-（FISH阴性），Stage IIA。已看过多位医生（second opinion）。正在做PET/CT和基因检测。计划：讨论化疗、手术+放疗、内分泌治疗降低复发风险。生活方式建议（饮食、运动、睡眠、减压）。Oncotype已送出等结果。Full code。无药物。无实验室结果。

**提取审查**：
- Type_of_Cancer: "ER+/PR+/HER2- IDC" ✓（原文"HR+, her 2 2+, fish negative"）
- Stage: "Stage IIA" ✓
- Metastasis: "No" ✓
- goals: "curative" ✓（Stage IIA，讨论新辅助治疗）
- genetic_testing_plan: "sent and pending" ✓
- second opinion: "yes" ✓（原文"several opinions"）

**信件逐句审查**：
1. "medical oncology consult about your newly diagnosed breast cancer" ✓
2. "ER+/PR+/HER2-...grows in response to hormones (estrogen) and does not have extra HER2 protein" — 受体解释正确 ✓
3. "early stage, called Stage IIA" ✓
4. "no sign that the cancer has spread" ✓
5. "1.7 cm tumor in the upper-outer part, and a nearby lymph node is also affected" ✓
6. "not currently taking any medications" ✓
7. "chemotherapy to lower the chance of the cancer coming back" ✓
8. "surgery and radiation to lower the risk" ✓
9. "PET scan and genetic testing are being done" ✓
10. "follow up after the results" ✓
11. 情绪支持 ✓

**遗漏（P2）**：
- goals_of_treatment: "curative" 未明确说出。虽然信件说了"lower the chance of coming back"间接暗示了治愈意图，但没有直接说"Our goal is to cure"
- 内分泌治疗（hormonal blockade）讨论未提及 — 原文A/P第4点明确讨论了
- 生活方式建议（饮食、运动、睡眠）未提及
- Advance care (full code) 未提及

**总评**：P0=0, P1=0, P2=2（goals未显式说出、内分泌治疗讨论遗漏）。信件清晰通俗，受体解释正确，Stage IIA→"early stage"恰当。患者能理解基本诊断和下一步计划。

---

### SAMPLE 4 (coral 144) — 复发转移ER+/HER2-乳腺癌随访

**原文关键事实**：31岁绝经前女性，Stage III ER+/HER2- IDC左乳，行双侧乳房切除+AC化疗3周期，未放疗。现活检证实转移复发（左颈部淋巴结）。正接受leuprolide+anastrozole+palbociclib。脑MRI正常。颈椎MRI示淋巴结病变和左臂丛受累。耐受治疗良好，偶尔用Zofran。计划：继续治疗，Rad Onc转诊（左颈/臂丛症状），CT+骨扫描复查，月度实验室检查。

**提取审查**：
- Type_of_Cancer/Stage/Metastasis/current_meds 均准确 ✓
- response_assessment: 有详细CT结果（颈部淋巴结缩小、纵隔淋巴结稳定、部分增大）✓
- 但response_assessment未出现在信件中

**信件逐句审查**：
1-2. 就诊原因、受体解释 ✓
3. "...making it advanced stage, the cancer has spread" — **语法模式**重复出现（同Sample 1），读起来断裂。P2
4. "creatinine level of 0.6 mg/dL, which is within the normal range" — 一个正常的肌酐值从几个月前拿来，对患者没有意义。P2（无用细节）
5. "lump in your left neck and involvement of the left brachial plexus, which may be causing pain in your left arm" ✓ — 通俗解释臂丛受累
6-9. 用药、继续治疗、Rad Onc转诊、检查计划均准确 ✓

**遗漏（P2）**：
- response_assessment — 患者会想知道治疗效果如何，CT显示部分缩小部分增大，这是重要信息
- 脑MRI正常 — 对患者来说是好消息，值得告知
- goals_of_treatment (palliative) — 虽然"manage symptoms and control cancer"间接暗示

**总评**：P0=0, P1=0, P2=3（语法、无用肌酐、疗效评估遗漏）。受体解释正确，用药信息准确，Rad Onc转诊清晰。但缺少治疗效果评估是较大遗漏。

---

### SAMPLE 5 (coral 145) — 早期ER+/HER2-乳腺癌，辅助治疗启动

**原文关键事实**：34岁女性，右乳ER+/PR+/HER2-（FISH阴性）IDC，1.5cm，淋巴结阴性，Grade 1。已行双侧乳房切除+扩张器。一个月前开始zoladex，今日开始letrozole。合并双相情感障碍 — 讨论tamoxifen可能引发躁狂（精神科同意gabapentin）。有遗传学转诊。计划：letrozole至少3年→序贯tamoxifen。月度检测estradiol。3个月后复诊。

**提取审查**：各字段准确 ✓。goals "curative" 正确（早期，辅助治疗）。

**信件逐句审查**：
1. "follow-up about your breast cancer" ✓
2. "grows in response to hormones (estrogen) and does not have extra HER2 protein" — 受体解释正确 ✓
3. "surgery to remove both breasts...left breast was healthy, right breast had a small cancer that did not spread to the lymph nodes" — 通俗准确，术后结果清楚 ✓
4. "blood tests are mostly normal" ✓（所有值均在正常范围）
5-6. 用药信息 ✓
7. "goal of your treatment is to cure the cancer" ✓ — 早期curative intent
8. "letrozole for at least 3 years. You can also take tamoxifen later" ✓
9. "estradiol levels every month" ✓
10. "3 months or sooner" ✓

**遗漏（P2）**：遗传学转诊未提及。其余覆盖完整。

**总评**：P0=0, P1=0, P2=1（Genetics遗漏）。非常干净的信件，阅读等级5.6（最低之一），简单直白。早期病例信息量适中，患者容易理解。**本组最佳信件之一。**

---

### SAMPLE 6 (coral 146) — 转移性HER2+乳腺癌，second opinion

**原文关键事实**：1998年Stage II左乳IDC，lumpectomy+ALND，AC+T+XRT，未用内分泌治疗。原始生物标记"结果不清"。2008年复发（左锁骨上淋巴结+纵隔），活检ER-/PR-/HER2+。多线治疗后进展。今日second opinion，已停药一周。影像示左乳/胸壁可能轻度进展。

**提取审查**：
- Type_of_Cancer: "Originally ER+/PR+/HER2+, metastatic biopsy ER-/PR-/HER2+" — **原始标记有疑问**：原文说"Biomarker results unclear"，提取却写了ER+/PR+。可能是推断，但原文不支持。P2
- Stage/Metastasis/goals 准确 ✓
- response_assessment: 左乳/胸壁可能轻度进展 ✓

**信件逐句审查**：
1. "second opinion about your breast cancer" ✓
2. "originally...grew in response to hormones (estrogen) and had extra HER2 protein" — **基于不确定的提取**。原文说1998年生物标记"不清楚"，这句话可能不准确。P2
3. "cancer has spread to other parts of your body, including your chest and neck" — 通俗化锁骨上/纵隔转移 ✓
4. "small increase in the size of the cancer" — 与"probable mild progression"一致 ✓
5. "stopping your current treatment, which includes **a specific treatment**, Herceptin, and Taxotere" — "[REDACTED]→a specific treatment"占位符读起来非常奇怪，患者会困惑 P2
6. "make you as comfortable as possible and control the cancer's growth" — palliative恰当 ✓
7. "start you on a new treatment called **a specific treatment**" — 同上占位符问题 P2
8. "check your blood tests again before starting" ✓

**关键遗漏**：
- **受体变化未清楚传达** — 这是重要临床信息：原始可能ER+，复发变为ER-/PR-/HER2+。信件只说了"originally grew in response to hormones"但没说现在受体状况已改变
- response_assessment未单独提及

**总评**：P0=0, P1=0, P2=3（原始标记可能不准确、占位符尴尬、受体变化未传达）。信件通俗性好，但受体变化的处理需要改进。"a specific treatment"占位符是系统性问题。

---

### SAMPLE 7 (coral 147) — Stage III HER2+/ER- IDC，术后辅助治疗讨论

**原文关键事实**：29岁绝经前，Stage II-III ER-/PR-/HER2+（IHC 3+，FISH 5.7）左乳IDC。TCHP化疗不规律（仅3个不完整周期）。行lumpectomy/ALND — **病理完全缓解（无残留肿瘤）**。现讨论后续辅助治疗。计划：AC x 4 → T-DM1 → 放疗。社工转诊。

**信件审查**：
- 受体解释："does not respond to hormones and has extra HER2 protein" ✓（ER-/HER2+正确）
- Stage III→"locally advanced" ✓
- 治疗计划（AC→T-DM1→radiation）清晰 ✓
- 社工转诊 ✓

**遗漏（P2）**：
- **病理完全缓解（pCR）未告知** — 这是好消息！手术组织无残留肿瘤。应告诉患者"手术显示之前的治疗有效，切除的组织中没有发现癌细胞"
- 句3-4有些重复（都说没有远处扩散）

**总评**：P0=0, P1=0, P2=2（pCR好消息未传达、轻度重复）。治疗计划清晰，受体解释正确。

---

### SAMPLE 8 (coral 148) — Stage II HR+/HER2-，术后恢复+辅助计划

**原文关键事实**：Stage II右乳IDC，HR+/HER2-。已完成4周期[化疗]+每周Taxol x 12。行双侧乳房切除：3.84cm肿瘤，1个淋巴结阳性。神经病变改善。骨质减少→Fosamax。计划：放疗转诊→之后letrozole。今日患者流泪。Full code。

**信件审查**：
- 受体解释正确 ✓
- Stage II→"early stage" ✓
- "surgery removed it successfully. One lymph node has cancer cells, but the cancer has not spread" — 对手术结果的通俗总结清晰准确 ✓
- "goal is to cure the cancer" ✓
- "After radiation therapy, you will start taking...Letrozole to block the effects of estrogen" ✓ — 治疗序列正确
- "drains will be removed on Thursday" — 具体、实用的信息 ✓
- 情绪支持 ✓（原文记录患者今日流泪，情绪支持恰当）

**总评**：P0=0, P1=0, P2=0。干净准确的信件，覆盖诊断、手术结果、下一步计划。阅读等级6.6，通俗。**无问题。**

---

### SAMPLE 9 (coral 149) — Stage II HR+/HER2-，术后随访

**原文关键事实**：左乳HR+/HER2- IDC，Stage II。已行左乳切除术（8.8cm肿瘤，多个淋巴结阳性）。Oncotype低风险→不需化疗。继续letrozole。下周开始放疗模拟。DEXA扫描。Full code。

**信件审查**：
- 受体解释："HR+ invasive ductal carcinoma, HER2- **— in plain language:** grows in response to hormones" — **"— in plain language:"标记泄漏到信件文本中**！P2（已知问题）
- Stage II→"early stage" ✓
- letrozole、放疗、DEXA计划均准确 ✓
- 遗漏：Oncotype低风险+不需化疗（好消息未告知）

**总评**：P0=0, P1=0, P2=1（标记泄漏）。除标记泄漏外，信件通俗准确。

---

### SAMPLE 10 (coral 150) — 转移性乳腺癌骨转移，Faslodex治疗中

**原文关键事实**：Stage IIIC→IV转移性乳腺癌，骨转移。On Faslodex+Denosumab。下颌放疗后改善但麻木持续。右腿疼痛/麻木加重。PET/CT示左下颌骨病变活动增加。鹅口疮。计划：继续Faslodex/Denosumab，PETCT评估股骨/足趾，Mycelex治疗鹅口疮。

**信件审查**：
- 诊断、分期、转移部位准确 ✓
- "blood tests are mostly normal" ✓（CBC值正常）
- 症状描述（颌骨麻木、腿痛）准确 ✓
- "cancer is not responding well to the current treatment" — PET显示活动增加，这个表述准确 ✓
- Mycelex/鹅口疮、盐苏打漱口、影像计划 ✓
- 治疗目标palliative恰当 ✓

**总评**：P0=0, P1=0, P2=0。**准确全面的信件。**覆盖症状、治疗反应、计划。阅读等级7.1。

---

### SAMPLE 11 (coral 151) — 转移性ER+/PR+/HER2+乳腺癌，脑/肺/骨转移

**原文关键事实**：50岁女性，ER+/PR+/HER2+转移性乳腺癌。转移至脑（多次伽马刀）、肺、骨。On herceptin+letrozole。CT CAP稳定。**但MRI脑显示2个新病灶**。DNR/DNI。因多种药物不耐受已停化疗。计划：继续herceptin+letrozole，等Rad Onc意见，CT/MRI每4个月。

**信件审查**：
- 受体解释正确 ✓
- 转移部位（脑、肺、骨）✓
- **句4："Recent imaging shows that the cancer is stable, with no new problems in your lungs or heart"** — **P1**：虽然CT CAP和超声心动确实稳定，但**脑MRI显示2个新病灶**！信件完全未提及脑部新病灶，给患者一个"一切稳定"的错误印象。这对患者来说是重大信息遗漏
- 句7提到等Rad Onc意见但没解释为什么 — 患者会困惑
- herceptin+letrozole用药准确 ✓
- DNR/DNI未提及（提取记录"Not discussed during this visit"）

**总评**：P0=0, **P1=1（脑部新病灶进展未告知，创造误导性乐观印象）**, P2=1（DNR/DNI遗漏）。这是61个样本中发现的第一个P1问题。根本原因：信件选择性报告了稳定的body CT结果，但遗漏了进展的brain MRI结果。

---

### SAMPLE 13 (coral 153) — 转移性ER+乳腺癌，低剂量化疗+免疫

**原文关键事实**：58岁女性，de novo转移性ER+/HER2-乳腺癌至骨/肝/淋巴结。S/p脊柱手术+XRT。最近停palbociclib/fulvestrant。现低剂量化疗（[药名]+Gemcitabine+Docetaxel）+免疫疫苗+pamidronate。肋骨疼痛。计划：CT+脊柱MRI 5月，实验室每两周，PT转诊。

**信件审查**：覆盖全面 — 用药变化、低剂量化疗方案、pamidronate副作用（寒战→减量）、大麻外用、Cymbalta处方、影像/实验室计划、PT转诊均提及 ✓。归因链100%完整。P2=1（"a specific treatment"占位符）。

**总评**：P0=0, P1=0, P2=1。**归因链完美。**

---

### SAMPLE 16 (coral 156) — 早期ER+/PR+/HER2-，术后辅助内分泌治疗讨论

**原文关键事实**：53岁女性，左乳IDC 0.8cm，Grade 2，ER+/PR+/HER2-，margin阴性，LN 0/5阴性。S/p lumpectomy。10-15%复发风险。讨论tamoxifen vs AI（取决于绝经状态）。需放疗。DXA。遗传学+营养转诊。

**信件审查**：非常全面 — 诊断、手术结果、治疗选择（tamoxifen vs AI）、放疗、DXA、遗传学、营养师、随访计划均覆盖 ✓。受体解释正确 ✓。句7/10轻微重复（都提检查激素水平）。Coverage 88%。

**总评**：P0=0, P1=0, P2=1（轻微重复）。全面准确的信件。

---

### SAMPLE 17 (coral 157) — Stage I ER+/PR+/HER2-，术后辅助计划

**原文关键事实**：左乳IDC 8mm + 包裹性乳头状癌。pT1b，大约Stage I。LN有微小转移。S/p lumpectomy。计划：辅助内分泌治疗5-10年，Rad Onc评估，DEXA。

**信件审查**：所有要点覆盖 ✓。TNM正确转换为"Stage I-II" ✓。"very small amounts of cancer cells in one of them"准确描述LN微转移 ✓。"hormone-blocking medicine for 5 to 10 years"通俗 ✓。"bone density test (DEXA)"解释了DEXA是什么 ✓。

**总评**：P0=0, P1=0, P2=0。**干净准确的信件。**

---

### SAMPLE 19 (coral 159) — 转移性ER+/HER2-，新开letrozole+palbociclib

**原文关键事实**：75岁女性，ER+/HER2- IDC。2009年诊断→双侧乳房切除+5年tamoxifen。现转移至骨和淋巴结。开始letrozole+palbociclib。MRI/CT/PET/Foundation One/Rad Onc转诊。

**信件审查**：覆盖全面 ✓。受体解释正确 ✓。月度血检提醒 ✓。影像/遗传检测/Rad Onc ✓。P2=2：（1）"blood tests show a normal glucose level" — 一个2013年的旧血糖值，对患者无意义；（2）letrozole重复提及两次（句6和7）。

**总评**：P0=0, P1=0, P2=2。

---

### SAMPLE 21 (coral 161) — 转移性ER+/HER2-，abemaciclib因肺炎停药

**原文关键事实**：72岁女性，复杂病史（L DCIS 1994 + R Stage II IDC 2000）。2020年转移至骨/胸壁/淋巴结。Abemaciclib因肺炎停用。现anastrozole+denosumab。PET示良好反应。计划：PET CT→稳定则继arimidex→进展则faslodex+[突变靶向药]。

**信件审查**："lung irritation (pneumonitis)" 通俗解释 ✓。条件性计划（稳定vs进展）清晰传达 ✓。PET良好反应提及 ✓。P2=1（占位符"specific mutation"）。信件通俗准确。

**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 26 (coral 166) — 转移性骨转移，稳定病变

**原文关键事实**：转移性ER+/HER2-乳腺癌骨转移。On FEMARA/ZOLADEX/zolendronic acid。PET稳定/略降。新症状：头晕、背痛、瘀伤、尿频。计划：继续用药、考虑MRI脊柱、CBC查瘀伤、UA查UTI。

**信件审查**：所有要点覆盖 ✓。"treatment is working well, as the cancer activity in your bones has slightly decreased" — 治疗有效的好消息准确传达 ✓。症状、检查计划、随访均提及。归因链100%。

**总评**：P0=0, P1=0, P2=0。**准确全面。**

---

### SAMPLE 28 (coral 168) — 早期ER+/HER2-，Oncotype低风险

**原文关键事实**：59岁女性，多灶性Grade 2 IDC，ER+/PR+/HER2-，SLN微转移(0.5mm)，Oncotype低风险。开始letrozole。需re-excision。放疗计划。9月手术。

**信件审查**：TNM→"early stage" ✓（无TNM术语）。"small area of cancer...tiny spot in a lymph node" ✓。letrozole、钙、骨密度、放疗、手术均提及 ✓。归因链100%。P2=1（letrozole重复提及句5/6）。

**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 29 (coral 169) — Stage II-III HER2+，新辅助化疗计划

**原文关键事实**：64岁女性，ER-/PR-/HER2+ IDC。右乳9cm肿块+腋窝/皮下LN。计划：新辅助化疗（paclitaxel+trastuzumab或TCHP）→手术→放疗。Mediport。TTE心脏评估。

**信件审查**："does not respond to hormones but has extra HER2 protein" ✓。"chemotherapy before surgery to shrink the cancer" — 新辅助治疗通俗解释 ✓。Mediport、TTE、放疗、决定治疗地点均准确 ✓。情绪支持 ✓。

**总评**：P0=0, P1=0, P2=0。**优秀信件。**

---

### SAMPLE 32 (coral 172) — ER+/HER2- 小叶癌，adjuvant letrozole随访

**信件审查**：所有要点覆盖 ✓。"no signs of the cancer coming back" ✓。letrozole、钙、维D、NSAIDs ✓。6个月随访 ✓。受体解释正确。

**总评**：P0=0, P1=0, P2=0。

---

### SAMPLE 33 (coral 173) — ER+/PR- IDC，局部复发→tamoxifen+胸壁放疗

**信件审查**：局部复发准确描述 ✓。"grows in response to hormones" ✓。arimidex→tamoxifen转换 ✓。胸壁放疗转诊 ✓。"no evidence of spreading" ✓。6个月随访 ✓。

**总评**：P0=0, P1=0, P2=0。

---

### SAMPLE 35 (coral 175) — 年轻患者ER+/HER2- Grade III，Abraxane化疗中

**信件审查**：Taxol过敏反应→Abraxane转换 ✓。"zoladex to protect your ovaries during chemotherapy" 通俗解释 ✓。低RBC/Hgb ✓。右臂肿胀+doppler DVT排查 ✓。Rad Onc转诊 ✓。P2=1："mucinous carcinoma"（黏液性癌）未解释（prompt要求解释为"a type of cancer that makes mucus"）。

**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 36 (coral 176) — Stage IIA TNBC，术后辅助化疗计划

**信件审查**："triple negative...cancer cells lack three common receptors" TNBC解释正确 ✓。"dd AC followed by Taxol" ✓。无放疗/内分泌治疗明确说明 ✓。P2=1："adjuvant"未解释（prompt要求解释为"treatment given after surgery to prevent the cancer from coming back"）。

**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 39 (coral 179) — Stage II ER+/HER2-，letrozole启动

**提取特殊情况**：Type_of_Cancer提取为原始数据"ER 95, PR 5, HER2 2+ FISH negative"，但信件正确简化为"responds to the hormone estrogen but does not have extra HER2 protein" ✓ — **预处理清洗有效**。手术margin clear ✓。DEXA ✓。PT ✓。P2=1（letrozole句5/6重复）。

**总评**：P0=0, P1=0, P2=1。

---

### SAMPLE 40 (coral 180) — ATM突变，ER+/HER2-，AC-Taxol计划
P0=0, P1=0, P2=0。用药/手术/卵巢抑制/port均准确。

### SAMPLE 41 (coral 181) — PR+ IDC，tamoxifen启动
P0=0, P1=0, P2=1。HER2提取为"not tested"但信件说"does not have extra HER2 protein" — 推断合理（tamoxifen治疗暗示HER2-）但技术上未验证。

### SAMPLE 42 (coral 182) — TNBC Stage I，taxol+carboplatin
P0=0, P1=0, P2=0。TNBC解释正确。化疗方案清晰。

### SAMPLE 43 (coral 183) — ER+/HER2-，BRCA1，残留病变+放疗临床试验
P0=0, P1=0, P2=0。**非常全面**：残留病变、临床试验（3vs5周放疗）、AI、卵巢切除、CT随访、营养/PT转诊全覆盖。

### SAMPLE 45 (coral 185) — ER+/PR-/HER2-，pT2N2，margin阳性需再切
P0=0, P1=0, P2=0。复杂外科/治疗计划清晰传达：再切、腋窝清扫可能、letrozole→abemaciclib序列。

### SAMPLE 48 (coral 188) — ER+/PR+/HER2-，Stage II，术前讨论
P0=0, P1=0, P2=0。手术计划、tamoxifen VTE风险、放疗讨论、**Advance care（配偶为代理决策人）**均准确传达。

### SAMPLE 49 (coral 189) — HR+/HER2-，de novo Stage IV，ibrance控制良好
P0=0, P1=0, P2=0。"cancer is under good control...However, progression in the left breast" — 平衡报道 ✓。多线治疗+mastectomy考虑 ✓。

### SAMPLE 51 (coral 191) — 35岁ER+/HER2-，初诊+生育力保存
P0=0, P1=0, P2=0。**生育力保存优先**准确传达 ✓。CT/bone scan/基因检测计划 ✓。

### SAMPLE 52 (coral 192) — ER+/PR+/HER2+，Stage II/III
P0=0, P1=0, P2=1。seroma解释为"pocket of clear fluid" ✓（医学词汇表生效）。"adjuvant"未解释。阅读等级9.7偏高。

### SAMPLE 53 (coral 193) — ER+/PR-/HER2-，骨转移Stage IV
P0=0, P1=0, P2=0。稳定病变准确报告 ✓。palbociclib+放疗序列 ✓。

### SAMPLE 56 (coral 196) — TNBC局部晚期，术后AC化疗
P0=0, P1=0, P2=0。TNBC解释 ✓。化疗后神经病变提及 ✓。放疗+遗传咨询 ✓。

### SAMPLE 58 (coral 198) — ER+/PR+/HER2- Stage I，换药exemestane
P0=0, P1=0, P2=0。无复发证据 ✓。letrozole→exemestane转换解释清楚 ✓。

### SAMPLE 60 (coral 200) — ER+/PR+/HER2-，新诊+手术计划
P0=0, P1=0, P2=0。手术日期、Oncotype待查、tamoxifen vs卵巢抑制选择 ✓。

### SAMPLE 62 (coral 202) — ER+/PR-/HER2- Stage IIIA，近完全缓解
P0=0, P1=0, P2=1。"tumor nearly disappearing on imaging" ✓。abemaciclib剂量+减量选项 ✓。**阅读等级10.1偏高**。

### SAMPLE 63 (coral 203) — ER+/PR+/HER2-，可能寡转移至胸骨
P0=0, P1=0, P2=1（占位符"specific treatment"）。条件性计划（活检阳性→xgeva）清晰 ✓。

### SAMPLE 64 (coral 204) — ER弱阳(2%)，局部晚期，新辅助化疗
P0=0, P1=0, P2=0。"responds a little to the hormone estrogen" — **ER 2%的创造性通俗表达** ✓。新辅助化疗="treatment before surgery to shrink the cancer" ✓。

### SAMPLE 65 (coral 205) — 化生性癌（罕见类型），新辅助化疗
P0=0, P1=0, P2=0。"rare type of breast cancer called metaplastic carcinoma" ✓（医学词汇表生效）。

### SAMPLE 67 (coral 207) — 多灶ER+，TCHP后近完全缓解
P0=0, P1=0, **P2=1（ER+未解释）**。信件说"ER+ breast cancer"但没有parenthetical解释ER+含义。TCHP反应好 ✓。mastectomy推荐 ✓。儿子遗传检测建议 ✓。

### SAMPLE 69 (coral 209) — 双侧乳腺癌（左ILC+右IDC）
P0=0, P1=0, P2=1（术后病理细节过多——对患者来说太技术化）。但受体解释正确 ✓。letrozole ✓。

### SAMPLE 71 (coral 211) — ER+/PR-/HER2- Stage I，术后letrozole+Oncotype
P0=0, P1=0, P2=1（letrozole重复提及）。TNM→"Stage I-II" ✓。手术结果清晰 ✓。Oncotype待查 ✓。

### SAMPLE 72 (coral 212) — ER+/PR+/HER2- Stage III，脂肪坏死
P0=0, P1=0, P2=0。"fat necrosis, which is a common reaction after breast surgery" — **医学术语通俗解释优秀** ✓。100%归因链。

### SAMPLE 77 (coral 217) — TNBC Stage IV，肝+LN转移，疾病进展
P0=0, P1=0, P2=1（"Advance care planning was not discussed"措辞生硬）。疾病进展准确报告 ✓。临床试验兴趣 ✓。Rad Onc ✓。

### SAMPLE 79 (coral 219) — ER+/PR+/HER2-，化疗即将开始
P0=0, P1=0, P2=0。化疗日期、放疗计划、支持药物均清晰。

### SAMPLE 81 (coral 221) — ER+/PR+/HER2-，混合导管/小叶癌
P0=0, **P1=1（"spread to 24 lymph nodes"几乎肯定是脱敏文本误读）**。原文"November 24 lymph nodes positive"中的"November 24"极可能是日期或脱敏后的数字（如"2/4"），而非24个阳性淋巴结。Stage II不可能有24个阳性LN。P2=1（无关的血糖值）。

### SAMPLE 82 (coral 222) — 小叶癌，新辅助letrozole，反应良好
P0=0, P1=0, P2=0。"responding well...decreased significantly" ✓。简短但准确。

### SAMPLE 83 (coral 223) — ER+/PR-/HER2-，CHEK2突变，脑/肝/骨转移
P0=0, P1=0, P2=0。复杂转移病例处理出色。"ER+ (grows in response to hormones like estrogen) and PR- (does not respond to progesterone)" — 详细的受体解释 ✓。脑膜转移可能性 ✓。腰穿 ✓。

### SAMPLE 84 (coral 224) — ER+/PR-/HER2- ILC，多线治疗后脑转移
P0=0, P1=0, P2=0。"spot in your brain...causing headaches and facial numbness" ✓。"make you as comfortable as possible" ✓。临床试验筛选 ✓。

### SAMPLE 85 (coral 225) — ER+/PR+/HER2+，骨/肝/脑转移
P0=0, P1=0, P2=0。治疗更换（→fulvestrant+/-everolimus）✓。"bones more active, liver stable" ✓。100%归因链。

### SAMPLE 86 (coral 226) — ER+/PR+/HER2-，术后辅助治疗
P0=0, P1=0, P2=0。**"4 out of 19 lymph nodes"清晰** ✓。"get rid of the cancer completely" ✓。100%归因链。

### SAMPLE 87 (coral 227) — ER+/PR+/HER2- Stage IIIB→IV，脑/肺转移
P0=0, P1=0, P2=0。Xeloda ✓。HER2复检计划 ✓。脑手术+放疗描述 ✓。

### SAMPLE 89 (coral 229) — 腺癌，AC化疗中
P0=0, P1=0, P2=1（"adenocarcinoma"未解释——prompt要求解释为"cancer that started in gland cells"）。

### SAMPLE 90 (coral 230) — ER+/PR+ Stage IV骨转移，治疗反应不佳
P0=0, P1=0, P2=1（"advanced stage, the cancer has spread"语法模式）。"treatment is not working as well as we hoped" — 诚实评估 ✓。

### SAMPLE 91 (coral 231) — ER+/PR-/HER2-，肝转移，Epirubicin
P0=0, P1=0, P2=2（化疗剂量太详细"25 mg/m2 Day 1,8,15"、"advance care planning was not discussed"措辞生硬）。肝脏改善准确报告 ✓。

### SAMPLE 93 (coral 233) — ER+/PR+/HER2- Stage IIA，Oncotype低风险
P0=0, P1=0, P2=1（占位符"specific treatment"）。"low risk test → no chemo" 通俗解释 ✓。

### SAMPLE 94 (coral 234) — ER+/HER2-，临床试验→AC→放疗→capecitabine
P0=0, P1=0, P2=0。治疗反应、后续治疗序列均准确 ✓。100%归因链。

### SAMPLE 96 (coral 236) — ER+/PR+/HER2-，Stage I，MS合并
P0=0, P1=0, P2=0。TNM→"Stage I-II" ✓。"GILENYA for your multiple sclerosis"恰当提及合并症 ✓。

### SAMPLE 99 (coral 239) — ER+/PR+ 肝转移，Gemzar，患者想休息
P0=0, P1=0, P2=0。**"treating your cancer to help you feel better and live longer, not to cure it"** — **对困难处境的最佳通俗表达之一**。"walking for 10 minutes three times a day" ✓。诚实、有共情、可操作。

---

## 全局总结

### 61个样本逐一审查完毕

| 严重等级 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| **P0 (幻觉/编造)** | **0** | 0% | 全部61个样本中没有发现任何编造信息 |
| **P1 (重大错误)** | **2** | 3.3% | Sample 11: 脑部新病灶遗漏; Sample 81: 淋巴结数误读 |
| **P2 (小问题)** | **~25** | — | 语法、重复、占位符、术语未解释等 |
| **无问题** | **~36** | 59% | 完全准确、通俗、无遗漏 |

### P1 详情

| Sample | 问题 | 根本原因 |
|--------|------|----------|
| **11** (coral 151) | 信件说"cancer is stable, no new problems"但脑MRI显示2个新病灶，遗漏了进展信息 | response_assessment中有脑部新病灶数据，但信件生成时选择性报告了body CT（稳定）而忽略了brain MRI（进展）|
| **81** (coral 221) | 信件说"spread to 24 lymph nodes"，但Stage II不可能有24个阳性LN | 原文"November 24 lymph nodes positive"是脱敏后的文本，"November 24"极可能是日期或被脱敏的数字（如"2/4"），被误读为24个淋巴结 |

### P2 常见模式

| 模式 | 频次 | 涉及Samples |
|------|------|------------|
| "advanced stage, the cancer has spread"语法断裂 | 4 | 1, 4, 90, + others |
| letrozole/用药重复提及 | 5 | 19, 28, 39, 71, + others |
| [REDACTED]→"a specific treatment"占位符 | 4 | 6, 13, 63, 93 |
| "Advance care planning was not discussed"措辞生硬 | 2 | 77, 91 |
| 医学术语未按prompt解释 | 3 | 36 (adjuvant), 67 (ER+), 89 (adenocarcinoma) |
| 化疗剂量过于详细（mg/m2） | 2 | 1, 91 |
| 无关实验室值（旧血糖、正常肌酐） | 2 | 4, 19 |

### 亮点 — 医生视角的积极发现

1. **受体解释始终正确**：ER+→"grows in response to hormones (estrogen)"，HER2-→"does not have extra HER2 protein"，TNBC→"cancer cells lack three common receptors"。61个样本中未发现受体方向性错误。

2. **医学术语通俗化出色**：
   - peritoneum → "the lining of your abdomen (belly area)" (Sample 0)
   - seroma → "a pocket of clear fluid that can form after surgery" (Sample 52)
   - fat necrosis → "a common reaction after breast surgery" (Sample 72)
   - metaplastic carcinoma → "rare type of breast cancer" (Sample 65)
   - pneumonitis → "lung irritation" (Sample 21)
   - neoadjuvant → "treatment before surgery to shrink the cancer" (Sample 29)

3. **情绪支持恰当**：当原文记录患者"distressed"、"very scared"、"tearful"时，信件包含温暖但不说教的支持句 ✓

4. **治疗反应诚实传达**：
   - "treatment is not working as well as we hoped" (Sample 90)
   - "cancer is not responding well" (Sample 10)
   - "treating your cancer to help you feel better and live longer, not to cure it" (Sample 99)

5. **TNM术语成功转换**：pT1c(m)N1(sn)M0 → "early stage" (Sample 28), pT2N2 → "locally advanced" (Sample 45)

6. **归因链可靠**：大多数句子可追溯到原文出处，归因链完整率84%

### 作为医生的整体评价

这些患者信件整体质量很高。如果我是患者的主治医生，我会愿意把这些信件（经过P1/P2修正后）发给我的患者。它们做到了：

- **不编造** — 0个P0，所有信息都有原文依据
- **不太漏** — 覆盖了诊断、治疗、计划、随访等重要信息
- **通俗** — 平均阅读等级7.3（82%在8年级以下），大多数患者能看懂
- **温暖** — 情绪支持自然、不居高临下

需要修正的2个P1问题都是可修复的：Sample 11需要在response_assessment中区分body CT和brain MRI的不同结果；Sample 81需要更好地处理脱敏文本导致的数字误读。

