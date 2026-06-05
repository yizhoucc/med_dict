# 临床上下文规则（药物分类的"医生常识"层）

> 定位：这是 harness 的**临床知识层**——把"同一药在不同语境是否为癌症相关"的医生经验编入。
> **可辩护**：编入的是真实临床知识（非硬编码测试答案），且**可推广**——换癌种只需继续编入更多医生常识。
> 论文叙事：少量临床规则已显著提升；规模化编入医生经验，效果更好。

## 三层机制
1. **通用药典**(data/drug_dictionary.tsv): 药名→ ONCOLOGY / SUPPORTIVE / SUPPORTIVE_OR_HOME / NON_CANCER。
2. **上下文规则**(SUPPORTIVE_OR_HOME): 同一成分既可癌症支持、又可家庭慢病用 → 只在 A/P 治疗语境出现时算支持药。
3. **忠实性**: 支持药必须在原文出现，否则视为幻觉删除。

## 已编入的临床上下文规则（示例，可扩充）
| 药 | 癌症支持语境 | 家庭语境 | 规则 |
|---|---|---|---|
| gabapentin/pregabalin/duloxetine | 化疗神经病变(CIPN) | 糖尿病/慢性痛/抑郁 | SUPPORTIVE_OR_HOME：仅 A/P 出现才算支持 |
| promethazine (单方) | 化疗止吐(Phenergan) | — | SUPPORTIVE(止吐) |
| promethazine-**dextromethorphan** | — | 复方止咳糖浆 | NON_CANCER（复方剂型≠止吐单方） |
| omeprazole/PPI、lorazepam、acetaminophen/NSAID、furosemide、prednisone | 化疗 GI 预防/预期恶心/癌痛/输液反应/方案类固醇 | 慢性 GERD/焦虑/OTC 止痛/CHF/慢性炎症 | SUPPORTIVE_OR_HOME：A/P 语境决定 |
| 眼药/鼻喷/外用/维生素/降压/降糖/他汀/抗组胺 | — | 纯家庭慢病 | NON_CANCER：直接剔除 |
| pancrelipase/Creon、octreotide | 胰腺癌酶替代/梗阻 | — | SUPPORTIVE（但须在原文出现，防幻觉）|

## 为什么这经得起 challenge
- 编入的是**临床事实**（"gabapentin 治 CIPN 是支持治疗"是医学共识），不是"ROW9 答案=X"。
- 规则**与具体测试笔记无关**，对任何新笔记同样适用 → 可泛化。
- 透明可审：规则成文于此，审稿人可逐条核对其临床合理性。
