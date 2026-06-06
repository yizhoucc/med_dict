# 根因深挖：分期/转移 + 疗效评估（为"拉开 PL vs BL 差距"做准备）

> 2026-06-05。目标：把当前打平/BL占优的维度（Q10 STAGE、Q11 NOHALLUC、Q7 RESP、T6 DISTMET）
> 找出根因 → 改 PL 的多层 hardness → 让 PL 在这些维度也碾压 BL，从而能加回这些评分题。
> 数据：pipeline_*_FIXED.txt（最新，含药典+T4 fix）。WSL+vllm 已就绪可重跑验证。

## 当前比分（淘汰题，来自 SCORING_RESULTS_40.md）
- Q10 STAGE   PL10 : BL9  (0.51) — 残留 stage bug 拖累
- Q11 NOHALLUC PL4 : BL9  (0.44) — 幻觉拖累
- Q7  RESP    PL8 : BL2 TIE30 (0.57) — 大量打平
- T6  DISTMET PL6 : BL7  (0.49) — BL 保守更安全

---

## A. 分期/转移 根因（逐样本核对原文得出）

### 根因 A1：双字段冗余且不 reconcile —— PL 自找的矛盾
PL 有 `Distant Metastasis` 和 `Metastasis` 两个语义重叠字段，模型分别填 → 互相打架。BL 只有一个字段，没这病。
明确矛盾样本：
- **breast ROW2**：DistMet="No" 但 Metastasis="Yes (chest wall, axillary LN, liver)"
- **pdac ROW6**：DistMet="No" 但 Metastasis="Yes, to liver"，Stage 还写 "Stage IV"（三方打架）
- **pdac ROW20**：DistMet="No" 但 Metastasis="Yes (liver, peritoneum)"，Stage="Stage IV"
现有 hook：POST-STAGE-DISTMET 只清 `Distant Metastasis`，**`Metastasis` 字段无任何 hook 清理** → 漏网根源。

### 根因 A2：局部复发 / regional 病灶 被误判为 distant met → 武断升 Stage IV
- **breast ROW4（P0，与原文直接矛盾）**：原文 PET/CT 白纸黑字 "right lateral breast mass compatible with
  **recurrent disease. No evidence of hypermetabolic metastatic disease**" + "Bones: No metastases"。
  PL 却判 Stage IV / DistMet="Yes, to right lateral breast"。把**原发部位的局部复发**当远处转移。
- **breast ROW2**：chest wall + axillary LN（regional）+ liver（原文是 cyst 囊肿）全塞进 Metastasis。
现有 hook：POST-DISTMET-NOMET（原文"no metastatic disease"→改No）、POST-DISTMET-REGIONAL（regional→No）
**都没在 ROW4 生效**：(1) 没同步下调 Stage 字段；(2) regex 未覆盖"recurrent disease at 原发部位"。

### 根因 A3：疑似（suspicious/early evidence）被武断升为确诊 Stage IV
- **pdac ROW6**：原文 "early evidence of disease recurrence" + 肝灶 "suspicious for / suggestive of metastatic
  disease"（疑似，未确诊）→ PL 写 "Stage IV / metastatic"。BL 写 "locally advanced" + "Not sure"（对）。
- 对比：breast ROW6/ROW20、pdac ROW1 正确用了 "Suspected Stage IV (pending confirmation)" / "Not sure"。
  → 模型**行为不一致**：有时保守标 suspected，有时武断升确诊。需规则统一。

### 根因 A4：编造 "Originally Stage X"（幻觉）
- **breast ROW2** "Originally Stage IIA"（原文 1994 是 1cm grade1 N0，约 Stage I，未明说 IIA）
- **pdac ROW2 / ROW6** "Originally Stage IIB" —— 原文**完全没提 stage**（grep 核实无输出）= 纯套话幻觉
- （pdac ROW8 原文确有 AJCC 病理分期，那个有依据，不算幻觉）
现有 hook：POST-STAGE-VERIFY-ORIG / POST-STAGE-VERIFY-NOTE 存在但**对 IIA/IIB 漏网**，验证逻辑不够强。

---

## B. 疗效评估 根因

### 根因 E：verbose —— 复制整段 CT/findings 报告进 response
- **breast ROW2**：response 长 1263 字符（把整个 findings 段抄进来）
- pdac 多数 400–767 字符（治疗中需对比 before/after 部分合理，但复制 CT 细节过头）
- BL 普遍简洁 → 很多 TIE。PL 要赢得靠"简洁+结论更准"，不是堆细节。

### 根因 F：无抗癌治疗却下 response 结论 + 自相矛盾
- **breast ROW2**：仅在用 zoledronic acid（支持药，非抗癌治疗），PL 却判 "stable / no progression"，
  且该患者是 locally recurrent，结论自相矛盾。BL "Not assessed" 反而安全。

### 根因 G：response 字段混入 plan
- **breast ROW19**：response 写 "NED on exam. **Continue exemestane 25mg daily. Recommend checking
  estradiol level...**" —— 后半是治疗计划，不该出现在 response（答非所问）。

---

## C. 修复方案（全部 = 通用临床规则，非照 test note 硬编码 → on-thesis 可泛化）

| 根因 | 修复 | 改动位置 | 预期翻盘样本 |
|---|---|---|---|
| A1 双字段矛盾 | 新 hook **POST-MET-RECONCILE-FULL**：Stage / Distant Met / Metastasis 三字段强制一致（以最保守且有原文支持者为准）；清空/同步 `Metastasis` 字段 | run.py ~2194 | breast2, pdac6, pdac20 |
| A2 局部复发误判 | 强化 POST-DISTMET-NOMET：命中原文"no (evidence of) metastatic disease"时**同步下调 Stage**；原发部位"recurrent"不计 distant | run.py ~2523 | breast4, breast2 |
| A3 疑似升确诊 | 新规则：原文含 suspicious/suggestive/early evidence/cannot exclude → 标 "Suspected Stage IV (pending confirmation)"，不写确诊 | run.py stage 段 | pdac6 |
| A4 编造 Originally | 强化 POST-STAGE-VERIFY：`Originally Stage X` 的 X 必须在原文出现，否则删该前缀 | run.py ~2245 | breast2, pdac2, pdac6 |
| E verbose | 新 gate：response 压缩到「结论句 + ≤1 句关键证据」，禁止整段复制 findings | ult.py / letter 段 | breast2 + pdac 普遍 |
| F 无治疗下结论 | response gate 先判「是否在抗癌治疗」（支持药不算）；否则只陈述基线疾病状态 | ult.py response gate | breast2 |
| G 混入 plan | response gate 剔除 Continue/Recommend/will 等 plan 动词句 | ult.py response gate | breast19 |

## D. 预期收益（修复后重跑 40 + 重评）
- Q10 STAGE   PL10:BL9 → 预期 PL 大胜（breast2/4 + pdac6 翻正，BL 的 9 胜多来自这些 bug）
- Q11 NOHALLUC PL4:BL9 → 预期翻正（清除 Originally-Stage 幻觉 + 局部复发误判）
- Q7  RESP    0.57 → 预期过 0.65（更简洁更准，PDAC 治疗中样本 PL 给对 stable/progression 结论而 BL 漏）
- T6  DISTMET 0.49 → 预期翻正（三字段一致 + 保守升级）
→ 可加回 **分期一致性、幻觉、疗效** 三类评分题，正好是你想要的"分期转移/疗效/特定类别幻觉时态"。

## E. 诚实边界
所有修复都是通用临床知识规则（regional≠distant、疑似≠确诊、原文无 stage 不编、response 不复制报告），
不读 test note 特异细节定制。审稿人问"换 40 个新 note 行不行"→ 站得住。
不碰的：lab plan（结构性 BL 占优）、referral（已定位打平）。
</content>
</invoke>
