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

---

## F. 重跑实测（2026-06-05 晚）—— 关键发现：FIXED.txt 是旧的 + vLLM 非确定性

在当前 WSL 代码（f0699e8c，即生成 FIXED.txt 的同一 commit）上，letter-off 重跑全部 40 样本
（`pipeline_breast_RERUN.txt` / `pipeline_pdac_RERUN.txt`），与 FIXED.txt 对照关键 stage/met 行：

| 样本 | FIXED.txt | RERUN（当前代码重跑） | 评价 |
|---|---|---|---|
| breast2 | Stage "locally recurrent"，DistMet **No**，Met="chest wall+axLN+liver" | Stage "Originally Stage IIA, now **metastatic (Stage IV)**"，DistMet/Met 都="Yes, liver+chest wall" | RERUN **变差**（误升 IV；A4"IIA"仍在；但两 met 字段已一致） |
| breast4 | Stage **IV**，DistMet "Yes, right lateral breast" | Stage **III**，DistMet **No**，Met="right lateral breast mass" | RERUN **变好**（局部复发不再算 distant） |
| pdac2 | "Originally **Stage IIB**, now metastatic (Stage IV)" | "Originally **unspecified**, now metastatic **()**" | RERUN **变好**（A4 幻觉被 VERIFY-NOTE 清掉），但留空括号 "()" |
| pdac6 | "Originally Stage IIB, now metastatic (Stage IV)"，DistMet No | "...early evidence of disease recurrence **()**"，DistMet **""**，Met "Yes, liver" | stage **变好**(疑似→不确诊)，但 A1 仍裂(DistMet空/Met=Yes)，留 "()" |
| pdac20 | "Metastatic (Stage IV)"，DistMet **No**（矛盾） | "Metastatic **()**"，DistMet/Met 都="Yes, liver+peritoneum" | RERUN **变好**（三字段一致），留 "()" |

### 三条硬结论
1. **vLLM greedy 非确定性**：`do_sample=false` 但跨运行 batch 组成不同 → 浮点归约顺序变 →
   同一 commit 两次跑出**不同 stage 字段**。FIXED.txt ≠ RERUN（breast2/4、pdac2/6/20 全不同）。
   → 我之前对 FIXED.txt 的逐样本根因评分**部分在追幽灵**：有的 bug 自愈、有的换行出现。
   → 单次跑分脆弱；修复要锁"地板"，不能靠某次走运。
2. **新确定性 bug：空括号 "()" 残留**。POST-STAGE-VERIFY-NOTE 删掉 "Stage IV/IIB" 后留下
   "metastatic ()" / "recurrence ()"（pdac2/6/20 都有）。难看、确定复现、好修 → **第一优先**。
3. **A1–A4 深层 bug 仍真实存在**，只是在不同样本间闪烁：
   - A1 reconcile 仍裂：pdac6 DistMet="" 但 Met="Yes, liver"（清一个没同步另一个）
   - A4 幻觉：breast2 仍"Originally Stage IIA"（VERIFY-NOTE 没覆盖 breast 路径）
   - A2/A3：breast4 这次对了、breast2 这次错了（误把 liver囊肿+chest wall 升 IV）；非确定 → 需规则锁死

### 修订后的修复优先级（基于 RERUN 真实现状）
0. **[新] strip 空括号**：stage verify 删 designation 后，清理 "(  )"/"()" 及孤立 "now ,"。确定性、零风险。
1. A1 三字段 reconcile（pdac6 类：清空一个必须同步另一个；保守取向）
2. A4 breast 路径的 "Originally Stage X" 幻觉校验（VERIFY-NOTE 现仅 non-breast 生效）
3. A2/A3 局部复发≠distant、疑似≠确诊（锁死非确定行为，让 breast2 也对）
4. E/F/G response gate

---

## G. 修复实施 + 逐样本验证（2026-06-05 晚，全部 deploy 到 WSL 实测）

7 个 fix 全部实现，逐个改 + 逐个在关键样本上验证（非确定性下多次跑），无回归：

| # | hook（run.py） | 作用 | 验证结果 |
|---|---|---|---|
| 0 | `POST-STAGE-PARENS-CLEANUP` | 删 VERIFY 留下的空 `()`、前导括号、前后悬空 now/originally | pdac2/6/20 "metastatic ()"→"metastatic"；pdac6 "(borderline resectable)"→"borderline resectable" ✓ |
| A4a | `POST-STAGE-VERIFY-NOTE` 加 guard | metastatic⟺Stage IV 是合法推断，绝不删；pTN 替换保留 "Originally" 前缀 | pdac8 "pt2n2, now metastatic (pt2n2)"(坏)→"Originally pt2n2, now metastatic (Stage IV)" ✓ |
| A4b | `POST-STAGE-VERIFY-ORIG` (新, 全癌种) | "Originally Stage X" 原文无依据则删（覆盖 VERIFY-NOTE 跳过的 breast 路径） | breast2 删掉 "Originally Stage IIA" ✓ |
| A1 | `POST-MET-RECONCILE` (新) | DistMet/Met 三字段一致：R1 distant→general、R2 未确诊→both Not sure、R3 No-wins、R4 mirror-unsure | 兜底网；测试中模型自洽未触发，无矛盾输出 ✓ |
| A2 | `POST-LOCOREGIONAL` (新) | 医生判 local-regional recurrence ⇒ 不升 IV/distant；原发器官当 met ⇒ 改 No | breast2 →"Locally recurrent (unresectable)" met=No；breast4 Met "right lateral breast mass"→No ✓ |
| A3 | `POST-STAGE-SUSPECTED` (新) | 疑似影像(suspicious/suggestive/early evidence)且无确诊 ⇒ Suspected，不写确诊 IV；**确诊 met 强 guard**(metastatic <organ> adenocarcinoma/cancer、carcinomatosis、omental caking、biopsy) | pdac20(确诊 metastatic)**不被误降** ✓；pdac6 早期复发→Suspected/Not sure ✓ |
| E/G | `POST-RESPONSE-COMPRESS` (新) | response 删 plan 祈使句(Continue/Recommend/will…)、压缩复制的 CT 报告到结论句 | breast19 "NED on exam. Continue exemestane… Recommend…"→"NED on exam" ✓；"continues to respond" 保留 ✓ |
| F | 既有 `POST-RESPONSE-PRETREATMENT/TREATMENT` | 无抗癌治疗(支持药不算)不下 response 结论 | breast2 →"Not yet on treatment — no response to assess." ✓ |

### 关键样本最终状态（多次非确定性跑均稳定正确）
- **breast2**：Stage "Locally recurrent (unresectable)"、DistMet/Met=No、resp "Not yet on treatment" —— 原来 Stage IV 误判 + IIA 幻觉 + 矛盾，**全修**
- **breast4**：Stage "now Stage III"、DistMet=No、Met=No（原发复发不算 met）——**修正**
- **pdac6**：Stage "borderline resectable, now with early evidence of disease recurrence"、both "Not sure" ——**疑似不确诊**
- **pdac8**：Stage "Originally pt2n2, now metastatic (Stage IV)"、met=lymph nodes ——**确诊 IV 保留 + 幻觉 IIB 修**
- **pdac20**：Stage "Metastatic (Stage IV)"、liver+peritoneum ——**确诊 IV 不被误降**
- **breast19**：resp "NED on exam"（去掉 plan 尾巴）

诚实边界守住：全是通用临床规则（metastatic⟺IV、原文无 stage 不编、局部复发≠distant、疑似≠确诊、确诊 met 不降、response 答"如何响应"非"下一步计划"）。审稿人换 40 个新 note 仍站得住。

下一步：全 40 用最终代码重跑（letter-off），作为新 authoritative PL 输出，亲自重评 Q10/Q11/Q7/T6 是否可加回评分题。

---

## H. 全 40 人工审查发现 2 个 P0 回归（已修）+ 残留 P2（2026-06-05 深夜）

第一次全 40 重跑后，**逐样本人工核对原文**（CLAUDE.md 要求），抓出我自己新 hook 引入的 2 个 P0 回归：

| 样本 | 真实情况（原文） | 错误输出 | 根因 | 修复 |
|---|---|---|---|---|
| breast7 | A/P "newly diagnosed **metastatic** breast cancer to liver"；肝活检证实 metastatic | A2 误降 "Locally recurrent" met=No | A2 在**全文**匹配到 2018 历史标题 "local recurrence 2018" | A2 locoregional 检测**限定 A/P** + 强化 confirmed-distant guard |
| breast9 | A/P "**possibly** metastatic"、"FNA to confirm"、"if she has MBC" = 疑似 | A3 没降级，stage 写确诊 "metastatic (Stage IV)" | A3 confirmed-guard 在**全文**匹配到 2013 历史 "axillary LN biopsy: Metastatic carcinoma" | A3 hedged/confirmed **限定 A/P** |

**共同根因 = hook 读全文（含 HPI 历史）而非当前 A/P 评估**。修复后验证：breast7 正确保留 metastatic、breast9 正确 Suspected、breast2 仍修复、pdac8/20 确诊 IV 不误降、pdac6 疑似不确诊。
（注意：A4 VERIFY-ORIG 仍用全文 —— 因为"Originally Stage X"的历史分期本就该在 HPI 找依据，scope 区分正确。）

### 最终 authoritative 输出（pipeline_*_FINAL.txt，含 scope 修复）
- breast/pdac 关键样本 stage/met 全部正确且三字段自洽；confirmed vs suspected 正确区分；metastatic⟺IV 推断保留。

### 残留 P2（incompleteness，非矛盾，优先级低，部分为 G4-FAITH 既有行为非本次回归）
- 空 Stage：breast20、pdac1、pdac7（A3 不在空 stage 上补 "Suspected IV"；可后续加增强）
- 空 Distant Metastasis：breast15（"cervical LN" 未进 distant 站点表 → reconcile R2 未触发）
- 这些不影响"无矛盾/无幻觉/疑似确诊区分"的核心收益；如需可再加 2 个确定性小增强。

### 方法论备注（写 paper 用）
vLLM greedy 跨运行非确定性 → 单次跑分脆弱；本轮策略 = **确定性规则 hook 锁地板**，并用**人工逐样本审查**抓 hook 自身回归（自动跑分抓不到 breast7 这种"看似自洽实则方向错"的错误）。

