# 第三轮迭代追踪 (Round 3) — PL 全方位碾压 BL

> 本文档追踪第 3 轮 hook+prompt 迭代的全过程：我们做了什么、获得什么结果、发现什么问题、准备做什么。持久化、跨上下文可续。

## 总目标 (不变)
PL(pipeline: 多阶段提取+5 gate+POST hook+词典) 在全 40 held-out sample 的**每个字段**全方位碾压 BL(单 prompt 裸跑同底座 Qwen2.5-32B)。消融不变量=同底座模型 + 同字段 schema；可自由操作 PL 侧(加 stage/加 hook/改多层 prompt)。四不可妥协原则：①精确忠实(绝不幻觉)②不遗漏③简单词④通俗。

---

## 已完成的过程与结果 (Round 1-2 回顾)

### Round 1 (14 个 hook 修复)
基于上一次全字段审查(`FULL_AUDIT_40.md`, 发现 2 P0 + ~55 P1)，修了 14 个 hook：
1. POST-STAGE-NOBASIS (b4 stage 幻觉清零)
2. POST-MEDS-REGIMEN-FAB (b12/b17 化疗方案幻觉清空) + 后续 guard(仅 explicit option/refusal 框架才删, 去 'receiving' 假阳)
3. 删 ' pt ' 触发 (physical therapy 幻觉)
4. AUTO_SUPPLEMENT_SKIP + 同义词去重 (Medication_Plan "; also:" 噪声)
5. POST-SUPP-SUPPLEMENT (Creon/Xarelto/Lovenox 恢复, 9 pdac)
6. IV-CHECK regimen 增强 + POST-MEDS-ENZYME-STRIP (active 化疗)
7. POST-PLAN-ROUTING (FNA/port/germline 归位) + POST-LAB +TTE/MUGA/EKG
8. POST-PLAN-TEMPORAL (已完成影像/labs 清空)
9. POST-GENETIC-PENDING (UCSF500)
11. POST-GOALS-PALLIATIVE-CHECK (b2/b4)
12. POST-TYPE-MET-CONSISTENCY (疑似/局部≠转移确诊)
13. POST-STAGE-LOCALLY-ADVANCED (pdac1)
14. POST-MEDS-DOUBLET (b7/pdac3)
+ CROSSCHECK 去括号 (pdac10 FOLFOX)

### Round 2 (全量重跑 FIX2 + 全 40 重审)
- 全量重跑 40(letter-off, run.py+vLLM port 8000, 0 错误)→ `pipeline_breast_FIX2.txt` / `pipeline_pdac_FIX2.txt`
- 全 40 subagent 逐字段重审(主复核 P0/P1) → `REAUDIT_40_v2.md`
- **结果**: 14 修复 100% 生效、零回归; P0 2→1, P1 ~55→~33; 逐 sample PL≥BL 全部, 但 ~13 sample 个别字段仍被 BL 反超.

---

## 发现的问题 (Round 2 重审, 11 类主题, ~33 P1 + 1 P0)

1. **[P0] 亚型幻觉**(b20): Type 写 "invasive ductal carcinoma" 但病理仅 "invasive cancer with some lobular differentiation"(全文无 ductal).
2. **[P1 高频~8] plan 类含已完成/无依据项**: imaging_plan 把已做检查当计划(b4 mammogram/b13 brain MRI/b16 CT/pdac13 CT Chest); genetic_plan 把已出结果检测当计划(b17 BRCA/b18 mammaprint).
3. **[P1~6] current_meds 完整性/时态**: pdac9/18 漏 gemcitabine(gem 别名+resume 双药回填); pdac5 folfirinox 已完成(s/p+chemo break→空); pdac4 误清空(明列药勿删); pdac10 过窄; pdac12 CA19-9 取最新.
4. **[P1~5] Stage**: pdac7 vessel encasement>180°→III; pdac16 专 "Cancer Staging" 段漏读; b13 过度 III(应 IIB/not-staged); b15 MBC/cervical→IV; b20 "early stage" 漏.
5. **[P1~4] goals/response surveillance & 疑似**: pdac6/15 s/p resection+无 active→surveillance; b2 ordering bug; response 疑似软化.
6. **[P1~5] 字段路由残留**: b5 激素→procedure 剔除; b17 echo→lab 剔除; b20 port 残留 lab; pdac19 ERCP→procedure; pdac18 referral incoming("has been seen by").
7. **[P1~3] findings 纯度**: b19 findings=plan; pdac5/17 重复 lab; 重复 type/stage.
8. **[P1~3] supportive_meds**: pdac7 "Patient not taking" 过滤; pdac14 home 癌痛药纳入; b4 alendronate.
9. **[P1~3] plan regex 乱码/截断**: pdac11 Lab_Plan; pdac13/19 截断.
10. **[P1~3] therapy_plan 被拒药/None**: pdac7 None(continue 时禁); pdac12 ipilimumab + b14 tamoxifen(剔除 "chose X vs"/"declined").
11. **[P1~3] ACP/完整性**: pdac3/12 hospice; CA19-9 non-secretor(pdac4/13); recent_changes 漏 pause(pdac13); Referral SMS(pdac19).

---

## 第三轮准备做什么 (本轮计划)

用户指示: **先打包修 #1 → #2 → #3 → #6, 修完统一全量 40 测试, 再重审.**

### 本轮 4 个 targeted 修复包
- **#1 [P0] 亚型幻觉**: 新 hook — Type_of_Cancer 含 "ductal/IDC" 但病理/note 仅有 "lobular differentiation" 且无 "ductal"/"IDC" 字样 → 改写为 "invasive carcinoma with lobular differentiation"(不臆造 ductal). 守诚实边界.
- **#2 [高频] plan 类含已完成项**: 强化 imaging_plan + genetic_plan 时态过滤 — 排除已出结果/过去日期/已完成检查; imaging_plan 只保留 A/P 有 order 依据的 modality. 覆盖 b4/b13/b16/pdac13(imaging) + b17/b18(genetic).
- **#3 current_meds 完整性/时态**: (a) IV-CHECK 把 "gem"/"gemcitabine" 别名识别为双药, resume/hold regimen 从 HPI 回填完整方案(pdac9/18); (b) s/p+completed N cycles+chemo break → 清空(pdac5); (c) 排查 current_meds 误清空(pdac4, 勿删 "patient states to be taking").
- **#6 字段路由残留**: (a) procedure_plan 剔除激素/全身治疗药名(b5 AI); (b) lab_plan 剔除 echo/port(b17/b20); (c) procedure_plan 捕获 ERCP/biopsy 且有 procedure 时禁 "No procedures planned"(pdac19); (d) Referral 剔除 incoming("has been seen by"/"was seen by")(pdac18).

### 执行规则
- 每个 fix: 读样本 → 设计 hook(通用临床/文本规则, 不硬编码测试集) → 单元测试(目标样本触发 + 防回归控制) → commit+push.
- 4 个修完 → WSL 全量重跑 40(letter-off, fix_breast_nl/fix_pdac_nl) → 下载 FIX3 → 全 40 重审(subagent+主复核).
- 不碰 #4/#5/#7/#8/#9/#10/#11(留第 4 轮).

### STATUS
- [x] #1 亚型幻觉 — POST-SUBTYPE-VERIFY (b20 ductal→invasive carcinoma) ✓单测
- [x] #2 plan 已完成项 — POST-PLAN-TEMPORAL bare-imaging 须 A/P future-order(b4/b13/b16/pdac13) + POST-GENETIC-PLAN-COMPLETED(b17/b18) ✓单测
- [x] #3 current_meds — DOUBLET 扫note+标准双药对(pdac9/18) + POST-MEDS-COMPLETED-CHEMO(pdac5/17) ✓单测; pdac4确认非癌药PL空正确(不改)
- [x] #6 字段路由 — PROC黑名单+激素(b5)/POST-LAB+echo保护lab内容(b17/b20)/POST-PROCEDURE-ENDO ERCP(pdac19)/POST-REFERRAL-INCOMING(pdac18) ✓单测
- [x] 全量重跑 FIX3 — FIX3-final (含5个patch), 0错误, 11/11 targeted fix验证通过
- [x] 全 40 重审完成 — P0=0(从1清零), P1 ~33→~25, round3修复全held(pdac2自伤已修待验,pdac13 imaging 1边缘); 剩余~25 P1全是deferred主题#4/#5/#7/#8/#9/#10/#11 → round4. 详见 REAUDIT_40_v3.md

---

## 关键文件
- PL 输出(当前): `pipeline_breast_FIX2.txt` / `pipeline_pdac_FIX2.txt`; 重跑后→ FIX3
- BL 输出: `baseline_extract_breast_json.txt` / `baseline_extract_pdac_json.txt`
- 审查记录: `REAUDIT_40_v2.md`(Round 2) / 本文档(Round 3)
- 代码: `run.py`(POST hooks, FINAL 实际路径) / `prompts/extraction.yaml` + `prompts/pdac/extraction.yaml` + `prompts/plan_extraction.yaml` + `prompts/pdac/plan_extraction.yaml`
- WSL: `ssh wsl`, python=`/home/yc/miniconda3/envs/medllm/bin/python`, 跑法 systemd-run --user(linger已开), vLLM port 8000


## FIX3-final 验证 (11/11 OK)
- #1 b20 ductal→invasive carcinoma ✓ | #2 b4/b16(CT删US留)/b17/b18 imaging+genetic已完成清空 ✓ | #3 pdac9/18 +gemcitabine, pdac5 chemo-break清空 ✓ | #6 b5 AI出procedure, b17 echo出lab, pdac19 ERCP捕获, pdac18 incoming→None ✓
- 注: FIX3首跑暴露5个'被后置hook覆盖'缺口(b16多modality/b17b18 genetic-SEARCH重填/pdac5 'currently on break'/pdac19 ERCP被覆盖), 已加patch(clause-split/RESULT-CHECK增强/active正则\b/ENDO-FINAL)并最终重跑验证.

---

# Round 4 (deferred 7 主题) — 进行记录
## 修了什么 (commits d29b0ea1..ed382b2c)
- #4 Stage: NOBASIS锚点放宽(stage IIA/(m) TNM,b8) + POST-STAGE-EXPLICIT(明写"Stage X(cTNM)",pdac16/b8) + POST-STAGE-MBC(de novo MBC→IV,b15) + LOCALLY-ADVANCED血管包绕(pdac7)+防generic(b13)
- #10: POST-PLAN-REJECTED-DRUG(b14 tamoxifen/pdac12 ipilimumab) + POST-THERAPY-NONE-FIX(pdac7)
- #9: POST-PLAN-GARBAGE-CLEAN(pdac11/19 lab乱码, pdac11/13 genetic截断)
- #5: POST-GOALS-FINAL(pdac6/15 surveillance, b2 ordering) + POST-RESPONSE-SUSPECTED-SOFTEN(pdac6)
- #8: POST-SUPP-NOTTAKING(pdac7) + POST-SUPP-HOMEPAIN(pdac14)
- #11: ACP hospice(pdac3/12) + POST-NONSECRETOR(pdac4/13) + POST-GERMLINE-PENDING(b20) + POST-REFERRAL-SMS(pdac19)
- new: IV-CHECK continue+on(pdac8) + POST-PATIENT-TYPE-ONGOING(pdac19)
- residual(FIX4验证后补): REJECTED窗口90+trial信号 / NOTTAKING窗口140 / HOMEPAIN非癌痛guard

## FIX4 全量重跑验证 (14/17 spot-check OK)
✅ b8 IIA / pdac16 IIB / b15 IV / pdac7 III / b20 Early / b14 no-tamoxifen / pdac7 therapy填充 / pdac11 lab清 / pdac6+pdac15+b2 goals / pdac14 Tylenol / b19 findings净 / pdac19 Follow up / ACP hospice / non-secretor / germline pending.
⚠️ 3个residual(已code-fix+单测,待下次重跑确认): pdac12 ipilimumab(窗口) / pdac7 ondansetron(窗口)+oxycodone(非癌痛误加) / b13 "Stage III"(model直输cT2N1,难/可辩护,留下轮).

## STATUS
- [x] #4/#5/#7/#8/#9/#10/#11 + new 全部 code 完成 + 单测
- [x] FIX4 全量重跑 (0错误, 14/17 landed)
- [x] 3 residual code-fix + 单测
- [ ] 最终确认重跑 (验证3 residual) + 全40重审 → 待用户决定
