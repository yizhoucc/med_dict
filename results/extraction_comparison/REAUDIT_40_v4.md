# 重审 v4 (Round4 FIX5/FINAL 全 40) — 确认 P0=0 + 无 round4 回归 + 最终 PL vs BL

> PL=pipeline_*_FINAL.txt(=FIX5), BL=baseline_extract_*_json.txt. subagent极简初审+主复核P0/P1. 禁脚本判断.
> 已知边缘(非确定性/可辩护): b13/b15 stage. round4修复(7主题)FIX4/FIX5已spot验证landed.

## STATUS: [ ]breast1-20 [ ]pdac1-20

## breast

## pdac

## breast 1-20 (v4) — P0=0 全部 ✅
- round4验证held: b8 "Stage IIA(pT2N1a)"✓ / b14 no-tamoxifen✓ / b19 findings净(NED)✓ / current_meds过滤 5/5✓(BL系统性灌非癌药) / b2 curative✓.
- PL残留 P1 (均stage字段, 少量):
  - b10 Stage="Stage IIA (pT2N1)" 过度细化(note仅"clinical stage II",应cT2N1)——model直输+NOBASIS留(TNM锚).
  - b11 Type "PR+" 但note "PR pending"(把pending当阳性).
  - b13 Stage "III"(node+推断,可辩护,BL="Not specified"保守).
  - b15 Stage "III" 应IV(de novo MBC, 非确定性: FIX4=IV/FIX5=III; MBC hook与regional-downgrade竞争).
  - b20 Stage "Early stage"(我的early-stage capture触发,但DistMet=Suspected liver/lung→矛盾,应加mets-guard).
- BL系统性弱点: current_meds灌全部非癌家用药(多数sample)、goals方向错(b1 palliative/b15 curative-for-MBC)、Stage漏判.
- breast计: PL P0=0, P1≈5(全stage类+1 PR), 逐sample PL≥BL绝大多数; BL反超仅b13/b15/b20 stage保守 + b18 radiotherapy-reason.

## pdac 1-20 (v4) — P0=0 全部 ✅
- round4验证held: pdac5 current_meds清空✓ / pdac6 surveillance✓ / pdac11 lab净✓ / pdac12 ipilimumab→None✓ / pdac14 Tylenol✓ / pdac16 Stage IIB✓ / pdac4 non-secretor捕获✓.
- current_meds: PL 5/5每批碾压BL(BL系统性把非癌门诊药当现用药+漏化疗).
- PL残留 P1:
  - pdac7/8 current_meds空漏active化疗(gem+Abraxane/"continue treatment"; recent_changes/therapy有捕获,gate过保守).
  - **pdac8 Advance_care "transition to hospice"=编造**(note仅"prioritize QoL"无hospice)——round4 ACP hook过宽,**已收紧+commit**(仅显式hospice触发).
  - pdac10 goals "palliative" 应curative-intent(downstaging手术,可辩).
  - pdac11 Patient type "Follow up" 应New patient(second opinion新患者).
  - pdac13 current_meds含s/p gemcitabine(pause systemic therapy).
  - pdac15 stage pT2N3 vs ypT3N2(可辩).

## ✅✅✅ Round4 全 40 v4 重审 — 总评
- **P0 = 0 (全 40)** ✅✅ —— 三轮从 2→1→0→**0 维持**.
- **round4 七主题修复全部确认 held**: #4 stage(b8/pdac16/b15/pdac7/b20)、#5 surveillance(pdac6/15/b2)、#7 findings(b19)、#8 supportive(pdac7/14)、#9 regex(pdac11)、#10 否决药(b14/pdac12/pdac7)、#11(ACP/non-secretor/germline/SMS).
- **PL 残留 P1 ≈ 9**(全 40): 多为 stage 字段细节(b10/b13/b15/b20/pdac10/15)+ current_meds active-chemo gate 过保守(pdac7/8)+ 个别(b11 PR-pending/pdac11 Patient type). 无 P0.
- **1 自伤回归已修**: pdac8 ACP hospice 过宽 → 收紧.
- **BL**: P0=0 但**几乎每个 sample 都有 P1**——最系统性=current_meds 倾倒非癌门诊药+漏真正化疗(全 40 反复); 加 goals 方向错、stage 漏判.
- **PL vs BL 字段级**: PL 在绝大多数 sample/字段碾压 BL; BL 反超 PL 仅极少数(b13/b15/b20/pdac10 stage 保守、pdac11 Patient type、b18 radiotherapy-reason). 
- **核心护城河**: current_meds 三分法+药物词典(全 40 完胜 BL)、stage 规则、goals 决策树、疑似≠确诊、字段路由、plan 时态. 这些正是 harness 相对裸跑同底座的增益.
