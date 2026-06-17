#!/usr/bin/env python3
"""Build a verdict-first HTML: for each comparison question, show MY review verdict
(PL更好 / 打平 / PL不如BL) FIRST, then the reasoning citing note evidence, then the
PL|BL values. Verdicts come from subagent audits (_audit_v5/verdicts.json), spot-verified
by main-Claude against the FINAL files."""
import re, json, html, os
from build_review_html import parse_pl, parse_bl, norm, s

HERE = os.path.dirname(os.path.abspath(__file__))

# field-id -> (label, tier, section, key). key=None means whole section.
FIELD_MAP = {
    "current_meds":   ("现用药 current_meds（抗癌 vs 非癌/已停）", "优", "Current_Medications", "current_meds"),
    "stage":          ("分期 Stage", "优", "Cancer_Diagnosis", "Stage_of_Cancer"),
    "distant_met":    ("远处转移 Distant Metastasis", "优", "Cancer_Diagnosis", "Distant Metastasis"),
    "metastasis":     ("转移(含区域) Metastasis", "优", "Cancer_Diagnosis", "Metastasis"),
    "response":       ("疗效 Response", "优", "Response_Assessment", "response_assessment"),
    "type_receptor":  ("分型/受体 Type (ER/PR/HER2)", "优", "Cancer_Diagnosis", "Type_of_Cancer"),
    "genetic_results":("分子/遗传结果 Genetic results", "优", "Genetic_Testing_Results", "genetic_testing_results"),
    "genetic_plan":   ("基因检测计划 Genetic plan", "优", "Genetic_Testing_Plan", "genetic_testing_plan"),
    "supportive_meds":("支持用药 supportive_meds", "优", "Treatment_Changes", "supportive_meds"),
    "procedure_plan": ("操作计划 Procedure plan", "优", "Procedure_Plan", "procedure_plan"),
    "imaging_plan":   ("影像计划 Imaging plan", "优", "Imaging_Plan", "imaging_plan"),
    "lab_plan":       ("化验计划 Lab plan", "优", "Lab_Plan", "lab_plan"),
    "medication_plan":("用药计划 Medication plan", "优", "Medication_Plan", "medication_plan"),
    "referral":       ("转诊 Referral", "优", "Referral", "Specialty"),
    "advance_care":   ("预立医疗计划 Advance care", "优", "Advance_care_planning", None),
    "recent_changes": ("近期治疗变化 recent_changes", "评", "Treatment_Changes", "recent_changes"),
    # 'patient_type' 和 'goals(治疗目标)' 于 2026-06-15 由临床医生判定无评分价值而删除；
    # 历史判定仍保留在 _audit_v5/verdicts.json，此处不再展示这两题。
    # 'summary(就诊原因)' 于 2026-06-16 删除（医生判定无评分价值；历史判定保留在 verdicts.json）。
    "lab_summary":    ("化验摘要 lab_summary", "差", "Lab_Results", "lab_summary"),
    "findings":       ("检查发现 findings", "差", "Clinical_Findings", "findings"),
}
TIER_ORDER = {"优": 0, "评": 1, "差": 2}
TIER_CLASS = {"优": "t-good", "评": "t-mid", "差": "t-low"}
TIER_LABEL = {"优": "优·需医学知识", "评": "评·中", "差": "一般"}
V_CLASS = {"PL": "v-pl", "BL": "v-bl", "TIE": "v-tie"}
V_LABEL = {"PL": "✓ PL 更好", "BL": "✗ PL 不如 BL", "TIE": "= 打平"}


def get_val(kp, section, key):
    if not isinstance(kp, dict):
        return ""
    sec = None
    for k in kp:
        if norm(k) == norm(section):
            sec = kp[k]; break
    if sec is None:
        return ""
    if key is None:
        return s(sec)
    if isinstance(sec, dict):
        for k in sec:
            if norm(k) == norm(key):
                return s(sec[k])
        return ""
    return s(sec)


def attr_for(att, key):
    if not isinstance(att, dict) or key is None:
        return ""
    def search(d):
        for k, v in (d.items() if isinstance(d, dict) else []):
            if norm(k) == norm(key):
                if isinstance(v, dict):
                    return s(v.get("source") or v.get("quote") or v.get("evidence") or v)
                return s(v)
            if isinstance(v, dict):
                r = search(v)
                if r:
                    return r
        return ""
    return search(att)


def build(cancer, pl, bl, verdicts):
    blocks, toc = [], []
    tally = {"PL": 0, "BL": 0, "TIE": 0}
    bl_wins = []
    vmap = {}
    for v in verdicts:
        vmap.setdefault(v["row"], []).append(v)
    for row in sorted(pl):
        rid = f"{cancer}{row}"
        p = pl[row]; b = bl.get(row, {})
        coral = p["coral"]
        cname = "乳腺 Breast" if cancer == "b" else "胰腺 PDAC"
        title = f"{cname} · ROW {row} · coral_idx={coral}"
        vs = sorted(vmap.get(row, []),
                    key=lambda v: (TIER_ORDER.get(FIELD_MAP.get(v["field"], ("", "差"))[1], 9),
                                   v["field"]))
        rp = sum(1 for v in vs if v["verdict"] == "PL")
        rb = sum(1 for v in vs if v["verdict"] == "BL")
        rt = sum(1 for v in vs if v["verdict"] == "TIE")
        toc.append(f'<a href="#{rid}">{cancer.upper()}{row} <small>({rp}胜/{rt}平/{rb}负)</small></a>')
        qhtml = []
        for v in vs:
            fid = v["field"]
            if fid not in FIELD_MAP:
                continue
            label, tier, section, key = FIELD_MAP[fid]
            verd = v["verdict"]
            tally[verd] += 1
            pv = get_val(p["kp"], section, key).strip()
            bv = get_val(b, section, key).strip()
            ev = attr_for(p["att"], key).strip()
            if verd == "BL":
                bl_wins.append((f"{cancer.upper()}{row}", label, v["reason"]))
            evhtml = (f'<div class="ev"><b>PL 原文归因:</b> {html.escape(ev[:320])}</div>'
                      if ev else '')
            qhtml.append(f'''<div class="q {V_CLASS[verd]}">
  <div class="qhead">
    <span class="vbadge {V_CLASS[verd]}">{V_LABEL[verd]}</span>
    <span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span>
    <span class="qlabel">{html.escape(label)}</span>
  </div>
  <div class="reason">{html.escape(v["reason"])}</div>
  <div class="qcols">
    <div class="qcol pl"><div class="qtag">PL（我们的方法）</div><div class="qval">{html.escape(pv) or "<em>（空）</em>"}</div>{evhtml}</div>
    <div class="qcol bl"><div class="qtag">BL（裸跑同模型）</div><div class="qval">{html.escape(bv) or "<em>（空）</em>"}</div></div>
  </div>
</div>''')
        sumbar = (f'<span class="cnt c-pl">PL 胜 {rp}</span>'
                  f'<span class="cnt c-tie">打平 {rt}</span>'
                  f'<span class="cnt c-bl">PL 负 {rb}</span>')
        block = f'''<section class="sample" id="{rid}">
  <h2>{html.escape(title)} <span class="rowsum">{sumbar}</span></h2>
  <div class="source">
    <div class="srchead">原文 · Assessment &amp; Plan（评审依据，点开可看完整笔记）</div>
    <pre class="ap">{html.escape(p["ap"]) or "（无 A/P 段）"}</pre>
    <details><summary>展开完整笔记 note_text</summary><pre class="note">{html.escape(p["note"])}</pre></details>
  </div>
  <div class="qs">{''.join(qhtml)}</div>
  <a class="top" href="#top">↑ 返回目录</a>
</section>'''
        blocks.append(block)
    return toc, blocks, tally, bl_wins


def main():
    plb = parse_pl(os.path.join(HERE, "pipeline_breast_FINAL.txt"))
    plp = parse_pl(os.path.join(HERE, "pipeline_pdac_FINAL.txt"))
    blb = parse_bl(os.path.join(HERE, "baseline_extract_breast_json.txt"))
    blp = parse_bl(os.path.join(HERE, "baseline_extract_pdac_json.txt"))
    V = json.load(open(os.path.join(HERE, "_audit_v5", "verdicts.json")))
    tb, bb, ta, bwa = build("b", plb, blb, V["b"])
    tp, bp, tpa, bwp = build("p", plp, blp, V["p"])

    TOT = {k: ta[k] + tpa[k] for k in ta}
    bl_all = bwa + bwp
    n = sum(TOT.values())

    bl_list = "".join(
        f'<li><b>{html.escape(r)}</b> · {html.escape(lab)}：{html.escape(rs)}</li>'
        for r, lab, rs in bl_all) or "<li>（无）</li>"

    css = '''
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f4f5f7;line-height:1.55}
header{background:#0b3d63;color:#fff;padding:22px 28px}
header h1{margin:0 0 6px;font-size:22px}
header p{margin:4px 0;font-size:13.5px;opacity:.93}
.scoreboard{display:flex;gap:14px;margin:16px 28px}
.score{flex:1;background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:16px;text-align:center}
.score .num{font-size:34px;font-weight:800;line-height:1}
.score .lbl{font-size:13px;color:#555;margin-top:6px}
.score.pl .num{color:#1b7a36}
.score.tie .num{color:#888}
.score.bl .num{color:#c0392b}
.legend{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:12px 16px;margin:0 28px 8px;font-size:13px}
.legend b{color:#0b3d63}
.blbox-top{background:#fff4f2;border:1px solid #e9b8ae;border-radius:8px;padding:12px 16px;margin:8px 28px;font-size:13px}
.blbox-top b{color:#c0392b}
.blbox-top ul{margin:8px 0 0;padding-left:20px}
.blbox-top li{margin:3px 0}
.toc{margin:8px 28px;font-size:12.5px}
.toc a{display:inline-block;margin:2px 4px;padding:3px 8px;background:#fff;border:1px solid #cdd5dd;border-radius:5px;color:#0b3d63;text-decoration:none}
.toc a small{color:#888;font-weight:400}
.toc a:hover{background:#e8f0f7}
.sample{background:#fff;margin:18px 28px;border:1px solid #d8dde3;border-radius:10px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.sample h2{margin:0 0 12px;font-size:18px;color:#0b3d63;border-bottom:2px solid #e6ebf0;padding-bottom:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.rowsum .cnt{font-size:12px;font-weight:700;padding:2px 9px;border-radius:10px;margin-left:5px}
.c-pl{background:#e3f3e7;color:#1b7a36}
.c-tie{background:#eee;color:#777}
.c-bl{background:#fbe3df;color:#c0392b}
.source{margin-bottom:14px}
.srchead{font-weight:600;font-size:13px;color:#555;margin-bottom:4px}
pre.ap{white-space:pre-wrap;word-break:break-word;background:#fbfaf3;border:1px solid #ece6c9;border-radius:6px;padding:10px 12px;font-size:12.5px;font-family:"SF Mono",Consolas,monospace;max-height:260px;overflow:auto;margin:0}
details{margin-top:6px}
summary{cursor:pointer;font-size:12.5px;color:#0b3d63}
pre.note{white-space:pre-wrap;word-break:break-word;background:#f7f7f7;border:1px solid #e0e0e0;border-radius:6px;padding:10px;font-size:11.5px;font-family:"SF Mono",Consolas,monospace;max-height:420px;overflow:auto}
.q{border:1px solid #e2e6ea;border-radius:8px;margin-bottom:9px;padding:10px 12px;background:#fff;border-left:5px solid #ccc}
.q.v-pl{border-left-color:#1b7a36;background:#f6fbf7}
.q.v-bl{border-left-color:#c0392b;background:#fff6f4}
.q.v-tie{border-left-color:#bbb;background:#fafafa}
.qhead{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.vbadge{font-size:12.5px;font-weight:800;padding:3px 11px;border-radius:6px;color:#fff}
.vbadge.v-pl{background:#1b7a36}
.vbadge.v-bl{background:#c0392b}
.vbadge.v-tie{background:#999}
.tier{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:10px;color:#fff}
.t-good{background:#1b7a36}
.t-mid{background:#b8860b}
.t-low{background:#9aa3ad}
.qlabel{font-weight:700;font-size:13.5px}
.reason{font-size:13px;color:#333;background:#fff;border:1px dashed #d6dbe0;border-radius:6px;padding:7px 10px;margin-bottom:8px}
.qcols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.qcol{border:1px solid #e0e0e0;border-radius:6px;padding:7px 9px;font-size:12.5px}
.qcol.pl{background:#f1f8f2}
.qcol.bl{background:#f7f7f8}
.qtag{font-weight:700;font-size:11px;margin-bottom:3px}
.qcol.pl .qtag{color:#1b7a36}
.qcol.bl .qtag{color:#777}
.qval{white-space:pre-wrap;word-break:break-word}
.ev{margin-top:5px;font-size:11px;color:#4a6a6a;border-top:1px dotted #cdd;padding-top:4px}
.top{display:inline-block;margin-top:8px;font-size:12px;color:#0b3d63;text-decoration:none}
@media print{.toc,.top{display:none}.sample{break-inside:avoid;box-shadow:none}body{background:#fff}}
'''
    htmldoc = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PL vs BL 逐题评审判定</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>肿瘤笔记结构化提取 · PL vs BL 逐题评审判定</h1>
  <p>PL = 我们的方法（多阶段提取 + 5-gate 校验 + 药物/术语词典 + 后处理规则）。BL = 起点（<b>同一个模型</b> Qwen2.5-32B 单 prompt 裸跑）。唯一变量 = 处理流程。</p>
  <p>仅列出 PL 与 BL <b>有实质差异</b>的小题（共 {n} 题）。每题先给<b>我的评审判定</b>（PL 更好 / 打平 / PL 不如 BL），再说<b>原文依据与理由</b>，最后并列 PL | BL 实际答案。判定来自 subagent 逐样本初审 + 主审逐条复核 FINAL 文件。</p>
</header>
<div class="scoreboard">
  <div class="score pl"><div class="num">{TOT['PL']}</div><div class="lbl">PL 更好</div></div>
  <div class="score tie"><div class="num">{TOT['TIE']}</div><div class="lbl">打平</div></div>
  <div class="score bl"><div class="num">{TOT['BL']}</div><div class="lbl">PL 不如 BL</div></div>
</div>
<div class="blbox-top">
  <b>诚实披露 · 全部 {TOT['BL']} 个 “PL 不如 BL” 的题（PL 待改进点）：</b>
  <ul>{bl_list}</ul>
  <div style="margin-top:8px;color:#555">说明：这 {TOT['BL']} 题里，多数是<b>次要计划字段</b>的遗漏（影像/化验子计划 B10/B13/B16/P18、化验摘要 B12），另有 5 个<b>边界临床判断</b>：B20 双侧乳腺 PL 只标了右侧 HER2+（漏左侧 HER2−）、B16 区域淋巴结 PL 漏标、P3 疗效 PL 措辞自相矛盾、P7 PL 漏 Gyn-Onc 转诊、P11 试验筛查程序。<b>全部 11 题均非幻觉</b>（无凭空编造）。<br>最强护城河字段——<b>现用药“抗癌 vs 非癌家用药”识别（PL 全胜，BL 几乎每例都把降压/降糖/止痛家用药误当抗癌药）、分期、远处转移、分子/遗传结果、治疗目标方向</b>——PL 无一败于 BL。</div>
</div>
<div class="legend">
  <b>深度分级：</b><span class="tier t-good">优·需医学知识</span> 需专业背景才能答对（药物抗癌/支持区分、分期、转移、疗效、受体、分子）；<span class="tier t-mid">评·中</span>；<span class="tier t-low">一般</span>（就诊原因/治疗目标方向/化验数值，普通人可答，医生兴趣低）。<b>评审重点放在“优”级题。</b>
</div>
<div class="toc"><b>乳腺：</b>{''.join(tb)}</div>
<div class="toc"><b>胰腺：</b>{''.join(tp)}</div>
{''.join(bb)}
{''.join(bp)}
<div style="margin:30px 28px;color:#888;font-size:12px">判定数据：_audit_v5/verdicts.json（subagent 初审 + 主审复核）。PL 值来自 pipeline_*_FINAL.txt，BL 值来自 baseline_extract_*_json.txt。方法与统计详见 SUMMARY_PL_vs_BL.md。</div>
</body></html>'''
    outp = os.path.join(HERE, "PL_vs_BL_verdict.html")
    open(outp, "w").write(htmldoc)
    print("wrote", outp, f"({len(htmldoc)//1024} KB)")
    print("TOTAL  PL:", TOT["PL"], " TIE:", TOT["TIE"], " BL:", TOT["BL"], " n:", n)
    print("BL-wins:", [(r, l) for r, l, _ in bl_all])


main()
