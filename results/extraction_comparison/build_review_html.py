#!/usr/bin/env python3
"""Build a self-contained HTML comparing PL (pipeline) vs BL (baseline) per sample,
for clinician review. Layout per sample: source note (1 col) -> PL | BL full extraction
(2 col) -> per-field comparison "questions" (2 col, depth-tiered, diffs highlighted)."""
import re, json, html, os

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_pl(path):
    T = open(path).read()
    parts = re.split(r'(?m)^RESULTS FOR ROW (\d+)\s*$', T)
    out = {}
    for i in range(1, len(parts), 2):
        row = int(parts[i]); b = parts[i+1]
        def grab(pat):
            m = re.search(pat, b, re.S); return m.group(1).strip() if m else ""
        note = grab(r'--- Column: note_text ---\n(.*?)\n--- Column: assessment_and_plan')
        ap = grab(r'--- Column: assessment_and_plan ---\n(.*?)\n--- Column: keypoints')
        kp_raw = grab(r'--- Column: keypoints ---\n(.*?)\n--- Column: attribution')
        att_raw = grab(r'--- Column: attribution ---\n(.*?)\n--- Column: (?:letter|traceability)')
        coral = grab(r'--- Column: coral_idx ---\n(.*?)\n')
        try: kp = json.loads(kp_raw)
        except Exception: kp = {}
        try: att = json.loads(att_raw)
        except Exception: att = {}
        out[row] = dict(note=note, ap=ap, kp=kp, att=att, coral=coral)
    return out

def parse_bl(path):
    T = open(path).read()
    parts = re.split(r'(?m)^ROW (\d+) \(coral_idx=(\d+)\)\s*$', T)
    out = {}
    for i in range(1, len(parts), 3):
        row = int(parts[i]); blk = parts[i+3-1]
        mj = re.search(r'\{.*\}', blk, re.S)
        try: out[row] = json.loads(mj.group(0)) if mj else {}
        except Exception: out[row] = {}
    return out

def norm(k): return re.sub(r'[^a-z0-9]', '', k.lower())

def leaf(kp, section, key):
    """nested lookup with normalized matching"""
    if not isinstance(kp, dict): return ""
    sec = None
    for k in kp:
        if norm(k) == norm(section): sec = kp[k]; break
    if isinstance(sec, dict):
        for k in sec:
            if norm(k) == norm(key): return sec[k]
        return ""
    if sec is not None and key is None:
        return sec
    return ""

def s(v):
    if v is None: return ""
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return str(v)

# canonical comparison fields: (label, tier, section, key)
FIELDS = [
    ("现用药 current_meds（抗癌 vs 非癌/已停）", "优", "Current_Medications", "current_meds"),
    ("分期 Stage", "优", "Cancer_Diagnosis", "Stage_of_Cancer"),
    ("远处转移 Distant Metastasis", "优", "Cancer_Diagnosis", "Distant Metastasis"),
    ("转移(含区域) Metastasis", "优", "Cancer_Diagnosis", "Metastasis"),
    ("疗效 Response", "优", "Response_Assessment", "response_assessment"),
    ("分型/受体 Type (ER/PR/HER2)", "优", "Cancer_Diagnosis", "Type_of_Cancer"),
    ("分子/遗传结果 Genetic results", "优", "Genetic_Testing_Results", "genetic_testing_results"),
    ("基因检测计划 Genetic plan", "优", "Genetic_Testing_Plan", "genetic_testing_plan"),
    ("支持用药 supportive_meds", "优", "Treatment_Changes", "supportive_meds"),
    ("操作计划 Procedure plan", "优", "Procedure_Plan", "procedure_plan"),
    ("影像计划 Imaging plan", "优", "Imaging_Plan", "imaging_plan"),
    ("化验计划 Lab plan", "优", "Lab_Plan", "lab_plan"),
    ("用药计划 Medication plan", "优", "Medication_Plan", "medication_plan"),
    ("近期治疗变化 recent_changes", "评", "Treatment_Changes", "recent_changes"),
    ("就诊类型 Patient type", "评", "Reason_for_Visit", "Patient type"),
    ("治疗目标 goals（方向）", "差", "Treatment_Goals", "goals_of_treatment"),
    ("就诊原因 summary", "差", "Reason_for_Visit", "summary"),
    ("化验摘要 lab_summary", "差", "Lab_Results", "lab_summary"),
    ("检查发现 findings", "差", "Clinical_Findings", "findings"),
]

# full extraction render order (middle two columns)
SECTION_ORDER = ["Reason_for_Visit","Cancer_Diagnosis","Lab_Results","Clinical_Findings",
    "Current_Medications","Treatment_Changes","Treatment_Goals","Response_Assessment",
    "Genetic_Testing_Results","Genetic_Testing_Plan","Medication_Plan","Therapy_plan",
    "radiotherapy_plan","Procedure_Plan","Imaging_Plan","Lab_Plan","Referral",
    "follow_up_next_visit","Advance_care_planning"]

def render_kp(kp):
    rows = []
    for sec in SECTION_ORDER:
        secval = None
        for k in kp:
            if norm(k) == norm(sec): secval = kp[k]; sec = k; break
        if secval is None: continue
        rows.append(f'<div class="sec"><div class="secname">{html.escape(sec)}</div>')
        if isinstance(secval, dict):
            for k, v in secval.items():
                vv = s(v).strip()
                if not vv: continue
                rows.append(f'<div class="fld"><span class="fk">{html.escape(k)}:</span> <span class="fv">{html.escape(vv)}</span></div>')
        else:
            vv = s(secval).strip()
            if vv: rows.append(f'<div class="fld"><span class="fv">{html.escape(vv)}</span></div>')
        rows.append('</div>')
    return "\n".join(rows)

def attr_for(att, section, key):
    """find attribution evidence quote for a field, best-effort."""
    if not isinstance(att, dict): return ""
    # attribution may be nested same as keypoints, or flat by field
    def search(d):
        for k, v in (d.items() if isinstance(d, dict) else []):
            if norm(k) == norm(key):
                if isinstance(v, dict):
                    return s(v.get("source") or v.get("quote") or v.get("evidence") or v)
                return s(v)
            if isinstance(v, dict):
                r = search(v)
                if r: return r
        return ""
    return search(att)

TIER_CLASS = {"优":"t-good","评":"t-mid","差":"t-low"}
TIER_LABEL = {"优":"优·需医学知识","评":"评·中","差":"一般·普通人可答"}

def build(cancer, pl, bl, start_id):
    blocks = []
    toc = []
    for row in sorted(pl):
        rid = f"{cancer}{row}"
        p = pl[row]; b = bl.get(row, {})
        coral = p["coral"]
        title = f"{'乳腺 Breast' if cancer=='b' else '胰腺 PDAC'} · ROW {row} · coral_idx={coral}"
        toc.append(f'<a href="#{rid}">{cancer.upper()}{row}</a>')
        # comparison rows
        comp = []
        for label, tier, section, key in FIELDS:
            pv = s(leaf(p["kp"], section, key)).strip()
            bv = s(leaf(b, section, key)).strip()
            if not pv and not bv: continue
            diff = norm(pv) != norm(bv)
            ev = attr_for(p["att"], section, key).strip()
            evhtml = f'<div class="ev">PL依据: {html.escape(ev[:300])}</div>' if ev else ""
            comp.append(f'''<div class="qrow {'diff' if diff else 'same'}">
  <div class="qlabel"><span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span> {html.escape(label)}</div>
  <div class="qcols">
    <div class="qcol pl"><div class="qtag">PL</div><div class="qval">{html.escape(pv) or '<em>（空）</em>'}</div>{evhtml}</div>
    <div class="qcol bl"><div class="qtag">BL</div><div class="qval">{html.escape(bv) or '<em>（空）</em>'}</div></div>
  </div>
</div>''')
        block = f'''<section class="sample" id="{rid}">
  <h2>{html.escape(title)}</h2>
  <div class="source">
    <div class="srchead">原文 · Assessment &amp; Plan</div>
    <pre class="ap">{html.escape(p["ap"]) or "（无 A/P 段）"}</pre>
    <details><summary>展开完整笔记 note_text</summary><pre class="note">{html.escape(p["note"])}</pre></details>
  </div>
  <div class="twocol">
    <div class="col plbox"><div class="colhead">PL（我们的方法 · pipeline）</div>{render_kp(p["kp"])}</div>
    <div class="col blbox"><div class="colhead">BL（起点 · 单 prompt 裸跑同模型）</div>{render_kp(b)}</div>
  </div>
  <div class="compare">
    <div class="cmphead">逐字段对比小题（按深度分级；高亮=PL/BL 不同。请逐题判：PL 更好 / BL 更好 / 打平）</div>
    {''.join(comp)}
  </div>
  <a class="top" href="#top">↑ 返回目录</a>
</section>'''
        blocks.append(block)
    return toc, blocks

def main():
    plb = parse_pl(os.path.join(HERE, "pipeline_breast_FINAL.txt"))
    plp = parse_pl(os.path.join(HERE, "pipeline_pdac_FINAL.txt"))
    blb = parse_bl(os.path.join(HERE, "baseline_extract_breast_json.txt"))
    blp = parse_bl(os.path.join(HERE, "baseline_extract_pdac_json.txt"))
    tb, bb = build("b", plb, blb, 0)
    tp, bp = build("p", plp, blp, 0)

    css = '''
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f4f5f7;line-height:1.5}
header{background:#0b3d63;color:#fff;padding:22px 28px}
header h1{margin:0 0 6px;font-size:22px}
header p{margin:4px 0;font-size:13.5px;opacity:.93}
.legend{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:12px 16px;margin:16px 28px;font-size:13px}
.legend b{color:#0b3d63}
.toc{margin:0 28px 8px;font-size:13px}
.toc a{display:inline-block;margin:2px 4px;padding:3px 8px;background:#fff;border:1px solid #cdd5dd;border-radius:5px;color:#0b3d63;text-decoration:none}
.toc a:hover{background:#e8f0f7}
.sample{background:#fff;margin:18px 28px;border:1px solid #d8dde3;border-radius:10px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.sample h2{margin:0 0 12px;font-size:18px;color:#0b3d63;border-bottom:2px solid #e6ebf0;padding-bottom:8px}
.source{margin-bottom:14px}
.srchead{font-weight:600;font-size:13px;color:#555;margin-bottom:4px}
pre.ap{white-space:pre-wrap;word-break:break-word;background:#fbfaf3;border:1px solid #ece6c9;border-radius:6px;padding:10px 12px;font-size:12.5px;font-family:"SF Mono",Consolas,monospace;max-height:280px;overflow:auto;margin:0}
details{margin-top:6px}
summary{cursor:pointer;font-size:12.5px;color:#0b3d63}
pre.note{white-space:pre-wrap;word-break:break-word;background:#f7f7f7;border:1px solid #e0e0e0;border-radius:6px;padding:10px;font-size:11.5px;font-family:"SF Mono",Consolas,monospace;max-height:420px;overflow:auto}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0 16px}
.col{border:1px solid #dde3ea;border-radius:8px;padding:10px 12px;font-size:12.5px}
.plbox{background:#f1f8f2;border-color:#bfe0c4}
.blbox{background:#f6f6f7;border-color:#d8dde3}
.colhead{font-weight:700;margin-bottom:8px;font-size:13px}
.plbox .colhead{color:#1b7a36}
.blbox .colhead{color:#555}
.sec{margin-bottom:7px}
.secname{font-weight:600;color:#0b3d63;font-size:11.5px;text-transform:uppercase;letter-spacing:.3px;border-bottom:1px dotted #ccd;margin-bottom:2px}
.fld{margin:1px 0 1px 4px}
.fk{color:#666;font-weight:600}
.fv{color:#111}
.compare{margin-top:10px}
.cmphead{font-weight:700;font-size:13.5px;color:#0b3d63;background:#eef3f8;border:1px solid #d3deea;border-radius:6px;padding:8px 12px;margin-bottom:8px}
.qrow{border:1px solid #e2e6ea;border-radius:7px;margin-bottom:7px;padding:8px 10px;background:#fff}
.qrow.diff{background:#fff8ec;border-color:#f0d39a}
.qrow.same{opacity:.78}
.qlabel{font-weight:600;font-size:13px;margin-bottom:6px}
.tier{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:10px;margin-right:6px;vertical-align:middle}
.t-good{background:#1b7a36;color:#fff}
.t-mid{background:#b8860b;color:#fff}
.t-low{background:#9aa3ad;color:#fff}
.qcols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.qcol{border:1px solid #e0e0e0;border-radius:6px;padding:7px 9px;font-size:12.5px}
.qcol.pl{background:#f1f8f2}
.qcol.bl{background:#f7f7f8}
.qtag{font-weight:700;font-size:11px;margin-bottom:3px}
.qcol.pl .qtag{color:#1b7a36}
.qcol.bl .qtag{color:#777}
.qval{white-space:pre-wrap;word-break:break-word}
.ev{margin-top:5px;font-size:11px;color:#577;border-top:1px dotted #cdd;padding-top:3px}
.top{display:inline-block;margin-top:8px;font-size:12px;color:#0b3d63;text-decoration:none}
@media print{.toc,.top{display:none}.sample{break-inside:avoid;box-shadow:none}body{background:#fff}}
'''
    htmldoc = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PL vs BL 提取对比（医生评审）</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>肿瘤笔记结构化提取 · PL vs BL 对比（医生评审用）</h1>
  <p>PL = 我们的方法（多阶段提取 + 5-gate 校验 + 药物/术语词典 + 后处理规则）。BL = 起点（<b>同一个模型</b> Qwen2.5-32B 单 prompt 裸跑）。唯一变量 = 处理流程。</p>
  <p>40 个去标识化样本（20 乳腺 + 20 胰腺，UCSF CORAL，已 ***** 脱敏）。每个样本：① 原文 → ② PL（左）/ BL（右）完整提取 → ③ 逐字段对比小题。</p>
</header>
<div class="legend">
  <b>怎么读：</b>每个对比小题给出同一字段 PL 与 BL 的答案；<span style="background:#fff8ec;border:1px solid #f0d39a;padding:0 4px">高亮</span>= 两者不同（值得判断）。请对每个小题判断 <b>PL 更好 / BL 更好 / 打平</b>。<br>
  <b>深度分级：</b><span class="tier t-good">优·需医学知识</span> 需专业背景才能答对（药物抗癌/支持区分、分期、转移、疗效、受体、分子）；<span class="tier t-mid">评·中</span> 中等；<span class="tier t-low">一般·普通人可答</span>（就诊原因/治疗目标方向/化验数值等，医生兴趣低，仅供参考）。<br>
  <b>评审重点放在"优"级题。</b> PL 字段下的 “PL依据” 为模型给出的原文归因，便于核对忠实性。
</div>
<div class="toc"><b>乳腺：</b>{''.join(tb)}</div>
<div class="toc"><b>胰腺：</b>{''.join(tp)}</div>
{''.join(bb)}
{''.join(bp)}
<div style="margin:30px 28px;color:#888;font-size:12px">生成自 pipeline_*_FINAL.txt（PL）+ baseline_extract_*_json.txt（BL）。完整方法与统计见 SUMMARY_PL_vs_BL.md。</div>
</body></html>'''
    outp = os.path.join(HERE, "PL_vs_BL_review.html")
    open(outp, "w").write(htmldoc)
    print("wrote", outp, f"({len(htmldoc)//1024} KB), samples: breast {len(plb)} + pdac {len(plp)}")

main()
