#!/usr/bin/env python3
"""Build a FULL scoring HTML for clinician review — every sample shows ALL 19 shared
questions (not just the differing ones), PL | BL side by side, with a per-question
scoring control (PL 更好 / 打平 / BL 更好). No pre-filled verdict (avoid biasing the
doctor). PL attribution shown to help check faithfulness. Self-contained: localStorage
persistence + live tally + CSV export."""
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
        row = int(parts[i]); blk = parts[i+2]
        mj = re.search(r'\{.*\}', blk, re.S)
        try: out[row] = json.loads(mj.group(0)) if mj else {}
        except Exception: out[row] = {}
    return out


def norm(k): return re.sub(r'[^a-z0-9]', '', k.lower())


def s(v):
    if v is None: return ""
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return str(v)


def get_val(kp, section, key):
    if not isinstance(kp, dict): return ""
    sec = None
    for k in kp:
        if norm(k) == norm(section): sec = kp[k]; break
    if sec is None: return ""
    if key is None: return s(sec)
    if isinstance(sec, dict):
        for k in sec:
            if norm(k) == norm(key): return s(sec[k])
        return ""
    return s(sec)


def attr_for(att, key):
    if not isinstance(att, dict) or key is None: return ""
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


# 19 shared questions: (field-id, label, tier, section, key, question-text)
QUESTIONS = [
    ("current_meds",   "现用药 current_meds", "优", "Current_Medications", "current_meds", "当前在用的【抗癌药】是哪些？（须排除降压/降糖/止痛等非癌家用药、已停药、计划未用药）"),
    ("stage",          "分期 Stage", "优", "Cancer_Diagnosis", "Stage_of_Cancer", "癌症分期是什么？"),
    ("distant_met",    "远处转移 Distant Metastasis", "优", "Cancer_Diagnosis", "Distant Metastasis", "有无远处转移、转移到哪？（腋窝/锁骨上=区域，不算远处；疑似≠确诊）"),
    ("metastasis",     "转移(含区域) Metastasis", "优", "Cancer_Diagnosis", "Metastasis", "有无淋巴结/区域受累？（区分区域淋巴结 vs 远处转移）"),
    ("response",       "疗效 Response", "优", "Response_Assessment", "response_assessment", "当前治疗的疗效如何？（副作用≠疗效；未开始治疗→无可评估）"),
    ("type_receptor",  "分型/受体 Type (ER/PR/HER2)", "优", "Cancer_Diagnosis", "Type_of_Cancer", "病理类型与受体状态（ER/PR/HER2；双侧需分别写）"),
    ("genetic_results","分子/遗传结果 Genetic results", "优", "Genetic_Testing_Results", "genetic_testing_results", "已出的分子/基因检测结果"),
    ("genetic_plan",   "基因检测计划 Genetic plan", "优", "Genetic_Testing_Plan", "genetic_testing_plan", "计划做哪些基因/分子检测"),
    ("supportive_meds","支持用药 supportive_meds", "优", "Treatment_Changes", "supportive_meds", "支持性用药（止吐/止痛/胰酶/骨改良药等，非抗癌）"),
    ("procedure_plan", "操作计划 Procedure plan", "优", "Procedure_Plan", "procedure_plan", "计划的操作/手术"),
    ("imaging_plan",   "影像计划 Imaging plan", "优", "Imaging_Plan", "imaging_plan", "计划的影像检查"),
    ("lab_plan",       "化验计划 Lab plan", "优", "Lab_Plan", "lab_plan", "计划的化验"),
    ("medication_plan","用药计划 Medication plan", "优", "Medication_Plan", "medication_plan", "计划/续用的药物方案"),
    ("recent_changes", "近期治疗变化 recent_changes", "评", "Treatment_Changes", "recent_changes", "最近治疗方案的变更（起始/停用/换药/暂停）"),
    ("patient_type",   "就诊类型 Patient type", "评", "Reason_for_Visit", "Patient type", "新患者 vs 随访（初诊会诊/第二意见=New patient）"),
    ("goals",          "治疗目标 goals（方向）", "差", "Treatment_Goals", "goals_of_treatment", "治疗意图是 curative 还是 palliative"),
    ("summary",        "就诊原因 summary", "差", "Reason_for_Visit", "summary", "本次就诊的概述"),
    ("lab_summary",    "化验摘要 lab_summary", "差", "Lab_Results", "lab_summary", "化验结果摘要（不含影像/病理/基因）"),
    ("findings",       "检查发现 findings", "差", "Clinical_Findings", "findings", "客观检查发现（不写主观症状）"),
]
TIER_CLASS = {"优": "t-good", "评": "t-mid", "差": "t-low"}
TIER_LABEL = {"优": "优·需医学知识", "评": "评·中", "差": "一般"}


def build(cancer, pl, bl):
    blocks, toc = [], []
    for row in sorted(pl):
        rid = f"{cancer}{row}"
        p = pl[row]; b = bl.get(row, {})
        coral = p["coral"]
        cname = "乳腺 Breast" if cancer == "b" else "胰腺 PDAC"
        title = f"{cname} · ROW {row} · coral_idx={coral}"
        toc.append(f'<a href="#{rid}">{cancer.upper()}{row}</a>')
        qhtml = []
        for qi, (fid, label, tier, section, key, qtext) in enumerate(QUESTIONS, 1):
            pv = get_val(p["kp"], section, key).strip()
            bv = get_val(b, section, key).strip()
            both_empty = (not pv) and (not bv)
            ev = attr_for(p["att"], key).strip()
            evhtml = (f'<div class="ev"><b>PL 原文归因:</b> {html.escape(ev[:320])}</div>'
                      if ev else '')
            name = f"{rid}__{fid}"
            na_note = ' <span class="na">（PL/BL 都为空，可记 N/A）</span>' if both_empty else ''
            qhtml.append(f'''<div class="q" data-q="{name}">
  <div class="qhead">
    <span class="qnum">Q{qi}</span>
    <span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span>
    <span class="qlabel">{html.escape(label)}</span>{na_note}
  </div>
  <div class="qtext">{html.escape(qtext)}</div>
  <div class="qcols">
    <div class="qcol pl"><div class="qtag">PL（我们的方法）</div><div class="qval">{html.escape(pv) or "<em>（空）</em>"}</div>{evhtml}</div>
    <div class="qcol bl"><div class="qtag">BL（裸跑同模型）</div><div class="qval">{html.escape(bv) or "<em>（空）</em>"}</div></div>
  </div>
  <div class="score" role="radiogroup">
    <label class="opt o-pl"><input type="radio" name="{name}" value="PL"> PL 更好</label>
    <label class="opt o-tie"><input type="radio" name="{name}" value="TIE"> 打平</label>
    <label class="opt o-bl"><input type="radio" name="{name}" value="BL"> BL 更好</label>
    <label class="opt o-na"><input type="radio" name="{name}" value="NA"> N/A</label>
    <input class="cmt" type="text" name="{name}__c" placeholder="备注（可选）">
  </div>
</div>''')
        block = f'''<section class="sample" id="{rid}">
  <h2>{html.escape(title)} <span class="rowprog" id="prog_{rid}">0/19</span></h2>
  <div class="source">
    <div class="srchead">原文 · Assessment &amp; Plan（评分依据，点开看完整笔记）</div>
    <pre class="ap">{html.escape(p["ap"]) or "（无 A/P 段）"}</pre>
    <details><summary>展开完整笔记 note_text</summary><pre class="note">{html.escape(p["note"])}</pre></details>
  </div>
  <div class="qs">{''.join(qhtml)}</div>
  <a class="top" href="#top">↑ 返回目录</a>
</section>'''
        blocks.append(block)
    return toc, blocks


def main():
    plb = parse_pl(os.path.join(HERE, "pipeline_breast_FINAL.txt"))
    plp = parse_pl(os.path.join(HERE, "pipeline_pdac_FINAL.txt"))
    blb = parse_bl(os.path.join(HERE, "baseline_extract_breast_json.txt"))
    blp = parse_bl(os.path.join(HERE, "baseline_extract_pdac_json.txt"))
    tb, bb = build("b", plb, blb)
    tp, bp = build("p", plp, blp)
    total_q = (len(plb) + len(plp)) * len(QUESTIONS)

    css = '''
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f4f5f7;line-height:1.55}
header{background:#0b3d63;color:#fff;padding:20px 28px}
header h1{margin:0 0 6px;font-size:21px}
header p{margin:4px 0;font-size:13px;opacity:.93}
.bar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid #ccd;padding:10px 28px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.bar .stat{font-size:13px;font-weight:700}
.bar .stat span{padding:2px 9px;border-radius:10px;margin-left:4px}
.s-pl{background:#e3f3e7;color:#1b7a36}.s-tie{background:#eee;color:#777}.s-bl{background:#fbe3df;color:#c0392b}.s-na{background:#eef;color:#558}.s-done{background:#0b3d63;color:#fff}
.bar button{font-size:13px;padding:6px 14px;border:1px solid #0b3d63;background:#0b3d63;color:#fff;border-radius:6px;cursor:pointer}
.bar button.ghost{background:#fff;color:#0b3d63}
.legend{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:12px 16px;margin:14px 28px;font-size:13px}
.legend b{color:#0b3d63}
.toc{margin:8px 28px;font-size:12.5px}
.toc a{display:inline-block;margin:2px 4px;padding:3px 8px;background:#fff;border:1px solid #cdd5dd;border-radius:5px;color:#0b3d63;text-decoration:none}
.toc a.done{background:#e3f3e7;border-color:#9ed0ab}
.toc a:hover{background:#e8f0f7}
.sample{background:#fff;margin:18px 28px;border:1px solid #d8dde3;border-radius:10px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.sample h2{margin:0 0 12px;font-size:18px;color:#0b3d63;border-bottom:2px solid #e6ebf0;padding-bottom:8px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.rowprog{font-size:12px;font-weight:700;background:#eee;color:#777;padding:2px 10px;border-radius:10px}
.rowprog.full{background:#1b7a36;color:#fff}
.source{margin-bottom:14px}
.srchead{font-weight:600;font-size:13px;color:#555;margin-bottom:4px}
pre.ap{white-space:pre-wrap;word-break:break-word;background:#fbfaf3;border:1px solid #ece6c9;border-radius:6px;padding:10px 12px;font-size:12.5px;font-family:"SF Mono",Consolas,monospace;max-height:260px;overflow:auto;margin:0}
details{margin-top:6px}summary{cursor:pointer;font-size:12.5px;color:#0b3d63}
pre.note{white-space:pre-wrap;word-break:break-word;background:#f7f7f7;border:1px solid #e0e0e0;border-radius:6px;padding:10px;font-size:11.5px;font-family:"SF Mono",Consolas,monospace;max-height:420px;overflow:auto}
.q{border:1px solid #e2e6ea;border-radius:8px;margin-bottom:9px;padding:10px 12px;background:#fff;border-left:5px solid #ccd}
.q.answered{border-left-color:#0b3d63;background:#fcfdff}
.qhead{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:3px}
.qnum{font-weight:800;color:#0b3d63;font-size:13px}
.tier{display:inline-block;font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:10px;color:#fff}
.t-good{background:#1b7a36}.t-mid{background:#b8860b}.t-low{background:#9aa3ad}
.qlabel{font-weight:700;font-size:13.5px}
.na{font-size:11px;color:#88a}
.qtext{font-size:12.5px;color:#555;margin-bottom:7px}
.qcols{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px}
.qcol{border:1px solid #e0e0e0;border-radius:6px;padding:7px 9px;font-size:12.5px}
.qcol.pl{background:#f1f8f2}.qcol.bl{background:#f7f7f8}
.qtag{font-weight:700;font-size:11px;margin-bottom:3px}
.qcol.pl .qtag{color:#1b7a36}.qcol.bl .qtag{color:#777}
.qval{white-space:pre-wrap;word-break:break-word}
.ev{margin-top:5px;font-size:11px;color:#4a6a6a;border-top:1px dotted #cdd;padding-top:4px}
.score{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.opt{font-size:12.5px;font-weight:600;padding:5px 11px;border:1.5px solid #ccd;border-radius:18px;cursor:pointer;user-select:none}
.opt input{margin-right:4px;vertical-align:middle}
.o-pl.sel{background:#1b7a36;color:#fff;border-color:#1b7a36}
.o-tie.sel{background:#888;color:#fff;border-color:#888}
.o-bl.sel{background:#c0392b;color:#fff;border-color:#c0392b}
.o-na.sel{background:#557;color:#fff;border-color:#557}
.cmt{flex:1;min-width:140px;font-size:12px;padding:5px 8px;border:1px solid #d3d8de;border-radius:6px}
.top{display:inline-block;margin-top:8px;font-size:12px;color:#0b3d63;text-decoration:none}
@media print{.bar,.toc,.top,.score{display:none}.sample{break-inside:avoid;box-shadow:none}body{background:#fff}}
'''
    js = '''
const KEY="pl_bl_scoring_v1", TOTAL=%d;
function load(){try{return JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){return{}}}
function save(d){localStorage.setItem(KEY,JSON.stringify(d))}
let data=load();
function applySaved(){
  for(const [k,v] of Object.entries(data)){
    if(k.endsWith("__c")){const el=document.querySelector(`input[name="${CSS.escape(k)}"]`);if(el)el.value=v;continue;}
    const el=document.querySelector(`input[name="${CSS.escape(k)}"][value="${v}"]`);
    if(el){el.checked=true;markSel(el);}
  }
  recount();
}
function markSel(el){
  const grp=el.closest(".score");
  grp.querySelectorAll(".opt").forEach(o=>o.classList.remove("sel"));
  if(el.checked)el.closest(".opt").classList.add("sel");
  el.closest(".q").classList.add("answered");
}
function recount(){
  let c={PL:0,TIE:0,BL:0,NA:0};
  const rowDone={};
  for(const [k,v] of Object.entries(data)){
    if(k.endsWith("__c"))continue;
    if(c[v]!==undefined)c[v]++;
    const rid=k.split("__")[0];
    rowDone[rid]=(rowDone[rid]||0)+1;
  }
  const done=c.PL+c.TIE+c.BL+c.NA;
  document.getElementById("c_pl").textContent=c.PL;
  document.getElementById("c_tie").textContent=c.TIE;
  document.getElementById("c_bl").textContent=c.BL;
  document.getElementById("c_na").textContent=c.NA;
  document.getElementById("c_done").textContent=done+"/"+TOTAL;
  document.querySelectorAll(".rowprog").forEach(p=>{
    const rid=p.id.replace("prog_","");const n=rowDone[rid]||0;
    p.textContent=n+"/19";p.classList.toggle("full",n>=19);
    const a=document.querySelector(`.toc a[href="#${rid}"]`);if(a)a.classList.toggle("done",n>=19);
  });
}
document.addEventListener("change",e=>{
  if(e.target.matches('.score input[type=radio]')){
    data[e.target.name]=e.target.value;markSel(e.target);save(data);recount();
  }else if(e.target.matches('.cmt')){
    if(e.target.value)data[e.target.name]=e.target.value;else delete data[e.target.name];
    save(data);
  }
});
function exportCSV(){
  let rows=[["sample","field","score","comment"]];
  const seen=new Set();
  for(const k of Object.keys(data)){
    const base=k.replace(/__c$/,"");if(seen.has(base))continue;seen.add(base);
    const [rid,fid]=base.split("__");
    rows.push([rid,fid,data[base]||"",data[base+"__c"]||""]);
  }
  const csv=rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(",")).join("\\n");
  const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="doctor_scores.csv";a.click();
}
function exportJSON(){
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="doctor_scores.json";a.click();
}
function resetAll(){if(confirm("清空所有评分？")){data={};save(data);location.reload();}}
window.addEventListener("DOMContentLoaded",applySaved);
''' % total_q

    htmldoc = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PL vs BL 医生打分（完整版）</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>肿瘤笔记结构化提取 · PL vs BL 医生打分（完整版）</h1>
  <p>PL = 我们的方法（多阶段提取 + 5-gate + 词典 + 后处理）。BL = 同一个模型 Qwen2.5-32B 单 prompt 裸跑。唯一变量 = 处理流程。</p>
  <p>40 个去标识化样本（20 乳腺 + 20 胰腺，UCSF CORAL，已 ***** 脱敏）。每个样本对<b>全部 19 道题</b>逐题打分：PL 更好 / 打平 / BL 更好 / N/A。评分自动存浏览器本地，可随时导出。</p>
</header>
<div class="bar">
  <span class="stat">进度<span class="s-done" id="c_done">0/{total_q}</span></span>
  <span class="stat">PL<span class="s-pl" id="c_pl">0</span></span>
  <span class="stat">打平<span class="s-tie" id="c_tie">0</span></span>
  <span class="stat">BL<span class="s-bl" id="c_bl">0</span></span>
  <span class="stat">N/A<span class="s-na" id="c_na">0</span></span>
  <button onclick="exportCSV()">导出 CSV</button>
  <button class="ghost" onclick="exportJSON()">导出 JSON</button>
  <button class="ghost" onclick="resetAll()">清空</button>
</div>
<div class="legend">
  <b>怎么打分：</b>每题给出同一字段 PL 与 BL 的提取结果，请对照上方原文（A/P 段，点开可看完整笔记），判断哪个更忠实/更完整/更正确，选 <b>PL 更好 / 打平 / BL 更好</b>；两者都为空且本笔记确无此项时选 N/A。<br>
  <b>深度分级：</b><span class="tier t-good">优·需医学知识</span> 需专业背景（药物抗癌/支持区分、分期、转移、疗效、受体、分子）；<span class="tier t-mid">评·中</span>；<span class="tier t-low">一般</span>（就诊原因/治疗目标方向/化验数值，普通人可答）。<b>请把精力放在“优”级题。</b><br>
  <b>PL 原文归因</b> = 模型给出的原文出处，便于核对 PL 是否忠实。<b>本页未预填任何判定，请独立评分。</b>
</div>
<div class="toc"><b>乳腺：</b>{''.join(tb)}</div>
<div class="toc"><b>胰腺：</b>{''.join(tp)}</div>
{''.join(bb)}
{''.join(bp)}
<div style="margin:30px 28px;color:#888;font-size:12px">PL 值来自 pipeline_*_FINAL.txt，BL 值来自 baseline_extract_*_json.txt。题目定义见 QUESTIONS.txt。</div>
<script>{js}</script>
</body></html>'''
    outp = os.path.join(HERE, "PL_vs_BL_scoring.html")
    open(outp, "w").write(htmldoc)
    print("wrote", outp, f"({len(htmldoc)//1024} KB), samples: breast {len(plb)} + pdac {len(plp)}, total questions:", total_q)


main()
