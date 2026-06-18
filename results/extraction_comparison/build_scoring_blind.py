#!/usr/bin/env python3
"""BLIND scoring HTML: same samples/questions as the main scoring page, but the two
extractions are shown only as System A (left) and System B (right) — the rater cannot tell
which is which. A = PL (our method), B = BL (baseline), fixed left/right (not randomized),
but this mapping is NOT shown anywhere in the page.

Identity labels removed vs the normal page (the *content* is shown verbatim):
- column labels PL/BL -> A/B; option labels -> "A better / B better"
- header / legend / hints carry no "PL/BL/our method/baseline/pipeline" wording
- A and B columns use identical neutral styling (no green=good hint)
- output filename and localStorage key carry no PL/BL
Attribution IS kept (it is part of the model output being judged); its label is made
neutral. Note: only A (PL) produces attribution, so its presence is itself a real output
difference the rater sees.
Scores are stored as A/B; map back with A=PL, B=BL when analyzing.
"""
import html, os
from build_scoring_html import (parse_pl, parse_bl, get_val, attr_for, QUESTIONS, PER_CANCER,
                                qset_for, TIER_CLASS, TIER_LABEL)

HERE = os.path.dirname(os.path.abspath(__file__))

# Neutral reminders (no PL/BL wording).
RULE_FULL = ('We are grading <b>extraction</b>, not summary. A system that lists the raw '
             'detail without forcing a tidy conclusion is doing the job, and you judge it; a '
             'vague summary that happens to read as correct still skipped the extraction, so '
             'that alone is <b>not</b> a reason to prefer it.')
RULE_SHORT = 'We are grading <b>extraction</b> here, not summary.'


def build(cancer, pl, bl):
    blocks, toc = [], []
    qset = qset_for(cancer)
    nq = len(qset)
    for row in sorted(pl):
        rid = f"{cancer}{row}"
        p = pl[row]; b = bl.get(row, {})
        coral = p["coral"]
        cname = "Breast" if cancer == "b" else "PDAC"
        title = f"{cname} · ROW {row} · coral_idx={coral}"
        toc.append(f'<a href="#{rid}">{cancer.upper()}{row}</a>')
        qhtml = []
        for qi, (fid, label, tier, section, key, qtext) in enumerate(qset, 1):
            if fid in PER_CANCER and cancer in PER_CANCER[fid]:
                label, qtext = PER_CANCER[fid][cancer]
            av = get_val(p["kp"], section, key).strip()   # A = PL
            bv = get_val(b, section, key).strip()          # B = BL
            both_empty = (not av) and (not bv)
            ev = attr_for(p["att"], key).strip()           # model-produced attribution (A only)
            evhtml = (f'<div class="ev"><b>Cited source</b> <span class="evsrc">(quote the model pulled from the note)</span>: {html.escape(ev[:320])}</div>'
                      if ev else '')
            name = f"{rid}__{fid}"
            na_note = ' <span class="na">(both empty, you can mark N/A)</span>' if both_empty else ''
            full_rule = (qi - 1) % 5 == 0
            hint = RULE_FULL if full_rule else RULE_SHORT
            qtext_html = f'<div class="qtext">{html.escape(qtext)} <span class="qhint">{hint}</span></div>'
            qhtml.append(f'''<div class="q" data-q="{name}">
  <div class="qhead">
    <span class="qnum">Q{qi}</span>
    <span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span>
    <span class="qlabel">{html.escape(label)}</span>{na_note}
  </div>
  {qtext_html}
  <div class="qcols">
    <div class="qcol a"><div class="qtag">A</div><div class="qval">{html.escape(av) or "<em>(empty)</em>"}</div>{evhtml}</div>
    <div class="qcol b"><div class="qtag">B</div><div class="qval">{html.escape(bv) or "<em>(empty)</em>"}</div></div>
  </div>
  <div class="score" role="radiogroup">
    <label class="opt o-a"><input type="radio" name="{name}" value="A"> A better</label>
    <label class="opt o-tie"><input type="radio" name="{name}" value="TIE"> Tie</label>
    <label class="opt o-b"><input type="radio" name="{name}" value="B"> B better</label>
    <label class="opt o-na"><input type="radio" name="{name}" value="NA"> N/A</label>
    <input class="cmt" type="text" name="{name}__c" placeholder="Note (optional)">
  </div>
</div>''')
        block = f'''<section class="sample" id="{rid}">
  <h2>{html.escape(title)} <span class="rowprog" id="prog_{rid}" data-nq="{nq}">0/{nq}</span></h2>
  <div class="source">
    <div class="srchead">Source · Assessment &amp; Plan (basis for scoring; expand for the full note)</div>
    <pre class="ap">{html.escape(p["ap"]) or "(no A/P section)"}</pre>
    <details><summary>Expand full note_text</summary><pre class="note">{html.escape(p["note"])}</pre></details>
  </div>
  <div class="qs">{''.join(qhtml)}</div>
  <a class="top" href="#top">↑ Back to top</a>
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
    NQ_B = len(qset_for("b")); NQ_P = len(qset_for("p"))
    total_q = len(plb) * NQ_B + len(plp) * NQ_P

    css = '''
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f4f5f7;line-height:1.55}
header{background:#0b3d63;color:#fff;padding:20px 28px}
header h1{margin:0 0 6px;font-size:21px}
header p{margin:4px 0;font-size:13px;opacity:.93}
.bar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid #ccd;padding:10px 28px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.bar .stat{font-size:13px;font-weight:700}
.bar .stat span{padding:2px 9px;border-radius:10px;margin-left:4px}
.s-a{background:#e6eef7;color:#3b6ea5}.s-tie{background:#eee;color:#777}.s-b{background:#ece9f7;color:#6a5acd}.s-na{background:#eef;color:#558}.s-done{background:#0b3d63;color:#fff}
.bar button{font-size:13px;padding:6px 14px;border:1px solid #0b3d63;background:#0b3d63;color:#fff;border-radius:6px;cursor:pointer}
.bar button.ghost{background:#fff;color:#0b3d63}
.barrule{font-size:12px;color:#7a5b00;background:#fff7e0;border:1px solid #f0d79a;border-radius:6px;padding:4px 10px}
.barrule b{color:#6b4f00}
.legend{background:#fff;border:1px solid #d8dde3;border-radius:8px;padding:12px 16px;margin:14px 28px;font-size:13px}
.legend>div{margin:6px 0}
.legend b{color:#0b3d63}
.principle{background:#fff7e0;border:1px solid #f0d79a;border-left:4px solid #d9a400;border-radius:6px;padding:9px 13px}
.principle b{color:#6b4f00}
.principle ul{margin:6px 0 0;padding-left:20px}
.principle li{margin:3px 0}
.tiers{background:#f8fafc;border:1px solid #e3e8ee;border-radius:6px;padding:8px 12px}
.tiers>div{margin:4px 0}
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
.qcol{border:1px solid #e0e0e0;border-radius:6px;padding:7px 9px;font-size:12.5px;background:#fafbfc}
.qtag{font-weight:700;font-size:11px;margin-bottom:3px;color:#555}
.qcol.a .qtag{color:#3b6ea5}.qcol.b .qtag{color:#6a5acd}
.qval{white-space:pre-wrap;word-break:break-word}
.ev{margin-top:5px;font-size:11px;color:#4a6a6a;border-top:1px dotted #cdd;padding-top:4px}
.evsrc{color:#8a8f95;font-style:italic}
.qhint{color:#9a7b2a;font-style:italic}
.qhint b{color:#7a5b00;font-style:normal}
.score{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.opt{font-size:12.5px;font-weight:600;padding:5px 11px;border:1.5px solid #ccd;border-radius:18px;cursor:pointer;user-select:none}
.opt input{margin-right:4px;vertical-align:middle}
.o-a.sel{background:#3b6ea5;color:#fff;border-color:#3b6ea5}
.o-tie.sel{background:#888;color:#fff;border-color:#888}
.o-b.sel{background:#6a5acd;color:#fff;border-color:#6a5acd}
.o-na.sel{background:#557;color:#fff;border-color:#557}
.cmt{flex:1;min-width:140px;font-size:12px;padding:5px 8px;border:1px solid #d3d8de;border-radius:6px}
.top{display:inline-block;margin-top:8px;font-size:12px;color:#0b3d63;text-decoration:none}
@media print{.bar,.toc,.top,.score{display:none}.sample{break-inside:avoid;box-shadow:none}body{background:#fff}}
'''
    js = '''
const KEY="blind_scoring_v1", TOTAL=%d;
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
  let c={A:0,TIE:0,B:0,NA:0};
  const rowDone={};
  for(const [k,v] of Object.entries(data)){
    if(k.endsWith("__c"))continue;
    if(c[v]!==undefined)c[v]++;
    const rid=k.split("__")[0];
    rowDone[rid]=(rowDone[rid]||0)+1;
  }
  const done=c.A+c.TIE+c.B+c.NA;
  document.getElementById("c_a").textContent=c.A;
  document.getElementById("c_tie").textContent=c.TIE;
  document.getElementById("c_b").textContent=c.B;
  document.getElementById("c_na").textContent=c.NA;
  document.getElementById("c_done").textContent=done+"/"+TOTAL;
  document.querySelectorAll(".rowprog").forEach(p=>{
    const rid=p.id.replace("prog_","");const n=rowDone[rid]||0;
    const nq=+p.dataset.nq||0;
    p.textContent=n+"/"+nq;p.classList.toggle("full",nq>0&&n>=nq);
    const a=document.querySelector(`.toc a[href="#${rid}"]`);if(a)a.classList.toggle("done",nq>0&&n>=nq);
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
  a.download="blind_scores.csv";a.click();
}
function exportJSON(){
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="blind_scores.json";a.click();
}
function resetAll(){if(confirm("Clear all scores?")){data={};save(data);location.reload();}}
window.addEventListener("DOMContentLoaded",applySaved);
''' % total_q

    htmldoc = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extraction Scoring (blind A/B)</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>Oncology Note Structured Extraction · Blind Scoring (A vs B)</h1>
  <p>Two automated systems, <b>A</b> and <b>B</b>, each extracted structured fields from the same clinical note. They are shown side by side without revealing which system is which. Please judge, for each field, which extraction is better.</p>
  <p>There are 40 de-identified samples (20 breast and 20 PDAC, from UCSF CORAL, with ***** redacted). Score every question on each sample (<b>16 for breast, 15 for PDAC</b>; the type/receptor question is breast-only, since PDAC has no ER/PR/HER2 markers) as A better, Tie, B better, or N/A. Your scores save in the browser as you go, and you can export them anytime.</p>
</header>
<div class="bar">
  <span class="stat">Progress<span class="s-done" id="c_done">0/{total_q}</span></span>
  <span class="stat">A<span class="s-a" id="c_a">0</span></span>
  <span class="stat">Tie<span class="s-tie" id="c_tie">0</span></span>
  <span class="stat">B<span class="s-b" id="c_b">0</span></span>
  <span class="stat">N/A<span class="s-na" id="c_na">0</span></span>
  <button onclick="exportCSV()">Export CSV</button>
  <button class="ghost" onclick="exportJSON()">Export JSON</button>
  <button class="ghost" onclick="resetAll()">Reset</button>
  <span class="barrule">⚑ We are grading <b>extraction</b>, not summary. A vague summary that happens to read as correct still skipped the job, so it does not win.</span>
</div>
<div class="legend">
  <div class="principle"><b>★ Scoring principle: grade extraction, not summary or inference.</b> The job is to <b>pull the concrete facts out of the note</b>. So:
    <ul>
      <li>When a system lists the raw details (drugs, sizes, nodes, dates) without forcing a tidy conclusion, that is a <b>good</b> outcome. It means the details got pulled out, and you, the clinician, can judge them.</li>
      <li>When a system gives a short, vague summary that drops the details but happens to read as correct, that is <b>off task</b>. It may sound right, but it skipped the extraction job, so do not prefer it for that.</li>
      <li>Prefer the other side only when it actually extracted something <b>true and specific that the first missed or got wrong</b>, not just because it reads shorter or smoother.</li>
    </ul>
  </div>
  <div><b>How to score:</b> For each question you see the same field as extracted by System A and by System B. Using the source note above (the A/P section, or expand for the full note), decide which is more faithful, more complete, and more correct, then choose <b>A better, Tie, or B better</b>. If both are empty and the note really has no such item, choose N/A.</div>
  <div class="tiers">
    <div><b>Depth tiers:</b></div>
    <div><span class="tier t-good">Deep · needs medical knowledge</span> needs clinical background: telling anticancer drugs from supportive ones, stage, metastasis, response, receptors, molecular results.</div>
    <div><span class="tier t-mid">Medium</span> moderate.</div>
    <div><span class="tier t-low">Basic · layperson</span> things like lab values, answerable without medical training.</div>
    <div><b>Please put your effort into the Deep questions.</b></div>
  </div>
  <div><b>Nothing is pre-filled, and neither side is labeled — please score on your own.</b></div>
</div>
<div class="toc"><b>Breast:</b> {''.join(tb)}</div>
<div class="toc"><b>PDAC:</b> {''.join(tp)}</div>
{''.join(bb)}
{''.join(bp)}
<div style="margin:30px 28px;color:#888;font-size:12px">Two systems compared on the same notes. Question definitions in QUESTIONS.txt.</div>
<script>{js}</script>
</body></html>'''
    outp = os.path.join(HERE, "scoring_blind_AB.html")
    open(outp, "w").write(htmldoc)
    print("wrote", outp, f"({len(htmldoc)//1024} KB), samples: breast {len(plb)} + pdac {len(plp)}, total questions:", total_q)


main()
