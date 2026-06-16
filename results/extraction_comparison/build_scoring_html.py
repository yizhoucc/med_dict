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
    ("current_meds",   "Current medications", "deep", "Current_Medications", "current_meds", "Which drugs is the patient currently taking that are ANTI-CANCER? (Exclude non-cancer home meds — BP/diabetes/pain — discontinued drugs, and planned-but-not-started drugs.)"),
    ("stage",          "Cancer stage", "deep", "Cancer_Diagnosis", "Stage_of_Cancer", "What is the cancer stage?"),
    ("distant_met",    "Distant metastasis", "deep", "Cancer_Diagnosis", "Distant Metastasis", "Is there distant metastasis, and where? (Axillary / supraclavicular = regional, NOT distant; suspected ≠ confirmed.)"),
    ("metastasis",     "Metastasis (incl. regional)", "deep", "Cancer_Diagnosis", "Metastasis", "Is there nodal / regional involvement? (Distinguish regional lymph nodes vs distant metastasis.)"),
    ("response",       "Treatment response", "deep", "Response_Assessment", "response_assessment", "How is the cancer responding to current treatment? (Side effects ≠ response; not yet on treatment → nothing to assess.)"),
    ("type_receptor",  "Type / receptors (ER/PR/HER2)", "deep", "Cancer_Diagnosis", "Type_of_Cancer", "Pathology type and receptor status (ER/PR/HER2; bilateral disease must list each side)."),
    ("genetic_results","Molecular / genetic results", "deep", "Genetic_Testing_Results", "genetic_testing_results", "Molecular / genetic test results already available."),
    ("genetic_plan",   "Genetic testing plan", "deep", "Genetic_Testing_Plan", "genetic_testing_plan", "Which genetic / molecular tests are planned."),
    ("supportive_meds","Supportive medications", "deep", "Treatment_Changes", "supportive_meds", "Supportive meds (antiemetics / analgesics / pancreatic enzymes / bone agents, etc.; non-anticancer)."),
    ("procedure_plan", "Procedure plan", "deep", "Procedure_Plan", "procedure_plan", "Planned procedures / surgery."),
    ("imaging_plan",   "Imaging plan", "deep", "Imaging_Plan", "imaging_plan", "Planned imaging."),
    ("lab_plan",       "Lab plan", "deep", "Lab_Plan", "lab_plan", "Planned labs."),
    ("medication_plan","Medication plan", "deep", "Medication_Plan", "medication_plan", "Planned / continued drug regimen."),
    ("recent_changes", "Recent treatment changes", "med", "Treatment_Changes", "recent_changes", "Recent changes to the treatment regimen (start / stop / switch / hold)."),
    # NOTE: 'patient_type' and 'goals (treatment intent)' were dropped 2026-06-15 — a
    # clinician judged both clinically useless to score. Their historical PL-vs-BL
    # verdicts are kept in _audit_v5/verdicts.json; they are removed from scoring/HTML/figs.
    ("summary",        "Reason for visit (summary)", "basic", "Reason_for_Visit", "summary", "Summary of the reason for this visit."),
    ("lab_summary",    "Lab summary", "basic", "Lab_Results", "lab_summary", "Summary of lab results (excluding imaging / pathology / genetics)."),
    ("findings",       "Clinical findings", "basic", "Clinical_Findings", "findings", "Objective exam / imaging findings (not subjective symptoms)."),
]
TIER_CLASS = {"deep": "t-good", "med": "t-mid", "basic": "t-low"}
TIER_LABEL = {"deep": "Deep · needs medical knowledge", "med": "Medium", "basic": "Basic · layperson"}
NQ = len(QUESTIONS)


def build(cancer, pl, bl):
    blocks, toc = [], []
    for row in sorted(pl):
        rid = f"{cancer}{row}"
        p = pl[row]; b = bl.get(row, {})
        coral = p["coral"]
        cname = "Breast" if cancer == "b" else "PDAC"
        title = f"{cname} · ROW {row} · coral_idx={coral}"
        toc.append(f'<a href="#{rid}">{cancer.upper()}{row}</a>')
        qhtml = []
        for qi, (fid, label, tier, section, key, qtext) in enumerate(QUESTIONS, 1):
            pv = get_val(p["kp"], section, key).strip()
            bv = get_val(b, section, key).strip()
            both_empty = (not pv) and (not bv)
            ev = attr_for(p["att"], key).strip()
            evhtml = (f'<div class="ev"><b>PL attribution:</b> {html.escape(ev[:320])}</div>'
                      if ev else '')
            name = f"{rid}__{fid}"
            na_note = ' <span class="na">(both empty — may mark N/A)</span>' if both_empty else ''
            qhtml.append(f'''<div class="q" data-q="{name}">
  <div class="qhead">
    <span class="qnum">Q{qi}</span>
    <span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span>
    <span class="qlabel">{html.escape(label)}</span>{na_note}
  </div>
  <div class="qtext">{html.escape(qtext)}</div>
  <div class="qcols">
    <div class="qcol pl"><div class="qtag">PL (our method)</div><div class="qval">{html.escape(pv) or "<em>(empty)</em>"}</div>{evhtml}</div>
    <div class="qcol bl"><div class="qtag">BL (same model, single prompt)</div><div class="qval">{html.escape(bv) or "<em>(empty)</em>"}</div></div>
  </div>
  <div class="score" role="radiogroup">
    <label class="opt o-pl"><input type="radio" name="{name}" value="PL"> PL better</label>
    <label class="opt o-tie"><input type="radio" name="{name}" value="TIE"> Tie</label>
    <label class="opt o-bl"><input type="radio" name="{name}" value="BL"> BL better</label>
    <label class="opt o-na"><input type="radio" name="{name}" value="NA"> N/A</label>
    <input class="cmt" type="text" name="{name}__c" placeholder="Note (optional)">
  </div>
</div>''')
        block = f'''<section class="sample" id="{rid}">
  <h2>{html.escape(title)} <span class="rowprog" id="prog_{rid}">0/{NQ}</span></h2>
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
.legend>div{margin:6px 0}
.legend b{color:#0b3d63}
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
const KEY="pl_bl_scoring_v2", TOTAL=%d, NQ=%d;
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
    p.textContent=n+"/"+NQ;p.classList.toggle("full",n>=NQ);
    const a=document.querySelector(`.toc a[href="#${rid}"]`);if(a)a.classList.toggle("done",n>=NQ);
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
function resetAll(){if(confirm("Clear all scores?")){data={};save(data);location.reload();}}
window.addEventListener("DOMContentLoaded",applySaved);
''' % (total_q, NQ)

    htmldoc = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PL vs BL — Clinician Scoring (Full)</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>Oncology Note Structured Extraction · PL vs BL — Clinician Scoring (Full)</h1>
  <p>PL = our method (multi-stage extraction + 5-gate verification + drug/term dictionaries + post-processing). BL = the same model (Qwen2.5-32B) run with a single prompt and no post-processing. The only variable is the processing pipeline.</p>
  <p>40 de-identified samples (20 breast + 20 PDAC, UCSF CORAL, ***** redacted). For each sample, score <b>all {NQ} questions</b>: PL better / Tie / BL better / N/A. Scores auto-save in your browser and can be exported anytime.</p>
</header>
<div class="bar">
  <span class="stat">Progress<span class="s-done" id="c_done">0/{total_q}</span></span>
  <span class="stat">PL<span class="s-pl" id="c_pl">0</span></span>
  <span class="stat">Tie<span class="s-tie" id="c_tie">0</span></span>
  <span class="stat">BL<span class="s-bl" id="c_bl">0</span></span>
  <span class="stat">N/A<span class="s-na" id="c_na">0</span></span>
  <button onclick="exportCSV()">Export CSV</button>
  <button class="ghost" onclick="exportJSON()">Export JSON</button>
  <button class="ghost" onclick="resetAll()">Reset</button>
</div>
<div class="legend">
  <div><b>How to score:</b> For each question you'll see the same field as extracted by PL and by BL. Using the source note above (A/P section; expand for the full note), decide which is more faithful / complete / correct, and choose <b>PL better / Tie / BL better</b>. If both are empty and the note truly has no such item, choose N/A.</div>
  <div class="tiers">
    <div><b>Depth tiers:</b></div>
    <div><span class="tier t-good">Deep · needs medical knowledge</span> requires clinical background — anticancer-vs-supportive drug distinction, stage, metastasis, response, receptors, molecular.</div>
    <div><span class="tier t-mid">Medium</span> moderate.</div>
    <div><span class="tier t-low">Basic · layperson</span> reason for visit, treatment-goal direction, lab values — answerable without medical training.</div>
    <div><b>Please focus your effort on the Deep questions.</b></div>
  </div>
  <div><b>PL attribution</b> = the source span the model cited, to help you check PL's faithfulness. <b>No verdict is pre-filled — please score independently.</b></div>
</div>
<div class="toc"><b>Breast:</b> {''.join(tb)}</div>
<div class="toc"><b>PDAC:</b> {''.join(tp)}</div>
{''.join(bb)}
{''.join(bp)}
<div style="margin:30px 28px;color:#888;font-size:12px">PL values from pipeline_*_FINAL.txt; BL values from baseline_extract_*_json.txt. Question definitions in QUESTIONS.txt.</div>
<script>{js}</script>
</body></html>'''
    outp = os.path.join(HERE, "PL_vs_BL_scoring.html")
    open(outp, "w").write(htmldoc)
    print("wrote", outp, f"({len(htmldoc)//1024} KB), samples: breast {len(plb)} + pdac {len(plp)}, total questions:", total_q)


main()
