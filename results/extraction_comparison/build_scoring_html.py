#!/usr/bin/env python3
"""Build the FULL clinician scoring HTML — every sample shows all questions, PL | BL side
by side, with a per-question scoring control. No pre-filled verdict (avoid biasing the
doctor). PL attribution shown to help check faithfulness. Self-contained: localStorage
persistence + live tally + CSV export.
The user-facing copy (question texts, scoring rule, legend, header) is humanized with the
blader/humanizer skill so it reads like a clinician wrote it, not an AI."""
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


# Question texts humanized with the blader/humanizer skill: no em dashes, varied phrasing,
# and the repetitive "Extract the specific X ... We want Y, not a vague Z" template replaced
# with plain questions a clinician would actually ask.
QUESTIONS = [
    ("current_meds",   "Current medications", "deep", "Current_Medications", "current_meds", "Which anticancer drugs is the patient on right now? List them by name. Leave out the home meds for things like blood pressure, diabetes, or pain, anything they have stopped, and anything that is only being considered. A vague 'on chemo' is not enough."),
    ("stage",          "Cancer stage", "deep", "Cancer_Diagnosis", "Stage_of_Cancer", "What stage is the cancer? Give the actual stage or TNM, either stated in the note or worked out from tumor size, nodes, and spread. Do not settle for 'advanced', and do not write 'not mentioned' if the note gives you enough to stage it."),
    ("distant_met",    "Distant metastasis", "deep", "Cancer_Diagnosis", "Distant Metastasis", "Is there distant spread, and to where? Name the sites. Axillary and supraclavicular nodes count as regional, not distant, and flag anything that is only suspected rather than confirmed. A bare yes or no does not cut it."),
    ("metastasis",     "Metastasis (incl. regional)", "deep", "Cancer_Diagnosis", "Metastasis", "Is there nodal or regional involvement? Say which nodes and how many. Keep regional nodes separate from distant spread."),
    ("response",       "Treatment response", "deep", "Response_Assessment", "response_assessment", "How is the cancer responding to treatment, and on what basis? Point to the evidence, like a change on imaging or a moving tumor marker. Side effects are not a response, and if treatment has not started, say so. 'Doing well' tells us nothing."),
    ("type_receptor",  "Type / receptors", "deep", "Cancer_Diagnosis", "Type_of_Cancer", "What is the tumor type and the receptor status? Give each value rather than a vague label. (See the per-cancer note.)"),
    ("genetic_results","Molecular / genetic results", "deep", "Genetic_Testing_Results", "genetic_testing_results", "What molecular or genetic results are already back? Give the gene, the variant, and the status, like BRCA2 positive, MMR intact, or CA19-9 non-secretor. List each one. 'Testing was done' does not answer it."),
    ("genetic_plan",   "Genetic testing plan", "deep", "Genetic_Testing_Plan", "genetic_testing_plan", "Which genetic or molecular tests are still coming up, ordered but not yet back (Oncotype, UCSF500, a germline panel)? Only count tests that have not resulted yet; ones already back belong under molecular results. Name them rather than writing 'will test'."),
    ("supportive_meds","Supportive medications", "deep", "Treatment_Changes", "supportive_meds", "What supportive medications is the patient on, by name? Think antiemetics, pain meds, pancreatic enzymes, bone agents, nothing that treats the cancer itself. 'Supportive care given' is not specific enough."),
    ("procedure_plan", "Procedure plan", "deep", "Procedure_Plan", "procedure_plan", "What procedures or surgery are planned and still to come (a port, a biopsy, a resection)? Count only what has not happened yet, not procedures already done. Name them rather than writing 'procedure planned'."),
    ("imaging_plan",   "Imaging plan", "deep", "Imaging_Plan", "imaging_plan", "What imaging is planned and still to come (CT, MRI, PET, DEXA), with timing if the note gives it? Count only scans that have not been done yet. Name them rather than writing 'imaging to follow'."),
    ("lab_plan",       "Lab plan", "deep", "Lab_Plan", "lab_plan", "What labs are planned and still to come, like a CA19-9 follow-up or electrolyte monitoring? Count only labs that have not been drawn yet. Name them rather than writing 'labs to follow'."),
    ("medication_plan","Medication plan", "deep", "Medication_Plan", "medication_plan", "What is the plan for medications going forward, including drugs to start, keep, or stop, and any chemo hold or break? Count only what is upcoming, not drugs already finished. Watch the edge case: if the patient wants a chemo break, that is a change, so 'no medications mentioned' would be wrong."),
    ("recent_changes", "Recent treatment changes", "med", "Treatment_Changes", "recent_changes", "What recently changed in the treatment, and which drug? Something started, stopped, switched, or held. Name the actual change instead of 'treatment ongoing'."),
    # NOTE: 'patient_type' and 'goals (treatment intent)' were dropped 2026-06-15, and
    # 'summary (reason for visit)' was dropped 2026-06-16. Historical verdicts kept in
    # _audit_v5/verdicts.json; removed from scoring/HTML/figs.
    ("lab_summary",    "Lab summary", "basic", "Lab_Results", "lab_summary", "What do the labs show? Give the actual values, but leave out imaging, pathology, and genetics. 'Labs stable' is not an answer."),
    ("findings",       "Clinical findings", "basic", "Clinical_Findings", "findings", "What did the exam and imaging actually find, with sizes and sites? Stick to objective findings, not the patient's symptoms. List them instead of giving a one-line impression."),
]
TIER_CLASS = {"deep": "t-good", "med": "t-mid", "basic": "t-low"}
TIER_LABEL = {"deep": "Deep · needs medical knowledge", "med": "Medium", "basic": "Basic · layperson"}

# Per-cancer overrides of (label, question-text). Breast and PDAC don't share the same
# biology — e.g. PDAC has no ER/PR/HER2 receptors, so Q6 must be phrased per cancer.
# Q6 (type/receptor) is breast-only (see qset_for) — PDAC has no ER/PR/HER2 markers.
PER_CANCER = {
    "type_receptor": {
        "b": ("Type / receptors (ER/PR/HER2)",
              "What is the tumor type, and what are the ER, PR, and HER2 results? Give each receptor value, and list both sides if the disease is bilateral. 'Hormone-positive' on its own is not specific enough."),
    },
}

# The reminder so raters don't get misled (see legend). The full amber box shows every 5th
# question; the rest just get a short hint folded into the end of the question text.
SCORING_RULE = ('⚑ We are grading <b>extraction</b>, not summary. If PL pulls out more raw detail '
                'without forcing a conclusion, that is good, and you are the one who judges it. If BL '
                'gives a vague summary that happens to read as correct, it still skipped the job, so '
                'that is <b>not</b> a reason to mark "BL better".')
SCORING_RULE_SHORT = 'We are grading <b>extraction</b> here, not summary.'


def qset_for(cancer):
    # Q6 (type/receptor) is breast-only: PDAC has no ER/PR/HER2 markers, so we drop it.
    return [q for q in QUESTIONS if not (cancer == "p" and q[0] == "type_receptor")]


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
            pv = get_val(p["kp"], section, key).strip()
            bv = get_val(b, section, key).strip()
            both_empty = (not pv) and (not bv)
            ev = attr_for(p["att"], key).strip()
            evhtml = (f'<div class="ev"><b>PL attribution</b> <span class="evsrc">(quote the model pulled from the note)</span>: {html.escape(ev[:320])}</div>'
                      if ev else '')
            name = f"{rid}__{fid}"
            na_note = ' <span class="na">(both empty, you can mark N/A)</span>' if both_empty else ''
            # Full amber reminder every 5th question (Q1, Q6, Q11, Q16); for the rest, fold a
            # short hint into the end of the question text so there's no separate box.
            full_rule = (qi - 1) % 5 == 0
            if full_rule:
                qtext_html = f'<div class="qtext">{html.escape(qtext)}</div>'
                rule_html = f'<div class="rule">{SCORING_RULE}</div>'
            else:
                qtext_html = f'<div class="qtext">{html.escape(qtext)} <span class="qhint">{SCORING_RULE_SHORT}</span></div>'
                rule_html = ''
            qhtml.append(f'''<div class="q" data-q="{name}">
  <div class="qhead">
    <span class="qnum">Q{qi}</span>
    <span class="tier {TIER_CLASS[tier]}">{TIER_LABEL[tier]}</span>
    <span class="qlabel">{html.escape(label)}</span>{na_note}
  </div>
  {qtext_html}
  <div class="qcols">
    <div class="qcol pl"><div class="qtag">PL (our method)</div><div class="qval">{html.escape(pv) or "<em>(empty)</em>"}</div>{evhtml}</div>
    <div class="qcol bl"><div class="qtag">BL (same model, single prompt)</div><div class="qval">{html.escape(bv) or "<em>(empty)</em>"}</div></div>
  </div>
  {rule_html}
  <div class="score" role="radiogroup">
    <label class="opt o-pl"><input type="radio" name="{name}" value="PL"> PL better</label>
    <label class="opt o-tie"><input type="radio" name="{name}" value="TIE"> Tie</label>
    <label class="opt o-bl"><input type="radio" name="{name}" value="BL"> BL better</label>
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
.s-pl{background:#e3f3e7;color:#1b7a36}.s-tie{background:#eee;color:#777}.s-bl{background:#fbe3df;color:#c0392b}.s-na{background:#eef;color:#558}.s-done{background:#0b3d63;color:#fff}
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
.qcol{border:1px solid #e0e0e0;border-radius:6px;padding:7px 9px;font-size:12.5px}
.qcol.pl{background:#f1f8f2}.qcol.bl{background:#f7f7f8}
.qtag{font-weight:700;font-size:11px;margin-bottom:3px}
.qcol.pl .qtag{color:#1b7a36}.qcol.bl .qtag{color:#777}
.qval{white-space:pre-wrap;word-break:break-word}
.ev{margin-top:5px;font-size:11px;color:#4a6a6a;border-top:1px dotted #cdd;padding-top:4px}
.evsrc{color:#8a8f95;font-style:italic}
.rule{font-size:11.5px;color:#7a5b00;background:#fff7e0;border:1px solid #f0d79a;border-left:4px solid #d9a400;border-radius:5px;padding:5px 9px;margin:0 0 6px}
.rule b{color:#6b4f00}
.qhint{color:#9a7b2a;font-style:italic}
.qhint b{color:#7a5b00;font-style:normal}
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
const KEY="pl_bl_scoring_v5", TOTAL=%d;
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
  a.download="doctor_scores.csv";a.click();
}
function exportJSON(){
  const blob=new Blob([JSON.stringify(data,null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="doctor_scores.json";a.click();
}
function resetAll(){if(confirm("Clear all scores?")){data={};save(data);location.reload();}}
window.addEventListener("DOMContentLoaded",applySaved);
''' % total_q

    htmldoc = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PL vs BL: Clinician Scoring (plain-language questions)</title><style>{css}</style></head>
<body><a id="top"></a>
<header>
  <h1>Oncology Note Structured Extraction · PL vs BL Clinician Scoring</h1>
  <p>PL is our method (multi-stage extraction, a 5-gate verification cascade, drug and term dictionaries, and post-processing). BL is the same model (Qwen2.5-32B) run from a single prompt with no post-processing. The pipeline is the only thing that differs.</p>
  <p>There are 40 de-identified samples (20 breast and 20 PDAC, from UCSF CORAL, with ***** redacted). Score every question on each sample (<b>16 for breast, 15 for PDAC</b>; the type/receptor question is breast-only, since PDAC has no ER/PR/HER2 markers) as PL better, Tie, BL better, or N/A. Your scores save in the browser as you go, and you can export them anytime.</p>
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
  <span class="barrule">⚑ We are grading <b>extraction</b>, not summary. A vague summary that happens to read as correct still skipped the job, so it does not win.</span>
</div>
<div class="legend">
  <div class="principle"><b>★ Scoring principle: grade extraction, not summary or inference.</b> The job is to <b>pull the concrete facts out of the note</b>. So:
    <ul>
      <li>When PL lists the raw details (drugs, sizes, nodes, dates) without forcing a tidy conclusion, that is a <b>good</b> outcome. It means the details got pulled out, and you, the clinician, can judge them.</li>
      <li>When BL gives a short, vague summary that drops the details but happens to read as correct, that is <b>off task</b>. It may sound right, but it skipped the extraction job, so do not mark "BL better" for it.</li>
      <li>Mark "BL better" only when BL actually extracted something <b>true and specific that PL missed or got wrong</b>, not just because BL reads shorter or smoother.</li>
    </ul>
  </div>
  <div><b>How to score:</b> For each question you see the same field as PL extracted it and as BL extracted it. Using the source note above (the A/P section, or expand for the full note), decide which is more faithful, more complete, and more correct, then choose <b>PL better, Tie, or BL better</b>. If both are empty and the note really has no such item, choose N/A.</div>
  <div class="tiers">
    <div><b>Depth tiers:</b></div>
    <div><span class="tier t-good">Deep · needs medical knowledge</span> needs clinical background: telling anticancer drugs from supportive ones, stage, metastasis, response, receptors, molecular results.</div>
    <div><span class="tier t-mid">Medium</span> moderate.</div>
    <div><span class="tier t-low">Basic · layperson</span> things like lab values, answerable without medical training.</div>
    <div><b>Please put your effort into the Deep questions.</b></div>
  </div>
  <div><b>PL attribution</b> is a quote that <b>the model itself produced</b>: a second model pass was asked to cite the exact phrase from the note that backs up the value. It comes from the model, not from us, so use it to check whether PL is faithful, but still confirm it against the note. <b>Nothing is pre-filled, so please score on your own.</b></div>
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
