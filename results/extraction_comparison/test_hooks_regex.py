#!/usr/bin/env python3
# Deterministic regex validation for the 9 bug-fix hooks (no GPU). Parses the FINAL PL outputs to
# get the REAL note_text / assessment_and_plan / keypoint values, then runs each hook's trigger
# logic (copied verbatim from run.py) against its target sample + negative-control samples.
import re, json, sys

def parse_rows(path):
    txt = open(path, encoding="utf-8").read()
    rows = {}
    for chunk in re.split(r'RESULTS FOR ROW ', txt)[1:]:
        rid = int(re.match(r'(\d+)', chunk).group(1))
        def col(name):
            m = re.search(r'--- Column: ' + re.escape(name) + r' ---\n(.*?)(?:\n\n--- Column:|\n\n\n)', chunk, re.DOTALL)
            return m.group(1).strip() if m else ""
        nt = col("note_text")
        ap = col("assessment_and_plan")
        # note_text / ap are JSON-quoted single strings
        for k in ("nt", "ap"):
            v = nt if k == "nt" else ap
            if v.startswith('"'):
                try: v = json.loads(v)
                except Exception: v = v.strip('"')
            if k == "nt": nt = v
            else: ap = v
        kp = {}
        m = re.search(r'--- Column: keypoints ---\n(\{.*?\n\})\n', chunk, re.DOTALL)
        if m:
            try: kp = json.loads(m.group(1))
            except Exception: kp = {}
        rows[rid] = {"note_text": nt, "assessment_and_plan": ap, "keypoints": kp}
    return rows

B = parse_rows("pipeline_breast_FINAL.txt")
P = parse_rows("pipeline_pdac_FINAL.txt")

results = []
def check(label, got, want):
    ok = got == want
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")

# ---- bug6 POST-METASTATIC-UPGRADE trigger ----
HEDGE_MU = re.compile(r'\b(?:no|not|without|negative for|r/o|rule out|resolv\w*|resolution|possib\w*|'
                      r'concern\w*|suspicious|suspect\w*|question\w*|may|might|likely|could|evaluat\w*|'
                      r'assess\w*|differential|cannot exclude|worrisome|if )\b')
def upgrade_fires(ap, nt, stage):
    already = bool(re.search(r'stage\s*iv|metastatic|suspected', stage, re.IGNORECASE))
    txt = (ap + " " + nt).lower()
    marker = None
    for pat, site in [(r'peritoneal carcinomatosis','p'),(r'\bcarcinomatosis\b','c'),
                      (r'omental cak(?:e|ing)','o'),(r'biopsy[\s-]*(?:proven|confirmed)[^.]{0,30}metasta','b')]:
        for mm in re.finditer(pat, txt):
            ctx = txt[max(0,mm.start()-32):mm.start()]
            if not HEDGE_MU.search(ctx):
                marker = site; break
        if marker: break
    return bool(marker and not already)
r = P[12]; check("bug6 pdac12 upgrade", upgrade_fires(r["assessment_and_plan"], r["note_text"], r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]), True)
for neg in (9, 20, 3):  # already-IV pdac → should NOT upgrade (guard already_iv)
    r = P[neg]; st = r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]
    check(f"bug6 pdac{neg} no-upgrade(alreadyIV)", upgrade_fires(r["assessment_and_plan"], r["note_text"], st), False)
for neg in (6, 9):  # breast suspected (no carcinomatosis) → no upgrade
    r = B[neg]; st = r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]
    check(f"bug6 breast{neg} no-upgrade", upgrade_fires(r["assessment_and_plan"], r["note_text"], st), False)
# pdac6: only "possibility of peritoneal carcinomatosis" + laparoscopy unremarkable → must NOT upgrade
r = P[6]; st = r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]
check("bug6 pdac6 no-upgrade(hedged+laparoscopy neg)", upgrade_fires(r["assessment_and_plan"], r["note_text"], st), False)

# ---- bug4 POST-DISTMET-BENIGN trigger ----
def benign_fires(ap, nt, dm, m, stage, goals):
    dm_u = any(k in dm.lower() for k in ("not sure","unsure","suspect","possible"))
    m_u = any(k in m.lower() for k in ("not sure","unsure","suspect","possible"))
    cur = ("curative" in goals or "adjuvant" in goals or "risk reduction" in goals)
    nm = ("iv" not in stage.lower() and "metastatic" not in stage.lower())
    if not((dm_u or m_u) and cur and nm): return False
    nb = nt.lower(); ab = ap.lower()
    benign = re.search(r'most likely (?:a |an )?(?:meningioma|hemangioma|benign|cyst|lipoma|adenoma)|'
        r'(?:consistent with|favor|likely)\s+(?:a |an )?(?:meningioma|hemangioma|benign|cyst|lipoma)|'
        r'metasta\w*\s+(?:is|are|remains?)\s+(?:an?\s+)?(?:unlikely|very unlikely)|'
        r'unlikely\s+(?:to\s+(?:represent|be)\s+)?(?:a\s+)?metasta', nb)
    rp = re.search(r'pending confirmation|follow[\s-]?up on (?:the )?(?:lung|liver|bone|lesion)|'
        r'suspicious for (?:distant\s+)?metasta|concerning for (?:distant\s+)?metasta|'
        r'biopsy[^.]{0,30}(?:lesion|nodule|met)|nodules? pending', ab+" "+dm.lower()+" "+m.lower())
    return bool(benign and not rp)
def cd(row):
    c = row["keypoints"].get("Cancer_Diagnosis",{}); g = row["keypoints"].get("Treatment_Goals",{})
    return (c.get("Distant Metastasis","") or "", c.get("Metastasis","") or "", c.get("Stage_of_Cancer","") or "", (g.get("goals_of_treatment","") or "").lower())
r=B[13]; dm,m,st,g=cd(r); check("bug4 breast13 benign-fires", benign_fires(r["assessment_and_plan"],r["note_text"],dm,m,st,g), True)
r=B[20]; dm,m,st,g=cd(r); check("bug4 breast20 NO-fire(real pending)", benign_fires(r["assessment_and_plan"],r["note_text"],dm,m,st,g), False)
r=B[6]; dm,m,st,g=cd(r); check("bug4 breast6 NO-fire(palliative)", benign_fires(r["assessment_and_plan"],r["note_text"],dm,m,st,g), False)

# ---- bug3 POST-DISTMET-PENDING trigger ----
def pending_fires(ap, dm, stage):
    dml=dm.strip().lower(); nm=("iv" not in stage.lower() and "metastatic" not in stage.lower())
    noemp=(dml in ("","no","no.","none") or dml.startswith("no "))
    if not(nm and noemp): return False
    a=ap.lower()
    ps=re.search(r'staging\s+(imaging|scans?|work[\s-]?up|ct|pet)[^.]{0,50}(metasta|spread|assess|stage|distant)|'
        r'(obtain|order|recommend|will\s+(get|obtain|order)|role of|plan(?:ning)?\s+(?:for|to (?:get|obtain)))'
        r'[^.]{0,40}(pet[\s/]*ct|ct\s+(?:chest|c/?a/?p|of the chest)|bone scan|staging)[^.]{0,40}(metasta|assess|stage|spread)|'
        r'(imaging|pet[\s/]*ct|scans?)\s+to\s+(assess|evaluate|look)[^.]{0,25}(metasta|spread|for distant)', a)
    cn=re.search(r'no evidence of (distant\s+)?metasta|staging[^.]{0,20}negative|negative for (distant\s+)?metasta|'
        r'(w/?u|workup)\s+negative|no distant (disease|metasta)', a)
    return bool(ps and not cn)
r=B[1]; dm,m,st,g=cd(r); check("bug3 breast1 pending-fires", pending_fires(r["assessment_and_plan"],dm,st), True)
r=B[3]; dm,m,st,g=cd(r); check("bug3 breast3 NO-fire(no-mets stated)", pending_fires(r["assessment_and_plan"],dm,st), False)

# ---- bug7 POST-RESPONSE-TREATMENT tightened on_treatment + SURVEILLANCE ----
def on_treatment(ap):
    return bool(re.search(r'(?:currently on|on cycle|cycle\s*\d|c\d+\s*d\d+|'
        r'(?:continue|continuing|on)\s+(?:\w+\s+){0,2}\w*'
        r'(?:oxifen|zole|mab|lib|nib|platin|tabine|rubicin|taxel|fluorouracil|'
        r'gemcitabine|capecitabine|folfirinox|folfox|folfiri|pembrolizumab|chemo))', ap.lower()))
r=P[15]; check("bug7 pdac15 on_treatment(continue creon)=False", on_treatment(r["assessment_and_plan"]), False)
def surv_fires(ap, nt, cur_meds, rv):
    ctx=(ap+" "+nt).lower()
    res=re.search(r'\b(?:resected|s/p\s+(?:resection|whipple|pancrea\w*ectomy|mastectomy|lumpectomy|surgery)|'
        r'status post (?:resection|surgery)|post[\s-]?(?:operative|surgical resection))', ctx)
    sv=re.search(r'surveillance|rising (?:marker|ca\s*19|cea)|high risk for recurrence|'
        r'monitor(?:ing)? for recurrence|recheck (?:ca\s*19|cea|markers)|concern\w* for recurrence', ctx)
    rl=rv.lower(); mis=(("on treatment" in rl and "not" not in rl[:30]) or "not yet on treatment" in rl or "not on treatment" in rl or not rl.strip())
    return bool(res and sv and not cur_meds and mis)
r=P[15]; c=r["keypoints"]["Cancer_Diagnosis"]; rv=r["keypoints"]["Response_Assessment"]["response_assessment"]
cm=(r["keypoints"]["Current_Medications"]["current_meds"] or "").strip()
check("bug7 pdac15 surveillance-fires", surv_fires(r["assessment_and_plan"],r["note_text"],cm,rv), True)

# ---- bug9 POST-STAGE-PTNM-VERIFY ----
def ptnm_fix(ap, nt, stage):
    vt=re.search(r'(?:yp|p|c)?T\s*\d[a-d]?\s*N\s*\d[a-c]?', stage, re.IGNORECASE)
    if not vt: return stage
    src=ap+" "+nt
    n=re.search(r'\b(yp|p)T(\d)([a-d]?)\s*,?\s*N(\d)([a-c]?)', src, re.IGNORECASE)
    if not n: return stage
    ns=f"{(n.group(1) or '').lower()}T{n.group(2)}{n.group(3) or ''}N{n.group(4)}{n.group(5) or ''}"
    norm=lambda s: re.sub(r'\s','',s).lower()
    if norm(vt.group(0))!=norm(ns):
        return re.sub(r'(?:yp|p|c)?T\s*\d[a-d]?\s*N\s*\d[a-c]?', ns, stage, count=1, flags=re.IGNORECASE)
    return stage
r=P[15]; check("bug9 pdac15 pTNM→pT2N3", ptnm_fix(r["assessment_and_plan"],r["note_text"],r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]), "pT2N3")

# ---- bug2 POST-STAGE-CTNM ----
def ctnm_fill(ap, nt, stage):
    sl=stage.strip().lower()
    if sl and sl not in ("not mentioned","not mentioned in note","not available","not available (redacted)","not specified","not staged in note","not specified in the note",""): return stage
    for src in (ap, nt):
        m=re.search(r'\b(?:(clinical|pathologic|path)\s+)?(c|p|yp)?T(\d)([a-d]?)\s*,?\s*N([0-3X])([a-c]?)', src, re.IGNORECASE)
        if m:
            cw,pfx,t,ts,n,nx=m.groups(); pfx=(pfx or "").lower()
            if not pfx and cw and cw.lower().startswith("clinic"): pfx="c"
            tnm=f"{pfx}T{t}{ts or ''}N{n.upper()}{nx or ''}"
            return tnm+(" (clinical staging)" if pfx=="c" else "")
    return stage
r=B[18]; check("bug2 breast18 cT2NX", ctnm_fill(r["assessment_and_plan"],r["note_text"],r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]), "cT2NX (clinical staging)")

# ---- bug1 POST-STAGE-BILATERAL ----
def bilateral(ap):
    pairs=re.findall(r'stage\s+(IV|III[ABC]?|II[ABC]?|I[ABC]?)\s*\(([^)]*)\)[^.]{0,90}?\b(left|right)\s+breast', ap, re.IGNORECASE)
    sides={}
    for st,tn,side in pairs:
        sk=side.capitalize()
        if sk not in sides: sides[sk]=f"Stage {st.upper()} ({tn.strip()})"
    if len(sides)>=2:
        return "; ".join(f"{s}: {sides[s]}" for s in ("Left","Right") if s in sides)
    return None
r=B[5]; check("bug1 breast5 bilateral", bilateral(r["assessment_and_plan"]), "Left: Stage III (T3N1); Right: Stage I (T1cN0)")

# ---- bug5 cervical in DISTANT + R2 site-preserving ----
DISTANT_RC=["liver","hepatic","lung","pulmonary","bone","osseous","brain","cerebral","peritone","pleural","adrenal","distant","contralateral","spine","spinal","mediastin","retroperitone","mesenteric","omentum","omental","metastatic","cervical"]
def r2(dm, m):
    has_d=any(s in m.lower() for s in DISTANT_RC)
    dm_empty=(dm.strip()=="")
    m_yes=("yes" in m.lower())
    if m_yes and has_d and dm_empty:
        sites=re.sub(r'(?i)^\s*yes[\s,:.-]*(?:to\s+)?','',m).strip()
        return f"Suspected, to {sites}" if sites and sites.lower() not in ("","yes") else "Not sure"
    return None
r=B[15]; c=r["keypoints"]["Cancer_Diagnosis"]
check("bug5 breast15 R2 suspected-sites", r2(c.get("Distant Metastasis","") or "", c.get("Metastasis","") or ""),
      "Suspected, to right cervical lymph nodes, right axillary lymph nodes")

# ---- bug10 POST-DISTMET-SITES: REMOVED (fired on negated "No osseous lesions" → hallucination). ----

if __name__ == "__main__":
    print(f"\n==== {sum(results)}/{len(results)} PASS ====")
    sys.exit(0 if all(results) else 1)
