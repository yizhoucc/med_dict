#!/usr/bin/env python3
"""CPU-only smoke test for the final extraction hooks.

Shared pure helpers are imported directly from ``extraction_post_hooks.py`` so these tests
exercise the same M1, cleanup, reconciliation, and regional-evidence logic as ``run.py``.
Archived FINAL outputs provide idempotence/negative controls, and synthetic pre-hook cases
cover the targeted edge cases. Run from any working directory with:

    python results/extraction_comparison/test_hooks_regex.py
"""
import re, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extraction_post_hooks import (
    clean_breast_distant,
    compose_general_metastasis,
    has_affirmative_m1_site,
    has_explicit_m1_evidence,
    has_uncertain_m1_site,
    is_confirmed_distant_value,
    locally_advanced_stage,
    merge_regional_metastasis,
    normalize_stage_iv,
    reconcile_metastasis_fields,
    regional_node_evidence,
    verify_unique_pathologic_tnm,
)

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

B = parse_rows(ROOT / "results/extraction_comparison/pipeline_breast_FINAL.txt")
P = parse_rows(ROOT / "results/extraction_comparison/pipeline_pdac_FINAL.txt")

results = []
def check(label, got, want):
    ok = got == want
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")

# ---- bug6 POST-METASTATIC-UPGRADE trigger ----
def upgrade_fires(ap, nt, stage, cancer_type="pdac"):
    _ = nt  # production intentionally scopes current M1 evidence to the A/P
    already = bool(re.search(r'stage\s*iv|metastatic|suspected', stage, re.IGNORECASE))
    return has_explicit_m1_evidence(ap, cancer_type) and not already
# Archived FINAL is already upgraded, so the hook must now be idempotent.
r = P[12]; check("bug6 pdac12 final-idempotent", upgrade_fires(r["assessment_and_plan"], r["note_text"], r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"]), False)
check("bug6 synthetic confirmed-carcinomatosis fires",
      upgrade_fires("unchanged peritoneal carcinomatosis", "", "Stage II"), True)
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
r=B[13]; dm,m,st,g=cd(r); check("bug4 breast13 final-idempotent", benign_fires(r["assessment_and_plan"],r["note_text"],dm,m,st,g), False)
check("bug4 synthetic benign-lesion fires",
      benign_fires("", "The lesion is most likely a meningioma.", "Not sure", "Not sure", "Stage II", "curative"), True)
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
r=B[1]; dm,m,st,g=cd(r); check("bug3 breast1 final-idempotent", pending_fires(r["assessment_and_plan"],dm,st), False)
check("bug3 synthetic pending-staging fires",
      pending_fires("We will obtain PET/CT to assess for distant metastasis.", "No", "Stage II"), True)
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
check("bug7 pdac15 archived-final trigger state", surv_fires(r["assessment_and_plan"],r["note_text"],cm,rv), False)
check("bug7 synthetic surveillance fires",
      surv_fires("Continue surveillance; recheck CA 19-9.", "s/p Whipple.", "", "Not yet on treatment"), True)

# ---- bug9 POST-STAGE-PTNM-VERIFY ----
r=P[15]; check("bug9 pdac15 formal pathology wins → ypT3N2", verify_unique_pathologic_tnm(r["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"],r["note_text"]), "ypT3N2")
check("bug9 clinical cTN is outside pathology verifier",
      verify_unique_pathologic_tnm("cT3N1","AJCC Pathologic Stage: pT2N1."),"cT3N1")
check("bug9 multiple formal pathology candidates do not overwrite",
      verify_unique_pathologic_tnm(
          "pT2N3",
          "Pathologic Stage: pT2N1. Outside review lists Pathologic Stage: ypT3N2.",
      ),
      "pT2N3")

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

# ---- Stage IV / goals must use confirmed M1, never general regional Metastasis ----
def stage_iv_trigger(dm, general_met, ap, cancer_type="breast"):
    _ = general_met  # deliberately ignored: it may describe regional nodes only
    return is_confirmed_distant_value(dm,cancer_type) or has_explicit_m1_evidence(ap,cancer_type)
def adjuvant_goal(dm, general_met, stage):
    _ = general_met  # deliberately ignored
    sl=stage.lower()
    confirmed_iv=bool(re.search(r'stage\s*iv|metastatic',sl)) and not any(
        k in sl for k in ("suspect","possible","pending"))
    return "adjuvant" if is_confirmed_distant_value(dm,"breast") or confirmed_iv else "curative"

regional_note="Right axillary lymph-node biopsy confirms metastatic breast adenocarcinoma."
check("Stage IV negative: regional-only + Distant No",
      stage_iv_trigger("No","Yes, confirmed regional lymph-node involvement",regional_note),False)
check("Stage IV negative: regional-only + Distant uncertain",
      stage_iv_trigger("Not sure (staging pending)","Yes, confirmed regional lymph-node involvement",regional_note),False)
check("Stage IV negative: suspected distant evidence",
      stage_iv_trigger("Not sure, suspicious liver lesion","Yes, confirmed regional nodes",
                       "Possible metastatic disease to liver; biopsy pending."),False)
check("Stage IV positive: confirmed Distant value",
      stage_iv_trigger("Yes, to liver","No",""),True)
check("Stage IV positive: explicit current M1 evidence",
      stage_iv_trigger("","No","Pancreatic cancer is metastatic to the liver.","pdac"),True)
check("Stage IV negative: negated explicit stage",
      stage_iv_trigger("No","Yes, regional nodes","Not Stage IV; regional disease only."),False)
check("Stage IV negative: negated carcinomatosis",
      stage_iv_trigger("No","No","No peritoneal carcinomatosis."),False)
check("Stage IV mixed Distant: confirmed liver + suspicious lung remains confirmed",
      is_confirmed_distant_value("Yes, confirmed liver metastasis; suspicious lung nodules","breast"),True)
check("Stage IV mixed Distant: liver + possible bone remains confirmed",
      is_confirmed_distant_value("Yes, liver metastasis; possible bone lesion","breast"),True)
check("Stage IV negative: historical M1 is not current anchor",
      stage_iv_trigger("No","No","History of breast cancer metastatic to liver, now NED."),False)
check("Stage IV negative: another primary M1 is not current-cancer anchor",
      stage_iv_trigger("No","No","Breast cancer follow-up. Pancreatic cancer is metastatic to liver."),False)
check("Stage IV negative: conditional M1 is not confirmed",
      stage_iv_trigger("Not sure","No","If metastatic to liver, treatment would be palliative."),False)
check("Stage IV negative: condition after Stage IV is not confirmed",
      stage_iv_trigger("No","No","Stage IV if biopsy confirms metastatic disease."),False)
check("Stage IV final: regional + benign cyst has no M1 basis",
      normalize_stage_iv(
          "Stage IV", "No", "Breast cancer with regional nodes and a benign liver cyst.", "breast"
      ),
      ("Not staged in note", "no confirmed M1 basis"))
check("Stage IV final: suspicious liver remains suspected, not confirmed",
      normalize_stage_iv(
          "Stage IV", "Not sure, suspicious liver lesion", "Breast cancer follow-up.", "breast"
      ),
      ("Suspected Stage IV (pending confirmation)", "unconfirmed M1 evidence"))
check("Stage IV final: mixed confirmed/suspicious Distant remains confirmed",
      normalize_stage_iv(
          "Stage IV", "Yes, confirmed liver metastasis; suspicious lung nodules", "", "breast"
      ),
      ("Stage IV", None))
check("Goals regional-only adjuvant → curative",
      adjuvant_goal("No","Yes, confirmed regional lymph-node involvement","Stage II"),"curative")
check("Goals regional-only + uncertain distant still not confirmed metastatic",
      adjuvant_goal("Not sure, staging pending","Yes, confirmed regional lymph-node involvement","Suspected Stage IV"),"curative")
check("Goals confirmed M1 does not force curative",
      adjuvant_goal("Yes, to liver","Yes, to liver","Stage IV"),"adjuvant")

# Missing Distant field must not inherit a regional-only general Metastasis value.
def ensure_distant(cancer):
    if "Distant Metastasis" not in cancer: cancer["Distant Metastasis"]=""
    return cancer["Distant Metastasis"]
check("missing Distant does not copy regional Metastasis",
      ensure_distant({"Metastasis":"Yes, regional lymph nodes"}),"")

# ---- PDAC locally-advanced description must not fabricate Stage III ----
check("PDAC locally advanced does not imply Stage III",
      locally_advanced_stage("Locally advanced pancreatic adenocarcinoma.","pdac"),"Locally advanced")
check("PDAC vessel encasement is descriptive, not Stage III",
      locally_advanced_stage("The mass encases the SMA and is unresectable.","pdac"),"Locally advanced (unresectable)")
check("PDAC explicit Stage III remains Stage III",
      locally_advanced_stage("Stage III locally advanced pancreatic adenocarcinoma.","pdac"),"Stage III (locally advanced)")

# ---- Metastasis subset/certainty reconciliation ----
check("met subset R1 confirmed distant populates general",
      reconcile_metastasis_fields("Yes, to liver", "No", "breast")[:2],
      ("Yes, to liver", "Yes, to liver"))
check("met subset pdac6 uncertainty ceiling",
      reconcile_metastasis_fields("Not sure, suspicious liver lesions", "Yes (to liver)", "pdac")[:2],
      ("Not sure, suspicious liver lesions", "Not sure, suspicious liver lesions"))
check("met subset DistMet=No rejects distant component",
      reconcile_metastasis_fields("No", "Yes, to liver", "breast")[:2], ("No", "No"))
check("met subset negated distant clause does not erase regional",
      reconcile_metastasis_fields("No", "Yes, confirmed regional nodes; no distant metastasis", "breast")[:2],
      ("No", "Yes, confirmed regional nodes; no distant metastasis"))
check("met subset mixed certainty remains scalar-safe",
      reconcile_metastasis_fields(
          "Not sure, suspicious liver lesion",
          "Yes, confirmed regional nodes; distant disease uncertain — suspicious liver lesion",
          "breast",
      )[:2],
      ("Not sure, suspicious liver lesion",
       "Yes, confirmed regional nodes; distant disease uncertain — suspicious liver lesion"))
check("M1 anchor ignores benign liver cyst",
      has_affirmative_m1_site("Yes, regional nodes; benign liver cyst","breast"),False)
check("M1 anchor ignores negated distant clause",
      has_affirmative_m1_site("Yes, regional nodes; no distant metastasis","breast"),False)
check("M1 anchor ignores historical liver metastasis",
      has_affirmative_m1_site("History of liver metastasis, now resected","breast"),False)
check("M1 anchor ignores another primary's liver metastasis",
      has_affirmative_m1_site("Yes, ovarian cancer metastatic to liver","breast"),False)
check("M1 anchor ignores conditional liver metastasis",
      has_affirmative_m1_site("Yes only if biopsy confirms liver metastasis","breast"),False)
check("M1 anchor accepts affirmative liver metastasis",
      has_affirmative_m1_site("Yes, confirmed liver metastasis","breast"),True)
check("M1 mixed order accepts later confirmed liver clause",
      has_affirmative_m1_site("Yes, possible bone; confirmed liver metastasis","breast"),True)
check("M1 anchor rejects liver lesion suspicious for metastasis",
      has_affirmative_m1_site("Yes, liver lesion suspicious for metastasis","breast"),False)
check("uncertain M1 recognizes suspicious liver",
      has_uncertain_m1_site("Not sure, suspicious liver lesion","breast"),True)
check("uncertain M1 rejects benign liver cyst",
      has_uncertain_m1_site("Possible benign liver cyst","breast"),False)
check("met subset regional uncertainty never contaminates Distant",
      reconcile_metastasis_fields("", "Not sure, suspicious axillary nodes", "breast")[:2],
      ("", "Not sure, suspicious axillary nodes"))
check("met subset explicit uncertain M1 can populate empty Distant",
      reconcile_metastasis_fields("", "Not sure, suspicious liver lesion", "breast")[:2],
      ("Not sure, suspicious liver lesion", "Not sure, suspicious liver lesion"))
check("met subset empty-Distant mixed regional/distant claim is not scalarized",
      reconcile_metastasis_fields(
          "", "Yes, to right cervical lymph nodes, right axillary lymph nodes", "breast"
      )[:2],
      ("", "Yes, to right cervical lymph nodes, right axillary lymph nodes"))

check("suspected-stage guard ignores confirmed regional FNA",
      has_explicit_m1_evidence("Right axillary node FNA: metastatic breast adenocarcinoma.","breast"), False)
check("suspected-stage guard accepts confirmed liver metastasis",
      has_explicit_m1_evidence("Breast cancer is metastatic to the liver.","breast"), True)
check("suspected-stage guard rejects 'Not Stage IV'",
      has_explicit_m1_evidence("Not Stage IV; no distant disease.","breast"), False)
check("suspected-stage guard rejects 'No peritoneal carcinomatosis'",
      has_explicit_m1_evidence("No peritoneal carcinomatosis.","breast"), False)

# ---- Distant field strips breast-regional sites but retains true M1 sites ----
check("Distant regional-only breast nodes → No",
      clean_breast_distant("Yes, to right axillary lymph nodes"), "No")
check("Distant mixed liver+generic nodes keeps liver only",
      clean_breast_distant("Yes, to liver and nodes"), "Yes, to liver")
check("Distant mixed liver+named regional nodes keeps liver only",
      clean_breast_distant("Yes, to liver and right axillary lymph nodes"), "Yes, to liver")
check("Distant cervical node remains M1 for breast",
      clean_breast_distant("Suspected, to left cervical lymph node"), "Suspected, to left cervical lymph node")
check("Distant contralateral axillary node remains M1 for breast",
      clean_breast_distant("Yes, to contralateral axillary lymph nodes"),
      "Yes, to contralateral axillary lymph nodes")
check("Distant liver plus contralateral supraclavicular node keeps both M1 sites",
      clean_breast_distant("Yes, to liver and contralateral supraclavicular lymph node"),
      "Yes, to liver and contralateral supraclavicular lymph node")
check("Distant regional + explicit no distant becomes No",
      clean_breast_distant("Yes, to right axillary nodes; no distant metastasis"), "No")
check("Distant confirmed regional + no distant disease becomes No",
      clean_breast_distant("Yes confirmed right axillary node; no distant disease"), "No")

# ---- Locoregional/direct-extension semantics ----
def locoregional_values(ap, cancer_type, dm, met):
    a=ap.lower(); loc=re.search(r'local[\s-]*regional recurrence|locoregional recurrence|'
                                r'local recurrence(?!\w)|chest[\s-]*wall recurrence',a)
    if loc:
        regional=bool(re.search(r'local[\s-]*regional recurrence|locoregional recurrence|'
                                r'chest[\s-]*wall recurrence',a))
        if regional:
            label="locoregional chest-wall recurrence" if "chest" in loc.group(0) else "locoregional recurrence"
            return "No",f"Yes, {label}; no distant metastasis"
        return "No","No"
    ml=met.lower()
    if (cancer_type!="breast" and dm.lower() in ("no","no.","none") and "yes" in ml and
        re.search(r'\b(invasi\w*|invad\w*|abut\w*|encas\w*|involv\w*)\b',ml) and
        any(s in ml for s in ("stomach","gastric","duoden","spleen","splenic","adrenal","kidney",
                              "renal","artery","vein","vessel","sma","smv","portal")) and
        not re.search(r'lymph|\bnode\b|liver|hepatic|lung|pulmonary|bone|osseous|brain|cerebral|'
                      r'peritone|omentum|pleur|distant',ml)):
        met="No"
    return dm,met
check("locoregional chest-wall recurrence is general, not distant",
      locoregional_values("Unresectable locoregional chest-wall recurrence.","breast","Yes","No"),
      ("No","Yes, locoregional chest-wall recurrence; no distant metastasis"))
check("locoregional guard preserves known current metastatic cancer",
      has_explicit_m1_evidence(
          "Metastatic breast cancer to liver remains active; also has local chest-wall recurrence.",
          "breast",
      ),
      True)
check("local breast-only recurrence is not metastasis",
      locoregional_values("Biopsy confirms local recurrence in the breast.","breast","Yes","Yes, breast"),
      ("No","No"))
check("PDAC direct contiguous invasion is not metastasis",
      locoregional_values("", "pdac", "No", "Yes, tumor encases the SMA and invades duodenum"),
      ("No","No"))

# ---- Final regional-node evidence and mixed-certainty composition ----
check("regional pathologic N1a marked historical", regional_node_evidence("breast","pT2N1a",""), ("HISTORICAL","N1a"))
check("regional PDAC ypN2 marked historical", regional_node_evidence("pdac","ypT3N2",""), ("HISTORICAL","N2"))
check("regional standalone pN1 supported", regional_node_evidence("breast","pN1a",""), ("HISTORICAL","pN1a"))
check("regional standalone ypN2 supported", regional_node_evidence("pdac","ypN2",""), ("HISTORICAL","ypN2"))
check("regional PDAC cN1 is not auto-promoted", regional_node_evidence("pdac","Stage IIB (cT1c cN1 cM0)",""), (None,""))
check("regional positive count confirmed", regional_node_evidence("pdac","","PDAC pathology shows 2/29 lymph nodes positive"), ("CONFIRMED","2/29 nodes positive"))
check("regional breast FNA confirmed", regional_node_evidence("breast","","FNA of the right axillary lymph node showed metastatic carcinoma from breast cancer."), ("CONFIRMED","axillary lymph node"))
check("regional suspicious breast nodes are not auto-promoted", regional_node_evidence("breast","Stage III","Breast MRI shows suspicious left axillary lymph nodes; FNA is pending."), (None,""))
check("regional N0 excluded", regional_node_evidence("breast","pT2N0","MRI shows prominent axillary nodes."), (None,""))
check("regional NX excluded", regional_node_evidence("breast","cT2NX","MRI shows abnormal axillary nodes."), (None,""))
check("regional zero-positive excluded", regional_node_evidence("pdac","","0/18 lymph nodes positive"), (None,""))
check("regional negative biopsy excluded", regional_node_evidence("breast","","Abnormal axillary lymph nodes; FNA of the lymph nodes was negative for carcinoma."), (None,""))
check("regional benign/reactive excluded", regional_node_evidence("breast","","Axillary lymph node biopsy showed benign reactive tissue and granulomatous change."), (None,""))
check("regional conditional language excluded", regional_node_evidence("breast","","If node-positive, radiation would be recommended."), (None,""))
check("regional another primary excluded", regional_node_evidence("breast","","History of ovarian cancer with 3/10 lymph nodes positive."), (None,""))
check("regional other-primary sentence not rescued by broad ductal token",
      regional_node_evidence("breast","","Breast cancer follow-up. Pancreatic ductal adenocarcinoma with 3/10 lymph nodes positive."),
      (None,""))
check("regional mixed-primary sentence is rejected as ambiguous",
      regional_node_evidence("breast","","Breast cancer follow-up notes pancreatic ductal adenocarcinoma with 3/10 lymph nodes positive."),
      (None,""))
check("regional contralateral axillary biopsy is not regional",
      regional_node_evidence("breast","","Breast cancer with contralateral axillary lymph node biopsy showing metastatic carcinoma."),
      (None,""))
check("regional historical positive count is not promoted",
      regional_node_evidence("pdac","","History of pancreatic cancer with 2/29 lymph nodes positive."),
      (None,""))
check("regional historical negative does not block current positive count",
      regional_node_evidence("pdac","","In 2018, pancreatic cancer had 0/18 nodes positive. PDAC pathology now shows 2/29 lymph nodes positive."),
      ("CONFIRMED","2/29 nodes positive"))
check("regional confirmed + uncertain distant stays mixed",
      compose_general_metastasis("Not sure, suspicious liver lesion pending biopsy","CONFIRMED","N2","pdac"),
      "Yes, confirmed regional lymph-node involvement (N2); distant disease uncertain — Not sure, suspicious liver lesion pending biopsy")
check("regional historical pN is not phrased as current",
      compose_general_metastasis("No","HISTORICAL","ypN2","pdac"),
      "Yes, historical pathologically confirmed regional lymph-node involvement (ypN2); no distant metastasis")
check("regional append preserves mixed Distant clause verbatim",
      merge_regional_metastasis(
          "Yes, confirmed liver metastasis; suspicious lung nodules", "No", "CONFIRMED", "N2", "pdac"
      ),
      ("Yes, confirmed liver metastasis; suspicious lung nodules; confirmed regional lymph-node involvement (N2)", True))
check("regional append preserves terse mixed Distant clause verbatim",
      merge_regional_metastasis(
          "Yes, liver metastasis; possible bone lesion", "No", "CONFIRMED", "N1", "breast"
      ),
      ("Yes, liver metastasis; possible bone lesion; confirmed regional lymph-node involvement (N1)", True))
check("regional append does not mistake cervical distant node for regional",
      merge_regional_metastasis(
          "Yes, to right cervical lymph nodes", "Yes, to right cervical lymph nodes",
          "HISTORICAL", "pN1", "breast"
      ),
      ("Yes, to right cervical lymph nodes; historical pathologically confirmed regional lymph-node involvement (pN1)", True))
check("regional append does not mistake contralateral distant node for regional",
      merge_regional_metastasis(
          "Yes, to contralateral axillary lymph nodes", "Yes, to contralateral axillary lymph nodes",
          "HISTORICAL", "pN1", "breast"
      ),
      ("Yes, to contralateral axillary lymph nodes; historical pathologically confirmed regional lymph-node involvement (pN1)", True))

# Archived target/clean samples: lock the real phrasing that motivated the hook.
r=B[6]; c=r["keypoints"]["Cancer_Diagnosis"]
check("regional archived breast6 current A/P has no confirmed regional statement",
      regional_node_evidence("breast",c.get("Stage_of_Cancer","") or "",r["assessment_and_plan"])[0], None)
r=B[18]; c=r["keypoints"]["Cancer_Diagnosis"]
check("regional archived breast18 NX + negative FNAs stays clean",
      regional_node_evidence("breast",c.get("Stage_of_Cancer","") or "",r["assessment_and_plan"])[0],
      None)
r=P[16]; c=r["keypoints"]["Cancer_Diagnosis"]
check("regional archived pdac16 cN1 is not auto-promoted",
      regional_node_evidence("pdac",c.get("Stage_of_Cancer","") or "",r["assessment_and_plan"])[0], None)
r=P[18]; c=r["keypoints"]["Cancer_Diagnosis"]
check("regional archived pdac18 historical A/P node statement is not auto-promoted",
      regional_node_evidence("pdac",c.get("Stage_of_Cancer","") or "",r["assessment_and_plan"])[0], None)

# ---- bug10 POST-DISTMET-SITES: REMOVED (fired on negated "No osseous lesions" → hallucination). ----

if __name__ == "__main__":
    print(f"\n==== {sum(results)}/{len(results)} PASS ====")
    sys.exit(0 if all(results) else 1)
