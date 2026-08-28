#!/usr/bin/env python3
"""CPU-only smoke test for the final extraction hooks.

Shared pure helpers are imported directly from ``extraction_post_hooks.py`` so these tests
exercise the same M1, response, cleanup, reconciliation, and regional-evidence logic as ``run.py``.
Archived FINAL outputs provide idempotence/negative controls, and synthetic pre-hook cases
cover the targeted edge cases. Run from any working directory with:

    python results/extraction_comparison/test_hooks_regex.py
"""
import re, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extraction_post_hooks import (
    align_stage_with_confirmed_distant,
    clear_held_anticancer_meds,
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
    recover_completed_genetic_results,
    regional_node_evidence,
    resolve_current_anticancer_meds,
    sanitize_distant_metastasis_by_site,
    sanitize_general_metastasis,
    sanitize_genetic_testing_results,
    sanitize_breast_recurrence_receptors,
    sanitize_response_assessment,
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
PV21 = parse_rows(ROOT / "results/extraction_comparison/pipeline_pdac_matched_v21.txt")
BV21 = parse_rows(ROOT / "results/extraction_comparison/pipeline_breast_matched_v21.txt")

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

# ---- POST-RESPONSE-FINAL shared sanitizer ----
def sanitize_row(row):
    kp = row["keypoints"]
    return sanitize_response_assessment(
        kp.get("Response_Assessment", {}).get("response_assessment", ""),
        row["note_text"],
        row["assessment_and_plan"],
        current_meds=kp.get("Current_Medications", {}).get("current_meds", ""),
        recent_changes=kp.get("Treatment_Changes", {}).get("recent_changes", ""),
        findings=kp.get("Clinical_Findings", {}).get("findings", ""),
    )[0]

fixed = sanitize_row(B[4])
check("response breast4 recurrence/growth replaces untreated", "not yet on treatment" in fixed.lower(), False)
check("response breast4 preserves measured growth", "growth" in fixed.lower() and "2.7" in fixed, True)

fixed = sanitize_row(B[14])
check("response breast14 same-day goserelin has started", fixed,
      "Anticancer treatment started today; too early to assess its response.")

fixed = sanitize_row(P[7])
check("response pdac7 slight decrease is not upgraded to PR", "partial response" in fixed.lower(), False)
check("response pdac7 preserves slight decrease", "slight" in fixed.lower() and "decreas" in fixed.lower(), True)

fixed = sanitize_row(P[13])
check("response pdac13 keeps stable disease", "stable disease" in fixed.lower(), True)
check("response pdac13 removes isolated biliary findings",
      bool(re.search(r"pneumobilia|biliary duct", fixed, re.I)), False)

fixed = sanitize_row(P[16])
check("response pdac16 preserves treatment-linked mass disappearance",
      "mass is no longer seen" in fixed.lower() and "treatment" in fixed.lower(), True)

fixed = sanitize_row(P[18])
check("response pdac18 removes pure postoperative vascular/fluid changes",
      bool(re.search(r"portal vein|superior mesenteric vein|free fluid|cut edge", fixed, re.I)), False)
check("response pdac18 falls back to direct tumor status",
      "no evidence of metastatic disease" in fixed.lower(), True)

for row_id in (13, 18):
    row = P[row_id]
    current = row["keypoints"]["Current_Medications"]["current_meds"]
    check(
        f"current meds pdac{row_id} held regimen clears",
        clear_held_anticancer_meds(
            current, row["note_text"], row["assessment_and_plan"]
        )[0],
        "",
    )

# ---- POST-MEDS-FINAL shared current-regimen resolver ----
def resolve_meds(current_meds, note="", ap="", changes=""):
    return resolve_current_anticancer_meds(
        current_meds, note, ap, recent_changes=changes
    )[0]

check(
    "meds active Gem/Abrax from explicit current treatment",
    resolve_meds(
        "",
        "She started gemcitabine and Abraxane and is currently receiving the alternate-week regimen.",
        "Will continue on with treatment without dose or schedule modification.",
    ),
    "gemcitabine, abraxane",
)
check(
    "meds active Gem/Abrax after cycles plus continuation",
    resolve_meds(
        "",
        "She started gemcitabine/nab-paclitaxel and is tolerating current chemotherapy well.",
        "Continue current treatment.",
    ),
    "gemcitabine, abraxane",
)
check(
    "meds FOLFOX only removes historical FOLFIRINOX",
    resolve_meds(
        "folfox, folfirinox",
        "Initially treated with mFOLFIRINOX.",
        "Omitted irinotecan since C3. Will continue with FOLFOX only going forward.",
    ),
    "FOLFOX",
)
check(
    "meds omitted irinotecan component removed",
    resolve_meds(
        "folfox, irinotecan",
        "",
        "Omitted irinotecan since C3. Will continue with FOLFOX only going forward.",
    ),
    "FOLFOX",
)
check(
    "meds whole chemotherapy hold clears Gem/Cape",
    resolve_meds(
        "gemcitabine, capecitabine",
        "Postoperative treatment has used gemcitabine and capecitabine.",
        "We will hold her chemotherapy until the hand-foot syndrome resolves, then resume treatment.",
    ),
    "",
)
check(
    "meds explicit not-taking clears capecitabine",
    resolve_meds(
        "capecitabine",
        "Capecitabine (Patient not taking: reported today).",
        "",
    ),
    "",
)
check(
    "meds one postponed cycle remains active",
    resolve_meds(
        "FOLFIRINOX",
        "She is on dose-modified FOLFIRINOX.",
        "She presents for C3 today; postpone today's infusion because of abnormal liver tests.",
    ),
    "FOLFIRINOX",
)
check(
    "meds one cancelled dose remains active",
    resolve_meds(
        "gemcitabine, abraxane",
        "Currently receiving gemcitabine and Abraxane on an alternate-week schedule.",
        "Day 8 was cancelled for neutropenia; continue the regimen.",
    ),
    "gemcitabine, abraxane",
)
check(
    "meds normal off-week remains active",
    resolve_meds(
        "capecitabine",
        "She is currently on capecitabine and this is her scheduled week off.",
        "Continue capecitabine.",
    ),
    "capecitabine",
)
check(
    "meds completed course plus holiday clears",
    resolve_meds(
        "FOLFIRINOX",
        "Completed 12 cycles of FOLFIRINOX and is now on a treatment holiday.",
        "Continue surveillance.",
    ),
    "",
)
check(
    "meds completed course plus surveillance clears",
    resolve_meds(
        "gemcitabine, abraxane",
        "Finished all six cycles of gemcitabine and Abraxane; now under surveillance.",
        "No active chemotherapy is planned.",
    ),
    "",
)
check(
    "meds planned-only regimen clears",
    resolve_meds(
        "FOLFIRINOX",
        "No treatment has started.",
        "Recommend starting FOLFIRINOX next week.",
    ),
    "",
)
check(
    "meds treatment options only clear",
    resolve_meds(
        "gemcitabine, abraxane",
        "The patient is treatment-naive.",
        "Options include FOLFIRINOX versus gemcitabine/nab-paclitaxel.",
    ),
    "",
)
check(
    "meds current AP overrides stale historical regimen",
    resolve_meds(
        "FOLFIRINOX, FOLFOX",
        "Previously received FOLFIRINOX.",
        "The patient is currently receiving FOLFOX only.",
    ),
    "FOLFOX",
)
check(
    "meds current cycle-day recovers doublet",
    resolve_meds(
        "",
        "The patient presents for C4D1 gemcitabine/nab-paclitaxel today.",
        "Proceed with today's cycle.",
    ),
    "gemcitabine, abraxane",
)
check(
    "meds bare status-post cycles do not become current",
    resolve_meds("", "Status post 6 cycles of FOLFIRINOX.", ""),
    "",
)
check(
    "meds current AP bridges status-post cycles to next cycle",
    resolve_meds(
        "",
        "",
        "Patient is now s/p 2 cycles of FOLFIRINOX. She presents for C3 today.",
    ),
    "FOLFIRINOX",
)
check(
    "meds hold one drug preserves concurrent targeted therapy",
    resolve_meds(
        "trastuzumab, capecitabine",
        "",
        "Hold capecitabine for toxicity; continue trastuzumab.",
    ),
    "trastuzumab",
)
check(
    "meds conditional future switch does not replace current regimen",
    resolve_meds(
        "FOLFIRINOX",
        "",
        "Continue FOLFIRINOX; consider FOLFOX if toxicity worsens.",
    ),
    "FOLFIRINOX",
)
check(
    "meds actual same-day administration is active",
    resolve_meds("", "", "Received cycle 1 gemcitabine/Abraxane today."),
    "gemcitabine, abraxane",
)
check(
    "meds conditional same-day plan is not active",
    resolve_meds("", "", "Will start gemcitabine/Abraxane today if labs permit."),
    "",
)
check(
    "meds supportive-only source does not populate current meds",
    resolve_meds("", "Continue Creon and ondansetron.", ""),
    "",
)
check(
    "meds literature regimen does not populate current meds",
    resolve_meds("", "A trial reported outcomes in patients receiving gemcitabine.", ""),
    "",
)
check(
    "meds historical multi-regimen timeline remains empty",
    resolve_meds(
        "",
        "06/05/18 C1D1 FOLFIRINOX. 11/29/18 C1D1 gemcitabine/Abraxane. Both courses were completed.",
        "Continue surveillance.",
    ),
    "",
)
check(
    "meds AP-current regimen wins over historical multi-regimen timeline",
    resolve_meds(
        "FOLFIRINOX, gemcitabine, abraxane",
        "06/05/18 C1D1 FOLFIRINOX. 11/29/18 C1D1 gemcitabine/Abraxane.",
        "The patient is currently receiving FOLFOX only.",
    ),
    "FOLFOX",
)

for row_id, expected in (
    (7, "gemcitabine, abraxane"),
    (8, "gemcitabine, abraxane"),
    (10, "FOLFOX"),
    (18, ""),
    (19, "folfirinox"),
):
    row = PV21[row_id]
    kp = row["keypoints"]
    got = resolve_meds(
        kp.get("Current_Medications", {}).get("current_meds", ""),
        row["note_text"],
        row["assessment_and_plan"],
        kp.get("Treatment_Changes", {}).get("recent_changes", ""),
    )
    check(f"meds matched-v21 pdac{row_id}", got, expected)

# affected temporal-window case remains prompt-owned: the conservative helper must not invent
# a replacement from ambiguous pre-regimen imaging.
check("response breast7 cross-regimen ambiguity unchanged by regex", sanitize_row(B[7]),
      B[7]["keypoints"]["Response_Assessment"]["response_assessment"])

# 30% clean controls plus an extra truly-untreated guard.
for cancer, row_id, row in (
    ("breast", 11, B[11]),
    ("pdac", 5, P[5]),
    ("pdac", 9, P[9]),
    ("pdac", 14, P[14]),
):
    original = row["keypoints"]["Response_Assessment"]["response_assessment"]
    check(f"response clean control {cancer}{row_id} unchanged", sanitize_row(row), original)

check(
    "response same-day future prescription does not fire",
    sanitize_response_assessment(
        "Not yet on treatment — no response to assess.",
        "",
        "We will start gemcitabine today after authorization.",
        current_meds="",
        recent_changes="Gemcitabine prescribed today",
    )[0],
    "Not yet on treatment — no response to assess.",
)
check(
    "response tumor-linked biliary change is retained",
    sanitize_response_assessment(
        "Biliary duct dilation worsened because the pancreatic tumor increased in size.",
        "",
        "The pancreatic tumor increased in size and caused worsening biliary duct dilation.",
        current_meds="gemcitabine",
    )[0],
    "Biliary duct dilation worsened because the pancreatic tumor increased in size.",
)
check(
    "response explicit formal PR is retained",
    sanitize_response_assessment(
        "Partial response on restaging CT.",
        "Restaging CT documents a partial response.",
        "Partial response after four cycles.",
        current_meds="gemcitabine",
    )[0],
    "Partial response on restaging CT.",
)
check(
    "response mass disappearance does not overwrite simultaneous progression",
    sanitize_response_assessment(
        "New liver metastasis indicates progression.",
        "",
        "New liver metastasis indicates progression. The pancreatic mass is no longer seen, likely related to treatment.",
        current_meds="gemcitabine",
    )[0],
    "New liver metastasis indicates progression. The pancreatic mass is no longer seen, likely related to treatment.",
)
historical_pr = sanitize_response_assessment(
    "This suggests a partial response to treatment.",
    "Last year the patient had a partial response on FOLFIRINOX.",
    "Current CT shows a slight decrease in the pancreatic mass.",
    current_meds="gemcitabine",
    findings="Current CT shows a slight decrease in the pancreatic mass.",
)[0]
check("response historical PR does not authorize current formal PR",
      "partial response" in historical_pr.lower(), False)
check("response historical PR downgrade preserves current slight decrease",
      "slight decrease" in historical_pr.lower(), True)
check(
    "response organ-qualified new metastasis preserves progression beside disappearance",
    sanitize_response_assessment(
        "A new liver metastasis developed, indicating progression.",
        "",
        "A new liver metastasis developed. The pancreatic mass is no longer seen, likely related to treatment.",
        current_meds="gemcitabine",
    )[0],
    "A new liver metastasis developed, indicating progression. The pancreatic mass is no longer seen, likely related to treatment.",
)
mixed_pr = sanitize_response_assessment(
    "Mixed response: partial response in the pancreatic mass, but a new liver metastasis developed.",
    "Last year the patient had a partial response on FOLFIRINOX.",
    "Current CT shows a slight decrease in the pancreatic mass but a new liver metastasis developed.",
    current_meds="gemcitabine",
    findings="Current CT shows a slight decrease in the pancreatic mass but a new liver metastasis developed.",
)[0]
check("response mixed PR downgrade retains new-organ progression",
      "new liver metastasis" in mixed_pr.lower(), True)
check("response mixed PR downgrade is local, not whole-field replacement",
      "mixed response" in mixed_pr.lower() and "slight decrease" in mixed_pr.lower()
      and "partial response" not in mixed_pr.lower(), True)
check(
    "response empty meds plus planned-only treatment is not current treatment",
    sanitize_response_assessment(
        "Currently on treatment; response is not yet available.",
        "",
        "Plan to start gemcitabine next week.",
        current_meds="",
        recent_changes="Gemcitabine planned for next week",
    )[0],
    "Not yet on treatment — no response to assess.",
)
check(
    "response empty meds plus held treatment uses direct tumor status",
    sanitize_response_assessment(
        "Currently receiving chemotherapy; response not available.",
        "",
        "Chemotherapy is currently on hold. CT shows stable disease.",
        current_meds="",
        recent_changes="Chemotherapy held for toxicity",
        findings="CT shows stable disease.",
    )[0],
    "CT shows stable disease.",
)
check(
    "response toxicity clause with explicit response is retained",
    sanitize_response_assessment(
        "Partial response with grade 2 neuropathy.",
        "",
        "CT documents a partial response with grade 2 neuropathy.",
        current_meds="FOLFIRINOX",
    )[0],
    "Partial response with grade 2 neuropathy.",
)
check(
    "response treatment-linked portal-vein resolution is retained",
    sanitize_response_assessment(
        "Portal vein encasement resolved after chemotherapy, consistent with treatment response.",
        "Status post diagnostic laparoscopy.",
        "Portal vein encasement resolved after chemotherapy, consistent with treatment response.",
        current_meds="gemcitabine",
    )[0],
    "Portal vein encasement resolved after chemotherapy, consistent with treatment response.",
)
check(
    "response received cycle one today counts as started",
    sanitize_response_assessment(
        "Not yet on treatment — no response to assess.",
        "",
        "Received cycle 1 today.",
        current_meds="gemcitabine",
        recent_changes="Received cycle 1 today",
    )[0],
    "Anticancer treatment started today; too early to assess its response.",
)
check(
    "response short drug name started today counts as started",
    sanitize_response_assessment(
        "Not yet on treatment — no response to assess.",
        "",
        "5-FU started today.",
        current_meds="5-FU",
        recent_changes="5-FU started today",
    )[0],
    "Anticancer treatment started today; too early to assess its response.",
)
check(
    "response explicit current stable control overrides fabricated progression",
    sanitize_response_assessment(
        "Imaging findings raise concern for possible recurrence or progression, but this is not confirmed.",
        "A historical liver lesion was once considered suspicious for metastasis.",
        "Metastatic pancreatic adenocarcinoma with continued good disease control on surveillance.",
        current_meds="",
    )[0],
    "Metastatic pancreatic adenocarcinoma with continued good disease control on surveillance.",
)
check(
    "response current progression prevents stable override",
    sanitize_response_assessment(
        "Current imaging shows progression.",
        "",
        "Previously stable disease, but the current scan shows progression.",
        current_meds="gemcitabine",
    )[0],
    "Current imaging shows progression.",
)

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
check("suspected-stage guard rejects biopsy for definitive Stage IV diagnosis",
      has_explicit_m1_evidence(
          "Bone lesions are suspicious; biopsy is planned for a definitive Stage IV diagnosis.",
          "breast",
      ), False)
check("suspected-stage guard rejects biopsy to confirm Stage IV",
      has_explicit_m1_evidence(
          "We will biopsy the iliac lesion to confirm Stage IV disease.",
          "breast",
      ), False)

check(
    "stage confirmed distant updates historical pTN stage",
    align_stage_with_confirmed_distant("pT2N2", "Yes, to liver", "pdac"),
    ("Originally pT2N2; now Stage IV (metastatic)",
     "confirmed distant disease updates historical/nonmetastatic stage"),
)
check(
    "stage suspected distant does not upgrade",
    align_stage_with_confirmed_distant("Stage III", "Not sure/Suspected, to bone", "breast"),
    ("Stage III", None),
)

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

# ---- Final broad-Metastasis sanitizer: synthetic evidence/certainty cases only ----
template_general = (
    "Yes — confirmed regional nodes; distant disease uncertain — suspicious organ lesion pending biopsy"
)
for label, note in (
    ("breast no nodal evidence", "Breast cancer follow-up. No axillary adenopathy on examination."),
    ("breast negative node biopsy", "Breast cancer. Axillary lymph-node biopsy was negative for carcinoma."),
    ("breast reactive nodes", "Breast cancer. Axillary lymph nodes showed benign reactive granulomatous change."),
):
    check(
        f"final Metastasis removes template regional claim: {label}",
        sanitize_general_metastasis("No", template_general, "Stage II", note, "", "breast")[0],
        "No",
    )

for label, note in (
    ("local invasion only", "Pancreatic adenocarcinoma directly invades the duodenum and encases the SMA."),
    ("nonspecific porta hepatis nodes", "Pancreatic adenocarcinoma with similar porta hepatis lymphadenopathy measuring 1.3 cm."),
    ("negative nodal staging", "Pancreatic adenocarcinoma. No pathologically enlarged lymph nodes."),
    ("inflammatory chest nodes", "Pancreatic cancer with mediastinal nodes favored reactive to pneumonia."),
    ("another primary node disease", "Pancreatic cancer follow-up. Prior prostate cancer had metastatic pelvic lymph nodes."),
    ("benign node pathology", "Pancreatic adenocarcinoma. Lymph-node biopsy showed benign tissue."),
    ("no nodes mentioned", "Pancreatic adenocarcinoma remains locally advanced and unresectable."),
):
    check(
        f"final Metastasis removes unsupported confirmed nodes: {label}",
        sanitize_general_metastasis("No", template_general, "Locally advanced", note, "", "pdac")[0],
        "No",
    )

check(
    "final Metastasis cN1 downgrades to clinically suspected",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "Stage IIB (cT1c cN1 cM0)",
        "Pancreatic adenocarcinoma with no distant metastasis.", "", "pdac",
    )[0],
    "Yes, clinically suspected regional lymph-node involvement (cN1); no distant metastasis",
)
check(
    "final Metastasis preserves historical 11/46 pathologic nodes",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "",
        "History of pancreatic adenocarcinoma resection in 2020; pancreatic surgical pathology showed 11/46 lymph nodes positive.",
        "", "pdac",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (11/46 nodes positive); no distant metastasis",
)
check(
    "final Metastasis preserves historical 2/29 pathologic nodes",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "",
        "Prior Whipple resection for pancreatic adenocarcinoma showed 2/29 lymph nodes positive.",
        "", "pdac",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (2/29 nodes positive); no distant metastasis",
)
check(
    "final Metastasis supports SLN-plus micrometastasis notation",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "",
        "Breast surgical pathology showed 1/2 SLN+ (micrometastasis).", "", "breast",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (1/2 nodes positive (micrometastasis)); no distant metastasis",
)
check(
    "final Metastasis supports sentinel nodes with micrometastasis wording",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "",
        "Breast surgical pathology found 1 of 2 sentinel nodes with micrometastasis.", "", "breast",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (1/2 nodes positive (micrometastasis)); no distant metastasis",
)
check(
    "final Metastasis supports current A/P positive-LN micrometastasis wording",
    sanitize_general_metastasis(
        "No", "Yes, confirmed ipsilateral axillary node", "Left: Stage III (T3N1); Right: Stage I (T1cN0)",
        "", "TAILORx does not apply to her given her positive LN, even as a micrometastasis.", "breast",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (micrometastatic regional node); no distant metastasis",
)
check(
    "final Metastasis restores omitted pathologic regional disease",
    sanitize_general_metastasis(
        "No", "No", "",
        "Pancreatic adenocarcinoma surgical pathology: ypT3N2.", "", "pdac",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (N2); no distant metastasis",
)
check(
    "final Metastasis labels imaging-explicit nodal metastases as radiographic",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "Locally advanced",
        "", "CT shows peripancreatic nodal metastases from pancreatic adenocarcinoma.", "pdac",
    )[0],
    "Yes, radiographically involved regional lymph nodes; no distant metastasis",
)
check(
    "final Metastasis uses preceding mesenteric context for terse nodal-metastases impression",
    sanitize_general_metastasis(
        "Yes, to liver", "Yes, confirmed regional nodes; confirmed liver metastases", "Stage IV",
        "", "CT abdomen shows increased size of multiple upper abdominal and mesenteric lymph nodes, some centrally necrotic. Impression: increased size and number of hepatic and nodal metastases.",
        "pdac",
    )[0],
    "Yes, to liver; radiographically involved regional lymph nodes",
)
check(
    "final Metastasis does not relabel distant nodal metastases as regional",
    sanitize_general_metastasis(
        "Yes, to cervical lymph nodes", "Yes, confirmed regional nodes", "Stage IV",
        "", "CT shows enlarging cervical lymph nodes and nodal metastases from pancreatic adenocarcinoma.",
        "pdac",
    )[0],
    "Yes, to cervical lymph nodes",
)
check(
    "final Metastasis does not borrow another primary's nodal metastases",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "Stage II",
        "", "Breast cancer follow-up. CT for ovarian cancer shows pelvic nodal metastases.", "breast",
    )[0],
    "No",
)
check(
    "final Metastasis Distant No retains biopsy-confirmed regional disease",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes; confirmed liver metastasis", "Stage III",
        "", "Core biopsy of the ipsilateral axillary lymph node was positive for metastatic breast carcinoma.",
        "breast",
    )[0],
    "Yes, pathologically confirmed regional lymph-node involvement (ipsilateral axillary lymph node); no distant metastasis",
)
check(
    "final Metastasis preserves suspected distant site certainty",
    sanitize_general_metastasis(
        "Not sure, suspicious liver lesion", "Yes, confirmed regional nodes; confirmed liver metastasis", "",
        "Pancreatic adenocarcinoma with an indeterminate liver lesion. No regional nodal disease is described.",
        "", "pdac",
    )[0],
    "Not sure, suspicious liver lesion",
)
check(
    "final Metastasis recognizes named pelvic bone sites as M1",
    sanitize_general_metastasis(
        "Yes, to left ilium, bilateral sacral ala", "No", "Stage IV",
        "Breast MRI shows lesions in the left iliac bone and bilateral sacral ala.", "", "breast",
    )[0],
    "Yes, to left ilium, bilateral sacral ala",
)
check(
    "final Metastasis preserves two-of-twenty-nine original pathology wording",
    sanitize_general_metastasis(
        "No", "Yes, confirmed regional nodes", "",
        "On 03/01/22, she underwent a distal pancreatectomy and splenectomy. She was found to have moderately differentiated pancreatic adenocarcinoma with negative margins and 2 of 29 lymph nodes were positive.",
        "", "pdac",
    )[0],
    "Yes, historical pathologically confirmed regional lymph-node involvement (2/29 nodes positive); no distant metastasis",
)
check(
    "final Metastasis preserves true locoregional chest-wall recurrence",
    sanitize_general_metastasis(
        "No", "Yes, locoregional chest-wall recurrence; no distant metastasis", "",
        "Biopsy confirms locoregional chest-wall recurrence of breast cancer.", "", "breast",
    )[0],
    "Yes, locoregional chest-wall recurrence; no distant metastasis",
)

# ---- POST-GENETIC-RESULTS-FINAL shared value sanitizer ----
def clean_genetic(value):
    return sanitize_genetic_testing_results(value)[0]


GENETIC_FALLBACK = "No genetic testing results in note."

check("genetic relative BRCA2 removed",
      clean_genetic("BRCA2 mutation found in brother."), GENETIC_FALLBACK)
check("genetic relative clause removed but patient result retained",
      clean_genetic("Patient's brother has a BRCA2 mutation; patient germline panel was negative."),
      "patient germline panel was negative.")
check("genetic pending STRATA removed",
      clean_genetic("STRATA pending."), GENETIC_FALLBACK)
check("genetic ordered germline panel removed",
      clean_genetic("Germline panel ordered."), GENETIC_FALLBACK)
check("genetic sent Invitae panel removed",
      clean_genetic("Invitae 126-gene panel sent."), GENETIC_FALLBACK)
check("genetic in-process FoundationOne removed",
      clean_genetic("FoundationOne testing is in process."), GENETIC_FALLBACK)
check("genetic completed MammaPrint retained",
      clean_genetic("MammaPrint high risk (-0.622)."), "MammaPrint high risk (-0.622).")
check("genetic completed Oncotype retained",
      clean_genetic("Oncotype DX recurrence score 23."), "Oncotype DX recurrence score 23.")
check("genetic completed germline negative retained",
      clean_genetic("Germline panel was negative for pathogenic variants."),
      "Germline panel was negative for pathogenic variants.")
check("genetic completed pathogenic panel retained",
      clean_genetic("Invitae panel identified a pathogenic ATM variant."),
      "Invitae panel identified a pathogenic ATM variant.")
check("genetic Foundation mutation retained",
      clean_genetic("FoundationOne detected KRAS G12D and TP53 mutations."),
      "FoundationOne detected KRAS G12D and TP53 mutations.")
check("genetic STRATA mutation retained",
      clean_genetic("STRATA showed a KRAS G12V mutation."),
      "STRATA showed a KRAS G12V mutation.")
check("genetic UCSF500 VUS retained",
      clean_genetic("UCSF500 found a VUS in RECQL4."),
      "UCSF500 found a VUS in RECQL4.")
check("genetic Tempus MSI TMB retained",
      clean_genetic("Tempus: MSS, TMB 5 muts/Mb."), "Tempus: MSS, TMB 5 muts/Mb.")
check("genetic Guardant ctDNA retained",
      clean_genetic("Guardant ctDNA detected PIK3CA H1047R."),
      "Guardant ctDNA detected PIK3CA H1047R.")
check("genetic MMR IHC retained",
      clean_genetic("MMR proteins intact by IHC."), "MMR proteins intact by IHC.")
check("genetic PD-L1 IHC retained",
      clean_genetic("PD-L1 CPS 10 by IHC."), "PD-L1 CPS 10 by IHC.")
check("genetic Foundation ERBB2 amplification retained",
      clean_genetic("FoundationOne: ERBB2 amplification."),
      "FoundationOne: ERBB2 amplification.")
check("genetic CA19-9 non-secretor retained",
      clean_genetic("CA 19-9 non-secretor (marker not useful for tracking)"),
      "CA 19-9 non-secretor (marker not useful for tracking)")
check("genetic routine breast receptors removed",
      clean_genetic("ER 95%, PR 80%, HER2 negative, Ki-67 20%."), GENETIC_FALLBACK)
check("genetic HER2 FISH removed",
      clean_genetic("HER2 negative by FISH."), GENETIC_FALLBACK)
check("genetic pure surgical pathology removed",
      clean_genetic("Grade 2 IDC with negative margins and 0/8 lymph nodes."), GENETIC_FALLBACK)
check("genetic mixed completed plus pending keeps completed clause",
      clean_genetic("MammaPrint high risk; Oncotype DX ordered."), "MammaPrint high risk.")
check("genetic mixed MMR IHC plus HER2 IHC keeps MMR only",
      clean_genetic("MMR intact by IHC; HER2 IHC 2+ with FISH non-amplified."),
      "MMR intact by IHC.")
check("genetic fallback normalization",
      clean_genetic("None"), GENETIC_FALLBACK)
check("genetic completed result after sent remains",
      clean_genetic("Germline panel was sent and resulted negative."),
      "Germline panel was sent and resulted negative.")

check("genetic matched breast3 removes sent/pending/pathology",
      clean_genetic(BV21[3]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
      GENETIC_FALLBACK)
check("genetic matched breast17 keeps BRCA and removes HER2 FISH",
      clean_genetic(BV21[17]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
      "BRCA test negative by report (Ambry - not sure whether panel or only BRCA).")
check("genetic matched breast19 keeps MammaPrint only",
      clean_genetic(BV21[19]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
      "Mammaprint - low risk IDC of the left breast.")
check("genetic matched pdac11 removes brother plus pending STRATA",
      clean_genetic(PV21[11]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
      GENETIC_FALLBACK)
check("genetic matched pdac13 preserves CA19-9 non-secretor",
      clean_genetic(PV21[13]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
      PV21[13]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"])

check(
    "genetic source recovery adds completed MMR to empty result",
    recover_completed_genetic_results(
        "No genetic testing results in note.",
        "Fine needle biopsy confirmed adenocarcinoma. MMR proteins all intact by IHC.",
    )[0],
    "MMR proteins intact by IHC (pMMR).",
)
check(
    "genetic source recovery recognizes four intact proteins",
    recover_completed_genetic_results(
        "No genetic testing results in note.",
        "MLH1 expression: Present. PMS2 expression: Present. MSH2 expression: Present. MSH6 expression: Present.",
    )[0],
    "MMR proteins intact by IHC (pMMR).",
)
check(
    "genetic source recovery ignores pending MMR",
    recover_completed_genetic_results(
        "No genetic testing results in note.",
        "MMR by IHC pending.",
    )[0],
    "No genetic testing results in note.",
)

check(
    "recurrent HR-only disease does not borrow historical PR HER2",
    sanitize_breast_recurrence_receptors(
        "ER+/PR-/HER2- grade 1 IDC (initial diagnosis); ER+/PR-/HER2- (current recurrent disease)",
        "Locally recurrent, unresectable, strongly hormone-receptor positive breast cancer.",
    )[0],
    "ER+/PR-/HER2- grade 1 IDC (initial diagnosis); HR+ (PR/HER2 not specified; current recurrent disease)",
)
check(
    "recurrent explicit receptor profile remains unchanged",
    sanitize_breast_recurrence_receptors(
        "Original ER+/PR-/HER2- IDC; current recurrent disease ER+/PR+/HER2-",
        "Current recurrent biopsy: ER positive, PR positive, HER2 negative.",
    )[0],
    "Original ER+/PR-/HER2- IDC; current recurrent disease ER+/PR+/HER2-",
)

for label, value in (
    ("breast7 MSH2 completed", BV21[7]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
    ("breast14 Myriad plus MammaPrint", BV21[14]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
    ("breast18 ATM plus MammaPrint", BV21[18]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
    ("pdac8 ATM plus MMR IHC", PV21[8]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
    ("pdac16 Foundation profile", PV21[16]["keypoints"]["Genetic_Testing_Results"]["genetic_testing_results"]),
):
    check(f"genetic clean control unchanged: {label}", clean_genetic(value), value)

# ---- POST-DISTMET-SITE-CERTAINTY: monotonic per-site evidence ceiling ----
def dm_fix(distant, general="", note="", ap="", cancer="pdac", stage=""):
    return sanitize_distant_metastasis_by_site(
        distant, general, stage, note, ap, cancer
    )[0]


for cancer, row_id, row, expected in (
    ("breast", 2, BV21[2], "No"),
    ("breast", 3, BV21[3], "Not sure/Suspected, to right adrenal nodule"),
    ("breast", 6, BV21[6], "Not sure/Suspected, to bone"),
    ("breast", 13, BV21[13], "No"),
    ("breast", 15, BV21[15], "Not sure/Suspected, to right cervical lymph nodes"),
    ("breast", 18, BV21[18], "No"),
    ("pdac", 4, PV21[4], "Not sure/Suspected, to liver"),
    ("pdac", 6, PV21[6], "Not sure/Suspected, to liver"),
    ("pdac", 12, PV21[12], "Yes, to peritoneum and omentum; Not sure/Suspected, to liver"),
    ("pdac", 20, PV21[20], "Yes, to peritoneum; Not sure/Suspected, to liver"),
):
    diagnosis = row["keypoints"]["Cancer_Diagnosis"]
    check(
        f"distant matched-v21 {cancer}{row_id}",
        dm_fix(
            diagnosis.get("Distant Metastasis", ""),
            diagnosis.get("Metastasis", ""),
            row["note_text"],
            row["assessment_and_plan"],
            cancer,
            diagnosis.get("Stage_of_Cancer", ""),
        ),
        expected,
    )

for cancer, row_id, row in (
    ("breast", 1, BV21[1]),
    ("breast", 9, BV21[9]),
    ("pdac", 3, PV21[3]),
    ("pdac", 9, PV21[9]),
    ("pdac", 14, PV21[14]),
):
    diagnosis = row["keypoints"]["Cancer_Diagnosis"]
    original = diagnosis.get("Distant Metastasis", "")
    check(
        f"distant matched-v21 clean control {cancer}{row_id}",
        dm_fix(
            original,
            diagnosis.get("Metastasis", ""),
            row["note_text"],
            row["assessment_and_plan"],
            cancer,
            diagnosis.get("Stage_of_Cancer", ""),
        ),
        original,
    )

check(
    "distant confirmed liver control unchanged",
    dm_fix("Yes, to liver", note="Liver biopsy confirmed metastatic pancreatic adenocarcinoma."),
    "Yes, to liver",
)
check(
    "distant suspected liver control unchanged",
    dm_fix("Not sure/Suspected, to liver", note="A liver lesion is suspicious for metastasis."),
    "Not sure/Suspected, to liver",
)
check(
    "distant peritoneum confirmed liver suspected stays mixed",
    dm_fix(
        "Yes, to peritoneum and liver",
        note="CT demonstrates peritoneal carcinomatosis. Liver lesions are suspicious for metastases.",
    ),
    "Yes, to peritoneum; Not sure/Suspected, to liver",
)
check(
    "distant confirmed liver does not upgrade suspicious lung",
    dm_fix(
        "Yes, to liver and lung",
        note="Liver biopsy confirmed metastatic adenocarcinoma. Pulmonary nodules are indeterminate.",
    ),
    "Yes, to liver; Not sure/Suspected, to lung",
)
check(
    "distant No never imports a general-field liver",
    dm_fix(
        "No", "Yes, to liver", note="No evidence of distant metastatic disease."
    ),
    "No",
)
check(
    "distant unsupported named liver is removed to unknown",
    dm_fix("Yes, to liver", note="Pancreatic adenocarcinoma without completed staging."),
    "Not sure",
)
check(
    "distant breast regional axillary node is removed",
    dm_fix(
        "Yes, to right axillary lymph nodes",
        note="Right axillary node biopsy was positive. No distant metastatic disease.",
        cancer="breast",
    ),
    "No",
)
check(
    "distant breast chest-wall recurrence is local",
    dm_fix(
        "Suspected, to parasternal chest wall",
        note="Biopsy confirms a locoregional parasternal chest-wall recurrence. No other sites of disease.",
        cancer="breast",
    ),
    "No",
)
check(
    "distant benign falx meningioma is cleared",
    dm_fix(
        "Not sure/Suspected, to falx cerebri",
        note="MRI shows a parafalcine lesion most likely a meningioma. No evidence of distant metastasis.",
        cancer="breast",
    ),
    "No",
)
check(
    "distant other-primary lung disease is not borrowed",
    dm_fix(
        "Yes, to lung",
        note="History of renal cell carcinoma metastatic to lung. Current pancreatic cancer staging is incomplete.",
    ),
    "Not sure",
)
check(
    "distant confirmed abdominal-wall control unchanged",
    dm_fix(
        "Yes, to abdominal wall",
        note="Abdominal wall biopsy confirmed metastatic pancreatic adenocarcinoma.",
    ),
    "Yes, to abdominal wall",
)
check(
    "distant omental caking remains confirmed",
    dm_fix("Yes, to omentum", note="CT demonstrates diffuse omental caking."),
    "Yes, to omentum",
)
check(
    "distant generic Yes borrows only existing suspicious bone site",
    dm_fix(
        "Yes", "Distant disease uncertain — suspicious bone lesions pending biopsy",
        note="PET/CT shows osseous lesions suspicious for metastatic disease.",
    ),
    "Not sure/Suspected, to bone",
)
check(
    "distant generic Not sure cannot upgrade general confirmed liver",
    dm_fix(
        "Not sure", "Yes, to liver",
        note="A liver lesion is suspicious for metastasis and biopsy is planned.",
    ),
    "Not sure/Suspected, to liver",
)
check(
    "distant cystic liver metastasis is not treated as benign cyst",
    dm_fix("Yes, to liver", note="CT shows known cystic liver metastases."),
    "Yes, to liver",
)
check(
    "distant neutral adrenal description remains unchanged",
    dm_fix("Yes, to adrenal gland", note="An adrenal lesion is described; characterization is unavailable."),
    "Yes, to adrenal gland",
)
check(
    "distant explicit indeterminate adrenal survives generic no-definite statement",
    dm_fix(
        "Not sure/Suspected, to adrenal gland",
        note="There is an indeterminate adrenal nodule. No definite distant metastatic disease.",
        cancer="breast",
    ),
    "Not sure/Suspected, to adrenal gland",
)
check(
    "distant generic established M1 remains generic Yes",
    dm_fix("Yes", ap="Metastatic adenocarcinoma of the pancreas remains under treatment."),
    "Yes",
)
check(
    "distant explicit M0 control remains No",
    dm_fix("No", note="Clinical staging is cT2N1M0."),
    "No",
)
check(
    "distant unsupported No remains unchanged under conservative policy",
    dm_fix("No", note="Breast MRI shows the primary tumor and regional axillary nodes.", cancer="breast"),
    "No",
)
check(
    "distant historical biopsy-proven lung site retained as historical",
    dm_fix(
        "Yes, to lung",
        note="History of lung biopsy positive for metastatic pancreatic adenocarcinoma.",
    ),
    "Yes, historically confirmed to lung",
)
check(
    "distant historical confirmed value remains scalar-confirmed",
    is_confirmed_distant_value("Yes, historically confirmed to lung", "pdac"),
    True,
)
breast2_distant = dm_fix(
    BV21[2]["keypoints"]["Cancer_Diagnosis"]["Distant Metastasis"],
    BV21[2]["keypoints"]["Cancer_Diagnosis"]["Metastasis"],
    BV21[2]["note_text"],
    BV21[2]["assessment_and_plan"],
    "breast",
)
check(
    "distant breast2 cleanup preserves locoregional recurrence downstream",
    sanitize_general_metastasis(
        breast2_distant,
        BV21[2]["keypoints"]["Cancer_Diagnosis"]["Metastasis"],
        BV21[2]["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"],
        BV21[2]["note_text"],
        BV21[2]["assessment_and_plan"],
        "breast",
    )[0],
    "Yes, locoregional recurrence; no distant metastasis",
)
pdac4_distant = dm_fix(
    PV21[4]["keypoints"]["Cancer_Diagnosis"]["Distant Metastasis"],
    PV21[4]["keypoints"]["Cancer_Diagnosis"]["Metastasis"],
    PV21[4]["note_text"],
    PV21[4]["assessment_and_plan"],
    "pdac",
)
check(
    "distant pdac4 downgrade retains physician-explicit metastatic stage",
    normalize_stage_iv(
        PV21[4]["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"],
        pdac4_distant,
        PV21[4]["assessment_and_plan"],
        "pdac",
    ),
    (PV21[4]["keypoints"]["Cancer_Diagnosis"]["Stage_of_Cancer"], None),
)
mixed_distant = dm_fix(
    "Yes, to peritoneum and liver",
    note="CT demonstrates peritoneal carcinomatosis. Liver lesions are suspicious for metastases.",
)
check(
    "distant mixed certainty survives final general sanitizer",
    sanitize_general_metastasis(
        mixed_distant,
        "Yes, confirmed liver and peritoneal metastases",
        "Stage IV",
        "CT demonstrates peritoneal carcinomatosis. Liver lesions are suspicious for metastases.",
        "",
        "pdac",
    )[0],
    mixed_distant,
)

# ---- bug10 POST-DISTMET-SITES: REMOVED (fired on negated "No osseous lesions" → hallucination). ----

if __name__ == "__main__":
    print(f"\n==== {sum(results)}/{len(results)} PASS ====")
    sys.exit(0 if all(results) else 1)
