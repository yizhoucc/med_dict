#!/usr/bin/env python3
"""Reproducible figure for the progress report.
Recomputes the PL-vs-BL per-field outcome matrix on the 40 held-out samples
(_audit_v6/verdicts.json for differing items; identical values -> Tie; both empty -> N/A
excluded) and renders a single 2-panel report figure:
  (a) per-question PL/Tie/BL outcome, sorted by PL advantage
  (b) overall outcome among the questions where the two systems differ
Outputs figs/report_fig.png (+ .pdf) and prints the exact counts used in the report text.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from build_scoring_html import parse_pl, parse_bl, get_val, QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
PL_C, TIE_C, BL_C, INK = "#2e7d4f", "#d9dee4", "#c0392b", "#222426"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#9aa0a6",
                     "axes.linewidth": .8, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "legend.frameon": False})

# ---- recompute matrix (same convention as make_figs.py) ----
plb = parse_pl(os.path.join(HERE, "pipeline_breast_FINAL.txt"))
plp = parse_pl(os.path.join(HERE, "pipeline_pdac_FINAL.txt"))
blb = parse_bl(os.path.join(HERE, "baseline_extract_breast_json.txt"))
blp = parse_bl(os.path.join(HERE, "baseline_extract_pdac_json.txt"))
V = json.load(open(os.path.join(HERE, "_audit_v6", "verdicts.json")))
vd = {}
for c in ("b", "p"):
    for v in V[c]:
        vd[(c, v["row"], v["field"])] = v["verdict"]

FIELDS = [q[0] for q in QUESTIONS]
LABEL = {q[0]: q[1] for q in QUESTIONS}
SEC = {q[0]: (q[3], q[4]) for q in QUESTIONS}
# readable labels for the report
RLAB = {"current_meds": "Current medications", "stage": "Cancer stage",
        "distant_met": "Distant metastasis", "metastasis": "Metastasis (regional)",
        "response": "Treatment response", "type_receptor": "Type / receptors",
        "genetic_results": "Molecular / genetic results", "genetic_plan": "Genetic testing plan",
        "supportive_meds": "Supportive meds", "procedure_plan": "Procedure plan",
        "imaging_plan": "Imaging plan", "lab_plan": "Lab plan",
        "medication_plan": "Medication plan", "recent_changes": "Recent tx changes",
        "lab_summary": "Lab summary", "findings": "Clinical findings"}

def blank(): return {"PL": 0, "TIE": 0, "BL": 0, "NA": 0}
stats = {f: blank() for f in FIELDS}

def run(c, pl, bl):
    for row in sorted(pl):
        p = pl[row]; b = bl.get(row, {})
        for f in FIELDS:
            if c == "p" and f == "type_receptor":
                continue  # breast-only
            pv = get_val(p["kp"], *SEC[f]).strip(); bv = get_val(b, *SEC[f]).strip()
            k3 = (c, row, f)
            if k3 in vd: stats[f][vd[k3]] += 1
            elif not pv and not bv: stats[f]["NA"] += 1
            else: stats[f]["TIE"] += 1
run("b", plb, blb); run("p", plp, blp)

GPL = sum(stats[f]["PL"] for f in FIELDS); GTIE = sum(stats[f]["TIE"] for f in FIELDS)
GBL = sum(stats[f]["BL"] for f in FIELDS); N = GPL + GTIE + GBL
dpl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "PL")
dtie = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "TIE")
dbl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "BL")
nd = dpl + dtie + dbl
print(f"scored={N}  PL={GPL} Tie={GTIE} BL={GBL}")
print(f"differing-only n={nd}  PL={dpl} Tie={dtie} BL={dbl}  (PL share={dpl/nd*100:.0f}%)")
print("current_meds:", stats["current_meds"])

# ---- figure ----
order = sorted(FIELDS, key=lambda f: (stats[f]["PL"] - stats[f]["BL"], stats[f]["PL"]))
fig = plt.figure(figsize=(12, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1], wspace=0.32)

# (a) per-question
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(order))
pls = np.array([stats[f]["PL"] for f in order]); tie = np.array([stats[f]["TIE"] for f in order])
bls = np.array([stats[f]["BL"] for f in order])
ax.barh(y, pls, color=PL_C, height=.74, label="PL better")
ax.barh(y, tie, left=pls, color=TIE_C, height=.74, label="Tie / same")
ax.barh(y, bls, left=pls + tie, color=BL_C, height=.74, label="BL better")
for i in range(len(order)):
    if pls[i]: ax.text(pls[i] - .4, i, str(pls[i]), ha="right", va="center", color="white", fontsize=8.5, fontweight="bold")
    if bls[i]: ax.text(pls[i] + tie[i] + bls[i] + .3, i, str(bls[i]), ha="left", va="center", color=BL_C, fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([RLAB[f] for f in order], fontsize=9.5)
ax.set_xlabel("number of samples"); ax.set_xlim(0, 40)
ax.set_title("(a) PL vs BL by field (sorted by PL advantage)", fontsize=11.5, fontweight="bold", color=INK)
ax.legend(loc="lower right", fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.xaxis.grid(True, color="#eef0f2", lw=.8); ax.set_axisbelow(True)

# (b) overall among differing
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar([0], [dpl], color=PL_C, width=.6, label="PL better")
ax2.bar([0], [dtie], bottom=[dpl], color=TIE_C, width=.6, label="Tie")
ax2.bar([0], [dbl], bottom=[dpl + dtie], color=BL_C, width=.6, label="BL better")
ax2.text(0, dpl/2, f"PL better\n{dpl}  ({dpl/nd*100:.0f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
ax2.text(0, dpl + dtie/2, f"Tie {dtie}", ha="center", va="center", color=INK, fontweight="bold", fontsize=9)
ax2.text(0, dpl + dtie + dbl/2 + 3, f"BL better {dbl} ({dbl/nd*100:.0f}%)", ha="center", va="bottom", color=BL_C, fontweight="bold", fontsize=9)
ax2.set_xticks([]); ax2.set_xlim(-.6, .6); ax2.set_ylabel("field-comparisons where systems differ")
ax2.set_title(f"(b) Where they differ (n={nd})", fontsize=11.5, fontweight="bold", color=INK)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.suptitle("Pipeline (PL) vs single-prompt baseline (BL) on 40 held-out oncology notes — same base model, harness as the only variable",
             fontsize=12.5, fontweight="bold", y=1.01)
fig.savefig(os.path.join(FIG, "report_fig.png"))
fig.savefig(os.path.join(FIG, "report_fig.pdf"))
print("wrote figs/report_fig.png (+ .pdf)")
