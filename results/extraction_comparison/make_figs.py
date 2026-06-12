#!/usr/bin/env python3
"""Figures for PL vs BL extraction comparison. Recomputes the full 19-question x
40-sample matrix (verdicts.json for differing items; identical values -> Tie; both
empty -> N/A excluded), then renders a set of charts."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_scoring_html import parse_pl, parse_bl, get_val, QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

PL_C, TIE_C, BL_C = "#1b7a36", "#c2c8cf", "#c0392b"
plt.rcParams.update({"font.size": 11})

# ---- recompute matrix ----
plb = parse_pl(os.path.join(HERE, "pipeline_breast_FINAL.txt"))
plp = parse_pl(os.path.join(HERE, "pipeline_pdac_FINAL.txt"))
blb = parse_bl(os.path.join(HERE, "baseline_extract_breast_json.txt"))
blp = parse_bl(os.path.join(HERE, "baseline_extract_pdac_json.txt"))
V = json.load(open(os.path.join(HERE, "_audit_v5", "verdicts.json")))
vd = {}
for c in ("b", "p"):
    for v in V[c]:
        vd[(c, v["row"], v["field"])] = v["verdict"]

FIELDS = [q[0] for q in QUESTIONS]
LABEL = {q[0]: q[1] for q in QUESTIONS}
TIER = {q[0]: q[2] for q in QUESTIONS}
SEC = {q[0]: (q[3], q[4]) for q in QUESTIONS}

# stats[field][cancer] = dict counts
def blank():
    return {"PL": 0, "TIE": 0, "BL": 0, "NA": 0}
stats = {f: {"b": blank(), "p": blank()} for f in FIELDS}

def run(c, pl, bl):
    for row in sorted(pl):
        p = pl[row]; b = bl.get(row, {})
        for f in FIELDS:
            sec, key = SEC[f]
            pv = get_val(p["kp"], sec, key).strip()
            bv = get_val(b, sec, key).strip()
            k3 = (c, row, f)
            if k3 in vd:
                stats[f][c][vd[k3]] += 1
            elif not pv and not bv:
                stats[f][c]["NA"] += 1
            else:
                stats[f][c]["TIE"] += 1  # identical / no-diff
run("b", plb, blb); run("p", plp, blp)

def tot(f):
    d = blank()
    for c in ("b", "p"):
        for k in d: d[k] += stats[f][c][k]
    return d

GPL = sum(tot(f)["PL"] for f in FIELDS)
GTIE = sum(tot(f)["TIE"] for f in FIELDS)
GBL = sum(tot(f)["BL"] for f in FIELDS)
GNA = sum(tot(f)["NA"] for f in FIELDS)
N_SCORED = GPL + GTIE + GBL
print(f"scored={N_SCORED} (PL {GPL} / Tie {GTIE} / BL {GBL}), N/A {GNA}")

# ============ FIG 1: overall outcome (two framings, both on the 19-question universe) ============
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

def stacked(ax, pl, tie, bl, total, title):
    ax.barh([0], [pl], color=PL_C, label="PL better")
    ax.barh([0], [tie], left=[pl], color=TIE_C, label="Tie")
    ax.barh([0], [bl], left=[pl + tie], color=BL_C, label="BL better")
    ax.text(pl/2, 0, f"{pl}\n{pl/total*100:.0f}%", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.text(pl+tie/2, 0, f"{tie}\n{tie/total*100:.0f}%", ha="center", va="center",
            color="white" if tie/total > .08 else "#333", fontweight="bold", fontsize=10)
    # BL label outside the bar (it is small), in BL color
    ax.text(pl+tie+bl + total*0.012, 0, f"BL {bl}  ({bl/total*100:.0f}%)",
            ha="left", va="center", color=BL_C, fontweight="bold", fontsize=9.5)
    ax.set_yticks([]); ax.set_xlim(0, total*1.16)
    ax.set_title(title, fontsize=11)

# (a) all 19-question scores (Tie includes identical/no-diff)
stacked(axes[0], GPL, GTIE, GBL, N_SCORED, f"All scored questions (n={N_SCORED})")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
# (b) only the 19-question items where PL and BL actually differ
dpl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "PL")
dtie = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "TIE")
dbl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "BL")
nd = dpl + dtie + dbl
stacked(axes[1], dpl, dtie, dbl, nd, f"Only questions where PL≠BL (n={nd})")
fig.suptitle("PL vs BL — overall outcome", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG, "fig1_overall.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: per-question stacked bars (sorted by PL wins) ============
order = sorted(FIELDS, key=lambda f: (tot(f)["PL"] - tot(f)["BL"], tot(f)["PL"]))
labels = [LABEL[f] for f in order]
pls = np.array([tot(f)["PL"] for f in order])
ties = np.array([tot(f)["TIE"] for f in order])
bls = np.array([tot(f)["BL"] for f in order])
nas = np.array([tot(f)["NA"] for f in order])
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(y, pls, color=PL_C, label="PL better")
ax.barh(y, ties, left=pls, color=TIE_C, label="Tie / same")
ax.barh(y, bls, left=pls + ties, color=BL_C, label="BL better")
for i, f in enumerate(order):
    if pls[i]: ax.text(pls[i]-0.3, i, str(pls[i]), ha="right", va="center", color="white", fontsize=8.5, fontweight="bold")
    if bls[i]: ax.text(pls[i]+ties[i]+bls[i]+0.2, i, str(bls[i]), ha="left", va="center", color=BL_C, fontsize=8.5, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("number of samples (out of 40)")
ax.set_title("PL vs BL by question  (sorted by PL advantage)", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.set_xlim(0, 40)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_per_question.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ============ FIG 3: net advantage diverging (PL wins - BL wins) ============
order2 = sorted(FIELDS, key=lambda f: tot(f)["PL"] - tot(f)["BL"])
net = [tot(f)["PL"] - tot(f)["BL"] for f in order2]
labels2 = [LABEL[f] for f in order2]
cols = [PL_C if n > 0 else (BL_C if n < 0 else TIE_C) for n in net]
fig, ax = plt.subplots(figsize=(9.5, 8))
yy = np.arange(len(order2))
ax.barh(yy, net, color=cols)
for i, n in enumerate(net):
    ax.text(n + (0.4 if n >= 0 else -0.4), i, f"{n:+d}", ha="left" if n >= 0 else "right",
            va="center", fontsize=9, fontweight="bold", color=cols[i])
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels(labels2, fontsize=9.5)
ax.set_xlabel("net PL advantage  (# PL-better  −  # BL-better)")
ax.set_title("Net PL advantage by question", fontsize=12, fontweight="bold")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_net.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ============ FIG 4: by depth tier ============
tiers = ["deep", "med", "basic"]
TNAME = {"deep": "Deep\n(needs medical knowledge)", "med": "Medium", "basic": "Basic\n(layperson)"}
tp = {t: blank() for t in tiers}
for f in FIELDS:
    d = tot(f)
    for k in tp[TIER[f]]: tp[TIER[f]][k] += d[k]
fig, ax = plt.subplots(figsize=(8.5, 4))
x = np.arange(len(tiers))
sc = [tp[t]["PL"] + tp[t]["TIE"] + tp[t]["BL"] for t in tiers]
plf = [tp[t]["PL"] for t in tiers]; tif = [tp[t]["TIE"] for t in tiers]; blf = [tp[t]["BL"] for t in tiers]
plp_ = [a/b*100 for a, b in zip(plf, sc)]
tip_ = [a/b*100 for a, b in zip(tif, sc)]
blp_ = [a/b*100 for a, b in zip(blf, sc)]
ax.bar(x, plp_, color=PL_C, label="PL better")
ax.bar(x, tip_, bottom=plp_, color=TIE_C, label="Tie / same")
ax.bar(x, blp_, bottom=[a+b for a, b in zip(plp_, tip_)], color=BL_C, label="BL better")
for i, t in enumerate(tiers):
    ax.text(i, plp_[i]/2, f"{plf[i]}\n({plp_[i]:.0f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    if blf[i]: ax.text(i, plp_[i]+tip_[i]+blp_[i]/2, f"{blf[i]}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([TNAME[t] for t in tiers], fontsize=9.5)
ax.set_ylabel("% of scored questions"); ax.set_ylim(0, 100)
ax.set_title("Outcome by question depth tier", fontsize=12, fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_by_tier.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ============ FIG 5: breast vs PDAC ============
fig, ax = plt.subplots(figsize=(8.5, 4))
groups = [("Breast", "b"), ("PDAC", "p")]
x = np.arange(len(groups))
def agg(c):
    d = blank()
    for f in FIELDS:
        for k in d: d[k] += stats[f][c][k]
    return d
data = {g[0]: agg(g[1]) for g in groups}
for gi, (gn, _) in enumerate(groups):
    d = data[gn]; sc = d["PL"]+d["TIE"]+d["BL"]
    pl_, ti_, bl_ = d["PL"]/sc*100, d["TIE"]/sc*100, d["BL"]/sc*100
    ax.bar(gi, pl_, color=PL_C); ax.bar(gi, ti_, bottom=pl_, color=TIE_C)
    ax.bar(gi, bl_, bottom=pl_+ti_, color=BL_C)
    ax.text(gi, pl_/2, f"PL {d['PL']}\n{pl_:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    if d["BL"]: ax.text(gi, pl_+ti_+bl_/2, f"BL {d['BL']}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("% of scored questions"); ax.set_ylim(0, 100)
ax.set_title("Outcome by cancer type", fontsize=12, fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PL_C, label="PL better"), Patch(color=TIE_C, label="Tie / same"), Patch(color=BL_C, label="BL better")],
          loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=9)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_by_cancer.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

print("wrote 5 figures to", FIG)
