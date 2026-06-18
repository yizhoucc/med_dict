#!/usr/bin/env python3
"""Paper-style figures for PL vs BL extraction comparison.
Recomputes the full 19-question x 40-sample matrix (verdicts.json for differing items;
identical values -> Tie; both empty -> N/A excluded), then renders publication-quality
charts + a 40x19 paired heatmap."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
from build_scoring_html import parse_pl, parse_bl, get_val, QUESTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

# ---------- paper style ----------
PL_C, TIE_C, BL_C, NA_C = "#2e7d4f", "#e3e7ec", "#c0392b", "#ffffff"
INK = "#222426"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#9aa0a6",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": 11,
    "text.color": INK,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
})

def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=3, width=0.8)

# ---------- recompute matrix ----------
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
TIER = {q[0]: q[2] for q in QUESTIONS}
SEC = {q[0]: (q[3], q[4]) for q in QUESTIONS}

def blank(): return {"PL": 0, "TIE": 0, "BL": 0, "NA": 0}
stats = {f: {"b": blank(), "p": blank()} for f in FIELDS}
# cell[(c,row)][field] = code  PL=2 TIE=1 BL=0 NA=nan
cell = {}

def run(c, pl, bl):
    for row in sorted(pl):
        p = pl[row]; b = bl.get(row, {})
        cell[(c, row)] = {}
        for f in FIELDS:
            if c == "p" and f == "type_receptor":
                continue  # Q6 type/receptor is breast-only (PDAC has no ER/PR/HER2)
            sec, key = SEC[f]
            pv = get_val(p["kp"], sec, key).strip()
            bv = get_val(b, sec, key).strip()
            k3 = (c, row, f)
            if k3 in vd:
                v = vd[k3]; stats[f][c][v] += 1
                cell[(c, row)][f] = {"PL": 2, "TIE": 1, "BL": 0}[v]
            elif not pv and not bv:
                stats[f][c]["NA"] += 1; cell[(c, row)][f] = np.nan
            else:
                stats[f][c]["TIE"] += 1; cell[(c, row)][f] = 1
run("b", plb, blb); run("p", plp, blp)

def tot(f):
    d = blank()
    for c in ("b", "p"):
        for k in d: d[k] += stats[f][c][k]
    return d

GPL = sum(tot(f)["PL"] for f in FIELDS); GTIE = sum(tot(f)["TIE"] for f in FIELDS)
GBL = sum(tot(f)["BL"] for f in FIELDS); GNA = sum(tot(f)["NA"] for f in FIELDS)
N_SCORED = GPL + GTIE + GBL
print(f"scored={N_SCORED} (PL {GPL} / Tie {GTIE} / BL {GBL}), N/A {GNA}")

# order fields by PL advantage (asc, for horizontal bars bottom->top)
order_adv = sorted(FIELDS, key=lambda f: (tot(f)["PL"] - tot(f)["BL"], tot(f)["PL"]))

# ===================================================================
# FIG 1 — overall outcome (two framings)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))
def stacked(ax, pl, tie, bl, total, title):
    ax.barh([0], [pl], color=PL_C)
    ax.barh([0], [tie], left=[pl], color=TIE_C)
    ax.barh([0], [bl], left=[pl + tie], color=BL_C)
    ax.text(pl/2, 0, f"{pl}\n{pl/total*100:.0f}%", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)
    ax.text(pl+tie/2, 0, f"{tie}\n{tie/total*100:.0f}%", ha="center", va="center",
            color=INK if tie/total > .08 else "#333", fontweight="bold", fontsize=11)
    ax.text(pl+tie+bl + total*0.015, 0, f"BL {bl} ({bl/total*100:.0f}%)",
            ha="left", va="center", color=BL_C, fontweight="bold", fontsize=10)
    ax.set_yticks([]); ax.set_xlim(0, total*1.18); ax.set_ylim(-0.6, 0.6)
    ax.set_title(title, fontsize=11.5)
    despine(ax, keep=("bottom",))
stacked(axes[0], GPL, GTIE, GBL, N_SCORED, f"All scored questions  (n = {N_SCORED})")
dpl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "PL")
dtie = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "TIE")
dbl = sum(1 for (c, r, f), v in vd.items() if f in FIELDS and v == "BL")
nd = dpl + dtie + dbl
stacked(axes[1], dpl, dtie, dbl, nd, f"Only questions where PL ≠ BL  (n = {nd})")
handles = [Patch(color=PL_C, label="PL better"), Patch(color=TIE_C, label="Tie / same"),
           Patch(color=BL_C, label="BL better")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("PL vs BL — overall outcome", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_overall.png")); plt.close(fig)

# ===================================================================
# FIG 2 — per-question stacked bars
# ===================================================================
labels = [LABEL[f] for f in order_adv]
pls = np.array([tot(f)["PL"] for f in order_adv]); ties = np.array([tot(f)["TIE"] for f in order_adv])
bls = np.array([tot(f)["BL"] for f in order_adv])
y = np.arange(len(order_adv))
fig, ax = plt.subplots(figsize=(10.5, 7.5))
ax.barh(y, pls, color=PL_C, label="PL better", height=0.74)
ax.barh(y, ties, left=pls, color=TIE_C, label="Tie / same", height=0.74)
ax.barh(y, bls, left=pls + ties, color=BL_C, label="BL better", height=0.74)
for i in range(len(order_adv)):
    if pls[i]: ax.text(pls[i]-0.4, i, str(pls[i]), ha="right", va="center", color="white", fontsize=9, fontweight="bold")
    if bls[i]: ax.text(40.3, i, str(bls[i]), ha="left", va="center", color=BL_C, fontsize=9, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels)
# colour-tag tier on y labels
tcol = {"deep": PL_C, "med": "#b8860b", "basic": "#8a929b"}
for tick, f in zip(ax.get_yticklabels(), order_adv):
    tick.set_color(INK)
ax.set_xlabel("number of samples (out of 40)")
ax.set_title("PL vs BL by question  (sorted by PL advantage)")
ax.set_xlim(0, 40); ax.margins(y=0.01)
ax.legend(loc="lower right")
despine(ax)
ax.xaxis.grid(True, color="#eef0f2", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_per_question.png")); plt.close(fig)

# ===================================================================
# FIG 3 — net advantage diverging
# ===================================================================
net = [tot(f)["PL"] - tot(f)["BL"] for f in order_adv]
cols = [PL_C if n > 0 else (BL_C if n < 0 else "#9aa0a6") for n in net]
fig, ax = plt.subplots(figsize=(8.5, 7.5))
ax.barh(y, net, color=cols, height=0.72)
for i, n in enumerate(net):
    ax.text(n + (0.45 if n >= 0 else -0.45), i, f"{n:+d}", ha="left" if n >= 0 else "right",
            va="center", fontsize=9.5, fontweight="bold", color=cols[i])
ax.axvline(0, color="#555", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("net PL advantage   (# PL-better  −  # BL-better)")
ax.set_title("Net PL advantage by question")
ax.set_xlim(-3, 38)
despine(ax, keep=("bottom",)); ax.tick_params(left=False)
ax.xaxis.grid(True, color="#eef0f2", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_net.png")); plt.close(fig)

# ===================================================================
# FIG 4 — by depth tier
# ===================================================================
tiers = ["deep", "med", "basic"]
TNAME = {"deep": "Deep\n(needs medical knowledge)", "med": "Medium", "basic": "Basic\n(layperson)"}
tp = {t: blank() for t in tiers}
for f in FIELDS:
    d = tot(f)
    for k in tp[TIER[f]]: tp[TIER[f]][k] += d[k]
fig, ax = plt.subplots(figsize=(7.8, 4.2))
x = np.arange(len(tiers))
sc = [tp[t]["PL"]+tp[t]["TIE"]+tp[t]["BL"] for t in tiers]
plp_ = [tp[t]["PL"]/s*100 for t, s in zip(tiers, sc)]
tip_ = [tp[t]["TIE"]/s*100 for t, s in zip(tiers, sc)]
blp_ = [tp[t]["BL"]/s*100 for t, s in zip(tiers, sc)]
ax.bar(x, plp_, color=PL_C, width=0.62, label="PL better")
ax.bar(x, tip_, bottom=plp_, color=TIE_C, width=0.62, label="Tie / same")
ax.bar(x, blp_, bottom=[a+b for a, b in zip(plp_, tip_)], color=BL_C, width=0.62, label="BL better")
for i, t in enumerate(tiers):
    ax.text(i, plp_[i]/2, f"PL {tp[t]['PL']}\n{plp_[i]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    if tp[t]["BL"]:
        ax.text(i + 0.34, 100 - blp_[i]/2, f"BL {tp[t]['BL']}", ha="left", va="center", color=BL_C, fontweight="bold", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([TNAME[t] for t in tiers])
ax.set_ylabel("% of scored questions"); ax.set_ylim(0, 100)
ax.set_title("Outcome by question depth tier")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3)
despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_by_tier.png")); plt.close(fig)

# ===================================================================
# FIG 5 — breast vs PDAC
# ===================================================================
fig, ax = plt.subplots(figsize=(7.8, 4.2))
groups = [("Breast", "b"), ("PDAC", "p")]
def agg(c):
    d = blank()
    for f in FIELDS:
        for k in d: d[k] += stats[f][c][k]
    return d
for gi, (gn, c) in enumerate(groups):
    d = agg(c); s = d["PL"]+d["TIE"]+d["BL"]
    pl_, ti_, bl_ = d["PL"]/s*100, d["TIE"]/s*100, d["BL"]/s*100
    ax.bar(gi, pl_, color=PL_C, width=0.5); ax.bar(gi, ti_, bottom=pl_, color=TIE_C, width=0.5)
    ax.bar(gi, bl_, bottom=pl_+ti_, color=BL_C, width=0.5)
    ax.text(gi, pl_/2, f"PL {d['PL']}\n{pl_:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    if d["BL"]: ax.text(gi+0.28, 100-bl_/2, f"BL {d['BL']}", ha="left", va="center", color=BL_C, fontweight="bold", fontsize=9)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("% of scored questions"); ax.set_ylim(0, 100); ax.set_xlim(-0.6, 1.8)
ax.set_title("Outcome by cancer type")
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3)
despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_by_cancer.png")); plt.close(fig)

# ===================================================================
# FIG 6 — 40 x 19 paired heatmap
# ===================================================================
col_order = order_adv[::-1]  # strongest PL advantage on the left
rows = [("b", r) for r in sorted(plb)] + [("p", r) for r in sorted(plp)]
M = np.full((len(rows), len(col_order)), np.nan)
for ri, rk in enumerate(rows):
    for ci, f in enumerate(col_order):
        M[ri, ci] = cell[rk].get(f, np.nan)
cmap = ListedColormap([BL_C, TIE_C, PL_C])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
cmap.set_bad(NA_C)
fig, ax = plt.subplots(figsize=(10.5, 11))
ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
# gridlines between cells
ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="minor", length=0)
# heavy divider between breast and pdac
ax.axhline(len(plb)-0.5, color=INK, lw=1.6)
ax.set_xticks(range(len(col_order)))
ax.set_xticklabels([LABEL[f] for f in col_order], rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{c.upper()}{r}" for c, r in rows], fontsize=7.5)
ax.set_title("Paired outcome per sample × question\n(breast B1–20 top, PDAC P1–20 bottom)", fontsize=12.5)
for s in ax.spines.values(): s.set_visible(False)
leg = [Patch(facecolor=PL_C, label="PL better"), Patch(facecolor=TIE_C, label="Tie / same"),
       Patch(facecolor=BL_C, label="BL better"),
       Patch(facecolor=NA_C, edgecolor="#bbb", label="N/A (both empty)")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_heatmap.png")); plt.close(fig)

print("wrote 6 figures to", FIG)
