# -*- coding: utf-8 -*-
"""Nature-portfolio-standard figure set for the AM-DGCN paper.
Generates Figures 1-5 (PDF + PNG @300dpi + SVG) from real experimental data only.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
from paths import FIGURES, RESULTS

OUT = FIGURES
O = RESULTS
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

NATURE = {
    "blue": "#4878CF", "red": "#D65F5F", "green": "#6ACC65",
    "orange": "#EE854A", "purple": "#956CB4", "teal": "#82C6E2",
    "gray": "#8C8C8C", "cb_blue": "#0072B2", "cb_orange": "#E69F00",
}

ORDER = ["Full AM-DGCN", "w/o Multi-Scale", "w/o Adaptive Adjacency", "w/o PLV",
         "w/o Cross-Scale MP", "w/o Scale Attention", "w/o Linear Shortcut",
         "Single Scale (GCN+Linear)", "Single Scale (GCN)", "Linear Only (no GCN)"]
SHORT = ["Full", "w/o MS", "w/o AA", "w/o PLV", "w/o CSMP",
         "w/o SA", "w/o LS", "SS(G+Lin)", "SS(GCN)", "Lin-only"]

def bar_color(k):
    if k == "Full AM-DGCN":
        return NATURE["blue"]
    if "Linear Shortcut" in k or k == "Single Scale (GCN)":
        return NATURE["red"]
    if k == "Single Scale (GCN+Linear)":
        return NATURE["cb_orange"]
    if k == "Linear Only (no GCN)":
        return NATURE["green"]
    return NATURE["gray"]

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    print("saved", name)

def load_ablation(path):
    d = json.load(open(path, encoding="utf-8"))
    means, stds, pts = [], [], []
    for k in ORDER:
        v = d[k]
        means.append(v["mean_accuracy"] * 100)
        stds.append(v["std_accuracy"] * 100)
        pa = v.get("per_subject_accuracy")
        pts.append([a * 100 for a in pa] if pa else None)
    return np.array(means), np.array(stds), pts

# ---------------- Figure 2 & 3: ablation (single-panel bars, jittered points for retained per-subject) ----
# Each entry is (fresh_source, legacy_source, outname, title, ylim).
# The fresh source is the single-seed=20260918 re-run (run_ablation_fresh_all.py), which
# retains per-subject arrays for ALL ten variants; the legacy source is kept as a fallback
# so the script still renders before a dataset's re-run has finished.
ABL = [
    ("seed_ablation_checkpoint.json",
     "SEED_ablation_final_20260819_220515/ablation_results.json",
     "figure2_seed_ablation",
     "Ablation on SEED (subject-independent, LOSO)", (38, 88)),
    ("seediv_ablation_checkpoint.json",
     "SEEDIV_ablation_final_20260819_220550/ablation_results.json",
     "figure3_seediv_ablation",
     "Ablation on SEED-IV (subject-independent, LOSO)", (22, 72)),
]
# NOTE on the ylim values above: they are set to the full observed range of the
# per-subject points, not to the range of the bars.  The earlier limits (42, 80)
# for SEED and (28, 72) for SEED-IV clipped the jittered subject markers while
# leaving the mean+/-sd error bars inside -- 7 of the 10 SEED variants have a
# subject above 80 (max 84.44) and 2 SEED-IV variants have a subject below 28
# (min 23.61), so 17 points were drawn truncated or not at all.  A figure that
# hides part of its own scatter is a data-integrity problem, not a cosmetic one,
# so do not "tidy" these limits back to round numbers without re-checking
# max/min of per_subject_accuracy in the two checkpoints.
for fresh, legacy, outname, title, ylim in ABL:
    src = fresh if os.path.exists(os.path.join(O, fresh)) else legacy
    tag = "FRESH" if src == fresh else "LEGACY"
    print(f"[{outname}] data source = {tag}: {src}")
    m, s, pts = load_ablation(os.path.join(O, src))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(m))
    cols = [bar_color(k) for k in ORDER]
    ax.bar(x, m, color=cols, edgecolor="black", linewidth=0.5, width=0.7, zorder=2)
    ax.errorbar(x, m, yerr=s, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
    # jittered individual points (per-subject scores retained for every variant in the fresh run)
    rng = np.random.default_rng(0)
    n_drawn = 0
    for xi, (k, p) in enumerate(zip(ORDER, pts)):
        if p:
            jx = xi + rng.uniform(-0.12, 0.12, len(p))
            ax.plot(jx, p, "o", ms=3, mfc="none", mec="black", mew=0.5, alpha=0.8, zorder=4)
            n_drawn += len(p)
    print(f"    per-subject points drawn: {n_drawn}")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(*ylim)
    # Decisive drop annotation (w/o Linear Shortcut is index 6).
    # The label sits in the empty band just under the top spine instead of at
    # ylim[1]-2.5, where its second line ran into the top cap of the
    # neighbouring SS(GCN+Linear) error bar and into that bar's jittered
    # markers.  The tallest subject marker inside the label's x-range is 82.22
    # on SEED and 63.89 on SEED-IV, so the band above it is clear on both.
    _rng = ylim[1] - ylim[0]
    ax.annotate("", xy=(6, m[6] + s[6] + 1.0), xytext=(6, ylim[1] - 0.14 * _rng),
                arrowprops=dict(arrowstyle="-|>", color=NATURE["red"], lw=1.0))
    ax.text(6.25, ylim[1] - 0.006 * _rng, "linear\nshortcut", ha="left", va="top",
            fontsize=6.5, color=NATURE["red"], fontweight="bold")
    save(fig, outname)

# ---------------- Figure 4: hyperparameter sweeps (3 panels a/b/c) ----------------
def load_hp(path, keys):
    d = json.load(open(path, encoding="utf-8"))
    return (np.array([d[k]["mean_accuracy"] * 100 for k in keys]),
            np.array([d[k]["std_accuracy"] * 100 for k in keys]))

a_m, a_s = load_hp(os.path.join(O, "hyperparam_alpha_SEED_checkpoint.json"),
                   ["alpha_0.0", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1.0"])
l_m, l_s = load_hp(os.path.join(O, "hyperparam_layers_SEED_checkpoint.json"),
                   ["layers_1", "layers_2", "layers_3", "layers_4"])
h_m, h_s = load_hp(os.path.join(O, "hyperparam_hidden_SEED_checkpoint.json"),
                   ["hidden_32", "hidden_64", "hidden_128", "hidden_256"])

fig = plt.figure(figsize=(7.2, 2.7))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5)
panels = [(a_m, a_s, ["0.0", "0.25", "0.5", "0.75", "1.0"], "alpha init", "a"),
          (l_m, l_s, ["1", "2", "3", "4"], "GCN layers", "b"),
          (h_m, h_s, ["32", "64", "128", "256"], "hidden dim", "c")]
for i, (mm, ss, xx, xl, lab) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    xi = np.arange(len(xx))
    ax.plot(xi, mm, "-o", color=NATURE["cb_blue"], linewidth=1.2, markersize=4, zorder=3)
    ax.errorbar(xi, mm, yerr=ss, fmt="none", ecolor=NATURE["cb_blue"],
                elinewidth=1.0, capsize=3, zorder=2)
    ax.set_xticks(xi); ax.set_xticklabels(xx, fontsize=7)
    ax.set_xlabel(xl, fontsize=7.5)
    ax.set_ylabel("Accuracy (%)" if i == 0 else "", fontsize=7.5)
    ax.set_ylim(54, 78)
    ax.text(-0.16, 1.08, lab, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")
save(fig, "figure4_hyperparams")

# ---------------- Figure 5: LOSO ceiling vs SI, with paired-test significance ----------------
ceil = {"SEED": (52.00, 10.82), "SEED-IV": (39.81, 6.87),
        "DEAP-val": (55.08, 8.81), "DEAP-aro": (50.86, 13.93)}
si = {"SEED": (64.44, 11.48), "SEED-IV": (49.63, 9.43),
      "DEAP-val": (59.92, 7.49), "DEAP-aro": (62.19, 11.38)}
sig = ["****", "***", "*", "****"]  # paired t p: SEED 6.0e-6 / SEED-IV 3.8e-4 / val 2.9e-2 / aro 8.8e-5
datasets = ["SEED", "SEED-IV", "DEAP-val", "DEAP-aro"]
labels = ["SEED", "SEED-IV", "DEAP\n(valence)", "DEAP\n(arousal)"]
cm, cs = np.array([ceil[d] for d in datasets]).T
sm, ss = np.array([si[d] for d in datasets]).T

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(datasets)); w = 0.36
ax.bar(x - w/2, cm, w, yerr=cs, color=NATURE["gray"], edgecolor="black", linewidth=0.5,
       capsize=3, label="LOSO linear ceiling")
ax.bar(x + w/2, sm, w, yerr=ss, color=NATURE["blue"], edgecolor="black", linewidth=0.5,
       capsize=3, label="AM-DGCN (SI)")
for i in range(len(datasets)):
    ax.annotate(f"+{sm[i]-cm[i]:.1f}", xy=(x[i] + w/2, sm[i] + ss[i] + 0.8),
                ha="center", fontsize=7, color=NATURE["blue"], fontweight="bold")
    if sig[i]:
        ax.text(x[i] + w/2, sm[i] + ss[i] + 2.6, sig[i], ha="center", fontsize=9,
                color="black", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(30, 82)
# The legend used to sit at loc="upper left", which is directly above the SEED
# group -- exactly where this panel prints the paired-significance stars and the
# "+12.4" gap for SEED (x = 0.18, y = 76.7 / 78.5).  The two overlapped and the
# legend text was unreadable.  The empty band above the two middle groups
# (SEED-IV and DEAP-val top out at 49.3 and 66.5) is clear, so the legend goes
# there; nothing in the panel reaches it.
ax.legend(frameon=False, fontsize=7.5, loc="upper center")
save(fig, "figure5_ceiling")

# ---------------- Figure 1: architecture schematic ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

def box(ax, x, y, w, h, text, fc, tc="white", fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=fc, ec="black", lw=0.8, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, weight="bold")

def arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=10, color=color, lw=1.2, zorder=1))

box(ax, 3.6, 10.6, 2.8, 1.0, "DE features\n(C ch x B bands)", NATURE["teal"])
# The three scale boxes start at x = 1.5, not 0.4.  The red "Linear shortcut"
# box spans x 0.3-1.3 over the full height y 6.2-10.6, so the old
# electrode-level box (0.4-3.0) was painted on top of it and its label began
# inside the red box.  Three 2.4-wide boxes now share the band 1.5-9.5 evenly
# (centres 2.7 / 5.5 / 8.3), which clears the red box by 0.2.
box(ax, 1.5, 8.2, 2.4, 1.0, "Electrode-level\ngraph", NATURE["purple"])
box(ax, 4.3, 8.2, 2.4, 1.0, "Region-level\ngraph", NATURE["purple"])
box(ax, 7.1, 8.2, 2.4, 1.0, "Global-level\ngraph", NATURE["purple"])
arrow(ax, 5.0, 10.6, 2.7, 9.2); arrow(ax, 5.0, 10.6, 5.5, 9.2); arrow(ax, 5.0, 10.6, 8.3, 9.2)
# White backing: the middle arrow of this fan passes through the label's box,
# and the arrow used to be drawn across the text.
ax.text(5.0, 9.72, "Multi-scale graph construction", ha="center", fontsize=6.8,
        style="italic", color="#555", zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
box(ax, 1.4, 6.2, 7.2, 1.0, "Multi-scale GCN encoding (per scale)", NATURE["blue"])
arrow(ax, 2.7, 8.2, 3.0, 7.2); arrow(ax, 5.5, 8.2, 5.5, 7.2); arrow(ax, 8.3, 8.2, 8.0, 7.2)
box(ax, 1.4, 4.4, 7.2, 1.0, "Bidirectional cross-scale message passing", NATURE["cb_blue"])
arrow(ax, 5.0, 6.2, 5.0, 5.4)
box(ax, 1.4, 2.6, 7.2, 1.0, "Scale-aware attention fusion", NATURE["cb_blue"])
arrow(ax, 5.0, 4.4, 5.0, 3.6)
box(ax, 3.0, 0.5, 4.0, 1.0, "Classifier", NATURE["orange"])
arrow(ax, 5.0, 2.6, 5.0, 1.5)
box(ax, 0.3, 6.2, 1.0, 4.4, "Linear\nshortcut", NATURE["red"], fs=7)
arrow(ax, 5.0, 10.6, 0.8, 9.2, color=NATURE["red"])
arrow(ax, 0.8, 6.2, 5.0, 1.0, color=NATURE["red"])
save(fig, "figure1_architecture")

# ---------------- Figure 4 (file: figure7_shortcut_signal): shortcut-vs-graph decomposition ----------
# Panel a -- the Full -> {remove shortcut} / {remove graph} decomposition on SEED.
# Panel b -- where the same-protocol third-party baselines sit on the graph<->shortcut axis,
#            for both datasets.  This panel exists because the SEED-IV placement result
#            ("the re-run baselines cluster with the graph pathway, not the shortcut") is a
#            statement about relative position, and relative position is what a reader has to
#            be able to see, not just read in three rounded means.
#
# EVERY number below is read from a fresh artefact (ablation checkpoint, baseline checkpoint,
# baseline paired test, linear-ceiling JSON) -- nothing is hardcoded, so the figure cannot go
# stale when any of those are re-run.  The "*" flags are computed from the paired-test JSON's
# significant_05 field, not typed in.
_JSON_CACHE = {}

def _json(path):
    if path not in _JSON_CACHE:
        with open(path, encoding="utf-8") as f:
            _JSON_CACHE[path] = json.load(f)
    return _JSON_CACHE[path]

def _abl_mean(path, key):
    return _json(path)[key]["mean_accuracy"] * 100

def _paired_p(ds, a, b):
    pt = os.path.join(O, "ablation_paired_test.json")
    if not os.path.exists(pt):
        return None
    node = _json(pt).get(ds, {}).get("vs_full", {}).get(b)
    if node is None:
        return None
    return node["t_pvalue"]

def _minus(s):
    return s.replace("-", "\u2212")

def fmt_p(p):
    if p is None:
        return "-"
    return "n.s." if p >= 0.05 else f"p = {p:.3f}"

# --- panel b plumbing: the two pathways, the four baselines, both datasets ---------------
GRAPH_VAR = "w/o Linear Shortcut"      # graph pathway, shortcut cut out  (red in Figs 5/6)
SHORT_VAR = "Linear Only (no GCN)"     # shortcut pathway, graph cut out  (green in Figs 5/6)
BL_ORDER = [("DGCNN", "DGCNN", NATURE["purple"], True),
            ("DGCNN-matched", "DGCNN (matched)", NATURE["purple"], False),
            ("RGNN", "RGNN", NATURE["cb_orange"], True),
            ("RGNN-plain", "RGNN (plain)", NATURE["cb_orange"], False)]
BL_FILE = os.path.join(O, "baseline_checkpoint.json")
BPT_FILE = os.path.join(O, "baseline_paired_test.json")
FIG4_ROWS = [("SEED-IV", "SEED-IV (4-class)", "seediv_ablation_checkpoint.json",
              "linear_ceiling_SEED-IV.json", (30.5, 54.5)),
             ("SEED", "SEED (3-class)", "seed_ablation_checkpoint.json",
              "linear_ceiling_SEED.json", (49.0, 70.5))]

_ABL_SRC = os.path.join(O, "seed_ablation_checkpoint.json")
if os.path.exists(_ABL_SRC):
    full = _abl_mean(_ABL_SRC, "Full AM-DGCN")
    noshort = _abl_mean(_ABL_SRC, GRAPH_VAR)
    linonly = _abl_mean(_ABL_SRC, SHORT_VAR)
    p_short = _paired_p("SEED", "Full AM-DGCN", GRAPH_VAR)
    p_lin = _paired_p("SEED", "Full AM-DGCN", SHORT_VAR)
    d_short = noshort - full
    d_lin = linonly - full

    def box3(ax, x, y, w, h, lines, fc, ec, tcs, fss):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.14",
                                    fc=fc, ec=ec, lw=0.9, zorder=2))
        n = len(lines)
        for i, (tx, tc, fs) in enumerate(zip(lines, tcs, fss)):
            yy = y + h - (i + 0.5) * (h / n)
            ax.text(x + w / 2, yy, tx, ha="center", va="center",
                    fontsize=fs, color=tc, zorder=3,
                    fontweight="bold" if i == 0 else "normal")

    fig = plt.figure(figsize=(7.2, 4.1))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.86, 1.74], wspace=0.14)

    # ---------------- panel a: decomposition ----------------
    axa = fig.add_subplot(gs[0])
    axa.set_xlim(0, 10); axa.set_ylim(0, 11.2); axa.axis("off")
    axa.text(0, 11.1, "a", fontsize=9, fontweight="bold", va="top", ha="left")

    box3(axa, 2.1, 9.0, 5.8, 1.9,
         ["Full AM-DGCN", f"{full:.2f}%"],
         "#E1F5EE", "#0F6E56", ["#085041", "#0F6E56"], [7, 9])
    box3(axa, 0.1, 4.8, 4.8, 3.0,
         ["remove linear", "shortcut", f"{noshort:.2f}%",
          _minus(f"{d_short:+.2f} pp") + "  " + fmt_p(p_short)],
         "#FAECE7", "#993C1D", ["#712B13", "#712B13", "#712B13", "#993C1D"], [6.5, 6.5, 8.5, 6])
    box3(axa, 5.1, 4.8, 4.8, 3.0,
         ["remove graph", "pathway", f"{linonly:.2f}%",
          f"{d_lin:+.2f} pp  " + fmt_p(p_lin)],
         "#F1EFE8", "#5F5E5A", ["#444441", "#444441", "#444441", "#5F5E5A"], [6.5, 6.5, 8.5, 6])
    arrow(axa, 3.6, 9.0, 2.5, 7.8, color="#993C1D")
    arrow(axa, 6.4, 9.0, 7.5, 7.8, color="#5F5E5A")
    box3(axa, 0.1, 1.7, 9.8, 2.1,
         ["SEED \u00b7 LOSO", "the shortcut carries the signal",
          "the graph pathway adds nothing"],
         "#E6F1FB", "#185FA5", ["#0C447C", "#0C447C", "#0C447C"], [6.5, 6.5, 6.5])
    axa.text(0.1, 0.95, "* baseline differs from the\n   shortcut-only variant (paired t, p < 0.05)\n"
                        "   thin bar = mean \u00b1 1 population sd",
             fontsize=6, va="top", ha="left", color="#444441")

    # ---------------- panel b: where the baselines land ----------------
    if os.path.exists(BL_FILE):
        gsb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.42)
        for ri, (ds, rowlab, ablname, ceilname, xlim) in enumerate(FIG4_ROWS):
            ax = fig.add_subplot(gsb[ri])
            abl = os.path.join(O, ablname)
            if not os.path.exists(abl):
                ax.axis("off"); continue
            g = _abl_mean(abl, GRAPH_VAR)
            s = _abl_mean(abl, SHORT_VAR)
            fm = _abl_mean(abl, "Full AM-DGCN")
            cl = _json(os.path.join(O, ceilname))["si_accuracy_mean"] * 100
            bck = _json(BL_FILE)
            bmean = {n: bck["%s/%s" % (ds, n)]["mean_accuracy"] * 100 for n, _, _, _ in BL_ORDER}
            bsd = {n: bck["%s/%s" % (ds, n)]["std_accuracy"] * 100 for n, _, _, _ in BL_ORDER}
            sig = {}
            if os.path.exists(BPT_FILE):
                sig = _json(BPT_FILE)["datasets"].get(ds, {}) \
                    .get("vs_variants", {}).get(SHORT_VAR, {}).get("baselines", {})

            # the interval a third-party model has to land in, and its two poles
            ax.axvspan(g, s, color="#F1F0EA", zorder=0)
            ax.axvline(g, color=NATURE["red"], lw=1.1, ls=(0, (4, 2)), zorder=1)
            ax.axvline(s, color=NATURE["green"], lw=1.1, ls=(0, (4, 2)), zorder=1)
            # pole labels sit just above the strip, clear of the row label at y=1.19
            ax.text(g, 1.02, "graph-only", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color=NATURE["red"])
            ax.text(s, 1.02, "shortcut-only", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.5, color="#3E8E3A")

            levels = [("__FULL__", 0.86)] + [(n, 0.70 - 0.14 * i) for i, (n, _, _, _) in enumerate(BL_ORDER)] \
                     + [("__CEIL__", 0.12)]
            for key, y in levels:
                if key == "__FULL__":
                    m, sd, mk, col, filled, lab = fm, None, "D", NATURE["blue"], True, "AM-DGCN (Full)"
                elif key == "__CEIL__":
                    m, sd, mk, col, filled, lab = cl, None, "^", NATURE["gray"], True, "linear ceiling"
                else:
                    nm, lab, col, filled = next(e for e in BL_ORDER if e[0] == key)
                    m, sd, mk = bmean[nm], bsd[nm], ("o" if nm.startswith("DGCNN") else "s")
                    lab = lab + ("*" if sig.get(nm, {}).get("significant_05") else "")
                if sd is not None:
                    ax.plot([m - sd, m + sd], [y, y], "-", color=col, lw=0.6, alpha=0.40, zorder=2)
                ax.plot([m], [y], marker=mk, ms=4.2, mfc=(col if filled else "white"),
                        mec=col, mew=0.9, zorder=4)
                ax.text(m + (xlim[1] - xlim[0]) * 0.022, y, lab, fontsize=6, va="center",
                        ha="left", zorder=5,
                        bbox=dict(fc="white", ec="none", pad=0.7, alpha=0.85))

            # live-computed position summary: distance to each pole, as a range
            dg = [bmean[n] - g for n, _, _, _ in BL_ORDER]
            dsh = [bmean[n] - s for n, _, _, _ in BL_ORDER]
            ax.text(1.0, 1.19, "vs graph-only [%+.1f, %+.1f] pp   vs shortcut-only [%+.1f, %+.1f] pp"
                    % (min(dg), max(dg), min(dsh), max(dsh)),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#444441")
            ax.text(0.0, 1.19, rowlab, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7, fontweight="bold")
            if ri == 0:
                ax.text(-0.085, 1.19, "b", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, fontweight="bold")

            ax.set_xlim(*xlim); ax.set_ylim(0, 1.02)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(np.arange(np.ceil(xlim[0] / 5) * 5, xlim[1] + 0.1, 5))
            ax.tick_params(axis="x", labelsize=6.5)
            if ri == 0:
                ax.tick_params(axis="x", labelbottom=False)
            if ri == len(FIG4_ROWS) - 1:
                ax.set_xlabel("Accuracy (%)", fontsize=7.5)
    save(fig, "figure7_shortcut_signal")
    print(f"    full={full:.2f}  w/oLS={noshort:.2f} ({d_short:+.2f}, {fmt_p(p_short)})  "
          f"LinOnly={linonly:.2f} ({d_lin:+.2f}, {fmt_p(p_lin)})")
else:
    print("[figure7_shortcut_signal] skipped: fresh SEED checkpoint absent")

print("ALL NATURE FIGURES SAVED to", OUT)
